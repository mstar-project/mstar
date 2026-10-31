# Copyright (c) 2026 Robbyant Team.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: the pose math split between the
# data worker and the DiT's device, the rays folded per latent cell.
#
# Original files (Robbyant/lingbot-world, wan/utils/cam_utils.py and
# wan/image2video_fast.py) were released under Apache-2.0, with the full license
# text available at https://github.com/Robbyant/lingbot-world/blob/main/LICENSE.txt.
#
# This modified file is released under the same license.

"""Camera conditioning: a c2w pose trajectory to per-latent-frame Plucker rays.

Ported from LingBot-World v1's ``wan/utils/cam_utils.py`` and the camera
preprocessing in ``WanI2VFast.generate`` (Robbyant/lingbot-world, Apache-2.0),
which the V2 reference runs with the same math; the Plucker ray construction
follows hehao13/CameraCtrl (Apache-2.0). The pose math runs on the data worker
(``latent_camera_poses``, numpy/scipy on CPU, as the reference does before it
moves the poses to the GPU); the Plucker rays are built on the DiT's device
(``plucker_embedding``), where the reference builds them, because their matmul
runs under the engine's TF32 matmul precision there.

Two reference behaviours a streaming session cannot keep as is:
  * translations are normalized by the largest framewise step over the WHOLE
    trajectory, so the scale of chunk 0 depends on poses the client has not
    sent yet;
  * the trajectory is resampled from video frames to latent frames over its
    whole length (``linspace(0, F - 1, lat_f)``), not frame by frame.
An interactive session needs a fixed translation scale and a per-chunk
resampling rule; both are open design points, not ported behaviour.
"""

import numpy as np
import torch
from einops import rearrange


def _interpolate_camera_poses(
    src_indices: np.ndarray, src_rot_mat: np.ndarray, src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    interp_func_trans = interp1d(
        src_indices, src_trans_vec, axis=0, kind="linear", bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)
    quats = Rotation.from_matrix(src_rot_mat).as_quat().copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    slerp_func_rot = Slerp(src_indices, Rotation.from_quat(quats))
    interpolated_rot_mat = slerp_func_rot(tgt_indices).as_matrix()
    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def _se3_inverse(T: torch.Tensor) -> torch.Tensor:
    rot = T[:, :3, :3]
    trans = T[:, :3, 3:]
    r_inv = rot.transpose(-1, -2)
    t_inv = -torch.bmm(r_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = r_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def _relative_poses_framewise(c2ws: torch.Tensor) -> torch.Tensor:
    """``compute_relative_poses(framewise=True, normalize_trans=True)``."""
    ref_w2cs = _se3_inverse(c2ws[0:1])
    relative = torch.matmul(ref_w2cs, c2ws)
    relative[0] = torch.eye(4, device=c2ws.device, dtype=c2ws.dtype)
    relative[1:] = torch.bmm(_se3_inverse(relative[:-1]), relative[1:])
    translations = relative[:, :3, 3]
    max_norm = torch.norm(translations, dim=-1).max()
    if max_norm > 0:
        relative[:, :3, 3] = translations / max_norm
    return relative


def latent_camera_poses(c2ws: np.ndarray, num_latent_frames: int) -> torch.Tensor:
    """``[num_latent_frames, 4, 4]`` float32 framewise relative poses from the
    request's ``[F, 4, 4]`` OpenCV c2w trajectory (already cut to the video's
    frame count)."""
    n = len(c2ws)
    return _relative_poses_framewise(
        _interpolate_camera_poses(
            src_indices=np.linspace(0, n - 1, n),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, n - 1, num_latent_frames),
        )
    )


def scale_intrinsics(
    intrinsics: torch.Tensor, height_org: int, width_org: int, height: int, width: int,
) -> torch.Tensor:
    """``get_Ks_transformed`` with no crop: the first frame's ``[fx, fy, cx, cy]``
    rescaled from the size the intrinsics were given at to the output size. The
    reference uses only the first frame's intrinsics for every frame."""
    fx, fy, cx, cy = intrinsics.float().reshape(-1, 4)[0]
    scale_x = width / width_org
    scale_y = height / height_org
    return torch.stack([fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y])


def plucker_embedding(
    c2ws: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int,
    lat_h: int, lat_w: int,
) -> torch.Tensor:
    """``[1, 6 * s * s, F, lat_h, lat_w]`` Plucker rays (origin, direction) with
    each latent cell's ``s x s`` pixels folded into channels, for ``c2ws``
    ``[F, 4, 4]`` and one ``[fx, fy, cx, cy]`` for every frame."""
    n_frames = c2ws.shape[0]
    device, dtype = c2ws.device, c2ws.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid_xy = (torch.stack([grid_x, grid_y], dim=-1).view(-1, 2) + 0.5)[None].repeat(n_frames, 1, 1)
    Ks = intrinsics.to(device).reshape(1, 4).repeat(n_frames, 1)
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)
    i, j = grid_xy[..., 0], grid_xy[..., 1]
    zs = torch.ones_like(i)
    dirs = torch.stack([(i - cx) / fx * zs, (j - cy) / fy * zs, zs], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    rays_d = dirs @ c2ws[:, :3, :3].transpose(-1, -2)
    rays_o = c2ws[:, :3, 3][:, None, :].expand_as(rays_d)
    plucker = torch.cat([rays_o, rays_d], dim=-1).view(n_frames, height, width, 6)
    plucker = rearrange(
        plucker, "f (h c1) (w c2) c -> (f h w) (c c1 c2)",
        c1=height // lat_h, c2=width // lat_w,
    )[None]
    return rearrange(plucker, "b (f h w) c -> b c f h w", f=n_frames, h=lat_h, w=lat_w)
