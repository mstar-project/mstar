"""Graph-walk names shared by the Cosmos3 model graph and its submodules."""

PREFILL_WALK = "prefill"
# Image/video-conditioned generation prefills the same understanding tower, but
# also VAE-encodes the conditioning frame into a clean anchor latent (see
# Cosmos3DiTSubmodule._encode_conditioning). It is a separate walk from the
# text-only prefill because the graph node only fires once all of its declared
# inputs arrive, so the conditioning image has to be one of them.
PREFILL_COND_WALK = "prefill_cond"
# Action inverse-dynamics conditions on a full video rather than a single frame,
# so it gets its own conditioned prefill that takes the video among its inputs.
PREFILL_COND_VIDEO_WALK = "prefill_cond_video"
IMAGE_GEN_WALK = "image_gen"
VIDEO_GEN_WALK = "video_gen"
VIDEO_SOUND_GEN_WALK = "video_sound_gen"
ACTION_GEN_WALK = "action_gen"
# Forward-dynamics runs the same joint video+action denoise but emits the
# predicted video (VAE-decoded) instead of the action, so it has its own walk.
ACTION_VIDEO_GEN_WALK = "action_video_gen"
