"""Graph-walk names shared by the Cosmos3 model graph and its submodules."""

# Video-to-video conditioning defaults (reference pipeline recipe): the request
# pins these clean latent frames from the conditioning video, keeps the
# first/last frames of a longer input, and denoises with the V2V flow shift.
DEFAULT_CONDITION_FRAME_INDEXES_VISION = (0, 1)
DEFAULT_CONDITION_VIDEO_KEEP = "first"
V2V_DEFAULT_FLOW_SHIFT = 10.0

# Truncation cap on the prompt token count (before the eos + vision_start
# markers), matching the reference serving pipeline: it only bounds the UND
# pathway / GEN cross-attention cost for pathologically long prompts.
DEFAULT_MAX_SEQUENCE_LENGTH = 4096

PREFILL_WALK = "prefill"
# Image/video-conditioned generation prefills the same understanding tower
# while, in parallel, the vae_encoder node encodes the conditioning frame into
# clean anchor latents (Cosmos3VAEEncoderSubmodule). Separate walks from the
# text-only prefill because they route the conditioning input to the encoder.
PREFILL_COND_WALK = "prefill_cond"
# Action inverse-dynamics / video-to-video condition on a video rather than a
# single frame, so they get their own conditioned prefill walk.
PREFILL_COND_VIDEO_WALK = "prefill_cond_video"
IMAGE_GEN_WALK = "image_gen"
VIDEO_GEN_WALK = "video_gen"
VIDEO_SOUND_GEN_WALK = "video_sound_gen"
ACTION_GEN_WALK = "action_gen"
# Forward-dynamics runs the same joint video+action denoise but emits the
# predicted video (VAE-decoded) instead of the action, so it has its own walk.
ACTION_VIDEO_GEN_WALK = "action_video_gen"
