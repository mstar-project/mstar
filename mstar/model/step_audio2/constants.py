"""Step-Audio2 constants the API server reads too, kept free of heavy imports."""

# Per-request voice cloning needs a voice-store resource for the vocoder's prepared voice
# caches, which is not built yet; both frontends' adapters and the model refuse it alike.
REF_AUDIO_REFUSAL = "Step-Audio2 does not clone a voice from ref_audio yet; choose one of its voices with 'voice'"
