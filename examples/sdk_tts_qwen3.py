"""Qwen3-TTS through the Python SDK: built-in voices, voice design and voice cloning.

Start one server per checkpoint, e.g.:
    mstar serve qwen3_tts_1p7b          # built-in speakers + style instructions
    mstar serve qwen3_tts_voicedesign   # voice described by an instruction
    mstar serve qwen3_tts_base          # voice cloned from a reference clip
"""

import sys

from mstar import MStarClient

client = MStarClient("http://localhost:8000")
variant = sys.argv[1] if len(sys.argv) > 1 else "custom_voice"

if variant == "custom_voice":
    audio = client.tts(
        "Hello from M star! It is a beautiful day for a walk.",
        voice="vivian",
        language="English",
        instruct="Speak with great enthusiasm.",  # 1.7B only; drop it for the 0.6B checkpoint
    )
elif variant == "voice_design":
    audio = client.tts(
        "Hello from M star! It is a beautiful day for a walk.",
        language="English",
        instruct="A calm, warm adult female voice with a slight British accent.",
    )
elif variant == "voice_clone":
    audio = client.tts(
        "Hello from M star! It is a beautiful day for a walk.",
        language="English",
        reference_audio="reference.wav",          # 3-10 s clip of the target speaker
        ref_text="Transcript of the reference clip.",  # or x_vector_only_mode=True
    )
else:
    sys.exit(f"unknown variant {variant!r}: custom_voice | voice_design | voice_clone")

audio.to_wav("out.wav")
print(f"wrote out.wav — {len(audio)} samples @ {audio.sample_rate} Hz")
