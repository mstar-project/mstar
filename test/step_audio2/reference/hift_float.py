"""The reference's HiFT half of ``Token2wav.stream`` replayed on the recorded window mels, in
float: per window the vocoder's raw speech and source and the emitted samples, to localise a
sub-LSB difference. Checks that its PCM equals the recording's.

Usage: python -I hift_float.py <github/Step-Audio2> <hf/Step-Audio-2-mini> <oracle/rec>
"""
import os
import sys

ref_dir, model_dir, rec_dir = sys.argv[1:4]
sys.path.insert(0, ref_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import sf_shim  # noqa: E402,F401
import torch  # noqa: E402
from flashcosyvoice.modules.hifigan import HiFTGenerator  # noqa: E402
from token2wav import fade_in_out  # noqa: E402

# HIFT_DETERMINISTIC=1: cuDNN's deterministic algorithms (the recording ran without)
DET = os.environ.get("HIFT_DETERMINISTIC") == "1"
torch.backends.cudnn.deterministic = DET
rec = torch.load(os.path.join(rec_dir, "token2wav_stream.pt"))
hift = HiFTGenerator()
hift.load_state_dict({k.replace("generator.", ""): v for k, v in torch.load(
    f"{model_dir}/token2wav/hift.pt", map_location="cpu", weights_only=True).items()}, strict=True)
hift.cuda().eval()
window = torch.from_numpy(np.hamming(2 * 3840)).cuda()
cache = dict(mel=torch.zeros(1, 80, 0, device="cuda"), source=torch.zeros(1, 1, 0, device="cuda"),
             speech=torch.zeros(1, 0, device="cuda"))
out = []
torch.manual_seed(0)
torch.cuda.manual_seed(0)
with torch.inference_mode():
    for k, ((_codes, last), chunk_mel) in enumerate(zip(rec["windows"], rec["window_mels"], strict=True)):
        mel = torch.concat([cache["mel"], chunk_mel.cuda()], dim=2)
        speech, source = hift(mel, cache["source"])
        raw = speech.clone()
        if cache["speech"].shape[-1] > 0:
            speech = fade_in_out(speech, cache["speech"], window)
        cache = dict(mel=mel[..., -8:].clone(), source=source[:, :, -3840:].clone(), speech=speech[:, -3840:].clone())
        if not last:
            speech = speech[:, :-3840]
        pcm = (np.clip(speech.cpu().numpy(), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        print(k, "pcm equals the recording:", pcm == rec["window_pcm"][k])
        out.append({"raw": raw.cpu(), "source": source.cpu(), "speech": speech.cpu()})
torch.save(out, os.path.join(rec_dir, "hift_float_det.pt" if DET else "hift_float.pt"))
