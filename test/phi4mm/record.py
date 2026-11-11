"""Record reference tensors for Phi-4-multimodal parity (run in /root/oracle-venv).

Per case: the processor's outputs, the vision tower's hidden_states[-2] per crop and
each image's projected embeddings, the conformer's output and projected embeddings,
the prefill's per-layer hidden states (every layer for short prompts, every 4th plus
the last for long ones), last-token logits, and greedy tokens with each step's top-2
logit margin.

usage: record.py <attn: sdpa|eager> <out_dir> [case ...]
"""
import glob
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

PATH = glob.glob('/root/hf/hub/models--microsoft--Phi-4-multimodal-instruct/snapshots/*/')[0]
ORACLE = Path(os.environ.get('PHI4MM_ORACLE', '/workspace/phi4mm/oracle'))
MEDIA = ORACLE / 'media'
U, A, E = '<|user|>', '<|assistant|>', '<|end|>'


def img(name):
    return Image.open(MEDIA / name).convert('RGB')


def wav(name):
    w, sr = sf.read(MEDIA / name)
    return (w, sr)


CASES = {
    'text': (f'{U}What is the capital of France? Answer in one sentence.{E}{A}', {}),
    'text_long': (
        f'{U}' + 'Repeat after me: the quick brown fox jumps over the lazy dog. ' * 400 + f'Now summarize.{E}{A}', {},
    ),
    'image_bagel': (f'{U}<|image_1|>Describe this image in two sentences.{E}{A}', {'images': ['bagel.png']}),
    'image_food_small': (f'{U}<|image_1|>What food is this?{E}{A}', {'images': ['food_small.png']}),
    'image_food_wide': (f'{U}<|image_1|>Describe the image in detail.{E}{A}', {'images': ['food_wide.png']}),
    'two_images': (
        f'{U}<|image_1|><|image_2|>Compare the two images.{E}{A}', {'images': ['bagel.png', 'food_small.png']},
    ),
    'audio_short': (f'{U}<|audio_1|>Transcribe the audio clip into text.{E}{A}', {'audios': ['speech_short.wav']}),
    'audio_long': (f'{U}<|audio_1|>Transcribe the audio clip into text.{E}{A}', {'audios': ['speech_long.wav']}),
    'image_audio': (f'{U}<|image_1|><|audio_1|>Answer the spoken words, then describe the image.{E}{A}',
                    {'images': ['bagel.png'], 'audios': ['speech_short.wav']}),
}


def main():
    attn, out_dir = sys.argv[1], Path(sys.argv[2])
    names = sys.argv[3:] or list(CASES)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    proc = AutoProcessor.from_pretrained(PATH, trust_remote_code=True)
    # `<attn>+fa2vision` / `<attn>+fa2vision_sdpa`: the vision tower on FA2's padded-patch
    # semantics, on the remote code's matmul kernels or on torch SDPA's; `+nolora`: the
    # base model for every prompt, a negative control for the served checks
    attn, *options = attn.split('+')
    fa2_vision = bool({'fa2vision', 'fa2vision_sdpa'} & set(options))
    vision_kernel = 'sdpa' if 'fa2vision_sdpa' in options else 'manual'
    model = AutoModelForCausalLM.from_pretrained(
        PATH, trust_remote_code=True, torch_dtype=torch.bfloat16, _attn_implementation=attn,
    ).cuda().eval()
    if fa2_vision:
        # the checkpoint's default FA2 semantics for the vision tower's padded patches
        # (record_vision.py's `attention`, manual kernels, masked queries zeroed)
        import functools
        import types

        from record_vision import attention
        for layer in model.model.embed_tokens_extend.image_embed.img_processor.encoder.layers:
            layer.self_attn.forward = types.MethodType(
                functools.partial(attention, kernel=vision_kernel, zero_masked=True), layer.self_attn)
    if 'nolora' in options:
        model.set_lora_adapter = lambda _name: model.unset_lora_adapter()
    ext = model.model.embed_tokens_extend
    cap = {}

    def keep(name):
        def hook(_mod, _inp, out):
            cap.setdefault(name, []).append(out)
        return hook

    def keep_tower(_mod, _inp, out):
        cap.setdefault('vision_hidden', []).append(out.hidden_states[-2].detach().cpu())
    ext.image_embed.img_processor.register_forward_hook(keep_tower)
    ext.image_embed.img_projection.register_forward_hook(
        lambda m, i, o: cap.setdefault('image_embeds', []).append(o.detach().cpu()))
    ext.audio_embed.encoder.register_forward_hook(
        lambda m, i, o: cap.setdefault('audio_encoded', []).append(o[0].detach().cpu()))
    for mode in ('speech', 'vision'):
        ext.audio_embed.audio_projection[mode].register_forward_hook(
            lambda m, i, o, mode=mode: cap.setdefault(f'audio_embeds_{mode}', []).append(o.detach().cpu()))

    for name in names:
        prompt, media = CASES[name]
        kw = {}
        if 'images' in media:
            kw['images'] = [img(n) for n in media['images']]
        if 'audios' in media:
            kw['audios'] = [wav(n) for n in media['audios']]
        inputs = proc(text=prompt, return_tensors='pt', **kw).to('cuda')
        cap.clear()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, num_logits_to_keep=1)
        hs = out.hidden_states
        n = inputs['input_ids'].shape[1]
        layers = list(range(len(hs))) if n <= 1024 else sorted(set(range(0, len(hs), 4)) | {len(hs) - 1})
        record = {
            'prompt': prompt, 'media': media,
            'inputs': {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()},
            'hidden_layers': layers,
            'hidden': {i: hs[i][0].cpu() for i in layers},
            'last_logits': out.logits[0, -1].float().cpu(),
            **{k: v for k, v in cap.items()},
        }
        cap.clear()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False, num_logits_to_keep=1,
                                 output_scores=True, return_dict_in_generate=True)
        gen = out.sequences
        record['greedy'] = gen[0, n:].cpu()
        # each step's top-2 logit gap: a gap of a bf16 step or two is a tie the kernels decide
        top2 = torch.stack([s[0].float().topk(2).values for s in out.scores])
        record['greedy_margin'] = (top2[:, 0] - top2[:, 1]).cpu()
        record['greedy_text'] = proc.batch_decode(gen[:, n:], skip_special_tokens=True)[0]
        cap.clear()
        torch.save(record, out_dir / f'{name}.pt')
        print(f'{name}: {n} tokens, mode {int(inputs["input_mode"][0])}: {record["greedy_text"][:100]!r}', flush=True)


if __name__ == '__main__':
    main()
