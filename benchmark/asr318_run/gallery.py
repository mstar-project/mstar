"""Build the examples page the user reads, looks at and listens to: every prompt, with its
input media, next to each engine's output and the oracle's.

Inputs:
  --prompts prompts.jsonl   the same file parity.py used ("text", optional "image"/"video"/"audio" paths)
  --parity DIR              parity.py's out dir: <tag>.json per engine, report.json (oracle text) if present
  --media DIR               optional: generated media, named <tag>/<prompt index>.<ext>
                            (wav/mp3/flac, png/jpg/webp, mp4/webm), e.g. media/mstar_a/0.wav
  --out DIR                 writes DIR/gallery.html and DIR/examples/NNN.md; media are copied into DIR/files/
                            so the folder can be opened, zipped or rsynced as one unit

  python gallery.py --prompts prompts.jsonl --parity out/parity --media out/media --out out/gallery
"""
import argparse
import html
import json
import shutil
from pathlib import Path

AUDIO = {".wav", ".mp3", ".flac", ".ogg"}
IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO = {".mp4", ".webm", ".mov"}


def media_tag(rel: str) -> str:
    ext = Path(rel).suffix.lower()
    src = html.escape(rel)
    if ext in AUDIO:
        return f'<audio controls preload="none" src="{src}"></audio><div class="f">{src}</div>'
    if ext in IMAGE:
        return f'<a href="{src}"><img src="{src}"></a>'
    if ext in VIDEO:
        return f'<video controls preload="metadata" src="{src}"></video><div class="f">{src}</div>'
    return f'<a href="{src}">{src}</a>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--parity")
    ap.add_argument("--media")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="mstar correctness examples")
    a = ap.parse_args()
    out = Path(a.out)
    files = out / "files"
    (out / "examples").mkdir(parents=True, exist_ok=True)
    files.mkdir(exist_ok=True)

    def copy(src: Path, name: str) -> str:
        dest = files / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return str(dest.relative_to(out))

    prompts = [json.loads(line) for line in Path(a.prompts).read_text().splitlines() if line.strip()]
    texts: dict[str, list] = {}
    oracle: dict = {}
    if a.parity:
        pdir = Path(a.parity)
        for f in sorted(pdir.glob("*.json")):
            if f.stem.startswith("report"):
                continue
            texts[f.stem] = json.loads(f.read_text())
        rep = pdir / "report.json"
        if rep.exists():
            oracle = json.loads(rep.read_text()).get("oracle", {})
    media: dict[str, dict[int, Path]] = {}
    if a.media:
        for f in sorted(Path(a.media).glob("*/*")):
            if f.stem.isdigit():
                media.setdefault(f.parent.name, {})[int(f.stem)] = f
    tags = sorted(set(texts) | set(media), key=lambda t: (not t.startswith("mstar"), t))

    cards = []
    for i, p in enumerate(prompts):
        inputs = [copy(Path(p[k]), f"inputs/{i}_{k}{Path(p[k]).suffix}") for k in ("image", "video", "audio")
                  if p.get(k)]
        cols = []
        if str(i) in oracle or i in oracle:
            cols.append(("oracle (transformers)", oracle.get(str(i), oracle.get(i)), None))
        for t in tags:
            row = next((r for r in texts.get(t, []) if r.get("prompt") == i), None)
            m = media.get(t, {}).get(i)
            cols.append((t, row["text"] if row else None, copy(m, f"{t}/{i}{m.suffix}") if m else None))
        md = [f"# Example {i}", "", "## Prompt", "", p["text"], ""]
        md += [f"Input: `{x}`" for x in inputs]
        for name, text, m in cols:
            md += ["", f"## {name}", ""]
            if m:
                md.append(f"Media: `{m}`")
            if text is not None:
                md += ["```", text, "```"]
        (out / "examples" / f"{i:03d}.md").write_text("\n".join(md) + "\n")
        body = "".join(
            f'<div class="col"><h3>{html.escape(name)}</h3>{media_tag(m) if m else ""}'
            f'{"<pre>" + html.escape(text) + "</pre>" if text is not None else ""}</div>'
            for name, text, m in cols)
        cards.append(f'<section><h2>Example {i}</h2><p class="prompt">{html.escape(p["text"])}</p>'
                     f'<div class="inputs">{"".join(media_tag(x) for x in inputs)}</div>'
                     f'<div class="grid">{body}</div></section>')

    page = f"""<!doctype html><meta charset="utf-8"><title>{html.escape(a.title)}</title>
<style>
:root {{ color-scheme: light dark; --bd: #8884; }}
body {{ font: 14px/1.45 system-ui, sans-serif; margin: 2rem auto; max-width: 1400px; padding: 0 1rem; }}
section {{ border-top: 1px solid var(--bd); padding: 1rem 0; }}
.prompt {{ font-size: 15px; white-space: pre-wrap; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }}
.col {{ border: 1px solid var(--bd); border-radius: 6px; padding: .5rem .75rem; }}
pre {{ white-space: pre-wrap; margin: .25rem 0; }}
img, video {{ max-width: 100%; border-radius: 4px; }}
.inputs img, .inputs video {{ max-height: 240px; width: auto; }}
audio {{ width: 100%; }} .f {{ font-size: 11px; opacity: .6; }}
</style>
<h1>{html.escape(a.title)}</h1>
<p>{len(prompts)} prompts; runs: {html.escape(", ".join(tags) or "-")}. Same prompts and settings on every engine.</p>
{"".join(cards)}
"""
    (out / "gallery.html").write_text(page)
    print(f"wrote {out / 'gallery.html'} and {len(prompts)} examples in {out / 'examples'}")


if __name__ == "__main__":
    main()
