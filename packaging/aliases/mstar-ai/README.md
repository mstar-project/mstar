# mstar-ai

Alias package for the M* multimodal inference engine. It has no code of its
own; installing it pulls in the real `m-star` distribution.

```
pip install mstar-ai
pip install "mstar-ai[bagel]"   # extras forward to m-star
pip install "mstar-ai[all]"
```

is equivalent to installing `m-star` with the same extras. Either way you
`import mstar`.
