# mstar-serve

Alias package for the M* multimodal inference engine. It has no code of its
own; installing it pulls in the real `mstar-ai` distribution.

```
pip install mstar-serve
pip install "mstar-serve[bagel]"   # extras forward to mstar-ai
pip install "mstar-serve[all]"
```

is equivalent to installing `mstar-ai` with the same extras. Either way you
`import mstar`.
