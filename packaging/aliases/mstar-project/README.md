# mstar-project

Alias package for the M* multimodal inference engine. It has no code of its
own; installing it pulls in the real `mstar-ai` distribution.

```
pip install mstar-project
pip install "mstar-project[bagel]"   # extras forward to mstar-ai
pip install "mstar-project[all]"
```

is equivalent to installing `mstar-ai` with the same extras. Either way you
`import mstar`.
