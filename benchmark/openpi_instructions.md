1. Make a python3.12 environment 
2. Clone the `openpi` reop
3. Run the following in your ennvironment:
```
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
4. From the mminf repo, run, e.g. for coriander,:
```
python benchmark/download_pi05_ckpt.py[3:21 PM]mkdir /m-coriander/coriander/naomi/openpi-cache
mv /home/$USER/.cache/openpi/* /m-coriander/coriander/$USER/openpi-cache/
```
5. Start the server with:
```
CUDA_VISIBLE_DEVICES=4 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=/m-coriander/coriander/$USER/openpi-cache/openpi-assets/checkpoints/pi05_droid
```