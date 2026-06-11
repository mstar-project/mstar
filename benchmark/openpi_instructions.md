1. Make a python3.12 environment 
2. Clone the `openpi` reop
3. Run the following in your ennvironment:
```
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
4. From the mstar repo, run:
```
pip install gsutil
python benchmark/download_pi05_ckpt.py
mkdir <DESIRED_OPENPI_CACHE_DIR>
mv /home/$USER/.cache/openpi/* <DESIRED_OPENPI_CACHE_DIR>
```
5. Start the server with:
```
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=<DESIRED_OPENPI_CACHE_DIR>/openpi-assets/checkpoints/pi05_droid
```