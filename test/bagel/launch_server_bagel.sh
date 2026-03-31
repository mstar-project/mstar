#!/bin/bash

WHO=naomi

CACHE_DIR=/mnt/storage/$WHO/mminf/bagel/
DEVICES=0,1,2
TENSOR_PROTOCOL=RDMA # faster

# coriander settings
# CACHE_DIR=/m-coriander/coriander/$WHO/mminf_cache/bagel/
# TENSOR_PROTOCOL=TCP # Needed for coriander!!
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

TCP_DEVICE=lo # loop-back because we're on the same node

CUDA_VISIBLE_DEVICES=$DEVICES python mminf/api_server/entrypoint.py \
    --config configs/bagel_cfg_parallel.yaml \
    --cache-dir $CACHE_DIR \
    --socket-path-prefix /tmp/mminf_$WHO/ \
    --upload-dir /tmp/mminf_uploads_$WHO/ \
    --port 8000 \
    --mooncake-port 8081 \
    --tensor-comm-protocol $TENSOR_PROTOCOL \
    --tcp-transfer-device $TCP_DEVICE
    # --log-level DEBUG
