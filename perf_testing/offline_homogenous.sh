#!/bin/bash

BATCH_SIZES=(1 2 4 8 16)
NUM_BATCHES="${NUM_BATCHES:-10}"
MIN_NUM_REQUESTS="${MIN_NUM_REQUESTS:-20}"

compute_num_requests() {
  local bs=$1
  local req=$((NUM_BATCHES * bs))
  if [ "$req" -lt "$MIN_NUM_REQUESTS" ]; then
    req=$MIN_NUM_REQUESTS
  fi
  echo $req
}

# text_to_text (with DATASET=text)
for bs in "${BATCH_SIZES[@]}"; do
  NUM_REQUESTS=$(compute_num_requests $bs)
  NUM_REQUESTS=$NUM_REQUESTS TASK=text_to_text DATASET=text BATCH_SIZE=$bs benchmark/run_benchmark.sh
done

# image_to_text
for bs in "${BATCH_SIZES[@]}"; do
  NUM_REQUESTS=$(compute_num_requests $bs)
  NUM_REQUESTS=$NUM_REQUESTS TASK=image_to_text BATCH_SIZE=$bs benchmark/run_benchmark.sh
done

# text_to_image
for bs in "${BATCH_SIZES[@]}"; do
  NUM_REQUESTS=$(compute_num_requests $bs)
  NUM_REQUESTS=$NUM_REQUESTS TASK=text_to_image BATCH_SIZE=$bs benchmark/run_benchmark.sh
done

BATCH_SIZES=(1 2 4 8)

# image_to_image
for bs in "${BATCH_SIZES[@]}"; do
  NUM_REQUESTS=$(compute_num_requests $bs)
  NUM_REQUESTS=$NUM_REQUESTS TASK=image_to_image BATCH_SIZE=$bs benchmark/run_benchmark.sh
done