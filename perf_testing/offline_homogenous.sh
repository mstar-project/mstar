#!/bin/bash

#!/bin/bash

NUM_REQUESTS="${NUM_REQUESTS:-128}"

BATCH_SIZES=(1 2 4 8 16)

# text_to_text (with DATASET=text)
for bs in "${BATCH_SIZES[@]}"; do
  NUM_REQUESTS=$NUM_REQUESTS TASK=text_to_text DATASET=text BATCH_SIZE=$bs benchmark/run_benchmark.sh
done

# image_to_text
for bs in "${BATCH_SIZES[@]}"; do
  NUM_REQUESTS=$NUM_REQUESTS TASK=image_to_text BATCH_SIZE=$bs benchmark/run_benchmark.sh
done

NUM_REQUESTS="${NUM_REQUESTS_GEN:-64}"

# text_to_image
for bs in "${BATCH_SIZES[@]}"; do
  NUM_REQUESTS=$NUM_REQUESTS TASK=text_to_image BATCH_SIZE=$bs benchmark/run_benchmark.sh
done

# image_to_image
for bs in "${BATCH_SIZES[@]}"; do
  NUM_REQUESTS=$NUM_REQUESTS TASK=image_to_image BATCH_SIZE=$bs benchmark/run_benchmark.sh
done