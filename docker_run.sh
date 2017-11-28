#!/bin/bash

CLEVRDIR=$1
if [ -z "$CLEVRDIR" ]; then
	printf "Please specify CLEVR dataset base directory as first argument\n"
else
	nvidia-docker run --rm -it --ipc=host \
		--volume=$PWD:/app \
		--volume=$CLEVRDIR:/clevr \
		-e CUDA_VISIBLE_DEVICES=0  \
		pytorch-rn python3 /app/main.py --clevr-dir=/clevr
fi
