#!/bin/bash

CLEVRDIR=$1
NUMGPUS=$2
if [ -z "$CLEVRDIR" ]; then
	printf "Please specify CLEVR dataset base directory as first argument\n"

elif [ -z "$NUMGPUS" ]; then
	printf "Please specify number of available GPUs as second argument\n"

else
	nvidia-docker run --rm -it --ipc=host \
		-p 0.0.0.0:6006:6006 \
		--volume=$PWD:/app \
		--volume=$CLEVRDIR:/clevr \
		pytorch-rn-raytune python3 /app/train.py --model 'sd-innetaggr' --clevr-dir=/clevr --num-gpus $NUMGPUS
fi
