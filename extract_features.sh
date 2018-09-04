#!/bin/bash

set -e

CLEVRDIR=$1
if [ -z "$CLEVRDIR" ]; then
	echo "Please specify CLEVR dataset base directory as first argument"
	exit
fi

if [ ! -d "$CLEVRDIR" ]; then
	echo "Specified CLEVR directory does not exist!"
	exit
fi

if [ ! -f .venvok ]; then
	echo "Building virtual environment for extracting features"
	mkdir extraction_env
	virtualenv -p /usr/bin/python3 extraction_env
fi

source ./extraction_env/bin/activate

if [ ! -f .venvok ]; then
	echo "Installing dependencies..."
	which pip3
	pip3 install -r requirements.txt

	touch .venvok
else 
	echo "Extraction environment already installed"
fi

#extract 2S-RN features
python3 extract.py --clevr-dir $CLEVRDIR --model 'ir-fp' --checkpoint pretrained_models/ir_fp_epoch_312.pth
#extract RN features
python3 extract.py --clevr-dir $CLEVRDIR --model 'ir-fp' --checkpoint pretrained_models/ir_fp_epoch_312.pth --extr-layer-idx -1

printf "Deactivating extraction virtual environment...\n"
deactivate
