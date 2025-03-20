#!/bin/bash

# Change to the base directory (parent of scripts directory)
cd "$(dirname "$0")/.." || exit 1

name='Digits_DER'
debug='1'
comments='None'
expid='digits'

# Now we're in the base directory, so Python can find the main module
if [ "$debug" = "0" ]; then
    python -m main train with "./configs/${expid}.yaml" \
        exp.name="${name}" \
        exp.savedir="./logs/" \
        exp.ckptdir="./logs/" \
        exp.tensorboard_dir="./tensorboard/" \
        trial=0 \
        --name="${name}" \
        -D \
        -p \
        -c "${comments}" \
        --force \
        --mongo_db=10.10.10.100:30620:classil
else
    python -m main train with "./configs/${expid}.yaml" \
        exp.name="${name}" \
        exp.savedir="./logs/" \
        exp.ckptdir="./logs/" \
        exp.tensorboard_dir="./tensorboard/" \
        exp.debug=True \
        --name="${name}" \
        -D \
        -p \
        --force 
    # --mongo_db=10.10.10.100:30620:debug
fi
