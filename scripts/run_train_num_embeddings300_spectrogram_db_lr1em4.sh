#!/bin/bash

python /home/ubuntu/speech2speech/scripts/train_model.py --verbose --debug\
    --base-dir '/home/ubuntu/speech2speech/test_num_embeddings_500'\
    --data-dir '/home/ubuntu/speech2speech/data/raw/VCTK-Corpus'\
    --spectrogram-dir '/home/ubuntu/speech2speech/data/interim/spectogram_array_trim_30db'\
    --time_length 100\
    --train-data-fraction 0.8\
    --validation-data-fraction 0.1\
    --num-epochs 20\
    --batch-size 10\
    --num-hiddens 768\
    --num-residual-hiddens 32\
    --num-residual-layers 2\
    --embedding-dim 64\
    --num-embeddings 300\
    --speaker-embedding-dim 20\
    --commitment-cost 0.25\
    --decay 0\
    --learning-rate 1e-4
