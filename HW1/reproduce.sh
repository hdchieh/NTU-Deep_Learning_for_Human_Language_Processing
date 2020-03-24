#!/bin/bash

# Download dataset from Google Drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Cq_q2x_Qk0RH1Jn87MFUR23svo3fId8N' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Cq_q2x_Qk0RH1Jn87MFUR23svo3fId8N" -O ckpt.tar.gz && rm -rf /tmp/cookies.txt

# Unzip the dataset
tar zxvf ckpt.tar.gz --no-same-owner > log && rm log && rm ckpt.tar.gz

sed -i "10c\    test_path: '${1}'" config/dlhlp/decode_dlhlp.yaml

python3 main.py --config config/dlhlp/decode_dlhlp.yaml --test --njobs ${NJOBS}

python3 format.py result/decode_dlhlp_test_output.csv "${2}"
