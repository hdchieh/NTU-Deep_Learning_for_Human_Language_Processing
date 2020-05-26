#!/bin/bash

# Download dataset from Google Drive
FILEID='1x_MnvIRDb70fy6EKXU0wI8rKs_XeZH5b'

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $1 && rm -rf /tmp/cookies.txt
