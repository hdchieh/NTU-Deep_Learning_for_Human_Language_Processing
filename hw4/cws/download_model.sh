#!/bin/bash

# Download dataset from Google Drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uwp-dL-LYF1nebP0J8wZQjJ0T5B88swV' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uwp-dL-LYF1nebP0J8wZQjJ0T5B88swV" -O ./my_cws_bert.pt && rm -rf /tmp/cookies.txt
