#!/bin/bash

echo "測試使用"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tanji

cd ~/crontab 
python ustock.py > ~/crontab/ustock.log 2>&1


echo "結束"