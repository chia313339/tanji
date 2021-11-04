#!/bin/bash

echo "測試使用"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate tanji

cd /home/coder/crontab 
python stock_list_stats.py >> /home/coder/crontab/stock_list_stats.log 2>&1


echo "結束"