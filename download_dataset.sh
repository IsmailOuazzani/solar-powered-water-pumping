#!/bin/bash

set -e
list_path="/home/ismail/code/solar-powered-water-pumping/dataset_lists/subset_M2T1NXRAD_5.12.4_20250329_192227_.txt"
token="$EARTHDATA_TOKEN"
output_dir="/media/ismail/BIG/datasets/M2T1NXRAD_5-2015-2025_only_SWGDN"

mkdir ${output_dir}

cat ${list_path} | tr -d '\r' | xargs -n 1 -P 50 -I {} \
  curl -LJO -H "Authorization: bearer ${token}" --output-dir "${output_dir}" "{}"