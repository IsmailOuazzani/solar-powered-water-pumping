#!/bin/bash

set -e
list_path="/home/ismail/code/spwp/dataset_lists/subset_M2T1NXRAD_5.12.4_20241118_214524_.txt"
token="$EARTHDATA_TOKEN"
output_dir="/media/ismail/BIG/datasets/M2T1NXRAD_5-1995/"

mkdir ${output_dir}

cat ${list_path} | tr -d '\r' | xargs -n 1 -P 30 -I {} \
  curl -LJO -H "Authorization: bearer ${token}" --output-dir "${output_dir}" "{}"