#!/bin/bash
# mkdir -p ../models/_pretrained_weights
# # download load weights for gma and store under pretrained weights
base_url=https://github.com/zacjiang/GMA/raw/2f1fd29468a86a354d44dd25d107930b3f175043/checkpoints/
file_names=(
    gma-kitti.pth
    gma-sintel.pth
    gma-things.pth
    gma-chairs.pth
)
for name in "${file_names[@]}"; do
    url=${base_url}${name}
    echo "$name"
    echo "$url"
    wget -L $url
    mv -f $name ../models/_pretrained_weights/
done