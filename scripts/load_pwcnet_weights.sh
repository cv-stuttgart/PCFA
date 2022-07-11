#!/bin/bash
mkdir -p ../models/_pretrained_weights
# download load weights for pwcnet and store under pretrained weights
wget https://github.com/NVlabs/PWC-Net/raw/master/PyTorch/pwc_net_chairs.pth.tar
mv  pwc_net_chairs.pth.tar ../models/_pretrained_weights