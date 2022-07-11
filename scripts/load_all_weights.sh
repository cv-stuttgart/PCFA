#!/bin/bash
mkdir -p ../models/_pretrained_weights
bash load_raft_weights.sh
bash load_gma_weights.sh
bash load_pwcnet_weights.sh
bash load_flownet2_weights.sh
bash load_spynet_weights.sh