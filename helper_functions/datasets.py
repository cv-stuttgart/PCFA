'''
This file is partially taken from https://github.com/princeton-vl/RAFT/, which is distributed under the following license:


BSD 3-Clause License

Copyright (c) 2020, princeton-vl
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F

import os
import math
import random
from glob import glob
import os.path as osp

from helper_functions import frame_utils
from helper_functions.config_paths import Paths


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.sparse = sparse

        self.has_gt = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.enforce_dimensions = False
        self.image_x_dim = 0
        self.image_y_dim = 0

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        valid = None

        if self.has_gt:
            if self.sparse:
                flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
            else:
                flow = frame_utils.read_gen(self.flow_list[index])
            flow = np.array(flow).astype(np.float32)

            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

            if valid is not None:
                valid = torch.from_numpy(valid)
            else:
                valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        else:
            (img_x, img_y, img_chann) = img1.shape

            flow = np.zeros((img_x, img_y, 2)) # make correct size for flow (2 dimensions for u,v instead of 3 [r,g,b] for image )
            valid = False

            flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        if self.enforce_dimensions:
            dims = img1.size()
            x_dims = dims[-2]
            y_dims = dims[-1]

            diff_x = self.image_x_dim - x_dims
            diff_y = self.image_y_dim - y_dims

            img1 = F.pad(img1, (0,diff_y,0,diff_x), "constant", 0)
            img2 = F.pad(img2, (0,diff_y,0,diff_x), "constant", 0)

            flow = F.pad(flow, (0,diff_y,0,diff_x), "constant", 0)
            if self.has_gt:
                valid = F.pad(valid, (0,diff_y,0,diff_x), "constant", False)


        return img1, img2, flow, valid


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)

    def has_groundtruth(self):
        return self.has_gt


class MpiSintel(FlowDataset):
    def __init__(self, aug_params=None, split='training', root=Paths.config("sintel_mpi"), dstype='clean', has_gt=False):
        super(MpiSintel, self).__init__(aug_params)
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        self.has_gt = has_gt

        for scene in os.listdir(image_root):
            image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
            for i in range(len(image_list)-1):
                self.image_list += [ [image_list[i], image_list[i+1]] ]
                self.extra_info += [ (scene, i) ] # scene and frame_id

            if split != 'test':
                self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))

        if len(self.image_list) == 0:
            raise RuntimeWarning("No MPI Sintel data found at dataset root '%s'. Check the configuration file under helper_functions/config_paths.py and add the correct path to the MPI Sintel dataset." % root)


class KITTI(FlowDataset):
    def __init__(self, aug_params=None, split='training', root=Paths.config("kitti15"), has_gt=False):
        super(KITTI, self).__init__(aug_params, sparse=True)

        self.has_gt = has_gt

        root = osp.join(root, split)
        images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
        images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

        for img1, img2 in zip(images1, images2):
            frame_id = img1.split('/')[-1]
            self.extra_info += [ [frame_id] ]
            self.image_list += [ [img1, img2] ]

        if self.has_gt:
            self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))

        self.enforce_dimensions = True
        self.image_x_dim = 375
        self.image_y_dim = 1242

        if len(self.image_list) == 0:
            raise RuntimeWarning("No KITTI data found at dataset root '%s'. Check the configuration file under helper_functions/config_paths.py and add the correct path to the KITTI dataset." % root)