import json
import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from argparse import Namespace
import os
import sys
#required to prevent ModuleNotFoundError for 'flow_plot'. The flow_library is a submodule, which imports its own functions and can therefore not be imported with flow_library.flow_plot
sys.path.append("flow_library")


from PIL import Image

from torch.utils.data import DataLoader, Subset
from helper_functions import datasets
from helper_functions.config_paths import Paths, Conf
from flow_plot import colorplot_light


class InputPadder:
	"""Pads images such that dimensions are divisible by divisor

	This method is taken from https://github.com/princeton-vl/RAFT/blob/master/core/utils/utils.py
	"""
	def __init__(self, dims, divisor=8, mode='sintel'):
		self.ht, self.wd = dims[-2:]
		pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
		pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
		if mode == 'sintel':
			self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
		else:
			self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

	def pad(self, *inputs):
		"""Pad a batch of input images such that the image size is divisible by the factor specified as divisor

		Returns:
			list: padded input images
		"""
		return [F.pad(x, self._pad, mode='replicate') for x in inputs]

	def get_dimensions(self):
		"""get the original spatial dimension of the image

		Returns:
			int: original image height and width
		"""
		return self.ht, self.wd

	def unpad(self,x):
		"""undo the padding and restore original spatial dimension

		Args:
			x (tensor): a tensor with padded dimensions

		Returns:
			tesnor: tensor with removed padding (i.e. original spatial dimension)
		"""
		ht, wd = x.shape[-2:]
		c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
		return x[..., c[0]:c[1], c[2]:c[3]]

def import_and_load(net='RAFT', make_unit_input=False, variable_change=False, device=torch.device("cpu"), make_scaled_input_model=False, **kwargs):
	"""import a model and load pretrained weights for it

	Args:
		net (str, optional):
			the desired network to load. Defaults to 'RAFT'.
		make_unit_input (bool, optional):
			model will assume input images in range [0,1] and transform to [0,255]. Defaults to False.
		variable_change (bool, optional):
			apply change of variables (COV). Defaults to False.
		device (torch.device, optional):
			changes the selected device. Defaults to torch.device("cpu").
		make_scaled_input_model (bool, optional):
			load a scaled input model which uses make_unit_input and variable_change as specified. Defaults to False.

	Raises:
		RuntimeWarning: Unknown model type

	Returns:
		torch.nn.Module: PyTorch optical flow model with loaded weights 
	"""

	if make_unit_input==True or variable_change==True or make_scaled_input_model:
		from helper_functions.own_models import ScaledInputModel
		model = ScaledInputModel(net, make_unit_input=make_unit_input, variable_change=variable_change, device=device, **kwargs)
		print("--> transforming model to 'make_unit_input'=%s, 'variable_change'=%s\n" % (str(make_unit_input), str(variable_change)))

	else:
		model = None
		try:
			if net == 'RAFT':
				from models.raft.raft import RAFT

				# set the path to the corresponding weights for initializing the model
				path_weights = 'models/_pretrained_weights/raft-sintel.pth'

				# possible adjustements to the config can be made in the file
				# found under models/_config/raft_config.json
				with open("models/_config/raft_config.json") as file:
					config = json.load(file)

				model = torch.nn.DataParallel(RAFT(config))
				# load pretrained weights
				model.load_state_dict(torch.load(path_weights, map_location=device))

			elif net == 'GMA':
				from models.gma.network import RAFTGMA

				# set the path to the corresponding weights for initializing the model
				path_weights = 'models/_pretrained_weights/gma-sintel.pth'

				# possible adjustements to the config file can be made
				# under models/_config/gma_config.json
				with open("models/_config/gma_config.json") as file:
					config = json.load(file)
					# GMA accepts only a Namespace object when initializing
					config = Namespace(**config)

				model = torch.nn.DataParallel(RAFTGMA(config))

				model.load_state_dict(torch.load(path_weights, map_location=device))

			elif net =='PWCNet':
				from models.PWCNet.PWCNet import PWCDCNet

				# set path to pretrained weights:
				path_weights = 'models/_pretrained_weights/pwc_net_chairs.pth.tar'

				model = PWCDCNet()

				weights = torch.load(path_weights, map_location=device)
				if 'state_dict' in weights.keys():
					model.load_state_dict(weights['state_dict'])
				else:
					model.load_state_dict(weights)
				model.to(device)

			elif net =='SpyNet':
				from models.SpyNet.SpyNet import Network as SpyNet
				# weights for SpyNet are loaded during initialization
				model = SpyNet(nlevels=6, pretrained=True)
				model.to(device)

			elif net == "FlowNet2":
				from models.FlowNet.FlowNet2 import FlowNet2

				# hard coding configuration for FlowNet2
				args_fn = Namespace(fp16=False, rgb_max=255.0)

				# set path to pretrained weights
				path_weights = 'models/_pretrained_weights/FlowNet2_checkpoint.pth.tar'
				model = FlowNet2(args_fn, div_flow=20, batchNorm=False)

				weights = torch.load(path_weights, map_location=device)
				model.load_state_dict(weights['state_dict'])

				model.to(device)

			if model is None:
				raise RuntimeWarning('The network %s is not a valid model option for import_and_load(network). No model was loaded. Use "RAFT", "GMA", "FlowNetC", "PWCNet" or "SpyNet" instead.' % (net))
		except FileNotFoundError as e:
			print("\nLoading the model failed, because the checkpoint path was invalid. Are the checkpoints placed in models/_pretrained_weights/? If this folder is empty, consider to execute the checkpoint loading script from scripts/load_all_weights.sh. The full error that caused the loading failure is below:\n\n%s" % e)
			exit()

		print("--> flow network is set to %s" % net)
	return model

def prepare_dataloader(mode='training', dataset='Sintel', shuffle=False, batch_size=1, small_run=False, sintel_subsplit=False, dstype='clean'):
	"""Get a PyTorch dataloader for the specified dataset

	Args:
		mode (str, optional):
			Specify the split of the dataset [training | evaluation]. Defaults to 'training'.
		dataset (str, optional):
			Specify the dataset used [Sintel | Kitti15]. Defaults to 'Sintel'.
		shuffle (bool, optional):
			Use random sampling. Defaults to False.
		batch_size (int, optional):
			Defaults to 1.
		small_run (bool, optional):
			For debugging: Will load only 32 images. Defaults to False.
		sintel_subsplit (bool, optional): 
			Specific for Sintel dataset. If a subsplit is available (and the path correctly specified under config_paths.py),
			will use the subsplit for Sintel. Defaults to False.
		dstype (str, optional):
			Specific for Sintel dataset. Dataset type [clean | final] . Defaults to 'clean'.

	Raises:
		ValueError: Unknown mode.
		ValueError: Unkown dataset.

	Returns:
		torch.utils.data.DataLoader: Dataloader which can be used for FGSM.
	"""

	if dataset == 'Sintel':
		if not sintel_subsplit:
			if mode == 'training':
				dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"),
					root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=True)
			elif mode == 'evaluation':
				# with this option, ground truth and valid are None!!
				dataset = datasets.MpiSintel(split=Paths.splits("sintel_eval"),
					root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=False)
			else:
				raise ValueError(f'The specified mode: {mode} is unknown.')
		else:
			if mode == 'training':
				dataset = datasets.MpiSintelSubsplit(split=Paths.splits("sintel_sub_train"),
					root=Paths.config("sintel_subsplit"), dstype=dstype, has_gt=True)
			elif mode == 'evaluation':
				dataset = datasets.MpiSintelSubsplit(split=Paths.splits("sintel_sub_eval"),
					root=Paths.config("sintel_subsplit"), dstype=dstype, has_gt=True)
			else:
				raise ValueError(f'The specified mode: {mode} is unknown.')


	elif dataset == 'Kitti15':
		if mode == 'training':
			dataset = datasets.KITTI(split=Paths.splits("kitti_train"), aug_params=None, root=Paths.config("kitti15"), has_gt=True)
		elif mode == 'evaluation':
			dataset = datasets.KITTI(split=Paths.splits("kitti_eval"), aug_params=None, root=Paths.config("kitti15"), has_gt=False)

		else: 
			raise ValueError("Unknown dataset %s, use either 'Sintel' or 'Kitti15'." %(dataset))

	# if e.g. the evaluation dataset does not provide a ground truth this is specified
	ds_has_gt = dataset.has_groundtruth()

	if small_run:
		rand_indices = np.random.randint(0, len(dataset), 32)
		indices = np.arange(0, 32)
		dataset = Subset(dataset, indices)

	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), ds_has_gt


def preprocess_img(network, *images):
	"""Manipulate input images, such that the specified network is able to handle them

	Args:
		network (str):
			Specify the network to which the input images are adapted

	Returns:
		InputPadder, *tensor:
			returns the Padder object used to adapt the image dimensions as well as the transformed images
	"""
	if network == 'RAFT' or network == "GMA":
		padder = InputPadder(images[0].shape)
		output = padder.pad(*images)

	elif network == 'PWCNet':
		images = [(img / 255.) for img in images]
		padder = InputPadder(images[0].shape, divisor=64)
		output = padder.pad(*images)

	elif network == 'SpyNet':
		# normalize images to [0, 1]
		images = [ img / 255. for img in images ]
		# make image divisibile by 64
		padder = InputPadder(images[0].shape, divisor=64)
		output = padder.pad(*images)

	elif network[:7] == 'FlowNet':
		# normalization only for FlowNet, not FlowNet2
		if not network[:8] == 'FlowNet2':
			images = [ img / 255. for img in images ]
		# make image divisibile by 64
		padder = InputPadder(images[0].shape, divisor=64)
		output = padder.pad(*images)

	else:
		padder = None
		output = images

	return padder, output


def	postprocess_flow(network, padder, *flows):
	"""Manipulate the output flow by removing the padding

	Args:
		network (str): name of the network used to create the flow
		padder (InputPadder): instance of InputPadder class used during preprocessing
		flows (*tensor): (batch) of flow fields

	Returns:
		*tensor: output with removed padding
	"""

	if padder != None:
		# remove padding
		return [padder.unpad(flow).cpu() for flow in flows]
	else:
		return flows


def compute_flow(model, network, x1, x2, test_mode=True, **kwargs):
	"""subroutine to call the forward pass of the network

	Args:
		model (torch.nn.module):
			instance of optical flow model
		network (str):
			name of the network. [scaled_input_model | RAFT | GMA | FlowNet2 | SpyNet | PWCNet]
		x1 (tensor):
			first image of a frame sequence
		x2 (tensor):
			second image of a frame sequence
		test_mode (bool, optional):
			applies only to RAFT and GMA such that the forward call yields only the final flow field. Defaults to True.

	Returns:
		tensor: optical flow field
	"""
	if network == "scaled_input_model":
		flow = model(x1,x2, test_mode=True, **kwargs)

	elif network == 'RAFT':
		_, flow = model(x1, x2, test_mode=test_mode, **kwargs)

	elif network == 'GMA':
		_, flow = model(x1, x2, iters=6, test_mode=test_mode, **kwargs)

	elif network[:7] == 'FlowNet':
		# all flow net types need image tensor of dimensions [batch, colors, image12, x, y] = [b,3,2,x,y]
		x = torch.stack((x1, x2), dim=-3)
		# FlowNet2-variants: all fine now, input [0,255] is taken.

		if not network[:8] == 'FlowNet2':
			# FlowNet variants need input in [-1,1], which is achieved by substracting the mean rgb value from the image in [0,1]
			rgb_mean = x.contiguous().view(x.size()[:2]+(-1,)).mean(dim=-1).view(x.size()[:2] + (1,1,1,)).detach()
			x = x - rgb_mean

		flow = model(x)

	else: # works for PWCNet, SpyNet
		flow = model(x1,x2, **kwargs)
	return flow



def model_takes_unit_input(model):
	"""Boolean check if a network needs input in range [0,1] or [0,255]

	Args:
		model (str):
			name of the model

	Returns:
		bool: True -> [0,1], False -> [0,255]
	"""
	model_takes_unit_input = False
	if model in ["PWCNet", "SpyNet"]:
		model_takes_unit_input = True
	return model_takes_unit_input


def flow_length(flow):
	"""Calculates the length of the flow vectors of a flow field

	Args:
		flow (tensor):
			flow field tensor of dimensions (b,2,H,W) or (2,H,W)

	Returns:
		torch.float: length of the flow vectors f_ij, computed as sqrt(u_ij^2 + v_ij^2) in a tensor of (b,1,H,W) or (1,H,W)
	"""
	flow_pow = torch.pow(flow,2)
	flow_norm_pow = torch.sum(flow, -3, keepdim=True)

	return torch.sqrt(flow_norm_pow)


def maximum_flow(flow):
	"""Calculates the length of the longest flow vector of a flow field

	Args:
		flow (tensor):
			a flow field tensor of dimensions (b,2,H,W) or (2,H,W)

	Returns:
		float: length of the longest flow vector f_ij, computed as sqrt(u_ij^2 + v_ij^2)
	"""
	return torch.max(flow_length(flow)).cpu().detach().numpy()


def quickvis_tensor(t, filename):
	"""Saves a tensor with three dimensions as image to a specified file location.

	Args:
		t (tensor):
			3-dimensional tensor, following the dimension order (c,H,W)
		filename (str):
			name for the image to save, including path and file extension
	"""
	valid = False
	if len(t.size())==3:
		img = t.detach().cpu().numpy()
		valid = True

	elif len(t.size())==4 and t.size()[0] == 1:
		img = t[0,:,:,:].detach().cpu().numpy()
		valid = True 

	else:
		print("Encountered invalid tensor dimensions %s, abort printing." %str(t.size()))

	if valid:
		img = np.rollaxis(img, 0, 3)
		data = img.astype(np.uint8)
		data = Image.fromarray(data)
		data.save(filename)



def quickvisualization_tensor(t, filename):
	"""Saves a batch (>= 1) of image tensors with three dimensions as images to a specified file location.

	Args:
		t (tensor):
			batch of 3-dimensional tensor, following the dimension order (b,c,H,W)
		filename (str):
			name for the image to save, including path and file extension. Batches will append a number at the end of the filename.
	"""
	if len(t.size())==3 or (len(t.size())==4 and t.size()[0] == 1):
		quickvis_tensor(t, filename)

	elif len(t.size())==4:
		for i in range(t.size()[0]):
			if i == 0:
				quickvis_tensor(t[i,:,:,:], filename)
			else:
				quickvis_tensor(t[i,:,:,:], filename+"_"+str(i)+".png")

	else:
		print("Encountered unprocessable tensor dimensions %s, abort printing." %str(t.size()))


def quickvis_flow(flow, filename, auto_scale=True, max_scale=-1):
	"""Saves a flow field tensor with two dimensions as image to a specified file location.

	Args:
		flow (tensor):
			2-dimensional tensor (c=2), following the dimension order (c,H,W) or (1,c,H,W)
		filename (str):
			name for the image to save, including path and file extension.
		auto_scale (bool, optional):
			automatically scale color values. Defaults to True.
		max_scale (int, optional):
			if auto_scale is false, scale flow by this value. Defaults to -1.
	"""
	valid = False
	if len(flow.size())==3:
		flow_img = flow.clone().detach().cpu().numpy()
		valid = True

	elif len(flow.size())==4 and flow.size()[0] == 1:
		flow_img = flow[0,:,:,:].clone().detach().cpu().numpy()
		valid = True 

	else:
		print("Encountered invalid tensor dimensions %s, abort printing." %str(flow.size()))

	if valid:
		# make directory and ignore if it exists
		if not os.path.dirname(filename) == "":
			os.makedirs(os.path.dirname(filename), exist_ok=True)
		# write flow
		flow_img = np.rollaxis(flow_img, 0, 3)
		data = colorplot_light(flow_img, auto_scale=auto_scale, max_scale=max_scale, return_max=False)
		data = data.astype(np.uint8)
		data = Image.fromarray(data)
		data.save(filename)


def quickvisualization_flow(flow, filename, auto_scale=True, max_scale=-1):
	"""Saves a batch (>= 1) of 2-dimensional flow field tensors as images to a specified file location.

	Args:
		flow (tensor):
			single or batch of 2-dimensional flow tensors, following the dimension order (c,H,W) or (b,c,H,W)
		filename (str):
			name for the image to save, including path and file extension.
		auto_scale (bool, optional):
			automatically scale color values. Defaults to True.
		max_scale (int, optional):
			if auto_scale is false, scale flow by this value. Defaults to -1.
	"""
	if len(flow.size())==3 or (len(flow.size())==4 and flow.size()[0] == 1):
		quickvis_flow(flow, filename, auto_scale=True, max_scale=-1)

	elif len(flow.size())==4:
		for i in range(flow.size()[0]):
			if i == 0:
				quickvis_flow(flow[i,:,:,:], filename, auto_scale=True, max_scale=-1)
			else:
				quickvis_flow(flow[i,:,:,:], filename+"_"+str(i)+".png", auto_scale=True, max_scale=-1)

	else:
		print("Encountered unprocessable tensor dimensions %s, abort printing." %str(flow.size()))


def torchfloat_to_float64(torch_float):
	"""helper function to convert a torch.float to numpy float

	Args:
		torch_float (torch.float):
			scalar floating point number in torch

	Returns:
		numpy.float: floating point number in numpy
	"""
	float_val = np.float(torch_float.detach().cpu().numpy())
	return float_val









