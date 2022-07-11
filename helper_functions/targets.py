import torch
import torch.nn.functional as F
import numpy as np
import os
import os.path
from helper_functions import ownutilities, frame_utils


def zero_flow(flow):
	"""Create a zero tensor with the same size as flow

	Args:
		flow (tensor): input

	Returns:
		tensor: containing zeros with same dimension as the input
	"""
	return torch.zeros_like(flow)


def neg_flow(flow):
	"""Mirror the input flow by 180 degree

	Args:
		flow (tensor): input flow field

	Returns:
		tensor: reversed flow field
	"""
	return - flow


def custom_target(flow, path_to_custom_target, device):
	"""Load a (custom) flow field and crop / pad the spatial dimension, such that it can be used as a target during PCFA.
	

	Args:
		flow (tensor):
			unattacked flow field of the network (the targets spatial dimensions will be adapted to this reference)
		path_to_custom_target (str):
			Path to a .npy perturbation file
		device (torch.device):
			changes the selected device

	Raises:
		AssertionError: file or file path are invalid

	Returns:
		tensor: perturbation which has the same batch and spatial size as the flow field (b,c,h,w) or (c,h,w)
	"""
	try:
		target_data = frame_utils.read_gen(path_to_custom_target, pil=False)
		if len(target_data) < 2:
			raise AssertionError()
		target_data = np.array(target_data).astype(np.float32)
		target_data = torch.from_numpy(target_data).permute(2, 0, 1).float()
		target_data = target_data.to(device)


		target_size = target_data.size()
		flow_size = flow.size()
		if len(target_size) == 4:
			target_data = target_data[0,:,:,:]

		if flow_size[-1] < target_size[-1]: # if flow smaller than target, crop target
			target_data = target_data[:,:,:flow_size[-1]]
		elif flow_size[-1] > target_size[-1]: # if flow greater than target, padd target
			pd = (0, flow_size[-1]- target_size[-1])
			target_data = F.pad(target_data, pd, "reflect")

		if flow_size[-2] < target_size[-2]: # if flow smaller than target, crop target
			target_data = target_data[:,:flow_size[-2],:]
		elif flow_size[-2] > target_size[-2]: # if flow greater than target, padd target
			pd = (0, 0, 0, flow_size[-2]- target_size[-2])
			target_data = F.pad(target_data, pd, "reflect")

		if len(flow_size) == 4:
			target_data = target_data.unsqueeze(0).repeat(flow_size[0], 1, 1, 1)

	except AssertionError:
		print("WARNING: The specified custom target file is not a valid flow file at %s" % path_to_custom_target)
		print("Please specify a valid flow file via --custom_target_path")
		print("\nExiting attack.")
		exit()
	
	return target_data


def get_target(target_name, flow_pred_init, custom_target_path="", device=None):
	"""Getter method which yields a specified target flow used during PCFA 

	Args:
		target_name (str):
			description the attack target. Options: [zero | negative | custom]
		flow_pred_init (tensor):
			unattacked flow field
		custom_target_path (str, optional):
			if custom target is desired provide the path to a .npy perturbation file. Defaults to "".
		device (_type_, optional): _description_. Defaults to None.

	Raises:
		ValueError: Undefined choice for target.

	Returns:
		tensor: target flow field used during PCFA
	"""
	if target_name == 'zero':
		target = zero_flow(flow_pred_init)
	elif target_name == 'neg_flow':
		target = neg_flow(flow_pred_init)
	elif target_name == 'custom':
		target = custom_target(flow_pred_init, custom_target_path, device)
	else:
		raise ValueError('The specified target type "' + target_name + '" is not defined and cannot be used. Select one of "zero", "neg_flow" or "custom". Aborting.')
	return target