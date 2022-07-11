import argparse

def create_parser(stage=None, attack_type=None):
	stage = stage.lower()
	attack_type = attack_type.lower()
	if stage not in ['training','evaluation']:
		raise ValueError('To create a parser the stage has to be specified. Please choose one of "training" or "evaluation"')
	if attack_type not in ["fgsm", "pcfa"]:
			raise ValueError('To create a parser the attack type has to be specified. Please choose one of "fgsm" or "pcfa"')


	parser = argparse.ArgumentParser(usage='%(prog)s [options (see below)]')

	# network arguments
	net_args = parser.add_argument_group(title='network arguments')
	net_args.add_argument('--net', default='SpyNet', choices=['RAFT', 'GMA', 'PWCNet', 'SpyNet', 'FlowNet2'],
		help="specify the network under attack")

	# Dataset arguments
	dataset_args = parser.add_argument_group(title="dataset arguments")
	dataset_args.add_argument('--dataset', default='Kitti15', choices=['Kitti15', 'Sintel'],
		help="specify the dataset which should be used for evaluation")
	dataset_args.add_argument('--dataset_stage', default='evaluation', choices=['training', 'evaluation'],
		help="specify the dataset stage ('training' or 'evaluation') that should be used.")
	dataset_args.add_argument('--small_run', action='store_true',
		help="for testing purposes: if specified the dataloader will on load 32 images")
	# Sintel specific:
	sintel_args = parser.add_argument_group(title="sintel specific arguments")	
	sintel_args.add_argument('--dstype', default='final', choices=['clean', 'final'],
		help="[only sintel] specify the dataset type for the sintel dataset")

	# Data saving
	data_save_args = parser.add_argument_group(title="data saving arguments")
	data_save_args.add_argument('--output_folder', default='experiment_data',
		help="data that is logged during training and evaluation will be saved there")
	data_save_args.add_argument('--small_save', action='store_true',
		help="if specified potential extended output will only be produced for the first 32 images.")
	data_save_args.add_argument('--save_frequency', type=int, default=1,
			help="specifies after how many batches intermediate results (patch, input images, flows) should be saved. Default: 1 (save after every batch/image). If --no_save is specified, this overwrites any save_frequency.")
	data_save_args.add_argument('--no_save', action='store_true',
		help="if specified no extended output (like distortions/patches) will be written. This overwrites any value specified by save_frequency.")
	data_save_args.add_argument('--unregistered_artifacts', action='store_true', default=False,
		help="if this flag is used, artifacts are saved to the output folder but not registered. This might save time during training.")


	# Global Distortion Attack specific arguments
	if attack_type in ['fgsm', 'pcfa']:
		global_dist_args = parser.add_argument_group(title="global distortion attack arguments")

		global_dist_args.add_argument('--joint_perturbation', action='store_true', default=False,
			help="this flag should be used if the same global perturbation should be applied to network input images 1 and 2.")
		global_dist_args.add_argument('--steps', default=20, type=int,
			help="the number of optimization steps per image (for non-universal perturbations only).")

		# FGSM unique arguments
		if attack_type in ['fgsm']:
			fgsm_args = parser.add_argument_group(title="fgsm arguments")
			fgsm_args.add_argument('--epsilon', default=0.00025, type=float,
				help="the step size for FGSM attack step")

		# PCFA unique arguments
		if attack_type in ['pcfa']:
			pcfa_args = parser.add_argument_group(title="pcfa arguments")
			pcfa_args.add_argument('--universal_perturbation', action='store_true', default=False,
					help="train an universal perturbation for multiple images from a dataset.")
			pcfa_args.add_argument('--boxconstraint', default='change_of_variables', choices=['clipping', 'change_of_variables'],
				help="the way to enfoce the box constraint on the distortion. Options: 'clipping', 'change_of_variables'.")
			pcfa_args.add_argument('--batch_size', default=4, type=int,
				help="[universal perturbation only] the batch size.")

			if stage == "training":
				pcfa_args.add_argument('--delta_bound', default=0.005, type=float,
					help="This bound should be enforced on the L2 norm of the trained image perturbation delta per pixel.")
				pcfa_args.add_argument('--mu', default=-1, type=float,
					help="The PCFA attempts to solve for a given --delta_bound with a penalty procedure. Mu specifies the weight of the L2 norm constraint on delta. If mu=-1, PCFA will attempt to set mu on a heuristic that works reasonably well for Kitti15 and Sintel. If the optimization fails, hand-tuning this parameter might be required.")
				pcfa_args.add_argument('--epochs', default=25, type=int,
					help="[universal perturbation only] the epochs.")


			if stage == "evaluation":
				pcfa_args.add_argument('--perturbation_sourcefolder',
					help="when evaluating PCFA, please provide the folder that contains the trained patches/perturbations to evaluate. Alternatively, provide a path to a perturbation file as .npy")
				pcfa_args.add_argument('--origin_net',
					help="the network that trained the perturbations which are tested now.")


	# Arguments for training:
	if stage == 'training':
		train_args = parser.add_argument_group(title="training arguments")
		train_args.add_argument('--target', default='zero', choices=['zero', 'neg_flow', 'custom'],
			help="specify the attack target as one flow type out of 'zero', 'neg_flow' and 'custom'. Additionally provide a '--custom_target_path' if 'custom' is chosen")
		train_args.add_argument('--custom_target_path', default='',
			help="specify path to a custom target flow")
		train_args.add_argument('--loss', default='aee', choices=['aee', 'mse', 'cosim'],
			help="specify the loss function as one of 'aee', 'cosim' or 'mse'")


	return parser 
