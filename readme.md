# Perturbation Constrained Flow Attack (PCFA)
This repository contains the source code for our paper

> **A Perturbation Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow**,<br>
> J. Schmalfuss, P. Scholze and A. Bruhn<br>
> 2022,

which is accepted at ECCV 2022. Also refer to our [preprint](https://arxiv.org/abs/2203.13214) for details on the method.


# Initial setup

## Setup virtual environment

```shell
python3 -m venv pcfa
source pcfa/bin/activate
```

## Install required packages:
Change into scripts folder and execute the script which installs all required packages via pip. As each package is installed succesively, you can debug errors for specific packages later.

```shell
cd scripts
bash install_packages.sh
```

### Spatial Correlation Sampler 
If the installation of the `spatial-correlation-sampler` works and you have a cuda capable machine, open `helper_functions/config_paths.py` and make sure to set the variable `"correlationSamplerOnlyCPU":` to `False`. This will speed up computations when using PWCNet.

If the spatial-correlation-sampler does not install run the following script to install a cpu-only version:
```shell
cd scripts
bash install_scs_cpu.sh
```

## Loading Flow Models

Download the weights for a specific model by changing into the `scripts/` directory and executing the bash script for a specific model:
```shell
cd scripts
./load_[model]_weights.sh
```
Here `[model]` should be replaced by one of the following options:
```	
[ all | raft | gma | spynet | pwcnet | flownet2]
```
Note: the load_model scripts remove .git files, which are often write-protected and then require an additional confirmation on removal. To automate this process, consider to execute instead
```shell
yes | ./load_[model]_weights.sh
```

### Compiling Cuda Extensions for FlowNet2

Please refer to the [pytorch documentation](https://github.com/NVIDIA/flownet2-pytorch.git) how to compile the channelnorm, correlation and resample2d extensions.
If all else fails, go to the extension folders `/models/FlowNet/{channelnorm,correlation,resample2d}_package`, manually execute
```
python3 setup.py install
```
and potentially replace `cxx_args = ['-std=c++11']` by `cxx_args = ['-std=c++14']`, and the list of `nvcc_args` by `nvcc_args = []` in every setup.py file.
If manually compiling worked, you may need to add the paths to the respective .egg files in the `{channelnorm,correlation,resample2d}.py files`, e.g. for channelnorm via
```python
sys.path.append("/lib/pythonX.X/site-packages/channelnorm_cuda-0.0.0-py3.6-linux-x86_64.egg")
import channelnorm_cuda
```
The `site-packages` folder location varies depending on your operation system and python version. 

## Datasets

For training and evaluation in PCFA we use the [*Sintel*](http://sintel.is.tue.mpg.de/) and [*KITTI 2015*](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) dataset.
Set correct paths to datasets (after downloading them) in `helper_functions/config_paths.py`


# Code Usage

## Training Perturbations with PCFA

To train perturbations with PCFA, execute
```
python3 attack_PCFA.py --net=[SpyNet,PWCNet,RAFT,GMA,FlowNet2] 
```
By default, this trains disjoint image-specific perturbation with L2 bound 0.005 and a zero-flow target on KITTI15 evaluation, and saves output after every image.
All available argument options are displayed via 
```
python3 attack_PCFA.py --help
```
The following arguments are useful to reproduce the paper results:
```
--net=[SpyNet,PWCNet,RAFT,GMA,FlowNet2] | The network for which to execute
--delta_bound=0.005             | The average L2 bound that is used to train the perturbation, std=0.005

--joint_perturbation            | If this argument is passed, a joint perturbation is trained.
--universal_perturbation        | If this argument is passed, a universal perturbation is trained.

--target=[zero,neg_flow,custom] | Allows to set a target. If a custom target is desired, specify the path to a flow-containing file via --custom_target_path.
--loss=[aee,mse,cosim]          | Allows to specify the loss function
--steps=20                      | The number of LBFGS steps per image. Consider a lower number for --universal_perturbations (e.g. 1).
--boxconstraint=[clipping,change_of_variables]   | Allows to specify the box constraint
```
By default, universal perturbations are trained for `--epochs=25` with a batch-size of `--batch_size=4`; both values are treated as 1 for non-universal perturbations.
Universal perturbations do not return average Adversarial Robustness or Attack strength scores, because the trained perturbations still have to be evaluated on a full dataset via `evaluate_PCFA.py` (see next section).

To specify the datasets and handle how much output is produced, the following arguments help:
```
--dataset=[Kitti15,Sintel]            | Which dataset is used
--dataset_stage=[evaluation,training] | Use evaluation or training split
--dstype=[clean,final]                | Sintel only, specify clean or final data

--save_frequency=1                    | After so many batches are sample outputs (flow fields, targets, perturbations) written
--no_save                             | Setting this option prevents any but the most important output
```

## Evaluating Existing Perturbations from PCFA

To evaluate perturbations, a path to the perturbation(folder) and the network for which the perturbations were trained have to be specified with the `--perturbation_sourcefolder` and `--origin_net` arguments.
The `--perturbation_sourcefolder` can either be the output folder from a universal perturbation training, or a path to a .npy perturbation file.
Additionally, the perturbation specification via `--joint_perturbation` and `--universal_perturbation` should match the trained patch.

```
python3 evaluate_PCFA.py --net=[SpyNet,PWCNet,RAFT,GMA,FlowNet2] --origin_net=<specify_network> --perturbation_sourcefolder=<specify> [--joint_perturbation --universal_perturbation]
```
Further, the dataset on which to evaluate should be specified. Note that perturbations trained for one dataset can only be tested on this dataset (no mixing between KITTI and Sintel perturbations is supported).

Also, the data will be batched according to 
```
--batch_size=4                        | The testing batch size
```

To specify the datasets and handle how much output is produced, the following arguments help:
```
--dataset=[Kitti15,Sintel]            | Which dataset is used
--dataset_stage=[evaluation,training] | Use evaluation or training split
--dstype=[clean,final]                | Sintel only, specify clean or final data

--save_frequency=1                    | After so many batches are sample outputs (flow fields, targets, perturbations) written
--no_save                             | Setting this option prevents any but the most important output
```

## Training and Evaluating (I-)FGSM perturbations

Additionally, we provide a training routine that can generate FGSM perturbations (image specific only).
```
python3 attack_FGSM.py --net=[SpyNet,PWCNet,RAFT,GMA,FlowNet2] 
```
Parameters to provide are the number of iterations via `--steps`, the step size `--epsilon` as well as the normal attack parameters target and loss.
```
--steps=20                      | The number I-FGSM steps per image.
--epsilon=0.00025               | The I-FGSM step size

--joint_perturbation            | If this argument is passed, a joint perturbation is trained.

--target=[zero,neg_flow,custom] | Allows to set a target. If a custom target is desired, specify the path to a flow-containing file via --custom_target_path.
--loss=[aee,mse,cosim]          | Allows to specify the loss function
```
The dataset parameters work as explained for PCFA in the previous sections.


# Data Logging and Progress Tracking

Training progress and output images are tracked with MLFlow in `mlruns/`, and output images and flows are additionally saved in `experiment_data/`.
In `experiment_data/`, the folder structure is `<networkname>_<attacktype>_<perturbationtype>/`, where each subfolder contains different runs of the same network with a specific perturbation type.

To view the mlflow data locally, navitage to the root folder of this repository, execute

```shell
mlflow ui

```
and follow the link that is displayed. This leads to the web interface of mlflow.

If the data is on a remote host, the below procedure will get the mlflow data displayed.

## Progress tracking with MLFlow (remote server)

Identify the remote's public IP adress via

```shell
curl ifconfig.me
```
then start mlflow on remote machine:
```shell
mlflow server --host 0.0.0.0
```
On your local PC, replace 0.0.0.0 with the public IP and visit the following address in a web-browser:
```shell
http://0.0.0.0:5000
```


# Adding External Models

The framework is built such that custom (PyTorch) models can be included. To add an own model, perform the following steps:
1. Create a directory `models/your_model` containing all the required files for the model.
2. Make sure that all import calls are updated to the correct folder. I.e change:
	```python
	from your_utils import your_functions # old

	# should be changed to:
	from models.your_model.your_utils import your_functions # new
	```

3. In [`helper_functions/ownutilities.py`](helper_functions/ownutilities.py) modify the following functions:
	- [`import_and_load()`](helper_functions/ownutilities.py#L64): Add the following lines:
		```python
		elif net == 'your_model':
			# mandatory: import your model i.e:
			from models.your_model import your_model

			# optional: you can outsource the configuration of your model e.g. as a .json file in models/_config/
			with open("models/_config/your_model_config.json") as file:
				config = json.load(file)
			# mandatory: initialize model with your_model and load pretrained weights
			model = your_model(config)
			weights = torch.load(path_weights, map_location=device)
			model = load_state_dict(weights)
		```
	- [`preprocess_img()`](helper_functions/ownutilities.py#L241): Make sure that the input is adapted to the forward pass of your model.
	The dataloader provides rgb images with range `[0, 255]`. The image dimensions differ with the dataset.
	You can use the padder class make the spatial dimensions divisible by a certain divisor.
		```python
		elif network == 'your_model':
			# example: normalize rgb range to [0,1]
			images = [(img / 255.) for img in images]
			# example: initialize padder to make spatial dimension divisible by 64
			padder = InputPadder(images[0].shape, divisor=64)
			# example: apply padding
			output = padder.pad(*images)
		```
		
	- [`model_takes_unit_input()`](helper_functions/ownutilities.py#L347): Add your model to the respective list, if it expects input images in `[0,1]` rather than `[0,255]`.

	- [`compute_flow()`](helper_functions/ownutilities.py#L302): Has to return a tensor `flow` originating from the forward pass of your model with the input images `x1` and `x2`.
	If your model needs further preprocessing like concatenation perform it here:
		```python
		elif network == 'your_model':
			# optional: 
			model_input = torch.cat((x1, x2), dim=0)
			# mandatory: perform forward pass
			flow = model(model_input)
		```
	- `postprocess_flow()`: Rescale the spatial dimension of the output `flow`, such that they coincide with the original image dimensions. If you used the padder class during preprocessing it will be automatically reused here.

4. Add your model to the possible choices for `--net` in [`helper_functions/parsing_file.py`](helper_functions/parsing_file.py#L16) (i.e. `[... | your_model]`)



# External Models and Dependencies

## Models

The models provided under [`models/`](models/) are supposed to help recreate the paper results, but are not part of the published attack.

- [*RAFT*](https://github.com/princeton-vl/RAFT)
- [*GMA*](https://github.com/zacjiang/GMA.git)
- [*SpyNet*](https://github.com/sniklaus/pytorch-spynet.git) and [Flow Attack](https://github.com/anuragranj/flowattack.git)
- [*PWC-Net*](https://github.com/NVlabs/PWC-Net.git) and [Flow Attack](https://github.com/anuragranj/flowattack.git)
- [*FlowNet*](https://github.com/NVIDIA/flownet2-pytorch.git)

## Additional code

- Augmentation and dataset handling (`datasets.py` `frame_utils.py` `InputPadder`) from [RAFT](https://github.com/princeton-vl/RAFT)

- Path configuration (`conifg_specs.py`) inspired by [this post](https://stackoverflow.com/questions/6198372/most-pythonic-way-to-provide-global-configuration-variables-in-config-py)

- File parsing (`parsing_file.py`): idea from [this post](https://stackoverflow.com/a/60418265/13810868)