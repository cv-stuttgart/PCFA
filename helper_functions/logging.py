import numpy as np
from datetime import datetime
from os import makedirs, path, listdir
import mlflow
from mlflow import log_metric, log_param, log_artifacts, log_artifact

from helper_functions import losses, ownutilities



def createDateFolder(parent_path, custom_extension=""):
    """
    Creates a new folder with the current time as name in the parent-directory.
    If the parent directory doesn't exit, it is also created.
    
    Args:
        parent_path (str): path to parent folder
        custom_extension (str, optional): an extension to the datestring.
    
    Returns:
        tuple: a tuple containing the full path to the new folder (no tailing "/")
               and the raw date string that is used as name
    """

    time = datetime.now()
    datestr = time.strftime("%Y-%m-%d_%H:%M:%S:%f")
    if not custom_extension == "":
        folder_name = "%s_%s" % (datestr, custom_extension)
    else:
        folder_name = datestr

    folder_path = create_subfolder(parent_path, folder_name)

    return folder_path, folder_name, datestr


def create_subfolder(main_folder, subfolder_name):
    """
    Creates a subfolder in the main folder.

    Args:
        main_folder (str): The main folder
        subfolder_name (str): the subfolder name

    Returns:
        str: path of the newly created folder
    """
    subfolder_path = create_subfolder_name(main_folder, subfolder_name)
    makedirs(subfolder_path, exist_ok=True)

    return subfolder_path

def create_subfolder_name(main_folder, subfolder_name):
    """
    Returns the name of main_folder/subfolder_name

    Args:
        main_folder (str): the main folder
        subfolder_name (str): name for the subfolder

    Returns:
        str: the concatenated folder names
    """
    return path.join(main_folder, subfolder_name)


def mlflow_experimental_setup(exp_basefolder, network_name, attack_name, common_perturbation, universal_perturbation, stage='train'):
    """
    Sets up an mlflow experiment based on network name and attack name.
    If it does not yet exist, it creates a new mlflow experiment (and returns its ID) and an experiment folder within the exp_basefolder.
    exp_basefolder/<nw_name>_<attack_name>/

    Within the new mlflow experiment folder, a folder for a new experimental run is created with a naming
    that uses a mix of the current date, network name and attack name.
    exp_basefolder/<nw_name>_<attack_name>/<curr_date>_<nw_name>_<attack_name>
    
    Args:
        exp_basefolder (str): The main folder to which the data should be logged
        network_name (str): The network name
        attack_name (str): The attack name
        common_perturbation (bool): Indicates if a common perturbation (same delta for both input images 1 and 2) is trained
        universal_perturbation (bool): Indicates if an universal perturbation (same delta for multiple input images) is trained
    
    Returns:
        float, str, str: The experiment id, the folder path for the experiment run and the name of the experimental run folder.
    """

    c_p = "dd"
    u_p = "-"
    if common_perturbation:
        c_p = "cd"
    if universal_perturbation:
        u_p = "u"

    exp_name = "_".join([network_name, attack_name, c_p, u_p])
    if stage=="eval":
        exp_name += "_eval"

    try:
        mlflow.create_experiment(exp_name)#, artifact_location=folder_path)
        _ = create_subfolder(exp_basefolder, exp_name)
    except mlflow.exceptions.MlflowException:
        pass

    folder_path, folder_name, datestr = createDateFolder(create_subfolder(exp_basefolder, exp_name), exp_name)


    exp = mlflow.get_experiment_by_name(exp_name)
    exp_id = exp.experiment_id

    return exp_id, folder_path, folder_name


def log_model_params(model_name, model_takes_unit_input):
    """
    loggs all model related parameters to mlflow
    
    Args:
        model_name (str): the network name
        model_takes_unit_input (bool): a boolean value that specifies if the model (without modification) takes input images in [0,1]
    """
    log_param("model", model_name)
    log_param("model_unitinput", model_takes_unit_input)


def log_dataset_params(dataset_name, dataset_batchsize, dataset_epochs, dataset_subsplit, dataset_type, dataset_stage):
    """
    loggs all dataset related parameters to mlflow
    
    Args:
        dataset_name (str): the dataset name
        dataset_batchsize (int): the batch size to be used for the dataset
        dataset_subsplit (bool): if True this signifies that a subsplit of the dataset is used
    """
    log_param("dataset_name", dataset_name)
    log_param("dataset_bsize", dataset_batchsize)
    log_param("dataset_epochs", dataset_epochs)
    log_param("dataset_subsplit", dataset_subsplit)
    if dataset_name == 'Sintel':
        log_param("dataset_type", dataset_type)
    log_param("dataset_stage", dataset_stage)


def log_attack_params(attack_name, attack_loss, attack_target, attack_commonperturbation, attack_universalperturbation, random_scale=1., custom_target_path=""):
    """
    loggs all attack related parameters to mlflow
    
    Args:
        attack_name (str): the attack name
        attack_loss (str): the loss function used for the attack
        attack_target (str): the attack target
        attack_commonperturbation (bool): if true, a joint perturbation is trained for input images 1 and 2
    """
    log_param("attack_name", attack_name)
    log_param("attack_loss", attack_loss)
    log_param("attack_target", attack_target)
    log_param("attack_common_perturbation", attack_commonperturbation)
    log_param("attack_universal_perturbation", attack_universalperturbation)
    if attack_target == "scaled_random_flow":
        log_param("attack_target_scale", random_scale)
    if attack_target == "custom":
        log_param("attack_target_path", custom_target_path)


def calc_metrics_adv(flow_pred, target, flow_pred_init):
    """Compute the average endpoint errors of
    1. (adversarial) flow <-> target flow and
    2. (adversarial) flow <-> (unattacked) initial flow

    Args:
        flow_pred (tensor): flow field (resulting from perturbed input images)
        target (tensor): target flow field (for which the perturbation is optimized)
        flow_pred_init (_type_): initial (unattacked) flow field

    Returns:
        float, float: scalar values of the endpoint errors
    """

    epe_pred_target = losses.avg_epe(flow_pred, target)
    epe_pred_pred_init = losses.avg_epe(flow_pred, flow_pred_init)

    # print("avg ADV-EPE pr-init: %.12f, avg ADV-EPE tgt    : %.12f" % (epe_pred_pred_init,epe_pred_target))

    return float(epe_pred_target), float(epe_pred_pred_init)


def calc_metrics_adv_gt(flow_pred, flow_gt):
    """Compute the average end point error between predicted flow field and ground truth

    Args:
        flow_pred (tensor): predicted flow field
        flow_gt (tensor): ground truth flow field

    Returns:
        float: scalar end point error
    """

    epe_pred_gt = losses.avg_epe(flow_pred, flow_gt)

    # print("avg ADV-EPE      gt: %.12f" % (epe_pred_gt))

    return float(epe_pred_gt)


def calc_metrics_const(target, flow_pred_init):
    """Compute the average end point error between target and unattacked initial flow field

    Args:
        target (tensor): flow field representing the target of an attack
        flow_pred_init (tensor): initial (unattacked) flow field

    Returns:
        float: scalar end point error
    """

    epe_target_pred_init = losses.avg_epe(target, flow_pred_init)

    # print("                                     avg EPE tgt-pr-init: %.12f" % (epe_target_pred_init))

    return float(epe_target_pred_init)


def calc_metrics_const_gt(target, flow_pred_init, flow_gt):
    """Compute the average endpoint errors of
    1. ground truth flow <-> target flow and
    2. ground truth flow <-> (unattacked) initial flow

    Args:
        target (_type_): _description_
        flow_pred_init (_type_): _description_
        flow_pred_init (_type_): initial (unattacked) flow field
        flow_gt (_type_): _description_
        flow_gt (tensor): ground truth flow field 

    Returns:
        float, float: scalar values of the endpoint errors
    """

    epe_target_gt = losses.avg_epe(target, flow_gt)
    epe_pred_init_gt = losses.avg_epe(flow_pred_init, flow_gt)

    # print("avg EPE  pr-init-gt: %.12f\navg EPE      tgt-gt: %.12f" % (epe_pred_init_gt, epe_target_gt))

    return float(epe_target_gt), float(epe_pred_init_gt)


def calc_delta_metrics(delta1, delta2, step):
    """
    Computes the L2 norm for image distortions and loggs them in MLFlow for a given step.
    
    Args:
        delta1 (tensor): the distortion for image 1
        delta2 (tensor): the distortion for image 2
        step (int): the training step
    """
    l2_delta1 = ownutilities.torchfloat_to_float64(losses.two_norm_avg(delta1))
    l2_delta2 = ownutilities.torchfloat_to_float64(losses.two_norm_avg(delta2))
    l2_delta12 = ownutilities.torchfloat_to_float64(losses.two_norm_avg_delta(delta1, delta2))

    # print("avg delta1         : %.12f, avg delta2         : %.12f, avg delta          : %.12f" % (l2_delta1, l2_delta2, l2_delta12))

    return l2_delta1, l2_delta2, l2_delta12


def save_tensor(tens, tensor_name, batch, output_folder, unregistered_artifacts=True):
    """
    Saves a distortion tensor as .npy object to a specified output folder.
    In case the perturbation for image 1 and 2 are the same, setting common_perturbation=True only saves one instead of both distortions to save memory.
    
    Args:
        delta1 (tensor): the distortion tensor for image 1
        delta2 (tensor): the distortion tensor for image 2
        batch (int): a sample counter
        output_folder (str): the folder to which the distortion files should be saved
        common_perturbation (bool, optional): If true, only delta1 is saved because both delta1 and delta2 are assumed to be the same.
    """

    number = "{:05d}".format(batch)

    filename = "_".join([number, tensor_name + ".npy"])
    filepath = path.join(output_folder, filename)

    tensor_data = tens.clone().detach().cpu().numpy()
    np.save(filepath, tensor_data)
    if not unregistered_artifacts:
        log_artifact(filepath)


def save_image(image_data, batch, output_folder, image_name='image', unit_input=True, normalize_max=None, unregistered_artifacts=True):
    """
    Saves a distortion tensor as .npy object to a specified output folder.
    In case the perturbation for image 1 and 2 are the same, setting common_perturbation=True only saves one instead of both distortions to save memory.
    
    Args:
        delta1 (tensor): the distortion tensor for image 1
        delta2 (tensor): the distortion tensor for image 2
        batch (int): a sample counter
        output_folder (str): the folder to which the distortion files should be saved
        common_perturbation (bool, optional): If true, only delta1 is saved because both delta1 and delta2 are assumed to be the same.
    """

    number = "{:05d}".format(batch)

    filename = "_".join([number, image_name + ".png"])
    filepath = path.join(output_folder, filename)

    image_data = image_data.clone().detach()

    if normalize_max is not None:
        image_data = image_data / normalize_max / 2. + 0.5
        unit_input = True
    if unit_input:
        image_data = image_data * 255.

    ownutilities.quickvisualization_tensor(image_data, filepath)
    if not unregistered_artifacts:
        log_artifact(filepath)


def save_flow(flow, batch, output_folder, flow_name='flowgt', auto_scale=True, max_scale=-1, unregistered_artifacts=True):
    """
    Saves a distortion tensor as .npy object to a specified output folder.
    In case the perturbation for image 1 and 2 are the same, setting common_perturbation=True only saves one instead of both distortions to save memory.
    
    Args:
        delta1 (tensor): the distortion tensor for image 1
        delta2 (tensor): the distortion tensor for image 2
        batch (int): a sample counter
        output_folder (str): the folder to which the distortion files should be saved
    """
    number = "{:05d}".format(batch)

    filename = "_".join([number, flow_name + ".png"])
    filepath = path.join(output_folder, filename)

    flow_data = flow.clone()
    ownutilities.quickvisualization_flow(flow_data, filepath, auto_scale=auto_scale, max_scale=max_scale)
    if not unregistered_artifacts:
        log_artifact(filepath)



def log_metrics(step, *args):
    """
    Loggs given tuples of (metric_name, metric_value) for the specified step.
    with MLFlow.
    
    Args:
        step (int): The training step
        *args: for every metric a tuple with (name, value).
    """
    for (metric_name, metric_value) in args:
        if metric_value is not None:
            log_metric(key=metric_name, value=metric_value, step=step)


def calc_log_averages(numsteps, *args):
    """
    Calculates averages from accumulated values over a specified number of steps and loggs the averages.
    
    Args:
        numsteps (int): the number of steps, over which the averages should be computed
        *args: tuples of (name, sum) where name specifies under which name the average should be logged, and sum is the accumulated sum of the metric over numsteps steps.
    """
    for (logname, value) in args:
        if value is not None:
            avg = value / numsteps
            log_metric(logname, avg)
            # print(logname + ": " + str(avg))
        # else:
            # print(logname + ": " + str(value))

