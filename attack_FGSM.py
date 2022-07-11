from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
from mlflow import log_metric, log_param, log_artifacts

from helper_functions import ownutilities, datasets, losses, parsing_file, targets, logging
from helper_functions.config_paths import Conf


# FGSM attack code
def fgsm_attack_step(image1, image2, epsilon, image1_grad, image2_grad, image_min=0., image_max=1., clipping=True, common_perturb=False):
    """
    Performs a step of the Fast Gradient Sign Method (FGSM) by Goodfellow, Shlens, Szegedy "Explaining and harnessing adversarial examples" (2014)

    Args:
        image1 (tensor): The first input image to the Optical Flow estimator
        image2 (tensor): The second input image to the Optical Flow estimator
        epsilon (float): Learning rate or step size. It scales the sign of the image gradients
        image1_grad (tensor): Gradient of Optical Flow estimimator w.r.t. the first image
        image2_grad (tensor): Gradient of Optical Flow estimimator w.r.t. the second image
        image_min (float, optional): minimal image value for clipping after the FGSM update
        image_max (float, optional): maximal image value for clipping after the FGSM update
        clipping (bool, optional): If True, the images are clipped to [image_min, image_max] after the FGSM update.
    
    Returns:
        tensor, tensor: perturbed images
    """
    # Collect the element-wise sign of the data gradient
    if not common_perturb:
        sign_image1_grad = image1_grad.sign()
        sign_image2_grad = image2_grad.sign()
    else:
        image_avg_grad = 0.5 * (image1_grad + image2_grad)
        sign_image1_grad = image_avg_grad.sign()
        sign_image2_grad = image_avg_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    image1_perturbed = image1 - epsilon*sign_image1_grad
    image2_perturbed = image2 - epsilon*sign_image2_grad

    if clipping:
        # Adding clipping to maintain [0,1] range
        image1_perturbed = torch.clamp(image1_perturbed, image_min, image_max)
        image2_perturbed = torch.clamp(image2_perturbed, image_min, image_max)
    # Return the perturbed image

    return image1_perturbed, image2_perturbed


def attack(args):

    """
    Performs an FGSM attack on a given model and for all images of a specified dataset.
    Note that the original FGSM attack assumes the image data to be in [0,1].
    
    The step size epsilon is assumed to scale with the image data in [0,1]. 
    Should the model use data in [0,255] instead, the epsilon will automatically be scaled by 255 to match the image range.
    
    Args: 
        model (string): The model that is used to predict the optical flow.
        dataset (string): The dataset on which the optical flow is predicted
        sintel_subsplit (bool): Indicates if the sintel subsplit should be used.
        epsilon (float): Step size for the FGSM attack, assumes image scale from [0,1]. 
                    It is automatically scaled by 255 if the network indicates a required image range of [0,255] for the model.
        net (): A string to describe the network model. Any of
                'RAFT', 'RAFT_scaled'
    """

    experiment_id, folder_path, folder_name = logging.mlflow_experimental_setup(args.output_folder, args.net, "FGSM", args.joint_perturbation, False)

    print("\nStarting Fast Gradient Sign Method (FGSM) for Optical Flow:")
    print()
    print("\tModel:                   %s" % (args.net))
    print("\tPerturbation universal:  %s" % (str(False)))
    print("\tPerturbation joint:      %s" % (str(args.joint_perturbation)))
    print()
    print("\tTarget:                  %s" % (args.target))
    print("\tOptimizer steps:         %d" % (args.steps))
    print("\tOptimizer stepsize:      %f" % (args.epsilon))
    print()
    print("\tOutputfolder:            %s" % (folder_path))
    print()

    with mlflow.start_run(experiment_id=experiment_id, run_name=folder_name):

        log_param("outputfolder", folder_path)
        distortion_folder_name = "patches"
        distortion_folder_path = folder_path
        distortion_folder = logging.create_subfolder(distortion_folder_path, distortion_folder_name)
        log_artifacts(distortion_folder)

        epsilon = args.epsilon

        logging.log_model_params(args.net, ownutilities.model_takes_unit_input(args.net))
        logging.log_dataset_params(args.dataset, 1, 1, False, args.dstype, args.dataset_stage)
        logging.log_attack_params("FGSM", args.loss, args.target, args.joint_perturbation, False)
        log_param("fgsm_eps", epsilon)
        log_param("optimizer", 'FGSM')
        log_param("optimizer_boxconstraint", 'clipping')
        log_param("optimizer_lr", epsilon)
        log_param("optimizer_steps", args.steps)

        print("Preparing data from %s %s\n" % (args.dataset, args.dataset_stage))
        data_loader, has_gt = ownutilities.prepare_dataloader(args.dataset_stage, 
                                                      dataset=args.dataset,
                                                      shuffle=False, 
                                                      small_run=args.small_run, 
                                                      sintel_subsplit=False, 
                                                      dstype=args.dstype)

        # Define what device we are using

        if Conf.config('useCPU') or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        print("Setting Device to %s\n" % device)

        # Initialize the network
        # load model that takes images scaled to [0,1] as input
        print("Loading model %s:" % (args.net))
        model = ownutilities.import_and_load(args.net, make_unit_input=not ownutilities.model_takes_unit_input(args.net), variable_change=False, make_scaled_input_model=True, device=device)

        # Set the model in evaluation mode. This can be needed for Dropout layers, and is also required for the BatchNorm2dLayers in RAFT (that would otherwise still change in training)
        model.eval()
        # Make sure the model is not trained:
        for param in model.parameters():
            param.requires_grad = False

        # Initialize statistics and Logging
        sum_aee_gt = 0.
        sum_aee_tgt = 0.
        sum_aee_gt_tgt = 0.
        sum_aee_adv_gt = 0.
        sum_aee_adv_tgt = 0.
        sum_aee_adv_pred = 0.
        sum_l2_delta12 = 0.
        tests = 0

        # Loop over all examples in test set
        print("Starting Attack on %s %s\n" % (args.dataset, args.dataset_stage))
        for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):

            curr_step = batch*args.steps

            log_metric(key="batch", value=batch, step=curr_step)
            log_metric(key="steps", value=0, step=curr_step)

            # prepare and rescale images
            image1, image2 = image1.to(device), image2.to(device)
            flow = flow.to(device)
            if not ownutilities.model_takes_unit_input(args.net):
                image1 = image1/255.
                image2 = image2/255.
                img_min = 0.
                img_max = 1.
            else: # Currently not needed, because every non-unit model will be transformed in one that takes unit input by import_and_load.
                img_min = 0.
                img_max = 1.

            # RAFT input padding
            padder, [image1, image2] = ownutilities.preprocess_img(args.net, image1, image2)

            # Set requires_grad attribute of tensor. Important for Attack
            nw_input1 = image1.clone().detach().to(device)
            nw_input2 = image2.clone().detach().to(device)
            nw_input1.requires_grad = True
            nw_input2.requires_grad = True

            # Forward pass the data through the model
            flow_pred = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True)
            [flow_pred] = ownutilities.postprocess_flow(args.net, padder, flow_pred)
            flow_pred = flow_pred.to(device)

            flow_pred_init = flow_pred.detach().clone()
            flow_pred_init.requires_grad = False

            # define attack target
            target = targets.get_target(args.target, flow_pred_init, custom_target_path=args.custom_target_path, device=device)
            target = target.to(device)
            target.requires_grad = False

            aee_tgt            = logging.calc_metrics_const(target, flow_pred_init)
            aee_gt_tgt, aee_gt = logging.calc_metrics_const_gt(target, flow_pred_init, flow) if has_gt else (None, None)

            logging.log_metrics(curr_step, ("aee_pred-tgt", aee_tgt), 
                                           ("aee_gt-tgt", aee_gt_tgt), 
                                           ("aee_pred-gt", aee_gt))

            for step in range(args.steps):
                curr_step = batch*args.steps + step

                # Calculate the loss
                loss = losses.get_loss(args.loss, flow_pred, target)

                # Zero all existing gradients
                model.zero_grad()
                # Calculate gradients of model in backward pass
                loss.backward()

                # Collect datagrad
                nw_input1_grad = nw_input1.grad.data
                nw_input2_grad = nw_input2.grad.data

                # Call FGSM Attack
                nw_input1, nw_input2 = fgsm_attack_step(nw_input1, nw_input2, epsilon, nw_input1_grad, nw_input2_grad, image_min=img_min, image_max=img_max, clipping=True, common_perturb=args.joint_perturbation)

                delta1 = torch.clamp(nw_input1, 0., 1.) - image1
                delta2 = torch.clamp(nw_input2, 0., 1.) - image2
                delta1.detach()
                delta2.detach()

                nw_input1 = nw_input1.detach()
                nw_input2 = nw_input2.detach()
                nw_input1.requires_grad = True
                nw_input2.requires_grad = True

                flow_pred = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True)
                [flow_pred] = ownutilities.postprocess_flow(args.net, padder, flow_pred)
                flow_pred = flow_pred.to(device)

                # More aee statistics, now for attacked images
                aee_adv_tgt, aee_adv_pred = logging.calc_metrics_adv(flow_pred, target, flow_pred_init)
                aee_adv_gt                = logging.calc_metrics_adv_gt(flow_pred, flow) if has_gt else None
                logging.log_metrics(curr_step, ("aee_predadv-tgt", aee_adv_tgt),
                                               ("aee_pred-predadv", aee_adv_pred), 
                                               ("aee_predadv-gt", aee_adv_gt))

                l2_delta1, l2_delta2, l2_delta12 = logging.calc_delta_metrics(delta1, delta2, curr_step)
                logging.log_metrics(curr_step, ("l2_delta1", l2_delta1),
                                               ("l2_delta2", l2_delta2), 
                                               ("l2_delta-avg", l2_delta12))


            if ((batch % args.save_frequency == 0 and not args.small_save) or (args.small_save and batch < 32)) and not args.no_save:

                logging.save_tensor(delta1, "delta1_final", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(delta2, "delta2_final", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(image1, "image1", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(image2, "image2", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(target, "target", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(flow_pred, "flow_pred_final", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(flow_pred_init, "flow_pred_init", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                if has_gt: 
                    logging.save_tensor(flow, "flow_gt", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)

                logging.save_image(image1, batch, distortion_folder, image_name='image1', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_image(image2, batch, distortion_folder, image_name='image2', unit_input=True, normalize_max=None, unregistered_artifacts=args)


                max_delta = np.max([ownutilities.torchfloat_to_float64(torch.max(delta1)), 
                                    ownutilities.torchfloat_to_float64(torch.max(delta2))])

                logging.save_image(delta1, batch, distortion_folder, image_name='delta1', unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)
                if not args.joint_perturbation:
                    logging.save_image(delta2, batch, distortion_folder, image_name='delta2', unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)

                max_flow_gt = 0
                if has_gt:
                    max_flow_gt = ownutilities.maximum_flow(flow)
                max_flow = np.max([max_flow_gt, 
                                   ownutilities.maximum_flow(flow_pred_init), 
                                   ownutilities.maximum_flow(flow_pred)])

                logging.save_flow(flow_pred, batch, distortion_folder, flow_name='flow_pred_final', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_flow(flow_pred_init, batch, distortion_folder, flow_name='flow_pred_init', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_flow(target, batch, distortion_folder, flow_name='flow_target', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
                if has_gt:
                    logging.save_flow(flow, batch, distortion_folder, flow_name='flow_gt', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)


            sum_aee_tgt += aee_tgt
            sum_aee_adv_tgt += aee_adv_tgt
            sum_aee_adv_pred += aee_adv_pred
            sum_l2_delta12 += l2_delta12
            if has_gt:
                sum_aee_gt += aee_gt
                sum_aee_gt_tgt += aee_gt_tgt
                sum_aee_adv_gt += aee_adv_gt
            tests += 1


        # Calculate final accuracy
        if (not args.unregistered_artifacts) and (not args.no_save):
            log_artifacts(distortion_folder)
        logging.calc_log_averages(tests,
                ("aee_avg_pred-gt",sum_aee_gt), 
                ("aee_avg_pred-tgt", sum_aee_tgt),
                ("aee_avg_gt-tgt",sum_aee_gt_tgt),
                ("aee_avg_predadv-gt", sum_aee_adv_gt),
                ("aee_avg_predadv-tgt", sum_aee_adv_tgt),
                ("aee_avg_pred-predadv", sum_aee_adv_pred),
                ("l2_avg_delta12", sum_l2_delta12))

        print("\nFinished attacking with FGSM. The best achieved values are")
        print("\tAEE(f_adv, f_init)=%f" % (sum_aee_adv_pred / tests))
        print("\tAEE(f_adv, f_targ)=%f" % (sum_aee_adv_tgt / tests))
        print("\tL2(perturbation)  =%f" % (sum_l2_delta12 / tests))
        print()


if __name__ == '__main__':

    parser = parsing_file.create_parser(stage='training', attack_type='fgsm')

    args = parser.parse_args()

    print(args)

    attack(args)