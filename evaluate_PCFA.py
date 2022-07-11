from __future__ import print_function
import argparse
import os
import re
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from torchvision import datasets, transforms
from tqdm import tqdm
from mlflow import log_metric, log_param

from helper_functions import ownutilities, datasets, parsing_file, logging, losses
from helper_functions.config_paths import Conf, Paths


def extract_epoch_patchlist(path):

    delta1_list = []
    delta2_list = []
    epochs = 0
    print("Loading existing perturbation(s) from\n%s" % path)
    if os.path.isfile(path):
        epochs = 1
        _, extension = os.path.splitext(path)

        if extension == ".npy":
            delta1_list = [path]
        else:
            raise ValueError("Invalid extension %s for perturbation file, please use a .npy file instead of %s" % (extension, path))

        print("\tFound path to a perturbation file. Evaluating one perturbation (epochs=1) only.")

    else:
        base_folder = os.path.join(path, "patches")

        # searches for strings of the form "BBBBB_delta1_eEE.npy" where BBBBB and EE are counters for batch and epoch respectively; Files of this sort are produced by attack_PCFA --universal_perturbation over multiple epochs.
        pattern1 = re.compile("[0-9]{5}_delta1_e[0-9]*.npy") 
        pattern2 = re.compile("[0-9]{5}_delta2_e[0-9]*.npy")
        for file in os.listdir(base_folder):
            if pattern1.match(file):
                delta1_list += [os.path.join(base_folder,file)]
            if pattern2.match(file):
                delta2_list += [os.path.join(base_folder,file)]

        delta1_list = np.sort(delta1_list)
        delta2_list = np.sort(delta2_list)

        epochs = int(delta1_list[-1].split("_")[-1].split(".")[0][1:])
        epochs = epochs+1 # logging starts to count epochs with 0, hence add 1

        print("\tFound path to folder that contains perturbation files from %d epochs. Evaluating each epoch perturbation." % epochs)

    return epochs, delta1_list, delta2_list

def convert_perturbationsizes(delta, image, network_training, network_eval, dataset):
    nws_fnetpadd = ["PWCNet", "SpyNet", "FlowNet2"]
    nws_raftpadd = ["RAFT", "GMA"]
    nws_unitinput = ["PWCNet", "SpyNet"]
    if (network_training in nws_fnetpadd and network_eval in nws_fnetpadd) or (network_training in nws_raftpadd and network_eval in nws_raftpadd):
        delta_repadded = delta
    else:
        print("Changing padding when importing perturbation trained for %s to evaluate it on %s" % (network_training, network_eval))
        padder_train, _ = ownutilities.preprocess_img(network_training, image.detach().clone())

        delta_unpadded = padder_train.unpad(delta)
        delta_unpadded = torch.unsqueeze(delta_unpadded, 0)

        # this step might return delta/255
        padder_eval, [delta_repadded] = ownutilities.preprocess_img(network_eval, delta_unpadded.detach().clone())

        if network_eval in nws_unitinput: # This is necessary because FlowNetC, PWCNet and Spynet change the images [0,255] to range [0,1]. However, delta is already in [0,1], hence the scaling by 1/255 has to be reset:
            delta_repadded = delta_repadded * 255.

    return delta_repadded






def eval_l2_universal(args):

    experiment_id, folder_path, folder_name = logging.mlflow_experimental_setup(args.output_folder, args.net, "PCFA", args.joint_perturbation, args.universal_perturbation, stage="eval")

    print("Evaluating a Perturbation Constrained Flow Attack:")
    print()
    print("\tModel (evaluation, now): %s" % (args.net))
    print("\tModel (training):        %s" % (args.origin_net))
    print("\tPerturbation universal:  %s" % (str(args.universal_perturbation)))
    print("\tPerturbation joint:      %s" % (str(args.joint_perturbation)))
    print()
    print("\tOutputfolder:            %s" % (folder_path))
    print()

    with mlflow.start_run(experiment_id=experiment_id, run_name=folder_name):
        log_param("perturbation_sourcefolder", args.perturbation_sourcefolder)
        log_param("stage", "eval")
        log_param("outputfolder", folder_path)
        if args.origin_net is None:
            raise ValueError("args.origin_net is not allowed to be empty. Please state which network was used to train the perturbations via the --origin_net argument.")
        log_param("origin_net", args.origin_net)
        distortion_folder_name = "patches"
        distortion_folder_path = folder_path
        distortion_folder = logging.create_subfolder(distortion_folder_path, distortion_folder_name)

        eps_box = 1e-7

        print("Evaluating perturbations trained for %s on %s.\n" % (args.origin_net, args.net))
        epochs, delta1_paths, delta2_paths = extract_epoch_patchlist(args.perturbation_sourcefolder)

        logging.log_model_params(args.net, ownutilities.model_takes_unit_input(args.net))
        logging.log_dataset_params(args.dataset, args.batch_size, epochs, False, args.dstype, args.dataset_stage)
        log_param("attack_joint_perturbation", args.joint_perturbation)
        log_param("attack_universal_perturbation", args.universal_perturbation)

        print("Preparing data from %s %s\n" % (args.dataset, args.dataset_stage))
        data_loader, has_gt = ownutilities.prepare_dataloader(args.dataset_stage, 
                                                      dataset=args.dataset,
                                                      batch_size=args.batch_size,
                                                      shuffle=False, 
                                                      small_run=args.small_run, 
                                                      sintel_subsplit=False, 
                                                      dstype=args.dstype)

        image1_init, image2_init, flow_init, _ = next(iter(data_loader))
        image1_init = image1_init.detach()
        image2_init = image2_init.detach()
        flow_init = flow_init.detach()
        image1_init.requires_grad = False
        image2_init.requires_grad = False
        flow_init.requires_grad = False


        # Define what device we are using

        if Conf.config('useCPU') or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        print("Setting Device to %s\n" % device)

        # Initialize the network
        # load model that uses RAFT, which takes images scaled to [0,1] as input
        # Make sure that model is configured for the change of variables, if PCFA is supposed to run with it.
        print("Loading model %s\n" % (args.net))
        if args.boxconstraint in ['change_of_variables']:
            model = ownutilities.import_and_load(args.net, make_unit_input=not ownutilities.model_takes_unit_input(args.net), variable_change=True, make_scaled_input_model=True, device=device, eps_box=eps_box)
        else:
            model = ownutilities.import_and_load(args.net, make_unit_input=not ownutilities.model_takes_unit_input(args.net), variable_change=False, make_scaled_input_model=True, device=device)

        # Set the model in evaluation mode. This can be needed for Dropout layers, and is also required for the BatchNorm2dLayers in RAFT (that would otherwise still change in training)
        model.eval()
        # Make sure the model is not trained:
        for param in model.parameters():
            param.requires_grad = False

        total_images = 0


        print("Evaluating perturbations on %s %s\n" % (args.dataset, args.dataset_stage))
        for epoch in range(epochs):

            print("Evaluation for perturbation from epoch %d" % epoch)

            delta1 = torch.from_numpy(np.load(delta1_paths[epoch]))
            delta1 = convert_perturbationsizes(delta1, image1_init, args.origin_net, args.net, args.dataset)
            if args.universal_perturbation:
                delta2 = delta1
            else:
                delta2 = torch.from_numpy(np.load(delta2_paths[epoch]))
                delta2 = convert_perturbationsizes(delta2, image2_init, args.origin_net, args.net, args.dataset)
            delta1 = delta1.to(device)
            delta2 = delta2.to(device)

            delta1 = delta1.detach()
            delta2 = delta2.detach()
            delta1.requires_grad = False
            delta2.requires_grad = False

            images_passed = 0
            sum_aee_adv_pred = 0.

            for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):

                delta1 = delta1.detach()
                delta2 = delta2.detach()
                delta1.requires_grad = False
                delta2.requires_grad = False
                image1 = image1.detach()
                image2 = image2.detach()
                image1.requires_grad = False
                image2.requires_grad = False

                image1, image2 = image1.to(device), image2.to(device)


                if not ownutilities.model_takes_unit_input(args.net):
                    image1 = image1/255.
                    image2 = image2/255.

                # RAFT input padding
                padder, [image1, image2] = ownutilities.preprocess_img(args.net, image1, image2)

                # Set requires_grad attribute of tensor. Important for Attack
                image1 = image1.detach()
                image2 = image2.detach()
                image1.requires_grad = False
                image2.requires_grad = False

                flow_pred_init = None

                # Predict flow for undisturbed images (for statistics)
                flow_pred_init = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True)
                [flow_pred_init] = ownutilities.postprocess_flow(args.net, padder, flow_pred_init)
                flow_pred_init = flow_pred_init.to(device).detach()
                flow_pred_init.requires_grad = False

                if args.joint_perturbation:
                    flow_pred = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True, delta1=delta1) # this expands delta to batched input size
                else:
                    flow_pred = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True, delta1=delta1, delta2=delta2)
                [flow_pred] = ownutilities.postprocess_flow(args.net, padder, flow_pred)
                flow_pred = flow_pred.to(device).detach()
                flow_pred.requires_grad = False

                # take care of batched input!

                images_per_batch = image1.size()[0]

                for i in range(images_per_batch):

                    curr_step = total_images + images_passed + i

                    log_metric(key="steps", value=images_passed + i, step=curr_step)
                    log_metric(key="batch", value=batch, step=curr_step)
                    log_metric(key="epoch", value=epoch, step=curr_step)

                    flow_pred_i = flow_pred[i:i+1,:,:,:]
                    flow_pred_init_i = flow_pred_init[i:i+1,:,:,:]
                    image1_i = image1[i:i+1,:,:,:]
                    image2_i = image2[i:i+1,:,:,:]

                    aee_adv_pred = ownutilities.torchfloat_to_float64(losses.avg_epe(flow_pred_i, flow_pred_init_i))
                    sum_aee_adv_pred += aee_adv_pred

                    logging.log_metrics(curr_step, ("aee_pred-predadv", aee_adv_pred))

                    if (((images_passed+i) % args.save_frequency == 0 and not args.small_save) or (args.small_save and (images_passed+i) < 32)) and not args.no_save:

                        logging.save_tensor(delta1, "delta1", curr_step, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                        logging.save_tensor(delta2, "delta2", curr_step, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                        logging.save_tensor(image1_i, "image1", curr_step, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                        logging.save_tensor(image2_i, "image2", curr_step, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                        logging.save_tensor(flow_pred_i, "flow_pred", curr_step, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                        logging.save_tensor(flow_pred_init_i, "flow_pred_init", curr_step, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)


                        logging.save_image(image1_i, curr_step, distortion_folder, image_name='image1', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
                        logging.save_image(image2_i, curr_step, distortion_folder, image_name='image2', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
                        logging.save_image(image1_i+delta1, curr_step, distortion_folder, image_name='image1_delta', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
                        logging.save_image(image2_i+delta2, curr_step, distortion_folder, image_name='image2_delta', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)

                        max_flow = np.max([ownutilities.maximum_flow(flow_pred_init_i), 
                                           ownutilities.maximum_flow(flow_pred_i)])

                        logging.save_flow(flow_pred_i, curr_step, distortion_folder, flow_name='flow_pred', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
                        logging.save_flow(flow_pred_init_i, curr_step, distortion_folder, flow_name='flow_pred_init', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)

                images_passed += images_per_batch

            avg_aee_adv_pred = sum_aee_adv_pred / images_passed

            total_images += images_passed

            logging.log_metrics(total_images-1, ("epoch_aee_pred-predadv", avg_aee_adv_pred))
            l2_delta1, l2_delta2, l2_delta12 = logging.calc_delta_metrics(delta1, delta2, total_images-1)
            logging.log_metrics(total_images-1, ("l2_delta1", l2_delta1),
                                                ("l2_delta2", l2_delta2), 
                                                ("l2_delta-avg", l2_delta12))

            max_delta = np.max([ownutilities.torchfloat_to_float64(torch.max(torch.abs(delta1))), 
                                ownutilities.torchfloat_to_float64(torch.max(torch.abs(delta2)))])

            logging.save_image(delta1, total_images-1, distortion_folder, image_name='delta1_e'+str(epoch), unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)
            if not args.joint_perturbation:
                logging.save_image(delta2, total_images-1, distortion_folder, image_name='delta2_e'+str(epoch), unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)

            logging.save_image(delta1, total_images-1, distortion_folder, image_name='delta1_e'+str(epoch), unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)
            if not args.joint_perturbation:
                logging.save_image(delta2, total_images-1, distortion_folder, image_name='delta2_e'+str(epoch), unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)

            print("Finished attacking epoch %d" % epoch)
            print("\tAEE(f_adv, f_init)=%f" % avg_aee_adv_pred)
            print("\tL2(perturbation)  =%f\n" % l2_delta12)



if __name__ == '__main__':

    parser = parsing_file.create_parser(stage='evaluation', attack_type='pcfa')
    args = parser.parse_args()
    print(args)

    if args.universal_perturbation:
      eval_l2_universal(args)
    else:
      raise ValueError("An additional evaluation for non-universal perturbations is not implemented.")