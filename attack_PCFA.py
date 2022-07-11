from __future__ import print_function
import argparse
import mlflow
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm
from torchvision import datasets, transforms
from mlflow import log_metric, log_param, log_artifacts

from helper_functions import ownutilities, datasets, losses, parsing_file, targets, logging
from helper_functions.config_paths import Conf, Paths


def extract_deltas(nw_input1, nw_input2, image1, image2, boxconstraint, eps_box=0.):

    if boxconstraint in ['change_of_variables']:
        delta1 = (1./2.)*1./(1.-eps_box)*(torch.tanh(nw_input1) + (1.-eps_box)) - image1
        delta2 = (1./2.)*1./(1.-eps_box)*(torch.tanh(nw_input2) + (1.-eps_box)) - image2
    else: # case clipping (ScaledInputModel also treats everything that is not change_of_variables by clipping it into range before feeding it to network.)
        delta1 = torch.clamp(nw_input1, 0., 1.) - image1
        delta2 = torch.clamp(nw_input2, 0., 1.) - image2

    return delta1, delta2


def extract_deltas_joint(nw_delta, images_max, images_min):

    delta_upper = torch.clamp(nw_delta+images_max, 0., 1.) - images_max
    delta       = torch.clamp(delta_upper+images_min, 0., 1.) - images_min

    return delta, delta


def pcfa_attack(model, image1, image2, flow, batch, distortion_folder, eps_box, device, has_gt, optim_mu, args, statistics_in_every_step=True):
    torch.autograd.set_detect_anomaly(True)

    curr_step = batch*args.steps

    # prepare and rescale images

    aee_gt = 0.
    aee_tgt = 0.
    aee_gt_tgt = 0.
    aee_adv_gt = 0.
    aee_adv_tgt = 0.
    aee_adv_pred = 0.

    image1, image2 = image1.to(device), image2.to(device)
    flow = flow.to(device)
    # If the model takes unit input, ownutilities.preprocess_img will transform images into [0,1].
    # Otherwise, do transformation here
    if not ownutilities.model_takes_unit_input(args.net):
        image1 = image1/255.
        image2 = image2/255.

    # RAFT input padding
    padder, [image1, image2] = ownutilities.preprocess_img(args.net, image1, image2)

    # Set requires_grad attribute of tensor. Important for Attack
    image1.requires_grad = False
    image2.requires_grad = False
    images_max = torch.max(image1, image2).detach().to(device)
    images_min = torch.min(image1, image2).detach().to(device)

    # initialize perturbation and auxiliary variables:
    delta1 = torch.zeros_like(image1)
    delta2 = torch.zeros_like(image2)
    delta1 = delta1.to(device)
    delta2 = delta2.to(device)

    nw_input1 = None
    nw_input2 = None
    nw_delta = None

    flow_pred_init = None

    # Set up the optimizer and variables if a common perturbation delta should be trained
    if args.joint_perturbation:
        delta1.requires_grad = False
        delta2.requires_grad = False

        nw_delta = delta1
        nw_delta.requires_grad = True

        if args.boxconstraint in ['change_of_variables']:
            raise ValueError("Training a --joint_perturbation with --boxconstraint=change_of_variables is not defined. Please use --boxconstraint=clipping.")
        else:
            nw_input1 = image1
            nw_input2 = image2

        optimizer = optim.LBFGS([nw_delta], max_iter=10)

    # Set up the optimizer and variables if individual perturbations delta1 and delta2 for images 1 and 2 should be trained
    else:
        delta1.requires_grad = False
        delta2.requires_grad = False

        if args.boxconstraint in ['change_of_variables']:
            nw_input1 = torch.atanh( 2. * (1.- eps_box) * (image1 + delta1) - (1 - eps_box)  )
            nw_input2 = torch.atanh( 2. * (1.- eps_box) * (image2 + delta2) - (1 - eps_box)  )
        else:
            nw_input1 = image1 + delta1
            nw_input2 = image2 + delta2

        nw_input1.requires_grad = True
        nw_input2.requires_grad = True

        optimizer = optim.LBFGS([nw_input1, nw_input2], max_iter=10)


    # Predict the flow
    flow_pred = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True)
    [flow_pred] = ownutilities.postprocess_flow(args.net, padder, flow_pred)
    flow_pred = flow_pred.to(device)

    # define the initial flow, the target, and update mu
    flow_pred_init = flow_pred.detach().clone()
    flow_pred_init.requires_grad = False


    # define target (potentially based on first flow prediction)
    # define attack target
    target = targets.get_target(args.target, flow_pred_init, custom_target_path=args.custom_target_path, device=device)
    target = target.to(device)
    target.requires_grad = False

    # Some aee statistics for the unattacked flow
    aee_tgt            = logging.calc_metrics_const(target, flow_pred_init)
    aee_gt_tgt, aee_gt = logging.calc_metrics_const_gt(target, flow_pred_init, flow) if has_gt else (None, None)

    logging.log_metrics(curr_step, ("aee_pred-tgt", aee_tgt), 
                                   ("aee_gt-tgt", aee_gt_tgt), 
                                   ("aee_pred-gt", aee_gt))

    log_metric(key="optim_mu", value=optim_mu, step=curr_step)

    # Zero all existing gradients
    model.zero_grad()
    optimizer.zero_grad()

    delta_below_threshold=False
    delta12_min_val = float('inf')
    aee_adv_tgt_min_val = float('inf')
    aee_adv_pred_min_val = 0.
    delta1_min = None
    delta2_min = None
    flow_pred_min = None

    for steps in range(args.steps):

        curr_step = batch*args.steps + steps
        log_metric(key="batch", value=batch, step=curr_step)
        log_metric(key="steps", value=steps, step=curr_step)
        log_metric(key="epoch", value=0, step=curr_step)

        # Calculate the deltas from the quantities that go into the network
        if args.joint_perturbation:
            delta1, delta2 = extract_deltas_joint(nw_delta, images_max, images_min)
        else:
            delta1, delta2 = extract_deltas(nw_input1, nw_input2, image1, image2, args.boxconstraint, eps_box=eps_box)

        # Calculate the loss
        loss = losses.loss_delta_constraint(flow_pred, target, delta1, delta2, device, delta_bound=args.delta_bound, mu=optim_mu,  f_type=args.loss)


        # Update the optimization parameters
        loss.backward()

        def closure():
            optimizer.zero_grad()
            if args.joint_perturbation:
                flow_closure = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True, delta1=nw_delta)
            else:
                flow_closure = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True)
            [flow_closure] = ownutilities.postprocess_flow(args.net, padder, flow_closure)
            flow_closure = flow_closure.to(device)
            if args.joint_perturbation:
                delta1_closure, delta2_closure = extract_deltas_joint(nw_delta, images_max, images_min)
            else:
                delta1_closure, delta2_closure = extract_deltas(nw_input1, nw_input2, image1, image2, args.boxconstraint, eps_box=eps_box)
            loss_closure = losses.loss_delta_constraint(flow_closure, target, delta1_closure, delta2_closure, device, delta_bound=args.delta_bound, mu=optim_mu,  f_type=args.loss)
            loss_closure.backward()
            return loss_closure

        # Update the optimization parameters
        optimizer.step(closure)

        # calculate the magnitude of the updated distortion, and with it the new network inputs:
        if args.joint_perturbation:
            delta1, delta2 = extract_deltas_joint(nw_delta, images_max, images_min)
            if args.boxconstraint in ['change_of_variables']:
                raise ValueError("Training a --joint_perturbation with --boxconstraint=change_of_variables is not defined. Please use --boxconstraint=clipping.")
            else:
                nw_input1 = image1
                nw_input2 = image2
        else:
            delta1, delta2 = extract_deltas(nw_input1, nw_input2, image1, image2, args.boxconstraint, eps_box=eps_box)
            # The nw_inputs remain unchanged in this case, and can be directly fed into the network again for further perturbation training

        # Re-predict flow with the perturbed image, and update the flow prediction for the next iteration
        if args.joint_perturbation:
            flow_pred = ownutilities.compute_flow(model, "scaled_input_model", nw_input1, nw_input2, test_mode=True, delta1=nw_delta)
        else:
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

        update_minima = False
        if not delta_below_threshold:
            if l2_delta12 < delta12_min_val or (l2_delta12 == delta12_min_val and aee_adv_tgt < aee_adv_tgt_min_val):

                update_minima = True
                if l2_delta12 <= args.delta_bound:
                    delta_below_threshold = True
        else:
            if l2_delta12 <= args.delta_bound and aee_adv_tgt < aee_adv_tgt_min_val:
                update_minima = True

        if update_minima:
            delta12_min_val = l2_delta12
            aee_adv_tgt_min_val = aee_adv_tgt
            aee_adv_pred_min_val = aee_adv_pred
            delta1_min = delta1.detach().clone()
            delta2_min = delta2.detach().clone()
            flow_pred_min = flow_pred.detach().clone()

        logging.log_metrics(curr_step, ("aee_pred-tgt_min", aee_adv_tgt_min_val), 
                                   ("l2_delta-avg_min", delta12_min_val), 
                                   ("aee_pred-predadv_min", aee_adv_pred_min_val))


    # Final saving of images:
    if ((batch % args.save_frequency == 0 and not args.small_save) or (args.small_save and batch < 32)) and not args.no_save:

        logging.save_tensor(delta1, "delta1_final", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(delta2, "delta2_final", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(delta1_min, "delta1_best", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(delta2_min, "delta2_best", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(image1, "image1", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(image2, "image2", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(target, "target", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(flow_pred, "flow_pred_final", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(flow_pred_min, "flow_pred_best", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_tensor(flow_pred_init, "flow_pred_init", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
        if has_gt: 
            logging.save_tensor(flow, "flow_gt", batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)


        logging.save_image(image1, batch, distortion_folder, image_name='image1', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_image(image2, batch, distortion_folder, image_name='image2', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_image(image1+delta1_min, batch, distortion_folder, image_name='image1_delta_best', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_image(image2+delta2_min, batch, distortion_folder, image_name='image2_delta_best', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)


        max_delta = np.max([ownutilities.torchfloat_to_float64(torch.max(torch.abs(delta1_min))), 
                            ownutilities.torchfloat_to_float64(torch.max(torch.abs(delta2_min)))])

        logging.save_image(delta1_min, batch, distortion_folder, image_name='delta1_best', unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)
        if not args.joint_perturbation:
            logging.save_image(delta2_min, batch, distortion_folder, image_name='delta2_best', unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)


        max_flow_gt = 0
        if has_gt:
            max_flow_gt = ownutilities.maximum_flow(flow)
        max_flow = np.max([max_flow_gt, 
                           ownutilities.maximum_flow(flow_pred_init), 
                           ownutilities.maximum_flow(flow_pred_min)])

        logging.save_flow(flow_pred_min, batch, distortion_folder, flow_name='flow_pred_best', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_flow(flow_pred_init, batch, distortion_folder, flow_name='flow_pred_init', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
        logging.save_flow(target, batch, distortion_folder, flow_name='flow_target', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
        if has_gt:
            logging.save_flow(flow, batch, distortion_folder, flow_name='flow_gt', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)

    return aee_gt, aee_tgt, aee_gt_tgt, aee_adv_gt, aee_adv_tgt, aee_adv_pred, l2_delta1, l2_delta2, l2_delta12, aee_adv_tgt_min_val, aee_adv_pred_min_val, delta12_min_val


def attack_l2_universal(args):
    torch.autograd.set_detect_anomaly(True)

    experiment_id, folder_path, folder_name = logging.mlflow_experimental_setup(args.output_folder, args.net, "PCFA", args.joint_perturbation, args.universal_perturbation)

    optim_mu = args.mu
        # update mu in case it was not provided
    if optim_mu == -1.:
        optim_mu = 2500./args.delta_bound
        if args.target not in ['zero']:
            optim_mu = 1.5*optim_mu
        print("The optimizer penalty factor mu was choosen automatically to %d, because no value was provided via --mu. Note that this choice may not be optimal, and that you can tune it by hand for better results.\n" % optim_mu)

    print("\nStarting Perturbation Constrained Flow Attack (PCFA):")
    print()
    print("\tModel:                   %s" % (args.net))
    print("\tPerturbation universal:  %s" % (str(args.universal_perturbation)))
    print("\tPerturbation joint:      %s" % (str(args.joint_perturbation)))
    print("\tPerturbation bound:      %f" % (args.delta_bound))
    print()
    print("\tTarget:                  %s" % (args.target))
    print("\tOptimizer steps:         %d" % (args.steps))
    print("\tOptimizer boxconstraint: clipping")
    print("\tOptimizer mu:            %f" % (optim_mu))
    print()
    print("\tOutputfolder:            %s" % (folder_path))
    print()

    with mlflow.start_run(experiment_id=experiment_id, run_name=folder_name):
        log_param("outputfolder", folder_path)
        distortion_folder = logging.create_subfolder(folder_path, "patches")
        log_artifacts(distortion_folder)

        eps_box = 1e-7
        model_takes_unit_input = ownutilities.model_takes_unit_input(args.net)

        logging.log_model_params(args.net, model_takes_unit_input)
        logging.log_dataset_params(args.dataset, args.batch_size, args.epochs, False, args.dstype, args.dataset_stage)
        logging.log_attack_params("PCFA", args.loss, args.target, args.joint_perturbation, args.universal_perturbation, custom_target_path=args.custom_target_path)
        log_param("box_eps", eps_box)
        log_param("pcfa_delta_bound", args.delta_bound)
        log_param("optimizer", "LBFGS")
        log_param("optimizer_mu", args.mu)
        log_param("optimizer_boxconstraint", 'clipping')
        log_param("optimizer_steps", args.steps)

        print("Preparing data from %s %s\n" % (args.dataset, args.dataset_stage))
        data_loader, has_gt = ownutilities.prepare_dataloader(args.dataset_stage, 
                                                      dataset=args.dataset,
                                                      batch_size=args.batch_size,
                                                      shuffle=True, 
                                                      small_run=args.small_run, 
                                                      sintel_subsplit=False, 
                                                      dstype=args.dstype)

        image1_init, image2_init, flow_init, _ = next(iter(data_loader))
        padder_init, [image1_init, image2_init] = ownutilities.preprocess_img(args.net, image1_init, image2_init)

        # Define what device we are using
        if Conf.config('useCPU') or not torch.cuda.is_available():
             device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        print("Setting Device to %s\n" % device)

        # Initialize the network
        # The black box attack assumes clipping and is not implemented for change_of_variables
        print("Loading model %s:" % (args.net))
        model = ownutilities.import_and_load(args.net, make_unit_input=not model_takes_unit_input, variable_change=False, make_scaled_input_model=True, device=device)

        # Set the model in evaluation mode. In this case of the tutorial this was for the Dropout layers - not sure if it is required for RAFT
        model.eval()
        # Make sure the model is not trained:
        for param in model.parameters():
            param.requires_grad = False

        nw_delta1 = torch.zeros_like(image1_init[0,:,:,:])
        nw_delta2 = torch.zeros_like(image2_init[0,:,:,:])
        nw_delta1 = nw_delta1.to(device)
        nw_delta2 = nw_delta2.to(device)

        if args.joint_perturbation:
            nw_delta1.requires_grad = True
            nw_delta2.requires_grad = False

            optimizer = optim.LBFGS([nw_delta1], max_iter=10)

        else:
            nw_delta1.requires_grad = True
            nw_delta2.requires_grad = True

            optimizer = optim.LBFGS([nw_delta1, nw_delta2], max_iter=10)


        batch_ctr = -1
        # Loop over all examples in test set
        print("Starting Attack on %s %s\n" % (args.dataset, args.dataset_stage))
        for epoch in range(args.epochs):
            print("epoch: %d" % epoch)
            for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):

                if has_gt:
                    flow = flow.to(device)

                batch_ctr += 1
                curr_step = batch_ctr*args.steps

                torch.autograd.set_detect_anomaly(True)

                image1, image2 = image1.to(device), image2.to(device)
                if not model_takes_unit_input:
                    image1 = image1/255.
                    image2 = image2/255.

                # define attack target
                target = None

                # Input padding
                padder, [image1, image2] = ownutilities.preprocess_img(args.net, image1, image2)

                # Set requires_grad attribute of tensor. Important for Attack
                image1.requires_grad = False
                image2.requires_grad = False
                images_max = torch.max(image1, image2).detach()
                images_min = torch.min(image1, image2).detach()

                flow_pred_init = None


                # Predict flow for undisturbed images (for target creation and statistics)
                flow_pred_init = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True)
                [flow_pred_init] = ownutilities.postprocess_flow(args.net, padder, flow_pred_init)
                flow_pred_init = flow_pred_init.to(device).detach()
                flow_pred_init.requires_grad = False

                if args.joint_perturbation:
                    flow_pred = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True, delta1=nw_delta1)
                else:
                    flow_pred = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True, delta1=nw_delta1, delta2=nw_delta2)
                [flow_pred] = ownutilities.postprocess_flow(args.net, padder, flow_pred)
                flow_pred = flow_pred.to(device)

                # Set up target
                target = targets.get_target(args.target, flow_pred_init, custom_target_path=args.custom_target_path, device=device)
                target = target.to(device)
                target.requires_grad = False

                aee_tgt            = logging.calc_metrics_const(target, flow_pred_init)
                aee_gt_tgt, aee_gt = logging.calc_metrics_const_gt(target, flow_pred_init, flow) if has_gt else (None, None)

                logging.log_metrics(curr_step, ("aee_pred-tgt", aee_tgt), 
                                               ("aee_gt-tgt", aee_gt_tgt), 
                                               ("aee_pred-gt", aee_gt))

                # Zero all existing gradients
                model.zero_grad()
                optimizer.zero_grad()

                for steps in range(args.steps):

                    curr_step = batch_ctr*args.steps + steps
                    log_metric(key="steps", value=steps, step=curr_step)
                    log_metric(key="batch", value=batch, step=curr_step)
                    log_metric(key="epoch", value=epoch, step=curr_step)

                    # extract deltas
                    if args.joint_perturbation:
                        delta1, delta2 = nw_delta1, nw_delta1
                    else:
                        delta1, delta2 = nw_delta1, nw_delta2

                    # Calculate the loss
                    loss = losses.loss_delta_constraint(flow_pred, target, delta1, delta2, device, delta_bound=args.delta_bound, mu=optim_mu,  f_type=args.loss)


                    # Update the optimization parameters
                    loss.backward()

                    def closure():
                        optimizer.zero_grad()
                        if args.joint_perturbation:
                            flow_closure = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True, delta1=nw_delta1)
                            delta1_closure, delta2_closure = nw_delta1, nw_delta1
                        else:
                            flow_closure = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True, delta1=nw_delta1, delta2=nw_delta2)
                            delta1_closure, delta2_closure = nw_delta1, nw_delta2
                        [flow_closure] = ownutilities.postprocess_flow(args.net, padder, flow_closure)
                        flow_closure = flow_closure.to(device)
                        loss_closure = losses.loss_delta_constraint(flow_closure, target, delta1_closure, delta2_closure, device, delta_bound=args.delta_bound, mu=optim_mu,  f_type=args.loss)
                        loss_closure.backward()
                        return loss_closure

                    # Update the optimization parameters
                    optimizer.step(closure)


                    # extract deltas
                    if args.joint_perturbation:
                        delta1, delta2 = nw_delta1, nw_delta1
                    else:
                        delta1, delta2 = nw_delta1, nw_delta2


                    if args.joint_perturbation:
                        flow_pred = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True, delta1=nw_delta1)
                    else:
                        flow_pred = ownutilities.compute_flow(model, "scaled_input_model", image1, image2, test_mode=True, delta1=nw_delta1, delta2=nw_delta2)
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

                if ((batch_ctr % args.save_frequency == 0 and not args.small_save) or (args.small_save and batch_ctr < 32)) and not args.no_save:

                    logging.save_tensor(delta1, "delta1_b" + str(batch_ctr), batch_ctr, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                    logging.save_tensor(delta2, "delta2_b" + str(batch_ctr), batch_ctr, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)

            logging.save_tensor(delta1, "delta1_e" + str(epoch), batch_ctr, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)

            max_delta = np.max([ownutilities.torchfloat_to_float64(torch.max(torch.abs(delta1))), 
                                ownutilities.torchfloat_to_float64(torch.max(torch.abs(delta2)))])
            logging.save_image(delta1, batch_ctr, distortion_folder, image_name='delta1_e'+str(epoch), unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)
            if not args.joint_perturbation:
                logging.save_image(delta2, batch_ctr, distortion_folder, image_name='delta2_e'+str(epoch), unit_input=True, normalize_max=max_delta, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(delta2, "delta2_e" + str(epoch), batch_ctr, distortion_folder,unregistered_artifacts=args.unregistered_artifacts)
            logging.save_image(image1+delta1.repeat([image2.size()[0],1,1,1]), batch_ctr, distortion_folder, image_name='image1_delta_e'+str(epoch), unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
            logging.save_image(image2+delta2.repeat([image2.size()[0],1,1,1]), batch_ctr, distortion_folder, image_name='image2_delta_e'+str(epoch), unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)

            max_flow_gt = 0
            if has_gt:
                max_flow_gt = ownutilities.maximum_flow(flow)
            max_flow = np.max([max_flow_gt, 
                               ownutilities.maximum_flow(flow_pred_init), 
                               ownutilities.maximum_flow(flow_pred)])
            logging.save_flow(flow_pred, batch_ctr, distortion_folder, flow_name='flow_pred_e'+str(epoch), auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)

            if epoch == 0:
                logging.save_tensor(image1, "image1_e" + str(epoch), batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(image2, "image2_e" + str(epoch), batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(target, "target_e" + str(epoch), batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(flow_pred, "flow_pred_e" + str(epoch), batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_tensor(flow_pred_init, "flow_pred_init_e" + str(epoch), batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)
                if has_gt: 
                    logging.save_tensor(flow, "flow_gt_e" + str(epoch), batch, distortion_folder, unregistered_artifacts=args.unregistered_artifacts)


                logging.save_image(image1, batch, distortion_folder, image_name='image1', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_image(image2, batch, distortion_folder, image_name='image2', unit_input=True, normalize_max=None, unregistered_artifacts=args.unregistered_artifacts)


                logging.save_flow(target, batch, distortion_folder, flow_name='flow_target', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
                logging.save_flow(flow_pred_init, batch, distortion_folder, flow_name='flow_pred_init', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)
                if has_gt:
                    logging.save_flow(flow, batch, distortion_folder, flow_name='flow_gt', auto_scale=False, max_scale=max_flow, unregistered_artifacts=args.unregistered_artifacts)

        print("\nFinished attacking with PCFA, universal perturbations have been produced and are logged at\n%s" % folder_path )
        print("To evaluate the trained perturbations after each epoch, please pass this folder to evaluate_PCFA.py via the --perturbation_folder flag, state the used model as --origin_net and specify the dataset on which to evaluate (e.g. the dataset used to train the perturbations):")
        print()
        print("   $ python3 evaluate_PCFA.py --net=%s --origin_net=%s --dataset=%s --dataset_stage=%s --perturbation_sourcefolder=%s --dstype=%s --universal_perturbation --boxconstraint=clipping %s" % (args.net, args.net, args.dataset, args.dataset_stage, folder_path, args.dstype, "--joint_perturbation" if args.joint_perturbation else ""))
        print()



def attack_l2(args):

    """
    Performs an PCFA attack on a given model and for all images of a specified dataset.
    """

    experiment_id, folder_path, folder_name = logging.mlflow_experimental_setup(args.output_folder, args.net, "PCFA", args.joint_perturbation, args.universal_perturbation)

    optim_mu = args.mu
    # update mu in case it was not provided
    if optim_mu == -1.:
        optim_mu = 2500./args.delta_bound
        if args.target not in ['zero']:
            optim_mu = 1.5*optim_mu
        print("The optimizer penalty factor mu was choosen automatically to %d, because no value was provided via --mu. Note that this choice may not be optimal, and that you can tune it by hand for better results.\n" % optim_mu)

    print("\nStarting Perturbation Constrained Flow Attack (PCFA):")
    print()
    print("\tModel:                   %s" % (args.net))
    print("\tPerturbation universal:  %s" % (str(args.universal_perturbation)))
    print("\tPerturbation joint:      %s" % (str(args.joint_perturbation)))
    print("\tPerturbation bound:      %f" % (args.delta_bound))
    print()
    print("\tTarget:                  %s" % (args.target))
    print("\tOptimizer steps:         %d" % (args.steps))
    print("\tOptimizer boxconstraint: %s" % (args.boxconstraint))
    print("\tOptimizer mu:            %f" % (optim_mu))
    print()
    print("\tOutputfolder:            %s" % (folder_path))
    print()

    with mlflow.start_run(experiment_id=experiment_id, run_name=folder_name):

        log_param("outputfolder", folder_path)
        distortion_folder_name = "patches"
        distortion_folder_path = folder_path
        distortion_folder = logging.create_subfolder(distortion_folder_path, distortion_folder_name)

        eps_box = 1e-7
        model_takes_unit_input = ownutilities.model_takes_unit_input(args.net)

        logging.log_model_params(args.net, model_takes_unit_input)
        logging.log_dataset_params(args.dataset, 1, 1, False, args.dstype, args.dataset_stage)
        logging.log_attack_params("PCFA", args.loss, args.target, args.joint_perturbation, args.universal_perturbation, custom_target_path=args.custom_target_path)
        log_param("box_eps", eps_box)
        log_param("pcfa_delta_bound", args.delta_bound)
        log_param("optimizer", "LBFGS")
        log_param("optimizer_mu", args.mu)
        log_param("optimizer_boxconstraint", args.boxconstraint)
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
        # load model that uses RAFT, which takes images scaled to [0,1] as input
        # Make sure that model is configured for the change of variables, if PCFA is supposed to run with it.
        print("Loading model %s:" % (args.net))
        if args.boxconstraint in ['change_of_variables']:
            model = ownutilities.import_and_load(args.net, make_unit_input=not model_takes_unit_input, variable_change=True, make_scaled_input_model=True, device=device, eps_box=eps_box)
        else:
            model = ownutilities.import_and_load(args.net, make_unit_input=not model_takes_unit_input, variable_change=False, make_scaled_input_model=True, device=device)

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
        sum_aee_adv_tgt_min = 0.
        sum_aee_adv_pred_min = 0.
        sum_l2_delta12_min = 0.
        tests = 0


        # Loop over all examples in test set
        print("Starting Attack on %s %s\n" % (args.dataset, args.dataset_stage))
        for batch, (image1, image2, flow, _) in enumerate(tqdm(data_loader)):

            aee_gt, aee_tgt, aee_gt_tgt, aee_adv_gt, aee_adv_tgt, aee_adv_pred, l2_delta1, l2_delta2, l2_delta12, aee_adv_tgt_min_val, aee_adv_pred_min_val, delta12_min_val = pcfa_attack(model, image1, image2, flow, batch, distortion_folder, eps_box, device, has_gt, optim_mu, args)
            sum_aee_tgt += aee_tgt
            sum_aee_adv_tgt += aee_adv_tgt
            sum_aee_adv_pred += aee_adv_pred
            sum_l2_delta12 += l2_delta12
            sum_aee_adv_tgt_min += aee_adv_tgt_min_val
            sum_aee_adv_pred_min += aee_adv_pred_min_val
            sum_l2_delta12_min += delta12_min_val
            if has_gt:
                sum_aee_gt += aee_gt
                sum_aee_gt_tgt += aee_gt_tgt
                sum_aee_adv_gt += aee_adv_gt
            tests += 1

        # Calculate final accuracy
        logging.calc_log_averages(tests,
                ("aee_avg_pred-gt",sum_aee_gt), 
                ("aee_avg_pred-tgt", sum_aee_tgt),
                ("aee_avg_gt-tgt",sum_aee_gt_tgt),
                ("aee_avg_predadv-gt", sum_aee_adv_gt),
                ("aee_avg_predadv-tgt", sum_aee_adv_tgt),
                ("aee_avg_pred-predadv", sum_aee_adv_pred),
                ("l2_avg_delta12", sum_l2_delta12),
                ("aee_avg_predadv-tgt_min", sum_aee_adv_tgt_min),
                ("aee_avg_pred-predadv_min", sum_aee_adv_pred_min),
                ("l2_avg_delta12_min", sum_l2_delta12_min))

        print("\nFinished attacking with PCFA. The best achieved values are")
        print("\tAEE(f_adv, f_init)=%f" % (sum_aee_adv_pred_min / tests))
        print("\tAEE(f_adv, f_targ)=%f" % (sum_aee_adv_tgt_min / tests))
        print("\tL2(perturbation)  =%f" % (sum_l2_delta12_min / tests))
        print()



if __name__ == '__main__':

    parser = parsing_file.create_parser(stage='training', attack_type='pcfa')
    args = parser.parse_args()
    print(args)

    if args.universal_perturbation:
        attack_l2_universal(args)
    else:
        attack_l2(args)