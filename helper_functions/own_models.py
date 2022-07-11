import torch
import torch.nn as nn
import logging

from helper_functions import ownutilities



class ScaledInputModel(nn.Module):
    def __init__(self,  net, make_unit_input=False, variable_change=False, **kwargs):
        """
        Initializes a Model with rescaled input requirements. It calls a pretrained model that takes inputs in [0,255], but the ScaledInputModel version of it can handle inputs in [0,1] and transforms them before passing them to the pretrained model.
        This might be necessary for attacks that work on different paramter spaces, or with substitued variables
        Args:
            net: a string describing the network type
            make_unit_input (bool): if True, the Model assumes that it receives inputs in [0,1], but that the pretrained model requires them in [0,255]
            variable_change (bool): if True, the Model assumes that it will be given a variable as required for the change of variable proposed in the Carlini&Wagner attack.
            **kwargs: Futher arguments for ownutilities.import_and_load()
        """
        super(ScaledInputModel, self).__init__()

        self.make_unit_input = make_unit_input
        self.var_change = variable_change
        self.model_name = net
        logging.info("Creating a Model with scaled input and the following parameters:")
        logging.info("\tmake_unit_input=%s" % (str(make_unit_input)))

        self.eps_box = 0.
        # ToDo: Chekc if var_change is really checked
        if 'eps_box' in kwargs:
            self.eps_box = kwargs.get("eps_box")
            logging.info("\teps_box=%s" % (str(self.eps_box)))

        elif variable_change:
            logging.warning("The ScaledInputModel did receive 'variable_change'=True, but no epsilon value for the CW attack was given. This might lead to numerical instabilities. Consider a small float value to the ScaledInputModel. Setting 'eps_box'=0.0.")

        self.model_loaded = ownutilities.import_and_load(net, **kwargs)


    def forward(self, image1, image2, delta1=None, delta2=None, test_mode=True, *args, **kwargs):
        """
        Performs a forward pass of the pretrained model as specified in self.model_loaded.
        Optionally, it transforms the input image1 and image2 previous to feeding them into the model.
        When specifying delta1 or delta2, one or both of these tensors are added as perturbations to the input images.
        If only "delta1" is specified, this tensor is added to both images and the result is cliped to the allowed image range [0,1].
        If make_unit_input=True was specified for the ScaledInputModel, images1 and 2 are assumed to be in [0,1] when they enter forward, but will be rescaled to [0,255] before being passed to the pretrained model.
        If variable_change=True was specified for the ScaledInputModel, images1 and 2 are assumed to be not the image information, but the w-variable from the Carlini&Wagner model. Hence they are transformed into their image representations, before being fed to to the model.

        Args:
            image1 (tensor): first input image for model in [0,1]
            image2 (tensor): second input image for model in [0,1]
            *args: additional arguments for model
            delta1 (optional, tensor): optional distortion to be added to image1 in [0,1]
            delta2 (optional, tensor): optional distortion to be added to image2 in [0,1]
            **kwargs: additional kwargs for model
        
        Returns:
            tensor: returns the output of a forward pass of the pytorch.model on image1 and image2, after the specified transformations to images1 and 2.
        """

        # Add delta to the input images if a delta was given
        if delta1 is not None:
            batch_delta1 = delta1.repeat([image1.size()[0],1,1,1])
            image1 = image1 + batch_delta1
            if delta2 is None:
                batch_delta2 = delta1.repeat([image2.size()[0],1,1,1])
                image2 = image2 + batch_delta2
        if delta2 is not None:
            batch_delta2 = delta2.repeat([image2.size()[0],1,1,1])
            image2 = image2 + batch_delta2

        # Perform the Carlini&Wagner Change of Variables, if the ScaledInputModel was configured to do so.
        if self.var_change:
            image1 = (1./2.) * 1. / (1. - self.eps_box) * (torch.tanh(image1) + (1 - self.eps_box) )
            image2 = (1./2.) * 1. / (1. - self.eps_box) * (torch.tanh(image2) + (1 - self.eps_box) )


        # Clipping case, which will only clip something if change of variables was not defined. otherwise, the change of variables has already brought the iamges into the range [0,1]
        image1 = torch.clamp(image1, 0., 1.)
        image2 = torch.clamp(image2, 0., 1.)

        # If model expects images in [0,255], transform them from [0,1]
        if self.make_unit_input:
            image1 = 255.*image1
            image2 = 255.*image2

        # return self.model_loaded(image1, image2, *args, **kwargs)
        return ownutilities.compute_flow(self.model_loaded, self.model_name, image1, image2, test_mode=test_mode, *args, **kwargs)
