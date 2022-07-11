import torch

def avg_epe(flow1, flow2):
    """"
    Compute the average endpoint errors (AEE) between two flow fields.
    The epe measures the euclidean- / 2-norm of the difference of two optical flow vectors
    (u0, v0) and (u1, v1) and is defined as sqrt((u0 - u1)^2 + (v0 - v1)^2).

    Args:
        flow1 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component
        flow2 (tensor):
            represents a flow field with dimension (2,M,N) or (b,2,M,N) where M ~ u-component and N ~v-component

    Raises:
        ValueError: dimensons not valid

    Returns:
        float: scalar average endpoint error 
    """
    diff_squared = (flow1 - flow2)**2
    if len(diff_squared.size()) == 3:
        # here, dim=0 is the 2-dimension (u and v direction of flow [2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.mean(torch.sum(diff_squared, dim=0).sqrt())
    elif len(diff_squared.size()) == 4:
        # here, dim=0 is the 2-dimension (u and v direction of flow [b,2,M,N]) , which needs to be added BEFORE taking the square root. To get the length of a flow vector, we need to do sqrt(u_ij^2 + v_ij^2)
        epe = torch.mean(torch.sum(diff_squared, dim=1).sqrt())
    else:
        raise ValueError("The flow tensors for which the EPE should be computed do not have a valid number of dimensions (either [b,2,M,N] or [2,M,N]). Here: " + str(flow1.size()) + " and " + str(flow1.size()))
    return epe

def avg_mse(flow1, flow2):
    """Computes mean squared error between two flow fields.

    Args:
        flow1 (tensor):
            flow field, which must have the same dimension as flow2
        flow2 (tensor):
            flow field, which must have the same dimension as flow1

    Returns:
        float: scalar average squared end-point-error
    """
    return torch.mean((flow1 - flow2)**2)

def f_epe(pred, target):
    """Wrapper function to compute the average endpoint error between prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar average endpoint error 
    """
    return avg_epe(pred, target)


def f_mse(pred, target):
    """Wrapper function to compute the mean squared error between prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar average squared end-point-error
    """
    return avg_mse(pred, target)


def f_cosim(pred, target):
    """Compute the mean cosine similarity between the two flow fields prediction and target

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Returns:
        float: scalar mean cosine similarity
    """
    return 1 - torch.sum( pred * target) / torch.sqrt(torch.sum(pred*pred)) * torch.sqrt(torch.sum(target*target))


def two_norm_avg_delta(delta1, delta2):
    """Computes the mean of the L2-norm of two perturbations used during PCFA.

    Args:
        delta1 (tensor):
            perturbation applied to the first image
        delta2 (tensor):
            perturbation applied to the second image

    Returns:
        float: scalar average L2-norm of two perturbations
    """
    numels_delta1 = torch.numel(delta1)
    numels_delta2 = torch.numel(delta2)
    sqrt_numels = (numels_delta1 + numels_delta2)**(0.5)
    two_norm = torch.sqrt(torch.sum(torch.pow(torch.flatten(delta1), 2)) + torch.sum(torch.pow(torch.flatten(delta2), 2)))
    return two_norm / sqrt_numels


def two_norm_avg_delta_squared(delta1, delta2):
    """Computes the mean of the squared L2-norm of two perturbations used during PCFA.

    Args:
        delta1 (tensor):
            perturbation applied to the first image
        delta2 (tensor):
            perturbation applied to the second image

    Returns:
        float: scalar average squared L2-norm of two perturbations
    """
    numels_delta1 = torch.numel(delta1)
    numels_delta2 = torch.numel(delta2)
    numels = numels_delta1 + numels_delta2
    two_norm = torch.sum(torch.pow(torch.flatten(delta1), 2)) + torch.sum(torch.pow(torch.flatten(delta2), 2))
    return two_norm / numels


def two_norm_avg(x):
    """Computes the L2-norm of the input normalized by the root of the number of elements.

    Args:
        x (tensor):
            input tensor with variable dimensions

    Returns:
        float: normalized L2-norm
    """
    numels_x = torch.numel(x)
    sqrt_numels = numels_x**0.5
    two_norm = torch.sqrt(torch.sum(torch.pow(torch.flatten(x), 2)))
    return two_norm / sqrt_numels


def get_loss(f_type, pred, target):
    """Wrapper to return a specified loss metric. 

    Args:
        f_type (str):
            specifies the returned metric. Options: [aee | mse | cosim]
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)

    Raises:
		NotImplementedError: Unknown metric.

    Returns:
        float: scalar representing the loss measured with the specified norm
    """

    similarity_term = None

    if f_type == "aee":
        similarity_term = f_epe(pred, target)
    elif f_type == "cosim":
        similarity_term = f_cosim(pred, target)
    elif f_type == "mse":
        similarity_term = f_mse(pred, target)
    else:
        raise(NotImplementedError, "The requested loss type %s does not exist. Please choose one of 'aee', 'mse' or 'cosim'" % (f_type))

    return similarity_term


def relu_penalty(delta1, delta2, device, delta_bound=0.001):
    """Implementation of the penalty term.
    The penalty function linearly penalizes deviations from a constraint and is otherwise zero.
    This is implemented using the ReLU function.

    Args:
        delta1 (tensor):
            perturbation for image1
        delta2 (tensor):
            perturbation for image2
        device (torch.device):
            changes the selected device
        delta_bound (float, optional):
            L2-constraint for the perturbation. Defaults to 0.001.

    Returns:
        float: scalar penalty value
    """
    zero_tensor = torch.tensor(0.).to(device)
    delta_minus_bound = two_norm_avg_delta_squared(delta1, delta2) - torch.tensor(delta_bound**2).to(device)
    return torch.max(zero_tensor, delta_minus_bound) # This is relu( ||delta||**2-delta_bond**2).


def loss_delta_constraint(pred, target, delta1, delta2, device, delta_bound=0.001, mu=100., f_type="aee"):
    """Penalty method to optimize the perturbations.
    An exact penalty function is used to transform the inequality constrained problem into an
    unconstrained optimization problem.

    Args:
        pred (tensor):
            predicted flow field (must have same dimensions as target)
        target (tensor):
            specified target flow field (must have same dimensions as prediction)
        delta1 (tensor):
            perturbation for image1
        delta2 (tensor):
            perturbation for image2
        device (torch.device):
            changes the selected device
        delta_bound (float, optional):
            L2-constraint for the perturbation. Defaults to 0.001.
        mu (_type_, optional):
            penalty parameter which enforces the unconstrained the specified constraint. Defaults to 100..
        f_type (str, optional):
            specifies the metric used for comparing prediction and target. Options: [aee | mse | cosim]. Defaults to "aee".

    Returns:
        _type_: _description_
    """

    similarity_term = get_loss(f_type, pred, target)
    penalty_term = relu_penalty(delta1, delta2, device, delta_bound) # This is relu( ||delta||**2-delta_bond**2).

    return similarity_term + mu * penalty_term