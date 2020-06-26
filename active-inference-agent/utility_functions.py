import math

import numpy as np
import torch
from torch.autograd.gradcheck import zero_gradients


def min_max_norm(data, min, max):
    """
    Min-max normalisation
    :param data: data to be normalised
    :param min: min value (value that corresponds to -1 after normalisation)
    :param max: max value (value that corresponds to 1 after normalisation)
    :return: normalised data
    """
    return ((data - min) / (max - min)) * 2 - 1


def min_max_norm_dr(data, data_range):
    """
    Min-max normalisation with a given data range
    :param data: data to be normalised (must be of shape (1,2))
    :param data_range: range for normalisation
    :return: normalised data (1,2)
    """
    return np.array([[min_max_norm(data[0, 0], *data_range[0]), min_max_norm(data[0, 1], *data_range[1])]])


def undo_min_max_norm(data, min, max):
    """
    Transform data normalised between -1 and 1 to the provided range
    :param data: data to be de-normalised
    :param min: min value (values corresponding to -1 will get this value)
    :param max: max value (values corresponding to 1 will get this value)
    :return: de-normalised data
    """
    return (((data + 1) / 2) * (max - min)) + min


def undo_min_max_norm_dr(data, data_range):
    """
    Transform data normalised between -1 and 1 to the provided range
    :param data: data to be de-normalised (must be of shape (1,2))
    :param data_range: range for de-normalisation
    :return: de-normalised data (shape (1,2))
    """
    return np.array([[undo_min_max_norm(data[0, 0], *data_range[0]), undo_min_max_norm(data[0, 1], *data_range[1])]])


def add_gaussian_noise(array, mean, variance):
    """
    Add Gaussian noise to an array
    :param array: input array
    :param mean: mean for Gaussian noise
    :param variance: variance for Gaussian noise
    :return: noisy array
    """
    sigma = variance ** 0.5
    return array + np.random.normal(mean, sigma, array.shape)


# Retrieved from https://github.com/ast0414/adversarial-example/blob/master/craft.py
# Original license: GNU GPL v3.0
def compute_jacobian(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    n_rows = output.size()[1]
    n_collumns = output.size()[2]

    jacobian = torch.zeros(n_rows, n_collumns, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for r in range(n_rows):
        for c in range(n_collumns):
            zero_gradients(inputs)
            grad_output.zero_()
            grad_output[:, r, c] = 1
            output.backward(grad_output, retain_graph=True)
            jacobian[r, c] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)


# Written by Cansu Sancaktar (https://github.com/cansu97/PixelAI)
def shuffle_unison(a, b):
    """
    Shuffle two arrays in the same way
    :param a: array 1
    :param b: array 2
    :return: shuffled a and b
    """
    assert a.shape[0] == len(b)
    p = np.random.permutation(len(b))
    return a[p], b[p]


def forward_kinematics(shoulder_action, elbow_action):
    """
    Compute attempted end-effector action based on attempt shoulder and elbow rotation
    Note: only with the real arm locked in the center position
    :param shoulder_action: the attempted shoulder action
    :param elbow_action: the attempted elbow action
    :return: the attempted end-effector action
    """
    return shoulder_action * 0.3595 * math.cos(math.radians(31.218)) + elbow_action * 0.3694 * math.cos(
        math.radians(90 - 31.218))
