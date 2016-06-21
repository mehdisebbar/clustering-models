"""
    Some tools to work on matrices
"""
import numpy as np


def check_zero_matrix(mat_list):
    """
    Return the list of matrices ids which are non empty
    :param mat_list: List of matrices, usually covariance matrices
    :return: list of ids of non empty matrices
    """
    non_zero_list = []
    for i in range(len(mat_list)):
        if np.count_nonzero(mat_list[i]) is not 0:
            non_zero_list.append(i)
    return non_zero_list


def clean_nans(x):
    if np.isnan(x).any():
        return np.nan_to_num(x)
    else:
        return x


def weights_compare(pi1, pi2):
    if len(pi1) == len(pi2):
        return ((np.array(pi1) - np.array(pi2)) ** 2).sum()
    elif len(pi1) < len(pi2):
        return ((np.array(pi1 + [0] * (len(pi2) - len(pi1))) - np.array(pi2)) ** 2).sum()
    else:
        return ((np.array(pi2 + [0] * (len(pi1) - len(pi2))) - np.array(pi1)) ** 2).sum()
