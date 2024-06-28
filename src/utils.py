import os
import sys
import warnings
import torch
import pickle
import numpy as np
from itertools import product

def to_pth(np_array, pth_dtype=torch.FloatTensor):
    return torch.from_numpy(np_array).type(pth_dtype)

def to_sqpth(np_array, pth_dtype=torch.FloatTensor):
    return torch.squeeze(to_pth(np_array, pth_dtype=pth_dtype))

def to_np(torch_tensor):
    return torch_tensor.data.numpy()

def to_sqnp(torch_tensor, dtype=np.float64):
    return np.array(np.squeeze(to_np(torch_tensor)), dtype=dtype)


def enumerated_product(*args):
    # https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


def ignore_warnings():
    if not sys.warnoptions:
        warnings.simplefilter("ignore")


def pickle_save_dict(input_dict, save_path):
    """Save the dictionary

    Parameters
    ----------
    input_dict : type
        Description of parameter `input_dict`.
    save_path : type
        Description of parameter `save_path`.

    """
    with open(save_path, 'wb') as handle:
        pickle.dump(input_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load_dict(fpath):
    """load the dict

    Parameters
    ----------
    fpath : type
        Description of parameter `fpath`.

    Returns
    -------
    type
        Description of returned object.

    """
    return pickle.load(open(fpath, "rb"))

def mkdir(dir_name, verbose=False):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        if verbose:
            print(f'Dir created: {dir_name}')
    else:
        if verbose:
            print(f'Dir exist: {dir_name}')



def one_hot_vector(i, k):
    """
    Create a k-dimensional one-hot vector with the i-th dimension set to 1.

    Parameters:
    - i (int): The index of the dimension to set to 1.
    - k (int): The total number of dimensions.

    Returns:
    - list: A one-hot vector represented as a list.
    """
    if i < 0 or i >= k:
        raise ValueError("Invalid index i for dimension k")

    # Create a list of zeros with length k
    one_hot = [0] * k

    # Set the i-th dimension to 1
    one_hot[i] = 1

    return np.array(one_hot)


if __name__ == "__main__":
    '''how to use'''

    # Example usage:
    i = 2  # Index of the dimension to set to 1
    k = 5  # Total number of dimensions
    result = one_hot_vector(i, k)
    print(result)  # Output: [0, 0, 1, 0, 0]
