import os, struct
import lmdb
import numpy as np

import theano
import theano.tensor as T
from array import array as pyarray


def load_mnist(dataset="training", digits=np.arange(10), path="../data"):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype='float32')
    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))/127.5 - 1.

    return images

