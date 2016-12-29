import timeit
import inspect
import sys
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


from theano.tensor.nnet import conv2d
from theano.tensor.extra_ops import repeat
from nn import ReLU, batchnorm, leakyRelu
from nn import adam, generate_noise

from utils import load_mnist


nkerns = [32, 64, 128]

def gen(Z, w, w1, w2, w3):
    h0 = ReLU(batchnorm(T.dot(Z, w)))
    h1 = ReLU(batchnorm(T.dot(h0, w1)))
    h1_output = h1.reshape((h1.shape[0], nkerns[2], 7, 7))
    h2_input = repeat(repeat(h1_output, 2, 2), 2, 3)
    h2 = ReLU(batchnorm(conv2d(h2_input, w2, border_mode='half')))
    h3_input = repeat(repeat(h2, 2, 2), 2, 3)
    h3 = T.tanh(conv2d(h3_input, w3, border_mode='half'))
    return h3

def discrim(X, w, w1, w2, w3):
    h0 = leakyRelu(conv2d(X.reshape((X.shape[0], 1, 28, 28)), w, border_mode='half', subsample=(2, 2)))
    h1 = leakyRelu(batchnorm(conv2d(h0, w1, border_mode='half', subsample=(2, 2))))
    h2 = leakyRelu(batchnorm(T.dot(h1.flatten(2), w2)))
    h3 = T.nnet.sigmoid(T.dot(h2, w3))
    return h3


def train_dcgan(n_epochs=60, batch_size=128, verbose=True):

    rng = np.random.RandomState(123)

    #########################
    # LOAD DATA: CHECK CODE #
    #########################
    # AND ADD FROM * IMPORT #
    #########################

    train_set_x = load_mnist(dataset="training")
    valid_set_x = load_mnist(dataset="testing")

    #########################

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0]
    n_valid_batches = valid_set_x.shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    X = T.ftensor4('X')    # the data is presented as rasterized images
    Z = T.fmatrix('Z')   # the input of the generator is random matrix
                         # of size batch_size x 100 uniform

    # Set up Y variables
    half_batch = batch_size//2
    ones = np.ones(batch_size).astype(dtype='int32')
    zeros = np.zeros(batch_size).astype(dtype='int32')
    ones_zeros = np.concatenate((ones[:half_batch], zeros[:half_batch]))
    
    # Define theano shared variables
    gw = theano.shared(np.asarray(rng.normal(0.0, 0.02, (100, 1024)),
                                  dtype=theano.config.floatX), name='gw')
    gw1 = theano.shared(np.asarray(rng.normal(0.0, 0.02, (1024, 128*7*7)),
                                  dtype=theano.config.floatX), name='gw1')
    gw2 = theano.shared(np.asarray(rng.normal(0.0, 0.02, (nkerns[1], nkerns[2], 5, 5)),
                                  dtype=theano.config.floatX), name='gw2')
    gw3 = theano.shared(np.asarray(rng.normal(0.0, 0.02, (1, nkerns[1], 5, 5)),
                                  dtype=theano.config.floatX), name='gw3')
    
    dw = theano.shared(np.asarray(rng.normal(0.0, 0.02, (nkerns[1], 1, 5, 5)),
                                  dtype=theano.config.floatX), name='dw')
    dw1 = theano.shared(np.asarray(rng.normal(0.0, 0.02, (nkerns[2], nkerns[1], 5, 5)),
                                  dtype=theano.config.floatX), name='dw1')
    dw2 = theano.shared(np.asarray(rng.normal(0.0, 0.02, (128*7*7, 1024)),
                                  dtype=theano.config.floatX), name='dw2')
    dw3 = theano.shared(np.asarray(rng.normal(0.0, 0.02, 1024),
                                  dtype=theano.config.floatX), name='dw3')
    
    # define parameters of the generator and discriminator
    g_params = [gw, gw1, gw2, gw3]
    d_params = [dw, dw1, dw2, dw3]
    
    gX = gen(Z, gw, gw1, gw2, gw3)
    
    p_real = discrim(X, dw, dw1, dw2, dw3)
    p_gen = discrim(gX, dw, dw1, dw2, dw3)
    
    # Define the cost function of the model
    d_cost_real = T.nnet.binary_crossentropy(p_real, T.ones(p_real.shape)).mean()
    d_cost_gen = T.nnet.binary_crossentropy(p_gen, T.zeros(p_gen.shape)).mean()
    g_cost = T.nnet.binary_crossentropy(p_gen, T.ones(p_gen.shape)).mean()
    
    d_cost = d_cost_real + d_cost_gen
    
    cost = [g_cost, d_cost, d_cost_real, d_cost_gen]
    
    # Define the parameter update rule for both models
    updates_d = adam(d_params, d_cost)
    updates_g = adam(g_params, g_cost)
    
    # compile functions
    train_d = theano.function([X, Z], cost, updates=updates_d)
    train_g = theano.function([X, Z], cost, updates=updates_g)
    generate = theano.function([Z], gX)
    
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 15000  # look as this many examples regardless
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    # test_score = 0.
    start_time = timeit.default_timer()
    
    for epoch in range(n_epochs):
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter % 1000 == 0) and verbose:
                print('training @ iter = ', iter)
            
            xmb = train_set_x[minibatch_index * batch_size: (minibatch_index + 1) * batch_size].reshape((batch_size, 1, 28, 28))
            zmb = generate_noise(rng, batch_size)
            
            cost_d = train_d(xmb, zmb)
            cost_g = train_g(xmb, zmb)
            
            if (iter + 1) % validation_frequency == 0:

                if verbose:
                    print('epoch %i, minibatch %i/%i' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches))
                    print 'cost_d', cost_d, 'cost_g', cost_g

    end_time = timeit.default_timer()
    calframe = inspect.getouterframes(curframe, 2)

    print('Optimization complete.')
    print(('The training process for function ' +
           calframe[1][3] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), sys.stderr)

    return g_params, d_params

