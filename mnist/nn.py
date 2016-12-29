
import timeit
import inspect
import sys
import numpy
import theano
import theano.tensor as T


def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
    """
    batchnorm with support for not using scale and shift parameters
    as well as inference values (u and s) and partial batchnorm (via a)
    will detect and use convolutional or fully connected version
    """
    if X.ndim == 4:
        if u is not None and s is not None:
            b_u = u.dimshuffle('x', 0, 'x', 'x')
            b_s = s.dimshuffle('x', 0, 'x', 'x')
        else:
            b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
        if a != 1:
            b_u = (1. - a)*0. + a*b_u
            b_s = (1. - a)*1. + a*b_s
        X = (X - b_u) / T.sqrt(b_s + e)
        if g is not None and b is not None:
            X = X*g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
    elif X.ndim == 2:
        if u is None and s is None:
            u = T.mean(X, axis=0)
            s = T.mean(T.sqr(X - u), axis=0)
        if a != 1:
            u = (1. - a)*0. + a*u
            s = (1. - a)*1. + a*s
        X = (X - u) / T.sqrt(s + e)
        if g is not None and b is not None:
            X = X*g + b
    else:
        raise NotImplementedError
    return X

def ReLU(x):
    return T.switch(x < 0., 0., x)

def leakyRelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

# We find the adam update rules on the website below
# https://github.com/EderSantana/top/blob/master/top/update_rules.py
floatX = theano.config.floatX
def adam(params, cost, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, grad_clip=None):
    """Adam algorithm proposed was proposed in Adam: A Method for Stochastic
    Optimization.
    This code was modified from Newmu's (Alec Radford) code:
    https://gist.github.com/Newmu/acb738767acb4788bac3
    :param params: list of :class:theano.shared variables to be optimized
    :param cost: cost function that should be minimized in the optimization
    :param float lr: learning rate
    :param float b1: ToDo: WRITEME
    :param float b2: ToDo: WRITEME
    :param float e: ToDO: WRITEME
    """
    updates = []
    grads = T.grad(cost, params)
    zero = numpy.zeros(1).astype(floatX)[0]
    i = theano.shared(zero)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        if grad_clip is not None:
            gnorm = T.sqrt(T.sqr(g).sum())
            ggrad = T.switch(T.ge(gnorm,grad_clip),
                             grad_clip*g/gnorm, g)
        else:
            ggrad = g
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * ggrad) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(ggrad)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def generate_noise(rng, batch_size):
    return rng.uniform(-1.0, 1.0, size=(batch_size, 100)).astype(dtype=theano.config.floatX)

