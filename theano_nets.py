import numpy as np
import theano
from theano import tensor as T
import h5py

"""
This script implements necessary functions and layers for
constructing recurrent neural networks.
"""

""" Global variable setting """
_DTYPE = theano.config.floatX
_EPSILON = 1e-8

""" Initializers """
def init_choices():
    return ['orthogonal', 'glorot_uniform', 'uniform', 'normal']
""" Saxe et al. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. 2013. """
def orthogonal(n_dim=None):
    assert n_dim is not None
    w = np.random.randn(n_dim, n_dim)
    u,_,_ = np.linalg.svd(w)
    return u.astype(_DTYPE)

""" Glorot and Bengio. Understanding the difficulties of training deep feedforward neural networks. 2010. """
def glorot_uniform(n_in=None, n_out=None):
    assert n_in is not None
    if n_out is None: n_out = n_in
    w = np.random.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)
        )
    return w.astype(_DTYPE)

def uniform(n_in=None, n_out=None):
    assert n_in is not None
    if n_out is None: n_out = n_in
    w = np.random.uniform(
        low=-1,
        high=1,
        size=(n_in, n_out)
        )
    return w.astype(_DTYPE) * 0.02

def normal(n_in=None, n_out=None):
    assert n_in is not None
    if n_out is None: n_out = n_in
    w = np.random.randn(n_in, n_out) * 0.01
    return w.astype(_DTYPE)

""" Shortcut functions """
def tanh(x):
    return T.tanh(x)

def softmax(x):
    if x.ndim == 2:
        return T.nnet.softmax(x)
    elif x.ndim == 3:
        exp_x = T.exp(x)
        sum_exp_x = exp_x.sum(2)[:,:,None]
        return exp_x / sum_exp_x
    else:
        raise ValueError('softmax only accepts input dimension of either 2 or 3')

def sigmoid(x):
    return T.nnet.sigmoid(x)

def relu(x):
    return T.nnet.relu(x)

def linear(x):
    return x

""" Regularizers """
def reg_choices():
    return ['L2', 'L1', 'L1L2']
def L2(params):
    reg_term = 0
    for p in params:
        # only apply to weight but not bias
        if p.ndim > 1:
            reg_term += T.sum(p**2)
    return reg_term

def L1(params):
    reg_term = 0
    for p in params:
        if p.ndim > 1:
            reg_term += T.sum(T.abs_(p))
    return reg_term

def L1L2(params):
    reg_term = 0
    for p in params:
        if p.ndim > 1:
            reg_term += T.sum(p**2) + T.sum(T.abs_(p))
    return reg_term

""" Optimization algorithms """
def optim_choices():
    return ['rmsprop', 'adam', 'sgd']
def rmsprop(params, grads, learn_rate):
    #learn_rate = theano.shared(np.asarray(learn_rate, dtype=_DTYPE))
    rho = theano.shared(np.asarray(0.9, dtype=_DTYPE))
    epsilon = theano.shared(np.asarray(1e-8, dtype=_DTYPE))
    updates = []

    shapes = [p.get_value().shape for p in params]
    accums = [theano.shared(np.zeros(shape, dtype=_DTYPE)) for shape in shapes]

    for p,g,a in zip(params, grads, accums):
        new_a = rho * a + (1. - rho) * T.sqr(g)
        new_p = p - learn_rate * g / (T.sqrt(new_a) + epsilon)

        updates.append((a, new_a))
        updates.append((p, new_p))

    return updates

""" Kingma and Ba. Adam: a method for stochastic optimization. In ICLR, 2014. """
def adam(params, grads, learn_rate):
    #learn_rate = theano.shared(np.asarray(learn_rate, dtype=_DTYPE))
    epsilon = theano.shared(np.asarray(1e-8, dtype=_DTYPE))
    beta1 = theano.shared(np.asarray(0.9, dtype=_DTYPE))
    beta2 = theano.shared(np.asarray(0.999, dtype=_DTYPE))
    t = theano.shared(np.asarray(1, dtype=_DTYPE))
    updates = []

    shapes = [p.get_value().shape for p in params]
    ms = [theano.shared(np.zeros(shape, dtype=_DTYPE)) for shape in shapes]
    vs = [theano.shared(np.zeros(shape, dtype=_DTYPE)) for shape in shapes]

    alpha = learn_rate * T.sqrt(1 - beta2**t) / (1 - beta1**t)

    for p,g,m,v in zip(params, grads, ms, vs):
        new_m = beta1 * m + (1 - beta1) * g
        new_v = beta2 * v + (1 - beta2) * (g**2)
        new_p = p - alpha * new_m / (T.sqrt(new_v) + epsilon)
        updates.append((m, new_m))
        updates.append((v, new_v))
        updates.append((p, new_p))

    updates.append((t, t+1))

    return updates

def sgd(params, grads, learn_rate):
    #learn_rate = theano.shared(np.asarray(learn_rate, dtype=_DTYPE))
    momentum = theano.shared(np.asarray(0.9, dtype=_DTYPE))
    updates = []

    shapes = [p.get_value().shape for p in params]
    vs = [theano.shared(np.zeros(shape, dtype=_DTYPE)) for shape in shapes]

    for p,g,v in zip(params, grads, vs):
        new_v = momentum * v + learn_rate * g
        new_p = p - new_v
        updates.append((v, new_v))
        updates.append((p, new_p))

    return updates

""" Layers
Note: minibatch training is not considered here due to reward computation """
def Dropout(state_below, trng, dropout_factor):
    return T.switch(dropout_factor,
                    state_below *
                    trng.binomial(state_below.shape, p=0.5, n=1, dtype=state_below.dtype),
                    state_below * 0.5)

class FC(object):
    def __init__(self,
                 state_below=None,
                 input_dim=None,
                 output_dim=None,
                 activation='sigmoid',
                 W_init='uniform',
                 use_bias=True,
                 layer_name='fc',
                 model_file=None
                 ):
        self.state_below = state_below
        self.n_in = input_dim
        self.n_out = output_dim
        self.activation = activation
        self.W_init = W_init
        self.use_bias = use_bias
        self.layer_name = layer_name
        self.model_file = model_file
        self.build()
        self.output = self.step(self.state_below)

    def build(self):
        self.params = []
        if self.model_file is None:
            self.W = theano.shared(eval(self.W_init)(self.n_in, self.n_out), name=self.layer_name + '_W')
            self.params += [self.W]
            if self.use_bias:
                self.b = theano.shared(np.zeros((self.n_out), dtype=_DTYPE), name=self.layer_name + '_b')
                self.params += [self.b]
        else:
            with h5py.File(self.model_file, 'r') as saved_model:
                self.W = theano.shared(saved_model[self.layer_name + '_W'][...].astype(_DTYPE), name=self.layer_name + '_W')
                self.params += [self.W]
                if self.use_bias:
                    self.b = theano.shared(saved_model[self.layer_name + '_b'][...].astype(_DTYPE), name=self.layer_name + '_b')
                    self.params += [self.b]

    def step(self, x):
        yraw = T.dot(x, self.W)
        if self.use_bias:
            yraw += self.b
        y = eval(self.activation)(yraw)
        return y

    def reset_params(self):
        self.W.set_value(eval(self.W_init)(self.n_in, self.n_out))
        if self.use_bias:
            self.b.set_value(np.zeros((self.n_out), dtype=_DTYPE))

    def load_params(self, model_file=None):
        assert model_file is not None
        with h5py.File(model_file, 'r') as saved_model:
            self.W.set_value(saved_model[self.layer_name + '_W'][...].astype(_DTYPE))
            if self.use_bias:
                self.b.set_value(saved_model[self.layer_name + '_b'][...].astype(_DTYPE))

class GRU(object):
    def __init__(self,
                 state_below=None,
                 input_dim=None,
                 output_dim=None,
                 W_init='normal',
                 U_init='normal',
                 init_state=None,
                 go_backwards=False,
                 layer_name='gru',
                 model_file=None
                 ):
        self.state_below = state_below
        self.n_in = input_dim
        self.n_out = output_dim
        self.W_init = W_init
        self.U_init = U_init
        self.init_state = init_state
        self.go_backwards = go_backwards
        self.layer_name = layer_name
        self.model_file = model_file
        self.build()
        self.output = self.step(self.state_below)

    def build(self):
        if self.model_file is None:
            self.W = theano.shared(np.concatenate([eval(self.W_init)(self.n_in, self.n_out),
                                                   eval(self.W_init)(self.n_in, self.n_out),
                                                   eval(self.W_init)(self.n_in, self.n_out)], axis=1), name=self.layer_name + '_W')
            self.U = theano.shared(np.concatenate([eval(self.U_init)(self.n_out),
                                                   eval(self.U_init)(self.n_out),
                                                   eval(self.U_init)(self.n_out)], axis=1), name=self.layer_name + '_U')
            self.b = theano.shared(np.zeros((3 * self.n_out), dtype=_DTYPE), name=self.layer_name + '_b')
        else:
            with h5py.File(self.model_file, 'r') as saved_model:
                self.W = theano.shared(saved_model[self.layer_name + '_W'][...].astype(_DTYPE), name=self.layer_name + '_W')
                self.U = theano.shared(saved_model[self.layer_name + '_U'][...].astype(_DTYPE), name=self.layer_name + '_U')
                self.b = theano.shared(saved_model[self.layer_name + '_b'][...].astype(_DTYPE), name=self.layer_name + '_b')
        self.params = [self.W, self.U, self.b]

    def step(self, x):
        if self.init_state == None:
            init_state = T.alloc(0., self.n_out)
        else:
            init_state = self.init_state

        def _slice(_x, n, dim):
            return _x[n*dim:(n+1)*dim]

        def _recurrence(_x, _h):
            matrix_r = _slice(_x, 0, self.n_out)
            matrix_z = _slice(_x, 1, self.n_out)
            matrix_h = _slice(_x, 2, self.n_out)
            inner_r = T.dot(_h, self.U[:,:self.n_out])
            inner_z = T.dot(_h, self.U[:,self.n_out:2*self.n_out])
            r = sigmoid(matrix_r + inner_r)
            z = sigmoid(matrix_z + inner_z)
            inner_h = T.dot(r*_h, self.U[:,2*self.n_out:])
            h_p = tanh(matrix_h + inner_h)
            h = (1 - z) * _h + z * h_p
            return h

        x = T.dot(x, self.W) + self.b
        rval,_ = theano.scan(
          fn=_recurrence,
          sequences=x,
          outputs_info=init_state,
          go_backwards=self.go_backwards
        )
        if self.go_backwards:
            rval = rval[::-1,...]
        return rval

    def reset_params(self):
        self.W.set_value(np.concatenate([eval(self.W_init)(self.n_in, self.n_out),
                                         eval(self.W_init)(self.n_in, self.n_out),
                                         eval(self.W_init)(self.n_in, self.n_out)], axis=1))
        self.U.set_value(np.concatenate([eval(self.U_init)(self.n_out),
                                         eval(self.U_init)(self.n_out),
                                         eval(self.U_init)(self.n_out)], axis=1))
        self.b.set_value(np.zeros((3 * self.n_out), dtype=_DTYPE))

    def load_params(self, model_file=None):
        assert model_file is not None
        with h5py.File(model_file, 'r') as saved_model:
            self.W.set_value(saved_model[self.layer_name + '_W'][...].astype(_DTYPE))
            self.U.set_value(saved_model[self.layer_name + '_U'][...].astype(_DTYPE))
            self.b.set_value(saved_model[self.layer_name + '_b'][...].astype(_DTYPE))

class LSTM(object):
    def __init__(self,
                 state_below=None,
                 input_dim=None,
                 output_dim=None,
                 W_init='glorot_uniform',
                 U_init='orthogonal',
                 init_state=None,
                 init_memory=None,
                 go_backwards=False,
                 layer_name='lstm',
                 model_file=None
                 ):
        self.state_below = state_below
        self.n_in = input_dim
        self.n_out = output_dim
        self.W_init = W_init
        self.U_init = U_init
        self.init_state = init_state
        self.init_memory = init_memory
        self.go_backwards = go_backwards
        self.layer_name = layer_name
        self.model_file = model_file
        self.build()
        self.output = self.step(self.state_below)

    def build(self):
        if self.model_file is None:
            self.W = theano.shared(np.concatenate([eval(self.W_init)(self.n_in, self.n_out),
                                                   eval(self.W_init)(self.n_in, self.n_out),
                                                   eval(self.W_init)(self.n_in, self.n_out),
                                                   eval(self.W_init)(self.n_in, self.n_out)], axis=1), name=self.layer_name + '_W')
            self.U = theano.shared(np.concatenate([eval(self.U_init)(self.n_out),
                                                   eval(self.U_init)(self.n_out),
                                                   eval(self.U_init)(self.n_out),
                                                   eval(self.U_init)(self.n_out)], axis=1), name=self.layer_name + '_U')
            self.b = theano.shared(np.zeros((4 * self.n_out), dtype=_DTYPE), name=self.layer_name + '_b')
        else:
            with h5py.File(self.model_file, 'r') as saved_model:
                self.W = theano.shared(saved_model[self.layer_name + '_W'][...].astype(_DTYPE), name=self.layer_name + '_W')
                self.U = theano.shared(saved_model[self.layer_name + '_U'][...].astype(_DTYPE), name=self.layer_name + '_U')
                self.b = theano.shared(saved_model[self.layer_name + '_b'][...].astype(_DTYPE), name=self.layer_name + '_b')
        self.params = [self.W, self.U, self.b]

    def step(self, x):
        if self.init_state == None:
            init_state = T.alloc(0., self.n_out)
        else:
            init_state = self.init_state
        if self.init_memory == None:
            init_memory = T.alloc(0., self.n_out)
        else:
            init_memory = self.init_memory

        def _slice(_x, n, dim):
            return _x[n*dim:(n+1)*dim]

        def _recurrence(_x, _h, _c):
            preact = T.dot(_h, self.U)
            preact += _x
            i = _slice(preact, 0, self.n_out)
            f = _slice(preact, 1, self.n_out)
            o = _slice(preact, 2, self.n_out)
            g = _slice(preact, 3, self.n_out)
            i = sigmoid(i)
            f = sigmoid(f)
            o = sigmoid(o)
            g = tanh(g)
            c = f * _c + i * g
            h = o * tanh(c)
            rval = [h, c]
            return rval

        x = T.dot(x, self.W) + self.b
        rval,_ = theano.scan(
          fn=_recurrence,
          sequences=x,
          outputs_info=[init_state, init_memory],
          go_backwards=self.go_backwards
        )
        h = rval[0]
        if self.go_backwards:
            h = h[::-1,...]
        return h

    def reset_params(self):
        self.W.set_value(np.concatenate([eval(self.W_init)(self.n_in, self.n_out),
                                         eval(self.W_init)(self.n_in, self.n_out),
                                         eval(self.W_init)(self.n_in, self.n_out),
                                         eval(self.W_init)(self.n_in, self.n_out)], axis=1))
        self.U.set_value(np.concatenate([eval(self.U_init)(self.n_out),
                                         eval(self.U_init)(self.n_out),
                                         eval(self.U_init)(self.n_out),
                                         eval(self.U_init)(self.n_out)], axis=1))
        self.b.set_value(np.zeros((4 * self.n_out), dtype=_DTYPE))

    def load_params(self, model_file=None):
        assert model_file is not None
        with h5py.File(model_file, 'r') as saved_model:
            self.W.set_value(saved_model[self.layer_name + '_W'][...].astype(_DTYPE))
            self.U.set_value(saved_model[self.layer_name + '_U'][...].astype(_DTYPE))
            self.b.set_value(saved_model[self.layer_name + '_b'][...].astype(_DTYPE))
