import numpy as np
import theano
from theano import tensor as T
from theano import In
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano_nets import *
import os, h5py

"""
Implementation of recurrent neural networks for video summarization.
"""

""" Global variable setting """
_DTYPE = theano.config.floatX

class reinforceRNN(object):
    def __init__(self, model_opts):
        input_dim = model_opts['input_dim']
        hidden_dim = model_opts['hidden_dim']
        weight_decay = model_opts['weight_decay']
        regularizer = model_opts['regularizer']
        W_init = model_opts['W_init']
        U_init = model_opts['U_init']
        optimizer = model_opts['optimizer']
        model_file = model_opts['model_file']
        n_episodes = model_opts['n_episodes']
        alpha = model_opts['alpha'] # coefficient for summary length penalty

        # Constructing computational graph
        x = T.matrix('data_matrix') # (n_steps, n_dim)
        learn_rate = T.scalar('learn_rate')
        baseline_reward = T.scalar('baseline_reward')
        L_dissim_mat = T.matrix('dissimilarity_matrix')
        L_distance_mat = T.matrix('distance_matrix')
        trng = RandomStreams(1234)
        
        self.fwd_rnn = LSTM(
          state_below=x, input_dim=input_dim, output_dim=hidden_dim,
          W_init=W_init, U_init=U_init, layer_name='fwd_rnn', model_file=model_file,
          init_state=None, init_memory=None, go_backwards=False
        )
        fwd_h_state = self.fwd_rnn.output
        self.bwd_rnn = LSTM(
          state_below=x, input_dim=input_dim, output_dim=hidden_dim,
          W_init=W_init, U_init=U_init, layer_name='bwd_rnn', model_file=model_file,
          init_state=None, init_memory=None, go_backwards=True
        )
        bwd_h_state = self.bwd_rnn.output

        h_state = T.concatenate([fwd_h_state, bwd_h_state], axis=1)
        h_state_dim = hidden_dim * 2

        self.fc_action = FC(
          state_below=h_state, input_dim=h_state_dim, output_dim=1,
          activation='sigmoid', W_init=W_init, layer_name='fc_action', model_file=model_file
        )
        probs = self.fc_action.output
        probs = probs.flatten()

        self.layers = [self.fwd_rnn, self.bwd_rnn, self.fc_action]
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        repeated_probs = T.extra_ops.repeat(probs[None,:], repeats=n_episodes, axis=0) # (n_episodes, n_steps)
        actions = trng.binomial(n=1, p=repeated_probs, size=repeated_probs.shape) # (n_episodes, n_steps)

        # Note: when modifing the reward computation function, make sure the corresponding single
        # function is modified too.
        def _compute_reward(_a, _L, _Ld):
            """
            _a: actions at timestep t
            _L: dissimilarity matrix
            _Ld: distance matrix
            """
            picks = _a.nonzero()[0]
            # compute diversity score
            Ly = _L[picks,:][:,picks]
            den_tmp = T.switch(_a.sum()-1, _a.sum()*(_a.sum()-1), 1.)
            den = T.switch(den_tmp, den_tmp, 1.)
            reward_div = Ly.sum() / den
            # compute representativeness score
            L_p = _Ld[:,picks]
            D_p = T.min(L_p, axis=1)
            reward_rep = T.exp(- T.mean(D_p))
            extra_mul = T.switch(_a.sum(), 1., 0.) # no actions, no reward
            reward_div *= extra_mul
            reward_rep *= extra_mul
            return reward_div,reward_rep

        (rewards_div,rewards_rep),_ = theano.scan(
          fn=_compute_reward,
          sequences=actions,
          non_sequences=[L_dissim_mat, L_distance_mat]
        )
        rewards = 0.5 * rewards_div + 0.5 * rewards_rep

        # Construct a cost function to be minimized
        cost = 0
        expected_reward = T.log(actions * repeated_probs + (1 - actions) * (1 - repeated_probs)) * (rewards - baseline_reward)[:,None]
        expected_reward = expected_reward.mean()
        cost -= expected_reward # maximizing expected reward equals to minimizing the negative version

        summary_length_penalty = (probs.mean() - 0.5)**2
        cost += summary_length_penalty * alpha

        if weight_decay > 0:
            weight_decay = theano.shared(np.asarray(weight_decay, dtype=_DTYPE))
            weight_reg = eval(regularizer)(self.params)
            cost += weight_decay * weight_reg

        grads = [T.grad(cost=cost, wrt=p) for p in self.params]
        updates = eval(optimizer)(self.params, grads, learn_rate)
        
        self.model_train = theano.function(
          inputs=[x, learn_rate, L_dissim_mat, L_distance_mat, baseline_reward],
          outputs=[rewards, rewards_div, rewards_rep],
          updates=updates,
          on_unused_input='ignore',
          allow_input_downcast=True
        )
        self.model_inference = theano.function(
          inputs=[x],
          outputs=probs,
          on_unused_input='ignore'
        )

    def get_n_params(self):
        n_params = 0
        for p in self.params:
            shp = p.get_value().shape
            if len(shp) == 1:
                n_params += shp[0]
            elif len(shp) == 2:
                n_params += shp[0] * shp[1]
        return n_params

    def reset_net(self):
        for layer in self.layers:
            layer.reset_params()

    def save_net(self, save_dir='saved_models', model_name='reinforceRNN'):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        file_name = os.path.join(save_dir, model_name + '.h5')
        model_file = h5py.File(file_name, 'w')
        for p in self.params:
            model_file[p.name] = p.get_value()
        model_file.close()

    def load_net(self, model_file=None):
        assert model_file is not None
        for layer in self.layers:
            layer.load_params(model_file)
