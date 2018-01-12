import theano
from theano import tensor as T

import theano_nets
from model_reinforceRNN import reinforceRNN

import numpy as np
from datetime import datetime
import time, math, os, sys, h5py, logging, vsum_tools, argparse
from scipy.spatial.distance import cdist

_DTYPE = theano.config.floatX

def train(n_episodes=5,
          input_dim=1024,
          hidden_dim=256,
          W_init='normal',
          U_init='normal',
          weight_decay=1e-5,
          regularizer='L2',
          optimizer='adam',
          base_lr=1e-5,
          decay_rate=0.1,
          max_epochs=60,
          decay_stepsize=30,
          ignore_distant_sim=True,
          distant_sim_thre=20,
          alpha=0.01,
          model_file=None,
          disp_freq=1,
          train_dataset_path='datasets/eccv16_dataset_tvsum_google_pool5.h5',
          ):
    model_options = locals().copy()

    log_dir = 'log-train'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logging.basicConfig(
      filename=log_dir+'/log.txt',
      filemode='w',
      format='%(asctime)s %(message)s',
      datefmt='[%d/%m/%Y %I:%M:%S]',
      level=logging.INFO
    )

    logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s',datefmt='[%d/%m/%Y %I:%M:%S]')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('model options: ' + str(model_options))
    logger.info('initializing net model')
    net = reinforceRNN(model_options)
    if model_file is not None: logger.info('loaded model from %s' % model_file)
    n_params = net.get_n_params()
    logger.info('net params size is %d' % (n_params))

    logger.info('loading training data from %s' % (train_dataset_path))
    dataset = h5py.File(train_dataset_path, 'r')
    dataset_keys = dataset.keys()
    n_videos = len(dataset_keys)
    logger.info('total number of videos for training is %d' % n_videos)

    logger.info('=> training')
    start_time = time.time()
    blrwds = {name:np.array(0).astype(_DTYPE) for name in dataset_keys} # baseline rewards

    reward_history = []

    for i_epoch in range(max_epochs):
        indices = np.arange(n_videos)
        np.random.shuffle(indices)
        epoch_reward = 0.
        epoch_reward_div = 0.
        epoch_reward_rep = 0.

        if decay_stepsize != -1 and i_epoch >= decay_stepsize:
            power_n = int(i_epoch/decay_stepsize)
            learn_rate = base_lr * (decay_rate**power_n)
        else:
            learn_rate = base_lr

        for index in indices:
            key = dataset_keys[index]
            data_x = dataset[key]['features'][...].astype(_DTYPE)
            L_distance_mat = cdist(data_x, data_x, 'euclidean')
            L_dissim_mat = 1 - np.dot(data_x, data_x.T)
            if ignore_distant_sim:
                inds = np.arange(data_x.shape[0])[:,None]
                inds_dist = cdist(inds, inds, 'minkowski', 1)
                L_dissim_mat[inds_dist > distant_sim_thre] = 1
            rewards, rewards_div, rewards_rep = net.model_train(data_x, learn_rate, L_dissim_mat, L_distance_mat, blrwds[key])
            blrwds[key] = 0.9 * blrwds[key] + 0.1 * rewards.mean()
            epoch_reward += rewards.mean()
            epoch_reward_div += rewards_div.mean()
            epoch_reward_rep += rewards_rep.mean()

        epoch_reward /= n_videos
        epoch_reward_div /= n_videos
        epoch_reward_rep /= n_videos

        if (i_epoch+1) % disp_freq == 0 or (i_epoch+1) == max_epochs:
            logger.info('epoch %3d/%d. reward %f. reward-div %f. reward-rep %f.' % \
                       (i_epoch+1, max_epochs, epoch_reward, epoch_reward_div, epoch_reward_rep))

        reward_history.append((epoch_reward, epoch_reward_div, epoch_reward_rep))

    elapsed_time = time.time() - start_time
    logger.info('elapsed time %.2f s' % (elapsed_time))
    net.save_net(save_dir=log_dir, model_name='model_reinforceRNN')

    dataset.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-epi', type=int, default=5,
                        help="number of episodes for REINFORCE")
    parser.add_argument('--input-dim', type=int, default=1024,
                        help="input dimension, i.e. dimension of CNN features")
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help="hidden dimension of RNN")
    parser.add_argument('--W-init', type=str, default='normal', choices=theano_nets.init_choices(),
                        help="initialization method for non-recurrent weights")
    parser.add_argument('--U-init', type=str, default='normal', choices=theano_nets.init_choices(),
                        help="initialization method for recurrent weights")
    parser.add_argument('--weight-decay', type=float, default=1e-5, 
                        help="coefficient for regularization on weight parameters")
    parser.add_argument('--reg', type=str, default='L2', choices=theano_nets.reg_choices(),
                        help="regularizer for weight parameters")
    parser.add_argument('--optim', type=str, default='adam', choices=theano_nets.optim_choices())
    parser.add_argument('--base-lr', type=float, default=1e-5, help="base learning rate")
    parser.add_argument('--decay-rate', type=float, default=0.1, help="learning rate decay")
    parser.add_argument('--max-epochs', type=int, default=60, help="maximum training epochs")
    parser.add_argument('--decay-stepsize', type=int, default=-1,
                        help="stepsize to decay learning rate, if -1, then learning rate decay is disabled")
    parser.add_argument('--ignore-distant-sim', action='store_true',
                        help="whether to ignore the similarity between distant frames")
    parser.add_argument('--distant-sim-thre', type=int, default=20,
                        help="threshold to ignore similarity between distant frames")
    parser.add_argument('--alpha', type=float, default=0.01, help="coefficient for summary length penalty")
    parser.add_argument('--disp-freq', type=int, default=1, help="display frequency")
    parser.add_argument('--dataset', type=str, default='datasets/eccv16_dataset_summe_google_pool5.h5')

    args = parser.parse_args()

    train(n_episodes=args.n_epi,
          input_dim=args.input_dim,
          hidden_dim=args.hidden_dim,
          W_init=args.W_init,
          U_init=args.U_init,
          weight_decay=args.weight_decay,
          regularizer=args.reg,
          optimizer=args.optim,
          base_lr=args.base_lr,
          decay_rate=args.decay_rate,
          max_epochs=args.max_epochs,
          decay_stepsize=args.decay_stepsize,
          ignore_distant_sim=args.ignore_distant_sim,
          distant_sim_thre=args.distant_sim_thre,
          alpha=args.alpha,
          disp_freq=args.disp_freq,
          train_dataset_path=args.dataset,)