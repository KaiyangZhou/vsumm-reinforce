import theano
from theano import tensor as T

import theano_nets
from model_reinforceRNN import reinforceRNN

import numpy as np
from datetime import datetime
import time, math, os, sys, h5py, logging, vsum_tools, argparse
from scipy.spatial.distance import cdist

_DTYPE = theano.config.floatX

def test(n_episodes=5,
         input_dim=1024,
         hidden_dim=256,
         W_init='normal',
         U_init='normal',
         weight_decay=1e-5,
         regularizer='L2',
         optimizer='adam',
         alpha=0.01,
         model_file='',
         eval_dataset='summe',
         verbose=True,
         ):
    assert eval_dataset in ['summe', 'tvsum']
    assert os.path.isfile(model_file)

    if eval_dataset == 'summe':
        eval_metric = 'max'
    elif eval_dataset == 'tvsum':
        eval_metric = 'avg'
    model_options = locals().copy()

    log_dir = 'log-test'
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

    logger.info('initializing net model')
    net = reinforceRNN(model_options)

    logger.info('loading %s data' % (eval_dataset))
    h5f_path = 'datasets/eccv16_dataset_' + eval_dataset + '_google_pool5.h5'
    dataset = h5py.File(h5f_path, 'r')
    if sys.version_info[0] == 3:
        dataset_keys = list(dataset.keys())
    else:
        dataset_keys = dataset.keys()
    n_videos = len(dataset_keys)

    logger.info('=> testing')
    start_time = time.time()
    fms = []
    precs = []
    recs = []

    for i_video in range(n_videos):
        key = dataset_keys[i_video]
        data_x = dataset[key]['features'][...].astype(_DTYPE)
        probs = net.model_inference(data_x)

        cps = dataset[key]['change_points'][...]
        n_frames = dataset[key]['n_frames'][()]
        nfps = dataset[key]['n_frame_per_seg'][...].tolist()
        positions = dataset[key]['picks'][...]

        machine_summary = vsum_tools.generate_summary(probs, cps, n_frames, nfps, positions)
        user_summary = dataset[key]['user_summary'][...]
        fm,prec,rec = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
        fms.append(fm)
        precs.append(prec)
        recs.append(rec)
        if verbose: logger.info('video %s. fm=%f' % (key, fm))

    mean_fm = np.mean(fms)
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)

    logger.info('========================= conclusion =========================')
    logger.info('-- recap of model options')
    logger.info(str(model_options))
    logger.info('-- final outcome')
    logger.info('f-measure {:.1%}. precision {:.1%}. recall {:.1%}.'.format(mean_fm, mean_prec, mean_rec))
    elapsed_time = time.time() - start_time
    logger.info('elapsed time %.2f s' % (elapsed_time))
    logger.info('==============================================================')

    dataset.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='', metavar='PATH')
    parser.add_argument('-d', type=str, default='tvsum')
    parser.add_argument('--in-dim', type=int, default=1024,
                        help="input dimension, i.e. dimension of CNN features")
    parser.add_argument('--h-dim', type=int, default=256,
                        help="hidden dimension of RNN")
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    test(input_dim=args.in_dim,
         hidden_dim=args.h_dim,
         model_file=args.model,
         eval_dataset=args.d,
         verbose=args.verbose)