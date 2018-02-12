# Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward
We propose a reinforcement learning framework to train recurrent neural network for unsupervised video summarization. The reward function consists of a diversity reward and a representativeness reward. It assesses how diverse and representative the generated video summaries are during training, while the network tries to earn higher rewards by producing more diverse and more representative summaries. The reward function is fully unsupervised, so no labels or human interactions are required at all. We have tested our method on two commonly used benchmark datasets, SumMe and TVSum. Our results not only outperform other state-of-the-art unsupervised methods, but also are comparable or even better than most supervised methods.

We implemented our method using [Theano](http://deeplearning.net/software/theano/) (version `0.9.0`). Paper is available [here](https://arxiv.org/abs/1801.00054).

## Preparation
To get the datasets and models, you will need `wget`.

Run the following commands in order
```
git clone https://github.com/KaiyangZhou/vsumm-reinforce
cd vsumm-reinforce
# download datasets.tar.gz (173.5 MB)
wget http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/datasets.tar.gz
tar -xvzf datasets.tar.gz
# download models.tar.gz (39.4 MB)
wget http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/models.tar.gz
tar -xvzf models.tar.gz
```

(Email me if you are unable to download these files)

## How to train
Training code is implemented in `vsum_train.py`. To train a RNN, run
```
python vsum_train.py --dataset datasets/eccv16_dataset_tvsum_google_pool5.h5 --max-epochs 60 --hidden-dim 256
```

## How to test
Test code is implemented in `vsum_test.py`. For example, to test with our models, simply run
```
python vsum_test.py -model models/model_tvsum_reinforceRNN.h5 -d tvsum
python vsum_test.py -model models/model_tvsum_reinforceRNN_sup.h5 -d tvsum
python vsum_test.py -model models/model_summe_reinforceRNN.h5 -d summe
python vsum_test.py -model models/model_summe_reinforceRNN_sup.h5 -d summe
```

## Citation
```
@article{zhou2017reinforcevsumm, 
   title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward},
   author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao}, 
   journal={arXiv:1801.00054}, 
   year={2017} 
}
```
