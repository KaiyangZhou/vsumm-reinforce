# Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward
This is a [Theano](http://deeplearning.net/software/theano/) implementation of our AAAI18 paper - [Deep RL for Unsupervised Video Summarization](https://arxiv.org/abs/1801.00054). We tested our code with Theano version `0.9.0`.

## Preparation
To get the datasets and models, you will need `wget`.

Run the following commands in order
```
git clone https://github.com/KaiyangZhou/vsumm-reinforce
cd vsumm-reinforce
wget http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/datasets.tar.gz
tar -xvzf datasets.tar.gz
wget http://www.eecs.qmul.ac.uk/~kz303/vsumm-reinforce/models.tar.gz
tar -xvzf models.tar.gz
```

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
   author={Zhou, Kaiyang and Qiao, Yu}, 
   journal={arXiv:1801.00054}, 
   year={2017} 
}
```
