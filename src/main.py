"""
Usage :  python main.py  --exp lstm

"""

from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import torch
import pickle

from lstm.model import RNNModel
from data import get_data
from train import train

def dump_pkl(content, file_name):
    file = open(file_name, 'wb')
    pickle.dump(content, file)
    file.close()

def load_lstm():
    model_location = '../state_dict.pt'  
    lstm = RNNModel('LSTM', 50001, 650, 650, 2)
    lstm.load_state_dict(torch.load(model_location))

    with open('lstm/vocab.txt') as f:
        w2i = {w.strip(): i for i, w in enumerate(f)}

    vocab = defaultdict(lambda: w2i["<unk>"])
    vocab.update(w2i)
    return lstm, vocab


def execute_experiment(exp):
    if exp == "lstm":
        model, tokenizer = load_lstm()

    train_probe = True # of False
    if train_probe:
        print("Training the probe..")
        data = get_data(model, tokenizer)
        print("Data has been loaded.")
        
        n_epochs = 40
        test_uuas = train(data["train"], data["dev"], data["test"], n_epochs, exp)
        
        ## Rank Dim Experiment ##
        # rank_dim_list =  [pow(2,_) for _ in range(0,10)]
        # test_uuas_list = []
        # for rank_dim in rank_dim_list:
        #    test_uuas = train(data["train"], data["dev"], data["test"], n_epochs, exp, rank_dim=rank_dim)
        #    test_uuas_list.append(test_uuas)
        #    print("Test UUAS : {}, Rank Dim : {}".format(test_uuas, rank_dim))

        # data = {"rankdim":rank_dim_list, "uuas":test_uuas_list}
        # dump_pkl(data, "results/plots/ranks_{}.pkl".format(exp))
        



if __name__ == '__main__':
  argp = ArgumentParser()
 
  argp.add_argument('--seed', default=0, type=int)
  argp.add_argument('--exp', default="lstm")
  args = argp.parse_args()
  if args.seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
 
  execute_experiment(args.exp)
