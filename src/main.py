"""
Usage :  python main.py --exp lstm --model linear --device cuda:0

"""

from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import torch
import pickle

from lstm.model import RNNModel
from data import get_data
from train import train, get_best_model
from visualize import get_heatmaps
from transformers import *

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

def load_gpt():
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    return model, tokenizer


def execute_experiment(exp, language, s_model, batch_size, device):
    device = torch.device(device)
    
    if exp == "lstm":
        model, tokenizer = load_lstm()
    elif exp == "gpt":
        model, tokenizer = load_gpt()

    train_probe = True # or False
    generate_visualization = False  # or False

    loaders = get_data(model, tokenizer, language, exp, batch_size, device)
    print("Data has been loaded.")

    if train_probe:
        print("Training the probe..")
        print("Language:", language)
        n_epochs = 40
        emb_dim = 650
        if exp == "gpt":
            emb_dim = 768
        test_uuas = train(loaders, n_epochs, exp, language, emb_dim=emb_dim, model=s_model, device=device)

    if generate_visualization:
        best_model = get_best_model(exp, language, emb_dim=emb_dim, model=s_model, device=device)
        if best_model:
            get_heatmaps(best_model, data["test"], language)
            # fig2(best_model, data["test"], language)
            

        
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
    argp.add_argument('--lang', default="english")
    argp.add_argument('--model', default="linear")
    argp.add_argument('--batchsize', default=32, type=int)
    argp.add_argument('--device', default="cuda:0")
    
    args = argp.parse_args()
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    execute_experiment(args.exp, args.lang, args.model, args.batchsize, args.device)
