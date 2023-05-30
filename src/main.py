"""
Usage :  python main.py --exp lstm --model linear --device cuda:0

"""

from argparse import ArgumentParser
from collections import defaultdict
import numpy as np
import torch
import pickle
import os

from lstm.model import RNNModel
from data import get_data
from train import train, get_best_model
from visualize import get_heatmaps
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertModel

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
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    return model, tokenizer

def load_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained('bert-base-cased')
    return model, tokenizer

def load_bertL():
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BertModel.from_pretrained('bert-large-cased')
    return model, tokenizer

def execute_experiment(exp, language, s_model, batch_size, layer_index, device):
    device = torch.device(device)
    
    if exp == "lstm":
        model, tokenizer = load_lstm()
        emb_dim = 650
    elif exp == "gpt":
        model, tokenizer = load_gpt()
        emb_dim = 768
    elif exp == "bert":
        model, tokenizer = load_bert()
        emb_dim = 768
    elif exp == "bertL":
        model, tokenizer = load_bertL()
        emb_dim = 1024
    else:
        print(f"{exp} IS NOT A SUPPORTED MODEL!!")

        
    ###################################################
    ###################################################
    train_probe = False # or False
    generate_visualization = False  # or False
    layer_index_probing = True
    ###################################################
    ###################################################
    

    loaders = get_data(model, tokenizer, language, exp, batch_size, layer_index, device)
    print("Data has been loaded.")

    if train_probe:
        print("Training the probe..")
        print("Language:", language)
        n_epochs = 100

        test_uuas = train(loaders, n_epochs, exp, language, emb_dim=emb_dim, model=s_model, device=device)
        
    if layer_index_probing:
        print("Language:", language)
        n_epochs = 200
        print(f"Probing {exp} at layer {layer_index}")
        s_model_list = ["poly", "rbf", "sigmoid"]
#         s_model_list = ["linear"]
        
        uuas_score_dict = {
            'model' : exp,
            'layer' : layer_index,
            'uuas_score' : [],
        }
        
        for s_model in s_model_list:
            print(f"\n\nTraining the {s_model} probe..")
            test_uuas = train(loaders, n_epochs, exp, language, emb_dim=emb_dim, model=s_model, layer_idx=layer_index, device=device)
            uuas_score_dict['uuas_score'].append({s_model : test_uuas})
            
        score_save_path = f"results/layer_probing/{exp}"
        os.makedirs(score_save_path, exist_ok=True)
        
        dump_pkl(uuas_score_dict, os.path.join(score_save_path, f"layer{layer_index}.pkl"))

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
    argp.add_argument('--layer_index', default=-1, type=int)
    argp.add_argument('--device', default="cuda:0")
    
    args = argp.parse_args()
    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    execute_experiment(args.exp, args.lang, args.model, args.batchsize, args.layer_index, args.device)
