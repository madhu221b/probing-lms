import os
import torch
import torch.nn as nn
import pickle
from typing import List
from conllu import parse_incr, TokenList
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import lstm_utils, tree_utils, gpt_utils

dict_= {"english":"en_ewt",
        "tamil": "ta_ttb"
        }

def load_pickle(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def dump_pkl(content, file_name):
    file = open(file_name, 'wb')
    pickle.dump(content, file)
    file.close()

def parse_corpus(filename: str) -> List[TokenList]:
    data_file = open(filename, encoding="utf-8")
    ud_parses = list(parse_incr(data_file)) 
    return ud_parses


def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer, concat) -> Tensor:    
    rep = []
    # print(model)
    if "GPT2LMHeadModel" in str(model):
        for ud_parse in tqdm(ud_parses):
            rep.append(gpt_utils.get_gpt_representations([ud_parse], model, tokenizer))
        if concat:
            rep = nn.utils.rnn.pad_sequence(rep, batch_first=True)
        return rep
    else:
        for ud_parse in tqdm(ud_parses):
            rep.append(lstm_utils.get_lstm_representations([ud_parse], model, tokenizer))
        if concat:
            rep = nn.utils.rnn.pad_sequence(rep, batch_first=True)
        return rep


def init_corpus(path, model, tokenizer, concat=False, cutoff=None):
    """ Initialises the data of a corpus.
    
    Parameters
    ----------
    path : str
        Path to corpus location
    concat : bool, optional
        Optional toggle to concatenate all the tensors
        returned by `fetch_sen_reps`.
    cutoff : int, optional
        Optional integer to "cutoff" the data in the corpus.
        This allows only a subset to be used, alleviating 
        memory usage.
    """
    corpus = parse_corpus(path)[:cutoff]

    embs = fetch_sen_reps(corpus, model, tokenizer, concat=concat)    
    gold_distances = tree_utils.create_gold_distances(corpus)

    
    lengths = [sent.size(0) for sent in gold_distances]
    # lengths = [len(sent) for sent in corpus]
    maxlen = int(max(lengths))
    # print("max len: ", maxlen)
    label_maxshape = [maxlen for _ in gold_distances[0].shape]
    labels = [-torch.ones(*label_maxshape) for _ in range(len(lengths))]

    for idx, gold_dist in enumerate(gold_distances):
        length = lengths[idx]
        # print("labels idx", labels[idx].size())
        labels[idx][:length,:length] = gold_dist
    
    labels = torch.stack(labels)
    print("labels size in data: ", labels.size(), embs.size())
    return labels, embs, torch.Tensor(lengths)


class TokenRepresentations(Dataset):
    def __init__(self, data, device):
        super(Dataset).__init__()
        self.embeddings = data[1].to(device)
        self.labels = data[0].to(device)
        self.sentence_lengths = data[2].to(device)
        self.total_data = self.embeddings.size(0)
        
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.sentence_lengths[idx]
    
    def __len__(self):
        return self.total_data


def get_data(model, tokenizer, language="english", exp="lstm", batch_size=64, device=None):
#     TRAIN_DATA_PATH = '../data/sample/{}-ud-train.conllu'.format(dict_[language])
#     DEV_DATA_PATH = '../data/sample/{}-ud-dev.conllu'.format(dict_[language])
#     TEST_DATA_PATH = '../data/sample/{}-ud-test.conllu'.format(dict_[language])

    TRAIN_DATA_PATH = '../data/{}-ud-train.conllu'.format(dict_[language])
    DEV_DATA_PATH = '../data/{}-ud-dev.conllu'.format(dict_[language])
    TEST_DATA_PATH = '../data/{}-ud-test.conllu'.format(dict_[language])
    DATA_PATH = 'results/data/{}_{}.pkl'.format(exp, language)
    
    if os.path.exists(DATA_PATH):
        data = load_pickle(DATA_PATH)
        print("Loading data from path: {}".format(DATA_PATH))
    else:
        print("Generating representations...")
        train_data = init_corpus(TRAIN_DATA_PATH, model, tokenizer, concat=True)
        dev_data = init_corpus(DEV_DATA_PATH, model, tokenizer, concat=True)
        test_data = init_corpus(TEST_DATA_PATH, model, tokenizer, concat=True)
        data = {"train":train_data, "dev":dev_data, "test":test_data}
        dump_pkl(data, DATA_PATH )
        print("Data dumped in path: {}".format(DATA_PATH))

    loaders = []
    
    train_dataset = TokenRepresentations(data["train"], device=device)
    loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
    
    dev_dataset = TokenRepresentations(data["dev"], device=device)
    loaders.append(DataLoader(dev_dataset, batch_size=batch_size, shuffle=True))
    
    test_dataset = TokenRepresentations(data["test"], device=device)
    loaders.append(DataLoader(test_dataset, batch_size=batch_size, shuffle=True))
    
    return loaders

