import torch
import torch.nn as nn

from typing import List
from conllu import parse_incr, TokenList
from torch import Tensor

from utils import lstm_utils, tree_utils, gpt_utils

dict_= {"english":"en_ewt",
        "tamil": "ta_ttb"
        }


def parse_corpus(filename: str) -> List[TokenList]:
    data_file = open(filename, encoding="utf-8")
    ud_parses = list(parse_incr(data_file)) 
    return ud_parses


def fetch_sen_reps(ud_parses: List[TokenList], model, tokenizer, concat) -> Tensor:    
    rep = []
    print(model)
    if "GPT2LMHeadModel" in str(model):
        for ud_parse in ud_parses:
            rep.append(gpt_utils.get_gpt_representations([ud_parse], model, tokenizer))
        if concat:
            rep = nn.utils.rnn.pad_sequence(rep, batch_first=True)
        return rep
    else:
        for ud_parse in ud_parses:
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

def get_data(model, tokenizer,language="english"):
    TRAIN_DATA_PATH = '../data/sample/{}-ud-train.conllu'.format(dict_[language])
    DEV_DATA_PATH = '../data/sample/{}-ud-dev.conllu'.format(dict_[language])
    TEST_DATA_PATH = '../data/sample/{}-ud-test.conllu'.format(dict_[language])

    train_data = init_corpus(TRAIN_DATA_PATH, model, tokenizer, concat=True)
    dev_data = init_corpus(DEV_DATA_PATH, model, tokenizer, concat=True)
    test_data = init_corpus(TEST_DATA_PATH, model, tokenizer, concat=True)
    return {"train":train_data, "dev":dev_data, "test":test_data}

