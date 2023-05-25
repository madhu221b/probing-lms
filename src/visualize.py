import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from data import parse_corpus, dict_
from utils.tree_utils import create_gold_distances, create_mst, edges
from train import get_best_model

import seaborn as sns
sns.set(style="darkgrid")

def load_pickle(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def line_plot_model_rank_vs_uuas():
    file_name = "./results/plots/ranks_gpt.pkl"
    data = load_pickle(file_name)
    x, y = data["rankdim"], data["uuas"]
    plt.plot(x, y, label = "GPT",marker='o',linestyle="-")
    # plt.xticks(x)
    plt.xlabel("Probe Maximum Rank")
    plt.ylabel("UUAS")
    plt.legend()
    # plt.show()
    plt.savefig("./results/plots/rank.png")
       

def get_heatmaps(model, data, language):
    labels, test_x, test_sent_lens = data
    TEST_DATA_PATH = '../data/sample/{}-ud-test.conllu'.format(dict_[language])
    corpus = parse_corpus(TEST_DATA_PATH)[:]
    print(labels.size(),test_x.size(), test_sent_lens.size(), len(corpus))
    test_predictions = model(test_x)
    i = 0
    for test_prediction, length, text, label in zip(test_predictions, test_sent_lens, corpus, labels):
        length = int(length)
        prediction = test_prediction[:length,:length].cpu().detach().numpy()
        label = label[:length,:length].cpu()
        words = [_["form"] for _ in text ]
        fontsize = 5*( 1 + np.sqrt(len(words))/200)
        plt.clf()
        ax = sns.heatmap(label)
        ax.set_title('Gold Parse Distance')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='top')
        plt.tight_layout()
        plt.savefig("results/plots/gold_heatmaps/heatmap_{}.png".format(i), dpi=300)

        plt.clf()
        ax = sns.heatmap(prediction)
        ax.set_title('Predicted Parse Distance (squared)')
        ax.set_xticks(np.arange(len(words)))
        ax.set_yticks(np.arange(len(words)))
        ax.set_xticklabels(words, rotation=90, fontsize=fontsize, ha='center')
        ax.set_yticklabels(words, rotation=0, fontsize=fontsize, va='center')
        plt.tight_layout()
        plt.savefig("results/plots/predict_heatmaps/heatmap_{}.png".format(i), dpi=300)
        i += 1
    print("Generated all Visualizations of heat map")

def print_tikz(predicted_edges, gold_edges, words):
    """ Turns edge sets on word (nodes) into tikz dependency LaTeX.
    Parameters
    ----------
    predicted_edges : Set[Tuple[int, int]]
        Set (or list) of edge tuples, as predicted by your probe.
    gold_edges : Set[Tuple[int, int]]
        Set (or list) of gold edge tuples, as obtained from the treebank.
    words : List[str]
        List of strings representing the tokens in the sentence.
    """

    string = """\\begin{dependency}[hide label, edge unit distance=.5ex]
    \\begin{deptext}[column sep=0.05cm]
    """

    string += (
        "\\& ".join([x.replace("$", "\$").replace("&", "+") for x in words])
        + " \\\\\n"
    )
    string += "\\end{deptext}" + "\n"
    for i_index, j_index in gold_edges:
        string += "\\depedge[-]{{{}}}{{{}}}{{{}}}\n".format(i_index, j_index, ".")
    for i_index, j_index in predicted_edges:
        string += f"\\depedge[-,edge style={{red!60!}}, edge below]{{{i_index}}}{{{j_index}}}{{.}}\n"
    string += "\\end{dependency}\n"
    print(string, file=open("tree.txt", "a"))
    print("Tree has been dumped in tree.txt file")

def get_tree_dets(probe, _data, i, length):
    probe.eval()    
    # we do this only for one sample
   #  _data = _data[0].unsqueeze(0)
    y, x, sent_lens = _data
    y, x, sent_lens = y[i].unsqueeze(0), x[i].unsqueeze(0), sent_lens[i].unsqueeze(0)
    preds = probe(x)
    preds_new, y_new = [], []
 
    preds_resized, y_resized = preds[0, :length, :length], y[0, :length, :length]
    preds_new.append(preds_resized)
    y_new.append(y_resized)
 
    pred_edges, gold_edges = get_pred_gold_edges(preds_new, y_new)
    return pred_edges, gold_edges

def get_pred_gold_edges(pred_distances, gold_distances):    
    for pred_matrix, gold_matrix in zip(pred_distances, gold_distances):     
        pred_mst = create_mst(pred_matrix.to(torch.device('cpu')))
        gold_mst = create_mst(gold_matrix.to(torch.device('cpu')))
        pred_edges = edges(pred_mst)
        gold_edges = edges(gold_mst)
    return pred_edges, gold_edges


# def fig2(model, data, language):
#     i = 0
#     labels, test_x, test_sent_lens = data
#     TEST_DATA_PATH = '../data/sample/{}-ud-test.conllu'.format(dict_[language])
#     corpus = parse_corpus(TEST_DATA_PATH)[:]
#     test_predictions = model(test_x)
#     i = 0
#     for test_prediction, length, text, label in zip(test_predictions, test_sent_lens, corpus, labels):
#         plt.clf()
#         length = int(length)
#         prediction = test_prediction[:length,:length].cpu().detach().numpy()
#         label = label[:length,:length].cpu()
#         words = [_["form"] for _ in text ]
#         fontsize = 6
#         cumdist = 0
#         for index, (word, gold, pred) in enumerate(zip(words, label, prediction)):
#           print(type(gold))
#           plt.text(cumdist*3, gold*2, word, fontsize=fontsize, ha='center')
#           # plt.text(cumdist*3, pred*2, word, fontsize=fontsize, color='red', ha='center')
#           cumdist = cumdist + (np.square(len(word)) + 1)
        
     
#         plt.ylim(0,20)
#         plt.xlim(0,cumdist*3.5)
#         plt.title('Dependency Parse Tree Depth Prediction', fontsize=10)
#         # plt.ylabel('Tree Depth', fontsize=10)
#         # plt.xlabel('Linear Absolute Position',fontsize=10)
#         # plt.tight_layout()
#         # plt.xticks(fontsize=5)
#         # plt.yticks(fontsize=5)
#         plt.savefig("results/depth.png", dpi=300)
#         i += 1
#     print("Depth")



if __name__ == '__main__':
    # line_plot_model_rank_vs_uuas()
    try: 
        ## Generate tree ## 
        device = "cpu"
        exp = "lstm" # Mention model name
        s_model = "rbf" # Mention model: linear, rbf, poly ,sigmoid
        data_file = "results/data/{}_english.pkl".format(exp)
        test_data_path = '../data/en_ewt-ud-test.conllu'
        if exp == "lstm":  emb_dim = 650
        elif exp == "gpt" or exp == "bert":  emb_dim = 768

        test_data = load_pickle(data_file)["test"]
        index = 7
        ud_parses = parse_corpus(test_data_path)
        i = -9
        sent = ud_parses[i]
        words = [_["form"] for _ in sent]
        probe = get_best_model(exp, "english", rank=64, emb_dim=emb_dim, model=s_model, device=torch.device(device))
        pred_edges, gold_edges = get_tree_dets(probe, test_data, i, len(words))
        # print("pred edges", pred_edges)
        # print("gold edges: ", gold_edges)
        # print(len(pred_edges), len(gold_edges), len(words))
        print_tikz(pred_edges, gold_edges, words)
    except Exception as error:
        print("Error in visual file: ", error)


    






