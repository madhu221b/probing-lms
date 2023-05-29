import torch
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np
from data import parse_corpus, dict_
from utils.tree_utils import create_gold_distances, create_mst, edges
from train import get_best_model

import seaborn as sns
sns.set(style="darkgrid")


color_maps = {
  "0": (0,0,0),
"1":(1,0.0337,0),
"2":(1,0.2647,0.0033),
"3":(1,0.4870,0.1411),
"4":(1,0.6636,0.3583),
"5":(1,0.7992,0.6045),
"6":(1,0.9019,0.8473),
"6.5":(1,1,1),
"7":(0.9337,0.9150,1),
"8":(0.7874,0.8187,1),
"9":(0.6925,0.7522,1),
"10":(0.6268,0.7039,1),
"11":(0.3277,0.5022,1),
"12":(0.3277,0.5022,1)
}

def get_color(value):
    color_tup = (0,0,1)
    if  0 <= value <= 1: # 0 to 6.5
        if 0.25 <= value < 0.5: # 1 to 3
          mid = (0.5+0.25)/2
          if 0 <= value < mid: color_tup = color_maps["1"]
          elif value == mid:  color_tup = color_maps["2"]
          else: color_tup = color_maps["3"]
        elif 0.5 <= value < 1: # 4 to 6
          mid = (0.5+1)/2
          if 0.5 <= value < mid: color_tup = color_maps["4"]
          elif value == mid:  color_tup = color_maps["5"]
          else: color_tup = color_maps["6"]
        elif value == 1:
           color_tup = color_maps["6.5"]
    elif 1 < value <= 2: # 7 to 9
        mid = (1+2)/2
        if 1 < value <= mid: color_tup = color_maps["7"]
        elif value == mid: color_tup = color_maps["8"]
        else: color_tup = color_maps["9"]
    elif 2 < value <= 4:
        mid = (2+4)/2
        if 2 < value <= mid: color_tup = color_maps["10"]
        elif value == mid: color_tup = color_maps["11"]
        else: color_tup = color_maps["12"]
    else:
        color_tup = color_maps["12"]
    
    return color_tup

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
    for i_index, j_index,_ in predicted_edges:
        string += f"\\depedge[-,edge style={{red!60!}}, edge below]{{{i_index}}}{{{j_index}}}{{.}}\n"
    string += "\\end{dependency}\n"
    print(string, file=open("tree.txt", "a"))
    print("Tree has been dumped in tree.txt file")

def print_tikz_color(predicted_edges, gold_edges, words):
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
    print(gold_edges)
    for i_index, j_index in gold_edges:
        i_index += 1
        j_index += 1
        string += "\\depedge[-]{{{}}}{{{}}}{{{}}}\n".format(i_index, j_index, ".")
    print(predicted_edges)
    for i_index, j_index, color in predicted_edges:
        i_index += 1
        j_index += 1
        color_str = "color"+str(i_index)+str(j_index)
        
        if color != -999:
            
            color_tup = get_color(color)
            print("coming here to get colored", color_tup)
            string_color = f"\\definecolor{{{color_str}}}{{rgb}}{{{color_tup[0],color_tup[1],color_tup[2]}}}\n".replace("(","").replace(")","")
            string += string_color
            # color_p = "{}!60!, ultra thick".format(color_str)
            color_p = "{}!60!".format(color_str)
            string_edge  = "\\depedge[-,edge style={{{}}}, edge below]{{{}}}{{{}}}{{{}}}\n".format(color_p,i_index,j_index,".")
             
            string += string_edge
        elif 0 <= color <= 0.2: 
            string_color = f"\\definecolor{{{color_str}}}{{rgb}}{{{255- round((255*color),2),0,0}}}\n".replace("(","").replace(")","")
            string += string_color
            color_p = "{}!60!, densely dotted".format(color_str)
            string_edge  = "\\depedge[-,edge style={{{}}}, edge below]{{{}}}{{{}}}{{{}}}\n".format(color_p,i_index,j_index,".") 
            string += string_edge
        else:
          string += f"\\definecolor{{{color_str}}}{{rgb}}{{{0,255- round((255*color),2),0}}}\n".replace("(","").replace(")","")
          string += "\\depedge[-,edge style={}!60!, edge below]{{{}}}{{{}}}{{{}}}\n".format(color_str,i_index,j_index,".")
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
 
    pred_edges, gold_edges = get_pred_gold_edges_v2(preds_new, y_new)
    return pred_edges, gold_edges

def get_pred_gold_edges(pred_distances, gold_distances):    
    pred_matrix, gold_matrix = pred_distances[0], gold_distances[0]
    pred_mst = create_mst(pred_matrix.to(torch.device('cpu')))
    gold_mst = create_mst(gold_matrix.to(torch.device('cpu')))
    diff = torch.abs(gold_matrix- pred_matrix)
    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)
    diff_scores = [diff[i][j].item() for (i,j) in pred_edges]
    diff_n = [(x - min(diff_scores)) / (max(diff_scores) - min(diff_scores)) for x in diff_scores]
    pred_edges_new  =  [(i,j,round(score,2))  for (i,j),score in zip(pred_edges, diff_n)]
    return pred_edges_new, gold_edges

def get_pred_gold_edges_v2(pred_distances, gold_distances):    
    pred_matrix, gold_matrix = pred_distances[0], gold_distances[0]
    pred_mst = create_mst(pred_matrix.to(torch.device('cpu')))
    gold_mst = create_mst(gold_matrix.to(torch.device('cpu')))

    pred_edges = edges(pred_mst)
    gold_edges = edges(gold_mst)
    pred_edges_new = []
    for i, j in pred_edges:
        if gold_matrix[i][j] == 0:
            ratio = -999
        else:
              ratio = pred_matrix[i][j]/gold_matrix[i][j]
        pred_edges_new.append((i,j,ratio.item()))
    return pred_edges_new, gold_edges

def generate_line_plots(path):
    layer_index = []
    linear, sigmoid, rbf, poly = [], [], [], []
    for file_name in os.listdir(path):
        
        data = load_pickle(os.path.join(path, file_name))
  
        layer_index.append(data["layer"])
      
        linear.append(data["uuas_score"][0]["linear"])
        poly.append(data["uuas_score"][1]["poly"])
        rbf.append(data["uuas_score"][2]["rbf"])
        sigmoid.append(data["uuas_score"][3]["sigmoid"])
 

    idxs = np.array(layer_index).argsort()   
    layer_index = np.array(layer_index)[idxs]
    linear =  np.array(linear)[idxs]
    plt.plot(layer_index, np.array(linear)[idxs], label ='Linear', marker="o")
    plt.plot(layer_index, np.array(poly)[idxs], label ='Polynomial', marker="o")
    plt.plot(layer_index, np.array(rbf)[idxs], label ='RBF', marker="o")
    plt.plot(layer_index, np.array(sigmoid)[idxs], label ='Sigmoid', marker="o") 
    
    plt.xticks(layer_index)
    plt.xlabel("Layer Index")
    plt.ylabel("Test UUAS Score")
    plt.legend()
    plt.savefig("./results/plots/layer_gpt.png")





def print_tree():
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
        ud_parses = parse_corpus(test_data_path)
        i = -10
        sent = ud_parses[i]
        words = [_["form"] for _ in sent]
        probe = get_best_model(exp, "english", rank=64, emb_dim=emb_dim, model=s_model, device=torch.device(device))
        pred_edges, gold_edges = get_tree_dets(probe, test_data, i, len(words))
        # print("pred edges", pred_edges)
        # print("gold edges: ", gold_edges)
        # print(len(pred_edges), len(gold_edges), len(words))
        print_tikz_color(pred_edges, gold_edges, words)
        # print_tikz(pred_edges, gold_edges, words)
    except Exception as error:
        print("Error in visual file: ", error)

if __name__ == '__main__':
    # line_plot_model_rank_vs_uuas()

    folder_path = "./results/layer_probing/gpt/"
    generate_line_plots(folder_path)

    # print_tree()

    






