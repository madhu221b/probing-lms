"""
1. Call print tree layer function in the main caller of this file.
2. Specify the experiment you are running - bert, bertL etc
3. Make sure you have your test reps in this folder results/data/layer_test_representation
4. Make sure you have reqd model pkls.  in result > models folder
5. The tree is generated in .txt file by layer. (make sure you always delete the older .txt files before
  creating new)
6. Copy paste the latex code in latex and generate the tree. 

"""

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
"7":(0.8,0.83,1),
"8":(0.7874,0.8187,1),
"9":(0.6925,0.7522,1),
"10":(0.6268,0.7039,1),
"11":(0.3277,0.5022,1),
"12":(0.3277,0.5022,1)
}

def get_color(value):
    color_tup = (0,0,1)
    if  0 <= value <= 1: # 0 to 6.5
        if 0 <= value < 0.25:
            color_tup = color_maps["1"]
        elif 0.25 <= value < 0.5: # 1 to 3
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
    
    idx = int([key for key, val in color_maps.items() if val == color_tup][0])
    return color_tup, idx

def load_pickle(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data
def dump_pkl(content, file_name):
    file = open(file_name, 'wb')
    pickle.dump(content, file)
    file.close()

def line_plot_model_rank_vs_uuas():
    # file_name = "./results/plots/ranks_gpt.pkl"
    folder_name = "./results/rank"
    for file_name in os.listdir(folder_name):
        print(file_name)
        data = load_pickle(os.path.join(folder_name,file_name))
        smodel = data["smodel"]
        if smodel == "linear":
            label, color  = "Linear", "#069AF3"
        elif smodel == "poly":
            label, color = "Polynomial","#DDA0DD"
        elif smodel == "rbf":
            label,color = "RBF","#15B01A"
        else:
            label,color = "Sigmoid","#FA8072"
        x, y = data["rankdim"], data["uuas"]
        plt.plot(x, y, label = label ,marker='o', color=color,linestyle="-")

    
   
    plt.xticks(x)
    plt.xlabel("Probe Maximum Rank (for BERT layer 12)")
    plt.ylabel("Test UUAS Score")
    plt.legend()
    # # plt.show()
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
    # print("Predicted edges", predicted_edges)
    for i_index, j_index, color in predicted_edges:
        # print(i_index, j_index)
        i_index += 1
        j_index += 1
        color_str = "color"+str(i_index)+str(j_index)
        
        if color != -999:
            
            color_tup, idx = get_color(color)
            # print("coming here to get colored", color_tup, idx)
            string_color = f"\\definecolor{{{color_str}}}{{rgb}}{{{color_tup[0],color_tup[1],color_tup[2]}}}\n".replace("(","").replace(")","")
            string += string_color
            # if idx > 6:
            color_p = "{}!60!, ultra thick".format(color_str)
            # else: 
            #        color_p = "{}!60!".format(color_str)
            string_edge  = "\\depedge[-,edge style={{{}}}, edge below]{{{}}}{{{}}}{{{}}}\n".format(color_p,i_index,j_index,".")
            string += string_edge
        else:
          print("COMES HERE")
          # string += f"\\definecolor{{{color_str}}}{{rgb}}{{{0,255- round((255*color),2),0}}}\n".replace("(","").replace(")","")
          # string += "\\depedge[-,edge style={}!60!, edge below]{{{}}}{{{}}}{{{}}}\n".format(color_str,i_index,j_index,".")
    string += "\\end{dependency}\n"
    return string

def get_tree_dets(probe, _data, i, length):
    probe.eval()    
    # we do this only for one sample
   #  _data = _data[0].unsqueeze(0)
    print(len(_data))
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
        # print("pred",pred_matrix[i][j], "gold: ",gold_matrix[i][j])
        if gold_matrix[i][j] == 0:
            ratio = -999
        else:
              ratio = pred_matrix[i][j]/gold_matrix[i][j]
        pred_edges_new.append((i,j,ratio.item()))
    return pred_edges_new, gold_edges

def generate_line_plots(path,exp):
    layer_index = []
    linear, sigmoid, rbf, poly = [], [], [], []
    for file_name in os.listdir(path):
        print(file_name)
        data = load_pickle(os.path.join(path, file_name))
        # print(data)
        layer_index.append(data["layer"])
       
        linear.append(data["uuas_score"][0]["linear"])
        poly.append(data["uuas_score"][1]["poly"])
        rbf.append(data["uuas_score"][2]["rbf"])
        sigmoid.append(data["uuas_score"][3]["sigmoid"])
 

    idxs = np.array(layer_index).argsort()   
    
    layer_index = np.array(layer_index)[idxs]
 
    plt.plot(layer_index, np.array(linear)[idxs], label ='Linear', marker="o",color="#069AF3")
    plt.plot(layer_index, np.array(poly)[idxs], label ='Polynomial', marker="o",color="#DDA0DD")
    plt.plot(layer_index, np.array(rbf)[idxs], label ='RBF', marker="o",color="#15B01A")
    plt.plot(layer_index, np.array(sigmoid)[idxs], label ='Sigmoid', marker="o",color="#FA8072") 
    
    plt.xticks(layer_index)
    plt.xlabel("Layer Index")
    plt.ylabel("Test UUAS Score")
    plt.legend()
    plt.savefig("./results/plots/layer_{}.png".format(exp))
    return layer_index, np.array(linear)[idxs], np.array(poly)[idxs], np.array(rbf)[idxs], np.array(sigmoid)[idxs]



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
        # i = -1
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

def print_tree_layer(exp, s_model, layer):
    try: 
        ## Generate tree ## 
        device = "cpu"
        data_file = "results/data/layer_test_representation/{}/{}_layer{}_english_test.pkl".format(exp,exp,layer)
        test_data_path = '../data/en_ewt-ud-test.conllu'
        if exp == "lstm":  emb_dim = 650
        elif exp == "gpt" or exp == "bert":  emb_dim = 768

        test_data = load_pickle(data_file)["test"]
        # print(test_data["test"])
        ud_parses = parse_corpus(test_data_path)
        i = -10
        sent = ud_parses[i]
        words = [_["form"] for _ in sent]
        probe = get_best_model(exp, "english", rank=64, emb_dim=emb_dim, s_model=s_model, layer=layer, device=torch.device(device))
        
        pred_edges, gold_edges = get_tree_dets(probe, test_data, i, len(words))
        # print("pred edges", pred_edges)
        # print("gold edges: ", gold_edges)
        # print(len(pred_edges), len(gold_edges), len(words))
        str = print_tikz_color(pred_edges, gold_edges, words)
        tree_path = "tree_{}.txt".format(s_model)
        print(str, file=open(tree_path, "a"))
        print("Tree has been dumped in {} file".format(tree_path))
        # print_tikz(pred_edges, gold_edges, words)
    except Exception as error:
        print("Error in visual file: ", error)

def plot_bert_and_bertL():
    
    exp = "bert"
    folder_path = "./results/layer_probing/{}/".format(exp)
    layeridx, linear, poly, rbf, sigmoid = generate_line_plots(folder_path,exp)

    exp2 = "bertL"
    folder_path = "./results/layer_probing/{}/".format(exp2)
    layeridx_L, linear_L, poly_L, rbf_L, sigmoid_L = generate_line_plots(folder_path,exp2)
    
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax2.grid(False)
    labels = ["Linear", "Polynomial", "RBF", "Sigmoid"]
    for label in labels:
        if label == "Linear":
            lineL = ax.plot(layeridx, linear,  marker="o",linestyle='dashed',color="#069AF3")
            lineLL = ax2.plot(layeridx_L, linear_L,marker="o",color=lineL[-1].get_color())
            
            # first_legend = ax.legend(handles=[lineL], loc='upper right')
            # ax.add_artist(first_legend)
            curves = lineL + lineLL
            labs = ["BERT", "BERTLarge"]
            ax.legend(curves, labs, loc="upper right",bbox_to_anchor=(1,1))
            # ax2.legend("Linear")
        elif label == "Polynomial":
            lineP = ax.plot(layeridx, poly,  marker="o",linestyle='dashed',color="#DDA0DD")
            lineLP = ax2.plot(layeridx_L, poly_L,marker="o",color=lineP[-1].get_color())
        elif label == "RBF":
            lineR = ax.plot(layeridx, rbf, marker="o",linestyle='dashed',color="#15B01A")
            lineRL = ax2.plot(layeridx_L, rbf_L, marker="o", color=lineR[-1].get_color())
        else:
            lineS = ax.plot(layeridx, sigmoid,linestyle='dashed',color="#FA8072")
            lineSL = ax2.plot(layeridx_L, sigmoid_L, marker="o",color=lineS[-1].get_color()) 

            
            # ax.legend(handles=[lineL[], loc='lower right')






      

    # lineL = ax.plot(layeridx, linear, label ='Linear', marker="o",linestyle='dashed')
    # lineP = ax.plot(layeridx, poly, label ='Polynomial', marker="o",linestyle='dashed')
    # lineR = ax.plot(layeridx, rbf, label ='RBF', marker="o",linestyle='dashed')
    # lineS = ax.plot(layeridx, sigmoid, label ='Sigmoid', marker="o",linestyle='dashed') 

    # lineLL = ax.plot(layeridx_L, linear_L, label ='Linear', marker="o",color=lineL[-1].get_color())
    # linePL = ax.plot(layeridx_L, poly_L, label ='Polynomial', marker="o",color=lineP[-1].get_color())
    # lineRL = ax.plot(layeridx_L, rbf_L, label ='RBF', marker="o", color=lineR[-1].get_color())
    # lineSL = ax.plot(layeridx_L, sigmoid_L, label ='Sigmoid', marker="o",color=lineS[-1].get_color()) 
    
    # ax.xticks(layeridx_L)
    ax.set_xticks(layeridx_L) 
    ax.set_xticklabels(layeridx_L, fontsize=12)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Test UUAS Score")
    # ax.legend()
    
    # ax.legend(handles=[lineLL])
    ax2.legend(labels, loc = "lower right",bbox_to_anchor=(0.5,0))
    fig.savefig("./results/plots/layer_{}_{}.png".format(exp,exp2))

if __name__ == '__main__':
    # line_plot_model_rank_vs_uuas()
     
    # exp = "bert" 
    # folder_path = "./results/layer_probing/{}/".format(exp)
    # generate_line_plots(folder_path, exp)
    # plot_bert_and_bertL()

    exp = "bert"
    layer = 12
    # # ["linear","poly", "rbf", "sigmoid"]
    for s_model in ["poly","sigmoid"]:
        print_tree_layer(exp, s_model, layer)

    

    

    






