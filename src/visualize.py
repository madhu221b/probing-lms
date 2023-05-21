import pickle
import matplotlib.pyplot as plt
import numpy as np
from data import parse_corpus, dict_

import seaborn as sns
sns.set(style="darkgrid")

def load_pickle(file_name):
    file = open(file_name, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def line_plot_model_rank_vs_uuas():
    file_name = "./results/plots/ranks_lstm.pkl"
    data = load_pickle(file_name)
    x, y = data["rankdim"], data["uuas"]
    plt.plot(x, y, label = "Colorful RNN",marker='o',linestyle="-")
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
    line_plot_model_rank_vs_uuas()

    






