import pickle
import matplotlib.pyplot as plt

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
       

if __name__ == '__main__':
    line_plot_model_rank_vs_uuas()

    






