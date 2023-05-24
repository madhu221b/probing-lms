import torch
from torch import optim
from tqdm import tqdm
import sys
from models import StructuralProbe, PolynomialProbe, RbfProbe, SigmoidProbe
from loss import L1DistanceLoss
from utils.tree_utils import calc_uuas

def evaluate_probe(probe, loss_function, _data):
    probe.eval()
    y, x, sent_lens = _data
    preds = probe(x)
    loss_score, _ = loss_function(preds, y, sent_lens)
    
    preds_new, y_new = [], []
    for i, length in enumerate(sent_lens):
        length = int(length)
        preds_resized, y_resized = preds[i, :length, :length], y[i, :length, :length]
        preds_new.append(preds_resized)
        y_new.append(y_resized)
 
    uuas_score = calc_uuas(preds_new, y_new)
    return loss_score, uuas_score

def train(_data, _dev_data, _test_data, epochs, experiment_name, language, rank_dim=64,emb_dim=650,model="linear"):
    emb_dim = emb_dim
    rank = rank_dim
    lr = 10e-4
    batch_size = 11
    model_file_path = "results/models/model_{}_{}_{}_{}.pt".format(experiment_name,rank_dim,language,model)
    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    print("Model used: {}".format(model))
    if model == "linear":
        print("Using Structural Probe")
        probe = StructuralProbe(emb_dim, rank)
    elif model == "poly":
        print("Using Polynomial Probe")
        probe = PolynomialProbe(emb_dim, rank)
    elif model == "rbf":
        print("Using Rbf Probe")
        probe = RbfProbe(emb_dim, rank)
    elif model == "sigmoid":
        print("Using Sigmoid Probe")
        probe = SigmoidProbe(emb_dim, rank)

    optimizer = optim.Adam(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
    loss_function =  L1DistanceLoss()

    train_y, train_x, train_sent_lens = _data
    # print("train y: ", train_y.size(), train_x.size(), train_sent_lens.size())
    len_batch = train_y.size(0)
    epochs = 40
    for epoch_index in range(epochs):
        epoch_train_loss, epoch_dev_loss = 0,0
        epoch_train_epoch_count,epoch_dev_epoch_count = 0, 0
        epoch_train_loss_count, epoch_dev_loss_count = 0, 0

        for i in range(0, len_batch, batch_size):
            probe.train()
            optimizer.zero_grad()

            _train_batch = train_x[i:i+batch_size]
            _train_labels = train_y[i:i+batch_size]
            _train_lengths = train_sent_lens[i:i+batch_size]
            _preds = probe(_train_batch)
            
            batch_loss, count = loss_function(_preds, _train_labels, _train_lengths)

            epoch_train_loss += (batch_loss*count)
            epoch_train_epoch_count += 1
            epoch_train_loss_count += count 

            batch_loss.backward()
            optimizer.step()
        
        for i in range(0, len_batch, batch_size):
            optimizer.zero_grad()
            probe.eval()

            _train_batch = train_x[i:i+batch_size]
            _train_labels = train_y[i:i+batch_size]
            _train_lengths = train_sent_lens[i:i+batch_size]
            _preds = probe(_train_batch)
            batch_loss, count = loss_function(_preds, _train_labels, _train_lengths)

            epoch_dev_loss += (batch_loss*count)
            epoch_dev_epoch_count += 1
            epoch_dev_loss_count += count 
            
        scheduler.step(epoch_dev_loss)
        _, dev_uuas = evaluate_probe(probe, loss_function, _dev_data)
        tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}, Dev uuas: {}'.format(epoch_index, epoch_train_loss/epoch_train_loss_count,
                  epoch_dev_loss/epoch_dev_loss_count, dev_uuas))
        if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.0001:
            torch.save(probe.state_dict(),model_file_path)
            min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
            min_dev_loss_epoch = epoch_index
            tqdm.write('Saving probe parameters')
        elif min_dev_loss_epoch < epoch_index - 4:
            tqdm.write('Early stopping')
            break

        

    best_probe = StructuralProbe(emb_dim, rank)
    best_probe.load_state_dict(torch.load(model_file_path, map_location="cpu")) 
    test_loss, test_uuas = evaluate_probe(best_probe,loss_function,_test_data)
    print("Test Loss: {}, Test uuas: {}".format(test_loss, test_uuas))
    return round(test_uuas*100.0, 2)

def get_best_model(exp,language,rank=64,emb_dim=650,model="linear"):
    emb_dim = emb_dim
    model_file_path = "results/models/model_{}_{}__{}_{}.pt".format(exp,rank,model,language)
    probe = StructuralProbe(emb_dim, rank)
    try:
        probe.load_state_dict(torch.load(model_file_path, map_location="cpu")) 
        print("Model has been loaded.")
        return probe
    except Exception as error:
        print("Error while loading model: ", error)
        return None
