import torch
from torch import optim
from tqdm import tqdm
import sys
from models import StructuralProbe, PolynomialProbe, RbfProbe, SigmoidProbe
from loss import L1DistanceLoss
from utils.tree_utils import calc_uuas


def evaluate_probe(probe, loss_function, data_loader):
    probe.eval()
    
    epoch_loss = 0
    epoch_count = 0
    epoch_loss_count = 0
    epoch_uuas = 0
    
    for eval_batch in tqdm(data_loader):
        eval_x = eval_batch[0]
        eval_y = eval_batch[1]
        eval_lens = eval_batch[2]

        preds = probe(eval_x)

        batch_loss, count = loss_function(preds, eval_y, eval_lens)

        epoch_loss += (batch_loss.item()*count)
        epoch_count += 1
        epoch_loss_count += count
    
        preds_new, y_new = [], []
        
        for i, length in enumerate(eval_lens):
            length = int(length)
            preds_resized, y_resized = preds[i, :length, :length], eval_y[i, :length, :length]
            preds_new.append(preds_resized)
            y_new.append(y_resized)

        epoch_uuas += (calc_uuas(preds_new, y_new)*count)
        
        del preds, batch_loss
    
    uuas_score = epoch_uuas/epoch_loss_count
    loss_score = epoch_loss/epoch_loss_count
    
    return loss_score, uuas_score


def train(
    loaders, 
    epochs, 
    experiment_name, 
    language, 
    rank=64, 
    emb_dim=650, 
    model="linear",
    lr=10e-4,
    device=None,
):
    
    train_loader, dev_loader, test_loader = loaders
    model_file_path = "results/models/model_{}_{}_{}_{}.pt".format(experiment_name, rank, language, model)
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

    probe = probe.to(device)
    
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    loss_function =  L1DistanceLoss()
    
    for epoch_index in range(epochs):
        probe.train()
        
        epoch_train_loss = 0
        epoch_train_epoch_count = 0
        epoch_train_loss_count = 0

#         for train_batch in tqdm(train_loader):
        with tqdm(train_loader, unit="batch") as tepoch:
            for train_batch in tepoch:
                
                tepoch.set_description(f"Epoch {epoch_index}")
                optimizer.zero_grad()

                train_x = train_batch[0]
                train_y = train_batch[1]
                train_lens = train_batch[2]

                preds = probe(train_x)

                batch_loss, count = loss_function(preds, train_y, train_lens)

                epoch_train_loss += (batch_loss.item()*count)
                epoch_train_epoch_count += 1
                epoch_train_loss_count += count 

                batch_loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(train_loss=batch_loss.item(), lr=optimizer.param_groups[0]["lr"])
                
                del preds, batch_loss
            
        epoch_train_loss = epoch_train_loss/epoch_train_loss_count
        dev_loss, dev_uuas = evaluate_probe(probe, loss_function, dev_loader)
        scheduler.step(dev_loss)

        tqdm.write('Epoch: {}, Train loss: {}, Dev loss: {}, Dev uuas: {}'.format(epoch_index, epoch_train_loss, dev_loss, dev_uuas))
        
        if dev_loss < min_dev_loss - 0.0001:
            torch.save(probe.state_dict(), model_file_path)
            min_dev_loss = dev_loss
            min_dev_loss_epoch = epoch_index
            tqdm.write('Saving probe parameters')
            
        elif min_dev_loss_epoch < epoch_index - 4:
            tqdm.write('Early stopping')
            break

    best_probe = StructuralProbe(emb_dim, rank)
    best_probe.load_state_dict(torch.load(model_file_path, map_location='cpu'))
    
    best_probe = best_probe.to(device)
    
    test_loss, test_uuas = evaluate_probe(best_probe, loss_function, test_loader)
    print("Test Loss: {}, Test uuas: {}".format(test_loss, test_uuas))
    
    return round(test_uuas.item()*100.0, 2)


def get_best_model(exp, language, rank=64, emb_dim=650, model="linear", device=None):
    emb_dim = emb_dim
    model_file_path = "results/models/model_{}_{}__{}_{}.pt".format(exp,rank,model,language)
    probe = StructuralProbe(emb_dim, rank)
    try:
        probe.load_state_dict(torch.load(model_file_path, map_location="cpu"))
        probe = probe.to(device)
        print("Model has been loaded.")
        return probe
    except Exception as error:
        print("Error while loading model: ", error)
        return None
