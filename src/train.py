from torch import optim

from models import StructuralProbe
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

def train(_data, _dev_data, _test_data, epochs):
    emb_dim = 650
    rank = 64
    lr = 10e-4
    batch_size = 15

    probe = StructuralProbe(emb_dim, rank)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,patience=1)
    loss_function =  L1DistanceLoss()

    train_y, train_x, train_sent_lens = _data
    len_batch = train_y.size(0)

    for epoch in range(epochs):

        for i in range(0, len_batch, batch_size):
            probe.train()
            optimizer.zero_grad()

            _train_batch = train_x[i:i+batch_size]
            _train_labels = train_y[i:i+batch_size]
            _train_lengths = train_sent_lens[i:i+batch_size]
            _preds = probe(_train_batch)
            batch_loss, total_sents = loss_function(_preds, _train_labels, _train_lengths)
            batch_loss.backward()
            optimizer.step()

        dev_loss, dev_uuas = evaluate_probe(probe, loss_function, _dev_data)
        print("Dev Loss: {}, Dev uuas: {}".format(dev_loss, dev_uuas))
        # Using a scheduler is up to you, and might require some hyper param fine-tuning
        scheduler.step(dev_loss)

    test_loss, test_uuas = evaluate_probe(probe,loss_function,_test_data)
    print("Test Loss: {}, Test uuas: {}".format(test_loss, test_uuas))