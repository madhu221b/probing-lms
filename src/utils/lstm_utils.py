import torch

def get_lstm_representations(ud_parses, model, tokenizer):
        with torch.no_grad():
            model.eval()
            hidden = model.init_hidden(1)
            arr = []
            for sent in ud_parses:
                for token in sent:
                    form = token["form"]
                    if form in tokenizer:
                        arr.append(tokenizer[form])
                    else:
                        arr.append(tokenizer["<unk>"])
            ids = torch.tensor(arr)
            rep = model(ids.unsqueeze(0), hidden)
            rep = rep.squeeze(0)
            return rep