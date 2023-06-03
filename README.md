# Probing Language Models

Repository for the NLP2 assignment on probing language models.

All the required pkl model files can be found here: https://amsuni-my.sharepoint.com/:f:/r/personal/madhura_pawar_student_uva_nl/Documents/Probing?csf=1&web=1&e=SDNzKy. Please  request for access. 
## Main Caller function to train the probes 

The experiement name values can be: lstm, bert, bertL, gpt

(The state_dict.pt for Gulordava LSTM model can be found here: https://drive.google.com/file/d/19Lp3AM4NEPycp_IBgoHfLc_V456pmUom/view?usp=sharing)

And the layer index values can be 1 to 12 for gpt and bert and 1 to 24 for bertL.

### Run single probe training
```bash
python src/main.py  --exp "experiment name" --model "structural probe" --layer_index "layer index" --train_probe
```
The structural probe could be: linear, poly, rbf, sigmoid

### Probe a language model with all probe types for a layer index
Generates pickle file that stores test UUAS scores for all probes for a given layer index.
```bash
python src/main.py  --exp "experiment name" --layer_index "layer index" --layer_index_probing
```

### Rank experiment
Generates pickle file that stores test UUAS scores at various ranks of the probe for a given layer index.
```bash
python src/main.py  --exp "experiment name" --layer_index "layer index" --experiment_rank_dim
```
