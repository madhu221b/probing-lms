# Probing Language Models

Repository for the NLP2 assignment on probing language models.

## Main Caller function to train the probes 

```bash
python src/main.py  --exp "experiment_name"  --layer_index "layer_index"
```

The experiement name values can be: bert, bertL, gpt
And the layer index values can be 1 to 12 for gpt and bert and 1 to 24 for bertL.
