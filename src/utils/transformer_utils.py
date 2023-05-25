import torch

def get_transformer_representations(ud_parses, model, tokenizer, layer_index):
    model.eval()
    
    with torch.no_grad():
        all_sentence_reps = []
        for sentence in ud_parses:            
            no_of_original_tokens = len(sentence)
            inputs_ids_list = []
            
            for tok_idx in range(no_of_original_tokens):
                token = sentence[tok_idx]['form']
                inputs_ids_list.append(tokenizer(token, return_tensors="pt")['input_ids'])
                
            inputs_tensor = torch.cat(inputs_ids_list, -1)
            
            outputs = model(input_ids=inputs_tensor, output_hidden_states=True)
            final_reps = outputs.hidden_states[layer_index][0]
            
            combined_reps = []
            idx = 0
            
            for input_ids in inputs_ids_list:
                i_len = input_ids.size(-1)
                combined_reps.append(final_reps[idx:idx+i_len].mean(0))
                idx += i_len
        
            sentence_rep = torch.stack(combined_reps)
            
            all_sentence_reps.append(sentence_rep)
    return torch.cat(all_sentence_reps)
