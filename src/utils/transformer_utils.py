import torch

def get_gpt_representations(ud_parses, model, tokenizer, layer_index):
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


def get_bert_representations(ud_parses, model, tokenizer, layer_index):
    model.eval()
    
    with torch.no_grad():
        all_sentence_reps = []
        for sentence in ud_parses:
            no_of_original_tokens = len(sentence)
            original_tokens = []
            recon_sent = ""
            
            for tok_idx in range(no_of_original_tokens):
                token = sentence[tok_idx]['form']
                original_tokens.append(token)
                recon_sent = recon_sent + token + ' '
                    
            recon_sent = '[CLS] ' + recon_sent.strip() + ' [SEP]'
            tokenized_text = tokenizer.wordpiece_tokenizer.tokenize(recon_sent)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segment_ids = [1 for _ in tokenized_text]
            
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segment_ids])
            
            outputs = model(tokens_tensor, segments_tensors, output_hidden_states=True)
            hidden_state = outputs.hidden_states[layer_index][0]
            
            idx2 = 0
            combine_word = ''
            combine_representations = []
            all_word_representations = []
            combine_word_list = []
            
            for idx, bert_token in enumerate(tokenized_text[:-1]):
                if bert_token in ['[CLS]', 'SEP']:
                    continue
                    
                if bert_token[:2] == '##' and bert_token not in recon_sent:
                    bert_token = bert_token[2:]
                    
                combine_word = combine_word + bert_token
                combine_representations.append(hidden_state[idx])
                
                if combine_word == original_tokens[idx2] or combine_word == '[UNK]':
#                     print("This is combine word", combine_word)
                    combine_word_list.append(combine_word)
                    all_word_representations.append(torch.stack(combine_representations).mean(0))
                    idx2 += 1
                    combine_word = ''
                    combine_representations = []
            
            try:
                sentence_rep = torch.stack(all_word_representations)
            except Exception as error:
                print(error)
                print(recon_sent)
                print(tokenized_text)
                print(combine_word_list)
                
            all_sentence_reps.append(sentence_rep)
    return torch.cat(all_sentence_reps)