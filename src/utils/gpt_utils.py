import torch

def get_gpt_representations(ud_parses, model, tokenizer):
    model.eval()
    
    with torch.no_grad():
        all_sentence_reps = []
        for sentence in ud_parses:
            reformed_sent = ""
            no_of_original_tokens = len(sentence)
            
            for tok_idx in range(no_of_original_tokens):
                token = sentence[tok_idx]['form']
                
                if (
                    (
                    sentence[tok_idx]['misc'] is None or 
                    sentence[tok_idx]['form'] in ['"', "[", "]", "(", ")", ".", ",", "?", "'", "other", "can"]
                    ) and 
                    tok_idx != no_of_original_tokens-1
                ):
                    reformed_sent = reformed_sent + token + " "
                else:
                    reformed_sent = reformed_sent + token
            
            inputs = tokenizer(reformed_sent, return_tensors="pt")
                
            outputs = model(**inputs, output_hidden_states=True)
            final_reps = outputs.hidden_states[-1][0]

            idx2 = 0
            start_again = True
            combined_reps = []
            old_token = ""

            for idx, input_id in enumerate(inputs['input_ids'][0]):
                token = tokenizer.convert_ids_to_tokens(input_id.item())

                #### Handling n't ####
                
                if idx2 != no_of_original_tokens-1 and sentence[idx2+1]['form'] == "n't":
                    sentence[idx2]['form'] += sentence[idx2+1]['form'][0]
                    sentence[idx2+1]['form'] = sentence[idx2+1]['form'][1:]

                ######################
                
                if start_again:
                    combined_word_piece = ""
                    curr_rep = []
                    start_again = False

                #### Handling Déjà ####

                if token == 'Ã©':
                    token = 'é'
                if token == 'Ãł':
                    token = 'à'

                #######################
                
                if token[0] == 'Ġ':
                    cur_word_piece = token[1:]
                else:
                    cur_word_piece = token

                #### Handling 's ####
                
                if token == 'âĢ':
                    for _idx in range(3):
                        curr_rep.append(final_reps[idx+_idx])
                    curr_rep = torch.stack(curr_rep)
                    combined_reps.append(curr_rep.mean(0))
                    start_again = True
                    old_token = token
                    idx2 += 1
                    continue
                elif old_token == 'âĢ' and token == 'Ļ':
                    start_again = True
                    old_token = token
                    continue
                elif old_token == 'Ļ' and token == 's':
                    start_again = True
                    old_token = token
                    continue

                #####################

                combined_word_piece = combined_word_piece + cur_word_piece
                curr_rep.append(final_reps[idx])

                if combined_word_piece == sentence[idx2]['form']:
                    curr_rep = torch.stack(curr_rep)
                    combined_reps.append(curr_rep.mean(0))
                    start_again = True
                    old_token = token
                    idx2 += 1
        
            sentence_rep = torch.stack(combined_reps)

            #### DEBUG CODE ####

            if sentence_rep.size(0) != no_of_original_tokens:
                print(f"SENT REP SIZE: {sentence_rep.size(0)} ORIG SIZE: {no_of_original_tokens}")
                print("SENT REC: ", reformed_sent)

                for tok_idx in range(no_of_original_tokens):
                    token = sentence[tok_idx]['form']
                    print(token, end=' ')
                    
                print()
                inputs = tokenizer(reformed_sent, return_tensors="pt")

                for idx, input_id in enumerate(inputs['input_ids'][0]):
                    token = tokenizer.convert_ids_to_tokens(input_id.item())
                    print(token, end=' ')
                print()

            ####################
            
            all_sentence_reps.append(sentence_rep)
        return torch.cat(all_sentence_reps)
