import pandas as pd
import json
import tqdm

import torch
#from transformers import RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer, RobertaForMaskedLM
from transformers import BertTokenizer, BertModel, BertForMaskedLM, pipeline
import string
import csv

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertConfig.from_pretrained('roberta-base')
 # Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('roberta-base')
model.eval()
model.to('cuda')

def predict_token(predict, masked_index):
    #predict = predictions[0, masked_index]
    predicted_index = torch.argmax(predict).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    while predicted_token[0] == '#' or predicted_token =='。' or predicted_token == '॥' or predicted_token == '[UNK]' or predicted_token in string.punctuation or predicted_token == '...':
        predict[predicted_index] = float('-inf')
        predicted_index = torch.argmax(predict).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

    predict[predicted_index] = float('-inf')
    return predicted_token, predict

def predict_word(tokenized_text, masked_index):
    
    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0] * len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    predict = predictions[0, masked_index]
    predicted_token1, predict = predict_token(predict, masked_index)
    predicted_token2, predict = predict_token(predict, masked_index)
    predicted_token3, predict = predict_token(predict, masked_index)

    #assert predicted_token == 'henson'
    return predicted_token1, predicted_token2, predicted_token3

df = pd.read_csv("v4_atomic_all_agg.csv",index_col=0)
events = df.iloc[:,:0].apply(lambda col: col.apply(json.loads)).index

with open('v4_atomic_all_agg_complete_sentence.csv', 'w', encoding='utf8') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ')
    for item in tqdm.tqdm(events,desc="Predicting"):
        sentence = item.strip().split(" ")
        #sentence = list(orig_sentence)
        sentence = [sub.replace('PersonX', 'Alex') for sub in sentence]
        sentence = [sub.replace('PersonY', 'John') for sub in sentence]
        ssentence = [sub.replace('PersonZ', 'Mike') for sub in sentence]
        replace = "___"
        if replace in sentence:
            result = sentence.index("___")
            sentence[result] = '[MASK]'
            sentence1 = ['[CLS]'] + sentence + ['.'] + ['[SEP]']
            prediction1, prediction2, prediction3 = predict_word(sentence1, result+1)
            spamwriter.writerow([str(index)] + [','] + [prediction1])
            spamwriter.writerow([str(index)] + [','] + [prediction2])
            spamwriter.writerow([str(index)] +[',']+ [prediction3])
            #print(orig_sentence, sentence)
        else:
            spamwriter.writerow([str(index)] + [','])
    
       
    
