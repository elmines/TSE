import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertweetTokenizer

STRING = '<pad>'
sequence_length = 50

def convert_data_to_ids(tokenizer, text):
    
    input_ids, seg_ids, attention_masks, sent_len = [], [], [], []
    for sent in text:
        encoded_dict = tokenizer.encode_plus(' '.join(sent), add_special_tokens = True, max_length = 128, padding = 'max_length', return_attention_mask = True)
        
        if len(encoded_dict['input_ids']) > 128:
            # print(len(encoded_dict['input_ids']))
            encoded_dict['input_ids']      = encoded_dict['input_ids'][:128]
            encoded_dict['token_type_ids'] = encoded_dict['token_type_ids'][:128]
            encoded_dict['attention_mask'] = encoded_dict['attention_mask'][:128]

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        seg_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        sent_len.append(sum(encoded_dict['attention_mask']))
    
    return input_ids, seg_ids, attention_masks, sent_len

def data_helper_bert(x_train_all, x_val_all, x_test_all, model_select):
    
    x_train, x_train_target = x_train_all[0], x_train_all[1]                                                
    x_val, x_val_target     = x_val_all[0],   x_val_all[1]
    x_test, x_test_target   = x_test_all[0],  x_test_all[1]
    
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        
    # tokenization
    x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len = convert_data_to_ids(tokenizer, x_train)
    x_val_input_ids,   x_val_seg_ids,   x_val_atten_masks,   x_val_len   = convert_data_to_ids(tokenizer, x_val)
    x_test_input_ids,  x_test_seg_ids,  x_test_atten_masks,  x_test_len  = convert_data_to_ids(tokenizer, x_test)

    x_train_all = [x_train_input_ids, x_train_seg_ids, x_train_atten_masks, x_train_len, x_train_target]
    x_val_all   = [x_val_input_ids,   x_val_seg_ids,   x_val_atten_masks,   x_val_len,   x_val_target]
    x_test_all  = [x_test_input_ids,  x_test_seg_ids,  x_test_atten_masks,  x_test_len,  x_test_target]
    
    return x_train_all, x_val_all, x_test_all

def data_loader(x_all, batch_size, data_type):
    
    x_input_ids   = torch.tensor(x_all[0], dtype=torch.long).cuda()
    x_seg_ids     = torch.tensor(x_all[1], dtype=torch.long).cuda()
    x_atten_masks = torch.tensor(x_all[2], dtype=torch.long).cuda()
    x_len         = torch.tensor(x_all[3], dtype=torch.long).cuda()
    y             = torch.tensor(x_all[4], dtype=torch.long).cuda()

    tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,x_len,y)

    if data_type == 'train':
        data_loader = DataLoader(tensor_loader, shuffle=True, batch_size=batch_size)
    else:
        data_loader = DataLoader(tensor_loader, shuffle=False, batch_size=batch_size)

    return x_input_ids, x_seg_ids, x_atten_masks, x_len, y, data_loader
