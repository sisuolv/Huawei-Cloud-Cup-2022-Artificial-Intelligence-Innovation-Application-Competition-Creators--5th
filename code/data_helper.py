import random
from functools import partial
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from tqdm import tqdm as tqdm
import pandas as pd
import numpy as np
import os
import io

from sklearn.model_selection import StratifiedKFold, KFold
import joblib

import glob
import tqdm
from transformers import BertTokenizer

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from transformers import BertModel
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import re

def create_dataloaders(args):


    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    def tokenize_text(text: str) -> tuple:
            encoded_inputs =tokenizer(text, max_length=1024, padding='max_length', truncation=True)
            input_ids = torch.LongTensor(encoded_inputs['input_ids'])
            mask = torch.LongTensor(encoded_inputs['attention_mask'])
            return input_ids, mask
        
    label_path = 'labels.txt'
    id2label = []
    with open(label_path, 'r', encoding='utf8') as f:
        for line in f:
            id2label.append(line.strip())    
    def get_diagnosis( pred):
        pred_index = [i for i in range(len(pred)) if pred[i] == 1]
        pred_diagnosis = [id2label[index] for index in pred_index]
        return pred_diagnosis
        
        

    with open(args.traindata_path, 'r', encoding='utf8') as f:
        df_train = f.readlines()

    
    
    # print(  df_train )
    split_index = int(len(df_train) * 0.9)
    random.seed(args.seed)
    random.shuffle(df_train)
    train_da = df_train[:split_index]
    val_da = df_train[split_index:]
    
    tmp = np.zeros((len(df_train), 2))
    tmp_all = pd.DataFrame(tmp)
    for t in tqdm.tqdm(range(len(df_train))):
        tmp = json.loads(df_train[t])
        tmp_all.iloc[t, 0] =  int(tmp['emr_id'][2:])
        tmp_all.iloc[t, 1] =  '，'.join(tmp['diagnosis'])

    tmp_all.to_csv('./dic.csv',  index=0) 
    

    for t2 in range(len(train_da)):
        tempv1 = json.loads(train_da[t2])
        aug_text1 = tempv1['history_of_present_illness']
        if len(tempv1['diagnosis'])<=2:
            aug_label1 = tempv1['diagnosis']
            index2= int(np.random.choice(len(train_da), 1))
            tempv2 = json.loads(train_da[index2])
            aug_text2 = tempv2['history_of_present_illness']
            aug_label2 = tempv2['diagnosis']
            aug_text3 = aug_text1 + '，' + aug_text2
            aug_label3 = aug_label1 + aug_label2
            aug_label3 =list(set(aug_label3))
            tempv1['history_of_present_illness'] = aug_text3
            tempv1['diagnosis'] = aug_label3
            # print(tempv1['diagnosis'])
            train_da.append(json.dumps(tempv1))
            
    
#     def feat_jia(df_train):
        
#     ##########增强
#         dic ={}
#         from model_hunliu_small import MultiModal
#         model = MultiModal(args.bert_dir)
#         model.load_state_dict(torch.load('./save/model_hunliu_small_best_202160186.pt', map_location='cpu'))
#         model = model.to(DEVICE)
#         model.eval()

#         all_datadd = []
#         for t in tqdm.tqdm(range(len(df_train))):
#             tmp = json.loads(df_train[t])
#             text_plus =  pad_clip_text(tmp)
#             title_input, title_mask = tokenize_text(text_plus)
#             output = model(title_input.unsqueeze(0).to(DEVICE), title_mask.unsqueeze(0).to(DEVICE))
#             output = torch.sigmoid(output)
#             pred_labels = output.cpu().detach().numpy()
#             pred_labels = [1 if pred > 0.5 else 0 for pred in pred_labels[0]]
#             pred_diagnosis = get_diagnosis(pred_labels)
#             aug_label3 =pred_diagnosis 
#             aug_label3 =list(set(aug_label3))
#             tmp['feat'] = '，'.join(aug_label3)
#             all_datadd.append(json.dumps(tmp))
#         df_train = all_datadd
#         return df_train
    
#     train_da = feat_jia(train_da)
#     val_da = feat_jia(val_da)
    
    
    
    
    np.save('./train_da.npy', np.array(train_da))
    np.save('./val_da.npy', np.array(val_da))
 
    train_dataset = MyDataset(args, train_da, mode='train')
    val_dataset = MyDataset(args, val_da, mode='val')
    print('训练shape:', len(train_dataset), '验证shape:', len(val_dataset))
    dataloader_class = partial(DataLoader, pin_memory=False, num_workers=args.num_workers,
                               prefetch_factor=args.prefetch)
    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True,
                                        )
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.batch_size,
                                      sampler=val_sampler,
                                      drop_last=False,
                                      )
    return train_dataloader, val_dataloader

class MyDataset(Dataset):
    def __init__(self, args, datax, mode='train'):
        self.datax = datax
        self.args = args
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = mode
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir)

        with open("./labels.txt", 'r', encoding='utf8') as f:
            labels = f.readlines()
        label_icd = []
        label2id = {}
        for idx, label in enumerate(labels):
            label_icd.append(label.strip())
            label2id[label.strip()] = idx
        self.label2id = label2id
        
        self.label_icd = [ ]
        for tt in label_icd:
            self.label_icd.append(tt)
        print(self.label_icd)
    def __len__(self):
        return len(self.datax)

    def tokenize_text(self, text: str) -> tuple:
        encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def clean_txt2(self, pp):
        if pp == None:
            return '0'
        pp = pp.replace(' ', '')
        pp = pp.replace('\n', '')
        pp = pp.replace('\r', '')
        if pp == '':
            pp = '0'
        return pp
    
   
    
    def category_id_to_lv2id(self, targets):
        label = [0] * len(self.label2id)
        for target in targets:
            if target in self.label2id:
                idx = self.label2id[target]
                label[idx] = 1
        return label
    
    def pad_clip_text(self, data):
        input0 = self.clean_txt2(data["age"])
        input1 = self.clean_txt2(data["gender"])
        input2 = self.clean_txt2(data["chief_complaint"])
        input3 = self.clean_txt2(data["supplementary_examination"])
        input4 = self.clean_txt2(data["history_of_present_illness"])
        input5 = self.clean_txt2(data["past_history"])
        input6 = self.clean_txt2(data["physical_examination"])
        text_plus = '[CLS]'+ input0 +'[SEP]'  +  input1 +  '[SEP]' + input2 + '[SEP]' + input3 + '[SEP]' + input4 + '[SEP]' + input5 + '[SEP]' + input6 + '[SEP]'
        return text_plus

        
    def __getitem__(self, idx):
        data = json.loads(self.datax[idx])
        text_plus= self.pad_clip_text(data)
#         while len(text_plus) <1300:
#             text_plus +=  text_plus
#         text_plus =  '[CLS]' +text_plus
        title_input, title_mask = self.tokenize_text(text_plus)
        data_dic = dict(
                title_input=title_input,
                title_mask=title_mask,
                helpfu =torch.FloatTensor([len(data['diagnosis'])]), 
                    )
        if self.test_mode in ['train', 'val']:
            label = self.category_id_to_lv2id(data['diagnosis'])
            data_dic['label'] = torch.FloatTensor([label])
        return data_dic
    

        



