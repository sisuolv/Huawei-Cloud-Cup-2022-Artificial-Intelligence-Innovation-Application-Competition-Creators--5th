# -*- coding: utf-8 -*-
import logging
import os

import torch
import json
from transformers import BertTokenizer

from model_service.pytorch_model_service import PTServingBaseService
from model_hunliu_small import MultiModal
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(DEVICE)
import re

class DaService(PTServingBaseService):

    def __init__(self, model_name, model_path):
        super(PTServingBaseService, self).__init__(model_name, model_path)
        dir_path = os.path.dirname(os.path.realpath(model_path))
        bert_path = os.path.join(dir_path, 'medbertbase')


        self.bert_seq_length = 1024
        self.model1 = MultiModal(bert_path)
        self.model1.load_state_dict(torch.load(os.path.join(dir_path, 'model_hunliu_small_best_2021602.pt'), map_location='cpu'))
        
#         self.model2 = MultiModal(bert_path)
#         self.model2.load_state_dict(torch.load(os.path.join(dir_path, 'm2/model_hunliu_small_best_2018_6377.pt'), map_location='cpu'))
        

        self.model1.to(DEVICE)
        # self.model2.to(DEVICE)
        
        
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)


        label_path = os.path.join(dir_path, 'labels.txt')
        self.id2label = []
        with open(label_path, 'r', encoding='utf8') as f:
            for line in f:
                self.id2label.append(line.strip())

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
        text_plus =  input0 +'[SEP]'  +  input1 +  '[SEP]' + input2 + '[SEP]' + input3 + '[SEP]' + input4 + '[SEP]' + input5 + '[SEP]' + input6 + '[SEP]'
        return text_plus


    def _get_diagnosis(self, pred):
        pred_index = [i for i in range(len(pred)) if pred[i] == 1]
        pred_diagnosis = [self.id2label[index] for index in pred_index]
        return pred_diagnosis

    def _preprocess(self, data):
        data_dict = data.get('json_line')
        for v in data_dict.values():
            infer_dict = json.loads(v.read())
            return infer_dict

    def _inference(self, data):
        emr_id = data.get('emr_id')
        text_plus= self.pad_clip_text(data)
        # while len(text_plus) <1300:
        #     text_plus +=  text_plus
        # text_plus =  '[CLS]' +text_plus
        title_input, title_mask = self.tokenize_text(text_plus)

        self.model1.eval()
        output1 = self.model1(title_input.unsqueeze(0).to(DEVICE), title_mask.unsqueeze(0).to(DEVICE))
        output1 = torch.sigmoid(output1)
        
#         self.model2.eval()
#         output2 = self.model2(title_input.unsqueeze(0).to(DEVICE), title_mask.unsqueeze(0).to(DEVICE))
#         output2 = torch.sigmoid(output2)

        result = {emr_id: output1}
        return result

    def _postprocess(self, data):
        infer_output = None
        for k, v in data.items():
            pred_labels = v.cpu().detach().numpy()
            # pred_labels2 = v[1].cpu().detach().numpy()
            # pred_labels = pred_labels1*0.6 + pred_labels2*0.4
            
            pred_labels = [1 if pred > 0.5 else 0 for pred in pred_labels[0]]
            pred_diagnosis = self._get_diagnosis(pred_labels)
            infer_output = {k: pred_diagnosis}
        return infer_output
