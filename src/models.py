import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers import XLMRobertaConfig, XLMRobertaModel


 
class XLMRobertaLargeTC(nn.Module):
    def __init__(self):
        super(XLMRobertaLargeTC, self).__init__()
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-large', output_hidden_states=True)
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-large', config=config)
        
        self.fc = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)
        
        # initialize weight
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)
        
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        _, o2, _ = self.xlm_roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask)
        
        o2 = self.dropout(o2)
        logits = self.fc(o2)
        
        return logits
    
    
class XLMRobertaBaseTC(nn.Module):
    def __init__(self):
        super(XLMRobertaBaseTC, self).__init__()
        config = XLMRobertaConfig.from_pretrained('xlm-roberta-base', output_hidden_states=True)
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base', config=config)
        
        self.fc = nn.Linear(config.hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)
        
        # inititalize weights
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)
        
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        _, o2, _ = self.xlm_roberta(
            input_ids=input_ids, 
            attention_mask=attention_mask)
        
        o2 = self.dropout(o2)
        logits = self.fc(o2)
        
        return logits
    
    
class BertMultilingualCaseTC(nn.Module):
    def __init__(self):
        super(BertMultilingualCaseTC, self).__init__()
        config = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-cased', config=config)
        self.fc = nn.Linear(config.hidden_size, 1)
        
        # initialize weights
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        _, o2, _ = self.bert_model(
            input_ids=input_ids, 
            attention_mask=attention_mask)
        
        o2 = self.dropout(o2)
        logits = self.fc(o2)
        
        return logits
    
    
class BertMultilingualUncasedTC(nn.Module):
    def __init__(self):
        super(BertMultilingualUncasedTC, self).__init__()
        config = BertModel.from_pretrained('bert-base-multilingual-uncased', output_hidden_states=True)
        self.bert_model = BertModel.from_pretrained('bert-base-multilingual-uncased', config=config)
        self.fc = nn.Linear(config.hidden_size, 1)
        
        # initliaze weights
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):
        _, o2, _ = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        o2 = self.dropout(o2)
        logits = self.fc(o2)
        
        return logits
