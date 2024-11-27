from transformers import AutoTokenizer, RobertaModel, BertModel
import torch
import torch.nn as nn 
import torch.nn.functional as F

class PathScorer(nn.Module):
    def __init__(self, device, model_name):
        super(PathScorer, self).__init__()
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.roberta = RobertaModel.from_pretrained(model_name).to(self.device)
        self.backbone = BertModel.from_pretrained(model_name).to(self.device)
        self.scorer = nn.Sequential(nn.Dropout(p=0.1, inplace=False), nn.Linear(self.backbone.config.hidden_size, 1)).to(self.device)
        self.gt_threshold = 0.65
        self.eps = 1e-15

    def forward(self, input_batch):
        #input_batch {"rel":rel_list, "path":list, "score":list} 
        input = self.tokenizer(input_batch["rel"], input_batch["path"], padding=True, truncation=True, max_length=200, return_tensors="pt")
        for key, val in input.items():
            input[key] = val.to(self.device)

        #roberta output dict. {"last_hidden_state":tensor[batch,seqlen,768], "pooler_output":tensor[batch,768]}
        gt_scores = torch.tensor(input_batch["score"]).float().to(self.device) #[batch]
        # gt_scores = (gt_scores > self.gt_threshold).float() #T,F -> 1,0. if fuse score > threshold, as pos sample. get cls id = 1. 
        pred_scores = self.scorer(self.backbone(**input)["pooler_output"]).view(-1) # [batch, 1]->[batch]

        # loss_fn = nn.BCEWithLogitsLoss()
        # loss = loss_fn(pred_scores, gt_scores)
        loss_fn = nn.BCELoss()
        pred_scores = torch.sigmoid(pred_scores) # nn.Sigmoid(pred_scores)
        loss = loss_fn(pred_scores, gt_scores)
        return loss

    
    def infer(self, input_batch):
        #input_batch {"rel":list, "path":list}
        with torch.no_grad():
            input = self.tokenizer(input_batch["rel"], input_batch["path"], padding=True, truncation=True, max_length=200, return_tensors="pt")
            for key, val in input.items():
                input[key] = val.to(self.device)

            #roberta output dict. {"last_hidden_state":tensor[batch,seqlen,768], "pooler_output":tensor[batch,768]}
            scores = self.scorer(self.backbone(**input)["pooler_output"]).view(-1) #[batch] logits
            scores = scores.cpu()
            #for val in input.values():
                #del val # free tensors
            torch.cuda.empty_cache()
            return scores
    

