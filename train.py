import torch
import json
from torch.utils.data import Dataset, DataLoader
from path_scorer import PathScorer
from torch.optim import AdamW
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import time
from datetime import datetime
import math
from tqdm import tqdm
import argparse
import os
import logging


class scorerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        dic = self.data[index]
        res = dict()
        for key, val in dic.items():
            res[key] = val
        return res
    
class scorer_trainer:
    def __init__(self, args):
        self.args = args
        self.output_path = os.path.join(self.args.ckpt_path, datetime.now().strftime("%Y-%m-%d-%H-%M"))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.logger = self.get_logger()
        self.model = PathScorer(device=args.device, model_name=args.encoder_model)
        
        if os.path.exists(args.ckpt_file):
            ckpt = torch.load(args.ckpt_file)
            self.model.load_state_dict(ckpt)


        self.optimizer = optim.SGD([{"params": self.model.backbone.parameters(), "lr": args.roberta_lr}, 
                                {"params": self.model.scorer.parameters(), "lr": args.scorer_lr}])

        self.train_loader = self.get_dataloader(args.train_data_file)
        self.val_loader = self.get_dataloader(args.val_data_file)
        self.min_loss = float("inf")
        self.update_lr_ep = 5


    def get_dataloader(self, data_file):
        data = json.load(open(data_file, "r")) #list of dicts. {"rel":str, "pos":str, "neg":str, "margin":str}
        self.logger.info(f"preparing dataloader from {data_file}. data length = {len(data)}")
        dataset = scorerDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.args.encoder_bsz, shuffle=True)
    
        return dataloader
    
    def get_logger(self):
        log_f = os.path.join(self.output_path, f"train.log")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_f)
        console_handler = logging.StreamHandler()
        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter) 
        console_handler.setFormatter(formatter) 
        
        logger.addHandler(file_handler) 
        logger.addHandler(console_handler)
        for arg, val in vars(self.args).items():
            logger.info(f"{arg}: {val}")

        return logger

    def train_per_epoch(self, epoch_id):
        # if epoch_id == self.update_lr_ep:
        #     self.optimizer.param_groups[0]['lr'] *= 0.1
        #     self.optimizer.param_groups[1]['lr'] *= 0.1
        #     self.logger.info(f"lr updated.")

        self.model.train()
        # self.model.base = 0
        train_loss = 0.
        start = time.time()
        for batch in tqdm(self.train_loader, desc=f"epoch{epoch_id} train"): #batch: {"rel":rel_list, "path":list, "score":list}
            self.optimizer.zero_grad()
            loss = self.model(batch)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()
            torch.cuda.empty_cache()

        train_loss /= len(self.train_loader)
        end = time.time()
        self.logger.info(f"epoch{epoch_id} train fin: time = {math.ceil(end-start)}s, avg train loss = {train_loss}, roberta lr = {self.optimizer.param_groups[0]['lr']:.3e}, ff lr = {self.optimizer.param_groups[1]['lr']:.3e}")


    def val_per_epoch(self, epoch_id):
        self.model.eval()
        val_loss = 0.
        start = time.time()
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"epoch{epoch_id} valid"):
                loss = self.model(batch)
                val_loss += loss.item()
                torch.cuda.empty_cache()

        val_loss /= len(self.val_loader)
        end = time.time()
        self.logger.info(f"epoch{epoch_id} val fin: time = {math.ceil(end-start)}s, avg val loss = {val_loss}")

  
        if self.min_loss > val_loss:
            save_file = os.path.join(self.output_path, f"encoder_ep{epoch_id}.ckpt")
            torch.save(self.model.state_dict(), save_file)
            self.logger.info(f"model saved to {save_file}")
            self.min_loss = val_loss
 
    def main(self):
        for epoch in range(self.args.start_ep, self.args.start_ep+self.args.epoch_num):
            self.train_per_epoch(epoch_id=epoch)
            self.val_per_epoch(epoch_id=epoch)            
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_num", type=int, default=20)
    parser.add_argument("--ckpt_path", type=str, default="output")
    parser.add_argument("--encoder_bsz", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--encoder_model", type=str, default="bert-base-uncased") #FacebookAI/roberta-base
    parser.add_argument("--roberta_lr", type=float, default=1e-3)
    parser.add_argument("--scorer_lr", type=float, default=1e-3)
    parser.add_argument("--train_data_file", type=str, default="data/train_data_q1_balanced2.json")
    parser.add_argument("--val_data_file", type=str, default="data/val_data_q1_balanced2.json")
    parser.add_argument("--start_ep", type=int, default=0)
    parser.add_argument("--ckpt_file", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)

    trainer = scorer_trainer(args)
    trainer.main()
