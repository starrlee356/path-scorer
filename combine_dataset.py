import json
from tqdm import tqdm
import random
import os 

trainQ_train_q1 = json.load(open("data_trainQ/train_data_q1_balanced2.json", "r"))
trainQ_train_q2 = json.load(open("data_trainQ/train_data_q2_balanced2.json", "r"))
trainQ_val_q1 = json.load(open("data_trainQ/val_data_q1_balanced2.json", "r"))
trainQ_val_q2 = json.load(open("data_trainQ/val_data_q2_balanced2.json", "r"))

validQ_train_q1 = json.load(open("data_validQ/train_data_q1_balanced2.json", "r"))
validQ_train_q2 = json.load(open("data_validQ/train_data_q2_balanced2.json", "r"))
validQ_val_q1 = json.load(open("data_validQ/val_data_q1_balanced2.json", "r"))
validQ_val_q2 = json.load(open("data_validQ/val_data_q2_balanced2.json", "r"))


if not os.path.exists("data_combine"):
    os.makedirs("data_combine")

com_train_q1 = random.sample(validQ_train_q1, len(validQ_train_q1)//2) + random.sample(trainQ_train_q1, len(trainQ_train_q1)//2)
print(f"com_train_q1: {len(com_train_q1)} = {len(validQ_train_q1)//2} + {len(trainQ_train_q1)//2}")

com_val_q1 = random.sample(validQ_val_q1, len(validQ_val_q1)//2) + random.sample(trainQ_val_q1, len(trainQ_val_q1)//2)
print(f"com_val_q1: {len(com_val_q1)} = {len(validQ_val_q1)//2} + {len(trainQ_val_q1)//2}")

com_train_q2 = random.sample(validQ_train_q2, len(validQ_train_q2)//2) + random.sample(trainQ_train_q2, len(trainQ_train_q2)//2)
print(f"com_train_q2: {len(com_train_q2)} = {len(validQ_train_q2)//2} + {len(trainQ_train_q2)//2}")

com_val_q2 = random.sample(validQ_val_q2, len(validQ_val_q2)//2) + random.sample(trainQ_val_q2, len(trainQ_val_q2)//2)
print(f"com_val_q2: {len(com_val_q2)} = {len(validQ_val_q2)//2} + {len(trainQ_val_q2)//2}")


obj_l = [com_train_q1, com_val_q1, com_train_q2, com_val_q2]
file_name_l = ["data_combine/train_data_q1.json", "data_combine/val_data_q1.json", "data_combine/train_data_q2.json", "data_combine/val_data_q2.json"]
for i in range(4):
    with open(file_name_l[i], "w") as f:
        json.dump(obj_l[i], f)
