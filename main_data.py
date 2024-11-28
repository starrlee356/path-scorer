import argparse
import os
from tqdm import tqdm
import pickle as pkl
from global_config import *
from collections import defaultdict
import csv

def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = value.union(dict_1[key])
   return dict_3

def merge_easy_hard():
     modes = ["valid", "test"]
     for mode in modes:
          ans_file = os.path.join(args.data_path, f"{mode}-answers.pkl")

          if not os.path.exists(ans_file):
            easy_ans_file = os.path.join(args.data_path, f"{mode}-easy-answers.pkl")
            hard_ans_file = os.path.join(args.data_path, f"{mode}-hard-answers.pkl")
            easy_ans = pkl.load(open(easy_ans_file, "rb")) #query:ans
            hard_ans = pkl.load(open(hard_ans_file, "rb"))
            ans = mergeDictionary(easy_ans, hard_ans)
            
            with open(ans_file, "wb") as f:
                pkl.dump(ans, f)

def merge_train_valid_test():
        modes = ["train", "valid", "test"]
        q_dict_list = []
        a_dict_list = []
        for mode in modes:
            query_dict = pkl.load(open(os.path.join(args.data_path, f"{mode}-queries.pkl"), "rb"))
            answer_dict = pkl.load(open(os.path.join(args.data_path, f"{mode}-answers.pkl"), "rb")) #query: ans list
            q_dict_list.append(query_dict)
            a_dict_list.append(answer_dict)
        query_dict = mergeDictionary(mergeDictionary(q_dict_list[0], q_dict_list[1]), q_dict_list[2])
        answer_dict = mergeDictionary(mergeDictionary(a_dict_list[0], a_dict_list[1]), a_dict_list[2])
        with open(os.path.join(args.data_path, f"all-answers.pkl"), "wb") as f:
            pkl.dump(answer_dict, f)
        with open(os.path.join(args.data_path, f"all-queries.pkl"), "wb") as f:
            pkl.dump(query_dict, f)

def gen_answers_id2q_qnum():
    ans_dir = os.path.join(args.output_path, f"{args.QA_mode}_sorted_answers")
    if not os.path.exists(ans_dir):
        os.makedirs(ans_dir)
    
    query_dict = pkl.load(open(os.path.join(args.data_path, f"{args.QA_mode}-queries.pkl"), "rb")) # qpattern：set of queries of this qtype
    answer_dict = pkl.load(open(os.path.join(args.data_path, f"{args.QA_mode}-answers.pkl"), "rb")) #query: ans list
        
    id2q = defaultdict()
    qtype2cnt = defaultdict()
    for qtype, qpattern in QUERY_STRUCTS.items():
        queries = query_dict[qpattern]
        qtype2cnt[qtype] = len(queries)
        id2q[qtype] = {i:q for i,q in enumerate(queries)}
        
        for i, q in tqdm(id2q[qtype].items(), desc=f"{qtype}"):
            save_file = os.path.join(ans_dir, f"{qtype}_{i}_answer.txt")
            text = ", ".join(map(str, answer_dict[q]))
            with open(save_file, "w") as f:
                print(text, file=f)

    id2q_file = os.path.join(args.output_path, f"{args.QA_mode}-id2q.pkl") 
    qtype2cnt_file = os.path.join(args.output_path, f"{args.QA_mode}-qtype2cnt.pkl")            
    with open(id2q_file, "wb") as f:
        pkl.dump(id2q, f)
    with open(qtype2cnt_file, "wb") as f:
        pkl.dump(qtype2cnt, f)
      
def gen_LARK_triplets():
    entity_triplets, relation_triplets = {}, {}
    if args.graph_mode == "all":
        triplet_files = [os.path.join(f"{args.data_path}","train.txt"), 
                        os.path.join(f"{args.data_path}","valid.txt"), 
                        os.path.join(f"{args.data_path}","test.txt")]
    else:
        triplet_files = [os.path.join(f"{args.data_path}", f"{args.graph_mode}.txt")]
   
    for triplet_file in triplet_files:
        with open(triplet_file,"r") as kg_data_file:
            kg_tsv_file = csv.reader(kg_data_file, delimiter="\t")
            for line in kg_tsv_file:
                e1, r, e2 = map(int,line)
                triplet = (e1, r, e2)
                if e1 in entity_triplets: entity_triplets[e1].add(triplet)
                else: entity_triplets[e1] = set([triplet])
                if e2 in entity_triplets: entity_triplets[e2].add(triplet)
                else: entity_triplets[e2] = set([triplet])
                if r in relation_triplets: relation_triplets[r].add(triplet)
                else: relation_triplets[r] = set([triplet])

    dir_name = os.path.join(args.output_path, "LARK")
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(os.path.join(dir_name, f"{args.graph_mode}-entity-triplets.pkl"),"wb") as entity_triplets_file:
        pkl.dump(entity_triplets, entity_triplets_file)
    with open(os.path.join(dir_name, f"{args.graph_mode}-relation-triplets.pkl"),"wb") as relation_triplets_file:
        pkl.dump(relation_triplets, relation_triplets_file)

def gen_llmR_triplets():
    
    entity_triplets = {}
    if args.graph_mode == "all":
        triplet_files = [os.path.join(f"{args.data_path}","train.txt"), 
                        os.path.join(f"{args.data_path}","valid.txt"), 
                        os.path.join(f"{args.data_path}","test.txt")]
    else:
        triplet_files = [os.path.join(f"{args.data_path}", f"{args.graph_mode}.txt")]

    for triplet_file in triplet_files:
        with open(triplet_file,"r") as kg_data_file:
            kg_tsv_file = csv.reader(kg_data_file, delimiter="\t")
            for line in kg_tsv_file:
                e1, r, e2 = map(int,line)
                triplet = (e1, r, e2)
                if e1 in entity_triplets: entity_triplets[e1].add(triplet)
                else: entity_triplets[e1] = set([triplet])
                if e2 in entity_triplets: entity_triplets[e2].add(triplet)
                else: entity_triplets[e2] = set([triplet])

    dir_name = os.path.join(args.output_path, "llmR") 
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(os.path.join(dir_name, f"{args.graph_mode}-entity-triplets.pkl") , "wb") as entity_triplets_file:
        pkl.dump(entity_triplets, entity_triplets_file)

def gen_headent_triplets():#head ent id: set of triplets
    
    entity_triplets = {}
    if args.graph_mode == "all":
        triplet_files = [os.path.join(f"{args.data_path}","train.txt"), 
                        os.path.join(f"{args.data_path}","valid.txt"), 
                        os.path.join(f"{args.data_path}","test.txt")]
    else:
        triplet_files = [os.path.join(f"{args.data_path}", f"{args.graph_mode}.txt")]

    for triplet_file in triplet_files:
        with open(triplet_file,"r") as kg_data_file:
            kg_tsv_file = csv.reader(kg_data_file, delimiter="\t")
            for line in kg_tsv_file:
                e1, r, e2 = map(int,line)
                triplet = (e1, r, e2)
                if e1 in entity_triplets: entity_triplets[e1].add(triplet)
                else: entity_triplets[e1] = set([triplet])
                #if e2 in entity_triplets: entity_triplets[e2].add(triplet)
                #else: entity_triplets[e2] = set([triplet])

    dir_name = os.path.join(args.output_path, "llmR") 
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(os.path.join(dir_name, f"{args.graph_mode}-headent-triplets.pkl") , "wb") as entity_triplets_file:
        pkl.dump(entity_triplets, entity_triplets_file)

def gen_rel2path():
    ent2tri = pkl.load(open(os.path.join(args.output_path, "llmR", f"{args.graph_mode}-headent-triplets.pkl"), "rb"))
    rel2path = {}
    for ent, triplets in tqdm(ent2tri.items()):
        tail2path = {}
        for (_, r0, t0) in triplets:
            path = [r0]
            for (_, r1, t1) in ent2tri[t0]:
                path.append(r1)
                if t1 in tail2path:
                    tail2path[t1].add(tuple(path))
                else:
                    tail2path[t1] = {tuple(path)}
                for (_, r2, t2) in ent2tri[t1]:
                    path.append(r2)
                    if t2 in tail2path:
                        tail2path[t2].add(tuple(path))
                    else:
                        tail2path[t2] = {tuple(path)}
                    path.pop()
                path.pop()
        for (_, r0, t0) in triplets:
            if r0 in rel2path:
                rel2path[r0].union(tail2path[t0])
            else:
                rel2path[r0] = tail2path[t0]
    with open(os.path.join(args.output_path, "llmR", "rel2path.pkl"), "wb") as f:
        pkl.dump(rel2path, file=f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="data/NELL0/processed")
    parser.add_argument("--data_path", type=str, default="data/NELL0")
    parser.add_argument("--graph_mode", type=str, default="train", help="train, test, valid (.txt)")
    parser.add_argument("--QA_mode", type=str, default="valid", help="train, test, valid, all")
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    #merge_easy_hard()
    #merge_train_valid_test()
    gen_answers_id2q_qnum()
    # gen_LARK_triplets()
    # gen_llmR_triplets()
    # gen_headent_triplets()
    # gen_rel2path()

