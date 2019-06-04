import os
import sys
import numpy as np
import pickle
import json
import os

from argparse import ArgumentParser

from tqdm import tqdm

from collections import defaultdict

import random
import torch
import jieba

import itertools
from itertools import chain
from collections import defaultdict

from fairseq.tokenizer_jieba import JiebaTokenizer
from fairseq.tokenizer_bert import BertTokenizer

import pdb

def get_profile_id(response_profile, profiles, uid):
    if response_profile not in profiles:
        if uid==0:
            return 1
        else:
            return 0
    else:
        return profiles.index(response_profile)

def tokenizer_profile(profile):
    if isinstance(profile["tag"], list):
        tag = profile["tag"][0]
    else:
        tag = profile["tag"]
    loc = profile["loc"]
    gender = profile["gender"]
    new_profile = []
    if tag:
        new_profile.append(' '.join(jieba.lcut(tag)))
    else:
        new_profile.append('')
    new_profile.append(loc)
    if gender:
        new_profile.append(' '.join(jieba.lcut(gender)))
    else:
        new_profile.append('')

    return new_profile

def convert_to_standard(data):
    response_profile = data["response_profile"] if "responder_profile" not in data else data["responder_profile"]
    profiles = data["profile"]
    dialogs = data["dialog"].copy()
    uid = data["uid"].copy()
    golden_response = data["golden_response"][0]
    response_uid = get_profile_id(response_profile, profiles, uid[-1])
    uid.append(response_uid)
    lm_mask = [1]*len(uid)
    lm_mask[-1] = 0
    profile = tokenizer_profile(response_profile)
    dialog = [talk[0] for talk in dialogs]
    dialog.append(golden_response)
    try:
        assert len(uid)==len(dialog)
    except Exception:
        pdb.set_trace()
    return {
        "uid":uid,
        "dialog":dialog,
        "lm_mask":lm_mask,
        "profile":profile,
        "p_id":response_uid
    }

def preprocess_valid_data(data_json):
    return [convert_to_standard(data) for data in data_json]

def mask_person(uid_list,uid):
    """
    1：被mask，不用于计算当前的损失
    0：不被mask，用于计算当前的损失
    """
    mask_list = list()
    for id in uid_list:
        if id == uid:
            mask_list.append(0)
        else:
            mask_list.append(1)
    return mask_list

def get_profile_tag_list(profile_tag:dict):
    tag_name = ['tag','loc','gender']
    return [profile_tag[x] for x in tag_name]

def preprocess_train_data(line):
    all_step_dialog = list()
    new_format = list()
    train_one = dict()
    dialog = line['dialog']
    uid_list = line['uid']
    person_num = len(list(uid_list))

    # profile 处理
    profiles = {i:get_profile_tag_list(y) for i,y in enumerate(line['profile'])}
    #print(profiles)
    for key,value in profiles.items():
        #print(value)
        value[0] = ' '.join(jieba.lcut(value[0][0]))

    # 构建mask列表
    for uid in profiles.keys():
        train_one = dict()
        lm_mask = mask_person(uid_list,uid)
        train_one['dialog'] = list(map(lambda x:x[0],dialog))
        train_one['profile'] = profiles[uid]
        train_one['p_id'] = uid
        train_one['lm_mask'] = lm_mask
        train_one['uid'] = uid_list

        lm_mask[0] = 1 
        if sum(lm_mask)==len(lm_mask):
            continue
        pro = ""
        for a in train_one['profile']:
            pro += a
        if not pro:
            continue
        #train_one = json.dumps(train_one,ensure_ascii=False)
        new_format.append(train_one)

    # 构建新的格式的训练语料
    for x in new_format:
        all_step_dialog += step_dialog(x)
    #print(all_step_dialog)
    return all_step_dialog

def step_dialog(person_dialog):
    step_dialog = list()
    for index,mask in enumerate(person_dialog['lm_mask']):
        if mask == 1:
            continue
        #pdb.set_trace()
        new = dict(person_dialog)
        new['lm_mask'] = [1] * index + [0]
        new['dialog'] = list(new['dialog'])[:index+1]
        new['uid'] = list(new['uid'])[:index+1]
        #new = json.dumps(new,ensure_ascii=False)
        step_dialog.append(new)

    return step_dialog

def build_train_data(dialog_dict, bert_dict, jieba_dict):
    # pdb.set_trace()
    dialog_len = len(dialog_dict["dialog"])
    uid = dialog_dict["uid"][:-1]
    response = dialog_dict["dialog"].pop()
    dialog_ids = [bert_dict.convert_text_to_ids(d) for index,d in enumerate(dialog_dict["dialog"])]
    assert dialog_len == len(dialog_ids)+1
    assert len(uid)==len(dialog_ids)

    response_jieba = jieba_dict.convert_text_to_ids(response.strip())
    response_bert = [bert_dict.convert_text_to_ids(token) for token in response.strip().split()]
    assert len(response_bert)==len(response_jieba)

    profiles = [bert_dict.convert_text_to_ids(d) for d in dialog_dict["profile"]]
    assert len(profiles)==3

    instance = {
        "dialog_history": dialog_ids,
        "uid": uid,
        "profile": profiles,
        "response_bert": response_bert,
        "response_jieba": response_jieba,
    }
    return instance

def read_dialog(file):
    """
    Read dialogs from file
    :param file: str, file path to the dataset
    :return: list, a list of dialogue (context) contained in file
    """
    with open(file) as f:
        contents = [i.strip() for i in f.readlines() if len(i.strip()) != 0]
    return [json.loads(i) for i in contents]

def build_data(bert_dict, jieba_dict, train_json):
    train_data = read_dialog(train_json)
    data_text = []
    for sample in tqdm(train_data, desc="preprocess :"):
        data_text.extend(preprocess_train_data(sample))
    print("| Before build data size {}".format(len(data_text)))
    data_text = filter(lambda x:max(x["uid"])<12, data_text)
    data = []
    for d in tqdm(data_text, desc="Build :"):
        data.append(build_train_data(d, bert_dict, jieba_dict))
    output = train_json[:train_json.rindex(".")]+"-bert-jieba.pk"
    with open(output, "wb") as f:
        pickle.dump(data, f)

def main():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="", help="Path or url of the dataset.")
    parser.add_argument("--tokenizer", type=str, help="Path of the vocab")
    parser.add_argument('--train', action='store_true', default=False, help='if set, train dataset')
    args = parser.parse_args()

    jieba_dict = JiebaTokenizer(os.path.join(args.tokenizer, "vocab_jieba.txt"))
    bert_dict = BertTokenizer.build_tokenizer(os.path.join(args.tokenizer, "vocab_bert.txt"),do_lower_case=True)
    if args.train:
        build_data(bert_dict, jieba_dict, args.data)
    else:
        data_json = read_dialog(args.data)
        data_text = preprocess_valid_data(data_json)
        data = []
        for d in tqdm(data_text, desc="Build :"):
            data.append(build_train_data(d, bert_dict, jieba_dict))
        output = args.data[:args.data.rindex(".")]+"-bert-jieba.pk"
        with open(output, "wb") as f:
            pickle.dump(data, f)



if __name__ == "__main__":
    main()
    


