import pickle
import json

from fairseq.data import indexed_dataset
import numpy as np
import torch
import torch.nn.functional as F
import random
import fairseq.tokenizer_bert as tokenizer_bert
tokenizer = tokenizer_bert.BertTokenizer.build_tokenizer("/nas/qsj/bert-model/bert-base-uncased/vocab_file.txt", True)


def convert_marco_eval(cand_in, cand_map, ref_in, cand_out="candidate-file.json", ref_out="reference-file.json"):
    """
    cand_in pickle datamc: dict[str-dict]
    cand_map pickle data: dict[str-str]
    ref-in str: line-json
    """
    candidate_data = pickle.load(open(cand_in, "rb"))
    candidate_map = pickle.load(open(cand_map, "rb"))
    reference = open(ref_in)
    reference_data = {}
    for line in reference:
        l = json.loads(line)
        query_id = l["query_id"]
        reference_data[str(query_id)] = l
    candidate_out = open(cand_out, "w")
    reference_out = open(ref_out, "w")
    for i,index in candidate_map.items():
        g_d = candidate_data[i]
        if index not in reference_data:
            raise Exception("can not find {}_index".format(index))
        r_d = reference_data[index]
        c_d = {"query_id":index, "answers":[g_d["g-target"]]}
        candidate_out.write(json.dumps(c_d)+"\n")
        reference_out.write(json.dumps(r_d)+"\n")
    candidate_out.close()
    reference_out.close()

def convert_json_ids(json_file, out_file="eval_dict.pk"):
    json_data=json.load(open(json_file, "r"))
    has_select = "is_selected" in json_data["passages"]["0"][0]
    has_answers = "answers" in json_data
    keys = list(json_data["passages"].keys())
    data_dict = {}
    for key in keys:
        d = {}
        d["ids"] = key
        answerable = True
        if has_answers:
            ans = [a.strip()=="" for a in json_data["answers"][key]]
            if all(ans):
                continue
            if json_data["answers"][key][0]=="No Answer Present.":
                answerable = False
        d["answerable"] = answerable
        ids_passages = []
        selected = []
        for p in json_data["passages"][key]:
            text = p["passage_text"]
            if has_select:
                selected.append(p["is_selected"])
            else:
                selected.append(1)
            ids_passages.append(tokenizer.convert_text_to_ids(text))
        d["ids_passages"] = ids_passages
        d["selected"] = selected
        ids_query = tokenizer.convert_text_to_ids(json_data["query"][key])
        d["ids_query"] = ids_query

        ids_answers = []
        if has_answers:
            for a in json_data["answers"][key]:
                if a.strip()=="":
                    continue
                ids_answers.append(tokenizer.convert_text_to_ids(a))
        else:
            ids_answers.append(tokenizer.convert_text_to_ids("fake answers"))

        d["ids_answers"] = ids_answers
        d["query_id"] = str(json_data["query_id"][key])

        data_dict[key] = d

    out_f = open(out_file, "wb")
    pickle.dump(data_dict, out_f)


def make_select_databin(data_ids, prefix_dir="eval-select-bin", prefix_name="eval"):
    # prefix = "valid"
    def consumer(ds, ids):
        ds.add_item(torch.IntTensor(ids))
    
    def consumer_float(ds, ids):
        ds.add_item(torch.FloatTensor(ids))

    query_bin = "/nas/qsj/data-bin-v2/{}/{}-query.bin".format(prefix_dir, prefix_name)
    query_idx = "/nas/qsj/data-bin-v2/{}/{}-query.idx".format(prefix_dir, prefix_name)

    passage_bin = "/nas/qsj/data-bin-v2/{}/{}-passage.bin".format(prefix_dir, prefix_name)
    passage_idx = "/nas/qsj/data-bin-v2/{}/{}-passage.idx".format(prefix_dir, prefix_name)


    target_bin = "/nas/qsj/data-bin-v2/{}/{}-target.bin".format(prefix_dir, prefix_name)
    target_idx = "/nas/qsj/data-bin-v2/{}/{}-target.idx".format(prefix_dir, prefix_name)

    query_ds = indexed_dataset.IndexedDatasetBuilder(query_bin)

    passage_ds = indexed_dataset.IndexedDatasetBuilder(passage_bin)

    target_ds = indexed_dataset.IndexedDatasetBuilder(target_bin, dtype=np.float)

    id_to_index_file = open("/nas/qsj/data-bin-v2/{}/{}-id-to-index.pk".format(prefix_dir, prefix_name), "wb")
    id_to_index = {}

    data_dict = pickle.load(open(data_ids, "rb"))
    id_ = 0

    above_10 = 0
    below_10 = 0
    for key,data in data_dict.items():
        id_to_index[str(id_)] = key
        id_ += 1
        ids_query = data["ids_query"]
        ids_passages = data["ids_passages"]
        ids_selected = data["selected"]
        assert len(ids_passages)==len(ids_selected)

        len_ = len(ids_passages)
        if len_ > 10:
            ids_passages = ids_passages[:10]
            ids_selected = ids_selected[:10]
            above_10 += 1
        if len_ < 10:
            for _ in range(len_, 10):
                ids_passages.append(ids_passages[-1])
                ids_selected.append(ids_selected[-1])
                below_10 += 1
        assert len(ids_passages)==10
        assert len(ids_passages)==len(ids_selected)
        for ids_p in ids_passages:
            consumer(query_ds, ids_query)
            consumer(passage_ds, ids_p)
        if prefix_name == "train":
            total_sum = sum(ids_selected)
            if total_sum > 1:
                for i in range(10):
                    ids_selected[i] /= total_sum
        consumer_float(target_ds, ids_selected)

    query_ds.finalize(query_idx)
    passage_ds.finalize(passage_idx)
    target_ds.finalize(target_idx)

    pickle.dump(id_to_index, id_to_index_file)
    print("| above_10 {}, below_10 {}".format(above_10, below_10))


def get_datas():
    train_data = open("/nas/qsj/corpus/msmarco/multi-answer-msmarco/new-train-msmarco-span-rougel-F1.pk", "rb")
    train_data = pickle.load(train_data)
    train_ids_data = {}

    for d in train_data:
        idx = d["idx"]
        train_ids_data[idx] = d

    train_o_file = open("/nas/qsj/corpus/msmarco/format_train_v2.1.json")
    train_o_data = json.load(train_o_file)

    train_index_to_probs_file = open("/nas/qsj/data/msmarco-probs/train-index-to-probs.pk", "rb")
    train_index_to_probs = pickle.load(train_index_to_probs_file)

    well_form_keys = {}

    for key,value in train_o_data["wellFormedAnswers"].items():
        if isinstance(value, list):
            well_form_keys[key] = [tokenizer.convert_text_to_ids(t) for t in value]

    count = 0
    well_form_data = []
    for key,value in well_form_keys.items():
        if key not in train_ids_data or key not in train_index_to_probs:
            count += 1
            continue
        if train_ids_data[key]["is_error"]:
            count += 1
            continue
        ids_data = train_ids_data[key]
        for v in value:
            d = {}
            d["idx"] = key
            d["ids_passages"] = ids_data["ids_passages"]
            d["ids_answer"] = v
            d["ids_query"] = [0]+ids_data["ids_query"]
            d["probs"] = train_index_to_probs[key]
            well_form_data.append(d)
            well_form_data.append(d.copy())
            well_form_data.append(d.copy())

    qa_data = []

    count = 0
    for key,value in train_ids_data.items():
        if key not in train_index_to_probs:
            count += 1
            continue
        if value["is_error"]:
            count += 1
            continue
        for ans in train_o_data["answers"][key]:
            d = {}
            d["idx"] = key
            d["ids_passages"] = value["ids_passages"]
            d["ids_answer"] = tokenizer.convert_text_to_ids(ans)
            d["ids_query"] = [1] + value["ids_query"]
            d["probs"] = train_index_to_probs[key]
            qa_data.append(d)

    all_data = []
    all_data.extend(well_form_data)
    all_data.extend(qa_data)

    random.shuffle(all_data)
    random.shuffle(all_data)

    def probs_select(data):
        probs = np.array(data["probs"])
        return np.argsort(probs)[::-1].tolist()
    
    passage_re = []
    query_re = []
    answer_re = []

    for d in all_data:
        probs = d["probs"]
        max_prob_index = probs_select(probs)[:5]

        top_one = max_prob_index[0]
        top_two = max_prob_index[1]
        top_three = max_prob_index[2]
        top_four = max_prob_index[3]
        top_five = max_prob_index[4]

        top_one = min(top_one, len(d["ids_passages"])-1)
        top_two = min(top_two, len(d["ids_passages"])-1)
        top_three = min(top_three, len(d["ids_passages"])-1)
        top_four = min(top_four, len(d["ids_passages"])-1)
        top_five = min(top_five, len(d["ids_passages"])-1)
        
        ids_passage_one = d["ids_passages"][top_one]
        ids_passage_two = d["ids_passages"][top_two]
        ids_passage_three = d["ids_passages"][top_three]
        ids_passage_four = d["ids_passages"][top_four]
        ids_passage_five = d["ids_passages"][top_five]

        ids_query = d["ids_query"]

        ids_answer = d["ids_answer"]

        passage_re.append([ids_passage_one, ids_passage_two, ids_passage_three, ids_passage_four, ids_passage_five])

        query_re.append(ids_query)
        answer_re.append(ids_answer)


    return passage_re, query_re, answer_re
    



def augument_shffle_top5(passages, queries, answers, prefix="train", dir_file="top-5-qa+nlg+shuffle"):

    def consumer(ds, ids):
        ds.add_item(torch.IntTensor(ids))

    query_bin = "/nas/qsj/data-bin-v2/{}/{}-query.bin".format(dir_file, prefix)
    query_idx = "/nas/qsj/data-bin-v2/{}/{}-query.idx".format(dir_file, prefix)

    passage_1_bin = "/nas/qsj/data-bin-v2/{}/{}-passage-1.bin".format(dir_file, prefix)
    passage_1_idx = "/nas/qsj/data-bin-v2/{}/{}-passage-1.idx".format(dir_file, prefix)

    passage_2_bin = "/nas/qsj/data-bin-v2/{}/{}-passage-2.bin".format(dir_file, prefix)
    passage_2_idx = "/nas/qsj/data-bin-v2/{}/{}-passage-2.idx".format(dir_file, prefix)

    passage_3_bin = "/nas/qsj/data-bin-v2/{}/{}-passage-3.bin".format(dir_file, prefix)
    passage_3_idx = "/nas/qsj/data-bin-v2/{}/{}-passage-3.idx".format(dir_file, prefix)

    passage_4_bin = "/nas/qsj/data-bin-v2/{}/{}-passage-4.bin".format(dir_file, prefix)
    passage_4_idx = "/nas/qsj/data-bin-v2/{}/{}-passage-4.idx".format(dir_file, prefix)

    passage_5_bin = "/nas/qsj/data-bin-v2/{}/{}-passage-5.bin".format(dir_file, prefix)
    passage_5_idx = "/nas/qsj/data-bin-v2/{}/{}-passage-5.idx".format(dir_file, prefix)

    target_bin = "/nas/qsj/data-bin-v2/{}/{}-target.bin".format(dir_file, prefix)
    target_idx = "/nas/qsj/data-bin-v2/{}/{}-target.idx".format(dir_file, prefix)

    query_ds = indexed_dataset.IndexedDatasetBuilder(query_bin)

    passage_1_ds = indexed_dataset.IndexedDatasetBuilder(passage_1_bin)

    passage_2_ds = indexed_dataset.IndexedDatasetBuilder(passage_2_bin)

    passage_3_ds = indexed_dataset.IndexedDatasetBuilder(passage_3_bin)

    passage_4_ds = indexed_dataset.IndexedDatasetBuilder(passage_4_bin)

    passage_5_ds = indexed_dataset.IndexedDatasetBuilder(passage_5_bin)

    target_ds = indexed_dataset.IndexedDatasetBuilder(target_bin)

    for passage, query, answer in zip(passages, queries, answers):
        passage_shuffle = passage[2:]

        consumer(query_ds, query)
        consumer(passage_1_ds, passage[0])
        consumer(passage_2_ds, passage[1])
        consumer(passage_3_ds, passage_shuffle[0])
        consumer(passage_4_ds, passage_shuffle[1])
        consumer(passage_5_ds, passage_shuffle[2])
        consumer(target_ds, answer)

        # random.shuffle(passage_shuffle)
        # consumer(query_ds, query)
        # consumer(passage_1_ds, passage[0])
        # consumer(passage_2_ds, passage[1])
        # consumer(passage_3_ds, passage_shuffle[0])
        # consumer(passage_4_ds, passage_shuffle[1])
        # consumer(passage_5_ds, passage_shuffle[2])
        # consumer(target_ds, answer)
    query_ds.finalize(query_idx)
    passage_1_ds.finalize(passage_1_idx)
    passage_2_ds.finalize(passage_2_idx)
    passage_3_ds.finalize(passage_3_idx)
    passage_4_ds.finalize(passage_4_idx)
    passage_5_ds.finalize(passage_5_idx)
    target_ds.finalize(target_idx)


def make_top_5(select_probs, index_to_ids, ids_data, query_type, prefix, dir_file):

    def probs_select(data):
        probs = np.array(data)
        return np.argsort(probs)[::-1].tolist()

    if isinstance(select_probs, str):
        with open(select_probs, "rb") as f:
            select_probs= pickle.load(f)

    if isinstance(index_to_ids, str):
        with open(index_to_ids, "rb") as f:
            index_to_ids= pickle.load(f)

    if isinstance(ids_data, str):
        with open(ids_data, "rb") as f:
            ids_data= pickle.load(f)

    passage_re = []
    query_re = []
    answer_re = []

    for index, ids in index_to_ids.items():
        probs = select_probs[index]["probs"]

        d = ids_data[ids]


        max_prob_index = probs_select(probs)[:5]

        top_one = max_prob_index[0]
        top_two = max_prob_index[1]
        top_three = max_prob_index[2]
        top_four = max_prob_index[3]
        top_five = max_prob_index[4]

        top_one = min(top_one, len(d["ids_passages"])-1)
        top_two = min(top_two, len(d["ids_passages"])-1)
        top_three = min(top_three, len(d["ids_passages"])-1)
        top_four = min(top_four, len(d["ids_passages"])-1)
        top_five = min(top_five, len(d["ids_passages"])-1)

        ids_passage_one = d["ids_passages"][top_one]
        ids_passage_two = d["ids_passages"][top_two]
        ids_passage_three = d["ids_passages"][top_three]
        ids_passage_four = d["ids_passages"][top_four]
        ids_passage_five = d["ids_passages"][top_five]

        ids_query = [query_type]+d["ids_query"]

        ids_answer = d["ids_answers"][0]

        passage_re.append([ids_passage_one, ids_passage_two, ids_passage_three, ids_passage_four, ids_passage_five])

        query_re.append(ids_query)
        answer_re.append(ids_answer)

    augument_shffle_top5(passage_re, query_re, answer_re, prefix=prefix, dir_file=dir_file)



