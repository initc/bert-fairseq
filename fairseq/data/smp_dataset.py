# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from fairseq import utils

from . import data_utils, FairseqDataset

def collate(
    samples, encoder_dict, decoder_dict, max_history
):
    if len(samples) == 0:
        print("hahahaha")
        return {}
    batch_size = len(samples)
    def convert_bert(dialog, profile, uid):
        if max_history !=-1:
            dialog = dialog[-max_history:]
            uid = dialog[-max_history:]
        word_idx = [encoder_dict.cls()]+profile[0]+profile[1]+profile[2]+[encoder_dict.sep()]
        token_type = [0]*len(word_idx)
        position_ids = list(range(len(word_idx)))
        for dia, _uid in zip(dialog, uid):
            word_idx += dia+[encoder_dict.sep()]
            token_type += [uid[_uid]+1]*(len(dia)+1)
        position_ids += list(range(len(word_idx)-len(position_ids)))
        attention_mask = [1]*len(position_ids)
        return word_idx, token_type, position_ids, attention_mask

    def convert_jieba(response):
        target = response+[decoder_dict.eos()]
        prev_output_tokens = [decoder_dict.eos()]+response
        return prev_output_tokens, target


    encoder_idxs = [convert_bert(s["dialog_history"], s["profile"], s["uid"]) for s in samples]
    decoder_idxs = [convert_jieba(s["response_jieba"]) for s in samples]
    max_encoder = max(len(d[0]) for d in encoder_idxs)
    max_decoder = max(len(d[0]) for d in decoder_idxs)

    input_ids = torch.LongTensor(batch_size, max_encoder).fill_(encoder_dict.pad())
    token_type_ids = torch.LongTensor(batch_size, max_encoder).fill_(encoder_dict.pad())
    attention_mask = torch.LongTensor(batch_size, max_encoder).fill_(encoder_dict.pad())
    position_ids = torch.LongTensor(batch_size, max_encoder).fill_(511)
    prev_output_tokens = torch.LongTensor(batch_size, max_decoder).fill_(decoder_dict.pad())
    target = torch.LongTensor(batch_size, max_decoder).fill_(decoder_dict.pad())
    for i, (word_idx, token_type, positions, mask) in enumerate(encoder_idxs):
        word_size = len(word_idx)
        input_ids[i,:word_size] = torch.Tensor(word_idx)
        token_type_ids[i, :word_size] = torch.Tensor(token_type)
        position_ids[i,:word_size] = torch.Tensor(positions)
        attention_mask[i,:word_size] = torch.Tensor(mask)
    ntokens = 0
    for i,(prev_tokens, tgt) in enumerate(decoder_idxs):
        word_size = len(prev_tokens)
        ntokens += word_size
        prev_output_tokens[i,:word_size] = torch.Tensor(prev_tokens)
        target[i,:word_size] = torch.Tensor(tgt)

    batch = {
        'id': id,
        'nsentences': len(samples),
        "ntokens": ntokens,
        'net_input': {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            "attention_mask": attention_mask,
            "prev_output_tokens":prev_output_tokens,
            "position_ids":position_ids,
        },
        "target": target
    }
    return batch

class SMPDataset(FairseqDataset):

    def __init__(
        self, data, 
        encoder_dict, decoder_dict,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, max_history=-1,
    ):
        self.data = data
        self.encoder_dict = encoder_dict
        self.decoder_dict = decoder_dict
        assert encoder_dict.pad()==decoder_dict.pad()
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.max_history = max_history


    def __getitem__(self, index):
        item = self.data[index]
        item["id"] = index
        return item

    def __len__(self):
        return len(self.data)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
        """
        return collate(
            samples, self.encoder_dict, self.decoder_dict, self.max_history
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.max_source_positions, self.max_target_positions),
        )
        bsz = 2
        target = torch.Tensor(tgt_len).uniform_(2, 20).long().tolist()
        return self.collater([
            {
                'id': i,
                'dialog_history': [torch.Tensor(src_len//2).uniform_(2, 20).long().tolist() for _ in range(3)],
                'uid': [0,1,2],
                "profile": torch.Tensor(tgt_len).uniform_(2, 20).long().tolist(),
                "response_jieba":target,
                "response_bert": [[4,5,6] for _ in range(len(target))]
            }
            for i in range(bsz)
        ])


    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        # target_sizes = self.target_sizes[index]
        instance = self.data[index]
        dialog = instance["dialog_history"] if self.max_history==-1 else instance["dialog_history"][-self.max_history:]
        response = instance["response_jieba"]
        return sum([len(d) for d in dialog])+len(response)

    def get_origin_text(self, sample_id):
        pass
        # return self.tokenizer.str_tokens(self.a_data[sample_id])

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        instance = self.data[index]
        dialog = instance["dialog_history"] if self.max_history==-1 else instance["dialog_history"][-self.max_history:]
        response = instance["response_jieba"]
        return max(sum([len(d) for d in dialog]), len(response))

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        lens = []
        for index in indices:
            lens.append(self.num_tokens(index))
        indices = indices[np.argsort(np.array(lens), kind='mergesort')]
        return indices


    @property
    def supports_prefetch(self):
        return (
            hasattr(self.data, 'supports_prefetch')
            and self.data.supports_prefetch
        )
