# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import numpy as np
import torch

from fairseq import search, utils
from fairseq.models import FairseqIncrementalDecoder


class SequenceGenerator(object):
    def __init__(
        self, models, task
    ):
        self.args = task.args
        self.models = models
        self.tokenizer = task.tokenizer
        
    def cuda(self):
        for model in self.models:
            model.cuda()
        return self

    def generate_batched_itr(
        self, data_itr, cuda=False
    ):
        """Iterate over a batched dataset and yield individual translations.
        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            s = utils.move_to_cuda(sample) if cuda else sample
            if 'net_input' not in s:
                continue
            input = s['net_input']
            targets = s["target"]
            origin_target = self.convert_to_tokens(targets)

            with torch.no_grad():
                tokens, ppl_probs = self.generate(
                    input
                )
            for i, id in enumerate(s['id'].data):
                yield id, self.merge_bert_tokens(tokens[i]), self.merge_bert_tokens(origin_target[i]), ppl_probs[i]/len(tokens[i])

    def generate(self, encoder_input):

        with torch.no_grad():

            model= self.models[0]
            encoder_out =  model.greedy_generater(**encoder_input, max_lens=self.args.max_tokens_generate)
            generate_ids = encoder_out[0]
            ppl_probs = encoder_out[1]
            tokens = self.convert_to_tokens(generate_ids)
            return tokens, ppl_probs.tolist()

    def convert_to_tokens(self, ids):

        def to_tokens(t):
            return self.tokenizer.convert_ids_to_tokens(t)

        eos_ids = self.tokenizer.eos()
        if not isinstance(ids, list):
            ids = ids.tolist()
        tokens = []
        for item in ids:
            if eos_ids in item:
                eos_index = item.index(eos_ids)
                item = item[:eos_index]
            tokens.append(to_tokens(item))
        return tokens

    def merge_bert_tokens(self, tokens):
        new_tokens = []
        new_tokens.append(tokens[0])
        for tok in tokens[1:]:
            if tok.startswith("##"):
                new_tokens[-1] += tok[2:]
            else:
                new_tokens.append(tok)
        return new_tokens





