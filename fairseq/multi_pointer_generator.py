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
        self, models
    ):
       
        self.models = models

        self.hit_one = 0
        self.hit_two = 0
        self.hit_three = 0
        self.hit_four = 0
        self.hit_five = 0
        self.answer_n = 0
        self.no_answer = 0
        
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

            with torch.no_grad():
                probs = self.generate(
                    input
                )
                self.record(probs, targets)
            for i, id in enumerate(s['id'].data):
                yield id, probs[i]

    def generate(self, encoder_input):

        with torch.no_grad():
            net_outputs = []
            for model in self.models:
                net_outputs.append(model(**encoder_input))
            net_output = net_outputs[0]
            probs = model.get_normalized_probs(net_output, log_probs=False)
            return probs

    def eval_report(self):
        str_report = "**RESULT**  hit_one {}, hit_two {}, hit_three {}, hit_four {}, hit_five {}, answerable {}, non-answerable {}".format(self.hit_one/self.answer_n, self.hit_two/self.answer_n, self.hit_three/self.answer_n, self.hit_four/self.answer_n, self.hit_five/self.answer_n, self.answer_n, self.no_answer)
        return str_report

    def record(self, probs, targets):
        probs = probs.detach().cpu()
        targets = targets.detach().cpu()
        hit_one, hit_two, hit_three, hit_four, hit_five, answer_n, no_answer = self.validation(targets, probs)
        self.hit_one += hit_one
        self.hit_two += hit_two
        self.hit_three += hit_three
        self.hit_four += hit_four
        self.hit_five += hit_five
        self.answer_n += answer_n
        self.no_answer += no_answer

    def validation(self, target, probs):
        target = target.detach().cpu()
        probs = probs.detach().cpu()
        target = target.numpy()
        probs = probs.numpy()
        return self.get_histest_score(target, probs)

    def get_histest_score(self, targets, probs):
        hit_one = 0
        hit_two = 0
        hit_three = 0
        hit_four = 0
        hit_five = 0
        answer_n = 0

        no_answer = 0
        for t,p in zip(targets, probs):
            if sum(t) == 0:
                no_answer += 1
                continue
            answer_n += 1
            indic = np.argsort(-p)
            if t[indic[0]] == 1:
                hit_one += 1
            if t[indic[0]]==1 or t[indic[1]]==1:
                hit_two += 1
            if t[indic[0]]==1 or t[indic[1]]==1 or t[indic[2]]==1:
                hit_three += 1
            if t[indic[0]]==1 or t[indic[1]]==1 or t[indic[2]]==1 or t[indic[3]]==1:
                hit_four += 1 
            if t[indic[0]]==1 or t[indic[1]]==1 or t[indic[2]]==1 or t[indic[3]]==1 or t[indic[4]]==1:
                hit_five += 1
        return hit_one, hit_two, hit_three, hit_four, hit_five, answer_n, no_answer    

    
