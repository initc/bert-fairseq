# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion
import torch


@register_criterion('ner_acc_cross_entropy')
class NERCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx,
                          reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        max_args = net_output.argmax(-1)
        target = model.get_targets(sample, net_output)
        correct_num, predict_num, target_num = model.info_acc(max_args.cpu(), target.cpu())
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'correct': correct_num,
            'predict': predict_num,
            'target':target_num,
            'training':model.training
        }
        return loss, sample_size, logging_output

    def acc(self, predict, target):
        b, t = predict.size()
        acc = [1 for a,b in zip(predict, target) if torch.equal(a, b)]
        return len(acc), b


    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        global global_lens
        global global_acc
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)


        if logging_outputs[0]['training']:
            agg_output = {
                'loss': loss_sum / sample_size / math.log(2),
                'ntokens': ntokens,
                'nsentences': nsentences,
                'sample_size': sample_size
            }
        else:
            correct_num = sum(log.get('correct', 0) for log in logging_outputs)
            predict_num = sum(log.get('predict', 0) for log in logging_outputs)
            target_num = sum(log.get('target', 0) for log in logging_outputs)
            agg_output = {
                'loss': loss_sum / sample_size / math.log(2),
                'ntokens': ntokens,
                'nsentences': nsentences,
                'sample_size': sample_size,
                'correct': correct_num,
                'predict': predict_num,
                'target': target_num
            }


        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
