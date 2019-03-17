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
import pdb

class SequenceGenerator(object):
    def __init__(
        self, models, task, beam_size, max_lens
    ):
        self.args = task.args
        self.models = models
        self.tokenizer = task.tokenizer
        self.eos = self.tokenizer.eos()
        self.start_idx = None
        self.minlen = 1
        self.pad = self.tokenizer.pad()
        self.vocab_size = len(self.tokenizer)
        self.normalize_scores = True
        self.len_penalty = 1
        self.beam_size = beam_size
        self.max_lens = max_lens
        self.search = search.BeamSearch(self.tokenizer)

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
                tokens = self.generate(
                    input, self.beam_size, self.max_lens
                )
            for i, id in enumerate(s['id'].data):
                yield id, self.merge_bert_tokens(tokens[i]), self.merge_bert_tokens(origin_target[i]), 1

    def generate(self, encoder_input, beam_size, max_lens):

        with torch.no_grad():
            generate_ids = self.generate_beam(encoder_input, beam_size, max_lens)
            tokens = self.convert_to_tokens(generate_ids)
            return tokens

    def generate_beam(self, encoder_input, beam_size=None, maxlen=None, prefix_tokens=None):
        """Generate a batch of translations.

        Args:
            encoder_input (dict): dictionary containing the inputs to
                *model.encoder.forward*.
            beam_size (int, optional): overriding the beam size
                (default: *self.beam_size*).
            max_len (int, optional): maximum length of the generated sequence
            prefix_tokens (LongTensor, optional): force decoder to begin with
                these tokens
        """
        return self._generate(encoder_input, beam_size, maxlen, prefix_tokens)

    def _generate(self, encoder_input, beam_size=None, maxlen=None, prefix_tokens=None):
        """See generate"""
        src_tokens = encoder_input['input_ids']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        bsz, srclen = src_tokens.size()
        # maxlen = min(maxlen, self.maxlen) if maxlen is not None else self.maxlen

        # the max beam size is the dictionary size - 1, since we never select pad
        beam_size = beam_size if beam_size is not None else self.beam_size
        beam_size = min(beam_size, self.vocab_size - 1)

        encoder_outs = []
        incremental_states = {}
        for model in self.models:
            model.eval()
            # if isinstance(model.decoder, FairseqIncrementalDecoder):
            incremental_states[model] = {}
            # else:
            #     incremental_states[model] = None

            # compute the encoder output for each beam
            encoder_out, start_idx = model.encoder_out(**encoder_input)
            self.start_idx = start_idx
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
            new_order = new_order.to(src_tokens.device).long()
            encoder_out = model.encoder.reorder_encoder_out(encoder_out, new_order)
            encoder_outs.append(encoder_out)

        # initialize buffers
        scores = src_tokens.data.new(bsz * beam_size, maxlen + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.data.new(bsz * beam_size, maxlen + 2).fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.start_idx
        # attn, attn_buf = None, None
        nonpad_idxs = None

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        # worst_finalized = [{'idx': None, 'score': -math.inf} for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                # if self.stop_early or step == maxlen or unfinalized_scores is None:
                return True
                # stop if the best unfinalized score is worse than the worst
                # finalized one
                # best_unfinalized_score = unfinalized_scores[sent].max()
                # if self.normalize_scores:
                #     best_unfinalized_score /= maxlen ** self.len_penalty
                # if worst_finalized[sent]['score'] >= best_unfinalized_score:
                #     return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            tokens_clone[:, step] = self.eos
            # attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                # if self.match_source_len and step > src_lengths[unfin_idx]:
                #     score = -math.inf

                def get_hypo():

                    # if attn_clone is not None:
                    #     # remove padding tokens from attn scores
                    #     hypo_attn = attn_clone[i][nonpad_idxs[sent]]
                    #     _, alignment = hypo_attn.max(dim=0)
                    # else:
                    #     hypo_attn = None
                    #     alignment = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        # 'attention': hypo_attn,  # src_len x tgt_len
                        # 'alignment': alignment,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                # elif not self.stop_early and score > worst_finalized[sent]['score']:
                #     # replace worst hypo for this sentence with new/better one
                #     worst_idx = worst_finalized[sent]['idx']
                #     if worst_idx is not None:
                #         finalized[sent][worst_idx] = get_hypo()

                #     # find new worst finalized hypo for this sentence
                #     idx, s = min(enumerate(finalized[sent]), key=lambda r: r[1]['score'])
                #     worst_finalized[sent] = {
                #         'score': s['score'],
                #         'idx': idx,
                #     }

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfinalized_scores):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(maxlen + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                for i, model in enumerate(self.models):
                    # if isinstance(model.decoder, FairseqIncrementalDecoder):
                    # pdb.set_trace()
                    model.decoder.reorder_incremental_state(incremental_states[model], reorder_state)
                    encoder_outs[i] = model.encoder.reorder_encoder_out(encoder_outs[i], reorder_state)
            # pdb.set_trace()
            lprobs = self._decode(tokens[:, :step + 1], encoder_outs, incremental_states, step=step, max_lens = maxlen)

            lprobs[:, self.pad] = -math.inf  # never select pad
            # lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # if self.no_repeat_ngram_size > 0:
            #     # for each beam and batch sentence, generate a list of previous ngrams
            #     gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
            #     for bbsz_idx in range(bsz * beam_size):
            #         gen_tokens = tokens[bbsz_idx].tolist()
            #         for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
            #             gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
            #                     gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            # if avg_attn_scores is not None:
            #     if attn is None:
            #         attn = scores.new(bsz * beam_size, src_tokens.size(1), maxlen + 2)
            #         attn_buf = attn.clone()
            #         nonpad_idxs = src_tokens.ne(self.pad)
            #     attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < maxlen:
                self.search.set_src_lengths(src_lengths)
                cand_scores, cand_indices, cand_beams = self.search.step(
                    step,
                    lprobs.view(bsz, -1, self.vocab_size),
                    scores.view(bsz, beam_size, -1)[:, :, :step],
                )
            else:
                # make probs contain cumulative scores for each hypothesis
                lprobs.add_(scores[:, step - 1].unsqueeze(-1))

                # finalize all active hypotheses once we hit maxlen
                # pick the hypothesis with the highest prob of EOS right now
                torch.sort(
                    lprobs[:, self.eos],
                    descending=True,
                    out=(eos_scores, eos_bbsz_idx),
                )
                num_remaining_sent -= len(finalize_hypos(step, eos_bbsz_idx, eos_scores))
                assert num_remaining_sent == 0
                break

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            eos_mask = cand_indices.eq(self.eos)

            finalized_sents = set()
            if step >= self.minlen:
                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores, cand_scores)
                    num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            # 提前结束
            if num_remaining_sent == 0:
                break
            assert step < maxlen

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                # 只保留new_bsz个原属
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                # if prefix_tokens is not None:
                #     prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                # if attn is not None:
                #     attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                #     attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos
            active_mask = buffer('active_mask')
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            # 去除了eos后，最靠前的K个
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(_ignore, active_hypos)
            )

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)
            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            # if attn is not None:
            #     torch.index_select(
            #         attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
            #         out=attn_buf[:, :, :step + 2],
            #     )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            # if attn is not None:
            #     attn, attn_buf = attn_buf, attn
            # pdb.set_trace()
            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            sorted_tokens = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
            finalized[sent] = sorted_tokens[0]["tokens"].tolist()

        return finalized

    def _decode(self, tokens, encoder_outs, incremental_states, step, max_lens):
        if len(self.models) == 1:
            return self._decode_one(tokens, self.models[0], encoder_outs[0], incremental_states, step=step, log_probs=True, max_lens=max_lens)

        log_probs = []
        # avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs = self._decode_one(tokens, model, encoder_out, incremental_states, step=step, log_probs=True, max_lens=max_lens)
            log_probs.append(probs)
            # if attn is not None:
            #     if avg_attn is None:
            #         avg_attn = attn
            #     else:
            #         avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        # if avg_attn is not None:
        #     avg_attn.div_(len(self.models))
        return avg_probs

    def _decode_one(self, tokens, model, encoder_out, incremental_states, step, log_probs, max_lens):
        with torch.no_grad():
            decoder_out = model.decoder_one_step(tokens, encoder_out, incremental_states[model], step, max_lens)

            decoder_out = decoder_out[:, -1, :]
            # attn = decoder_out[1]
            # if type(attn) is dict:
            #     attn = attn['attn']
            # if attn is not None:
            #     if type(attn) is dict:
            #         attn = attn['attn']
            #     attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        return probs

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





