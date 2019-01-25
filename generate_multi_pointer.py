#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import data, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.bert_generator import SequenceGenerator
import pickle

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    generate_file = args.generate_file
    if generate_file is not None:
        generate_file = open(generate_file, "wb")

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)
    print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _ = utils.load_ensemble_for_inference(args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides))


    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        required_batch_size_multiple=8,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()

    translator = SequenceGenerator(
        models
    )

    if use_cuda:
        translator.cuda()

    # Generate and compute BLEU score
    # scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    generate_out = {}
    with progress_bar.build_progress_bar(args, itr) as t:

        translations = translator.generate_batched_itr(
            t,cuda=use_cuda
        )

        for sample_id, hypos in translations:

            src_str = task.dataset(args.gen_subset).get_origin_text(sample_id.item())

            if not args.quiet:
                print('ID-{}\tQUERY--{}\t{}'.format(sample_id.item(), src_str, hypos.tolist()))
            da = {"probs":hypos.tolist(), "query":src_str}
            generate_out[str(sample_id.item())] = da
            num_sentences += 1
            if num_sentences % args.log_interval == 0:
                print("| {}".format(translator.eval_report()))
    if generate_file is not None:
        pickle.dump(generate_out, generate_file)

    print('| Msmarco {} sentences'.format(
        num_sentences))
    print("| Msmarco reutl -- {}".format(translator.eval_report()))


if __name__ == '__main__':
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
