#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from monmt.utils.logging import init_logger
from monmt.utils.misc import split_corpus
from monmt.translate.translator import build_translator
from monmt.utils.parse import ArgumentParser
import monmt.opts as opts
import os


def remake_dir(dir, models):
	dir_name = os.path.dirname(dir)
	file_name = os.path.basename(dir)
	model_name =os.path.basename(models[0])
	real_dir = dir_name + '/' + model_name + '/'
	return real_dir + file_name


def main(opt):
	ArgumentParser.validate_translate_opts(opt)

	opt.log_file = remake_dir(opt.log_file, opt.models)
	opt.src2 = remake_dir(opt.src2, opt.models)
	opt.output = remake_dir(opt.output, opt.models)

	logger = init_logger(opt.log_file)
	logger.warning(opt.description)
	logger.info(opt.log_file)

	translator = build_translator(opt, report_score=True, logger=logger)
	logger.info('Building test data...')

	src_shards = split_corpus(opt.src, opt.shard_size)
	tmpl_shards = split_corpus(opt.tmpl, opt.shard_size) \
		if opt.tmpl is not None else [None] * opt.shard_size
	if len(opt.models) == 2:
		src2_shards = split_corpus(opt.src2, opt.shard_size)
		tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
			if opt.tgt is not None else [None] * opt.shard_size
	else:
		src2_shards = [None] * opt.shard_size
		tgt_shards = [None] * opt.shard_size
	shard_pairs = zip(src_shards, tmpl_shards, src2_shards, tgt_shards)

	for i, (src_shard, tmpl_shard, src2_shard, tgt_shard) in enumerate(shard_pairs):
		logger.info("Translating shard %d." % i)
		translator.translate(
			src=src_shard,
			tmpl=tmpl_shard,
			src2=src2_shard,
			tgt=tgt_shard,
			src_seq_len=opt.max_src_len,
			tgt_seq_len=opt.max_tgt_len,
			batch_size=opt.batch_size,
			attn_debug=opt.attn_debug,
		)


def translate_them_all(opt):
	model_path = os.path.dirname(opt.models[0])
	pt_name = os.path.basename(opt.models[0])
	import re
	model_name = re.findall('[A-Za-z0-9\-]+_step', pt_name)[0]

	src2 = opt.src2
	log_file = opt.log_file
	output = opt.output

	for i in range(0, 1000000, 1000):
		_model_name = '%s/%s_%d.pt' % (model_path, model_name, i)
		if not os.path.exists(_model_name): continue
		opt.src2 = src2
		opt.log_file = log_file
		opt.output = output
		if opt.pass1:
			opt.models = [_model_name]
		else:
			opt.models = [_model_name, _model_name.replace('-1_step', '-2_step')]
		main(opt)


if __name__ == "__main__":
	parser = ArgumentParser(description='translate.py')

	opts.config_opts(parser)
	opts.translate_opts(parser)

	opt = parser.parse_args()
	# main(opt)
	translate_them_all(opt)