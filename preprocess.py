#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

import configargparse
import glob
import sys
import gc
import os
import codecs
import torch
import cjjpy as cjj

from monmt.utils.misc import split_corpus
from monmt.utils.logging import init_logger, logger
import monmt.inputters as inputters
import monmt.opts as opts


def check_existing_pt_files(opt):
	""" Check if there are existing .pt files to avoid overwriting them """
	pattern = opt.save_data + '.{}*.pt'
	for t in ['train', 'valid', 'vocab']:
		path = pattern.format(t)
		if glob.glob(path):
			sys.stderr.write("Please backup existing pt files: %s, "
			                 "to avoid overwriting them!\n" % path)
			sys.exit(1)


def parse_args():
	parser = configargparse.ArgumentParser(
		description='preprocess.py',
		config_file_parser_class=configargparse.YAMLConfigFileParser,
		formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

	opts.config_opts(parser)
	opts.preprocess_opts(parser)

	opt = parser.parse_args()
	torch.manual_seed(opt.seed)

	check_existing_pt_files(opt)

	return opt


def build_save_dataset(corpus_type, fields, opt):
	assert corpus_type in ['train', 'valid']

	logger.info("Reading source and target from kb.")
	dataset_paths = []

	if corpus_type == 'train':
		_src = opt.train_src
		_tmpl = opt.train_tmpl
		_src2 = opt.train_src2
		_tgt = opt.train_tgt
	else:
		_src = opt.valid_src
		_tmpl = opt.valid_tmpl
		_src2 = opt.valid_src2
		_tgt = opt.valid_tgt

	src_shards = split_corpus(_src, opt.shard_size)
	tmpl_shards = split_corpus(_tmpl, opt.shard_size)
	src2_shards = split_corpus(_tmpl, opt.shard_size)
	tgt_shards = split_corpus(_tgt, opt.shard_size)

	shard_pairs = zip(src_shards, tmpl_shards, src2_shards, tgt_shards)

	for i, (src_shard, tmpl_shard, src2_shard, tgt_shard) in enumerate(shard_pairs):
		assert len(src_shard) == len(tgt_shard) == len(tmpl_shard)
		logger.info("Building shard %d." % i)
		dataset = inputters.build_dataset(
			fields,
			src=src_shard,
			tgt=tgt_shard,
			src2=src2_shard,
			tmpl=tmpl_shard,
			src_seq_len=opt.max_src_len,
			tmpl_seq_len=opt.max_tgt_len,
			src2_seq_len=opt.max_tgt_len,
			tgt_seq_len=opt.max_tgt_len,
			use_filter_pred=False
		)

		data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, corpus_type, i)
		cjj.MakeDir(os.path.dirname(data_path))
		dataset_paths.append(data_path)

		logger.info(" * saving %sth %s data shard to %s."
		            % (i, corpus_type, data_path))

		dataset.save(data_path)

		del dataset.examples
		gc.collect()
		del dataset
		gc.collect()

	return dataset_paths


def build_save_vocab(train_dataset, fields, opt):
	fields = inputters.build_vocab(
		train_dataset, fields, opt.share_vocab,
		opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
		opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency
	)
	vocab_path = opt.save_data + '.vocab.pt'
	torch.save(fields, vocab_path)


def count_features(path):
	"""
	path: location of a corpus file with whitespace-delimited tokens and
					￨-delimited features within the token
	returns: the number of features in the dataset
	"""
	with codecs.open(path, "r", "utf-8") as f:
		first_tok = f.readline().split(None, 1)[0]
		return len(first_tok.split(u"￨")) - 1


def main(opt):

	init_logger(opt.log_file)
	logger.info("Extracting features...")

	src_nfeats = count_features(opt.train_src)
	tgt_nfeats = count_features(opt.train_tgt)  # tgt always text so far
	src2_nfeats = count_features(opt.train_src2)
	tmpl_nfeats = count_features(opt.train_tmpl)
	logger.info(" * number of source features: %d." % src_nfeats)
	logger.info(" * number of template features: %d." % tmpl_nfeats)
	logger.info(" * number of source2 features: %d." % src2_nfeats)
	logger.info(" * number of target features: %d." % tgt_nfeats)

	logger.info("Building `Fields` object...")
	fields = inputters.get_fields(n_src_feats=src_nfeats,
	                              n_tmpl_feats=tmpl_nfeats,
	                              n_src2_feats=src2_nfeats,
	                              n_tgt_feats=tgt_nfeats)

	logger.info("Building & saving training data...")
	train_datasets = build_save_dataset('train', fields, opt)

	logger.info("Building & saving validation data...")
	build_save_dataset('valid', fields, opt)

	logger.info("Building & saving vocabulary...")
	build_save_vocab(train_datasets, fields, opt)


if __name__ == "__main__":
	parser = configargparse.ArgumentParser(description='preprocess.py')

	opts.config_opts(parser)
	opts.preprocess_opts(parser)
	opt = parse_args()

	logger.warning(opt.description)
	main(opt)
