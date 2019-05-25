# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2019/2/21 13:35
@Contact    : mi0134sher@hotmail.com
@Description: 
'''

import cjjpy as cjj
import os, configargparse, torch, sys
from raw.knowledge_base import kbinp_adjust_to_onmt, KnowledgeBase
import monmt.opts as opts


def make_dataset(kb, opt):
	def makedir(file):
		if file:
			path = os.path.dirname(file)
			cjj.MakeDir(path)

	def write_list(mlist, file):
		with open(file, 'w') as f:
			for line in mlist:
				f.write(line + '\n')

	for i in [opt.train_src, opt.train_tgt, opt.valid_src, opt.valid_tgt]:
		makedir(i)
	src, tgt, tmpl = kbinp_adjust_to_onmt(kb, opt)
	if kb.role == 'train':
		write_list(src, opt.train_src)
		write_list(tgt, opt.train_tgt)
		write_list(tmpl, opt.train_src2)
		write_list(tmpl, opt.train_tmpl)
	elif kb.role == 'valid':
		write_list(src, opt.valid_src)
		write_list(tgt, opt.valid_tgt)
		write_list(tmpl, opt.valid_src2)
		write_list(tmpl, opt.valid_tmpl)
	else:
		write_list(src, opt.test_src)
		write_list(tgt, opt.test_tgt)
		write_list(tmpl, opt.test_tmpl)


def parse_args():
	parser = configargparse.ArgumentParser(
		description='preprocess.py',
		config_file_parser_class=configargparse.YAMLConfigFileParser,
		formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

	opts.config_opts(parser)
	opts.preprocess_opts(parser)

	opt = parser.parse_args()
	torch.manual_seed(opt.seed)

	return opt


if __name__ == '__main__':
	opt = parse_args()
	import re
	data_size = re.findall('[0-9]+K', getattr(opt, 'train_src'))[0]
	prefix = 'raw/infobox/data%s/' % data_size
	trainkb = KnowledgeBase(prefix + 'data%s_train.infobox' % data_size,
	                        prefix + 'data%s.vocab' % data_size, role='train')
	validkb = KnowledgeBase(prefix + 'data%s_valid.infobox' % data_size,
	                        prefix + 'data%s.vocab' % data_size, role='valid')
	testkb = KnowledgeBase(prefix + 'data%s_test.infobox' % data_size,
	                       prefix + 'data%s.vocab' % data_size, role='test')

	make_dataset(trainkb, opt)
	make_dataset(validkb, opt)
	make_dataset(testkb, opt)
