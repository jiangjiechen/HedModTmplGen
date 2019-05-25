#!/usr/bin/env python
"""
    Training on a single process
"""

import os
import torch

from monmt.inputters.inputter import build_dataset_iter, \
	load_old_vocab, old_style_vocab
from monmt.model_builder import build_model
from monmt.utils.optimizers import build_optim
from monmt.utils.misc import set_random_seed
from monmt.trainer import build_trainer
from monmt.models import build_model_saver
from monmt.utils.logging import init_logger, logger
from monmt.utils.parse import ArgumentParser


def _check_save_model_path(save_path):
	save_model_path = os.path.abspath(save_path)
	model_dirname = os.path.dirname(save_model_path)
	if not os.path.exists(model_dirname):
		os.makedirs(model_dirname)


def _tally_parameters(model):
	enc = 0
	dec = 0
	for name, param in model.named_parameters():
		if 'encoder' in name:
			enc += param.nelement()
		else:
			dec += param.nelement()
	return enc + dec, enc, dec


def configure_process(opt, device_id):
	if device_id >= 0:
		torch.cuda.set_device(device_id)
	set_random_seed(opt.seed, device_id >= 0)


def main(opt, device_id):
	# NOTE: It's important that ``opt`` has been validated and updated
	# at this point.
	configure_process(opt, device_id)
	init_logger(opt.log_file, from_scratch=opt.from_scratch)
	os.system('cp %s %s' % (opt.config, os.path.dirname(opt.log_file) + '/'))

	logger.warning(opt.description)
	logger.warning('Joint learning' if opt.joint else 'Pipeline learning')
	# Load checkpoint if we resume from a previous training.
	if opt.train_from:
		logger.info('Loading checkpoint from %s' % opt.train_from)
		checkpoint = torch.load(opt.train_from,
		                        map_location=lambda storage, loc: storage)

		model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
		ArgumentParser.update_model_opts(model_opt)
		ArgumentParser.validate_model_opts(model_opt)
		logger.info('Loading vocab from checkpoint at %s.' % opt.train_from)
		vocab = checkpoint['vocab']
	else:
		checkpoint = None
		model_opt = opt
		vocab = torch.load(opt.data + '.vocab.pt')

	# check for code where vocab is saved instead of fields
	# (in the future this will be done in a smarter way)
	if old_style_vocab(vocab):
		logger.warning('Using old style vocab')
		fields = load_old_vocab(vocab)
	else:
		fields = vocab

	# Report src and tgt vocab sizes, including for features
	for side in ['src', 'tmpl', 'src2', 'tgt']:
		f = fields[side]
		try:
			f_iter = iter(f)
		except TypeError:
			f_iter = [(side, f)]
		for sn, sf in f_iter:
			if sf.use_vocab:
				logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))

	# Build model.

	if not opt.joint:
		_check_save_model_path(opt.save_model1)
		_check_save_model_path(opt.save_model2)

		model, model2 = build_model(model_opt, opt, fields, checkpoint)

		n_params, enc, dec = _tally_parameters(model)
		logger.info('encoder: %d' % enc)
		logger.info('decoder: %d' % dec)
		logger.info('* number of parameters: %d' % n_params)

		n_params2, enc2, dec2 = _tally_parameters(model2)
		logger.info('encoder: %d' % enc2)
		logger.info('decoder: %d' % dec2)
		logger.info('* number of parameters: %d' % n_params2)

		# Build optimizer.
		optim = build_optim(model, opt, checkpoint)
		optim2 = build_optim(model2, opt, checkpoint)

		# Build model saver
		model1_saver = build_model_saver(model_opt, opt, model, fields, optim,
		                                 save_path=opt.save_model1)
		model2_saver = build_model_saver(model_opt, opt, model2, fields, optim,
		                                 save_path=opt.save_model2)

	else:
		assert opt.save_model1[:-2] == opt.save_model2[:-2]
		_save_model = opt.save_model1[:-2]
		_check_save_model_path(_save_model)

		model = build_model(model_opt, opt, fields, checkpoint)
		n_params, enc, dec = _tally_parameters(model)
		logger.info('encoder: %d' % enc)
		logger.info('decoder: %d' % dec)
		logger.info('* number of parameters: %d' % n_params)

		optim = build_optim(model, opt, checkpoint)

		model1_saver = build_model_saver(model_opt, opt, model, fields, optim,
		                                 save_path=_save_model)
		model2 = None
		optim2 = None
		model2_saver = None

	trainer = build_trainer(opt, device_id, model, model2, fields, optim, optim2,
	                        model1_saver=model1_saver, model2_saver=model2_saver)

	train_iter = build_dataset_iter("train", fields, opt)
	valid_iter = build_dataset_iter("valid", fields, opt, is_train=False)

	if len(opt.gpu_ranks):
		logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
	else:
		logger.info('Starting training on CPU, could be very slow')
	train_steps = opt.train_steps

	trainer.train(
		train_iter,
		train_steps,
		valid_iter=valid_iter,
		valid_steps=opt.valid_steps)

	if opt.tensorboard:
		trainer.report_manager.tensorboard_writer.close()
