# -*- coding: utf-8 -*-

'''
@Author     : Jiangjie Chen
@Time       : 2019/2/8 12:28
@Contact    : mi0134sher@hotmail.com
@Description: 
'''

import torch, re
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import monmt.inputters as inputters
import monmt.modules
from monmt.encoders.rnn_encoder import RNNEncoder
from monmt.encoders.transformer import TransformerEncoder
from monmt.encoders.cnn_encoder import CNNEncoder
from monmt.encoders.mean_encoder import MeanEncoder

from monmt.decoders.decoder import InputFeedRNNDecoder, StdRNNDecoder
from monmt.decoders.decoder2 import InputFeedRNNDecoder2
from monmt.decoders.transformer import TransformerDecoder
from monmt.decoders.cnn_decoder import CNNDecoder

from monmt.modules import Embeddings, CopyGenerator
from monmt.utils.misc import use_gpu
from monmt.utils.logging import logger


def build_embeddings(opt, word_field, feat_fields, for_encoder=True):
	"""
	Args:
		opt: the option in current environment.
		word_dict(Vocab): words dictionary.
		feature_dicts([Vocab], optional): a list of feature dictionary.
		for_encoder(bool): build Embeddings for encoder or decoder?
	"""
	emb_dim = opt.src_word_vec_size if for_encoder else opt.tgt_word_vec_size

	word_padding_idx = word_field.vocab.stoi[word_field.pad_token]
	num_word_embeddings = len(word_field.vocab)

	feat_pad_indices = [ff.vocab.stoi[ff.pad_token] for ff in feat_fields]
	num_feat_embeddings = [len(ff.vocab) for ff in feat_fields]

	emb = Embeddings(
		word_vec_size=emb_dim,
		position_encoding=opt.position_encoding,
		feat_merge=opt.feat_merge,
		feat_vec_exponent=opt.feat_vec_exponent,
		feat_vec_size=opt.feat_vec_size,
		dropout=opt.dropout,
		word_padding_idx=word_padding_idx,
		feat_padding_idx=feat_pad_indices,
		word_vocab_size=num_word_embeddings,
		feat_vocab_sizes=num_feat_embeddings,
		sparse=opt.optim == "sparseadam"
	)
	return emb


def build_encoder(opt, embeddings):
	if opt.encoder_type == 'transformer':
		encoder = TransformerEncoder(
			opt.enc_layers,
			opt.enc_rnn_size,
			opt.heads,
			opt.transformer_ff,
			opt.dropout,
			embeddings
		)
	elif opt.encoder_type == "cnn":
		encoder = CNNEncoder(
			opt.enc_layers,
			opt.enc_rnn_size,
			opt.cnn_kernel_width,
			opt.dropout,
			embeddings)
	elif opt.encoder_type == "mean":
		encoder = MeanEncoder(
			opt.enc_layers,
			embeddings,
			opt.src_word_vec_size,
			attn_hidden=opt.attn_hidden,
			dropout=opt.dropout)
	else:
		encoder = RNNEncoder(
			opt.rnn_type,
			opt.brnn,
			opt.enc_layers,
			opt.enc_rnn_size,
			opt.dropout,
			embeddings,
			opt.bridge
		)
	return encoder


def build_decoder(opt, embeddings):
	if opt.decoder_type == "transformer":
		decoder = TransformerDecoder(
			opt.dec_layers,
			opt.dec_rnn_size,
			opt.heads,
			opt.transformer_ff,
			opt.global_attention,
			opt.copy_attn,
			opt.self_attn_type,
			opt.dropout,
			embeddings
		)
	elif opt.decoder_type == "cnn":
		decoder = CNNDecoder(
			opt.dec_layers,
			opt.dec_rnn_size,
			opt.global_attention,
			opt.copy_attn,
			opt.cnn_kernel_width,
			opt.dropout,
			embeddings
		)
	else:
		dec_class = InputFeedRNNDecoder if opt.input_feed else StdRNNDecoder
		decoder = dec_class(
			opt.rnn_type,
			opt.brnn,
			opt.dec_layers,
			opt.dec_rnn_size,
			opt.global_attention,
			opt.global_attention_function,
			opt.coverage_attn,
			opt.context_gate,
			opt.copy_attn,
			opt.dropout,
			embeddings,
			opt.reuse_copy_attn
		)
	return decoder


def build_encoder2(opt, embeddings, brnn2=False):
	if opt.encoder2_type == 'transformer':
		encoder = TransformerEncoder(
			opt.enc_layers,
			opt.enc_rnn_size,
			opt.heads,
			opt.transformer_ff,
			opt.dropout,
			embeddings
		)
	elif opt.encoder2_type == "cnn":
		encoder = CNNEncoder(
			opt.enc_layers,
			opt.enc_rnn_size,
			opt.cnn_kernel_width,
			opt.dropout,
			embeddings)
	elif opt.encoder2_type == "mean":
		encoder = MeanEncoder(
			opt.enc_layers,
			embeddings,
			opt.src_word_vec_size,
			attn_hidden=opt.attn_hidden,
			dropout=opt.dropout)
	else:
		encoder = RNNEncoder(
			opt.rnn_type,
			brnn2,
			opt.enc_layers,
			opt.enc_rnn_size,
			opt.dropout,
			embeddings,
			opt.bridge
		)
	return encoder


def build_decoder2(opt, embeddings, brnn2=False):
	if opt.decoder2_type == "transformer":
		decoder = TransformerDecoder(
			opt.dec_layers,
			opt.dec_rnn_size,
			opt.heads,
			opt.transformer_ff,
			opt.global_attention,
			opt.copy_attn,
			opt.self_attn_type,
			opt.dropout,
			embeddings
		)
	elif opt.decoder2_type == "cnn":
		decoder = CNNDecoder(
			opt.dec_layers,
			opt.dec_rnn_size,
			opt.global_attention,
			opt.copy_attn,
			opt.cnn_kernel_width,
			opt.dropout,
			embeddings
		)
	else:
		decoder = InputFeedRNNDecoder2(
			opt.rnn_type,
			brnn2,
			opt.dec_layers,
			opt.dec_rnn_size,
			opt.global_attention,
			opt.global_attention_function,
			opt.coverage_attn,
			opt.context_gate2,
			opt.copy_attn,
			opt.dropout,
			embeddings,
			opt.reuse_copy_attn
		)
	return decoder


def load_joint_model(opt):
	pass


def load_test_model(opt):
	model1_path = opt.models[0]
	pass1 = True
	checkpoint1 = torch.load(model1_path,
	                         map_location=lambda storage, loc: storage)
	model_opt = checkpoint1['opt']
	vocab = checkpoint1['vocab']

	if len(opt.models) > 1:
		model2_path = opt.models[1]
		pass1 = False
		checkpoint2 = torch.load(model2_path, map_location=lambda storage, loc: storage)

	if inputters.old_style_vocab(vocab):
		fields = inputters.load_old_vocab(vocab)
	else:
		fields = vocab

	model1, emb_weight = build_base_model(model_opt, fields, use_gpu(opt),
	                                      checkpoint1, opt.gpu)
	model1.eval()
	model1.generator.eval()

	if pass1:
		return fields, model1, model_opt
	else:
		model2 = build_base_model2(model_opt, fields, use_gpu(opt),
		                           checkpoint2, opt.gpu, prev_emb_w=emb_weight)
		model2.eval()
		model2.generator.eval()
		return fields, (model1, model2), model_opt


def build_base_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
	src_field = fields['src']
	feat_fields = [fields[k] for k in inputters.collect_features(fields, 'src')]
	src_emb = build_embeddings(model_opt, src_field, feat_fields)
	encoder = build_encoder(model_opt, src_emb)

	tgt_field = fields['tmpl']
	feat_fields = [fields[k]
	               for k in inputters.collect_features(fields, 'tmpl')]
	tgt_emb = build_embeddings(model_opt, tgt_field, feat_fields, for_encoder=False)

	# Share the embedding matrix - preprocess with share_vocab required.
	if model_opt.share_embeddings:
		# src/tgt vocab should be the same if `-share_vocab` is specified.
		assert src_field.vocab == tgt_field.vocab, \
			"preprocess with -share_vocab if you use share_embeddings"

		tgt_emb.word_lut.weight = src_emb.word_lut.weight

	decoder = build_decoder(model_opt, tgt_emb)

	if gpu:
		if gpu_id is not None:
			device = torch.device("cuda", gpu_id)
		else:
			device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	model = monmt.models.NMTModel(encoder, decoder)

	# Build Generator.
	tgt_base_field = fields['tmpl']
	gen_func = nn.LogSoftmax(dim=-1)
	generator = nn.Sequential(
		nn.Linear(model_opt.dec_rnn_size, len(tgt_base_field.vocab)),
		gen_func
	)
	if model_opt.share_decoder_embeddings:
		generator[0].weight = decoder.embeddings.word_lut.weight

	# Load the model states from checkpoint or initialize them.
	if checkpoint is not None:
		# This preserves backward-compat for models using customed layernorm
		def fix_key(s):
			s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
			           r'\1.layer_norm\2.bias', s)
			s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
			           r'\1.layer_norm\2.weight', s)
			return s

		checkpoint['model'] = {fix_key(k): v
		                       for k, v in checkpoint['model'].items()}
		# end of patch for backward compatibility

		model.load_state_dict(checkpoint['model'], strict=False)
		generator.load_state_dict(checkpoint['generator'], strict=False)
	else:
		if model_opt.param_init != 0.0:
			for p in model.parameters():
				p.data.uniform_(-model_opt.param_init, model_opt.param_init)
			for p in generator.parameters():
				p.data.uniform_(-model_opt.param_init, model_opt.param_init)
		if model_opt.param_init_glorot:
			for p in model.parameters():
				if p.dim() > 1:
					xavier_uniform_(p)
			for p in generator.parameters():
				if p.dim() > 1:
					xavier_uniform_(p)

		if hasattr(model.encoder, 'embeddings'):
			model.encoder.embeddings.load_pretrained_vectors(
				model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
		if hasattr(model.decoder, 'embeddings'):
			model.decoder.embeddings.load_pretrained_vectors(
				model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

	model.generator = generator
	model.to(device)

	return model, src_emb.word_lut.weight


def build_base_model2(model_opt, fields, gpu, checkpoint=None, gpu_id=None, prev_emb_w=None):
	src2_field = fields['src2']
	feat2_fields = [fields[k] for k in inputters.collect_features(fields, 'src2')]
	src2_emb = build_embeddings(model_opt, src2_field, feat2_fields)
	if prev_emb_w is not None:
		src2_emb.word_lut.weight = prev_emb_w
	encoder2 = build_encoder2(model_opt, src2_emb, brnn2=False)

	tgt_field = fields['tgt']
	feat_fields = [fields[k]
	               for k in inputters.collect_features(fields, 'tgt')]
	tgt_emb = build_embeddings(model_opt, tgt_field, feat_fields, for_encoder=model_opt.brnn2)

	# Share the embedding matrix - preprocess with share_vocab required.
	if model_opt.share_embeddings:
		tgt_emb.word_lut.weight = prev_emb_w

	decoder2 = build_decoder2(model_opt, tgt_emb, brnn2=model_opt.brnn2)

	if gpu:
		if gpu_id is not None:
			device = torch.device("cuda", gpu_id)
		else:
			device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	model = monmt.models.Model2(encoder2, decoder2)

	# Build Generator.

	tgt_base_field = fields['tgt']
	vocab_size = len(tgt_base_field.vocab)
	if model_opt.copy_attn:
		pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
		generator = CopyGenerator(model_opt.dec_rnn_size,
		                          vocab_size,
		                          pad_idx)
	else:
		gen_func = nn.LogSoftmax(dim=-1)
		generator = nn.Sequential(
			nn.Linear(model_opt.dec_rnn_size, vocab_size),
			gen_func
		)
		if model_opt.share_decoder_embeddings:
			generator[0].weight = decoder2.embeddings.word_lut.weight

	# Load the model states from checkpoint or initialize them.
	if checkpoint is not None:
		# This preserves backward-compat for models using customed layernorm
		def fix_key(s):
			s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
			           r'\1.layer_norm\2.bias', s)
			s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
			           r'\1.layer_norm\2.weight', s)
			return s

		checkpoint['model'] = {fix_key(k): v
		                       for k, v in checkpoint['model'].items()}
		# end of patch for backward compatibility

		model.load_state_dict(checkpoint['model'], strict=False)
		generator.load_state_dict(checkpoint['generator'], strict=False)
	else:
		if model_opt.param_init != 0.0:
			for p in model.parameters():
				p.data.uniform_(-model_opt.param_init, model_opt.param_init)
			for p in generator.parameters():
				p.data.uniform_(-model_opt.param_init, model_opt.param_init)
		if model_opt.param_init_glorot:
			for p in model.parameters():
				if p.dim() > 1:
					xavier_uniform_(p)
			for p in generator.parameters():
				if p.dim() > 1:
					xavier_uniform_(p)

		if hasattr(model.encoder, 'embeddings'):
			model.encoder.embeddings.load_pretrained_vectors(
				model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
		if hasattr(model.decoder, 'embeddings'):
			model.decoder.embeddings.load_pretrained_vectors(
				model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

	model.generator = generator
	model.to(device)

	return model


def build_joint_model(model_opt, fields, gpu, checkpoint=None, gpu_id=None):
	src_field = fields['src']
	feat_fields = [fields[k] for k in inputters.collect_features(fields, 'src')]
	src_emb = build_embeddings(model_opt, src_field, feat_fields)
	encoder1 = build_encoder(model_opt, src_emb)

	src2_field = fields['src2']
	feat_fields = [fields[k] for k in inputters.collect_features(fields, 'src2')]
	src2_emb = build_embeddings(model_opt, src2_field, feat_fields)
	encoder2 = build_encoder2(model_opt, src2_emb, brnn2=False)

	tmpl_field = fields['tmpl']
	feat_fields = [fields[k] for k in inputters.collect_features(fields, 'tmpl')]
	tmpl_emb = build_embeddings(model_opt, tmpl_field, feat_fields, for_encoder=False)

	tgt_field = fields['tgt']
	feat_fields = [fields[k] for k in inputters.collect_features(fields, 'tgt')]
	tgt_emb = build_embeddings(model_opt, tgt_field, feat_fields, for_encoder=False)

	# Share the embedding matrix - preprocess with share_vocab required.
	if model_opt.share_embeddings:
		tmpl_emb.word_lut.weight = src_emb.word_lut.weight
		src2_emb.word_lut.weight = src_emb.word_lut.weight
		tgt_emb.word_lut.weight = src_emb.word_lut.weight

	decoder1 = build_decoder(model_opt, tmpl_emb)
	decoder2 = build_decoder2(model_opt, tgt_emb, brnn2=False)

	if gpu:
		if gpu_id is not None:
			device = torch.device("cuda", gpu_id)
		else:
			device = torch.device("cuda")
	else:
		device = torch.device("cpu")

	model = monmt.models.JointModel(
		encoder1=encoder1, encoder2=encoder2,
		decoder1=decoder1, decoder2=decoder2
	)

	# TODO: Build Generator.

	tgt_base_field = fields['tmpl']
	gen_func = nn.LogSoftmax(dim=-1)
	generator1 = nn.Sequential(
		nn.Linear(model_opt.dec_rnn_size, len(tgt_base_field.vocab)),
		gen_func
	)

	tgt_base_field = fields['tgt']
	vocab_size = len(tgt_base_field.vocab)
	pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
	generator2 = CopyGenerator(model_opt.dec_rnn_size,
	                           vocab_size,
	                           pad_idx)

	# Load the model states from checkpoint or initialize them.
	if checkpoint is not None:
		# This preserves backward-compat for models using customed layernorm
		def fix_key(s):
			s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
			           r'\1.layer_norm\2.bias', s)
			s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
			           r'\1.layer_norm\2.weight', s)
			return s

		checkpoint['model'] = {fix_key(k): v
		                       for k, v in checkpoint['model'].items()}
		# end of patch for backward compatibility

		model.load_state_dict(checkpoint['model'], strict=False)
		generator1.load_state_dict(checkpoint['generator1'], strict=False)
		generator2.load_state_dict(checkpoint['generator2'], strict=False)
	else:
		if model_opt.param_init != 0.0:
			for p in model.parameters():
				p.data.uniform_(-model_opt.param_init, model_opt.param_init)
			for p in generator1.parameters():
				p.data.uniform_(-model_opt.param_init, model_opt.param_init)
			for p in generator2.parameters():
				p.data.uniform_(-model_opt.param_init, model_opt.param_init)
		if model_opt.param_init_glorot:
			for p in model.parameters():
				if p.dim() > 1:
					xavier_uniform_(p)
			for p in generator1.parameters():
				if p.dim() > 1:
					xavier_uniform_(p)
			for p in generator2.parameters():
				if p.dim() > 1:
					xavier_uniform_(p)

	model.generator1 = generator1
	model.generator2 = generator2
	model.to(device)

	return model


def build_model(model_opt, opt, fields, checkpoint):
	logger.info('Building model...')
	if opt.joint:
		model = build_joint_model(model_opt, fields, use_gpu(opt), checkpoint)
		logger.info(model)
		return model
	else:
		model, emb_word_lut = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
		logger.info(model)
		model2 = build_base_model2(model_opt, fields, use_gpu(opt), checkpoint,
		                           prev_emb_w=emb_word_lut)
		logger.info(model2)
		return model, model2
