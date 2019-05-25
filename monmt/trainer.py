"""
    This is the loadable seq2seq trainer library that is
    in charge of training details, loss compute, and statistics.
    See train.py for a use case of this library.

    Note: To make this a general library, we implement *only*
          mechanism things here(i.e. what to do), and leave the strategy
          things to users(i.e. how to do it). Also see train.py(one of the
          users of this library) for the strategy things we do.
"""

import torch
import monmt.inputters as inputters
import monmt.utils

from monmt.utils.logging import logger


def build_trainer(opt, device_id, model, model2, fields,
                  optim, optim2, model1_saver=None, model2_saver=None):
	"""
	Simplify `Trainer` creation based on user `opt`s*

	Args:
		opt (:obj:`Namespace`): user options (usually from argument parsing)
		model (:obj:`monmt.models.NMTModel`): the model to train
		fields (dict): dict of fields
		optim (:obj:`monmt.utils.Optimizer`): optimizer used during training
		model_saver(:obj:`monmt.models.ModelSaverBase`): the utility object
			used to save the model
	"""

	train_loss1 = monmt.utils.loss.build_loss_compute(
		model, fields["tmpl"], opt, pass1=True)
	valid_loss1 = monmt.utils.loss.build_loss_compute(
		model, fields["tmpl"], opt, train=False, pass1=True)

	_model2 = model if model2 is None else model2

	train_loss2 = monmt.utils.loss.build_loss_compute(
		_model2, fields["tgt"], opt, pass1=False)
	valid_loss2 = monmt.utils.loss.build_loss_compute(
		_model2, fields["tgt"], opt, train=False, pass1=False)

	trunc_size = opt.truncated_decoder  # Badly named...
	shard_size = opt.max_generator_batches
	norm_method = opt.normalization
	grad_accum_count = opt.accum_count
	n_gpu = opt.world_size
	if device_id >= 0:
		gpu_rank = opt.gpu_ranks[device_id]
	else:
		gpu_rank = 0
		n_gpu = 0
	gpu_verbose_level = opt.gpu_verbose_level

	report_manager = monmt.utils.build_report_manager(opt)
	trainer = monmt.Trainer(model, model2, train_loss1, valid_loss1,
	                        train_loss2, valid_loss2, optim, optim2,
	                        trunc_size, shard_size, norm_method,
	                        grad_accum_count, n_gpu, gpu_rank,
	                        gpu_verbose_level, report_manager,
	                        model1_saver=model1_saver,
	                        model2_saver=model2_saver,
	                        vocab=fields['tgt'].vocab)
	return trainer


class Trainer(object):
	"""
	Class that controls the training process.

	Args:
			model(:py:class:`monmt.models.model.NMTModel`): translation model
				to train
			train_loss(:obj:`monmt.utils.loss.LossComputeBase`):
			   training loss computation
			valid_loss(:obj:`monmt.utils.loss.LossComputeBase`):
			   training loss computation
			optim(:obj:`monmt.utils.optimizers.Optimizer`):
			   the optimizer responsible for update
			trunc_size(int): length of truncated back propagation through time
			shard_size(int): compute loss in shards of this size for efficiency
			norm_method(string): normalization methods: [sents|tokens]
			grad_accum_count(int): accumulate gradients this many times.
			report_manager(:obj:`monmt.utils.ReportMgrBase`):
				the object that creates reports, or None
			model_saver(:obj:`monmt.models.ModelSaverBase`): the saver is
				used to save a checkpoint.
				Thus nothing will be saved if this parameter is None
	"""

	def __init__(self, model, model2, train_loss, valid_loss, train_loss2, valid_loss2,
	             optim, optim2, trunc_size=0, shard_size=32,
	             norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
	             gpu_verbose_level=0, report_manager=None,
	             model1_saver=None, model2_saver=None, vocab=None):
		# Basic attributes.
		self.model1 = model
		self.model2 = model2
		self.train_loss1 = train_loss
		self.valid_loss1 = valid_loss
		self.train_loss2 = train_loss2
		self.valid_loss2 = valid_loss2
		self.optim1 = optim
		self.optim2 = optim2
		self.trunc_size = trunc_size
		self.shard_size = shard_size
		self.norm_method = norm_method
		self.grad_accum_count = grad_accum_count
		self.n_gpu = n_gpu
		self.gpu_rank = gpu_rank
		self.gpu_verbose_level = gpu_verbose_level
		self.report_manager = report_manager
		self.model1_saver = model1_saver
		self.model2_saver = model2_saver
		self.vocab = vocab

		assert grad_accum_count > 0
		if grad_accum_count > 1:
			assert (self.trunc_size == 0), \
				"""To enable accumulated gradients,
				   you must disable target sequence truncating."""

		# Set model in training mode.
		self.model1.train()
		if self.model2 is None:
			assert self.optim2 is None and self.model2_saver is None
			self.joint = True
		else:
			self.model2.train()
			self.joint = False

	def train(self,
	          train_iter,
	          train_steps,
	          valid_iter=None,
	          valid_steps=10000):
		"""
		The main training loop by iterating over `train_iter` and possibly
		running validation on `valid_iter`.

		Args:
		    train_iter: A generator that returns the next training batch.
		    train_steps: Run training for this many iterations.
		    valid_iter: A generator that returns the next validation batch.
		    valid_steps: Run evaluation every this many iterations.

		Returns:
		    The gathered statistics.
		"""
		if valid_iter is None:
			logger.info('Start training loop without validation...')
		else:
			logger.info('Start training loop and validate every %d steps...',
			            valid_steps)

		step = self.optim1._step + 1
		true_batchs = []
		accum = 0
		normalization1 = 0
		normalization2 = 0

		total_stats = monmt.utils.Statistics()
		report_stats = monmt.utils.Statistics()
		total_stats2 = monmt.utils.Statistics()
		report_stats2 = monmt.utils.Statistics()
		self._start_report_manager(start_time=total_stats.start_time)

		while step <= train_steps:

			reduce_counter = 0
			for i, batch in enumerate(train_iter):
				if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):
					if self.gpu_verbose_level > 1:
						logger.info("GpuRank %d: index: %d accum: %d"
						            % (self.gpu_rank, i, accum))

					true_batchs.append(batch)

					if self.norm_method == "tokens":
						num_tokens = batch.tmpl[1:].ne(
							self.train_loss1.padding_idx).sum()
						normalization1 += num_tokens.item()
					else:
						normalization1 += batch.batch_size

					if self.norm_method == "tokens":
						num_tokens = batch.tgt[1:].ne(
							self.train_loss2.padding_idx).sum()
						normalization2 += num_tokens.item()
					else:
						normalization2 += batch.batch_size

					accum += 1
					if accum == self.grad_accum_count:
						reduce_counter += 1
						if self.gpu_verbose_level > 0:
							logger.info("GpuRank %d: reduce_counter: %d \
                                        n_minibatch %d"
							            % (self.gpu_rank, reduce_counter,
							               len(true_batchs)))
						if self.n_gpu > 1:
							normalization1 = sum(monmt.utils.distributed
							                     .all_gather_list
							                     (normalization1))
							normalization2 = sum(monmt.utils.distributed
							                     .all_gather_list
							                     (normalization2))
						if self.joint:
							self._gradient_accumulation_joint(
								true_batchs, normalization1, total_stats, report_stats,
								normalization2, total_stats2, report_stats2)
						else:
							self._gradient_accumulation(
								true_batchs, normalization1, total_stats, report_stats,
								normalization2, total_stats2, report_stats2)

						report_stats = self._maybe_report_training(
							step, train_steps,
							self.optim1.learning_rate,
							report_stats)

						if not self.joint:
							report_stats2 = self._maybe_report_training(
								step, train_steps,
								self.optim2.learning_rate,
								report_stats2)
						else:
							report_stats2 = self._maybe_report_training(
								step, train_steps,
								self.optim1.learning_rate,
								report_stats2)

						true_batchs = []
						accum = 0
						normalization1 = 0
						normalization2 = 0
						if (step % valid_steps == 0):
							if self.gpu_verbose_level > 0:
								logger.info('GpuRank %d: validate step %d'
								            % (self.gpu_rank, step))

							if self.joint:
								valid_stats1, valid_stats2 = self.validate_joint(valid_iter)
							else:
								valid_stats1, valid_stats2 = self.validate(valid_iter)

							if self.gpu_verbose_level > 0:
								logger.info('GpuRank %d: gather valid stat \
                                            step %d' % (self.gpu_rank, step))
							valid_stats1 = self._maybe_gather_stats(valid_stats1)

							if self.gpu_verbose_level > 0:
								logger.info('GpuRank %d: report stat step %d'
								            % (self.gpu_rank, step))

							self._report_step(self.optim1.learning_rate,
							                  step, valid_stats=valid_stats1)

							valid_stats2 = self._maybe_gather_stats(valid_stats2)
							if self.joint:
								self._report_step(self.optim1.learning_rate,
								                  step, valid_stats=valid_stats2)
							else:
								self._report_step(self.optim2.learning_rate,
								                  step, valid_stats=valid_stats2)

						if self.gpu_rank == 0:
							self._maybe_save(step)
						step += 1
						if step > train_steps:
							break

			if self.gpu_verbose_level > 0:
				logger.info('GpuRank %d: we completed an epoch \
                            at step %d' % (self.gpu_rank, step))

		return total_stats

	def validate(self, valid_iter):
		""" Validate model.
			valid_iter: validate data iterator
		Returns:
			:obj:`nmt.Statistics`: validation loss statistics
		"""
		# Set model in validating mode.
		self.model1.eval()
		self.model2.eval()

		def id2sent(self, ids):
			if self.vocab is not None:
				itos = {v: k for (k, v) in self.vocab.stoi.items()}
				return [itos.get(w.tolist(), '<unk>') for w in ids]

		with torch.no_grad():
			stats = monmt.utils.Statistics()
			stats2 = monmt.utils.Statistics()

			show = True

			for batch in valid_iter:
				src1 = inputters.make_features(batch, 'src')
				_, src1_lengths = batch.src

				tgt1 = inputters.make_features(batch, 'tmpl')

				# F-prop through the model.
				outputs1, attns1, memory_bank1, src1_lengths = \
					self.model1(src1, tgt1, src1_lengths)

				# Compute loss.
				batch_stats1 = self.valid_loss1.monolithic_compute_loss(
					batch, outputs1, attns1, pass1=True, show=show)

				if show:
					batch_stats1, preds = batch_stats1
					preds = preds.view(-1, batch.batch_size)[:, 0]
					sent = id2sent(self, preds)
					logger.warning('* predicted template: %s' % sent)

				# Update statistics.
				stats.update(batch_stats1)

				src2 = inputters.make_features(batch, 'src2')

				_, src2_lengths = batch.src2
				tgt2 = inputters.make_features(batch, 'tgt')

				# F-prop through the model.
				outputs2, attns2 = self.model2(
					src2=src2, tgt=tgt2, lengths2=src2_lengths,
					src1=src1, memory_bank1=memory_bank1, lengths1=src1_lengths)

				# Compute loss.
				batch_stats2 = self.valid_loss2.monolithic_compute_loss(
					batch, outputs2, attns2, pass1=False, show=show)

				if show:
					batch_stats2, preds = batch_stats2
					preds = preds.view(-1, batch.batch_size)[:, 0]
					tgt = tgt2.squeeze(-1)[:, 0]
					sent = id2sent(self, tgt)
					logger.warning('* golden target: %s' % sent)
					sent = id2sent(self, preds)
					logger.warning('* predicted target: %s' % sent)
					show = False

				# Update statistics.
				stats2.update(batch_stats2)

		# Set model back to training mode.
		self.model1.train()
		self.model2.train()
		return stats, stats2

	def validate_joint(self, valid_iter):
		""" Validate model.
			valid_iter: validate data iterator
		Returns:
			:obj:`nmt.Statistics`: validation loss statistics
		"""
		# Set model in validating mode.
		self.model1.eval()

		with torch.no_grad():
			stats = monmt.utils.Statistics()
			stats2 = monmt.utils.Statistics()

			for batch in valid_iter:
				src1 = inputters.make_features(batch, 'src')
				_, src1_lengths = batch.src

				tgt1 = inputters.make_features(batch, 'tmpl')

				src2 = inputters.make_features(batch, 'src2')

				_, src2_lengths = batch.src2
				tgt2 = inputters.make_features(batch, 'tgt')

				# F-prop through the model.
				outputs1, attns1, outputs2, attns2 = self.model1(
					src=src1,
					tmpl=tgt1,
					src2=src2,
					tgt=tgt2,
					lengths1=src1_lengths,
					lengths2=src2_lengths
				)

				# Compute loss.
				batch_stats1 = self.valid_loss1.monolithic_compute_loss(
					batch, outputs1, attns1, pass1=True)

				# Update statistics.
				stats.update(batch_stats1)

				# Compute loss.
				batch_stats2 = self.valid_loss2.monolithic_compute_loss(
					batch, outputs2, attns2, pass1=False)

				# Update statistics.
				stats2.update(batch_stats2)

		# Set model back to training mode.
		self.model1.train()
		return stats, stats2

	def _gradient_accumulation(self, true_batchs,
	                           normalization1, total_stats1, report_stats1,
	                           normalization2=None, total_stats2=None, report_stats2=None):
		if self.grad_accum_count > 1:
			self.model1.zero_grad()
			self.model2.zero_grad()

		for batch in true_batchs:
			# pass 1
			target_size = batch.tmpl.size(0)
			# Truncated BPTT: reminder not compatible with accum > 1
			trunc_size = self.trunc_size if self.trunc_size else target_size

			# dec_state = None
			src1 = inputters.make_features(batch, 'src')
			_, src1_lengths = batch.src
			report_stats1.n_src_words += src1_lengths.sum().item()

			tgt1_outer = inputters.make_features(batch, 'tmpl')

			for j in range(0, target_size - 1, trunc_size):
				# 1. Create truncated target.
				tgt1 = tgt1_outer[j: j + trunc_size]

				# 2. F-prop all but generator.
				if self.grad_accum_count == 1:
					self.model1.zero_grad()

				outputs1, attns1, memory_bank1, src1_lengths \
					= self.model1(src1, tgt1, src1_lengths)

				# 3. Compute loss in shards for memory efficiency.
				batch_stats = self.train_loss1.sharded_compute_loss(
					batch, outputs1, attns1, j,
					trunc_size, self.shard_size, normalization1, retain_graph=True)
				total_stats1.update(batch_stats)
				report_stats1.update(batch_stats)

				# If truncated, don't backprop fully.
				# TO CHECK
				# if dec_state is not None:
				#    dec_state.detach()
				if self.model1.decoder.state is not None:
					self.model1.decoder.detach_state()

			# pass 2
			target_size = batch.tgt.size(0)
			trunc_size = self.trunc_size if self.trunc_size else target_size
			_, src2_lengths = batch.src2
			report_stats2.n_src_words += src2_lengths.sum()

			src2 = inputters.make_features(batch, 'src2')

			# logger.warning('Using tgt for pass 2')
			tgt_outer = inputters.make_features(batch, 'tgt')
			for j in range(0, target_size - 1, trunc_size):
				# 1. Create truncated target.
				tgt2 = tgt_outer[j: j + trunc_size]

				# 2. F-prop all but generator.
				if self.grad_accum_count == 1:
					self.model2.zero_grad()

				outputs2, attns2 = self.model2(
					src2=src2, tgt=tgt2, lengths2=src2_lengths,
					src1=src1, memory_bank1=memory_bank1, lengths1=src1_lengths
				)

				# retain_graph is false for the final truncation
				retain_graph = (j + trunc_size) < (target_size - 1)

				# 3. Compute loss in shards for memory efficiency.
				batch_stats = self.train_loss2.sharded_compute_loss(
					batch, outputs2, attns2, j,
					trunc_size, self.shard_size, normalization2, retain_graph=retain_graph)
				total_stats2.update(batch_stats)
				report_stats2.update(batch_stats)

				# 4. Update the parameters and statistics.
				if self.grad_accum_count == 1:
					# Multi GPU gradient gather
					if self.n_gpu > 1:
						grads = [p.grad.data for p in self.model2.parameters()
						         if p.requires_grad
						         and p.grad is not None]
						monmt.utils.distributed.all_reduce_and_rescale_tensors(
							grads, float(1))
					self.optim2.step()

				# If truncated, don't backprop fully.
				# TO CHECK
				# if dec_state is not None:
				#    dec_state.detach()
				if self.model2.decoder.state is not None:
					self.model2.decoder.detach_state()

			# 4. Update the parameters and statistics.
			if self.grad_accum_count == 1:
				# Multi GPU gradient gather
				if self.n_gpu > 1:
					grads = [p.grad.data for p in self.model1.parameters()
					         if p.requires_grad
					         and p.grad is not None]
					monmt.utils.distributed.all_reduce_and_rescale_tensors(
						grads, float(1))
				self.optim1.step()

		# in case of multi step gradient accumulation,
		# update only after accum batches
		if self.grad_accum_count > 1:
			logger.warning('Grad_accum_count > 1!!!')
			if self.n_gpu > 1:
				grads = [p.grad.data for p in self.model1.parameters()
				         if p.requires_grad
				         and p.grad is not None]
				monmt.utils.distributed.all_reduce_and_rescale_tensors(
					grads, float(1))

				grads2 = [p.grad.data for p in self.model2.parameters()
				          if p.requires_grad
				          and p.grad is not None]
				monmt.utils.distributed.all_reduce_and_rescale_tensors(
					grads2, float(1))
			self.optim1.step()
			self.optim2.step()

	def _gradient_accumulation_joint(self, true_batchs,
	                                 normalization1, total_stats1, report_stats1,
	                                 normalization2=None, total_stats2=None, report_stats2=None):
		if self.grad_accum_count > 1:
			self.model1.zero_grad()

		for batch in true_batchs:
			# pass 1
			target1_size = batch.tmpl.size(0)
			# Truncated BPTT: reminder not compatible with accum > 1
			trunc1_size = self.trunc_size if self.trunc_size else target1_size
			target2_size = batch.tgt.size(0)
			trunc2_size = self.trunc_size if self.trunc_size else target2_size

			# dec_state = None
			src1 = inputters.make_features(batch, 'src')
			src2 = inputters.make_features(batch, 'src2')

			_, src1_lengths = batch.src
			_, src2_lengths = batch.src2
			report_stats1.n_src_words += src1_lengths.sum().item()
			report_stats2.n_src_words += src2_lengths.sum().item()

			tgt1_outer = inputters.make_features(batch, 'tmpl')
			tgt2_outer = inputters.make_features(batch, 'tgt')

			for j in range(0, target1_size - 1, trunc1_size):
				# 1. Create truncated target.
				tgt1 = tgt1_outer[j: j + trunc1_size]
				tgt2 = tgt2_outer[j: j + trunc2_size]

				# 2. F-prop all but generator.
				if self.grad_accum_count == 1:
					self.model1.zero_grad()

				outputs1, attns1, outputs2, attns2 = self.model1(
					src=src1,
					tmpl=tgt1,
					src2=src2,
					tgt=tgt2,
					lengths1=src1_lengths,
					lengths2=src2_lengths
				)

				# retain_graph = (j + trunc_size) < (target_size - 1)
				# 3. Compute loss in shards for memory efficiency.

				joint_loss = zip(
					self.train_loss1.sharded_compute_loss_joint(
						batch, outputs1, attns1, j,
						trunc1_size, self.shard_size, normalization1, retain_graph=True),
					self.train_loss2.sharded_compute_loss_joint(
						batch, outputs2, attns2, j,
						trunc2_size, self.shard_size, normalization2, retain_graph=True)
				)

				batch_stats = monmt.utils.Statistics()
				batch_stats2 = monmt.utils.Statistics()
				for (loss1, stats1), (loss2, stats2) in joint_loss:
					loss = loss1 + loss2
					loss.backward()
					batch_stats.update(stats1)
					batch_stats2.update(stats2)

				total_stats1.update(batch_stats)
				report_stats1.update(batch_stats)

				total_stats2.update(batch_stats2)
				report_stats2.update(batch_stats2)

				if self.model1.decoder1.state is not None:
					self.model1.decoder1.detach_state()
				if self.model1.decoder2.state is not None:
					self.model1.decoder2.detach_state()

			# 4. Update the parameters and statistics.
			if self.grad_accum_count == 1:
				# Multi GPU gradient gather
				if self.n_gpu > 1:
					grads = [p.grad.data for p in self.model1.parameters()
					         if p.requires_grad
					         and p.grad is not None]
					monmt.utils.distributed.all_reduce_and_rescale_tensors(
						grads, float(1))
				self.optim1.step()

		# in case of multi step gradient accumulation,
		# update only after accum batches
		if self.grad_accum_count > 1:
			logger.warning('Grad_accum_count > 1!!!')
			if self.n_gpu > 1:
				grads = [p.grad.data for p in self.model1.parameters()
				         if p.requires_grad
				         and p.grad is not None]
				monmt.utils.distributed.all_reduce_and_rescale_tensors(
					grads, float(1))
			self.optim1.step()

	def _start_report_manager(self, start_time=None):
		"""
		Simple function to start report manager (if any)
		"""
		if self.report_manager is not None:
			if start_time is None:
				self.report_manager.start()
			else:
				self.report_manager.start_time = start_time

	def _maybe_gather_stats(self, stat):
		"""
		Gather statistics in multi-processes cases

		Args:
			stat(:obj:monmt.utils.Statistics): a Statistics object to gather
				or None (it returns None in this case)

		Returns:
			stat: the updated (or unchanged) stat object
		"""
		if stat is not None and self.n_gpu > 1:
			return monmt.utils.Statistics.all_gather_stats(stat)
		return stat

	def _maybe_report_training(self, step, num_steps, learning_rate,
	                           report_stats):
		"""
		Simple function to report training stats (if report_manager is set)
		see `monmt.utils.ReportManagerBase.report_training` for doc
		"""
		if self.report_manager is not None:
			return self.report_manager.report_training(
				step, num_steps, learning_rate, report_stats,
				multigpu=self.n_gpu > 1)

	def _report_step(self, learning_rate, step, train_stats=None,
	                 valid_stats=None):
		"""
		Simple function to report stats (if report_manager is set)
		see `monmt.utils.ReportManagerBase.report_step` for doc
		"""
		if self.report_manager is not None:
			return self.report_manager.report_step(
				learning_rate, step, train_stats=train_stats,
				valid_stats=valid_stats)

	def _maybe_save(self, step):
		"""
		Save the model if a model saver is set
		"""
		if self.model1_saver is not None:
			self.model1_saver.maybe_save(step)
		if self.model2_saver is not None:
			self.model2_saver.maybe_save(step)
