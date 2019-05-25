""" Implementation of all available options """
from __future__ import print_function


def config_opts(parser):
	parser.add('-config', '--config', required=False,
	           is_config_file_arg=True, help='config file path')
	parser.add('-save_config', '--save_config', required=False,
	           is_write_out_config_file_arg=True,
	           help='config file save path')
	parser.add('-description', '--description', required=True, type=str,
	           help='description of this config file')


def model_opts(parser):
	"""
	These options are passed to the construction of the model.
	Be careful with these as they will be used during translation.
	"""

	# Embedding Options
	group = parser.add_argument_group('Model-Embeddings')
	group.add('--src_word_vec_size', '-src_word_vec_size',
	          type=int, default=500,
	          help='Word embedding size for src.')
	group.add('--tgt_word_vec_size', '-tgt_word_vec_size',
	          type=int, default=500,
	          help='Word embedding size for tgt.')
	group.add('--word_vec_size', '-word_vec_size', type=int, default=-1,
	          help='Word embedding size for src and tgt.')
	group.add('--attn_hidden', '-attn_hidden', type=int, default=-1,
	          help='Self-attention hidden size')

	group.add('--share_decoder_embeddings', '-share_decoder_embeddings',
	          action='store_true',
	          help="""Use a shared weight matrix for the input and
                       output word  embeddings in the decoder.""")
	group.add('--share_embeddings', '-share_embeddings', action='store_true',
	          help="""Share the word embeddings between encoder
                       and decoder. Need to use shared dictionary for this
                       option.""")
	group.add('--position_encoding', '-position_encoding', action='store_true',
	          help="""Use a sin to mark relative words positions.
                       Necessary for non-RNN style models.
                       """)

	group = parser.add_argument_group('Model-Embedding Features')
	group.add('--feat_merge', '-feat_merge', type=str, default='concat',
	          choices=['concat', 'sum', 'mlp'],
	          help="""Merge action for incorporating features embeddings.
                       Options [concat|sum|mlp].""")
	group.add('--feat_vec_size', '-feat_vec_size', type=int, default=-1,
	          help="""If specified, feature embedding sizes
                       will be set to this. Otherwise, feat_vec_exponent
                       will be used.""")
	group.add('--feat_vec_exponent', '-feat_vec_exponent',
	          type=float, default=0.7,
	          help="""If -feat_merge_size is not set, feature
                       embedding sizes will be set to N^feat_vec_exponent
                       where N is the number of values the feature takes.""")

	# Encoder-Decoder Options
	group = parser.add_argument_group('Model- Encoder-Decoder')
	group.add('--model_type', '-model_type', default='text',
	          help="""Type of source model to use. Allows
                       the system to incorporate non-text inputs.
                       Options are [text|img|audio].""")

	group.add('--encoder_type', '-encoder_type', type=str, default='rnn',
	          choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn', 'embed'],
	          help="""Type of encoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|brnn|mean|transformer|cnn|embed].""")
	group.add('--decoder_type', '-decoder_type', type=str, default='rnn',
	          choices=['rnn', 'transformer', 'cnn'],
	          help="""Type of decoder layer to use. Non-RNN layers
                       are experimental. Options are
                       [rnn|transformer|cnn].""")

	group.add('--encoder2_type', '-encoder2_type', type=str, default='rnn',
	          choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn', 'embed'],
	          help="""Type of encoder2 layer to use. Non-RNN layers
	                       are experimental. Options are
	                       [rnn|brnn|mean|transformer|cnn|embed].""")
	group.add('--decoder2_type', '-decoder2_type', type=str, default='rnn',
	          choices=['rnn', 'transformer', 'cnn'],
	          help="""Type of decoder2 layer to use. Non-RNN layers
	                       are experimental. Options are
	                       [rnn|transformer|cnn].""")

	group.add('--layers', '-layers', type=int, default=-1,
	          help='Number of layers in enc/dec.')
	group.add('--enc_layers', '-enc_layers', type=int, default=2,
	          help='Number of layers in the encoder')
	group.add('--dec_layers', '-dec_layers', type=int, default=2,
	          help='Number of layers in the decoder')
	group.add('--rnn_size', '-rnn_size', type=int, default=-1,
	          help="""Size of rnn hidden states. Overwrites
                       enc_rnn_size and dec_rnn_size""")
	group.add('--enc_rnn_size', '-enc_rnn_size', type=int, default=500,
	          help="""Size of encoder rnn hidden states.
                       Must be equal to dec_rnn_size except for
                       speech-to-text.""")
	group.add('--dec_rnn_size', '-dec_rnn_size', type=int, default=500,
	          help="""Size of decoder rnn hidden states.
                       Must be equal to enc_rnn_size except for
                       speech-to-text.""")
	group.add('--cnn_kernel_width', '-cnn_kernel_width', type=int, default=3,
	          help="""Size of windows in the cnn, the kernel_size is
                       (cnn_kernel_width, 1) in conv layer""")

	group.add('--input_feed', '-input_feed', type=int, default=1,
	          help="""Feed the context vector at each time step as
                       additional input (via concatenation with the word
                       embeddings) to the decoder.""")
	group.add('--bridge', '-bridge', action="store_true",
	          help="""Have an additional layer between the last encoder
                       state and the first decoder state""")
	group.add('--rnn_type', '-rnn_type', type=str, default='GRU',
	          choices=['LSTM', 'GRU'],
	          help="""The gate type to use in the RNNs""")
	# group.add('--residual', '-residual',   action="store_true",
	#                     help="Add residual connections between RNN layers.")

	group.add('--context_gate', '-context_gate', type=str, default='target',
	          choices=['source', 'target', 'both'],
	          help="""Type of context gate to use.
                       Do not select for no context gate.""")
	group.add('--context_gate2', '-context_gate2', type=str, default='target',
	          choices=['source', 'target', 'both'],
	          help="""Type of context gate to use.
	                       Do not select for no context gate.""")

	# Attention options
	group = parser.add_argument_group('Model- Attention')
	group.add('--global_attention', '-global_attention',
	          type=str, default='general', choices=['dot', 'general', 'mlp'],
	          help="""The attention type to use:
                       dotprod or general (Luong) or MLP (Bahdanau)""")
	group.add('--global_attention_function', '-global_attention_function',
	          type=str, default="softmax", choices=["softmax", "sparsemax"])
	group.add('--self_attn_type', '-self_attn_type',
	          type=str, default="scaled-dot",
	          help="""Self attention type in Transformer decoder
                       layer -- currently "scaled-dot" or "average" """)
	group.add('--heads', '-heads', type=int, default=8,
	          help='Number of heads for transformer self-attention')
	group.add('--transformer_ff', '-transformer_ff', type=int, default=2048,
	          help='Size of hidden transformer feed-forward')

	# Generator and loss options.
	group.add('--copy_attn', '-copy_attn', action="store_true",
	          help='Train copy attention layer.')
	group.add('--generator_function', '-generator_function', default="softmax",
	          choices=["softmax", "sparsemax"],
	          help="""Which function to use for generating
              probabilities over the target vocabulary (choices:
              softmax, sparsemax)""")
	group.add('--copy_attn_force', '-copy_attn_force', action="store_true",
	          help='When available, train to copy.')
	group.add('--reuse_copy_attn', '-reuse_copy_attn', action="store_true",
	          help="Reuse standard attention for copy")
	group.add('--copy_loss_by_seqlength', '-copy_loss_by_seqlength',
	          action="store_true",
	          help="Divide copy loss by length of sequence")
	group.add('--coverage_attn', '-coverage_attn', action="store_true",
	          help='Train a coverage attention layer.')
	group.add('--lambda_coverage', '-lambda_coverage', type=float, default=0.5,
	          help='Lambda value for coverage.')


def preprocess_opts(parser):
	""" Pre-procesing options """
	# Dictionary options, for text corpus
	group = parser.add_argument_group('Data')
	group.add('--save_data', '-save_data', required=True,
	          help="Output file for the prepared data")

	group.add('--train_src', '-train_src', required=True,
	          help="Path to the training source data")
	group.add('--train_tgt', '-train_tgt', required=True,
	          help="Path to the training target data")
	group.add('--train_src2', '-train_src2', required=True,
	          help="Path to the training source2 data")
	group.add('--train_tmpl', '-train_tmpl', required=True,
	          help="Path to the training template data")

	group.add('--valid_src', '-valid_src',
	          help="Path to the validation source data")
	group.add('--valid_tgt', '-valid_tgt',
	          help="Path to the validation target data")
	group.add('--valid_src2', '-valid_src2',
	          help="Path to the validation source2 data")
	group.add('--valid_tmpl', '-valid_tmpl',
	          help="Path to the validation template data")

	group.add('--test_src', '-test_src',
	          help="Path to the test source data")
	group.add('--test_tgt', '-test_tgt',
	          help="Path to the test target data")
	group.add('--test_tmpl', '-test_tmpl',
	          help="Path to the test template data")

	group.add('--shard_size', '-shard_size', type=int, default=10000,
	          help="""Divide src and tgt (if applicable) into
		          smaller multiple src and tgt files, then
		          build shards, each shard will have
		          opt.shard_size samples except last shard.
	              shard_size=0 means no segmentation
	              shard_size>0 means segment dataset into multiple shards,
	              each shard has shard_size samples""")

	group = parser.add_argument_group('Vocab')
	group.add('--src_vocab', '-src_vocab', default='',
	          help="""Path to an existing source vocabulary. Format:
	                one word per line.""")
	group.add('--tgt_vocab', '-tgt_vocab', default='',
	          help="""Path to an existing target vocabulary. Format:
	          	    one word per line.""")
	group.add('--features_vocabs_prefix', '-features_vocabs_prefix',
	          type=str, default='',
	          help="Path prefix to existing features vocabularies")

	group.add('--src_words_min_frequency',
	          '-src_words_min_frequency', type=int, default=0)
	group.add('--tgt_words_min_frequency',
	          '-tgt_words_min_frequency', type=int, default=0)

	group.add('--src_vocab_size', '-src_vocab_size', type=int, default=50000,
	          help="Size of the source vocabulary")
	group.add('--tgt_vocab_size', '-tgt_vocab_size', type=int, default=50000,
	          help="Size of the target vocabulary")

	group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
	          help="Create dynamic dictionaries")
	group.add('--share_vocab', '-share_vocab', action='store_true', default=True,
	          help="Share source and target vocabulary")

	# Truncation options, for text corpus
	group = parser.add_argument_group('Pruning')
	group.add('--max_src_len', '-max_src_len', type=int, default=160,
	          help="Maximum source sequence length")
	group.add('--max_val_words_num', '-max_val_words_num', type=int, default=20,
	          help="Maximum word number of a value")
	group.add('--max_tgt_len', '-max_tgt_len', type=int, default=60,
	          help="Maximum target sequence length to keep.")

	# Data processing options
	group = parser.add_argument_group('Random')
	group.add('--shuffle', '-shuffle', type=int, default=True,
	          help="Shuffle data")
	group.add('--seed', '-seed', type=int, default=3435,
	          help="Random seed")

	group = parser.add_argument_group('Logging')
	group.add('--report_every', '-report_every', type=int, default=100000,
	          help="Report status every this many sentences")
	group.add('--log_file', '-log_file', type=str, default="logs/",
	          help="Output logs to a file under this path.")


def train_opts(parser):
	""" Training and saving options """

	group = parser.add_argument_group('General')

	group.add('--from_scratch', '-from_scratch', action="store_true",
	          help='clear log files if start from scratch')

	group.add('--joint', '-joint', action="store_true",
	          help='true for joint training')
	group.add('--loss_gamma', '-loss_gamma', type=float, default='0.7',
	          help='gamma ratio of loss2 when joint learning')

	group.add('--log_file', '-log_file', type=str, default="logs/",
	          help="Output logs to a file under this path.")
	group.add('--data', '-data', required=True,
	          help="""Path prefix to input data""")

	group.add('--save_model1', '-save_model1', default='models',
	          help="""Model filename (the model will be saved as
                       <save_model1>_N.pt where N is the number
                       of steps""")
	group.add('--save_model2', '-save_model2', default='models',
	          help="""Model filename (the model will be saved as
	                       <save_model2>_N.pt where N is the number
	                       of steps""")

	group.add('--save_checkpoint_steps', '-save_checkpoint_steps',
	          type=int, default=5000,
	          help="""Save a checkpoint every X steps""")
	group.add('--keep_checkpoint', '-keep_checkpoint', type=int, default=-1,
	          help="""Keep X checkpoints (negative: keep all)""")

	# GPU
	group.add('--gpuid', '-gpuid', default=[], nargs='*', type=int,
	          help="Deprecated see world_size and gpu_ranks.")
	group.add('--gpu_ranks', '-gpu_ranks', default=[], nargs='*', type=int,
	          help="list of ranks of each process.")
	group.add('--world_size', '-world_size', default=1, type=int,
	          help="total number of distributed processes.")
	group.add('--gpu_backend', '-gpu_backend',
	          default="nccl", type=str,
	          help="Type of torch distributed backend")
	group.add('--gpu_verbose_level', '-gpu_verbose_level', default=0, type=int,
	          help="Gives more info on each process per GPU.")
	group.add('--master_ip', '-master_ip', default="localhost", type=str,
	          help="IP of master for torch.distributed training.")
	group.add('--master_port', '-master_port', default=10000, type=int,
	          help="Port of master for torch.distributed training.")

	group.add('--seed', '-seed', type=int, default=-1,
	          help="""Random seed used for the experiments
	                      reproducibility.""")

	# Init options
	group = parser.add_argument_group('Initialization')
	group.add('--param_init', '-param_init', type=float, default=0.1,
	          help="""Parameters are initialized over uniform distribution
                       with support (-param_init, param_init).
                       Use 0 to not use initialization""")
	group.add('--param_init_glorot', '-param_init_glorot', action='store_true',
	          help="""Init parameters with xavier_uniform.
                       Required for transfomer.""")

	group.add('--train_from', '-train_from', default='', type=str,
	          help="""If training from a checkpoint then this is the
                       path to the pretrained model's state_dict.""")
	group.add('--start_epoch', '-start_epoch', default=0, type=int,
	          help="""Start training epoch when using checkpoint.""")
	group.add('--reset_optim', '-reset_optim', default='none',
	          choices=['none', 'all', 'states', 'keep_states'],
	          help="""Optimization resetter when train_from.""")

	# Pretrained word vectors
	group.add('--pre_word_vecs_enc', '-pre_word_vecs_enc',
	          help="""If a valid path is specified, then this will load
	                       pretrained word embeddings on the encoder side.
	                       See README for specific formatting instructions.""")
	group.add('--pre_word_vecs_dec', '-pre_word_vecs_dec',
	          help="""If a valid path is specified, then this will load
	                       pretrained word embeddings on the decoder side.
	                       See README for specific formatting instructions.""")
	# Fixed word vectors
	group.add('--fix_word_vecs_enc', '-fix_word_vecs_enc',
	          action='store_true',
	          help="Fix word embeddings on the encoder side.")
	group.add('--fix_word_vecs_dec', '-fix_word_vecs_dec',
	          action='store_true',
	          help="Fix word embeddings on the decoder side.")

	# Optimization options
	group = parser.add_argument_group('Optimization- Type')
	group.add('--batch_size', '-batch_size', type=int, default=32,
	          help='Maximum batch size for training')
	group.add('--batch_type', '-batch_type', default='sents',
	          choices=["sents", "tokens"],
	          help="""Batch grouping for batch_size. Standard
                               is sents. Tokens will do dynamic batching""")
	group.add('--normalization', '-normalization', default='sents',
	          choices=["sents", "tokens"],
	          help='Normalization method of the gradient.')
	group.add('--accum_count', '-accum_count', type=int, default=1,
	          help="""Accumulate gradient this many times.
                       Approximately equivalent to updating
                       batch_size * accum_count batches at once.
                       Recommended for Transformer.""")
	group.add('--valid_steps', '-valid_steps', type=int, default=10000,
	          help='Perfom validation every X steps')
	group.add('--valid_batch_size', '-valid_batch_size', type=int, default=32,
	          help='Maximum batch size for validation')
	group.add('--max_generator_batches', '-max_generator_batches',
	          type=int, default=2,
	          help="""Maximum batches of words in a sequence to run
                        the generator on in parallel. Higher is faster, but
                        uses more memory.""")
	group.add('--train_steps', '-train_steps', type=int, default=100000,
	          help='Number of training steps')
	group.add('--epochs', '-epochs', type=int, default=0,
	          help='Deprecated epochs see train_steps')
	group.add('--optim', '-optim', default='sgd',
	          choices=['sgd', 'adagrad', 'adadelta', 'adam',
	                   'sparseadam', 'adafactor'],
	          help="""Optimization method.""")
	group.add('--adagrad_accumulator_init', '-adagrad_accumulator_init',
	          type=float, default=0,
	          help="""Initializes the accumulator values in adagrad.
                       Mirrors the initial_accumulator_value option
                       in the tensorflow adagrad (use 0.1 for their default).
                       """)
	group.add('--max_grad_norm', '-max_grad_norm', type=float, default=5,
	          help="""If the norm of the gradient vector exceeds this,
                       renormalize it to have the norm equal to
                       max_grad_norm""")
	group.add('--dropout', '-dropout', type=float, default=0.3,
	          help="Dropout probability; applied in LSTM stacks.")
	group.add('--truncated_decoder', '-truncated_decoder', type=int, default=0,
	          help="""Truncated bptt.""")
	group.add('--adam_beta1', '-adam_beta1', type=float, default=0.9,
	          help="""The beta1 parameter used by Adam.
                       Almost without exception a value of 0.9 is used in
                       the literature, seemingly giving good results,
                       so we would discourage changing this value from
                       the default without due consideration.""")
	group.add('--adam_beta2', '-adam_beta2', type=float, default=0.999,
	          help="""The beta2 parameter used by Adam.
                       Typically a value of 0.999 is recommended, as this is
                       the value suggested by the original paper describing
                       Adam, and is also the value adopted in other frameworks
                       such as Tensorflow and Kerras, i.e. see:
                       https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
                       https://keras.io/optimizers/ .
                       Whereas recently the paper "Attention is All You Need"
                       suggested a value of 0.98 for beta2, this parameter may
                       not work well for normal models / default
                       baselines.""")
	group.add('--label_smoothing', '-label_smoothing', type=float, default=0.0,
	          help="""Label smoothing value epsilon.
                       Probabilities of all non-true labels
                       will be smoothed by epsilon / (vocab_size - 1).
                       Set to zero to turn off label smoothing.
                       For more detailed information, see:
                       https://arxiv.org/abs/1512.00567""")
	# learning rate
	group = parser.add_argument_group('Optimization- Rate')
	group.add('--learning_rate', '-learning_rate', type=float, default=1.0,
	          help="""Starting learning rate.
                       Recommended settings: sgd = 1, adagrad = 0.1,
                       adadelta = 1, adam = 0.001""")
	group.add('--learning_rate_decay', '-learning_rate_decay',
	          type=float, default=0.5,
	          help="""If update_learning_rate, decay learning rate by
                       this much if steps have gone past
                       start_decay_steps""")
	group.add('--start_decay_steps', '-start_decay_steps',
	          type=int, default=50000,
	          help="""Start decaying every decay_steps after
                       start_decay_steps""")
	group.add('--decay_steps', '-decay_steps', type=int, default=10000,
	          help="""Decay every decay_steps""")

	group.add('--decay_method', '-decay_method', type=str, default="none",
	          choices=['noam', 'rsqrt', 'none'],
	          help="Use a custom decay rate.")
	group.add('--warmup_steps', '-warmup_steps', type=int, default=4000,
	          help="""Number of warmup steps for custom decay.""")

	group = parser.add_argument_group('Logging')
	group.add('--report_every', '-report_every', type=int, default=50,
	          help="Print stats at this interval.")
	# Use TensorboardX for visualization during training
	group.add('--tensorboard', '-tensorboard', action="store_true",
	          help="""Use tensorboardX for visualization during training.
                       Must have the library tensorboardX.""")
	group.add_argument("--tensorboard_log_dir", "-tensorboard_log_dir",
	                   type=str, default="runs/onmt",
	                   help="""Log directory for Tensorboard.
                       This is also the name of the run.
                       """)


def translate_opts(parser):
	""" Translation / inference options """
	group = parser.add_argument_group('Model')

	group.add('--pass1', '-pass1', action="store_true",
	          help="true: translate w/ the first model, generating tmpl, "
	               "false: second")

	group.add('--model', '-model', dest='models', metavar='MODEL',
	          nargs='+', type=str, default=[], required=True,
	          help='Path to model .pt file(s). '
	               'Multiple models can be specified, '
	               'for ensemble decoding.')
	group.add('--avg_raw_probs', '-avg_raw_probs', action='store_true',
	          help="""If this is set, during ensembling scores from
              different models will be combined by averaging their
              raw probabilities and then taking the log. Otherwise,
              the log probabilities will be averaged directly.
              Necessary for models whose output layers can assign
              zero probability.""")

	group = parser.add_argument_group('Data')

	group.add('--src', '-src', required=True,
	          help="""Source sequence to decode (one line per
                       sequence)""")
	group.add('--tgt', '-tgt',
	          help='True target sequence (optional)')
	group.add('--src2', '-src2',
	          help="""Second source sequence to decode (one line per
	                   sequence)""")
	group.add('--tmpl', '-tmpl',
	          help='True template sequence (optional)')

	group.add('--max_src_len', '-max_src_len', type=int, default=200,
	          help="Maximum source sequence length")
	group.add('--max_tgt_len', '-max_tgt_len', type=int, default=60,
	          help="Maximum target sequence length to keep.")
	group.add('--shard_size', '-shard_size', type=int, default=10000,
	          help="""Divide src and tgt (if applicable) into
              smaller multiple src and tgt files, then
              build shards, each shard will have
              opt.shard_size samples except last shard.
              shard_size=0 means no segmentation
              shard_size>0 means segment dataset into multiple shards,
              each shard has shard_size samples""")

	group.add('--output', '-output', default='pred.txt',
	          help="""Path to output the predictions (each line will
                       be the decoded sequence""")

	group.add('--report_bleu', '-report_bleu', action='store_true',
	          help="""Report bleu score after translation,
                       call tools/multi-bleu.perl on command line""")
	group.add('--report_rouge', '-report_rouge', action='store_true',
	          help="""Report rouge 1/2/3/L/SU4 score after translation
                       call tools/test_rouge.py on command line""")
	group.add('--report_time', '-report_time', action='store_true',
	          help="Report some translation time metrics")

	# Options most relevant to summarization.
	group.add('--dynamic_dict', '-dynamic_dict', action='store_true',
	          help="Create dynamic dictionaries")
	group.add('--share_vocab', '-share_vocab', action='store_true',
	          help="Share source and target vocabulary")

	group = parser.add_argument_group('Random Sampling')
	group.add('--random_sampling_topk', '-random_sampling_topk',
	          default=1, type=int,
	          help="""Set this to -1 to do random sampling from full
                      distribution. Set this to value k>1 to do random
                      sampling restricted to the k most likely next tokens.
                      Set this to 1 to use argmax or for doing beam
                      search.""")
	group.add('--random_sampling_temp', '-random_sampling_temp',
	          default=1., type=float,
	          help="""If doing random sampling, divide the logits by
                       this before computing softmax during decoding.""")
	group.add('--seed', '-seed', type=int, default=829,
	          help="Random seed")

	group = parser.add_argument_group('Beam')
	group.add('--beam_size', '-beam_size', type=int, default=5,
	          help='Beam size')
	group.add('--min_length', '-min_length', type=int, default=0,
	          help='Minimum prediction length')
	group.add('--max_length', '-max_length', type=int, default=100,
	          help='Maximum prediction length.')

	# Alpha and Beta values for Google Length + Coverage penalty
	# Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
	group.add('--stepwise_penalty', '-stepwise_penalty', action='store_true',
	          help="""Apply penalty at every decoding step.
                       Helpful for summary penalty.""")
	group.add('--length_penalty', '-length_penalty', default='none',
	          choices=['none', 'wu', 'avg'],
	          help="""Length Penalty to use.""")
	group.add('--coverage_penalty', '-coverage_penalty', default='none',
	          choices=['none', 'wu', 'summary'],
	          help="""Coverage Penalty to use.""")
	group.add('--alpha', '-alpha', type=float, default=0.,
	          help="""Google NMT length penalty parameter
                        (higher = longer generation)""")
	group.add('--beta', '-beta', type=float, default=-0.,
	          help="""Coverage penalty parameter""")
	group.add('--block_ngram_repeat', '-block_ngram_repeat',
	          type=int, default=0,
	          help='Block repetition of ngrams during decoding.')
	group.add('--ignore_when_blocking', '-ignore_when_blocking',
	          nargs='+', type=str, default=[],
	          help="""Ignore these strings when blocking repeats.
                       You want to block sentence delimiters.""")
	group.add('--replace_unk', '-replace_unk', action="store_true",
	          help="""Replace the generated UNK tokens with the
                       source token that had highest attention weight. If
                       phrase_table is provided, it will lookup the
                       identified source token and give the corresponding
                       target token. If it is not provided(or the identified
                       source token does not exist in the table) then it
                       will copy the source token""")

	group = parser.add_argument_group('Logging')
	group.add('--verbose', '-verbose', action="store_true",
	          help='Print scores and predictions for each sentence')
	group.add('--log_file', '-log_file', type=str, default="",
	          help="Output logs to a file under this path.")
	group.add('--attn_debug', '-attn_debug', action="store_true",
	          help='Print best attn for each word')
	group.add('--dump_beam', '-dump_beam', type=str, default="",
	          help='File to dump beam information to.')
	group.add('--n_best', '-n_best', type=int, default=1,
	          help="""If verbose is set, will output the n_best
                       decoded sentences""")

	group = parser.add_argument_group('Efficiency')
	group.add('--batch_size', '-batch_size', type=int, default=32,
	          help='Batch size')
	group.add('--gpu', '-gpu', type=int, default=-1,
	          help="Device to run on")
