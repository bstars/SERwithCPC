import torch

class Config:

	# global parameters
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	RANDOM_SEED = 826 # Makes sure the consistent train/test split


	# parameters across the whole pipeline
	SAMPLE_RATE = 16000
	DOWNSAMPLING_FACTOR = 8000 # this is hard-coded into the CNN architecture of CPC




	# Unlabeled dataset for cpc pretraining
	UNLABELED_DATA_PATH = './SER'
	CPC_NUM_SAMPLE = 16
	CPC_TIMESTEP = 12
	CPC_CNN_DIM = 256
	CPC_RNN_DIM = 256
	CPC_RNN_LAYERS = 2
	CPC_LEARNING_RATE = 1e-4
	CPC_WEIGHT_DECAY = 1e-4

	CPC_CKPT_PATH = './ckpts/cpc_90.pth'


	# SER dataset parameters
	SER_LABELS = ['normal', 'abnormal']
	SERLABELS_DICT = {
		l: i for (i, l) in enumerate(SER_LABELS)
	}
	SER_DATA_PATH = './SER'
	train_split = './splits/3/train_split.txt'
	test_split = './splits/3/test_split.txt'
	SER_CUT_RATIO_FRONT = 0.2
	SER_CUT_RATIO_END = 0.3
	SER_NORM_CUT_THRESHOLD = 0.015


	# SER RNN parameters if we use RNN as decoder
	SER_BATCH_SIZE = 2
	SER_RNN_DIM = 256
	SER_RNN_LAYERS = 1

	# SER training parameters
	SER_LEARNING_RATE = 1e-4
	SER_WEIGHT_DECAY = 1e-5



	# evaluate parameters
	SER_CKPT_PATH = './ckpts/ser_145.pth'


