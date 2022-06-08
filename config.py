import torch

class Config:

	# global parameters
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	RANDOM_SEED = 826


	# parameters across the whole pipeline
	SAMPLE_RATE = 16000
	DOWNSAMPLING_FACTOR = 160 # this is hard-coded into the CNN architecture of CPC


	# CPC pretraining data (CASIA) parameters
	CASIA_LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
	CASIA_LABELS_DICT = {
		l: i for (i, l) in enumerate(CASIA_LABELS)
	}
	CASIA_DATA_PATH = './CASIA'

	CPC_NUM_SAMPLE = 8
	CPC_TIMESTEP = 4
	CPC_CNN_DIM = 512
	CPC_RNN_DIM = 512
	CPC_RNN_LAYERS = 2
	CPC_LEARNING_RATE = 1e-3



