import numpy as np
import torch
from torch import nn

from cpc import CPC
from config import Config

class SER_CPC_RNN(nn.Module):
	def __init__(self, cpc_ckpt_path=None):
		super(SER_CPC_RNN, self).__init__()
		self.cpc = CPC()

		# load the pretrained cpc encoder and freeze the cpc parameters
		# note: use filter( lambda p : p.requires_grad, model.parameters()) during model training

		if cpc_ckpt_path is not None:
			ckpt = torch.load(cpc_ckpt_path, map_location=torch.device(Config.DEVICE))
			self.cpc.load_state_dict(ckpt['model_state_dict'])
			for p in self.cpc.parameters():
				p.requires_grad = False

		self.rnn_decoder = nn.GRU(
			input_size=Config.CPC_CNN_DIM,
			hidden_size=Config.SER_RNN_DIM,
			num_layers=Config.SER_RNN_LAYERS,
			bidirectional=False,
			batch_first=True
		)

		self.prediction = nn.Sequential(
			nn.Linear(Config.SER_RNN_DIM, 64), nn.ReLU(),
			nn.Linear(64, 16), nn.ReLU(),
			nn.Linear(16, 1)
		)
		self.to(Config.DEVICE)

	def forward(self, x, lengths):
		"""
		:param x: torch.Tensor, [batch_size, T]
		:param lengths: np.array, [batch_size]
		:return:
		"""

		batch_size, T = x.shape

		x = x[:,None,:] # [batch_size, 1, T]
		x = self.cpc.cnn(x) # [batch_size, cnn_dim, seq]
		x = torch.transpose(x, 1,2) # [batch_size, seq, cnn_dim]

		x, hidden = self.rnn_decoder(x) # [batch_size, seq, ser_rnn_dim]
		del hidden

		x = x[
			np.arange(batch_size), lengths // Config.DOWNSAMPLING_FACTOR - 1, :
		] # [batch_size, ser_rnn_dim]
		x = self.prediction(x) # [batch_size, 1]

		return x[:,0]

	def load_ckpt(self, path):
		ckpt = torch.load(path, map_location=torch.device(Config.DEVICE))
		self.load_state_dict(ckpt['model_state_dict'])



if __name__ == '__main__':
	batch_size, T = 2, 320000
	x = torch.randn(batch_size, T)
	model = SER_CPC_RNN()
	y = model(x, np.array([32000, 64000]))
	print(y.shape)
