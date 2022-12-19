import torch
from torch import nn
import numpy as np

from config import Config

# A pytorch implementation of contrastive predictive coding
# from https://arxiv.org/pdf/1807.03748.pdf

# We will pretrain the cnn encoder for raw audio feature extraction
# (and the rnn if we use rnn for downstream task)
# on CASIA dataset, and transfer the cnn encoder (and rnn) to downstream task.
class CPC(nn.Module):
	def __init__(self):
		super(CPC, self).__init__()
		self.time_step = Config.CPC_TIMESTEP
		self.cnn_dim = Config.CPC_CNN_DIM
		self.rnn_dim = Config.CPC_RNN_DIM
		self.rnn_layers = Config.CPC_RNN_LAYERS

		self.cnn = nn.Sequential(
			nn.Conv1d(1, 64, kernel_size=11, stride=5, padding=3, bias=False),  # downsampling by 5
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(0.05),

			nn.Conv1d(64, 128, kernel_size=11, stride=5, padding=3, bias=False),  # downsampling by 5
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Dropout(0.05),

			nn.Conv1d(128, 128, kernel_size=11, stride=5, padding=3, bias=False),  # downsampling by 5
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Dropout(0.05),

			nn.Conv1d(128, 128, kernel_size=8, stride=4, padding=2, bias=False),  # downsampling by 4
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Dropout(0.05),

			nn.Conv1d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(0.05),

			nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2, bias=False),  # downsampling by 4
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Dropout(0.1),

			nn.Conv1d(128, self.cnn_dim, kernel_size=8, stride=4, padding=2, bias=False),  # downsampling by 4
			nn.BatchNorm1d(self.cnn_dim),
			nn.ReLU(),


		)  # downsampling by factor=8000

		self.rnn = nn.GRU(
			input_size=self.cnn_dim,
			hidden_size=self.rnn_dim,
			num_layers=self.rnn_layers,
			bidirectional=False,
			batch_first=True)
		self.Wk = nn.ModuleList([
			nn.Linear(self.rnn_dim, self.cnn_dim) for _ in range(self.time_step)
		])
		# self.softmax = nn.Softmax(dim=-1)
		self.log_softmax = nn.LogSoftmax(dim=0)

		self.to(Config.DEVICE)
		self.device = Config.DEVICE

	def forward(self, x):
		"""
		:param x: [batch_size, T]
		:return:
		:rtype:
		"""
		# extract feature representation
		batch_size, T = x.shape
		x = x[:, None, :] # [batch_size, 1, T]
		z = self.cnn(x)  # [batch_size, cnn_dim, seq]
		z = torch.transpose(z, 1, 2)  # [batch_size, seq, cnn_dim]

		# sample time step t
		t = np.random.randint(0, T // Config.DOWNSAMPLING_FACTOR - self.time_step)
		z_tk = z[:, np.arange(t + 1, t + self.time_step + 1), :]  # [batch_size, time_step, cnn_dim]

		# extract ct
		cts, _ = self.rnn(z[:, :t+1, :]) # [batch_size, prev_len, rnn_dim]
		ct = cts[:, -1, :] # [batch_size, rnn_dim]
		wk_ct = torch.stack([
			linear(ct) for linear in self.Wk
		], dim=0) # # [time_step, batch_size, cnn_dim]

		# compute InfoNCE loss and prediction accuracy
		nce = 0
		acc = 0
		for k in range(self.time_step): # TODO: broadcast this for-loop
			# use z_tk from other batches as random samples
			total = torch.mm(z_tk[:, k,:], wk_ct[k,:,:].transpose(0,1)) # [batch_size, batch_size]
			nce -= torch.sum(
				torch.diag(
					self.log_softmax(total)
				)
			) / batch_size

			acc += torch.sum(
				torch.argmax(total, dim=0) == torch.arange(batch_size).to(self.device)
			).item() / batch_size

		acc = acc / self.time_step
		nce = nce / self.time_step
		return nce, acc