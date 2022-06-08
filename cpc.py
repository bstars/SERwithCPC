import torch
from torch import nn
import numpy as np

# A pytorch implementation of contrastive predictive coding
# from https://arxiv.org/pdf/1807.03748.pdf

# We will pretrain the cnn encoder for raw audio feature extraction
# (and the rnn if we use rnn for downstream task)
# on CASIA dataset, and transfer the cnn encoder (and rnn) to downstream task.
class CPC(nn.Module):
	def __init__(self, time_step, cnn_dim=256, rnn_dim=256, device='cpu'):
		super(CPC, self).__init__()
		self.time_step = time_step
		self.cnn_dim = cnn_dim
		self.rnn_dim = rnn_dim

		self.encoder = nn.Sequential(
			nn.Conv1d(1, 64, kernel_size=11, stride=5, padding=3, bias=False),  # downsampling by 5
			nn.BatchNorm1d(64),
			nn.ReLU(),

			nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2, bias=False),  # downsampling by 4
			nn.BatchNorm1d(128),
			nn.ReLU(),

			nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # downsampling by 2
			nn.BatchNorm1d(256),
			nn.ReLU(),

			nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),  # downsampling by 2
			nn.BatchNorm1d(256),
			nn.ReLU(),

			nn.Conv1d(256, cnn_dim, kernel_size=4, stride=2, padding=1, bias=False),  # downsampling by 2
			nn.BatchNorm1d(cnn_dim),
			nn.ReLU(),
		)  # downsampling by factor=160

		self.rnn = nn.GRU(input_size=cnn_dim, hidden_size=rnn_dim, num_layers=2, bidirectional=False, batch_first=True)
		self.Wk = nn.ModuleList([
			nn.Linear(rnn_dim, cnn_dim) for _ in range(time_step)
		])
		# self.softmax = nn.Softmax(dim=-1)
		self.log_softmax = nn.LogSoftmax(dim=0)

		self.to(device)
		self.device = device

	def forward(self, x):
		"""
		:param x: [batch_size, T]
		:return:
		:rtype:
		"""
		# extract feature representation
		batch_size, T = x.shape
		x = x[:, None, :] # [batch_size, 1, T]
		z = self.encoder(x)  # [batch_size, cnn_dim, seq]
		z = torch.transpose(z, 1, 2)  # [batch_size, seq, cnn_dim]

		# sample time step t
		t = np.random.randint(0, T // 160 - self.time_step)
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

if __name__ == '__main__':
	x = torch.randn(3, 16000)
	model = CPC(4)
	model(x)





