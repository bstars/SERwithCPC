import os
import re
import sys
import librosa

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import Config




def get_data_path():
	"""
	Get all wav file path
	:param data_path:
	:param class_labels:
	:type class_labels:
	:return:
	"""
	paths = []
	labels = []

	for i in range(1,5):
		for label in Config.CASIA_LABELS:
			for j in range(201, 251):
				paths.append(
					os.path.join(Config.CASIA_DATA_PATH, str(i), label, str(j) + '.wav')
				)
				labels.append(Config.CASIA_LABELS_DICT[label])
	return paths, labels

class CasiaCPCPretrain(Dataset):
	def __init__(self, train=True):
		super(CasiaCPCPretrain, self).__init__()
		paths, labels = get_data_path()
		path_train, path_test, y_train, y_test = train_test_split(paths, labels, test_size=0.2, random_state=Config.RANDOM_SEED)

		self.paths = path_train if train else path_test
		self.labels = y_train if train else y_test

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):
		signal, _ = librosa.load(self.paths[idx], sr=Config.SAMPLE_RATE)
		return signal

def casia_collate_fn(batches):
	"""
	Randomly cut a fragment from each signal
	:param batches:
	:return:
	"""

	start_time = [
		np.random.randint(0, len(signal) - (Config.CPC_TIMESTEP + 1) * Config.DOWNSAMPLING_FACTOR) for signal in batches
	]

	batches = [
		batches[i][start_time[i]:] for i in range(len(batches))
	]

	min_length = np.min([
		len(x) for x in batches
	])

	min_length = min_length - min_length % Config.DOWNSAMPLING_FACTOR # make the sequence length a multiply of the downsampling rate
	x = np.stack([
		signal[:min_length] for signal in batches
	])

	return torch.Tensor(x)


if __name__ == '__main__':
	ds = CasiaCPCPretrain(train=False)
	dl = DataLoader(ds, batch_size=3, collate_fn=casia_collate_fn)

	for x in dl:
		print(x.shape)
		break