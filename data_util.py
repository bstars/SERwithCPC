import os
import re
import sys
import librosa
from imblearn.over_sampling import RandomOverSampler


import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from config import Config


class Unlabeled(Dataset):
	def __init__(self, train = True):
		super(Unlabeled, self).__init__()
		file = open(
			'train_split.txt' if train else 'test_split.txt'
		)
		self.paths = [l.strip() for l in file.readlines()]

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):
		signal, _ = librosa.load(self.paths[idx], sr=Config.SAMPLE_RATE)
		idx = np.where(np.abs(signal) >= Config.SER_NORM_CUT_THRESHOLD)[0][0]
		signal = signal[idx:]
		return signal


def unlabeled_collate_fn(batches):

	start_time = [
		np.random.randint(0, len(signal) - (Config.CPC_TIMESTEP + 1) * Config.DOWNSAMPLING_FACTOR) for signal in batches
	]

	batches = [
		batches[i][start_time[i]:] for i in range(len(batches))
	]

	min_length = np.min([
		len(x) for x in batches
	])

	min_length = min_length - min_length % Config.DOWNSAMPLING_FACTOR  # make the sequence length a multiply of the downsampling rate
	x = np.stack([
		signal[:min_length] for signal in batches
	])

	return torch.Tensor(x)



def ser_train_test_split():
	normal_paths = []
	abnormal_paths = []

	for path, folder, files in sorted(os.walk(os.path.join(Config.SER_DATA_PATH, 'normal'))):
		for fname in files:
			if '.wav' in fname:
				normal_paths.append(os.path.join(path, fname))

	for path, folder, files in sorted(os.walk(os.path.join(Config.SER_DATA_PATH, 'abnormal'))):
		for fname in files:
			if '.wav' in fname:
				abnormal_paths.append(os.path.join(path, fname))


	# trainging split
	file = open('train_split.txt', 'w')
	for i in range(200):
		file.write(normal_paths[i] + '\n')
		file.write(abnormal_paths[i] + '\n')
	file.close()

	# testing split
	file  = open('test_split.txt', 'w')
	for i in range(200, len(normal_paths)):
		file.write(normal_paths[i] + '\n')

	for i in range(200, len(abnormal_paths)):
		file.write(abnormal_paths[i] + '\n')
	file.close()


class SERUsingAI(Dataset):
	def __init__(self, train=True):
		file = open(
			Config.train_split if train else Config.test_split
		)

		self.paths = [l.strip() for l in file.readlines()]

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx, mel_spectrom=False):
		path = self.paths[idx]
		signal, _ = librosa.load(self.paths[idx], sr=Config.SAMPLE_RATE)
		idx = np.where(np.abs(signal) >= Config.SER_NORM_CUT_THRESHOLD)[0][0]
		signal = signal[idx:]

		if 'abnormal' in path or 'newdataset' in path:
			label = Config.SERLABELS_DICT['abnormal']
		else:
			label = Config.SERLABELS_DICT['normal']

		if mel_spectrom:
			pass
		else:
			return torch.Tensor(signal), label

def ser_collate_fn(batches, eval=False):
	signals, labels, lengths = [], [], []
	for (signal, y) in batches:

		if not eval:

			# start_idx = int(
			# 	len(signal) * np.random.uniform(Config.SER_CUT_RATIO_FRONT - 0.075,
			# 	                                Config.SER_CUT_RATIO_FRONT + 0.075)
			# )
			start_idx = 0
			end_idx = -int(
				len(signal) * np.random.uniform(Config.SER_CUT_RATIO_END - 0.075, Config.SER_CUT_RATIO_END + 0.075)
			)
		else:
			# start_idx = int(len(signal) * Config.SER_CUT_RATIO_FRONT)
			start_idx = 0
			end_idx = -int(len(signal) * Config.SER_CUT_RATIO_END)

		signal = signal[start_idx: end_idx]
		lengths.append(len(signal))
		signals.append(signal)
		labels.append(y)

	signals = pad_sequence(signals, batch_first=True)
	return signals, torch.Tensor(labels), np.array(lengths).astype(int)

class NewDataset(Dataset):
	def __init__(self, path):
		super(NewDataset, self).__init__()
		self.paths = []
		for root, subdir, files in os.walk(path):
			for file in files:
				self.paths.append(os.path.join(path, file))

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):
		signal, _ = librosa.load(self.paths[idx], sr=Config.SAMPLE_RATE)
		idx = np.where(np.abs(signal) >= Config.SER_NORM_CUT_THRESHOLD)[0][0]
		signal = signal[idx:]
		return torch.Tensor(signal), Config.SERLABELS_DICT['abnormal']




if __name__ == '__main__':
	# ser_train_test_split()
	# ds = SERUsingAI(train=True)
	# for signal, label in ds:
	# 	print(label)

	# ds = NewDataset('./CASIA')
	# print(ds.paths)

	temp_string = './SER/abnormal/2018082875560637345.wav'
	label = 1 if 'abnormal' in temp_string or 'newdataset' in temp_string else 0
	print(label)



