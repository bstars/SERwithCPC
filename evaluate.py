import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score, f1_score, roc_curve
from scipy.io import savemat
import time



from ser import SER_CPC_RNN
from data_util import SERUsingAI, ser_collate_fn, NewDataset
from config import Config




class Predictor():
	def __init__(self, model:nn.Module):
		self.model = model


	def predict(self, fname, device=Config.DEVICE):
		self.model.to(device)
		self.model.eval()
		signal, _ = librosa.load(fname, sr=Config.SAMPLE_RATE)
		idx = np.where(np.abs(signal) >= Config.SER_NORM_CUT_THRESHOLD)[0][0]
		signal = signal[idx:]
		# start_idx = int(len(signal) * Config.SER_CUT_RATIO_FRONT)
		start_idx = 0
		end_idx = -int(len(signal) * Config.SER_CUT_RATIO_END)
		signal = signal[start_idx: end_idx]
		signal = torch.Tensor(signal).to(device)[None,:]

		pred = self.model(signal, np.array([len(signal)]))
		pred = torch.sigmoid(pred)
		score = pred.cpu().detach().numpy()[0]
		return score # a score between 0 and 1




def evaluate2(predictor:Predictor, ds:SERUsingAI or NewDataset, device=Config.DEVICE):

	wrong_paths = []

	num_correct = 0
	num_all = 0
	ys = []
	scores = []
	for i in range(len(ds)):
		signal, label = ds[i]
		path = ds.paths[i]
		score = predictor.predict(path, device)

		scores.append(score)
		ys.append(label)

		num_all += 1
		num_correct += int(label == (score >= 0.5))

		if label != (score >= 0.5):
			wrong_paths.append(path)

		print(path)
		print('%d/%d = %.5f, y=%d' % (num_correct, num_all, num_correct/num_all, label))

	# print('wrong paths')
	# for path in wrong_paths:
	# 	print(path)

	ys = np.array(ys)
	scores = np.array(scores)

	# accuracy
	print(
		'Accuracy: %.5f' % (np.sum(ys == (scores>=0.5))/ len(ys))
	)

	# recall
	print(
		"Recall: %.5f" % (
			recall_score(ys, scores>=0.5)
		)
	)

	# auc
	fpr, tpr, thresholds = roc_curve(ys, scores)
	auc = roc_auc_score(ys, scores)
	print(
		"AUC: %.5f" % (
			auc
		)
	)
	# plt.plot(fpr, tpr, label="%.4f" % (auc))

	# f1
	print(
		"F1: %.5f" % (
			f1_score(ys, scores>=0.5)
		)
	)

	# confusion matrix
	print(confusion_matrix(ys, scores>=0.5))







if __name__ == '__main__':
	tic = time.time()
	model = SER_CPC_RNN(cpc_ckpt_path=None)
	model.load_ckpt(Config.SER_CKPT_PATH)
	ds = SERUsingAI(train=False)
	# ds = NewDataset('./newdataset')

	evaluate2(Predictor(model), ds)
	toc=time.time()
	print(toc-tic)

