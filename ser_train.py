import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from data_util import SERUsingAI, ser_collate_fn
from ser import SER_CPC_RNN
from config import Config

def ser_train(model:nn.Module,
              training_set:Dataset,
              val_set:Dataset,
              batch_size,
              learning_rate,
              epoches,
              device='cpu',
              ckpt_path=None):
	model.to(device)

	if ckpt_path is not None:
		ckpt = torch.load(ckpt_path, map_location=torch.device(device))
		model.load_state_dict(ckpt['model_state_dict'])


	training_loader = DataLoader(training_set, batch_size, shuffle=True, collate_fn=ser_collate_fn)
	val_loader = DataLoader(val_set, batch_size, shuffle=True, collate_fn=ser_collate_fn)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=learning_rate, weight_decay=Config.SER_WEIGHT_DECAY
	)


	for e in range(epoches):
		for i, (signal_train, y_train, lengths) in enumerate(training_loader):

			torch.cuda.empty_cache()
			model.train()
			signal_train = signal_train.to(device)
			y_train = y_train.to(device)

			optimizer.zero_grad()
			pred_train = model(signal_train, lengths)
			loss_train = criterion(pred_train, y_train)
			loss_train.backward()
			optimizer.step()


			del signal_train
			if i % 10 == 0:
				with torch.no_grad():
					model.eval()
					signal_val, y_val, lengths = next(iter(val_loader))
					signal_val = signal_val.to(device)
					y_val = y_val.to(device)
					pred_val = model(signal_val, lengths)
					loss_val = criterion(pred_val, y_val)

					# acc_train
					acc_train = torch.sum( (pred_train>0) == y_train ).item() / batch_size
					acc_val = torch.sum( (pred_val>0) == y_val ).item() / batch_size

					print()
					print('%d epochs, %d/%d' % (e, i, len(training_loader)))
					print('training loss: %.5f, training acc: %.5f' % (loss_train, acc_train))
					print('val      loss: %.5f, val      acc: %.5f' % (loss_val, acc_val))
					print()

		if e % 5 == 0:
			torch.save(
				{ 'model_state_dict' : model.state_dict() },
				'ser_%d.pth' % (e)
			)


def overfit(model:nn.Module,
              training_set:Dataset,
              batch_size,
              learning_rate,
              device='cpu'):
	model.to(device)
	training_loader = DataLoader(training_set, batch_size, shuffle=True, collate_fn=ser_collate_fn)

	criterion = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(
		filter(lambda p: p.requires_grad, model.parameters()),
		lr=learning_rate, # weight_decay=Config.SER_WEIGHT_DECAY
	)


	for i, (signal_train, y_train, lengths) in enumerate(training_loader):

		model.train()
		signal_train = signal_train.to(device)
		y_train = y_train.to(device)

		for _ in range(10000):
			optimizer.zero_grad()
			pred_train = model(signal_train, lengths)
			loss_train = criterion(pred_train, y_train)
			loss_train.backward()
			optimizer.step()

			acc = torch.sum((pred_train > 0) == y_train).item() / batch_size
			print(loss_train.item(), acc)




if __name__ == '__main__':
	training_set = SERUsingAI(train=True)
	val_set = SERUsingAI(train=False)
	model = SER_CPC_RNN(Config.CPC_CKPT_PATH)
	ser_train(
		model,
		training_set,
		val_set,
		batch_size=Config.SER_BATCH_SIZE,
		learning_rate=Config.SER_LEARNING_RATE,
		epoches=100,
		device=Config.DEVICE,
		ckpt_path=Config.SER_CKPT_PATH
	)

	# overfit(model, training_set, 8, 1e-4, Config.DEVICE)


