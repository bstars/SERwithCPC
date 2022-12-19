import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from data_util import Unlabeled, unlabeled_collate_fn
from cpc import CPC
from config import Config

def cpc_pretrain(model:CPC,
                 training_set:Dataset,
                 val_set:Dataset,
                 batch_size,
                 learning_rate,
                 epoches,
                 device='cpu'):

	model.to(device)
	training_loader = DataLoader(training_set, batch_size, shuffle=True, collate_fn=unlabeled_collate_fn)
	val_loader = DataLoader(val_set, batch_size, shuffle=True, collate_fn=unlabeled_collate_fn)

	optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=Config.CPC_WEIGHT_DECAY)

	for e in range(epoches):
		for i, x in enumerate(training_loader):
			# for _ in range(1000):
			model.train()
			x = x.to(device)

			optimizer.zero_grad()
			training_nce, training_acc = model(x)
			training_nce.backward()
			optimizer.step()

			if i % 5 == 0:
				model.eval()
				with torch.no_grad():
					x_val = next(iter(val_loader))
					x_val = x_val.to(device)
					val_nce, val_acc = model(x_val)

				print()
				print('%d epochs, %d/%d' % (e, i, len(training_loader)))
				print('training InfoNCE: %.5f, training acc: %.5f' % (training_nce, training_acc))
				print('val      InfoNCE: %.5f, val      acc: %.5f' % (val_nce, val_acc))
				print()

		if e % 10 == 0:
			torch.save(
				{ 'model_state_dict' : model.state_dict() },
				'cpc_%d.pth' % (e)
			)

if __name__ == '__main__':
	training_set = Unlabeled(train=True)
	val_set = Unlabeled(train=False)
	model = CPC()
	ckpt = torch.load('./cpc_90.pth', map_location=torch.device(Config.DEVICE))
	model.load_state_dict(ckpt['model_state_dict'])
	cpc_pretrain(model,
	             training_set,
	             val_set,
	             batch_size=Config.CPC_NUM_SAMPLE,
	             learning_rate=Config.CPC_LEARNING_RATE,
	             epoches=100,
	             device=Config.DEVICE)