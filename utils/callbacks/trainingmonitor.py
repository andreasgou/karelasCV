# import the necessary packages
from keras.callbacks import BaseLogger
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import math
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
	def __init__(self, title, modPath, figPath, logPath=None, startAt=0):
		# store the output path for the figure, the path to the JSON
		# serialized file, and the starting epoch
		super(TrainingMonitor, self).__init__()
		self.modPath = modPath
		self.figPath = figPath
		self.logPath = logPath
		self.startAt = startAt
		self.best_metrics = {"val_loss": 1.0}
		self.title = title
		
		# initialize the history dictionary
		self.H = {}
		if self.logPath is not None:
			if os.path.exists(self.logPath):
				self.H = json.loads(open(self.logPath).read())
				self.best_metrics["val_loss"] = min(self.H["val_loss"])

	def get_epochs_trained(self):
		if "loss" in self.H:
			return len(self.H["loss"])
		else:
			return 0

	def plot_history(self, show=True, save=False):
		if "loss" in self.H:
			lr = round(self.H["lr"][-1], 5)

			# plot init
			# plt.close()
			plt.style.use("ggplot")
			plt.figure(figsize=(12.8, 7.2))
			plt.suptitle("{} [Epochs {}]".format(self.title, len(self.H["loss"])))
			N = np.arange(0, len(self.H["loss"]))
			
			# subplot training metrics
			fig = plt.subplot(121)
			plt.title("Metrics")
			plt.plot(N+1, self.H["loss"], label="train_loss")
			plt.plot(N+1, self.H["val_loss"], label="val_loss")
			plt.plot(N+1, self.H["acc"], label="train_acc")
			plt.plot(N+1, self.H["val_acc"], label="val_acc")
			
			# spot min val_loss and val_acc epoch with an arrow
			val_loss_best = min(self.H["val_loss"])
			val_loss_epoch = self.H["val_loss"].index(val_loss_best) + 1
			plt.annotate(str((val_loss_epoch, round(val_loss_best, 5))),
			             xy=(val_loss_epoch, val_loss_best),
			             xytext=(-50, 50),
			             textcoords='offset pixels',
			             arrowprops=dict(facecolor='black', shrink=0.05))
			val_acc_best = max(self.H["val_acc"])
			val_acc_epoch = self.H["val_acc"].index(val_acc_best) + 1
			plt.annotate(str((val_acc_epoch, round(val_acc_best, 5))),
			             xy=(val_acc_epoch, val_acc_best),
			             xytext=(-50, -50),
			             textcoords='offset pixels',
			             arrowprops=dict(facecolor='black', shrink=0.05))

			# paint val_loss improvement areas
			s2 = np.zeros(len(self.H["val_loss"]))
			vlm = 9
			for idx, vl in np.ndenumerate(np.asarray(self.H["val_loss"])):
				if vl < vlm:
					s2[idx] = vl
					vlm = vl
			
			ax = fig.axes
			ax.fill_between(N+1, 0, y2=s2, step='pre', color='xkcd:lavender')
			# collection = collections.BrokenBarHCollection.span_where(
			# 	N, ymin=0, ymax=val_loss_min, where=s2, facecolor='xkcd:lavender')
			# ax.add_collection(collection)
			
			plt.xlabel("Epoch #")
			plt.ylabel("Loss/Accuracy")
			plt.legend()
			
			# subplot hyper params
			N = np.arange(0, len(self.H["lr"]))
			plt.subplot(122)
			plt.title("Hyperparams")
			plt.plot(N+1, self.H["lr"], label="learning_rate")
			plt.xlabel("Epoch #")
			plt.ylabel("Learning Rate")
			plt.legend()
			
			plt.subplots_adjust(hspace=0.25, wspace=0.5)
			
			if save:
				plt.savefig(self.figPath)

			if show:
				plt.show()

	def on_train_begin(self, logs={}):
		# Load the training history
		if self.logPath is not None:
			# check to see if a starting epoch was supplied
			if self.startAt > 0:
				# loop over the entries in the history log and
				# trim any entries that are past the starting
				# epoch
				for k in self.H.keys():
					self.H[k] = self.H[k][:self.startAt]

	def on_epoch_end(self, epoch, logs={}):
		# loop over the logs and update the loss, accuracy, etc.
		# for the entire training process
		for (k, v) in logs.items():
			l = self.H.get(k, [])
			l.append(v)
			self.H[k] = l
		
		# append current learning rate to metrics
		# read the current lr after decay applied (required)
		lr = self.model.optimizer.lr
		decay = self.model.optimizer.decay
		iterations = self.model.optimizer.iterations
		lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
		# print("\n\nLearning Rate:", K.eval(lr), "Decay: ", K.eval(decay), "lr_with_decay:", K.eval(lr_with_decay))

		l = self.H.get("lr", [])
		l.append(K.eval(lr_with_decay).item())
		self.H["lr"] = l
		
		# Check validation loss improvements
		epochs_trained = self.get_epochs_trained()
		old_val = round(self.best_metrics["val_loss"], 5)
		new_val = round(logs["val_loss"], 5)
		if (new_val < old_val):
			print("\n[INFO] Epoch {}: val_loss improved from {} to {}, saving model to {}".format(
				epochs_trained, old_val, new_val, self.modPath
			))
			self.best_metrics["val_loss"] = logs["val_loss"]
			self.model.save(self.modPath)
		else:
			print("\n[INFO] Epoch {}: val_loss did not improved. Model is not stored".format(epochs_trained))
			
		
		# check to see if the training history should be serialized
		# to file
		if self.logPath is not None:
			f = open(self.logPath, "w")
			f.write(json.dumps(self.H))
			f.close()

		# ensure at least X epochs have passed before plotting ?
		# (epoch starts at zero)
		if len(self.H["loss"]) > 0:
			self.plot_history(show=False)

			# save the figure
			plt.savefig(self.figPath)
			plt.close()
