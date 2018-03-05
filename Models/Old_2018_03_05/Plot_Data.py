import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

##############################################################################
Name = 'r_alpha'
##############################################################################

avg_loss_plot = np.load(Name + '_avg_loss_plot.npy')
val_accuracy_plot = np.load(Name + '_val_accuracy_plot.npy')
test_accuracy_plot = np.load(Name + '_test_accuracy_plot.npy')

plt.figure(1)
plt.plot(range(1, len(avg_loss_plot) + 1), avg_loss_plot)
plt.title('Training loss for ' + Name)
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
plt.savefig(Name + '_TrainLoss.jpg')

plt.figure(2)
plt.plot(range(1, len(val_accuracy_plot) + 1), val_accuracy_plot, label='Validation Accuracy')
plt.plot(range(1, len(test_accuracy_plot) + 1), test_accuracy_plot, label='Test Accuracy')
plt.title('Accuracy for ' + Name)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(Name + '_Acccuracy.jpg')
