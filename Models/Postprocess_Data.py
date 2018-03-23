import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

##############################################################################
parameter = 'alpha'
channel = 'red'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_3_classes_LeNet_150/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_3_classes_VGG16_150/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_10_classes_LeNet_150/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_10_classes_VGG16_224/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_5_classes_VGG16_150/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_10_classes_VGG16_150/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_5_classes_VGG16_224/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_10_classes_VGG16_224_preprocessed/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_10_classes_VGG16_224_no_preprocessing/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_10_classes_VGG16_224_sgd/'
# call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_10_classes_VGG19_224_no_preprocessing/'
call_folder = '/home/dario/Desktop/SemThes/Models/model_r_alpha_10_classes_VGG16_150/'
##############################################################################

avg_loss_plot = np.load(call_folder + channel + '_' + parameter + '_avg_loss_plot.npy')
val_accuracy_plot = np.load(call_folder + channel + '_' + parameter + '_val_accuracy_plot.npy')
test_accuracy_plot = np.load(call_folder + channel + '_' + parameter + '_test_accuracy_plot.npy')

plt.figure(1)
plt.plot(range(1, len(avg_loss_plot) + 1), avg_loss_plot)
plt.title('Training loss for ' + channel + '_' + parameter)
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
plt.savefig(call_folder + channel + '_' + parameter + '_TrainLoss.jpg')

plt.figure(2)
plt.plot(range(1, len(val_accuracy_plot) + 1), val_accuracy_plot, label='Validation Accuracy')
plt.plot(range(1, len(test_accuracy_plot) + 1), test_accuracy_plot, label='Test Accuracy')
plt.title('Accuracy for ' + channel + '_' + parameter)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(call_folder + channel + '_' + parameter + '_Acccuracy.jpg')
