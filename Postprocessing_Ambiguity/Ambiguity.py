import os
import numpy as np
import fnmatch
from PIL import Image

call_folder = '../Data_Ambiguity/Images'
classes = 16

nr_images = len(fnmatch.filter(os.listdir(call_folder), '*.jpg'))
print('Total number of images: ', nr_images)


for i in range(3): # Loop for channel
    counter = 0
    MSE_array_tot = np.full((classes, classes),0.)
    
    for m in range(0,nr_images,classes):
        MSE_array = np.full((classes, classes),0.)
        counter = counter + 1
        print(m)
        print(counter, '\n')
        
        for j in range(classes):
            img_basis = np.array(Image.open(os.path.join(call_folder, str(m+j+1) + '.jpg')))[:,:,i:i+1]
            MSE = np.full((classes,1),0.)
            
            for k in range(classes):
                img_trial = np.array(Image.open(os.path.join(call_folder, str(m+k+1) + '.jpg')))[:,:,i:i+1]
                sqrt_err = np.sum(np.subtract(np.divide(img_basis,255), np.divide(img_trial,255))**2)
                MSE[k,0] = sqrt_err
                
            MSE_array[:, j] = MSE[:,0]
            
        np.add(MSE_array_tot, MSE_array)
        
    np.divide(MSE_array_tot,counter)
        
    if i == 0:
        np.savetxt("statistics_red.csv", MSE_array, delimiter=",")
    elif i == 1:
        np.savetxt("statistics_green.csv", MSE_array, delimiter=",")
    elif i == 2:
        np.savetxt("statistics_blue.csv", MSE_array, delimiter=",")
    else:
        print('ERROR!')