import numpy as np
from models import interpolation , SRCNN_train , SRCNN_predict , DNCNN_train
from scipy.io import loadmat

if __name__ == "__main__":
	# Load datasets
	real = loadmat("Perfect_"+ ""VehA".mat")["My_perfect_H"]
	noisy_input = loadmat("Noisy_" + "VehA" + "_" + "SNR_" + "22" + ".mat") ["VehA"+"_noisy_"+ str(SNR)]

	interp_noisy = interpolation(noisy_input , SNR , 48 , 'rbf')

	real_image = np.zeros((len(real),72,14,2))
	real_image[:,:,:,0] = np.real(real)
	real_image[:,:,:,1] = np.imag(real)
	real_image = np.concatenate((real_image[:,:,:,0], real_image[:,:,:,1]), axis=0).reshape(2*len(real), 72, 14, 1)

	# Train SRCNN
	idx_random = np.random.rand(len(real_image)) < (1/9)  # uses 32000 from 36000 as training and the rest as validation
	val_data, val_label = interp_noisy[~idx_random,:,:,:] , real_image[~idx_random,:,:,:]
	train_data, train_label = interp_noisy[idx_random,:,:,:] , real_image[idx_random,:,:,:]
	SRCNN_train(train_data ,train_label, val_data , val_label , "VehA" , 48 , SNR )

	# Predict
	srcnn_pred_train = SRCNN_predict(train_data, "VehA" , num_pilots , SNR)
	srcnn_pred_validation = SRCNN_predict(train_data, "VehA" , num_pilots , SNR)

	# Train DNCNN
	DNCNN_train(input_data, "VehA" , num_pilots , SNR)