"""
Code developed by:

Prof. Dr. Sidney Marlon Lopes de Lima
Federal University of Pernambuco
Department of Electronics and Systems
"""

from math import *
from random import *
from time import process_time
from sklearn.model_selection import KFold

import math
import time
import struct
import sys,string
import argparse
import numpy as np
import pandas as pd

#========================================================================
class melm():
	def main(self,TrainingData_File,TestingData_File,AllData_File,Elm_Type,NumberofHiddenNeurons,ActivationFunction,nSeed,kfold,virusNorm,sep,verbose):
				
		if ActivationFunction is None:
			ActivationFunction = 'linear'
		if verbose: print('kernel ' + ActivationFunction)
		
		if nSeed is None:
			nSeed = 1
		else:
			if verbose: print('seed ' + nSeed)
			nSeed = int(nSeed)
		seed(nSeed)
		np.random.seed(nSeed)
		
		Elm_Type = int(Elm_Type)
		if verbose: print ('Macro definition')
		#%%%%%%%%%%% Macro definition
		self.REGRESSION=0
		self.CLASSIFIER=1

		if AllData_File is None:
			if sep:
				sep_character = sep
			else:
				sep_character = " "
				
			if verbose: print ('Loading training dataset')
			#%%%%%%%%%%% Loading training dataset
			train_data = pd.read_csv(TrainingData_File, sep=sep_character, decimal=".", header=None)
			train_data = eliminateNaN_All_data(train_data)
			if verbose: print ('Loading testing dataset')
			#%%%%%%%%%%% Loading testing dataset
			test_data = pd.read_csv(TestingData_File, sep=sep_character, decimal=".", header=None)
			test_data = eliminateNaN_All_data(test_data)
									#%   Release raw testing data array
			TrainingAccuracy, TestingAccuracy, TrainingTime, TestingTime = mElmLearning(train_data,test_data,1,Elm_Type,NumberofHiddenNeurons,ActivationFunction,nSeed,kfold,virusNorm,verbose)
		else:
			#%%%%%%%%%%% Load all dataset
			if verbose: print ('Loading all dataset')
			all_data, samples_index = mElmStruct(AllData_File,Elm_Type,sep,verbose)			
			#---------------------------------------------------------
			if kfold is False:
				peace_train = round(0.9*np.size(samples_index,0))		#% 		90% to train
				train_samples_index = samples_index[0: peace_train]
				test_samples_index = samples_index[peace_train+1: np.size(all_data,0)-1]
				train_data =  all_data[train_samples_index]
				test_data = all_data[test_samples_index]
				del(train_samples_index)
				del(test_samples_index)
				mElmLearning(train_data,test_data,1,Elm_Type,NumberofHiddenNeurons,ActivationFunction,nSeed,kfold,virusNorm,verbose)
			else: 
				#%%%%%%%%%%% k-fold
				if verbose: print ('K-fold validation')
				kf = KFold(n_splits=int(kfold), shuffle=True)
				# Repeat the process k times
				accuracies_train = []
				accuracies_test = []
				timing_train = []
				timing_test = []
				execution = -1
				for train_samples_index, test_samples_index in kf.split(samples_index):
					execution += 1
					train_data =  all_data[train_samples_index]
					test_data = all_data[test_samples_index]
					TrainingAccuracy, TestingAccuracy, TrainingTime, TestingTime = mElmLearning(train_data,test_data,execution, Elm_Type,NumberofHiddenNeurons,ActivationFunction,nSeed,kfold,virusNorm,verbose)
					accuracies_train.append(TrainingAccuracy)
					accuracies_test.append(TestingAccuracy)
					timing_train.append(TrainingTime)
					timing_test.append(TestingTime)
				
	
				# Calculate the mean and standard deviation of training accuracy
				mean_accuracy_train = np.mean(accuracies_train)
				std_accuracy_train = np.std(accuracies_train)
				mean_timing_train = np.mean(timing_train)
				std_timing_train = np.std(timing_train)
				
				# Calculate the mean and standard deviation of test accuracy
				mean_accuracy_test = np.mean(accuracies_test)
				std_accuracy_test = np.std(accuracies_test)
				mean_timing_test = np.mean(timing_test)
				std_timing_test = np.std(timing_test)	
				print('...................K-fold mean results........................')
				if Elm_Type==0:
					print(f"Train RMSE (Root Mean-Square Error): {mean_accuracy_train} ± {std_accuracy_train}")
					print(f"Test RMSE (Root Mean-Square Error): {mean_accuracy_test} ± {std_accuracy_test}")
					print(f"Train Mean Timing: {mean_timing_train:.2f} sec. ± {std_timing_train:.2f} sec.")
					print(f"Test Mean Timing: {mean_timing_test:.2f} sec. ± {std_timing_test:.2f} sec.")
				else:
					print(f"Train Mean Accuracy: {100*mean_accuracy_train:.2f}% ± {std_accuracy_train:.2f}%")
					print(f"Test Mean Accuracy: {100*mean_accuracy_test:.2f}% ± {std_accuracy_test:.2f}%")
					print(f"Train Mean Timing: {mean_timing_train:.2f} sec. ± {std_timing_train:.2f} sec.")
					print(f"Test Mean Timing: {mean_timing_test:.2f} sec. ± {std_timing_test:.2f} sec.")
#========================================================================
def eliminateNaN(vector):

	# Select the first line as a list
	first_row = vector.tolist()

	# Remove elements that are NaN
	first_row = [elem for elem in first_row if not pd.isna(elem)]

	return first_row
#========================================================================
def eliminateNaN_All_data(all_data):
		
	all_data = all_data[:].to_numpy()
	all_data = all_data.astype(float)

	for ii in reversed(range(np.size(all_data,1))):
		if np.all(np.isnan(all_data[:,ii])):
			all_data = np.delete(all_data, ii, axis=1)
	return all_data
#========================================================================
def mElmStruct(AllData_File,Elm_Type,sep,verbose):
				
	if sep:
		sep_character = sep
	else:
		sep_character = ';'
		
	all_data = pd.read_csv(AllData_File, sep=sep_character, decimal=".", low_memory=False, header=None)
	
	training_files = all_data.iloc[:, 0]  # Select the first column
	training_files = training_files.iloc[1:]  # Remove the first element
	training_files = eliminateNaN(training_files)
	
	features = all_data.iloc[0, :]  # Select the first line
	features = features.iloc[2:]  #  Remove the first element
	features = eliminateNaN(features)
	
	all_data = all_data.loc[1:np.size(all_data,0), 1:np.size(all_data,1)]	
	all_data = eliminateNaN_All_data(all_data)
					
	if Elm_Type!=0:
		if verbose: print('Permutation of the order of the input data')
		samples_index = np.random.permutation(np.size(all_data,0))
	else:
		samples_index = range(0, np.size(all_data,0))		#		in time series, order is important
	return all_data, samples_index			
#========================================================================
def loadingDataset(dataset):

	T=np.transpose(dataset[:,0])
	P=np.transpose(dataset[:,1:np.size(dataset,1)])
	del(dataset)
	return T, P      
#========================================================================
def mElmLearning(train_data, test_data,execution,Elm_Type,NumberofHiddenNeurons,ActivationFunction,nSeed,kfold,virusNorm,verbose):

	[T,P] = loadingDataset(train_data)
	[TVT, TVP] = loadingDataset(test_data)
		
	NumberofTrainingData=np.size(P,1)
	NumberofTestingData=np.size(TVP,1)
	NumberofInputNeurons=np.size(P,0)
	NumberofHiddenNeurons = int(NumberofHiddenNeurons)

	if Elm_Type!=0: #classification
		if verbose: print ('Preprocessing the data of classification')
		#%%%%%%%%%%%% Preprocessing the data of classification
		sorted_target=np.sort(np.concatenate((T, TVT), axis=0))
		label = []                #%   Find and save in 'label' class label from training and testing data sets
		label.append(sorted_target[0])
		j=0
		for i in range(1, NumberofTrainingData+NumberofTestingData):
			if sorted_target[i] != label[j]:
				j=j+1
				label.append(sorted_target[i])

		number_class=j+1
		NumberofOutputNeurons=number_class

		if verbose: print ('Processing the targets of training')
		#%%%%%%%%%% Processing the targets of training
		temp_T=np.zeros((NumberofOutputNeurons, NumberofTrainingData))

		for i in range(0, NumberofTrainingData):
			for j in range(0, number_class):
				if label[j] == T[i]:
					break
			temp_T[j][i]=1
		T=temp_T*2-1

		if verbose: print ('Processing the targets of testing')
		#%%%%%%%%%% Processing the targets of testing
		temp_TV_T=np.zeros((NumberofOutputNeurons, NumberofTestingData))
		for i in range(0, NumberofTestingData):
			for j in range(0, number_class):
				if label[j] == TVT[i]:
					break
			temp_TV_T[j][i]=1
		TVT=temp_TV_T*2-1

	if verbose: print ('Calculate weights & biases')
	#%%%%%%%%%%% Calculate weights & biases
	start_time_train = process_time()

	if verbose: print ('Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons')
	#%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
	
	if Elm_Type == 0: #Regression
		if (ActivationFunction == 'erosion') or (ActivationFunction == 'ero')  or (ActivationFunction == 'dilation') or (ActivationFunction == 'dil') or (ActivationFunction == 'fuzzy-erosion') or (ActivationFunction == 'fuzzy_erosion') or (ActivationFunction == 'fuzzy-dilation') or (ActivationFunction == 'fuzzy_dilation') or (ActivationFunction == 'bitwise-erosion') or (ActivationFunction == 'bitwise_erosion') or (ActivationFunction == 'bitwise-dilation') or (ActivationFunction == 'bitwise_dilation') :
			InputWeight = np.random.uniform(np.amin(np.amin(P)), np.amax(np.amax(P)), (NumberofHiddenNeurons,NumberofInputNeurons))
		else:
			InputWeight=np.random.rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
	else:
		InputWeight=np.random.rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
	
	if virusNorm: 
		InputWeight = virusNormFunction(InputWeight,verbose);
		
	BiasofHiddenNeurons=np.random.rand(NumberofHiddenNeurons,1);
	if verbose: print ('Calculate hidden neuron output matrix H')
	#%%%%%%%%%%% Calculate hidden neuron output matrix H
	H = switchActivationFunction(ActivationFunction,InputWeight,BiasofHiddenNeurons,P)
	
	if verbose: print ('Calculate output weights OutputWeight (beta_i)')
	#%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)
	OutputWeight = np.dot(np.linalg.pinv(np.transpose(H)), np.transpose(T))
	
	end_time_train =  process_time()
	TrainingTime=end_time_train-start_time_train       # %   Calculate CPU time (seconds) spent for training ELM

	if verbose: print ('Calculate the training accuracy')
	#%%%%%%%%%%% Calculate the training accuracy
	Y = np.transpose(np.dot(np.transpose(H), OutputWeight))                     #%   Y: the actual output of the training data

	TrainingAccuracy = 0
	if Elm_Type == 0:        
		if verbose: print ('Calculate training accuracy (RMSE) for regression case')  
		#   Calculate training accuracy (RMSE) for regression case
		TrainingAccuracy = np.square(np.subtract(T, Y)).mean()
		TrainingAccuracy = round(TrainingAccuracy, 6) 
	del(H)
	
	if verbose: print ('Calculate the output of testing input')  
	start_time_test = process_time()
	#%%%%%%%%%%% Calculate the output of testing input
	tempH_test = switchActivationFunction(ActivationFunction,InputWeight,BiasofHiddenNeurons,TVP)
	del(TVP)

	TY = np.transpose(np.dot(np.transpose(tempH_test), OutputWeight))                     #%   Y: the actual output of the training data

	end_time_test = process_time()
	TestingTime=end_time_test-start_time_test           #%   Calculate CPU time (seconds) spent by ELM predicting the whole testing data

	TestingAccuracy = 0
	if Elm_Type == 0:          
		if verbose: print ('Calculate testing accuracy (RMSE) for regression case')  
		#   Calculate testing accuracy (RMSE) for regression case
		TestingAccuracy = np.square(np.subtract(TVT, TY)).mean()
		TestingAccuracy = round(TestingAccuracy, 6) 
		if kfold: print('..................k: '+ str(execution) +', k-fold: ' + str(kfold) + '............................')
		else: print('....................................................................')
		print('Training RMSE (Root Mean-Square Error): ' + str(TrainingAccuracy)+' ( ' + str(np.size(Y,0)) + ' samples) (regression)')
		print('Testing RMSE (Root Mean-Square Error): ' + str(TestingAccuracy)+' ( ' + str(np.size(TY,0)) + ' samples) (regression)')
		print('Training Time: ' + str(round(TrainingTime,2)) + ' sec.')
		print('Testing Time: ' + str(round(TestingTime,2)) + ' sec.')
	else:
		if verbose: print ('Calculate training & testing classification accuracy')  
		#%%%%%%%%%% Calculate training & testing classification accuracy
		MissClassificationRate_Training=0
		MissClassificationRate_Testing=0

		label_index_train_expected = np.argmax(T, axis=0)   # Maxima along the second axis
		label_index_train_actual = np.argmax(Y, axis=0)   # Maxima along the second axis
					
		for i in range(0, np.size(label_index_train_expected,0)):
			if label_index_train_actual[i]!=label_index_train_expected[i]:
				MissClassificationRate_Training=MissClassificationRate_Training+1

		TrainingAccuracy=1-MissClassificationRate_Training/np.size(label_index_train_expected,0)
		TrainingAccuracy = round(TrainingAccuracy, 6) 

		
		label_index_expected = np.argmax(TVT, axis=0)   # Maxima along the second axis
		label_index_actual = np.argmax(TY, axis=0)   # Maxima along the second axis

		for i in range(0, np.size(label_index_expected,0)):
			if label_index_actual[i]!=label_index_expected[i]:
				MissClassificationRate_Testing=MissClassificationRate_Testing+1

		TestingAccuracy=1-MissClassificationRate_Testing/np.size(label_index_expected,0)
		
		TestingAccuracy = round(TestingAccuracy, 6) 
		if kfold: print('..................k: '+ str(execution) +', k-fold: ' + str(kfold) + '............................')
		else: print('....................................................................')
		print('Training Accuracy: ' + str(round(TrainingAccuracy*100,2))+' % (',str(np.size(label_index_train_expected,0)-MissClassificationRate_Training),'/',str(np.size(label_index_train_expected,0)),') (classification)')
		print('Testing Accuracy: ' + str(round(TestingAccuracy*100,2))+' % (',str(np.size(label_index_expected,0)-MissClassificationRate_Testing),'/',str(np.size(label_index_expected,0)),') (classification)')
		
		print('Training Time: ' + str(round(TrainingTime,2)) + ' sec.')
		print('Testing Time: ' + str(round(TestingTime,2)) + ' sec.')
	return TrainingAccuracy, TestingAccuracy, TrainingTime, TestingTime
#========================================================================
def virusNormFunction(matrix, verbose):
	#Normalizes the weights of the matrix to the interval [rb, ra].
	if verbose: print('virusNorm is a normalization technique. It uses malware samples from the VirusShare database as a reference.')
	vector = matrix.flatten()
	maxi = np.max(vector)
	mini = np.min(vector)

	ra = 0.9
	rb = 0.1

	R = (((ra - rb) * (matrix - mini)) / (maxi - mini)) + rb
	return R

#========================================================================
def switchActivationFunction(ActivationFunction,InputWeight,BiasofHiddenNeurons,P):
	
	if (ActivationFunction == 'sig') or (ActivationFunction == 'sigmoid'):
		H = sig_kernel(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'sin') or (ActivationFunction == 'sine'):
		H = sin_kernel(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'hardlim'):
		H = hardlim_kernel(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'tribas'): 
		H = tribas_kernel(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'radbas'):
		H = radbas_kernel(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'erosion') or (ActivationFunction == 'ero'):
		H = erosion(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'dilation') or (ActivationFunction == 'dil'):
		H = dilation(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'fuzzy-erosion') or (ActivationFunction == 'fuzzy_erosion'):
		H = fuzzy_erosion(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'fuzzy-dilation') or (ActivationFunction == 'fuzzy_dilation'):
		H = fuzzy_dilation(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'bitwise-erosion') or (ActivationFunction == 'bitwise_erosion'):
		H = bitwise_erosion(InputWeight, BiasofHiddenNeurons, P)
	elif (ActivationFunction == 'bitwise-dilation') or (ActivationFunction == 'bitwise_dilation'):
		H = bitwise_dilation(InputWeight, BiasofHiddenNeurons, P)
	else: #'linear'
		H = linear_kernel(InputWeight, BiasofHiddenNeurons, P)
	return H
#========================================================================
def sig_kernel(w1, b1, samples):
	#%%%%%%%% Sigmoid
	tempH = np.dot(w1, samples) + b1
	H = 1 / (1 + np.exp(-tempH))
	return H
#========================================================================
def sin_kernel(w1, b1, samples):
	#%%%%%%%% Sine
	tempH = np.dot(w1, samples) + b1
	H = np.sin(tempH)   
	return H
#========================================================================
def hardlim_kernel(w1, b1, samples):
	#%%%%%%%% Hard Limit
	#hardlim(n)	= 1 if n ≥ 0
	#		= 0 otherwise
	tempH = np.dot(w1, samples) + b1
	H = tempH
	for ii in range(np.size(tempH,0)):
		for jj in range(np.size(tempH,1)):
			if tempH[ii][jj] >= 0:
				H[ii][jj] = 1
			else:
				H[ii][jj] = 0
	return H
#========================================================================
def tribas_kernel(w1, b1, samples):
	#%%%%%%%% Triangular basis function
	#a = tribas(n) 	= 1 - abs(n), if -1 <= n <= 1
		#	 	= 0, otherwise
	tempH = np.dot(w1, samples) + b1
	H = tempH
	for ii in range(np.size(tempH,0)):
		for jj in range(np.size(tempH,1)):
			if (tempH[ii][jj] >= -1) and (tempH[ii][jj] <= 1):
				H[ii][jj] = 1 - abs(tempH[ii][jj])
			else:
				H[ii][jj] = 0
	return H
#========================================================================
def radbas_kernel(w1, b1, samples):
	#%%%%%%%% Radial basis function
	#radbas(n) = exp(-n^2)
	tempH = np.dot(w1, samples) + b1
	H = np.exp(-np.power(tempH, 2))
	return H
#========================================================================
def linear_kernel(w1, b1, samples):
	H = np.dot(w1, samples) + b1
	return H
#========================================================================
def erosion(w1, b1, samples):

	H = np.zeros((np.size(w1,0), np.size(samples,1)))
	x = np.zeros(np.size(w1,1))

	for s_index in range(np.size(samples,1)):
		ss = samples[:,s_index]
		for i in range(np.size(w1,0)):
			for j in range(np.size(w1,1)):
				x[j] = max(ss[j], 1-w1[i][j])
			H[i][s_index] = min(x)+b1[i][0]

	return H
#========================================================================
def dilation(w1, b1, samples):

	H = np.zeros((np.size(w1,0), np.size(samples,1)))
	x = np.zeros(np.size(w1,1))
	for s_index in range(np.size(samples,1)):
		ss = samples[:,s_index]
		for i in range(np.size(w1,0)):
			for j in range(np.size(w1,1)):
				x[j] = min(ss[j], w1[i][j])
			H[i][s_index] = max(x)+b1[i][0]

	return H
#========================================================================
def fuzzy_erosion(w1, b1, samples):
	
	tempH = np.dot(w1, samples) + b1
	H = np.ones((np.size(w1,0), np.size(samples,1)))
	
	for s_index in range(np.size(samples,1)):
		ss = samples[:,s_index]
		for i in range(np.size(w1,0)):
			H[i][s_index] = 1- tempH[i][s_index] 
	return H
#========================================================================
def fuzzy_dilation(w1, b1, samples):

	tempH = np.dot(w1, samples) + b1
	H = np.ones((np.size(w1,0), np.size(samples,1)))
	
	for s_index in range(np.size(samples,1)):
		ss = samples[:,s_index]
		for i in range(np.size(w1,0)):
			for j in range(np.size(w1,1)):
				H[i][s_index] = H[i][s_index] * (1 - tempH[i][j])

	H = 1 - H
	return H
#========================================================================
def bitwise_erosion(w1, b1, samples):

	H = np.zeros((np.size(w1,0), np.size(samples,1)))
	x = np.zeros(np.size(w1,1),dtype=bytearray)

	for s_index in range(np.size(samples,1)):
		ss = samples[:,s_index]
		for i in range(np.size(w1,0)):
			for j in range(np.size(w1,1)):
				x[j] = bytes_or(ss[j], 1-w1[i][j])
			result = x[0]
			for j in range(1, np.size(w1,1)):
				result = bytes_and(result, x[j])
			temp = struct.unpack('d', result)[0]
			if math.isnan(temp): temp = 0.0 
			H[i][s_index] = temp + b1[i][0]
	return H
#========================================================================
def bitwise_dilation(w1, b1, samples):

	H = np.zeros((np.size(w1,0), np.size(samples,1)))
	x = np.zeros(np.size(w1,1),dtype=bytearray)

	for s_index in range(np.size(samples,1)):
		ss = samples[:,s_index]
		for i in range(np.size(w1,0)):
			for j in range(np.size(w1,1)):
				x[j] = bytes_and(ss[j], w1[i][j])
			result = x[0]
			for j in range(1, np.size(w1,1)):
				result = bytes_or(result, x[j])
			temp = struct.unpack('d', result)[0]
			if math.isnan(temp): temp = 0.0 
			H[i][s_index] = temp + b1[i][0]
	return H

#========================================================================	
def bytes_and(a, b) :
	a1 = bytearray(a)
	b1 = bytearray(b)
	c = bytearray(len(a1))
	
	for i in range(len(a1)):
		c[i] = a1[i] & b1[i]
	return c
	
#========================================================================	
def bytes_or(a, b) :
	a1 = bytearray(a)
	b1 = bytearray(b)
	c = bytearray(len(a1))
	
	for i in range(len(a1)):
		c[i] = a1[i] | b1[i]
	return c
	
#========================================================================
def setOpts(argv):                         
	parser = argparse.ArgumentParser()
	parser.add_argument('-tr', '--TrainingData_File',dest='TrainingData_File',action='store', 
		help="Filename of training data set")
	parser.add_argument('-ts', '--TestingData_File',dest='TestingData_File',action='store',
		help="Filename of testing data set")
	parser.add_argument('-tall', '--AllData_File',dest='AllData_File',action='store',
		help="Filename of all data set, including training and testing")
	parser.add_argument('-ty', '--Elm_Type',dest='Elm_Type',action='store',required=True,
		help="0 for regression; 1 for (both binary and multi-classes) classification")
	parser.add_argument('-nh', '--nHiddenNeurons',dest='nHiddenNeurons',action='store',required=True,
		help="Number of hidden neurons assigned to the ELM")
	parser.add_argument('-af', '--ActivationFunction',dest='ActivationFunction',action='store', 
		help="Type of activation function:")
	parser.add_argument('-sd', '--seed',dest='nSeed',action='store', 
		help="random number generator seed:")
	parser.add_argument('-kfold', dest='kfold', action='store',default=False,
		help="K-fold validation. It must be an integer value.")
	parser.add_argument('-virusNorm', dest='virusNorm', action='store_true',default=False,
		help="Normalization according to the range of VirusShare sample attributes.")  
	parser.add_argument('-sep', dest='sep', action='store_true',default=False,
		help="Character or regex pattern to treat as the delimiter.Default; space for TrainingData_File and TestingData_Filea and ; character for AllData_File")          
	parser.add_argument('-v', dest='verbose', action='store_true',default=False,
		help="Verbose output")
	arg = parser.parse_args()
	return(arg.__dict__['TrainingData_File'], arg.__dict__['TestingData_File'], arg.__dict__['AllData_File'], arg.__dict__['Elm_Type'], arg.__dict__['nHiddenNeurons'], 		
		arg.__dict__['ActivationFunction'], arg.__dict__['nSeed'], arg.__dict__['kfold'], arg.__dict__['virusNorm'], arg.__dict__['sep'], arg.__dict__['verbose'])	
#========================================================================
if __name__ == "__main__":
	opts = setOpts(sys.argv[1:])
	ff = melm()
	ff.main(opts[0], opts[1], opts[2], opts[3], opts[4], opts[5], opts[6], opts[7], opts[8], opts[9], opts[10])
#========================================================================
