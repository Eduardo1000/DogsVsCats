# Módulo para carregar o Dataset

import numpy as np
import cv2
from os import listdir
from tqdm import tqdm

# Função para carregar o dataset
def load_X(path, width, height, channel, slice=None):
	images_list = listdir( path )
	if slice != None:
		images_list = images_list[:slice]
	tamanho = len(images_list)
	(n, w, h, c) = (tamanho, width, height, channel) # Tensor de numero de imagens, tamanho_x, tamanho_y, canais
	X = np.zeros( (n, w, h, c), dtype = np.uint8)
	Y = np.zeros((n), dtype = np.float32)
	for i, image in enumerate(tqdm(images_list)) :
		if image[:3] == 'dog': Y[i] = 0
		else: Y[i] = 1
		image_path = path + '/' + image
		# Load an color image
		image = cv2.imread( image_path )
		image = cv2.resize( image, (w, h) )

		X[i] = image
	return X, Y

# Módulo para visualizar as imagens caso necessário
def view_X(X, Y, label):
	size = X.shape[0]
	for i in range(size):
		# print(X[i].shape, Y[i])
		# Dislay Image
		cv2.imshow( label, X[i])
		cv2.waitKey( 0 )

# Divide o Dataset em Treino, Validação e Teste
def split_dataset(X,Y, train=None, validation=None, test=None):
	size = X.shape[0]
	if train == None or validation == None or test == None:
		# 70% train 10% test 20% validation DEFAULT
		train = 0.7; validation = 0.2; test = 0.1
	else:
		if type(train) != float or type(validation) != float or type(test) != float:
			return print('Not valid type. Train, validation and test must be float')
		sum = train+validation+test
		sum = float( '%g' % (sum) )
		if sum != 1.0:
			return print('Train, test e and validation must sum 1')

	size_train = int(train*size)
	size_validation = int( float( '%g' % ( train+validation ) ) *size)
	Xtrain = X[:size_train]
	Ytrain = Y[:size_train]
	Xvalidation = X[ size_train : size_validation ]
	Yvalidation = Y[ size_train : size_validation ]
	Xtest = X[ size_validation : ]
	Ytest = Y[ size_validation : ]
	return Xtrain, Ytrain, Xvalidation, Yvalidation, Xtest, Ytest

# Divide o Dataset em batches, que são conjuntos menores do dado
# Para alimentar a rede geralmente é utilizado conjunto menores
def split_tensor(X,Y,n):
	tensor_list_X = []
	tensor_list_Y = []
	size = Y.shape[0]
	division = int(size/n)

	for i in range(division):
		tensor_list_X.append( X[i*n: (i+1)*n] )
		tensor_list_Y.append( Y[i*n: (i+1)*n] )
	if size % n > 0 :
		tensor_list_X.append( X[division * n : size] )
		tensor_list_Y.append( Y[division * n : size] )
	return tensor_list_X, tensor_list_Y