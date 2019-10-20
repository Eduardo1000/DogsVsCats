import numpy as np
from utils.dataset_loader import load_X
import tensorflow as tf
import cv2

# Método para compara as predições com as marcações
if __name__ == '__main__':
	# Path to images
	(train_path, test_path) = ('DogsVsCats/all/train', 'DogsVsCats/all/test1')
	# Resize image parameter
	(width, height, channel) = (64,64, 3)
	# Load images
	dataset, Y = load_X(train_path,  width, height, slice=2)
	print('Train Shape = ', dataset.shape)

	# Cria o Filto de Imagem Vertical e horizontal
	filters = np.zeros(shape = (7,7,channel,2), dtype = np.float32)
	filters[:, 3, :, 0] = 1
	filters[3, :, :, 1] = 1

	# Realiza a Convolução
	X = tf.placeholder(tf.float32, shape=(None, height, width, channel))
	convolution = tf.nn.conv2d( X, filters, strides=[1,2,2,1], padding="SAME")

	with tf.Session() as sess:
		output = sess.run(convolution, feed_dict={X : dataset})

	for image in dataset:
		cv2.imshow( 'original  ', image )
		cv2.waitKey( 0 )
		# cv2.imshow( 'image', output[0,:,:,1] )
