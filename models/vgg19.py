# VGG é uma arquitetura de Rede Neural muito conhecida
# A Rede VGG16 foi campeã da competição ILSRV de Imagens
# A VGG19 é uma evolução da mesma
# As redes VGG são muito utilizada por serem de fácil implementação
# São basicamento uma sequencia de convuluções e pooling

import tensorflow as tf

def build_vgg19(input_width, input_height, input_channels, n_classes):
	# Por praticidade os tensores são salvos neste dicionário
	Dic = { }

	# ---- INICIO DA REDE NEURAL ----
	# placeholders
	placeholder_X = tf.placeholder( tf.float32, shape = (None, input_width, input_height, input_channels) )
	Dic["placeholder_X"] = placeholder_X

	placeholder_Y = tf.placeholder( tf.int64, shape = (None) )
	Dic["placeholder_Y"] = placeholder_Y

	initializer = tf.contrib.layers.xavier_initializer( seed = 0 )

	# --------------------------- BLOCO 1 -------------------------------
	# camada convolucao 1_1
	conv1_1 = tf.layers.conv2d( inputs = placeholder_X, filters = 64, kernel_size = [3, 3], strides = 1,
	                             activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv1_1"] = conv1_1
	print( "conv1_1", conv1_1.get_shape( ) )

	# camada convolucao 1_2
	conv1_2 = tf.layers.conv2d( inputs = conv1_1, filters = 64, kernel_size = [3, 3], strides = 1,
	                             activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv1_2"] = conv1_2
	print( "conv1_2", conv1_2.get_shape( ) )

	# camada max pooling 1

	maxpool_1 = tf.layers.max_pooling2d( inputs = conv1_2, pool_size = [2, 2], strides = 2, padding = 'SAME' )
	Dic["maxpool_1"] = maxpool_1
	print( "max_pool_1", maxpool_1.get_shape( ) )

	# --------------------------- BLOCO 2 -------------------------------
	# camada convolucao 2_1
	conv2_1 = tf.layers.conv2d( inputs = maxpool_1, filters = 128, kernel_size = [3, 3], strides = 1,
	                             activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv2_1"] = conv2_1
	print( "conv2_1", conv2_1.get_shape( ) )

	# TODO camada convolucao 2_2
	conv2_2 = tf.layers.conv2d( inputs = conv2_1, filters = 128, kernel_size = [3, 3], strides = 1,
	                             activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv2_2"] = conv2_2
	print( "conv2_2", conv2_2.get_shape( ) )

	# camada pooling 2
	maxpool_2 = tf.layers.max_pooling2d( inputs = conv2_2, pool_size = [2, 2], strides = 2, padding = 'SAME' )
	Dic["maxpool_2"] = maxpool_2
	print( "maxpool_2", maxpool_2.get_shape( ) )

	# --------------------------- BLOCO 3 -------------------------------
	# camada convolucao 3_1
	conv3_1 = tf.layers.conv2d( inputs = maxpool_2, filters = 256, kernel_size = [3, 3], strides = 1,
	                             activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv3_1"] = conv3_1
	print( "conv3_1", conv3_1.get_shape( ) )

	# TODO camada convolucao 3_2
	conv3_2 = tf.layers.conv2d( inputs = conv3_1, filters = 256, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv3_2"] = conv3_2
	print( "conv3_2", conv3_2.get_shape( ) )

	# TODO camada convolucao 3_3
	conv3_3 = tf.layers.conv2d( inputs = conv3_2, filters = 256, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv3_3"] = conv3_3
	print( "conv3_3", conv3_3.get_shape( ) )

	# TODO camada convolucao 3_4
	conv3_4 = tf.layers.conv2d( inputs = conv3_3, filters = 256, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv3_4"] = conv3_4
	print( "conv3_4", conv3_4.get_shape( ) )

	# camada pooling 3
	maxpool_3 = tf.layers.max_pooling2d( inputs = conv3_4, pool_size = [2, 2], strides = 2, padding = 'SAME' )
	Dic["maxpool_3"] = maxpool_3
	print( "maxpool_3", maxpool_3.get_shape( ) )

	# --------------------------- BLOCO 4 -------------------------------
	# TODO camada convolucao 4_1
	conv4_1 = tf.layers.conv2d( inputs = maxpool_3, filters = 512, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv4_1"] = conv4_1
	print( "conv4_1", conv4_1.get_shape( ) )

	# TODO camada convolucao 4_2
	conv4_2 = tf.layers.conv2d( inputs = conv4_1, filters = 512, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv4_2"] = conv4_2
	print( "conv4_2", conv4_2.get_shape( ) )

	# TODO camada convolucao 4_3
	conv4_3 = tf.layers.conv2d( inputs = conv4_2, filters = 512, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv4_3"] = conv4_3
	print( "conv4_3", conv4_3.get_shape( ) )

	# TODO camada convolucao 4_4
	conv4_4 = tf.layers.conv2d( inputs = conv4_3, filters = 512, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv4_4"] = conv4_4
	print( "conv4_4", conv4_4.get_shape( ) )

	# TODO camada pooling 4
	maxpool_4 = tf.layers.max_pooling2d( inputs = conv4_4, pool_size = [2, 2], strides = 2, padding = 'SAME' )
	Dic["maxpool_4"] = maxpool_4
	print( "maxpool_4", maxpool_4.get_shape( ) )

	# --------------------------- BLOCO 5 -------------------------------
	# TODO camada convolucao 5_1
	conv5_1 = tf.layers.conv2d( inputs = maxpool_4, filters = 512, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv5_1"] = conv5_1
	print( "conv5_1", conv5_1.get_shape( ) )

	# TODO camada convolucao 5_2
	conv5_2 = tf.layers.conv2d( inputs = conv5_1, filters = 512, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv5_2"] = conv5_2
	print( "conv5_2", conv5_2.get_shape( ) )

	# TODO camada convolucao 5_3
	conv5_3 = tf.layers.conv2d( inputs = conv5_2, filters = 512, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv5_3"] = conv5_3
	print( "conv5_3", conv5_3.get_shape( ) )

	# TODO camada convolucao 5_4
	conv5_4 = tf.layers.conv2d( inputs = conv5_3, filters = 512, kernel_size = [3, 3], strides = 1,
	                            activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv5_4"] = conv5_4
	print( "conv5_4", conv5_4.get_shape( ) )

	# TODO camada pooling 5
	maxpool_5 = tf.layers.max_pooling2d( inputs = conv5_4, pool_size = [2, 2], strides = 2, padding = 'SAME' )
	Dic["maxpool_5"] = maxpool_5
	print( "maxpool_5", maxpool_5.get_shape( ) )

	# flatten
	flatten = tf.contrib.layers.flatten( maxpool_5 )

	# fc1
	fc1 = tf.contrib.layers.fully_connected( flatten, num_outputs = 4096, activation_fn = tf.nn.relu )

	# fc2
	fc2 = tf.contrib.layers.fully_connected( fc1, num_outputs = 4096, activation_fn = tf.nn.relu )

	# output (fully_connected)
	out = tf.contrib.layers.fully_connected( fc2, num_outputs = n_classes, activation_fn = None )

	# adaptando o Label Y para o modelo One-Hot Label
	one_hot = tf.one_hot( placeholder_Y, depth = n_classes )

	# Função de perda/custo/erro
	loss = tf.losses.softmax_cross_entropy( onehot_labels = one_hot, logits = out )
	Dic["loss"] = loss

	# Otimizador
	opt = tf.train.AdamOptimizer( learning_rate = 0.003 ).minimize( loss )
	Dic["opt"] = opt

	# Softmax
	softmax = tf.nn.softmax( out )
	Dic["softmax"] = softmax

	# Classe
	class_ = tf.argmax( softmax, 1 )
	Dic["class"] = class_

	# Acurácia
	compare_prediction = tf.equal( class_, placeholder_Y )
	accuracy = tf.reduce_mean( tf.cast( compare_prediction, tf.float32 ) )
	Dic["accuracy"] = accuracy

	return Dic