import tensorflow as tf

def build_cnn(input_width, input_height, input_channels, n_classes) :
	# Por praticidade os tensores são salvos neste dicionário
	Dic = { }

	# ---- INICIO DA REDE NEURAL ----
	# placeholders
	placeholder_X = tf.placeholder( tf.float32, shape = (None, input_width, input_height, input_channels) )
	Dic["placeholder_X"] = placeholder_X

	placeholder_Y = tf.placeholder( tf.int64, shape = (None) )
	Dic["placeholder_Y"] = placeholder_Y

	initializer = tf.contrib.layers.xavier_initializer( seed = 0 )

	# camada convolucao 1
	conv2d_1 = tf.layers.conv2d( inputs = placeholder_X, filters = 4, kernel_size = [3, 3], strides = 1,
	                             activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv2d_1"] = conv2d_1
	print( "conv2d_1", conv2d_1.get_shape( ) )

	# camada convolucao 2
	conv2d_2 = tf.layers.conv2d( inputs = conv2d_1, filters = 4, kernel_size = [3, 3], strides = 1,
	                             activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv2d_2"] = conv2d_2
	print( "conv2d_2", conv2d_2.get_shape( ) )

	# camada max pooling 1
	maxpool_1 = tf.layers.max_pooling2d( inputs = conv2d_2, pool_size = [8, 8], strides = 2, padding = 'SAME' )
	Dic["maxpool_1"] = maxpool_1
	print( "max_pool2d_1", maxpool_1.get_shape( ) )

	# camada convolucao 3
	conv2d_3 = tf.layers.conv2d( inputs = maxpool_1, filters = 8, kernel_size = [3, 3], strides = 1,
	                             activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv2d_3"] = conv2d_3
	print( "conv2d_3", conv2d_3.get_shape( ) )

	# camada pooling 2
	maxpool_2 = tf.layers.max_pooling2d( inputs = conv2d_3, pool_size = [8, 8], strides = 2, padding = 'SAME' )
	Dic["maxpool_2"] = maxpool_2
	print( "max_pool2d_2", maxpool_2.get_shape( ) )

	# camada convolucao 4
	conv2d_4 = tf.layers.conv2d( inputs = maxpool_2, filters = 16, kernel_size = [3, 3], strides = 1,
	                             activation = tf.nn.relu, padding = 'SAME', kernel_initializer = initializer )
	Dic["conv2d_4"] = conv2d_3
	print( "conv2d_4", conv2d_3.get_shape( ) )

	# camada pooling 3
	maxpool_3 = tf.layers.max_pooling2d( inputs = conv2d_4, pool_size = [8, 8], strides = 2, padding = 'SAME' )
	Dic["maxpool_3"] = maxpool_3
	print( "max_pool2d_3", maxpool_3.get_shape( ) )

	# flatten
	flatten = tf.contrib.layers.flatten( maxpool_3 )

	# output (fully_connected)
	out = tf.contrib.layers.fully_connected( flatten, num_outputs = n_classes, activation_fn = None )

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
