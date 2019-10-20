import tensorflow as tf
from models.vgg19 import build_vgg19
from utils.dataset_loader import load_X, split_dataset, split_tensor

if __name__ == '__main__':
	# Path to train and test folders
	# in this example we are just using train - Supervised training
	(train_path, test_path) = ('all/train', 'all/test1')
	# Size of the resized images
	(width, height, channel) = (64, 64, 3)
	# Load images limited to slice quantity - Ignore slice to load all images  in folder
	X, Y = load_X( train_path, width, height, channel, slice = 500 )
	print( 'dataset shape', X.shape )

	# Split dataset into
	X_train, Y_train, X_validation, Y_validation, X_test, Y_test = split_dataset(X,Y, train=0.7, validation=0.2, test=0.1 )
	print( 'train shape', X_train.shape, Y_train.shape )
	print( 'validation shape', X_validation.shape, Y_validation.shape )
	print( 'test shape', X_test.shape, Y_test.shape )

	# create minibatches of size n
	n = 16
	X_tensor, Y_tensor = split_tensor( X_train, Y_train, n = n)
	print( 'tensor shape %s n = %s' % (len(X_tensor), n) )

	# Iniciando
	sess = tf.InteractiveSession( )

	# construindo o modelo de rede - 2 means 0 and 1 for dogs and cats
	# Rede com o modelo Inicial
	# Dic_cg = build_cnn( width, height, channel, 2 )
	# Rede com a VGG19
	Dic_cg = build_vgg19( width, height, channel, 2 )

	# inicializando as variveis do tensorflow
	sess.run( tf.global_variables_initializer( ) )


	# definindo o número de épocas
	epochs = 100
	seed = 0
	best_acc = 0
	saver = tf.train.Saver( )
	models = []
	loss_list = []

	for i in range( epochs ) :
		# treinamento com os batch
		for j, minibatch in enumerate(X_tensor) :
			_ , erro_treino = sess.run( [Dic_cg["opt"], Dic_cg["loss"]],
			          feed_dict = { Dic_cg["placeholder_X"] : X_tensor[j], Dic_cg["placeholder_Y"] :  Y_tensor[j] } )

		# a cada 10 épocas o erro é impresso
		if i % 10 == 0 :
			erro_validation, acc_validation = sess.run( [Dic_cg["loss"],Dic_cg["accuracy"]], feed_dict = { Dic_cg["placeholder_X"] : X_validation,
			                                                     Dic_cg["placeholder_Y"] : Y_validation } )
			models.append( [i, erro_validation, acc_validation] )
			loss_list.append(erro_treino)
			if acc_validation > best_acc:
				best_acc = acc_validation
				save_path = saver.save( sess, "/tmp/model.ckpt" )
				print( "Model saved in path: %s" % save_path, models )

			print( "O erro na época", i, "é", erro_validation, ' e a acurácia é ', acc_validation )

	print('Erros e acurácia das épocas ',models)
	print('Erros das épocas ',loss_list)

	# calculando a acurácia
	acc = sess.run( Dic_cg["accuracy"],
	                feed_dict = { Dic_cg["placeholder_X"] : X_test, Dic_cg["placeholder_Y"] : Y_test } )
	print( "A accurácia é:", acc )

	#USANDO A REDE PARA REALIZAR PREDIÇÕES
	index = 10
	index_cut = index+1

	probs = sess.run(Dic_cg["softmax"], feed_dict={Dic_cg["placeholder_X"]: X_test[index:index_cut]})
	print("Softmax da imagem é:", probs)

	# view_X(X_tensor[0], Y_tensor[0], label='train')
	# view_X(Xtest, Ytest, label = 'test')


