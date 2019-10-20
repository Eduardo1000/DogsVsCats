# Loop through every image in train database
# for image in listdir( test_path ):
# 	image_path = test_path + '/' + image
# 	print(image_path)
# Load an color image

# # Add Border
# bordersize = 10
# GREEN = [0, 255, 0]
# RED = [0,0,255]
# image = cv2.copyMakeBorder( image, top = bordersize, bottom = bordersize, left = bordersize, right = bordersize,
#                              borderType = cv2.BORDER_CONSTANT, value = GREEN )

# Resize this image
# image = cv2.resize( image, (rx, ry) )
#
# # Dislay Image
# cv2.imshow('image',image)
#
# key = cv2.waitKey(300)
# if key == 27:
# 	cv2.destroyAllWindows()
# 	break


import models.vgg as vgg
vgg.vgg_16(inputs,
           num_classes=2,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False)