import numpy as np
import scipy.misc
import io
from keras.models import  model_from_json
from keras.optimizers import SGD
from PIL import Image
import tensorflow as tf
import matplotlib.image as mping
#load data
model_architecture = 'cifar_model.json'
model_weights = 'cifar_weights'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

# img_names = ['cat.jpg', 'dog.jpg']
# imgs = [np.transpose(scipy.misc.imresize(scipy.misc.imread(img_name), (32,32)), (1, 0, 2)).astype('float32')
#         for img_name in img_names]
# imgs = np.array(imgs)/ 255
# img = tf.gfile.FastGFile('cat.jpeg', 'rb')
# encode_jpg = img.read()
# encode_jpg_io = io.BytesIO(encode_jpg)
# try:
#         image = Image.open(encode_jpg_io)
#         heigth, width = image.size
# except(OSError, NameError):
#         print('Oserror')
# print('done')
imgs = mping.imread('cat.jpg')
imgs.resize((32,32))

optim = SGD()
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])

predictions = model.predict_classes(imgs)
print(predictions)