from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.optimizers import SGD,RMSprop,Adam
from keras.applications.vgg16 import VGG16
import  numpy as np
import matplotlib.pyplot as plt
import cv2

model = VGG16(weights = 'imagenet', include_top = True)
sgd = SGD(lr=0.1, decay= 1e-6,momentum=0.9, nesterov=True )

im = cv2.resize(cv2.imread('steam-locomotive.jpg'), (224, 224))
im = np.expand_dims(im, axis=0)

out = model.predict(im)
plt.plot(out.ravel())
plt.show()
print(np.argmax(out))