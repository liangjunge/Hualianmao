from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

#load model
base_mode = VGG16(weights = 'imagenet', include_top = True)
for i, layer in enumerate(base_mode.layers):
    print(i, layer.name, layer.output_shape)

model = Model(input_shape=base_mode.input, output = base_mode.get_layer('block4_pool').output)
img_path = 'cat.jpg'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

feature = model.predict(x)