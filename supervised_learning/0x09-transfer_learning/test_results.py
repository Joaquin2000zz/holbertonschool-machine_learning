"""
Test the results of the training
"""
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as K


img_shape = (32, 32)
img_classes = {0: 'airplane',
               1: 'automobile',
               2: 'bird',
               3: 'cat',
               4: 'deer',
               5: 'dog',
               6: 'frog',
               7: 'horse',
               8: 'ship',
               9: 'truck'}
img_list = []
img_list_adjusted = []


for key, value in img_classes.items():
    img_list.append(Image.open(f'{value}.jpg'))
    prepared = np.array(img_list[key].resize(img_shape)) / 255.0
    img_list_adjusted.append(prepared)

model = K.models.load_model('cifar10.h5')
results = model.predict(np.array([img for img in img_list_adjusted]))

fig = plt.figure(figsize=(400, 400))
for key in img_classes.keys():
    fig.add_subplot(200, 200, key + 1)
    plt.imshow(img_list[key])
    plt.title(img_classes[np.argmax(results[key])])
    plt.axis('off')
plt.tight_layout()
plt.show()