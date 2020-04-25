import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import img_loader

x_train, y_train = img_loader.load_data(r"C:\Users\const\Documents\scripts\dl\data\dogs-vs-cats\train\all")
x_train = x_train / 255.0

model = tf.keras.Sequential()
model.add(Conv2D(200, kernel_size=(10,10), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(6,6)))

model.add(Conv2D(100, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Conv2D(50, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, batch_size=50, epochs=20)
model.save("model1.h5")