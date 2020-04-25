import tensorflow as tf
from tensorflow.keras.layers import *
import numpy as np
import img_loader

x_train, y_train = img_loader.load_data(r"C:\Users\const\Documents\scripts\dl\data\dogs-vs-cats\train\all")
x_train = x_train / 255.0

model = tf.keras.Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, batch_size=30, epochs=20)
model.save("model3.h5")