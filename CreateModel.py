import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_folder = 'dataset1'

patient_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]

labels = pd.read_csv('tags.csv')

image_size = (256, 256)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    os.path.join(data_folder, 'train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    os.path.join(data_folder, 'validation'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

base_model = NASNetLarge(include_top=False, weights='imagenet', input_shape=(331, 331, 3))

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 10

model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

model.save('modelo.h5')