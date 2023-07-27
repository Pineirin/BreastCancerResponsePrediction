import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

data_folder = 'dataset1'
labels_file = 'tags1.csv'

patient_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]

labels = pd.read_csv(labels_file, delimiter=';')

# Convertir las etiquetas a valores numéricos en el rango de 0 a 2
labels['response'] = pd.Categorical(labels['response'])
labels['response'] = labels['response'].cat.codes

# Convertir las etiquetas a one-hot encoding
labels['response'] = to_categorical(labels['response'], num_classes=3)  # Cambiar num_classes a 3

# Resto del código sigue igual

image_size = (256, 256)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Utilizar 'categorical' como class_mode para el generador de flujo de datos de entrenamiento
train_generator = datagen.flow_from_directory(
    os.path.join(data_folder, 'train'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',  # Cambiar a 'sparse'
    shuffle=True
)

# Utilizar 'sparse_categorical' como class_mode para el generador de flujo de datos de validación
validation_generator = datagen.flow_from_directory(
    os.path.join(data_folder, 'validation'),
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse'  # Cambiar a 'sparse'
)

base_model = NASNetLarge(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(3, activation='softmax')  # Utilizar 3 unidades para clasificación multiclase
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 10

model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

model.save('modelo.h5')