import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ruta de la carpeta de datos
data_folder = 'dataset1'

# Obtener la lista de subcarpetas (pacientes)
patient_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]

# Obtener las etiquetas de los pacientes
labels = pd.read_csv('tags.csv')

# Preprocesamiento de los datos
image_size = (331, 331)  # Tamaño de entrada requerido por NasNetLarge
batch_size = 32

# Crear una lista para almacenar los datos de cada paciente
X_data = []
y_data = []

for folder in patient_folders:
    patient_images = []
    label = labels.loc[labels['Patient'] == folder, 'Answer'].values[0]  # Obtener la etiqueta correspondiente al paciente

    image_folder = os.path.join(data_folder, folder)
    image_files = [file for file in os.listdir(image_folder) if file.endswith('.dcm')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = load_img(image_path, target_size=image_size)
        image = img_to_array(image)
        image = preprocess_input(image)

        patient_images.append(image)

    X_data.extend(patient_images)
    y_data.extend([label] * len(patient_images))

# Convertir las listas a arrays numpy
X_data = np.array(X_data)
y_data = np.array(y_data)

# Calcular la cantidad de padding requerido
max_samples = max([len(patient_images) for patient_images in X_data])
padding_samples = max_samples - np.array([len(patient_images) for patient_images in X_data])

# Realizar el padding
X_data_padded = []
y_data_padded = []

for i, patient_images in enumerate(X_data):
    padding = np.zeros_like(patient_images[0])  # Crear una imagen de padding con todos los valores en cero

    padded_images = np.concatenate([patient_images, np.tile(padding, (padding_samples[i], 1, 1, 1))], axis=0)

    X_data_padded.append(padded_images)
    y_data_padded.extend([y_data[i]] * max_samples)

# Convertir las listas a arrays numpy
X_data_padded = np.array(X_data_padded)
y_data_padded = np.array(y_data_padded)

# Preparar los generadores de datos
datagen = ImageDataGenerator(rescale=1./255)  # Normalización de píxeles entre 0 y 1

# Generador de datos para las imágenes de entrenamiento
train_generator = datagen.flow(
    X_data_padded,
    y_data_padded,
    batch_size=batch_size
)

# Crear el modelo basado en NasNetLarge
base_model = NASNetLarge(include_top=False, weights='imagenet', input_shape=(331, 331, 3))

# Congelar las capas del modelo base
base_model.trainable = False

# Agregar capas adicionales para adaptar el modelo a la clasificación binaria
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
epochs = 10

model.fit(
    train_generator,
    epochs=epochs
)