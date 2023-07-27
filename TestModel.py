import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ruta de la carpeta de datos
data_folder = 'dataset'

# Cargar el modelo entrenado
model = tf.keras.models.load_model('modelo.h5')

# Obtener las etiquetas de los pacientes
labels = pd.read_csv('etiquetas.csv')

# Obtener la lista de subcarpetas (pacientes) de prueba
test_patient_folders = [folder for folder in os.listdir(os.path.join(data_folder, 'test')) if os.path.isdir(os.path.join(data_folder, 'test', folder))]

# Preprocesamiento de las imágenes de prueba
image_size = (331, 331)  # Tamaño de entrada requerido por NasNetLarge

X_test = []  # Lista para almacenar las imágenes de prueba
y_test = []  # Lista para almacenar las etiquetas de prueba

for folder in test_patient_folders:
    image_path = os.path.join(data_folder, 'test', folder, 'imagen1.dcm')  # Tomamos la primera imagen de cada paciente como ejemplo
    label = labels.loc[labels['Paciente'] == folder, 'Respuesta'].values[0]  # Obtener la etiqueta correspondiente al paciente

    image = load_img(image_path, target_size=image_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    
    X_test.append(image)
    y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Evaluación del modelo
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')