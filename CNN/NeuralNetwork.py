import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC, Recall

# Define una función para cargar los datos de segmentación
def load_data(batch_size, target_size):
    train_dir = 'dataset1_jpg/ti/'  # Directorio de imágenes de entrenamiento
    train_mask_dir = 'dataset1_jpg/tm/'     # Directorio de máscaras de segmentación de entrenamiento
    validation_dir = 'dataset1_jpg/vi/'  # Directorio de imágenes de validación
    validation_mask_dir = 'dataset1_jpg/vm/'     # Directorio de máscaras de segmentación de validación

    train_image_datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_mask_datagen = ImageDataGenerator(rescale=1.0/255.0)
    validation_image_datagen = ImageDataGenerator(rescale=1.0/255.0)
    validation_mask_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_image_generator = train_image_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    train_mask_generator = train_mask_datagen.flow_from_directory(
        train_mask_dir,
        target_size=(80,80),
        batch_size=batch_size,
        class_mode=None,  # Configura como None para cargar las máscaras como imágenes en blanco y negro
        color_mode='grayscale',
        shuffle=False  
    )

    validation_image_generator = validation_image_datagen.flow_from_directory(
        validation_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    validation_mask_generator = validation_mask_datagen.flow_from_directory(
        validation_mask_dir,
        target_size=(80,80),
        batch_size=batch_size,
        class_mode=None,  # Configura como None para cargar las máscaras como imágenes en blanco y negro
        color_mode='grayscale',
        shuffle=False
    )


    # Combinar los generadores de imágenes y máscaras
    train_data_generator = zip(train_image_generator, train_mask_generator)
    validation_data_generator = zip(validation_image_generator, validation_mask_generator)

    return train_data_generator, validation_data_generator

def dice_coefficient(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + 1e-5) / (union + 1e-5)

# Define la arquitectura de U-Net para segmentación
def create_unet(input_shape):
    # Entrada
    inputs = keras.Input(shape=input_shape)
    
    # Capa de convolución
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    
    # Capa de pooling
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Otras capas de convolución y pooling (puedes personalizar esto según tus necesidades)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # Capa de convolución de salida con 1 canal de salida (para máscaras binarias)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    # Crea el modelo
    modelo = keras.Model(inputs, outputs)
    
    return modelo

def main():
    batch_size = 8
    target_size = (320, 320)
    epochs = 200
    
    # Cargar datos de segmentación para entrenamiento y validación
    train_data_generator, validation_data_generator = load_data(batch_size, target_size)

    # Crear modelo U-Net
    model = create_unet((*target_size, 3))

    # Compilar el modelo con pérdida de entropía cruzada binaria y métrica de precisión
    model.compile(optimizer=Adam(lr=1e-4), 
                  loss='binary_crossentropy',  # Cambiar a binary_crossentropy
                  metrics=['accuracy', Recall(), AUC(), dice_coefficient])

    # Entrenar el modelo con los datos de entrenamiento y validar con los datos de validación
    history = model.fit(
        train_data_generator,
        steps_per_epoch=1582 // batch_size,
        epochs=epochs,
        validation_data=validation_data_generator,
        validation_steps=57 // batch_size
    )

    # Visualizar resultados, por ejemplo, imágenes de entrada y máscaras segmentadas

    model.save('modelo_jpg.h5')

if __name__ == "__main__":
    # Asegúrate de tener los directorios 'dataset/ti', 'dataset/tm', 'dataset/vi' y 'dataset/vm' con las imágenes y máscaras correspondientes.
    main()