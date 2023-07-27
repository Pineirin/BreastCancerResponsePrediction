import tensorflow as tf
import SimpleITK as sitk
import os
import numpy as np
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model


# Obtiene el directorio del archivo actual
current_dir = os.path.dirname(os.path.abspath(__file__))
# Cambia el directorio de trabajo al directorio actual
os.chdir(current_dir)

def load_dicom_images(f_path, target_shape=(256, 256)):

    loaded_images = []

    image_paths = os.listdir(f_path)

    for image_path in image_paths:  
        # Read the image.
        image = sitk.ReadImage(os.path.join(f_path, image_path))
        # Obtener el número de rebanas (slices) en el volumen
        num_slices = image.GetSize()[2]
        # Seleccionar la rebanada intermedia (slice)
        slice_index = num_slices // 2
        image_slice = image[:, :, slice_index]

        # Redimensionar la imagen para tener el mismo tamaño
        target_size = (target_shape[0], target_shape[1])
        image_resized = sitk.GetArrayFromImage(sitk.Resample(image_slice, target_size, sitk.Transform(), sitk.sitkLinear, image_slice.GetOrigin(), (1.0, 1.0), image_slice.GetDirection(), 0.0, image_slice.GetPixelID()))

        # Agregar una dimensión extra para convertir la imagen en RGB (3 canales)
        image_rgb = np.expand_dims(image_resized, axis=-1)
        image_rgb = np.concatenate([image_rgb] * 3, axis=-1)

        loaded_images.append(image_rgb)

    images_array = np.array(loaded_images)

    # Escalar las intensidades de píxeles al rango [0, 1]
    images_array = images_array.astype('float32') / 255.0

    return images_array

def load_dicom_masks(f_path, target_shape=(256, 256)):
    loaded_masks = []

    mask_paths = os.listdir(f_path)

    for mask_path in mask_paths:
        # Read the mask.
        mask = sitk.ReadImage(os.path.join(f_path, mask_path))
        # Obtener el número de rebanas (slices) en la máscara
        num_slices = mask.GetSize()[2]
        # Seleccionar la rebanada intermedia (slice)
        slice_index = num_slices // 2
        mask_slice = mask[:, :, slice_index]

        # Redimensionar la máscara para tener el mismo tamaño
        target_size = (target_shape[0], target_shape[1])
        mask_resized = sitk.GetArrayFromImage(sitk.Resample(mask_slice, target_size, sitk.Transform(), sitk.sitkLinear, mask_slice.GetOrigin(), (1.0, 1.0), mask_slice.GetDirection(), 0.0, mask_slice.GetPixelID()))

        loaded_masks.append(mask_resized)

    masks_array = np.array(loaded_masks)

    # Escalar las intensidades de píxeles al rango [0, 1]
    masks_array = masks_array.astype('float32') / 255.0

    return masks_array

def unet_nasnet(input_shape, num_classes):
    # Cargar la parte de convolución de NasNetLarge sin las capas totalmente conectadas y sin pesos preentrenados
    nasnet_base = NASNetLarge(input_shape=input_shape, include_top=False, weights=None)

    # Obtener la última capa activada del modelo NasNetLarge
    nasnet_output = nasnet_base.layers[-1].output

    # Agregar capas personalizadas para la tarea de segmentación
    up1 = UpSampling2D(size=(4, 4))(nasnet_output)
    conv1 = Conv2D(512, 3, activation='relu', padding='same')(up1)

    up2 = UpSampling2D(size=(2, 2))(conv1)
    conv2 = Conv2D(256, 3, activation='relu', padding='same')(up2)

    up3 = UpSampling2D(size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(up3)

    up4 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(up4)

    # Output
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(conv4)

    model = Model(inputs=nasnet_base.input, outputs=outputs)

    return model

# Obtiene el directorio del archivo actual
current_dir = os.path.dirname(os.path.abspath(__file__))
# Cambia el directorio de trabajo al directorio actual
os.chdir(current_dir)


# Definir las dimensiones de entrada de las imágenes y el número de clases (1 en este caso, ya que la máscara es binaria)
input_shape = (256, 256, 3)  # Asegúrate de que las imágenes de entrada tengan 3 canales si usas NasNetLarge
num_classes = 1

# Crear el modelo U-Net con NasNet preentrenada
model = unet_nasnet(input_shape, num_classes)

# Compilar el modelo con la función de pérdida adecuada para la segmentación (por ejemplo, binary_crossentropy)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_images = load_dicom_images("dataset1/train/images/")
train_masks = load_dicom_masks("dataset1/train/masks/")
val_images = load_dicom_images("dataset1/validation/images/")
val_masks = load_dicom_masks("dataset1/validation/masks/")

# Entrenar el modelo con tus datos de imágenes y máscaras
# Por ejemplo:
model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=10, batch_size=16)

# Guardar el modelo entrenado para poder utilizarlo más tarde
model.save('modelo1.h5')
