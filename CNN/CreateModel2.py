import tensorflow as tf
import SimpleITK as sitk
import os
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

def load_dicom_images(f_path):

    loaded_images = []

    image_paths = os.listdir(f_path)

    for image_path in image_paths:  
            # Read the image.
            image = sitk.ReadImage(os.path.join(f_path, image_path))
            image_array = sitk.GetArrayFromImage(image)  # Convertir la imagen a una matriz NumPy
            loaded_images.append(image_array)

    return loaded_images

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

# Definir las dimensiones de entrada de las imágenes y el número de clases (1 en este caso, ya que la máscara es binaria)
input_shape = (256, 256, 3)  # Asegúrate de que las imágenes de entrada tengan 3 canales si usas NasNetLarge
num_classes = 1

# Crear el modelo U-Net con NasNet preentrenada
model = unet_nasnet(input_shape, num_classes)

# Compilar el modelo con la función de pérdida adecuada para la segmentación (por ejemplo, binary_crossentropy)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_images = load_dicom_images("./dataset1/train/images/")
train_masks = load_dicom_images("./dataset1/train/maks/")
val_images = load_dicom_images("./dataset1/validation/images/")
val_masks = load_dicom_images("./dataset1/validation/maks/")

# Entrenar el modelo con tus datos de imágenes y máscaras
# Por ejemplo:
model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=10, batch_size=16)

# Guardar el modelo entrenado para poder utilizarlo más tarde
model.save('modelo1.h5')
