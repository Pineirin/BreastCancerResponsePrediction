import SimpleITK as sitk
import os
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

# Directorio que contiene las imágenes DICOM
input_directory = 'dataset1/validation/images/'
output_directory = 'dataset1_jpg/validation/images/'

# Lista de nombres de archivos DICOM en el directorio
dicom_file_list = [f for f in os.listdir(input_directory) if f.endswith('.dcm')]

# Itera a través de los archivos DICOM y conviértelos a JPEG
for dicom_file in dicom_file_list:
    # Ruta completa del archivo DICOM
    dicom_path = os.path.join(input_directory, dicom_file)
    
    # Lee la imagen DICOM
    image = sitk.ReadImage(dicom_path)

    # Asegúrate de que la imagen sea 3D
    if image.GetDimension() != 3:
        raise ValueError(f"La imagen {dicom_file} no es 3D")

    # Extrae una rebanada específica (por ejemplo, la rebanada del medio)
    slice_index = image.GetSize()[2] // 2  # Tomamos la rebanada del medio
    image_slice = image[:, :, slice_index]

    # Convierte la rebanada en un arreglo NumPy
    image_array = sitk.GetArrayFromImage(image_slice)

    # Normaliza los valores de píxeles para que estén en el rango 0-255
    image_array = ((image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255).astype(np.uint8)

    # Convierte el arreglo NumPy a una imagen PIL
    image_pil = Image.fromarray(image_array)

    # Nombre del archivo JPEG de salida (mismo nombre base que el archivo DICOM)
    output_file = os.path.splitext(dicom_file)[0] + '.jpg'
    
    # Guarda la rebanada como JPEG en el mismo directorio original
    output_path = os.path.join(output_directory, output_file)
    image_pil.save(output_path)

    print(f"La rebanada de la imagen DICOM {dicom_file} se ha convertido y guardado como {output_file}")