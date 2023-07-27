import SimpleITK as sitk
import os
import numpy as np
import csv
from PIL import Image

# Variable global para guardar los ids de los pacientes, los usaremos para saber para que pacientes necesitamos definir la etiqueta de su respuesta a la PCR.
patient_ids = {}
# Número de pacientes que serán usados para entrenar el modelo, el resto serán usados para validarlo. 
train = 250

class PatientImage:
  original = None
  mask = None

# Convierte un archivo .dcm a uno jpeg.
def dicom_to_image(dicom):
    pixels = sitk.GetArrayFromImage(dicom)
    
    pixels = np.squeeze(pixels)  
    pixels = np.transpose(pixels)
    pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels))
    pixels = (pixels * 255).astype(np.uint8)
    return Image.fromarray(pixels)

def save_images(patient_id, n_dataset, saved_images):
    if train <= 0:
        output_folder = f"./dataset{n_dataset}/validation/{patient_id}/"
    else:
        output_folder = f"./dataset{n_dataset}/train/{patient_id}/"
    

    images = os.path.join(output_folder, "images/")
    masks = os.path.join(output_folder, "masks/")
    os.makedirs(images, exist_ok=True)
    os.makedirs(masks, exist_ok=True)
    count = 1
    for saved_image in saved_images:

        current_image = saved_images[saved_image]
        if current_image.original is not None and current_image.mask is not None:

            current_mask = dicom_to_image(current_image.mask)
            current_original = dicom_to_image(current_image.original)

            output_mask = os.path.join(masks, f"mask{count}.jpeg")
            current_mask.save(output_mask)

            output_original = os.path.join(images, f"image{count}.jpeg")
            current_original.save(output_original)
            count+= 1

# Itera sobre los estudios
# Selecciona las imagenes que están en una serie "TRACE" o "MASK"
def iterate_studies(folder_path, n_dataset, patient_id):

    saved_images = {}
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    folders.reverse()

    for folder in folders:

        #print("Series: " + folder)

        f_path = f"{folder_path}\\{folder}"
        image_paths = os.listdir(f_path)

        if not ("MASK" in  folder or "TRACE" in folder):
            continue
        
        for image_path in image_paths:
            i=1

            image = sitk.ReadImage(os.path.join(f_path, image_path))

            metadatos = image.GetMetaDataKeys()

            for clave in metadatos:
                if clave == "0020|0013":
                    n_imagen = image.GetMetaData(clave)

            if "MASK" in folder:
                saved_image = PatientImage()
                saved_image.mask = image
                saved_images[n_imagen] = saved_image

            else:
                if n_imagen in saved_images:

                    rescaler = sitk.RescaleIntensityImageFilter()
                    rescaled_image = rescaler.Execute(image)

                    laplacian_filter = sitk.LaplacianSharpeningImageFilter()
                    enhanced_image = laplacian_filter.Execute(rescaled_image)

                    saved_images[n_imagen].original = enhanced_image
            i += 1

    save_images(patient_id, n_dataset, saved_images)


#Itera sobre las series de cada paciente, perteneciendo cada una a una de las 4 fechas de su seguimiento.
def iterate_dates(folder_path, patient_id):
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    n_dataset = 1

    for folder in folders:

        #print("Date: " + folder)

        fpath = os.path.join(folder_path, folder)
        iterate_studies(fpath, n_dataset, patient_id)
        n_dataset += 1



# Itera sobre todos los pacientes.
def iterate_patients(folder_path):
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    global train

    for folder in folders:
        
        print("Patient: " + str(train))
        #print("Patient: " + folder)

        fpath = os.path.join(folder_path, folder)
        iterate_dates(fpath, folder)

        train -= 1

def remove_empty_folders_define_required_tags():
    n_dataset = 1

    while n_dataset < 5:

        ids = []
        folder_path = f"./dataset{n_dataset}/train/"

        folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

        for folder in folders:

            path = os.path.join(folder_path, folder)
            if len(os.listdir(path)) == 0:
                os.rmdir(path)
            else:
                ids.append(folder)

        folder_path = f"./dataset{n_dataset}/validation/"

        folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

        for folder in folders:
            path = os.path.join(folder_path, folder)
            if len(os.listdir(path)) == 0:
                os.rmdir(path)
            else:
                ids.append(folder)
        
        patient_ids[n_dataset] = ids

        n_dataset += 1



# Crear un archivo tags.csv etiquetando a los pacientes que hemos guardado en patient_ids, usando los datos metadata.csv. Los datos se guardan como id_paciente;respuesta
def retreive_tags():
    values1 = {}
    values2 = {}
    values3 = {}
    values4 = {}

    with open("metadata.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        next(csv_reader)
        for row in csv_reader:
            if row[1] in patient_ids[1]:
                values1[row[1]] = row[29]
            if row[1] in patient_ids[2]:
                values2[row[1]] = row[29]
            if row[1] in patient_ids[3]:
                values3[row[1]] = row[29]
            if row[1] in patient_ids[4]:
                values4[row[1]] = row[29]

    with open("tags1.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')

        for key, value in values1.items():
            print([key, value])
            writer.writerow([key, value])

    with open("tags2.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')

        for key, value in values2.items():
            print([key, value])
            writer.writerow([key, value])

    with open("tags3.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')

        for key, value in values3.items():
            print([key, value])
            writer.writerow([key, value])

    with open("tags4.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')

        for key, value in values4.items():
            print([key, value])
            writer.writerow([key, value])


def main():
    # Todos los datos vienen definidos en la carpeta entrada
    #iterate_patients("E:\\entrada")
    remove_empty_folders_define_required_tags()
    retreive_tags()

if __name__ == "__main__":
    main()

