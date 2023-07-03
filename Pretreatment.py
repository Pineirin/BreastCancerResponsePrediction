import SimpleITK as sitk
import os
import numpy as np
import csv

class Image:
  original = None
  mask = None

def iterate_studies(folder_path, n_dataset, n_patient):

    imagenes_con_mascara = []

    saved_images = {}

    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    folders.reverse()

    for folder in folders:

        f_path = f"{folder_path}\\{folder}"
        image_paths = os.listdir(f_path)

        if not ("MASK" in  folder or "TRACE" in folder):
            continue
        
        print("Series: " + folder)

        for image_path in image_paths:
            
            i=1

            image = sitk.ReadImage(os.path.join(f_path, image_path))

            metadatos = image.GetMetaDataKeys()

            for clave in metadatos:
                if clave == "0020|0013":
                    n_imagen = image.GetMetaData(clave)

            if "MASK" in folder:
                imagenes_con_mascara.append(n_imagen)

                saved_image = Image()
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

    if n_patient > 15:
        output_folder = f"./dataset{n_dataset}/validation/patient{n_patient}/"
    else:
        output_folder = f"./dataset{n_dataset}/train/patient{n_patient}/"
    os.makedirs(output_folder, exist_ok=True)
    count = 1

    for saved_image in saved_images:

        current_image = saved_images[saved_image]


        if current_image.original is not None and current_image.mask is not None:              
            output_mask = os.path.join(output_folder, f"mask{count}.dcm")
            sitk.WriteImage(current_image.mask, output_mask)

            output_original = os.path.join(output_folder, f"image{count}.dcm")
            sitk.WriteImage(current_image.original, output_original)
            count+= 1


def iterate_dates(folder_path, n_patient):
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    n_dataset = 1

    for folder in folders:

        print("Date: " + folder)

        fpath = os.path.join(folder_path, folder)
        iterate_studies(fpath, n_dataset, n_patient)
        n_dataset += 1

def iterate_patients(folder_path):
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    n_patient = 1

    for folder in folders:
        
        print("Patient: " + folder)
        patient_ids.append(folder)

        fpath = os.path.join(folder_path, folder)
        iterate_dates(fpath, n_patient)
        n_patient += 1

def retreive_tags():
    values = {}

    with open("metadata.csv", 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        next(csv_reader)
        i=1
        for row in csv_reader:
            if row[1] in patient_ids:
                values[f"patient{i}"] = row[29]
                i+=1



    with open("tags.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')

        for key, value in values.items():
            print([key, value])
            writer.writerow([key, value])



patient_ids = []
train = 15

iterate_patients(".\\ACRIN-6698")
retreive_tags()