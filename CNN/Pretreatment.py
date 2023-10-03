import SimpleITK as sitk
import os
from datetime import datetime

# Create the dirs in which the new data will be stored.
def create_dirs():
    for i in range(4):
        val_images = f"dataset{i+1}/validation/images"
        val_masks = f"dataset{i+1}/validation/masks"
        train_images = f"dataset{i+1}/train/images"
        train_masks = f"dataset{i+1}/train/masks"

        os.makedirs(val_images, exist_ok=True)
        os.makedirs(val_masks, exist_ok=True)
        os.makedirs(train_images, exist_ok=True)
        os.makedirs(train_masks, exist_ok=True)

# Store the images and masks in the validation or the train folder.
def save_images(n_dataset, images_masks, patient_id):
    # If we have enough patients for the train part, we start storing new images in the validation folder
    if train <= 0:
        output_folder = f"dataset{n_dataset}/validation/"
    else:
        output_folder = f"dataset{n_dataset}/train/"


    for image_id, image_mask in images_masks.items():

        # In some cases the mask doesn't have a matching image, we need a 1:1 relationship, so we just don't save anything.
        if len(image_mask) != 2:
            print(f"--> No image for mask: {image_id} in dataset {n_dataset}")
            continue

        image = image_mask[1]
        mask = image_mask[0]

        # Store the treated image.
        output_image = os.path.join(output_folder, f"images/image_{patient_id}_{image_id}.dcm")
        sitk.WriteImage(image, output_image)

        # Store the treated mask.
        output_mask = os.path.join(output_folder, f"masks/mask_{patient_id}_{image_id}.dcm")
        sitk.WriteImage(mask, output_mask)

# Iterate over the studies of the series.
# We want to store all the masks defined in each study and for each mask find it's original image.
def iterate_studies(folder_path, n_dataset, patient_id):

    # Store the images and the masks
    images_masks = {}

    # We fetch the folders in reverse order because we want the MASKs to appears before the TRACEs
    folders = sorted([name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))], reverse=True)

    for folder in folders:

        f_path = f"{folder_path}\\{folder}"
        image_paths = os.listdir(f_path)

        # We only work with the folders with the masks or the images used to create the masks
        if not ("MASK" in  folder or "TRACE" in folder):
            continue
        
        for image_path in image_paths:
            
            # Read the image.
            image = sitk.ReadImage(os.path.join(f_path, image_path))

            # Fetch the id of the image
            metadata = image.GetMetaDataKeys()
            for key in metadata:
                if key == "0020|0013":
                    image_id = image.GetMetaData(key)

            # If it is a mask, we store it.
            if "MASK" in folder:
                images_masks[image_id] = [image]

            # If it is the original image, we make sure that we have a matching mask; if so, we store it.
            else:
                if image_id in images_masks:

                    # We rescale the intensity of the pixels to enhance further use of the image.
                    rescaler = sitk.RescaleIntensityImageFilter()
                    rescaled_image = rescaler.Execute(image)
                    # We enhance the borders and details of the image.
                    laplacian_filter = sitk.LaplacianSharpeningImageFilter()
                    enhanced_image = laplacian_filter.Execute(rescaled_image)

                    images_masks[image_id].append(enhanced_image)

    save_images(n_dataset, images_masks, patient_id)

# Give a folder_name that starts with a date in %m-%d-%Y format, it extracts the date.
def extract_date_from_folder_name(folder_name):
    date_str = folder_name[:10]
    return datetime.strptime(date_str, "%m-%d-%Y").date()

# Iterate over each series of the patients.
def iterate_series(folder_path, patient_id):
    folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    #Sort by date, instead of sorting by name.
    folders = sorted(folders, key=lambda x: extract_date_from_folder_name(x))

    # We need this variable to select in which folder we store the images.
    n_dataset = 1

    for folder in folders:

        fpath = os.path.join(folder_path, folder)
        iterate_studies(fpath, n_dataset, patient_id)
        n_dataset += 1

# Iterate over each patient.
def iterate_patients(folder_path):
    folders = sorted([name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))])

    global train

    for folder in folders:
        print("Patient: " + str(train))

        fpath = os.path.join(folder_path, folder)
        iterate_series(fpath, folder)

        train -= 1

# Number of patients that will be used to train the model (the rest will be used to validate).
train = 250

def main():

    # Create the folders to store the treated data.
    create_dirs()
    # Retrieve all data  
    iterate_patients("E:\\entrada")

if __name__ == "__main__":
    # We want to use the relative path of the file (instead of the project one).
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    main()

