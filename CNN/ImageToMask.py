import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import os

def load_and_preprocess_image(image_path, target_shape=(256, 256)):
    image = sitk.ReadImage(image_path)
        
    # As it is a 3D image, we want to get a slice to make it a 2D (our images only have one slice, but we pick the middle one just in case.)
    num_slices = image.GetSize()[2]
    slice_index = num_slices // 2
    image_slice = image[:, :, slice_index]

    # Resize the image to the target size.
    target_size = (target_shape[0], target_shape[1])
    image_resized = sitk.GetArrayFromImage(sitk.Resample(image_slice, target_size, sitk.Transform(), sitk.sitkLinear, image_slice.GetOrigin(), (1.0, 1.0), image_slice.GetDirection(), 0.0, image_slice.GetPixelID()))

    # Expand the image to three identical channels to match the pre-trained network.
    image_rgb = np.expand_dims(image_resized, axis=-1)
    image_rgb = np.concatenate([image_rgb] * 3, axis=-1)

    # Convert the image to the appropriate data type and scale to [0, 1].
    image_rgb = image_rgb.astype('float32') / 255.0

    return image_rgb

def is_mask_completely_black(mask):
    return np.all(mask == 0)

def main():
    # Load the trained model
    model = tf.keras.models.load_model('modelo1.h5')

    # Load and preprocess the input image
    input_image_path = 'image_ACRIN-6698-102212_11.dcm'
    input_image = load_and_preprocess_image(input_image_path)
    
    # Generate the mask using the model
    predicted_mask = model.predict(np.expand_dims(input_image, axis=0))[0]
    normalized_predicted_mask = (predicted_mask - np.min(predicted_mask)) / (np.max(predicted_mask) - np.min(predicted_mask))

    predicted_mask_binary = (normalized_predicted_mask > 0.0000000000001).astype(np.uint8)

    # Convert the mask to SimpleITK format
    predicted_mask_sitk = sitk.GetImageFromArray(predicted_mask_binary)

    # Print the coordinates of pixels with value 1 in the mask
    mask_array = sitk.GetArrayFromImage(predicted_mask_sitk)
    mask_pixels_with_value_1 = np.argwhere(mask_array == 1)
    
    if mask_pixels_with_value_1.shape[0] > 0:
        print("Píxeles con valor 1 en la máscara:")
        for pixel_coord in mask_pixels_with_value_1:
            print("Coordenadas:", pixel_coord)
    else:
        print("No hay píxeles con valor 1 en la máscara.")


    result = is_mask_completely_black(predicted_mask_sitk)

    if result:
        print("La máscara está completamente en negro.")
    else:
        print("La máscara no está completamente en negro.")

    # Save the generated mask
    mask_save_path = 'predicted_mask.dcm'
    sitk.WriteImage(predicted_mask_sitk, mask_save_path)

    print("Predicted mask saved:", mask_save_path)

if __name__ == "__main__":
    # We want to use the relative path of the file (instead of the project one).
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    main()