import tensorflow as tf
import SimpleITK as sitk
import os
import numpy as np
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import AUC, Recall

# Load the images of a folder a prepare them for the Neural Network
def load_dicom_images(f_path, target_shape=(256, 256)):

    loaded_images = []

    image_paths = os.listdir(f_path)

    for image_path in image_paths:  
        # Read the image.
        image = sitk.ReadImage(os.path.join(f_path, image_path))
        
        # As it is a 3D image, we want to get an slice to make it a 2D (our images only have one slice, but we pick the middle one just in case)
        num_slices = image.GetSize()[2]
        slice_index = num_slices // 2
        image_slice = image[:, :, slice_index]

        # We make sure our images have the target shape
        target_size = (target_shape[0], target_shape[1])
        image_resized = sitk.GetArrayFromImage(sitk.Resample(image_slice, target_size, sitk.Transform(), sitk.sitkLinear, image_slice.GetOrigin(), (1.0, 1.0), image_slice.GetDirection(), 0.0, image_slice.GetPixelID()))

        loaded_images.append(image_resized)

    # Numpy array for the nueral network
    images_array = np.array(loaded_images)

    # Scale the image to the range [0, 1] for the neural network
    images_array = images_array.astype('float32') / 255.0

    return images_array

# Creates a Nasnet model for a segmentation task
def unet_nasnet(input_shape, num_classes):

    # We create a NasNetLarge and add convolutional layers to it.
    nasnet_base = NASNetLarge(input_shape=input_shape, include_top=False, weights=None)

    nasnet_output = nasnet_base.layers[-1].output

    up1 = UpSampling2D(size=(4, 4))(nasnet_output)
    conv1 = Conv2D(512, 3, activation='relu', padding='same')(up1)

    up2 = UpSampling2D(size=(2, 2))(conv1)
    conv2 = Conv2D(256, 3, activation='relu', padding='same')(up2)

    up3 = UpSampling2D(size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(up3)

    up4 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(64, 3, activation='relu', padding='same')(up4)

    outputs = Conv2D(num_classes, 1, activation='sigmoid')(conv4)

    model = Model(inputs=nasnet_base.input, outputs=outputs)

    return model

def main():
    # Define image size and color channels
    input_shape = (256, 256, 1)  
    num_classes = 1

    # Create the model
    model = unet_nasnet(input_shape, num_classes)

    # Compile it for a binary classification using the most relevant metrics
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(), AUC()])

    # Load all images and masks.
    train_images = load_dicom_images("dataset1/train/images/")
    train_masks = load_dicom_images("dataset1/train/masks/")
    val_images = load_dicom_images("dataset1/validation/images/")
    val_masks = load_dicom_images("dataset1/validation/masks/")

    # Train the model
    model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=200, batch_size=16)

    # Save the trained model
    model.save('modelo1.h5')

if __name__ == "__main__":
    # We want to use the relative path of the file (instead of the project one).
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    main()
