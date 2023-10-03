import cv2
import os
import numpy as np
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Recall

# Load the images of a folder and prepare them for the Neural Network
def load_jpg_images(f_path, target_shape):
    loaded_images = []

    image_paths = os.listdir(f_path)

    for image_path in image_paths:
        image = cv2.imread(os.path.join(f_path, image_path))

        image_resized = cv2.resize(image, target_shape)

        loaded_images.append(image_resized)

    images_array = np.array(loaded_images)

    images_array = images_array.astype('float32') / 255.0

    return images_array

# Load the images of a folder and prepare them for the Neural Network
def load_jpg_masks(f_path, target_shape):
    loaded_masks = []

    mask_paths = os.listdir(f_path)

    for mask_path in mask_paths:
        mask = cv2.imread(os.path.join(f_path, mask_path), cv2.IMREAD_GRAYSCALE)

        mask_resized = cv2.resize(mask, target_shape)

        mask_resized = np.expand_dims(mask_resized, axis=-1)

        print(f"Shape of loaded mask '{mask_path}': {mask_resized.shape}")

        loaded_masks.append(mask_resized)

    masks_array = np.array(loaded_masks)

    masks_array = masks_array.astype('float32') / 255.0

    return masks_array

# Create a NasNetLarge pretrained using ImageNet, freezing all prertained layers.
def unet_nasnet(input_shape, num_classes):
    nasnet_base = NASNetLarge(input_shape=input_shape, include_top=False, weights='imagenet')

    for layer in nasnet_base.layers:
        layer.trainable = False

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
    shapeX, shapeY = 256, 256
    input_shape = (shapeX, shapeY, 3)  
    num_classes = 1

    # Create model and compile it using as loss function BinaryCrossentropy and defining as performance metrics the accuracy, recall and AUC.
    model = unet_nasnet(input_shape, num_classes)
    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy', Recall(), AUC()])

    # Load all images and masks.
    train_images = load_jpg_images("./dataset1_jpg/train/images/", (shapeX, shapeY))
    train_masks = load_jpg_masks("./dataset1_jpg/train/masks/", (shapeX, shapeY))
    val_images = load_jpg_images("./dataset1_jpg/validation/images/", (shapeX, shapeY))
    val_masks = load_jpg_masks("./dataset1_jpg/validation/masks/", (shapeX, shapeY))

    # Train the model
    model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=200, batch_size=16)

    # Save the trained model
    model.save('model.h5')


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    main()