import os
import pandas as pd
import numpy as np
from keras.applications import NASNetLarge
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from PIL import Image

train_data_dir = 'dataset1/train'
validation_data_dir = 'dataset1/validation'

df = pd.read_csv("tags1.csv", delimiter=';')
df['response'] = df['response'].astype(str)

patient_response_map = dict(zip(df['patient_id'], df['response']))

train_patients = os.listdir(train_data_dir)
validation_patients = os.listdir(validation_data_dir)

train_images = []
train_labels = []
validation_images = []
validation_labels = []

for patient in train_patients:
    patient_dir = os.path.join(train_data_dir, patient)
    image_dir = os.path.join(patient_dir, 'images')
    mask_dir = os.path.join(patient_dir, 'masks')
    images = os.listdir(image_dir)
    
    for image_file in images:
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        train_images.append(img_array)
        train_labels.append(patient_response_map[patient])

for patient in validation_patients:
    patient_dir = os.path.join(validation_data_dir, patient)
    image_dir = os.path.join(patient_dir, 'images')
    mask_dir = os.path.join(patient_dir, 'masks')
    images = os.listdir(image_dir)
    
    for image_file in images:
        image_path = os.path.join(image_dir, image_file)
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        validation_images.append(img_array)
        validation_labels.append(patient_response_map[patient])

train_images = np.array(train_images)
train_labels = np.array(train_labels)
validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)

train_labels = to_categorical(train_labels, num_classes=3)
validation_labels = to_categorical(validation_labels, num_classes=3)

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True 
)

validation_datagen = ImageDataGenerator(rescale=1.0/255.0)

base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='softmax')) 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 16
train_generator = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
validation_generator = validation_datagen.flow(validation_images, validation_labels, batch_size=batch_size)

steps_per_epoch = len(train_images) // batch_size
validation_steps = len(validation_images) // batch_size

model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=10, validation_data=validation_generator, validation_steps=validation_steps)

model.save('model.h5')

