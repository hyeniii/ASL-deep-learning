import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

import src.image_processor as ip

# Process training and validation images
ip.process_images_indirect("./data/roboflow/raw/train", 
                           "./data/roboflow/processed/train_bb", 
                           "./annotations/train_annotations.csv")

ip.process_images_indirect("./data/roboflow/raw/validation", 
                           "./data/roboflow/processed/validation_bb", 
                           "./annotations/validation_annotations.csv")

# Define variables (extract from yaml file eventually)
train_dir = "data/roboflow/processed/train_bb"
validation_dir = "data/roboflow/processed/validation_bb"
batch_size = 32
target_size = (64, 64)
class_mode = "categorical"

# Define training generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode="nearest")

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode=class_mode)

# Define validation generator
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    class_mode=class_mode)

# Define number of classes
num_classes = len(train_generator.class_indices)

# Define model architecture (first decent model)
# model = Sequential([
#     Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 3)),
#     BatchNormalization(),
#     Conv2D(64, (3, 3), activation="relu", padding="same"),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(128, (3, 3), activation="relu", padding="same"),
#     BatchNormalization(),
#     Conv2D(128, (3, 3), activation="relu", padding="same"),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(256, (3, 3), activation="relu", padding="same"),
#     BatchNormalization(),
#     Conv2D(256, (3, 3), activation="relu", padding="same"),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(512, (3, 3), activation="relu", padding="same"),
#     BatchNormalization(),
#     Conv2D(512, (3, 3), activation="relu", padding="same"),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Flatten(),
#     Dense(1024, activation="relu", kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     Dropout(0.6),
#     Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
#     BatchNormalization(),
#     Dropout(0.6),
#     Dense(num_classes, activation="softmax")
# ])

# Define model architecture (current best model)
model = Sequential([
    Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(64, 64, 3)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(512, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(2048, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(1024, activation="relu", kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint("./artifacts/best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1)
callbacks_list = [checkpoint, early_stopping]

# Train model
with tf.device("/GPU:0"):
    epochs = 300
    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=len(validation_generator),
                        callbacks=callbacks_list,
                        verbose=1)
