from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define variables (extract from yaml file eventually)
train_dir = "data/roboflow/processed/train_bb"
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
