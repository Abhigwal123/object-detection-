import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification: laptop or mobile
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data generators for training, validation, and testing
train_datagen = ImageDataGenerator(rescale=1.0/255)
train_generator = train_datagen.flow_from_directory(
    'training',
    target_size=(200, 200),
    batch_size=20,
    class_mode='binary'  # Binary classification
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
    'validation',
    target_size=(200, 200),
    batch_size=20,
    class_mode='binary'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)

test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    'testing',
    target_size=(200, 200),
    batch_size=1,
    class_mode=None,  # Set class_mode to None for prediction
    shuffle=False  # Keep the order of images for evaluation
)

import os

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# Get the list of image file names in the testing directory
test_image_filenames = test_generator.filenames

# Make predictions
predictions = model.predict(test_generator)
binary_predictions = np.round(predictions)
true_labels = test_generator.classes
# Calculate accuracy
accuracy = np.mean(binary_predictions == true_labels)

print(f"Prediction accuracy: {accuracy * 100:.2f}%")

# Iterate through the images, their filenames, and predictions
for i in range(len(test_image_filenames)):
    img = image.load_img(os.path.join('testing', test_image_filenames[i]), target_size=(150, 150))
    plt.imshow(img)
    
    if predictions[i] > 0.5:
        label = "Mobile"
    else:
        label = "Laptop"
    
    plt.title(label)
    plt.show()
