# mnist_model.py

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical

def create_mnist_model():
    # Load and preprocess the MNIST dataset
    (train_images, train_labels), (_, _) = mnist.load_data()
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    train_labels = to_categorical(train_labels, num_classes=10)

    # Create the custom MNIST model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model using the provided train_images and train_labels
    model.fit(train_images, train_labels, epochs=5, batch_size=32)

    # Save the entire model to a file (including architecture and weights)
    model.save('custom_mnist_model.h5')

    return model



def process_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return None

    # Apply edge detection using the Canny algorithm
    edges = cv2.Canny(image, threshold1=100, threshold2=200)

    # Ensure that the edges image is not empty
    if np.sum(edges) == 0:
        print(f"Warning: No edges detected in image '{image_path}'")
        return None

    # Resize and normalize the edges image to 28x28
    processed_image = cv2.resize(edges, (28, 28)) / 255.0
    processed_image = processed_image.reshape(1, 28, 28, 1).astype('float32')  # Add batch dimension

    return processed_image

# def process_and_predict_mnist():
#     # Load the model architecture from the JSON file
#     with open('custom_mnist_model_architecture.json', 'r') as json_file:
#         model_json = json_file.read()

#     # Reconstruct the model from the saved architecture
#     model = tf.keras.models.model_from_json(model_json)

#     # Load the trained weights from the .h5 file
#     model.load_weights('custom_mnist_model_weights.h5')

#     # Compile the model (if you want to use it for predictions or further training)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # Process and predict the cell images as mentioned in the previous response
#     root_dir = "row"
#     row_predictions_sum = np.zeros(6, dtype=int)

#     for i in range(1, 7):  # Loop through each row
#         row_dir = os.path.join(root_dir, f"row_{i}")
#         for j in range(1, 6):  # Loop through each cell in half1 and half2  
#             cell_path = os.path.join(row_dir, f"half1/cell_{j}.png")
#             input_image = process_image(cell_path)

#             # Check if input_image is not None (edges detected and not empty)
#             if input_image is not None:
#                 # Make predictions using the pre-trained model
#                 predictions = model.predict(input_image)
#                 predicted_class = np.argmax(predictions)
#                 row_predictions_sum[i-1] += predicted_class
#             else:
#                 print(f"Warning: No valid edges detected for image {cell_path}")

#     return row_predictions_sum

# # Call the function to process and predict MNIST rows
# row_predictions_sum = process_and_predict_mnist()
# print(row_predictions_sum)

def process_and_predict_mnist():
    # Load the model from the .h5 file
    model = load_model('model.h5')

    # Process and predict the cell images as mentioned in the previous response
    root_dir = "row"
    row_predictions_sum = np.zeros(6, dtype=int)

    for i in range(1, 7):  # Loop through each row
        row_dir = os.path.join(root_dir, f"row_{i}")
        for j in range(1, 6):  # Loop through each cell in half1 and half2  
            cell_path = os.path.join(row_dir, f"half1/cell_{j}.png")
            input_image = process_image(cell_path)

            # Check if input_image is not None (edges detected and not empty)
            if input_image is not None:
                # Make predictions using the pre-trained model
                predictions = model.predict(input_image)
                predicted_class = np.argmax(predictions)
                row_predictions_sum[i-1] += predicted_class
            else:
                print(f"Warning: No valid edges detected for image {cell_path}")

    return row_predictions_sum

# Call the function to process and predict MNIST rows
row_predictions_sum = process_and_predict_mnist()
print(row_predictions_sum)