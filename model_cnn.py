import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical

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

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=32)

# Save the model architecture to a JSON file
with open('custom_mnist_model_architecture.json', 'w') as json_file:
    json_file.write(model.to_json())

# Save the trained weights to a file in .h5 format
model.save_weights('custom_mnist_model_weights.h5')
