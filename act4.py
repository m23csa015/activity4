import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN model
def create_simple_cnn(input_shape, num_classes):
    model = models.Sequential()

    # First convolutional layer with 32 filters, 3x3 kernel size, and ReLU activation
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Second convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional layer with 64 filters, 3x3 kernel size, and ReLU activation
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Flatten the output of the convolutional layers
    model.add(layers.Flatten())

    # Fully connected layer with 64 units and ReLU activation
    model.add(layers.Dense(64, activation='relu'))

    # Output layer with softmax activation for classification
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Define input shape (e.g., for 28x28 grayscale images)
input_shape = (28, 28, 1)  # Height, Width, Channels

# Define the number of classes (e.g., for a 10-class classification task)
num_classes = 15

# Create the CNN model
simple_cnn_model = create_simple_cnn(input_shape, num_classes)

# Compile the model with appropriate loss function, optimizer, and metrics
simple_cnn_model.compile(optimizer='sgd',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

# Print the model summary
simple_cnn_model.summary()
print('change optimizer')
