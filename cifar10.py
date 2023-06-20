# Object Recognition Demo using CNN
# Trained and tested a CNN model to detect objects.
# cifar10 Two-Layer Dense Network (Linear Classifier) Demo


# Remove the comment of below imports if running on colab
# !pip install tf2onnx
# !pip install onnxruntime

import tensorflow as tf
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout
from keras.models import Sequential
from keras.datasets import cifar10
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from keras.optimizers import Adam, RMSprop
import numpy as np
from keras.utils import to_categorical
import tf2onnx
import cv2
import onnxruntime as ort

# Load cifar10 datasets for training and testing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize inputs to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

num_epochs = 200                        # One epoch of training means processing each examplar in the training set exactly once. 
batch_size = 100                        # Number of exemplars processed at once. Larger batches speed up computation but need more memory.
size_input = (32, 32, 3)                   # We have 3-dimensional input arrays representing 32x32x3 pixel intensity values
size_output = len(np.unique(y_train))   # Number of output-layer neurons must match number of individual classes (here: 10 classes, one for each digit)

num_train_exemplars = x_train.shape[0] 

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

num_epochs = 50
batch_size = 128                     # Number of exemplars processed at once. Larger batches speed up computation but need more memory.
size_input = (32, 32, 3)                   # We have 3-dimensional input arrays representing 32x32x3 pixel intensity values
size_output = len(np.unique(y_train))   # Number of output-layer neurons must match number of individual classes (here: 10 classes, one for each digit)

num_train_exemplars = x_train.shape[0] 

# Build the model (computational graph)
cifar10_model = Sequential(
    [Input(shape=size_input, name='input_layer'),
    BatchNormalization(),
    Conv2D(32, kernel_size=(3, 3), padding='same', activation='ReLU', name='conv_1'),
    BatchNormalization(),
    Conv2D(32, kernel_size=(3, 3), padding='same', activation='ReLU', name='conv_2'),
    BatchNormalization(),
    MaxPool2D((2, 2), name='maxpool_1'),
    Dropout(0.4),
    Conv2D(64, kernel_size=(3, 3), padding='same', activation='ReLU', name='conv_3'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), padding='same', activation='ReLU', name='conv_4'),
    BatchNormalization(),
    MaxPool2D((2, 2), name='maxpool_2'),
    Dropout(0.4),
    Conv2D(128, kernel_size=(3, 3), padding='same', activation='ReLU', name='conv_5'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3), padding='same', activation='ReLU', name='conv_6'),
    BatchNormalization(),
    MaxPool2D((2, 2), name='maxpool_3'),
    Dropout(0.5),
    Flatten(name='flat_layer'),
    Dense(256, activation='ReLU', name='dense_layer_1'),
    Dropout(0.6),
    BatchNormalization(),
    Dense(size_output, activation='softmax', name='output_layer')])     # Output layer uses softmax activation, which is a good choice for classification tasks

# Print a summary of the model's layers, including their number of neurons and weights (parameters) in each layer
cifar10_model.summary()

cifar10_model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics='accuracy')

cifar10_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

# Converted the model in ONNX format and saved it. Below is the code and then tested with 3 images. """
# Specify the input format, convert the trained network into ONNX format, and save it.
spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
_ = tf2onnx.convert.from_keras(cifar10_model,  input_signature=spec, output_path='cifar10_model.onnx')

def get_inference(img_path):
  # Creating cifar10 label
  label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

  img = cv2.imread(img_path)        # Always convert to grayscale
  img_resized = 255 - cv2.resize(img, (32, 32))           # Resize the input image to 28x28 pixels. 
  img_resized = np.expand_dims(np.array(img_resized, dtype=np.float32), axis=0)
  print(img_resized.shape)
  
  network_sess = ort.InferenceSession('cifar10_model.onnx', providers=['CUDAExecutionProvider'])    # Load the network. If you are using a CPU instead of a GPU, put 'CPUExecutionProvider' here instead.

  inputName = network_sess.get_inputs()[0].name       # Get the names of the input and output layers so that you cna refer to them in the call below
  outputName = network_sess.get_outputs()[0].name

  result = network_sess.run([outputName], {inputName: img_resized})     # Run the network on the input
  print(np.around(result[0]), label[np.argmax(result[0])])    # Print the result. Since we only have one output layer, only result[0] is relevant.

img_path = 'ship.jpg'      # Filename of image to be classified
get_inference(img_path)

img_path = 'airplane.jpg'      # Filename of image to be classified
get_inference(img_path)

img_path = 'frog.jpg'      # Filename of image to be classified
get_inference(img_path)