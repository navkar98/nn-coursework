# Face Recognition Demo using CNN
# Trained and tested a CNN model to detect faces.

import tensorflow as tf
from tensorflow.keras import utils, optimizers, losses, callbacks, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten

import urllib
from io import BytesIO
import numpy as np
import pickle
import datetime
import matplotlib.pyplot as plt

"""Load face dataset"""

# Commented out IPython magic to ensure Python compatibility.
print('Reading face dataset...')

x = urllib.request.urlopen('https://www.cs.umb.edu/~marc/batch_00.npz').read()
x = np.load(BytesIO(x), allow_pickle=True)['arr_0']
print(f'Read {x.shape[0]} grayscale images of size {x.shape[1]}x{x.shape[2]}')

y = urllib.request.urlopen('https://www.cs.umb.edu/~marc/batch_00_labels.npz').read()
y = np.load(BytesIO(y), allow_pickle=True)['arr_0']

print('Done!\nWriting copy to drive for faster reloading...')

with open('x_data.pickle', 'wb') as f:
    pickle.dump(x, f)
with open('y_data.pickle', 'wb') as f:
    pickle.dump(y, f)

print('Done!\n')

"""Preparing to train and test dataset"""

# Below is the function you typically use for splitting a dataset into training and test sets:

# Print basic info about a set of exemplars by analyzing their desired outputs, i.e., the vector y of class indices
def print_exemplar_info(y):
    class_labels, label_counts = np.unique(y, return_counts=True)
    num_classes = len(class_labels)
    print(f'Total of {len(y)} images of {num_classes} individual people')
    print(f'Number of images per person ranges from {min(label_counts)} to {max(label_counts)}')

with open('x_data.pickle', 'rb') as f:
    x = pickle.load(f)

with open('y_data.pickle', 'rb') as f:
    y = pickle.load(f)

train_samples_per_person = 20
test_samples_per_person = 5
total_samples_per_person = train_samples_per_person + test_samples_per_person

print_exemplar_info(y)

print(f'\nRemoving data of people with less than {total_samples_per_person} images:')
print(f'({train_samples_per_person}) for training, ({test_samples_per_person}) for testing)')

class_labels, labels_inverse, label_counts = np.unique(y, return_inverse=True, return_counts=True)
selected_images = label_counts[labels_inverse] >= total_samples_per_person       # Create array indicating whether a person has enough images in the dataset
x, y = x[selected_images], y[selected_images]                                    # Remove the exemplars of people who fail this test
_, y, final_label_counts = np.unique(y, return_inverse=True, return_counts=True) # Make the remaining class indices consecutive, i.e., [0, 1, ..., len(y) - 1]
num_classes = len(final_label_counts)                                            # Number of remaining individual people (classes)
sample_type = np.zeros(len(y), dtype=int)                                        # Will hold result of splitting: 0: discard, 1: training set, 2: test set

for cl in range(num_classes):
    indices = np.nonzero(y == cl)[0]                                             # Create array of the indices of all exemplars in currret class cl
    train_subset = np.random.choice(indices, train_samples_per_person, replace=False)   # Randomly pick the desired number of training exemplars from this array
    remaining_samples = np.setdiff1d(indices, train_subset)                                     # From among the remaining exemplars, ...
    test_subset = np.random.choice(remaining_samples, test_samples_per_person, replace=False)   # randomly pick the desired number of test exemplars
    sample_type[train_subset] = 1
    sample_type[test_subset] = 2

x = (np.expand_dims(x, 3) / 255.0).astype(np.float32) # Remember: Conv2D layers need 3-dim. tensors of floats as input. Scaling values to interval [0, 1] is typical

print('\nGenerating training and test sets:')

x_train = x[sample_type == 1]
y_train = y[sample_type == 1]
x_test  = x[sample_type == 2]
y_test  = y[sample_type == 2]

del x, y    # Let us free some memory to avoid out-of-memory errors

print('\nTraining set:')
print_exemplar_info(y_train)

print('\nTest set:')
print_exemplar_info(y_test)

"""Building neural network"""

num_epochs = 30
batch_size = 200                        # Number of exemplars processed in each learning step
                                        # Larger batches -> faster training per epoch, more GPU memory used, may need slightly more epochs
size_input = (56, 56, 1)                # Shape of an individual input image (see comment above)
size_output = len(np.unique(y_train))   # Number of output-layer neurons must match number of classes

# Define the face recognition model (computational graph) using the sequential model
facerec_model = Sequential(
    [Input(shape=size_input, name='input_layer'),
    BatchNormalization(),   # Learn the distribution (in terms of 4 parameters) of inputs for each channel (here: 1 channel) and standardize it accordingly (mean = 0, std. dev. = 1)
    Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv_1'),   # Use 64 conv. filters of size 3x3 and shift them in 1-pixel steps
    Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', name='conv_2'),   # Use 32 conv. filters of size 3x3x64 because the input has 64 channels
    BatchNormalization(),
    MaxPool2D((2, 2), name='maxpool_1'),    # Max pooling with a window size of 2x2 pixels. Default stride equals window size, i.e., no window overlap
    Dropout(0.3),                           # Deactivate random subset of 30% of neurons in the previous layer in each learning step to avoid overfitting
    Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu', name='conv_3'),
    Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', name='conv_4'),
    BatchNormalization(),
    MaxPool2D((2, 2), name='maxpool_2'),
    Dropout(0.3),
    Flatten(name='flat_layer'),             # Reshape the input tensor provided the previous layer into a vector (1-dim. array) required by dense layers
    Dense(128, activation='relu', name='dense_layer_1'),    # A dense layer of 100 neurons ("dense" implies complete connections to all of its inputs)
    Dropout(0.3),
    BatchNormalization(),
    Dense(size_output, activation='softmax', name='output_layer')])  # The output layer uses softmax activation, which is typical for classification tasks

# Set up the model for training
Adam = optimizers.Adam()    # Adam optimizer uses both first and second loss-function derivatives for learning and is often more efficient than SGD

# Use categorical (one class per input) cross-entropy as loss function
# Its sparse version only requires a vector y of class indices, e.g. [2, 3, 0, ...], so our y_train and y_test are already in the correct form.
# Its regular version would require a matrix y with each row representing the desired activations for the entire output layer ("one-hot encoding").
# For the example [2, 3, 0, ...], y would then be [[0, 0, 1, 0, 0, ...], [0, 0, 0, 1, 0, ...], [1, 0, 0, 0, 0, ...], ...]
facerec_model.compile(loss=losses.sparse_categorical_crossentropy, optimizer=Adam, metrics='accuracy')

# Print a summary of the model's layers, including their number of neurons and weights (parameters) in each layer
facerec_model.summary()

"""Training the model and recording data using TensorBoard"""

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)   # Record data for every single training epoch

# Train the model using x_test and y_test as validation data to assess the performance of the network after every epoch
facerec_model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[tensorboard_callback], verbose=2)

"""Shows the results in TensorBoard"""

# Compute matrix of output vectors. Each row represents the output for one input image.
# Since we use softmax, the sum along each row must equal 1 (probability distribution).
y_test_pred = facerec_model.predict(x_test)

# For each output vector (row), find the column index of the "winner" with maximum activation.
# This column index is the class (= person) index that the network determined for the given input.
class_pred = np.argmax(y_test_pred, axis=1)

# The confidence in each classification equals the activation of the "winner" neuron.
confidence = np.max(y_test_pred, axis=1)

"""Inspecting individual classification results"""

test_index = 21         # Pick the index of any test exemplar
max_sample_outputs = 5  # Max. number of sample images shown for a given class

def show_class_exemplars(cl_index, fig_index):
    print(f'\nExamples of training images for class {cl_index}:')
    ex_indices = np.nonzero(y_train == cl_index)[0]     # Get indices of training examplars for this class
    if len(ex_indices) > max_sample_outputs:
        ex_indices = np.random.choice(ex_indices, max_sample_outputs, replace=False)
    montage = np.squeeze(np.concatenate(x_train[ex_indices], axis=1))
    plt.figure(fig_index)
    plt.imshow(montage, cmap='gray')
    plt.axis('off')
    plt.show()

actual_class = y_test[test_index]
print(f'Test exemplar {test_index} belongs to class {actual_class} and has the following input:')
plt.figure(1)
plt.imshow(np.squeeze(x_test[test_index]), cmap='gray')
plt.axis('off')
plt.show()

pred_class = class_pred[test_index]
print(f'\nNetwork prediction: class {pred_class} with confidence {confidence[test_index]:.3f}')

if pred_class == actual_class:
    print('Correct!')
    show_class_exemplars(actual_class, 2)
else:
    print('Incorrect!')
    show_class_exemplars(actual_class, 2)
    show_class_exemplars(pred_class, 3)