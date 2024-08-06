import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.model_selection import KFold
import glob
import tensorflow.keras.backend as K
from keras import initializers
from tensorflow.keras.initializers import glorot_uniform
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


### LOAD DATA AND PREP MODEL ###

def read_data():
    data_dir = ''
    labels = ['electron', 'kaon', 'muon', 'photon', 'pion', 'pion_zero']

    # Load data
    X = []
    y = []

    files = sorted(glob.glob(data_dir + '/*.npy') )

    for file in files:
        filename = str(os.path.basename(file))
        if filename.startswith('electron'):
            label = 'electron'
        elif filename.startswith('kaon'):
            label = 'kaon'
        elif filename.startswith('muon'):
            label = 'muon'
        elif filename.startswith('photon'):
            label = 'photon'
        elif filename.startswith('pion_zero'):
            label = 'pion_zero'
        elif filename.startswith('pion') and not filename.startswith('pion_zero'):
            label = 'pion'
        else:
            print('Unknown label')
            continue

        data = np.load(data_dir + '/' + filename)
        X.append(data)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Reshape X to include the channel dimension, channel = 1 as greyscale
    X = X.reshape(X.shape[0], 240, 146, 1) # (number_of_samples, 240, 146, 1)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X, y_categorical, label_encoder


def construct_model():
    # Build model
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(240, 146, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        kernel_initializer=initializers.RandomNormal(stddev=0.01, seed=np.random.randint(1000)),
        bias_initializer=initializers.Zeros()),
        Dropout(0.5),

        Dense(32, activation='relu', 
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        kernel_initializer=initializers.RandomNormal(stddev=0.01, seed=np.random.randint(1000)),
        bias_initializer=initializers.Zeros()),
        Dropout(0.5),

        Dense(6, activation='softmax')
    ])

    reset_weights(model)

    # Compile the model with a smaller learning rate
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    return model


def normalise(image):
    """
    Normalizes the given npy files by scaling its pixel values between 0.1 and 1.1. 
    Can cause issues with CNN if not normalised

    Parameters:
    image (numpy.ndarray): The npy fileto be normalized. 

    Returns:
    numpy.ndarray: The normalized array.
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image)) + 0.1

def reset_weights(model):
    """
    Resets the weights of a given model and its submodels recursively.

    Parameters:
    model (tf.keras.Model): The model whose weights need to be reset.

    Returns:
    None
    """
    for layer in model.layers: 
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) # recursive call
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue # skip non-initializer attributes
            # find the corresponding variable, like kernel or bias
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))
            # older versions of tf/keras might not have `assign` method, use the following line in that case
            # var = initializer(var.shape, var.dtype)

def add_gaussian_noise(image, mean=0.0, stddev=0.01):
    """
    Add Gaussian noise to an image.
    :param image: Input image
    :param mean: Mean of the Gaussian noise
    :param stddev: Standard deviation of the Gaussian noise
    :return: Image with Gaussian noise added
    """
    noise = np.random.normal(mean, stddev, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0., 1.)  # Ensure values are within [0, 1]
    return noisy_image



X, y_categorical, label_encoder = read_data()
X = np.array([normalise(image) for image in X])


X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, shuffle = True) # shuffle = True??
X_train = np.array([add_gaussian_noise(image) for image in X_train])


data_augmentation = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

data_augmentation.fit(X_train)

model = construct_model()
model.fit(X_train, y_train, epochs=6, batch_size=32, verbose=1)


def test_model(X_test, y_test):
    y_pred = model.predict(X_test)

    y_pred = np.argmax(y_pred, axis=1)
    #y_test = np.argmax(y_test, axis=1)

    y_test = np.argmax(y_test, axis=1)


    average_accuracy = accuracy_score(y_test, y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    conf_matrix = conf_matrix * 100
    # Round the confusion matrix to 1 significant digit
    conf_matrix_rounded = np.round(conf_matrix, 0).astype(int)

    class_labels = ['electron', 'Kaon', 'Muon', 'Photon', 'Pion ±', 'Pion 0']

    print("----------------------------------------------------")
    print("Average accuracy: ", average_accuracy)


    print("----------------------------------------------------")
    print("Confusion matrix:")
    # Print header
    header = " " * 10  # Adjust spacing based on your terminal size or longest label
    for label in class_labels:
        header += f"{label:>10}"
    print(header)
    print("-" * len(header))  # Separator line

    # Print each row of the confusion matrix
    for i, row in enumerate(conf_matrix_rounded):
        row_str = f"{class_labels[i]:>10}"  # Start with the row label
        for val in row:
            row_str += f"{val:>10}"  # Append each value in the row
        print(row_str)

test_model(X_test, y_test)
model.save('k_fold_model_2.h5')
np.save('X_test_k_fold_2.npy', X_test)
np.save('y_test_k_fold_2.npy', y_test)
