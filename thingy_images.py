############################################################################################################
"""
FEASIBILITY OF CONVOLUTIONAL NEURAL NETWORKS FOR PARTICLE IDENTIFICATION AT LARIAT

This is a project by Lukas Wystemp, second year physics summer intern at the University of Manchester.
The project is supervised by Elena Gramellini in collaboration with LArIAT at Fermilab.
Many thanks to Maria Gabriela Manuel Alves who produced the input data of the collection plane 
with a Monte-Carlo simulation.

The project is about using Convolutional Neural Networks (CNN) to classify particle interactions in a 
the liquid argon time projection chamber (LArTPC) of the LArIAT experiment.

It investigated:
- Can CNNs be applied to LArTPCs? 
- What is the optimal model configuration and structure with the highest accuracy?
- How can we investigate and improve our models for LArIAT? 

This script focuses mainly on the last point. 

This script contains various visualisation techniques for trained CNN models. 
 - Grad-CAM
 - Saliency maps
 - Smooth Grad Saliency Map
 - Integrated Gradients
 - Rectified Gradient
 - Guided Backpropagation
 - Guided Grad-CAM backprop

It also displays the input image with a bounding box around the object of interest and the model summary. 
"""
############################################################################################################

### Imports
import os
import numpy as np
import tensorflow.compat.v1 as tf
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
import pickle
from sklearn.utils import class_weight
import keras_cv
import focal_loss
import glob
import keras
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow.keras.backend as K
from keras import initializers
from tensorflow.keras.initializers import glorot_uniform
import tensorflow.keras.models
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from IPython.display import Image, display
from keras.models import Model
from tensorflow.python.framework import ops
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency


# Define the classes
classes = ['electron', 'kaon', 'muon', 'photon', 'pion', 'pion_zero']

# Set the working directory, change this to your own directory
os.chdir('/Users/lukaswystemp/Desktop')

# Init
K.clear_session()
tf.compat.v1.reset_default_graph()

### FUNCTIONS ###

### Prep ###
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


def load_image(img):
    """
    Loads the npy file and prepares it for the model

    Parameters:
    img (str): The path to the npy file. 

    Returns:
    numpy.ndarray: The reshaped image. 
    """
    try:
        img = np.load(img)
    except FileNotFoundError:
        print("Image doesn't exist")
        return None
    except Exception as e:
        print("An error occurred: ", e)
        sys.exit()
    img = img.reshape(1, 240, 146, 1)
    #img = np.random.rand(1, 240, 146, 1)
    img = normalise(img)
    return img


### Grad-CAM ###
def grad_cam(input_model, image, cls, layer_name):
    """
    GradCAM method for visualizing input saliency.

    :param input_model: model to compute the GradCAM for
    :param image: image to compute GradCAM for
    :param cls: class to compute GradCAM for
    :param layer_name: name of the layer to compute GradCAM for
    :return: heatmap of GradCAM
    """
    grad_model = Model(
        inputs=[input_model.inputs],
        outputs=[input_model.get_layer(layer_name).output, input_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image) # Conv_outputs: (1, height, widht, kernels), 
        # predictions: (1, num_classes)
        pred = predictions[:, cls] # (1,)

    grads = tape.gradient(pred, conv_outputs) # grad of prediction value wrt feature maps, how changes in 
    # the convolutional output affect the loss (1, height, width, kernels)
    guided_grads = tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads # only 
    # positive activations in feature maps and positive gradients, highlighting areas that positively affect 
    # the prediction of the class of interest
    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[0]

    weights = tf.reduce_mean(guided_grads, axis=(0, 1)) # average gradients over spatial dimensions for each 
    # kernel (kernels,)
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1) # weighted sum of feature maps 
    # (height, width)

    # ReLU by hand
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam


def call_grad_cam(model, img, class_idx, layer_name):
    grad_cam_img = grad_cam(model, img, class_idx, layer_name)
    grad_cam_img = np.expand_dims(grad_cam_img, axis=-1)
    grad_cam_img = tf.image.resize(grad_cam_img, [240, 146], method=tf.image.ResizeMethod.BILINEAR)

    plt.imshow(grad_cam_img, cmap='jet', alpha=1)
    plt.imshow(np.squeeze(img), cmap='gray', alpha = 0.6)
    plt.title(f'Grad-CAM\n Predicted Class: {classes[class_idx]}\n {layer_name}')
    plt.axis('off')
    plt.show()


### Saliency map ###
def compute_saliency(model, img, class_idx):
    """
    Computes the saliency map for the given image and model.

    Parameters:
    modeL (tf.keras.Model): The trained model.
    img (numpy.ndarray): The input image of shape (1, height, width, channels).
    class_idx (int): The index of the target class.

    Returns:
    numpy.ndarray: The saliency map of same dimension as img
    """
    img = tf.convert_to_tensor(img)
    # for tf 2 use tape
    with tf.GradientTape() as tape:
        tape.watch(img)
        preds = model(img)
        loss = preds[:, class_idx]

    grads = tape.gradient(preds, img) # gradients of loss wrt img (1, height, width, channels)
    dgrad_abs = tf.math.abs(grads)

    # Normalise and prep for visualisation
    dgrad_max_ = np.max(dgrad_abs, axis=-1)[0] # max grads across colour channels (height, width)
    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_show = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
    return grad_show


def compute_smoothgrad(model, img, class_idx, num_samples=50, noise_level=0.1):
    """
    Computes SmoothGrad saliency maps.
    
    Parameters:
    model (tf.keras.Model): The trained model.
    img (numpy.ndarray): The input image.
    class_idx (int): The index of the target class.
    num_samples (int): The number of noisy samples to average.
    noise_level (float): The standard deviation of the noise to be added.
    
    Returns:
    numpy.ndarray: The SmoothGrad saliency map.
    """
    stdev = noise_level * (np.max(img) - np.min(img))
    smooth_grad = np.zeros_like(img, dtype='float64')
    smooth_grad = np.squeeze(smooth_grad)

    
    for _ in range(num_samples):
        noise = np.random.normal(0, stdev, img.shape)
        noisy_img = img + noise
        grad = compute_saliency(model, noisy_img, class_idx)

        smooth_grad += grad
    
    smooth_grad /= num_samples
    #smooth_grad *= np.squeeze(img)
    return smooth_grad


def increase_contrast(img, gamma):
    """
    Increases the contrast of the given image.
    
    Parameters:
    img (numpy.ndarray): The input image.
    
    Returns:
    numpy.ndarray: The image with increased contrast.
    """
    # Normalize the image to [0, 1]
    img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-18)
    
    # Apply gamma correction for contrast enhancement
    #gamma = 0.1
    img_contrast_enhanced = np.power(img_normalized, gamma)
    
    # Scale back to [0, 255]
    img_contrast_enhanced = img_contrast_enhanced * 255
    return img


def call_smooth_grad_sal(model, img, class_idx):
    """
    Calls the SmoothGrad saliency map function and displays the result.
    """
    # Compute SmoothGrad saliency map
    smoothgrad_saliency = compute_smoothgrad(model, img, class_idx)
    # Increase contrast for visualization
    smoothgrad_saliency = increase_contrast(smoothgrad_saliency, 0.1)

    #img = np.squeeze(img)
    #smoothgrad_saliency = img * smoothgrad_saliency

    #plt.imshow(np.squeeze(img), cmap='gray')
    plt.imshow(smoothgrad_saliency, cmap='viridis', alpha=1)
    plt.title(f'SmoothGrad Saliency Map\n Predicted Class: {classes[class_idx]}')
    plt.axis('off')
    plt.show()



### Integrated Gradients ###
def compute_integrated_gradients(model, img, class_idx, baseline=None, steps=50):
    """
    Computes Integrated Gradients for the given image and model.

    This function is based on the implementation from the TensorFlow Lucid library: 
    https://www.tensorflow.org/tutorials/interpretability/integrated_gradients

    Use this as documentation and reference. Function calls integral_approximation, interpolate_images, 
    one_batch, and compupute_gradients all of which are from tf
    
    Parameters:
    model (tf.keras.Model): The trained model.
    img (numpy.ndarray): The input image.
    class_idx (int): The index of the target class.
    baseline (numpy.ndarray): The baseline image to start from.
    steps (int): The number of steps for the integration.
    
    Returns:
    numpy.ndarray: The Integrated Gradients saliency map.
    """

    batch_size = 32
    if baseline is None:
        baseline = np.zeros_like(img)
        #baseline = np.random.uniform(0, 1, img.shape)
    
    img = (img - np.min(img)) / (np.max(img) - np.min(img))


    img = tf.convert_to_tensor(img, dtype=tf.float32)
    baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)


    alphas = tf.cast(tf.linspace(start=0.0, stop=1.0, num=steps+1), dtype=tf.float32)
    gradient_batches = []

    for alpha in tf.range(0, len(alphas), batch_size):
        from_ = alpha
        to = tf.minimum(alpha + batch_size, len(alphas))
        alpha_batch = alphas[from_:to]

        gradient_batch = one_batch(baseline, img, alpha_batch, model, class_idx)
        gradient_batches.append(gradient_batch)

    total_gradients = tf.concat(gradient_batches, axis=0)

    total_gradients = tf.convert_to_tensor(total_gradients, dtype=tf.float32)

    avg_gradients = integral_approximation(total_gradients)
    integrated_gradients = (img - baseline) * avg_gradients

    
    integrated_gradients = np.abs(integrated_gradients)
    return integrated_gradients

def integral_approximation(gradients):
  # riemann_trapezoidal
  grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
  integrated_gradients = tf.math.reduce_mean(grads, axis=0)
  return integrated_gradients

@tf.function
def one_batch(baseline, image, alpha_batch, model, target_class_idx):

    #print(baseline.shape)
    #print(image.shape)
    #print(alpha_batch.shape)
    
    # Generate interpolated inputs between baseline and input.
    interpolated_path_input_batch = interpolate_images(baseline=baseline,
                                                       image=image,
                                                       alphas=alpha_batch)
    
    # Compute gradients between model outputs and interpolated inputs.
    gradient_batch = compute_gradients(images=interpolated_path_input_batch,
                                       target_class_idx=target_class_idx, model=model)
    return gradient_batch


def interpolate_images(baseline, image, alphas):
  baseline = tf.convert_to_tensor(baseline, dtype=tf.float32)
  image = tf.convert_to_tensor(image, dtype=tf.float32)
  alphas = tf.convert_to_tensor(alphas, dtype=tf.float32)


  alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
  baseline_x = tf.expand_dims(baseline, axis=0)
  input_x = tf.expand_dims(image, axis=0)
  delta = input_x - baseline_x
  images = baseline_x +  alphas_x * delta
  return images[0]

def compute_gradients(images, target_class_idx, model):
    with tf.GradientTape() as tape:
        tape.watch(images)
        logits = model(images)
        probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx]
  
    return tape.gradient(probs, images)


def call_int_grads(model, img, class_idx):
    # Compute Integrated Gradients saliency map
    integrated_grads_saliency = compute_integrated_gradients(model, img, class_idx)
    #integrated_grads_saliency = increase_contrast(integrated_grads_saliency, 0.01)
    # Display the Integrated Gradients saliency map
    plt.imshow(np.squeeze(img), cmap='gray')
    plt.imshow(np.squeeze(integrated_grads_saliency), cmap='jet', alpha=0.9)
    plt.title(f'Integrated Gradients Saliency Map\n Predicted Class: {classes[class_idx]}')
    plt.axis('off')
    plt.show()


### Guided Backpropagation ###
def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)


def modify_backprop(model, name):
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = model
    return new_model


def guided_backpropagation(img_tensor, model, activation_layer):
    img_tensor = tf.convert_to_tensor(img_tensor)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        # Forward pass through the model to get activations
        layer_output = model.get_layer(activation_layer).output

        # Create a model that outputs the activations of the specified layer
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=layer_output)
        intermediate_output = intermediate_model(img_tensor)

        # Compute the maximum activation
        max_output = tf.reduce_max(intermediate_output, axis=3)

    # Compute the gradient of the max output with respect to the input image tensor
    grads = tape.gradient(max_output, img_tensor)

    # Apply guided backpropagation (set negative gradients to zero)
    guided_grads = tf.where(grads > 0, grads, tf.zeros_like(grads))

    return guided_grads.numpy()


def guided_grad_cam(img_tensor, model, activation_layer, class_idx):
    """
    Perform Guided Grad-CAM to combine guided backpropagation with Grad-CAM.

    Parameters:
    img_tensor (tf.Tensor): Input image tensor.
    model (tf.keras.Model): Pre-trained Keras model.
    activation_layer (str): The name of the layer to visualize.
    class_idx (int): Index of the target class.

    Returns:
    np.ndarray: Guided Grad-CAM result.
    """
    guided_grads = guided_backpropagation(img_tensor, model, activation_layer)
    cam_heatmap = grad_cam(model, img_tensor, class_idx, activation_layer)
    #print("cam", cam_heatmap.shape)
    #print("grad", guided_grads.shape)

    cam_heatmap = np.expand_dims(cam_heatmap, axis=-1)
    #print(cam_heatmap.shape)

    cam_heatmap = tf.image.resize(cam_heatmap, [240, 146], method=tf.image.ResizeMethod.BILINEAR)
    #print(cam_heatmap.shape)


    guided_gradcam = guided_grads[0] * cam_heatmap

    return guided_grads[0]


def call_guided_backprop(model, img, class_idx, activation_layer):
    register_gradient()
    guided_model = modify_backprop(model, 'GuidedBackProp')
    gradient = guided_backpropagation(img, guided_model, activation_layer)
    gradient = np.squeeze(gradient)
    plt.figure()
    plt.title(f"Guided BackProp\n {activation_layer}")
    plt.imshow(gradient, cmap='viridis')
    plt.axis('off')
    plt.show()


    guided_gradcam = guided_grad_cam(img, guided_model, activation_layer, class_idx)
    plt.figure()
    plt.title(f"Guided Grad-CAM \n Predicted Class: {classes[class_idx]}\n {activation_layer}")
    plt.imshow(guided_gradcam, cmap='jet')
    plt.axis('off')
    plt.show()


### Visualisation of input ###
def draw_bounding_box(model, img, class_idx, layer_name=None):
    """
    Draws a bounding box around the object of interest in the input image. Uses Grad-CAM for orientation
    hence calls it again. need's layer_name to be specified. Deeper layers will result in more
    spread out bounding boxes. 

    Args:
        model (tf.keras.Model): The pre-trained model used for inference.
        img (numpy.ndarray): The input image.
        class_idx (int): The index of the class corresponding to the object of interest.
        layer_name (str): The name of the layer to visualize.

    Returns:
        None
    """
    def get_first_conv_layer_name(model):
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None

    if layer_name is None:
        layer_name = get_first_conv_layer_name(model)

    cam = grad_cam(model, img, class_idx, layer_name)
    cam = np.expand_dims(cam, axis=-1)
    cam = tf.image.resize(cam, [240, 146], method=tf.image.ResizeMethod.BILINEAR)
    
    threshold = 0.2 * np.max(cam)

    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')

    indices = np.argwhere(cam > threshold)
    if indices.size == 0:
        min_x = min_y = 0
        max_x = max_y = 0
    else:
        min_y, min_x, _ = np.min(indices, axis=0)
        max_y, max_x, _ = np.max(indices, axis=0)

    # Draw the bounding box
    img = np.squeeze(img)
    plt.imshow(img, cmap='jet', interpolation='nearest')
    plt.gca().add_patch(plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, fill=False, edgecolor='red', linewidth=1))
    plt.title("Input image")
    plt.text(min_x, max_y, f"{classes[class_idx]} candidate", color='red', fontsize=9, ha='left', va='top')
    plt.axis('off')
    
    plt.show()


### RectGrad ###
def sal(model, img, class_idx):
    replace2linear = ReplaceToLinear()

    score = CategoricalScore([class_idx])
    saliency = Saliency(model, model_modifier=replace2linear, clone=True)
    saliency_map = saliency(score, img, smooth_samples=20, smooth_noise=0.2)

    plt.imshow(saliency_map[0], cmap='viridis')
    plt.title(f"SmoothGrad Saliency Map \n Predicted Class: {classes[class_idx]}")
    plt.axis('off')
    plt.show()


def compute_saliency_for_rectgrad(model, img, class_idx):
    with tf.GradientTape() as tape:
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        tape.watch(img)
        predictions = model(img)
        loss = predictions[:, class_idx]
    gradients = tape.gradient(loss, img)
    return gradients.numpy()

def apply_rectgrad(saliency_map, threshold_level):
    threshold = np.percentile(saliency_map, 100 - threshold_level)
    rectgrad_map = np.where(saliency_map > threshold, saliency_map, 0)
    return rectgrad_map

def compute_rectgrad(model, img, class_idx, num_samples=50, noise_level=0.1, threshold_level=99):
    """
    Computes SmoothGrad saliency maps with RectGrad applied.
    
    Parameters:
    model (tf.keras.Model): The trained model.
    img (numpy.ndarray): The input image.
    class_idx (int): The index of the target class.
    num_samples (int): The number of noisy samples to average.
    noise_level (float): The standard deviation of the noise to be added.
    threshold_level (float): The percentile for thresholding in RectGrad.
    
    Returns:
    numpy.ndarray: The SmoothGrad saliency map with RectGrad applied.
    """

    stdev = noise_level * (np.max(img) - np.min(img))
    rect_grad = np.zeros_like(img, dtype='float64')
    #smooth_grad = np.squeeze(smooth_grad)

    for _ in range(num_samples):
        noise = np.random.normal(0, stdev, img.shape)
        noisy_img = img + noise
        grad = compute_saliency_for_rectgrad(model, noisy_img, class_idx)
        grad = apply_rectgrad(grad, threshold_level)
        rect_grad += grad
        #smooth_grad = apply_rectgrad(smooth_grad, threshold_level)
        
    rect_grad /= num_samples
    #smooth_grad = apply_rectgrad(smooth_grad, threshold_level)

    return rect_grad


def compute_saliency_with_rectgrad(model, img, class_idx):
    #build model without softmax
    model = keras.Model(inputs=[model.input], outputs=[model.layers[-2].output])

    g = tf.Graph()
    with g.as_default():
        with g.gradient_override_map({"Relu": "RectifiedRelu"}):
            with tf.GradientTape() as tape:
                img = tf.convert_to_tensor(img, dtype=tf.float32)
                tape.watch(img)
                predictions = model(img)
                loss = predictions[:, class_idx]
            gradients = tape.gradient(loss, img)
    return gradients.numpy()


def call_grad_rectgrad(model, img, class_idx):
    smoothgrad_saliency = compute_rectgrad(model, img, class_idx)
    plt.imshow(np.abs(smoothgrad_saliency[0]), cmap='jet', alpha = 1)
    plt.title(f'SmoothGrad Saliency Map\n with Rectified Gradient added\n Predicted Class: {classes[class_idx]} [1]')
    plt.axis('off')
    #plt.colorbar()
    plt.show()


### Main ###
def main():
    # Load model
    model = tf.keras.models.load_model('/Users/lukaswystemp/Desktop/k_fold_model_2.h5')
    model.trainable = False
    
    # Get overview of model  
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    Image('model.png')
    #display(Image('model.png'))
    model.summary()

    # import image
    img = '/Users/lukaswystemp/Documents/University/UoM_Internship/npData/electron_111.npy'  
    img = load_image(img)
    if img is None:
        sys.exit()

    # Predict class, 
    #labels = ['electron', 'kaon', 'michel', 'muon', 'photon', 'pion', 'pion_zero']
    print("0: electron, 1: kaon, 2: muon, 3: photon, 4: pion, 5: pion_zero")
    class_idx = np.argmax(model.predict(img))
    print(class_idx)


    # visualise functions
    # Input
    draw_bounding_box(model, img, class_idx)

    # layer name for grad-cam
    layer_name = 'conv2d_5'

    call_grad_cam(model, img, class_idx, layer_name)

    # SmoothGrad Saliency map
    sal(model, img, class_idx)

    # SmoothGrad Saliency map with integrated gradients
    call_int_grads(model, img, class_idx)

    # Legacy
    #grad = compute_saliency_with_rectgrad(model, img, class_idx)

    # Rectified Gradient
    call_grad_rectgrad(model, img, class_idx)

    # Guided Backpropagation
    activation_layer = 'conv2d_2'
    #call_guided_backprop(model, img, class_idx, activation_layer)

if __name__ == '__main__':
    main()