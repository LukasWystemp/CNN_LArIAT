# Feasability of CNNs for particle identification at LArIAT

This was my summer project in second year physics at the University of Manchester in collaboration with the Fermilab liquid argon in a testbeam (LArIAT) detector. 

The question it aimed to address was:
- Can CNNs which have been developed for information dense photographs with various textures, colours, etc. be applied to LArTPC data which is very information sparse. Can high accuracies be achieved?
- What is the optimal model configuration and structure with the highest accuracy?
- Utilising gradient based localisation to visually represent the neural network, can I gain insights into its classification mechanisms to maximise the model’s performance, ensure applicability at LArIAT and other LArTPCs, and improve resilience to noise?

# File overview

| Project files | What they do |
| --------------| -------------|
| make_train_model.py | Makes the model and trains it in the same file |
| Visualise_data_distribution.py | Overview of LArIAT data distribution |
| original_model.h5 | Trained model weights, this model experienced issues with overfitting: 75% accuracy|
| new_model.h5 | Trained model weights, overfitting is mitigated at the cost of some accuracy: 64% accuracy |
| model_visualisation.py | Different gradient-based visualisation techniques |


# Gradient based CNN visualisation techniques
'model_visualisation.py' employs several gradient-based visualisation:
| Method | What it does |
| -------| -------------|
| Input | Input image, uses Grad-CAM to plot a boundary box |
| Grad-CAM | Weight the 2D activations by the average gradient |
| Saliency Map | highlights input regions contributing most to the final output |
| SmoothGrad Saliency Map | Adds noise to Saliency Map to reduce noise |
| Integrated Gradients Saliency Map | Integrates over the contribution of the features from a baseline image to the input image |
| Rectified Gradients Smoothgrad Saliency Map | removes irrelevant features with positive pre- activation values passing through ReLU functions |
| Guided Backpropagation | pixels that are detected by the neurons, discarding the pixels that suppress the neurons |
| Guided Grad-CAM |combiens GBP and Grad-CAM |

# Visual Examples
These examples use real test data from a Monte-Carlo simulation of the collection plane at LArIAT. 

![image](https://github.com/user-attachments/assets/28f2fc43-3a20-46b8-bfc6-bbafff0b3e6e)
![image](https://github.com/user-attachments/assets/1c881eac-e911-4607-aaa0-f768f0894a77)
![image](https://github.com/user-attachments/assets/409315d0-1cc1-4352-a10b-000008ec0ee7)
![image](https://github.com/user-attachments/assets/e6f5e6b5-c4b1-4b26-9688-0c45f44fdfe5)
![image](https://github.com/user-attachments/assets/f3b364f3-064e-43fe-bc36-c5d302a9fd58)
![image](https://github.com/user-attachments/assets/650ef56f-d4a2-4a56-97c4-c00fb6d2a74a)
![image](https://github.com/user-attachments/assets/47e089f8-8093-4d60-a0e8-dfa999ed2b73)


# Use this code
1. Download `model_visualisation.py` and `new_model.h5` or your own model but it should have a similar architecture to `make_train_model.py`
2. In `model_visualisation.py` change `model = tf.keras.models.load_model()` and `img = ''` to a str with the file location of the model (.h5) and the input image (.npy). The image should be a two dimensional array, e.g. (240, 146). If you want to use different length arrays you have to adapt the functions `load_image()`, `grad_cam()`, `guided_grad_cam()`, `draw_bounding_box()` as they rely on knowledge about the input array dimension
3. Change `layer_name = 'conv2d_2'` and `activation_layer = 'conv2d_2'` to whatever layer you want to investigate according to `model.summary()`. 
4. Change class names if you are investigating your own model

Do not use a newer version of tensorflow than 2.13 as this will lead to errors when importing `new_model.h5`. 


# References
1. Ioffe, S. and Szegedy, C. (2015) Batch Normalization: Accelerating Deep Network Training
by Reducing Internal Covariate Shift [Preprint]. Available at:https://arxiv.org/pdf/1502.03167.
2. Zhao, L. and Zhang, Z. (2024) A improved pooling method for Convolutional Neural Net- works, Nature News. Available at: https://www.nature.com/articles/s41598-024-51258-
6: :text=Max%20pooling%20is%20a%20commonly,each%20small%20window%20or%20region. (Accessed: 03 August 2024).
3. Bohra, Y. (2024) The challenge of vanishing/exploding gradients in deep neural networks, Analytics Vidhya. Available at: https://www.analyticsvidhya.com/blog/2021/06/the-challenge- of-vanishing-exploding-gradients-in-deep-neural-networks/ (Accessed: 03 August 2024).
4. P.A. Zyla et al. (2020), Particle Data Group, Prog. Theor. Exp. Phys. 2020, 083C01
5. Selvaraju, R.R. et al. (2017) ‘Grad-CAM: Visual explanations from deep networks via gradient-based localization’, 2017 IEEE International Conference on Computer Vision (ICCV) [Preprint]. doi:10.1109/iccv.2017.74
6. Mokuwe, M., Burke, M. and Bosman, A.S. (2020) Black-box saliency map generation using Bayesian optimisation, arXiv.org. Available at: https://arxiv.org/abs/2001.11366 (Accessed: 31 July 2024).
7. Sundararajan, M., Yan, Q. and Taly, A. (2013) Axiomatic Attribution for Deep Networks. Available at: https://arxiv.org/pdf/1703.01365 (Accessed: 31 July 2024).
8. Kim, B. et al. (2019) ‘Why are saliency maps noisy? Cause of and solution to noisy saliency maps’, 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW) [Preprint]. doi:10.1109/iccvw.2019.00510



 
