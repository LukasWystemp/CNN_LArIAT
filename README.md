# Feasability of CNNs for particle identification at LArIAT

This was my summer project in second year physics at the University of Manchester in collaboration with the Fermilab liquid argon in a testbeam (LArIAT) detector. 

The question it aimed to address was:
- Can CNNs which have been developed for information dense photographs with various textures, colours, etc. be applied to LArTPC data which is very information sparse. Can high accuracies be achieved?
- What is the optimal model configuration and structure with the highest accuracy?
- Utilising gradient based localisation to visually represent the neural network, can I gain insights into its classification mechanisms to maximise the model’s performance, ensure applicability at LArIAT and other LArTPCs, and improve resilience to noise?

| Project files | What they do |
| --------------| -------------|
| make_train_model.py | Makes the model and trains it in the same file |
| Visualise_data_distribution.py | Overview of LArIAT data distribution |
| original_model.h5 | Trained model weights, this model experienced issues with overfitting: 75% accuracy|
| new_model.h5 | Trained model weights, overfitting is mitigated at the cost of some accuracy: 64% accuracy |
| model_visualisation.py | Different gradient-based visualisation techniques |


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





 
