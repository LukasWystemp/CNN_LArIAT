Feasability of CNNs for particle identification at LArIAT

This was a summer project by Lukas Wystemp at the University of Manchester in collaboration with the Fermilab Liquid argon in a testbeam (LArIAT) detector. 

The question it aimed to address was:**
- Can CNNs which have been developed for information dense photographs with various textures, colours, etc. be applied to LArTPC data which is very information sparse. Can high accuracies be achieved?
- What is the optimal model configuration and structure with the highest accuracy?
- Utilising gradient based localisation to visually represent the neural network, can I gain insights into its classification mechanisms to maximise the model’s performance, ensure applicability at LArIAT and other LArTPCs, and improve resilience to noise?

There are several files for this project:
- Visualise_data_distribution.py
- make_train_model.py
- model_visualisation.py
- new_model.h5
- original_model.h5

| Project files | What they do |
| --------------| -------------|
| make_train_model.py | Makes the model and trains it in the same file |
| Visualise_data_distribution.py | Overview of LArIAT data distribution |
| original_model.h5 | Trained model weights, this model experienced issues with overfitting: 75% accuracy|
| new_model.h5 | Trained model weights, overfitting is mitigated at the cost of some accuracy: 64% accuracy |
| model_visualisation.py | Different gradient-based visualisation techniques |


 
