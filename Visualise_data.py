##############################################################################################################
"""
FEASIBILITY OF CONVOLUTIONAL NEURAL NETWORKS FOR PARTICLE IDENTIFICATION AT LARIAT

This is a project by Lukas Wystemp, second year physics summer intern at the University of Manchester.
The project is supervised by Elena Gramellini in collaboration with LArIAT at Fermilab.
Many thanks to Maria Gabriela Manuel Alves for providing the input data of the collection plane 
with a Monte-Carlo simulation.

This script visualises the data distribution per class in the dataset. The data is stored in .npy files
and the labels are electron, kaon, muon, photon, pion and pion_zero.
"""
import os
import glob
import matplotlib.pyplot as plt

data_dir = '/Users/lukaswystemp/Documents/University/UoM_Internship/npData_2'
labels = ['electron', 'kaon', 'muon', 'photon', 'pion', 'pion_zero']
files = sorted(glob.glob(data_dir + '/*.npy') )

electron = 0
kaon = 0
muon = 0
photon = 0
pion = 0
pion_zero = 0

for file in files:
    filename = str(os.path.basename(file))
    if filename.startswith('electron'):
        electron += 1
    elif filename.startswith('kaon'):
        kaon += 1
    elif filename.startswith('muon'):
        muon += 1
    elif filename.startswith('photon'):
        photon += 1
    elif filename.startswith('pion_zero'):
        pion_zero += 1
    elif filename.startswith('pion') and not filename.startswith('pion_zero'):
        pion += 1
    else:
        print('Unknown label')
        continue

plt.figure()
plt.grid(True)

plt.title('Data distribution per class')
plt.bar(labels, [electron, kaon, muon, photon, pion, pion_zero])
plt.show()

print("Electron: ", electron)
print("Kaon: ", kaon)
print("Muon: ", muon)
print("Photon: ", photon)
print("Pion: ", pion)
print("Pion Zero: ", pion_zero)