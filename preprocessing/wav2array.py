import numpy as np
import sys
import os
import librosa
import matplotlib.pyplot as plt

hop_length_in = 512
n_bins_in = 252
bins_octaves_in = 36
win_step = 0.01
number_notes = 88
num_cep_def = 40
num_filt_def = 40
length_per_file = 4000000

source_List = sys.argv[1];
source_WAV = sys.argv[2];
source_Txt = sys.argv[3];
out_mat = sys.argv[4];

train2mat = []
labels2mat = []
count = 8

# Get the name of the list
source_list_split = source_List.split('.')
source_list_split = source_list_split[0].split('/')
list_name = source_list_split[-1]

# Open the list
file_List = open( source_List , "r")

# Iterate on every file
for filename in file_List:

    filename_split = filename.split('.')

    #### Constant QT extraction ####
    # Transform to raw data from wav. Get the sampling rate 2
    sampling_freq, stereo_vector = wavfile.read(source_WAV + filename_split[0] + '.wav')
    win_len = 512/float(sampling_freq)
    #plt.imshow( np.array(np.absolute(cqt_feat)))
    #plt.show()
    # Transform to mono
    mono_vector = np.mean(stereo_vector, axis = 1)
    # Extract cqt_features
    cqt_feat = np.absolute(librosa.cqt(mono_vector, sampling_freq, hop_length=hop_length_in,n_bins=n_bins_in,bins_per_octave=bins_octaves_in)).transpose()
    #### LABELING ####
    # Number of frames in the file
    number_Frames = np.max( cqt_feat.shape[0])
    # Aux_Vector of times
    vector_aux = np.arange(1, number_Frames + 1)*win_len
    # Binary labels - we need multiple labels at the same time to represent the chords
    labels = np.zeros((number_Frames, number_notes))


    # Open the align txt labels
    file = open( source_Txt + filename_split[0] + '.txt' , "r")
    #f = open(out_mat + filename_split[0] + 'label.lst','w')
    # Loop over all the lines 
    for line in file: 
        line_split = line.split()
        if (len(line_split) > 0):
            if line_split[0] == "OnsetTime":
                print("Preprocessing operations . . .")
            else:
    	    # Get the values from the text
                init_range, fin_range, pitch = float(line_split[0]), float(line_split[1]), int(line_split[2])
                # Pitch move to 0-87 range
                pitch = pitch - 21;
                # Get the range indexes
                index_min = np.where(vector_aux >= init_range)
                index_max = np.where(vector_aux - 0.01 > int((fin_range)*100)/float(100))
                labels[index_min[0][0]:index_max[0][0],pitch] = 1
    
    """
    file.close()
    plt.figure()
    plt.imshow( np.array(labels.transpose()),aspect='auto')
    plt.figure()
    plt.imshow( np.array(np.absolute(cqt_feat)), aspect='auto')
    plt.show()
    """
    while (len(train2mat) + len(cqt_feat)) >= length_per_file:
        size_to_add = length_per_file - len(train2mat)
        # Append to add to npz
        train2mat.extend(cqt_feat[0:size_to_add,:])
        # Append the labels 
        labels2mat.extend(labels[0:size_to_add,:])
        train2mat = np.array(train2mat)
        labels2mat = np.array(labels2mat)
        # Plotting stuff
        print(" Shape of CQT is " + str(train2mat.shape) + " - Saved in " + out_mat + list_name + '/' + str(count) + list_name)
        print(" Shape of Labels is " + str(labels2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(count) + list_name)
        np.save('{}_X'.format(out_mat + list_name + '/' + str(count) + list_name ), train2mat)
        np.save('{}_y'.format(out_mat + list_name + '/' + str(count) + list_name), labels2mat)
        count = count + 1;
        train2mat = []
        labels2mat = []
        cqt_feat = cqt_feat[size_to_add:,:]
        labels = labels[size_to_add:,:]
    if len(cqt_feat) == length_per_file:
        # Append to add to npz
        train2mat.extend(cqt_feat)
        # Append the labels 
        labels2mat.extend(labels)
        train2mat = np.array(train2mat)
        labels2mat = np.array(labels2mat)
        # Plotting stuff
        print(" Shape of CQT is " + str(train2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(count) + list_name)
        print(" Shape of Labels is " + str(labels2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(count) + list_name)
        np.save('{}_X'.format(out_mat + list_name + '/' + str(count) + list_name ), train2mat)
        np.save('{}_y'.format(out_mat + list_name + '/' + str(count) + list_name), labels2mat)
        count = count + 1;
        train2mat = []
        labels2mat = []
    elif len(cqt_feat) > 0:
        # Append to add to npz
        train2mat.extend(cqt_feat)
        # Append the labels 
        labels2mat.extend(labels)

train2mat = np.array(train2mat)
labels2mat = np.array(labels2mat)


plt.figure()
plt.imshow( np.array(labels2mat.transpose()),aspect='auto')
plt.colorbar()
plt.figure()
plt.imshow( np.array(train2mat.transpose()), aspect='auto')
plt.colorbar()
plt.show()


# Plotting stuff
print(" Shape of CQT is " + str(train2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(count) + list_name)
print(" Shape of Labels is " + str(labels2mat.shape)  + " - Saved in " + out_mat + list_name + '/' + str(count) + list_name)

np.save('{}_X'.format(out_mat + list_name + '/' + str(count) + list_name ), train2mat)
np.save('{}_y'.format(out_mat + list_name + '/' + str(count) + list_name), labels2mat)
