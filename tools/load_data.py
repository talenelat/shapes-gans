from __future__ import print_function, division 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def encode_labels(file_name='metadata.csv'):
    meta = pd.read_csv(f'./{file_name}', header=0)
    meta.columns = ['image', 'background', 'shape_type', 'shape_color']
    #print(meta.head())

    labels = []
    for entry in meta.index:
        # print(meta.iloc[entry].values.tolist())
        labels.append(meta.iloc[entry].values.tolist()[1] + ' ' + meta.iloc[entry].values.tolist()[2] + ' ' + meta.iloc[entry].values.tolist()[3])
        
    #print(labels)
    values = np.array(labels)

    # Integer Encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)

    # One Hot Encoding
    binary_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    binary_encoded = binary_encoder.fit_transform(integer_encoded)
    #print(binary_encoded)

    final = pd.DataFrame(binary_encoded)
    final.to_csv('y.csv')