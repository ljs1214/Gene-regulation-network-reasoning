import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
import numpy as np
from dataProcess import rebuiltDataset
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf
from net import TransformerAutoencoder
from train import trainModel
from recogNet import recogNet

trainList, targets, gene_pairs = rebuiltDataset('/Users/nianhua/Nutstore Files/ned/jupyter/DSproject/DREAM5_NetworkInference_GoldStandard_Network2 - S. aureus.txt','/Users/nianhua/Nutstore Files/ned/jupyter/DSproject/net2_expression_data.tsv')

negtiveTrain = []
for i in tqdm(range(len(targets))):
    if targets[i] == 0:
        negtiveTrain.append(trainList[i])
negtiveTrain = np.array(negtiveTrain)



history, model = trainModel(negtiveTrain)
history.save('model')
#model = keras.models.load_model('path/to/location')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

recogNet(model, trainList, targets)

