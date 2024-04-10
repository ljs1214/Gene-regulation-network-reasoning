from net import TransformerAutoencoder
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from recogNet import recogNet

def trainModel(negtiveTrain, num_layers = 4,d_model = 128,num_heads = 8,dff = 512,maximum_position_encoding = 10000):
    transformer_autoencoder = TransformerAutoencoder(num_layers=num_layers, d_model=d_model,
                                                  num_heads=num_heads, dff=dff)

    x_train = negtiveTrain
    x_val = negtiveTrain
    transformer_autoencoder.compile(optimizer=Adam(1e-3), loss='mean_squared_error')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, mode='min', verbose=1)
    history = transformer_autoencoder.fit(x_train, x_train,
                                        batch_size=128,
                                        epochs=10,
                                        validation_data=(x_val, x_val),
                                        callbacks=[early_stopping, reduce_lr],
                                        verbose = 2)
    return history, transformer_autoencoder

    
