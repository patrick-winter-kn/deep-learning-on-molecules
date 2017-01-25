from keras.callbacks import Callback, ReduceLROnPlateau
import math


def train(model_constructor, data, batch_size, epochs, latent_rep_size):
    model = model_constructor(data.shape[1], data.shape[2], latent_rep_size)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001)
    model.fit(data, data, shuffle='batch', nb_epoch=epochs, batch_size=batch_size, callbacks=[reduce_lr])
    return model
