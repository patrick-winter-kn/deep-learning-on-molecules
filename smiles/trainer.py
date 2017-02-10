from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import math
import os


def train(model_path, model_constructor, data, batch_size, epochs, latent_rep_size):
    model_path = os.path.expanduser(model_path)
    model = model_constructor(model_path, data.shape[1], data.shape[2], latent_rep_size)
    checkpointer = ModelCheckpoint(filepath=model_path, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001)
    model.fit(data, data, nb_epoch=epochs, batch_size=batch_size, callbacks=[checkpointer, reduce_lr])
    return model
