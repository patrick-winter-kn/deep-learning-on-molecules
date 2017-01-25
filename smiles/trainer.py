from keras.callbacks import Callback, ReduceLROnPlateau
import math
from tqdm import tqdm_notebook


def train(model_constructor, data, batch_size, epochs, latent_rep_size):
    model = model_constructor(data.shape[1], data.shape[2], latent_rep_size)
    steps = epochs * math.ceil(data.shape[0] / batch_size)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
    # model.fit(data, data, shuffle='batch', nb_epoch=epochs, batch_size=batch_size, callbacks=[ProgressBar(steps)])
    model.fit(data, data, shuffle='batch', nb_epoch=epochs, batch_size=batch_size)
    return model


class ProgressBar(Callback):

    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.current_step = 0
        self.progress = None

    def on_train_begin(self, logs=None):
        self.progress = tqdm_notebook(total=self.num_steps)

    def on_batch_end(self, batch, logs=None):
        self.current_step += 1
        self.progress.update(1)

    def on_train_end(self, logs={}):
        self.progress.close()
