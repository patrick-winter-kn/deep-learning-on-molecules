import os
import math
from . import models
from progressbar import ProgressBar


def predict(model_path, data, charset, latent_rep_size, out_file):
    model_path = os.path.expanduser(model_path)
    model = models.autoencoder(model_path, data.shape[1], data.shape[2], latent_rep_size)
    data_set = None
    with ProgressBar(max_value=len(data)) as progress:
        chunk_size = 100
        number_chunks = math.ceil(len(data)/chunk_size)
        for i in range(number_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(data))
            results = model.predict(data[start:end])
            if data_set is None:
                data_set = out_file.create_dataset('structure', (len(data),), dtype=('S' + str(results.shape[1])))
            write_decoded_smiles_chunk(results, charset, data_set, start)
            progress.update(end)


def write_decoded_smiles(data, charset, out_file):
    data_set = out_file.create_dataset('structure', (len(data),), dtype=('S' + str(data.shape[1])))
    with ProgressBar(max_value=len(data)) as progress:
        chunk_size = 100
        number_chunks = math.ceil(len(data)/chunk_size)
        for i in range(number_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(data))
            write_decoded_smiles_chunk(data[start:end], charset, data_set, start)
            progress.update(end)


def write_decoded_smiles_chunk(chunk_data, charset, out_data_set, offset):
    for i in range(len(chunk_data)):
        smiles = ''
        for j in range(len(chunk_data[i])):
            smiles += decode_char(chunk_data[i][j], charset)
        out_data_set[offset + i] = smiles.strip().encode('utf-8')


def decode_char(vector, charset):
    max_index = 0
    for i in range(len(vector)):
        if vector[i] > vector[max_index]:
            max_index = i
    return charset[max_index].decode('utf-8')


def encode(model_path, data, latent_rep_size, out_file):
    model_path = os.path.expanduser(model_path)
    model = models.encoder(model_path, data.shape[1], data.shape[2], latent_rep_size)
    data_set = out_file.create_dataset('latent_vectors', (len(data), latent_rep_size))
    with ProgressBar(max_value=len(data)) as progress:
        chunk_size = 100
        number_chunks = math.ceil(len(data)/chunk_size)
        for i in range(number_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(data))
            data_set[start:end] = model.predict(data[start:end])
            progress.update(end)


def decode(model_path, data, charset, max_length, out_file):
    model_path = os.path.expanduser(model_path)
    model = models.decoder(model_path, max_length, len(charset), data.shape[1])
    data_set = None
    with ProgressBar(max_value=len(data)) as progress:
        chunk_size = 100
        number_chunks = math.ceil(len(data)/chunk_size)
        for i in range(number_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(data))
            results = model.predict(data[start:end])
            if data_set is None:
                data_set = out_file.create_dataset('structure', (len(data),), dtype=('S' + str(results.shape[1])))
            write_decoded_smiles_chunk(results, charset, data_set, start)
            progress.update(end)
