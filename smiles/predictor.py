import os


def predict(model_path, model_constructor, data, charset, latent_rep_size, out_file):
    model_path = os.path.expanduser(model_path)
    model = model_constructor(model_path, data.shape[1], data.shape[2], latent_rep_size)
    # TODO progress reporting
    results = model.predict(data)
    write_decoded_smiles(results, charset, out_file)


def write_decoded_smiles(data, charset, out_file):
    data_set = out_file.create_dataset('structure', (len(data),), dtype=('S' + str(len(data[0]))))
    for i in range(len(data)):
        smiles = ''
        for j in range(len(data[i])):
            smiles += decode_char(data[i][j], charset)
        data_set[i] = smiles.strip().encode('utf-8')


def decode_char(vector, charset):
    max_index = 0
    for i in range(len(vector)):
        if vector[i] > vector[max_index]:
            max_index = i
    return charset[max_index].decode('utf-8')
