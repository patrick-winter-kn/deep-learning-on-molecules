from rdkit import Chem
from rdkit.Chem import AllChem
import numpy
import h5py
from progressbar import ProgressBar


def write_fingerprints(smiles_file, smiles_data_set_name, fingerprints_file, fingerprints_data_set_name, fingerprint_size):
    smiles_h5 = h5py.File(smiles_file, 'r')
    fingerprints_h5 = h5py.File(fingerprints_file, 'w')
    smiles = smiles_h5[smiles_data_set_name]
    fingerprints = fingerprints_h5.create_dataset(fingerprints_data_set_name, (smiles.shape[0], fingerprint_size), 'I')
    generate(smiles, fingerprints, fingerprint_size)
    smiles_h5.close()
    fingerprints_h5.close()


def generate(smiles, fingerprints, size):
    print('Generating fingerprints')
    with ProgressBar(max_value=len(smiles)) as progress:
        for i in range(len(smiles)):
            mol = Chem.MolFromSmiles(smiles[i], sanitize=False)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, size)
            fingerprints[i] = numpy.array(fingerprint)
            progress.update(i+1)
