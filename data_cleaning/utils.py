
from rdkit import Chem

def smiles_to_inchikey(smiles):
    # Convert SMILES to a molecule object
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        # Get the InChIKey
        inchikey = Chem.MolToInchiKey(mol)
        return inchikey
    else:
        return None


import requests
import urllib.parse


def smiles_to_npclassifier(smiles):
    """
    convert smiles to npclassifier
    """
    smiles = urllib.parse.quote(smiles, safe='')
    url = f'https://npclassifier.gnps2.org/classify?smiles={smiles}'
    try:
        response = requests.get(url, timeout=5)
        _class_info = response.json()
        return _class_info['class_results'], _class_info['superclass_results'], _class_info['pathway_results']
    except:
        return None, None, None