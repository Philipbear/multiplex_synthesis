from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def smiles_to_formula_inchikey(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        # Get the molecular formula
        formula = rdMolDescriptors.CalcMolFormula(mol)
        # Get the InChIKey
        inchikey = Chem.MolToInchiKey(mol)

        return formula, inchikey
    else:
        return None, None



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
        return (ensure_list(_class_info['class_results']),
                ensure_list(_class_info['superclass_results']),
                ensure_list(_class_info['pathway_results']))
    except:
        return None, None, None


def ensure_list(obj):
    if not obj:
        return None
    if not isinstance(obj, list):
        return [obj]
    return obj


from molmass import Formula


def calc_monoisotopic_mass(formula):
    """
    Calculate the exact mass for a given formula string
    """
    try:
        f = Formula(formula)
        return f.monoisotopic_mass
    except:
        return None
