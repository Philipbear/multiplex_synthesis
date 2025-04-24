import pandas as pd
import pickle


def gen_data():
    df = pd.read_csv('../../raw_data/all_ms2_df.tsv', sep='\t', low_memory=False)
    # df = pd.read_csv('../../raw_data/Old_results_without_library_results_combined.tsv', sep='\t', low_memory=False)

    print('unique cmpd inchis:', len(df['INCHI'].unique()))
    print('unique cmpd smiles:', len(df['SMILES'].unique()))

    # dereplicate by inchis
    df = df.drop_duplicates(subset=['INCHI'], keep='first')

    df['formula'] = df['SMILES'].apply(smiles_to_formula)
    df = df[df['formula'].notnull()].reset_index(drop=True)

    df['mono_mass'] = df['formula'].apply(calc_monoisotopic_mass)
    df = df[df['mono_mass'].notnull()].reset_index(drop=True)

    mono_mass_ls = df['mono_mass'].tolist()

    # save
    with open('mono_mass_ls.pkl', 'wb') as f:
        pickle.dump(mono_mass_ls, f)


from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def smiles_to_formula(smiles):
    # Convert SMILES to a molecule object
    mol = Chem.MolFromSmiles(smiles)

    if mol:
        # Get the molecular formula
        formula = rdMolDescriptors.CalcMolFormula(mol)
        return formula
    else:
        return None


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


if __name__ == '__main__':
    gen_data()

