import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

"""
['name', 'exact_mass', 'scan', 'smiles', 'inchi', 'usi', 'mz', 'adduct',
       'formula', 'inchikey', '2d_inchikey']
"""


def plot_isomers(filtered=False, save_path='library_analysis/basic_stats/data'):
    
    
    if filtered:
        df = pd.read_pickle('data_cleaning/cleaned_data/ms2_filtered_df.pkl')
    else:
        df = pd.read_pickle('data_cleaning/cleaned_data/ms2_all_df.pkl')
    
    df['isomer_count'] = df['name'].apply(lambda x: x.split('(known isomers: ')[1].split(';')[0])
    df['isomer_count'] = df['isomer_count'].astype(int)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Set style
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.linestyle': '--', 'grid.linewidth': 0.5})
    plt.figure(figsize=(4, 2))
    
    # Plot histogram
    ax = sns.histplot(data=df, x='isomer_count', binwidth=1, kde=False)
    
    # Add labels and title
    plt.xlabel('Number of isomers', fontsize=6, labelpad=2)
    plt.ylabel('Count', fontsize=6, labelpad=2)
    title = 'Distribution of Isomer Counts' + (' (Filtered)' if filtered else '')
    # plt.title(title)
    
    # tick parameters
    plt.tick_params(axis='x', which='major', length=1, width=0.8, pad=1,
                    colors='0.2', labelsize=5.5)
    plt.tick_params(axis='y', which='major', length=1, width=0.8, pad=1,
                    colors='0.2', labelsize=5.5)
    
    # x axis limits
    plt.xlim(0, 50)
    
    # Calculate percentage statistics
    total_spectra = len(df)
    zero_isomers = (df['isomer_count'] == 0).sum() / total_spectra * 100
    less_than_eq_3 = (df['isomer_count'] <= 3).sum() / total_spectra * 100
    less_than_eq_5 = (df['isomer_count'] <= 5).sum() / total_spectra * 100
    median_val = df['isomer_count'].median()
    
    stats_text = (f"Median: {median_val:.0f}\n"
                 f"0 isomers: {zero_isomers:.1f}%\n"
                 f"≤3 isomers: {less_than_eq_3:.1f}%\n"
                 f"≤5 isomers: {less_than_eq_5:.1f}%")
    
    plt.annotate(stats_text, xy=(0.7, 0.6), xycoords='axes fraction', fontsize=6,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure as SVG
    filename = f"isomer_distribution{'_filtered' if filtered else ''}.svg"
    save_file = os.path.join(save_path, filename)
    plt.savefig(save_file, format='svg', bbox_inches='tight')
    print(f"Figure saved to: {save_file}")
    
    # # save as PNG
    # save_file_png = os.path.join(save_path, filename.replace('.svg', '.png'))
    # plt.savefig(save_file_png, format='png', bbox_inches='tight', dpi=300)
    # print(f"Figure saved to: {save_file_png}")
    
    # Show plot
    plt.show()
    
    return df

if __name__ == "__main__":
    # Plot both versions
    plot_isomers(filtered=False)
    # plot_isomers(filtered=True)