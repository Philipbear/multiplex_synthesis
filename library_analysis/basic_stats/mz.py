import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def gen_data():
    df = pd.read_pickle('all_lib/data/ms2_all_df.pkl')

    print('unique cmpd inchis:', len(df['inchi'].unique()))
    print('unique cmpd smiles:', len(df['smiles'].unique()))

    # dereplicate by inchikey
    df = df.drop_duplicates(subset=['inchikey'], keep='first')

    df = df[df['exact_mass'].notnull()].reset_index(drop=True)

    mono_mass_ls = df['exact_mass'].tolist()

    # save
    with open('library_analysis/basic_stats/data/mono_mass_ls.pkl', 'wb') as f:
        pickle.dump(mono_mass_ls, f)


def plot_mz(fig_size):
    # Load the pickle file
    with open('library_analysis/basic_stats/data/mono_mass_ls.pkl', 'rb') as f:
        mono_mass_ls = pickle.load(f)

    # Convert to pandas Series for easier handling
    mono_mass_series = pd.Series(mono_mass_ls)

    # Create a figure with appropriate size
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=fig_size)

    # Create the KDE plot using seaborn
    sns.kdeplot(
        data=mono_mass_series,
        fill=False,
        color="#c7522a",
        alpha=1,
        linewidth=1
    )

    # Determine min and max values for setting x-axis limits
    min_mass = min(mono_mass_ls)
    max_mass = max(mono_mass_ls)
    
    print(f"Min mass: {min_mass}, Max mass: {max_mass}")

    # Round down min to nearest 250 and round up max to nearest 250
    min_tick = int(min_mass // 250) * 250
    max_tick = int(max_mass // 250 + 1) * 250

    # Create custom tick positions and labels at intervals of 250
    x_ticks = np.arange(min_tick, max_tick + 1, 250)

    # Set the custom tick positions and labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(x)) for x in x_ticks], fontsize=6)
    
    # Set the x-axis limits
    ax.set_xlim(0, 1500)

    # Customize the plot
    # plt.title("Mass distribution", fontsize=16)
    plt.xlabel("Monoisotopic mass", fontsize=8, labelpad=3)
    plt.ylabel("Density", fontsize=8, labelpad=3.5)

    ax.tick_params(axis='x', which='major', length=2, width=0.8, pad=1,
                   colors='0.4', labelsize=6.5)
    
    # Set y-ticks to be empty
    ax.tick_params(axis='y', which='major', length=1, width=0.8, pad=1,
                   colors='0.4', labelsize=6.5, labelleft=False)

    # plt.grid(True, linestyle='--', alpha=0.7)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color('0.4')

    # Tight layout to ensure everything fits nicely
    plt.tight_layout()

    # Save the plot as SVG
    plt.savefig("library_analysis/basic_stats/data/mono_mass_distribution.svg", format="svg", bbox_inches="tight", transparent=True)

    # Display the plot (optional)
    # plt.show()

    print(f"KDE plot saved as 'mono_mass_distribution.svg'")

    # Optional: Print some statistics about the data
    print("\nData Statistics:")
    print(f"Number of compounds: {len(mono_mass_ls)}")
    print(f"Min mass: {min(mono_mass_ls):.2f}")
    print(f"Max mass: {max(mono_mass_ls):.2f}")
    print(f"Mean mass: {np.mean(mono_mass_ls):.2f}")
    print(f"Median mass: {np.median(mono_mass_ls):.2f}")


if __name__ == '__main__':
    gen_data()
    plot_mz((2.5, 1.1))