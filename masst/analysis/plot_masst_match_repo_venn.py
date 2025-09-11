import pandas as pd
import os
import matplotlib.pyplot as plt
# from matplotlib_venn import venn3, venn3_circles
from venn import venn as vennN
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


def print_venn_plot_data():
    """
    Load the generated TSV files and print data needed for Venn plots
    """
    print("\n=== Venn Plot Data ===")
    
    # Load USI repo presence matrix
    print("Loading USI repo presence data...")
    usi_presence = pd.read_csv("data/usi_repo_presence_matrix.tsv", sep='\t', index_col=0)
    
    usi_repos = list(usi_presence.columns)
    print(f"\nRepositories found in USI data: {usi_repos}")
    
    print("\n=== USI Venn Plot Data ===")
    usi_sets = {}
    for repo in usi_repos:
        usi_sets[repo] = set(usi_presence[usi_presence[repo] == 1].index)
        print(f"Repository {repo}: {len(usi_sets[repo]):,} USIs")
    
    if len(usi_repos) >= 2:
        print("\nUSI pairwise overlaps:")
        for i, r1 in enumerate(usi_repos):
            for r2 in usi_repos[i+1:]:
                print(f"{r1} ∩ {r2}: {len(usi_sets[r1] & usi_sets[r2]):,}")
    if len(usi_repos) >= 3:
        print("\nUSI triple overlaps:")
        from itertools import combinations
        for r1, r2, r3 in combinations(usi_repos, 3):
            print(f"{r1} ∩ {r2} ∩ {r3}: {len(usi_sets[r1] & usi_sets[r2] & usi_sets[r3]):,}")
    if len(usi_repos) >= 4:
        first4 = usi_repos[:4]
        quad = len(usi_sets[first4[0]] & usi_sets[first4[1]] & usi_sets[first4[2]] & usi_sets[first4[3]])
        print(f"\n4-way overlap ({' ∩ '.join(first4)}): {quad:,}")
    
    return {
        'usi_sets': usi_sets,
        'usi_repos': usi_repos,
    }

def plot_venn_plots():
    """
    Generate Venn (4 sets) plots for USI distribution across repositories
    """    
    print("\n=== Generating Venn Plots ===")
    venn_data = print_venn_plot_data()
    usi_sets = venn_data['usi_sets']
    usi_repos = venn_data['usi_repos']
    
    plt.rcParams['font.family'] = 'Arial'
    
    print("\nGenerating USI Venn plots...")
    os.makedirs("plots", exist_ok=True)
    
    # Optional mapping (edit as needed)
    label_map = {
        'MS': 'GNPS/MassIVE',
        'MT': 'MetaboLights',
        'NO': 'NORMAN',
        'ST': 'Metabolomics Workbench',
    }
    
    _create_venn_plot_4(
        usi_sets,
        usi_repos[:4],
        'USI',
        'plots/usi_venn_4way',
        set_label_map=label_map
    )


def _create_venn_plot_4(
    data_sets,
    repos,
    data_type,
    output_prefix,
    set_label_map=None,
    colors=None
):
    """
    Create a 4-set Venn/Euler-like diagram using the 'venn' library.

    Parameters
    ----------
    data_sets : dict[str, set]
        Mapping repo -> set of identifiers.
    repos : list[str]
        Order of repositories to plot (length 4).
    data_type : str
        Descriptor for title.
    output_prefix : str
        Output file path prefix (no extension).
    set_label_map : dict[str,str] | None
        Optional mapping from repo code to display label.
    colors : list[str] | None
        Optional list of 4 hex/rgb colors.
    """
    subset_dict = {repo: data_sets[repo] for repo in repos}

    display_labels = [
        set_label_map.get(r, r) if set_label_map else r
        for r in repos
    ]

    if colors is None:
        colors = ['#377eb8', '#e41a1c', '#4daf4a', '#984ea3']  # ColorBrewer-like

    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Draw venn; fontsize controls numeric subset labels
    vennN(
        subset_dict,
        ax=ax,
        fontsize=6,
        cmap=cmap,
        alpha=0.1
    )

    # Adjust set label font sizes (they are regular Text objects with those names)
    for txt in ax.texts:
        if txt.get_text() in display_labels:
            txt.set_fontsize(7)

    # Build legend (one entry per set)
    handles = [
        mpatches.Patch(facecolor=(*mcolors.to_rgba(colors[i])[:3], 0.2), edgecolor=colors[i], label=display_labels[i], linewidth=0.7)
        for i in range(len(repos))
    ]
    ax.legend(
        handles=handles,
        fontsize=7,
        frameon=False,
        loc='center left',
        bbox_to_anchor=(0.95, 0.45)
    )

    # ax.set_title(f"{data_type} distribution across 4 repositories", fontsize=8)
    plt.tight_layout()

    fname_svg = f"{output_prefix}.svg"
    plt.savefig(fname_svg, format='svg', bbox_inches='tight', transparent=True)
    plt.close()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    plot_venn_plots()