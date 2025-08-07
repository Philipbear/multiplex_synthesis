import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

microbe_class_dict = {
        'Coriobacteriia': 'Actinomycetota',
        'Actinomycetes': 'Actinomycetota',
        'Clostridia': 'Bacillota',
        'Bacilli': 'Bacillota',
        'Tissierellia': 'Bacillota',
        'Erysipelotrichia': 'Bacillota',
        'Negativicutes': 'Bacillota',
        'Deinococci': 'Deinococcota',
        'Cyanophyceae': 'Cyanobacteriota',
        'Alphaproteobacteria': 'Pseudomonadota',
        'Betaproteobacteria': 'Pseudomonadota',
        'Gammaproteobacteria': 'Pseudomonadota',
        'Myxococcia': 'Myxococcota',
        'Epsilonproteobacteria': 'Campylobacterota',
        'Spirochaetia': 'Spirochaetota',
        'Fusobacteriia': 'Fusobacteriota',
        'Chitinophagia': 'Bacteroidota',
        'Bacteroidia': 'Bacteroidota',
        'Flavobacteriia': 'Bacteroidota',
        'Sphingobacteriia': 'Bacteroidota',
        'Cytophagia': 'Bacteroidota',
        'Verrucomicrobiia': 'Verrucomicrobiota',
        'Ustilaginomycetes': 'Basidiomycota',
        'Malasseziomycetes': 'Basidiomycota',
        'Tremellomycetes': 'Basidiomycota',
        'Microbotryomycetes': 'Basidiomycota',
        'Agaricomycetes': 'Basidiomycota',
        'Eurotiomycetes': 'Ascomycota',
        'Leotiomycetes': 'Ascomycota',
        'Sordariomycetes': 'Ascomycota',
        'Dothideomycetes': 'Ascomycota',
        'Basidiobolomycetes': 'Zoopagomycota',
        'Mortierellomycetes': 'Mucoromycota',
        'Halobacteria': 'Euryarchaeota'
    }

def prepare_data():
    files = os.listdir('microbemasst_plots/source_data_microbemasst')
    files = [f for f in files if f.endswith('barplot.tsv')]
    
    all_df = pd.DataFrame()
    for file in files:
        df = pd.read_csv(os.path.join('microbemasst_plots/source_data_microbemasst', file), sep='\t')
        class_name = file.split('_')[0]
        
        # rename 'sum' to class_name
        df.rename(columns={'sum': class_name}, inplace=True)
        all_df = pd.merge(all_df, df, on='microbe', how='outer') if not all_df.empty else df
        
    # reorder rows by microbe name    
    microbe_class = [x for x in microbe_class_dict.keys()]

    # print if any microbe is missing in the microbe_class
    missing_microbes = set(all_df['microbe']) - set(microbe_class)
    if missing_microbes:
        print(f"Warning: The following microbes are missing in the microbe_class: {missing_microbes}")

    # reorder the DataFrame
    all_df['microbe'] = pd.Categorical(all_df['microbe'], categories=microbe_class, ordered=True)
    all_df = all_df.sort_values('microbe').reset_index(drop=True)
    
    # fill NaN values with 0 (other than 'microbe' column)
    for col in all_df.columns:
        if col != 'microbe':
            all_df[col] = all_df[col].fillna(0)
    
    # for all cols, make them first letter uppercase
    all_df.columns = [col.capitalize() for col in all_df.columns]
    
    # order columns alphabetically except 'Microbe'
    cols = ['Microbe'] + sorted([col for col in all_df.columns if col != 'Microbe'])
    all_df = all_df[cols]
    
    # save to tsv
    all_df.to_csv('microbemasst_plots/microbemass_barplot.tsv', sep='\t', index=False)


def create_barplots(input_file='microbemasst_plots/microbemass_barplot.tsv', 
                   output_path='microbemasst_plots/microbemass_barplot.svg',
                   figsize=(6.8, 2.4)):
    """
    Create vertical stacked bar plots with microbes on x-axis
    
    Args:
        input_file: Path to the TSV file with microbe data
        output_path: Path to save the plot
        figsize: Figure size tuple
    """
    # Load the data
    df = pd.read_csv(input_file, sep='\t')
    
    # Set microbe as index
    df.set_index('Microbe', inplace=True)
    
    # Set font
    plt.rcParams['font.family'] = 'Arial'
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors for each column
    colors = plt.cm.Set3(np.linspace(0, 1, len(df.columns)))
    
    # Create vertical stacked bar plot
    df.plot(kind='bar', stacked=True, ax=ax, color=colors, 
            width=0.8, edgecolor='white', linewidth=0.5)
    
    # Customize the plot
    ax.set_ylabel('Number of MASST\nspectral matches', fontsize=7, labelpad=5)
    ax.set_xlabel('', fontsize=7)
    # ax.set_title('Microbe Distribution Across Categories', fontsize=7, pad=20)
    
    # Customize legend
    ax.legend(bbox_to_anchor=(0.66, 0.37), loc='lower left', fontsize=6.5)

    # Customize ticks
    ax.tick_params(axis='x', which='major', pad=3, rotation=90, labelsize=6.5, length=2, width=0.5, color='0.2')
    ax.tick_params(axis='y', which='major', pad=1.5, labelsize=6, length=2, color='0.2')

    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, format='svg', bbox_inches='tight', transparent=True)
    # plt.show()
    plt.close()
    
    print(f'Bar plot saved to {output_path}')


def create_barplots_with_dendrogram(input_file='microbemasst_plots/microbemass_barplot.tsv', 
                                     output_path='microbemasst_plots/microbemass_barplot_dendro.svg',
                                     figsize=(7.5, 3.5)):
    """
    Create vertical stacked bar plots with a dendrogram-like structure at the bottom.
    
    Args:
        input_file: Path to the TSV file with microbe data.
        output_path: Path to save the plot.
        figsize: Figure size tuple.
    """
    # Load the data
    df = pd.read_csv(input_file, sep='\t')
    
    # Set microbe as index
    df.set_index('Microbe', inplace=True)
    
    # Set font
    plt.rcParams['font.family'] = 'Arial'
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors for each column
    colors = plt.cm.Set3(np.linspace(0, 1, len(df.columns)))
    
    # Create vertical stacked bar plot
    df.plot(kind='bar', stacked=True, ax=ax, color=colors, 
            width=0.8, edgecolor='white', linewidth=0.5)
    
    # --- Dendrogram drawing ---
    # Get class names (x-tick labels)
    class_names = df.index.tolist()
    
    # Group classes by phylum
    phylum_groups = {}
    for i, class_name in enumerate(class_names):
        phylum = microbe_class_dict.get(class_name)
        if phylum:
            if phylum not in phylum_groups:
                phylum_groups[phylum] = []
            phylum_groups[phylum].append(i)

    # Define y-positions for dendrogram lines in axis coordinates
    y_base = -1.1
    y_phylum_line = -1.2
    y_phylum_text = -1.23

    # Use the axis transform for y-coordinates
    transform = ax.get_xaxis_transform()

    # Draw lines for each phylum group
    for phylum, indices in phylum_groups.items():
        if not indices:
            continue
        
        start_idx, end_idx = min(indices), max(indices)
        
        # Horizontal line connecting classes within a phylum
        ax.plot([start_idx, end_idx], [y_base, y_base], color='black', lw=0.8, clip_on=False, transform=transform)
        
        # Vertical lines from each class to the horizontal line
        if len(indices) > 2:
            # For groups with more than 2 classes, only draw lines for the first and last class
            ax.plot([start_idx, start_idx], [y_base, y_base + 0.1], color='black', lw=0.8, clip_on=False, transform=transform)
            ax.plot([end_idx, end_idx], [y_base, y_base + 0.1], color='black', lw=0.8, clip_on=False, transform=transform)
        else:
            # For groups with 1 or 2 classes, draw lines for all classes
            for i in indices:
                ax.plot([i, i], [y_base, y_base + 0.1], color='black', lw=0.8, clip_on=False, transform=transform)
            
        # Phylum line and label
        mid_point = (start_idx + end_idx) / 2.0
        ax.plot([mid_point, mid_point], [y_base, y_phylum_line], color='black', lw=0.8, clip_on=False, transform=transform)
        ax.text(mid_point, y_phylum_text, phylum, ha='center', va='top', fontsize=7, rotation=90, transform=transform)

    # Add "Phylum" and "Class" labels
    ax.text(-1.5, -1.5, 'Phylum', ha='center', va='center', fontsize=7, weight='bold', transform=transform, rotation=90)
    ax.text(-1.5, -0.33, 'Class', ha='center', va='center', fontsize=7, weight='bold', transform=transform, rotation=90)

    # --- Plot customization ---
    ax.set_ylabel('Number of MASST\nspectral matches', fontsize=7, labelpad=5)
    ax.set_xlabel('', fontsize=7)
    
    # Customize legend
    ax.legend(bbox_to_anchor=(0.7, 1.1), loc='upper left', fontsize=6.5,
              ncol=1, labelspacing=0.3,
              handlelength=1.5, handletextpad=0.5)

    # Customize ticks
    ax.tick_params(axis='x', which='major', pad=1, rotation=90, labelsize=6.5, length=1.5, width=0.5, color='0.2')
    ax.tick_params(axis='y', which='major', pad=1.5, labelsize=6, length=2, color='0.2')

    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=0.3)
    ax.set_axisbelow(True)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # Adjust layout to prevent labels from being cut off
    plt.subplots_adjust(bottom=0.6)
    
    # Save the plot
    plt.savefig(output_path, format='svg', bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    
    print(f'Bar plot with dendrogram saved to {output_path}')
    

if __name__ == "__main__":
    # prepare_data()
    # create_barplots()
    create_barplots_with_dendrogram()
    