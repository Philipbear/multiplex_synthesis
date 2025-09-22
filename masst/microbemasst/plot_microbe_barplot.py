import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def sort_microbe_data_taxonomically(df):
    """
    Sort microbe data by taxonomic order, grouping related phyla and classes
    """
    # Define taxonomic order for phyla (broader groupings)
    phylum_order = [
        # Bacteria
        'Actinomycetota',
        'Bacillota', 
        'Bacteroidota',
        'Pseudomonadota',
        'Spirochaetota',
        'Verrucomicrobiota',
        'Cyanobacteriota',
        'Fusobacteriota',
        'Campylobacterota',
        'Myxococcota',
        'Bacillariophyta',
        
        # Archaea
        'Euryarchaeota',
        
        # Fungi
        'Ascomycota',
        'Basidiomycota',
        'Mucoromycota',
        'Zoopagomycota',
        
        # Eukaryotes
        'Chlorophyta',
        'Chordata'
    ]
    
    # Define class order within each phylum
    class_order_within_phylum = {
        'Actinomycetota': ['Actinomycetes', 'Coriobacteriia'],
        
        'Bacillota': ['Bacilli', 'Clostridia', 'Erysipelotrichia', 'Tissierellia', 'Negativicutes'],
        
        'Bacteroidota': ['Bacteroidia', 'Flavobacteriia', 'Cytophagia', 'Chitinophagia', 'Sphingobacteriia'],
        
        'Pseudomonadota': ['Gammaproteobacteria', 'Alphaproteobacteria', 'Betaproteobacteria', 'Epsilonproteobacteria'],
        
        'Spirochaetota': ['Spirochaetia'],
        
        'Verrucomicrobiota': ['Verrucomicrobiia'],
        
        'Cyanobacteriota': ['Cyanophyceae'],
        
        'Fusobacteriota': ['Fusobacteriia'],
        
        'Campylobacterota': ['Epsilonproteobacteria'],
        
        'Myxococcota': ['Myxococcia'],
        
        'Bacillariophyta': ['Coscinodiscophyceae'],
        
        'Euryarchaeota': ['Halobacteria', 'Methanobacteria'],
        
        'Ascomycota': ['Dothideomycetes', 'Sordariomycetes', 'Eurotiomycetes', 'Leotiomycetes', 
                       'Saccharomycetes', 'Pezizomycetes', 'Schizosaccharomycetes'],
        
        'Basidiomycota': ['Agaricomycetes', 'Ustilaginomycetes', 'Tremellomycetes', 'Microbotryomycetes',
                          'Dacrymycetes', 'Exobasidiomycetes', 'Malasseziomycetes'],
        
        'Mucoromycota': ['Mortierellomycetes'],
        
        'Zoopagomycota': ['Basidiobolomycetes'],
        
        'Chlorophyta': ['Trebouxiophyceae'],
        
        'Chordata': ['Mammalia']
    }
    
    # Create sorting keys
    def get_phylum_sort_key(phylum):
        if phylum in phylum_order:
            return phylum_order.index(phylum)
        else:
            return len(phylum_order)  # Put unknown phyla at the end
    
    def get_class_sort_key(row):
        phylum = row['phylum']
        class_name = row['class']
        
        if phylum in class_order_within_phylum:
            class_list = class_order_within_phylum[phylum]
            if class_name in class_list:
                return class_list.index(class_name)
            else:
                return len(class_list)  # Put unknown classes at the end of their phylum
        else:
            return 0
    
    # Add sorting columns
    df_sorted = df.copy()
    df_sorted['phylum_sort_key'] = df_sorted['phylum'].apply(get_phylum_sort_key)
    df_sorted['class_sort_key'] = df_sorted.apply(get_class_sort_key, axis=1)
    
    # Sort by phylum first, then by class within phylum
    df_sorted = df_sorted.sort_values(['phylum_sort_key', 'class_sort_key'])
    
    # Remove the sorting columns
    df_sorted = df_sorted.drop(['phylum_sort_key', 'class_sort_key'], axis=1)
    
    return df_sorted.reset_index(drop=True)


def create_class_barplots(input_file='masst/microbemasst/data/microbemasst_class_summary.tsv', 
                         output_path='masst/microbemasst/plots/microbemasst_class_barplot.svg',
                         figsize=(7.8, 3.5)):
    """
    Create vertical stacked bar plots with microbial classes on x-axis and phylum dendrogram
    
    Args:
        input_file: Path to the TSV file with class data
        output_path: Path to save the plot
        figsize: Figure size tuple
    """
    # Load the data
    df = pd.read_csv(input_file, sep='\t')
    
    # # Sort by phylum first, then by class
    # df = df.sort_values(['phylum', 'class'], ascending=[True, True])
    df = sort_microbe_data_taxonomically(df)
    
    # Set class as index
    df_plot = df.set_index('class')
    
    # Select only the pathway columns for plotting - REVERSED ORDER
    pathway_cols = ['Unclassified', 'Terpenoids', 'Shikimates and phenylpropanoids', 
                   'Fatty acids', 'Amino acids and peptides', 'Alkaloids']
    df_plot = df_plot[pathway_cols]
    
    # Set font
    plt.rcParams['font.family'] = 'Arial'
    # font size
    plt.rcParams['font.size'] = 7
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate colors for each pathway
    colors = plt.cm.Set3(np.linspace(0, 1, len(pathway_cols)))
    colors = colors[::-1]  # Reverse colors to match order in legend
    
    # Create vertical stacked bar plot
    df_plot.plot(kind='bar', stacked=True, ax=ax, color=colors, 
                width=0.8, edgecolor='white', linewidth=0.5)
    
    # --- Dendrogram drawing ---
    # Get class names and their phylums
    class_names = df['class'].tolist()
    phylums = df['phylum'].tolist()
    
    # Group classes by phylum
    phylum_groups = {}
    for i, (class_name, phylum) in enumerate(zip(class_names, phylums)):
        if phylum not in phylum_groups:
            phylum_groups[phylum] = []
        phylum_groups[phylum].append(i)
    
    # Define y-positions for dendrogram lines in axis coordinates (matching microbemasst_barplot)
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
            ax.plot([start_idx, start_idx], [y_base, y_base + 0.05], color='black', lw=0.8, clip_on=False, transform=transform)
            ax.plot([end_idx, end_idx], [y_base, y_base + 0.05], color='black', lw=0.8, clip_on=False, transform=transform)
        else:
            # For groups with 1 or 2 classes, draw lines for all classes
            for i in indices:
                ax.plot([i, i], [y_base, y_base + 0.1], color='black', lw=0.8, clip_on=False, transform=transform)
                
        # Phylum line and label
        mid_point = (start_idx + end_idx) / 2.0
        ax.plot([mid_point, mid_point], [y_base, y_phylum_line], color='black', lw=0.8, clip_on=False, transform=transform)
        ax.text(mid_point, y_phylum_text, phylum, ha='center', va='top', fontsize=7, rotation=90, transform=transform)
    
    # Add "Phylum" and "Class" labels (matching microbemasst_barplot)
    ax.text(-1.5, -1.5, 'Phylum', ha='center', va='center', fontsize=7, weight='bold', transform=transform, rotation=90)
    ax.text(-1.5, -0.33, 'Class', ha='center', va='center', fontsize=7, weight='bold', transform=transform, rotation=90)
    
    # --- Plot customization ---
    ax.set_ylabel('Number of microbeMASST\nspectral matches', fontsize=7, labelpad=5)
    ax.set_xlabel('', fontsize=7)
    
    # Customize legend (matching microbemasst_barplot)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.65, 1.1), loc='upper left', fontsize=6.5,
              ncol=1, labelspacing=0.3,
              handlelength=1.5, handletextpad=0.5)
    
    # Customize ticks (matching microbemasst_barplot)
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
    
    # Adjust layout to prevent labels from being cut off (matching microbemasst_barplot)
    plt.subplots_adjust(bottom=0.6)
    
    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, format='svg', bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f'Class bar plot with dendrogram saved to {output_path}')
    
    # Print summary statistics
    print(f'\nSummary:')
    print(f'Total classes: {len(df)}')
    print(f'Total phylums: {df["phylum"].nunique()}')
    print(f'Total entries: {df["total_entries"].sum():,}')
    
    print(f'\nTop 10 classes by total entries:')
    top_classes = df.nlargest(10, 'total_entries')[['class', 'phylum', 'total_entries']]
    for _, row in top_classes.iterrows():
        print(f'  {row["class"]} ({row["phylum"]}): {row["total_entries"]:,}')

if __name__ == "__main__":
    
    create_class_barplots()
    