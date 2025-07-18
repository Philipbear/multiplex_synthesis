import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def load_ms2_lib_metadata():
    """Load MS2 library metadata to get conjugate status"""
    try:
        df = pd.read_pickle('masst/umap/data/ms2_all_df_unique_usi.pkl')
        print(f"Loaded MS2 library metadata: {len(df)} entries")
        
        # Create USI to conjugate status mapping
        usi_to_is_conjugate = df.set_index('usi')['name'].apply(lambda x: '_' in str(x) if pd.notna(x) else False).to_dict()
        
        return usi_to_is_conjugate
    except FileNotFoundError:
        print("MS2 library metadata not found. Run get_ms2_lib_info.py first.")
        return {}


def load_usi_labeling_data(base_umap_dir):
    """Load USI labeling data created by prepare_data_for_umap.py"""
    try:
        labeling_path = f'{base_umap_dir}/data/usi_labeling_table.pkl'
        df = pd.read_pickle(labeling_path)
        print(f"Loaded USI labeling data: {len(df)} entries")
        
        return df
    except FileNotFoundError:
        print(f"USI labeling data not found at {labeling_path}. Run prepare_data_for_umap.py first.")
        return None


def get_rank_counts(usi_list, labeling_df, rank):
    """Get rank counts for USIs based on the specified taxonomic rank"""
    rank_counts = []
    unknown_count = 0
    
    # Create USI to rank count mapping
    usi_to_rank_count = labeling_df.set_index('usi')[f'{rank}_id_count'].to_dict()
    
    for usi in usi_list:
        if usi and usi in usi_to_rank_count:
            count = usi_to_rank_count[usi]
            
            # Handle potential None or NaN values
            if count is None or (isinstance(count, float) and pd.isna(count)):
                rank_counts.append(0)
                unknown_count += 1
            else:
                rank_counts.append(int(count))
        else:
            rank_counts.append(0)
            unknown_count += 1
    
    print(f"Found {rank} counts for {len(usi_list) - unknown_count}/{len(usi_list)} USIs")
    print(f"USIs with missing {rank} counts (set to 0): {unknown_count}")

    return rank_counts


def load_processed_data(output_dir, conjugates_only=False):
    """Load already processed data"""
    suffix = '_conjugates' if conjugates_only else ''
    
    # Load feature matrix
    feature_matrix = np.load(os.path.join(output_dir, f'umap_feature_matrix_clean{suffix}.npy'))
    
    # Load USI IDs
    with open(os.path.join(output_dir, f'usi_ids_clean{suffix}.txt'), 'r') as f:
        usi_ids = [line.strip() for line in f]
    
    # Load UMAP embedding
    embedding = np.load(os.path.join(output_dir, f'umap_embedding{suffix}.npy'))
    
    print(f"Loaded processed data{' (conjugates only)' if conjugates_only else ''}: {len(usi_ids)} USIs, {feature_matrix.shape[1]} features")
    print(f"UMAP embedding shape: {embedding.shape}")
    
    return feature_matrix, usi_ids, embedding


def plot_umap_by_rank_count(embedding, usi_ids, rank_counts, rank, conjugates_only=False, output_folder='plots'):
    """Create UMAP plots colored by number of taxonomic ranks"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert rank counts to categorical labels
    rank_labels = []
    for count in rank_counts:
        if count == 0:
            rank_labels.append('0')
        elif count <= 10:
            rank_labels.append(str(count))
        else:
            rank_labels.append('>10')
    
    # Get unique labels and their counts
    unique_labels = sorted(set(rank_labels), key=lambda x: (x == '0', x == '>10', int(x) if x.isdigit() else float('inf')))
    label_counts = pd.Series(rank_labels).value_counts()
    
    print(f"\n{rank.capitalize()} count distribution:")
    for label in unique_labels:
        if label in label_counts.index:
            count = label_counts[label]
            print(f"  {label} {rank}s: {count} USIs ({count/len(rank_labels)*100:.1f}%)")
    
    # Define FIXED category order for consistent legends across all ranks
    fixed_category_order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '>10']
    
    # Only include categories that are actually present in the data
    category_order = [cat for cat in fixed_category_order if cat in unique_labels]
    
    # Define plotting order - 0 first (background), then the rest
    plotting_order = ['0'] + [cat for cat in category_order if cat != '0']
    
    # Custom color palette - gradual progression from light to dark
    # 0 gets a light gray, 1-10 get a color gradient, >10 gets a distinct color
    base_colors = [
        '#f7f7f7',  # 0 - very light gray
        '#d9f0a3',  # 1 - light green
        '#addd8e',  # 2 - 
        '#78c679',  # 3 - 
        '#41ab5d',  # 4 - 
        '#238443',  # 5 - 
        '#006837',  # 6 - dark green
        '#004529',  # 7 - darker green
        '#003d26',  # 8 - 
        '#003322',  # 9 - 
        '#002a1e',  # 10 - darkest green
        '#8b0000'   # >10 - dark red
    ]
    
    # Create color mapping
    label_to_color = {}
    
    for i, label in enumerate(['0'] + [str(j) for j in range(1, 11)] + ['>10']):
        if label in unique_labels:
            if i < len(base_colors):
                label_to_color[label] = base_colors[i]
            else:
                label_to_color[label] = '#666666'  # fallback gray
    
    # Set font for all text elements
    plt.rcParams['font.family'] = 'Arial'
    
    # Create UMAP plot WITHOUT legend
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot in plotting order (0 first as background)
    scatter_handles = {}
    for label in plotting_order:
        if label in unique_labels:
            mask = np.array(rank_labels) == label
            if np.any(mask):
                alpha = 0.3 if label == '0' else 0.7
                size = 0.4 if label == '0' else 0.6
                handle = ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                                  c=[label_to_color[label]], label=f'{label}',
                                  alpha=alpha, s=size, edgecolor='none')
                scatter_handles[label] = handle

    ax.set_xlabel('UMAP 1', fontsize=6, labelpad=1.5, color='0.2')
    ax.set_ylabel('UMAP 2', fontsize=6, labelpad=1.5, color='0.2')

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Remove default spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # set color and linewidth for remaining spines
    ax.spines['left'].set_color('0.2')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('0.2')
    ax.spines['bottom'].set_linewidth(0.5)
    
    plt.tight_layout()
    
    # Save UMAP plot WITHOUT legend
    conjugate_suffix = '_conjugates' if conjugates_only else ''
    filename_base = f'umap_{rank}_by_{rank}_count{conjugate_suffix}'
    plt.savefig(os.path.join(output_folder, f'{filename_base}.png'), dpi=600, bbox_inches='tight', transparent=True)
    print(f"UMAP plot saved to {output_folder}/{filename_base}.png")
    plt.close()
    
    # Create separate legend figure
    fig_legend, ax_legend = plt.subplots(figsize=(1.2, 2))
    ax_legend.axis('off')
    
    # Create legend handles and labels in fixed order
    legend_handles = []
    legend_labels = []
    for label in category_order:  # Use the fixed order
        if label in scatter_handles:
            # Create a new scatter plot point for the legend
            handle = ax_legend.scatter([], [], c=[label_to_color[label]], 
                                     alpha=0.7, s=10, edgecolor='none')
            legend_handles.append(handle)
            legend_labels.append(f'{label}' if label != '>10' else label)
    
    # Create legend with title
    legend_title = f'{rank.capitalize()} count'
    legend = ax_legend.legend(legend_handles, legend_labels, 
                             loc='center', fontsize=8, markerscale=1.5, 
                             frameon=False, ncol=1,
                             title=legend_title, title_fontsize=8)
    legend.get_title().set_color('0.2')
    
    # Save legend separately
    plt.savefig(os.path.join(output_folder, f'{filename_base}_legend.svg'), format='svg',
                bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(output_folder, f'{filename_base}_legend.png'), dpi=600,
                bbox_inches='tight', transparent=True)
    print(f"Legend saved to {output_folder}/{filename_base}_legend.svg")
    plt.close()


def plot_rank_count_distribution(rank_counts, rank, conjugates_only=False, output_folder='plots'):
    """Create distribution plot showing how many USIs have each rank count"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Convert to numpy array for easier manipulation
    rank_counts_array = np.array(rank_counts)
    
    # Filter out zero counts for the distribution plot
    non_zero_counts = rank_counts_array[rank_counts_array > 0]
    
    if len(non_zero_counts) == 0:
        print(f"No USIs found with {rank} counts > 0")
        return
    
    # Create bins for histogram starting from 1
    max_count = non_zero_counts.max()
    if max_count <= 10:
        bins = list(range(1, max_count + 2))  # Bins from 1 to max_count+1
        labels = [str(i) for i in range(1, max_count + 1)]
    else:
        # Create bins for 1, 2, ..., 10, and >10
        bins = list(range(1, 12)) + [max_count + 1]  # Bins: [1, 2, ..., 11, max+1] -> 11 bins
        labels = [str(i) for i in range(1, 11)] + ['>10'] # Labels: '1', '2', ..., '10', '>10'
    
    # Create histogram data
    hist, bin_edges = np.histogram(non_zero_counts, bins=bins)
    
    # Set font for all text elements
    plt.rcParams['font.family'] = 'Helvetica'
    
    # Create the distribution plot
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Create bars with same color scheme as UMAP plots (skip the first color since we start from 1)
    colors = ['#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', 
              '#006837', '#004529', '#003d26', '#003322', '#002a1e', '#8b0000']
    
    bars = ax.bar(range(len(hist)), hist, 
                  color=colors[:len(hist)], 
                  alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel(f'Number of {rank}s per USI', fontsize=6, color='0.2')
    ax.set_ylabel('Number of USIs', fontsize=6, color='0.2')
    ax.set_title(f'Distribution of {rank.capitalize()} Counts per USI', 
                 fontsize=8, fontweight='bold', color='0.2', pad=10)
    
    # Set x-axis labels - make sure we have the right number of labels
    ax.set_xticks(range(len(hist)))
    ax.set_xticklabels(labels)  # Now labels should match hist length
    
    # tick parameters
    ax.tick_params(axis='x', labelsize=6, colors='0.4', length=2, width=0.5, pad=1)
    ax.tick_params(axis='y', labelsize=6, colors='0.4', length=2, width=0.5, pad=1)
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, hist)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(hist)*0.01,
                   f'{count:,}', ha='center', va='bottom', fontsize=6, color='0.2')
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('0.4')
    ax.spines['bottom'].set_color('0.4')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save the plot
    conjugate_suffix = '_conjugates' if conjugates_only else ''
    filename = f'distribution_{rank}_count{conjugate_suffix}.png'
    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to {output_folder}/{filename}")
    plt.close()
    

def main(rank='phylum', conjugates_only=False, base_umap_dir='masst/umap'):
    """
    Main analysis pipeline for taxonomic rank-based UMAP colored by rank count
    
    Args:
        rank: Taxonomic rank to use ('phylum', 'class', 'order', 'family', 'genus')
        conjugates_only: Whether to only include conjugate USIs (name contains '_')
        base_umap_dir: Base directory for UMAP data
    """
    
    print(f"Running UMAP analysis for taxonomic rank: {rank}")
    print(f"Coloring by number of {rank}s per USI")
    print(f"Conjugates only: {conjugates_only}")
    
    # Set up directories based on rank
    data_dir = f'{base_umap_dir}/{rank}_based/data'
    output_dir = f'{base_umap_dir}/{rank}_based/plots'
    
    # Check if processed data exists
    suffix = '_conjugates' if conjugates_only else ''
    required_files = [
        f'umap_feature_matrix_clean{suffix}.npy',
        f'usi_ids_clean{suffix}.txt',
        f'umap_embedding{suffix}.npy'
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            print(f"Error: Required processed data file not found: {os.path.join(data_dir, file)}")
            print(f"Please run data processing pipeline first for rank '{rank}' with conjugates_only={conjugates_only}")
            return
    
    # Load USI labeling data
    print("Loading USI labeling data...")
    labeling_df = load_usi_labeling_data(base_umap_dir)
    
    if labeling_df is None:
        print("No labeling data found. Please run prepare_data_for_umap.py first.")
        return
    
    # Load processed data
    feature_matrix, usi_ids, embedding = load_processed_data(data_dir, conjugates_only)

    # Get rank counts for USIs
    print(f"\nMapping USIs to {rank} counts...")
    rank_counts = get_rank_counts(usi_ids, labeling_df, rank)
    
    # # Create plots
    # print("\nGenerating plots...")
    # os.makedirs(output_dir, exist_ok=True)
    # plot_umap_by_rank_count(embedding, usi_ids, rank_counts, rank, conjugates_only, output_folder=output_dir)
    
    # Create distribution plots
    print("\nGenerating distribution plots...")
    plot_rank_count_distribution(rank_counts, rank, conjugates_only, output_folder=output_dir)

    # Save final results with rank counts
    conjugate_suffix = '_conjugates' if conjugates_only else ''
    results_df = pd.DataFrame({
        'usi': usi_ids,
        'umap_1': embedding[:, 0],
        'umap_2': embedding[:, 1],
        f'{rank}_count': rank_counts,
        'taxonomic_rank': rank,
        'conjugates_only': conjugates_only
    })
    
    results_df.to_csv(f'{data_dir}/umap_results_{rank}_count{conjugate_suffix}.csv', index=False)
    print(f"\nResults saved to '{data_dir}/umap_results_{rank}_count{conjugate_suffix}.csv'")
    
    conjugate_text = ' (conjugates only)' if conjugates_only else ''
    print(f"\nAnalysis complete for rank '{rank}' using {rank} count coloring{conjugate_text}")
    print(f"Data directory: {data_dir}")
    print(f"Plots directory: {output_dir}")


if __name__ == "__main__":

    for rank in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
        print(f"\n{'='*60}")
        print(f"Processing rank: {rank}")
        print('='*60)
        
        # Run for all USIs
        main(
            rank=rank,
            conjugates_only=False,
            base_umap_dir='masst/umap'
        )
        
        # Run for conjugates only
        main(
            rank=rank,
            conjugates_only=True,
            base_umap_dir='masst/umap'
        )