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
        
        # Create USI to label mapping
        usi_to_label = df.set_index('usi')['label'].to_dict()
        
        return usi_to_label
    except FileNotFoundError:
        print(f"USI labeling data not found at {labeling_path}. Run prepare_data_for_umap.py first.")
        return {}


def get_show_name_labels(usi_list, usi_to_label):
    """Get show name labels for USIs"""
    labels = []
    unknown_count = 0
    
    for usi in usi_list:
        if usi and usi in usi_to_label:
            label = usi_to_label[usi]
            
            # Handle potential None or NaN values
            if label is None or (isinstance(label, float) and pd.isna(label)):
                labels.append('Others')
                unknown_count += 1
            else:
                labels.append(str(label))
        else:
            labels.append('Others')
            unknown_count += 1
    
    #########
    # only keep labels with one cat
    labels = [label if ' + ' not in label else 'Others' for label in labels]
    
    print(f"Found labels for {len(usi_list) - unknown_count}/{len(usi_list)} USIs")
    print(f"Unlabeled USIs (set to 'Others'): {unknown_count}")

    return labels


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

def plot_umap(embedding, usi_ids, labels, rank, conjugates_only=False, output_folder='plots'):
    """Create UMAP plots colored by show name labels"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Get unique labels and their counts
    unique_labels = list(set(labels))
    label_counts = pd.Series(labels).value_counts()
    
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} USIs ({count/len(labels)*100:.1f}%)")
    
    # Define FIXED category order for consistent legends across all ranks
    fixed_category_order = ['Humans', 'Rodents', 'Plants', 'Animals', 'Fungi', 'Bacteria', 'Others']
    
    # Only include categories that are actually present in the data
    category_order = [cat for cat in fixed_category_order if cat in unique_labels]
    # Add any unexpected categories at the end
    for label in unique_labels:
        if label not in fixed_category_order:
            category_order.append(label)
    
    # Define plotting order - Others first (background), then the rest
    plotting_order = ['Others'] + [cat for cat in category_order if cat != 'Others']
        
    # Custom color palette for show names
    label_colors = {
        "Humans": "#e74c3c",      # Red
        "Rodents": "#f39c12",     # Orange
        "Plants": "#27ae60",      # Green
        "Animals": "#3498db",     # Blue
        "Fungi": "#9b59b6",       # Purple
        "Bacteria": "#34495e",    # Dark gray
        "Others": "#bdc3c7"       # Light gray
    }
    
    # Fallback colors for combination categories
    combination_colors = [
        '#e67e22',  # Orange (darker)
        '#2ecc71',  # Light green
        '#8e44ad',  # Purple (darker)
        '#16a085',  # Teal
        '#c0392b',  # Red (darker)
        '#d35400',  # Orange (darkest)
        '#7f8c8d',  # Gray
    ]
    
    # Create color mapping
    label_to_color = {}
    combination_index = 0
    
    for label in unique_labels:
        if label in label_colors:
            label_to_color[label] = label_colors[label]
        elif ' + ' in label:
            # Use combination colors for mixed categories
            if combination_index < len(combination_colors):
                label_to_color[label] = combination_colors[combination_index]
                combination_index += 1
            else:
                label_to_color[label] = '#95a5a6'  # Default gray for additional combinations
        else:
            label_to_color[label] = '#95a5a6'  # Default gray for other categories
    
    # Set font for all text elements
    plt.rcParams['font.family'] = 'Arial'
    
    # Create UMAP plot WITHOUT legend
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot in plotting order (Others first as background)
    scatter_handles = {}
    for label in plotting_order:
        if label in unique_labels:
            mask = np.array(labels) == label
            if np.any(mask):
                alpha = 0.4 if label == 'Others' else 0.65
                size = 0.5 if label == 'Others' else 0.65
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
    filename_base = f'umap_{rank}_by_show_names{conjugate_suffix}'
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
                                     alpha=0.65, s=10, edgecolor='none')
            legend_handles.append(handle)
            legend_labels.append(label)
    
    # Create legend
    legend = ax_legend.legend(legend_handles, legend_labels, 
                             loc='center', fontsize=8, markerscale=1.5, 
                             frameon=False, ncol=1)
    
    # Save legend separately
    plt.savefig(os.path.join(output_folder, f'{filename_base}_legend.svg'), format='svg', bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(output_folder, f'{filename_base}_legend.png'), dpi=600, bbox_inches='tight', transparent=True)
    print(f"Legend saved to {output_folder}/{filename_base}_legend.svg")
    plt.close()


def main(rank='phylum', conjugates_only=False, base_umap_dir='masst/umap'):
    """
    Main analysis pipeline for taxonomic rank-based UMAP colored by show names
    
    Args:
        rank: Taxonomic rank to use ('phylum', 'class', 'order', 'family', 'genus')
        conjugates_only: Whether to only include conjugate USIs (name contains '_')
        base_umap_dir: Base directory for UMAP data
    """
    
    print(f"Running UMAP analysis for taxonomic rank: {rank}")
    print(f"Using show name labeling")
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
    usi_to_label = load_usi_labeling_data(base_umap_dir)
    
    if not usi_to_label:
        print("No labeling data found. Please run prepare_data_for_umap.py first.")
        return
    
    # Load processed data
    feature_matrix, usi_ids, embedding = load_processed_data(data_dir, conjugates_only)

    # Get show name labels for USIs
    print("\nMapping USIs to show name labels...")
    labels = get_show_name_labels(usi_ids, usi_to_label)
    labeled_count = sum(1 for l in labels if l != 'Others')
    print(f"Labeled {labeled_count}/{len(labels)} USIs with specific show names")
    
    # Create plots
    print("\nGenerating plots...")
    os.makedirs(output_dir, exist_ok=True)
    plot_umap(embedding, usi_ids, labels, rank, conjugates_only, output_folder=output_dir)

    # Save final results with labels
    conjugate_suffix = '_conjugates' if conjugates_only else ''
    results_df = pd.DataFrame({
        'usi': usi_ids,
        'umap_1': embedding[:, 0],
        'umap_2': embedding[:, 1],
        'show_name_label': labels,
        'taxonomic_rank': rank,
        'conjugates_only': conjugates_only
    })
    
    results_df.to_csv(f'{data_dir}/umap_results_{rank}_show_names{conjugate_suffix}.csv', index=False)
    print(f"\nResults saved to '{data_dir}/umap_results_{rank}_show_names{conjugate_suffix}.csv'")
    
    conjugate_text = ' (conjugates only)' if conjugates_only else ''
    print(f"\nAnalysis complete for rank '{rank}' using show name labeling{conjugate_text}")
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