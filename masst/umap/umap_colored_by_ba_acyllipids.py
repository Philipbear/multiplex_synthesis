import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def load_ms2_lib_metadata():
    """Load MS2 library metadata with NPClassifier results"""
    try:
        df = pd.read_pickle('masst/umap/data/ms2_all_df_unique_usi.pkl')
        print(f"Loaded MS2 library metadata: {len(df)} entries")
        
        # Create USI to np_superclass mapping and conjugate status
        usi_to_np_superclass = df.set_index('usi')['np_superclass'].to_dict()
        usi_to_is_conjugate = df.set_index('usi')['name'].apply(lambda x: '_' in str(x) if pd.notna(x) else False).to_dict()
        
        return usi_to_np_superclass, usi_to_is_conjugate
    except FileNotFoundError:
        print("MS2 library metadata not found. Run get_ms2_lib_info.py first.")
        return {}, {}


def get_compound_classes(usi_list, usi_to_superclass):
    """Get chemical superclasses for USIs and map to specific categories"""
    classes = []
    unknown_count = 0
    
    for usi in usi_list:
        if usi and usi in usi_to_superclass:
            class_info = usi_to_superclass[usi]
            
            # Handle numpy arrays or pandas Series - take the first element
            if hasattr(class_info, '__len__') and not isinstance(class_info, str):
                if len(class_info) > 0:
                    class_info = class_info[0] if hasattr(class_info, '__getitem__') else str(class_info)
                else:
                    class_info = None
            
            # Handle NaN, None, empty string cases
            if class_info is None or (hasattr(class_info, '__len__') and len(class_info) == 0):
                classes.append('Others')
                unknown_count += 1
            elif pd.isna(class_info) or class_info == '':
                classes.append('Others')
                unknown_count += 1
            else:
                # Convert to string and strip whitespace
                class_str = str(class_info).strip()
                if class_str == '' or class_str.lower() == 'nan':
                    classes.append('Others')
                    unknown_count += 1
                else:
                    # Split by ';' and take the first part
                    if ';' in class_str:
                        class_str = class_str.split(';')[0].strip()
                    
                    # Check again if after splitting it's empty
                    if class_str == '':
                        classes.append('Others')
                        unknown_count += 1
                    else:
                        # Map to specific categories
                        if class_str == 'Steroids':
                            classes.append('Steroids')
                        elif class_str == 'Fatty amides':
                            classes.append('Fatty amides')
                        else:
                            classes.append('Others')
        else:
            classes.append('Others')
            unknown_count += 1
    
    print(f"Found classes for {len(usi_list) - unknown_count}/{len(usi_list)} USIs")
    print(f"Unclassified USIs: {unknown_count}")

    return classes


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

def plot_umap(embedding, usi_ids, classes, rank, conjugates_only=False, output_folder='plots'):
    os.makedirs(output_folder, exist_ok=True)
    
    # Get unique classes and their counts
    unique_classes = list(set(classes))
    
    # Define FIXED category order for consistent legends across all ranks
    fixed_category_order = ['Steroids', 'Fatty amides', 'Others']
    
    # Only include categories that are actually present in the data
    category_order = [cls for cls in fixed_category_order if cls in unique_classes]
    
    # Define plotting order - Others first (background), then the rest
    plotting_order = ['Others'] + [cls for cls in category_order if cls != 'Others']
    
    # Custom color palette for the three categories
    class_colors = {
        "Steroids": "#e74c3c",      # Red
        "Fatty amides": "#3498db",    # Blue  
        "Others": "#bdc3c7"           # Light gray
    }
    
    # Set font for all text elements
    plt.rcParams['font.family'] = 'Arial'
    
    # Create UMAP plot WITHOUT legend
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot in plotting order (Others first as background)
    scatter_handles = {}
    for cls in plotting_order:
        if cls in unique_classes:
            mask = np.array(classes) == cls
            if np.any(mask):
                alpha = 0.4 if cls == 'Others' else 0.75
                size = 0.5 if cls == 'Others' else 0.8
                handle = ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                                  c=[class_colors[cls]], label=f'{cls}',
                                  alpha=alpha, s=size, edgecolor='none')
                scatter_handles[cls] = handle

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
    filename_base = f'umap_{rank}_by_ba_acyllipids{conjugate_suffix}'
    plt.savefig(os.path.join(output_folder, f'{filename_base}.png'), dpi=600, bbox_inches='tight', transparent=True)
    print(f"UMAP plot saved to {output_folder}/{filename_base}.png")
    plt.close()
    
    # Create separate legend figure
    fig_legend, ax_legend = plt.subplots(figsize=(2.5, 2))
    ax_legend.axis('off')
    
    # Create legend handles and labels in fixed order
    legend_handles = []
    legend_labels = []
    for cls in category_order:  # Use the fixed order
        if cls in scatter_handles:
            # Create a new scatter plot point for the legend
            handle = ax_legend.scatter([], [], c=[class_colors[cls]], 
                                     alpha=0.75, s=10, edgecolor='none')
            legend_handles.append(handle)
            legend_labels.append(cls)
    
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
    Main analysis pipeline for taxonomic rank-based UMAP colored by steroids and fatty amides

    Args:
        rank: Taxonomic rank to use ('phylum', 'class', 'order', 'family', 'genus')
        conjugates_only: Whether to only include conjugate USIs (name contains '_')
        base_umap_dir: Base directory for UMAP data
    """
    
    print(f"Running UMAP analysis for taxonomic rank: {rank}")
    print(f"Coloring by steroids and fatty amides")
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
    
    # Load chemical class metadata
    print("Loading chemical class metadata...")
    usi_to_np_superclass, usi_to_is_conjugate = load_ms2_lib_metadata()
    
    # Load processed data
    print(f"Loading processed {rank} data...")
    feature_matrix, usi_ids, embedding = load_processed_data(data_dir, conjugates_only)

    # Get chemical classes for USIs
    print("\nMapping USIs to steroids and fatty amides...")
    classes = get_compound_classes(usi_ids, usi_to_np_superclass)
    
    # Print classification statistics
    class_counts = pd.Series(classes).value_counts()
    print(f"\nClassification results:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} USIs ({count/len(classes)*100:.1f}%)")
    
    # Create plots
    print("\nGenerating plots...")
    os.makedirs(output_dir, exist_ok=True)
    plot_umap(embedding, usi_ids, classes, rank, conjugates_only, output_folder=output_dir)

    # Save final results with classes
    conjugate_suffix = '_conjugates' if conjugates_only else ''
    results_df = pd.DataFrame({
        'usi': usi_ids,
        'umap_1': embedding[:, 0],
        'umap_2': embedding[:, 1],
        'chemical_class': classes,
        'taxonomic_rank': rank,
        'conjugates_only': conjugates_only
    })
    
    results_df.to_csv(f'{data_dir}/umap_results_{rank}_ba_acyllipids{conjugate_suffix}.csv', index=False)
    print(f"\nResults saved to '{data_dir}/umap_results_{rank}_ba_acyllipids{conjugate_suffix}.csv'")
    
    conjugate_text = ' (conjugates only)' if conjugates_only else ''
    print(f"Data directory: {data_dir}")
    print(f"Plots directory: {output_dir}")


if __name__ == "__main__":
    
    # Run for multiple ranks
    for rank in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
        print(f"\n{'='*60}")
        print(f"Processing rank: {rank}")
        print('='*60)
        
        # # Run for all USIs
        # main(
        #     rank=rank,
        #     conjugates_only=False,
        #     base_umap_dir='masst/umap'
        # )
        
        # Run for conjugates only
        main(
            rank=rank,
            conjugates_only=True,
            base_umap_dir='masst/umap'
        )