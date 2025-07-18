import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os
import pickle


def load_ms2_lib_metadata():
    """Load MS2 library metadata with NPClassifier results"""
    try:
        df = pd.read_pickle('masst/umap/data/ms2_all_df_unique_usi.pkl')
        print(f"Loaded MS2 library metadata: {len(df)} entries")
        
        # Create USI to classifier mapping
        usi_to_np_class = df.set_index('usi')['np_class'].to_dict()
        usi_to_np_superclass = df.set_index('usi')['np_superclass'].to_dict()
        usi_to_np_pathway = df.set_index('usi')['np_pathway'].to_dict()
        
        # Create USI to conjugate status mapping
        usi_to_is_conjugate = df.set_index('usi')['name'].apply(lambda x: '_' in str(x) if pd.notna(x) else False).to_dict()
        
        return usi_to_np_class, usi_to_np_superclass, usi_to_np_pathway, usi_to_is_conjugate
    except FileNotFoundError:
        print("MS2 library metadata not found. Run get_ms2_lib_info.py first.")
        return {}, {}, {}, {}


def filter_conjugates(usi_list, usi_to_is_conjugate, conjugates_only=False):
    """Filter USIs based on conjugate status"""
    if not conjugates_only:
        return usi_list
    
    filtered_usis = []
    conjugate_count = 0
    
    for usi in usi_list:
        if usi in usi_to_is_conjugate and usi_to_is_conjugate[usi]:
            filtered_usis.append(usi)
            conjugate_count += 1
    
    print(f"Conjugate filtering: {conjugate_count}/{len(usi_list)} USIs are conjugates")
    return filtered_usis


def get_compound_classes(usi_list, usi_to_classifier):
    """Get chemical classes for USIs"""
    classes = []
    unknown_count = 0
    
    for usi in usi_list:
        if usi and usi in usi_to_classifier:
            class_info = usi_to_classifier[usi]
            
            # Handle numpy arrays or pandas Series - take the first element
            if hasattr(class_info, '__len__') and not isinstance(class_info, str):
                if len(class_info) > 0:
                    class_info = class_info[0] if hasattr(class_info, '__getitem__') else str(class_info)
                else:
                    class_info = None
            
            # Handle NaN, None, empty string cases
            if class_info is None or (hasattr(class_info, '__len__') and len(class_info) == 0):
                classes.append('Unclassified')
                unknown_count += 1
            elif pd.isna(class_info) or class_info == '':
                classes.append('Unclassified')
                unknown_count += 1
            else:
                # Convert to string and strip whitespace
                class_str = str(class_info).strip()
                if class_str == '' or class_str.lower() == 'nan':
                    classes.append('Unclassified')
                    unknown_count += 1
                else:
                    # Split by ';' and take the first part
                    if ';' in class_str:
                        class_str = class_str.split(';')[0].strip()
                    
                    # Check again if after splitting it's empty
                    if class_str == '':
                        classes.append('Unclassified')
                        unknown_count += 1
                    else:
                        classes.append(class_str)
        else:
            classes.append('Unclassified')
            unknown_count += 1
    
    print(f"Found classes for {len(usi_list) - unknown_count}/{len(usi_list)} USIs")
    print(f"Unclassified USIs: {unknown_count}")

    return classes


def check_processed_data_exists(output_dir, conjugates_only=False):
    """Check if processed data files exist"""
    suffix = '_conjugates' if conjugates_only else ''
    
    required_files = [
        f'umap_feature_matrix_clean{suffix}.npy',
        f'usi_ids_clean{suffix}.txt',
        f'umap_embedding{suffix}.npy'
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(output_dir, file)):
            return False
    return True


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


def create_umap_embedding(feature_matrix, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean'):
    """Create UMAP embedding"""
    print(f"Creating UMAP embedding with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42,
        verbose=True
    )
    
    embedding = reducer.fit_transform(feature_matrix)
    
    return embedding, reducer

def plot_umap(embedding, usi_ids, classes, classifier_name, rank, conjugates_only=False, output_folder='plots'):
    """Create UMAP plots colored by chemical classes"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Get unique classes and their counts
    unique_classes = list(set(classes))
    
    # Define FIXED category order for consistent legends across all ranks
    fixed_category_order = [
        "Fatty acids",
        "Shikimates and Phenylpropanoids", 
        "Terpenoids",
        "Alkaloids",
        "Amino acids and Peptides",
        "Carbohydrates",
        "Polyketides",
        "Unclassified"
    ]
    
    # Only include categories that are actually present in the data
    category_order = [cls for cls in fixed_category_order if cls in unique_classes]
    # Add any unexpected categories at the end
    for cls in unique_classes:
        if cls not in fixed_category_order:
            category_order.append(cls)
    
    # Define plotting order - Unclassified first (background), then the rest
    plotting_order = ['Unclassified'] + [cls for cls in category_order if cls != 'Unclassified']
        
    # Custom color palette
    class_colors = {
        "Fatty acids": "#4e639e", 
        "Shikimates and Phenylpropanoids": "#e54616", 
        "Terpenoids": "#dba053",
        "Alkaloids": "#ff997c", 
        "Amino acids and Peptides": "#7fbfdd", 
        "Carbohydrates": "#96a46b",
        "Polyketides": "#760f00",
        "Unclassified": "0.8"  # grey for unclassified
    }
    
    # Fallback colors for classes not in predefined list
    fallback_colors = [
        '#17becf',  # cyan
        '#bcbd22',  # olive
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#2ca02c',  # green
        '#d62728',  # red
    ]
    
    # Create color mapping
    class_to_color = {}
    fallback_index = 0
    
    for cls in unique_classes:
        if cls in class_colors:
            class_to_color[cls] = class_colors[cls]
        else:
            # Use fallback colors for classes not in the predefined list
            if fallback_index < len(fallback_colors):
                class_to_color[cls] = fallback_colors[fallback_index]
                fallback_index += 1
            else:
                class_to_color[cls] = '0.5'  # grey for additional classes
    
    # Set font for all text elements
    plt.rcParams['font.family'] = 'Arial'
    
    # Create UMAP plot WITHOUT legend
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot in plotting order (Unclassified first as background)
    scatter_handles = {}
    for cls in plotting_order:
        if cls in unique_classes:
            mask = np.array(classes) == cls
            if np.any(mask):
                handle = ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                                  c=[class_to_color[cls]], label=f'{cls}',
                                  alpha=0.65, s=0.65, edgecolor='none')
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
    filename_base = f'umap_{rank}_by_{classifier_name}{conjugate_suffix}'
    plt.savefig(os.path.join(output_folder, f'{filename_base}.png'), dpi=600, bbox_inches='tight', transparent=True)
    print(f"UMAP plot saved to {output_folder}/{filename_base}.png")
    plt.close()
    
    # Create separate legend figure
    fig_legend, ax_legend = plt.subplots(figsize=(0.8, 2))
    ax_legend.axis('off')
    
    # Create legend handles and labels in fixed order
    legend_handles = []
    legend_labels = []
    for cls in category_order:  # Use the fixed order
        if cls in scatter_handles:
            # Create a new scatter plot point for the legend
            handle = ax_legend.scatter([], [], c=[class_to_color[cls]], 
                                     alpha=0.65, s=10, edgecolor='none')
            legend_handles.append(handle)
            legend_labels.append(cls)
    
    # Create legend
    legend = ax_legend.legend(legend_handles, legend_labels, 
                             loc='center', fontsize=8, markerscale=1.5, 
                             frameon=False, ncol=1)
    
    # Save legend separately
    plt.savefig(os.path.join(output_folder, f'{filename_base}_legend.svg'), format='svg',
                bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(output_folder, f'{filename_base}_legend.png'), dpi=600,
                bbox_inches='tight', transparent=True)
    print(f"Legend saved to {output_folder}/{filename_base}_legend.svg")
    plt.close()


def process_data_pipeline(matrix_path, output_dir, usi_to_is_conjugate, conjugates_only=False, 
                         n_neighbors=20, min_dist=0.5):
    """Complete data processing pipeline directly to UMAP"""
    print(f"Loading raw UMAP data{' (conjugates only)' if conjugates_only else ''}...")
    umap_matrix = pd.read_pickle(matrix_path)
    
    print(f"Original data: {umap_matrix.shape[0]} USIs × {umap_matrix.shape[1]} shown_ranks")
    
    # Filter for conjugates if requested
    if conjugates_only:
        conjugate_usis = [usi for usi in umap_matrix.index if usi in usi_to_is_conjugate and usi_to_is_conjugate[usi]]
        umap_matrix = umap_matrix.loc[conjugate_usis]
        print(f"After conjugate filtering: {umap_matrix.shape[0]} USIs × {umap_matrix.shape[1]} shown_ranks")
    
    # Convert to numpy array for processing
    feature_matrix = umap_matrix.values
    usi_ids = umap_matrix.index.tolist()
    
    # Binarize the feature matrix (1 for any non-zero match, 0 otherwise)
    feature_matrix_binary = (feature_matrix > 0).astype(float)
    
    print(f"Binary matrix shape: {feature_matrix_binary.shape}")
    print(f"Sparsity: {1 - np.count_nonzero(feature_matrix_binary) / feature_matrix_binary.size:.2%}")
    
    # Create UMAP embedding directly from binary matrix
    print("Creating UMAP embedding...")
    embedding, reducer = create_umap_embedding(
        feature_matrix_binary, 
        n_neighbors=n_neighbors,
        min_dist=min_dist, 
        metric='jaccard'  # jaccard is good for binary data
    )
    
    # Save processed data with appropriate suffix
    suffix = '_conjugates' if conjugates_only else ''
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f'umap_feature_matrix_clean{suffix}.npy'), feature_matrix_binary)
    with open(os.path.join(output_dir, f'usi_ids_clean{suffix}.txt'), 'w') as f:
        for usi_id in usi_ids:
            f.write(f"{usi_id}\n")
    np.save(os.path.join(output_dir, f'umap_embedding{suffix}.npy'), embedding)
    print(f"Processed data{' (conjugates)' if conjugates_only else ''} saved for future use")
    
    return feature_matrix_binary, usi_ids, embedding


def main(reprocess_data=False, classifier='np_pathway', rank='phylum', conjugates_only=False,
         base_umap_dir='masst/umap', n_neighbors=20, min_dist=0.5):
    """
    Main analysis pipeline for taxonomic rank-based UMAP
    
    Args:
        reprocess_data: Whether to reprocess the data
        classifier: Chemical classifier to use ('np_class', 'np_superclass', 'np_pathway')
        rank: Taxonomic rank to use ('phylum', 'class', 'order', 'family', 'genus')
        conjugates_only: Whether to only include conjugate USIs (name contains '_')
        base_umap_dir: Base directory for UMAP data
    """
    
    print(f"Running UMAP analysis for taxonomic rank: {rank}")
    print(f"Using chemical classifier: {classifier}")
    print(f"Conjugates only: {conjugates_only}")
    
    # Set up directories based on rank
    data_dir = f'{base_umap_dir}/{rank}_based/data'
    output_dir = f'{base_umap_dir}/{rank}_based/plots'
    
    # Check if the rank-based data exists
    matrix_path = f"{data_dir}/umap_match_matrix.pkl"
    if not os.path.exists(matrix_path):
        print(f"Error: Data for rank '{rank}' not found at {matrix_path}")
        print(f"Please run prepare_data_for_umap.py first to generate data for rank '{rank}'")
        return
    
    # Load chemical class metadata
    print("Loading chemical class metadata...")
    usi_to_np_class, usi_to_np_superclass, usi_to_np_pathway, usi_to_is_conjugate = load_ms2_lib_metadata()
    
    # Check if processed data already exists
    if reprocess_data or not check_processed_data_exists(data_dir, conjugates_only):
        print(f"Processing {rank} data for UMAP...")
        feature_matrix, usi_ids, embedding = process_data_pipeline(
            matrix_path=matrix_path,
            output_dir=data_dir,
            usi_to_is_conjugate=usi_to_is_conjugate,
            conjugates_only=conjugates_only,
            n_neighbors=n_neighbors,
            min_dist=min_dist
        )
    else:
        print(f"Processed {rank} data found. Loading existing data...")
        feature_matrix, usi_ids, embedding = load_processed_data(data_dir, conjugates_only)

    # Get chemical classes for USIs
    print("\nMapping USIs to chemical classes...")
    
    # Use specified classifier
    classifier_mapping = {
        'np_class': usi_to_np_class,
        'np_superclass': usi_to_np_superclass,
        'np_pathway': usi_to_np_pathway
    }
    
    classifier_dict = classifier_mapping[classifier]
    classes = get_compound_classes(usi_ids, classifier_dict)
    classified_count = sum(1 for c in classes if c != 'Unclassified')
    print(f"Using {classifier} for classification ({classified_count}/{len(classes)} classified)")
    
    # Create plots
    print("\nGenerating plots...")
    os.makedirs(output_dir, exist_ok=True)
    plot_umap(embedding, usi_ids, classes, classifier, rank, conjugates_only, output_folder=output_dir)

    # Save final results with classes
    conjugate_suffix = '_conjugates' if conjugates_only else ''
    results_df = pd.DataFrame({
        'usi': usi_ids,
        'umap_1': embedding[:, 0],
        'umap_2': embedding[:, 1],
        'chemical_class': classes,
        'classifier_used': classifier,
        'taxonomic_rank': rank,
        'conjugates_only': conjugates_only
    })
    
    results_df.to_csv(f'{data_dir}/umap_results_{rank}_{classifier}{conjugate_suffix}.csv', index=False)
    print(f"\nResults saved to '{data_dir}/umap_results_{rank}_{classifier}{conjugate_suffix}.csv'")
    
    conjugate_text = ' (conjugates only)' if conjugates_only else ''
    print(f"\nAnalysis complete for rank '{rank}' using classifier '{classifier}'{conjugate_text}")
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
        #     reprocess_data=False,
        #     classifier='np_pathway',
        #     rank=rank,
        #     conjugates_only=False,
        #     base_umap_dir='masst/umap',
        #     n_neighbors=25,
        #     min_dist=0.7
        # )
        
        # Run for conjugates only
        main(
            reprocess_data=False,
            classifier='np_pathway',
            rank=rank,
            conjugates_only=True,
            base_umap_dir='masst/umap',
            n_neighbors=25,
            min_dist=0.7
        )