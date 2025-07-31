import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyparsing import line
import seaborn as sns


def load_data():
    df = pd.read_csv('target_drugs/5-ASA-PP/datasets/MSV000084775_df.tsv', sep='\t', low_memory=False)
    
    df = df[['intensity', 'ASA_Exposure', 'Current_5ASA', 'Diagnosis']]
    # rename columns
    df.columns = ['intensity', '5ASA_prior_use', '5ASA_current_use', 'diagnosis']

    df['diagnosis'] = df['diagnosis'].apply(lambda x: 'IBD' if x in ['UC', 'CD'] else 'Healthy')

    # remove rows with 'Missing' in '5ASA_current_use' and 'IBD' in 'diagnosis'
    _mask = (df['5ASA_current_use'] == 'Missing') & (df['diagnosis'] == 'IBD')
    df = df[~_mask].reset_index(drop=True)

    # Convert to boolean: '0' means exposed/using, other values mean not exposed/not using
    df['5ASA_prior_use'] = df['5ASA_prior_use'].apply(lambda x: True if x == '1' else False)
    df['5ASA_current_use'] = df['5ASA_current_use'].apply(lambda x: True if x == '1' else False)

    # Create combined exposure categories
    def get_exposure_category(row):
        if row['5ASA_current_use']:
            return 'Current use'
        elif row['5ASA_prior_use']:
            return 'Prior use only'
        else:
            return 'No exposure'
    
    df['5ASA_exposure_category'] = df.apply(get_exposure_category, axis=1)

    print("5-ASA Prior use counts:", df['5ASA_prior_use'].value_counts())
    print("5-ASA Current use counts:", df['5ASA_current_use'].value_counts())
    print("5-ASA Exposure categories:", df['5ASA_exposure_category'].value_counts())

    return df


def create_boxplot(df, output_path):
    """
    Create box plot for log intensity values grouped by diagnosis and colored by 5-ASA exposure category.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'intensity', '5ASA_exposure_category', 'diagnosis'
        output_path (str): Path to save the output SVG file
    """
    # Set Arial font
    plt.rcParams['font.family'] = 'Arial'
    # set font size
    plt.rcParams['font.size'] = 7
    
    # Calculate log intensity (add small value to avoid log(0))
    df['log_intensity'] = np.log10(df['intensity'] + 1)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(1.3, 1.85))
    
    # Define colors for 5-ASA exposure categories
    colors = {
        'Current use': '#e74c3c',      # Red
        'Prior use only': '#f39c12',   # Orange
        'No exposure': '#3498db'       # Blue
    }
    
        # Create box plot
    box_data = []
    box_labels = []
    box_positions = []
    
    diagnoses = ['Healthy', 'IBD']
    
    for i, diagnosis in enumerate(diagnoses):
        diagnosis_data = df[df['diagnosis'] == diagnosis]['log_intensity']
        n_samples = len(diagnosis_data)
        box_data.append(diagnosis_data)
        box_labels.append(f'{diagnosis}\n(n = {n_samples})')
        box_positions.append(i + 1)
    
    # Create box plot
    bp = ax.boxplot(box_data, positions=box_positions, labels=box_labels, 
                    patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor='0.8', alpha=0.3, linewidth=0.5),
                    medianprops=dict(color='0.15', linewidth=0.75),
                    whiskerprops=dict(color='0.5'),
                    capprops=dict(color='0.5'),
                    flierprops=dict(marker='', markersize=0))
    
    # Add individual points colored by 5-ASA exposure category
    legend_added = set()  # Track which labels we've already added
    
    for i, diagnosis in enumerate(diagnoses):
        diagnosis_df = df[df['diagnosis'] == diagnosis]
        
        # Separate by 5-ASA exposure category
        exposure_categories = ['No exposure', 'Prior use only', 'Current use']
        
        for category in exposure_categories:
            category_data = diagnosis_df[diagnosis_df['5ASA_exposure_category'] == category]
            
            if len(category_data) > 0:
                # Add jitter to x-coordinates
                jitter_width = 0.25
                x_jittered = np.random.normal(i + 1, jitter_width * 0.3, len(category_data))
                
                # Only add label if this category hasn't been labeled yet
                label_to_use = category if category not in legend_added else None
                if label_to_use:
                    legend_added.add(category)
                
                ax.scatter(x_jittered, category_data['log_intensity'], 
                          c=colors[category], alpha=0.85, s=10, 
                          edgecolors='0.4', linewidth=0.3,
                          label=label_to_use)
    
    # Customize the plot
    ax.set_xlabel('Diagnosis', fontsize=6, color='0.2', labelpad=2)
    ax.set_ylabel('5-ASA-phenylpropionate\npeak area (log10)', fontsize=5.5, color='0.2')
    ax.set_title('Pediatric IBD cohort', fontsize=7, color='0.2', pad=30)
    
    # Customize tick parameters
    ax.tick_params(axis='y', colors='0.2', labelsize=5.5,
                   length=2, width=0.5, pad=2)
    ax.tick_params(axis='x', colors='0.2', labelsize=6, length=0)  # Remove x-axis tick marks
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('0.5')
    ax.spines['bottom'].set_color('0.5')
    
    # Add legend
    ax.legend(loc='lower left', frameon=False, fontsize=6, ncol=1, 
              # vertical spacing
              labelspacing=0.3,
              handletextpad=0.2,
              bbox_to_anchor=(0., 1))
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, format='svg', bbox_inches='tight', transparent=True)
    
    print(f"Box plot saved to: {output_path}")
    print(f"Data summary by diagnosis and exposure category:")
    print(df.groupby(['diagnosis', '5ASA_exposure_category']).size().unstack(fill_value=0))
    
    plt.show()


if __name__ == "__main__":
    
    df = load_data()
    
    # Create and save the box plot
    create_boxplot(df, 'target_drugs/5-ASA-PP/datasets/plots/boxplot_1.svg')
    