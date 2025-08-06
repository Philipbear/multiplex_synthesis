import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_data():
    df = pd.read_csv('target_drugs/5-ASA-PP/datasets/raw/df_melted_rito_RA_COHORT.csv', low_memory=False)

    df = df[['Peak_Area', 'sulfasalazine']]

    # Convert to boolean: 'user' means using, other values mean not using
    df['5ASA_use'] = df['sulfasalazine'].apply(lambda x: True if x == 'user' else False)

    # Create use categories for grouping
    df['use_group'] = df['5ASA_use'].apply(lambda x: 'Yes' if x else 'No')

    print("5-ASA use counts:", df['5ASA_use'].value_counts())
    print("Use groups:", df['use_group'].value_counts())
    
    # active users, how many intensities are > 1
    active_users = df[df['5ASA_use'] == True]['Peak_Area'] > 1
    print("Active users with intensity > 1:", active_users.sum())

    return df


def create_boxplot(df, output_path):
    """
    Create box plot for log peak area values grouped by 5-ASA use.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'Peak_Area', 'use_group'
        output_path (str): Path to save the output SVG file
    """
    # Set Arial font
    plt.rcParams['font.family'] = 'Arial'
    # set font size
    plt.rcParams['font.size'] = 7
    
    # Calculate log peak area (add small value to avoid log(0))
    df['log_peak_area'] = np.log10(df['Peak_Area'] + 1)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(1.3, 1.85))
    
    # Define colors for the two groups
    colors = {
        'Yes': '#e74c3c',      # Red
        'No': '#3498db'    # Blue
    }
    
    # Create box plot based on use groups
    box_data = []
    box_labels = []
    box_positions = []

    use_groups = ['No', 'Yes']  # Groups based on 5-ASA use

    for i, use_group in enumerate(use_groups):
        group_data = df[df['use_group'] == use_group]['log_peak_area']
        n_samples = len(group_data)
        box_data.append(group_data)
        box_labels.append(f'{use_group}\n(n = {n_samples})')
        box_positions.append(i + 1)
    
    # Create box plot
    bp = ax.boxplot(box_data, positions=box_positions, labels=box_labels, 
                    patch_artist=True, widths=0.6,
                    boxprops=dict(facecolor='0.8', alpha=0.3, linewidth=0.5),
                    medianprops=dict(color='0.15', linewidth=0.75),
                    whiskerprops=dict(color='0.5'),
                    capprops=dict(color='0.5'),
                    flierprops=dict(marker='', markersize=0))
    
    # Add individual points colored by group
    for i, use_group in enumerate(use_groups):
        group_df = df[df['use_group'] == use_group]
        
        if len(group_df) > 0:
            # Add jitter to x-coordinates
            jitter_width = 0.25
            x_jittered = np.random.normal(i + 1, jitter_width * 0.3, len(group_df))
            
            ax.scatter(x_jittered, group_df['log_peak_area'], 
                      c=colors[use_group], alpha=0.85, s=10, 
                      edgecolors='0.4', linewidth=0.3,
                      label=use_group)
    
    # Customize the plot
    ax.set_xlabel('Sulfasalazine use', fontsize=6, color='0.2', labelpad=2)
    ax.set_ylabel('5-ASA-phenylpropionate\npeak area (log10)', fontsize=5.5, color='0.2')
    ax.set_title('RA cohort', fontsize=7, color='0.2', pad=30)
    
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
              labelspacing=0.3,
              handletextpad=0.2,
              bbox_to_anchor=(0.45, 1.07))
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, format='svg', bbox_inches='tight', transparent=True)
    
    print(f"Box plot saved to: {output_path}")
    print(f"Data summary by use group:")
    print(df['use_group'].value_counts())
    
    plt.show()


if __name__ == "__main__":
    
    df = load_data()
    
    # Create and save the box plot
    create_boxplot(df, 'target_drugs/5-ASA-PP/datasets/plots/boxplot_3.svg')