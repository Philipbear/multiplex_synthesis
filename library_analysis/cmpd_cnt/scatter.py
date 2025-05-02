import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_dual_scatter_plot(df, column1, column2, figsize=(12, 8), save_path=None):
    """
    Create a scatter plot of two columns sorted from high to low with different colors

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    column1 : str
        First column name to plot
    column2 : str
        Second column name to plot
    figsize : tuple, optional
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
    """
    # Sort dataframe by the first column in descending order
    sorted_df1 = df.sort_values(by=column1, ascending=False).reset_index(drop=True)
    # Sort dataframe by the second column in descending order
    sorted_df2 = df.sort_values(by=column2, ascending=False).reset_index(drop=True)

    # Create x values (just the row numbers)
    x1 = np.arange(len(sorted_df1))
    x2 = np.arange(len(sorted_df2))

    # Create the plot
    plt.rcParams['font.family'] = 'Arial'

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Set y-axis to logarithmic scale
    ax.set_yscale('log')

    # Set y-axis limits
    y_max = max(sorted_df1[column1].max(), sorted_df2[column2].max())
    ax.set_ylim(1, y_max * 1.2)

    # Plot the scatter points with different colors

    scatter2 = ax.scatter(x2, sorted_df2[column2],
                          color='#809bce',
                          s=0.05,
                          alpha=1,
                          edgecolors='#809bce',  # Match to fill color to ensure fully filled dots
                          label='MS/MS reference spectra')

    scatter1 = ax.scatter(x1, sorted_df1[column1],
                          color='#c7522a',
                          s=0.05,
                          alpha=1,
                          edgecolors='#c7522a',  # Match to fill color to ensure fully filled dots
                          label='Structures observed')


    ax.tick_params(axis='both', which='major', length=2, width=0.8, pad=1.5,
                   colors='0', labelsize=5.5)

    # Also set minor tick parameters for log scale
    ax.tick_params(axis='y', which='minor', length=1, width=0.5, colors='0')

    ax.set_xlabel('Synthesis reactions', fontsize=7)
    ax.set_ylabel('Count', fontsize=7)

    # Add legend
    legend = ax.legend(loc='upper right', fontsize=7, frameon=False, markerscale=8, handletextpad=0.1)

    # Style adjustments
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Tight layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='svg', transparent=True)

    # Show plot
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv('cmpd_cnt_summary.tsv', sep='\t', low_memory=False)

    # Create plot with both columns
    create_dual_scatter_plot(df, 'observed_cmpd_no', 'observed_spec_no',
                             figsize=(2.8, 1.8),
                             save_path='dual_scatter_plot.svg')