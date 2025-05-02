import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def plot_product_comparison():
    # Load the data
    reaction_df = pd.read_csv('cmpd_cnt_summary.tsv', sep='\t')

    # Make sure we have both spectrum and compound counts
    valid_data = reaction_df.dropna(subset=['observed_cmpd_no', 'observed_spec_no']).copy()

    # Calculate cumulative reaction count for x-axis
    valid_data['cumulative_reactions'] = range(1, len(valid_data) + 1)

    # Calculate cumulative sum of observed compounds and spectra
    valid_data['cumulative_compounds'] = valid_data['observed_cmpd_no'].cumsum()
    valid_data['cumulative_spectra'] = valid_data['observed_spec_no'].cumsum()

    # Create the plot
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(2.7, 1.8))

    # Plot both lines
    plt.plot(valid_data['cumulative_reactions'], valid_data['cumulative_compounds'],
             label='Observed compounds', color='blue', linewidth=1)
    plt.plot(valid_data['cumulative_reactions'], valid_data['cumulative_spectra'],
             label='Observed spectra', color='red', linewidth=1)

    # Add labels and title
    ax.set_xlabel('Number of reactions', fontsize=7, labelpad=3.5)
    ax.set_ylabel('Cumulative count', fontsize=7, labelpad=3.5)
    ax.tick_params(axis='both', which='major', length=2, width=0.8, pad=1.5,
                   colors='0', labelsize=5.5)
    # Add thousand separators (commas) to axis labels
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Add legend
    plt.legend(fontsize=7, frameon=False, loc='upper left', handlelength=1.5, handletextpad=0.5)

    # Ensure there's a bit of padding around the plot
    plt.tight_layout()

    # Save the figure
    plt.savefig('compounds_vs_spectra.svg', transparent=True, bbox_inches='tight')

    # Show the plot
    plt.show()

    # Print some statistics
    total_compounds = valid_data['observed_cmpd_no'].sum()
    total_spectra = valid_data['observed_spec_no'].sum()
    print(f"Total observed compounds: {total_compounds}")
    print(f"Total observed spectra: {total_spectra}")
    print(f"Ratio (spectra/compounds): {total_spectra / total_compounds:.2f}")


if __name__ == '__main__':
    plot_product_comparison()