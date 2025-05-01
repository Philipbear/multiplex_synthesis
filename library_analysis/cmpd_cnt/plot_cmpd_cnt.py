import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def plot_product_comparison():
    # Load the data
    reaction_df = pd.read_csv('reaction_df.tsv', sep='\t')

    # Make sure we have both expected and observed product counts
    valid_data = reaction_df.dropna(subset=['No_of_products_expected', 'observed_products']).copy()

    # Sort data if needed (assuming we're using the original order)
    # valid_data = valid_data.sort_values('Reaction_ID')

    # Calculate cumulative reaction count for x-axis
    valid_data['cumulative_reactions'] = range(1, len(valid_data) + 1)

    # Calculate cumulative sum of products (both expected and observed)
    valid_data['cumulative_expected'] = valid_data['No_of_products_expected'].cumsum()
    valid_data['cumulative_observed'] = valid_data['observed_products'].cumsum()

    # Create the plot
    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(2.7, 1.8))

    # Plot both lines
    plt.plot(valid_data['cumulative_reactions'], valid_data['cumulative_expected'],
             label='Expected products', color='blue', linewidth=1)
    plt.plot(valid_data['cumulative_reactions'], valid_data['cumulative_observed'],
             label='Observed products', color='red', linewidth=1)

    # Add labels and title
    ax.set_xlabel('Number of reactions', fontsize=7, labelpad=3.5)
    ax.set_ylabel('Number of products', fontsize=7, labelpad=3.5)
    ax.tick_params(axis='both', which='major', length=2, width=0.8, pad=1.5,
                   colors='0', labelsize=5.5)
    # Add thousand separators (commas) to axis labels
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Add legend
    plt.legend(fontsize=7, frameon=False, loc='upper left', handlelength=1.5, handletextpad=0.5)

    # # Add grid for better readability
    # plt.grid(True, alpha=0.3)

    # Ensure there's a bit of padding around the plot
    plt.tight_layout()

    # Save the figure
    plt.savefig('expected_vs_observed_products.svg', transparent=True, bbox_inches='tight')

    # Show the plot
    plt.show()

    # Print some statistics
    total_expected = valid_data['No_of_products_expected'].sum()
    total_observed = valid_data['observed_products'].sum()
    print(f"Total expected products: {total_expected}")
    print(f"Total observed products: {total_observed}")
    print(f"Ratio (observed/expected): {total_observed / total_expected:.2f}")


if __name__ == '__main__':
    plot_product_comparison()