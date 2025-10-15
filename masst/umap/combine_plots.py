import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import numpy as np


def load_plot_and_legend(base_dir, rank, plot_type, conjugates_only=False):
    """Load both plot image and legend image from the file system"""
    conjugate_suffix = '_conjugates' if conjugates_only else ''
    
    # Define plot filename patterns based on plot type
    filename_patterns = {
        'rank_count': f'umap_{rank}_by_{rank}_count{conjugate_suffix}',
        'show_names': f'umap_{rank}_by_show_names{conjugate_suffix}', 
        'ba_acyllipids': f'umap_{rank}_by_ba_acyllipids{conjugate_suffix}',
        'nppathway': f'umap_{rank}_by_np_pathway{conjugate_suffix}',
        'distribution': f'distribution_{rank}_count{conjugate_suffix}'
    }
    
    plot_dir = f"{base_dir}/{rank}_based/plots"
    filename_base = filename_patterns[plot_type]
    
    # Load UMAP plot (PNG)
    plot_filepath = os.path.join(plot_dir, f'{filename_base}.png')
    plot_img = None
    if os.path.exists(plot_filepath):
        try:
            plot_img = mpimg.imread(plot_filepath)
        except Exception as e:
            print(f"Error loading plot {plot_filepath}: {e}")
    else:
        print(f"Plot file not found: {plot_filepath}")
    
    # Load legend (only for UMAP plots, not distribution plots)
    legend_img = None
    if plot_type != 'distribution':
        legend_filepath = os.path.join(plot_dir, f'{filename_base}_legend.png')
        if os.path.exists(legend_filepath):
            try:
                legend_img = mpimg.imread(legend_filepath)
            except Exception as e:
                print(f"Error loading legend {legend_filepath}: {e}")
        else:
            print(f"Legend file not found: {legend_filepath}")
    
    return plot_img, legend_img, plot_filepath, legend_filepath if plot_type != 'distribution' else None


def create_combined_pdf(base_umap_dir='masst/umap', 
                        conjugates_only_lst=[False, True],
                        output_dir='masst/umap/combined_plots'):
    """Create combined PDFs with all UMAP plots and their legends"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define ranks and plot types
    ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    plot_types = ['rank_count', 'show_names', 'ba_acyllipids', 'nppathway', 'distribution']
    
    # Create PDFs for both conjugates and all data
    for conjugates_only in conjugates_only_lst:
        suffix = '_conjugates' if conjugates_only else '_all'
        pdf_filename = f"umap_combined_plots{suffix}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)
        
        print(f"Creating PDF: {pdf_path}")
        
        with PdfPages(pdf_path) as pdf:
            # Create one page per rank
            for rank in ranks:
                print(f"  Processing rank: {rank}")
                
                # Create figure with landscape orientation (horizontal) and higher DPI
                fig = plt.figure(figsize=(20, 14), dpi=600)
                fig.suptitle(f'{rank.capitalize()} UMAP Plots', fontsize=20, fontweight='bold', y=0.92)

                # Create 2x5 subplot grid (2 rows, 5 columns)
                # Top row: 4 UMAP plots with legends (plot + legend pairs)
                # Bottom row: 1 distribution plot (spans 2 columns) + empty space
                gs = fig.add_gridspec(2, 5, hspace=0.1, wspace=-0.1, 
                                      width_ratios=[3, 1.2, 3, 1.8, 2], 
                                      left=0.05, right=0.95, top=0.90, bottom=0.05)
                
                # Plot and legend positions: (row, col_plot, col_legend) for each plot type
                plot_positions = {
                    'rank_count': (0, 0, 1),      # Top left: plot at (0,0), legend at (0,1)
                    'nppathway': (0, 2, 3),      # Top right: plot at (0,2), legend at (0,3)
                    'show_names': (1, 0, 1),     # Bottom left: plot at (1,0), legend at (1,1)
                    'ba_acyllipids': (1, 2, 3),  # Bottom right: plot at (1,2), legend at (1,3)
                    'distribution': (0, 4, None) # Right side: spans both rows at column 4
                }
                
                loaded_plots = 0
                
                for plot_type in plot_types:
                    # Load both plot and legend images
                    plot_img, legend_img, plot_path, legend_path = load_plot_and_legend(
                        base_umap_dir, rank, plot_type, conjugates_only)
                    
                    if plot_img is not None:
                        # Get positions
                        row, col_plot, col_legend = plot_positions[plot_type]
                        
                        if plot_type == 'distribution':
                            # Distribution plot spans both rows in the last column
                            ax_plot = fig.add_subplot(gs[:, col_plot])
                        else:
                            # Create subplot for the UMAP plot
                            ax_plot = fig.add_subplot(gs[row, col_plot])
                        
                        ax_plot.imshow(plot_img, interpolation='bilinear')
                        ax_plot.axis('off')
                        
                        # Create subplot for the legend (if available and not distribution)
                        if plot_type != 'distribution' and legend_img is not None:
                            ax_legend = fig.add_subplot(gs[row, col_legend])
                            ax_legend.imshow(legend_img, interpolation='bilinear')
                            ax_legend.axis('off')
                        elif plot_type != 'distribution':
                            # Create empty subplot if legend is missing
                            ax_legend = fig.add_subplot(gs[row, col_legend])
                            ax_legend.axis('off')
                            ax_legend.text(0.5, 0.5, 'Legend\nNot Found', 
                                         ha='center', va='center', fontsize=10, 
                                         transform=ax_legend.transAxes, color='gray')
                        
                        loaded_plots += 1
                    else:
                        print(f"    Warning: Could not load {plot_type} plot")
                
                if loaded_plots == 0:
                    # If no plots were loaded, add a message
                    ax = fig.add_subplot(1, 1, 1)
                    ax.text(0.5, 0.5, f'No plots found for {rank}', 
                           ha='center', va='center', fontsize=20, transform=ax.transAxes)
                    ax.axis('off')
                
                # Save page to PDF with high quality
                pdf.savefig(fig, bbox_inches='tight', dpi=600, facecolor='white')
                
                # Save individual rank plot as PNG with
                png_filename = f"umap_combined_{rank}{suffix}.png"
                png_path = os.path.join(output_dir, png_filename)
                fig.savefig(png_path, bbox_inches='tight', dpi=600, facecolor='white')
                print(f"    Saved PNG: {png_filename}")
                
                plt.close(fig)
        
        print(f"PDF saved: {pdf_path}")


def main(conjugates_only_lst=[False, True]):
    """Main function to create combined PDFs"""
    print("Creating combined UMAP plot PDFs...")
    
    # Check if base directory exists
    base_umap_dir = 'masst/umap'
    if not os.path.exists(base_umap_dir):
        print(f"Error: Base UMAP directory not found: {base_umap_dir}")
        return
    
    # Create combined PDFs
    create_combined_pdf(conjugates_only_lst=conjugates_only_lst, base_umap_dir=base_umap_dir)

    print("\nCombined PDF creation complete!")
    if False in conjugates_only_lst:
        print("  - masst/umap/combined_plots/umap_combined_plots_all.pdf")
        print("  - Individual PNG files for each rank (all data)")
    if True in conjugates_only_lst:
        print("  - masst/umap/combined_plots/umap_combined_plots_conjugates.pdf")
        print("  - Individual PNG files for each rank (conjugates only)")


if __name__ == "__main__":
    main(conjugates_only_lst=[True])