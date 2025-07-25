import matplotlib.pyplot as plt
from pyteomics import mzml
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def get_xics(mzml_file, target_mz, ms_level=1, rt_range=None, mz_tol=0.01):
    """
    Function to extract XIC for a single m/z value.
    :param mzml_file: Path to the mzML file.
    :param target_mz: Target m/z value.
    :param ms_level: MS level to consider (1 or 2).
    :param rt_range: Retention time range as a tuple (start, end). If None, extract for all.
    :param mz_tol: Tolerance for the m/z value.
    :return: Dictionary with 'times' and 'intensities' lists.
    """
    times = []
    intensities = []

    with mzml.read(mzml_file) as reader:
        for spectrum in reader:
            if spectrum['ms level'] == ms_level:
                time = spectrum['scanList']['scan'][0]['scan start time']

                # Check if the retention time is within the specified range
                if rt_range is not None and not (rt_range[0] <= time <= rt_range[1]):
                    continue

                if len(times) == 0 or times[-1] != time:
                    times.append(time)

                    mz_array = spectrum['m/z array']
                    intensity_array = spectrum['intensity array']

                    xic_intensity = np.sum(
                        intensity_array[(mz_array >= target_mz - mz_tol) & (mz_array <= target_mz + mz_tol)])
                    intensities.append(xic_intensity)

    return {'times': times, 'intensities': intensities}


def plot_xics(xic_data_list, labels=None, rt_range=None,
              fig_size=(3, 6), linewidth=2.5,
              x_axis_bottom_only=True,
              save=False, name='xics.svg'):
    """
    Function to plot multiple XICs vertically with shared x-axis.
    :param xic_data_list: List of XIC dictionaries (each with 'times' and 'intensities').
    :param labels: List of labels for each XIC.
    :param rt_range: Retention time range as a tuple (start, end) in minutes.
    :param fig_size: Figure size as a tuple (width, height).
    :param linewidth: Line width for plotting.
    :param x_axis_bottom_only: If True, only show x-axis on bottom subplot.
    :param save: Whether to save the plot.
    :param name: Filename for saving.
    """
    # colors = ['#56648a', '#6280a5', '#8ca5c0', '#8d7e95', '#facaa9', '#ca9a96']
    
    colors = ['#c7522a', '#008585']
    
    n_plots = len(xic_data_list)
    
    # font settings
    plt.rcParams['font.family'] = 'Arial'
    
    fig, axes = plt.subplots(n_plots, 1, figsize=fig_size, sharex=True)
    
    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = [axes]
    
    if len(xic_data_list) > len(colors):
        # Create a custom colormap
        cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(xic_data_list))
        color_list = [cmap(i) for i in np.linspace(0, 1, len(xic_data_list))]
    else:
        color_list = colors

    for i, (xic_data, ax) in enumerate(zip(xic_data_list, axes)):
        times = np.array(xic_data['times'])
        intensities = xic_data['intensities']
        
        ax.plot(times, intensities, color=color_list[i % len(color_list)], linewidth=linewidth)
        
        # Add label if provided
        if labels and i < len(labels):
            ax.text(0.0, 0.9, labels[i], transform=ax.transAxes, 
                   fontname='Arial', fontsize=6, color='0.2', 
                   verticalalignment='top')
        
        # Remove top, right, and left spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Keep bottom spine visible for all subplots
        ax.spines['bottom'].set_visible(True)
        
        # Set the color of the frame (axes spines)
        for spine in ax.spines.values():
            spine.set_color('0.5')
        
        # Hide y-axis
        ax.yaxis.set_visible(False)
        
        # Set the color of tick labels to 0.2
        ax.tick_params(axis='x', colors='0.2', length=2, width=0.5, labelsize=6, pad=1)
        
        # Hide x-axis ticks and labels for all but the bottom plot if specified
        if x_axis_bottom_only and i < n_plots - 1:
            ax.tick_params(axis='x', which='both', length=2, labelbottom=True)
        else:
            ax.tick_params(axis='x', which='both', length=2, labelbottom=True)
        
        # # Hide x-axis for all but the bottom plot if specified
        # if x_axis_bottom_only and i < n_plots - 1:
        #     ax.xaxis.set_visible(False)
        #     ax.spines['bottom'].set_visible(False)
    
    # Set x-axis label only on the bottom plot
    if x_axis_bottom_only:
        axes[-1].set_xlabel('RT (min)', fontsize=6, labelpad=1.5, color='0.2')
    else:
        for ax in axes:
            ax.set_xlabel('RT (min)', fontsize=6, labelpad=1.5, color='0.2')

    if rt_range:
        axes[0].set_xlim(rt_range)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(name, transparent=True, format='svg', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':

    #############
    target_mz = 350.2326
    mz_ppm = 10

    # Get XIC data for standard
    std_xic = get_xics('/Users/shipei/Documents/projects/multiplex_synthesis/target_drugs/ibuprofen-carnitine/mzml/100nM_STANDARD_MIX.mzML', 
                       target_mz, mz_tol=target_mz * mz_ppm / 1e6)
    
    # Get XIC data for biological sample
    bio_xic = get_xics('/Users/shipei/Documents/projects/multiplex_synthesis/target_drugs/ibuprofen-carnitine/mzml/IBD_BIOLOGICAL_2.mzML', 
                       target_mz, mz_tol=target_mz * mz_ppm / 1e6)

    # Plot both XICs
    plot_xics([std_xic, bio_xic], 
            #   labels=[f'Chemical\nstandard', f'IBD feces\nsample'],
              fig_size=(1.2, 1.35), linewidth=1,
              rt_range=(3.3, 6), 
              save=True, name='target_drugs/ibuprofen-carnitine/plots/xics.svg')