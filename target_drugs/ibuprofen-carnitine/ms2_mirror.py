import matplotlib.pyplot as plt
import numpy as np
from pyteomics import mzml


def get_spectrum_by_scan(mzml_file, scan_ls):
    """
    Function to get a spectrum by scan number from an mzML file.
    :param mzml_file: Path to the mzML file.
    :param scan_ls: List of scan numbers.
    :return: A dictionary of spectra, where the key is the scan number and the value is peaks
    """

    scan_ls = [int(x) for x in scan_ls]
    scan_ls = set(scan_ls)

    out_spec_ls = []
    with mzml.read(mzml_file) as reader:
        for spectrum in reader:
            this_scan_number = spectrum['index'] + 1  # mzML scan numbers are 1-based, but index is 0-based
            if this_scan_number in scan_ls:
                this_spectrum = {
                    'title': f"{mzml_file}:scan:{this_scan_number}",
                    'prec_mz': float(spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']),
                    'mz_ls': spectrum['m/z array'],
                    'intensity_ls': spectrum['intensity array']
                }
                out_spec_ls.append(this_spectrum)

    return out_spec_ls


def create_mirror_plot(peaks_1, peaks_2, fig_size=(3.6, 2.2),
                       ms2_tol=0.05, peak_int_power=1.0,
                       intensity_threshold_label=50, label_decimals=2,
                       max_x=None,
                       up_matched_peak_color='#c7522a', down_matched_peak_color='#008585',
                       up_spec_peak_color='0.6', down_spec_peak_color='0.6',
                       show_unmatched_peaks=True, title=None,
                       save=False,
                       output_name='demo.svg'):
    plt.rcParams['font.family'] = 'Helvetica'
    fig, ax = plt.subplots(figsize=fig_size)

    peaks_1[:, 1] = np.power(peaks_1[:, 1], peak_int_power)
    peaks_2[:, 1] = np.power(peaks_2[:, 1], peak_int_power)

    # Set axis limits and labels
    if max_x is None:
        max_x = max(np.max(peaks_1[:, 0]), np.max(peaks_2[:, 0])) * 1.02

    peaks_1 = peaks_1[peaks_1[:, 0] < max_x]
    peaks_2 = peaks_2[peaks_2[:, 0] < max_x]

    # Normalize intensities
    peaks_1[:, 1] = peaks_1[:, 1] / np.max(peaks_1[:, 1]) * 100
    peaks_2[:, 1] = peaks_2[:, 1] / np.max(peaks_2[:, 1]) * 100

    # Add line at y=0
    plt.axhline(y=0, color='0.5', linewidth=0.4)

    # Find matched peaks
    matched_indices = []
    matched_indices_ref = []
    for i, mz in enumerate(peaks_1[:, 0]):
        matches = np.where(np.abs(peaks_2[:, 0] - mz) <= ms2_tol)[0]
        if matches.size > 0:
            print(f'{mz:.4f} matched to {peaks_2[:, 0][matches]}')
            matched_indices.extend([i] * len(matches))
            matched_indices_ref.extend(matches)

    # Plot unmatched peaks
    if show_unmatched_peaks:
        plt.vlines(peaks_1[:, 0], 0, peaks_1[:, 1], color=up_spec_peak_color, linewidth=0.7)
        plt.vlines(peaks_2[:, 0], 0, -peaks_2[:, 1], color=down_spec_peak_color, linewidth=0.7)

    # Plot matched peaks
    plt.vlines(peaks_1[:, 0][matched_indices], 0, peaks_1[:, 1][matched_indices],
               color=up_matched_peak_color, linewidth=0.7)
    plt.vlines(peaks_2[:, 0][matched_indices_ref], 0, -peaks_2[:, 1][matched_indices_ref],
               color=down_matched_peak_color, linewidth=0.7)

    plt.xlim(0, max_x)
    plt.ylim(-130, 130)

    # Add labels for peaks above threshold
    for mz, intensity in zip(peaks_1[:, 0], peaks_1[:, 1]):
        if intensity >= intensity_threshold_label and mz < max_x:
            plt.text(mz, intensity + 0.5, f'{mz:.{label_decimals}f}', ha='center', va='bottom', fontsize=5)

    # for mz, intensity in zip(peaks_2[:, 0], peaks_2[:, 1]):
    #     if intensity >= intensity_threshold_label and mz < max_x:
    #         plt.text(mz, -intensity - 2.5, f'{mz:.{label_decimals}f}', ha='center', va='top', fontsize=5)

    # plt.xlabel(r'$\mathit{m/z}$', fontsize=5, labelpad=2, color='0.2')
    plt.ylabel('Intensity (%)', fontsize=6, labelpad=0.3, color='0.2')

    plt.gca().xaxis.set_visible(False)
    # plt.gca().yaxis.set_visible(False)

    # Remove spines
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)

    # Set the color of the frame (axes spines)
    for spine in plt.gca().spines.values():
        spine.set_color('0.5')
        spine.set_linewidth(0.4)

    # Remove grid
    # plt.grid(False)

    # Adjust tick parameters
    ax.tick_params(axis='y', which='major', length=1, width=0.4, pad=0.5, colors='0.5', labelsize=5.5)

    # Modify y-axis ticks and labels
    yticks = plt.gca().get_yticks()
    yticks = [y for y in yticks if abs(y) <= 100]
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels([str(abs(int(y))) for y in yticks])

    if title:
        plt.title(title, fontsize=5, color='0.2', pad=2)

    # Save the plot
    plt.tight_layout()
    if save:
        plt.savefig(output_name, transparent=True)
    plt.show()
    plt.close()


if __name__ == "__main__":

    ref = get_spectrum_by_scan('target_drugs/ibuprofen-carnitine/mzml/100nM_STANDARD_MIX.mzML', [1917])[0]
    bio = get_spectrum_by_scan('target_drugs/ibuprofen-carnitine/mzml/IBD_BIOLOGICAL_2.mzML', [1781])[0]

    peaks_1 = np.column_stack((ref['mz_ls'], ref['intensity_ls']))
    peaks_2 = np.column_stack((bio['mz_ls'], bio['intensity_ls']))
    create_mirror_plot(peaks_1, peaks_2, 
                       fig_size=(2.1, 1.3),
                       ms2_tol=0.05, peak_int_power=1.0,
                       intensity_threshold_label=50, label_decimals=2,
                       max_x=385,
                       up_matched_peak_color='#c7522a', down_matched_peak_color='#008585',
                       up_spec_peak_color='0.6', down_spec_peak_color='0.6',
                       show_unmatched_peaks=True, title=None,
                       save=True, output_name='target_drugs/ibuprofen-carnitine/plots/mirror_plot.svg')


    from matchms import Spectrum
    from matchms.similarity import CosineGreedy

    spec_1 = Spectrum(mz=ref['mz_ls'], intensities=ref['intensity_ls'], metadata={'precursor_mz': 350.2326})
    spec_2 = Spectrum(mz=bio['mz_ls'], intensities=bio['intensity_ls'], metadata={'precursor_mz': 350.2326})

    # construct a similarity function
    cosine_greedy = CosineGreedy(tolerance=0.05)

    score = cosine_greedy.pair(spec_1, spec_2)
    print(f"Cosine similarity: {score}")