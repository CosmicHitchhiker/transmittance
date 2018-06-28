import numpy as np
from matplotlib import pyplot as plt
import sys
import glob
import os
from astropy.io import fits, ascii
from astropy.table import Table
import pandas as pd
import scipy.signal as sig
from math import *
import re
from scipy.ndimage.interpolation import shift


def average_fits(name_mask, return_data=False, dir_name='./', name=None):
    """Average several fits images.

    Take an array of fits images and apply numpy.mean to it.

    Parameters
    ----------
    name_mask : string
        Mask for glob function to get names of fits files.
    return_data : bool, optional
        If True, return the resulting array instead of writing into file.
    dir_name : string, optional
        Name of directory to write result in. Default is './'
    name : string, optional
        Prefered name for the result.


    Returns
    -------
    name : string
        Name of generated fits file. (Returned when return_data is False)
    res : dictionary
        Dictionary with name as keyword and result as value. (Returned
        when return_data is True)
    """
    list_of_names = glob.glob(name_mask)
    # 3D array with image as every 2D component
    all_data = np.array(list(map(lambda x: fits.getdata(x), list_of_names)))
    # Averaged 2D image
    res_data = np.mean(all_data, axis=0)
    if (not name):      # Create name if not mentioned
        # Result header is equal to the header of fitrst file
        new_header = fits.getheader(list_of_names[0])
        name = new_header['TARNAME'] + '_' + new_header['UPPER'] + '.fits'
        if (list_of_names[0].upper().count('SKY')):
            name = 'SKY_' + name
    if (return_data):
        return {name: res_data}
    else:
        name = dir_name + name
        fits.writeto(name, res_data, header=new_header, overwrite=True)
        return name


def extract_spectrum(file_name, band, return_data=True, dir_name='./', name=None):
    """Extract spectrum from ASTRONIRCAM spectrograph image.

    Open fits file with data from ASTRONIRCAM. Fit curve with spectrum
    by polynom of degree 2, summarize flux from each row of this curve.
    Then transfom y-coordinate to wavelenght with dispersional equation
    of degree 4 for Y, J and H bands and of degree 5 for K band. Return
    summarized flux for every wavelenght.

    Parameters
    ----------
    file_name : string
        Path to file
    band : string
        Must be 'Y', 'J', 'H' or 'K'
    return_data : bool, optional
        If True, return the resulting array instead of writing into file.
    dir_name : string, optional
        Name of directory to write result in. Default is './'
    name : string, optional
        Prefered name for the result.

    Returns
    -------
    specrum : ndarray
        First row is an array of wavelenghts, second is an array of
        fluxes corresponded to each wavelenght. (Returned when return_data
        is True)
    name : string
        Name of generated fits file. (Returned when return_data is False)
    """
    band = band.upper()     # To avoid errors in letter size
    obs = fits.getdata(file_name)
    dx = 40         # Width of field to summarize
    # c - coefficients for curve-fit equation
    # p - coefficients for dispersional equation
    if band == 'Y':
        c = [2.3456E-6, 0.30176, 518.70]
        p = [7.6050E-15, -4.4590E-11, 1.0088E-7, 1.2629E-4, 0.90406]
    elif band == 'J':
        c = [1.4842e-5, 0.33344, 757.65]
        p = [9.1734E-16, -1.1372E-11, 3.9892E-8, 2.2548E-4, 1.0454]
    elif band == 'H':
        c = [4.9498E-6, 0.16552, 712.99]
        p = [-4.6344E-15, 1.2196E-11, 5.1964E-9, 3.2047E-4, 1.2856]
    elif band == 'K':
        c = [9.8195E-6, 0.21731, 936.31]
        p = [-9.2514E-18, 4.3516E-14, -9.0009E-11, 1.1712E-7, 3.7964E-4, 1.7039]
    else:
        print("Band must be Y,J,H or K (but not ", band, ")!")
        sys.exit(1)
    y = np.arange(20, 2020, dtype=int)
    x = np.array(np.polyval(c, y), dtype=int)
    # summarize flux in significant pixels of necessary range
    sum_obs = np.array(list(map(lambda a, b: np.ma.sum(obs[a, b:b + dx]), y, x)))
    wavelenght = np.polyval(p, y)       # Transform y to lambda by dispersional equation
    spectrum = [wavelenght, sum_obs]
    if return_data:
        return np.array(spectrum)
    else:
        table = Table(spectrum, names=('wavelenght', 'flux'))
        hdr = fits.getheader(file_name)
        hdr['BAND'] = band
        name = hdr['TARNAME'] + '_' + band + '.fits'
        if file_name.upper().count('SKY'):
            name = 'SKY_' + name
        fits.BinTableHDU(data=table, header=hdr).writeto(dir_name + name, overwrite=True)
        return name


def extract_spectra(file_name):
    """Extract spectra from ASTRONIRCAM fits image.

    Apply extract_spectra to both of the bands existing on an image (according
    to the fits header information)

    Parameters
    ----------
    file_name : string
        Path to file

    Returns
    -------
    specrum : ndarray
        First row is an array of wavelenghts, second is an array of
        fluxes corresponded to each wavelenght.
    """
    data_header = fits.getheader(file_name)
    if data_header['UPPER'].count('YJ'):
        Y_spectra = extract_spectrum(file_name, 'Y', return_data=False)
        J_spectra = extract_spectrum(file_name, 'J', return_data=False)
        return [Y_spectra, J_spectra]
    elif data_header['UPPER'].count('HK'):
        H_spectra = extract_spectrum(file_name, 'H', return_data=False)
        K_spectra = extract_spectrum(file_name, 'K', return_data=False)
        return [H_spectra, K_spectra]
    else:
        print("Can't find mentiond band in the UPPER string of the header of the file.")
        sys.exit(1)


def get_magnitudes(hip_id, catalogue='A0V.csv'):
    """Get J, H and K magnitudes from the mentioned catalogue.

    It's highly recomended to use telluric standarts catalogue from
    the author's github repo:
    https://github.com/CosmicHitchhiker/transmittance/blob/master/A0V.csv

    Parameters
    ----------
    hip_id : string or int
        Id of the star in HIPPARCOS catalogue
    catalogue: string, optional
        Path to the catalogue file

    Returns
    -------
    magnitudes : dictionary
        Dictionary with the band names ('J', 'H', 'K') as keys and
        corresponding magnitudes as values
    """
    cat = pd.read_csv(catalogue, sep='\s+')  # read catalogue in pandas dataframe
    hip_id = str(hip_id).lower()   # format of star name in catalogue is 'hip<NUMBER'
    if hip_id.isnumeric():      # if ony humber is given, make it 'hip<NUMBER>'
        hip_id = 'hip' + hip_id
    star = cat.loc[cat['HIP_ID_STR'] == hip_id]     # find star in catalogue
    magnitudes = {'J': star['FLUX_J'].values[0], 'H': star['FLUX_H'].values[0],
                  'K': star['FLUX_K'].values[0]}
    return magnitudes


def clean_spectrum(spec_name, sky_name, return_data=False, dir_name='./'):
    """Subtract sky spectrum, remove noise, normalize strar specrum

    Open mentioned sky and star spectra, subtract sky from sky spectrum.
    Fix some bugs (negative values), then apply median and wiener filter
    to the given data. Take magnitude for the band mentioned in header (because
    of lack of Y magnitude in catalogue, it's set to J magnitude) and transform
    flux as it would be star of 5th magnitude. Then cut spectrum to the band
    borders.

    Parameters
    ----------
    spec_name : string
        Path to star spectrum (assuming that it's in fits-table format)
    sky_name : string
        Path to sky spectrum
    return_data : bool, optional
        If True, return the resulting spectrum instead of writing into file.
    dir_name : string, optional
        Name of directory to write result in. Default is './'

    Returns
    -------
    specrum : ndarray
        First row is an array of wavelenghts, second is an array of
        fluxes corresponded to each wavelenght. (Returned when return_data
        is True)
    name : string
        Name of generated fits file. (Returned when return_data is False)
    """
    # Borders for every band
    Y_band = [0.95, 1.3]
    J_band = [1.15, 1.35]
    H_band = [1.50, 1.80]
    K_band = [1.95, 2.40]
    borders = {'Y': Y_band, 'J': J_band, 'H': H_band, 'K': K_band}

    spec = fits.open(spec_name)[1].data     # Spectra of star
    sky = fits.open(sky_name)[1].data       # Spectra of sky
    data = spec.field(1) - sky.field(1)     # Just remove sky from star spectrum
    wavelenghts = spec.field(0)
    data[data < 0] = 0          # Remove negative values
    data = sig.medfilt(data, 5)         # Simple median filter
    data = sig.wiener(data, 10)         # Wiener filter (removes sharp noise)

    # Nomalize spectrum according to the band, mentioned in header
    # Magnitude for this star must be written in A0V catalogue file`
    hdr = fits.getheader(spec_name, 1)
    band = hdr['BAND']
    if band == 'Y':
        mag = get_magnitudes(hdr['TARNAME'])['J']
    else:
        mag = get_magnitudes(hdr['TARNAME'])[band]
    # Let this star have magnitude equal to 5!
    data = data * 100 ** (0.2 * (mag - 5.0))

    # Cut spectrum
    data = data[wavelenghts > borders[band][0]]
    wavelenghts = wavelenghts[wavelenghts > borders[band][0]]
    data = data[wavelenghts < borders[band][1]]
    wavelenghts = wavelenghts[wavelenghts < borders[band][1]]

    # Write spectrum as a "list" structure
    spectrum = [wavelenghts, data]
    if return_data:
        # return list with wavelenghts in the first position and normalized flux in the second
        return spectrum
    else:
        # Create fits table with wavelenghts and coresponding flux
        table = Table(spectrum, names=('wavelenght', 'flux'))
        # Name is <TARNAME>_<BAND>_CLEAR.fits (for example "HIP007_Y_CLEAR.fits")
        name = hdr['TARNAME'] + '_' + hdr['BAND'] + '_CLEAR.fits'
        fits.BinTableHDU(data=table, header=hdr).writeto(dir_name + name, overwrite=True)
        # Return name of the created fits
        return name


def clean_spectra(list_of_names):
    """ Apply clean_spectrum to every star in the given files.

    Assume that in the given list of names the number of
    sky spectra and star spectra is equal, and the only difference is the
    word SKY in the names of sky spectra. Separate sky and stars and apply
    clean_spectrum to the sorted list of sky and stars spectra names.

    Parameters
    ----------
    list_of_names : list of strings
        List with the filenames of all necessary spectra (both sky and star)

    Returns
    -------
    res : list of strings
        List with filenames of resulting clear spectra
    """

    all = set(list_of_names)        # Names of both sky and stars spectra
    r = re.compile('.*SKY.*')
    sky = set(filter(r.match, list_of_names))   # Only names of sky spectra
    stars = list(all - sky)         # Only names of stars spectra
    sky = list(sky)
    stars.sort()        # Sort both of name lists
    sky.sort()          # To make index of sky and star specrum of same object (in same band)
    res = list(map(clean_spectrum, stars, sky))     # Get clear spectrum for every star in every band
    return res


def airmass(zt):
    """Calculate airmass on the given coaltitude (zenith distance)

    Parameters
    ----------
    zt : float
        Coaltitude

    Returns
    -------
    airmass : float
        Calculated airmass
    """
    c = np.cos(np.radians(zt))
    k = [1.002432, 0.148386, 0.0096467, 0.149864, 0.0102963, 0.000303978]
    return ((k[0] * c ** 2 + k[1] * c + k[2]) / (c ** 3 + k[3] * c ** 2 + k[4] * c + k[5]))


def transmittance(data, M):
    """Calculate transmittance for every wavelenght in given data.

    See "theory" section in the documentation for explanation of this function.

    Parameters
    ----------
    data : 2D list or ndarray
        array of fluxes of every star
    M : 1D list or ndarray
        array of airmasses of every star

    Returns
    -------
    k : ndarray
        optical depth in zenith for every wavelenght
    """
    data = np.log((data / data[0])[1:])     # Left part of linear equation
    M = np.array([(M - M[0])[1:]])          # Right part of linear equation
    k = np.linalg.lstsq(M.T, data, rcond=None)[0]   # Solution of equation
    return k


def get_transmittance(filenames):
    """Calculate transmittance in one band.

    Load spectra from fits files, calculate airmass (get altitude from header)
    and calculate transmittance for every wavelenght.

    Parameters
    ----------
    filenames : list of strings
        Names of files with clear stars spectra

    Returns
    -------
    2D ndarray
        first row is wavelenghts, second is transmittance
    """
    # Load all spectra
    all_data = list(map(lambda x: fits.open(x)[1], filenames))
    data = list(map(lambda x: x.data.field(1), all_data))  # Flux arrays of every star
    wl = all_data[0].data.field(0)      # Wavelenghts (should be equal for all stars)
    h = np.array(list(map(lambda x: x.header['CURALT'], all_data)))  # Altitudes
    M = airmass(90 - h)     # Airmasses
    p = np.exp(transmittance(data, M)).T        # Transmittance
    return(np.array([wl, p.flatten()]))


def get_all_transmittance(filenames, plot=True, save=True, dir_name='./'):
    """Calculate transmittance in all bands.

    Just apply get_transmittance to all band one-by-one.

    Parameters
    ----------
    filenames : list of strings
        Names of files with clear stars spectra
    plot : bool, optional
        if True, transmittance will be plotted
    save : bool, optional
        if True, results will be saved in fits tables
    dir_name : string, optional
        where to save the files

    Returns
    -------
    names : list of names
        names of saved files (only if 'save' is True)
    """
    res = dict.fromkeys(['Y', 'J', 'H', 'K'])
    for band in res.keys():
        r = re.compile('.*_' + band + '_.*')
        res[band] = get_transmittance(list(filter(r.match, filenames)))

    if plot:
        a = list(map(lambda x, y: plt.plot(x[0], x[1], label=y), res.values(), res.keys()))
        plt.legend()
        plt.show()
    if save:
        names = []
        for band in res.keys():
            print(res[band])
            table = Table(np.array(res[band]).T, names=('wavelenght', 'transmittance'))
            name = band + '_TRANSMITTANCE.fits'
            fits.BinTableHDU(data=table).writeto(dir_name + name, overwrite=True)
            names.append(name)
        return names


def main(args):
    """First argument is the name of directory with raw fits files"""

    # Take all "fts" files from the mentioned directory
    files = glob.glob(sys.argv[1] + "*.fts")
    # We do not need last 4 parts of name
    files_mask = set(list(map(lambda x: '-'.join(x.split('-')[:-4]) + '*',
                              files)))
    # Get mean fits for every object
    mean_raw = list(map(average_fits, list(files_mask)))
    # print(mean_raw)

    # Extract all spectra for every object
    spectra = (np.array(list(map(extract_spectra, mean_raw))).flatten()).tolist()
    # print(spectra)

    # Remove noise, sky and normalize spectra
    clear = clean_spectra(spectra)
    # print(clear)

    # Get transmittance for every band
    get_all_transmittance(clear)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
