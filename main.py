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
    dir_name : string, optional
        Name of directory to write result in. Default is './'
    return_data : bool, optional
        If True, return the resulting array instead of writing into file.
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
    dir_name : string, optional
        Name of directory to write result in. Default is './'
    return_data : bool, optional
        If True, return the resulting array instead of writing into file.
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
    band = band.upper()
    obs = fits.getdata(file_name)
    dx = 40
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
    sum_obs = np.array(list(map(lambda a, b: np.ma.sum(obs[a, b:b + dx]), y, x)))
    wavelenght = np.polyval(p, y)
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


def extract_spectra(file_name, return_data=False, dir_name='./', name=None):
    """Extract spectra from ASTRONIRCAM fits image.

    Apply extract_spectra to both of the bands existing on an image.

    Parameters
    ----------
    file_name : string
        Path to file
    band : string
        Must be 'Y', 'J', 'H' or 'K'

    Returns
    -------
    specrum : ndarray
        First row is an array of wavelenghts, second is an array of
        fluxes corresponded to each wavelenght.
    """
    data_header = fits.getheader(file_name)
    if data_header['UPPER'].count('YJ'):
        Y_spectra = extract_spectrum(file_name, 'Y', return_data=return_data, dir_name=dir_name)
        J_spectra = extract_spectrum(file_name, 'J', return_data=return_data, dir_name=dir_name)
        return [Y_spectra, J_spectra]
    elif data_header['UPPER'].count('HK'):
        H_spectra = extract_spectrum(file_name, 'H', return_data=return_data, dir_name=dir_name)
        K_spectra = extract_spectrum(file_name, 'K', return_data=return_data, dir_name=dir_name)
        return [H_spectra, K_spectra]
    else:
        # print("Can't find mentiond band in the UPPER string of the header of the file.")
        sys.exit(1)


def get_magnitudes(hip_id, catalogue='~/Documents/SAI/atm-tran/transmittance/A0V.csv'):
    cat = pd.read_csv(catalogue, sep='\s+')
    hip_id = str(hip_id).lower()
    if hip_id.isnumeric():
        hip_id = 'hip' + hip_id
    star = cat.loc[cat['HIP_ID_STR'] == hip_id]
    magnitudes = {'J': star['FLUX_J'].values[0], 'H': star['FLUX_H'].values[0],
                  'K': star['FLUX_K'].values[0]}
    return magnitudes


def clear_spectrum(spec_name, sky_name, return_data=False, dir_name='./', name=None):
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


def clear_spectra(list_of_names):
    all = set(list_of_names)        # Names of both sky and stars spectra
    r = re.compile('.*SKY.*')
    sky = set(filter(r.match, list_of_names))   # Only names of sky spectra
    stars = list(all - sky)         # Only names of stars spectra
    sky = list(sky)
    stars.sort()        # Sort both of name lists
    sky.sort()          # To make index of sky and star specrum of same object (in same band)
    res = list(map(clear_spectrum, stars, sky))     # Get clean spectrum for every star in every band
    return res


def airmass(zt):
    c = np.cos(np.radians(zt))
    k = [1.002432, 0.148386, 0.0096467, 0.149864, 0.0102963, 0.000303978]
    return ((k[0] * c ** 2 + k[1] * c + k[2]) / (c ** 3 + k[3] * c ** 2 + k[4] * c + k[5]))


def corr2(a, b, mmv=10):
    ''' len(a) shouldn't be less than len(b) '''
    b = b[mmv: -mmv]
    move = np.argmax(np.correlate(a, b))
    return move


def max_corr(data, mv=10):
    ''' Returns corellated data, number of reference line and ref movement
        data - 1-D lines of numbers (y-values with the same x[0])
        mv - maximum movement, every line will loose at least 2*mmv elemets
    '''

    len_data = list(map(len, data))     # Делаем массив длин
    print("len:" + str(len_data))
    i_ref = np.argmax(len_data)         # Получаем номер опорного массива (с наибольшим количеством) элементов
    ref_data = data[i_ref]
    del data[i_ref]
    move = np.array(list(map(lambda x: corr2(ref_data, x, mmv=mv), data)))
    ref_move = np.max(move)
    move = (move * (-1) + ref_move).tolist()
    data = list(map(lambda x: x[mv:-mv], data))
    data.insert(i_ref, ref_data)
    move.insert(i_ref, ref_move)
    data = list(map(lambda x, y: x[y:], data, move))
    max_len = np.min(list(map(len, data)))
    data = np.array(list(map(lambda x: x[:max_len], data)))
    return data, i_ref, ref_move


def transmission(data, M):
    data = np.log((data / data[0])[1:])
    M = np.array([(M - M[0])[1:]])
    k = np.linalg.lstsq(M.T, data, rcond=None)[0]
    return k


def get_transmittance(args):
    # Load all spectra
    all_data = list(map(lambda x: fits.open(x)[1], args))
    data = list(map(lambda x: x.data.field(1), all_data))
    # Cut all spectra to universal beginning
    # data, i_ref, ref_move = max_corr(data)
    # wl = (all_data[i_ref].data.field(0))[ref_move: np.shape(data)[1] + ref_move]
    wl = all_data[0].data.field(0)
    h = np.array(list(map(lambda x: x.header['CURALT'], all_data)))
    M = airmass(90 - h)
    p = np.exp(transmission(data, M)).T
    return(np.array([wl, p.flatten()]))


def get_all_transmittance(filenames):
    r = re.compile('.*_Y_.*')
    Y = get_transmittance(list(filter(r.match, filenames)))
    r = re.compile('.*_J_.*')
    J = get_transmittance(list(filter(r.match, filenames)))
    r = re.compile('.*_H_.*')
    H = get_transmittance(list(filter(r.match, filenames)))
    r = re.compile('.*_K_.*')
    K = get_transmittance(list(filter(r.match, filenames)))
    a = list(map(lambda x: plt.plot(x[0], x[1]), [Y, J, H, K]))
    plt.show()


def main(args):
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
    clean = clear_spectra(spectra)
    # print(clean)

    # Get transmittance for every band
    get_all_transmittance(clean)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
