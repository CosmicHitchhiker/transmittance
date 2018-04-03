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
        Prefered name for result.


    Returns
    -------
    name : string
        Name of generated fits file. (Returned when return_data is False)
    res : dictionary
        Dictionary with name as keyword and result as value. (Returned
        when return_data is True)
    """
    list_of_names = glob.glob(name_mask)
    all_data = np.array(list(map(lambda x: fits.getdata(x), list_of_names)))
    res_data = np.mean(all_data, axis=0)
    if (not name):
        new_header = fits.getheader(list_of_names[0])
        name = new_header['TARNAME'] + '_' + new_header['UPPER'] + '.fits'
        new_header['SKY'] = 'No'
        if (list_of_names[0].upper().count('SKY')):
            name = 'SKY_' + name
            new_header['SKY'] = 'Yes'
    if (return_data):
        return {name: res_data}
    else:
        name = dir_name + name
        fits.writeto(name, res_data, header=new_header)
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

    Returns
    -------
    specrum : ndarray
        First row is an array of wavelenghts, second is an array of
        fluxes corresponded to each wavelenght.
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
    print(c)
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
        fits.BinTableHDU(data=table, header=hdr).writeto(dir_name + name)
        return name


def extract_spectra(file_name, return_data=False, dir_name='./', name=None):
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
        print("Can't find mentiond band in the UPPER string of the header of the file.")
        sys.exit(1)


def get_magnitudes(hip_id, catalogue='./A0V.csv'):
    cat = pd.read_csv(catalogue, sep='\s+')
    hip_id = str(hip_id).lower()
    if hip_id.isnumeric():
        hip_id = 'hip' + hip_id
    star = cat.loc[cat['HIP_ID_STR'] == hip_id]
    magnitudes = {'J': star['FLUX_J'].values[0], 'H': star['FLUX_H'].values[0],
                  'K': star['FLUX_K'].values[0]}
    return magnitudes


def clear_spectrum(spec_name, sky_name, return_data=False, dir_name='./', name=None):
    spec = fits.open(spec_name)[1].data
    sky = fits.open(sky_name)[1].data
    data = spec.field(1) - sky.field(1)
    data[data < 0] = 0
    data = sig.medfilt(data, 5)
    data = sig.wiener(data, 10)
    hdr = fits.getheader(spec_name, 1)
    band = hdr['BAND']
    if band == 'Y':
        band = 'J'
    mag = get_magnitudes(hdr['TARNAME'])[band]
    data = data * 100 ** (0.2 * (mag - 5.0))
    spectrum = [spec.field(0), data]
    if return_data:
        return spectrum
    else:
        table = Table(spectrum, names=('wavelenght', 'flux'))
        name = hdr['TARNAME'] + '_' + hdr['BAND'] + '_CLEAR.fits'
        fits.BinTableHDU(data=table, header=hdr).writeto(dir_name + name)
        return name


def clear_spectra(list_of_names):
    return None


def main(args):
    files = glob.glob(sys.argv[1] + "*.fts")
    # # We do not need last 4 parts of name
    files_mask = set(list(map(lambda x: '-'.join(x.split('-')[:-4]) + '*',
                              files)))
    mean_raw = list(map(average_fits, list(files_mask)))
    print(mean_raw)

    spectra = list(map(extract_spectra, mean_raw))
    print(spectra)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
