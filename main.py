import numpy as np
from matplotlib import pyplot as plt
import sys
import glob
import os
from astropy.io import fits, ascii
import pandas as pd
import scipy.signal as sig
from math import *


def average_fits(name_mask, dir_name='./', return_data=False, name=None):
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
        name = new_header['TARNAME'] + '_' + new_header['UPPER'] + '_' + new_header['CURALT'] + '.fits'
        if (list_of_names[0].upper().count('SKY')):
            name = 'SKY_' + name
    if (return_data):
        return {name: res_data}
    else:
        name = dir_name + name
        fits.writeto(name, res_data, header=new_header)
        return name


def extract_spectrum(file_name, band):
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
    wavelenght : ndarray
        Float array of wavelenghts.
    sum_obs : ndarray
        Array of summarized flux for each represented wavelenght.
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
    return wavelenght, sum_obs


def get_magnitudes(hip_id, catalogue='./A0V.csv'):
    cat = pd.read_csv(catalogue, sep='\s+')
    hip_id = str(hip_id).lower()
    if hip_id.isnumeric():
        hip_id = 'hip' + hip_id
    star = cat.loc[cat['HIP_ID_STR'] == name.lower()]
    magnitudes = {'J': star['FLUX_J'].values[0], 'H': star['FLUX_H'].values[0],
                  'K': star['FLUX_K'].values[0]}
    return magnitudes


def main(args):
    files = glob.glob(sys.argv[1] + "*.fts")
    # # We do not need last 4 parts of name
    files_mask = set(list(map(lambda x: '-'.join(x.split('-')[:-4]) + '*',
                              files)))
    print(list(map(average_fits, list(files_mask))))

    extract_spectrum(args[1], args[2])
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
