"""
Compute massflux parametrisation from radar data using Kumar et al. 2015 paper.

@title: massflux
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Monash University
@date: 17/05/2019
@version: 1

.. autosummary::
    :toctree: generated/

    chunks
    main
"""

import os
import glob
import uuid
import datetime

import netCDF4
import numpy as np
import xarray as xr


def make_param(eth_conv, zhwt_conv, heights):
    wu = np.zeros((len(zhwt_conv), len(heights))) + np.NaN
    wu_mean= np.zeros((len(zhwt_conv))) + np.NaN
    Tz = np.zeros((len(zhwt_conv), len(heights)))

    height_2d = np.tile(heights, (wu.shape[0], 1))
    eth_conv_tile = np.tile(eth_conv, (len(heights), 1)).T

    wu[(eth_conv < 7e3), :] = 0.040 * heights + 0.992
    wu[(eth_conv >= 7e3) & (eth_conv < 15e3), :] = -0.0016 * heights ** 4 + 0.052 * heights ** 3 - 0.571 * heights ** 2 + 2.7 * heights - 2.735
    wu[(eth_conv >= 15e3), :] = -0.045 * heights ** 2 + 1.089 * heights - 0.896
    wu[wu <= 0] = np.NaN
    wu[height_2d > eth_conv_tile / 1e3] = np.NaN

    wu_mean = np.nanmean(wu, axis=1)

    wd_tmp = 0.0339 * heights ** 2 + 0.4109 * heights - 1.6852
    wd = np.tile(wd_tmp, (wu.shape[0], 1))
    wd[wd > 0] = np.NaN

    wres = 4.391 - 1.238 * (eth_conv / 1e3) + (-0.061 + 0.021 * eth_conv / 1e3) * zhwt_conv
    for idx, h in enumerate(heights):
        Tz[:, idx] = (wres + wu_mean) / (wu_mean) * (wu[:, idx] / wu_mean) ** 0.5

    wu[np.isnan(wu)] = 0
    wd[np.isnan(wd)] = 0
    Tz[np.isnan(Tz)] = 0

    wtot = (wu * Tz + wd)
    wtot[heights > (eth_conv_tile / 1e3)] = np.NaN
    wtot = np.ma.masked_invalid(wtot)
    return wtot


def massflux(infile_stein, infile_eth, infile_zhwt, heights):
    with netCDF4.Dataset(infile_stein) as ncid:
        stein = ncid['steiner_echo_classification'][:]

    with netCDF4.Dataset(infile_eth) as ncid:
        eth = ncid['echo_top_height'][:]

    with netCDF4.Dataset(infile_zhwt) as ncid:
        zhwt = ncid['height_weighted_sum_reflectivity'][:]

    pos_conv = (stein == 2) & (eth > 2.5e3)
    eth_conv = eth[pos_conv]
    zhwt_conv = zhwt[pos_conv]

    wtot = make_param(eth_conv, zhwt_conv, heights)

    pos_tot = np.zeros((144, 117, 117, len(heights)), dtype=pos_conv.dtype)
    for cnt in range(41):
        pos_tot[:, :, :, cnt] = pos_conv

    wprof = np.zeros((144, 117, 117, len(heights))) + np.NaN
    wprof[pos_conv] = wtot
    wprof = np.ma.masked_invalid(wprof)

    return wprof


def make_daily(infile_stein, infile_eth, infile_zhwt, output_directory):
    # Read Input file and get date
    datas = xr.open_dataset(infile_zhwt)
    datestr = infile_stein[-11:-3]
    outfilename = f"twp1440cpol.massflux.c1.{datestr}.nc"
    outfilename = os.path.join(output_directory, outfilename)
    if os.path.isfile(outfilename):
        print(f"{outfilename} already exists.")
        return None

    # Compute massflux
    wprof = massflux(infile_stein, infile_eth, infile_zhwt, np.linspace(0, 20, 41))  # height in km

    metadata = datas.attrs.copy()
    metadata['version'] = "2019.04_level2"
    metadata['created'] = datetime.datetime.now().isoformat()
    metadata['uuid'] = str(uuid.uuid4())
    metadata['creator_email'] = "valentin.louf@monash.edu"
    metadata['institution'] = "Monash University"

    ndata = xr.Dataset({'x': datas.x,
                        'y': datas.y,
                        'z': (('z'), np.linspace(0, 20000, 41, dtype=np.int32)),
                        'time': datas.time,
                        "massflux": (('time', 'y', 'x', 'z'), wprof.astype(np.float32)),
                        'isfile': datas.isfile,
                        'latitude': datas.latitude,
                        'longitude': datas.longitude,})

    ndata.z.attrs["standard_name"] = "projection_z_coordinate"
    ndata.z.attrs["long_name"] = "Height above radar"
    ndata.z.attrs["units"] = "m"

    ndata.massflux.attrs['_FillValue'] = np.NAN
    ndata.massflux.attrs['short_name'] = "massflux"
    ndata.massflux.attrs['long_name'] = "radar_estimated_convective_mass_flux"
    ndata.massflux.attrs['units'] = "m s-1"
    ndata.massflux.attrs['reference'] = "doi: 10.1175/JAMC-D-15-0193.1"
    ndata.massflux.attrs['comments'] = "Estimation of the convective mass flux using V. Kumar parameterization."

    ndata.attrs = metadata
    args = dict()
    for k in ['massflux', ]:
        args[k] = {'zlib': True, 'least_significant_digit': 4}

    ndata.to_netcdf(outfilename, encoding=args)
    return None