"""
Compute massflux parametrisation from radar data using Kumar et al. 2015 paper.

@title: radar_pack
@author: Valentin Louf <valentin.louf@monash.edu>
@institution: Monash University
@date: 17/05/2019
@version: 1

.. autosummary::
    :toctree: generated/

    chunks
    main
"""
# Python Standard Library
import os
import sys
import glob
import time
import argparse
import datetime
import warnings
import traceback

import crayons
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main(inargs):
    """
    It calls the production line and manages it. Buffer function that is used
    to catch any problem with the processing line without screwing the whole
    multiprocessing stuff.

    Parameters:
    ===========
    infile: str
        Name of the input radar file.
    outpath: str
        Path for saving output data.
    """
    infile_stein, infile_eth, infile_zhwt, output_directory = inargs

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import matplotlib
        matplotlib.use('Agg')

        import massflux

        try:
            massflux.make_daily(infile_stein, infile_eth, infile_zhwt, output_directory)
        except Exception:
            traceback.print_exc()
            return None

    return None


if __name__ == '__main__':
    """
    Global variables definition.
    """
    # Parse arguments
    parser_description = "Processing of radar data from level 1a to level 1b."
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        '-o',
        '--output-dir',
        dest='outdir',
        default="/g/data/hj10/cpol_level_2/v2018/grid_150km_2500m/reflectivity_vertical_profile/",
        type=str,
        help='Output directory.')
    parser.add_argument(
        '-n',
        '--ncpu',
        dest='ncpu',
        default=16,
        type=int,
        help='Number of CPUs for multiprocessing.')

    args = parser.parse_args()
    OUTPATH = args.outdir
    NCPU = args.ncpu

    fstein = sorted(glob.glob('/g/data/hj10/cpol_level_2/v2018/grid_150km_2500m/STEINER_ECHO_CLASSIFICATION/*.nc'))
    feth = sorted(glob.glob('/g/data/hj10/cpol_level_2/v2018/grid_150km_2500m/ECHO_TOP_HEIGHT/*.nc'))
    fzhwt = sorted(glob.glob('/g/data/hj10/cpol_level_2/v2019/grid_150km_2500m/height_weighted_sum_reflectivity/*.nc'))

    if len(fstein) != len(feth) or len(feth) != len(fzhwt):
        print("Not the right number of files.")
        sys.exit()
    if len(fstein) == 0:
        print('No file found.')
        sys.exit()

    print(f'{len(fstein)} files found.')
    arglist = [(a, b, c, OUTPATH) for a, b, c in zip(fstein, feth, fzhwt)]

    sttime = time.time()
    for chunk in chunks(arglist, NCPU * 2):
        with ProcessPool(max_workers=NCPU) as pool:
            future = pool.map(main, chunk, timeout=120)
            iterator = future.result()
            while True:
                try:
                    result = next(iterator)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print("function took longer than %d seconds" % error.args[1])
                except ProcessExpired as error:
                    print("%s. Exit code: %d" % (error, error.exitcode))
                except Exception as error:
                    print("function raised %s" % error)
                    print(error.traceback)  # Python's traceback of remote process

    print(crayons.green(f"Process completed in {time.time() - sttime:0.2f}."))