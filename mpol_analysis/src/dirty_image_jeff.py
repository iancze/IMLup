from mpol import coordinates, gridding
import numpy as np
from src import loaddata
import argparse

def main():

    parser = argparse.ArgumentParser(
        description="Make a dirty image with the visibilities."
    )
    parser.add_argument("file", help="Path to asdf file")
    args = parser.parse_args()
        
    uu, vv, vis, weight = loaddata.get_basic_data(args.file)
    data_re, data_im = vis.real, vis.imag

    cell_size, npix = 0.005103605411684434, 940
    coords = coordinates.GridCoords(cell_size=cell_size, npix=npix)

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im
    )

    imager = gridding.DirtyImager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im
    )


if __name__=="__main__":
    main()