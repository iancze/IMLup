from mpol import coordinates, gridding
import numpy as np
from src import loaddata
import argparse

def init_dirty_imager(filename):
    uu, vv, data, weight = loaddata.get_basic_data(filename)

    coords = coordinates.GridCoords(cell_size=0.005, npix=800)
    return gridding.DirtyImager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=np.real(data),
        data_im=np.imag(data),
    )

def main():

    parser = argparse.ArgumentParser(
        description="Make a dirty image with the visibilities."
    )
    parser.add_argument("file", help="Path to asdf file")
    parser.add_argument("outfile", help="Output file") 
    args = parser.parse_args()
    
    imager = init_dirty_imager(args.file)

    img, beam = imager.get_dirty_image(weighting="briggs", robust=0.0)

if __name__=="__main__":
    main()