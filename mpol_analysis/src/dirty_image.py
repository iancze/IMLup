from mpol import coordinates, gridding
import numpy as np
from src import loaddata

def main():

    uu, vv, data, weight = loaddata.get_basic_data()

    coords = coordinates.GridCoords(cell_size=0.005, npix=800)
    imager = gridding.DirtyImager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=np.real(data),
        data_im=np.imag(data),
    )


    img, beam = imager.get_dirty_image(weighting="briggs", robust=0.0)

if __name__=="__main__":
    main()