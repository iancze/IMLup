from astropy.io import fits
from astropy.constants import c
import numpy as np
import casatasks
from aksco_common import constants

from aksco_common.image_utils import clear_extensions, exportfits


def dirty_image_cont(ms, imagename, specmode, robust=0.0, imsize=3000, cell=0.003, restfreq=constants.f12):
    """Make a zero-iteration CLEAN image of the whole field for inspection purposes.
    Args:
        ms (string): input measurement set
        imagename (string): base string for image outputs
        specmode (string): mode to use for imaging
        robust (float): value between -2.0 and 2.0
        
    Returns:
        None
    """
    
    clear_extensions(imagename)

    casatasks.tclean(vis=ms,
        imagename=imagename,
        specmode=specmode,
        deconvolver="multiscale",
        scales=[0, 5, 30, 100, 200],
        weighting="briggs",
        robust=robust,
        imsize=imsize,
        cell="{:.5f}arcsec".format(cell),
        niter=0,
        nterms=1,
        restfreq="{:.8f}Hz".format(restfreq) if specmode=="cube" else "")

    exportfits(imagename + ".image", imagename + ".fits")


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Make a dirty image of the measurement set.")
    
    parser.add_argument("ms")
    parser.add_argument("imagename", default="", help="Base of output, e.g. `mydir/myimage`")
    parser.add_argument("--robust", default=0.0, type=float, help="Briggs Robust Value, between -2.0 and 2.0.")
    parser.add_argument("--imsize", type=int, default=6000, help="Number of pixels on each side of image.")
    parser.add_argument("--cell", type=float, default=0.003, help="Pixel size in arcseconds")
    parser.add_argument("--specmode", choices=["mfs", "cube"], default="mfs", help="Which imaging mode to use. 'mfs' is for continuum, 'cube' is for line.")
    parser.add_argument(
        "--restfreq", type=float, default=constants.f12, help="Default 12CO J=2-1."
    )
    args = parser.parse_args()
    
    dirty_image_cont(args.ms, args.imagename, args.specmode, args.robust, imsize=args.imsize, cell=args.cell, restfreq=args.restfreq)

if __name__=="__main__":
    main()
