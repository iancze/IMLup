from astropy.io import fits
from astropy.constants import c
import numpy as np
import casatasks

from aksco_common.image_utils import clear_extensions, exportfits

def tclean_image_cont(ms, imagename, robust=0.0, imsize=1000, cell=0.003, mask="", threshold=0.08, scales=[0, 5, 30, 100]):
    """Make a CLEAN image.
    
    Args:
        ms (string): input measurement set
        imagename (string): base string for image outputs
        robust (float): value between -2.0 and 2.0
        imsize (int): number of pixels
        cell (float): cell size in arcseconds
        mask (string): path to mask file
        threshold (float): stopping threshold of point source in mJy (mJy/beam).
        uvtaper : None
        
    Returns:
        None
    """

    clear_extensions(imagename)

    casatasks.tclean(vis=ms,
            imagename=imagename,
            specmode="mfs",
            deconvolver="multiscale",
            scales=scales,
            weighting="briggs",
            robust=robust,
            imsize=imsize,
            cell="{:.5f}arcsec".format(cell),
            niter=50000,
            threshold="{:.4f}mJy".format(threshold), # / beam
            nterms=1,
            savemodel="modelcolumn",
            mask=mask)

    exportfits(imagename + ".image", imagename + ".fits")


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Make a CLEAN image of the measurement set, selecting only one ddid")
    
    parser.add_argument("ms")
    parser.add_argument("imagename", default="", help="Base of output, e.g. `mydir/myimage`")
    parser.add_argument("--robust", default=0.0, type=float, help="Briggs Robust Value, between -2.0 and 2.0.")
    parser.add_argument("--imsize", type=int, default=1000, help="Number of pixels on each side of image.")
    parser.add_argument("--cell", type=float, default=0.003, help="Pixel size in arcseconds")
    parser.add_argument("--mask", default="", help="Path to mask file")
    parser.add_argument("--threshold", default=0.08, type=float, help="Stopping threshold in mJy (mJy/beam).")
    parser.add_argument("--scales", type=int, nargs="+", default=[0, 5, 30, 100])
    parser.add_argument("--uvtaper", type=str, nargs=3)
    args = parser.parse_args()
    
    tclean_image_cont(args.ms, args.imagename, args.robust, args.imsize, args.cell, args.mask, args.threshold, args.scales, args.uvtaper)

if __name__=="__main__":
    main()