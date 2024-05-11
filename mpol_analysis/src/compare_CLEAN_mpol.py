import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpol import coordinates, precomposed, utils
from mpol.constants import arcsec
from mpol.input_output import ProcessFitsImage
from astropy.visualization.mpl_normalize import simple_norm

def main():
    parser = argparse.ArgumentParser(description="Compare image to DSHARP image")
    parser.add_argument("load_checkpoint", metavar="load-checkpoint", help="Path to checkpoint from which to resume.")
    parser.add_argument("dsharpfits", help="Path to DSHARP FITS file.")
    parser.add_argument("outfile")
    args = parser.parse_args()

    coords = coordinates.GridCoords(cell_size=0.005, npix=1028)
    checkpoint = torch.load(args.load_checkpoint, map_location=torch.device('cpu'))
    base_cube = checkpoint["model_state_dict"]["bcube.base_cube"]

    # create a new SimpleNet, using the basecube
    
    model = precomposed.SimpleNet(coords=coords, nchan=1, base_cube=base_cube)
    with torch.no_grad():
        # perform a forward pass to save products to FourierCube layer
        model.forward()
        # access products
        sky_image = utils.torch2npy(torch.squeeze(model.icube.sky_cube))
        
    fig, ax = plt.subplots(ncols=2, figsize=(9,4))


    fits_obj = ProcessFitsImage(args.dsharpfits)
    clean_im, clean_im_ext, clean_beam = fits_obj.get_image(beam=True)

    BMAJ, BMIN, BPA = clean_beam 

    # calculate beam area
    beam_area = (np.pi / (4 * np.log(2))) * BMAJ * BMIN # arcsec^2
    jy_arcsec = 1e-3 * clean_im / beam_area

    from astropy.io import fits
    hdu_list = fits.open(args.dsharpfits)
    hdu = hdu_list[0]

    # calculate the total flux of each image 
    # convert to Jy/pixel then sum
    RA, DEC, ext = fits_obj.get_extent(hdu.header)
    pixel_size_clean = np.abs((RA[1] - RA[0]) * (DEC[1] - DEC[0])) # arcsec^2
    print("flux clean", np.sum(jy_arcsec * pixel_size_clean), "Jy")

    pixel_size_mpol = (coords.dl / arcsec)**2
    print("flux MPoL", np.sum(sky_image * pixel_size_mpol))

    norm_clean = simple_norm(jy_arcsec, stretch='asinh', asinh_a=0.02)
    norm_mpol = simple_norm(sky_image, stretch='asinh', asinh_a=0.02, 
                        min_cut=np.min(sky_image), max_cut=1.0)
    
    im_clean = ax[0].imshow(jy_arcsec, extent=clean_im_ext, origin="lower", cmap="inferno", norm=norm_clean)
    plt.colorbar(im_clean, ax=ax[0])

    im_mpol = ax[1].imshow(sky_image, extent=coords.img_ext, origin="lower", cmap="inferno", norm=norm_mpol)
    plt.colorbar(im_mpol, ax=ax[1])

    r = 1.5
    for a in ax:
        a.set_xlim(r, -r)
        a.set_ylim(-r, r)

    fig.subplots_adjust(wspace=0.25)
    fig.savefig(args.outfile, dpi=300)


if __name__ == "__main__":
    main()