import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sgd import Net
from mpol import coordinates, images, precomposed, utils
from mpol.constants import arcsec
from mpol.input_output import ProcessFitsImage
from astropy.io import fits
from astropy.visualization.mpl_normalize import simple_norm

def main():
    parser = argparse.ArgumentParser(description="Compare image to DSHARP image")
    parser.add_argument("load_checkpoint", metavar="load-checkpoint", help="Path to checkpoint from which to resume.")
    parser.add_argument("dsharpfits", help="Path to DSHARP FITS file.")
    parser.add_argument("plotfile")
    parser.add_argument("--fitsfile")
    args = parser.parse_args()

    # load the CLEAN image
    fits_obj = ProcessFitsImage(args.dsharpfits)
    clean_im, clean_im_ext, clean_beam = fits_obj.get_image(beam=True)

    BMAJ, BMIN, BPA = clean_beam 

    # calculate beam area
    beam_area = (np.pi / (4 * np.log(2))) * BMAJ * BMIN # arcsec^2
    print("Beam area", beam_area)
    jy_arcsec = 1e-3 * clean_im / beam_area
    
    hdu_list = fits.open(args.dsharpfits)
    hdu = hdu_list[0]

    # calculate the total flux
    # convert to Jy/pixel then sum
    RA, DEC, ext = fits_obj.get_extent(hdu.header)
    pixel_size_clean = np.abs((RA[1] - RA[0]) * (DEC[1] - DEC[0])) # arcsec^2
    print("CLEAN peak", np.max(jy_arcsec), "Jy/arcsec^2")
    print("CLEAN flux", np.sum(jy_arcsec * pixel_size_clean), "Jy")


    # get the MPoL image from the checkpoint
    coords = coordinates.GridCoords(cell_size=0.005, npix=1028)
    checkpoint = torch.load(args.load_checkpoint, map_location=torch.device('cpu'))

    # get the image cube in packed format and run through an ImageCube
    icube = images.ImageCube(coords=coords)
    icube(checkpoint["model_state_dict"]["icube.packed_cube"])

    if args.fitsfile is not None:
        # export to FITS
        icube.to_FITS(args.fitsfile)

    print("MPoL peak", torch.max(icube.packed_cube).item(), "Jy/arcsec^2")
    print("MPoL flux", icube.flux.item(), "Jy")

    mpol_img = torch.squeeze(icube.sky_cube)

    # start plotting things
    
    # NeurIPS width is 5.5 in
    
    lmargin = 0.0
    rmargin = lmargin
    mmargin = 0.1 # sep between left and right axes
    XX = 5.5 #in 
    ax_width = (XX - lmargin - rmargin - mmargin) / 2
    ax_height = ax_width

    cax_height = 0.2
    cax_sep = 0.05
    cax_cut = 0.05
    cax_width = ax_width - 2 * cax_cut
    tmargin = 0.25
    bmargin = 0.0
    YY = bmargin + ax_height + cax_sep + cax_height + tmargin

    fig = plt.figure(figsize=(XX,YY))

    ax_clean = fig.add_axes([lmargin/XX, bmargin/YY, ax_width/XX, ax_height/YY])
    ax_mpol = fig.add_axes([(lmargin + ax_width + mmargin)/XX, bmargin/YY, ax_width/XX, ax_height/YY])
    ax = [ax_clean, ax_mpol]

    cax_clean = fig.add_axes([(lmargin + cax_cut)/XX, (bmargin + ax_height + cax_sep)/YY, cax_width/XX, cax_height/YY])
    cax_mpol = fig.add_axes([(lmargin + ax_width + mmargin + cax_cut)/XX, (bmargin + ax_height + cax_sep)/YY, cax_width/XX, cax_height/YY])

    max_cut = 0.7
    # norm_clean = simple_norm(jy_arcsec, stretch='asinh', asinh_a=0.02, vmax=max_cut*np.max(jy_arcsec))
    norm_clean = simple_norm(jy_arcsec, stretch='sqrt', vmax=max_cut*np.max(jy_arcsec))
    max_cut = 0.4
    # norm_mpol = simple_norm(mpol_img, stretch='asinh', asinh_a=0.02, 
    norm_mpol = simple_norm(mpol_img, stretch='sqrt', vmin=0.0, vmax=max_cut*torch.max(mpol_img))
    
    im_clean = ax_clean.imshow(jy_arcsec, extent=clean_im_ext, origin="lower", cmap="inferno", norm=norm_clean)

    cbar_kwargs = {"orientation":"horizontal", "ticklocation":"top"}
    cbar_clean = plt.colorbar(im_clean, cax=cax_clean, **cbar_kwargs)
    cbar_clean.ax.tick_params(labelsize=9) 


    im_mpol = ax_mpol.imshow(mpol_img, extent=coords.img_ext, origin="lower", cmap="inferno", norm=norm_mpol)
    cbar_mpol = plt.colorbar(im_mpol, cax=cax_mpol, **cbar_kwargs)
    cbar_mpol.ax.tick_params(labelsize=9) 

    r = 1.5
    for a in ax:
        a.set_xlim(r, -r)
        a.set_ylim(-r, r)
        a.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    fig.subplots_adjust(wspace=0.25)
    fig.savefig(args.plotfile, dpi=300)


if __name__ == "__main__":
    main()