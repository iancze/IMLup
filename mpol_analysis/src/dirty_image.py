import mpol
from mpol import coordinates, gridding
import numpy as np
import matplotlib.pyplot as plt
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

def plot_beam_and_image(beam, img, extent):

# set plot dimensions
    xx = 8 # in
    cax_width = 0.2 # in 
    cax_sep = 0.1 # in
    mmargin = 1.2
    lmargin = 0.7
    rmargin = 0.9
    tmargin = 0.3
    bmargin = 0.5

    npanels = 2
    # the size of image axes + cax_sep + cax_width
    block_width = (xx - lmargin - rmargin - mmargin * (npanels - 1) )/npanels
    ax_width = block_width - cax_width - cax_sep
    ax_height = ax_width 
    yy = bmargin + ax_height + tmargin
    
    kw = {"origin": "lower", "interpolation": "none", "extent": extent, "cmap":"inferno"}

    fig = plt.figure(figsize=(xx, yy))
    ax = []
    cax = []
    for i in range(npanels):
        ax.append(fig.add_axes([(lmargin + i * (block_width + mmargin))/xx, bmargin/yy, ax_width/xx, ax_height/yy]))
        cax.append(fig.add_axes([(lmargin + i * (block_width + mmargin) + ax_width + cax_sep)/xx, bmargin/yy, cax_width/xx, ax_height/yy]))

    # single-channel image cube    
    chan = 0

    im_beam = ax[0].imshow(beam[chan], **kw)
    cbar = plt.colorbar(im_beam, cax=cax[0])
    ax[0].set_title("beam")
    # zoom in a bit
    r = 0.3
    ax[0].set_xlim(r, -r)
    ax[0].set_ylim(-r, r)

    im = ax[1].imshow(img[chan], **kw)
    ax[1].set_title("dirty image")
    cbar = plt.colorbar(im, cax=cax[1])
    cbar.set_label(r"Jy/beam")

    for a in ax:
        a.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
        a.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")

    return fig

def main():

    parser = argparse.ArgumentParser(
        description="Make a dirty image with the visibilities."
    )
    parser.add_argument("file", help="Path to asdf file")
    parser.add_argument("outfile", help="Output file") 
    args = parser.parse_args()
    
    imager = init_dirty_imager(args.file)

    img, beam = imager.get_dirty_image(weighting="briggs", robust=0.0)
    fig = plot_beam_and_image(beam, img, imager.coords.img_ext)
    
    fig.savefig(args.outfile, dpi=300)


if __name__=="__main__":
    main()