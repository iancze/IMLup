#!/usr/bin/env python3

from visread import process, scatter, visualization
import casatools
import os
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

msmd = casatools.msmetadata()

def plot_raw(fname, outdir, residual=True, datacolumn="corrected_data"):

    p = Path(outdir)

    msmd.open(fname)
    spws = msmd.datadescids() 
    msmd.done()

    for spw in spws:
        fig = visualization.plot_scatter_datadescid(fname, spw, residual=residual, datacolumn=datacolumn)
        fig.savefig(p / "{:02d}.png".format(spw), dpi=300)

    plt.close("all")


def plot_rescaled_resid(fname, outdir, residual=True, datacolumn="corrected_data"):

    p = Path(outdir)

    msmd.open(fname)
    spws = msmd.datadescids() 
    msmd.done()

    for spw in spws:
        # calculate rescale factor and replot
        sigma_rescale = scatter.get_sigma_rescale_datadescid(fname, spw, datacolumn=datacolumn)
        fig = visualization.plot_scatter_datadescid(fname, spw, sigma_rescale=sigma_rescale, residual=True, datacolumn=datacolumn)
        fig.suptitle(r"rescale $\sigma$ by ${:.2f}$".format(sigma_rescale))
        fig.savefig(p / "{:02d}.png".format(spw), dpi=300)

        plt.close("all")


def plot_averaged(fname):
    """
    Does the scatter of the polarization-averaged visibilities agree with with the weight value? Make a plot for each spectral window contained in the measurement set. Save under 'averaged/00.png'

    Args:
        fname (string): path to the measurement set

    Returns:
        None
    """
    # make a directory for the rescaled plots
    avgdir = "averaged"
    if not os.path.exists(avgdir):
        os.makedirs(avgdir)

    msmd.open(fname)
    spws = msmd.datadescids() 
    msmd.done()

    for spw in spws:
        # calculate rescale factor and replot
        sigma_rescale = scatter.get_sigma_rescale_datadescid(fname, spw)

        # get processed visibilities
        d = process.get_processed_visibilities(fname, spw, sigma_rescale=sigma_rescale, incl_model_data=True)

        scatter_visibilities = scatter.get_averaged_scatter(d["data"], d["model_data"], d["weight"], flag=d["flag"])

        fig = visualization.plot_averaged_scatter(scatter_visibilities)
        fig.savefig("{:}/{:02d}.png".format(avgdir, spw), dpi=300)

        plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Examine the scatter in each spectral window of a measurement set."
    )
    parser.add_argument("ms", help="Filename of measurement set")
    parser.add_argument("outdir", help="Base of the output directory.")
    parser.add_argument("scale", choices=["raw", "raw-resid", "rescale", "pol_average"], help="How and if the visibilities should be rescaled. 'raw' does not apply rescaling, 'rescale' calculates and applies a rescale factor, and 'pol_average' rescales and averages, mimicking the pol-averaged exported product for visibility analysis.")
    parser.add_argument("--datacolumn", default="corrected_data", help="Which column to use.")
        
    args = parser.parse_args()
    
    if args.scale == "raw":
        plot_raw(args.ms, args.outdir, residual=False, datacolumn=args.datacolumn)

    if args.scale == "raw-resid":
        plot_raw(args.ms, args.outdir, datacolumn=args.datacolumn)
        
    if args.scale == "rescale":
        plot_rescaled_resid(args.ms, args.outdir, datacolumn=args.datacolumn)

    # if args.averaged:
    #     plot_averaged(args.filename)


if __name__ == "__main__":
    main()