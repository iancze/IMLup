import casatasks


def apparentsens(ms, robust=0.0, imsize=1000, cell=0.003):
    return casatasks.apparentsens(
        vis=ms,
        specmode="mfs",
        weighting="briggs",
        robust=robust,
        imsize=imsize,
        cell="{:.5f}arcsec".format(cell),
    )


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Use casatasks.apparentsens to estimate the apparent sensitivity of our dataset."
    )
    parser.add_argument(
        "ms", help="The path to the measurement set to operate on."
    )
    parser.add_argument(
        "--robust",
        default=0.0,
        type=float,
        help="Briggs Robust Value, between -2.0 and 2.0.",
    )
    parser.add_argument(
        "--imsize",
        type=int,
        default=1000,
        help="Number of pixels on each side of image.",
    )
    parser.add_argument(
        "--cell", type=float, default=0.003, help="Pixel size in arcseconds"
    )
    parser.add_argument("--outfile", help="Write results to this file", required=True)
    args = parser.parse_args()

    result = apparentsens(args.ms, args.robust, args.imsize, args.cell)

    with open(args.outfile, "w") as f:
        f.write("Sensitivity calculated using casatasks.apparentsens\n")
        f.write("robust = {:.2f}\n".format(args.robust))
        f.write("imsize = {}\n".format(args.imsize))
        f.write("cell = {}\n".format(args.cell))
        f.write("RMS Point source sensitivity : {:.4f} mJy\n".format(result['effSens'] * 1e3))
        f.write("Factor of sensitivity lost relative to (maximal) natural weighting {:.3f}\n".format(result["relToNat"]))


if __name__ == "__main__":
    main()
