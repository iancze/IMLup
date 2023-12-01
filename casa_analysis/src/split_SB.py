# Use listobs to understand the content of the measurement set
# Record spectral windows, channels, resolution, etc.
import casatasks
from pathlib import Path
import shutil


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Split the short baselines into their own MS."
    )

    parser.add_argument("ms")
    parser.add_argument("outfile")
    args = parser.parse_args()

    spw_string = ",".join(["{:}".format(i) for i in range(0, 18)])
    
    # split the EB
    casatasks.split(
        vis=args.ms,
        field="im_lup",
        outputvis=args.outfile,
        datacolumn="data",
        spw=spw_string
    )


if __name__ == "__main__":
    main()
