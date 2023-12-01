# Use listobs to understand the content of the measurement set
# Record spectral windows, channels, resolution, etc.
import casatasks
import casatools
from pathlib import Path
import os 

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Determine the column names present in a measurement set, and print to file.")
    
    parser.add_argument("ms")
    parser.add_argument("outfile")

    args = parser.parse_args()

    tb = casatools.table()
    tb.open(args.ms)

    with open(args.outfile, "w") as f:
        f.write(" ".join(tb.colnames()))
    
    tb.close()


if __name__=="__main__":
    main()