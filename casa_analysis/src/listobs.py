# Use listobs to understand the content of the measurement set
# Record spectral windows, channels, resolution, etc.
import casatasks
from pathlib import Path
import os 

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Provides a light wrapper to casa/listobs with default printing to a file.")
    
    parser.add_argument("ms")
    parser.add_argument("outfile")

    args = parser.parse_args()

    p = Path(args.outfile)
    if p.exists():
        os.remove(p)
    casatasks.listobs(vis=args.ms, listfile=args.outfile)

if __name__=="__main__":
    main()