import casatools
import analysisUtils as au
import numpy as np 

msmd = casatools.msmetadata()

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Get a list of all antenna names in the measurement set.")
    
    parser.add_argument("ms")
    parser.add_argument("--outfile", help="If specified, write names to file. Otherwise print.")

    args = parser.parse_args()

    msmd.open(args.ms)
    antenna_names = np.unique(msmd.antennanames())
    msmd.close()


    frac = 0.9
    unflagged_antennas = au.getUnflaggedAntennas(args.ms, flaggedFraction=frac)

    if args.outfile is not None:
        with open(args.outfile, "w") as f:
            print("Antenna names from casatools.msmd.antennanames:\n", file=f)
            print(antenna_names, file=f)
            print("\nUnflagged antennas from analysisUtils.getUnflaggedAntennas\n", file=f)
            print("unflagged antennas frac {:.2f}".format(frac), file=f)
            print(unflagged_antennas, file=f)
    else:
        print(antenna_names)
        print(unflagged_antennas)
        
if __name__=="__main__":
    main()