import casatools
import analysisUtils as au

msmd = casatools.msmetadata()
ms = casatools.ms()


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Plot histograms of the baseline distributions for each antenna individually.")
    
    parser.add_argument("ms")
    parser.add_argument("outfile", help="File to save list.")

    args = parser.parse_args()

    output = au.getBaselineLengths(args.ms, sort=True, unflagged=True)

    with open(args.outfile, "w") as f:
        f.write("only baselines between unflagged antennas")
        for (antpair, length) in output:
            f.write("{} : {}m\n".format(antpair, length))
        
if __name__=="__main__":
    main()