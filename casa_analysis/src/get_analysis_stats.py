import analysisUtils as au
from pathlib import Path


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Get stats about observation using analysisUtils.")
    
    parser.add_argument("ms")
    parser.add_argument("outfile")

    args = parser.parse_args()

    with open(args.outfile, "w") as f:
        # get min and max baselines
        ans = au.getBaselineStats(msFile=args.ms, field="im_lup")
        f.write("Projected baselines\n")
        f.write("min: {:.1f}m \nmax: {:.1f}m \n".format(ans[1], ans[2]))
    
        # get total science time in minutes
        ans = au.timeOnSource(args.ms)
        f.write("Science Time on Source: {:.2f}m\n".format(ans["minutes_on_science"]))

        # We could use lstrange
        # au.lstrange(args.ms)

        mjdsecmin = au.getObservationStart(args.ms)
        mjdsecmax = au.getObservationStop(args.ms)

        [mjdmin,utmin] = au.mjdSecondsToMJDandUT(mjdsecmin)
        [mjdmax,utmax] = au.mjdSecondsToMJDandUT(mjdsecmax)
        meanJD = au.mjdToJD(0.5*(mjdmin+mjdmax))
        
        f.write("Mean JD: {:.3f}\n".format(meanJD))
        f.write("UT range: {:} to {:}\n".format(utmin, utmax))

        ans = au.getUnflaggedAntennas(args.ms)
        ants = ans[0]
        f.write("Number of unflagged antennas {:}\n".format(len(ants)))

if __name__=="__main__":
    main()