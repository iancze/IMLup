import casatools
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Plot baselines for all datadesc IDs.")
    
    parser.add_argument("ms")
    parser.add_argument("outdir")

    args = parser.parse_args()
    p = Path(args.outdir)

    # get list of datadescids
    msmd = casatools.msmetadata()
    msmd.open(args.ms)
    ddids = msmd.datadescids()
    msmd.done() 

    ms = casatools.ms() 
    ms.open(args.ms)

    for ddid in ddids:
        ms.selectinit(reset=True)
        ms.selectinit(datadescid=ddid)
        d = ms.getdata(["uvw", "flag"])  
        # d["uvw"] is an array of float64 with shape [3, nvis]
        uu, vv, ww = d["uvw"] 
        # uu is (nvis,)
        # flag is (2, nchan, nvis)
        # flag if any pol, but only if all channels
        flag_init = d["flag"]
        flag_chan = np.any(d["flag"], axis=0) # True means data is bad
        flag = np.all(flag_chan, axis=0)
        # now flag is (nvis,)

        fig, ax = plt.subplots(nrows=1, figsize=(5, 5))
        ax.scatter(uu[~flag], vv[~flag], s=1.5, rasterized=True, linewidths=0.0, c="k")
        ax.scatter(uu[flag], vv[flag], s=1.5, rasterized=True, linewidths=0.0, c="r", label="flagged")
        ax.set_xlabel(r"$u$ [m]")
        ax.set_ylabel(r"$v$ [m]")
        ax.legend()
        fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8)
        fig.savefig(p / "{:}.png".format(ddid), dpi=300)
        plt.close("all")

    ms.done()

if __name__=="__main__":
    main()



