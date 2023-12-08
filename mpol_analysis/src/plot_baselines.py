import matplotlib.pyplot as plt 
import numpy as np
import argparse
from src import loaddata

def main():
    parser = argparse.ArgumentParser(
        description="Plot the visibilities."
    )
    parser.add_argument("file", help="Path to asdf file")
    parser.add_argument("outfile", help="Output file") 
    args = parser.parse_args()
    
    uu, vv, data, weight = loaddata.get_basic_data(args.file)
    
    # augment to include complex conjugates
    uu = np.concatenate([uu, -uu])
    vv = np.concatenate([vv, -vv])

    fig, ax = plt.subplots(nrows=1, figsize=(6,6))
    ax.scatter(uu, vv, s=1.5, rasterized=True, linewidths=0.0, c="k")
    ax.set_xlabel(r"$u$ [k$\lambda$]")
    ax.set_ylabel(r"$v$ [k$\lambda$]")
    ax.axis('equal')
    fig.subplots_adjust(left=0.15, right=0.85)
    fig.savefig(args.outfile, dpi=300)

if __name__=="__main__":
    main()