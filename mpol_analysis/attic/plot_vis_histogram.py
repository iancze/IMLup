import matplotlib.pyplot as plt 
import numpy as np
import argparse
from mpol.plot import vis_histogram_fig
from src import loaddata


def main():
    parser = argparse.ArgumentParser(
        description="Plot the visibility histogram."
    )
    parser.add_argument("outfile", help="Output file") 
    args = parser.parse_args()
    
    uu, vv, data, weight = loaddata.get_basic_data()
    
    # augment to include complex conjugates
    uu = np.concatenate([uu, -uu])
    vv = np.concatenate([vv, -vv])

    # use mpol.plots.vis_histogram_fig

if __name__=="__main__":
    main()