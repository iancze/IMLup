import numpy as np
from visread import process
import asdf
import argparse

def pack_file(ms, outfile):
    # goal is to export visibilities with as minimal processing as possible
    # with the expectation that we might modify our MPoL scripts to perform
    # additional processing before beginning the fit.

    # I think we do want to average polarizations
    # but otherwise keep continuum channels separate

    # group by ObsID
    # group by ddid
    obsid = {0:[0,1,2,3,4], 1:[5,6,7,8,9,10], 2:[11,12], 3:[13,14], 4:[15,16,17], 5:[18,19,20,21], 6:[22,23,24,25]}

    obs_tree = {"obsids":{}}
    
    for key, ddids in obsid.items():
        ddid_tree = {"ddids":{}}

        # for a given spw
        for ddid in ddids:
            # get processed visibilities
            # includes flipping frequency, if necessary
            # including complex conjugation
            # no channel-averaging (assuming DSHARP did this to the maximal extent possible)
            d = process.get_processed_visibilities(ms, ddid)

            # drop the "model_data" key from this dictionary, we no longer need it
            d.pop("model_data")

            # add this sub-dictionary to a larger dictionary
            ddid_tree["ddids"][ddid] = d

        # add this to a yet-larger tree
        obs_tree["obsids"][key] = ddid_tree

    # Create the ASDF file object from our data tree
    af = asdf.AsdfFile(obs_tree)

    # Write the data to a new file
    af.write_to(outfile)


def main():
    parser = argparse.ArgumentParser(
        description="Take visibilities in the measurement set and put them into an asdf file for use with software without CASA dependencies."
    )
    parser.add_argument("ms", help="Filename of measurement set")
    parser.add_argument("outfile", help="Output file") 
    args = parser.parse_args()
    
    pack_file(args.ms, args.outfile)

if __name__ == "__main__":
    main()