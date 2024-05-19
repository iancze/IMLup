import torch
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Calculate per EB scaling factors.")
    parser.add_argument("load_checkpoint", metavar="load-checkpoint", help="Path to checkpoint from which to resume.")
    args = parser.parse_args()

    checkpoint = torch.load(args.load_checkpoint, map_location=torch.device('cpu'))

    # get the image cube in packed format and run through an ImageCube
    log10_sigma_factors = checkpoint["model_state_dict"]["log10_sigma_factors"]
    sigma_factors = 10**log10_sigma_factors.numpy()
    
    # load the relevant checkpoint

    # use obsid_dict to find the mapping from spw to EB
    obsid_dict = {0:[0,1,2,3,4], 1:[5,6,7,8,9,10], 2:[11,12], 3:[13,14], 4:[15,16,17], 5:[18,19,20,21], 6:[22,23,24,25]}

    for key, spws in obsid_dict.items():
        # average all SB factors within same EB to report value
        spw_sigma_factors = sigma_factors[spws]
        print(key, np.average(spw_sigma_factors))




if __name__=="__main__":
    main()