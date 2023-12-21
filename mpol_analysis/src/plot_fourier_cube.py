import numpy as np
import loaddata
import torch
import argparse
import matplotlib.colors as mco
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Sampler
from mpol import coordinates, gridding, precomposed, utils


def calculate_amp_cell_scaling(filename, coords):
    uu, vv, data, weight = loaddata.get_basic_data(filename)

    gridder = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=np.real(data),
        data_im=np.imag(data),
    )

    dataset = gridder.to_pytorch_dataset(check_visibility_scatter=False)
    # find max weight 
    max_weight = torch.max(dataset.weight_gridded)
    min_sigma = 1/torch.sqrt(max_weight)
    print("Max weight", max_weight)
    print("Min sigma", min_sigma, "Jy")
    
    return min_sigma
    

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Experiment with plotting the Fourier Cube")
    parser.add_argument("load_checkpoint", metavar="load-checkpoint", help="Path to checkpoint from which to resume.")
    parser.add_argument("--filename")
    args = parser.parse_args()

    coords = coordinates.GridCoords(cell_size=0.005, npix=1028)

    min_sigma = None
    if args.filename is not None:
        min_sigma = calculate_amp_cell_scaling(args.filename, coords)


    checkpoint = torch.load(args.load_checkpoint, map_location=torch.device('cpu'))
    base_cube = checkpoint["model_state_dict"]["bcube.base_cube"]

    # create a new SimpleNet, using the basecube
    
    model = precomposed.SimpleNet(coords=coords, nchan=1, base_cube=base_cube)
    with torch.no_grad():
        # perform a forward pass to save products to FourierCube layer
        model.forward()
        # access products
        amp = utils.torch2npy(torch.squeeze(model.fcube.ground_amp))
        phase = utils.torch2npy(torch.squeeze(model.fcube.ground_phase))
        
    fig, ax = plt.subplots(ncols=2, figsize=(9,4))

    # set vmin to noise level?
    # calculate what the highest weight gridded cell is
    # then estimate sigma from that
    # and then set that as min

    if min_sigma is not None:
        log_norm = mco.LogNorm(vmin=min_sigma, vmax=np.max(amp))
    else:
        log_norm = mco.LogNorm(vmin=np.min(amp), vmax=np.max(amp))
    im_amp = ax[0].imshow(amp, extent=coords.vis_ext, origin="lower", norm=log_norm, cmap="inferno")
    plt.colorbar(im_amp, ax=ax[0])

    im_phase = ax[1].imshow(phase, extent=coords.vis_ext, origin="lower", cmap="twilight")
    plt.colorbar(im_phase, ax=ax[1])

    fig.subplots_adjust(wspace=0.25)
    fig.savefig("fourier_cube.png", dpi=300)


if __name__ == "__main__":
    main()
