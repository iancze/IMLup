import numpy as np
import loaddata
import torch
from mpol import coordinates, fourier


# set up the devices
device = torch.device("cuda")

# load the full dataset
uu, vv, data, weight = loaddata.get_basic_data("data/raw/imlup.asdf")

coords = coordinates.GridCoords(cell_size=0.005, npix=1028)

# create a random sky image
img = torch.rand((1, coords.npix, coords.npix), requires_grad=True, device=device)

# create a NuFFT instance
nufft = fourier.NuFFT(coords=coords)

# send relevant items to gpu 
nufft = nufft.to(device)

# test with and without tensor input
uu_t = torch.tensor(uu, device=device)
vv_t = torch.tensor(vv, device=device)

# def all_numpy():
#     # calculate all of the residuals for all visibility points
#     model_vis = nufft(img, uu, vv)

# def all_numpy_no_grad():
#     with torch.no_grad():
#         model_vis = nufft(img, uu, vv)

def all_tensor():
    # calculate all of the residuals for all visibility points
    model_vis = nufft(img, uu_t, vv_t)
    

def all_tensor_no_grad():
    with torch.no_grad():
        model_vis = nufft(img, uu_t, vv_t)
    

all_tensor()
all_tensor_no_grad()

# how long did this take and what was our memory usage?
import torch.utils.benchmark as benchmark

t0 = benchmark.Timer(
    stmt='all_tensor()',
    setup='from __main__ import all_tensor')

t1 = benchmark.Timer(
    stmt='all_tensor_no_grad()',
    setup='from __main__ import all_tensor_no_grad')

print("all_tensor", t0.timeit(3))
print("all_tensor_no_grad", t1.timeit(3))

# honestly no difference in prediction. Took only 0.5s... kind of hard to believe? 
# so we might as well just keep it as is and predict for dirty image

# create a mock image

# try to predict the loose visibilities for slices of all baselines
# see if there is a memory problem, or if this is better done in batches
# is there a way we can eliminate "grad" from this routine to speed everything up
# do it on the GPU

