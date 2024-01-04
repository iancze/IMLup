import numpy as np
import loaddata
import torch

# create our new loss function
def log_likelihood(model_vis, data_vis, weight):
    # asssumes tensors are the same shape
    N = len(torch.ravel(data_vis)) # number of complex visibilities
    resid = data_vis - model_vis

    # derivation from real and imag in notebook
    return - N * np.log(2 * np.pi) + torch.sum(torch.log(weight)) - 0.5 * torch.sum(weight * torch.abs(resid)**2)

def neg_log_likelihood_avg(model_vis, data_vis, weight):
    N = len(torch.ravel(data_vis)) # number of complex visibilities
    ll = log_likelihood(model_vis, data_vis, weight)
    # factor of 2 is because of complex calculation
    return - ll / (2 * N)

for N in np.logspace(4,5, num=10):
    # create fake model, resid, and weight 

    N = int(N)

    mean = torch.zeros(N)
    std = 0.2 * torch.ones(N)
    weight = 1/std**2

    model_real = torch.ones(N)
    model_imag = torch.zeros(N)
    model = torch.complex(model_real, model_imag)

    noise_real = torch.normal(mean, std)
    noise_imag = torch.normal(mean, std)
    noise = torch.complex(noise_real, noise_imag)

    data = model + noise

    nlla = neg_log_likelihood_avg(model, data, weight)
    print("N", N, "nlla", nlla)