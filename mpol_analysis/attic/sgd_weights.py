import numpy as np
import loaddata
import torch
import argparse
import matplotlib.colors as mco
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Sampler
from mpol import coordinates, gridding, fourier, images, utils, plot
import visread
import visread.visualization

from torch.utils.tensorboard import SummaryWriter

# following structure from https://github.com/pytorch/examples/blob/main/mnist/main.py

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


# hand this a dictionary that contains the obsid as keys and ddid as lists in those
# the routine will look 

# create a model that uses the NuFFT to predict
class Net(torch.nn.Module):
    def __init__(
        self,
        coords=None,
        nchan=1,
        obsid_dict=None,
        base_cube=None,
        freeze_weights=False,
    ):
        super().__init__()

        # these should be saved as registered variables, so they are serialized on save
        self.coords = coords
        self.nchan = nchan

        # create parameters that store weights
        
        # assumes ddids indexed from 0
        # mapping from ddid to obsid
        ddid2obsid = []
        for obsid, ddids in obsid_dict.items():
            for ddid in ddids:
                ddid2obsid.append(obsid)
        
        # use this for amplitude and astrometric offsets 
        self.register_buffer("ddid2obsid", torch.tensor(ddid2obsid))

        # one for each ddid
        self.log10_sigma_factors = torch.nn.Parameter(torch.zeros(len(ddid2obsid)), requires_grad=(not freeze_weights))

        self.bcube = images.BaseCube(
            coords=self.coords, nchan=self.nchan, base_cube=base_cube
        )

        self.conv_layer = images.HannConvCube(nchan=self.nchan)

        self.icube = images.ImageCube(
            coords=self.coords, nchan=self.nchan, passthrough=True
        )
        self.nufft = fourier.NuFFT(coords=self.coords, nchan=self.nchan)


    def adjust_weights(self, ddid, weight):
        r"""
        Adjust the weights according to the rescale factor on the parameter.
        """

        # get the obsID that corresponds to all ddid
        # obsid = self.ddid2obsid[ddid]
        log10_sigma_factors = self.log10_sigma_factors[ddid]
        
        return weight / (10**log10_sigma_factors)**2

    def forward(self, uu, vv):
        r"""
        Predict visibilities at uu, vv.

        Feed forward to calculate the model visibilities. In this step, a :class:`~mpol.images.BaseCube` is fed to a :class:`~mpol.images.HannConvCube` is fed to a :class:`~mpol.images.ImageCube` is fed to a :class:`~mpol.fourier.NuFFT` to produce the model visibilities.

        Returns: 1D complex torch tensor of model visibilities.
        """
        x = self.bcube()
        x = self.conv_layer(x)
        x = self.icube(x)
        vis = self.nufft(x, uu, vv)
        return vis


class DdidBatchSampler(Sampler):
    """
    Custom Sampler to gather mini-batches having the same ddid (usually corresponds to a unique spw for a unique obs id).
    """
    def __init__(self, ddids, batch_size, shuffle=True):
        self.ddids = ddids
        self.ids = np.arange(len(ddids))
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.unique_ddids = np.unique(self.ddids)

        # calculate how many batches we will have for each ddid sub-group
        self.batches_per_ddid = {}
        for ddid in self.unique_ddids:
            # find all indices with ddid match
            ids_ddid = self.ids[(self.ddids == ddid)]
            # calculate the length
            nbatches = (len(ids_ddid) + self.batch_size - 1) // self.batch_size
            self.batches_per_ddid[ddid] = nbatches

    def __len__(self):
        "returns the number of batches to cover all the data"
        return np.sum([b for b in self.batches_per_ddid.values()])

    def __iter__(self):
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self.unique_ddids)

        for ddid in self.unique_ddids:
            # find all indices with ddid match
            ids_ddid = self.ids[(self.ddids == ddid)]

            for batch in torch.chunk(torch.tensor(ids_ddid), self.batches_per_ddid[ddid]):
                yield batch.tolist()


def plots(model, step, writer):
    """
    Plot images to the Tensorboard instance.
    """

    r = 1
    img = np.squeeze(utils.torch2npy(model.icube.sky_cube))
    fig, ax = plt.subplots(nrows=1)
    plot.plot_image(img, extent=model.coords.img_ext, ax=ax)
    # set zoom a little
    ax.set_xlim(r, -r)
    ax.set_ylim(-r, r)
    writer.add_figure("image", fig, step)

    norm_asinh = plot.get_image_cmap_norm(img, stretch='asinh')
    fig, ax = plt.subplots(nrows=1)
    plot.plot_image(img, extent=model.coords.img_ext, norm=norm_asinh, ax=ax)
    # set zoom a little
    ax.set_xlim(r, -r)
    ax.set_ylim(-r, r)
    writer.add_figure("asinh", fig, step)

    bcube = np.squeeze(utils.torch2npy(utils.packed_cube_to_sky_cube(model.bcube.base_cube)))
    norm = mco.Normalize(vmin=np.min(bcube), vmax=np.max(bcube))
    fig, ax = plt.subplots(nrows=1)
    plot.plot_image(bcube, extent=model.coords.img_ext, ax=ax, norm=norm)
    writer.add_figure("bcube", fig, step)

    # get gradient as it exists on model from root node
    b_grad = np.squeeze(utils.torch2npy(utils.packed_cube_to_sky_cube(model.bcube.base_cube.grad)))
    norm_sym = plot.get_image_cmap_norm(b_grad, symmetric=True)
    fig, ax = plt.subplots(nrows=1)
    plot.plot_image(b_grad, extent=model.coords.img_ext, norm=norm_sym, ax=ax, cmap="bwr_r")
    writer.add_figure("b_grad", fig, step)


def residual_dirty_image(coords, model_vis, uu, vv, data, weight, step, writer):
    # calculate residual dirty image for this *batch*
    resid = data - model_vis

    # convert all quantities to numpy arrays 
    uu = utils.torch2npy(uu)
    vv = utils.torch2npy(vv)
    resid = np.squeeze(utils.torch2npy(resid))
    weight = utils.torch2npy(weight)
    # print(uu.shape, vv.shape, resid.shape, weight.shape)
    imager = gridding.DirtyImager(coords=coords, uu=uu, vv=vv, weight=weight, data_re=np.real(resid), data_im=np.imag(resid))
    img, beam = imager.get_dirty_image(weighting="briggs", robust=0.0, check_visibility_scatter=False)

    # plot the two
    # set plot dimensions
    xx = 8 # in
    cax_width = 0.2 # in 
    cax_sep = 0.1 # in
    mmargin = 1.2
    lmargin = 0.7
    rmargin = 0.9
    tmargin = 0.3
    bmargin = 0.5

    npanels = 2
    # the size of image axes + cax_sep + cax_width
    block_width = (xx - lmargin - rmargin - mmargin * (npanels - 1) )/npanels
    ax_width = block_width - cax_width - cax_sep
    ax_height = ax_width 
    yy = bmargin + ax_height + tmargin
    
    kw = {"origin": "lower", "interpolation": "none", "extent": imager.coords.img_ext, "cmap":"inferno"}

    fig = plt.figure(figsize=(xx, yy))
    ax = []
    cax = []
    for i in range(npanels):
        ax.append(fig.add_axes([(lmargin + i * (block_width + mmargin))/xx, bmargin/yy, ax_width/xx, ax_height/yy]))
        cax.append(fig.add_axes([(lmargin + i * (block_width + mmargin) + ax_width + cax_sep)/xx, bmargin/yy, cax_width/xx, ax_height/yy]))

    # single-channel image cube    
    chan = 0

    im_beam = ax[0].imshow(beam[chan], **kw)
    cbar = plt.colorbar(im_beam, cax=cax[0])
    ax[0].set_title("beam")
    # zoom in a bit on the beam
    r = 0.4
    ax[0].set_xlim(r, -r)
    ax[0].set_ylim(-r, r)

    im = ax[1].imshow(img[chan], **kw)
    ax[1].set_title("dirty image")
    cbar = plt.colorbar(im, cax=cax[1])
    cbar.set_label(r"Jy/beam")

    for a in ax:
        a.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
        a.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")

    writer.add_figure("dirty_image", fig, step)

# plot histogram of residuals normalized to weight
def plot_residual_histogram(model_vis, data, weight, ddid, step, writer):
    # convert all quantities to numpy arrays 
    model_vis = utils.torch2npy(torch.squeeze(model_vis))
    data = utils.torch2npy(torch.squeeze(data))
    weight = utils.torch2npy(torch.squeeze(weight))

    ddid = utils.torch2npy(ddid)
    unique_ddid = np.unique(ddid)

    # compute normalized scatter 
    scatter = visread.scatter.get_averaged_scatter(data, model_vis, weight)
    fig = visread.visualization.plot_averaged_scatter(scatter)
    fig.suptitle(unique_ddid)

    writer.add_figure("residual_scatter", fig, step)

# model will need to move to one that includes a key to index the amplitude and/or weight rescaling
# or at least a dictionary that can look up obsid from ddid

# plot the baseline distribution of the batch samples (potentially more relevant with inter-EB switching)

# can train on just long baselines, just short baselines, etc, and see what happens to the model and residuals.

def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for i_batch, (uu, vv, data, weight, ddid) in enumerate(train_loader):
        # send all values to device
        uu, vv, data, weight, ddid = (
            uu.to(device),
            vv.to(device),
            data.to(device),
            weight.to(device),
            ddid.to(device)
        )

        optimizer.zero_grad()
        # get model visibilities
        vis = model(uu, vv)

        # correct the weights
        weight_adjusted = model.adjust_weights(ddid, weight)

        # calculate loss using adjusted weights
        loss = neg_log_likelihood_avg(vis, data, weight_adjusted)
        loss.backward()
        optimizer.step()
        

        # log results
        if i_batch % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    i_batch * len(data),
                    len(train_loader.dataset),
                    100.0 * i_batch / len(train_loader),
                    loss.item(),
                )
            )

            step = i_batch + epoch * len(train_loader)
            writer.add_scalar("loss", loss.item(), step)
            sigma_dict = {str(obsid):10**val.item() for obsid, val in enumerate(model.log10_sigma_factors)}
            writer.add_scalars("sigma_scale_factors", sigma_dict, step)
            plots(model, step, writer)
            residual_dirty_image(model.coords, vis, uu, vv, data, weight_adjusted, step, writer)
            plot_residual_histogram(vis, data, weight_adjusted, ddid, step, writer)
            if args.dry_run:
                break


def validate(model, device, validate_loader):
    model.eval()
    validate_loss = 0
    # speed up calculation by disabling gradients
    with torch.no_grad():
        for uu, vv, data, weight, ddid in validate_loader:
            # send all values to device
            uu, vv, data, weight, ddid = (
                uu.to(device),
                vv.to(device),
                data.to(device),
                weight.to(device),
                ddid.to(device)
            )

            # get model visibilities
            vis = model(uu, vv)

            # correct the weights
            weight_adjusted = model.adjust_weights(ddid, weight)

            # calculate loss as total over all chunks
            validate_loss += neg_log_likelihood_avg(vis, data, weight_adjusted).item()

    # re-normalize to total number of data points
    validate_loss /= len(validate_loader.dataset)
    return validate_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="IM Lup SGD Example")
    parser.add_argument(
        "asdf", help="Input path to .asdf file containing visibilities."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100000,
        help="input batch size for training",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=100000,
        help="input batch size for validation",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="The fraction of the dataset to use for training. The remainder will be used for validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        help="learning rate",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        help="The log dir to which tensorboard files should be written.",
    )
    parser.add_argument("--load-checkpoint", help="Path to checkpoint from which to resume.")
    parser.add_argument(
        "--save-checkpoint",
        help="Path to which checkpoint where finished model and optimizer state should be saved.",
    )
    parser.add_argument("--sampler", choices=["default", "ddid"], default="default")
    parser.add_argument("--sb_only", action="store_true", default=False)
    parser.add_argument("--freeze_weights", action="store_true")
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)

    # set up the devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # load the full dataset
    uu, vv, data, weight, ddid = loaddata.get_ddid_data(args.asdf, args.sb_only)

    nvis = len(uu)
    print("Number of Visibility Points:", nvis)
    # 42 million

    # split the dataset into train / test by creating a list of indices
    assert (args.train_fraction > 0) and (
        args.train_fraction < 1
    ), "--train-fraction must be greater than 0 and less than 1.0. You provided {}".format(
        args.train_fraction
    )
    full_indices = np.arange(len(uu))

    # randomly split into train and validation sets
    train_size = round(len(uu) * args.train_fraction)
    rng = np.random.default_rng()
    train_indices = rng.choice(full_indices, size=train_size, replace=False)
    validation_indices = np.setdiff1d(full_indices, train_indices)

    # initialize a PyTorch data object for each set
    # TensorDataset can be indexed just like a numpy array
    def init_dataset(indices):
        return TensorDataset(
            torch.tensor(uu[indices]),
            torch.tensor(vv[indices]),
            torch.tensor(data[indices]),
            torch.tensor(weight[indices]),
            torch.tensor(ddid[indices])
        )

    train_dataset = init_dataset(train_indices)
    validation_dataset = init_dataset(validation_indices)

    # set the batch sizes for the loaders
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True}
    else:
        cuda_kwargs = {}    
    
    if args.sampler == "default":
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **cuda_kwargs)
        validation_loader = DataLoader(validation_dataset, batch_size=args.validation_batch_size, shuffle=True, **cuda_kwargs)

    elif args.sampler == "ddid":
        train_ddids = ddid[train_indices]
        train_sampler = DdidBatchSampler(train_ddids, args.batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **cuda_kwargs)

        validation_ddids = ddid[validation_indices]
        validation_sampler = DdidBatchSampler(validation_ddids, args.validation_batch_size)
        validation_loader = DataLoader(validation_dataset, batch_sampler=validation_sampler, **cuda_kwargs)

    # create the model and send to device
    obsid_dict = {0:[0,1,2,3,4], 1:[5,6,7,8,9,10], 2:[11,12], 3:[13,14], 4:[15,16,17], 5:[18,19,20,21], 6:[22,23,24,25]}
    coords = coordinates.GridCoords(cell_size=0.005, npix=1028)

    # initialize from the base_cube, just until we get a run through
    # checkpoint = torch.load(args.load_checkpoint, map_location=torch.device('cpu'))
    # base_cube = checkpoint["model_state_dict"]["bcube.base_cube"]
    # model = Net(coords, obsid_dict=obsid_dict, base_cube=base_cube).to(device)
    model = Net(coords, obsid_dict=obsid_dict, freeze_weights=args.freeze_weights).to(device)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # here is where we could set up a scheduler, if desired

    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])

    # set up TensorBoard instance
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

    # enter the loop
    for epoch in range(0, args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        vloss_avg = validate(model, device, validation_loader)
        print("Logging validation")
        writer.add_scalar("vloss_avg", vloss_avg, epoch)
        optimizer.step()

    # save checkpoint
    if args.save_checkpoint is not None:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            args.save_checkpoint,
        )

    writer.close()


if __name__ == "__main__":
    main()
