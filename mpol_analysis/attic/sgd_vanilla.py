import numpy as np
import loaddata
import dirty_image
import torch
import argparse
import matplotlib.colors as mco
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Sampler
from mpol import coordinates, gridding, fourier, images, losses, utils, plot
import visread
import visread.visualization

from torch.utils.tensorboard import SummaryWriter

# following structure from https://github.com/pytorch/examples/blob/main/mnist/main.py

class Net(torch.nn.Module):
    def __init__(
        self,
        coords=None,
        nchan=1,
        FWHM=0.05,
    ):
        super().__init__()

        self.coords = coords
        self.nchan = nchan

        self.bcube = images.BaseCube(coords=self.coords, nchan=self.nchan)
        self.conv_layer = images.GaussConvFourier(coords=self.coords, FWHM_maj=FWHM, FWHM_min=FWHM)
        self.icube = images.ImageCube(coords=self.coords, nchan=self.nchan)
        self.nufft = fourier.NuFFT(coords=self.coords, nchan=self.nchan)

    def forward(self, uu, vv):
        r"""
        Predict model visibilities at baseline locations.

        Parameters
        ----------
        uu, vv : torch.Tensor
            spatial frequencies. Units of :math:`\lambda`.

        Returns
        -------
        torch.Tensor
            1D complex torch tensor of model visibilities.
        """
        # Feed-forward network passes base representation through "layers"
        # to create model visibilities
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

    imager = gridding.DirtyImager.from_tensors(
        coords=coords, uu=uu, vv=vv, weight=weight, data=resid
    )
    img, beam = imager.get_dirty_image(
        weighting="briggs",
        robust=0.0,
        check_visibility_scatter=False,
        unit="Jy/arcsec^2",
    )

    fig = dirty_image.plot_beam_and_image(beam, img, imager.coords.img_ext)
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

def train(args, model, device, train_loader, optimizer, epoch, lam_ent, writer):
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
        vis = model(uu, vv)[0] # take only the first channel

        # calculate loss
        loss = losses.neg_log_likelihood_avg(
            vis, data, weight
        ) + lam_ent * losses.entropy(
            model.icube.packed_cube, prior_intensity=1e-4, tot_flux=0.253
        )
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
            plots(model, step, writer)
            residual_dirty_image(model.coords, vis, uu, vv, data, weight, step, writer)
            plot_residual_histogram(vis, data, weight, ddid, step, writer)
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
            vis = model(uu, vv)[0] # take first channel

            # calculate loss as total over all chunks
            validate_loss += losses.neg_log_likelihood_avg(vis, data, weight).item()

    # re-normalize to total number of batches
    validate_loss /= len(validate_loader)
    return validate_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="IM Lup SGD")
    parser.add_argument(
        "asdf", help="Input path to .asdf file containing visibilities."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="input batch size for training",
    )
    parser.add_argument(
        "--validation-batch-size",
        type=int,
        default=10000,
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
        default=20,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate",
    )
    parser.add_argument("--FWHM", type=float, default=0.05, help="FWHM of Gaussian Base layer in arcseconds.")
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
        default=4,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        help="The log dir to which tensorboard files should be written.",
    )
    parser.add_argument("--lam-ent", type=float, default=0.0)
    parser.add_argument("--load-checkpoint", help="Path to checkpoint from which to resume.")
    parser.add_argument(
        "--save-checkpoint",
        help="Path to which checkpoint where finished model and optimizer state should be saved.",
    )
    parser.add_argument("--sampler", choices=["default", "ddid"], default="default")
    parser.add_argument("--sb_only", action="store_true", default=False)
    args = parser.parse_args()

    # set seed
    torch.manual_seed(args.seed)

    # choose the compute device, preference cuda > mps > cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
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
    cuda_kwargs = {"num_workers": 1, "pin_memory": True}
    
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
    coords = coordinates.GridCoords(cell_size=0.005, npix=1028)
    model = Net(coords, nchan=1, FWHM=args.FWHM).to(device)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # here is where we could set up a scheduler, if desired

    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    # set up TensorBoard instance
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

    # enter the loop
    for epoch in range(0, args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, args.lam_ent, writer)
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
