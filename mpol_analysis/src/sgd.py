import numpy as np
import loaddata
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from mpol import coordinates, fourier, images, losses

from torch.utils.tensorboard import SummaryWriter

# following structure from https://github.com/pytorch/examples/blob/main/mnist/main.py


# create a model that uses the NuFFT to predict, following SimpleNet
class SGDNet(torch.nn.Module):
    def __init__(
        self,
        coords=None,
        nchan=1,
        base_cube=None,
    ):
        super().__init__()

        self.coords = coords
        self.nchan = nchan

        self.bcube = images.BaseCube(
            coords=self.coords, nchan=self.nchan, base_cube=base_cube
        )

        self.conv_layer = images.HannConvCube(nchan=self.nchan)

        self.icube = images.ImageCube(
            coords=self.coords, nchan=self.nchan, passthrough=True
        )
        self.nufft = fourier.NuFFT(coords=self.coords, nchan=self.nchan)

    @classmethod
    def from_image_properties(cls, cell_size, npix, nchan, base_cube):
        coords = coordinates.GridCoords(cell_size, npix)
        return cls(coords, nchan, base_cube)

    def forward(self, uu, vv):
        r"""
        Feed forward to calculate the model visibilities. In this step, a :class:`~mpol.images.BaseCube` is fed to a :class:`~mpol.images.HannConvCube` is fed to a :class:`~mpol.images.ImageCube` is fed to a :class:`~mpol.fourier.NuFFT` to produce the model visibilities.

        Returns: 1D complex torch tensor of model visibilities.
        """
        x = self.bcube()
        x = self.conv_layer(x)
        x = self.icube(x)
        vis = self.nufft(x, uu, vv)
        return vis


coords = coordinates.GridCoords(cell_size=0.005, npix=1028)
rml = SGDNet(coords=coords)
optimizer = torch.optim.SGD(rml.parameters(), lr=3e4)

loss_tracker = []
nepochs = 20
for epoch in range(nepochs):
    # mini-batches
    for i, batch in enumerate(train_dloader):
        uu, vv, data, weight = batch

        rml.zero_grad()

        # get the predicted visibilities for this batch
        vis = rml(uu, vv)

        # calculate a loss
        loss = losses.nll(vis, data, weight)

        loss_tracker.append(loss.item())
        print("Epoch: {:} Batch: {:} Loss {:}".format(epoch, i, loss.item()))

        writer.add_scalar("loss", loss.item(), i)

        # calculate gradients of parameters
        loss.backward()

        # update the model parameters
        optimizer.step()

    # after training, compute the validation loss in batches
    # for i, vbatch in enumerate(validation_dloader):
    # pass


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


# we're looking at 3.4.4: https://d2l.ai/chapter_linear-regression/linear-regression-scratch.html#training
# what is this doing?
# for batch in self.val_dataloader:
# with torch.no_grad():
#     self.model.validation_step(self.prepare_batch(batch))
# self.val_batch_idx += 1


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="IM Lup SGD Example")
    parser.add_argument("adsf", help="Input path to .asdf file containing visibilities.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000000,
        help="input batch size for training",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100000,
        help="input batch size for testing",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
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
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    fname = "imlup.asdf"
    uu, vv, data, weight = loaddata.get_basic_data(fname)

    nvis = len(uu)
    print("Number of Visibility Points:", nvis)
    # 42 million

    # create indicies for each data point
    train_frac = 0.8
    full_indices = np.arange(len(uu))

    # randomly split into train and validation sets
    train_size = round(len(uu) * train_frac)
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
        )

    train_dataset = init_dataset(train_indices)
    validation_dataset = init_dataset(validation_indices)

    # create a DataLoader to feed batches of Dataset
    n_batches_per_dataset = 1000
    import math

    batch_size = math.ceil(nvis / n_batches_per_dataset)
    print("Batch size", batch_size)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True
    )

    # set up TensorBoard instance
    writer = SummaryWriter()

    # TODO: see how long NuFFT scales with number of data points, on CPU and GPU
    # TODO: see what the variation in number of baselines / dirty image is for each batch... is it informative?
    # TODO: how many epochs do we need to converge to something reasonable?
    # TODO: potentially investigate multi-core data-loader + parallelism for CPUs

    writer.add_image("images", grid, 0)
    writer.add_graph(model, images)
    writer.close()


if __name__ == "__main__":
    main()
