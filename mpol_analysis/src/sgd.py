import numpy as np
import loaddata
import torch
import argparse
from torch.utils.data import TensorDataset, DataLoader
from mpol import coordinates, fourier, images, losses

from torch.utils.tensorboard import SummaryWriter

# following structure from https://github.com/pytorch/examples/blob/main/mnist/main.py

# create a model that uses the NuFFT to predict
class Net(torch.nn.Module):
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


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for i_batch, (uu, vv, data, weight) in enumerate(train_loader):
        # send all values to device
        uu, vv, data, weight = uu.to(device), vv.to(device), data.to(device), weight.to(device)
        
        optimizer.zero_grad()
        # get model visibilities
        vis = model(uu, vv)

        # calculate loss
        loss = losses.nll(vis, data, weight)
        loss.backward()
        optimizer.step()

        # log results
        if i_batch % args.log_interval == 0:
            writer.add_scalar("loss", loss.item())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i_batch * len(data), len(train_loader.dataset),
                100. * i_batch / len(train_loader), loss.item()))
            if args.dry_run:
                break

def validate(model, device, validate_loader):
    model.eval()
    validate_loss = 0
    with torch.no_grad():
        for (uu, vv, data, weight) in validate_loader:
            # send all values to device
            uu, vv, data, weight = uu.to(device), vv.to(device), data.to(device), weight.to(device)
                    
            # get model visibilities
            vis = model(uu, vv)

            # calculate loss as total over all chunks
            validate_loss += losses.nll(vis, data, weight).item()  
    
    # re-normalize to total number of data points
    validate_loss /= len(validate_loader.dataset)
    return validate_loss



def main():
    # Training settings
    parser = argparse.ArgumentParser(description="IM Lup SGD Example")
    parser.add_argument("asdf", help="Input path to .asdf file containing visibilities.")
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
        "--train-fraction", type=float, default=0.8, help="The fraction of the dataset to use for training. The remainder will be used for validation."
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
        "--tensorboard-log-dir",
        help="The log dir to which tensorboard files should be written."
    )
    parser.add_argument(
        "--save-path",
        help="Provide path to save the current model",
    )
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
    uu, vv, data, weight = loaddata.get_basic_data(args.asdf)

    nvis = len(uu)
    print("Number of Visibility Points:", nvis)
    # 42 million

    # split the dataset into train / test by creating a list of indices
    assert (args.train_fraction > 0) and (args.train_fraction < 1), "--train-fraction must be greater than 0 and less than 1.0. You provided {}".format(args.train_fraction)
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
        )

    train_dataset = init_dataset(train_indices)
    validation_dataset = init_dataset(validation_indices)

    # set the batch sizes for the loaders
    train_kwargs = {"batch_size": args.batch_size}
    validation_kwargs = {"batch_size": args.validation_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    validation_loader = DataLoader(
        validation_dataset, **validation_kwargs
    )

    # create the model and send to device
    coords = coordinates.GridCoords(cell_size=0.005, npix=1028)
    model = Net(coords).to(device)

    # create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # here is where we could set up a scheduler, if desired

    # set up TensorBoard instance
    writer = SummaryWriter(log_dir=args.tensorboard_log_dir)

    # enter the loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        vloss_avg = validate(model, device, validation_loader)
        writer.add_scalar("vloss_avg", vloss_avg)
        optimizer.step()

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)


    # TODO: see how long NuFFT scales with number of data points, on CPU and GPU
    # TODO: see what the variation in number of baselines / dirty image is for each batch... is it informative?
    
    writer.close()


if __name__ == "__main__":
    main()
