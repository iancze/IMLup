import numpy as np
import loaddata
import torch
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Sampler
from mpol import coordinates, gridding, fourier, images, losses, utils, plot

def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Implement and test the per-EB dataloader")
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
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    
    args = parser.parse_args()


    # load the full dataset
    uu, vv, data, weight, ddid = loaddata.get_ddid_data(args.asdf)

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
    train_ddids = ddid[train_indices]
    validation_dataset = init_dataset(validation_indices)

    # set the batch sizes for the loaders
    train_kwargs = {"batch_size": args.batch_size}
    validation_kwargs = {"batch_size": args.validation_batch_size}
    
    cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_kwargs.update(cuda_kwargs)
    validation_kwargs.update(cuda_kwargs)

    # https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
    # the sampler is the object that chooses which indices to provide in each mini-batch
    # want to pass this to batch_sampler argument instead of sampler

    # define a new type of sampler
    # we want each batch to have exactly the same ddid
    # but we might only want part of the ddid 
    # do we want a wrapped sampler?

    class EBBatchSampler(Sampler):
        def __init__(self, ddids, batch_size, shuffle=True):
            self.ddids = ddids
            self.ids = np.arange(len(ddids))
            self.batch_size = batch_size
            self.shuffle = shuffle
            
            # create an index of ddids that exist
            self.unique_ddids = np.unique(self.ddids)

            print("Batch size", self.batch_size)
            print("Unique ddids", self.unique_ddids)

            # calculate how many batches we will have for each ddid sub-group
            self.batches_per_ddid = {}
            for ddid in self.unique_ddids:
                # find all indices with ddid match
                ids_ddid = self.ids[(self.ddids == ddid)]
                # calculate the length
                nbatches = (len(ids_ddid) + self.batch_size - 1) // self.batch_size
                self.batches_per_ddid[ddid] = nbatches

            print("batches_per_didd", self.batches_per_ddid)

        def __len__(self):
            "returns the number of batches to cover all the data"
            return np.sum([b for b in self.batches_per_ddid.values()])

        def __iter__(self):
            # we want an iterator that will give us mini-batches containing the same ddid

            if self.shuffle:
                rng = np.random.default_rng()
                rng.shuffle(self.unique_ddids)

            for ddid in self.unique_ddids:
                print("iterating on ddid", ddid)
                # find all indices with ddid match
                ids_ddid = self.ids[(self.ddids == ddid)]

                for batch in torch.chunk(torch.tensor(ids_ddid), self.batches_per_ddid[ddid]):
                    yield batch.tolist()

    # these can now be iterated upon in train and validation loops
    # train_loader = DataLoader(train_dataset, **train_kwargs)
    # validation_loader = DataLoader(validation_dataset, **validation_kwargs)

    sampler = EBBatchSampler(train_ddids, args.batch_size)

    train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    print("Number of batches", len(train_loader))

    for i, batch in enumerate(train_loader):
        uu, vv, data, weight, ddid = batch 
        print("Batch {:} Num points in batch: {:} Unique ddids".format(i, len(ddid)), torch.unique(ddid))

    print("restarting")

    for i, batch in enumerate(train_loader):
        uu, vv, data, weight, ddid = batch 
        print("Batch {:} Num points in batch: {:} Unique ddids".format(i, len(ddid)), torch.unique(ddid))



if __name__=="__main__":
    main()