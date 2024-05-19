import asdf
import numpy as np 
import visread.process
import argparse

def get_basic_data(filename):
    """Load all of the data contained in the .asdf file, and convert it into 1D. 
    Destroys channel information.
    
    Returns:
        uu, vv, data, weight as 1D arrays.
    """
    
    data_list = []
    weight_list = []
    uu_list = []
    vv_list = []

    with asdf.open(filename) as af:
        # print(af.info())

        # obsids contained in measurement set
        obsids = af["obsids"].keys()
        for obsid in obsids:
            # ddids contained in each obsid
            ddids = af["obsids"][obsid]["ddids"].keys()
            for ddid in ddids:
                d = af["obsids"][obsid]["ddids"][ddid]
                # each ddid should have the following keys
                # "frequencies", "uu", "vv", "antenna1", "antenna2", 
                # "time", "data, "flag", "weight"

                # but we'll load only what we need for now
                # frequencies should be (nchan,)
                # uu and vv should each be (nvis,)
                # flag should be (nchan, nvis)
                # data should be (nchan, nvis)
                # weight should be (nvis,) assuming the same for all channels
                
                # convert uu, vv from meters to klambda
                uu, vv = visread.process.broadcast_and_convert_baselines(d["uu"], d["vv"], d["frequencies"])

                # broadcast weights
                weight = visread.process.broadcast_weights(d["weight"], d["data"].shape)

                # flag relevant baselines, data, and weight
                flag = d["flag"]
                
                uu = uu[~flag] # keep the good ones
                vv = vv[~flag]

                data = d["data"][~flag]
                weight = weight[~flag]

                # destroy channel axis and concatenate
                uu_list.append(uu.flatten())
                vv_list.append(vv.flatten())
                data_list.append(data.flatten())
                weight_list.append(weight.flatten())

    # concatenate all files at the end
    uu = np.concatenate(uu_list)
    vv = np.concatenate(vv_list)
    data = np.concatenate(data_list)
    weight = np.concatenate(weight_list)

    return uu, vv, data, weight

def get_ddid_data(filename, sb_only=False):
    """Load all of the data contained in the .asdf file, and convert it into 1D. 
    Provides ddid argument, too.
    Destroys channel information.
    
    Returns:
        uu, vv, data, weight as 1D arrays.
    """
    
    data_list = []
    weight_list = []
    uu_list = []
    vv_list = []
    ddid_list = []

    with asdf.open(filename) as af:
        # print(af.info())

        # obsids contained in measurement set
        obsids = af["obsids"].keys()
        for obsid in obsids:

            if sb_only and (obsid > 4):
                continue
            # ddids contained in each obsid
            ddids = af["obsids"][obsid]["ddids"].keys()
            for ddid in ddids:
                d = af["obsids"][obsid]["ddids"][ddid]
                # each ddid should have the following keys
                # "frequencies", "uu", "vv", "antenna1", "antenna2", 
                # "time", "data, "flag", "weight"

                # but we'll load only what we need for now
                # frequencies should be (nchan,)
                # uu and vv should each be (nvis,)
                # flag should be (nchan, nvis)
                # data should be (nchan, nvis)
                # weight should be (nvis,) assuming the same for all channels
                
                # convert uu, vv from meters to klambda
                uu, vv = visread.process.broadcast_and_convert_baselines(d["uu"], d["vv"], d["frequencies"])

                # broadcast weights
                weight = visread.process.broadcast_weights(d["weight"], d["data"].shape)

                # flag relevant baselines, data, and weight
                flag = d["flag"]
                
                uu = uu[~flag] # keep the good ones
                vv = vv[~flag]

                data = d["data"][~flag]
                weight = weight[~flag]

                # destroy channel axis and concatenate
                uu_list.append(uu.flatten())
                vv_list.append(vv.flatten())
                data_list.append(data.flatten())
                weight_list.append(weight.flatten())
                ddid_list.append(ddid * np.ones_like(uu.flatten(), dtype=np.int64))
            



    # concatenate all files at the end
    uu = np.concatenate(uu_list)
    vv = np.concatenate(vv_list)
    data = np.concatenate(data_list)
    weight = np.concatenate(weight_list)
    ddid = np.concatenate(ddid_list)

    return uu, vv, data, weight, ddid


def main():
    parser = argparse.ArgumentParser(
        description="Load an .asdf file."
    )
    parser.add_argument("file", help="Path to asdf file") 
    args = parser.parse_args()

    uu, vv, data, weight = get_basic_data(args.file)
    print("uu", uu)
    print("vv", vv)
    print("data", data)
    print("weight", weight)

    print("dtype", uu.dtype)

    print("Number of visibilities", len(data))

if __name__=="__main__":
    main()