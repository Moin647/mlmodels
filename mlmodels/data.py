from __future__ import print_function

import os
from pathlib import Path


def import_data_tch(name="", mode="train", node_id=0, data_folder_root=""):
    from torchvision import datasets, transforms

    if name == "mnist":
        data_folder = os.path.join(data_folder_root, "data-%d" % node_id)
        dataset = datasets.MNIST(
            data_folder,
            train=True if mode == "train" else False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return dataset


def import_data_fromfile(**kw):
    """
        data_pars["data_path"]

    """

    m = kw
    extension = Path(m['data_path']).suffix.lower()
    if m.get("use_dask", False):
        import dask.dataframe as dd
        if extension in [".csv", ".txt"]: df = dd.read_csv(m["data_path"])
        elif extension in [".pkl"]: df = dd.read_pickle(m["data_path"])
        elif extension in [".npz"]: df = dd.read_pickle(m["data_path"])
        else: raise Exception(f"Not support extension {extension}")
        return df
    else:
        import pandas as pd
        import numpy as np
        if extension in [".csv", ".txt"]: df = pd.read_csv(m["data_path"])
        elif extension in [".pkl"]: df = pd.read_pickle(m["data_path"])
        elif extension in [".npz"]: df = np.load(m["data_path"])
        else:raise Exception(f"Not support extension {extension}")
        return df
