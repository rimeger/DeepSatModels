import geopandas as gpd
import os
import numpy as np
import pickle
import datetime
import torch
import argparse
import json


def get_doy(date):
    date = str(date)
    Y = date[:4]
    m = date[4:6]
    d = date[6:]
    date = "%s.%s.%s" % (Y, m, d)
    dt = datetime.datetime.strptime(date, '%Y.%m.%d')
    return dt.timetuple().tm_yday


def unfold_reshape(img, HW):
    # img can be (T,C,H,W), (C,H,W), or (H,W)

    if len(img.shape) == 4:              # T,C,H,W
        T, C, H, W = img.shape
        img = img.unfold(2, HW, HW).unfold(3, HW, HW)
        img = img.reshape(T, C, -1, HW, HW).permute(2, 0, 1, 3, 4)

    elif len(img.shape) == 3:            # C,H,W (rare for your dataset)
        C, H, W = img.shape
        img = img.unfold(1, HW, HW).unfold(2, HW, HW)
        img = img.reshape(C, -1, HW, HW).permute(1, 0, 2, 3)

    elif len(img.shape) == 2:            # H,W  ← labels!
        H, W = img.shape
        img = img.unfold(0, HW, HW).unfold(1, HW, HW)
        img = img.reshape(-1, HW, HW)

    else:
        raise ValueError("Unsupported shape: " + str(img.shape))

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CSCL pre-training')
    parser.add_argument('--rootdir', type=str, default="", help='PASTIS24 root dir')
    parser.add_argument('--savedir', type=str, default="", help='where to save new data')
    parser.add_argument('--HWout', type=int, default=24, help='size of extracted windows')
    args = parser.parse_args()

    rootdir = args.rootdir
    savedir = args.savedir
    HWin = 128
    HWout = args.HWout

    meta_patch = gpd.read_file(os.path.join(rootdir, "metadata.geojson"))

    for i in range(meta_patch.shape[0]):
        print(f'doing file {i} of {meta_patch.shape[0]}')

        patch_id = meta_patch['ID_PATCH'].iloc[i]

        img = np.load(os.path.join(rootdir, f'DATA_S2/S2_{patch_id}.npy'))
        lab = np.load(os.path.join(rootdir, f'ANNOTATIONS/TARGET_{patch_id}.npy'))
        ids = np.load(os.path.join(rootdir, f'ANNOTATIONS/ParcelIDs_{patch_id}.npy'))

        dates = meta_patch['dates-S2'].iloc[i]
        if isinstance(dates, str):
            dates = json.loads(dates)

        doy = np.array([get_doy(d) for d in dates.values()])
        idx = np.argsort(doy)

        img = img[idx]
        doy = doy[idx]

        unfolded_images = unfold_reshape(torch.tensor(img), HWout).numpy()
        unfolded_labels = unfold_reshape(torch.tensor(lab), HWout).numpy()

        # FIXED iteration
        for j in range(unfolded_images.shape[0]):
            sample = {
                'img': unfolded_images[j],
                'labels': unfolded_labels[j],
                'doy': doy.copy()
            }

            with open(os.path.join(savedir, f"{patch_id}_{j}.pickle"), "wb") as output_file:
                pickle.dump(sample, output_file)
