# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from pathlib import Path

from tqdm import tqdm
from torchaudio.datasets.utils import _extract_tar
from torch.hub import download_url_to_file
from joblib import Parallel, delayed
from functools import partial
from voxpopuli import LANGUAGES, LANGUAGES_V2, YEARS, DOWNLOAD_BASE_URL


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", "-r", type=str, required=True, help="data root path"
    )
    parser.add_argument(
        "--subset", "-s", type=str, required=True,
        choices=["400k", "100k", "10k", "asr"] + LANGUAGES + LANGUAGES_V2,
        help="data subset to download"
    )
    parser.add_argument("--num_workers", "-j", type=int, default=4,
                        help="number of workers to download data")
    return parser.parse_args()


def process_url(url, out_root):
    print(f"Downloading {url}...")
    tar_path = out_root / Path(url).name
    download_url_to_file(url, tar_path.as_posix(), hash_prefix=None)
    _extract_tar(tar_path.as_posix())
    os.remove(tar_path)
    print(f"Downloaded and extracted {url} to {tar_path}")
    return tar_path


def download(args):
    if args.subset in LANGUAGES_V2:
        languages = [args.subset.split("_")[0]]
        years = YEARS + [f"{y}_2" for y in YEARS]
    elif args.subset in LANGUAGES:
        languages = [args.subset]
        years = YEARS
    else:
        languages = {
            "400k": LANGUAGES,
            "100k": LANGUAGES,
            "10k": LANGUAGES,
            "asr": ["original"]
        }.get(args.subset, None)
        years = {
            "400k": YEARS + [f"{y}_2" for y in YEARS],
            "100k": YEARS,
            "10k": [2019, 2020],
            "asr": YEARS
        }.get(args.subset, None)

    url_list = []
    for l in languages:
        for y in years:
            url_list.append(f"{DOWNLOAD_BASE_URL}/audios/{l}_{y}.tar")

    out_root = Path(args.root) / "raw_audios"
    out_root.mkdir(exist_ok=True, parents=True)
    print(f"{len(url_list)} files to download...")

    process_func = partial(process_url, out_root=out_root)

    results = Parallel(n_jobs=args.num_workers)(delayed(process_func)(url) for url in url_list)
    # for url in tqdm(url_list):
    #     tar_path = out_root / Path(url).name
    #     download_url_to_file(url, tar_path.as_posix(), hash_prefix=None)
    #     _extract_tar(tar_path.as_posix())
    #     os.remove(tar_path)


def main():
    args = get_args()
    download(args)


if __name__ == '__main__':
    main()
