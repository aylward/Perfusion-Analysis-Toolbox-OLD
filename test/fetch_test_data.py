#!/usr/bin/env python3

import os
from dataclasses import dataclass
from urllib.request import urlretrieve


@dataclass
class TestDataFile:
    name: str
    config_param: str
    filepath: str
    url: str


test_data_info = [
    TestDataFile(
        "mask",
        "--mask-path",
        "Data/CTP04_ct_brain_mask_15x15x5.nii.gz",
        "https://data.kitware.com/api/v1/item/64da58ac1c6956f5031e4342/download",
    ),
    TestDataFile(
        "image",
        "--image-path",
        "Data/CTP04-4D_reg_15x15x5.nii.gz",
        "https://data.kitware.com/api/v1/item/64da58b31c6956f5031e4345/download",
    ),
    TestDataFile(
        "vessel-image",
        "--vessel-path",
        "Data/vesscenterlines.mha",
        "https://data.kitware.com/api/v1/item/64da58c41c6956f5031e4351/download",
    ),
]


def fetch_test_data():
    """Fetch test data from the server to the local disk if not already available"""
    for file_info in test_data_info:
        if not os.path.exists(file_info.filepath):
            os.makedirs(os.path.dirname(file_info.filepath), exist_ok=True)
            urlretrieve(file_info.url, file_info.filepath)
