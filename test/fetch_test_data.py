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
        "Data/CTP04_CT_BrainMask_15x15x5.nii.gz",
        "https://data.kitware.com/api/v1/item/6614490b2357cf6b55ca8bfc/download",
    ),
    TestDataFile(
        "image",
        "--image-path",
        "Data/CTP04-4DCTP_Registered_15x15x5.nii.gz",
        "https://data.kitware.com/api/v1/item/661449122357cf6b55ca8bff/download",
    ),
    TestDataFile(
        "vessel-image",
        "--vessel-path",
        "Data/CTP04_CT_VesselCenterlinesMask_15x15x5.nii.gz",
        "https://data.kitware.com/api/v1/item/66144af22357cf6b55ca8c06/download",
    ),
]


def fetch_test_data():
    """Fetch test data from the server to the local disk if not already available"""
    for file_info in test_data_info:
        if not os.path.exists(file_info.filepath):
            os.makedirs(os.path.dirname(file_info.filepath), exist_ok=True)
            urlretrieve(file_info.url, file_info.filepath)

if __name__ == "__main__":
    fetch_test_data()
