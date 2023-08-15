#!/usr/bin/env python3

"""
Purpose: Validate that perfusion image processing pipeline
         returns expected results for expected inputs
"""

import sys
import os
import unittest.mock

import itk

from fetch_test_data import test_data_info, fetch_test_data

sys.path.append("src")
itk.auto_progress(2)

TEST_OUTPUT_DIR = "Data/output"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
fetch_test_data()


def test_import():
    """Basic test to verify imports succeed"""
    import perfusion_analysis_toolbox.main  # noqa: F401
    import perfusion_analysis_toolbox.config  # noqa: F401


def test_calc():
    from perfusion_analysis_toolbox.main import main

    argv = ["pyproject.toml", "--save-path", TEST_OUTPUT_DIR]
    for data_info in test_data_info:
        argv.append(data_info.config_param)
        argv.append(data_info.filepath)

    with unittest.mock.patch("sys.argv", argv):
        print(f"Mock sys.argv: {sys.argv}")
        main()

    # TODO validate images
