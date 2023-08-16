#!/usr/bin/env python3

"""
Command line utility to compute perfusion parameters over a given CT input image.
"""

import torch

from .lib.utils import datestr
from .lib.signal_reader import read_signal
from .lib.main_calculator import MainCalculator
from .config import parse_config


def main():
    """
    Application entrypoint.

    Parse arguments, load and preprocess data, and compute perfusion parameters.
    """
    print(datestr())
    config = parse_config()

    device = torch.device("cpu")

    # Read and preprocess (brain-region extraction, low-pass filtering)
    # raw signal (size: (slice, row, column, time))
    RawPerfImg, mask, vessels, origin, spacing, direction = read_signal(
        config.image_path,
        config.mask_path,
        config.vessel_path,
        ToTensor=config.to_tensor,
    )

    # Calculate perfusion parameters
    calculator = MainCalculator(
        RawPerfImg,
        mask,
        vessels,
        origin,
        spacing,
        direction,
        config,
        config.save_path,
        device,
    )
    calculator.run()


########################################################################################################################

if __name__ == "__main__":
    main()
