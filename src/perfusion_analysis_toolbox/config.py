#!/usr/bin/env python3

"""
Helper utility to parse input parameters for a given run.
"""

import argparse


def parse_config() -> argparse.Namespace:
    """
    Parse command line arguments to configure the run.
    """

    parser = argparse.ArgumentParser(description="CTP Colormaps Calculation")

    parser.add_argument(
        "--mask-path", type=str, required=True, help="Path to mask image"
    )
    parser.add_argument(
        "--image-path", type=str, required=True, help="Path to input image"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to folder to save results",
    )
    parser.add_argument(
        "--vessel-path",
        type=str,
        required=True,
        help="Path to vessel centerlines input file",
    )

    parser.add_argument(
        "--image_type", type=str, default="CTP", help="Image type: CTP/MRP"
    )

    ################## Constants Settings ##################
    parser.add_argument(
        "--k_ct",
        type=float,
        default=1.0,
        help="Constant k_ct (g/ml/HU) for CTP",
    )
    parser.add_argument(
        "--k_mr", type=float, default=1.0, help="Constant k_mr for MRP"
    )
    parser.add_argument(
        "--TE", type=float, default=0.025, help="Constant TE (ms) for MRP"
    )
    parser.add_argument(
        "--TR", type=float, default=1.55, help="Constant TR (s) for MRP"
    )

    parser.add_argument(
        "--use_filter",
        type=bool,
        default=True,
        help="Whether use low-pass filtering for CTC. "
        "Usually need filter for MRP, no need for CTP.",
    )
    parser.add_argument(
        "--mrp_s0_threshold",
        type=float,
        default=0.05,
        help="Threshold for finding MRP bolus arrival time ",
    )
    parser.add_argument(
        "--ctp_s0_threshold",
        type=float,
        default=0.05,
        help="Threshold for finding CTP bolus arrival time ",
    )
    parser.add_argument(
        "--to_tensor",
        type=bool,
        default=True,
        help="Whether need to convert to torch.tensor",
    )
    parser.add_argument(
        "--mask",
        type=list,
        default=[[], [0, 138], [0, 138]],
        help="Used as BackGround Code for MRP, while BrainMask -300 for CTP (UNC)",
    )

    args = parser.parse_args()

    return args
