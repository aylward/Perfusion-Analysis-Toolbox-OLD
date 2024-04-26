# Perfusion-Analysis-Toolbox
Compute perfusion parameters for a 4D perfusion image using an Arterial Input Function extracted from vessel centerlines.

## Prerequisites

- Git version control
- Python 3.8 or later

## Setup

Follow these steps to run `Perfusion-Analysis-Toolbox` scripts on your local data:

### 1. Clone the project to your PC
```sh
git clone git@github.com:KitwareMedical/Perfusion-Analysis-Toolbox.git
```

### 2. Install the project and its dependencies

```sh
python -m pip install path/to/Perfusion-Analysis-Toolbox
```

### 3. (Optional) Retrieve sample perfusion imaging data

Sample data files are provided at [`data.kitware.com`](https://data.kitware.com/#collection/661410cf2357cf6b55ca8bda).

We recommend downloading that data into the directory
```
Perfusion-Analysis-Toolbox/Data
```

In the Data directory, you can run the provided data-download.sh script to download the testing / demonstration data:
```sh
pip install girder-client
./data-download.sh
```

Alternatively, you may provide your own 4D_CTP, 3D_CT_BrainMask, and 3D_CT_VesselCenterlinesMask images for perfusion analysis.

### 4. Run the Command Line Application

```sh
perfusion-compute-cli --image-path <image-path> --mask-path <mask-path> --vessel-path <vessel-path> --save-path <save-path>
```
or

```sh
python -m perfusion-compute-cli --image-path <path/to/4D-CT-Perfusion-Image> --mask-path <path/to/3D-Mask-Image> --vessel-path <path/to/3D-Vessel-Centerline-Image> --save-path <path/to/save-directory>
```

To run the sample data, once it has been downloaded (see above), type

```sh
mkdir Results
perfusion-compute-cli --image-path ./Data/CTP04_4DCTP_Registered_15x15x5.nii.gz --mask-path ./Data/CTP04_CT_BrainMask_15x15x5.nii.gz --vessel-path ./Data/CTP04_CT_VesselCenterlinesMask_15x15x5.nii.gz --save-path Results
```

### 5. View Results
1) CBV, CBF, CTC and MTT images are saved under the `--save-path` folder.
2) To view gamma variate fit for the chosen AIF, navigate to src and open high-res-aif.png.

## Contributing

Community contributions are welcome. Follow these steps to contribute to `Perfusion-Analysis-Toolbox`:

1. Fork the `Perfusion-Analysis-Toolbox` repository under your user account on GitHub.
2. Clone the project to your PC.
3. Make local code changes.
4. Install developer tools.
```sh
python -m pip install path/to/Perfusion-Analysis-Toolbox[develop]
```
5. Run auto-formatting with `black` and `ruff` linting tools.
```sh
black ./src
ruff check ./src --fix --show-fixes
```
6. Commit changes to a development branch under your user account.
7. Create a [pull request](https://github.com/KitwareMedical/Perfusion-Analysis-Toolbox/pulls).

## References
1. Peruzzo, Denis, et al. “Automatic Selection of Arterial Input Function on Dynamic Contrast-Enhanced MR Images.” Computer Methods and Programs in Biomedicine, vol. 104, no. 3, 2011, https://doi.org/10.1016/j.cmpb.2011.02.012.
2. https://github.com/marcocastellaro/dsc-mri-toolbox
3. https://github.com/KitwareMedical/ITKTubeTK-CTHead-Perfusion/tree/main
4. https://github.com/peirong26/Perfusion-Analysis-Toolbox

