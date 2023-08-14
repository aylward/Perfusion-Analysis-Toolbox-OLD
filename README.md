# Perfusion-Analysis-Toolbox
Compute perfusion parameters for a 4D perfusion image using an Arterial Input Function extracted from vessel centerlines.

## Prerequisites

- Git version control
- Python 3.8 or later
- `itk`, `tensorboardX`, and `nibabel` PyPI packages

Run the following command in a shell prompt to install Python dependencies from the Python Package Index (PyPI):
```sh
python -m pip install itk tensorboardX nibabel
```

## Setup

Follow these steps to run `Perfusion-Analysis-Toolbox` scripts on your local data:

### 1. Clone the project to your PC
```sh
git clone git@github.com:KitwareMedical/Perfusion-Analysis-Toolbox.git
```

### 2. (Optional) Retrieve sample perfusion imaging data

Sample data files are provided at [`data.kitware.com`](https://data.kitware.com/#folder/64da58971c6956f5031e433d).
We recommend downloading [Perfusion-Image-Input-Data.zip](https://data.kitware.com/api/v1/item/64da58e11c6956f5031e4354/download)
and extracting its contents into `Perfusion-Analysis-Toolbox/Data`.

Alternatively, you may provide your own CT images for perfusion analysis.

### 3. Load Data and Set Parameters
1) Set correct parameters in src/config.py. 
2) Set correct file paths in src/paths.py.
    1) FileName -> Input CTP image (4D)
    2) MaskName -> Mask (3D)
    3) VesselName -> Vessel Centerline Image (3D)

### 4. Run Code
```sh
cd path/to/this/folder/src
python main.py
```

### 5. View Results
1) CBV, CBF, CTC and MTT images are saved under the src folder.
2) To view gamma variate fit for the chosen AIF, navigate to src and open high-res-aif.png.

## 5. References
1. Peruzzo, Denis, et al. “Automatic Selection of Arterial Input Function on Dynamic Contrast-Enhanced MR Images.” Computer Methods and Programs in Biomedicine, vol. 104, no. 3, 2011, https://doi.org/10.1016/j.cmpb.2011.02.012.
2. https://github.com/marcocastellaro/dsc-mri-toolbox
3. https://github.com/KitwareMedical/ITKTubeTK-CTHead-Perfusion/tree/main
4. https://github.com/peirong26/Perfusion-Analysis-Toolbox

