# Perfusion-Analysis-Toolbox
Compute perfusion parameters for a 4D perfusion image using an Arterial Input Function extracted from vessel centerlines.

## 1. Load Data and Set Parameters
1) Set correct parameters in src/config.py. 
2) Set correct file paths in src/paths.py.
    1) FileName -> Input CTP image (4D)
    2) MaskName -> Mask (3D)
    3) VesselName -> Vessel Centerline Image (3D)

## 2. Install Python Packages
Install required packages:

```
pip install itk
pip install tensorboardX
pip install nibabel

```
## 3. Run Code
```
cd path/to/this/folder/src
python main.py
```
## 4. View Results
1) CBV, CBF, CTC and MTT images are saved under the src folder.
2) To view gamma variate fit for the chosen AIF, navigate to src and open high-res-aif.png.

## 5. References
1. Peruzzo, Denis, et al. “Automatic Selection of Arterial Input Function on Dynamic Contrast-Enhanced MR Images.” Computer Methods and Programs in Biomedicine, vol. 104, no. 3, 2011, https://doi.org/10.1016/j.cmpb.2011.02.012.
2. https://github.com/marcocastellaro/dsc-mri-toolbox
3. https://github.com/KitwareMedical/ITKTubeTK-CTHead-Perfusion/tree/main
4. https://github.com/peirong26/Perfusion-Analysis-Toolbox

