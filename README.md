# Perfusion-Analysis-Toolbox
Compute various perfusion parameters given a 4D perfusion image. 

## 1. Functions
a) Start from main.py, set correct parameters in config.py 
b) Set correct file paths in paths.py; 
    i) FileName -> Input CTP image (4D)
    ii) MaskName -> Mask (3D)
    iii) VesselName -> Vessel Centerline Image (3D)
c) main_calculator.py: aggregate the main parameters calculators in ./ParamsCalculator;
d) To view gamma variate fit for chosen AIF - navigate to src and open high-res-aif.png;

## 2. Python Packages
Install required packages:

```
pip install itk
pip install tensorboardX
pip install nibabel

```
## 3. Usage 

```
cd path/to/this/folder
python main.py
```
## 4. References
1. Peruzzo, Denis, et al. “Automatic Selection of Arterial Input Function on Dynamic Contrast-Enhanced MR Images.” Computer Methods and Programs in Biomedicine, vol. 104, no. 3, 2011, https://doi.org/10.1016/j.cmpb.2011.02.012.
2. https://github.com/marcocastellaro/dsc-mri-toolbox
3. https://github.com/KitwareMedical/ITKTubeTK-CTHead-Perfusion/tree/main
4. https://github.com/peirong26/Perfusion-Analysis-Toolbox

