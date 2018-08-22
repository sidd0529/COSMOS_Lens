# COSMOS_Lens

## What does this project do?
The codes in this repository use a combination of C and Python codes, and Cython as a bridge to generate convergence and lensing maps from the COSMOS2015 galaxy catalog (https://arxiv.org/abs/1604.02350). 

## Why is this project useful?
This repository is useful for astrophysicists who are working with precision weak lensing measurements from COSMOS2015 and other cosmological surveys. The codes in this repository are highly optimized, and analyze half a million galaxies inside a few (~1) minutes.

Below is a picture of a convergence map generated by scripts in this code:

<img src="https://user-images.githubusercontent.com/26308648/44443139-3360c680-a5a4-11e8-84ec-3c80dcf7b6ee.png" width="620">

The potential map obtained from the aforementioned convergence map is shown below:

<img src="https://user-images.githubusercontent.com/26308648/44443160-5a1efd00-a5a4-11e8-95d6-026b6c80e475.png" width="620">

 ## How to get started with this project?
 ```
 $ git clone https://github.com/sidd0529/COSMOS_Lens.git
 ```
 
 Download COSMOS2015 data from: ``` ftp://ftp.iap.fr/pub/from_users/hjmcc/COSMOS2015/ ```
 
 Extract ```COSMOS2015_Laigle_v1.1.fits``` from ```COSMOS2015_Laigle_v1.1.fits.gz``` using:
 
 ```
 $ gunzip COSMOS2015_Laigle_v1.1.fits.gz
 ```
 
 Run the code using:
 
 ```
 $ python setup.py build_ext --inplace
 ```
 
 ## Where can you get help with this project?
 I will be very happy to help in case you have any questions regarding this project. You can find me at siddharthsatpathy.ss@gmail.com .

