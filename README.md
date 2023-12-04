# 2D Transdimensional Reversible Jump McMC Tomography

This Python program was used to create the azimuthal anisotropic phase-velocity maps in

**Anisotropic reversible-jump MCMC shear-velocity tomography of the eastern Alpine crust (Kästle and Tilmann, 2024).**

The scripts are mostly documented, but some options are still under development and thus not yet properly described. If you plan to use these scripts in your work, please cite the article above.

The functionality of the program goes beyond what is published in the article, there are different options that were not used (sensitivity kernels, parallel tempering, etc.). Many of those are still in an experimental stage.

## Installation

You need a Python (3.xx) installation and the following Python packages (I recommend installing Anaconda and installing the packages with __conda install package_name__):
numpy
scipy
mpi4py
matplotlib
cartopy
pyproj
gzip
cmcrameri (for colormaps https://pypi.org/project/cmcrameri/)
scikit-fmm (Python implementation of the fast-marching method https://github.com/scikit-fmm/scikit-fmm)
pyekfmm (optional, for azimuthally anisotropic ray tracing https://github.com/aaspip/pyekfmm)
pywt (optional, for wavelet parameterization)

Once these packages are installed, the tomography scripts simply have to be copied to the folder where they are supposed to be executed. You can run them from a console window by typing

_starting a model search_
```bash
python run_rjtransdim2d.py
```

_starting a model search on 8 parallel cores_
```bash
mpirun -np 8 python run_rjtransdim2d.py
```

## Scripts

__run_rjtransdim2d.py__
This is the main script that contains all the user parameters. You have to modify the search parameters and priors in the header of this script. All available options are explained inside the script.

__rjtransdim2d_helper_classes.py__
The functions and classes in this script control all operations that are necessary to set up the model search and that deal with communication between parallel processes. This includes distributed calculation of ray paths/kernels, parallel tempering, averaging of final models, etc.

__rjtransdim2d.py__
This script contains the class for handling a single model search (=chain). It takes the parameters and priors defined by the user and sets up the model search. Proposing model updates (birth, death, move, etc.) is handled by this class, followed by doing the forward calculation (travel times from proposed model), calculating the likelihood and the acceptance term. Accepted models are stored within each chain.

__rjtransdim2d_parameterization.py__
This script contains several classes, each describing a different parameterization (Voronoi cells = nearest neighbor interpolation, Delaunay triangulation = linear interpolation, blocks, wavelets, etc.). Users can create their own parameterization and add them to this file.

__rjtransdim2d_plot.py__
This is just a convenience script that calls the plot routing in __rjtransdim2d_helper_classes.py__. Modify path inside the file and run by typing 'python rjtransdim2d_plot.py' in a console window.

__FMM.py__
A collection on functions necessary to calculate the ray paths or the sensitivity kernels.

## Examples

There are two example datasets in the folder

__dataset_eastalps.dat__
__synthetic_measurements_rayleigh.txt__

These are subsets of the original datasets used in the publication of Kaestle and Tilmann, 2023. You can use them to reproduce Figs.4 and 8 (partially). You should be able to use them for a test run of the program.


## Further information

Feel free to contact E. Kaestle in case of questions.
