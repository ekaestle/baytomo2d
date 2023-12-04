#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:16:33 2021

@author: emanuel
"""

import rjtransdim2d_helper_classes as rhc
import os, glob

folderlist = glob.glob("./rayleigh_zz_aniso_2.5")

for filepath in folderlist:
    # folder with chain files
    #filepath = "./results_curta/rayleigh_zz_anisotropic_staggered_20.0"
    # plot also single chain output images
    plotsinglechains = True
    # specify the projection of the X,Y coordinates in the chain files
    projection = None # recommended: None, i.e. read from chain file
    
    
    if not os.path.exists(filepath):
        raise Exception("path",filepath,"does not exist.")
    
    rhc.plot_avg_model(filepath,projection=projection,plotsinglechains=plotsinglechains)
