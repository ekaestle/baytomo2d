#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:18:22 2018

@author: emanuel
"""

import numpy as np
#np.seterr(under='ignore')
#np.seterr(over='ignore')
from scipy.spatial import Delaunay
from scipy.ndimage import uniform_filter
from scipy.sparse import csc_matrix
from scipy.special import i0
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
import matplotlib, os
if not bool(os.environ.get('DISPLAY', None)):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import time
from copy import deepcopy
from cmcrameri import cm
import rjtransdim2d_parameterization


class RJMC(object):
    
    def __init__(self,chain_no,indata,prior,params):

        self.chain_no = chain_no
        self.logfile = os.path.join(params.logfile_path,"logfile_%d.txt" %(self.chain_no))
        with open(self.logfile,"w") as f:
            f.write("%s\nStarting chain %d\n\n" %(time.ctime(),self.chain_no))
        #print("Chain %d: initialize..." %self.chain_no)

        # interpolation method       
        self.interpolation_type = params.interpolation_type
        self.selfcheck=False
        if self.selfcheck:
            print("Warning: selfcheck turned on. Calculations will take much longer.")
            
        self.misfit_norm = params.misfit_norm
        self.outlier_model = params.outlier_model        
                
        #self.prior = prior
        # get all prior values
        self.velmin = prior.velmin
        self.velmax = prior.velmax
        self.min_datastd = prior.min_datastd
        self.max_datastd = prior.max_datastd        
        self.nmin_points = prior.nmin_points
        if not "nearest_neighbor" in self.interpolation_type and self.nmin_points<4:
            self.nmin_points=4 # minimum 4 corner points are needed
        self.nmax_points = prior.nmax_points
        
        self.projection = params.projection
        
        self.minx,self.maxx,self.miny,self.maxy = params.minx,params.maxx,params.miny,params.maxy
        self.x = np.arange(params.minx, params.maxx+params.xgridspacing, params.xgridspacing)
        self.y = np.arange(params.miny, params.maxy+params.ygridspacing, params.ygridspacing)
        self.xgridspacing = params.xgridspacing
        self.ygridspacing = params.ygridspacing
        if self.interpolation_type=='wavelets':
            xsize,ysize = rjtransdim2d_parameterization.wavelets.optimize_gridsize(
                self.minx,self.maxx,self.miny,self.maxy,
                self.xgridspacing,self.ygridspacing)
            self.x = np.linspace(self.minx,self.maxx,xsize)
            self.xgridspacing = self.x[1]-self.x[0]
            self.y = np.linspace(self.miny,self.maxy,ysize)
            self.ygridspacing = self.y[1]-self.y[0]
            if self.chain_no==1:
                print(f"Warning: setting xgridspacing from {params.xgridspacing} "+
                      f"to {self.xgridspacing} to make the x-axis length a "+
                      "multiple of 2 (necessary for the wavelet transform).")
                print(f"Warning: setting ygridspacing from {params.ygridspacing} "+
                      f"to {self.ygridspacing} to make the y-axis length a "+
                      "multiple of 2 (necessary for the wavelet transform).")
        X,Y = np.meshgrid(self.x,self.y)
        self.shape = X.shape
        self.gridpoints = np.column_stack((X.flatten(),Y.flatten()))
        #self.grid_kdtree = KDTree(self.gridpoints)
        # gridlines (lines between gridpoints, delimiting cells) 
        self.xgrid = np.append(self.x-params.xgridspacing/2.,
                               self.x[-1]+params.xgridspacing/2.)
        self.ygrid = np.append(self.y-params.ygridspacing/2.,
                               self.y[-1]+params.ygridspacing/2.)
        self.wavelength_dependent_gridspacing = params.wavelength_dependent_gridspacing
        
        
        # for the calculation of fat rays/Eikonal rays
        self.kernel_type = params.kernel_type
        self.period = params.period
        self.meanslow = 1./params.meanvel
        self.wavelength = self.period / self.meanslow
        
        self.error_type = params.data_std_type
        self.propsigmastd = params.propsigmastd
        
        self.propvelstd_dimchange = params.propvelstd_dimchange
        self.propvelstd_pointupdate = params.propvelstd_pointupdate
        self.propvelstd = params.propvelstd

        self.propmovestd = params.propmovestd
        self.temperature = params.temperature
        
        self.target_iterations = params.target_iterations
        self.nburnin = params.nburnin
        self.update_paths_interval = params.update_paths_interval
        
        self.delayed_rejection = params.delayed_rejection
        self.store_models = params.store_models
        
        self.init_no_points = params.init_no_points
        self.init_vel_points = params.init_vel_points
        
        self.print_stats_step = params.print_stats_step
            
        self.collect_step = params.collect_step
        
        self.anisotropic = params.anisotropic
        if prior.aniso_ampmin != 0.:
            print("Warning: minimum anisotropic amplitude is set to zero (originally set to %.3f)" %prior.aniso_ampmin)
            prior.aniso_ampmin = 0.
            # could otherwise cause problems if I start with a purely isotropic search
        if self.anisotropic:
            self.propstd_anisoamp = params.propstd_anisoamp
            self.propstd_anisodir = params.propstd_anisodir
            self.aniso_ampmin = prior.aniso_ampmin
            self.aniso_ampmax = prior.aniso_ampmax
        else:
            self.aniso_ampmin = self.aniso_ampmax = 0

        # get data
        self.data_units = 'velocity' # 'velocity' or 'travel_times'
        self.sources = {}
        self.receivers = {}
        self.data = {}
        self.data_std = {}
        self.model_predictions = {}
        self.idxselect = {}
        #self.COV = {}
        self.datasets = []
        if self.outlier_model:
            self.foutlier = {}
            self.outlierwidth = {}
            self.widthoutlier = 0.
        self.minresidual = 0 #.05
        if self.minresidual > 0.:
            print("Warning: Setting a minimum residual of",self.minresidual)
            print("Data residuals can therefore not be lower than this value (experimental).")
        for dataset in indata.ttimes:
            self.datasets.append(dataset)
            # sort everything (speeds up eikonal calculations)
            #sort_idx = np.lexsort((indata.sources[dataset][:,0],
            #                       indata.sources[dataset][:,1]))
            sort_idx = np.arange(len(indata.sources[dataset]))
            # sources and receivers are normally not a list of unique sources and receivers
            # it results from the input list which contains source receiver traveltime for every measurement
            self.sources[dataset] = indata.sources[dataset][sort_idx]
            self.receivers[dataset] = indata.receivers[dataset][sort_idx]
            self.data[dataset] = indata.ttimes[dataset][sort_idx]
            self.data_std[dataset] = indata.datastd[dataset][sort_idx]
            if np.shape(self.data_std[dataset]) == ():
                raise Exception("standard deviation has to be a vector of the same length as the input data")
            if len(self.sources[dataset]) != len(self.receivers[dataset]) != len(self.data[dataset]):
                raise Exception("Problem with the input data format!")
            self.model_predictions[dataset] = np.zeros(np.shape(self.data[dataset]))

            if self.outlier_model:
                self.foutlier[dataset] = 0.1
                self.outlierwidth[dataset] = [0.,0.]
                path_dists = np.sqrt(np.sum((self.sources[dataset]-self.receivers[dataset])**2,axis=1))
                if self.data_units == 'velocity':
                    residuals = path_dists/self.data[dataset] - np.mean(path_dists/self.data[dataset])
                else:
                    residuals = self.data[dataset] - path_dists/np.mean(path_dists/self.data[dataset])
                #print("Warning: modified outlierwidth!")
                self.widthoutlier = np.max([self.widthoutlier,(np.max(residuals)-np.min(residuals))])
                if self.chain_no == 1:
                    print("    Outlier width W is set to",np.around(self.widthoutlier,3),"(Outlier model of Tilmann et al. 2020)")
            else:
                self.foutlier = None
            self.idxselect[dataset] = np.arange(len(self.data[dataset]),dtype=int)
        if self.outlier_model:
            with open(self.logfile,"a") as f:
                f.write("width outliers: %.1f\n" %(self.widthoutlier))
         
            
        if 'relative' in self.error_type:
            self.std_intercept = {}
            self.std_scaling_factor = {}
            for dataset in self.data: 
                self.std_intercept[dataset] = np.mean([self.max_datastd,self.min_datastd])
                self.std_scaling_factor[dataset] = 0. #(0.5*self.max_datastd-self.std_intercept)/np.max(self.data)
                self.data_std[dataset] = np.ones(len(self.data[dataset])) * self.std_intercept[dataset]
        elif 'absolute' in self.error_type:
            for dataset in self.data:
                self.data_std[dataset] = np.ones(len(self.data[dataset])) * np.mean([self.max_datastd,self.min_datastd])#*self.data/np.mean(self.data)
        if 'stationerrors' in self.error_type:
            self.stations = {}
            self.stationerrors_index = {}
            self.stationerrors = {}
            for dataset in self.data:
                stations = np.vstack((self.sources[dataset],self.receivers[dataset]))
                self.stations[dataset],stats_uidx = np.unique(stations,return_inverse=True,axis=0)
                self.stationerrors_index[dataset] = np.column_stack(np.split(stats_uidx,2))
                self.stationerrors[dataset] = np.zeros(len(self.stations[dataset]))
            
        # make velocity updates more likely (in Bodin every even step is a vel update)
        self.actions = ["birth","death","move","velocity_update","velocity_update"]

        correct_bias = False
        if correct_bias:
            self.actions += ["bias_correction",]
            print("trying to correct for a source distribution bias")
        self.bootstrapping = False
        if self.bootstrapping:
            print("Warning: Experimentally using bootstrapping to randomly "+
                  "select a subset of xx% of the data at each iteration")
        #print("Warning: adding parameter swap!")
        #self.actions += ["parameter_swap"]
        self.accepted_steps = {}
        self.rejected_steps = {}
        self.acceptance_rate = {}
        for action in np.unique(self.actions):
            self.accepted_steps[action] = 0
            self.rejected_steps[action] = 1
            self.acceptance_rate[action] = np.zeros(100)
            # gives the CURRENT values, relative to the last 1000 steps:

        # hierarchical steps
        if self.error_type != 'fixed':
            self.actions += ["error_update"]
              
        self.total_steps = 0  
        
        # if user wants to visualize progress:
        self.visualize = params.visualize
        self.visualize_step = params.visualize_step
        if self.visualize:
            self.visualize_progress(initialize=True)
            
        self.model_smoothing = False
        if self.model_smoothing:
            print("model smoothing is set to True")

        self.collection_points = [[None,None,-1e99]]
        self.collection_models = []
        self.collection_no_points = []
        self.collection_datastd = {}
        self.collection_psi2 = []
        self.collection_psi2amp = []
        for dataset in self.datasets:
            self.collection_datastd[dataset] = []
        self.collection_loglikelihood = []
        self.collection_residuals = []
        self.collection_temperatures = []
        
        if self.interpolation_type=='wavelets':
            self.actions.remove("move")
        elif self.interpolation_type=='nocells':
            self.actions.remove("move")
            self.actions.remove("birth")
            self.actions.remove("death")
            
            #self.actions.remove("birth")
            #self.actions.remove("death")
            
        # include 2 psi anisotropy
        if self.anisotropic:
            self.accepted_steps['anisotropy_update_direction'] = 0
            self.accepted_steps['anisotropy_update_amplitude'] = 0
            self.rejected_steps['anisotropy_update_direction'] = 1
            self.rejected_steps['anisotropy_update_amplitude'] = 1
            self.acceptance_rate['anisotropy_update_direction'] = np.zeros(100)
            self.acceptance_rate['anisotropy_update_amplitude'] = np.zeros(100)
            self.accepted_steps['anisotropy_birth'] = 0
            self.accepted_steps['anisotropy_death'] = 0
            self.rejected_steps['anisotropy_birth'] = 1
            self.rejected_steps['anisotropy_death'] = 1
            self.acceptance_rate['anisotropy_birth'] = np.zeros(100)
            self.acceptance_rate['anisotropy_death'] = np.zeros(100)

        stats = []
        for dataset in self.datasets:
            stats.append(np.vstack((self.sources[dataset],self.receivers[dataset])))
        self.stations = np.unique(np.vstack(stats),axis=0)
        self.src_rcv_idx = {}
        for dataset in self.datasets:
            self.src_rcv_idx[dataset] = np.zeros((len(self.data[dataset]),2),dtype=int)
            for dataidx in range(len(self.data[dataset])):
                idx1 = np.where(np.sum(
                    self.stations==self.sources[dataset][dataidx],axis=1)==2)[0][0]
                idx2 = np.where(np.sum(
                    self.stations==self.receivers[dataset][dataidx],axis=1)==2)[0][0]
                self.src_rcv_idx[dataset][dataidx] = np.array([idx1,idx2])
      
        self.azimuths = {}
        self.bias = {}
        self.prop_bias = None
        self.biasmod_intcpt = 0.
        self.biasmod_scale = 1.
        self.sourcedist = np.ones(180)*0.5 # azimuths between 0 and 180 deg
        for dataset in self.datasets:
            self.bias[dataset] = np.zeros(len(self.data[dataset]))
            self.azimuths[dataset] = np.arctan2(self.sources[dataset][:,1]-self.receivers[dataset][:,1],
                                                self.sources[dataset][:,0]-self.receivers[dataset][:,0])
            self.azimuths[dataset][self.azimuths[dataset]<0.] += np.pi
            #self.azimuths[dataset][self.azimuths[dataset]>np.pi/2.] = np.abs(np.pi-self.azimuths[dataset][self.azimuths[dataset]>np.pi/2.])

        
    def initialize_model(self):
        
        if np.shape(self.init_vel_points)==():
            if self.init_vel_points =='random':
                init_vels = np.random.uniform(self.velmax,self.velmin,
                                              size=self.init_no_points)
                #meanvel = np.mean([self.velmax,self.velmin])
                #init_vels = np.random.uniform(meanvel-0.05,meanvel+0.05,
                #                              size=self.init_no_points)
            else:
                init_vels = np.ones(self.init_no_points)*self.init_vel_points
        elif len(self.init_vel_points)==self.init_no_points:
            init_vels = self.init_vel_points
        else:
            print("    Chain %d: ERROR: initial velocity parameter has not the right format." %self.chain_no)
         
        if self.interpolation_type=='nearest_neighbor':
            
            if self.kernel_type == 'rays':
                data_azimuths = self.get_binned_azimuths()
                data_idx = self.get_dataidx_in_gridcells()
            else:
                data_azimuths = data_idx = None

            if self.wavelength_dependent_gridspacing:
                gridsize = self.wavelength/4.
            else:
                gridsize = None
            self.para = rjtransdim2d_parameterization.voronoi_cells(
                self.gridpoints,self.shape,self.velmin,self.velmax,
                self.init_no_points,
                psi2ampmin=self.aniso_ampmin,psi2ampmax=self.aniso_ampmax,
                anisotropic=self.anisotropic,
                data_azimuths=data_azimuths,data_idx=data_idx,
                min_azimuthal_coverage=None,min_data=None,
                gridspacing_staggered=gridsize)
            #print("Warning: setting minimum coverage!")
            self.para.vs = init_vels[:len(self.para.vs)]
            #print("Warning! Fixing velocities at station locations!")
            #tree = KDTree(self.gridpoints)
            #nndist,nnidx = tree.query(self.stations,k=4)
            #self.para.stationgridpoints = np.unique(nnidx)

        elif self.interpolation_type=='weighted_nearest_neighbor':
            
            self.para = rjtransdim2d_parameterization.gravity_cells(
                self.gridpoints,self.shape,self.init_no_points)
            self.para.vs = init_vels
            
        elif self.interpolation_type=='linear':
            
            self.para = rjtransdim2d_parameterization.linear_interpolation(
                self.gridpoints,self.shape,self.velmin,self.velmax,
                self.init_no_points,anisotropic=self.anisotropic,
                psi2ampmin=self.aniso_ampmin,psi2ampmax=self.aniso_ampmax)
            self.para.vs = init_vels
            
        elif self.interpolation_type=='wavelets':
            
            self.para = rjtransdim2d_parameterization.wavelets_anisotropic(
                self.gridpoints,self.shape,self.velmin,self.velmax,
                init_no_coeffs=self.init_no_points,
                psi2ampmin=self.aniso_ampmin,psi2ampmax=self.aniso_ampmax,
                anisotropic=self.anisotropic)
            self.maxlevel = self.para.levels-1 # if there are 5 levels, maxlevel=4 levels=[0,1,2,3,4]
            self.update_steps = np.around([(self.nburnin/2.)/2**i for i in range(self.maxlevel)]).astype(int)[::-1]
            #self.maxlevel = 5
            
        elif self.interpolation_type=='nocells':
            self.para = rjtransdim2d_parameterization.nocells(
                self.gridpoints, self.shape, self.velmin, self.velmax,
                 self.init_no_points,psi2ampmin=self.aniso_ampmin,
                 psi2ampmax=self.aniso_ampmax,anisotropic=self.anisotropic,)

        elif self.interpolation_type=='blocks':
            self.para = rjtransdim2d_parameterization.blocks(
                self.gridpoints, self.shape, self.velmin, self.velmax,
                 self.init_no_points,psi2ampmin=self.aniso_ampmin,
                 psi2ampmax=self.aniso_ampmax,anisotropic=self.anisotropic,)
            
        elif self.interpolation_type=='dist_weighted_means':

            self.para = rjtransdim2d_parameterization.dist_weighted_means(
                self.gridpoints,self.shape,self.velmin,self.velmax,
                self.init_no_points,
                psi2ampmin=self.aniso_ampmin,psi2ampmax=self.aniso_ampmax,
                anisotropic=self.anisotropic,
                gridspacing_staggered=None,metric='euclidean')
                       
        else:
            raise Exception(f"Interpolation type {self.interpolation_type} unknown.")
            
        if self.anisotropic:
            self.AA1 = {}
            self.AA2 = {}
            for dataset in self.datasets:
                self.AA1[dataset] = deepcopy(self.A[dataset])
                self.AA1[dataset].data *= np.cos(2*self.PHI[dataset].data)
                self.AA2[dataset] = deepcopy(self.A[dataset])
                self.AA2[dataset].data *= np.sin(2*self.PHI[dataset].data)
            self.model_slowness,self.psi2amp,self.psi2 = self.para.get_model(anisotropic=True)
        else:
            self.model_slowness = self.para.get_model()
       
        # initialize average model        
        self.average_model = 1./self.model_slowness.copy()
        self.modelsumsquared = self.average_model.copy()**2
        self.average_model_counter = 1
        
        if self.anisotropic:
            self.average_anisotropy = np.column_stack((np.cos(2*self.psi2)*self.psi2amp,
                                                       np.sin(2*self.psi2)*self.psi2amp))
            self.anisotropy_sumsquared =  self.average_anisotropy**2
            self.psi2amp_sumsquared = self.psi2amp**2
            self.average_model_counter_aniso = 1
            
            
    def resume(self,prior,params):
               
        #self.selfcheck=False
        if self.selfcheck:
            print("WARNING: selfcheck turned on")
            
        #if self.interpolation_type=='wavelets':
        #    print("Warning! Setting new maxlevel for wavelet decomposition")
            #self.para.update_maxlevel()
            #self.para.update_maxlevel()
        #print("setting temperature to 1")
        #self.temperature=1
        
        self.target_iterations = self.update_parameter(self.target_iterations,params.target_iterations,
                                                       "parameter(target iterations)")
        self.nburnin = self.update_parameter(self.nburnin, params.nburnin,
                                             "parameter(nburnin)")
        
        with open(self.logfile,"a") as f:
                f.write(f"\nResuming at {self.total_steps} iterations. Calculating additional {self.target_iterations-self.total_steps} iterations.\n")

        self.min_datastd = self.update_parameter(self.min_datastd, prior.min_datastd,
                                                 "prior(min data std)")
        self.max_datastd = self.update_parameter(self.max_datastd, prior.max_datastd,
                                                 "prior(max data std)")
        for dataset in self.datasets:
            if np.mean(self.data_std[dataset])<self.min_datastd:
                self.data_std[dataset][:] = self.min_datastd
            elif np.mean(self.data_std[dataset])>self.max_datastd:
                self.data_std[dataset][:] = self.max_datastd
        self.nmin_points = self.update_parameter(self.nmin_points, prior.nmin_points,
                                                 "prior(min no points)")
        self.nmax_points = self.update_parameter(self.nmax_points, prior.nmax_points,
                                                 "prior(max no points)")
    
        self.propsigmastd = self.update_parameter(self.propsigmastd, params.propsigmastd,
                                                  "parameter(proposal std data std)")
        if self.propvelstd != 'adaptive' or params.propvelstd != "adaptive":
            self.propvelstd = self.update_parameter(self.propvelstd, params.propvelstd,
                                                    "parameter(proposal std type)")
            self.propvelstd_dimchange = self.update_parameter(self.propvelstd_dimchange, params.propvelstd_dimchange,
                                                              "parameter(proposal std birth/death)")
            self.propvelstd_pointupdate = self.update_parameter(self.propvelstd_pointupdate, params.propvelstd_pointupdate,
                                                                "parameter(proposal std velocity update)")
            self.propmovestd = self.update_parameter(self.propmovestd, params.propmovestd,
                                                     "parameter(proposal std move)")
            
        #self.temperature = self.update_parameter(self.temperature, params.temperature,
        #                                         "parameter(temperature)")

        self.update_paths_interval = self.update_parameter(self.update_paths_interval, params.update_paths_interval,
                                                           "parameter(update path interval)")

        self.collect_step = self.update_parameter(self.collect_step, params.collect_step,
                                                  "parameter(model collect interval)")
        self.print_stats_step = self.update_parameter(self.print_stats_step, params.print_stats_step,
                                                      "parameter(print stats interval)")

       # include 2 psi anisotropy
        self.anisotropic = self.update_parameter(self.anisotropic, params.anisotropic,
                                                 "parameter(anisotropic search)")
        if self.anisotropic:
            self.propstd_anisoamp = self.update_parameter(self.propstd_anisoamp, params.propstd_anisoamp,
                                                          "parameter(proposal std anisotropic amplitude)")
            self.propstd_anisodir = self.update_parameter(self.propstd_anisodir, params.propstd_anisodir,
                                                          "parameter(proposal std anisotropic direction)")
            self.aniso_ampmin = self.update_parameter(self.aniso_ampmin, prior.aniso_ampmin,
                                                      "prior(min aniso amplitude)")
            if self.aniso_ampmin != 0.:
                print("Warning: minimum anisotropic amplitude is set to zero")
                self.aniso_ampmin = 0.
                # could otherwise cause problems if I start with a purely isotropic search
            self.aniso_ampmax = self.update_parameter(self.aniso_ampmax, prior.aniso_ampmax,
                                                      "parameter(prior(max aniso amplitude)")

        self.update_chain()
        
        
    def update_parameter(self,oldparam,newparam,varname):
        if newparam is None:
            return oldparam
        if oldparam != newparam:
            with open(self.logfile,"a") as f:
                f.write(f"    Updating {varname}: {oldparam} -> {newparam}\n")
        return newparam  
         
    
    def propose_jump(self,action='random'):
                                                          
        self.total_steps += 1
        
        if self.interpolation_type=='wavelets':
            if self.total_steps < self.update_steps[-1]:
                if (self.total_steps > self.update_steps[self.para.maxlevel] and 
                    self.para.maxlevel<self.maxlevel):
                    print("Chain %d iteration %d" %(self.chain_no,self.total_steps))
                    self.para.update_maxlevel()

        # start adding anisotropy only after a certain number of iterations
        aniso_start = 1#int(self.nburnin/2)
        if self.anisotropic and self.total_steps>=aniso_start:
            anisotropic = True
            if self.total_steps==aniso_start:
                with open(self.logfile,"a") as f:
                    f.write("Starting to adapt anisotropic parameters.\n")
                # should actually all be zero
                self.average_anisotropy /= self.average_model_counter_aniso
                self.psi2amp_sumsquared /= self.average_model_counter_aniso
                self.anisotropy_sumsquared /= self.average_model_counter_aniso
                self.average_model_counter_aniso = 1
                self.actions += ["anisotropy_update","anisotropy_update"]
                if self.interpolation_type!='wavelets' and self.interpolation_type!='nocells':
                    self.actions += ["anisotropy",]
        else:
            anisotropic = False
          
        if False:#self.total_steps % 100 == 0:
            self.idxselect = {}
            for dataset in self.data:
                self.idxselect[dataset] = np.random.choice(
                    np.arange(len(self.data[dataset])),
                    size=int(0.5*len(self.data[dataset])),
                    replace=False)
            self.loglikelihood_current, self.residual = self.loglikelihood(
                self.data, self.model_predictions, self.data_std, self.foutlier)
            
                
        # if the proposal standard deviations are not fixed
        if self.propvelstd == 'adaptive':
            if self.total_steps % 100 == 0:        
                self.update_proposal_stds() 
                  
        if action=='random':
            # 50% chance of updating the parameters of an existing point
            action = np.random.choice(self.actions)
        self.action = action

        # if self.total_steps<50000:
        #     self.min_datastd = 0.1
        #     if action in ['birth','death']:
        #         action = 'velocity_update'
        # else:
        #     self.min_datastd = 0.01
            
        #if action in ['death']:
        #    action = 'move'

        #if self.outlier_model:
        #    if self.total_steps < 2000 or self.total_steps%1000==0:
        #        self.update_widthoutlier(self.data,self.model_predictions,
        #                                 self.data_std,self.foutlier)
                
        # if self.interpolation_type == 'wavelets':
        #     if self.total_steps == int(np.product(self.para.tree) / 
        #                                np.product(self.para.full_tree) *
        #                                self.nburnin ):
        #         self.para.update_maxlevel()

        #print(action)
        
        # experimental: try to fit potential bias in the measured data
        if action == 'bias_correction':
            
            
            # for dataset in self.datasets:
            #     velbias = -0.0002*self.path_dists[dataset] + 0.1
            #     vels = self.path_dists[dataset]/self.data[dataset]
            #     self.bias[dataset] = self.data[dataset]-self.path_dists[dataset]/(vels+velbias)
            
            # loglikelihood_prop,_ = self.loglikelihood(
            #     self.data,self.model_predictions,self.data_std,self.foutlier)
            # self.store_model(action,"accept")
            # return 1,self.loglikelihood_current
            
            """
            Bias dependent on the propagation azimuth
            """
            # prop_biasmod_intcpt = self.biasmod_intcpt
            # prop_biasmod_scale  = self.biasmod_scale
            
            # perturb_action = np.random.choice(["scaling_factor","intercept"])
            # if perturb_action == 'scaling_factor':
            #     prop_biasmod_scale += np.random.normal(0,0.001)
            # if perturb_action == 'intercept':
            #     prop_biasmod_intcpt += np.random.normal(0,0.001)
            
            # self.prop_bias = {}
            # for dataset in self.datasets:
            #     velbias = prop_biasmod_scale*self.azimuths[dataset] + prop_biasmod_intcpt
            #     #velbias = prop_biasmod_scale*self.path_dists[dataset] + prop_biasmod_intcpt
            #     vels = self.path_dists[dataset]/self.data[dataset]
            #     if (np.abs(velbias)>0.3).any():
            #         self.prop_bias = None
            #         self.store_model(action,"reject")
            #         return 0,self.loglikelihood_current
            #     self.prop_bias[dataset] = self.data[dataset]-self.path_dists[dataset]/(vels+velbias)
            
            """
            Bias that is related to the non-homogenous noise source distribution
            
            This bias is often very small and only becomes important at long
            periods and small interstation distances (about 1-2 wavelength
            interstation distance). E.g. synthetic tests in Kaestle et al. 2016
            and Weaver et al. 2009 indicate around 0.5% velocity error at two
            interstation distances and up to 2.5% at one interstation distances
            (in practice probably lower). This seems to be too low compared to
            the data standard deviation to be properly resolvable.
            """
            def vonmises(x,mu,kappa):
                return  (np.exp(kappa*np.cos(x-mu)) / (2*np.pi*i0(kappa)))
            #def vonmises_d1(x,mu,kappa):
            #    # first derivative
            #    return (np.exp(kappa*np.cos(x-mu))*-1*kappa*np.sin(x-mu)) / (2*np.pi*i0(kappa))
            def vonmises_d2(x,mu,kappa):
                # second derivative
                return (np.exp(kappa*np.cos(x-mu))*kappa*(kappa*np.sin(x-mu)**2-np.cos(x-mu))) / (2*np.pi*i0(kappa))
            
            perturb_action = np.random.choice(["sourcedist","biasscale"])
            
            prop_sourcedist = self.sourcedist.copy()
            # the source distribution model is only defined between 0 and
            # 180 degrees. Since we work with symmetrized spectra, we
            # can only resolve the source distribution on a half circle
            azi = np.linspace(0,np.pi,180)
            
            # the vonmises distribution is calculated on the full circle
            # but then compressed to 180 degrees. This only changes kappa
            # which is random anyways
            dsource = vonmises(np.linspace(0,2*np.pi,len(azi)),
                               np.random.uniform(0,2*np.pi),
                               np.random.uniform(0,20))
            dsource /= np.max(dsource)
            dsource *= np.random.uniform(0,0.1)
            prop_sourcedist += dsource
            prop_sourcedist /= np.max(prop_sourcedist)

            # idx = np.random.randint(0,len(self.soucebias_params))
            # params = self.sourcebias_params.copy()
            # params[idx] += np.random.normal()
            # if (params<0).any():
            #     self.prop_bias = None
            #     self.store_model(action,"reject")
            #     return 0,self.loglikelihood_current
            
            # azi = np.linspace(0,2*np.pi,360)
            # A0,A1,A2,A3,M1,M2,M3,K1,K2,K3 = params
            # sourceintensity = (A0 + 
            #                     A1*vonmises(azi,M1,K1) + 
            #                     A2*vonmises(azi,M2,K2) +
            #                     A3*vonmises(azi,M3,K3) )
            # sourceintensity /= np.max(sourceintensity)
            
                      
            B2 = np.gradient(np.gradient(prop_sourcedist,edge_order=2)/
                             np.gradient(azi))/np.gradient(azi)
            func = interp1d(azi,B2/prop_sourcedist,kind='nearest')
            
            # bias in terms of traveltime
            self.prop_bias = {}
            for dataset in self.datasets:
                self.prop_bias[dataset] = ( 
                    func(self.azimuths[dataset])/np.pi /
                    (2*self.meanslow*self.path_dists[dataset]*(2*np.pi/self.period)**2))
                # np.pi factor may be wrong?
            
            loglikelihood_prop,_ = self.loglikelihood(
                self.data,self.model_predictions,self.data_std,self.foutlier)
            
            log_acceptance_prob = loglikelihood_prop - self.loglikelihood_current
            
            if np.log(np.random.rand(1)) <= log_acceptance_prob:
                         
                #accepted
                #self.biasmod_intcpt = prop_biasmod_intcpt
                self.sourcedist = prop_sourcedist
                #self.biasmod_scale = prop_biasmod_scale 
                self.loglikelihood_current = loglikelihood_prop
                self.bias = deepcopy(self.prop_bias)
                         
                self.prop_bias = None
                self.store_model(action,"accept")
                return 1,self.loglikelihood_current
                
            else:
                
                self.prop_bias = None
                self.store_model(action,"reject")
                return 0,self.loglikelihood_current
            
        
        
        if action == 'error_update':
            
            loglikelihood_current,_ = self.loglikelihood(
                self.data,self.model_predictions,self.data_std,self.foutlier,
                error_update=True)
            
            dataset = np.random.choice(self.datasets)
            
            update_foutlier = False
            
            if self.outlier_model and np.random.rand(1)>0.66:
                
                update_foutlier = True
                prop_foutlier = deepcopy(self.foutlier)
                prop_foutlier[dataset] = self.foutlier[dataset] + np.random.normal(loc=0.0,scale=0.01)
                
                if prop_foutlier[dataset] < 0. or prop_foutlier[dataset] > 0.8:
                    prop_foutlier = None
                    self.store_model(action,"reject")
                    return 0,self.loglikelihood_current
                
                loglikelihood_prop,_ = self.loglikelihood(self.data,self.model_predictions,
                                                          self.data_std,prop_foutlier,
                                                          error_update=True)
                    
            else:
                
                prop_data_std = deepcopy(self.data_std)
                
                #self.current_data_std = self.data_std
                if 'absolute' in self.error_type:
                    # I could also skip this since we know that the maximum
                    # likelihood solution is always
                    # data_std = np.sqrt(np.mean(residuals**2))
                    # see for example Bishop Pattern recognition and machine learning equation 1.56
                    
                    prop_data_std[dataset] = self.data_std[dataset] + np.random.normal(loc=0.0,scale=self.propsigmastd)# * self.data/np.mean(self.data)
                    
                # linear scaling with the path distance data_std = std_scaling_factor * path_dists + std_intercept
                elif 'relative' in self.error_type:
                    
                    perturb_action = np.random.choice(["scaling_factor","intercept"])
                    if perturb_action == "scaling_factor":
                        # if the scaling factor is perturbed, the proposal std
                        # is scaled by the mean path distance
                        prop_std_scaling_factor = self.std_scaling_factor[dataset] + np.random.normal(loc=0.0,scale=self.propsigmastd/np.mean(self.path_dists[dataset]))
                        prop_data_std[dataset] = prop_std_scaling_factor * self.path_dists[dataset] + self.std_intercept[dataset]
                        if prop_std_scaling_factor <= 0.:
                            prop_data_std = None
                            self.store_model(action,"reject")
                            return 0,self.loglikelihood_current
                    else:
                        prop_std_intercept = self.std_intercept[dataset] + np.random.normal(loc=0.0,scale=self.propsigmastd)
                        prop_data_std[dataset] = self.std_scaling_factor[dataset] * self.path_dists[dataset] + prop_std_intercept
                    
                else:
                    raise Exception("error type not understood:",self.error_type)
                
                # experimental: station-wise additional standard deviations 
                if 'stationerrors' in self.error_type:
                    
                    prop_stationerrors = None
                    if self.total_steps>self.nburnin/4.:
                        # randomly choose a station for which to update the error
                        prop_stationerrors = self.stationerrors[dataset].copy()
                        updateidx = np.random.randint(low=0,high=len(self.stations[dataset]))
                        prop_stationerrors[updateidx] += np.random.normal(loc=0.0,scale=self.propsigmastd)
                        if prop_stationerrors[updateidx] < 0.:
                            prop_stationerrors[updateidx] = self.stationerrors[dataset][updateidx]
                        prop_data_std[dataset] += np.sum(
                            prop_stationerrors[self.stationerrors_index[dataset]],axis=1)
                         

                if ((np.mean(prop_data_std[dataset]) <= self.min_datastd).any() or
                    (np.mean(prop_data_std[dataset]) >= self.max_datastd).any() or
                    (prop_data_std[dataset] <= 0.0).any()):
                    prop_data_std = None
                    self.store_model(action,"reject")
                    return 0,self.loglikelihood_current
                
                loglikelihood_prop,_ = self.loglikelihood(
                    self.data,self.model_predictions,prop_data_std,self.foutlier,
                    error_update=True)
                
            log_acceptance_prob = loglikelihood_prop - loglikelihood_current
            # hyperparameter update, don't use the self.check_acceptance function
            if np.log(np.random.rand(1)) <= log_acceptance_prob:
                                
                #accepted
                if update_foutlier:
                    self.foutlier = prop_foutlier
                else:
                    self.data_std = prop_data_std
                    if 'relative' in self.error_type:
                        if perturb_action == 'scaling_factor':
                            self.std_scaling_factor[dataset] = prop_std_scaling_factor
                        elif perturb_action == 'intercept':
                            self.std_intercept[dataset] = prop_std_intercept
                    if ('stationerrors' in self.error_type) and (prop_stationerrors is not None):
                        self.stationerrors[dataset] = prop_stationerrors
                         
                self.loglikelihood_current,_ = self.loglikelihood(
                    self.data,self.model_predictions,self.data_std,self.foutlier)   
                         
                self.store_model(action,"accept")
                return 1,self.loglikelihood_current
                
            else:
                
                prop_data_std = None
                prop_foutlier = None
                self.store_model(action,"reject")
                return 0,self.loglikelihood_current
                
            
        
        if action == 'velocity_update':
                      
            idx_point = np.random.randint(0,len(self.para.points))
                        
            for second_try in [False,True]: #delayed rejection
            
                # if first move was rejected, try a second move with smaller std
                # dx and propdist_std is needed to calculate the correct acceptance
                # probability for delayed proposals
                if second_try:
                    self.propdist_std = self.propvelstd_pointupdate * 0.25
                else:
                    self.propdist_std = self.propvelstd_pointupdate
                      
                self.dx = self.propdist_std * np.random.normal(loc=0.0,scale=1.0)
                
                check_update = self.para.vs_update(idx_point,self.dx)
                
                if not check_update:
                    if self.delayed_rejection and not second_try:
                        self.dx1 = self.dx
                        self.propdist_std1 = self.propdist_std
                        self.para.reject_mod()
                        continue
                    else:
                        self.reject_model(action)
                        return 0,self.loglikelihood_current
                
                # calculate traveltimes, loglikelihood and residual in new model
                self.update_model_predictions(action)
                    
                # check acceptance according to Metropolis-Hastings criterion
                if self.check_acceptance(delayed=second_try):
                    self.accept_model(action)
                    return 1,self.loglikelihood_current
                    
                else:                   
                    if self.delayed_rejection and not(second_try):
                        self.dx1 = self.dx
                        self.propdist_std1 = self.propdist_std
                        self.para.reject_mod()
                        continue
                    else:
                        self.reject_model(action)                
                        return 0,self.loglikelihood_current     
        

        if action == 'anisotropy_update':
               
            anisochange = np.random.choice(["amplitude","direction"])
            
            if self.interpolation_type in ['wavelets','nocells']:
                idx_point = np.random.choice(np.arange(len(self.para.points)))
            else:
                ind_pnts = np.where(self.para.psi2amp > 0.)[0]
                if len(ind_pnts)==0:
                    self.reject_model(action+'_'+anisochange)
                    return 0,self.loglikelihood_current
                idx_point = np.random.choice(ind_pnts)
                                                    
            for second_try in [False,True]: #delayed rejection
                        
                if anisochange == "amplitude": # change fast axis amplitude
                    # if first move was rejected, try a second move with smaller std
                    if second_try:
                        self.propdist_std = self.propstd_anisoamp * 0.25
                    else:
                        self.propdist_std = self.propstd_anisoamp
                        
                    self.dx = self.propdist_std * np.random.normal(loc=0.0,scale=1.0)
                    check_update = self.para.psi2amp_update(idx_point, self.dx)

                else: # change fast axis direction
                    # if first move was rejected, try a second move with smaller std
                    if second_try:
                        self.propdist_std = self.propstd_anisodir * 0.25
                    else:
                        self.propdist_std = self.propstd_anisodir
                    
                    self.dx = self.propdist_std * np.random.normal(loc=0.0,scale=1.0)
                    check_update = self.para.psi2_update(idx_point, self.dx)
                    
                if not check_update:
                    if self.delayed_rejection and not second_try:
                        self.dx1 = self.dx
                        self.propdist_std1 = self.propdist_std
                        self.para.reject_mod()
                        continue
                    else:
                        self.reject_model(action+'_'+anisochange)
                        return 0,self.loglikelihood_current
                    
                # calculate traveltimes, loglikelihood and residual in new model
                self.update_model_predictions(action+'_'+anisochange)
                       
                # check acceptance according to Metropolis-Hastings criterion
                if self.check_acceptance(delayed=second_try):
                    self.accept_model(action+"_"+anisochange) 
                    return 1,self.loglikelihood_current
                else: # rejected
                    if self.delayed_rejection and not(second_try):
                        self.dx1 = self.dx
                        self.propdist_std1 = self.propdist_std
                        self.para.reject_mod()
                        continue
                    else:    
                        self.reject_model(action+"_"+anisochange)                
                        return 0,self.loglikelihood_current  


        if action == 'anisotropy': # for anisotropy-birth / anisotropy-death
            
            anisochange = np.random.choice(['birth','death'])
            
            if anisochange == 'birth':
                ind_pnts = np.where(self.para.psi2amp == 0.)[0]
            else:
                ind_pnts = np.where(self.para.psi2amp != 0.)[0]
                
            if len(ind_pnts) > 0:
                idx_point = np.random.choice(ind_pnts)
            else:
                self.reject_model(action+"_"+anisochange)
                return 0,self.loglikelihood_current

       
            if anisochange == 'birth':
                check_update = self.para.psi2amp_update(
                    idx_point,np.random.uniform(self.aniso_ampmin,
                                                self.aniso_ampmax))
                self.para.psi2_update(idx_point,np.random.uniform(0,2*np.pi))
                 
            else:
                check_update = self.para.psi2amp_update(
                    idx_point,-self.para.psi2amp[idx_point])
                #self.para.psi2_update(idx_point,0.)
                
            if not check_update:    
                self.reject_model(action+'_'+anisochange)
                return 0,self.loglikelihood_current
            
            # calculate traveltimes, loglikelihood and residual in updated model
            self.update_model_predictions(action+"_"+anisochange)

            # check acceptance according to Metropolis-Hastings criterion
            if self.check_acceptance():
                self.accept_model(action+"_"+anisochange)
                return 1,self.loglikelihood_current
            else: # rejected
                self.reject_model(action+"_"+anisochange)                
                return 0,self.loglikelihood_current 
            
        
        
        if action=='birth': # birth always creates isotropic cells
                        
            # check that there are not more points than defined in the prior        
            if len(self.para.points)+1 > self.nmax_points:
                self.reject_model(action)
                return 0,self.loglikelihood_current
  
            # add a new point at a random location
            birth_check = self.para.add_point(anisotropic=self.anisotropic,
                                              birth_prop=self.propvelstd_dimchange)

            # reject model if the new parameter is outside the prior range or
            # if there is no free node in the tree (wavelet option)
            if not birth_check:
                self.reject_model(action)
                return 0,self.loglikelihood_current
            # get traveltimes, likelihoods and residuals in new model
            self.update_model_predictions(action)
            
            # check acceptance according to Metropolis-Hastings criterion
            if self.check_acceptance():
                # model accepted
                self.accept_model(action)
                return 1,self.loglikelihood_current
                
            else:
                self.reject_model(action)
                return 0,self.loglikelihood_current



        if action == 'death':
            
            # check that there are not fewer points than defined in the prior
            if len(self.para.points)-1 < self.nmin_points:
                self.reject_model(action)
                return 0,self.loglikelihood_current
         
            # remove a randomly chosen point
            remove_check = self.para.remove_point(anisotropic=self.anisotropic) 
            
            # check whether it was possible to remove a point
            if not remove_check:
                self.reject_model(action)
                return 0,self.loglikelihood_current
            
            # calculate the traveltimes in the new model and calculate new
            # likelihood and residuals
            self.update_model_predictions(action)

            # check the acceptance according to the metropolis hastings criterion
            if self.check_acceptance():
                # model accepted
                self.accept_model(action)
                return 1,self.loglikelihood_current
            else:
                self.reject_model(action)
                return 0,self.loglikelihood_current
                          

        if action == 'move':
            
            idx_move = None
     
            for second_try in [False,True]:
            
                # if first move was rejected, try a second move with smaller std
                if second_try:
                    self.propdist_std = self.propmovestd * 0.25
                else:
                    self.propdist_std = self.propmovestd
                    
                movepoint_old, movepoint_new = self.para.move_point(self.propdist_std,
                                                                    index=idx_move)
                idx_move = self.para.idx_mod
                self.dx = movepoint_new - movepoint_old
                
                # if the new position is outsite the coordinate limits
                if np.isnan(movepoint_new).any():
                    if self.delayed_rejection and not second_try:
                        self.dx1 = self.dx
                        self.propdist_std1 = self.propdist_std
                        self.para.reject_mod()
                        continue
                    else:
                        self.reject_model(action)
                        return 0,self.loglikelihood_current
                       
                # calculate the traveltimes in the new model
                self.update_model_predictions(action)
                
                # check the acceptance according to the metropolis hastings criterion
                if self.check_acceptance(delayed=second_try):
                    # model accepted
                    self.accept_model(action)
                    return 1,self.loglikelihood_current
                else:
                    if self.delayed_rejection and not(second_try):
                        self.dx1 = self.dx
                        self.propdist_std1 = self.propdist_std
                        self.para.reject_mod()
                        continue
                    else: 
                        self.reject_model(action)          
                        return 0,self.loglikelihood_current


        else:
            
            raise Exception(f"not a valid action: {action}")
            
    


    def update_model_predictions(self,action):

        self.valid_model = True
        
        # calculate the new model        
        if self.anisotropic:
            self.prop_model_slowness, self.prop_psi2amp, self.prop_psi2 = (
                self.para.update_model(fields=(self.model_slowness,
                                               self.psi2amp,
                                               self.psi2),
                                       anisotropic=True))
        else:
            self.prop_model_slowness = self.para.update_model(fields=
                                                self.model_slowness)
            
        # experimental: check gradient between voronoi cells
        check_grad = self.check_gradients()
        if not check_grad:
            self.valid_model = False
            return
        
        # invalid wavlet models result in nan in model vector
        if self.interpolation_type == 'wavelets':
            if np.isnan(self.prop_model_slowness).any():
                self.valid_model = False
                return
             
        # for debugging
        if self.selfcheck:
            test_slow_field = self.para.get_model()
            if (self.prop_model_slowness-test_slow_field > 1e-10).any():
                self.para.plot()
                raise Exception(action)
            if np.isnan(self.prop_model_slowness).any():
                raise Exception(f"nan in vel mod during {action}")
            if self.anisotropic:
                test_slow_field,test_psi2amp,test_psi2 = self.para.get_model(anisotropic=True)
                if ((self.prop_model_slowness-test_slow_field > 1e-10).any() or
                    (self.prop_psi2amp-test_psi2amp > 1e-10).any() or
                    (self.prop_psi2-test_psi2 > 1e-10).any() ):
                    self.para.plot()
                    raise Exception(action)

        # calculate the traveltimes in the new model
        if len(self.para.idx_mod_gpts)>0:
            self.calculate_ttimes(idx_modified=self.para.idx_mod_gpts,
                                  anisotropic=self.anisotropic)
        else:
            self.prop_model_predictions = deepcopy(self.model_predictions)
            
        
        if self.bootstrapping:
            idx_bootstrapping = {}
            for dataset in self.data:
                idx_bootstrapping[dataset] = np.random.choice(
                    np.arange(len(self.data[dataset]),dtype=int),
                    size=int(0.5*len(self.data[dataset])),replace=False)
            self.loglikelihood_current, self.residual = self.loglikelihood(
                self.data,self.model_predictions,self.data_std,self.foutlier,
                bootstrap_idx=idx_bootstrapping)
        else:
            idx_bootstrapping=None
            
        # this function is executed after model modifications (parameter update
        # birth, death, move), model predictions change, but not the hyperparameters.
        self.loglikelihood_prop, self.residual_prop = self.loglikelihood(
            self.data,self.prop_model_predictions,self.data_std,self.foutlier,
            bootstrap_idx=idx_bootstrapping)
            
        
    def check_acceptance(self,delayed=False):
        
        if not self.valid_model:
            return False
        
        llike_current = self.loglikelihood_current
        llike_proposed = self.loglikelihood_prop
        
        if delayed:
            log_a_m1_m2 = 1./self.temperature * (self.llike_prop_m1-llike_proposed)
            if self.log_a_m1_m0 >= 0. or log_a_m1_m2 >=0.:
                return False
            log_prior_prop_ratios = (
                # this is log(q1(m'|m'')/q1(m'|m)) from eq. 3.39 of the thesis of T.Bodin
                # where the q-terms are given in eq. 3.13 or 3.15.
                # notice that it's both q1, meaning that most terms cancel out, only exp terms remain
                (np.sum(self.dx1**2)-np.sum(self.dx**2))/(2*self.propdist_std1**2) +
                # this is {1-a1(m'|m'')}/{1-a1(m'|m)} from eq. 3.39 of the thesis of T.Bodin
                np.log((1-np.exp(log_a_m1_m2))/(1-np.exp(self.log_a_m1_m0))) )
        else:
            log_prior_prop_ratios = self.para.get_prior_proposal_ratio()
        
        log_acceptance_prob = (
            log_prior_prop_ratios + 1./self.temperature * 
            (llike_proposed - llike_current) )
            
        # store the value for the delayed rejection scheme
        self.log_a_m1_m0 = log_acceptance_prob
        self.llike_prop_m1 = llike_proposed
        
        return np.log(np.random.rand(1)) <= log_acceptance_prob
    
    
    def check_gradients(self):
        return True
        maxgrad = 0.2#self.velmax-self.velmin
        V = np.reshape(1./self.prop_model_slowness,self.shape)
        grady,gradx = np.gradient(V)
        if np.max(np.abs(gradx)) > maxgrad or np.max(np.abs(grady)) > maxgrad:
            return False
        else:
            return True

    # forward calculation using a matrix vector multiplication
    def calculate_ttimes(self,idx_modified=None,anisotropic=False):
            
        if idx_modified is not None: # is sorting necessary?
            idx_modified = np.sort(idx_modified)
           
        self.prop_model_predictions = {}
        
        for dataset in self.A:
            
            if not anisotropic:
                
                if idx_modified is None:
                    
                    if self.kernel_type == 'sens_kernels':
                        m0 = 1./self.meanslow
                        m = 1./self.prop_model_slowness
                        self.prop_model_predictions[dataset] = (
                            self.path_dists[dataset] * self.meanslow + (
                            self.A[dataset]*(m-m0)))

                    else:
                        self.prop_model_predictions[dataset] = (
                            self.A[dataset] * self.prop_model_slowness )
                        
                else:
                    
                    m0 = self.model_slowness[idx_modified]
                    m = self.prop_model_slowness[idx_modified]
                    if self.kernel_type == 'sens_kernels':
                        m0 = 1./m0
                        m = 1./m
                    self.prop_model_predictions[dataset] = (
                        self.model_predictions[dataset] + 
                        self.A[dataset][:,idx_modified] * (m-m0))
                    
            # anisotropic        
            else:

                m = 1./self.prop_model_slowness

                if idx_modified is None:
                                        
                    if self.kernel_type == 'sens_kernels':
                        m0 = 1./self.meanslow
                        nz = self.A[dataset].nonzero()
                        # since the matrix is csc (column sorted), we have to sort
                        # the nonzero indices according to their column (nz[1])
                        cidx = np.sort(nz[1])
                        dm = m[cidx] * (1. + self.prop_psi2amp[cidx]*
                                        np.cos(2*(self.PHI[dataset].data -
                                                  self.prop_psi2[cidx]))) - m0
                        dM = csc_matrix((dm,self.A[dataset].indices,
                                         self.A[dataset].indptr),
                                        shape=self.A[dataset].shape)
                        self.prop_model_predictions[dataset] = (
                            self.path_dists[dataset] * self.meanslow + 
                            np.asarray((self.A[dataset].multiply(dM)).sum(axis=1).T)[0])
                        
                    else:
                        nz = self.A[dataset].nonzero()
                        cidx = np.sort(nz[1])
                        m_aniso = m[cidx] * (1. + self.prop_psi2amp[cidx]*
                                             np.cos(2*(self.PHI[dataset].data - 
                                                       self.prop_psi2[cidx])))
                        M_aniso = csc_matrix((1./m_aniso,
                                              self.A[dataset].indices,
                                              self.A[dataset].indptr),
                                             shape=self.A[dataset].shape)
                        self.prop_model_predictions[dataset] = (
                            np.asarray((self.A[dataset].multiply(M_aniso)).sum(axis=1).T)[0])
                        
                elif True: # exact calculation
                    
                    m0 = 1./self.model_slowness
                    Asub = self.A[dataset][:,idx_modified]
                    phi_data = self.PHI[dataset][:,idx_modified].data
                    cidx = np.hstack(self.nonzero[dataset][idx_modified])
                    m_aniso = m[cidx] * (1. + self.prop_psi2amp[cidx] * 
                        np.cos(2*(phi_data - self.prop_psi2[cidx])))
                    m0_aniso = m0[cidx] * (1. + self.psi2amp[cidx] * 
                        np.cos(2*(phi_data - self.psi2[cidx])))
                    if self.kernel_type != 'sens_kernels':
                        m_aniso = 1./m_aniso
                        m0_aniso = 1./m0_aniso
                    dM = csc_matrix((m_aniso-m0_aniso,Asub.indices,Asub.indptr),
                                    shape=Asub.shape)
                    self.prop_model_predictions[dataset] = (
                        self.model_predictions[dataset] + np.asarray((
                            Asub.multiply(dM)).sum(axis=1).T)[0])
                    
                else: # approximate calculation based on Liu et al. 2019 (Direct Inversion for Three-Dim..., eq.7)
                    # this is faster than the above (~2/3 of calculation time)
                    # but it does not work well, the error seems to be so large
                    # that the anisotropy cannot be fitted.
                    m = 1./self.prop_model_slowness[idx_modified]
                    m0 = 1./self.model_slowness[idx_modified]
                    #m0 = 1./self.meanslow
                    maa1 = self.prop_psi2amp[idx_modified]*np.cos(2*self.prop_psi2[idx_modified])
                    maa2 = self.prop_psi2amp[idx_modified]*np.sin(2*self.prop_psi2[idx_modified])
                    self.prop_model_predictions[dataset] = (
                        self.model_predictions[dataset] +
                            self.A[dataset][:,idx_modified] * ((m0-m)/m0**2) -
                            self.AA1[dataset][:,idx_modified] * (maa1/m0**2) -
                            self.AA2[dataset][:,idx_modified] * (maa2/m0**2) )
                    
        
        if self.selfcheck and idx_modified is not None:
            prop_model_predictions = deepcopy(self.prop_model_predictions)
            self.calculate_ttimes(idx_modified=None,anisotropic=anisotropic)
            for dataset in self.A:
                if not np.allclose(prop_model_predictions[dataset],
                                   self.prop_model_predictions[dataset]):
                    raise Exception("Error in forward calculation!")
            self.prop_model_predictions = prop_model_predictions
                    
        
    def get_binned_azimuths(self,data_set=None):
        # this function returns an array for each gridcell
        # array([array([1,2,8,16]),array([2,5,16,17]),array([0,1,2,3,4,5,6,7,8,9...17]),...])
        # the indices show in which of the 18 bins we have coverage
        azimuths_in_bins = []
        # 19 steps from -90deg to +90deg gives 18 bins
        bins = np.linspace(-np.pi/2.,np.pi/2.,19) # 10deg bins
        for grididx in range(len(self.gridpoints)):
            binidx = []
            for dataset in self.datasets:
                if data_set is not None and data_set != dataset:
                    continue
                azims = self.PHI[dataset][:,grididx].data
                if len(azims)>0:
                    binned_count = binned_statistic(azims,[],
                                                    statistic='count',
                                                    bins=bins)
                    binidx.append(np.where(binned_count.statistic>0.)[0])
                else:
                    binidx.append([])
            binidx = np.unique(np.hstack(binidx)).astype(int)
            azimuths_in_bins.append(binidx)
        return np.array(azimuths_in_bins,dtype='object')

    def get_azimuthal_coverage(self,data_set=None):
        # This function evaluates the azimuthal coverage in each grid cell
        # returned values range from 0 (no coverage) to 180 (perfect coverage)
        azimuthal_coverage = np.array(list(map(len,self.get_binned_azimuths(data_set=data_set))))*10
        azimuthal_coverage = np.reshape(azimuthal_coverage,self.shape)
        # a bit of smoothing
        azimuthal_coverage = uniform_filter(azimuthal_coverage, size=3, mode='constant')
        
        return azimuthal_coverage.flatten()

                
    def get_data_coverage(self):
        
        # get data coverage
        data_coverage = {}
        for dataset in self.datasets:
            data_coverage[dataset] = np.array(np.abs(self.A[dataset]).sum(axis=0))[0]
        #data_coverage = data_coverage>0.
        return data_coverage
    
    def get_data_coverage_independent(self):
        """
        Same as get_data_coverage, but only independent rays are counted in
        each grid cell. That is, rays where both source and receiver stations
        are different.
        """
           
        data_coverage = np.zeros(len(self.gridpoints),dtype=int)
        for dataset in self.datasets:
            for grididx in range(len(data_coverage)):
                dataidx = self.A[dataset][:,grididx].indices
                if len(dataidx)==0:
                    continue
                sources = self.sources[dataset][dataidx]
                receivers = self.receivers[dataset][dataidx]
                
                while True:
                    data_coverage[grididx] += 1
                    src = sources[0]
                    rcv = receivers[0]
                    dependent = np.sum((sources==src)+(receivers==src)+
                                       (sources==rcv)+(receivers==rcv),
                                       axis=1,dtype=bool)
                    sources = sources[~dependent]
                    receivers = receivers[~dependent]
                    if len(sources)==0 or len(receivers)==0:
                        break

        return data_coverage
    
    def get_dataidx_in_gridcells(self):
        
        data_idx = []
        for grididx in range(len(self.gridpoints)):
            dataidx = []
            for dataset in self.datasets:
                dataidx.append(self.A[dataset][:,grididx].indices)
            data_idx.append(np.hstack(dataidx))#+idx0)
            #idx0 = np.max(np.hstack(data_idx))+1
        return np.array(data_idx,dtype='object')

    
    
    def find_data_in_dense_hull(self,coverage=None):
        """
        Parameters
        ----------
        coverage : array of same length as self.gridpoints
            Defines the criterion after which measurements are classified as
            within the dense coverage region or outside. Can be either
            ray coverage from self.get_data_coverage() or the azimuthal
            coverage from self.get_azimuthal_coverage().

        Returns
        -------
        self.dataidx_inhull checks whether the entire path from source
        to receiver is within the region of good data/azimuthal coverage.
        The array is of the same size as the number of data points.
        """
        
        if coverage is None:
            coverage = self.get_data_coverage_independent()
            lim = 10
        elif np.max(coverage)==180:
            lim = 135 # everything above 135 deg coverage is considered good
        else:
            lim = np.max([np.median(coverage[coverage>0]),10])
            
        #self.central_region = coverage > 100
        self.dataidx_inhull = {}
        for dataset in self.datasets:
            if self.kernel_type == 'rays' and not self.model_smoothing:
                B = self.A[dataset].copy()
                B[B.nonzero()] = 1
                B = B.multiply(coverage).tocsr()
                min_coverage = np.array([np.min(B[i].data) for i in range(B.shape[0])])
                
                self.dataidx_inhull[dataset] = min_coverage>=lim
                if np.sum(self.dataidx_inhull[dataset])==0:
                    if self.chain_no==1:
                        print("Warning! No ray path is within the in good coverage region for dataset",dataset)
                elif np.sum(self.dataidx_inhull[dataset])/len(self.dataidx_inhull[dataset]) < 0.2:
                    if self.chain_no==1:
                        print("less than 20 percent of the ray paths are within the in good coverage region of the model.")
                    self.dataidx_inhull[dataset] = min_coverage>=np.median(min_coverage)
            else:
                hull = Delaunay(self.gridpoints[coverage>lim])
                self.dataidx_inhull[dataset] = (
                    (hull.find_simplex(self.sources[dataset])>0)*
                    (hull.find_simplex(self.receivers[dataset])>0) )
                
                
        # #%% create a testplot
        # segments = []
        # for i in range(len(self.sources[dataset])):
        #     if not self.dataidx_inhull[dataset][i]:
        #         continue
        #     segments.append([[self.sources[dataset][i][0],self.sources[dataset][i][1]],[self.receivers[dataset][i][0],self.receivers[dataset][i][1]]])
        # lc1 = LineCollection(segments,linewidths=0.5)
        # segments = []
        # for i in range(len(self.sources[dataset])):
        #     if self.dataidx_inhull[dataset][i]:
        #         continue
        #     segments.append([[self.sources[dataset][i][0],self.sources[dataset][i][1]],[self.receivers[dataset][i][0],self.receivers[dataset][i][1]]])
        # lc2 = LineCollection(segments,linewidths=0.5)
        # plt.figure()
        # plt.pcolormesh(X,Y,np.reshape(coverage,self.shape))
        # plt.plot(self.sources[dataset][:,0],self.sources[dataset][:,1],'rv')
        # plt.gca().add_collection(lc1)
        # plt.colorbar()
        # plt.show()
        # #%%

    
    def accept_model(self,action):
        
        if np.isnan(self.prop_model_slowness).any():
            raise Exception("nan in velocity model after",action)
                    
        self.para.accept_mod(selfcheck=self.selfcheck)
        
        self.model_slowness = self.prop_model_slowness
        self.loglikelihood_current = self.loglikelihood_prop
        self.residual = self.residual_prop
        self.model_predictions = self.prop_model_predictions
        if self.anisotropic:
            self.psi2amp = self.prop_psi2amp
            self.psi2 = self.prop_psi2
              
        # the acceptance rate gives the CURRENT acceptance rate, i.e. relative to the last 100 steps
        self.acceptance_rate[action][0] = 1
        self.acceptance_rate[action] = np.roll(self.acceptance_rate[action],1)
        self.accepted_steps[action] += 1
            
        self.store_model(action,"accept")
        
        

    def reject_model(self,action):
 
        #if np.isnan(self.prop_model_slowness).any():
        #    raise Exception("nan in velocity model while rejecting",action)       
 
        self.para.reject_mod()
 
        # the acceptance rate gives the CURRENT acceptance rate, i.e. relative to the last 100 steps
        self.acceptance_rate[action][0] = 0
        self.acceptance_rate[action] = np.roll(self.acceptance_rate[action],1)
        self.rejected_steps[action] += 1
            
        self.store_model(action,"reject")
        
    
    #store_model is executed after EVERY iteration (also error updates)
    def store_model(self,action,decision):
            
        if self.selfcheck:
            if self.loglikelihood_current != self.loglikelihood(
                    self.data,self.model_predictions,self.data_std,self.foutlier)[0]:
                raise Exception(f"loglikelihood_current not right after {action}")
                
            test_slow_field = self.para.get_model()
            if (self.model_slowness-test_slow_field > 1e-10).any():
                self.para.plot()
                raise Exception(f"model not right after {action} ({decision})")
            
    
        if self.total_steps == self.nburnin: # pre-burnin average models are discarded (just a single average is kept)
            with open(self.logfile,"a") as f:
                f.write("\n###\nChain %d: Burnin phase finished.\n###\n" %self.chain_no)
            self.average_model /= self.average_model_counter
            self.modelsumsquared = self.average_model.copy()**2
            if self.anisotropic:
                self.average_anisotropy /= self.average_model_counter_aniso
                self.psi2amp_sumsquared /= self.average_model_counter_aniso
                self.anisotropy_sumsquared /= self.average_model_counter_aniso
                self.average_model_counter_aniso = 1
            self.average_model_counter = 1
            if self.interpolation_type!="wavelets":
                self.point_density = np.histogram2d(self.para.points[:,0],
                                                    self.para.points[:,1],
                                                    bins=[self.xgrid,self.ygrid])[0].T
                
            # resetting statistics
            for dataset in self.datasets:
                self.collection_datastd[dataset] = []
                
        if self.total_steps%self.collect_step == 0 or self.total_steps==1:
        
            self.collection_temperatures.append(self.temperature)
            self.collection_no_points.append(len(self.para.points))
            for dataset in self.datasets:
                self.collection_datastd[dataset].append(np.mean(self.data_std[dataset]))
            self.collection_loglikelihood.append(self.loglikelihood_current)
            self.collection_residuals.append(self.residual)

            if not self.store_models or self.total_steps < self.nburnin:
                if self.loglikelihood_current > self.collection_points[-1][-1] and self.interpolation_type!='wavelets':
                    self.collection_points = [[self.para.points.astype('float32'),
                                               self.para.vs.astype('float16'),
                                               self.loglikelihood_current]]
            
            if self.temperature==1:
            
                self.average_model_counter += 1
                self.average_model += 1./self.model_slowness
                # modelsumsquared is needed for the calculation of the model uncertainty
                self.modelsumsquared += (1./self.model_slowness)**2
                if self.selfcheck:
                    if (self.modelsumsquared - self.average_model**2/self.average_model_counter < -0.1).any():
                        raise Exception("error in modeluncertainty calculation") 
                if self.anisotropic:
                    self.average_model_counter_aniso += 1
                    # the final average angles are then given by
                    # 0.5*np.arctan2(self.average_anisotropy[:,1]/self.average_model_counter,
                    #                self.average_anisotropy[:,0]/self.average_model_counter)
                    # the amplitude by 
                    # sqrt((self.average_anisotropy[:,0]/self.average_model_counter)**2 +
                    #      (self.average_anisotropy[:,1]/self.average_model_counter)**2)
                    self.average_anisotropy += np.column_stack((np.cos(2*self.psi2)*self.psi2amp,
                                                                np.sin(2*self.psi2)*self.psi2amp))
                    self.anisotropy_sumsquared += np.column_stack(((np.cos(2*self.psi2)*self.psi2amp)**2,
                                                                   (np.sin(2*self.psi2)*self.psi2amp)**2))
                    self.psi2amp_sumsquared += self.psi2amp**2
                    #self.psi2_sumsquared += (np.sin(self.psi2))**2
                
            if self.total_steps > self.nburnin:
                if self.store_models:
                    # float16 saves diskspace and should be precise enough
                    self.collection_models.append((1./self.model_slowness).astype('float16'))
                    if self.anisotropic:
                        # needed to create the histogram plot
                        self.collection_psi2.append(self.psi2.astype('float16'))
                        self.collection_psi2amp.append(self.psi2amp.astype('float16'))
                if self.interpolation_type!="wavelets":
                    self.point_density += np.histogram2d(
                        self.para.points[:,0],self.para.points[:,1],
                        bins=[self.xgrid,self.ygrid])[0].T
                    if self.store_models:
                        self.collection_points.append([self.para.points.astype('float32'),
                                                       self.para.vs.astype('float16'),
                                                       self.loglikelihood_current])
                    
    
        if self.total_steps % self.print_stats_step == 0 or self.total_steps==1:
            self.print_statistics()
            
        self.cleanup(action,decision)
        

    def update_proposal_stds(self):
            
        # I should check whether it makes sense to increase the acceptance rate
        # for the delayed rejection scheme. Some preliminary tests have indicated
        # that the standard deviation may be too large if the rate is fixed 
        # below 50%
        if self.delayed_rejection:
            minrate = 65
            maxrate = 70
        else:
            minrate = 42
            maxrate = 48
        # velocity udates
        if np.sum(self.acceptance_rate['velocity_update']) < minrate and self.propvelstd_pointupdate > 0.001:
            self.propvelstd_pointupdate *= 0.99
        elif np.sum(self.acceptance_rate['velocity_update']) > maxrate:
            self.propvelstd_pointupdate *= 1.01   
        # moving points
        if np.sum(self.acceptance_rate['move']) < minrate:
            if self.propmovestd*0.99 > np.min([self.xgridspacing/10.,self.ygridspacing/10.]):
                self.propmovestd *= 0.99
        elif np.sum(self.acceptance_rate['move']) > maxrate:
            self.propmovestd *= 1.01
        # for birth/death steps it is often not possible to get a good
        # acceptance rate and thus there is no good way to modify the proposal
        # standard deviation. Therefore, we simply take the same as for the
        # point updates
        if not isinstance(self.propvelstd_dimchange,str):
            self.propvelstd_dimchange = self.propvelstd_pointupdate
        
       

    def cleanup(self,action,decision):

        # optional progress visualization every visualize_step iteration
        if self.visualize and self.total_steps%self.visualize_step == 0:
            self.visualize_progress()

        # for the acceptance calculation in case of anisotropic birth/death
        self.aniso_factor = 0.

        # this is just to be sure that the variables are not reused and everything works properly
        # could be removed in the future
        self.prop_model_slowness = []
        self.prop_model_predictions = []
        self.loglikelihood_prop = []
        self.residual_prop = []
        if self.anisotropic:
            self.prop_psi2amp = []
            self.prop_psi2 = []
        self.dx1 = self.propdist_std1 = []                         
    
    def loglikelihood(self,data,predictions,data_std,foutlier=None,
                      error_update=False,bootstrap_idx=None):

        # this is the likelihood P(d|m) for a hierarchical Bayesian formulation
        # meaning that the std is also considered unknown.
        # however, if it is fixed during the model search the first term will just cancel out   

        # For testing: reconstruct the prior by setting the likelihood to a 
        # fixed value. This way it cancels out in all calculations above
        # equivalent to having no data. This way, the algorithm just samples
        # the prior.
        # For the number of cells this means a uniform distribution if the 
        # velocity for a birth step is drawn from a uniform distribution
        # otherwise it is a normal distribution with a peak around the mean
        # number of cells. The width of the normal distribution depends on the
        # propvelstd_dimchange parameter.
        # For the data std a uniform distribution in case
        # data_std = 'absolute' is reconstructed
        # return 0,0 # for testing, should reconstruct the prior

        # the standard deviation has be be a vector of the same size as data here.
        # otherwise, the first term has to be multiplied by len(data)
        # P(d|m) ~ 1/sigma**N * exp(-(G(m)-d)**2/2*sigma**2) 
        # -> logspace: P(d|m) ~ -log(sigma)*N + (-(G(m)-d)**2)/2*sigma**2
               
        loglikelihood = 0.
        total_residual = 0.

        for dataset in self.datasets:
            
            #self.minresidual = np.mean(data_std[dataset])
            
            if self.prop_bias is None:
                data_corr = data[dataset] - self.bias[dataset]
            else:
                data_corr = data[dataset] - self.prop_bias[dataset]
            
            if self.data_units == 'velocity':
                residuals = (self.path_dists[dataset]/predictions[dataset] - 
                             self.path_dists[dataset]/data_corr )
                # setting a minimum error
                if self.minresidual>0.:
                    residuals[np.abs(residuals)<self.minresidual] = self.minresidual
            else:
                residuals = predictions[dataset] - data_corr
                
            residuals = residuals
            total_residual += np.sum(np.abs(residuals))
            stadev = data_std[dataset]#.copy()
            
            if self.total_steps%200000==0:
                if self.chain_no==1:
                    print("    info: mean abs residuals in central region: %.3f; in other regions: %.3f" %(
                          np.mean(np.abs(residuals[self.dataidx_inhull[dataset]])),
                          np.mean(np.abs(residuals[~self.dataidx_inhull[dataset]]))))
            
            # experimental, using only the data in the central region for
            # error updates
            if False: #error_update:
                idx = self.dataidx_inhull[dataset]
                residuals = residuals[idx]
                stadev = stadev[idx]
            if bootstrap_idx is not None:
                residuals = residuals[bootstrap_idx[dataset]]
                stadev = stadev[bootstrap_idx[dataset]]          
                
            # stadev that maximizes the loglikelihood (see Sambridge 2013):
            # std_opt = np.sqrt(1./len(residuals)*np.sum(residuals**2))
            
            if False: # experimental, does not make a big difference
                # multimodal gaussian mixuture, assuming 40% of measurements are shifted by one cycle
                if self.data_units != 'velocity':
                    residuals1 = residuals-self.period
                    residuals2 = residuals+self.period
                else:
                    residuals1 = (self.path_dists[dataset]/predictions[dataset] - 
                                 self.path_dists[dataset]/(data_corr+self.period) )
                    residuals2 = (self.path_dists[dataset]/predictions[dataset] - 
                                 self.path_dists[dataset]/(data_corr-self.period) )
                loglikelihood += np.sum(np.log(
                    0.2*1./(np.sqrt(2*np.pi)*stadev)*np.exp((-(residuals1/stadev)**2)/2.) +
                    0.6*1./(np.sqrt(2*np.pi)*stadev)*np.exp((-(residuals /stadev)**2)/2.) +
                    0.2*1./(np.sqrt(2*np.pi)*stadev)*np.exp((-(residuals2/stadev)**2)/2.)
                    + 1e-300 ))
            
            elif not self.outlier_model:
                if self.misfit_norm == 'L2':
                    # the np.sqrt(2*np.pi) term is not necessary (always cancels out in the
                    # likelihood ratios), but is kept here to make it more comparable to
                    # the outlier model likelihoods which include the 2pi term.
                    loglikelihood += ((-np.sum(np.log(np.sqrt(2*np.pi)*stadev) ) - 
                                       np.sum((residuals/stadev)**2)/2.))
                elif self.misfit_norm == 'L1':
                    loglikelihood += (-np.sum(np.log(2*stadev) ) - 
                                       np.sum(np.abs(residuals/stadev)))
                
            else:
                if self.misfit_norm == 'L2':
                    likelihood_gauss = ((1-foutlier[dataset]) *
                                        1./(np.sqrt(2*np.pi)*stadev) *
                                        np.exp((-(residuals/stadev)**2)/2.))
                    likelihood_uniform = np.zeros(len(residuals))
                    likelihood_uniform[np.abs(residuals)<=self.widthoutlier/2.] = foutlier[dataset]/self.widthoutlier
                    # waterlevel 1e-300 to make sure that there is no inf in np.log
                    loglikelihood += np.sum( np.log( likelihood_gauss + likelihood_uniform + 1e-300 ) )
    
                elif self.misfit_norm == 'L1': # is that correct?
                    raise Exception("outliers and L1 norm currently not implemented.")

        return loglikelihood, total_residual
    
    
    def update_widthoutlier(self,data,predictions,data_std,foutlier):

        raise Exception("currently not used")
        self.widthoutlier = 0.
        for dataset in self.datasets:
            
            #ttime_mean = self.A[dataset]*np.ones(len(self.gridpoints))*0.5041
            #residuals = self.data[dataset] - ttime_mean
            #self.widthoutlier[dataset] = np.max(residuals)-np.min(residuals)
            self.widthoutlier = np.max([self.widthoutlier,
                                        np.max(self.data[dataset])-np.min(self.data[dataset])])
            # try:
            #     self.collection_widthoutlier[dataset]
            # except:
            #     self.collection_widthoutlier = {}
            # residuals = predictions[dataset] - data[dataset]
            
            # if (residuals > self.outlierwidth[dataset][1]).any():
            #     self.outlierwidth[dataset][1] = np.max(residuals)
            #     self.widthoutlier[dataset] = np.diff(self.outlierwidth[dataset])
            # elif (residuals < self.outlierwidth[dataset][0]).any():
            #     self.outlierwidth[dataset][0] = np.min(residuals)
            #     self.widthoutlier[dataset] = np.diff(self.outlierwidth[dataset])[0]
            # self.widthoutlier[dataset] = 400.
            # try:
            #     self.collection_widthoutlier[dataset].append(self.widthoutlier[dataset])
            # except:
            #     self.collection_widthoutlier[dataset] = [self.widthoutlier[dataset]]
                
        # the loglikelihood current value changes with updated widthoutlier,
        # so it is necessary to overwrite the old value otherwise the algorithm
        # may get stuck.
        self.loglikelihood_current, self.residual = self.loglikelihood(data,predictions,
                                                        data_std,foutlier)  


    def update_chain(self):
        
        # this is necessary after the paths have changed
        with open(self.logfile,"a") as f:
            f.write("\nUpdating model predictions and current loglikelihood.\n")
            
        if self.kernel_type == 'sens_kernels' and self.anisotropic:
            print("Warning! setting PHI matrix for sens kernels!")
            for dataset in self.data:
                dx = self.sources[dataset]-self.receivers[dataset]
                phi = np.arctan(dx[:,1]/(dx[:,0]+1e-16))
                PHI = self.PHI[dataset].tocsr()
                for i in range(len(self.data[dataset])):
                    PHI[i].data[:] = phi[i]
                self.PHI[dataset] = PHI.tocsc()
        
        if self.total_steps==0:
            try:
                self.data_coverage = self.get_azimuthal_coverage()
            except:
                if self.chain_no==1:
                    print("    info: could not get azimuthal data coverage (no PHI matrix defined?)")
                self.data_coverage = self.get_data_coverage_independent()
            self.find_data_in_dense_hull(coverage=self.data_coverage)


        if self.anisotropic:
            self.prop_model_slowness,self.prop_psi2amp,self.prop_psi2 = (
                self.para.get_model(anisotropic=True))
            self.psi2amp = self.prop_psi2amp
            self.psi2 = self.prop_psi2
            # storing column indices of nonzero elments of the A matrices
            # this is faster than getting them at each iteration
            self.nonzero = {}
            for dataset in self.A:
                self.nonzero[dataset] = []
                for i in range(len(self.gridpoints)):
                    self.nonzero[dataset].append(i*np.ones(len(
                        self.A[dataset][:,i].nonzero()[1]),dtype=np.int32))
                self.nonzero[dataset] = np.array(self.nonzero[dataset],
                                                 dtype='object')
        else:
            self.prop_model_slowness = self.para.get_model()
        self.model_slowness = self.prop_model_slowness
        
        check_grad = self.check_gradients()
        if not check_grad:
            raise Exception("invalid model! gradients too large.")
            
        # path dists may have changed, therefore update also data stds
        if True:#self.error_type not in ["fixed","absolute"]:
            self.path_dists = {}
            for dataset in self.datasets:
                if self.kernel_type == 'sens_kernels' or self.model_smoothing:
                    self.path_dists[dataset] = np.sqrt(np.sum((self.sources[dataset] - self.receivers[dataset])**2,axis=1))
                else:
                    self.path_dists[dataset] = np.asarray(self.A[dataset].sum(axis=1).T)[0]
                if 'relative' in self.error_type:
                    self.data_std[dataset] = self.std_scaling_factor[dataset] * self.path_dists[dataset] + self.std_intercept[dataset]
                    if (self.data_std[dataset] <= 0.0).any():
                        self.std_intercept = np.mean(self.data_std[dataset]) 
                        self.std_scaling_factor[dataset] = 0.
                        self.data_std[dataset] = self.std_scaling_factor[dataset] * self.path_dists[dataset] + self.std_intercept[dataset]
                if 'stationerrors' in self.error_type:
                    self.data_std[dataset] += np.sum(
                        self.stationerrors[dataset][self.stationerrors_index[dataset]],axis=1)
               
        if True:
            # for testing: create a plot showing data coverage (azimuthal coverage or ray coverage)
            if self.chain_no==1:
                plt.ioff()
                fig = plt.figure(figsize=(14,8))
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)
                X,Y = np.meshgrid(self.x,self.y)
                coverage = np.reshape(self.data_coverage,self.shape)
                cbar = ax2.pcolormesh(X,Y,coverage,shading='nearest')
                plt.colorbar(cbar,shrink=0.5,pad=0.02)
                ax1.pcolormesh(X,Y,coverage,shading='nearest')
                segments_in = []
                segments_out = []
                for dataset in self.data:
                    for i in range(len(self.data[dataset])):
                        x1,y1 = self.sources[dataset][i]
                        x2,y2 = self.receivers[dataset][i]
                        if self.dataidx_inhull[dataset][i]:
                            segments_in.append([[x1,y1],[x2,y2]])
                        else:
                            segments_out.append([[x1,y1],[x2,y2]])
                lc2 = LineCollection(segments_out, linewidths=0.3,alpha=0.4,color='red',label='outside good coverage area')
                ax1.add_collection(lc2) 
                lc = LineCollection(segments_in, linewidths=0.3,alpha=0.4,color='black',label='inside good coverage area')
                ax1.add_collection(lc)
                ax1.legend()
                ax1.set_aspect('equal')
                ax2.set_aspect('equal')
                plt.savefig(os.path.join(os.path.dirname(self.logfile),
                            "data_coverage.png"), bbox_inches='tight',dpi=200)
                plt.close(fig)
          
        self.calculate_ttimes(anisotropic=self.anisotropic)
        self.model_predictions = self.prop_model_predictions

        self.loglikelihood_current, self.residual = self.loglikelihood(
            self.data,self.model_predictions,self.data_std,self.foutlier)

        with open(self.logfile,"a") as f:
            f.write("Temperature: %.3f, Loglikelihood: %.1f, Residual: %.1f\n" %(
                self.temperature,self.loglikelihood_current,self.residual))
 
    
    def print_statistics(self):
        
        #if self.interpolation_type=='wavelets':
        #    print(self.para.coeff_range)

        with open(self.logfile,"a") as f:
            f.write("\n################################\n")
            f.write(time.ctime()+"\n")
            f.write("Chain %d: (Temperature: %.3f, Loglikelihood: %.1f, Residual: %.1f)\n" %(self.chain_no,self.temperature,self.loglikelihood_current, self.residual))
            f.write("    Iteration: %d\n" %self.total_steps)
            f.write("    Acceptance rate parameter update: %d %%\n" %(np.sum(self.acceptance_rate['velocity_update'])))
            f.write("    Acceptance rate birth: %d %%\n" %(np.sum(self.acceptance_rate['birth'])))
            f.write("    Acceptance rate death: %d %%\n" %(np.sum(self.acceptance_rate['death'])))
            f.write("    Acceptance rate move: %d %%\n" %(np.sum(self.acceptance_rate['move'])))
            #f.write("    Acceptance rate swap: %d\n" %(np.sum(self.acceptance_rate['parameter_swap'])))
            if self.anisotropic:# and self.total_steps>int(self.nburnin/2):
                f.write("    Acceptance rate anisotropy update direction: %d %%\n" %(np.sum(self.acceptance_rate['anisotropy_update_direction'])))
                f.write("    Acceptance rate anisotropy update amplitude: %d %%\n" %(np.sum(self.acceptance_rate['anisotropy_update_amplitude'])))
                f.write("    Acceptance rate anisotropy birth: %d %%\n" %(np.sum(self.acceptance_rate['anisotropy_birth'])))
                f.write("    Acceptance rate anisotropy death: %d %%\n" %(np.sum(self.acceptance_rate['anisotropy_death'])))

            #f.write("    Acceptance rate birth/death should be around 40-50%. The Acceptance rate move/cell update will be higher due to the delayed rejection.")
            f.write("    Proposal standard deviation parameter update: %.3f  move: %.2f \n" %(self.propvelstd_pointupdate,self.propmovestd))
            if self.total_steps>100:
                f.write("    Mean no of parameters: %d\n" %(np.mean(self.collection_no_points[-100:])))
            else:
                f.write("    Start no of parameters: %d\n" %(len(self.para.points)))
            for dataset in self.datasets:
                if self.outlier_model:
                     f.write("    %s : Mean data std: %.2f    Outlier fraction: %.1f %%\n" %(dataset, np.mean(self.data_std[dataset]),self.foutlier[dataset]*100))
                else:
                    f.write("    %s : Mean data std: %.2f\n" %(dataset,np.mean(self.data_std[dataset])))
            f.write("# # # # # # # # # # # # # # \n\n")

    def visualize_progress(self,initialize=False):
        
        X,Y = np.meshgrid(self.x,self.y)
        if initialize:
            fig = plt.figure(figsize=(14,10))
            gs = gridspec.GridSpec(4,5,width_ratios=[1,1,1,1,0.05],height_ratios=[0.7,0.7,0.7,1],wspace=0.5)
            self.ax11 = fig.add_subplot(gs[0:3,0:2])
            self.ax12 = fig.add_subplot(gs[0:3,2:4])
            self.cbarax = fig.add_subplot(gs[1,4])
            self.ax11.set_aspect('equal')
            self.ax12.set_aspect('equal')
            self.ax21 = fig.add_subplot(gs[3,0])
            self.ax22 = fig.add_subplot(gs[3,1])
            self.ax23 = fig.add_subplot(gs[3,2])
            self.ax24 = fig.add_subplot(gs[3,3])
            plt.ion()    
            
            cbar = self.ax12.pcolormesh(X,Y,
                                        np.ones_like(X)*np.mean([self.velmin,self.velmax]),
                                        vmin=self.velmin,vmax=self.velmax)
            self.cbar = plt.colorbar(cbar,cax=self.cbarax,
                                     label='velocity')
            self.ax11.set_title('current proposal')
            self.ax12.set_title('average model')
            self.ax11.set_aspect('equal')
            self.ax12.set_aspect('equal')

        else:
            avg_model = self.average_model.reshape(np.shape(X))/self.average_model_counter
            vmax = np.max(avg_model)
            vmin = np.min(avg_model)
            self.ax11.clear()
            self.ax12.clear()
            if self.prop_model_slowness == []:
                self.ax11.pcolormesh(X,Y,
                                     np.reshape(1./self.model_slowness,X.shape),
                                     vmin=vmin,vmax=vmax,cmap=cm.roma)
            else:
                self.ax11.pcolormesh(X,Y,
                                     np.reshape(1./self.prop_model_slowness,X.shape),
                                     vmin=vmin,vmax=vmax,cmap=cm.roma)                
            self.ax11.plot(self.para.points[:,0],self.para.points[:,1],'ko')
            self.ax11.set_title('current proposal')
            cbar = self.ax12.pcolormesh(X,Y,
                                        self.average_model.reshape(np.shape(X))/self.average_model_counter,
                                        vmin=vmin,vmax=vmax,cmap=cm.roma)
            self.ax12.set_title('average model')
            self.ax12.plot(self.stations[:,0],self.stations[:,1],'rv',markeredgecolor='black')
            
            self.cbar.update_normal(cbar)
            
            self.ax21.clear()
            no_cells = np.min([len(np.unique(self.collection_no_points)),25])
            self.ax21.hist(self.collection_no_points,no_cells)
            self.ax21.set_xlabel('no of voronoi cells',fontsize=8)
            self.ax21.set_ylabel('no of models',fontsize=8)
            self.ax21.tick_params(axis='both', which='major', labelsize=8)
            
            self.ax22.clear()
            for dataset in self.collection_datastd:
                self.ax22.hist(self.collection_datastd[dataset],25,label=dataset)
            self.ax22.legend(loc='upper right',fontsize=6)
            self.ax22.set_xlabel('data std',fontsize=8)
            self.ax22.set_ylabel('no of models',fontsize=8)
            self.ax22.tick_params(axis='both', which='major', labelsize=8)
            
            self.ax23.clear()
            self.ax23.plot(np.arange(len(self.collection_loglikelihood))*self.collect_step,
                           self.collection_loglikelihood)
            self.ax23.set_ylabel('log likelihood',fontsize=8)
            self.ax23.set_xlabel('iteration no',fontsize=8)
            self.ax23.tick_params(axis='both', which='major', labelsize=8)
            
            self.ax24.clear()
            self.ax24.text(0,0.95,"Chain number: %d" %self.chain_no)
            self.ax24.text(0,0.85,"Iterations: {:,}".format(self.total_steps))
            self.ax24.text(0,0.75,"Burnin samples: {:,}".format(self.nburnin))
            self.ax24.text(0,0.65,"Std proposal data uncertainty: %.3f" %self.propsigmastd)               
            self.ax24.text(0,0.55,"Std proposal velocity update: %.3f" %self.propvelstd_pointupdate)
            if isinstance(self.propvelstd_dimchange,str):
                self.ax24.text(0,0.45, "Std proposal birth/death: %s" %self.propvelstd_dimchange)
            else:
                self.ax24.text(0,0.45, "Std proposal birth/death: %.3f" %self.propvelstd_dimchange)
            self.ax24.text(0,0.35,"Std proposal move cell: %.3f" %self.propmovestd)
            self.ax24.text(0,0.25,"Temperature: %.1f" %self.temperature)
            self.ax24.text(0,0.15,"Average of every %dth model" %self.collect_step)
            if self.total_steps>self.update_paths_interval:
                self.ax24.text(0,0.05,"Paths updated every {:,}th iteration.".format(self.update_paths_interval))
            self.ax24.axis('off')
            
        plt.draw()
        plt.savefig("visualization/img_%d.jpg" %self.total_steps,dpi=100,bbox_inches='tight')
        plt.pause(0.01)


    def plot_cur_mod(self):
        from scipy.spatial import delaunay_plot_2d
        
        mod = self.para.get_model()
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        cbar = ax1.pcolormesh(self.x,self.y,1./mod.reshape(self.shape),cmap=cm.roma,shading='nearest')
        ax1.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.')
        tri = Delaunay(self.para.points)
        _ = delaunay_plot_2d(tri,ax=ax1)
        for i in range(len(self.para.points)):
            plt.text(self.para.points[i,0],self.para.points[i,1],"%d" %i,fontsize=5)
        stations = []
        for dataset in self.datasets:
            stations.append(np.vstack((self.sources[dataset],self.receivers[dataset])))
        stations = np.unique(np.vstack(stations),axis=0)
        ax1.plot(stations[:,0],stations[:,1],'rv',markeredgecolor='black')
        plt.colorbar(cbar,shrink=0.5)
        plt.show()

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        cbar = ax1.pcolormesh(self.x,self.y,1./mod.reshape(self.shape),cmap=plt.roma,shading='nearest')
        stations = []
        for dataset in self.datasets:
            stations.append(np.vstack((self.sources[dataset],self.receivers[dataset])))
        stations = np.unique(np.vstack(stations),axis=0)
        ax1.plot(stations[:,0],stations[:,1],'rv',markeredgecolor='black')
        for dataset in self.data:
            idx = self.dataidx_inhull[dataset]
            for k in idx:
                plt.plot([self.sources[dataset][k,0],self.receivers[dataset][k,0]],
                         [self.sources[dataset][k,1],self.receivers[dataset][k,1]],'k-',linewidth=0.2)
        plt.colorbar(cbar,shrink=0.5)
        plt.show()
        

        

    def plot(self,saveplot = False, output_location = "./", suffix=""):
        
        if saveplot:
            plt.ioff()
        #%%
        X,Y = np.meshgrid(self.x,self.y)
        if self.anisotropic:
            # mean angle given by np.arctan(y/x) -> arctan(sum(sin(phi))/sum(cos(phi)))
            anisotropy_dirmean = 0.5*np.arctan2(self.average_anisotropy[:,1],
                                                self.average_anisotropy[:,0])
            anisotropy_ampmean = np.sqrt((self.average_anisotropy[:,0]/self.average_model_counter_aniso)**2 +
                                         (self.average_anisotropy[:,1]/self.average_model_counter_aniso)**2)
    
            fig = plt.figure(figsize=(14,6))
            ax1 = fig.add_subplot(121)
            cbar = ax1.pcolormesh(X,Y,
                                  100*anisotropy_ampmean.reshape(np.shape(X)),
                                  shading='nearest')
            plt.colorbar(cbar,shrink=0.6)
            ax2 = fig.add_subplot(122)
            q = ax2.quiver(X[::2,::2],Y[::2,::2],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2],
                           angles=anisotropy_dirmean.reshape(np.shape(X))[::2,::2]/np.pi*180,
                           headwidth=0,headlength=0,headaxislength=0,
                           pivot='middle',scale=70.,width=0.005,
                           color='yellow',edgecolor='k',#scale=80.,width=0.0035,
                           linewidth=0.5)
            if saveplot:
                plt.savefig(os.path.join(output_location,"result_chain%d_anisotropy%s.png" %(self.chain_no,suffix)), bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()
                
            if len(self.collection_psi2)>100:
                from matplotlib.gridspec import GridSpec
                from matplotlib.patches import ConnectionPatch
                anisodirs = np.vstack(self.collection_psi2)
                anisoamps = np.vstack(self.collection_psi2amp)
                fig = plt.figure(figsize=(16,12))
                gs = GridSpec(5, 5)
                axmap = fig.add_subplot(gs[1:4,1:4])
                cbar = axmap.pcolormesh(
                    X,Y,self.average_model.reshape(np.shape(X))/self.average_model_counter,
                    cmap=plt.cm.jet_r,shading='nearest')
                #plt.colorbar(cbar,shrink = 0.5)
                q = axmap.quiver(X[::5,::5],Y[::5,::5],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::5,::5],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::5,::5],
                           angles=anisotropy_dirmean.reshape(np.shape(X))[::5,::5]/np.pi*180,
                           headwidth=0,headlength=0,headaxislength=0,
                           pivot='middle',scale=70.,width=0.005,
                           color='yellow',edgecolor='k',#scale=80.,width=0.0035,
                           linewidth=0.5)
                qk = axmap.quiverkey(q, X=0.06, Y=0.04, U=3, label='3%',
                                   labelpos='E',#edgecolor='w',linewidth=0.5,
                                   fontproperties=dict(size=10))
                gsidx = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,4),(2,0),(2,4),
                         (3,0),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]
                gsanchors = [(90,0),(0,0),(0,0),(0,0),(-90,0),
                             (90,10),(-90,10),(90,10),(-90,10),(90,10),(-90,10),
                             (90,20),(0,20),(0,20),(0,20),(-90,20)]
                if anisodirs.shape[1] == 28350: # for Kaestle et al. publication plot
                    plot_idx = [24970, 21800, 21840, 18450 ,23400, 17620, 15400, 13840,
                                12600, 10600,  12200,  7400,  7430,  16500, 2600, 9000]
                else:
                    plot_idx = np.sort(np.random.choice(np.arange(anisodirs.shape[1],dtype=int),replace=False,size=16))
                for jj,j in enumerate(plot_idx):
                    ax = fig.add_subplot(gs[gsidx[jj]])
                    dirs = anisodirs[:,j]
                    amps = anisoamps[:,j]
                    dirs = dirs[amps>0.005] # greater 0.5%
                    # dirs is in range 0 to 2pi, bring to -pi/2 to pi/2 range using arctan(tan(..))
                    ax.hist(np.arctan(np.tan(dirs))/np.pi*180,bins=np.linspace(-90,90,37))
                    ax.vlines(anisotropy_dirmean[j]/np.pi*180,0,20,color='red')
                    xyA = gsanchors[jj]
                    xyB = (X.flatten()[j],Y.flatten()[j])
                    axmap.plot(xyB[0],xyB[1],'ko')
                    con = ConnectionPatch(
                        xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                        axesA=ax, axesB=axmap, color="black",linestyle='solid',
                        linewidth=0.5)
                    ax.add_artist(con)
                if saveplot:
                    plt.savefig(os.path.join(output_location,"result_chain%d_anisotropy_histograms%s.png" %(self.chain_no,suffix)), bbox_inches='tight')
                    plt.close(fig)
                else:
                    plt.show()
                

        #%%
        if saveplot:
            plt.ioff()
        fig = plt.figure(figsize=(14,10))
        gs = gridspec.GridSpec(2,4,height_ratios=[2,1],wspace=0.5)
        ax11 = fig.add_subplot(gs[0,0:2])
        ax12 = fig.add_subplot(gs[0,2:4])
        ax11.set_aspect('equal')
        ax12.set_aspect('equal')
        ax21 = fig.add_subplot(gs[1,0])
        ax22 = fig.add_subplot(gs[1,1])
        ax23 = fig.add_subplot(gs[1,2])
        ax24 = fig.add_subplot(gs[1,3])
        
        vmax = np.max(self.average_model.reshape(np.shape(X))/self.average_model_counter)
        vmin = np.min(self.average_model.reshape(np.shape(X))/self.average_model_counter)
        ax11.clear()
        ax12.clear()
        if len(self.collection_points)>0:
            idx_bestfit = np.array(self.collection_points,dtype='object')[:,2].argmax()
            best_pointmod = np.array(self.collection_points,dtype='object')[idx_bestfit,0]
            if best_pointmod is not None:
                best_vsmod = np.array(self.collection_points,dtype='object')[idx_bestfit,1]
                bestfit_slowfield = self.para.get_model(best_pointmod, best_vsmod).reshape(np.shape(X))
                ax11.pcolormesh(X,Y,1./bestfit_slowfield,shading='nearest',
                                     vmin=self.velmin,vmax=self.velmax,cmap=cm.roma)
                ax11.set_title('best fitting model')
                ax11.plot(np.array(self.collection_points,dtype='object')[idx_bestfit,0][:,0],
                               np.array(self.collection_points,dtype='object')[idx_bestfit,0][:,1],'ko')
        cbar = ax12.pcolormesh(X,Y,
                               self.average_model.reshape(np.shape(X))/self.average_model_counter,
                               vmin=self.velmin,vmax=self.velmax,
                               cmap=cm.roma,shading='nearest')
        stations = []
        for dataset in self.datasets:
            stations.append(np.vstack((self.sources[dataset],self.receivers[dataset])))
        stations = np.unique(np.vstack(stations),axis=0)
        ax12.plot(stations[:,0],stations[:,1],'rv',markeredgecolor='black')
        ax12.set_title('average model')
        cbarax = ax12.inset_axes([1.05,0.05,0.03,0.6])
        cbar = plt.colorbar(cbar,cax=cbarax,label='velocity')
        
        ax21.clear()
        no_cells = np.min([len(np.unique(self.collection_no_points)),25])
        ax21.hist(self.collection_no_points,no_cells)
        ax21.set_xlabel('no of voronoi cells',fontsize=8)
        ax21.set_ylabel('no of models',fontsize=8)
        ax21.tick_params(axis='both', which='major', labelsize=8)
        
        ax22.clear()
        for dataset in self.datasets:
            ax22.hist(self.collection_datastd[dataset],25,label=dataset)
        ax22.legend(loc='upper right',fontsize=6)
        ax22.set_xlabel('data std',fontsize=8)
        ax22.set_ylabel('no of models',fontsize=8)
        ax22.tick_params(axis='both', which='major', labelsize=8)
        
        ax23.clear()
        iterations = np.arange(0,self.total_steps+1,self.collect_step)
        it_burnin = int(self.nburnin/self.collect_step)
        ax23.plot(iterations[int(it_burnin/4):],self.collection_loglikelihood[int(it_burnin/4):])   
        if self.total_steps>self.nburnin:
            ax23.plot([self.nburnin,self.nburnin],
                      [np.min(self.collection_loglikelihood[int(it_burnin/4):]),
                       np.max(self.collection_loglikelihood)],'k',lw=0.4)            
            #ax23.set_xlim(int(self.nburnin/4.),self.total_steps+1)
            #ax23.set_ylim(np.min(self.collection_loglikelihood[int(self.nburnin/4):]),
            #                   np.max(self.collection_loglikelihood))
        ax23.set_ylabel('log likelihood',fontsize=8)
        ax23.set_xlabel('iteration no',fontsize=8)
        ax23.tick_params(axis='both', which='major', labelsize=8)
        
        ax24.clear()
        ax24.text(0,0.95,"Chain number: %d" %self.chain_no)
        ax24.text(0,0.85,"Iterations: {:,}".format(self.total_steps))
        ax24.text(0,0.75,"Burnin samples: {:,}".format(self.nburnin))
        ax24.text(0,0.65,"Std proposal data uncertainty: %.3f" %self.propsigmastd)
        ax24.text(0,0.55,"Std proposal velocity update: %.3f" %self.propvelstd_pointupdate)
        if isinstance(self.propvelstd_dimchange,str):
            ax24.text(0,0.45, "Std proposal birth/death: %s" %self.propvelstd_dimchange)
        else:
            ax24.text(0,0.45, "Std proposal birth/death: %.3f" %self.propvelstd_dimchange)
        ax24.text(0,0.35,"Std proposal move cell: %.3f" %self.propmovestd)
        ax24.text(0,0.25,"Temperature: %.3f" %self.temperature)
        ax24.text(0,0.15,"Average of every %dth model" %self.collect_step)
        if self.total_steps>self.update_paths_interval:
            ax24.text(0,0.05,"Paths updated every {:,}th iteration.".format(self.update_paths_interval))
        ax24.axis('off')
        if saveplot:
            plt.savefig(os.path.join(output_location,"result_chain%d%s.png" %(self.chain_no,suffix)), bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
            
        # try:
        #     fig = plt.figure()
        #     plt.plot(self.sourcedist)
        #     plt.savefig(os.path.join(output_location,"sourcedist_chain%d.png" %self.chain_no),
        #                 dpi=100,bbox_inches='tight')
        #     plt.close(fig)
        # except:
        #     pass
        #%%
        #ax24.text(0,0.1,"Iterations: %d" %self.total_steps)
        
            
