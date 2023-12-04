#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 17:14:27 2018

@author: emanuel
"""

#import multiprocessing
import os, re
import matplotlib
if not bool(os.environ.get('DISPLAY', None)):
    matplotlib.use('Agg')
#print("Warning! importing rjtransdim2d_test.py")
from rjtransdim2d import RJMC
import numpy as np
import time, pickle, glob, copy
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
import cmcrameri.cm as cmcram
import cartopy.crs as ccrs
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
import pyproj 
#from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter  
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.sparse import vstack, lil_matrix, save_npz, load_npz, csc_matrix
#from mpi4py import MPI
import gzip
import FMM

# CLASSES # # # 
class initialize_params(object):
    
    def __init__(self, minx=None, maxx=None, miny=None, maxy=None,
                 xgridspacing = None, ygridspacing = None,
                 projection = None,
                 interpolation_type = None,
                 wavelength_dependent_gridspacing = None,
                 misfit_norm = None,
                 model_outliers = None,
                 init_no_points = None,
                 init_vel_points = None,
                 propvelstd_velocity_update = None,
                 propvelstd_birth_death = None,
                 fixed_propvel_std = None,
                 propmove_std = None,
                 delayed_rejection = None,
                 number_of_iterations = None,
                 number_of_burnin_samples = None,
                 update_path_interval = None,
                 period = None,
                 meanvel = None,
                 kernel_type = None,
                 propsigma_std = None,
                 data_std_type = None,
                 temperature = None,
                 collect_step = None,
                 anisotropic = None,
                 propstd_anisoamp = None,
                 propstd_anisodir = None,
                 store_models = None,
                 print_stats_step = None,
                 logfile_path = None,
                 visualize = False,
                 visualize_step = None,
                 resume = False):
        
        loc = locals()
        exception = False
        for argument in loc:
            if (resume and
                argument in ["minx","maxx","miny","maxy","meanvel",
                             "xgridspacing","ygridspacing","projection",
                             "wavelength_dependent_gridspacing"]):
                continue                
            if loc[argument] is None:
                if anisotropic:
                    print(f"{argument} undefined in initialize_params!")
                    exception = True
                elif not "aniso" in argument:
                    print(f"{argument} undefined in initialize_params!")
                    exception = True
        if exception:
            raise Exception("Undefined parameters")
            
            
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.xgridspacing = xgridspacing
        self.ygridspacing = ygridspacing
        self.wavelength_dependent_gridspacing = wavelength_dependent_gridspacing
           
        self.projection = projection
        
        self.interpolation_type = interpolation_type
        
        self.misfit_norm = misfit_norm
        
        self.outlier_model = model_outliers
        
        self.init_no_points = init_no_points
        self.init_vel_points = init_vel_points
        
        self.target_iterations = number_of_iterations
        self.nburnin = number_of_burnin_samples
        
        self.delayed_rejection = delayed_rejection
        
        self.update_paths_interval = update_path_interval
        self.period = period
        self.meanvel = meanvel
        self.kernel_type = kernel_type
        
        if fixed_propvel_std:
            self.propvelstd = 'fixed'
        else:
            self.propvelstd = 'adaptive'
        self.propvelstd_pointupdate = propvelstd_velocity_update
        self.propvelstd_dimchange = propvelstd_birth_death
        self.propmovestd = propmove_std
        self.propsigmastd = propsigma_std
        self.data_std_type = data_std_type
        
        self.temperature = temperature
                
        self.anisotropic = anisotropic
        self.propstd_anisoamp = propstd_anisoamp
        self.propstd_anisodir = propstd_anisodir
        
        self.collect_step = collect_step
        
        self.store_models = store_models
        
        self.print_stats_step = print_stats_step
        
        self.logfile_path = logfile_path
        
        self.visualize = visualize
        self.visualize_step = visualize_step



class initialize_prior(object):
    
    def __init__(self, min_velocity=None, max_velocity=None,
                 min_no_points=None, max_no_points=None,
                 min_datastd=None, max_datastd=None,
                 aniso_ampmin=None, aniso_ampmax=None):
        
        self.velmin = min_velocity
        self.velmax = max_velocity
        
        self.min_datastd = min_datastd
        self.max_datastd = max_datastd
        
        if min_no_points < 3:
            print("WARNING: minimum number of points has been set to 3 for the triangulation to work.")
            min_no_points = 3
        self.nmin_points = min_no_points
        self.nmax_points = max_no_points
        
        self.aniso_ampmin = aniso_ampmin
        self.aniso_ampmax = aniso_ampmax
        


class initialize_data(object):
    
    def __init__(self,sourcexy,receiverxy,ttime,std):
        
        self.sources = sourcexy
        self.receivers = receiverxy
        self.ttimes = ttime
        self.datastd = std
        



""" Class for an MPI parallelized search """
# works also if only a single process is started
class parallel_search(object):
    
    
    def __init__(self, output_location, parallel_tempering, adapt_temp_profile,
                 nchains, update_path_interval, refine_fmm_grid_factor, 
                 kernel_type, anisotropic,store_paths, savetofile_step, mpi_comm):
        
        self.output_location = output_location
        self.parallel_tempering = parallel_tempering
        self.chainlist = []
        self.no_chains = nchains        
        self.no_cores = mpi_comm.Get_size()
        self.update_path_interval = update_path_interval
        if np.size(update_path_interval) > 1:
            raise Exception("The paths should be updated at the same time for all chains! (Only single value for 'update_path_interval')")
        self.refine_fmm_grid_factor = refine_fmm_grid_factor
        self.kernel_type = kernel_type
        self.anisotropic = anisotropic
        self.store_paths = store_paths
        self.savetofile_step = savetofile_step
                
        if parallel_tempering:
            
            self.accepted_swaps = 0
            self.rejected_swaps = 1
            self.loglikelihoods = np.zeros(nchains)
            self.temperatures = np.zeros(nchains)
            self.accepted_swaps_classic = 0
            self.rejected_swaps_classic = 1
            self.adapt_temperature_profile = adapt_temp_profile
        
        return
           
    
    def setup_model_search(self, input_list, mpi_rank, mpi_size, mpi_comm,
                           resume = False):
          
        A_matrix = None
        #if self.anisotropic:
        PHI_matrix = None
        
        for rank in range(self.no_cores):
            if mpi_rank == rank:
                if os.path.isfile(os.path.join(self.output_location,"matrices","A_matrix.pgz")):
                    with gzip.open(os.path.join(self.output_location,
                                                "matrices","A_matrix.pgz"), "rb") as f:
                        A_matrix = pickle.load(f)
                    #if self.anisotropic:
                    if os.path.isfile(os.path.join(self.output_location,
                                                "matrices","PHI_matrix.pgz")):
                        with gzip.open(os.path.join(self.output_location,
                                                "matrices","PHI_matrix.pgz"), "rb") as f:
                            PHI_matrix = pickle.load(f)
                    else:
                        PHI_matrix = None
            mpi_comm.Barrier()
                                    
        #A_matrix = mpi_comm.bcast(A_matrix, root=0)
        #if self.anisotropic:
        #PHI_matrix = mpi_comm.bcast(PHI_matrix, root=0)
       
        
        if resume:
    
            if A_matrix is None:
                raise Exception("could not read matrix file in ",os.path.isfile(os.path.join(self.output_location,"matrices","A_matrix.pgz")))
            # maybe this should also be done sequentially...
            for i,(chain_no,dummy,prior,params) in enumerate(input_list[mpi_rank::mpi_size]):
                
                with gzip.open(os.path.join(self.output_location,"rjmcmc_chain%d.pgz" %(chain_no)), "rb") as f:
                    chain = pickle.load(f)
                chain.A = A_matrix
                #if self.anisotropic:
                chain.PHI = PHI_matrix

                chain.resume(prior,params)

                self.chainlist.append(chain)
   
        else:

            for i,(chain_no,data,prior,params) in enumerate(input_list[mpi_rank::mpi_size]):
   
                chain = RJMC(chain_no, data, prior, params)
                self.chainlist.append(chain)
                    
            if A_matrix is not None:
                for dataset in chain.datasets:
                    if A_matrix[dataset].shape != (len(chain.data[dataset]),len(chain.gridpoints)):
                        print("Matrix shape mismatch - recalculating matrices.")
                        A_matrix = None
                        PHI_matrix = None
                        break

            # parallel computation of paths and matrix creation
            self.update_matrices(mpi_rank, mpi_comm,A=A_matrix,PHI=PHI_matrix,
                                 initialize=True)
                    
        # self.chain_dict = {}
        # for i,chain in enumerate(self.chainlist):
        #     self.chain_dict[chain.chain_no] = (mpi_rank,i)
        # chain_dict = mpi_comm.gather(self.chain_dict, root=0)
        # if mpi_rank==0:
        #     self.chain_dict = {}
        #     for entry in chain_dict:
        #         self.chain_dict.update(entry)
        # self.chain_dict = mpi_comm.bcast(self.chain_dict, root=0)
        
        temps = []
        for chain in self.chainlist:
            temps.append(chain.temperature)
        temps = mpi_comm.gather(temps,root=0)
        if mpi_rank==0:
            temps = np.hstack(temps)
            self.temperatures = np.unique(temps)
            print("temperature levels:",self.temperatures)
            self.swap_acceptance_rate = []
            for temp in self.temperatures:
                self.swap_acceptance_rate.append(np.zeros(100,dtype=int))
                self.swap_acceptance_rate[-1][::2] = 1
            self.temp1swaps = 0
            self.swap_dir = []
            for i in range(len(temps)):
                if temps[i] == np.min(self.temperatures):
                    self.swap_dir.append("up")
                elif temps[i] == np.max(self.temperatures):
                    self.swap_dir.append("down")
                else:
                    self.swap_dir.append(np.random.choice(['up','down']))
                #if temps[chain_no]==np.min(self.temperatures):
                #    self.swap_dir[chain_no+1] = 'up'
                #elif temps[chain_no]==np.max(self.temperatures):
                #    self.swap_dir[chain_no+1] = 'down'
                #else:
                #    self.swap_dir[chain_no+1] = np.random.choice(['up','down'])
            self.swap_dir = np.array(self.swap_dir)
        return
        

        
        
    def submit_jobs(self,mpi_rank,mpi_comm):
            
        chain_iterations = []
        chain_target_iterations = []
        nburnins = []
            
        chain_numbers = []
        for chain in self.chainlist:
            chain_iterations.append(chain.total_steps)
            chain_target_iterations.append(chain.target_iterations)
            nburnins.append(chain.nburnin)
            chain_numbers.append(chain.chain_no)

        nburnins = np.array(nburnins)

        #if np.min(chain_iterations) != np.max(chain_iterations):
        #    raise Exception("MPI process",mpi_rank,"is working on",len(self.chainlist),"chains that have not the same number of starting iterations. This may cause unwanted behavior and is therefore aborted")

        print("MPI Process",mpi_rank,"working on chains:",chain_numbers,flush=True)

        # make sure that the update_path_interval parameter is identical for all chains
        path_updates = None
        path_updates = mpi_comm.gather(self.update_path_interval,root=0)
        path_updates = mpi_comm.bcast(path_updates,root=0)
        if np.min(path_updates) != np.max(path_updates):
            raise Exception("Paths have to be updated at the same interval for all chains. Other options are currently not implemented. Aborting.")
            

        # communicate the minimum number of burnin steps
        min_nburnin = None
        min_nburnin = mpi_comm.gather(np.min(nburnins),root=0)
        min_nburnin = mpi_comm.bcast(np.min(min_nburnin),root=0)
        
        # communicate the maximum number of iterations to all processes
        max_iterations = None
        max_iterations = mpi_comm.gather(np.max(chain_target_iterations),root=0)
        max_iterations = mpi_comm.bcast(np.max(max_iterations),root=0)
        #start_iterations = None
        #start_iterations = mpi_comm.gather(np.min(chain_iterations),root=0)
        #start_iterations = mpi_comm.bcast(np.min(start_iterations),root=0)
        start_iterations = np.min(chain_iterations)
        
        for i in range(start_iterations,max_iterations):
                                        
            for j,chain in enumerate(self.chainlist):
                
                if chain.total_steps < chain.target_iterations:
                    
                    acceptance,loglikelihood = chain.propose_jump()   
                    #self.loglikelihoods[chain.chain_no-1] = loglikelihood 

                    if chain.total_steps%self.savetofile_step == 0:
                        
                        dumpchain = copy.copy(chain)
                        dumpchain.A = []
                        dumpchain.PHI = []
                        dumpchain = pickle.dumps(dumpchain)
                        dumpchain = gzip.compress(dumpchain)
                        time.sleep(mpi_rank/10.) # so that not all chains write to the disk at the same time
                        with open(os.path.join(self.output_location,"dumpchain%d.pgz" %(chain.chain_no)), "wb") as f:
                            f.write(dumpchain)
                        dumpchain = []
                        os.rename(os.path.join(self.output_location,"dumpchain%d.pgz" %(chain.chain_no)),
                                  os.path.join(self.output_location,"rjmcmc_chain%d.pgz" %(chain.chain_no)))
                        #chain.plot(saveplot=True, output_location=self.output_location,suffix="_%d" %(chain.total_steps))
                        print("    Chain %d: iteration %d/%d" %(chain.chain_no,chain.total_steps,chain.target_iterations),flush=True)  

            # Recalculate Eikonal paths
            if i%self.update_path_interval == 0 and i>0:
                # recalculate paths/kernels and update matrices
                self.update_matrices(mpi_rank,mpi_comm)
                        
            # parallel tempering, start after half the number of burnin steps
            # if the chains have different number of burnin steps, take the smallest number
            # currently every 10th step, maybe better every single step?
            if self.parallel_tempering and i%10==0:# and i >= min_nburnin/4.:

                self.swap_temperatures(mpi_rank,mpi_comm,i)
                if i%10000==0:
                    if mpi_rank==0:
                        print("\nIteration: %d\naverage swap acceptance rate = %.2f" %(i,self.accepted_swaps/(self.rejected_swaps+self.accepted_swaps)*100))
                        #print("swap acceptance rate classic = %.2f" %(self.accepted_swaps_classic/(self.rejected_swaps_classic+self.accepted_swaps_classic)*100))
                        print("Temperature swaps of maximum likelihood chains:",self.temp1swaps)
                        for ti in range(len(self.swap_acceptance_rate))[::int(np.ceil(len(self.swap_acceptance_rate)/20))]:
                            print("    T = %.3f" %self.temperatures[ti],np.sum(self.swap_acceptance_rate[ti]))
                    self.accepted_swaps = 0
                    self.rejected_swaps = 1
                    self.accepted_swaps_classic = 0
                    self.rejected_swaps_classic = 1
            # average models from all chains to obtain faster convergence
            #self.mix_model_step = 2500005
            #if i%self.mix_model_step == 0 and i>0:
            #    self.mix_models(mpi_rank, mpi_comm)
            # probably not recommendable, as it will limit the parameter space
            # that is being explored...?
                
    
        for chain in self.chainlist:
            dumpchain = copy.deepcopy(chain)
            dumpchain.para.data_azimuths = None
            dumpchain.para.data_idx = None
            dumpchain.A = []
            #dumpchain.Acsr = []
            dumpchain.PHI = []
            #dumpchain.PHIcsr = []
            #dumpchain.Aprime = []
            #dumpchain.matrix_ind = []
            dumpchain = pickle.dumps(dumpchain)
            dumpchain = gzip.compress(dumpchain)
            time.sleep(mpi_rank/10.) # so that not all chains write to the disk at the same time
            with open(os.path.join(self.output_location,"dumpchain%d.pgz" %(chain.chain_no)), "wb") as f:
                f.write(dumpchain)
            dumpchain = []
            os.rename(os.path.join(self.output_location,"dumpchain%d.pgz" %(chain.chain_no)),
                      os.path.join(self.output_location,"rjmcmc_chain%d.pgz" %(chain.chain_no)))
            ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
    


    def update_matrices(self,mpi_rank,mpi_comm,A=None,PHI=None,
                        initialize=False):
        
        if mpi_rank==0:
            if self.no_cores==1:
                print("\n(Re)calculating paths and updating matrices.")
            else:
                print("\n(Re)calculating paths on %d parallel cores and updating matrices." %self.no_cores)
                
        if A is None or PHI is None:
            if self.kernel_type == "rays":
                if initialize:
                    A,PHI = self.calculate_paths(mpi_rank, mpi_comm, 
                                                 rays='straight')
                else:
                    A,PHI = self.calculate_paths(mpi_rank, mpi_comm, 
                                                 rays='fmm')
            elif self.kernel_type == "fat_rays":
                if initialize:
                    A,PHI = self.calculate_fat_rays(mpi_rank, mpi_comm, 
                                                    kernel_type='constant')
                else:
                    A,PHI = self.calculate_fat_rays(mpi_rank, mpi_comm, 
                                                    kernel_type='empirical')
            elif self.kernel_type == "sens_kernels":
                if initialize:
                    A,PHI = self.calculate_sensitivity_kernels(
                        mpi_rank, mpi_comm, model_type='constant')                
                else:
                    A,PHI = self.calculate_sensitivity_kernels(
                        mpi_rank, mpi_comm, model_type='empirical')
            else:
                raise Exception("Did not recognize option",self.kernel_type)
                
            if mpi_rank == 0:
                if not os.path.exists(os.path.join(self.output_location,"matrices")):
                    os.makedirs(os.path.join(self.output_location,"matrices"))
                with gzip.open(os.path.join(
                        self.output_location,"matrices","A_matrix.pgz"), "wb") as f:
                    pickle.dump(A,f)
                with gzip.open(os.path.join(
                        self.output_location,"matrices","PHI_matrix.pgz"), "wb") as f:
                    pickle.dump(PHI,f)
                
        for chain in self.chainlist:
            chain.A = A
            chain.PHI = PHI
            if initialize:
                chain.initialize_model()
            chain.update_chain()
           
            
    def calculate_paths(self,mpi_rank,mpi_comm,rays='straight'):
        
        def running_mean(x, N):
            if N>len(x):
                print("Warning: Length of the array is shorter than the number of samples to apply the running mean. Setting N=%d." %len(x))
                N=len(x)
            if N<=1:
                return x
            if N%2 == 0:
                N+=1
            idx0 = int((N-1)/2)
            runmean = np.zeros(len(x))
            cumsum = np.cumsum(np.insert(x, 0, 0)) 
            runmean[idx0:-idx0] = (cumsum[N:] - cumsum[:-N]) / N
            for i in range(idx0):
                runmean[i] = np.mean(x[:2*i+1])
                runmean[-i-1] = np.mean(x[-2*i-1:])
            return runmean
            
        chain = self.chainlist[0]

        xfine = chain.x
        yfine = chain.y
        if rays == 'fmm':
            global_average_model = self.get_global_average_model(mpi_rank, mpi_comm)
            global_average_model = np.reshape(global_average_model, chain.shape)
                
            if self.refine_fmm_grid_factor>1:            
                func = RectBivariateSpline(chain.x,chain.y,global_average_model.T)
                xfine = np.linspace(chain.minx,chain.maxx,len(chain.x)*self.refine_fmm_grid_factor-1)
                yfine = np.linspace(chain.miny,chain.maxy,len(chain.y)*self.refine_fmm_grid_factor-1)
                global_average_model = func(xfine,yfine).T
                
            if self.anisotropic:
                global_average_psi2,global_average_psi2amp = self.get_global_average_anisotropy(mpi_rank, mpi_comm)
                global_average_psi2 = np.reshape(global_average_psi2, chain.shape)
                global_average_psi2amp = np.reshape(global_average_psi2amp, chain.shape)
                if self.refine_fmm_grid_factor>1:
                    func = RectBivariateSpline(chain.x,chain.y,global_average_psi2amp.T)
                    global_average_psi2amp = func(xfine,yfine).T
                    Xfine,Yfine = np.meshgrid(xfine,yfine)
                    func = RegularGridInterpolator((chain.y,chain.x), global_average_psi2, method='nearest')
                    global_average_psi2 = func((Yfine,Xfine))
            else:
                global_average_psi2amp = global_average_psi2 = None
                    
        else:
            global_average_model = global_average_psi2amp = global_average_psi2 = None

        if rays!='straight':
            savefig_location = os.path.join(self.output_location,"figures_"+rays)
            if mpi_rank==0:
                if not os.path.exists(savefig_location):
                    os.makedirs(savefig_location)
        else:
            savefig_location = None

        path_dictionary = {}
        A_dictionary = {}
        PHI_dictionary = {}
        t0 = time.time()
        for dataset in chain.datasets:
            
            sources = chain.sources[dataset]
            receivers = chain.receivers[dataset]
            # sorting can speed up the calculation
            sortind = np.lexsort((receivers[:,0],sources[:,1],sources[:,0]))
            sources = sources[sortind]
            receivers = receivers[sortind]
            
            startidx = mpi_rank*int(len(chain.data[dataset])/self.no_cores)
            endidx = (mpi_rank+1)*int(len(chain.data[dataset])/self.no_cores)
            if (mpi_rank+1) == self.no_cores:
                endidx += len(chain.data[dataset])%self.no_cores
            
            #print("MPI Process No",mpi_rank+1,"calculating straight paths from",
            #      startidx,"to",endidx,flush=True)
            paths = get_paths(xfine,yfine,sources[startidx:endidx],
                              receivers[startidx:endidx],
                              chain.projection,rays=rays,verbose=False,
                              model=global_average_model,
                              psi2=global_average_psi2,
                              psi2amp=global_average_psi2amp,
                              savefig_location=savefig_location)
                
            all_paths = []
            # gather gives a list ordered by rank (which is important!)
            paths = mpi_comm.gather(paths, root=0)
            
            if mpi_rank == 0:
                for pathlist in paths:
                    all_paths = all_paths + pathlist
                # undo sorting and restore original order
                reverse_sortind = sortind.argsort()
                all_paths = [all_paths[i] for i in reverse_sortind]
                # print("Warning! Smoothing paths")
                # for path in all_paths:
                #     path[:,0] = running_mean(path[:,0],np.min([51,len(path)]))
                #     path[:,1] = running_mean(path[:,1],np.min([51,len(path)]))
                path_dictionary[dataset] = all_paths
                if self.store_paths:
                    if not os.path.exists(os.path.join(self.output_location,"paths")):
                        os.makedirs(os.path.join(self.output_location,"paths"))
                    np.save(os.path.join(self.output_location,"paths","%s_paths_%s.npy" %(rays,dataset)),all_paths)

            all_paths = mpi_comm.bcast(all_paths, root=0)
            path_dictionary[dataset] = all_paths
            
            if mpi_rank == 0:
                print("Finished calculating paths for dataset %s (time = %ds)" %(dataset,int(time.time()-t0)))
                t0 = time.time()
                print("Creating matrices")
                
            if endidx>0:
                paths = {dataset:all_paths[startidx:endidx]}
                if False:
                    print("warning: setting A matrix smoothing")
                    wavelength = 1./chain.meanslow * chain.period
                    smoothing_matrix = create_smoothing_matrix(chain.x, chain.y, smooth_width=wavelength/4.)
                else:
                    smoothing_matrix = None
                A,PHI = create_A_matrix(chain.x,chain.y,paths=paths,smoothing_matrix=smoothing_matrix,verbose=False)
            else:
                A = []
                PHI = []
                
            A_list = []
            # gather gives a list ordered by rank (which is important!)
            A_list = mpi_comm.gather(A, root=0)
            PHI_list = mpi_comm.gather(PHI, root=0)
            
            if mpi_rank == 0:
                A_list = [A_part[dataset] for A_part in A_list if A_part!=[]]
                A_dictionary[dataset] = vstack(A_list).tocsc()
                PHI_list = [PHI_part[dataset] for PHI_part in PHI_list if PHI_part!=[]]
                PHI_dictionary[dataset] = vstack(PHI_list).tocsc()
                print("Finished creating matrices (time = %ds)" %(int(time.time()-t0)))

        A_dictionary = mpi_comm.bcast(A_dictionary, root=0)
        PHI_dictionary = mpi_comm.bcast(PHI_dictionary, root=0)
        
        return A_dictionary,PHI_dictionary            


    def calculate_sensitivity_kernels(self,mpi_rank,mpi_comm,model_type='constant'):
        
        chain = self.chainlist[0]
        A = {}
        PHI = {}

        savefig_location = os.path.join(self.output_location,"figures_sens_kernels")
        if mpi_rank==0:
            if not os.path.exists(savefig_location):
                os.makedirs(savefig_location)
                
        if model_type=='constant':
            c = 1./chain.meanslow
        else:
            c = self.get_global_average_model(mpi_rank, mpi_comm)
            c = np.reshape(c, chain.shape)
            if mpi_rank==0:
                print("    info: smoothing model before empirical kernel calculation")
            c = gaussian_filter(c,4)
            
        minfreq = 1./chain.period
        k = 2*np.pi*minfreq / np.min(c)
        required_gridspacing = 1./np.max(k)
        
        if required_gridspacing < np.max([chain.xgridspacing,chain.ygridspacing]):
            if mpi_rank==0:
                print("Warning: grid spacing is too coarse to generate accurate " +
                      "sensitivity kernels at this period. This will introduce " +
                      "a small error. Consider using a finer "+
                      "grid or a ray approximation instead.")
            xfine = np.linspace(np.min(chain.x),np.max(chain.x),
                                int(np.ceil((chain.x[-1]-chain.x[0])/required_gridspacing)+1))
            yfine = np.linspace(np.min(chain.y),np.max(chain.y),
                                int(np.ceil((chain.y[-1]-chain.y[0])/required_gridspacing)+1))
            if model_type != 'constant':
                func = RectBivariateSpline(chain.x,chain.y,c.T)
                c = func(xfine,yfine).T
        else:
            xfine = chain.x
            yfine = chain.y
             
        X,Y = np.meshgrid(xfine,yfine)
        
        for dataset in chain.datasets:
            
            if mpi_rank==0:
                print("Calculating sensitivity kernels for dataset",dataset)
            
            startidx = mpi_rank*int(len(chain.data[dataset])/self.no_cores)
            endidx = (mpi_rank+1)*int(len(chain.data[dataset])/self.no_cores)
            if (mpi_rank+1) == self.no_cores:
                endidx += len(chain.data[dataset])%self.no_cores
            
            kernels = get_senskernels(chain.sources[dataset][startidx:endidx],
                                      chain.receivers[dataset][startidx:endidx],
                                      chain.x,chain.y,
                                      X,Y,c,1./chain.meanslow,chain.period,
                                      modeltype=model_type,
                                      savefig_location=savefig_location)
            
            # gather gives a list ordered by rank (which is important!)
            #kernels = mpi_comm.gather(kernels, root=0)
            save_npz(os.path.join(self.output_location,"kernel_matrix_%d.npz" %mpi_rank),
                                  kernels.tocoo())
            mpi_comm.Barrier()
            for rank in range(self.no_cores): # force sequential reading
                if mpi_rank == rank:
                    print("mpi rank",mpi_rank,"starting to read..",flush=True)
                    kernels = []
                    for i in range(self.no_cores):
                        kernels.append(load_npz(os.path.join(
                            self.output_location,"kernel_matrix_%d.npz" %i)))
                    print("mpi rank",mpi_rank,"done reading",flush=True)
                mpi_comm.Barrier()
            kernels = vstack(kernels)
            kernels = kernels.tocsc()
            A[dataset] = kernels
                
        mpi_comm.Barrier()
        if mpi_rank==0:
            print("Done calculating sensitivity kernels.")
            for i in range(self.no_cores):
              os.remove(os.path.join(self.output_location,"kernel_matrix_%d.npz" %i))
                    
        #A = mpi_comm.bcast(A, root=0)
        if self.anisotropic:
            print("Warning! PHI matrix for sens kernels needs fix!")
            PHI = copy.deepcopy(A)
            
        return A, PHI
         


    def calculate_fat_rays(self,mpi_rank,mpi_comm,kernel_type='constant'):
               
        chain = self.chainlist[0]
        A = {}
        PHI = {}

        savefig_location = os.path.join(self.output_location,"figures_fat_rays")
        if mpi_rank==0:
            if not os.path.exists(savefig_location):
                os.makedirs(savefig_location)
                   
        if kernel_type=='constant':
            c = np.ones(chain.shape)*1./chain.meanslow
        else:
            c = self.get_global_average_model(mpi_rank, mpi_comm)
            c = np.reshape(c, chain.shape)
                 
        X,Y = np.meshgrid(chain.x,chain.y)
        
        for dataset in chain.datasets:
            
            if mpi_rank==0:
                print("Calculating fat ray kernels for dataset",dataset)
                                                
            #root_mpi = ranks[0]
            startidx = mpi_rank*int(len(chain.data[dataset])/self.no_cores)
            endidx = (mpi_rank+1)*int(len(chain.data[dataset])/self.no_cores)
            if (mpi_rank+1) == self.no_cores:
                endidx += len(chain.data[dataset])%self.no_cores

            A_part, PHI_part = get_fat_rays(
                chain.sources[dataset][startidx:endidx],
                chain.receivers[dataset][startidx:endidx],
                X,Y,c,chain.period,savefig_location=savefig_location)
            
            # gather gives a list ordered by rank (which is important!)
            A_part = mpi_comm.gather(A_part, root=0)
            PHI_part = mpi_comm.gather(PHI_part, root=0)
            
            if mpi_rank==0:
                A_part = vstack(A_part)
                A_part = A_part.tocsc()
                A[dataset] = A_part
                PHI_part = vstack(PHI_part)
                PHI_part = PHI_part.tocsc()
                PHI[dataset] = PHI_part
                                            
        A = mpi_comm.bcast(A, root=0)
        PHI = mpi_comm.bcast(PHI, root=0)
            
        return A, PHI


    def get_global_average_model(self,mpi_rank,mpi_comm):
              
        Nmods = 0
        avg_mod = np.zeros_like(self.chainlist[0].average_model)
        for chain in self.chainlist:
            Nmods += chain.average_model_counter
            avg_mod += chain.average_model

        recvbuf = None
        if mpi_rank == 0:
            recvbuf = np.empty((self.no_cores,len(chain.gridpoints)), dtype=float)
        mpi_comm.Gather(avg_mod, recvbuf, root=0)
            
        no_models = None
        no_models = mpi_comm.gather(Nmods,root=0)
        no_models = np.sum(no_models)

        if mpi_rank == 0:
            global_average_model = np.sum(recvbuf,axis=0)/no_models
        else:
            global_average_model = np.empty(len(chain.gridpoints))
        mpi_comm.Bcast(global_average_model, root=0)
                        
        return global_average_model            
        
        
    def get_global_average_anisotropy(self,mpi_rank,mpi_comm):
        
        avg_aniso = np.zeros_like(self.chainlist[0].average_anisotropy)
        Nmods = 0
        for i,chain in enumerate(self.chainlist):
            avg_aniso += chain.average_anisotropy
            Nmods += chain.average_model_counter_aniso

        recvbuf_aniso = None
        if mpi_rank == 0:
            recvbuf_aniso = np.empty((self.no_cores,len(chain.gridpoints),2), dtype=float)
        mpi_comm.Gather(avg_aniso, recvbuf_aniso, root=0)
            
        no_models_aniso = None
        no_models_aniso = mpi_comm.gather(Nmods,root=0)
        no_models_aniso = np.sum(no_models_aniso)
            

        if mpi_rank == 0:
            average_anisotropy = np.sum(recvbuf_aniso,axis=0)
            global_average_psi2 = 0.5*np.arctan2(average_anisotropy[:,1]/no_models_aniso,
                                                 average_anisotropy[:,0]/no_models_aniso)
            global_average_psi2amp = np.sqrt((average_anisotropy[:,0]/no_models_aniso)**2 +
                                             (average_anisotropy[:,1]/no_models_aniso)**2)
        else:
            global_average_psi2 = np.empty(len(chain.gridpoints))
            global_average_psi2amp = np.empty(len(chain.gridpoints))
            
        mpi_comm.Bcast(global_average_psi2, root=0)
        mpi_comm.Bcast(global_average_psi2amp, root=0)
        
        return global_average_psi2,global_average_psi2amp


    
    def swap_temperatures(self,mpi_rank,mpi_comm,iteration):
        
        #mpi_comm.Barrier()
        #t0 = time.time()
        
        sendbuf = np.zeros((len(self.chainlist),3))
        for i,chain in enumerate(self.chainlist):
            sendbuf[i] = np.array([chain.chain_no,chain.temperature,
                                   chain.loglikelihood_current])

        recvbuf = None
        if mpi_rank == 0:
            recvbuf = np.empty((self.no_chains,3), dtype=float)
        mpi_comm.Gather(sendbuf,recvbuf, root=0)
        
        if mpi_rank==0:
            
            chain_no = recvbuf[:,0].astype(int)
            temps = recvbuf[:,1]
            likes = recvbuf[:,2]
            
            #print("temperatures:",self.temperatures)
            #print("loglikelihoods:",self.loglikelihoods)
                    
            # # VERSION 1: SWAPPING ADJACENT TEMPERATURE LEVELS
            # for ti in np.random.choice(np.arange(len(self.temperatures)-1),len(self.temperatures)-1,replace=False):#range(len(self.temperatures)-1):
                
            #     idx1 = np.random.choice(np.where(temps==self.temperatures[ti])[0])
            #     idx2 = np.random.choice(np.where(temps==self.temperatures[ti+1])[0])
            #     #if self.swap_dir[idx1]!='up' or self.swap_dir[idx2]!='down':
            #     #    continue
                
            #     if (np.log(np.random.rand(1)) <= (likes[idx1]-likes[idx2]) *
            #         (1./temps[idx2]-1./temps[idx1]) ):
                    
            #         #if self.swap_dir[idx1]=='up' and self.swap_dir[idx2]=='down':
            #         temps[idx1],temps[idx2] = temps[idx2],temps[idx1]

            #         self.accepted_swaps += 1
            #         if True:# ti==0 and likes[idx1]==np.max(likes) or ti>0:
            #             self.swap_acceptance_rate[ti][0] = 1
            #             if temps[idx1] == 1. or temps[idx2] == 1.:
            #                 self.temp1swaps += 1
            #     else:
            #         if True:#ti==0 and likes[idx1]==np.max(likes) or ti>0:
            #             self.rejected_swaps += 1
            #             self.swap_acceptance_rate[ti][0] = 0

            #     self.swap_acceptance_rate[ti] = np.roll(self.swap_acceptance_rate[ti],1)
            
            # self.swap_dir[temps==np.min(self.temperatures)] = 'up'
            # self.swap_dir[temps==np.max(self.temperatures)] = 'down'
            
            # swap_acceptance_rate is defined for swap between i and i+1,
            # therefore undefined for the last temperature level
            #self.swap_acceptance_rate[-1] = self.swap_acceptance_rate[-2]                     


            # # VERSION 1.5: SWAPPING ALMOST RANDOM TEMPERATURE LEVELS (Ray 2021)
            # for ti in np.arange(len(self.temperatures)-1,0,-1):
                
            #     idx1 = np.random.choice(np.where(temps==self.temperatures[ti])[0])
                
            #     ti2 = np.random.randint(0,ti+1)
            #     idx2 = np.random.choice(np.where(temps==self.temperatures[ti2])[0])
                
            #     if temps[idx1]==temps[idx2]:
            #         continue
                
            #     if np.log(np.random.rand(1)) <= ((likes[idx1]-likes[idx2]) *
            #                                       (1./temps[idx2]-1./temps[idx1])):
                    
            #         temps[idx1],temps[idx2] = temps[idx2],temps[idx1]

            #         self.accepted_swaps += 1
            #         if ti==0 and likes[idx1]==np.max(likes) or ti>0:
            #             self.swap_acceptance_rate[ti][0] = 1
            #             self.swap_acceptance_rate[ti2][0] = 1
            #             if temps[idx1] == 1. or temps[idx2] == 1.:
            #                 self.temp1swaps += 1
            #     else:
            #         if ti==0 and likes[idx1]==np.max(likes) or ti>0:
            #             self.rejected_swaps += 1
            #             self.swap_acceptance_rate[ti][0] = 0
            #             self.swap_acceptance_rate[ti2][0] = 0

            #     self.swap_acceptance_rate[ti] = np.roll(self.swap_acceptance_rate[ti],1)
            #     self.swap_acceptance_rate[ti2] = np.roll(self.swap_acceptance_rate[ti2],1)

            
            # VERSION 2: SWAPPING RANDOM TEMPERATURE LEVELS
            chain_indices = np.arange(len(temps))
            np.random.shuffle(chain_indices)
            nchains2 = int(len(chain_indices)/2.)
            for pair in chain_indices[:nchains2*2].reshape(nchains2,2):
                
                if temps[pair[0]] == temps[pair[1]]:
                    continue
                
                idx1 = np.where(temps[pair[0]]==self.temperatures)[0][0]
                idx2 = np.where(temps[pair[1]]==self.temperatures)[0][0]
            
                if np.log(np.random.rand(1)) <= ((likes[pair[1]]-likes[pair[0]]) *
                                                  (1./temps[pair[0]]-1./temps[pair[1]])):
                    
                    self.accepted_swaps += 1
                    temps[pair[0]],temps[pair[1]] = temps[pair[1]],temps[pair[0]]
                    self.swap_acceptance_rate[idx1][0] = 1
                    self.swap_acceptance_rate[idx2][0] = 1
                    # additionally, count accepted swaps for the chain with the highest likelihood
                    if ((temps[pair[0]]==1 and likes[pair[0]]==np.max(likes)) or 
                        (temps[pair[1]]==1 and likes[pair[1]]==np.max(likes))):
                        self.temp1swaps += 1

                else:
                    self.rejected_swaps += 1
                    self.swap_acceptance_rate[idx1][0] = 0
                    self.swap_acceptance_rate[idx2][0] = 0
                            
                self.swap_acceptance_rate[idx1] = np.roll(self.swap_acceptance_rate[idx1],1)
                self.swap_acceptance_rate[idx2] = np.roll(self.swap_acceptance_rate[idx2],1)
                        
            # OPTIMIZE TEMPERATURE SWAP ACCEPTANCE RATE
            # the temperatures are modified so that we get an acceptance rate
            # of 15 - 30% for temperature swaps.
            #(is this working? if we lower the temperatures, the likelihood
            #for a swap is increased, however, the chains will not be any
            #more explorative which may lead to all chains being trapped in 
            #local minima without any swaps)
            if iteration>1000 and np.random.uniform()<0.01 and self.adapt_temperature_profile: # and iteration%10000000000000000==0:
                                
                # TEST 1
                # logtempdiffs = np.log(np.diff(self.temperatures))[:-1]
                # logtempdiffs += -0.1*(100000/(iteration+100000))*np.diff(
                #     np.sum(self.swap_acceptance_rate,axis=1)/100.)[:-1]
                # tempdiffs = np.exp(logtempdiffs)
                # tempdiffs[tempdiffs<0.001] = 0.001
                # newtemps = np.cumsum(tempdiffs)
                # newtemps[newtemps>=self.temperatures[-1]-2] = (
                #     np.linspace(self.temperatures[-1]-2,
                #                 self.temperatures[-1]-1.1,
                #                 np.sum(newtemps>=self.temperatures[-1]-2)))
                # newtemps = np.hstack((self.temperatures[0],
                #                       self.temperatures[0]+newtemps,
                #                       self.temperatures[-1]))                
                # sortidx = temps.argsort()
                # reverse_sortidx = sortidx.argsort()
                # temps = temps[sortidx]
                # temps[temps>1.] = newtemps[1:]
                # temps = temps[reverse_sortidx]
                # if not np.array_equal(newtemps,np.unique(newtemps)):
                #     raise Exception("here")
                # self.temperatures = newtemps
            
                # TEST 2
                # always maintain a log-spaced temperature profile
                # acceptance rate for temperature=1 swaps
                acc_rate_1 = np.sum(self.swap_acceptance_rate[0])
                if acc_rate_1 < 10. or acc_rate_1 > 30.:
                    sortidx = temps.argsort()
                    reverse_sortidx = sortidx.argsort()
                    temps = temps[sortidx]
                    ntemps = len(temps[temps>1.])
                    # leave the upper limit untouched
                    tmax = self.temperatures[-2]
                    if acc_rate_1 < 10.:
                        factor = 0.95
                        # the upper boundary shall not drop below 1.1
                        new_tmax = np.max([tmax*factor,1.1])
                    else:
                        factor = 1.05
                        new_tmax = np.min([tmax*factor,self.temperatures[-1]*0.94])
                    newtemps = np.around(np.logspace(np.log10(1),np.log10(new_tmax),num=ntemps),4)
                    temps[temps>1.] = np.append(newtemps[1:],self.temperatures[-1])
                    temps = temps[reverse_sortidx]
                    self.temperatures = np.unique(temps)

        #self.temperatures = mpi_comm.bcast(self.temperatures, root=0)            
        # this should also in priniple maintain the order
        # temps = mpi_comm.scatter(temps,root=0)
        
        if mpi_rank == 0:
            swaplist = np.column_stack((chain_no,temps))
        else:
            swaplist = None
            #swaplist = np.zeros((self.no_chains,2))
        swaplist = mpi_comm.bcast(swaplist,root=0)
        for chain in self.chainlist:
            #print(chain.temperature)
            chain.temperature = swaplist[swaplist[:,0]==chain.chain_no,1][0]
        
        return
        
    

        """
        # This is slower on my 4 core PC, but maybe faster for more parallel
        # processes?
        
        mpi_comm.Barrier()
        t0 = time.time()
        pairs = None
        probs = None
        if mpi_rank == 0:
            chain_indices = np.arange(self.no_chains)+1
            np.random.shuffle(chain_indices)
            pairs = chain_indices[:int(len(chain_indices)/2.)*2].reshape(int(len(chain_indices)/2),2)
            probs = np.random.rand(len(pairs))
        pairs,probs = mpi_comm.bcast((pairs,probs),root=0)
        
        for i,pair in enumerate(pairs):
            mpi_comm.Barrier()
            #if mpi_rank==0:
            #    print("\n#######\n",flush=True)
            #time.sleep(2)
            
            if self.chain_dict[pair[0]][0] == mpi_rank:
                partner_rank = self.chain_dict[pair[1]][0]
                list_idx = self.chain_dict[pair[0]][1]
            elif self.chain_dict[pair[1]][0] == mpi_rank:
                partner_rank =  self.chain_dict[pair[0]][0]
                list_idx = self.chain_dict[pair[1]][1]
            else:
                continue
            
            if partner_rank == mpi_rank:
            
                chain1 = self.chainlist[self.chain_dict[pair[0]][1]]
                chain2 = self.chainlist[self.chain_dict[pair[1]][1]]
                
                if chain1.temperature == chain2.temperature:
                    continue
                
                loglike11 = chain1.loglikelihood_current
                loglike12,_ = chain1.loglikelihood(
                    chain1.data,chain2.model_predictions,chain1.data_std,chain1.foutlier)
                loglike22 = chain2.loglikelihood_current
                loglike21,_ = chain2.loglikelihood(
                    chain2.data,chain1.model_predictions,chain2.data_std,chain2.foutlier)
                
                log_acc_prob = ((loglike12/chain1.temperature - loglike11/chain1.temperature) + 
                                (loglike21/chain2.temperature - loglike22/chain2.temperature))
            
                log_acc_prob_classic = ((loglike11 - loglike22) * (1./chain2.temperature - 1./chain1.temperature))

                if np.log(probs[i]) <= log_acc_prob_classic:
                    
                    t1 = chain1.temperature
                    #chain1.temperature = chain2.temperature
                    #chain2.temperature = t1
                    self.accepted_swaps_classic += 1
                    
                else:
                    self.rejected_swaps_classic += 1            
            
                if np.log(probs[i]) <= log_acc_prob:
                    
                    t1 = chain1.temperature
                    chain1.temperature = chain2.temperature
                    chain2.temperature = t1
                    self.accepted_swaps += 1
                    
                else:
                    self.rejected_swaps += 1
                    
            else:
                
                chain = self.chainlist[list_idx]
                model_predictions_partner = mpi_comm.sendrecv(chain.model_predictions, partner_rank)
                
                loglike11 = chain.loglikelihood_current
                loglike12,_ = chain.loglikelihood(
                    chain.data,model_predictions_partner,chain.data_std,chain.foutlier)
                
                loglike22, loglike21, partner_temp = mpi_comm.sendrecv(
                    (loglike11,loglike12,chain.temperature), partner_rank)
                
                if chain.temperature == partner_temp:
                    continue
                
                #print("hello from rank",mpi_rank,"with partner",partner_rank,"ll11 =",loglike11,"ll12 =",loglike12,"ll22 =",loglike22,"ll21 =",loglike21)
                
                log_acc_prob = ((loglike12/chain.temperature - loglike11/chain.temperature) + 
                                (loglike21/partner_temp - loglike22/partner_temp))
                
                log_acc_prob_classic = ((loglike11 - loglike22) * (1./partner_temp - 1./chain.temperature))
                
                if np.log(probs[i]) <= log_acc_prob_classic:
                    self.accepted_swaps_classic += 1
                    #chain.temperature = partner_temp
                else:
                    self.rejected_swaps_classic += 1
                
                accept = False
                if np.log(probs[i]) <= log_acc_prob:
                    
                    chain.temperature = partner_temp
                    accept = True
                    self.accepted_swaps += 1
                    
                else:
                    self.rejected_swaps += 1
                
                partner_accept = mpi_comm.sendrecv(accept, partner_rank)
                if partner_accept != accept:
                    raise Exception("acceptance of partner is not identical to own acceptance.")
                
        mpi_comm.Barrier()
        # if mpi_rank==0:
        #     print(time.time()-t0)
        #     print("")
        
        return
        """

    
 
    # def swap_parameterization(self,mpi_rank,mpi_comm):
 
    #     if self.chain.interpolation_type != 'nearest_neighbor':
    #         print("Warning: swap parameterization is not properly implemented for interpolation methods other than nearest neighbor (Voronoi cells).")
               
    #     pairs = None
    #     if mpi_rank == 0:
    #         chain_indices = np.arange(self.no_chains)
    #         np.random.shuffle(chain_indices)
    #         pairs = chain_indices[:int(len(chain_indices)/2.)*2].reshape(int(len(chain_indices)/2),2)
    #     pairs = mpi_comm.bcast(pairs,root=0)
        
    #     pair_idx,swap_idx = np.where(pairs==mpi_rank)
        
    #     if len(pair_idx)>0:
            
    #         pair = pairs[pair_idx][0]
    #         if swap_idx == 0:
    #             partner_rank = pair[1]
    #         else:
    #             partner_rank = pair[0]

            
    #         self.chain.points = mpi_comm.sendrecv(self.chain.points,partner_rank)
    #         self.chain.gpts_idx = mpi_comm.sendrecv(self.chain.gpts_idx,partner_rank)
    #         self.chain.tri = mpi_comm.sendrecv(self.chain.tri,partner_rank)
            
    #         self.chain.prop_model_slowness = np.ones_like(self.chain.model_slowness)
                                               
    #         for i in range(len(self.chain.gpts_idx)):
    #             if len(self.chain.gpts_idx[i])>0:
    #                 mean_slow = np.mean(self.chain.model_slowness[self.chain.gpts_idx[i]])
    #                 self.chain.prop_model_slowness[self.chain.gpts_idx[i]] = mean_slow
    #                 self.chain.points[i,2] = 1./mean_slow
                
    #         self.chain.update_current_likelihood(slowfield = self.chain.prop_model_slowness)
            
    #     mpi_comm.Barrier()
                

    # # this is exactly the same as the swap_parameterization function just
    # # a bit faster
    # def swap_model(self,mpi_rank,mpi_comm):

    #     if self.chain.interpolation_type != 'nearest_neighbor':
    #         print("Warning: swap model is not properly implemented for interpolation methods other than nearest neighbor (Voronoi cells).")

    #     pairs = None
    #     if mpi_rank == 0:
    #         chain_indices = np.arange(self.no_chains)
    #         np.random.shuffle(chain_indices)
    #         pairs = chain_indices[:int(len(chain_indices)/2.)*2].reshape(int(len(chain_indices)/2),2)
    #     pairs = mpi_comm.bcast(pairs,root=0)
        
    #     pair_idx,swap_idx = np.where(pairs==mpi_rank)
        
    #     if len(pair_idx)>0:
            
    #         pair = pairs[pair_idx][0]
    #         if swap_idx == 0:
    #             partner_rank = pair[1]
    #         else:
    #             partner_rank = pair[0]
                
    #         for i in range(len(self.chain.gpts_idx)):
    #             if len(self.chain.gpts_idx[i])>0:
    #                 mean_slow = np.mean(self.chain.prop_model_slowness[self.chain.gpts_idx[i]])
    #                 self.chain.model_slowness[self.chain.gpts_idx[i]] = mean_slow
    #                 self.chain.points[i,2] = 1./mean_slow

    #         if np.isnan(self.chain.points[:,2]).any():
    #             raise Exception("MPI rank",mpi_rank,"has a nan in point distribution after model swap and updating")                
                
    #         self.chain.update_current_likelihood()
            
    #     mpi_comm.Barrier()
        
        

    def plot(self, chainno=None, saveplot=False):

        if chainno is None:
            
            for chain in self.chainlist:
                
                chain.plot(saveplot = saveplot,
                           output_location = self.output_location)

        else:
            
            if np.shape(chainno) == ():
                chainno = [chainno]
            
            for chain in self.chainlist:
                
                if chain.chain_no in chainno:
                    
                    chain.plot(saveplot = saveplot,
                               output_location = self.output_location)
            
            
            

######################
# # FUNCTIONS

def check_list(variable,index,no_searches):
    
    if np.shape(variable) == ():
        return variable
    elif len(variable) == no_searches:
        return variable[index]
    elif len(variable) < no_searches:
        if index >= len(variable):  
            print("Warning! More model searches than assigned parameters. "
                  "The same parameter as for model search number 1 will be assigned.")
            return variable[0]
        else:
            return variable[index]
    elif len(variable) > no_searches:
        print("Warning! Less model searches (%d) than assigned parameters (%d). " %(no_searches,len(variable)))
        print("All additional parameters are being ignored.",variable)
        return variable[index]
    else:
        print(variable)
        print(len(variable))
        print(no_searches)
        print(index)
        raise Exception("Unknown error in the check_list function in rjtransdim_helper_classes.")



# Calculate a single straight path (on a regular, equidistant grid)
def calc_straight_ray_path(source,receiver,stepsize,projection):
    """

    Parameters
    ----------
    source : TYPE
        DESCRIPTION.
    receiver : TYPE
        DESCRIPTION.
    stepsize : TYPE
        DESCRIPTION.
    projection : TYPE
        DESCRIPTION.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    if source[0]==receiver[0] and source[1]==receiver[1]:
        raise Exception("Source and receiver position are identical.")
    
    #stepsize = np.min([self.xgridspacing,self.ygridspacing])/3.
    distance = np.sqrt((source[0]-receiver[0])**2 + (source[1]-receiver[1])**2)
    if distance <= stepsize*2.:
        stepsize = distance/2.
        
    g = pyproj.Geod(ellps='WGS84')
    p = pyproj.Proj(projection)
    lon1,lat1 = p(source[0]*1000,source[1]*1000,inverse=True)
    lon2,lat2 = p(receiver[0]*1000,receiver[1]*1000,inverse=True)
    lonlats = g.npts(lon1,lat1,lon2,lat2,int(np.ceil(distance/stepsize)))
    lonlats = np.vstack(((lon1,lat1),lonlats,(lon2,lat2)))

    x_regular = np.interp(np.linspace(0,1,int(distance/stepsize)),[0,1],[source[0],receiver[0]])
    y_regular = np.interp(np.linspace(0,1,int(distance/stepsize)),[0,1],[source[1],receiver[1]])
    
    x_regular,y_regular = p(lonlats[:,0],lonlats[:,1])
    x_regular /= 1000.
    y_regular /= 1000.

    return np.column_stack((x_regular,y_regular))



# Calculate a single Eikonal path between a source and a receiver
def calc_eikonal_path(x,y,model,source,receiver,psi2=None,psi2amp=None,fmm_out=None):
    """
    x: 1-d array x-axis
    y: 1-d array y-axis
    model: 2-d array velocity model
    source: tuple of source coordinates (x,y)
    receiver: tuple of receiver coordinates (x,y)
    fmm_out: output from a previous traveltime field calculation (travel-
             time field is identical if only the receiver position changes)
    """
    
    if fmm_out is not None:
        xnew,ynew,ttimefield = fmm_out
    elif psi2 is None and psi2amp is None:
        xnew,ynew,ttimefield = FMM.calculate_ttime_field(x,y,model,source)
        if type(ttimefield) == type(None):
            raise Exception("Could not calculate travel time field!")
    else:
        ttimefield = FMM.calculate_ttime_field_aniso(x,y,model,psi2,psi2amp,source)
        xnew = x
        ynew = y
        
    path = FMM.shoot_ray(xnew,ynew,ttimefield,source,receiver)[0] # FMM.shoot_ray returns a list      
    
    return path,(xnew,ynew,ttimefield)     


# Calculate paths for all source-receiver combinations
def get_paths(x,y,sources,receivers,projection,rays='straight',model=None,
              psi2=None,psi2amp=None,
              return_ttimes=False,verbose=False,
              savefig_location=None):
    """
    it is assumed that sources has the same length as receivers. Each
    source-receiver couple represents one travel-time measurement
    """
        
    paths = []
    ttimes = []
    
    t0 = time.time()
    if rays=='straight':
        stepsize = np.min([np.min(np.abs(np.diff(x))),
                           np.min(np.abs(np.diff(y)))])/3.
        if return_ttimes:
            raise Exception("return_traveltimes works only in combination with the FMM option.")
        if verbose:
            print("Calculating straight rays")
        for i in range(len(sources)):
            paths.append(calc_straight_ray_path(sources[i],receivers[i],
                                                stepsize,projection))
    
    elif rays=='fmm':
        if x is None or y is None or model is None:
            raise Exception("Please define x,y and the model for the FMM calculations.")
        if verbose:
            print("Calculating Eikonal rays with the Fast Marching Method.")
        for i in range(len(sources)):
            if verbose:
                if i%1000 == 0:
                    calc_time = time.time()-t0
                    print("  Eikonal path: %d/%d (calc time: %ds)" %(i,len(sources),calc_time))            
                        
            if i==0 or (sources[i]!=sources[i-1]).any():
                fmm_out = None # if the source location changes, we need to recalculate the ttime field
                ttime_interpolation_func = None
            path,fmm_out = calc_eikonal_path(x,y,model,sources[i],receivers[i],
                                             psi2=psi2,psi2amp=psi2amp,fmm_out=fmm_out)
            if return_ttimes:
                if ttime_interpolation_func is None:
                    ttimefunc = RectBivariateSpline(fmm_out[0],fmm_out[1],fmm_out[2].T)
                ttimes.append(ttimefunc(receivers[i][0],receivers[i][1]))
                
            paths.append(path)
            
            if i%1000==0 and savefig_location is not None:
                plt.ioff()
                fig = plt.figure(figsize=(7,6))
                ax1 = fig.add_subplot(111)
                X,Y = np.meshgrid(x,y)
                cmap = ax1.pcolormesh(x,y,model,cmap=cmcram.roma,shading='nearest')
                ax1.plot(sources[i][0],sources[i][1],'gv',label='source')
                ax1.plot(receivers[i][0],receivers[i][1],'rv',label='receiver')
                ax1.plot(path[:,0],path[:,1],'k--',linewidth=0.3)
                plt.legend()
                plt.colorbar(cmap,shrink=0.5)
                ax1.set_aspect('equal')
                figname = os.path.join(savefig_location,
                                       "img_%.3f_%.3f_%.3f_%.3f.png" 
                    %(sources[i][0],sources[i][1],receivers[i][0],receivers[i][1]))
                plt.savefig(figname)
                plt.close(fig)
    
    # elif rays=='fat_rays':
    #     if x is None or y is None or model is None:
    #         raise Exception("Please define x,y and the model for the FMM calculations.")
    #     if verbose:
    #         print("Calculating fat rays with the Fast Marching Method.")
    #     # calculating after sorting sources can be faster because we don't
    #     # have to re-calculate the travel time field if the source remains the same
    #     sortind = np.lexsort((receivers[:,0],sources[:,1],sources[:,0]))
    #     sources = sources[sortind]
    #     receivers = receivers[sortind]
    #     for i in range(len(sources)):
    #         if verbose:
    #             if i%1000 == 0:
    #                 calc_time = time.time()-t0
    #                 print("  path %d/%d (calc time: %ds)\n" %(i,len(sources),calc_time))            
                        
    #         if i==0 or (sources[i]!=sources[i-1]).any():
    #             fmm_out = None # if the source location changes, we need to recalculate the ttime field
    #             ttime_interpolation_func = None
    #         path,fmm_out = calc_eikonal_path(x,y,model,sources[i],receivers[i],fmm_out=fmm_out)
            
    
    else:
        raise Exception("Undefined ray type for path calculation! Choose between 'straight' and 'fmm'. Current value: %s" %(rays))

    if verbose:
        print("  Calculating paths took: %ds\n" %(time.time()-t0))          
    

    if return_ttimes:
        return paths,ttimes
    else:
        return paths
    
    
    
def get_senskernels(sources,receivers,xmod,ymod,X,Y,vel_model,meanvel,period,
                    modeltype='constant',savefig_location=None):
    """

    Parameters
    ----------
    sources : TYPE
        DESCRIPTION.
    receivers : TYPE
        DESCRIPTION.
    xmod : TYPE
        x axis of the model (regularly spaced).
    ymod : TYPE
        y axis of the model (regularly spaced.
    X : TYPE
        X grid for the calculation of the sensitivity kernels (can coincide
        with the model grid).
    Y : TYPE
        Y grid for the calculation of the sensitivity kernels (can coincide
        with the model grid).
    vel_model : TYPE
        DESCRIPTION.
    meanvel : TYPE
        DESCRIPTION.
    period : TYPE
        DESCRIPTION.
    modeltype : TYPE, optional
        DESCRIPTION. The default is 'constant'.
    savefig_location : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    kernels : TYPE
        DESCRIPTION.

    """
    
    kernels = lil_matrix((len(sources),len(xmod)*len(ymod)),dtype='float32')
    
    Xmod,Ymod = np.meshgrid(xmod,ymod)
    xgridspacing = np.abs(np.diff(xmod))[0]
    ygridspacing = np.abs(np.diff(ymod))[0]
    
    iprint = int(len(sources)/10.)
    iprint = np.max([iprint,1])
    
    print("Warning! Setting maxorbits to 2.6!")
    maxorbits = 2.6 # standard: 5
    
    for i in range(len(sources)):
        
        #if i%iprint==0 and self.chain_no==1:
        #    print(i,"/",len(sources))
        
        source = sources[i]
        receiver = receivers[i]
        distance = np.sqrt(np.sum((source-receiver)**2))
        t0 = distance*1./meanvel
        
        kernel_a = FMM.analytical_sensitivity_kernel(X,Y,source,receiver,
                                                     meanvel,1./period,
                                                     maxorbits=maxorbits)
        
        if X.shape != Xmod.shape or Y.shape != Ymod.shape:
            func = RegularGridInterpolator((X[0],Y[:,0]),kernel_a.T)
            kernel_a = func((Xmod,Ymod))
            
        kernel_a *= xgridspacing * ygridspacing * t0 / meanvel
        
        if modeltype == 'constant':
            
            hybrid_kernel = kernel_a
            path = np.array([[source[0],source[1]],
                             [receiver[0],receiver[1]]])
            
        else:

            ttimefield_src = FMM.calculate_ttime_field_samegrid(
                X,Y,vel_model,source,refine_source_grid=True)
            path = FMM.shoot_ray(X[0],Y[:,0],ttimefield_src,source,receiver)[0]
            ttimefield_rcv = FMM.calculate_ttime_field_samegrid(
                X,Y,vel_model,receiver,refine_source_grid=True)
        
            kernel_e,ttime_direct = FMM.empirical_sensitivity_kernel(
                X,Y,source,receiver,ttimefield_src,ttimefield_rcv,
                1./period,synthetic=True,maxorbits=maxorbits)
                            
            kernel_e *= xgridspacing * ygridspacing * ttime_direct / vel_model
            
            if X.shape != Xmod.shape or Y.shape != Ymod.shape:
                func = RegularGridInterpolator((X[0],Y[:,0]),kernel_e.T)
                kernel_e = func((Xmod,Ymod))
                    
            hybrid_kernel = 0.5 * (kernel_e + kernel_a)
            
        hybrid_kernel = hybrid_kernel.flatten()
        hybrid_kernel[np.abs(hybrid_kernel)<0.01*np.max(np.abs(hybrid_kernel))] = 0.
        
        if i%iprint==0 and savefig_location is not None:
            plt.ioff()
            fig = plt.figure(figsize=(14,6))
            ax1 = fig.add_subplot(121)
            if vel_model.shape!=Xmod.shape:
                vel_model = np.ones_like(X)*vel_model
            cmap = ax1.pcolormesh(X,Y,vel_model,cmap=cmcram.roma,shading='nearest')
            ax1.plot(source[0],source[1],'gv',label='source')
            ax1.plot(receiver[0],receiver[1],'rv',label='receiver')
            ax1.plot(path[:,0],path[:,1],'k--',linewidth=0.3)
            plt.legend()
            plt.colorbar(cmap,shrink=0.5)
            ax1.set_aspect('equal')
            ax2 = fig.add_subplot(122)
            vmax = np.max(np.abs(hybrid_kernel))
            cmap = ax2.pcolormesh(Xmod,Ymod,hybrid_kernel.reshape(Xmod.shape),
                                  vmin=-0.5*vmax,vmax=0.5*vmax,cmap=plt.cm.PuOr,
                                  shading='nearest')
            ax2.plot(source[0],source[1],'gv',label='source')
            ax2.plot(receiver[0],receiver[1],'rv',label='receiver')
            ax2.plot(path[:,0],path[:,1],'k--',linewidth=0.3)
            plt.legend()
            plt.colorbar(cmap,shrink=0.5)
            ax2.set_aspect('equal')
            figname = os.path.join(savefig_location,
                "img_%.3f_%.3f_%.3f_%.3f.png" 
                %(source[0],source[1],receiver[0],receiver[1]))
            plt.savefig(figname)
            plt.close(fig)
        
        kernels[i,np.abs(hybrid_kernel)>0.] = hybrid_kernel[np.abs(hybrid_kernel)>0.]
        
        
    return kernels



def get_fat_rays(sources,receivers,X,Y,vel_model,period,
                 savefig_location = None):
    # Simplified version of the first orbit of a sensitivity kernel
    
    A = lil_matrix((len(sources),len(X.flatten())),dtype='float32')
    PHI = lil_matrix(A.shape,dtype='float32')
    iprint = int(len(sources)/10)
    
    for i in range(len(sources)):
        
        if i%iprint==0:
            print(i,"/",len(sources))
        
        source = sources[i]
        receiver = receivers[i]
        
        kernel,phi_kernel,path = FMM.fat_ray_kernel(
            X,Y,source,receiver,vel_model,1./period)

        kernel = kernel.flatten()
        phi_kernel = phi_kernel.flatten()
        phi_kernel[phi_kernel==0.] = 0.01 # avoid accidental zeros
        if (kernel==0.).all():
            raise Exception()        
        if (kernel<0.).any():
            raise Exception()
        
        A[i,kernel>0.] = kernel[kernel>0.]
        PHI[i,kernel>0.] = phi_kernel[kernel>0.]
        
        if i%iprint==0 and savefig_location is not None:
            plt.ioff()
            fig = plt.figure(figsize=(14,6))
            ax1 = fig.add_subplot(121)
            cmap = ax1.pcolormesh(X,Y,vel_model,cmap=cmcram.roma,shading='nearest')
            ax1.plot(source[0],source[1],'gv',label='source')
            ax1.plot(receiver[0],receiver[1],'rv',label='receiver')
            ax1.plot(path[:,0],path[:,1],'k--',linewidth=0.3)
            plt.legend()
            plt.colorbar(cmap,shrink=0.5)
            ax1.set_aspect('equal')
            ax2 = fig.add_subplot(122)
            cmap = ax2.pcolormesh(X,Y,kernel.reshape(X.shape),
                                  cmap=plt.cm.magma_r,shading='nearest')
            # q = ax2.quiver(X,Y,1,1,angles=phi_kernel.reshape(X.shape)/np.pi*180.,
            #     headwidth=0,headlength=0,headaxislength=0,
            #     pivot='middle',scale=70.,width=0.005,
            #     color='yellow',edgecolor='k',#scale=80.,width=0.0035,
            #     linewidth=0.5)
            ax2.plot(source[0],source[1],'gv',label='source')
            ax2.plot(receiver[0],receiver[1],'rv',label='receiver')
            ax2.plot(path[:,0],path[:,1],'k--',linewidth=0.3)
            plt.legend()
            plt.colorbar(cmap,shrink=0.5)
            ax2.set_aspect('equal')
            figname = os.path.join(savefig_location,
                "img_%.3f_%.3f_%.3f_%.3f.png" 
                %(source[0],source[1],receiver[0],receiver[1]))
            plt.savefig(figname)
            plt.close(fig)

    return A, PHI


def trunc_normal(x, mu, sig, sig_trunc=3):
    
    if sig==0.:
        sig = 1e-10
    gauss = (1. / (np.sqrt(2 * np.pi) * sig) * np.exp(
                -0.5 * np.square(x - mu) / np.square(sig)))

    bounds = (mu-sig_trunc*sig,mu+sig_trunc*sig)
    gauss[x < bounds[0]] = 0
    gauss[x > bounds[1]] = 0

    return gauss/np.sum(gauss)
    
def create_smoothing_matrix(x,y,smooth_width=25):
    
    model_size = len(x)*len(y)
    smoothing_matrix = lil_matrix((model_size,model_size),dtype='float32')
    print("Trying to apply gaussian model smoothing with %.1fkm standard deviation." %smooth_width)
    X,Y = np.meshgrid(x,y)
    gridpoints = np.column_stack((X.flatten(),Y.flatten()))
    for idx in range(model_size):
        dists = np.sqrt(np.sum((gridpoints[idx]-gridpoints)**2,axis=1))
        weights = trunc_normal(dists,0,smooth_width,sig_trunc=2)
        # remove elements with very small influence to reduce the matrix size
        weights[weights<0.01*weights.max()] = 0.
        # normalize to 1
        weights /= weights.sum()
        smoothing_matrix[idx] = weights
        
    return smoothing_matrix

def create_A_matrix(x,y,paths,smoothing_matrix=None,verbose=False):
    
    # from shapely.geometry import Polygon
    # import matplotlib.patches as patches
    # wavelength = 3.2*15
    # print("Warning, A matrix with fat rays at 15s period!")
    
    if verbose:
        print("filling A matrix...")
    
    A = {}
    PHI = {}
    
    # cells = []
    # corners = []
    # for gp in self.gridpoints:
    #     cell = [(gp[0]-self.xgridspacing/2.,gp[1]+self.ygridspacing/2.),
    #             (gp[0]+self.xgridspacing/2.,gp[1]+self.ygridspacing/2.),
    #             (gp[0]+self.xgridspacing/2.,gp[1]-self.ygridspacing/2.),
    #             (gp[0]-self.xgridspacing/2.,gp[1]-self.ygridspacing/2.)]
    #     corners.append(cell)
    #     cells.append(Polygon(cell))
        
    # gridtree = cKDTree(self.gridpoints)
       
    # for dataset in paths:
        
    #     # initializing sparse matrix
    #     self.A[dataset] = lil_matrix((len(self.data[dataset]),len(self.gridpoints)))
    #     PHI = lil_matrix(self.A[dataset].shape)
 
    #     if len(paths[dataset]) != len(self.data[dataset]):
    #         raise Exception("Number of paths and number of data points is not identical! Cannot create A matrix.")

    #     for i in range(len(self.data[dataset])):
            
    #         if i%1000==0:
    #             print(i,"/",len(self.data[dataset]))
                
    #         x = paths[dataset][i][:,0]
    #         y = paths[dataset][i][:,1]
                                
    #         pathdist = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    #         steps = np.linspace(0,1,int(pathdist/np.min([self.xgridspacing/4.,self.ygridspacing/4.])))
    #         x_reg = np.interp(steps,np.linspace(0,1,len(x)),x)
    #         y_reg = np.interp(steps,np.linspace(0,1,len(y)),y)
            
    #         k=16
    #         distance_upper_bound = np.max([wavelength/7.,np.sqrt(self.xgridspacing**2+self.ygridspacing**2)])
    #         nndist,nnidx = gridtree.query(np.column_stack((x_reg,y_reg)),k=k,distance_upper_bound=distance_upper_bound)
    #         while not np.isinf(nndist[:,-1]).all():
    #             k*=2
    #             nndist,nnidx = gridtree.query(np.column_stack((x_reg,y_reg)),k=k,distance_upper_bound=distance_upper_bound)
    #             if k>=len(self.gridpoints):
    #                 break
    #         idx = np.unique(nnidx)[:-1] # last index is for the inf points
                   
    #         # normal vectors
    #         norm = np.arctan2(np.gradient(y_reg),np.gradient(x_reg)) + np.pi/2.
            
    #         parallel1 = np.column_stack((x_reg+wavelength/8*np.cos(norm),
    #                                      y_reg+wavelength/8*np.sin(norm)))
    #         parallel2 = np.column_stack((x_reg-wavelength/8*np.cos(norm),
    #                                      y_reg-wavelength/8*np.sin(norm)))
            
    #         # shapely polygon
    #         polyshape = np.vstack((parallel1,parallel2[::-1]))
    #         poly = Polygon(polyshape)
            
    #         weights = np.zeros(len(self.gridpoints))
    #         idx_valid = []
    #         for grididx in idx:
    #             area = poly.intersection(cells[grididx]).area
    #             if area>0.:
    #                 weights[grididx] = area
    #                 idx_valid.append(grididx)
                
    #         idx = idx_valid
    #         weights /= np.sum(weights)
    #         weights *= pathdist
                    
                    
    #         self.A[dataset][i,idx] = weights[idx]

    #         # plt.figure()
    #         # patch = patches.Polygon(polyshape,alpha=0.6)
    #         # for xg in self.xgrid:
    #         #     plt.plot([xg,xg],[np.min(self.ygrid),np.max(self.ygrid)],'k')
    #         # for yg in self.ygrid:
    #         #     plt.plot([np.min(self.xgrid),np.max(self.xgrid)],[yg,yg],'k')
    #         # plt.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.',ms=2)
    #         # plt.plot(paths[dataset][i][:,0],paths[dataset][i][:,1])
    #         # plt.plot(parallel1[:,0],parallel1[:,1],'--')
    #         # plt.plot(parallel2[:,0],parallel2[:,1],'--')
    #         # plt.plot(self.gridpoints[idx][:,0],self.gridpoints[idx][:,1],'g.')
    #         # for i in idx:
    #         #     plt.text(self.gridpoints[i,0],self.gridpoints[i,1],"%.3f" %weights[i])
    #         # plt.gca().add_patch(patch)
    #         # plt.gca().set_aspect('equal')
    #         # plt.show()    
    #         # pause
    
    if not (np.allclose(np.std(np.diff(x)),0.) and np.allclose(np.std(np.diff(x)),0.)):
        raise Exception("x and y need to be on a regular axis.")
        
    xgridspacing = np.diff(x)[0]
    ygridspacing = np.diff(y)[0]
    
    # gridlines (lines between gridpoints, delimiting cells) 
    xgrid = np.append(x-xgridspacing/2.,x[-1]+xgridspacing/2.)
    ygrid = np.append(y-ygridspacing/2.,y[-1]+ygridspacing/2.)
        
    for dataset in paths:
    
        # initializing sparse matrix
        A[dataset] = lil_matrix((len(paths[dataset]),len(x)*len(y)),dtype='float32')
        PHI[dataset] = lil_matrix(A[dataset].shape,dtype='float32')
 
        # it can cause problems if the start or end points of the path are 
        # exactly on the grid lines. Points are then moved by this threshold
        xthresh = xgridspacing/1000.
        ythresh = ygridspacing/1000.

        for i in range(len(paths[dataset])):
                        
            xpath = paths[dataset][i][:,0]
            ypath = paths[dataset][i][:,1]
            
            # x-points directly on one of the xgridlines:
            idx_x = np.where(np.abs(xgrid[:, None] - xpath[None, :]) < xthresh)[1]
            xpath[idx_x] += xthresh
            # y-points directly on one of the ygridlines:
            idx_y = np.where(np.abs(ygrid[:, None] - ypath[None, :]) < ythresh)[1]
            ypath[idx_y] += ythresh            
            
            t = np.arange(len(xpath))
            
            idx_grid_x, idx_t = np.where((xgrid[:, None] - xpath[None, :-1]) * (xgrid[:, None] - xpath[None, 1:]) <= 0)
            tx = idx_t + (xgrid[idx_grid_x] - xpath[idx_t]) / (xpath[idx_t+1] - xpath[idx_t])
                
            idx_grid_y, idx_t = np.where((ygrid[:, None] - ypath[None, :-1]) * (ygrid[:, None] - ypath[None, 1:]) <= 0)
            ty = idx_t + (ygrid[idx_grid_y] - ypath[idx_t]) / (ypath[idx_t+1] - ypath[idx_t])
                            
            t2 = np.sort(np.r_[t, tx, tx, ty, ty])
            
            # if the path crosses exactly a grid crossing, it creates a double
            # point which can cause problems
            t2unique,idxunique,cntunique = np.unique(np.around(t2,5),
                                                     return_index=True,
                                                     return_counts=True)
            idx_double = idxunique[np.where(cntunique==4)[0]]
            t2 = np.delete(t2,np.hstack((idx_double,idx_double+1)))
            
            x2 = np.interp(t2, t, xpath)
            y2 = np.interp(t2, t, ypath)
            
            dists = np.sqrt( np.diff(x2)**2 + np.diff(y2)**2 )
  
            # loc gives the x2-/y2-indices where the path is crossing a gridline
            loc = np.where(np.diff(t2) == 0)[0] + 1     
            
            pnts = np.column_stack((x2[np.hstack((0,loc,-1))],
                                    y2[np.hstack((0,loc,-1))]))
            
            # the angles from np.arctan lie always in the range [-pi/2,+pi/2]
            phi = np.arctan(np.diff(pnts[:,1])/(np.diff(pnts[:,0])+1e-16))
            # the azimuthal coverage check will not work with arctan2!
            # arctan2 determines the correct quadrant, angles are in [-pi,pi]
            #phi = np.arctan2(np.diff(pnts[:,1]),np.diff(pnts[:,0]))
            
            midpoints = pnts[:-1]+np.diff(pnts,axis=0)/2.
       
            xind = np.abs(midpoints[:,0][None,:]-x[:,None]).argmin(axis=0)
            yind = np.abs(midpoints[:,1][None,:]-y[:,None]).argmin(axis=0)
            
            # idx gives the nearest neighbor index of the self.gridpoints array
            idx = yind*len(x)+xind
                
            distlist = np.split(dists, loc)
            d = np.array(list(map(np.sum,distlist)))
            if np.sum(d) == 0:
                print(i,[xpath,ypath])
                raise Exception("a zero distance path is not valid!")
            weights = d.copy()
       
            # plt.figure()
            # for xg in self.xgrid:
            #     plt.plot([xg,xg],[np.min(self.ygrid),np.max(self.ygrid)],'k')
            # for yg in self.ygrid:
            #     plt.plot([np.min(self.xgrid),np.max(self.xgrid)],[yg,yg],'k')
            # plt.plot(self.X.flatten(),self.Y.flatten(),'k.',ms=2)
            # plt.plot(x2[loc],y2[loc],'o')
            # plt.plot(midpoints[:,0],midpoints[:,1],'.')
            # plt.plot(self.gridpoints[idx][:,0],self.gridpoints[idx][:,1],'g.')
            # plt.plot([x2[0],x2[-1]],[y2[0],y2[-1]],'r.')
            # for mp in range(len(midpoints)):
            #     plt.text(midpoints[mp,0],midpoints[mp,1],"%.3f" %d[mp])
            # plt.gca().set_aspect('equal')
            # plt.show()    
            # pause
       
            # sometimes a path can traverse a grid cell twice
            idxunique, sortidx, cntunique= np.unique(idx,return_counts=True,return_index=True)
            if (cntunique>1).any():
                for doubleidx in np.where(cntunique>1)[0]:
                    weights[idx==idxunique[doubleidx]] = np.sum(weights[idx==idxunique[doubleidx]])
                    # taking the mean angle
                    phi[idx==idxunique[doubleidx]] = np.arctan2(np.sum(np.sin(phi[idx==idxunique[doubleidx]])),
                                                                np.sum(np.cos(phi[idx==idxunique[doubleidx]])))
   
            # we have to make sure that none of the phi values is exactly
            # zero by coincidence. This would result in different matrix
            # shapes between the A matrix and the PHI matrix
            phi[phi==0.] += 0.001
            if (weights==0.).any():
                print("Warning! One of the weights in the A matrix is zero!")
            A[dataset][i,idx[sortidx]] = weights[sortidx]
            PHI[dataset][i,idx[sortidx]] = phi[sortidx]
        
            if not np.allclose(np.sum(A[dataset][i]),np.sum(d)):
                print(np.sum(A[dataset][i]),np.sum(d))
                raise Exception("matrix filling error")
            if (weights==0.).all():
                raise Exception("weights have to be greater than zero!")
                            
        if verbose:
            print("Matrix filling done")
          
        if smoothing_matrix is not None:
            print("applying matrix model smoothing")
            A[dataset] = A[dataset]*smoothing_matrix
            sources = np.vstack([p[0] for p in paths[dataset]])
            receivers = np.vstack([p[-1] for p in paths[dataset]])
            dx,dy = sources[:,0]-receivers[:,0],sources[:,1]-receivers[:,1]
            meanphi = np.arctan(dy/(dx+1e-10))
            PHI[dataset] = A[dataset].copy()
            nonzero = PHI[dataset].nonzero()
            PHI[dataset].data[:] = meanphi[nonzero[0]]
            
        A[dataset] = A[dataset].tocsc() # much faster calculations. But slower while filling
        # csr would be even a little bit faster with matrix vector product
        # calculations, but csc is much faster when extracting columns which is
        # necessary when updating the model predictions (see calculate_ttimes)
        #if self.anisotropic:
        PHI[dataset] = PHI[dataset].tocsc()
    
    return A,PHI


def plot_matrix(A,gridpoints,indices,sources=None,receivers=None):
    
    if type(indices) == type(1):
        indices = [indices]
        
    x = np.unique(gridpoints[:,0])
    y = np.unique(gridpoints[:,1])
    xgridspacing = np.diff(x)[0]
    ygridspacing = np.diff(y)[0]
    # gridlines (lines between gridpoints, delimiting cells) 
    xgrid = np.append(x-xgridspacing/2.,x[-1]+xgridspacing/2.)
    ygrid = np.append(y-ygridspacing/2.,y[-1]+ygridspacing/2.)
    
    plt.figure()
    for xg in xgrid:
        plt.plot([xg,xg],[np.min(ygrid),np.max(ygrid)],'k')
    for yg in ygrid:
        plt.plot([np.min(xgrid),np.max(xgrid)],[yg,yg],'k')
    plt.plot(gridpoints[:,0],gridpoints[:,1],'k.',ms=2)
    for idx in indices:
        weights = A[idx].toarray()[0]
        path = np.column_stack((gridpoints,weights))
        path = path[weights!=0.] 
        plt.plot(path[:,0],path[:,1],'.')
        if sources is not None:
            plt.plot(sources[idx][0],sources[idx][1],'o')
        if receivers is not None:
            plt.plot(receivers[idx][0],receivers[idx][1],'o')
        if sources is not None and receivers is not None:
            plt.plot([sources[idx][0],receivers[idx][0]],
                     [sources[idx][1],receivers[idx][1]])
        for mp in range(len(path)):
            plt.text(path[mp,0],path[mp,1],"%.3f" %path[mp,2])
    plt.gca().set_aspect('equal')
    plt.show()
 
    
 
def forward_calculation(A,m,m0=None,ttimes0=None,idx_mod=None,
                        kernel_type='rays',anisotropic=False,
                        PHI=None,psi2=None,psi2amp=None,psi20=None,psi2amp0=None):
       
    ttimes = {}
        
    if kernel_type == 'sens_kernels':
        
        if not anisotropic:
            if idx_mod is None:
                for dataset in A:
                    ttimes[dataset] = ttimes0[dataset] + (
                        A[dataset]*(m-m0))
            else:
                for dataset in A:
                    ttimes[dataset] = ttimes0[dataset] + (
                        A[dataset][:,idx_mod] * (m[idx_mod]-m0[idx_mod]))
                    
        else:
            if idx_mod is None:
                for dataset in A:
                    nz = A[dataset].nonzero()
                    # since the matrix is csc (column sorted), we have to sort
                    # the nonzero indices according to their column (nz[1])
                    cidx = np.sort(nz[1])
                    # csc_indices = np.hstack([
                    #     A[dataset].indices[A[dataset].indptr[i]:A[dataset].indptr[i+1]] 
                    #     for i in range(A[dataset].shape[1])])
                    #m_aniso = m[cidx] * (1. - psi2amp[cidx]*np.cos(2*(PHI[dataset].data-psi2[cidx])))
                    #M_aniso = csc_matrix((m_aniso,A[dataset].indices,A[dataset].indptr),shape=A[dataset].shape)
                    dm = m[cidx] * (1. + psi2amp[cidx]*np.cos(2*(PHI[dataset].data-psi2[cidx]))) - m0[cidx]
                    dM = csc_matrix((dm,A[dataset].indices,A[dataset].indptr),shape=A[dataset].shape)
                    ttimes[dataset] = ttimes0[dataset] + np.asarray((A[dataset].multiply(dM)).sum(axis=1).T)
                    # slower:
                    #ttimes = np.bincount(nz[0][nz[1].argsort()],weights = A[dataset].data * dm)
                    
            else:
                for dataset in A:
                    nz = A[dataset][:,idx_mod].nonzero()
                    cidx = np.sort(nz[1])
                    m_aniso = m[idx_mod][cidx] * (
                        1. + psi2amp[idx_mod][cidx] * 
                        np.cos(2*(PHI[dataset][:,idx_mod].data - 
                                  psi2[idx_mod][cidx])))
                    m0_aniso = m0[idx_mod][cidx] * (
                        1. + psi2amp0[idx_mod][cidx] * 
                        np.cos(2*(PHI[dataset][:,idx_mod].data - 
                                  psi20[idx_mod][cidx])))
                    dM = csc_matrix((m_aniso-m0_aniso,
                                     A[dataset][:,idx_mod].indices,
                                     A[dataset][:,idx_mod].indptr),
                                    shape=A[dataset][:,idx_mod].shape)
                    ttimes[dataset] = ttimes0[dataset] + np.asarray((A[dataset][:,idx_mod].multiply(dM)).sum(axis=1).T)
    
    
    else:
        
        if idx_mod is None:
            for dataset in A:
                nz = A[dataset].nonzero()
                cidx = np.sort(nz[1])
                m_aniso = m[cidx] * (1. + psi2amp[cidx]*np.cos(2*(PHI[dataset].data-psi2[cidx])))
                M_aniso = csc_matrix((1./m_aniso,A[dataset].indices,A[dataset].indptr),shape=A[dataset].shape)
                ttimes[dataset] = np.asarray((A[dataset].multiply(M_aniso)).sum(axis=1).T)
            
        else:
            for dataset in A:
                nz = A[dataset][:,idx_mod].nonzero()
                cidx = np.sort(nz[1])
                m_aniso = m[idx_mod][cidx] * (1. - psi2amp[idx_mod][cidx] * 
                                      np.cos(2*(PHI[dataset][:,idx_mod].data - 
                                                psi2[idx_mod][cidx])))
                m0_aniso = m0[idx_mod][cidx] * (1. - psi2amp0[idx_mod][cidx] * 
                                        np.cos(2*(PHI[dataset][:,idx_mod].data - 
                                                  psi20[idx_mod][cidx])))
                dM = csc_matrix((1./m_aniso-1./m0_aniso,
                                     A[dataset][:,idx_mod].indices,
                                     A[dataset][:,idx_mod].indptr),
                                    shape=A[dataset][:,idx_mod].shape)
                ttimes[dataset] = ttimes0[dataset] + np.asarray((A[dataset][:,idx_mod].multiply(dM)).sum(axis=1).T)

                
    return ttimes
    

def plot_avg_model(filepath,projection=None,plotsinglechains=False):
    #%%
    def atoi(text): # used for string sorting
        return int(text) if text.isdigit() else text

    def natural_keys(text): # used for string sorting
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [ atoi(c) for c in re.split(r'(\d+)', text) ]
    
    no_cells = []
    burnin_steps = []
    total_steps = []
    
    meanstds = []
    likelihoods = []
    residuals = []
    average_models = []
    modelsumsquared = []
    average_model_counters = []
    average_model_counters_aniso = []
    point_density = []
    # only used for stationwise errors:
    stationerrors = {}
    stationerrors_stations = {}
    # anisotropic
    anisotropic = False
    aniso_models = []
    psi2amp_sumsquared = []
    aniso_sumsquared = []
    psi2all = []
    psi2ampall = []
    temperatures = []

    #voronoimodels = []
    
    filepaths = glob.glob(os.path.join(filepath,"*.pgz"))
    filepaths.sort(key=natural_keys)
    for i,chainpath in enumerate(filepaths):
        
        print("reading chain",chainpath)
        with gzip.open(chainpath, "rb") as f:
            chain = pickle.load(f) 
        
        print("    chain number:",chain.chain_no,"temperature:",chain.temperature)
        
        if plotsinglechains:
            chain.plot(saveplot=True,output_location=filepath)
            
        if i == 0:
            X,Y = np.meshgrid(chain.x,chain.y)
            stations = []
            for dataset in chain.datasets:
                stations.append(np.vstack((chain.sources[dataset],chain.receivers[dataset])))
            stations = np.unique(np.vstack(stations),axis=0)
            datasets = chain.datasets
            if projection==None:
                projection = pyproj.Proj(chain.projection)
            
        average_models.append(chain.average_model)
        modelsumsquared.append(chain.modelsumsquared)
        #voronoimodels.append([vmod[0] for vmod in chain.collection_points])
        if i>0:
            if np.shape(chain.gridpoints[:,0]) != np.shape(average_models[0]):
                raise Exception("Chains have different parameterization. Calculating the average model in this case is currently not implemented.")
        average_model_counters.append(chain.average_model_counter)
        print("    models contributed to the average:",chain.average_model_counter)
        if chain.anisotropic:
            average_model_counters_aniso.append(chain.average_model_counter_aniso)
            aniso_models.append(chain.average_anisotropy)
            psi2amp_sumsquared.append(chain.psi2amp_sumsquared)
            aniso_sumsquared.append(chain.anisotropy_sumsquared)
            anisotropic = True
            try:
                psi2all += chain.collection_psi2
                psi2ampall += chain.collection_psi2amp
            except:
                pass
        no_cells.append(chain.collection_no_points)
        meanstds.append(chain.collection_datastd)
        likelihoods.append(chain.collection_loglikelihood)
        residuals.append(chain.collection_residuals)
        total_steps.append(chain.total_steps)
        burnin_steps.append(chain.nburnin)
        temperatures.append(chain.collection_temperatures)
        if chain.total_steps >= chain.nburnin:
            try:
                point_density.append(chain.point_density)
            except:
                print("could not get point density")
        if 'stationerrors' in chain.error_type:
            for dataset in datasets:
                if not dataset in stationerrors.keys():
                    stationerrors[dataset] = []
                    stationerrors_stations[dataset] = chain.stations[dataset]
                stationerrors[dataset].append(chain.stationerrors[dataset])
                if not np.array_equal(stationerrors_stations[dataset],chain.stations[dataset]):
                    print("ERROR! station errors are not correctly associated")                       

    no_chains = i+1
    
    print_prec = np.min([np.min(np.diff(X[0])),np.min(np.diff(Y[:,0]))])/1000.
    if print_prec > 1.:
        print_prec = 1 # only one decimal
    else:
        for ifloat,decimal in enumerate(str(print_prec).split(".")[-1]):
            if decimal != "0":
                print_prec = ifloat+1
                break

    for subselection in [100]:  
        #print("Taking a subselection of %d percent of the best models" %subselection)
        if no_chains > 1:
            # sort out the bad chains
            store_intervals = np.array(total_steps)/(np.array(list(map(len,likelihoods)))-1)
            if (np.abs(np.around(store_intervals) - store_intervals) > 0.).any():
                print("Warning: length of likelihoods array not correct!")
            store_intervals = store_intervals.astype(int)
            likelihoodmeans = np.array([np.mean(like[int(burnin_steps[k]/store_intervals[k]):]) for k,like in enumerate(likelihoods)])#np.mean(likelihoods,axis=1)
            sortidx = likelihoodmeans.argsort()[::-1]
            good_chains = sortidx[:int(np.around(subselection/100.*len(sortidx)))]
        else:
            store_intervals = [int(total_steps[0]/(len(likelihoods[0])-1))]
            good_chains = np.array([0])
            
        #good_chains = np.where(likelihoodmeans>=np.median(likelihoodmeans)-1*np.std(likelihoodmeans))[0]
        #good_chains = np.random.choice(good_chains,int(subselection/100*len(good_chains)),replace=False)
        
        print("keeping",len(good_chains),"chains of a total of",no_chains)
        print("keeping:",np.sort(good_chains))
        
        #keep only good chains
        Nmodels = np.sum(np.array(average_model_counters)[good_chains])
        avg_model = np.sum(np.array(average_models)[good_chains],axis=0) / Nmodels
        if len(stationerrors) > 0:
            for dataset in datasets:
                stationerrors[dataset] = np.array(stationerrors[dataset])[good_chains]
        # average model variance sum(model-avg_model)**2 = sum( model**2 - 2model*avg_model + avg_model**2)
        # model std = sqrt( variance / N)        
        modeluncertainties = np.sqrt((np.sum(np.array(modelsumsquared)[good_chains],axis=0) - 
                              2*avg_model * np.sum(np.array(average_models)[good_chains],axis=0) +
                              Nmodels * avg_model**2) / Nmodels)       
        
        modeluncertainties = modeluncertainties.reshape(np.shape(X))
        avg_model = avg_model.reshape(np.shape(X))
        
        if len(point_density) == no_chains:
            nucleii_density = np.sum(np.array(point_density)[good_chains],axis=0) / Nmodels
            nucleii_density_smooth = gaussian_filter(nucleii_density,2)

        # not necessary anymore
        # t0 = time.time()  
        # i=0     
        # reduce_factor = 1
        # X_low,Y_low = X[::reduce_factor,::reduce_factor],Y[::reduce_factor,::reduce_factor]
        # avg_model_low = avg_model[::reduce_factor,::reduce_factor]
        # gridpoints = np.column_stack((X_low.flatten(),Y_low.flatten()))
        # varsum = np.zeros(np.shape(X_low))
        # for good_chain_idx in good_chains:
        #     vnoicollection = voronoimodels[good_chain_idx]
        #     for vnoimod in vnoicollection:
        #         # interpolate voronoicells on the grid
        #         prop_voronoi_kdtree = cKDTree(vnoimod[:,:2])
        #         test_point_dist, test_point_regions = prop_voronoi_kdtree.query(gridpoints)
        #         model = np.reshape(vnoimod[test_point_regions,2], np.shape(X_low))
        #         varsum += (model-avg_model_low)**2
        #         i+=1
        # modeluncertainties = np.sqrt(varsum / (i-1))
        # print("getting modelstd took",time.time()-t0)
        
     
        LON,LAT = projection(X*1000.,Y*1000., inverse=True)
        statlon,statlat = projection(stations[:,0]*1000.,stations[:,1]*1000,inverse=True)
        proj = ccrs.TransverseMercator(central_longitude=np.mean(LON),
                                       central_latitude=np.mean(LAT),
                                       approx=False)
       
        rel_vels = (avg_model-np.mean(avg_model))/np.mean(avg_model)*100
        #dmax = 12
    
        plt.ioff()
    
        fig = plt.figure(figsize=(14,10))
        axm = fig.add_subplot(111,projection=proj)
        cbar = axm.pcolormesh(LON,LAT,avg_model,cmap=cmcram.roma,
                              #vmin=2.8,vmax=3.6,
                              shading='nearest',rasterized=True,transform=ccrs.PlateCarree())
                       #vmin=1.9,vmax=3.9)
                       #rasterized=True,vmin=-dmax,vmax=dmax)
        #cbar = plt.tricontourf(X_plot.flatten(),Y_plot.flatten(),avg_model.flatten(),
        #                       cmap=cm.GMT_haxby_r,levels=50)
        try:
            reader = shpreader.Reader("/home/user/Maps Alpine Chains & Mediterranean/GIS_digitization/tectonic_maps_4dmb/shape_files/faults_alcapadi")
            faults = reader.records()
            lines = []
            linemarkers = []
            linewidths = []
            for fault in faults:
                if fault.attributes["fault_type"] == 1:
                    linemarkers.append('solid')
                    linewidths.append(1.5)
                elif fault.attributes["fault_type"] == 2:
                    linemarkers.append('solid')
                    linewidths.append(1.)
                else:
                    linemarkers.append('dashed')
                    linewidths.append(1.5)
                lines.append(np.array(fault.geometry.coords))
        except:
            lines = []
            linewidths = []
            linemarkers = []
        norm_alpha = plt.Normalize(0,np.max(modeluncertainties),clip=True)
        overlay = np.ones(LON.shape+(4,))
        overlay[:,:,-1] = norm_alpha(modeluncertainties)
        #plt.imshow(overlay,origin='lower',zorder=1,
        #           extent=(np.min(X_plot),np.max(X_plot),np.min(Y_plot),np.max(Y_plot)))  
        axm.coastlines(resolution='50m')
        axm.add_feature(cf.BORDERS.with_scale('50m'))
        axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
        axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
        axm.add_collection(LineCollection(lines,linewidths=linewidths,
                                          linestyles=linemarkers,colors='red',
                                          transform=ccrs.PlateCarree()))
        gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                           linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        colax = fig.add_axes([0.55,0.05,0.25,0.02])
        plt.colorbar(cbar,label='phase velocity [%]',cax=colax,orientation='horizontal')
        #plt.annotate("Rayleigh %.2fs" %period,xycoords='figure fraction',xy=(0.15,0.9),fontsize=16,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.7))
        plt.savefig(os.path.join(filepath,"pv_map_plot.png"),bbox_inches='tight',transparent=True)
        plt.close(fig)
     
        fig = plt.figure(figsize=(14,10))
        axm = fig.add_subplot(projection=proj)
        axm.coastlines(resolution='50m')
        axm.add_feature(cf.BORDERS.with_scale('50m'))
        axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
        axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')   
        cbar = axm.pcolormesh(LON,LAT,modeluncertainties,cmap=plt.cm.hot,
                       shading='nearest',transform=ccrs.PlateCarree())
        axm.plot(statlon,statlat,'gv',ms=1,transform=ccrs.PlateCarree())
        plt.colorbar(cbar,shrink=0.5,label='velocity standard deviation')
        plt.savefig(os.path.join(filepath,"model_std.png"), bbox_inches='tight')
        plt.close(fig)
        
        fig = plt.figure(figsize=(14,10))
        axm = fig.add_subplot(projection=proj)
        cbar = axm.pcolormesh(LON,LAT,modeluncertainties,cmap=plt.cm.hot,
                              shading='nearest',rasterized=True,transform=ccrs.PlateCarree())
        axm.add_collection(LineCollection(lines,linewidths=linewidths,
                                          linestyles=linemarkers,colors='white',
                                          transform=ccrs.PlateCarree()))
        norm_alpha = plt.Normalize(0,np.max(modeluncertainties),clip=True)
        overlay = np.ones(LON.shape+(4,))
        overlay[:,:,-1] = norm_alpha(modeluncertainties)
        axm.coastlines(resolution='50m')
        axm.add_feature(cf.BORDERS.with_scale('50m'))
        axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
        axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
        gl = axm.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                           #xlocs=[6,10,14,18],ylocs=[43,45,47,49],
                           linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        colax = fig.add_axes([0.55,0.05,0.25,0.02])
        plt.colorbar(cbar,label='phase velocity uncertainty',cax=colax,orientation='horizontal')
        #plt.annotate("Rayleigh %.2fs" %period,xycoords='figure fraction',xy=(0.15,0.9),fontsize=16,bbox=dict(boxstyle="square,pad=0.2",fc='white',ec='None', alpha=0.7))
        plt.savefig(os.path.join(filepath,"model_std_map.png"),bbox_inches='tight',transparent=True)
        plt.close(fig)
    
        if len(point_density) > 0:
            fig = plt.figure(figsize=(14,10))
            axm = fig.add_subplot(111,projection=proj)
            axm.coastlines(resolution='50m')
            axm.add_feature(cf.BORDERS.with_scale('50m'))
            axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
            axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
            cbar = axm.pcolormesh(LON,LAT,nucleii_density_smooth,cmap=plt.cm.hot,
                           shading='nearest',transform=ccrs.PlateCarree())
            axm.plot(statlon,statlat,'gv',ms=1,transform=ccrs.PlateCarree())
            plt.colorbar(cbar,shrink=0.5,label='relative point density')
            plt.savefig(os.path.join(filepath,"nucleii_density smoothed.png"), bbox_inches='tight')
            plt.close(fig)
        
        fig = plt.figure(figsize=(14,10))
        axm = fig.add_subplot(projection=proj)
        cbar = axm.pcolormesh(LON,LAT,avg_model,cmap=plt.cm.jet_r,shading='nearest',transform=ccrs.PlateCarree())
        axm.plot(statlon,statlat,'kv',ms=1,transform=ccrs.PlateCarree())
        plt.colorbar(cbar,shrink=0.5,)
        plt.savefig(os.path.join(filepath,"result_average_model_%d_percent.png" %subselection), bbox_inches='tight')
        plt.close(fig)
    
        fig = plt.figure(figsize=(14,10))
        axm = fig.add_subplot(projection=proj)
        cbar = axm.pcolormesh(LON,LAT,(avg_model-np.mean(avg_model))/np.mean(avg_model)*100,
                       cmap=plt.cm.jet_r,shading='nearest',transform=ccrs.PlateCarree())
        axm.plot(statlon,statlat,'kv',ms=1,transform=ccrs.PlateCarree())
        plt.colorbar(cbar,shrink=0.5,)
        plt.savefig(os.path.join(filepath,"result_average_model_relative.png"), bbox_inches='tight')
        plt.close(fig)
            
        # LINE PLOTS
        colors = [mcolors.to_rgba(c)
          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        
        fig = plt.figure(figsize=(10,7))
        plt.title("loglikelihood")
        lines = []
        for li,ll in enumerate(likelihoods):
            if total_steps[li]>burnin_steps[li]:
                idx0 = int(burnin_steps[li]/store_intervals[li]*(3/4.))
            else:
                idx0 = 0
            iterations = np.arange(total_steps[li]+1)
            lines.append(np.column_stack((
                iterations[::store_intervals[li]][idx0:],ll[idx0:])))
            plt.plot([burnin_steps[li],burnin_steps[li]],
                     [np.min(ll[idx0:]),np.max(ll[idx0:])],'k')
        line_segments = LineCollection(lines,colors=colors,linewidths=1,linestyles='solid')
        plt.gca().add_collection(line_segments)
        plt.ylabel("loglikelihood")
        if idx0>0:    
            plt.gca().set_xlim(int(3/4*np.min(burnin_steps)),np.max(total_steps))
        else:
            plt.gca().set_xlim(0,np.max(total_steps))
        plt.savefig(os.path.join(filepath,"result_loglikelihoods.png"), bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10,7))
        plt.title("joint residuals")
        lines = []
        for li,ll in enumerate(residuals):
            if total_steps[li]>burnin_steps[li]:
                idx0 = int(burnin_steps[li]/store_intervals[li]*(3/4.))
            else:
                idx0 = 0
            iterations = np.arange(total_steps[li]+1)
            lines.append(np.column_stack((
                iterations[::store_intervals[li]][idx0:],ll[idx0:])))
            plt.plot([burnin_steps[li],burnin_steps[li]],
                     [np.min(ll[idx0:]),np.max(ll[idx0:])],'k')
        line_segments = LineCollection(lines,colors=colors,linewidths=1,linestyles='solid')
        plt.gca().add_collection(line_segments)
        plt.ylabel("residual")
        if idx0>0:
            plt.gca().set_xlim(int(3/4*np.min(burnin_steps)),np.max(total_steps))
        else:
            plt.gca().set_xlim(0,np.max(total_steps))
        plt.savefig(os.path.join(filepath,"result_residuals.png"), bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(10,7))
        plt.title("number of points")
        lines = []
        for pnts in no_cells:
            iterations = np.arange(total_steps[li]+1)
            lines.append(np.column_stack((
                iterations[::store_intervals[li]],pnts)))
            plt.plot([burnin_steps[li],burnin_steps[li]],
                     [np.min(pnts),np.max(pnts)],'k')
        line_segments = LineCollection(lines,colors=colors,linewidths=1,linestyles='solid')
        plt.gca().add_collection(line_segments)
        plt.gca().set_xlim(0,np.max(total_steps))
        plt.ylabel("no of points")
        plt.savefig(os.path.join(filepath,"result_no_cells_evolution.png"), bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(figsize=(16,12))
        plt.subplot(221)
        plt.title("Temperature evolution")
        lines = []
        for ti,temps in enumerate(temperatures):
            iterations = np.arange(total_steps[ti]+1)
            lines.append(np.column_stack((
                iterations[::store_intervals[ti]],temps)))
        plt.plot([burnin_steps[ti],burnin_steps[ti]],
                 [1,np.max(np.hstack(temperatures))],'k')
        line_segments = LineCollection(lines,colors=colors,linewidths=1,linestyles='solid')
        plt.gca().add_collection(line_segments)
        plt.gca().set_xlim(0,np.max(total_steps))
        plt.ylabel("temperatures")
        plt.xlabel("iteration")
        plt.subplot(222)
        for ti,temps in enumerate(temperatures):
            temps_p = np.unique(temps[int(len(temps)*burnin_steps[ti]/total_steps[ti]):])
            plt.plot(np.ones(len(temps_p))*(ti+1),temps_p,'.')
        plt.yscale('log')
        plt.ylabel("temperature")
        plt.xlabel("chain number")
        plt.subplot(223)
        for ti,temps in enumerate(temperatures):
            plt.plot(ti,np.sum(np.array(temps)==1),'o')
        plt.xlabel("chain number")
        plt.ylabel("no samples temp=1")
        plt.savefig(os.path.join(filepath,"result_temperature_evolution.png"), bbox_inches='tight')
        plt.close(fig)

    
        fig = plt.figure(figsize=(10,7))
        plt.title("no of points")
        binsmin = np.min(np.hstack(no_cells))
        binsmax = np.max(np.hstack(no_cells))
        if binsmax-binsmin > 500:
            bins = np.linspace(binsmin,binsmax,500)
        else:
            bins = np.arange(binsmin,binsmax+1,1) 
        for nc in no_cells:
            plt.hist(nc,bins=bins,alpha=0.5)
        plt.savefig(os.path.join(filepath,"result_no_cells_hist.png"), bbox_inches='tight')
        plt.close(fig)
        writelist = []
        for i,nc in enumerate(no_cells):
            writelist.append(np.column_stack((nc,np.ones(len(nc))*(i+1))))
        np.savetxt(os.path.join(filepath,"result_no_cells_%s.txt" %dataset),
                   np.vstack(writelist), fmt="%.3f %d")
            
        fig = plt.figure(figsize=(7,4*len(datasets)))
        for i,dataset in enumerate(datasets):
            ax = fig.add_subplot(len(datasets),1,i+1)
            ax.set_title(dataset,loc='left')
            for ds in meanstds:
                ax.hist(ds[dataset],bins=30,alpha=0.5)
        #plt.suptitle("data standard deviation")
        plt.savefig(os.path.join(filepath,"result_data_std.png"), bbox_inches='tight')
        plt.close(fig)
        for dataset in datasets:
            writelist = []
            for i,ds in enumerate(meanstds):
                writelist.append(np.column_stack((ds[dataset],np.ones(len(ds[dataset]))*(i+1))))
            np.savetxt(os.path.join(filepath,"result_data_std_%s.txt" %dataset),
                       np.vstack(writelist), fmt="%.3f %d")
        
        if anisotropic:
            aniso_sum = np.sum(np.array(aniso_models)[good_chains],axis=0)/np.sum(np.array(average_model_counters_aniso)[good_chains])
            #print(np.shape(aniso_sum))
            anisotropy_dirmean = 0.5*np.arctan2(aniso_sum[:,1],aniso_sum[:,0])
            anisotropy_ampmean = np.sqrt(aniso_sum[:,0]**2 + aniso_sum[:,1]**2)
            # this is identical to the sqrt(sum((x-xmean)**2 + (y-ymean)**2)/Nmodels)
            # uncertainty defined below
            #psi2uncertainty = np.sqrt((np.sum(np.array(psi2amp_sumsquared)[good_chains],axis=0) - 
            #                            Nmodels * anisotropy_ampmean**2) / Nmodels)
            aniso_sumsquared_sum = np.sum(np.array(aniso_sumsquared)[good_chains],axis=0)
            psi2uncertainty = np.sqrt((aniso_sumsquared_sum[:,0] - Nmodels * aniso_sum[:,0]**2 +
                                       aniso_sumsquared_sum[:,1] - Nmodels * aniso_sum[:,1]**2) / Nmodels)
            xpsi2_mean = aniso_sum[:,0]
            xpsi2_std = np.sqrt((aniso_sumsquared_sum[:,0] - 2*aniso_sum[:,0] * 
                                 np.sum(np.array(aniso_models)[good_chains],axis=0)[:,0] +
                                 Nmodels * aniso_sum[:,0]**2) / Nmodels)
            ypsi2_mean = aniso_sum[:,1]
            ypsi2_std = np.sqrt((aniso_sumsquared_sum[:,1] - 2*aniso_sum[:,1] *
                                 np.sum(np.array(aniso_models)[good_chains],axis=0)[:,1] +
                                 Nmodels * aniso_sum[:,1]**2) / Nmodels)
            
            np.savetxt(os.path.join(filepath,"result_average_model_%d_percent.xyv" %subselection),
                       np.column_stack((X.flatten(),Y.flatten(),LON.flatten(),LAT.flatten(),
                                        avg_model.flatten(),modeluncertainties.flatten(),
                                        100*anisotropy_ampmean,anisotropy_dirmean,100*psi2uncertainty,
                                        xpsi2_mean,xpsi2_std,ypsi2_mean,ypsi2_std)),
                       header='Mean model and model standard deviation\nAverage of %d chains\n' %len(good_chains) +
                              'projection used: %s\n' %projection.definition_string() +
                              'x[km] y[km] lon[deg] lat[deg] v[km/s] std(v)[km/s] PSI2amp[%] PSI2dir[rad] std(PSI2) xPSI2 xstdPSI2 yPSI2 ystdPSI2',
                       fmt="%%.%df %%.%df %%.%df %%.%df %%.5f %%.6f %%.1f %%.3f %%.1f %%.3f %%.4f %%.3f %%.4f" %(print_prec,print_prec,print_prec*3,print_prec*3))
            
        else:
            np.savetxt(os.path.join(filepath,"result_average_model_%d_percent.xyv" %subselection),
                       np.column_stack((X.flatten(),Y.flatten(),LON.flatten(),LAT.flatten(),
                                        avg_model.flatten(),modeluncertainties.flatten())),
                       header='Mean model and model standard deviation\nAverage of %d chains\n' %len(good_chains) +
                              'projection used: %s\n' %projection.definition_string() +
                              'x[km] y[km] lon[deg] lat[deg] v[km/s] std(v)[km/s]',
                       fmt="%%.%df %%.%df %%.%df %%.%df %%.5f %%.6f" %(print_prec,print_prec,print_prec*3,print_prec*3))


        # plotting
        if anisotropic:
            thresh = 0.03
            
            fig = plt.figure(figsize=(18,10))
            ax1 = fig.add_subplot(231,projection=proj)
            ax1.coastlines(resolution='50m')
            ax1.add_feature(cf.BORDERS.with_scale('50m'))
            ax1.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
            ax1.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
            cbar = ax1.pcolormesh(LON,LAT,
                                  100*anisotropy_ampmean.reshape(np.shape(X)),
                                  shading='nearest',transform=ccrs.PlateCarree())
            plt.colorbar(cbar,fraction=0.1,shrink=0.5)
            ax2 = fig.add_subplot(232,projection=proj)
            ax2.coastlines(resolution='50m')
            ax2.add_feature(cf.BORDERS.with_scale('50m'))
            ax2.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
            ax2.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
            threshold = psi2uncertainty.reshape(X.shape)[::2,::2] < thresh
            q = ax2.quiver(LON[::2,::2][threshold],LAT[::2,::2][threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2][threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2][threshold],
                           angles=anisotropy_dirmean.reshape(np.shape(X))[::2,::2][threshold]/np.pi*180,
                           headwidth=0,headlength=0,headaxislength=0,
                           pivot='middle',scale=70.,width=0.005,
                           color='yellow',edgecolor='k',#scale=80.,width=0.0035,
                           linewidth=0.5,transform=ccrs.PlateCarree()) 
            q2 = ax2.quiver(LON[::2,::2][~threshold],LAT[::2,::2][~threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2][~threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2][~threshold],
                           angles=anisotropy_dirmean.reshape(np.shape(X))[::2,::2][~threshold]/np.pi*180,
                           headwidth=0,headlength=0,headaxislength=0,
                           pivot='middle',scale=70.,width=0.005,
                           color='orange',edgecolor='k',#scale=80.,width=0.0035,
                           linewidth=0.5,transform=ccrs.PlateCarree()) 
            ax3 = fig.add_subplot(233,projection=proj)
            ax3.coastlines(resolution='50m')
            ax3.add_feature(cf.BORDERS.with_scale('50m'))
            ax3.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
            ax3.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
            cbar = ax3.pcolormesh(LON,LAT,psi2uncertainty.reshape(X.shape),
                                  cmap=plt.cm.hot,shading='nearest',
                                  transform=ccrs.PlateCarree())
            plt.colorbar(cbar,fraction=0.1,shrink=0.5)          

            ax4 = fig.add_subplot(234,projection=proj)
            ax4.coastlines(resolution='50m')
            ax4.add_feature(cf.BORDERS.with_scale('50m'))
            ax4.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
            ax4.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
            threshold = psi2uncertainty < thresh
            anisotropy_ampmean[~threshold] = np.nan
            cbar = ax4.pcolormesh(LON,LAT,
                                  100*anisotropy_ampmean.reshape(np.shape(X)),
                                  shading='nearest',transform=ccrs.PlateCarree())
            plt.colorbar(cbar,fraction=0.1,shrink=0.5)
            ax5 = fig.add_subplot(235,projection=proj)
            ax5.coastlines(resolution='50m')
            ax5.add_feature(cf.BORDERS.with_scale('50m'))
            ax5.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
            ax5.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
            threshold = psi2uncertainty.reshape(X.shape)[::2,::2] < thresh
            q = ax5.quiver(LON[::2,::2][threshold],LAT[::2,::2][threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2][threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2][threshold],
                           angles=anisotropy_dirmean.reshape(np.shape(X))[::2,::2][threshold]/np.pi*180,
                           headwidth=0,headlength=0,headaxislength=0,
                           pivot='middle',scale=70.,width=0.005,
                           color='yellow',edgecolor='k',#scale=80.,width=0.0035,
                           linewidth=0.5,transform=ccrs.PlateCarree()) 

            ax6 = fig.add_subplot(236,projection=proj)
            ax6.coastlines(resolution='50m')
            ax6.add_feature(cf.BORDERS.with_scale('50m'))
            ax6.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
            ax6.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
            cbar = ax6.pcolormesh(LON,LAT,avg_model,cmap=plt.cm.jet_r,
                                  shading='nearest',transform=ccrs.PlateCarree())
            plt.colorbar(cbar,shrink = 0.5)
            threshold = psi2uncertainty.reshape(X.shape)[::2,::2] < thresh
            q = ax6.quiver(LON[::2,::2][threshold],LAT[::2,::2][threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2][threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::2,::2][threshold],
                           angles=anisotropy_dirmean.reshape(np.shape(X))[::2,::2][threshold]/np.pi*180,
                           headwidth=0,headlength=0,headaxislength=0,
                           pivot='middle',scale=70.,width=0.005,
                           color='yellow',edgecolor='k',#scale=80.,width=0.0035,
                           linewidth=0.5,transform=ccrs.PlateCarree()) 
            qk = ax6.quiverkey(q, X=0.06, Y=0.04, U=3, label='3%',
                               labelpos='E',#edgecolor='w',linewidth=0.5,
                               fontproperties=dict(size=10))
            qk.text.set_path_effects([path_effects.withStroke(linewidth=2,foreground='w')])
            
            plt.savefig(os.path.join(filepath,"result_anisotropy.png"), bbox_inches='tight',dpi=200)
            plt.close(fig)
            
            # make a plot with histograms of the psi2 directions
            print("creating a plot of histograms of the psi2 directions. This may have to be adapted by the user.")
            if len(psi2all)>0: # if individual models are collected
                anisodirs = np.vstack(psi2all)
                anisodirs = 0.5*np.arctan2(np.sin(2*anisodirs),np.cos(2*anisodirs))
                anisoamps = np.vstack(psi2ampall)
            else: # if individual models are not collected, make histogram from means of chains
                print("searches did not collect invididual models, histograms are created from chain means (for a better plot, set store_models=True)")
                anisodirs = []
                anisoamps = []
                for i,aniso_mod in enumerate(aniso_models):
                    anisodirs.append(0.5*np.arctan2(aniso_mod[:,1],aniso_mod[:,0]))
                    anisoamps.append(np.sqrt(
                        (aniso_mod[:,1]/average_model_counters_aniso[i])**2+
                        (aniso_mod[:,0]/average_model_counters_aniso[i])**2))
                anisodirs = np.array(anisodirs)
                anisoamps = np.array(anisoamps)
            fig = plt.figure(figsize=(16,12))
            gs = GridSpec(5, 5)
            axmap = fig.add_subplot(gs[1:4,1:4],projection=proj)
            axmap.coastlines(resolution='50m')
            axmap.add_feature(cf.BORDERS.with_scale('50m'))
            axmap.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
            axmap.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
            cbar = axmap.pcolormesh(LON,LAT,avg_model,cmap=plt.cm.jet_r,
                                  shading='nearest',transform=ccrs.PlateCarree())
            #plt.colorbar(cbar,shrink = 0.5)
            thresh = 0.05
            threshold = psi2uncertainty.reshape(X.shape)[::5,::5] < thresh
            q = axmap.quiver(LON[::5,::5][threshold],LAT[::5,::5][threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::5,::5][threshold],
                           100*anisotropy_ampmean.reshape(np.shape(X))[::5,::5][threshold],
                           angles=anisotropy_dirmean.reshape(np.shape(X))[::5,::5][threshold]/np.pi*180,
                           headwidth=0,headlength=0,headaxislength=0,
                           pivot='middle',scale=70.,width=0.005,
                           color='yellow',edgecolor='k',#scale=80.,width=0.0035,
                           linewidth=0.5,transform=ccrs.PlateCarree()) 
            qk = axmap.quiverkey(q, X=0.06, Y=0.04, U=3, label='3%',
                               labelpos='E',#edgecolor='w',linewidth=0.5,
                               fontproperties=dict(size=10))
            qk.text.set_path_effects([path_effects.withStroke(linewidth=2,foreground='w')])
            gsidx = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,4),(2,0),(2,4),
                     (3,0),(3,4),(4,0),(4,1),(4,2),(4,3),(4,4)]
            yheight = anisodirs.shape[0]/2.
            gsanchors = [(90,0),(0,0),(0,0),(0,0),(-90,0),
                         (90,0.5*yheight),(-90,0.5*yheight),(90,0.5*yheight),
                         (-90,0.5*yheight),(90,0.5*yheight),(-90,0.5*yheight),
                         (90,yheight*1.1),(0,yheight*1.1),(0,yheight*1.1),(0,yheight*1.1),(-90,yheight*1.1)]
            if anisodirs.shape[1] == 28350: # for Kaestle et al. publication plot
                plot_idx = [24970, 21800, 21840, 18450 ,23400, 17620, 15400, 13840,
                            12600, 10600,  12200,  7400,  7430,  16500, 2600, 9000]
            else:
                plot_idx = np.sort(np.random.choice(np.arange(anisodirs.shape[1],dtype=int),replace=False,size=16))
            for jj,j in enumerate(plot_idx):
                ax = fig.add_subplot(gs[gsidx[jj]])
                directions = 0.5*np.arctan2(np.sin(2*anisodirs[:,j]),np.cos(2*anisodirs[:,j]))
                directions = directions[anisoamps[:,j]>0.005] # greater 0.5% to count in histogram
                dirstd = np.abs(directions - anisotropy_dirmean[j])
                dirstd[dirstd>np.pi/2.] -= np.pi
                if len(dirstd)>0:
                    dirstd = np.sqrt(np.sum(dirstd**2,axis=0) / len(dirstd))
                else:
                    dirstd = 0.
                if psi2uncertainty[j]<=0.02:
                    ax.hist(directions/np.pi*180,bins=np.linspace(-90,90,37),color='black')
                else:
                    ax.hist(directions/np.pi*180,bins=np.linspace(-90,90,37),color='grey')        
                ax.vlines(anisotropy_dirmean[j]/np.pi*180,0,0.9*yheight,color='red',linewidth=3)
                ax.set_yticklabels([])
                ax.set_ylim(0,yheight*1.1)
                ax.text(-88,0.85*yheight,"uncertainty $e$: %.3f\nstd($\Psi_2$): %d$^{\circ}$" %(psi2uncertainty[j],dirstd/np.pi*180.),fontsize=10)
                ax.set_xticks([-90,-45,0,45,90])
                ax.set_xticklabels(["-$90^{\circ}S$","-$45^{\circ}$","$0^{\circ}E$",
                                    "$45^{\circ}$","$90^{\circ}N$"])
                xyA = gsanchors[jj]
                xyB = proj.transform_point(LON.flatten()[j],LAT.flatten()[j],ccrs.PlateCarree())
                axmap.plot(xyB[0],xyB[1],'ko')
                con = ConnectionPatch(
                    xyA=xyA, xyB=xyB, coordsA="data", coordsB="data",
                    axesA=ax, axesB=axmap, color="black",linestyle='solid',
                    linewidth=0.5)
                ax.add_artist(con)
            plt.savefig(os.path.join(filepath,"anisotropy_histograms.png"), bbox_inches='tight',dpi=100)
            plt.close(fig)
            
            

       
        if len(stationerrors) > 0:
            for dataset in stationerrors:

                if len(stationerrors[dataset])>1:
                    meanstationerrors = np.mean(stationerrors[dataset],axis=0)
                else:
                    meanstationerrors = stationerrors[dataset][0]                    
            
                fig = plt.figure(figsize=(14,10))
                plt.title("mean station errors %s" %dataset)
                cbar = plt.scatter(stationerrors_stations[dataset][:,0],
                                   stationerrors_stations[dataset][:,1],c=meanstationerrors)
                plt.colorbar(cbar)
                plt.savefig(os.path.join(filepath,"result_stationerrors_%s.png" %dataset), bbox_inches='tight')
                plt.close(fig)

    #%%
