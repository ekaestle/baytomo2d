#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
"""

@author: E. Kaestle (FU Berlin), based on the work of F. Tilmann (GFZ Potsdam) and T. Bodin (Lyon)

run this program on a machine with MPI installed (e.g. openmpi) by typing in a console

    mpirun -np number_of_cores python run_rjtransdim2d.py
    
can also be run sequentially without MPI (mpi4py still needs to be installed)

    python run_rjtransdim2d.py

"""
import numpy as np
""" ####################################################### """
""" ## USER DEFINED PARAMETERS ## """
""" ####################################################### """

plot = True # plot input data and results

# number of searches/chains that will be executed on the number of cores
# defined in mpirun -np no_cores
# If this number is lower than the assigned number of cores, it will be
# automatically increased to the available number of cores so no cores are idle.
# If it is greater than no_cores, the jobs will be split so that the workload
# is equally distributed between no_cores cores.
no_searches = 4

# folder where a separate file for each search is created (careful, older files
# will be overwritten when a new search is started, unless resume_job=True)
output_location = "./rayleigh_zz_aniso"

# if resume_job = True, the program will try to read existing files in the
# output_location folder and continue with additional iterations until the
# total number_of_iterations as defined below is reached.
# the follwing parameters can be changed when resuming the job, all others are 
# ignored: priors, proposal_stds, nburnin, update_paths, anisotropic_search,
# collect_step, print_stats_step 
# if output_location does not exists, it is automatically set to False
resume_job = False


""" # # # # # INPUT DATA # # # # # # # """
# The input file(s) should contain (at least) 5 columns
# LAT1 LON1 LAT2 LON2 TRAVELTIME
# the 6th column, if available, is interpreted as traveltime standard deviation
# LAT1 LON1 LAT2 LON2 TRAVELTIME STD
# if the input file contains more than 6 columns, the additional columns are
# interpreted as traveltime values at different periods. If there is no measurement
# for a certain source-receiver pair, give nan in the input file. The traveltime
# standard deviation is always assumed to be in the last column.
# LAT1 LON1 LAT2 LON2 TRAVELTIME_1 TRAVELTIME_2 TRAVELTIME_3 ... TRAVELTIME_N STD
# if a list of several files is given, all datasets at identical periods are
# inverted jointly. datasets may have individual data std
input_files = ["dataset_eastalps.dat",
               ]

# choose between 'latlon' and 'xy' (in km) for the input coordinate system
coordinates = 'latlon'


""" # # # # SEARCH PARAMETERS # # # # #  """
# desired grid spacing in x and y direction. The point model will be 
# interpolated on a regular grid and the forward problem is solved on this grid.
xgridspacing = 5 #in km
ygridspacing = 5 #in km

# Interpolation type can be 
# 'nearest_neighbor' (= Voronoi cells),
# 'linear' (= linear interpolation)
# 'wavelets' (e.g. Sambridge & Hawkins "Transdimensional trees...")
# 'blocks' (rectangular blocks that may overlap)
# 'nocells' (no parameterization, just random block-wise model updates, fast but inefficient)
# standard should be 'nearest_neighbor', the other types are just experimental
interpolation_type = 'nearest_neighbor'

# Optionally, set the gridspacing such that it is never smaller than 
# 1/4 of the average wavelength. Additionally, the grids of each
# chain will be slightly offset to each other, to avoid grid artifacts.
# currently only implemented for nearest neighbor interpolation
wavelength_dependent_gridspacing = True # should normally be False 


# ANY OF THE FOLLOWING PARAMETERS CAN ALSO BE A LIST OF LENGTH no_searches
# IN THIS CASE, THE SEARCHES ARE PERFORMED WITH DIFFERENT PARAMETERS
# e.g. no_searches = 3, number_of_burnin_samples = [2000,5000,6000]
# will start 3 searches, with different numbers of burnin samples

# total number of iterations
number_of_iterations = 500000

# number of burnin samples, these samples are discarded
number_of_burnin_samples = 100000 # has to be smaller than number_of_iterations

# number of points (i.e. voronoi cells) at the beginning (can also be 'random',
# then for each search, a random number in the prior range (below) is chosen)
init_no_points = list(np.linspace(100,800,no_searches).astype(int)) # [20,100,200,500] # 'random'

# initial point velocity (can also be 'random' or a vector of the length
# of init_no_points)
init_vel_points = 'random' # 'random' or velocity in km/s


# misfit norm, choose between L1 and L2
misfit_norm = 'L2' # standard is L2; L1 is more resistant to outliers

# At each iteration the model is perturbed by one of the following operations:
# Changing the velocity in one cell (velocity update)
# Moving the position of one cell (move)
# Creating or destroying one cell (birth/death)
# For each of these operations, the updated value is drawn from a Gaussian
# distribution, defined by the proposal standard deviation that controls
# how far the updated value deviates from the previous model. These values
# can be fixed (normal case) or the algorithm can try to adapt them automatically
# to obtain a good acceptance rate (optimally around 45%).

# if False, the algorithm will modify the proposal stds and try to achieve an
# acceptance ratio close to 45%. The proposal_std_ values below only serve as
# starting values in this case. Otherwise the proposal stds are fixed.
fixed_propvel_std = False # choose from False, True, recommended False

# standard deviation when proposing a velocity update, large values are more
# explorative, but the model may not converge, low values can cause a very
# slow convergence
proposal_std_velocity_update = 0.1 # units in km/s
# standard deviation when proposing the velocity of a newborn point or when
# deleting a point
proposal_std_birth_death = 'uniform'#0.1 # units km/s
# standard deviation when proposing to move a point to a new location
proposal_std_move = 10. # units are same as x,y units (km)

# If delayed_rejection = True, a bad velocity update or move operation is not
# directly rejected, but a second try with lower std is performed, improving
# the convergence rate (see PhD thesis of Thomas Bodin)
delayed_rejection = False

# The data standard deviation can also be a search parameter (hierarchical
# algorithm). In this case choose a standard deviation for proposing a std update
proposal_std_data_uncertainty = 0.05 # has no effect if data_std_type = 'fixed'
# The data std can be 'fixed' (not hierarchical), 'absolute' (same std for all data
# points), 'relative' (std follows a linear relationship scaling with distance)
# or 'stationerrors' (experimental, trying to adjust data std stationwise)
data_std_type = 'absolute' # choose from 'fixed','absolute','relative','stationerrors'

# the probability density function of the data standard deviation is usually 
# assumed to follow a gaussian distribution. With the approach of Tilmann et 
# al. (2020) ["Another look at the treatment of data uncertainty in...] it is
# possible to include outliers in the error model so that they have a lower
# impact on the final result. This results in 1 additional search parameter:
# the fraction of outliers (between 0 - 100%).
model_outliers = True


# Parallel tempering (temperatures are exchanged between chains)
# If parallel tempering is switched off (False) you can still assign a temp-
# erature to each chain. But there will be no temperature exchanges.
parallel_tempering = False # choose between True, False 

# Chain temperatures. Chains running on higher temperatures are more explorative
# and accept more 'bad' model perturbations
# If 'auto' then chain temperatures are assigned according to Atchadé et al. (2011)
# This results in a logarithmic increase from 1 to 1000 (too high in my opinion)
#temperatures = 1# 'auto' Ti = 10**(3*(i − 1)/(n − 1)), (i = 1, …, n) as proposed by Atchadé et al. (2011)
if no_searches>1 and parallel_tempering:
    temperatures = list(np.append(np.ones(int(no_searches/4.)),
                                  np.around(np.logspace(np.log10(1),np.log10(100),num=no_searches-int(no_searches/4.)-1),4)))
    temperatures += [1000]
else:
    # Or fix all temperatures to 1 (standard case, no temperature values)
    temperatures = 1.
    
# if parallel tempering is True, the algorithm can try to adapt the chain
# temperatures such that the temperature swap acceptance rate is between 10-30%
adapt_temperature_profile = True # True or False, has no effect without tempering

# the ray paths are recalculated using the Fast Marching Method (FMM) every
# update_path_interval iteration (if it is greater than the no of iterations
# it will never update the paths, i.e. use only straight paths)
update_path_interval = int(80000)

# When Eikonal paths (FMM method) are calculated, it is often good to refine
# the grid. Especially when the grid spacing is very coarse or the velocity
# model very rough. A higher refine_factor will cost more calculation time.
refine_fmm_grid_factor = 2 # can be 1 if x/y-gridspacing is very small, otherwise between 2 and 10

# Choose between ray/sensitivity kernel types:
# 'rays' means either straight or Eikonal rays of infinitely small width
# 'fat_rays' uses the shape of the first sensitivity kernel orbit (Experimental)
# 'sens_kernels' means that sensitivity kernels are calculated according to Lin & Ritzwoller 2010 (Experimental)
kernel_type = 'rays'

# only every collect_step model (after burnin) is used to calculate the average model
# collect step should be large enough so that models can be considered independent
collect_step = 200

# parameters that do not influence the result of the model search

# The paths can be stored as an array for plotting purposes or to check that
# the paths actually look correct. If False, the ray paths are discarded after
# the A matrix has been set up. This saves a bit of disk space.
store_paths = False # choose from True, False

# all models can be stored as complete models interpolated on the x-y grid
# this requires more disk space (but is necessary to create the histogram
# map of the anisotropy).
store_models = True

# print the convergence rate every print_stats_step iteration to log file
print_stats_step = 10000
# save chain files every saveto_file_step
saveto_file_step = 100000

visualize = False # visualize progress (slower, only for testing)
visualize_step = 100 # visualize every 'visualize_step' iteration

""" # # # # # # PRIORS # # # # # # """
# ANY OF THESE CAN ALSO BE A LIST OF LENGTH no_searches
# IN THIS CASE, THE SEARCHES ARE PERFORMED WITH DIFFERENT PRIORS
# min/max velocity for the individual points. Can be set to 'auto', in this case
# the min/max values are determined from the min/max velocities in the input
# data minus/plus 5%.
min_velocity = 'auto' # set to value in km/s or 'auto'
max_velocity = 'auto' # set to value in km/s or 'auto'
# allowed number of points (Voronoi cells in case of nearest neighbor interpolation)
min_no_points = 10
max_no_points = 4000
# the following range is only relevant if the data standard deviation is treated
# as an unknown (hierarchical approach). If data_std_type='fixed', it is ignored.
min_datastd = 0.01 # expected minimum standard deviation of the input data
max_datastd = 2.00 # expected maximum standard deviation of the input data


""" # # # # # ANISOTROPIC SEARCH # # # # # """
# trying to fit 2psi anisotropy; search will always start with an isotropic
# model and only start varying the anisotropic parameters after
# half of the burnin samples
anisotropic_search = True

# choose priors for the anisotropic search
min_aniso_amplitude = 0. # recommended to be zero
max_aniso_amplitude = 0.05 # 0.1 means 10%
proposal_std_aniso_direction = 20/180.*np.pi # 10 degrees
proposal_std_aniso_amplitude = 0.02 # 0.03 = three percent


""" ##################################################### """
""" # END OF USER DEFINED PARAMETERS # """
""" ##################################################### """

from mpi4py import MPI
import scipy.version as scipy_version
scv1,scv2,scv3 = (scipy_version.version).split(".")
if float(scv1)<=1 and float(scv2)<=3 and anisotropic_search:
    raise Exception("You are using an old version of scipy (%s), " %scipy_version.version +
                    "this may cause problems for anisotropic searches " +
                    "because of the sparse matrix formulation. Please " +
                    "upgrade to a version >= 1.4.")
import os
import matplotlib
if not bool(os.environ.get('DISPLAY', None)):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import rjtransdim2d_helper_classes as rhc
import time, pickle
#from mpl_toolkits.basemap import cm
import cartopy.crs as ccrs
import cartopy.feature as cf
import pyproj
import gzip

    

if __name__ == '__main__': 

    # Initialize MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size() 
    
    if mpi_size > no_searches:
        if mpi_rank == 0:
            print("Warning! The number of searches has been increased to the number of assigned MPI processes")
            print("Initial number of searches: %d, now: %d" %(no_searches,mpi_size))
        no_searches = mpi_size
    if parallel_tempering and no_searches == 1:
        if mpi_rank == 0:
            print("Parallel tempering requires more than one chain. Deactivating parallel tempering.")
        parallel_tempering = False
        
    """ READ INPUT DATA """ 
    periods = []
    datasets = []
    if type(input_files) != type([]):
        input_files = [input_files]
    if mpi_rank==0:
        
        stations = []
        dataset_dict = {}
        i = 0
        for fpath in input_files: 
            i += 1

            input_data = np.loadtxt(fpath)
            #print("Warning! Selecting a subarea of the total dataset")
            #idxvalid = ((input_data[:,0]<49.)*(input_data[:,0]>43.)*(input_data[:,2]<49.)*(input_data[:,2]>43)*
            #            (input_data[:,1]>8.)*(input_data[:,1]<19.)*(input_data[:,3]>8.)*(input_data[:,3]<19.) )
            #input_data = input_data[idxvalid]
            # read input file header
            with open(fpath,"r") as f:
                line1 = f.readline()   
                if not(line1[0] == '#'):
                    print("Info: input file contains no header")
                    dataset_name = os.path.basename(fpath)
                    if np.shape(input_data)[1] == 5:
                        no_periods = 1
                        input_data = np.column_stack((input_data,np.ones(len(input_data))))
                    else:
                        no_periods = np.shape(input_data)[1]-5
                    periods = np.arange(no_periods)
                else:
                    dataset_name = line1[1:].strip().lower()
                    if dataset_name in datasets:
                        dataset_name += " %d" %i
                    datasets.append(dataset_name)
                    line2 = f.readline()
                    line3 = f.readline()
                    periods = np.array(line3.split()[2:]).astype(float)
                    if np.shape(input_data)[1] == 5:
                        no_periods = 1
                        input_data = np.column_stack((input_data,np.ones(len(input_data))))
                    elif np.shape(input_data)[1]-4 == len(periods):
                        no_periods = len(periods)
                        input_data = np.column_stack((input_data,np.ones(len(input_data))))
                    elif np.shape(input_data)[1]-5 == len(periods):
                        no_periods = len(periods)
                    else:
                        print(fpath)
                        print("Columns with traveltime values:",no_periods)
                        print("Period labels in header:",len(periods))
                        print(periods)
                        raise Exception("The number of columns and column labels in the header do not match.")
                    print(dataset_name,"Periods:",periods)
                    
            for pidx,period in enumerate(periods):
                data = input_data.copy()
                data = data[:,(0,1,2,3,pidx+4,-1)]
                data = data[~np.isnan(data[:,-2])]
                data = data[data[:,-2]>0.0]
                #if len(data)>1000:
                #    print("Warning, taking only a subselection of the total data!")
                #    data = data[np.random.choice(np.arange(len(data)),size=1000,replace=False)]
                if len(data)==0:
                    print("no data for dataset",dataset_name,"at period",period)
                    continue
                if not period in dataset_dict.keys():
                    dataset_dict[period] = {}
                dataset_dict[period][dataset_name] = data
    
                stats = np.unique(np.vstack((data[:,:2],data[:,2:4])),axis=0)
                if len(stations) > 0:
                    stations = np.unique(np.vstack((stations,stats)),axis=0)
                else:
                    stations = stats

        periods = []
        datasets = []
        for period in dataset_dict:
            periods.append(period)
            for dataset in dataset_dict[period]:
                datasets.append(dataset)
        periods = np.unique(np.hstack(periods))
        datasets = np.unique(np.hstack(datasets))

        
    periods = mpi_comm.bcast(periods,root=0) 

    output_location_base = output_location
        
    for pidx,period in enumerate(periods):
        
        #if period!=15:
        #    continue
        #print("warning: jumping periods")
        
        output_location = output_location_base+"_"+str(period)
        
        if mpi_rank==0:
            print("\n\nWorking on period ",period)
            if len(dataset_dict[period].keys()) > 1:
                print("Datasets:",dataset_dict[period].keys())
    
        input_list = []
        
        modelsearch = rhc.parallel_search(output_location,
                                          parallel_tempering,
                                          adapt_temperature_profile,
                                          no_searches,
                                          update_path_interval,
                                          refine_fmm_grid_factor,
                                          kernel_type,
                                          anisotropic_search,
                                          store_paths,
                                          saveto_file_step,
                                          mpi_comm)
    
        input_srcxy = {}
        input_rcvxy = {}
        input_ttimes = {}
        input_datastd = {}
        
        if mpi_rank==0:
            
            if not os.path.exists(output_location):
                if resume_job:
                    print("Warning: Cannot resume job because output_folder does not exist. Treating as new job.")
                resume_job = False
                os.makedirs(output_location)
            
            if not resume_job:
            
                if coordinates == 'latlon':
                    central_lon = np.around(np.mean(stations[:,1]),1)
                    central_lat = np.around(np.mean(stations[:,0]),1)
                else:
                    central_lon = central_lat = 0
                g = pyproj.Geod(ellps='WGS84')
                projection_str ="+proj=tmerc +datum=WGS84 +lat_0=%f +lon_0=%f" %(central_lat,central_lon)
                p = pyproj.Proj(projection_str)
                if coordinates == 'latlon':
                    stations_xy = np.column_stack(p(stations[:,1],stations[:,0]))/1000.      
                else:
                    stations_xy = stations
                    
                velocities = []
                
                for dataset in dataset_dict[period]:
                    
                    data = dataset_dict[period][dataset]   
        
                    stats = np.unique(np.vstack((data[:,:2],data[:,2:4])),axis=0)
                    
                    if coordinates == 'latlon':
                    
                        srcxy = p(data[:,1],data[:,0])
                        rcvxy = p(data[:,3],data[:,2])
                        statsxy = p(stats[:,1],stats[:,0])
                        az,baz,dist_real = g.inv(data[:,1], data[:,0], data[:,3], data[:,2])
                        
                        dist_proj = np.sqrt((srcxy[0]-rcvxy[0])**2 + (srcxy[1]-rcvxy[1])**2)
                        distortion_factor = dist_proj/dist_real#((dist_proj/dist_real)-1)*20+1
                        if np.max(distortion_factor)>1.02 or np.min(distortion_factor)<0.98:
                            print("WARNING: significant distortion by the coordinate projection!")
                        
                    elif coordinates == 'xy':
                        
                        srcxy = data[:,(0,1)].T*1000. # in meter
                        rcvxy = data[:,(2,3)].T*1000. # in meter
                        statsxy = stats
                        dist_real = np.sqrt((srcxy[0]-rcvxy[0])**2 + (srcxy[1]-rcvxy[1])**2)
                        distortion_factor = np.ones_like(dist_real)

                    vels = dist_real/data[:,-2]/1000.
                    
                    # print("warning, randomizing data!")
                    # vels = np.random.normal(loc=np.mean(vels),scale=np.std(vels),size=len(vels))
                    # vels[::2] = np.random.uniform(np.mean(vels)-0.5,np.mean(vels)+0.5,size=int(np.ceil(len(vels)/2.)))
                    # data[:,-2] = dist_real/(vels*1000.)
                    
                    velocities.append(vels) # in km/s
                
                    # variables that serve as input for the model search
                    #idxvalid = np.where((vels>np.mean(vels)-1*np.std(vels))*
                    #                    (vels<np.mean(vels)+1*np.std(vels)))[0]
                    #print("warning discarding data!!")
                    #idxvalid = np.random.choice(np.arange(len(vels)),size=1000,replace=False)
                    idxvalid = np.arange(len(vels)) # <- all data points are valid (above is just for testing)
                    #idxvalid = np.where((dist_real/1000.>180)*(dist_real/1000.<250))[0]
                    #print(len(idxvalid),"/",len(vels))
                    if False:
                        print("Warning! Cutting paths!")
                        from rjtransdim2d_helper_classes import get_paths
                        paths = get_paths([0,0+xgridspacing],[0,0+ygridspacing],
                                          np.column_stack(srcxy)/1000.,
                                          np.column_stack(rcvxy)/1000.,
                                          projection_str)
                        new_srcxy = (np.zeros(len(srcxy[0])),np.zeros(len(srcxy[0])))
                        new_rcvxy = (np.zeros(len(srcxy[0])),np.zeros(len(srcxy[0])))
                        new_ttimes = np.zeros(len(data))
                        for ipath,path in enumerate(paths):
                            path = path[3:-3]
                            dist = np.sum(np.sqrt(np.sum(np.diff(path,axis=0)**2,axis=1)))
                            new_ttimes[ipath] = dist/vels[ipath]
                            new_srcxy[0][ipath] = path[0,0]
                            new_srcxy[1][ipath] = path[0,1]
                            new_rcvxy[0][ipath] = path[-1,0]
                            new_rcvxy[1][ipath] = path[-1,1]
                        srcxy = (new_srcxy[0]*1000,new_srcxy[1]*1000)
                        rcvxy = (new_rcvxy[0]*1000,new_rcvxy[1]*1000)
                        data[:,4] = new_ttimes
                    input_srcxy[dataset] = np.column_stack(srcxy)[idxvalid]/1000.
                    input_rcvxy[dataset] = np.column_stack(rcvxy)[idxvalid]/1000.
                    input_ttimes[dataset] = data[idxvalid,4]*distortion_factor[idxvalid]
                    #print("WARNING: not correcting for projection distortion")
                    input_datastd[dataset] = data[idxvalid,5]
                    input_datastd[dataset][:] = 0.1

                    if plot:
                        if not os.path.exists(os.path.join(output_location,"indata_figures")):
                            os.makedirs(os.path.join(output_location,"indata_figures"))
                        if len(data)>5000:
                            randidx = np.random.choice(np.arange(len(data)),5000,replace=False)
                        else:
                            randidx = np.arange(len(data))
                        #randidx = np.where(vels<3.2)[0]
                        plt.ioff()
                        fig = plt.figure(figsize=(12,10))
                        if coordinates=='latlon':
                            # uses a WGS84 ellipsoid as default to create the transverse mercator projection
                            proj = ccrs.TransverseMercator(central_longitude=central_lon,
                                                           central_latitude=central_lat,
                                                           approx=False)
                            axm = plt.axes(projection=proj)
                            segments = np.hstack((np.split(np.column_stack(srcxy)[randidx],len(randidx)),
                                                  np.split(np.column_stack(rcvxy)[randidx],len(randidx))))
                            lc = LineCollection(segments, linewidths=0.3)
                            lc.set(array=vels[randidx],cmap='jet_r')
                            # PlateCarree is the projection of the input coordinates, i.e. "no" projection (lon,lat)
                            # it is also possible to use ccrs.Geodetic() here 
                            axm.add_collection(lc)
                            axm.plot(stats[:,1],stats[:,0],'rv',ms = 2,transform = ccrs.PlateCarree())
                            axm.coastlines(resolution='50m')
                            axm.add_feature(cf.BORDERS.with_scale('50m'))
                            axm.add_feature(cf.LAND.with_scale('50m'),facecolor='lightgrey')
                            axm.add_feature(cf.OCEAN.with_scale('50m'),facecolor='grey')
                        else:
                            axm = fig.add_subplot(111)
                            segments = np.hstack((np.split(np.column_stack(srcxy/1000.)[randidx],len(randidx)),
                                                  np.split(np.column_stack(rcvxy/1000.)[randidx],len(randidx))))
                            lc = LineCollection(segments, linewidths=0.3)
                            lc.set(array=vels[randidx],cmap='jet_r')
                            axm.add_collection(lc)
                            axm.plot(stats[:,0],stats[:,1],'rv',ms=2)
                            axm.set_aspect('equal')
                        plt.colorbar(lc,fraction=0.05,shrink=0.5,label='velocity')
                        plt.savefig(os.path.join(output_location,"indata_figures","input_data_%s_%.1f.png" %(dataset,period)),bbox_inches='tight')
                        plt.close(fig)
              
                
                # # # MODELSEARCH PARAMETERS # # 
                #print("Increasing map size for sensitivity kernels!")
                xmargin = np.max([2*xgridspacing,0.05*(np.max(stations_xy[:,0])-np.min(stations_xy[:,0]))])
                ymargin = np.max([2*ygridspacing,0.05*(np.max(stations_xy[:,1])-np.min(stations_xy[:,1]))])
                minx = np.floor((np.min(stations_xy[:,0])-xmargin)/xgridspacing) * xgridspacing
                maxx = np.ceil((np.max(stations_xy[:,0])+xmargin)/xgridspacing) * xgridspacing
                miny = np.floor((np.min(stations_xy[:,1])-ymargin)/ygridspacing) * ygridspacing
                maxy = np.ceil((np.max(stations_xy[:,1])+ymargin)/ygridspacing) * ygridspacing
                
                velocities = np.hstack(velocities)
                if min_velocity == 'auto' or max_velocity == 'auto':
                    # assume 0.5 percent outliers
                    vmin = np.percentile(velocities,0.25) 
                    vmax = np.percentile(velocities,99.75)
                if min_velocity == 'auto':
                    # 25% smaller than the minimum velocity wrt the total velocity range                     
                    min_vel = np.floor((vmin-(vmax-vmin)*0.25)*100.)/100.
                    print("Minimum interstation velocity in the dataset is",np.around(np.min(velocities),3))
                    print("Minimum velocity prior for model search set to",min_vel,"km/s")
                else:
                    min_vel = min_velocity
                    print("Prior minimum velocity is",min_vel,"km/s. Minimum " +
                          "interstation velocity in the dataset is",np.min(velocities))
                if max_velocity == 'auto':
                    # 25% larger than the maximum velocity wrt the total velocity range
                    max_vel = np.ceil((vmax+(vmax-vmin)*0.25)*100.)/100.
                    print("Maximum interstation velocity in the dataset is",np.around(np.max(velocities),3))
                    print("Maximum velocity prior for model search set to",max_vel,"km/s")
                else:
                    max_vel = max_velocity
                    print("Prior maximum velocity is",max_vel,"km/s. Maximum " +
                          "interstation velocity in the dataset is",np.max(velocities))
                    
                meanvel = np.mean(velocities)
                        
                if init_no_points == 'random':
                    init_no_points = list(np.random.randint(min_no_points,max_no_points,no_searches))
        
            else: # if resuming job these variables aren't needed
                minx = maxx = miny = maxy = projection_str = None
                meanvel = None

                if max_velocity == 'auto':
                    max_vel = None
                else:
                    max_vel = max_velocity
                if min_velocity == 'auto':
                    min_vel = None
                else:
                    min_vel = min_velocity
                    
                    
            if isinstance(temperatures,str):
                if temperatures == 'auto':
                    temperatures = 10**(3*np.arange(no_searches)/(no_searches - 1)) #, (i = 1, …, n) as proposed by Atchadé et al. (2011)
                    print("Temperature levels:",temperatures)
                              
            for i in range(no_searches):
                # # # INITIALIZE VARIABLES # # # 
                params = rhc.initialize_params(
                    minx=minx, maxx=maxx, miny=miny,maxy=maxy,
                    xgridspacing = xgridspacing,
                    ygridspacing = ygridspacing,
                    projection = projection_str,
                    interpolation_type = interpolation_type,
                    wavelength_dependent_gridspacing = wavelength_dependent_gridspacing,
                    number_of_iterations = rhc.check_list(number_of_iterations,i,no_searches),
                    misfit_norm = rhc.check_list(misfit_norm,i,no_searches),
                    model_outliers = rhc.check_list(model_outliers,i,no_searches),
                    init_no_points = rhc.check_list(init_no_points,i,no_searches),
                    init_vel_points = rhc.check_list(init_vel_points,i,no_searches),
                    propvelstd_velocity_update = rhc.check_list(proposal_std_velocity_update,i,no_searches),
                    propvelstd_birth_death = rhc.check_list(proposal_std_birth_death,i,no_searches),
                    fixed_propvel_std = rhc.check_list(fixed_propvel_std,i,no_searches),
                    propmove_std = rhc.check_list(proposal_std_move,i,no_searches),
                    delayed_rejection = rhc.check_list(delayed_rejection,i,no_searches),
                    temperature = rhc.check_list(temperatures,i,no_searches),
                    number_of_burnin_samples = rhc.check_list(number_of_burnin_samples,i,no_searches),
                    update_path_interval = rhc.check_list(update_path_interval,i,no_searches),
                    period = period,
                    meanvel = meanvel,
                    kernel_type = rhc.check_list(kernel_type,i,no_searches),
                    propsigma_std = rhc.check_list(proposal_std_data_uncertainty,i,no_searches),
                    data_std_type = rhc.check_list(data_std_type,i,no_searches),
                    anisotropic = rhc.check_list(anisotropic_search,i,no_searches),
                    propstd_anisodir = rhc.check_list(proposal_std_aniso_direction,i,no_searches),
                    propstd_anisoamp = rhc.check_list(proposal_std_aniso_amplitude,i,no_searches),
                    collect_step = rhc.check_list(collect_step,i,no_searches),
                    store_models = rhc.check_list(store_models,i,no_searches),
                    print_stats_step = rhc.check_list(print_stats_step,i,no_searches),
                    logfile_path = rhc.check_list(output_location,i,no_searches),
                    visualize = rhc.check_list(visualize,i,no_searches),
                    visualize_step = rhc.check_list(visualize_step,i,no_searches),
                    resume = resume_job)
                prior = rhc.initialize_prior(
                    min_velocity = rhc.check_list(min_vel,i,no_searches),
                    max_velocity=rhc.check_list(max_vel,i,no_searches),
                    min_no_points=rhc.check_list(min_no_points,i,no_searches),
                    max_no_points=rhc.check_list(max_no_points,i,no_searches),
                    min_datastd = rhc.check_list(min_datastd,i,no_searches),
                    max_datastd = rhc.check_list(max_datastd,i,no_searches),
                    aniso_ampmin = rhc.check_list(min_aniso_amplitude,i,no_searches),
                    aniso_ampmax = rhc.check_list(max_aniso_amplitude,i,no_searches))
                
                if resume_job:
                    if i==0:
                        print("Resuming jobs")
                    try:
                        with gzip.open(os.path.join(output_location,"rjmcmc_chain%d.pgz" %(i+1)), "rb") as f:
                            chain = pickle.load(f)
                        input_list.append([i+1,"dummy",prior,params])
                    except:
                        print("Warning! Could not read",os.path.join(output_location,"rjmcmc_chain%d.pgz" %(i+1)),"File corrupt?")
                    
                else:

                    indata = rhc.initialize_data(input_srcxy,input_rcvxy,
                                                 input_ttimes,input_datastd)
                    input_list.append([i+1,indata,prior,params])
                    
            if len(input_list) < mpi_size:
                raise Exception("Less chains (%d) than MPI processes (%d), this may cause problems." %(len(input_list),mpi_size))

    
        # share data among processes                    
        input_list = mpi_comm.bcast(input_list,root=0) 
        resume_job = mpi_comm.bcast(resume_job,root=0)
        
        modelsearch.setup_model_search(input_list,
                                       mpi_rank,mpi_size,mpi_comm,
                                       resume=resume_job)
        
        # free up space
        input_list = []
        
        # start only after all chains are done initializing
        mpi_comm.Barrier()
        
        
        # Run
        if mpi_rank==0:
            print("Starting model search")
        t0 = time.time()
        _result = modelsearch.submit_jobs(mpi_rank,mpi_comm)
        
        # wait until all chaines are finished
        mpi_comm.Barrier()
                

        if mpi_rank==0:
            print("total time: %d s" %(time.time()-t0))
              
        
        if plot:
    
            modelsearch.plot(saveplot=True)
            
            if mpi_rank == 0:
                print("plotting average model")
                rhc.plot_avg_model(output_location)
        



       
