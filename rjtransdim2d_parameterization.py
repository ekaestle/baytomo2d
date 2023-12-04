#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:35:55 2021

@author: emanuel
"""

import numpy as np
from operator import itemgetter
from copy import deepcopy
from scipy.spatial import KDTree, Delaunay, distance_matrix, ConvexHull, distance
from scipy.sparse import lil_matrix
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import pickle
try:
    import pywt
except:
    print("pywavelets module not installed.")

            
##############################################################################
##############################################################################
##############################################################################

class voronoi_cells(object):
    
    def __init__(self,gridpoints,shape,velmin,velmax,init_no_points,
                 psi2ampmin=0.,psi2ampmax=0.1,anisotropic=False,
                 data_azimuths=None,data_idx=None,
                 min_azimuthal_coverage=0.,min_data=0,
                 gridspacing_staggered=None): 
        
        self.gridpoints = gridpoints
        self.shape = shape
        self.anisotropic = anisotropic
        
        self.minx = np.min(self.gridpoints[:,0])
        self.maxx = np.max(self.gridpoints[:,0])
        self.miny = np.min(self.gridpoints[:,1])
        self.maxy = np.max(self.gridpoints[:,1])

        if gridspacing_staggered is not None:
            dx = np.diff(np.unique(self.gridpoints[:,0]))[0]
            dy = np.diff(np.unique(self.gridpoints[:,1]))[0]
            if gridspacing_staggered > np.max([dx,dy]):
                print("Warning! introducing staggered grid of size",np.around(gridspacing_staggered,3))
                dx = dy = gridspacing_staggered
                x = np.arange(self.minx+np.random.uniform(-0.5,0.5)*dx,
                              self.maxx+np.random.uniform(-0.5,0.5)*dx+dx,
                              dx)
                y = np.arange(self.miny+np.random.uniform(-0.5,0.5)*dy,
                              self.maxy+np.random.uniform(-0.5,0.5)*dy+dy,dy)
                X,Y = np.meshgrid(x,y)
                gridpoints = np.column_stack((X.flatten(),Y.flatten()))
                kdtree = KDTree(gridpoints)
                nndist,nnidx = kdtree.query(self.gridpoints)
                self.gridpoints = gridpoints[nnidx]

        # if velmin < 0 and velmax > 0:
        #     self.slomin = 1./velmin
        #     self.slomax = 1./velmax
        # else:
        #     self.slomin = 1./velmax
        #     self.slomax = 1./velmin
        # self.slorange = self.slomax-self.slomin
        self.velmax = velmax
        self.velmin = velmin
        self.velrange = self.velmax-self.velmin        
                    
        self.data_azimuths = data_azimuths
        self.data_idx = data_idx
        self.min_azimuthal_coverage = min_azimuthal_coverage
        self.min_data = min_data

        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.grid_nndists_backup = None
        self.grid_nnidx_backup = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None                        
        self.propvelstd_dimchange = 'uniform' # will be adapted when running
        self.slowfield_voronoi = None
        self.slowfield_voronoi_backup = None
        self.psi2amp_voronoi = None
        self.psi2amp_voronoi_backup = None
        self.psi2_voronoi = None
        self.psi2_voronoi_backup = None        
        # min_point_dist controls that neighboring cells have a minimum
        # distance given by the self.min_point_dist interpolation function.
        self.min_point_dist = None
        #
        
        self.smooth_model = False
        if self.smooth_model:
            self.smoothing_matrix = lil_matrix((len(self.gridpoints),len(self.gridpoints)))
            #self.smoothing_matrix[[np.arange(len(self.gridpoints),dtype=int),
            #                       np.arange(len(self.gridpoints),dtype=int)]] = 1.
            print("Trying to apply model smoothing with 25km radius")
            for idx in range(len(self.gridpoints)):
                dists = np.sqrt(np.sum((self.gridpoints[idx]-self.gridpoints)**2,axis=1))
                weights = trunc_normal(dists,0,25,sig_trunc=2)
                # remove elements with very small influence to reduce the matrix size
                weights[weights<0.01*weights.max()] = 0.
                # normalize to 1
                weights /= weights.sum()
                self.smoothing_matrix[idx] = weights
            self.smoothing_matrix = self.smoothing_matrix.tocsc()       
        
        # initialize points
        self.points = np.column_stack((
            np.random.uniform(low=self.minx,high=self.maxx),
            np.random.uniform(low=self.miny,high=self.maxy)))
        self.vs = np.random.uniform(self.velmin,self.velmax,size=(1,))
        kdt = KDTree(self.points)
        self.grid_nndists,self.grid_nnidx = kdt.query(self.gridpoints)
        
        if anisotropic:
            self.psi2ampmin=psi2ampmin
            self.psi2ampmax=psi2ampmax
            self.psi2amp = np.zeros(len(self.points))
            self.psi2 = np.random.uniform(-np.pi,np.pi,size=len(self.points))
        else:
            self.psi2amp = None
            self.psi2 = None       
        
        # add points until init_no_points is reached
        k = 0
        while True:
            valid = self.add_point(anisotropic=anisotropic)
            k += 1
            if valid:
                self.accept_mod()
            else:
                self.reject_mod()
            if len(self.points) == init_no_points:
                break
            if k > init_no_points+10000:
                print("starting model has less cells than required")
                break


    def psi2amp_update(self,idx,delta,backup=True):
        
        if backup:
            self.psi2amp_backup = self.psi2amp.copy()
            
        self.psi2amp[idx] += delta
        self.action='psi2amp_update'
        self.idx_mod = idx
        
        if self.psi2amp[idx]<self.psi2ampmin or self.psi2amp[idx]>self.psi2ampmax:
            return False
        if (self.vs[idx]*(1+self.psi2amp[idx]) > self.velmax or 
            self.vs[idx]*(1-self.psi2amp[idx]) < self.velmin):
            return False
        
        self.idx_mod_gpts = self.get_modified_gridpoints(
            self.points,'update',self.idx_mod)
        
        return True
    
        
    def psi2_update(self,idx,delta,backup=True):
        
        if backup:
            self.psi2_backup = self.psi2.copy()
            
        # modulo makes sure it's always in the 0-2pi range
        # the Python modulo convention also correctly treats negative angles
        self.psi2[idx] = (self.psi2[idx]+delta)%(2*np.pi)
        if self.action=='psi2amp_update':
            self.action='anisotropy_birth_death'
        else:
            self.action='psi2_update'
        self.idx_mod = idx
        
        self.idx_mod_gpts = self.get_modified_gridpoints(
            self.points,'update',self.idx_mod)
        
        return True
        

    def vs_update(self,idx,dvs,backup=True):
        
        if backup:
            self.vs_backup = self.vs.copy()
            
        self.action='velocity_update'
        self.idx_mod = idx
            
        self.vs[idx] += dvs
        
        if self.vs[idx]<self.velmin or self.vs[idx]>self.velmax:
            return False
        if self.psi2amp is not None:
            if (self.vs[idx]*(1+self.psi2amp[idx]) > self.velmax or 
                self.vs[idx]*(1-self.psi2amp[idx]) < self.velmin):
                return False
        
        self.idx_mod_gpts = self.get_modified_gridpoints(
            self.points,'update',self.idx_mod)
        
        return True
        
        
    def add_point(self,anisotropic=False,birth_prop='uniform',backup=True):
        
        if backup:
            self.backup_mod()
        
        self.action='birth'
        self.idx_mod = len(self.vs)
        
        prop_point_x = np.random.uniform(self.minx,self.maxx)
        prop_point_y = np.random.uniform(self.miny,self.maxy)
        point = np.hstack((prop_point_x,prop_point_y))
        if False:#self.min_point_dist is not None:
            points = np.vstack((self.points,point))
            idx_subset_points = np.append(
                self.get_neighbors(points,self.idx_mod),self.idx_mod)
            kdt = KDTree(points[idx_subset_points])
            nndist,nnidx = kdt.query(points[idx_subset_points],k=2)
            while (nndist[:,1]<self.min_point_dist(points[idx_subset_points])).any():
                prop_point_x = np.random.uniform(self.minx,self.maxx)
                prop_point_y = np.random.uniform(self.miny,self.maxy)
                point = np.hstack((prop_point_x,prop_point_y))
                points = np.vstack((self.points,point))
                idx_subset_points = np.append(
                    self.get_neighbors(points,self.idx_mod),self.idx_mod)
                kdt = KDTree(points[idx_subset_points])
                nndist,nnidx = kdt.query(points[idx_subset_points],k=2)
            self.points = points
        else:
            self.points = np.vstack((self.points,point))
            
        if anisotropic:
            self.psi2amp = np.append(self.psi2amp,0.)
            self.psi2 = np.append(self.psi2,0.)
            
        self.idx_mod_gpts = self.get_modified_gridpoints(
            self.points,'birth',self.idx_mod)

        if birth_prop=='uniform':
            vs_birth = np.random.uniform(self.velmin,self.velmax)
        else:
            self.prop_dv = np.random.normal(loc=0.0,scale=birth_prop)
            if len(self.idx_mod_gpts)==0:
                # this happens if the newborn Voronoi cell is so small that there are no gridpoints inside
                # avoiding that vs_birth is NaN (maybe should rather be rejected?)
                vs_birth = np.random.uniform(self.velmin,self.velmax)
            else:
                vs_birth = np.mean(self.vs[self.grid_nnidx_backup[self.idx_mod_gpts]]) + self.prop_dv
            if vs_birth > self.velmax or vs_birth < self.velmin:
                return False

            
        self.vs = np.append(self.vs,vs_birth)
        
        # for the prior_proposal_ratio calculation:
        self.propvelstd_dimchange = birth_prop
        
        valid_cells = self.check_min_coverage(
            min_azi_coverage=self.min_azimuthal_coverage,
            min_data=self.min_data)
        if not valid_cells:
            return False
        
        return True
        
        
        
    def remove_point(self,anisotropic=False,backup=True):
        
        if backup:
            self.backup_mod()
            
        self.action='death'
        
        if anisotropic:
            # choose only points without anisotropy
            ind_pnts = np.where(self.psi2amp == 0.)[0]
            if len(ind_pnts) > 0:
                self.idx_mod = np.random.choice(ind_pnts)
            else:
                return False
        else:
            # choose point to remove randomly
            self.idx_mod = np.random.randint(0,len(self.points))
        
        #pnt_remove = self.points[self.idx_mod]
        vs_remove = self.vs[self.idx_mod]
        
        self.points = np.delete(self.points,self.idx_mod,axis=0)
        self.vs = np.delete(self.vs,self.idx_mod)
        if anisotropic:
            self.psi2amp = np.delete(self.psi2amp,self.idx_mod)
            self.psi2 = np.delete(self.psi2,self.idx_mod)
        
        self.idx_mod_gpts = self.get_modified_gridpoints(
            self.points,'death',self.idx_mod)
        
        # now compare the velocity of the removed point with the velocity
        # at the empty spot (inverse birth operation)
        if self.propvelstd_dimchange != 'uniform' and len(self.idx_mod_gpts)>0:
            self.prop_dv = np.mean(self.vs[self.grid_nnidx[self.idx_mod_gpts]]) - vs_remove
        else:
            self.prop_dv = 0.
        
        return True
                    
            
    def move_point(self,propmovestd,index=None,backup=True):

        if backup:
            self.backup_mod()
        
        self.action = 'move'
        
        if index is None:
            index = np.random.randint(0,len(self.points))
        self.idx_mod = index

        oldxy = self.points[index].copy() # otherwise, newidx and oldidx may be identical when returned
        
        dx = np.random.normal(loc=0.0,scale=propmovestd,size=2)
        newxy = oldxy + dx
        
        if False:#self.min_point_dist is not None:
            self.points[index] = newxy
            idx_subset = np.append(self.get_neighbors(self.points,index),index)
            kdt = KDTree(self.points[idx_subset])
            nndist,nnidx = kdt.query(self.points[idx_subset],k=2)
            i = 0
            while (nndist[:,1]<self.min_point_dist(self.points[idx_subset])).any() and i<10:
                dx = np.random.normal(loc=0.0,scale=propmovestd,size=2)
                newxy = oldxy + dx
                self.points[index] = newxy
                idx_subset = np.append(self.get_neighbors(self.points,index),index)
                kdt = KDTree(self.points[idx_subset])
                nndist,nnidx = kdt.query(self.points[idx_subset],k=2)
                i += 1
            if i==10:
                return (np.nan,np.nan)
        else:
            self.points[index] = newxy
                
        if (newxy[0]>self.maxx or newxy[0]<self.minx or
            newxy[1]>self.maxy or newxy[1]<self.miny):
            return (np.nan,np.nan)

        self.idx_mod_gpts = self.get_modified_gridpoints(
            self.points, 'move', self.idx_mod)

        valid_cells = self.check_min_coverage(
            min_azi_coverage=self.min_azimuthal_coverage,
            min_data=self.min_data)
        if not valid_cells:
            return (np.nan,np.nan)
        
        return (oldxy,newxy)
        

    def get_modified_gridpoints(self,points,action,idx_point):
              
        if action=='update':
            idx_mod_gpts = np.where(self.grid_nnidx==idx_point)[0]
        elif action=='birth':
            dists = distance.cdist(self.gridpoints,points[-1:]).flatten()
            #dists = np.sqrt(np.sum((self.gridpoints-points[-1])**2,axis=1))
            idx_mod_gpts = np.where(dists<self.grid_nndists)[0]
            self.grid_nndists[idx_mod_gpts] = dists[idx_mod_gpts]
            self.grid_nnidx[idx_mod_gpts] = idx_point
        elif action=='death':
            idx_mod_gpts = np.where(self.grid_nnidx==idx_point)[0]
            kdt = KDTree(points)
            nndist,nnidx = kdt.query(self.gridpoints[idx_mod_gpts])
            self.grid_nndists[idx_mod_gpts] = nndist
            self.grid_nnidx[self.grid_nnidx>idx_point] -= 1
            self.grid_nnidx[idx_mod_gpts] = nnidx
        elif action=='move':
            idx_old = np.where(self.grid_nnidx==idx_point)[0]
            kdt = KDTree(points)
            nndist,nnidx = kdt.query(self.gridpoints[idx_old])
            self.grid_nndists[idx_old] = nndist
            self.grid_nnidx[idx_old] = nnidx
            dists = distance.cdist(self.gridpoints,points[idx_point:idx_point+1]).flatten()
            #dists = np.sqrt(np.sum((self.gridpoints-points[idx_point])**2,axis=1))
            idx_new = np.where(dists<self.grid_nndists)[0]
            self.grid_nndists[idx_new] = dists[idx_new]
            self.grid_nnidx[idx_new] = idx_point
            idx_mod_gpts = np.append(
                idx_old[self.grid_nnidx[idx_old]!=idx_point],idx_new)
        else:
            raise Exception("action undefined!",action)
            
        return idx_mod_gpts
    
        
    def get_prior_proposal_ratio(self):
        
        # this function returns the prior ratio times the proposal ratio which
        # is needed to calculate the acceptance probability  
        if 'update' in self.action or self.action == 'move' or self.action == 'anisotropy_birth_death':
            # is always log(1)=0, unless delayed rejegion which is currently
            # included in the main script
            return 0
        elif self.action == 'birth':
            if self.anisotropic:
                aniso_factor = np.log(len(self.points)/(len(self.points)+1))
            else:
                aniso_factor = 0.
            if self.propvelstd_dimchange == 'uniform':
                # if we draw from a uniform prior, everything cancels out
                return aniso_factor + 0
            else:
                # see for example equation A.34 of the PhD thesis of Thomas Bodin
                return (aniso_factor + 
                    np.log(self.propvelstd_dimchange*np.sqrt(2.*np.pi) / self.velrange) +  
                    (self.prop_dv**2 / (2*self.propvelstd_dimchange**2)))
        elif self.action == 'death':
            if self.anisotropic:
                aniso_factor = np.log(len(self.points)/(len(self.points)-1))
            else:
                aniso_factor = 0.
            if self.propvelstd_dimchange == 'uniform':
                return aniso_factor + 0
            else:
                return ( aniso_factor + 
                    np.log(self.velrange/(self.propvelstd_dimchange*np.sqrt(2.*np.pi))) -
                    (self.prop_dv**2 / (2*self.propvelstd_dimchange**2)))
        else:
            raise Exception("action undefined")
    
    def get_model(self,points=None,vs=None,psi2amp=None,psi2=None,anisotropic=False):
        
        if points is None:
            points = self.points
        if vs is None:
            vs = self.vs
            
        if anisotropic:
            if psi2amp is None:
                psi2amp = self.psi2amp 
            if psi2 is None:
                psi2 = self.psi2
            func = NearestNDInterpolator(points,np.column_stack((vs,psi2amp,psi2)))                  
            field = func(self.gridpoints)
            slowfield = 1./field[:,0]
            psi2amp = field[:,1]
            psi2 = field[:,2]
            if self.smooth_model:
                self.slowfield_voronoi = slowfield
                slowfield = self.smooth_slowfield()
                self.psi2amp_voronoi = psi2amp
                self.psi2_voronoi = psi2
                psi2amp,psi2 = self.smooth_anisofields()
            return (slowfield,psi2amp,psi2)
        
        else:
            func = NearestNDInterpolator(points,vs)
            slowfield = 1./func(self.gridpoints)
            if self.smooth_model:
                self.slowfield_voronoi = slowfield
                slowfield = self.smooth_slowfield()
            #slowfield[self.stationgridpoints] = 1./3.1
            return  slowfield
    
    
    def update_model(self,fields=None,anisotropic=False):
        
        slowfield_cp = (1./self.vs)[self.grid_nnidx]
        if anisotropic:
            slowfield, psi2amp, psi2 = fields
            psi2amp_cp = self.psi2amp[self.grid_nnidx]
            psi2_cp = self.psi2[self.grid_nnidx]
        else:
            slowfield = fields
                 
        if self.smooth_model:
            # internally, a unsmoothed velocity field is kept as slowfield_voronoi
            self.slowfield_voronoi_backup = self.slowfield_voronoi.copy()
            self.slowfield_voronoi = slowfield_cp
            slowfield_cp = self.smooth_slowfield(
                slowfield_old=slowfield,idx_mod=self.idx_mod_gpts)
            if anisotropic:
                self.psi2amp_voronoi_backup = self.psi2amp_voronoi.copy()
                self.psi2amp_voronoi = psi2amp_cp
                self.psi2_voronoi_backup = self.psi2_voronoi.copy()
                self.psi2_voronoi = psi2_cp
                psi2amp_cp, psi2_cp = self.smooth_anisofields(
                    psi2amp_old=psi2amp,psi2_old=psi2,idx_mod=self.idx_mod_gpts)
            self.idx_mod_gpts = np.where(slowfield_cp!=slowfield)[0]
            
        if anisotropic:   
            return slowfield_cp, psi2amp_cp, psi2_cp
        else:
            return slowfield_cp
        
        
    def smooth_slowfield(self,slowfield_old=None,idx_mod=None):
                
        if idx_mod is None:
            return self.smoothing_matrix*self.slowfield_voronoi
        else: # only update, slowfield is in this case the difference between
              # new and old voronoi models
            dslow = self.slowfield_voronoi-self.slowfield_voronoi_backup
            return (slowfield_old + 
                    self.smoothing_matrix[:,idx_mod] * dslow[idx_mod])
        
    def smooth_anisofields(self,psi2amp_old=None,psi2_old=None,idx_mod=None):
        
        #x = self.psi2amp_voronoi*np.cos(self.psi2_voronoi)
        #y = self.psi2amp_voronoi*np.sin(self.psi2_voronoi)
        x = (self.psi2amp*np.cos(2*self.psi2))[self.grid_nnidx]
        y = (self.psi2amp*np.sin(2*self.psi2))[self.grid_nnidx]
        if idx_mod is None:
            xsmooth = self.smoothing_matrix*x
            ysmooth = self.smoothing_matrix*y
        else:
            dx = x - self.psi2amp_voronoi_backup*np.cos(2*self.psi2_voronoi_backup)
            dy = y - self.psi2amp_voronoi_backup*np.sin(2*self.psi2_voronoi_backup)
            xold = psi2amp_old*np.cos(psi2_old)
            yold = psi2amp_old*np.sin(psi2_old)
            xsmooth = xold + self.smoothing_matrix[:,idx_mod] * dx[idx_mod]
            ysmooth = yold + self.smoothing_matrix[:,idx_mod] * dy[idx_mod]
          
        psi2amp_smooth = np.sqrt(xsmooth**2+ysmooth**2)
        psi2_smooth = 0.5*np.arctan2(ysmooth,xsmooth)
        return psi2amp_smooth,psi2_smooth


    def get_neighbors(self,points,idx):
        def func1(points,idx):
            tri = Delaunay(points)
            if len(tri.coplanar)>0:
                print("Warning: coplanar points!")
            intptr,neighbor_indices = tri.vertex_neighbor_vertices
            return neighbor_indices[intptr[idx]:intptr[idx+1]]
    
        def func2(idx):
            idx_flat = np.where(self.grid_nnidx==idx)[0]
            idx1,idx2 = np.unravel_index(idx_flat,shape=self.shape)
            idx_neighbors = np.vstack((np.column_stack((idx1-1,idx2-1)),
                                       np.column_stack((idx1-1,idx2)),
                                       np.column_stack((idx1-1,idx2+1)),
                                       np.column_stack((idx1,idx2-1)),
                                       np.column_stack((idx1,idx2+1)),
                                       np.column_stack((idx1+1,idx2-1)),
                                       np.column_stack((idx1+1,idx2)),
                                       np.column_stack((idx1+1,idx2+1))))
            idx_neighbors = np.unique(idx_neighbors,axis=0)
            n1 = idx_neighbors[:,0]
            n2 = idx_neighbors[:,1]
            idx_flat = np.ravel_multi_index((n1,n2),self.shape)
            neighbor_indices = np.unique(self.grid_nnidx[idx_flat])
            neighbor_indices = neighbor_indices[neighbor_indices!=idx]
            return neighbor_indices
        
        test = np.zeros(self.shape)
        test[idx1,idx2] = 1
        test[n1,n2] += 1
        plt.figure()
        plt.pcolormesh(test)
        plt.colorbar()
        plt.show()
                   
    
    
        
    def check_min_coverage(self,min_azi_coverage=135.,min_data=3,idx_points=None):
        
        if min_azi_coverage == 0. and min_data == 0:
            return True
        if min_azi_coverage is None and min_data is None:
            return True
        if self.data_azimuths is None and self.data_idx is None:
            return True
        
        if idx_points is None:
            idx_points = np.unique(
                np.append(self.grid_nnidx_backup[self.idx_mod_gpts],
                          self.grid_nnidx[self.idx_mod_gpts]))
        for idx in idx_points:
            grididx = np.where(self.grid_nnidx==idx)[0]
            valid = self.get_coverage(
                grididx, min_azi_coverage=min_azi_coverage,
                min_data=min_data)
            if not valid:
                return False
                
        return True


    def get_coverage(self, grididx, min_azi_coverage=135.,
                     min_data=3):
        
        if len(grididx) == 0:
            return True

        if min_azi_coverage > 0 and self.data_azimuths is not None:
            coverage = len(np.unique(np.hstack(self.data_azimuths[grididx])))*10
            #coverage = np.max(self.data_azimuths[grididx])
            if coverage > 0 and coverage < min_azi_coverage:
                return False
        
        if min_data > 0 and self.data_idx is not None:
            n_data  = len(np.unique(np.hstack(self.data_idx[grididx])))
            if n_data > 0 and n_data < min_data:
                return False
            
        return True
                   

    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        self.vs_backup = self.vs.copy()
        self.grid_nndists_backup = self.grid_nndists.copy()
        self.grid_nnidx_backup = self.grid_nnidx.copy()
        if self.psi2amp is not None:
            self.psi2amp_backup = self.psi2amp.copy()
        if self.psi2 is not None:
            self.psi2_backup = self.psi2.copy()
        
        
    def reject_mod(self):
        
        if self.points_backup is not None:
            self.points = self.points_backup
        if self.vs_backup is not None:
            self.vs = self.vs_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        if self.slowfield_voronoi_backup is not None:
            self.slowfield_voronoi = self.slowfield_voronoi_backup
        if self.psi2amp_voronoi_backup is not None:
            self.psi2amp_voronoi = self.psi2amp_voronoi_backup
            self.psi2_voronoi = self.psi2_voronoi_backup
        if self.grid_nndists_backup is not None:
            self.grid_nndists = self.grid_nndists_backup
        if self.grid_nnidx_backup is not None:
            self.grid_nnidx = self.grid_nnidx_backup
            
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = 1e99
        self.idx_mod = 1e99
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.slowfield_voronoi_backup = None
        self.psi2amp_voronoi_backup = None
        self.psi2_voronoi_backup = None
        
    def accept_mod(self,selfcheck=False):
                    
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = 1e99
        self.idx_mod = 1e99
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.slowfield_voronoi_backup = None
        self.psi2amp_voronoi_backup = None
        self.psi2_voronoi_backup = None
        self.grid_nndists_backup = None
        self.grid_nnidx_backup = None
        
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):
        
        from scipy.spatial import delaunay_plot_2d
        tri = Delaunay(self.points)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _ = delaunay_plot_2d(tri,ax=ax)
        ax.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.')
        if idx_mod_gpts is not None:
            ax.plot(self.gridpoints[idx_mod_gpts,0],self.gridpoints[idx_mod_gpts,1],'rx',zorder=3)
        cbar = ax.scatter(self.points[:,0],self.points[:,1],c=self.vs,s=40,zorder=3)
        if idx_neighbor_points is not None:
            ax.scatter(self.points[:,0],self.points[:,1],c=self.vs,s=60,zorder=3)
        plt.colorbar(cbar)
        plt.show()
        
        
##############################################################################
##############################################################################
##############################################################################

class dist_weighted_means(object):
    
    def __init__(self,gridpoints,shape,velmin,velmax,init_no_points,
                 psi2ampmin=0.,psi2ampmax=0.1,anisotropic=False,
                 data_azimuths=None,data_idx=None,
                 min_azimuthal_coverage=0.,min_data=0,
                 metric='euclidean',
                 gridspacing_staggered=None): 
        
        self.gridpoints = gridpoints
        self.shape = shape
        self.anisotropic = anisotropic
        self.metric = metric
        self.smooth_radius = 25 # needs testing, if close to xgridspacing, then
        # it looks like Voronoi cells, if larger, Voronoi cells start melting
        # into each other. If too low, cells may not cover the entire region,
        # rest is then covered with the average velocity.
        
        self.minx = np.min(self.gridpoints[:,0])
        self.maxx = np.max(self.gridpoints[:,0])
        self.miny = np.min(self.gridpoints[:,1])
        self.maxy = np.max(self.gridpoints[:,1])

        if gridspacing_staggered is not None:
            dx = np.diff(np.unique(self.gridpoints[:,0]))[0]
            dy = np.diff(np.unique(self.gridpoints[:,1]))[0]
            if gridspacing_staggered > np.max([dx,dy]):
                print("Warning! introducing staggered rough grid")
                dx = dy = gridspacing_staggered
                x = np.arange(self.minx+np.random.uniform(-0.5,0.5)*dx,
                              self.maxx+np.random.uniform(-0.5,0.5)*dx+dx,
                              dx)
                y = np.arange(self.miny+np.random.uniform(-0.5,0.5)*dy,
                              self.maxy+np.random.uniform(-0.5,0.5)*dy+dy,dy)
                X,Y = np.meshgrid(x,y)
                gridpoints = np.column_stack((X.flatten(),Y.flatten()))
                kdtree = KDTree(gridpoints)
                nndist,nnidx = kdtree.query(self.gridpoints)
                self.gridpoints = gridpoints[nnidx]

        self.velmax = velmax
        self.velmin = velmin
        self.velrange = self.velmax-self.velmin

        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None                        
        self.propvelstd_dimchange = 'uniform' # will be adapted when running
        self.psi2xsum_backup = None
        self.psi2ysum_backup = None
        self.vssum_backup = None
        self.weightsum_backup = None
        
        # initialize points
        self.points = np.column_stack((
            np.random.uniform(low=self.minx,high=self.maxx),
            np.random.uniform(low=self.miny,high=self.maxy)))
        self.vs = np.random.uniform(self.velmin,self.velmax,size=(1,))
        
        self.anisotropic = anisotropic
        if anisotropic:
            self.psi2ampmin=psi2ampmin
            self.psi2ampmax=psi2ampmax
            self.psi2amp = np.zeros(len(self.points))
            self.psi2 = np.random.uniform(-np.pi,np.pi,size=len(self.points))
            self.psi2xsum = np.zeros(len(self.gridpoints))
            self.psi2ysum = np.zeros(len(self.gridpoints))
        else:
            self.psi2amp = None
            self.psi2 = None
            
        self.weightsum = np.zeros(len(self.gridpoints))
        self.vssum = np.zeros(len(self.gridpoints))
        for i in range(len(self.points)):
            weights = self.get_weights(self.points[i])
            self.weightsum += weights
            self.vssum += self.vs[i]*weights
            if anisotropic:
                psi2x = self.psi2amp[i]*np.cos(2*self.psi2[i])
                psi2y = self.psi2amp[i]*np.sin(2*self.psi2[i])
                self.psi2xsum += psi2x
                self.psi2ysum += psi2y
       
        # add points until init_no_points is reached
        k = 0
        while True:
            valid = self.add_point(anisotropic=anisotropic)
            k += 1
            if valid:
                self.accept_mod()
            else:
                self.reject_mod()
            if len(self.points) == init_no_points:
                break
            if k > init_no_points+10000:
                print("starting model has less cells than required")
                break
            


    def psi2amp_update(self,idx,delta,backup=True):
        
        if backup:
            self.psi2amp_backup = self.psi2amp.copy()
            self.psi2_backup = self.psi2.copy()
            self.psi2xsum_backup = self.psi2xsum.copy()
            self.psi2ysum_backup = self.psi2ysum.copy()
            
        self.psi2amp[idx] += delta
        self.action='psi2amp_update'
        self.idx_mod = idx
        
        if self.psi2amp[idx]<self.psi2ampmin or self.psi2amp[idx]>self.psi2ampmax:
            return False
        if (self.vs[idx]*(1+self.psi2amp[idx]) > self.velmax or 
            self.vs[idx]*(1-self.psi2amp[idx]) < self.velmin):
            return False

        self.aniso_update(idx)
        
        return True
    
        
    def psi2_update(self,idx,delta,backup=True):
        
        # if psi2amp_update was executed previously, it's an anisotropy birth/death step
        if self.action=='psi2amp_update':
            self.action='anisotropy_birth_death'
            # this is a bit more complicated, othewisse the double execution 
            # of aniso_update will mess up the psi2xsum and psi2ysum arrays.
            newamp = self.psi2amp[idx]
            self.psi2amp = self.psi2amp_backup.copy()
            self.psi2amp[idx] = newamp
            self.psi2xsum = self.psi2xsum_backup.copy()
            self.psi2ysum = self.psi2ysum_backup.copy()
        else:
            self.action='psi2_update'
            
        if backup and self.action!='anisotropy_birth_death':
            self.psi2_backup = self.psi2.copy()   
            self.psi2amp_backup = self.psi2amp.copy()
            self.psi2xsum_backup = self.psi2xsum.copy()
            self.psi2ysum_backup = self.psi2ysum.copy()

        # second, update anisotropy
        # modulo makes sure it's always in the 0-2pi range
        # the Python modulo convention also correctly treats negative angles
        self.psi2[idx] = (self.psi2[idx]+delta)%(2*np.pi)
        self.idx_mod = idx
        
        self.aniso_update(idx)
        
        return True

    def aniso_update(self,idx):

        # remove (old) anisotropy contribution            
        weights = self.get_weights(self.points[idx])
        psi2x = self.psi2amp_backup[idx]*np.cos(2*self.psi2_backup[idx])
        psi2y = self.psi2amp_backup[idx]*np.sin(2*self.psi2_backup[idx])
        self.psi2xsum -= weights*psi2x
        self.psi2ysum -= weights*psi2y

        # add (new) anisotropy contribution       
        psi2x = self.psi2amp[idx]*np.cos(2*self.psi2[idx])
        psi2y = self.psi2amp[idx]*np.sin(2*self.psi2[idx])
        self.psi2xsum += weights*psi2x
        self.psi2ysum += weights*psi2y
            

    def vs_update(self,idx,dvs,backup=True):

        vsnew = self.vs[idx] + dvs
        if vsnew<self.velmin or vsnew>self.velmax:
            return False
        if self.psi2amp is not None:
            if (vsnew*(1+self.psi2amp[idx]) > self.velmax or 
                vsnew*(1-self.psi2amp[idx]) < self.velmin):
                return False        
        
        if backup:
            self.vs_backup = self.vs.copy()
            self.vssum_backup = self.vssum.copy()
            
        self.action='velocity_update'
        self.idx_mod = idx
        
        weights = self.get_weights(self.points[idx])
        self.vssum += weights*dvs
        self.vs[idx] = vsnew
  
        return True
        
        
    def add_point(self,anisotropic=False,birth_prop='uniform',backup=True):
        
        if backup:
            self.backup_mod()
        
        self.action='birth'
        self.idx_mod = len(self.vs)
        
        prop_point_x = np.random.uniform(self.minx,self.maxx)
        prop_point_y = np.random.uniform(self.miny,self.maxy)
        point = np.hstack((prop_point_x,prop_point_y))
        self.points = np.vstack((self.points,point))
            
        weights = self.get_weights(self.points[-1])
        # get the velocity at the location of the newborn cell
        idx_loc = weights.argmax()
        vs_location = self.vssum[idx_loc]/self.weightsum[idx_loc]
        # update the distance array
        self.weightsum += weights
        
        # propose the new velocity, either from a uniform distribution or a
        # Gaussian distribution centered around the velocity at the point location
        if birth_prop=='uniform':
            vs_birth = np.random.uniform(self.velmin,self.velmax)
        else:
            self.prop_dv = np.random.normal(loc=0.0,scale=birth_prop)
            vs_birth = vs_location + self.prop_dv
            if vs_birth > self.velmax or vs_birth < self.velmin:
                return False

        self.vs = np.append(self.vs,vs_birth)
        self.vssum += weights*self.vs[self.idx_mod]
        
        if anisotropic:
            self.psi2amp = np.append(self.psi2amp,0.)
            self.psi2 = np.append(self.psi2,0.)
            # psi2xsum and psi2ysum remain the same since psi2amp is zero
        
        # for the prior_proposal_ratio calculation:
        self.propvelstd_dimchange = birth_prop
        
        return True
        
        
        
    def remove_point(self,anisotropic=False,backup=True):
        
        if backup:
            self.backup_mod()
            
        self.action='death'
        
        if anisotropic:
            # choose only points without anisotropy
            ind_pnts = np.where(self.psi2amp == 0.)[0]
            if len(ind_pnts) > 0:
                self.idx_mod = np.random.choice(ind_pnts)
            else:
                return False
        else:
            # choose point to remove randomly
            self.idx_mod = np.random.randint(0,len(self.points))
        
        #pnt_remove = self.points[self.idx_mod]
        vs_remove = self.vs[self.idx_mod]
        weights = self.get_weights(self.points[self.idx_mod])
        self.weightsum = self.weightsum - weights
        self.vssum = self.vssum - weights*vs_remove
        
        self.points = np.delete(self.points,self.idx_mod,axis=0)
        self.vs = np.delete(self.vs,self.idx_mod)
        if anisotropic:
            self.psi2amp = np.delete(self.psi2amp,self.idx_mod)
            self.psi2 = np.delete(self.psi2,self.idx_mod)
            # psi2xsum and psi2ysum remain the same since psi2amp is zero
        
        # get the velocity that is at the grid location after removing the point
        idx_loc = weights.argmax()
        vs_location = self.vssum[idx_loc]/self.weightsum[idx_loc]
        
        # now compare the velocity of the removed point with the velocity
        # at the empty spot (inverse birth operation)
        if self.propvelstd_dimchange != 'uniform':
            self.prop_dv = vs_remove - vs_location
        else:
            self.prop_dv = 0.
            
        return True
                    
            
    def move_point(self,propmovestd,index=None,backup=True):

        self.action = 'move'
        
        if index is None:
            index = np.random.randint(0,len(self.points))
        self.idx_mod = index

        oldxy = self.points[index].copy()    
        newxy = np.random.normal(loc=oldxy,scale=propmovestd,size=2)
        if (newxy[0]>self.maxx or newxy[0]<self.minx or
            newxy[1]>self.maxy or newxy[1]<self.miny):
            return (np.nan,np.nan)        

        if backup:
            self.backup_mod()

        # now set the new point position        
        self.points[index] = newxy
            
        # as in a death operation, first of all, remove point contribution
        weights1 = self.get_weights(oldxy)
        self.vssum -= self.vs[self.idx_mod]*weights1
        self.weightsum -= weights1        
        # as in birth operation, add contribution from point at new position
        weights2 = self.get_weights(newxy)
        self.vssum += self.vs[self.idx_mod]*weights2
        self.weightsum += weights2
        
        if self.anisotropic:
            psi2x = self.psi2amp[index]*np.cos(2*self.psi2[index])
            psi2y = self.psi2amp[index]*np.sin(2*self.psi2[index])
            self.psi2xsum = self.psi2xsum - psi2x*weights1 + psi2x*weights2
            self.psi2ysum = self.psi2ysum - psi2y*weights1 + psi2y*weights2
        
        return (oldxy,newxy)
    
        
    def get_prior_proposal_ratio(self):
        
        # this function returns the prior ratio times the proposal ratio which
        # is needed to calculate the acceptance probability  
        if 'update' in self.action or self.action == 'move' or self.action == 'anisotropy_birth_death':
            # is always log(1)=0, unless delayed rejegion which is currently
            # included in the main script
            return 0
        elif self.action == 'birth':
            if self.anisotropic:
                aniso_factor = np.log(len(self.points)/(len(self.points)+1))
            else:
                aniso_factor = 0.
            if self.propvelstd_dimchange == 'uniform':
                # if we draw from a uniform prior, everything cancels out
                return aniso_factor + 0
            else:
                # see for example equation A.34 of the PhD thesis of Thomas Bodin
                return (aniso_factor + 
                    np.log(self.propvelstd_dimchange*np.sqrt(2.*np.pi) / self.velrange) +  
                    (self.prop_dv**2 / (2*self.propvelstd_dimchange**2)))
        elif self.action == 'death':
            if self.anisotropic:
                aniso_factor = np.log(len(self.points)/(len(self.points)-1))
            else:
                aniso_factor = 0.
            if self.propvelstd_dimchange == 'uniform':
                return aniso_factor + 0
            else:
                return ( aniso_factor + 
                    np.log(self.velrange/(self.propvelstd_dimchange*np.sqrt(2.*np.pi))) -
                    (self.prop_dv**2 / (2*self.propvelstd_dimchange**2)))
        else:
            raise Exception("action undefined")
    
    def get_model(self,points=None,vs=None,psi2amp=None,psi2=None,anisotropic=False):
        
        if points is None:
            points = self.points
        if vs is None:
            vs = self.vs
        if anisotropic:
            if psi2amp is None:
                psi2amp = self.psi2amp 
            if psi2 is None:
                psi2 = self.psi2
            psi2x = psi2amp*np.cos(2*psi2)
            psi2y = psi2amp*np.sin(2*psi2)
            psi2xsum = np.zeros(len(self.gridpoints))
            psi2ysum = np.zeros(len(self.gridpoints))
            
        vssum = np.zeros(len(self.gridpoints))
        weightsum = np.zeros(len(self.gridpoints))
        for i in range(len(points)):
            weights = self.get_weights(points[i])
            vssum += vs[i]*weights
            weightsum += weights
            if anisotropic:
                psi2xsum += psi2x[i]*weights
                psi2ysum += psi2y[i]*weights
        
        slowfield = 1./np.around(vssum/(weightsum+1e-300),2)
        if anisotropic:
            psi2ampfield = np.around(np.sqrt((psi2xsum/(weightsum+1e-300))**2 +
                                             (psi2ysum/(weightsum+1e-300))**2),3)
            psi2field = np.around(0.5*np.arctan2(psi2ysum/(weightsum+1e-300),
                                                 psi2xsum/(weightsum+1e-300)),2)
            return slowfield,psi2ampfield,psi2field
        else:
            return slowfield
    
    
    def update_model(self,fields=None,anisotropic=False):
        
        if anisotropic:
            slowfield_old, psi2ampfield_old, psi2field_old = fields
        else:
            slowfield_old = fields
            
        slowfield = 1./np.around(self.vssum/(self.weightsum+1e-300),2)
        if anisotropic:
            psi2ampfield = np.around(np.sqrt((self.psi2xsum/(self.weightsum+1e-300))**2 +
                                             (self.psi2ysum/(self.weightsum+1e-300))**2),3)
            psi2field = np.around(0.5*np.arctan2(self.psi2ysum/(self.weightsum+1e-300),
                                                 self.psi2xsum/(self.weightsum+1e-300)),2)
        self.idx_mod_gpts = np.where(slowfield!=slowfield_old)[0]
        
        if anisotropic:
            return slowfield,psi2ampfield,psi2field
        else:
            return slowfield

    def get_weights(self,point):
        # gaussian decay with distance of the weights
        dists = distance.cdist(self.gridpoints,point[None],metric=self.metric).flatten()
        return np.exp(-0.5 * np.square(dists) / np.square(self.smooth_radius))

    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        self.vs_backup = self.vs.copy()
        self.vssum_backup = self.vssum.copy()
        self.weightsum_backup = self.weightsum.copy()
        if self.psi2amp is not None:
            self.psi2amp_backup = self.psi2amp.copy()
            self.psi2xsum_backup = self.psi2xsum.copy()
            self.psi2ysum_backup = self.psi2ysum.copy()
        if self.psi2 is not None:
            self.psi2_backup = self.psi2.copy()
        
        
    def reject_mod(self):
        
        if self.points_backup is not None:
            self.points = self.points_backup
        if self.vs_backup is not None:
            self.vs = self.vs_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        if self.vssum_backup is not None:
            self.vssum = self.vssum_backup
        if self.weightsum_backup is not None:
            self.weightsum = self.weightsum_backup
        if self.psi2xsum_backup is not None:
            self.psi2xsum = self.psi2xsum_backup
        if self.psi2ysum_backup is not None:
            self.psi2ysum = self.psi2ysum_backup
            
        self.reset_backup()
        
    def accept_mod(self,selfcheck=False):
                    
        self.reset_backup()
        
    def reset_backup(self):
        
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = 1e99
        self.idx_mod = 1e99
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.weightsum_backup = None
        self.vssum_backup = None
        self.psi2xsum_backup = None
        self.psi2ysum_backup = None
        
        
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):
        
        from scipy.spatial import delaunay_plot_2d
        tri = Delaunay(self.points)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _ = delaunay_plot_2d(tri,ax=ax)
        ax.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.')
        if idx_mod_gpts is not None:
            ax.plot(self.gridpoints[idx_mod_gpts,0],self.gridpoints[idx_mod_gpts,1],'rx',zorder=3)
        cbar = ax.scatter(self.points[:,0],self.points[:,1],c=self.vs,s=40,zorder=3)
        if idx_neighbor_points is not None:
            ax.scatter(self.points[:,0],self.points[:,1],c=self.vs,s=60,zorder=3)
        plt.colorbar(cbar)
        plt.show()  
        
##############################################################################
##############################################################################
##############################################################################

class linear_interpolation(object):
    
    def __init__(self,gridpoints,shape,velmin,velmax,init_no_points,
                 anisotropic=False,psi2ampmax=None,psi2ampmin=None): 
        
        self.gridpoints = gridpoints
        self.shape = shape
        self.anisotropic = anisotropic
        
        self.minx = np.min(self.gridpoints[:,0])
        self.maxx = np.max(self.gridpoints[:,0])
        self.miny = np.min(self.gridpoints[:,1])
        self.maxy = np.max(self.gridpoints[:,1])
        self.xrange = self.maxx-self.minx
        self.yrange = self.maxy-self.miny
        
        self.velmin = velmin
        self.velmax = velmax
        self.velrange = velmax-velmin
        
        # add a bit extra otherwise the triangles at the borders of the model
        # domain are very flat
        self.minx -= 0.02*self.xrange
        self.maxx += 0.02*self.xrange
        self.miny -= 0.02*self.yrange
        self.maxy += 0.02*self.yrange
        
        # INITIALIZE PARAMETERIZATION
        # for the lin. interpolation, it is important to always have points at
        # the model edges.
        # it is also better if the points are not exactly at the gridpointlocations
        # the Delaunay triangulation may then not uniquely attribute the gridpoint
        if init_no_points < 4:
            print("set initial number of points to minimum requirement of 4.")
            init_no_points = 4
        points_x = np.random.uniform(low=self.minx,
                                     high=self.maxx,
                                     size=init_no_points)
        points_y = np.random.uniform(low=self.miny,
                                     high=self.maxy,
                                     size=init_no_points)
        xgridspacing = np.max(np.diff(self.gridpoints[:,0]))
        ygridspacing = np.max(np.diff(self.gridpoints[:,1]))   
        points_x[:4] = np.array([self.minx-xgridspacing/10.,
                                 self.minx-xgridspacing/10.,
                                 self.maxx+xgridspacing/10.,
                                 self.maxx+xgridspacing/10.])
        points_y[:4] = np.array([self.miny-ygridspacing/10.,
                                 self.maxy+ygridspacing/10.,
                                 self.miny-ygridspacing/10.,
                                 self.maxy+ygridspacing/10.])
        self.points = np.column_stack((points_x,points_y))
        if anisotropic:
            self.psi2amp = np.zeros(len(self.points))
            self.psi2 = np.zeros(len(self.points))
            self.psi2ampmin = psi2ampmin
            self.psi2ampmax = psi2ampmax
        else:
            self.psi2amp = None
            self.psi2 = None
        
        self.gpts_dict, self.tri = self.get_gpts_dictionary()

        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.idx_subset_points = None
        self.prop_tri = None
        self.propvelstd_dimchange = 'uniform' # will be adapted when running
        self.psi2amp_backup = None
        self.psi2_backup = None
                    

    def psi2amp_update(self,idx,delta,backup=True):
        
        if backup:
            self.psi2amp_backup = self.psi2amp.copy()
            
        self.psi2amp[idx] += delta
        self.action='psi2amp_update'
        self.idx_mod = idx
        
        if self.psi2amp[idx]<self.psi2ampmin or self.psi2amp[idx]>self.psi2ampmax:
            return False
        if (self.vs[idx]*(1+self.psi2amp[idx]) > self.velmax or 
            self.vs[idx]*(1-self.psi2amp[idx]) < self.velmin):
            return False
        
        self.idx_subset_points = np.append(idx,self.get_neighbors(idx))
        self.idx_mod_gpts = self.gpts_dict[idx]
        
        return True
    
        
    def psi2_update(self,idx,delta,backup=True):
        
        if backup:
            self.psi2_backup = self.psi2.copy()
            
        # modulo makes sure it's always in the 0-2pi range
        # the Python modulo convention also correctly treats negative angles
        self.psi2[idx] = (self.psi2[idx]+delta)%(2*np.pi)
        if self.action=='psi2amp_update':
            self.action='anisotropy_birth_death'
        else:
            self.action='psi2_update'
        self.idx_mod = idx
        
        self.idx_subset_points = np.append(idx,self.get_neighbors(idx))
        self.idx_mod_gpts = self.gpts_dict[idx]
        
        return True
    

    def vs_update(self,idx,dvs,backup=True):
        
        if backup:
            self.vs_backup = self.vs.copy()
            
        self.action='velocity_update'
            
        self.vs[idx] += dvs
        if self.vs[idx]>self.velmax or self.vs[idx]<self.velmin:
            return False
        if self.psi2amp is not None:
            if (self.vs[idx]*(1+self.psi2amp[idx]) > self.velmax or 
                self.vs[idx]*(1-self.psi2amp[idx]) < self.velmin):
                return False
        
        self.idx_subset_points = np.append(idx,self.get_neighbors(idx))

        self.idx_mod_gpts = self.gpts_dict[idx]
        self.idx_mod = idx
        
        return True
        
        
    def add_point(self,anisotropic=False,birth_prop='uniform',backup=True):
        
        if backup:
            self.backup_mod()
            
        self.action='birth'
        self.idx_mod = len(self.vs)
        
        # propose new point position
        # trying to avoid flat triangles
        prop_point_x = np.random.uniform(self.minx,self.maxx)
        prop_point_y = np.random.uniform(self.miny,self.maxy)
        idx_simplex = self.tri.find_simplex((prop_point_x,prop_point_y))
        if idx_simplex > -1:
            triangle_indices = self.tri.simplices[idx_simplex]
            dists = distance_matrix(self.points[triangle_indices[:2]],
                                    self.points[triangle_indices[1:]])
            ind = np.unravel_index(np.argmax(dists, axis=None), dists.shape)
            idx_point1 = triangle_indices[ind[0]]
            idx_point2 = triangle_indices[ind[1]+1]
            prop_point_x,prop_point_y = np.mean((self.points[idx_point1],
                                                 self.points[idx_point2]),axis=0)
        point = np.hstack((prop_point_x,prop_point_y))

        self.points = np.vstack((self.points,point))
        
        #### finding the updated model region ###
        # getting all neighbours of new point
        self.idx_subset_points = self.get_neighbors(self.idx_mod,
                                                    points=self.points)

        if birth_prop=='uniform':
            vs_birth = np.random.uniform(self.velmin,self.velmax)
        else:
            self.prop_dv = birth_prop * np.random.normal(loc=0.0,scale=1.0)
            dists = np.sqrt(np.sum((self.points[self.idx_subset_points,:2] -
                                    point[:2])**2,axis=1))
            vs_birth = np.average(self.vs[self.idx_subset_points],
                                  weights=1./dists) + self.prop_dv
            if vs_birth > self.velmax or vs_birth < self.velmin:
                return False
            
        self.vs = np.append(self.vs,vs_birth)
        if anisotropic:
            # newborn cell has not anisotropy
            self.psi2amp = np.append(self.psi2amp,0.)
            self.psi2 = np.append(self.psi2,0.)
            
        # interpolate new velocities only in the area affected by the change
        idx_points_subset = np.append(self.idx_mod,self.idx_subset_points)

        # getting all indices of all gridpoints of the neighboring points
        # (this automatically includes the gridpoints in the new area)
        idx_modified_gridpoints_all = np.unique(np.hstack(
            itemgetter(*self.idx_subset_points)(self.gpts_dict)))
        
        subset_tri = Delaunay(self.points[idx_points_subset,:2])
        # we can reduce this to the gridpoints that are actually related to the new point position
        simplices = subset_tri.find_simplex(self.gridpoints[idx_modified_gridpoints_all])
        valid_simplices = simplices>=0
        points = subset_tri.simplices[simplices[valid_simplices]]
        # get all gridpoints that belong to point 0 (0 means the first
        # element in the idx_points_subset, i.e. the birth point)
        self.idx_mod_gpts = idx_modified_gridpoints_all[valid_simplices][
            np.where(np.sum(np.isin(points,0),axis=1)>0)[0]]
        
        # for the prior_proposal_ratio calculation:
        self.propvelstd_dimchange = birth_prop
        
        return True


    def remove_point(self,anisotropic=False,backup=True):
        
        if backup:
            self.backup_mod()
            
        self.action='death'
            
        if anisotropic:
            # choose only points without anisotropy, do not remove 4 corner points
            ind_pnts = np.where(self.psi2amp[4:] == 0.)[0]+4
            if len(ind_pnts) > 0:
                self.idx_mod = np.random.choice(ind_pnts)
            else:
                return False
        else:
            # choose point to remove randomly, do not remove corner points
            self.idx_mod = np.random.randint(4,len(self.points))
        
        pnt_remove = self.points[self.idx_mod]
        vs_remove = self.vs[self.idx_mod]
        
        # getting all neighbor points of removed vertice
        self.idx_subset_points = self.get_neighbors(self.idx_mod)

        # getting indices of all gridpoints that will change their velocity
        self.idx_mod_gpts = self.gpts_dict[self.idx_mod]
        
        self.points = np.delete(self.points,self.idx_mod,axis=0)
        self.vs = np.delete(self.vs,self.idx_mod)
        if anisotropic:
            self.psi2amp = np.delete(self.psi2amp,self.idx_mod)
            self.psi2 = np.delete(self.psi2,self.idx_mod)
        
        # gives the neighbor indices of the removed point. correcting indices
        # for the neighbors after removal
        self.idx_subset_points[self.idx_subset_points>self.idx_mod] -= 1
        
        # now compare the velocity of the removed point with the velocity
        # at the empty spot (inverse birth operation)
        # the given idx_subset_points are for the NEW point indices of
        # the neighbors (after point removal)
        if len(self.idx_subset_points)>0:
            dists = np.sqrt(np.sum((self.points[self.idx_subset_points,:2] -
                                    pnt_remove[:2])**2,axis=1))
            self.prop_dv = (vs_remove - 
                            np.average(self.vs[self.idx_subset_points],
                                       weights=1./dists) )
        else:
            self.prop_dv = 0.
        
        return True
                    
            
    def move_point(self,propmovestd,index=None,backup=True):

        if backup:
            self.backup_mod()
        
        self.action = 'move'
        
        if index is None:
            # don't move the points at the edges
            index = np.random.randint(4,len(self.points))
        self.idx_mod = index

        oldxy = self.points[index].copy()
        
        dx = np.random.normal(loc=0.0,scale=propmovestd,size=2)
        newxy = oldxy + dx
        
        if (newxy[0]>self.maxx or newxy[0]<self.minx or
            newxy[1]>self.maxy or newxy[1]<self.miny):
            return (np.nan,np.nan)
                    
        old_neighbors = self.get_neighbors(index)
        self.points[index] = newxy
        new_neighbors = self.get_neighbors(index,points=self.points)
        
        idx_neighbor_indices = np.unique(np.hstack((old_neighbors,new_neighbors)))
        
        # indices of all points that might be affected by the change
        # (old neighbors, new neighbors and move point itself) 
        # make sure the idx_move index is the first in the array
        self.idx_subset_points = np.append(index,idx_neighbor_indices)       
        subset_tri = Delaunay(self.points[self.idx_subset_points])

        # indices of all gridpoints that may be affected by this
        # operation, i.e. gridpoints in idx_move and in new neighbors
        idx_modified_gridpoints_all = np.unique(np.hstack(itemgetter(*np.hstack((new_neighbors,index)))(self.gpts_dict)))
        
        # we can reduce this to the gridpoints that are actually related to the new point position
        if np.array_equal(np.sort(old_neighbors),np.sort(new_neighbors)):
            self.idx_mod_gpts = self.gpts_dict[index]
        else:
            simplices = subset_tri.find_simplex(self.gridpoints[idx_modified_gridpoints_all])
            valid_simplices = simplices>=0
            points = subset_tri.simplices[simplices[valid_simplices]]
            idx_mod_gpts_newpos = idx_modified_gridpoints_all[valid_simplices][np.where(np.sum(np.isin(points,0),axis=1)>0)[0]]
            idx_mod_gpts_oldpos = self.gpts_dict[index]
            self.idx_mod_gpts = np.unique(np.append(idx_mod_gpts_newpos,
                                                    idx_mod_gpts_oldpos))
        
        return (oldxy,newxy)
        

    def get_neighbors(self,ind,points=None):
                
        if points is None:
            intptr,neighbor_indices = self.tri.vertex_neighbor_vertices
        else:
            self.prop_tri = Delaunay(points)
            if len(self.prop_tri.coplanar)>0:
                print("Warning: coplanar points!")
            intptr,neighbor_indices = self.prop_tri.vertex_neighbor_vertices 

        neighbors = neighbor_indices[intptr[ind]:intptr[ind+1]]
          
        # return a copy. This is important because otherwise, if neighbors is
        # changed, it changes also the neighbor_indices of the Delaunay
        # triangulation (self.tri or self.prop_tri) itself. 
        return neighbors.copy()
    
    
    def get_prior_proposal_ratio(self):
        
        # this function returns the prior ratio times the proposal ratio which
        # is needed to calculate the acceptance probability  
        if 'update' in self.action or self.action == 'move' or self.action == 'anisotropy_birth_death':
            # is always log(1)=0, unless delayed rejegion which is currently
            # included in the main script
            return 0
        elif self.action == 'birth':
            if self.anisotropic:
                aniso_factor = np.log(len(self.points)/(len(self.points)+1))
            else:
                aniso_factor = 0.
            if self.propvelstd_dimchange == 'uniform':
                # if we draw from a uniform prior, everything cancels out
                return aniso_factor + 0
            else:
                # see for example equation A.34 of the PhD thesis of Thomas Bodin
                return (aniso_factor + 
                    np.log(self.propvelstd_dimchange*np.sqrt(2.*np.pi) / self.velrange) +  
                    (self.prop_dv**2 / (2*self.propvelstd_dimchange**2)))
        elif self.action == 'death':
            if self.anisotropic:
                aniso_factor = np.log(len(self.points)/(len(self.points)-1))
            else:
                aniso_factor = 0.
            if self.propvelstd_dimchange == 'uniform':
                return aniso_factor + 0
            else:
                return ( aniso_factor + 
                    np.log(self.velrange/(self.propvelstd_dimchange*np.sqrt(2.*np.pi))) -
                    (self.prop_dv**2 / (2*self.propvelstd_dimchange**2)))
        else:
            raise Exception("action undefined")            
        
        
    def get_model(self,points=None,vs=None,psi2amp=None,psi2=None,anisotropic=False):
        
        if points is None:
            points = self.points
        if vs is None:
            vs = self.vs
            
        if anisotropic:
            if psi2amp is None:
                psi2amp = self.psi2amp 
            if psi2 is None:
                psi2 = self.psi2
            psi2x = psi2amp*np.cos(2*psi2)
            psi2y = psi2amp*np.sin(2*psi2)
            func = NearestNDInterpolator(points,np.column_stack((vs,psi2x,psi2y)))                  
            field = func(self.gridpoints)
            slowfield = 1./field[:,0]
            psi2xfield = field[:,1]
            psi2yfield = field[:,2]
            psi2ampfield = np.sqrt(psi2xfield**2+psi2yfield**2)
            psi2field = 0.5*np.arctan2(psi2yfield,psi2xfield)
            return (slowfield,psi2ampfield,psi2field)
        
        else:
            func = LinearNDInterpolator(points,vs)
            slowfield = 1./func(self.gridpoints)
            return  slowfield
    
    
    def update_model(self,fields=None,anisotropic=False):
        
        if anisotropic:
            slowfield, psi2amp, psi2 = fields
            psi2amp_cp = psi2amp.copy()
            psi2_cp = psi2.copy()
        else:
            slowfield = fields
        slowfield_cp = slowfield.copy()

        if self.action=='birth':
            # add the birth point (only neighbors are in the original self.idx_subset_points)
            idx_subset_points = np.append(self.idx_mod,self.idx_subset_points)
        else:
            idx_subset_points = self.idx_subset_points
                    
        if len(self.idx_mod_gpts) == 0:
            if anisotropic:
                return slowfield_cp,psi2amp_cp,psi2_cp
            else:
                return slowfield_cp
            
        if anisotropic:
            psi2x = self.psi2amp[idx_subset_points]*np.cos(2*self.psi2[idx_subset_points])
            psi2y = self.psi2amp[idx_subset_points]*np.sin(2*self.psi2[idx_subset_points])
            # interpolate only in the region where the point was modified
            func = LinearNDInterpolator(self.points[idx_subset_points],
                                        np.column_stack((self.vs[idx_subset_points],
                                                         psi2x,psi2y)))
            # only evaluate the interpolation function at the modified gridpoints
            field = func(self.gridpoints[self.idx_mod_gpts])
            # fill in new fields
            slowfield_cp[self.idx_mod_gpts] = 1./field[:,0]
            psi2amp_cp[self.idx_mod_gpts] = np.sqrt(field[:,1]**2+field[:,2]**2)
            psi2_cp[self.idx_mod_gpts] = 0.5*np.arctan2(field[:,2],field[:,1]) 
            return slowfield_cp,psi2amp_cp,psi2_cp
        else:
            # interpolate only in the region where the point was modified
            func = LinearNDInterpolator(self.points[idx_subset_points],
                                        self.vs[idx_subset_points])
            # only evaluate the interpolation function at the modified gridpoints
            vs = func(self.gridpoints[self.idx_mod_gpts])
            slowfield_cp[self.idx_mod_gpts] = 1./vs
            return slowfield_cp
                      
        
    def get_gpts_dictionary(self,points=None,gridpoints=None):
        
        gpts_dict = {}
        
        if points is None:
            points = self.points
        if gridpoints is None:
            gridpoints = self.gridpoints
            
        tri = Delaunay(points)
        simplices = tri.find_simplex(gridpoints)
        vertices = tri.simplices[simplices]
        for i in range(len(points)):
            gpts_dict[i] = np.where(np.sum(np.isin(vertices,i),axis=1)>0)[0]

        return gpts_dict, tri
        
        
    def update_gpts_dict(self,selfcheck=False):
        
        if (self.action is None or self.idx_mod_gpts is None or 
            self.idx_subset_points is None):
            raise Exception("cannot update gpts dictionary!")
        
        if self.action=='birth':
            idx_modified_gridpoints_all = np.unique(np.hstack((
                itemgetter(*self.idx_subset_points)(self.gpts_dict))))
            # add the birth point (only neighbors are in the original self.idx_subset_points)
            idx_subset_points = np.append(self.idx_mod,self.idx_subset_points)

        elif self.action=='death':
            idx_subset_points = self.idx_subset_points
            # we have to get the entries from the gpts dict that are still
            # using the old indices
            idx_subset_points[idx_subset_points>=self.idx_mod] += 1
            idx_modified_gridpoints_all = np.unique(np.hstack((
                itemgetter(*idx_subset_points)(self.gpts_dict))))
            # now we go back to the new indices
            idx_subset_points[idx_subset_points>self.idx_mod] -= 1
            # first, we correct the vertex indices, because idx_death is missing now
            for idx in range(self.idx_mod,len(self.points)):
                self.gpts_dict[idx] = self.gpts_dict[idx+1]
            # the dictionary is one entry shorter now
            del self.gpts_dict[len(self.points)] # remove last entry

        elif self.action=='move':
            idx_subset_points = self.idx_subset_points
            idx_modified_gridpoints_all = np.unique(np.hstack((
                itemgetter(*idx_subset_points)(self.gpts_dict))))

        if self.action in ['birth','death','move']:
            if self.prop_tri is None:
                self.tri = Delaunay(self.points)
            else:
                self.tri = self.prop_tri
            simplices = self.tri.find_simplex(self.gridpoints[idx_modified_gridpoints_all])
            vertices = self.tri.simplices[simplices]
            for i in idx_subset_points:
                self.gpts_dict[i] = idx_modified_gridpoints_all[np.where(np.sum(np.isin(vertices,i),axis=1)>0)[0]]
            
            
        if selfcheck:
            gpts_dict_test, tri = self.get_gpts_dictionary()
            for entry in gpts_dict_test:
                if not np.array_equal(np.sort(gpts_dict_test[entry]),np.sort(self.gpts_dict[entry])):
                    print(entry)
                    raise Exception(f"gpts dict not right after {self.action} operation")
        


    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        self.vs_backup = self.vs.copy()
        if self.anisotropic:
            self.psi2_backup = self.psi2.copy()
            self.psi2amp_backup = self.psi2amp.copy()
        
        
    def reject_mod(self):

        if self.points_backup is not None:
            self.points = self.points_backup
        if self.vs_backup is not None:
            self.vs = self.vs_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
            
        self.reset_backup()
        
        
    def accept_mod(self,selfcheck=False):
        
        self.update_gpts_dict(selfcheck=selfcheck)
        self.reset_backup()
        
        
    def reset_backup(self):
        
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.idx_subset_points = None
        self.prop_tri = None
        self.prop_dv = None
        self.psi2_backup = None
        self.psi2amp_backup = None
        
        
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):
        
        from scipy.spatial import delaunay_plot_2d
        tri = Delaunay(self.points)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _ = delaunay_plot_2d(tri,ax=ax)
        ax.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.')
        if idx_mod_gpts is not None:
            ax.plot(self.gridpoints[idx_mod_gpts,0],self.gridpoints[idx_mod_gpts,1],'rx',zorder=3)
        cbar = ax.scatter(self.points[:,0],self.points[:,1],c=self.vs,s=40,zorder=3)
        if idx_neighbor_points is not None:
            ax.scatter(self.points[idx_neighbor_points,0],self.points[idx_neighbor_points,1],
                       c=self.vs[idx_neighbor_points],s=100,zorder=3)
            ax.scatter(self.points[idx_neighbor_points,0],self.points[idx_neighbor_points,1],
                       c='red',s=10,zorder=3)
        plt.colorbar(cbar)
        plt.show()
        
        
        from scipy.spatial import delaunay_plot_2d
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _ = delaunay_plot_2d(self.tri,ax=ax)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        _ = delaunay_plot_2d(self.prop_tri,ax=ax)
        plt.show()
        

##############################################################################
##############################################################################
##############################################################################

class wavelets(object):
    
    
    def __init__(self,gridpoints,shape,velmin,velmax,init_no_coeffs=None): 
        
        
        self.gridpoints = gridpoints
        self.shape = shape
        
        self.wavelet = 'bior4.4'#'bior4.4'
        decomposition_level = int(np.log2(np.min(self.shape)))
        # minlevel gives the minimum level of the decomposition tree in which
        # the birth/death operations may take place. nodes smaller than
        # minlevel are always filled. Normally, minlevel=0 (no restriction)
        self.minlevel=0
        
        # # wavelet parameterization works with different levels from very rough
        # # to successively finer scales
        # subshape = shape
        # shapes = []
        # while True:
        #     subshape = (int(subshape[0]/2.)+subshape[0]%2,
        #                 int(subshape[1]/2.)+subshape[1]%2)
        #     shapes.append(subshape)
        #     if subshape[0]==1 or subshape[1]==1:
        #         break
        
        self.velmax = self.minmod = velmax
        self.velmin = self.maxmod = velmin
        
        """
        velmean = (self.velmax+self.velmin) / 2.
        velamp = (self.velmax-self.velmin) / 2.
        self.coeff_range = {}
        X = np.reshape(self.gridpoints[:,0],self.shape)
        xrange = X[0,-1]-X[0,0]
        kx = 0.01/xrange
        kxmax = 1/(2*(X[0,1]-X[0,0])) # Nyquist wavenumber
        Y = np.reshape(self.gridpoints[:,1],self.shape)
        yrange = Y[-1,0]-Y[0,0]
        ky = 0.01/yrange
        kymax = 1/(2*(Y[1,0]-Y[0,0]))
        while kx<kxmax and ky<kymax:
            start_mod = velmean + velamp * (np.cos(2*np.pi*kx*X + 
                                                    np.random.uniform(0,2*np.pi)))
                                            # np.cos(2*np.pi*ky*Y +
                                            #        np.random.uniform(0,2*np.pi)))
            if np.max(start_mod)>self.velmax or np.min(start_mod)<self.velmin:
                raise Exception()
            coeffs = pywt.wavedec2(start_mod, self.wavelet,
                                    level=decomposition_level,
                                    mode='periodization')

            for level in range(len(coeffs)):
                try:
                    self.coeff_range[level]
                except:
                    self.coeff_range[level] = {}
                    self.coeff_range[level]["max"] = -1e99
                    self.coeff_range[level]["min"] = 1e99
                if level==0:
                    self.coeff_range[0]['max'] = np.max([
                        self.coeff_range[0]['max'],np.max(coeffs[level])])
                    self.coeff_range[0]['min'] = np.min([
                        self.coeff_range[0]['min'],np.min(coeffs[level])])
                else:
                    for direction in [0,1,2]:
                        self.coeff_range[level]['max'] = np.max([
                            self.coeff_range[level]['max'],
                            np.max(coeffs[level][direction])])
                        self.coeff_range[level]['min'] = np.min([
                            self.coeff_range[level]['min'],
                            np.min(coeffs[level][direction])]) 
            kx*=1.1
            ky*=1.1
            
        for level in self.coeff_range:
            if level>0:
                self.coeff_range[level]['max'] = np.around(np.max([
                    self.coeff_range[level]['max'],
                    np.abs(self.coeff_range[level]['min'])]),3)
                self.coeff_range[level]['min'] = -self.coeff_range[level]['max']
        """
        
        # test_coeffs = deepcopy(coeffs)
        # for level in range(len(coeffs)):
        #     if level==0:
        #         test_coeffs[level][:] = mmax#self.coeff_range[level]["max"]
        #     else:
        #         for direction in [0,1,2]:
        #             test_coeffs[level][direction][:] = 0.
        #             test_coeffs[level][direction][:] = self.coeff_range[level]["max"]
                    
        
        # #%%
        # mode = "symmetric"
        # coeffs = pywt.wavedec2(np.ones(self.shape)*3.5, self.wavelet,level=None,
        #                             mode=mode)
        # for level,lvlcoeffs in enumerate(coeffs):
        #     print(level,np.shape(lvlcoeffs))
        #     if level==0:
        #         coeffs[level][:] = np.mean(coeffs[level])
        #     else:
        #         coeffs[level][0][:] = coeffs[level][1][:] = coeffs[level][2][:] = 0.
        # coeffs[0][5,5] = 29.
        # velfield = pywt.waverec2(coeffs, self.wavelet, mode=mode)
        # plt.figure()
        # plt.pcolormesh(velfield)
        # plt.colorbar()
        # plt.show()
        # #%%
        
        start_mod = np.random.uniform(low=self.velmin,high=self.velmax,
                                      size=self.shape)
        self.coeffs = pywt.wavedec2(start_mod, self.wavelet,
                                    level=decomposition_level,
                                    mode='periodization')
        mincoeff = np.min(pywt.wavedec2(np.ones_like(start_mod)*self.velmin, 
                                        self.wavelet,level=decomposition_level,
                                        mode='periodization')[0])
        maxcoeff = np.max(pywt.wavedec2(np.ones_like(start_mod)*self.velmax, 
                                        self.wavelet,level=decomposition_level,
                                        mode='periodization')[0])

        self.points = []
        for level in range(len(self.coeffs)):
            
            if level==0:
                shape = self.coeffs[level].shape
            elif level==1:
                shape = (3,) + shape
            else:
                shape = (3, shape[1]*2,shape[2]*2)
            if np.shape(self.coeffs[level])!=shape:
                raise Exception("bad tree structure")
                
            if level<self.minlevel or level==0:
                for direction in range(3):
                    if level==0 and direction>0:
                        continue
                    elif level==0:
                        direction=None
                    indices = np.column_stack(np.unravel_index(
                        np.arange(len(self.coeffs[level][direction].flatten())),
                        shape[-2:]))
                    if level==0:
                        direction=0
                    self.points.append(np.column_stack((np.ones(len(indices))*level,
                                                        np.ones(len(indices))*direction,
                                                        indices,
                                                        np.zeros(len(indices)))))
            else:
                self.coeffs[level][0][:] = self.coeffs[level][1][:] = self.coeffs[level][2][:] = 0.
        
        # points has 5 columns: level, direction (hor,ver,diag), yind, xind, isleaf
        # at level=0, the direction does not apply
        self.points = np.vstack(self.points).astype(int)
            
        # number of decomposition levels
        self.levels = level+1
        
        self.coeff_range = {}
        for level in range(self.levels):
            self.coeff_range[level] = {}
            self.coeff_range[level]['max'] = 0.
            self.coeff_range[level]['min'] = 0.
        self.coeff_range[0]['min'] = mincoeff
        self.coeff_range[0]['max'] = maxcoeff
        self.update_coeff_range()
        
        self.maxlevel = self.levels-1
        print("Warning, setting maximum level to",self.maxlevel)#self.levels-1)
        
        # potential children includes all new nodes that can be chosen from
        # in a birth operation (children of currently active nodes)
        self.potential_children = []
        for point in self.points:
            children = self.get_children(point)
            for child in children:
                idx1 = np.where((child[:4]==self.points[:,:4]).all(axis=1))[0]
                if len(idx1)==0:
                    self.potential_children.append(child)
        if len(self.potential_children)>0:
            self.potential_children = np.vstack(self.potential_children)
            self.potential_children = np.unique(self.potential_children,axis=0)
        else:
            self.potential_children = np.empty((0,5),dtype=int)
                
        # dictionary that stores the number of possible arrangements of nodes
        # for a certain tree structure
        try:
            with open("tree_dict.pkl","rb") as f:
                self.D = pickle.load(f)
            print("read tree dictionary from tree_dict.pkl")
        except:
            self.D = {}
        # getting the tree structure. the first number in the array gives the
        # total number of root nodes. The following number give the number of
        # children for each root node.
        # In a binary tree that has a maximum depth of 3, this would look like
        # self.tree = [1,2,2,2]
        # However, in this case we have already an array at the root, e.g.
        # 6*7 elements, resulting in 42 root nodes. Each root node has 3 child-
        # ren, one for each direction. For each direction, each node has 4
        # children, on the succesively finer grids
        # self.tree = [42, 3, 4, 4, 4]
        self.full_tree = np.array([len(self.coeffs[0].flatten()),3] + 
                                  (self.levels-2)*[4])
        # adapt tree to the minlevel and the maxlevel
        self.tree = np.append(np.product(self.full_tree[:self.minlevel+1]),
                              self.full_tree[self.minlevel+1:self.maxlevel+1])
        self.kmin = 0
        for i in range(self.minlevel):
            self.kmin += np.product(self.full_tree[:i+1])
        
        self.points_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp = None
        self.psi2 = None
        self.psi2amp_backup = None
        self.psi2_backup = None  
        self.coeffs_backup = None
        self.potential_children_backup = None
        

        # add points until the number of initial coefficients is reached
        if init_no_coeffs is not None:
            init_coeffs = np.max([0,init_no_coeffs-len(self.points)])
        else:
            init_coeffs = 0
        for i in range(init_coeffs):
            check_birth = self.add_point()
            if check_birth:
                slow_prop = self.update_model()
                if np.isnan(slow_prop).any():
                    self.reject_mod()
                else:
                    self.accept_mod()
        
        
        #slowfield = self.get_model()
        #if (1./slowfield > velmax).any() or (1./slowfield < velmin).any():
        #    raise Exception("bad starting model")        
        
        self.acceptance_rate = {}
        self.accepted_steps = {}
        self.rejected_steps = {}
        for level in range(self.levels):
            self.acceptance_rate[level] = np.zeros(100,dtype=int)
            self.accepted_steps[level] = 0
            self.rejected_steps[level] = 0
        
        # # array of decomposed wavelet coefficients
        # self.coeffs = []
        # for i,shape in enumerate(shapes[::-1]):
        #     if i==0:
        #         self.coeffs.append(np.zeros(shape))
        #     self.coeffs.append((np.zeros(shape),np.zeros(shape),np.zeros(shape)))
        
     
    def update_maxlevel(self):
        
        if self.maxlevel+1<self.levels:
            print("setting maximum decomposition level from:")
            print(self.maxlevel,self.tree)
            self.maxlevel += 1
            self.tree = np.append(np.product(self.full_tree[:self.minlevel+1]),
                              self.full_tree[self.minlevel+1:self.maxlevel+1])
            print("to")
            print(self.maxlevel,self.tree)
            
            for point in self.points[self.points[:,0]==self.maxlevel-1]:
                children = self.get_children(point)
                self.potential_children = np.vstack((self.potential_children,
                                                     children))
     
        
    def optimize_gridsize(minx,maxx,miny,maxy,xgridspacing,ygridspacing):
        
        # function to optimize the gridsize so that we can construct a tree
        # that has 4 children for every node.
        # During the wavelet decomposition the grid is split in successively
        # rougher grids by taking only every second sample of the original grid
        # 16 -> 8 -> 4 -> 2 -> 1
        # If the gridsize is not a power of 2, this will fail at some point.
        # 18 -> 9 -> !
        # In general, this is not a problem, because the wavelet decomposition
        # works also with non-integer decimations:
        # 18 -> 9 -> 5 -> 3 -> 2 -> 1
        # however, this means that not every node has exactly four children
        # which makes the calculations more difficult
        # this function will therefore try to avoid this by decreasing the
        # grid spacing, i.e. increasing the number of samples
        # 18 samples increased to 20
        # 20 -> 10 -> 5
        # the optimization will stop at some point, meaning that it will not
        # be possible to have only one single root node
        # instead, in the example above, there would be 5 root nodes

        #minx = -110.75607181407213
        #maxx = 104.96901884072297
        #xgridspacing = 1.
        
        xpoints = len(np.arange(minx,maxx+xgridspacing,xgridspacing))
        partition_levels = 0
        while True:
            if xpoints==1:
                xpartitions = int(xpoints * 2**partition_levels)
                break
            if xpoints%2==0:
                xpoints/=2
                partition_levels += 1
            else:
                xpoints = int(xpoints+1)
                xnew = np.linspace(minx,maxx,
                                   xpoints * 2**partition_levels)
                dxnew = xnew[1]-xnew[0]
                if dxnew < 0.5*xgridspacing:
                    xpoints-=1
                    xnew = np.linspace(minx,maxx,
                                       xpoints * 2**partition_levels)
                    dxnew = xnew[1]-xnew[0]
                    xpartitions = (xpoints * 2**partition_levels)
                    break
                
        ypoints = len(np.arange(miny,maxy+ygridspacing,ygridspacing))
        partition_levels = 0
        while True:
            if ypoints==1:
                ypartitions = int(ypoints * 2**partition_levels)
                break
            if ypoints%2==0:
                ypoints/=2
                partition_levels += 1
            else:
                ypoints = int(ypoints+1)
                ynew = np.linspace(miny,maxy,
                                   ypoints * 2**partition_levels)
                dynew = ynew[1]-ynew[0]
                if dynew < 0.5*ygridspacing:
                    ypoints-=1
                    ynew = np.linspace(miny,maxy,
                                       ypoints * 2**partition_levels)
                    dynew = ynew[1]-ynew[0]
                    ypartitions = (ypoints * 2**partition_levels)
                    break
                
        return xpartitions,ypartitions
                
                
    
    def memoize_arrangements(self,tree,k):
        
        # based on "Geophysical imaging using trans-dimensional trees" by
        # Hawkins and Sambridge, 2015, appendix A

        # this function will successively fill the D dictionary where the
        # number of possible arrangements in a tree is stored
        # for example in a ternary tree with 3 active nodes, there are 12
        # possible arrangements
        # tree = (1, 3, 3, 3, 3); k=3
        # D[((1,3,3,3,3),3)] = 12    

        kmax = 0
        for i in range(len(tree)):
            kmax += np.product(tree[:i+1])
        if k==0 or k==kmax:
            return 1
        if k<0 or k>kmax:
            return 0
        
        try:
            return self.D[(tuple(tree),k)]
        except:
            j = tree[0]
            if j==1:
                A = tree[1:]
                self.D[(tuple(tree),k)] = self.memoize_arrangements(A,k-1)
            elif j%2==1:
                A = tree.copy()
                A[0] = 1
                B = tree.copy()
                B[0] -= 1
                self.D[(tuple(tree),k)] = self.compute_subtrees(A,B,k)
            else:
                A = tree.copy()
                A[0] = tree[0]/2
                B = tree.copy()
                B[0] = tree[0]/2
                self.D[(tuple(tree),k)] = self.compute_subtrees(A,B,k)
            
        return self.D[(tuple(tree),k)]
            
            
    def compute_subtrees(self,A,B,k):
        arrangements = 0
        for i in range(k+1):
            a = self.memoize_arrangements(A,i)
            b = self.memoize_arrangements(B,k-i)
            arrangements += a*b
        return arrangements
    
        # # for comparison, possible arrangements in a ternary tree:
        # def compute_ternary(k):
        #     if k<=0:
        #         return 1
        #     arrangements = 0
        #     for i in range(k):
        #         a = compute_ternary(i)
        #         subsum = 0
        #         for j in range(k-i):
        #             b = compute_ternary(j)
        #             c = compute_ternary(k-i-j-1)
        #             subsum += b*c
        #         arrangements += a*subsum
        #     return arrangements
        
        # # comparison possible arrangements in a binary tree
        # def compute_binary_heighrestricted(k,h):
        #     if k<=0:
        #         return 1
        #     elif k<=0 and h<=0:
        #         return 1
        #     elif k>0 and h<=0:
        #         return 0
        #     elif k<=0 and h>=0:
        #         return 1
        #     else:
        #         arrangements = 0
        #         for i in range(k):
        #             a = compute_binary_heighrestricted(i,h-1)
        #             b = compute_binary_heighrestricted(k-i-1,h-1)
        #             arrangements += a*b
        #         return arrangements
            
 
    def get_children(self,point):
        
        level,direction,yind,xind,isleaf = point
        
        if level+1 > self.maxlevel:
            return []
        
        # a level 0 node has 3 children, x-direction child, y-direction child
        # and diagional child
        if level==0:
            return [(level+1,0,yind,xind,1),
                    (level+1,1,yind,xind,1),
                    (level+1,2,yind,xind,1)]
        
        # a higher level node has 4 children, the direction is the same as the
        # parent direction but the "pixel" gets split into 4 smaller ones
        children = [(level+1,direction,yind*2,xind*2,1)]
        # if the model dimensions are not a power of two, the border nodes
        # may have less than 4 children
        if yind*2+1 < self.coeffs[level+1][direction].shape[0]:
            children.append((level+1,direction,yind*2+1,xind*2,1))
        if xind*2+1 < self.coeffs[level+1][direction].shape[1]:
            children.append((level+1,direction,yind*2,xind*2+1,1))
        if len(children)==3:
            children.append((level+1,direction,yind*2+1,xind*2+1,1))
        return np.vstack(children)
    

    def get_parent_idx(self,point):
        
        level,direction,yind,xind,isleaf = point
        
        if level==0:
            raise Exception("level 0 has no parent")
            
        if level==1:
            idx_parent = np.where((self.points[:,0]==level-1)*
                                  (self.points[:,2]==yind)*
                                  (self.points[:,3]==xind))[0]
        
        else:
            idx_parent = np.where((self.points[:,0]==level-1)*
                                  (self.points[:,1]==direction)*
                                  (self.points[:,2]==int(yind/2))*
                                  (self.points[:,3]==int(xind/2)))[0]
        
        if len(idx_parent) > 1:
            raise Exception("there should only be one parent!")
        
        return idx_parent
    
        
    def vs_update(self,idx,dcoeff,backup=True):
        
        level,direction,yind,xind,isleaf = self.points[idx]
        
        # with increasing level, the coefficient ranges become smaller
        # this happens with an approximate factor of 0.5
        # this way, the acceptance at each level should be approximately equal
        # dcoeff *= *1.2**(8-level) # 5 could be replaced by any value, but should be adjusted to the proposal ratio
        # CURRENTLY NOT REALLY USEFUL
        
        if level>0:
            prop_coeff = self.coeffs[level][direction][yind,xind] + dcoeff
        else:
            prop_coeff = self.coeffs[level][yind,xind] + dcoeff           
        
        if (prop_coeff>self.coeff_range[level]['max'] or 
            prop_coeff<self.coeff_range[level]['min']):
            return False
        
        if backup:
            self.backup_mod()        
        self.action = 'update'
        
        if level>0:
            self.coeffs[level][direction][yind,xind] = prop_coeff
        else:
            self.coeffs[level][yind,xind] = prop_coeff

        self.idx_mod = idx
        
        return True
        
        
    def add_point(self,birth_prop='uniform',anisotropic=False,backup=True):
        
        if birth_prop!='uniform':
            raise Exception("only uniform birth proposals are currently implemented")
        
        # do a better implementation of the case where the maximum tree 
        # height is reached
        if len(self.potential_children) == 0:
            return False
        
        if backup:
            self.backup_mod()
            
        self.action='birth'
           
        # randomly choose one of the potential children
        idx_birth = np.random.randint(0,len(self.potential_children))
        
        level,direction,yind,xind,isleaf = self.potential_children[idx_birth]
        
        self.points = np.vstack((self.points,self.potential_children[idx_birth]))
        
        coeff_prop = np.random.uniform(
            self.coeff_range[level]['min'],self.coeff_range[level]['max'])
        if level==0:
            self.coeffs[level][yind,xind] = coeff_prop
        else:
            self.coeffs[level][direction][yind,xind] = coeff_prop
        
        # needed for the proposal ratio calculation
        self.no_birthnodes = len(self.potential_children)
        
        self.potential_children = np.delete(self.potential_children,
                                            idx_birth,axis=0)
        #self.potential_children.remove(tuple(self.points[-1]))
        children = self.get_children(self.points[-1])
        if len(children) > 0:
            self.potential_children = np.vstack((self.potential_children,
                                                 children))
        idx_parent = self.get_parent_idx(self.points[-1])
        self.points[idx_parent,-1] = 0
        
        return True


    def remove_point(self,anisotropic=False,backup=True):
        
        if backup:
            self.backup_mod()
            
        self.action='death'
           
        # randomly choose one of the leaf nodes to remove
        # leaf nodes are those that have no children
        self.no_deathnodes = np.sum(self.points[:,-1])
        if self.no_deathnodes > 0:
            idx_death = np.random.choice(np.where(self.points[:,-1])[0])
        else:
            return False
        
        self.point_removed = self.points[idx_death]
        level,direction,yind,xind,isleaf = self.point_removed
        if self.point_removed[0]<self.minlevel:
            raise Exception("should not happen!")
        
        self.points = np.delete(self.points,idx_death,axis=0)
        if level==0:
            self.coeffs[level][yind,xind] = 0.
        else:
            self.coeffs[level][direction][yind,xind] = 0.
            
        # the children of the removed point have to be removed from the list
        # of potential children (in a birth step a child from potential
        # children is chosen)
        children = self.get_children(self.point_removed)
        if len(children)>0:
            idx_remove = np.where(
                (self.potential_children[:,0]==children[0,0]) * 
                np.in1d(self.potential_children[:,1],children[:,1]) *
                np.in1d(self.potential_children[:,2],children[:,2]) *
                np.in1d(self.potential_children[:,3],children[:,3]))[0]
            self.potential_children = np.delete(self.potential_children,
                                                idx_remove,0)
        # the removed point has to be added to the list of potential children
        self.potential_children = np.vstack((self.potential_children,
                                             self.point_removed))
        # the parent might become a leaf node now (removable), do a check
        if self.point_removed[0]>self.minlevel+1:
            # check if the removed node had any siblings 
            level,direction,yind,xind,isleaf = self.point_removed      
            coeff_sum_siblings = (
                self.coeffs[level][direction][int(yind/2)*2,int(xind/2)*2] +
                self.coeffs[level][direction][int(yind/2)*2+1,int(xind/2)*2] +
                self.coeffs[level][direction][int(yind/2)*2,int(xind/2)*2+1] +
                self.coeffs[level][direction][int(yind/2)*2+1,int(xind/2)*2+1] -
                self.coeffs[level][direction][yind,xind])
            if coeff_sum_siblings == 0.: # i.e. no siblings
                # make parent node a leaf node
                idx_parent = self.get_parent_idx(self.point_removed)[0]
                self.points[idx_parent,-1] = 1
 
        return True
         
        
    def get_model(self,coeffs=None):
        
        if coeffs is None:
            coeffs = self.coeffs
        
        # reconstruct the field from the coefficients
        velfield = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        return 1./velfield.flatten()
    
        # coeffs = deepcopy(self.coeffs)
        # for level in range(self.levels):
        #     if level==0:
        #         coeffs[level][:] = 0.
        #     else:
        #         for direction in [0,1,2]:
        #             coeffs[level][direction][:] = 0.
        # coeffs[2][0][1,1] = 3.
        # velfield1 = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        # coeffs[2][0][1,1] = 0.
        # coeffs[3][1][1,1] = 1.
        # velfield2 = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        # coeffs[2][0][1,1] = 3.
        # velfield3 = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        
        
        
    def update_model(self,fields=None,anisotropic=False):
        
        if anisotropic:
            slowfield, psi2amp, psi2 = fields
        else:
            slowfield = fields
                                    
        velfield = pywt.waverec2(self.coeffs, self.wavelet, mode='periodization')
        
        if np.max(velfield)>self.velmax or np.min(velfield)<self.velmin:
            return np.nan
        
        slowfield_prop = 1./velfield.flatten()
        
        self.idx_mod_gpts = np.where(slowfield_prop!=slowfield)[0]
        
        return slowfield_prop
       
    
    def get_prior_proposal_ratio(self):
                        
        if self.action == 'update':
            # if the number of active nodes is unchanged and only the 
            # coefficients are changed, the probability from going from one
            # coefficient to another one is equal to the reverse step
            return 0
        
        elif self.action == 'birth':
            # assuming we draw from a uniform distribution
            # nominator: number of possible nodes to choose from during birth
            # denominator: number of possible death nodes to choose from after
            #              birth
            proposal_ratio = self.no_birthnodes / np.sum(self.points[:,-1])
            # eqs 13 & 16 from Hawkins & Sambridge 2015
            prior_ratio = (
                self.memoize_arrangements(self.tree, len(self.points)-1-self.kmin) /
                self.memoize_arrangements(self.tree, len(self.points)-self.kmin) )
            
        elif self.action == 'death':
            # same as birth but inverse
            proposal_ratio = self.no_deathnodes / len(self.potential_children)
            prior_ratio = (
                self.memoize_arrangements(self.tree, len(self.points)+1-self.kmin) /
                self.memoize_arrangements(self.tree, len(self.points)-self.kmin) )

        return np.log(proposal_ratio * prior_ratio) 
    
    
    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        self.coeffs_backup = deepcopy(self.coeffs)
        self.potential_children_backup = self.potential_children.copy()
        if self.psi2amp is not None:
            self.psi2amp_backup = self.psi2amp.copy()
        if self.psi2 is not None:
            self.psi2_backup = self.psi2.copy()
       
    
    def update_coeff_range(self):
        
        for level in self.coeff_range:
            if level == 0:
                coeff_range = (self.coeff_range[0]['max'] -
                               self.coeff_range[0]['min'])/2.
                # maxcoeff = np.max(np.abs(self.coeffs[level]))
                # mincoeff = np.min(np.abs(self.coeffs[level]))
                # self.coeff_range[level]['max'] = 1.01*maxcoeff
                # self.coeff_range[level]['min'] = 0.99*mincoeff
            else:   
                maxcoeff = np.max(np.abs([self.coeffs[level][0][:],
                                          self.coeffs[level][1][:],
                                          self.coeffs[level][2][:]]))
                maxcoeff = np.max([maxcoeff,0.2*self.coeff_range[level-1]['max']])
                if level>1:
                    maxcoeff = np.min([maxcoeff,0.5*self.coeff_range[level-1]['max']])
                else:
                    maxcoeff = np.min([maxcoeff,0.5*coeff_range])
                self.coeff_range[level]['max'] =  1.1*maxcoeff
                self.coeff_range[level]['min'] = -1.1*maxcoeff
     
        
    def reject_mod(self):
        
        if self.action=='update':
            self.rejected_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 0
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
            
        if self.points_backup is not None:
            self.points = self.points_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        if self.coeffs_backup is not None:
            self.coeffs = self.coeffs_backup
        if self.potential_children_backup is not None:
            self.potential_children = self.potential_children_backup
         
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not child in self.potential_children:
        #             raise Exception(f"error after reject {self.action}") 
                    
        self.points_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.point_removed = None
        self.coeffs_backup = None
        self.no_birthnodes = None
        self.no_deathnodes = None
        self.potential_children_backup = None
    
        
    def accept_mod(self,selfcheck=False):
        
        # testlist = []
        # for item in self.points:
        #     testlist.append(tuple(item[:-1]))
        # for point in self.points:
        #     if point[0]==0:
        #         continue
        #     if point[-1]==0:
        #         children = self.get_children(point)
        #         for child in children:
        #             if tuple(child[:-1]) in testlist:
        #                 break
        #         else:
        #             print(point)
        #             raise Exception("this node should be marked as leaf! error after",self.action) 
                    
        if self.action=='update':
            self.accepted_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 1
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
            
            if np.random.rand(1)>0.8:
                self.update_coeff_range()
         
        if selfcheck:
            for point in self.points:
                if (self.potential_children[:,:-1]==point[:-1]).all(axis=1).any():
                    idx_child = np.where((self.potential_children[:,:-1]==point[:-1]).all(axis=1))[0]
                    print(idx_child,self.potential_children[idx_child],point)
                    raise Exception(f"point is also in potential children after {self.action}")
                if np.sum((self.points[:,:-1]==point[:-1]).all(axis=1)) > 1:
                    idx_double = np.where((self.points[:,:-1]==point[:-1]).all(axis=1))[0]
                    print(idx_double,self.points[idx_double])
                    raise Exception(f"double points after {self.action}")
            
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not (self.potential_children==child).all(axis=1).any():
        #             raise Exception(f"error after accept {self.action}")
        #         if (self.points==child).all(axis=1).any():
        #             raise Exception(f"error after accept {self.action}, should not be leaf node!")
                   
        # for point in self.points:
        #     if point[0]==0:
        #         continue
        #     parent_idx = self.get_parent_idx(point)
        #     if len(parent_idx)!=1 or self.points[parent_idx,-1]==1:
        #         raise Exception("error after",self.action)
        
        # for point in self.points:
        #     children = self.get_children(point)
        #     isleaf = True
        #     for child in children:
        #         idx_child = np.where((self.points[:,:4]==child[:4]).all(axis=1))[0]
        #         if len(idx_child)>0 and point[-1]==1:
        #             raise Exception()
        #         if len(idx_child)>0:
        #             isleaf = False
        #     if point[-1]==0 and isleaf:
        #         raise Exception("should be leaf")
                
                
        self.points_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.point_removed = None
        self.coeffs_backup = None
        self.no_birthnodes = None
        self.no_deathnodes = None
        self.potential_children_backup = None
        
        
            
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):

        slowfield = self.get_model()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cbar = ax.pcolormesh(1./slowfield.reshape(self.shape),cmap=plt.cm.seismic_r)
        plt.colorbar(cbar)
        plt.show()
        
        


##############################################################################
##############################################################################
##############################################################################

class wavelets_anisotropic(object):
    
    
    def __init__(self,gridpoints,shape,velmin,velmax,init_no_coeffs=None,
                 psi2ampmin=0.,psi2ampmax=0.1,anisotropic=False):
        
        self.gridpoints = gridpoints
        self.shape = shape
        
        self.wavelet = 'bior4.4'#'bior4.4'
        decomposition_level = int(np.log2(np.min(self.shape)))
        # minlevel gives the minimum level of the decomposition tree in which
        # the birth/death operations may take place. nodes smaller than
        # minlevel are always filled. Normally, minlevel=0 (no restriction)
        self.minlevel=0
       
        self.velmax = self.minmod = velmax
        self.velmin = self.maxmod = velmin
        
        start_mod = np.random.uniform(low=self.velmin,high=self.velmax,
                                      size=self.shape)
        self.coeffs = pywt.wavedec2(start_mod, self.wavelet,
                                    level=decomposition_level,
                                    mode='periodization')
        mincoeff = np.min(pywt.wavedec2(np.ones_like(start_mod)*self.velmin, 
                                        self.wavelet,level=decomposition_level,
                                        mode='periodization')[0])
        maxcoeff = np.max(pywt.wavedec2(np.ones_like(start_mod)*self.velmax, 
                                        self.wavelet,level=decomposition_level,
                                        mode='periodization')[0])

        self.points = []
        for level in range(len(self.coeffs)):
            
            if level==0:
                shape = self.coeffs[level].shape
            elif level==1:
                shape = (3,) + shape
            else:
                shape = (3, shape[1]*2,shape[2]*2)
            if np.shape(self.coeffs[level])!=shape:
                raise Exception("bad tree structure")
                
            if level<self.minlevel or level==0:
                for direction in range(3):
                    if level==0 and direction>0:
                        continue
                    elif level==0:
                        direction=None
                    indices = np.column_stack(np.unravel_index(
                        np.arange(len(self.coeffs[level][direction].flatten())),
                        shape[-2:]))
                    if level==0:
                        direction=0
                    self.points.append(np.column_stack((np.ones(len(indices))*level,
                                                        np.ones(len(indices))*direction,
                                                        indices,
                                                        np.zeros(len(indices)))))
            else:
                self.coeffs[level][0][:] = self.coeffs[level][1][:] = self.coeffs[level][2][:] = 0.
        
        # points has 5 columns: level, direction (hor,ver,diag), yind, xind, isleaf
        # at level=0, the direction does not apply
        self.points = np.vstack(self.points).astype(int)
        
        self.anisotropic = anisotropic
        if self.anisotropic:
            print("anisotropic wavelet search")
            print("WARNING: this is currently only for testing, not yet fully "
                  +"implemented, should probably replace psi2amp and psi2 with "
                  +"x and y component of anisotropy. Also birth death is not yet "
                  +"symmetric, check acceptance probabilities.")
            self.coeffs_psi2amp = deepcopy(self.coeffs)
            for level in range(len(self.coeffs)):
                if level==0:
                    self.coeffs_psi2amp[level][:] = 0.
                    continue
                for direction in range(3):
                    self.coeffs_psi2amp[level][direction][:] = 0.
            self.coeffs_psi2 = pywt.wavedec2(np.random.uniform(
                -np.pi,np.pi,self.shape), 'haar', level=decomposition_level,
                mode='periodization')
            self.coeff_range_psi2 = {}
            maxcoeff = 0.
            for direction in range(3):
                maxcoeff = np.max([maxcoeff,np.max(np.abs(self.coeffs_psi2[-1][direction]))])
            maxcoeff = np.around(maxcoeff,1)
            for level in reversed(range(len(self.coeffs_psi2))):
                self.coeff_range_psi2[level] = maxcoeff
                maxcoeff *= 2
            self.psi2ampmin=psi2ampmin
            self.psi2ampmax=psi2ampmax
            
        # number of decomposition levels
        self.levels = decomposition_level+1
        
        self.coeff_range = {}
        for level in range(self.levels):
            self.coeff_range[level] = {}
            self.coeff_range[level]['max'] = 0.
            self.coeff_range[level]['min'] = 0.
        self.coeff_range[0]['min'] = mincoeff
        self.coeff_range[0]['max'] = maxcoeff
        self.update_coeff_range()
        
        self.maxlevel = 1
        #print("Warning, setting maximum level to",self.maxlevel)#self.levels-1)
        
        # potential children includes all new nodes that can be chosen from
        # in a birth operation (children of currently active nodes)
        self.potential_children = []
        for point in self.points:
            children = self.get_children(point)
            for child in children:
                idx1 = np.where((child[:4]==self.points[:,:4]).all(axis=1))[0]
                if len(idx1)==0:
                    self.potential_children.append(child)
        if len(self.potential_children)>0:
            self.potential_children = np.vstack(self.potential_children)
            self.potential_children = np.unique(self.potential_children,axis=0)
        else:
            self.potential_children = np.empty((0,5),dtype=int)
                
        # dictionary that stores the number of possible arrangements of nodes
        # for a certain tree structure
        try:
            with open("tree_dict.pkl","rb") as f:
                self.D = pickle.load(f)
            print("read tree dictionary from tree_dict.pkl")
        except:
            self.D = {}
        # getting the tree structure. the first number in the array gives the
        # total number of root nodes. The following number give the number of
        # children for each root node.
        # In a binary tree that has a maximum depth of 3, this would look like
        # self.tree = [1,2,2,2]
        # However, in this case we have already an array at the root, e.g.
        # 6*7 elements, resulting in 42 root nodes. Each root node has 3 child-
        # ren, one for each direction. For each direction, each node has 4
        # children, on the succesively finer grids
        # self.tree = [42, 3, 4, 4, 4]
        self.full_tree = np.array([len(self.coeffs[0].flatten()),3] + 
                                  (self.levels-2)*[4])
        # adapt tree to the minlevel and the maxlevel
        self.tree = np.append(np.product(self.full_tree[:self.minlevel+1]),
                              self.full_tree[self.minlevel+1:self.maxlevel+1])
        self.kmin = 0
        for i in range(self.minlevel):
            self.kmin += np.product(self.full_tree[:i+1])
        
        self.points_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.coeffs_psi2amp_backup = None
        self.coeffs_psi2_backup = None  
        self.coeffs_backup = None
        self.potential_children_backup = None
        

        # add points until the number of initial coefficients is reached
        if init_no_coeffs is not None:
            init_coeffs = np.max([0,init_no_coeffs-len(self.points)])
        else:
            init_coeffs = 0
        for i in range(init_coeffs):
            check_birth = self.add_point()
            if check_birth:
                slow_prop = self.update_model()
                if np.isnan(slow_prop).any():
                    self.reject_mod()
                else:
                    self.accept_mod()
        
        
        #slowfield = self.get_model()
        #if (1./slowfield > velmax).any() or (1./slowfield < velmin).any():
        #    raise Exception("bad starting model")        
        
        self.acceptance_rate = {}
        self.accepted_steps = {}
        self.rejected_steps = {}
        for level in range(self.levels):
            self.acceptance_rate[level] = np.zeros(100,dtype=int)
            self.accepted_steps[level] = 0
            self.rejected_steps[level] = 0
        
        # # array of decomposed wavelet coefficients
        # self.coeffs = []
        # for i,shape in enumerate(shapes[::-1]):
        #     if i==0:
        #         self.coeffs.append(np.zeros(shape))
        #     self.coeffs.append((np.zeros(shape),np.zeros(shape),np.zeros(shape)))
        
     
    def update_maxlevel(self):
        
        if self.maxlevel+1<self.levels:
            print("setting maximum decomposition level from:")
            print(self.maxlevel,self.tree)
            self.maxlevel += 1
            self.tree = np.append(np.product(self.full_tree[:self.minlevel+1]),
                              self.full_tree[self.minlevel+1:self.maxlevel+1])
            print("to")
            print(self.maxlevel,self.tree)
            
            for point in self.points[self.points[:,0]==self.maxlevel-1]:
                children = self.get_children(point)
                self.potential_children = np.vstack((self.potential_children,
                                                     children))
     
        
    def optimize_gridsize(minx,maxx,miny,maxy,xgridspacing,ygridspacing):
        
        # function to optimize the gridsize so that we can construct a tree
        # that has 4 children for every node.
        # During the wavelet decomposition the grid is split in successively
        # rougher grids by taking only every second sample of the original grid
        # 16 -> 8 -> 4 -> 2 -> 1
        # If the gridsize is not a power of 2, this will fail at some point.
        # 18 -> 9 -> !
        # In general, this is not a problem, because the wavelet decomposition
        # works also with non-integer decimations:
        # 18 -> 9 -> 5 -> 3 -> 2 -> 1
        # however, this means that not every node has exactly four children
        # which makes the calculations more difficult
        # this function will therefore try to avoid this by decreasing the
        # grid spacing, i.e. increasing the number of samples
        # 18 samples increased to 20
        # 20 -> 10 -> 5
        # the optimization will stop at some point, meaning that it will not
        # be possible to have only one single root node
        # instead, in the example above, there would be 5 root nodes

        #minx = -110.75607181407213
        #maxx = 104.96901884072297
        #xgridspacing = 1.
        
        xpoints = len(np.arange(minx,maxx+xgridspacing,xgridspacing))
        partition_levels = 0
        while True:
            if xpoints==1:
                xpartitions = int(xpoints * 2**partition_levels)
                break
            if xpoints%2==0:
                xpoints/=2
                partition_levels += 1
            else:
                xpoints = int(xpoints+1)
                xnew = np.linspace(minx,maxx,
                                   xpoints * 2**partition_levels)
                dxnew = xnew[1]-xnew[0]
                if dxnew < 0.5*xgridspacing:
                    xpoints-=1
                    xnew = np.linspace(minx,maxx,
                                       xpoints * 2**partition_levels)
                    dxnew = xnew[1]-xnew[0]
                    xpartitions = (xpoints * 2**partition_levels)
                    break
                
        ypoints = len(np.arange(miny,maxy+ygridspacing,ygridspacing))
        partition_levels = 0
        while True:
            if ypoints==1:
                ypartitions = int(ypoints * 2**partition_levels)
                break
            if ypoints%2==0:
                ypoints/=2
                partition_levels += 1
            else:
                ypoints = int(ypoints+1)
                ynew = np.linspace(miny,maxy,
                                   ypoints * 2**partition_levels)
                dynew = ynew[1]-ynew[0]
                if dynew < 0.5*ygridspacing:
                    ypoints-=1
                    ynew = np.linspace(miny,maxy,
                                       ypoints * 2**partition_levels)
                    dynew = ynew[1]-ynew[0]
                    ypartitions = (ypoints * 2**partition_levels)
                    break
                
        return xpartitions,ypartitions
                
                
    
    def memoize_arrangements(self,tree,k):
        
        # based on "Geophysical imaging using trans-dimensional trees" by
        # Hawkins and Sambridge, 2015, appendix A

        # this function will successively fill the D dictionary where the
        # number of possible arrangements in a tree is stored
        # for example in a ternary tree with 3 active nodes, there are 12
        # possible arrangements
        # tree = (1, 3, 3, 3, 3); k=3
        # D[((1,3,3,3,3),3)] = 12    

        kmax = 0
        for i in range(len(tree)):
            kmax += np.product(tree[:i+1])
        if k==0 or k==kmax:
            return 1
        if k<0 or k>kmax:
            return 0
        
        try:
            return self.D[(tuple(tree),k)]
        except:
            j = tree[0]
            if j==1:
                A = tree[1:]
                self.D[(tuple(tree),k)] = self.memoize_arrangements(A,k-1)
            elif j%2==1:
                A = tree.copy()
                A[0] = 1
                B = tree.copy()
                B[0] -= 1
                self.D[(tuple(tree),k)] = self.compute_subtrees(A,B,k)
            else:
                A = tree.copy()
                A[0] = tree[0]/2
                B = tree.copy()
                B[0] = tree[0]/2
                self.D[(tuple(tree),k)] = self.compute_subtrees(A,B,k)
            
        return self.D[(tuple(tree),k)]
            
            
    def compute_subtrees(self,A,B,k):
        arrangements = 0
        for i in range(k+1):
            a = self.memoize_arrangements(A,i)
            b = self.memoize_arrangements(B,k-i)
            arrangements += a*b
        return arrangements
    
        # # for comparison, possible arrangements in a ternary tree:
        # def compute_ternary(k):
        #     if k<=0:
        #         return 1
        #     arrangements = 0
        #     for i in range(k):
        #         a = compute_ternary(i)
        #         subsum = 0
        #         for j in range(k-i):
        #             b = compute_ternary(j)
        #             c = compute_ternary(k-i-j-1)
        #             subsum += b*c
        #         arrangements += a*subsum
        #     return arrangements
        
        # # comparison possible arrangements in a binary tree
        # def compute_binary_heighrestricted(k,h):
        #     if k<=0:
        #         return 1
        #     elif k<=0 and h<=0:
        #         return 1
        #     elif k>0 and h<=0:
        #         return 0
        #     elif k<=0 and h>=0:
        #         return 1
        #     else:
        #         arrangements = 0
        #         for i in range(k):
        #             a = compute_binary_heighrestricted(i,h-1)
        #             b = compute_binary_heighrestricted(k-i-1,h-1)
        #             arrangements += a*b
        #         return arrangements
            
 
    def get_children(self,point):
        
        level,direction,yind,xind,isleaf = point
        
        if level+1 > self.maxlevel:
            return []
        
        # a level 0 node has 3 children, x-direction child, y-direction child
        # and diagional child
        if level==0:
            return [(level+1,0,yind,xind,1),
                    (level+1,1,yind,xind,1),
                    (level+1,2,yind,xind,1)]
        
        # a higher level node has 4 children, the direction is the same as the
        # parent direction but the "pixel" gets split into 4 smaller ones
        children = [(level+1,direction,yind*2,xind*2,1)]
        # if the model dimensions are not a power of two, the border nodes
        # may have less than 4 children
        if yind*2+1 < self.coeffs[level+1][direction].shape[0]:
            children.append((level+1,direction,yind*2+1,xind*2,1))
        if xind*2+1 < self.coeffs[level+1][direction].shape[1]:
            children.append((level+1,direction,yind*2,xind*2+1,1))
        if len(children)==3:
            children.append((level+1,direction,yind*2+1,xind*2+1,1))
        return np.vstack(children)
    

    def get_parent_idx(self,point):
        
        level,direction,yind,xind,isleaf = point
        
        if level==0:
            raise Exception("level 0 has no parent")
            
        if level==1:
            idx_parent = np.where((self.points[:,0]==level-1)*
                                  (self.points[:,2]==yind)*
                                  (self.points[:,3]==xind))[0]
        
        else:
            idx_parent = np.where((self.points[:,0]==level-1)*
                                  (self.points[:,1]==direction)*
                                  (self.points[:,2]==int(yind/2))*
                                  (self.points[:,3]==int(xind/2)))[0]
        
        if len(idx_parent) > 1:
            raise Exception("there should only be one parent!")
        
        return idx_parent
    
        
    def vs_update(self,idx,dcoeff,backup=True):
        
        level,direction,yind,xind,isleaf = self.points[idx]
        
        # with increasing level, the coefficient ranges become smaller
        # this happens with an approximate factor of 0.5
        # this way, the acceptance at each level should be approximately equal
        # dcoeff *= *1.2**(8-level) # 5 could be replaced by any value, but should be adjusted to the proposal ratio
        # CURRENTLY NOT REALLY USEFUL
        
        if level>0:
            prop_coeff = self.coeffs[level][direction][yind,xind] + dcoeff
        else:
            prop_coeff = self.coeffs[level][yind,xind] + dcoeff           
        
        if (prop_coeff>self.coeff_range[level]['max'] or 
            prop_coeff<self.coeff_range[level]['min']):
            return False
        
        if backup:
            self.backup_mod()        
        self.action = 'update'
        
        if level>0:
            self.coeffs[level][direction][yind,xind] = prop_coeff
        else:
            self.coeffs[level][yind,xind] = prop_coeff

        self.idx_mod = idx
        
        return True
        
    
    def psi2_update(self,idx,_,backup=True):
        
        if backup:
            self.backup_mod()
            
        level,direction,yind,xind,isleaf = self.points[idx]

        # the actual value is not so important since there is a pi periodicity
        # 1pi == 2pi == 3pi ... so the values are not that imporant 
        # simply draw from a uniform distribution that will result in values
        # roughly in the range -2pi,2pi
        # might this be biased towards a certain direction?
        c_new = np.random.uniform(-self.coeff_range_psi2[level],self.coeff_range_psi2[level])
        if level>0:
            self.coeffs_psi2[level][direction][yind,xind] = c_new
        else:
            self.coeffs_psi2[level][yind,xind] = c_new
        
        self.action='psi2_update'
        self.idx_mod = idx
        
        return True
        
    
    def psi2amp_update(self,idx,delta,backup=True):
        
        if backup:
            self.backup_mod()
            
        level,direction,yind,xind,isleaf = self.points[idx]

        if level>0:
            self.coeffs_psi2amp[level][direction][yind,xind] += delta
        else:
            self.coeffs_psi2amp[level][yind,xind] += delta
            
        self.action='psi2amp_update'
        self.idx_mod = idx
        
        return True

        
        
    def add_point(self,birth_prop='uniform',anisotropic=False,backup=True):
        
        if birth_prop!='uniform':
            raise Exception("only uniform birth proposals are currently implemented")
        
        # do a better implementation of the case where the maximum tree 
        # height is reached
        if len(self.potential_children) == 0:
            return False
        
        if backup:
            self.backup_mod()
            
        self.action='birth'
           
        # randomly choose one of the potential children
        idx_birth = np.random.randint(0,len(self.potential_children))
        
        level,direction,yind,xind,isleaf = self.potential_children[idx_birth]
        
        self.points = np.vstack((self.points,self.potential_children[idx_birth]))
        
        coeff_prop = np.random.uniform(
            self.coeff_range[level]['min'],self.coeff_range[level]['max'])
        if level==0:
            self.coeffs[level][yind,xind] = coeff_prop
        else:
            self.coeffs[level][direction][yind,xind] = coeff_prop
        
        # needed for the proposal ratio calculation
        self.no_birthnodes = len(self.potential_children)
        
        self.potential_children = np.delete(self.potential_children,
                                            idx_birth,axis=0)
        #self.potential_children.remove(tuple(self.points[-1]))
        children = self.get_children(self.points[-1])
        if len(children) > 0:
            self.potential_children = np.vstack((self.potential_children,
                                                 children))
        idx_parent = self.get_parent_idx(self.points[-1])
        self.points[idx_parent,-1] = 0
        
        return True


    def remove_point(self,anisotropic=False,backup=True):
        
        if backup:
            self.backup_mod()
            
        self.action='death'
           
        # randomly choose one of the leaf nodes to remove
        # leaf nodes are those that have no children
        self.no_deathnodes = np.sum(self.points[:,-1])
        if self.no_deathnodes > 0:
            idx_death = np.random.choice(np.where(self.points[:,-1])[0])
        else:
            return False
        
        self.point_removed = self.points[idx_death]
        level,direction,yind,xind,isleaf = self.point_removed
        if self.point_removed[0]<self.minlevel:
            raise Exception("should not happen!")
        
        self.points = np.delete(self.points,idx_death,axis=0)
        if level==0:
            self.coeffs[level][yind,xind] = 0.
            if self.anisotropic:
                self.coeffs_psi2amp[level][yind,xind] = 0.
        else:
            self.coeffs[level][direction][yind,xind] = 0.
            if anisotropic:
                self.coeffs_psi2amp[level][direction][yind,xind] = 0.
                
        # the children of the removed point have to be removed from the list
        # of potential children (in a birth step a child from potential
        # children is chosen)
        children = self.get_children(self.point_removed)
        if len(children)>0:
            idx_remove = np.where(
                (self.potential_children[:,0]==children[0,0]) * 
                np.in1d(self.potential_children[:,1],children[:,1]) *
                np.in1d(self.potential_children[:,2],children[:,2]) *
                np.in1d(self.potential_children[:,3],children[:,3]))[0]
            self.potential_children = np.delete(self.potential_children,
                                                idx_remove,0)
        # the removed point has to be added to the list of potential children
        self.potential_children = np.vstack((self.potential_children,
                                             self.point_removed))
        # the parent might become a leaf node now (removable), do a check
        if self.point_removed[0]>self.minlevel+1:
            # check if the removed node had any siblings 
            level,direction,yind,xind,isleaf = self.point_removed      
            coeff_sum_siblings = (
                self.coeffs[level][direction][int(yind/2)*2,int(xind/2)*2] +
                self.coeffs[level][direction][int(yind/2)*2+1,int(xind/2)*2] +
                self.coeffs[level][direction][int(yind/2)*2,int(xind/2)*2+1] +
                self.coeffs[level][direction][int(yind/2)*2+1,int(xind/2)*2+1] -
                self.coeffs[level][direction][yind,xind])
            if coeff_sum_siblings == 0.: # i.e. no siblings
                # make parent node a leaf node
                idx_parent = self.get_parent_idx(self.point_removed)[0]
                self.points[idx_parent,-1] = 1
 
        return True
         
        
    def get_model(self,coeffs=None,anisotropic=False):
        
        if coeffs is None:
            coeffs = self.coeffs
        
        # reconstruct the field from the coefficients
        velfield = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        velfield = np.around(velfield,3)
        
        if anisotropic:
            psi2field = pywt.waverec2(self.coeffs_psi2, 'haar', mode='periodization')
            psi2field = np.around(psi2field,1)
            psi2ampfield = np.abs(pywt.waverec2(self.coeffs_psi2amp, 'haar', mode='periodization'))
            psi2ampfield = np.around(psi2ampfield,3)
            return 1./velfield.flatten(),psi2ampfield.flatten(),psi2field.flatten()
        else:
            return 1./velfield.flatten()
    
        # coeffs = deepcopy(self.coeffs)
        # for level in range(self.levels):
        #     if level==0:
        #         coeffs[level][:] = 0.
        #     else:
        #         for direction in [0,1,2]:
        #             coeffs[level][direction][:] = 0.
        # coeffs[2][0][1,1] = 3.
        # velfield1 = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        # coeffs[2][0][1,1] = 0.
        # coeffs[3][1][1,1] = 1.
        # velfield2 = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        # coeffs[2][0][1,1] = 3.
        # velfield3 = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        
        
        
    def update_model(self,fields=None,anisotropic=False):
        
        if anisotropic:
            slowfield, psi2amp, psi2 = fields
        else:
            slowfield = fields                 

        if self.action == 'psi2_update':
            psi2field = pywt.waverec2(self.coeffs_psi2, 'haar', mode='periodization')
            psi2field = np.around(psi2field,1)
            # we ignore changes in psi2 if the psi2amp at the same location is 0.
            # this means that the Aprime matrix in the rjtransdim2d script is
            # wrong, but the result will be the same, since it is updated as
            # soon as psi2amp is changed
            self.idx_mod_gpts = np.where((psi2field.flatten()!=psi2)*(psi2amp!=0.))[0]
            return slowfield,psi2amp,psi2field.flatten()
        elif self.action == 'psi2amp_update':
            psi2ampfield = np.abs(pywt.waverec2(self.coeffs_psi2amp, 'haar', mode='periodization'))
            psi2ampfield = np.around(psi2ampfield,3)
            if np.max(psi2ampfield)>self.psi2ampmax or np.min(psi2ampfield)<self.psi2ampmin:
                return np.nan,np.nan,np.nan
            self.idx_mod_gpts = np.where(psi2ampfield.flatten()!=psi2amp)[0]
            return slowfield,psi2ampfield.flatten(),psi2
        else:
            velfield = pywt.waverec2(self.coeffs, self.wavelet, mode='periodization')
            velfield = np.around(velfield,3)
            if np.max(velfield)>self.velmax or np.min(velfield)<self.velmin:
                if anisotropic:
                    return np.nan,np.nan,np.nan
                else:
                    return np.nan
            slowfield_prop = 1./velfield.flatten()
            if self.action == 'death' and anisotropic:
                psi2ampfield = np.abs(pywt.waverec2(self.coeffs_psi2amp, 
                                                    'haar', mode='periodization'))
                psi2ampfield = np.around(psi2ampfield.flatten(),3)
                if (np.max(psi2ampfield)>self.psi2ampmax or 
                    np.min(psi2ampfield)<self.psi2ampmin):
                    return np.nan,np.nan,np.nan
                self.idx_mod_gpts = np.where(np.logical_or(
                    slowfield_prop!=slowfield,psi2ampfield!=psi2amp))[0]
            elif anisotropic:
                self.idx_mod_gpts = np.where(slowfield_prop!=slowfield)[0]
                psi2ampfield = psi2amp
            else:
                self.idx_mod_gpts = np.where(slowfield_prop!=slowfield)[0]
            if anisotropic:
                return slowfield_prop,psi2ampfield,psi2
            else:
                return slowfield_prop
       
    
    def get_prior_proposal_ratio(self):
                        
        if self.action == 'update':
            # if the number of active nodes is unchanged and only the 
            # coefficients are changed, the probability from going from one
            # coefficient to another one is equal to the reverse step
            return 0
        
        elif self.action == 'birth':
            # assuming we draw from a uniform distribution
            # nominator: number of possible nodes to choose from during birth
            # denominator: number of possible death nodes to choose from after
            #              birth
            proposal_ratio = self.no_birthnodes / np.sum(self.points[:,-1])
            # eqs 13 & 16 from Hawkins & Sambridge 2015
            prior_ratio = (
                self.memoize_arrangements(self.tree, len(self.points)-1-self.kmin) /
                self.memoize_arrangements(self.tree, len(self.points)-self.kmin) )
            
        elif self.action == 'death':
            # same as birth but inverse
            proposal_ratio = self.no_deathnodes / len(self.potential_children)
            prior_ratio = (
                self.memoize_arrangements(self.tree, len(self.points)+1-self.kmin) /
                self.memoize_arrangements(self.tree, len(self.points)-self.kmin) )

        else:
            raise Exception("action undefined",self.action)

        return np.log(proposal_ratio * prior_ratio)
    
    
    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        self.coeffs_backup = deepcopy(self.coeffs)
        if self.anisotropic:
            self.coeffs_psi2_backup = deepcopy(self.coeffs_psi2)
            self.coeffs_psi2amp_backup = deepcopy(self.coeffs_psi2amp)
        self.potential_children_backup = self.potential_children.copy()
       
    
    def update_coeff_range(self):
        
        for level in self.coeff_range:
            if level == 0:
                coeff_range = (self.coeff_range[0]['max'] -
                               self.coeff_range[0]['min'])/2.
                # maxcoeff = np.max(np.abs(self.coeffs[level]))
                # mincoeff = np.min(np.abs(self.coeffs[level]))
                # self.coeff_range[level]['max'] = 1.01*maxcoeff
                # self.coeff_range[level]['min'] = 0.99*mincoeff
            else:   
                maxcoeff = np.max(np.abs([self.coeffs[level][0][:],
                                          self.coeffs[level][1][:],
                                          self.coeffs[level][2][:]]))
                maxcoeff = np.max([maxcoeff,0.2*self.coeff_range[level-1]['max']])
                if level>1:
                    maxcoeff = np.min([maxcoeff,0.5*self.coeff_range[level-1]['max']])
                else:
                    maxcoeff = np.min([maxcoeff,0.5*coeff_range])
                self.coeff_range[level]['max'] =  1.1*maxcoeff
                self.coeff_range[level]['min'] = -1.1*maxcoeff
     
        
    def reject_mod(self):
        
        if self.action=='update':
            self.rejected_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 0
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
            
        if self.points_backup is not None:
            self.points = self.points_backup
        if self.coeffs_backup is not None:
            self.coeffs = self.coeffs_backup
        if self.potential_children_backup is not None:
            self.potential_children = self.potential_children_backup
        if self.coeffs_psi2amp_backup is not None:
            self.coeffs_psi2amp = self.coeffs_psi2amp_backup
        if self.coeffs_psi2_backup is not None:
            self.coeffs_psi2 = self.coeffs_psi2_backup
            
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not child in self.potential_children:
        #             raise Exception(f"error after reject {self.action}") 
                    
        self.points_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.point_removed = None
        self.coeffs_backup = None
        self.no_birthnodes = None
        self.no_deathnodes = None
        self.potential_children_backup = None
        self.coeffs_psi2_backup = None
        self.coeffs_psi2amp_backup = None    
        
    def accept_mod(self,selfcheck=False):
        
        # testlist = []
        # for item in self.points:
        #     testlist.append(tuple(item[:-1]))
        # for point in self.points:
        #     if point[0]==0:
        #         continue
        #     if point[-1]==0:
        #         children = self.get_children(point)
        #         for child in children:
        #             if tuple(child[:-1]) in testlist:
        #                 break
        #         else:
        #             print(point)
        #             raise Exception("this node should be marked as leaf! error after",self.action) 
                    
        if self.action=='update':
            self.accepted_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 1
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
            
            if np.random.rand(1)>0.8:
                self.update_coeff_range()
         
        if selfcheck:
            for point in self.points:
                if (self.potential_children[:,:-1]==point[:-1]).all(axis=1).any():
                    idx_child = np.where((self.potential_children[:,:-1]==point[:-1]).all(axis=1))[0]
                    print(idx_child,self.potential_children[idx_child],point)
                    raise Exception(f"point is also in potential children after {self.action}")
                if np.sum((self.points[:,:-1]==point[:-1]).all(axis=1)) > 1:
                    idx_double = np.where((self.points[:,:-1]==point[:-1]).all(axis=1))[0]
                    print(idx_double,self.points[idx_double])
                    raise Exception(f"double points after {self.action}")
            
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not (self.potential_children==child).all(axis=1).any():
        #             raise Exception(f"error after accept {self.action}")
        #         if (self.points==child).all(axis=1).any():
        #             raise Exception(f"error after accept {self.action}, should not be leaf node!")
                   
        # for point in self.points:
        #     if point[0]==0:
        #         continue
        #     parent_idx = self.get_parent_idx(point)
        #     if len(parent_idx)!=1 or self.points[parent_idx,-1]==1:
        #         raise Exception("error after",self.action)
        
        # for point in self.points:
        #     children = self.get_children(point)
        #     isleaf = True
        #     for child in children:
        #         idx_child = np.where((self.points[:,:4]==child[:4]).all(axis=1))[0]
        #         if len(idx_child)>0 and point[-1]==1:
        #             raise Exception()
        #         if len(idx_child)>0:
        #             isleaf = False
        #     if point[-1]==0 and isleaf:
        #         raise Exception("should be leaf")
                
                
        self.points_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.point_removed = None
        self.coeffs_backup = None
        self.no_birthnodes = None
        self.no_deathnodes = None
        self.potential_children_backup = None
        self.coeffs_psi2_backup = None
        self.coeffs_psi2amp_backup = None
        
            
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):

        slowfield = self.get_model()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cbar = ax.pcolormesh(1./slowfield.reshape(self.shape),cmap=plt.cm.seismic_r)
        plt.colorbar(cbar)
        plt.show()
        
        



##############################################################################
##############################################################################
##############################################################################

class wavelets_simpletree(object):
    
    
    def __init__(self,gridpoints,shape,velmin,velmax,init_no_coeffs=None): 
        
        
        self.gridpoints = gridpoints
        self.shape = shape
        
        self.wavelet = 'bior4.4'#'bior4.4'
        decomposition_level = int(np.log2(np.min(self.shape)))
        
        # # wavelet parameterization works with different levels from very rough
        # # to successively finer scales
        # subshape = shape
        # shapes = []
        # while True:
        #     subshape = (int(subshape[0]/2.)+subshape[0]%2,
        #                 int(subshape[1]/2.)+subshape[1]%2)
        #     shapes.append(subshape)
        #     if subshape[0]==1 or subshape[1]==1:
        #         break
        
        self.velmax = velmax
        self.velmin = velmin
        
        velmean = (self.velmax+self.velmin) / 2.
        velamp = (self.velmax-self.velmin) / 2.
        
        
        self.coeff_range = {}
        X = np.reshape(self.gridpoints[:,0],self.shape)
        xrange = X[0,-1]-X[0,0]
        kx = 0.01/xrange
        kxmax = 1/(2*(X[0,1]-X[0,0])) # Nyquist wavenumber
        Y = np.reshape(self.gridpoints[:,1],self.shape)
        yrange = Y[-1,0]-Y[0,0]
        ky = 0.01/yrange
        kymax = 1/(2*(Y[1,0]-Y[0,0]))
        while kx<kxmax and ky<kymax:
            start_mod = velmean + velamp * (np.cos(2*np.pi*kx*X + 
                                                   np.random.uniform(0,2*np.pi)) +
                                            np.cos(2*np.pi*ky*Y +
                                                   np.random.uniform(0,2*np.pi)))
            coeffs = pywt.wavedec2(start_mod, self.wavelet,
                                   level=decomposition_level,
                                   mode='periodization')

            for level in range(len(coeffs)):
                try:
                    self.coeff_range[level]
                except:
                    self.coeff_range[level] = {}
                    self.coeff_range[level]["max"] = -1e99
                    self.coeff_range[level]["min"] = 1e99
                if level==0:
                    self.coeff_range[0]['max'] = np.max([
                        self.coeff_range[0]['max'],np.max(coeffs[level])])
                    self.coeff_range[0]['min'] = np.min([
                        self.coeff_range[0]['min'],np.min(coeffs[level])])
                else:
                    for direction in [0,1,2]:
                        self.coeff_range[level]['max'] = np.max([
                            self.coeff_range[level]['max'],
                            np.max(coeffs[level][direction])])
                        self.coeff_range[level]['min'] = np.min([
                            self.coeff_range[level]['min'],
                            np.min(coeffs[level][direction])]) 
            kx*=1.1
            ky*=1.1
            
        for level in self.coeff_range:
            if level>0:
                self.coeff_range[level]['max'] = np.around(np.max([
                    self.coeff_range[level]['max'],
                    np.abs(self.coeff_range[level]['min'])]),3)/3.
                self.coeff_range[level]['min'] = -self.coeff_range[level]['max']
        
        # test_coeffs = deepcopy(coeffs)
        # for level in range(len(coeffs)):
        #     if level==0:
        #         test_coeffs[level][:] = mmax#self.coeff_range[level]["max"]
        #     else:
        #         for direction in [0,1,2]:
        #             test_coeffs[level][direction][:] = 0.
        #             test_coeffs[level][direction][:] = self.coeff_range[level]["max"]
        
        # #%%
        # mode = "symmetric"
        # coeffs = pywt.wavedec2(np.ones(self.shape)*3.5, self.wavelet,level=None,
        #                             mode=mode)
        # for level,lvlcoeffs in enumerate(coeffs):
        #     print(level,np.shape(lvlcoeffs))
        #     if level==0:
        #         coeffs[level][:] = np.mean(coeffs[level])
        #     else:
        #         coeffs[level][0][:] = coeffs[level][1][:] = coeffs[level][2][:] = 0.
        # coeffs[0][5,5] = 29.
        # velfield = pywt.waverec2(coeffs, self.wavelet, mode=mode)
        # plt.figure()
        # plt.pcolormesh(velfield)
        # plt.colorbar()
        # plt.show()
        # #%%
        
        self.minlevel = 1       
        start_mod = np.random.uniform(low=self.velmin,high=self.velmax,
                                      size=self.shape)
        self.coeffs = pywt.wavedec2(start_mod, self.wavelet,
                                    level=decomposition_level,
                                    mode='periodization')
        
        self.vs = []
        self.points = []
        for level in range(len(self.coeffs)):
            
            if level==0:
                shape = self.coeffs[level].shape
            elif level==1:
                shape = (3,) + shape
            else:
                shape = (3, shape[1]*2,shape[2]*2)
            if np.shape(self.coeffs[level])!=shape:
                raise Exception("bad tree structure")
                
            if level<=self.minlevel:
                indices = np.column_stack(np.unravel_index(
                        np.arange(len(self.coeffs[level][0].flatten())),
                        shape[-2:]))
                if level==0:
                    self.vs.append(np.hstack((self.coeffs[0][0],0.,0.)))
                else:
                    self.vs.append(np.column_stack((
                        self.coeffs[level][0].flatten(),
                        self.coeffs[level][1].flatten(),
                        self.coeffs[level][2].flatten())))
                self.points.append(np.column_stack((np.ones(len(indices))*level,
                                                    indices,
                                                    np.zeros(len(indices)))))
            else:
                self.coeffs[level][0][:] = self.coeffs[level][1][:] = self.coeffs[level][2][:] = 0.
        
        # points has 4 columns: level, yind, xind, isleaf
        self.vs = np.vstack(self.vs)
        self.points = np.vstack(self.points).astype(int)
            
            
        # number of decomposition levels
        self.levels = level+1
        
        print("Warning, setting maximum level to",self.levels-1)
        self.maxlevel = self.levels-1
        
        
        # potential children includes all new nodes that can be chosen from
        # in a birth operation (children of currently active nodes)
        self.potential_children = []
        for point in self.points:
            children = self.get_children(point)
            if len(children)>0:
                self.potential_children.append(children)
        self.potential_children = np.vstack(self.potential_children)
        
            # if len(self.potential_children) == 0:
            #     self.potential_children = self.get_children(point)
            # else:
            #     self.potential_children = np.vstack((self.potential_children,
            #                                          self.get_children(point)))
                
        # dictionary that stores the number of possible arrangements of nodes
        # for a certain tree structure
        self.D = {}
        # getting the tree structure. the first number in the array gives the
        # total number of root nodes. The following number give the number of
        # children for each root node.
        # In a binary tree that has a maximum depth of 3, this would look like
        # self.tree = [1,2,2,2]
        # However, in this case we have already an array at the root, e.g.
        # 6*7 elements, resulting in 42 root nodes. Each root node has 3 child-
        # ren, one for each direction. For each direction, each node has 4
        # children, on the succesively finer grids
        # self.tree = [42, 3, 4, 4, 4]
        self.tree = np.array([len(self.coeffs[0].flatten()),1] + 
                             (self.levels-2)*[4])
        # adapt tree to the minlevel and the maxlevel
        self.tree = np.append(np.product(self.tree[:self.minlevel+1]),
                              self.tree[self.minlevel:self.maxlevel])
        
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp = None
        self.psi2 = None
        self.psi2amp_backup = None
        self.psi2_backup = None  
        self.coeffs_backup = None
        self.potential_children_backup = None
        

        # add points until the number of initial coefficients is reached
        if init_no_coeffs is not None:
            init_coeffs = init_no_coeffs-len(self.vs)
        else:
            init_coeffs = 0
        for i in range(init_coeffs):
            self.add_point()
            self.update_model()
            self.accept_mod()
        
        
        #slowfield = self.get_model()
        #if (1./slowfield > velmax).any() or (1./slowfield < velmin).any():
        #    raise Exception("bad starting model")        
        
        self.acceptance_rate = {}
        self.accepted_steps = {}
        self.rejected_steps = {}
        for level in range(self.levels):
            self.acceptance_rate[level] = np.zeros(100,dtype=int)
            self.accepted_steps[level] = 0
            self.rejected_steps[level] = 0
        
        # # array of decomposed wavelet coefficients
        # self.coeffs = []
        # for i,shape in enumerate(shapes[::-1]):
        #     if i==0:
        #         self.coeffs.append(np.zeros(shape))
        #     self.coeffs.append((np.zeros(shape),np.zeros(shape),np.zeros(shape)))
        
     
        
    def optimize_gridsize(minx,maxx,miny,maxy,xgridspacing,ygridspacing):
        
        # function to optimize the gridsize so that we can construct a tree
        # that has 4 children for every node.
        # During the wavelet decomposition the grid is split in successively
        # rougher grids by taking only every second sample of the original grid
        # 16 -> 8 -> 4 -> 2 -> 1
        # If the gridsize is not a power of 2, this will fail at some point.
        # 18 -> 9 -> !
        # In general, this is not a problem, because the wavelet decomposition
        # works also with non-integer decimations:
        # 18 -> 9 -> 5 -> 3 -> 2 -> 1
        # however, this means that not every node has exactly four children
        # which makes the calculations more difficult
        # this function will therefore try to avoid this by decreasing the
        # grid spacing, i.e. increasing the number of samples
        # 18 samples increased to 20
        # 20 -> 10 -> 5
        # the optimization will stop at some point, meaning that it will not
        # be possible to have only one single root node
        # instead, in the example above, there would be 5 root nodes

        #minx = -110.75607181407213
        #maxx = 104.96901884072297
        #xgridspacing = 1.
        
        xpoints = len(np.arange(minx,maxx+xgridspacing,xgridspacing))
        partition_levels = 0
        while True:
            if xpoints==1:
                xpartitions = int(xpoints * 2**partition_levels)
                break
            if xpoints%2==0:
                xpoints/=2
                partition_levels += 1
            else:
                xpoints = int(xpoints+1)
                xnew = np.linspace(minx,maxx,
                                   xpoints * 2**partition_levels)
                dxnew = xnew[1]-xnew[0]
                if dxnew < 0.5*xgridspacing:
                    xpoints-=1
                    xnew = np.linspace(minx,maxx,
                                       xpoints * 2**partition_levels)
                    dxnew = xnew[1]-xnew[0]
                    xpartitions = (xpoints * 2**partition_levels)
                    break
                
        ypoints = len(np.arange(miny,maxy+ygridspacing,ygridspacing))
        partition_levels = 0
        while True:
            if ypoints==1:
                ypartitions = int(ypoints * 2**partition_levels)
                break
            if ypoints%2==0:
                ypoints/=2
                partition_levels += 1
            else:
                ypoints = int(ypoints+1)
                ynew = np.linspace(miny,maxy,
                                   ypoints * 2**partition_levels)
                dynew = ynew[1]-ynew[0]
                if dynew < 0.5*ygridspacing:
                    ypoints-=1
                    ynew = np.linspace(miny,maxy,
                                       ypoints * 2**partition_levels)
                    dynew = ynew[1]-ynew[0]
                    ypartitions = (ypoints * 2**partition_levels)
                    break
                
        return xpartitions,ypartitions
                
                
    
    def memorize_arrangements(self,tree,k):
        
        # based on "Geophysical imaging using trans-dimensional trees" by
        # Hawkins and Sambridge, 2015, appendix A

        # this function will successively fill the D dictionary where the
        # number of possible arrangements in a tree is stored
        # for example in a ternary tree with 3 active nodes, there are 12
        # possible arrangements
        # tree = (1, 3, 3, 3, 3); k=3
        # D[((1,3,3,3,3),3)] = 12    

        kmax = 0
        for i in range(len(tree)):
            kmax += np.product(tree[:i+1])
        if k==0 or k==kmax:
            return 1
        if k<0 or k>kmax:
            return 0
        
        try:
            return self.D[(tuple(tree),k)]
        except:
            j = tree[0]
            if j==1:
                A = tree[1:]
                self.D[(tuple(tree),k)] = self.memorize_arrangements(A,k-1)
            elif j%2==1:
                A = tree.copy()
                A[0] = 1
                B = tree.copy()
                B[0] -= 1
                self.D[(tuple(tree),k)] = self.compute_subtrees(A,B,k)
            else:
                A = tree.copy()
                A[0] = tree[0]/2
                B = tree.copy()
                B[0] = tree[0]/2
                self.D[(tuple(tree),k)] = self.compute_subtrees(A,B,k)
            
        return self.D[(tuple(tree),k)]
            
            
    def compute_subtrees(self,A,B,k):
        arrangements = 0
        for i in range(k+1):
            a = self.memorize_arrangements(A,i)
            b = self.memorize_arrangements(B,k-i)
            arrangements += a*b
        return arrangements
    
        # # for comparison, possible arrangements in a ternary tree:
        # def compute_ternary(k):
        #     if k<=0:
        #         return 1
        #     arrangements = 0
        #     for i in range(k):
        #         a = compute_ternary(i)
        #         subsum = 0
        #         for j in range(k-i):
        #             b = compute_ternary(j)
        #             c = compute_ternary(k-i-j-1)
        #             subsum += b*c
        #         arrangements += a*subsum
        #     return arrangements
                
         
 
    def get_children(self,point):
        
        level,yind,xind,isleaf = point
        
        if level+1 >= self.maxlevel or level<self.minlevel:
            return []
        
        # a level 0 node has 3 children, x-direction child, y-direction child
        # and diagional child
        if level==0:
            return [(level+1,yind,xind,1)]
        
        # a higher level node has 4 children, the direction is the same as the
        # parent direction but the "pixel" gets split into 4 smaller ones
        children = [(level+1,yind*2,xind*2,1)]
        # if the model dimensions are not a power of two, the border nodes
        # may have less than 4 children
        if yind*2+1 < self.coeffs[level+1][0].shape[0]:
            children.append((level+1,yind*2+1,xind*2,1))
        if xind*2+1 < self.coeffs[level+1][0].shape[1]:
            children.append((level+1,yind*2,xind*2+1,1))
        if len(children)==3:
            children.append((level+1,yind*2+1,xind*2+1,1))
        return np.vstack(children)
    

    def get_parent_idx(self,point):
        
        level,yind,xind,isleaf = point
        
        if level==0:
            raise Exception("level 0 has no parent")
            
        if level==1:
            idx_parent = np.where((self.points[:,0]==level-1)*
                                  (self.points[:,1]==yind)*
                                  (self.points[:,2]==xind))[0]
        
        else:
            idx_parent = np.where((self.points[:,0]==level-1)*
                                  (self.points[:,1]==int(yind/2))*
                                  (self.points[:,2]==int(xind/2)))[0]
        
        if len(idx_parent) > 1:
            raise Exception("there should only be one parent!")
        
        return idx_parent
    
        
    def vs_update(self,idx,dcoeff,backup=True):
        
        if backup:
            self.backup_mod()
            #self.vs_backup = self.vs.copy()
            
        self.action = 'update'
            
        # with increasing level, the coefficient ranges become smaller
        # this happens with an approximate factor of 0.5
        # this way, the acceptance at each level should be approximately equal
        level = self.points[idx,0]
        if level==0:
            direction=0
        else:
            direction = np.random.randint(0,3)
        self.vs[idx,direction] += dcoeff#*1.2**(8-level) # 5 could be replaced by any value, but should be adjusted to the proposal ratio
        
        self.idx_mod = idx
        
        
    def add_point(self,birth_prop='uniform',anisotropic=False,backup=True):
        
        if birth_prop!='uniform':
            raise Exception("only uniform birth proposals are currently implemented")
        
        # do a better implementation of the case where the maximum tree 
        # height is reached
        if len(self.potential_children) == 0:
            self.idx_mod_gpts = []
            return
        
        if backup:
            self.backup_mod()
            
        self.action='birth'
           
        # randomly choose one of the potential children
        idx_birth = np.random.randint(0,len(self.potential_children))
        
        self.points = np.vstack((self.points,self.potential_children[idx_birth]))
        level = self.points[-1,0]
        self.vs = np.vstack((self.vs,np.random.uniform(
            self.coeff_range[level]['min'],self.coeff_range[level]['max'],3)))
        
        # needed for the proposal ratio calculation
        self.no_birthnodes = len(self.potential_children)
        
        self.potential_children = np.delete(self.potential_children,
                                            idx_birth,axis=0)
        #self.potential_children.remove(tuple(self.points[-1]))
        children = self.get_children(self.points[-1])
        if len(children) > 0:
            self.potential_children = np.vstack((self.potential_children,
                                                 children))
        idx_parent = self.get_parent_idx(self.points[-1])
        self.points[idx_parent,-1] = 0


    def remove_point(self,anisotropic=False,backup=True):
        
        if backup:
            self.backup_mod()
            
        self.action='death'
           
        # randomly choose one of the leaf nodes to remove
        # leaf nodes are those that have no children
        self.no_deathnodes = np.sum(self.points[:,-1])
        if self.no_deathnodes > 0:
            idx_death = np.random.choice(np.where(self.points[:,-1])[0])
        else:
            return (np.nan,np.nan)
        
        self.point_removed = self.points[idx_death]
        if self.point_removed[0]<=self.minlevel:
            raise Exception("should not happen!")
        vs_removed = self.vs[idx_death]
        
        self.points = np.delete(self.points,idx_death,axis=0)
        self.vs = np.delete(self.vs,idx_death,axis=0)
        
        # the children of the removed point have to be removed from the list
        # of potential children (in a birth step a child from potential
        # children is chosen)
        children = self.get_children(self.point_removed)
        if len(children)>0:
            idx_remove = np.where(
                (self.potential_children[:,0]==children[0,0]) * 
                np.in1d(self.potential_children[:,1],children[:,1]) *
                np.in1d(self.potential_children[:,2],children[:,2]))[0]
            self.potential_children = np.delete(self.potential_children,
                                                idx_remove,0)
        # the removed point has to be added to the list of potential children
        self.potential_children = np.vstack((self.potential_children,
                                             self.point_removed))
        # the parent might become a leaf node now (removable), do a check
        if self.point_removed[0]>self.minlevel+1:
            # check if the removed node had any siblings 
            level,yind,xind,isleaf = self.point_removed      
            coeff_sum_siblings = (
                self.coeffs[level][0][int(yind/2)*2,int(xind/2)*2] +
                self.coeffs[level][0][int(yind/2)*2+1,int(xind/2)*2] +
                self.coeffs[level][0][int(yind/2)*2,int(xind/2)*2+1] +
                self.coeffs[level][0][int(yind/2)*2+1,int(xind/2)*2+1] -
                self.coeffs[level][0][yind,xind])
            if coeff_sum_siblings == 0.: # i.e. no siblings
                # make parent node a leaf node
                idx_parent = self.get_parent_idx(self.point_removed)[0]
                self.points[idx_parent,-1] = 1

        
        return self.point_removed,vs_removed
         
        
    def get_model(self,points=None,vs=None,psi2amp=None,psi2=None,anisotropic=False):
        
        coeffs = deepcopy(self.coeffs)
        
        # first, set everything to zero
        for level in range(self.levels):
            if level==0:
                coeffs[level][:] = 0.
            else:
                for direction in [0,1,2]:
                    coeffs[level][direction][:] = 0.
            
        # then, fill in the coefficients from the self.vs array 
        for i,(level,yind,xind,isleaf) in enumerate(self.points):
            if level==0:
                coeffs[level][yind,xind] = self.vs[i,0]
            else:
                for direction in [0,1,2]:
                    coeffs[level][direction][yind,xind] = self.vs[i,direction]
            
        # reconstruct the field from the coefficients
        velfield = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        return 1./velfield.flatten()

        if False:
            self.vs[:] = 0.
            self.coeffs[1][0][:] = 0.
            self.coeffs[1][1][:] = 0.
            self.coeffs[1][2][:] = 0.
            #self.vs[32] = vs[32]
            self.coeffs[0] = self.vs.reshape(self.coeffs[0].shape)
            self.coeffs[1][2][(8,11)] = 10
            self.coeffs[2][0][:] = 0.
            #self.coeffs[2][2][(2*8,2*9+1)] = 10.
            velfield = pywt.waverec2(self.coeffs, self.wavelet, mode='periodization')
            plt.figure()
            plt.pcolormesh(velfield)
            plt.colorbar()
            plt.show()

        
        
    def update_model(self,fields=None,anisotropic=False):
        
        if anisotropic:
            slowfield, psi2amp, psi2 = fields
        else:
            slowfield = fields
                       
        if self.action == 'update':
            level,yind,xind,isleaf = self.points[self.idx_mod]
            coefficient = self.vs[self.idx_mod]
            if ((coefficient>self.coeff_range[level]['max']).any() or
                (coefficient<self.coeff_range[level]['min']).any()):
                return np.nan
        elif self.action == 'birth':
            level,yind,xind,isleaf = self.points[-1]
            coefficient = self.vs[-1]
        elif self.action == 'death':
            level,yind,xind,isleaf = self.point_removed
            coefficient = [0.,0.,0.]
        else:
            raise Exception("cannot update model, action undefined")
               
        if level == 0:
            self.coeffs[level][yind,xind] = coefficient[0]
        else:
            for direction in [0,1,2]:
                self.coeffs[level][direction][yind,xind] = coefficient[direction]
             
        velfield = pywt.waverec2(self.coeffs, self.wavelet, mode='periodization')
        
        #if np.max(velfield)>self.velmax or np.min(velfield)<self.velmin:
        #    return np.nan
        
        slowfield_prop = 1./velfield.flatten()
        
        self.idx_mod_gpts = np.where(slowfield_prop!=slowfield)[0]
        
        return slowfield_prop
       
    
    def get_prior_proposal_ratio(self):
        
        if self.action == 'update':
            # if the number of active nodes is unchanged and only the 
            # coefficients are changed, the probability from going from one
            # coefficient to another one is equal to the reverse step
            return 0
        elif self.action == 'birth':
            # assuming we draw from a uniform distribution
            # nominator: number of possible nodes to choose from during birth
            # denominator: number of possible death nodes to choose from after
            #              birth
            proposal_ratio = self.no_birthnodes / np.sum(self.points[:,-1])
            # eqs 13 & 16 from Hawkins & Sambridge 2015
            prior_ratio = (
                self.memorize_arrangements(self.tree, len(self.points)-1) /
                self.memorize_arrangements(self.tree, len(self.points)) )
        elif self.action == 'death':
            # same as birth but inverse
            proposal_ratio = self.no_deathnodes / len(self.potential_children)
            prior_ratio = (
                self.memorize_arrangements(self.tree, len(self.points)+1) /
                self.memorize_arrangements(self.tree, len(self.points)) )

        return np.log(proposal_ratio * prior_ratio)
    
    
    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        self.vs_backup = self.vs.copy()
        self.coeffs_backup = deepcopy(self.coeffs)
        self.potential_children_backup = self.potential_children.copy()
        if self.psi2amp is not None:
            self.psi2amp_backup = self.psi2amp.copy()
        if self.psi2 is not None:
            self.psi2_backup = self.psi2.copy()
        
        
    def reject_mod(self):
        
        if self.action=='update':
            self.rejected_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 0
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
            
        if self.points_backup is not None:
            self.points = self.points_backup
        if self.vs_backup is not None:
            self.vs = self.vs_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        if self.coeffs_backup is not None:
            self.coeffs = self.coeffs_backup
        if self.potential_children_backup is not None:
            self.potential_children = self.potential_children_backup
         
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not child in self.potential_children:
        #             raise Exception(f"error after reject {self.action}") 
                    
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.point_removed = None
        self.coeffs_backup = None
        self.no_birthnodes = None
        self.no_deathnodes = None
        self.potential_children_backup = None
    
        
    def accept_mod(self,selfcheck=False):
        
        # testlist = []
        # for item in self.points:
        #     testlist.append(tuple(item[:-1]))
        # for point in self.points:
        #     if point[0]==0:
        #         continue
        #     if point[-1]==0:
        #         children = self.get_children(point)
        #         for child in children:
        #             if tuple(child[:-1]) in testlist:
        #                 break
        #         else:
        #             print(point)
        #             raise Exception("this node should be marked as leaf! error after",self.action) 
                    
        if self.action=='update':
            self.accepted_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 1
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
         
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not child in self.potential_children:
        #             raise Exception(f"error after accept {self.action}")
                   
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.point_removed = None
        self.coeffs_backup = None
        self.no_birthnodes = None
        self.no_deathnodes = None
        self.potential_children_backup = None
        
            
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):

        slowfield = self.get_model()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cbar = ax.pcolormesh(1./slowfield.reshape(self.shape),cmap=plt.cm.seismic_r)
        plt.colorbar(cbar)
        plt.show()
        
        
        
##############################################################################
##############################################################################
##############################################################################

class wavelets_notree(object):
    
    
    def __init__(self,gridpoints,shape,velmin,velmax,init_no_coeffs=None): 
        
        
        self.gridpoints = gridpoints
        self.shape = shape
        
        self.wavelet = 'bior4.4'
        decomposition_level = int(np.log2(np.min(self.shape)))
        
        # # wavelet parameterization works with different levels from very rough
        # # to successively finer scales
        # subshape = shape
        # shapes = []
        # while True:
        #     subshape = (int(subshape[0]/2.)+subshape[0]%2,
        #                 int(subshape[1]/2.)+subshape[1]%2)
        #     shapes.append(subshape)
        #     if subshape[0]==1 or subshape[1]==1:
        #         break
        
        self.velmax = velmax
        self.velmin = velmin
        
        velmean = (self.velmax+self.velmin) / 2.
        velamp = (self.velmax-self.velmin) / 2.
        
        
        self.coeff_range = {}
        X = np.reshape(self.gridpoints[:,0],self.shape)
        xrange = X[0,-1]-X[0,0]
        kx = 0.01/xrange
        kxmax = 1/(2*(X[0,1]-X[0,0])) # Nyquist wavenumber
        Y = np.reshape(self.gridpoints[:,1],self.shape)
        yrange = Y[-1,0]-Y[0,0]
        ky = 0.01/yrange
        kymax = 1/(2*(Y[1,0]-Y[0,0]))
        while kx<kxmax and ky<kymax:
            start_mod = velmean + velamp * (np.cos(2*np.pi*kx*X + 
                                                   np.random.uniform(0,2*np.pi)) +
                                            np.cos(2*np.pi*ky*Y +
                                                   np.random.uniform(0,2*np.pi)))
            coeffs = pywt.wavedec2(start_mod, self.wavelet,
                                   level=decomposition_level,
                                   mode='periodization')

            for level in range(len(coeffs)):
                try:
                    self.coeff_range[level]
                except:
                    self.coeff_range[level] = {}
                    self.coeff_range[level]["max"] = -1e99
                    self.coeff_range[level]["min"] = 1e99
                if level==0:
                    self.coeff_range[0]['max'] = np.max([
                        self.coeff_range[0]['max'],np.max(coeffs[level])])
                    self.coeff_range[0]['min'] = np.min([
                        self.coeff_range[0]['min'],np.min(coeffs[level])])
                else:
                    for direction in [0,1,2]:
                        self.coeff_range[level]['max'] = np.max([
                            self.coeff_range[level]['max'],
                            np.max(coeffs[level][direction])])
                        self.coeff_range[level]['min'] = np.min([
                            self.coeff_range[level]['min'],
                            np.min(coeffs[level][direction])]) 
            kx*=1.1
            ky*=1.1
            
        for level in self.coeff_range:
            if level>0:
                self.coeff_range[level]['max'] = np.around(np.max([
                    self.coeff_range[level]['max'],
                    np.abs(self.coeff_range[level]['min'])]),3)/3.
                self.coeff_range[level]['min'] = -self.coeff_range[level]['max']
        
        

        start_mod = np.random.uniform(low=self.velmin,high=self.velmax,
                                      size=self.shape)
        self.coeffs = pywt.wavedec2(start_mod, self.wavelet,
                                    level=decomposition_level,
                                    mode='periodization')

        for level in range(len(self.coeffs)):
            if level==0:
                self.vs = self.coeffs[level].flatten()
                indices = np.column_stack(np.unravel_index(
                    np.arange(len(self.vs)),self.coeffs[level].shape))
                shape = self.coeffs[level].shape
            else:
                if level==1:
                    shape = (3,) + shape
                else:
                    shape = (3, shape[1]*2,shape[2]*2)
                if np.shape(self.coeffs[level])!=shape:
                    raise Exception("bad tree structure")
                self.coeffs[level][0][:] = self.coeffs[level][1][:] = self.coeffs[level][2][:] = 0.
            
        # number of decomposition levels
        self.levels = level+1
        
        #print("Warning, setting maximum level to",self.levels-2)
        self.maxlevel = self.levels
        
        # points has 5 columns: level, direction (hor,ver,diag), yind, xind, isleaf
        # at level=0, the direction does not apply
        self.points = np.column_stack((np.zeros_like(indices),indices)).astype(int)
        
        # potential children includes all new nodes that can be chosen from
        # in a birth operation
        self.free_nodes = []
        for level in range(self.levels):
            if level==0 or level>self.maxlevel:
                continue
            for direction in range(3):
                for yind in range(np.shape(self.coeffs[level][direction])[0]):
                    for xind in range(np.shape(self.coeffs[level][direction])[1]):
                        self.free_nodes.append([level,direction,yind,xind])
        self.free_nodes = np.array(self.free_nodes)
            
        
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp = None
        self.psi2 = None
        self.psi2amp_backup = None
        self.psi2_backup = None  
        self.coeffs_backup = None
        self.free_nodes_backup = None


        # add points until the number of initial coefficients is reached
        if init_no_coeffs is not None:
            init_coeffs = init_no_coeffs-len(self.vs)
        else:
            init_coeffs = 0
        for i in range(init_coeffs):
            self.add_point()
            self.update_model()
            self.accept_mod()
            
        #slowfield = self.get_model()
        #if (1./slowfield > velmax).any() or (1./slowfield < velmin).any():
        #    raise Exception("bad starting model")        
        
        self.acceptance_rate = {}
        self.accepted_steps = {}
        self.rejected_steps = {}
        for level in range(self.levels):
            self.acceptance_rate[level] = np.zeros(100,dtype=int)
            self.accepted_steps[level] = 0
            self.rejected_steps[level] = 0
        
        # # array of decomposed wavelet coefficients
        # self.coeffs = []
        # for i,shape in enumerate(shapes[::-1]):
        #     if i==0:
        #         self.coeffs.append(np.zeros(shape))
        #     self.coeffs.append((np.zeros(shape),np.zeros(shape),np.zeros(shape)))
        
     
        
    def optimize_gridsize(minx,maxx,miny,maxy,xgridspacing,ygridspacing):
        
        # function to optimize the gridsize so that we can construct a tree
        # that has 4 children for every node.
        # During the wavelet decomposition the grid is split in successively
        # rougher grids by taking only every second sample of the original grid
        # 16 -> 8 -> 4 -> 2 -> 1
        # If the gridsize is not a power of 2, this will fail at some point.
        # 18 -> 9 -> !
        # In general, this is not a problem, because the wavelet decomposition
        # works also with non-integer decimations:
        # 18 -> 9 -> 5 -> 3 -> 2 -> 1
        # however, this means that not every node has exactly four children
        # which makes the calculations more difficult
        # this function will therefore try to avoid this by decreasing the
        # grid spacing, i.e. increasing the number of samples
        # 18 samples increased to 20
        # 20 -> 10 -> 5
        # the optimization will stop at some point, meaning that it will not
        # be possible to have only one single root node
        # instead, in the example above, there would be 5 root nodes

        #minx = -110.75607181407213
        #maxx = 104.96901884072297
        #xgridspacing = 1.
        
        xpoints = len(np.arange(minx,maxx+xgridspacing,xgridspacing))
        partition_levels = 0
        while True:
            if xpoints==1:
                xpartitions = int(xpoints * 2**partition_levels)
                break
            if xpoints%2==0:
                xpoints/=2
                partition_levels += 1
            else:
                xpoints = int(xpoints+1)
                xnew = np.linspace(minx,maxx,
                                   xpoints * 2**partition_levels)
                dxnew = xnew[1]-xnew[0]
                if dxnew < 0.5*xgridspacing:
                    xpoints-=1
                    xnew = np.linspace(minx,maxx,
                                       xpoints * 2**partition_levels)
                    dxnew = xnew[1]-xnew[0]
                    xpartitions = (xpoints * 2**partition_levels)
                    break
                
        ypoints = len(np.arange(miny,maxy+ygridspacing,ygridspacing))
        partition_levels = 0
        while True:
            if ypoints==1:
                ypartitions = int(ypoints * 2**partition_levels)
                break
            if ypoints%2==0:
                ypoints/=2
                partition_levels += 1
            else:
                ypoints = int(ypoints+1)
                ynew = np.linspace(miny,maxy,
                                   ypoints * 2**partition_levels)
                dynew = ynew[1]-ynew[0]
                if dynew < 0.5*ygridspacing:
                    ypoints-=1
                    ynew = np.linspace(miny,maxy,
                                       ypoints * 2**partition_levels)
                    dynew = ynew[1]-ynew[0]
                    ypartitions = (ypoints * 2**partition_levels)
                    break
                
        return xpartitions,ypartitions
                

    def vs_update(self,idx,dcoeff,backup=True):
        
        if backup:
            self.backup_mod()
            #self.vs_backup = self.vs.copy()
            
        self.action = 'update'
            
        # with increasing level, the coefficient ranges become smaller
        # this happens with an approximate factor of 0.5
        # this way, the acceptance at each level should be approximately equal
        level = self.points[idx,0]
        self.vs[idx] += dcoeff#*1.2**(8-level) # 5 could be replaced by any value, but should be adjusted to the proposal ratio
        
        self.idx_mod = idx      
        
        
    def add_point(self,birth_prop='uniform',anisotropic=False,backup=True):
        
        if birth_prop!='uniform':
            raise Exception("only uniform birth proposals are currently implemented")
        
        # do a better implementation of the case where the maximum tree 
        # height is reached
        if len(self.free_nodes) == 0:
            self.idx_mod_gpts = []
            return
        
        if backup:
            self.backup_mod()
            
        self.action='birth'
           
        # randomly choose one of the potential children
        #idx_birth = np.random.randint(0,len(self.free_nodes))
        # put higher weight on nodes closer to the root of the tree
        probs = self.free_nodes[:,0]**(-3.)/np.sum(self.free_nodes[:,0]**(-3.))
        idx_birth = np.random.choice(np.arange(len(self.free_nodes)),
                                     p=probs)
        self.birth_prob = probs[idx_birth] # to calculate the proposal ratio
        
        
        self.points = np.vstack((self.points,self.free_nodes[idx_birth]))
        level = self.points[-1,0]
        self.vs = np.append(self.vs,np.random.uniform(
            self.coeff_range[level]['min'],self.coeff_range[level]['max']))
                
        self.free_nodes = np.delete(self.free_nodes,idx_birth,axis=0)


    def remove_point(self,anisotropic=False,backup=True):
        
        if backup:
            self.backup_mod()
            
        self.action='death'
           
        # randomly choose one of the leaf nodes to remove
        # leaf nodes are those that have no children
        self.no_deathnodes = len(self.points)-1
        if self.no_deathnodes > 0:
            idx_death = np.random.randint(1,len(self.points))
        else:
            return (np.nan,np.nan)
        
        self.point_removed = self.points[idx_death]
        vs_removed = self.vs[idx_death]
        
        self.points = np.delete(self.points,idx_death,axis=0)
        self.vs = np.delete(self.vs,idx_death)
        
        self.free_nodes = np.vstack((self.free_nodes,self.point_removed))

        # the probability that this node was chosen in an inverse (birth) step
        self.birth_prob = (self.free_nodes[-1,0]**(-3.) / 
                           np.sum(self.free_nodes[:,0]**(-3.)))

        
        return self.point_removed,vs_removed
         
        
    def get_model(self,points=None,vs=None,psi2amp=None,psi2=None,anisotropic=False):
        
        coeffs = deepcopy(self.coeffs)
        
        # first, set everything to zero
        for level in range(self.levels):
            if level==0:
                coeffs[level][:] = 0.
            else:
                for direction in [0,1,2]:
                    coeffs[level][direction][:] = 0.
            
        # then, fill in the coefficients from the self.vs array 
        for i,(level,direction,yind,xind) in enumerate(self.points):
            if level==0:
                coeffs[level][yind,xind] = self.vs[i]
            else:
                coeffs[level][direction][yind,xind] = self.vs[i]
            
        # reconstruct the field from the coefficients
        velfield = pywt.waverec2(coeffs, self.wavelet, mode='periodization')
        return 1./velfield.flatten()

        if False:
            self.vs[:] = 0.
            self.coeffs[1][0][:] = 0.
            self.coeffs[1][1][:] = 0.
            self.coeffs[1][2][:] = 0.
            #self.vs[32] = vs[32]
            self.coeffs[0] = self.vs.reshape(self.coeffs[0].shape)
            self.coeffs[1][2][(8,11)] = 10
            self.coeffs[2][0][:] = 0.
            #self.coeffs[2][2][(2*8,2*9+1)] = 10.
            velfield = pywt.waverec2(self.coeffs, self.wavelet, mode='periodization')
            plt.figure()
            plt.pcolormesh(velfield)
            plt.colorbar()
            plt.show()

        
        
    def update_model(self,fields=None,anisotropic=False):
           
        if anisotropic:
            slowfield, psi2amp, psi2 = fields
        else:
            slowfield = fields
                       
        if self.action == 'update':
            level,direction,yind,xind = self.points[self.idx_mod]
            coefficient = self.vs[self.idx_mod]
            if (coefficient>self.coeff_range[level]['max'] or
                coefficient<self.coeff_range[level]['min']):
                return np.nan
        elif self.action == 'birth':
            level,direction,yind,xind = self.points[-1]
            coefficient = self.vs[-1]
        elif self.action == 'death':
            level,direction,yind,xind = self.point_removed
            coefficient = 0.
        else:
            raise Exception("cannot update model, action undefined")
                        
            
        if level == 0:
            self.coeffs[level][yind,xind] = coefficient
        else:
            self.coeffs[level][direction][yind,xind] = coefficient
            
        velfield = pywt.waverec2(self.coeffs, self.wavelet, mode='periodization')
        
        #if np.max(velfield)>self.velmax or np.min(velfield)<self.velmin:
        #    return np.nan
        
        slowfield_prop = 1./velfield.flatten()
        
        self.idx_mod_gpts = np.where(slowfield_prop!=slowfield)[0]
        
        return slowfield_prop
       
    
    def get_prior_proposal_ratio(self):
        
        # if the prior on the number of nodes and the coefficient in each
        # node is uniform, and if the proposal is also uniform, all terms
        # cancel out and the proposal_ratio*prior_ratio=1
        # see eqs. 3.10, 3.23, 3.32 & 3.33 from T. Bodin (thesis)
        # return 1
        
        # if we make use of a proposal in which it is more likely to draw
        # a birth node closer to the root of the tree, it is necessary to
        # calculate the prior and proposal ratios
                
        if self.action == 'update':
            # if the number of active nodes is unchanged and only the 
            # coefficients are changed, the probability from going from one
            # coefficient to another one is equal to the reverse step
            return 0
        elif self.action == 'birth':
            # compare eq. 3.32 from thesis Bodin (in eq 3.32, n is the number
            # of nodes before the birth operation, here len(self.points) is
            # after the birth operation, i.e. len(self.points)=n+1)
            prior_ratio = len(self.points)/(len(self.free_nodes)+1)
            # the birth node is chosen according to eq. 25 in Hawkins & Sambr.
            # death nodes are chosen uniformly from all available points
            proposal_ratio = 1./(len(self.points)*self.birth_prob)
        elif self.action == 'death':
            # same as birth but inverse
            prior_ratio = (len(self.free_nodes)+1)/len(self.points)
            proposal_ratio = self.birth_prob*(len(self.points)+1)

        return np.log(proposal_ratio * prior_ratio)
    
    
    def backup_mod(self):
        
        self.points_backup = self.points.copy()
        self.vs_backup = self.vs.copy()
        self.free_nodes_backup = self.free_nodes.copy()
        self.coeffs_backup = deepcopy(self.coeffs)
        if self.psi2amp is not None:
            self.psi2amp_backup = self.psi2amp.copy()
        if self.psi2 is not None:
            self.psi2_backup = self.psi2.copy()
        
        
    def reject_mod(self):
        
        if self.action=='update':
            self.rejected_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 0
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
            
        if self.points_backup is not None:
            self.points = self.points_backup
        if self.vs_backup is not None:
            self.vs = self.vs_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        if self.coeffs_backup is not None:
            self.coeffs = self.coeffs_backup
        if self.free_nodes_backup is not None:
            self.free_nodes = self.free_nodes_backup
         
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not child in self.potential_children:
        #             raise Exception(f"error after reject {self.action}") 
                    
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.point_removed = None
        self.coeffs_backup = None
        self.free_nodes_backup = None
        self.birth_prob = None
    
        
    def accept_mod(self,selfcheck=False):
        
        # testlist = []
        # for item in self.points:
        #     testlist.append(tuple(item[:-1]))
        # for point in self.points:
        #     if point[0]==0:
        #         continue
        #     if point[-1]==0:
        #         children = self.get_children(point)
        #         for child in children:
        #             if tuple(child[:-1]) in testlist:
        #                 break
        #         else:
        #             print(point)
        #             raise Exception("this node should be marked as leaf! error after",self.action) 
                    
        if self.action=='update':
            self.accepted_steps[self.points[self.idx_mod,0]] += 1
            self.acceptance_rate[self.points[self.idx_mod,0]][0] = 1
            self.acceptance_rate[self.points[self.idx_mod,0]] = np.roll(
                self.acceptance_rate[self.points[self.idx_mod,0]],1)
         
        # for point in self.points:
        #     if point[-1]==0:
        #         continue
        #     children = self.get_children(point)
        #     for child in children:
        #         if not child in self.potential_children:
        #             raise Exception(f"error after accept {self.action}")
                   
        self.points_backup = None
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = None
        self.idx_mod = None
        self.psi2amp_backup = None
        self.psi2_backup = None
        self.point_removed = None
        self.coeffs_backup = None
        self.free_nodes_backup = None
        self.birth_prob = None
        
            
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):

        slowfield = self.get_model()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cbar = ax.pcolormesh(1./slowfield.reshape(self.shape),cmap=plt.cm.seismic_r)
        plt.colorbar(cbar)
        plt.show()


##############################################################################
##############################################################################
##############################################################################

class nocells(object):
    
    def __init__(self,gridpoints,shape,velmin,velmax,init_no_points,
                 psi2ampmin=0.,psi2ampmax=0.1,anisotropic=False,
                 gridspacing_staggered=None): 
        
        self.gridpoints = gridpoints
        self.shape = shape
        
        self.minx = np.min(self.gridpoints[:,0])
        self.maxx = np.max(self.gridpoints[:,0])
        self.miny = np.min(self.gridpoints[:,1])
        self.maxy = np.max(self.gridpoints[:,1])

        if gridspacing_staggered is not None:
            dx = np.diff(np.unique(self.gridpoints[:,0]))[0]
            dy = np.diff(np.unique(self.gridpoints[:,1]))[0]
            if gridspacing_staggered > np.max([dx,dy]):
                print("Warning! introducing staggered rough grid")
                dx = dy = gridspacing_staggered
                x = np.arange(self.minx+np.random.uniform(-0.5,0.5)*dx,
                              self.maxx+np.random.uniform(-0.5,0.5)*dx+dx,
                              dx)
                y = np.arange(self.miny+np.random.uniform(-0.5,0.5)*dy,
                              self.maxy+np.random.uniform(-0.5,0.5)*dy+dy,dy)
                X,Y = np.meshgrid(x,y)
                gridpoints = np.column_stack((X.flatten(),Y.flatten()))
                kdtree = KDTree(gridpoints)
                nndist,nnidx = kdtree.query(self.gridpoints)
                self.gridpoints = gridpoints[nnidx]
        
        self.points = self.gridpoints
        # if velmin < 0 and velmax > 0:
        #     self.slomin = 1./velmin
        #     self.slomax = 1./velmax
        # else:
        #     self.slomin = 1./velmax
        #     self.slomax = 1./velmin
        # self.slorange = self.slomax-self.slomin
        self.velmax = velmax
        self.velmin = velmin
        self.velrange = self.velmax-self.velmin        

        self.action = None
        self.idx_mod_gpts = None
        self.propdist = 'uniform' # will be adapted when running
        #self.vs = np.random.uniform(self.velmin,self.velmax,len(self.gridpoints))
        self.vs = np.ones(len(self.gridpoints))*np.mean([self.velmin,self.velmax])
        self.vs_backup = None
        self.psi2amp = np.zeros_like(self.vs)
        self.psi2amp_backup = None
        self.psi2 = np.random.uniform(-np.pi/2.,np.pi/2.,len(self.gridpoints))
        self.psi2_backup = None        
        
        # self.smooth_model = False
        # if self.smooth_model:
        #     self.smoothing_matrix = lil_matrix((len(self.gridpoints),len(self.gridpoints)))
        #     #self.smoothing_matrix[[np.arange(len(self.gridpoints),dtype=int),
        #     #                       np.arange(len(self.gridpoints),dtype=int)]] = 1.
        #     print("Trying to apply model smoothing with 25km radius")
        #     for idx in range(len(self.gridpoints)):
        #         dists = np.sqrt(np.sum((self.gridpoints[idx]-self.gridpoints)**2,axis=1))
        #         weights = trunc_normal(dists,0,25,sig_trunc=2)
        #         # remove elements with very small influence to reduce the matrix size
        #         weights[weights<0.01*weights.max()] = 0.
        #         # normalize to 1
        #         weights /= weights.sum()
        #         self.smoothing_matrix[idx] = weights
        #     self.smoothing_matrix = self.smoothing_matrix.tocsc()       
        
        # initialize model
        k = 0
        while True:
            idx = np.random.randint(0,len(self.points))
            dvs = np.random.uniform(0,self.velrange)
            valid = self.vs_update(idx,dvs)
            if valid:
                self.accept_mod()
                k += 1
            else:
                self.reject_mod()
            if k == init_no_points:
                break
        
        if anisotropic:
            self.psi2ampmin=psi2ampmin
            self.psi2ampmax=psi2ampmax
            

    def psi2amp_update(self,idx,delta,backup=True):
           
        if backup:
            self.psi2amp_backup = self.psi2amp.copy()
            
        self.action='psi2amp_update'
        self.idx_mod = idx
        
        self.idx_mod_gpts = self.get_modified_gridpoints(idx)
        if len(self.idx_mod_gpts)==0:
            return False
        
        self.psi2amp[self.idx_mod_gpts] += delta
        if ((self.psi2amp[self.idx_mod_gpts]<self.psi2ampmin).any() or 
            (self.psi2amp[self.idx_mod_gpts]>self.psi2ampmax).any() or
            (self.vs[self.idx_mod_gpts]*(1+self.psi2amp[self.idx_mod_gpts])>self.velmax).any() or
            (self.vs[self.idx_mod_gpts]*(1-self.psi2amp[self.idx_mod_gpts])<self.velmin).any()):
            self.psi2amp[self.idx_mod_gpts] -= delta
            return False
        
        return True
    
        
    def psi2_update(self,idx,delta,backup=True):
        
        if backup:
            self.psi2_backup = self.psi2.copy()
            
        self.action='psi2_update'
        self.idx_mod = idx

        self.idx_mod_gpts = self.get_modified_gridpoints(idx)
        if len(self.idx_mod_gpts)==0:
            return False
        
        # modulo makes sure it's always in the 0-2pi range
        # the Python modulo convention also correctly treats negative angles
        self.psi2[self.idx_mod_gpts] = (self.psi2[self.idx_mod_gpts]+delta)%(2*np.pi)
        
        return True
        

    def vs_update(self,idx,dvs,backup=True):
            
        if backup:
            self.vs_backup = self.vs.copy()
            
        self.action='velocity_update'
        self.idx_mod = idx
            
        self.idx_mod_gpts = self.get_modified_gridpoints(idx)
        if len(self.idx_mod_gpts)==0:
            return False
        
        self.vs[self.idx_mod_gpts] += dvs
        if ((self.vs[self.idx_mod_gpts]<self.velmin).any() or 
            (self.vs[self.idx_mod_gpts]>self.velmax).any()):
            self.vs[self.idx_mod_gpts] -= dvs
            return False            
        if self.psi2amp is not None:
            if ((self.vs[self.idx_mod_gpts]*(1+self.psi2amp[self.idx_mod_gpts])>self.velmax).any() or
                (self.vs[self.idx_mod_gpts]*(1-self.psi2amp[self.idx_mod_gpts])<self.velmin).any()):
                self.vs[self.idx_mod_gpts] -= dvs
                return False
        
        return True
        

    def get_modified_gridpoints(self,idx):
        
        idx2d = np.unravel_index(idx, self.shape)
        idx_arr = np.reshape(np.arange(len(self.gridpoints),dtype=int),self.shape)
        
        height = np.random.randint(1,int(self.shape[0]/10))
        width = np.random.randint(1,int(self.shape[1]/10))

        idx_mod_gpts = idx_arr[np.max([0,idx2d[0]-height]):idx2d[0]+height,np.max([0,idx2d[1]-width]):idx2d[1]+width]
            
        return idx_mod_gpts.flatten()
    
        
    def get_prior_proposal_ratio(self):
        
        # this function returns the prior ratio times the proposal ratio which
        # is needed to calculate the acceptance probability
        # is always 0 since the forward and backward steps are equally likely
        return 0 
    
    def get_model(self,points=None,vs=None,psi2amp=None,psi2=None,anisotropic=False):
        
        if vs is None:
            vs = self.vs
        if psi2amp is None:
            psi2amp = self.psi2amp
        if psi2 is None:
            psi2 = self.psi2
        
        if anisotropic:
            return 1./vs,psi2amp,psi2
        else:
            return 1./vs
    
    def update_model(self,fields=None,anisotropic=False):
        
        if anisotropic:
            return 1./self.vs,self.psi2amp,self.psi2
        else:
            return 1./self.vs
        
        
    def reject_mod(self):
        
        if self.vs_backup is not None:
            self.vs = self.vs_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
            
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = 1e99
        self.idx_mod = 1e99
        self.psi2amp_backup = None
        self.psi2_backup = None
        
    def accept_mod(self,selfcheck=False):
                    
        self.vs_backup = None
        self.action = None
        self.idx_mod_gpts = 1e99
        self.idx_mod = 1e99
        self.psi2amp_backup = None
        self.psi2_backup = None
        
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cbar = ax.pcolormesh(np.unique(self.gridpoints[:,0]),
                             np.unique(self.gridpoints[:,1]),
                             np.reshape(self.vs,self.shape),
                             cmap=plt.cm.coolwarm_r,shading='nearest')
        #ax.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.')
        if idx_mod_gpts is not None:
            ax.plot(self.gridpoints[idx_mod_gpts,0],self.gridpoints[idx_mod_gpts,1],'rx',zorder=3)
        plt.colorbar(cbar)
        plt.show()
        
##############################################################################
##############################################################################
##############################################################################

class blocks(object):
    
    def __init__(self,gridpoints,shape,velmin,velmax,init_no_points,
                 psi2ampmin=0.,psi2ampmax=0.1,anisotropic=False,
                 gridspacing_staggered=None): 
        
        self.gridpoints = gridpoints
        self.shape = shape
        self.idx_arr = np.reshape(np.arange(len(self.gridpoints),dtype=int),self.shape)
        
        self.minx = np.min(self.gridpoints[:,0])
        self.maxx = np.max(self.gridpoints[:,0])
        self.miny = np.min(self.gridpoints[:,1])
        self.maxy = np.max(self.gridpoints[:,1])

        if gridspacing_staggered is not None:
            dx = np.diff(np.unique(self.gridpoints[:,0]))[0]
            dy = np.diff(np.unique(self.gridpoints[:,1]))[0]
            if gridspacing_staggered > np.max([dx,dy]):
                print("Warning! introducing staggered rough grid")
                dx = dy = gridspacing_staggered
                x = np.arange(self.minx+np.random.uniform(-0.5,0.5)*dx,
                              self.maxx+np.random.uniform(-0.5,0.5)*dx+dx,
                              dx)
                y = np.arange(self.miny+np.random.uniform(-0.5,0.5)*dy,
                              self.maxy+np.random.uniform(-0.5,0.5)*dy+dy,dy)
                X,Y = np.meshgrid(x,y)
                gridpoints = np.column_stack((X.flatten(),Y.flatten()))
                kdtree = KDTree(gridpoints)
                nndist,nnidx = kdtree.query(self.gridpoints)
                self.gridpoints = gridpoints[nnidx]
        
        # if velmin < 0 and velmax > 0:
        #     self.slomin = 1./velmin
        #     self.slomax = 1./velmax
        # else:
        #     self.slomin = 1./velmax
        #     self.slomax = 1./velmin
        # self.slorange = self.slomax-self.slomin
        self.velmax = velmax
        self.velmin = velmin
        self.velrange = self.velmax-self.velmin     
        
        self.anisotropic = anisotropic
        if self.anisotropic:
            raise Exception("anisotropy currently not implemented for block parameterization")
            self.psi2ampmin=psi2ampmin
            self.psi2ampmax=psi2ampmax

        self.action = None
        self.idx_mod_gpts = None
        self.propvelstd_dimchange = 'uniform' # will be adapted when running
        #self.vs = np.random.uniform(self.velmin,self.velmax,len(self.gridpoints))
        self.vsfield = np.ones(len(self.gridpoints))*np.mean([self.velmin,self.velmax])
        self.vsfield_backup = None
        self.psi2ampfield = np.zeros_like(self.vsfield)
        self.psi2ampfield_backup = None
        self.psi2field = np.random.uniform(-np.pi/2.,np.pi/2.,len(self.gridpoints))
        self.psi2field_backup = None
        
        self.vs = np.empty((0,))
        self.vs_backup = None
        self.points = np.empty((0,4),dtype=int)
        self.points_backup = None
        self.psi2amp = np.empty((0,))
        self.psi2amp_backup = None
        self.psi2 = np.empty((0,))
        self.psi2_backup = None
        
        # self.smooth_model = False
        # if self.smooth_model:
        #     self.smoothing_matrix = lil_matrix((len(self.gridpoints),len(self.gridpoints)))
        #     #self.smoothing_matrix[[np.arange(len(self.gridpoints),dtype=int),
        #     #                       np.arange(len(self.gridpoints),dtype=int)]] = 1.
        #     print("Trying to apply model smoothing with 25km radius")
        #     for idx in range(len(self.gridpoints)):
        #         dists = np.sqrt(np.sum((self.gridpoints[idx]-self.gridpoints)**2,axis=1))
        #         weights = trunc_normal(dists,0,25,sig_trunc=2)
        #         # remove elements with very small influence to reduce the matrix size
        #         weights[weights<0.01*weights.max()] = 0.
        #         # normalize to 1
        #         weights /= weights.sum()
        #         self.smoothing_matrix[idx] = weights
        #     self.smoothing_matrix = self.smoothing_matrix.tocsc()       
        
        # initialize model
        k = 0
        while True:
            valid = self.add_point(anisotropic=anisotropic)
            if valid:
                self.accept_mod()
                k += 1
            else:
                self.reject_mod()
            if k == init_no_points:
                break
            

    def psi2amp_update(self,idx,delta,backup=True):
           
        raise Exception("anisotropy not implemented yet")
        
        if backup:
            self.psi2amp_backup = self.psi2amp.copy()
            
        self.action='psi2amp_update'
        self.idx_mod = idx
        
        self.idx_mod_gpts = self.get_modified_gridpoints(idx)
        if len(self.idx_mod_gpts)==0:
            return False
        
        self.psi2amp[self.idx_mod_gpts] += delta
        if ((self.psi2amp[self.idx_mod_gpts]<self.psi2ampmin).any() or 
            (self.psi2amp[self.idx_mod_gpts]>self.psi2ampmax).any() or
            (self.vs[self.idx_mod_gpts]*(1+self.psi2amp[self.idx_mod_gpts])>self.velmax).any() or
            (self.vs[self.idx_mod_gpts]*(1-self.psi2amp[self.idx_mod_gpts])<self.velmin).any()):
            self.psi2amp[self.idx_mod_gpts] -= delta
            return False
        
        return True
    
        
    def psi2_update(self,idx,delta,backup=True):
        
        if backup:
            self.psi2_backup = self.psi2.copy()
            
        self.action='psi2_update'
        self.idx_mod = idx

        self.idx_mod_gpts = self.get_modified_gridpoints(idx)
        if len(self.idx_mod_gpts)==0:
            return False
        
        # modulo makes sure it's always in the 0-2pi range
        # the Python modulo convention also correctly treats negative angles
        self.psi2[self.idx_mod_gpts] = (self.psi2[self.idx_mod_gpts]+delta)%(2*np.pi)
        
        return True
        

    def vs_update(self,idx,dvs,backup=True):
            
        if backup:
            self.vs_backup = self.vs.copy()
            self.vsfield_backup = self.vsfield.copy()
            
        self.action='velocity_update'
        self.idx_mod = idx
        
        idx00,idx01,idx10,idx11 = self.points[idx]
            
        self.idx_mod_gpts = self.idx_arr[idx00:idx01,idx10:idx11].flatten()
        if len(self.idx_mod_gpts)==0:
            return False
        
        self.vs[idx] += dvs
        self.vsfield[self.idx_mod_gpts] += dvs
        
        valid = self.check_valid()
        if not valid:
            return False
        else:
            return True
    
    def add_point(self,birth_prop='uniform',anisotropic=False,backup=True):
        
        if backup:
            self.vs_backup = self.vs.copy()
            self.points_backup = self.points.copy()
            self.vsfield_backup = self.vsfield.copy()
            
        self.action='birth'
        self.idx_mod = len(self.points)
        
        idx_center = np.random.choice(self.idx_arr.flatten())
        idx_center = np.unravel_index(idx_center, self.shape)
        height = np.random.randint(1,int(self.shape[0]/4))
        width = np.random.randint(1,int(self.shape[1]/4))
        
        idx00 = int(np.max([0,idx_center[0]-height]))
        idx01 = int(np.min([self.shape[0],idx_center[0]+height]))
        idx10 = int(np.max([0,idx_center[1]-width]))
        idx11 = int(np.min([self.shape[1],idx_center[1]+width]))
        
        self.points = np.vstack((self.points,np.array([idx00,idx01,idx10,idx11])))
        
        if birth_prop=='uniform':
            dvs = np.random.uniform(-self.velrange/2,self.velrange/2)
            self.vs = np.append(self.vs,dvs)
        else:
            raise Exception("not implemented")
            
        self.idx_mod_gpts = self.idx_arr[idx00:idx01,idx10:idx11].flatten()
        self.vsfield[self.idx_mod_gpts] += dvs
        
        valid = self.check_valid()
        if not valid:
            return False
        else:
            return True

        return True
            
        
    def remove_point(self,anisotropic=False,backup=True):
        
        if backup:
            self.vs_backup = self.vs.copy()
            self.points_backup = self.points.copy()
            self.vsfield_backup = self.vsfield.copy()
        
        self.action='death'
        
        if anisotropic:
            # choose only points without anisotropy
            ind_pnts = np.where(self.psi2amp == 0.)[0]
            if len(ind_pnts) > 0:
                self.idx_mod = np.random.choice(ind_pnts)
            else:
                return False
        else:
            # choose point to remove randomly
            self.idx_mod = np.random.randint(0,len(self.points))
        
        vs_remove = self.vs[self.idx_mod]
        idx00,idx01,idx10,idx11 = self.points[self.idx_mod]
        
        self.points = np.delete(self.points,self.idx_mod,axis=0)
        self.vs = np.delete(self.vs,self.idx_mod)
        if anisotropic:
            self.psi2amp = np.delete(self.psi2amp,self.idx_mod)
            self.psi2 = np.delete(self.psi2,self.idx_mod)
        
        self.idx_mod_gpts = self.idx_arr[idx00:idx01,idx10:idx11].flatten()

        # now compare the velocity of the removed point with the velocity
        # at the empty spot (inverse birth operation)
        if self.propvelstd_dimchange != 'uniform' and len(self.idx_mod_gpts)>0:
            self.prop_dv = np.mean(self.vs[self.grid_nnidx[self.idx_mod_gpts]]) - vs_remove
        else:
            self.prop_dv = 0.
            
        self.vsfield[self.idx_mod_gpts] -= vs_remove
        valid = self.check_valid()
        if not valid:
            return False
        else:
            return True
        
        return True
    

    def move_point(self,propmovestd,index=None,backup=True):

        if backup:
            self.points_backup = self.points.copy()
            self.vsfield_backup = self.vsfield.copy()
            
        self.action='move'
        if index is None:
            index = np.random.randint(0,len(self.points))
        self.idx_mod = index
        
        dvs = self.vs[self.idx_mod]
        idx00,idx01,idx10,idx11 = self.points[self.idx_mod]
        pnt_old = np.array([idx00+idx01,idx10+idx11])
        idx_mod_gpts0 = self.idx_arr[idx00:idx01,idx10:idx11].flatten()
        self.vsfield[idx_mod_gpts0] -= dvs
        
        mod = np.random.normal(loc=0,scale=propmovestd,size=4)
        idx00 = int(np.max([0,idx00+mod[0]]))
        idx01 = int(np.min([self.shape[0],idx01+mod[1]]))
        idx00,idx01 = np.sort([idx00,idx01])
        idx10 = int(np.max([0,idx10+mod[2]]))
        idx11 = int(np.min([self.shape[1],idx11+mod[3]]))
        idx10,idx11 = np.sort([idx10,idx11])
        pnt_new = np.array([idx00+idx01,idx10+idx11])
        
        self.points[self.idx_mod] = np.array([idx00,idx01,idx10,idx11])
        
        idx_mod_gpts1 = self.idx_arr[idx00:idx01,idx10:idx11].flatten()
        self.vsfield[idx_mod_gpts1] += dvs
        
        self.idx_mod_gpts = np.array(list(set(idx_mod_gpts0).symmetric_difference(set(idx_mod_gpts1))))
        
        if len(self.idx_mod_gpts)==0 or len(idx_mod_gpts1)==0:
            return (np.nan,np.nan)
        
        if idx00>=idx01 or idx10>=idx11:
            print(idx00,idx01,idx10,idx11)
            print(self.points[self.idx_mod])
            raise Exception("bad indices")
        
        valid = self.check_valid()
        if not valid:
            return (np.nan,np.nan)
        else:
            return (pnt_old,pnt_new)
    

    def check_valid(self):

        if ((self.vsfield[self.idx_mod_gpts]<self.velmin).any() or 
            (self.vsfield[self.idx_mod_gpts]>self.velmax).any()):
            return False            
        if self.psi2amp is not None and len(self.psi2amp)>0:
            if ((self.vsfield[self.idx_mod_gpts]*(1+self.psi2ampfield[self.idx_mod_gpts])>self.velmax).any() or
                (self.vsfield[self.idx_mod_gpts]*(1-self.psi2ampfield[self.idx_mod_gpts])<self.velmin).any()):
                return False
        
        return True
            
    def get_prior_proposal_ratio(self):
        
        # this function returns the prior ratio times the proposal ratio which
        # is needed to calculate the acceptance probability  
        if 'update' in self.action or self.action == 'move':
            # is always log(1)=0, unless delayed rejection which is currently
            # included in the main script
            return 0
        elif self.action == 'birth':
            if self.anisotropic:
                aniso_factor = np.log(len(self.points)/(len(self.points)+1))
            else:
                aniso_factor = 0.
            if self.propvelstd_dimchange == 'uniform':
                # if we draw from a uniform prior, everything cancels out
                return aniso_factor + 0
            else:
                # see for example equation A.34 of the PhD thesis of Thomas Bodin
                return (aniso_factor + 
                    np.log(self.propvelstd_dimchange*np.sqrt(2.*np.pi) / self.velrange) +  
                    (self.prop_dv**2 / (2*self.propvelstd_dimchange**2)))
        elif self.action == 'death':
            if self.anisotropic:
                aniso_factor = np.log(len(self.points)/(len(self.points)-1))
            else:
                aniso_factor = 0.
            if self.propvelstd_dimchange == 'uniform':
                return aniso_factor + 0
            else:
                return ( aniso_factor + 
                    np.log(self.velrange/(self.propvelstd_dimchange*np.sqrt(2.*np.pi))) -
                    (self.prop_dv**2 / (2*self.propvelstd_dimchange**2)))
        else:
            raise Exception("action undefined")
    
    def get_model(self,points=None,vs=None,psi2amp=None,psi2=None,anisotropic=False):
        
        if points is None:
            points = self.points
        if vs is None:
            vs = self.vs
        if psi2amp is None:
            psi2amp = self.psi2amp
        if psi2 is None:
            psi2 = self.psi2
            
        velfield = np.ones(self.shape)*np.mean([self.velmin,self.velmax])
        psi2ampfield = np.zeros_like(velfield)
        psi2field = np.zeros_like(velfield)
        for i in range(len(points)):
            idx00,idx01,idx10,idx11 = points[i].astype(int)
            velfield[idx00:idx01,idx10:idx11] += vs[i]
            if anisotropic:
                psi2ampfield[idx00:idx01,idx10:idx11] += psi2amp[i]
                psi2field[idx00:idx01,idx10:idx11] += psi2[i]

        self.vsfield = velfield.flatten()

        if anisotropic:
            self.psi2ampfield = psi2ampfield
            self.psi2field = psi2field
            return 1./velfield.flatten(),psi2ampfield.flatten(),psi2field.flatten()
        else:
            return 1./velfield.flatten()
    
    def update_model(self,fields=None,anisotropic=False):
        
        if anisotropic:
            slowfield,psi2ampfield,psi2field = fields
        else:
            slowfield = fields
            
        if not np.array_equal(slowfield,1./self.vsfield_backup):
            raise Exception()
        
        if anisotropic:
            return 1./self.vsfield,self.psi2ampfield,self.psi2field
        else:
            return 1./self.vsfield
        
        
    def reject_mod(self):
        
        if self.vs_backup is not None:
            self.vs = self.vs_backup
        if self.psi2amp_backup is not None:
            self.psi2amp = self.psi2amp_backup
        if self.psi2_backup is not None:
            self.psi2 = self.psi2_backup
        if self.points_backup is not None:
            self.points = self.points_backup
        if self.vsfield_backup is not None:
            self.vsfield = self.vsfield_backup
        if self.psi2ampfield_backup is not None:
            self.psi2ampfield = self.psi2ampfield_backup
        if self.psi2field_backup is not None:
            self.psi2field = self.psi2field_backup
            
        self.points_backup = None
        self.vs_backup = None
        self.vsfield_backup = None
        self.psi2ampfield_backup = None
        self.psi2field_backup = None        
        self.action = None
        self.idx_mod_gpts = 1e99
        self.idx_mod = 1e99
        self.psi2amp_backup = None
        self.psi2_backup = None
        
    def accept_mod(self,selfcheck=False):
                    
        self.points_backup = None
        self.vs_backup = None
        self.vsfield_backup = None
        self.psi2ampfield_backup = None
        self.psi2field_backup = None        
        self.action = None
        self.idx_mod_gpts = 1e99
        self.idx_mod = 1e99
        self.psi2amp_backup = None
        self.psi2_backup = None
        
    def plot(self,idx_mod_gpts=None,idx_neighbor_points=None):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cbar = ax.pcolormesh(np.unique(self.gridpoints[:,0]),
                             np.unique(self.gridpoints[:,1]),
                             np.reshape(self.vs,self.shape),
                             cmap=plt.cm.coolwarm_r,shading='nearest')
        #ax.plot(self.gridpoints[:,0],self.gridpoints[:,1],'k.')
        if idx_mod_gpts is not None:
            ax.plot(self.gridpoints[idx_mod_gpts,0],self.gridpoints[idx_mod_gpts,1],'rx',zorder=3)
        plt.colorbar(cbar)
        plt.show()
  

# functions
def trunc_normal(x, mu, sig, sig_trunc=3):
    
    if sig==0.:
        sig = 1e-10
    gauss = (1. / (np.sqrt(2 * np.pi) * sig) * np.exp(
                -0.5 * np.square(x - mu) / np.square(sig)))

    bounds = (mu-sig_trunc*sig,mu+sig_trunc*sig)
    gauss[x < bounds[0]] = 0
    gauss[x > bounds[1]] = 0

    return gauss/np.sum(gauss)
