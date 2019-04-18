#clustering_single_variable.py
#history
#Copy of 2019-02-clustering_couvreux, the purpose is to cluster all cells in which the variable exceeds a specific value. 
#Pretty much exclusively used to cluster clouds together using ql and a ql_min to detect clouds.  
#Saves lists of the cells and their indexes into a pkl file that is then later loaded to calculate the cluster properties or plot them. 
#Most important part is the cluster_3D_v2 function

#ToDo expand to run automatically for a list of directories, am confused that this wasn't already implemented


#Settings:
var_name = 'ql'
var_min = 1e-6
directory_in = '/data/testbed/lasso/sims/'
dates  = ['20160611']
directory_out = '/data/testbed/lasso/clustering/'
boundary_periodic =1

#var_name = 'w'
#var_min = 0.1



import numpy as np
import math
from netCDF4 import Dataset
import os

#from unionfind import Unionfind
from cusize_functions import *
import time as ttiimmee
import sys
import pickle





for date in dates:
    directory = directory_in+date+'/'
    for filename in os.listdir(directory):
        if filename.endswith('.nc') and filename.startswith(var_name):
            clustering_file = var_name+'_'+date+'_clustering.pkl'
            print('to be saved to: ',clustering_file)
            file_var  =  Dataset(directory+filename,read='r') 
            timesteps = len(file_var.variables['time'][:])
            #timesteps = 4
            nz,nx,ny = get_zxy_dimension(directory+filename,var_name)
            cloud_cell_list_time = []
            idx_3d_cloudy_cells_time = []
            print('looking at file ',filename,' with so many timesteps: ',timesteps)
    
           
            
            for t in range(timesteps):
                
                print('clustering timestep :',t, ' of ',timesteps)
                
                
                #First to get binary field of where the variable exceeds the minimum value
                var_binary_3d = grab_3d_binary_field(file_var,t,var_name,var_min) 
                
                               
                print('at timestep ',t,' total number of cells which fulfil ' + var_name + ' greater than ' + str(var_min) + ' : ',len(var_binary_3d[var_binary_3d>0]))
                print('which is in fraction',len(var_binary_3d[var_binary_3d>0])/(nx*ny*nz))
                if np.max(var_binary_3d)>0:
                    time1 = ttiimmee.time()
                    cloud_cell_list,idx_3d_cloudy_cells  = cluster_3D_v2(var_binary_3d,boundary_periodic)
                    time2 = ttiimmee.time()
                    print(' cluster_3D_v2 took',(time2-time1),'at timestep ',t)
                    print(' for this many clusters',len(cloud_cell_list))
                    cloud_cell_list_time.append(cloud_cell_list)
                    idx_3d_cloudy_cells_time.append(idx_3d_cloudy_cells)
                else:
                    cloud_cell_list_time.append([])
                    idx_3d_cloudy_cells_time.append([])
                #Saving clustering\n,
            with open(directory_out+clustering_file, 'wb') as f: 
               print('saved clustering results via pickle in ',directory_out+clustering_file)
            
               pickle.dump([ cloud_cell_list_time,idx_3d_cloudy_cells_time],f)



