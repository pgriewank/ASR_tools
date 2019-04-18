#history
#19-02 trying to filter all the smallest clusters and those above the CBL with cluster_3D_v2_filt
#will have to add a CBL detection though

#Detects updrafts with the help of the couvreux scaler c_s
#We will first try it with the basic assumption: 
#c_s>mean(c_s)+max(sig_s_c,sig_min) & w>0

#We later filter out later all updrafts that don't start below a specific layer and have a too small vertical extent

import numpy as np
import math
from netCDF4 import Dataset
import os

#from unionfind import Unionfind
from cusize_functions import *
import time as ttiimmee
import sys
import pickle

#Settings:
sig_number = 2.0 #how many std deviations difference are used to determine what counts as a plume
ql_min     = 1e-6    #Minimum cloud liquid water content
w_var_cbl  = 0.08   #Minimum vertical velocity variance used to determine cbl height
w_min      =0.0
directory_in  = '/data/testbed/lasso/sims/'
directory_out = '/data/testbed/lasso/clustering/'
dates  = ['bomex','20160611_micro']
boundary_periodic =1
sig_s_c_min_min = 1e-30
z_cbl=10 #Minimum layer of convective boundary layer depth. Set to 10 because  BOMEX case has very low variances in the lowest layers. 




for date in dates:
    directory = directory_in+date+'/'

    for filename in os.listdir(directory):
        if filename.endswith('.nc') and filename.startswith('couvreux'):
            clustering_file = 'couvreux_'+date+'_clustering_20sig_filt_v2.pkl'
            print('to be saved to: ',clustering_file)
            file_c_s  =  Dataset(directory+filename,read='r') 
            file_w   =  Dataset(directory+'w.nc',read='r') 
            file_ql  =  Dataset(directory+'ql.nc',read='r') 
            timesteps = len(file_ql.variables['time'][:])
            #timesteps = 4
            nz,nx,ny = get_zxy_dimension(directory+filename,'couvreux')
            cloud_cell_list_time = []
            idx_3d_cloudy_cells_time = []
            print('looking at file ',filename,' with so many timesteps: ',timesteps)
            #loading dales profile to determine minimum base height
            
            if date == 'bomex':
                file_micro    =  Dataset(directory_in+date+'/bomex.default.0000000.nc',read='r')
            else: 
                file_micro    =  Dataset(directory_in+date+'/testbed.default.0000000.nc',read='r')
    
           
            #Just for development purposes
            sig_c_s_z_t  = np.zeros((timesteps,nz))
            mean_c_s_z_t = np.zeros((timesteps,nz))
            
            for t in range(timesteps):
                
                print('clustering timestep :',t, ' of ',timesteps)
                
                #Calculate CBL height index z_pbl to filter out all clusters which do not extend below the PBL
                #The two requirements are that variance w >0.08 and ql < 1e-6
                ql = file_micro['ql'][t,:]
                w2 = file_micro['w2'][t,:]
                w_var = 1.
                ql_lvl = 0.
                while w_var > w_var_cbl and ql_lvl<1e-6:
                    z_cbl += 1
                    w_var = w2[z_cbl]
                    ql_lvl = ql[z_cbl]
                print(' cbl height index: ',z_cbl, ', w var: ',str(w_var)[:5],', mean ql:', "{:.1e}".format(ql_lvl))
                
                
                #First to get binary field of where couvreux fulfills c_s>mean(c_s)+max(sig_s_c,sig_min)
                c_s_3d = grab_3d_field(file_c_s,t,'couvreux') 
                #For the meantime we are limiting c_s to be at least zero to deal with pesky negative values
                c_s_3d[c_s_3d<0.] = 0.0
                
                c_s_binary_3d = np.zeros((nz,nx,ny), dtype=np.int)  
                sig_c_s_z = np.zeros(nz)
                mean_c_s_z = np.zeros(nz)
                
                
                
                for k in range(nz):
                    sig_c_s_z[k]=np.sqrt(np.var(c_s_3d[k,:,:]))
                    mean_c_s_z[k] = np.mean(c_s_3d[k,:,:])
                    sig_s_c_min = np.mean(sig_c_s_z[k])*0.05
                    sig_s_c_min=max(sig_s_c_min,sig_s_c_min_min)
                    c_s_binary_3d[k,:,:] = np.where(c_s_3d[k,:,:]>(mean_c_s_z[k]+max(sig_c_s_z[k],sig_s_c_min)*sig_number),1,0) 
                
    
    
    
                #Just for development purposes
                sig_c_s_z_t[t,:] = sig_c_s_z 
                mean_c_s_z_t[t,:] = mean_c_s_z 
                
                #ql_binary_3d = grab_3d_binary_field(file_ql,t,'ql',ql_min)
                w_binary_3d = grab_3d_binary_field(file_w,t,'w',w_min)
                w_cs_binary_3d = c_s_binary_3d * w_binary_3d
                
                print('at timestep ',t,' total number of cells which fulfil w>0 and c_s conditions: ',len(w_cs_binary_3d[w_cs_binary_3d>0]))
                print('which is in fraction',str(len(w_cs_binary_3d[w_cs_binary_3d>0])/(nx*ny*nz))[:5])
                if np.max(w_cs_binary_3d)>0:
                    time1 = ttiimmee.time()
                    #cloud_cell_list,idx_3d_cloudy_cells  = cluster_3D_v2(w_cs_binary_3d,boundary_periodic)
                    cloud_cell_list,idx_3d_cloudy_cells  = cluster_3D_v2_filt(w_cs_binary_3d,boundary_periodic,z_cbl=z_cbl)
                    time2 = ttiimmee.time()
                    print(' cluster_3D_v2_filt took',int((time2-time1)/60.),' seconds at timestep ',t)
                    print(' for ',len(cloud_cell_list),' clusters which have an 8 cell extent and start below ',z_cbl,' levels ',len(cloud_cell_list),'and this many clouds number ',idx_3d_cloudy_cells.shape[1])
                    cloud_cell_list_time.append(cloud_cell_list)
                    idx_3d_cloudy_cells_time.append(idx_3d_cloudy_cells)
                else:
                    cloud_cell_list_time.append([])
                    idx_3d_cloudy_cells_time.append([])
                #Saving clustering\n,
            with open(directory_out+clustering_file, 'wb') as f: 
               print('saved clustering results via pickle in ',directory_out+clustering_file)
            
               pickle.dump([ cloud_cell_list_time,idx_3d_cloudy_cells_time],f)
    
    
    
