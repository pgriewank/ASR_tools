#cusize_functions.py
#intended as a general collection of functions used to load LES data and cluster it




import numpy as np
import math
from netCDF4 import Dataset
import os
from plot_plume_functions import *


try:
    from unionfind import UnionFind
except:
    print('hope you dont need unionfind')
import time

###########################################################################################################################
#INPUT OUTPUT
###########################################################################################################################
def get_zxy_dimension(filename,var_name):
    #Grabs the 3D field of the given file and varname, rearranges it to z,x,y, and returns the dimenions of z,x,y in that order.
    
    time=0
    #first checks if time is a variable, then uses the variable time to select the right timestep
    var_file =  Dataset(filename,read='r')
    var_dim = var_file.variables[var_name].dimensions
    if 'time' in var_dim:
        idx_time = var_dim.index('time')
        if idx_time == 0:
            var = var_file.variables[var_name][time,:]
        if idx_time == 1:
            var = var_file.variables[var_name][:,time,:,:]
        if idx_time == 2:
            var = var_file.variables[var_name][:,:,time,:]
        if idx_time == 3:
            var = var_file.variables[var_name][:,:,:,time]
            
        #now we convert the var_dim tuple to a list remove time from var_dim so it can be used to rearrange z,x, and y
        var_dim_space = list(var_dim)
        var_dim_space.remove('time')
        
    else: 
        var = var_file.variables[var_name][:]
        
        #now we convert the var_dim tuple to a list
        var_dim_space = list(var_dim)
            
    if var_dim_space[2]=='zt':
        var_dim_space = [var_dim_space[2],var_dim_space[1],var_dim_space[0]]
        var = var.swapaxes(0,2)
    if var_dim_space[2]=='zm':
        var_dim_space = [var_dim_space[2],var_dim_space[1],var_dim_space[0]]
        var = var.swapaxes(0,2)
    if var_dim_space[2]=='xt':
        var_dim_space = [var_dim_space[0],var_dim_space[2],var_dim_space[1]]
        var = var.swapaxes(1,2)
    if var_dim_space[2]=='x':
        var_dim_space = [var_dim_space[0],var_dim_space[2],var_dim_space[1]]
        var = var.swapaxes(1,2)
    
    
        
    return var.shape



def grab_3d_field(var_file,time,var_name):
    #Grabs and returns the 3D field of the given variable at the given timestep
    #also makes sure to rearrange the matrix into z, x, y 
    
        #first checks if time is a variable, then uses the variable time to select the right timestep
    var_dim = var_file.variables[var_name].dimensions
    if 'time' in var_dim:
        idx_time = var_dim.index('time')
        if idx_time == 0:
            var = var_file.variables[var_name][time,:]
        if idx_time == 1:
            var = var_file.variables[var_name][:,time,:,:]
        if idx_time == 2:
            var = var_file.variables[var_name][:,:,time,:]
        if idx_time == 3:
            var = var_file.variables[var_name][:,:,:,time]
            
        #now we convert the var_dim tuple to a list remove time from var_dim so it can be used to rearrange z,x, and y
        var_dim_space = list(var_dim)
        var_dim_space.remove('time')
        
    else: 
        var = var_file.variables[var_name][:]
        
        #now we convert the var_dim tuple to a list
        var_dim_space = list(var_dim)
            
    #rearange to z,x,y
    #print(var.shape)

    #print(var_dim_space)
    if var_dim_space[2]=='zt':
        var_dim_space = [var_dim_space[2],var_dim_space[1],var_dim_space[0]]
        var = var.swapaxes(0,2)
    if var_dim_space[2]=='zm':
        var_dim_space = [var_dim_space[2],var_dim_space[1],var_dim_space[0]]
        var = var.swapaxes(0,2)
    if var_dim_space[2]=='xt':
        var_dim_space = [var_dim_space[0],var_dim_space[2],var_dim_space[1]]
        var = var.swapaxes(1,2)
    if var_dim_space[2]=='x':
        var_dim_space = [var_dim_space[0],var_dim_space[2],var_dim_space[1]]
        var = var.swapaxes(1,2)
    
    #print(var_dim_space)

    return var



def grab_2d_binary_field(var_file,time,var_name,var_min,flip_flag=0):
    #Grabs the 3D field of the given variable at the given timestep and returns a 2D binary 0/1 field
    #the threshold value used is so far always a minimal value
    #also makes sure to rearrange the matrix into z, x, y 
    #Changed 2019-02 added invert flag. If set to something other than zero it looks for things smaller rather than bigger the var_min
    
    #first checks if time is a variable, then uses the variable time to select the right timestep
    var_dim = var_file.variables[var_name].dimensions
    if 'time' in var_dim:
        idx_time = var_dim.index('time')
        if idx_time == 0:
            var = var_file.variables[var_name][time,:]
        if idx_time == 1:
            var = var_file.variables[var_name][:,time,:,:]
        if idx_time == 2:
            var = var_file.variables[var_name][:,:,time,:]
        if idx_time == 3:
            var = var_file.variables[var_name][:,:,:,time]
            
        #now we convert the var_dim tuple to a list remove time from var_dim so it can be used to rearrange z,x, and y
        var_dim_space = list(var_dim)
        var_dim_space.remove('time')
        
    else: 
        var = var_file.variables[var_name][:]
        
        #now we convert the var_dim tuple to a list
        var_dim_space = list(var_dim)
            
    #rearange to z,x,y
    if var_dim_space[2]=='zt':
        var_dim_space = [var_dim_space[2],var_dim_space[1],var_dim_space[0]]
        var = var.swapaxes(0,2)
    if var_dim_space[2]=='zm':
        var_dim_space = [var_dim_space[2],var_dim_space[1],var_dim_space[0]]
        var = var.swapaxes(0,2)
    if var_dim_space[2]=='xt':
        var_dim_space = [var_dim_space[0],var_dim_space[2],var_dim_space[1]]
        var = var.swapaxes(1,2)
    if var_dim_space[2]=='x':
        var_dim_space = [var_dim_space[0],var_dim_space[2],var_dim_space[1]]
        var = var.swapaxes(1,2)
    
    var_proj = np.sum(var,axis=0)
    if flip_flag==0:
        var_binary = np.where(var_proj>var_min,1,0)
    else:
        var_binary = np.where(var_proj<var_min,1,0)
    return var_binary
    


def grab_3d_binary_field(var_file,time,var_name,var_min,flip_flag=0):
    #Grabs the 3D field of the given variable at the given timestep and returns a binary 0/1 field
    #the threshold value used is so far always a minimal value
    #also makes sure to rearrange the matrix into z, x, y 
    #Changed 2019-02 added invert flag. If set to something other than zero it looks for things smaller rather than bigger the var_min
    
    #first checks if time is a variable, then uses the variable time to select the right timestep
    var_dim = var_file.variables[var_name].dimensions
    if 'time' in var_dim:
        idx_time = var_dim.index('time')
        if idx_time == 0:
            var = var_file.variables[var_name][time,:]
        if idx_time == 1:
            var = var_file.variables[var_name][:,time,:,:]
        if idx_time == 2:
            var = var_file.variables[var_name][:,:,time,:]
        if idx_time == 3:
            var = var_file.variables[var_name][:,:,:,time]
            
        #now we convert the var_dim tuple to a list remove time from var_dim so it can be used to rearrange z,x, and y
        var_dim_space = list(var_dim)
        var_dim_space.remove('time')
        
    else: 
        var = var_file.variables[var_name][:]
        
        #now we convert the var_dim tuple to a list
        var_dim_space = list(var_dim)
            
    #rearange to z,x,y
    if var_dim_space[2]=='zt':
        var_dim_space = [var_dim_space[2],var_dim_space[1],var_dim_space[0]]
        var = var.swapaxes(0,2)
    if var_dim_space[2]=='zm':
        var_dim_space = [var_dim_space[2],var_dim_space[1],var_dim_space[0]]
        var = var.swapaxes(0,2)
    if var_dim_space[2]=='xt':
        var_dim_space = [var_dim_space[0],var_dim_space[2],var_dim_space[1]]
        var = var.swapaxes(1,2)
    if var_dim_space[2]=='x':
        var_dim_space = [var_dim_space[0],var_dim_space[2],var_dim_space[1]]
        var = var.swapaxes(1,2)
    
    
    if flip_flag==0:
        var_binary = np.array(np.where(var>var_min,1,0),dtype=bool)
    else:
        var_binary = np.array(np.where(var<var_min,1,0),dtype=bool)
    return var_binary



###########################################################################################################################
#CLUSTERING
###########################################################################################################################
#Exists in a 2d and 3D 
#Both work very similar using unionfind zu detect all neighbouring cloud cells into a single cloud

def cluster_2D_v2(var_binary,boundary_periodic):
    #For a given binary 2d it returns a list of which cloudy cells belong to a shared cloud, and a array of the 2D indexes of those cells
    #initially used random indexes and where search, which turned out to be to costly for large fields.
    #So now I switched to using 1D vectors with a predifined 1D index which is id1D = n_y*i_1+i_2
    #important is that the array has to be in the right order when imported
    #This turned into a bit of a confusing mess, because we now have 2 1d indexes.
    #The first 1D index is of all cells and can be turned back into the 3D indexes using unravel
    #The second 1D index is just the list of all cloudy cells which is used for the union find
#boundary_periodic flag has to be passed along as well
    n_x,n_y = var_binary.shape
    #2D to 1D
    var_binary  = var_binary.ravel()
    
    #1D indexes of cloudy cells
    #Compute where cloudy cells are by hand (instead of with np.where) to be able to link neighbours later on.
    #Isn't fast, but only has to be done once.
    idx_1d_to_idx_cloudy_cells = [0]*len(var_binary)
    idx_1d_cloudy_cells = []
    counter = 0
    for i in range(len(var_binary)):
        if var_binary[i]==1:
            idx_1d_cloudy_cells.append(i)
            idx_1d_to_idx_cloudy_cells[i]=counter
            counter += 1
       
   
    nr_cloudy_cells = len(idx_1d_cloudy_cells)
    
   

    #Initiating unionfind class that will be used later on to find each individual cloud
    clouds_uf = UnionFind(range(nr_cloudy_cells))
   
    #For each cloudy cell the original 2d index is found before adding the 1d neighbours, I hardcoded it in 1D with n_x, and n_y. I could also have assigned the neighbours in 2D and then used ravel index somehow.
    #No neighbours are added across boundary domain

    for j in range(0,nr_cloudy_cells):
        idx_cell_1d = idx_1d_cloudy_cells[j]
        #print('idx_cell_1d',idx_cell_1d)
        idx_cell_2d = np.unravel_index(idx_cell_1d,(n_x,n_y))
        #print('idx_cell_2d',idx_cell_2d)
        cell_neighbours = []
       
       
        if idx_cell_2d[0]>0:
            cell_neighbours.append(idx_cell_1d-n_y)
        if idx_cell_2d[0]<n_x-1:
            cell_neighbours.append(idx_cell_1d+n_y)
        if idx_cell_2d[1]>0:
            cell_neighbours.append(idx_cell_1d-1)
        if idx_cell_2d[1]<n_y-1:
            cell_neighbours.append(idx_cell_1d+1)
       
      
        #If the boundaries are periodic x-y neighbours are added across the boundary
        if boundary_periodic:
            if idx_cell_2d[1]==0:
                cell_neighbours.append(idx_cell_1d-1+n_y)
            if idx_cell_2d[1]==n_y-1:
                cell_neighbours.append(idx_cell_1d+1-n_y)
            if idx_cell_2d[0]==0:
                cell_neighbours.append(idx_cell_1d+n_y*(n_x-1))
            if idx_cell_2d[0]==n_x-1:
                cell_neighbours.append(idx_cell_1d-n_y*(n_x-1))


        #Check if the neighbouring cells are also cloudy
        for i in range(len(cell_neighbours)):
            cell_neighbour = cell_neighbours[i]
           
            #If the neighbouring cell is cloudy the index is found on the vector of cloudy cells is found list and linked in unionfind
           
            if var_binary[cell_neighbour]:
                neighbour_cloud = idx_1d_to_idx_cloudy_cells[cell_neighbour] #Here we go from the total 1D index to only those with clouds
                clouds_uf.union(j,neighbour_cloud)
               
       
   
    #time1 = ttiimmee.time()
   
    cloud_cell_list_sets = clouds_uf.components() #Gives a list of sets of which cells belong to which cloud, but the cells are numbered from 0 to N-1
    
    cloud_cell_list = map(list,cloud_cell_list_sets) #Convert to list
    
    #time2 = ttiimmee.time()
    #print('so how long was unionfind.componentes in seconds?',(time2-time1)*1.0)
    nr_clouds = len(cloud_cell_list)
   
    #Last part is getting the 2d index for each of the cells listed cloud_cell_list
    idx_2d_cloudy_cells = np.unravel_index(np.array(idx_1d_cloudy_cells),(n_x,n_y))
    idx_2d_cloudy_cells = np.array(idx_2d_cloudy_cells)
   

    return cloud_cell_list,idx_2d_cloudy_cells








def cluster_3D_v2_filt(var_binary,boundary_periodic,z_ext_min=8,z_cbl=40):
    #slight modification of cluster_3D_v2 that filters out all clusters which have an extent smaller than 8 cells 
    #And which are above the convective boundary layer height given in a z level
    #Filters out a lot of small clusters, which don't reduce the clustering file by much since the 1D list of indexes of cells to be clustered 
    #is 3 times bigger than the list of cells in each cluster,  and I won't be trimming that.
    #But does reduce the number of clusters by factor of 10 or so, making it much quicker to calculate and save the following cluster properties
    #Should really only be used when dealing with plumes detected from the couvreux tracer

    #requires the input field to be in the following shape, z,x,y, although it will be switched to 1D later on
    n_z,n_x,n_y = var_binary.shape
    print(var_binary.shape)
   
    #initially used random indexes and where search, which turned out to be to costly for large fields.
    #So now I switched to using 1D vectors with a predifined 1D index which is id1D = i_1*n_x*n_y+n_y*i_2+i_3
    #important is that the array has to be in the right order when imported
    #This turned into a bit of a confusing mess, because we now have 2 1d indexes.
    #The first 1D index is of all cells and can be turned back into the 3D indexes using unravel
    #The second 1D index is just the list of all cloudy cells which is used for the union find
   
    #So in the end what is returned is cloud_cell_list, where all cells in each connected cloud are listed using the 1D
    #idx of the cloudy cells, and a 2D array idx_3d_cloudy_cells  which contains the 3d indexes for each cloudy cell mentioned in the 
    #cloud_cell_list
   
    #Memory trick, we will keep calling var_binary var_binary despite changing the dimension to avoid creating two gigantic arrays if the array is already really large
   

    #3D to 1D
    var_binary  = var_binary.ravel()
    #try to save memory

    #1D indexes of cloudy cells
    #Compute where cloudy cells are by hand (instead of with np.where) to be able to link neighbours later on.
    #Isn't fast, but only has to be done once.
    idx_1d_to_idx_cloudy_cells = [0]*len(var_binary)
    idx_1d_cloudy_cells = []
    counter = 0
    for i in range(len(var_binary)):
        if var_binary[i]==1:
            idx_1d_cloudy_cells.append(i)
            idx_1d_to_idx_cloudy_cells[i]=counter
            counter += 1
       
   
    nr_cloudy_cells = len(idx_1d_cloudy_cells)
   
   
    #Initiating unionfind class that will be used later on to find each individual cloud
    clouds_uf = UnionFind(range(nr_cloudy_cells))
   
    #For each cloudy cell the original 3d index is found before adding the 1d neighbours, I hardcoded it in 1D with n_z,n_x, and n_y. I could also have assigned the neighbours in 3D and then used ravel index somehow.
    #No neighbours are added across boundary domain

    for j in range(0,nr_cloudy_cells):
        idx_cell_1d = idx_1d_cloudy_cells[j]
        #print('idx_cell_1d',idx_cell_1d)
        idx_cell_3d = np.unravel_index(idx_cell_1d,(n_z,n_x,n_y))
        #print('idx_cell_3d',idx_cell_3d)
        cell_neighbours = []
       
       
        if idx_cell_3d[0]>0:
            cell_neighbours.append(idx_cell_1d-n_x*n_y)
        if idx_cell_3d[0]<n_z-1:
            cell_neighbours.append(idx_cell_1d+n_x*n_y)
        if idx_cell_3d[1]>0:
            cell_neighbours.append(idx_cell_1d-n_y)
        if idx_cell_3d[1]<n_x-1:
            cell_neighbours.append(idx_cell_1d+n_y)
        if idx_cell_3d[2]>0:
            cell_neighbours.append(idx_cell_1d-1)
        if idx_cell_3d[2]<n_y-1:
            cell_neighbours.append(idx_cell_1d+1)
       
      
        #If the boundaries are periodic x-y neighbours are added across the boundary
        if boundary_periodic:
            if idx_cell_3d[2]==0:
                cell_neighbours.append(idx_cell_1d-1+n_y)
            if idx_cell_3d[2]==n_y-1:
                cell_neighbours.append(idx_cell_1d+1-n_y)
            if idx_cell_3d[1]==0:
                cell_neighbours.append(idx_cell_1d+n_y*(n_x-1))
            if idx_cell_3d[1]==n_x-1:
                cell_neighbours.append(idx_cell_1d-n_y*(n_x-1))

           
        #Check if the neighbouring cells are also cloudy
        for i in range(len(cell_neighbours)):
            cell_neighbour = cell_neighbours[i]
           
            #If the neighbouring cell is cloudy the index is found on the vector of cloudy cells is found list and linked in unionfind
           
            if var_binary[cell_neighbour]:
                neighbour_cloud = idx_1d_to_idx_cloudy_cells[cell_neighbour] #Here we go from the total 1D index to only those with clouds
                clouds_uf.union(j,neighbour_cloud)
               
       
   
    #time1 = ttiimmee.time()
   
    cloud_cell_list_sets = clouds_uf.components() #Gives a list of sets of which cells belong to which cloud, but the cells are numbered from 0 to N-1
    

    cloud_cell_list = list(map(list,cloud_cell_list_sets)) #Convert to list
    
    #Last part is getting the 3d index for each of the cells listed cloud_cell_list
    idx_3d_cloudy_cells = np.unravel_index(np.array(idx_1d_cloudy_cells),(n_z,n_x,n_y))
    idx_3d_cloudy_cells = np.array(idx_3d_cloudy_cells)
    
    
    #First filter step, getting rid of everything which doesn't have at least z_ext_min just to make the following filter steps quicker easier
    cloud_cell_list_minsize =  [s for s in cloud_cell_list if len(s) > z_ext_min]
    #Second filter step, getting rid of everything that doesn't have the minimum vertical extent of z_ext_min and a minimum below the cbl height 
    #We will start out with a loop because it is easier to code, hopefully python realizes this is fully parallizable
    n_clusters = len(cloud_cell_list_minsize)
    cloud_cell_list_filt = []
    n_filtered_cluster_cells = 0
    for n in range(n_clusters):
        idx_z = idx_3d_cloudy_cells[0][cloud_cell_list_minsize[n]]
        min(idx_z)
        if min(idx_z)<z_cbl and max(idx_z)-min(idx_z)>=z_ext_min:
          cloud_cell_list_filt.append(cloud_cell_list_minsize[n])
          #I want to count the total amount of cells which are in the clusters that pass the filtering, to find out if it worth writting a new idx_3d_cloudy_cells
          #which only contains the cells in the filtered clusters
          n_filtered_cluster_cells += len(cloud_cell_list_minsize[n])
    

    #time2 = ttiimmee.time()
    #print('so how long was unionfind.componentes in seconds?',(time2-time1)*1.0)
    #nr_clouds = len(cloud_cell_list)
    print('total number of cells contained in the filtered clusters: ',n_filtered_cluster_cells) 
   

    return cloud_cell_list_filt,idx_3d_cloudy_cells



def cluster_3D_v2(var_binary,boundary_periodic):
    #requires the input field to be in the following shape, z,x,y, although it will be switched to 1D later on
    n_z,n_x,n_y = var_binary.shape
    print(var_binary.shape)
    #initially used random indexes and where search, which turned out to be to costly for large fields.
    #So now I switched to using 1D vectors with a predifined 1D index which is id1D = i_1*n_x*n_y+n_y*i_2+i_3
    #important is that the array has to be in the right order when imported
    #This turned into a bit of a confusing mess, because we now have 2 1d indexes.
    #The first 1D index is of all cells and can be turned back into the 3D indexes using unravel
    #The second 1D index is just the list of all cloudy cells which is used for the union find
   
    #So in the end what is returned is cloud_cell_list, where all cells in each connected cloud are listed using the 1D
    #idx of the cloudy cells, and a 2D array idx_3d_cloudy_cells  which contains the 3d indexes for each cloudy cell mentioned in the 
    #cloud_cell_list
   
    #Memory trick, we will keep calling var_binary var_binary despite changing the dimension to avoid creating two gigantic arrays if the array is already really large
   
    #3D to 1D
    var_binary  = var_binary.ravel()
    #try to save memory

    #1D indexes of cloudy cells
    #Compute where cloudy cells are by hand (instead of with np.where) to be able to link neighbours later on.
    #Isn't fast, but only has to be done once.
    idx_1d_to_idx_cloudy_cells = [0]*len(var_binary)
    idx_1d_cloudy_cells = []
    counter = 0
    for i in range(len(var_binary)):
        if var_binary[i]==1:
            idx_1d_cloudy_cells.append(i)
            idx_1d_to_idx_cloudy_cells[i]=counter
            counter += 1
       
   
    nr_cloudy_cells = len(idx_1d_cloudy_cells)
   
   
    #Initiating unionfind class that will be used later on to find each individual cloud
    clouds_uf = UnionFind(range(nr_cloudy_cells))
   
    #For each cloudy cell the original 3d index is found before adding the 1d neighbours, I hardcoded it in 1D with n_z,n_x, and n_y. I could also have assigned the neighbours in 3D and then used ravel index somehow.
    #No neighbours are added across boundary domain

    for j in range(0,nr_cloudy_cells):
        idx_cell_1d = idx_1d_cloudy_cells[j]
        #print('idx_cell_1d',idx_cell_1d)
        idx_cell_3d = np.unravel_index(idx_cell_1d,(n_z,n_x,n_y))
        #print('idx_cell_3d',idx_cell_3d)
        cell_neighbours = []
       
       
        if idx_cell_3d[0]>0:
            cell_neighbours.append(idx_cell_1d-n_x*n_y)
        if idx_cell_3d[0]<n_z-1:
            cell_neighbours.append(idx_cell_1d+n_x*n_y)
        if idx_cell_3d[1]>0:
            cell_neighbours.append(idx_cell_1d-n_y)
        if idx_cell_3d[1]<n_x-1:
            cell_neighbours.append(idx_cell_1d+n_y)
        if idx_cell_3d[2]>0:
            cell_neighbours.append(idx_cell_1d-1)
        if idx_cell_3d[2]<n_y-1:
            cell_neighbours.append(idx_cell_1d+1)
       
      
        #If the boundaries are periodic x-y neighbours are added across the boundary
        if boundary_periodic:
            if idx_cell_3d[2]==0:
                cell_neighbours.append(idx_cell_1d-1+n_y)
            if idx_cell_3d[2]==n_y-1:
                cell_neighbours.append(idx_cell_1d+1-n_y)
            if idx_cell_3d[1]==0:
                cell_neighbours.append(idx_cell_1d+n_y*(n_x-1))
            if idx_cell_3d[1]==n_x-1:
                cell_neighbours.append(idx_cell_1d-n_y*(n_x-1))

           
        #Check if the neighbouring cells are also cloudy
        for i in range(len(cell_neighbours)):
            cell_neighbour = cell_neighbours[i]
           
            #If the neighbouring cell is cloudy the index is found on the vector of cloudy cells is found list and linked in unionfind
           
            if var_binary[cell_neighbour]:
                neighbour_cloud = idx_1d_to_idx_cloudy_cells[cell_neighbour] #Here we go from the total 1D index to only those with clouds
                clouds_uf.union(j,neighbour_cloud)
               
       
   
    #time1 = ttiimmee.time()
   
    cloud_cell_list_sets = clouds_uf.components() #Gives a list of sets of which cells belong to which cloud, but the cells are numbered from 0 to N-1
    

    cloud_cell_list = list(map(list,cloud_cell_list_sets)) #Convert to list
    
    #time2 = ttiimmee.time()
    #print('so how long was unionfind.componentes in seconds?',(time2-time1)*1.0)
    nr_clouds = len(cloud_cell_list)
   
    #Last part is getting the 3d index for each of the cells listed cloud_cell_list
    idx_3d_cloudy_cells = np.unravel_index(np.array(idx_1d_cloudy_cells),(n_z,n_x,n_y))
    idx_3d_cloudy_cells = np.array(idx_3d_cloudy_cells)
   

    return cloud_cell_list,idx_3d_cloudy_cells





###########################################################################################################################
#Cloud properties
###########################################################################################################################

###Function for projected horizontal area of cloud
#takes the x and y indexes and turns them into 1D indexes, and uses np.unique to find the unique horizontal grid points
#multiplies with the cell Area to get the total area
def func_proj_A(idx_x,idx_y,dA):
    idx_1D = idx_x*(max(idx_y)+1)+idx_y
    n_hor = len(np.unique(idx_1D))
    A_proj = n_hor*dA
    return A_proj


###Function for vertical flux profile of passed variable
#goes through all unique z indexes
#returns a numpy array of n_z length with nan values where no levels are
#Also returns number of cells per level 
def func_vert_flux(idx_z,idx_x,idx_y,w,var):
    z_unique = np.unique(idx_z)
    #print(z_unique)
    vert_prof    = np.zeros(var.shape[0])
    vert_prof[:] = 'nan'
    vert_n    = np.zeros(var.shape[0])
    vert_n[:] = 'nan'
    for z in z_unique:
        ind=np.where(idx_z==z)
        #print(ind)
        #print(idx_z[ind],idx_x[ind],idx_y[ind])
        vert_prof[z]=np.sum(w[idx_z[ind],idx_x[ind],idx_y[ind]]*var[idx_z[ind],idx_x[ind],idx_y[ind]])
        vert_n[z]=len(ind[0])*1.0
    return vert_prof, vert_n

###Function for mean vertical profile of passed variable
#goes through all unique z indexes
#returns a numpy array of n_z length with nan values where no levels are
#Also returns number of cells per level 
def func_vert_mean(idx_z,idx_x,idx_y,var):
    z_unique = np.unique(idx_z)
    #print(z_unique)
    vert_prof    = np.zeros(var.shape[0])
    vert_prof[:] = 'nan'
    vert_n    = np.zeros(var.shape[0])
    vert_n[:] = 'nan'
    for z in z_unique:
        ind=np.where(idx_z==z)
        #print(ind)
        #print(idx_z[ind],idx_x[ind],idx_y[ind])
        vert_prof[z]=np.mean(var[idx_z[ind],idx_x[ind],idx_y[ind]])
        vert_n[z]=len(ind[0])*1.0
    return vert_prof, vert_n

#Optimized version of func_vert_mean that should (fingers crossed) be a lot faster for bigger plumes
#It seems that for some reason idx_z is already sorted, so it could be further sped up by replacing z_sort with np.arange
def func_vert_mean_sorted(idx_z,idx_x,idx_y,var):
    z_unique = np.unique(idx_z)
    #print(z_unique)
    vert_prof    = np.zeros(var.shape[0])
    vert_prof[:] = 'nan'
    vert_n    = np.zeros(var.shape[0])
    vert_n[:] = 'nan'
    z_sort = np.argsort(idx_z, kind='mergesort')
    idx_z_sort= np.split(z_sort, np.cumsum(np.bincount(idx_z)[:-1]))
    for z in z_unique:
        ind = idx_z_sort[z]
        #ind=np.where(idx_z==z)
        #print(ind)
        #print(idx_z[ind],idx_x[ind],idx_y[ind])
        vert_prof[z]=np.mean(var[idx_z[ind],idx_x[ind],idx_y[ind]])
        vert_n[z]=len(ind)*1.0
    return vert_prof, vert_n


def func_vert_mean_sorted_argv(idx_z,idx_x,idx_y,*argv):
    """
        A quicker version of func_vert_mean
        Speed up thanks to:
        I: getting rid of np.where and using the sort indexing
        II: ability to pass multiple to be processed so that the indexes don't need to be found again for each var

        parameters: 
        z, x, y, indexes of the field to be averaged over

    """
    z_unique = np.unique(idx_z)
    var = argv[0]
    vert_prof    = np.zeros([var.shape[0],len(argv)])

    vert_prof[:,:] = 'nan'
    vert_n    = np.zeros(var.shape[0])
    vert_n[:] = 'nan'
    z_sort = np.argsort(idx_z, kind='mergesort')
    idx_z_sort= np.split(z_sort, np.cumsum(np.bincount(idx_z)[:-1]))
    for z in z_unique:
        ind = idx_z_sort[z]
        vert_n[z]=len(ind)*1.0
        for a in range(len(argv)):
            var = argv[a]
            vert_prof[z,a]=np.mean(var[idx_z[ind],idx_x[ind],idx_y[ind]])
    return vert_prof, vert_n

###Similar to func_vert_mean, but calculates the percentile instead
def func_vert_percentile(idx_z,idx_x,idx_y,var,percentile):
    z_unique = np.unique(idx_z)
    #print(z_unique)
    vert_prof    = np.zeros(var.shape[0])
    vert_prof[:] = 'nan'
    vert_n    = np.zeros(var.shape[0])
    vert_n[:] = 'nan'
    for z in z_unique:
        ind=np.where(idx_z==z)
        #print(ind)
        #print(idx_z[ind],idx_x[ind],idx_y[ind])
        vert_prof[z]=np.percentile(var[idx_z[ind],idx_x[ind],idx_y[ind]],percentile)
        vert_n[z]=len(ind[0])*1.0
    return vert_prof, vert_n

###########################################################################################################################
#Binning
###########################################################################################################################


#Creates the bins for a given width and max value
#Bin width is linear and starts with 0
#Then detects the indexes of which clouds are in which bin
#Also directly calculates the cloud size distribution (CSD)
#Before deviding by the width turn it into a CSDD (cloud size density distribution)
def linear_binner(width,var):
     max_val   = max(var)
     bin_n     = int(np.ceil(max_val/width))
     bins      = np.linspace(0,bin_n*width,bin_n+1)
     ind       = np.digitize(var,bins)
     CSD       = np.zeros(bin_n)
     for b in range(bin_n):
#C   SD[b] = len(cloud_w_all[ind==b])
         CSD[b] = float(np.count_nonzero(ind==b+1))/width

     return bin_n, bins, ind, CSD


#Based on linear binner, but immediately returns an array of a variable binned by another one.
#Returns a nan where no data available in bin. 
def var_linear_binner(var,bin_var,bin_width=100):
    
    bin_n, bins, ind, CSD =linear_binner(100,bin_var)
    binned_var = np.zeros(bin_n)
    for b in range(bin_n):
        if max(ind==b+1):
            binned_var[b] = np.nanmean(var[ind==b+1])
        else:
            binned_var[b] = 'nan'

    return binned_var, bin_n, bins, CSD 
                                                                                    
#Creates the bins for a given width and max value
#Bin width is logarithmic, starts with bin_min, and increases from there on out 
#Then detects the indexes of which clouds are in which bin
#Also directly calculates the cloud size distribution (CSD)
#Before deviding by the width turn it into a CSD (cloud size density distribution)
def log_binner(var,bin_min=25,step_ratio=2,N_min=0):
    max_val   = max(var)
    min_val   = min(var)
    #bin_min = max(min_val,bin_min)


    log_bin_dist = np.log10(step_ratio)
    max_log = np.log10(max_val/bin_min)

    bins = bin_min*10**(np.arange(0,max_log+log_bin_dist,log_bin_dist))
    bin_n = len(bins)-1
    ind       = np.digitize(var,bins)
    CSD       = np.zeros(bin_n)
    for b in range(bin_n):
        if len(ind[ind==b+1])>N_min:
            CSD[b] = float(np.count_nonzero(ind==b+1))/(bins[b+1]-bins[b])
        else:
            CSD[b] = 'nan'
    return bin_n, bins, ind, CSD 


#Based on linear binner, but immediately returns an array of a variable binned by another one.
#Returns a nan where less than N_min points in bin. 
def var_log_binner(var,bin_var,bin_min=0,step_ratio=2,N_min=10):

    bin_n, bins, ind, CSD =log_binner(bin_var,bin_min=bin_min,step_ratio=step_ratio)
    binned_var = np.zeros(bin_n)
    for b in range(bin_n):
        if len(ind[ind==b+1])>N_min:
            binned_var[b] = np.nanmean(var[ind==b+1])
        else:
            binned_var[b] = 'nan'
            CSD[b] = 'nan'

    return binned_var, bin_n, bins, CSD







def func_A_base_binner(clus_prop,max_bin=5,prescribed_width=0,percentile=70.,t_window=0,
                                  plume_min=1,n_z=256,dz=25,w_sorting_flag=0,z_sorting_flag=1,bin_var='sq Area',cluster_per=50):
    """     
    Ok, this is a reworking of the original func_A_base_binner. The original one was not designed to be what it turned into. 
  
    So the way this should work is that instead of first determining the plume height and then moving plumes up, it should iteratively move clusters from plume to plume until the bin is stabil.

    Also, I chucked w_sorting for the mean time. I couldn't really get it to do what I wanted.  

    The basic idea of the function is that it moves clusters which are abnormally high in comparison to the other plumes in a bin to a higher bin. 
    Abnormally high is decided if the height percentile (defined by cluster_per) is higher than the plume height calculated from the percentile Uses height to determine if things should be bumped up. 
   
    An important limitation is that the height of the plume of a specific size bin can not be lower than that of a smaller bin size. 

    Function was originally derived from the full reconstruction plot function in a hurry in time for the paracon workshop, which explains a lot of its naming. In it's current format it is stabil, but takes a while. 

    IMPORTANT:
    -The method used to determine the plume height is a function of the total area of all plumes in a bin. For the smallest bin size this can be a problem, because the clustering detects very many small plumes close to the surface with a big area. This can be adressed by first filtering out all plumes which do not reach until a certain height, which will in turn undersample close to the surface.

    -Despite the amount of time I have spent toying around with this function, it really is only a vary crude tool. Do not read to much into it. 
    
    Parameters:
        max_bin:            Number of bins
        percentile:         percentile used to determine height from prof_height variable
        precribed_width:    defines bin width
        t_window:           time before and after which are included in the average in seconds, makes for a smoother height value
        plume_min:          numbers of plumes needed for the height to be able to be determined. If not fulfilled height is super heigh, all plumes remain in that bin. This isn't great
        bin_var:            variable used to sort into bins. Default is 'sq Area' 
        cluster_per:        Determines which plume area percentile a cluster can be above the plume "height" before it is moved to the next plume. 
    
    Returns:
        bin_n, bins,bin_ind: same as the linear binner, but with no size distribution as it doesn't really make sense..
    """
    
    
    plume_init_bin = clus_prop[bin_var]
    prof_height = 'Area profile'

    #determining bin width using the 99.9th percentile
    max_var = np.percentile(plume_init_bin,99.9)
    bin_width = round_down(max_var/max_bin)
    print('bin_width:',bin_width)
    if prescribed_width>0:
        bin_width = prescribed_width
        print('bin_width override:',bin_width)


    bin_n, bins, bin_ind, csd = linear_binner(bin_width,plume_init_bin)


    time_vec = np.unique(clus_prop['time'])

    #limiting t_steps to avoid issues
    t_steps = len(time_vec)

    prof_z=np.linspace(0,n_z*dz,n_z)+dz/2.

    #Calculating height percentile of clusters used later on to determine if clusters are too high. 
    tmp_array = np.zeros(len(clus_prop))
    print('calculating per height')
    for i in range(len(clus_prop)):
        #I use the cheap alternative to Area_percentile_x to speed things up. 
        tmp_array[i]=Area_percentile_cheap(prof_z,clus_prop['Area profile'].iloc[i],cluster_per)

    #Now a quick fix for all clusters which have only a single horizontal layer
    idx_singlelayer =   np.where(clus_prop['height']==dz)[0]
    tmp_array[idx_singlelayer] = clus_prop['base'].iloc[idx_singlelayer]+dz*cluster_per/100.
    clus_prop['per height'] = tmp_array

    del(tmp_array)
    print('done calculating per height')
    
    
    #Plan is to iterate through the bins, moving things up until nothing happens anymore. 
    clus_prop['weighted w'] = clus_prop['Area profile']*clus_prop['w profile']                  
    
    #Creating height matrix
    plume_height = np.zeros([max_bin,t_steps])
    
    
    #Ok, now that the Area binning is ready, we begin by iterating over each time step

    t_iterable = np.arange(t_steps).astype(int)
    for t in t_iterable:
        #expanding to include a time window before and after. First calculating t difference to current timestep
        delta_t = abs((clus_prop['time']-time_vec[t])/np.timedelta64(1, 's'))

        for b in range(max_bin):
            bump_iteration = 1 #Is set to zero when nothing more to bump up

            while bump_iteration>0:
                index_select =np.where(np.logical_and(bin_ind==b+1,delta_t<=t_window))[0] 
                clus_tmp = clus_prop.iloc[index_select]

                if len(clus_tmp)>plume_min:
                    
                    #Getting height according to percentile at 1 m percision using linear interpolation 
                    prof_sum_height = func_prof_sum(clus_tmp,prof_height,n_z)
                    
                    z_percentile    = Area_percentile_x(prof_z,prof_sum_height,percentile)
                    z_scaled = z_percentile*100./percentile
                    
                    if b>0 and z_scaled<plume_height[b-1,t]:
                        plume_height[b,t] = plume_height[b-1,t]
                        #print('WARNING, plume hight being boosted to thinner plume')
                    else:
                        plume_height[b,t] = z_scaled
                    
                    
                    #Now we have the plume height, time to bump shit up according to the cluster vs plume height.
                    bump_up_idx = np.where((bin_ind==b+1) & (clus_prop['time']==time_vec[t]) & ((clus_prop['per height'])>plume_height[b,t]))[0]
                    
                    if bump_up_idx.size>0:
                        bin_ind[bump_up_idx] = bin_ind[bump_up_idx] + 1
                        #print(bump_up_idx)
                        #print('bump iteration, t, b, n bump, bump it',t,b,bump_up_idx.size,bump_iteration)
                        #print('plume height and number',z_scaled,len(clus_tmp))
                        #print('clus_tmp size',len(clus_tmp))
                        bump_iteration += 1
                    else:
                        #print('FINAL HEIGHT ',plume_height[b,t],' bin: ',t,b, ' iterations ',bump_iteration-1)
                        bump_iteration = 0




                else:
                    plume_height[b,t] = n_z*dz
                    bump_iteration =0



    return bin_n, bins,bin_ind

################################################################################################
#Random small stuff
################################################################################################


#Little subplot labeling script found online from https://gist.github.com/tacaswell/9643166
#Put here for no good reason
#Warning, if you define your colorbar using an additional axes it will give that a label as well, which is pretty funny but also very annoying. 

import string
from itertools import cycle
from six.moves import zip
def label_axes_abcd(fig, labels=None, loc=None, **kwargs):
    """
    Walks through axes and labels each.
    kwargs are collected and passed to `annotate`
    Parameters
    ----------
    fig : Figure
         Figure object to work on
    labels : iterable or None
        iterable of strings to use to label the axes.
        If None, lower case letters are used.
    
    loc : len=2 tuple of floats
        Where to put the label in axes-fraction units
    """
    if labels is None:
        labels = string.ascii_lowercase
    
    # re-use labels rather than stop labeling
    labels = cycle(labels)
    if loc is None:
        loc = (.9, .9)
    for ax, lab in zip(fig.axes, labels):
        ax.annotate(lab, xy=loc,
                    xycoords='axes fraction',
                    **kwargs)
###########################################################################################################################
#Graveyard
###########################################################################################################################

#def cluster_3D(var_binary,boundary_periodic):
#    n_z,n_x,n_y = var_binary.shape
#    print(n_z,n_x,n_y)
#    idx_cloudy_cells = np.array(np.where(var_binary))
#    idx_cloudy_cells = idx_cloudy_cells.transpose()
#    nr_cloudy_cells = len(idx_cloudy_cells[:,0])
#    
#    clouds_uf = unionfind(nr_cloudy_cells)
#   
#    for j in range(0,nr_cloudy_cells):
#        idx_cell = idx_cloudy_cells[j,:]
#       
#        #adding a little timer
#        if j/10000.==np.floor(j/1000):
#            print('j ',j,nr_cloudy_cells) 
#        
#        
#        #list of all neighbouting cells
#        cell_neighbours = np.vstack((idx_cell + [1,0,0],idx_cell - [1,0,0],idx_cell + [0,1,0],idx_cell - [0,1,0]))
#        cell_neighbours = np.vstack((cell_neighbours,idx_cell + [0,0,1],idx_cell - [0,0,1]))
#
#             
#        
#        #If the boundaries are periodic -1 are replaced with n-1, and n with 0
#        if boundary_periodic:
#            #print(idx_cell)
#            #print(cell_neighbours)
#            cell_neighbours[cell_neighbours[:,2]==-1    ,2] =n_x-1
#            cell_neighbours[cell_neighbours[:,1]==-1    ,1] =n_y-1
#            cell_neighbours[cell_neighbours[:,2]==n_x ,2] =0
#            cell_neighbours[cell_neighbours[:,1]==n_y ,1] =0
#            #print(cell_neighbours)
#        
#        #Now to get rid of annoying -1 that lead to weird connections across edges
#        if np.min(cell_neighbours)==-1:
#            cell_neighbours[cell_neighbours==-1]=1e10
#            
#        #Check the neighbouring cells if they are in the list of all cloudy cells. 
#        for i in range(len(cell_neighbours[:,0])):
#            cell_neighbour = cell_neighbours[i,:]
#            neighbour_cloud = np.where((idx_cloudy_cells == cell_neighbour).all(axis=1))[0]
#            #If the neighbouring cell is in the cloud list they are linked in unionfind
#            if neighbour_cloud.size:
#                clouds_uf.unite(j,neighbour_cloud)
#                #print('connection ',idx_cell, ' to ',cell_neighbour)
#    #list is generated of all connected cloudy cells for each cloud from unionfind
#    cloud_idx_list = clouds_uf.groups()
#    nr_clouds = len(cloud_idx_list)
#    
#    #print('number of clouds ',len(set(clouds_uf.parent)))
#    
#
#    return cloud_idx_list,idx_cloudy_cells 
#
#def cluster_3D_new(var_binary,boundary_periodic):
#    #requires the field to be in the following shape, z,x,y
#    n_z,n_x,n_y = var_binary.shape
#    print(var_binary.shape)
#    #initially used random indexes and where search, which turned out to be to costly for large fields.
#    #So now I switched to using 1D vectors with a predifined 1D index which is id1D = i_1*n_x*n_y+n_y*i_2+i_3 
#    #important is that the array has to be in the right order when imported
#    #This turned into a bit of a confusing mess, because we now have 2 1d indexes.
#    #The first 1D index is of all cells and can be turned back into the 3D indexes using unravel
#    #The second 1D index is just the list of all cloudy cells which is used for the union find
#    
#    #So in the end what is returned is cloud_cell_list, where all cells in each connected cloud are listed using the 1D
#    #idx of the cloudy cells, and a 2D array idx_3d_cloudy_cells  which contains the 3d indexes for each cloudy cell mentioned in the  
#    #cloud_cell_list
#    
#    
#    #3D to 1D
#    var_binary_1d  = var_binary.ravel()
#    
#    #1D indexes of cloudy cells
#    #Compute where cloudy cells are by hand to be able to link neighbours later on. 
#    #Isn't fast, but only has to be done once. 
#    idx_1d_to_idx_cloudy_cells = [0]*len(var_binary_1d)
#    idx_1d_cloudy_cells = []
#    counter = 0
#    for i in range(len(var_binary_1d)):
#        if var_binary_1d[i]==1: 
#            idx_1d_cloudy_cells.append(i)
#            idx_1d_to_idx_cloudy_cells[i]=counter
#            counter += 1
#        
#    
#    nr_cloudy_cells = len(idx_1d_cloudy_cells)
#    
#    
#    #Initiating unionfind class that will be used later on to find each individual cloud
#    clouds_uf = unionfind(nr_cloudy_cells)
#    
#    #For each cloudy cell the original 3d index is found before adding the 1d neighbours
#    #No neighbours are added across boundary domain
#    for j in range(0,nr_cloudy_cells):
#        
#        #adding a little timer
#        if j/10000.==np.floor(j/10000):
#            print('j ',j,nr_cloudy_cells) 
#        
#        
#        idx_cell_1d = idx_1d_cloudy_cells[j]
#        #print('idx_cell_1d',idx_cell_1d)
#        idx_cell_3d = np.unravel_index(idx_cell_1d,(n_z,n_x,n_y))
#        #print('idx_cell_3d',idx_cell_3d)
#        cell_neighbours = []
#        
#        
#        if idx_cell_3d[0]>0:
#            cell_neighbours.append(idx_cell_1d-n_x*n_y) 
#        if idx_cell_3d[0]<n_z-1:
#            cell_neighbours.append(idx_cell_1d+n_x*n_y) 
#        if idx_cell_3d[1]>0:
#            cell_neighbours.append(idx_cell_1d-n_y) 
#        if idx_cell_3d[1]<n_x-1:
#            cell_neighbours.append(idx_cell_1d+n_y) 
#        if idx_cell_3d[2]>0:
#            cell_neighbours.append(idx_cell_1d-1) 
#        if idx_cell_3d[2]<n_y-1:
#            cell_neighbours.append(idx_cell_1d+1) 
#        
#       
#        #If the boundaries are periodic x-y neighbours are added accross the boundary
#        if boundary_periodic:
#            if idx_cell_3d[2]==0:
#                cell_neighbours.append(idx_cell_1d-1+n_y)
#            if idx_cell_3d[2]==n_y-1:
#                cell_neighbours.append(idx_cell_1d+1-n_y)
#            if idx_cell_3d[1]==0:
#                cell_neighbours.append(idx_cell_1d+n_y*(n_x-1))
#            if idx_cell_3d[1]==n_x-1:
#                cell_neighbours.append(idx_cell_1d-n_y*(n_x-1))
#
#        
#            
#        #Check if the neighbouring cells are also cloudy 
#        #for i in range(len(cell_neighbours[:,0])):
#        for i in range(len(cell_neighbours)):
#            cell_neighbour = cell_neighbours[i]
#            
#            #If the neighbouring cell is cloudy the index is found on the vector of cloudy cells is found list and linked in unionfind
#            
#            #if var_binary[tuple(cell_neighbour)]:
#            if var_binary_1d[cell_neighbour]:
#                neighbour_cloud = idx_1d_to_idx_cloudy_cells[cell_neighbour]
#                clouds_uf.unite(j,neighbour_cloud)
#                #print('connection ',idx_cell, ' to ',cell_neighbour)
#    
#    print('now we use union find')
#    time1 = time.time()
#    cloud_cell_list = clouds_uf.groups() #Gives the list of which cells belong to which cloud, but the cells are numbered from 0 to N
#    time2 = time.time()
#    print('so how long was unionfind?',(time2-time1)*1000.0)
#    nr_clouds = len(cloud_cell_list)
#    
#    
#    idx_3d_cloudy_cells = np.unravel_index(np.array(idx_1d_cloudy_cells),(n_z,n_x,n_y))
#    idx_3d_cloudy_cells = np.array(idx_3d_cloudy_cells)
#    return cloud_cell_list,idx_3d_cloudy_cells 
##Uses new unionfind I found on github



#def cluster_2D(var_binary,boundary_periodic):
#    n_x,n_y = var_binary.shape
#    idx_cloudy_cells = np.array(np.where(var_binary))
#    idx_cloudy_cells = idx_cloudy_cells.transpose()
#    nr_cloudy_cells = len(idx_cloudy_cells[:,0])
#    clouds_uf = unionfind(nr_cloudy_cells)
#    for j in range(0,nr_cloudy_cells):
#        idx_cell = idx_cloudy_cells[j,:]
#
#        #array of the neighbouring cells
#        cell_neighbours = np.vstack((idx_cell + [1,0],idx_cell - [1,0],idx_cell + [0,1],idx_cell - [0,1]))
#
#        #If the boundaries are periodic -1 are replaced with n-1, and n with 0
#        if boundary_periodic:
#            cell_neighbours[cell_neighbours[:,0]==-1    ,0] =n_x-1
#            cell_neighbours[cell_neighbours[:,1]==-1    ,1] =n_y-1
#            cell_neighbours[cell_neighbours[:,0]==n_x ,0] =0
#            cell_neighbours[cell_neighbours[:,1]==n_y ,1] =0
#        #print(cell_neighbours)
#        
#        #Now to get rid of annoying -1 that lead to weird connections across edges
#        if np.min(cell_neighbours)==-1:
#            cell_neighbours[cell_neighbours==-1]=1e10
#            
#
#        #Check the neighbouring cells if they are in the list of all cloudy cells. 
#        for i in range(len(cell_neighbours[:,0])):
#            cell_neighbour = cell_neighbours[i,:]
#            neighbour_cloud = np.where((idx_cloudy_cells == cell_neighbour).all(axis=1))[0]
#            #If the neighbouring cell is in the cloud list they are linked in unionfind
#            if neighbour_cloud.size:
#                clouds_uf.unite(j,neighbour_cloud)
#                #print('connection ',idx_cell, ' to ',cell_neighbour)
#    
#    
#    #list is generated of all connected cloudy cells for each cloud from unionfind
#    cloud_idx_list = clouds_uf.groups()
#    nr_clouds = len(cloud_idx_list)
#    
#    #print('number of clouds ',len(set(clouds_uf.parent)))
#    
#
#    return cloud_idx_list,idx_cloudy_cells 
#
#def func_A_base_binner(clus_prop,max_bin=5,prescribed_width=0,percentile=80.,t_window=0,
#                                  plume_min=1,n_z=256,dz=25,w_sorting_flag=0,z_sorting_flag=1,bin_var='sq Area'):
#    """     
#    A binner that first uses a linear area binner as a first guess, and then bumps high plumes up a bin
#   
#    z_sorting_flag ==1:
#    Uses height to determine if things should be bumped up. 
#    The precise definition of what constitutes too high is very arbitrary. Current version uses when
#    the middle plume height is higher than the calculated height of the plume bin.
#    
#    If w_sorting_flag = 1:
#    After applying the height limiter it sorts all plumes that have a "too high" vertical velocity into a higher bin. 
#    Too high is of course also super arbitrary
#    Ideas: if plume vertical velocity above the binned mean plume height is higher than the mean plume velocity below. 
#    Very much under construction!
#   
#    
#    Function was derived from the full reconstruction plot function in a hurry in time for the paracon workshop, which explains a lot of its naming. 
#    
#    Parameters:
#        max_bin:            Number of bins
#        w_sorting_flag:     0: nothing, 1: move plumes which are too fast at plume height
#        z_sorting_flag:     0: nothing, 1: move plumes which are too high  
#        percentile:         percentile used to determine height from prof_height variable
#        precribed_width:    defines bin width
#        t_window:           time before and after which are included in the average in seconds, makes for a smoother height value
#        plume_min:          numbers of plumes needed for the height to be able to be determined. If not fulfilled height is super heigh, all plumes remain in that bin. This isn't great
#        bin_var:            variable used to sort into bins. Default is 'sq Area' 
#    
#    Returns:
#        bin_n, bins,bin_ind,csd: same as the linear binner. 
#    """
#    
#    
#    plume_init_bin = clus_prop[bin_var]
#    prof_height = 'Area profile'
#
#    #determining bin width using the 99.9th percentile
#    max_var = np.percentile(plume_init_bin,99.9)
#    bin_width = round_down(max_var/max_bin)
#    print('bin_width:',bin_width)
#    if prescribed_width>0:
#        bin_width = prescribed_width
#        print('bin_width override:',bin_width)
#
#
#    bin_n, bins, bin_ind, csd = linear_binner(bin_width,plume_init_bin)
#
#
#    time_vec = np.unique(clus_prop['time'])
#
#    #limiting t_steps to avoid issues
#    t_steps = len(time_vec)
#
#    prof_z=np.linspace(0,n_z*dz,n_z)+dz/2.
#
#    
#    
#    #We are turning this into a loop to see if it converges.
#
#    clus_prop['weighted w'] = clus_prop['Area profile']*clus_prop['w profile']                  
#    for i in range(5):
#    
#      #Creating height matrix used for height filtering
#      plume_height = np.zeros([max_bin,t_steps])
#      
#      #Creating w matrix for w filtering
#      plume_w_below_height = np.zeros([max_bin,t_steps])
#
#      
#      
#      #Ok, now that the Area binning is ready, we calculate the height for each bin at each time step
#
#      for b in range(max_bin):
#
#          clus_tmp = clus_prop.iloc[bin_ind==b+1]
#          t_iterable = np.arange(t_steps).astype(int)
#          for t in t_iterable:
#              #expanding to include a time window before and after. First calculating t difference to current timestep
#              delta_t = abs((clus_tmp['time']-time_vec[t])/np.timedelta64(1, 's'))
#
#              idx_time = np.where(delta_t<=t_window)[0] 
#              clus_tmp_t = clus_tmp.iloc[idx_time]
#
#              if len(idx_time)>plume_min:
#                  
#                  #Getting height according to percentile at 1 m percision using linear interpolation 
#                  prof_sum_height = func_prof_sum(clus_tmp_t,prof_height,n_z)
#                  
#                  z_percentile    = Area_percentile_x(prof_z,prof_sum_height,percentile)
#                  z_scaled = z_percentile*100./percentile
#                  
#                  plume_height[b,t] = z_scaled
#                  
#                  #adding an additional trick, if the smaller bin is higher, we set it to the same height as the current bin
#                  if b>0 and  plume_height[b-1,t]>plume_height[b,t]:
#                      #print('bin height correction activated in ',b,t)
#                      plume_height[b-1,t] = plume_height[b,t]
#                  
#                  prof_mean_w_area_weighted = func_prof_sum(clus_tmp_t,'weighted w',n_z)/func_prof_sum(clus_tmp_t,'Area profile',n_z)
#                  #func_prof_mean(clus_tmp_t,'w profile',n_z)
#                  #print('w below height',prof_mean_w_area_weighted,b,t)
#
#                  #So the new new idea is to move up plumes which are higher than where the area weighted mean becomes negative 
#                  #plume_w_below_height[b,t]=np.nanmean(prof_mean_w[:int(z_scaled/dz)])
#                  if np.nanmin(prof_mean_w_area_weighted)<0: 
#                      plume_w_below_height[b,t]=np.where(prof_mean_w_area_weighted<0)[0][0]*dz
#                      #print('w below height',np.where(prof_mean_w_area_weighted<0)[0][0],b,t)
#                  else:
#                      plume_w_below_height[b,t]=n_z*dz
#
#
#
#              else:
#                  plume_height[b,t] = n_z*dz
#                  plume_w_below_height[b,t] = 100.
#
#      #if w_sorting_flag ==1:
#      #    print('plume_w_below b=0 ',plume_w_below_height[0,:])
#      #    w_matrix = np.vstack(clus_prop['w profile'].values)
#
#      for b in range(max_bin):
#          for t in t_iterable:
#              #bump_up_idx = np.where((bin_ind==b+1) & (clus_prop['time']==time_vec[t]) & (clus_prop['base']>plume_height[b,t]))
#              #lets try making it more restrictibe, namely if half the plume is above the detected height
#              if z_sorting_flag==1:
#              
#                  #This is more aggresive, bumps clusters up when their mean height is above the plume height. 
#                  bump_up_idx = np.where((bin_ind==b+1) & (clus_prop['time']==time_vec[t]) & ((clus_prop['base']+clus_prop['height']/2)>plume_height[b,t]))
#                  
#                  #This only bumps up when the base of the cluster is above the plume height
#                  #bump_up_idx = np.where((bin_ind==b+1) & (clus_prop['time']==time_vec[t]) & ((clus_prop['base'])>plume_height[b,t]))
#                  #print(b,t,bump_up_idx)
#                  bin_ind[bump_up_idx] = bin_ind[bump_up_idx] + 1
#              
#              
#              if w_sorting_flag==1:
#                  bump_up_idx = np.where((bin_ind==b+1) & (clus_prop['time']==time_vec[t]) & (clus_prop['base']>plume_w_below_height[b,t]))
#                  
#                  #print(b,t,bump_up_idx) 
#                  #if len(bump_up_idx)>1:
#                  print('w bump',b,t,len(bump_up_idx[0]))
#                  bin_ind[bump_up_idx] = bin_ind[bump_up_idx] + 1
#                  #z_idx = int(plume_height[b,t]/dz) 
#                  #bump_up_idx = np.where((bin_ind==b+1) & (clus_prop['time']==time_vec[t]) & ((clus_prop['base'])>plume_height[b,t]))
#                  #if z_idx<n_z:
#                      #bump_up_idx = np.where((bin_ind==b+1) & (clus_prop['time']==time_vec[t]) & (w_matrix[:,z_idx]>plume_w_below_height[b,t]))
#                      #print(b,t,bump_up_idx,plume_w_below_height[b,t])
#                      
#                      #bin_ind[bump_up_idx] = bin_ind[bump_up_idx] + 1
#    
#
#
#
#
#    #And then we have to recalculated the size distribution         
#    for b in range(bin_n):
#        if len(bin_ind[bin_ind==b+1])>plume_min:
#            csd[b] = float(np.count_nonzero(bin_ind==b+1))/(bins[b+1]-bins[b])
#        else:
#            csd[b] = 'nan'
#
#    return bin_n, bins,bin_ind,csd
