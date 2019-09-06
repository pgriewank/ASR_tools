#proc_watershed.py

#proc_watershed contains the functions to cluster output using a couvreux tracer, as well as the function to calculate the resulting plume properties. 
#Full details of how that happens are in the relevant files. 



from skimage.morphology import watershed
from skimage.segmentation import relabel_sequential
from skimage.segmentation import find_boundaries

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
import time as ttiimmee
import math
from datetime import datetime,timedelta
import time as ttiimmee
import sys
import pickle
import pandas as pd
import glob
sys.path.insert(0, "/home/pgriewank/code/2019-chords-plumes/")
from cusize_functions import *
import scipy.ndimage as ndi
from random import random




# In[2]:



def add_buffer(A,n_extra):
    """Adds n_extra cells/columns in x and y direction to array A. Works with 2d and 3d arrays, super advanced stuff right here. """
    if A.ndim == 2:
        A_extra = np.vstack([A[-n_extra:,:],A,A[:n_extra,:]])
        A_extra = np.hstack([A_extra[:,-n_extra:],A_extra,A_extra[:,:n_extra]])
    if A.ndim == 3:
        A_extra = np.concatenate((A[:,-n_extra:,:],A,A[:,:n_extra,:]),axis=1)
        A_extra = np.concatenate((A_extra[:,:,-n_extra:],A_extra,A_extra[:,:,:n_extra]),axis=2)
        
    
    return A_extra
 


# In[3]:


def sort_and_tidy_labels(segmentation):
    """
    For a given 3D integer array sort_and_tidy_labels will renumber the array 
    so no gaps are between the the integer values and replace them beginning with 0 upward. 
    Also, the integer values will be sorted according to their frequency.
    
    
    1D example: 
    [4,4,1,4,1,4,4,3,3,3,3,4,4]
    -> 
    [0,0,2,0,2,0,0,1,1,1,1,0,0]
    """

    unique_labels, unique_label_counts = np.unique(segmentation,return_counts=True)
    n_labels = len(unique_labels)
    unique_labels_sorted = [x for _,x in sorted(zip(unique_label_counts,unique_labels))][::-1]
    new_labels = np.arange(n_labels)

    lin_idx       = np.argsort(segmentation.ravel(), kind='mergesort')
    lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(segmentation.ravel())[:-1]))
    #think that I can now remove lin_idx, as it is an array with the size of the full domain. 
    del(lin_idx)

    for l in range(n_labels):
        c = unique_labels_sorted[l]
        if l!=c: #C==l should only happen for 0 and maybe a few more labels. 
            idx_z,idx_x,idx_y = np.unravel_index(lin_idx_split[c],segmentation.shape)
            segmentation[idx_z,idx_x,idx_y] = new_labels[l]

    return segmentation 
    





def cluster_watershed_couvreux_3D(data_c,data_w,time,
                                  z_level=0,n_max=0,buffer_size=10,
                                  c_thresh_mask=1,c_thresh_marker=2,w_min=0., 
                                  clus_flag=0 ,merge_flag=0):
    """
    Function that for a given 3D couvreux scalar and w file detects and return clusters in a labelled 3D array. Should take about 5 minutes for a 256x1024x1024 field.  
   
    Clustering is done as follows:
    0. Expand data in x and y direction to deal with periodic boundary domains. 
    1. Calculate the relative couvreaux concentration in each grid box compared to the mean and standard deviation (c_rel = (c-c_mean)/c_std)
       Important to note that both c_mean and c_std have minimal values to avoid detecting plumes high up due to numerical issues. 
    2. Apply c_thresholds to calculate c_rel. Depending on the clustering flag used c_rel is determined diffently.
       To determine contionous areas to serve as markers c_rel>c_thresh_marker is used, and the mask of total area to be filled us guven by c_rel>c_thresh_mask.
    3. Use watershed to fill the masked area around the markers. If a masked area is not connected to a marker, it will be ignored.
    4. If merge_flag =1, merge together all connection plumes with a sufficient size difference. Expensive, see cluster_sizemerge_neigh_3D.
    4. Throw out all plumes for which the mean horizontal point does not lie within the origninal area.
    5. If the segments extends beyond the boarders of the original domain the affected areas need to be remapped  
    Cancelled: 6. Reorganize labels to get rid of gaps created by step 4, and organize so that the segment volume decreases with segment numbering. 0 always labels the largest non plume area. 
    Step 6 was removed to save time (approx 20% faster) and to not mess up the matching labeling of markers and plumes/segments.
    
    Parameters:
      file_c:            the couvreux file
      file_w:            the w file
      time:              time index
      z_level:           only load fields till this height, useful for reducing size to speed things up and avoid getting into areas where couv is too low. 
      n_max:             only load fields till this x and y number, horribly named. Useful for reducing size for quick testing
      buffer_size:       percentage of field to add as a buffer to each edge to deal with periodic boundary domains. Large value slows things down, small one increases chance a freak plume reaches the wall. 
      c_thresh_mask:     couvreux threshold which determines total area to be filled by plumes. The lower the bigger the domain
      c_thresh_masker:   couvreux threshold which determines areas to be used as marker for watershed algorithm. The higher the less number of plumes detected
      w_min:             Additioanl constraint on mask and markers regarding w. eyeballing seems to indicate that it mostly only removes cells along the plume edge. 
      clus_flag:         0: uses couvreux thresholds and w value, 1: uses w*couvreux, no w threshold, 2: only couvreux threshold, no w limitation
      merge_flag:        0 nothing, 1: clusters are merged together using cluster_sizemerge_neigh_3D. Warning, can take a while. 
      
      
    Output:  
      segmentation_orig:    3D array of integer values showing which label it belongs to each plume. Zero is always the non plumes
      labeled_markers_orig: 3D array of integer values showing which label it belongs to which marker. Zero is no marker
      mask_orig:            3D array of zeros and ones showing which parts of the array are filled by the watershed algorithm
    
    
    
    IMPORTANT! The mean and standard deviation used to calculate c_rel in step 1 can only be taken from the original field, not the extended field used to deal with periodic boundary conditions! The mean and std can change when adding the extra to the sides. 
    WARNING: Will easily use 20 GB of data to run on a 256x1024x1024 file.  
    """
    #Uses a default periodic boundary domain
 
    c_std_min  = 1e-7  #Minimum sigma, intended to avoid issues with numerical precision when couvreux scalar is almost zero everywhere and also avoid deviding by zero
    c_mea_min  = 1e-10 #Minimum mean, couvreux sampling approach doesn't work well if almost no tracer has reached a level yet. 
                       #Is set to its current value just by eyeballing the output, started with 1e-8 which was too high.
                       #Is also devided by 10 and used to set the couvreux density where the density is below zero. 
    
    if z_level ==0:
        test_vec = data_c.variables['couvreux'][0,:,0,0]
        z_level = test_vec.size
    if n_max ==0:
        test_vec = data_c.variables['couvreux'][0,0,:,0]
        n_max   = test_vec.size
        
    
    
    c_3D_orig = grab_3d_field(data_c,time,'couvreux')
    w_3D_orig = grab_3d_field(data_w,time,'w')
    
    
    #Important, when the couvreux scalar is below zero we set it to the mean min
    c_3D_orig[c_3D_orig<0.0]=c_mea_min/10.
    
    #Now limiting to z_level 
    c_3D_orig = c_3D_orig[:z_level,:,:]
    w_3D_orig = w_3D_orig[:z_level,:,:]
    
    n_buffer = int(buffer_size/100.*n_max)

    #Explanding c and w fields with a buffer on each edge to take periodic boundaries into account. 
    c_3D  = add_buffer(c_3D_orig,n_buffer)
    w_3D  = add_buffer(w_3D_orig,n_buffer)
    c_rel = np.zeros_like(c_3D)
        
    
    if clus_flag==0 or clus_flag==2:
        #Original version, uses only couvreux to determine tracers and imposes the w>0 separately as well as the c_std and s_mea limiters
        #If clus_flag ==2 then the w_min limiting isn't applied
        
        #It is important that the mean and std are only calculated from inside the original domain
        c_rel = np.zeros_like(c_3D)
        for i in range(c_rel.shape[0]):
            c_std = np.std( c_3D_orig[i,:,:].ravel())
            c_mea = np.mean(c_3D_orig[i,:,:].ravel())
            c_2D=c_3D[i,:,:]
            c_std = max(c_std,c_std_min)
            c_mea = max(c_mea,c_mea_min)
            c_rel[i,:,:] = (c_2D-c_mea)/c_std

        if clus_flag==0:
            #Adding the w restriction straight to the c_rel variable, makes it easier to visualize using c_rel 
            c_rel[w_3D<w_min] = 'nan'

    
       

        
        
    if clus_flag==1:
        #Uses w*couvreux instead of only couvreux. No restrictions or limits so far, but function is only called when couvreux has a minimal value
        for i in range(c_rel.shape[0]):
            c_std = np.std( c_3D_orig[i,:,:].ravel()*w_3D_orig[i,:,:].ravel())
            c_mea = np.mean(c_3D_orig[i,:,:].ravel()*w_3D_orig[i,:,:].ravel())
            #c_std = max(c_std,c_std_min)
            #c_mea = max(c_mea,c_mea_min)
            
            c_2D=c_3D[i,:,:]*w_3D[i,:,:]
            c_rel[i,:,:] = (c_2D-c_mea)/c_std

         




    markers         = np.zeros_like(c_rel).astype(int)
    segmentation    = np.zeros_like(c_rel).astype(int)
    labeled_plumes  = np.zeros_like(c_rel).astype(int)
    mask = np.zeros_like(c_rel).astype(int)
    
    #creating a mask to get rid of everything where c_rel above c_thresh_mask
    mask[c_rel>=c_thresh_mask] =1

    #markers! All markers need to have an individual number to avoid them bleeding into each other
    #so now I am only looking into where c_rel greater than c_thresh_marker, and then using ndi.label to give each an individual number 
    markers[c_rel>c_thresh_marker] =1
        
    
    #No longer needed. 
    del(w_3D,c_3D)
    
    #This is already very impressive, ndi.label detects all areas with marker =1 that are connected and gives each resulting cluster an individual integer value 
    labeled_markers, _ = ndi.label(markers)
    
    
    #This is where the actualy magic happens, the masked area is flooded from the markers using -c_rel to determine speed. Is pretty bloody fast all things considered. 
    segmentation = watershed(-c_rel, labeled_markers,mask=mask)
    
    
    #Now if merging should happen, it has to be done here. 
    #Annoyingly it needs to call sort_and_tidy beforehand, ruins the the labeled_marker to segmentation connection, and requires relabel_sequential afterwards. 
    if merge_flag==1:
        time_beg = ttiimmee.time()
        sort_and_tidy_labels(segmentation);
        segmentation,bla,bla   = cluster_sizemerge_neigh_3D(segmentation)
        segmentation, bla, bla = relabel_sequential(segmentation)
        print('cluster merging with relabelling done in minutes:',str((ttiimmee.time()-time_beg)/60)[:4])
        del(bla)
    
    # Going back from the padded field back to the original size requires removing all dublicate plumes and relabelling things around the edge. 
    # OK, calculate index means, then only look at those with a mean inside the original box
    # We ignore the cells with the mean outside, they will be cut off or overwritten
    # For those inside we check if they have something outside original box, and if so a very ugly hard coded overwritting is done. 
    # In the very end the segmentation box is cut back down to the iriginal size
    # WARNING! segmentation no longer contains all indexes. Might be worth to change that. 

    n_cluster = np.max(segmentation)
    #print('number of clusters including buffer in timestep ',time,n_cluster)
    
    if n_cluster>0:
        lin_idx       = np.argsort(segmentation.ravel(), kind='mergesort')
        lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(segmentation.ravel())[:-1]))
        del(lin_idx)


        for c in range(1,n_cluster+1): 

            idx_z,idx_x,idx_y = np.unravel_index(lin_idx_split[c],segmentation.shape)
            idx_x_m = np.mean(idx_x)
            idx_y_m = np.mean(idx_y)

            if idx_x_m< n_buffer or idx_x_m>n_buffer+n_max or idx_y_m< n_buffer or idx_y_m>n_buffer+n_max:
                #cluster is outside, chuck it
                #print(c,'cluster out of bounds',idx_x,idx_y)
                #segmentation_cp[segmentation==c] = 0
                bla = 1

            else:
                idx_x_max = np.max(idx_x)
                idx_x_min = np.min(idx_x)
                idx_y_min = np.min(idx_y)
                idx_y_max = np.max(idx_y)
                if idx_x_min< n_buffer or idx_x_max>n_buffer+n_max or idx_y_min< n_buffer or idx_y_max>n_buffer+n_max:
                    #print(c,'this is our guniea pig')
                    if idx_x_min<n_buffer:
                        idx_x_sel = idx_x[idx_x<n_buffer]+n_max
                        idx_y_sel = idx_y[idx_x<n_buffer]
                        idx_z_sel = idx_z[idx_x<n_buffer]
                        segmentation[idx_z_sel,idx_x_sel,idx_y_sel] = c
                    if idx_x_max>=n_buffer+n_max:
                        idx_x_sel = idx_x[idx_x>=n_buffer+n_max]-n_max
                        idx_y_sel = idx_y[idx_x>=n_buffer+n_max]
                        idx_z_sel = idx_z[idx_x>=n_buffer+n_max]
                        segmentation[idx_z_sel,idx_x_sel,idx_y_sel] = c
                    if idx_y_min<n_buffer:
                        idx_x_sel = idx_x[idx_y<n_buffer]
                        idx_y_sel = idx_y[idx_y<n_buffer]+n_max
                        idx_z_sel = idx_z[idx_y<n_buffer]
                        segmentation[idx_z_sel,idx_x_sel,idx_y_sel] = c
                    if idx_y_max>=n_buffer+n_max:
                        idx_x_sel = idx_x[idx_y>=n_buffer+n_max]
                        idx_y_sel = idx_y[idx_y>=n_buffer+n_max]-n_max
                        idx_z_sel = idx_z[idx_y>=n_buffer+n_max]
                        segmentation[idx_z_sel,idx_x_sel,idx_y_sel] = c
    else:
        labeled_markers = segmentation



    
    
    #Now cut to the original domain size again
    segmentation_orig = segmentation[:,n_buffer:-n_buffer,n_buffer:-n_buffer]
    labeled_markers_orig = labeled_markers[:,n_buffer:-n_buffer,n_buffer:-n_buffer]
    mask_orig = mask[:,n_buffer:-n_buffer,n_buffer:-n_buffer]
    c_rel_orig = c_rel[:,n_buffer:-n_buffer,n_buffer:-n_buffer]
   
    

    return segmentation_orig,labeled_markers_orig ,mask_orig,c_rel_orig



    



def cluster_sizemerge_neigh_3D(c_mask, size_ratio_thresh=100.):
    """
    For a given segmentation cmask array this function will merge smallest cells into neighbouring ones if they are sufficiently larger.  
   
    In general we don't care too much about things being 100% perfect, as the clustering itself isn't devine either. 
    
    Clustering is done as follows:
    I first filter out which clusters are too small to eat a different cluster, e.g. smaller than size_ratio_thresh cells
    Then I loop over all big enough clusters checking for edible neighbours. 
    If there are non, I note in a list that this cluster can not eat anything nearby and does not have to be revisited.
    If it can eat a neighbour, that neighbour is eaten.
    I then repeat the loop over all clustes which have eaten to see if they can now eat more clusters. 
    This loop is repeated until all clusters can not eat anything anymore. 

    Abandoned approaches. 
    I first started off by going through all small clusters and deciding if they should be merged into a big one. This was nice and fast, but didn't account for the large clusters changing in size and shape as they grew.  
    Secondly I start with the biggest plumes and had each of them eat as many plumes as possible iteratively until no more could be eaten, which had the dissadvantage of the larger plumes growing very strongly, eating up smaller plumes until they hit a big plume. 
    The current approach avoids this my having the plumes all eat only their immediate neighbours in a row, ensuring that small plumes are eaten by their immediate neighbour. 

    Sadly, this is not a fast routine, it can currently take over 5 minutes for a single timestep.   
    Despite my crude optimization attempts detecting the neighbouring cells of the plumes has to be done again and again which just isn't cheap. 
    A lot of time was saved by adding the new indexes of each eaten plume to its eater directly to avoid having to recalculate cluster indexes each time they change. 
    The time needed should be fine for my purposes. 

    Important! 
    This function takes much longer when clusters cross the periodic boundary domain. 
    This is because to save time I only search for neighbours in a rectangular box around the cluster, and if that cluster crosses the domain so does the bounding rectangular box. 
    It would be possible to avoid this by being a bit smarter when cutting the bounding box, but I didn't bother because I normally apply it 
    to the expanded fields segmenation domain which occurs before taking care of the periodic boundary conditions. 
    
    
    Parameters:
      c_mask:            the 3D segmentation field, with each cluster being labeled with a unique integer. 
      size_ratio_thresh: how much bigger the neighbourng plumes volume has to be to be merged. Just started with 100, seems to work fine.  
      
      
     
    Output: 
      c_mask_merged:    merged c_mask, warning not sorted. 
      list_of_iterat:   processing stats, says for each cluster how many iterations if neighbour eating occured. 
      list_of_eaters:   processing stats, says for each cluster of many neighbours were eaten.
    
    
    """
 
    #First of all get cluster_sizes and also total number, as well as the dimensions
    [nz,nx,ny]  = c_mask.shape
    
    c_mask_merged = c_mask+0
    cluster_label, cluster_size = np.unique(c_mask,return_counts=True)
    n_cluster = cluster_label[-1]
    
    #then we got to do the index thing (takes about 30 s)
    
    #Temorary speed boost
    lin_idx       = np.argsort(c_mask.ravel(), kind='mergesort')
    lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(c_mask.ravel())[:-1]))
    
    del(lin_idx)
    
    c_3D_zeros = (np.zeros_like(c_mask)).astype(int)
    
    
    first_cluster = np.where(cluster_size/cluster_size[1]<1/size_ratio_thresh)[0][0]
    print(first_cluster,'first_cluster which could be eaten')
    last_cluster = np.where(cluster_size>size_ratio_thresh)[0][-1]
    print(last_cluster,'last cluster that could eat someone')
    
    n_merged=0
    list_of_merged = np.zeros_like(cluster_label) #if zero that idx has not been merged already, if 1 it has finished being merged
    list_of_eaters = np.zeros_like(cluster_label) #lists the amount of clusters eaten by each cluster
    list_of_iterat = np.zeros_like(cluster_label) #lists the number of times things were calculated for each plume
    
    #Ruling out all that can't eat anyone, which avoids calculating their neighbours. 
    list_of_merged[last_cluster:] =  1 
    #And of course I need to get rid of the background first, which I totally didn't forget the first time. 
    list_of_merged[0] =  1 
    
    
    if first_cluster>1: #Checking if anything can be eaten. 
        while min(list_of_merged)==0:i #keep going until no cluster can eat anything anymore. 
            time0 = ttiimmee.time()
            for c in range(1,last_cluster):
                if list_of_merged[c]==0:
                    time1 = ttiimmee.time()
                
                
                    #To speed up the boundary hunting we make a rectangular box around the cluster. 
                    idx_z,idx_x,idx_y = np.unravel_index(lin_idx_split[c],c_mask.shape)
                    c_3D_zeros[idx_z,idx_x,idx_y] = 1
                    z_min = max(np.min(idx_z)-1,0   )
                    z_max = min(np.max(idx_z)+1,nz )
                    x_min = max(np.min(idx_x)-1,0   )
                    x_max = min(np.max(idx_x)+1,nx)
                    y_min = max(np.min(idx_y)-1,0   )
                    y_max = min(np.max(idx_y)+1,ny)


                    boundaries = find_boundaries(c_3D_zeros[z_min:z_max,x_min:x_max,y_min:y_max],mode='outer').astype(np.uint8) 
                    c_3D_zeros[idx_z,idx_x,idx_y] = 0

                    #Now making a small c_mask to match the boundaries
                    c_mask_small = c_mask_merged[z_min:z_max,x_min:x_max,y_min:y_max]
                    neighbours = np.unique(c_mask_small[boundaries==1])

                    #now finding which neighbours are small enough to be eaten
                    prey = neighbours[cluster_size[c]/cluster_size[neighbours]>size_ratio_thresh]
                    if prey.size:
                        for p in prey:
                            list_of_eaters[c] = list_of_eaters[c]+1
                            #print(c,' is eating ',p, ttiimmee.time()-time0,x_max-x_min,y_max-y_min)
                            idx_z_prey,idx_x_prey,idx_y_prey = np.unravel_index(lin_idx_split[p],c_mask.shape)

                            c_mask_merged[idx_z_prey,idx_x_prey,idx_y_prey]=c
                            cluster_size[c]=cluster_size[c]+cluster_size[p]
                            lin_idx_split[c]=np.hstack([lin_idx_split[c],lin_idx_split[p]])
                            n_merged = n_merged+1

                            list_of_merged[p] =1 #eaten clusters will no longer be checked for neighbours
                        list_of_iterat[c]=list_of_iterat[c]+1
                        #print('cluster ',c, ' ate ',list_of_eaters[c], ' in iteration ',list_of_iterat[c],',eating clusters left',n_cluster-sum(list_of_merged), ' time: ',ttiimmee.time()-time1)

                    else:
                        list_of_merged[c]=1 #Nothing to eat, time to move on



                        #Abandoned first version
                        #size_largest_neighbour=cluster_size[largest_neighbour]
                        #largest_neighbour = np.min(neighbours[neighbours>0])

                        #if size_largest_neighbour/size_ratio_thresh>cluster_size[c]:
                            #When merged all the c_mask of the to be merged cluster is rewritten, the indexes are added to the lin_inx_plit, and the volume is added to the large ones volume

                            #print(c,' we got a merger with ', largest_neighbour,ttiimmee.time()-time0,x_max-x_min,y_max-y_min)
                        #    c_mask_merged[idx_z,idx_x,idx_y]=largest_neighbour
                        #    lin_idx_split[largest_neighbour]=np.hstack([lin_idx_split[largest_neighbour],lin_idx_split[c]])
                        #    cluster_size[largest_neighbour]=cluster_size[largest_neighbour]+cluster_size[c]
                        #else:
                            #print(c,' no wedding',ttiimmee.time()-time0,x_min,x_max,y_min,y_max)

    print(n_merged,' clusters eaten by larger clusters from an initial ',n_cluster)
   
    return c_mask_merged,list_of_iterat,list_of_eaters



    


def proc_clustering_watershed_couv_w(folder_str,
                                     input_dir='/data/testbed/lasso/sims/',
                                     output_dir='/data/testbed/lasso/sims/',
                                     c_thresh_mask=1.0,c_thresh_marker=2.0,w_min=0.,special_name='',
                                     t_start=0,t_end=0,
                                     clus_flag=0,merge_flag=0):
    """
    Function to cluster plumes for a full day. For the details on the clustering is conducted see cluster_watershed_couvreux_3D.
    
    For optimization purposes clustering is limited to the layers which have a mean couvreux tracer concentration above a hard coded c_mea_min value.
    c_mea_min is set to 1e-10 by eye, original value of 1e-8 was too strict.
    
    To avoid fumbling around with netcdf properties, a copy of an existing file is made with cdo with a changed variable name. 
    
    WARNING: File is not created new if it already exists, which means that old results can still be in the old file if only a few time slices are calculated. 
    
    Parameters:
        c_thresh_mask:     couvreux threshold which determines total area to be filled by plumes. The lower the bigger the domain
        c_thresh_masker:   couvreux threshold which determines areas to be used as marker for watershed algorithm. The higher the less number of plumes detected
        w_min:             Additional constraint on mask and markers regarding w. eyeballing seems to indicate that it mostly only removes cells along the plume edge. 
        special_name:      A string that can be used to add text to the end of the c_mask file.
        t_start:           First timestep that is looked at 
        t_end:             Last timestep that is computed           
        clus_flag:         Decides if either just couvreux clustering is used, or w*couvreux, or couvreux without w restriction
        merge_flag:        If activated merges much smaller clusters with ones it is in contact with.
        
    Don't return anything. 
    """
    import subprocess
    import os.path
    
    c_mea_min  = 1e-10  #Minimum mean couvreux in profile, used to set the z_level to speed things up and avoid issues where 
                        #Is set by eyeballing things
   
    
    
    #import sys
    time0 = ttiimmee.time()
    #Need to get cdo command working, first attempt using subprocess
    output_file = output_dir+folder_str+'/c_mask_'+str(c_thresh_mask*100)[:3]+'c_mark_'+str(c_thresh_marker*100)[:3]+'_cf'+str(clus_flag)+'_mf'+str(merge_flag)+special_name+'.nc'
    file_c = output_dir+folder_str+'/couvreux.nc'
    file_w = output_dir+folder_str+'/w.nc'
    filename_prof=glob.glob(output_dir+folder_str+'/*default?0*.nc')[0]
    
    
    
    #Checking if the output file already exists, if not create it
    if os.path.isfile(output_file): 
        print('output file already exists, will overwrite: ',output_file)
    else:
        cdo_command_str = 'cdo -b I32 setname,c_mask '+file_c+' '+output_file
        print('executing: ',cdo_command_str)
        #aborted reducing accuracy of c_mask file, the saving in size just isn't worth the hassle of ensuring the numbers don't exceed the expected
        subprocess.check_call(cdo_command_str,shell=True)
        #print(cdo_command_str)
    
    data_w = Dataset(file_w)
    data_c = Dataset(file_c)
    
    file_prof  =  Dataset(filename_prof,read='r')
    
    #So if we managed to make the output file, time to do the loop, we will test with z_level = 5 to speed things up
    #get dimensions
    n_time = data_c.variables['couvreux'][:,0,0,0].size
    [n_z, n_x, n_y] =  get_zxy_dimension(file_c,'couvreux')
    
    if t_end ==0:
        t_end = n_time
    
        
    
    for t in range(t_start,t_end):
        time1 = ttiimmee.time()
        print('timestep: ',t)
        
        #Determine z_level from couvreoux profile in default profile 
        couv_prof = file_prof['couvreux'][t,:]
        z_level = np.argwhere(couv_prof < c_mea_min)[0][0]
        print('z_level for timetstep ',t,' :',z_level, 'which means no clusters above ',z_level*25)
        
        if z_level>1:
            
            #Only couvreux clustering
            segmentation, bla, bla, bla= cluster_watershed_couvreux_3D(data_c,data_w,t,z_level=z_level,n_max=n_x,buffer_size=10,
                                                                       c_thresh_mask=c_thresh_mask,c_thresh_marker=c_thresh_marker,w_min=w_min,
                                                                       clus_flag=clus_flag,merge_flag=merge_flag)
            

             
            sort_and_tidy_labels(segmentation);

            #Output
            file_cmask = Dataset(output_file,"r+")
            
            #Not proud of this, but because I use zxy for coordinates but the c_mask file is in zyx I need to x and y before writting. 
            
            
            dim_space = list(file_cmask.variables['c_mask'].dimensions)
            
            if 'z' in dim_space[1] and 'x' in dim_space[3] and 'y' in dim_space[2]:

                file_cmask.variables['c_mask'][t,:z_level,:,:] =  segmentation.swapaxes(1,2)
                file_cmask.variables['c_mask'][t,z_level:,:,:] =  0.
            else:
                print('dimension mismatch writting to ',output_file,'aborting')
                []+1
            file_cmask.close()
            time2 = ttiimmee.time()
            print('timestep: ',t,' took ',str((time2-time1)/60)[:4], ' minutes for n_clusters: ', np.max(segmentation))
        else:
            print('so no clustering chap')
    
   
        
   
    time4 = ttiimmee.time()
    time_total =  str((time4-time0)/60.)
    print('done clustering in ',time_total[:4],' minutes, output written to: ',output_file)
    
    return 




def proc_couv_prop(folder_str = '20160611',c_name = '/c_mask_100c_mark_200_cf0.nc',
                    special_name = '',
                    directory_data          = '/data/testbed/lasso/sims/' ,
                    directory_output        = '/data/testbed/lasso/clustering/',
                    t_start = 0,
                    t_end = 0
                    ):
    
    """
    Calculates the properties of the clusters written in  a c_mask file.
    
    Importantly, it does not apply any filters beyond a minimum size filt_N_min. 
    
    Sorry, needs documentation.

    Important parameters: filt_N_min, minimum number of cells needed for cluster. 
    """
    
    dx = 25
    dy = 25


    
    
    #So we assume that the field files are in the same folder as the segmentation file


    #Input

    ql_min = 1e-6 #cutoff for cloudy


    #Filters
    #filt_h_min = 8  #given in grid boxes, so each plume must be this high. Is intended to filter out small turbulence with an extent of less than 200 m
    #filt_V_min = filt_h_min*4 # The idea is that the smallest volume is equal to the minimum height but with 50x50 m. 
    filt_N_min = 30 # Thanks to clusters already being sorted by number of cells, it is easiest to just cut off everything beyond a certain index.  






    time0 = ttiimmee.time()


    #Assigning name
    filename_w = directory_data+folder_str+'/w.nc'
    filename_ql= directory_data+folder_str+'/ql.nc' 
    filename_qt= directory_data+folder_str+'/qt.nc' 
    filename_couv = directory_data+folder_str+'/couvreux.nc'
    

    filename_c = directory_data+folder_str+c_name

    filename_out = directory_output + 'prop_'+folder_str+'_'+c_name[1:-3]+special_name+'.pkl'

    filename_prof=glob.glob(directory_data+folder_str+'/*default?0*.nc')[0]
    file_prof  =  Dataset(filename_prof,read='r')

    


    #Getting dz
    z_prof = file_prof['z'][:]
    dz = z_prof[1]-z_prof[0]

    dA = dx*dy
    dV = dx*dy*dz

    n_z,n_x,n_y = get_zxy_dimension(filename_w,'w')

    #got to make timesteps automated


    #Time loop that loads all the cluss identified via clustering and calculates their properties



    #loading w 
    file_w    =  Dataset(filename_w ,read='r')
    file_ql   =  Dataset(filename_ql,read='r')
    file_qt   =  Dataset(filename_qt,read='r')
    file_c    =  Dataset(filename_c ,read='r')
    file_couv =  Dataset(filename_couv ,read='r')
    
    print('beginning to calculate the cluster properties of ',filename_c)


    seconds_since_start = file_w.variables['time'][:]

    timesteps = len(seconds_since_start)

    try:
        time_init = datetime.strptime(dates[d][0:7]+'0600','%Y%m%d%H%M')
    except:
        time_init = datetime.strptime('2020'+'01'+'01'+'0600','%Y%m%d%H%M')
    #loading clustering from file 
    time1 = ttiimmee.time()



    couv_A_all = np.zeros(0)
    couv_w_all   = np.zeros(0)
    couv_V_all   = np.zeros(0)
    couv_h_all   = np.zeros(0)


    couv_base_all   = np.zeros(0)
    couv_max_cf_all = np.zeros(0)

    couv_prof_w_all  = np.zeros([0,n_z])
    couv_prof_w90_all  = np.zeros([0,n_z])
    couv_prof_A_all  = np.zeros([0,n_z])
    couv_prof_qv_all  = np.zeros([0,n_z])
    couv_prof_ql_all  = np.zeros([0,n_z])
    couv_prof_cf_all  = np.zeros([0,n_z])
    couv_prof_couv_all  = np.zeros([0,n_z])

    #fluxes
    couv_prof_flux_w_all  = np.zeros([0,n_z]) #Poor mans mass flux
    couv_prof_flux_qt_all  = np.zeros([0,n_z]) 
    couv_prof_total_flux_qt_all  = np.zeros([0,n_z]) 

    #fluctuations
    couv_prof_fluc_qt_all  = np.zeros([0,n_z]) 



    couv_t_all = []
    couv_x_max_all = []
    couv_y_max_all = []
    couv_z_max_all = []

    couv_wet_w_all   = np.zeros(0)
    couv_wet_V_all   = np.zeros(0)
    couv_wet_A_all   = np.zeros(0)
    couv_wet_h_all   = np.zeros(0)
    couv_dry_w_all   = np.zeros(0)
    couv_dry_V_all   = np.zeros(0)
    couv_dry_A_all   = np.zeros(0)
    couv_dry_h_all   = np.zeros(0)

    couv_wet_prof_A_all  = np.zeros([0,n_z])
    couv_dry_prof_A_all  = np.zeros([0,n_z])


    if t_end ==0: t_end=timesteps
    

    for t in range(t_start,t_end):
        c_mask = grab_3d_field(file_c,t,'c_mask')

        ncluss = np.max(c_mask)
        #(cluster_cell_list_time[t])
        time_now = time_init + timedelta(seconds=float(seconds_since_start[t]))
        print('timestep and ncluss ',t,ncluss)
        print('datetime ',t,time_now)

        if ncluss>0:
            time1 = ttiimmee.time()

            #load data
            w  = grab_3d_field(file_w,t,'w')
            ql = grab_3d_field(file_ql,t,'ql')
            qt = grab_3d_field(file_qt,t,'qt')
            couv = grab_3d_field(file_couv,t,'couvreux')
            qt_mean_prof = file_prof.variables['qt'][t,:]

            qv = qt-ql


            w_qt_fluc = qt*0.0
            for n in range(n_z):
                w_qt_fluc[n,:,:]=w[n,:,:]*(qt[n,:,:]-qt_mean_prof[n])
            w_qt = w*qt


            #This sorting into lin_idx_split is crucial as a way to quickly find the indexes and size of each cluster
            lin_idx       = np.argsort(c_mask.ravel(), kind='mergesort')
            lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(c_mask.ravel())[:-1]))
            c_mask_shape = c_mask.shape
            
            time11 = ttiimmee.time()


            #This should probably be possible to be done much quicker with skimage.morphology.remove_small_objects. 
            
            
            #Before making all the variables to fill, I first determine which clusters are too small to be included
            ncluss_short = 1
            while len(lin_idx_split[ncluss_short])>filt_N_min:
                ncluss_short+=1
            print('reduced ncluss from ',ncluss, ' to ',ncluss_short-1,' by cutting out all cluster with fewer cells than ',filt_N_min)

            #Now we need to add all small plumes that were excluded back to 0, and then recalculate the indexes (having to recalculate this is not great time wise)
            for i in range(ncluss_short,ncluss):
                idx_z,idx_x,idx_y = np.unravel_index(lin_idx_split[i],c_mask_shape)
                c_mask[idx_z,idx_x,idx_y]  = 0

            #This sorting into lin_idx_split is crucial as a way to quickly find the indexes and size of each cluster
            lin_idx       = np.argsort(c_mask.ravel(), kind='mergesort')
            lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(c_mask.ravel())[:-1]))
            
            print('time needed to get rid of small plumes in minutes:',(ttiimmee.time()-time11)/60. )


            del(lin_idx,c_mask)

            ncluss = ncluss_short-1



            couv_w = np.zeros(ncluss)
            couv_V = np.zeros(ncluss) 
            couv_A = np.zeros(ncluss)
            couv_h = np.zeros(ncluss)


            couv_max_cf = np.zeros(ncluss)
            couv_base = np.zeros(ncluss)

            couv_prof_w = np.zeros([ncluss,n_z]) 
            couv_prof_A = np.zeros([ncluss,n_z]) 
            couv_prof_ql = np.zeros([ncluss,n_z]) 
            couv_prof_qv = np.zeros([ncluss,n_z]) 
            couv_prof_cf = np.zeros([ncluss,n_z]) 
            couv_prof_couv = np.zeros([ncluss,n_z]) 

            couv_prof_flux_w = np.zeros([ncluss,n_z])
            couv_prof_flux_qt = np.zeros([ncluss,n_z])
            couv_prof_total_flux_qt = np.zeros([ncluss,n_z])
            couv_prof_fluc_qt = np.zeros([ncluss,n_z])

            couv_t = [] 
            couv_x_max = []
            couv_y_max = []
            couv_z_max = []

            couv_wet_w   = np.zeros(ncluss)
            couv_wet_V   = np.zeros(ncluss)
            couv_wet_A   = np.zeros(ncluss)
            couv_wet_h   = np.zeros(ncluss)
            couv_dry_w   = np.zeros(ncluss)
            couv_dry_V   = np.zeros(ncluss)
            couv_dry_A   = np.zeros(ncluss)
            couv_dry_h   = np.zeros(ncluss)

            couv_wet_prof_A = np.zeros([ncluss,n_z]) 
            couv_dry_prof_A = np.zeros([ncluss,n_z]) 

            #setting to nan incase they don't exist
            couv_wet_w[:]   = 'nan'
            couv_wet_V[:]   = 'nan'
            couv_wet_A[:]   = 'nan'
            couv_wet_h[:]   = 'nan'
            couv_dry_w[:]   = 'nan'
            couv_dry_V[:]   = 'nan'
            couv_dry_A[:]   = 'nan'
            couv_dry_h[:]   = 'nan'






    #             #Calculating cbl height + 300m using a critical value for the horizontal variability of w following Lareau
    #             w_var = 1.0
    #             z_var=0
    #             while w_var > 0.08:
    #                 z_var += 1
    #                 w_var = np.var(w[z_var,:])
    #             cbl_idx = z_var
    #             cbl_idx_max = cbl_idx+int(300/dz)

    #             print('cbl height + 300 m index :',cbl_idx_max)


            #calculate the z lvl of maximum amount of cluster fraction at that timestep
            idx_z,idx_x,idx_y = np.unravel_index(np.hstack(lin_idx_split[1:]),c_mask_shape)
            z_max_cf  = np.argmax(np.bincount(idx_z))




            for i in range(0,ncluss):
            #for i in range(1,ncluss):
                #To speed things up each set of indexes (all, wet, dry) is used to calculate profiles only once!
                #So everything needs to be gathered together before being passed on to func_vert_mean_sorted_argv
                iiii = 0
                time_tmp = ttiimmee.time()
                idx_z,idx_x,idx_y = np.unravel_index(lin_idx_split[i],c_mask_shape)
                #print('cluster ',i,iiii,'time spent loading indexes ',ttiimmee.time()-time_tmp); iiii += 1; time_tmp = ttiimmee.time() 

                couv_w[i]   = np.mean(w[idx_z,idx_x,idx_y])
                couv_V[i]   = (float(len(idx_z))*dV)**(1./3.)

                #This is a bit of hard coded speed up for the zero index, which covers almost the full domain
                if i>0:
                    couv_A[i]   = func_proj_A(idx_x,idx_y,dA)
                    couv_h[i]   = (np.max(idx_z)-np.min(idx_z)+1)*dz
                    couv_base[i]=np.min(idx_z)*dz
                    couv_x_max.append(np.argmax(np.bincount(idx_x)))
                    couv_y_max.append(np.argmax(np.bincount(idx_y)))
                    couv_z_max.append(np.argmax(np.bincount(idx_z)))

                else:
                    couv_A[i]   = n_x*n_y*dA
                    couv_h[i]   = n_z*dz
                    couv_base[i]= 0 
                    couv_x_max.append(0)
                    couv_y_max.append(0)
                    couv_z_max.append(0)

                couv_max_cf[i] = z_max_cf
                #print('cluster ',i,iiii,'time spent calculating A, h, base, V, w, x,y,z:',ttiimmee.time()-time_tmp); iiii += 1; time_tmp = ttiimmee.time() 



                prof_tmp,tmp =func_vert_mean_sorted_argv(idx_z,idx_x,idx_y,w,qv,ql,w_qt,w_qt_fluc,couv)
                couv_prof_w [i,:]            = prof_tmp[:,0]
                couv_prof_qv[i,:]            = prof_tmp[:,1]
                couv_prof_ql[i,:]            = prof_tmp[:,2]
                couv_prof_A[i,:]             = tmp*dA
                couv_prof_total_flux_qt[i,:] = prof_tmp[:,3] *couv_prof_A[i,:]
                couv_prof_flux_qt[i,:]       = prof_tmp[:,4] 
                couv_prof_couv[i,:]          = prof_tmp[:,5]

                #print('cluster ',i,iiii,'time spent calculating profiles:',ttiimmee.time()-time_tmp); iiii += 1; time_tmp = ttiimmee.time() 


                #Now everything that can be computed from the other fluxes 
                couv_prof_flux_w [i,:] = couv_prof_w[i,:]*couv_prof_A[i,:]
                couv_prof_fluc_qt[i,:] = couv_prof_qv[i,:]+couv_prof_ql[i,:]-qt_mean_prof



                couv_t.append(time_now) 



                #Now we start looking into the wet and dry parts of the clusters
                #Now the new part, separating the cluster into dry and wet

                ql_tmp = ql[idx_z,idx_x,idx_y]

                idx_z_dry = idx_z[ql_tmp<ql_min] 
                idx_x_dry = idx_x[ql_tmp<ql_min] 
                idx_y_dry = idx_y[ql_tmp<ql_min] 

                if idx_z_dry.size>0:
                    couv_dry_w[i]   = np.mean(w[idx_z_dry,idx_x_dry,idx_y_dry])
                    couv_dry_V[i]   = (float(len(idx_z_dry))*dV)**(1./3.)
                    if i>0:
                        couv_dry_A[i]   = func_proj_A(idx_x_dry,idx_y_dry,dA)
                        couv_dry_h[i]   = (max(idx_z_dry)-min(idx_z_dry)+1)*dz
                    else:
                        couv_dry_A[i]   = n_x*n_y*dA
                        couv_dry_h[i]   = n_z*dz

                #print('cluster ',i,iiii,'time spent dry stuff:',ttiimmee.time()-time_tmp); iiii += 1; time_tmp = ttiimmee.time() 




                idx_z_wet = idx_z[ql_tmp>=ql_min] 
                idx_x_wet = idx_x[ql_tmp>=ql_min] 
                idx_y_wet = idx_y[ql_tmp>=ql_min] 

                if idx_z_wet.size>0:


                    couv_wet_w[i]   = np.mean(w[idx_z_wet,idx_x_wet,idx_y_wet])
                    couv_wet_V[i]   = (float(len(idx_z_wet))*dV)**(1./3.)
                    couv_wet_A[i]   = func_proj_A(idx_x_wet,idx_y_wet,dA)
                    couv_wet_h[i]   = (max(idx_z_wet)-min(idx_z_wet)+1)*dz

                    if couv_wet_V[i]>couv_V[i]:
                        print('wtf is happening',couv_wet_V[i],couv_V[i])

                    #print('cluster ',i,iiii,'time spent on wet stuff:',ttiimmee.time()-time_tmp); iiii += 1; time_tmp = ttiimmee.time() 

                    #I try to get the cf by calculating the Area profiles of both dry and wet and the ratio
                    #actually, it might be best to calculate that anyway
                    #I have a feeling this will throw up some weird shit when 
                    tmp1,tmpwet =func_vert_mean_sorted(idx_z_wet,idx_x_wet,idx_y_wet,w)
                    couv_wet_prof_A[i,:] = tmpwet*dA
                    tmp1,tmpdry =func_vert_mean_sorted(idx_z_dry,idx_x_dry,idx_y_dry,w)
                    couv_dry_prof_A[i,:] = tmpdry*dA



                    couv_prof_cf[i,:] = tmpwet/(tmpdry+tmpwet)
                    #print('cluster ',i,iiii,'time spent on A wet and dry profs:',ttiimmee.time()-time_tmp); iiii += 1; time_tmp = ttiimmee.time() 

                else:

                    couv_prof_cf[i,:] = couv_prof_A[i,:]*0.0




            couv_V_all = np.hstack([couv_V_all,couv_V])
            couv_w_all = np.hstack([couv_w_all,couv_w])
            couv_A_all = np.hstack([couv_A_all,couv_A])
            couv_h_all = np.hstack([couv_h_all,couv_h])


            couv_max_cf_all = np.hstack([couv_max_cf_all,couv_max_cf])
            couv_base_all = np.hstack([couv_base_all,couv_base])
            couv_t_all.extend(couv_t)
            couv_x_max_all.extend(couv_x_max)
            couv_y_max_all.extend(couv_y_max)
            couv_z_max_all.extend(couv_z_max)

            couv_wet_V_all = np.hstack([couv_wet_V_all,couv_wet_V])
            couv_wet_w_all = np.hstack([couv_wet_w_all,couv_wet_w])
            couv_wet_A_all = np.hstack([couv_wet_A_all,couv_wet_A])
            couv_wet_h_all = np.hstack([couv_wet_h_all,couv_wet_h])

            couv_dry_V_all = np.hstack([couv_dry_V_all,couv_dry_V])
            couv_dry_w_all = np.hstack([couv_dry_w_all,couv_dry_w])
            couv_dry_A_all = np.hstack([couv_dry_A_all,couv_dry_A])
            couv_dry_h_all = np.hstack([couv_dry_h_all,couv_dry_h])

            couv_prof_w_all = np.vstack([couv_prof_w_all,couv_prof_w])
            couv_prof_A_all = np.vstack([couv_prof_A_all,couv_prof_A])
            couv_prof_ql_all = np.vstack([couv_prof_ql_all,couv_prof_ql])
            couv_prof_qv_all = np.vstack([couv_prof_qv_all,couv_prof_qv])
            couv_prof_cf_all = np.vstack([couv_prof_cf_all,couv_prof_cf])
            couv_prof_couv_all = np.vstack([couv_prof_couv_all,couv_prof_couv])


            couv_prof_flux_w_all  = np.vstack([couv_prof_flux_w_all ,couv_prof_flux_w ])
            couv_prof_flux_qt_all = np.vstack([couv_prof_flux_qt_all,couv_prof_flux_qt ])
            couv_prof_total_flux_qt_all = np.vstack([couv_prof_total_flux_qt_all,couv_prof_total_flux_qt ])

            couv_prof_fluc_qt_all = np.vstack([couv_prof_fluc_qt_all,couv_prof_fluc_qt ])

            couv_dry_prof_A_all = np.vstack([couv_dry_prof_A_all,couv_dry_prof_A])
            couv_wet_prof_A_all = np.vstack([couv_wet_prof_A_all,couv_wet_prof_A])



            time2 = ttiimmee.time()
            print('time needed to calculate cluster properties for timestep ',t,' in minutes:',(time2-time1)/60)

    #Now calculate clus radius and area square root. 
    couv_sqA_all   = np.sqrt(couv_A_all)
    couv_rad_all   = np.sqrt(couv_A_all/math.pi)
    #height averaged area, aka sqrt(V/h) 
    couv_V_h_all   = np.sqrt(couv_V_all**3./couv_h_all)



    file_w.close()
    file_ql.close()
    file_qt.close()
    file_c.close() 
    file_prof.close()



    #saving as panda
    data_for_panda = list(zip(couv_V_all,couv_sqA_all,couv_rad_all,couv_A_all,couv_h_all,couv_V_h_all,
                              couv_w_all,
                              couv_prof_w_all,couv_prof_ql_all,couv_prof_qv_all,couv_prof_A_all,couv_prof_couv_all,
                              couv_wet_h_all,couv_wet_A_all,couv_wet_V_all,couv_wet_w_all,
                              couv_dry_h_all,couv_dry_A_all,couv_dry_V_all,couv_dry_w_all,
                              couv_t_all,couv_x_max_all,couv_y_max_all,couv_z_max_all,couv_base_all,couv_max_cf_all,
                              couv_prof_flux_w_all,couv_prof_flux_qt_all,couv_prof_total_flux_qt_all,
                              couv_prof_fluc_qt_all))

    # #Got distracted, started cleaning this up as well    
    col_names = ['Volume','sq Area','Radius','Area','height','V_h',
                 'w',
                 'w profile','ql profile','qv profile','Area profile','couv profile',
                 'wet h','wet A','wet V','wet w',            
                 'dry h','dry A','dry V','dry w',            
                 'time','x','y','z','base','z max cf',
                 'w flux','qt flux','qt total flux',
                 'qt fluc']  


    df = pd.DataFrame(data = data_for_panda, columns=col_names)
    df.to_pickle(filename_out)
    print('saved clus properties as panda in ',filename_out, 'after ',(ttiimmee.time()-time0)/60,' minutes')
    return





