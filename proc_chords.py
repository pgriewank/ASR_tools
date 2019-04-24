#Contains the functions needed to process both chords and regularized beards
#proc_chords is used for chords
#proc_beard_regularize for generating beards
#Both have a large overlap, but I split them in two to keep the one script from getting to confusing. 

import numpy as np
import math
from netCDF4 import Dataset
import os
import time as ttiimmee
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import sys
sys.path.insert(0, "/home/pgriewank/code/2019-chords-plumes/")


from unionfind import UnionFind
from cusize_functions import *
import matplotlib.pyplot as plt
import pandas as pd
import gc


#turned into a function
#removed the possibility to loop over multiple dates, if you want to do that call the function repeatedly 
#Full list of variables to analyze is unclear, I will try to include everything available, but this might break the memory bank
#want to keep the automatic x and y calculation
#Scaling shouldn't be needed, as all chord properties should be indepenent of wind direction (right?)
#Similarly, no basedefinition is needed, all values are relative to cloud base

#Should be able to work for any variable in the column output, or for any 3D variable as long as it is named the same as the file. 
#Changing 3D output
#Default is now to always go over x and y directions

#TODO
#plot_flag disabled for the mean time

def proc_chords(          date_str='20160611',
                          directory_input='/data/testbed/lasso/sims/',
                          directory_output='/data/testbed/lasso/chords/',
                          data_dim_flag=1,
                          base_percentile = 25,
                          special_name='',
                          chord_times = 0,
                          N_it_min=0,
                          N_it_max=1e9):

    # plot_curtains_flag: 0 nothing, 1 plots pre regularization plots, currently dissabled
    # data_dim_flag: 1 = column, 3 = 3D snapshot
    # chord_times: 0 use Neils values, use values that fit model output exactly with not gap possible
    # directory_input         = '/data/testbed/lasso/sims/' #+date
    # N_it_max = maximum number of iterables, 3D timesteps or column files. Used for testing things quickly
    # N_it_min = start number of iterables, 3D timesteps or column files. Only reall makes sense for 3D to avoid some weird initial fields. 
    
    

    dz = 25.0 #39.0625 #should be overwritten after the profile data is loaded
    dx = 25.0

    date = date_str

    n_percentiles = 7 #Number of percentiles
    percentiles = np.array([5,10,35,50,65,90,95])

    
    #1D clustering parameters in seconds, taken to agree with Lareau
    if chord_times == 0:
        t_gap = 20
        t_min = 30
        t_max = 1200*100 #Made a 100 times longer
        cell_min = 3 #Minimal number of cells needed per chord

    # #1D clustering parameters, 
    #set super strict, but goes on for a loooong time as well
    if chord_times == 1:
        t_gap = 0. #should be pretty strict, no gaps allowed!
        t_min = 0.0
        t_max = 1e9
        cell_min = 3 #Minimal number of cells needed per chord
        
   


    ql_min = 1e-5 #value used to determine existence of cloud
    z_min  = 10 #Index of minimum  z_vlvl of the cbl



 
    
    print('looking into date: ',date)
    if data_dim_flag==1:
        filename_column = []
        if date_str=='bomex':
            filename_column.append(directory_input+date+'/bomex.column.00512.00512.0000000.nc')
        else:
            #filename_column.append(directory_input+date+'/testbed.column.00000.00000.0000000.nc')
            #filename_column.append(directory_input+date+'/testbed.column.00000.00512.0000000.nc')
            #filename_column.append(directory_input+date+'/testbed.column.00512.00000.0000000.nc')
            filename_column.append(directory_input+date+'/testbed.column.00512.00512.0000000.nc')
    if data_dim_flag==3:
        filename_w   = directory_input+date+'/w.nc'
        filename_l   = directory_input+date+'/ql.nc'
        filename_qt  = directory_input+date+'/qt.nc'
        filename_thl = directory_input+date+'/thl.nc'
        file_w       = Dataset(filename_w,read='r')
        file_ql      = Dataset(filename_l,read='r')
        file_thl     = Dataset(filename_thl,read='r')
        file_qt      = Dataset(filename_qt,read='r')
        [nz, nx, ny] = get_zxy_dimension(filename_l,'ql')
        
        
#         #getting variable to be regularized
#         filename_var = directory_input+date+'/'+reg_var+'.nc'
#         file_var     = Dataset(filename_var,read='r')





    filename_prof=directory_input+date+'/testbed.default.0000000.nc'
    
    if date=='bomex':
        filename_prof=directory_input+date+'/bomex.default.0000000.nc'
        

    file_prof  =  Dataset(filename_prof,read='r')


    
    

    n_chords = 0
    #I will try lists first, which I will then convert to arrays in the end before saving in pandas
    chord_timesteps           = []
    chord_length              = []
    chord_duration            = []
    chord_time                = []
    chord_height              = [] #percentile of cloud base
    chord_w                   = []
    chord_w_up                = [] #mean over updrafts
    chord_w_base              = []
    chord_w_flux              = [] #Sum of w below
    chord_thl_anom            = []
    chord_qt_anom             = []
    #Coming next
    chord_w_per               = np.zeros([0,n_percentiles])
    chord_w_per_up            = np.zeros([0,n_percentiles])
    


    #This now a bit trickier then for the 3D version. Will have to calculate a vector for the lower time resolution of the profile,
    #Then latter apply the nearest value to the full 1d time vec
    #First loading surface variables from default profile

    print('calculating cbl height from profile file')
    T = file_prof['thl'][:,0]
    p = file_prof['p'][:,0]*0.0+99709
    qt = file_prof['qt'][:,0]
    w2 = file_prof['w2'][:,:]
    thl_prof = file_prof['thl'][:,:]
    qt_prof = file_prof['qt'][:,:]
    nz_prof = w2.shape[1]
    z_prof = file_prof['z'][:]
    dz = z_prof[1]-z_prof[0]
    print('dz: ',dz)
    
    
    time_prof = file_prof['time'][:]
    cbl_1d_prof = time_prof*0.0

    #Hack together the Lifting condensation level LCL
    qt_pressure = p*qt
    sat_qv = 6.112*100 * np.exp(17.67 * (T - 273.15) / (T - 29.65 ))
    #rel_hum = np.asmatrix(qt_pressure/sat_qv)[0]
    rel_hum = qt_pressure/sat_qv
    #Dewpoint
    A = 17.27
    B = 237.7
    alpha = ((A * (T- 273.15)) / (B + (T-273.15)))
    alpha = alpha + np.log(rel_hum)
    dewpoint = (B * alpha) / (A - alpha)
    dewpoint = dewpoint + 273.15
    LCL = 125.*(T-dewpoint)
    LCL_index = np.floor(LCL/dz)

    #now calculate the cbl top for each profile time
    for tt in range(len(time_prof)):
        w_var = 1.0
        z=z_min
        while w_var > 0.08:
            z += 1
            w_var = w2[tt,z]
            #w_var = np.var(w_1d[z,:])

        #Mimimum of LCL +100 or variance plus 300 m 
        cbl_1d_prof[tt] = min(z+300/dz,LCL_index[tt])
        #To avoid issues later on I set the maximum cbl height to 60 % of the domain height, but spit out a warning if it happens
        if cbl_1d_prof[tt]>0.6*nz_prof:
            print('warning, cbl height heigher than 0.6 domain height, could crash regularization later on, timestep: ',tt)
            cbl_1d_prof[tt] = math.floor(nz*0.6)
    print('resulting indexes of cbl over time: ',cbl_1d_prof)
    print('calculated LCL: ',LCL_index)


    #Now we either iterate over columns or timesteps
    if data_dim_flag==1:
        n_iter =len(filename_column)
    if data_dim_flag==3:
        n_iter =len(time_prof)





    #for col in filename_column:
    n_iter = min(n_iter,N_it_max)
    for it in range(N_it_min,n_iter):
    #for it in range(14,16):
    #for it in range(16,17):
        print('n_chords: ',n_chords)
        

        time1 = ttiimmee.time()
        if data_dim_flag ==1:
            print('loading column: ',filename_column[it])
            file_col = Dataset(filename_column[it],read='r')

            w_2d = file_col.variables['w'][:]
            w_2d = w_2d.transpose()
            ql_2d = file_col.variables['ql'][:]
            ql_2d = ql_2d.transpose()
            t_1d = file_col.variables['time'][:]
            print('t_1d',t_1d)
            thl_2d = file_col.variables['thl'][:]
            thl_2d = thl_2d.transpose()
            qt_2d  = file_col.variables['qt'][:]
            qt_2d  = qt_2d.transpose()
            u_2d  = file_col.variables['u'][:]
            u_2d  = u_2d.transpose()
            v_2d  = file_col.variables['v'][:]
            v_2d  = v_2d.transpose()
            
            
            #The needed cbl height
            cbl_1d = t_1d*0

            #Now we go through profile time snapshots and allocate the closest full time values to the profile values
            dt_2 = (time_prof[1]-time_prof[0])/2
            for tt in range(len(time_prof)):
                cbl_1d[abs(t_1d-time_prof[tt])<dt_2] = cbl_1d_prof[tt]
                
                
                
            #to get anomalies of thl and qt we subtract the closet mean profile
            for tt in range(len(time_prof)):
                
                #globals().update(locals())                
                tmp_matrix =  thl_2d[:,abs(t_1d-time_prof[tt])<dt_2]
                tmp_vector =  thl_prof[tt,:]
                #because the vectors don't perfectly align
                thl_2d[:,abs(t_1d-time_prof[tt])<dt_2] = (tmp_matrix.transpose() - tmp_vector).transpose()

                tmp_matrix =  qt_2d[:,abs(t_1d-time_prof[tt])<dt_2]
                tmp_vector =  qt_prof[tt,:]
                #because the vectors don't perfectly align
                qt_2d[:,abs(t_1d-time_prof[tt])<dt_2] = (tmp_matrix.transpose() - tmp_vector).transpose()

                # = var_2d[:,abs(t_1d-time_prof[tt])<dt_2]-var_prof[tt,:]
                
                


        if data_dim_flag ==3:


            if sum(file_prof['ql'][it,:])>0.0:

                print('loading timestep: ',it)

                ql_3d   = grab_3d_field(file_ql  ,it,'ql')
                w_3d    = grab_3d_field(file_w   ,it,'w')
                qt_3d   = grab_3d_field(file_qt  ,it,'qt')
                thl_3d  = grab_3d_field(file_thl  ,it,'thl')

                #Here we have to do all the fuckery to turn the 3D fields into 2d slices with an imaginary time vector
                w_2d   = np.array(w_3d.reshape((nz,nx*ny)))
                ql_2d  = np.array(ql_3d.reshape((nz,nx*ny)))
                qt_2d  = np.array(qt_3d.reshape((nz,nx*ny)))
                thl_2d = np.array(thl_3d.reshape((nz,nx*ny)))
                
                #Now we do the same thing with the transposed field, use to be an either or, now just add it on
                w_3d   = np.transpose( w_3d,  (0, 2, 1))
                ql_3d  = np.transpose(ql_3d,  (0, 2, 1))
                qt_3d  = np.transpose(qt_3d,  (0, 2, 1))
                thl_3d = np.transpose(thl_3d, (0, 2, 1))
                
                w_2d     = np.hstack([w_2d   ,np.array(w_3d.reshape((nz,nx*ny)))])
                ql_2d    = np.hstack([ql_2d  ,np.array(ql_3d.reshape((nz,nx*ny)))])
                thl_2d   = np.hstack([thl_2d ,np.array(thl_3d.reshape((nz,nx*ny)))])
                qt_2d    = np.hstack([qt_2d  ,np.array(qt_3d.reshape((nz,nx*ny)))])
                
                
                
                
                #Should now be able to delete 3d fields as they aren't needed anymore, not sure if that helps save any memory though
                del w_3d
                del ql_3d
                del thl_3d
                del qt_3d
                #hopefully this helps
                gc.collect()
                
                #Getting anomalies of thl and qt
                qt_2d[:,:]  = (qt_2d.transpose()  - qt_prof[it,:]).transpose()
                thl_2d[:,:] = (thl_2d.transpose() - thl_prof[it,:]).transpose()
                
                
                #to get the fake time vector we load the wind from the profile data, which devided by the grid spacing gives us a fake time resolution
                #we use the calculated cbl+300 meter or lcl as reference height 
                ref_lvl = cbl_1d_prof[tt]

                u_ref = file_prof['u'][it,ref_lvl]
                v_ref = file_prof['v'][it,ref_lvl]

                V_ref = np.sqrt(u_ref**2+v_ref**2) 

                time_resolution = dx/V_ref
                #if time_resolution > t_gap:
                #    print('t_gap too small:', t_gap)
                #    t_gap = time_resolution*1.5
                #    print('changed t_gap to:', t_gap)

                print('time iterative, V_ref, time_resolution',it, str(V_ref)[:4], str(time_resolution)[:4] )
                #fake t vector, 
                t_1d = np.linspace(0,2*nx*ny*time_resolution,2*nx*ny)#+nx*ny*time_resolution*it
                #dt_1d   = t_1d*0
                #dt_1d[1:] = t_1d[1:]-t_1d[:-1]   
                



            else:
                #If no clouds are present we pass a very short empty fields over to the chord searcher
                print('skipping timestep: ',it,' cause no clouds')
                ql_2d    = np.zeros((nz,1))
                w_2d     = np.zeros((nz,1))
                thl_2d   = np.zeros((nz,1))
                qt_2d    = np.zeros((nz,1))
                t_1d     = np.zeros(1)

            #The needed cbl height, which constant everywhere
            cbl_1d = t_1d*0
            cbl_1d[:] = cbl_1d_prof[it]






        time2 = ttiimmee.time()
        print('loading time:',(time2-time1)*1.0,)



        ### Detecting lowest cloud cell is within 300 m of CBL

        nt = len(cbl_1d)
        cl_base = np.zeros(nt)

        #Detecting all cloudy cells
        #Use to have a different method using nans that doesn:t work anymore somehow. Now I just set it really high where there is no cloud. 
        for t in range(nt):
            if np.max(ql_2d[:,t])>ql_min :
                cl_base[t]=np.argmax(ql_2d[:,t]>1e-6)
            else:
                cl_base[t]=10000000

        cl_base=cl_base.astype(int)

        #Now find c base lower than the max height
        cbl_cl_idx = np.where((cl_base-cbl_1d[:nt])*dz<0)[0]

        cbl_cl_binary = cl_base*0
        cbl_cl_binary[cbl_cl_idx]=1
        t_cbl_cl=t_1d[cbl_cl_idx]


        ### Clustering 1D

        #Now we simply go through all cloudy timesteps and detect chords
        #If they fulful chord time requirements and have a number of values which fulfills cell_min they are counted as a chord
        #and their properties are calculatted immediately
        t_cloudy_idx = 0
        #n_chords = 0
        chord_idx_list = []

        print('iterating through step ',it,'which contains ',len(cbl_cl_idx),'cloudy columns')


        chord_idx_list = []

        while t_cloudy_idx < len(cbl_cl_idx)-1:# and n_curtain<100*it:      ####################################GO HERE TO SET MAXIMUM CURTAIN
            #print(t_chord_begin)
            t_chord_begin = t_cloudy_idx
            #now connecting all cloudy indexes
            #Originally only cared if they fulfilled cloud criteria, but now I also hard coded that neighboring cells always count
            ##Check if the index of the next cloudy cell is the same as the next index in total, if so the cells are connected
            #if cbl_cl_idx[t_cloudy_idx+1]==cbl_cl_idx[t_cloudy_idx]+1:
                
            
            while t_cloudy_idx < len(cbl_cl_idx)-1 and (cbl_cl_idx[t_cloudy_idx+1]==cbl_cl_idx[t_cloudy_idx]+1 or t_cbl_cl[t_cloudy_idx+1]-t_cbl_cl[t_cloudy_idx]<t_gap):
                t_cloudy_idx += 1 
            t_chord_end = t_cloudy_idx
            #print('t_chord_end',t_chord_end)


            #Checking if it fulfils chord criteria regaring time
            #we also added a minimum height of 100 m to screen out fog/dew stuff at the surface
            if t_chord_end-t_chord_begin>cell_min:
                chord_z_min = np.min(cl_base[cbl_cl_idx[t_chord_begin:t_chord_end]])
                ch_duration = t_cbl_cl[t_chord_end]-t_cbl_cl[t_chord_begin]
            else:
                chord_z_min = 0
                ch_duration = 0
            if ch_duration>t_min and ch_duration<t_max and chord_z_min > 4:


                if t_chord_end-t_chord_begin>cell_min-1:
                    n_chords += 1
                    
                    #Getting the chord beginning and end
                    idx_beg_chord = cbl_cl_idx[t_chord_begin]
                    idx_end_chord = cbl_cl_idx[t_chord_end]
                    time_beg_chord = t_1d[idx_beg_chord]
                    time_end_chord = t_1d[idx_end_chord]
                    #chord_idx_list.append(list(cbl_cl_idx[t_chord_begin:t_chord_end]))
                    #list of relevant chord indexes
                    ch_idx_l = list(cbl_cl_idx[t_chord_begin:t_chord_end])
                    
                    
                    #getting V_ref if data_dim_flag==1. Is calculated directly from the cloud base speeds
                    if data_dim_flag==1:
                        u_ref=np.mean(u_2d[cl_base[ch_idx_l],ch_idx_l])
                        v_ref=np.mean(v_2d[cl_base[ch_idx_l],ch_idx_l])
                        V_ref=np.sqrt(u_ref**2+v_ref**2) 
                     ### Now appending chord properties
                
                    chord_timesteps.append(t_chord_end-t_chord_begin)
                    chord_duration.append(ch_duration)
                    chord_length.append(ch_duration*V_ref)
                    chord_height.append(np.percentile(cl_base[ch_idx_l],base_percentile)*dz) #25th percentile of cloud base
                    chord_w_base.append(np.mean(w_2d[cl_base[ch_idx_l],ch_idx_l]))
                    chord_w.append(np.mean(w_2d[cl_base[ch_idx_l]-1,ch_idx_l]))
                    chord_w_flux.append(np.sum(w_2d[cl_base[ch_idx_l]-1,ch_idx_l]))
                    chord_thl_anom.append(np.mean(thl_2d[cl_base[ch_idx_l]-1,ch_idx_l]))
                    chord_qt_anom.append(np.mean(qt_2d[cl_base[ch_idx_l]-1,ch_idx_l]))

                    w_base_vec = w_2d[cl_base[ch_idx_l]-1,ch_idx_l]
                    chord_w_up.append(np.mean(w_base_vec[w_base_vec>0.0]))
                    tmp_w_per    = np.percentile(w_base_vec,percentiles)
                    
                    if len(w_base_vec[w_base_vec>0.0])>0:
                        tmp_w_per_up = np.percentile(w_base_vec[w_base_vec>0.0],percentiles)
                    else:
                        tmp_w_per_up = np.zeros(n_percentiles)
                        tmp_w_per_up[:] = 'nan'
                        
                    
                    
                    chord_w_per               = np.vstack([chord_w_per,tmp_w_per])
                    chord_w_per_up            = np.vstack([chord_w_per,tmp_w_per_up])


                    if data_dim_flag==1:
                        chord_time.append(np.mean(t_1d[ch_idx_l]))
                    if data_dim_flag==3:
                        chord_time.append(time_prof[it])


            t_cloudy_idx += 1
        time3 = ttiimmee.time()
        
        print('iterable: ',it)
        print('n_chords: ',n_chords)
        print('number of time points included: ',len(cbl_cl_idx))

       
                
            

       
    
    
    #Does it matter if I turn these from lists to arrays? Fuck it, will do it anyway
    
    chord_timesteps=np.asarray(chord_timesteps)
    chord_duration =np.asarray(chord_duration)
    chord_length   =np.asarray(chord_length)
    chord_height   =np.asarray(chord_height)
    chord_w_base   =np.asarray(chord_w_base)
    chord_w        =np.asarray(chord_w)
    chord_w_up     =np.asarray(chord_w_up)
    chord_w_flux   =np.asarray(chord_w_flux)
    chord_thl_anom =np.asarray(chord_thl_anom)
    chord_qt_anom  =np.asarray(chord_qt_anom)
    chord_time     =np.asarray(chord_time)
    
    
    #Saving
    
    print('all chords: ',len(chord_duration))
        
    save_string_base = 'chord_prop_'+date+'_d'+str(data_dim_flag)+'_ct'+str(chord_times)+'_'+special_name+'_N'+str(n_chords)

                   
    filename_chord_panda = directory_output+save_string_base+'.pkl'
    
    
    data_for_panda = list(zip(chord_timesteps,chord_duration,chord_length,chord_height,chord_w_base,chord_w,chord_w_flux,chord_thl_anom,chord_qt_anom,chord_time,chord_w_up,chord_w_per,chord_w_per_up))
    df = pd.DataFrame(data = data_for_panda, columns=['timesteps','duration','length','height','w_base','w','w_flux','thl_anom','qt_anom','time','w up','w per','w per up'])
    df.to_pickle(filename_chord_panda)
    print('chordlength properties saved as panda in ',filename_chord_panda)
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')

def func_curtain_reg(input_2d_field):
    #function regularizes to cloud base
    #2019-03-20: added smoother to hopefully avoid impact of harsch jumps
    #2019-03-28: Added simplified version for base_smoothing_flag == 2 which gets rid of 1D pre interpolation 
    #I originally used interp2d, tried griddata but it was a lot slower
    
    #Calculating the regularized t axis but for original resolution
    #It is expected to go a bit beyond -1.5 and 1.5, total width defined by curtain_extra
    #takes the original time vector, subtracts it by mean time, then scales it by 1/(time_end_chord-time_beg_chord)
    t_reg_orig = t_1d[idx_beg_curtain:idx_end_curtain]-(time_beg_chord+time_end_chord)/2.
    t_reg_orig = t_reg_orig/(time_end_chord-time_beg_chord)

    #Now we calculate the new regularized grid with the correct vertical but low/original horizontal/time resolution
    #mesh_t_low_z_high_x,mesh_t_low_z_high_z = np.meshgrid(t_reg_orig,z_reg_mid) #seems not to be needed
    var_t_low_z_high = np.zeros([curtain_cells,n_z_reg])


    #introducing z_idx_base vector
    #Assigning reference cloud base where no cloud present
    z_idx_base=cl_base*1.0+0.0
    z_idx_base[:] = z_idx_base_default
    for i in range(idx_beg_chord,idx_end_chord):
        if i>idx_beg_chord-1 and i<idx_end_chord and cl_base[i]<cbl_1d[i]:
                z_idx_base[i] = cl_base[i]
            
    

    #Here the smoother comes into play:
    #We started with a simple 5 cell running mean, 
    #But now we are making it a function of the chordlength, using a 0.1 running mean
    
    if base_smoothing_flag ==1:
        z_idx_base_smooth = z_idx_base*1.0
        N = int(np.floor(idx_end_chord-idx_beg_chord)*0.1)
        for i in range(idx_beg_chord-N,idx_end_chord+N):
            z_idx_base_smooth[i] = sum(z_idx_base[i-N:i+N])/(2*N)
        z_idx_base[:] = z_idx_base_smooth[:]
        
    if base_smoothing_flag==2:
        #just put the percentile back

        z_idx_base[:] = z_idx_base_default
        
        
    
    
        
        
    #default version for variable base height    
    if base_smoothing_flag<2:
        #Now for each of the columns of the original curtain a vertical interpolation is done
        for i in range(idx_beg_curtain,idx_end_curtain):

            #assigining column value

            var_orig_col = input_2d_field[:,i]





            #Regularizing the z axes so that cloud base is at 1
            d_z_tmp = 1.0/z_idx_base[i]
            nz = var_orig_col.shape[0]
            z_reg_orig_top = d_z_tmp*nz- d_z_tmp/2
            z_reg_orig = np.linspace(0+d_z_tmp/2,z_reg_orig_top,nz)

            #HAve to add 0  to the z_reg_orig to enable interpolation 
            z_reg_orig = np.hstack([[0],z_reg_orig])
            var_orig_col   = np.hstack([var_orig_col[0],var_orig_col])


            #1D vertical interpolation to get the right columns and asign them one by one to w_x_low_z_high
            f = interp1d(z_reg_orig, var_orig_col, kind='next')
            try:

                var_reg_inter = f(z_reg_mid)
            except:
                print(z_idx_base[i])
                #plt.plot(z_idx_base[c])
                print(z_reg_orig)
                print(z_reg_mid)

            var_t_low_z_high[i-idx_beg_curtain,:] = var_reg_inter

        #Now that w_x_low_z_high we have to interpolate 2D onto the rull regularized grid
        #print(t_reg_orig.shape,z_reg_mid.shape)
        f = interp2d(t_reg_orig, z_reg_mid, var_t_low_z_high.transpose(), kind='linear')
        var_curtain = f(t_reg_mid,z_reg_mid)
    
    
    #constant base height version
    if base_smoothing_flag==2:
        #Regularizing the z axes so that cloud base is at 1, since z_idx_base is the same everywhere I just use idx_beg_curtain as one. 
        i=idx_beg_curtain
        d_z_tmp = 1.0/z_idx_base[i]
        var_orig_2d = input_2d_field[:,idx_beg_curtain:idx_end_curtain]

        nz = var_orig_2d.shape[0]
        z_reg_orig_top = d_z_tmp*nz- d_z_tmp/2
        z_reg_orig = np.linspace(0+d_z_tmp/2,z_reg_orig_top,nz)
        
        #Have to add 0  to the z_reg_orig to enable interpolation 
        z_reg_orig    = np.hstack([[0],z_reg_orig])
        var_orig_2d   = np.vstack([var_orig_2d[0,:],var_orig_2d])

        
        f = interp2d(t_reg_orig, z_reg_orig,var_orig_2d, kind='linear')
        var_curtain = f(t_reg_mid,z_reg_mid)

        
    
    
    
    
    
    return var_curtain




#turned into a function
#removed the possibility to loop over multiple dates, if you want to do that call the function repeatedly 
#Should be able to work for any variable in the column output, or for any 3D variable as long as it is named the same as the file. 
#Changing 3D output
#Default is now to always go over x and y directions
#If scaling is applied 

#TODO
#plot_flag disabled for the mean time

def proc_beard_regularize(reg_var = 'w',
                          date_str='20160611',
                          directory_input='/data/testbed/lasso/sims/',
                          data_dim_flag=1,
                          base_smoothing_flag=2,
                          plot_curtains_flag = 0,
                          base_percentile = 25,
                          special_name='',
                          scale_flag=0,
                          chord_times = 0,
                          anomaly_flag = 0,
                          N_it_max=1e9,
                          N_it_min=0,
                          size_bin_flag=0,
                          N_bins=12,
                          bin_size = 250,
                          curtain_extra = 1.0
                         ):

    # reg_var = variable that will be regularized
    # plot_curtains_flag: 0 nothing, 1 plots pre and post regularization plots of reg_var
    # data_dim_flag: 1 = column, 3 = 3D snapshot
    # time_slice_curtain: 0 only puts out the total sums, 1: adds a seperate output for each time slice, is needed for scale_flag
    # scale_flag: If 0, nothing, if 1, it scales the output by u/sqrt(u^2+v^2) and flips the vector if u>0. Is set to 0 if data_dim_flag==1
    #
    # base_smoothing_flag: 0 use mix of percentile and cloud base as done my Neil, 1: smooth out base after setting it with running average 2: just use percentile defined by base_percentile
    # base_percentile: percentile used to find chordlength bottom
    # chord_times: 0 use Neils values, use values that fit model output exactly with not gap possible
    # anomaly_flag: 0 use reg_var as it is. 1 use reg_var - profile. Works easiest for 3d output, 1d_flag needs to use the closet mean profile
    # directory_input         = '/data/testbed/lasso/sims/' #+date
    # N_it_max = maximum number of iterables, 3D timesteps or column files. Used for testing things quickly
    # size_bin_flag bins the beards by their chord_lenth. Currently using 8 bins of 250 meters length to get started. The lowest bin should be empty, because we only calculate curtains when at least curtain_min is used
    # curtain_extra: Regularized chord length before and after in the curtain, default is 1
    
    directory_output = 'data_curtains/'


    dz = 25.0 #39.0625 #Is recalculated from the profile file later on
    dx = 25.0

    date = date_str

    #1D clustering parameters in seconds, taken to agree with Lareau
    if chord_times == 0:
        t_gap = 20
        t_min = 30
        t_max = 120000
        cell_min = 3 #Minimal number of cells needed per chord
        curtain_min = 10 #Minimal number of cells needed to convert into a curtain

    # #1D clustering parameters, 
    #set super strict 
    if chord_times == 1:
        t_gap = 0.#should be pretty strict gaps allowed!
        t_min = 0
        t_max = 1e9
        cell_min = 10 #Minimal number of cells needed per chord
        curtain_min = 10 #Minimal number of cells needed per curtain


    #value used to determine existence of cloud
    ql_min = 1e-5

    z_min  = 10 #Index of minimum  z_vlvl of the cbl


    #Flag clean up
    if data_dim_flag==1:
        scale_flag=0


    

    
    
    #Creating dictionary to save all properties
    settings_dict = {
    'reg_var': reg_var,
    'date_str':date_str,
    'directory_input':directory_input,
    'data_dim_flag':data_dim_flag,
    'base_smoothing_flag':base_smoothing_flag,
    'plot_curtains_flag' :plot_curtains_flag,
    'base_percentile':base_percentile,
    'special_name':special_name,
    'scale_flag':scale_flag,
    'chord_times':chord_times,
    'anomaly_flag':anomaly_flag,
    'N_it_max':N_it_max,
    'N_it_min':N_it_min,
    'size_bin_flag':size_bin_flag,
    'bin_size':bin_size,
    'N_bins':N_bins,
    'curtain_extra':curtain_extra
    }
    
    
    
    
    #Creating regularized grid.
    d_reg = 0.005
    n_z_reg = int(1.5/d_reg)
    n_t_reg = int((1+2*curtain_extra)/d_reg)


    t_reg_bound      = np.linspace(-0.5-curtain_extra,0.5+curtain_extra ,n_t_reg+1)
    t_reg_mid        = np.linspace(-0.5-curtain_extra+d_reg/2,0.5+curtain_extra-d_reg/2  ,n_t_reg)
    z_reg_bound      = np.linspace(0,1.5                     ,n_z_reg+1)
    z_reg_mid        = np.linspace(0+d_reg/2,1.5-d_reg/2     ,n_z_reg)

    mesh_curtain_t,mesh_curtain_z = np.meshgrid(t_reg_mid,z_reg_mid)
    var_curtain     = np.zeros([n_t_reg,n_z_reg])
    var_curtain_sum = np.zeros([n_t_reg,n_z_reg])
    
    var_curtain_up_sum = np.zeros([n_t_reg,n_z_reg])
    var_curtain_dw_sum = np.zeros([n_t_reg,n_z_reg])

    n_curtain        = 0
    n_curtain_up     = 0
    n_curtain_dw     = 0

    if size_bin_flag==1:
        N_bins = 12
        n_curtain_bin          = np.zeros([N_bins])
        n_curtain_bin_up       = np.zeros([N_bins])
        n_curtain_bin_dw       = np.zeros([N_bins])
        var_curtain_bin_sum    = np.zeros([N_bins,n_t_reg,n_z_reg])
        var_curtain_bin_up_sum = np.zeros([N_bins,n_t_reg,n_z_reg])
        var_curtain_bin_dw_sum = np.zeros([N_bins,n_t_reg,n_z_reg])
        
        mid_bin_size = np.linspace(125,-125+N_bins*250,N_bins)
        print('mid_bin_size',mid_bin_size)



    
    print('looking into date: ',date)
    if data_dim_flag==1:
        filename_column = []
        if date_str=='bomex':
            filename_column.append(directory_input+date+'/bomex.column.00512.00512.0000000.nc')
        else:
            #filename_column.append(directory_input+date+'/testbed.column.00000.00000.0000000.nc')
            #filename_column.append(directory_input+date+'/testbed.column.00000.00512.0000000.nc')
            #filename_column.append(directory_input+date+'/testbed.column.00512.00000.0000000.nc')
            filename_column.append(directory_input+date+'/testbed.column.00512.00512.0000000.nc')
    if data_dim_flag==3:
        filename_w   = directory_input+date+'/w.nc'
        filename_l   = directory_input+date+'/ql.nc'
        file_w       = Dataset(filename_w,read='r')
        file_ql      = Dataset(filename_l,read='r')
        [nz, nx, ny] = get_zxy_dimension(filename_l,'ql')
        
        
        #getting variable to be regularized
        filename_var = directory_input+date+'/'+reg_var+'.nc'
        file_var     = Dataset(filename_var,read='r')





    filename_prof=directory_input+date+'/testbed.default.0000000.nc'
    
    if date=='bomex':
        filename_prof=directory_input+date+'/bomex.default.0000000.nc'

    file_prof  =  Dataset(filename_prof,read='r')


    extra_string = ''

    

    n_chords = 0


    #This now a bit trickier then for the 3D version. Will have to calculate a vector for the lower time resolution of the profile,
    #Then latter apply the nearest value to the full 1d time vec
    #First loading surface variables from default profile

    print('calculating cbl height from profile file')
    T = file_prof['thl'][:,0]
    p = file_prof['p'][:,0]*0.0+99709
    qt = file_prof['qt'][:,0]
    w2 = file_prof['w2'][:,:]
    nz_prof = w2.shape[1]
    var_prof =  file_prof[reg_var][:,:] #needed for anomaly processing
    #Just grabbing this to calculate dz
    z_prof = file_prof['z'][:]
    dz = z_prof[1]-z_prof[0]
    print('dz: ',dz)
    

    time_prof = file_prof['time'][:]
    cbl_1d_prof = time_prof*0.0

    #Hack together the Lifting condensation level LCL
    qt_pressure = p*qt
    sat_qv = 6.112*100 * np.exp(17.67 * (T - 273.15) / (T - 29.65 ))
    #rel_hum = np.asmatrix(qt_pressure/sat_qv)[0]
    rel_hum = qt_pressure/sat_qv
    #Dewpoint
    A = 17.27
    B = 237.7
    alpha = ((A * (T- 273.15)) / (B + (T-273.15)))
    alpha = alpha + np.log(rel_hum)
    dewpoint = (B * alpha) / (A - alpha)
    dewpoint = dewpoint + 273.15
    LCL = 125.*(T-dewpoint)
    LCL_index = np.floor(LCL/dz)

    #now calculate the cbl top for each profile time
    for tt in range(len(time_prof)):
        w_var = 1.0
        z=z_min
        while w_var > 0.08:
            z += 1
            w_var = w2[tt,z]
            #w_var = np.var(w_1d[z,:])

        #Mimimum of LCL +100 or variance plus 300 m 
        cbl_1d_prof[tt] = min(z+300/dz,LCL_index[tt])
        #To avoid issues later on I set the maximum cbl height to 60 % of the domain height, but spit out a warning if it happens
        if cbl_1d_prof[tt]>0.6*nz_prof:
            print('warning, cbl height heigher than 0.6 domain height, could crash regularization later on, timestep: ',tt)
            cbl_1d_prof[tt] = math.floor(nz*0.6)
    print('resulting indexes of cbl over time: ',cbl_1d_prof)
    print('calculated LCL: ',LCL_index)


    #Now we either iterate over columns or timesteps
    if data_dim_flag==1:
        n_iter =len(filename_column)
    if data_dim_flag==3:
        n_iter =len(time_prof)


    #Setting curtains for var
    var_curtain_sum = np.zeros([n_t_reg,n_z_reg])
    var_curtain_up_sum = np.zeros([n_t_reg,n_z_reg])
    var_curtain_dw_sum = np.zeros([n_t_reg,n_z_reg])

    n_curtain        = 0
    n_chord        = 0
    n_curtain_up     = 0
    n_curtain_dw     = 0




    #for col in filename_column:
    n_iter = min(n_iter,N_it_max)
    for it in range(N_it_min,n_iter):
        print('n_chords: ',n_chords)
        print('n_curtain: ',n_curtain)




        time1 = ttiimmee.time()
        if data_dim_flag ==1:
            print('loading column: ',filename_column[it])
            file_col = Dataset(filename_column[it],read='r')

            w_2d = file_col.variables['w'][:]
            w_2d = w_2d.transpose()
            ql_2d = file_col.variables['ql'][:]
            ql_2d = ql_2d.transpose()
            t_1d = file_col.variables['time'][:]
            u_2d  = file_col.variables['u'][:]
            u_2d  = u_2d.transpose()
            v_2d  = file_col.variables['v'][:]
            v_2d  = v_2d.transpose()
            print('t_1d',t_1d)
            #Load the var file, even if means that we doable load w_2d or ql_2d
            var_2d = file_col.variables[reg_var][:]
            
            
            var_2d = var_2d.transpose()
            
            
            #The needed cbl height
            cbl_1d = t_1d*0

            #Now we go through profile time snapshots and allocate the closest full time values to the profile values
            dt_2 = (time_prof[1]-time_prof[0])/2
            for tt in range(len(time_prof)):
                cbl_1d[abs(t_1d-time_prof[tt])<dt_2] = cbl_1d_prof[tt]
                
            #to get anomalies we subtract the closet mean profile
            if anomaly_flag==1:
                for tt in range(len(time_prof)):
                    #print('var_2d[.shape',var_2d.shape)
                    #print('var_prof.shape',var_prof.shape)
                    #globals().update(locals())
                    tmp_matrix =  var_2d[:,abs(t_1d-time_prof[tt])<dt_2]
                    tmp_vector =  var_prof[tt,:]
                    #because the vectors don't perfectly align
                    var_2d[:,abs(t_1d-time_prof[tt])<dt_2] = (tmp_matrix.transpose() - tmp_vector).transpose()

                    # = var_2d[:,abs(t_1d-time_prof[tt])<dt_2]-var_prof[tt,:]
                
                


        if data_dim_flag ==3:


            if sum(file_prof['ql'][it,:])>0.0:

                print('loading timestep: ',it)

                ql_3d   = grab_3d_field(file_ql  ,it,'ql')
                w_3d    = grab_3d_field(file_w   ,it,'w')
                var_3d  = grab_3d_field(file_var ,it,reg_var)

                #Here we have to do all the fuckery to turn the 3D fields into 2d slices with an imaginary time vector
                w_2d   = np.array(w_3d.reshape((nz,nx*ny)))
                ql_2d  = np.array(ql_3d.reshape((nz,nx*ny)))
                var_2d = np.array(var_3d.reshape((nz,nx*ny)))
                
                #Now we do the same thing with the transposed field, use to be an either or, now just add it on
                w_3d   = np.transpose( w_3d,  (0, 2, 1))
                ql_3d  = np.transpose(ql_3d,  (0, 2, 1))
                var_3d = np.transpose(var_3d, (0, 2, 1))
                
                #globals().update(locals())                
                w_2d     = np.hstack([w_2d   ,np.array(w_3d.reshape((nz,nx*ny)))])
                ql_2d    = np.hstack([ql_2d  ,np.array(ql_3d.reshape((nz,nx*ny)))])
                var_2d   = np.hstack([var_2d ,np.array(var_3d.reshape((nz,nx*ny)))])
                
                
                
                ##Transposing 3D fields to sample the clouds in another direction
                #if direction_slice=='y':
                #    w_3d   = np.transpose( w_3d,  (0, 2, 1))
                #    ql_3d  = np.transpose(ql_3d,  (0, 2, 1))
                #    var_3d = np.transpose(var_3d, (0, 2, 1))
                
                #Should now be able to delete 3d fields as they aren't needed anymore, not sure if that helps save any memory though
                del w_3d
                del ql_3d
                del var_3d
                
                gc.collect()
                
                
                #Switching to anomalies if anomaly flag is used
                if anomaly_flag==1:
                    #tmp_matrix =  var_2d[:,abs(t_1d-time_prof[tt])<dt_2]
                    #tmp_vector =  var_prof[tt,:]
                    #because the vectors don't perfectly align
                    var_2d[:,:] = (var_2d.transpose() - var_prof[it,:]).transpose()
                
                
                #to get the fake time vector we load the wind from the profile data, which devided by the grid spacing gives us a fake time resolution
                #we use the calculated cbl+300 meter or lcl as reference height 
                ref_lvl = cbl_1d_prof[tt]

                u_ref = file_prof['u'][it,ref_lvl]
                v_ref = file_prof['v'][it,ref_lvl]

                V_ref = np.sqrt(u_ref**2+v_ref**2) 

                time_resolution = dx/V_ref
                #if time_resolution > t_gap:
                #    print('t_gap too small:', t_gap)
                #    t_gap = time_resolution*1.5
                #    print('changed t_gap to:', t_gap)

                print('time iterative, V_ref, time_resolution',it, V_ref, time_resolution )
                #fake t vector, 
                t_1d = np.linspace(0,2*nx*ny*time_resolution,2*nx*ny)#+nx*ny*time_resolution*it
                #dt_1d   = t_1d*0
                #dt_1d[1:] = t_1d[1:]-t_1d[:-1]   
                

                #calculate scaling factors if needed
                if scale_flag == 1:

                    scaling_factor_x = u_ref/np.sqrt(u_ref**2+v_ref**2)
                    scaling_factor_y = v_ref/np.sqrt(u_ref**2+v_ref**2)
                    print('Scaling: u_ref: ,',u_ref,' v_ref: ', v_ref, ' scaling factor_x: ',scaling_factor_x,' scaling factor_y: ',scaling_factor_y,)



            else:
                #If no clouds are present we pass a very short empty fields over to the chord searcher
                print('skipping timestep: ',it,' cause no clouds')
                ql_2d  = np.zeros((nz,1))
                w_2d   = np.zeros((nz,1))
                var_2d = np.zeros((nz,1))
                t_1d   = np.zeros(1)

            #The needed cbl height, which constant everywhere
            cbl_1d = t_1d*0
            cbl_1d[:] = cbl_1d_prof[it]






        time2 = ttiimmee.time()
        print('loading time:',(time2-time1)*1.0,)



        ### Detecting lowest cloud cell is within 300 m of CBL

        nt = len(cbl_1d)
        cl_base = np.zeros(nt)

        #Detecting all cloudy cells
        #Use to have a different method using nans that doesn:t work anymore somehow. Now I just set it really high where there is no cloud. 
        for t in range(nt):
            if np.max(ql_2d[:,t])>ql_min :
                cl_base[t]=np.argmax(ql_2d[:,t]>1e-6)
            else:
                cl_base[t]=10000000

        cl_base=cl_base.astype(int)

        #Now find c base lower than the max height
        cbl_cl_idx = np.where((cl_base-cbl_1d[:nt])*dz<0)[0]

        cbl_cl_binary = cl_base*0
        cbl_cl_binary[cbl_cl_idx]=1
        t_cbl_cl=t_1d[cbl_cl_idx]


        ### Clustering 1D

        #Now we simply go through all cloudy timesteps
        #As long as the difference to the next cloudy timestep is lower than t_gap it counts as the same cloud
        #As an additional contraint, if the cloudy cells are right next to each other they are always counted as consecutive, not matter the time distance between them. 
        #if the difference is larger than 20s the cloud is over, and a chordlength is created which is a list of all timesteps that below to that chordlength
        #However if the duration of the chordlength is lower than t_min  or higher than t_max seconds it isn't
        #I added an additional constraint that each chord must include at least cell_min cells, because it is possible to get 
        #Small chord lengths with more than t_min which are mostly gaps. 


        t_cloudy_idx = 0
        #n_chords = 0
        chord_idx_list = []

        #t_cloudy_idx = int(np.floor(len(cbl_cl_idx)*0.99))+500 #just to make things short and easier to handle
        #t_cloudy_idx = 289933-1
        print('iterating through step ',it,'which contains ',len(cbl_cl_idx),'cloudy columns')


        while t_cloudy_idx < len(cbl_cl_idx)-1:# and n_curtain<100*it:      ####################################GO HERE TO SET MAXIMUM CURTAIN
            #print('t_chord_begin',t_chord_begin)
            t_chord_begin = t_cloudy_idx
            #now connecting all cloudy indexes
            while t_cloudy_idx < len(cbl_cl_idx)-1 and (cbl_cl_idx[t_cloudy_idx+1]==cbl_cl_idx[t_cloudy_idx]+1 or t_cbl_cl[t_cloudy_idx+1]-t_cbl_cl[t_cloudy_idx]<t_gap):
                t_cloudy_idx += 1 
            t_chord_end = t_cloudy_idx
            #print('t_chord_end',t_chord_end)


            #Checking if it fulfils chord criteria regaring time
            #we also added a minimum height of 100 m to screen out fog/dew stuff at the surface
            if t_chord_end-t_chord_begin>cell_min:
                chord_z_min = np.min(cl_base[cbl_cl_idx[t_chord_begin:t_chord_end]])
                chord_duration = t_cbl_cl[t_chord_end]-t_cbl_cl[t_chord_begin]
            else:
                chord_z_min = 0
                chord_duration = 0
                
                
            
            
            
            if chord_duration>t_min and chord_duration<t_max and chord_z_min > 4:


                if t_chord_end-t_chord_begin>cell_min-1:
                    n_chords += 1
                    #chord_idx_list.append(list(cbl_cl_idx[t_chord_begin:t_cloudy_idx]))

                    #Here we start the interpolation stuff
                    #Getting the chord beginning and end
                    idx_beg_chord = cbl_cl_idx[t_chord_begin]
                    idx_end_chord = cbl_cl_idx[t_chord_end]
                    time_beg_chord = t_1d[idx_beg_chord]
                    time_end_chord = t_1d[idx_end_chord]

                    #Calculate the beginning and end of the curtain, we add a bit to to each side to make interpolation easy
                    idx_beg_curtain = (np.abs(t_1d - (time_beg_chord-curtain_extra*(time_end_chord-time_beg_chord)))).argmin()-1
                    idx_end_curtain = (np.abs(t_1d - (time_end_chord+curtain_extra*(time_end_chord-time_beg_chord)))).argmin()+2
                    idx_end_curtain = min(idx_end_curtain,nt-1)
                    time_beg_curtain = t_1d[idx_beg_curtain]
                    time_end_curtain = t_1d[idx_end_curtain]


                    chord_cells = t_chord_end-t_chord_begin
                    curtain_cells = idx_end_curtain-idx_beg_curtain

                    #If curtain has more than curtain_min cells and curtain tail noes not extend beyond end of 2d field or the beginning extend before 
                    #I added 2 cells buffer at the beginning and end, because for the interpolation a bit of overlap is used. 
                    if idx_end_curtain<nt-2 and idx_beg_curtain>2 and len(cbl_cl_idx[t_chord_begin:t_chord_end])>curtain_min-1:
                        n_curtain += 1

                            
                        #First thing to do is calculate the chord base using the 25 percentile in agreement with Neil
                        z_idx_base_default = math.floor(np.percentile(cl_base[cbl_cl_idx[t_chord_begin:t_cloudy_idx]],base_percentile))

                        #Regularized curtains, I am too lazy to pass on all my variables to func_curtain_reg so I use something that everyone says is horrible
                        #and make all local variables global ones :)
                        #print('idx_beg_curtain: ',idx_beg_curtain,' idx_end_curtain: ',idx_end_curtain )
                        globals().update(locals())
                        var_curtain_tmp = (func_curtain_reg(var_2d)).transpose()
                        

                        #Finally add it to the mean one and track one more curtain
                        #detecting if chord base has a positive or negative w, then adds to the sum of up or downdraft chords
                        w_tmp = w_2d[cl_base[cbl_cl_idx[t_chord_begin:t_cloudy_idx]]-1,cbl_cl_idx[t_chord_begin:t_chord_end]]
                        #print(w_tmp)
                            
                        if scale_flag==0:
                            var_curtain_sum  = var_curtain_sum+var_curtain_tmp
                            if np.mean(w_tmp)>0.:
                                n_curtain_up += 1
                                var_curtain_up_sum += var_curtain_tmp
                            elif np.mean(w_tmp)<0.:
                                n_curtain_dw += 1
                                var_curtain_dw_sum += var_curtain_tmp
                            else:
                                print('wtf how is this zero: ',np.mean(w_tmp),w_tmp)

                        #Scaling is now added here, 
                        #Things are applied twice so that deviding by n it comes out fin
                        #We assume here that n_x and n_y are roughly same
                        #Could be made cleaner later on
                        if scale_flag==1 and data_dim_flag==3:
                            #find out if we need scaling_factor_x or y by seeing if we are in the first or second half
                            if idx_end_curtain<nt/2:
                                scaling_factor = 2*scaling_factor_x
                            else:
                                scaling_factor = 2*scaling_factor_y
                                
                            
                            if scaling_factor>0:
                                var_curtain_tmp = var_curtain_tmp[::-1,:]
                                
                            var_curtain_sum    = var_curtain_sum+abs(scaling_factor) * var_curtain_tmp
                            if np.mean(w_tmp)>0.:
                                n_curtain_up += 1
                                var_curtain_up_sum += abs(scaling_factor) * var_curtain_tmp
                            elif np.mean(w_tmp)<0.:
                                n_curtain_dw += 1
                                var_curtain_dw_sum += abs(scaling_factor) * var_curtain_tmp
                            else:
                                print('wtf how is this zero: ',np.mean(w_tmp),w_tmp)
                                
                        
                        ###############################################################################################################################################
                        ################## SIZE BINNING  ##############################################################################################################
                        ###############################################################################################################################################
                        
                        if size_bin_flag:
                            
                            #getting V_ref if data_dim_flag==1. Is calculated directly from the cloud base speeds
                            if data_dim_flag==1:
                                ch_idx_l = list(cbl_cl_idx[t_chord_begin:t_chord_end])
                                u_ref=np.mean(u_2d[cl_base[ch_idx_l],ch_idx_l])
                                v_ref=np.mean(v_2d[cl_base[ch_idx_l],ch_idx_l])
                                V_ref=np.sqrt(u_ref**2+v_ref**2)
                                
                            ch_duration  = t_cbl_cl[t_chord_end]-t_cbl_cl[t_chord_begin]
                            chord_length = ch_duration*V_ref

                            if scale_flag==0:
                                scaling_factor=1.
                            
                                
                            #find index of bin close to mid size bin
                            bin_idx = np.where(np.abs(chord_length-mid_bin_size)<125)[0]
                            if bin_idx.size>0:
                                #print('bin_idx,chord_length',bin_idx,chord_length)
                                n_curtain_bin[bin_idx] += 1                                                   
                                var_curtain_bin_sum[bin_idx,:,:]    = var_curtain_bin_sum[bin_idx,:,:]+abs(scaling_factor) * var_curtain_tmp

                                if np.mean(w_tmp)>0.:
                                    n_curtain_bin_up[bin_idx] += 1
                                    var_curtain_bin_up_sum[bin_idx,:,:] += abs(scaling_factor) * var_curtain_tmp
                                elif np.mean(w_tmp)<0.:
                                    n_curtain_bin_dw[bin_idx] += 1
                                    var_curtain_bin_dw_sum[bin_idx,:,:] += abs(scaling_factor) * var_curtain_tmp
                                else:
                                    print('wtf how is this zero: ',np.mean(w_tmp),w_tmp)
                            
                            
                        

                        


                        ##############################################################################################################################
                        #PLOTS
                        ##############################################################################################################################
                        #If the plot flag is set the pre regularization curtains are plotted. 
                        if plot_curtains_flag ==1:
                            print('plotting not implemented yet')






            t_cloudy_idx += 1
        time3 = ttiimmee.time()

        print('curtain processing:',(time3-time2)*1.0,)
    
    save_string_base = '_curt_'+date+'_d'+str(data_dim_flag)+'_cb'+str(base_smoothing_flag)+'_an'+str(anomaly_flag)+'_ct'+str(chord_times)+'_'+special_name+'_N'+str(n_curtain)

    if scale_flag ==1:
        save_string_base = save_string_base+'_scaled'
    
    save_string = 'data_curtains/'+ reg_var+save_string_base
    #if data_dim_flag==3:
    #    save_string= save_string+'_'+direction_slice
       
    save_string = save_string+'.npz'

    np.savez(save_string,var_curtain_sum=var_curtain_sum,var_curtain_up_sum=var_curtain_up_sum,var_curtain_dw_sum=var_curtain_dw_sum,n_curtain=n_curtain,n_curtain_up=n_curtain_up,n_curtain_dw=n_curtain_dw)
    print('saved curtains to '+save_string)
    
    #Adding dictionary to save all properties
    save_string = 'data_curtains/settings_'+ reg_var+save_string_base+'.pkl'
    with open(save_string, 'wb') as f:
        pickle.dump(settings_dict, f, pickle.HIGHEST_PROTOCOL)
    print('saved settings dict to '+save_string)
    
    
    
    if size_bin_flag==1: 
        save_string = 'data_curtains/'+ reg_var+save_string_base+'_sizebin'
        save_string = save_string+'.npz'
        np.savez(save_string,
                 var_curtain_sum    = var_curtain_bin_sum,
                 var_curtain_up_sum = var_curtain_bin_up_sum,
                 var_curtain_dw_sum = var_curtain_bin_dw_sum,
                 n_curtain    =n_curtain_bin,
                 n_curtain_up =n_curtain_bin_up,
                 n_curtain_dw =n_curtain_bin_dw)
        print('saved size binned curtains to '+save_string)
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    print(':')
    
        

        


            
    return








   
    
