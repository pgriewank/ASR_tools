#todo
#make timesteps adaptive
#read in dz


import numpy as np
import math
from netCDF4 import Dataset
import os
from datetime import datetime,timedelta

#from unionfind import Unionfind
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.pyplot import cm 
from cusize_functions import *
import time as ttiimmee
import sys
import pickle
#from cdo import *
import pandas as pd



directory_clustering = '/data/inscape/phil/lasso_clustering/'
directory_clustering = '/data/testbed/lasso/clustering/'
directory_data          = '/data/testbed/lasso/sims/' #+date
directory_data          = '/data/inscape/lasso/sims/' #+date
dates = ['20160611_micro']
dates = ['20160611']
dates = ['20160830']
dates = ['bomex']
filename_w = []
filename_ql = []
filename_qt = []
filename_clus =[]
filename_couv =[]
for date in dates:
    filename_w.append(directory_data+date+'/w.nc')
    filename_ql.append(directory_data+date+'/ql.nc') 
    filename_qt.append(directory_data+date+'/qt.nc') 
    #filename_clus.append(directory_data+date+'/couvreux_clustering_20sig_filt_v2.pkl')
    filename_clus.append(directory_clustering+'couvreux_'+date+'_clustering_20sig_filt_v2.pkl')
    #filename_couv.append(directory_clustering+'couv_prop_sig2_filt_'+date+'.pkl')
    filename_couv.append(directory_clustering+'couv_prop_sig2_filt_'+date+'.pkl')
    
#col_names = ['Volume','sq Area','Radius','V_h','w','w profile','ql profile','qv profile','Area profile','Area','height','cl h','cl A','cl V','dry w','dry h','dry A','dry V','dry w','time','x','y','z','base','z max cf','w_ref','w_ref_vec','w_below_vec','w_below','w_bottom']
#We are getting rid of the bottom/below things because they are tricky and maybe no necessary
col_names = ['Volume','sq Area','Radius','V_h','w',                 'w profile','ql profile','qv profile','Area profile','Area','height',                    'wet h','wet A','wet V','wet w',            'dry w','dry h','dry A','dry V'           ,'time','x','y','z','base','z max cf','w flux','qt flux','qt total flux','qt fluc','w90 profile']   

    

dx = 25
dy = 25
dz = 25
dz = 23.4375
dA = dx*dy
dV = dx*dy*dz

ql_min = 1e-6 #cutoff for cloudy

n_z,n_x,n_y = get_zxy_dimension(filename_w[0],'w')

#got to make timesteps automated
timesteps = 70


#Time loop that loads all the cluss identified via clustering and calculates their properties


for d in range(len(dates)):
    #loading w 
    file_w    =  Dataset(filename_w[d],read='r')
    file_ql   =  Dataset(filename_ql[d],read='r')
    file_qt   =  Dataset(filename_qt[d],read='r')
    
    seconds_since_start = file_w.variables['time'][:]

    try:
        time_init = datetime.strptime(dates[d][0:7]+'0600','%Y%m%d%H%M')
    except:
        time_init = datetime.strptime('2020'+'01'+'01'+'0600','%Y%m%d%H%M')
    #loading clustering from file 
    time1 = ttiimmee.time()

    with open(filename_clus[d],'rb') as f:  # Python 3: open(..., 'rb')
        print('reading in clustering results via pickle in '+filename_clus[d])

        cluster_cell_list_time, idx_3d_cluster_cells_time = pickle.load(f,encoding='latin1')
    time2 = ttiimmee.time()
    print(' reading in the clustering took so many seconds:',(time2-time1))
    
    if timesteps > len(cluster_cell_list_time):

        timesteps = len(cluster_cell_list_time)
        print(' limiting timesteps to len(cluster_cell_list_time):',timesteps)



    #and calculating varous sizes as well as mean vertical velocity

    
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
    
    
    
    #couv_w_below_all   = np.zeros(0)
    #couv_w_bottom_all   = np.zeros(0)
    #couv_w_ref_all   = np.zeros(0)
    #couv_w_ref_vec_all   = []
    #couv_w_below_vec_all   = []


    
    for t in range(timesteps):
    #for t in range(20,21):
        
        ncluss = len(cluster_cell_list_time[t])
        time_now = time_init + timedelta(seconds=float(seconds_since_start[t]))
        print('timestep and ncluss ',t,ncluss)
        print('datetime ',t,time_now)
        
        if ncluss>0:
            time1 = ttiimmee.time()
            
            #load data
            w  = grab_3d_field(file_w,t,'w')
            ql = grab_3d_field(file_ql,t,'ql')
            qt = grab_3d_field(file_qt,t,'qt')
            qv = qt-ql
            qt_mean_prof = np.mean(qt,axis=1)
            qt_mean_prof = np.mean(qt_mean_prof,axis=1)
            
            w_qt_fluc = qt*0.0
            for n in range(n_z):
                w_qt_fluc[n,:,:]=w[n,:,:]*(qt[n,:,:]-qt_mean_prof[n])
            w_qt = w*qt
            
            couv_w = np.zeros(ncluss)
            couv_V = np.zeros(ncluss) 
            couv_A = np.zeros(ncluss)
            couv_h = np.zeros(ncluss)
            
            
            couv_max_cf = np.zeros(ncluss)
            couv_base = np.zeros(ncluss)
            
            couv_prof_w = np.zeros([ncluss,n_z]) 
            couv_prof_w90 = np.zeros([ncluss,n_z]) 
            couv_prof_A = np.zeros([ncluss,n_z]) 
            couv_prof_ql = np.zeros([ncluss,n_z]) 
            couv_prof_qv = np.zeros([ncluss,n_z]) 
            couv_prof_cf = np.zeros([ncluss,n_z]) 
            
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
            

            #couv_w_ref_vec = []
            #couv_w_below_vec = []
            #couv_w_below = np.zeros(ncluss) #imitate Lareau, get the layer beneath the clus everywhere within 300 m of the CBL
            #couv_w_bottom = np.zeros(ncluss) #get the lowest clus layer everywhere within 300 m of the CBL
            #couv_w_ref = np.zeros(ncluss)
                    
    
    
            #calculate the z lvl of maximum amount of clus fraction at that timestep
            idx_z = idx_3d_cluster_cells_time[t][0]
            z_max_cf  = np.argmax(np.bincount(idx_z))
            print('z_max_cf :',z_max_cf)
    
    
#             #Calculating cbl height + 300m using a critical value for the horizontal variability of w following Lareau
#             w_var = 1.0
#             z_var=0
#             while w_var > 0.08:
#                 z_var += 1
#                 w_var = np.var(w[z_var,:])
#             cbl_idx = z_var
#             cbl_idx_max = cbl_idx+int(300/dz)
            
#             print('cbl height + 300 m index :',cbl_idx_max)
    
            

            for i in range(ncluss):
                
                idx_z = idx_3d_cluster_cells_time[t][0][cluster_cell_list_time[t][i]]
                idx_x = idx_3d_cluster_cells_time[t][1][cluster_cell_list_time[t][i]]
                idx_y = idx_3d_cluster_cells_time[t][2][cluster_cell_list_time[t][i]]

                couv_w[i]   = np.mean(w[idx_z,idx_x,idx_y])
                couv_V[i]   = (float(len(cluster_cell_list_time[t][i]))*dV)**(1./3.)
                couv_A[i]   = func_proj_A(idx_x,idx_y,dA)
                couv_h[i]   = (max(idx_z)-min(idx_z)+1)*dz
                
                couv_max_cf[i] = z_max_cf
                
                
                
                couv_prof_w[i,:],tmp =func_vert_mean(idx_z,idx_x,idx_y,w)
                couv_prof_w90[i,:],tmp =func_vert_percentile(idx_z,idx_x,idx_y,w,90)
                couv_prof_A[i,:] = tmp*dA
                couv_prof_qv[i,:],tmp =func_vert_mean(idx_z,idx_x,idx_y,qv)
                couv_prof_ql[i,:],tmp =func_vert_mean(idx_z,idx_x,idx_y,ql)
                
                
                #getting the fluxes
                couv_prof_flux_w[i,:],tmp =func_vert_mean(idx_z,idx_x,idx_y,w)
                couv_prof_flux_w[i,:] = couv_prof_flux_w[i,:]*couv_prof_A[i,:]
                
                #couv_prof_flux_qt[i,:],tmp =func_vert_mean(idx_z,idx_x,idx_y,w*qt)
                #lets see if it is cheaper to calculate this once
                couv_prof_total_flux_qt[i,:],tmp =func_vert_mean(idx_z,idx_x,idx_y,w_qt)
                couv_prof_total_flux_qt[i,:] = couv_prof_total_flux_qt[i,:]*couv_prof_A[i,:]
                
                couv_prof_flux_qt[i,:],tmp =func_vert_mean(idx_z,idx_x,idx_y,w_qt_fluc)
                couv_prof_flux_qt[i,:] = couv_prof_flux_qt[i,:]*couv_prof_A[i,:]
                
                couv_prof_fluc_qt[i,:] = couv_prof_qv[i,:]+couv_prof_ql[i,:]-qt_mean_prof

                
                
#                 #Speed stuff which is currently not used
#                 couv_w_ref[i] = couv_prof_w[i,z_max_cf]
#                 ############################################################################################
#                 #Here we determine the cells which are 1 cell below the clus and no higher than cbl_idx_max
#                 #Using this we calculate the mean value of w
#                 #As well as the 95th percentile
#                 #And for a reason I forgot I also pass along all w values
#                 ############################################################################################
#                 idx_xy = np.vstack([idx_x,idx_y])
#                 idx_xy_unique = np.unique(idx_xy,axis=1)

#                 idx_xy_z_below = np.zeros(idx_xy_unique.shape[1],dtype=np.uint8) 
#                 for ii in range(idx_xy_unique.shape[1]):
#                     #Searches for where the x and y values of the unique value 
#                     idx_z_xy = idx_z[np.where((idx_xy_unique[0,ii]==idx_xy[0])*(idx_xy_unique[1,ii]==idx_xy[1]))[0]]
#                     #z_min = min(idx_z_xy)-1
#                     idx_xy_z_below[ii] = min(idx_z_xy)-1
#                     #print(z_min)
#                 w_below_vec = w[idx_xy_z_below,idx_xy_unique[0,:],idx_xy_unique[1,:]].ravel()
#                 w_below_vex_cbl300 = w_below_vec[idx_xy_z_below<cbl_idx_max]
#                 if w_below_vex_cbl300.size:
#                     couv_w_below_vec.append(w_below_vec)
#                     couv_w_below[i] = np.mean(w_below_vec)
                    
#                     w_below_up =  w_below_vex_cbl300[ w_below_vex_cbl300>0.1]
#                     if w_below_up.size:
#                         couv_w_below_95_up[i] = np.percentile(w_below_up,95)
#                     else:
#                         couv_w_below_95_up[i] = 'nan'
                    
#                 else:
#                     couv_w_below_vec.append([])
#                     couv_w_below[i] = 'nan'
                
#                 #Getting the bottom clus layer
#                 w_bottom_vec = w[idx_xy_z_below+1,idx_xy_unique[0,:],idx_xy_unique[1,:]].ravel()
#                 w_bottom_vec_cbl300 = w_bottom_vec[idx_xy_z_below+1<cbl_idx_max]
#                 if w_bottom_vec_cbl300.size:
#                     couv_w_bottom[i] = np.mean(w_bottom_vec)
#                 else:
#                     couv_w_bottom[i] = 'nan'
#                 #print('idx_xy_z_below',idx_xy_z_below)
#                 #print('idx_z',idx_z)
#                 #print('w_bottom_vec',w_bottom_vec)
#                 #print('w',w[idx_z,idx_x,idx_y])
#                 #print('wtf clus bottom vs clus w',couv_w_bottom[i],couv_w[i])
#                 #if i>100:
#                 #    []+absd
    
#                 ############################################################################################
#                 #getting the 95th percentile of all cells in the z_ref level
#                 #As well as the mean w at that level
#                 #And for good measure I also just pass along the whole vector of w
#                 ############################################################################################
#                 ind_z_ref = np.where(idx_z==z_max_cf)[0]
#                 w_ref_lvl = w[idx_z[ind_z_ref],idx_x[ind_z_ref],idx_y[ind_z_ref]].ravel()
#                 w_ref_up = w_ref_lvl[w_ref_lvl>0.1]
#                 if w_ref_lvl.size:
#                     couv_w_ref_95[i]    = np.percentile(w_ref_lvl,95)
#                     couv_w_ref_vec.append(w_ref_lvl)
#                 else:
#                     couv_w_ref_95[i]    = 'nan'
#                     couv_w_ref_vec.append([])
#                 if w_ref_up.size:
#                     couv_w_ref_95_up[i] = np.percentile(w_ref_up,95)
#                     couv_w_ref_up_n[i] = len(w_ref_up)
                    
                                              
#                 else:
#                     couv_w_ref_95_up[i] = 'nan'
#                     couv_w_ref_up_n[i]  = 0
                
                
                
#                 if np.isnan(np.nanmax(tmp)):
#                     print('wtf')
#                     print(tmp)
                    
                couv_base[i]=np.min(idx_z)*dz
                couv_t.append(time_now) 
                couv_x_max.append(np.argmax(np.bincount(idx_x)))
                couv_y_max.append(np.argmax(np.bincount(idx_y)))
                couv_z_max.append(np.argmax(np.bincount(idx_z)))
                
                
                #Now the new part, separating the cluster into dry and wet
                ql_tmp = ql[idx_z,idx_x,idx_y]
                
                idx_z_dry = idx_z[ql_tmp<ql_min] 
                idx_x_dry = idx_x[ql_tmp<ql_min] 
                idx_y_dry = idx_y[ql_tmp<ql_min] 
                
                if idx_z_dry.size>0:
                    couv_dry_w[i]   = np.mean(w[idx_z_dry,idx_x_dry,idx_y_dry])
                    couv_dry_V[i]   = (float(len(idx_z_dry))*dV)**(1./3.)
                    couv_dry_A[i]   = func_proj_A(idx_x_dry,idx_y_dry,dA)
                    couv_dry_h[i]   = (max(idx_z_dry)-min(idx_z_dry)+1)*dz
                
                
                idx_z_wet = idx_z[ql_tmp>=ql_min] 
                idx_x_wet = idx_x[ql_tmp>=ql_min] 
                idx_y_wet = idx_y[ql_tmp>=ql_min] 
                
                if idx_z_wet.size>0:
                    
                    #print('idx_z_wet',idx_z_wet)
                    #print('ql_tmp',ql_tmp)
                    #print('ql[idx_z_wet,idx_x_wet,idx_y_wet]',ql[idx_z_wet,idx_x_wet,idx_y_wet])
                    #print((float(len(idx_z_wet))*dV)**(1./3.),'(float(len(idx_z_wet))*dV)**(1./3.)')
                    #print((float(len(idx_z))*dV)**(1./3.),'(float(len(idx_z))*dV)**(1./3.)')
                    #print((float(len(idx_z))*dV)**(1./3.)-(float(len(idx_z_wet))*dV)**(1./3.))
                    
                    couv_wet_w[i]   = np.mean(w[idx_z_wet,idx_x_wet,idx_y_wet])
                    couv_wet_V[i]   = (float(len(idx_z_wet))*dV)**(1./3.)
                    couv_wet_A[i]   = func_proj_A(idx_x_wet,idx_y_wet,dA)
                    couv_wet_h[i]   = (max(idx_z_wet)-min(idx_z_wet)+1)*dz
                
                    if couv_wet_V[i]>couv_V[i]:
                        print('wtf is happening',couv_wet_V[i],couv_V[i])
                    
                    #I try to get the cf by calculating the Area profiles of both dry and wet and the ratio
                    #actually, it might be best to calculate that anyway
                    #I have a feeling this will throw up some weird shit when 
                    tmp1,tmpwet =func_vert_mean(idx_z_wet,idx_x_wet,idx_y_wet,w)
                    couv_wet_prof_A[i,:] = tmpwet*dA
                    tmp1,tmpdry =func_vert_mean(idx_z_dry,idx_x_dry,idx_y_dry,w)
                    couv_dry_prof_A[i,:] = tmpdry*dA
                
                
                
                    couv_prof_cf[i,:] = tmpwet/(tmpdry+tmpwet)
                
                else:
                    
                    couv_prof_cf[i,:] = couv_prof_A[i,:]*0.0
                    
                    
                
                
                
                
                
                
                
                


            couv_V_all = np.hstack([couv_V_all,couv_V])
            couv_w_all = np.hstack([couv_w_all,couv_w])
            couv_A_all = np.hstack([couv_A_all,couv_A])
            couv_h_all = np.hstack([couv_h_all,couv_h])
            
#             couv_w_ref_all = np.hstack([couv_w_ref_all,couv_w_ref])
#             couv_w_below_all = np.hstack([couv_w_below_all,couv_w_below])
#             couv_w_bottom_all = np.hstack([couv_w_bottom_all,couv_w_bottom])
#             couv_w_ref_95_all = np.hstack([couv_w_ref_95_all,couv_w_ref_95])
#             couv_w_ref_95_up_all = np.hstack([couv_w_ref_95_up_all,couv_w_ref_95_up])
#             couv_w_below_95_up_all = np.hstack([couv_w_below_95_up_all,couv_w_below_95_up])
#             couv_w_ref_up_n_all = np.hstack([couv_w_ref_up_n_all,couv_w_ref_up_n])
#             couv_w_ref_vec_all.extend(couv_w_ref_vec)
#             couv_w_below_vec_all.extend(couv_w_below_vec)
            
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
            couv_prof_w90_all = np.vstack([couv_prof_w90_all,couv_prof_w90])
            couv_prof_A_all = np.vstack([couv_prof_A_all,couv_prof_A])
            couv_prof_ql_all = np.vstack([couv_prof_ql_all,couv_prof_ql])
            couv_prof_qv_all = np.vstack([couv_prof_qv_all,couv_prof_qv])
            couv_prof_cf_all = np.vstack([couv_prof_cf_all,couv_prof_cf])
            

            couv_prof_flux_w_all  = np.vstack([couv_prof_flux_w_all ,couv_prof_flux_w ])
            couv_prof_flux_qt_all = np.vstack([couv_prof_flux_qt_all,couv_prof_flux_qt ])
            couv_prof_total_flux_qt_all = np.vstack([couv_prof_total_flux_qt_all,couv_prof_total_flux_qt ])

            couv_prof_fluc_qt_all = np.vstack([couv_prof_fluc_qt_all,couv_prof_fluc_qt ])

            couv_dry_prof_A_all = np.vstack([couv_dry_prof_A_all,couv_dry_prof_A])
            couv_wet_prof_A_all = np.vstack([couv_wet_prof_A_all,couv_wet_prof_A])
            
            
            
            time2 = ttiimmee.time()
            print('time needed to calculate clus properties',(time2-time1))

    #Now calculate clus radius and area square root. 
    couv_sqA_all   = np.sqrt(couv_A_all)
    couv_rad_all   = np.sqrt(couv_A_all/math.pi)
    #height averaged area, aka sqrt(V/h) 
    couv_V_h_all   = np.sqrt(couv_V_all**3./couv_h_all)
    
    
    #with open(filename_clus[d], 'wb') as f: 
    #    pickle.dump([ couv_V_all,couv_sqA_all,couv_rad_all,couv_V_h_all,couv_w_all,couv_prof_w_all,couv_prof_A_all,couv_A_all,couv_h_all,couv_t],f)
    #    print('saved clus properties in',filename_clus[d])

    #saving as panda
    data_for_panda = list(zip(couv_V_all,couv_sqA_all,couv_rad_all,couv_V_h_all,couv_w_all,couv_prof_w_all,couv_prof_ql_all,couv_prof_qv_all,couv_prof_A_all,couv_A_all,couv_h_all,couv_wet_h_all,couv_wet_A_all,couv_wet_V_all,couv_wet_w_all,couv_dry_w_all,couv_dry_h_all,couv_dry_A_all,couv_dry_V_all,couv_t_all,couv_x_max_all,couv_y_max_all,couv_z_max_all,couv_base_all,couv_max_cf_all,couv_prof_flux_w_all,couv_prof_flux_qt_all,couv_prof_total_flux_qt_all,couv_prof_fluc_qt_all,couv_prof_w90_all))
    
    
    #list(zip(couv_V_all,couv_sqA_all,couv_rad_all,couv_V_h_all,couv_w_all,couv_prof_w_all,couv_prof_ql_all,couv_prof_qv_all,couv_prof_A_all,couv_A_all,couv_h_all,couv_wet_h,couv_wet_A,couv_wet_V,couv_wet_w,couv_dry_w,couv_dry_h,couv_dry_A,couv_dry_V,couv_t_all,couv_x_max_all,couv_y_max_all,couv_z_max_all,couv_base_all,couv_max_cf_all))
    #,couv_w_ref_all,couv_w_ref_95_all,couv_w_ref_95_up_all,couv_w_ref_up_n_all,couv_w_ref_vec_all,couv_w_below_vec_all,couv_w_below_all,couv_w_below_95_up_all,couv_w_bottom_all))

    
    #list(zip(couv_V_all,couv_sqA_all,couv_rad_all,couv_V_h_all,couv_w_all,couv_prof_w_all,couv_prof_A_all,couv_A_all,couv_h_all,couv_t_all,couv_x_max_all,couv_y_max_all,couv_z_max_all,couv_base_all,couv_max_cf_all,couv_w_ref_all,couv_w_ref_95_all,couv_w_ref_95_up_all,couv_w_ref_up_n_all,couv_w_ref_vec_all,couv_w_below_vec_all,couv_w_below_all,couv_w_below_95_up_all,couv_w_bottom_all))
    df = pd.DataFrame(data = data_for_panda, columns=col_names)
    df.to_pickle(filename_couv[d])
    print('saved clus properties as panda in ',filename_couv[d])


# In[11]:

