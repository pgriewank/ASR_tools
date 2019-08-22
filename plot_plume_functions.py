#Contains important plot scripts to plot plumes and distributions, as well as some support functions to deal with weird panda formats and round down and so on
#Most important is  plot_prof_full_reconstruction
# coding: utf-8




import numpy as np
import math
from netCDF4 import Dataset
import os
from datetime import datetime,timedelta
from scipy import stats

#from unionfind import Unionfind
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.pyplot import cm 
from matplotlib.colors import LogNorm
from cusize_functions import *
import time as ttiimmee
import sys
import pickle
#from cdo import *
import pandas as pd




#dirty function to mean over all profiles in a pandas

def func_prof_mean(clus_prop,var_name,n_z):
    
    flux_matrix = np.zeros([n_z,len(clus_prop)])
    flux_vec   = np.zeros(n_z)
    flux_vec[:]    = 'nan'
    flux_matrix[:] = 'nan'
    
    for i in range(len(clus_prop)):
        flux_matrix[:,i] = clus_prop[var_name].iloc[i][:].ravel()
    for k in range(n_z):
        flux_vec[k]   = np.nanmean(flux_matrix[k,:])
    
    return(flux_vec)




#dirty function to sum over all profiles in a pandas

def func_prof_sum(clus_prop,var_name,n_z):
    
    flux_matrix = np.zeros([n_z,len(clus_prop)])
    flux_vec   = np.zeros(n_z)
    for i in range(len(clus_prop)):
        flux_matrix[:,i] = clus_prop[var_name].iloc[i][:].ravel()
    for k in range(n_z):
        flux_vec[k]   = np.nansum(flux_matrix[k,:])
    
    return(flux_vec)




def round_down(n):
    order = np.floor(math.log(n,10))
    scaled_n = n*10.**(-order)
    floor_scaled = np.floor(scaled_n)
    
    return floor_scaled*10**order







def plot_prof_binned_time(clus_prop,var_bin,max_bin=5,min_bin=0,t_steps=20,prof_var='w profile',bin_string='',prescribed_width=0,n_z=256,dz=25):
    #Plots all plumes profiles individually of a pandas file, after binning for a given parameter var_bin olors all plumes according to time
    #max_bin: Number of bins
    #min_bin: lowest bin to plot, increasing this greatly speeds up plotting
    #t_steps: number of timesteps to plot
    #prof_car: variable to plot (muss be profile)
    #bin_string: text string for bin label
    #precribed_width: defines bin width
    #n_z, dz vertical axes


    #determining bin width from the 99.9th percentile instead
    max_var = np.percentile(var_bin,99.9)
    bin_width = round_down(max_var/max_bin)
    print('bin_width:',bin_width)
    if prescribed_width>0:
        bin_width = prescribed_width
        print('bin_width override:',bin_width)
        

    bin_n, bins, bin_ind, csd = linear_binner(bin_width,var_bin)

    x_bin = (bins[:-1]+bins[1:])/2.

    rainbow = plt.get_cmap('cool')

    time_vec = np.unique(clus_prop['time'])
    
    #limiting t_steps to avoid issues
    t_steps = min(t_steps,len(time_vec))
    
    #trying to determine alpha from sample size
    alpha_vec = np.sqrt(csd[max_bin-1]/csd[:max_bin])*0.5

    prof_z=np.linspace(0,n_z*dz,n_z)+dz/2.

    
    
    fig, axes = plt.subplots(1, max_bin-min_bin,figsize=[20,10],sharey=True,sharex=True)
    #for the legend
    b=min_bin
    clus_tmp = clus_prop.iloc[bin_ind==b+1]


    for t in range(t_steps):
        if max(clus_tmp['time']==time_vec[t]):
            color_rain = rainbow(0 + float(t)/float(t_steps))
            idx_time = np.where(clus_tmp['time']==time_vec[t])[0] 

            clus_tmp_t = clus_tmp.iloc[idx_time]
            tlabel = str(time_vec[t])

            axes[b-min_bin].plot(clus_tmp_t[prof_var].iloc[0],-prof_z,color=color_rain,alpha=1,label=tlabel[11:16])

    l1 = axes[b-min_bin].legend(ncol=2)



    for b in range(min_bin,max_bin):



        clus_tmp = clus_prop.iloc[bin_ind==b+1]
        for t in range(t_steps):
            color_rain = rainbow(0 + float(t)/float(t_steps))
            idx_time = np.where(clus_tmp['time']==time_vec[t])[0] 

            clus_tmp_t = clus_tmp.iloc[idx_time]
            for n in range(len(clus_tmp_t)):
                axes[b-min_bin].plot(clus_tmp_t[prof_var].iloc[n],prof_z,color=color_rain,alpha=alpha_vec[b])#,label=str(x_A[b]))
        axes[b-min_bin].set_title(bin_string+'bin: ' +str(bins[b])[:-2] + '-' + str(bins[b+1])[:-2])
        axes[b-min_bin].set_xlabel(prof_var)
        
        if prof_var == 'w profile':
            axes[b-min_bin].set_xlim([0,6])





    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    
    #Getting height according to 70 percentile at 1 m percision using linear interpolation to get automatically scaled height 
    percentile=70
    prof_sum_height = func_prof_sum(clus_tmp,'Area profile',n_z)

    z_percentile=Area_percentile_x(prof_z,prof_sum_height,percentile)
    z_scaled = z_percentile*100./percentile
    
    ax1=axes[0]
    #ax1.set_ylim([0,3500])
    ax1.set_ylim([0,z_scaled*1.1])
    ax1.set_ylabel('z in m')

    return fig




def Area_percentile_x(x,fx,percentile):
    #Calculates the x_per value at which the integral from x[0] to x_per of fx is the percentile of the whole value
    #Only really works if fx>0
    #Tries to get a somewhat more precise value by using some linear interpolation, for the intended puposes here the default precision is 1 m
    #Is shit, can't believe that I couldn't find a stats function to do this
    x_interp = np.linspace(x[0],x[-1],x[-1]-x[0]+1)
    fx_interp = np.interp(x_interp, x, fx)
    cy = np.cumsum(fx_interp)
    cy_norm = cy/cy[-1]
    idx=np.argmin(abs(cy_norm-percentile/100.))

    return x_interp[idx]













def plot_prof_full_reconstruction(clus_prop,var_bin,max_bin=5,t_steps=20,prof_var='w profile',bin_string='',prescribed_width=0,
                                  prof_height='Area profile',percentile=90.,height_plot_flag=2,t_window=0,
                                  fixed_scaling = 0,plume_min=1,simple_mean_flag=0,n_z=256,dz=25,n_col=2,
                                  xmax=0,single_t=0,A_base_sorting = 0):
    """     
    Reconstructs EDMFn comparable plumes for given bins from Courvreux plumes. Then plots them colored according to time.
    
    See project report 2019-01 for full desciption of reconstruction, but it involves an area weighting and Volume compensation
    
    Parameters:
        max_bin: Number of bins
        t_steps: number of timesteps to plot
        prof_var: variable to plot. Must be profile, and must be in deviations from background state.
        prof_height: profile variable to determine plume height from
        percentile: percentile used to determine height from prof_height variable
        height_plot_flag 0: keine height information, 1: baren, 2: punkt
        bin_string: text string for bin label
        precribed_width: defines bin width
        t_window: time before and after which are included in the average in seconds
        fixed_scaling: if not zero a this parameter is used 
        plume_min: numbers of plumes needed to be plotted
        simple_mean_flag=0: if set to one uses a simple mean instead of reconstructing the profiles
        xmax sets the right xlim
        single_t: if bigger than 1 plots only a single subplot with all size bins at the single_t timestep
        A_base_sorting: if 1 it overrides the standard linear binner with the A_base binner, should help clean up the smallest bins 
    
    Returns:
        fig: the figure
        axes: the figure axes
        plume_profiles_btz: profiles of reconstructed profiles for each bin and time 
        plume_height: height of plumes for each bin and time
    """
    from cusize_functions import func_A_base_binner
    from cusize_functions import linear_binner
    
    #First calculate area weighted profile: 
    clus_prop['weighted var'] = clus_prop['Area profile']*clus_prop[prof_var]


    #determining bin width using the 99.9th percentile
    max_var = np.percentile(var_bin,99.9)
    bin_width = round_down(max_var/max_bin)
    print('bin_width:',bin_width)
    if prescribed_width>0:
        bin_width = prescribed_width
        print('bin_width override:',bin_width)
        

    if A_base_sorting: 
        bin_n, bins, bin_ind, csd = func_A_base_binner(clus_prop,max_bin=max_bin,prescribed_width=prescribed_width,
                                    percentile=percentile,t_window=t_window,plume_min=plume_min,n_z=n_z,dz=dz)
    else: 
        bin_n, bins, bin_ind, csd = linear_binner(bin_width,var_bin)

    x_bin = (bins[:-1]+bins[1:])/2.

    rainbow = plt.get_cmap('cool')

    time_vec = np.unique(clus_prop['time'])
    
    #limiting t_steps to avoid issues
    t_steps = min(t_steps,len(time_vec))


    prof_z=np.linspace(0,n_z*dz,n_z)+dz/2.
    
    #Creating height matrix to enable plotting it later on
    plume_height = np.zeros([max_bin,t_steps])
    plume_x      = np.zeros([max_bin,t_steps])

    #Creating bin, time, z profile  matrix to pass back when desired
    plume_profiles_btz = np.zeros([max_bin,t_steps,n_z])

    
    if single_t==0:   fig, axes = plt.subplots(1, max_bin,figsize=[20,10],sharey=True,sharex=True)
    else: fig, axes = plt.subplots(1, 1,figsize=[5,5])

    for b in range(max_bin):

        clus_tmp = clus_prop.iloc[bin_ind==b+1]
        blabel =  str(bins[b])[:-2] + '-' + str(bins[b+1])[:-2]
        if single_t >0: color_rain = rainbow(0 + float(b)/float(max_bin))

        if single_t==0:
            t_iterable = np.arange(t_steps).astype(int)
        else:
            t_iterable = [single_t]
        for t in t_iterable:
            if single_t ==0: color_rain = rainbow(0 + float(t)/float(t_steps))
            #idx_time = np.where(clus_tmp['time']==time_vec[t])[0] 
            
            #expanding to include a time window before and after. First calculating t difference to current timestep
            delta_t = abs((clus_tmp['time']-time_vec[t])/np.timedelta64(1, 's'))
            
            idx_time = np.where(delta_t<=t_window)[0] 
            clus_tmp_t = clus_tmp.iloc[idx_time]
            tlabel = str(time_vec[t])

            if len(idx_time)>plume_min:
                number_of_plumes = float(len(idx_time))
                
                
                #Getting height according to percentile at 1 m percision using linear interpolation 
                prof_sum_height = func_prof_sum(clus_tmp_t,prof_height,n_z)

                z_percentile=Area_percentile_x(prof_z,prof_sum_height,percentile)
                z_scaled = z_percentile*100./percentile
                
                if simple_mean_flag==0:
                    if fixed_scaling == 0: 

                        #print(b,t,z_percentile)
                        V_edmf = x_bin[b]*x_bin[b]*z_scaled
                        V_LES  = np.mean(clus_tmp_t['Volume']*clus_tmp_t['Volume']*clus_tmp_t['Volume'])

                        N_Vc = number_of_plumes*V_LES/V_edmf

                        prof_plot = func_prof_sum(clus_tmp_t,'weighted var',n_z)/N_Vc/x_bin[b]/x_bin[b]

                    else:
                        prof_plot = func_prof_sum(clus_tmp_t,'weighted var',n_z)*fixed_scaling/number_of_plumes/x_bin[b]/x_bin[b]
                else:
                    prof_plot=func_prof_mean(clus_tmp_t,prof_var,n_z)
                    

                if single_t ==0: 
                    axes[b].plot(prof_plot,prof_z,color=color_rain,alpha=1,label=tlabel[11:16])
                else:
                    axes.plot(prof_plot,prof_z,color=color_rain,alpha=1,label=blabel)

                # Saving height data to plot later on
                plume_height[b,t] = z_scaled
                plume_x     [b,t] = np.nanmax(prof_plot)


                #Saving pplume profile
                plume_profiles_btz[b,t,:] = prof_plot
                
                    
        if single_t==0:         ax = axes[b]
        else: ax = axes
        #plotting height info
        if height_plot_flag>0:
            for t in range(t_steps):
                if single_t == 0:    color_rain = rainbow(0 + float(t)/float(t_steps))
                if single_t == 1:    color_rain = rainbow(0 + float(b)/float(max_bin))
                if height_plot_flag==1:
                    ax.hlines(plume_height[b,t],0,plume_x[b,t],color=color_rain)
                if height_plot_flag==2:
                    ax.plot(plume_x[b,t],plume_height[b,t],color='k',markersize=6,marker='o')
                    ax.plot(plume_x[b,t],plume_height[b,t],color=color_rain,markersize=4,marker='o')
                if height_plot_flag==3:
                    ax.hlines(plume_height[b,t],0,1,color=color_rain)
                    ax.vlines(1,0,plume_height[b,t],color=color_rain)
                    #ax.plot(plume_x[b,t],plume_height[b,t],color='k',markersize=6,marker='o')
                    #ax.plot(plume_x[b,t],plume_height[b,t],color=color_rain,markersize=4,marker='o')
                                    

        if single_t ==0: 
            ax.set_title(bin_string+'bin: \n ' +str(bins[b])[:-2] + '-' + str(bins[b+1])[:-2])
            ax.set_xlabel(prof_var)
        else: 
            ax.set_title('time=: ' +tlabel)
            ax.set_xlabel(prof_var)
        

        
    
    if n_col>0:
        if single_t ==0:
            if t_window>0:
                l1 = ax.legend(ncol=n_col,frameon=False,title='+- ' + str(t_window)+ ' s')
            else:
                l1 = ax.legend(ncol=n_col,frameon=False)
        else:
            l1 = ax.legend(ncol=n_col,frameon=False, title = r'bin sizes in $\sqrt{A}$ [m]')
             

    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    
    ax.set_ylim([0,3500])
    ax.set_xlim([0,6])
    
    ax.set_ylim([0,1.1*np.nanmax(plume_height.ravel())])
    ax.set_xlim([0,1.1*np.nanmax(plume_x.ravel())])
   
    if xmax>0: 
        ax.set_xlim(right=xmax)


    ax.set_ylabel('z in m')
    return fig, axes, plume_profiles_btz,plume_height




def plot_psd_edmf_fit(var_bin,bin_width,bin_begin=300,norm=1000):
    #plot that takes a variable, bins by bin_width, plots that, also looks for the best fit, plots that, and also plots the default EDMFn slope. 
    #should only use bins before the first zero

    #bin_begin: from where on to begin the regularization
    bin_begin_n = int(np.ceil((bin_begin-1)/bin_width))
    #through which point the lines are laid
    norm_n = int(np.ceil((norm-1)/bin_width))

    bin_n, bins, bin_ind, psd = linear_binner(bin_width,var_bin)
    x_bins = bins[1:]/2+bins[:-1]/2
    psd_reg = psd/sum(psd)

    bin_end = np.where(psd==0)[0][0]
    
    psd_reg_nz=psd_reg[psd_reg>0]

    x_bins_nz = x_bins[psd_reg>0]

    #To play around with some linear regression
    x_log = np.log10(x_bins_nz[bin_begin_n:bin_end])
    psd_log = np.log10(psd_reg_nz[bin_begin_n:bin_end])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log,psd_log)

    psd_fit_log = intercept+slope*np.log10(x_bins_nz)
    psd_fit = intercept*x_bins_nz**slope

    #slope

    #EDMF powerlat, scaled to match psd at some point
    zb1 = -1.98

    N_EDMF  = x_bins_nz**zb1 
    N_EDMF = N_EDMF/N_EDMF[norm_n]*psd_reg[norm_n]

    fig, ax = plt.subplots(figsize=[5,5])


    plt.plot(np.log10(x_bins_nz),np.log10(N_EDMF),marker='o',label='EDMF slope -1.98',alpha=0.5)
    plt.plot(np.log10(x_bins_nz),psd_fit_log, marker='o',label='microhh fit = '+str(slope)[:5],alpha=0.5)
    plt.plot(np.log10(x_bins_nz), np.log10(psd_reg_nz),marker='o',label='microhh',alpha=0.5)
    ax.set_xlabel('plume square Area 10^ m')
    ax.set_ylabel('log occurence')
   

    ax.legend()
    
    return fig, ax

def plot_edmf_plumes(filename,t_steps = 24,var_string = 'w',ensemble_flag = 0,max_size = 1000):
#loads and plots the profiles of a dales_edmf run
#ensemble_flag = 0 plots one x,y column, 1 is ensemble


    f_edmf = Dataset(filename,'r')

    prof_z = f_edmf.variables['zt'][:]

    if ensemble_flag ==0:
        var = f_edmf.variables[var_string][:,1:,:,1,1]
    else:
        var_xy = f_edmf.variables[var_string][:,1:,:,:,:]
        var_x  = np.mean(var_xy,axis=4)
        var    = np.mean(var_x,axis=3)

    max_bin = var.shape[1]
    t_steps=min(t_steps,var.shape[0])



    rainbow = plt.get_cmap('cool')

    d_size = max_size/max_bin
    x_A = np.linspace(d_size/2,max_size-d_size/2,max_bin)
    bin_beg = np.linspace(0,max_size-d_size,max_bin).astype(int)#*np.pi**0.5
    bin_end = np.linspace(d_size,max_size,max_bin).astype(int)#*np.pi**0.5
    
    fig, axes = plt.subplots(1, max_bin,figsize=[20,10],sharey=True,sharex=True)

    time_vec = f_edmf.variables['time'][:]/3600+6

    z_max_idx = 0
    for b in range(max_bin):
        for t in range(t_steps):
            color_rain = rainbow(0 + float(t)/float(t_steps))
            tlabel = str(time_vec[t])
            prof_plot = var[t,b,:]
            axes[b].plot(prof_plot,prof_z,color=color_rain,alpha=1,label=tlabel)#,linewidth=1)
            if np.max(prof_plot)>0.0:
                z_max_idx = max(z_max_idx,np.max(np.where(prof_plot>0.)))

        axes[b].set_title('radius: \n' +str(bin_beg[b]) + '-' + str(bin_end[b]) + 'm')
        axes[b].set_xlabel(var_string)

    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    l1 = axes[0].legend(title='time [h]',ncol=1,frameon=False)

    ax1=axes[0]
    ax1.set_ylim([0,prof_z[z_max_idx+3]])
    ax1.set_ylabel('z [m]')

    return fig
