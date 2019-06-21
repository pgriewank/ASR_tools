#!/usr/bin/env python3 

#Python script that should be run once for each lasso microhh simulation. The functions themselves are all in proc_chords_xarray.py
#The day for which it is to run should needs to be added as a string after the command, e.g. > ./run_postprocessing.py '201608230'



#I prefer running with nohup and -u to get ongoing output: nohup python3 -u run_postprocessing.py '20160611_micro' > output_postpro.txt 

#Requires proc_chords_xarray.py and cusize_functions.py from https://github.com/pgriewank/ASR_tools be in the same folder. 

#Output folder should be changed as necessary


#Individual functions to call. Each saves one or two files:
#proc_chords d3, ct0 
#proc_chords d1, ct1, all chords
#proc_reg_beards, 3d,  w, ct0, sizebin, curtain_extra = 8, N_it_min = 6 (avoid spin up issues) 
#proc_reg_beards, 1d,  w, ct1, sizebin, curtain_extra = 8 all columns
#proc_reg_beards, 1d, qt, ct1, sizebin, curtain_extra = 8 an-1, all columns
#proc_reg_beards, 1d,  w, ct1, loopall colums 
#wstar_pdf
#w_pdf
#proc_reg_beards, 3d, qt, ct0, sizebin, curtain_extra = 8 an=1

from proc_chords_xarray import *
import glob
import sys

try:
    date_str = sys.argv[1]
except:
    print('did you forget to add the date as a string? e.g. ./run_postprocessing.py 201608230')
    sys.exit()


#Should not change:
directory_input = '/data/lasso/sims/'
directory_output = '/home/philipp/post_output/'

#First test
special_name=''
N_it_min = 5



reg_var = 'w'
ddflag = 3
anom_flag = 0


#Starting with chord properties in 3d and 1d:
ddflag  = 3
ct_flag = 1



print('###########################################################################################################')
print('starting 3d proc chords')
print('###########################################################################################################')
proc_chords(        date_str=date_str  , data_dim_flag=ddflag , special_name=special_name , chord_times=ct_flag,
                                         directory_input=directory_input , directory_output=directory_output,
                                         N_it_min=N_it_min)#,N_it_max=N_it_max)

print('###########################################################################################################')
print('finished 3d proc chords')
print('###########################################################################################################')
print('starting 1d proc chords')
print('###########################################################################################################')

ddflag  = 1
ct_flag = 0
proc_chords(        date_str=date_str  , data_dim_flag=ddflag , special_name=special_name , chord_times=ct_flag,
                                         directory_input=directory_input , directory_output=directory_output)
print('###########################################################################################################')
print('finished 1d proc chords')
print('###########################################################################################################')
print('starting 3d beard regularization')
print('###########################################################################################################')

#Now various beards, starting with the extra wide 3D and 1D w
#For 3D we skip the first 3 hours to hopefully avoid spin up issues
ddflag        = 3
szb_flag      = 1
curtain_extra = 8
ct_flag       = 1
reg_var       = 'w'
proc_beard_regularize(date_str=date_str        ,reg_var=reg_var,data_dim_flag=ddflag,special_name=special_name,
                                                chord_times=ct_flag,boundary_scaling_flag=0,size_bin_flag=szb_flag,
                                                curtain_extra=curtain_extra,anomaly_flag = anom_flag,
                                                directory_input=directory_input,directory_output=directory_output,
                                                N_it_min=N_it_min)#,N_it_max=N_it_max)
print('###########################################################################################################')
print('finished 3d beard regularization')
print('###########################################################################################################')
print('starting 1d beard regularization, qt and w')
print('###########################################################################################################')

#Next the 1D columns 
ddflag        = 1
ct_flag       = 0
proc_beard_regularize(date_str=date_str        ,reg_var=reg_var,data_dim_flag=ddflag,special_name=special_name,
                                                chord_times=ct_flag,boundary_scaling_flag=0,size_bin_flag=szb_flag,
                                                curtain_extra=curtain_extra,anomaly_flag = anom_flag,
                                                directory_input=directory_input,directory_output=directory_output)

anom_flag = 1
reg_var   = 'qt'

proc_beard_regularize(date_str=date_str        ,reg_var=reg_var,data_dim_flag=ddflag,special_name=special_name,
                                                chord_times=ct_flag,boundary_scaling_flag=0,size_bin_flag=szb_flag,
                                                curtain_extra=curtain_extra,anomaly_flag = anom_flag,
                                                directory_input=directory_input,directory_output=directory_output)

print('###########################################################################################################')
print('finished 1d beard regularization, qt and w')
print('###########################################################################################################')
print('starting 1d single column beard regularization')
print('###########################################################################################################')

#Now the individual column chords, with no more needed extra thick curtain or individual size bins
#This is a bit of a botch, but I'll use glob to get the number of columns, than loop over that. 
anom_flag     = 0
special_name  = 'singles_test'
reg_var       = 'w'
curtain_extra = 1
szb_flag      = 0
ddflag        = 1

column_files =     glob.glob(directory_input+date_str+'/*column?0*.nc')

print(column_files)


for n in range(len(column_files)):
    proc_beard_regularize(date_str=date_str    ,reg_var=reg_var,data_dim_flag=ddflag,special_name=special_name,
                                                chord_times=ct_flag,boundary_scaling_flag=0,size_bin_flag=szb_flag,
                                                curtain_extra=curtain_extra,anomaly_flag = anom_flag,
                                                directory_input=directory_input,directory_output=directory_output,
                                                N_it_min=n,N_it_max=n+1)
print('###########################################################################################################')
print('finished 1d single column beard regularization')
print('###########################################################################################################')
print('strating w and wstar pdf ')
print('###########################################################################################################')

#Lastly the the wpdfs at cloud base from the 3D fields. One with and one without star scaling
special_name='postpro_test'
proc_pdf(date_str=date_str,data_dim_flag=3,special_name=special_name,boundary_scaling_flag=1,
                                                directory_input=directory_input,directory_output=directory_output,
                                                N_it_min=N_it_min)#,N_it_max=N_it_max)
proc_pdf(date_str=date_str,data_dim_flag=3,special_name=special_name,boundary_scaling_flag=0,
                                                directory_input=directory_input,directory_output=directory_output,
                                                N_it_min=N_it_min)#,N_it_max=N_it_max)
print('###########################################################################################################')
print('finished w and wstar pdf ')
print('###########################################################################################################')


