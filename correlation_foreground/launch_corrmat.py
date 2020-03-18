import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
from toolbox import *
from corrmat_tools import *


#path = "/home/pspipe/workspace/PSpipe/project/maps2params/test/"
noise_data_path = "sim_data/noise_tot_test/"
save_path_fig='Saves/Figures/'
save_path_dat='Saves/Datas/'

if not os.path.isdir(save_path_dat):
    os.makedirs(save_path_dat)
if not os.path.isdir(save_path_fig):
    os.makedirs(save_path_fig)


name_param_corrmat = [r'$H_0$',r'$\Omega_bh^2$',r'$\Omega_ch^2$',r'$A_s$',r'$n_s$',r'$\tau$',r'$a_{tSZ}$',r'$a_{kSZ}$',r'$a_p$',r'$\beta_p$',r'$a_c$',r'$\beta_c$',r'$a_s$']
names = ['H0','Ombh2','Omch2','As','ns','tau','atSZ','akSZ','ap','betap','ac','betac','as']
planck_parameters = np.array([67.4,0.02207,0.1196,2.22645e-9,0.9616,0.097])
fg_parameters = np.array([3.3,1.66,6.91,2.07,4.88,2.2,3.09])
cl_path = 'pre_calc/'
if not os.path.isdir(cl_path):
    os.makedirs(cl_path)
frequency_list = [145,93,39,27,225,280]

mnu = 0.06
omk = 0
r = 0
fsky = 0.4
pts = 4449

################################################################################

calculate_data = False
plot_correlation = True
plot_fisher = False
plot_cosmo_parameters = True

if calculate_data:
    from pre_calc_corr import *
    pre_calculation(planck_parameters,fg_parameters,pts,frequency_list,noise_data_path,mnu,omk,r,cl_path,names)

if plot_correlation:
    corrmat_evol(frequency_list,name_param_corrmat,save_path_fig,save_path_dat,fsky,names,cl_path)

if plot_fisher:
    fisher_norm_evol(frequency_list,name_param_corrmat,save_path_fig,fsky,names,cl_path)

if plot_cosmo_parameters:
    cosmo_parameters(planck_parameters,fg_parameters,frequency_list,name_param_corrmat,save_path_fig,save_path_dat)
