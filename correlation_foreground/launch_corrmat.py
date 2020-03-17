import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
import camb
from camb import model,initialpower
from toolbox import *
from corrmat_tools import *
from pre_calc_corr import *

#path = "/home/pspipe/workspace/PSpipe/project/maps2params/test/"
noise_data_path = "sim_data/noise_tot_test/"
save_path='Figures/fisher_fg/'

name_param_corrmat = [r'$H_0$',r'$\Omega_bh^2$',r'$\Omega_ch^2$',r'$A_s$',r'$n_s$',r'$\tau$',r'$a_{tSZ}$',r'$a_{kSZ}$',r'$a_p$',r'$\beta_p$',r'$a_c$',r'$\beta_c$',r'$a_s$']
names = ['H0','Ombh2','Omch2','As','ns','tau','atSZ','akSZ','ap','betap','ac','betac','as']
planck_parameters = np.array([67.4,0.02207,0.1196,2.22645e-9,0.9616,0.097])
fg_parameters = np.array([3.3,1.66,6.91,2.07,4.88,2.2,3.09])
cl_path = 'pre_calc/'
frequency_list = [145,93,39,27,225,280]

mnu = 0.06
omk = 0
r = 0
fsky = 0.4
pts = 4449

################################################################################

### Enlever le # pour pré-calculer les CLs et les dérivées
#pre_calculation(planck_parameters,fg_parameters,pts,frequency_list,path,noise_data_path,mnu,omk,r,cl_path,names)

corrmat_evol(frequency_list,name_param_corrmat,save_path,fsky,names,cl_path)
