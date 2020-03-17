import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
import camb
from camb import model,initialpower
from toolbox import *
from pspy import pspy_utils, so_dict
from itertools import combinations_with_replacement as cwr
import so_noise_calculator_public_20180822 as noise_calc
from copy import deepcopy
import mflike as mfl
import time

def fisher(freq_list,fsky,names,cl_path):
    start_time = time.time()
    deriv = [np.load(cl_path+"deriv_"+names[i]+".npy") for i in range(len(names))]
    C_ell = np.load(cl_path+"CL.npy")
    print("Importation : %s secondes" % (time.time() - start_time))
    lenght = len(deriv)
    Nf = len(freq_list)
    F = np.zeros((lenght,lenght))
    start_time = time.time()
    for i in range(lenght):
        for j in range(i+1):
            ls = np.arange(len(C_ell)+2)[2:]
            pf1 = fsky*(2*ls+1)/2
            matprod = np.array([np.trace(np.linalg.inv(C_ell[a][:Nf,:Nf]).dot(deriv[i][a][:Nf,:Nf].dot(np.linalg.inv(C_ell[a][:Nf,:Nf]).dot(deriv[j][a][:Nf,:Nf])))) for a in range(len(C_ell))])
            F[i,j] = np.sum(pf1*matprod)
            F[j,i] = np.sum(pf1*matprod)
    print("Construction de Fisher : %s secondes" % (time.time() - start_time))
    #print(np.linalg.eigvals(F))
    return(F)

def constraints(freq_list,fsky,names,cl_path):
    F = fisher(freq_list,fsky,names,cl_path)
    C = np.linalg.inv(F)
    return(C)

def corrmat_evol(freq_list,name_param,save_path,fsky,names,cl_path):
    fig = plt.figure(figsize=(24,13.5))

    for i in range(len(freq_list)):
        print("Debut de la boucle a %s frequences" %(i+1))
        start_time_boucle = time.time()
        covar = constraints(freq_list[:i+1],fsky,names,cl_path)
        np.savetxt(save_path+str(freq_list[:i+1])+".dat",covar)
        corr = cov2corr(covar,remove_diag=False)
        ax = fig.add_subplot(231+i)
        im = ax.imshow(corr,vmin=-1,vmax=+1,cmap='seismic')
        ax.set_xticks(np.arange(0, len(name_param), 1));
        ax.set_yticks(np.arange(0, len(name_param), 1));
        ax.set_xticklabels(name_param)
        ax.set_yticklabels(name_param)
        ax.set_title(r'f = '+str(freq_list[:i+1]))
        print("Fin de la boucle, temps d'execution : %s secondes" % (time.time() - start_time_boucle))
    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    fig.savefig(save_path+"corrmat_var.png",dpi=300)
