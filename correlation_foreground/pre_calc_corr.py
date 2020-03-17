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

def noise_freq(freq_list,noise_data_path,pts):
    NL_diag_T = []
    for element in freq_list:
        NL_T = np.loadtxt(noise_data_path+"noise_t_LAT_"+str(element)+"xLAT_"+str(element)+".dat")[:,1][:(pts)+51-2]
        NL_diag_T.append(NL_T)
    NL_ndiag_T = []
    NL_ndiag_T.append(np.loadtxt(noise_data_path+"noise_t_LAT_27xLAT_39.dat")[:,1][:(pts)+51-2])
    NL_ndiag_T.append(np.loadtxt(noise_data_path+"noise_t_LAT_93xLAT_145.dat")[:,1][:(pts)+51-2])
    NL_ndiag_T.append(np.loadtxt(noise_data_path+"noise_t_LAT_225xLAT_280.dat")[:,1][:(pts)+51-2])
    NL_t = []
    for a in range(pts+51-2):
        Nmatrix = np.zeros((len(freq_list),len(freq_list)))
        for i in range(len(freq_list)):
            if i==1:
                Nmatrix[i,i] = NL_diag_T[i][a]
            else:
                Nmatrix[i,i] = NL_diag_T[i][a]
            for j in range(i):
                if j==2 and i==3:
                    Nmatrix[i,j] = NL_ndiag_T[0][a]
                    Nmatrix[j,i] = NL_ndiag_T[0][a]
                if j==0 and i==1:
                    Nmatrix[i,j] = NL_ndiag_T[1][a]
                    Nmatrix[j,i] = NL_ndiag_T[1][a]
                if j==4 and i==5:
                    Nmatrix[i,j] = NL_ndiag_T[2][a]
                    Nmatrix[j,i] = NL_ndiag_T[2][a]
        NL_t.append(Nmatrix)
    return(np.array(NL_t))

def CL(theta,fg_parameters,pts,freq_list,dict_path,N,mnu,omk,r):

    d = so_dict.so_dict()
    d.read_from_file(dict_path+"global_healpix_example.dict")
    fg_norm = d["fg_norm"]
    components = {"tt": d["fg_components"],"ee": [],"te": []}
    fg_model = {"normalisation": fg_norm,"components": components}
    fg_params = {'a_tSZ': fg_parameters[0],'a_kSZ': fg_parameters[1],'a_p': fg_parameters[2],'beta_p': fg_parameters[3],
                    'a_c': fg_parameters[4],'beta_c': fg_parameters[5],'n_CIBC': 1.2,'a_s': fg_parameters[6],
                    'T_d': 9.6}
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=theta[0],ombh2=theta[1],omch2=theta[2],mnu=mnu,omk=omk,tau=theta[5])
    pars.InitPower.set_params(As=theta[3],ns=theta[4],r=r)
    pars.set_for_lmax(pts,lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars,CMB_unit="muK")
    totCL=powers['total']

    TT = totCL[:,0][2:]
    ls = np.arange(len(TT)+2)[2:]
    CL = []
    #NL = noise_freq(freq_list,noise_data_path,pts)
    NL = [element[:len(freq_list),:len(freq_list)] for element in N]
    fg_dict = mfl.get_foreground_model(fg_params,fg_model,freq_list,ls)
    for a in range(len(ls)):
        Cmatrix = np.ones((len(freq_list),len(freq_list)))*TT[a]
        for i in range(len(freq_list)):
            for j in range(i+1):
                Cmatrix[i,j] = Cmatrix[i,j] + NL[a][i,j] + fg_dict['tt','all',freq_list[i],freq_list[j]][a]
                Cmatrix[j,i] = Cmatrix[j,i] + NL[a][j,i] + fg_dict['tt','all',freq_list[j],freq_list[i]][a]
        CL.append(Cmatrix)
    return(np.array(CL))

def dCL(theta,fg_parameters,pts,freq_list,dict_path,N,mnu,omk,r):
    length1 = len(theta)
    length2 = len(fg_parameters)
    epsilon_theta = theta/100
    epsilon_fg = fg_parameters/100
    var_temp = []
    for i in range(length1):
        eps = epsilon_theta[i]*np.eye(1,length1,i)
        eps = eps.flatten()
        CL_plus = CL(theta+eps,fg_parameters,pts,freq_list,dict_path,N,mnu,omk,r)
        CL_moins = CL(theta-eps,fg_parameters,pts,freq_list,dict_path,N,mnu,omk,r)
        der = (CL_plus-CL_moins)/(2*epsilon_theta[i])
        var_temp.append(der)
    for j in range(length2):
        eps = epsilon_fg[j]*np.eye(1,length2,j)
        eps = eps.flatten()
        CL_plus = CL(theta,fg_parameters+eps,pts,freq_list,dict_path,N,mnu,omk,r)
        CL_moins = CL(theta,fg_parameters-eps,pts,freq_list,dict_path,N,mnu,omk,r)
        der = (CL_plus-CL_moins)/(2*epsilon_fg[j])
        var_temp.append(der)
    return(var_temp)

def pre_calculation(theta,fg_parameters,pts,freq_list,dict_path,noise_data_path,mnu,omk,r,save_path,names):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    N = noise_freq(freq_list,noise_data_path,pts)
    np.save(save_path+'noise',N)
    Cl = CL(theta,fg_parameters,pts,freq_list,dict_path,N,mnu,omk,r)
    np.save(save_path+'CL',Cl)
    deriv = dCL(theta,fg_parameters,pts,freq_list,dict_path,N,mnu,omk,r)
    for i in range(len(names)):
        np.save(save_path+'deriv_'+names[i],deriv[i])
