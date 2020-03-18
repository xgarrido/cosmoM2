import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
import camb
from camb import model,initialpower
from toolbox import *
from pspy import pspy_utils, so_dict
import mflike as mfl

def noise_freq(freq_list,noise_data_path,ell_max):
    noise_list_diag = []

    for element in freq_list:
        noise_t_spectrum = np.loadtxt(noise_data_path+"noise_t_LAT_"+str(element)+"xLAT_"+str(element)+".dat")[:,1][:ell_max-1]
        noise_list_diag.append(noise_t_spectrum)

    noise_list_matrix = []

    for a in range(ell_max-1):
        noise_matrix = np.zeros((len(freq_list),len(freq_list)))
        for i in range(len(freq_list)):
            noise_matrix[i,i] = noise_list_diag[i][a]
        noise_list_matrix.append(noise_matrix)

    return(np.array(noise_list_matrix))

def CL(theta,fg_parameters,ell_max,freq_list,noise_list_matrix,mnu,omk,r):

    d = so_dict.so_dict()
    d.read_from_file("global_healpix_example.dict")
    fg_norm = d["fg_norm"]
    components = {"tt": d["fg_components"],"ee": [],"te": []}
    fg_model = {"normalisation": fg_norm,"components": components}
    fg_params = {'a_tSZ': fg_parameters[0],'a_kSZ': fg_parameters[1],'a_p': fg_parameters[2],'beta_p': fg_parameters[3],
                    'a_c': fg_parameters[4],'beta_c': fg_parameters[5],'n_CIBC': 1.2,'a_s': fg_parameters[6],
                    'T_d': 9.6}

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=theta[0],ombh2=theta[1],omch2=theta[2],mnu=mnu,omk=omk,tau=theta[5])
    pars.InitPower.set_params(As=theta[3],ns=theta[4],r=r)
    pars.set_for_lmax(ell_max-1,lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars,CMB_unit="muK")
    totCL=powers['total']

    TT = totCL[:,0][2:]
    ell_list = np.arange(len(TT)+2)[2:]

    power_spectrum_matrix = []

    noise_list_matrix = [element[:len(freq_list),:len(freq_list)] for element in noise_list_matrix]

    fg_dict = mfl.get_foreground_model(fg_params,fg_model,freq_list,ell_list)

    for a in range(len(ell_list)):
        c_matrix = np.ones((len(freq_list),len(freq_list)))*TT[a]
        for i in range(len(freq_list)):
            for j in range(i+1):
                c_matrix[i,j] = c_matrix[i,j] + noise_list_matrix[a][i,j] + fg_dict['tt','all',freq_list[i],freq_list[j]][a]
                c_matrix[j,i] = c_matrix[j,i] + noise_list_matrix[a][j,i] + fg_dict['tt','all',freq_list[j],freq_list[i]][a]
        power_spectrum_matrix.append(c_matrix)

    return(np.array(power_spectrum_matrix))

def dCL(theta,fg_parameters,ell_max,freq_list,noise_list_matrix,mnu,omk,r):

    length_cosmo = len(theta)
    length_fg = len(fg_parameters)
    epsilon_cosmo = theta/100
    epsilon_fg = fg_parameters/100
    var_temp = []

    for i in range(length_cosmo):
        eps = epsilon_cosmo[i]*np.eye(1,length_cosmo,i)
        eps = eps.flatten()
        CL_plus = CL(theta+eps,fg_parameters,ell_max,freq_list,noise_list_matrix,mnu,omk,r)
        CL_moins = CL(theta-eps,fg_parameters,ell_max,freq_list,noise_list_matrix,mnu,omk,r)
        der = (CL_plus-CL_moins)/(2*epsilon_cosmo[i])
        var_temp.append(der)

    for j in range(length_fg):
        eps = epsilon_fg[j]*np.eye(1,length_fg,j)
        eps = eps.flatten()
        CL_plus = CL(theta,fg_parameters+eps,ell_max,freq_list,noise_list_matrix,mnu,omk,r)
        CL_moins = CL(theta,fg_parameters-eps,ell_max,freq_list,noise_list_matrix,mnu,omk,r)
        der = (CL_plus-CL_moins)/(2*epsilon_fg[j])
        var_temp.append(der)

    return(var_temp)

def pre_calculation(theta,fg_parameters,ell_max,freq_list,noise_data_path,mnu,omk,r,save_path,names):

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    noise_list_matrix = noise_freq(freq_list,noise_data_path,ell_max)
    np.save(save_path+'noise',noise_list_matrix)

    power_spectrums = CL(theta,fg_parameters,ell_max,freq_list,noise_list_matrix,mnu,omk,r)
    np.save(save_path+'CL',power_spectrums)

    deriv = dCL(theta,fg_parameters,ell_max,freq_list,noise_list_matrix,mnu,omk,r)
    for i in range(len(names)):
        np.save(save_path+'deriv_'+names[i],deriv[i])
