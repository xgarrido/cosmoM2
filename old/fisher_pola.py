import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower

path = 'Figures/polar/'
parameters = np.array([67.5,0.022,0.122,2e-9,0.965,0.06])
planck_parameters = np.array([67.4,0.02207,0.1196,2.22645e-9,0.9616,0.097])
name_param = ['$H_0$','$\Omega_bh^2$','$\Omega_ch^2$','$A_s$','$n_s$','$\tau$']
mnu = 0.06
omk = 0
r = 0

def CL_mat(theta,pts,NL):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=theta[0],ombh2=theta[1],omch2=theta[2],mnu=mnu,omk=omk,tau=theta[5])
    pars.InitPower.set_params(As=theta[3],ns=theta[4],r=r)
    pars.set_for_lmax(pts,lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars,CMB_unit="muK")
    totCL=powers['total']
    ls = np.arange(totCL.shape[0])
    TT = totCL[:,0]
    EE = totCL[:,1]
    TE = totCL[:,3]
    CL = []
    for element in ls:
        Cmatrix = np.zeros((2,2))
        Cmatrix[0,0] = TT[element] + NL[element]
        Cmatrix[1,1] = EE[element] + NL[element]
        Cmatrix[0,1], Cmatrix[1,0] = TE[element],TE[element]
        CL.append(Cmatrix)
    return(np.array(CL)[2:])

def dCL_mat(theta,pts,NL):
    length = len(theta)
    epsilon = theta/100
    temp = []
    for i in range(length):
        eps = epsilon[i]*np.eye(1,length,i)
        eps = eps.flatten()
        CLp = CL_mat(theta+eps,pts,NL)
        #print(np.shape(CLp))
        CLm = CL_mat(theta-eps,pts,NL)
        der = (CLp-CLm)/(2*epsilon[i])
        temp.append(der)
    return(np.array(temp))
