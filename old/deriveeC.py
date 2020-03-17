import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import camb
from camb import model, initialpower

#HO,Omb,Omc,As,ns
parameters = np.array([67.5,0.022,0.122,2e-9,0.965,0.06])
planck_parameters = np.array([67.4,0.02207,0.1196,2.22645e-9,0.9616,0.097])
name_param = ['$H_0$','$\Omega_bh^2$','$\Omega_ch^2$','$A_s$','$n_s$','$\tau$']
tau = 0.06
mnu = 0.06
omk = 0
r = 0

def CL(theta,pts):
    ## Calcule les CL à partir de la liste des parametres cosmo theta
    pars = camb.CAMBparams()
    if len(theta) == 5:
        pars.set_cosmology(H0=theta[0],ombh2=theta[1],omch2=theta[2],mnu=mnu,omk=omk,tau=tau)
    else:
        pars.set_cosmology(H0=theta[0],ombh2=theta[1],omch2=theta[2],mnu=mnu,omk=omk,tau=theta[5])
    pars.InitPower.set_params(As=theta[3],ns=theta[4],r=r)
    pars.set_for_lmax(pts,lens_potential_accuracy=0);
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars,CMB_unit="muK")
    totCL=powers['total']
    totCL = totCL[:,0]
    return(totCL)

def dCL(theta,pts):
    ## Calcule la derivée des Cl par raport à chaque parametre cosmo et renvoie une matrice de taille nb_param*pts (pts = nbr pts)
    length = len(theta)
    epsilon = theta/100
    L = []
    for i in range(length):
        eps = epsilon[i]*np.eye(1,length,i)
        eps = eps.flatten()
        CL2 = CL(theta+eps,pts)
        CL3 = CL(theta-eps,pts)
        der = (CL2-CL3)/epsilon[i]/2
        L.append(der)
    L = np.array(L)
    return(L)

def plot_der(theta):
    if not(os.path.isdir('Figures/der_figs')):
        os.mkdir('Figures/der_figs/')
    deriv = dCL(theta,pts)
    print(deriv)
    for i in range(len(deriv)):
        plt.figure()
        plt.grid(True,linestyle='dotted')
        ls = np.arange(len(deriv[i]))
        plt.plot(ls,deriv[i],color="darkred")
        plt.xlabel(r"$l$")
        plt.ylabel(r'Dérivée par rapport à '+name_param[i])
        plt.savefig('Figures/der_figs/'+name_param[i]+'.png',dpi=300)
