import deriveeC
import numpy as np
name_param = [r'$H_0$',r'$\Omega_bh^2$',r'$\Omega_ch^2$',r'$A_s$',r'$n_s$',r'$\tau$']
import matplotlib.pyplot as plt
import sys, platform, os
from ellipse import Ellipse

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))/(np.sqrt(2*np.pi*sig*sig))


def fischer(theta,pts):
    # Calcule la matrice de fisher à partir des paramètres cosmologiques choisis (et du nombre de points)
    deriv = deriveeC.dCL(theta,pts)
    n_p = len(theta)
    F = np.zeros((n_p,n_p))
    for i in range(n_p):
        for j in range(n_p):
            ls = np.arange(len(deriv[0]))
            NL = np.zeros(len(deriv[0]))
            pf1 = 0.6*(2*ls[2:]+1)/2
            pf2 = 1/((deriveeC.CL(theta,pts)[2:]+NL[2:])**2)
            F[i,j] = np.sum(pf1*pf2*deriv[i][2:]*deriv[j][2:])
    return(F)

def cov2corr(cov,remove_diag=True):
    d= np.sqrt(cov.diagonal())
    corr = ((cov.T/d).T)/d
    if remove_diag:
        corr -= np.identity(cov.shape[0])
    return(corr)

def constraints(theta,pts):
    ### Renvoie les contraintes sur les parametres sous la forme [Standard deviations, Covariance matrix, Correlation matrix]
    F = fischer(theta,pts)
    C = np.linalg.inv(F)
    #Cp = np.copy(C)
    stds = []
    for i in range(len(C)):
        stds.append(np.sqrt(C[i,i]))
        #for j in range(len(C)):
            #Cp[i,j] = C[i,j]/np.sqrt(C[i,i]*C[j,j])
    Cp = cov2corr(C,remove_diag=False)
    if len(theta) == 5:
        print("Stds for [H0,Ombh2,Omch2,As,ns] : ",stds)
    else:
        print("Stds for [H0,Ombh2,Omch2,As,ns,tau] : ",stds)
    print("Correlation : ",Cp)
    print("Covariance : ",C)
    return(stds,C,Cp)

planck_values = [1.4,0.00033,0.0031,1.075e-10,0.0094,0.038]
def plot_planck(theta,pts):
    std_l= constraints(theta,pts)[0]
    for i in range(len(theta)):
        plt.figure()
        plt.grid(True,linestyle='dotted')
        stds = std_l[i]
        abs = np.linspace(theta[i]-4*planck_values[i],theta[i]+4*planck_values[i],500)
        ord = gaussian(abs,theta[i],stds)
        ord2 = gaussian(abs,theta[i],planck_values[i])
        plt.plot(abs,ord,color='darkred',label ="Fischer")
        plt.plot(abs,ord2,color='darkblue',label="Planck")
        plt.xlabel(name_param[i])
        plt.legend()
        if not(os.path.isdir('Figures/error_figs')):
            os.mkdir('Figures/error_figs/')
        plt.savefig('Figures/error_figs/error'+str(i)+".png",dpi=300)

def plot_subplanck(theta,pts):
    std_l= constraints(theta,pts)[0]
    fig = plt.figure(figsize=(15,7.5))
    for i in range(len(theta)):
        ax = fig.add_subplot(231+i)
        ax.grid(True,linestyle='dotted')
        stds = std_l[i]
        textstr = r'$\frac{\sigma_p}{\sigma_F} =$ '+str(round(planck_values[i]/stds,3))
        props = dict(boxstyle='round',facecolor='white')
        ax.text(0.05,0.95,textstr,transform=ax.transAxes,verticalalignment='top',bbox=props)
        abs = np.linspace(theta[i]-4*planck_values[i],theta[i]+4*planck_values[i],500)
        ord = gaussian(abs,theta[i],stds)
        ord2 = gaussian(abs,theta[i],planck_values[i])
        ax.plot(abs,ord/np.max(ord),color='darkred',label ="Fischer")
        ax.plot(abs,ord2/np.max(ord2),color='darkblue',label="Planck")
        ax.set_xlabel(name_param[i])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(r'$L_{max}$ = '+str(pts))
    if not(os.path.isdir('Figures/error_figs_tau')):
        os.mkdir('Figures/error_figs_tau/')
    fig.savefig('Figures/error_figs_tau/error_plot.png',dpi=300)

def ellipseplot(theta,pts):
    ### Trace les ellipses de corrélation des paramètres pour les param cosmo theta
    covarmat = constraints(theta,pts)[1]
    fig,axes = plt.subplots(figsize=(15,15),ncols=len(theta)-1,nrows=len(theta)-1)
    for i in range(len(theta)-1):
        for j in range(len(theta)-1):
            if i<j:
                axes[i,j].axis('off')
            else:
                a2=(covarmat[i+1,i+1]+covarmat[j,j])/2 + np.sqrt((covarmat[i+1,i+1]-covarmat[j,j])**2/4 + covarmat[i+1,j]**2)
                b2=(covarmat[i+1,i+1]+covarmat[j,j])/2 - np.sqrt((covarmat[i+1,i+1]-covarmat[j,j])**2/4 + covarmat[i+1,j]**2)
                a = 1.52*np.sqrt(a2)
                b = 1.52*np.sqrt(b2)
                print(a)
                print(b)
                tan2T=2*covarmat[i+1,j]/(covarmat[j,j]-covarmat[i+1,i+1])
                T=0.5*np.arctan(tan2T)
                print(T*180/np.pi)
                ellipse = Ellipse((theta[j],theta[i+1]),a,b,T)
                axes[i,j].plot(ellipse[0],ellipse[1],color='darkred', lw=2)
                if j==0:
                    axes[i,j].set_ylabel(name_param[i+1])
                if i==4:
                    axes[i,j].set_xlabel(name_param[j])
                if j!=0:
                    axes[i,j].set_yticklabels([])
                if i!=4:
                    axes[i,j].set_xticklabels([])
    fig.savefig('Figures/error_figs_tau/cov_plot.png',dpi=300)
