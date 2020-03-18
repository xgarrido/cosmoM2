import sys, platform, os
import matplotlib.pyplot as plt
import numpy as np
from toolbox import *
import time

def compute_fisher(freq_list, fsky,names, cl_path):

    start_time = time.time()

    deriv = [np.load(cl_path + "deriv_" + names[i] + ".npy") for i in range(len(names))]
    C_ell = np.load(cl_path + "CL.npy")

    print("Importation : %s secondes" % (time.time() - start_time))

    n_params = len(deriv)
    n_freqs = len(freq_list)
    fisher = np.zeros((n_params, n_params))

    start_time = time.time()

    for ell in range(len(C_ell)):
        trace_fisher = []
        ls = np.arange(len(C_ell) + 2)[2:]
        pre_fact = fsky * (2 * ls + 1)/2
        inverse_c_ell = np.linalg.inv(C_ell[ell][:n_freqs,:n_freqs])
        for i in range(n_params):
            for j in range(n_params):
                trace_mat_prod = np.trace(inverse_c_ell.dot(deriv[i][ell][:n_freqs, :n_freqs].dot(inverse_c_ell.dot(deriv[j][ell][:n_freqs, :n_freqs]))))
                fisher[i, j] += pre_fact[ell] * trace_mat_prod

    print("Construction de Fisher : %s secondes" % (time.time() - start_time))

    #print(np.linalg.eigvals(F))
    return(fisher)

def constraints(freq_list, fsky,names, cl_path):

    fisher = compute_fisher(freq_list, fsky, names, cl_path)
    print(np.linalg.eigvals(fisher))
    cov = np.linalg.inv(fisher)

    return(cov)

def norm_fisher(freq_list, fsky, names, cl_path):

    fisher = compute_fisher(freq_list, fsky, names, cl_path)
    norm_fisher = cov2corr(fisher, remove_diag=False)
    eigen = np.linalg.eigvals(fisher)
    #print("\n")
    #print("Normalized Fisher Matrix : ",norm_F)
    #print("\n")
    #print("Eigenvalues : ",eigen)
    return(norm_fisher)

def corrmat_evol(freq_list, name_param, save_path_fig, save_path_dat, fsky,names, cl_path):

    fig = plt.figure(figsize=(24, 13.5))
    sig_mat = []
    n_freqs = len(freq_list)
    for i in range(1, n_freqs):

        print("\n")
        print("-" * 15)
        print("Debut de la boucle a %s frequences" %(i + 1))

        start_time_boucle = time.time()
        covar = constraints(freq_list[:i + 1], fsky, names, cl_path)
        sig = np.sqrt(np.diagonal(covar))
        sig_mat.append(sig)
        np.savetxt(save_path_dat + str(freq_list[:i + 1]) + ".dat", covar)
        corr = cov2corr(covar, remove_diag=False)
        ax = fig.add_subplot(231 + i)
        im = ax.imshow(corr, vmin= - 1, vmax= + 1, cmap='seismic')
        ax.set_xticks(np.arange(0, len(name_param), 1));
        ax.set_yticks(np.arange(0, len(name_param), 1));
        ax.set_xticklabels(name_param)
        ax.set_yticklabels(name_param)
        ax.set_title(r'f = ' + str(freq_list[:i + 1]))

        print("Fin de la boucle, temps d'execution : %s secondes" % (time.time() - start_time_boucle))
        print("-" * 15)

    cb_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    sig_mat = np.array(sig_mat)
    np.savetxt(save_path_dat + "sigmas.dat",sig_mat)
    fig.savefig(save_path_fig + "corrmat_var.png", dpi=300)

def fisher_norm_evol(freq_list, name_param, save_path_fig, fsky, names, cl_path):

    n_freqs = freq_list
    for a in range(n_freqs):
        fig = plt.figure(figsize=(24, 24))
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        print("\n")
        print("-" * 15)
        print("Debut de la boucle a %s frequences" %(a + 1))
        start_time_boucle = time.time()
        Fnorm = norm_fisher(freq_list[:a + 1], fsky, names, cl_path)
        ax = fig.add_subplot(111)
        im = ax.imshow(Fnorm, vmin= - 1, vmax= + 1, cmap='seismic')
        ax.set_xticks(np.arange(0, len(name_param), 1));
        ax.set_yticks(np.arange(0, len(name_param), 1));
        ax.set_xticklabels(name_param)
        ax.set_yticklabels(name_param)
        ax.set_title(r'f = ' + str(freq_list[:a + 1]), fontsize=20)
        for i in range(len(Fnorm)):
            for j in range(len(Fnorm)):
                ax.text(j, i, "{:0.3f}".format(Fnorm[i, j]), horizontalalignment="center",
                     color="black",fontsize=20)
        print("Fin de la boucle, temps d'execution : %s secondes" % (time.time() - start_time_boucle))
        print("-" * 15)
        fig.tight_layout()
        fig.savefig(save_path_fig + "fisher_norm_var" + str(a) + ".png",dpi=300)

def cosmo_parameters(theta, fg_parameters, freq_list, name_param, save_path_fig, save_path_dat):

    fig = plt.figure(figsize=(24, 13.5))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    colors = ['darkred', 'darkorange', 'darkgreen', 'darkblue', 'darkviolet']
    for i in range(len(theta)):
        sigma = np.loadtxt(save_path_dat + "sigmas.dat")[:, i]
        ax = fig.add_subplot(231 + i)
        ax.grid(True,linestyle='--')
        abs = np.linspace(theta[i] - 4 * np.max(sigma), theta[i] + 4 * np.max(sigma),  500)
        for j in range(len(sigma)):
            ord = gaussian(abs, theta[i], sigma[j])
            ax.plot(abs, ord/np.max(ord), label=r'$N_{freq}$ = %s' %(j + 2),color=colors[j])
        ax.set_title(name_param[i], fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=18)
    fig.tight_layout()
    fig.savefig(save_path_fig + "forecast_var.png",dpi=300)

    fig = plt.figure(figsize=(20, 20))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    colors = ['darkred', 'darkorange', 'darkgreen', 'darkblue', 'darkviolet']
    for i in range(len(fg_parameters)):
        sigma = np.loadtxt(save_path_dat + "sigmas.dat")[:, i + 6]
        ax = fig.add_subplot(331 + i)
        ax.grid(True, linestyle='--')
        abs = np.linspace(fg_parameters[i] - 4 * np.max(sigma), fg_parameters[i] + 4 * np.max(sigma), 500)
        for j in range(len(sigma)):
            ord = gaussian(abs, fg_parameters[i], sigma[j])
            ax.plot(abs, ord/np.max(ord), label=r'$N_{freq}$ = %s' %(j + 2),color=colors[j])
        ax.set_title(name_param[i + 6], fontsize=20)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.6,0.1), fontsize=24)
    fig.tight_layout()
    fig.savefig(save_path_fig + "forecast_var_fg.png",dpi=300)
