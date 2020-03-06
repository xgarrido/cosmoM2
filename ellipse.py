import numpy as np
from matplotlib import pyplot as plt
from math import pi, cos, sin

### Petite fonction pour tracer des ellipses

def Ellipse(center,a,b,t_rot):
    u,v = center
    t = np.linspace(0, 2*pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])
    #u,v removed to keep the same center location
    R_rot = np.array([[cos(t_rot) , -sin(t_rot)],[sin(t_rot) , cos(t_rot)]])
    #2-D rotation matrix
    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])
    return(u+Ell_rot[0,:],v+Ell_rot[1,:])
