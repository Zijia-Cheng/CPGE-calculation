import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit(nopython = True)
def levi_civita(x1, x2, x3):
    if x1 == x2 or x2 == x3 or x1 == x3:
        return 0  # Two or more indices are equal, so the symbol is 0.
    elif (x1, x2, x3) in [(0, 1, 2), (2, 0, 1), (1, 2, 0)]:
        return 1  # The sequence is an even permutation of (0, 1, 2).
    else:
        return -1  # The sequence is an odd permutation of (0, 1, 2).



@jit(nopython = True)
def calculate_cpge(GRID, energy, H_k, idc_i, idc_j, kx_GRID, ky_GRID, kz_GRID, differ_GRID = True, H_k_d = None):
    """
    energy is photon energy
    H_k is a function that return the Hamiltonian at a given k point
    delta is the linewidth
    idc_i : first index of beta_ij
    dic_j : second index of beta_ij
    i,j are the indices x : 0, y : 1, z : 2
    The calculatiion is done in the zero temperature limit
    differ_GRID is a boolean that determines if the derivative is calculated using the central difference or the analytical derivative
    H_k_d is a function that returns the derivative of the Hamiltonian at a given k point, if differ_GRID is False

    """
    sigma_abc = 0 + 0j
    delta = 0.0025 #delta is the linewidth
    stepa = kx_GRID[1] - kx_GRID[0]
    stepb = ky_GRID[1] - ky_GRID[0]
    stepc = kz_GRID[1] - kz_GRID[0]

    for i in range(GRID):
        for j in range(GRID):
            for k in range(GRID):
                H_0 = H_k(kx_GRID[i], ky_GRID[j], kz_GRID[k])
                E, V = np.linalg.eigh(H_0)
                # central difference
                H_D = np.zeros((3, H_0.shape[0], H_0.shape[1]), dtype = np.complex128)
                if differ_GRID:
                    H_D[0,:,:] = (H_k(kx_GRID[(i+1)%GRID], ky_GRID[j], kz_GRID[k]) - H_k(kx_GRID[(i-1)%GRID], ky_GRID[j], kz_GRID[k]))/(2*stepa)
                    H_D[1,:,:] = (H_k(kx_GRID[i], ky_GRID[(j+1)%GRID], kz_GRID[k]) - H_k(kx_GRID[i], ky_GRID[(j-1)%GRID], kz_GRID[k]))/(2*stepb)
                    H_D[2,:,:] = (H_k(kx_GRID[i], ky_GRID[j], kz_GRID[(k+1)%GRID]) - H_k(kx_GRID[i], ky_GRID[j], kz_GRID[(k-1)%GRID]))/(2*stepc)            
                else:
                    for direc in range(3):
                        H_D[direc,:,:] = H_k_d(direc, kx_GRID[i], ky_GRID[j], kz_GRID[k])
                for m in range(H_D[0].shape[0]):
                    for n in range(H_D[0].shape[0]):
                        for direc1 in range(3):
                            for direc2 in range(3):
                                levi = levi_civita(idc_j,direc1,direc2)
                                if levi != 0:
                                    f = 0
                                    if E[n] <= 0.0 and E[m] > 0.0: #zero temperature
                                        f = 1
                                    elif E[n] > 0.0 and E[m] <= 0.0:
                                        f = -1
                                    if f != 0:
                                        sigma_abc += f * levi * (V[:, n].conj().T @ H_D[idc_i] @ V[:, n] - V[:, m].conj().T @ H_D[idc_i] @ V[:, m]) \
                                            * (V[:, n].conj().T @ H_D[direc1] @ V[:, m]) * (V[:, m].conj().T @ H_D[direc2] @ V[:, n]) \
                                            * delta/np.pi /((energy - E[m] + E[n])**2 + delta**2)                                
                                    
    return np.imag(sigma_abc)*stepa*stepb*stepc*np.pi/(2*np.pi)**3/energy**2
