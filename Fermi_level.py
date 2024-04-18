import numpy as np
import matplotlib.pyplot as plt
from numba import jit



#We want to calculate the states that below the Fermi level and return it as a number
@jit(nopython = True)
def filling_states_3d(Hk, Fermi_level, kx, ky, kz):
    """
    Hk is the Hamiltonian at a given k point
    Fermi_level is the Fermi energy
    grid is the dimension of the points in the k space
    """

    states = 0
    for i in kx:
        for j in ky:
            for k in kz:
                H = Hk(i,j,k)
                E, V = np.linalg.eigh(H)
                for e in E:
                    if e <= Fermi_level:
                        states += 1
    
    return states

@jit(nopython = True)
def calculate_fermi_level_3d(Hk,basis_dim, filling, kx, ky, kz):
    """
    Hk is the Hamiltonian at a given k point
    filling is the number of states below the Fermi level (Depending the grid size)
    grid is the dimension of the points in the k space
    kx,ky,kz are k grids: np.float64 arrays

    """
    energy_of_states = np.zeros(len(kx)*len(ky)*len(kz)*basis_dim, dtype = np.float64)
    index = 0
    for i in kx:
        for j in ky:
            for k in kz:
                H = Hk(i,j,k)
                E, V = np.linalg.eigh(H)
                for e in E:
                    energy_of_states[index] = e
                    index += 1
    energy_of_states = np.sort(energy_of_states)
    if filling > len(energy_of_states):
        raise ValueError("The filling is larger than the number of states")
    return energy_of_states[int(filling)]