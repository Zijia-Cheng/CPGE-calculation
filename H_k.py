import numpy as np
import matplotlib.pyplot as plt
from numba import jit


sigma_0 = np.array([[1, 0], [0, 1]], dtype = np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype = np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype = np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype = np.complex128)






@jit(nopython = True)
def H_k_0(kx, ky, kz, A, theta, k1):
    # Compute each term of the Hamiltonian
    term_0 = (A * np.sin(kx) * np.sin(kz) + A*np.sin(k1)) * sigma_0
    term_x = (np.cos(kx) - np.cos(k1)) * sigma_x
    term_y = np.sin(ky) * sigma_y
    term_z = (1 - np.cos(kz) - np.cos(ky)) * sigma_z
    H = term_0 + term_x + term_y + term_z
    return H


@jit(nopython = True)
def H_k_mag(kx, ky, kz, t, m , gap):
    # Compute each term of the Hamiltonian
    term0 = 2*(-t*np.sin(kx)*np.kron(sigma_x,sigma_0))
    term1 = 2*(t*np.sin(ky)*np.kron(sigma_y,sigma_0))
    term2 = 2*(t*np.cos(kz)*np.kron(sigma_z,sigma_x))
    term3 = -m*(2 - np.cos(kx) - np.cos(ky))*np.kron(sigma_z,sigma_0)
    term4 = gap*np.kron(sigma_z,sigma_z)

    return term0 + term1 + term2 + term3 + term4

@jit(nopython = True)
def H_k_mag_d(direc, kx, ky, kz, t, m, gap):
    if direc == 0:
        return 2*(-t*np.cos(kx)*np.kron(sigma_x,sigma_0)) - m*np.sin(kx)*np.kron(sigma_z,sigma_0)
    elif direc == 1:
        return 2*(t*np.cos(ky)*np.kron(sigma_y,sigma_0)) - m*np.sin(ky)*np.kron(sigma_z,sigma_0)
    elif direc == 2:
        return 2*(-t*np.sin(kz)*np.kron(sigma_z,sigma_x))
    


@jit(nopython = True)
def Hk_r2k(kx, ky, kz, basis_dim, terms, lattice, position):
    """
    kx, ky, kz are the k points: np.float64
    basis_dim is the dimension of the Hamiltonian: int
    terms is the list of terms in the Hamiltonian: np.complex128
    lattice is the lattice vectors: np.complex128
    position is the position of the atoms: np.complex128
    
    """
    h_k = np.zeros((basis_dim,basis_dim),dtype=np.complex128)
    k_vector = np.array([kx,ky,kz], dtype = np.complex128)
    for term in terms:
        R = term[3] * lattice[0] + term[4] * lattice[1] + term[5] * lattice[2] + position[int(term[2].real)] @ lattice - position[int(term[1].real)] @ lattice
        phase = term[0]*np.exp(-1j*np.dot(k_vector,R))
        h_k[int(term[2].real),int(term[1].real)] += phase
        h_k[int(term[1].real),int(term[2].real)] += phase.conjugate()
    return h_k




@jit(nopython = True)
def Hk_r2k_diff(direc, kx, ky, kz,basis_dim, terms, lattice, position):
    #Direcly calculate the derivative of the Hamiltonian along the direction direc
    #direc = 0,1,2 for x,y,z
    h_k_d = np.zeros((basis_dim,basis_dim),dtype=np.complex128)
    k_vector = np.array([kx,ky,kz], dtype = np.complex128)
    for term in terms:
        R = term[3] * lattice[0] + term[4] * lattice[1] + term[5] * lattice[2] + position[int(term[2].real)] @ lattice - position[int(term[1].real)] @ lattice
        phase = term[0]*np.exp(-1j*np.dot(k_vector,R))
        h_k_d[int(term[2].real),int(term[1].real)] += -1j*R[direc]*phase
        h_k_d[int(term[1].real),int(term[2].real)] += 1j*R[direc]*phase.conjugate()
    return h_k_d
