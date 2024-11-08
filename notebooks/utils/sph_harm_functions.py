"""Author: Douwe Orij"""

import numpy as np
import scipy as sp
from matplotlib.tri import Triangulation
import pyvista as pv


# Define values of spherical harmonics at given phi and theta
def sph_harm(theta, phi, l_max=14):

    num_orbitals = (l_max + 1) ** 2
    list_orbitals = np.arange(num_orbitals)

    sph = num_orbitals * [None]

    for i, j in enumerate(list_orbitals):
        l = int(np.floor(np.sqrt(j)))
        m = int(j - l**2 - l)

        Y_lm = sp.special.sph_harm(m, l, theta, phi)
        if m < 0:
            Y_lm = np.sqrt(2) * Y_lm.imag
        elif m == 0:
            Y_lm = Y_lm.real
        elif m > 0:
            Y_lm = np.sqrt(2) * Y_lm.real

        sph[i] = Y_lm

    return np.array(sph).T


# Define new grid
def make_grid(grid):
    theta, phi = np.linspace(0, 2 * np.pi, grid[1]), np.linspace(0, np.pi, grid[0])
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = np.ravel(theta), np.ravel(phi)

    return theta, phi


# Make faces on regular grid to close 3D object
def make_faces(theta, phi):
    faces = Triangulation(theta, phi).triangles
    faces[:, [1, 2]] = faces[:, [2, 1]]
    col = np.repeat(3, len(faces))
    faces = np.c_[col, faces].reshape((1, -1))

    return faces


# Cartesian --> Spherical
def cart2sph(vertices):
    [X, Y, Z] = vertices.T
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arctan2(Y, X)
    phi = np.arccos(Z / r)

    return r, theta, phi


# Spherical --> Cartesian
def sph2cart(sph, weights, theta, phi):
    X = np.dot(sph, weights) * np.sin(phi) * np.cos(theta)
    Y = np.dot(sph, weights) * np.sin(phi) * np.sin(theta)
    Z = np.dot(sph, weights) * np.cos(phi)

    return np.array([X, Y, Z]).T


def normalize(data, rot=[0, 0, 0]):
    # Import data and center to origin
    data = data.translate(np.array(data.center) * -1, inplace=False)
    data = data.rotate_x(rot[0], inplace=True)
    data = data.rotate_y(rot[1], inplace=True)
    data = data.rotate_z(rot[2], inplace=True)
    return data


def stl_maker(weights, grid):
    l_max = int(np.sqrt(weights.shape[0]) - 1)

    # Define new grid
    theta, phi = np.linspace(0, 2 * np.pi, grid[1]), np.linspace(0, np.pi, grid[0])
    theta, phi = np.meshgrid(theta, phi)
    theta, phi = np.ravel(theta), np.ravel(phi)

    # Define spherical harmonics on new grid
    sph = sph_harm(theta, phi, l_max)

    # Determine vertices and faces
    vertices = sph2cart(sph, weights, theta, phi)
    faces = Triangulation(theta, phi).triangles
    faces[:, [1, 2]] = faces[:, [2, 1]]

    # Using PyVista
    col = np.repeat(3, len(faces))
    faces = np.c_[col, faces].reshape((1, -1))

    cloud = pv.PolyData(vertices, faces)

    return cloud


# def stl_from_vertices(r,grid):
#     theta,phi,faces = make_grid(grid)

#     X = r * np.sin(phi) * np.cos(theta)
#     Y = r * np.sin(phi) * np.sin(theta)
#     Z = r * np.cos(phi)

#     cloud = pv.PolyData(np.array([X,Y,Z]).T,faces)

#     return cloud


def weights_from_stl(cloud, rot=[0, 0, 0], l_max=14):
    num_orbitals = (l_max + 1) ** 2
    list_orbitals = np.arange(num_orbitals)

    cloud = normalize(cloud, rot)

    r, theta, phi = cart2sph(np.array(cloud.points))
    sph = sph_harm(
        theta, phi
    )  # NOTE: I removed this since stl_2_sph_harm.py otherwise gives errors.          #, list_orbitals=list_orbitals)

    weights, res, rank, S = np.linalg.lstsq(sph, r)

    return weights, cloud, res


# def stl_maker_list(weights,grid=[100,100],l_max=0,num_orbitals=0,list_orbitals=[]):

#     if l_max != 0:
#         num_orbitals = (l_max+1)**2
#         list_orbitals = np.arange(num_orbitals)

#     elif num_orbitals != 0:
#         list_orbitals = np.arange(num_orbitals)

#     elif len(list_orbitals) != 0:
#         num_orbitals = len(list_orbitals)

#     else:
#         l_max = int(np.sqrt(weights.shape[0])-1)
#         num_orbitals = (l_max+1)**2
#         list_orbitals = np.arange(num_orbitals)

#     # Define new grid
#     theta,phi,faces = make_grid(grid)

#     # Define spherical harmonics on new grid
#     sph = sph_harm(theta,phi,list_orbitals=list_orbitals)
#     vertices = sph2cart(sph,weights,theta,phi)

#     cloud = pv.PolyData(vertices,faces)

#     return cloud


def plot(stl, stl_sph, save_dir=None):
    # Make plot
    pl = pv.Plotter(shape=(1, 3), window_size=[1500, 500])
    pl.subplot(0, 0)
    pl.add_mesh(stl, color="red")
    pl.add_title("Ground Truth")
    pl.subplot(0, 1)
    pl.add_mesh(stl_sph, color="blue")
    pl.add_title("Spherical Harmonics")
    pl.subplot(0, 2)
    pl.add_mesh(stl, color="red", style="wireframe")
    pl.add_mesh(stl_sph, color="blue", style="wireframe")
    pl.add_title("Overlap")
    pl.show()
    if save_dir:
        pl.screenshot(f"{save_dir}/comparison.png")
