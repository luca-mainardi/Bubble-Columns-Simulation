"""Author: Douwe Orij"""

import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import os
import time
import utils.sph_harm_functions as sph_harm


# Combine all different datasets before rendering
def combine_data(dir):
    df = pd.DataFrame()

    for file in os.listdir(dir):

        # Read and load data files
        fname = os.path.join(dir, file)
        df1 = pd.read_pickle(fname)

        df = pd.concat([df, df1], ignore_index=True)

    return df


class bubble:
    # NOTE: This class is designed to be called from the root directory (ie have the 2amm40-spherical-harmonics directory as your working directory)
    # If you call it from a different location, you may need to update self.dir, or other references to file locations.
    def __init__(self, df, save=False, l_max=14, res=100) -> None:
        self.dir = ""  # os.path.dirname(os.path.realpath(__file__))

        if save:
            self.results = os.path.join(self.dir, "figures")
            os.makedirs(self.results, exist_ok=True)
        else:
            self.results = None

        # Define the spherical harmonics grid and faces for 3D mesh
        self.theta, self.phi = sph_harm.make_grid([res, res])
        self.faces = sph_harm.make_faces(self.theta, self.phi)

        # Define the spherical harmonics and the maximum l value
        self.l_max = l_max
        self.sph = sph_harm.sph_harm(self.theta, self.phi, self.l_max)

        # Get the data
        self.df = combine_data(df)
        self.bubbles = self.df["id"].unique()
        self.sims = self.df["sim"].unique()

    def get_orbs(self, data):
        columns = [col for col in self.df.columns if "orb" in col]
        orbs = data[columns].values.astype(np.float64)

        return orbs

    def sort_and_get(self, i, data=None):
        if data is None:
            data = self.df

        data = data[data["id"] == self.bubbles[i]]
        data = data.sort_values("time [s]")
        return data

    def make_bubble(self, orbs):
        vertices = sph_harm.sph2cart(self.sph, orbs, self.theta, self.phi)
        stl_sph = pv.PolyData(vertices, self.faces)

        return stl_sph

    def display_bubble(self, i):
        # Extract data
        data = self.df.iloc[i]

        # Get orbs and make bubble
        orbs = self.get_orbs(data)
        stl_sph = self.make_bubble(orbs)

        # Read stl file
        data_stl = data["stl"]

        # Replace '\' with '/' for unix
        if os.name == "posix":
            data_stl = data_stl.replace("\\", "/")

        stl = os.path.join(self.dir, data_stl)
        stl = pv.read(stl)
        stl = sph_harm.normalize(stl)

        # Plot the bubble
        sph_harm.plot(stl, stl_sph, save_dir=self.results)

    def display_orbs(self, i, orbs_to_show=None):
        # Extract data
        data = self.sort_and_get(i)

        # Get orbs into numpy array
        orbs = self.get_orbs(data)

        # Select specific orbs
        if not orbs_to_show:
            orbs_to_show = range(orbs.shape[1])
        orbs = orbs[:, orbs_to_show]

        # Get the time values
        time = data["time [s]"].values

        # Make plot
        plt.figure()
        plt.plot(time, orbs)
        plt.xlabel("Time [s]")
        plt.ylabel("Orbital scale [-]")
        plt.title("Orbital scale over time")
        plt.legend([f"orb_{i}" for i in range(orbs.shape[1])])
        plt.tight_layout()
        if self.results:
            plt.savefig(os.path.join(self.results, "orbital_scale.png"))
        plt.show()

    def display_trajectory(self, i):
        # Extract data
        data = self.sort_and_get(i)

        # Get the position values
        pos = data[["pos_x", "pos_y", "pos_z"]].values

        # Make plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Bubble position over time")
        if self.results:
            plt.savefig(os.path.join(self.results, "bubble_position.png"))
        plt.show()

    def display_trajectories(self, i):
        # Extract data
        df = self.df[self.df["sim"] == self.sims[i]]

        # Make a new figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for i in range(df["id"].nunique()):
            data = self.sort_and_get(i, df)

            # Get the position values
            pos = data[["pos_x", "pos_y", "pos_z"]].values

            # Make plot
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])

        # Finish layout
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Bubble position over time")
        fig.tight_layout()

        if self.results:
            plt.savefig(os.path.join(self.results, "bubble_positions.png"))
        plt.show()

    def display_shape(self, i):
        # Extract data
        data = self.sort_and_get(i)

        # Get orbs and convert to vertices
        orbs = self.get_orbs(data)
        vertices = [
            sph_harm.sph2cart(self.sph, row, self.theta, self.phi) for row in orbs
        ]

        # Convert first set of vertices to mesh
        mesh = pv.PolyData(vertices[0], self.faces)

        # Make an updating plot
        pl = pv.Plotter()
        pl.add_mesh(mesh, color="blue")
        pl.show(interactive_update=True)
        if self.results:
            pl.open_gif(os.path.join(self.results, "bubble_shape.gif"))

        # Iterate over the vertices
        for i in range(len(vertices)):
            mesh.points = vertices[i]
            pl.add_title(f'Bubble shape at time {data.iloc[i]["time [s]"]:.2f} s')
            pl.update()
            # Write 5 frames to the gif to slow down the animation
            if self.results:
                [pl.write_frame() for j in range(5)]
            time.sleep(0.3)

        pl.close()

    def display_simulation(self, i):
        # Make an updating plot
        pl = pv.Plotter()

        df = self.df[self.df["sim"] == self.sims[i]]

        meshes = []
        verts = []
        for i in range(df["id"].nunique()):
            # Extract data
            data = self.sort_and_get(i, df)

            # Get orbs and convert to vertices
            pos = data[["pos_x", "pos_y", "pos_z"]].values
            orbs = self.get_orbs(data)
            vertices = [
                sph_harm.sph2cart(self.sph, row, self.theta, self.phi) + p
                for row, p in zip(orbs, pos)
            ]

            # Convert first set of vertices to mesh
            mesh = pv.PolyData(vertices[0], self.faces)
            pl.add_mesh(mesh, color="blue")

            verts.append(vertices)
            meshes.append(mesh)

        pl.camera_position = "xz"
        pl.show(interactive_update=True)
        if self.results:
            pl.open_gif(os.path.join(self.results, "bubble_simulation.gif"))

        # Iterate over the vertices
        for i in range(len(vertices)):
            for mesh, vert in zip(meshes, verts):
                mesh.points = vert[i]

            pl.add_title(f'Simulation at time {data.iloc[i]["time [s]"]:.2f} s')
            pl.camera_position = "xz"
            pl.update()
            # Write 5 frames to the gif to slow down the animation
            if self.results:
                [pl.write_frame() for j in range(5)]
            time.sleep(0.3)

        pl.close()


if __name__ == "__main__":
    print(os.listdir("."))


if False:
    # ---------------------------------------------------------------- #
    loc = os.path.dirname(os.path.realpath(__file__))
    dir = os.path.join(loc, "data", "pickle_files_FT")
    save = True  # Save the plots
    orbs_to_show = range(0, 5)  # Select the orbs to show

    b = bubble(dir, save)
    print(b.df.head())

    bubble_0_df = b.df.loc[(b.df["bub_num"] == 0) & (b.df["sim"] == "4mm_eps05")]
    print(bubble_0_df.head())
    print("Num rows:", len(bubble_0_df))
    b.display_bubble(
        0
    )  # Displays a single bubble at a certain timestep to compare the ground truth with the spherical harmonics
    b.display_orbs(
        0, orbs_to_show
    )  # Displays the orbital weights over time for a single bubble
    b.display_trajectory(0)  # Displays the position of a single bubble over time
    b.display_trajectories(0)  # Displays the position of a simulation over time
    b.display_shape(0)  # Displays the shape of a single bubble over time
    b.display_simulation(0)  # Displays the simulation of all bubbles over time
