import os
import h5py
import numpy as np


# --- Output ---
def write_hdf5(step, u, X, Y, unames, output_dir):
    """
    Write the current state of the simulation to an HDF5 file.
    Args:
        step (int): The current time step.
        u (numpy.ndarray): The wave field data.
        X (numpy.ndarray): The x-coordinates.
        Y (numpy.ndarray): The y-coordinates.
        output_dir (str): Directory where the HDF5 file will be saved.
    """

    fname = f"{output_dir}/wave_{step:05d}.h5"
    with h5py.File(fname, "w") as f:
        Z = np.array([0.0])
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("Z", data=Z)
        for m in range(len(u)):
            d1 = np.transpose(u[m])
            f.create_dataset(unames[m], data=d1)


def write_xdmf(output_dir, Nt, Nx, Ny, unames, output_interval, dt):
    Nz = 1
    with open(os.path.join(output_dir, "wave.xdmf"), "w") as f:
        f.write(
            """<?xml version="1.0" ?>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n"""
        )
        for n in range(0, Nt + 1, output_interval):
            f.write(
                f"""      <Grid Name="wave_{n}" GridType="Uniform">
        <Time Value="{n*dt}"/>
        <Topology TopologyType="3DRectMesh" Dimensions="{Nz} {Ny} {Nx}"/>
        <Geometry GeometryType="VXVYVZ">
          <DataItem Name="X" Dimensions="{Nx}" NumberType="Float" Precision="8" Format="HDF">wave_{n:05d}.h5:/X</DataItem>
          <DataItem Name="Y" Dimensions="{Ny}" NumberType="Float" Precision="8" Format="HDF">wave_{n:05d}.h5:/Y</DataItem>
          <DataItem Name="Z" Dimensions="{Nz}" NumberType="Float" Precision="8" Format="HDF">wave_{n:05d}.h5:/Z</DataItem>
        </Geometry>\n"""
            )
            for m in range(len(unames)):
                f.write(
                    f"""        <Attribute Name="{unames[m]}" AttributeType="Scalar" Center="Node">
          <DataItem Dimensions="{Nz} {Ny} {Nx}" NumberType="Float" Precision="8" Format="HDF">wave_{n:05d}.h5:/{unames[m]}</DataItem>
        </Attribute>\n"""
                )

            f.write("      </Grid>\n")

        f.write("    </Grid>\n  </Domain>\n</Xdmf>\n")
