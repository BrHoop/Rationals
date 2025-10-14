import os
import tomllib
import numpy as np
import utils.sowave
import utils.ioxdmf as iox

from utils.grid import Grid2D
from utils.sowave import ScalarField

def rk2(eqs, g, dt):
    nu = len(eqs.u)

    up = []
    k1 = []
    for i in range(nu):
        ux = np.empty_like(eqs.u[0], dtype=object)
        kx = np.empty_like(eqs.u[0], dtype=object)
        up.append(ux)
        k1.append(kx)
    eqs.rhs(k1, eqs.u ,g)
    for i in range(nu):
        up[i][:] = eqs.u[i][:] + 0.5 * dt * k1[i][:]
    eqs.rhs(k1, up, g)
    for i in range(nu):
        eqs.u[i][:] = eqs.u[i][:] + dt * k1[i][:]


def main(parfile):

    #Read the parfile
    with open(parfile,"rb") as f:
        params = tomllib.load(f)
    g = Grid2D(params)
    x = g.xi[0]
    y = g.xi[1]
    dx = g.dx[0]
    dy = g.dx[1]

    eqs = ScalarField(2, g, params["bound_cond"])
    eqs.initialize(g, params)

    output_dir = params["output_dir"]
    output_interval = params["output_interval"]
    os.makedirs(output_dir, exist_ok=True)

    dt = params["cfl"] * dx

    time = 0.0
    func_names = ["phi","chi"]
    iox.write_hdf5(0,eqs.u,x,y,func_names,output_dir)

    Nt = params["Nt"]

    #TODO make it so that I can set the finitederivs opps

    for i in range(1, Nt +1):
        rk2(eqs, g, dt)
        time += dt
        print(f"Step {i:d} t={time:.2e}")
        if i % output_interval == 0:
            iox.write_hdf5(i,eqs.u,x,y,func_names,output_dir)
    iox.write_xdmf(output_dir, Nt, g.shp[0], g.shp[1], func_names, output_interval, dt)




