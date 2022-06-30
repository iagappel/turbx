# turbx

Tools for analysis of turbulent flow datasets.

Documentation available at: https://iagappel.github.io/turbx

```
git clone git@github.com:iagappel/turbx.git
```

It can be installed using `pip` by running

```
pip install --upgrade --user turbx
```

`turbx` runs in `python3` and uses parallel `HDF5` (wrapped by `h5py`) for high-performance collective MPI-IO with `mpi4py`. This requires:

- A `python3` installation (3.8+ recommended)
- An MPI implementation such as `OpenMPI`
- A parallel `HDF5` installation (must be compiled with `--enable-parallel`) 
- `mpi4py` (optionally compiled from source)
- `h5py` compiled with parallel configuration

An environment configuration guide can be found here: https://iagappel.github.io/turbx/env

Visualization of `HDF5` datasets is possible using `Paraview` with the use of `xdmf` data descriptor files, which are written automatically by calling `.make_xdmf()` on `turbx` data class (such as `rgd`) class instances.
