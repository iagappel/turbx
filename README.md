# turbx
[![PyPI version](https://badge.fury.io/py/turbx.svg)](https://badge.fury.io/py/turbx)
[![Downloads](https://pepy.tech/badge/turbx)](https://pepy.tech/project/turbx)

Extensible toolkit for analyzing turbulent flow datasets.

Install with `pip`:

```
pip install --upgrade --user turbx
```

`turbx` runs in `python3` and uses parallel `HDF5` (wrapped by `h5py`) for high-performance collective MPI-IO with `mpi4py`. This requires:

- A `python3` installation (3.8+ recommended)
- An MPI implementation such as `OpenMPI`
- A parallel `HDF5` installation (must be compiled with `--enable-parallel`) 
- `mpi4py` (optionally compiled from source)
- `h5py` compiled with parallel configuration

Visualization of `HDF5` datasets is possible using `Paraview` with the use of `xdmf` data descriptor files, which are written automatically by calling `.make_xdmf()` on `turbx` data class (such as `rgd`) class instances.
