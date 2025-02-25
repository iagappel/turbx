# turbx
[![PyPI version](https://badge.fury.io/py/turbx.svg)](https://badge.fury.io/py/turbx)
[![Downloads](https://pepy.tech/badge/turbx)](https://pepy.tech/project/turbx)

`turbx` is a `python3` module which contains tools for organization, storage and parallelized processing of turbulent flow datasets, including `super()`ed wrappers of `h5py.File` that streamline data & metadata access.

```
python3 -m pip install turbx
```

`turbx` runs in `python3` and uses parallel `HDF5` (wrapped by `h5py`) for high-performance collective MPI-IO with `mpi4py`. This requires:

- A `python3` installation (3.11+ recommended)
- An MPI implementation such as `OpenMPI`
- A parallel `HDF5` installation (must be compiled with `--enable-parallel`) 
- `mpi4py`
- `h5py` compiled with parallel configuration

Visualization of `HDF5` datasets in `Paraview` is supported through the use of `XML`/`XDMF` sidecar descriptor files. All major data classes (such as `rgd`) can automatically generate the descriptor files by calling `.make_xdmf()`.
