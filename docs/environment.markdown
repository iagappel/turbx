---
layout: page
title: Environment
permalink: /env/
nav_order: 2
---

## Environment

`turbx` runs in `python3` and uses parallel `HDF5` (wrapped by `h5py`) for high-performance collective MPI-IO with `mpi4py`. This requires:

- A `python3` installation (3.8+ recommended)
- An MPI implementation such as `OpenMPI`
- A parallel `HDF5` installation (must be compiled with `--enable-parallel`) 
- `mpi4py` (optionally compiled from source)
- `h5py` compiled with parallel configuration

Visualization of `HDF5` datasets is possible using `Paraview` with the use of `xdmf` data descriptor files, which are written automatically by calling `.make_xdmf()` on `turbx` data class (such as `rgd`) instances.

------------------------------------------------------------------------

### **OpenMPI**

Information regarding compiling and installing `OpenMPI` can be found in the [OpenMPI FAQs](https://www.open-mpi.org/faq/?category=building).

``` bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.2.tar.gz
tar -zxvf openmpi-4.1.2.tar.gz
cd openmpi-4.1.2/
./configure --prefix=/path/to/software/openmpi/openmpi_4.1.2 --enable-shared --enable-static --without-verbs
make all install
```

After a successful installation, the install location should be added to the environment.

``` bash
export LD_LIBRARY_PATH=/path/to/software/openmpi/openmpi_4.1.2/lib:${LD_LIBRARY_PATH}
export PATH=/path/to/software/openmpi/openmpi_4.1.2/bin:${PATH}
```

------------------------------------------------------------------------

### **HDF5**

Information on compiling HDF5 can be found directly in the [HDF5 source code](https://www.hdfgroup.org/downloads/hdf5/source-code/) package.

An example of how to compile and install `HDF5` is given below (note the `--enable-parallel` flag).

``` bash
export LD_LIBRARY_PATH=/path/to/software/openmpi/openmpi_4.1.2/lib:${LD_LIBRARY_PATH}
export PATH=/path/to/software/software/openmpi/openmpi_4.1.2/bin:${PATH}
export CC=/path/to/software/openmpi/openmpi_4.1.2/bin/mpicc
export FC=/path/to/software/openmpi/openmpi_4.1.2/bin/mpif90

wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz
tar -zxvf hdf5-1.12.1.tar.gz
cd hdf5-1.12.1

./configure --enable-fortran --enable-parallel --prefix=/path/to/software/hdf5/hdf5_1.12.1_ompi4.1.2
make install
```

After a successful installation, the install location should be added to the environment.

``` bash
export PATH=/path/to/software/hdf5/hdf5_1.12.1_ompi/bin:${PATH}
export HDF5_DIR=/path/to/software/hdf5/hdf5_1.12.1_ompi
```

------------------------------------------------------------------------

### **python3**

Source code for `python` can be downloaded from the [Python Source Releases](https://www.python.org/downloads/source/) webpage. Installation instructions can be found in the source code package or in any of several install guides such as [this one](https://realpython.com/installing-python/). If you don't already have `python3` or want a customizable user install in a HPC environment, you can compile it from source as follows.

``` bash
wget https://www.python.org/ftp/python/3.9.10/Python-3.9.10.tgz
tar -xvzf Python-3.9.10.tgz
cd Python-3.9.10
export CC=/path/to/gnu/9.2.0/bin/gcc
./configure --with-ensurepip=install --enable-optimizations --prefix=/path/to/software/python/python_3.9.10
make && make install
```

Once complete, run `python3` in the terminal and confirm the installation. The install location should be appended to the enironment `PATH`.

``` bash
export PATH=/path/to/software/python/python_3.9.10/bin:${PATH}
```

Use `pip3` to install additional `python3` modules.

``` bash
pip3 install --upgrade --user pip
pip3 install --upgrade --user setuptools wheel
pip3 install --upgrade --user cython
pip3 install --upgrade --user numpy scipy
pip3 install --upgrade --user psutil
pip3 install --upgrade --user tqdm
pip3 install --upgrade --user matplotlib cmocean colorcet cmasher
pip3 install --upgrade --user scikit-image
pip3 install --upgrade --user vtk
pip3 install --upgrade --user pyvista
```

Keep your `python3` environment clean and up-to-date by periodically running the following snippet in a `python3` terminal.

``` python
import pkg_resources
from subprocess import call
packages = [dist.project_name for dist in pkg_resources.working_set]
call('pip3 install --upgrade --user ' + ' '.join(packages), shell=True)
```

------------------------------------------------------------------------

### **mpi4py**

Information on installing `mpi4py` can be found in the [mpi4py docs](https://mpi4py.readthedocs.io/en/stable/appendix.html#building-mpi)

``` bash
export CC=/path/to/software/openmpi/openmpi_4.1.2/bin/mpicc
export LD_LIBRARY_PATH=/path/to/software/openmpi/openmpi_4.1.2/lib:${LD_LIBRARY_PATH}
export PATH=/path/to/software/software/openmpi/openmpi_4.1.2/bin:${PATH}

wget https://github.com/mpi4py/mpi4py/releases/download/3.1.3/mpi4py-3.1.3.tar.gz
tar -zxvf mpi4py-3.1.3.tar.gz
cd mpi4py-3.1.3
python3 setup.py build
python3 setup.py install --user
```

------------------------------------------------------------------------

### **h5py**

To enable collective reading and writing with `h5py`, it must be compiled against a parallel installation of `HDF5` i.e. with the `--enable-parallel` flag activated in its configuration (see above).

More information can be found in the [h5py docs](https://docs.h5py.org/en/latest/build.html).

``` bash
export LD_LIBRARY_PATH=/path/to/software/hdf5/hdf5_1.12.1_ompi/lib:${PATH}
export PATH=/path/to/software/hdf5/hdf5_1.12.1_ompi/bin:${PATH}
export LD_LIBRARY_PATH=/path/to/software/openmpi/openmpi_4.1.2/lib:${LD_LIBRARY_PATH}
export PATH=/path/to/software/software/openmpi/openmpi_4.1.2/bin:${PATH}

export CC=/path/to/software/openmpi/openmpi_4.1.2/bin/mpicc

export HDF5_DIR=/path/to/software/hdf5/hdf5_1.12.1_ompi
export HDF5_MPI="ON"

wget https://github.com/h5py/h5py/releases/download/3.6.0/h5py-3.6.0.tar.gz
tar -zxvf h5py-3.6.0.tar.gz
cd h5py-3.6.0
H5PY_SETUP_REQUIRES=0 python3 setup.py build
H5PY_SETUP_REQUIRES=0 python3 setup.py install --user
```

The `h5py` installation can be verified in a `python3` terminal with:

``` python
import h5py
print(h5py.version.info)
```
