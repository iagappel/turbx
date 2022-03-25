#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, re, glob, time, traceback
import h5py
import mpi4py
from mpi4py import MPI
import pickle
import gzip
import shutil
import psutil
import pathlib
from pathlib import Path, PurePosixPath
import subprocess
import io
import copy
import datetime
import timeit
import textwrap
import tqdm
from tqdm import tqdm
import math
import numpy as np
import scipy as sp
from scipy import interpolate, integrate, signal, stats, special

# ## waiting until numba supports numpy >1.22
# import numba
# from numba import jit, njit
## try:
##     from numba import cuda
##     #numba.cuda.detect()
##     gpus = numba.cuda.gpus
##     print('CUDA Enabled Devices\n--------------------')
##     for i in range(len(gpus)):
##         print('%i --> %s'%(i,gpus[i].name.decode('ascii')))
## except ImportError:
##     print('No CUDA Devices Found')

import skimage
from skimage import color
import matplotlib as mpl
#import PyQt5
#mpl.use('Qt5Agg') ## maybe beneficial on Windows
#mpl.use('Agg') ## non-GUI backend
import matplotlib.pyplot as plt
import cmocean
import colorcet
import cmasher

#import vtk
#from vtk.util import numpy_support

## required for EAS3
import struct

'''
------------------------------------------------------------

Description
-----------

Tools for analysis of turbulent flow datasets
--> dev version Mar/April 2022

Notes
-----
- HDF5 Documentation : https://docs.h5py.org/_/downloads/en/3.2.1/pdf/
- compiling parallel HDF5 & h5py: https://docs.h5py.org/en/stable/mpi.html#building-against-parallel-hdf5

------------------------------------------------------------
'''

# data container interface classes
# ------------------------------------------------------------

class rgd(h5py.File):
    '''
    Rectilinear Grid Data (RGD)
    ---------------------------
    - super()'ed h5py.File class
    - HDF5-based data storage format
    - MPI (mpi4py) support
    - stores data in 4D (temporal operations like mean, FFT, etc benefit heavily)
    - Paraview support
    - pop-open viewer support through:
        - pyvista (VTK rendering)
        - plotoptix (NVIDIA Optix rendering)
    
    to clear:
    ---------
    > os.system('h5clear -s tmp.h5')
    > hf = h5py.File('tmp.h5', 'r', libver='latest')
    > hf.close()
    
    Structure
    ---------
    
    rgd.h5
    │
    ├── header/
    │   └── udef_char
    │   └── udef_real
    │
    ├── dims/ --> 1D
    │   └── x
    │   └── y
    │   └── z
    │   └── t
    │
    └-─ data/<<scalar>> --> 4D [t,z,y,x]
    
    '''
    
    def __init__(self, *args, **kwargs):
        
        self.fname, openMode = args
        
        ## check if running with MPI
        if ('comm' in kwargs):
            self.comm = kwargs['comm']
            self.n_ranks = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        else:
            self.comm = None
            self.n_ranks = 1
            self.rank = 0
        
        if ('info' in kwargs):
            self.mpi_info = kwargs['info']
        else:
            mpi_info = MPI.Info.Create()
            mpi_info.Set('romio_ds_write' , 'disable'   )                             
            mpi_info.Set('romio_ds_read'  , 'disable'   )
            mpi_info.Set('romio_cb_read'  , 'automatic' )
            mpi_info.Set('romio_cb_write' , 'automatic' )
            mpi_info.Set('collective_buffering' , 'true' )
            mpi_info.Set('cb_block_size'  , str(int(round(    2*1024**2))))
            mpi_info.Set('cb_buffer_size' , str(int(round( 64*2*1024**2))))
            kwargs['info'] = mpi_info
            self.mpi_info = mpi_info
        
        if ('driver' not in kwargs) and ('info' in kwargs):
            del kwargs['info']
        
        # ## if opened in serial mode, delete info kwarg
        # if (self.comm is None):
        #     del kwargs['info']
        
        if ('rdcc_nbytes' in kwargs):
            pass
        else:
            kwargs['rdcc_nbytes']=4*1024**3
        
        ## rgd() unique kwargs --> pop() rather than get()
        verbose = kwargs.pop('verbose',False)
        force   = kwargs.pop('force',False)
        
        if (openMode == 'w') and (force is False) and os.path.isfile(self.fname):
            if (self.rank==0):
                print('\n'+72*'-')
                print(self.fname+' already exists! opening with \'w\' would overwrite.\n')
                openModeInfoStr = '''
                                  r       --> Read only, file must exist
                                  r+      --> Read/write, file must exist
                                  w       --> Create file, truncate if exists
                                  w- or x --> Create file, fail if exists
                                  a       --> Read/write if exists, create otherwise
                                  
                                  or use force=True arg:
                                  
                                  >>> with rgd(<<fname>>,'w',force=True) as f:
                                  >>>     ...
                                  '''
                print(textwrap.indent(textwrap.dedent(openModeInfoStr), 2*' ').strip('\n'))
                print(72*'-'+'\n')
            
            if (self.comm is not None):
                self.comm.Barrier()
            raise FileExistsError()
        
        ## remove file, touch, stripe
        elif (openMode == 'w') and (force is True) and os.path.isfile(self.fname):
            if (self.rank==0):
                os.remove(self.fname)
                Path(self.fname).touch()
                if shutil.which('lfs') is not None:
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 2M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        ## touch, stripe
        elif (openMode == 'w') and not os.path.isfile(self.fname):
            if (self.rank==0):
                Path(self.fname).touch()
                if shutil.which('lfs') is not None:
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 2M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        else:
            pass
        
        if (self.comm is not None):
            self.comm.Barrier()
        
        ## call actual h5py.File.__init__()
        super(rgd, self).__init__(*args, **kwargs)
        self.get_header(verbose=verbose)
    
    def __enter__(self):
        '''
        for use with python 'with' statement
        '''
        #return self
        return super(rgd, self).__enter__()
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        '''
        for use with python 'with' statement
        '''
        if (self.rank==0):
            if exception_type is not None:
                print('\nsafely closed RGD HDF5 due to exception')
                print(72*'-')
                print('exception type : '+exception_type.__name__)
            if exception_value is not None:
                print('exception_value : '+str(exception_value))
            if exception_traceback is not None:
                print(72*'-')
                #print('exception_traceback : '+str(exception_traceback))
                print('exception_traceback : \n'+traceback.format_exc().rstrip())
            if exception_type is not None:
                print(72*'-')
        return super(rgd, self).__exit__()
    
    def get_header(self,**kwargs):
        '''
        initialize header attributes of RGD class instance
        '''
        
        verbose = kwargs.get('verbose',True)
        
        if (self.rank!=0):
            verbose=False
        
        # === attrs
        if ('duration_avg' in self.attrs.keys()):
            self.duration_avg = self.attrs['duration_avg']
        
        # === udef
        
        if ('header' in self):
            
            udef_real = np.copy(self['header/udef_real'][:])
            udef_char = np.copy(self['header/udef_char'][:]) ## the unpacked numpy array of |S128 encoded fixed-length character objects
            udef_char = [s.decode('utf-8') for s in udef_char] ## convert it to a python list of utf-8 strings
            self.udef = dict(zip(udef_char, udef_real)) ## just make udef_real a dict with udef_char as keys
            
            # === characteristic values
            
            self.Ma          = self.udef['Ma']
            self.Re          = self.udef['Re']
            self.Pr          = self.udef['Pr']
            self.kappa       = self.udef['kappa']
            self.R           = self.udef['R']
            self.p_inf       = self.udef['p_inf']
            self.T_inf       = self.udef['T_inf']
            self.C_Suth      = self.udef['C_Suth']
            self.S_Suth      = self.udef['S_Suth']
            self.mu_Suth_ref = self.udef['mu_Suth_ref']
            self.T_Suth_ref  = self.udef['T_Suth_ref']
            
            if verbose: print(72*'-')
            if verbose: even_print('Ma'          , '%0.2f [-]'           % self.Ma          )
            if verbose: even_print('Re'          , '%0.1f [-]'           % self.Re          )
            if verbose: even_print('Pr'          , '%0.3f [-]'           % self.Pr          )
            if verbose: even_print('T_inf'       , '%0.3f [K]'           % self.T_inf       )
            if verbose: even_print('p_inf'       , '%0.1f [Pa]'          % self.p_inf       )
            if verbose: even_print('kappa'       , '%0.3f [-]'           % self.kappa       )
            if verbose: even_print('R'           , '%0.3f [J/(kg·K)]'    % self.R           )
            if verbose: even_print('mu_Suth_ref' , '%0.6E [kg/(m·s)]'    % self.mu_Suth_ref )
            if verbose: even_print('T_Suth_ref'  , '%0.2f [K]'           % self.T_Suth_ref  )
            if verbose: even_print('C_Suth'      , '%0.5e [kg/(m·s·√K)]' % self.C_Suth      )
            if verbose: even_print('S_Suth'      , '%0.2f [K]'           % self.S_Suth      )
            
            # === characteristic values : derived
            
            rho_inf = self.rho_inf = self.p_inf/(self.R * self.T_inf)
            mu_inf  = self.mu_inf  = 14.58e-7*self.T_inf**1.5/(self.T_inf+110.4)
            nu_inf  = self.nu_inf  = self.mu_inf/self.rho_inf
            a_inf   = self.a_inf   = np.sqrt(self.kappa*self.R*self.T_inf)
            U_inf   = self.U_inf   = self.Ma*self.a_inf
            cp      = self.cp      = self.R*self.kappa/(self.kappa-1.)
            cv      = self.cv      = self.cp/self.kappa                         
            r       = self.r       = self.Pr**(1/3)
            Tw      = self.Tw      = self.T_inf
            Taw     = self.Taw     = self.T_inf + self.r*self.U_inf**2/(2*self.cp)
            lchar   = self.lchar   = self.Re*self.nu_inf/self.U_inf
            
            if verbose: print(72*'-')
            if verbose: even_print('rho_inf' , '%0.3f [kg/m³]'    % self.rho_inf )
            if verbose: even_print('mu_inf'  , '%0.6E [kg/(m·s)]' % self.mu_inf  )
            if verbose: even_print('nu_inf'  , '%0.6E [m²/s]'     % self.nu_inf  )
            if verbose: even_print('a_inf'   , '%0.6f [m/s]'      % self.a_inf   )
            if verbose: even_print('U_inf'   , '%0.6f [m/s]'      % self.U_inf   )
            if verbose: even_print('cp'      , '%0.3f [J/(kg·K)]' % self.cp      )
            if verbose: even_print('cv'      , '%0.3f [J/(kg·K)]' % self.cv      )
            if verbose: even_print('r'       , '%0.6f [-]'        % self.r       )
            if verbose: even_print('Tw'      , '%0.3f [K]'        % self.Tw      )
            if verbose: even_print('Taw'     , '%0.3f [K]'        % self.Taw     )
            if verbose: even_print('lchar'   , '%0.6E [m]'        % self.lchar   )
            if verbose: print(72*'-'+'\n')
            
            # === write the 'derived' udef variables to a dict attribute of the RGD instance
            udef_char_deriv = ['rho_inf', 'mu_inf', 'nu_inf', 'a_inf', 'U_inf', 'cp', 'cv', 'r', 'Tw', 'Taw', 'lchar']
            udef_real_deriv = [ rho_inf,   mu_inf,   nu_inf,   a_inf,   U_inf,   cp,   cv,   r,   Tw,   Taw,   lchar ]
            self.udef_deriv = dict(zip(udef_char_deriv, udef_real_deriv))
        
        else:
            pass
        
        # === coordinate vectors
        
        if all([('dims/x' in self),('dims/y' in self),('dims/z' in self)]):
            
            x   = self.x   = np.copy(self['dims/x'][:])
            y   = self.y   = np.copy(self['dims/y'][:])
            z   = self.z   = np.copy(self['dims/z'][:])
            nx  = self.nx  = x.size
            ny  = self.ny  = y.size
            nz  = self.nz  = z.size
            ngp = self.ngp = nx*ny*nz
            
            if verbose: print(72*'-')
            if verbose: even_print('nx', '%i'%nx )
            if verbose: even_print('ny', '%i'%ny )
            if verbose: even_print('nz', '%i'%nz )
            if verbose: even_print('ngp', '%i'%ngp )
            if verbose: print(72*'-')
            
            if verbose: even_print('x_min', '%0.2f'%x.min())
            if verbose: even_print('x_max', '%0.2f'%x.max())
            if verbose: even_print('dx begin : end', '%0.3E : %0.3E'%( (x[1]-x[0]), (x[-1]-x[-2]) ))
            if verbose: even_print('y_min', '%0.2f'%y.min())
            if verbose: even_print('y_max', '%0.2f'%y.max())
            if verbose: even_print('dy begin : end', '%0.3E : %0.3E'%( (y[1]-y[0]), (y[-1]-y[-2]) ))
            if verbose: even_print('z_min', '%0.2f'%z.min())
            if verbose: even_print('z_max', '%0.2f'%z.max())        
            if verbose: even_print('dz begin : end', '%0.3E : %0.3E'%( (z[1]-z[0]), (z[-1]-z[-2]) ))
            if verbose: print(72*'-'+'\n')
        
        else:
            pass
        
        # === 1D grid filters
        
        self.hasGridFilter=False
        if ('dims/xfi' in self):
            xfi = np.copy(self['dims/xfi'][:])
            self.xfi = xfi
            if not np.array_equal(xfi, np.array(range(nx), dtype=np.int64)):
                self.hasGridFilter=True
        if ('dims/yfi' in self):
            yfi = np.copy(self['dims/yfi'][:])
            self.yfi = yfi
            if not np.array_equal(yfi, np.array(range(ny), dtype=np.int64)):
                self.hasGridFilter=True
        if ('dims/zfi' in self):
            zfi = np.copy(self['dims/zfi'][:])
            self.zfi = zfi
            if not np.array_equal(xfi, np.array(range(nz), dtype=np.int64)):
                self.hasGridFilter=True
        
        # === time vector
        
        if ('dims/t' in self):
            self.t = np.copy(self['dims/t'][:])
            
            if ('data' in self): ## check t dim and data arr agree
                nt,_,_,_ = self['data/%s'%list(self['data'].keys())[0]].shape ## 4D
                if (nt!=self.t.size):
                    raise AssertionError('nt!=self.t.size : %i!=%i'%(nt,self.t.size))
            
            nt = self.t.size
            
            try:
                self.dt = self.t[1] - self.t[0]
            except IndexError:
                self.dt = 0.
            
            self.nt       = nt       = self.t.size
            self.duration = duration = self.t[-1] - self.t[0]
            self.ti       = ti       = np.array(range(self.nt), dtype=np.int64)
        
        elif all([('data' in self),('dims/t' not in self)]): ## data but no time
            nt,_,_,_ = self['data/%s'%self['data'].keys()[0]].shape ## 4D --> fragile
            self.nt  = nt
            self.t   =      np.array(range(self.nt), dtype=np.float64)
            self.ti  = ti = np.array(range(self.nt), dtype=np.int64)
            self.dt  = 1.
            self.duration = duration = self.t[-1]-self.t[0]
        
        else:
            self.t  = np.array([], dtype=np.float64)
            self.ti = np.array([], dtype=np.int64)
            self.nt = nt = 0
            self.dt = 0.
            self.duration = duration = 0.
        
        if verbose: print(72*'-')
        if verbose: even_print('nt', '%i'%self.nt )
        if verbose: even_print('dt', '%0.6f'%self.dt)
        if verbose: even_print('duration', '%0.2f'%self.duration )
        if hasattr(self, 'duration_avg'):
            if verbose: even_print('duration_avg', '%0.2f'%self.duration_avg )
        if verbose: print(72*'-'+'\n')
        
        # === ts group names & scalars
        
        if ('data' in self):
            self.scalars = list(self['data'].keys()) ## 4D : string names of scalars : ['u','v','w'],...
            self.n_scalars = len(self.scalars)
            self.scalars_dtypes = []
            for scalar in self.scalars:
                self.scalars_dtypes.append(self['data/%s'%scalar].dtype)
            self.scalars_dtypes_dict = dict(zip(self.scalars, self.scalars_dtypes)) ## dict {<<scalar>>: <<dtype>>}
        else:
            self.scalars = []
            self.n_scalars = 0
            self.scalars_dtypes = []
            self.scalars_dtypes_dict = dict(zip(self.scalars, self.scalars_dtypes))
        return
    
    # === I/O funcs
    
    @staticmethod
    def chunk_sizer(nxi, **kwargs):
        '''
        solve for an appropriate HDF5 chunk size based on dims
        '''
        
        size_kb     = kwargs.get('size_kb'    , 1024) ## target chunk size [KB]
        data_byte   = kwargs.get('data_byte'  , 4)    ## dtype size [B]
        constraint  = kwargs.get('constraint' , None) ## nxi constraints (fixed )
        base        = kwargs.get('base'       , 1)
        
        if not hasattr(nxi, '__iter__'):
            raise AssertionError('\'nxi\' must be an iterable')
        if constraint is None:
            constraint = [None for i in range(len(nxi))]
        if not hasattr(constraint, '__iter__'):
            raise AssertionError('\'constraint\' must be an iterable')
        if not (len(nxi)==len(constraint)):
            raise ValueError('nxi and constraint must have same size')
        if not isinstance(base, int):
            raise TypeError('\'base\' must be type int')
        
        if False: ## increment divisor on all axes
            
            div=2
            cc=-1
            while True:
                cc+=1
                chunks=[]
                for i in range(len(nxi)):
                    if constraint[i] is not None:
                        chunks.append(constraint[i])
                    else:
                        aa = max(math.ceil(nxi[i]/div),1)
                        bb = max(nxi[i]//div,1)
                        if (bb==1):
                            chunks.append(1)
                        else:
                            chunks.append(aa)
                
                chunk_size_kb = np.prod(chunks)*data_byte / 1024.
                #print('chunk size %0.1f [KB]'%chunk_size_kb)
                
                if (chunk_size_kb<=size_kb):
                    break
                else:
                    div+=1
                
                if (cc>int(1e5)):
                    raise RuntimeError('max iterations in rgd.chunk_sizer()')
        
        if True: ## increment divisor on largest axis
            
            nxi_ = [n for n in nxi] ## copy
            div  = [1 for i in range(len(nxi))]
            i_flexible = [i for i in range(len(nxi)) if constraint[i] is None]
            while True:
                chunks = [ max(math.ceil(nxi_[i]/div[i]),1) if (constraint[i] is None) else constraint[i] for i in range(len(nxi))]
                
                chunk_size_kb = np.prod(chunks)*data_byte / 1024.
                #print('chunk size %0.1f [KB]'%chunk_size_kb)
                
                if (chunk_size_kb<=size_kb):
                    break
                else:
                    aa = [i for i, j in enumerate(chunks) if (constraint[i] is None)]
                    bb = [j for i, j in enumerate(chunks) if (constraint[i] is None)]
                    i_gt = aa[np.argmax(bb)]
                    if (base==1):
                        div[i_gt] += 1
                    else:
                        div[i_gt] *= base
        
        return tuple(chunks)
    
    def init_from_eas4(self, fn_eas4, **kwargs):
        '''
        initialize an RGD from an EAS4 (NS3D output format)
        -----
        - x_min/max xi_min/max : min/max coord/index
        - stride filters (sx,sy,sz)
        '''
        
        verbose = kwargs.get('verbose',True)
        if (self.rank!=0):
            verbose=False
        
        # === spatial resolution filter : take every nth grid point
        sx = kwargs.get('sx',1)
        sy = kwargs.get('sy',1)
        sz = kwargs.get('sz',1)
        #st = kwargs.get('st',1)
        
        # === spatial resolution filter : set x/y/z bounds
        x_min = kwargs.get('x_min',None)
        y_min = kwargs.get('y_min',None)
        z_min = kwargs.get('z_min',None)
        
        x_max = kwargs.get('x_max',None)
        y_max = kwargs.get('y_max',None)
        z_max = kwargs.get('z_max',None)
        
        xi_min = kwargs.get('xi_min',None)
        yi_min = kwargs.get('yi_min',None)
        zi_min = kwargs.get('zi_min',None)
        
        xi_max = kwargs.get('xi_max',None)
        yi_max = kwargs.get('yi_max',None)
        zi_max = kwargs.get('zi_max',None)
        
        if verbose: print('\n'+'rgd.init_from_eas4()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        # even_print('infile', os.path.basename(fn_eas4))
        # even_print('infile size', '%0.2f [GB]'%(os.path.getsize(fn_eas4)/1024**3))
        # even_print('outfile', self.fname)
        
        if verbose: print('>>> infile : %s'%fn_eas4)
        if verbose: print('>>> infile size : %0.2f [GB]'%(os.path.getsize(fn_eas4)/1024**3))
        if verbose: print('>>> outfile : %s'%self.fname)
        
        with eas4(fn_eas4, 'r', verbose=False, driver='mpio', comm=MPI.COMM_WORLD, libver='latest') as hf_eas4:
            
            # === copy over header info if needed
            if all([('header/udef_real' in self),('header/udef_char' in self)]):
                raise ValueError('udef already present')
            else:
                udef         = hf_eas4.udef
                udef_real    = list(udef.values())
                udef_char    = list(udef.keys())
                udef_real_h5 = np.array(udef_real, dtype=np.float64)
                udef_char_h5 = np.array([s.encode('ascii', 'ignore') for s in udef_char], dtype='S128')
                
                self.create_dataset('header/udef_real', data=udef_real_h5, dtype=np.float64)
                self.create_dataset('header/udef_char', data=udef_char_h5, dtype='S128')
                self.udef      = udef
                self.udef_real = udef_real
                self.udef_char = udef_char
             
            # === copy over dims info
            if all([('dims/x' in self),('dims/y' in self),('dims/z' in self)]):
                pass
                ## future: 2D/3D handling here
            else:
                x, y, z = hf_eas4.x, hf_eas4.y, hf_eas4.z
                nx  = x.size
                ny  = y.size
                nz  = z.size
                ngp = nx*ny*nz
                # nt = hf_eas4.nt --> no time data available yet
                
                if any([(xi_min is not None),\
                        (xi_max is not None),\
                        (yi_min is not None),\
                        (yi_max is not None),\
                        (zi_min is not None),\
                        (zi_max is not None),\
                        (x_min is not None),\
                        (x_max is not None),\
                        (y_min is not None),\
                        (y_max is not None),\
                        (z_min is not None),\
                        (z_max is not None),\
                        (sx!=1),\
                        (sy!=1),\
                        (sz!=1)]):
                            hasFilters=True
                            if verbose: print('filtered dim info\n'+72*'-')
                else:
                    hasFilters=False
                
                # === index arrays along each axis
                xfi = np.array(range(nx),  dtype=np.int64)
                yfi = np.array(range(ny),  dtype=np.int64)
                zfi = np.array(range(nz),  dtype=np.int64)
                #tfi = np.array(range(nt), dtype=np.int64) --> no time data available yet
                # === total bounds clip (physical nondimensional distance)
                if (x_min is not None):
                    xfi = np.array([i for i in xfi if (x[i] >= x_min)])
                    if verbose: even_print('x_min', '%0.3f'%x_min)
                if (x_max is not None):
                    xfi = np.array([i for i in xfi if (x[i] <= x_max)])
                    if verbose: even_print('x_max', '%0.3f'%x_max)
                if (y_min is not None):
                    yfi = np.array([i for i in yfi if (y[i] >= y_min)])
                    if verbose: even_print('y_min', '%0.3f'%y_min)
                if (y_max is not None):
                    yfi = np.array([i for i in yfi if (y[i] <= y_max)])
                    if verbose: even_print('y_max', '%0.3f'%y_max)
                if (z_min is not None):
                    zfi = np.array([i for i in zfi if (z[i] >= z_min)])
                    if verbose: even_print('z_min', '%0.3f'%z_min)
                if (z_max is not None):
                    zfi = np.array([i for i in zfi if (z[i] <= z_max)])
                    if verbose: even_print('z_max', '%0.3f'%z_max)
                
                # === total bounds clip (coordinate index)
                if (xi_min is not None):
                    
                    xfi_ = []
                    if verbose: even_print('xi_min', '%i'%xi_min)
                    for c in xfi:
                        if (xi_min<0) and (c>=(nx+xi_min)): ## support negative indexing
                            xfi_.append(c)
                        elif (xi_min>=0) and (c>=xi_min):
                            xfi_.append(c)
                    xfi=np.array(xfi_, dtype=np.int64)

                if (xi_max is not None):
                    
                    xfi_ = []
                    if verbose: even_print('xi_max', '%i'%xi_max)
                    for c in xfi:
                        if (xi_max<0) and (c<=(nx+xi_max)): ## support negative indexing
                            xfi_.append(c)
                        elif (xi_max>=0) and (c<=xi_max):
                            xfi_.append(c)
                    xfi=np.array(xfi_, dtype=np.int64)
                
                if (yi_min is not None):
                    
                    yfi_ = []
                    if verbose: even_print('yi_min', '%i'%yi_min)
                    for c in yfi:
                        if (yi_min<0) and (c>=(ny+yi_min)): ## support negative indexing
                            yfi_.append(c)
                        elif (yi_min>=0) and (c>=yi_min):
                            yfi_.append(c)
                    yfi=np.array(yfi_, dtype=np.int64)

                if (yi_max is not None):
                    
                    yfi_ = []
                    if verbose: even_print('yi_max', '%i'%yi_max)
                    for c in yfi:
                        if (yi_max<0) and (c<=(ny+yi_max)): ## support negative indexing
                            yfi_.append(c)
                        elif (yi_max>=0) and (c<=yi_max):
                            yfi_.append(c)
                    yfi=np.array(yfi_, dtype=np.int64)
                
                if (zi_min is not None):
                    
                    zfi_ = []
                    if verbose: even_print('zi_min', '%i'%zi_min)
                    for c in zfi:
                        if (zi_min<0) and (c>=(nz+zi_min)): ## support negative indexing
                            zfi_.append(c)
                        elif (zi_min>=0) and (c>=zi_min):
                            zfi_.append(c)
                    zfi=np.array(zfi_, dtype=np.int64)

                if (zi_max is not None):
                    
                    zfi_ = []
                    if verbose: even_print('zi_max', '%i'%zi_max)
                    for c in zfi:
                        if (zi_max<0) and (c<=(nz+zi_max)): ## support negative indexing
                            zfi_.append(c)
                        elif (zi_max>=0) and (c<=zi_max):
                            zfi_.append(c)
                    zfi=np.array(zfi_, dtype=np.int64)
                
                # === resolution filter (skip every n grid points in each direction)
                if (sx!=1):
                    if verbose: even_print('sx', '%i'%sx)
                    xfi = xfi[::sx]
                if (sy!=1):
                    if verbose: even_print('sy', '%i'%sy)
                    yfi = yfi[::sy]
                if (sz!=1):
                    if verbose: even_print('sz', '%i'%sz)
                    zfi = zfi[::sz]
                # if (st!=1):
                #     even_print('st', '%i'%st)
                #     tfi = tfi[::st]
                # ===
                self.xfi = xfi
                self.yfi = yfi
                self.zfi = zfi
                #self.tfi = tfi
                # ===
                self.create_dataset('dims/xfi', data=xfi)
                self.create_dataset('dims/yfi', data=yfi)
                self.create_dataset('dims/zfi', data=zfi)
                #self.create_dataset('dims/tfi', data=tfi, maxshape=np.shape(tfi))
                
                if (xfi.size==0):
                    raise ValueError('x grid filter is empty... check!')
                if (yfi.size==0):
                    raise ValueError('y grid filter is empty... check!')
                if (zfi.size==0):
                    raise ValueError('z grid filter is empty... check!')
                #if (tfi.size==0):
                #    raise ValueError('t grid filter is empty... check!')
                
                # === determine if gridFilter is active
                self.hasGridFilter=False
                if not np.array_equal(xfi, np.array(range(nx), dtype=np.int64)):
                    self.hasGridFilter=True
                if not np.array_equal(yfi, np.array(range(ny), dtype=np.int64)):
                    self.hasGridFilter=True
                if not np.array_equal(zfi, np.array(range(nz), dtype=np.int64)):
                    self.hasGridFilter=True
                #if not np.array_equal(tfi, np.array(range(nt), dtype=np.int64)):
                #    self.hasGridFilter=True
                # === overwrite & write 1D filter arrays
                x = np.copy(x[xfi])
                y = np.copy(y[yfi])
                z = np.copy(z[zfi])
                #t = np.copy(t[tfi])
                
                ### if self.hasGridFilter:
                ###     print('>>> grid filter is present')
                ### else:
                ###     print('>>> no grid filter present')
                
                # === (over)write coord vecs if filter present
                if self.hasGridFilter:
                    
                    nx = x.size    
                    ny = y.size    
                    nz = z.size    
                    ngp = nx*ny*nz 
                    #nt = t.size   
                    
                    if verbose: even_print('nx',  '%i'%nx  )
                    if verbose: even_print('ny',  '%i'%ny  )
                    if verbose: even_print('nz',  '%i'%nz  )
                    if verbose: even_print('ngp', '%i'%ngp )
                    #if verbose: even_print('nt', '%i'%nt )
                    
                    self.nx  = nx
                    self.ny  = ny
                    self.nz  = nz
                    self.ngp = ngp
                    #self.nt = nt
                
                # === write 1D coord arrays
                if ('dims/x' in self):
                    del self['dims/x']
                self.create_dataset('dims/x', data=x)
                if ('dims/y' in self):
                    del self['dims/y']
                self.create_dataset('dims/y', data=y)
                if ('dims/z' in self):
                    del self['dims/z']
                self.create_dataset('dims/z', data=z)
                # if ('dims/t' in self):
                #     del self['dims/t']
                # self.create_dataset('dims/t', data=t, maxshape=t.shape)
                
                # === write 3D coord arrays
                if False:
                    size3DCoordArrays = 3*ngp*8/1024**2 ## [MB]
                    #print('>>> size of 3D coord arrays : %0.2f [GB]'%size3DCoordArrays)
                    if (size3DCoordArrays < 100): ## if less than 100 [MB]
                        xxx, yyy, zzz = np.meshgrid(x, y, z, indexing='ij')
                        self.create_dataset('dims/xxx', data=xxx.T)
                        self.create_dataset('dims/yyy', data=yyy.T)
                        self.create_dataset('dims/zzz', data=zzz.T)
        
        if hasFilters: print(72*'-'+'\n')
        self.get_header(verbose=True)
        return
    
    def init_from_rgd(self, fn_rgd, **kwargs):
        '''
        initialize an RGD from an RGD (copy over header data & coordinate data)
        '''
        
        t_info = kwargs.get('t_info',True)
        
        verbose = kwargs.get('verbose',True)
        if (self.rank!=0):
            verbose=False
        
        with rgd(fn_rgd, 'r', driver='mpio', comm=MPI.COMM_WORLD, libver='latest') as hf_ref:
            
            # === copy over header info if needed
            
            if all([('header/udef_real' in self),('header/udef_char' in self)]):
                raise ValueError('udef already present')
            else:
                udef         = hf_ref.udef
                udef_real    = list(udef.values())
                udef_char    = list(udef.keys())
                udef_real_h5 = np.array(udef_real, dtype=np.float64)
                udef_char_h5 = np.array([s.encode('ascii', 'ignore') for s in udef_char], dtype='S128')
                
                self.create_dataset('header/udef_real', data=udef_real_h5, maxshape=np.shape(udef_real_h5), dtype=np.float64)
                self.create_dataset('header/udef_char', data=udef_char_h5, maxshape=np.shape(udef_char_h5), dtype='S128')
                self.udef      = udef
                self.udef_real = udef_real
                self.udef_char = udef_char
            
            # === copy over spatial dim info
            
            x, y, z = hf_ref.x, hf_ref.y, hf_ref.z
            nx  = self.nx  = x.size
            ny  = self.ny  = y.size
            nz  = self.nz  = z.size
            ngp = self.ngp = nx*ny*nz
            if ('dims/x' in self):
                del self['dims/x']
            if ('dims/y' in self):
                del self['dims/y']
            if ('dims/z' in self):
                del self['dims/z']
            
            self.create_dataset('dims/x', data=x)
            self.create_dataset('dims/y', data=y)
            self.create_dataset('dims/z', data=z)
            
            # === copy over temporal dim info
            
            if t_info:
                self.t  = hf_ref.t
                self.nt = self.t.size
                self.create_dataset('dims/t', data=hf_ref.t)
            else:
                t = np.array([0.], dtype=np.float64)
                if ('dims/t' in self):
                    del self['dims/t']
                self.create_dataset('dims/t', data=t)
        
        self.get_header(verbose=False)
        return
    
    def import_eas4(self, fn_eas4_list, **kwargs):
        '''
        import data from a series of EAS4 files
        '''
        
        if (self.rank!=0):
            verbose=False
        else:
            verbose=True
        
        if verbose: print('\n'+'rgd.import_eas4()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        ti_min = kwargs.get('ti_min',None)
        ti_max = kwargs.get('ti_max',None)
        tt_min = kwargs.get('tt_min',None)
        tt_max = kwargs.get('tt_max',None)
        
        chunk_kb = kwargs.get('chunk_kb',2048)
        
        ## bt = kwargs.get('bt',1) ## buffer size t
        
        # === check for an often made mistake
        ts_min = kwargs.get('ts_min',None)
        ts_max = kwargs.get('ts_max',None)
        if (ts_min is not None):
            raise AssertionError('ts_min is not an option --> did you mean ti_min or tt_min?')
        if (ts_max is not None):
            raise AssertionError('ts_max is not an option --> did you mean ti_max or tt_max?')
        
        # === check that iterable of EAS4 files is OK
        if not hasattr(fn_eas4_list, '__iter__'):
            raise AssertionError('first arg \'fn_eas4_list\' must be iterable')
        for fn_eas4 in fn_eas4_list:
            if not os.path.isfile(fn_eas4):
                raise FileNotFoundError('%s not found!'%fn_eas4)
        
        # === ranks
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        rt = kwargs.get('rt',1)
        
        if (rx*ry*rz*rt != self.n_ranks):
            raise AssertionError('rx*ry*rz*rt != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        
        ## skip dimensions --> spatial skips done in init_from_XXX()
        # sx = kwargs.get('sx',1)
        # sy = kwargs.get('sy',1)
        # sz = kwargs.get('sz',1)
        st = kwargs.get('st',1)
        
        ## update this RGD's header and attributes
        self.get_header(verbose=False)
        
        # === get all time info
        
        comm_eas4 = MPI.COMM_WORLD
        
        t = np.array([], dtype=np.float64)
        for fn_eas4 in fn_eas4_list:
            with eas4(fn_eas4, 'r', verbose=False, driver='mpio', comm=comm_eas4, libver='latest') as hf_eas4:
                t = np.concatenate((t, hf_eas4.t))
        
        comm_eas4.Barrier()
        
        if verbose: even_print('n EAS4 files','%i'%len(fn_eas4_list))
        if verbose: even_print('nt all files','%i'%t.size)
        
        ## check no zero distance elements
        if (np.diff(t).size - np.count_nonzero(np.diff(t))) != 0.:
            raise AssertionError('t arr has zero-distance elements')
        else:
            if verbose: even_print('check: Δt!=0','passed')
        
        ## check monotonically increasing
        if not np.all(np.diff(t) > 0.):
            raise AssertionError('t arr not monotonically increasing')
        else:
            if verbose: even_print('check: t mono increasing','passed')
        
        ## check constant Δt
        dt0 = np.diff(t)[0]
        if not np.all(np.isclose(np.diff(t), dt0, rtol=1e-3)):
            if (self.rank==0): print(np.diff(t))
            raise AssertionError('t arr not uniformly spaced')
        else:
            if verbose: even_print('check: constant Δt','passed')
        
        # === resolution filter (skip every n timesteps)
        tfi = self.tfi = np.arange(t.size, dtype=np.int64)
        if (st!=1):
            if verbose: even_print('st', '%i'%st)
            #print('>>> st : %i'%st)
            tfi = self.tfi = tfi[::st]
        
        # === get doRead vector
        doRead = np.full((t.size,), True, dtype=bool)
        
        ## skip filter
        if hasattr(self, 'tfi'):
            doRead[np.isin(np.arange(t.size),self.tfi,invert=True)] = False
        
        ## min/max index filter
        if (ti_min is not None):
            if not isinstance(ti_min, int):
                raise TypeError('ti_min must be type int')
            doRead[:ti_min] = False
        if (ti_max is not None):
            if not isinstance(ti_max, int):
                raise TypeError('ti_max must be type int')
            doRead[ti_max:] = False
        
        if (tt_min is not None):
            if (tt_min>=0.):
                doRead[np.where((t-t.min())<tt_min)] = False
            elif (tt_min<0.):
                doRead[np.where((t-t.max())<tt_min)] = False
        
        if (tt_max is not None):
            if (tt_max>=0.):
                doRead[np.where((t-t.min())>tt_max)] = False
            elif (tt_max<0.):
                doRead[np.where((t-t.max())>tt_max)] = False
        
        # === RGD times
        self.t  = np.copy(t[doRead])
        self.nt = self.t.size
        self.ti = np.arange(self.nt, dtype=np.int64)
        
        # === write back self.t to file
        if ('dims/t' in self):
            del self['dims/t']
        self.create_dataset('dims/t', data=self.t)
        
        comm4d = self.comm.Create_cart(dims=[rx,ry,rz,rt], periods=[False,False,False,False], reorder=False)
        t4d = comm4d.Get_coords(self.rank)
        
        rxl_ = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
        ryl_ = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
        rzl_ = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
        #rtl_ = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))
        
        rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        # === determine RGD scalars (from EAS4 scalars)
        if not hasattr(self, 'scalars') or (len(self.scalars)==0):
            with eas4(fn_eas4_list[0], 'r', verbose=False, driver='mpio', comm=comm_eas4, libver='latest') as hf_eas4:
                self.scalars   = hf_eas4.scalars
                self.n_scalars = len(self.scalars)
        comm_eas4.Barrier()
        
        data_gb = 4*self.nt*self.nz*self.ny*self.nx / 1024**3
        
        # === initialize datasets
        for scalar in self.scalars:
            if verbose:
                even_print('initializing data/%s'%(scalar,),'%0.2f [GB]'%(data_gb,))
            
            shape  = (self.nt,self.nz,self.ny,self.nx)
            chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
            
            # chunk_kb_ = np.prod(chunks)*4 / 1024. ## actual
            # if verbose:
            #     even_print('chunk shape (t,z,y,x)','%s'%str(chunks))
            #     even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            dset = self.create_dataset('data/%s'%scalar, 
                                       shape=shape, 
                                       dtype=np.float32,
                                       #fillvalue=0.,
                                       #chunks=(1,min(self.nz//3,256),min(self.ny//3,256),min(self.nx//3,256)),
                                       chunks=chunks,
                                      )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
        if verbose: print(72*'-')
        
        # === report size of RGD after initialization
        if verbose: tqdm.write(even_print(os.path.basename(self.fname), '%0.2f [GB]'%(os.path.getsize(self.fname)/1024**3), s=True))
        if verbose: print(72*'-')
        
        # === open EAS4s, read, write to RGD
        if verbose:
            progress_bar = tqdm(total=(self.nt*self.n_scalars), ncols=100, desc='import', leave=False, file=sys.stdout)
        
        data_gb = 4*self.nx*self.ny*self.nz / 1024**3 ## per EAS4 ts
        data_gb_read  = 0.
        data_gb_write = 0.
        t_read  = 0.
        t_write = 0.
        
        #self.atomic = True
        
        tii  = -1 ## counter full series
        tiii = -1 ## counter RGD-local
        for fn_eas4 in fn_eas4_list:
            with eas4(fn_eas4, 'r', verbose=False, driver='mpio', comm=comm_eas4, libver='latest') as hf_eas4:
                #hf_eas4.atomic = True
                
                if verbose: tqdm.write(even_print(os.path.basename(fn_eas4), '%0.2f [GB]'%(os.path.getsize(fn_eas4)/1024**3), s=True))
                #if verbose: tqdm.write(even_print('gmode_dim1', hf_eas4.gmode_dim1, s=True))
                #if verbose: tqdm.write(even_print('gmode_dim2', hf_eas4.gmode_dim2, s=True))
                if verbose: tqdm.write(even_print('gmode_dim3', hf_eas4.gmode_dim3, s=True))
                if verbose: tqdm.write(even_print('duration', '%0.2f'%hf_eas4.duration, s=True))
                
                # === write buffer
                
                # ## 5D [scalar][x,y,z,t] structured array
                # buff = np.zeros(shape=(nxr, nyr, nzr, bt), dtype={'names':self.scalars, 'formats':self.scalars_dtypes})
                
                # ===
                
                domainName = 'DOMAIN_000000' ## only one domain supported
                for ti in range(hf_eas4.nt):
                    tii += 1 ## EAS4 series counter
                    if doRead[tii]:
                        tiii += 1 ## RGD counter
                        for scalar in hf_eas4.scalars:
                            if (scalar in self.scalars):
                                
                                # === collective read
                                
                                dset_path = 'Data/%s/ts_%06d/par_%06d'%(domainName,ti,hf_eas4.scalar_n_map[scalar])
                                dset = hf_eas4[dset_path]
                                
                                comm_eas4.Barrier()
                                t_start = timeit.default_timer()
                                with dset.collective:
                                    data = dset[rx1:rx2,ry1:ry2,rz1:rz2]
                                comm_eas4.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                
                                t_read       += t_delta
                                data_gb_read += data_gb
                                
                                if False:
                                    if verbose:
                                        txt = even_print('read', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                        tqdm.write(txt)
                                
                                # === collective write
                                
                                dset = self['data/%s'%scalar]
                                
                                self.comm.Barrier()
                                t_start = timeit.default_timer()
                                with dset.collective:
                                    dset[tiii,rz1:rz2,ry1:ry2,rx1:rx2] = data.T
                                self.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                
                                t_write       += t_delta
                                data_gb_write += data_gb
                                
                                if False:
                                    if verbose:
                                        txt = even_print('write', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                        tqdm.write(txt)
                                
                                if verbose:
                                    progress_bar.update()
        
        if verbose:
            progress_bar.close()
        
        comm_eas4.Barrier()
        self.comm.Barrier()
        self.get_header(verbose=False)
        
        if verbose: print(72*'-')
        if verbose: even_print('nt',       '%i'%self.nt )
        if verbose: even_print('dt',       '%0.6f'%self.dt )
        if verbose: even_print('duration', '%0.2f'%self.duration )
        if verbose: print(72*'-')
        if verbose: even_print('time read',format_time_string(t_read))
        if verbose: even_print('time write',format_time_string(t_write))
        if verbose: even_print(self.fname, '%0.2f [GB]'%(os.path.getsize(self.fname)/1024**3))
        if verbose: even_print('avg read speed','%0.3f [GB/s]'%(data_gb_read/t_read))
        if verbose: even_print('avg write speed','%0.3f [GB/s]'%(data_gb_write/t_write))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.import_eas4() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    # === test data populators
    
    def populate_abc_flow(self, **kwargs):
        '''
        populate (unsteady) ABC flow dummy data
        -----
        https://en.wikipedia.org/wiki/Arnold%E2%80%93Beltrami%E2%80%93Childress_flow
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.populate_abc_flow()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        ##
        chunk_kb = kwargs.get('chunk_kb',1024)
        
        self.nx = nx = kwargs.get('nx',100)
        self.ny = ny = kwargs.get('ny',100)
        self.nz = nz = kwargs.get('nz',100)
        self.nt = nt = kwargs.get('nt',100)
        
        data_gb = 3 * 4*nx*ny*nz*nt / 1024.**3
        if verbose: even_print(self.fname, '%0.2f [GB]'%(data_gb,))
        
        self.x = x = np.linspace(0., 2*np.pi, nx, dtype=np.float32)
        self.y = y = np.linspace(0., 2*np.pi, ny, dtype=np.float32)
        self.z = z = np.linspace(0., 2*np.pi, nz, dtype=np.float32)
        #self.t = t = np.linspace(0., 10.,     nt, dtype=np.float32)
        self.t = t = 0.1 * np.arange(nt, dtype=np.float32)
        
        if (rx*ry*rz != self.n_ranks):
            raise AssertionError('rx*ry*rz != self.n_ranks')
        
        # ===
        
        comm4d = self.comm.Create_cart(dims=[rx,ry,rz], periods=[False,False,False], reorder=False)
        t4d = comm4d.Get_coords(self.rank)
        
        rxl_ = np.array_split(np.arange(self.nx,dtype=np.int64),min(rx,self.nx))
        ryl_ = np.array_split(np.arange(self.ny,dtype=np.int64),min(ry,self.ny))
        rzl_ = np.array_split(np.arange(self.nz,dtype=np.int64),min(rz,self.nz))
        #rtl_ = np.array_split(np.arange(self.nt,dtype=np.int64),min(rt,self.nt))
        
        rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        ## per-rank dim range
        xr = x[rx1:rx2]
        yr = y[ry1:ry2]
        zr = z[rz1:rz2]
        #tr = t[rt1:rt2]
        tr = np.copy(t)
        
        ## write dims
        self.create_dataset('dims/x', data=x)
        self.create_dataset('dims/y', data=y)
        self.create_dataset('dims/z', data=z)
        self.create_dataset('dims/t', data=t)
        
        shape  = (self.nt,self.nz,self.ny,self.nx)
        chunks = rgdmpi.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
        
        ## initialize
        data_gb = 4*nx*ny*nz*nt / 1024.**3
        for scalar in ['u','v','w']:
            if ('data/%s'%scalar in self):
                del self['data/%s'%scalar]
            if verbose:
                even_print('initializing data/%s'%(scalar,),'%0.2f [GB]'%(data_gb,))
            dset = self.create_dataset('data/%s'%scalar, 
                                        shape=shape,
                                        dtype=np.float32,
                                        chunks=chunks,
                                        )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
        # === make 4D ABC flow data
        
        t_start = timeit.default_timer()
        A = np.sqrt(3)
        B = np.sqrt(2)
        C = 1.
        na = np.newaxis
        u = (A + 0.5 * tr[na,na,na,:] * np.sin(np.pi*tr[na,na,na,:])) * np.sin(zr[na,na,:,na]) + \
            B * np.cos(yr[na,:,na,na]) + \
            0.*xr[:,na,na,na]
        v = B * np.sin(xr[:,na,na,na]) + \
            C * np.cos(zr[na,na,:,na]) + \
            0.*yr[na,:,na,na] + \
            0.*tr[na,na,na,:]
        w = C * np.sin(yr[na,:,na,na]) + \
            (A + 0.5 * tr[na,na,na,:] * np.sin(np.pi*tr[na,na,na,:])) * np.cos(xr[:,na,na,na]) + \
            0.*zr[na,na,:,na]
        
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('calc flow','%0.3f [s]'%(t_delta,))
        
        # ===
        
        data_gb = 4*nx*ny*nz*nt / 1024.**3
        
        self.comm.Barrier()
        t_start = timeit.default_timer()
        ds = self['data/u']
        with ds.collective:
            ds[:,rz1:rz2,ry1:ry2,rx1:rx2] = u.T
        self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('write: u','%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        self.comm.Barrier()
        t_start = timeit.default_timer()
        ds = self['data/v']
        with ds.collective:
            ds[:,rz1:rz2,ry1:ry2,rx1:rx2] = v.T
        self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('write: v','%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        self.comm.Barrier()
        t_start = timeit.default_timer()
        ds = self['data/w']
        with ds.collective:
            ds[:,rz1:rz2,ry1:ry2,rx1:rx2] = w.T
        self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('write: w','%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        # ===
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.populate_abc_flow() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    def populate_perlin_noise(self, **kwargs):
        '''
        populate Perlin noise
        '''
        raise NotImplementedError('populate_perlin_noise() not yet implmented')
        return
    
    # === post-processing
    
    def get_mean(self, **kwargs):
        '''
        get mean in [t] --> leaves [x,y,z,1]
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.get_mean()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        #rt = kwargs.get('rt',1)
        #rt = 1
        
        fn_rgd_mean  = kwargs.get('fn_rgd_mean',None)
        #sfm         = kwargs.get('scalars',None) ## scalars to take (for mean)
        favre        = kwargs.get('favre',True)
        reynolds     = kwargs.get('reynolds',True)
        force        = kwargs.get('force',False)
        
        chunk_kb     = kwargs.get('chunk_kb',2048)
        
        if (rx*ry*rz != self.n_ranks):
            raise AssertionError('rx*ry*rz != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        
        comm4d = self.comm.Create_cart(dims=[rx,ry,rz], periods=[False,False,False], reorder=False)
        t4d = comm4d.Get_coords(self.rank)
        
        rxl_ = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
        ryl_ = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
        rzl_ = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
        #rtl_ = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))
        
        rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        # === mean file name (for writing)
        if (fn_rgd_mean is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_mean_h5_base = fname_root+'_mean.h5'
            #fn_rgd_mean = os.path.join(fname_path, fname_mean_h5_base)
            fn_rgd_mean = str(PurePosixPath(fname_path, fname_mean_h5_base))
            #fn_rgd_mean = Path(fname_path, fname_mean_h5_base)
        
        if verbose: even_print('fn_rgd'       , self.fname   )
        if verbose: even_print('fn_rgd_mean'  , fn_rgd_mean  )
        #if verbose: even_print('fn_rgd_prime' , fn_rgd_prime )
        if verbose: even_print('do Favre avg' , str(favre)   )
        if verbose: even_print('do Reynolds avg' , str(reynolds)   )
        if verbose: print(72*'-')
        
        t_read = 0.
        t_write = 0.
        data_gb_read = 0.
        data_gb_write = 0.
        
        data_gb      = 4*self.nx*self.ny*self.nz*self.nt / 1024**3
        data_gb_mean = 4*self.nx*self.ny*self.nz*1       / 1024**3
        
        scalars_re = ['u','v','w','p','T','rho']
        scalars_fv = ['u','v','w','p','T','rho']
        
        with rgd(fn_rgd_mean, 'w', force=force, driver='mpio', comm=MPI.COMM_WORLD, libver='latest') as hf_mean:
            
            hf_mean.attrs['duration_avg'] = self.t[-1] - self.t[0] ## add attribute for duration of mean
            #hf_mean.attrs['duration_avg'] = self.duration
            
            hf_mean.init_from_rgd(self.fname) ## initialize the mean file from the rgd file
            
            # === initialize mean datasets
            for scalar in self.scalars:
                
                shape  = (1,self.nz,self.ny,self.nx)
                chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
                
                if reynolds:
                    if ('data/%s'%scalar in hf_mean):
                        del hf_mean['data/%s'%scalar]
                    if (self.rank==0):
                        even_print('initializing data/%s'%(scalar,),'%0.3f [GB]'%(data_gb_mean,))
                    dset = hf_mean.create_dataset('data/%s'%scalar,
                                                  shape=shape,
                                                  dtype=np.float32,
                                                  chunks=chunks,
                                                  )
                    hf_mean.scalars.append('data/%s'%scalar)
                    
                    chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                    if verbose:
                        even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                        even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
                
                if favre:
                    if (scalar in scalars_fv):
                        if ('data/%s_fv'%scalar in hf_mean):
                            del hf_mean['data/%s_fv'%scalar]
                        if (self.rank==0):
                            even_print('initializing data/%s_fv'%(scalar,),'%0.3f [GB]'%(data_gb_mean,))
                        dset = hf_mean.create_dataset('data/%s_fv'%scalar,
                                                      shape=shape,
                                                      dtype=np.float32,
                                                      chunks=chunks,
                                                      )
                        hf_mean.scalars.append('data/%s_fv'%scalar)
                        
                        chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                        if verbose:
                            even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                            even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            self.comm.Barrier()
            if verbose: print(72*'-')
            
            # === read rho
            if favre:
                dset = self['data/rho']
                self.comm.Barrier()
                t_start = timeit.default_timer()
                with dset.collective:
                    rho = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                self.comm.Barrier()
                
                t_delta = timeit.default_timer() - t_start
                if (self.rank==0):
                    txt = even_print('read: rho', '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                    tqdm.write(txt)
                
                t_read       += t_delta
                data_gb_read += data_gb
                
                rho_mean = np.mean(rho, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32) ## mean in [t] --> leave [x,y,z]
            
            # === read, do mean, write
            for scalar in self.scalars:
                
                # === collective read
                dset = self['data/%s'%scalar]
                self.comm.Barrier()
                t_start = timeit.default_timer()
                with dset.collective:
                    data = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                
                if (self.rank==0):
                    txt = even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                    tqdm.write(txt)
                
                t_read       += t_delta
                data_gb_read += data_gb
                
                # === do mean
                if reynolds:
                    data_mean    = np.mean(data,     axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
                if favre:
                    data_mean_fv = np.mean(data*rho, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32) / rho_mean
                
                # === write
                if reynolds:
                    if scalar in scalars_re:
                        
                        dset = hf_mean['data/%s'%scalar]
                        self.comm.Barrier()
                        t_start = timeit.default_timer()
                        with dset.collective:
                            dset[:,rz1:rz2,ry1:ry2,rx1:rx2] = data_mean.T
                        self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        
                        if (self.rank==0):
                            txt = even_print('write: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                            tqdm.write(txt)
                        
                        t_write       += t_delta
                        data_gb_write += data_gb_mean
                
                if favre:
                    if scalar in scalars_fv:
                        
                        dset = hf_mean['data/%s_fv'%scalar]
                        self.comm.Barrier()
                        t_start = timeit.default_timer()
                        with dset.collective:
                            dset[:,rz1:rz2,ry1:ry2,rx1:rx2] = data_mean_fv.T.astype(np.float32)
                        self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        
                        if (self.rank==0):
                            txt = even_print('write: %s_fv'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                            tqdm.write(txt)
                        
                        t_write       += t_delta
                        data_gb_write += data_gb_mean
            
            self.comm.Barrier()
            if verbose: print(72*'-')
            
            # === replace dims/t array --> take last time of series
            t = np.array([self.t[-1]],dtype=np.float64)
            if ('dims/t' in hf_mean):
                del hf_mean['dims/t']
            hf_mean.create_dataset('dims/t', data=t)
            
            if hasattr(hf_mean, 'duration_avg'):
                if verbose: even_print('duration_avg', '%0.2f'%hf_mean.duration_avg)
        
        if verbose: print(72*'-')
        if verbose: even_print('time read',format_time_string(t_read))
        if verbose: even_print('time write',format_time_string(t_write))
        if verbose: even_print(fn_rgd_mean, '%0.2f [GB]'%(os.path.getsize(fn_rgd_mean)/1024**3))
        if verbose: even_print('avg read speed','%0.3f [GB/s]'%(data_gb_read/t_read))
        if verbose: even_print('avg write speed','%0.3f [GB/s]'%(data_gb_write/t_write))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.get_mean() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    def get_prime(self, **kwargs):
        '''
        get mean-removed (prime) variables
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.get_prime()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        #rt = kwargs.get('rt',1)
        ct = kwargs.get('ct',1) ## n chunks [t]
        
        fn_rgd_mean  = kwargs.get('fn_rgd_mean',None)
        fn_rgd_prime = kwargs.get('fn_rgd_prime',None)
        sfp          = kwargs.get('scalars',None) ## scalars to take (for prime)
        favre        = kwargs.get('favre',True)
        reynolds     = kwargs.get('reynolds',True)
        force        = kwargs.get('force',False)
        
        chunk_kb     = kwargs.get('chunk_kb',2048) ## 2 [MB]
        
        if (rx*ry*rz != self.n_ranks):
            raise AssertionError('rx*ry*rz != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        
        # === ranks
        
        comm4d = self.comm.Create_cart(dims=[rx,ry,rz], periods=[False,False,False], reorder=False)
        t4d = comm4d.Get_coords(self.rank)
        
        rxl_ = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
        ryl_ = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
        rzl_ = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
        #rtl_ = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))
        
        rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        # === chunks
        
        ctl_ = np.array_split(np.arange(self.nt),min(ct,self.nt))
        ctl  = [[b[0],b[-1]+1] for b in ctl_ ]
        if verbose:
            even_print('ct','%i'%ct)
        
        # === mean file name (for reading)
        if (fn_rgd_mean is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_mean_h5_base = fname_root+'_mean.h5'
            #fn_rgd_mean = os.path.join(fname_path, fname_mean_h5_base)
            fn_rgd_mean = str(PurePosixPath(fname_path, fname_mean_h5_base))
            #fn_rgd_mean = Path(fname_path, fname_mean_h5_base)
        
        if not os.path.isfile(fn_rgd_mean):
            raise FileNotFoundError('%s not found!'%fn_rgd_mean)
        
        # === prime file name (for writing)
        if (fn_rgd_prime is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_prime_h5_base = fname_root+'_prime.h5'
            #fn_rgd_prime = os.path.join(fname_path, fname_prime_h5_base)
            fn_rgd_prime = str(PurePosixPath(fname_path, fname_prime_h5_base))
            #fn_rgd_prime = Path(fname_path, fname_prime_h5_base)
        
        if verbose: even_print('fn_rgd'          , self.fname    )
        if verbose: even_print('fn_rgd_mean'     , fn_rgd_mean   )
        if verbose: even_print('fn_rgd_prime'    , fn_rgd_prime  )
        if verbose: even_print('do Favre avg'    , str(favre)    )
        if verbose: even_print('do Reynolds avg' , str(reynolds) )
        if verbose: print(72*'-')
        
        t_read = 0.
        t_write = 0.
        data_gb_read = 0.
        data_gb_write = 0.
        
        data_gb      = 4*self.nx*self.ny*self.nz*self.nt / 1024**3
        data_gb_mean = 4*self.nx*self.ny*self.nz*1       / 1024**3
        
        scalars_fv = ['u','v','w','p','T','rho']
        scalars_re = ['u','v','w','p','T','rho']
        
        # ===
        
        comm_rgd_prime = MPI.COMM_WORLD
        
        with rgd(fn_rgd_prime, 'w', force=force, driver='mpio', comm=comm_rgd_prime, libver='latest') as hf_prime:
            hf_prime.init_from_rgd(self.fname)
            
            # === initialize prime datasets
            for scalar in self.scalars:
                
                shape  = (self.nt,self.nz,self.ny,self.nx)
                chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
                
                if reynolds:
                    if (scalar in scalars_re):
                        if ('data/%sI'%scalar in hf_prime):
                            del hf_prime['data/%sI'%scalar]
                        if verbose:
                            even_print('initializing data/%sI'%(scalar,),'%0.1f [GB]'%(data_gb,))
                        dset = hf_prime.create_dataset('data/%sI'%scalar,
                                                       shape=shape,
                                                       dtype=np.float32,
                                                       chunks=chunks,
                                                       )
                        hf_prime.scalars.append('data/%sI'%scalar)
                        
                        chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                        if verbose:
                            even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                            even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
                
                if favre:
                    if (scalar in scalars_fv):
                        if ('data/%sII'%scalar in hf_prime):
                            del hf_prime['data/%sII'%scalar]
                        if verbose:
                            even_print('initializing data/%sII'%(scalar,),'%0.1f [GB]'%(data_gb,))
                        dset = hf_prime.create_dataset('data/%sII'%scalar,
                                                       shape=shape,
                                                       dtype=np.float32,
                                                       chunks=chunks,
                                                       )
                        hf_prime.scalars.append('data/%sII'%scalar)
                        
                        chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                        if verbose:
                            even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                            even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            comm_rgd_prime.Barrier()
            self.comm.Barrier()
            if verbose: print(72*'-')
            
            # === read unsteady + mean, do difference, write
            
            comm_rgd_mean = MPI.COMM_WORLD
            
            with rgd(fn_rgd_mean, 'r', driver='mpio', comm=comm_rgd_mean, libver='latest') as hf_mean:
                
                if verbose:
                    progress_bar = tqdm(total=ct*self.n_scalars, ncols=100, desc='prime', leave=False, file=sys.stdout)
                
                for ctl_ in ctl:
                    ct1, ct2 = ctl_
                    ntc = ct2 - ct1
                    
                    data_gb = 4*self.nx*self.ny*self.nz*ntc / 1024**3 ## this chunk [GB]
                    
                    # === read rho (if necessary)
                    if favre:
                        dset = self['data/rho']
                        self.comm.Barrier()
                        t_start = timeit.default_timer()
                        with dset.collective:
                            rho = dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2].T
                        self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        # if verbose:
                        #     txt = even_print('read: rho', '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                        #     tqdm.write(txt)
                        t_read       += t_delta
                        data_gb_read += data_gb
                    
                    for scalar in self.scalars:
                        
                        if (scalar in scalars_re) or (scalar in scalars_fv):
                            
                            # === read RGD data
                            dset = self['data/%s'%scalar]
                            self.comm.Barrier()
                            t_start = timeit.default_timer()
                            with dset.collective:
                                data = dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2].T
                            self.comm.Barrier()
                            t_delta = timeit.default_timer() - t_start
                            # if verbose:
                            #     txt = even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                            #     tqdm.write(txt)
                            t_read       += t_delta
                            data_gb_read += data_gb
                            
                            # === do prime Reynolds
                            if (scalar in scalars_re) and reynolds:
                                
                                ## read Reynolds avg
                                dset = hf_mean['data/%s'%scalar]
                                hf_mean.comm.Barrier()
                                t_start = timeit.default_timer()
                                with dset.collective:
                                    data_mean_re = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                                hf_mean.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                # if verbose:
                                #     txt = even_print('read: %s (Re avg)'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                                #     tqdm.write(txt)
                                
                                ## calc mean-removed Reynolds
                                data_prime_re = data - data_mean_re
                                
                                ## write Reynolds prime
                                dset = hf_prime['data/%sI'%scalar]
                                hf_prime.comm.Barrier()
                                t_start = timeit.default_timer()
                                with dset.collective:
                                    dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2] = data_prime_re.T
                                hf_prime.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                t_write       += t_delta
                                data_gb_write += data_gb
                                # if verbose:
                                #     txt = even_print('write: %sI'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                #     tqdm.write(txt)
                            
                            # === do prime Favre
                            if (scalar in scalars_fv) and favre:
                                
                                ## read Favre avg
                                dset = hf_mean['data/%s_fv'%scalar]
                                hf_mean.comm.Barrier()
                                t_start = timeit.default_timer()
                                with dset.collective:
                                    data_mean_fv = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                                hf_mean.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                # if verbose:
                                #     txt = even_print('read: %s (Fv avg)'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                                #     tqdm.write(txt)
                                
                                ## do Favre prime
                                data_prime_fv = data - data_mean_fv
                                
                                ## write Favre prime
                                dset = hf_prime['data/%sII'%scalar]
                                hf_prime.comm.Barrier()
                                t_start = timeit.default_timer()
                                with dset.collective:
                                    dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2] = data_prime_fv.T
                                hf_prime.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                t_write       += t_delta
                                data_gb_write += data_gb
                                # if verbose:
                                #     txt = even_print('write: %sII'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                #     tqdm.write(txt)
                        
                        self.comm.Barrier()
                        comm_rgd_prime.Barrier()
                        comm_rgd_mean.Barrier()
                        if verbose: progress_bar.update()
                
                if verbose:
                    progress_bar.close()
            
            comm_rgd_mean.Barrier()
        comm_rgd_prime.Barrier()
        self.comm.Barrier()
        
        # ===
        
        #if verbose: print(72*'-')
        if verbose: even_print('time read',format_time_string(t_read))
        if verbose: even_print('time write',format_time_string(t_write))
        if verbose: even_print(fn_rgd_prime, '%0.2f [GB]'%(os.path.getsize(fn_rgd_prime)/1024**3))
        if verbose: even_print('avg read speed','%0.3f [GB/s]'%(data_gb_read/t_read))
        if verbose: even_print('avg write speed','%0.3f [GB/s]'%(data_gb_write/t_write))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.get_prime() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    def get_mean_dim(self, **kwargs):
        '''
        get dimensionalized [x,z] mean
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.get_mean_dim()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        rt = kwargs.get('rt',1)
        
        ## always only one rank in y and t!!!
        if (ry != 1):
            raise AssertionError('ry != 1')
        if (rt != 1):
            raise AssertionError('rt != 1')
        
        if (rx*rz != self.n_ranks):
            raise AssertionError('rx*rz != self.n_ranks')
        if (self.nt != 1):
            raise AssertionError('self.nt != 1 --> rgd.get_mean_dim() should only be run on mean files, i.e. output from rgd.get_mean()')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        
        comm4d = self.comm.Create_cart(dims=[rx,ry,rz,rt], periods=[False,False,False,False], reorder=False)
        t4d = comm4d.Get_coords(self.rank)
        
        rxl_ = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
        #ryl_ = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
        rzl_ = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
        #rtl_ = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))
        
        rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        #ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        #ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        nyr = self.ny
        
        fn_dat_mean_dim  = kwargs.get('fn_dat_mean_dim',None)
        #fn_h5_mean_dim   = kwargs.get('fn_h5_mean_dim',None)
        
        # === mean (dim) file name (for writing) : .dat
        if (fn_dat_mean_dim is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            #fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            #fname_dat_mean_base = fname_root+'_mean_dim.dat'
            fname_dat_mean_base = fname_root+'_dim.dat'
            fn_dat_mean_dim = str(PurePosixPath(fname_path, fname_dat_mean_base))
        
        # # === mean (dim) file name (for writing) : .h5
        # if (fn_h5_mean_dim is None):
        #     fname_path = os.path.dirname(self.fname)
        #     fname_base = os.path.basename(self.fname)
        #     fname_root, fname_ext = os.path.splitext(fname_base)
        #     fname_h5_mean_base = fname_root+'_dim.h5'
        #     fn_h5_mean_dim = str(PurePosixPath(fname_path, fname_h5_mean_base))
        
        #if verbose: even_print('fn_rgd_mean'     , self.fname       )
        if verbose: even_print('fn_dat_mean_dim' , fn_dat_mean_dim  )
        if verbose: print(72*'-')
        
        # ===
        
        data = {} ## the container dict that will be returned
        
        nx = self.nx ; data['nx'] = nx
        ny = self.ny ; data['ny'] = ny
        nz = self.nz ; data['nz'] = nz
        
        # === write all header attributes (and derived ones)
        for key in self.udef:
            data[key] = self.udef[key]
        for key in self.udef_deriv:
            data[key] = self.udef_deriv[key]
        
        xs = np.copy(self.x) ; data['xs'] = xs ## dimensionless (inlet)
        ys = np.copy(self.y) ; data['ys'] = ys ## dimensionless (inlet)
        zs = np.copy(self.z) ; data['zs'] = zs ## dimensionless (inlet)
        
        x = self.x * self.lchar ; data['x'] = x ## dimensional [m]
        y = self.y * self.lchar ; data['y'] = y ## dimensional [m]
        z = self.z * self.lchar ; data['z'] = z ## dimensional [m]
        
        # === 5D [scalar][x,y,z,t] structured array
        dataScalar = np.zeros(shape=(nxr, self.ny, nzr, 1), dtype={'names':self.scalars, 'formats':self.scalars_dtypes})
        
        data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
        
        for scalar in self.scalars:
            dset = self['data/%s'%scalar]
            self.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                dataScalar[scalar] = dset[:,rz1:rz2,:,rx1:rx2].T
            self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            if (self.rank==0):
                even_print('read: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        # === dimensionalize
        
        u   =  self.U_inf                     * np.squeeze(np.copy(dataScalar['u'])   ) ; data['u']   = u
        v   =  self.U_inf                     * np.squeeze(np.copy(dataScalar['v'])   ) ; data['v']   = v
        w   =  self.U_inf                     * np.squeeze(np.copy(dataScalar['w'])   ) ; data['w']   = w
        rho =  self.rho_inf                   * np.squeeze(np.copy(dataScalar['rho']) ) ; data['rho'] = rho
        p   =  (self.rho_inf * self.U_inf**2) * np.squeeze(np.copy(dataScalar['p'])   ) ; data['p']   = p
        T   =  self.T_inf                     * np.squeeze(np.copy(dataScalar['T'])   ) ; data['T']   = T
        
        umag = np.sqrt(u**2+v**2+w**2) ; data['umag'] = umag
        mflux = umag*rho               ; data['mflux'] = mflux
        
        mu = (14.58e-7 * T**1.5) / (T+110.4)      ; data['mu']  = mu
        M  = u / np.sqrt(self.kappa * self.R * T) ; data['M']   = M
        nu = mu / rho                             ; data['nu']  = nu
        
        hiOrder=True
        if hiOrder:
            
            dudx   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dudy   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dvdx   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dvdy   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dTdx   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dTdy   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dpdx   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dpdy   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            drhodx = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            drhody = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            
            ## y-gradients
            if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='grad', leave=False, file=sys.stdout)
            for i in range(nxr):
                for k in range(nzr):
                    
                    if True:
                        dudy[i,:,k]   = sp.interpolate.CubicSpline(y,   u[i,:,k], bc_type='natural')(y,1)
                        dvdy[i,:,k]   = sp.interpolate.CubicSpline(y,   v[i,:,k], bc_type='natural')(y,1)
                        dTdy[i,:,k]   = sp.interpolate.CubicSpline(y,   T[i,:,k], bc_type='natural')(y,1)
                        dpdy[i,:,k]   = sp.interpolate.CubicSpline(y,   p[i,:,k], bc_type='natural')(y,1)
                        drhody[i,:,k] = sp.interpolate.CubicSpline(y, rho[i,:,k], bc_type='natural')(y,1)
                    
                    if False:
                        dudy[i,:,k]   = sp.interpolate.pchip(y,   u[i,:,k])(y,1)
                        dvdy[i,:,k]   = sp.interpolate.pchip(y,   v[i,:,k])(y,1)
                        dTdy[i,:,k]   = sp.interpolate.pchip(y,   T[i,:,k])(y,1)
                        dpdy[i,:,k]   = sp.interpolate.pchip(y,   p[i,:,k])(y,1)
                        drhody[i,:,k] = sp.interpolate.pchip(y, rho[i,:,k])(y,1)
                    
                    if verbose: progress_bar.update()
            
            ## x-gradients --> only need for pseudovel --> not available (currently) MPI implementation
            if False:
                for j in range(self.ny):
                    for k in range(nzr):
                        
                        if True:
                            dudx[:,j,k]   = sp.interpolate.CubicSpline(x,   u[:,j,k], bc_type='natural')(x,1)
                            dvdx[:,j,k]   = sp.interpolate.CubicSpline(x,   v[:,j,k], bc_type='natural')(x,1)
                            dTdx[:,j,k]   = sp.interpolate.CubicSpline(x,   T[:,j,k], bc_type='natural')(x,1)
                            dpdx[:,j,k]   = sp.interpolate.CubicSpline(x,   p[:,j,k], bc_type='natural')(x,1)
                            drhodx[:,j,k] = sp.interpolate.CubicSpline(x, rho[:,j,k], bc_type='natural')(x,1)
                        
                        if False:
                            dudx[:,j,k]   = sp.interpolate.pchip(x,   u[:,j,k])(x,1)
                            dvdx[:,j,k]   = sp.interpolate.pchip(x,   v[:,j,k])(x,1)
                            dTdx[:,j,k]   = sp.interpolate.pchip(x,   T[:,j,k])(x,1)
                            dpdx[:,j,k]   = sp.interpolate.pchip(x,   p[:,j,k])(x,1)
                            drhodx[:,j,k] = sp.interpolate.pchip(x, rho[:,j,k])(x,1)
                    
                    if verbose: progress_bar.update()
            if verbose: progress_bar.close()
        
        else: ## lower order gradients
            
            dudy   = np.gradient(u,   y, edge_order=1, axis=1)
            dvdy   = np.gradient(v,   y, edge_order=1, axis=1)
            dpdy   = np.gradient(p,   y, edge_order=1, axis=1)
            dTdy   = np.gradient(T,   y, edge_order=1, axis=1)
            drhody = np.gradient(rho, y, edge_order=1, axis=1)
            
            if False:
                dudx   = np.gradient(u,   x, edge_order=1, axis=0)
                dvdx   = np.gradient(v,   x, edge_order=1, axis=0)
                dpdx   = np.gradient(p,   x, edge_order=1, axis=0)
                dTdx   = np.gradient(T,   x, edge_order=1, axis=0)
                drhodx = np.gradient(rho, x, edge_order=1, axis=0)
        
        # ===
        
        if False: # no x-gradients
            data['dudx']   = dudx  
            data['dvdx']   = dvdx  
            data['dTdx']   = dTdx  
            data['dpdx']   = dpdx  
            data['drhodx'] = drhodx
        
        data['dudy']   = dudy
        data['dvdy']   = dvdy
        data['dTdy']   = dTdy
        data['dpdy']   = dpdy
        data['drhody'] = drhody
        
        # ===
        
        if False:
            vort_z = dvdx - dudy
            data['vort_z'] = vort_z
        
        ## wall-adjacent values
        dudy_wall = np.squeeze(dudy[:,0,:])
        rho_wall  = np.squeeze(rho[:,0,:])
        nu_wall   = np.squeeze(nu[:,0,:])
        mu_wall   = np.squeeze(mu[:,0,:])
        T_wall    = np.squeeze(T[:,0,:])
        tau_wall  = mu_wall * dudy_wall
        q_wall    = self.cp * mu_wall / self.Pr * np.squeeze(dTdy[:,0,:]) ### wall heat flux
        
        data['dudy_wall'] = dudy_wall
        data['rho_wall']  = rho_wall
        data['nu_wall']   = nu_wall
        data['mu_wall']   = mu_wall
        data['T_wall']    = T_wall
        data['tau_wall']  = tau_wall
        data['q_wall']    = q_wall
        
        u_tau  = np.sqrt(tau_wall/rho_wall)
        y_plus = y[np.newaxis,:,np.newaxis] * u_tau[:,np.newaxis,:] / nu_wall[:,np.newaxis,:]
        u_plus = u / u_tau[:,np.newaxis,:]
        
        data['u_tau']  = u_tau
        data['y_plus'] = y_plus
        data['u_plus'] = u_plus
        
        # === BL edge & 99 values
        
        j_edge     = np.zeros(shape=(nxr,nzr), dtype=np.int32)
        y_edge     = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        u_edge     = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        v_edge     = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        T_edge     = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        p_edge     = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        rho_edge   = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        nu_edge    = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        #psvel_edge = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        M_edge     = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        
        d99        = np.zeros(shape=(nxr,nzr), dtype=np.float64) ## δ₉₉ --> interpolated
        d99j       = np.zeros(shape=(nxr,nzr), dtype=np.int32)   ## closest y-index to δ₉₉
        d99g       = np.zeros(shape=(nxr,nzr), dtype=np.float64) ## δ₉₉ at nearest grid point
        
        u99        = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        v99        = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        T99        = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        p99        = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        rho99      = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        nu99       = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        #psvel99    = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        M99        = np.zeros(shape=(nxr,nzr), dtype=np.float64)

        # === get pseudo-velocity (wall-normal integration of z-vorticity)
        if False:
            psvel = np.zeros(shape=(nxr,nyr,nzr), dtype=np.float64)
            if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='psvel', leave=False, file=sys.stdout)
            for i in range(nxr):
                for k in range(nzr):
                    psvel[i,:,k] = sp.integrate.cumtrapz(-1*vort_z[i,:,k], y, initial=0.)
                    if verbose: progress_bar.update()
            if verbose: progress_bar.close()
            
            psvel_ddy = np.zeros(shape=(nxr,nyr,nzr), dtype=np.float64)
            if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='psvel_ddy', leave=False, file=sys.stdout)
            for i in range(nxr):
                for k in range(nzr):
                    psvel_ddy[i,:,k] = sp.interpolate.CubicSpline(y,psvel[i,:,k],bc_type='natural')(y,1)
                    if verbose: progress_bar.update()
            if verbose: progress_bar.close()
            
            data['psvel']     = psvel
            data['psvel_ddy'] = psvel_ddy
        
        ## u criteria
        if True:
            j_edge_3 = np.zeros(shape=(nx,nz), dtype=np.int32)
            if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='umax', leave=False, file=sys.stdout)
            for i in range(nxr):
                for k in range(nzr):
                    umax=u[i,:,k].max()
                    for j in range(nyr):
                        if math.isclose(u[i,j,k], umax, rel_tol=1.1*8e-4):
                            j_edge_3[i,k] = j
                            break
                        if (u[i,j,k]>umax):
                            j_edge_3[i,k] = j-1
                            break
                    if verbose: progress_bar.update()
            if verbose: progress_bar.close()
        
        ## umag criteria
        if True:
            j_edge_4 = np.zeros(shape=(nx,nz), dtype=np.int32)
            if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='umagmax', leave=False, file=sys.stdout)
            for i in range(nxr):
                for k in range(nzr):
                    umagmax=umag[i,:,k].max()
                    for j in range(nyr):
                        if math.isclose(umag[i,j,k], umagmax, rel_tol=1.1*8e-4):
                            j_edge_4[i,k] = j
                            break
                        if (umag[i,j,k]>umagmax):
                            j_edge_4[i,k] = j-1
                            break
                    if verbose: progress_bar.update()
            if verbose: progress_bar.close()
        
        ## mass flux criteria
        if True:
            j_edge_5 = np.zeros(shape=(nx,nz), dtype=np.int32)
            if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='mfluxmax', leave=False, file=sys.stdout)
            for i in range(nxr):
                for k in range(nzr):
                    mfluxmax = mflux[i,:,k].max()
                    for j in range(nyr):
                        if math.isclose(mflux[i,j,k], mfluxmax, rel_tol=1.2*2e-3):
                            j_edge_5[i,k] = j
                            break
                        if (mflux[i,j,k]>mfluxmax):
                            j_edge_5[i,k] = j-1
                            break
                    if verbose: progress_bar.update()
            if verbose: progress_bar.close()
        
        # j_edge_1 = np.argmax(psvel_ddy<=1000., axis=1)            ## (index of) threshold on ddy(psvel)
        # j_edge_2 = np.argmax(psvel, axis=1)                       ## (index of) max of pseudovel
        # j_edge   = np.amin(np.stack((j_edge_1,j_edge_2)), axis=0) ## minimum of collective minima : shape [nx,nz]
        
        j_edge = j_edge_3 ## u criteria
        #j_edge = j_edge_4 ## umag criteria
        #j_edge = j_edge_5 ## mass flux criteria
        #j_edge   = np.amin(np.stack((j_edge_3,j_edge_4,j_edge_5)), axis=0) ## minimum of collective minima : shape [nx,nz]
        
        # === populate edge arrays (always grid snapped)
        
        for i in range(nxr):
            for k in range(nzr):
                je              =   j_edge[i,k]
                y_edge[i,k]     =         y[je]
                u_edge[i,k]     =     u[i,je,k]
                v_edge[i,k]     =     v[i,je,k]
                T_edge[i,k]     =     T[i,je,k]
                p_edge[i,k]     =     p[i,je,k]
                rho_edge[i,k]   =   rho[i,je,k]
                nu_edge[i,k]    =    nu[i,je,k]
                #psvel_edge[i,k] = psvel[i,je,k]
                M_edge[i,k]     =     M[i,je,k]
        
        data['j_edge']     = j_edge
        data['y_edge']     = y_edge
        data['u_edge']     = u_edge
        data['v_edge']     = v_edge
        data['T_edge']     = T_edge
        data['p_edge']     = p_edge
        data['rho_edge']   = rho_edge
        data['nu_edge']    = nu_edge
        #data['psvel_edge'] = psvel_edge
        data['M_edge']     = M_edge
        
        # ===
        
        if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='d99', leave=False, file=sys.stdout)
        for i in range(nxr):
            for k in range(nzr):
                
                ## populate d99 arrays with default 'grid-snapped' values
                for j in range(nyr):
                    
                    if False:
                        if (psvel[i,j,k] >= 0.99*psvel_edge[i,k]):
                            d99j[i,k] =   j-1
                            d99[i,k]  = y[j-1]
                            d99g[i,k] = y[j-1] ## at grid
                            break
                    
                    if True:
                        if (u[i,j,k] >= 0.99*u_edge[i,k]):
                            d99j[i,k] =   j-1
                            d99[i,k]  = y[j-1]
                            d99g[i,k] = y[j-1] ## at grid
                            break
                
                # === use spline interpolation to find d99
                
                je = j_edge[i,k]+5 ## add a couple points to accurately loft a higher order spline
                
                if False:
                    psvel_spl = sp.interpolate.CubicSpline(y[:je],psvel[i,:je,k]-(0.99*psvel_edge[i,k]),bc_type='natural')
                    roots = psvel_spl.roots() #discontinuity=False,extrapolate=False ## vulcan
                
                if True:
                    u_spl = sp.interpolate.CubicSpline(y[:je],u[i,:je,k]-(0.99*u_edge[i,k]),bc_type='natural')
                    roots = u_spl.roots(discontinuity=False,extrapolate=False)
                    #u_spl = sp.interpolate.pchip(y[:je],u[i,:je,k]-(0.99*u_edge[i,k]))
                    #roots = u_spl.roots(extrapolate=False,discontinuity=False)
                
                ## stupid thing for python<3.6 / scipy when 'extrapolate=False' not allowed
                ## --> Vulcan system python install
                if False:
                    roots2=[]
                    for root in roots:
                        if (root>0) and (root<y.max()):
                            roots2.append(root)
                    roots=np.array(roots2)
                
                # === populate d99 with root, recalculate & overwrite index/snap values
                
                if (roots.size>0):
                    d99_ = roots[0] ## lowest (and usually only) root
                    if (d99_<y_edge[i,k]): ## dont let it be greater than max location
                        d99[i,k]  =   d99_
                        d99j[i,k] =   np.abs(y-d99_).argmin()  ## closest index to interped value
                        d99g[i,k] = y[np.abs(y-d99_).argmin()] ## d99 at nearest grid point (overwrite)
                    else:
                        raise ValueError('root is > max! : xi=%i'%i)
                else:
                    d99_ = d99[i,k]
                    #raise ValueError('no root found at xi=%i'%i)
                    print('WARNING: no root was found --> taking grid snapped value')
                    print('-->check turbx.rgd.get_mean_dim()')
                
                # === get other quantities @ d99
                
                u99[i,k]     = sp.interpolate.interp1d(y[:je],     u[i,:je,k] )(d99_)
                rho99[i,k]   = sp.interpolate.interp1d(y[:je],   rho[i,:je,k] )(d99_)
                nu99[i,k]    = sp.interpolate.interp1d(y[:je],    nu[i,:je,k] )(d99_)
                T99[i,k]     = sp.interpolate.interp1d(y[:je],     T[i,:je,k] )(d99_)
                p99[i,k]     = sp.interpolate.interp1d(y[:je],     p[i,:je,k] )(d99_)
                v99[i,k]     = sp.interpolate.interp1d(y[:je],     v[i,:je,k] )(d99_)
                #psvel99[i,k] = sp.interpolate.interp1d(y[:je], psvel[i,:je,k] )(d99_)
                M99[i,k]     = sp.interpolate.interp1d(y[:je],     M[i,:je,k] )(d99_)
                
                if verbose: progress_bar.update()
        if verbose: progress_bar.close()
        
        data['u99']     = u99
        data['rho99']   = rho99
        data['nu99']    = nu99
        data['T99']     = T99
        data['p99']     = p99
        data['v99']     = v99
        #data['psvel99'] = psvel99
        data['M99']     = M99
        
        data['d99']     = d99
        data['d99j']    = d99j
        data['d99g']    = d99g
        
        # === θ, δ*, Re_θ, Re_τ
        
        Re_theta      = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        Re_theta_wall = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        Re_tau        = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        Re_d99        = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        Re_x          = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        H12           = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        H12_inc       = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        theta         = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        dstar         = np.zeros(shape=(nxr,nzr), dtype=np.float64)
        
        u_vd = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64) ## Van Driest scaled u
        
        if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='Re', leave=False, file=sys.stdout)
        for i in range(nxr):
            for k in range(nzr):
                je   = j_edge[i,k]
                yl   = np.copy(    y[:je+1]  )
                ul   = np.copy(  u[i,:je+1,k])
                rhol = np.copy(rho[i,:je+1,k])
                
                integrand_theta_inc = (ul/u_edge[i,k])*(1-(ul/u_edge[i,k]))
                integrand_dstar_inc = 1-(ul/u_edge[i,k])
                theta_inc           = sp.integrate.trapz(integrand_theta_inc, x=yl)
                dstar_inc           = sp.integrate.trapz(integrand_dstar_inc, x=yl)
                
                integrand_theta_cmp = (ul*rhol)/(u_edge[i,k]*rho_edge[i,k])*(1-(ul/u_edge[i,k]))
                integrand_dstar_cmp = (1-((ul*rhol)/(u_edge[i,k]*rho_edge[i,k])))
                theta_cmp           = sp.integrate.trapz(integrand_theta_cmp, x=yl)
                dstar_cmp           = sp.integrate.trapz(integrand_dstar_cmp, x=yl)
                
                integrand_u_vd   = np.sqrt(T_wall[i,k]/T[i,:,k])
                u_vd[i,:,k]      = sp.integrate.cumtrapz(integrand_u_vd, u[i,:,k], initial=0)
                
                # =====
                
                theta[i,k]         = theta_cmp
                dstar[i,k]         = dstar_cmp
                H12[i,k]           = dstar_cmp/theta_cmp
                H12_inc[i,k]       = dstar_inc/theta_inc
                Re_tau[i,k]        = d99[i,k]*u_tau[i,k]/nu_wall[i,k]
                Re_theta[i,k]      = theta_cmp*u_edge[i,k]/nu_edge[i,k]
                Re_theta_wall[i,k] = rho_edge[i,k]*theta_cmp*u_edge[i,k]/mu_wall[i,k]
                Re_d99[i,k]        = d99[i,k]*u_edge[i,k]/nu_edge[i,k]
                Re_x[i,k]          = u_edge[i,k]*(x[i]-x[0])/nu_edge[i,k]
                
                if verbose: progress_bar.update()
        if verbose: progress_bar.close()
        
        # ===
        
        data['Re_theta']      = Re_theta     
        data['Re_theta_wall'] = Re_theta_wall
        data['Re_tau']        = Re_tau       
        data['Re_d99']        = Re_d99       
        data['Re_x']          = Re_x         
        data['H12']           = H12          
        data['H12_inc']       = H12_inc      
        data['theta']         = theta        
        data['dstar']         = dstar    
        
        # ===
        
        u_plus_vd = u_vd / u_tau[:,np.newaxis,:] ## Van Driest scaled velocity (wall units)
        data['u_plus_vd'] = u_plus_vd
        
        # === gather everything
        
        data2 = {}
        for key,val in data.items():
            
            if isinstance(data[key], np.ndarray) and (data[key].shape==(nxr,nyr,nzr)):
                data2[key] = np.zeros((self.nx,self.ny,self.nz), dtype=data[key].dtype)
                arr = self.comm.gather([rx1,rx2,rz1,rz2, np.copy(val)], root=0)
                arr = self.comm.bcast(arr, root=0)
                for i in range(len(arr)):
                    data2[key][arr[i][0]:arr[i][1],:,arr[i][2]:arr[i][3]] = arr[i][-1]
            
            elif isinstance(data[key], np.ndarray) and (data[key].shape==(nxr,nzr)):
                data2[key] = np.zeros((self.nx,self.nz), dtype=data[key].dtype)
                arr = self.comm.gather([rx1,rx2,rz1,rz2, np.copy(val)], root=0)
                arr = self.comm.bcast(arr, root=0)
                for i in range(len(arr)):
                    data2[key][arr[i][0]:arr[i][1],arr[i][2]:arr[i][3]] = arr[i][-1]
            
            else:
                data2[key] = val
            
            self.comm.Barrier()
        data = data2
        
        # ===
        
        if False: ## to check
            for key,val in data.items():
                if (self.rank==0):
                    if isinstance(data[key], np.ndarray):
                        print('%s: %s'%(key, str(data[key].shape)))
                    else:
                        print('%s: %s'%(key, type(data[key])))
        
        # === report physical scales
        
        if verbose: print(72*'-')
        
        d99_avg      = np.mean(data['d99'],      axis=(0,1))
        u99_avg      = np.mean(data['u99'],      axis=(0,1))
        nu_wall_avg  = np.mean(data['nu_wall'],  axis=(0,1))
        u_tau_avg    = np.mean(data['u_tau'],    axis=(0,1))
        Re_tau_avg   = np.mean(data['Re_tau'],   axis=(0,1))
        Re_theta_avg = np.mean(data['Re_theta'], axis=(0,1))
        
        if verbose: even_print('Re_τ'   ,'%0.1f'%Re_tau_avg)
        if verbose: even_print('Re_θ'   ,'%0.1f'%Re_theta_avg)
        
        if verbose: even_print('δ99'   ,'%0.5e [m]'%d99_avg)
        if verbose: even_print('u_τ'   ,'%0.3f [m/s]'%u_tau_avg)
        if verbose: even_print('ν_wall','%0.5e [m²/s]'%nu_wall_avg)
        
        t_meas = self.duration_avg * (self.lchar / self.U_inf)
        if verbose: even_print('t_meas','%0.5e [s]'%t_meas)
        
        t_eddy = t_meas / (d99_avg/u_tau_avg)
        if verbose: even_print('t_eddy=t_meas/(δ99/u_τ)', '%0.2f'%t_eddy)
        if verbose: even_print('t_meas/(δ99/u99)'       , '%0.2f'%(t_meas/(d99_avg/u99_avg)))
        if verbose: even_print('t_meas/(20·δ99/u99)'    , '%0.2f'%(t_meas/(20*d99_avg/u99_avg)))
        
        if verbose: print(72*'-')
        
        # ===
        
        if (self.rank==0):
            with open(fn_dat_mean_dim,'wb') as f:
                pickle.dump(data,f,protocol=4)
            size = os.path.getsize(fn_dat_mean_dim)
        
        self.comm.Barrier()
        if verbose: even_print(fn_dat_mean_dim, '%0.2f [GB]'%(os.path.getsize(fn_dat_mean_dim)/1024**3))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.get_mean_dim() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    def calc_lambda2(self, **kwargs):
        '''
        calculate λ-2 & Q, save to RGD
        -----
        Jeong & Hussain (1996) : https://doi.org/10.1017/S0022112095000462
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        hiOrder      = kwargs.get('hiOrder',False) ## passthrough to get_grad()
        save_Q       = kwargs.get('save_Q',True)
        save_lambda2 = kwargs.get('save_lambda2',True)
        
        rt           = kwargs.get('rt',self.n_ranks)
        
        # ===
        
        if verbose: print('\n'+'rgd.calc_lambda2()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        if all([(save_Q is False),(save_lambda2 is False)]):
            raise AssertionError('neither λ-2 nor Q set to be solved')
        
        if (rt>self.nt):
            raise AssertionError('rt>self.nt')
        
        if (rt != self.n_ranks):
            raise AssertionError('rt != self.n_ranks')
        
        if verbose: even_print('save_Q','%s'%save_Q)
        if verbose: even_print('save_lambda2','%s'%save_lambda2)
        
        ## profiling not implemented (non-collective r/w)
        t_read   = 0.
        t_write  = 0.
        t_q_crit = 0.
        t_l2     = 0. 
        
        ## take advantage of collective r/w
        useReadBuffer  = False
        useWriteBuffer = False
        
        # === memory requirements
        
        mem_total_gb = psutil.virtual_memory().total/1024**3
        mem_avail_gb = psutil.virtual_memory().available/1024**3
        mem_free_gb  = psutil.virtual_memory().free/1024**3
        if verbose: even_print('mem total', '%0.1f [GB]'%mem_total_gb)
        if verbose: even_print('mem available', '%0.1f [GB]'%mem_avail_gb)
        if verbose: even_print('mem free', '%0.1f [GB]'%mem_free_gb)
        
        fsize = os.path.getsize(self.fname)/1024**3
        if verbose: even_print(os.path.basename(self.fname),'%0.1f [GB]'%fsize)
        
        # ===
        
        if verbose: even_print('rt','%i'%rt)
        
        #comm4d = self.comm.Create_cart(dims=[rx,ry,rz,rt], periods=[False,False,False,False], reorder=False)
        #t4d = comm4d.Get_coords(self.rank)
        
        #rxl_ = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
        #ryl_ = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
        #rzl_ = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
        rtl_ = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))
        
        #rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        #ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        #rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        #rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        #ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        #rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        rt1,rt2 = rtl[self.rank]; ntr = rt2 - rt1
        
        data_gb = 4*self.nx*self.ny*self.nz*self.nt / 1024**3
        
        # === initialize 4D arrays
        if save_lambda2:
            if verbose: even_print('initializing data/lambda2','%0.2f [GB]'%(data_gb,))
            if ('data/lambda2' in self):
                del self['data/lambda2']
            self.create_dataset('data/lambda2', 
                                 shape=(self.nt,self.nz,self.ny,self.nx), 
                                 dtype=self['data/u'].dtype,
                                 #chunks=(1,min(self.nz//2,64),min(self.ny//2,64),min(self.nx//2,64)),
                                 #chunks=(1,True,True,True),
                                 chunks=True,
                                 )
        
        if save_Q:
            if verbose: even_print('initializing data/Q','%0.2f [GB]'%(data_gb,))
            if ('data/Q' in self):
                del self['data/Q']
            self.create_dataset('data/Q', 
                                 shape=(self.nt,self.nz,self.ny,self.nx), 
                                 dtype=self['data/u'].dtype,
                                 #chunks=(1,min(self.nz//2,64),min(self.ny//2,64),min(self.nx//2,64)),
                                 #chunks=(1,True,True,True),
                                 chunks=True,
                                 )
        
        # === check if strains exist
        
        if all([('data/dudx' in self),('data/dvdx' in self),('data/dwdx' in self),\
                ('data/dudy' in self),('data/dvdy' in self),('data/dwdy' in self),\
                ('data/dudz' in self),('data/dvdz' in self),('data/dwdz' in self)]):
            strainsAvailable = True
        else:
            strainsAvailable = False
        if verbose: even_print('strains available','%s'%str(strainsAvailable))
        
        # === collective reads
        
        if useReadBuffer:
            dset = self['data/u']
            self.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                u = dset[rt1:rt2,:,:,:].T
            self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
            if verbose:
                print('read u : %0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
            
            dset = self['data/v']
            self.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                v = dset[rt1:rt2,:,:,:].T
            self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
            if verbose:
                print('read v : %0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
            
            dset = self['data/w']
            self.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                w = dset[rt1:rt2,:,:,:].T
            self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
            if verbose:
                print('read w : %0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        ## write buffers
        if useWriteBuffer:
            if save_lambda2:
                l2_buff = np.zeros_like(u)
            if save_Q:
                q_buff = np.zeros_like(u)
        
        mem_total_gb = psutil.virtual_memory().total/1024**3
        mem_avail_gb = psutil.virtual_memory().available/1024**3
        mem_free_gb  = psutil.virtual_memory().free/1024**3
        if verbose: even_print('mem total', '%0.1f [GB]'%mem_total_gb)
        #if verbose: even_print('mem available', '%0.1f [GB]'%mem_avail_gb)
        if verbose: even_print('mem free', '%0.1f [GB]'%mem_free_gb)
        
        if verbose:
            progress_bar = tqdm(total=ntr, ncols=100, desc='calc λ2', leave=False, file=sys.stdout)
        
        tii = -1
        for ti in range(rt1,rt2):
            tii += 1
            
            if useReadBuffer:
                u_ = np.squeeze(u[:,:,:,tii]) ## read from buffer
                v_ = np.squeeze(v[:,:,:,tii])
                w_ = np.squeeze(w[:,:,:,tii])
            else:
                t_start = timeit.default_timer()
                u_ = np.squeeze( self['data/u'][ti,:,:,:].T ) ## independent reads
                v_ = np.squeeze( self['data/v'][ti,:,:,:].T )
                w_ = np.squeeze( self['data/w'][ti,:,:,:].T )
                t_delta = timeit.default_timer() - t_start
                data_gb = (u_.nbytes + v_.nbytes + w_.nbytes) / 1024**3
                #if verbose:
                #    tqdm.write( 'read u,v,w : %0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)) )
            
            # === get the velocity gradient tensor
            t_start = timeit.default_timer()
            strain = get_grad(u_, v_, w_, self.x, self.y, self.z, do_stack=True, hiOrder=hiOrder, verbose=verbose)
            t_delta = timeit.default_timer() - t_start
            #if verbose:
            #    tqdm.write( even_print('get_grad()','%0.3f [s]'%(t_delta,), s=True) )
            
            # === get the shear rate & vorticity tensors
            S = 0.5*(strain + np.transpose(strain, axes=(0,1,2,4,3))) ## strain rate tensor (symmetric)
            O = 0.5*(strain - np.transpose(strain, axes=(0,1,2,4,3))) ## rotation rate tensor (anti-symmetric)
            # np.testing.assert_allclose(S+O, strain, atol=1.e-6)
            
            # === Q : second invariant of characteristics equation: λ³ + Pλ² + Qλ + R = 0
            if save_Q:
                
                # === second invariant : Q
                t_start = timeit.default_timer()
                O_norm  = np.linalg.norm(O, ord='fro', axis=(3,4))
                S_norm  = np.linalg.norm(S, ord='fro', axis=(3,4))
                Q       = 0.5*(O_norm**2 - S_norm**2)
                t_delta = timeit.default_timer() - t_start
                #if verbose: tqdm.write(even_print('calc: Q','%s'%format_time_string(t_delta), s=True))
                if useWriteBuffer:
                    q_buff[:,:,:,tii]
                else:
                    self['data/Q'][ti,:,:,:] = Q.T ## non-collective write (memory minimizing)
                
                # === second invariant : Q --> an equivalent formulation using eigenvalues (much slower)
                if False:
                    t_start = timeit.default_timer()
                    eigvals = np.linalg.eigvals(strain)
                    P       = -1*np.sum(eigvals, axis=-1) ## first invariant : P
                    SijSji  = np.einsum('xyzij,xyzji->xyz', S, S)
                    OijOji  = np.einsum('xyzij,xyzji->xyz', O, O)
                    Q       = 0.5*(P**2 - SijSji - OijOji)
                    t_delta = timeit.default_timer() - t_start
                    #if verbose: tqdm.write(even_print('calc: Q','%s'%format_time_string(t_delta), s=True))
                    #np.testing.assert_allclose(Q.imag, np.zeros_like(Q.imag, dtype=np.float32), atol=1e-6)
                    if useWriteBuffer:
                        q_buff[:,:,:,tii]
                    else:
                        self['data/Q'][ti,:,:,:] = Q.T ## non-collective write (memory minimizing)
                    pass
            
            # === λ-2
            if save_lambda2:
                
                t_start = timeit.default_timer()
                
                # === S² and Ω²
                SikSkj = np.einsum('xyzik,xyzkj->xyzij', S, S)
                OikOkj = np.einsum('xyzik,xyzkj->xyzij', O, O)
                #np.testing.assert_allclose(np.matmul(S,S), SikSkj, atol=1e-6)
                #np.testing.assert_allclose(np.matmul(O,O), OikOkj, atol=1e-6)
                
                # === Eigenvalues of (S²+Ω²) --> a real symmetric (Hermitian) matrix
                eigvals            = np.linalg.eigvalsh(SikSkj+OikOkj, UPLO='L')
                #eigvals_sort_order = np.argsort(np.abs(eigvals), axis=3) ## sort order of λ --> magnitude (wrong)
                eigvals_sort_order = np.argsort(eigvals, axis=3) ## sort order of λ
                eigvals_sorted     = np.take_along_axis(eigvals, eigvals_sort_order, axis=3) ## do sort
                lambda2            = np.squeeze(eigvals_sorted[:,:,:,1]) ## λ-2 is the second eigenvalue
                t_delta            = timeit.default_timer() - t_start
                #if verbose: tqdm.write(even_print('calc: λ2','%s'%format_time_string(t_delta), s=True))
                
                if useWriteBuffer:
                    l2_buff[:,:,:,tii]
                else:
                    self['data/lambda2'][ti,:,:,:] = lambda2.T ## non-collective write (memory minimizing)
                pass
            
            if verbose: progress_bar.update()
        if verbose: progress_bar.close()
        
        # === collective writes --> better write performance, but memory limiting
        
        if useWriteBuffer:
            self.comm.Barrier()
            
            if save_lambda2:
                dset = self['data/lambda2']
                self.comm.Barrier()
                t_start = timeit.default_timer()
                with dset.collective:
                    dset[rt1:rt2,:,:,:] = l2_buff.T
                self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
                if (self.rank==0):
                    print('write λ2 : %0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
            
            if save_Q:
                dset = self['data/Q']
                self.comm.Barrier()
                t_start = timeit.default_timer()
                with dset.collective:
                    dset[rt1:rt2,:,:,:] = q_buff.T
                self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
                if (self.rank==0):
                    print('write Q : %0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        # ===
        
        self.get_header(verbose=False)
        if verbose: even_print(self.fname, '%0.2f [GB]'%(os.path.getsize(self.fname)/1024**3))
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.calc_lambda2() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
    
    def calc_fft(self, **kwargs):
        '''
        calculate FFT in [t] at every [x,y,z], avg in [x,z]
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.calc_fft()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        rt = kwargs.get('rt',1)
        
        overlap_fac_nom = kwargs.get('overlap_fac_nom',0.50)
        n_win           = kwargs.get('n_win',4)
        
        fn_h5_fft       = kwargs.get('fn_h5_fft',None)
        fn_dat_fft      = kwargs.get('fn_dat_fft',None)
        fn_dat_mean_dim = kwargs.get('fn_dat_mean_dim',None)
        
        ## for now only distribute data in [y] --> allows [x,z] mean before Send/Recv
        if (rx!=1):
            raise AssertionError('rx!=1')
        if (rz!=1):
            raise AssertionError('rz!=1')
        if (rt!=1):
            raise AssertionError('rt!=1')
        
        if (rx*ry*rz*rt != self.n_ranks):
            raise AssertionError('rx*ry*rz*rt != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        if (rt>self.nt):
            raise AssertionError('rt>self.nt')
        
        # ===
        
        #comm4d = self.comm.Create_cart(dims=[rx,ry,ry,rt], periods=[False,False,False,False], reorder=False)
        #t4d = comm4d.Get_coords(self.rank)
        
        #rxl_ = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
        ryl_ = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
        #rzl_ = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
        #rtl_ = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))

        #rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        #rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        #rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        #ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        #rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        ry1,ry2 = ryl[self.rank]; nyr = ry2 - ry1
        
        # === mean file name (for reading) : .dat
        if (fn_dat_mean_dim is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_dat_mean_base = fname_root+'_mean_dim.dat'
            fn_dat_mean_dim = str(PurePosixPath(fname_path, fname_dat_mean_base))
        
        # === fft file name (for writing)
        if (fn_h5_fft is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_fft_h5_base = fname_root+'_fft.h5'
            fn_h5_fft = str(PurePosixPath(fname_path, fname_fft_h5_base))
        
        # === fft file name (for writing) : dat
        if (fn_dat_fft is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_fft_dat_base = fname_root+'_fft.dat'
            fn_dat_fft = str(PurePosixPath(fname_path, fname_fft_dat_base))
        
        if verbose: even_print('fn_rgd_prime'    , self.fname       )
        if verbose: even_print('fn_dat_mean_dim' , fn_dat_mean_dim  )
        if verbose: even_print('fn_h5_fft'       , fn_h5_fft        )
        if verbose: even_print('fn_dat_fft'      , fn_dat_fft       )
        if verbose: print(72*'-')
        
        if not os.path.isfile(fn_dat_mean_dim):
            raise FileNotFoundError('%s not found!'%fn_dat_mean_dim)
        
        # === read in data (mean dim) --> every rank gets full [x,z]
        with open(fn_dat_mean_dim,'rb') as f:
            data_mean_dim = pickle.load(f)
        fmd = type('foo', (object,), data_mean_dim)
        
        self.comm.Barrier()
        
        # === 2D dimensional quantities --> [x,z]
        u_tau    = fmd.u_tau
        nu_wall  = fmd.nu_wall
        d99      = fmd.d99
        u99      = fmd.u99
        Re_tau   = fmd.Re_tau
        Re_theta = fmd.Re_theta
        
        u_tau_avg    = np.mean(fmd.u_tau    , axis=(0,1))
        nu_wall_avg  = np.mean(fmd.nu_wall  , axis=(0,1))
        d99_avg      = np.mean(fmd.d99      , axis=(0,1))
        u99_avg      = np.mean(fmd.u99      , axis=(0,1))
        Re_tau_avg   = np.mean(fmd.Re_tau   , axis=(0,1))
        Re_theta_avg = np.mean(fmd.Re_theta , axis=(0,1))
        
        # === 2D inner scales --> [x,z]
        sc_l_in = nu_wall / u_tau
        sc_u_in = u_tau
        sc_t_in = nu_wall / u_tau**2
        
        # === 2D outer scales --> [x,z]
        sc_l_out = d99
        sc_u_out = u99
        sc_t_out = d99/u99
        
        # === check
        np.testing.assert_allclose(fmd.lchar   , self.lchar   , rtol=1e-8)
        np.testing.assert_allclose(fmd.U_inf   , self.U_inf   , rtol=1e-8)
        np.testing.assert_allclose(fmd.rho_inf , self.rho_inf , rtol=1e-8)
        np.testing.assert_allclose(fmd.T_inf   , self.T_inf   , rtol=1e-8)
        np.testing.assert_allclose(fmd.nx      , self.nx      , rtol=1e-8)
        np.testing.assert_allclose(fmd.ny      , self.ny      , rtol=1e-8)
        np.testing.assert_allclose(fmd.nz      , self.nz      , rtol=1e-8)
        np.testing.assert_allclose(fmd.xs      , self.x       , rtol=1e-8)
        np.testing.assert_allclose(fmd.ys      , self.y       , rtol=1e-8)
        np.testing.assert_allclose(fmd.zs      , self.z       , rtol=1e-8)
        
        lchar   = self.lchar
        U_inf   = self.U_inf
        rho_inf = self.rho_inf
        T_inf   = self.T_inf
        
        nx = self.nx
        ny = self.ny
        nz = self.nz
        nt = self.nt
        
        ## dimless (inlet)
        xd = self.x
        yd = self.y
        zd = self.z
        td = self.t
        
        ## dimensional [m] / [s]
        x      = self.x * lchar 
        y      = self.y * lchar
        z      = self.z * lchar
        t      = self.t * (lchar/U_inf)
        t_meas = t[-1]-t[0]
        dt     = self.dt * (lchar/U_inf)
        
        np.testing.assert_equal(nx,x.size)
        np.testing.assert_equal(ny,y.size)
        np.testing.assert_equal(nz,z.size)
        np.testing.assert_equal(nt,t.size)
        np.testing.assert_allclose(dt, t[1]-t[0], rtol=1e-8)
        
        # === report
        if verbose:
            even_print('nx'     , '%i'        %nx     )
            even_print('ny'     , '%i'        %ny     )
            even_print('nz'     , '%i'        %nz     )
            even_print('nt'     , '%i'        %nt     )
            even_print('dt'     , '%0.5e [s]' %dt     )
            even_print('t_meas' , '%0.5e [s]' %t_meas )
            print(72*'-')
        
        if verbose:
            even_print('Re_τ'   , '%0.1f'        % Re_tau_avg   )
            even_print('Re_θ'   , '%0.1f'        % Re_theta_avg )
            even_print('δ99'    , '%0.5e [m]'    % d99_avg      )
            even_print('U_inf'  , '%0.3f [m/s]'  % U_inf        )
            even_print('u_τ'    , '%0.3f [m/s]'  % u_tau_avg    )
            even_print('ν_wall' , '%0.5e [m²/s]' % nu_wall_avg  )
            print(72*'-')
        
        t_eddy = t_meas / (d99_avg/u_tau_avg)
        
        if verbose:
            even_print('t_meas/(δ99/u_τ) = t_eddy' , '%0.2f'%t_eddy)
            even_print('t_meas/(δ99/u99)'          , '%0.2f'%(t_meas/(d99_avg/u99_avg)))
            even_print('t_meas/(20·δ99/u99)'       , '%0.2f'%(t_meas/(20*d99_avg/u99_avg)))
            print(72*'-')
        
        # === establish windowing
        win_len, overlap = get_overlapping_window_size(nt, n_win, overlap_fac_nom)
        overlap_fac = overlap / win_len
        tw, n_win, n_pad = get_overlapping_windows(t, win_len, overlap)
        
        t_meas_per_win = (win_len-1)*dt
        t_eddy_per_win = t_meas_per_win / (d99_avg/u_tau_avg)
        
        if verbose:
            even_print('overlap_fac (nominal)' , '%0.5f'%overlap_fac_nom    )
            even_print('n_win'                 , '%i'%n_win                 )
            even_print('win_len'               , '%i'%win_len               )
            even_print('overlap'               , '%i'%overlap               )
            even_print('overlap_fac'           , '%0.5f'%overlap_fac        )
            even_print('n_pad'                 , '%i'%n_pad                 )
            even_print('t_win/(δ99/u_τ)'       , '%0.2f [-]'%t_eddy_per_win )
            print(72*'-')
        
        # === get frequency vector --> here for short time FFT!
        freq_full = sp.fft.fftfreq(n=win_len, d=dt)
        fp        = np.where(freq_full>0)
        freq      = np.copy(freq_full[fp])
        df        = freq[1]-freq[0]
        nf        = freq.size
        
        if verbose:
            even_print('freq min','%0.1f [Hz]'%freq.min())
            even_print('freq max','%0.1f [Hz]'%freq.max())
            even_print('df','%0.1f [Hz]'%df)
            even_print('nf','%i'%nf)
            
            # freq_plus = freq * np.mean(sc_t_in, axis=(0,1)) ## [?]
            # df_plus   = df   * np.mean(sc_t_in, axis=(0,1)) ## [?]
            # even_print('freq+ min','%0.5e [-]'%freq_plus.min())
            # even_print('freq+ max','%0.5f [-]'%freq_plus.max())
            # even_print('df+','%0.5e [-]'%df_plus)
            
            period_eddy = (1/freq) / np.mean((d99/u_tau), axis=(0,1))
            period_plus = (1/freq) / np.mean(sc_t_in,     axis=(0,1))
            even_print('period+ min'     , '%0.5e [-]'%period_plus.min())
            even_print('period+ max'     , '%0.5f [-]'%period_plus.max())
            even_print('period eddy min' , '%0.5e [-]'%period_eddy.min())
            even_print('period eddy max' , '%0.5f [-]'%period_eddy.max())
            
            print(72*'-')
        
        # === kx, λx
        
        na = np.newaxis
        
        # === prevent zeros since later we divide by wall u to get kx
        #fmd.u = np.maximum(fmd.u, np.ones_like(fmd.u)*1e-8)
        fmd.u[:,0,:] = fmd.u[:,1,:]*1e-4
        
        ## kx
        kx = (2*np.pi*freq[na,na,na,:]/fmd.u[:,:,:,na])
        kx = np.mean(kx, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        ## kx·δ99 --> dimless outer
        kxd = (2*np.pi*freq[na,na,na,:]/fmd.u[:,:,:,na]) * sc_l_out[:,na,:,na]
        kxd = np.mean(kxd, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        ## kx+ --> dimless inner
        kxp = (2*np.pi*freq[na,na,na,:]/fmd.u[:,:,:,na]) * sc_l_in[:,na,:,na]
        kxp = np.mean(kxp, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        ## λx
        lx = (fmd.u[:,:,:,na]/freq[na,na,na,:])
        lx = np.mean(lx, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        ## λx/δ99 --> dimless outer
        lxd = (fmd.u[:,:,:,na]/freq[na,na,na,:]) / sc_l_out[:,na,:,na]
        lxd = np.mean(lxd, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        ## λx+ = λx/(ν/u_tau) --> dimless inner
        lxp = (fmd.u[:,:,:,na]/freq[na,na,na,:]) / sc_l_in[:,na,:,na]
        lxp = np.mean(lxp, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        # === read in data (prime) --> still dimless!
        
        # if verbose: print('starting read')
        
        scalars = ['uI','vI','wI']
        scalars_dtypes = [self.scalars_dtypes_dict[s] for s in scalars]
        
        ## 5D [scalar][x,y,z,t] structured array
        data = np.zeros(shape=(self.nx, nyr, self.nz, self.nt), dtype={'names':scalars, 'formats':scalars_dtypes})
        
        for scalar in scalars:
            dset = self['data/%s'%scalar]
            self.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                data[scalar] = dset[:,:,ry1:ry2,:].T
            self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
            if verbose:
                even_print('read: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        # === redimensionalize prime data
        
        data['uI'] *= U_inf
        data['vI'] *= U_inf
        data['wI'] *= U_inf
        
        # === initialize arrays
        
        Euu      = np.zeros((self.nx, nyr, self.nz, nf), dtype=np.float32)
        uIuI_avg = np.zeros((self.nx, nyr, self.nz),     dtype=np.float32)
        
        energy_norm_fac_arr = np.zeros((self.nx, nyr, self.nz), dtype=np.float32) ## just for monitoring
        
        window_type = 'tukey'
        if (window_type=='tukey'):
            window = sp.signal.windows.tukey(win_len, alpha=overlap_fac_nom)
        elif (window_type is None):
            window = np.ones(win_len, dtype=data['uI'].dtype)
        
        if True:
            if verbose: progress_bar = tqdm(total=self.nx*nyr*self.nz, ncols=100, desc='fft', leave=False)
            for xi in range(self.nx):
                for yi in range(nyr):
                    for zi in range(self.nz):
                        
                        uI_ = np.copy(data['uI'][xi,yi,zi,:])
                        #vI_ = np.copy(data['vI'][xi,yi,zi,:])
                        #wI_ = np.copy(data['wI'][xi,yi,zi,:])
                        
                        uIuI_avg_ijk = np.mean(uI_*uI_, dtype=np.float64).astype(np.float32)
                        uIuI_avg[xi,yi,zi] = uIuI_avg_ijk
                        
                        ## window uI_ into several overlapping windows
                        uI_, nw, n_pad = get_overlapping_windows(uI_, win_len, overlap)
                        
                        ## do fft for each segment
                        Euu_ijk = np.zeros((nw,nf), dtype=np.float32)
                        for wi in range(nw):
                            ui    = np.copy(uI_[wi,:])
                            uj    = np.copy(uI_[wi,:])
                            n     = ui.size
                            #A_ui = sp.fft.fft(ui)[fp] / n
                            #A_uj = sp.fft.fft(uj)[fp] / n
                            ui   *= window ## window
                            uj   *= window
                            #ui  -= np.mean(ui) ## de-trend
                            #uj  -= np.mean(uj)
                            A_ui          = sp.fft.fft(ui)[fp] / np.sum(np.sqrt(window))
                            A_uj          = sp.fft.fft(uj)[fp] / np.sum(np.sqrt(window))
                            Euu_ijk[wi,:] = 2 * np.real(A_ui*np.conj(A_uj)) / df
                        
                        ## mean across segments
                        Euu_ijk = np.mean(Euu_ijk, axis=0, dtype=np.float64).astype(np.float32)
                        
                        ## normalize by covariance
                        if True:
                            if (uIuI_avg_ijk!=0.):
                                energy_norm_fac = np.sum(df*Euu_ijk) / uIuI_avg_ijk
                            else:
                                energy_norm_fac = 1.
                            ##
                            Euu_ijk /= energy_norm_fac
                            energy_norm_fac_arr[xi,yi,zi] = energy_norm_fac ## just for monitoring
                        
                        ## write
                        Euu[xi,yi,zi,:] = Euu_ijk
                        
                        if verbose: progress_bar.update()
            if verbose:
                progress_bar.close()
        
        ## just for monitoring --> delete eventually
        energy_norm_fac_min = energy_norm_fac_arr.min()
        energy_norm_fac_max = energy_norm_fac_arr.max()
        energy_norm_fac_arr = np.mean(energy_norm_fac_arr, axis=(0,1,2), dtype=np.float64).astype(np.float32)
        if verbose:
            even_print('energy norm fac min', '%0.6f'%energy_norm_fac_min)
            even_print('energy norm fac max', '%0.6f'%energy_norm_fac_max)
            even_print('energy norm fac avg', '%0.6f'%energy_norm_fac_arr)
        
        # === non-dimensionalize
        
        Euu      = Euu / ( sc_t_in[:,na,:,na] * sc_u_in[:,na,:,na]**2 )
        kxEuu    = np.copy( Euu * (2*np.pi*freq[na,na,na,:]/fmd.u[:,ry1:ry2,:,na]) * sc_l_in[:,na,:,na] )
        uIuI_avg = uIuI_avg / sc_u_in[:,na,:]**2
        
        # === average in [x,z] --> leave [y,f]
        
        Euu      = np.mean(Euu      , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y,f]
        kxEuu    = np.mean(kxEuu    , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y,f]
        uIuI_avg = np.mean(uIuI_avg , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y]
        
        # === gather all results
        
        self.comm.Barrier()
        Euu_all      = None
        kxEuu_all    = None
        uIuI_avg_all = None
        
        # === 
        
        if (self.rank==0):
            
            #i=j=k=0
            j=0
            
            Euu_all      = np.zeros( (ny,nf) , dtype=Euu.dtype      )
            kxEuu_all    = np.zeros( (ny,nf) , dtype=kxEuu.dtype    )
            uIuI_avg_all = np.zeros( (ny,)   , dtype=uIuI_avg.dtype )
            
            # ## data this rank
            # Euu_all[rxl[i][0]:rxl[i][1],ryl[j][0]:ryl[j][1],rzl[k][0]:rzl[k][1],:]    = Euu
            # kxEuu_all[rxl[i][0]:rxl[i][1],ryl[j][0]:ryl[j][1],rzl[k][0]:rzl[k][1],:]  = kxEuu
            # uIuI_avg_all[rxl[i][0]:rxl[i][1],ryl[j][0]:ryl[j][1],rzl[k][0]:rzl[k][1]] = uIuI_avg
            
            ## data this rank
            Euu_all[ryl[j][0]:ryl[j][1],:]    = Euu
            kxEuu_all[ryl[j][0]:ryl[j][1],:]  = kxEuu
            uIuI_avg_all[ryl[j][0]:ryl[j][1]] = uIuI_avg
        
        for ri in range(1,self.n_ranks):
            
            # t4d = comm4d.Get_coords(ri)
            # i,j,k = t4d[0], t4d[1], t4d[2]
            
            j = ri
            
            self.comm.Barrier()
            if (self.rank==ri):
                self.comm.Send(Euu,      dest=0, tag=1*ri)
                self.comm.Send(kxEuu,    dest=0, tag=2*ri)
                self.comm.Send(uIuI_avg, dest=0, tag=3*ri)
                ## print('rank %i : Euu.shape=%s'%(rank,str(Euu.shape)))
            elif (self.rank==0):
                #nxri = rxl[i][1] - rxl[i][0]
                nyri = ryl[j][1] - ryl[j][0]
                #nzri = rzl[k][1] - rzl[k][0]
                
                # recvbuf1 = np.zeros((nxri,nyri,nzri,nf), dtype=Euu.dtype)
                # recvbuf2 = np.zeros((nxri,nyri,nzri,nf), dtype=kxEuu.dtype)
                # recvbuf3 = np.zeros((nxri,nyri,nzri),    dtype=uIuI_avg.dtype)
                
                recvbuf1 = np.zeros( (nyri,nf) , dtype=Euu.dtype      )
                recvbuf2 = np.zeros( (nyri,nf) , dtype=kxEuu.dtype    )
                recvbuf3 = np.zeros( (nyri,)   , dtype=uIuI_avg.dtype )
                
                #print('rank %i : recvbuf1.shape=%s'%(rank,str(recvbuf.shape)))
                self.comm.Recv(recvbuf1, source=ri, tag=1*ri)
                self.comm.Recv(recvbuf2, source=ri, tag=2*ri)
                self.comm.Recv(recvbuf3, source=ri, tag=3*ri)
                
                # Euu_all[rxl[i][0]:rxl[i][1],ryl[j][0]:ryl[j][1],rzl[k][0]:rzl[k][1],:]    = recvbuf1
                # kxEuu_all[rxl[i][0]:rxl[i][1],ryl[j][0]:ryl[j][1],rzl[k][0]:rzl[k][1],:]  = recvbuf2
                # uIuI_avg_all[rxl[i][0]:rxl[i][1],ryl[j][0]:ryl[j][1],rzl[k][0]:rzl[k][1]] = recvbuf3
                
                Euu_all[ryl[j][0]:ryl[j][1],:]    = recvbuf1
                kxEuu_all[ryl[j][0]:ryl[j][1],:]  = recvbuf2
                uIuI_avg_all[ryl[j][0]:ryl[j][1]] = recvbuf3
                
            else:
                pass
        
        # === average --> now do before Send/Recv
        
        ## if (self.rank==0):
        ##     Euu      = np.mean(Euu_all      , axis=(0,2)) ## avg in [x,z] --> leave [y,f]
        ##     kxEuu    = np.mean(kxEuu_all    , axis=(0,2)) ## avg in [x,z] --> leave [y,f]
        ##     uIuI_avg = np.mean(uIuI_avg_all , axis=(0,2)) ## avg in [x,z] --> leave [y]
        
        # === overwrite
        
        Euu      = np.copy( Euu_all      )
        kxEuu    = np.copy( kxEuu_all    )
        uIuI_avg = np.copy( uIuI_avg_all )
        
        # === save results
        if (self.rank==0):
            
            sc_l_in  = np.mean(sc_l_in  , axis=(0,1))
            sc_u_in  = np.mean(sc_u_in  , axis=(0,1))
            sc_t_in  = np.mean(sc_t_in  , axis=(0,1))
            sc_l_out = np.mean(sc_l_out , axis=(0,1))
            sc_u_out = np.mean(sc_u_out , axis=(0,1))
            sc_t_out = np.mean(sc_t_out , axis=(0,1))
            
            data = {}
            data['Euu']      = Euu
            data['kxEuu']    = kxEuu
            data['uIuI_avg'] = uIuI_avg
            #data['Euu_all']  = Euu_all
            data['x']        = x
            data['y']        = y
            data['z']        = z
            data['t']        = t
            data['freq']     = freq
            data['kx']       = kx
            data['kxp']      = kxp
            data['kxd']      = kxd
            data['lx']       = lx
            data['lxp']      = lxp
            data['lxd']      = lxd
            data['sc_l_in']  = sc_l_in
            data['sc_l_out'] = sc_l_out
            data['sc_t_in']  = sc_t_in
            data['sc_t_out'] = sc_t_out
            data['sc_u_in']  = sc_u_in
            data['sc_u_out'] = sc_u_out
            data['lchar']    = lchar
            data['Re_tau']   = Re_tau
            data['Re_theta'] = Re_theta
            
            with open(fn_dat_fft,'wb') as f:
                pickle.dump(data, f, protocol=4)
            print('--w-> %s : %0.2f [MB]'%(fn_dat_fft,os.path.getsize(fn_dat_fft)/1024**2))
        
        # ===
        
        self.comm.Barrier()
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.calc_fft() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    def calc_corr_span(self, **kwargs):
        '''
        calculate autocorrelation in [z] and avg in [x,t]
        '''
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.calc_corr_span()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        rt = kwargs.get('rt',1)
        
        fn_dat_corr_span = kwargs.get('fn_dat_corr_span',None)
        fn_dat_mean_dim  = kwargs.get('fn_dat_mean_dim',None)
        
        ## for now only distribute data in [y]
        if (rx!=1):
            raise AssertionError('rx!=1')
        if (rz!=1):
            raise AssertionError('rz!=1')
        if (rt!=1):
            raise AssertionError('rt!=1')
        
        if (rx*ry*rz*rt != self.n_ranks):
            raise AssertionError('rx*ry*rz*rt != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        if (rt>self.nt):
            raise AssertionError('rt>self.nt')
        
        # ===
        
        #comm4d = self.comm.Create_cart(dims=[rx,ry,ry,rt], periods=[False,False,False,False], reorder=False)
        #t4d = comm4d.Get_coords(self.rank)
        
        #rxl_ = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
        ryl_ = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
        #rzl_ = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
        #rtl_ = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))

        #rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        #rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        #rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        #ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        #rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        ry1,ry2 = ryl[self.rank]; nyr = ry2 - ry1
        
        # === mean dimensional file name (for reading) : .dat
        if (fn_dat_mean_dim is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_dat_mean_base = fname_root+'_mean_dim.dat'
            fn_dat_mean_dim = str(PurePosixPath(fname_path, fname_dat_mean_base))
        
        # === (auto)corr file name (for writing) : dat
        if (fn_dat_corr_span is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_corr_span_dat_base = fname_root+'_corr_span.dat'
            fn_dat_corr_span = str(PurePosixPath(fname_path, fname_corr_span_dat_base))
        
        if verbose: even_print('fn_rgd_prime'     , self.fname       )
        if verbose: even_print('fn_dat_mean_dim'  , fn_dat_mean_dim  )
        if verbose: even_print('fn_dat_corr_span' , fn_dat_corr_span )
        if verbose: print(72*'-')
        
        if not os.path.isfile(fn_dat_mean_dim):
            raise FileNotFoundError('%s not found!'%fn_dat_mean_dim)
        
        if True:
            # === read in data (mean dim) --> every rank gets full [x,z]
            with open(fn_dat_mean_dim,'rb') as f:
                data_mean_dim = pickle.load(f)
            fmd = type('foo', (object,), data_mean_dim)
            
            # === 2D dimensional quantities --> [x,z]
            u_tau    = fmd.u_tau
            nu_wall  = fmd.nu_wall
            d99      = fmd.d99
            u99      = fmd.u99
            Re_tau   = fmd.Re_tau
            Re_theta = fmd.Re_theta
            
            u_tau_avg    = np.mean(fmd.u_tau    , axis=(0,1))
            nu_wall_avg  = np.mean(fmd.nu_wall  , axis=(0,1))
            d99_avg      = np.mean(fmd.d99      , axis=(0,1))
            u99_avg      = np.mean(fmd.u99      , axis=(0,1))
            Re_tau_avg   = np.mean(fmd.Re_tau   , axis=(0,1))
            Re_theta_avg = np.mean(fmd.Re_theta , axis=(0,1))
            
            # === 2D inner scales --> [x,z]
            sc_l_in = nu_wall / u_tau
            sc_u_in = u_tau
            sc_t_in = nu_wall / u_tau**2
            
            # === 2D outer scales --> [x,z]
            sc_l_out = d99
            sc_u_out = u99
            sc_t_out = d99/u99
            
            # === check
            np.testing.assert_allclose(fmd.lchar   , self.lchar   , rtol=1e-8)
            np.testing.assert_allclose(fmd.U_inf   , self.U_inf   , rtol=1e-8)
            np.testing.assert_allclose(fmd.rho_inf , self.rho_inf , rtol=1e-8)
            np.testing.assert_allclose(fmd.T_inf   , self.T_inf   , rtol=1e-8)
            np.testing.assert_allclose(fmd.nx      , self.nx      , rtol=1e-8)
            np.testing.assert_allclose(fmd.ny      , self.ny      , rtol=1e-8)
            np.testing.assert_allclose(fmd.nz      , self.nz      , rtol=1e-8)
            np.testing.assert_allclose(fmd.xs      , self.x       , rtol=1e-8)
            np.testing.assert_allclose(fmd.ys      , self.y       , rtol=1e-8)
            np.testing.assert_allclose(fmd.zs      , self.z       , rtol=1e-8)
            
            lchar   = self.lchar
            U_inf   = self.U_inf
            rho_inf = self.rho_inf
            T_inf   = self.T_inf
            
            nx = self.nx
            ny = self.ny
            nz = self.nz
            nt = self.nt
            
            ## dimless (inlet)
            xd = self.x
            yd = self.y
            zd = self.z
            td = self.t
            
            ## dimensional [m] / [s]
            x = self.x * lchar 
            y = self.y * lchar
            z = self.z * lchar
            t = self.t * (lchar/U_inf)
            
            t_meas = t[-1]-t[0]
            dt     = self.dt * (lchar/U_inf)
            
            np.testing.assert_equal(nx,x.size)
            np.testing.assert_equal(ny,y.size)
            np.testing.assert_equal(nz,z.size)
            np.testing.assert_equal(nt,t.size)
            np.testing.assert_allclose(dt, t[1]-t[0], rtol=1e-8)
            
            # === report
            if verbose:
                even_print('nx'     , '%i'        %nx     )
                even_print('ny'     , '%i'        %ny     )
                even_print('nz'     , '%i'        %nz     )
                even_print('nt'     , '%i'        %nt     )
                even_print('dt'     , '%0.5e [s]' %dt     )
                even_print('t_meas' , '%0.5e [s]' %t_meas )
                print(72*'-')
            
            if verbose:
                even_print('Re_τ'   , '%0.1f'        % Re_tau_avg   )
                even_print('Re_θ'   , '%0.1f'        % Re_theta_avg )
                even_print('δ99'    , '%0.5e [m]'    % d99_avg      )
                even_print('U_inf'  , '%0.3f [m/s]'  % U_inf        )
                even_print('u_τ'    , '%0.3f [m/s]'  % u_tau_avg    )
                even_print('ν_wall' , '%0.5e [m²/s]' % nu_wall_avg  )
                print(72*'-')
            
            t_eddy = t_meas / (d99_avg/u_tau_avg)
            
            if verbose:
                even_print('t_meas/(δ99/u_τ) = t_eddy' , '%0.2f'%t_eddy)
                even_print('t_meas/(δ99/u99)'          , '%0.2f'%(t_meas/(d99_avg/u99_avg)))
                even_print('t_meas/(20·δ99/u99)'       , '%0.2f'%(t_meas/(20*d99_avg/u99_avg)))
                print(72*'-')
        
        # ===
        
        scalars = ['uI','vI','wI','TI']
        scalars_dtypes = [self.scalars_dtypes_dict[s] for s in scalars]
        
        ## 5D [scalar][x,y,z,t] structured array
        data = np.zeros(shape=(nx, nyr, nz, nt), dtype={'names':scalars, 'formats':scalars_dtypes})
        
        for scalar in scalars:
            dset = self['data/%s'%scalar]
            self.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                data[scalar] = dset[:,:,ry1:ry2,:].T
            self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
            if (self.rank==0):
                even_print('read: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        # === redimensionalize prime data --> is this really necessary here??? i dont think so.
        
        data['uI'] *= U_inf
        data['vI'] *= U_inf
        data['wI'] *= U_inf
        data['TI'] *= T_inf
        
        ## 5D [scalar][x,y,z,t] structured array
        scalars_R = ['R_uIuI','R_vIvI','R_wIwI','R_TITI'] #, 'R_uIvI','R_uIwI']
        scalars_dtypes_R = [np.float32 for s in scalars_R]
        ##
        data_R = np.zeros(shape=(nx, nyr, nz*2-1, nt), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
        #data_R = np.zeros(shape=(nx, nyr, nz, nt), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
        
        if True:
            if verbose: progress_bar = tqdm(total=nx*nyr*nt, ncols=100, desc='corr_span()', leave=False, file=sys.stdout)
            for xi in range(nx):
                for yi in range(nyr):
                    for ti in range(nt):
                        
                        uI_ = np.copy(data['uI'][xi,yi,:,ti])
                        vI_ = np.copy(data['vI'][xi,yi,:,ti])
                        wI_ = np.copy(data['wI'][xi,yi,:,ti])
                        TI_ = np.copy(data['TI'][xi,yi,:,ti])
                        
                        # ===
                        
                        # aaa = np.zeros((2*nz-1), dtype=np.float32)
                        # for zi1 in np.arange(nz):
                        #     for zi2 in np.flip(np.arange(nz)):
                        #         aaa[zi1]
                        
                        aaa = ( uI_[0] * uI_[20] ) / ( np.sqrt(uI_[0]**2) * np.sqrt(uI_[20]**2) )
                        
                        # ===
                        
                        conv_uIuI = sp.signal.correlate(uI_, uI_, mode='full', method='direct')
                        norm_uIuI = np.sqrt( np.sum( uI_**2 ) ) * np.sqrt( np.sum( uI_**2 ) )
                        if (norm_uIuI==0.):
                            norm_uIuI = 1e-8
                        data_R['R_uIuI'][xi,yi,:,ti] = conv_uIuI / norm_uIuI
                        
                        conv_vIvI = sp.signal.correlate(vI_, vI_, mode='full', method='direct')
                        norm_vIvI = np.sqrt( np.sum( vI_**2 ) ) * np.sqrt( np.sum( vI_**2 ) )
                        if (norm_vIvI==0.):
                            norm_vIvI = 1e-8
                        data_R['R_vIvI'][xi,yi,:,ti] = conv_vIvI / norm_vIvI
                        
                        conv_wIwI = sp.signal.correlate(wI_, wI_, mode='full', method='direct')
                        norm_wIwI = np.sqrt( np.sum( wI_**2 ) ) * np.sqrt( np.sum( wI_**2 ) )
                        if (norm_wIwI==0.):
                            norm_wIwI = 1e-8
                        data_R['R_wIwI'][xi,yi,:,ti] = conv_wIwI / norm_wIwI
                        
                        conv_TITI = sp.signal.correlate(TI_, TI_, mode='full', method='direct')
                        norm_TITI = np.sqrt( np.sum( TI_**2 ) ) * np.sqrt( np.sum( TI_**2 ) )
                        if (norm_TITI==0.):
                            norm_TITI = 1e-8
                        data_R['R_TITI'][xi,yi,:,ti] = conv_TITI / norm_TITI
                        
                        if verbose: progress_bar.update()
            
            if verbose:
                progress_bar.close()
        
        # === average in [x,t] --> leave [y,z]
        
        #data_R_avg = np.zeros(shape=(nyr, nz), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
        data_R_avg = np.zeros(shape=(nyr, 2*nz-1), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
        for scalar_R in scalars_R:
            data_R_avg[scalar_R] = np.mean(data_R[scalar_R], axis=(0,3))
        
        # === gather all results
        
        self.comm.Barrier()
        data_R_all = None
        if (self.rank==0):
            
            j=0
            data_R_all = np.zeros(shape=(ny,nz*2-1), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
            #data_R_all = np.zeros(shape=(ny,nz), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
            
            ## data this rank
            for scalar_R in scalars_R:
                data_R_all[scalar_R][ryl[j][0]:ryl[j][1],:] = data_R_avg[scalar_R]
        
        for scalar_R in scalars_R:
            for ri in range(1,self.n_ranks):
                j = ri
                self.comm.Barrier()
                if (self.rank==ri):
                    sendbuf = np.copy(data_R_avg[scalar_R])
                    self.comm.Send(sendbuf, dest=0, tag=ri)
                    #print('rank %i sending %s'%(ri,scalar_R))
                elif (self.rank==0):
                    #print('rank %i receiving %s'%(self.rank,scalar_R))
                    nyri = ryl[j][1] - ryl[j][0]
                    ##
                    recvbuf = np.zeros((nyri,nz*2-1), dtype=data_R_avg[scalar_R].dtype)
                    #recvbuf = np.zeros((nyri,nz), dtype=data_R_avg[scalar_R].dtype)
                    ##
                    #print('rank %i : recvbuf.shape=%s'%(rank,str(recvbuf.shape)))
                    self.comm.Recv(recvbuf, source=ri, tag=ri)
                    data_R_all[scalar_R][ryl[j][0]:ryl[j][1],:] = recvbuf
                else:
                    pass
        
        # === overwrite
        
        if (self.rank==0):
            R = np.copy(data_R_all)
        
        # === save results
        if (self.rank==0):
            
            sc_l_in  = np.mean(sc_l_in  , axis=(0,1)) ## avg in [x,z] --> leave 0D scalar
            sc_u_in  = np.mean(sc_u_in  , axis=(0,1))
            sc_t_in  = np.mean(sc_t_in  , axis=(0,1))
            sc_l_out = np.mean(sc_l_out , axis=(0,1))
            sc_u_out = np.mean(sc_u_out , axis=(0,1))
            sc_t_out = np.mean(sc_t_out , axis=(0,1))
            
            data = {}
            data['R']        = R
            ##
            ## dimensional [m]
            data['x']        = x
            data['y']        = y
            data['z']        = z
            data['t']        = t
            ##
            data['sc_l_in']  = sc_l_in
            data['sc_l_out'] = sc_l_out
            data['lchar']    = lchar
            data['Re_tau']   = Re_tau
            data['Re_theta'] = Re_theta
            
            with open(fn_dat_corr_span,'wb') as f:
                pickle.dump(data, f, protocol=4)
            print('--w-> %s : %0.2f [MB]'%(fn_dat_corr_span,os.path.getsize(fn_dat_corr_span)/1024**2))
        
        # ===
        
        self.comm.Barrier()
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.calc_corr_span() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    # ===
    
    @staticmethod
    def calc_corr_stream(fn_rgd, comm, xe1, xe2, xe3, **kwargs):
        '''
        calculate autocorrelation in [x] and avg in [z,t]
        '''
        
        #fn_rgd             = kwargs.get('fn_rgd',None)
        fn_rgd_prime       = kwargs.get('fn_rgd_prime',None)
        fn_dat_corr_stream = kwargs.get('fn_dat_corr_stream',None)
        fn_dat_mean_dim    = kwargs.get('fn_dat_mean_dim',None)
        ##
        #xe1 = kwargs.get('xe1',None)
        #xe2 = kwargs.get('xe2',None)
        #xe3 = kwargs.get('xe3',None)
        
        rank    = comm.Get_rank()
        n_ranks = comm.Get_size()
        
        if (rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.calc_corr_stream()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        rt = kwargs.get('rt',1)
        
        ## for now only distribute data in [y]
        if (rx!=1):
            raise AssertionError('rx!=1')
        if (rz!=1):
            raise AssertionError('rz!=1')
        if (rt!=1):
            raise AssertionError('rt!=1')
        
        if (rx*ry*rz*rt != n_ranks):
            raise AssertionError('rx*ry*rz*rt != n_ranks')
        
        if not os.path.isfile(fn_rgd):
            raise FileNotFoundError('%s not found!'%fn_rgd)
        
        # === prime file name (for reading)
        if (fn_rgd_prime is None):
            fname_path = os.path.dirname(fn_rgd)
            fname_base = os.path.basename(fn_rgd)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_prime_h5_base = fname_root+'_prime.h5'
            fn_rgd_prime = str(PurePosixPath(fname_path, fname_prime_h5_base))
        
        if not os.path.isfile(fn_rgd_prime):
            raise FileNotFoundError('%s not found!'%fn_rgd_prime)
        
        # === mean dimensional file name (for reading) : .dat
        if (fn_dat_mean_dim is None):
            fname_path = os.path.dirname(fn_rgd)
            fname_base = os.path.basename(fn_rgd)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_dat_mean_base = fname_root+'_mean_dim.dat'
            fn_dat_mean_dim = str(PurePosixPath(fname_path, fname_dat_mean_base))
        
        if not os.path.isfile(fn_dat_mean_dim):
            raise FileNotFoundError('%s not found!'%fn_dat_mean_dim)
        
        # === corr file name (for writing) : dat
        if (fn_dat_corr_stream is None):
            fname_path = os.path.dirname(fn_rgd)
            fname_base = os.path.basename(fn_rgd)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_corr_stream_dat_base = fname_root+'_corr_stream.dat'
            fn_dat_corr_stream = str(PurePosixPath(fname_path, fname_corr_stream_dat_base))
        
        if verbose: even_print('fn_rgd'             , fn_rgd             )
        if verbose: even_print('fn_rgd_prime'       , fn_rgd_prime       )
        if verbose: even_print('fn_dat_mean_dim'    , fn_dat_mean_dim    )
        if verbose: even_print('fn_dat_corr_stream' , fn_dat_corr_stream )
        if verbose: print(72*'-')
        
        hf_rgd   = rgd(fn_rgd,       'r', driver='mpio', comm=comm, libver='latest')
        hf_prime = rgd(fn_rgd_prime, 'r', driver='mpio', comm=comm, libver='latest')
        #hf_mean  = rgd(fn_rgd_mean,  'r', driver='mpio', comm=comm, libver='latest')
        
        if (xe3 != hf_rgd.nx):
            raise AssertionError('xe3 != hf_rgd.nx')
        
        if (hf_rgd.nx!=hf_prime.nx):
            raise AssertionError('hf_rgd.nx!=hf_prime.nx')
        if (hf_rgd.ny!=hf_prime.ny):
            raise AssertionError('hf_rgd.ny!=hf_prime.ny')
        if (hf_rgd.nz!=hf_prime.nz):
            raise AssertionError('hf_rgd.nz!=hf_prime.nz')
        if (hf_rgd.nt!=hf_prime.nt):
            raise AssertionError('hf_rgd.nt!=hf_prime.nt')
        
        if (rx>hf_rgd.nx):
            raise AssertionError('rx>nx')
        if (ry>hf_rgd.ny):
            raise AssertionError('ry>ny')
        if (rz>hf_rgd.nz):
            raise AssertionError('rz>nz')
        if (rt>hf_rgd.nt):
            raise AssertionError('rt>nt')
        
        # ===
        
        #comm4d = hf_rgd.comm.Create_cart(dims=[rx,ry,ry,rt], periods=[False,False,False,False], reorder=False)
        #t4d = comm4d.Get_coords(hf_rgd.rank)
        
        #rxl_ = np.array_split(np.array(range(hf_rgd.nx),dtype=np.int64),min(rx,hf_rgd.nx))
        ryl_ = np.array_split(np.array(range(hf_rgd.ny),dtype=np.int64),min(ry,hf_rgd.ny))
        #rzl_ = np.array_split(np.array(range(hf_rgd.nz),dtype=np.int64),min(rz,hf_rgd.nz))
        #rtl_ = np.array_split(np.array(range(hf_rgd.nt),dtype=np.int64),min(rt,hf_rgd.nt))
        
        #rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        #rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        #rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        #ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        #rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        ry1,ry2 = ryl[hf_rgd.rank]; nyr = ry2 - ry1
        
        # === read mean dim data
        
        if True:
            
            # === read in data (mean dim) --> every rank gets full [x,z]
            with open(fn_dat_mean_dim,'rb') as f:
                data_mean_dim = pickle.load(f)
            fmd = type('foo', (object,), data_mean_dim)
            
            # === 2D dimensional quantities --> [x,z]
            u_tau    = fmd.u_tau
            nu_wall  = fmd.nu_wall
            d99      = fmd.d99
            u99      = fmd.u99
            Re_tau   = fmd.Re_tau
            Re_theta = fmd.Re_theta
            
            u_tau_avg    = np.mean(fmd.u_tau    , axis=(0,1))
            nu_wall_avg  = np.mean(fmd.nu_wall  , axis=(0,1))
            d99_avg      = np.mean(fmd.d99      , axis=(0,1))
            u99_avg      = np.mean(fmd.u99      , axis=(0,1))
            Re_tau_avg   = np.mean(fmd.Re_tau   , axis=(0,1))
            Re_theta_avg = np.mean(fmd.Re_theta , axis=(0,1))
            
            # === 2D inner scales --> [x,z]
            sc_l_in = nu_wall / u_tau
            sc_u_in = u_tau
            sc_t_in = nu_wall / u_tau**2
            
            # === 2D outer scales --> [x,z]
            sc_l_out = d99
            sc_u_out = u99
            sc_t_out = d99/u99
            
            # === check
            np.testing.assert_allclose(fmd.lchar   , hf_rgd.lchar   , rtol=1e-8)
            np.testing.assert_allclose(fmd.U_inf   , hf_rgd.U_inf   , rtol=1e-8)
            np.testing.assert_allclose(fmd.rho_inf , hf_rgd.rho_inf , rtol=1e-8)
            np.testing.assert_allclose(fmd.T_inf   , hf_rgd.T_inf   , rtol=1e-8)
            np.testing.assert_allclose(fmd.nx      , hf_rgd.nx      , rtol=1e-8)
            np.testing.assert_allclose(fmd.ny      , hf_rgd.ny      , rtol=1e-8)
            np.testing.assert_allclose(fmd.nz      , hf_rgd.nz      , rtol=1e-8)
            np.testing.assert_allclose(fmd.xs      , hf_rgd.x       , rtol=1e-8)
            np.testing.assert_allclose(fmd.ys      , hf_rgd.y       , rtol=1e-8)
            np.testing.assert_allclose(fmd.zs      , hf_rgd.z       , rtol=1e-8)
            
            lchar   = hf_rgd.lchar
            U_inf   = hf_rgd.U_inf
            rho_inf = hf_rgd.rho_inf
            T_inf   = hf_rgd.T_inf
            
            nx = hf_rgd.nx
            ny = hf_rgd.ny
            nz = hf_rgd.nz
            nt = hf_rgd.nt
            
            ## dimless (inlet)
            xd = hf_rgd.x
            yd = hf_rgd.y
            zd = hf_rgd.z
            td = hf_rgd.t
            
            ## dimensional [m] & [s]
            x = hf_rgd.x * lchar 
            y = hf_rgd.y * lchar
            z = hf_rgd.z * lchar
            t = hf_rgd.t * (lchar/U_inf)
            
            t_meas = t[-1]-t[0]
            dt     = hf_rgd.dt * (lchar/U_inf)
            
            np.testing.assert_equal(nx,x.size)
            np.testing.assert_equal(ny,y.size)
            np.testing.assert_equal(nz,z.size)
            np.testing.assert_equal(nt,t.size)
            np.testing.assert_allclose(dt, t[1]-t[0], rtol=1e-8)
            
            # === report
            if verbose:
                even_print('xe1','%i'%xe1)
                even_print('xe2','%i'%xe2)
                even_print('xe3','%i'%xe3)
            
            if verbose:
                even_print('nx'     , '%i'        %nx     )
                even_print('ny'     , '%i'        %ny     )
                even_print('nz'     , '%i'        %nz     )
                even_print('nt'     , '%i'        %nt     )
                even_print('dt'     , '%0.5e [s]' %dt     )
                even_print('t_meas' , '%0.5e [s]' %t_meas )
                print(72*'-')
            
            if verbose:
                even_print('Re_τ'   , '%0.1f'        % Re_tau_avg   )
                even_print('Re_θ'   , '%0.1f'        % Re_theta_avg )
                even_print('δ99'    , '%0.5e [m]'    % d99_avg      )
                even_print('U_inf'  , '%0.3f [m/s]'  % U_inf        )
                even_print('u_τ'    , '%0.3f [m/s]'  % u_tau_avg    )
                even_print('ν_wall' , '%0.5e [m²/s]' % nu_wall_avg  )
                print(72*'-')
            
            t_eddy = t_meas / (d99_avg/u_tau_avg)
            
            if verbose:
                even_print('t_meas/(δ99/u_τ) = t_eddy' , '%0.2f'%t_eddy)
                even_print('t_meas/(δ99/u99)'          , '%0.2f'%(t_meas/(d99_avg/u99_avg)))
                even_print('t_meas/(20·δ99/u99)'       , '%0.2f'%(t_meas/(20*d99_avg/u99_avg)))
                print(72*'-')
        
        # === x-grid index start / end (self-similar)
        
        d99_1D = np.mean(fmd.d99, axis=(1,)) ## avg in [z] --> leave [x]
        
        d99_end = d99_1D[xe2] / lchar ## dimless (inlet)
        
        ## xe2ss   = np.abs(xd - (xd[xe2]-((5+0)*d99_end))).argmin() ## self-similar end @ x_uniform_end -  N    * d99
        ## xe1ss   = np.abs(xd - (xd[xe2]-((5+2)*d99_end))).argmin() ## self-similar end @ x_uniform_end - (N+M) * d99
        
        xe2ss   = np.abs(xd - (xd[xe2]-((0.1)*d99_end))).argmin() ## self-similar end @ x_uniform_end -  N    * d99
        xe1ss   = np.abs(xd - (xd[xe2]-((0.1+2)*d99_end))).argmin() ## self-similar end @ x_uniform_end - (N+M) * d99
        
        if verbose: even_print('xe1ss','%i'%xe1ss)
        if verbose: even_print('xe2ss','%i'%xe2ss)
        
        nxss = xe2ss - xe1ss
        if verbose: even_print('nxss','%i'%nxss)
        if verbose: print(72*'-')
        
        # === do read
        ss1 = ['T','rho'] ## from RGD
        ss2 = ['uI','vI','wI','TI'] ## from RGD prime
        
        fmts1 = [hf_rgd.scalars_dtypes_dict[s]   for s in ss1]
        fmts2 = [hf_prime.scalars_dtypes_dict[s] for s in ss2]
        
        ## 5D [scalar][x,y,z,t] structured array
        data = np.zeros(shape=(nxss, nyr, nz, nt), dtype={'names':ss1+ss2, 'formats':fmts1+fmts2})
        
        for ss in ss1:
            dset = hf_rgd['data/%s'%ss]
            hf_rgd.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                data[ss] = dset[:,:,ry1:ry2,xe1ss:xe2ss].T
            hf_rgd.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4 * nxss * hf_rgd.ny * hf_rgd.nz * hf_rgd.nt / 1024**3
            if (hf_rgd.rank==0):
                even_print('read: %s'%ss, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        for ss in ss2:
            dset = hf_prime['data/%s'%ss]
            hf_prime.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                data[ss] = dset[:,:,ry1:ry2,xe1ss:xe2ss].T
            hf_prime.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4 * nxss * hf_prime.ny * hf_prime.nz * hf_prime.nt / 1024**3
            if (hf_prime.rank==0):
                even_print('read: %s'%ss, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        hf_rgd.close()
        hf_prime.close()
        #hf_mean.close()
        
        # === redimensionalize hf_rgd & hf_prime data
        
        for var in data.dtype.names:
            if var in ['u','v','w','uI','vI','wI']:
                data[var] *= U_inf
            elif var in ['T','TI']:
                data[var] *= T_inf
            elif var in ['rho','rhoI']:
                data[var] *= rho_inf
            elif var in ['p','pI']:
                data[var] *= (rho_inf * U_inf**2)
            else:
                raise ValueError('condition needed for redimensionalizing \'%s\''%var)
        
        ## get μ from T
        mu = np.copy(14.58e-7*data['T']**1.5/(data['T']+110.4))
        
        # === get a specific y
        
        if False:
            
            yi = 60
            for ri in range(n_ranks):
                j = ri
                if (ryl[j][1] > yi) and (ryl[j][0] <= yi):
                    riX = ri ## this rank has the data at yi
            
            if (rank==riX):
                j = riX
                yi_ = yi - ryl[j][0]
                uI_100 = np.copy(data['uI'][:,yi_,:,:])
            
            comm.Barrier()
            
            if (rank!=riX):
                uI_100 = None
            uI_100 = comm.bcast(uI_100, root=riX)
            
            comm.Barrier()
        
        # === get τ_w′ : fluctuating wall shear stress (first rank only)
        
        if (rank==0):
            tau_wall_I = np.zeros((nxss,1,nz,nt), dtype=data['uI'].dtype)
            
            y_ = np.copy( y[ry1:ry2] )
            
            if verbose: progress_bar = tqdm(total=nxss*nz*nt, ncols=100, desc='τ_w′', leave=False, file=sys.stdout)
            for xi in range(nxss):
                for zi in range(nz):
                    for ti in range(nt):
                        uI_    = np.copy(data['uI'][xi,:,zi,ti])
                        duIdy_ = sp.interpolate.CubicSpline(y_, uI_, bc_type='natural')(y_,1)
                        ##
                        mu_wall_    = mu[xi,0,zi,ti]
                        duIdy_wall_ = duIdy_[0]
                        ##
                        tau_wall_I_ = mu_wall_ * duIdy_wall_
                        tau_wall_I[xi,0,zi,ti] = tau_wall_I_
                        ##
                        if verbose:
                            progress_bar.update()
            if verbose:
                progress_bar.close()
        
        comm.Barrier()
        
        ## send tau_wall_I to all ranks
        if (rank!=0):
            tau_wall_I = None
        tau_wall_I = comm.bcast(tau_wall_I, root=0)
        
        # ===
        
        ## 5D [scalar][x,y,z,t] structured array
        scalars_R = ['R_uIuI','R_uItauwI','R_vIvI','R_wIwI','R_TITI']
        scalars_dtypes_R = [np.float32 for s in scalars_R]
        data_R = np.zeros(shape=(nxss*2-1, nyr, nz, nt), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
        
        # ===
        
        if True:
            if verbose: progress_bar = tqdm(total=nz*nyr*nt, ncols=100, desc='corr_stream()', leave=False, file=sys.stdout)
            for zi in range(nz):
                for ti in range(nt):
                    ## uI_100_     = np.copy(uI_100[:,zi,ti])
                    tau_wall_I_ = np.copy(tau_wall_I[:,0,zi,ti])
                    for yi in range(nyr):
                        
                        uI_ = np.copy(data['uI'][:,yi,zi,ti])
                        vI_ = np.copy(data['vI'][:,yi,zi,ti])
                        wI_ = np.copy(data['wI'][:,yi,zi,ti])
                        TI_ = np.copy(data['TI'][:,yi,zi,ti])
                        
                        ## ## convolve & correlate test
                        ## conv_uIuI_1 = sp.signal.convolve(uI_,  np.flip(uI_), mode='full', method='direct')
                        ## conv_uIuI_2 = sp.signal.correlate(uI_, uI_,          mode='full', method='direct')
                        ## np.testing.assert_allclose(conv_uIuI_1, conv_uIuI_2, rtol=1e-8)
                        
                        ## norm_uIuI_1 = np.sum( uI_ * uI_ , axis=0)
                        ## norm_uIuI_2 = np.sqrt( np.sum( uI_**2 ) ) * np.sqrt( np.sum( uI_**2 ) )
                        ## #norm_uIuI_2 = np.sqrt( np.sum( uI_**2 ) * np.sum( uI_**2 ) )
                        ## #norm_uIuI_2 = np.sum( np.sqrt( uI_**2 ) ) * np.sum( np.sqrt( uI_**2 ) )
                        ## np.testing.assert_allclose(norm_uIuI_1, norm_uIuI_2, rtol=1e-6)
                        
                        conv_uIuI = sp.signal.correlate(uI_, uI_, mode='full', method='direct')
                        norm_uIuI = np.sqrt( np.sum( uI_**2 ) ) * np.sqrt( np.sum( uI_**2 ) )
                        
                        if (norm_uIuI==0.):
                            norm_uIuI = 1e-8
                        data_R['R_uIuI'][:,yi,zi,ti] = conv_uIuI / norm_uIuI
                        
                        # ===
                        
                        conv_uItauwI = sp.signal.correlate(uI_, tau_wall_I_, mode='full', method='direct')
                        norm_uItauwI = np.sqrt( np.sum( uI_**2 ) ) * np.sqrt( np.sum( tau_wall_I_**2 ) )
                        if (norm_uItauwI==0.):
                            norm_uItauwI = 1e-8
                        data_R['R_uItauwI'][:,yi,zi,ti] = conv_uItauwI / norm_uItauwI
                        
                        # ===
                        
                        conv_vIvI = sp.signal.correlate(vI_, vI_, mode='full', method='direct')
                        norm_vIvI = np.sqrt( np.sum( vI_**2 ) ) * np.sqrt( np.sum( vI_**2 ) )
                        if (norm_vIvI==0.):
                            norm_vIvI = 1e-8
                        data_R['R_vIvI'][:,yi,zi,ti] = conv_vIvI / norm_vIvI
                        
                        conv_wIwI = sp.signal.correlate(wI_, wI_, mode='full', method='direct')
                        norm_wIwI = np.sqrt( np.sum( wI_**2 ) ) * np.sqrt( np.sum( wI_**2 ) )
                        if (norm_wIwI==0.):
                            norm_wIwI = 1e-8
                        data_R['R_wIwI'][:,yi,zi,ti] = conv_wIwI / norm_wIwI
                        
                        conv_TITI = sp.signal.correlate(TI_, TI_, mode='full', method='direct')
                        norm_TITI = np.sqrt( np.sum( TI_**2 ) ) * np.sqrt( np.sum( TI_**2 ) )
                        if (norm_TITI==0.):
                            norm_TITI = 1e-8
                        data_R['R_TITI'][:,yi,zi,ti] = conv_TITI / norm_TITI
                        
                        if verbose: progress_bar.update()
            
            if verbose:
                progress_bar.close()
        
        # === average in [z,t] --> leave [x,y]
        
        data_R_avg = np.zeros(shape=(nxss*2-1, nyr), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
        for scalar_R in scalars_R:
            data_R_avg[scalar_R] = np.mean(data_R[scalar_R], axis=(2,3))
        
        # === gather all results
        
        comm.Barrier()
        data_R_all = None
        if (rank==0):
            
            j=0
            data_R_all = np.zeros(shape=(nxss*2-1,ny), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
            
            ## data this rank
            for scalar_R in scalars_R:
                data_R_all[scalar_R][:,ryl[j][0]:ryl[j][1]] = data_R_avg[scalar_R]
        
        comm.Barrier()
        for scalar_R in scalars_R:
            for ri in range(1,n_ranks):
                j = ri
                
                if (rank==ri):
                    sendbuf = np.copy(data_R_avg[scalar_R])
                    comm.Send(sendbuf, dest=0, tag=ri)
                    #print('rank %i sending %s'%(ri,scalar_R))
                elif (rank==0):
                    #print('rank %i receiving %s'%(rank,scalar_R))
                    nyri = ryl[j][1] - ryl[j][0]
                    #recvbuf = np.zeros((nyri,nz), dtype=data_R_avg[scalar_R].dtype)
                    #recvbuf = np.zeros((nxss,nyri), dtype=data_R_avg[scalar_R].dtype)
                    recvbuf = np.zeros((nxss*2-1,nyri), dtype=data_R_avg[scalar_R].dtype)
                    #print('rank %i : recvbuf.shape=%s'%(rank,str(recvbuf.shape)))
                    comm.Recv(recvbuf, source=ri, tag=ri)
                    data_R_all[scalar_R][:,ryl[j][0]:ryl[j][1]] = recvbuf
                else:
                    pass
                
                comm.Barrier()
            comm.Barrier()
        
        # === overwrite
        
        if (rank==0):
            R = np.copy(data_R_all)
        
        # === save results
        if (rank==0):
            
            sc_l_in  = np.mean(sc_l_in  , axis=(1,)) ## avg in [z] --> leave [x]
            sc_u_in  = np.mean(sc_u_in  , axis=(1,))
            sc_t_in  = np.mean(sc_t_in  , axis=(1,))
            sc_l_out = np.mean(sc_l_out , axis=(1,))
            sc_u_out = np.mean(sc_u_out , axis=(1,))
            sc_t_out = np.mean(sc_t_out , axis=(1,))
            
            data = {}
            data['R']        = R
            ##
            ## dimensional [m]
            data['x']        = x[xe1ss:xe2ss]
            data['y']        = y
            data['z']        = z
            data['t']        = t
            ##
            data['sc_l_in']  = sc_l_in[xe1ss:xe2ss]
            data['sc_l_out'] = sc_l_out[xe1ss:xe2ss]
            data['lchar']    = lchar
            data['Re_tau']   = Re_tau[xe1ss:xe2ss,:]
            data['Re_theta'] = Re_theta[xe1ss:xe2ss,:]
            
            with open(fn_dat_corr_stream,'wb') as f:
                pickle.dump(data, f, protocol=4)
            print('--w-> %s : %0.2f [MB]'%(fn_dat_corr_stream,os.path.getsize(fn_dat_corr_stream)/1024**2))
        
        # ===
        
        comm.Barrier()
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.calc_corr_stream() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    def calc_turb_budget(self, **kwargs):
        '''
        calculate turbulent kinetic energy (k) budget
        -----
        --> dimensional [SI]
        '''
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.calc_turb_budget()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        rt = kwargs.get('rt',1)
        
        force                   = kwargs.get('force',False)
        chunk_kb                = kwargs.get('chunk_kb',1024)
        hiOrder                 = kwargs.get('hiOrder',False)
        #fn_rgd                 = kwargs.get('fn_rgd',None) ## self
        fn_rgd_mean             = kwargs.get('fn_rgd_mean',None)
        fn_rgd_prime            = kwargs.get('fn_rgd_prime',None)
        fn_rgd_turb_budget      = kwargs.get('fn_rgd_turb_budget',None)
        fn_rgd_turb_budget_mean = kwargs.get('fn_rgd_turb_budget_mean',None)
        ##
        save_unsteady           = kwargs.get('save_unsteady',False)
        
        ## for now only distribute data in [y]
        if (rx!=1):
            raise AssertionError('rx!=1')
        if (ry!=1):
            raise AssertionError('ry!=1')
        if (rz!=1):
            raise AssertionError('rz!=1')
        
        if (rx*ry*rz*rt != self.n_ranks):
            raise AssertionError('rx*ry*rz*rt != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        if (rt>self.nt):
            raise AssertionError('rt>self.nt')
        
        # ===
        
        #comm4d = self.comm.Create_cart(dims=[rx,ry,ry,rt], periods=[False,False,False,False], reorder=False)
        #t4d = comm4d.Get_coords(self.rank)
        
        #rxl_ = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
        #ryl_ = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
        #rzl_ = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
        rtl_ = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))

        #rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        #ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        #rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        #rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        #ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        #rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        rt1,rt2 = rtl[self.rank]; ntr = rt2 - rt1
        
        # === mean file name (for reading)
        if (fn_rgd_mean is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_mean_h5_base = fname_root+'_mean.h5'
            #fn_rgd_mean = os.path.join(fname_path, fname_mean_h5_base)
            fn_rgd_mean = str(PurePosixPath(fname_path, fname_mean_h5_base))
            #fn_rgd_mean = Path(fname_path, fname_mean_h5_base)
        
        if not os.path.isfile(fn_rgd_mean):
            raise FileNotFoundError('%s not found!'%fn_rgd_mean)
        
        # === prime file name (for reading)
        if (fn_rgd_prime is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_prime_h5_base = fname_root+'_prime.h5'
            #fn_rgd_prime = os.path.join(fname_path, fname_prime_h5_base)
            fn_rgd_prime = str(PurePosixPath(fname_path, fname_prime_h5_base))
            #fn_rgd_prime = Path(fname_path, fname_prime_h5_base)
        
        if not os.path.isfile(fn_rgd_prime):
            raise FileNotFoundError('%s not found!'%fn_rgd_prime)
        
        # === turb_budget file name (for writing)
        if (fn_rgd_turb_budget is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_turb_budget_h5_base = fname_root+'_turb_budget.h5'
            #fn_rgd_turb_budget = os.path.join(fname_path, fname_turb_budget_h5_base)
            fn_rgd_turb_budget = str(PurePosixPath(fname_path, fname_turb_budget_h5_base))
            #fn_rgd_turb_budget = Path(fname_path, fname_turb_budget_h5_base)
        
        if os.path.isfile(fn_rgd_turb_budget) and (force is False):
            raise FileNotFoundError('%s already present & force=False'%fn_rgd_turb_budget)
        
        # === turb_budget (mean) file name (for writing)
        if (fn_rgd_turb_budget_mean is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fn_rgd_turb_budget_mean_base = fname_root+'_turb_budget_mean.h5'
            fn_rgd_turb_budget_mean = str(PurePosixPath(fname_path, fn_rgd_turb_budget_mean_base))
        
        if os.path.isfile(fn_rgd_turb_budget_mean) and (force is False):
            raise FileNotFoundError('%s already present & force=False'%fn_rgd_turb_budget_mean)
        
        if verbose: even_print('fn_rgd'       , self.fname   )
        if verbose: even_print('fn_rgd_mean'  , fn_rgd_mean  )
        if verbose: even_print('fn_rgd_prime' , fn_rgd_prime )
        if verbose: even_print('fn_rgd_turb_budget' , fn_rgd_turb_budget )
        if verbose: even_print('fn_rgd_turb_budget_mean' , fn_rgd_turb_budget_mean )
        if verbose: print(72*'-')
        
        # ===
        
        if verbose: even_print('nx' , '%i'%self.nx)
        if verbose: even_print('ny' , '%i'%self.ny)
        if verbose: even_print('nz' , '%i'%self.nz)
        if verbose: even_print('nt' , '%i'%self.nt)
        if verbose: even_print('save_unsteady', str(save_unsteady))
        if verbose: print(72*'-')
        
        # === init outfiles
        
        ## unsteady turb budget
        if save_unsteady:
            
            with rgd(fn_rgd_turb_budget, 'w', force=force, driver='mpio', comm=self.comm, libver='latest') as f1:
                f1.init_from_rgd(self.fname)
                
                shape = (f1.nt,f1.nz,f1.ny,f1.nx)
                chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
                
                data_gb = 4*f1.nt*f1.nz*f1.ny*f1.nx / 1024**3
                
                for dss in ['unsteady_production','unsteady_dissipation','unsteady_transport','unsteady_diffusion','unsteady_diffusion2','unsteady_p_dilatation','unsteady_p_diffusion']:
                    
                    if verbose:
                        even_print('initializing data/%s'%(dss,),'%0.1f [GB]'%(data_gb,))
                    
                    dset = f1.create_dataset('data/%s'%dss, 
                                             shape=shape, 
                                             dtype=np.float32,
                                             chunks=chunks)
                    
                    chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                    if verbose:
                        even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                        even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            if verbose: print(72*'-')
        
        ## avg turb budget
        with rgd(fn_rgd_turb_budget_mean, 'w', force=force, driver='mpio', comm=self.comm, libver='latest') as f1:
            
            f1.attrs['duration_avg'] = self.t[-1] - self.t[0] ## add attribute for duration of mean
            #f1_mean.attrs['duration_avg'] = self.duration
            
            f1.init_from_rgd(self.fname, t_info=False)
            
            shape = (1,f1.nz,f1.ny,f1.nx)
            chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
            
            data_gb = 4*1*f1.nz*f1.ny*f1.nx / 1024**3
            
            for dss in ['production','dissipation','transport','diffusion','diffusion2','p_dilatation','p_diffusion']:
                
                if verbose:
                    even_print('initializing data/%s'%(dss,),'%0.1f [GB]'%(data_gb,))
                
                dset = f1.create_dataset('data/%s'%dss, 
                                         shape=shape, 
                                         dtype=np.float32,
                                         chunks=chunks)
                
                chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                if verbose:
                    even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                    even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            # === replace dims/t array --> take last time of series
            if ('dims/t' in f1):
                del f1['dims/t']
            f1.create_dataset('dims/t', data=np.array([self.t[-1]],dtype=np.float64))
            
            if hasattr(f1, 'duration_avg'):
                if verbose: even_print('duration_avg', '%0.2f'%f1.duration_avg)
        
        if verbose: print(72*'-')
        
        self.comm.Barrier()
        
        # ===
        
        ## with rgd(fn_rgd, 'r', driver='mpio', comm=comm, verbose=False) as hf_rgd: #self
        with rgd(fn_rgd_prime, 'r', driver='mpio', comm=self.comm, libver='latest') as hf_prime:
            with rgd(fn_rgd_mean, 'r', driver='mpio', comm=self.comm, libver='latest') as hf_mean:
                
                lchar   = self.lchar
                U_inf   = self.U_inf
                rho_inf = self.rho_inf
                T_inf   = self.T_inf
                ## p_inf = self.p_inf  ## shouldnt be used for redimensionalization of p
                
                x = lchar * self.x
                y = lchar * self.y
                z = lchar * self.z
                
                nx = self.nx
                ny = self.ny
                nz = self.nz
                nt = self.nt
                
                u_re = U_inf * hf_mean['data/u'][0,:,:,:].T
                v_re = U_inf * hf_mean['data/v'][0,:,:,:].T
                w_re = U_inf * hf_mean['data/w'][0,:,:,:].T
                u_fv = U_inf * hf_mean['data/u_fv'][0,:,:,:].T
                v_fv = U_inf * hf_mean['data/v_fv'][0,:,:,:].T
                w_fv = U_inf * hf_mean['data/w_fv'][0,:,:,:].T
                
                dudx_ij_fv = get_grad(u_fv, v_fv, w_fv, x, y, z, do_stack=False, hiOrder=hiOrder, verbose=False)
                dudx_ij_re = get_grad(u_re, v_re, w_re, x, y, z, do_stack=False, hiOrder=hiOrder, verbose=False)
                
                dudx_fv = dudx_ij_fv['dadx'][:,:,:,np.newaxis] ; dudx_re = dudx_ij_re['dadx'][:,:,:,np.newaxis]
                dudy_fv = dudx_ij_fv['dady'][:,:,:,np.newaxis] ; dudy_re = dudx_ij_re['dady'][:,:,:,np.newaxis]
                dudz_fv = dudx_ij_fv['dadz'][:,:,:,np.newaxis] ; dudz_re = dudx_ij_re['dadz'][:,:,:,np.newaxis]
                dvdx_fv = dudx_ij_fv['dbdx'][:,:,:,np.newaxis] ; dvdx_re = dudx_ij_re['dbdx'][:,:,:,np.newaxis]
                dvdy_fv = dudx_ij_fv['dbdy'][:,:,:,np.newaxis] ; dvdy_re = dudx_ij_re['dbdy'][:,:,:,np.newaxis]
                dvdz_fv = dudx_ij_fv['dbdz'][:,:,:,np.newaxis] ; dvdz_re = dudx_ij_re['dbdz'][:,:,:,np.newaxis]
                dwdx_fv = dudx_ij_fv['dcdx'][:,:,:,np.newaxis] ; dwdx_re = dudx_ij_re['dcdx'][:,:,:,np.newaxis]
                dwdy_fv = dudx_ij_fv['dcdy'][:,:,:,np.newaxis] ; dwdy_re = dudx_ij_re['dcdy'][:,:,:,np.newaxis]
                dwdz_fv = dudx_ij_fv['dcdz'][:,:,:,np.newaxis] ; dwdz_re = dudx_ij_re['dcdz'][:,:,:,np.newaxis]
                
                # === do read
                ss1 = ['u','v','w','p','T','rho']
                ss2 = ['uI','vI','wI','uII','vII','wII','pI','TI','rhoI']
                
                formats = [np.float32 for s in ss1+ss2]
                dd = np.zeros(shape=(nx,ny,nz,ntr), dtype={'names':ss1+ss2, 'formats':formats}, order='C')
                
                for ss in ss1:
                    dset = self['data/%s'%ss]
                    self.comm.Barrier()
                    t_start = timeit.default_timer()
                    with dset.collective:
                        dd[ss] = dset[rt1:rt2,:,:,:].T
                    self.comm.Barrier()
                    t_delta = timeit.default_timer() - t_start
                    data_gb = 4*nx*ny*nz*nt/1024**3
                    if verbose:
                        even_print('read: %s'%ss, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
                
                for ss in ss2:
                    dset = hf_prime['data/%s'%ss]
                    self.comm.Barrier()
                    t_start = timeit.default_timer()
                    with dset.collective:
                        dd[ss] = dset[rt1:rt2,:,:,:].T
                    self.comm.Barrier()
                    t_delta = timeit.default_timer() - t_start
                    data_gb = 4*nx*ny*nz*nt/1024**3
                    if verbose:
                        even_print('read: %s'%ss, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
                
                if verbose: print(72*'-')
                
                data_gb = dd.nbytes / 1024**3
                if verbose:
                    tqdm.write(even_print('data_gb (rank)', '%0.1f [GB]'%data_gb, s=True))
                
                mem_avail_gb = psutil.virtual_memory().available/1024**3
                mem_free_gb  = psutil.virtual_memory().free/1024**3
                if verbose:
                    tqdm.write(even_print('mem free', '%0.1f [GB]'%mem_free_gb, s=True))
                
                # === dimensionalize
                
                u    = dd['u']   * U_inf
                v    = dd['v']   * U_inf
                w    = dd['w']   * U_inf
                uI   = dd['uI']  * U_inf
                vI   = dd['vI']  * U_inf
                wI   = dd['wI']  * U_inf
                uII  = dd['uII'] * U_inf
                vII  = dd['vII'] * U_inf
                wII  = dd['wII'] * U_inf
                
                rho  = dd['rho']  * rho_inf
                rhoI = dd['rhoI'] * rho_inf
                p    = dd['p']    * (rho_inf * U_inf**2)
                pI   = dd['pI']   * (rho_inf * U_inf**2)
                T    = dd['T']    * T_inf
                TI   = dd['TI']   * T_inf
                
                mu  = np.copy(14.58e-7*T**1.5/(T+110.4))
                
                dd = None
                del dd
                
                # === get gradients
                
                t_start = timeit.default_timer()
                
                dudx   = np.gradient(u,   x, edge_order=1, axis=0)
                dudy   = np.gradient(u,   y, edge_order=1, axis=1)
                dudz   = np.gradient(u,   z, edge_order=1, axis=2)
                dvdx   = np.gradient(v,   x, edge_order=1, axis=0)
                dvdy   = np.gradient(v,   y, edge_order=1, axis=1)
                dvdz   = np.gradient(v,   z, edge_order=1, axis=2)
                dwdx   = np.gradient(w,   x, edge_order=1, axis=0)
                dwdy   = np.gradient(w,   y, edge_order=1, axis=1)
                dwdz   = np.gradient(w,   z, edge_order=1, axis=2)
                ##
                duIdx  = np.gradient(uI,  x, edge_order=1, axis=0)
                duIdy  = np.gradient(uI,  y, edge_order=1, axis=1)
                duIdz  = np.gradient(uI,  z, edge_order=1, axis=2)
                dvIdx  = np.gradient(vI,  x, edge_order=1, axis=0)
                dvIdy  = np.gradient(vI,  y, edge_order=1, axis=1)
                dvIdz  = np.gradient(vI,  z, edge_order=1, axis=2)
                dwIdx  = np.gradient(wI,  x, edge_order=1, axis=0)
                dwIdy  = np.gradient(wI,  y, edge_order=1, axis=1)
                dwIdz  = np.gradient(wI,  z, edge_order=1, axis=2)
                ##
                duIIdx = np.gradient(uII, x, edge_order=1, axis=0)
                duIIdy = np.gradient(uII, y, edge_order=1, axis=1)
                duIIdz = np.gradient(uII, z, edge_order=1, axis=2)
                dvIIdx = np.gradient(vII, x, edge_order=1, axis=0)
                dvIIdy = np.gradient(vII, y, edge_order=1, axis=1)
                dvIIdz = np.gradient(vII, z, edge_order=1, axis=2)
                dwIIdx = np.gradient(wII, x, edge_order=1, axis=0)
                dwIIdy = np.gradient(wII, y, edge_order=1, axis=1)
                dwIIdz = np.gradient(wII, z, edge_order=1, axis=2)
                
                t_delta = timeit.default_timer() - t_start
                
                if verbose:
                    tqdm.write(even_print('get gradients', format_time_string(t_delta), s=True))
                
                self.comm.Barrier()
                mem_avail_gb = psutil.virtual_memory().available/1024**3
                mem_free_gb  = psutil.virtual_memory().free/1024**3
                if verbose:
                    tqdm.write(even_print('mem available', '%0.1f [GB]'%mem_avail_gb, s=True))
                    tqdm.write(even_print('mem free',      '%0.1f [GB]'%mem_free_gb,  s=True))
                
                # === stack into tensors
                
                t_start = timeit.default_timer()
                
                uI_i  = np.copy(np.stack((uI,  vI,  wI ), axis=-1))
                uII_i = np.copy(np.stack((uII, vII, wII), axis=-1))
                
                duIdx_ij = np.stack((np.stack((duIdx, duIdy, duIdz), axis=4),
                                     np.stack((dvIdx, dvIdy, dvIdz), axis=4),
                                     np.stack((dwIdx, dwIdy, dwIdz), axis=4)), axis=5)
                
                duIIdx_ij = np.stack((np.stack((duIIdx, duIIdy, duIIdz), axis=4),
                                      np.stack((dvIIdx, dvIIdy, dvIIdz), axis=4),
                                      np.stack((dwIIdx, dwIIdy, dwIIdz), axis=4)), axis=5)
                
                t_delta = timeit.default_timer() - t_start
                if verbose:
                    tqdm.write(even_print('tensor stacking',format_time_string(t_delta), s=True))
                
                self.comm.Barrier()
                mem_avail_gb = psutil.virtual_memory().available/1024**3
                mem_free_gb  = psutil.virtual_memory().free/1024**3
                if verbose:
                    tqdm.write(even_print('mem available', '%0.1f [GB]'%mem_avail_gb, s=True))
                    tqdm.write(even_print('mem free',      '%0.1f [GB]'%mem_free_gb,  s=True))
                
                # === production
                if True:
                    
                    if verbose: print(72*'-')
                    t_start = timeit.default_timer()
                    
                    r_uII_uII = rho*uII*uII
                    r_uII_vII = rho*uII*vII
                    r_uII_wII = rho*uII*wII
                    
                    r_vII_uII = rho*vII*uII
                    r_vII_vII = rho*vII*vII
                    r_vII_wII = rho*vII*wII
                    
                    r_wII_uII = rho*wII*uII
                    r_wII_vII = rho*wII*vII
                    r_wII_wII = rho*wII*wII
                    
                    ## unsteady_production_ = - ( r_uII_uII * dudx_fv + r_uII_vII * dudy_fv + r_uII_wII * dudz_fv \
                    ##                          + r_uII_vII * dvdx_fv + r_vII_vII * dvdy_fv + r_vII_wII * dvdz_fv \
                    ##                          + r_uII_wII * dwdx_fv + r_vII_wII * dwdy_fv + r_wII_wII * dwdz_fv )
                    
                    r_uIIuII_ij = np.stack((np.stack((r_uII_uII, r_uII_vII, r_uII_wII), axis=4),
                                            np.stack((r_vII_uII, r_vII_vII, r_vII_wII), axis=4),
                                            np.stack((r_wII_uII, r_wII_vII, r_wII_wII), axis=4)), axis=5)
                    
                    dudx_fv_ij = np.stack((np.stack((dudx_fv, dudy_fv, dudz_fv), axis=4),
                                           np.stack((dvdx_fv, dvdy_fv, dvdz_fv), axis=4),
                                           np.stack((dwdx_fv, dwdy_fv, dwdz_fv), axis=4)), axis=5)
                    
                    unsteady_production = -1*np.einsum('xyztij,xyztij->xyzt', r_uIIuII_ij, dudx_fv_ij)
                    
                    ## np.testing.assert_allclose(unsteady_production, unsteady_production_, atol=20000)
                    ## print('check passed : np.einsum()')
                    
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('production', format_time_string(t_delta), s=True))
                    
                    # === do avg
                    t_start = timeit.default_timer()
                    unsteady_production_sum   = np.zeros((nx,ny,nz,1), dtype=np.float64, order='C')
                    unsteady_production_sum_i = np.sum(unsteady_production, axis=3, keepdims=True, dtype=np.float64)
                    self.comm.Reduce([unsteady_production_sum_i, MPI.DOUBLE], [unsteady_production_sum, MPI.DOUBLE], op=MPI.SUM, root=0)
                    if (self.rank==0):
                        production = ((1/nt)*unsteady_production_sum).astype(np.float32)
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('t avg production', format_time_string(t_delta),s=True))
                    
                    # === write avg
                    with rgd(fn_rgd_turb_budget_mean, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                        if (self.rank==0):
                            f1['data/production'][:,:,:,:] = production.T
                    self.comm.Barrier()
                    
                    # === write unsteady data
                    if save_unsteady:
                        with rgd(fn_rgd_turb_budget, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                            dset = f1['data/unsteady_production']
                            self.comm.Barrier()
                            t_start = timeit.default_timer()
                            with dset.collective:
                                dset[rt1:rt2,:,:,:] = unsteady_production.T
                            t_delta = timeit.default_timer() - t_start
                            data_gb = 4*nx*ny*nz*nt/1024**3
                            if verbose:
                                even_print('write: unsteady production', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
                    
                    # === release mem
                    r_uII_uII           = None; del r_uII_uII
                    r_uII_vII           = None; del r_uII_vII
                    r_uII_wII           = None; del r_uII_wII
                    r_vII_uII           = None; del r_vII_uII
                    r_vII_vII           = None; del r_vII_vII
                    r_vII_wII           = None; del r_vII_wII
                    r_wII_uII           = None; del r_wII_uII
                    r_wII_vII           = None; del r_wII_vII
                    r_wII_wII           = None; del r_wII_wII
                    r_uIIuII_ij         = None; del r_uIIuII_ij
                    dudx_fv_ij          = None; del dudx_fv_ij
                    unsteady_production = None; del unsteady_production
                    
                    self.comm.Barrier()
                    mem_avail_gb = psutil.virtual_memory().available/1024**3
                    mem_free_gb  = psutil.virtual_memory().free/1024**3
                    if verbose:
                        tqdm.write(even_print('mem available', '%0.1f [GB]'%mem_avail_gb, s=True))
                        tqdm.write(even_print('mem free',      '%0.1f [GB]'%mem_free_gb,  s=True))
                
                # === dissipation ε
                if True:
                    
                    if verbose: print(72*'-')
                    t_start = timeit.default_timer()
                    
                    duIIdx_ij_duIdx_ij = np.einsum('xyztij,xyztij->xyzt', duIIdx_ij, duIdx_ij)
                    duIIdx_ij_duIdx_ji = np.einsum('xyztij,xyztji->xyzt', duIIdx_ij, duIdx_ij)
                    
                    ## # === from pp_turbulent_budget.F90
                    ## unsteady_dissipation_ = mu * ( (duIdx + duIdx) * duIIdx  +  (duIdy + dvIdx) * duIIdy  +  (duIdz + dwIdx) * duIIdz \
                    ##                              + (dvIdx + duIdy) * dvIIdx  +  (dvIdy + dvIdy) * dvIIdy  +  (dvIdz + dwIdy) * dvIIdz \
                    ##                              + (dwIdx + duIdz) * dwIIdx  +  (dwIdy + dvIdz) * dwIIdy  +  (dwIdz + dwIdz) * dwIIdz )
                    
                    unsteady_dissipation = mu*(duIIdx_ij_duIdx_ij + duIIdx_ij_duIdx_ji)
                    
                    ## np.testing.assert_allclose(unsteady_dissipation, unsteady_dissipation_, rtol=1e-4)
                    ## print('check passed : np.einsum() : dissipation')
                    
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('dissipation', format_time_string(t_delta),s=True))
                    
                    # === do avg
                    t_start = timeit.default_timer()
                    unsteady_dissipation_sum   = np.zeros((nx,ny,nz,1), dtype=np.float64, order='C')
                    unsteady_dissipation_sum_i = np.sum(unsteady_dissipation, axis=3, keepdims=True, dtype=np.float64)
                    self.comm.Reduce([unsteady_dissipation_sum_i, MPI.DOUBLE], [unsteady_dissipation_sum, MPI.DOUBLE], op=MPI.SUM, root=0)
                    if (self.rank==0):
                        dissipation = ((1/nt)*unsteady_dissipation_sum).astype(np.float32)
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('t avg dissipation', format_time_string(t_delta),s=True))
                    
                    # === write avg
                    with rgd(fn_rgd_turb_budget_mean, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                        if (self.rank==0):
                            f1['data/dissipation'][:,:,:,:] = dissipation.T
                    self.comm.Barrier()
                    
                    # === write unsteady data
                    if save_unsteady:
                        with rgd(fn_rgd_turb_budget, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                            dset = f1['data/unsteady_dissipation']
                            self.comm.Barrier()
                            t_start = timeit.default_timer()
                            with dset.collective:
                                dset[rt1:rt2,:,:,:] = unsteady_dissipation.T
                            t_delta = timeit.default_timer() - t_start
                            data_gb = 4*nx*ny*nz*nt/1024**3
                            if verbose:
                                even_print('write: unsteady dissipation', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
                    
                    # === release mem
                    duIIdx_ij_duIdx_ij   = None; del duIIdx_ij_duIdx_ij
                    duIIdx_ij_duIdx_ji   = None; del duIIdx_ij_duIdx_ji
                    unsteady_dissipation = None; del unsteady_dissipation
                    
                    self.comm.Barrier()
                    mem_avail_gb = psutil.virtual_memory().available/1024**3
                    mem_free_gb  = psutil.virtual_memory().free/1024**3
                    if verbose:
                        tqdm.write(even_print('mem available', '%0.1f [GB]'%mem_avail_gb, s=True))
                        tqdm.write(even_print('mem free',      '%0.1f [GB]'%mem_free_gb,  s=True))
                
                # === transport
                if True:
                    
                    if verbose: print(72*'-')
                    t_start = timeit.default_timer()
                    
                    tc = np.einsum('xyzt,xyzti,xyzti,xyztj->xyztj', rho, uII_i, uII_i, uII_i) ## triple correlation
                    tc_ddx = np.gradient(tc, x, axis=0, edge_order=1)
                    tc_ddy = np.gradient(tc, y, axis=1, edge_order=1)
                    tc_ddz = np.gradient(tc, z, axis=2, edge_order=1)
                    unsteady_transport = -0.5*(tc_ddx[:,:,:,:,0] + tc_ddy[:,:,:,:,1] + tc_ddz[:,:,:,:,2])
                    
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('transport', format_time_string(t_delta),s=True))
                    
                    # === do avg
                    t_start = timeit.default_timer()
                    unsteady_transport_sum   = np.zeros((nx,ny,nz,1), dtype=np.float64, order='C')
                    unsteady_transport_sum_i = np.sum(unsteady_transport, axis=3, keepdims=True, dtype=np.float64)
                    self.comm.Reduce([unsteady_transport_sum_i, MPI.DOUBLE], [unsteady_transport_sum, MPI.DOUBLE], op=MPI.SUM, root=0)
                    if (self.rank==0):
                        transport = ( (1/nt)*unsteady_transport_sum ).astype(np.float32)
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('t avg transport', format_time_string(t_delta),s=True))
                    
                    # === write avg
                    with rgd(fn_rgd_turb_budget_mean, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                        if (self.rank==0):
                            f1['data/transport'][:,:,:,:] = transport.T
                    self.comm.Barrier()
                    
                    # === write unsteady data
                    if save_unsteady:
                        with rgd(fn_rgd_turb_budget, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                            dset = f1['data/unsteady_transport']
                            self.comm.Barrier()
                            t_start = timeit.default_timer()
                            with dset.collective:
                                dset[rt1:rt2,:,:,:] = unsteady_transport.T
                            t_delta = timeit.default_timer() - t_start
                            data_gb = 4*nx*ny*nz*nt/1024**3
                            if verbose:
                                even_print('write: unsteady transport', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
                    
                    # === release mem
                    tc     = None; del tc
                    tc_ddx = None; del tc_ddx
                    tc_ddy = None; del tc_ddy
                    tc_ddz = None; del tc_ddz
                    unsteady_transport = None; del unsteady_transport
                    
                    self.comm.Barrier()
                    mem_avail_gb = psutil.virtual_memory().available/1024**3
                    mem_free_gb  = psutil.virtual_memory().free/1024**3
                    if verbose:
                        tqdm.write(even_print('mem available', '%0.1f [GB]'%mem_avail_gb, s=True))
                        tqdm.write(even_print('mem free',      '%0.1f [GB]'%mem_free_gb,  s=True))
                
                # === (viscous) diffusion
                if True:
                    
                    if verbose: print(72*'-')
                    t_start = timeit.default_timer()
                    
                    omega_ij = duIdx_ij + np.transpose(duIdx_ij, axes=(0,1,2,3,5,4))
                    
                    if False:
                        
                        omega_ij_2 = np.stack((np.stack(((duIdx+duIdx), (dvIdx+duIdy), (dwIdx+duIdz)), axis=4),
                                               np.stack(((duIdy+dvIdx), (dvIdy+dvIdy), (dwIdy+dvIdz)), axis=4),
                                               np.stack(((duIdz+dwIdx), (dvIdz+dwIdy), (dwIdz+dwIdz)), axis=4)), axis=5)
                        
                        np.testing.assert_allclose(omega_ij, omega_ij_2, rtol=1e-8)
                        
                        if verbose:
                            print('check passed : np.einsum()')
                    
                    ## ## this one is likely wrong
                    ## omega_ij = np.stack((np.stack(((duIdx+duIdx), (dvIdx+duIdy), (dwIdx+duIdz)), axis=4),
                    ##                      np.stack(((duIdy+duIdx), (dvIdy+duIdy), (dwIdy+duIdz)), axis=4),
                    ##                      np.stack(((duIdz+duIdx), (dvIdz+duIdy), (dwIdz+duIdz)), axis=4)), axis=5)
                    
                    A = np.einsum('xyzt,xyzti,xyztij->xyztj', mu, uI_i, omega_ij)
                    A_ddx = np.gradient(A[:,:,:,:,0], x, axis=0, edge_order=1)
                    A_ddy = np.gradient(A[:,:,:,:,1], y, axis=1, edge_order=1)
                    A_ddz = np.gradient(A[:,:,:,:,2], z, axis=2, edge_order=1)
                    unsteady_diffusion = A_ddx + A_ddy + A_ddz
                    
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('diffusion', format_time_string(t_delta),s=True))
                    
                    # === do avg
                    t_start = timeit.default_timer()
                    unsteady_diffusion_sum   = np.zeros((nx,ny,nz,1), dtype=np.float64, order='C')
                    unsteady_diffusion_sum_i = np.sum(unsteady_diffusion, axis=3, keepdims=True, dtype=np.float64)
                    self.comm.Reduce([unsteady_diffusion_sum_i, MPI.DOUBLE], [unsteady_diffusion_sum, MPI.DOUBLE], op=MPI.SUM, root=0)
                    if (self.rank==0):
                        diffusion = ( (1/nt)*unsteady_diffusion_sum ).astype(np.float32)
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('t avg diffusion', format_time_string(t_delta),s=True))
                    
                    # === write avg
                    with rgd(fn_rgd_turb_budget_mean, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                        if (self.rank==0):
                            f1['data/diffusion'][:,:,:,:] = diffusion.T
                    self.comm.Barrier()
                    
                    # === write unsteady data
                    if save_unsteady:
                        with rgd(fn_rgd_turb_budget, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                            dset = f1['data/unsteady_diffusion']
                            self.comm.Barrier()
                            t_start = timeit.default_timer()
                            with dset.collective:
                                dset[rt1:rt2,:,:,:] = unsteady_diffusion.T
                            t_delta = timeit.default_timer() - t_start
                            data_gb = 4*nx*ny*nz*nt/1024**3
                            if verbose:
                                even_print('write: unsteady diffusion', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
                    
                    # === release mem
                    omega_ij = None; del omega_ij
                    A        = None; del A
                    A_ddx    = None; del A_ddx
                    A_ddy    = None; del A_ddy
                    A_ddz    = None; del A_ddz
                    unsteady_diffusion = None; del unsteady_diffusion
                    
                    self.comm.Barrier()
                    mem_avail_gb = psutil.virtual_memory().available/1024**3
                    mem_free_gb  = psutil.virtual_memory().free/1024**3
                    if verbose:
                        tqdm.write(even_print('mem available', '%0.1f [GB]'%mem_avail_gb, s=True))
                        tqdm.write(even_print('mem free',      '%0.1f [GB]'%mem_free_gb,  s=True))
                
                # === (viscous) diffusion2 --> wrong, currently in NS3D
                if True:
                    
                    if verbose: print(72*'-')
                    t_start = timeit.default_timer()
                    
                    ## omega_ij = duIdx_ij + np.transpose(duIdx_ij, axes=(0,1,2,3,5,4)) ## i think this is actually right but this is not whats in pp_turbulent_budget.F90
                    
                    ## this one is likely wrong
                    omega_ij = np.stack((np.stack(((duIdx+duIdx), (dvIdx+duIdy), (dwIdx+duIdz)), axis=4),
                                         np.stack(((duIdy+duIdx), (dvIdy+duIdy), (dwIdy+duIdz)), axis=4),
                                         np.stack(((duIdz+duIdx), (dvIdz+duIdy), (dwIdz+duIdz)), axis=4)), axis=5)
                    
                    A = np.einsum('xyzt,xyzti,xyztij->xyztj', mu, uI_i, omega_ij)
                    A_ddx = np.gradient(A[:,:,:,:,0], x, axis=0, edge_order=1)
                    A_ddy = np.gradient(A[:,:,:,:,1], y, axis=1, edge_order=1)
                    A_ddz = np.gradient(A[:,:,:,:,2], z, axis=2, edge_order=1)
                    unsteady_diffusion2 = A_ddx + A_ddy + A_ddz
                    
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('diffusion2', format_time_string(t_delta),s=True))
                    
                    # === do avg
                    t_start = timeit.default_timer()
                    unsteady_diffusion2_sum   = np.zeros((nx,ny,nz,1), dtype=np.float64, order='C')
                    unsteady_diffusion2_sum_i = np.sum(unsteady_diffusion2, axis=3, keepdims=True, dtype=np.float64)
                    self.comm.Reduce([unsteady_diffusion2_sum_i, MPI.DOUBLE], [unsteady_diffusion2_sum, MPI.DOUBLE], op=MPI.SUM, root=0)
                    if (self.rank==0):
                        diffusion2 = ( (1/nt)*unsteady_diffusion2_sum ).astype(np.float32)
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('t avg diffusion2', format_time_string(t_delta),s=True))
                    
                    # === write avg
                    with rgd(fn_rgd_turb_budget_mean, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                        if (self.rank==0):
                            f1['data/diffusion2'][:,:,:,:] = diffusion2.T
                    self.comm.Barrier()
                    
                    # === write unsteady data
                    if save_unsteady:
                        with rgd(fn_rgd_turb_budget, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                            dset = f1['data/unsteady_diffusion2']
                            self.comm.Barrier()
                            t_start = timeit.default_timer()
                            with dset.collective:
                                dset[rt1:rt2,:,:,:] = unsteady_diffusion2.T
                            t_delta = timeit.default_timer() - t_start
                            data_gb = 4*nx*ny*nz*nt/1024**3
                            if verbose:
                                even_print('write: unsteady diffusion2', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
                    
                    # === release mem
                    omega_ij = None; del omega_ij
                    A        = None; del A
                    A_ddx    = None; del A_ddx
                    A_ddy    = None; del A_ddy
                    A_ddz    = None; del A_ddz
                    unsteady_diffusion2 = None; del unsteady_diffusion2
                    
                    self.comm.Barrier()
                    mem_avail_gb = psutil.virtual_memory().available/1024**3
                    mem_free_gb  = psutil.virtual_memory().free/1024**3
                    if verbose:
                        tqdm.write(even_print('mem available', '%0.1f [GB]'%mem_avail_gb, s=True))
                        tqdm.write(even_print('mem free',      '%0.1f [GB]'%mem_free_gb,  s=True))
                
                # === pressure diffusion
                if True:
                    
                    if verbose: print(72*'-')
                    t_start = timeit.default_timer()
                    
                    A = np.einsum('xyzti,xyzt->xyzti', uII_i, pI)
                    A_ddx = np.gradient(A[:,:,:,:,0], x, axis=0, edge_order=1)
                    A_ddy = np.gradient(A[:,:,:,:,1], y, axis=1, edge_order=1)
                    A_ddz = np.gradient(A[:,:,:,:,2], z, axis=2, edge_order=1)
                    unsteady_p_diffusion = A_ddx + A_ddy + A_ddz
                    
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('p_diffusion', format_time_string(t_delta),s=True))
                    
                    # === do avg
                    t_start = timeit.default_timer()
                    unsteady_p_diffusion_sum   = np.zeros((nx,ny,nz,1), dtype=np.float64, order='C')
                    unsteady_p_diffusion_sum_i = np.sum(unsteady_p_diffusion, axis=3, keepdims=True, dtype=np.float64)
                    self.comm.Reduce([unsteady_p_diffusion_sum_i, MPI.DOUBLE], [unsteady_p_diffusion_sum, MPI.DOUBLE], op=MPI.SUM, root=0)
                    if (self.rank==0):
                        p_diffusion = ( (1/nt)*unsteady_p_diffusion_sum ).astype(np.float32)
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('t avg p_diffusion', format_time_string(t_delta),s=True))
                    
                    # === write avg
                    with rgd(fn_rgd_turb_budget_mean, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                        if (self.rank==0):
                            f1['data/p_diffusion'][:,:,:,:] = p_diffusion.T
                    self.comm.Barrier()
                    
                    # === write unsteady data
                    if save_unsteady:
                        with rgd(fn_rgd_turb_budget, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                            dset = f1['data/unsteady_p_diffusion']
                            self.comm.Barrier()
                            t_start = timeit.default_timer()
                            with dset.collective:
                                dset[rt1:rt2,:,:,:] = unsteady_p_diffusion.T
                            t_delta = timeit.default_timer() - t_start
                            data_gb = 4*nx*ny*nz*nt/1024**3
                            if verbose:
                                even_print('write: unsteady p diffusion', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
                    
                    # === release mem
                    A        = None; del A
                    A_ddx    = None; del A_ddx
                    A_ddy    = None; del A_ddy
                    A_ddz    = None; del A_ddz
                    unsteady_p_diffusion = None; del unsteady_p_diffusion
                    
                    self.comm.Barrier()
                    mem_avail_gb = psutil.virtual_memory().available/1024**3
                    mem_free_gb  = psutil.virtual_memory().free/1024**3
                    if verbose:
                        tqdm.write(even_print('mem available', '%0.1f [GB]'%mem_avail_gb, s=True))
                        tqdm.write(even_print('mem free',      '%0.1f [GB]'%mem_free_gb,  s=True))
                
                # === pressure dilatation
                if True:
                    
                    if verbose: print(72*'-')
                    t_start = timeit.default_timer()
                    
                    # A = np.einsum('xyzt,xyzti->xyzti', pI, uII_i)
                    # A_ddx = np.gradient(A[:,:,:,:,0], x, axis=0, edge_order=1)
                    # A_ddy = np.gradient(A[:,:,:,:,1], y, axis=1, edge_order=1)
                    # A_ddz = np.gradient(A[:,:,:,:,2], z, axis=2, edge_order=1)
                    # unsteady_p_dilatation = A_ddx + A_ddy + A_ddz
                    
                    unsteady_p_dilatation = pI * ( duIIdx + dvIIdy + dwIIdz )
                    
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('p_dilatation', format_time_string(t_delta),s=True))
                    
                    # === do avg
                    t_start = timeit.default_timer()
                    unsteady_p_dilatation_sum   = np.zeros((nx,ny,nz,1), dtype=np.float64, order='C')
                    unsteady_p_dilatation_sum_i = np.sum(unsteady_p_dilatation, axis=3, keepdims=True, dtype=np.float64)
                    self.comm.Reduce([unsteady_p_dilatation_sum_i, MPI.DOUBLE], [unsteady_p_dilatation_sum, MPI.DOUBLE], op=MPI.SUM, root=0)
                    if (self.rank==0):
                        p_dilatation = ( (1/nt)*unsteady_p_dilatation_sum ).astype(np.float32)
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('t avg p_dilatation', format_time_string(t_delta),s=True))
                    
                    # === write avg
                    with rgd(fn_rgd_turb_budget_mean, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                        if (self.rank==0):
                            f1['data/p_dilatation'][:,:,:,:] = p_dilatation.T
                    self.comm.Barrier()
                    
                    # === write unsteady data
                    if save_unsteady:
                        with rgd(fn_rgd_turb_budget, 'a', driver='mpio', comm=self.comm, libver='latest') as f1:
                            dset = f1['data/unsteady_p_dilatation']
                            self.comm.Barrier()
                            t_start = timeit.default_timer()
                            with dset.collective:
                                dset[rt1:rt2,:,:,:] = unsteady_p_dilatation.T
                            t_delta = timeit.default_timer() - t_start
                            data_gb = 4*nx*ny*nz*nt/1024**3
                            if verbose:
                                even_print('write: unsteady p dilatation', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
                    
                    # === release mem
                    unsteady_p_dilatation = None; del unsteady_p_dilatation
                    
                    self.comm.Barrier()
                    mem_avail_gb = psutil.virtual_memory().available/1024**3
                    mem_free_gb  = psutil.virtual_memory().free/1024**3
                    if verbose:
                        tqdm.write(even_print('mem available', '%0.1f [GB]'%mem_avail_gb, s=True))
                        tqdm.write(even_print('mem free',      '%0.1f [GB]'%mem_free_gb,  s=True))
        
        if verbose: print(72*'-')
        
        # === make .xdmf
        
        if save_unsteady:
            with rgd(fn_rgd_turb_budget, 'r', driver='mpio', comm=self.comm, libver='latest') as f1:
                f1.make_xdmf()
        with rgd(fn_rgd_turb_budget_mean, 'r', driver='mpio', comm=self.comm, libver='latest') as f1:
            f1.make_xdmf()
        
        # ===
        
        self.comm.Barrier()
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.calc_turb_budget() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    # === Eulerian to Lagrangian transform --> 'converts' RGD to LPD
    
    def time_integrate(self, **kwargs):
        '''
        do Lagrangian-frame time integration of [u,v,w] field
        -----
        --> output LPD (Lagrangian Particle Data) file / lpd() instance
        '''
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.time_integrate()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        force    = kwargs.get('force',False)
        chunk_kb = kwargs.get('chunk_kb',1024)
        fn_lpd   = kwargs.get('fn_lpd','pts.h5')
        scheme   = kwargs.get('scheme','RK4')
        npts     = kwargs.get('npts',1e4)
        
        #rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        #rz = kwargs.get('rz',1)
        #rt = kwargs.get('rt',1)
        
        #if (rx*ry*rz != self.n_ranks):
        #    raise AssertionError('rx*ry*rz != self.n_ranks')
        if (ry != self.n_ranks):
            raise AssertionError('ry != self.n_ranks')
        #if (rx>self.nx):
        #    raise AssertionError('rx>self.nx')
        #if (ry>self.ny):
        #    raise AssertionError('ry>self.ny')
        #if (rz>self.nz):
        #    raise AssertionError('rz>self.nz')
        
        #comm4d = self.comm.Create_cart(dims=[rx,ry,rz], periods=[False,False,False], reorder=False)
        #t4d = comm4d.Get_coords(self.rank)
        
        # === the 'standard' non-abutting / non-overlapping index split
        
        #rxl_ = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
        ryl_ = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
        #rzl_ = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
        #rtl_ = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))
        
        #rxl = [[b[0],b[-1]+1] for b in rxl_ ]
        ryl = [[b[0],b[-1]+1] for b in ryl_ ]
        #rzl = [[b[0],b[-1]+1] for b in rzl_ ]
        #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
        
        #rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
        #ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
        #rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
        #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        
        ry1,ry2 = ryl[self.rank]; nyr = ry2 - ry1
        
        # === here we actually want to overlap points i.e. share abutting pts
        
        if (self.rank!=(self.n_ranks-1)):
            ry2 += 1
            nyr += 1
        
        ## these should be abutting
        y_max = self.y[ry1:ry2].max()
        y_min = self.y[ry1:ry2].min()
        
        ## overlap in middle ranks
        if (self.rank==0):
            ry1 -= 0
            ry2 += 6
        elif (self.rank==self.n_ranks-1):
            ry1 -= 6
            ry2 += 0
        else:
            ry1 -= 6
            ry2 += 6
        
        nyr = ry2 - ry1
        
        ## these should be overlapping / intersecting
        y_max_ov = self.y[ry1:ry2].max()
        y_min_ov = self.y[ry1:ry2].min()
        
        ## check rank / grid distribution
        if False:
            for ri in range(self.n_ranks):
                self.comm.Barrier()
                if (self.rank == ri):
                    print('rank %04d : ry1=%i ry2=%i y_min=%0.8f y_max=%0.8f y_min_ov=%0.8f y_max_ov=%0.8f'%(self.rank,ry1,ry2,y_min,y_max,y_min_ov,y_max_ov))
                    sys.stdout.flush()
            self.comm.Barrier()
        
        t_read = 0.
        t_write = 0.
        data_gb_read = 0.
        data_gb_write = 0.
        
        # === info about the domain
        
        fn_dat_mean_dim = None
        
        if (fn_dat_mean_dim is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_dat_mean_base = fname_root+'_mean_dim.dat'
            fn_dat_mean_dim = str(PurePosixPath(fname_path, fname_dat_mean_base))
        
        if not os.path.isfile(fn_dat_mean_dim):
            raise FileNotFoundError('%s not found!'%fn_dat_mean_dim)
        
        if verbose: even_print('fn_rgd'             , self.fname      )
        if verbose: even_print('fn_dat_mean_dim'    , fn_dat_mean_dim )
        if verbose: even_print('fn_lpd'             , fn_lpd          )
        if verbose: print(72*'-')
        
        if True: ## mean dimensional data [x,z]
            
            # === read in data (mean dim) --> every rank gets full [x,z]
            with open(fn_dat_mean_dim,'rb') as f:
                data_mean_dim = pickle.load(f)
            fmd = type('foo', (object,), data_mean_dim)
            
            # === 2D dimensional quantities --> [x,z]
            u_tau    = fmd.u_tau
            nu_wall  = fmd.nu_wall
            d99      = fmd.d99
            u99      = fmd.u99
            Re_tau   = fmd.Re_tau
            Re_theta = fmd.Re_theta
            
            u_tau_avg_end    = np.mean(fmd.u_tau[-1,:]    , axis=(0,))
            nu_wall_avg_end  = np.mean(fmd.nu_wall[-1,:]  , axis=(0,))
            d99_avg_end      = np.mean(fmd.d99[-1,:]      , axis=(0,))
            u99_avg_end      = np.mean(fmd.u99[-1,:]      , axis=(0,))
            Re_tau_avg_end   = np.mean(fmd.Re_tau[-1,:]   , axis=(0,))
            Re_theta_avg_end = np.mean(fmd.Re_theta[-1,:] , axis=(0,))
            
            u_tau_avg_begin    = np.mean(fmd.u_tau[0,:]    , axis=(0,))
            nu_wall_avg_begin  = np.mean(fmd.nu_wall[0,:]  , axis=(0,))
            d99_avg_begin      = np.mean(fmd.d99[0,:]      , axis=(0,))
            u99_avg_begin      = np.mean(fmd.u99[0,:]      , axis=(0,))
            Re_tau_avg_begin   = np.mean(fmd.Re_tau[0,:]   , axis=(0,))
            Re_theta_avg_begin = np.mean(fmd.Re_theta[0,:] , axis=(0,))
            
            # === 2D inner scales --> [x,z]
            sc_l_in = nu_wall / u_tau
            sc_u_in = u_tau
            sc_t_in = nu_wall / u_tau**2
            
            # === 2D outer scales --> [x,z]
            sc_l_out = d99
            sc_u_out = u99
            sc_t_out = d99/u99
            
            # === check
            np.testing.assert_allclose(fmd.lchar   , self.lchar   , rtol=1e-8)
            np.testing.assert_allclose(fmd.U_inf   , self.U_inf   , rtol=1e-8)
            np.testing.assert_allclose(fmd.rho_inf , self.rho_inf , rtol=1e-8)
            np.testing.assert_allclose(fmd.T_inf   , self.T_inf   , rtol=1e-8)
            np.testing.assert_allclose(fmd.nx      , self.nx      , rtol=1e-8)
            np.testing.assert_allclose(fmd.ny      , self.ny      , rtol=1e-8)
            np.testing.assert_allclose(fmd.nz      , self.nz      , rtol=1e-8)
            np.testing.assert_allclose(fmd.xs      , self.x       , rtol=1e-8)
            np.testing.assert_allclose(fmd.ys      , self.y       , rtol=1e-8)
            np.testing.assert_allclose(fmd.zs      , self.z       , rtol=1e-8)
            
            lchar   = self.lchar
            U_inf   = self.U_inf
            rho_inf = self.rho_inf
            T_inf   = self.T_inf
            
            nx = self.nx
            ny = self.ny
            nz = self.nz
            nt = self.nt
            
            ## dimless (inlet)
            xd = self.x
            yd = self.y
            zd = self.z
            td = self.t
            
            ## dimensional [m] / [s]
            x = self.x * lchar 
            y = self.y * lchar
            z = self.z * lchar
            t = self.t * (lchar/U_inf)
            
            t_meas = t[-1]-t[0]
            dt     = self.dt * (lchar/U_inf)
            
            np.testing.assert_equal(nx,x.size)
            np.testing.assert_equal(ny,y.size)
            np.testing.assert_equal(nz,z.size)
            np.testing.assert_equal(nt,t.size)
            np.testing.assert_allclose(dt, t[1]-t[0], rtol=1e-8)
            
            # === report
            if verbose:
                even_print('nx'     , '%i'        %nx     )
                even_print('ny'     , '%i'        %ny     )
                even_print('nz'     , '%i'        %nz     )
                even_print('nt'     , '%i'        %nt     )
                even_print('dt'     , '%0.5e [s]' %dt     )
                even_print('t_meas' , '%0.5e [s]' %t_meas )
                even_print('domain x' , '%0.5e [m]' % (x.max()-x.min()) )
                ##
                even_print('U_inf'  , '%0.3f [m/s]'  % U_inf        )
                even_print('U_inf·t_meas/(domain x)' , '%0.3f' % (U_inf*t_meas/(x.max()-x.min())) )
                print(72*'-')
            
            if verbose:
                print('begin x'+'\n'+'-------')
                even_print('Re_τ'   , '%0.1f'        % Re_tau_avg_begin   )
                even_print('Re_θ'   , '%0.1f'        % Re_theta_avg_begin )
                even_print('δ99'    , '%0.5e [m]'    % d99_avg_begin      )
                even_print('u_τ'    , '%0.3f [m/s]'  % u_tau_avg_begin    )
                even_print('ν_wall' , '%0.5e [m²/s]' % nu_wall_avg_begin  )
                even_print('dt+'    , '%0.4f'        % (dt/(nu_wall_avg_begin / u_tau_avg_begin**2)) )
                
                t_eddy_begin = t_meas / (d99_avg_begin/u_tau_avg_begin)
                
                even_print('t_meas/(δ99/u_τ) = t_eddy' , '%0.2f'%t_eddy_begin)
                even_print('t_meas/(δ99/u99)'          , '%0.2f'%(t_meas/(d99_avg_begin/u99_avg_begin)))
                even_print('t_meas/(20·δ99/u99)'       , '%0.2f'%(t_meas/(20*d99_avg_begin/u99_avg_begin)))
                ##
                print('end x'+'\n'+'-----')
                even_print('Re_τ'   , '%0.1f'        % Re_tau_avg_end   )
                even_print('Re_θ'   , '%0.1f'        % Re_theta_avg_end )
                even_print('δ99'    , '%0.5e [m]'    % d99_avg_end      )
                even_print('u_τ'    , '%0.3f [m/s]'  % u_tau_avg_end    )
                even_print('ν_wall' , '%0.5e [m²/s]' % nu_wall_avg_end  )
                even_print('dt+'    , '%0.4f'        % (dt/(nu_wall_avg_end / u_tau_avg_end**2)) )
                
                t_eddy_end = t_meas / (d99_avg_end/u_tau_avg_end)
                
                even_print('t_meas/(δ99/u_τ) = t_eddy' , '%0.2f'%t_eddy_end)
                even_print('t_meas/(δ99/u99)'          , '%0.2f'%(t_meas/(d99_avg_end/u99_avg_end)))
                even_print('t_meas/(20·δ99/u99)'       , '%0.2f'%(t_meas/(20*d99_avg_end/u99_avg_end)))
                print(72*'-')
        
        # === make initial particle lists
        
        tc  = np.copy(self.t) ## keep convection time = RGD time
        ntc = tc.size
        dtc = tc[1]-tc[0]
        
        if False: ## initial points @ grid points
            
            xc = np.copy(self.x)[::5]
            yc = np.copy(self.y[ry1:ry2]) ## [::3]
            zc = np.copy(self.z)[::5]
            
            xxxp, yyyp, zzzp = np.meshgrid(xc, yc, zc, indexing='ij')
            xp = xxxp.ravel(order='F')
            yp = yyyp.ravel(order='F')
            zp = zzzp.ravel(order='F')
            npts = xp.shape[0] ## this rank
            
            ## 4D
            tp = tc[0]*np.ones((npts,), dtype=xp.dtype)
            xyztp = np.stack((xp,yp,zp,tp)).T
            
            ## ## 3D
            ## xyzp = np.stack((xp,yp,zp)).T
            
            ## get total number of particles all ranks
            G = self.comm.gather([npts, self.rank], root=0)
            G = self.comm.bcast(G, root=0)
            npts_total = sum([x[0] for x in G])
            
            ## get the particle ID array for this rank (pnum)
            rp1  = sum([ G[i][0] for i in range(len(G)) if (i<self.rank) ])
            rp2  = rp1 + npts
            pnum = np.arange(rp1,rp2, dtype=np.int64)
            
            ## check particles per rank
            if False:
                for ri in range(self.n_ranks):
                    self.comm.Barrier()
                    if (self.rank == ri):
                        print('rank %04d : %s'%(self.rank,str(pnum)))
                self.comm.Barrier()
            
            if verbose: even_print('n pts','%i'%npts)
            if verbose: even_print('n pts (total)','%i'%npts_total)
            
            npts_total_orig = npts_total
        
        else: ## initial points @ random locations
            
            ## random number generator
            rng = np.random.default_rng(seed=1)
            
            ## pts already in domain at t=0
            if True:
                
                npts_init     = int(round(npts))
                volume_domain = (self.x.max()-self.x.min()) * (self.y.max()-self.y.min()) * (self.z.max()-self.z.min())
                pts_density   = npts_init / volume_domain ## [n pts / m^3]
                area_inlet    = (self.y.max()-self.y.min()) * (self.z.max()-self.z.min())
                volume_flow   = area_inlet * 1 ## still dimless, so U_inf=1 ## [m^2]*[m/s] = [m^3 / s]
                fac           = (10372 - ((5390083-5000000)/3288))/10372 ## approximates U(y)dy integral
                volume_flow  *= fac ## since U(y)!=1 across BL
                volume_per_dt = dtc * volume_flow ## [m^3]
                
                pts_per_dt_inlet          = int(round(pts_density*volume_per_dt))
                pts_per_dt_inlet_arr      = pts_per_dt_inlet * np.ones((ntc-1,), dtype=np.int32)
                pts_per_dt_inlet_arr[-5:] = 0 ## dont add any pts in the last N ts
                
                npts_init -= pts_per_dt_inlet ## added back at beginning of loop
                
                if verbose: even_print('volume_domain','%0.5e'%volume_domain)
                if verbose: even_print('pts_density','%0.5e'%pts_density)
                if verbose: even_print('area_inlet','%0.5e'%area_inlet)
                if verbose: even_print('pts_per_dt_inlet','%i'%pts_per_dt_inlet)
                if verbose: print(72*'-')
                
                xp = rng.uniform(self.x.min(), self.x.max(), size=(npts_init,))
                yp = rng.uniform(self.y.min(), self.y.max(), size=(npts_init,))
                zp = rng.uniform(self.z.min(), self.z.max(), size=(npts_init,))
                tp = tc[0]*np.ones((npts_init,), dtype=xp.dtype)
                xyztp = np.stack((xp,yp,zp,tp)).T
                pnum  = np.arange(npts_init, dtype=np.int64)
                
                offset = npts_init
                
                if (self.rank==0):
                    ii = np.where(xyztp[:,1]<=y_max)
                elif (self.rank==self.n_ranks-1):
                    ii = np.where(xyztp[:,1]>y_min)
                else:
                    ii = np.where((xyztp[:,1]>y_min) & (xyztp[:,1]<=y_max))
                
                xyztp = np.copy(xyztp[ii])
                pnum  = np.copy(pnum[ii])
                
                npts_total = npts_init
            
            else: ## no initial particles
                
                xyztp      = None
                pnum       = None
                offset     = 0
                npts_total = 0
                pts_per_dt_inlet = int(10e3)
            
            ## function to replenish pts
            def pts_initializer(rng, npts, tt, offset):
                '''
                the 'new particle' replenishment func
                '''
                zp = rng.uniform(self.z.min(), self.z.max(), size=(npts,))
                yp = rng.uniform(self.y.min(), self.y.max(), size=(npts,))
                xp = 0.5*(self.x[0]+self.x[1]) * np.ones((npts,), dtype=zp.dtype)
                tp = tt * np.ones((npts,), dtype=zp.dtype)
                ##
                xyztp  = np.stack((xp,yp,zp,tp)).T
                pnum = int(offset) + np.arange(npts, dtype=np.int64)
                ##
                return xyztp, pnum
        
        ## get the total number of points that will exist for all times
        npts_all_ts = npts_total
        for tci in range(ntc-1):
            xyztp_, pnum_ = pts_initializer(rng, pts_per_dt_inlet_arr[tci], tc[tci], 0)
            npts,_ = xyztp_.shape
            npts_all_ts += npts
        
        # === check if file exists / delete / touch / chmod
        
        ## self.comm.Barrier()
        ## if (self.rank==0):
        ##     if os.path.isfile(fn_lpd):
        ##         os.remove(fn_lpd)
        ##     Path(fn_lpd).touch()
        ##     os.chmod(fn_lpd, int('770', base=8))
        ##     if shutil.which('lfs') is not None:
        ##         return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 2M %s > /dev/null 2>&1'%('particles.h5',), shell=True)
        ## self.comm.Barrier()
        
        # ===
        
        pcomm = MPI.COMM_WORLD
        with lpd(fn_lpd, 'w', force=force, driver='mpio', comm=pcomm, libver='latest') as hf_lpd:
            
            ## hf_lpd.atomic = True
            
            ## 'self' passed here is RGD / h5py.File instance
            ## this copies over all the header info from the RGD: U_inf, lchar, etc
            hf_lpd.init_from_rgd(self, t_info=False)
            
            ## shape & HDF5 chunk scheme for datasets
            shape = (npts_all_ts, ntc-1)
            chunks = rgd.chunk_sizer(nxi=shape, constraint=(None,1), size_kb=chunk_kb, base=2)
            
            scalars = ['x','y','z', 'u','v','w', 't','id']
            
            for scalar in scalars:
                
                if verbose:
                    even_print('initializing',scalar)
                
                if ('data/%s'%scalar in hf_lpd):
                    del hf_lpd['data/%s'%scalar]
                dset = hf_lpd.create_dataset('data/%s'%scalar, 
                                          shape=shape, 
                                          dtype=np.float32,
                                          fillvalue=np.nan,
                                          #compression='gzip', ## this causes segfaults :( :(
                                          #compression_opts=5,
                                          #shuffle=True,
                                          chunks=chunks,
                                          )
                
                chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                if verbose:
                    even_print('chunk shape (pts,nt)','%s'%str(dset.chunks))
                    even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            ## write time vector
            if ('dims/t' in hf_lpd):
                del hf_lpd['dims/t']
            hf_lpd.create_dataset('dims/t',
                                data=tc[:-1],
                                dtype=np.float32,
                                chunks=True)
            
            pcomm.Barrier()
            self.comm.Barrier()
            
            if True: ## convect fwd
                
                if verbose:
                    progress_bar = tqdm(total=ntc-1, ncols=100, desc='convect fwd', leave=False, file=sys.stdout)
                
                t_write = 0.
                
                if (dtc!=self.dt):
                    raise AssertionError('dtc!=self.dt')
                
                ## the global list of all particle IDs that have left volume
                pnum_nan_global = np.array([],dtype=np.int64)
                
                for tci in range(ntc-1):
                    
                    if verbose: tqdm.write('---')
                    
                    ## get global new pts this ts
                    xyztp_new, pnum_new = pts_initializer(rng, pts_per_dt_inlet_arr[tci], tc[tci], offset)
                    npts_new, _ = xyztp_new.shape
                    offset += npts_new
                    
                    ## take pts in rank bounds
                    if (self.rank==0):
                        ii = np.where(xyztp_new[:,1]<=y_max)
                    elif (self.rank==self.n_ranks-1):
                        ii = np.where(xyztp_new[:,1]>y_min)
                    else:
                        ii = np.where((xyztp_new[:,1]>y_min) & (xyztp_new[:,1]<=y_max))
                    ##
                    xyztp = np.concatenate((xyztp, xyztp_new[ii]), axis=0)
                    pnum  = np.concatenate((pnum,  pnum_new[ii]),  axis=0)
                    
                    if verbose: tqdm.write(even_print('tci', '%i'%(tci,), s=True))
                    
                    # ===
                    
                    ti1  = tci
                    ti2  = tci+1+1
                    tseg = tc[ti1:ti2]
                    
                    ## RK4 times
                    tbegin = tseg[0]
                    tend   = tseg[-1]
                    tmid   = 0.5*(tbegin+tend)
                    
                    ## update pts list time
                    xyztp[:,3] = tc[tci]
                    
                    ## assert : tc[tci] == segment begin time
                    if not np.allclose(tc[tci], tbegin, rtol=1e-8):
                        raise AssertionError('tc[tci] != tbegin')
                    
                    # === read RGD rectilinear data
                    
                    scalars = ['u','v','w']
                    scalars_dtypes = [np.float32 for s in scalars]
                    data = np.zeros(shape=(self.nx, nyr, self.nz, 2), dtype={'names':scalars, 'formats':scalars_dtypes})
                    
                    for scalar in scalars:
                        
                        data_gb = 4*self.nx*self.ny*self.nz*2 / 1024**3
                        
                        dset = self['data/%s'%scalar]
                        self.comm.Barrier()
                        t_start = timeit.default_timer()
                        with dset.collective:
                            data[scalar] = dset[ti1:ti2,:,ry1:ry2,:].T
                        
                        self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        
                        if verbose:
                            tqdm.write(even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)) 
                        pass
                    
                    # ===
                    
                    ## ## make 4D scalar interpolators
                    ## f_u = sp.interpolate.RegularGridInterpolator((self.x,self.y[ry1:ry2],self.z,self.t[ti1:ti2]), data['u'], method='linear', fill_value=np.nan, bounds_error=False)
                    ## f_v = sp.interpolate.RegularGridInterpolator((self.x,self.y[ry1:ry2],self.z,self.t[ti1:ti2]), data['v'], method='linear', fill_value=np.nan, bounds_error=False)
                    ## f_w = sp.interpolate.RegularGridInterpolator((self.x,self.y[ry1:ry2],self.z,self.t[ti1:ti2]), data['w'], method='linear', fill_value=np.nan, bounds_error=False)
                    
                    uvw   = np.stack((data['u'], data['v'], data['w']), axis=4)
                    f_uvw = sp.interpolate.RegularGridInterpolator((self.x,self.y[ry1:ry2],self.z,self.t[ti1:ti2]), uvw, method='linear', fill_value=np.nan, bounds_error=False)
                    
                    # === Trilinear interpolation at beginning of time segment (1/2) --> this may have NaNs
                    
                    xyztp_k1 = np.copy(xyztp)
                    ## u_k1     = f_u(xyztp_k1)
                    ## v_k1     = f_v(xyztp_k1)
                    ## w_k1     = f_w(xyztp_k1)
                    
                    uvw_k1 = f_uvw(xyztp_k1)
                    u_k1 = uvw_k1[:,0]
                    
                    x_ = np.copy( xyztp[:,0] )
                    y_ = np.copy( xyztp[:,1] )
                    z_ = np.copy( xyztp[:,2] )
                    t_ = np.copy( xyztp[:,3] )
                    
                    # === separate out NaN positions / velocities
                    
                    ii_nan    = np.where(  np.isnan(x_) |  np.isnan(u_k1) )
                    ii_notnan = np.where( ~np.isnan(x_) & ~np.isnan(u_k1) )
                    ##
                    pnum_nan  = np.copy(pnum[ii_nan])
                    xyztp_nan = np.copy(xyztp[ii_nan])
                    ##
                    G = self.comm.gather([np.copy(pnum_nan), self.rank], root=0)
                    G = self.comm.bcast(G, root=0)
                    pnum_nan_global_this_ts = np.concatenate( [g[0] for g in G] )
                    pnum_nan_global         = np.concatenate( (pnum_nan_global_this_ts , pnum_nan_global) )
                    ##
                    npts_nan = pnum_nan_global.shape[0]
                    
                    ## take only non-NaN position particles
                    pnum  = np.copy(pnum[ii_notnan])
                    xyztp = np.copy(xyztp[ii_notnan])
                    
                    if True: ## check global pnum (pt id) vector
                        
                        G = self.comm.gather([np.copy(pnum), self.rank], root=0)
                        G = self.comm.bcast(G, root=0)
                        pnum_global = np.sort( np.concatenate( [g[0] for g in G] ) )
                        pnum_global = np.sort( np.concatenate((pnum_global,pnum_nan_global)) )
                        
                        ## make sure that the union of the current (non-nan) IDs and nan IDs is 
                        ##    equal to the arange of the total number of particles instantiated to
                        ##    this point
                        if not np.array_equal(pnum_global, np.arange(offset, dtype=np.int64)):
                            raise AssertionError('pnum_global!=np.arange(offset, dtype=np.int64)')
                        
                        if verbose: tqdm.write(even_print('pnum check', 'passed', s=True)) 
                    
                    # === Trilinear interpolation at beginning of time segment (2/2) --> again after NaN filter
                    
                    xyztp_k1 = np.copy(xyztp)
                    ## u_k1     = f_u(xyztp_k1)
                    ## v_k1     = f_v(xyztp_k1)
                    ## w_k1     = f_w(xyztp_k1)
                    uvw_k1   = f_uvw(xyztp_k1)
                    u_k1     = uvw_k1[:,0]
                    v_k1     = uvw_k1[:,1]
                    w_k1     = uvw_k1[:,2]
                    
                    x_ = np.copy( xyztp[:,0] )
                    y_ = np.copy( xyztp[:,1] )
                    z_ = np.copy( xyztp[:,2] )
                    t_ = np.copy( xyztp[:,3] )
                    
                    ## this passes
                    if False:
                        if np.isnan(np.min(u_k1)):
                            print('u_k1 has NaNs')
                            pcomm.Abort(1)
                        if np.isnan(np.min(v_k1)):
                            print('v_k1 has NaNs')
                            pcomm.Abort(1)
                        if np.isnan(np.min(w_k1)):
                            print('w_k1 has NaNs')
                            pcomm.Abort(1)
                    
                    # === Gather/Bcast all data
                    
                    pcomm.Barrier()
                    t_start = timeit.default_timer()
                    
                    G = self.comm.gather([self.rank, pnum, x_, y_, z_, t_, u_k1, v_k1, w_k1], root=0)
                    G = self.comm.bcast(G, root=0)
                    
                    pcomm.Barrier()
                    t_delta = timeit.default_timer() - t_start
                    if verbose:
                        tqdm.write(even_print('MPI Gather/Bcast', '%0.2f [s]'%(t_delta,), s=True))
                    
                    npts_total = sum([g[1].shape[0] for g in G])
                    pnum_gl    = np.concatenate([g[1] for g in G], axis=0)
                    x_gl       = np.concatenate([g[2] for g in G], axis=0)
                    y_gl       = np.concatenate([g[3] for g in G], axis=0)
                    z_gl       = np.concatenate([g[4] for g in G], axis=0)
                    t_gl       = np.concatenate([g[5] for g in G], axis=0)
                    u_gl       = np.concatenate([g[6] for g in G], axis=0)
                    v_gl       = np.concatenate([g[7] for g in G], axis=0)
                    w_gl       = np.concatenate([g[8] for g in G], axis=0)
                    ##
                    if verbose: tqdm.write(even_print('n pts initialized', '%i'%(offset,),      s=True))
                    if verbose: tqdm.write(even_print('n pts in domain',   '%i'%(npts_total,),  s=True))
                    if verbose: tqdm.write(even_print('n pts left domain', '%i'%(npts_nan,),    s=True))
                    if verbose: tqdm.write(even_print('n pts all time',    '%i'%(npts_all_ts,), s=True))
                    
                    # === add NaN IDs, pad scalar vectors, do sort
                    
                    pnum_gl = np.concatenate([pnum_gl,pnum_nan_global], axis=0)
                    npts_total_incl_nan = pnum_gl.shape[0]
                    
                    if (npts_total_incl_nan!=offset):
                        raise AssertionError('npts_total_incl_nan!=offset')
                    
                    aa = np.empty((npts_nan,), dtype=np.float32)
                    aa[:] = np.nan
                    x_gl = np.concatenate([x_gl,aa], axis=0)
                    y_gl = np.concatenate([y_gl,aa], axis=0)
                    z_gl = np.concatenate([z_gl,aa], axis=0)
                    t_gl = np.concatenate([t_gl,aa], axis=0)
                    u_gl = np.concatenate([u_gl,aa], axis=0)
                    v_gl = np.concatenate([v_gl,aa], axis=0)
                    w_gl = np.concatenate([w_gl,aa], axis=0)
                    ##
                    sort_order = np.argsort(pnum_gl, axis=0)
                    pnum_gl    = np.copy(pnum_gl[sort_order])
                    x_gl       = np.copy(x_gl[sort_order])
                    y_gl       = np.copy(y_gl[sort_order])
                    z_gl       = np.copy(z_gl[sort_order])
                    t_gl       = np.copy(t_gl[sort_order])
                    u_gl       = np.copy(u_gl[sort_order])
                    v_gl       = np.copy(v_gl[sort_order])
                    w_gl       = np.copy(w_gl[sort_order])
                    
                    ## yet another check
                    if not np.array_equal(pnum_gl, np.arange(offset, dtype=np.int64)):
                        raise AssertionError('pnum_gl!=np.arange(offset, dtype=np.int64)')
                    
                    # === get collective write bounds
                    
                    rpl_ = np.array_split(np.arange(npts_total_incl_nan,dtype=np.int64) , self.n_ranks)
                    rpl = [[b[0],b[-1]+1] for b in rpl_ ]
                    rp1,rp2 = rpl[self.rank]
                    
                    # === write
                    
                    #data_gb = 4 * npts_total / 1024**3
                    data_gb = 4 * npts_total_incl_nan / 1024**3
                    
                    for key, value in {'id':pnum_gl, 'x':x_gl, 'y':y_gl, 'z':z_gl, 't':t_gl, 'u':u_gl, 'v':v_gl, 'w':w_gl}.items():
                        
                        dset = hf_lpd['data/%s'%key]
                        pcomm.Barrier()
                        t_start = timeit.default_timer()
                        with dset.collective:
                            #dset[0:npts_total_incl_nan,tci] = value.astype(np.float32)
                            dset[rp1:rp2,tci] = value[rp1:rp2].astype(np.float32)
                        pcomm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        t_write += t_delta
                        if verbose:
                            tqdm.write(even_print('write: %s'%key, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)) 
                    
                    # ===
                    
                    ## ## write the IDs of the particles which have left the domain for sort in a second step
                    ## ## --> this is now done in the step above
                    ## if (npts_nan>0):
                    ##     dset = hf_lpd['data/id']
                    ##     with dset.collective:
                    ##         dset[-npts_nan:,tci] = pnum_nan_global.astype(np.float32)
                    
                    # === Time Integration
                    
                    pcomm.Barrier()
                    t_start = timeit.default_timer()
                    
                    if (scheme=='RK4'):
                        
                        xyztp_    = np.copy(xyztp)
                        xyztp_k2  = np.zeros_like(xyztp)
                        xyztp_k3  = np.zeros_like(xyztp)
                        xyztp_k4  = np.zeros_like(xyztp)
                        
                        xyztp_k2[:,0] = xyztp_[:,0] + u_k1*dtc*0.5
                        xyztp_k2[:,1] = xyztp_[:,1] + v_k1*dtc*0.5
                        xyztp_k2[:,2] = xyztp_[:,2] + w_k1*dtc*0.5
                        xyztp_k2[:,3] = tmid
                        ## u_k2          = f_u(xyztp_k2)
                        ## v_k2          = f_v(xyztp_k2)
                        ## w_k2          = f_w(xyztp_k2)
                        uvw_k2 = f_uvw(xyztp_k2)
                        u_k2   = uvw_k2[:,0]
                        v_k2   = uvw_k2[:,1]
                        w_k2   = uvw_k2[:,2]
                        
                        xyztp_k3[:,0] = xyztp_[:,0] + u_k2*dtc*0.5
                        xyztp_k3[:,1] = xyztp_[:,1] + v_k2*dtc*0.5
                        xyztp_k3[:,2] = xyztp_[:,2] + w_k2*dtc*0.5
                        xyztp_k3[:,3] = tmid
                        ## u_k3          = f_u(xyztp_k3)
                        ## v_k3          = f_v(xyztp_k3)
                        ## w_k3          = f_w(xyztp_k3)
                        uvw_k3 = f_uvw(xyztp_k3)
                        u_k3   = uvw_k3[:,0]
                        v_k3   = uvw_k3[:,1]
                        w_k3   = uvw_k3[:,2]
                        
                        xyztp_k4[:,0] = xyztp_[:,0] + u_k3*dtc*1.0
                        xyztp_k4[:,1] = xyztp_[:,1] + v_k3*dtc*1.0
                        xyztp_k4[:,2] = xyztp_[:,2] + w_k3*dtc*1.0
                        xyztp_k4[:,3] = tend
                        ## u_k4          = f_u(xyztp_k4)
                        ## v_k4          = f_v(xyztp_k4)
                        ## w_k4          = f_w(xyztp_k4)
                        uvw_k4 = f_uvw(xyztp_k4)
                        u_k4   = uvw_k4[:,0]
                        v_k4   = uvw_k4[:,1]
                        w_k4   = uvw_k4[:,2]
                        
                        ## the vel components (RK4 weighted avg) for time integration
                        u = (1./6.) * (1.*u_k1 + 2.*u_k2 + 2.*u_k3 + 1.*u_k4)
                        v = (1./6.) * (1.*v_k1 + 2.*v_k2 + 2.*v_k3 + 1.*v_k4)
                        w = (1./6.) * (1.*w_k1 + 2.*w_k2 + 2.*w_k3 + 1.*w_k4)
                    
                    elif (scheme=='Euler Explicit'):
                        
                        u = u_k1
                        v = v_k1
                        w = w_k1
                    
                    else:
                        raise NotImplementedError('integration scheme \'%s\' not yet implemented'%scheme)
                    
                    ## actual time integration with higher order 'average' [u,v,w] over time segment 
                    xyztp[:,0] = xyztp[:,0] + u*dtc
                    xyztp[:,1] = xyztp[:,1] + v*dtc
                    xyztp[:,2] = xyztp[:,2] + w*dtc
                    
                    pcomm.Barrier()
                    t_delta = timeit.default_timer() - t_start
                    t_write += t_delta
                    if verbose:
                        tqdm.write(even_print('%s'%scheme, '%0.2f [s]'%(t_delta,), s=True))
                    
                    # === MPI Gather/Bcast points between domains
                    
                    ## get pts that leave rank bounds in dim [y]
                    i_exit_top = np.where(xyztp[:,1]>y_max)
                    i_exit_bot = np.where(xyztp[:,1]<y_min)
                    
                    ## lists for the pts that leave
                    xyztp_out = np.concatenate((np.copy(xyztp[i_exit_top]) , np.copy(xyztp[i_exit_bot])), axis=0)
                    pnum_out  = np.concatenate((np.copy(pnum[i_exit_top])  , np.copy(pnum[i_exit_bot])),  axis=0)
                    
                    ## delete those from local lists
                    i_del = np.concatenate((i_exit_top[0],i_exit_bot[0]))
                    xyztp = np.delete(xyztp, (i_del,), axis=0)
                    pnum  = np.delete(pnum,  (i_del,), axis=0)
                    
                    # === MPI : Gather/Bcast all inter-domain pts
                    G = self.comm.gather([np.copy(xyztp_out), np.copy(pnum_out), self.rank], root=0)
                    G = self.comm.bcast(G, root=0)
                    xyztpN = np.concatenate([x[0] for x in G], axis=0)
                    pnumN  = np.concatenate([x[1] for x in G], axis=0)
                    
                    ## get indices to 'take' from inter-domain points
                    if (self.rank==0):
                        i_take = np.where(xyztpN[:,1]<=y_max)
                    elif (self.rank==self.n_ranks-1):
                        i_take = np.where(xyztpN[:,1]>=y_min)
                    else:
                        i_take = np.where((xyztpN[:,1]<y_max) & (xyztpN[:,1]>=y_min))
                    ##
                    xyztp = np.concatenate((xyztp, xyztpN[i_take]), axis=0)
                    pnum  = np.concatenate((pnum,  pnumN[i_take]), axis=0)
                    
                    if verbose:
                        progress_bar.update()
                    
                    self.comm.Barrier()
                    pcomm.Barrier()
                
                if verbose:
                    progress_bar.close()
        
        if verbose: print(72*'-'+'\n')
        
        # === sort at each timestep by pnum (pt id)
        
        if False:
            
            if verbose: print('sort' + '\n' + 72*'-')
            
            with lpd(fn_lpd, 'a', driver='mpio', comm=pcomm, libver='latest') as hf_lpd:
                
                # hf_lpd.atomic = True
                
                rt = self.n_ranks
                #dset = hf_lpd['data/pdata']
                #npts, ntc, _ = dset.shape
                
                npts = hf_lpd.npts
                ntc  = hf_lpd.nt
                
                ## shape  = (npts,ntc)
                ## chunks = rgd.chunk_sizer(nxi=shape, constraint=(None,1), size_kb=chunk_kb, base=2)
                
                if verbose:
                    even_print('npts','%i'%npts)
                    even_print('ntc','%i'%ntc)
                
                if (rt>ntc):
                    raise AssertionError('more ranks than timesteps! rt>ntc')
                
                rtl_ = np.array_split(np.arange(ntc, dtype=np.int64), min(rt,ntc))
                rtl  = [[b[0],b[-1]+1] for b in rtl_ ]
                rt1, rt2 = rtl[self.rank]; ntr = rt2 - rt1
                
                ## read all pts in, data distributed in [t] across ranks
                
                ## numpy structured array
                pdata = np.empty(shape=(npts,ntr), dtype={'names':hf_lpd.scalars, 'formats':hf_lpd.scalars_dtypes})
                pdata[:,:] = np.nan
                
                for scalar in hf_lpd.scalars:
                    dset = hf_lpd['data/%s'%scalar]
                    pcomm.Barrier()
                    t_start = timeit.default_timer()
                    with dset.collective:
                        pdata[scalar] = dset[:,rt1:rt2]
                    pcomm.Barrier()
                    t_delta = timeit.default_timer() - t_start
                    data_gb = 4 * ntc * npts / 1024**3
                    if verbose:
                        tqdm.write(even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True))
                
                ## sort by ID at each time, overwrite
                pcomm.Barrier()
                t_start = timeit.default_timer()
                for ti in range(ntr):
                    
                    id_ = np.copy(pdata['id'][:,ti])
                    
                    if True: ## check that non-NaN ids are contiguous
                        ii_notnan = np.where( ~np.isnan(id_) )
                        id_notnan = np.copy( id_[ii_notnan] )
                        npts_notnan = id_notnan.shape[0]
                        #np.testing.assert_equal(np.sort(id_notnan), np.arange(npts_notnan, dtype=np.float32))
                        if not np.array_equal(np.sort(id_notnan), np.arange(npts_notnan, dtype=np.float32)):
                            #raise AssertionError('asdf')
                            print('[rank %02d] non-NaN particle id array does not match non-NaN scalar length'%self.rank)
                            pcomm.Abort(1)
                        
                        sort_order = np.argsort(id_, axis=0)
                        for scalar in hf_lpd.scalars:
                            dd_                 = np.copy(pdata[scalar][:,ti])
                            #pdata[scalar][:,ti] = dd_[id_.argsort()]
                            pdata[scalar][:,ti] = np.copy(dd_[sort_order])
                
                pcomm.Barrier()
                t_delta = timeit.default_timer() - t_start
                
                if verbose:
                    tqdm.write(even_print('sort', '%0.2f [s]'%(t_delta,), s=True))
                
                ## write [x,y,z,u,v,w,t,id] back out collectively
                for scalar in hf_lpd.scalars:
                    dset = hf_lpd['data/%s'%scalar]
                    pcomm.Barrier()
                    t_start = timeit.default_timer()
                    with dset.collective:
                        dset[:,rt1:rt2] = pdata[scalar]
                    pcomm.Barrier()
                    t_delta = timeit.default_timer() - t_start
                    data_gb = pdata.nbytes * self.n_ranks / 1024**3 ## approx
                    if verbose:
                        tqdm.write(even_print('write %s'%scalar, '%0.2f [s]'%(t_delta,), s=True))
            
            if verbose: print(72*'-')
        
        pcomm.Barrier()
        self.comm.Barrier()
        
        ## make XDMF/XMF2
        with lpd(fn_lpd, 'r', driver='mpio', comm=pcomm, libver='latest') as hf_lpd:
            hf_lpd.make_xdmf()
        
        # ===
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.time_integrate() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    # === Spectral Filtering
    
    def do_fft_filter(self, **kwargs):
        '''
        apply spectral filter to RGD
        '''
        pass
        return
    
    # === checks
    
    def check_wall_T(self, **kwargs):
        '''
        check T at wall for zeros (indicative of romio bug)
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.check_wall_T()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        T_wall = self['data/T'][:,:,0,:].T
        print('T_wall.shape = %s'%str(T_wall.shape))
        
        nT0 = T_wall.size - np.count_nonzero(T_wall)
        if (nT0>0):
            print('>>> WARNING : n grid points with T==0 : %i'%nT0)
        else:
            print('check passed: no grid points with T==0')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.check_wall_T() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    # === Paraview
    
    def make_xdmf(self, **kwargs):
        '''
        generate an XDMF/XMF2 from RGD for processing with Paraview
        -----
        --> https://www.xdmf.org/index.php/XDMF_Model_and_Format
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        makeVectors = kwargs.get('makeVectors',True) ## write [u,v,w] and [vort_x,vort_y,vort_z] vectors to XDMF
        makeTensors = kwargs.get('makeTensors',True) ## write stress or strain tensors to XDMF
        
        fname_path            = os.path.dirname(self.fname)
        fname_base            = os.path.basename(self.fname)
        fname_root, fname_ext = os.path.splitext(fname_base)
        fname_xdmf_base       = fname_root+'.xmf2'
        fname_xdmf            = os.path.join(fname_path, fname_xdmf_base)
        
        if verbose: print('\n'+'rgd.make_xdmf()'+'\n'+72*'-')
        
        dataset_precision_dict = {} ## holds dtype.itemsize ints i.e. 4,8
        dataset_numbertype_dict = {} ## holds string description of dtypes i.e. 'Float','Integer'
        
        # === dims
        for scalar in ['x','y','z']:
            if ('dims/'+scalar in self):
                data = self['dims/'+scalar]
                dataset_precision_dict[scalar] = data.dtype.itemsize
                if (data.dtype.name=='float32') or (data.dtype.name=='float64'):
                    dataset_numbertype_dict[scalar] = 'Float'
                elif (data.dtype.name=='int8') or (data.dtype.name=='int16') or (data.dtype.name=='int32') or (data.dtype.name=='int64'):
                    dataset_numbertype_dict[scalar] = 'Integer'
                else:
                    raise ValueError('dtype not recognized, please update script accordingly')
        
        # === scalar names dict --> could integrate units support --> just make 'dumb' for now
        if False:
            units = 'dimless' ## hardcode for now
            if (units=='SI') or (units=='si'): ## m,s,kg,K
                scalar_names = {\
                'x':'x [m]',   'y':'y [m]',   'z':'z [m]', \
                'u':'u [m/s]', 'v':'v [m/s]', 'w':'w [m/s]', \
                'xxx':'x [m]', 'yyy':'y [m]', 'zzz':'z [m]', \
                'T':'T [K]',   'rho':'rho [kg/m^3]', 'p':'p [Pa]', \
                }
            elif (units=='dimless') or (units=='dimensionless'):
                scalar_names = {\
                'x':'x [dimless]',   'y':'y [dimless]',   'z':'z [dimless]', \
                'u':'u [dimless]',   'v':'v [dimless]',   'w':'w [dimless]', \
                'xxx':'x [dimless]', 'yyy':'y [dimless]', 'zzz':'z [dimless]', \
                'T':'T [dimless]',   'rho':'rho [dimless]', 'p':'p [dimless]', \
                }
            else:
                raise ValueError('choice of units not recognized : %s --> options are : %s / %s'%(units,'SI','dimless'))
        else:
            scalar_names = {} ## dummy
        
        # === refresh header
        # if not hasattr(self, 'scalars'):
        #     self.get_header(verbose=False)
        # if not hasattr(self, 't'):
        #     self.get_header(verbose=False)
        self.get_header(verbose=False)
        
        #print('\n'+'scalar dtype.itemsize dtype.name dtype.byteorder'+'\n'+72*'-')
        for scalar in self.scalars:
            data = self['data/%s'%scalar]
            
            dataset_precision_dict[scalar] = data.dtype.itemsize
            txt = '%s%s%s%s%s'%(data.dtype.itemsize, ' '*(4-len(str(data.dtype.itemsize))), data.dtype.name, ' '*(10-len(str(data.dtype.name))), data.dtype.byteorder)
            if verbose: even_print(scalar, txt)
            
            if (data.dtype.name=='float32') or (data.dtype.name=='float64'):
                dataset_numbertype_dict[scalar] = 'Float'
            elif (data.dtype.name=='int8') or (data.dtype.name=='int16') or (data.dtype.name=='int32') or (data.dtype.name=='int64'):
                dataset_numbertype_dict[scalar] = 'Integer'
            else:
                raise TypeError('dtype not recognized, please update script accordingly')
        
        if verbose: print(72*'-')
        
        # === write to .xdmf/.xmf2 file
        if (self.rank==0):
            with open(fname_xdmf,'w') as xdmf:
                
                xdmf_str='''
                         <?xml version="1.0" encoding="utf-8"?>
                         <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
                         <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
                           <Domain>
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 0*' '))
                
                ## Dimensions can also be NumberOfElements
                xdmf_str='''
                         <Topology TopologyType="3DRectMesh" NumberOfElements="%i %i %i"/>
                         <Geometry GeometryType="VxVyVz">
                           <DataItem Dimensions="%i" NumberType="%s" Precision="%i" Format="HDF">
                             %s:/dims/%s
                           </DataItem>
                           <DataItem Dimensions="%i" NumberType="%s" Precision="%i" Format="HDF">
                             %s:/dims/%s
                           </DataItem>
                           <DataItem Dimensions="%i" NumberType="%s" Precision="%i" Format="HDF">
                             %s:/dims/%s
                           </DataItem>
                         </Geometry>
                         ''' % \
                          (self.nz, self.ny, self.nx, \
                           self.nx, dataset_numbertype_dict['x'], dataset_precision_dict['x'], fname_base, 'x', \
                           self.ny, dataset_numbertype_dict['y'], dataset_precision_dict['y'], fname_base, 'y', \
                           self.nz, dataset_numbertype_dict['z'], dataset_precision_dict['z'], fname_base, 'z' )
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 4*' '))
                
                # =====
                
                xdmf_str='''
                         <!-- ==================== time series ==================== -->
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 4*' '))
                
                # ===== the time series
                
                xdmf_str='''
                         <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 4*' '))
                
                for ti in range(len(self.t)):
                    dset_name = 'ts_%06d'%ti
                    #dset_name = self.tss[ti]
                    
                    xdmf_str='''
                             <!-- ============================================================ -->
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # =====
                    
                    xdmf_str='''
                             <Grid Name="%s" GridType="Uniform">
                               <Time TimeType="Single" Value="%E"/>
                               <Topology Reference="/Xdmf/Domain/Topology[1]" />
                               <Geometry Reference="/Xdmf/Domain/Geometry[1]" />
                             ''' % (dset_name, self.t[ti])
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # ===== .xdmf : <Grid> per scalar
                    
                    for scalar in self.scalars:
                        
                        #dset_hf_path = 'data/%s/%s'%(dset_name,scalar) ## 3D
                        dset_hf_path = 'data/%s'%scalar ## 4D
                        
                        try:
                            scalar_name = scalar_names[scalar]
                        except KeyError:
                            scalar_name = scalar
                        
                        xdmf_str='''
                                 <!-- ===== scalar : %s ===== -->
                                 <Attribute Name="%s" AttributeType="Scalar" Center="Node">
                                   <DataItem ItemType="HyperSlab" Dimensions="%i %i %i" Type="HyperSlab">
                                     <DataItem Dimensions="3 4" NumberType="Integer" Format="XML">
                                       %4i %4i %4i %4i
                                       %4i %4i %4i %4i
                                       %4i %4i %4i %4i
                                     </DataItem>
                                     <DataItem Dimensions="%i %i %i %i" NumberType="%s" Precision="%i" Format="HDF">
                                       %s:/%s
                                     </DataItem>
                                   </DataItem>
                                 </Attribute>
                                 ''' % \
                                 (scalar_name, 
                                  scalar_name, 
                                  self.nx, self.ny, self.nz, 
                                  ## Hyperslab: start, stride, count
                                  ti, 0, 0, 0,
                                  1,  1, 1, 1,
                                  1,  self.nz, self.ny, self.nx,
                                  self.nx, self.ny, self.nz, self.nt,
                                  dataset_numbertype_dict[scalar], 
                                  dataset_precision_dict[scalar], 
                                  fname_base, 
                                  dset_hf_path)
                        
                        xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    if makeVectors:
                        
                        # ===== .xdmf : <Grid> per vector : velocity vector
                        
                        if ('u' in self.scalars) and ('v' in self.scalars) and ('w' in self.scalars):
                            
                            scalar_name = 'velocity'
                            
                            if False: ## 3D
                                
                                dset_hf_path_i = 'data/%s/u'%(dset_name,)
                                dset_hf_path_j = 'data/%s/v'%(dset_name,)
                                dset_hf_path_k = 'data/%s/w'%(dset_name,)
                                
                                xdmf_str='''
                                          <!-- ===== vector : %s ===== -->
                                          <Attribute Name="%s" AttributeType="Vector" Center="Node">
                                            <DataItem Dimensions="%i %i %i 3" Function="JOIN($0, $1, $2)" ItemType="Function">
                                              <DataItem Dimensions="%i %i %i 1" NumberType="%s" Precision="%i" Format="HDF">
                                                %s:/%s
                                              </DataItem>
                                              <DataItem Dimensions="%i %i %i 1" NumberType="%s" Precision="%i" Format="HDF">
                                                %s:/%s
                                              </DataItem>
                                              <DataItem Dimensions="%i %i %i 1" NumberType="%s" Precision="%i" Format="HDF">
                                                %s:/%s
                                              </DataItem>
                                            </DataItem>
                                          </Attribute>
                                         ''' % \
                                         (scalar_name,
                                          scalar_name,
                                          self.nx, self.ny, self.nz, 
                                          self.nx, self.ny, self.nz, dataset_numbertype_dict['u'], dataset_precision_dict['u'], fname_base, dset_hf_path_i, \
                                          self.nx, self.ny, self.nz, dataset_numbertype_dict['v'], dataset_precision_dict['v'], fname_base, dset_hf_path_j, \
                                          self.nx, self.ny, self.nz, dataset_numbertype_dict['w'], dataset_precision_dict['w'], fname_base, dset_hf_path_k)
                                
                                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                            
                            if True: ## 4D
                                
                                dset_hf_path_i = 'data/u'
                                dset_hf_path_j = 'data/v'
                                dset_hf_path_k = 'data/w'
                                
                                xdmf_str = '''
                                <!-- ===== vector : %s ===== -->
                                <Attribute Name="%s" AttributeType="Vector" Center="Node">
                                  <DataItem Dimensions="%i %i %i 3" Function="JOIN($0, $1, $2)" ItemType="Function">
                                    <!-- 1 -->
                                    <DataItem ItemType="HyperSlab" Dimensions="%i %i %i 1" Type="HyperSlab">
                                      <DataItem Dimensions="3 4" Format="XML">
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                      </DataItem>
                                      <DataItem Dimensions="%i %i %i %i" NumberType="%s" Precision="%i" Format="HDF">
                                        %s:/%s
                                      </DataItem>
                                    </DataItem>
                                    <!-- 2 -->
                                    <DataItem ItemType="HyperSlab" Dimensions="%i %i %i 1" Type="HyperSlab">
                                      <DataItem Dimensions="3 4" Format="XML">
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                      </DataItem>
                                      <DataItem Dimensions="%i %i %i %i" NumberType="%s" Precision="%i" Format="HDF">
                                        %s:/%s
                                      </DataItem>
                                    </DataItem>
                                    <!-- 3 -->
                                    <DataItem ItemType="HyperSlab" Dimensions="%i %i %i 1" Type="HyperSlab">
                                      <DataItem Dimensions="3 4" Format="XML">
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                      </DataItem>
                                      <DataItem Dimensions="%i %i %i %i" NumberType="%s" Precision="%i" Format="HDF">
                                        %s:/%s
                                      </DataItem>
                                    </DataItem>
                                    <!-- - -->
                                  </DataItem>
                                </Attribute>
                                ''' % \
                                (scalar_name,
                                 scalar_name,
                                 self.nx, self.ny, self.nz, 
                                 #
                                 self.nx, self.ny, self.nz, 
                                 ti, 0, 0, 0,
                                 1,  1, 1, 1,
                                 1,    self.nz, self.ny, self.nx,
                                 self.nx, self.ny, self.nz, self.nt,
                                 dataset_numbertype_dict['u'], 
                                 dataset_precision_dict['u'], 
                                 fname_base, 
                                 dset_hf_path_i,
                                 #
                                 self.nx, self.ny, self.nz, 
                                 ti, 0, 0, 0,
                                 1,  1, 1, 1,
                                 1,    self.nz, self.ny, self.nx,
                                 self.nx, self.ny, self.nz, self.nt,
                                 dataset_numbertype_dict['v'], 
                                 dataset_precision_dict['v'], 
                                 fname_base, 
                                 dset_hf_path_j,
                                 #
                                 self.nx, self.ny, self.nz, 
                                 ti, 0, 0, 0,
                                 1,  1, 1, 1,
                                 1,    self.nz, self.ny, self.nx,
                                 self.nx, self.ny, self.nz, self.nt,
                                 dataset_numbertype_dict['w'], 
                                 dataset_precision_dict['w'], 
                                 fname_base, 
                                 dset_hf_path_k)
                                
                                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                        
                        # ===== .xdmf : <Grid> per vector : vorticity vector
                        
                        if ('vort_x' in self.scalars) and ('vort_y' in self.scalars) and ('vort_z' in self.scalars):
                            
                            scalar_name = 'vorticity'
                            
                            if False: ## 3D
                                
                                dset_hf_path_i = 'data/%s/vort_x'%(dset_name,)
                                dset_hf_path_j = 'data/%s/vort_y'%(dset_name,)
                                dset_hf_path_k = 'data/%s/vort_z'%(dset_name,)
                                
                                xdmf_str='''
                                          <!-- ===== vector : %s ===== -->
                                          <Attribute Name="%s" AttributeType="Vector" Center="Node">
                                            <DataItem Dimensions="%i %i %i 3" Function="JOIN($0, $1, $2)" ItemType="Function">
                                              <DataItem Dimensions="%i %i %i 1" NumberType="%s" Precision="%i" Format="HDF">
                                                %s:/%s
                                              </DataItem>
                                              <DataItem Dimensions="%i %i %i 1" NumberType="%s" Precision="%i" Format="HDF">
                                                %s:/%s
                                              </DataItem>
                                              <DataItem Dimensions="%i %i %i 1" NumberType="%s" Precision="%i" Format="HDF">
                                                %s:/%s
                                              </DataItem>
                                            </DataItem>
                                          </Attribute>
                                         ''' % \
                                         (scalar_name,
                                          scalar_name,
                                          self.nx, self.ny, self.nz, 
                                          self.nx, self.ny, self.nz, dataset_numbertype_dict['vort_x'], dataset_precision_dict['vort_x'], fname_base, dset_hf_path_i, \
                                          self.nx, self.ny, self.nz, dataset_numbertype_dict['vort_y'], dataset_precision_dict['vort_y'], fname_base, dset_hf_path_j, \
                                          self.nx, self.ny, self.nz, dataset_numbertype_dict['vort_z'], dataset_precision_dict['vort_z'], fname_base, dset_hf_path_k)
                                
                                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                            
                            if True: ## 4D
                                
                                dset_hf_path_i = 'data/vort_x'
                                dset_hf_path_j = 'data/vort_y'
                                dset_hf_path_k = 'data/vort_z'
                                
                                xdmf_str = '''
                                <!-- ===== vector : %s ===== -->
                                <Attribute Name="%s" AttributeType="Vector" Center="Node">
                                  <DataItem Dimensions="%i %i %i 3" Function="JOIN($0, $1, $2)" ItemType="Function">
                                    <!-- 1 -->
                                    <DataItem ItemType="HyperSlab" Dimensions="%i %i %i 1" Type="HyperSlab">
                                      <DataItem Dimensions="3 4" Format="XML">
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                      </DataItem>
                                      <DataItem Dimensions="%i %i %i %i" NumberType="%s" Precision="%i" Format="HDF">
                                        %s:/%s
                                      </DataItem>
                                    </DataItem>
                                    <!-- 2 -->
                                    <DataItem ItemType="HyperSlab" Dimensions="%i %i %i 1" Type="HyperSlab">
                                      <DataItem Dimensions="3 4" Format="XML">
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                      </DataItem>
                                      <DataItem Dimensions="%i %i %i %i" NumberType="%s" Precision="%i" Format="HDF">
                                        %s:/%s
                                      </DataItem>
                                    </DataItem>
                                    <!-- 3 -->
                                    <DataItem ItemType="HyperSlab" Dimensions="%i %i %i 1" Type="HyperSlab">
                                      <DataItem Dimensions="3 4" Format="XML">
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                        %4i %4i %4i %4i
                                      </DataItem>
                                      <DataItem Dimensions="%i %i %i %i" NumberType="%s" Precision="%i" Format="HDF">
                                        %s:/%s
                                      </DataItem>
                                    </DataItem>
                                    <!-- - -->
                                  </DataItem>
                                </Attribute>
                                ''' % \
                                (scalar_name,
                                 scalar_name,
                                 self.nx, self.ny, self.nz, 
                                 #
                                 self.nx, self.ny, self.nz, 
                                 ti, 0, 0, 0,
                                 1,  1, 1, 1,
                                 1,  self.nz, self.ny, self.nx,
                                 self.nx, self.ny, self.nz, self.nt,
                                 dataset_numbertype_dict['vort_x'], 
                                 dataset_precision_dict['vort_x'], 
                                 fname_base, 
                                 dset_hf_path_i,
                                 #
                                 self.nx, self.ny, self.nz, 
                                 ti, 0, 0, 0,
                                 1,  1, 1, 1,
                                 1,  self.nz, self.ny, self.nx,
                                 self.nx, self.ny, self.nz, self.nt,
                                 dataset_numbertype_dict['vort_y'], 
                                 dataset_precision_dict['vort_y'], 
                                 fname_base, 
                                 dset_hf_path_j,
                                 #
                                 self.nx, self.ny, self.nz, 
                                 ti, 0, 0, 0,
                                 1,  1, 1, 1,
                                 1,  self.nz, self.ny, self.nx,
                                 self.nx, self.ny, self.nz, self.nt,
                                 dataset_numbertype_dict['vort_z'], 
                                 dataset_precision_dict['vort_z'], 
                                 fname_base, 
                                 dset_hf_path_k)
                                
                                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    if makeTensors:
                        if all([('dudx' in self.scalars),('dvdx' in self.scalars),('dwdx' in self.scalars),
                                ('dudy' in self.scalars),('dvdy' in self.scalars),('dwdy' in self.scalars),
                                ('dudz' in self.scalars),('dvdz' in self.scalars),('dwdz' in self.scalars)]):
                            pass
                            pass ## TODO
                            pass
                    
                    # === .xdmf : end Grid : </Grid>
                    
                    xdmf_str='''
                             </Grid>
                             '''
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                
                # ===
                
                xdmf_str='''
                             </Grid>
                           </Domain>
                         </Xdmf>
                         '''
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 0*' '))
        
        if verbose: print('--w-> %s'%fname_xdmf_base)
        return

class eas4(h5py.File):
    '''
    Interface class for EAS4 files
    '''
    
    def __init__(self, *args, **kwargs):
        
        self.fname, openMode = args
        
        ## check if running with MPI
        if ('comm' in kwargs):
            self.comm = kwargs['comm']
            self.n_ranks = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        else:
            self.comm = None
            self.n_ranks = 1
            self.rank = 0
        
        if ('info' in kwargs):
            self.mpi_info = kwargs['info']
        else:
            mpi_info = MPI.Info.Create()
            mpi_info.Set('romio_ds_write' , 'disable'   )                             
            mpi_info.Set('romio_ds_read'  , 'disable'   )
            mpi_info.Set('romio_cb_read'  , 'automatic' )
            mpi_info.Set('romio_cb_write' , 'automatic' )
            mpi_info.Set('collective_buffering' , 'true' )
            mpi_info.Set('cb_block_size'  , str(int(round(    2*1024**2))))
            mpi_info.Set('cb_buffer_size' , str(int(round( 64*2*1024**2))))
            kwargs['info'] = mpi_info
            self.mpi_info = mpi_info
        
        if ('rdcc_nbytes' in kwargs):
            pass
        else:
            kwargs['rdcc_nbytes']=4*1024**3
        
        ## eas4() unique kwargs --> pop rather than get
        verbose = kwargs.pop('verbose',False)
        self.verbose = verbose
        #force   = kwargs.pop('force',False)
        
        if (openMode != 'r'):
            raise NotImplementedError('eas4(): opening EAS4 in anything but read mode \'r\' is not allowed')
        
        ## call actual h5py.File.__init__()
        super(eas4, self).__init__(*args, **kwargs)
        self.get_header(verbose=verbose)
    
    def __enter__(self):
        '''
        for use with python 'with' statement
        '''
        #return self
        return super(eas4, self).__enter__()
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        '''
        for use with python 'with' statement
        '''
        if (self.rank==0):
            if exception_type is not None:
                print('\nsafely closed EAS4 HDF5 due to exception')
                print(72*'-')
                print('exception type : '+exception_type.__name__)
            if exception_value is not None:
                print('exception_value : '+str(exception_value))
            if exception_traceback is not None:
                print(72*'-')
                #print('exception_traceback : '+str(exception_traceback))
                print('exception_traceback : \n'+traceback.format_exc().rstrip())
            if exception_type is not None:
                print(72*'-')
        return super(eas4, self).__exit__()
    
    def get_header(self,**kwargs):
        
        EAS4=1
        IEEES=1; IEEED=2
        EAS4_NO_G=1; EAS4_X0DX_G=2; EAS4_UDEF_G=3; EAS4_ALL_G=4; EAS4_FULL_G=5
        gmode_dict = {1:'EAS4_NO_G', 2:'EAS4_X0DX_G', 3:'EAS4_UDEF_G', 4:'EAS4_ALL_G', 5:'EAS4_FULL_G'}
        
        # === characteristic values
        
        if self.verbose: print(72*'-')
        Ma    = self['Kennsatz']['FLOWFIELD_PROPERTIES'].attrs['Ma'][0]
        Re    = self['Kennsatz']['FLOWFIELD_PROPERTIES'].attrs['Re'][0]
        Pr    = self['Kennsatz']['FLOWFIELD_PROPERTIES'].attrs['Pr'][0]
        kappa = self['Kennsatz']['FLOWFIELD_PROPERTIES'].attrs['kappa'][0]
        R     = self['Kennsatz']['FLOWFIELD_PROPERTIES'].attrs['R'][0]
        p_inf = self['Kennsatz']['FLOWFIELD_PROPERTIES'].attrs['p_inf'][0]
        T_inf = self['Kennsatz']['FLOWFIELD_PROPERTIES'].attrs['T_inf'][0]
        # !!!!! ----- ACHTUNG ACHTUNG ACHTUNG ----- !!!!! #
        C_Suth_in_file_which_is_mislabelled = self['Kennsatz']['VISCOUS_PROPERTIES'].attrs['C_Suth'][0]
        S_Suth      = C_Suth_in_file_which_is_mislabelled
        # !!!!! ----- ACHTUNG ACHTUNG ACHTUNG ----- !!!!! #
        mu_Suth_ref = self['Kennsatz']['VISCOUS_PROPERTIES'].attrs['mu_Suth_ref'][0]
        T_Suth_ref  = self['Kennsatz']['VISCOUS_PROPERTIES'].attrs['T_Suth_ref'][0]
        
        C_Suth = mu_Suth_ref/(T_Suth_ref**(3/2))*(T_Suth_ref + S_Suth) ## [kg/(m·s·√K)]
        
        if self.verbose: even_print('Ma'          , '%0.2f [-]'           % Ma          )
        if self.verbose: even_print('Re'          , '%0.1f [-]'           % Re          )
        if self.verbose: even_print('Pr'          , '%0.3f [-]'           % Pr          )
        if self.verbose: even_print('T_inf'       , '%0.3f [K]'           % T_inf       )
        if self.verbose: even_print('p_inf'       , '%0.1f [Pa]'          % p_inf       )
        if self.verbose: even_print('kappa'       , '%0.3f [-]'           % kappa       )
        if self.verbose: even_print('R'           , '%0.3f [J/(kg·K)]'    % R           )
        if self.verbose: even_print('mu_Suth_ref' , '%0.6E [kg/(m·s)]'    % mu_Suth_ref )
        if self.verbose: even_print('T_Suth_ref'  , '%0.2f [K]'           % T_Suth_ref  )
        if self.verbose: even_print('C_Suth'      , '%0.5e [kg/(m·s·√K)]' % C_Suth      )
        if self.verbose: even_print('S_Suth'      , '%0.2f [K]'           % S_Suth      )
        
        # === characteristic values : derived
        
        if self.verbose: print(72*'-')
        rho_inf = p_inf/(R * T_inf)
        
        mu_inf_1 = 14.58e-7*T_inf**1.5/(T_inf+110.4)
        mu_inf_2 = mu_Suth_ref*(T_inf/T_Suth_ref)**(3/2)*(T_Suth_ref+S_Suth)/(T_inf+S_Suth)
        mu_inf_3 = C_Suth*T_inf**(3/2)/(T_inf+S_Suth)
        
        if not np.isclose(mu_inf_1, mu_inf_2, rtol=1e-08):
            raise AssertionError('inconsistency in Sutherland calc --> check')
        if not np.isclose(mu_inf_2, mu_inf_3, rtol=1e-08):
            raise AssertionError('inconsistency in Sutherland calc --> check')
        
        mu_inf = mu_inf_3
        
        nu_inf  = mu_inf/rho_inf                   
        a_inf   = np.sqrt(kappa*R*T_inf)           
        U_inf   = Ma*a_inf
        cp      = R*kappa/(kappa-1.)
        cv      = cp/kappa                         
        r       = Pr**(1/3)                        
        Tw      = T_inf                            
        Taw     = T_inf + r*U_inf**2/(2*cp)        
        lchar   = Re*nu_inf/U_inf                  
        
        if self.verbose: even_print('rho_inf' , '%0.3f [kg/m³]'    % rho_inf )
        if self.verbose: even_print('mu_inf'  , '%0.6E [kg/(m·s)]' % mu_inf  )
        if self.verbose: even_print('nu_inf'  , '%0.6E [m²/s]'     % nu_inf  )
        if self.verbose: even_print('a_inf'   , '%0.6f [m/s]'      % a_inf   )
        if self.verbose: even_print('U_inf'   , '%0.6f [m/s]'      % U_inf   )
        if self.verbose: even_print('cp'      , '%0.3f [J/(kg·K)]' % cp      )
        if self.verbose: even_print('cv'      , '%0.3f [J/(kg·K)]' % cv      )
        if self.verbose: even_print('r'       , '%0.6f [-]'        % r       )
        if self.verbose: even_print('Tw'      , '%0.3f [K]'        % Tw      )
        if self.verbose: even_print('Taw'     , '%0.3f [K]'        % Taw     )
        if self.verbose: even_print('lchar'   , '%0.6E [m]'        % lchar   )
        if self.verbose: print(72*'-'+'\n')
        
        # ===
        
        self.Ma           = Ma
        self.Re           = Re
        self.Pr           = Pr
        self.kappa        = kappa
        self.R            = R
        self.p_inf        = p_inf
        self.T_inf        = T_inf
        self.C_Suth       = C_Suth
        self.S_Suth       = S_Suth
        self.mu_Suth_ref  = mu_Suth_ref
        self.T_Suth_ref   = T_Suth_ref
        ##
        self.rho_inf      = rho_inf
        self.mu_inf       = mu_inf
        self.nu_inf       = nu_inf
        self.a_inf        = a_inf
        self.U_inf        = U_inf
        self.cp           = cp
        self.cv           = cv
        self.r            = r
        self.Tw           = Tw
        self.Taw          = Taw
        self.lchar        = lchar
        
        # === grid info
        
        domainName = 'DOMAIN_000000' ## assume only one domain for now
        
        if self.verbose: print('grid info\n'+72*'-')
        x = np.copy(self['Kennsatz/GEOMETRY/%s/dim01'%domainName][:])
        y = np.copy(self['Kennsatz/GEOMETRY/%s/dim02'%domainName][:])
        z = np.copy(self['Kennsatz/GEOMETRY/%s/dim03'%domainName][:])
        
        ndim1 = self['Kennsatz/GEOMETRY/%s'%domainName].attrs['DOMAIN_SIZE'][0] #; print(ndim1)
        ndim2 = self['Kennsatz/GEOMETRY/%s'%domainName].attrs['DOMAIN_SIZE'][1] #; print(ndim2)
        ndim3 = self['Kennsatz/GEOMETRY/%s'%domainName].attrs['DOMAIN_SIZE'][2] #; print(ndim3)
        
        gmode_dim1 = self['Kennsatz/GEOMETRY/%s'%domainName].attrs['DOMAIN_GMODE'][0] # ; print(gmode_dim1)
        gmode_dim2 = self['Kennsatz/GEOMETRY/%s'%domainName].attrs['DOMAIN_GMODE'][1] # ; print(gmode_dim2)
        gmode_dim3 = self['Kennsatz/GEOMETRY/%s'%domainName].attrs['DOMAIN_GMODE'][2] # ; print(gmode_dim3) ## --> 2 in r882
        
        size_dim1  = self['Kennsatz/GEOMETRY/%s/dim01'%domainName].shape #; print(size_dim1)
        size_dim2  = self['Kennsatz/GEOMETRY/%s/dim02'%domainName].shape #; print(size_dim2)
        size_dim3  = self['Kennsatz/GEOMETRY/%s/dim03'%domainName].shape #; print(size_dim3)
        
        # === If geometry mode > EAS4_NO_G for dimensions 1 to 3
        
        if gmode_dim1 > EAS4_NO_G:
            dim1_data = self['Kennsatz/GEOMETRY/%s/dim01'%domainName][()]
            #np.where(dim1_data < 1e-12, 0., dim1_data)
        
        if gmode_dim2 > EAS4_NO_G:
            dim2_data = self['Kennsatz/GEOMETRY/%s/dim02'%domainName][()]
            #np.where(dim1_data < 1e-12, 0., dim2_data)
        
        if gmode_dim3 > EAS4_NO_G:
            dim3_data = self['Kennsatz/GEOMETRY/%s/dim03'%domainName][()]
        ## else:
        ##     dim3_data = 0. ## --> what was this for?
        
        self.gmode_dim1 = gmode_dict[gmode_dim1]
        self.gmode_dim2 = gmode_dict[gmode_dim2]
        self.gmode_dim3 = gmode_dict[gmode_dim3]
        
        # print(dim3_data)
        
        # ### 'x', 'y', 'z' --> assume dim 1/2/3 = x/y/z
        # attr_dim1 = hf['Kennsatz/GEOMETRY/%s/dim01'%domainName].attrs['DIM_ATTR'][()][0].decode().strip() ; print(attr_dim1)
        # attr_dim2 = hf['Kennsatz/GEOMETRY/%s/dim02'%domainName].attrs['DIM_ATTR'][()][0].decode().strip() ; print(attr_dim2)
        # attr_dim3 = hf['Kennsatz/GEOMETRY/%s/dim03'%domainName].attrs['DIM_ATTR'][()][0].decode().strip() ; print(attr_dim3)
        
        # === convert GMODE 
        
        change_gmode = True
        if change_gmode:
            if gmode_dim1==EAS4_X0DX_G:
                dim1_data = np.linspace(dim1_data[0],dim1_data[0]+dim1_data[1]*(ndim1-1), ndim1)
                gmode_dim1=EAS4_ALL_G
            if gmode_dim2==EAS4_X0DX_G:
                dim2_data = np.linspace(dim2_data[0],dim2_data[0]+dim2_data[1]*(ndim2-1), ndim2)
                gmode_dim2=EAS4_ALL_G
            if gmode_dim3==EAS4_X0DX_G:
                dim3_data = np.linspace(dim3_data[0],dim3_data[0]+dim3_data[1]*(ndim3-1), ndim3)
                gmode_dim3=EAS4_ALL_G
        
        ### dim1   = [dim1_data, ndim1, gmode_dim1]
        ### dim2   = [dim2_data, ndim2, gmode_dim2]
        ### dim3   = [dim3_data, ndim3, gmode_dim3]
        ### dim = [dim1, dim2, dim3]
        ### #self.geom.append(dim)
        
        # # Create new structure for returning values
        # self.time   = [time_data, nzs, gmode_time]
        # self.par    = [param, npar, gmode_param]
        # self.attr   = [time_step, attr_time, attr_param, [attr_dim1, attr_dim2, attr_dim3]]
        
        # ===
        
        x = np.copy(dim1_data)
        y = np.copy(dim2_data)
        z = np.copy(dim3_data)
        nx = x.size
        ny = y.size
        nz = z.size
        ngp = nx*ny*nz
        
        ## bug (r882) check
        if (z.size > 1):
            if np.all(np.isclose(z,z[0],rtol=1e-12)):
                raise AssertionError('z has size > 1 but all grid coords are identical!')
        
        if self.verbose: even_print('nx', '%i'%nx )
        if self.verbose: even_print('ny', '%i'%ny )
        if self.verbose: even_print('nz', '%i'%nz )
        if self.verbose: even_print('ngp', '%i'%ngp )
        if self.verbose: print(72*'-')
        
        if self.verbose: even_print('x_min', '%0.2f'%x.min())
        if self.verbose: even_print('x_max', '%0.2f'%x.max())
        if self.verbose: even_print('dx begin : end', '%0.3E : %0.3E'%( (x[1]-x[0]), (x[-1]-x[-2]) ))
        if self.verbose: even_print('y_min', '%0.2f'%y.min())
        if self.verbose: even_print('y_max', '%0.2f'%y.max())
        if self.verbose: even_print('dy begin : end', '%0.3E : %0.3E'%( (y[1]-y[0]), (y[-1]-y[-2]) ))
        if self.verbose: even_print('z_min', '%0.2f'%z.min())
        if self.verbose: even_print('z_max', '%0.2f'%z.max())        
        if self.verbose: even_print('dz begin : end', '%0.3E : %0.3E'%( (z[1]-z[0]), (z[-1]-z[-2]) ))
        if self.verbose: print(72*'-'+'\n')
        
        self.x   = x
        self.y   = y
        self.z   = z
        self.nx  = nx
        self.ny  = ny
        self.nz  = nz
        self.ngp = ngp
        
        # === time & scalar info
        
        if self.verbose: print('time & scalar info\n'+72*'-')
        
        n_scalars   = self['Kennsatz/PARAMETER'].attrs['PARAMETER_SIZE'][0]
        
        if ('Kennsatz/PARAMETER/PARAMETERS_ATTRS' in self):
            scalars =  [ s.decode('utf-8').strip() for s in self['Kennsatz/PARAMETER/PARAMETERS_ATTRS'][()] ]
        else:
            ## this is the older gen structure
            scalars = [ self['Kennsatz/PARAMETER'].attrs['PARAMETERS_ATTR_%06d'%i][0].decode('utf-8').strip() for i in range(n_scalars) ]
        
        scalar_n_map = dict(zip(scalars, range(n_scalars)))
        
        self.scalars_dtypes = []
        for scalar in scalars:
            dset_path = 'Data/%s/ts_%06d/par_%06d'%(domainName,0,scalar_n_map[scalar])
            self.scalars_dtypes.append(self[dset_path].dtype)
        
        nt          = self['Kennsatz/TIMESTEP'].attrs['TIMESTEP_SIZE'][0] 
        
        gmode_time  = self['Kennsatz/TIMESTEP'].attrs['TIMESTEP_MODE'][0]
        t           = self['Kennsatz/TIMESTEP/TIMEGRID'][:]
        
        if (gmode_time==EAS4_X0DX_G): ##2 --> i.e. more than one timestep
            t = np.linspace(t[0],t[0]+t[1]*(nt - 1), nt  )
            gmode_time=EAS4_ALL_G
        else:
            #print('gmode_time : '+str(gmode_time))
            pass
        
        if (t.size>1):
            dt = t[1] - t[0]
            duration = t[-1] - t[0]
        else:
            dt = 0.
            duration = 0.
        
        if self.verbose: even_print('nt', '%i'%nt )
        if self.verbose: even_print('dt', '%0.6f'%dt)
        if self.verbose: even_print('duration', '%0.2f'%duration )
        
        self.n_scalars = n_scalars
        self.scalars = scalars
        self.scalar_n_map = scalar_n_map
        self.t  = t
        self.dt = dt
        self.nt = nt
        self.duration = duration
        
        self.ti = np.arange(self.nt, dtype=np.float64)
        
        # ===
        
        if ('/Kennsatz/AUXILIARY/AVERAGING' in self):
            #print(self['/Kennsatz/AUXILIARY/AVERAGING'].attrs['total_avg_time'])
            #print(self['/Kennsatz/AUXILIARY/AVERAGING'].attrs['total_avg_iter_count'])
            self.total_avg_time       = self['/Kennsatz/AUXILIARY/AVERAGING'].attrs['total_avg_time'][0]
            self.total_avg_iter_count = self['/Kennsatz/AUXILIARY/AVERAGING'].attrs['total_avg_iter_count'][0]
            if self.verbose: even_print('total_avg_time', '%0.2f'%self.total_avg_time)
            if self.verbose: even_print('total_avg_iter_count', '%i'%self.total_avg_iter_count)
            self.measType = 'mean'
        else:
            self.measType = 'unsteady'
        
        if self.verbose: print(72*'-'+'\n')
        
        # ===
        
        udef_char = [     'Ma',     'Re',     'Pr',      'kappa',    'R',    'p_inf',    'T_inf',    'C_Suth',    'S_Suth',    'mu_Suth_ref',    'T_Suth_ref' ]
        udef_real = [ self.Ma , self.Re , self.Pr ,  self.kappa, self.R, self.p_inf, self.T_inf, self.C_Suth, self.S_Suth, self.mu_Suth_ref, self.T_Suth_ref  ]
        self.udef = dict(zip(udef_char, udef_real))
        return

class lpd(h5py.File):
    '''
    Lagrangian Particle Data (LPD)
    ------------------------------
    - super()'ed h5py.File class
    
    Structure
    ---------
    
    lpd.h5
    │
    ├── header/
    │   └── udef_char
    │   └── udef_real
    │
    ├── dims/ --> 1D (rectilinear coords of source volume for reference)
    │   └── x
    │   └── y
    │   └── z
    │   └── t
    │
    └-─ data/
        └── <<scalar>> --> 2D [pts,time]
    '''
    
    def __init__(self, *args, **kwargs):
        
        self.fname, openMode = args
        
        ## check if running with MPI
        if ('comm' in kwargs):
            self.comm = kwargs['comm']
            self.n_ranks = self.comm.Get_size()
            self.rank = self.comm.Get_rank()
        else:
            self.comm = None
            self.n_ranks = 1
            self.rank = 0
        
        if ('info' in kwargs):
            self.mpi_info = kwargs['info']
        else:
            mpi_info = MPI.Info.Create()
            mpi_info.Set('romio_ds_write' , 'disable'   )                             
            mpi_info.Set('romio_ds_read'  , 'disable'   )
            mpi_info.Set('romio_cb_read'  , 'automatic' )
            mpi_info.Set('romio_cb_write' , 'automatic' )
            mpi_info.Set('collective_buffering' , 'true' )
            mpi_info.Set('cb_block_size'  , str(int(round(    2*1024**2))))
            mpi_info.Set('cb_buffer_size' , str(int(round( 64*2*1024**2))))
            kwargs['info'] = mpi_info
            self.mpi_info = mpi_info
        
        if ('driver' not in kwargs) and ('info' in kwargs):
            del kwargs['info']
        
        if ('rdcc_nbytes' in kwargs):
            pass
        else:
            kwargs['rdcc_nbytes']=4*1024**3
        
        ## lpd() unique kwargs --> pop() rather than get()
        verbose = kwargs.pop('verbose',False)
        force   = kwargs.pop('force',False)
        
        if (openMode == 'w') and (force is False) and os.path.isfile(self.fname):
            if (self.rank==0):
                print('\n'+72*'-')
                print(self.fname+' already exists! opening with \'w\' would overwrite.\n')
                openModeInfoStr = '''
                                  r       --> Read only, file must exist
                                  r+      --> Read/write, file must exist
                                  w       --> Create file, truncate if exists
                                  w- or x --> Create file, fail if exists
                                  a       --> Read/write if exists, create otherwise
                                  
                                  or use force=True arg:
                                  
                                  >>> with lpd(<<fname>>,'w',force=True) as f:
                                  >>>     ...
                                  '''
                print(textwrap.indent(textwrap.dedent(openModeInfoStr), 2*' ').strip('\n'))
                print(72*'-'+'\n')
            
            if (self.comm is not None):
                self.comm.Barrier()
            raise FileExistsError()
        
        ## remove file, touch, stripe
        elif (openMode == 'w') and (force is True) and os.path.isfile(self.fname):
            if (self.rank==0):
                os.remove(self.fname)
                Path(self.fname).touch()
                if shutil.which('lfs') is not None:
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 2M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        ## touch, stripe
        elif (openMode == 'w') and not os.path.isfile(self.fname):
            if (self.rank==0):
                Path(self.fname).touch()
                if shutil.which('lfs') is not None:
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 2M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        else:
            pass
        
        if (self.comm is not None):
            self.comm.Barrier()
        
        ## call actual h5py.File.__init__()
        super(lpd, self).__init__(*args, **kwargs)
        self.get_header(verbose=verbose)
    
    def __enter__(self):
        '''
        for use with python 'with' statement
        '''
        #return self
        return super(lpd, self).__enter__()
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        '''
        for use with python 'with' statement
        '''
        if (self.rank==0):
            if exception_type is not None:
                print('\nsafely closed LPD HDF5 due to exception')
                print(72*'-')
                print('exception type : '+exception_type.__name__)
            if exception_value is not None:
                print('exception_value : '+str(exception_value))
            if exception_traceback is not None:
                print(72*'-')
                #print('exception_traceback : '+str(exception_traceback))
                print('exception_traceback : \n'+traceback.format_exc().rstrip())
            if exception_type is not None:
                print(72*'-')
        return super(lpd, self).__exit__()
    
    def get_header(self,**kwargs):
        '''
        initialize header attributes of LPD class instance
        '''
        verbose = kwargs.get('verbose',True)
        
        if (self.rank!=0):
            verbose=False
        
        # === scalars, npts
        
        if ('data' in self):
            self.scalars = list(self['data'].keys()) ## string names of scalars : ['u','v','w'] ...
            npts,nt = self['data/%s'%self.scalars[0]].shape
            self.npts = npts
            self.n_scalars = len(self.scalars)
            self.scalars_dtypes = []
            for scalar in self.scalars:
                self.scalars_dtypes.append(self['data/%s'%scalar].dtype)
            self.scalars_dtypes_dict = dict(zip(self.scalars, self.scalars_dtypes)) ## dict {<<scalar>>: <<dtype>>}
        else:
            self.scalars = []
            self.n_scalars = 0
            self.scalars_dtypes = []
            self.scalars_dtypes_dict = dict(zip(self.scalars, self.scalars_dtypes))
        
        # === time vector
        
        if ('dims/t' in self):
            self.t = np.copy(self['dims/t'][:])
            
            if ('data' in self): ## check t dim and data arrays agree
                if (nt!=self.t.size):
                    raise AssertionError('nt!=self.t.size : %i!=%i'%(nt,self.t.size))
            
            try:
                self.dt = self.t[1] - self.t[0]
            except IndexError:
                self.dt = 0.
            
            self.nt       = nt       = self.t.size
            self.duration = duration = self.t[-1] - self.t[0]
            self.ti       = ti       = np.array(range(self.nt), dtype=np.int64)
        else:
            self.t  = np.array([], dtype=np.float64)
            self.ti = np.array([], dtype=np.int64)
            self.nt = nt = 0
            self.dt = 0.
            self.duration = duration = 0.
        
        return
    
    def init_from_rgd(self, rgd_instance, **kwargs):
        '''
        initialize an LPD from an RGD (copy over header data & coordinate data)
        '''
        
        t_info = kwargs.get('t_info',True)
        
        verbose = kwargs.get('verbose',True)
        if (self.rank!=0):
            verbose=False
        
        #with rgd(fn_rgd, 'r', driver='mpio', comm=MPI.COMM_WORLD, libver='latest') as hf_ref:
        hf_ref = rgd_instance
        
        # === copy over header info if needed
        
        if all([('header/udef_real' in self),('header/udef_char' in self)]):
            raise ValueError('udef already present')
        else:
            udef         = hf_ref.udef
            udef_real    = list(udef.values())
            udef_char    = list(udef.keys())
            udef_real_h5 = np.array(udef_real, dtype=np.float64)
            udef_char_h5 = np.array([s.encode('ascii', 'ignore') for s in udef_char], dtype='S128')
            
            self.create_dataset('header/udef_real', data=udef_real_h5, maxshape=np.shape(udef_real_h5), dtype=np.float64)
            self.create_dataset('header/udef_char', data=udef_char_h5, maxshape=np.shape(udef_char_h5), dtype='S128')
            self.udef      = udef
            self.udef_real = udef_real
            self.udef_char = udef_char
        
        # === copy over spatial dim info
        
        x, y, z = hf_ref.x, hf_ref.y, hf_ref.z
        nx  = self.nx  = x.size
        ny  = self.ny  = y.size
        nz  = self.nz  = z.size
        ngp = self.ngp = nx*ny*nz
        if ('dims/x' in self):
            del self['dims/x']
        if ('dims/y' in self):
            del self['dims/y']
        if ('dims/z' in self):
            del self['dims/z']
        
        self.create_dataset('dims/x', data=x)
        self.create_dataset('dims/y', data=y)
        self.create_dataset('dims/z', data=z)
        
        # === copy over temporal dim info
        
        if t_info:
            self.t  = hf_ref.t
            self.nt = self.t.size
            self.create_dataset('dims/t', data=hf_ref.t)
        else:
            t = np.array([0.], dtype=np.float64)
            if ('dims/t' in self):
                del self['dims/t']
            self.create_dataset('dims/t', data=t)
        
        self.get_header(verbose=False)
        return
    
    # ===
    
    def calc_acceleration(self,**kwargs):
        '''
        calculate velocity time derivatives (acceleration and jerk)
        '''
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'lpd.calc_acceleration()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        n_chunks = kwargs.get('n_chunks',1)
        chunk_kb = kwargs.get('chunk_kb',1024)
        
        if verbose:
            even_print('fn_lpd',  self.fname)
            even_print('npts'  , '%i'%self.npts)
            even_print('nt'    , '%i'%self.nt)
            print(72*'-')
        
        ## particle list bounds this rank
        rpl_ = np.array_split(np.arange(self.npts, dtype=np.int64), self.n_ranks )
        rpl  = [[b[0],b[-1]+1] for b in rpl_ ]
        rp1, rp2 = rpl[self.rank]; npr = rp2 - rp1
        
        ## for local rank bound, subdivide into chunks for collective reads
        cpl_ = np.array_split(np.arange(rp1, rp2, dtype=np.int64), n_chunks )
        cpl  = [[b[0],b[-1]+1] for b in cpl_ ]
        
        # === initialize (new) datasets
        
        ## shape & HDF5 chunk scheme for datasets
        shape = (self.npts, self.nt)
        chunks = rgd.chunk_sizer(nxi=shape, constraint=(None,1), size_kb=chunk_kb, base=2)
        
        scalars_in         = ['u','v','w','t','id']
        scalars_in_dtypes  = [self.scalars_dtypes_dict[s] for s in scalars_in]
        
        scalars_out        = ['ax','ay','az', 'jx','jy','jz']
        scalars_out_dtypes = [np.float32 for i in scalars_out]
        
        for scalar in scalars_out:
            
            if verbose:
                even_print('initializing',scalar)
            
            if ('data/%s'%scalar in self):
                del self['data/%s'%scalar]
            dset = self.create_dataset('data/%s'%scalar, 
                                       shape=shape, 
                                       dtype=np.float32,
                                       fillvalue=np.nan,
                                       #compression='gzip', ## this causes segfaults :( :(
                                       #compression_opts=5,
                                       #shuffle=True,
                                       chunks=chunks,
                                       )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (pts,nt)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
        if verbose:
            tqdm.write(72*'-')
        
        # ===
        
        if verbose: progress_bar = tqdm(total=npr, ncols=100, desc='calc accel', leave=False, file=sys.stdout)
        
        for ci in range(n_chunks):
            cp1, cp2 = cpl[ci]
            npc = cp2 - cp1
            
            ## local buffer array (for read)
            pdata_in = np.empty(shape=(npc,self.nt), dtype={'names':scalars_in, 'formats':scalars_in_dtypes})
            pdata_in[:,:] = np.nan
            
            ## local buffer array (for write)
            pdata_out = np.empty(shape=(npc,self.nt), dtype={'names':scalars_out, 'formats':scalars_out_dtypes})
            pdata_out[:,:] = np.nan
            
            ## read data
            if verbose: tqdm.write(even_print('chunk', '%i/%i'%(ci+1,n_chunks), s=True))
            #for scalar in self.scalars:
            for scalar in scalars_in:
                dset = self['data/%s'%scalar]
                self.comm.Barrier()
                t_start = timeit.default_timer()
                with dset.collective:
                    #pdata_in[scalar] = dset[rp1:rp2,:]
                    pdata_in[scalar] = dset[cp1:cp2,:]
                self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                #data_gb = 4 * self.nt * self.npts / 1024**3
                data_gb = ( 4 * self.nt * self.npts / 1024**3 ) / n_chunks
                if verbose:
                    tqdm.write(even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True))
            
            # === iterate over (chunk of) particle tracks
            
            t_start = timeit.default_timer()
            
            #for pi in range(npr): 
            for pi in range(npc): 
                
                u   = pdata_in['u'][pi,:]
                v   = pdata_in['v'][pi,:]
                w   = pdata_in['w'][pi,:]
                t   = pdata_in['t'][pi,:]
                pid = pdata_in['id'][pi,:]
                
                ii_real = np.where(~np.isnan(u))
                ii_nan  = np.where( np.isnan(u))
                n_real  = len(ii_real[0])
                n_nan   = len(ii_nan[0])
                
                if True: ## check non-NaNs have contiguous time indices (no NaN gaps) --> passes
                    aa = ii_real[0]
                    if (aa.shape[0]>1):
                        if not np.all( np.diff(aa) == np.diff(aa)[0] ):
                            #raise AssertionError('non-NaN index arr not const')
                            print('non-NaN index arr not const')
                            self.comm.Abort(1)
                
                if False: ## check non-NaN index vecs are same for every scalar --> passes
                    iiu = np.where(~np.isnan(u))
                    iiv = np.where(~np.isnan(v))
                    iiw = np.where(~np.isnan(w))
                    iit = np.where(~np.isnan(t))
                    
                    if not np.array_equal(iiu, iiv):
                        #raise AssertionError('real index vecs not same between scalars')
                        print('real index vecs not same between scalars')
                        self.comm.Abort(1)
                    if not np.array_equal(iiv, iiw):
                        print('real index vecs not same between scalars')
                        self.comm.Abort(1)
                    if not np.array_equal(iiw, iit):
                        print('real index vecs not same between scalars')
                        self.comm.Abort(1)
                
                ## take where scalar(t) vector isnt NaN
                u   = np.copy(   u[ii_real] )
                v   = np.copy(   v[ii_real] )
                w   = np.copy(   w[ii_real] )
                t   = np.copy(   t[ii_real] )
                pid = np.copy( pid[ii_real] )
                
                if False: ## check ID scalar is constant for this particle --> passes
                    if not np.all(pid == pid[0]):
                        #raise AssertionError('pt ID not same')
                        print('pt ID not same')
                        print(pid)
                        self.comm.Abort(1)
                
                #if (n_real>2):
                if (n_real>2):
                    
                    if False:
                        dudt   = np.gradient(u,    t, axis=0, edge_order=1)
                        dvdt   = np.gradient(v,    t, axis=0, edge_order=1)
                        dwdt   = np.gradient(w,    t, axis=0, edge_order=1)
                        d2udt2 = np.gradient(dudt, t, axis=0, edge_order=1)
                        d2vdt2 = np.gradient(dvdt, t, axis=0, edge_order=1)
                        d2wdt2 = np.gradient(dwdt, t, axis=0, edge_order=1)
                    
                    if True:
                        dudt   = sp.interpolate.CubicSpline(t,u,bc_type='natural')(t,1)
                        dvdt   = sp.interpolate.CubicSpline(t,v,bc_type='natural')(t,1)
                        dwdt   = sp.interpolate.CubicSpline(t,w,bc_type='natural')(t,1)
                        d2udt2 = sp.interpolate.CubicSpline(t,u,bc_type='natural')(t,2)
                        d2vdt2 = sp.interpolate.CubicSpline(t,v,bc_type='natural')(t,2)
                        d2wdt2 = sp.interpolate.CubicSpline(t,w,bc_type='natural')(t,2)
                    
                    ## write to buffer
                    pdata_out['ax'][pi,ii_real] = dudt  
                    pdata_out['ay'][pi,ii_real] = dvdt  
                    pdata_out['az'][pi,ii_real] = dwdt  
                    pdata_out['jx'][pi,ii_real] = d2udt2
                    pdata_out['jy'][pi,ii_real] = d2vdt2
                    pdata_out['jz'][pi,ii_real] = d2wdt2
                
                if verbose: progress_bar.update()
            
            t_delta = timeit.default_timer() - t_start
            if verbose:
                tqdm.write(even_print('calc accel & jerk', '%0.2f [s]'%(t_delta,), s=True))
            
            # === write buffer out
            
            for scalar in scalars_out:
                dset = self['data/%s'%scalar]
                self.comm.Barrier()
                t_start = timeit.default_timer()
                with dset.collective:
                    dset[cp1:cp2,:] = pdata_out[scalar]
                self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                data_gb = ( 4 * self.nt * self.npts / 1024**3 ) / n_chunks
                if verbose:
                    tqdm.write(even_print('write: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True))
            
            # ===
            
            if verbose:
                tqdm.write(72*'-')
        
        if verbose: progress_bar.close()
        
        # ===
        
        self.comm.Barrier()
        self.make_xdmf()
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : lpd.calc_acceleration() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    # === Paraview
    
    def make_xdmf(self, **kwargs):
        '''
        generate an XDMF/XMF2 from LPD for processing with Paraview
        -----
        --> https://www.xdmf.org/index.php/XDMF_Model_and_Format
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        makeVectors = kwargs.get('makeVectors',True) ## write [u,v,w] and [vort_x,vort_y,vort_z] vectors to XDMF
        makeTensors = kwargs.get('makeTensors',True) ## write stress or strain tensors to XDMF
        
        fname_path            = os.path.dirname(self.fname)
        fname_base            = os.path.basename(self.fname)
        fname_root, fname_ext = os.path.splitext(fname_base)
        fname_xdmf_base       = fname_root+'.xmf2'
        fname_xdmf            = os.path.join(fname_path, fname_xdmf_base)
        
        if verbose: print('\n'+'lpd.make_xdmf()'+'\n'+72*'-')
        
        dataset_precision_dict = {} ## holds dtype.itemsize ints i.e. 4,8
        dataset_numbertype_dict = {} ## holds string description of dtypes i.e. 'Float','Integer'
        
        # === refresh header
        self.get_header(verbose=False)
        
        for scalar in self.scalars:
            data = self['data/%s'%scalar]
            
            dataset_precision_dict[scalar] = data.dtype.itemsize
            txt = '%s%s%s%s%s'%(data.dtype.itemsize, ' '*(4-len(str(data.dtype.itemsize))), data.dtype.name, ' '*(10-len(str(data.dtype.name))), data.dtype.byteorder)
            if verbose: even_print(scalar, txt)
            
            if (data.dtype.name=='float32') or (data.dtype.name=='float64'):
                dataset_numbertype_dict[scalar] = 'Float'
            elif (data.dtype.name=='int8') or (data.dtype.name=='int16') or (data.dtype.name=='int32') or (data.dtype.name=='int64'):
                dataset_numbertype_dict[scalar] = 'Integer'
            else:
                raise TypeError('dtype not recognized, please update script accordingly')
        
        if verbose: print(72*'-')
        
        # === make .xdmf/.xmf2 file
        
        if (self.rank==0):
            
            with open(fname_xdmf,'w') as xdmf:
                
                xdmf_str='''
                         <?xml version="1.0" encoding="utf-8"?>
                         <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
                         <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
                           <Domain>
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 0*' '))
                
                # ===
                
                xdmf_str='''
                         <!-- ==================== time series ==================== -->
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 4*' '))
                
                # === the time series
                
                xdmf_str='''
                         <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 4*' '))
                
                for ti in range(len(self.t)):
                    dset_name = 'ts_%06d'%ti
                    
                    xdmf_str='''
                             <!-- ============================================================ -->
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # ===
                    
                    xdmf_str=f'''
                             <Grid Name="{dset_name}" GridType="Uniform">
                               <Time TimeType="Single" Value="{self.t[ti]:E}"/>
                             '''
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    xdmf_str=f'''
                             <Topology TopologyType="Polyvertex" NumberOfElements="{self.npts:d}"/>
                             <!-- === -->
                             <Geometry GeometryType="X_Y_Z">
                             '''
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))

                    # ===
                    
                    xdmf_str=f'''
                              <DataItem ItemType="HyperSlab" Dimensions="{self.npts:d}" Type="HyperSlab">
                                <DataItem Dimensions="3 2" NumberType="Integer" Format="XML">
                                     0        {ti:d}
                                     1        1
                                     {self.npts:d}  1
                                </DataItem>
                                <DataItem Dimensions="{self.npts:d} {self.nt:d}" NumberType="{dataset_numbertype_dict['x']}" Precision="{dataset_precision_dict['x']:d}" Format="HDF">
                                  {fname_base}:/data/x
                                </DataItem>
                              </DataItem>
                              <DataItem ItemType="HyperSlab" Dimensions="{self.npts:d}" Type="HyperSlab">
                                <DataItem Dimensions="3 2" NumberType="Integer" Format="XML">
                                     0        {ti:d}
                                     1        1
                                     {self.npts:d}  1
                                </DataItem>
                                <DataItem Dimensions="{self.npts:d} {self.nt:d}" NumberType="{dataset_numbertype_dict['y']}" Precision="{dataset_precision_dict['y']:d}" Format="HDF">
                                  {fname_base}:/data/y
                                </DataItem>
                              </DataItem>
                              <DataItem ItemType="HyperSlab" Dimensions="{self.npts:d}" Type="HyperSlab">
                                <DataItem Dimensions="3 2" NumberType="Integer" Format="XML">
                                     0        {ti:d}
                                     1        1
                                     {self.npts:d}  1
                                </DataItem>
                                <DataItem Dimensions="{self.npts:d} {self.nt:d}" NumberType="{dataset_numbertype_dict['z']}" Precision="{dataset_precision_dict['z']:d}" Format="HDF">
                                  {fname_base}:/data/z
                                </DataItem>
                              </DataItem>
                            </Geometry>
                            <!-- === -->
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    # === .xdmf : <Grid> per scalar
                    
                    for scalar in self.scalars:
                        
                        dset_hf_path = 'data/%s'%scalar
                        
                        xdmf_str=f'''
                                 <!-- ===== scalar : {scalar} ===== -->
                                 <Attribute Name="{scalar}" AttributeType="Scalar" Center="Node">
                                       <DataItem ItemType="HyperSlab" Dimensions="{self.npts:d}" Type="HyperSlab">
                                         <DataItem Dimensions="3 2" NumberType="Integer" Format="XML">
                                              0        {ti:d}
                                              1        1
                                              {self.npts:d}  1
                                         </DataItem>
                                         <DataItem Dimensions="{self.npts:d} {self.nt:d}" NumberType="{dataset_numbertype_dict[scalar]}" Precision="{dataset_precision_dict[scalar]:d}" Format="HDF">
                                           {fname_base}:{dset_hf_path}
                                         </DataItem>
                                       </DataItem>
                                 </Attribute>
                                 '''
                        
                        xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    # === .xdmf : end Grid : </Grid>
                    
                    xdmf_str='''
                             </Grid>
                             '''
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                
                # ===
                
                xdmf_str='''
                             </Grid>
                           </Domain>
                         </Xdmf>
                         '''
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 0*' '))
        
        if verbose: print('--w-> %s'%fname_xdmf_base)
        return

# data container interface class for EAS3 files
# ------------------------------------------------------------

class eas3:
    '''
    Interface class for EAS3 files
    '''
    
    def __init__(self, fname, **kwargs):
        '''
        initialize class instance
        '''
        self.fname   = fname
        self.verbose = kwargs.get('verbose',True)
        
        if isinstance(fname, str):
            self.f = open(fname,'rb')
        elif isinstance(fname, io.BytesIO):
            self.f = fname
        else:
            raise TypeError('fname should be type str or io.BytesIO')
        
        self.udef    = self.get_header()
    
    def __enter__(self):
        '''
        for use with python 'with' statement
        '''
        #print('opening from enter() --> used with statement')
        return self
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        '''
        for use with python 'with' statement
        '''
        # =====
        self.f.close()
        # =====
        if exception_type is not None:
            print('\nsafely closed EAS3 due to exception')
            print(72*'-')
            print('exception type : '+str(exception_type))
        if exception_value is not None:
            print('exception_value : '+str(exception_value))
        if exception_traceback is not None:
            print('exception_traceback : '+str(exception_traceback))
        if exception_type is not None:
            print(72*'-')
    
    def close(self):
        '''
        close() passthrough to HDF5
        '''
        self.f.close()
    
    def get_header(self, **kwargs):
        
        ATTRLEN     = kwargs.get('ATTRLEN',10)
        UDEFLEN     = kwargs.get('UDEFLEN',20)
        ChangeGmode = kwargs.get('ChangeGmode',1)
        
        ## Definitions
        EAS2=1; EAS3=2
        IEEES=1; IEEED=2; IEEEQ=3
        EAS3_NO_ATTR=1; EAS3_ALL_ATTR=2
        EAS3_NO_G=1; EAS3_X0DX_G=2; EAS3_UDEF_G=3; EAS3_ALL_G=4; EAS3_FULL_G=5
        EAS3_NO_UDEF=1; EAS3_ALL_UDEF=2; EAS3_INT_UDEF=3
        
        self.IEEES = IEEES
        self.IEEED = IEEED
        self.IEEEQ = IEEEQ
        
        ## Identifier: 20 byte character
        identifier = self.f.read(20).strip()
        
        ## File type: 8 byte integer
        file_type = struct.unpack('!q',self.f.read(8))[0]
        
        ## Accuracy: 8 byte integer
        accuracy = struct.unpack('!q',self.f.read(8))[0]
        self.accuracy = accuracy
        
        if (self.accuracy == self.IEEES):
            self.dtype = np.float32
        elif (self.accuracy == self.IEEED):
            self.dtype = np.float64
        elif (self.accuracy == self.IEEEQ):
            self.dtype = np.float128
        else:
            raise ValueError('precision not identifiable')
        
        ## Array sizes: each 8 byte integer
        nzs   = struct.unpack('!q',self.f.read(8))[0]
        npar  = struct.unpack('!q',self.f.read(8))[0]
        ndim1 = struct.unpack('!q',self.f.read(8))[0]
        ndim2 = struct.unpack('!q',self.f.read(8))[0]
        ndim3 = struct.unpack('!q',self.f.read(8))[0]
        
        self.ndim1 = ndim1
        self.ndim2 = ndim2
        self.ndim3 = ndim3
        self.npar  = npar
        self.nzs   = nzs
        
        ## Attribute mode: 8 byte integer
        attribute_mode = struct.unpack('!q',self.f.read(8))[0]
        
        ## Geometry mode: each 8 byte integer
        gmode_time  = struct.unpack('!q',self.f.read(8))[0]
        gmode_param = struct.unpack('!q',self.f.read(8))[0]
        gmode_dim1  = struct.unpack('!q',self.f.read(8))[0]  
        gmode_dim2  = struct.unpack('!q',self.f.read(8))[0]
        gmode_dim3  = struct.unpack('!q',self.f.read(8))[0]
        
        ## Array sizes for geometry data: each 8 byte integer
        size_time  = struct.unpack('!q',self.f.read(8))[0]
        size_param = struct.unpack('!q',self.f.read(8))[0]
        size_dim1  = struct.unpack('!q',self.f.read(8))[0]
        size_dim2  = struct.unpack('!q',self.f.read(8))[0]
        size_dim3  = struct.unpack('!q',self.f.read(8))[0]
        
        ## Specification of user defined data: 8 byte integer
        udef = struct.unpack('!q',self.f.read(8))[0]
        
        ## Array sizes for used defined data: each 8 byte integer
        udef_char_size = struct.unpack('!q',self.f.read(8))[0]
        udef_int_size  = struct.unpack('!q',self.f.read(8))[0]
        udef_real_size = struct.unpack('!q',self.f.read(8))[0]
        
        ## Time step array: nzs x 8 byte
        time_step = np.zeros(nzs,int)
        for it in range(nzs):
            time_step[it] = struct.unpack('!q',self.f.read(8))[0]
        
        if attribute_mode==EAS3_ALL_ATTR:
            ## Time step attributes
            attr_time = [ self.f.read(ATTRLEN).decode('UTF-8').strip() ]
            for it in range(1,nzs):
                attr_time.append( self.f.read(ATTRLEN).decode('UTF-8').strip() )
            ## Parameter attributes
            attr_param = [ self.f.read(ATTRLEN).decode('UTF-8').strip() ]
            for it in range(1,npar):
                attr_param.append( self.f.read(ATTRLEN).decode('UTF-8').strip() )
            
            # Spatial attributes
            attr_dim1 = self.f.read(ATTRLEN).decode('UTF-8').strip()
            attr_dim2 = self.f.read(ATTRLEN).decode('UTF-8').strip()
            attr_dim3 = self.f.read(ATTRLEN).decode('UTF-8').strip()
        
        ## If geometry mode > EAS3_NO_G for time
        if gmode_time == EAS3_X0DX_G:
            time_data = np.zeros(2)
            for it in range(2):
                time_data[it] = struct.unpack('!d',self.f.read(8))[0]
        elif gmode_time == EAS3_ALL_G:
            time_data = np.zeros(size_time)
            for it in range(size_time):
                time_data[it] = struct.unpack('!d',self.f.read(8))[0]
        else: time_data = np.zeros(1)

        ## If geometry mode > EAS3_NO_G for parameters
        if gmode_param > EAS3_NO_G:
            param = np.zeros(size_param)
            for it in range(size_param):
                param[it] = struct.unpack('!d',self.f.read(8))[0]

        ## If geometry mode > EAS3_NO_G for dimensions 1 to 3
        dim1_data = np.zeros(size_dim1)
        if gmode_dim1 > EAS3_NO_G:
            for it in range(size_dim1):
                dim1_data[it] = struct.unpack('!d',self.f.read(8))[0]
            if abs(dim1_data[0]) < 1e-18: dim1_data[0] = 0.   
        dim2_data = np.zeros(size_dim2)
        if gmode_dim2 > EAS3_NO_G:
            for it in range(size_dim2):
                dim2_data[it] = struct.unpack('!d',self.f.read(8))[0]
            if abs(dim2_data[0]) < 1e-18: dim2_data[0] = 0.
        dim3_data = np.zeros(size_dim3)
        if gmode_dim3 > EAS3_NO_G:
            for it in range(size_dim3):
                dim3_data[it] = struct.unpack('!d',self.f.read(8))[0]
        else: dim3_data = 0.
        
        ## If user-defined data is chosen 
        if udef==EAS3_ALL_UDEF:
            udef_char = []
            for it in range(udef_char_size):
                udef_char.append(self.f.read(UDEFLEN).decode('UTF-8').strip())
            udef_int = np.zeros(udef_int_size,int)
            for it in range(udef_int_size):
                udef_int[it] = struct.unpack('!q',self.f.read(8))[0]
            udef_real = np.zeros(udef_real_size)
            for it in range(udef_real_size):
                udef_real[it] = struct.unpack('!d',self.f.read(8))[0]
        
        ## Option: convert gmode=EAS3_X0DX_G to gmode=EAS3_ALL_G
        if ChangeGmode==1:
            if gmode_dim1==EAS3_X0DX_G:
                dim1_data = np.linspace(dim1_data[0],dim1_data[0]+dim1_data[1]*(ndim1-1), ndim1)
                gmode_dim1=EAS3_ALL_G
            if gmode_dim2==EAS3_X0DX_G:
                dim2_data = np.linspace(dim2_data[0],dim2_data[0]+dim2_data[1]*(ndim2-1), ndim2)
                gmode_dim2=EAS3_ALL_G
            if gmode_dim3==EAS3_X0DX_G:
                dim3_data = np.linspace(dim3_data[0],dim3_data[0]+dim3_data[1]*(ndim3-1), ndim3)
                gmode_dim3=EAS3_ALL_G
            if gmode_time==EAS3_X0DX_G:
                time_data = np.linspace(time_data[0],time_data[0]+time_data[1]*(nzs  -1), nzs  )
                gmode_time=EAS3_ALL_G
        
        # ===
        
        self.attr_param = attr_param
        self.scalars    = attr_param
        self.t          = time_data
        self.nt         = self.t.size
        
        if   (attr_dim1=='x'):
            self.x = dim1_data
        elif (attr_dim1=='y'):
            self.y = dim1_data
        elif (attr_dim1=='z'):
            self.z = dim1_data
        else:
            raise ValueError('attr_dim1 = %s not identifiable as any x,y,z'%attr_dim1)
        
        if   (attr_dim2=='x'):
            self.x = dim2_data
        elif (attr_dim2=='y'):
            self.y = dim2_data
        elif (attr_dim2=='z'):
            self.z = dim2_data
        else:
            raise ValueError('attr_dim2 = %s not identifiable as any x,y,z'%attr_dim2)
        
        if   (attr_dim3=='x'):
            self.x = dim3_data
        elif (attr_dim3=='y'):
            self.y = dim3_data
        elif (attr_dim3=='z'):
            self.z = dim3_data
        else:
            raise ValueError('attr_dim3 = %s not identifiable as any x,y,z'%attr_dim3)
        
        # === transpose order to [xyz]
        
        if all([(attr_dim1=='x'),(attr_dim2=='y'),(attr_dim3=='z')]):
            self.axes_transpose_xyz = (0,1,2)
        elif all([(attr_dim1=='y'),(attr_dim2=='x'),(attr_dim3=='z')]):
            self.axes_transpose_xyz = (1,0,2)
        elif all([(attr_dim1=='z'),(attr_dim2=='y'),(attr_dim3=='x')]):
            self.axes_transpose_xyz = (2,1,0)
        elif all([(attr_dim1=='x'),(attr_dim2=='z'),(attr_dim3=='y')]):
            self.axes_transpose_xyz = (0,2,1)
        elif all([(attr_dim1=='y'),(attr_dim2=='z'),(attr_dim3=='x')]):
            self.axes_transpose_xyz = (2,0,1)
        elif all([(attr_dim1=='z'),(attr_dim2=='x'),(attr_dim3=='y')]):
            self.axes_transpose_xyz = (1,2,0)
        else:
            raise ValueError('could not figure out transpose axes')
        
        # ===
        
        self.nx  = self.x.size
        self.ny  = self.y.size
        self.nz  = self.z.size
        self.ngp = self.nx*self.ny*self.nz
        
        if self.verbose: print(72*'-')
        if self.verbose: even_print('nx', '%i'%self.nx )
        if self.verbose: even_print('ny', '%i'%self.ny )
        if self.verbose: even_print('nz', '%i'%self.nz )
        if self.verbose: even_print('ngp', '%i'%self.ngp )
        if self.verbose: print(72*'-')
        
        if self.verbose: even_print('x_min', '%0.2f'%self.x.min())
        if self.verbose: even_print('x_max', '%0.2f'%self.x.max())
        if self.verbose: even_print('dx begin : end', '%0.3E : %0.3E'%( (self.x[1]-self.x[0]), (self.x[-1]-self.x[-2]) ))
        if self.verbose: even_print('y_min', '%0.2f'%self.y.min())
        if self.verbose: even_print('y_max', '%0.2f'%self.y.max())
        if self.verbose: even_print('dy begin : end', '%0.3E : %0.3E'%( (self.y[1]-self.y[0]), (self.y[-1]-self.y[-2]) ))
        if self.verbose: even_print('z_min', '%0.2f'%self.z.min())
        if self.verbose: even_print('z_max', '%0.2f'%self.z.max())        
        if self.verbose: even_print('dz begin : end', '%0.3E : %0.3E'%( (self.z[1]-self.z[0]), (self.z[-1]-self.z[-2]) ))
        if self.verbose: print(72*'-'+'\n')
        
        # ===
        
        udef_dict = {}
        for i in range(len(udef_char)):
            if (udef_char[i]!=''):
                if (udef_int[i]!=0):
                    udef_dict[udef_char[i]] = int(udef_int[i])
                elif (udef_real[i]!=0.):
                    udef_dict[udef_char[i]] = float(udef_real[i])
                else:
                    udef_dict[udef_char[i]] = 0.
        
        if self.verbose:
            print('udef from EAS3\n' + 72*'-')
            for key in udef_dict:
                if isinstance(udef_dict[key],float):
                    even_print(key, '%0.8f'%udef_dict[key])
                elif isinstance(udef_dict[key],int):
                    even_print(key, '%i'%udef_dict[key])
                else:
                    print(type(udef_dict[key]))
                    raise TypeError('udef dict item not float or int')
            print(72*'-'+'\n')
        
        self.Ma    = udef_dict['Ma']
        self.Re    = udef_dict['Re']
        self.Pr    = udef_dict['Pr']
        self.kappa = udef_dict['kappa']
        
        if ('T_unend' in udef_dict):
            self.T_inf = udef_dict['T_unend']
        elif ('T_inf' in udef_dict):
            self.T_inf = udef_dict['T_inf']
        elif ('Tinf' in udef_dict):
            self.T_inf = udef_dict['Tinf']
        else:
            if self.verbose: print('WARNING! No match in udef for any T_unend, T_inf, Tinf')
            if self.verbose: print('--> setting T_inf = (273.15+15) [K]')
            self.T_inf = 273.15 + 15
        
        if ('R' in udef_dict):
            self.R = udef_dict['R']
        ## elif ('cv' in udef_dict):
        ##     print('WARNING! No match in udef for R, but cv given')
        ##     self.cv = udef_dict['cv']
        ##     self.R  = self.cv / (5/2)
        ##     print('--> assuming air, taking R = cv/(5/2) = %0.3f [J/(kg·K)]'%self.R)
        else:
            #if self.verbose: print('WARNING! No match in udef for R and no cv given')
            if self.verbose: print('WARNING! No match in udef for R')
            if self.verbose: print('--> assuming air, setting R = 287.055 [J/(kg·K)]')
            self.R = 287.055
        
        if ('p_unend' in udef_dict):
            self.p_inf = udef_dict['p_unend']
        elif ('p_inf' in udef_dict):
            self.p_inf = udef_dict['p_inf']
        elif ('pinf' in udef_dict):
            self.p_inf = udef_dict['pinf']
        else:
            if self.verbose: print('WARNING! No match in udef for any p_unend, p_inf, pinf')
            if self.verbose: print('--> setting p_inf = 101325 [Pa]')
            self.p_inf = 101325.

        self.rho_inf     = self.p_inf/(self.R*self.T_inf) ## mass density [kg/m³]
        
        self.S_Suth      = 110.4    ## [K] --> Sutherland temperature
        self.mu_Suth_ref = 1.716e-5 ## [kg/(m·s)] --> μ of air at T_Suth_ref = 273.15 [K]
        self.T_Suth_ref  = 273.15   ## [K]
        self.C_Suth      = self.mu_Suth_ref/(self.T_Suth_ref**(3/2))*(self.T_Suth_ref + self.S_Suth) ## [kg/(m·s·√K)]
        #mu_inf      = self.C_Suth*self.T_inf**(3/2)/(self.T_inf+self.S_Suth)
        self.mu_inf      = self.mu_Suth_ref*(self.T_inf/self.T_Suth_ref)**(3/2)*(self.T_Suth_ref+self.S_Suth)/(self.T_inf+self.S_Suth) ## [Pa s] | [N s m^-2]
        
        self.nu_inf = self.mu_inf / self.rho_inf ## kinematic viscosity [m²/s] --> momentum diffusivity
        
        # ===
        
        if self.verbose: print(72*'-')
        if self.verbose: even_print('Ma'          , '%0.2f [-]'           % self.Ma          )
        if self.verbose: even_print('Re'          , '%0.1f [-]'           % self.Re          )
        if self.verbose: even_print('Pr'          , '%0.3f [-]'           % self.Pr          )
        if self.verbose: even_print('T_inf'       , '%0.3f [K]'           % self.T_inf       )
        if self.verbose: even_print('p_inf'       , '%0.1f [Pa]'          % self.p_inf       )
        if self.verbose: even_print('kappa'       , '%0.3f [-]'           % self.kappa       )
        if self.verbose: even_print('R'           , '%0.3f [J/(kg·K)]'    % self.R           )
        if self.verbose: even_print('mu_Suth_ref' , '%0.6E [kg/(m·s)]'    % self.mu_Suth_ref )
        if self.verbose: even_print('T_Suth_ref'  , '%0.2f [K]'           % self.T_Suth_ref  )
        if self.verbose: even_print('C_Suth'      , '%0.5e [kg/(m·s·√K)]' % self.C_Suth      )
        if self.verbose: even_print('S_Suth'      , '%0.2f [K]'           % self.S_Suth      )
        if self.verbose: print(72*'-')
        
        self.a_inf = np.sqrt(self.kappa*self.R*self.T_inf)           
        self.U_inf = self.Ma*self.a_inf
        self.cp    = self.R*self.kappa/(self.kappa-1.)
        self.cv    = self.cp/self.kappa                         
        self.r     = self.Pr**(1/3)                        
        self.Tw    = self.T_inf                            
        self.Taw   = self.T_inf + self.r*self.U_inf**2/(2*self.cp)        
        self.lchar = self.Re*self.nu_inf/self.U_inf
        
        if self.verbose: even_print('rho_inf' , '%0.3f [kg/m³]'    % self.rho_inf )
        if self.verbose: even_print('mu_inf'  , '%0.6E [kg/(m·s)]' % self.mu_inf  )
        if self.verbose: even_print('nu_inf'  , '%0.6E [m²/s]'     % self.nu_inf  )
        if self.verbose: even_print('a_inf'   , '%0.6f [m/s]'      % self.a_inf   )
        if self.verbose: even_print('U_inf'   , '%0.6f [m/s]'      % self.U_inf   )
        if self.verbose: even_print('cp'      , '%0.3f [J/(kg·K)]' % self.cp      )
        if self.verbose: even_print('cv'      , '%0.3f [J/(kg·K)]' % self.cv      )
        if self.verbose: even_print('r'       , '%0.6f [-]'        % self.r       )
        if self.verbose: even_print('Tw'      , '%0.3f [K]'        % self.Tw      )
        if self.verbose: even_print('Taw'     , '%0.3f [K]'        % self.Taw     )
        if self.verbose: even_print('lchar'   , '%0.6E [m]'        % self.lchar   )
        if self.verbose: print(72*'-'+'\n')
        
        # ===
        
        udef_char = [    'Ma',     'Re',     'Pr',     'kappa',    'R',    'p_inf',    'T_inf',    'C_Suth',    'mu_Suth_ref',    'T_Suth_ref' ]
        udef_real = [self.Ma , self.Re , self.Pr , self.kappa, self.R, self.p_inf, self.T_inf, self.C_Suth, self.mu_Suth_ref, self.T_Suth_ref  ]
        udef = dict(zip(udef_char, udef_real))
        
        return udef

# 2D & spanwise average functions
# ------------------------------------------------------------

def get_span_avg_data(path,**kwargs):
    '''
    get data from EAS4 files, return data as dict
    -----
    --> mean_flow_mpi.eas
    --> favre_mean_flow_mpi.eas
    --> ext_rms_fluctuation_mpi.eas
    --> ext_favre_fluctuation_mpi.eas
    --> turbulent_budget_mpi.eas
    -----
    - assumes all 5x files are present
    '''
    dz = kwargs.get('dz',None) ## dz should be input as dimless (output from tgg) --> gets dimensionalized during this func!
    nz = kwargs.get('nz',None)
    dt = kwargs.get('dt',None)
    
    dataFolder        = Path(path) ## pathlib
    fname_Re_mean     = Path(dataFolder, 'mean_flow_mpi.eas')
    fname_Favre_mean  = Path(dataFolder, 'favre_mean_flow_mpi.eas')
    fname_Re_fluct    = Path(dataFolder, 'ext_rms_fluctuation_mpi.eas')
    fname_Favre_fluct = Path(dataFolder, 'ext_favre_fluctuation_mpi.eas')
    fname_turb_budget = Path(dataFolder, 'turbulent_budget_mpi.eas')
    
    data = {} ## the container dict that will be returned
    
    if (dt is not None):
        data['dt'] = dt
    if (nz is not None):
        data['nz'] = nz
    
    if fname_Re_mean.exists():
        print('--r-> %s'%fname_Re_mean.name)
        with eas4(str(fname_Re_mean),'r',verbose=False) as f1:
            
            meanData = f1.get_mean()
            
            Reyn_mean_total_avg_time               = f1.total_avg_time
            Reyn_mean_total_avg_iter_count         = f1.total_avg_iter_count
            Reyn_mean_dt                           = Reyn_mean_total_avg_time/Reyn_mean_total_avg_iter_count
            data['Reyn_mean_total_avg_time']       = Reyn_mean_total_avg_time
            data['Reyn_mean_total_avg_iter_count'] = Reyn_mean_total_avg_iter_count
            data['Reyn_mean_dt']                   = Reyn_mean_dt
            
            t_meas = f1.total_avg_time * (f1.lchar/f1.U_inf) ## dimensional [s]
            data['t_meas'] = t_meas
            
            nx = f1.nx                   ; data['nx']          = nx
            ny = f1.ny                   ; data['ny']          = ny
            
            Ma          = f1.Ma          ; data['Ma']          = Ma
            Re          = f1.Re          ; data['Re']          = Re
            Pr          = f1.Pr          ; data['Pr']          = Pr
            T_inf       = f1.T_inf       ; data['T_inf']       = T_inf
            p_inf       = f1.p_inf       ; data['p_inf']       = p_inf
            kappa       = f1.kappa       ; data['kappa']       = kappa
            R           = f1.R           ; data['R']           = R
            mu_Suth_ref = f1.mu_Suth_ref ; data['mu_Suth_ref'] = mu_Suth_ref
            T_Suth_ref  = f1.T_Suth_ref  ; data['T_Suth_ref']  = T_Suth_ref
            C_Suth      = f1.C_Suth      ; data['C_Suth']      = C_Suth
            S_Suth      = f1.S_Suth      ; data['S_Suth']      = S_Suth
            
            rho_inf = f1.rho_inf ; data['rho_inf'] = rho_inf
            mu_inf  = f1.mu_inf  ; data['mu_inf']  = mu_inf 
            nu_inf  = f1.nu_inf  ; data['nu_inf']  = nu_inf 
            a_inf   = f1.a_inf   ; data['a_inf']   = a_inf  
            U_inf   = f1.U_inf   ; data['U_inf']   = U_inf  
            cp      = f1.cp      ; data['cp']      = cp     
            cv      = f1.cv      ; data['cv']      = cv     
            r       = f1.r       ; data['r']       = r      
            Tw      = f1.Tw      ; data['Tw']      = Tw     
            Taw     = f1.Taw     ; data['Taw']     = Taw    
            lchar   = f1.lchar   ; data['lchar']   = lchar  
            
            xs = np.copy(f1.x) ; data['xs'] = xs ## dimensionless (inlet)
            ys = np.copy(f1.y) ; data['ys'] = ys ## dimensionless (inlet)
            
            x = f1.x * lchar ; data['x'] = x ## dimensional [m]
            y = f1.y * lchar ; data['y'] = y ## dimensional [m]
            
            xxs, yys = np.meshgrid(f1.x, f1.y, indexing='ij') ; data['xxs'] = xxs ; data['yys'] = yys ## dimensionless (inlet)
            xx,  yy  = np.meshgrid(x,    y,    indexing='ij') ; data['xx']  = xx  ; data['yy']  = yy  ## dimensional
            
            dx = np.insert(np.diff(x,n=1), 0, 0.) ; data['dx'] = dx ## 1D
            dy = np.insert(np.diff(y,n=1), 0, 0.) ; data['dy'] = dy
            
            if dz is not None: ## dimensionalize
                dz = dz * f1.lchar ; data['dz'] = dz ## 0D (float)
            
            np.testing.assert_allclose(np.cumsum(dx), x, rtol=1e-8)
            np.testing.assert_allclose(np.cumsum(dy), y, rtol=1e-8)
            
            dxx = np.broadcast_to(dx, (ny,nx)).T ; data['dxx'] = dxx ## 2D
            dyy = np.broadcast_to(dy, (nx,ny))   ; data['dyy'] = dyy
            
            # if dz is not None:
            #     dzz = dz * np.ones((nx,ny), dtype=np.float64) ## 2D but all == dz
            
            ### dxx_ = np.concatenate([np.zeros((1,ny)), np.diff(xx,axis=0)], axis=0)
            ### dyy_ = np.concatenate([np.zeros((nx,1)), np.diff(yy,axis=1)], axis=1)
            ### np.testing.assert_allclose(dxx, dxx_, rtol=1e-8)
            ### np.testing.assert_allclose(dyy, dyy_, rtol=1e-8)
            
            # === redimensionalize quantities (inlet scaling)
            u   = meanData['u']   * U_inf                ; data['u']   = u
            v   = meanData['v']   * U_inf                ; data['v']   = v
            w   = meanData['w']   * U_inf                ; data['w']   = w
            rho = meanData['rho'] * rho_inf              ; data['rho'] = rho
            p   = meanData['p']   * (rho_inf * U_inf**2) ; data['p']   = p
            T   = meanData['T']   * T_inf                ; data['T']   = T
            mu  = meanData['mu']  * mu_inf               ; data['mu']  = mu
            M   = u / np.sqrt(kappa * R * T)             ; data['M']   = M
            nu  = mu / rho                               ; data['nu']  = nu
            
            # ===== verify : mu == Suth(T)
            mu_from_Suth_1 = (14.58e-7 * T**1.5) / (T+110.4)
            mu_from_Suth_2 = C_Suth*T**(3/2)/(T+S_Suth)
            if not np.allclose(mu_from_Suth_1, mu_from_Suth_2, rtol=1e-8):
                raise AssertionError('Sutherland inconsistency')
            # if not np.allclose(mu, mu_from_Suth_2, rtol=1e-8): # --> fails : rtol_max = ~0.0018
            #     raise AssertionError('mu != Suth(T)')
            # np.testing.assert_allclose(mu, mu_from_Suth_1, rtol=1e-8) # --> fails : rtol_max = ~0.0018
            # np.testing.assert_allclose(mu, mu_from_Suth_2, rtol=1e-8) # --> fails : rtol_max = ~0.0018
            # =====
            
            # === gradients with O3 Spline + natural BCs
            hiOrder=True
            if hiOrder:
                dudx = np.zeros(shape=(nx,ny), dtype=np.float64)
                dudy = np.zeros(shape=(nx,ny), dtype=np.float64)
                dvdx = np.zeros(shape=(nx,ny), dtype=np.float64)
                dvdy = np.zeros(shape=(nx,ny), dtype=np.float64)
                dTdx = np.zeros(shape=(nx,ny), dtype=np.float64)
                dTdy = np.zeros(shape=(nx,ny), dtype=np.float64)
                dpdx = np.zeros(shape=(nx,ny), dtype=np.float64)
                dpdy = np.zeros(shape=(nx,ny), dtype=np.float64)
                for i in range(nx):
                    dudy[i,:] = sp.interpolate.CubicSpline(y,u[i,:],bc_type='natural')(y,1)
                    dvdy[i,:] = sp.interpolate.CubicSpline(y,v[i,:],bc_type='natural')(y,1)
                    dTdy[i,:] = sp.interpolate.CubicSpline(y,T[i,:],bc_type='natural')(y,1)
                    dpdy[i,:] = sp.interpolate.CubicSpline(y,p[i,:],bc_type='natural')(y,1)
                for j in range(ny):
                    dudx[:,j] = sp.interpolate.CubicSpline(x,u[:,j],bc_type='natural')(x,1)
                    dvdx[:,j] = sp.interpolate.CubicSpline(x,v[:,j],bc_type='natural')(x,1)
                    dTdx[:,j] = sp.interpolate.CubicSpline(x,T[:,j],bc_type='natural')(x,1)
                    dpdx[:,j] = sp.interpolate.CubicSpline(x,p[:,j],bc_type='natural')(x,1)
            else: ### numpy.gradient() --> 1st or 2nd order
                dudy = np.gradient(u, y, axis=1, edge_order=1)
                dvdy = np.gradient(v, y, axis=1, edge_order=1)
                dudx = np.gradient(u, x, axis=0, edge_order=1)
                dvdx = np.gradient(v, x, axis=0, edge_order=1)
            
            # === verify fderiv module & effect on dudy --> extremely minimal difference w.r.t sp.interpolate.CubicSpline()
            if False:
                
                sys.path.append('T:/phd/work/iagappel-beta')
                import postproc_turb.PYTHON_AUX.differentiate as df
                
                dudy_fderiv = np.zeros(shape=(nx,ny), dtype=np.float64)
                for i in range(nx):
                    dudy_fderiv[i,:] = np.transpose(df.fderiv(np.transpose(u[i,:]), 'dim1', y, ny))
                
                np.testing.assert_allclose(dudy[:,0], dudy_fderiv[:,0], rtol=3e-2)
                np.testing.assert_allclose(dudy[:,1], dudy_fderiv[:,1], rtol=3e-2)
                print('check passed : fderiv within 3%')
                
                plt.close('all')
                fig1 = plt.figure(frameon=True, figsize=(3.2*1.5, 3.2*1.5/(24/15*2)), dpi=320) ## powerpoint half height
                ax1 = plt.gca()
                #ax1.set_yscale('log', base=10)
                ln1, = ax1.plot(np.array(range(nx)), dudy[:,1],        c='red',  zorder=20)
                ln1, = ax1.plot(np.array(range(nx)), dudy_fderiv[:,1], c='blue', zorder=20)
                fig1.tight_layout(pad=0.15)
                #fig1.savefig('dudy_compare.png', pad_inches=0.15, dpi=png_px_x/plt.gcf().get_size_inches()[0])
                plt.show()
                
                plt.close('all')
                fig1 = plt.figure(frameon=True, figsize=(3.2*1.5, 3.2*1.5/(24/15*2)), dpi=320) ## powerpoint half height
                ax1 = plt.gca()
                #ax1.set_yscale('log', base=10)
                ln1, = ax1.plot(np.array(range(ny)), dudy[500,:],        c='red',  zorder=20)
                ln1, = ax1.plot(np.array(range(ny)), dudy_fderiv[500,:], c='blue', zorder=20)
                fig1.tight_layout(pad=0.15)
                #fig1.savefig('dudy_compare2.png', pad_inches=0.15, dpi=png_px_x/plt.gcf().get_size_inches()[0])
                plt.show()
            
            vort_z = dvdx - dudy
            
            data['dudx']   = dudx
            data['dudy']   = dudy
            data['dvdx']   = dvdx
            data['dvdy']   = dvdy
            data['dTdx']   = dTdx
            data['dTdy']   = dTdy
            data['dpdx']   = dpdx
            data['dpdy']   = dpdy
            data['vort_z'] = vort_z
            
            # === wall-adjacent values
            dudy_wall = dudy[:,0]
            rho_wall  = rho[:,0]
            nu_wall   = nu[:,0]
            mu_wall   = mu[:,0]
            T_wall    = T[:,0]
            tau_wall  = mu_wall * dudy_wall
            q_wall    = cp * mu_wall / Pr * dTdy[:,0] ### wall heat flux
            
            data['dudy_wall'] = dudy_wall
            data['rho_wall']  = rho_wall
            data['nu_wall']   = nu_wall
            data['mu_wall']   = mu_wall
            data['T_wall']    = T_wall
            data['tau_wall']  = tau_wall
            data['q_wall']    = q_wall
            
            u_tau  = np.sqrt(tau_wall/rho_wall)
            y_plus = y * u_tau[:,np.newaxis] / nu_wall[:,np.newaxis]
            u_plus = u / u_tau[:,np.newaxis]
            M_tau  = u_tau / np.sqrt(kappa * R * T_wall)
            
            data['u_tau']   = u_tau
            data['y_plus']  = y_plus
            data['u_plus']  = u_plus
            data['M_tau']   = M_tau
            
            # === BL edge & 99 values
            j_edge     = np.zeros(shape=(nx,), dtype=np.int64)
            y_edge     = np.zeros(shape=(nx,), dtype=np.float64)
            u_edge     = np.zeros(shape=(nx,), dtype=np.float64)
            v_edge     = np.zeros(shape=(nx,), dtype=np.float64)
            T_edge     = np.zeros(shape=(nx,), dtype=np.float64)
            p_edge     = np.zeros(shape=(nx,), dtype=np.float64)
            rho_edge   = np.zeros(shape=(nx,), dtype=np.float64)
            nu_edge    = np.zeros(shape=(nx,), dtype=np.float64)
            psvel_edge = np.zeros(shape=(nx,), dtype=np.float64)
            M_edge     = np.zeros(shape=(nx,), dtype=np.float64)
            
            d99     = np.zeros(shape=(nx,), dtype=np.float64) ## δ₉₉ --> interpolated
            d99j    = np.zeros(shape=(nx,), dtype=np.int64)   ## closest y-index to δ₉₉
            d99g    = np.zeros(shape=(nx,), dtype=np.float64) ## δ₉₉ at nearest grid point
            
            u99     = np.zeros(shape=(nx,), dtype=np.float64)
            v99     = np.zeros(shape=(nx,), dtype=np.float64)
            T99     = np.zeros(shape=(nx,), dtype=np.float64)
            p99     = np.zeros(shape=(nx,), dtype=np.float64)
            rho99   = np.zeros(shape=(nx,), dtype=np.float64)
            nu99    = np.zeros(shape=(nx,), dtype=np.float64)
            psvel99 = np.zeros(shape=(nx,), dtype=np.float64)
            M99     = np.zeros(shape=(nx,), dtype=np.float64)
            
            # === get pseudo-velocity (wall-normal integration of z-vorticity)
            psvel = np.zeros(shape=(nx,ny), dtype=np.float64)
            for i in range(nx):
                psvel[i,:] = sp.integrate.cumtrapz(-1*vort_z[i,:], y, initial=0.)
            
            ## low order
            #psvel_ddy = np.gradient(psvel, y, edge_order=1, axis=1) ## 1st order
            
            ## high order
            psvel_ddy = np.zeros(shape=(nx,ny), dtype=np.float64)
            for i in range(nx):
                psvel_ddy[i,:] = sp.interpolate.CubicSpline(y,psvel[i,:],bc_type='natural')(y,1)
            
            ## u-driven criteria
            if False:
                for i in range(nx):
                    #umax=np.copy(u[i,:]).max()
                    umax=u[i,-1]
                    for j in range(ny):
                        if math.isclose(u[i,j], umax, rel_tol=1e-4) or (u[i,j]>=umax):
                            j_edge_3[i] = j
            
            j_edge_1 = np.argmax(psvel_ddy<=1000., axis=1)            ## (index of) threshold on ddy(psvel)
            j_edge_2 = np.argmax(psvel, axis=1)                       ## (index of) max of pseudovel
            j_edge   = np.amin(np.stack((j_edge_1,j_edge_2)), axis=0) ## minimum of collective minima
            
            # === populate edge arrays (always grid snapped)
            for i in range(nx):
                je = j_edge[i]
                y_edge[i]     = y[je]
                u_edge[i]     = u[i,je]
                v_edge[i]     = v[i,je]
                T_edge[i]     = T[i,je]
                p_edge[i]     = p[i,je]
                rho_edge[i]   = rho[i,je]
                nu_edge[i]    = nu[i,je]
                psvel_edge[i] = psvel[i,je]
                M_edge[i]     = M[i,je]
            
            data['j_edge']     = j_edge
            data['y_edge']     = y_edge
            data['u_edge']     = u_edge
            data['v_edge']     = v_edge
            data['T_edge']     = T_edge
            data['p_edge']     = p_edge
            data['rho_edge']   = rho_edge
            data['nu_edge']    = nu_edge
            data['psvel_edge'] = psvel_edge
            data['M_edge']     = M_edge
            
            # === get d99 interpolated values
            for i in range(nx):
                
                ## populate d99 arrays with default 'grid-snapped' values
                for j in range(ny):
                    #if (u[i,j] >= 0.99*u_edge[i]):
                    if (psvel[i,j] >= 0.99*psvel_edge[i]):
                        d99j[i] =   j-1
                        d99[i]  = y[j-1]
                        d99g[i] = y[j-1] ## at grid
                        break
                
                # === use spline interpolation to find d99
                
                je = j_edge[i]+2 ## add a couple points to accurately loft a higher order spline
                
                ## splrep --> this is CubicSpline with 'not-a-knot' BCs
                if False:
                    #u_spl = sp.interpolate.splrep(y[:je], u[i,:je]-(0.99*u_edge[i]), k=3, s=0.)
                    #roots = sp.interpolate.sproot(u_spl)
                    psvel_spl = sp.interpolate.splrep(y[:je],psvel[i,:je]-(0.99*psvel_edge[i]), k=3, s=0.)
                    roots = sp.interpolate.sproot(psvel_spl)
                
                ## Piecewise Cubic Hermite Interpolating Polynomial : 'pchip'
                if False: 
                    #u_spl = sp.interpolate.pchip(y[:je], u[i,:je]-(0.99*u_edge[i]))
                    #roots = u_spl.roots(discontinuity=False,extrapolate=False)
                    psvel_spl = sp.interpolate.pchip(y[:je],psvel[i,:je]-(0.99*psvel_edge[i]))
                    roots = psvel_spl.roots(discontinuity=False,extrapolate=False)
                
                ## Cubic Spline O3 --> this is the best + allows for explicit BCs
                if True:
                    #u_spl = sp.interpolate.CubicSpline(y[:je],u[i,:je]-(0.99*u_edge[i]),bc_type='natural')
                    #roots = u_spl.roots(discontinuity=False,extrapolate=False)
                    psvel_spl = sp.interpolate.CubicSpline(y[:je],psvel[i,:je]-(0.99*psvel_edge[i]),bc_type='natural')
                    roots = psvel_spl.roots(discontinuity=False,extrapolate=False)
                
                # === check roots
                
                if (roots.size>0):
                    d99_ = roots[0]
                    if (d99_<y_edge[i]): ## dont let it be greater than max location
                        if True:
                            d99[i]  =   d99_
                            d99j[i] =   np.abs(y-d99_).argmin()  ## closest index to interped value
                            d99g[i] = y[np.abs(y-d99_).argmin()] ## d99 at nearest grid point (overwrite)
                        else:
                            pass ## take default (grid-snapped d99)
                    else:
                        print('root is > max! : xi=%i'%i)
                else:
                    d99_ = d99[i]
                    #warnings.warn('no root found at xi=%i'%i, category=UserWarning)
                    print('no root found at xi=%i'%i)
                
                # ===
                
                u99[i]     = sp.interpolate.interp1d(y[:je], u[i,:je]    )(d99_)
                rho99[i]   = sp.interpolate.interp1d(y[:je], rho[i,:je]  )(d99_)
                nu99[i]    = sp.interpolate.interp1d(y[:je], nu[i,:je]   )(d99_)
                T99[i]     = sp.interpolate.interp1d(y[:je], T[i,:je]    )(d99_)
                p99[i]     = sp.interpolate.interp1d(y[:je], p[i,:je]    )(d99_)
                v99[i]     = sp.interpolate.interp1d(y[:je], v[i,:je]    )(d99_)
                psvel99[i] = sp.interpolate.interp1d(y[:je], psvel[i,:je])(d99_)
                M99[i]     = sp.interpolate.interp1d(y[:je], M[i,:je]    )(d99_)
            
            data['u99']     = u99
            data['rho99']   = rho99
            data['nu99']    = nu99
            data['T99']     = T99
            data['p99']     = p99
            data['v99']     = v99
            data['psvel99'] = psvel99
            data['M99']     = M99
            
            data['d99']     = d99
            data['d99j']    = d99j
            data['d99g']    = d99g
            
            # === outer scales
            data['sc_l_out'] = np.copy(d99)
            data['sc_u_out'] = np.copy(u99)
            data['sc_t_out'] = np.copy(u99/d99)
            
            # === eddy scale
            data['sc_t_eddy'] = np.copy(u_tau/d99)
            
            # === θ, δ*, Re_θ, Re_τ
            Re_theta      = np.zeros(shape=(nx,), dtype=np.float64)
            Re_theta_wall = np.zeros(shape=(nx,), dtype=np.float64)
            Re_tau        = np.zeros(shape=(nx,), dtype=np.float64)
            Re_d99        = np.zeros(shape=(nx,), dtype=np.float64)
            Re_x          = np.zeros(shape=(nx,), dtype=np.float64)
            H12           = np.zeros(shape=(nx,), dtype=np.float64)
            H12_inc       = np.zeros(shape=(nx,), dtype=np.float64)
            theta         = np.zeros(shape=(nx,), dtype=np.float64)
            dstar         = np.zeros(shape=(nx,), dtype=np.float64)
            
            # === Van Driest scaled u
            u_vd = np.zeros(shape=(nx,ny), dtype=np.float64) 
            
            for i in range(nx):
                
                je   = j_edge[i]
                yl   = np.copy(    y[:je+1])
                ul   = np.copy(  u[i,:je+1])
                rhol = np.copy(rho[i,:je+1])
                
                integrand_theta_inc = (ul/u_edge[i])*(1-(ul/u_edge[i]))
                integrand_dstar_inc = 1-(ul/u_edge[i])
                theta_inc           = sp.integrate.trapezoid(integrand_theta_inc, x=yl)
                dstar_inc           = sp.integrate.trapezoid(integrand_dstar_inc, x=yl)
                
                integrand_theta_cmp = (ul*rhol)/(u_edge[i]*rho_edge[i])*(1-(ul/u_edge[i]))
                integrand_dstar_cmp = (1-((ul*rhol)/(u_edge[i]*rho_edge[i])))
                theta_cmp           = sp.integrate.trapezoid(integrand_theta_cmp, x=yl)
                dstar_cmp           = sp.integrate.trapezoid(integrand_dstar_cmp, x=yl)
                
                integrand_u_vd   = np.sqrt(T_wall[i]/T[i,:])
                u_vd[i,:]        = sp.integrate.cumtrapz(integrand_u_vd, u[i,:], initial=0)
                
                theta[i]         = theta_cmp
                dstar[i]         = dstar_cmp
                H12[i]           = dstar_cmp/theta_cmp
                H12_inc[i]       = dstar_inc/theta_inc
                Re_tau[i]        = d99[i]*u_tau[i]/nu_wall[i]
                Re_theta[i]      = theta_cmp*u_edge[i]/nu_edge[i]
                Re_theta_wall[i] = rho_edge[i]*theta_cmp*u_edge[i]/mu_wall[i]
                Re_d99[i]        = d99[i]*u_edge[i]/nu_edge[i]
                Re_x[i]          = u_edge[i]*(x[i]-x[0])/nu_edge[i]
            
            data['Re_theta']      = Re_theta
            data['Re_theta_wall'] = Re_theta_wall
            data['Re_tau']        = Re_tau       
            data['Re_d99']        = Re_d99
            data['Re_x']          = Re_x
            data['H12']           = H12
            data['H12_inc']       = H12_inc
            data['theta']         = theta
            data['dstar']         = dstar
            
            # === Van Driest scaled velocity (wall units)
            u_plus_vd = u_vd / u_tau[:,np.newaxis] 
            data['u_plus_vd'] = u_plus_vd
            
            # === inner scales
            sc_u_in = np.copy( u_tau              ) ; data['sc_u_in'] = sc_u_in
            sc_l_in = np.copy( nu_wall / u_tau    ) ; data['sc_l_in'] = sc_l_in
            sc_t_in = np.copy( nu_wall / u_tau**2 ) ; data['sc_t_in'] = sc_t_in
            #np.testing.assert_allclose(sc_t_in, sc_l_in/sc_u_in, rtol=1e-8)
            
            dy_plus  = dyy              * u_tau[:,np.newaxis] / nu_wall[:,np.newaxis]
            dy_plus_ = dy[np.newaxis,:] * u_tau[:,np.newaxis] / nu_wall[:,np.newaxis] ## 2D field
            np.testing.assert_allclose(dy_plus_, dy_plus, rtol=1e-8)
            
            #dx_plus  = dxx * u_tau[:,np.newaxis]/nu_wall[:,np.newaxis] ## 2D but uniform in y / axis=1
            dx_plus  = dx * u_tau / nu_wall ## 1D
            
            if dz is not None:
                #dz_plus  = dzz  * u_tau[:,np.newaxis]/nu_wall[:,np.newaxis] ## 2D but uniform in y / axis=1
                dz_plus  = dz * u_tau / nu_wall ## 1D
            
            dy_plus_wall = np.copy(dy_plus[:,1])
            
            data['dx_plus']      = dx_plus
            data['dy_plus']      = dy_plus
            data['dy_plus_wall'] = dy_plus_wall
            if dz is not None:
                data['dz_plus']  = dz_plus
            
            # === curve fit : Re_θ(x)
            
            ### def func_curve(x, a, b, c, d):
            ###     return a + x**b + c*x**d
            ### popt, pcov = sp.optimize.curve_fit(func_curve, xs[xe1:xe2], Re_theta[xe1:xe2], maxfev=100000)
            ### # print(list(popt)) ## coefficients
            ### #x_trend = np.linspace(xs[xe1],2000,1000) ## extrapolate
            ### x_trend = np.linspace(xs[xe1],xs[xe2],1000)
            ### Re_theta_trend = func_curve(x_trend, *popt)
            
            # ===
            
            dy_plus_99 = np.zeros(shape=(nx,), dtype=np.float64)
            for i in range(nx):
                je = d99j[i]
                je_plus_5 = je+5 ## add some pts for high-order spline
                dy_plus_99[i]  = sp.interpolate.interp1d(y[:je_plus_5], dy_plus[i,:je_plus_5])(d99[i])
            data['dy_plus_99'] = dy_plus_99
            
            pass
    
    if fname_Favre_mean.exists():
        print('--r-> %s'%fname_Favre_mean.name)
        with eas4(str(fname_Favre_mean),'r',verbose=False) as f1:
            
            meanData = f1.get_mean()
            
            Favre_mean_total_avg_time       = f1.total_avg_time
            Favre_mean_total_avg_iter_count = f1.total_avg_iter_count
            data['Favre_mean_total_avg_time'] = Favre_mean_total_avg_time
            data['Favre_mean_total_avg_iter_count'] = Favre_mean_total_avg_iter_count
            
            u_favre   = meanData['u']   * U_inf              ; data['u_favre']   = u_favre
            v_favre   = meanData['v']   * U_inf              ; data['v_favre']   = v_favre
            w_favre   = meanData['w']   * U_inf              ; data['w_favre']   = w_favre
            rho_favre = meanData['rho'] * rho_inf            ; data['rho_favre'] = rho_favre
            T_favre   = meanData['T']   * T_inf              ; data['T_favre']   = T_favre
            p_favre   = meanData['p']   * rho_inf * U_inf**2 ; data['p_favre']   = p_favre
            mu_favre  = meanData['mu']  * mu_inf             ; data['mu_favre']  = mu_favre
            uu_favre  = meanData['uu']  * U_inf**2           ; data['uu_favre']  = uu_favre
            uv_favre  = meanData['uv']  * U_inf**2           ; data['uv_favre']  = uv_favre
            
            ### # === gradients with O3 Spline + natural BCs
            ### if True:
            ###     dudy_favre = np.zeros(shape=(nx,ny), dtype=np.float64)
            ###     for i in range(nx):
            ###         dudy_favre[i,:] = sp.interpolate.CubicSpline(y,u_favre[i,:],bc_type='natural')(y,1)
            ### data['dudy_favre'] = dudy_favre
            ### 
            ### dudy_wall_favre         = dudy_favre[:,0]
            ### mu_wall_favre           = mu_favre[:,0]
            ### rho_wall_favre          = rho_favre[:,0]
            ### 
            ### data['dudy_wall_favre'] = dudy_wall_favre
            ### data['mu_wall_favre']   = mu_wall_favre
            ### data['rho_wall_favre']  = rho_wall_favre
            ### 
            ### tau_wall_favre  = mu_wall_favre * dudy_wall_favre
            ### u_tau_favre     = np.sqrt(tau_wall_favre/rho_wall_favre)
            ### data['tau_wall_favre']  = tau_wall_favre
            ### data['u_tau_favre']     = u_tau_favre
            pass
    
    if fname_Re_fluct.exists():
        print('--r-> %s'%fname_Re_fluct.name)
        with eas4(str(fname_Re_fluct),'r',verbose=False) as f1:
            
            meanData = f1.get_mean()
            
            Reyn_fluct_total_avg_time       = f1.total_avg_time
            Reyn_fluct_total_avg_iter_count = f1.total_avg_iter_count
            data['Reyn_fluct_total_avg_time'] = Reyn_fluct_total_avg_time
            data['Reyn_fluct_total_avg_iter_count'] = Reyn_fluct_total_avg_iter_count 
            
            uI_uI = meanData["u'u'"] * U_inf**2                ; data['uI_uI']   = uI_uI  
            vI_vI = meanData["v'v'"] * U_inf**2                ; data['vI_vI']   = vI_vI  
            wI_wI = meanData["w'w'"] * U_inf**2                ; data['wI_wI']   = wI_wI  
            uI_vI = meanData["u'v'"] * U_inf**2                ; data['uI_vI']   = uI_vI  
            uI_wI = meanData["u'w'"] * U_inf**2                ; data['uI_wI']   = uI_wI  
            vI_wI = meanData["v'w'"] * U_inf**2                ; data['vI_wI']   = vI_wI  
            
            uI_TI = meanData["u'T'"] * (U_inf*T_inf)           ; data['uI_TI']   = uI_TI  
            vI_TI = meanData["v'T'"] * (U_inf*T_inf)           ; data['vI_TI']   = vI_TI  
            wI_TI = meanData["w'T'"] * (U_inf*T_inf)           ; data['wI_TI']   = wI_TI  
            
            TI_TI = meanData["T'T'"] * T_inf**2                ; data['TI_TI']   = TI_TI  
            pI_pI = meanData["p'p'"] * (rho_inf * U_inf**2)**2 ; data['pI_pI']   = pI_pI  
            rI_rI = meanData["r'r'"] * rho_inf**2              ; data['rI_rI']   = rI_rI  
            muI_muI = meanData["mu'mu'"] * mu_inf**2           ; data['muI_muI'] = muI_muI
            
            # === verify : mu_inf == Suth(T_inf)
            mu_inf_in_file = mu_inf
            mu_inf_from_T_inf_1 = (14.58e-7 * T_inf**1.5) / (T_inf+110.4)
            mu_inf_from_T_inf_2 = C_Suth*T_inf**(3/2)/(T_inf+S_Suth)
            
            if not np.isclose(mu_inf_from_T_inf_1, mu_inf_from_T_inf_2, rtol=1e-8):
                raise AssertionError('Sutherland inconsistency for air --> check')
            if not np.isclose(mu_inf_in_file, mu_inf_from_T_inf_2, rtol=1e-8):
                raise AssertionError('mu_inf != Suth(T_inf)')
            
            # === fluctuating shear stresses
            tauI_xx = meanData["tau'_xx"] * mu_inf * U_inf / lchar ; data['tauI_xx'] = tauI_xx
            tauI_yy = meanData["tau'_yy"] * mu_inf * U_inf / lchar ; data['tauI_yy'] = tauI_yy
            tauI_zz = meanData["tau'_zz"] * mu_inf * U_inf / lchar ; data['tauI_zz'] = tauI_zz
            tauI_xy = meanData["tau'_xy"] * mu_inf * U_inf / lchar ; data['tauI_xy'] = tauI_xy
            tauI_xz = meanData["tau'_xz"] * mu_inf * U_inf / lchar ; data['tauI_xz'] = tauI_xz
            tauI_yz = meanData["tau'_yz"] * mu_inf * U_inf / lchar ; data['tauI_yz'] = tauI_yz
            
            # ===== RMS values
            uI_uI_rms = np.sqrt(       meanData["u'u'"]  * U_inf**2 )                               ; data['uI_uI_rms']   = uI_uI_rms
            vI_vI_rms = np.sqrt(       meanData["v'v'"]  * U_inf**2 )                               ; data['vI_vI_rms']   = vI_vI_rms
            wI_wI_rms = np.sqrt(       meanData["w'w'"]  * U_inf**2 )                               ; data['wI_wI_rms']   = wI_wI_rms
            uI_vI_rms = np.sqrt(np.abs(meanData["u'v'"]) * U_inf**2 ) * np.sign(meanData["u'v'"])   ; data['uI_vI_rms']   = uI_vI_rms
            uI_wI_rms = np.sqrt(np.abs(meanData["u'w'"]) * U_inf**2 ) * np.sign(meanData["u'w'"])   ; data['uI_wI_rms']   = uI_wI_rms
            vI_wI_rms = np.sqrt(np.abs(meanData["v'w'"]) * U_inf**2 ) * np.sign(meanData["v'w'"])   ; data['vI_wI_rms']   = vI_wI_rms
            
            uI_TI_rms = np.sqrt(np.abs(meanData["u'T'"]) * U_inf*T_inf) * np.sign(meanData["u'T'"]) ; data['uI_TI_rms']   = uI_TI_rms
            vI_TI_rms = np.sqrt(np.abs(meanData["v'T'"]) * U_inf*T_inf) * np.sign(meanData["v'T'"]) ; data['vI_TI_rms']   = vI_TI_rms
            wI_TI_rms = np.sqrt(np.abs(meanData["w'T'"]) * U_inf*T_inf) * np.sign(meanData["w'T'"]) ; data['wI_TI_rms']   = wI_TI_rms
            
            rI_rI_rms   = np.sqrt( meanData["r'r'"]   * rho_inf**2 )                                ; data['rI_rI_rms']   = rI_rI_rms
            TI_TI_rms   = np.sqrt( meanData["T'T'"]   * T_inf**2 )                                  ; data['TI_TI_rms']   = TI_TI_rms
            pI_pI_rms   = np.sqrt( meanData["p'p'"]   * (rho_inf * U_inf**2)**2 )                   ; data['pI_pI_rms']   = pI_pI_rms
            muI_muI_rms = np.sqrt( meanData["mu'mu'"] * mu_inf**2 )                                 ; data['muI_muI_rms'] = muI_muI_rms
            
            M_rms = uI_uI_rms / np.sqrt(kappa * R * T)                                              ; data['M_rms'] = M_rms
            
            # === flucutating wall shear
            uI_uI_ddy = np.zeros(shape=(nx,ny), dtype=np.float64)
            for i in range(nx):
                uI_uI_ddy[i,:] = sp.interpolate.CubicSpline(y,uI_uI_rms[i,:],bc_type='natural')(y,1)
            
            tau_uIuI_wall = mu[:,0] * uI_uI_ddy[:,0]
            
            data['uI_uI_ddy']     = uI_uI_ddy
            data['tau_uIuI_wall'] = tau_uIuI_wall
            
            tau_wall_rms = tau_uIuI_wall / (u_tau**2 * rho_wall)
            data['tau_wall_rms'] = tau_wall_rms
            
            # === Alfredsson et. al 1988 : tau_wall_rms is limit of local streamwise turbulence intensity as y-->0
            tau_wall_rms_lim = uI_uI_rms[:,1] / u[:,1]
            data['tau_wall_rms_lim'] = tau_wall_rms_lim
            
            # === turbulent kinetic energy
            tke = 0.5 * (uI_uI_rms**2 + vI_vI_rms**2 + wI_wI_rms**2)
            data['tke'] = tke
    
    if fname_Favre_fluct.exists():
        print('--r-> %s'%fname_Favre_fluct.name)
        with eas4(str(fname_Favre_fluct),'r',verbose=False) as f1:
            
            meanData = f1.get_mean()
            
            r_uII_uII   = meanData["r u''u''"]   * rho_inf * U_inf**2                ; data['r_uII_uII']   = r_uII_uII
            r_vII_vII   = meanData["r v''v''"]   * rho_inf * U_inf**2                ; data['r_vII_vII']   = r_vII_vII
            r_wII_wII   = meanData["r w''_w''"]  * rho_inf * U_inf**2                ; data['r_wII_wII']   = r_wII_wII
            r_uII_vII   = meanData["r u''v''"]   * rho_inf * U_inf**2                ; data['r_uII_vII']   = r_uII_vII
            r_uII_wII   = meanData["r u''w''"]   * rho_inf * U_inf**2                ; data['r_uII_wII']   = r_uII_wII
            r_vII_wII   = meanData["r w''v''"]   * rho_inf * U_inf**2                ; data['r_vII_wII']   = r_vII_wII
            
            r_uII_TII   = meanData["r u''T''"]   * rho_inf * U_inf * T_inf           ; data['r_uII_TII']   = r_uII_TII
            r_vII_TII   = meanData["r v''T''"]   * rho_inf * U_inf * T_inf           ; data['r_vII_TII']   = r_vII_TII
            r_wII_TII   = meanData["r w''T''"]   * rho_inf * U_inf * T_inf           ; data['r_wII_TII']   = r_wII_TII
            
            r_TII_TII   = meanData["r T''T''"]   * rho_inf * T_inf**2                ; data['r_TII_TII']   = r_TII_TII
            r_pII_pII   = meanData["r p''p''"]   * rho_inf * (rho_inf * U_inf**2)**2 ; data['r_pII_pII']   = r_pII_pII
            r_rII_rII   = meanData["r r''r''"]   * rho_inf * rho_inf**2              ; data['r_rII_rII']   = r_rII_rII
            r_muII_muII = meanData["r mu''mu''"] * rho_inf * mu_inf**2               ; data['r_muII_muII'] = r_muII_muII
            
            # =====
            
            tke_favre = 0.5 * (r_uII_uII + r_vII_vII + r_wII_wII) ### check implementation (Pirozzoli?)
            data['tke_favre'] = tke_favre
    
    if fname_turb_budget.exists():
        print('--r-> %s'%fname_turb_budget.name)
        with eas4(str(fname_turb_budget),'r',verbose=False) as f1:
            
            meanData = f1.get_mean()
            
            turb_budget_total_avg_time               = f1.total_avg_time
            turb_budget_total_avg_iter_count         = f1.total_avg_iter_count
            data['turb_budget_total_avg_time']       = turb_budget_total_avg_time
            data['turb_budget_total_avg_iter_count'] = turb_budget_total_avg_iter_count
            
            production     = meanData['prod.']     * U_inf**3 * rho_inf / lchar    ; data['production']     = production
            dissipation    = meanData['dis.']      * U_inf**2 * mu_inf  / lchar**2 ; data['dissipation']    = dissipation
            turb_transport = meanData['t-transp.'] * U_inf**3 * rho_inf / lchar    ; data['turb_transport'] = turb_transport
            visc_diffusion = meanData['v-diff.']   * U_inf**2 * mu_inf  / lchar**2 ; data['visc_diffusion'] = visc_diffusion
            p_diffusion    = meanData['p-diff.']   * U_inf**3 * rho_inf / lchar    ; data['p_diffusion']    = p_diffusion
            p_dilatation   = meanData['p-dilat.']  * U_inf**3 * rho_inf / lchar    ; data['p_dilatation']   = p_dilatation
            rho_terms      = meanData['rho-terms'] * U_inf**3 * rho_inf / lchar    ; data['rho_terms']      = rho_terms
    
    # === total enthalpy
    if 'tke' in globals():
        h_tot_mean               = cp * T       + 0.5*(u**2       + v**2                   ) + tke
        h_tot_mean_favre         = cp * T_favre + 0.5*(u_favre**2 + v_favre**2 + w_favre**2) + tke_favre/rho
        data['h_tot_mean']       = h_tot_mean
        data['h_tot_mean_favre'] = h_tot_mean_favre
    
    # === skin friction coefficient
    cf = 2.0 * (u_tau / u_edge)**2 * (rho_wall/rho_edge)
    data['cf'] = cf
    
    if 'dissipation' in locals():
        Kolm_len = (nu**3 / np.abs(dissipation))**(1/4)
        data['Kolm_len'] = Kolm_len
    
    # === get the boundaries of the log law & get n log law decades
    doGetLogLawBounds = True
    if doGetLogLawBounds:
        
        logLawTol = 0.05 ## 5% tolerance
        
        logLaw_yp_lo   = np.zeros(shape=(nx,), dtype=np.float64)
        logLaw_yp_md   = np.zeros(shape=(nx,), dtype=np.float64)
        logLaw_yp_hi   = np.zeros(shape=(nx,), dtype=np.float64)
        
        logLaw_decades = np.zeros(shape=(nx,), dtype=np.float64)
        
        logLaw_y_lo    = np.zeros(shape=(nx,), dtype=np.float64)
        logLaw_y_md    = np.zeros(shape=(nx,), dtype=np.float64)
        logLaw_y_hi    = np.zeros(shape=(nx,), dtype=np.float64)
        
        # === "decades of log layer" calc
        for xi in range(nx):
            aa = (1/0.41) * np.log(np.maximum(y_plus[xi,:],1e-16)) + 5.2 ## log law line
            bb = u_plus_vd[xi,:]
            cc = np.where(np.abs((bb-aa)/aa)<logLawTol)[0] ## indices where within tolerance of log law line --> includes intersection with wake!
            dd = cc[np.where(cc<=(d99j[xi]+2))] ## take only indices less than d99 index +2
            
            try:
                lli1  = dd[0]
                lli2  = dd[-1]
                llyp1 = y_plus[xi,:][lli1]
                llyp2 = y_plus[xi,:][lli2]
                llypm = 10**(0.5*(np.log10(llyp1)+np.log10(llyp2)))
                lldec = np.log10(llyp2)-np.log10(llyp1)
                logLaw_yp_lo[xi]   = llyp1
                logLaw_yp_md[xi]   = llypm
                logLaw_yp_hi[xi]   = llyp2
                logLaw_decades[xi] = lldec
                # ===
                #print('y+ log start : %0.1f'%llyp1)
                #print('y+ log end   : %0.1f'%llyp2)
                #print('y+ log mid   : %0.1f'%llypm)
                #print('log decades  : %0.3f'%lldec)
                # ===
                logLaw_y_lo[xi] = y[lli1]
                logLaw_y_md[xi] = 10**(0.5*(np.log10(y[lli1])+np.log10(y[lli2])))
                logLaw_y_hi[xi] = y[lli2]
            except IndexError: ## no log law present
                logLaw_yp_lo[xi]   = np.nan
                logLaw_yp_md[xi]   = np.nan
                logLaw_yp_hi[xi]   = np.nan
                logLaw_decades[xi] = np.nan
                logLaw_y_lo[xi]    = np.nan
                logLaw_y_md[xi]    = np.nan
                logLaw_y_hi[xi]    = np.nan
        
        data['logLaw_yp_lo']   = logLaw_yp_lo
        data['logLaw_yp_md']   = logLaw_yp_md
        data['logLaw_yp_hi']   = logLaw_yp_hi
        data['logLaw_decades'] = logLaw_decades
        data['logLaw_y_lo']    = logLaw_y_lo
        data['logLaw_y_md']    = logLaw_y_md
        data['logLaw_y_hi']    = logLaw_y_hi
    
    # === get the Kolmogorov resolution
    doGetKolmRes = True
    if doGetKolmRes and ('Kolm_len' in locals()):
        Kolm_res_dx = dxx / Kolm_len ; data['Kolm_res_dx'] = Kolm_res_dx
        Kolm_res_dy = dyy / Kolm_len ; data['Kolm_res_dy'] = Kolm_res_dy
        Kolm_res_dz = dz  / Kolm_len ; data['Kolm_res_dz'] = Kolm_res_dz
        
        # === the Kolmogorov length & (relative) resolution @ d99
        
        Kolm_len_99      = np.zeros(shape=(nx,), dtype=np.float64)
        Kolm_len_99g     = np.zeros(shape=(nx,), dtype=np.float64)
        
        Kolm_len_logLaw_y_lo = np.zeros(shape=(nx,), dtype=np.float64)
        Kolm_len_logLaw_y_md = np.zeros(shape=(nx,), dtype=np.float64)
        Kolm_len_logLaw_y_hi = np.zeros(shape=(nx,), dtype=np.float64)
        
        Kolm_res_dx_99   = np.zeros(shape=(nx,), dtype=np.float64)
        Kolm_res_dx_99g  = np.zeros(shape=(nx,), dtype=np.float64)
        
        Kolm_res_dy_99   = np.zeros(shape=(nx,), dtype=np.float64)
        Kolm_res_dy_99g  = np.zeros(shape=(nx,), dtype=np.float64)
        
        if dz is not None:
            Kolm_res_dz_99   = np.zeros(shape=(nx,), dtype=np.float64)
            Kolm_res_dz_99g  = np.zeros(shape=(nx,), dtype=np.float64)
        
        # === the maximum Kolmogorov (relative) resolution
        
        Kolm_res_dy_max  = np.zeros(shape=(nx,), dtype=np.float64)
        y_Kolm_res_dy_max = np.zeros(shape=(nx,), dtype=np.float64)
        
        ### ## interpolated --> not really helpful
        ### Kolm_res_dy_maxg = np.zeros(shape=(nx,), dtype=np.float64)
        ### Kolm_res_dy_maxj = np.zeros(shape=(nx,), dtype=np.float64)
        ### y_Kolm_res_dy_maxg = np.zeros(shape=(nx,), dtype=np.float64)
        
        Kolm_res_dx_max  = np.zeros(shape=(nx,), dtype=np.float64)
        y_Kolm_res_dx_max = np.zeros(shape=(nx,), dtype=np.float64)
        
        if dz is not None:
            Kolm_res_dz_max  = np.zeros(shape=(nx,), dtype=np.float64)
            y_Kolm_res_dz_max = np.zeros(shape=(nx,), dtype=np.float64)
        
        ### dy at d99 (grid snapped 'g' and interpolated)
        dy99g = np.zeros(shape=(nx,), dtype=np.float64)
        dy99  = np.zeros(shape=(nx,), dtype=np.float64)
        
        for i in range(nx):
            
            je = d99j[i]
            je_plus_5 = je+5 ## add some pts for high-order spline
            
            # === Kolm len @ d99
            
            Kolm_len_99g[i] = Kolm_len[i,je] ## Kolm length @ d99g (grid-snapped)
            Kolm_len_99[i]  = sp.interpolate.interp1d(y[:je_plus_5], Kolm_len[i,:je_plus_5])(d99[i]) ## Kolm length @ d99
            
            Kolm_res_dy_99g[i] = Kolm_res_dy[i,je] ## ratio dy/Kolm_len @ d99g (grid-snapped)
            Kolm_res_dy_99[i]  = sp.interpolate.interp1d(y[1:je_plus_5], Kolm_res_dy[i,1:je_plus_5])(d99[i]) ## skip 1st cell (dy=0 there : /0 error)
            
            # === Kolm len @ points in log layer (begin,middle,end)
            
            Kolm_len_logLaw_y_lo[i] = sp.interpolate.interp1d(y[:je_plus_5], Kolm_len[i,:je_plus_5])(logLaw_y_lo[i])
            Kolm_len_logLaw_y_md[i] = sp.interpolate.interp1d(y[:je_plus_5], Kolm_len[i,:je_plus_5])(logLaw_y_md[i])
            Kolm_len_logLaw_y_hi[i] = sp.interpolate.interp1d(y[:je_plus_5], Kolm_len[i,:je_plus_5])(logLaw_y_hi[i])
            
            # === in x direction
            
            if (i==0): ## because dx[0]=0
                ii=1
            else:
                ii=i
            Kolm_res_dx_99g[i] = Kolm_len[i,je] / dx[ii]
            Kolm_res_dx_99[i]  = sp.interpolate.interp1d(y[1:je_plus_5], Kolm_res_dx[ii,1:je_plus_5])(d99[i])
            
            if dz is not None:
                Kolm_res_dz_99g[i] = Kolm_len[i,je] / dz
                Kolm_res_dz_99[i]  = sp.interpolate.interp1d(y[1:je_plus_5], Kolm_res_dz[i,1:je_plus_5])(d99[i])
            
            # === maximum
            
            ### ### interpolated --> not really helpful
            ### spl = sp.interpolate.CubicSpline(y[1:-5], Kolm_res_dy[i,1:-5], bc_type='natural')
            ### def splf(yt):
            ###     return -1.*spl(yt)
            ### result = sp.optimize.brute(splf, ranges=[(y[1], y[-4])], full_output=True, finish=sp.optimize.fmin)
            ### y_Kolm_res_dy_max_   = result[0][0]
            ### y_Kolm_res_dy_max[i] = y_Kolm_res_dy_max_
            ### Kolm_res_dy_max[i]   = spl(y_Kolm_res_dy_max_)
            ### Kolm_res_dy_maxj_    = np.abs(y-y_Kolm_res_dy_max_).argmin()
            ### Kolm_res_dy_maxj[i]  = Kolm_res_dy_maxj_
            ### Kolm_res_dy_maxg[i]  = Kolm_res_dy[i,Kolm_res_dy_maxj_]
            ### y_Kolm_res_dy_maxg[i]  = y[Kolm_res_dy_maxj_]
            
            Kolm_res_dy_maxj = Kolm_res_dy[i,:].argmax()
            Kolm_res_dy_max[i] = Kolm_res_dy[i,Kolm_res_dy_maxj]
            y_Kolm_res_dy_max[i] = y[Kolm_res_dy_maxj]
            
            Kolm_res_dx_maxj = Kolm_res_dx[ii,:].argmax()
            Kolm_res_dx_max[i] = Kolm_res_dx[ii,Kolm_res_dx_maxj]
            y_Kolm_res_dx_max[i] = y[Kolm_res_dx_maxj]
            
            if dz is not None:
                Kolm_res_dz_maxj = Kolm_res_dz[i,:].argmax()
                Kolm_res_dz_max[i] = Kolm_res_dz[i,Kolm_res_dz_maxj]
                y_Kolm_res_dz_max[i] = y[Kolm_res_dz_maxj]
            
            # ===
            
            dy99g[i] = dy[je]
            dy99[i]  = sp.interpolate.interp1d(y[1:je_plus_5], dy[1:je_plus_5])(d99[i])
        
        ## @ d99
        data['dy99g'] = dy99g
        data['dy99'] = dy99
        data['Kolm_len_99']  = Kolm_len_99
        data['Kolm_len_99g'] = Kolm_len_99g
        
        data['Kolm_len_logLaw_y_lo'] = Kolm_len_logLaw_y_lo
        data['Kolm_len_logLaw_y_md'] = Kolm_len_logLaw_y_md
        data['Kolm_len_logLaw_y_hi'] = Kolm_len_logLaw_y_hi
        
        data['Kolm_res_dy_99']  = Kolm_res_dy_99
        data['Kolm_res_dy_99g'] = Kolm_res_dy_99g
        data['Kolm_res_dx_99']  = Kolm_res_dx_99
        data['Kolm_res_dx_99g'] = Kolm_res_dx_99g
        if dz is not None:
            data['Kolm_res_dz_99']  = Kolm_res_dz_99
            data['Kolm_res_dz_99g'] = Kolm_res_dz_99g
        
        ### ### fails : product of linear interped values is != the linear interped value of the product
        ### Kolm_res_dy_99_  = Kolm_len_99  / dy99
        ### np.testing.assert_allclose(Kolm_res_dy_99, Kolm_res_dy_99_, rtol=1e-8)
        
        Kolm_res_dy_99g_ = dy99g / Kolm_len_99g
        np.testing.assert_allclose(Kolm_res_dy_99g, Kolm_res_dy_99g_, rtol=1e-8)
        
        # ===== max Kolmogorov (relative) res
        
        ### ### interpolated --> not really helpful
        ### data['Kolm_res_dy_maxg']   =   Kolm_res_dy_maxg
        ### data['Kolm_res_dy_maxj']   =   Kolm_res_dy_maxj
        ### data['y_Kolm_res_dy_maxg'] =  y_Kolm_res_dy_maxg
        
        data['Kolm_res_dy_max']   = Kolm_res_dy_max
        data['y_Kolm_res_dy_max'] = y_Kolm_res_dy_max
        
        data['Kolm_res_dx_max']   = Kolm_res_dx_max
        data['y_Kolm_res_dx_max'] = y_Kolm_res_dx_max
        
        if dz is not None:
            data['Kolm_res_dz_max']   = Kolm_res_dz_max
            data['y_Kolm_res_dz_max'] = y_Kolm_res_dz_max
    
    # === u_rms tangent --> this was never finished and is purely experimental
    doGetUrmsDoubleTan = False
    if doGetUrmsDoubleTan:
        
        def get_two_point_tangent(pts,spl):
            '''
            func for solution with sp.optimize.root()
            '''
            x1, x2 = pts[0], pts[1]
            y1 = spl(np.log10(x1))
            y2 = spl(np.log10(x2))
            m1 = spl(np.log10(x1),1)
            m2 = spl(np.log10(x2),1)
            eq1 = m1 - m2
            eq2 = m1 - ((y2-y1)/(np.log10(x2/x1)))
            return np.array([eq1,eq2])
        
        urms_tan_yp_1 = np.zeros(shape=(nx,), dtype=np.float64)
        urms_tan_yp_2 = np.zeros(shape=(nx,), dtype=np.float64)
        urms_tan_up_1 = np.zeros(shape=(nx,), dtype=np.float64)
        urms_tan_up_2 = np.zeros(shape=(nx,), dtype=np.float64)
        urms_tan_m    = np.zeros(shape=(nx,), dtype=np.float64)
        
        for i in range(nx):
            
            print(i)
            
            try:
                bb = np.copy(uI_uI_rms[i,:] / u_tau[i])
                aa = np.copy(y_plus[i,:])
                jmin = bb.argmax() - 1 ## remove lower y+
                aa = np.copy(aa[jmin:])
                bb = np.copy(bb[jmin:])
                #jmax = np.abs(bb-0.3).argmin() ## remove lower (urms/u_tau)
                jmax = d99j[i] ## not doing what you think it's doing --> theres an offset
                aa = np.copy(aa[:jmax])
                bb = np.copy(bb[:jmax])
                
                spl = sp.interpolate.CubicSpline(np.log10(aa),bb,bc_type='natural',extrapolate=False)
                
                ## # ===== get best initial guess
                ## tmp = []
                ## for ii in range(aa.size):
                ##     if (ii>30):
                ##         x1T = aa[2]
                ##         x2T = aa[ii]
                ##         y1T = spl(np.log10(x1T))
                ##         y2T = spl(np.log10(x2T))
                ##         mT = ((y2T-y1T)/(np.log10(x2T/x1T)))
                ##         tmp.append(mT)
                ##     else:
                ##         tmp.append(-10000)
                ## tmp = np.array(tmp)
                ## jguess = tmp.argmax()
                ## print(jguess)
                ## # =====
                
                jguess=int(round(aa.size*0.66))
            
                state_init = np.array([aa[3], aa[jguess]])
                
                sol = sp.optimize.root(get_two_point_tangent, state_init,
                                       method='krylov', 
                                       args=(spl),
                                       tol=1e-8,
                                       options={'maxiter':20000 ,'disp':False,'jac_options':{'method':'minres'}}) #'xtol':1e-10,
                
                state = sol.x
                x1, x2 = state[0], state[1]
                y1 = spl(np.log10(x1))
                y2 = spl(np.log10(x2))
                m1 = spl(np.log10(x1),1)
                m2 = spl(np.log10(x2),1)
                
                if np.isclose(m1, m2, rtol=1e-5):
                    tanLineWasFound = True
                else:
                    print('double tangent line not found')
                    tanLineWasFound = False
                    print(m1)
                    print(m2)
                
                if np.isclose(np.log10(x1), np.log10(x2), rtol=1e-3):
                    print('optimization failed : np.log10(x1)==np.log10(x2) : %0.2f==%0.2f'%(np.log10(x1),np.log10(x2)))
                    tanLineWasFound = False
                    print(np.log10(x1))
                    print(np.log10(x2))
                    print(x1)
                    print(x2)
                
                if tanLineWasFound:
                    urms_tan_yp_1[i] = x1
                    urms_tan_yp_2[i] = x2
                    urms_tan_up_1[i] = y1
                    urms_tan_up_2[i] = y2
                    urms_tan_m[i]    = m1
                else:
                    urms_tan_yp_1[i] = None
                    urms_tan_yp_2[i] = None
                    urms_tan_up_1[i] = None
                    urms_tan_up_2[i] = None
                    urms_tan_m[i]    = None
            
            except RuntimeError:
                
                print('ERROR')
                
                urms_tan_yp_1[i] = None
                urms_tan_yp_2[i] = None
                urms_tan_up_1[i] = None
                urms_tan_up_2[i] = None
                urms_tan_m[i]    = None
        
        data['urms_tan_yp_1'] = urms_tan_yp_1
        data['urms_tan_yp_2'] = urms_tan_yp_2
        data['urms_tan_up_1'] = urms_tan_up_1
        data['urms_tan_up_2'] = urms_tan_up_2
        data['urms_tan_m']    = urms_tan_m
    
    # === get peak u'u'
    doGetPeakuIuI = True
    if doGetPeakuIuI and ('uI_uI' in locals()):
        uIuIp_peak     = np.zeros(shape=(nx,), dtype=np.float64)
        uIuIp_peak_y   = np.zeros(shape=(nx,), dtype=np.float64)
        uIIuIIp_peak   = np.zeros(shape=(nx,), dtype=np.float64)
        uIIuIIp_peak_y = np.zeros(shape=(nx,), dtype=np.float64)
        
        for xi in range(nx):
            
            # === Reynolds avg: u'u'
            bb = uI_uI[xi,:]/u_tau[xi]**2
            aa = y_plus[xi,:]
            cc = np.where((aa>5) & (aa<35))
            aa = aa[cc]
            bb = bb[cc]
            spl = sp.interpolate.CubicSpline(np.log10(aa),bb,bc_type='natural',extrapolate=False)
            roots = spl.derivative(nu=1).roots(discontinuity=False,extrapolate=False)
            if not (roots.size>0):
                #raise AssertionError('no roots')
                uIuIp_peak_y[xi] = np.nan
                uIuIp_peak[xi]   = np.nan
            elif not (roots.size==1):
                #raise AssertionError('multiple roots')
                uIuIp_peak_y[xi] = np.nan
                uIuIp_peak[xi]   = np.nan
            else:
                uIuIp_peak_y_    = roots[0] 
                uIuIp_peak_y[xi] = 10**uIuIp_peak_y_ * sc_l_in[xi]
                
                uIuIp_peak_      = spl(uIuIp_peak_y_)
                uIuIp_peak[xi]   = uIuIp_peak_
            
            # === Favre avg: u''u''
            if ('r_uII_uII' in locals()):
                bb = r_uII_uII[xi,:]/(u_tau[xi]**2 * rho[xi,:])
                aa = y_plus[xi,:]
                cc = np.where((aa>5) & (aa<35))
                aa = aa[cc]
                bb = bb[cc]
                spl = sp.interpolate.CubicSpline(np.log10(aa),bb,bc_type='natural',extrapolate=False)
                roots = spl.derivative(nu=1).roots(discontinuity=False,extrapolate=False)
                if not (roots.size>0):
                    #raise AssertionError('no roots')
                    uIIuIIp_peak_y[xi] = np.nan
                    uIIuIIp_peak[xi]   = np.nan
                elif not (roots.size==1):
                    #raise AssertionError('multiple roots')
                    uIIuIIp_peak_y[xi] = np.nan
                    uIIuIIp_peak[xi]   = np.nan
                else:
                    uIIuIIp_peak_y_    = roots[0] 
                    uIIuIIp_peak_y[xi] = 10**uIIuIIp_peak_y_ * sc_l_in[xi]
                    
                    uIIuIIp_peak_      = spl(uIIuIIp_peak_y_)
                    uIIuIIp_peak[xi]   = uIIuIIp_peak_
        
        data['uIuIp_peak']     = uIuIp_peak
        data['uIuIp_peak_y']   = uIuIp_peak_y
        if ('r_uII_uII' in locals()):
            data['uIIuIIp_peak']   = uIIuIIp_peak
            data['uIIuIIp_peak_y'] = uIIuIIp_peak_y
    
    return data

# post-processing : vector & tensor ops
# ------------------------------------------------------------

def get_grad(a,b,c, x,y,z, **kwargs):
    '''
    get the 3D gradient tensor (∂Ai/∂xj) from vector A=[a,b,c]
    -----
    - a,b,c are 3D arrays
    - x,y,z are 1D arrays (coord vectors)
    '''
    
    do_stack = kwargs.get('do_stack',True)
    hiOrder  = kwargs.get('hiOrder',True)
    verbose  = kwargs.get('verbose',False)
    
    nx = x.size; ny = y.size; nz = z.size
    
    dtype = a.dtype
    
    # === gradients with O3 Spline + natural BCs
    if hiOrder:
        '''
        this could be parallelized with multiprocessing + async threads
        '''
        dadx = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dady = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dadz = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dbdx = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dbdy = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dbdz = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dcdx = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dcdy = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dcdz = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        
        if verbose: progress_bar = tqdm(total=((nx*ny)+(ny*nz)+(nz*nx)), ncols=100, desc='get_grad()', leave=False, file=sys.stdout)
        
        for i in range(nx):
            for j in range(ny):
                dadz[i,j,:] = sp.interpolate.CubicSpline(z,a[i,j,:],bc_type='natural')(z,1)
                dbdz[i,j,:] = sp.interpolate.CubicSpline(z,b[i,j,:],bc_type='natural')(z,1)
                dcdz[i,j,:] = sp.interpolate.CubicSpline(z,c[i,j,:],bc_type='natural')(z,1)
                if verbose: progress_bar.update()
        for j in range(ny):
            for k in range(nz):
                dadx[:,j,k] = sp.interpolate.CubicSpline(x,a[:,j,k],bc_type='natural')(x,1)
                dbdx[:,j,k] = sp.interpolate.CubicSpline(x,b[:,j,k],bc_type='natural')(x,1)
                dcdx[:,j,k] = sp.interpolate.CubicSpline(x,c[:,j,k],bc_type='natural')(x,1)
                if verbose: progress_bar.update()
        for k in range(nz):
            for i in range(nx):
                dady[i,:,k] = sp.interpolate.CubicSpline(y,a[i,:,k],bc_type='natural')(y,1)
                dbdy[i,:,k] = sp.interpolate.CubicSpline(y,b[i,:,k],bc_type='natural')(y,1)
                dcdy[i,:,k] = sp.interpolate.CubicSpline(y,c[i,:,k],bc_type='natural')(y,1)
                if verbose: progress_bar.update()
        
        if verbose: progress_bar.close()
    
    else: ## numpy.gradient() --> 1st (or 2nd?) order
        if verbose: progress_bar = tqdm(total=9, ncols=100, desc='get_grad()', leave=False, file=sys.stdout)
        
        dadx = np.gradient(a, x, edge_order=1, axis=0)
        if verbose: progress_bar.update()
        dady = np.gradient(a, y, edge_order=1, axis=1)
        if verbose: progress_bar.update()
        dadz = np.gradient(a, z, edge_order=1, axis=2)
        if verbose: progress_bar.update()
        dbdx = np.gradient(b, x, edge_order=1, axis=0)
        if verbose: progress_bar.update()
        dbdy = np.gradient(b, y, edge_order=1, axis=1)
        if verbose: progress_bar.update()
        dbdz = np.gradient(b, z, edge_order=1, axis=2)
        if verbose: progress_bar.update()
        dcdx = np.gradient(c, x, edge_order=1, axis=0)
        if verbose: progress_bar.update()
        dcdy = np.gradient(c, y, edge_order=1, axis=1)
        if verbose: progress_bar.update()
        dcdz = np.gradient(c, z, edge_order=1, axis=2)
        if verbose: progress_bar.update()
        
        if verbose: progress_bar.close()
    
    if do_stack:
        dAdx_ij = np.stack((np.stack((dadx, dady, dadz), axis=3),
                            np.stack((dbdx, dbdy, dbdz), axis=3),
                            np.stack((dcdx, dcdy, dcdz), axis=3)), axis=4)
        return dAdx_ij
    else:
        dAdx_ij = dict(zip(['dadx', 'dady', 'dadz', 'dbdx', 'dbdy', 'dbdz', 'dcdx', 'dcdy', 'dcdz'], 
                           [ dadx,   dady,   dadz,   dbdx,   dbdy,   dbdz,   dcdx,   dcdy,   dcdz]))
        return dAdx_ij

def get_curl(a,b,c, x,y,z, **kwargs):
    '''
    get 3D curl vector ∇⨯A , A=[a,b,c]
    '''
    
    do_stack = kwargs.get('do_stack',True)
    hiOrder  = kwargs.get('hiOrder',True)
    verbose  = kwargs.get('verbose',False)
    
    nx = x.size; ny = y.size; nz = z.size
    
    dtype = u.dtype
    
    ## gradients with O3 Spline + natural BCs
    if hiOrder:
        '''
        this could be easily parallelized
        '''
        
        #dadx = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dady = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dadz = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dbdx = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        #dbdy = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dbdz = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dcdx = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        dcdy = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        #dcdz = np.zeros(shape=(nx,ny,nz), dtype=dtype)
        
        if verbose: progress_bar = tqdm(total=((nx*ny)+(ny*nz)+(nz*nx)), ncols=100, desc='get_curl()', leave=False, file=sys.stdout)
        
        for i in range(nx):
            for j in range(ny):
                dadz[i,j,:] = sp.interpolate.CubicSpline(z,a[i,j,:],bc_type='natural')(z,1)
                dbdz[i,j,:] = sp.interpolate.CubicSpline(z,b[i,j,:],bc_type='natural')(z,1)
                #dcdz[i,j,:] = sp.interpolate.CubicSpline(z,c[i,j,:],bc_type='natural')(z,1)
                if verbose: progress_bar.update()
        for j in range(ny):
            for k in range(nz):
                #dadx[:,j,k] = sp.interpolate.CubicSpline(x,a[:,j,k],bc_type='natural')(x,1)
                dbdx[:,j,k] = sp.interpolate.CubicSpline(x,b[:,j,k],bc_type='natural')(x,1)
                dcdx[:,j,k] = sp.interpolate.CubicSpline(x,c[:,j,k],bc_type='natural')(x,1)
                if verbose: progress_bar.update()
        for k in range(nz):
            for i in range(nx):
                dady[i,:,k] = sp.interpolate.CubicSpline(y,a[i,:,k],bc_type='natural')(y,1)
                #dbdy[i,:,k] = sp.interpolate.CubicSpline(y,b[i,:,k],bc_type='natural')(y,1)
                dcdy[i,:,k] = sp.interpolate.CubicSpline(y,c[i,:,k],bc_type='natural')(y,1)
                if verbose: progress_bar.update()
        
        if verbose: progress_bar.close()
    
    else: ## numpy.gradient() --> 1st (or 2nd?) order
        if verbose: progress_bar = tqdm(total=9, ncols=100, desc='get_curl()', leave=False, file=sys.stdout)
        
        #dadx = np.gradient(a, x, edge_order=1, axis=0)
        #if verbose: progress_bar.update()
        dady = np.gradient(a, y, edge_order=1, axis=1)
        if verbose: progress_bar.update()
        dadz = np.gradient(a, z, edge_order=1, axis=2)
        if verbose: progress_bar.update()
        dbdx = np.gradient(b, x, edge_order=1, axis=0)
        if verbose: progress_bar.update()
        #dbdy = np.gradient(b, y, edge_order=1, axis=1)
        #if verbose: progress_bar.update()
        dbdz = np.gradient(b, z, edge_order=1, axis=2)
        if verbose: progress_bar.update()
        dcdx = np.gradient(c, x, edge_order=1, axis=0)
        if verbose: progress_bar.update()
        dcdy = np.gradient(c, y, edge_order=1, axis=1)
        if verbose: progress_bar.update()
        #dcdz = np.gradient(c, z, edge_order=1, axis=2)
        #if verbose: progress_bar.update()
        
        if verbose: progress_bar.close()
    
    # ===
    curl_x = dcdy - dbdz
    curl_y = dadz - dcdx
    curl_z = dbdx - dady
    
    if do_stack:
        curl = np.stack((curl_x, curl_y, curl_z), axis=3)
        return curl
    else:
        curl = dict(zip(['curl_x', 'curl_y', 'curl_z'], 
                        [ curl_x,   curl_y,   curl_z]))
        return curl

# post-processing : spectral & wavelet
# ------------------------------------------------------------

def get_overlapping_window_size(asz, n_win, overlap_fac):
    '''
    get window length and overlap given a
    desired number of windows and a nominal overlap factor
    -----
    --> the output should be passed to get_overlapping_windows()
        to do the actual padding & windowing
    '''
    if not isinstance(asz, int):
        raise TypeError('arg asz must be type int')
    if not isinstance(n_win, int):
        raise TypeError('arg n_win must be type int')
    if (overlap_fac >= 1.):
        raise ValueError('arg overlap_fac must be <1')
    if (overlap_fac < 0.):
        raise ValueError('arg overlap_fac must be >0')
    n_ends = n_win+1
    n_mids = n_win
    
    # === solve for float-valued window 'mid' size & 'end' size
    def eqn(soltup, asz=asz, overlap_fac=overlap_fac):
        (endsz,midsz) = soltup
        eq1 = asz - n_ends*endsz - n_mids*midsz
        eq2 = overlap_fac*(midsz+2*endsz) - endsz
        return [eq1, eq2]
    
    guess = asz*0.5
    endsz,midsz = sp.optimize.fsolve(eqn, (guess,guess), (asz,overlap_fac))
    win_len     = midsz + 2*endsz
    overlap     = endsz
    
    win_len = max(math.ceil(win_len),1)
    overlap = max(math.floor(overlap),0)
    
    return win_len, overlap

def get_overlapping_windows(a, win_len, overlap):
    '''
    subdivide 1D array into overlapping windows
    '''
    #pad_mode = kwargs.get('pad_mode','append')
    ##
    if not isinstance(a, np.ndarray):
        raise TypeError('arg a must be type np.ndarray')
    if not isinstance(win_len, int):
        raise TypeError('arg win_len must be type int')
    if not isinstance(overlap, int):
        raise TypeError('arg overlap must be type int')
    ##
    asz   = a.size
    skip  = win_len - overlap
    n_pad = (win_len - asz%skip)%skip
    #a_pad = np.concatenate(( np.zeros(n_pad,dtype=a.dtype) , np.copy(a) )) ## prepend
    a_pad = np.concatenate(( np.copy(a) , np.zeros(n_pad,dtype=a.dtype) )) ## append
    ##
    b = np.lib.stride_tricks.sliding_window_view(a_pad, win_len, axis=0)
    b = np.copy(b[::skip,:])
    n_win = b.shape[0]
    ##
    if (n_pad > 0.5*win_len):
        print('WARNING: n_pad > overlap')
    ##
    return b, n_win, n_pad

# binary I/O
# ------------------------------------------------------------

def gulp(fname, **kwargs):
    '''
    read a complete binary file into memory, return 'virtual file'
    -----
    - returned handle can be opened via h5py as if it were on disk (with very high performance)
    - of course this only works if the entire file fits into memory
    - best use case: medium size file, large number of high-frequency read ops
    '''
    verbose = kwargs.get('verbose',True)
    f_size_gb = os.path.getsize(fname)/1024**3
    if verbose: tqdm.write('>>> gulp() : %s : %0.2f [GB]'%(os.path.basename(fname),f_size_gb))
    t_start = timeit.default_timer()
    with open(fname, 'rb') as fnb:
        bytes_in_mem = io.BytesIO(fnb.read())
    t_delta = timeit.default_timer() - t_start
    if verbose: tqdm.write('>>> gulp() : %s : %0.2f [GB/s]'%(format_time_string(t_delta), (f_size_gb/t_delta)))
    return bytes_in_mem

# utilities
# ------------------------------------------------------------

def format_time_string(tsec):
    '''
    format seconds as dd:hh:mm:ss
    '''
    m, s = divmod(tsec,60)
    h, m = divmod(m,60)
    d, h = divmod(h,24)
    time_str = '%dd:%dh:%02dm:%02ds'%(d,h,m,s)
    return time_str

def even_print(label, output, **kwargs):
    '''
    print an evenly spaced list to the terminal
    '''
    terminal_width = kwargs.get('terminal_width',72)
    s              = kwargs.get('s',False) ## return string
    
    ndots = (terminal_width-2) - len(label) - len(output)
    text = label+' '+ndots*'.'+' '+output
    if s:
        return text
    else:
        #sys.stdout.write(text)
        print(text)
        return

# plotting & matplotlib
# ------------------------------------------------------------

def set_mpl_env(**kwargs):
    '''
    Setup the matplotlib environment
    
    - styles   : https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    - rcParams : https://matplotlib.org/stable/tutorials/introductory/customizing.html
    
    TrueType / OpenType Fonts
    -------------------------
    
    - IBM Plex
        - https://github.com/IBM/plex/archive/refs/heads/master.zip
        - https://www.fontsquirrel.com/fonts/download/ibm-plex --> doesnt contain 'Condensed'
        - test at : https://www.ibm.com/plex/plexness/
        
        ----- Linux / Ubuntu / WSL2
        $> wget https://github.com/IBM/plex/archive/refs/heads/master.zip
        $> unzip master.zip
        $> cd plex-master
        $> sudo mkdir -p /usr/share/fonts/opentype/ibm-plex
        $> sudo mkdir -p /usr/share/fonts/truetype/ibm-plex
        $> sudo find . -name '*.otf' -exec cp -v {} /usr/share/fonts/opentype/ibm-plex/ \;
        $> sudo find . -name '*.ttf' -exec cp -v {} /usr/share/fonts/truetype/ibm-plex/ \;
        $> fc-cache -f -v
        $> fc-list | grep 'IBM'
        ----- regenerate matplotlib font cache (just delete, gets regenerated)
        $> rm ~/.cache/matplotlib/fontlist-v330.json
        $> rm -rf ~/.cache/matplotlib/tex.cache
    
    - Latin Modern Math
        - http://www.gust.org.pl/projects/e-foundry/lm-math/download/latinmodern-math-1959.zip
    
    - Latin Modern (lmodern in LaTeX)
        - https://www.fontsquirrel.com/fonts/download/Latin-Modern-Roman.zip
        - http://www.gust.org.pl/projects/e-foundry/latin-modern/download/lm2.004otf.zip
        
        --> usually already installed at : /usr/share/texmf/fonts/opentype/public/lm/
        ----- Linux / Ubuntu / WSL2
        $> wget https://www.fontsquirrel.com/fonts/download/Latin-Modern-Roman.zip
        $> unzip Latin-Modern-Roman.zip -d Latin-Modern-Roman
        $> cd Latin-Modern-Roman
        $> sudo mkdir -p /usr/share/fonts/opentype/lmodern
        $> sudo find . -name '*.otf' -exec cp -v {} /usr/share/fonts/opentype/lmodern/ \;
        $> fc-cache -f -v
        $> fc-list | grep 'Latin'
        ----- regenerate matplotlib font cache (just delete, gets regenerated)
        $> rm ~/.cache/matplotlib/fontlist-v330.json
        $> rm -rf ~/.cache/matplotlib/tex.cache
    
    - Computer Modern (default in LaTeX)
        - https://www.fontsquirrel.com/fonts/download/computer-modern.zip
        - http://mirrors.ctan.org/fonts/cm/ps-type1/bakoma.zip
        
        ----- Linux / Ubuntu / WSL2
        $> wget https://www.fontsquirrel.com/fonts/download/computer-modern.zip
        $> unzip computer-modern.zip -d computer-modern
        $> cd computer-modern
        $> sudo mkdir -p /usr/share/fonts/truetype/cmu
        $> sudo find . -name '*.ttf' -exec cp -v {} /usr/share/fonts/truetype/cmu/ \;
        $> fc-cache -f -v
        $> fc-list | grep 'CMU'
        ----- regenerate matplotlib font cache (just delete, gets regenerated)
        $> rm ~/.cache/matplotlib/fontlist-v330.json
        $> rm -rf ~/.cache/matplotlib/tex.cache
    
    Windows
    -------
    --> download, install, then delete : C:/Users/%USERNAME%/.matplotlib/fontlist-v330.json
    --> this JSON file will get regenerated with newly installed fonts
    
    MikTeX .sty files --> global (sometimes needed for journals)
    -----------------
    - https://miktex.org/faq/local-additions
    - in C:/Users/%USERNAME%/AppData/Local  : make : mytextmf\tex\latex\mystuff
    - register as root directory in MikTeX
    - put .sty files in there
    
    Dimension Presets: Springer 'svjour3' Template
    ----------------------------------------------
    ltx_textwidth  = 6.85066
    ltx_hsize      = 3.30719 # \linewidth
    ltx_textheight = 9.2144 * 0.90 ### error if fig is actually this tall, so effective max *= 0.90
    ltx_vsize      = 9.2144 * 0.90
    
    '''
    
    useTex   = kwargs.get('useTex',False) ## use LaTeX text rendering
    darkMode = kwargs.get('darkMode',True)
    
    if darkMode:
        mpl.style.use('dark_background') ## dark mode
    else:
        mpl.style.use('default')
    
    if useTex:
        
        #mpl.rcParams.update(mpl.rcParamsDefault) ## reset to defaults
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['pgf.texsystem'] = 'pdflatex'
        mpl.rcParams['text.latex.preamble'] = '\n'.join([\
                                                         #r'\usepackage[utf8]{inputenc}',
                                                         r'\usepackage[T1]{fontenc}',
                                                         r'\usepackage{amsmath}', 
                                                         r'\usepackage{amsfonts}',
                                                         r'\usepackage{xfrac}',
                                                         #r'\usepackage{nicefrac}',
                                                         ###
                                                         #r'\usepackage{lmodern}', ## Latin Modern
                                                         #r'\usepackage{gensymb}', ## Generic symbols 
                                                         #r'\usepackage{txfonts}', ## Times-like fonts mathtext symbols
                                                         ###
                                                         r'\usepackage{plex-sans}', ## IBM Plex Sans
                                                         r'\renewcommand{\familydefault}{\sfdefault}', ## sans as default family
                                                         r'\renewcommand{\seriesdefault}{c}', ## condensed {*} as default series
                                                         r'\usepackage[italic]{mathastext}', ## use default font in math mode
                                                        ])
        #mpl.rc('font',**{'family':'serif','serif':['Times New Roman'],'size':10,'weight':'normal'}) ## Times New Roman in LaTeX --> activate txfonts
        #mpl.rc('font',**{'family':'serif','serif':['Times'],'size':10,'weight':'normal'}) ## Times in LaTeX
        pass
    
    else:
        
        # === Register OTF/TTF Fonts (only necessary once)
        
        ### # === register (new) fonts : Windows --> done automatically if you delete ~/.cache/matplotlib/fontlist-v330.json
        ### ##mpl.font_manager.findSystemFonts(fontpaths='C:/Windows/Fonts', fontext='ttf')
        ### #mpl.font_manager.findSystemFonts(fontpaths='C:/Users/'+os.path.expandvars('%USERNAME%')+'/AppData/Local/Microsoft/Windows/Fonts', fontext='ttf')
        ### mpl.font_manager.findSystemFonts(fontpaths=mpl.font_manager.win32FontDirectory(), fontext='ttf')
        
        ### # === register (new) fonts : Linux / WSL2 --> done automatically if you delete C:/Users/%USERNAME%/.matplotlib/fontlist-v330.json
        ### mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        ### mpl.font_manager.findSystemFonts(fontpaths=None, fontext='otf')
        
        ### # === example: list all TTF font properties
        ### fonts = mpl.font_manager.fontManager.ttflist
        ### #fonts = [f for f in fonts if all([('IBM' in f.name),('Condensed' in f.name)])] ## filter list
        ### for f in fonts:
        ###     print(f.name)
        ###     print(Path(f.fname).stem)
        ###     print('weight  : %s'%str(f.weight))
        ###     print('style   : %s'%str(f.style))
        ###     print('stretch : %s'%str(f.stretch))
        ###     print('variant : %s'%str(f.variant))
        
        ### # === example: list all font families registered in matplotlib
        ### fontlist = mpl.font_manager.get_fontconfig_fonts()
        ### fontnames = sorted(list(set([mpl.font_manager.FontProperties(fname=fname).get_name() for fname in fontlist])))
        ### for i in range(len(fontnames)):
        ###     print(fontnames[i])
        ### print('\n')
        
        # === TTF/OTF fonts (when NOT using LaTeX rendering)
        
        ## Times New Roman
        if False:
            mpl.rcParams['font.family'] = 'Times New Roman'
            mpl.rcParams['font.weight'] = 'normal'
            mpl.rcParams['mathtext.fontset'] = 'custom'
            mpl.rcParams['mathtext.default'] = 'it'
            mpl.rcParams['mathtext.rm'] = 'Times New Roman:normal'
            mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
            mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
        
        ## IBM Plex Sans
        if False:
            mpl.rcParams['font.family'] = 'IBM Plex Sans'
            mpl.rcParams['font.weight'] = '400' ## 'light'
            #mpl.rcParams['font.stretch'] = 'normal' ## always 'normal' for family
            mpl.rcParams['mathtext.fontset'] = 'custom'
            mpl.rcParams['mathtext.default'] = 'it'
            mpl.rcParams['mathtext.rm'] = 'IBM Plex Sans:regular'
            mpl.rcParams['mathtext.it'] = 'IBM Plex Sans:italic:regular'
            mpl.rcParams['mathtext.bf'] = 'IBM Plex Sans:bold'
        
        ## IBM Plex Sans Condensed
        if True:
            mpl.rcParams['font.family'] = 'IBM Plex Sans Condensed'
            mpl.rcParams['font.weight'] = 'regular' ## 200, 300/'light', 400, 450
            #mpl.rcParams['font.style'] = 'normal' ## 'normal', 'italic', 'oblique'
            #mpl.rcParams['font.variant'] = 'normal' ## 'normal', 'small-caps'
            #mpl.rcParams['font.stretch'] = 'condensed' ## always 'condensed' for family
            mpl.rcParams['mathtext.fontset'] = 'custom'
            mpl.rcParams['mathtext.default'] = 'it'
            mpl.rcParams['mathtext.rm'] = 'IBM Plex Sans Condensed:regular'
            mpl.rcParams['mathtext.it'] = 'IBM Plex Sans Condensed:italic:regular'
            mpl.rcParams['mathtext.bf'] = 'IBM Plex Sans Condensed:bold'
            mpl.rcParams['mathtext.cal'] = 'Latin Modern Roman:italic'
        
        ## Latin Modern Roman (lmodern in LaTeX, often used)
        if False:
            mpl.rcParams['font.family'] = 'Latin Modern Roman'
            mpl.rcParams['font.weight'] = '400'
            mpl.rcParams['font.style'] = 'normal'
            mpl.rcParams['font.variant'] = 'normal'
            #mpl.rcParams['font.stretch'] = 'condensed'
            mpl.rcParams['mathtext.fontset'] = 'custom'
            mpl.rcParams['mathtext.default'] = 'it'
            mpl.rcParams['mathtext.rm'] = 'Latin Modern Roman:normal'
            mpl.rcParams['mathtext.it'] = 'Latin Modern Roman:italic'
            mpl.rcParams['mathtext.bf'] = 'Latin Modern Roman:bold'
        
        ## Computer Modern (LaTeX default)
        if False:
            mpl.rcParams['font.family'] = 'CMU Serif'
            mpl.rcParams['font.weight'] = 'regular'
            mpl.rcParams['font.style'] = 'normal'
            mpl.rcParams['mathtext.fontset'] = 'custom'
            mpl.rcParams['mathtext.default'] = 'it'
            mpl.rcParams['mathtext.rm'] = 'CMU Serif:regular'
            mpl.rcParams['mathtext.it'] = 'CMU Serif:italic:regular'
            mpl.rcParams['mathtext.bf'] = 'CMU Serif:bold'
        
        ## Manually point to a TTF/OTF file
        if False:
            #fe = mpl.font_manager.FontEntry(fname='/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf', name='10erLatin')
            fe = mpl.font_manager.FontEntry(fname='C:/Users/'+os.path.expandvars('%USERNAME%')+'/AppData/Local/Microsoft/Windows/Fonts/lmroman10-regular.otf', name='10erLatin')
            mpl.font_manager.fontManager.ttflist.insert(0, fe)
            mpl.rcParams['font.family'] = fe.name
            mpl.rcParams['font.weight'] = '400'
            mpl.rcParams['font.style'] = 'normal'
            mpl.rcParams['font.variant'] = 'normal'
            mpl.rcParams['mathtext.fontset'] = 'custom'
            mpl.rcParams['mathtext.default'] = 'it'
            mpl.rcParams['mathtext.rm'] = fe.name+':normal'
            mpl.rcParams['mathtext.it'] = fe.name+':italic'
            mpl.rcParams['mathtext.bf'] = fe.name+':bold'
    
    # ===
    
    ### # === list all options
    ### print(mpl.rcParams.keys())
    
    fontsize = 10
    axesAndTickWidth = 0.5
    
    mpl.rcParams['figure.figsize'] = 4, 4/(32/15)
    mpl.rcParams['figure.dpi']     = 300
    #mpl.rcParams['figure.facecolor'] = 'k'
    #mpl.rcParams['figure.autolayout'] = True ### tight_layout() --> just use instead : fig1.tight_layout(pad=0.20)
    
    #mpl.rcParams['figure.constrained_layout.use'] = True
    #mpl.rcParams['figure.constrained_layout.h_pad']  = 0.0 ## Padding around axes objects. Float representing
    #mpl.rcParams['figure.constrained_layout.w_pad']  = 0.0 ## inches. Default is 3/72 inches (3 points)
    #mpl.rcParams['figure.constrained_layout.hspace'] = 0.2 ## Space between subplot groups. Float representing
    #mpl.rcParams['figure.constrained_layout.wspace'] = 0.2 ## a fraction of the subplot widths being separated.
    
    #mpl.rcParams['figure.subplot.bottom'] = 0.02
    #mpl.rcParams['figure.subplot.top'] = 0.98
    #mpl.rcParams['figure.subplot.left'] = 0.02
    #mpl.rcParams['figure.subplot.right'] = 0.98
    #mpl.rcParams['figure.subplot.hspace'] = 0.02
    #mpl.rcParams['figure.subplot.wspace'] = 0.2
    
    mpl.rcParams['pdf.compression'] = 2 ## 0-9
    #mpl.rcParams['pdf.fonttype'] = 42  # Output Type 3 (Type3) or Type 42 (TrueType)
    #mpl.rcParams['pdf.use14corefonts'] = False
    
    mpl.rcParams['savefig.pad_inches'] = 0.20
    mpl.rcParams['savefig.dpi']        = 300
    
    mpl.rcParams['xtick.major.size']  = 2.5
    mpl.rcParams['xtick.major.width'] = axesAndTickWidth
    mpl.rcParams['xtick.minor.size']  = 1.4
    mpl.rcParams['xtick.minor.width'] = axesAndTickWidth*1.0
    #mpl.rcParams['xtick.color'] = 'k' ## set with mpl.style.use()
    mpl.rcParams['xtick.direction']   = 'in'
    
    mpl.rcParams['ytick.major.size']  = 2.5
    mpl.rcParams['ytick.major.width'] = axesAndTickWidth
    mpl.rcParams['ytick.minor.size']  = 1.4
    mpl.rcParams['ytick.minor.width'] = axesAndTickWidth*1.0
    #mpl.rcParams['ytick.color'] = 'k' ## set with mpl.style.use()
    mpl.rcParams['ytick.direction'] = 'in'
    
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    
    mpl.rcParams['xtick.major.pad'] = 3.0
    mpl.rcParams['ytick.major.pad'] = 1.0
    
    mpl.rcParams['lines.linewidth']  = 0.5
    mpl.rcParams['lines.linestyle']  = 'solid'
    mpl.rcParams['lines.marker']     = 'None' #'o'
    mpl.rcParams['lines.markersize'] = 1.2
    mpl.rcParams['lines.markeredgewidth'] = 0.
    
    #mpl.rcParams['axes.facecolor'] = 'k' ## set with mpl.style.use()
    mpl.rcParams['axes.linewidth'] = axesAndTickWidth
    mpl.rcParams['axes.labelpad']  = 3.0
    mpl.rcParams['axes.titlesize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    #mpl.rcParams['axes.axisbelow'] = False ## dont allow axes, ticks to be under lines --> doesn't work for artist objects with zorder >2.5
    
    mpl.rcParams['legend.fontsize'] = fontsize*0.8
    mpl.rcParams['legend.shadow']   = False
    mpl.rcParams['legend.borderpad'] = 0.2
    mpl.rcParams['legend.framealpha'] = 1.0
    mpl.rcParams['legend.edgecolor']  = 'inherit'
    mpl.rcParams['legend.handlelength'] = 1.0
    mpl.rcParams['legend.handletextpad'] = 0.4
    mpl.rcParams['legend.borderaxespad'] = 0.25
    mpl.rcParams['legend.columnspacing'] = 0.5
    mpl.rcParams['legend.fancybox'] = False
    
    ## display scaling with Qt5Agg backend for interactive plotting
    ## --> make Qt5Agg backend available with PyQt5 : pip3 install PyQt5 / pythonw -m pip install PyQt5
    if plt.get_backend() == 'Qt5Agg':
        from matplotlib.backends.qt_compat import QtWidgets
        qApp = QtWidgets.QApplication(sys.argv)
        physical_dpi = qApp.desktop().physicalDpiX()
        mpl.rcParams['figure.dpi'] = physical_dpi
    
    return

def get_Lch_colors(hues,**kwargs):
    '''
    given a list of hues [0-360], chroma & luminance, return
        colors in hex (html) or rgb format
    -----
    Lch color picker : https://css.land/lch/
    '''
    # ===
    c = kwargs.get('c',110) ## chroma
    L = kwargs.get('L',65) ## luminance
    fmt = kwargs.get('fmt','rgb') ## output format : hex or rgb tuples
    test_plot = kwargs.get('test_plot',False) ## plot to test colors
    # ===
    colors_rgb=[]
    colors_Lab=[]
    #import colormath
    #from colormath import color_objects, color_conversions
    #cspace_out = colormath.color_objects.sRGBColor
    #cspace_out.rgb_gamma = 2.2
    for h in hues:
        hX = h*np.pi/180
        LX,a,b = skimage.color.lch2lab([L,c,hX])
        colors_Lab.append([LX,a,b])
        # =====
        ### cc = colormath.color_objects.LCHuvColor(L, c, h-18, observer='2', illuminant='d65')
        ### #cc = colormath.color_objects.LCHabColor(L, c, h-18, observer='2', illuminant='d65')
        ### cc2 = colormath.color_conversions.convert_color(cc, cspace_out)
        ### cc3 = cc2.get_value_tuple()
        ### cc4 = (cc2.clamped_rgb_r, cc2.clamped_rgb_g, cc2.clamped_rgb_b)
        ### print(cc4)
        ### colors_rgb.append(cc4)
        # =====
        r,g,b = skimage.color.lab2rgb([LX,a,b])
        colors_rgb.append([r,g,b])
    
    if test_plot:
        nc = len(colors_rgb)
        x = np.linspace(0,2*np.pi,1000)
        plt.close('all')
        fig1 = plt.figure(frameon=True, figsize=(4, 4*(9/16)), dpi=300)
        ax1 = plt.gca()
        for i in range(nc):
            ax1.plot(x, np.cos(x-i*np.pi/nc), c=colors_rgb[i])
        fig1.tight_layout(pad=0.15)
        plt.show()
    
    if (fmt=='rgb'):
        colors=colors_rgb
    elif (fmt=='hex'):
        colors=[mpl.colors.to_hex(c) for c in colors_rgb]
    elif (fmt=='Lab'):
        colors=colors_Lab
    else:
        raise NameError('fmt=%s not a valid option'%fmt)
    
    return colors

def hex2rgb(hexstr,**kwargs):
    '''
    return (r,g,b) [0-1] from html/hexadecimal
    '''
    base = kwargs.get('base',1)
    hexstr = hexstr.lstrip('#')
    c = tuple(int(hexstr[i:i+2], 16) for i in (0, 2, 4))
    if (base==1):
        c = tuple(i/255. for i in c)
    return c

def hsv_adjust_hex(hex_list,h_fac,s_fac,v_fac,**kwargs):
    '''
    adjust the (h,s,v) values of a list of html color codes
    --> if single #XXXXXX is passed, returns single
    --> margin : adjust proportional to available margin
    '''
    margin = kwargs.get('margin',False)
    # ===
    if isinstance(hex_list, str):
        single=True
        hex_list = [hex_list]
    else:
        single=False
    # ===
    colors_rgb = [ hex2rgb(c) for c in hex_list ]
    colors_hsv = mpl.colors.rgb_to_hsv(colors_rgb)
    for ci in range(len(colors_hsv)):
        c = colors_hsv[ci]
        h, s, v = c
        if margin:
            h_margin = 1. - h
            s_margin = 1. - s
            v_margin = 1. - v
        else:
            h_margin = 1.
            s_margin = 1.
            v_margin = 1.
        h = max(0.,min(1.,h + h_margin*h_fac))
        s = max(0.,min(1.,s + s_margin*s_fac))
        v = max(0.,min(1.,v + v_margin*v_fac))
        colors_hsv[ci] = (h,s,v)
    colors_rgb = mpl.colors.hsv_to_rgb(colors_hsv)
    hex_list_out = [mpl.colors.to_hex(c).upper() for c in colors_rgb]
    if single:
        hex_list_out=hex_list_out[0] ## just the one tuple
    return hex_list_out

def cmap_hsv_adjust(cmap):
    pass
    return cmap

def analytical_u_plus_y_plus():
    '''
    return viscous, transitional (Spalding), and log law curves
    '''
    y_plus_viscousLayer = np.logspace(np.log10(0.1), np.log10(12.1), num=200, base=10.)
    u_plus_viscousLayer = np.logspace(np.log10(0.1), np.log10(12.1), num=200, base=10.)
    
    y_plus_logLaw = np.logspace(np.log10(9), np.log10(3000), num=20, base=10.)
    u_plus_logLaw = 1 / 0.41 * np.log(y_plus_logLaw) + 5.2
    
    u_plus_spalding = np.logspace(np.log10(0.1), np.log10(13.1), num=200, base=10.)
    y_plus_spalding = u_plus_spalding + \
                      0.1108*(np.exp(0.4*u_plus_spalding) - 1 \
                      -  0.4*u_plus_spalding \
                      - (0.4*u_plus_spalding)**2/(2*1) \
                      - (0.4*u_plus_spalding)**3/(3*2*1) \
                      - (0.4*u_plus_spalding)**4/(4*3*2*1) )
    
    return y_plus_viscousLayer, u_plus_viscousLayer, y_plus_logLaw , u_plus_logLaw, y_plus_spalding, u_plus_spalding

def fig_trim(fig, list_of_axes, **kwargs):
    '''
    trims the figure in (y) direction
    - first axis is main axis
    - future : update to work in x-direction also
    - typical use case : single equal aspect figure needs to be scooted / trimmed
    '''
    offset_px = kwargs.get('offset_px',2)
    fig_px_x, fig_px_y = fig.get_size_inches()*fig.dpi ; print('fig size px : %0.3f %0.3f'%(fig_px_x, fig_px_y))
    transFigInv = fig.transFigure.inverted()
    mainAxis = list_of_axes[0]
    # ===
    x0,  y0,  dx,  dy  = mainAxis.get_position().bounds
    x0A, y0A, dxA, dyA = mainAxis.get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=False).bounds ### pixel values of the axis tightbox
    #print('x0A, y0A, dxA, dyA : %0.2f %0.2f %0.2f %0.2f'%(x0A, y0A, dxA, dyA))
    dy_pct = dyA / fig_px_y #; print('dy_pct : %0.6f' % dy_pct)
    x0A, y0A = transFigInv.transform_point([x0A, y0A])
    dxA, dyA = transFigInv.transform_point([dxA, dyA])
    #y_shift = 1.0 - (y0A+dyA)
    y_shift = y0A
    # ===
    w = fig.get_figwidth()
    h = fig.get_figheight()
    fig.set_size_inches(w,h*1.02*dy_pct, forward=True)
    fig_px_x, fig_px_y = fig.get_size_inches()*fig.dpi ; print('fig size px : %0.3f %0.3f'%(fig_px_x, fig_px_y))
    w_adj = fig.get_figwidth()
    h_adj = fig.get_figheight()
    # ===
    for axis in list_of_axes:
        x0, y0, dx, dy  = axis.get_position().bounds
        x0n = x0
        y0n = y0-y_shift+(offset_px/fig_px_y)
        dxn = dx
        dyn = dy
        axis.set_position([x0n,y0n*(h/h_adj),dxn,dyn*(h/h_adj)])
    return

def axs_grid_compress(fig,axs,**kwargs):
    '''
    compress ax grid
    '''
    dim = kwargs.get('dim',1)
    offset_px = kwargs.get('offset_px',5)
    transFigInv = fig.transFigure.inverted()
    
    ### this is the ON SCREEN pixels size... has nothing to do with output
    fig_px_x, fig_px_y = fig.get_size_inches()*fig.dpi #; print('fig size px : %0.3f %0.3f'%(fig_px_x, fig_px_y))
    
    cols, rows = axs.shape
    for j in range(rows-1):
        
        ### determine the min x0 in each row
        top_row_y0s = []
        low_row_y1s = []
        for i in range(cols):
            
            x0,  y0,  dx,  dy  = axs[i,j+1].get_position().bounds
            
            ### pixel values of the axis tightbox
            x0A, y0A, dxA, dyA = axs[i,j+0].get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=False).bounds
            x0B, y0B, dxB, dyB = axs[i,j+1].get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=False).bounds
            #print('x0A, y0A, dxA, dyA : %0.2f %0.2f %0.2f %0.2f'%(x0A, y0A, dxA, dyA))
            #print('x0B, y0B, dxB, dyB : %0.2f %0.2f %0.2f %0.2f'%(x0B, y0B, dxB, dyB))
            
            ### convert pixel vals to dimless
            x0A, y0A = transFigInv.transform_point([x0A, y0A])
            dxA, dyA = transFigInv.transform_point([dxA, dyA])
            x0B, y0B = transFigInv.transform_point([x0B, y0B])
            dxB, dyB = transFigInv.transform_point([dxB, dyB])
            #print('x0A, y0A, dxA, dyA : %0.2f %0.2f %0.2f %0.2f'%(x0A, y0A, dxA, dyA))
            #print('x0B, y0B, dxB, dyB : %0.2f %0.2f %0.2f %0.2f'%(x0B, y0B, dxB, dyB))
            #print('\n')
            
            top_row_y0s.append(y0A)
            low_row_y1s.append(y0B+dyB)
        
        y_shift = min(top_row_y0s) - max(low_row_y1s)
        
        # =====
        
        for i in range(cols):
            
            x0,  y0,  dx,  dy  = axs[i,j+1].get_position().bounds
            
            # x0A, y0A, dxA, dyA = axs[i,j+0].get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=False).bounds
            # x0B, y0B, dxB, dyB = axs[i,j+1].get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=False).bounds
            # #print('x0A, y0A, dxA, dyA : %0.2f %0.2f %0.2f %0.2f'%(x0A, y0A, dxA, dyA))
            # #print('x0B, y0B, dxB, dyB : %0.2f %0.2f %0.2f %0.2f'%(x0B, y0B, dxB, dyB))
            # 
            # ### convert pixel vals to dimless
            # x0A, y0A = transFigInv.transform_point([x0A, y0A])
            # dxA, dyA = transFigInv.transform_point([dxA, dyA])
            # x0B, y0B = transFigInv.transform_point([x0B, y0B])
            # dxB, dyB = transFigInv.transform_point([dxB, dyB])
            # #print('x0A, y0A, dxA, dyA : %0.2f %0.2f %0.2f %0.2f'%(x0A, y0A, dxA, dyA))
            # #print('x0B, y0B, dxB, dyB : %0.2f %0.2f %0.2f %0.2f'%(x0B, y0B, dxB, dyB))
            # #print('\n')
            
            x0n = x0
            #y0n = y0+(y0A-(y0B+dyB))-(offset_px/fig_px_y)
            y0n = y0+y_shift-(offset_px/fig_px_y)
            dxn = dx
            dyn = dy
            
            axs[i,j+1].set_position([x0n,y0n,dxn,dyn])
    
    return

def tight_layout_helper_ax_with_cbar(fig,ax,cax,**kwargs):
    '''
    shift cbar in x, then expand main axis dx to fill space
    --> call tight_layout(pad=X) first to set base padding
    '''
    
    transFigInv = fig.transFigure.inverted()
    transFig = fig.transFigure
    
    ### this is the ON SCREEN pixels size... has nothing to do with output
    fig_px_x, fig_px_y = fig.get_size_inches()*fig.dpi
    #print('fig size px : %0.3f %0.3f'%(fig_px_x, fig_px_y))
    
    ### axis position (dimless) --> NOT tightbox!
    x0A, y0A, dxA, dyA  = ax.get_position().bounds
    #print('x0A, y0A, dxA, dyA : %0.6f %0.6f %0.6f %0.6f'%(x0A, y0A, dxA, dyA))
    
    ### pixel values of axis frame (NOT tightbox)
    x0App, y0App = transFig.transform_point([x0A, y0A])
    dxApp, dyApp = transFig.transform_point([dxA, dyA])
    #x0Bpp, y0Bpp = transFig.transform_point([x0B, y0B])
    #dxBpp, dyBpp = transFig.transform_point([dxB, dyB])
    #print('x0App, y0App, dxApp, dyApp : %0.6f %0.6f %0.6f %0.6f'%(x0App, y0App, dxApp, dyApp))
    #print('x0Bpp, y0Bpp, dxBpp, dyBpp : %0.6f %0.6f %0.6f %0.6f'%(x0Bpp, y0Bpp, dxBpp, dyBpp))
    
    x0B, y0B, dxB, dyB  = cax.get_position().bounds
    #print('x0B, y0B, dxB, dyB : %0.6f %0.6f %0.6f %0.6f'%(x0B, y0B, dxB, dyB))
    
    ### pixel values of the axis tightbox (NOT the axis position)
    x0Ap, y0Ap, dxAp, dyAp = ax.get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=True).bounds
    #print('x0Ap, y0Ap, dxAp, dyAp : %0.6f %0.6f %0.6f %0.6f'%(x0Ap, y0Ap, dxAp, dyAp))
    
    x0Bp, y0Bp, dxBp, dyBp = cax.get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=True).bounds
    #print('x0Bp, y0Bp, dxBp, dyBp : %0.6f %0.6f %0.6f %0.6f'%(x0Bp, y0Bp, dxBp, dyBp))
    
    ### in pixels, figure out the x0 of the cbar axis
    ### use x0Ap : the number of pixels to the L of the main axis --> determined independently by tight_layout()
    x0Bpn = fig_px_x - x0Ap - dxBp
    x0Bn, _ = transFigInv.transform_point([x0Bpn, 0.])
    cax.set_position([x0Bn, y0B, dxB, dyB])
    
    end_ax_px_target = x0Bpn-2*x0Ap
    end_ax_px        = x0Ap + dxAp
    delta_px = end_ax_px_target - end_ax_px
    
    fac = (dxApp + delta_px)/dxApp
    
    ax.set_position([x0A, y0A, dxA*fac, dyA])
    x0Ap, y0Ap, dxAp, dyAp = ax.get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=True).bounds
    #print('x0Ap, y0Ap, dxAp, dyAp : %0.6f %0.6f %0.6f %0.6f'%(x0Ap, y0Ap, dxAp, dyAp))
    #print(x0Ap+dxAp)
    
    return

def cmap_convert_mpl_to_pview(cmap,fname,cmap_name,**kwargs):
    '''
    convert python cmap to JSON for Paraview
    '''
    
    # === we dont want to mess up the current cmap/norm when we set_array() and autoscale()
    cmapX = copy.deepcopy(cmap)
    #normX = copy.deepcopy(norm)
    
    N = kwargs.get('N',256)
    #lo = kwargs.get('lo',0.)
    #hi = kwargs.get('hi',1.)
    #norm = kwargs.get('norm',mpl.colors.Normalize(vmin=0.0, vmax=1.0))
    
    sclMap = mpl.cm.ScalarMappable(cmap=cmapX)
    x      = np.linspace(0,1,N+1)
    sclMap.set_array(x)
    sclMap.autoscale()
    colors = sclMap.to_rgba(x)
    
    # === output .json formatted ascii file
    #if os.path.exists(fname):
    #    sys.exit('file %s exists. exiting.'%fname)
    f = open(fname,'w')
    
    out_str='''[
    {
        "ColorSpace" : "RGB",
        "Name" : "%s",
        "RGBPoints" : 
        ['''%(cmap_name, )
    
    f.write(out_str)
    
    for i in range(len(x)):
        c = x[i]
        if (i==len(x)-1):
            maybeComma=''
        else:
            maybeComma=','
        color=colors[i]
        out_str='''
			%0.6f,
			%0.17f,
			%0.17f,
			%0.17f%s'''%(c,color[0],color[1],color[2],maybeComma)
        f.write(out_str)
    
    out_str = '''\n%s]\n%s}\n]'''%(8*' ',4*' ')
    f.write(out_str)
    
    f.close()
    print('--w-> %s'%fname)
    return

# main()
# ------------------------------------------------------------

if __name__ == '__main__':
    
    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    n_ranks = comm.Get_size()
    
    darkMode = True
    set_mpl_env(useTex=False, darkMode=darkMode)
    save_pdf = False
    png_px_x = 3840//2
    
    # ===
    
    hues = [323,282,190,130,92,60,30] ## Lch(ab) hues [degrees] : https://css.land/lch/
    if darkMode:
        L = 85; c = 155 ## luminance & chroma
    else:
        L = 55; c = 100 ## luminance & chroma
    
    colors = get_Lch_colors(hues,L=L,c=c,fmt='hex',test_plot=False)
    purple, blue, cyan, green, yellow, orange, red = colors
    colors_dark = hsv_adjust_hex(colors,0,0,-0.3)
    purple_dk, blue_dk, cyan_dk, green_dk, yellow_dk, orange_dk, red_dk = colors_dark
    cl1 = blue; cl2 = yellow; cl3 = red; cl4 = green; cl5 = purple; cl6 = orange; cl7 = cyan
    
    # ===
    
    if False: ## test data (ABC flow)
        with rgd('abc_flow.h5', 'w', force=True, driver='mpio', comm=comm, libver='latest') as f1:
            f1.populate_abc_flow(rx=2, ry=2, rz=2, nx=100, ny=100, nz=100, nt=100)
            f1.make_xdmf()
    
    if False: ## λ-2
        with rgd('abc_flow.h5','a', verbose=False, driver='mpio', comm=comm, libver='latest') as f1:
            f1.calc_lambda2(hiOrder=False, save_Q=True, save_lambda2=True, rt=n_ranks)
            f1.make_xdmf()
    
    MPI.Finalize()