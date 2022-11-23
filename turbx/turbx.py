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

# import numba
# from numba import jit, njit

import skimage
from skimage import color

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean, colorcet, cmasher

# import vtk
# from vtk.util import numpy_support

## required for EAS3
import struct

'''
========================================================================

Description
-----------

Tools for analysis of turbulent flow datasets
--> version Fall 2022

$> wget https://raw.githubusercontent.com/iagappel/turbx/main/turbx/turbx.py
$> git clone git@gitlab.iag.uni-stuttgart.de:transi/turbx.git
$> git clone git@github.com:iagappel/turbx.git

Notes
-----
- HDF5 Documentation : https://docs.h5py.org/_/downloads/en/3.2.1/pdf/
- compiling parallel HDF5 & h5py: https://docs.h5py.org/en/stable/mpi.html#building-against-parallel-hdf5

========================================================================
'''

# data container interface classes for HDF5 containers
# ======================================================================

class cgd(h5py.File):
    '''
    Curvilinear Grid Data (CGD)
    ---------------------------
    - super()'ed h5py.File class
    '''
    
    def __init__(self, *args, **kwargs):
        
        self.fname, openMode = args
        
        self.fname_path = os.path.dirname(self.fname)
        self.fname_base = os.path.basename(self.fname)
        self.fname_root, self.fname_ext = os.path.splitext(self.fname_base)
        
        ## default to libver='latest' if none provided
        if ('libver' not in kwargs):
            kwargs['libver'] = 'latest'
        
        ## catch possible user error --> could prevent accidental EAS overwrites
        if (self.fname_ext=='.eas'):
            raise ValueError('EAS4 files should not be opened with turbx.cgd()')
        
        ## determine if using mpi
        if ('driver' in kwargs) and (kwargs['driver']=='mpio'):
            self.usingmpi = True
        else:
            self.usingmpi = False
        
        ## determine communicator & rank info
        if self.usingmpi:
            self.comm    = kwargs['comm']
            self.n_ranks = self.comm.Get_size()
            self.rank    = self.comm.Get_rank()
        else:
            self.comm    = None
            self.n_ranks = 1
            self.rank    = 0
        
        ## if not using MPI, remove 'driver' and 'comm' from kwargs
        if ( not self.usingmpi ) and ('driver' in kwargs):
            kwargs.pop('driver')
        if ( not self.usingmpi ) and ('comm' in kwargs):
            kwargs.pop('comm')
        
        ## | mpiexec --mca io romio321 -n $NP python3 ...
        ## | mpiexec --mca io ompio -n $NP python3 ...
        ## | ompi_info --> print ompi settings ('MCA io' gives io implementation options)
        ## | export ROMIO_FSTYPE_FORCE="lustre:" --> force Lustre driver over UFS --> causes crash
        ## | export ROMIO_FSTYPE_FORCE="ufs:"
        ## | export ROMIO_PRINT_HINTS=1 --> show available hints
        
        ## determine MPI info / hints
        if self.usingmpi:
            if ('info' in kwargs):
                self.mpi_info = kwargs['info']
            else:
                mpi_info = MPI.Info.Create()
                ##
                mpi_info.Set('romio_ds_write' , 'disable'   )
                mpi_info.Set('romio_ds_read'  , 'disable'   )
                mpi_info.Set('romio_cb_read'  , 'automatic' )
                mpi_info.Set('romio_cb_write' , 'automatic' )
                #mpi_info.Set('romio_cb_read'  , 'enable' )
                #mpi_info.Set('romio_cb_write' , 'enable' )
                mpi_info.Set('cb_buffer_size' , str(int(round(8*1024**2))) ) ## 8 [MB]
                ##
                kwargs['info'] = mpi_info
                self.mpi_info = mpi_info
        
        ## | rdcc_nbytes:
        ## | ------------
        ## | Integer setting the total size of the raw data chunk cache for this dataset in bytes.
        ## | In most cases increasing this number will improve performance, as long as you have 
        ## | enough free memory. The default size is 1 MB
        
        ## --> gets passed to H5Pset_chunk_cache
        if ('rdcc_nbytes' not in kwargs):
            kwargs['rdcc_nbytes'] = int(16*1024**2) ## 16 [MB]
        
        ## | rdcc_nslots:
        ## | ------------
        ## | Integer defining the number of chunk slots in the raw data chunk cache for this dataset.
        
        ## if ('rdcc_nslots' not in kwargs):
        ##     kwargs['rdcc_nslots'] = 521
        
        ## cgd() unique kwargs (not h5py.File kwargs) --> pop() rather than get()
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
                                  
                                  >>> with cgd(<<fname>>,'w',force=True) as f:
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
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 1M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        ## touch, stripe
        elif (openMode == 'w') and not os.path.isfile(self.fname):
            if (self.rank==0):
                Path(self.fname).touch()
                if shutil.which('lfs') is not None:
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 1M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        else:
            pass
        
        if (self.comm is not None):
            self.comm.Barrier()
        
        ## call actual h5py.File.__init__()
        super(cgd, self).__init__(*args, **kwargs)
        self.get_header(verbose=verbose)
    
    def __enter__(self):
        '''
        for use with python 'with' statement
        '''
        #return self
        return super(cgd, self).__enter__()
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        '''
        for use with python 'with' statement
        '''
        if (self.rank==0):
            if exception_type is not None:
                print('\nsafely closed CGD HDF5 due to exception')
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
        return super(cgd, self).__exit__()
    
    def get_header(self,**kwargs):
        '''
        initialize header attributes of CGD class instance
        '''
        
        verbose = kwargs.get('verbose',True)
        read_grid = kwargs.get('read_grid',True)
        
        if (self.rank!=0):
            verbose=False
        
        if verbose: print('\n'+'cgd.get_header()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
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
            
            #if verbose: print(72*'-')
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
            if verbose: print(72*'-')
            #if verbose: print(72*'-'+'\n')
            
            # === write the 'derived' udef variables to a dict attribute of the CGD instance
            udef_char_deriv = ['rho_inf', 'mu_inf', 'nu_inf', 'a_inf', 'U_inf', 'cp', 'cv', 'r', 'Tw', 'Taw', 'lchar']
            udef_real_deriv = [ rho_inf,   mu_inf,   nu_inf,   a_inf,   U_inf,   cp,   cv,   r,   Tw,   Taw,   lchar ]
            self.udef_deriv = dict(zip(udef_char_deriv, udef_real_deriv))
        
        else:
            pass
        
        # === read coordinate vectors
        # - this can get slow for big 3D meshes
        # - every rank reads full grid
        
        if all([('dims/x' in self),('dims/y' in self),('dims/z' in self)]) and read_grid:
            
            if self.usingmpi: self.comm.Barrier()
            t_start = timeit.default_timer()
            
            ## read 3D/1D coordinate arrays
            ## ( dont transpose right away --> allows for 1D storage )
            
            if self.usingmpi:
                
                dset = self['dims/x']
                with dset.collective:
                    x = self.x = np.copy( dset[()] )
                
                self.comm.Barrier()
                
                dset = self['dims/y']
                with dset.collective:
                    y = self.y = np.copy( dset[()] )
                
                self.comm.Barrier()
                
                dset = self['dims/z']
                with dset.collective:
                    z = self.z = np.copy( dset[()] )
            
            else:
                
                x = self.x = np.copy( self['dims/x'][()] )
                y = self.y = np.copy( self['dims/y'][()] )
                z = self.z = np.copy( self['dims/z'][()] )
            
            if self.usingmpi: self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = ( x.nbytes + y.nbytes + z.nbytes ) * self.n_ranks / 1024**3
            if verbose:
                even_print('read x,y,z (full)', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
            
            if True: ## transpose the coordinate arrays
                
                '''
                nx,ny,nz should probably just be stored as attributes
                '''
                
                if (x.ndim==1):
                    nx = self.nx = x.shape[0]
                elif (x.ndim==3):
                    x  = self.x  = np.copy( x.T )
                    nx = self.nx = x.shape[0]
                else:
                    raise AssertionError('x.ndim=%i'%(x.ndim,))
                
                if (y.ndim==1):
                    ny = self.ny = y.shape[0]
                elif (y.ndim==3):
                    y  = self.y  = np.copy( y.T )
                    ny = self.ny = y.shape[1]
                else:
                    raise AssertionError('y.ndim=%i'%(y.ndim,))
                
                if (z.ndim==1):
                    nz = self.nz = z.shape[0]
                elif (z.ndim==3):
                    z  = self.z  = np.copy( z.T )
                    nz = self.nz = z.shape[2]
                else:
                    raise AssertionError('z.ndim=%i'%(z.ndim,))
            
            ## #if verbose:
            ## if (self.rank==0):
            ##     print('in header()')
            ##     print('x.min() = %0.3e , x.max() = %0.3e'%(self.x.min(),self.x.max()))
            ##     print('y.min() = %0.3e , y.max() = %0.3e'%(self.y.min(),self.y.max()))
            ##     print('z.min() = %0.3e , z.max() = %0.3e'%(self.z.min(),self.z.max()))
            
            ngp = self.ngp = nx*ny*nz
            
            if verbose: even_print('nx',  '%i'%nx  )
            if verbose: even_print('ny',  '%i'%ny  )
            if verbose: even_print('nz',  '%i'%nz  )
            if verbose: even_print('ngp', '%i'%ngp )
            #if verbose: print(72*'-')
            
            if False:
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
        
        # === time vector
        
        if ('dims/t' in self):
            
            self.t = np.copy(self['dims/t'][:])
            
            if ('data' in self): ## check t dim and data arr agree
                nt,_,_,_ = self['data/%s'%list(self['data'].keys())[0]].shape
                if (nt!=self.t.size):
                    raise AssertionError('nt!=self.t.size : %i!=%i'%(nt,self.t.size))
            
            try:
                self.dt = self.t[1] - self.t[0]
            except IndexError:
                self.dt = 0.
            
            self.nt       = nt       = self.t.size
            self.duration = duration = self.t[-1] - self.t[0]
            self.ti       = ti       = np.arange(self.nt, dtype=np.int64)
        
        elif all([ ('data' in self) , ('dims/t' not in self) ]): ## data but no time --> make dummy time vector
            self.scalars = list(self['data'].keys())
            nt,_,_,_ = self['data/%s'%self.scalars[0]].shape
            self.nt  = nt
            self.t   =      np.arange(self.nt, dtype=np.float64)
            self.ti  = ti = np.arange(self.nt, dtype=np.int64)
            self.dt  = 1.
            self.duration = duration = self.t[-1]-self.t[0]
        
        else: ## no data, no time
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
    
    def init_from_eas4(self, fn_eas4, **kwargs):
        '''
        initialize a CGD from an EAS4 (NS3D output format)
        '''
        
        EAS4=1
        IEEES=1; IEEED=2
        EAS4_NO_G=1; EAS4_X0DX_G=2; EAS4_UDEF_G=3; EAS4_ALL_G=4; EAS4_FULL_G=5
        gmode_dict = {1:'EAS4_NO_G', 2:'EAS4_X0DX_G', 3:'EAS4_UDEF_G', 4:'EAS4_ALL_G', 5:'EAS4_FULL_G'}
        
        verbose = kwargs.get('verbose',True)
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        
        if (rx*ry*rz != self.n_ranks):
            raise AssertionError('rx*ry*rz != self.n_ranks')
        
        if (self.rank!=0):
            verbose=False

        # # === spatial resolution filter : take every nth grid point
        # sx = kwargs.get('sx',1)
        # sy = kwargs.get('sy',1)
        # sz = kwargs.get('sz',1)
        # #st = kwargs.get('st',1)
        
        # # === spatial resolution filter : set x/y/z bounds
        # x_min = kwargs.get('x_min',None)
        # y_min = kwargs.get('y_min',None)
        # z_min = kwargs.get('z_min',None)
        # 
        # x_max = kwargs.get('x_max',None)
        # y_max = kwargs.get('y_max',None)
        # z_max = kwargs.get('z_max',None)
        # 
        # xi_min = kwargs.get('xi_min',None)
        # yi_min = kwargs.get('yi_min',None)
        # zi_min = kwargs.get('zi_min',None)
        # 
        # xi_max = kwargs.get('xi_max',None)
        # yi_max = kwargs.get('yi_max',None)
        # zi_max = kwargs.get('zi_max',None)
        
        ## grid filters are currently not supported for CGD
        self.hasGridFilter=False
        
        if verbose: print('\n'+'cgd.init_from_eas4()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        if verbose: even_print('infile', os.path.basename(fn_eas4))
        if verbose: even_print('infile size', '%0.2f [GB]'%(os.path.getsize(fn_eas4)/1024**3))
        if verbose: even_print('outfile', self.fname)
        
        with eas4(fn_eas4, 'r', verbose=False, driver=self.driver, comm=MPI.COMM_WORLD, libver='latest') as hf_eas4:
            
            if verbose: even_print( 'gmode dim1' , '%i / %s'%( hf_eas4.gmode_dim1_orig, gmode_dict[hf_eas4.gmode_dim1_orig] ) )
            if verbose: even_print( 'gmode dim2' , '%i / %s'%( hf_eas4.gmode_dim2_orig, gmode_dict[hf_eas4.gmode_dim2_orig] ) )
            if verbose: even_print( 'gmode dim3' , '%i / %s'%( hf_eas4.gmode_dim3_orig, gmode_dict[hf_eas4.gmode_dim3_orig] ) )
            
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
                raise ValueError('dims/x, dims/y, dims/z already in CGD file')
            
            x = np.copy(hf_eas4.x)
            y = np.copy(hf_eas4.y)
            z = np.copy(hf_eas4.z)
            
            nx = hf_eas4.nx
            ny = hf_eas4.ny
            nz = hf_eas4.nz
            
            ngp = nx*ny*nz
            ## nt = hf_eas4.nt --> no time data yet
            
            ## if dimensions are EAS4_ALL_G (4), then bcast to 3D
            if all([ (hf_eas4.gmode_dim1==4) , (hf_eas4.gmode_dim2==4) , (hf_eas4.gmode_dim3==4) ]):
                x, y, z = np.meshgrid(x, y, z, indexing='ij')
            
            if (x.ndim!=3):
                raise ValueError('turbx.cgd() requires FULLG / 3D x,y,z')
            if (y.ndim!=3):
                raise ValueError('turbx.cgd() requires FULLG / 3D x,y,z')
            if (z.ndim!=3):
                raise ValueError('turbx.cgd() requires FULLG / 3D x,y,z')
            
            ## broadcast in dimensions with shape=1
            ## EAS4_FULL_G=5
            if ( hf_eas4.gmode_dim1==5 ) and ( x.shape != (nx,ny,nz) ):
                x = np.broadcast_to(x, (nx,ny,nz))
                if verbose: print('broadcasted x')
            
            if ( hf_eas4.gmode_dim2==5 ) and ( y.shape != (nx,ny,nz) ):
                y = np.broadcast_to(y, (nx,ny,nz))
                if verbose: print('broadcasted y')
            
            if ( hf_eas4.gmode_dim3==5 ) and ( z.shape != (nx,ny,nz) ):
                z = np.broadcast_to(z, (nx,ny,nz))
                if verbose: print('broadcasted z')
            
            shape  = (nz,ny,nx)
            chunks = rgd.chunk_sizer(nxi=shape, constraint=(None,None,None), size_kb=2*1024, base=2, data_byte=8) ## 2 [MB]
            
            # === ranks
            
            if self.usingmpi:
                
                self.nx = nx
                self.ny = ny
                self.nz = nz
                
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
            else:
                nxr = self.nx
                nyr = self.ny
                nzr = self.nz
                #ntr = self.nt
            
            # === write coord arrays
            
            if ('dims/x' in self):
                del self['dims/x']
            if ('dims/y' in self):
                del self['dims/y']
            if ('dims/z' in self):
                del self['dims/z']
            
            if False: ## serial write
                
                if self.usingmpi: self.comm.Barrier()
                t_start = timeit.default_timer()
                
                dset = self.create_dataset('dims/x', data=x.T, shape=shape, chunks=chunks)
                dset = self.create_dataset('dims/y', data=y.T, shape=shape, chunks=chunks)
                dset = self.create_dataset('dims/z', data=z.T, shape=shape, chunks=chunks)
                
                if self.usingmpi: self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                data_gb = ( x.nbytes + y.nbytes + z.nbytes ) / 1024**3
                if verbose:
                    even_print('write x,y,z', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
            
            if True: ## collective write
                
                ## initialize datasets
                dset_x = self.create_dataset('dims/x', shape=shape, chunks=chunks, dtype=x.dtype)
                dset_y = self.create_dataset('dims/y', shape=shape, chunks=chunks, dtype=y.dtype)
                dset_z = self.create_dataset('dims/z', shape=shape, chunks=chunks, dtype=z.dtype)
                
                chunk_kb_ = np.prod(dset_x.chunks)*8 / 1024. ## actual
                if verbose:
                    even_print('chunk shape (z,y,x)','%s'%str(dset_x.chunks))
                    even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
                
                if self.usingmpi: self.comm.Barrier()
                t_start = timeit.default_timer()
                
                if self.usingmpi: 
                    with dset_x.collective:
                        dset_x[rz1:rz2,ry1:ry2,rx1:rx2] = x[rx1:rx2,ry1:ry2,rz1:rz2].T
                else:
                    dset_x[:,:,:] = x.T
                
                if self.usingmpi: self.comm.Barrier()
                
                if self.usingmpi: 
                    with dset_y.collective:
                        dset_y[rz1:rz2,ry1:ry2,rx1:rx2] = y[rx1:rx2,ry1:ry2,rz1:rz2].T
                else:
                    dset_y[:,:,:] = y.T
                
                if self.usingmpi: self.comm.Barrier()
                
                if self.usingmpi: 
                    with dset_z.collective:
                        dset_z[rz1:rz2,ry1:ry2,rx1:rx2] = z[rx1:rx2,ry1:ry2,rz1:rz2].T
                else:
                    dset_z[:,:,:] = z.T
                
                if self.usingmpi: self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                
                data_gb = ( x.nbytes + y.nbytes + z.nbytes ) / 1024**3
                
                if verbose:
                    even_print('write x,y,z', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
            
                # === write a preliminary time array --> e.g. for baseflow
                
                if ('dims/t' in self):
                    del self['dims/t']
                dset = self.create_dataset( 'dims/t', data=hf_eas4.t )
        
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        
        if verbose: print(72*'-')
        self.get_header(verbose=True, read_grid=True)
        
        return
    
    def init_from_cgd(self, fn_cgd, **kwargs):
        '''
        initialize an CGD from an CGD (copy over header data & coordinate data)
        '''
        
        t_info = kwargs.get('t_info',True)
        chunk_kb = kwargs.get('chunk_kb',4*1024) ## 2 [MB]
        
        verbose = kwargs.get('verbose',True)
        if (self.rank!=0):
            verbose=False
        
        with cgd(fn_cgd, 'r', driver=self.driver, comm=self.comm) as hf_ref:
            
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
            
            x   = self.x   = hf_ref.x
            y   = self.y   = hf_ref.y
            z   = self.z   = hf_ref.z
            nx  = self.nx  = hf_ref.nx
            ny  = self.ny  = hf_ref.ny
            nz  = self.nz  = hf_ref.nz
            ngp = self.ngp = hf_ref.ngp
            if ('dims/x' in self):
                del self['dims/x']
            if ('dims/y' in self):
                del self['dims/y']
            if ('dims/z' in self):
                del self['dims/z']
            
            ## 1D
            #self.create_dataset('dims/x', data=x)
            #self.create_dataset('dims/y', data=y)
            #self.create_dataset('dims/z', data=z)
            
            ## 3D
            shape  = (nz,ny,nx)
            chunks = rgd.chunk_sizer(nxi=shape, constraint=(None,None,None), size_kb=chunk_kb, base=2, data_byte=8)
            
            if self.usingmpi: self.comm.Barrier()
            t_start = timeit.default_timer()
            dset = self.create_dataset('dims/x', data=x.T, shape=shape, chunks=chunks)
            dset = self.create_dataset('dims/y', data=y.T, shape=shape, chunks=chunks)
            dset = self.create_dataset('dims/z', data=z.T, shape=shape, chunks=chunks)
            if self.usingmpi: self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = ( x.nbytes + y.nbytes + z.nbytes ) / 1024**3
            if verbose:
                even_print('write x,y,z', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
            
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
        import data from a series of EAS4 files to a CGD
        '''
        
        if (self.rank!=0):
            verbose=False
        else:
            verbose=True
        
        EAS4=1
        IEEES=1; IEEED=2
        EAS4_NO_G=1; EAS4_X0DX_G=2; EAS4_UDEF_G=3; EAS4_ALL_G=4; EAS4_FULL_G=5
        gmode_dict = {1:'EAS4_NO_G', 2:'EAS4_X0DX_G', 3:'EAS4_UDEF_G', 4:'EAS4_ALL_G', 5:'EAS4_FULL_G'}
        
        if verbose: print('\n'+'cgd.import_eas4()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        ti_min = kwargs.get('ti_min',None)
        ti_max = kwargs.get('ti_max',None)
        tt_min = kwargs.get('tt_min',None)
        tt_max = kwargs.get('tt_max',None)
        
        chunk_kb = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
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
        
        ## update this CGD's header and attributes
        self.get_header(verbose=False)
        
        # === get all time info
        
        comm_eas4 = MPI.COMM_WORLD
        t = np.array([], dtype=np.float64)
        for fn_eas4 in fn_eas4_list:
            with eas4(fn_eas4, 'r', verbose=False, driver=self.driver, comm=comm_eas4, libver='latest') as hf_eas4:
                t = np.concatenate((t, hf_eas4.t))
        comm_eas4.Barrier()
        
        if verbose: even_print('n EAS4 files','%i'%len(fn_eas4_list))
        if verbose: even_print('nt all files','%i'%t.size)
        
        if (t.size>1):
            
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
        
        # === get all grid info & check
        
        # TODO : compare coordinate arrays for series of EAS4 files
        
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
        
        # === CGD times
        self.t  = np.copy(t[doRead])
        self.nt = self.t.size
        self.ti = np.arange(self.nt, dtype=np.int64)
        
        # === write back self.t to file
        if ('dims/t' in self):
            del self['dims/t']
        self.create_dataset('dims/t', data=self.t)
        
        if self.usingmpi:
            comm4d = self.comm.Create_cart(dims=[rx,ry,rz,rt], periods=[False,False,False,False], reorder=False)
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
        else:
            nxr = self.nx
            nyr = self.ny
            nzr = self.nz
            #ntr = self.nt
        
        # === determine CGD scalars (from EAS4 scalars)
        if not hasattr(self, 'scalars') or (len(self.scalars)==0):
            with eas4(fn_eas4_list[0], 'r', verbose=False, driver=self.driver, comm=comm_eas4) as hf_eas4:
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
            
            dset = self.create_dataset('data/%s'%scalar, 
                                       shape=shape, 
                                       dtype=np.float32,
                                       chunks=chunks)
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
        if verbose: print(72*'-')
        
        # === report size of CGD after initialization
        if verbose: tqdm.write(even_print(os.path.basename(self.fname), '%0.2f [GB]'%(os.path.getsize(self.fname)/1024**3), s=True))
        if verbose: print(72*'-')
        
        # === open EAS4s, read, write to CGD
        
        if verbose: progress_bar = tqdm(total=(self.nt*self.n_scalars), ncols=100, desc='import', leave=False, file=sys.stdout)
        
        data_gb_read  = 0.
        data_gb_write = 0.
        t_read  = 0.
        t_write = 0.
        
        tii  = -1 ## counter full series
        tiii = -1 ## counter CGD-local
        for fn_eas4 in fn_eas4_list:
            with eas4(fn_eas4, 'r', verbose=False, driver=self.driver, comm=comm_eas4) as hf_eas4:
                
                if verbose: tqdm.write(even_print(os.path.basename(fn_eas4), '%0.2f [GB]'%(os.path.getsize(fn_eas4)/1024**3), s=True))
                ##
                # if verbose: tqdm.write(even_print('gmode_dim1' , '%i'%hf_eas4.gmode_dim1  , s=True))
                # if verbose: tqdm.write(even_print('gmode_dim2' , '%i'%hf_eas4.gmode_dim2  , s=True))
                # if verbose: tqdm.write(even_print('gmode_dim3' , '%i'%hf_eas4.gmode_dim3  , s=True))
                ##
                if verbose: tqdm.write(even_print( 'gmode dim1' , '%i / %s'%( hf_eas4.gmode_dim1_orig, gmode_dict[hf_eas4.gmode_dim1_orig] ), s=True ))
                if verbose: tqdm.write(even_print( 'gmode dim2' , '%i / %s'%( hf_eas4.gmode_dim2_orig, gmode_dict[hf_eas4.gmode_dim2_orig] ), s=True ))
                if verbose: tqdm.write(even_print( 'gmode dim3' , '%i / %s'%( hf_eas4.gmode_dim3_orig, gmode_dict[hf_eas4.gmode_dim3_orig] ), s=True ))
                ##
                if verbose: tqdm.write(even_print('duration'   , '%0.2f'%hf_eas4.duration , s=True))
                
                # === write buffer
                
                # ## 5D [scalar][x,y,z,t] structured array
                # buff = np.zeros(shape=(nxr, nyr, nzr, bt), dtype={'names':self.scalars, 'formats':self.scalars_dtypes})
                
                # ===
                
                #domainName = 'DOMAIN_000000' ## only one domain supported
                domainName = hf_eas4.domainName
                
                for ti in range(hf_eas4.nt):
                    tii += 1 ## EAS4 series counter
                    if doRead[tii]:
                        tiii += 1 ## CGD counter
                        for scalar in hf_eas4.scalars:
                            if (scalar in self.scalars):
                                
                                # === collective read
                                
                                dset_path = 'Data/%s/ts_%06d/par_%06d'%(domainName,ti,hf_eas4.scalar_n_map[scalar])
                                dset = hf_eas4[dset_path]
                                
                                if hf_eas4.usingmpi: comm_eas4.Barrier()
                                t_start = timeit.default_timer()
                                if hf_eas4.usingmpi: 
                                    with dset.collective:
                                        data = dset[rx1:rx2,ry1:ry2,rz1:rz2]
                                else:
                                    data = dset[()]
                                if hf_eas4.usingmpi: comm_eas4.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                
                                data_gb       = data.nbytes / 1024**3
                                t_read       += t_delta
                                data_gb_read += data_gb
                                
                                if False:
                                    if verbose:
                                        txt = even_print('read', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                        tqdm.write(txt)
                                
                                # === reduce precision (e.g. for restarts)
                                
                                if (data.dtype == np.float64):
                                    data = np.copy( data.astype(np.float32) )
                                data_gb = data.nbytes / 1024**3
                                
                                # === collective write
                                
                                dset = self['data/%s'%scalar]
                                
                                if self.usingmpi: self.comm.Barrier()
                                t_start = timeit.default_timer()
                                if self.usingmpi:
                                    with dset.collective:
                                        dset[tiii,rz1:rz2,ry1:ry2,rx1:rx2] = data.T
                                else:
                                    
                                    if self.hasGridFilter:
                                        data = data[self.xfi[:,np.newaxis,np.newaxis],
                                                    self.yfi[np.newaxis,:,np.newaxis],
                                                    self.zfi[np.newaxis,np.newaxis,:]]
                                    
                                    dset[tiii,:,:,:] = data.T
                                
                                if self.usingmpi: self.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                
                                t_write       += t_delta
                                data_gb_write += data_gb
                                
                                if False:
                                    if verbose:
                                        txt = even_print('write', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                        tqdm.write(txt)
                                
                                if verbose: progress_bar.update()
                if hf_eas4.usingmpi: comm_eas4.Barrier()
        
        if verbose: progress_bar.close()
        
        if self.usingmpi: self.comm.Barrier()
        self.get_header(verbose=False, read_grid=True)
        
        ## get read read/write totals all ranks
        if self.usingmpi:
            G = self.comm.gather([data_gb_read, data_gb_write, self.rank], root=0)
            G = self.comm.bcast(G, root=0)
            data_gb_read  = sum([x[0] for x in G])
            data_gb_write = sum([x[1] for x in G])
        
        if verbose: print(72*'-')
        if verbose: even_print('nt',       '%i'%self.nt )
        if verbose: even_print('dt',       '%0.6f'%self.dt )
        if verbose: even_print('duration', '%0.2f'%self.duration )
        
        if verbose: print(72*'-')
        if verbose: even_print('time read',format_time_string(t_read))
        if verbose: even_print('time write',format_time_string(t_write))
        if verbose: even_print(self.fname, '%0.2f [GB]'%(os.path.getsize(self.fname)/1024**3))
        if verbose: even_print('read total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_read,t_read,(data_gb_read/t_read)))
        if verbose: even_print('write total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_write,t_write,(data_gb_write/t_write)))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : cgd.import_eas4() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    # === test data populators
    
    def make_test_file(self, **kwargs):
        '''
        make a test CGD file
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'cgd.make_test_file()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        ##
        chunk_kb = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
        self.nx = nx = kwargs.get('nx',100)
        self.ny = ny = kwargs.get('ny',100)
        self.nz = nz = kwargs.get('nz',100)
        self.nt = nt = kwargs.get('nt',100)
        
        data_gb = 3 * 4*nx*ny*nz*nt / 1024.**3
        if verbose: even_print(self.fname, '%0.2f [GB]'%(data_gb,))
        
        if (rx*ry*rz != self.n_ranks):
            raise AssertionError('rx*ry*rz != self.n_ranks')
        
        # === rank mapping
        
        if self.usingmpi:
            
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
        
        else:
            
            nxr = self.nx
            nyr = self.ny
            nzr = self.nz
            #ntr = self.nt
        
        # ===
        
        ## 1D coordinates
        x = np.linspace(0., 2*np.pi, nx, dtype=np.float64)
        y = np.linspace(0., 2*np.pi, ny, dtype=np.float64)
        z = np.linspace(0., 2*np.pi, nz, dtype=np.float64)
        t = 0.1 * np.arange(nt, dtype=np.float64)
        
        ## 3D coordinates
        x,y,z = np.meshgrid(x,y,z, indexing='ij')
        
        ## per-rank dim range
        if self.usingmpi:
            xr = x[rx1:rx2,ry1:ry2,rz1:rz2]
            yr = y[rx1:rx2,ry1:ry2,rz1:rz2]
            zr = z[rx1:rx2,ry1:ry2,rz1:rz2]
            #tr = t[rt1:rt2]
            tr = np.copy(t)
        else:
            xr = np.copy(x)
            yr = np.copy(y)
            zr = np.copy(z)
            tr = np.copy(t)
        
        # === apply deformation to mesh
        
        x_ = np.copy(x)
        y_ = np.copy(y)
        z_ = np.copy(z)
        
        if False:
            
            ## xy
            x += 0.2*np.sin(1*y_)
            #x += 0.2*np.sin(1*z_)
            
            ## yz
            #y += 0.2*np.sin(1*z_)
            y += 0.2*np.sin(1*x_)
            
            ## zy
            #z += 0.2*np.sin(1*x_)
            #z += 0.2*np.sin(1*y_)
        
        if True:
            
            ## xy
            x += 0.2*np.sin(1*y_)
            x += 0.2*np.sin(1*z_)
            
            ## yz
            y += 0.2*np.sin(1*z_)
            y += 0.2*np.sin(1*x_)
            
            ## zy
            z += 0.2*np.sin(1*x_)
            z += 0.2*np.sin(1*y_)
        
        x_ = None; del x_
        y_ = None; del y_
        z_ = None; del z_
        
        # self.x = x
        # self.y = y
        # self.z = z
        
        # === write coord arrays
        
        shape  = (nz,ny,nx)
        chunks = rgd.chunk_sizer(nxi=shape, constraint=(None,None,None), size_kb=chunk_kb, base=2, data_byte=8)
        
        # === write coordinate datasets (independent)
        
        # dset = self.create_dataset('dims/x', data=x.T, shape=shape, chunks=chunks)
        # dset = self.create_dataset('dims/y', data=y.T, shape=shape, chunks=chunks)
        # dset = self.create_dataset('dims/z', data=z.T, shape=shape, chunks=chunks)
        
        # === initialize coordinate datasets
        
        data_gb = 4*nx*ny*nz / 1024.**3
        for scalar in ['x','y','z']:
            if ('data/%s'%scalar in self):
                del self['data/%s'%scalar]
            if verbose:
                even_print('initializing dims/%s'%(scalar,),'%0.2f [GB]'%(data_gb,))
            dset = self.create_dataset('dims/%s'%scalar, 
                                        shape=shape,
                                        dtype=np.float64,
                                        chunks=chunks )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
        if verbose: print(72*'-')
        
        # === write coordinate datasets (collective)
        
        data_gb = 8*nx*ny*nz / 1024.**3
        
        if self.usingmpi: self.comm.Barrier()
        t_start = timeit.default_timer()
        ds = self['dims/x']
        if self.usingmpi:
            with ds.collective:
                ds[rz1:rz2,ry1:ry2,rx1:rx2] = x[rx1:rx2,ry1:ry2,rz1:rz2].T
        else:
            ds[:,:,:] = x.T
        if self.usingmpi: self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('write: dims/x','%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        if self.usingmpi: self.comm.Barrier()
        t_start = timeit.default_timer()
        ds = self['dims/y']
        if self.usingmpi:
            with ds.collective:
                ds[rz1:rz2,ry1:ry2,rx1:rx2] = y[rx1:rx2,ry1:ry2,rz1:rz2].T
        else:
            ds[:,:,:] = y.T
        if self.usingmpi: self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('write: dims/y','%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        if self.usingmpi: self.comm.Barrier()
        t_start = timeit.default_timer()
        ds = self['dims/z']
        if self.usingmpi:
            with ds.collective:
                ds[rz1:rz2,ry1:ry2,rx1:rx2] = z[rx1:rx2,ry1:ry2,rz1:rz2].T
        else:
            ds[:,:,:] = z.T
        if self.usingmpi: self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('write: dims/z','%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        # ===
        
        shape  = (self.nt,self.nz,self.ny,self.nx)
        chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2, data_byte=4)
        
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
                                        chunks=chunks )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
        if verbose: print(72*'-')
        
        # === make 4D ABC flow data
        
        t_start = timeit.default_timer()
        A = np.sqrt(3)
        B = np.sqrt(2)
        C = 1.
        na = np.newaxis
        u = (A + 0.5 * tr[na,na,na,:] * np.sin(np.pi*tr[na,na,na,:])) * np.sin(zr[:,:,:,na]) + \
             B * np.cos(yr[:,:,:,na]) + \
             0.*xr[:,:,:,na]
        v = B * np.sin(xr[:,:,:,na]) + \
            C * np.cos(zr[:,:,:,na]) + \
            0.*yr[:,:,:,na] + \
            0.*tr[na,na,na,:]
        w = C * np.sin(yr[:,:,:,na]) + \
            (A + 0.5 * tr[na,na,na,:] * np.sin(np.pi*tr[na,na,na,:])) * np.cos(xr[:,:,:,na]) + \
            0.*zr[:,:,:,na]
        
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('calc flow','%0.3f [s]'%(t_delta,))
        
        # ===
        
        data_gb = 4*nx*ny*nz*nt / 1024.**3
        
        if self.usingmpi: self.comm.Barrier()
        t_start = timeit.default_timer()
        ds = self['data/u']
        if self.usingmpi:
            with ds.collective:
                ds[:,rz1:rz2,ry1:ry2,rx1:rx2] = u.T
        else:
            ds[:,:,:,:] = u.T
        if self.usingmpi: self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('write: u','%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        if self.usingmpi: self.comm.Barrier()
        t_start = timeit.default_timer()
        ds = self['data/v']
        if self.usingmpi:
            with ds.collective:
                ds[:,rz1:rz2,ry1:ry2,rx1:rx2] = v.T
        else:
            ds[:,:,:,:] = v.T
        if self.usingmpi: self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('write: v','%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        if self.usingmpi: self.comm.Barrier()
        t_start = timeit.default_timer()
        ds = self['data/w']
        if self.usingmpi:
            with ds.collective:
                ds[:,rz1:rz2,ry1:ry2,rx1:rx2] = w.T
        else:
            ds[:,:,:,:] = w.T
        if self.usingmpi: self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('write: w','%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        # ===
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.populate_abc_flow() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    # === post-processing
    
    def calc_grid_metrics(self,):
        '''
        calculate the metric tensor M and save it to the file
        '''
        pass
        return
    
    def get_mean(self, **kwargs):
        '''
        get mean in [t] --> leaves [x,y,z,1]
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'cgd.get_mean()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        #rt = kwargs.get('rt',1)
        #rt = 1
        
        fn_cgd_mean  = kwargs.get('fn_cgd_mean',None)
        #sfm         = kwargs.get('scalars',None) ## scalars to take (for mean)
        ti_min       = kwargs.get('ti_min',None)
        favre        = kwargs.get('favre',True)
        reynolds     = kwargs.get('reynolds',True)
        ##
        force        = kwargs.get('force',False)
        
        chunk_kb     = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
        if (rx*ry*rz != self.n_ranks):
            raise AssertionError('rx*ry*rz != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        if (ti_min is not None):
            if not isinstance(ti_min, int):
                raise TypeError('ti_min must be type int')
        
        if self.usingmpi:
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
        else:
            nxr = self.nx
            nyr = self.ny
            nzr = self.nz
            #ntr = self.nt
        
        # === mean file name (for writing)
        if (fn_cgd_mean is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_mean_h5_base = fname_root+'_mean.h5'
            #fn_cgd_mean = os.path.join(fname_path, fname_mean_h5_base)
            fn_cgd_mean = str(PurePosixPath(fname_path, fname_mean_h5_base))
            #fn_cgd_mean = Path(fname_path, fname_mean_h5_base)
        
        if verbose: even_print('fn_cgd'       , self.fname   )
        if verbose: even_print('fn_cgd_mean'  , fn_cgd_mean  )
        #if verbose: even_print('fn_cgd_prime' , fn_cgd_prime )
        if verbose: even_print('do Favre avg' , str(favre)   )
        if verbose: even_print('do Reynolds avg' , str(reynolds)   )
        if verbose: print(72*'-')
        if verbose: even_print('nx','%i'%self.nx)
        if verbose: even_print('ny','%i'%self.ny)
        if verbose: even_print('nz','%i'%self.nz)
        if verbose: even_print('nt','%i'%self.nt)
        if verbose: print(72*'-')
        
        ## get times to take for avg
        if (ti_min is not None):
            ti_for_avg = np.copy( self.ti[ti_min:] )
        else:
            ti_for_avg = np.copy( self.ti )
        
        nt_avg       = ti_for_avg.shape[0]
        t_avg_start  = self.t[ti_for_avg[0]]
        t_avg_end    = self.t[ti_for_avg[-1]]
        duration_avg = t_avg_end - t_avg_start
        
        if verbose: even_print('n timesteps avg','%i/%i'%(nt_avg,self.nt))
        if verbose: even_print('t index avg start','%i'%(ti_for_avg[0],))
        if verbose: even_print('t index avg end','%i'%(ti_for_avg[-1],))
        if verbose: even_print('t avg start','%0.2f [-]'%(t_avg_start,))
        if verbose: even_print('t avg end','%0.2f [-]'%(t_avg_end,))
        if verbose: even_print('duration avg','%0.2f [-]'%(duration_avg,))
        if verbose: print(72*'-')
        
        ## performance
        t_read = 0.
        t_write = 0.
        data_gb_read = 0.
        data_gb_write = 0.
        
        #data_gb      = 4*self.nx*self.ny*self.nz*self.nt / 1024**3
        data_gb      = 4*self.nx*self.ny*self.nz*nt_avg / 1024**3
        data_gb_mean = 4*self.nx*self.ny*self.nz*1      / 1024**3
        
        scalars_re = ['u','v','w','p','T','rho']
        scalars_fv = ['u','v','w','p','T','rho']
        
        #with cgd(fn_cgd_mean, 'w', force=force, driver='mpio', comm=MPI.COMM_WORLD) as hf_mean:
        with cgd(fn_cgd_mean, 'w', force=force, driver=self.driver, comm=self.comm) as hf_mean:
            
            hf_mean.attrs['duration_avg'] = duration_avg ## duration of mean
            #hf_mean.attrs['duration_avg'] = self.duration
            
            hf_mean.init_from_cgd(self.fname) ## initialize the mean file from the cgd file
            
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
            
            if self.usingmpi: self.comm.Barrier()
            if verbose: print(72*'-')
            
            # === read rho
            if favre:
                dset = self['data/rho']
                if self.usingmpi: self.comm.Barrier()
                t_start = timeit.default_timer()
                if self.usingmpi: 
                    with dset.collective:
                        #rho = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                        rho = dset[ti_min:,rz1:rz2,ry1:ry2,rx1:rx2].T
                else:
                    #rho = dset[()].T
                    rho = dset[ti_min:,:,:,:].T
                if self.usingmpi: self.comm.Barrier()
                
                t_delta = timeit.default_timer() - t_start
                if (self.rank==0):
                    txt = even_print('read: rho', '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                    tqdm.write(txt)
                
                t_read       += t_delta
                data_gb_read += data_gb
                
                ## mean ρ in [t] --> leave [x,y,z]
                rho_mean = np.mean(rho, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
            
            # === read, do mean, write
            for scalar in self.scalars:
                
                # === collective read
                dset = self['data/%s'%scalar]
                if self.usingmpi: self.comm.Barrier()
                t_start = timeit.default_timer()
                if self.usingmpi:
                    with dset.collective:
                        #data = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                        data = dset[ti_min:,rz1:rz2,ry1:ry2,rx1:rx2].T
                else:
                    #data = dset[()].T
                    data = dset[ti_min:,:,:,:].T
                if self.usingmpi: self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                
                if (self.rank==0):
                    txt = even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                    tqdm.write(txt)
                
                t_read       += t_delta
                data_gb_read += data_gb
                
                # === do mean in [t]
                if reynolds:
                    data_mean    = np.mean(data,     axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
                if favre:
                    data_mean_fv = np.mean(data*rho, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32) / rho_mean
                
                # === write
                if reynolds:
                    if scalar in scalars_re:
                        
                        dset = hf_mean['data/%s'%scalar]
                        if self.usingmpi: self.comm.Barrier()
                        t_start = timeit.default_timer()
                        if self.usingmpi:
                            with dset.collective:
                                dset[:,rz1:rz2,ry1:ry2,rx1:rx2] = data_mean.T
                        else:
                            dset[:,:,:,:] = data_mean.T
                        if self.usingmpi: self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        
                        if (self.rank==0):
                            txt = even_print('write: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                            tqdm.write(txt)
                        
                        t_write       += t_delta
                        data_gb_write += data_gb_mean
                
                if favre:
                    if scalar in scalars_fv:
                        
                        dset = hf_mean['data/%s_fv'%scalar]
                        if self.usingmpi: self.comm.Barrier()
                        t_start = timeit.default_timer()
                        if self.usingmpi:
                            with dset.collective:
                                dset[:,rz1:rz2,ry1:ry2,rx1:rx2] = data_mean_fv.T
                        else:
                            dset[:,:,:,:] = data_mean_fv.T
                        if self.usingmpi: self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        
                        if (self.rank==0):
                            txt = even_print('write: %s_fv'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                            tqdm.write(txt)
                        
                        t_write       += t_delta
                        data_gb_write += data_gb_mean
            
            if self.usingmpi: self.comm.Barrier()
            if verbose: print(72*'-')
            
            # === replace dims/t array --> take last time of series
            t = np.array([self.t[-1]],dtype=np.float64)
            if ('dims/t' in hf_mean):
                del hf_mean['dims/t']
            hf_mean.create_dataset('dims/t', data=t)
            
            if hasattr(hf_mean, 'duration_avg'):
                if verbose: even_print('duration avg', '%0.2f [-]'%hf_mean.duration_avg)
        
        if verbose: print(72*'-')
        if verbose: even_print('time read',format_time_string(t_read))
        if verbose: even_print('time write',format_time_string(t_write))
        if verbose: even_print(fn_cgd_mean, '%0.2f [GB]'%(os.path.getsize(fn_cgd_mean)/1024**3))
        if verbose: even_print('read total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_read,t_read,(data_gb_read/t_read)))
        if verbose: even_print('write total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_write,t_write,(data_gb_write/t_write)))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : cgd.get_mean() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    def get_prime(self, **kwargs):
        '''
        get mean-removed (prime) variables in [t]
        -----
        XI  : Reynolds primes : mean(XI)=0
        XII : Favre primes    : mean(ρ·XII)=0 --> mean(XII)≠0 !!
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'cgd.get_prime()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        #rt = kwargs.get('rt',1)
        ct = kwargs.get('ct',1) ## n chunks [t]
        
        fn_cgd_mean  = kwargs.get('fn_cgd_mean',None)
        fn_cgd_prime = kwargs.get('fn_cgd_prime',None)
        sfp          = kwargs.get('scalars',None) ## scalars (for prime)
        favre        = kwargs.get('favre',True)
        reynolds     = kwargs.get('reynolds',True)
        force        = kwargs.get('force',False)
        
        chunk_kb     = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
        ti_min       = kwargs.get('ti_min',None)
        
        ## if writing Favre primes, copy over ρ --> mean(ρ·XII)=0 / mean(XII)≠0 !!
        if favre:
            copy_rho = True
        else:
            copy_rho = False
        
        if (rx*ry*rz != self.n_ranks):
            raise AssertionError('rx*ry*rz != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        if (ti_min is not None):
            if not isinstance(ti_min, int):
                raise TypeError('ti_min must be type int')
        
        if (sfp is None):
            sfp = self.scalars
        
        # === ranks
        
        if self.usingmpi:
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
        else:
            nxr = self.nx
            nyr = self.ny
            nzr = self.nz
            #ntr = self.nt
        
        # === get times to take for prime
        if (ti_min is not None):
            ti_for_prime = np.copy( self.ti[ti_min:] )
        else:
            ti_for_prime = np.copy( self.ti )
        
        nt_prime       = ti_for_prime.shape[0]
        t_prime_start  = self.t[ti_for_prime[0]]
        t_prime_end    = self.t[ti_for_prime[-1]]
        duration_prime = t_prime_end - t_prime_start
        
        # === chunks
        #ctl_ = np.array_split(np.arange(self.nt),min(ct,self.nt))
        ctl_ = np.array_split(ti_for_prime,min(ct,nt_prime))
        ctl  = [[b[0],b[-1]+1] for b in ctl_ ]
        
        # === mean file name (for reading)
        if (fn_cgd_mean is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_mean_h5_base = fname_root+'_mean.h5'
            #fn_cgd_mean = os.path.join(fname_path, fname_mean_h5_base)
            fn_cgd_mean = str(PurePosixPath(fname_path, fname_mean_h5_base))
            #fn_cgd_mean = Path(fname_path, fname_mean_h5_base)
        
        if not os.path.isfile(fn_cgd_mean):
            raise FileNotFoundError('%s not found!'%fn_cgd_mean)
        
        # === prime file name (for writing)
        if (fn_cgd_prime is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_prime_h5_base = fname_root+'_prime.h5'
            #fn_cgd_prime = os.path.join(fname_path, fname_prime_h5_base)
            fn_cgd_prime = str(PurePosixPath(fname_path, fname_prime_h5_base))
            #fn_cgd_prime = Path(fname_path, fname_prime_h5_base)
        
        if verbose: even_print('fn_cgd'          , self.fname    )
        if verbose: even_print('fn_cgd_mean'     , fn_cgd_mean   )
        if verbose: even_print('fn_cgd_prime'    , fn_cgd_prime  )
        if verbose: even_print('do Favre avg'    , str(favre)    )
        if verbose: even_print('do Reynolds avg' , str(reynolds) )
        if verbose: even_print('copy rho'        , str(copy_rho) )
        if verbose: even_print('ct'              , '%i'%ct       )
        if verbose: print(72*'-')
        if verbose: even_print('nx','%i'%self.nx)
        if verbose: even_print('ny','%i'%self.ny)
        if verbose: even_print('nz','%i'%self.nz)
        if verbose: even_print('nt','%i'%self.nt)
        if verbose: print(72*'-')
        if verbose: even_print('n timesteps prime','%i/%i'%(nt_prime,self.nt))
        if verbose: even_print('t index prime start','%i'%(ti_for_prime[0],))
        if verbose: even_print('t index prime end','%i'%(ti_for_prime[-1],))
        if verbose: even_print('t prime start','%0.2f [-]'%(t_prime_start,))
        if verbose: even_print('t prime end','%0.2f [-]'%(t_prime_end,))
        if verbose: even_print('duration prime','%0.2f [-]'%(duration_prime,))
        if verbose: print(72*'-')
        
        ## performance
        t_read = 0.
        t_write = 0.
        data_gb_read = 0.
        data_gb_write = 0.
        
        #data_gb      = 4*self.nx*self.ny*self.nz*self.nt / 1024**3
        data_gb      = 4*self.nx*self.ny*self.nz*nt_prime / 1024**3
        data_gb_mean = 4*self.nx*self.ny*self.nz*1       / 1024**3
        
        scalars_re = ['u','v','w','T','p','rho']
        scalars_fv = ['u','v','w','T'] ## p'' and ρ'' are never really needed
        
        scalars_re_ = []
        for scalar in scalars_re:
            if (scalar in self.scalars) and (scalar in sfp):
                scalars_re_.append(scalar)
        scalars_re = scalars_re_
        
        scalars_fv_ = []
        for scalar in scalars_fv:
            if (scalar in self.scalars) and (scalar in sfp):
                scalars_fv_.append(scalar)
        scalars_fv = scalars_fv_
        
        # ===
        
        comm_cgd_prime = MPI.COMM_WORLD
        
        with cgd(fn_cgd_prime, 'w', force=force, driver=self.driver, comm=self.comm) as hf_prime:
            
            hf_prime.init_from_cgd(self.fname)
            
            #shape  = (self.nt,self.nz,self.ny,self.nx)
            shape  = (nt_prime,self.nz,self.ny,self.nx)
            chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
            
            # === initialize prime datasets + rho
            
            if copy_rho:
                if verbose:
                    even_print('initializing data/rho','%0.1f [GB]'%(data_gb,))
                dset = hf_prime.create_dataset('data/rho',
                                               shape=shape,
                                               dtype=np.float32,
                                               chunks=chunks)
                
                chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                if verbose:
                    even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                    even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            for scalar in self.scalars:
                
                if reynolds:
                    if (scalar in scalars_re):
                        ## if ('data/%sI'%scalar in hf_prime):
                        ##     del hf_prime['data/%sI'%scalar]
                        if verbose:
                            even_print('initializing data/%sI'%(scalar,),'%0.1f [GB]'%(data_gb,))
                        dset = hf_prime.create_dataset('data/%sI'%scalar,
                                                       shape=shape,
                                                       dtype=np.float32,
                                                       chunks=chunks )
                        hf_prime.scalars.append('%sI'%scalar)
                        
                        chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                        if verbose:
                            even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                            even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
                
                if favre:
                    if (scalar in scalars_fv):
                        ## if ('data/%sII'%scalar in hf_prime):
                        ##     del hf_prime['data/%sII'%scalar]
                        if verbose:
                            even_print('initializing data/%sII'%(scalar,),'%0.1f [GB]'%(data_gb,))
                        dset = hf_prime.create_dataset('data/%sII'%scalar,
                                                       shape=shape,
                                                       dtype=np.float32,
                                                       chunks=chunks )
                        hf_prime.scalars.append('%sII'%scalar)
                        
                        chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                        if verbose:
                            even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                            even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            if hf_prime.usingmpi: comm_cgd_prime.Barrier()
            if self.usingmpi: self.comm.Barrier()
            if verbose: print(72*'-')
            
            # === read unsteady + mean, do difference, write
            
            n_pbar = 0
            if favre or copy_rho:
                n_pbar += 1
            for scalar in self.scalars:
                if (scalar in scalars_re) and reynolds:
                    n_pbar += 1
                if (scalar in scalars_fv) and favre:
                    n_pbar += 1
            
            comm_cgd_mean = MPI.COMM_WORLD
            
            with cgd(fn_cgd_mean, 'r', driver=self.driver, comm=self.comm) as hf_mean:
                
                if verbose:
                    progress_bar = tqdm(total=ct*n_pbar, ncols=100, desc='prime', leave=False, file=sys.stdout)
                
                for ctl_ in ctl:
                    ct1, ct2 = ctl_
                    ntc = ct2 - ct1
                    
                    ## chunk range for writing to file (offset from read if using ti_min)
                    if (ti_min is not None):
                        #ct1w,ct2w = ct1-ti_min, ct2-ti_min ## doesnt work for (-) ti_min
                        ct1w,ct2w = ct1-ti_for_prime[0], ct2-ti_for_prime[0]
                    else:
                        ct1w,ct2w = ct1,ct2
                    
                    # ## debug report
                    # if verbose: tqdm.write('ct1,ct2 = %i,%i'%(ct1,ct2))
                    # if verbose: tqdm.write('ct1w,ct2w = %i,%i'%(ct1w,ct2w))
                    
                    data_gb = 4*self.nx*self.ny*self.nz*ntc / 1024**3 ## data this chunk [GB]
                    
                    if favre or copy_rho:
                        
                        ## read rho
                        dset = self['data/rho']
                        if self.usingmpi: self.comm.Barrier()
                        t_start = timeit.default_timer()
                        if self.usingmpi:
                            with dset.collective:
                                rho = dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2].T
                        else:
                            rho = dset[ct1:ct2,:,:,:].T
                        if self.usingmpi: self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        if verbose:
                            txt = even_print('read: rho', '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                            tqdm.write(txt)
                        t_read       += t_delta
                        data_gb_read += data_gb
                        
                        ## write a copy of rho to the prime file
                        dset = hf_prime['data/rho']
                        if hf_prime.usingmpi: hf_prime.comm.Barrier()
                        t_start = timeit.default_timer()
                        if self.usingmpi:
                            with dset.collective:
                                #dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2] = rho.T
                                dset[ct1w:ct2w,rz1:rz2,ry1:ry2,rx1:rx2] = rho.T
                        else:
                            #dset[ct1:ct2,:,:,:] = rho.T
                            dset[ct1w:ct2w,:,:,:] = rho.T
                        if hf_prime.usingmpi: hf_prime.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        t_write       += t_delta
                        data_gb_write += data_gb
                        if verbose:
                            txt = even_print('write: rho', '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                            tqdm.write(txt)
                        
                        if verbose: progress_bar.update()
                    
                    for scalar in self.scalars:
                        
                        if (scalar in scalars_re) or (scalar in scalars_fv):
                            
                            ## read CGD data
                            dset = self['data/%s'%scalar]
                            if self.usingmpi: self.comm.Barrier()
                            t_start = timeit.default_timer()
                            if self.usingmpi:
                                with dset.collective:
                                    data = dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2].T
                            else:
                                data = dset[ct1:ct2,:,:,:].T
                            if self.usingmpi: self.comm.Barrier()
                            t_delta = timeit.default_timer() - t_start
                            if verbose:
                                txt = even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                tqdm.write(txt)
                            t_read       += t_delta
                            data_gb_read += data_gb
                            
                            # === do prime Reynolds
                            
                            if (scalar in scalars_re) and reynolds:
                                
                                ## read Reynolds avg from mean file
                                dset = hf_mean['data/%s'%scalar]
                                if hf_mean.usingmpi: hf_mean.comm.Barrier()
                                t_start = timeit.default_timer()
                                if hf_mean.usingmpi:
                                    with dset.collective:
                                        data_mean_re = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                                else:
                                    data_mean_re = dset[()].T
                                if hf_mean.usingmpi: hf_mean.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                ## if verbose:
                                ##     txt = even_print('read: %s (Re avg)'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                                ##     tqdm.write(txt)
                                
                                ## calc mean-removed Reynolds
                                data_prime_re = data - data_mean_re
                                
                                ## if False:
                                ##     data_prime_re_mean = np.mean(data_prime_re, axis=-1, dtype=np.float64, keepdims=True).astype(np.float32)
                                ##     
                                ##     ## normalize [mean(prime)] by mean
                                ##     data_prime_re_mean = np.abs(np.divide(data_prime_re_mean,
                                ##                                           data_mean_re, 
                                ##                                           out=np.zeros_like(data_prime_re_mean), 
                                ##                                           where=data_mean_re!=0))
                                ##     
                                ##     # np.testing.assert_allclose( data_prime_re_mean , 
                                ##     #                             np.zeros_like(data_prime_re_mean, dtype=np.float32), atol=1e-4)
                                ##     if verbose:
                                ##         tqdm.write('max(abs(mean(%sI)/mean(%s)))=%0.4e'%(scalar,scalar,data_prime_re_mean.max()))
                                
                                ## write Reynolds prime
                                dset = hf_prime['data/%sI'%scalar]
                                if hf_prime.usingmpi: hf_prime.comm.Barrier()
                                t_start = timeit.default_timer()
                                if hf_prime.usingmpi:
                                    with dset.collective:
                                        #dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2] = data_prime_re.T
                                        dset[ct1w:ct2w,rz1:rz2,ry1:ry2,rx1:rx2] = data_prime_re.T
                                else:
                                    #dset[ct1:ct2,:,:,:] = data_prime_re.T
                                    dset[ct1w:ct2w,:,:,:] = data_prime_re.T
                                if hf_prime.usingmpi: hf_prime.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                t_write       += t_delta
                                data_gb_write += data_gb
                                if verbose:
                                    txt = even_print('write: %sI'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                    tqdm.write(txt)
                                pass
                                
                                if verbose: progress_bar.update()
                            
                            # === do prime Favre
                            
                            if (scalar in scalars_fv) and favre:
                                
                                ## read Favre avg from mean file
                                dset = hf_mean['data/%s_fv'%scalar]
                                if hf_mean.usingmpi: hf_mean.comm.Barrier()
                                t_start = timeit.default_timer()
                                if hf_mean.usingmpi:
                                    with dset.collective:
                                        data_mean_fv = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                                else:
                                    data_mean_fv = dset[()].T
                                if hf_mean.usingmpi: hf_mean.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                ## if verbose:
                                ##     txt = even_print('read: %s (Fv avg)'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                                ##     tqdm.write(txt)
                                
                                ## calc mean-removed Favre
                                ## data_prime_fv = ( data - data_mean_fv ) * rho ## pre-multiply with ρ (has zero mean) --> better to not do this here
                                data_prime_fv = data - data_mean_fv
                                
                                ## write Favre prime
                                dset = hf_prime['data/%sII'%scalar]
                                if hf_prime.usingmpi: hf_prime.comm.Barrier()
                                t_start = timeit.default_timer()
                                if hf_prime.usingmpi:
                                    with dset.collective:
                                        #dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2] = data_prime_fv.T
                                        dset[ct1w:ct2w,rz1:rz2,ry1:ry2,rx1:rx2] = data_prime_fv.T
                                else:
                                    #dset[ct1:ct2,:,:,:] = data_prime_fv.T
                                    dset[ct1w:ct2w,:,:,:] = data_prime_fv.T
                                if hf_prime.usingmpi: hf_prime.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                t_write       += t_delta
                                data_gb_write += data_gb
                                if verbose:
                                    txt = even_print('write: %sII'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                    tqdm.write(txt)
                                pass
                                
                                if verbose: progress_bar.update()
                        
                        if self.usingmpi: self.comm.Barrier()
                        if hf_prime.usingmpi: comm_cgd_prime.Barrier()
                        if hf_mean.usingmpi: comm_cgd_mean.Barrier()
                
                if verbose:
                    progress_bar.close()
            
            # === replace dims/t array in prime file (if ti_min was given)
            if (ti_min is not None):
                t = np.copy( self.t[ti_min:] )
                if ('dims/t' in hf_prime):
                    del hf_prime['dims/t']
                hf_prime.create_dataset('dims/t', data=t)
            
            if hf_mean.usingmpi: comm_cgd_mean.Barrier()
        if hf_prime.usingmpi: comm_cgd_prime.Barrier()
        if self.usingmpi: self.comm.Barrier()
        
        # ===
        
        #if verbose: print(72*'-')
        if verbose: even_print('time read',format_time_string(t_read))
        if verbose: even_print('time write',format_time_string(t_write))
        if verbose: even_print(fn_cgd_prime, '%0.2f [GB]'%(os.path.getsize(fn_cgd_prime)/1024**3))
        if verbose: even_print('read total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_read,t_read,(data_gb_read/t_read)))
        if verbose: even_print('write total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_write,t_write,(data_gb_write/t_write)))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : cgd.get_prime() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    def calc_lambda2(self, **kwargs):
        '''
        calculate λ-2 & Q, save to CGD
        -----
        --> this version is meant for curvilinear (non-rectilinear) meshes
        -----
        Jeong & Hussain (1996) : https://doi.org/10.1017/S0022112095000462
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        save_Q       = kwargs.get('save_Q',True)
        save_lambda2 = kwargs.get('save_lambda2',True)
        rt           = kwargs.get('rt',self.n_ranks)
        chunk_kb     = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
        # ===
        
        if verbose: print('\n'+'turbx.cgd.calc_lambda2()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        ## checks
        if all([(save_Q is False),(save_lambda2 is False)]):
            raise AssertionError('neither λ-2 nor Q set to be solved')
        if (rt>self.nt):
            raise AssertionError('rt>self.nt')
        if (rt != self.n_ranks):
            raise AssertionError('rt != self.n_ranks')
        
        if verbose: even_print('save_Q','%s'%save_Q)
        if verbose: even_print('save_lambda2','%s'%save_lambda2)
        if verbose: even_print('rt','%i'%rt)
        if verbose: print(72*'-')
        
        ## profiling not implemented (non-collective r/w)
        t_read   = 0.
        t_write  = 0.
        t_q_crit = 0.
        t_l2     = 0.
        
        ## get size of infile
        fsize = os.path.getsize(self.fname)/1024**3
        if verbose: even_print(os.path.basename(self.fname),'%0.1f [GB]'%fsize)
        if verbose: even_print('nx','%i'%self.nx)
        if verbose: even_print('ny','%i'%self.ny)
        if verbose: even_print('nz','%i'%self.nz)
        if verbose: even_print('ngp','%0.1f [M]'%(self.ngp/1e6,))
        if verbose: print(72*'-')
        
        ## report memory
        mem_total_gb = psutil.virtual_memory().total/1024**3
        mem_avail_gb = psutil.virtual_memory().available/1024**3
        mem_free_gb  = psutil.virtual_memory().free/1024**3
        if verbose: even_print('mem total', '%0.1f [GB]'%mem_total_gb)
        if verbose: even_print('mem available', '%0.1f [GB]'%mem_avail_gb)
        if verbose: even_print('mem free', '%0.1f [GB]'%mem_free_gb)
        if verbose: print(72*'-')
        
        # ===
        
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
        
        shape  = (self.nt,self.nz,self.ny,self.nx)
        chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
        
        # === initialize 4D arrays in HDF5
        
        if save_lambda2:
            if verbose: even_print('initializing data/lambda2','%0.2f [GB]'%(data_gb,))
            if ('data/lambda2' in self):
                del self['data/lambda2']
            dset = self.create_dataset('data/lambda2', 
                                        shape=shape, 
                                        dtype=self['data/u'].dtype,
                                        chunks=chunks,
                                        )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
        if save_Q:
            if verbose: even_print('initializing data/Q','%0.2f [GB]'%(data_gb,))
            if ('data/Q' in self):
                del self['data/Q']
            dset = self.create_dataset('data/Q', 
                                        shape=shape, 
                                        dtype=self['data/u'].dtype,
                                        chunks=chunks,
                                        )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
        if verbose: print(72*'-')
        
        # === check if strains already exist
        
        if all([('data/dudx' in self),('data/dvdx' in self),('data/dwdx' in self),\
                ('data/dudy' in self),('data/dvdy' in self),('data/dwdy' in self),\
                ('data/dudz' in self),('data/dvdz' in self),('data/dwdz' in self)]):
            strainsAvailable = True
        else:
            strainsAvailable = False
        if verbose: even_print('strains available','%s'%str(strainsAvailable))
        
        # === initialize r/w buffers (optional)
        
        ## take advantage of collective r/w
        useReadBuffer  = False
        useWriteBuffer = False
        
        ## if selected, initialize read buffer (read multiple timesteps into memory at once)
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
        
        ## if selected, initialize write buffer (hold multiple timesteps in memory before writing)
        if useWriteBuffer:
            if save_lambda2:
                l2_buff = np.zeros_like(u)
            if save_Q:
                q_buff = np.zeros_like(u)
        
        # === get metric tensor
        
        t_start = timeit.default_timer()
        ##
        acc = 2
        M = get_metric_tensor_3d(self.x, self.y, self.z, acc=acc, verbose=verbose)
        ##
        t_delta = timeit.default_timer() - t_start
        if verbose: tqdm.write( even_print('get metric tensor','%0.3f [s]'%(t_delta,), s=True) )
        
        # === split out the metric tensor components
        
        ddx_q1 = np.copy( M[:,:,:,0,0] ) ## ξ_x
        ddx_q2 = np.copy( M[:,:,:,1,0] ) ## η_x
        ddx_q3 = np.copy( M[:,:,:,2,0] ) ## ζ_x
        ##
        ddy_q1 = np.copy( M[:,:,:,0,1] ) ## ξ_y
        ddy_q2 = np.copy( M[:,:,:,1,1] ) ## η_y
        ddy_q3 = np.copy( M[:,:,:,2,1] ) ## ζ_y
        ##
        ddz_q1 = np.copy( M[:,:,:,0,2] ) ## ξ_z
        ddz_q2 = np.copy( M[:,:,:,1,2] ) ## η_z
        ddz_q3 = np.copy( M[:,:,:,2,2] ) ## ζ_z
        
        M = None; del M ## free memory
        
        mem_total_gb = psutil.virtual_memory().total/1024**3
        mem_avail_gb = psutil.virtual_memory().available/1024**3
        mem_free_gb  = psutil.virtual_memory().free/1024**3
        if verbose: even_print('mem total', '%0.1f [GB]'%mem_total_gb)
        #if verbose: even_print('mem available', '%0.1f [GB]'%mem_avail_gb)
        if verbose: even_print('mem free', '%0.1f [GB]'%mem_free_gb)
        if verbose: print(72*'-')
        
        ## the 'computational' grid (unit Cartesian)
        #x_comp = np.arange(nx, dtype=np.float64)
        #y_comp = np.arange(ny, dtype=np.float64)
        #z_comp = np.arange(nz, dtype=np.float64)
        x_comp = 1.
        y_comp = 1.
        z_comp = 1.
        
        # ===
        
        if verbose:
            progress_bar = tqdm(total=ntr, ncols=100, desc='calc λ2', leave=False, file=sys.stdout)
        
        tii = -1
        for ti in range(rt1,rt2):
            tii += 1
            
            # === read velocities
            
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
                if verbose:
                    tqdm.write( even_print('read u,v,w', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True ) )
            
            # ============================================================ #
            # get velocity gradient (strain) tensor ∂(u,v,w)/∂(x,y,z)
            # ============================================================ #
            
            t_start = timeit.default_timer()
            
            # === ∂(u)/∂(x,y,z)
            
            ## get computational grid gradient elements
            ddx_u_comp = gradient(u_, x_comp, axis=0, acc=acc, d=1)
            ddy_u_comp = gradient(u_, y_comp, axis=1, acc=acc, d=1)
            ddz_u_comp = gradient(u_, z_comp, axis=2, acc=acc, d=1)
            u_ = None; del u_ ## free memory
            
            ## get physical grid gradient elements by taking
            ##  inner product with metric tensor
            ddx_u = np.copy( ddx_u_comp * ddx_q1 + \
                             ddy_u_comp * ddx_q2 + \
                             ddz_u_comp * ddx_q3 )
            
            ddy_u = np.copy( ddx_u_comp * ddy_q1 + \
                             ddy_u_comp * ddy_q2 + \
                             ddz_u_comp * ddy_q3 )
            
            ddz_u = np.copy( ddx_u_comp * ddz_q1 + \
                             ddy_u_comp * ddz_q2 + \
                             ddz_u_comp * ddz_q3 )
            
            ddx_u_comp = None; del ddx_u_comp ## free memory
            ddy_u_comp = None; del ddy_u_comp ## free memory
            ddz_u_comp = None; del ddz_u_comp ## free memory
            
            # === ∂(v)/∂(x,y,z)
            
            ddx_v_comp = gradient(v_, x_comp, axis=0, acc=acc, d=1)
            ddy_v_comp = gradient(v_, y_comp, axis=1, acc=acc, d=1)
            ddz_v_comp = gradient(v_, z_comp, axis=2, acc=acc, d=1)
            v_ = None; del v_ ## free memory
            
            ddx_v = np.copy( ddx_v_comp * ddx_q1 + \
                             ddy_v_comp * ddx_q2 + \
                             ddz_v_comp * ddx_q3 )
            
            ddy_v = np.copy( ddx_v_comp * ddy_q1 + \
                             ddy_v_comp * ddy_q2 + \
                             ddz_v_comp * ddy_q3 )
            
            ddz_v = np.copy( ddx_v_comp * ddz_q1 + \
                             ddy_v_comp * ddz_q2 + \
                             ddz_v_comp * ddz_q3 )
            
            ddx_v_comp = None; del ddx_v_comp ## free memory
            ddy_v_comp = None; del ddy_v_comp ## free memory
            ddz_v_comp = None; del ddz_v_comp ## free memory
            
            # === ∂(w)/∂(x,y,z)
            
            ddx_w_comp = gradient(w_, x_comp, axis=0, acc=acc, d=1)
            ddy_w_comp = gradient(w_, y_comp, axis=1, acc=acc, d=1)
            ddz_w_comp = gradient(w_, z_comp, axis=2, acc=acc, d=1)
            w_ = None; del w_ ## free memory
            
            ddx_w = np.copy( ddx_w_comp * ddx_q1 + \
                             ddy_w_comp * ddx_q2 + \
                             ddz_w_comp * ddx_q3 )
            
            ddy_w = np.copy( ddx_w_comp * ddy_q1 + \
                             ddy_w_comp * ddy_q2 + \
                             ddz_w_comp * ddy_q3 )
            
            ddz_w = np.copy( ddx_w_comp * ddz_q1 + \
                             ddy_w_comp * ddz_q2 + \
                             ddz_w_comp * ddz_q3 )
            
            ddx_w_comp = None; del ddx_w_comp ## free memory
            ddy_w_comp = None; del ddy_w_comp ## free memory
            ddz_w_comp = None; del ddz_w_comp ## free memory
            
            # ===
            
            strain = np.copy( np.stack((np.stack((ddx_u, ddy_u, ddz_u), axis=3),
                                        np.stack((ddx_v, ddy_v, ddz_v), axis=3),
                                        np.stack((ddx_w, ddy_w, ddz_w), axis=3)), axis=4) )
            
            t_delta = timeit.default_timer() - t_start
            if verbose: tqdm.write( even_print('get strain','%0.3f [s]'%(t_delta,), s=True) )
            
            ddx_u = None; del ddx_u
            ddy_u = None; del ddy_u
            ddz_u = None; del ddz_u
            ddx_v = None; del ddx_v
            ddy_v = None; del ddy_v
            ddz_v = None; del ddz_v
            ddx_w = None; del ddx_w
            ddy_w = None; del ddy_w
            ddz_w = None; del ddz_w
            
            # === get the rate-of-strain & vorticity tensors
            
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
                if verbose: tqdm.write(even_print('calc Q','%s'%format_time_string(t_delta), s=True))
                
                if useWriteBuffer:
                    q_buff[:,:,:,tii]
                else:
                    
                    data_gb = Q.nbytes / 1024**3
                    t_start = timeit.default_timer()
                    self['data/Q'][ti,:,:,:] = Q.T ## non-collective write (memory minimizing)
                    t_delta = timeit.default_timer() - t_start
                    if verbose: tqdm.write( even_print('write Q', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True ) )
                
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
                eigvals_sorted     = np.take_along_axis(eigvals, eigvals_sort_order, axis=3) ## do λ sort
                lambda2            = np.squeeze(eigvals_sorted[:,:,:,1]) ## λ2 is the second eigenvalue (index=1)
                t_delta            = timeit.default_timer() - t_start
                
                if verbose: tqdm.write(even_print('calc λ2','%s'%format_time_string(t_delta), s=True))
                
                if useWriteBuffer:
                    l2_buff[:,:,:,tii]
                else:
                    data_gb = lambda2.nbytes / 1024**3
                    t_start = timeit.default_timer()
                    self['data/lambda2'][ti,:,:,:] = lambda2.T ## independent write
                    t_delta = timeit.default_timer() - t_start
                    if verbose: tqdm.write( even_print('write λ2', '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True ) )
                pass
            
            if verbose: progress_bar.update()
            if verbose and (ti<rt2-1): tqdm.write( '---' )
        if verbose: progress_bar.close()
        if verbose: print(72*'-')
        
        ## do collective writes (if selected)
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
        
        self.get_header(verbose=False, read_grid=False)
        if verbose: even_print(self.fname, '%0.2f [GB]'%(os.path.getsize(self.fname)/1024**3))
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : turbx.cgd.calc_lambda2() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    # === Paraview
    
    def make_xdmf(self, **kwargs):
        '''
        generate an XDMF/XMF2 from CGD for processing with Paraview
        -----
        --> https://www.xdmf.org/index.php/XDMF_Model_and_Format
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        makeVectors = kwargs.get('makeVectors',True) ## write vectors (e.g. velocity, vorticity) to XDMF
        makeTensors = kwargs.get('makeTensors',True) ## write 3x3 tensors (e.g. stress, strain) to XDMF
        
        fname_path            = os.path.dirname(self.fname)
        fname_base            = os.path.basename(self.fname)
        fname_root, fname_ext = os.path.splitext(fname_base)
        fname_xdmf_base       = fname_root+'.xmf2'
        fname_xdmf            = os.path.join(fname_path, fname_xdmf_base)
        
        if verbose: print('\n'+'cgd.make_xdmf()'+'\n'+72*'-')
        
        dataset_precision_dict = {} ## holds dtype.itemsize ints i.e. 4,8
        dataset_numbertype_dict = {} ## holds string description of dtypes i.e. 'Float','Integer'
        
        # === 1D coordinate dimension vectors --> get dtype.name
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
        
        # scalar names dict
        # --> labels for Paraview could be customized (e.g. units could be added) using a dict
        # --> the block below shows one such example dict, though it is currently inactive
        
        if False:
            units = 'dimless'
            if (units=='SI') or (units=='si'): ## m,s,kg,K
                scalar_names = {'x':'x [m]',
                                'y':'y [m]',
                                'z':'z [m]', 
                                'u':'u [m/s]',
                                'v':'v [m/s]',
                                'w':'w [m/s]', 
                                'T':'T [K]',
                                'rho':'rho [kg/m^3]',
                                'p':'p [Pa]'}
            elif (units=='dimless') or (units=='dimensionless'):
                scalar_names = {'x':'x [dimless]',
                                'y':'y [dimless]',
                                'z':'z [dimless]', 
                                'u':'u [dimless]',
                                'v':'v [dimless]',
                                'w':'w [dimless]',
                                'T':'T [dimless]',
                                'rho':'rho [dimless]',
                                'p':'p [dimless]'}
            else:
                raise ValueError('choice of units not recognized : %s --> options are : %s / %s'%(units,'SI','dimless'))
        else:
            scalar_names = {} ## dummy/empty 
        
        ## refresh header
        self.get_header(verbose=False, read_grid=False)
        
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
            
            #with open(fname_xdmf,'w') as xdmf:
            with io.open(fname_xdmf,'w',newline='\n') as xdmf:
                
                xdmf_str='''
                         <?xml version="1.0" encoding="utf-8"?>
                         <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
                         <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
                           <Domain>
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 0*' '))
                
                ## <Topology TopologyType="3DRectMesh" NumberOfElements="{self.nz:d} {self.ny:d} {self.nx:d}"/>
                ## <Geometry GeometryType="VxVyVz">
                
                xdmf_str=f'''
                         <Topology TopologyType="3DSMesh" NumberOfElements="{self.nz:d} {self.ny:d} {self.nx:d}"/>
                         <Geometry GeometryType="X_Y_Z">
                           <DataItem Dimensions="{self.nx:d} {self.ny:d} {self.nz:d}" NumberType="{dataset_numbertype_dict['x']}" Precision="{dataset_precision_dict['x']:d}" Format="HDF">
                             {fname_base}:/dims/{'x'}
                           </DataItem>
                           <DataItem Dimensions="{self.nx:d} {self.ny:d} {self.nz:d}" NumberType="{dataset_numbertype_dict['y']}" Precision="{dataset_precision_dict['y']:d}" Format="HDF">
                             {fname_base}:/dims/{'y'}
                           </DataItem>
                           <DataItem Dimensions="{self.nx:d} {self.ny:d} {self.nz:d}" NumberType="{dataset_numbertype_dict['z']}" Precision="{dataset_precision_dict['z']:d}" Format="HDF">
                             {fname_base}:/dims/{'z'}
                           </DataItem>
                         </Geometry>
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 4*' '))
                
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
                    
                    dset_name = 'ts_%08d'%ti
                    
                    xdmf_str='''
                             <!-- ============================================================ -->
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # =====
                    
                    xdmf_str=f'''
                             <Grid Name="{dset_name}" GridType="Uniform">
                               <Time TimeType="Single" Value="{self.t[ti]:0.8E}"/>
                               <Topology Reference="/Xdmf/Domain/Topology[1]" />
                               <Geometry Reference="/Xdmf/Domain/Geometry[1]" />
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # ===== .xdmf : <Grid> per 3D coordinate array
                    
                    for scalar in ['x','y','z']:
                        
                        dset_hf_path = 'dims/%s'%scalar
                        
                        ## get optional 'label' for Paraview (currently inactive)
                        if scalar in scalar_names:
                            scalar_name = scalar_names[scalar]
                        else:
                            scalar_name = scalar
                        
                        xdmf_str=f'''
                                 <!-- ===== scalar : {scalar} ===== -->
                                 <Attribute Name="{scalar_name}" AttributeType="Scalar" Center="Node">
                                   <DataItem Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict[scalar]}" Precision="{dataset_precision_dict[scalar]:d}" Format="HDF">
                                     {fname_base}:/{dset_hf_path}
                                   </DataItem>
                                 </Attribute>
                                 '''
                        
                        xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    # ===== .xdmf : <Grid> per scalar
                    
                    for scalar in self.scalars:
                        
                        dset_hf_path = 'data/%s'%scalar
                        
                        ## get optional 'label' for Paraview (currently inactive)
                        if scalar in scalar_names:
                            scalar_name = scalar_names[scalar]
                        else:
                            scalar_name = scalar
                        
                        xdmf_str=f'''
                                 <!-- ===== scalar : {scalar} ===== -->
                                 <Attribute Name="{scalar_name}" AttributeType="Scalar" Center="Node">
                                   <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                     <DataItem Dimensions="3 4" NumberType="Integer" Format="XML">
                                       {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                       {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                       {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                     </DataItem>
                                     <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict[scalar]}" Precision="{dataset_precision_dict[scalar]:d}" Format="HDF">
                                       {fname_base}:/{dset_hf_path}
                                     </DataItem>
                                   </DataItem>
                                 </Attribute>
                                 '''
                        
                        xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    if makeVectors:
                        
                        # === .xdmf : <Grid> per vector : velocity vector
                        
                        if ('u' in self.scalars) and ('v' in self.scalars) and ('w' in self.scalars):
                            
                            scalar_name    = 'velocity'
                            dset_hf_path_i = 'data/u'
                            dset_hf_path_j = 'data/v'
                            dset_hf_path_k = 'data/w'
                            
                            xdmf_str = f'''
                            <!-- ===== vector : {scalar_name} ===== -->
                            <Attribute Name="{scalar_name}" AttributeType="Vector" Center="Node">
                              <DataItem Dimensions="{self.nz:d} {self.ny:d} {self.nx:d} {3:d}" Function="JOIN($0, $1, $2)" ItemType="Function">
                                <!-- 1 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['u']}" Precision="{dataset_precision_dict['u']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_i}
                                  </DataItem>
                                </DataItem>
                                <!-- 2 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['v']}" Precision="{dataset_precision_dict['v']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_j}
                                  </DataItem>
                                </DataItem>
                                <!-- 3 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['w']}" Precision="{dataset_precision_dict['w']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_k}
                                  </DataItem>
                                </DataItem>
                                <!-- - -->
                              </DataItem>
                            </Attribute>
                            '''
                            
                            xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                        
                        # === .xdmf : <Grid> per vector : velocity vector
                        
                        if ('uI' in self.scalars) and ('vI' in self.scalars) and ('wI' in self.scalars):
                            
                            scalar_name    = 'velocityI'
                            dset_hf_path_i = 'data/uI'
                            dset_hf_path_j = 'data/vI'
                            dset_hf_path_k = 'data/wI'
                            
                            xdmf_str = f'''
                            <!-- ===== vector : {scalar_name} ===== -->
                            <Attribute Name="{scalar_name}" AttributeType="Vector" Center="Node">
                              <DataItem Dimensions="{self.nz:d} {self.ny:d} {self.nx:d} {3:d}" Function="JOIN($0, $1, $2)" ItemType="Function">
                                <!-- 1 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['uI']}" Precision="{dataset_precision_dict['uI']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_i}
                                  </DataItem>
                                </DataItem>
                                <!-- 2 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['vI']}" Precision="{dataset_precision_dict['vI']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_j}
                                  </DataItem>
                                </DataItem>
                                <!-- 3 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['wI']}" Precision="{dataset_precision_dict['wI']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_k}
                                  </DataItem>
                                </DataItem>
                                <!-- - -->
                              </DataItem>
                            </Attribute>
                            '''
                            
                            xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                        
                        # === .xdmf : <Grid> per vector : vorticity vector
                        
                        if ('vort_x' in self.scalars) and ('vort_y' in self.scalars) and ('vort_z' in self.scalars):
                            
                            scalar_name    = 'vorticity'
                            dset_hf_path_i = 'data/vort_x'
                            dset_hf_path_j = 'data/vort_y'
                            dset_hf_path_k = 'data/vort_z'
                            
                            xdmf_str = f'''
                            <!-- ===== vector : {scalar_name} ===== -->
                            <Attribute Name="{scalar_name}" AttributeType="Vector" Center="Node">
                              <DataItem Dimensions="{self.nz:d} {self.ny:d} {self.nx:d} {3:d}" Function="JOIN($0, $1, $2)" ItemType="Function">
                                <!-- 1 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['vort_x']}" Precision="{dataset_precision_dict['vort_x']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_i}
                                  </DataItem>
                                </DataItem>
                                <!-- 2 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['vort_y']}" Precision="{dataset_precision_dict['vort_y']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_j}
                                  </DataItem>
                                </DataItem>
                                <!-- 3 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['vort_z']}" Precision="{dataset_precision_dict['vort_z']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_k}
                                  </DataItem>
                                </DataItem>
                                <!-- - -->
                              </DataItem>
                            </Attribute>
                            '''
                            
                            xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    if makeTensors:
                        if all([('dudx' in self.scalars),('dvdx' in self.scalars),('dwdx' in self.scalars),
                                ('dudy' in self.scalars),('dvdy' in self.scalars),('dwdy' in self.scalars),
                                ('dudz' in self.scalars),('dvdz' in self.scalars),('dwdz' in self.scalars)]):
                            pass
                            pass ## TODO
                            pass
                    
                    # === .xdmf : end Grid for this timestep
                    
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

class rgd(h5py.File):
    '''
    Rectilinear Grid Data (RGD)
    ---------------------------
    - super()'ed h5py.File class
    - 4D dataset storage
    - dimension coordinates are 4x 1D arrays defining [x,y,z,t] 
    
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
        
        self.fname_path = os.path.dirname(self.fname)
        self.fname_base = os.path.basename(self.fname)
        self.fname_root, self.fname_ext = os.path.splitext(self.fname_base)
        
        ## default to libver='latest' if none provided
        if ('libver' not in kwargs):
            kwargs['libver'] = 'latest'
        
        ## catch possible user error --> could prevent accidental EAS overwrites
        if (self.fname_ext=='.eas'):
            raise ValueError('EAS4 files should not be opened with turbx.rgd()')
        
        ## determine if using mpi
        if ('driver' in kwargs) and (kwargs['driver']=='mpio'):
            self.usingmpi = True
        else:
            self.usingmpi = False
        
        ## determine communicator & rank info
        if self.usingmpi:
            self.comm    = kwargs['comm']
            self.n_ranks = self.comm.Get_size()
            self.rank    = self.comm.Get_rank()
        else:
            self.comm    = None
            self.n_ranks = 1
            self.rank    = 0
        
        ## if not using MPI, remove 'driver' and 'comm' from kwargs
        if ( not self.usingmpi ) and ('driver' in kwargs):
            kwargs.pop('driver')
        if ( not self.usingmpi ) and ('comm' in kwargs):
            kwargs.pop('comm')
        
        ## | mpiexec --mca io romio321 -n $NP python3 ...
        ## | mpiexec --mca io ompio -n $NP python3 ...
        ## | ompi_info --> print ompi settings ('MCA io' gives io implementation options)
        ## | export ROMIO_FSTYPE_FORCE="lustre:" --> force Lustre driver over UFS --> causes crash
        ## | export ROMIO_FSTYPE_FORCE="ufs:"
        ## | export ROMIO_PRINT_HINTS=1 --> show available hints
        
        ## determine MPI info / hints
        if self.usingmpi:
            if ('info' in kwargs):
                self.mpi_info = kwargs['info']
            else:
                mpi_info = MPI.Info.Create()
                ##
                mpi_info.Set('romio_ds_write' , 'disable'   )
                mpi_info.Set('romio_ds_read'  , 'disable'   )
                mpi_info.Set('romio_cb_read'  , 'automatic' )
                mpi_info.Set('romio_cb_write' , 'automatic' )
                #mpi_info.Set('romio_cb_read'  , 'enable' )
                #mpi_info.Set('romio_cb_write' , 'enable' )
                mpi_info.Set('cb_buffer_size' , str(int(round(8*1024**2))) ) ## 8 [MB]
                ##
                kwargs['info'] = mpi_info
                self.mpi_info = mpi_info
        
        ## | rdcc_nbytes:
        ## | ------------
        ## | Integer setting the total size of the raw data chunk cache for this dataset in bytes.
        ## | In most cases increasing this number will improve performance, as long as you have 
        ## | enough free memory. The default size is 1 MB
        
        ## --> gets passed to H5Pset_chunk_cache
        if ('rdcc_nbytes' not in kwargs):
            kwargs['rdcc_nbytes'] = int(16*1024**2) ## 16 [MB]
        
        ## | rdcc_nslots:
        ## | ------------
        ## | Integer defining the number of chunk slots in the raw data chunk cache for this dataset.
        
        ## if ('rdcc_nslots' not in kwargs):
        ##     kwargs['rdcc_nslots'] = 521
        
        ## rgd() unique kwargs (not h5py.File kwargs) --> pop() rather than get()
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
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 8M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        ## touch, stripe
        elif (openMode == 'w') and not os.path.isfile(self.fname):
            if (self.rank==0):
                Path(self.fname).touch()
                if shutil.which('lfs') is not None:
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 8M %s > /dev/null 2>&1'%self.fname, shell=True)
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
            if (self.nx>2):
                if verbose: even_print('dx begin : end', '%0.3E : %0.3E'%( (x[1]-x[0]), (x[-1]-x[-2]) ))
            if verbose: even_print('y_min', '%0.2f'%y.min())
            if verbose: even_print('y_max', '%0.2f'%y.max())
            if (self.ny>2):
                if verbose: even_print('dy begin : end', '%0.3E : %0.3E'%( (y[1]-y[0]), (y[-1]-y[-2]) ))
            if verbose: even_print('z_min', '%0.2f'%z.min())
            if verbose: even_print('z_max', '%0.2f'%z.max())
            if (self.nz>2):
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
            self.scalars = list(self['data'].keys())
            nt,_,_,_ = self['data/%s'%self.scalars[0]].shape
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
        
        EAS4=1
        IEEES=1; IEEED=2
        EAS4_NO_G=1; EAS4_X0DX_G=2; EAS4_UDEF_G=3; EAS4_ALL_G=4; EAS4_FULL_G=5
        gmode_dict = {1:'EAS4_NO_G', 2:'EAS4_X0DX_G', 3:'EAS4_UDEF_G', 4:'EAS4_ALL_G', 5:'EAS4_FULL_G'}
        
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
        
        with eas4(fn_eas4, 'r', verbose=False, driver=self.driver, comm=self.comm) as hf_eas4:
            
            if verbose: even_print( 'gmode dim1' , '%i / %s'%( hf_eas4.gmode_dim1_orig, gmode_dict[hf_eas4.gmode_dim1_orig] ) )
            if verbose: even_print( 'gmode dim2' , '%i / %s'%( hf_eas4.gmode_dim2_orig, gmode_dict[hf_eas4.gmode_dim2_orig] ) )
            if verbose: even_print( 'gmode dim3' , '%i / %s'%( hf_eas4.gmode_dim3_orig, gmode_dict[hf_eas4.gmode_dim3_orig] ) )
            
            # === check gmode (RGD should not have more than ALL_G)
            if (hf_eas4.gmode_dim1_orig > 4):
                raise ValueError('turbx.rgd cannot handle gmode > 4 (EAS4 gmode_dim1=%i)'%hf_eas4.gmode_dim1_orig)
            if (hf_eas4.gmode_dim2_orig > 4):
                raise ValueError('turbx.rgd cannot handle gmode > 4 (EAS4 gmode_dim2=%i)'%hf_eas4.gmode_dim2_orig)
            if (hf_eas4.gmode_dim3_orig > 4):
                raise ValueError('turbx.rgd cannot handle gmode > 4 (EAS4 gmode_dim3=%i)'%hf_eas4.gmode_dim3_orig)
            
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
        
        with rgd(fn_rgd, 'r', driver=self.driver, comm=self.comm) as hf_ref:
            
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
        
        chunk_kb = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
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
        
        # === get all time info & check
        
        comm_eas4 = MPI.COMM_WORLD
        t = np.array([], dtype=np.float64)
        for fn_eas4 in fn_eas4_list:
            with eas4(fn_eas4, 'r', verbose=False, driver=self.driver, comm=comm_eas4) as hf_eas4:
                t = np.concatenate((t, hf_eas4.t))
        comm_eas4.Barrier()
        
        if verbose: even_print('n EAS4 files','%i'%len(fn_eas4_list))
        if verbose: even_print('nt all files','%i'%t.size)
        
        if (t.size>1):
            
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
        
        # === get all grid info & check
        
        if ( len(fn_eas4_list) > 1 ):
            
            comm_eas4 = MPI.COMM_WORLD
            eas4_x_arr = []
            eas4_y_arr = []
            eas4_z_arr = []
            for fn_eas4 in fn_eas4_list:
                with eas4(fn_eas4, 'r', verbose=False, driver=self.driver, comm=comm_eas4) as hf_eas4:
                    eas4_x_arr.append( hf_eas4.x )
                    eas4_y_arr.append( hf_eas4.y )
                    eas4_z_arr.append( hf_eas4.z )
            comm_eas4.Barrier()
            
            ## check coordinate vectors are same
            if not np.all([np.allclose(eas4_z_arr[i],eas4_z_arr[0],rtol=1e-8) for i in range(len(fn_eas4_list))]):
                raise AssertionError('EAS4 files do not have the same z coordinates')
            else:
                if verbose: even_print('check: z coordinate vectors equal','passed')
            if not np.all([np.allclose(eas4_y_arr[i],eas4_y_arr[0],rtol=1e-8) for i in range(len(fn_eas4_list))]):
                raise AssertionError('EAS4 files do not have the same y coordinates')
            else:
                if verbose: even_print('check: y coordinate vectors equal','passed')
            if not np.all([np.allclose(eas4_x_arr[i],eas4_x_arr[0],rtol=1e-8) for i in range(len(fn_eas4_list))]):
                raise AssertionError('EAS4 files do not have the same x coordinates')
            else:
                if verbose: even_print('check: x coordinate vectors equal','passed')
        
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
        
        if self.usingmpi:
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
        else:
            nxr = self.nx
            nyr = self.ny
            nzr = self.nz
            #ntr = self.nt
        
        # === determine RGD scalars (from EAS4 scalars)
        if not hasattr(self, 'scalars') or (len(self.scalars)==0):
            with eas4(fn_eas4_list[0], 'r', verbose=False, driver=self.driver, comm=comm_eas4) as hf_eas4:
                self.scalars   = hf_eas4.scalars
                self.n_scalars = len(self.scalars)
        if self.usingmpi: comm_eas4.Barrier()
        
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
        
        data_gb_read  = 0.
        data_gb_write = 0.
        t_read  = 0.
        t_write = 0.
        
        tii  = -1 ## counter full series
        tiii = -1 ## counter RGD-local
        for fn_eas4 in fn_eas4_list:
            with eas4(fn_eas4, 'r', verbose=False, driver=self.driver, comm=self.comm) as hf_eas4:
                
                if verbose: tqdm.write(even_print(os.path.basename(fn_eas4), '%0.2f [GB]'%(os.path.getsize(fn_eas4)/1024**3), s=True))
                #if verbose: tqdm.write(even_print('gmode_dim1', '%i'%hf_eas4.gmode_dim1, s=True))
                #if verbose: tqdm.write(even_print('gmode_dim2', '%i'%hf_eas4.gmode_dim2, s=True))
                if verbose: tqdm.write(even_print('gmode_dim3', '%i'%hf_eas4.gmode_dim3, s=True))
                if verbose: tqdm.write(even_print('duration', '%0.2f'%hf_eas4.duration, s=True))
                
                # === write buffer
                
                # ## 5D [scalar][x,y,z,t] structured array
                # buff = np.zeros(shape=(nxr, nyr, nzr, bt), dtype={'names':self.scalars, 'formats':self.scalars_dtypes})
                
                # ===
                
                #domainName = 'DOMAIN_000000' ## only one domain supported
                domainName = hf_eas4.domainName
                
                for ti in range(hf_eas4.nt):
                    tii += 1 ## EAS4 series counter
                    if doRead[tii]:
                        tiii += 1 ## RGD counter
                        for scalar in hf_eas4.scalars:
                            if (scalar in self.scalars):
                                
                                # === collective read
                                
                                dset_path = 'Data/%s/ts_%06d/par_%06d'%(domainName,ti,hf_eas4.scalar_n_map[scalar])
                                dset = hf_eas4[dset_path]
                                
                                if hf_eas4.usingmpi: comm_eas4.Barrier()
                                t_start = timeit.default_timer()
                                if hf_eas4.usingmpi: 
                                    with dset.collective:
                                        data = dset[rx1:rx2,ry1:ry2,rz1:rz2]
                                else:
                                    data = dset[()]
                                if hf_eas4.usingmpi: comm_eas4.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                
                                data_gb       = data.nbytes / 1024**3
                                t_read       += t_delta
                                data_gb_read += data_gb
                                
                                if False:
                                    if verbose:
                                        txt = even_print('read: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                        tqdm.write(txt)
                                
                                # === reduce precision
                                
                                if (data.dtype == np.float64):
                                    data = np.copy( data.astype(np.float32) )
                                data_gb = data.nbytes / 1024**3
                                
                                # === collective write
                                
                                dset = self['data/%s'%scalar]
                                
                                if self.usingmpi: self.comm.Barrier()
                                t_start = timeit.default_timer()
                                if self.usingmpi: 
                                    with dset.collective:
                                        dset[tiii,rz1:rz2,ry1:ry2,rx1:rx2] = data.T
                                else:
                                    
                                    if self.hasGridFilter:
                                        data = data[self.xfi[:,np.newaxis,np.newaxis],
                                                    self.yfi[np.newaxis,:,np.newaxis],
                                                    self.zfi[np.newaxis,np.newaxis,:]]
                                    
                                    dset[tiii,:,:,:] = data.T
                                
                                if self.usingmpi: self.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                
                                t_write       += t_delta
                                data_gb_write += data_gb
                                
                                if False:
                                    if verbose:
                                        txt = even_print('write: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                        tqdm.write(txt)
                                
                                if verbose:
                                    progress_bar.update()
        
        if verbose:
            progress_bar.close()
        
        if hf_eas4.usingmpi: comm_eas4.Barrier()
        if self.usingmpi: self.comm.Barrier()
        self.get_header(verbose=False)
        
        if verbose: print(72*'-')
        if verbose: even_print('nt',       '%i'%self.nt )
        if verbose: even_print('dt',       '%0.6f'%self.dt )
        if verbose: even_print('duration', '%0.2f'%self.duration )
        
        if verbose: print(72*'-')
        if verbose: even_print('time read',format_time_string(t_read))
        if verbose: even_print('time write',format_time_string(t_write))
        if verbose: even_print(self.fname, '%0.2f [GB]'%(os.path.getsize(self.fname)/1024**3))
        if verbose: even_print('read total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_read,t_read,(data_gb_read/t_read)))
        if verbose: even_print('write total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_write,t_write,(data_gb_write/t_write)))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.import_eas4() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    @staticmethod
    def copy(fn_rgd_src, fn_rgd_tgt, **kwargs):
        '''
        copy header info, selected scalars, and [x,y,z,t] range to new RGD file
        '''
        
        #comm    = MPI.COMM_WORLD
        rank    = MPI.COMM_WORLD.Get_rank()
        n_ranks = MPI.COMM_WORLD.Get_size()
        
        if (rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.copy()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx       = kwargs.get('rx',1)
        ry       = kwargs.get('ry',1)
        rz       = kwargs.get('rz',1)
        rt       = kwargs.get('rt',1)
        force    = kwargs.get('force',False) ## overwrite or raise error if exists
        chunk_kb = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        ti_min   = kwargs.get('ti_min',None)
        ti_max   = kwargs.get('ti_max',None)
        scalars  = kwargs.get('scalars',None)
        ##
        xi_min = kwargs.get('xi_min',None) ## 4D coordinate 
        xi_max = kwargs.get('xi_max',None)
        yi_min = kwargs.get('yi_min',None)
        yi_max = kwargs.get('yi_max',None)
        zi_min = kwargs.get('zi_min',None)
        zi_max = kwargs.get('zi_max',None)
        ti_min = kwargs.get('ti_min',None)
        ti_max = kwargs.get('ti_max',None)
        ##
        ct = kwargs.get('ct',1) ## 'chunks' in time
        
        if (rt!=1):
            raise AssertionError('rt!=1')
        if (rx*ry*rz!=n_ranks):
            raise AssertionError('rx*ry*rz!=n_ranks')
        if not os.path.isfile(fn_rgd_src):
            raise FileNotFoundError('%s not found!'%fn_rgd_src)
        if os.path.isfile(fn_rgd_tgt) and not force:
            raise FileExistsError('%s already exists. delete it or use \'force=True\' kwarg'%fn_rgd_tgt)
        
        # ===
        
        with rgd(fn_rgd_src, 'r', comm=MPI.COMM_WORLD, driver='mpio', libver='latest') as hf_src:
            with rgd(fn_rgd_tgt, 'w', comm=MPI.COMM_WORLD, driver='mpio', libver='latest', force=force) as hf_tgt:
                
                ## copy over header info
                hf_tgt.init_from_rgd(fn_rgd_src)
                
                if (scalars is None):
                    scalars = hf_src.scalars
                
                if verbose:
                    even_print('fn_rgd_src' , fn_rgd_src )
                    even_print('nx' , '%i'%hf_src.nx )
                    even_print('ny' , '%i'%hf_src.ny )
                    even_print('nz' , '%i'%hf_src.nz )
                    even_print('nt' , '%i'%hf_src.nt )
                    if verbose: print(72*'-')
                
                if (rx>hf_src.nx):
                    raise AssertionError('rx>nx')
                if (ry>hf_src.ny):
                    raise AssertionError('ry>ny')
                if (rz>hf_src.nz):
                    raise AssertionError('rz>nz')
                if (rt>hf_src.nt):
                    raise AssertionError('rt>nt')
                
                x  = np.copy( hf_src.x )
                y  = np.copy( hf_src.y )
                z  = np.copy( hf_src.z )
                t  = np.copy( hf_src.t )
                
                xi  = np.arange(x.shape[0],dtype=np.int64) ## arange index vector, doesnt get touched!
                yi  = np.arange(y.shape[0],dtype=np.int64)
                zi  = np.arange(z.shape[0],dtype=np.int64)
                ti  = np.arange(t.shape[0],dtype=np.int64)
                
                xfi = np.arange(x.shape[0],dtype=np.int64) ## gets clipped depending on x/y/z/t_min/max opts
                yfi = np.arange(y.shape[0],dtype=np.int64)
                zfi = np.arange(z.shape[0],dtype=np.int64)
                tfi = np.arange(t.shape[0],dtype=np.int64)
                
                # === total bounds clip (coordinate index) --> supports negative indexing!
                
                if True: ## code folding
                    
                    if (xi_min is not None):
                        xfi_ = []
                        if verbose:
                            if (xi_min<0):
                                even_print('xi_min', '%i / %i'%(xi_min,xi[xi_min]))
                            else:
                                even_print('xi_min', '%i'%(xi_min,))
                        for c in xfi:
                            if (xi_min<0) and (c>=(hf_src.nx+xi_min)):
                                xfi_.append(c)
                            elif (xi_min>=0) and (c>=xi_min):
                                xfi_.append(c)
                        xfi=np.array(xfi_, dtype=np.int64)
                    else:
                        xi_min = 0
                    
                    if (xi_max is not None):
                        xfi_ = []
                        if verbose:
                            if (xi_max<0):
                                even_print('xi_max', '%i / %i'%(xi_max,xi[xi_max]))
                            else:
                                even_print('xi_max', '%i'%(xi_max,))
                        for c in xfi:
                            if (xi_max<0) and (c<=(hf_src.nx+xi_max)):
                                xfi_.append(c)
                            elif (xi_max>=0) and (c<=xi_max):
                                xfi_.append(c)
                        xfi=np.array(xfi_, dtype=np.int64)
                    else:
                        xi_max = xi[-1]
                    
                    ## check x
                    if ((xi[xi_max]-xi[xi_min]+1)<1):
                        raise ValueError('invalid xi range requested')
                    if (rx>(xi[xi_max]-xi[xi_min]+1)):
                        raise ValueError('more ranks than grid points in x')
                    
                    if (yi_min is not None):
                        yfi_ = []
                        if verbose:
                            if (yi_min<0):
                                even_print('yi_min', '%i / %i'%(yi_min,yi[yi_min]))
                            else:
                                even_print('yi_min', '%i'%(yi_min,))
                        for c in yfi:
                            if (yi_min<0) and (c>=(hf_src.ny+yi_min)):
                                yfi_.append(c)
                            elif (yi_min>=0) and (c>=yi_min):
                                yfi_.append(c)
                        yfi=np.array(yfi_, dtype=np.int64)
                    else:
                        yi_min = 0
                    
                    if (yi_max is not None):
                        yfi_ = []
                        if verbose:
                            if (yi_max<0):
                                even_print('yi_max', '%i / %i'%(yi_max,yi[yi_max]))
                            else:
                                even_print('yi_max', '%i'%(yi_max,))
                        for c in yfi:
                            if (yi_max<0) and (c<=(hf_src.ny+yi_max)):
                                yfi_.append(c)
                            elif (yi_max>=0) and (c<=yi_max):
                                yfi_.append(c)
                        yfi=np.array(yfi_, dtype=np.int64)
                    else:
                        yi_max = yi[-1]
                    
                    ## check y
                    if ((yi[yi_max]-yi[yi_min]+1)<1):
                        raise ValueError('invalid yi range requested')
                    if (ry>(yi[yi_max]-yi[yi_min]+1)):
                        raise ValueError('more ranks than grid points in y')
                    
                    if (zi_min is not None):
                        zfi_ = []
                        if verbose:
                            if (zi_min<0):
                                even_print('zi_min', '%i / %i'%(zi_min,zi[zi_min]))
                            else:
                                even_print('zi_min', '%i'%(zi_min,))
                        for c in zfi:
                            if (zi_min<0) and (c>=(hf_src.nz+zi_min)):
                                zfi_.append(c)
                            elif (zi_min>=0) and (c>=zi_min):
                                zfi_.append(c)
                        zfi=np.array(zfi_, dtype=np.int64)
                    else:
                        zi_min = 0
                    
                    if (zi_max is not None):
                        zfi_ = []
                        if verbose:
                            if (zi_max<0):
                                even_print('zi_max', '%i / %i'%(zi_max,zi[zi_max]))
                            else:
                                even_print('zi_max', '%i'%(zi_max,))
                        for c in zfi:
                            if (zi_max<0) and (c<=(hf_src.nz+zi_max)):
                                zfi_.append(c)
                            elif (zi_max>=0) and (c<=zi_max):
                                zfi_.append(c)
                        zfi=np.array(zfi_, dtype=np.int64)
                    else:
                        zi_max = zi[-1]
                    
                    ## check z
                    if ((zi[zi_max]-zi[zi_min]+1)<1):
                        raise ValueError('invalid zi range requested')
                    if (rz>(zi[zi_max]-zi[zi_min]+1)):
                        raise ValueError('more ranks than grid points in z')
                    
                    if (ti_min is not None):
                        tfi_ = []
                        if verbose:
                            if (ti_min<0):
                                even_print('ti_min', '%i / %i'%(ti_min,ti[ti_min]))
                            else:
                                even_print('ti_min', '%i'%(ti_min,))
                        for c in tfi:
                            if (ti_min<0) and (c>=(hf_src.nt+ti_min)):
                                tfi_.append(c)
                            elif (ti_min>=0) and (c>=ti_min):
                                tfi_.append(c)
                        tfi=np.array(tfi_, dtype=np.int64)
                    else:
                        ti_min = 0
                    
                    if (ti_max is not None):
                        tfi_ = []
                        if verbose:
                            if (ti_max<0):
                                even_print('ti_max', '%i / %i'%(ti_max,ti[ti_max]))
                            else:
                                even_print('ti_max', '%i'%(ti_max,))
                        for c in tfi:
                            if (ti_max<0) and (c<=(hf_src.nt+ti_max)):
                                tfi_.append(c)
                            elif (ti_max>=0) and (c<=ti_max):
                                tfi_.append(c)
                        tfi=np.array(tfi_, dtype=np.int64)
                    else:
                        ti_max = ti[-1]
                    
                    ## check t
                    if ((ti[ti_max]-ti[ti_min]+1)<1):
                        raise ValueError('invalid ti range requested')
                    if (ct>(ti[ti_max]-ti[ti_min]+1)):
                        raise ValueError('more chunks than timesteps')
                
                # ===
                
                x  = np.copy(x[xfi]) ## target file
                y  = np.copy(y[yfi])
                z  = np.copy(z[zfi])
                t  = np.copy(t[tfi])
                
                nx = x.shape[0] ## target file
                ny = y.shape[0]
                nz = z.shape[0]
                nt = t.shape[0]
                
                if verbose:
                    even_print('fn_rgd_tgt' , fn_rgd_tgt )
                    even_print('nx' , '%i'%nx )
                    even_print('ny' , '%i'%ny )
                    even_print('nz' , '%i'%nz )
                    even_print('nt' , '%i'%nt )
                    print(72*'-')
                
                ## replace coordinate dimension arrays in target file
                if ('dims/x' in hf_tgt):
                    del hf_tgt['dims/x']
                    hf_tgt.create_dataset('dims/x', data=x, dtype=np.float64, chunks=None)
                if ('dims/y' in hf_tgt):
                    del hf_tgt['dims/y']
                    hf_tgt.create_dataset('dims/y', data=y, dtype=np.float64, chunks=None)
                if ('dims/z' in hf_tgt):
                    del hf_tgt['dims/z']
                    hf_tgt.create_dataset('dims/z', data=z, dtype=np.float64, chunks=None)
                if ('dims/t' in hf_tgt):
                    del hf_tgt['dims/t']
                    hf_tgt.create_dataset('dims/t', data=t, dtype=np.float64, chunks=None)
                
                # === 3D/4D communicator
                
                comm4d = hf_src.comm.Create_cart(dims=[rx,ry,rz], periods=[False,False,False], reorder=False)
                t4d = comm4d.Get_coords(rank)
                
                rxl_ = np.array_split(xfi,rx)
                ryl_ = np.array_split(yfi,ry)
                rzl_ = np.array_split(zfi,rz)
                #rtl_ = np.array_split(tfi,rt)
                
                rxl = [[b[0],b[-1]+1] for b in rxl_ ]
                ryl = [[b[0],b[-1]+1] for b in ryl_ ]
                rzl = [[b[0],b[-1]+1] for b in rzl_ ]
                #rtl = [[b[0],b[-1]+1] for b in rtl_ ]
                
                rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
                ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
                rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
                #rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
                
                # ===
                
                rx1_ = rx1 - xi[xi_min] ## coords in target file
                ry1_ = ry1 - yi[yi_min]
                rz1_ = rz1 - zi[zi_min]
                #rt1_ = rt1 - ti[ti_min]
                
                rx2_ = rx2 - xi[xi_min] ## coords in target file
                ry2_ = ry2 - yi[yi_min]
                rz2_ = rz2 - zi[zi_min]
                #rt2_ = rt2 - ti[ti_min]
                
                ## time 'chunks' split (number of timesteps to read / write at a time)
                ctl_ = np.array_split(tfi,ct)
                ctl = [[b[0],b[-1]+1] for b in ctl_ ]
                
                shape  = (nt,nz,ny,nx) ## target
                chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
                
                hf_tgt.scalars = []
                data_gb = 4*nx*ny*nz*nt / 1024**3
                
                ## initialize datasets
                t_start = timeit.default_timer()
                for scalar in hf_src.scalars:
                    if (scalar in scalars):
                        if verbose:
                            even_print('initializing data/%s'%(scalar,),'%0.1f [GB]'%(data_gb,))
                        dset = hf_tgt.create_dataset('data/%s'%scalar,
                                                       shape=shape,
                                                       dtype=np.float32,
                                                       chunks=chunks)
                        hf_tgt.scalars.append(scalar)
                        
                        chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                        if verbose:
                            even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                            even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
                
                t_initialize = timeit.default_timer() - t_start
                if verbose:
                    even_print('time initialize',format_time_string(t_initialize))
                    print(72*'-')
                
                # ===
                
                hf_tgt.n_scalars = len(hf_tgt.scalars)
                
                # ===
                
                data_gb_read  = 0.
                data_gb_write = 0.
                t_read  = 0.
                t_write = 0.
                
                if verbose:
                    progress_bar = tqdm(total=len(ctl)*hf_tgt.n_scalars, ncols=100, desc='copy', leave=False, file=sys.stdout)
                
                for scalar in hf_tgt.scalars:
                    dset_src = hf_src['data/%s'%scalar]
                    dset_tgt = hf_tgt['data/%s'%scalar]
                    
                    for ctl_ in ctl:
                        
                        ct1, ct2 = ctl_
                        
                        ct1_ = ct1 - ti[ti_min] ## coords in target file
                        ct2_ = ct2 - ti[ti_min]
                        
                        ## read
                        hf_src.comm.Barrier()
                        t_start = timeit.default_timer()
                        with dset_src.collective:
                            data = dset_src[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2].T
                        hf_src.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        #data_gb = n_ranks * data.nbytes / 1024**3 ## approximate
                        data_gb = 4*nx*ny*nz*(ct2-ct1) / 1024**3
                        
                        t_read       += t_delta
                        data_gb_read += data_gb
                        
                        if verbose:
                            tqdm.write(even_print('read: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True))
                        
                        ## write
                        hf_tgt.comm.Barrier()
                        t_start = timeit.default_timer()
                        with dset_tgt.collective:
                            dset_tgt[ct1_:ct2_,rz1_:rz2_,ry1_:ry2_,rx1_:rx2_] = data.T
                        hf_tgt.flush() ## not strictly needed
                        hf_tgt.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        #data_gb = n_ranks * data.nbytes / 1024**3 ## approximate
                        data_gb = 4*nx*ny*nz*(ct2-ct1) / 1024**3
                        
                        t_write       += t_delta
                        data_gb_write += data_gb
                        
                        if verbose:
                            tqdm.write(even_print('write: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True))
                        
                        if verbose:
                            progress_bar.update()
                
                if verbose:
                    progress_bar.close()
        
        if verbose: print(72*'-')
        if verbose: even_print('time initialize',format_time_string(t_initialize))
        if verbose: even_print('time read',format_time_string(t_read))
        if verbose: even_print('time write',format_time_string(t_write))
        if verbose: even_print(fn_rgd_src, '%0.2f [GB]'%(os.path.getsize(fn_rgd_src)/1024**3))
        if verbose: even_print(fn_rgd_tgt, '%0.2f [GB]'%(os.path.getsize(fn_rgd_tgt)/1024**3))
        if verbose: even_print('read total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_read,t_read,(data_gb_read/t_read)))
        if verbose: even_print('write total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_write,t_write,(data_gb_write/t_write)))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.copy() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    def delete_scalars(self,scalar_list,**kwargs):
        '''
        delete scalars from RGD 
        '''
        pass
        return
    
    def read(self,**kwargs):
        '''
        read all data from file, return structured array
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.read()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        if verbose: even_print(self.fname,'%0.2f [GB]'%(os.path.getsize(self.fname)/1024**3))
        if verbose: print(72*'-')
        
        rx       = kwargs.get('rx',1)
        ry       = kwargs.get('ry',1)
        rz       = kwargs.get('rz',1)
        rt       = kwargs.get('rt',1)
        scalars_to_read = kwargs.get('scalars',None)
        
        if (rx*ry*rz*rt!=self.n_ranks):
            raise AssertionError('rx*ry*rz*rt!=self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        if (rt>self.nt):
            raise AssertionError('rt>self.nt')
        
        if (scalars_to_read is None):
            scalars_to_read = self.scalars
        
        if self.usingmpi:
            comm4d = self.comm.Create_cart(dims=[rx,ry,rz,rt], periods=[False,False,False,False], reorder=False)
            t4d    = comm4d.Get_coords(self.rank)
            ##
            rxl_   = np.array_split(np.array(range(self.nx),dtype=np.int64),min(rx,self.nx))
            ryl_   = np.array_split(np.array(range(self.ny),dtype=np.int64),min(ry,self.ny))
            rzl_   = np.array_split(np.array(range(self.nz),dtype=np.int64),min(rz,self.nz))
            rtl_   = np.array_split(np.array(range(self.nt),dtype=np.int64),min(rt,self.nt))
            rxl    = [[b[0],b[-1]+1] for b in rxl_ ]
            ryl    = [[b[0],b[-1]+1] for b in ryl_ ]
            rzl    = [[b[0],b[-1]+1] for b in rzl_ ]
            rtl    = [[b[0],b[-1]+1] for b in rtl_ ]
            ##
            rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
            ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
            rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
            rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
        else:
            nxr = self.nx
            nyr = self.ny
            nzr = self.nz
            ntr = self.nt
        
        t_read = 0.
        data_gb_read = 0.
        
        data_gb = 4*self.nx*self.ny*self.nz*self.nt / 1024**3
        
        # ===
        
        names   = [ s for s in scalars_to_read if s in self.scalars ]
        formats = [ self.scalars_dtypes_dict[n] for n in names ]
        
        ## 5D [scalar][x,y,z,t] structured array
        data = np.zeros(shape=(nxr,nyr,nzr,ntr), dtype={'names':names, 'formats':formats})
        
        for scalar in names:
            
            # === collective read
            dset = self['data/%s'%scalar]
            if self.usingmpi: self.comm.Barrier()
            t_start = timeit.default_timer()
            if self.usingmpi: 
                with dset.collective:
                    data[scalar] = dset[rt1:rt2,rz1:rz2,ry1:ry2,rx1:rx2].T
            else:
                data[scalar] = dset[()].T
            if self.usingmpi: self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            
            if verbose:
                txt = even_print('read: %s'%scalar, '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                tqdm.write(txt)
            
            t_read       += t_delta
            data_gb_read += data_gb
        
        # ===
        
        if verbose: print(72*'-')
        if verbose: even_print('read total', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_read,t_read,(data_gb_read/t_read)))
        if verbose: print(72*'-')
        
        ## if verbose: print('\n'+72*'-')
        ## if verbose: print('total time : rgd.read() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        ## if verbose: print(72*'-')
        
        return data
    
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
        chunk_kb = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
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
        chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
        
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
        
        if verbose: print(72*'-')
        
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
        populate Perlin noise (f^beta noise)
        '''
        raise NotImplementedError('populate_perlin_noise() not yet implemented')
    
    def populate_white_noise(self, **kwargs):
        '''
        populate white noise dummy data
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.populate_white_noise()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        rt = kwargs.get('rt',1)
        
        chunk_kb = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
        self.nx = nx = kwargs.get('nx',128)
        self.ny = ny = kwargs.get('ny',128)
        self.nz = nz = kwargs.get('nz',128)
        self.nt = nt = kwargs.get('nt',128)
        
        data_gb = 3 * 4*nx*ny*nz*nt / 1024.**3
        if verbose: even_print(self.fname, '%0.2f [GB]'%(data_gb,))
        
        self.x = x = np.linspace(0., 2*np.pi, nx, dtype=np.float32)
        self.y = y = np.linspace(0., 2*np.pi, ny, dtype=np.float32)
        self.z = z = np.linspace(0., 2*np.pi, nz, dtype=np.float32)
        #self.t = t = np.linspace(0., 10.,     nt, dtype=np.float32)
        self.t = t = 0.1 * np.arange(nt, dtype=np.float32)
        
        if (rx*ry*rz*rt != self.n_ranks):
            raise AssertionError('rx*ry*rz*rt != self.n_ranks')
        
        if self.usingmpi:
            comm4d = self.comm.Create_cart(dims=[rx,ry,rz,rt], periods=[False,False,False,False], reorder=False)
            t4d    = comm4d.Get_coords(self.rank)
            ##
            rxl_   = np.array_split(np.arange(self.nx,dtype=np.int64),min(rx,self.nx))
            ryl_   = np.array_split(np.arange(self.ny,dtype=np.int64),min(ry,self.ny))
            rzl_   = np.array_split(np.arange(self.nz,dtype=np.int64),min(rz,self.nz))
            rtl_   = np.array_split(np.arange(self.nt,dtype=np.int64),min(rt,self.nt))
            rxl    = [[b[0],b[-1]+1] for b in rxl_ ]
            ryl    = [[b[0],b[-1]+1] for b in ryl_ ]
            rzl    = [[b[0],b[-1]+1] for b in rzl_ ]
            rtl    = [[b[0],b[-1]+1] for b in rtl_ ]
            ##
            rx1, rx2 = rxl[t4d[0]]; nxr = rx2 - rx1
            ry1, ry2 = ryl[t4d[1]]; nyr = ry2 - ry1
            rz1, rz2 = rzl[t4d[2]]; nzr = rz2 - rz1
            rt1, rt2 = rtl[t4d[3]]; ntr = rt2 - rt1
            ##
            ## ## per-rank dim range
            ## xr = x[rx1:rx2]
            ## yr = y[ry1:ry2]
            ## zr = z[rz1:rz2]
            ## tr = t[rt1:rt2]
        else:
            nxr = nx
            nyr = ny
            nzr = nz
            ntr = nt
        
        ## write dims (independent)
        self.create_dataset('dims/x', data=x)
        self.create_dataset('dims/y', data=y)
        self.create_dataset('dims/z', data=z)
        self.create_dataset('dims/t', data=t)
        
        shape  = (self.nt,self.nz,self.ny,self.nx)
        chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
        
        self.scalars = ['u','v','w']
        self.scalars_dtypes = [np.float32 for s in self.scalars]
        
        ## initialize datasets
        data_gb = 4*nx*ny*nz*nt / 1024.**3
        self.usingmpi: self.comm.Barrier()
        t_start = timeit.default_timer()
        for scalar in self.scalars:
            if ('data/%s'%scalar in self):
                del self['data/%s'%scalar]
            if verbose:
                even_print('initializing data/%s'%(scalar,),'%0.2f [GB]'%(data_gb,))
            dset = self.create_dataset('data/%s'%scalar, 
                                        shape=shape,
                                        dtype=np.float32,
                                        chunks=chunks )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        self.usingmpi: self.comm.Barrier()
        t_initialize = timeit.default_timer() - t_start
        
        ## 5D [scalar][x,y,z,t] structured array --> data buffer
        data = np.zeros(shape=(nxr,nyr,nzr,ntr), dtype={'names':self.scalars, 'formats':self.scalars_dtypes})
        
        ## generate data
        if verbose: print(72*'-')
        
        self.usingmpi: self.comm.Barrier()
        t_start = timeit.default_timer()
        rng = np.random.default_rng(seed=self.rank)
        for scalar in self.scalars:
            data[scalar] = rng.uniform(-1, +1, size=(nxr,nyr,nzr,ntr)).astype(np.float32)
        self.usingmpi: self.comm.Barrier()
        t_delta = timeit.default_timer() - t_start
        if verbose: even_print('gen data','%0.3f [s]'%(t_delta,))
        if verbose: print(72*'-')
        
        ## write data
        data_gb_write = 0.
        t_write = 0.
        for scalar in self.scalars:
            ds = self['data/%s'%scalar]
            self.usingmpi: self.comm.Barrier()
            t_start = timeit.default_timer()
            if self.usingmpi:
                with ds.collective:
                    ds[rt1:rt2,rz1:rz2,ry1:ry2,rx1:rx2] = data[scalar].T
            else:
                ds[:,:,:,:] = data[scalar].T
            self.usingmpi: self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4*nx*ny*nz*nt / 1024**3
            
            t_write       += t_delta
            data_gb_write += data_gb
            
            if verbose:
                even_print('write: %s'%(scalar,), '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        if verbose: print(72*'-')
        if verbose: even_print('time initialize',format_time_string(t_initialize))
        if verbose: even_print('write total', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_write,t_write,(data_gb_write/t_write)))
        if verbose: even_print(self.fname, '%0.2f [GB]'%(os.path.getsize(self.fname)/1024**3))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.populate_white_noise() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
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
        ##
        force        = kwargs.get('force',False)
        
        chunk_kb     = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
        if (rx*ry*rz != self.n_ranks):
            raise AssertionError('rx*ry*rz != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        
        if self.usingmpi:
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
        else:
            nxr = self.nx
            nyr = self.ny
            nzr = self.nz
            #ntr = self.nt
        
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
        
        #with rgd(fn_rgd_mean, 'w', force=force, driver='mpio', comm=MPI.COMM_WORLD) as hf_mean:
        with rgd(fn_rgd_mean, 'w', force=force, driver=self.driver, comm=self.comm) as hf_mean:
            
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
            
            if self.usingmpi: self.comm.Barrier()
            if verbose: print(72*'-')
            
            # === read rho
            if favre:
                dset = self['data/rho']
                if self.usingmpi: self.comm.Barrier()
                t_start = timeit.default_timer()
                if self.usingmpi: 
                    with dset.collective:
                        rho = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                else:
                    rho = dset[()].T
                if self.usingmpi: self.comm.Barrier()
                
                t_delta = timeit.default_timer() - t_start
                if (self.rank==0):
                    txt = even_print('read: rho', '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                    tqdm.write(txt)
                
                t_read       += t_delta
                data_gb_read += data_gb
                
                ## mean ρ in [t] --> leave [x,y,z]
                rho_mean = np.mean(rho, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
            
            # === read, do mean, write
            for scalar in self.scalars:
                
                # === collective read
                dset = self['data/%s'%scalar]
                if self.usingmpi: self.comm.Barrier()
                t_start = timeit.default_timer()
                if self.usingmpi:
                    with dset.collective:
                        data = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                else:
                    data = dset[()].T
                if self.usingmpi: self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                
                if (self.rank==0):
                    txt = even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                    tqdm.write(txt)
                
                t_read       += t_delta
                data_gb_read += data_gb
                
                # === do mean in [t]
                if reynolds:
                    data_mean    = np.mean(data,     axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)
                if favre:
                    data_mean_fv = np.mean(data*rho, axis=-1, keepdims=True, dtype=np.float64).astype(np.float32) / rho_mean
                
                # === write
                if reynolds:
                    if scalar in scalars_re:
                        
                        dset = hf_mean['data/%s'%scalar]
                        if self.usingmpi: self.comm.Barrier()
                        t_start = timeit.default_timer()
                        if self.usingmpi:
                            with dset.collective:
                                dset[:,rz1:rz2,ry1:ry2,rx1:rx2] = data_mean.T
                        else:
                            dset[:,:,:,:] = data_mean.T
                        if self.usingmpi: self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        
                        if (self.rank==0):
                            txt = even_print('write: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                            tqdm.write(txt)
                        
                        t_write       += t_delta
                        data_gb_write += data_gb_mean
                
                if favre:
                    if scalar in scalars_fv:
                        
                        dset = hf_mean['data/%s_fv'%scalar]
                        if self.usingmpi: self.comm.Barrier()
                        t_start = timeit.default_timer()
                        if self.usingmpi:
                            with dset.collective:
                                dset[:,rz1:rz2,ry1:ry2,rx1:rx2] = data_mean_fv.T
                        else:
                            dset[:,:,:,:] = data_mean_fv.T
                        if self.usingmpi: self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        
                        if (self.rank==0):
                            txt = even_print('write: %s_fv'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                            tqdm.write(txt)
                        
                        t_write       += t_delta
                        data_gb_write += data_gb_mean
            
            if self.usingmpi: self.comm.Barrier()
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
        if verbose: even_print('read total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_read,t_read,(data_gb_read/t_read)))
        if verbose: even_print('write total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_write,t_write,(data_gb_write/t_write)))
        if verbose: print(72*'-')
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.get_mean() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        return
    
    def get_prime(self, **kwargs):
        '''
        get mean-removed (prime) variables in [t]
        -----
        XI  : Reynolds primes : mean(XI)=0
        XII : Favre primes    : mean(ρ·XII)=0 --> mean(XII)≠0 !!
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
        sfp          = kwargs.get('scalars',None) ## scalars (for prime)
        favre        = kwargs.get('favre',True)
        reynolds     = kwargs.get('reynolds',True)
        force        = kwargs.get('force',False)
        
        chunk_kb     = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
        ## if writing Favre primes, copy over ρ --> mean(ρ·XII)=0 / mean(XII)≠0 !!
        if favre:
            copy_rho = True
        else:
            copy_rho = False
        
        if (rx*ry*rz != self.n_ranks):
            raise AssertionError('rx*ry*rz != self.n_ranks')
        if (rx>self.nx):
            raise AssertionError('rx>self.nx')
        if (ry>self.ny):
            raise AssertionError('ry>self.ny')
        if (rz>self.nz):
            raise AssertionError('rz>self.nz')
        
        if (sfp is None):
            sfp = self.scalars
        
        # === ranks
        
        if self.usingmpi:
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
        else:
            nxr = self.nx
            nyr = self.ny
            nzr = self.nz
            #ntr = self.nt
        
        # === chunks
        ctl_ = np.array_split(np.arange(self.nt),min(ct,self.nt))
        ctl  = [[b[0],b[-1]+1] for b in ctl_ ]
        
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
        if verbose: even_print('copy rho'        , str(copy_rho) )
        if verbose: even_print('ct'              , '%i'%ct       )
        if verbose: print(72*'-')
        
        t_read = 0.
        t_write = 0.
        data_gb_read = 0.
        data_gb_write = 0.
        
        data_gb      = 4*self.nx*self.ny*self.nz*self.nt / 1024**3
        data_gb_mean = 4*self.nx*self.ny*self.nz*1       / 1024**3
        
        scalars_re = ['u','v','w','T','p','rho']
        scalars_fv = ['u','v','w','T'] ## p'' and ρ'' are never really needed
        
        scalars_re_ = []
        for scalar in scalars_re:
            if (scalar in self.scalars) and (scalar in sfp):
                scalars_re_.append(scalar)
        scalars_re = scalars_re_
        
        scalars_fv_ = []
        for scalar in scalars_fv:
            if (scalar in self.scalars) and (scalar in sfp):
                scalars_fv_.append(scalar)
        scalars_fv = scalars_fv_
        
        # ===
        
        comm_rgd_prime = MPI.COMM_WORLD
        
        with rgd(fn_rgd_prime, 'w', force=force, driver=self.driver, comm=self.comm) as hf_prime:
            
            hf_prime.init_from_rgd(self.fname)
            
            shape  = (self.nt,self.nz,self.ny,self.nx)
            chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
            
            # === initialize prime datasets + rho
            
            if copy_rho:
                if verbose:
                    even_print('initializing data/rho','%0.1f [GB]'%(data_gb,))
                dset = hf_prime.create_dataset('data/rho',
                                               shape=shape,
                                               dtype=np.float32,
                                               chunks=chunks)
                
                chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                if verbose:
                    even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                    even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            for scalar in self.scalars:
                
                if reynolds:
                    if (scalar in scalars_re):
                        ## if ('data/%sI'%scalar in hf_prime):
                        ##     del hf_prime['data/%sI'%scalar]
                        if verbose:
                            even_print('initializing data/%sI'%(scalar,),'%0.1f [GB]'%(data_gb,))
                        dset = hf_prime.create_dataset('data/%sI'%scalar,
                                                       shape=shape,
                                                       dtype=np.float32,
                                                       chunks=chunks )
                        hf_prime.scalars.append('%sI'%scalar)
                        
                        chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                        if verbose:
                            even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                            even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
                
                if favre:
                    if (scalar in scalars_fv):
                        ## if ('data/%sII'%scalar in hf_prime):
                        ##     del hf_prime['data/%sII'%scalar]
                        if verbose:
                            even_print('initializing data/%sII'%(scalar,),'%0.1f [GB]'%(data_gb,))
                        dset = hf_prime.create_dataset('data/%sII'%scalar,
                                                       shape=shape,
                                                       dtype=np.float32,
                                                       chunks=chunks )
                        hf_prime.scalars.append('%sII'%scalar)
                        
                        chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
                        if verbose:
                            even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                            even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
            
            if hf_prime.usingmpi: comm_rgd_prime.Barrier()
            if self.usingmpi: self.comm.Barrier()
            if verbose: print(72*'-')
            
            # === read unsteady + mean, do difference, write
            
            n_pbar = 0
            if favre or copy_rho:
                n_pbar += 1
            for scalar in self.scalars:
                if (scalar in scalars_re) and reynolds:
                    n_pbar += 1
                if (scalar in scalars_fv) and favre:
                    n_pbar += 1
            
            comm_rgd_mean = MPI.COMM_WORLD
            
            with rgd(fn_rgd_mean, 'r', driver=self.driver, comm=self.comm) as hf_mean:
                
                if verbose:
                    progress_bar = tqdm(total=ct*n_pbar, ncols=100, desc='prime', leave=False, file=sys.stdout) ## TODO!!
                
                for ctl_ in ctl:
                    ct1, ct2 = ctl_
                    ntc = ct2 - ct1
                    
                    data_gb = 4*self.nx*self.ny*self.nz*ntc / 1024**3 ## data this chunk [GB]
                    
                    if favre or copy_rho:
                        
                        ## read rho
                        dset = self['data/rho']
                        if self.usingmpi: self.comm.Barrier()
                        t_start = timeit.default_timer()
                        if self.usingmpi:
                            with dset.collective:
                                rho = dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2].T
                        else:
                            rho = dset[ct1:ct2,:,:,:].T
                        if self.usingmpi: self.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        # if verbose:
                        #     txt = even_print('read: rho', '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                        #     tqdm.write(txt)
                        t_read       += t_delta
                        data_gb_read += data_gb
                        
                        ## write a copy of rho to the prime file
                        dset = hf_prime['data/rho']
                        if hf_prime.usingmpi: hf_prime.comm.Barrier()
                        t_start = timeit.default_timer()
                        if self.usingmpi:
                            with dset.collective:
                                dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2] = rho.T
                        else:
                            dset[ct1:ct2,:,:,:] = rho.T
                        if hf_prime.usingmpi: hf_prime.comm.Barrier()
                        t_delta = timeit.default_timer() - t_start
                        t_write       += t_delta
                        data_gb_write += data_gb
                        # if verbose:
                        #     txt = even_print('write: rho', '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                        #     tqdm.write(txt)
                        
                        if verbose: progress_bar.update()
                    
                    for scalar in self.scalars:
                        
                        if (scalar in scalars_re) or (scalar in scalars_fv):
                            
                            ## read RGD data
                            dset = self['data/%s'%scalar]
                            if self.usingmpi: self.comm.Barrier()
                            t_start = timeit.default_timer()
                            if self.usingmpi:
                                with dset.collective:
                                    data = dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2].T
                            else:
                                data = dset[ct1:ct2,:,:,:].T
                            if self.usingmpi: self.comm.Barrier()
                            t_delta = timeit.default_timer() - t_start
                            # if verbose:
                            #     txt = even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                            #     tqdm.write(txt)
                            t_read       += t_delta
                            data_gb_read += data_gb
                            
                            # === do prime Reynolds
                            
                            if (scalar in scalars_re) and reynolds:
                                
                                ## read Reynolds avg from mean file
                                dset = hf_mean['data/%s'%scalar]
                                if hf_mean.usingmpi: hf_mean.comm.Barrier()
                                t_start = timeit.default_timer()
                                if hf_mean.usingmpi:
                                    with dset.collective:
                                        data_mean_re = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                                else:
                                    data_mean_re = dset[()].T
                                if hf_mean.usingmpi: hf_mean.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                ## if verbose:
                                ##     txt = even_print('read: %s (Re avg)'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                                ##     tqdm.write(txt)
                                
                                ## calc mean-removed Reynolds
                                data_prime_re = data - data_mean_re
                                
                                ## if False:
                                ##     data_prime_re_mean = np.mean(data_prime_re, axis=-1, dtype=np.float64, keepdims=True).astype(np.float32)
                                ##     
                                ##     ## normalize [mean(prime)] by mean
                                ##     data_prime_re_mean = np.abs(np.divide(data_prime_re_mean,
                                ##                                           data_mean_re, 
                                ##                                           out=np.zeros_like(data_prime_re_mean), 
                                ##                                           where=data_mean_re!=0))
                                ##     
                                ##     # np.testing.assert_allclose( data_prime_re_mean , 
                                ##     #                             np.zeros_like(data_prime_re_mean, dtype=np.float32), atol=1e-4)
                                ##     if verbose:
                                ##         tqdm.write('max(abs(mean(%sI)/mean(%s)))=%0.4e'%(scalar,scalar,data_prime_re_mean.max()))
                                
                                ## write Reynolds prime
                                dset = hf_prime['data/%sI'%scalar]
                                if hf_prime.usingmpi: hf_prime.comm.Barrier()
                                t_start = timeit.default_timer()
                                if hf_prime.usingmpi:
                                    with dset.collective:
                                        dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2] = data_prime_re.T
                                else:
                                    dset[ct1:ct2,:,:,:] = data_prime_re.T
                                if hf_prime.usingmpi: hf_prime.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                t_write       += t_delta
                                data_gb_write += data_gb
                                ## if verbose:
                                ##     txt = even_print('write: %sI'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                ##     tqdm.write(txt)
                                pass
                                
                                if verbose: progress_bar.update()
                            
                            # === do prime Favre
                            
                            if (scalar in scalars_fv) and favre:
                                
                                ## read Favre avg from mean file
                                dset = hf_mean['data/%s_fv'%scalar]
                                if hf_mean.usingmpi: hf_mean.comm.Barrier()
                                t_start = timeit.default_timer()
                                if hf_mean.usingmpi:
                                    with dset.collective:
                                        data_mean_fv = dset[:,rz1:rz2,ry1:ry2,rx1:rx2].T
                                else:
                                    data_mean_fv = dset[()].T
                                if hf_mean.usingmpi: hf_mean.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                ## if verbose:
                                ##     txt = even_print('read: %s (Fv avg)'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb_mean,t_delta,(data_gb_mean/t_delta)), s=True)
                                ##     tqdm.write(txt)
                                
                                ## calc mean-removed Favre
                                ## data_prime_fv = ( data - data_mean_fv ) * rho ## pre-multiply with ρ (has zero mean) --> better to not do this here
                                data_prime_fv = data - data_mean_fv
                                
                                ## write Favre prime
                                dset = hf_prime['data/%sII'%scalar]
                                if hf_prime.usingmpi: hf_prime.comm.Barrier()
                                t_start = timeit.default_timer()
                                if hf_prime.usingmpi:
                                    with dset.collective:
                                        dset[ct1:ct2,rz1:rz2,ry1:ry2,rx1:rx2] = data_prime_fv.T
                                else:
                                    dset[ct1:ct2,:,:,:] = data_prime_fv.T
                                if hf_prime.usingmpi: hf_prime.comm.Barrier()
                                t_delta = timeit.default_timer() - t_start
                                t_write       += t_delta
                                data_gb_write += data_gb
                                ## if verbose:
                                ##     txt = even_print('write: %sII'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True)
                                ##     tqdm.write(txt)
                                pass
                                
                                if verbose: progress_bar.update()
                        
                        if self.usingmpi: self.comm.Barrier()
                        if hf_prime.usingmpi: comm_rgd_prime.Barrier()
                        if hf_mean.usingmpi: comm_rgd_mean.Barrier()
                
                if verbose:
                    progress_bar.close()
            
            if hf_mean.usingmpi: comm_rgd_mean.Barrier()
        if hf_prime.usingmpi: comm_rgd_prime.Barrier()
        if self.usingmpi: self.comm.Barrier()
        
        # ===
        
        #if verbose: print(72*'-')
        if verbose: even_print('time read',format_time_string(t_read))
        if verbose: even_print('time write',format_time_string(t_write))
        if verbose: even_print(fn_rgd_prime, '%0.2f [GB]'%(os.path.getsize(fn_rgd_prime)/1024**3))
        if verbose: even_print('read total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_read,t_read,(data_gb_read/t_read)))
        if verbose: even_print('write total avg', '%0.2f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb_write,t_write,(data_gb_write/t_write)))
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
        
        xi1ss = kwargs.get('xi1ss',0) ## only relevant for when nx > N
        xi2ss = kwargs.get('xi2ss',-1)
        
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
        
        if self.usingmpi:
            
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
        
        else:
            
            nxr = self.nx
            nyr = self.ny
            nzr = self.nz
            #ntr = self.nt
        
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
            if self.usingmpi: self.comm.Barrier()
            t_start = timeit.default_timer()
            if self.usingmpi: 
                with dset.collective:
                    dataScalar[scalar] = dset[:,rz1:rz2,:,rx1:rx2].T
            else:
                dataScalar[scalar] = dset[()].T
            if self.usingmpi: self.comm.Barrier()
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
        
        if verbose: print(72*'-')
        
        # === get gradients
        
        hiOrder=True
        if hiOrder:
            
            ## dudx   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            ## dvdx   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            ## dTdx   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            ## dpdx   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            ## drhodx = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            
            dudy   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dvdy   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dTdy   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
            dpdy   = np.zeros(shape=(nxr,ny,nzr), dtype=np.float64)
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
            
            ## x-gradients --> only need for pseudovel --> not available (currently) in MPI implementation
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
            
            if True:
                dudy   = np.gradient(u,   y, edge_order=2, axis=1)
                dvdy   = np.gradient(v,   y, edge_order=2, axis=1)
                dpdy   = np.gradient(p,   y, edge_order=2, axis=1)
                dTdy   = np.gradient(T,   y, edge_order=2, axis=1)
                drhody = np.gradient(rho, y, edge_order=2, axis=1)
            
            if False:
                dudx   = np.gradient(u,   x, edge_order=2, axis=0)
                dvdx   = np.gradient(v,   x, edge_order=2, axis=0)
                dpdx   = np.gradient(p,   x, edge_order=2, axis=0)
                dTdx   = np.gradient(T,   x, edge_order=2, axis=0)
                drhodx = np.gradient(rho, x, edge_order=2, axis=0)
        
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
        w_edge     = np.zeros(shape=(nxr,nzr), dtype=np.float64)
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
        w99        = np.zeros(shape=(nxr,nzr), dtype=np.float64)
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
        
        ## This parameter becomes incredibly hard to tune for different Ma, etc
        ## - If large:
        ##    - The ddx(j_edge) is relatively smooth, BUT it's still in a region where the gradient is still (relatively high)
        ##       --> it is, after all, no longer really the 'edge'
        ##    - This results in the 99 values and their dependents (i.e. Re_theta) having digital 'steps'
        ## - If small:
        ##    - The 99 values are relatively smooth BUT...
        ##    - The ddx(j_edge) can be 'shock-like' and very uneven, usually producing single jumps in 99 values
        ## - The solution:
        ##    - First use large epsilon, THEN find point in between that point and the top boundary
        
        #epsilon = 1e-6
        epsilon = 1e-3
        
        ## u criteria
        if True:
            j_edge_3 = np.zeros(shape=(nx,nz), dtype=np.int32)
            if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='umax', leave=False, file=sys.stdout)
            for i in range(nxr):
                for k in range(nzr):
                    umax=u[i,:,k].max()
                    for j in range(nyr):
                        if math.isclose(u[i,j,k], umax, rel_tol=epsilon):
                            #j_edge_3[i,k] = j
                            j_edge_3[i,k] = np.abs(y-(y[j]+(1/8)*(y.max()-y[j]))).argmin() ## split difference to bound (distance)
                            break
                        if (u[i,j,k]>umax):
                            j_edge_3[i,k] = j-1
                            break
                    if verbose: progress_bar.update()
            if verbose: progress_bar.close()
        
        ## umag criteria
        if False:
            j_edge_4 = np.zeros(shape=(nx,nz), dtype=np.int32)
            if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='umagmax', leave=False, file=sys.stdout)
            for i in range(nxr):
                for k in range(nzr):
                    umagmax=umag[i,:,k].max()
                    for j in range(nyr):
                        if math.isclose(umag[i,j,k], umagmax, rel_tol=epsilon):
                            j_edge_4[i,k] = j
                            break
                        if (umag[i,j,k]>umagmax):
                            j_edge_4[i,k] = j-1
                            break
                    if verbose: progress_bar.update()
            if verbose: progress_bar.close()
        
        ## mass flux criteria
        if False:
            j_edge_5 = np.zeros(shape=(nx,nz), dtype=np.int32)
            if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='mfluxmax', leave=False, file=sys.stdout)
            for i in range(nxr):
                for k in range(nzr):
                    mfluxmax = mflux[i,:,k].max()
                    for j in range(nyr):
                        if math.isclose(mflux[i,j,k], mfluxmax, rel_tol=epsilon):
                            j_edge_5[i,k] = j
                            break
                        if (mflux[i,j,k]>mfluxmax):
                            j_edge_5[i,k] = j-1
                            break
                    if verbose: progress_bar.update()
            if verbose: progress_bar.close()
        
        j_edge = j_edge_3
        #j_edge   = np.amin(np.stack((j_edge_3,j_edge_4,j_edge_5)), axis=0) ## minimum of collective minima : shape [nx,nz]
        
        # === populate edge arrays (always grid snapped)
        
        for i in range(nxr):
            for k in range(nzr):
                je              =   j_edge[i,k]
                y_edge[i,k]     =         y[je]
                u_edge[i,k]     =     u[i,je,k]
                v_edge[i,k]     =     v[i,je,k]
                w_edge[i,k]     =     w[i,je,k]
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
        data['w_edge']     = w_edge
        data['T_edge']     = T_edge
        data['p_edge']     = p_edge
        data['rho_edge']   = rho_edge
        data['nu_edge']    = nu_edge
        #data['psvel_edge'] = psvel_edge
        data['M_edge']     = M_edge
        
        # ===
        
        if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='δ99', leave=False, file=sys.stdout)
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
                
                je = j_edge[i,k]+5 ## add points for interpolation
                
                if False:
                    psvel_spl = sp.interpolate.CubicSpline(y[:je],psvel[i,:je,k]-(0.99*psvel_edge[i,k]),bc_type='natural')
                    roots = psvel_spl.roots(discontinuity=False, extrapolate=False)
                
                if True:
                    u_spl = sp.interpolate.CubicSpline(y[:je],u[i,:je,k]-(0.99*u_edge[i,k]),bc_type='natural')
                    roots = u_spl.roots(discontinuity=False, extrapolate=False)
                    #u_spl = sp.interpolate.pchip(y[:je],u[i,:je,k]-(0.99*u_edge[i,k]))
                    #roots = u_spl.roots(extrapolate=False,discontinuity=False)
                
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
                w99[i,k]     = sp.interpolate.interp1d(y[:je],     w[i,:je,k] )(d99_)
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
        data['w99']     = w99
        #data['psvel99'] = psvel99
        data['M99']     = M99
        
        data['d99']     = d99
        data['d99j']    = d99j
        data['d99g']    = d99g
        
        if True:
            sc_l_in  = nu_wall / u_tau
            sc_u_in  = u_tau
            sc_t_in  = nu_wall / u_tau**2
            sc_l_out = d99
            sc_u_out = u99
            sc_t_out = d99/u99
            
            data['sc_l_in']  = sc_l_in
            data['sc_u_in']  = sc_u_in
            data['sc_t_in']  = sc_t_in
            data['sc_l_out'] = sc_l_out
            data['sc_u_out'] = sc_u_out
            data['sc_t_out'] = sc_t_out
        
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
        
        if verbose: progress_bar = tqdm(total=(nxr*nzr), ncols=100, desc='Re,θ,δ*', leave=False, file=sys.stdout)
        for i in range(nxr):
            for k in range(nzr):
                je   = j_edge[i,k]
                yl   = np.copy(     y[:je+1]   )
                ul   = np.copy(   u[i,:je+1,k] )
                rhol = np.copy( rho[i,:je+1,k] )
                
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
        
        u_plus_vd = u_vd / u_tau[:,np.newaxis,:] ## Van Driest scaled velocity (wall units)
        data['u_plus_vd'] = u_plus_vd
        
        # === gather everything
        
        if self.usingmpi:
            
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
            
            data = None; del data
            data = data2
            
            # ===
            
            if False: ## to check
                for key,val in data.items():
                    if (self.rank==0):
                        if isinstance(data[key], np.ndarray):
                            print('%s: %s'%(key, str(data[key].shape)))
                        else:
                            print('%s: %s'%(key, type(data[key])))
        
        # === report dimensional scales
        
        if verbose: print(72*'-')
        
        t_meas = self.duration_avg * (self.lchar / self.U_inf)
        
        if (nx<=20): ## if negligibly thin in [x] --> usually only the case for x-stripes
            
            d99_avg      = np.mean(data['d99'],      axis=(0,1)) ## avg in [x,z] --> leave 0D scalar
            u99_avg      = np.mean(data['u99'],      axis=(0,1))
            nu_wall_avg  = np.mean(data['nu_wall'],  axis=(0,1))
            u_tau_avg    = np.mean(data['u_tau'],    axis=(0,1))
            Re_tau_avg   = np.mean(data['Re_tau'],   axis=(0,1))
            Re_theta_avg = np.mean(data['Re_theta'], axis=(0,1))
            t_eddy = t_meas / (d99_avg/u_tau_avg)
            
            if verbose:
                even_print('Re_τ'                      , '%0.1f'%Re_tau_avg                    )
                even_print('Re_θ'                      , '%0.1f'%Re_theta_avg                  )
                even_print('δ99'                       , '%0.5e [m]'%d99_avg                   )
                even_print('u_τ'                       , '%0.3f [m/s]'%u_tau_avg               )
                even_print('ν_wall'                    , '%0.5e [m²/s]'%nu_wall_avg            )
                even_print('t_meas'                    , '%0.5e [s]'%t_meas                    )
                even_print('t_meas/(δ99/u_τ) = t_eddy' , '%0.2f'%t_eddy                        )
                even_print('t_meas/(δ99/u99)'          , '%0.2f'%(t_meas/(d99_avg/u99_avg))    )
                even_print('t_meas/(20·δ99/u99)'       , '%0.2f'%(t_meas/(20*d99_avg/u99_avg)) )
        
        else:
            
            d99_x       = np.mean(data['d99']      , axis=(1,)) ## avg in [z] --> leave [x]
            u99_x       = np.mean(data['u99']      , axis=(1,))
            nu_wall_x   = np.mean(data['nu_wall']  , axis=(1,))
            u_tau_x     = np.mean(data['u_tau']    , axis=(1,))
            Re_tau_x    = np.mean(data['Re_tau']   , axis=(1,))
            Re_theta_x  = np.mean(data['Re_theta'] , axis=(1,))
            sc_l_in_x   = np.mean(data['sc_l_in']  , axis=(1,))
            sc_u_in_x   = np.mean(data['sc_u_in']  , axis=(1,))
            sc_t_in_x   = np.mean(data['sc_t_in']  , axis=(1,))
            sc_l_out_x  = np.mean(data['sc_l_out'] , axis=(1,))
            sc_u_out_x  = np.mean(data['sc_u_out'] , axis=(1,))
            sc_t_out_x  = np.mean(data['sc_t_out'] , axis=(1,))
            
            d99_x_end   = d99_x[xi2ss]
            u_tau_x_end = u_tau_x[xi2ss]
            u99_x_end   = u99_x[xi2ss]
            x_range     = x[xi2ss]-x[xi1ss]
            z_range     = z[xi2ss]-z[xi1ss]
            t_eddy_end  = t_meas / ( d99_x_end / u_tau_x_end)
            
            sc_l_in_growth_fac   = (sc_l_in_x[xi2ss]  - sc_l_in_x[xi1ss])  / sc_l_in_x[xi2ss]
            sc_u_in_growth_fac   = (sc_u_in_x[xi2ss]  - sc_u_in_x[xi1ss])  / sc_u_in_x[xi2ss]
            sc_t_in_growth_fac   = (sc_t_in_x[xi2ss]  - sc_t_in_x[xi1ss])  / sc_t_in_x[xi2ss]
            sc_l_out_growth_fac  = (sc_l_out_x[xi2ss] - sc_l_out_x[xi1ss]) / sc_l_out_x[xi2ss]
            sc_u_out_growth_fac  = (sc_u_out_x[xi2ss] - sc_u_out_x[xi1ss]) / sc_u_out_x[xi2ss]
            sc_t_out_growth_fac  = (sc_t_out_x[xi2ss] - sc_t_out_x[xi1ss]) / sc_t_out_x[xi2ss]
            
            if verbose:
                even_print( 'xi1ss', '%i'%xi1ss )
                even_print( 'xi2ss', '%i'%xi2ss )
                ##
                even_print( 'x min:max' , '%0.5e : %0.5e'%(x.min(),x.max()) )
                even_print( 'y min:max' , '%0.5e : %0.5e'%(y.min(),y.max()) )
                even_print( 'z min:max' , '%0.5e : %0.5e'%(z.min(),z.max()) )
                ##
                even_print( 'x/lchar min:max' , '%0.6f : %0.6f'%(x.min()/self.lchar , x.max()/self.lchar) )
                even_print( 'y/lchar min:max' , '%0.6f : %0.6f'%(y.min()/self.lchar , y.max()/self.lchar) )
                even_print( 'z/lchar min:max' , '%0.6f : %0.6f'%(z.min()/self.lchar , z.max()/self.lchar) )
                ##
                even_print( 'x_range', '%0.5e [m]'%x_range )
                even_print( 'x_range/lchar', '%0.2f'%(x_range/self.lchar) )
                even_print( 'x_range/δ99 @ end', '%0.2f'%(x_range/d99_x_end) )
                even_print( 'z_range/δ99 @ end', '%0.2f'%(z_range/d99_x_end) )
                ##
                print(72*'-')
                even_print( 'rdiff(x) : ℓ_in  = ν_wall/u_τ'  , '%+0.6f'%sc_l_in_growth_fac  )
                even_print( 'rdiff(x) : u_in  = u_τ'         , '%+0.6f'%sc_u_in_growth_fac  )
                even_print( 'rdiff(x) : t_in  = ν_wall/u_τ²' , '%+0.6f'%sc_t_in_growth_fac  )
                even_print( 'rdiff(x) : ℓ_out = δ99'         , '%+0.6f'%sc_l_out_growth_fac )
                even_print( 'rdiff(x) : u_out = u99'         , '%+0.6f'%sc_u_out_growth_fac )
                even_print( 'rdiff(x) : t_out = d99/u99'     , '%+0.6f'%sc_t_out_growth_fac )
                ##
                print(72*'-')
                even_print('Re_τ @ end'   , '%0.1f'%Re_tau_x[xi2ss] )
                even_print('Re_θ @ end'   , '%0.1f'%Re_theta_x[xi2ss] )
                even_print('δ99 @ end'    , '%0.5e [m]'%d99_x[xi2ss] )
                even_print('u_τ @ end'    , '%0.3f [m/s]'%u_tau_x[xi2ss] )
                even_print('ν_wall @ end' , '%0.5e [m²/s]'%nu_wall_x[xi2ss] )
                print(72*'-')
                ##
                even_print('t_meas'                                , '%0.5e [s]'%t_meas )
                even_print('t_meas/(δ99/u_τ) = t_eddy @ end'       , '%0.2f'%t_eddy_end )
                even_print('t_meas/(δ99/u99) @ end'                , '%0.2f'%(t_meas/(d99_x_end/u99_x_end)) )
                even_print('t_meas/(20·δ99/u99) @ end'             , '%0.2f'%(t_meas/(20*d99_x_end/u99_x_end)) )
                even_print('t_meas/(x_range/U_inf) = n flowpasses' , '%0.2f'%(t_meas/(x_range/self.U_inf)) )
        
        if verbose: print(72*'-')
        
        # ===
        
        if (self.rank==0):
            with open(fn_dat_mean_dim,'wb') as f:
                pickle.dump(data,f,protocol=4)
            size = os.path.getsize(fn_dat_mean_dim)
        
        if self.usingmpi: self.comm.Barrier()
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
        
        chunk_kb     = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
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
        
        shape  = (self.nt,self.nz,self.ny,self.nx)
        chunks = rgd.chunk_sizer(nxi=shape, constraint=(1,None,None,None), size_kb=chunk_kb, base=2)
        
        # === initialize 4D arrays
        if save_lambda2:
            if verbose: even_print('initializing data/lambda2','%0.2f [GB]'%(data_gb,))
            if ('data/lambda2' in self):
                del self['data/lambda2']
            dset = self.create_dataset('data/lambda2', 
                                        shape=shape, 
                                        dtype=self['data/u'].dtype,
                                        chunks=chunks,
                                        )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
        if save_Q:
            if verbose: even_print('initializing data/Q','%0.2f [GB]'%(data_gb,))
            if ('data/Q' in self):
                del self['data/Q']
            dset = self.create_dataset('data/Q', 
                                        shape=shape, 
                                        dtype=self['data/u'].dtype,
                                        chunks=chunks,
                                        )
            
            chunk_kb_ = np.prod(dset.chunks)*4 / 1024. ## actual
            if verbose:
                even_print('chunk shape (t,z,y,x)','%s'%str(dset.chunks))
                even_print('chunk size','%i [KB]'%int(round(chunk_kb_)))
        
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
                    self['data/lambda2'][ti,:,:,:] = lambda2.T ## independent write
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
        
        return
    
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
        
        # === mean (dimensional) file name (for reading) : .dat
        if (fn_dat_mean_dim is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_dat_mean_base = fname_root+'_mean_dim.dat'
            fn_dat_mean_dim = str(PurePosixPath(fname_path, fname_dat_mean_base))
        
        ## # === fft file name (for writing) : .h5
        ## if (fn_h5_fft is None):
        ##     fname_path = os.path.dirname(self.fname)
        ##     fname_base = os.path.basename(self.fname)
        ##     fname_root, fname_ext = os.path.splitext(fname_base)
        ##     fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
        ##     fname_fft_h5_base = fname_root+'_fft.h5'
        ##     fn_h5_fft = str(PurePosixPath(fname_path, fname_fft_h5_base))
        
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
        ## if verbose: even_print('fn_h5_fft'       , fn_h5_fft        )
        if verbose: even_print('fn_dat_fft'      , fn_dat_fft       )
        if verbose: print(72*'-')
        
        if not os.path.isfile(fn_dat_mean_dim):
            raise FileNotFoundError('%s not found!'%fn_dat_mean_dim)
        
        # === read in data (mean dim) --> every rank gets full [x,z]
        with open(fn_dat_mean_dim,'rb') as f:
            data_mean_dim = pickle.load(f)
        fmd = type('foo', (object,), data_mean_dim)
        
        self.comm.Barrier()
        
        ## the data dictionary to be pickled later
        data = {}
        
        ## 2D dimensional quantities --> [x,z]
        u_tau    = fmd.u_tau    # ; data['u_tau']    = u_tau
        nu_wall  = fmd.nu_wall  # ; data['nu_wall']  = nu_wall
        rho_wall = fmd.rho_wall # ; data['rho_wall'] = rho_wall
        d99      = fmd.d99      # ; data['d99']      = d99
        u99      = fmd.u99      # ; data['u99']      = u99
        Re_tau   = fmd.Re_tau   # ; data['Re_tau']   = Re_tau
        Re_theta = fmd.Re_theta # ; data['Re_theta'] = Re_theta
        
        ## mean [x,z] --> leave 0D scalar
        u_tau_avg    = np.mean(fmd.u_tau    , axis=(0,1)) ; data['u_tau_avg']    = u_tau_avg
        nu_wall_avg  = np.mean(fmd.nu_wall  , axis=(0,1)) ; data['nu_wall_avg']  = nu_wall_avg
        rho_wall_avg = np.mean(fmd.rho_wall , axis=(0,1)) ; data['rho_wall_avg'] = rho_wall_avg
        d99_avg      = np.mean(fmd.d99      , axis=(0,1)) ; data['d99_avg']      = d99_avg
        u99_avg      = np.mean(fmd.u99      , axis=(0,1)) ; data['u99_avg']      = u99_avg
        Re_tau_avg   = np.mean(fmd.Re_tau   , axis=(0,1)) ; data['Re_tau_avg']   = Re_tau_avg
        Re_theta_avg = np.mean(fmd.Re_theta , axis=(0,1)) ; data['Re_theta_avg'] = Re_theta_avg
        
        ## mean [x,z] --> leave 1D [y]
        rho_avg = np.mean(fmd.rho,axis=(0,2))
        data['rho_avg'] = rho_avg
        
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
        
        lchar   = self.lchar   ; data['lchar']   = lchar
        U_inf   = self.U_inf   ; data['U_inf']   = U_inf
        rho_inf = self.rho_inf ; data['rho_inf'] = rho_inf
        T_inf   = self.T_inf   ; data['T_inf']   = T_inf
        
        data['Ma'] = self.Ma
        data['Pr'] = self.Pr
        
        nx = self.nx ; data['nx'] = nx
        ny = self.ny ; data['ny'] = ny
        nz = self.nz ; data['nz'] = nz
        nt = self.nt ; data['nt'] = nt
        
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
        
        data['x'] = x
        data['y'] = y
        data['z'] = z
        data['t'] = t
        data['t_meas'] = t_meas
        data['dt'] = dt
        
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
            even_print('Re_τ'   , '%0.1f'         % Re_tau_avg   )
            even_print('Re_θ'   , '%0.1f'         % Re_theta_avg )
            even_print('δ99'    , '%0.5e [m]'     % d99_avg      )
            even_print('U_inf'  , '%0.3f [m/s]'   % U_inf        )
            even_print('u_τ'    , '%0.3f [m/s]'   % u_tau_avg    )
            even_print('ν_wall' , '%0.5e [m²/s]'  % nu_wall_avg  )
            even_print('ρ_wall' , '%0.6f [kg/m³]' % rho_wall_avg  )
            print(72*'-')
        
        t_eddy = t_meas / ( d99_avg / u_tau_avg )
        
        if verbose:
            even_print('t_meas/(δ99/u_τ) = t_eddy' , '%0.2f'%t_eddy)
            even_print('t_meas/(δ99/u99)'          , '%0.2f'%(t_meas/(d99_avg/u99_avg)))
            even_print('t_meas/(20·δ99/u99)'       , '%0.2f'%(t_meas/(20*d99_avg/u99_avg)))
            print(72*'-')
        
        # === establish windowing
        win_len, overlap = get_overlapping_window_size(nt, n_win, overlap_fac_nom)
        overlap_fac = overlap / win_len
        tw, n_win, n_pad = get_overlapping_windows(t, win_len, overlap)
        
        data['win_len']     = win_len
        data['overlap_fac'] = overlap_fac
        data['overlap']     = overlap
        data['n_win']       = n_win
        
        t_meas_per_win = (win_len-1)*dt
        t_eddy_per_win = t_meas_per_win / (d99_avg/u_tau_avg)
        
        data['t_eddy_per_win'] = t_eddy_per_win
        
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
        
        data['freq'] = freq
        data['df']   = df
        data['nf']   = nf
        
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
        
        ## prevent zeros since later we divide by wall u to get kx
        #fmd.u = np.maximum(fmd.u, np.ones_like(fmd.u)*1e-8)
        fmd.u[:,0,:] = fmd.u[:,1,:]*1e-4
        
        ## kx
        kx = (2*np.pi*freq[na,na,na,:]/fmd.u[:,:,:,na])
        kx = np.mean(kx, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        # ## kx·δ99 --> dimless outer
        # kxd = (2*np.pi*freq[na,na,na,:]/fmd.u[:,:,:,na]) * sc_l_out[:,na,:,na]
        # kxd = np.mean(kxd, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        # ## kx+ --> dimless inner
        # kxp = (2*np.pi*freq[na,na,na,:]/fmd.u[:,:,:,na]) * sc_l_in[:,na,:,na]
        # kxp = np.mean(kxp, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        ## λx
        lx = (fmd.u[:,:,:,na]/freq[na,na,na,:])
        lx = np.mean(lx, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        # ## λx/δ99 --> dimless outer
        # lxd = (fmd.u[:,:,:,na]/freq[na,na,na,:]) / sc_l_out[:,na,:,na]
        # lxd = np.mean(lxd, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        # ## λx+ = λx/(ν/u_tau) --> dimless inner
        # lxp = (fmd.u[:,:,:,na]/freq[na,na,na,:]) / sc_l_in[:,na,:,na]
        # lxp = np.mean(lxp, axis=(0,2)) ## avg in [x,z] --> [y,f]
        
        data['kx']  = kx
        # data['kxp'] = kxp
        # data['kxd'] = kxd
        data['lx']  = lx
        # data['lxp'] = lxp
        # data['lxd'] = lxd
        
        # === read in data (prime) --> still dimless (inlet)
        
        scalars = [ 'uI','vI','wI', 'rho', 'uII','vII','wII' ]
        
        ## [var1, var2, density_scaling]
        fft_combis = [
                     [ 'uI'  , 'uI'  , False ],
                     [ 'vI'  , 'vI'  , False ],
                     [ 'wI'  , 'wI'  , False ],
                     [ 'uI'  , 'vI'  , False ],
                     [ 'uII' , 'uII' , True  ],
                     [ 'vII' , 'vII' , True  ],
                     [ 'wII' , 'wII' , True  ],
                     [ 'uII' , 'vII' , True  ],
                     ]
        
        scalars_dtypes = [self.scalars_dtypes_dict[s] for s in scalars]
        
        ## 5D [scalar][x,y,z,t] structured array
        data_prime = np.zeros(shape=(self.nx, nyr, self.nz, self.nt), dtype={'names':scalars, 'formats':scalars_dtypes})
        
        for scalar in scalars:
            dset = self['data/%s'%scalar]
            self.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                data_prime[scalar] = dset[:,:,ry1:ry2,:].T
            self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
            if verbose:
                even_print('read: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        # === redimensionalize prime data
        
        for var in data_prime.dtype.names:
            if var in ['u','v','w', 'uI','vI','wI', 'uII','vII','wII']:
                data_prime[var] *= U_inf
            elif var in ['r_uII','r_vII','r_wII']:
                data_prime[var] *= (U_inf*rho_inf)
            elif var in ['T','TI','TII']:
                data_prime[var] *= T_inf
            elif var in ['r_TII']:
                data_prime[var] *= (T_inf*rho_inf)
            elif var in ['rho','rhoI']:
                data_prime[var] *= rho_inf
            elif var in ['p','pI','pII']:
                data_prime[var] *= (rho_inf * U_inf**2)
            else:
                raise ValueError('condition needed for redimensionalizing \'%s\''%var)
        
        ## initialize buffers
        Euu_scalars         = [ '%s%s'%(cc[0],cc[1]) for cc in fft_combis ]
        Euu_scalars_dtypes  = [ np.float32 for s in Euu_scalars ]
        Euu                 = np.zeros(shape=(self.nx, nyr, self.nz, nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        uIuI_avg            = np.zeros(shape=(self.nx, nyr, self.nz)     , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        energy_norm_fac_arr = np.zeros(shape=(self.nx, nyr, self.nz)     , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes}) ## just for monitoring
        
        ## check memory
        mem_total_gb = psutil.virtual_memory().total/1024**3
        mem_avail_gb = psutil.virtual_memory().available/1024**3
        mem_free_gb  = psutil.virtual_memory().free/1024**3
        if verbose:
            even_print('mem total',     '%0.1f [GB]'%(mem_total_gb,))
            even_print('mem available', '%0.1f [GB] / %0.1f[%%]'%(mem_avail_gb,(100*mem_avail_gb/mem_total_gb)))
            even_print('mem free',      '%0.1f [GB] / %0.1f[%%]'%(mem_free_gb,(100*mem_free_gb/mem_total_gb)))
        
        ## the window function
        window_type = 'tukey'
        if (window_type=='tukey'):
            window = sp.signal.windows.tukey(win_len, alpha=overlap_fac_nom)
        elif (window_type is None):
            window = np.ones(win_len, dtype=np.float64)
        if verbose:
            even_print('window type', '\'%s\''%str(window_type))
        sum_sqrt_win = np.sum(np.sqrt(window))
        
        ## main loop
        self.comm.Barrier()
        if verbose: progress_bar = tqdm(total=self.nx*nyr*self.nz, ncols=100, desc='fft', leave=False)
        for xi in range(self.nx):
            for yi in range(nyr):
                for zi in range(self.nz):
                    for cci in range(len(fft_combis)):
                        
                        tag = Euu_scalars[cci]
                        cc = fft_combis[cci]
                        ccL,ccR,density_scaling = cc[0],cc[1],cc[2]
                        
                        uL      = np.copy( data_prime[ccL][xi,yi,zi,:]   )
                        uR      = np.copy( data_prime[ccR][xi,yi,zi,:]   )
                        rho     = np.copy( data_prime['rho'][xi,yi,zi,:] )
                        rho_avg = np.mean(rho, dtype=np.float64).astype(np.float32)
                        
                        if density_scaling:
                            uIuI_avg_ijk = np.mean(uL*uR*rho, dtype=np.float64).astype(np.float32) / rho_avg
                        else:
                            uIuI_avg_ijk = np.mean(uL*uR,     dtype=np.float64).astype(np.float32)
                        
                        uIuI_avg[tag][xi,yi,zi] = uIuI_avg_ijk
                        
                        ## window time series into several overlapping windows
                        uL,  nw, n_pad = get_overlapping_windows(uL,  win_len, overlap)
                        uR,  nw, n_pad = get_overlapping_windows(uR,  win_len, overlap)
                        rho, nw, n_pad = get_overlapping_windows(rho, win_len, overlap)
                        
                        ## do fft for each segment
                        Euu_ijk = np.zeros((nw,nf), dtype=np.float32)
                        for wi in range(nw):
                            if density_scaling:
                                ui    = np.copy( uL[wi,:] * rho[wi,:] )
                                uj    = np.copy( uR[wi,:] * rho[wi,:] )
                            else:
                                ui    = np.copy( uL[wi,:] )
                                uj    = np.copy( uR[wi,:] )
                            n     = ui.size
                            #A_ui = sp.fft.fft(ui)[fp] / n
                            #A_uj = sp.fft.fft(uj)[fp] / n
                            ui   *= window ## window
                            uj   *= window
                            #ui  -= np.mean(ui) ## de-trend
                            #uj  -= np.mean(uj)
                            A_ui          = sp.fft.fft(ui)[fp] / sum_sqrt_win
                            A_uj          = sp.fft.fft(uj)[fp] / sum_sqrt_win
                            Euu_ijk[wi,:] = 2 * np.real(A_ui*np.conj(A_uj)) / df
                        
                        ## mean across fft segments
                        Euu_ijk = np.mean(Euu_ijk, axis=0, dtype=np.float64).astype(np.float32)
                        
                        ## divide off mean density
                        if density_scaling:
                            Euu_ijk /= rho_avg**2
                        
                        ## normalize such that sum(PSD)=covariance
                        if (uIuI_avg_ijk!=0.):
                            energy_norm_fac = np.sum(df*Euu_ijk) / uIuI_avg_ijk
                        else:
                            energy_norm_fac = 1.
                        Euu_ijk /= energy_norm_fac
                        energy_norm_fac_arr[tag][xi,yi,zi] = energy_norm_fac
                        
                        ## write
                        Euu[tag][xi,yi,zi,:] = Euu_ijk
                    
                    if verbose: progress_bar.update()
        if verbose:
            progress_bar.close()
        
        ## report energy normalization factors --> tmp,only rank 0 currently!
        if verbose: print(72*'-')
        for tag in Euu_scalars:
            energy_norm_fac_min = energy_norm_fac_arr[tag].min()
            energy_norm_fac_max = energy_norm_fac_arr[tag].max()
            energy_norm_fac_avg = np.mean(energy_norm_fac_arr[tag], axis=(0,1,2), dtype=np.float64).astype(np.float32)
            if verbose:
                even_print('energy norm min/max/avg : %s'%tag, '%0.4f / %0.4f / %0.4f'%(energy_norm_fac_min,energy_norm_fac_max,energy_norm_fac_avg))
        energy_norm_fac_arr = None ; del energy_norm_fac_arr
        
        # === non-dimensionalize
        
        ## Euu_in       = np.zeros_like( Euu )
        ## kxEuu_in     = np.zeros_like( Euu )
        ## uIuI_avg_in  = np.zeros_like( uIuI_avg )
        ## 
        ## Euu_out      = np.zeros_like( Euu )
        ## kxEuu_out    = np.zeros_like( Euu )
        ## uIuI_avg_out = np.zeros_like( uIuI_avg )
        ## 
        ## for tag in Euu_scalars:
        ##     
        ##     Euu_in[tag]       = np.copy( Euu[tag] / ( sc_t_in[:,na,:,na] * sc_u_in[:,na,:,na]**2 ) )
        ##     kxEuu_in[tag]     = np.copy( Euu_in[tag] * (2*np.pi*freq[na,na,na,:]/fmd.u[:,ry1:ry2,:,na]) * sc_l_in[:,na,:,na] )
        ##     uIuI_avg_in[tag]  = np.copy( uIuI_avg[tag] / sc_u_in[:,na,:]**2 )
        ##     
        ##     Euu_out[tag]      = np.copy( Euu[tag] / ( sc_t_out[:,na,:,na] * sc_u_out[:,na,:,na]**2 ) )
        ##     kxEuu_out[tag]    = np.copy( Euu_out[tag] * (2*np.pi*freq[na,na,na,:]/fmd.u[:,ry1:ry2,:,na]) * sc_l_out[:,na,:,na] )
        ##     uIuI_avg_out[tag] = np.copy( uIuI_avg[tag] / sc_u_out[:,na,:]**2 )
        
        # === average in [x,z] --> leave [y,f]
        
        Euu_          = np.zeros(shape=(nyr,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        uIuI_avg_     = np.zeros(shape=(nyr,)   , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        
        ## Euu_in_       = np.zeros(shape=(nyr,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        ## kxEuu_in_     = np.zeros(shape=(nyr,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        ## uIuI_avg_in_  = np.zeros(shape=(nyr,)   , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        ## 
        ## Euu_out_      = np.zeros(shape=(nyr,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        ## kxEuu_out_    = np.zeros(shape=(nyr,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        ## uIuI_avg_out_ = np.zeros(shape=(nyr,)   , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        
        self.comm.Barrier()
        
        for tag in Euu_scalars:
            
            Euu_[tag]          = np.mean( Euu[tag]          , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y,f]
            uIuI_avg_[tag]     = np.mean( uIuI_avg[tag]     , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y]
            ## ##
            ## Euu_in_[tag]       = np.mean( Euu_in[tag]       , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y,f]
            ## kxEuu_in_[tag]     = np.mean( kxEuu_in[tag]     , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y,f]
            ## uIuI_avg_in_[tag]  = np.mean( uIuI_avg_in[tag]  , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y]
            ## ##
            ## Euu_out_[tag]      = np.mean( Euu_out[tag]      , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y,f]
            ## kxEuu_out_[tag]    = np.mean( kxEuu_out[tag]    , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y,f]
            ## uIuI_avg_out_[tag] = np.mean( uIuI_avg_out[tag] , axis=(0,2) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave [y]
        
        Euu          = np.copy( Euu_ )
        uIuI_avg     = np.copy( uIuI_avg_ )
        ## ##
        ## Euu_in       = np.copy( Euu_in_ )
        ## kxEuu_in     = np.copy( kxEuu_in_ )
        ## uIuI_avg_in  = np.copy( uIuI_avg_in_ )
        ## ##
        ## Euu_out      = np.copy( Euu_out_ )
        ## kxEuu_out    = np.copy( kxEuu_out_ )
        ## uIuI_avg_out = np.copy( uIuI_avg_out_ )
        
        self.comm.Barrier()
        
        # === gather all results --> arrays are very small at this point, just do 'lazy' gather/bcast, dont worry about buffers etc
        
        ## gather
        G = self.comm.gather([self.rank, 
                              Euu, uIuI_avg, 
                              #Euu_in, kxEuu_in, uIuI_avg_in, 
                              #Euu_out, kxEuu_out, uIuI_avg_out
                              ], root=0)
        G = self.comm.bcast(G, root=0)
        
        Euu          = np.zeros( (ny,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        uIuI_avg     = np.zeros( (ny,)   , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        
        # Euu_in       = np.zeros( (ny,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        # kxEuu_in     = np.zeros( (ny,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        # uIuI_avg_in  = np.zeros( (ny,)   , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        # 
        # Euu_out      = np.zeros( (ny,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        # kxEuu_out    = np.zeros( (ny,nf) , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        # uIuI_avg_out = np.zeros( (ny,)   , dtype={'names':Euu_scalars, 'formats':Euu_scalars_dtypes})
        
        for ri in range(self.n_ranks):
            j = ri
            for GG in G:
                if (GG[0]==ri):
                    for tag in Euu_scalars:
                        Euu[tag][ryl[j][0]:ryl[j][1],:]        = GG[1][tag]
                        uIuI_avg[tag][ryl[j][0]:ryl[j][1]]     = GG[2][tag]
                        ## ##
                        ## Euu_in[tag][ryl[j][0]:ryl[j][1],:]     = GG[3][tag]
                        ## kxEuu_in[tag][ryl[j][0]:ryl[j][1],:]   = GG[4][tag]
                        ## uIuI_avg_in[tag][ryl[j][0]:ryl[j][1]]  = GG[5][tag]
                        ## ##
                        ## Euu_out[tag][ryl[j][0]:ryl[j][1],:]    = GG[6][tag]
                        ## kxEuu_out[tag][ryl[j][0]:ryl[j][1],:]  = GG[7][tag]
                        ## uIuI_avg_out[tag][ryl[j][0]:ryl[j][1]] = GG[8][tag]
                else:
                    pass
        
        if verbose: print(72*'-')
        
        # === save results
        if (self.rank==0):
            
            data['Euu']      = Euu
            data['uIuI_avg'] = uIuI_avg
            ##
            ## data['Euu_in']      = Euu_in
            ## data['kxEuu_in']    = kxEuu_in
            ## data['uIuI_avg_in'] = uIuI_avg_in
            ## 
            ## data['Euu_out']      = Euu_out
            ## data['kxEuu_out']    = kxEuu_out
            ## data['uIuI_avg_out'] = uIuI_avg_out
            
            sc_l_in  = np.mean(sc_l_in  , axis=(0,1) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave 0D scalar
            sc_u_in  = np.mean(sc_u_in  , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_t_in  = np.mean(sc_t_in  , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_l_out = np.mean(sc_l_out , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_u_out = np.mean(sc_u_out , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_t_out = np.mean(sc_t_out , axis=(0,1) , dtype=np.float64).astype(np.float32)
            
            data['sc_l_in']  = sc_l_in
            data['sc_l_out'] = sc_l_out
            data['sc_t_in']  = sc_t_in
            data['sc_t_out'] = sc_t_out
            data['sc_u_in']  = sc_u_in
            data['sc_u_out'] = sc_u_out
            
            with open(fn_dat_fft,'wb') as f:
                pickle.dump(data, f, protocol=4)
            print('--w-> %s : %0.2f [MB]'%(fn_dat_fft,os.path.getsize(fn_dat_fft)/1024**2))
        
        # ===
        
        self.comm.Barrier()
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.calc_fft() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    def calc_ccor_span(self, **kwargs):
        '''
        calculate autocorrelation in [z] and avg in [x,t]
        '''
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.calc_ccor_span()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        rt = kwargs.get('rt',1)
        
        fn_dat_ccor_span = kwargs.get('fn_dat_ccor_span',None)
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
        
        # === cross-correlation file name (for writing) : dat
        if (fn_dat_ccor_span is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_ccor_span_dat_base = fname_root+'_ccor_span.dat'
            fn_dat_ccor_span = str(PurePosixPath(fname_path, fname_ccor_span_dat_base))
        
        if verbose: even_print('fn_rgd_prime'     , self.fname       )
        if verbose: even_print('fn_dat_mean_dim'  , fn_dat_mean_dim  )
        if verbose: even_print('fn_dat_ccor_span' , fn_dat_ccor_span )
        if verbose: print(72*'-')
        
        if not os.path.isfile(fn_dat_mean_dim):
            raise FileNotFoundError('%s not found!'%fn_dat_mean_dim)
        
        # === read in data (mean dim) --> every rank gets full [x,z]
        with open(fn_dat_mean_dim,'rb') as f:
            data_mean_dim = pickle.load(f)
        fmd = type('foo', (object,), data_mean_dim)
        
        self.comm.Barrier()
        
        ## the data dictionary to be pickled later
        data = {}
        
        ## 2D dimensional quantities --> [x,z]
        u_tau    = fmd.u_tau    # ; data['u_tau']    = u_tau
        nu_wall  = fmd.nu_wall  # ; data['nu_wall']  = nu_wall
        rho_wall = fmd.rho_wall # ; data['rho_wall'] = rho_wall
        d99      = fmd.d99      # ; data['d99']      = d99
        u99      = fmd.u99      # ; data['u99']      = u99
        Re_tau   = fmd.Re_tau   # ; data['Re_tau']   = Re_tau
        Re_theta = fmd.Re_theta # ; data['Re_theta'] = Re_theta
        
        ## mean [x,z] --> leave 0D scalar
        u_tau_avg    = np.mean(fmd.u_tau    , axis=(0,1)) ; data['u_tau_avg']    = u_tau_avg
        nu_wall_avg  = np.mean(fmd.nu_wall  , axis=(0,1)) ; data['nu_wall_avg']  = nu_wall_avg
        rho_wall_avg = np.mean(fmd.rho_wall , axis=(0,1)) ; data['rho_wall_avg'] = rho_wall_avg
        d99_avg      = np.mean(fmd.d99      , axis=(0,1)) ; data['d99_avg']      = d99_avg
        u99_avg      = np.mean(fmd.u99      , axis=(0,1)) ; data['u99_avg']      = u99_avg
        Re_tau_avg   = np.mean(fmd.Re_tau   , axis=(0,1)) ; data['Re_tau_avg']   = Re_tau_avg
        Re_theta_avg = np.mean(fmd.Re_theta , axis=(0,1)) ; data['Re_theta_avg'] = Re_theta_avg
        
        ## mean [x,z] --> leave 1D [y]
        rho_avg = np.mean(fmd.rho,axis=(0,2))
        data['rho_avg'] = rho_avg
        
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
        
        lchar   = self.lchar   ; data['lchar']   = lchar
        U_inf   = self.U_inf   ; data['U_inf']   = U_inf
        rho_inf = self.rho_inf ; data['rho_inf'] = rho_inf
        T_inf   = self.T_inf   ; data['T_inf']   = T_inf
        
        data['Ma'] = self.Ma
        data['Pr'] = self.Pr
        
        nx = self.nx ; data['nx'] = nx
        ny = self.ny ; data['ny'] = ny
        nz = self.nz ; data['nz'] = nz
        nt = self.nt ; data['nt'] = nt
        
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
        
        data['x'] = x
        data['y'] = y
        data['z'] = z
        data['t'] = t
        data['t_meas'] = t_meas
        data['dt'] = dt
        
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
            even_print('ρ_wall' , '%0.6f [kg/m³]' % rho_wall_avg  )
            print(72*'-')
        
        t_eddy = t_meas / (d99_avg/u_tau_avg)
        
        if verbose:
            even_print('t_meas/(δ99/u_τ) = t_eddy' , '%0.2f'%t_eddy)
            even_print('t_meas/(δ99/u99)'          , '%0.2f'%(t_meas/(d99_avg/u99_avg)))
            even_print('t_meas/(20·δ99/u99)'       , '%0.2f'%(t_meas/(20*d99_avg/u99_avg)))
            print(72*'-')
        
        # ===
        
        scalars = [ 'uI'  , 'vI'  , 'wI'  , 'TI'  , 'pI', 
                    'uII' , 'vII' , 'wII' , 'TII', 'rho'   ]
        scalars_dtypes = [self.scalars_dtypes_dict[s] for s in scalars]
        
        ## 5D [scalar][x,y,z,t] structured array
        data_prime = np.zeros(shape=(nx, nyr, nz, nt), dtype={'names':scalars, 'formats':scalars_dtypes})
        
        for scalar in scalars:
            dset = self['data/%s'%scalar]
            self.comm.Barrier()
            t_start = timeit.default_timer()
            with dset.collective:
                data_prime[scalar] = dset[:,:,ry1:ry2,:].T
            self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
            if (self.rank==0):
                even_print('read: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
        
        # === redimensionalize prime data
        
        for var in data_prime.dtype.names:
            if var in ['u','v','w', 'uI','vI','wI', 'uII','vII','wII']:
                data_prime[var] *= U_inf
            elif var in ['r_uII','r_vII','r_wII']:
                data_prime[var] *= (U_inf*rho_inf)
            elif var in ['T','TI','TII']:
                data_prime[var] *= T_inf
            elif var in ['r_TII']:
                data_prime[var] *= (T_inf*rho_inf)
            elif var in ['rho','rhoI']:
                data_prime[var] *= rho_inf
            elif var in ['p','pI','pII']:
                data_prime[var] *= (rho_inf * U_inf**2)
            else:
                raise ValueError('condition needed for redimensionalizing \'%s\''%var)
        
        ## get lags
        lags,_  = ccor( np.ones(nz,dtype=np.float32) , np.ones(nz,dtype=np.float32), get_lags=True )
        n_lags_ = nz*2-1
        n_lags  = lags.shape[0]
        if (n_lags!=n_lags_):
            raise AssertionError('possible problem with lags calc --> check!')
        
        ## [var1, var2, density_scaling]
        R_combis = [
                   [ 'uI'  , 'uI'  , False ],
                   [ 'vI'  , 'vI'  , False ],
                   [ 'wI'  , 'wI'  , False ],
                   [ 'uI'  , 'vI'  , False ],
                   [ 'uI'  , 'TI'  , False ],
                   [ 'TI'  , 'TI'  , False ],
                   [ 'uII' , 'uII' , True  ],
                   [ 'vII' , 'vII' , True  ],
                   [ 'wII' , 'wII' , True  ],
                   [ 'uII' , 'vII' , True  ],
                   [ 'uII' , 'TII' , True  ],
                   [ 'TII' , 'TII' , True  ],
                   ]
        
        ## 5D [scalar][x,y,z,t] structured array --> cross-correlation data buffer
        scalars_R        = [ 'R_%s%s'%(cc[0],cc[1]) for cc in R_combis ]
        scalars_dtypes_R = [ np.float32 for s in scalars_R ]
        data_R           = np.zeros(shape=(nx, nyr, n_lags, nt), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
        
        ## check memory
        self.comm.Barrier()
        mem_total_gb = psutil.virtual_memory().total/1024**3
        mem_avail_gb = psutil.virtual_memory().available/1024**3
        mem_free_gb  = psutil.virtual_memory().free/1024**3
        if verbose:
            even_print('mem available', '%0.1f [GB] / %0.1f[%%]'%(mem_avail_gb,(100*mem_avail_gb/mem_total_gb)))
            even_print('mem free',      '%0.1f [GB] / %0.1f[%%]'%(mem_free_gb,(100*mem_free_gb/mem_total_gb)))
            print(72*'-')
        
        ## main loop --> cross-correlation in [z] at every [x,y,t]
        if True:
            if verbose: progress_bar = tqdm(total=nx*nyr*nt, ncols=100, desc='ccor_span()', leave=False, file=sys.stdout)
            for xi in range(nx):
                for yi in range(nyr):
                    for ti in range(nt):
                        for cci in range(len(R_combis)):
                            
                            tag = scalars_R[cci]
                            cc  = R_combis[cci]
                            ccL,ccR,density_scaling = cc[0],cc[1],cc[2]
                            
                            uL      = np.copy( data_prime[ccL][xi,yi,:,ti]   )
                            uR      = np.copy( data_prime[ccR][xi,yi,:,ti]   )
                            rho     = np.copy( data_prime['rho'][xi,yi,:,ti] )
                            #rho_avg = np.mean(rho, dtype=np.float64).astype(np.float32)
                            
                            if density_scaling:
                                data_R[tag][xi,yi,:,ti] = ccor( rho*uL , rho*uR )
                            else:
                                data_R[tag][xi,yi,:,ti] = ccor( uL , uR )
                        
                        if verbose: progress_bar.update()
            
            if verbose:
                progress_bar.close()
        
        ## manually delete the prime data from memory
        data_prime = None; del data_prime
        self.comm.Barrier()
        
        ## average in [x,t] --> leave [y,z] (where z is actually lag in z or Δz)
        data_R_avg = np.zeros(shape=(nyr, n_lags), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
        for scalar_R in scalars_R:
            data_R_avg[scalar_R] = np.mean(data_R[scalar_R], axis=(0,3), dtype=np.float64).astype(np.float32)
        data_R = None; del data_R
        
        # === gather all results --> this could probably be simplified with a gather/bcast
        
        self.comm.Barrier()
        data_R_all = None
        if (self.rank==0):
            
            j=0
            data_R_all = np.zeros(shape=(ny,n_lags), dtype={'names':scalars_R, 'formats':scalars_dtypes_R})
            
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
                    recvbuf = np.zeros((nyri,n_lags), dtype=data_R_avg[scalar_R].dtype)
                    ##
                    #print('rank %i : recvbuf.shape=%s'%(rank,str(recvbuf.shape)))
                    self.comm.Recv(recvbuf, source=ri, tag=ri)
                    data_R_all[scalar_R][ryl[j][0]:ryl[j][1],:] = recvbuf
                else:
                    pass
        
        ## overwrite
        if (self.rank==0):
            R = np.copy(data_R_all)
        
        # === save results
        
        if (self.rank==0):
            
            data['R']    = R ## the main cross-correlation data array
            data['lags'] = lags
            
            sc_l_in  = np.mean(sc_l_in  , axis=(0,1) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave 0D scalar
            sc_u_in  = np.mean(sc_u_in  , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_t_in  = np.mean(sc_t_in  , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_l_out = np.mean(sc_l_out , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_u_out = np.mean(sc_u_out , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_t_out = np.mean(sc_t_out , axis=(0,1) , dtype=np.float64).astype(np.float32)
            
            data['sc_l_in']  = sc_l_in
            data['sc_l_out'] = sc_l_out
            data['sc_t_in']  = sc_t_in
            data['sc_t_out'] = sc_t_out
            data['sc_u_in']  = sc_u_in
            data['sc_u_out'] = sc_u_out
            
            with open(fn_dat_ccor_span,'wb') as f:
                pickle.dump(data, f, protocol=4)
            print('--w-> %s : %0.2f [MB]'%(fn_dat_ccor_span,os.path.getsize(fn_dat_ccor_span)/1024**2))
        
        # ===
        
        self.comm.Barrier()
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.calc_ccor_span() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    def calc_turb_budget(self, **kwargs):
        '''
        calculate turbulent kinetic energy (k) budget
        -----
        --> dimensional [SI]
        --> requires that get_prime() was run with option favre=True
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
        chunk_kb                = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
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
                
                for dss in ['unsteady_production','unsteady_dissipation','unsteady_transport','unsteady_diffusion','unsteady_p_dilatation','unsteady_p_diffusion']:
                    
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
            
            for dss in ['production','dissipation','transport','diffusion','p_dilatation','p_diffusion']:
                
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
                ## p_inf = self.p_inf  ## should not be used for redimensionalization of p
                
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
                
                dudx   = np.gradient(u,   x, edge_order=2, axis=0)
                dudy   = np.gradient(u,   y, edge_order=2, axis=1)
                dudz   = np.gradient(u,   z, edge_order=2, axis=2)
                dvdx   = np.gradient(v,   x, edge_order=2, axis=0)
                dvdy   = np.gradient(v,   y, edge_order=2, axis=1)
                dvdz   = np.gradient(v,   z, edge_order=2, axis=2)
                dwdx   = np.gradient(w,   x, edge_order=2, axis=0)
                dwdy   = np.gradient(w,   y, edge_order=2, axis=1)
                dwdz   = np.gradient(w,   z, edge_order=2, axis=2)
                ##
                duIdx  = np.gradient(uI,  x, edge_order=2, axis=0)
                duIdy  = np.gradient(uI,  y, edge_order=2, axis=1)
                duIdz  = np.gradient(uI,  z, edge_order=2, axis=2)
                dvIdx  = np.gradient(vI,  x, edge_order=2, axis=0)
                dvIdy  = np.gradient(vI,  y, edge_order=2, axis=1)
                dvIdz  = np.gradient(vI,  z, edge_order=2, axis=2)
                dwIdx  = np.gradient(wI,  x, edge_order=2, axis=0)
                dwIdy  = np.gradient(wI,  y, edge_order=2, axis=1)
                dwIdz  = np.gradient(wI,  z, edge_order=2, axis=2)
                ##
                duIIdx = np.gradient(uII, x, edge_order=2, axis=0)
                duIIdy = np.gradient(uII, y, edge_order=2, axis=1)
                duIIdz = np.gradient(uII, z, edge_order=2, axis=2)
                dvIIdx = np.gradient(vII, x, edge_order=2, axis=0)
                dvIIdy = np.gradient(vII, y, edge_order=2, axis=1)
                dvIIdz = np.gradient(vII, z, edge_order=2, axis=2)
                dwIIdx = np.gradient(wII, x, edge_order=2, axis=0)
                dwIIdy = np.gradient(wII, y, edge_order=2, axis=1)
                dwIIdz = np.gradient(wII, z, edge_order=2, axis=2)
                
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
                    tc_ddx = np.gradient(tc, x, axis=0, edge_order=2)
                    tc_ddy = np.gradient(tc, y, axis=1, edge_order=2)
                    tc_ddz = np.gradient(tc, z, axis=2, edge_order=2)
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
                    A_ddx = np.gradient(A[:,:,:,:,0], x, axis=0, edge_order=2)
                    A_ddy = np.gradient(A[:,:,:,:,1], y, axis=1, edge_order=2)
                    A_ddz = np.gradient(A[:,:,:,:,2], z, axis=2, edge_order=2)
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
                
                # === pressure diffusion
                if True:
                    
                    if verbose: print(72*'-')
                    t_start = timeit.default_timer()
                    
                    A = np.einsum('xyzti,xyzt->xyzti', uII_i, pI)
                    A_ddx = np.gradient(A[:,:,:,:,0], x, axis=0, edge_order=2)
                    A_ddy = np.gradient(A[:,:,:,:,1], y, axis=1, edge_order=2)
                    A_ddz = np.gradient(A[:,:,:,:,2], z, axis=2, edge_order=2)
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
                    # A_ddx = np.gradient(A[:,:,:,:,0], x, axis=0, edge_order=2)
                    # A_ddy = np.gradient(A[:,:,:,:,1], y, axis=1, edge_order=2)
                    # A_ddz = np.gradient(A[:,:,:,:,2], z, axis=2, edge_order=2)
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
    
    def calc_high_order_stats(self,**kwargs):
        '''
        calculate skewness, kurtosis ('flatness'), probability distribution function (PDF)
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        if verbose: print('\n'+'rgd.calc_high_order_stats()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        rx = kwargs.get('rx',1)
        ry = kwargs.get('ry',1)
        rz = kwargs.get('rz',1)
        rt = kwargs.get('rt',1)
        
        fn_dat_hos      = kwargs.get('fn_dat_hos',None) ## hos = 'high-order stats'
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
        
        # === mean (dimensional) file name (for reading) : .dat
        if (fn_dat_mean_dim is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_dat_mean_base = fname_root+'_mean_dim.dat'
            fn_dat_mean_dim = str(PurePosixPath(fname_path, fname_dat_mean_base))
        
        # === high-order stats ('hos') file name (for writing) : dat
        if (fn_dat_hos is None):
            fname_path = os.path.dirname(self.fname)
            fname_base = os.path.basename(self.fname)
            fname_root, fname_ext = os.path.splitext(fname_base)
            fname_root = re.findall('io\S+_mpi_[0-9]+', fname_root)[0]
            fname_hos_dat_base = fname_root+'_hos.dat'
            fn_dat_hos = str(PurePosixPath(fname_path, fname_hos_dat_base))
        
        # ===
        
        if verbose: even_print('fn_rgd'     , self.fname )
        if verbose: even_print('fn_dat_hos' , fn_dat_hos )
        if verbose: print(72*'-')

        if not os.path.isfile(fn_dat_mean_dim):
            raise FileNotFoundError('%s not found!'%fn_dat_mean_dim)
        
        if True: ## open mean (dimensional) data
            
            with open(fn_dat_mean_dim,'rb') as f:
                data_mean_dim = pickle.load(f)
            fmd = type('foo', (object,), data_mean_dim)

            self.comm.Barrier()
            
            ## the data dictionary to be pickled later
            data = {}
            
            ## 2D dimensional quantities --> [x,z]
            u_tau    = fmd.u_tau    # ; data['u_tau']    = u_tau
            nu_wall  = fmd.nu_wall  # ; data['nu_wall']  = nu_wall
            rho_wall = fmd.rho_wall # ; data['rho_wall'] = rho_wall
            d99      = fmd.d99      # ; data['d99']      = d99
            u99      = fmd.u99      # ; data['u99']      = u99
            Re_tau   = fmd.Re_tau   # ; data['Re_tau']   = Re_tau
            Re_theta = fmd.Re_theta # ; data['Re_theta'] = Re_theta
            
            ## mean [x,z] --> leave 0D scalar
            u_tau_avg    = np.mean(fmd.u_tau    , axis=(0,1)) ; data['u_tau_avg']    = u_tau_avg
            nu_wall_avg  = np.mean(fmd.nu_wall  , axis=(0,1)) ; data['nu_wall_avg']  = nu_wall_avg
            rho_wall_avg = np.mean(fmd.rho_wall , axis=(0,1)) ; data['rho_wall_avg'] = rho_wall_avg
            d99_avg      = np.mean(fmd.d99      , axis=(0,1)) ; data['d99_avg']      = d99_avg
            u99_avg      = np.mean(fmd.u99      , axis=(0,1)) ; data['u99_avg']      = u99_avg
            Re_tau_avg   = np.mean(fmd.Re_tau   , axis=(0,1)) ; data['Re_tau_avg']   = Re_tau_avg
            Re_theta_avg = np.mean(fmd.Re_theta , axis=(0,1)) ; data['Re_theta_avg'] = Re_theta_avg
            
            ## mean [x,z] --> leave 1D [y]
            rho_avg = np.mean(fmd.rho,axis=(0,2))
            data['rho_avg'] = rho_avg
            
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
            
            lchar   = self.lchar   ; data['lchar']   = lchar
            U_inf   = self.U_inf   ; data['U_inf']   = U_inf
            rho_inf = self.rho_inf ; data['rho_inf'] = rho_inf
            T_inf   = self.T_inf   ; data['T_inf']   = T_inf
            
            data['Ma'] = self.Ma
            data['Pr'] = self.Pr
            
            nx = self.nx ; data['nx'] = nx
            ny = self.ny ; data['ny'] = ny
            nz = self.nz ; data['nz'] = nz
            nt = self.nt ; data['nt'] = nt

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
            
            data['x'] = x
            data['y'] = y
            data['z'] = z
            data['t'] = t
            data['t_meas'] = t_meas
            data['dt'] = dt
            
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
                even_print('Re_τ'   , '%0.1f'         % Re_tau_avg   )
                even_print('Re_θ'   , '%0.1f'         % Re_theta_avg )
                even_print('δ99'    , '%0.5e [m]'     % d99_avg      )
                even_print('U_inf'  , '%0.3f [m/s]'   % U_inf        )
                even_print('u_τ'    , '%0.3f [m/s]'   % u_tau_avg    )
                even_print('ν_wall' , '%0.5e [m²/s]'  % nu_wall_avg  )
                even_print('ρ_wall' , '%0.6f [kg/m³]' % rho_wall_avg  )
                print(72*'-')
            
            t_eddy = t_meas / ( d99_avg / u_tau_avg )
            
            if verbose:
                even_print('t_meas/(δ99/u_τ) = t_eddy' , '%0.2f'%t_eddy)
                even_print('t_meas/(δ99/u99)'          , '%0.2f'%(t_meas/(d99_avg/u99_avg)))
                even_print('t_meas/(20·δ99/u99)'       , '%0.2f'%(t_meas/(20*d99_avg/u99_avg)))
                print(72*'-')
        
        if True: ## read RGD data & dimensionalize
            
            scalars = [ 'u','T' ] ## 'v','w','p','rho'
            scalars_dtypes = [self.scalars_dtypes_dict[s] for s in scalars]
            
            ## 5D [scalar][x,y,z,t] structured array
            data_prime = np.zeros(shape=(self.nx, nyr, self.nz, self.nt), dtype={'names':scalars, 'formats':scalars_dtypes})
            
            for scalar in scalars:
                dset = self['data/%s'%scalar]
                self.comm.Barrier()
                t_start = timeit.default_timer()
                with dset.collective:
                    data_prime[scalar] = dset[:,:,ry1:ry2,:].T
                self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                data_gb = 4 * self.nx * self.ny * self.nz * self.nt / 1024**3
                if verbose:
                    even_print('read: %s'%scalar, '%0.3f [GB]  %0.3f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)))
            
            # === redimensionalize prime data
            
            for var in data_prime.dtype.names:
                if var in ['u','v','w', 'uI','vI','wI', 'uII','vII','wII']:
                    data_prime[var] *= U_inf
                elif var in ['r_uII','r_vII','r_wII']:
                    data_prime[var] *= (U_inf*rho_inf)
                elif var in ['T','TI','TII']:
                    data_prime[var] *= T_inf
                elif var in ['r_TII']:
                    data_prime[var] *= (T_inf*rho_inf)
                elif var in ['rho','rhoI']:
                    data_prime[var] *= rho_inf
                elif var in ['p','pI','pII']:
                    data_prime[var] *= (rho_inf * U_inf**2)
                else:
                    raise ValueError('condition needed for redimensionalizing \'%s\''%var)
        
        ## initialize buffers
        hos_scalars = [ '%s_mean'%(s,)    for s in scalars ] + \
                      [ '%sI_median'%(s,) for s in scalars ] + \
                      [ '%s_std'%(s,)     for s in scalars ] + \
                      [ '%s_skew'%(s,)    for s in scalars ] + \
                      [ '%s_kurt'%(s,)    for s in scalars ]
        
        hos_scalars_dtypes = [ np.float64 for s in hos_scalars ]
        #hos                = np.zeros(shape=(self.nx, nyr, self.nz) , dtype={'names':hos_scalars, 'formats':hos_scalars_dtypes})
        hos                = np.zeros(shape=(nyr,) , dtype={'names':hos_scalars, 'formats':hos_scalars_dtypes})
        
        n_bins = 1000
        
        hist_scalars = [ '%s'%(s,) for s in scalars ]
        hist_scalars_dtypes = [ np.float64 for s in hist_scalars ]
        hist = np.zeros(shape=(nyr, n_bins) , dtype={'names':hist_scalars, 'formats':hist_scalars_dtypes})
        
        bins_scalars = [ '%s'%(s,) for s in scalars ]
        bins_scalars_dtypes = [ np.float64 for s in bins_scalars ]
        bins = np.zeros(shape=(nyr, n_bins+1) , dtype={'names':bins_scalars, 'formats':bins_scalars_dtypes})
        
        ## check memory
        mem_total_gb = psutil.virtual_memory().total/1024**3
        mem_avail_gb = psutil.virtual_memory().available/1024**3
        mem_free_gb  = psutil.virtual_memory().free/1024**3
        if verbose:
            even_print('mem total',     '%0.1f [GB]'%(mem_total_gb,))
            even_print('mem available', '%0.1f [GB] / %0.1f[%%]'%(mem_avail_gb,(100*mem_avail_gb/mem_total_gb)))
            even_print('mem free',      '%0.1f [GB] / %0.1f[%%]'%(mem_free_gb,(100*mem_free_gb/mem_total_gb)))
            print(72*'-')
        
        ## main loop
        self.comm.Barrier()
        #if verbose: progress_bar = tqdm(total=self.nx*nyr*self.nz, ncols=100, desc='high order stats', leave=False)
        if verbose: progress_bar = tqdm(total=nyr, ncols=100, desc='high order stats', leave=False)
        for yi in range(nyr):
            for s in scalars:
                
                ## all [x,z,t] at this [y]
                d        = np.copy( data_prime[s][:,yi,:,:] ).astype(np.float64).ravel()
                d_mean   = np.mean( d , dtype=np.float64 )
                
                dI        = d - d_mean
                dI_median = np.median( dI )
                
                hist_ , bin_edges_ = np.histogram( dI , bins=n_bins , density=True )
                hist[s][yi,:] = hist_
                bins[s][yi,:] = bin_edges_
                
                #d_var  = np.mean( dI**2 ) ## = d.std()**2
                d_std  = np.sqrt( np.mean( dI**2 , dtype=np.float64 ) ) ## = d.std()
                
                if np.isclose(d_std, 0., atol=1e-08):
                    d_skew = 0.
                    d_kurt = 0.
                else:
                    d_skew = np.mean( dI**3 , dtype=np.float64 ) / d_std**3
                    d_kurt = np.mean( dI**4 , dtype=np.float64 ) / d_std**4 ## = sp.stats.kurtosis(d,fisher=False)
                
                hos['%sI_median'%s][yi] = dI_median
                hos['%s_mean'%s][yi]    = d_mean
                hos['%s_std'%s][yi]     = d_std
                hos['%s_skew'%s][yi]    = d_skew
                hos['%s_kurt'%s][yi]    = d_kurt
                
                if verbose: progress_bar.update()
        if verbose:
            progress_bar.close()
        self.comm.Barrier()
        
        # === average in [x,z], leave [y] --> not needed if stats are calculated over [x,z,t] per [y]
        
        ## hos_ = np.zeros(shape=(nyr,) , dtype={'names':hos_scalars, 'formats':hos_scalars_dtypes})
        ## for tag in hos_scalars:
        ##     hos_[tag] = np.mean( hos[tag] , axis=(0,2) , dtype=np.float64 )
        ## hos = np.copy( hos_ )
        ## self.comm.Barrier()
        
        # === gather
        
        G = self.comm.gather([ self.rank, hos, hist, bins ], root=0)
        G = self.comm.bcast(G, root=0)
        
        hos  = np.zeros( (ny,)         , dtype={'names':hos_scalars  , 'formats':hos_scalars_dtypes}  )
        hist = np.zeros( (ny,n_bins)   , dtype={'names':hist_scalars , 'formats':hist_scalars_dtypes} )
        bins = np.zeros( (ny,n_bins+1) , dtype={'names':bins_scalars , 'formats':bins_scalars_dtypes} )
        
        for ri in range(self.n_ranks):
            j = ri
            for GG in G:
                if (GG[0]==ri):
                    for tag in hos_scalars:
                        hos[tag][ryl[j][0]:ryl[j][1],] = GG[1][tag]
                    for tag in hist_scalars:
                        hist[tag][ryl[j][0]:ryl[j][1],:] = GG[2][tag]
                    for tag in bins_scalars:
                        bins[tag][ryl[j][0]:ryl[j][1],:] = GG[3][tag]
                else:
                    pass
        
        # === save results
        if (self.rank==0):
            
            data['hos']  = hos
            data['hist'] = hist
            data['bins'] = bins
            
            sc_l_in  = np.mean(sc_l_in  , axis=(0,1) , dtype=np.float64).astype(np.float32) ## avg in [x,z] --> leave 0D scalar
            sc_u_in  = np.mean(sc_u_in  , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_t_in  = np.mean(sc_t_in  , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_l_out = np.mean(sc_l_out , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_u_out = np.mean(sc_u_out , axis=(0,1) , dtype=np.float64).astype(np.float32)
            sc_t_out = np.mean(sc_t_out , axis=(0,1) , dtype=np.float64).astype(np.float32)
            
            data['sc_l_in']  = sc_l_in
            data['sc_l_out'] = sc_l_out
            data['sc_t_in']  = sc_t_in
            data['sc_t_out'] = sc_t_out
            data['sc_u_in']  = sc_u_in
            data['sc_u_out'] = sc_u_out
            
            with open(fn_dat_hos,'wb') as f:
                pickle.dump(data, f, protocol=4)
            print('--w-> %s : %0.2f [MB]'%(fn_dat_hos,os.path.getsize(fn_dat_hos)/1024**2))
        
        # ===
        
        self.comm.Barrier()
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : rgd.calc_high_order_stats() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
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
        chunk_kb = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        fn_lpd   = kwargs.get('fn_lpd','pts.h5')
        scheme   = kwargs.get('scheme','RK4')
        npts     = kwargs.get('npts',1e4)
        ntc      = kwargs.get('ntc',None)
        
        if (ntc is not None):
            if not isinstance(ntc,int):
                raise ValueError('ntc should be of type int')
            if (ntc > self.nt):
                raise ValueError('more ts requested than exist')
        
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
        
        read_dat_mean_dim = False
        
        if read_dat_mean_dim:
            
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
        
        if verbose: even_print('fn_rgd'              , self.fname      )
        if read_dat_mean_dim:
            if verbose: even_print('fn_dat_mean_dim' , fn_dat_mean_dim )
        if verbose: even_print('fn_lpd'              , fn_lpd          )
        if verbose: even_print('n ts rgd'            , '%i'%self.nt    )
        
        if read_dat_mean_dim: ## mean dimensional data [x,z]
            
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
        
        if (ntc is None):
            tc  = np.copy(self.t).astype(np.float64)
            ntc = tc.size
            dtc = tc[1]-tc[0]
        else:
            tc  = np.copy(self.t[:ntc]).astype(np.float64)
            ntc = tc.size
            dtc = tc[1]-tc[0]
        
        if verbose: even_print('n ts for convection' , '%i'%ntc )
        if verbose: print(72*'-')
        
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
                fac           = 0.9500000 ## approximates U(y)dy integral
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
                
                xp = rng.uniform(self.x.min(), self.x.max(), size=(npts_init,)) ## double precision
                yp = rng.uniform(self.y.min(), self.y.max(), size=(npts_init,))
                zp = rng.uniform(self.z.min(), self.z.max(), size=(npts_init,))
                tp = tc[0]*np.ones((npts_init,), dtype=xp.dtype)
                
                xyztp = np.stack((xp,yp,zp,tp)).T
                
                ## check
                if (xyztp.dtype!=np.float64):
                    raise AssertionError
                
                pnum = np.arange(npts_init, dtype=np.int64)
                
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
                
                ## check
                if (xyztp.dtype!=np.float64):
                    raise AssertionError
                
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
        ##         return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 8M %s > /dev/null 2>&1'%('particles.h5',), shell=True)
        ## self.comm.Barrier()
        
        # ===
        
        pcomm = MPI.COMM_WORLD
        with lpd(fn_lpd, 'w', force=force, driver='mpio', comm=pcomm, libver='latest') as hf_lpd:
            
            ## 'self' passed here is RGD / h5py.File instance
            ## this copies over all the header info from the RGD: U_inf, lchar, etc
            hf_lpd.init_from_rgd(self, t_info=False)
            
            ## shape & HDF5 chunk scheme for datasets
            shape = (npts_all_ts, ntc-1)
            chunks = rgd.chunk_sizer(nxi=shape, constraint=(None,1), size_kb=chunk_kb, base=2)
            
            scalars = [ 'x','y','z', 
                        'u','v','w', 
                        't','id'     ]
            
            scalars_dtype = [ np.float64, np.float64, np.float64, 
                              np.float32, np.float32, np.float32, 
                              np.float64, np.int64                 ]
            
            for si in range(len(scalars)):
                
                scalar       = scalars[si]
                scalar_dtype = scalars_dtype[si]
                
                if verbose:
                    even_print('initializing',scalar)
                
                if ('data/%s'%scalar in hf_lpd):
                    del hf_lpd['data/%s'%scalar]
                dset = hf_lpd.create_dataset('data/%s'%scalar, 
                                          shape=shape, 
                                          dtype=scalar_dtype,
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
            hf_lpd.create_dataset('dims/t', data=tc[:-1], dtype=tc.dtype, chunks=True)
            
            pcomm.Barrier()
            self.comm.Barrier()
            
            if verbose:
                print(72*'-')
                even_print( os.path.basename(fn_lpd) , '%0.2f [GB]'%(os.path.getsize(fn_lpd)/1024**3))
                print(72*'-')
            
            if True: ## convect fwd
                
                if verbose:
                    progress_bar = tqdm(total=ntc-1, ncols=100, desc='convect fwd', leave=False, file=sys.stdout)
                
                t_write = 0.
                
                if (dtc!=self.dt):
                    raise AssertionError('dtc!=self.dt')
                
                ## the global list of all particle IDs that have left volume
                pnum_nan_global = np.array([],dtype=np.int64)
                
                for tci in range(ntc-1):
                    
                    if verbose:
                        if (tci>0):
                            tqdm.write('---')
                    
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
                    xyztp = np.concatenate((xyztp, xyztp_new[ii]), axis=0, casting='no')
                    pnum  = np.concatenate((pnum,  pnum_new[ii]),  axis=0, casting='no')
                    
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
                    pnum_nan_global_this_ts = np.concatenate( [g[0] for g in G] , casting='no' )
                    pnum_nan_global         = np.concatenate( (pnum_nan_global_this_ts , pnum_nan_global) , casting='no' )
                    ##
                    npts_nan = pnum_nan_global.shape[0]
                    
                    ## take only non-NaN position particles
                    pnum  = np.copy(pnum[ii_notnan])
                    xyztp = np.copy(xyztp[ii_notnan])
                    
                    if True: ## check global pnum (pt id) vector
                        
                        G = self.comm.gather([np.copy(pnum), self.rank], root=0)
                        G = self.comm.bcast(G, root=0)
                        pnum_global = np.sort( np.concatenate( [g[0] for g in G] , casting='no' ) )
                        pnum_global = np.sort( np.concatenate((pnum_global,pnum_nan_global), casting='no') )
                        
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
                    pnum_gl    = np.concatenate([g[1] for g in G], axis=0, casting='no',        dtype=np.int64   )
                    x_gl       = np.concatenate([g[2] for g in G], axis=0, casting='no',        dtype=np.float64 )
                    y_gl       = np.concatenate([g[3] for g in G], axis=0, casting='no',        dtype=np.float64 )
                    z_gl       = np.concatenate([g[4] for g in G], axis=0, casting='no',        dtype=np.float64 )
                    t_gl       = np.concatenate([g[5] for g in G], axis=0, casting='no',        dtype=np.float64 )
                    u_gl       = np.concatenate([g[6] for g in G], axis=0, casting='same_kind', dtype=np.float32 )
                    v_gl       = np.concatenate([g[7] for g in G], axis=0, casting='same_kind', dtype=np.float32 )
                    w_gl       = np.concatenate([g[8] for g in G], axis=0, casting='same_kind', dtype=np.float32 )
                    ##
                    if verbose: tqdm.write(even_print('n pts initialized', '%i'%(offset,),      s=True))
                    if verbose: tqdm.write(even_print('n pts in domain',   '%i'%(npts_total,),  s=True))
                    if verbose: tqdm.write(even_print('n pts left domain', '%i'%(npts_nan,),    s=True))
                    if verbose: tqdm.write(even_print('n pts all time',    '%i'%(npts_all_ts,), s=True))
                    
                    # === add NaN IDs, pad scalar vectors, do sort
                    
                    pnum_gl = np.concatenate([pnum_gl,pnum_nan_global], axis=0, casting='no', dtype=np.int64 )
                    npts_total_incl_nan = pnum_gl.shape[0]
                    
                    if (npts_total_incl_nan!=offset):
                        raise AssertionError('npts_total_incl_nan!=offset')
                    
                    nanpad_f32 = np.empty( (npts_nan,), dtype=np.float32 ); nanpad_f32[:] = np.nan
                    nanpad_f64 = np.empty( (npts_nan,), dtype=np.float64 ); nanpad_f64[:] = np.nan
                    #nanpad_i64 = np.empty( (npts_nan,), dtype=np.int64   ); nanpad_i64[:] = np.nan
                    
                    x_gl = np.concatenate( [x_gl,nanpad_f64], axis=0, dtype=np.float64, casting='no' )
                    y_gl = np.concatenate( [y_gl,nanpad_f64], axis=0, dtype=np.float64, casting='no' )
                    z_gl = np.concatenate( [z_gl,nanpad_f64], axis=0, dtype=np.float64, casting='no' )
                    t_gl = np.concatenate( [t_gl,nanpad_f64], axis=0, dtype=np.float64, casting='no' )
                    u_gl = np.concatenate( [u_gl,nanpad_f32], axis=0, dtype=np.float32, casting='no' )
                    v_gl = np.concatenate( [v_gl,nanpad_f32], axis=0, dtype=np.float32, casting='no' )
                    w_gl = np.concatenate( [w_gl,nanpad_f32], axis=0, dtype=np.float32, casting='no' )
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
                    
                    ## check that the global particle number / ID vector is simply arange(total an pts created)
                    if not np.array_equal(pnum_gl, np.arange(offset, dtype=np.int64)):
                        raise AssertionError('pnum_gl!=np.arange(offset, dtype=np.int64)')
                    
                    ## check that all particle times for this ts are equal --> passes
                    if False:
                        ii_notnan = np.where(~np.isnan(t_gl))
                        if not np.all( np.isclose(t_gl[ii_notnan], t_gl[ii_notnan][0], rtol=1e-14) ):
                            raise AssertionError('not all times are the same at this time integration step --> check!!!')
                    
                    # === get collective write bounds
                    
                    rpl_ = np.array_split(np.arange(npts_total_incl_nan,dtype=np.int64) , self.n_ranks)
                    rpl = [[b[0],b[-1]+1] for b in rpl_ ]
                    rp1,rp2 = rpl[self.rank]
                    
                    # === write
                    
                    for key, value in {'id':pnum_gl, 'x':x_gl, 'y':y_gl, 'z':z_gl, 't':t_gl, 'u':u_gl, 'v':v_gl, 'w':w_gl}.items():
                        
                        data_gb = value.itemsize * npts_total_incl_nan / 1024**3
                        
                        dset = hf_lpd['data/%s'%key]
                        pcomm.Barrier()
                        t_start = timeit.default_timer()
                        with dset.collective:
                            dset[rp1:rp2,tci] = value[rp1:rp2]
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
                        raise NotImplementedError('integration scheme \'%s\' not valid. options are: \'Euler Explicit\', \'RK4\''%scheme)
                    
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
                    xyztp_out = np.concatenate((np.copy(xyztp[i_exit_top]) , np.copy(xyztp[i_exit_bot])), axis=0, casting='no' )
                    pnum_out  = np.concatenate((np.copy(pnum[i_exit_top])  , np.copy(pnum[i_exit_bot])),  axis=0, casting='no' )
                    
                    ## delete those from local lists
                    i_del = np.concatenate( (i_exit_top[0], i_exit_bot[0]) , axis=0, casting='no' )
                    xyztp = np.delete(xyztp, (i_del,), axis=0)
                    pnum  = np.delete(pnum,  (i_del,), axis=0)
                    
                    # === MPI : Gather/Bcast all inter-domain pts
                    G = self.comm.gather([np.copy(xyztp_out), np.copy(pnum_out), self.rank], root=0)
                    G = self.comm.bcast(G, root=0)
                    xyztpN = np.concatenate([x[0] for x in G], axis=0, casting='no' )
                    pnumN  = np.concatenate([x[1] for x in G], axis=0, casting='no' )
                    
                    ## get indices to 'take' from inter-domain points
                    if (self.rank==0):
                        i_take = np.where(xyztpN[:,1]<=y_max)
                    elif (self.rank==self.n_ranks-1):
                        i_take = np.where(xyztpN[:,1]>=y_min)
                    else:
                        i_take = np.where((xyztpN[:,1]<y_max) & (xyztpN[:,1]>=y_min))
                    ##
                    xyztp = np.concatenate((xyztp, xyztpN[i_take]), axis=0, casting='no' )
                    pnum  = np.concatenate((pnum,  pnumN[i_take]),  axis=0, casting='no' )
                    
                    if verbose:
                        progress_bar.update()
                    
                    self.comm.Barrier()
                    pcomm.Barrier()
                
                if verbose:
                    progress_bar.close()
        
        if verbose: print(72*'-'+'\n')
        
        pcomm.Barrier()
        self.comm.Barrier()
        
        ## make XDMF/XMF2
        with lpd(fn_lpd, 'r', driver='mpio', comm=pcomm) as hf_lpd:
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
        
        makeVectors = kwargs.get('makeVectors',True) ## write vectors (e.g. velocity, vorticity) to XDMF
        makeTensors = kwargs.get('makeTensors',True) ## write 3x3 tensors (e.g. stress, strain) to XDMF
        
        fname_path            = os.path.dirname(self.fname)
        fname_base            = os.path.basename(self.fname)
        fname_root, fname_ext = os.path.splitext(fname_base)
        fname_xdmf_base       = fname_root+'.xmf2'
        fname_xdmf            = os.path.join(fname_path, fname_xdmf_base)
        
        if verbose: print('\n'+'rgd.make_xdmf()'+'\n'+72*'-')
        
        dataset_precision_dict = {} ## holds dtype.itemsize ints i.e. 4,8
        dataset_numbertype_dict = {} ## holds string description of dtypes i.e. 'Float','Integer'
        
        # === 1D coordinate dimension vectors --> get dtype.name
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
        
        # scalar names dict
        # --> labels for Paraview could be customized (e.g. units could be added) using a dict
        # --> the block below shows one such example dict, though it is currently inactive
        
        if False:
            units = 'dimless'
            if (units=='SI') or (units=='si'): ## m,s,kg,K
                scalar_names = {'x':'x [m]',
                                'y':'y [m]',
                                'z':'z [m]', 
                                'u':'u [m/s]',
                                'v':'v [m/s]',
                                'w':'w [m/s]', 
                                'T':'T [K]',
                                'rho':'rho [kg/m^3]',
                                'p':'p [Pa]'}
            elif (units=='dimless') or (units=='dimensionless'):
                scalar_names = {'x':'x [dimless]',
                                'y':'y [dimless]',
                                'z':'z [dimless]', 
                                'u':'u [dimless]',
                                'v':'v [dimless]',
                                'w':'w [dimless]',
                                'T':'T [dimless]',
                                'rho':'rho [dimless]',
                                'p':'p [dimless]'}
            else:
                raise ValueError('choice of units not recognized : %s --> options are : %s / %s'%(units,'SI','dimless'))
        else:
            scalar_names = {} ## dummy/empty 
        
        ## refresh header
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
        
        # === write to .xdmf/.xmf2 file
        if (self.rank==0):
            
            #with open(fname_xdmf,'w') as xdmf:
            with io.open(fname_xdmf,'w',newline='\n') as xdmf:
                
                xdmf_str='''
                         <?xml version="1.0" encoding="utf-8"?>
                         <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
                         <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
                           <Domain>
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 0*' '))
                
                ## Dimensions can also be NumberOfElements
                xdmf_str=f'''
                         <Topology TopologyType="3DRectMesh" NumberOfElements="{self.nz:d} {self.ny:d} {self.nx:d}"/>
                         <Geometry GeometryType="VxVyVz">
                           <DataItem Dimensions="{self.nx:d}" NumberType="{dataset_numbertype_dict['x']}" Precision="{dataset_precision_dict['x']:d}" Format="HDF">
                             {fname_base}:/dims/{'x'}
                           </DataItem>
                           <DataItem Dimensions="{self.ny:d}" NumberType="{dataset_numbertype_dict['y']}" Precision="{dataset_precision_dict['y']:d}" Format="HDF">
                             {fname_base}:/dims/{'y'}
                           </DataItem>
                           <DataItem Dimensions="{self.nz:d}" NumberType="{dataset_numbertype_dict['z']}" Precision="{dataset_precision_dict['z']:d}" Format="HDF">
                             {fname_base}:/dims/{'z'}
                           </DataItem>
                         </Geometry>
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 4*' '))
                
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
                    
                    dset_name = 'ts_%08d'%ti
                    
                    xdmf_str='''
                             <!-- ============================================================ -->
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # =====
                    
                    xdmf_str=f'''
                             <Grid Name="{dset_name}" GridType="Uniform">
                               <Time TimeType="Single" Value="{self.t[ti]:0.8E}"/>
                               <Topology Reference="/Xdmf/Domain/Topology[1]" />
                               <Geometry Reference="/Xdmf/Domain/Geometry[1]" />
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # ===== .xdmf : <Grid> per scalar
                    
                    for scalar in self.scalars:
                        
                        dset_hf_path = 'data/%s'%scalar
                        
                        ## get optional 'label' for Paraview (currently inactive)
                        if scalar in scalar_names:
                            scalar_name = scalar_names[scalar]
                        else:
                            scalar_name = scalar
                        
                        xdmf_str=f'''
                                 <!-- ===== scalar : {scalar} ===== -->
                                 <Attribute Name="{scalar_name}" AttributeType="Scalar" Center="Node">
                                   <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                     <DataItem Dimensions="3 4" NumberType="Integer" Format="XML">
                                       {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                       {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                       {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                     </DataItem>
                                     <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict[scalar]}" Precision="{dataset_precision_dict[scalar]:d}" Format="HDF">
                                       {fname_base}:/{dset_hf_path}
                                     </DataItem>
                                   </DataItem>
                                 </Attribute>
                                 '''
                        
                        xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    if makeVectors:
                        
                        # === .xdmf : <Grid> per vector : velocity vector
                        
                        if ('u' in self.scalars) and ('v' in self.scalars) and ('w' in self.scalars):
                            
                            scalar_name    = 'velocity'
                            dset_hf_path_i = 'data/u'
                            dset_hf_path_j = 'data/v'
                            dset_hf_path_k = 'data/w'
                            
                            xdmf_str = f'''
                            <!-- ===== vector : {scalar_name} ===== -->
                            <Attribute Name="{scalar_name}" AttributeType="Vector" Center="Node">
                              <DataItem Dimensions="{self.nz:d} {self.ny:d} {self.nx:d} {3:d}" Function="JOIN($0, $1, $2)" ItemType="Function">
                                <!-- 1 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['u']}" Precision="{dataset_precision_dict['u']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_i}
                                  </DataItem>
                                </DataItem>
                                <!-- 2 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['v']}" Precision="{dataset_precision_dict['v']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_j}
                                  </DataItem>
                                </DataItem>
                                <!-- 3 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['w']}" Precision="{dataset_precision_dict['w']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_k}
                                  </DataItem>
                                </DataItem>
                                <!-- - -->
                              </DataItem>
                            </Attribute>
                            '''
                            
                            xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                        
                        # === .xdmf : <Grid> per vector : vorticity vector
                        
                        if ('vort_x' in self.scalars) and ('vort_y' in self.scalars) and ('vort_z' in self.scalars):
                            
                            scalar_name    = 'vorticity'
                            dset_hf_path_i = 'data/vort_x'
                            dset_hf_path_j = 'data/vort_y'
                            dset_hf_path_k = 'data/vort_z'
                            
                            xdmf_str = f'''
                            <!-- ===== vector : {scalar_name} ===== -->
                            <Attribute Name="{scalar_name}" AttributeType="Vector" Center="Node">
                              <DataItem Dimensions="{self.nz:d} {self.ny:d} {self.nx:d} {3:d}" Function="JOIN($0, $1, $2)" ItemType="Function">
                                <!-- 1 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['vort_x']}" Precision="{dataset_precision_dict['vort_x']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_i}
                                  </DataItem>
                                </DataItem>
                                <!-- 2 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['vort_y']}" Precision="{dataset_precision_dict['vort_y']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_j}
                                  </DataItem>
                                </DataItem>
                                <!-- 3 -->
                                <DataItem ItemType="HyperSlab" Dimensions="{self.nz:d} {self.ny:d} {self.nx:d}" Type="HyperSlab">
                                  <DataItem Dimensions="3 4" Format="XML">
                                    {ti:<6d} {0:<6d} {0:<6d} {0:<6d}
                                    {1:<6d} {1:<6d} {1:<6d} {1:<6d}
                                    {1:<6d} {self.nz:<6d} {self.ny:<6d} {self.nx:<6d}
                                  </DataItem>
                                  <DataItem Dimensions="{self.nt:d} {self.nz:d} {self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict['vort_z']}" Precision="{dataset_precision_dict['vort_z']:d}" Format="HDF">
                                    {fname_base}:/{dset_hf_path_k}
                                  </DataItem>
                                </DataItem>
                                <!-- - -->
                              </DataItem>
                            </Attribute>
                            '''
                            
                            xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    if makeTensors:
                        if all([('dudx' in self.scalars),('dvdx' in self.scalars),('dwdx' in self.scalars),
                                ('dudy' in self.scalars),('dvdy' in self.scalars),('dwdy' in self.scalars),
                                ('dudz' in self.scalars),('dvdz' in self.scalars),('dwdz' in self.scalars)]):
                            pass
                            pass ## TODO
                            pass
                    
                    # === .xdmf : end Grid for this timestep
                    
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
    ------------------------------
    - super()'ed h5py.File class
    - EAS4 is the HDF5-based output format from the flow solver NS3D
    - 3D dataset storage ([x,y,z] per [t])
    '''
    
    def __init__(self, *args, **kwargs):
        
        self.fname, openMode = args
        
        self.fname_path = os.path.dirname(self.fname)
        self.fname_base = os.path.basename(self.fname)
        self.fname_root, self.fname_ext = os.path.splitext(self.fname_base)
        
        ## default to libver='latest' if none provided
        if ('libver' not in kwargs):
            kwargs['libver'] = 'latest'
        
        if (openMode!='r'):
            raise ValueError('turbx.eas4(): opening EAS4 in anything but read mode \'r\' is discouraged!')
        
        ## catch possible user error
        if (self.fname_ext!='.eas'):
            raise ValueError('turbx.eas4() should not be used to open non-EAS4 files')
        
        ## determine if using mpi
        if ('driver' in kwargs) and (kwargs['driver']=='mpio'):
            self.usingmpi = True
        else:
            self.usingmpi = False
        
        ## determine communicator & rank info
        if self.usingmpi:
            self.comm    = kwargs['comm']
            self.n_ranks = self.comm.Get_size()
            self.rank    = self.comm.Get_rank()
        else:
            self.comm    = None
            self.n_ranks = 1
            self.rank    = 0
            if ('comm' in kwargs):
                del kwargs['comm']
        
        ## set library version to latest (if not otherwise set)
        if ('libver' not in kwargs):
            kwargs['libver']='latest'
        
        ## determine MPI info / hints
        if self.usingmpi:
            if ('info' in kwargs):
                self.mpi_info = kwargs['info']
            else:
                mpi_info = MPI.Info.Create()
                ##
                mpi_info.Set('romio_ds_write' , 'disable'   )
                mpi_info.Set('romio_ds_read'  , 'disable'   )
                mpi_info.Set('romio_cb_read'  , 'automatic' )
                mpi_info.Set('romio_cb_write' , 'automatic' )
                #mpi_info.Set('romio_cb_read'  , 'enable' )
                #mpi_info.Set('romio_cb_write' , 'enable' )
                mpi_info.Set('cb_buffer_size' , str(int(round(8*1024**2))) ) ## 8 [MB]
                ##
                kwargs['info'] = mpi_info
                self.mpi_info = mpi_info
        
        if ('rdcc_nbytes' not in kwargs):
            kwargs['rdcc_nbytes'] = int(16*1024**2) ## 16 [MB]
        
        self.domainName = 'DOMAIN_000000' ## turbx only handles one domain for now
        
        ## eas4() unique kwargs (not h5py.File kwargs) --> pop() rather than get()
        verbose = kwargs.pop('verbose',False)
        self.verbose = verbose
        ## force = kwargs.pop('force',False) ## --> dont need, always read-only!
        
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
    
    def get_header(self, **kwargs):
        
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
        rho_inf = p_inf / ( R * T_inf )
        
        mu_inf_1 = 14.58e-7*T_inf**1.5/(T_inf+110.4)
        mu_inf_2 = mu_Suth_ref*(T_inf/T_Suth_ref)**(3/2)*(T_Suth_ref+S_Suth)/(T_inf+S_Suth)
        mu_inf_3 = C_Suth*T_inf**(3/2)/(T_inf+S_Suth)
        
        if not np.isclose(mu_inf_1, mu_inf_2, rtol=1e-08):
            raise AssertionError('inconsistency in Sutherland calc --> check')
        if not np.isclose(mu_inf_2, mu_inf_3, rtol=1e-08):
            raise AssertionError('inconsistency in Sutherland calc --> check')
        
        mu_inf    = mu_inf_3
        nu_inf    = mu_inf/rho_inf
        a_inf     = np.sqrt(kappa*R*T_inf)
        U_inf     = Ma*a_inf
        cp        = R*kappa/(kappa-1.)
        cv        = cp/kappa
        recov_fac = Pr**(1/3)
        Tw        = T_inf
        Taw       = T_inf + recov_fac*U_inf**2/(2*cp)
        lchar     = Re * nu_inf / U_inf
        
        if self.verbose: even_print('rho_inf'         , '%0.3f [kg/m³]'    % rho_inf   )
        if self.verbose: even_print('mu_inf'          , '%0.6E [kg/(m·s)]' % mu_inf    )
        if self.verbose: even_print('nu_inf'          , '%0.6E [m²/s]'     % nu_inf    )
        if self.verbose: even_print('a_inf'           , '%0.6f [m/s]'      % a_inf     )
        if self.verbose: even_print('U_inf'           , '%0.6f [m/s]'      % U_inf     )
        if self.verbose: even_print('cp'              , '%0.3f [J/(kg·K)]' % cp        )
        if self.verbose: even_print('cv'              , '%0.3f [J/(kg·K)]' % cv        )
        if self.verbose: even_print('recovery factor' , '%0.6f [-]'        % recov_fac )
        if self.verbose: even_print('Tw'              , '%0.3f [K]'        % Tw        )
        if self.verbose: even_print('Taw'             , '%0.3f [K]'        % Taw       )
        if self.verbose: even_print('lchar'           , '%0.6E [m]'        % lchar     )
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
        self.recov_fac    = recov_fac
        self.Tw           = Tw
        self.Taw          = Taw
        self.lchar        = lchar

        # === check if this a 2D average file like 'mean_flow_mpi.eas'
        
        if self.verbose: print(72*'-')
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
        if self.verbose: even_print('meas type', '\'%s\''%self.measType)
        if self.verbose: print(72*'-'+'\n')
        
        # === grid info
        
        ndim1 = self['Kennsatz/GEOMETRY/%s'%self.domainName].attrs['DOMAIN_SIZE'][0]
        ndim2 = self['Kennsatz/GEOMETRY/%s'%self.domainName].attrs['DOMAIN_SIZE'][1]
        ndim3 = self['Kennsatz/GEOMETRY/%s'%self.domainName].attrs['DOMAIN_SIZE'][2]

        nx = self.nx = ndim1
        ny = self.ny = ndim2
        nz = self.nz = ndim3
        
        if (self.measType=='mean'):
            nz = self.nz = 1
        
        ngp = self.ngp = nx*ny*nz
        
        if self.verbose: print('grid info\n'+72*'-')
        if self.verbose: even_print('nx',  '%i'%nx  )
        if self.verbose: even_print('ny',  '%i'%ny  )
        if self.verbose: even_print('nz',  '%i'%nz  )
        if self.verbose: even_print('ngp', '%i'%ngp )
        
        gmode_dim1 = self.gmode_dim1 = self['Kennsatz/GEOMETRY/%s'%self.domainName].attrs['DOMAIN_GMODE'][0]
        gmode_dim2 = self.gmode_dim2 = self['Kennsatz/GEOMETRY/%s'%self.domainName].attrs['DOMAIN_GMODE'][1]
        gmode_dim3 = self.gmode_dim3 = self['Kennsatz/GEOMETRY/%s'%self.domainName].attrs['DOMAIN_GMODE'][2]
        
        ## the original gmode (pre-conversion)
        gmode_dim1_orig = self.gmode_dim1_orig = gmode_dim1
        gmode_dim2_orig = self.gmode_dim2_orig = gmode_dim2
        gmode_dim3_orig = self.gmode_dim3_orig = gmode_dim3
        
        if self.verbose: even_print( 'gmode dim1' , '%i / %s'%(gmode_dim1,gmode_dict[gmode_dim1]) )
        if self.verbose: even_print( 'gmode dim2' , '%i / %s'%(gmode_dim2,gmode_dict[gmode_dim2]) )
        if self.verbose: even_print( 'gmode dim3' , '%i / %s'%(gmode_dim3,gmode_dict[gmode_dim3]) )
        if self.verbose: print(72*'-')
        
        ## read grid
        ## fails if >2[GB] and using driver=='mpio' and using one process --> https://github.com/h5py/h5py/issues/1052
        #try:
        if True:
            dim1_data = np.copy(self['Kennsatz/GEOMETRY/%s/dim01'%self.domainName][:])
            dim2_data = np.copy(self['Kennsatz/GEOMETRY/%s/dim02'%self.domainName][:])
            dim3_data = np.copy(self['Kennsatz/GEOMETRY/%s/dim03'%self.domainName][:])
        
        #except OSError:
        if False:
            if (gmode_dim1 == EAS4_FULL_G):
                dim1_data = np.zeros((nx,ny,nz), dtype = self['Kennsatz/GEOMETRY/%s/dim01'%self.domainName].dtype)
                for i in range(nx):
                    dim1_data[i,:,:] = self['Kennsatz/GEOMETRY/%s/dim01'%self.domainName][i,:,:]
            else:
                dim1_data = self['Kennsatz/GEOMETRY/%s/dim01'%self.domainName][:]
            
            if (gmode_dim2 == EAS4_FULL_G):
                dim2_data = np.zeros((nx,ny,nz), dtype = self['Kennsatz/GEOMETRY/%s/dim02'%self.domainName].dtype)
                for i in range(nx):
                    dim2_data[i,:,:] = self['Kennsatz/GEOMETRY/%s/dim02'%self.domainName][i,:,:]
            else:
                dim2_data = self['Kennsatz/GEOMETRY/%s/dim02'%self.domainName][:]
            
            if (gmode_dim3 == EAS4_FULL_G):
                dim3_data = np.zeros((nx,ny,nz), dtype = self['Kennsatz/GEOMETRY/%s/dim03'%self.domainName].dtype)
                for i in range(nx):
                    dim3_data[i,:,:] = self['Kennsatz/GEOMETRY/%s/dim03'%self.domainName][i,:,:]
            else:
                dim3_data = self['Kennsatz/GEOMETRY/%s/dim03'%self.domainName][:]
        
        ## check grid for span avg
        if (self.measType == 'mean'):
            if (gmode_dim1 == EAS4_FULL_G):
                if not np.allclose(dim1_data[:,:,0], dim1_data[:,:,1], rtol=1e-08):
                    raise AssertionError('check')
            if (gmode_dim2 == EAS4_FULL_G):
                if not np.allclose(dim2_data[:,:,0], dim2_data[:,:,1], rtol=1e-08):
                    raise AssertionError('check')
        
        ## convert EAS4_NO_G to EAS4_ALL_G (1 --> 4) --> always do this
        ## dim_n_data are already numpy arrays of shape (1,) --> no conversion necessary, just update 'gmode_dimn' attr
        if (gmode_dim1 == EAS4_NO_G):
            gmode_dim1 = self.gmode_dim1 = EAS4_ALL_G
        if (gmode_dim2 == EAS4_NO_G):
            gmode_dim2 = self.gmode_dim2 = EAS4_ALL_G
        if (gmode_dim3 == EAS4_NO_G):
            gmode_dim3 = self.gmode_dim3 = EAS4_ALL_G
        
        ## convert EAS4_X0DX_G to EAS4_ALL_G (2 --> 4) --> always do this
        if (gmode_dim1 == EAS4_X0DX_G):
            dim1_data  = np.linspace(dim1_data[0],dim1_data[0]+dim1_data[1]*(ndim1-1), ndim1)
            gmode_dim1 = self.gmode_dim1 = EAS4_ALL_G
        if (gmode_dim2 == EAS4_X0DX_G):
            dim2_data  = np.linspace(dim2_data[0],dim2_data[0]+dim2_data[1]*(ndim2-1), ndim2)
            gmode_dim2 = self.gmode_dim2 = EAS4_ALL_G
        if (gmode_dim3 == EAS4_X0DX_G):
            dim3_data  = np.linspace(dim3_data[0],dim3_data[0]+dim3_data[1]*(ndim3-1), ndim3)
            gmode_dim3 = self.gmode_dim3 = EAS4_ALL_G
        
        ## convert EAS4_ALL_G to EAS4_FULL_G (4 --> 5) --> only do this if at least one dimension is EAS4_FULL_G (5)
        if any([(gmode_dim1==5),(gmode_dim2==5),(gmode_dim3==5)]):
            
            self.isCurvilinear = True
            self.isRectilinear = False
            
            if (gmode_dim1 == EAS4_ALL_G):
                dim1_data  = np.broadcast_to(dim1_data, (ndim1,ndim2,ndim3))
                gmode_dim1 = self.gmode_dim1 = EAS4_FULL_G
            if (gmode_dim2 == EAS4_ALL_G):
                dim2_data  = np.broadcast_to(dim2_data, (ndim1,ndim2,ndim3))
                gmode_dim2 = self.gmode_dim2 = EAS4_FULL_G
            if (gmode_dim3 == EAS4_ALL_G):
                dim3_data  = np.broadcast_to(dim3_data, (ndim1,ndim2,ndim3))
                gmode_dim3 = self.gmode_dim3 = EAS4_FULL_G
        
        else:
            
            self.isCurvilinear = False
            self.isRectilinear = True
        
        # ===
        
        x = self.x = np.copy(dim1_data)
        y = self.y = np.copy(dim2_data)
        z = self.z = np.copy(dim3_data)
        
        # ## bug check
        # if (z.size > 1):
        #     if np.all(np.isclose(z,z[0],rtol=1e-12)):
        #         raise AssertionError('z has size > 1 but all grid coords are identical!')
        
        if self.verbose: even_print('x_min', '%0.2f'%x.min())
        if self.verbose: even_print('x_max', '%0.2f'%x.max())
        if self.isRectilinear:
            if (self.nx>2):
                if self.verbose: even_print('dx begin : end', '%0.3E : %0.3E'%( (x[1]-x[0]), (x[-1]-x[-2]) ))
        if self.verbose: even_print('y_min', '%0.2f'%y.min())
        if self.verbose: even_print('y_max', '%0.2f'%y.max())
        if self.isRectilinear:
            if (self.ny>2):
                if self.verbose: even_print('dy begin : end', '%0.3E : %0.3E'%( (y[1]-y[0]), (y[-1]-y[-2]) ))
        if self.verbose: even_print('z_min', '%0.2f'%z.min())
        if self.verbose: even_print('z_max', '%0.2f'%z.max())
        if self.isRectilinear:
            if (self.nz>2):
                if self.verbose: even_print('dz begin : end', '%0.3E : %0.3E'%( (z[1]-z[0]), (z[-1]-z[-2]) ))
        if self.verbose: print(72*'-'+'\n')
        
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
            dset_path = 'Data/%s/ts_%06d/par_%06d'%(self.domainName,0,scalar_n_map[scalar])
            if (dset_path in self):
                self.scalars_dtypes.append(self[dset_path].dtype)
            else:
                #self.scalars_dtypes.append(np.float64)
                raise AssertionError('dset not found: %s'%dset_path)
        
        nt          = self['Kennsatz/TIMESTEP'].attrs['TIMESTEP_SIZE'][0] 
        gmode_time  = self['Kennsatz/TIMESTEP'].attrs['TIMESTEP_MODE'][0]
        
        ## a baseflow will not have a TIMEGRID
        if ('Kennsatz/TIMESTEP/TIMEGRID' in self):
            t = self['Kennsatz/TIMESTEP/TIMEGRID'][:]
        else:
            t = np.array( [0.] , dtype=np.float64 )
        
        if (gmode_time==EAS4_X0DX_G): ## =2 --> i.e. more than one timestep
            t = np.linspace(t[0],t[0]+t[1]*(nt - 1), nt  )
            gmode_time = EAS4_ALL_G
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
        
        # === attach to instance
        
        self.n_scalars    = n_scalars
        self.scalars      = scalars
        self.scalar_n_map = scalar_n_map
        self.t            = t
        self.dt           = dt
        self.nt           = nt
        self.duration     = duration
        
        self.ti           = np.arange(self.nt, dtype=np.float64)
        
        if self.verbose: print(72*'-'+'\n')
        
        # === udef dictionary attached to instance
        
        udef_char = [     'Ma',     'Re',     'Pr',      'kappa',    'R',    'p_inf',    'T_inf',    'C_Suth',    'S_Suth',    'mu_Suth_ref',    'T_Suth_ref' ]
        udef_real = [ self.Ma , self.Re , self.Pr ,  self.kappa, self.R, self.p_inf, self.T_inf, self.C_Suth, self.S_Suth, self.mu_Suth_ref, self.T_Suth_ref  ]
        self.udef = dict(zip(udef_char, udef_real))
        return
    
    # ===
    
    def get_mean(self, **kwargs):
        '''
        get spanwise mean of 2D EAS4 file
        '''
        axis = kwargs.get('axis',(2,))
        
        if (self.measType!='mean'):
            raise NotImplementedError('get_mean() not yet valid for measType=\'%s\''%self.measType)
        
        ## numpy structured array
        data_mean = np.zeros(shape=(self.nx,self.ny), dtype={'names':self.scalars, 'formats':self.scalars_dtypes})
        
        for si, scalar in enumerate(self.scalars):
            scalar_dtype = self.scalars_dtypes[si]
            dset_path = 'Data/%s/ts_%06d/par_%06d'%(self.domainName,0,self.scalar_n_map[scalar])
            data = np.copy(self[dset_path][()])
            ## perform np.mean() with float64 accumulator!
            scalar_mean = np.mean(data, axis=axis, dtype=np.float64).astype(scalar_dtype)
            data_mean[scalar] = scalar_mean
        
        return data_mean
    
    # === Paraview
    
    def make_xdmf(self, **kwargs):
        print('make_xdmf() not yet implemented for turbx class EAS4')
        raise NotImplementedError
        return

class ztmd(h5py.File):
    '''
    span (z) & temporal (t) mean data (md)
    -----
    --> mean_flow_mpi.eas
    --> favre_mean_flow_mpi.eas
    --> ext_rms_fluctuation_mpi.eas
    --> ext_favre_fluctuation_mpi.eas
    --> turbulent_budget_mpi.eas
    -----
    '''
    
    def __init__(self, *args, **kwargs):
        
        self.fname, self.open_mode = args
        
        self.fname_path = os.path.dirname(self.fname)
        self.fname_base = os.path.basename(self.fname)
        self.fname_root, self.fname_ext = os.path.splitext(self.fname_base)
        
        ## default to libver='latest' if none provided
        if ('libver' not in kwargs):
            kwargs['libver'] = 'latest'
        
        ## catch possible user error --> could prevent accidental EAS overwrites
        if (self.fname_ext=='.eas'):
            raise ValueError('EAS4 files should not be opened with turbx.ztmd()')
        
        ## mpio driver for ZTMD currently not supported
        if ('driver' in kwargs) and (kwargs['driver']=='mpio'):
            raise ValueError('ZTMD class is currently not set up to be used with MPI')
        
        ## determine if using mpi
        if ('driver' in kwargs) and (kwargs['driver']=='mpio'):
            self.usingmpi = True
        else:
            self.usingmpi = False
        
        ## determine communicator & rank info
        if self.usingmpi:
            self.comm    = kwargs['comm']
            self.n_ranks = self.comm.Get_size()
            self.rank    = self.comm.Get_rank()
        else:
            self.comm    = None
            self.n_ranks = 1
            self.rank    = 0
        
        ## determine MPI info / hints
        if self.usingmpi:
            if ('info' in kwargs):
                self.mpi_info = kwargs['info']
            else:
                mpi_info = MPI.Info.Create()
                ##
                mpi_info.Set('romio_ds_write' , 'disable'   )
                mpi_info.Set('romio_ds_read'  , 'disable'   )
                mpi_info.Set('romio_cb_read'  , 'automatic' )
                mpi_info.Set('romio_cb_write' , 'automatic' )
                #mpi_info.Set('romio_cb_read'  , 'enable' )
                #mpi_info.Set('romio_cb_write' , 'enable' )
                mpi_info.Set('cb_buffer_size' , str(int(round(8*1024**2))) ) ## 8 [MB]
                ##
                kwargs['info'] = mpi_info
                self.mpi_info = mpi_info
        
        if ('rdcc_nbytes' not in kwargs):
            kwargs['rdcc_nbytes'] = int(16*1024**2) ## 16 [MB]
        
        ## ztmd() unique kwargs (not h5py.File kwargs) --> pop() rather than get()
        verbose = kwargs.pop('verbose',False)
        force   = kwargs.pop('force',False)
        
        if (self.open_mode == 'w') and (force is False) and os.path.isfile(self.fname):
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
                                  
                                  >>> with ztmd(<<fname>>,'w',force=True) as f:
                                  >>>     ...
                                  '''
                print(textwrap.indent(textwrap.dedent(openModeInfoStr), 2*' ').strip('\n'))
                print(72*'-'+'\n')
            
            if (self.comm is not None):
                self.comm.Barrier()
            raise FileExistsError()
        
        ## remove file, touch, stripe
        elif (self.open_mode == 'w') and (force is True) and os.path.isfile(self.fname):
            if (self.rank==0):
                os.remove(self.fname)
                Path(self.fname).touch()
                if shutil.which('lfs') is not None:
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 8M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        ## touch, stripe
        elif (self.open_mode == 'w') and not os.path.isfile(self.fname):
            if (self.rank==0):
                Path(self.fname).touch()
                if shutil.which('lfs') is not None:
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 8M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        else:
            pass
        
        if (self.comm is not None):
            self.comm.Barrier()
        
        self.mod_avail_tqdm = ('tqdm' in sys.modules)
        
        ## call actual h5py.File.__init__()
        super(ztmd, self).__init__(*args, **kwargs)
        self.get_header(verbose=verbose)
    
    def __enter__(self):
        '''
        for use with python 'with' statement
        '''
        #return self
        return super(ztmd, self).__enter__()
    
    def __exit__(self, exception_type, exception_value, exception_traceback):
        '''
        for use with python 'with' statement
        '''
        if (self.rank==0):
            if exception_type is not None:
                print('\nsafely closed ZTMD HDF5 due to exception')
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
        return super(ztmd, self).__exit__()
    
    def get_header(self,**kwargs):
        '''
        initialize header attributes of ZTMD class instance
        '''
        
        verbose = kwargs.get('verbose',True)
        
        if (self.rank!=0):
            verbose=False
        
        # === udef (header vector dset based)
        
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
            
            rho_inf   = self.rho_inf   = self.p_inf/(self.R * self.T_inf)
            mu_inf    = self.mu_inf    = 14.58e-7*self.T_inf**1.5/(self.T_inf+110.4)
            nu_inf    = self.nu_inf    = self.mu_inf/self.rho_inf
            a_inf     = self.a_inf     = np.sqrt(self.kappa*self.R*self.T_inf)
            U_inf     = self.U_inf     = self.Ma*self.a_inf
            cp        = self.cp        = self.R*self.kappa/(self.kappa-1.)
            cv        = self.cv        = self.cp/self.kappa
            recov_fac = self.recov_fac = self.Pr**(1/3)
            Tw        = self.Tw        = self.T_inf
            Taw       = self.Taw       = self.T_inf + self.r*self.U_inf**2/(2*self.cp)
            lchar     = self.lchar     = self.Re*self.nu_inf/self.U_inf
            
            if verbose: print(72*'-')
            if verbose: even_print('rho_inf'         , '%0.3f [kg/m³]'    % self.rho_inf   )
            if verbose: even_print('mu_inf'          , '%0.6E [kg/(m·s)]' % self.mu_inf    )
            if verbose: even_print('nu_inf'          , '%0.6E [m²/s]'     % self.nu_inf    )
            if verbose: even_print('a_inf'           , '%0.6f [m/s]'      % self.a_inf     )
            if verbose: even_print('U_inf'           , '%0.6f [m/s]'      % self.U_inf     )
            if verbose: even_print('cp'              , '%0.3f [J/(kg·K)]' % self.cp        )
            if verbose: even_print('cv'              , '%0.3f [J/(kg·K)]' % self.cv        )
            if verbose: even_print('recovery factor' , '%0.6f [-]'        % self.recov_fac )
            if verbose: even_print('Tw'              , '%0.3f [K]'        % self.Tw        )
            if verbose: even_print('Taw'             , '%0.3f [K]'        % self.Taw       )
            if verbose: even_print('lchar'           , '%0.6E [m]'        % self.lchar     )
            if verbose: print(72*'-'+'\n')
            
            # === write the 'derived' udef variables to a dict attribute of the ZTMD instance
            udef_char_deriv = ['rho_inf', 'mu_inf', 'nu_inf', 'a_inf', 'U_inf', 'cp', 'cv', 'recov_fac', 'Tw', 'Taw', 'lchar']
            udef_real_deriv = [ rho_inf,   mu_inf,   nu_inf,   a_inf,   U_inf,   cp,   cv,   recov_fac,   Tw,   Taw,   lchar ]
            self.udef_deriv = dict(zip(udef_char_deriv, udef_real_deriv))
        
        else:
            #print("dset 'header' not in ZTMD")
            pass
        
        # === udef (attr based)
        
        header_attr_str_list = ['Ma','Re','Pr','kappa','R','p_inf','T_inf','C_Suth','S_Suth','mu_Suth_ref','T_Suth_ref']
        if all([ attr_str in self.attrs.keys() for attr_str in header_attr_str_list ]):
            header_attr_based = True
        else:
            header_attr_based = False
        
        if header_attr_based:
            
            ## set all attributes
            for attr_str in header_attr_str_list:
                setattr( self, attr_str, self.attrs[attr_str] )
            
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
            
            mu_inf_1 = 14.58e-7*self.T_inf**1.5/(self.T_inf+110.4)
            mu_inf_2 = self.mu_Suth_ref*(self.T_inf/self.T_Suth_ref)**(3/2)*(self.T_Suth_ref+self.S_Suth)/(self.T_inf+self.S_Suth)
            mu_inf_3 = self.C_Suth*self.T_inf**(3/2)/(self.T_inf+self.S_Suth)
            #print(mu_inf_1)
            #print(mu_inf_2)
            #print(mu_inf_3)
            
            rho_inf   = self.rho_inf   = self.p_inf/(self.R*self.T_inf)
            #mu_inf    = self.mu_inf    = 14.58e-7*self.T_inf**1.5/(self.T_inf+110.4)
            mu_inf    = self.mu_inf    = mu_inf_3
            nu_inf    = self.nu_inf    = self.mu_inf/self.rho_inf
            a_inf     = self.a_inf     = np.sqrt(self.kappa*self.R*self.T_inf)
            U_inf     = self.U_inf     = self.Ma*self.a_inf
            cp        = self.cp        = self.R*self.kappa/(self.kappa-1.)
            cv        = self.cv        = self.cp/self.kappa
            recov_fac = self.recov_fac = self.Pr**(1/3)
            Tw        = self.Tw        = self.T_inf
            Taw       = self.Taw       = self.T_inf + self.recov_fac*self.U_inf**2/(2*self.cp)
            lchar     = self.lchar     = self.Re*self.nu_inf/self.U_inf
            
            if verbose: print(72*'-')
            if verbose: even_print('rho_inf'         , '%0.3f [kg/m³]'    % self.rho_inf   )
            if verbose: even_print('mu_inf'          , '%0.6E [kg/(m·s)]' % self.mu_inf    )
            if verbose: even_print('nu_inf'          , '%0.6E [m²/s]'     % self.nu_inf    )
            if verbose: even_print('a_inf'           , '%0.6f [m/s]'      % self.a_inf     )
            if verbose: even_print('U_inf'           , '%0.6f [m/s]'      % self.U_inf     )
            if verbose: even_print('cp'              , '%0.3f [J/(kg·K)]' % self.cp        )
            if verbose: even_print('cv'              , '%0.3f [J/(kg·K)]' % self.cv        )
            if verbose: even_print('recovery factor' , '%0.6f [-]'        % self.recov_fac )
            if verbose: even_print('Tw'              , '%0.3f [K]'        % self.Tw        )
            if verbose: even_print('Taw'             , '%0.3f [K]'        % self.Taw       )
            if verbose: even_print('lchar'           , '%0.6E [m]'        % self.lchar     )
            if verbose: print(72*'-'+'\n')
            
            # === write the 'derived' udef variables to a dict attribute of the ZTMD instance
            udef_char_deriv = ['rho_inf', 'mu_inf', 'nu_inf', 'a_inf', 'U_inf', 'cp', 'cv', 'recov_fac', 'Tw', 'Taw', 'lchar']
            udef_real_deriv = [ rho_inf,   mu_inf,   nu_inf,   a_inf,   U_inf,   cp,   cv,   recov_fac,   Tw,   Taw,   lchar ]
            self.udef_deriv = dict(zip(udef_char_deriv, udef_real_deriv))
        
        if ('duration_avg' in self.attrs.keys()):
            self.duration_avg = self.attrs['duration_avg']
        if ('nx' in self.attrs.keys()):
            self.nx = self.attrs['nx']
        if ('ny' in self.attrs.keys()):
            self.ny = self.attrs['ny']
        if ('p_inf' in self.attrs.keys()):
            self.p_inf = self.attrs['p_inf']
        if ('lchar' in self.attrs.keys()):
            self.lchar = self.attrs['lchar']
        if ('U_inf' in self.attrs.keys()):
            self.U_inf = self.attrs['U_inf']
        if ('Re' in self.attrs.keys()):
            self.Re = self.attrs['Re']

        if ('T_inf' in self.attrs.keys()):
            self.T_inf = self.attrs['T_inf']
        if ('rho_inf' in self.attrs.keys()):
            self.rho_inf = self.attrs['rho_inf']
        
        if ('dims/x' in self):
            self.x = np.copy( self['dims/x'][()].T )
        if ('dims/y' in self):
            self.y = np.copy( self['dims/y'][()].T )
        if ('dims/t' in self):
            self.t = t = np.copy( self['dims/t'][()] )
        
        if verbose: print(72*'-')
        if verbose and hasattr(self,'duration_avg'): even_print('duration_avg', '%0.5f'%self.duration_avg)
        if verbose: even_print('nx', '%i'%self.nx)
        if verbose: even_print('ny', '%i'%self.ny)
        if verbose: print(72*'-')
        
        # === ts group names & scalars
        
        if ('data' in self):
            self.scalars = list(self['data'].keys())
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
    
    def compile_data(self, path, **kwargs):
        '''
        Copy data from 2D EAS4 containers (output from NS3D) to a ZTMD container
        -----
        
        The 'path' directory should contain one or more of the following files:
        
        --> mean_flow_mpi.eas
        --> favre_mean_flow_mpi.eas
        --> ext_rms_fluctuation_mpi.eas
        --> ext_favre_fluctuation_mpi.eas
        --> turbulent_budget_mpi.eas
        
        /dims : 2D dimension datasets (x,y,..) and possibly 1D dimension datasets (s_wall,..)
        /data : 2D datasets (u,uIuI,..)
        
        Datasets are dimensionalized to SI units upon import!
        
        /dimless : copy the dimless datasets as a reference
        
        Curvilinear cases may have the following additional HDF5 groups
        
        /data_1Dx : 1D datsets in streamwise (x/s1) direction (μ_wall,ρ_wall,u_τ,..)
        /csys     : coordinate system transformation arrays (projection vectors, transform tensors, etc.)
        /dims_2Dw : alternate grid (e.g. wall-normal projected grid)
        /data_2Dw : data interpolated on alternate grid
        
        '''
        
        verbose = kwargs.get('verbose',True)
        
        if verbose: print('\n'+'turbx.ztmd.compile_data()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        ## dz should be input as dimless (characteristic/inlet) (output from tgg)
        ## --> gets dimensionalized during this func!
        dz = kwargs.get('dz',None)
        nz = kwargs.get('nz',None)
        dt = kwargs.get('dt',None)
        
        path_ztmean = Path(path)
        if not path_ztmean.is_dir():
            raise FileNotFoundError('%s does not exist.'%str(path_ztmean))
        fn_Re_mean     = Path(path_ztmean, 'mean_flow_mpi.eas')
        fn_Fv_mean     = Path(path_ztmean, 'favre_mean_flow_mpi.eas')
        fn_Re_fluct    = Path(path_ztmean, 'ext_rms_fluctuation_mpi.eas')
        fn_Fv_fluct    = Path(path_ztmean, 'ext_favre_fluctuation_mpi.eas')
        fn_turb_budget = Path(path_ztmean, 'turbulent_budget_mpi.eas')
        
        self.attrs['fn_Re_mean']     = str( fn_Re_mean.relative_to(Path())     )
        self.attrs['fn_Fv_mean']     = str( fn_Fv_mean.relative_to(Path())     )
        self.attrs['fn_Re_fluct']    = str( fn_Re_fluct.relative_to(Path())    )
        self.attrs['fn_Fv_fluct']    = str( fn_Fv_fluct.relative_to(Path())    )
        self.attrs['fn_turb_budget'] = str( fn_turb_budget.relative_to(Path()) )
        
        ## the timestep is not known from the file
        if (dt is not None):
            self.attrs['dt'] = dt
        
        if (nz is not None):
            self.attrs['nz'] = nz
        
        if verbose:
            even_print('nz' , '%i'%nz )
            even_print('dz' , '%0.6e'%dz )
            even_print('dt' , '%0.6e'%dt )
            print(72*'-')
        
        # ===
        
        if fn_Re_mean.exists():
            print('--r-> %s'%fn_Re_mean.relative_to(Path()) )
            with eas4(str(fn_Re_mean),'r',verbose=False) as f1:
                
                ## the EAS4 data is still organized by rank in [z], so perform average across ranks
                data_mean = f1.get_mean()
                
                ## assert mean data shape
                for i, key in enumerate(data_mean.dtype.names):
                    if (data_mean[key].shape[0]!=f1.nx):
                        raise AssertionError('mean data dim1 shape != nx')
                    if (data_mean[key].shape[1]!=f1.ny):
                        raise AssertionError('mean data dim2 shape != ny')
                    if (data_mean[key].ndim!=2):
                        raise AssertionError('mean data ndim != 2')
                
                nx = f1.nx ; self.attrs['nx'] = nx
                ny = f1.ny ; self.attrs['ny'] = ny
                
                Ma          = f1.Ma          ; self.attrs['Ma']          = Ma
                Re          = f1.Re          ; self.attrs['Re']          = Re
                Pr          = f1.Pr          ; self.attrs['Pr']          = Pr
                T_inf       = f1.T_inf       ; self.attrs['T_inf']       = T_inf
                p_inf       = f1.p_inf       ; self.attrs['p_inf']       = p_inf
                kappa       = f1.kappa       ; self.attrs['kappa']       = kappa
                R           = f1.R           ; self.attrs['R']           = R
                mu_Suth_ref = f1.mu_Suth_ref ; self.attrs['mu_Suth_ref'] = mu_Suth_ref
                T_Suth_ref  = f1.T_Suth_ref  ; self.attrs['T_Suth_ref']  = T_Suth_ref
                C_Suth      = f1.C_Suth      ; self.attrs['C_Suth']      = C_Suth
                S_Suth      = f1.S_Suth      ; self.attrs['S_Suth']      = S_Suth
                
                rho_inf   = f1.rho_inf   ; self.attrs['rho_inf']   = rho_inf
                mu_inf    = f1.mu_inf    ; self.attrs['mu_inf']    = mu_inf
                nu_inf    = f1.nu_inf    ; self.attrs['nu_inf']    = nu_inf
                a_inf     = f1.a_inf     ; self.attrs['a_inf']     = a_inf
                U_inf     = f1.U_inf     ; self.attrs['U_inf']     = U_inf
                cp        = f1.cp        ; self.attrs['cp']        = cp
                cv        = f1.cv        ; self.attrs['cv']        = cv
                recov_fac = f1.recov_fac ; self.attrs['recov_fac'] = recov_fac
                Tw        = f1.Tw        ; self.attrs['Tw']        = Tw
                Taw       = f1.Taw       ; self.attrs['Taw']       = Taw
                lchar     = f1.lchar     ; self.attrs['lchar']     = lchar
                
                tchar = f1.lchar/f1.U_inf ; self.attrs['tchar'] = tchar
                
                ## duration over which avg was performed, iteration count and the sampling period of the avg
                Re_mean_total_avg_time       = f1.total_avg_time * tchar
                Re_mean_total_avg_iter_count = f1.total_avg_iter_count
                Re_mean_dt                   = Re_mean_total_avg_time / Re_mean_total_avg_iter_count
                
                self.attrs['Re_mean_total_avg_time'] = Re_mean_total_avg_time
                self.attrs['Re_mean_total_avg_iter_count'] = Re_mean_total_avg_iter_count
                self.attrs['Re_mean_dt'] = Re_mean_dt
                
                t_meas = f1.total_avg_time * (f1.lchar/f1.U_inf) ## dimensional [s]
                #t_meas = f1.total_avg_time ## dimless (char)
                self.attrs['t_meas'] = t_meas
                dset = self.create_dataset('dims/t', data=np.array([t_meas],dtype=np.float64), chunks=None)
                
                # ===
                
                x = np.copy(f1.x) ## from EAS4, dimless (char)
                y = np.copy(f1.y)
                
                if True: ## check against tgg data file
                    
                    fn_dat = '../tgg/wall_distance.dat'
                    
                    if not os.path.isfile(fn_dat):
                        raise FileNotFoundError('file does not exist: %s'%str(fn_dat))
                    
                    with open(fn_dat,'rb') as f:
                        d_ = pickle.load(f)
                        xy2d_tmp = d_['xy2d']
                        np.testing.assert_allclose(xy2d_tmp[:,:,0], x[:,:,0], rtol=1e-14, atol=1e-14)
                        np.testing.assert_allclose(xy2d_tmp[:,:,1], y[:,:,0], rtol=1e-14, atol=1e-14)
                        if verbose: even_print('check passed' , 'x grid' )
                        if verbose: even_print('check passed' , 'y grid' )
                        d_ = None; del d_
                        xy2d_tmp = None; del xy2d_tmp
                
                if (f1.x.ndim==1) and (f1.y.ndim==1): ## rectilinear in [x,y]
                    
                    sys.exit('105019274029 --> has not yet been updated')
                    
                    xxs, yys = np.meshgrid(f1.x, f1.y, indexing='ij') ; data['xxs'] = xxs ; data['yys'] = yys ## dimensionless (inlet)
                    xx,  yy  = np.meshgrid(x,    y,    indexing='ij') ; data['xx']  = xx  ; data['yy']  = yy  ## dimensional
                    
                    dx = np.insert(np.diff(x,n=1), 0, 0.) ; data['dx'] = dx ## 1D Δx
                    dy = np.insert(np.diff(y,n=1), 0, 0.) ; data['dy'] = dy
                    
                    if dz is not None: ## dimensionalize
                        dz = dz * f1.lchar ; data['dz'] = dz ## 0D (float)
                    
                    np.testing.assert_allclose(np.cumsum(dx), x, rtol=1e-8) ## confirm 1D Δx calc
                    np.testing.assert_allclose(np.cumsum(dy), y, rtol=1e-8)
                    
                    dxx = np.broadcast_to(dx, (ny,nx)).T ; data['dxx'] = dxx ## 2D Δx
                    dyy = np.broadcast_to(dy, (nx,ny))   ; data['dyy'] = dyy
                    
                    ## if dz is not None:
                    ##     dzz = dz * np.ones((nx,ny), dtype=np.float64) ## 2D but all == dz
                    
                    ## dxx_ = np.concatenate([np.zeros((1,ny)), np.diff(xx,axis=0)], axis=0)
                    ## dyy_ = np.concatenate([np.zeros((nx,1)), np.diff(yy,axis=1)], axis=1)
                    ## np.testing.assert_allclose(dxx, dxx_, rtol=1e-8)
                    ## np.testing.assert_allclose(dyy, dyy_, rtol=1e-8)
                    
                    pass
                
                elif (f1.x.ndim==3) and (f1.y.ndim==3): ## curvilinear in [x,y]
                    
                    ## confirm that x,y coords are same in [z] direction
                    np.testing.assert_allclose( x[-1,-1,:] , x[-1,-1,0] , rtol=1e-14 , atol=1e-14 )
                    np.testing.assert_allclose( y[-1,-1,:] , y[-1,-1,0] , rtol=1e-14 , atol=1e-14 )
                    
                    ## take only 1 layer in [z]
                    x = np.squeeze( np.copy( x[:,:,0] ) ) ## dimless (char)
                    y = np.squeeze( np.copy( y[:,:,0] ) )
                    
                    # ## backup non-dimensional coordinate arrays
                    # dset = self.create_dataset('/dimless/dims/x', data=x.T, chunks=None)
                    # dset = self.create_dataset('/dimless/dims/y', data=y.T, chunks=None)
                    
                    ## dimensionalize & write
                    #x = np.copy( lchar * x )
                    #y = np.copy( lchar * y )
                    x *= lchar
                    y *= lchar
                    
                    self.create_dataset('dims/x', data=x.T, chunks=None, dtype=np.float64)
                    self.create_dataset('dims/y', data=y.T, chunks=None, dtype=np.float64)
                    
                    self.attrs['nx'] = nx
                    self.attrs['ny'] = ny
                    #self.attrs['nz'] = 1 ## NO
                    if verbose:
                        even_print('nx' , '%i'%nx )
                        even_print('ny' , '%i'%ny )
                
                else:
                    raise ValueError('case x.ndim=%i , y.ndim=%i not yet accounted for'%(f1.x.ndim,f1.y.ndim))
                
                # === redimensionalize quantities (by sim characteristic quantities)
                
                u   = np.copy( data_mean['u']   ) * U_inf 
                v   = np.copy( data_mean['v']   ) * U_inf
                w   = np.copy( data_mean['w']   ) * U_inf
                rho = np.copy( data_mean['rho'] ) * rho_inf
                p   = np.copy( data_mean['p']   ) * (rho_inf * U_inf**2)
                T   = np.copy( data_mean['T']   ) * T_inf
                mu  = np.copy( data_mean['mu']  ) * mu_inf
                data_mean = None; del data_mean
                
                a     = np.sqrt( kappa * R * T )
                nu    = mu / rho
                umag  = np.sqrt( u**2 + v**2 + w**2 )
                M     = umag / np.sqrt(kappa * R * T)
                mflux = umag * rho
                
                ## base scalars [u,v,w,ρ,p,T]
                dset = self.create_dataset('data/u'   , data=u.T   , chunks=None)
                dset = self.create_dataset('data/v'   , data=v.T   , chunks=None)
                dset = self.create_dataset('data/w'   , data=w.T   , chunks=None)
                dset = self.create_dataset('data/rho' , data=rho.T , chunks=None)
                dset = self.create_dataset('data/p'   , data=p.T   , chunks=None)
                dset = self.create_dataset('data/T'   , data=T.T   , chunks=None)
                ##
                dset = self.create_dataset('data/a'     , data=a.T     , chunks=None) #; dset.attrs['dimensional'] = True  ; dset.attrs['unit'] = '[m/s]'
                dset = self.create_dataset('data/mu'    , data=mu.T    , chunks=None) #; dset.attrs['dimensional'] = True  ; dset.attrs['unit'] = '[kg/(m·s)]'
                dset = self.create_dataset('data/nu'    , data=nu.T    , chunks=None) #; dset.attrs['dimensional'] = True  ; dset.attrs['unit'] = '[m²/s]'
                dset = self.create_dataset('data/umag'  , data=umag.T  , chunks=None) #; dset.attrs['dimensional'] = True  ; dset.attrs['unit'] = '[m/s]'
                dset = self.create_dataset('data/M'     , data=M.T     , chunks=None) #; dset.attrs['dimensional'] = False
                dset = self.create_dataset('data/mflux' , data=mflux.T , chunks=None) #; dset.attrs['dimensional'] = True  ; dset.attrs['unit'] = '[kg/(m²·s)]'
        
        if fn_Re_fluct.exists():
            print('--r-> %s'%fn_Re_fluct.relative_to(Path()) )
            with eas4(str(fn_Re_fluct),'r',verbose=False) as f1:
                
                data_mean = f1.get_mean()
                
                ## assert mean data shape
                for i, key in enumerate(data_mean.dtype.names):
                    if (data_mean[key].shape[0]!=f1.nx):
                        raise AssertionError('mean data dim1 shape != nx')
                    if (data_mean[key].shape[1]!=f1.ny):
                        raise AssertionError('mean data dim2 shape != ny')
                    if (data_mean[key].ndim!=2):
                        raise AssertionError('mean data ndim != 2')
                
                Re_fluct_total_avg_time       = f1.total_avg_time
                Re_fluct_total_avg_iter_count = f1.total_avg_iter_count
                Re_fluct_dt                   = Re_fluct_total_avg_time/Re_fluct_total_avg_iter_count
                
                self.attrs['Re_fluct_total_avg_time'] = Re_fluct_total_avg_time
                self.attrs['Re_fluct_total_avg_iter_count'] = Re_fluct_total_avg_iter_count
                self.attrs['Re_fluct_dt'] = Re_fluct_dt
                
                uI_uI = data_mean["u'u'"] * U_inf**2
                vI_vI = data_mean["v'v'"] * U_inf**2
                wI_wI = data_mean["w'w'"] * U_inf**2
                uI_vI = data_mean["u'v'"] * U_inf**2
                uI_wI = data_mean["u'w'"] * U_inf**2
                vI_wI = data_mean["v'w'"] * U_inf**2
                ##
                self.create_dataset('data/uI_uI', data=uI_uI.T, chunks=None)
                self.create_dataset('data/vI_vI', data=vI_vI.T, chunks=None)
                self.create_dataset('data/wI_wI', data=wI_wI.T, chunks=None)
                self.create_dataset('data/uI_vI', data=uI_vI.T, chunks=None)
                self.create_dataset('data/uI_wI', data=uI_wI.T, chunks=None)
                self.create_dataset('data/vI_wI', data=vI_wI.T, chunks=None)
                
                uI_TI = data_mean["u'T'"] * (U_inf*T_inf)
                vI_TI = data_mean["v'T'"] * (U_inf*T_inf)
                wI_TI = data_mean["w'T'"] * (U_inf*T_inf)
                ##
                self.create_dataset('data/uI_TI', data=uI_TI.T, chunks=None)
                self.create_dataset('data/vI_TI', data=vI_TI.T, chunks=None)
                self.create_dataset('data/wI_TI', data=wI_TI.T, chunks=None)
                
                TI_TI = data_mean["T'T'"] * T_inf**2
                pI_pI = data_mean["p'p'"] * (rho_inf * U_inf**2)**2
                rI_rI = data_mean["r'r'"] * rho_inf**2
                muI_muI = data_mean["mu'mu'"] * mu_inf**2
                ##
                self.create_dataset('data/TI_TI',   data=TI_TI.T,   chunks=None)
                self.create_dataset('data/pI_pI',   data=pI_pI.T,   chunks=None)
                self.create_dataset('data/rI_rI',   data=rI_rI.T,   chunks=None)
                self.create_dataset('data/muI_muI', data=muI_muI.T, chunks=None)
                
                tauI_xx = data_mean["tau'_xx"] * mu_inf * U_inf / lchar
                tauI_yy = data_mean["tau'_yy"] * mu_inf * U_inf / lchar
                tauI_zz = data_mean["tau'_zz"] * mu_inf * U_inf / lchar
                tauI_xy = data_mean["tau'_xy"] * mu_inf * U_inf / lchar
                tauI_xz = data_mean["tau'_xz"] * mu_inf * U_inf / lchar
                tauI_yz = data_mean["tau'_yz"] * mu_inf * U_inf / lchar
                ##
                self.create_dataset('data/tauI_xx', data=tauI_xx.T, chunks=None)
                self.create_dataset('data/tauI_yy', data=tauI_yy.T, chunks=None)
                self.create_dataset('data/tauI_zz', data=tauI_zz.T, chunks=None)
                self.create_dataset('data/tauI_xy', data=tauI_xy.T, chunks=None)
                self.create_dataset('data/tauI_xz', data=tauI_xz.T, chunks=None)
                self.create_dataset('data/tauI_yz', data=tauI_yz.T, chunks=None)
                
                # === RMS values
                
                if True: ## dimensional
                    
                    uI_uI_rms = np.sqrt(       data_mean["u'u'"]  * U_inf**2 )
                    vI_vI_rms = np.sqrt(       data_mean["v'v'"]  * U_inf**2 )
                    wI_wI_rms = np.sqrt(       data_mean["w'w'"]  * U_inf**2 )
                    uI_vI_rms = np.sqrt(np.abs(data_mean["u'v'"]) * U_inf**2 ) * np.sign(data_mean["u'v'"]) 
                    uI_wI_rms = np.sqrt(np.abs(data_mean["u'w'"]) * U_inf**2 ) * np.sign(data_mean["u'w'"])
                    vI_wI_rms = np.sqrt(np.abs(data_mean["v'w'"]) * U_inf**2 ) * np.sign(data_mean["v'w'"])
                    
                    uI_TI_rms = np.sqrt(np.abs(data_mean["u'T'"]) * U_inf*T_inf) * np.sign(data_mean["u'T'"])
                    vI_TI_rms = np.sqrt(np.abs(data_mean["v'T'"]) * U_inf*T_inf) * np.sign(data_mean["v'T'"])
                    wI_TI_rms = np.sqrt(np.abs(data_mean["w'T'"]) * U_inf*T_inf) * np.sign(data_mean["w'T'"])
                    
                    rI_rI_rms   = np.sqrt( data_mean["r'r'"]   * rho_inf**2              )
                    TI_TI_rms   = np.sqrt( data_mean["T'T'"]   * T_inf**2                )
                    pI_pI_rms   = np.sqrt( data_mean["p'p'"]   * (rho_inf * U_inf**2)**2 )
                    muI_muI_rms = np.sqrt( data_mean["mu'mu'"] * mu_inf**2               )
                    
                    M_rms = uI_uI_rms / np.sqrt(kappa * R * T)
                
                # if False: ## dimless
                #     
                #     uI_uI_rms = np.sqrt(        data_mean["u'u'"]  )
                #     vI_vI_rms = np.sqrt(        data_mean["v'v'"]  )
                #     wI_wI_rms = np.sqrt(        data_mean["w'w'"]  )
                #     uI_vI_rms = np.sqrt( np.abs(data_mean["u'v'"]) ) * np.sign(data_mean["u'v'"]) 
                #     uI_wI_rms = np.sqrt( np.abs(data_mean["u'w'"]) ) * np.sign(data_mean["u'w'"])
                #     vI_wI_rms = np.sqrt( np.abs(data_mean["v'w'"]) ) * np.sign(data_mean["v'w'"])
                #     
                #     uI_TI_rms = np.sqrt( np.abs(data_mean["u'T'"]) ) * np.sign(data_mean["u'T'"])
                #     vI_TI_rms = np.sqrt( np.abs(data_mean["v'T'"]) ) * np.sign(data_mean["v'T'"])
                #     wI_TI_rms = np.sqrt( np.abs(data_mean["w'T'"]) ) * np.sign(data_mean["w'T'"])
                #     
                #     rI_rI_rms   = np.sqrt( data_mean["r'r'"]   )
                #     TI_TI_rms   = np.sqrt( data_mean["T'T'"]   )
                #     pI_pI_rms   = np.sqrt( data_mean["p'p'"]   )
                #     muI_muI_rms = np.sqrt( data_mean["mu'mu'"] )
                #     
                #     # ...
                #     M_rms = np.sqrt( data_mean["u'u'"] * U_inf**2 ) / np.sqrt(kappa * R * (T*T_inf) )
                
                self.create_dataset('data/M_rms', data=M_rms.T, chunks=None)
        
        '''
        TODO: add Favre mean, fluct
        '''
        
        if fn_turb_budget.exists():
            print('--r-> %s'%fn_turb_budget.relative_to(Path()) )
            with eas4(str(fn_turb_budget),'r',verbose=False) as f1:
                
                data_mean = f1.get_mean() ## numpy structured array
                
                ## assert mean data shape
                for i, key in enumerate(data_mean.dtype.names):
                    if (data_mean[key].shape[0]!=f1.nx):
                        raise AssertionError('mean data dim1 shape != nx')
                    if (data_mean[key].shape[1]!=f1.ny):
                        raise AssertionError('mean data dim2 shape != ny')
                    if (data_mean[key].ndim!=2):
                        raise AssertionError('mean data ndim != 2')
                
                turb_budget_total_avg_time       = f1.total_avg_time
                turb_budget_total_avg_iter_count = f1.total_avg_iter_count
                turb_budget_dt                   = turb_budget_total_avg_time/turb_budget_total_avg_iter_count
                
                self.attrs['turb_budget_total_avg_time'] = turb_budget_total_avg_time
                self.attrs['turb_budget_total_avg_iter_count'] = turb_budget_total_avg_iter_count
                self.attrs['turb_budget_dt'] = turb_budget_dt
                
                production     = data_mean['prod.']     * U_inf**3 * rho_inf / lchar
                dissipation    = data_mean['dis.']      * U_inf**2 * mu_inf  / lchar**2
                turb_transport = data_mean['t-transp.'] * U_inf**3 * rho_inf / lchar
                visc_diffusion = data_mean['v-diff.']   * U_inf**2 * mu_inf  / lchar**2
                p_diffusion    = data_mean['p-diff.']   * U_inf**3 * rho_inf / lchar
                p_dilatation   = data_mean['p-dilat.']  * U_inf**3 * rho_inf / lchar
                rho_terms      = data_mean['rho-terms'] * U_inf**3 * rho_inf / lchar
                ##
                dset = self.create_dataset('data/production'     , data=production.T     , chunks=None) #; dset.attrs['dimensional'] = True
                dset = self.create_dataset('data/dissipation'    , data=dissipation.T    , chunks=None) #; dset.attrs['dimensional'] = True
                dset = self.create_dataset('data/turb_transport' , data=turb_transport.T , chunks=None) #; dset.attrs['dimensional'] = True
                dset = self.create_dataset('data/visc_diffusion' , data=visc_diffusion.T , chunks=None) #; dset.attrs['dimensional'] = True
                dset = self.create_dataset('data/p_diffusion'    , data=p_diffusion.T    , chunks=None) #; dset.attrs['dimensional'] = True
                dset = self.create_dataset('data/p_dilatation'   , data=p_dilatation.T   , chunks=None) #; dset.attrs['dimensional'] = True
                dset = self.create_dataset('data/rho_terms'      , data=rho_terms.T      , chunks=None) #; dset.attrs['dimensional'] = True
                
                if 'dissipation' in locals():
                    
                    #if not self.get('data/nu').attrs['dimensional']:
                    #    raise ValueError('nu is not dimensional')
                    
                    Kolm_len = (nu**3 / np.abs(dissipation))**(1/4)
                    self.create_dataset('data/Kolm_len', data=Kolm_len.T, chunks=None); dset.attrs['dimensional'] = True
        
        # ===
        
        self.get_header(verbose=True)
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : turbx.ztmd.compile_data() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    # === concave TBL (CTBL) specific
    
    def add_s_wall(self,**kwargs):
        '''
        add wall/top path length (numerically integrated, not continuous!)
        '''
        
        verbose = kwargs.get('verbose',True)
        
        dsw_   = np.sqrt( np.diff(self.x[:,0])**2 + np.diff(self.y[:,0])**2 )
        s_wall = np.cumsum(np.concatenate([[0.],dsw_]))
        
        dst_   = np.sqrt( np.diff(self.x[:,-1])**2 + np.diff(self.y[:,-1])**2 )
        s_top  = np.cumsum(np.concatenate([[0.],dst_]))
        
        if verbose: even_print('s_wall/lchar max','%0.4f'%(s_wall.max()/self.lchar))
        if verbose: even_print('s_top/lchar max','%0.4f'%(s_top.max()/self.lchar))
        
        if ('dims/s_wall' in self):
            del self['dims/s_wall']
        self.create_dataset('dims/s_wall', data=s_wall, chunks=None)
        
        if ('dims/s_top' in self):
            del self['dims/s_top']
        self.create_dataset('dims/s_top', data=s_top, chunks=None)
        
        return
    
    def add_wall_tang_norm_vecs(self, fn_dat=None, **kwargs):
        '''
        add data from pre-processor (turbulent grid generator / tgg) .dat file:
        - wall distance       --> shape (nx,ny)
        - wall normal vector  --> shape (nx,ny,2)
        - wall tangent vector --> shape (nx,ny,2)
        '''
        
        verbose = kwargs.get('verbose',True)
        
        # ## get baseflow profile
        # sys.path.append('../tgg')
        # import tgg
        # from tgg import grid_generator_tbl_concave, freestream_parameters, potential_flow_sector, piecewise_radial_tangent_curve
        #
        # fn_tgg = Path('../tgg/','gg.dat')
        # if verbose: print('--r-> %s'%str(fn_tgg))
        # with open(fn_tgg,'rb') as f:
        #     gg = pickle.load(f)
        # 
        # lchar_ref = gg.fsp.lchar
        #
        # rdiff = ( self.lchar - lchar_ref ) / self.lchar
        # if verbose: even_print('rdiff lchar', '%0.5e'%rdiff)
        
        if not os.path.isfile(fn_dat):
            raise FileNotFoundError('file does not exist: %s'%str(fn_dat))
        
        ## open data file from tgg
        with open(fn_dat,'rb') as f:
            
            # wall_distance, v_tang, v_norm = pickle.load(f)
            
            d_ = pickle.load(f)
            wall_distance = d_['wall_distance']
            v_tang        = d_['v_tang']
            v_norm        = d_['v_norm']
            s_wall_2d     = d_['s_wall_2d'] ## curve path length of point on wall (nx,ny)
            p_wall_2d     = d_['p_wall_2d'] ## projection point on wall (nx,ny,2)
            xy2d_tmp      = d_['xy2d']
            ##
            np.testing.assert_allclose(xy2d_tmp[:,:,0], self.x/self.lchar, rtol=1e-14, atol=1e-14)
            np.testing.assert_allclose(xy2d_tmp[:,:,1], self.y/self.lchar, rtol=1e-14, atol=1e-14)
            if verbose: even_print('check passed', 'x grid')
            if verbose: even_print('check passed', 'y grid')
            ##
            d_ = None; del d_
            xy2d_tmp = None; del xy2d_tmp
        
        ## re-dimensionalize
        wall_distance *= self.lchar
        
        if not (wall_distance.shape == (self.nx,self.ny)):
            raise ValueError('wall_distance.shape != (self.nx,self.ny)')
        if not (v_tang.shape == (self.nx,self.ny,2)):
            raise ValueError('v_tang.shape != (self.nx,self.ny,2)')
        if not (v_norm.shape == (self.nx,self.ny,2)):
            raise ValueError('v_norm.shape != (self.nx,self.ny,2)')
        
        ## write wall distance scalar (nx,ny)
        if ('data/wall_distance' in self): del self['data/wall_distance']
        self.create_dataset('data/wall_distance', data=wall_distance.T, chunks=None)
        
        ## write wall normal / tangent basis vectors
        if ('csys/v_tang' in self): del self['csys/v_tang']
        self.create_dataset('csys/v_tang', data=v_tang, chunks=None)
        if ('csys/v_norm' in self): del self['csys/v_norm']
        self.create_dataset('csys/v_norm', data=v_norm, chunks=None)
        
        ## write continuous wall point coordinate & wall path length
        if ('csys/s_wall_2d' in self): del self['csys/s_wall_2d']
        self.create_dataset('csys/s_wall_2d', data=s_wall_2d, chunks=None)
        if ('csys/p_wall_2d' in self): del self['csys/p_wall_2d']
        self.create_dataset('csys/p_wall_2d', data=p_wall_2d, chunks=None)
        
        return
    
    def calc_velocity_tang_norm_wall(self, **kwargs):
        '''
        calculate u_tang & u_norm
        '''
        
        verbose = kwargs.get('verbose',True)
        
        if not ('data/wall_distance' in self):
            raise ValueError('data/wall_distance not in hdf5')
        if not ('data/u' in self):
            raise ValueError('data/u not in hdf5')
        if not ('data/v' in self):
            raise ValueError('data/v not in hdf5')
        if not ('csys/v_tang' in self):
            raise ValueError('csys/v_tang not in hdf5')
        if not ('csys/v_norm' in self):
            raise ValueError('csys/v_norm not in hdf5')
        if not (self.open_mode=='a') or (self.open_mode=='w'):
            raise ValueError('not able to write to hdf5 file')
        
        ## read 2D velocities
        u  = np.copy( self['data/u'][()].T )
        v  = np.copy( self['data/v'][()].T )
        uv = np.stack((u,v), axis=-1)
        
        ## read unit vectors (wall tangent, wall norm) from HDF5
        v_tang = np.copy( self['csys/v_tang'][()] )
        v_norm = np.copy( self['csys/v_norm'][()] )
        
        ## inner product of velocity vector and basis vector (csys transform)
        u_tang = np.einsum('xyi,xyi->xy', v_tang, uv)
        u_norm = np.einsum('xyi,xyi->xy', v_norm, uv)
        
        # if self.get('data/u').attrs['dimensional']:
        #     raise AssertionError('u is dimensional')
        # if self.get('data/v').attrs['dimensional']:
        #     raise AssertionError('v is dimensional')
        
        if ('data/u_tang' in self): del self['data/u_tang']
        dset = self.create_dataset('data/u_tang', data=u_tang.T, chunks=None)
        #dset.attrs['dimensional'] = False
        if verbose: even_print('u tangent','%s'%str(u_tang.shape))
        
        if ('data/u_norm' in self): del self['data/u_norm']
        dset = self.create_dataset('data/u_norm', data=u_norm.T, chunks=None)
        #dset.attrs['dimensional'] = False
        if verbose: even_print('u normal','%s'%str(u_norm.shape))
        
        return
    
    def calc_gradients_crv(self, acc=6, edge_stencil='full', **kwargs):
        '''
        calculate spatial gradients of averaged quantities
        in curvilinear coordiantes
        '''
        
        verbose = kwargs.get('verbose',True)
        
        if (self.x.ndim!=2):
            raise ValueError('x.ndim!=2')
        
        # # === get csys unit vectors
        # if ('csys/v_tang' in self) and ('csys/v_norm' in self):
        #     v_tang = np.copy( self['csys/v_tang'][()] )
        #     v_norm = np.copy( self['csys/v_norm'][()] )
        #     #wall_trafo_mat = np.stack((v_tang,v_norm), axis=-1)
        
        # === get metric tensor 2D
        
        M = get_metric_tensor_2d(self.x, self.y, acc=acc, edge_stencil=edge_stencil, verbose=False)
        
        ddx_q1 = np.copy( M[:,:,0,0] ) ## ξ_x
        ddx_q2 = np.copy( M[:,:,1,0] ) ## η_x
        ddy_q1 = np.copy( M[:,:,0,1] ) ## ξ_y
        ddy_q2 = np.copy( M[:,:,1,1] ) ## η_y
        
        if verbose: even_print('ξ_x','%s'%str(ddx_q1.shape))
        if verbose: even_print('η_x','%s'%str(ddx_q2.shape))
        if verbose: even_print('ξ_y','%s'%str(ddy_q1.shape))
        if verbose: even_print('η_y','%s'%str(ddy_q2.shape))
        
        M = None; del M
        
        ## the 'computational' grid (unit Cartesian)
        #x_comp = np.arange(nx, dtype=np.float64)
        #y_comp = np.arange(ny, dtype=np.float64)
        x_comp = 1.
        y_comp = 1.
        
        # === get gradients of [u,v,p,T,ρ]
        
        if ('data/u' in self):
            
            u          = np.copy( self['data/u'][()].T )
            ddx_u_comp = gradient(u, x_comp, axis=0, acc=acc, d=1)
            ddy_u_comp = gradient(u, y_comp, axis=1, acc=acc, d=1)
            ddx_u      = ddx_u_comp*ddx_q1 + ddy_u_comp*ddx_q2
            ddy_u      = ddx_u_comp*ddy_q1 + ddy_u_comp*ddy_q2
            
            if ('data/ddx_u' in self): del self['data/ddx_u']
            self.create_dataset('data/ddx_u', data=ddx_u.T, chunks=None)
            
            if ('data/ddy_u' in self): del self['data/ddy_u']
            self.create_dataset('data/ddy_u', data=ddy_u.T, chunks=None)
            
            if verbose: even_print('ddx[u]','%s'%str(ddx_u.shape))
            if verbose: even_print('ddy[u]','%s'%str(ddy_u.shape))
        
        if ('data/v' in self):
            
            v          = np.copy( self['data/v'][()].T )
            ddx_v_comp = gradient(v, x_comp, axis=0, acc=acc, d=1)
            ddy_v_comp = gradient(v, y_comp, axis=1, acc=acc, d=1)
            ddx_v      = ddx_v_comp*ddx_q1 + ddy_v_comp*ddx_q2
            ddy_v      = ddx_v_comp*ddy_q1 + ddy_v_comp*ddy_q2
            
            if ('data/ddx_v' in self): del self['data/ddx_v']
            dset = self.create_dataset('data/ddx_v', data=ddx_v.T, chunks=None)
            
            if ('data/ddy_v' in self): del self['data/ddy_v']
            dset = self.create_dataset('data/ddy_v', data=ddy_v.T, chunks=None)
            
            if verbose: even_print('ddx[v]','%s'%str(ddx_v.shape))
            if verbose: even_print('ddy[v]','%s'%str(ddy_v.shape))
        
        if ('data/p' in self):
            
            p          = np.copy( self['data/p'][()].T )
            ddx_p_comp = gradient(p, x_comp, axis=0, acc=acc, d=1)
            ddy_p_comp = gradient(p, y_comp, axis=1, acc=acc, d=1)
            ddx_p      = ddx_p_comp*ddx_q1 + ddy_p_comp*ddx_q2
            ddy_p      = ddx_p_comp*ddy_q1 + ddy_p_comp*ddy_q2
            
            if ('data/ddx_p' in self): del self['data/ddx_p']
            dset = self.create_dataset('data/ddx_p', data=ddx_p.T, chunks=None)
            
            if ('data/ddy_p' in self): del self['data/ddy_p']
            dset = self.create_dataset('data/ddy_p', data=ddy_p.T, chunks=None)
            
            if verbose: even_print('ddx[p]','%s'%str(ddx_p.shape))
            if verbose: even_print('ddy[p]','%s'%str(ddy_p.shape))
        
        if ('data/T' in self):
            
            T          = np.copy( self['data/T'][()].T )
            ddx_T_comp = gradient(T, x_comp, axis=0, acc=acc, d=1)
            ddy_T_comp = gradient(T, y_comp, axis=1, acc=acc, d=1)
            ddx_T      = ddx_T_comp*ddx_q1 + ddy_T_comp*ddx_q2
            ddy_T      = ddx_T_comp*ddy_q1 + ddy_T_comp*ddy_q2
            
            if ('data/ddx_T' in self): del self['data/ddx_T']
            dset = self.create_dataset('data/ddx_T', data=ddx_T.T, chunks=None)
            
            if ('data/ddy_T' in self): del self['data/ddy_T']
            dset = self.create_dataset('data/ddy_T', data=ddy_T.T, chunks=None)
            
            if verbose: even_print('ddx[T]','%s'%str(ddx_T.shape))
            if verbose: even_print('ddy[T]','%s'%str(ddy_T.shape))
        
        if ('data/rho' in self):
            
            r          = np.copy( self['data/rho'][()].T )
            ddx_r_comp = gradient(r, x_comp, axis=0, acc=acc, d=1)
            ddy_r_comp = gradient(r, y_comp, axis=1, acc=acc, d=1)
            ddx_r      = ddx_r_comp*ddx_q1 + ddy_r_comp*ddx_q2
            ddy_r      = ddx_r_comp*ddy_q1 + ddy_r_comp*ddy_q2
            
            if ('data/ddx_r' in self): del self['data/ddx_r']
            dset = self.create_dataset('data/ddx_r', data=ddx_r.T, chunks=None)
            
            if ('data/ddy_r' in self): del self['data/ddy_r']
            dset = self.create_dataset('data/ddy_r', data=ddy_r.T, chunks=None)
            
            if verbose: even_print('ddx[ρ]','%s'%str(ddx_r.shape))
            if verbose: even_print('ddy[ρ]','%s'%str(ddy_r.shape))
        
        # === vorticity
        
        ## z-vorticity :: ω_z
        vort_z = ddx_v - ddy_u
        
        if ('data/vort_z' in self): del self['data/vort_z']
        self.create_dataset('data/vort_z', data=vort_z.T, chunks=None)
        if verbose: even_print('ω_z','%s'%str(vort_z.shape))
        
        ## divergence (in xy-plane)
        div_xy = ddx_u + ddy_v
        
        if ('data/div_xy' in self): del self['data/div_xy']
        self.create_dataset('data/div_xy', data=div_xy.T, chunks=None)
        if verbose: even_print('div_xy','%s'%str(div_xy.shape))
        
        # === 
        
        if ('data/u_tang' in self) and ('data/u_norm' in self):
            
            u_tang          = np.copy( self['data/u_tang'][()].T )
            ddx_u_tang_comp = gradient(u_tang, x_comp, axis=0, acc=acc, edge_stencil=edge_stencil, d=1)
            ddy_u_tang_comp = gradient(u_tang, y_comp, axis=1, acc=acc, edge_stencil=edge_stencil, d=1)
            ddx_u_tang      = ddx_u_tang_comp*ddx_q1 + ddy_u_tang_comp*ddx_q2
            ddy_u_tang      = ddx_u_tang_comp*ddy_q1 + ddy_u_tang_comp*ddy_q2
            
            if ('data/ddx_u_tang' in self): del self['data/ddx_u_tang']
            dset = self.create_dataset('data/ddx_u_tang', data=ddx_u_tang.T, chunks=None)
            if verbose: even_print('ddx[u_tang]','%s'%str(ddx_u_tang.shape))
            
            if ('data/ddy_u_tang' in self): del self['data/ddy_u_tang']
            dset = self.create_dataset('data/ddy_u_tang', data=ddy_u_tang.T, chunks=None)
            if verbose: even_print('ddy[u_tang]','%s'%str(ddy_u_tang.shape))
            
            u_norm          = np.copy( self['data/u_norm'][()].T )
            ddx_u_norm_comp = gradient(u_norm, x_comp, axis=0, acc=acc, edge_stencil=edge_stencil, d=1)
            ddy_u_norm_comp = gradient(u_norm, y_comp, axis=1, acc=acc, edge_stencil=edge_stencil, d=1)
            ddx_u_norm      = ddx_u_norm_comp*ddx_q1 + ddy_u_norm_comp*ddx_q2
            ddy_u_norm      = ddx_u_norm_comp*ddy_q1 + ddy_u_norm_comp*ddy_q2
            
            if ('data/ddx_u_norm' in self): del self['data/ddx_u_norm']
            dset = self.create_dataset('data/ddx_u_norm', data=ddx_u_norm.T, chunks=None)
            if verbose: even_print('ddx[u_norm]','%s'%str(ddx_u_norm.shape))
            
            if ('data/ddy_u_norm' in self): del self['data/ddy_u_norm']
            dset = self.create_dataset('data/ddy_u_norm', data=ddy_u_norm.T, chunks=None)
            if verbose: even_print('ddy[u_norm]','%s'%str(ddy_u_norm.shape))
        
        # ===
        
        return
    
    def get_wall_norm_mesh(self, **kwargs):
        '''
        get a new 'grid' which is extruded in the normal direction from the wall
        - this grid is good for post-processing in the wall-normal direction ONLY
        - orthogonal to the wall-normal direction may have jumps or be folded
        '''
        
        verbose = kwargs.get('verbose',True)
        
        ## the wall-normal unit tangent and normal vectors
        if ('csys/v_tang' in self) and ('csys/v_norm' in self):
            v_tang = np.copy( self['csys/v_tang'][()] )
            v_norm = np.copy( self['csys/v_norm'][()] )
            #wall_trafo_mat = np.stack((v_tang,v_norm), axis=-1)
        else:
            raise AssertionError('no v_norm/v_tang unit projection vector!')
        
        if ('data/wall_distance' in self):
            wall_dist = np.copy( self['data/wall_distance'][()].T )
        else:
            raise AssertionError('dset not present: data/wall_distance')
        
        xy2d_n1D          = np.zeros((self.nx,self.ny,2) , dtype=np.float64)
        wall_distance_n1D = np.zeros((self.nx,self.ny)   , dtype=np.float64)
        
        for i in range(self.nx):
            
            p0_ = np.array([self.x[i,0],self.y[i,0]], dtype=np.float64) ## wall point coordinate
            
            v_norm_ = np.copy( v_norm[i,0,:] ) ## unit normal vec @ wall at this x
            #v_tang_ = np.copy( v_tang[i,0,:] ) ## unit tangent vec @ wall at this x
            
            #x_  = np.copy( self.x[i,:] )
            #y_  = np.copy( self.y[i,:] )
            #dx_ = np.diff(x_,n=1)
            #dy_ = np.diff(y_,n=1)
            #ds_ = np.sqrt(dx_**2+dy_**2)
            #s_  = np.cumsum(np.concatenate(([0.,],ds_))) ## path length normal to wall @ this x
            
            s_ = np.copy( wall_dist[i,:] )
            
            wall_distance_n1D[i,:] = s_
            
            xy = p0_ + np.einsum( 'i,j->ij', s_, v_norm_ )
            
            xy2d_n1D[i,:,:] = xy
        
        if ('dims_2Dw/x' in self): del self['dims_2Dw/x']
        self.create_dataset('dims_2Dw/x', data=np.squeeze(xy2d_n1D[:,:,0]).T, chunks=None)
        
        if ('dims_2Dw/y' in self): del self['dims_2Dw/y']
        self.create_dataset('dims_2Dw/y', data=np.squeeze(xy2d_n1D[:,:,1]).T, chunks=None)
        
        if ('data_2Dw/wall_distance' in self): del self['data_2Dw/wall_distance']
        self.create_dataset('data_2Dw/wall_distance', data=wall_distance_n1D.T, chunks=None)
        
        # ===
        
        if False: ## debug plot
            
            lwg = 0.12 ## line width grid
            
            xy2d1 = np.copy( np.stack((self.x,self.y), axis=-1) / self.lchar )
            xy2d2 = np.copy( xy2d_n1D / self.lchar )
            
            plt.close('all')
            mpl.style.use('dark_background')
            fig1 = plt.figure(figsize=(8,8/2), dpi=230)
            ax1 = fig1.gca()
            ax1.set_aspect('equal')
            ax1.tick_params(axis='x', which='both', direction='out')
            ax1.tick_params(axis='y', which='both', direction='out')
            ##
            # grid_ln_y = mpl.collections.LineCollection(xy2d1,                       linewidth=lwg, edgecolors='red', zorder=19)
            # grid_ln_x = mpl.collections.LineCollection(np.transpose(xy2d1,(1,0,2)), linewidth=lwg, edgecolors='red', zorder=19)
            # ax1.add_collection(grid_ln_y)
            # ax1.add_collection(grid_ln_x)
            ##
            grid_ln_y = mpl.collections.LineCollection(xy2d2,                       linewidth=lwg, edgecolors=ax1.xaxis.label.get_color(), zorder=19)
            grid_ln_x = mpl.collections.LineCollection(np.transpose(xy2d2,(1,0,2)), linewidth=lwg, edgecolors=ax1.xaxis.label.get_color(), zorder=19)
            ax1.add_collection(grid_ln_y)
            ax1.add_collection(grid_ln_x)
            ##
            ax1.set_xlabel('$x/\ell_{char}$')
            ax1.set_ylabel('$y/\ell_{char}$')
            ##
            ax1.set_xlim(xy2d1[:,:,0].min()-3,xy2d1[:,:,0].max()+3)
            ax1.set_ylim(xy2d1[:,:,1].min()-3,xy2d1[:,:,1].max()+3)
            ##
            #ax1.set_xlim(self.plot_xlim[0],self.plot_xlim[1])
            #ax1.set_ylim(self.plot_ylim[0],self.plot_ylim[1])
            ##
            ax1.xaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
            ax1.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
            ax1.yaxis.set_major_locator(mpl.ticker.MultipleLocator(20))
            ax1.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
            ##
            fig1.tight_layout(pad=0.25)
            fig1.tight_layout(pad=0.25)
            ##
            #dpi_out = 2*2160/plt.gcf().get_size_inches()[1]
            #turbx.fig_trim_x(fig1, [ax1], offset_px=10, dpi=dpi_out)
            #fig1.savefig('grid.png', dpi=dpi_out)
            plt.show()
            pass
        
        return
    
    def interp_to_wall_norm_mesh(self, **kwargs):
        '''
        interpolate fields from original grid to 'wall-normal' grid
        '''
        
        verbose = kwargs.get('verbose',True)
        
        if verbose: print('\n'+'turbx.ztmd.interp_to_wall_norm_mesh()'+'\n'+72*'-')
        t_start_func = timeit.default_timer()
        
        # === interpolate
        
        ## read u_tangent, u_normal
        if ('data/u_tang' in self) and ('data/u_norm' in self):
            utang = np.copy( self['data/u_tang'][()].T ) ## (wall) tangential velocity, shape=(nx,ny)
            unorm = np.copy( self['data/u_norm'][()].T ) ## (wall) normal velocity, shape=(nx,ny)
        else:
            raise AssertionError('no u_tang/u_norm velocities!')
        
        if ('dims_2Dw/x' in self):
            x_wn = np.copy( self['dims_2Dw/x'][()].T )
        else:
            raise AssertionError('dset not present: dims_2Dw/x')
        
        if ('dims_2Dw/y' in self):
            y_wn = np.copy( self['dims_2Dw/y'][()].T )
        else:
            raise AssertionError('dset not present: dims_2Dw/y')
        
        u   = np.copy( self['data/u'][()].T   )
        v   = np.copy( self['data/v'][()].T   )
        w   = np.copy( self['data/w'][()].T   )
        p   = np.copy( self['data/p'][()].T   )
        T   = np.copy( self['data/T'][()].T   )
        rho = np.copy( self['data/rho'][()].T )
        nu  = np.copy( self['data/nu'][()].T  )
        mu  = np.copy( self['data/mu'][()].T  )
        ##
        vort_z = np.copy( self['data/vort_z'][()].T )
        M      = np.copy( self['data/M'][()].T      )
        umag   = np.copy( self['data/umag'][()].T   )
        
        if True: ## interpolate
            
            x2d_A = self.x
            y2d_A = self.y
            x2d_B = x_wn
            y2d_B = y_wn
            
            scalar_data_dict = {'u':u, 'v':v, 'w':w, 'p':p, 'T':T, 'rho':rho, 'nu':nu, 'mu':mu,
                                'utang':utang, 'unorm':unorm,
                                'vort_z':vort_z, 'M':M, 'umag':umag }
            
            if verbose: progress_bar = tqdm(total=len(scalar_data_dict), ncols=100, desc='interpolate 2D', leave=False, file=sys.stdout)
            for scalar_name, scalar_data in scalar_data_dict.items():
                
                if verbose: tqdm.write(even_print('start interpolate',scalar_name,s=True))
                scalar_data_wn = interp_2d_structured(x2d_A, y2d_A, x2d_B, y2d_B, scalar_data)
                if ('data_2Dw/%s'%scalar_name in self):
                    del self['data_2Dw/%s'%scalar_name]
                self.create_dataset('data_2Dw/%s'%scalar_name, data=scalar_data_wn.T, chunks=None)
                if verbose: tqdm.write(even_print('done interpolating',scalar_name,s=True))
                
                progress_bar.update()
            progress_bar.close()
        
        if verbose: print('\n'+72*'-')
        if verbose: print('total time : turbx.interp_to_wall_norm_mesh() : %s'%format_time_string((timeit.default_timer() - t_start_func)))
        if verbose: print(72*'-')
        
        return
    
    def calc_wall_quantities(self, acc=6, edge_stencil='full', **kwargs):
        '''
        get 1D wall quantities
        -----
        - [ ρ_wall, ν_wall, μ_wall, T_wall ]
        - τ_wall = μ_wall·ddn[u_tang] :: [kg/(m·s)]·[m/s]/[m] = [kg/(m·s²)] = [N/m²] = [Pa]
        - u_τ = (τ_wall/ρ_wall)^(1/2)
        '''
        
        verbose = kwargs.get('verbose',True)
        
        if ('data_2Dw/u_tang' not in self):
            raise AssertionError('data_2Dw/u_tang not present')
        
        ## wall-normal interpolation coordinates (2D)
        x_wn = np.copy( self['dims_2Dw/x'][()].T )
        y_wn = np.copy( self['dims_2Dw/y'][()].T )
        s_wn = np.copy( self['data_2Dw/wall_distance'][()].T )
        
        ## wall-normal interpolated scalars (2D)
        u_tang_wn = np.copy( self['data_2Dw/u_tang'][()].T )
        T_wn      = np.copy( self['data_2Dw/T'][()].T      )
        vort_z_wn = np.copy( self['data_2Dw/vort_z'][()].T )
        
        # === get ρ_wall, ν_wall, μ_wall, T_wall
        
        rho = np.copy( self['data/rho'][()].T )
        rho_wall = np.copy( rho[:,0] )
        if ('data_1Dx/rho_wall' in self): del self['data_1Dx/rho_wall']
        dset = self.create_dataset('data_1Dx/rho_wall', data=rho_wall, chunks=None)
        #dset.attrs['dimensional'] = False
        #dset.attrs['unit'] = 'none'
        
        nu = np.copy( self['data/nu'][()].T )
        nu_wall = np.copy( nu[:,0] )
        if ('data_1Dx/nu_wall' in self): del self['data_1Dx/nu_wall']
        dset = self.create_dataset('data_1Dx/nu_wall', data=nu_wall, chunks=None)
        #dset.attrs['dimensional'] = True
        #dset.attrs['unit'] = '[m²/s]'
        
        mu = np.copy( self['data/mu'][()].T )
        mu_wall = np.copy( mu[:,0] )
        if ('data_1Dx/mu_wall' in self): del self['data_1Dx/mu_wall']
        dset = self.create_dataset('data_1Dx/mu_wall', data=mu_wall, chunks=None)
        #dset.attrs['dimensional'] = True
        #dset.attrs['unit'] = '[kg/(m·s)]'
        
        T = np.copy( self['data/T'][()].T )
        T_wall = np.copy( T[:,0] )
        if ('data_1Dx/T_wall' in self): del self['data_1Dx/T_wall']
        dset = self.create_dataset('data_1Dx/T_wall', data=T_wall, chunks=None)
        #dset.attrs['dimensional'] = False
        #dset.attrs['unit'] = 'none'
        
        # === get ddn[]
        
        if True:
            
            ddn_utang  = np.zeros((self.nx,self.ny), dtype=np.float64) ## dimensional [m/s]/[m] = [1/s]
            ddn_vort_z = np.zeros((self.nx,self.ny), dtype=np.float64)
            ##
            progress_bar = tqdm(total=self.nx, ncols=100, desc='get ddn[u_tang]', leave=False, file=sys.stdout)
            for i in range(self.nx):
                ddn_utang[i,:]  = gradient(u_tang_wn[i,:] , s_wn[i,:], axis=0, acc=acc, edge_stencil=edge_stencil, d=1)
                ddn_vort_z[i,:] = gradient(vort_z_wn[i,:] , s_wn[i,:], axis=0, acc=acc, edge_stencil=edge_stencil, d=1)
                progress_bar.update()
            progress_bar.close()
            
            if ('data_2Dw/ddn_utang' in self): del self['data_2Dw/ddn_utang']
            dset = self.create_dataset('data_2Dw/ddn_utang', data=ddn_utang.T, chunks=None)
            #dset.attrs['dimensional'] = True; dset.attrs['unit'] = '1/s'
            
            if ('data_2Dw/ddn_vort_z' in self): del self['data_2Dw/ddn_vort_z']
            dset = self.create_dataset('data_2Dw/ddn_vort_z', data=ddn_vort_z.T, chunks=None)
            
            if ('data_1Dx/ddn_utang_wall' in self): del self['data_1Dx/ddn_utang_wall']
            dset = self.create_dataset('data_1Dx/ddn_utang_wall', data=ddn_utang[:,0], chunks=None)
            #dset.attrs['dimensional'] = True; dset.attrs['unit'] = '1/s'
        
        else:
            
            ddn_utang  = np.copy( self['data_2Dw/ddn_utang'][()].T  )
            ddn_vort_z = np.copy( self['data_2Dw/ddn_vort_z'][()].T )
        
        # === calculate τ_wall & u_τ
        
        #if not self.get('data/mu').attrs['dimensional']:
        #    raise AssertionError('mu is dimless')
        
        ## wall shear stress τ_wall
        #tau_wall = mu_wall * dudy_wall
        tau_wall = mu_wall * ddn_utang[:,0]
        
        if ('data_1Dx/tau_wall' in self): del self['data_1Dx/tau_wall']
        dset = self.create_dataset('data_1Dx/tau_wall', data=tau_wall, chunks=None)
        #dset.attrs['dimensional'] = True
        #dset.attrs['unit'] = '[N/m²]' ## = [kg/(m·s²)] = [N/m²] = [Pa]
        
        ## friction velocity u_τ
        u_tau = np.sqrt( tau_wall / rho_wall )
        
        if ('data_1Dx/u_tau' in self): del self['data_1Dx/u_tau']
        dset = self.create_dataset('data_1Dx/u_tau', data=u_tau, chunks=None)
        #dset.attrs['dimensional'] = True
        #dset.attrs['unit'] = '[m/s]'
        
        # === inner scales: length, velocity & time
        
        sc_u_in = np.copy( u_tau              )
        sc_l_in = np.copy( nu_wall / u_tau    )
        sc_t_in = np.copy( nu_wall / u_tau**2 )
        np.testing.assert_allclose(sc_t_in, sc_l_in/sc_u_in, rtol=1e-14, atol=1e-14)
        
        if ('data_1Dx/sc_u_in' in self): del self['data_1Dx/sc_u_in']
        dset = self.create_dataset('data_1Dx/sc_u_in', data=sc_u_in, chunks=None)
        #dset.attrs['dimensional'] = True
        #dset.attrs['unit'] = '[m/s]'
        
        if ('data_1Dx/sc_l_in' in self): del self['data_1Dx/sc_l_in']
        dset = self.create_dataset('data_1Dx/sc_l_in', data=sc_l_in, chunks=None)
        #dset.attrs['dimensional'] = True
        #dset.attrs['unit'] = '[m]'
        
        if ('data_1Dx/sc_t_in' in self): del self['data_1Dx/sc_t_in']
        dset = self.create_dataset('data_1Dx/sc_t_in', data=sc_t_in, chunks=None)
        #dset.attrs['dimensional'] = True
        #dset.attrs['unit'] = '[s]'
        
        return
    
    def add_grid_quality_metrics_2d(self, **kwargs):
        '''
        attach grid quality measures to ZTMD
        '''
        x = np.copy( self['dims/x'][()].T )
        y = np.copy( self['dims/y'][()].T )
        
        grid_quality_dict = get_grid_quality_metrics_2d(x,y,verbose=True)
        
        if ('data_cells/skew' in self): del self['data_cells/skew']
        dset = self.create_dataset('data_cells/skew', data=grid_quality_dict['skew'].T, chunks=None)
        
        if ('data/ds1avg' in self): del self['data/ds1avg']
        dset = self.create_dataset('data/ds1avg', data=grid_quality_dict['ds1avg'].T, chunks=None)
        
        if ('data/ds2avg' in self): del self['data/ds2avg']
        dset = self.create_dataset('data/ds2avg', data=grid_quality_dict['ds2avg'].T, chunks=None)
        
        return
    
    def add_cyl_coords(self, **kwargs):
        '''
        attach [θ,r] coords, calculated from [x,y]
        '''
        cx = kwargs.get('cx',0.)
        cy = kwargs.get('cy',0.)
        
        x = np.copy( self['dims/x'][()].T )
        y = np.copy( self['dims/y'][()].T )
        
        xy2d = np.stack((x,y), axis=-1)
        
        trz = rect_to_cyl(xy2d, cx=cx, cy=cy)
        
        if ('dims/theta' in self): del self['dims/theta']
        dset = self.create_dataset('dims/theta', data=trz[:,:,0].T, chunks=None)
        if ('dims/r' in self): del self['dims/r']
        dset = self.create_dataset('dims/r', data=trz[:,:,1].T, chunks=None)
        
        return
    
    # === post-processing
    
    def calc_d99(self, **kwargs):
        '''
        calculate δ99 and BL edge values
        --> here on wall-normal (interpolated) mesh
        -----
        - δ99
        - sc_l_out = δ99
        - sc_u_out = u99
        - sc_t_out = u99/d99
        - skin friction coefficient: cf
        '''
        
        verbose = kwargs.get('verbose',True)
        
        nx = self.nx
        ny = self.ny
        #x  = self.x
        #y  = self.y
        
        ## copy 2D datasets into memory
        ##  --> here wall-normal interpolated
        x         = np.copy( self['dims_2Dw/x'][()].T             )
        #y         = np.copy( self['dims_2Dw/y'][()].T             )
        wall_dist = np.copy( self['data_2Dw/wall_distance'][()].T )
        utang     = np.copy( self['data_2Dw/u_tang'][()].T        )
        p         = np.copy( self['data_2Dw/p'][()].T             )
        T         = np.copy( self['data_2Dw/T'][()].T             )
        rho       = np.copy( self['data_2Dw/rho'][()].T           )
        mu        = np.copy( self['data_2Dw/mu'][()].T            )
        vort_z    = np.copy( self['data_2Dw/vort_z'][()].T        )
        nu        = np.copy( self['data_2Dw/nu'][()].T            )
        M         = np.copy( self['data_2Dw/M'][()].T             )
        
        ## copy csys datasets into memory
        v_tang = np.copy( self['csys/v_tang'][()] )
        v_norm = np.copy( self['csys/v_norm'][()] )
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        y = np.copy(wall_dist)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        
        ## assertions
        if (x.shape != (self.nx,self.ny)):
            raise ValueError('x.shape != (self.nx,self.ny)')
        if (y.shape != (self.nx,self.ny)):
            raise ValueError('y.shape != (self.nx,self.ny)')
        
        ## copy 1D datasets into memory
        u_tau    = np.copy( self['data_1Dx/u_tau'][()]    )
        rho_wall = np.copy( self['data_1Dx/rho_wall'][()] )
        nu_wall  = np.copy( self['data_1Dx/nu_wall'][()]  )
        mu_wall  = np.copy( self['data_1Dx/mu_wall'][()]  )
        T_wall   = np.copy( self['data_1Dx/T_wall'][()]   )
        tau_wall = np.copy( self['data_1Dx/tau_wall'][()] )
        
        ## calculate & write pseudo-velocity
        psvel = np.zeros(shape=(nx,ny) , dtype=np.float64)
        for i in range(nx):
            psvel[i,:] = sp.integrate.cumtrapz(-1*vort_z[i,:], y[i,:], initial=0.)
        
        if ('data_2Dw/psvel' in self): del self['data_2Dw/psvel']
        dset = self.create_dataset('data_2Dw/psvel', data=psvel.T, chunks=None)
        
        # === edge index
        
        epsilon = 1e-6
        fac = 1/8
        
        ## 'epsilon' becomes difficult to tune across different Ma, etc
        ## - If large:
        ##    - The ddx(j_edge) is relatively smooth, BUT it's still in a region where the gradient
        ##        is still (relatively high). It is, after all, no longer really the 'edge'
        ##    - This results in the 99 values and their dependents (i.e. Re_theta) having digital 'steps'
        ## - If small:
        ##    - The 99 values are relatively smooth BUT...
        ##    - The ddx(j_edge) can be 'shock-like' and very uneven, usually producing single jumps in 99 values
        ## - The ad-hoc solution:
        ##    - First use large epsilon, THEN find point in between that point and the top boundary
        
        j_edge_psvel = np.zeros(shape=(nx,), dtype=np.int32)
        j_edge_utang = np.zeros(shape=(nx,), dtype=np.int32)
        
        if verbose: progress_bar = tqdm(total=nx, ncols=100, desc='edge', leave=False, file=sys.stdout)
        for i in range(nx):
            
            ## edge detector : pseudo-velocity
            psvelmax = psvel[i,:].max()
            for j in range(ny):
                if math.isclose(psvel[i,j], psvelmax, rel_tol=epsilon):
                    j_edge_psvel[i] = np.abs( y[i,:] - (y[i,j] + fac*(y[i,:].max()-y[i,j])) ).argmin()
                    break
                if (psvel[i,j]>psvelmax):
                    j_edge_psvel[i] = j-1
                    break
            
            ## edge detector : tangential velocity 'utang'
            utangmax = utang[i,:].max()
            for j in range(ny):
                if math.isclose(utang[i,j], utangmax, rel_tol=epsilon):
                    j_edge_utang[i] = np.abs( y[i,:] - (y[i,j] + fac*(y[i,:].max()-y[i,j])) ).argmin()
                    break
                if (utang[i,j]>utangmax):
                    j_edge_utang[i] = j-1
                    break
            
            if verbose: progress_bar.update()
        if verbose: progress_bar.close()
        
        j_edge = np.copy( j_edge_psvel )
        #j_edge = np.copy( j_edge_utang )
        
        if ('data_1Dx/j_edge' in self): del self['data_1Dx/j_edge']
        dset = self.create_dataset('data_1Dx/j_edge', data=j_edge, chunks=None)
        
        # === edge values
        
        y_edge     = np.zeros(shape=(nx,)  , dtype=np.float64 )
        y_edge_2d  = np.zeros(shape=(nx,2) , dtype=np.float64 )
        utang_edge = np.zeros(shape=(nx,)  , dtype=np.float64 )
        psvel_edge = np.zeros(shape=(nx,)  , dtype=np.float64 )
        T_edge     = np.zeros(shape=(nx,)  , dtype=np.float64 )
        p_edge     = np.zeros(shape=(nx,)  , dtype=np.float64 )
        rho_edge   = np.zeros(shape=(nx,)  , dtype=np.float64 )
        mu_edge    = np.zeros(shape=(nx,)  , dtype=np.float64 )
        nu_edge    = np.zeros(shape=(nx,)  , dtype=np.float64 )
        #a_edge     = np.zeros(shape=(nx,)  , dtype=np.float64 )
        M_edge     = np.zeros(shape=(nx,)  , dtype=np.float64 )
        
        for i in range(nx):
            
            je = j_edge[i]
            
            y_edge[i]     = y[i,je]
            utang_edge[i] = utang[i,je]
            psvel_edge[i] = psvel[i,je]
            
            #u_edge[i]     = u[i,je]
            #v_edge[i]     = v[i,je]
            #w_edge[i]     = w[i,je]
            #umag_edge[i]  = umag[i,je]
            #mflux_edge[i] = mflux[i,je]
            
            T_edge[i]     = T[i,je]
            p_edge[i]     = p[i,je]
            rho_edge[i]   = rho[i,je]
            mu_edge[i]    = mu[i,je]
            nu_edge[i]    = nu[i,je]
            #a_edge[i]     = a[i,je]
            M_edge[i]     = M[i,je]
            
            p0_      = np.array([self.x[i,0],self.y[i,0]], dtype=np.float64) ## wall point coordinate
            v_norm_  = np.copy( v_norm[i,0,:] ) ## unit normal vec @ wall at this x
            pt_edge_ = p0_ + np.dot( y[i,je] , v_norm_ )
            
            y_edge_2d[i,:] = pt_edge_
        
        if ('data_1Dx/y_edge' in self): del self['data_1Dx/y_edge']
        dset = self.create_dataset('data_1Dx/y_edge', data=y_edge, chunks=None)
        
        if ('data_1Dx/y_edge_2d' in self): del self['data_1Dx/y_edge_2d']
        dset = self.create_dataset('data_1Dx/y_edge_2d', data=y_edge_2d, chunks=None)
        
        if ('data_1Dx/utang_edge' in self): del self['data_1Dx/utang_edge']
        dset = self.create_dataset('data_1Dx/utang_edge', data=utang_edge, chunks=None)
        
        if ('data_1Dx/psvel_edge' in self): del self['data_1Dx/psvel_edge']
        dset = self.create_dataset('data_1Dx/psvel_edge', data=psvel_edge, chunks=None)
        
        if ('data_1Dx/T_edge' in self): del self['data_1Dx/T_edge']
        dset = self.create_dataset('data_1Dx/T_edge', data=T_edge, chunks=None)
        
        if ('data_1Dx/p_edge' in self): del self['data_1Dx/p_edge']
        dset = self.create_dataset('data_1Dx/p_edge', data=p_edge, chunks=None)
        
        if ('data_1Dx/rho_edge' in self): del self['data_1Dx/rho_edge']
        dset = self.create_dataset('data_1Dx/rho_edge', data=rho_edge, chunks=None)
        
        if ('data_1Dx/mu_edge' in self): del self['data_1Dx/mu_edge']
        dset = self.create_dataset('data_1Dx/mu_edge', data=mu_edge, chunks=None)
        
        if ('data_1Dx/nu_edge' in self): del self['data_1Dx/nu_edge']
        dset = self.create_dataset('data_1Dx/nu_edge', data=nu_edge, chunks=None)
        
        #if ('data_1Dx/a_edge' in self): del self['data_1Dx/a_edge']
        #dset = self.create_dataset('data_1Dx/a_edge', data=a_edge, chunks=None)
        
        if ('data_1Dx/M_edge' in self): del self['data_1Dx/M_edge']
        dset = self.create_dataset('data_1Dx/M_edge', data=M_edge, chunks=None)
        
        # === δ99 (interpolated) values
        
        d99     = np.zeros(shape=(nx,)  , dtype=np.float64)
        d99_2d  = np.zeros(shape=(nx,2) , dtype=np.float64)
        utang99 = np.zeros(shape=(nx,)  , dtype=np.float64)
        T99     = np.zeros(shape=(nx,)  , dtype=np.float64)
        nu99    = np.zeros(shape=(nx,)  , dtype=np.float64)
        
        if verbose: progress_bar = tqdm(total=nx, ncols=100, desc='δ99', leave=False, file=sys.stdout)
        for i in range(nx):
            
            je = j_edge[i]+5 ## add pts for interpolation
            
            # === find δ99 roots with interpolating function
            
            if True: ## CubicSpline
                
                f_psvel = sp.interpolate.CubicSpline(y[i,:je], psvel[i,:je]-(0.99*psvel_edge[i]), bc_type='natural')
                roots   = f_psvel.roots(discontinuity=False, extrapolate=False)
            
            # === check & write δ99 (to 1D buffer vec)
            
            if (roots.size==0):
                raise ValueError('no roots found at i=%i'%i)
            if (roots.size>1):
                raise ValueError('multiple roots found at i=%i'%i)
            
            d99_ = roots[0]
            if not (d99_<y_edge[i]):
                raise ValueError('δ99 root location is > edge location')
            
            d99[i] = d99_
            
            # === get the physical [x,y] coordinates of δ99 (project in wall-normal direction_
            
            p0_         = np.array([self.x[i,0],self.y[i,0]], dtype=np.float64) ## wall point coordinate
            v_norm_     = np.copy( v_norm[i,0,:] ) ## unit normal vec @ wall at this x
            p99_        = p0_ + np.dot( d99_ , v_norm_ ) ## start point + (n·δ99)
            d99_2d[i,:] = p99_
            
            # === interpolate other variables at δ99
            
            utang99[i] = sp.interpolate.interp1d(y[i,:je], utang[i,:je] )(d99_)
            T99[i]     = sp.interpolate.interp1d(y[i,:je],     T[i,:je] )(d99_)
            nu99[i]    = sp.interpolate.interp1d(y[i,:je],    nu[i,:je] )(d99_)
            
            # ===
            
            if verbose: progress_bar.update()
        if verbose: progress_bar.close()
        
        if ('data_1Dx/d99' in self): del self['data_1Dx/d99']
        dset = self.create_dataset('data_1Dx/d99', data=d99, chunks=None)
        
        if ('data_1Dx/d99_2d' in self): del self['data_1Dx/d99_2d']
        dset = self.create_dataset('data_1Dx/d99_2d', data=d99_2d, chunks=None)
        
        if ('data_1Dx/T99' in self): del self['data_1Dx/T99']
        dset = self.create_dataset('data_1Dx/T99', data=T99, chunks=None)
        
        if ('data_1Dx/nu99' in self): del self['data_1Dx/nu99']
        dset = self.create_dataset('data_1Dx/nu99', data=nu99, chunks=None)
        
        # === outer scales: length, velocity & time
        
        sc_u_out = np.copy( utang99     )
        sc_l_out = np.copy( d99         )
        sc_t_out = np.copy( d99/utang99 )
        np.testing.assert_allclose(sc_t_out, sc_l_out/sc_u_out, rtol=1e-14, atol=1e-14)
        
        sc_t_eddy = np.copy( d99/u_tau )
        
        if ('data_1Dx/sc_u_out' in self): del self['data_1Dx/sc_u_out']
        self.create_dataset('data_1Dx/sc_u_out', data=sc_u_out, chunks=None)
        
        if ('data_1Dx/sc_l_out' in self): del self['data_1Dx/sc_l_out']
        self.create_dataset('data_1Dx/sc_l_out', data=sc_l_out, chunks=None)
        
        if ('data_1Dx/sc_t_out' in self): del self['data_1Dx/sc_t_out']
        self.create_dataset('data_1Dx/sc_t_out', data=sc_t_out, chunks=None)
        
        if ('data_1Dx/sc_t_eddy' in self): del self['data_1Dx/sc_t_eddy']
        self.create_dataset('data_1Dx/sc_t_eddy', data=sc_t_eddy, chunks=None)
        
        # === skin friction
        
        cf_1 = 2. * (u_tau/utang_edge)**2 * (rho_wall/rho_edge)
        cf_2 = 2. * tau_wall / (rho_edge*utang_edge**2)
        np.testing.assert_allclose(cf_1, cf_2, rtol=1e-8)
        cf = np.copy(cf_2)
        
        if ('data_1Dx/cf' in self): del self['data_1Dx/cf']
        self.create_dataset('data_1Dx/cf', data=cf, chunks=None)
        
        return
    
    def calc_bl_1d_integral_quantities(self, **kwargs):
        '''
        θ, δ*, Re_θ, Re_τ
        '''
        
        verbose = kwargs.get('verbose',True)
        
        nx = self.nx
        ny = self.ny
        
        ## copy 2D datasets into memory
        ##  --> here wall-normal interpolated
        x         = np.copy( self['dims_2Dw/x'][()].T             )
        #y         = np.copy( self['dims_2Dw/y'][()].T             )
        wall_dist = np.copy( self['data_2Dw/wall_distance'][()].T )
        utang     = np.copy( self['data_2Dw/u_tang'][()].T        )
        vort_z    = np.copy( self['data_2Dw/vort_z'][()].T        )
        nu        = np.copy( self['data_2Dw/nu'][()].T            )
        T         = np.copy( self['data_2Dw/T'][()].T             )
        rho       = np.copy( self['data_2Dw/rho'][()].T           )
        
        ## copy 1D datasets into memory
        u_tau    = np.copy( self['data_1Dx/u_tau'][()]    )
        rho_wall = np.copy( self['data_1Dx/rho_wall'][()] )
        nu_wall  = np.copy( self['data_1Dx/nu_wall'][()]  )
        mu_wall  = np.copy( self['data_1Dx/mu_wall'][()]  )
        T_wall   = np.copy( self['data_1Dx/T_wall'][()]   )
        ##
        j_edge     = np.copy( self['data_1Dx/j_edge'][()]     )
        y_edge     = np.copy( self['data_1Dx/y_edge'][()]     )
        d99        = np.copy( self['data_1Dx/d99'][()]        )
        utang_edge = np.copy( self['data_1Dx/utang_edge'][()] )
        rho_edge   = np.copy( self['data_1Dx/rho_edge'][()]   )
        mu_edge    = np.copy( self['data_1Dx/mu_edge'][()]    )
        nu_edge    = np.copy( self['data_1Dx/nu_edge'][()]    )
        ##
        sc_l_out = np.copy( self['data_1Dx/sc_l_out'][()] )
        sc_u_out = np.copy( self['data_1Dx/sc_u_out'][()] )
        sc_t_out = np.copy( self['data_1Dx/sc_t_out'][()] )
        
        # ===
        
        theta_cmp = np.zeros(shape=(nx,), dtype=np.float64) ## momentum thickness
        theta_inc = np.zeros(shape=(nx,), dtype=np.float64)
        dstar_cmp = np.zeros(shape=(nx,), dtype=np.float64) ## displacement thickness
        dstar_inc = np.zeros(shape=(nx,), dtype=np.float64)
        
        utang_vd = np.zeros(shape=(nx,ny), dtype=np.float64)
        
        if verbose: progress_bar = tqdm(total=nx, ncols=100, desc='θ,δ*,Re_θ,Re_τ', leave=False, file=sys.stdout)
        for i in range(nx):
            
            je   = j_edge[i]
            y_   = wall_dist[i,:je+1]
            u_   =     utang[i,:je+1]
            rho_ =       rho[i,:je+1]
            
            integrand_theta_inc = (u_/utang_edge[i])*(1-(u_/utang_edge[i]))
            integrand_dstar_inc = 1-(u_/utang_edge[i])
            theta_inc[i]        = sp.integrate.trapezoid(integrand_theta_inc, x=y_)
            dstar_inc[i]        = sp.integrate.trapezoid(integrand_dstar_inc, x=y_)
            
            integrand_theta_cmp = (u_*rho_)/(utang_edge[i]*rho_edge[i])*(1-(u_/utang_edge[i]))
            integrand_dstar_cmp = (1-((u_*rho_)/(utang_edge[i]*rho_edge[i])))
            theta_cmp[i]        = sp.integrate.trapezoid(integrand_theta_cmp, x=y_)
            dstar_cmp[i]        = sp.integrate.trapezoid(integrand_dstar_cmp, x=y_)
            
            integrand_u_vd      = np.sqrt(T_wall[i]/T[i,:])
            #integrand_u_vd      = np.sqrt(rho[i,:]/rho_wall[i])
            
            utang_vd[i,:] = sp.integrate.cumtrapz(integrand_u_vd, utang[i,:], initial=0)
            
            if verbose: progress_bar.update()
        if verbose: progress_bar.close()
        
        if ('data_1Dx/theta_inc' in self): del self['data_1Dx/theta_inc']
        self.create_dataset('data_1Dx/theta_inc', data=theta_inc, chunks=None)
        
        if ('data_1Dx/dstar_inc' in self): del self['data_1Dx/dstar_inc']
        self.create_dataset('data_1Dx/dstar_inc', data=dstar_inc, chunks=None)
        
        if ('data_1Dx/theta_cmp' in self): del self['data_1Dx/theta_cmp']
        self.create_dataset('data_1Dx/theta_cmp', data=theta_cmp, chunks=None)
        
        if ('data_1Dx/dstar_cmp' in self): del self['data_1Dx/dstar_cmp']
        self.create_dataset('data_1Dx/dstar_cmp', data=dstar_cmp, chunks=None)
        
        if ('data_2Dw/utang_vd' in self): del self['data_2Dw/utang_vd']
        self.create_dataset('data_2Dw/utang_vd', data=utang_vd.T, chunks=None)
        
        # ===
        
        #theta   = np.copy(theta_cmp)
        #dstar   = np.copy(dstar_cmp)
        H12     = dstar_cmp/theta_cmp
        H12_inc = dstar_inc/theta_inc
        
        Re_tau        = d99*u_tau/nu_wall   
        Re_theta      = theta_cmp*utang_edge/nu_edge   
        Re_theta_inc  = theta_inc*utang_edge/nu_edge   
        Re_theta_wall = theta_cmp*utang_edge/(mu_wall/rho_edge) # (?)
        Re_d99        = d99*utang_edge/nu_edge
        
        if ('data_1Dx/H12' in self): del self['data_1Dx/H12']
        self.create_dataset('data_1Dx/H12', data=H12, chunks=None)
        
        if ('data_1Dx/H12_inc' in self): del self['data_1Dx/H12_inc']
        self.create_dataset('data_1Dx/H12_inc', data=H12_inc, chunks=None)
        
        if ('data_1Dx/Re_tau' in self): del self['data_1Dx/Re_tau']
        self.create_dataset('data_1Dx/Re_tau', data=Re_tau, chunks=None)
        
        if ('data_1Dx/Re_theta' in self): del self['data_1Dx/Re_theta']
        self.create_dataset('data_1Dx/Re_theta', data=Re_theta, chunks=None)
        
        if ('data_1Dx/Re_theta_inc' in self): del self['data_1Dx/Re_theta_inc']
        self.create_dataset('data_1Dx/Re_theta_inc', data=Re_theta_inc, chunks=None)
        
        if ('data_1Dx/Re_d99' in self): del self['data_1Dx/Re_d99']
        self.create_dataset('data_1Dx/Re_d99', data=Re_d99, chunks=None)
        
        return
    
    # === Paraview
    
    def make_xdmf(self, **kwargs):
        '''
        generate an XDMF/XMF2 from ZTMD for processing with Paraview
        -----
        --> https://www.xdmf.org/index.php/XDMF_Model_and_Format
        '''
        
        if (self.rank==0):
            verbose = True
        else:
            verbose = False
        
        makeVectors = kwargs.get('makeVectors',True) ## write vectors (e.g. velocity, vorticity) to XDMF
        makeTensors = kwargs.get('makeTensors',True) ## write 3x3 tensors (e.g. stress, strain) to XDMF
        
        fname_path            = os.path.dirname(self.fname)
        fname_base            = os.path.basename(self.fname)
        fname_root, fname_ext = os.path.splitext(fname_base)
        fname_xdmf_base       = fname_root+'.xmf2'
        fname_xdmf            = os.path.join(fname_path, fname_xdmf_base)
        
        if verbose: print('\n'+'ztmd.make_xdmf()'+'\n'+72*'-')
        
        dataset_precision_dict = {} ## holds dtype.itemsize ints i.e. 4,8
        dataset_numbertype_dict = {} ## holds string description of dtypes i.e. 'Float','Integer'
        
        # === 1D coordinate dimension vectors --> get dtype.name
        for scalar in ['x','y','r','theta']:
            if ('dims/'+scalar in self):
                data = self['dims/'+scalar]
                dataset_precision_dict[scalar] = data.dtype.itemsize
                if (data.dtype.name=='float32') or (data.dtype.name=='float64'):
                    dataset_numbertype_dict[scalar] = 'Float'
                elif (data.dtype.name=='int8') or (data.dtype.name=='int16') or (data.dtype.name=='int32') or (data.dtype.name=='int64'):
                    dataset_numbertype_dict[scalar] = 'Integer'
                else:
                    raise ValueError('dtype not recognized, please update script accordingly')
        
        ## refresh header
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
        
        # === write to .xdmf/.xmf2 file
        if (self.rank==0):
            
            #with open(fname_xdmf,'w') as xdmf:
            with io.open(fname_xdmf,'w',newline='\n') as xdmf:
                
                xdmf_str='''
                         <?xml version="1.0" encoding="utf-8"?>
                         <!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
                         <Xdmf xmlns:xi="http://www.w3.org/2001/XInclude" Version="2.0">
                           <Domain>
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 0*' '))
                
                xdmf_str=f'''
                         <Topology TopologyType="3DSMesh" NumberOfElements="{self.ny:d} {self.nx:d}"/>
                         <Geometry GeometryType="X_Y_Z">
                           <DataItem Dimensions="{self.nx:d} {self.ny:d}" NumberType="{dataset_numbertype_dict['x']}" Precision="{dataset_precision_dict['x']:d}" Format="HDF">
                             {fname_base}:/dims/{'x'}
                           </DataItem>
                           <DataItem Dimensions="{self.nx:d} {self.ny:d}" NumberType="{dataset_numbertype_dict['y']}" Precision="{dataset_precision_dict['y']:d}" Format="HDF">
                             {fname_base}:/dims/{'y'}
                           </DataItem>
                         </Geometry>
                         '''
                
                xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 4*' '))
                
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
                    
                    dset_name = 'ts_%08d'%ti
                    
                    xdmf_str='''
                             <!-- ============================================================ -->
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # ===
                    
                    xdmf_str=f'''
                             <Grid Name="{dset_name}" GridType="Uniform">
                               <Time TimeType="Single" Value="{self.t[ti]:0.8E}"/>
                               <Topology Reference="/Xdmf/Domain/Topology[1]" />
                               <Geometry Reference="/Xdmf/Domain/Geometry[1]" />
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # === .xdmf : <Grid> per 2D coordinate array
                    
                    for scalar in ['x','y','r','theta']:
                        
                        dset_hf_path = 'dims/%s'%scalar
                        
                        if dset_hf_path in self:
                            
                            ## get optional 'label' for Paraview (currently inactive)
                            #if scalar in scalar_names:
                            if False:
                                scalar_name = scalar_names[scalar]
                            else:
                                scalar_name = scalar
                            
                            xdmf_str=f'''
                                     <!-- ===== scalar : {scalar} ===== -->
                                     <Attribute Name="{scalar_name}" AttributeType="Scalar" Center="Node">
                                       <DataItem Dimensions="{self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict[scalar]}" Precision="{dataset_precision_dict[scalar]:d}" Format="HDF">
                                         {fname_base}:/{dset_hf_path}
                                       </DataItem>
                                     </Attribute>
                                     '''
                            
                            xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    # === .xdmf : <Grid> per scalar
                    
                    for scalar in self.scalars:
                        
                        dset_hf_path = 'data/%s'%scalar
                        
                        ## get optional 'label' for Paraview (currently inactive)
                        #if scalar in scalar_names:
                        if False:
                            scalar_name = scalar_names[scalar]
                        else:
                            scalar_name = scalar
                        
                        xdmf_str=f'''
                                 <!-- ===== scalar : {scalar} ===== -->
                                 <Attribute Name="{scalar_name}" AttributeType="Scalar" Center="Node">
                                   <DataItem Dimensions="{self.ny:d} {self.nx:d}" NumberType="{dataset_numbertype_dict[scalar]}" Precision="{dataset_precision_dict[scalar]:d}" Format="HDF">
                                     {fname_base}:/{dset_hf_path}
                                   </DataItem>
                                 </Attribute>
                                 '''
                        
                        xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    # === .xdmf : <Grid> per scalar (cell-centered values)
                    
                    if ('data_cells' in self):
                        scalars_cells = list(self['data_cells'].keys())
                        for scalar in scalars_cells:
                            
                            dset_hf_path = 'data_cells/%s'%scalar
                            dset = self[dset_hf_path]
                            dset_precision = dset.dtype.itemsize
                            scalar_name = scalar
                            
                            if (dset.dtype.name=='float32') or (dset.dtype.name=='float64'):
                                dset_numbertype = 'Float'
                            elif (data.dtype.name=='int8') or (data.dtype.name=='int16') or (data.dtype.name=='int32') or (data.dtype.name=='int64'):
                                dset_numbertype = 'Integer'
                            else:
                                raise TypeError('dtype not recognized, please update script accordingly')
                            
                            xdmf_str=f'''
                                     <!-- ===== scalar : {scalar} ===== -->
                                     <Attribute Name="{scalar_name}" AttributeType="Scalar" Center="Cell">
                                       <DataItem Dimensions="{(self.ny-1):d} {(self.nx-1):d}" NumberType="{dset_numbertype}" Precision="{dset_precision:d}" Format="HDF">
                                         {fname_base}:/{dset_hf_path}
                                       </DataItem>
                                     </Attribute>
                                     '''
                            
                            xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    xdmf_str='''
                             <!-- ===== end scalars ===== -->
                             '''
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 8*' '))
                    
                    # === .xdmf : end Grid for this timestep
                    
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
    ├── dims/ --> 1D (rectilinear coords of source volume) --> reference only!
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

        self.fname_path = os.path.dirname(self.fname)
        self.fname_base = os.path.basename(self.fname)
        self.fname_root, self.fname_ext = os.path.splitext(self.fname_base)
        
        ## default to libver='latest' if none provided
        if ('libver' not in kwargs):
            kwargs['libver'] = 'latest'
        
        ## catch possible user error --> could prevent accidental EAS overwrites
        if (self.fname_ext=='.eas'):
            raise ValueError('EAS4 files should not be opened with turbx.lpd()')
        
        ## determine if using mpi
        if ('driver' in kwargs) and (kwargs['driver']=='mpio'):
            self.usingmpi = True
        else:
            self.usingmpi = False
        
        ## determine communicator & rank info
        if self.usingmpi:
            self.comm    = kwargs['comm']
            self.n_ranks = self.comm.Get_size()
            self.rank    = self.comm.Get_rank()
        else:
            self.comm    = None
            self.n_ranks = 1
            self.rank    = 0
        
        ## if not using MPI, remove 'driver' and 'comm' from kwargs
        if ( not self.usingmpi ) and ('driver' in kwargs):
            kwargs.pop('driver')
        if ( not self.usingmpi ) and ('comm' in kwargs):
            kwargs.pop('comm')
        
        ## determine MPI info / hints
        if self.usingmpi:
            if ('info' in kwargs):
                self.mpi_info = kwargs['info']
            else:
                mpi_info = MPI.Info.Create()
                ##
                mpi_info.Set('romio_ds_write' , 'disable'   )
                mpi_info.Set('romio_ds_read'  , 'disable'   )
                mpi_info.Set('romio_cb_read'  , 'automatic' )
                mpi_info.Set('romio_cb_write' , 'automatic' )
                #mpi_info.Set('romio_cb_read'  , 'enable' )
                #mpi_info.Set('romio_cb_write' , 'enable' )
                mpi_info.Set('cb_buffer_size' , str(int(round(8*1024**2))) ) ## 8 [MB]
                ##
                kwargs['info'] = mpi_info
                self.mpi_info = mpi_info
        
        if ('rdcc_nbytes' not in kwargs):
            kwargs['rdcc_nbytes'] = int(16*1024**2) ## 16 [MB]
        
        ## lpd() unique kwargs (not h5py.File kwargs) --> pop() rather than get()
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
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 8M %s > /dev/null 2>&1'%self.fname, shell=True)
                else:
                    #print('striping with lfs not permitted on this filesystem')
                    pass
        
        ## touch, stripe
        elif (openMode == 'w') and not os.path.isfile(self.fname):
            if (self.rank==0):
                Path(self.fname).touch()
                if shutil.which('lfs') is not None:
                    return_code = subprocess.call('lfs migrate --stripe-count 16 --stripe-size 8M %s > /dev/null 2>&1'%self.fname, shell=True)
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
        chunk_kb = kwargs.get('chunk_kb',4*1024) ## 4 [MB]
        
        if verbose:
            even_print('fn_lpd',  self.fname)
            even_print('npts'  , '%i'%self.npts)
            even_print('nt'    , '%i'%self.nt)
            print(72*'-')
        
        ## particle list bounds this rank
        rpl_ = np.array_split(np.arange(self.npts, dtype=np.int64), self.n_ranks )
        rpl  = [[b[0],b[-1]+1] for b in rpl_ ]
        rp1, rp2 = rpl[self.rank]
        npr = rp2 - rp1
        
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
                if self.usingmpi: self.comm.Barrier()
                t_start = timeit.default_timer()
                
                if self.usingmpi:
                    with dset.collective:
                        #pdata_in[scalar] = dset[rp1:rp2,:]
                        pdata_in[scalar] = dset[cp1:cp2,:]
                else:
                    pdata_in[scalar] = dset[cp1:cp2,:]
                
                if self.usingmpi: self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                
                #data_gb = 4 * self.nt * self.npts / 1024**3
                data_gb = ( 4 * self.nt * self.npts / 1024**3 ) / n_chunks
                if verbose:
                    tqdm.write(even_print('read: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True))
            
            # === iterate over (chunk of) particle tracks
            
            if self.usingmpi: self.comm.Barrier()
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
                
                if (n_real>=12): ## must have at least N timesteps with data
                    
                    if False: ## numpy O2 gradient
                        dudt   = np.gradient(u,    t, axis=0, edge_order=2)
                        dvdt   = np.gradient(v,    t, axis=0, edge_order=2)
                        dwdt   = np.gradient(w,    t, axis=0, edge_order=2)
                        d2udt2 = np.gradient(dudt, t, axis=0, edge_order=2)
                        d2vdt2 = np.gradient(dvdt, t, axis=0, edge_order=2)
                        d2wdt2 = np.gradient(dwdt, t, axis=0, edge_order=2)
                    
                    if False: ## O3 Cubic Spline
                        dudt   = sp.interpolate.CubicSpline(t,u,bc_type='natural')(t,1)
                        dvdt   = sp.interpolate.CubicSpline(t,v,bc_type='natural')(t,1)
                        dwdt   = sp.interpolate.CubicSpline(t,w,bc_type='natural')(t,1)
                        d2udt2 = sp.interpolate.CubicSpline(t,u,bc_type='natural')(t,2)
                        d2vdt2 = sp.interpolate.CubicSpline(t,v,bc_type='natural')(t,2)
                        d2wdt2 = sp.interpolate.CubicSpline(t,w,bc_type='natural')(t,2)
                    
                    if True: ## Finite Difference O6
                        
                        acc = 6 ## order of truncated term (error term) in FD formulation
                        
                        dudt   = gradient(u, t, d=1, axis=0, acc=acc, edge_stencil='full')
                        dvdt   = gradient(v, t, d=1, axis=0, acc=acc, edge_stencil='full')
                        dwdt   = gradient(w, t, d=1, axis=0, acc=acc, edge_stencil='full')
                        d2udt2 = gradient(u, t, d=2, axis=0, acc=acc, edge_stencil='full')
                        d2vdt2 = gradient(v, t, d=2, axis=0, acc=acc, edge_stencil='full')
                        d2wdt2 = gradient(w, t, d=2, axis=0, acc=acc, edge_stencil='full')
                    
                    ## write to buffer
                    pdata_out['ax'][pi,ii_real] = dudt  
                    pdata_out['ay'][pi,ii_real] = dvdt  
                    pdata_out['az'][pi,ii_real] = dwdt  
                    pdata_out['jx'][pi,ii_real] = d2udt2
                    pdata_out['jy'][pi,ii_real] = d2vdt2
                    pdata_out['jz'][pi,ii_real] = d2wdt2
                
                if verbose: progress_bar.update()
            
            if self.usingmpi: self.comm.Barrier()
            t_delta = timeit.default_timer() - t_start
            if verbose:
                tqdm.write(even_print('calc accel & jerk', '%0.2f [s]'%(t_delta,), s=True))
            
            # === write buffer out
            
            for scalar in scalars_out:
                
                dset = self['data/%s'%scalar]
                
                if self.usingmpi: self.comm.Barrier()
                t_start = timeit.default_timer()
                
                if self.usingmpi:
                    with dset.collective:
                        dset[cp1:cp2,:] = pdata_out[scalar]
                else:
                    dset[cp1:cp2,:] = pdata_out[scalar]
                
                if self.usingmpi: self.comm.Barrier()
                t_delta = timeit.default_timer() - t_start
                
                data_gb = ( 4 * self.nt * self.npts / 1024**3 ) / n_chunks
                if verbose:
                    tqdm.write(even_print('write: %s'%scalar, '%0.2f [GB]  %0.2f [s]  %0.3f [GB/s]'%(data_gb,t_delta,(data_gb/t_delta)), s=True))
            
            # ===
            
            if verbose:
                tqdm.write(72*'-')
        
        if verbose: progress_bar.close()
        
        # ===
        
        if self.usingmpi: self.comm.Barrier()
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
            
            #with open(fname_xdmf,'w') as xdmf:
            with io.open(fname_xdmf,'w',newline='\n') as xdmf:
                
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
                    dset_name = 'ts_%08d'%ti
                    
                    xdmf_str='''
                             <!-- ============================================================ -->
                             '''
                    
                    xdmf.write(textwrap.indent(textwrap.dedent(xdmf_str.strip('\n')), 6*' '))
                    
                    # ===
                    
                    xdmf_str=f'''
                             <Grid Name="{dset_name}" GridType="Uniform">
                               <Time TimeType="Single" Value="{self.t[ti]:0.8E}"/>
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

# data container interface class for EAS3 (legacy NS3D format)
# ======================================================================

class eas3:
    '''
    Interface class for EAS3 files (legacy binary NS3D output format)
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

# 1D curve fitting
# ======================================================================

class curve_fitter(object):
    '''
    creates a curve fitter instance which is callable afterward
    - includes text for LaTeX and matplotlib
    '''
    def __init__(self, curveType, x, y):
        
        self.curveType = curveType
        
        if (curveType=='linear'):
            '''
            linear function : y=a+b·x
            a straight line on a lin-lin plot
            '''
            def curve(x, a, b):
                return a + b*x
        
        elif (curveType=='power'):
            '''
            power law y=a·x^b
            a straight line on a log-log plot
            '''
            def curve(x, a, b):
                return a*np.power(x,b)
        
        elif (curveType=='power_plus_const'):
            '''
            power law (plus a constant) y = a + b·(x^c)
            no longer a straight line on log-log but allows non-zero y-intercept
            '''
            def curve(x, a, b, c):
                return a + b*np.power(x,c)

        elif (curveType=='power_asymp'):
            '''
            power asymptotic
            '''
            def curve(x, a, b, c, d):
                return a/(b + c*np.power(x,d))
        
        elif (curveType=='exp'):
            '''
            exponential curve y = a + b·exp(c*x)
            '''
            def curve(x, a, b, c):
                return a + b*np.exp(c*x)
        
        elif (curveType=='log'):
            '''
            a straight line on a semi-log (lin-log) plot 
            '''
            def curve(x, a, b):
                return a + b*np.log(x)
        
        else:
            raise ValueError('curveType not recognized : %s'%str(curveType))
        
        self.__curve = curve ## private copy of curve() method
        self.popt, self.pcov = sp.optimize.curve_fit(self.__curve, x, y, maxfev=int(5e5), method='trf')
        
        # ===
        
        if (curveType=='linear'):
            a, b = self.popt
            self.txt = '%0.12e + %0.12e * x'%(a,b)
            self.latex = r'$%0.5f + %0.5f{\cdot}x$'%(a,b)
        
        elif (curveType=='power'):
            a, b = self.popt
            self.txt = '%0.12e * x**%0.12e'%(a,b)
            self.latex = r'$%0.5f{\cdot}x^{%0.5f}$'%(a,b)
        
        elif (curveType=='power_plus_const'):
            a, b, c = self.popt
            self.txt = '%0.12e + %0.12e * x**%0.12e'%(a,b,c)
            self.latex = r'$%0.5f + %0.5f{\cdot}x^{%0.5f}$'%(a,b,c)
        
        elif (curveType=='power_asymp'):
            a, b, c, d = self.popt
            self.txt = '%0.12e / (%0.12e + %0.12e * x**%0.12e)'%(a,b,c,d)
            self.latex = r'$%0.5f / (%0.5f + %0.5f {\cdot} x^{%0.5f})$'%(a,b,c,d)
        
        elif (curveType=='exp'):
            a, b, c = self.popt
            self.txt = '%0.12e + %0.12e * np.exp(%0.12e * x)'%(a,b,c)
            self.latex = r'$%0.5f + %0.5f{\cdot}\text{exp}(%0.5f{\cdot}x)$'%(a,b,c)
        
        elif (curveType=='log'):
            a, b = self.popt
            self.txt = '%0.12e + %0.12e * np.log(x)'%(a,b)
            self.latex = r'$%0.6f + %0.6f{\cdot}\text{ln}(x)$'%(a,b)
        
        else:
            raise NotImplementedError('curveType \'%s\' not recognized'%str(curveType))
    
    def __call__(self, xn):
        return self.__curve(xn, *self.popt)

# boundary layer
# ======================================================================

def Blasius_solution(eta):
    '''
    f·f′′ + 2·f′′′ = 0  ==>  f′′′ = -(1/2)·f·f′′
    BCs: f(0)=0, f′(0)=0, f′(∞)=1
    -----
    for solve_ivp(): d[f′′(η)]/dη = F(f(η), f′(η), f′′(η))
    y=[f,f′,f′′], y′=[ y[1], y[2], (-1/2)·y[0]·y[2] ]
    '''
    
    #Blasius = lambda t,y: [y[1],y[2],-0.5*y[0]*y[2]]
    
    def Blasius(t,y):
        return np.array([y[1], y[2], -0.5*y[0]*y[2]])
    
    if False: ## calculate c0
        import warnings
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
        def eq_root(c0,eta):
            sol = sp.integrate.solve_ivp( fun=Blasius, t_span=[0,eta[-1]], y0=[0,0,c0], t_eval=eta, method='RK45' )
            fp  = np.copy( sol.y[1] ) ## f′
            res = 1. - fp[-1] ## residual for BC: f′(∞)=0
            return res
        eta = np.linspace(0,1e3,int(1e3))
        sol = sp.optimize.fsolve(eq_root, x0=0.332, args=(eta,), xtol=1e-9)
        c0 = sol[0]
        #print('c0 = %0.14f'%c0)
    else:
        c0 = 0.33200221890262
    
    sol = sp.integrate.solve_ivp( fun=Blasius, 
                                  t_span=[0,eta[-1]], 
                                  y0=[ 0, 0, c0 ], 
                                  t_eval=eta, 
                                  method='RK45' )
    
    f, fp, fpp = sol.y ## f′=u/U
    
    ## i_99 = np.abs(fp-0.99).argmin()
    ## eta_99 = eta[i_99]
    ## print('η(f′=%0.14f) = %0.14f'%(fp[i_99],eta_99))
    ## ## η(f′=0.0.99000000002790)=4.91131834911318
    
    return f, fp, fpp

# numerical & grid
# ======================================================================

def interp_2d_structured(x2d_A, y2d_A, x2d_B, y2d_B, data_A):
    '''
    interpolate 2D array 'data_A' from grid A onto grid B, yielding 'data_B'
    --> based on sp.interpolate.griddata()
    --> default 'cubic' interpolation, where NaNs occur, fill with 'nearest'
    '''
    
    #if not isinstance(x2d_A, np.ndarray):
    #    raise ValueError('x2d_A should be a numpy array')
    
    # < need a lot of checks still >
    
    nx,ny = data_A.shape
    
    ## interp2d() --> gets OverflowError for big meshes
    # interpolant = sp.interpolate.interp2d(x2d_A.flatten(),
    #                                       y2d_A.flatten(),
    #                                       data_A.flatten(),
    #                                       kind='linear',
    #                                       copy=True,
    #                                       bounds_error=False,
    #                                       fill_value=np.nan)
    # u_tang_wn = interpolant( x2d_B.flatten(), 
    #                          y2d_B.flatten() )
    
    B_nearest = sp.interpolate.griddata( points=(x2d_A.flatten(), y2d_A.flatten()),
                                         values=data_A.flatten(),
                                         xi=(x2d_B.flatten(), y2d_B.flatten()),
                                         method='nearest',
                                         fill_value=np.nan )
    B_nearest = np.reshape(B_nearest, (nx,ny), order='C')
    
    B_cubic = sp.interpolate.griddata( points=(x2d_A.flatten(), y2d_A.flatten()),
                                       values=data_A.flatten(),
                                       xi=(x2d_B.flatten(), y2d_B.flatten()),
                                       method='cubic',
                                       fill_value=np.nan )
    B_cubic = np.reshape(B_cubic, (nx,ny), order='C')
    
    #nan_indices = np.nonzero(np.isnan(B_cubic))
    #n_nans = np.count_nonzero(np.isnan(B_cubic))
    
    data_B = np.where( np.isnan(B_cubic), B_nearest, B_cubic)
    
    if np.isnan(data_B).any():
        raise AssertionError('interpolated scalar field has NaNs')
    
    return data_B

def fd_coeff_calculator(stencil, d=1, x=None, dx=None):
    '''
    Calculate Finite Difference Coefficients for Arbitrary Stencil
    -----
    stencil : indices of stencil pts e.g. np.array([-2,-1,0,1,2])
    d       : derivative order
    x       : locations of grid points corresponding to stencil indices
    dx      : spacing of grid points in the case of uniform grid
    -----
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    https://web.media.mit.edu/~crtaylor/calculator.html
    -----
    Fornberg B. (1988) Generation of Finite Difference Formulas on
    Arbitrarily Spaced Grids, Mathematics of Computation 51, no. 184 : 699-706.
    http://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf
    '''
    
    stencil = np.asanyarray(stencil)
    
    if not isinstance(stencil, np.ndarray):
        raise ValueError('stencil should be of type np.ndarray')
    if (stencil.ndim!=1):
        raise ValueError('stencil should be 1D')
    if (stencil.shape[0]<2):
        raise ValueError('stencil size should be >=2')
    if (0 not in stencil):
        raise ValueError('stencil does not contain 0')
    if not np.issubdtype(stencil.dtype, np.integer):
        raise ValueError('stencil.dtype not a subdtype of np.integer')
    
    if not isinstance(d, int):
        raise ValueError('d (derivative order) should be of type int')
    if not (d>0):
        raise ValueError('d (derivative order) should be >0')
    
    if (dx is None) and (x is None):
        raise ValueError('one of args \'dx\' or \'x\' should be defined')
    if (dx is not None) and (x is not None):
        raise ValueError('only one of args \'dx\' or \'x\' should be defined')
    if (dx is not None):
        if not isinstance(dx, float):
            raise ValueError('dx should be of type float')
    
    if (x is not None):
        if not isinstance(x, np.ndarray):
            raise ValueError('x should be of type np.ndarray')
        if (x.shape[0] != stencil.shape[0]):
            raise ValueError('x, stencil should have same shape')
        if (not np.all(np.diff(x) > 0.)) and (not np.all(np.diff(x) < 0.)):
            raise AssertionError('x is not monotonically increasing/decreasing')
    
    ## overwrite stencil (int index) to be coordinate array (delta from 0 position)
    
    i0 = np.where(stencil==0)[0][0]
    
    if (x is not None):
        stencil = x - x[i0] 
    
    if (dx is not None):
        stencil = dx * stencil.astype(np.float64)
    
    nn = stencil.shape[0]
    
    dvec = np.zeros( (nn,) , dtype=np.float64 )
    #dvec = np.zeros( (nn,) , dtype=np.longdouble )
    dfac=1
    for i in range(d):
        dfac *= (i+1)
    dvec[d] = dfac
    
    ## increase precision
    #stencil = np.copy(stencil).astype(np.longdouble)
    
    stencil_abs_max         = np.abs(stencil).max()
    stencil_abs_min_nonzero = np.abs(stencil[[ i for i in range(stencil.size) if i!=i0 ]]).min()
    
    '''
    scale/normalize the coordinate stencil (to avoid ill-conditioning)
    - if coordinates are already small/large, the Vandermonde matrix becomes
       HIGHLY ill-conditioned due to row exponents
    - coordinates are normalized here so that smallest absolute non-zero delta coord. is =1
    - RHS vector (dvec) gets normalized too
    - FD coefficients are (theoretically) unaffected
    '''
    normalize_stencil = True
    
    if normalize_stencil:
        stencil /= stencil_abs_min_nonzero
    
    mat = np.zeros( (nn,nn) , dtype=np.float64)
    #mat = np.zeros( (nn,nn) , dtype=np.longdouble)
    for i in range(nn):
        mat[i,:] = np.power( stencil , i )
    
    ## condition_number = np.linalg.cond(mat, p=-2)
    
    # mat_inv = np.linalg.inv( mat )
    # coeffv  = np.dot( mat_inv , dvec )
    
    if normalize_stencil:
        for i in range(nn):
            dvec[i] /= np.power( stencil_abs_min_nonzero , i )
    
    #coeffv = np.linalg.solve(mat, dvec)
    coeffv = sp.linalg.solve(mat, dvec)
    
    return coeffv

def gradient(u, x=None, d=1, axis=0, acc=6, edge_stencil='full', return_coeffs=False):
    '''
    Numerical Gradient Approximation Using Finite Differences
    -----
    - calculates stencil given arbitrary accuracy & derivative order
    - handles non-uniform grids
    - accuracy order is only mathematically valid for:
       - uniform coordinate array
       - inner points which have full central stencil
    - handles N-D numpy arrays (gradient performed over axis denoted by axis arg)
    -----
    u    : input array to perform differentiation upon
    x    : coordinate vector (np.ndarray) OR dx (float) in the case of a uniform grid
    d    : derivative order
    axis : axis along which to perform gradient
    acc  : accuracy order (only fully valid for inner points with central stencil on uniform grid)
    -----
    edge_stencil  : type of edge stencil to use ('half','full')
    return_coeffs : if True, then return stencil & coefficient information
    -----
    # stencil_npts : number of index pts in (central) stencil
    #     --> no longer an input
    #     --> using 'acc' (accuracy order) instead and calculating npts from formula
    #     - stencil_npts=3 : stencil=[      -1,0,+1      ]
    #     - stencil_npts=5 : stencil=[   -2,-1,0,+1,+2   ]
    #     - stencil_npts=7 : stencil=[-3,-2,-1,0,+1,+2,+3]
    #     - edges are filled out with appropriate clipping of central stencil
    -----
    turbx.gradient( u , x , d=1 , acc=2 , edge_stencil='half' , axis=0 )
    ...reproduces...
    np.gradient(u, x, edge_order=1, axis=0)
    
    turbx.gradient( u , x , d=1 , acc=2 , edge_stencil='full' , axis=0 )
    ...reproduces...
    np.gradient(u, x, edge_order=2, axis=0)
    -----
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    https://web.media.mit.edu/~crtaylor/calculator.html
    -----
    Fornberg B. (1988) Generation of Finite Difference Formulas on
    Arbitrarily Spaced Grids, Mathematics of Computation 51, no. 184 : 699-706.
    http://www.ams.org/journals/mcom/1988-51-184/S0025-5718-1988-0935077-0/S0025-5718-1988-0935077-0.pdf
    '''
    
    u    = np.asanyarray(u)
    nd   = u.ndim
    
    if (nd==0):
        raise ValueError('turbx.gradient() requires input that is at least 1D')
    
    axes = tuple(range(nd))
    
    if not isinstance(axis, int):
        raise ValueError('axis should be of type int')
    if (axis not in axes):
        raise ValueError('axis=%i is not valid for array with u.ndim=%s'%(axis,str(u.ndim)))
    
    nx = u.shape[axis] ## size of axis over which gradient will be performed
    
    if (nx<3):
        raise ValueError('nx<3')
    
    if (x is not None):
        if isinstance(x, float):
            if (x<=0.):
                raise ValueError('if x is a float it should be >0.')
        elif isinstance(x, int):
            x = float(x)
        elif isinstance(x, np.ndarray):
            if (x.ndim!=1):
                raise ValueError('x should be 1D if it is of type np.ndarray')
            if (x.shape[0]!=nx):
                raise ValueError('size of x does not match data axis specified')
            if (not np.all(np.diff(x) > 0.)) and (not np.all(np.diff(x) < 0.)):
                    raise AssertionError('x is not monotonically increasing/decreasing')
            
            ## optimization: check if x is actually uniformly spaced, in which case x=Δx
            dx0 = x[1]-x[0]
            if np.all(np.isclose(np.diff(x), dx0, rtol=1e-8)): 
                #print('turbx.gradient() : x arr with x.shape=%s seems like it is actually uniformly spaced. applying x=%0.8e'%(str(x.shape),dx0))
                x = dx0
        
        else:
            raise ValueError('x should be a 1D np.ndarray or float')
    else:
        x = 1. ## if x not provided, assume uniform unit coordinate vector
    
    if isinstance(x, float):
        uniform_grid = True
    elif isinstance(x, np.ndarray):
        uniform_grid = False
    else:
        raise ValueError('turbx.gradient() : this should never happen... check!')
    
    if not isinstance(d, int):
        raise ValueError('d (derivative order) should be of type int')
    if not (d>0):
        raise ValueError('d (derivative order) should be >0')
    
    if not isinstance(acc, int):
        raise ValueError('acc (accuracy order) should be of type int')
    if not (acc>=2):
        raise ValueError('acc (accuracy order) should be >=2')
    if (acc%2!=0):
        raise ValueError('acc (accuracy order) should be an integer multiple of 2')
    
    ## for the d'th derivative with accuracy=acc, the following formula gives the n pts of the (central) stencil
    stencil_npts = 2*math.floor((d+1)/2) - 1 + acc
    
    if not isinstance(stencil_npts, int):
        raise ValueError('stencil_npts must be of type \'int\'')
    if (stencil_npts<3):
        raise ValueError('stencil_npts should be >=3')
    if ((stencil_npts-1)%2 != 0):
        raise ValueError('(stencil_npts-1) should be divisible by 2 (for central stencil)')
    if (stencil_npts > nx):
        raise ValueError('stencil_npts > nx')
    
    if all([ (edge_stencil!='half') , (edge_stencil!='full') ]):
        raise ValueError('edge_stencil=%s not valid. options are: \'full\', \'half\''%str(edge_stencil))
    
    # ===
    
    n_full_central_stencils = nx - stencil_npts + 1
    
    if ( n_full_central_stencils < 5 ):
        print('\nWARNING\n'+72*'-')
        print('n pts with full central stencils = %i (<5)'%n_full_central_stencils)
        #print('nx//3=%i'%(nx//3))
        print('--> consider reducing acc arg (accuracy order)')
        print(72*'-'+'\n')
    
    stencil_width = stencil_npts-1
    sw2           = stencil_width//2
    
    # === build up stencil & coefficients vector
    
    fdc_vec = [] ## vector of finite difference coefficient information
    
    ## left side
    for i in range(0,sw2):
        
        if (edge_stencil=='half'):
            stencil_L = np.arange(-i,sw2+1)
        elif (edge_stencil=='full'):
            stencil_L = np.arange(-i,stencil_width+1)
        else:
            raise ValueError('edge_stencil options are: \'full\', \'half\'')
        
        i_range  = np.arange( 0 , stencil_L.shape[0] )
        
        if uniform_grid:
            fdc = fd_coeff_calculator( stencil_L , d=d , dx=x )
        else:
            fdc = fd_coeff_calculator( stencil_L , d=d , x=x[i_range] )
        
        fdc_vec.append( [ fdc , i_range , stencil_L ] )
    
    ## inner pts
    stencil = np.arange(stencil_npts) - sw2
    if uniform_grid:
        fdc_inner = fd_coeff_calculator( stencil , d=d , dx=x )
    for i in range(sw2,nx-sw2):
        
        i_range  = np.arange(i-sw2,i+sw2+1)
        
        if uniform_grid:
            fdc = fdc_inner
        else:
            fdc = fd_coeff_calculator( stencil , d=d , x=x[i_range] )
        
        fdc_vec.append( [ fdc , i_range , stencil ] )
    
    ## right side
    for i in range(nx-sw2,nx):
        
        if (edge_stencil=='half'):
            stencil_R = np.arange(-sw2,nx-i)
        elif (edge_stencil=='full'):
            stencil_R = np.arange(-stencil_width,nx-i)
        else:
            raise ValueError('edge_stencil options are: \'full\', \'half\'')
        
        i_range  = np.arange( nx-stencil_R.shape[0] , nx )
        
        if uniform_grid:
            fdc = fd_coeff_calculator( stencil_R , d=d , dx=x )
        else:
            fdc = fd_coeff_calculator( stencil_R , d=d , x=x[i_range] )
        
        fdc_vec.append( [ fdc , i_range , stencil_R ] )
    
    # === evaluate gradient
    
    u_ddx = np.zeros_like(u)
    
    if (nd==1): ## 1D
        
        for i in range(len(fdc_vec)):
            fdc, i_range, stencil = fdc_vec[i]
            u_ddx[i] = np.dot( fdc , u[i_range] )
    
    else: ## N-D
        
        ## shift gradient axis to position 0
        u_ddx = np.swapaxes(u_ddx, axis, 0)
        u     = np.swapaxes(u    , axis, 0)
        
        shape_new       = u_ddx.shape
        size_all_but_ax = np.prod(np.array(shape_new)[1:])
        
        ## reshape N-D to 2D (gradient axis is 0, all other axes are flattened on axis=1)
        u_ddx = np.reshape(u_ddx, (nx, size_all_but_ax), order='C')
        u     = np.reshape(u    , (nx, size_all_but_ax), order='C')
        
        # ## slow loop
        # t_start = timeit.default_timer()
        # for i in range(nx):
        #     fdc, i_range, stencil = fdc_vec[i]
        #     for j in range(size_all_but_ax):
        #         u_ddx[i,j] = np.dot( fdc , u[i_range,j] )
        
        ## fast loop
        for i in range(nx):
            fdc, i_range, stencil = fdc_vec[i]
            #u_ = np.ascontiguousarray(u[i_range,:], dtype=u.dtype)
            u_ = u[i_range,:]
            u_ddx[i,:] = np.einsum('ij,i->j', u_, fdc)
        
        ## reshape 2D to N-D
        u_ddx = np.reshape(u_ddx, shape_new, order='C')
        u     = np.reshape(u    , shape_new, order='C')
        
        ## shift gradient axis back to original position
        u_ddx = np.swapaxes(u_ddx, 0, axis)
        u     = np.swapaxes(u    , 0, axis)
    
    if return_coeffs:
        return u_ddx, fdc_vec
    else:
        return u_ddx

def get_metric_tensor_3d(x3d, y3d, z3d, acc=2, edge_stencil='full', **kwargs):
    '''
    compute the grid metric tensor (inverse of grid Jacobian) for a 3D grid
    -----
    Computational Fluid Mechanics and Heat Transfer (2012) Pletcher, Tannehill, Anderson
    p.266-270, 335-337, 652
    '''
    
    verbose = kwargs.get('verbose',False)
    
    if not isinstance(x3d, np.ndarray):
        raise ValueError('x3d should be of type np.ndarray')
    if not isinstance(y3d, np.ndarray):
        raise ValueError('y3d should be of type np.ndarray')
    if not isinstance(z3d, np.ndarray):
        raise ValueError('z3d should be of type np.ndarray')
    
    if (x3d.ndim!=3):
        raise ValueError('x3d should have ndim=3 (xyz)')
    if (y3d.ndim!=3):
        raise ValueError('y3d should have ndim=3 (xyz)')
    if (z3d.ndim!=3):
        raise ValueError('z3d should have ndim=3 (xyz)')
    
    if not (x3d.shape==y3d.shape):
        raise ValueError('x3d.shape!=y3d.shape')
    if not (y3d.shape==z3d.shape):
        raise ValueError('y3d.shape!=z3d.shape')
    
    nx,ny,nz = x3d.shape
    
    ## the 'computational' grid (unit Cartesian)
    ## --> [x_comp,y_comp,z_comp ]= [ξ,η,ζ] = [q1,q2,q3]
    #x_comp = np.arange(nx, dtype=np.float64)
    #y_comp = np.arange(ny, dtype=np.float64)
    #z_comp = np.arange(nz, dtype=np.float64)
    x_comp = 1.
    y_comp = 1.
    z_comp = 1.
    
    # === get Jacobian :: ∂(x,y,z)/∂(q1,q2,q3)
    
    t_start = timeit.default_timer()
    
    dxdx = gradient(x3d, x_comp, axis=0, d=1, acc=acc, edge_stencil=edge_stencil)
    dydx = gradient(y3d, x_comp, axis=0, d=1, acc=acc, edge_stencil=edge_stencil)
    dzdx = gradient(z3d, x_comp, axis=0, d=1, acc=acc, edge_stencil=edge_stencil)
    ##
    dxdy = gradient(x3d, y_comp, axis=1, d=1, acc=acc, edge_stencil=edge_stencil)
    dydy = gradient(y3d, y_comp, axis=1, d=1, acc=acc, edge_stencil=edge_stencil)
    dzdy = gradient(z3d, y_comp, axis=1, d=1, acc=acc, edge_stencil=edge_stencil)
    ##
    dxdz = gradient(x3d, z_comp, axis=2, d=1, acc=acc, edge_stencil=edge_stencil)
    dydz = gradient(y3d, z_comp, axis=2, d=1, acc=acc, edge_stencil=edge_stencil)
    dzdz = gradient(z3d, z_comp, axis=2, d=1, acc=acc, edge_stencil=edge_stencil)
    
    J = np.stack((np.stack((dxdx, dydx, dzdx), axis=3),
                  np.stack((dxdy, dydy, dzdy), axis=3),
                  np.stack((dxdz, dydz, dzdz), axis=3)), axis=4)
    
    t_delta = timeit.default_timer() - t_start
    if verbose: tqdm.write( even_print('get J','%0.3f [s]'%(t_delta,), s=True) )
    
    # === get metric tensor M = J^-1 = ∂(q1,q2,q3)/∂(x,y,z) = ∂(ξ,η,ζ)/∂(x,y,z)
    
    if False: ## method 1
        
        t_start = timeit.default_timer()
        
        M = np.linalg.inv(J)
        
        # M_bak = np.copy(M)
        # for i in range(nx):
        #     for j in range(ny):
        #         for k in range(nz):
        #             M[i,j,k,:,:] = sp.linalg.inv( J[i,j,k,:,:] )
        # np.testing.assert_allclose(M_bak, M, atol=1e-12, rtol=1e-12)
        # print('check passed')
        
        t_delta = timeit.default_timer() - t_start
        if verbose: tqdm.write( even_print('get M','%0.3f [s]'%(t_delta,), s=True) )
    
    if True: ## method 2
        
        if ('M' in locals()):
            M_bak = np.copy(M)
            M = None; del M
        
        t_start = timeit.default_timer()
        
        a = J[:,:,:,0,0]
        b = J[:,:,:,0,1]
        c = J[:,:,:,0,2]
        d = J[:,:,:,1,0]
        e = J[:,:,:,1,1]
        f = J[:,:,:,1,2]
        g = J[:,:,:,2,0]
        h = J[:,:,:,2,1]
        i = J[:,:,:,2,2]
        
        # a = J[:,:,:,0,0]
        # b = J[:,:,:,1,0]
        # c = J[:,:,:,2,0]
        # d = J[:,:,:,0,1]
        # e = J[:,:,:,1,1]
        # f = J[:,:,:,2,1]
        # g = J[:,:,:,0,2]
        # h = J[:,:,:,1,2]
        # i = J[:,:,:,2,2]
        
        I = ( + a*e*i
              + b*f*g
              + c*d*h
              - c*e*g
              - b*d*i
              - a*f*h )
        
        M = np.zeros((nx,ny,nz,3,3), dtype=np.float64)
        M[:,:,:,0,0] = +( dydy * dzdz - dydz * dzdy ) / I ## ξ_x
        M[:,:,:,0,1] = -( dxdy * dzdz - dxdz * dzdy ) / I ## ξ_y
        M[:,:,:,0,2] = +( dxdy * dydz - dxdz * dydy ) / I ## ξ_z
        M[:,:,:,1,0] = -( dydx * dzdz - dydz * dzdx ) / I ## η_x
        M[:,:,:,1,1] = +( dxdx * dzdz - dxdz * dzdx ) / I ## η_y
        M[:,:,:,1,2] = -( dxdx * dydz - dxdz * dydx ) / I ## η_z
        M[:,:,:,2,0] = +( dydx * dzdy - dydy * dzdx ) / I ## ζ_x
        M[:,:,:,2,1] = -( dxdx * dzdy - dxdy * dzdx ) / I ## ζ_y
        M[:,:,:,2,2] = +( dxdx * dydy - dxdy * dydx ) / I ## ζ_z
        
        t_delta = timeit.default_timer() - t_start
        if verbose: tqdm.write( even_print('get M','%0.3f [s]'%(t_delta,), s=True) )
        
        if ('M_bak' in locals()):
            np.testing.assert_allclose(M[:,:,:,0,0], M_bak[:,:,:,0,0], atol=1e-14, rtol=1e-14)
            print('check passed: ξ_x')
            np.testing.assert_allclose(M[:,:,:,0,1], M_bak[:,:,:,0,1], atol=1e-14, rtol=1e-14)
            print('check passed: ξ_y')
            np.testing.assert_allclose(M[:,:,:,0,2], M_bak[:,:,:,0,2], atol=1e-14, rtol=1e-14)
            print('check passed: ξ_z')
            np.testing.assert_allclose(M[:,:,:,1,0], M_bak[:,:,:,1,0], atol=1e-14, rtol=1e-14)
            print('check passed: η_x')
            np.testing.assert_allclose(M[:,:,:,1,1], M_bak[:,:,:,1,1], atol=1e-14, rtol=1e-14)
            print('check passed: η_y')
            np.testing.assert_allclose(M[:,:,:,1,2], M_bak[:,:,:,1,2], atol=1e-14, rtol=1e-14)
            print('check passed: η_z')
            np.testing.assert_allclose(M[:,:,:,2,0], M_bak[:,:,:,2,0], atol=1e-14, rtol=1e-14)
            print('check passed: ζ_x')
            np.testing.assert_allclose(M[:,:,:,2,1], M_bak[:,:,:,2,1], atol=1e-14, rtol=1e-14)
            print('check passed: ζ_y')
            np.testing.assert_allclose(M[:,:,:,2,2], M_bak[:,:,:,2,2], atol=1e-14, rtol=1e-14)
            print('check passed: ζ_z')
            np.testing.assert_allclose(M, M_bak, atol=1e-14, rtol=1e-14)
            print('check passed: M')
    
    return M

def get_metric_tensor_2d(x2d, y2d, acc=2, edge_stencil='full', **kwargs):
    '''
    compute the grid metric tensor (inverse of grid Jacobian) for a 2D grid
    -----
    Computational Fluid Mechanics and Heat Transfer (2012) Pletcher, Tannehill, Anderson
    p.266-270, 335-337, 652
    '''
    
    verbose = kwargs.get('verbose',False)
    
    if not isinstance(x2d, np.ndarray):
        raise ValueError('x2d should be of type np.ndarray')
    if not isinstance(y2d, np.ndarray):
        raise ValueError('y2d should be of type np.ndarray')
    
    if (x2d.ndim!=2):
        raise ValueError('x2d should have ndim=2 (xy)')
    if (y2d.ndim!=2):
        raise ValueError('y2d should have ndim=2 (xy)')
    
    if not (x2d.shape==y2d.shape):
        raise ValueError('x2d.shape!=y2d.shape')
    
    nx,ny = x2d.shape
    
    ## the 'computational' grid (unit Cartesian)
    ## --> [x_comp,y_comp]= [ξ,η] = [q1,q2]
    #x_comp = np.arange(nx, dtype=np.float64)
    #y_comp = np.arange(ny, dtype=np.float64)
    x_comp = 1.
    y_comp = 1.
    
    # === get Jacobian :: ∂(x,y)/∂(q1,q2)
    
    t_start = timeit.default_timer()
    
    dxdx = gradient(x2d, x_comp, axis=0, d=1, acc=acc, edge_stencil=edge_stencil)
    dydx = gradient(y2d, x_comp, axis=0, d=1, acc=acc, edge_stencil=edge_stencil)
    dxdy = gradient(x2d, y_comp, axis=1, d=1, acc=acc, edge_stencil=edge_stencil)
    dydy = gradient(y2d, y_comp, axis=1, d=1, acc=acc, edge_stencil=edge_stencil)
    
    J = np.stack((np.stack((dxdx, dydx), axis=2),
                  np.stack((dxdy, dydy), axis=2)), axis=3)
    
    t_delta = timeit.default_timer() - t_start
    if verbose: tqdm.write( even_print('get J','%0.3f [s]'%(t_delta,), s=True) )
    
    # === get metric tensor M = J^-1 = ∂(q1,q2)/∂(x,y) = ∂(ξ,η)/∂(x,y)
    
    if False: ## method 1
        
        t_start = timeit.default_timer()
        
        M = np.linalg.inv(J)
        
        # M_bak = np.copy(M)
        # M = np.zeros((nx,ny,2,2),dtype=np.float64)
        # for i in range(nx):
        #     for j in range(ny):
        #         M[i,j,:,:] = sp.linalg.inv( J[i,j,:,:] )
        # np.testing.assert_allclose(M_bak, M, atol=1e-12, rtol=1e-12)
        # print('check passed')
        
        t_delta = timeit.default_timer() - t_start
        if verbose: tqdm.write( even_print('get M','%0.3f [s]'%(t_delta,), s=True) )
    
    if True: ## method 2
        
        if ('M' in locals()):
            M_bak = np.copy(M)
            M = None; del M
        
        t_start = timeit.default_timer()
        
        ## Jacobian determinant
        I = dxdx*dydy - dydx*dxdy
        
        # I_bak = np.copy(I)
        # I = None; del I
        # I = np.linalg.det(J)
        # np.testing.assert_allclose(I, I_bak, atol=1e-14, rtol=1e-14)
        # print('check passed')
        
        M = np.zeros((nx,ny,2,2), dtype=np.float64)
        M[:,:,0,0] = +dydy / I ## ξ_x
        M[:,:,0,1] = -dxdy / I ## ξ_y
        M[:,:,1,0] = -dydx / I ## η_x
        M[:,:,1,1] = +dxdx / I ## η_y
        
        t_delta = timeit.default_timer() - t_start
        if verbose: tqdm.write( even_print('get M','%0.3f [s]'%(t_delta,), s=True) )
        
        if ('M_bak' in locals()):
            np.testing.assert_allclose(M[:,:,0,0], M_bak[:,:,0,0], atol=1e-14, rtol=1e-14)
            print('check passed: ξ_x')
            np.testing.assert_allclose(M[:,:,0,1], M_bak[:,:,0,1], atol=1e-14, rtol=1e-14)
            print('check passed: ξ_y')
            np.testing.assert_allclose(M[:,:,1,0], M_bak[:,:,1,0], atol=1e-14, rtol=1e-14)
            print('check passed: η_x')
            np.testing.assert_allclose(M[:,:,1,1], M_bak[:,:,1,1], atol=1e-14, rtol=1e-14)
            print('check passed: η_y')
            np.testing.assert_allclose(M, M_bak, atol=1e-14, rtol=1e-14)
            print('check passed: M')
    
    return M

def get_grid_quality_metrics_2d(x2d, y2d, **kwargs):
    '''
    get 2d grid quality metrics
    -----
    - skew
    - avg diagonal length (not yet implemented)
    -----
    ## https://coreform.com/cubit_help/mesh_generation/mesh_quality_assessment/quadrilateral_metrics.htm
    '''
    
    verbose = kwargs.get('verbose',True)
    
    if not isinstance(x2d, np.ndarray):
        raise ValueError('x2d should be of type np.ndarray')
    if not isinstance(y2d, np.ndarray):
        raise ValueError('y2d should be of type np.ndarray')
    
    if (x2d.ndim!=2):
        raise ValueError('x2d should have ndim=2 (xy)')
    if (y2d.ndim!=2):
        raise ValueError('y2d should have ndim=2 (xy)')
    
    if not (x2d.shape==y2d.shape):
        raise ValueError('x2d.shape!=y2d.shape')
    
    nx,ny = x2d.shape
    xy2d = np.stack((x2d,y2d), axis=-1)
    
    '''
    there are very likely ways to vectorize the functions below
    '''
    
    ds1avg = np.zeros((nx,ny), dtype=np.float64)
    ds2avg = np.zeros((nx,ny), dtype=np.float64)
    
    if verbose: progress_bar = tqdm(total=(nx-1)*(ny-1), ncols=100, desc='grid ds ', leave=False, file=sys.stdout)
    for i in range(nx):
        for j in range(ny):
            
            ## W/E ds
            if (i==0):
                dsW = None
                dsE = sp.linalg.norm( xy2d[i+1,j,:] - xy2d[i,  j,:], ord=2)
                ds1avg[i,j] = dsE
            elif (i==nx-1):
                dsW = sp.linalg.norm( xy2d[i,  j,:] - xy2d[i-1,j,:], ord=2)
                dsE = None
                ds1avg[i,j] = dsW
            else:
                dsW = sp.linalg.norm( xy2d[i,  j,:] - xy2d[i-1,j,:], ord=2)
                dsE = sp.linalg.norm( xy2d[i+1,j,:] - xy2d[i,  j,:], ord=2)
                ds1avg[i,j] = 0.5*(dsW+dsE)
            
            ## S/N ds
            if (j==0):
                dsS = None
                dsN = sp.linalg.norm( xy2d[i,j+1,:] - xy2d[i,  j,:], ord=2)
                ds2avg[i,j] = dsN
            elif (j==ny-1):
                dsS = sp.linalg.norm( xy2d[i,  j,:] - xy2d[i,j-1,:], ord=2)
                dsN = None
                ds2avg[i,j] = dsS
            else:
                dsS = sp.linalg.norm( xy2d[i,  j,:] - xy2d[i,j-1,:], ord=2)
                dsN = sp.linalg.norm( xy2d[i,j+1,:] - xy2d[i,  j,:], ord=2)
                ds2avg[i,j] = 0.5*(dsS+dsN)
            
            if verbose: progress_bar.update()
    if verbose: progress_bar.close()
    
    grid_inner_angle_cosines = np.zeros((nx-1,ny-1,4), dtype=np.float64)
    if verbose: progress_bar = tqdm(total=(nx-1)*(ny-1), ncols=100, desc='grid skew angle', leave=False, file=sys.stdout)
    for i in range(nx-1):
        for j in range(ny-1):
            ## (SW-->NW)::(SW-->SE)
            v1 = xy2d[i,j+1,:] - xy2d[i,j,:]
            v2 = xy2d[i+1,j,:] - xy2d[i,j,:]
            v1mag = sp.linalg.norm(v1, ord=2)
            v2mag = sp.linalg.norm(v2, ord=2)
            grid_inner_angle_cosines[i,j,0] = np.abs(np.dot(v1,v2)/(v1mag*v2mag))
            ## (SE-->SW)::(SE-->NE)
            v1 = xy2d[i,j,:]     - xy2d[i+1,j,:]
            v2 = xy2d[i+1,j+1,:] - xy2d[i+1,j,:]
            v1mag = sp.linalg.norm(v1, ord=2)
            v2mag = sp.linalg.norm(v2, ord=2)
            grid_inner_angle_cosines[i,j,1] = np.abs(np.dot(v1,v2)/(v1mag*v2mag))
            ## (NE-->NW)::(NE-->SE)
            v1 = xy2d[i,j+1,:] - xy2d[i+1,j+1,:]
            v2 = xy2d[i+1,j,:] - xy2d[i+1,j+1,:]
            v1mag = sp.linalg.norm(v1, ord=2)
            v2mag = sp.linalg.norm(v2, ord=2)
            grid_inner_angle_cosines[i,j,2] = np.abs(np.dot(v1,v2)/(v1mag*v2mag))
            ## (NW-->NE)::(NW-->SW)
            v1 = xy2d[i+1,j+1,:] - xy2d[i,j+1,:]
            v2 = xy2d[i,j,:]     - xy2d[i,j+1,:]
            v1mag = sp.linalg.norm(v1, ord=2)
            v2mag = sp.linalg.norm(v2, ord=2)
            grid_inner_angle_cosines[i,j,3] = np.abs(np.dot(v1,v2)/(v1mag*v2mag))
            ##
        if verbose: progress_bar.update()
    if verbose: progress_bar.close()
    
    skew = np.max(grid_inner_angle_cosines, axis=-1)
    
    grid_quality_dict = { 'skew':skew, 'ds1avg':ds1avg, 'ds2avg':ds2avg }
    return grid_quality_dict

# csys
# ======================================================================

def rect_to_cyl(xyz,**kwargs):
    '''
    convert [x,y,<z>] to [θ,r,<z>]
    '''
    
    cx = kwargs.get('cx',0.)
    cy = kwargs.get('cy',0.)
    
    trz = np.zeros_like(xyz)
    
    if (xyz.ndim==1) and (xyz.shape[-1]==2): ## a single point, shape=(2,)
        xx = xyz[0]-cx
        yy = xyz[1]-cy
        trz[0] = np.arctan2(yy,xx)
        trz[1] = np.sqrt(xx**2 + yy**2)
    
    elif (xyz.ndim==1) and (xyz.shape[-1]==3): ## a single point, shape=(3,)
        xx = xyz[0]-cx
        yy = xyz[1]-cy
        trz[0] = np.arctan2(yy,xx)
        trz[1] = np.sqrt(xx**2 + yy**2)
        trz[2] = xyz[2]
    
    elif (xyz.ndim==2) and (xyz.shape[-1]==2): ## a 1D vector of 2D points, shape=(N,2)
        xx = xyz[:,0]-cx
        yy = xyz[:,1]-cy
        trz[:,0] = np.arctan2(yy,xx)
        trz[:,1] = np.sqrt(xx**2 + yy**2)
    
    elif (xyz.ndim==2) and (xyz.shape[-1]==3): ## a 1D vector of 3D points, shape=(N,3)
        xx = xyz[:,0]-cx
        yy = xyz[:,1]-cy
        trz[:,0] = np.arctan2(yy,xx)
        trz[:,1] = np.sqrt(xx**2 + yy**2)
        trz[:,2] = xyz[:,2]
    
    elif (xyz.ndim==3) and (xyz.shape[-1]==2): ## 2D, shape=(nx,ny,2)
        xx    = xyz[:,:,0] - cx
        yy    = xyz[:,:,1] - cy
        trz[:,:,0] = np.arctan2(yy,xx)
        trz[:,:,1] = np.sqrt(xx**2 + yy**2)
    
    elif (xyz.ndim==4) and (xyz.shape[-1]==3): ## 3D, shape=(nx,ny,nz,3)
        xx    = xyz[:,:,:,0] - cx
        yy    = xyz[:,:,:,1] - cy
        trz[:,:,:,0] = np.arctan2(yy,xx)
        trz[:,:,:,1] = np.sqrt(xx**2 + yy**2)
        trz[:,:,:,2] = xyz[:,:,:,2]
    
    else:
        raise ValueError('this input is not supported')
    
    return trz

def cyl_to_rect(trz,**kwargs):
    '''
    convert [θ,r,<z>] to [x,y,<z>]
    '''
    
    cx = kwargs.get('cx',0.)
    cy = kwargs.get('cy',0.)
    
    xyz = np.zeros_like(trz)
    
    if (trz.ndim==1) and (trz.shape[-1]==2): ## a single point, shape=(2,)
        tt = trz[0]
        rr = trz[1]
        xx = rr*np.cos(tt)
        yy = rr*np.sin(tt)
        xyz[0] = xx + cx
        xyz[1] = yy + cy
    
    elif (trz.ndim==1) and (trz.shape[-1]==3): ## a single point, shape=(3,)
        tt = trz[0]
        rr = trz[1]
        xx = rr*np.cos(tt)
        yy = rr*np.sin(tt)
        xyz[0] = xx + cx
        xyz[1] = yy + cy
        xyz[2] = trz[2]
    
    elif (trz.ndim==2) and (trz.shape[-1]==2): ## a 1D vector of 2D points, shape=(N,2)
        tt = trz[:,0]
        rr = trz[:,1]
        xx = rr*np.cos(tt)
        yy = rr*np.sin(tt)
        xyz[:,0] = xx + cx
        xyz[:,1] = yy + cy
    
    elif (trz.ndim==2) and (trz.shape[-1]==3): ## a 1D vector of 3D points, shape=(N,3)
        tt = trz[:,0]
        rr = trz[:,1]
        xx = rr*np.cos(tt)
        yy = rr*np.sin(tt)
        xyz[:,0] = xx + cx
        xyz[:,1] = yy + cy
        xyz[:,2] = trz[:,2]
    
    elif (trz.ndim==3) and (trz.shape[-1]==2): ## 2D, shape=(nx,ny,2)
        tt = trz[:,:,0]
        rr = trz[:,:,1]
        xx = rr*np.cos(tt)
        yy = rr*np.sin(tt)
        xyz[:,:,0] = xx + cx
        xyz[:,:,1] = yy + cy
    
    elif (trz.ndim==4) and (trz.shape[-1]==3): ## 3D, shape=(nx,ny,nz,3)
        tt = trz[:,:,:,0]
        rr = trz[:,:,:,1]
        xx = rr*np.cos(tt)
        yy = rr*np.sin(tt)
        xyz[:,:,:,0] = xx + cx
        xyz[:,:,:,1] = yy + cy
        xyz[:,:,:,2] = trz[:,:,:,2]
    
    else:
        raise ValueError('this input is not supported')
    
    return xyz

# post-processing : vector & tensor ops
# ======================================================================

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
    
    print('get_grad() has been deprecated --> needs to be updated with turbx.gradient()')
    sys.exit(1)
    
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
        
        dadx = np.gradient(a, x, edge_order=2, axis=0)
        if verbose: progress_bar.update()
        dady = np.gradient(a, y, edge_order=2, axis=1)
        if verbose: progress_bar.update()
        dadz = np.gradient(a, z, edge_order=2, axis=2)
        if verbose: progress_bar.update()
        dbdx = np.gradient(b, x, edge_order=2, axis=0)
        if verbose: progress_bar.update()
        dbdy = np.gradient(b, y, edge_order=2, axis=1)
        if verbose: progress_bar.update()
        dbdz = np.gradient(b, z, edge_order=2, axis=2)
        if verbose: progress_bar.update()
        dcdx = np.gradient(c, x, edge_order=2, axis=0)
        if verbose: progress_bar.update()
        dcdy = np.gradient(c, y, edge_order=2, axis=1)
        if verbose: progress_bar.update()
        dcdz = np.gradient(c, z, edge_order=2, axis=2)
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
    
    print('get_curl() has been deprecated --> needs to be updated with turbx.gradient()')
    sys.exit(1)
    
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
        
        #dadx = np.gradient(a, x, edge_order=2, axis=0)
        #if verbose: progress_bar.update()
        dady = np.gradient(a, y, edge_order=2, axis=1)
        if verbose: progress_bar.update()
        dadz = np.gradient(a, z, edge_order=2, axis=2)
        if verbose: progress_bar.update()
        dbdx = np.gradient(b, x, edge_order=2, axis=0)
        if verbose: progress_bar.update()
        #dbdy = np.gradient(b, y, edge_order=2, axis=1)
        #if verbose: progress_bar.update()
        dbdz = np.gradient(b, z, edge_order=2, axis=2)
        if verbose: progress_bar.update()
        dcdx = np.gradient(c, x, edge_order=2, axis=0)
        if verbose: progress_bar.update()
        dcdy = np.gradient(c, y, edge_order=2, axis=1)
        if verbose: progress_bar.update()
        #dcdz = np.gradient(c, z, edge_order=2, axis=2)
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

# post-processing : spectral, statistical
# ======================================================================

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

def ccor(ui,uj,**kwargs):
    '''
    normalized cross-correlation
    '''
    if (ui.ndim!=1):
        raise AssertionError('ui.ndim!=1')
    if (uj.ndim!=1):
        raise AssertionError('uj.ndim!=1')
    ##
    mode     = kwargs.get('mode','full')
    get_lags = kwargs.get('get_lags',False)
    ##
    lags = sp.signal.correlation_lags(ui.shape[0], uj.shape[0], mode=mode)
    ccor = sp.signal.correlate(ui, uj, mode=mode, method='direct')
    norm = np.sqrt( np.sum( ui**2 ) ) * np.sqrt( np.sum( uj**2 ) )
    ##
    if (norm==0.):
        #ccor = np.ones((lags.shape[0],), dtype=ui.dtype)
        ccor = np.zeros((lags.shape[0],), dtype=ui.dtype)
    else:
        ccor /= norm
    ##
    if get_lags:
        return lags, ccor
    else:
        return ccor

def ccor_naive(u,v,**kwargs):
    '''
    normalized cross-correlation (naive version)
    '''
    if (u.ndim!=1):
        raise AssertionError('u.ndim!=1')
    if (v.ndim!=1):
        raise AssertionError('v.ndim!=1')
    
    ii = np.arange(u.shape[0],dtype=np.int32)
    jj = np.arange(v.shape[0],dtype=np.int32)
    
    ## lags (2D)
    ll     = np.stack(np.meshgrid(ii,jj,indexing='ij'), axis=-1)
    ll     = ll[:,:,0] - ll[:,:,1]
    
    ## lags (1D)
    lmin   = ll.min()
    lmax   = ll.max()
    n_lags = lmax-lmin+1
    lags   = np.arange(lmin,lmax+1)
    
    uu, vv = np.meshgrid(u,v,indexing='ij')
    uv     = np.stack((uu,vv), axis=-1)
    uvp    = np.prod(uv,axis=-1)
    
    c=-1
    R = np.zeros(n_lags, dtype=np.float64)
    for lag in lags:
        c+=1
        X = np.where(ll==lag)
        N = X[0].shape[0]
        R_ = np.sum(uvp[X]) / ( np.sqrt(np.sum(u**2)) * np.sqrt(np.sum(v**2)) )
        R[c] = R_
    return lags, R

# binary I/O
# ======================================================================

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
# ======================================================================

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
    print/return a fixed width message
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
# ======================================================================

def set_mpl_env(**kwargs):
    '''
    Setup the matplotlib environment
    --------------------------------
    
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
    font     = kwargs.get('font',None)
    
    ## mpl.rcParams.update(mpl.rcParamsDefault) ## reset rcparams to defaults
    
    if darkMode:
        mpl.style.use('dark_background') ## dark mode
    else:
        mpl.style.use('default')
    
    if useTex:
        
        ## 'Text rendering with LaTeX'
        ## https://matplotlib.org/stable/tutorials/text/usetex.html
        
        mpl.rcParams['text.usetex'] = True
        #mpl.rcParams['pgf.texsystem'] = 'xelatex' ## 'xelatex', 'lualatex', 'pdflatex' --> xelatex seems to be fastest
        
        preamble_opts = [ r'\usepackage[T1]{fontenc}',
                          #r'\usepackage[utf8]{inputenc}',
                          r'\usepackage{amsmath}', 
                          r'\usepackage{amsfonts}',
                          #r'\usepackage{amssymb}',
                          r'\usepackage{gensymb}', ## Generic symbols 
                          r'\usepackage{xfrac}',
                          #r'\usepackage{nicefrac}',
                          ]
        
        if (font==None): ## default
            mpl.rcParams['font.family']= 'serif'
            mpl.rcParams['font.serif'] = 'Computer Modern Roman'
        
        elif (font=='IBM Plex Sans') or (font=='IBM Plex') or (font=='IBM') or (font=='ibm'):
            preamble_opts +=  [ r'\usepackage{plex-sans}', ## IBM Plex Sans
                                r'\renewcommand{\familydefault}{\sfdefault}', ## sans as default family
                                r'\renewcommand{\seriesdefault}{c}', ## condensed {*} as default series
                                r'\usepackage[italic]{mathastext}', ## use default font in math mode
                              ]
        
        elif (font=='times') or (font=='Times') or (font=='Times New Roman'):
            preamble_opts +=  [ r'\usepackage{txfonts}' ] ## Times-like fonts mathtext symbols
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif']  = 'Times'
        
        elif (font=='lmodern') or (font=='Latin Modern') or (font=='Latin Modern Roman') or (font=='lmr'):
            preamble_opts +=  [ r'\usepackage{lmodern}' ]
        
        elif (font=='Palatino') or (font=='palatino'):
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif']  = 'Palatino'
        
        elif (font=='Helvetica') or (font=='helvetica'):
            mpl.rcParams['font.family']      = 'sans-serif'
            mpl.rcParams['font.sans-serif']  = 'Helvetica'
            preamble_opts +=  [ r'\renewcommand{\familydefault}{\sfdefault}', ## sans as default family
                                r'\usepackage[italic]{mathastext}', ## use default font in math mode
                              ]
        
        elif (font=='Avant Garde'):
            mpl.rcParams['font.family']     = 'sans-serif'
            mpl.rcParams['font.sans-serif'] = 'Avant Garde'
            preamble_opts +=  [ r'\renewcommand{\familydefault}{\sfdefault}', ## sans as default family
                                r'\usepackage[italic]{mathastext}', ## use default font in math mode
                              ]
        
        elif (font=='Computer Modern Roman') or (font=='Computer Modern') or (font=='CMR') or (font=='cmr'):
            mpl.rcParams['font.family'] = 'serif'
            mpl.rcParams['font.serif']  = 'Computer Modern Roman'
        
        else:
            raise ValueError('font=%s not a valid option'%str(font))
        
        ## make preamble string
        mpl.rcParams['text.latex.preamble'] = '\n'.join(preamble_opts)
    
    else: ## use OpenType (OTF) / TrueType (TTF) fonts (and no TeX rendering)
        
        mpl.rcParams['text.usetex'] = False
        
        ## Register OTF/TTF Fonts (only necessary once)
        if False:
            # === register (new) fonts : Windows --> done automatically if you delete ~/.cache/matplotlib/fontlist-v330.json
            ##mpl.font_manager.findSystemFonts(fontpaths='C:/Windows/Fonts', fontext='ttf')
            #mpl.font_manager.findSystemFonts(fontpaths='C:/Users/'+os.path.expandvars('%USERNAME%')+'/AppData/Local/Microsoft/Windows/Fonts', fontext='ttf')
            mpl.font_manager.findSystemFonts(fontpaths=mpl.font_manager.win32FontDirectory(), fontext='ttf')
            
            # === register (new) fonts : Linux / WSL2 --> done automatically if you delete C:/Users/%USERNAME%/.matplotlib/fontlist-v330.json
            mpl.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
            mpl.font_manager.findSystemFonts(fontpaths=None, fontext='otf')
        
        ## example: list all TTF font properties
        if False:
            fonts = mpl.font_manager.fontManager.ttflist
            #fonts = [f for f in fonts if all([('IBM' in f.name),('Condensed' in f.name)])] ## filter list
            for f in fonts:
                print(f.name)
                print(Path(f.fname).stem)
                print('weight  : %s'%str(f.weight))
                print('style   : %s'%str(f.style))
                print('stretch : %s'%str(f.stretch))
                print('variant : %s'%str(f.variant))
                print('-----'+'\n')
        
        ## get list of names of all registered fonts
        if (font is not None):
            try:
                if hasattr(mpl.font_manager,'get_font_names'):
                    ## Matplotlib >3.6.X
                    fontnames = mpl.font_manager.get_font_names()
                elif hasattr(mpl.font_manager,'get_fontconfig_fonts'):
                    ## Matplotlib <=3.5.X
                    fontlist = mpl.font_manager.get_fontconfig_fonts()
                    fontnames = sorted(list(set([mpl.font_manager.FontProperties(fname=fname).get_name() for fname in fontlist])))
                else:
                    fontnames = None
            except:
                fontnames = None
        
        # === TTF/OTF fonts (when NOT using LaTeX rendering)
        
        if (font==None):
            pass ## do nothing, use system / matplotlib default font
        
        ## IBM Plex Sans
        elif (font=='IBM Plex Sans') or (font=='IBM Plex') or (font=='IBM') or (font=='ibm'):
            
            if (fontnames is not None) and ('IBM Plex Sans Condensed' in fontnames):
                
                ## condensed
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
                
                ## regular --> too wide
                # mpl.rcParams['font.family'] = 'IBM Plex Sans'
                # mpl.rcParams['font.weight'] = '400' ## 'light'
                # #mpl.rcParams['font.stretch'] = 'normal' ## always 'normal' for family
                # mpl.rcParams['mathtext.fontset'] = 'custom'
                # mpl.rcParams['mathtext.default'] = 'it'
                # mpl.rcParams['mathtext.rm'] = 'IBM Plex Sans:regular'
                # mpl.rcParams['mathtext.it'] = 'IBM Plex Sans:italic:regular'
                # mpl.rcParams['mathtext.bf'] = 'IBM Plex Sans:bold'
                pass
            
            else:
                #print('font not found: \'IBM Plex Sans Condensed\'')
                pass
        
        ## Latin Modern Roman (lmodern in LaTeX, often used)
        elif (font=='lmodern') or (font=='Latin Modern') or (font=='Latin Modern Roman') or (font=='lmr'):
            if (fontnames is not None) and ('Latin Modern Roman' in fontnames):
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
            else:
                #print('font not found: \'Latin Modern Roman\'')
                pass
        
        ## Times New Roman
        elif (font=='times') or (font=='Times') or (font=='Times New Roman'):
            if (fontnames is not None) and ('Times New Roman' in fontnames):
                mpl.rcParams['font.family'] = 'Times New Roman'
                mpl.rcParams['font.weight'] = 'normal'
                mpl.rcParams['mathtext.fontset'] = 'custom'
                mpl.rcParams['mathtext.default'] = 'it'
                mpl.rcParams['mathtext.rm'] = 'Times New Roman:normal'
                mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
                mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'
            else:
                #print('font not found: \'Times New Roman\'')
                pass
        
        ## Computer Modern (LaTeX default)
        elif (font=='Computer Modern Roman') or (font=='CMU Serif') or (font=='Computer Modern') or (font=='CMR') or (font=='cmr'):
            if (fontnames is not None) and ('CMU Serif' in fontnames):
                mpl.rcParams['font.family'] = 'CMU Serif'
                mpl.rcParams['font.weight'] = 'regular'
                mpl.rcParams['font.style'] = 'normal'
                mpl.rcParams['mathtext.fontset'] = 'custom'
                mpl.rcParams['mathtext.default'] = 'it'
                mpl.rcParams['mathtext.rm'] = 'CMU Serif:regular'
                mpl.rcParams['mathtext.it'] = 'CMU Serif:italic:regular'
                mpl.rcParams['mathtext.bf'] = 'CMU Serif:bold'
            else:
                #print('font not found: \'Times New Roman\'')
                pass
        
        else:
            raise ValueError('font=%s not a valid option'%str(font))
        
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
    
    ## ## list all options
    ## print(mpl.rcParams.keys())
    
    fontsize = 10
    axesAndTickWidth = 0.5
    
    mpl.rcParams['figure.figsize'] = 4, 4/(16/9)
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
    mpl.rcParams['legend.borderpad'] = 0.5
    mpl.rcParams['legend.framealpha'] = 1.0
    mpl.rcParams['legend.edgecolor']  = 'inherit'
    mpl.rcParams['legend.handlelength'] = 1.0
    mpl.rcParams['legend.handletextpad'] = 0.4
    mpl.rcParams['legend.borderaxespad'] = 0.7
    mpl.rcParams['legend.columnspacing'] = 0.5
    mpl.rcParams['legend.fancybox'] = False
    
    return

def colors_table():
    '''
    a table of color hues
    -----
    clrs = colors_table()
    red = clrs['red'][9]
    -----
    https://yeun.github.io/open-color/
    https://twitter.com/nprougier/status/1323575342204936192
    https://github.com/rougier/scientific-visualization-book/blob/master/code/colors/open-colors.py
    '''
    
    colors = {  "gray"   :  { 0: '#f8f9fa', 1: '#f1f3f5', 2: '#e9ecef', 3: '#dee2e6', 4: '#ced4da', 
                              5: '#adb5bd', 6: '#868e96', 7: '#495057', 8: '#343a40', 9: '#212529', },
                "red"    :  { 0: '#fff5f5', 1: '#ffe3e3', 2: '#ffc9c9', 3: '#ffa8a8', 4: '#ff8787', 
                              5: '#ff6b6b', 6: '#fa5252', 7: '#f03e3e', 8: '#e03131', 9: '#c92a2a', },
                "pink"   :  { 0: '#fff0f6', 1: '#ffdeeb', 2: '#fcc2d7', 3: '#faa2c1', 4: '#f783ac',
                              5: '#f06595', 6: '#e64980', 7: '#d6336c', 8: '#c2255c', 9: '#a61e4d', },
                "grape"  :  { 0: '#f8f0fc', 1: '#f3d9fa', 2: '#eebefa', 3: '#e599f7', 4: '#da77f2',
                              5: '#cc5de8', 6: '#be4bdb', 7: '#ae3ec9', 8: '#9c36b5', 9: '#862e9c', },
                "violet" :  { 0: '#f3f0ff', 1: '#e5dbff', 2: '#d0bfff', 3: '#b197fc', 4: '#9775fa',
                              5: '#845ef7', 6: '#7950f2', 7: '#7048e8', 8: '#6741d9', 9: '#5f3dc4', },
                "indigo" :  { 0: '#edf2ff', 1: '#dbe4ff', 2: '#bac8ff', 3: '#91a7ff', 4: '#748ffc', 
                              5: '#5c7cfa', 6: '#4c6ef5', 7: '#4263eb', 8: '#3b5bdb', 9: '#364fc7', },
                "blue"   :  { 0: '#e7f5ff', 1: '#d0ebff', 2: '#a5d8ff', 3: '#74c0fc', 4: '#4dabf7',
                              5: '#339af0', 6: '#228be6', 7: '#1c7ed6', 8: '#1971c2', 9: '#1864ab', },
                "cyan"   :  { 0: '#e3fafc', 1: '#c5f6fa', 2: '#99e9f2', 3: '#66d9e8', 4: '#3bc9db',
                              5: '#22b8cf', 6: '#15aabf', 7: '#1098ad', 8: '#0c8599', 9: '#0b7285', },
                "teal"   :  { 0: '#e6fcf5', 1: '#c3fae8', 2: '#96f2d7', 3: '#63e6be', 4: '#38d9a9',
                              5: '#20c997', 6: '#12b886', 7: '#0ca678', 8: '#099268', 9: '#087f5b', },
                "green"  :  { 0: '#ebfbee', 1: '#d3f9d8', 2: '#b2f2bb', 3: '#8ce99a', 4: '#69db7c',
                              5: '#51cf66', 6: '#40c057', 7: '#37b24d', 8: '#2f9e44', 9: '#2b8a3e', },
                "lime"   :  { 0: '#f4fce3', 1: '#e9fac8', 2: '#d8f5a2', 3: '#c0eb75', 4: '#a9e34b',
                              5: '#94d82d', 6: '#82c91e', 7: '#74b816', 8: '#66a80f', 9: '#5c940d', },
                "yellow" :  { 0: '#fff9db', 1: '#fff3bf', 2: '#ffec99', 3: '#ffe066', 4: '#ffd43b',
                              5: '#fcc419', 6: '#fab005', 7: '#f59f00', 8: '#f08c00', 9: '#e67700', },
                "orange" :  { 0: '#fff4e6', 1: '#ffe8cc', 2: '#ffd8a8', 3: '#ffc078', 4: '#ffa94d',
                              5: '#ff922b', 6: '#fd7e14', 7: '#f76707', 8: '#e8590c', 9: '#d9480f', }, }
    
    return colors

def get_Lch_colors(hues,**kwargs):
    '''
    given a list of hues [0-360], chroma & luminance, return
        colors in hex (html) or rgb format
    -----
    Lch color picker : https://css.land/lch/
    '''
    
    c = kwargs.get('c',110) ## chroma
    L = kwargs.get('L',65) ## luminance
    fmt = kwargs.get('fmt','rgb') ## output format : hex or rgb tuples
    test_plot = kwargs.get('test_plot',False) ## plot to test colors
    
    colors_rgb=[]
    colors_Lab=[]
    for h in hues:
        hX = h*np.pi/180
        LX,a,b = skimage.color.lch2lab([L,c,hX])
        colors_Lab.append([LX,a,b])
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
        raise NameError('fmt=%s not a valid option'%str(fmt))
    
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
    adjust the (h,s,v) values of a list of html color codes (#XXXXXX)
    --> if single #XXXXXX is passed, returns single
    --> margin : adjust proportional to available margin
    '''
    margin = kwargs.get('margin',False)
    
    if isinstance(hex_list, str):
        single=True
        hex_list = [hex_list]
    else:
        single=False
    
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

def nice_log_labels(L):
    '''
    nice compact axis labels for log plots
    '''
    L_out=[]
    for l in L:
        if (l<0.001):
            a = '%0.0e'%l
            a = a.replace('e-0','e-')
            L_out.append(a)
        elif (l>=0.001) and (l<0.01):
            a = '%0.0e'%l
            a = a.replace('e-0','e-')
            L_out.append(a)
        elif (l>=0.01) and (l<0.1):
            a = '%0.0e'%l
            a = a.replace('e-0','e-')
            L_out.append(a)
        elif (l>=0.1) and (l<1):
            L_out.append('%0.1f'%l)
        elif (l>=1):
            L_out.append('%i'%l)
        else:
            print(l)
            sys.exit('uh-oh')
    return L_out

def fig_trim_y(fig, list_of_axes, **kwargs):
    '''
    trims the figure in (y) / height dimension
    - typical use case : single equal aspect figure needs to be scooted / trimmed
    '''
    
    offset_px = kwargs.get('offset_px',10)
    dpi_out   = kwargs.get('dpi',None) ## this can be used to make sure output png px dims is divisible by N
    if (dpi_out is None):
        dpi_out = fig.dpi
    
    fig_px_x, fig_px_y = fig.get_size_inches()*fig.dpi
    #print('fig size px : %i %i'%(fig_px_x, fig_px_y))
    transFigInv = fig.transFigure.inverted()
    mainAxis = list_of_axes[0]
    ##
    x0,  y0,  dx,  dy  = mainAxis.get_position().bounds
    x0A, y0A, dxA, dyA = mainAxis.get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=False).bounds ### pixel values of the axis tightbox
    #print('x0A, y0A, dxA, dyA : %0.2f %0.2f %0.2f %0.2f'%(x0A, y0A, dxA, dyA))
    dy_pct = dyA / fig_px_y #; print('dy_pct : %0.6f' % dy_pct)
    x0A, y0A = transFigInv.transform_point([x0A, y0A])
    dxA, dyA = transFigInv.transform_point([dxA, dyA])
    #y_shift = 1.0 - (y0A+dyA)
    y_shift = y0A
    ##
    w = fig.get_figwidth()
    h = fig.get_figheight()
    w_inch_nom = w
    h_inch_nom = h*1.08*dy_pct ## cropped height [in]
    w_px_nom   = w_inch_nom * dpi_out
    h_px_nom   = h_inch_nom * dpi_out
    px_base    = 4
    h_px       = math.ceil(h_px_nom/px_base)*px_base ## make sure height in px divisible by N (video encoding)
    w_px       = int(round(w_px_nom))
    w_inch     = w_px / dpi_out
    h_inch     = h_px / dpi_out
    ##
    fig.set_size_inches(w_inch,h_inch,forward=True)
    fig_px_x, fig_px_y = fig.get_size_inches()*dpi_out # ; print('fig size px : %0.6f %0.6f'%(fig_px_x, fig_px_y))
    w_adj = fig.get_figwidth()
    h_adj = fig.get_figheight()
    ## do shift
    for axis in list_of_axes:
        x0, y0, dx, dy  = axis.get_position().bounds
        x0n = x0
        y0n = y0-y_shift+(offset_px/fig_px_y)
        dxn = dx
        dyn = dy
        axis.set_position([x0n,y0n*(h/h_adj),dxn,dyn*(h/h_adj)])
    return

def fig_trim_x(fig, list_of_axes, **kwargs):
    '''
    trims the figure in (x) / width dimension
    - typical use case : single equal aspect figure needs to be scooted / trimmed
    '''
    
    offset_px = kwargs.get('offset_px',10)
    dpi_out   = kwargs.get('dpi',None) ## this is used to make sure OUTPUT png px dims are divisible by N
    if (dpi_out is None):
        dpi_out = fig.dpi
    
    fig_px_x, fig_px_y = fig.get_size_inches()*dpi_out
    #print('fig size px : %i %i'%(fig_px_x, fig_px_y))
    transFigInv = fig.transFigure.inverted()
    
    w = fig.get_figwidth()
    h = fig.get_figheight()
    #print('w, h : %0.2f %0.2f'%(w, h))
    
    nax = len(list_of_axes)
    
    ax_tb_pct = np.zeros( (nax,4) , dtype=np.float64 )
    ax_tb_px  = np.zeros( (nax,4) , dtype=np.float64 )
    for ai, axis in enumerate(list_of_axes):
        
        ## percent values of the axis tightbox
        x0, y0, dx, dy  = axis.get_position().bounds
        #print('x0, y0, dx, dy : %0.2f %0.2f %0.2f %0.2f'%(x0, y0, dx, dy))
        ax_tb_pct[ai,:] = np.array([x0, y0, dx, dy])
        
        ## pixel values of the axis tightbox
        x0, y0, dx, dy = axis.get_tightbbox(fig.canvas.get_renderer(), call_axes_locator=True).bounds
        #print('x0, y0, dx, dy : %0.2f %0.2f %0.2f %0.2f'%(x0, y0, dx, dy))
        axis_tb_px  = np.array([x0, y0, dx, dy])
        axis_tb_px *= (dpi_out/fig.dpi) ## scale by dpi ratio [png:screen]
        ax_tb_px[ai,:] = axis_tb_px
    
    ## current width of (untrimmed) margins (in [x])
    marg_R_px = fig_px_x - (ax_tb_px[:,0] + ax_tb_px[:,2]).max()
    marg_L_px = ax_tb_px[:,0].min()
    #print('marg_L_px : %0.2f'%(marg_L_px,))
    #print('marg_R_px : %0.2f'%(marg_R_px,))
    
    ## n pixels to move all axes left by (fig canvas is 'trimmed' from right)
    x_shift_px = marg_L_px - offset_px
    #print('x_shift_px : %0.2f'%(x_shift_px,))
    
    ## get new canvas size
    ## make sure height in px divisible by N (important for video encoding)
    px_base = 8
    w_px    = fig_px_x - marg_L_px - marg_R_px + 2*offset_px
    w_px    = math.ceil(w_px/px_base)*px_base
    #w_px   += 1*px_base ## maybe helpful in case labels have \infty etc., where get_position() slightly underestimates
    h_px    = fig_px_y
    #print('w_px, h_px : %0.2f %0.2f'%(w_px, h_px))
    w_inch  = w_px / dpi_out
    h_inch  = h_px / dpi_out
    
    ## get shifted axis bound values
    for ai, axis in enumerate(list_of_axes):
        x0, y0, dx, dy = axis.get_position().bounds
        x0n = x0 - x_shift_px/fig_px_x
        y0n = y0
        dxn = dx
        dyn = dy
        ax_tb_pct[ai,:] = np.array([x0n, y0n, dxn, dyn])
        #print('x0n, y0n, dxn, dyn : %0.4f %0.4f %0.4f %0.4f'%(x0n, y0n, dxn, dyn))
    
    ## resize canvas
    fig.set_size_inches( w_inch, h_inch, forward=True )
    fig_px_x, fig_px_y = fig.get_size_inches()*dpi_out
    w_adj = fig.get_figwidth()
    h_adj = fig.get_figheight()
    #print('w_adj, h_adj : %0.2f %0.2f'%(w_adj, h_adj))
    #print('w_adj, h_adj : %0.2f %0.2f'%(w_adj*dpi_out, h_adj*dpi_out))
    
    ## do shift
    for ai, axis in enumerate(list_of_axes):
        x0n, y0n, dxn, dyn = ax_tb_pct[ai,:]
        axis.set_position( [ x0n*(w/w_adj) , y0n , dxn*(w/w_adj) , dyn ] )
    
    return

def axs_grid_compress(fig,axs,**kwargs):
    '''
    compress ax grid
    '''
    dim = kwargs.get('dim',1)
    offset_px = kwargs.get('offset_px',5)
    transFigInv = fig.transFigure.inverted()
    
    ## on screen pixel size
    fig_px_x, fig_px_y = fig.get_size_inches()*fig.dpi
    #print('fig size px : %0.3f %0.3f'%(fig_px_x, fig_px_y))
    
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
    transFig    = fig.transFigure
    
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
    convert python/matplotlib cmap object to JSON for Paraview
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
    
    ## output .json formatted ascii file
    #f = open(fname,'w')
    f = io.open(fname,'w',newline='\n')
    
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
# ======================================================================

if __name__ == '__main__':
    
    #mpl.use('Agg')
    mpl.use('GTK3Agg')
    
    comm    = MPI.COMM_WORLD
    rank    = comm.Get_rank()
    n_ranks = comm.Get_size()
    
    # === plotting env
    darkMode = True
    set_mpl_env(useTex=False, darkMode=darkMode, font='ibm')
    save_pdf = False
    
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
    
    figsize = (4,4/(16/9))
    dpi = 300
    fontsize_anno = 6
    fontsize_lgnd = 6
    
    # ===
    
    #chunk_kb = 4*1024 ## 4 [MB]
    chunk_kb = 1*1024 ## 1 [MB]
    
    if True: ## make test data (curvilinear)
        #with cgd('test.h5', 'w', force=True, driver='mpio', comm=comm) as f1:
        with cgd('test.h5', 'w', force=True) as f1:
            #f1.make_test_file(rx=2, ry=2, rz=2, nx=30*4, ny=40*4, nz=50*4, nt=3, chunk_kb=chunk_kb)
            f1.make_test_file(nx=30*4, ny=40*4, nz=50*4, nt=3, chunk_kb=chunk_kb)
            f1.make_xdmf()
    
    if True: ## λ-2 (curvilinear)
        #with cgd('test.h5', 'a', force=True, driver='mpio', comm=comm) as f1:
        with cgd('test.h5', 'a') as f1:
            f1.calc_lambda2(save_Q=True, save_lambda2=True, rt=n_ranks, chunk_kb=chunk_kb)
            f1.make_xdmf()
    
    if False: ## make test data (ABC flow)
        with rgd('abc_flow.h5', 'w', force=True, driver='mpio', comm=comm, libver='latest') as f1:
            f1.populate_abc_flow(rx=2, ry=2, rz=2, nx=100, ny=100, nz=100, nt=100)
            f1.make_xdmf()
    
    if False: ## λ-2
        with rgd('abc_flow.h5','a', verbose=False, driver='mpio', comm=comm, libver='latest') as f1:
            f1.calc_lambda2(hiOrder=False, save_Q=True, save_lambda2=True, rt=n_ranks)
            f1.make_xdmf()
    
    MPI.Finalize()