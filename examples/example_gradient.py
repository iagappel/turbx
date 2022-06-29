#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '..')
import turbx

import numpy as np
import scipy as sp
#from scipy import ndimage, signal
import math

import matplotlib as mpl
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import cmocean, cmasher, colorcet

#import pathlib
#from pathlib import Path, PurePosixPath

# ======================================================================

def test_func_1D_01(x,d=1):
    '''
    analytical 1D test function & derivatives
    '''
    if (d==0):
        u = -np.cos(x) - (3/10)*np.cos(10/3*x)
    elif (d==1):
        u = +np.sin(x) + (30/30)*np.sin(10/3*x)
    elif (d==2):
        u = +np.cos(x) + (300/90)*np.cos(10/3*x)
    elif (d==3):
        u = -np.sin(x) - (3000/270)*np.sin(10/3*x)
    elif (d==4):
        u = -np.cos(x) - (30000/810)*np.cos(10/3*x)
    else:
        raise ValueError('d=%i not implemented'%d)
    
    return u

def test_func_2D_01(xx,yy,d=1,axis=0):
    '''
    analytical 2D test function & derivatives
    '''
    if (axis==0):
        if (d==0):
            uu = np.sin(xx) * np.cos(yy)
        elif (d==1):
            uu = np.cos(xx) * np.cos(yy)
        elif (d==2):
            uu = np.sin(xx) * -np.cos(yy)
        elif (d==3):
            uu = -np.cos(xx) * np.cos(yy)
        elif (d==4):
            uu = np.sin(xx) * np.cos(yy)
        else:
            raise ValueError('d=%i not implemented'%d)
    elif (axis==1):
        if (d==0):
            uu = np.sin(xx) * np.cos(yy)
        elif (d==1):
            uu = -np.sin(xx) * np.sin(yy)
        elif (d==2):
            uu = np.sin(xx) * -np.cos(yy)
        elif (d==3):
            uu = np.sin(xx) * np.sin(yy)
        elif (d==4):
            uu = np.sin(xx) * np.cos(yy)
        else:
            raise ValueError('d=%i not implemented'%d)
    else:
        raise ValueError('axis=%i not valid'%axis)
    
    return uu

# main()
# ======================================================================

if __name__ == '__main__':
    
    save_pdf = True
    png_px_x = 3840
    figsize  = (6,6/(32/15))
    dpi      = 250
    fontsize_anno = 6
    fontsize_lgnd = 6
    
    if False: ## activate a customized plotting env
        
        darkMode = False
        turbx.set_mpl_env(useTex=True, darkMode=darkMode, font='ibm')
        
        hues = [323,282,190,130,92,60,30] ## Lch(ab) hues [degrees] : https://css.land/lch/
        if darkMode:
            L = 85; c = 155 ## luminance & chroma
        else:
            L = 55; c = 100 ## luminance & chroma
        
        colors = turbx.get_Lch_colors(hues,L=L,c=c,fmt='hex',test_plot=False)
        purple, blue, cyan, green, yellow, orange, red = colors
        colors_dark = turbx.hsv_adjust_hex(colors,0,0,-0.3)
        purple_dk, blue_dk, cyan_dk, green_dk, yellow_dk, orange_dk, red_dk = colors_dark
        cl1 = blue; cl2 = yellow; cl3 = red; cl4 = green; cl5 = purple; cl6 = orange; cl7 = cyan
    
    else:
        
        purple, blue, cyan, green, yellow, orange, red = 'purple', 'blue', 'cyan', 'green', 'yellow', 'orange', 'red'
    
    # === 1D
    
    if True:
        
        print('----- 1D')
        
        xmin, xmax = 2, 8
        nx      = 51
        d       = 1 ## degree of derivative
        uniform = True
        
        if uniform:
            x = np.linspace( xmin , xmax , nx )
        else:
            x = np.logspace( np.log10(2) , np.log10(10) , nx )
            x = xmin + (x-x.min())/(x.max()-x.min())*(xmax-xmin)
        
        # === turbx.gradient()
        
        u     = test_func_1D_01(x,d=0)
        ud_a  = test_func_1D_01(x,d=d) ## analytical solution for derivative order =d
        
        ud_fd = turbx.gradient(u, x=x, d=d, axis=0, acc=8, edge_stencil='full')
        
        rerr          = (ud_a-ud_fd)/ud_a
        rerr_abs_max  = np.abs(rerr).max()
        rerr_abs_mean = np.mean(np.abs(rerr))
        print('turbx.gradient() : rel err abs max  : %0.5e'%rerr_abs_max)
        print('turbx.gradient() : rel err abs mean : %0.5e'%rerr_abs_mean)
        
        # === numpy.gradient()
        
        if (d==1):
            
            ud_fd2         = np.gradient(u, x, axis=0, edge_order=2)
            
            rerr2          = (ud_a-ud_fd2)/ud_a
            rerr_abs_max2  = np.abs(rerr2).max()
            rerr_abs_mean2 = np.mean(np.abs(rerr2))
            print('numpy.gradient() : rel err abs max  : %0.5e'%rerr_abs_max2)
            print('numpy.gradient() : rel err abs mean : %0.5e'%rerr_abs_mean2)
        
        if True: ## plot
            plt.close('all')
            fig1 = plt.figure(figsize=figsize, dpi=dpi)
            ax1 = plt.gca()
            ax1.tick_params(axis='x', which='both', direction='out')
            ax1.tick_params(axis='y', which='both', direction='out')
            ln1, = ax1.plot(x , ud_a , c='#808080' , lw=0.8 , label='analytical')
            if ('ud_fd2' in locals()):
                ln1, = ax1.plot(x , ud_fd2 , c=blue , lw=0.8 , label='np.gradient()')
            ln1, = ax1.plot(x , ud_fd , c=red , lw=0.8 , label='turbx.gradient()')
            fig1.tight_layout(pad=0.25)
            fig1.tight_layout(pad=0.25)
            ##
            lg = ax1.legend(loc='best', ncol=1, fontsize=fontsize_lgnd, facecolor=ax1.get_facecolor()) ##, bbox_to_anchor=(0.995, 0.05))
            lg.get_frame().set_linewidth(0.2)
            lg.set_zorder(21)
            ##
            #if save_pdf: fig1.savefig('ud1D.pdf', format='pdf')
            #fig1.savefig('ud1D.png', dpi=png_px_x/plt.gcf().get_size_inches()[0])
            plt.show()
            pass
    
    # === 2D
    
    if True:
        
        print('----- 2D')
        
        xmin , xmax = (-2.+0.2)*np.pi , (+2.+0.2)*np.pi
        ymin , ymax = (-2.+0.2)*np.pi , (+2.+0.2)*np.pi
        
        nx = 151
        ny = 201
        
        d = 1 ## degree of derivative
        axis = 0
        uniform = False
        
        if uniform:
            x = np.linspace( xmin , xmax , nx )
            y = np.linspace( ymin , ymax , ny )
        else:
            x = np.logspace( np.log10(2) , np.log10(10) , nx )
            y = np.logspace( np.log10(2) , np.log10(10) , ny )
            x = xmin + (x-x.min())/(x.max()-x.min())*(xmax-xmin)
            y = ymin + (y-y.min())/(y.max()-y.min())*(ymax-ymin)
        
        if (axis==0):
            xi = x
        elif (axis==1):
            xi = y
        else:
            raise ValueError('axis=%i is invalid'%axis)
        
        xx, yy = np.meshgrid(x, y, indexing='ij')
        
        # === turbx.gradient()
        
        u     = test_func_2D_01(xx, yy, d=0, axis=axis)
        ud_a  = test_func_2D_01(xx, yy, d=d, axis=axis) ## analytical solution for derivative order =d
        
        ud_fd = turbx.gradient(u, x=xi, d=d, axis=axis, acc=8, edge_stencil='full')
        
        rerr          = (ud_a-ud_fd)/ud_a ## relative error --> has trouble @ f()==0
        rerr_abs_max  = np.abs(rerr).max()
        rerr_abs_mean = np.mean(np.abs(rerr))
        print('turbx.gradient() : rel err abs max  : %0.5e'%rerr_abs_max)
        print('turbx.gradient() : rel err abs mean : %0.5e'%rerr_abs_mean)
        
        nerr = (ud_a-ud_fd)/np.abs(ud_a).max() ## normed error --> good enough for when func has low dynamic range
        nerr_abs_max  = np.abs(nerr).max()
        nerr_abs_mean = np.mean(np.abs(nerr))
        print('turbx.gradient() : normed err abs max  : %0.5e'%nerr_abs_max)
        print('turbx.gradient() : normed err abs mean : %0.5e'%nerr_abs_mean)
        
        # === numpy.gradient()
        
        if (d==1):
            
            print('---')
            
            ud_fd2         = np.gradient(u, xi, axis=axis, edge_order=2)
            
            rerr2          = (ud_a-ud_fd2)/ud_a
            rerr_abs_max2  = np.abs(rerr2).max()
            rerr_abs_mean2 = np.mean(np.abs(rerr2))
            print('numpy.gradient() : rel err abs max  : %0.5e'%rerr_abs_max2)
            print('numpy.gradient() : rel err abs mean : %0.5e'%rerr_abs_mean2)
            
            nerr2          = (ud_a-ud_fd2)/ud_a
            nerr_abs_max2  = np.abs(nerr2).max()
            nerr_abs_mean2 = np.mean(np.abs(nerr2))
            print('numpy.gradient() : normed err abs max  : %0.5e'%nerr_abs_max2)
            print('numpy.gradient() : normed err abs mean : %0.5e'%nerr_abs_mean2)
        
        # ===
        
        if True: ## plot
            plt.close('all')
            fig1 = plt.figure(dpi=150) # figsize=(8,8/(16/9))
            ax1 = plt.gca()
            ax1.set_aspect('equal')
            ax1.tick_params(axis='x', which='both', direction='out')
            ax1.tick_params(axis='y', which='both', direction='out')
            ##
            cmap = mpl.cm.RdBu_r
            norm = mpl.colors.Normalize(vmin=-np.abs(ud_fd).max(), vmax=+np.abs(ud_fd).max())
            ##
            im = ax1.pcolormesh( xx,
                                 yy,
                                 ud_fd,
                                 shading='auto',
                                 cmap=cmap, 
                                 norm=norm, 
                                 rasterized=True )
            ##
            ax1.set_xlabel(r'$x$')
            ax1.set_ylabel(r'$y$')
            ax1.set_xlim(x.min(),x.max())
            ax1.set_ylim(x.min(),y.max())
            cbar = fig1.colorbar(im, ax=ax1, orientation='vertical', pad=0.01)
            cbar.ax.tick_params(axis='y', direction='out')
            fig1.tight_layout(pad=0.25)
            fig1.tight_layout(pad=0.25)
            ##
            #x0A, y0A, dxA, dyA = ax1.get_position().bounds
            #x0B, y0B, dxB, dyB = cbar.ax.get_position().bounds
            #cbar.ax.set_position([x0B, y0A, dxB, dyA])
            ##
            #if save_pdf: fig1.savefig('ud2D.pdf', format='pdf')
            #fig1.savefig('ud2D.png', dpi=png_px_x/plt.gcf().get_size_inches()[0])
            plt.show()
            pass
