'''
turbx

Tools for analysis of turbulent flow datasets.
'''

__version__ = '0.1.2'
__author__ = 'Jason Appelbaum'

from .turbx import rgd
from .turbx import cgd
from .turbx import eas4
from .turbx import lpd
from .turbx import eas3

from .turbx import curve_fitter

from .turbx import fd_coeff_calculator
from .turbx import gradient

from .turbx import get_grad
from .turbx import get_curl
from .turbx import get_overlapping_window_size
from .turbx import get_overlapping_windows
from .turbx import ccor
from .turbx import ccor_naive

from .turbx import gulp

from .turbx import format_time_string
from .turbx import even_print

from .turbx import set_mpl_env
from .turbx import get_color_list
from .turbx import get_Lch_colors
from .turbx import hex2rgb
from .turbx import hsv_adjust_hex
from .turbx import cmap_hsv_adjust
from .turbx import analytical_u_plus_y_plus
from .turbx import fig_trim
from .turbx import axs_grid_compress
from .turbx import tight_layout_helper_ax_with_cbar
from .turbx import cmap_convert_mpl_to_pview

__all__ = ['rgd',
           'cgd',
           'eas4',
           'lpd',
           'eas3',
           'curve_fitter',
           'fd_coeff_calculator',
           'gradient',
           'get_grad',
           'get_curl',
           'get_overlapping_window_size',
           'get_overlapping_windows',
           'ccor',
           'ccor_naive',
           'gulp',
           'format_time_string',
           'even_print',
           'set_mpl_env',
           'get_color_list',
           'get_Lch_colors',
           'hex2rgb',
           'hsv_adjust_hex',
           'cmap_hsv_adjust',
           'analytical_u_plus_y_plus',
           'fig_trim',
           'axs_grid_compress',
           'tight_layout_helper_ax_with_cbar',
           'cmap_convert_mpl_to_pview',
           ]