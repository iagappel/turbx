'''
turbx

Extensible toolkit for analyzing turbulent flow datasets
'''

__version__ = '0.3.9'
__author__ = 'Jason A'

from .turbx import cgd
from .turbx import rgd
from .turbx import eas4
from .turbx import ztmd
from .turbx import lpd
from .turbx import spd

from .turbx import h5_chunk_sizer
from .turbx import h5_visititems_print_attrs
from .turbx import h5_visit_container

from .turbx import eas3

from .turbx import curve_fitter

from .turbx import Blasius_solution
from .turbx import freestream_parameters
from .turbx import calc_profile_edge_1d
from .turbx import calc_d99_1d
from .turbx import calc_bl_integral_quantities_1d

from .turbx import interp_2d_structured
from .turbx import interp_1d
from .turbx import fd_coeff_calculator
from .turbx import assemble_1d_fd_coeff_vector_central
from .turbx import assemble_1d_fd_coeff_vector_custom
from .turbx import gradient
from .turbx import get_metric_tensor_3d
from .turbx import get_metric_tensor_2d
from .turbx import get_grid_quality_metrics_2d
from .turbx import smoothstep
from .turbx import stretch_1d_cluster_ends
from .turbx import time_integrate_2d_seeder
from .turbx import time_integrate_2d

from .turbx import rect_to_cyl
from .turbx import cyl_to_rect
from .turbx import rotate_2d

from .turbx import get_grad
from .turbx import get_curl

from .turbx import get_overlapping_window_size
from .turbx import get_overlapping_windows
from .turbx import ccor
from .turbx import ccor_naive

from .turbx import gulp

from .turbx import format_time_string
from .turbx import format_nbytes
from .turbx import even_print

from .turbx import set_mpl_env
from .turbx import mpl_typeset_templates
from .turbx import colors_table
from .turbx import get_standard_colors
from .turbx import cmap_Lch_edit
from .turbx import colors_test_plot
from .turbx import color_dict_to_tex
from .turbx import get_Lch_colors
from .turbx import hex2rgb
from .turbx import hsv_adjust_hex
from .turbx import analytical_u_plus_y_plus
from .turbx import nice_log_labels
from .turbx import fig_trim_y
from .turbx import fig_trim_x
from .turbx import axs_grid_initializer
from .turbx import axs_grid_compress
from .turbx import tight_layout_helper_ax_with_cbar
from .turbx import cmap_convert_mpl_to_pview

__all__ = ['cgd',
           'rgd',
           'eas4',
           'ztmd',
           'lpd',
           'spd',
           'h5_chunk_sizer',
           'h5_visititems_print_attrs',
           'h5_visit_container',
           'eas3',
           'curve_fitter',
           'Blasius_solution',
           'freestream_parameters',
           'calc_profile_edge_1d',
           'calc_d99_1d',
           'calc_bl_integral_quantities_1d',
           'interp_2d_structured',
           'interp_1d',
           'fd_coeff_calculator',
           'assemble_1d_fd_coeff_vector_central',
           'assemble_1d_fd_coeff_vector_custom',
           'gradient',
           'get_metric_tensor_3d',
           'get_metric_tensor_2d',
           'get_grid_quality_metrics_2d',
           'smoothstep',
           'stretch_1d_cluster_ends',
           'time_integrate_2d_seeder',
           'time_integrate_2d',
           'rect_to_cyl',
           'cyl_to_rect',
           'rotate_2d',
           'get_grad',
           'get_curl',
           'get_overlapping_window_size',
           'get_overlapping_windows',
           'ccor',
           'ccor_naive',
           'gulp',
           'format_time_string',
           'format_nbytes',
           'even_print',
           'set_mpl_env',
           'mpl_typeset_templates',
           'colors_table',
           'get_standard_colors',
           'cmap_Lch_edit',
           'colors_test_plot',
           'color_dict_to_tex',
           'get_Lch_colors',
           'hex2rgb',
           'hsv_adjust_hex',
           'analytical_u_plus_y_plus',
           'nice_log_labels',
           'fig_trim_y',
           'fig_trim_x',
           'axs_grid_initializer',
           'axs_grid_compress',
           'tight_layout_helper_ax_with_cbar',
           'cmap_convert_mpl_to_pview',
           ]
