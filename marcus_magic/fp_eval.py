"""
"""
#==.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===

import os
import sys
import time
import shutil
import argparse
#import datetime
import traceback

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

#rom fp_eval_common import make_timestamp, strftime_to_epoch, make_elapsed_time
#rom fp_eval_common import fmt0, fmt1
#rom fp_eval_common import prt_dict, prt_list
#
#rom fp_eval_common import make_x_coordinate_roi_reference_values
#
#rom fp_eval_common import get_dataset
#rom fp_eval_common import fix_nan_values
#rom fp_eval_common import make_laser_profiles_image_plot
#rom fp_eval_common import make_laser_profiles_overlay_plot
#rom fp_eval_common import make_laser_measts_overlay_plot
#rom fp_eval_common import make_laser_meast_zs_histogram_plot
#rom fp_eval_common import make_laser_measts_overlay_and_zs_histogram_plot
#
#rom fp_eval_common import imshow
#rom fp_eval_common import make_laser_profile_edges_image_plot
#rom fp_eval_common import make_gap_results_plot

#rom fp_eval_tow_ends import make_results_fields_ends
#rom fp_eval_tow_ends import get_located_tow_end_events_dataset

#rom fp_eval_tow_gaps import make_results_fields_gaps
#rom fp_eval_tow_gaps import get_located_tow_gap_events_dataset

try:
    # ... is successful during my code development.
    sys.path.append(r"C:\_my_python_reference_library")
    import my_python_reference_library as mprl
    print "\n... my python reference library ... is available\n"
    None if 1 else help(mprl)
except:
    # ... happens when this code is in production.
    print "\n... my python reference library ... is not available\n"
    #rint '\n... print_exc():\n', traceback.print_exc(file=sys.stdout)

np.set_printoptions(linewidth=235)  # was 220
np.set_printoptions(precision=3)  # was 4
np.set_printoptions(suppress=True)

pd.set_option('max_rows', 7000)
#d.set_option('max_columns', 24)
#d.set_option('max_columns', 20)  # was 48
pd.set_option('max_columns', 23)  # was 48
#d.set_option('display.width', 225)  # was 220
pd.set_option('display.width', 240)  # was 220
pd.set_option('display.precision', 3)  # was 4
pd.options.display.float_format = '{:,.3f}'.format  # was 4

# http://legacy.python.org/dev/peps/pep-0396/
#   PEP 396 -- Module Version Numbers
#
# Configuration Control "Contract":
#
#   The version identifier is ... "x.y.z" ...
#   ... where ...
#       x ... is the major release number
#       y ... is the minor release number
#       z ... is the build number
#
#   The software system "owner" assigns the major and minor release numbers.
#
#   The developer of this code assigns the build number (a serial number).
#
__version__ = '0.0.16'

#==.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===
# (below) initialization functions


def get_job_configuration_parameters(ngn, job_config_txt_abspath):
    """
    Returns a Pandas Series contining job configuration parameters.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '... '
    def_str = 'get_job_configuration_parameters'
    if prt_:
        print fmt0(just0)[0:] % ('(beg) %s' % def_str, ngn.in_)
        print "%s" % (mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True and prt_ if 1 else False

    #== === === === === === === === === === === === === === === === === === ===

    with open(job_config_txt_abspath, 'r') as f:
        lines = [line.strip() for line in f]

    if 1 and prt:
        print fmt1(just1)[0:] % (
            "(job_config_txt_abspath) lines", '\n'.join(lines))

    prt = True and prt_ if 0 else False

    ps = pd.Series()
    for line in lines:
        if 1 and prt:
            print fmt0(just1)[0:] % ("line", '"%s"' % line)

        if line.startswith('job_id'):
            job_id = line[line.find(':') + 1:].strip()
            ngn.job_id = job_id
            ps['job_id'] = job_id
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found job_id")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % (
                    "job_id", "'%s'" % job_id)

        if line.startswith('sensor'):
            sensor = line[line.find(':') + 1:].strip()
            ngn.sensor = sensor
            ps['sensor'] = sensor
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found sensor")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % (
                    "sensor", "'%s'" % sensor)

        if line.startswith('number_of_strips'):
            number_of_strips = np.int(
                line[line.find(':') + 1:])
            ngn.number_of_strips = number_of_strips
            ps['number_of_strips'] = number_of_strips
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found number_of_strips")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % (
                    "number_of_strips", number_of_strips)

        if line.startswith('strip_width_nominal_mm'):
            strip_width_nominal_mm = np.float(
                line[line.find(':') + 1:])
            ngn.strip_width_nominal_mm = strip_width_nominal_mm
            ps['strip_width_nominal_mm'] = strip_width_nominal_mm
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found strip_width_nominal_mm")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % (
                    "strip_width_nominal_mm", strip_width_nominal_mm)

        if line.startswith('gap_width_nominal_mm'):
            gap_width_nominal_mm = np.float(
                line[line.find(':') + 1:])
            ngn.gap_width_nominal_mm = gap_width_nominal_mm
            ps['gap_width_nominal_mm'] = gap_width_nominal_mm
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found gap_width_nominal_mm")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % (
                    "gap_width_nominal_mm", gap_width_nominal_mm)

        if line.startswith('strip_centers_xref_mm'):
            strip_centers_xref_mm = np.float(
                line[line.find(':') + 1:])
            ngn.strip_centers_xref_mm = strip_centers_xref_mm
            ps['strip_centers_xref_mm'] = strip_centers_xref_mm
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found strip_centers_xref_mm")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % (
                    "strip_centers_xref_mm", strip_centers_xref_mm)

        if line.startswith('nan_value_floor_threshold'):
            nan_value_floor_threshold = np.float(
                line[line.find(':') + 1:])
            ngn.nan_value_floor_threshold = nan_value_floor_threshold
            ps['nan_value_floor_threshold'] = nan_value_floor_threshold
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found nan_value_floor_threshold")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % (
                    "nan_value_floor_threshold", nan_value_floor_threshold)

        if line.startswith('nan_value_ceiling_threshold'):
            nan_value_ceiling_threshold = np.float(
                line[line.find(':') + 1:])
            ngn.nan_value_ceiling_threshold = nan_value_ceiling_threshold
            ps['nan_value_ceiling_threshold'] = (
                nan_value_ceiling_threshold)
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found nan_value_ceiling_threshold")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % ("nan_value_ceiling_threshold",
                    nan_value_ceiling_threshold)

        if line.startswith('half_window_pts'):
            half_window_pts = np.int(
                line[line.find(':') + 1:])
            ngn.half_window_pts = half_window_pts
            ps['half_window_pts'] = half_window_pts
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found half_window_pts")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % ("half_window_pts",
                    half_window_pts)

        if line.startswith('window_offset_pts'):
            window_offset_pts = np.int(
                line[line.find(':') + 1:])
            ngn.window_offset_pts = window_offset_pts
            ps['window_offset_pts'] = window_offset_pts
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found window_offset_pts")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % ("window_offset_pts",
                    window_offset_pts)

        if line.startswith('dzdys_threshold'):
            dzdys_threshold = np.float(
                line[line.find(':') + 1:])
            ngn.dzdys_threshold = dzdys_threshold
            ps['dzdys_threshold'] = dzdys_threshold
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found dzdys_threshold")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % ("dzdys_threshold",
                    dzdys_threshold)

        if line.startswith('zs_median_filter_size'):
            zs_median_filter_size = np.int(
                line[line.find(':') + 1:])
            ngn.zs_median_filter_size = zs_median_filter_size
            ps['zs_median_filter_size'] = zs_median_filter_size
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found zs_median_filter_size")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % ("zs_median_filter_size",
                    zs_median_filter_size)

        if line.startswith('zs_gaussian_filter_sigma'):
            zs_gaussian_filter_sigma = np.float(
                line[line.find(':') + 1:])
            ngn.zs_gaussian_filter_sigma = zs_gaussian_filter_sigma
            ps['zs_gaussian_filter_sigma'] = zs_gaussian_filter_sigma
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found zs_gaussian_filter_sigma")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % ("zs_gaussian_filter_sigma",
                    zs_gaussian_filter_sigma)

        if line.startswith('dzdxs_threshold'):
            dzdxs_threshold = np.float(
                line[line.find(':') + 1:])
            ngn.dzdxs_threshold = dzdxs_threshold
            ps['dzdxs_threshold'] = dzdxs_threshold
            if 1 and prt:
                print fmt0(just1)[1:] % ("",
                    "found dzdxs_threshold")
                print fmt0(just1)[1:] % (
                    "line.find(':')", line.find(':'))
                print fmt0(just1)[1:] % ("dzdxs_threshold",
                    dzdxs_threshold)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return ps


def make_x_coordinate_roi_reference_values(ngn):
    """
    Returns a Pandas Series contining job configuration parameters.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '... '
    def_str = 'make_x_coordinate_roi_reference_values'
    if prt_:
        print fmt0(just0)[0:] % ('(beg) %s' % def_str, ngn.in_)
        print "%s" % (mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True and prt_ if 1 else False

    #== === === === === === === === === === === === === === === === === === ===

    ps = pd.Series()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # generate the tow center x-coordinate reference values for the
    # within-course tows numbered from 1 to "number_of_strips" and the
    # left and right between-courses neighboring strips numbered 0 and
    # "number_of_strips + 1".

    tow_lane_mm = (ngn.job_cfg_ps.strip_width_nominal_mm +
        ngn.job_cfg_ps.gap_width_nominal_mm)
    if 1 and prt:
        print fmt0(just2)[0:] % ("ngn.job_cfg_ps.strip_width_nominal_mm",
                ngn.job_cfg_ps.strip_width_nominal_mm)
        print fmt0(just2)[1:] % ("ngn.job_cfg_ps.gap_width_nominal_mm",
                ngn.job_cfg_ps.gap_width_nominal_mm)
        print fmt0(just2)[1:] % ("ngn.job_cfg_ps.strip_centers_xref_mm",
                ngn.job_cfg_ps.strip_centers_xref_mm)
        print fmt0(just2)[1:] % ("tow_lane_mm", tow_lane_mm)

    tow_ids = np.arange(ngn.job_cfg_ps.number_of_strips + 2).astype(np.int)
    tow_center_idxs = tow_ids.copy()
    ps['tow_ids'] = tow_ids
    if 1 and prt:
        print fmt0(just0)[1:] % ("len(tow_ids)", len(tow_ids))
        print fmt1(just0)[1:] % ("tow_ids", tow_ids)
        print fmt1(just0)[1:] % ("tow_center_idxs", tow_center_idxs)
    None if 1 else sys.exit()

    tow_center_xrefs = tow_center_idxs.astype(np.float)
    if 1 and prt:
        print fmt0(just0)[0:] % ("len(tow_center_xrefs)",
            len(tow_center_xrefs))
        print fmt1(just0)[1:] % ("(init) tow_center_xrefs",
            tow_center_xrefs)

    tow_center_xrefs -= tow_center_xrefs[-1] / 2.
    if 0 and prt:
        print fmt1(just0)[1:] % ("(upd1) tow_center_xrefs",
            tow_center_xrefs)

    tow_center_xrefs *= tow_lane_mm
    if 0 and prt:
        print fmt1(just0)[1:] % ("(upd2) tow_center_xrefs",
            tow_center_xrefs)

    tow_center_xrefs += ngn.job_cfg_ps.strip_centers_xref_mm
    ps['tow_center_xrefs'] = tow_center_xrefs
    if 1 and prt:
        print fmt1(just0)[1:] % ("(finl) tow_center_xrefs",
            tow_center_xrefs)

    if 1 and prt:
        print fmt0(just0)[0:] % ("len(tow_ids)", len(tow_ids))
        print fmt1(just0)[1:] % (
            "np.vstack((tow_ids, tow_center_xrefs))",
            np.vstack((tow_ids, tow_center_xrefs)))

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # generate the tow edge x-coordinate reference values for the
    # within-course tows numbered from 1 to "number_of_strips"

    tow_edge_ids = (tow_ids[:-1] * 100) + (tow_ids[:-1] + 1)
    ps['tow_edge_ids'] = tow_edge_ids
    if 1 and prt:
        print fmt0(just0)[1:] % ("len(tow_edge_ids)", len(tow_edge_ids))
        print fmt1(just0)[0:] % ("tow_edge_ids", tow_edge_ids)

    tow_edge_xrefs = (tow_center_xrefs[:-1] + tow_center_xrefs[1:]) / 2.
    ps['tow_edge_xrefs'] = tow_edge_xrefs
    if 1 and prt:
        print fmt0(just0)[1:] % ("len(tow_edge_xrefs)",
            len(tow_edge_xrefs))
        print fmt1(just0)[1:] % ("tow_edge_xrefs", tow_edge_xrefs)

    if 1 and prt:
        print fmt0(just0)[0:] % ("len(tow_edge_ids)", len(tow_edge_ids))
        print fmt1(just0)[1:] % (
            "np.vstack((tow_edge_ids, tow_edge_xrefs))",
            np.vstack((tow_edge_ids, tow_edge_xrefs)))

    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return ps


def make_results_fields_ends(ngn):
    """
    Returns a list of tow ends analysis results fields.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '... '
    def_str = 'make_results_fields_ends'
    if prt_:
        print fmt0(just0)[0:] % ('(beg) %s' % def_str, ngn.in_)
        print "%s" % (mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    results_fields_ends = ngn.results_fields_ends_cols_to_copy[:]
    for idx in xrange(ngn.job_cfg_ps['number_of_strips']):
        tow = idx + 1
        tow_id = "%.2i" % tow
        results_fields_ends.extend([
            't%sd' % tow_id,
            't%si' % tow_id,
            't%sus' % tow_id,
            't%sxc' % tow_id,
        ])
        if 0 and prt:
            if idx == 0:
                print
            print fmt0(just0)[1:] % ("[idx, tow, tow_id]",
                [idx, tow, tow_id])

    if 0 and prt:
        print fmt1(just1)[0:] % ("results_fields_ends[:9]",
            prt_list(results_fields_ends[:9], -20))

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return results_fields_ends


def make_results_fields_gaps(ngn):
    """
    Returns a list of tow gaps analysis results fields.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '... '
    def_str = 'make_results_fields_ends'
    if prt_:
        print fmt0(just0)[0:] % ('(beg) %s' % def_str, ngn.in_)
        print "%s" % (mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    results_fields_gaps = ngn.results_fields_gaps_cols_to_copy[:]
    results_fields_gaps.extend(ngn.results_fields_gaps_profile_quality[:])
    for idx in xrange(ngn.job_cfg_ps.number_of_strips + 1):
        # gap results for each gap
        tow0 = idx
        tow1 = idx + 1
        gap_id = "%.2i%.2i" % (tow0, tow1)
        results_fields_gaps.extend([
            'Gap%sPresent' % gap_id,
            'Gap%sEdgeLfNnanIdx' % gap_id,
            'Gap%sEdgeLfIdx' % gap_id,
            'Gap%sEdgeLfX' % gap_id,
            'Gap%sEdgeRtNnanIdx' % gap_id,
            'Gap%sEdgeRtIdx' % gap_id,
            'Gap%sEdgeRtX' % gap_id,
            'Gap%sWidth' % gap_id,
            'Gap%sClass' % gap_id,
        ])
        if 0 and prt:
            if idx == 0:
                print
            print fmt0(just0)[1:] % ("[tow0, tow1, gap_id]",
                [tow0, tow1, gap_id])

    if 1 and prt:
        print fmt1(just1)[0:] % ("results_fields_gaps[:9]",
            prt_list(results_fields_gaps[:9], -20))

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return results_fields_gaps


# (above) initialization functions
#==.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===
# (below) utility functions


def make_timestamp(ngn, epoch_secs, strftime="%Y-%m-%d %H:%M:%S"):
    """
    Returns a time stamp representation of seconds (ticks) since epoch.
    """
    in_ = ngn.in_
    just0 = ngn.just0
    prt = False if 1 else True
    def_str = 'make_timestamp'
    if prt:
        print fmt0(just0)[0:] % ('(beg) def %s' % def_str, in_)
        print "%s" % ('--- ' * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    # references ... timestamp formatting ...
    #   http://www.tutorialspoint.com/python/time_strftime.htm
    #   https://www.google.com/#q=python+time.strftime+milliseconds

    #f 1 and prt:
    #   print fmt0(just0)[0:] % ('strftime', strftime)
    #   print fmt0(just0)[1:] % ('epoch_secs', epoch_secs)

    epoch_secs_timestamp = int(epoch_secs)
    local_time_timestamp = time.localtime(epoch_secs_timestamp)
    timestamp_secs = time.strftime(strftime, local_time_timestamp)
    #f 1 and prt:
    #   print fmt0(just0)[0:] % ('epoch_secs_timestamp', epoch_secs_timestamp)
    #   print fmt0(just0)[1:] % ('local_time_timestamp', local_time_timestamp)
    #   print fmt0(just0)[1:] % ('timestamp_secs', timestamp_secs)

    epoch_secs_mod = epoch_secs - int(epoch_secs)
    epoch_msecs_mod = ".%.0f" % (epoch_secs_mod * 1000.)
    timestamp_msecs = timestamp_secs + epoch_msecs_mod
    #f 1 and prt:
    #   print fmt0(just0)[0:] % ('epoch_secs_mod', epoch_secs_mod)
    #   print fmt0(just0)[1:] % ('epoch_msecs_mod', epoch_msecs_mod)
    #f 1 and prt:
    #   print fmt0(just0)[1:] % ('timestamp_msecs', timestamp_msecs)

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt:
        print "\n%s" % ('--- ' * ngn.mult)
        print fmt0(just0)[1:] % ('(end) def %s' % def_str, in_)

    return timestamp_msecs


def strftime_to_epoch(ngn, time_stamp):
    """
    Returns the seconds since epoch given a time_stamp.
    """
    in_ = ngn.in_
    just0 = ngn.just0
    prt = False if 1 else True
    def_str = 'strftime_to_epoch'
    if prt:
        print fmt0(just0)[0:] % ('(beg) def %s' % def_str, in_)
        print "%s" % ('--- ' * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    # http://stackoverflow.com/questions/11743019/ ...
    # ... convert-python-datetime-to-epoch-with-strftime
    time_stamp_strftime = "%Y-%m-%d %H:%M:%S"
    strptime = datetime.strptime(time_stamp, time_stamp_strftime)
    epoch_secs = (
        time.mktime(strptime.timetuple()) + strptime.microsecond * 1e-6)
    #f 1 and prt:
    #   print fmt0(just0)[0:] % ('time_stamp', time_stamp)
    #   print fmt0(just0)[1:] % ('strptime', strptime)
    #   print fmt0(just0)[1:] % ('epoch_secs', epoch_secs)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt:
        print "\n%s" % ('--- ' * ngn.mult)
        print fmt0(just0)[1:] % ('(end) def %s' % def_str, in_)

    return epoch_secs


def make_elapsed_time(ngn, epoch_secs):
    """
    Returns a string representation of elapsed time (in seconds).
    """
    in_ = ngn.in_
    just0 = ngn.just0
    prt = False if 1 else True
    def_str = 'make_elapsed_time'
    if prt:
        print fmt0(just0)[0:] % ('(beg) def %s' % def_str, in_)
        print "%s" % ('--- ' * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #f 1 and prt:
    #   print fmt0(just0)[0:] % ('epoch_secs', epoch_secs)

    elapsed_time = 'elapsed time: %.3f seconds' % (time.time() - epoch_secs)
    #f 1 and prt:
    #   print fmt0(just0)[0:] % ('elapsed_time', elapsed_time)

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt:
        print "\n%s" % ('--- ' * ngn.mult)
        print fmt0(just0)[1:] % ('(end) def %s' % def_str, in_)

    return elapsed_time


def fmt0(just0=0):
    """
    Returns a single-line string print format.
    """
    return "\n... %" + str(just0) + "s ...  %s"


def fmt1(just0=0):
    """
    Returns a multi-line string print format.
    """
    return "\n... %" + str(just0) + "s ...\n%s"


def prt_dict(dict_, just0=24, sort=False):
    """
    Returns a string representation of dict.
    """
    str_ = ''
    dict_items = sorted(dict_.items()) if sort else dict_.items()
    for k, v in dict_items:
        str_ += ("%" + str(just0) + "s :  %s\n") % (k, v)
    return str_


def prt_list(list_, just0=24, sort=False):
    """
    Returns a string representation of dict.
    """
    str_ = ''
    list_items = sorted(list_) if sort else list_
    for list_item in list_items:
        str_ += ("%" + str(just0) + "s\n") % list_item
    return str_


# (above) utility functions
#==.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===
# (below) python program container classes


class Ngn(object):
    """
    Container class for program parameters.
    """
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    in_ = os.path.basename(sys.argv[0]) if 1 else sys.argv[0]
    just0, just1, just2 = -24, -36, -48
    mult = 36
    stars_mult = 50

    #lib_abspath = os.path.abspath(r'..\_datasets')

    version_py = int(sys.modules[__name__].__version__.split('.')[-1])
    preview_dir = r'preview_py%.3i' % version_py
    results_dir = r'results_py%.3i' % version_py

    # tow event key fields for ends results
    results_fields_ends_cols_to_copy = ['ProfileID', 'MeastID',
        'TowPresentBits_Tow32toTow01', 'U-Sensor', 'U-TowChangeProgramed']

    half_window_pts = 10
    window_offset_pts = 7
    tow_ends_analysis_zs_median_filter_size = 11
    tow_ends_analysis_zs_gaussian_filter_sigma = 1.2

    # tow event key fields for gap results
    results_fields_gaps_cols_to_copy = ['ProfileID', 'MeastID',
            'GapPresentBits_Gap3233toGap0001', 'U-Sensor']

    # tow event profile quality fields for gap results
    results_fields_gaps_profile_quality = [
        'zs_idx_nnan_lf',
        'zs_idx_nnan_rt',
        'pts_fov',
        'pts_roi',
        'pts_value',
        'pts_drop'
    ]

    def job_init(self, args_parse):
        """
        Initializes job parameters.
        """
        just0, just1, just2 = self.just0, self.just1, self.just2
        prt = False if 1 else True
        prt_ = prt
        prt__ = False if 1 else True  # def print switch
        mult_str = '... '
        def_str = 'job_init'
        if prt_ or prt__:
            print fmt0(just0)[0:] % ('(beg) %s' % def_str, self.in_)
            print "%s" % (mult_str * self.mult)
        #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

        prt = True if 0 and prt_ else False

        if 0 and prt:
            print fmt1(just0)[0:] % ("(init) self", prt_obj(self))

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        ## required parameters
        job_xs_csv_abspath = args_parse.job_xs_csv_abspath
        job_zs_csv_abspath = args_parse.job_zs_csv_abspath
        job_us_csv_abspath = args_parse.job_us_csv_abspath
        job_config_txt = args_parse.job_config_txt
        #
        ## boolean parameters
        preview = args_parse.preview
        skip_tow_ends_analysis = args_parse.skip_tow_ends_analysis
        skip_tow_gaps_analysis = args_parse.skip_tow_gaps_analysis
        legacy_gap_analysis = args_parse.legacy_gap_analysis
        ###add_pyapp_version = args_parse.add_pyapp_version
        write_to_results_dir = args_parse.write_to_results_dir
        count_gap_rois_analyzed = args_parse.count_gap_rois_analyzed
        ###autothreshold_nan_values = args_parse.autothreshold_nan_values
        #
        # plotting options:
        make_survey_plots = args_parse.make_survey_plots
        make_gallery00_plots = args_parse.make_gallery00_plots
        make_gallery01_plots = args_parse.make_gallery01_plots
        make_gallery02_plots = args_parse.make_gallery02_plots
        make_gallery03_plots = args_parse.make_gallery03_plots
        make_gallery04_plots = args_parse.make_gallery04_plots
        make_gallery05_plots = args_parse.make_gallery05_plots

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        ## required parameters
        self.job_xs_csv_abspath = job_xs_csv_abspath
        self.job_zs_csv_abspath = job_zs_csv_abspath
        self.job_us_csv_abspath = job_us_csv_abspath
        ### self.job_config_txt = job_config_txt
        #
        ## boolean parameters
        self.preview = preview
        self.skip_tow_ends_analysis = skip_tow_ends_analysis
        self.skip_tow_gaps_analysis = skip_tow_gaps_analysis
        self.legacy_gap_analysis = legacy_gap_analysis
        ###self.add_pyapp_version = add_pyapp_version
        self.write_to_results_dir = write_to_results_dir
        self.count_gap_rois_analyzed = count_gap_rois_analyzed
        ###self.autothreshold_nan_values = autothreshold_nan_values
        #
        # plotting options:
        self.make_survey_plots = make_survey_plots
        self.make_gallery00_plots = make_gallery00_plots
        self.make_gallery01_plots = make_gallery01_plots
        self.make_gallery02_plots = make_gallery02_plots
        self.make_gallery03_plots = make_gallery03_plots
        self.make_gallery04_plots = make_gallery04_plots
        self.make_gallery05_plots = make_gallery05_plots

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        _, job_xs_csv = os.path.split(job_xs_csv_abspath)
        job_absdir, job_zs_csv = os.path.split(job_zs_csv_abspath)
        _, job_dir = os.path.split(job_absdir)
        #
        self.job_absdir = job_absdir
        self.job_xs_csv = job_xs_csv
        self.job_zs_csv = job_zs_csv
        self.job_dir = job_dir
        if 1 and prt:
            print fmt0(just1)[0:] % ("job_absdir", job_absdir)
            print fmt0(just1)[1:] % ("job_xs_csv", job_xs_csv)
            print fmt0(just1)[1:] % ("job_zs_csv", job_zs_csv)
            print fmt0(just1)[1:] % ("job_dir", job_dir)

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        job_config_txt_abspath = os.path.join(job_absdir, job_config_txt)
        if 1 and prt:
            #rint fmt0(just1)[1:] % ("job_config_txt", job_config_txt)
            print fmt0(just1)[1:] % (
                "job_config_txt_abspath", job_config_txt_abspath)

        ###job_cfg_ps = self.get_job_configuration_parameters(
        ###    job_config_txt_abspath)
        job_cfg_ps = (
            get_job_configuration_parameters(ngn, job_config_txt_abspath))
        self.job_cfg_ps = job_cfg_ps
        if 1 and prt:
            print fmt1(just1)[0:] % ("job_cfg_ps", job_cfg_ps)

        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # generate the tow center x-coordinate reference values for the
        # within-course tows numbered from 1 to "number_of_strips" and the
        # left and right between-courses neighboring strips numbered 0 and
        # "number_of_strips + 1".
        #
        # And, generate the tow edge x-coordinate reference values for the
        # within-course tows numbered from 1 to "number_of_strips"

        ps = make_x_coordinate_roi_reference_values(ngn)
        if 0 and prt:
            print fmt1(just1)[0:] % ("ps", ps)

        self.tow_ids = ps['tow_ids']
        self.tow_center_xrefs = ps['tow_center_xrefs']
        self.tow_edge_ids = ps['tow_edge_ids']
        self.tow_edge_xrefs = ps['tow_edge_xrefs']

        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        if preview:
            preview_absdir = os.path.join(job_absdir, self.preview_dir)
            self.preview_absdir = preview_absdir
            if 1 and prt:
                print fmt1(just1)[0:] % ("write_to_preview_dir",
                    write_to_preview_dir)
            if 1 and prt:
                print fmt0(just0)[0:] % ("job_absdir", job_absdir)
                print fmt0(just0)[1:] % ("self.preview_dir", self.preview_dir)
                print fmt0(just0)[1:] % ("preview_absdir", preview_absdir)

            None if os.path.isdir(preview_absdir) else (
                os.makedirs(preview_absdir))

        if write_to_results_dir:
            results_absdir = os.path.join(job_absdir, self.results_dir)
            self.results_absdir = results_absdir
            if 1 and prt:
                print fmt1(just1)[0:] % ("write_to_results_dir",
                    write_to_results_dir)
            if 1 and prt:
                print fmt0(just0)[0:] % ("job_absdir", job_absdir)
                print fmt0(just0)[1:] % ("self.results_dir", self.results_dir)
                print fmt0(just0)[1:] % ("results_absdir", results_absdir)

            None if os.path.isdir(results_absdir) else (
                os.makedirs(results_absdir))

            gallery00_dir = '_gallery00'
            gallery00_absdir = os.path.join(results_absdir, gallery00_dir)
            self.gallery00_dir = gallery00_dir
            self.gallery00_absdir = gallery00_absdir

            gallery01_dir = '_gallery01'
            gallery01_absdir = os.path.join(results_absdir, gallery01_dir)
            self.gallery01_dir = gallery01_dir
            self.gallery01_absdir = gallery01_absdir

            gallery02_dir = '_gallery02'
            gallery02_absdir = os.path.join(results_absdir, gallery02_dir)
            self.gallery02_dir = gallery02_dir
            self.gallery02_absdir = gallery02_absdir

            gallery03_dir = '_gallery03'
            gallery03_absdir = os.path.join(results_absdir, gallery03_dir)
            self.gallery03_dir = gallery03_dir
            self.gallery03_absdir = gallery03_absdir

            gallery04_dir = '_gallery04'
            gallery04_absdir = os.path.join(results_absdir, gallery04_dir)
            self.gallery04_dir = gallery04_dir
            self.gallery04_absdir = gallery04_absdir

            gallery05_dir = '_gallery05'
            gallery05_absdir = os.path.join(results_absdir, gallery05_dir)
            self.gallery05_dir = gallery05_dir
            self.gallery05_absdir = gallery05_absdir

            None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        results_fields_ends = make_results_fields_ends(self)
        self.results_fields_ends = results_fields_ends
        if 1 and prt:
            print fmt1(just1)[0:] % ("results_fields_ends[:9]",
                prt_list(results_fields_ends[:9], -20))

        None if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        results_ends_csv = (
            job_xs_csv.replace('z', '')
            .replace('.txt', '')
            .replace('.csv', '_results_ends.csv')
        )
        self.results_ends_csv = results_ends_csv
        if 1 and prt:
            print fmt0(just1)[0:] % ("results_ends_csv", results_ends_csv)

        #results_ends_csv_pyapp_ver = results_ends_csv.replace("ends.csv",
        #    "ends_pyv%.3i.csv" % self.version_py)
        #self.results_ends_csv_pyapp_ver = results_ends_csv_pyapp_ver

        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        ps_gap_init = pd.Series(*zip(*[
            [(-1, -1), 'gap_tows'],
            ['', 'gap_id'],
            #
            # negative edge information
            [np.nan, 'edge_neg_x_midp_nnan_idx'],
            [np.nan, 'edge_neg_x_midp_idx'],
            [np.nan, 'edge_neg_dzdx'],
            [np.nan, 'edge_neg_x_midp'],
            [np.nan, 'edge_neg_z'],
            #
            # positive edge information
            [np.nan, 'edge_pos_x_midp_nnan_idx'],
            [np.nan, 'edge_pos_x_midp_idx'],
            [np.nan, 'edge_pos_dzdx'],
            [np.nan, 'edge_pos_x_midp'],
            [np.nan, 'edge_pos_z'],
            #
            # gap information
            [np.nan, 'gap_flat_x_nnan_idx'],
            [np.nan, 'gap_flat_x_idx'],
            [np.nan, 'gap_flat_x'],
            [np.nan, 'gap_flat_z'],
            [np.nan, 'gap_center_z'],
            [np.nan, 'gap_width'],
        ]))
        self.ps_gap_init = ps_gap_init
        if 1 and prt:
            print fmt1(just1)[0:] % ("ps_gap_init", ps_gap_init)

        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        results_fields_gap = make_results_fields_gaps(self)
        self.results_fields_gap = results_fields_gap
        if 1 and prt:
            print fmt1(just1)[0:] % ("results_fields_gap[:9]",
                prt_list(results_fields_gap[:9], -20))

        None if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        results_gap_csv = (
            job_zs_csv.replace('z', '')
            .replace('.txt', '')
            .replace('.csv', '_results_gap.csv')
        )
        self.results_gap_csv = results_gap_csv
        if 1 and prt:
            print fmt1(just1)[0:] % ("results_gap_csv", results_gap_csv)

        #results_gap_csv_pyapp_ver = results_gap_csv.replace("gap.csv",
        #    "gap_pyv%.3i.csv" % self.version_py)
        #self.results_gap_csv_pyapp_ver = results_gap_csv_pyapp_ver

        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        #f 1 and prt:
        #   print fmt1(just0)[0:] % ("(updt) self", prt_obj(self))

        #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
        if prt_ or prt__:
            print "\n%s" % (mult_str * self.mult)
            print fmt0(just0)[1:] % ('(end) %s' % def_str, self.in_)
        None if 1 else sys.exit()


ngn = Ngn()


# (above)  python program container classes
#==.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===
# (below) 'Application' functions


def get_located_tow_end_events_dataset(pd_src_us):
    """
    Returns two Pandas DataFrame containing located tow presence & end events.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'get_located_tow_end_events_dataset'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("pd_src_us.shape", pd_src_us.shape)
        print fmt1(just1)[1:] % ("pd_src_us.columns.values",
            pd_src_us.columns.values)
    if 1 and prt:
        hd, tl = (4, 4) if 1 else (98, 98)
        print fmt1(just1)[0:] % ("pd_src_us.head(%i)" % hd, pd_src_us.head(hd))
        print fmt1(just1)[1:] % ("pd_src_us.tail(%i)" % tl, pd_src_us.tail(tl))
    if 0 and prt:
        print fmt1(just1)[0:] % ("pd_src_us", pd_src_us)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # get the pd_src_us_tow_present dataset

### ### using ... CNRC20170216_scan_parallel_base_layer_part3 ...
### #
### #   ... pd_src_us.columns.values             ...
### # ['MeastID' 'Ply ' 'Course' 'U-Roller' 'U-Sensor'
### #  'TowPresentBitmap''GapPresentBitmap'
### #  'ProfileID' 'Notes'
### #  'X' 'Y' 'Z' 'C' 'n1' 'n2' 'n3' 'j1' 'j2' 'j3' 'j4' 'j5' 'j6']
### #
### ### using ... CNRC20170522_06_PlyDrops ...
### ###       ... CNRC20170522_30ThouLap ...
### #
### # ... pd_src_us.columns.values             ...
### # ['MeastID' 'Ply' 'Course' 'U-Roller' 'U-Sensor'
### # 'U-TowChangeProgramed' 'TowPresentBitmap' 'GapPresentBitmap'
### # **********************
### # 'ProfileID' 'Notes'
### # 'X' 'Y' 'Z' 'C' 'n1' 'n2' 'n3' 'j1' 'j2' 'j3' 'j4' 'j5' 'j6']
###
### if 1 and prt:
###     print fmt1(just1)[0:] % ("ngn.results_fields_ends",
###         ngn.results_fields_ends)
###
### if 'CNRC20170216_scan_parallel_base_layer_part3' in ngn.job_dir:
###     # this 'pd_src_us' does not include the 'U-TowChangeProgramed' field
###
###     ngn.results_fields_ends = [s for s in ngn.results_fields_ends
###         if 'U-TowChangeProgramed' not in s]
###     if 0 and prt:
###         print fmt1(just1)[0:] % ("(updt) ngn.results_fields_ends",
###             ngn.results_fields_ends)
###
###     tow_key_names_init = [s for s in ngn.results_fields_ends[:4]
###         if 'TowPresentBits' not in s]
###     #f 1 and prt:
###     #   print fmt1(just1)[0:] % ("tow_key_names_init", tow_key_names_init)
###     #
###     # ... tow_key_names_init                   ...
###     # ['ProfileID', 'MeastID', 'U-Sensor']
###
### else:
    if 1:

        tow_key_names_init = [s for s in ngn.results_fields_ends[:5]
            if 'TowPresentBits' not in s]
        #f 1 and prt:
        #   print fmt1(just1)[0:] % ("tow_key_names_init", tow_key_names_init)
        #
        # ... tow_key_names_init                   ...
        # ['ProfileID', 'MeastID', 'U-Sensor', 'U-TowChangeProgramed']

    if 1 and prt:
        print fmt1(just1)[0:] % ("tow_key_names_init", tow_key_names_init)

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    pd_src_us_tow_present = pd_src_us[tow_key_names_init].copy()
    tow_present_bits = 'TowPresentBits_Tow32toTow01'
    pd_src_us_tow_present[tow_present_bits] = (
        pd_src_us['TowPresentBitmap'].astype(np.int).apply(
            lambda i: format(i, "%.3ib" % ngn.job_cfg_ps.number_of_strips))
    )
### if 'LD90_LB65536_DF101p4-Located-20170217' in ngn.job_us_csv_abspath:
###     # this is what it "is" ...
###     pd_src_us_tow_present['meast_idx'] = pd_src_us['ProfileID'] - 1
### elif 'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath:
###     # this is what it "is" ...
###     pd_src_us_tow_present['meast_idx'] = pd_src_us['ProfileID'] - 1
### else:
    if 1:
        # this is what it "should" be ...
        pd_src_us_tow_present['meast_idx'] = pd_src_us['MeastID'] - 1

    if 1 and prt:
        hd, tl = (4, 4) if 1 else (98, 98)
        print fmt0(just1)[0:] % ("(init) pd_src_us_tow_present.shape",
            pd_src_us_tow_present.shape)
        print fmt1(just1)[1:] % ("(init) pd_src_us_tow_present.head(%i)" %
            hd, pd_src_us_tow_present.head(hd))
        print fmt1(just1)[1:] % ("(init) pd_src_us_tow_present.tail(%i)" %
            tl, pd_src_us_tow_present.tail(tl))

    tow_key_names = list(pd_src_us_tow_present.columns.values)
    if 1 and prt:
        print fmt0(just1)[0:] % ("tow_key_names", tow_key_names)

    # NOTE:
    #
    #   Dataset "ViperCNRCSimulator_20170420175016 ..." does not have any
    #   laser profiles showing that last set of tow stop ends.
    #
    #   Having at least a half-window of the above is a requirement of the
    #   data acquisition system for the algorithm to identify all tow end
    #   events.

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    tow_present_names = []
    tow_diff_names = []
    tow_present_diff_names = []
    for idx in range(ngn.job_cfg_ps.number_of_strips):
        tow_num = ngn.job_cfg_ps.number_of_strips - idx
        #
        tow_present_name = 't%.2ip' % tow_num
        pd_src_us_tow_present[tow_present_name] = pd_src_us_tow_present[
            tow_present_bits].apply(lambda x: x[idx])
        #
        tow_diff_name = 't%.2id' % tow_num
        pd_src_us_tow_present[tow_diff_name] = pd_src_us_tow_present[
            tow_present_name].astype(np.int).diff()
        pd_src_us_tow_present.ix[0, tow_diff_name] = 0
        pd_src_us_tow_present[tow_diff_name] = pd_src_us_tow_present[
            tow_diff_name].astype(np.int)
        #
        tow_present_names.append(tow_present_name)
        tow_diff_names.append(tow_diff_name)
        tow_present_diff_names.extend([tow_present_name, tow_diff_name])
        if 0 and prt:
            print fmt0(just1)[0:] % (
                "[idx, tow_num, tow_present_name, tow_diff_name]",
                [idx, tow_num, tow_present_name, tow_diff_name])

    if 1 and prt:
        hd, tl = (4, 4) if 1 else (98, 98)
        print fmt0(just1)[0:] % ("(updt) pd_src_us_tow_present.shape",
            pd_src_us_tow_present.shape)
        print fmt1(just1)[0:] % ("(updt) pd_src_us_tow_present.head(%i)" %
            hd, pd_src_us_tow_present.head(hd))
        print fmt1(just1)[1:] % ("(updt) pd_src_us_tow_present.tail(%i)" %
            tl, pd_src_us_tow_present.tail(tl))

    if 0 and prt:
        print fmt1(just1)[0:] % ("tow_present_names", tow_present_names)
    if 1 and prt:
        print fmt1(just1)[0:] % ("tow_diff_names", tow_diff_names)
    if 1 and prt:
        print fmt1(just1)[0:] % ("tow_present_diff_names",
            tow_present_diff_names)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # get the pd_src_us_tow_diff dataset

    pd_src_us_tow_diff = pd_src_us_tow_present[tow_diff_names].astype(np.float)
    pd_src_us_tow_diff = pd_src_us_tow_diff.replace(0., np.nan)
    pd_src_us_tow_diff = pd_src_us_tow_diff.dropna(how='all')
    pd_src_us_tow_diff = pd_src_us_tow_diff.replace(np.nan, 0.).astype(np.int)
    if 1 and prt:
        print fmt0(just1)[0:] % ("pd_src_us_tow_diff.shape",
            pd_src_us_tow_diff.shape)
### #f 1 and prt:
### #   print fmt0(just1)[0:] % (
### #       "'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath",
### #       'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath)
### #   tow_diff_names_ = (
### #       [s for s in tow_diff_names
### #           if np.int(s.replace('t', '').replace('d', '')) >= 9 and
### #           np.int(s.replace('t', '').replace('d', '')) <= 24]
### #       if 'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath
### #       else tow_diff_names
### #   )
### #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff[tow_diff_names_]",
### #       pd_src_us_tow_diff[tow_diff_names_])
    if 0 and prt:
        print fmt1(just1)[0:] % ("pd_src_us_tow_diff[tow_diff_names]",
            pd_src_us_tow_diff[tow_diff_names])
    if 0 and prt:
        print fmt1(just1)[0:] % ("pd_src_us_tow_diff[tow_diff_names].head()",
            pd_src_us_tow_diff[tow_diff_names].head())
        print fmt1(just1)[1:] % ("pd_src_us_tow_diff[tow_diff_names].tail()",
            pd_src_us_tow_diff[tow_diff_names].tail())

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    tow_diff_idxs = pd_src_us_tow_diff.index
    if 0 and prt:
        print fmt0(just1)[0:] % (
            "pd_src_us_tow_present.ix[tow_diff_idxs, tow__key_names].shape",
            pd_src_us_tow_present.ix[tow_diff_idxs, tow_key_names].shape)
        print fmt1(just1)[1:] % (
            "pd_src_us_tow_present.ix[tow_diff_idxs, tow_key_names]",
            pd_src_us_tow_present.ix[tow_diff_idxs, tow_key_names])

    None if 1 else sys.exit()

    pd_src_us_tow_diff = pd.concat([
        pd_src_us_tow_present.ix[tow_diff_idxs, tow_key_names],
        pd_src_us_tow_diff,
    ], axis=1)
    pd_src_us_tow_diff.reset_index(drop=True, inplace=True)
    if 1 and prt:
        print fmt0(just1)[0:] % ("(updt) pd_src_us_tow_diff.shape",
            pd_src_us_tow_diff.shape)

### #f 1 and prt:
### #   print fmt0(just1)[0:] % (
### #       "'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath",
### #       'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath)
### #   tow_diff_names_ = tow_key_names + (
### #       [s for s in tow_diff_names
### #           if np.int(s.replace('t', '').replace('d', '')) >= 9 and
### #           np.int(s.replace('t', '').replace('d', '')) <= 24]
### #       if 'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath
### #       else tow_diff_names
### #   )
### #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff[tow_diff_names_]",
### #       pd_src_us_tow_diff[tow_diff_names_])
    if 0 and prt:
        print fmt1(just1)[0:] % ("pd_src_us_tow_diff[tow_diff_names]",
            pd_src_us_tow_diff[tow_diff_names])
    if 0 and prt:
        print fmt1(just1)[0:] % (
            "pd_src_us_tow_diff[tow_key_names + tow_diff_names].head()",
            pd_src_us_tow_diff[tow_key_names + tow_diff_names].head())
        print fmt1(just1)[1:] % (
            "pd_src_us_tow_diff[tow_key_names + tow_diff_names].tail()",
            pd_src_us_tow_diff[tow_key_names + tow_diff_names].tail())

    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return (pd_src_us_tow_present, pd_src_us_tow_diff, tow_key_names,
        tow_diff_names)


def get_located_tow_gap_events_dataset(ngn, pd_src_us):
    """
    Returns a Pandas DataFrame containing located gap presence & gap events.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'get_located_tow_gap_events_dataset'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("pd_src_us.shape", pd_src_us.shape)
        print fmt1(just1)[1:] % ("pd_src_us.columns.values",
            pd_src_us.columns.values)
    if 1 and prt:
        hd, tl = (4, 4) if 1 else (98, 98)
        print fmt1(just1)[0:] % ("pd_src_us.head(%i)" % hd, pd_src_us.head(hd))
        print fmt1(just1)[1:] % ("pd_src_us.tail(%i)" % tl, pd_src_us.tail(tl))
    if 0 and prt:
        print fmt1(just1)[0:] % ("pd_src_us", pd_src_us)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # get the pd_src_us_gap_present dataset

    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.results_fields_gap",
            ngn.results_fields_gap)

    gap_any_names_init = [s for s in ngn.results_fields_gap[:4]
        if 'GapPresentBits' not in s]
    if 1 and prt:
        print fmt1(just1)[0:] % ("gap_any_names_init", gap_any_names_init)

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    pd_src_us_gap_present = pd_src_us[gap_any_names_init].copy()
    gap_present_bits = 'GapPresentBits_Gap3233toGap0001'
    gap_present_bitmap = 'GapPresentBitmap'
    pd_src_us_gap_present[gap_present_bits] = (
        pd_src_us[gap_present_bitmap].astype(np.int).apply(lambda i:
            format(i, "%.3ib" % (ngn.job_cfg_ps.number_of_strips + 1)))
    )
    pd_src_us_gap_present['meast_idx'] = pd_src_us['MeastID'] - 1

    if 1 and prt:
        hd, tl = (4, 4) if 1 else (98, 98)
        print fmt0(just1)[0:] % ("(init) pd_src_us_gap_present.shape",
            pd_src_us_gap_present.shape)
        print fmt1(just1)[1:] % ("(init) pd_src_us_gap_present.head(%i)" %
            hd, pd_src_us_gap_present.head(hd))
        print fmt1(just1)[1:] % ("(init) pd_src_us_gap_present.tail(%i)" %
            tl, pd_src_us_gap_present.tail(tl))

    gap_any_names = list(pd_src_us_gap_present.columns.values)
    if 1 and prt:
        print fmt0(just1)[0:] % ("gap_any_names", gap_any_names)

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    gap_present_names = []
    for idx in range(ngn.job_cfg_ps.number_of_strips + 1):
        gap_num0 = ngn.job_cfg_ps.number_of_strips - idx
        gap_num1 = ngn.job_cfg_ps.number_of_strips - idx + 1
        #
        gap_present_name = 'Gap%.2i%.2iPresent' % (gap_num0, gap_num1)
        pd_src_us_gap_present[gap_present_name] = pd_src_us_gap_present[
            gap_present_bits].apply(lambda x: x[idx])
        #
        gap_present_names.append(gap_present_name)
        if 1 and prt:
            print fmt0(just1)[0:] % (
                "[idx, gap_num0, gap_num1, gap_present_name]",
                [idx, gap_num0, gap_num1, gap_present_name])

    if 1 and prt:
        hd, tl = (4, 4) if 1 else (98, 98)
        print fmt0(just1)[0:] % ("(updt) pd_src_us_gap_present.shape",
            pd_src_us_gap_present.shape)
        print fmt1(just1)[0:] % ("(updt) pd_src_us_gap_present.head(%i)" %
            hd, pd_src_us_gap_present.head(hd))
        print fmt1(just1)[1:] % ("(updt) pd_src_us_gap_present.tail(%i)" %
            tl, pd_src_us_gap_present.tail(tl))

    if 1 and prt:
        print fmt1(just1)[0:] % ("gap_present_names", gap_present_names)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # get the pd_src_us_gap_any dataset

    mask_gap_present_bitmap_gt0 = pd_src_us[gap_present_bitmap] > 0.
    where_gap_present_bitmap_gt0 = np.where(mask_gap_present_bitmap_gt0)[0]
    if 1 and prt:
        print fmt1(just1)[0:] % ("where_gap_present_bitmap_gt0",
            where_gap_present_bitmap_gt0)
    None if 1 else sys.exit()

    ##pd_src_us_gap_any = (
    ##    pd_src_us_gap_present.iloc[where_gap_present_bitmap_gt0, :].copy())
    pd_src_us_gap_any = pd_src_us_gap_present.ix[
        where_gap_present_bitmap_gt0, gap_any_names].copy()
    pd_src_us_gap_any.reset_index(drop=True, inplace=True)
    ##f 1 and prt:
    ##   print fmt0(just1)[0:] % ("pd_src_us_gap_any.shape",
    ##       pd_src_us_gap_any.shape)
    ##f 1 and prt:
    ##   print fmt0(just1)[0:] % (
    ##       "'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath",
    ##       'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath)
    ##   gap_present_names_ = gap_any_names + (
    ##       [s for s in gap_present_names if np.int(s.replace(
    ##           'Gap', '').replace('Present', '')) >= 809 and
    ##           np.int(s.replace('Gap', '').replace('Present', '')) <= 2425]
    ##       if 'ViperCNRCSimulator_20170420175016' in ngn.job_us_csv_abspath
    ##       else gap_present_names
    ##   )
    ##   #rint fmt1(just1)[0:] % ("pd_src_us_gap_any[gap_present_names_]",
    ##   #   pd_src_us_gap_any[gap_present_names_])
    ##   hd, tl = (4, 4) if 1 else (10, 10)
    ##   print fmt1(just1)[0:] % (
    ##       "pd_src_us_gap_any[gap_present_names_].head(%i)" % hd,
    ##       pd_src_us_gap_any[gap_present_names_].head(hd))
    ##   print fmt1(just1)[0:] % (
    ##       "pd_src_us_gap_any[gap_present_names_].tail(%i)" % tl,
    ##       pd_src_us_gap_any[gap_present_names_].tail(tl))
    ##
    ##pd_src_us_gap_any = pd_src_us_gap_any.ix[:, gap_any_names].copy()
    if 1 and prt:
        print fmt0(just1)[0:] % ("pd_src_us_gap_any.shape",
            pd_src_us_gap_any.shape)
    if 1 and prt:
        hd, tl = (4, 4) if 1 else (10, 10)
        print fmt1(just1)[0:] % ("pd_src_us_gap_any.head(%i)" % hd,
            pd_src_us_gap_any.head(hd))
        print fmt1(just1)[0:] % ("pd_src_us_gap_any.tail(%i)" % tl,
            pd_src_us_gap_any.tail(tl))

    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return pd_src_us_gap_present, pd_src_us_gap_any


def preview_laser_profiles_dataset_xs(np_src_xs, job_csv,
np_src_xs_str='np_src_xs'):
    """
    Genereates 'preview' statistics & graphics for dataset xs profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'preview_laser_profiles_dataset_xs'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs.shape", np_src_xs.shape)
        #rint fmt1(just1)[1:] % ("np_src_xs[:10, :10]", np_src_xs[:10, :10])

    np_src_xs_uniqs, np_src_xs_uniq_cnts = (
        np.unique(np_src_xs, return_counts=True))
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_uniqs.shape",
            np_src_xs_uniqs.shape)
        print fmt0(just1)[1:] % ("np_src_xs_uniqs[:10]", np_src_xs_uniqs[:10])
        print fmt0(just1)[1:] % ("np_src_xs_uniqs[-10:]",
            np_src_xs_uniqs[-10:])

    np_src_xs_uniqs_and_cnts = (
        np.vstack([np_src_xs_uniqs[::-1], np_src_xs_uniq_cnts[::-1]]))
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_uniqs_and_cnts.T.shape",
            np_src_xs_uniqs_and_cnts.T.shape)
        print fmt1(just1)[1:] % ("np_src_xs_uniqs_and_cnts.T",
            np_src_xs_uniqs_and_cnts.T)
        None if 1 else sys.exit()

    np_src_xs_uniqs_min = np_src_xs_uniqs.min()
    np_src_xs_uniqs_max = np_src_xs_uniqs.max()
    np_src_xs_uniqs_mid = (np_src_xs_uniqs_max + np_src_xs_uniqs_max) / 2.
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_uniqs_min",
            np_src_xs_uniqs_min)
        print fmt0(just1)[1:] % ("np_src_xs_uniqs_mid",
            np_src_xs_uniqs_mid)
        print fmt0(just1)[1:] % ("np_src_xs_uniqs_max",
            np_src_xs_uniqs_max)

    np_src_xs_uniq_cnts_min = np_src_xs_uniq_cnts.min()
    np_src_xs_uniq_cnts_mean = np_src_xs_uniq_cnts.astype(np.float).mean()
    np_src_xs_uniq_cnts_max = np_src_xs_uniq_cnts.max()
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_uniq_cnts_min",
            np_src_xs_uniq_cnts_min)
        print fmt0(just1)[1:] % ("np_src_xs_uniq_cnts_mean",
            np_src_xs_uniq_cnts_mean)
        print fmt0(just1)[1:] % ("np_src_xs_uniq_cnts_max",
            np_src_xs_uniq_cnts_max)

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    np_src_xs_uniqs_midp = (
        (np_src_xs_uniqs[1:] + np_src_xs_uniqs[:-1]) / 2.)
    if 0 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_uniqs_midp.shape",
            np_src_xs_uniqs_midp.shape)
        #rint fmt1(just1)[1:] % ("np_src_xs_uniqs_midp",
        #   np_src_xs_uniqs_midp)
        None if 1 else sys.exit()

    np_src_xs_uniq_cnts_diff = np.diff(np_src_xs_uniq_cnts)
    np_src_xs_uniqs_diff = np.diff(np_src_xs_uniqs)
    np_src_xs_uniq_cnts_diff_over_uniqs_diff = (
        np_src_xs_uniq_cnts_diff / np_src_xs_uniqs_diff)

    np_src_xs_uniqs_diff_and_cnts_diff = (
        np.vstack([np_src_xs_uniq_cnts_diff[::-1], np_src_xs_uniqs_diff[::-1],
        np_src_xs_uniq_cnts_diff_over_uniqs_diff[::-1]]))
    if 0 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_uniqs_diff_and_cnts_diff.T.shape",
            np_src_xs_uniqs_diff_and_cnts_diff.T.shape)
        print fmt1(just1)[1:] % ("np_src_xs_uniqs_diff_and_cnts_diff.T",
            np_src_xs_uniqs_diff_and_cnts_diff.T)
        None if 1 else sys.exit()

    np_src_xs_uniq_cnts_diff_over_uniqs_diff_min = (
        np_src_xs_uniq_cnts_diff_over_uniqs_diff.min())
    np_src_xs_uniq_cnts_diff_over_uniqs_diff_mean = (
        np_src_xs_uniq_cnts_diff_over_uniqs_diff.mean())
    np_src_xs_uniq_cnts_diff_over_uniqs_diff_max = (
        np_src_xs_uniq_cnts_diff_over_uniqs_diff.max())
    if 0 and prt:
        print fmt0(just1)[0:] % (
            "np_src_xs_uniq_cnts_diff_over_uniqs_diff_min",
            np_src_xs_uniq_cnts_diff_over_uniqs_diff_min)
        print fmt0(just1)[1:] % (
            "np_src_xs_uniq_cnts_diff_over_uniqs_diff_mean",
            np_src_xs_uniq_cnts_diff_over_uniqs_diff_mean)
        print fmt0(just1)[1:] % (
            "np_src_xs_uniq_cnts_diff_over_uniqs_diff_max",
            np_src_xs_uniq_cnts_diff_over_uniqs_diff_max)
        None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if 1:
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        fig = plt.gcf()
        fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
        fig.suptitle('%s:  %s' % (ngn.job_dir, np_src_xs_str))
        fig.subplots_adjust(top=0.84, left=0.15)
        gridspec = [2, 1]
        gs = mpl.gridspec.GridSpec(*gridspec)
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        #x1 = plt.gca()
        ax1 = plt.subplot(gs[0])  # xs
        #x1.set_xlabel('Coordinate Value')
        ax1.set_ylabel('Value Count')
        ax1.xaxis.grid(True)
        ax1.set_title("\n%s\n%s" % (
            "Values:  Min = %.3f,  Mean = %.3f,  Max = %.3f" % (
                np_src_xs_uniqs_min,
                np_src_xs_uniqs_mid,
                np_src_xs_uniqs_max,
            ),
            "Value Counts:  Min = %i,  Mid = %.1f,  Max = %i" % (
                np_src_xs_uniq_cnts_min,
                np_src_xs_uniq_cnts_mean,
                np_src_xs_uniq_cnts_max,
            ),
        ), fontsize=11.0)
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax1.set_ylim((
            np_src_xs_uniq_cnts_max * -0.05, np_src_xs_uniq_cnts_max * 1.30
        ))
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax1.plot(np_src_xs_uniqs, np_src_xs_uniq_cnts, '.-', mec='none')
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        ax2 = plt.subplot(gs[1], sharex=ax1)  # dzdxs
        ax2.set_xlabel('Coordinate Value')
        ax2.set_ylabel('del Value Count/ del Value')
        ax2.xaxis.grid(True)
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax_get_xlim_mid = (np_src_xs_uniqs[-1] + np_src_xs_uniqs[0]) / 2.
        ax_get_xlim_halfrng = (np_src_xs_uniqs[-1] - np_src_xs_uniqs[0]) / 2.
        ax_get_xlim_min = ax_get_xlim_mid - ax_get_xlim_halfrng * 1.10
        ax_get_xlim_max = ax_get_xlim_mid + ax_get_xlim_halfrng * 1.10
        ax2.set_xlim((ax_get_xlim_min, ax_get_xlim_max))
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax2_y_max_abs = np.max(np.abs(
            np_src_xs_uniq_cnts_diff_over_uniqs_diff))
        ax2_y_max_abs = 1. if ax2_y_max_abs < 1. else ax2_y_max_abs
        ax2.set_ylim((-ax2_y_max_abs * 1.1, ax2_y_max_abs * 1.1))
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax2.plot(np_src_xs_uniqs_midp,
            np_src_xs_uniq_cnts_diff_over_uniqs_diff, '.-', mec='none')
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        png = job_csv.replace(".csv", "__preview_%s.png" % np_src_xs_str)
        png_abspath = os.path.join(ngn.preview_absdir, png)
        if 0 and prt:
            print fmt0(just1)[0:] % ("ngn.preview_absdir", ngn.preview_absdir)
            print fmt0(just1)[1:] % ("job_csv", job_csv)
            print fmt0(just1)[1:] % ("np_src_xs_str", np_src_xs_str)
            print fmt0(just1)[1:] % ("png", png)
        print fmt1(just1)[0:] % ("png_abspath", png_abspath)
        fig.savefig(png_abspath)
        None if 1 else plt.show()
        plt.close()
        None if 1 else sys.exit()
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def preview_laser_profiles_dataset_zs(np_src_zs, job_csv,
nan_threshold_ceiling, nan_threshold_floor, count_threshold_floor,
np_src_zs_str='np_src_zs'):
    """
    Genereates 'preview' statistics & graphics for dataset zs profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'preview_laser_profiles_dataset_zs'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs.shape", np_src_zs.shape)
        #rint fmt1(just1)[1:] % ("np_src_zs[:10, :10]", np_src_zs[:10, :10])

    np_src_zs_uniqs, np_src_zs_uniq_cnts = (
        np.unique(np_src_zs, return_counts=True))
    if 1 and prt:
        print fmt0(just1)[0:] % ("(init) np_src_zs_uniqs.shape",
            np_src_zs_uniqs.shape)
        print fmt0(just1)[1:] % ("(init) np_src_zs_uniqs[:10]",
            np_src_zs_uniqs[:10])
        print fmt0(just1)[1:] % ("(init) np_src_zs_uniqs[-10:]",
            np_src_zs_uniqs[-10:])

    if 1 and prt:
        print fmt0(just1)[0:] % ("nan_threshold_ceiling",
            nan_threshold_ceiling)
        print fmt0(just1)[1:] % ("nan_threshold_floor",
            nan_threshold_floor)

    if nan_threshold_ceiling is not None:
        np_src_zs_uniqs_mask = np_src_zs_uniqs < nan_threshold_ceiling
        np_src_zs_uniqs = np_src_zs_uniqs[np_src_zs_uniqs_mask]
        np_src_zs_uniq_cnts = np_src_zs_uniq_cnts[np_src_zs_uniqs_mask]

    if nan_threshold_floor is not None:
        np_src_zs_uniqs_mask = np_src_zs_uniqs > nan_threshold_floor
        np_src_zs_uniqs = np_src_zs_uniqs[np_src_zs_uniqs_mask]
        np_src_zs_uniq_cnts = np_src_zs_uniq_cnts[np_src_zs_uniqs_mask]

    np_src_zs_uniqs_and_cnts = (
        np.vstack([np_src_zs_uniqs[::-1], np_src_zs_uniq_cnts[::-1]]))
    if 1 and prt:
        print fmt0(just1)[0:] % ("(updt) np_src_zs_uniqs_and_cnts.T.shape",
            np_src_zs_uniqs_and_cnts.T.shape)
        print fmt1(just1)[1:] % ("(updt) np_src_zs_uniqs_and_cnts.T",
            np_src_zs_uniqs_and_cnts.T)
        None if 1 else sys.exit()

    np_src_zs_uniqs_min = np_src_zs_uniqs.min()
    np_src_zs_uniqs_max = np_src_zs_uniqs.max()
    np_src_zs_uniqs_mid = (np_src_zs_uniqs_max + np_src_zs_uniqs_max) / 2.
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_uniqs_min",
            np_src_zs_uniqs_min)
        print fmt0(just1)[1:] % ("np_src_zs_uniqs_mid",
            np_src_zs_uniqs_mid)
        print fmt0(just1)[1:] % ("np_src_zs_uniqs_max",
            np_src_zs_uniqs_max)

    np_src_zs_uniq_cnts_min = np_src_zs_uniq_cnts.min()
    np_src_zs_uniq_cnts_mean = np_src_zs_uniq_cnts.astype(np.float).mean()
    np_src_zs_uniq_cnts_max = np_src_zs_uniq_cnts.max()
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_uniq_cnts_min",
            np_src_zs_uniq_cnts_min)
        print fmt0(just1)[1:] % ("np_src_zs_uniq_cnts_mean",
            np_src_zs_uniq_cnts_mean)
        print fmt0(just1)[1:] % ("np_src_zs_uniq_cnts_max",
            np_src_zs_uniq_cnts_max)

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    np_src_zs_uniqs_midp = (
        (np_src_zs_uniqs[1:] + np_src_zs_uniqs[:-1]) / 2.)
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_uniqs_midp.shape",
            np_src_zs_uniqs_midp.shape)
        #rint fmt1(just1)[1:] % ("np_src_zs_uniqs_midp",
        #   np_src_zs_uniqs_midp)
        None if 1 else sys.exit()

    np_src_zs_uniq_cnts_diff = np.diff(np_src_zs_uniq_cnts)
    np_src_zs_uniqs_diff = np.diff(np_src_zs_uniqs)
    np_src_zs_uniq_cnts_diff_over_uniqs_diff = (
        np_src_zs_uniq_cnts_diff / np_src_zs_uniqs_diff)

    np_src_zs_uniqs_diff_and_cnts_diff = (
        np.vstack([np_src_zs_uniq_cnts_diff[::-1], np_src_zs_uniqs_diff[::-1],
        np_src_zs_uniq_cnts_diff_over_uniqs_diff[::-1]]))
    if 0 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_uniqs_diff_and_cnts_diff.T.shape",
            np_src_zs_uniqs_diff_and_cnts_diff.T.shape)
        print fmt1(just1)[1:] % ("np_src_zs_uniqs_diff_and_cnts_diff.T",
            np_src_zs_uniqs_diff_and_cnts_diff.T)
        None if 1 else sys.exit()

    np_src_zs_uniq_cnts_diff_over_uniqs_diff_min = (
        np_src_zs_uniq_cnts_diff_over_uniqs_diff.min())
    np_src_zs_uniq_cnts_diff_over_uniqs_diff_mean = (
        np_src_zs_uniq_cnts_diff_over_uniqs_diff.mean())
    np_src_zs_uniq_cnts_diff_over_uniqs_diff_max = (
        np_src_zs_uniq_cnts_diff_over_uniqs_diff.max())
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "np_src_zs_uniq_cnts_diff_over_uniqs_diff_min",
            np_src_zs_uniq_cnts_diff_over_uniqs_diff_min)
        print fmt0(just1)[1:] % (
            "np_src_zs_uniq_cnts_diff_over_uniqs_diff_mean",
            np_src_zs_uniq_cnts_diff_over_uniqs_diff_mean)
        print fmt0(just1)[1:] % (
            "np_src_zs_uniq_cnts_diff_over_uniqs_diff_max",
            np_src_zs_uniq_cnts_diff_over_uniqs_diff_max)
        None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    if count_threshold_floor is not None:
        np_src_zs_uniq_cnts_mask_floor = (
            np_src_zs_uniq_cnts < count_threshold_floor)
        #
        np_src_zs_uniqs_lt_floor = np_src_zs_uniqs.copy()
        np_src_zs_uniqs_lt_floor[~ np_src_zs_uniq_cnts_mask_floor] = np.nan
        #
        np_src_zs_uniq_cnts_lt_floor = (
            np_src_zs_uniq_cnts.astype(np.float).copy())
        np_src_zs_uniq_cnts_lt_floor[~ np_src_zs_uniq_cnts_mask_floor] = np.nan
        #
        np_src_zs_uniqs_and_cnts_lt_floor = np.vstack([
            np_src_zs_uniqs_lt_floor[np_src_zs_uniq_cnts_mask_floor],
            np_src_zs_uniq_cnts_lt_floor[np_src_zs_uniq_cnts_mask_floor],
        ])
        if 0 and prt:
            print fmt0(just1)[0:] % ("count_threshold_floor",
                count_threshold_floor)
            print fmt0(just1)[1:] % (
                "np_src_zs_uniqs_and_cnts_lt_floor.T.shape",
                np_src_zs_uniqs_and_cnts_lt_floor.T.shape)
            print fmt1(just1)[1:] % ("np_src_zs_uniqs_and_cnts_lt_floor.T",
                np_src_zs_uniqs_and_cnts_lt_floor.T)
            None if 1 else sys.exit()

        np_src_zs_uniqs_ge_floor = (
            np_src_zs_uniqs[~ np_src_zs_uniq_cnts_mask_floor])
        np_src_zs_uniq_cnts_ge_floor = (
            np_src_zs_uniq_cnts[~ np_src_zs_uniq_cnts_mask_floor])
        #
        np_src_zs_uniqs_and_cnts_ge_floor = (
            np.vstack([np_src_zs_uniqs_ge_floor[::-1],
            np_src_zs_uniq_cnts_ge_floor[::-1]]))
        if 1 and prt:
            print "\n%s" % ('*** ' * ngn.mult)
            print fmt0(just1)[1:] % ("count_threshold_floor",
                count_threshold_floor)
            print fmt0(just1)[1:] % (
                "np_src_zs_uniqs_and_cnts_ge_floor.T.shape",
                np_src_zs_uniqs_and_cnts_ge_floor.T.shape)
            print fmt1(just1)[1:] % ("np_src_zs_uniqs_and_cnts_ge_floor.T",
                np_src_zs_uniqs_and_cnts_ge_floor.T)
            print "%s" % ('*** ' * ngn.mult)
            None if 1 else sys.exit()

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        np_src_zs_uniq_cnts_mask_floor_midp = (
            np_src_zs_uniq_cnts_mask_floor[1:] &
            np_src_zs_uniq_cnts_mask_floor[:-1])

        np_src_zs_uniqs_midp_lt_floor = np_src_zs_uniqs_midp.copy()
        np_src_zs_uniqs_midp_lt_floor[
            ~ np_src_zs_uniq_cnts_mask_floor_midp] = np.nan
        #
        np_src_zs_uniq_cnts_diff_over_uniqs_diff_lt_floor = (
            np_src_zs_uniq_cnts_diff_over_uniqs_diff.copy())
        np_src_zs_uniq_cnts_diff_over_uniqs_diff_lt_floor[
            ~ np_src_zs_uniq_cnts_mask_floor_midp] = np.nan
        #
        np_src_zs_uniqs_and_cnts_midp_lt_floor = np.vstack([
            np_src_zs_uniqs_midp_lt_floor[
                np_src_zs_uniq_cnts_mask_floor_midp][::-1],
            np_src_zs_uniq_cnts_diff_over_uniqs_diff_lt_floor[
                np_src_zs_uniq_cnts_mask_floor_midp][::-1],
        ])
        if 0 and prt:
            print fmt0(just1)[0:] % (
                "np_src_zs_uniqs_and_cnts_midp_lt_floor.T.shape",
                np_src_zs_uniqs_and_cnts_midp_lt_floor.T.shape)
            print fmt1(just1)[1:] % (
                "np_src_zs_uniqs_and_cnts_midp_lt_floor.T",
                np_src_zs_uniqs_and_cnts_midp_lt_floor.T)
            None if 1 else sys.exit()

        None if 1 else sys.exit()

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if 1:
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        fig = plt.gcf()
        fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
        fig.suptitle('%s:  %s' % (ngn.job_dir, np_src_zs_str))
        fig.subplots_adjust(top=0.81, left=0.15)
        gridspec = [2, 1]
        gs = mpl.gridspec.GridSpec(*gridspec)
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        #x1 = plt.gca()
        ax1 = plt.subplot(gs[0])  # xs
        #x1.set_xlabel('Coordinate Value')
        ax1.set_ylabel('Value Count')
        ax1.xaxis.grid(True)
        ax1.set_title("%s\n%s\n%s\n%s" % (
            "NaN Thresholds:  Floor = %s,  Ceiling = %s" % (
                nan_threshold_floor if nan_threshold_floor is None
                else "%.3f" % nan_threshold_floor,
                nan_threshold_ceiling if nan_threshold_ceiling is None
                else "%.3f" % nan_threshold_ceiling,
            ),
            "Values:  Min = %.3f,  Mean = %.3f,  Max = %.3f" % (
                np_src_zs_uniqs_min,
                np_src_zs_uniqs_mid,
                np_src_zs_uniqs_max,
            ),
            "Value Counts:  Min = %i,  Mid = %.1f,  Max = %i" % (
                np_src_zs_uniq_cnts_min,
                np_src_zs_uniq_cnts_mean,
                np_src_zs_uniq_cnts_max,
            ),
            "Count Threshold Floor = %s" % (
                count_threshold_floor if count_threshold_floor is None
                else "%.0f" % count_threshold_floor,
            ),
        ), fontsize=11.0)
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax1.set_ylim((
            np_src_zs_uniq_cnts_max * -0.05, np_src_zs_uniq_cnts_max * 1.30
        ))
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax1.plot(np_src_zs_uniqs, np_src_zs_uniq_cnts, '.-', mec='none')
        #
        if count_threshold_floor is not None:
            ax1.plot(np_src_zs_uniqs_lt_floor, np_src_zs_uniq_cnts_lt_floor,
                'ro-', mec='none')
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        ax2 = plt.subplot(gs[1], sharex=ax1)  # dzdxs
        ax2.set_xlabel('Coordinate Value')
        ax2.set_ylabel('del Value Count/ del Value')
        ax2.xaxis.grid(True)
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax_get_xlim_mid = (np_src_zs_uniqs[-1] + np_src_zs_uniqs[0]) / 2.
        ax_get_xlim_halfrng = (np_src_zs_uniqs[-1] - np_src_zs_uniqs[0]) / 2.
        ax_get_xlim_min = ax_get_xlim_mid - ax_get_xlim_halfrng * 1.10
        ax_get_xlim_max = ax_get_xlim_mid + ax_get_xlim_halfrng * 1.10
        ax2.set_xlim((ax_get_xlim_min, ax_get_xlim_max))
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax2_y_max_abs = np.max(np.abs(
            np_src_zs_uniq_cnts_diff_over_uniqs_diff))
        ax2_y_max_abs = 1. if ax2_y_max_abs < 1. else ax2_y_max_abs
        ax2.set_ylim((-ax2_y_max_abs * 1.1, ax2_y_max_abs * 1.1))
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax2.plot(np_src_zs_uniqs_midp,
            np_src_zs_uniq_cnts_diff_over_uniqs_diff, '.-', mec='none')
        #
        if count_threshold_floor is not None:
            ax2.plot(np_src_zs_uniqs_midp_lt_floor,
                np_src_zs_uniq_cnts_diff_over_uniqs_diff_lt_floor,
                'ro-', mec='none')
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        png = job_csv.replace(".csv", "__preview_%s.png" % np_src_zs_str)
        png_abspath = os.path.join(ngn.preview_absdir, png)
        if 0 and prt:
            print fmt0(just1)[0:] % ("ngn.preview_absdir", ngn.preview_absdir)
            print fmt0(just1)[1:] % ("job_csv", job_csv)
            print fmt0(just1)[1:] % ("np_src_zs_str", np_src_zs_str)
            print fmt0(just1)[1:] % ("png", png)
        print fmt1(just1)[0:] % ("png_abspath", png_abspath)
        fig.savefig(png_abspath)
        None if 1 else plt.show()
        plt.close()
        None if 1 else sys.exit()
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def preview_laser_profile_zs_image(np_src_zs, job_csv,
nan_threshold_ceiling=None, nan_threshold_floor=None, title=None,
np_src_zs_str='np_src_zs'):
    """
    Genereates 'preview' statistics & graphics for dataset (xs & zs) profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'preview_laser_profile_zs_image'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs.shape", np_src_zs.shape)

    # "fix" the np.nan values to a chosen value for the image plot
    image = np_src_zs.copy()
    image[np.isnan(image)] = np.nanmax(np_src_zs)
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "np.sum(np.isnan(image))", np.sum(np.isnan(image)))
        print fmt0(just1)[1:] % (
            "np.nanmin(image)", np.nanmin(image))
        print fmt0(just1)[1:] % (
            "np.nanmax(image)", np.nanmax(image))
    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    if nan_threshold_ceiling is None:
        nan_threshold_ceiling = np.nanmax(image)
    if nan_threshold_floor is None:
        nan_threshold_floor = np.nanmin(image)
    image = np.clip(image, nan_threshold_floor, nan_threshold_ceiling)
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "nan_threshold_ceiling", nan_threshold_ceiling)
        print fmt0(just1)[1:] % (
            "nan_threshold_floor", nan_threshold_floor)

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
    img_png = job_csv.replace(".csv",
        "__preview_%s_image.png" % np_src_zs_str)
    img_png_abspath = os.path.join(ngn.preview_absdir, img_png)
    if 0 and prt:
        print fmt0(just1)[0:] % ("ngn.preview_absdir", ngn.preview_absdir)
        print fmt0(just1)[1:] % ("job_csv", job_csv)
        print fmt0(just1)[1:] % ("np_src_zs_str", np_src_zs_str)
        print fmt0(just1)[1:] % ("img_png", img_png)
    print fmt1(just1)[0:] % ("img_png_abspath", img_png_abspath)

    title = ('Image Map of Z-coordinate Values\n%s' %
        os.path.join(ngn.job_dir, ngn.job_zs_csv) if title is None else title)
    cmap = 'jet'
    colorbar = True if 1 else False
    xlabel = 'Measurement Index (Y)'
    ylabel = 'Profile Index (X)'
    png = img_png_abspath if 1 else None
    imshow(image.T, title=title, cmap=cmap, xticklabel=True,
        yticklabel=True, xlabel=xlabel, ylabel=ylabel,
        colorbar=colorbar, png=png)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def preview_laser_profile(indy, np_src_xs, np_src_zs,
nan_threshold_ceiling, nan_threshold_floor,
median_filter_size, gaussian_filter_sigma, dzdxs_threshold,
course_edge_roi_lf, course_edge_roi_rt, make_plot=False,
fix_plot_z_max=None, fix_plot_z_min=None,
fix_plot_dz_max=None, fix_plot_dz_min=None):
    """
    Genereates 'preview' graphics for dataset xs versus zs profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'preview_laser_profile'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    ps = pd.Series({
        'course_edge_roi_lf_x': np.nan,
        'course_edge_roi_rt_x': np.nan,
    })
    if 0 and prt:
        print fmt1(just1)[0:] % ("(init) ps", ps)

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("indy", indy)

    np_src_xs_indy = np_src_xs[indy, :]
    np_src_zs_indy = np_src_zs[indy, :]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_indy.shape", np_src_xs_indy.shape)
        print fmt0(just1)[1:] % ("np_src_zs_indy.shape", np_src_zs_indy.shape)

    np_src_xs_zs_indy_all_nnan = (
        (~ np.isnan(np_src_xs_indy).any()) &
        (~ np.isnan(np_src_zs_indy).any()))
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_zs_indy_all_nnan",
            np_src_xs_zs_indy_all_nnan)

    if np_src_xs_zs_indy_all_nnan:
        np_src_xs_indy_midp = (np_src_xs_indy[1:] + np_src_xs_indy[:-1]) / 2.
        np_src_xs_indy_diff = (np_src_xs_indy[1:] - np_src_xs_indy[:-1])
        np_src_zs_indy_diff = (np_src_zs_indy[1:] - np_src_zs_indy[:-1])
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_src_xs_indy_midp.shape",
                np_src_xs_indy_midp.shape)
            print fmt0(just1)[1:] % ("np_src_zs_indy_diff.shape",
                np_src_zs_indy_diff.shape)
            print fmt0(just1)[1:] % ("np_src_zs_indy_diff.shape",
                np_src_zs_indy_diff.shape)

        np_src_xs_indy_diff_all_gt0 = (np_src_xs_indy_diff > 0.).all()
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_src_xs_indy_diff_all_gt0",
                np_src_xs_indy_diff_all_gt0)

        if np_src_xs_indy_diff_all_gt0:
            np_src_xs_indy_dxs = np_src_xs_indy_diff
        else:
            # estimate the mean dx
            np_src_xs_indy_del = np_src_xs_indy[-1] - np_src_xs_indy[0]
            np_src_xs_indy_dxs = (np_src_xs_indy_del /
                (np_src_xs_indy.shape[0] - 1.))
            if 1 and prt:
                print fmt0(just1)[0:] % ("np_src_xs_indy[-1]",
                    np_src_xs_indy[-1])
                print fmt0(just1)[1:] % ("np_src_xs_indy[0]",
                    np_src_xs_indy[0])
                print fmt0(just1)[1:] % ("np_src_xs_indy_del",
                    np_src_xs_indy_del)
                print fmt0(just1)[1:] % ("np_src_xs_indy.shape[0] - 1.",
                    np_src_xs_indy.shape[0] - 1.)
                print fmt0(just1)[0:] % ("np_src_xs_indy_dxs",
                    np_src_xs_indy_dxs)
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_src_xs_indy_dxs.shape[0]",
                np_src_xs_indy_dxs.shape[0])

        np_src_dzdxs_indy = np_src_zs_indy_diff / np_src_xs_indy_dxs
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_src_dzdxs_indy.shape",
                np_src_dzdxs_indy.shape)
        None if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        np_fltr_medn_zs_indy = (
            ndi.median_filter(np_src_zs_indy, median_filter_size))

        np_fltr_medn_zs_indy_diff = (
            np_fltr_medn_zs_indy[1:] - np_fltr_medn_zs_indy[:-1])

        np_fltr_medn_dzdxs_indy = (
            np_fltr_medn_zs_indy_diff / np_src_xs_indy_dxs)
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_fltr_medn_dzdxs_indy.shape",
                np_fltr_medn_dzdxs_indy.shape)
        None if 1 else sys.exit()

        np_fltr_gaus_zs_indy = (
            ndi.gaussian_filter(np_src_zs_indy, gaussian_filter_sigma))

        np_fltr_gaus_zs_indy_diff = (
            np_fltr_gaus_zs_indy[1:] - np_fltr_gaus_zs_indy[:-1])

        np_fltr_gaus_dzdxs_indy = (
            np_fltr_gaus_zs_indy_diff / np_src_xs_indy_dxs)
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_fltr_gaus_dzdxs_indy.shape",
                np_fltr_gaus_dzdxs_indy.shape)
        None if 1 else sys.exit()

        np_fltr_zs_indy = ndi.gaussian_filter(ndi.median_filter(
            np_src_zs_indy, median_filter_size), gaussian_filter_sigma)

        np_fltr_zs_indy_diff = (np_fltr_zs_indy[1:] - np_fltr_zs_indy[:-1])

        np_fltr_dzdxs_indy = np_fltr_zs_indy_diff / np_src_xs_indy_dxs
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_fltr_dzdxs_indy.shape",
                np_fltr_dzdxs_indy.shape)
        None if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        np_fltr_dzdxs_indy_mask_edges_neg = ((np_fltr_dzdxs_indy < 0.) &
            ndi.binary_dilation(np_fltr_dzdxs_indy <= -dzdxs_threshold))

        np_fltr_dzdxs_indy_edges_neg = np_fltr_dzdxs_indy.copy()
        np_fltr_dzdxs_indy_edges_neg[
            ~ np_fltr_dzdxs_indy_mask_edges_neg] = np.nan

        if course_edge_roi_rt is not None:
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            course_edge_roi_rt_mask = (
                (np_src_xs_indy_midp >= course_edge_roi_rt[0]) &
                (np_src_xs_indy_midp <= course_edge_roi_rt[1])
            )
            course_edge_roi_rt_where = np.where(course_edge_roi_rt_mask)[0]
            if 1 and prt:
                print fmt0(just1)[0:] % ("course_edge_roi_rt_mask.shape",
                    course_edge_roi_rt_mask.shape)
                print fmt0(just1)[1:] % ("course_edge_roi_rt_where.shape",
                    course_edge_roi_rt_where.shape)
                print fmt1(just1)[1:] % ("course_edge_roi_rt_where",
                    course_edge_roi_rt_where)
            if 1 and prt:
                print fmt1(just1)[0:] % (
                    "np_src_xs_indy_midp[course_edge_roi_rt_where]",
                    np_src_xs_indy_midp[course_edge_roi_rt_where])
            if 1 and prt:
                print fmt1(just1)[1:] % (
                    "np_fltr_dzdxs_indy[course_edge_roi_rt_where]",
                    np_fltr_dzdxs_indy[course_edge_roi_rt_where])
            if 1 and prt:
                print fmt1(just1)[1:] % (
                    "np_fltr_dzdxs_indy_edges_neg[course_edge_roi_rt_where]",
                    np_fltr_dzdxs_indy_edges_neg[course_edge_roi_rt_where])
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            np_fltr_dzdxs_indy_edges_neg_roi_rt = (
                np_fltr_dzdxs_indy_edges_neg.copy())
            np_fltr_dzdxs_indy_edges_neg_roi_rt[
                ~ course_edge_roi_rt_mask] = np.nan
            np_fltr_dzdxs_indy_edges_neg_roi_rt_isnan_all = (
                np.isnan(np_fltr_dzdxs_indy_edges_neg_roi_rt).all())
            if 1 and prt:
                print fmt0(just1)[1:] % (
                    "np_fltr_dzdxs_indy_edges_neg_roi_rt.shape",
                    np_fltr_dzdxs_indy_edges_neg_roi_rt.shape)
            if 0 and prt:
                print fmt0(just1)[1:] % (
                    "np.isnan(np_fltr_dzdxs_indy_edges_neg_roi_rt).all()",
                    np.isnan(np_fltr_dzdxs_indy_edges_neg_roi_rt).all())
            if 1 and prt:
                print fmt0(just1)[1:] % (
                    "not np_fltr_dzdxs_indy_edges_neg_roi_rt_isnan_all",
                    not np_fltr_dzdxs_indy_edges_neg_roi_rt_isnan_all)
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            if (course_edge_roi_rt is not None and
            not np_fltr_dzdxs_indy_edges_neg_roi_rt_isnan_all):
                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
                np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin = (
                    np.nanmin(np_fltr_dzdxs_indy_edges_neg_roi_rt))
                np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin_where = (
                    np.where(np_fltr_dzdxs_indy_edges_neg_roi_rt ==
                        np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin)[0])
                np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin_xs_midp = (
                    np_src_xs_indy_midp[
                        np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin_where])
                course_edge_roi_rt_x = np.mean(
                    np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin_xs_midp)
                ps.course_edge_roi_rt_x = np.mean(course_edge_roi_rt_x)
                if 1 and prt:
                    print fmt0(just1)[0:] % (
                        "np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin",
                        np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin_where",
                        np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin_xs_midp",
                        np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin_xs_midp)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("course_edge_roi_rt_x",
                        course_edge_roi_rt_x)
                None if 1 else sys.exit()
                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        #.. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        np_fltr_dzdxs_indy_mask_edges_pos = ((np_fltr_dzdxs_indy > 0.) &
            ndi.binary_dilation(np_fltr_dzdxs_indy >= dzdxs_threshold))

        np_fltr_dzdxs_indy_edges_pos = np_fltr_dzdxs_indy.copy()
        np_fltr_dzdxs_indy_edges_pos[
            ~ np_fltr_dzdxs_indy_mask_edges_pos] = np.nan

        if course_edge_roi_lf is not None:
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            course_edge_roi_lf_mask = (
                (np_src_xs_indy_midp >= course_edge_roi_lf[0]) &
                (np_src_xs_indy_midp <= course_edge_roi_lf[1])
            )
            course_edge_roi_lf_where = np.where(course_edge_roi_lf_mask)[0]
            if 1 and prt:
                print fmt0(just1)[0:] % ("course_edge_roi_lf_mask.shape",
                    course_edge_roi_lf_mask.shape)
                print fmt0(just1)[1:] % ("course_edge_roi_lf_where.shape",
                    course_edge_roi_lf_where.shape)
                print fmt1(just1)[1:] % ("course_edge_roi_lf_where",
                    course_edge_roi_lf_where)
            if 1 and prt:
                print fmt1(just1)[0:] % (
                    "np_src_xs_indy_midp[course_edge_roi_lf_where]",
                    np_src_xs_indy_midp[course_edge_roi_lf_where])
            if 1 and prt:
                print fmt1(just1)[1:] % (
                    "np_fltr_dzdxs_indy[course_edge_roi_lf_where]",
                    np_fltr_dzdxs_indy[course_edge_roi_lf_where])
            if 1 and prt:
                print fmt1(just1)[1:] % (
                    "np_fltr_dzdxs_indy_edges_pos[course_edge_roi_lf_where]",
                    np_fltr_dzdxs_indy_edges_pos[course_edge_roi_lf_where])
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            np_fltr_dzdxs_indy_edges_pos_roi_lf = (
                np_fltr_dzdxs_indy_edges_pos.copy())
            np_fltr_dzdxs_indy_edges_pos_roi_lf[
                ~ course_edge_roi_lf_mask] = np.nan
            np_fltr_dzdxs_indy_edges_pos_roi_lf_isnan_all = (
                np.isnan(np_fltr_dzdxs_indy_edges_pos_roi_lf).all())
            if 1 and prt:
                print fmt0(just1)[1:] % (
                    "np_fltr_dzdxs_indy_edges_pos_roi_lf.shape",
                    np_fltr_dzdxs_indy_edges_pos_roi_lf.shape)
            if 0 and prt:
                print fmt0(just1)[1:] % (
                    "np.isnan(np_fltr_dzdxs_indy_edges_pos_roi_lf).all()",
                    np.isnan(np_fltr_dzdxs_indy_edges_pos_roi_lf).all())
            if 1 and prt:
                print fmt0(just1)[1:] % (
                    "not np_fltr_dzdxs_indy_edges_pos_roi_lf_isnan_all",
                    not np_fltr_dzdxs_indy_edges_pos_roi_lf_isnan_all)
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            if (course_edge_roi_lf is not None and
            not np_fltr_dzdxs_indy_edges_pos_roi_lf_isnan_all):
                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
                np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax = (
                    np.nanmax(np_fltr_dzdxs_indy_edges_pos_roi_lf))
                #
                np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax_where = (
                    np.where(np_fltr_dzdxs_indy_edges_pos_roi_lf ==
                        np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax)[0])
                #
                np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax_xs_midp = (
                    np_src_xs_indy_midp[
                        np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax_where])
                #
                course_edge_roi_lf_x = np.mean(
                    np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax_xs_midp)
                #
                ps.course_edge_roi_lf_x = np.mean(course_edge_roi_lf_x)
                if 1 and prt:
                    print fmt0(just1)[0:] % (
                        "np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax",
                        np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax_where",
                        np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax_xs_midp",
                        np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax_xs_midp)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("course_edge_roi_lf_x",
                        course_edge_roi_lf_x)
                None if 1 else sys.exit()
                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    if 1 and prt:
        print fmt1(just1)[0:] % ("ps", ps)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if 1 and make_plot:
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        fig = plt.gcf()
        fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
        fig.suptitle('Laser Profile (X, Z) %i of %i:\n%s' % (
            indy + 1, len(np_src_zs),
            os.path.join(ngn.job_dir, ngn.job_zs_csv),
        ))
        fig.subplots_adjust(
            left=0.15,  # 0.125 default
            bottom=0.1,  # 0.1 default
            right=0.9,  # 0.9 default
            top=0.83,  # 0.9 default
            wspace=0.2,  # 0.2 default
            hspace=0.3  # 0.2 default
        ) if 1 else None
        gridspec = [2, 1]
        gs = mpl.gridspec.GridSpec(*gridspec)
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        #x1 = plt.gca()
        ax1 = plt.subplot(gs[0])  # xs
        #x1.set_xlabel('X-Coordinate')
        ax1.set_ylabel('Z-Coordinate')
        ax1.xaxis.grid(True)
        ax1.set_title("%s\n%s" % (
            "NaN Thresholds:  Floor = %s,  Ceiling = %s" % (
                nan_threshold_floor if nan_threshold_floor is None
                else "%.3f" % nan_threshold_floor,
                nan_threshold_ceiling if nan_threshold_ceiling is None
                else "%.3f" % nan_threshold_ceiling,
            ),
            "Median Filter Size = %i,  Gaussian_Filter_Sigma = %.2f" % (
                median_filter_size, gaussian_filter_sigma
            ),
        ), fontsize=11.0)  # if np_src_xs_zs_indy_all_nnan else None
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        if 1 and np_src_xs_zs_indy_all_nnan:
            if fix_plot_z_min is not None and fix_plot_z_max is not None:
                ax1.set_ylim((
                    fix_plot_z_min,
                    fix_plot_z_max,
                ))
            else:
                np_fltr_zs_indy_max = (np.max(np_fltr_zs_indy)
                    if 1 or nan_threshold_ceiling is None
                    else nan_threshold_ceiling)
                np_fltr_zs_indy_min = (np.min(np_fltr_zs_indy)
                    if 1 or nan_threshold_floor is None
                    else nan_threshold_floor)
                np_fltr_zs_indy_mid = (
                    (np_fltr_zs_indy_max + np_fltr_zs_indy_min) / 2.)
                np_fltr_zs_indy_hrng = (
                    (np_fltr_zs_indy_max - np_fltr_zs_indy_min) / 2.)
                ax1.set_ylim((
                    np_fltr_zs_indy_mid - np_fltr_zs_indy_hrng * 1.20,
                    np_fltr_zs_indy_mid + np_fltr_zs_indy_hrng * 1.20
                ))
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax1.axvspan(
            course_edge_roi_lf[0], course_edge_roi_lf[1],
            color='y', alpha=0.5, lw=0
        ) if course_edge_roi_lf is not None else None
        ax1.axvspan(
            course_edge_roi_rt[0], course_edge_roi_rt[1],
            color='y', alpha=0.5, lw=0
        ) if course_edge_roi_lf is not None else None
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax1.plot(np_src_xs_indy, np_src_zs_indy, 'c,-', mec='none')
        if np_src_xs_zs_indy_all_nnan:
            ax1.plot(np_src_xs_indy, np_fltr_medn_zs_indy,
                'g,-', mec='none') if 0 else None
            ax1.plot(np_src_xs_indy, np_fltr_gaus_zs_indy,
                ',-', c='orangered', mec='none') if 0 else None
            ax1.plot(np_src_xs_indy, np_fltr_zs_indy,
                'b,-', mec='none') if 1 else None
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        ax2 = plt.subplot(gs[1], sharex=ax1)  # dzdxs
        ax2.set_xlabel('X-Coordinate')
        ax2.set_ylabel('dZ/dX')
        ax2.xaxis.grid(True)
        ax2.set_title(
            'dzdxs_threshold:  +- %.3f' % dzdxs_threshold, fontsize=11.
        ) if dzdxs_threshold is not None else None
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax2_x_max_abs = np.nanmax(np.abs(np_src_xs_indy))
        ax2.set_xlim((-ax2_x_max_abs * 1.1, ax2_x_max_abs * 1.1))
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax2.axvspan(
            course_edge_roi_lf[0], course_edge_roi_lf[1],
            color='y', alpha=0.5, lw=0
        ) if course_edge_roi_lf is not None else None
        ax2.axvspan(
            course_edge_roi_rt[0], course_edge_roi_rt[1],
            color='y', alpha=0.5, lw=0
        ) if course_edge_roi_rt is not None else None
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        if np_src_xs_zs_indy_all_nnan:
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            if fix_plot_dz_min is not None and fix_plot_dz_max is not None:
                ax2.set_ylim((
                    fix_plot_dz_min,
                    fix_plot_dz_max,
                ))
            else:
                ax2_y_max_abs = np.nanmax(np.abs(np_fltr_dzdxs_indy))
                ax2_y_max_abs = 1. if ax2_y_max_abs < 1. else ax2_y_max_abs
                ax2.set_ylim((-ax2_y_max_abs * 1.2, ax2_y_max_abs * 1.2))
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            ax2.axhline(y=-dzdxs_threshold, xmin=0.02, xmax=0.98, c='y',
                ls='dashed', lw=2.) if dzdxs_threshold is not None else None
            ax2.axhline(y=dzdxs_threshold, xmin=0.02, xmax=0.98, c='y',
                ls='dashed', lw=2.) if dzdxs_threshold is not None else None
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            ax2.plot(np_src_xs_indy_midp, np_src_dzdxs_indy,
                'c,-', mec='none') if 1 else None
            ax2.plot(np_src_xs_indy_midp, np_fltr_medn_dzdxs_indy,
                'g,-', mec='none') if 0 else None
            ax2.plot(np_src_xs_indy_midp, np_fltr_gaus_dzdxs_indy,
                ',-', c='orangered', mec='none') if 0 else None
            ax2.plot(np_src_xs_indy_midp, np_fltr_dzdxs_indy,
                'b,-', mec='none') if 1 else None
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            ax2.plot(np_src_xs_indy_midp, np_fltr_dzdxs_indy_edges_neg,
                'm,-', mec='none') if 1 else None
            ax2.plot(np_src_xs_indy_midp, np_fltr_dzdxs_indy_edges_pos,
                'r,-', mec='none') if 1 else None
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

            if (course_edge_roi_rt is not None and
            not np_fltr_dzdxs_indy_edges_neg_roi_rt_isnan_all):
                ax2.plot(course_edge_roi_rt_x,
                    np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin,
                    'ko', mec='none', ms=4.)
                ax2.text(course_edge_roi_rt_x,
                    np_fltr_dzdxs_indy_edges_neg_roi_rt_nanmin,
                    'x: %.3f' % course_edge_roi_rt_x,
                    fontsize=10., ha='center', va='top')
                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

            if (course_edge_roi_lf is not None and
            not np_fltr_dzdxs_indy_edges_pos_roi_lf_isnan_all):
                ax2.plot(course_edge_roi_lf_x,
                    np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax,
                    'ko', mec='none', ms=4.)
                ax2.text(course_edge_roi_lf_x,
                    np_fltr_dzdxs_indy_edges_pos_roi_lf_nanmax,
                    'x: %.3f' % course_edge_roi_lf_x,
                    fontsize=10., ha='center', va='bottom')
                # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        png = ngn.job_zs_csv.replace(".csv",
            "__preview_meast_%.5i.png" % (indy + 1))
        png_absdir = os.path.join(ngn.preview_absdir, '_gallery01')
        None if os.path.isdir(png_absdir) else os.makedirs(png_absdir)
        png_abspath = os.path.join(png_absdir, png)
        if 1 and prt:
            print fmt0(just1)[0:] % ("ngn.preview_absdir", ngn.preview_absdir)
            print fmt0(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
            print fmt0(just1)[1:] % ("png", png)
        print fmt1(just1)[0:] % ("png_abspath", png_abspath)
        fig.savefig(png_abspath)
        None if 1 else plt.show()
        plt.close()
        None if 1 else sys.exit()
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return ps


def preview_laser_profile_image_masked(np_src_zs, job_csv,
indy_window_beg, indy_window_end, indx_window_beg, indx_window_end,
nan_threshold_ceiling=None, nan_threshold_floor=None, title=None,
np_src_zs_str='np_src_zs'):
    """
    Genereates 'preview' statistics & graphics for dataset (xs & zs) profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'preview_laser_profile_image_masked'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs.shape", np_src_zs.shape)

    np_src_zs_all_nnan = ~ np.isnan(np_src_zs).any()
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_all_nnan", np_src_zs_all_nnan)

    if nan_threshold_ceiling is None:
        nan_threshold_ceiling = np.nanmax(np_src_zs)
    if nan_threshold_floor is None:
        nan_threshold_floor = np.nanmin(np_src_zs)
    np_src_zs_clip = (
        np.clip(np_src_zs, nan_threshold_floor, nan_threshold_ceiling))
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "nan_threshold_ceiling", nan_threshold_ceiling)
        print fmt0(just1)[1:] % (
            "nan_threshold_floor", nan_threshold_floor)

    indys = np.arange(np_src_zs.shape[0])
    if 1 and prt:
        print fmt0(just1)[0:] % ("indys[:10]", indys[:10])
        print fmt0(just1)[1:] % ("indys[-10:]", indys[-10:])
    if 1 and prt:
        print fmt0(just1)[0:] % ("indy_window_beg", indy_window_beg)
        print fmt0(just1)[1:] % ("indy_window_end", indy_window_end)

    indys_window_mask = (
        (indys >= indy_window_beg) & (indys <= indy_window_end))
    if 1 and prt:
        print fmt0(just1)[0:] % ("indys_window_mask.shape",
            indys_window_mask.shape)

    indxs = np.arange(np_src_zs.shape[1])
    if 1 and prt:
        print fmt0(just1)[0:] % ("indxs[:10]", indxs[:10])
        print fmt0(just1)[1:] % ("indxs[-10:]", indxs[-10:])
    if 1 and prt:
        print fmt0(just1)[0:] % ("indx_window_beg", indx_window_beg)
        print fmt0(just1)[1:] % ("indx_window_end", indx_window_end)

    indxs_window_mask = (
        (indxs >= indx_window_beg) & (indxs <= indx_window_end))
    if 1 and prt:
        print fmt0(just1)[0:] % ("indxs_window_mask.shape",
            indxs_window_mask.shape)

    np_src_zs_clip_window = np_src_zs_clip.copy()
    np_src_zs_clip_window[~ indys_window_mask, :] = (
        nan_threshold_floor)
    np_src_zs_clip_window[:, ~ indxs_window_mask] = (
        nan_threshold_floor)

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    # "fix" the np.nan values to a chosen value for the image plot
    image = np_src_zs_clip_window.copy()
    image[np.isnan(image)] = np.nanmax(np_src_zs)
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "np.sum(np.isnan(image))", np.sum(np.isnan(image)))
        print fmt0(just1)[1:] % (
            "np.nanmin(image)", np.nanmin(image))
        print fmt0(just1)[1:] % (
            "np.nanmax(image)", np.nanmax(image))
    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
    img_png = job_csv.replace(".csv",
        "__preview_%s_image_masked.png" % np_src_zs_str)
    img_png_abspath = os.path.join(ngn.preview_absdir, img_png)
    if 0 and prt:
        print fmt0(just1)[0:] % ("ngn.preview_absdir", ngn.preview_absdir)
        print fmt0(just1)[1:] % ("job_csv", job_csv)
        print fmt0(just1)[1:] % ("np_src_zs_str", np_src_zs_str)
        print fmt0(just1)[1:] % ("img_png", img_png)
    print fmt1(just1)[0:] % ("img_png_abspath", img_png_abspath)

    title = ('Masked Image Map of Z Values\n%s' %
        os.path.join(ngn.job_dir, ngn.job_zs_csv) if title is None else title)
    cmap = 'jet'
    colorbar = True if 1 else False
    xlabel = 'Measurement Index (Y)'
    ylabel = 'Profile Index (X)'
    png = img_png_abspath if 1 else None
    imshow(image.T, title=title, cmap=cmap, xticklabel=True,
        yticklabel=True, xlabel=xlabel, ylabel=ylabel,
        colorbar=colorbar, png=png)
    plt.close('all')  # to avoid ttk "nasty-gram"
    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def preview_laser_meast_profiles(np_indys, np_indxs, np_src_zs,
indy_window_beg, indy_window_end, indx_window_beg, indx_window_end,
nan_threshold_ceiling, nan_threshold_floor,
median_filter_size, gaussian_filter_sigma,
dzdys_threshold, make_plot=False, np_src_zs_str='np_src_zs'):
    """
    Genereates 'preview' graphics for dataset xs versus zs profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'preview_laser_meast_profiles'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("np_indys.shape", np_indys.shape)
        print fmt0(just1)[1:] % ("np_indys.dtype", np_indys.dtype)
        print fmt0(just1)[1:] % ("np_indys[:9]", np_indys[:9])
        print fmt0(just1)[1:] % ("np_indys[-9:]", np_indys[-9:])

    if 1 and prt:
        print fmt0(just1)[0:] % ("indy_window_beg", indy_window_beg)
        print fmt0(just1)[1:] % ("indy_window_end", indy_window_end)

    np_indys_window = np_indys[indy_window_beg:indy_window_end + 1]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_indys_window.shape",
            np_indys_window.shape)
        print fmt0(just1)[1:] % ("np_indys_window[:9]", np_indys_window[:9])
        print fmt0(just1)[1:] % ("np_indys_window[-9:]", np_indys_window[-9:])

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    np_indys_midp = 0.5 + np_indys[:-1]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_indys_midp.shape", np_indys_midp.shape)
        print fmt0(just1)[1:] % ("np_indys_midp.dtype", np_indys_midp.dtype)
        print fmt0(just1)[1:] % ("np_indys_midp[:9]", np_indys_midp[:9])
        print fmt0(just1)[1:] % ("np_indys_midp[-9:]", np_indys_midp[-9:])

    np_indys_midp_iwindow = np_indys_window[:-1]
    np_indys_midp_window = np_indys_midp[np_indys_midp_iwindow]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_indys_midp_iwindow.shape",
            np_indys_midp_iwindow.shape)
        print fmt0(just1)[1:] % ("np_indys_midp_iwindow[:9]",
            np_indys_midp_iwindow[:9])
        print fmt0(just1)[1:] % ("np_indys_midp_iwindow[-9:]",
            np_indys_midp_iwindow[-9:])
    if 1 and prt:
        print fmt0(just1)[1:] % ("np_indys_midp_window.shape",
            np_indys_midp_window.shape)
        print fmt0(just1)[1:] % ("np_indys_midp_window[:9]",
            np_indys_midp_window[:9])
        print fmt0(just1)[1:] % ("np_indys_midp_window[-9:]",
            np_indys_midp_window[-9:])

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs.shape", np_src_zs.shape)
    np_src_zs_all_nnan = ~ np.isnan(np_src_zs).any()
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_all_nnan", np_src_zs_all_nnan)

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    np_src_zs_window = np_src_zs[indy_window_beg:indy_window_end + 1, :]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_window.shape",
            np_src_zs_window.shape)
        print fmt1(just1)[1:] % ("np_src_zs_window[:6, :]",
            np_src_zs_window[:6, :])
        print fmt1(just1)[1:] % ("np_src_zs_window[-6:, :]",
            np_src_zs_window[-6:, :])

    np_src_zs_diff = np_src_zs[1:, :] - np_src_zs[:-1, :]
    np_src_zs_diff_window = np_src_zs_diff[np_indys_midp_iwindow]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_diff_window.shape",
            np_src_zs_diff_window.shape)
        print fmt1(just1)[1:] % ("np_src_zs_diff_window[:6, :]",
            np_src_zs_diff_window[:6, :])
        print fmt1(just1)[1:] % ("np_src_zs_diff_window[-6:, :]",
            np_src_zs_diff_window[-6:, :])

    None if 1 else sys.exit()

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    if nan_threshold_ceiling is None:
        nan_threshold_ceiling = np.nanmax(np_src_zs)
    if nan_threshold_floor is None:
        nan_threshold_floor = np.nanmin(np_src_zs)
    np_src_zs_clip = (
        np.clip(np_src_zs, nan_threshold_floor, nan_threshold_ceiling))
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "nan_threshold_ceiling", nan_threshold_ceiling)
        print fmt0(just1)[1:] % (
            "nan_threshold_floor", nan_threshold_floor)

    np_src_zs_clip_window = (
        np_src_zs_clip[indy_window_beg:indy_window_end + 1, :])
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_clip_window.shape",
            np_src_zs_clip_window.shape)
        print fmt1(just1)[1:] % ("np_src_zs_clip_window[:6, :]",
            np_src_zs_clip_window[:6, :])
        print fmt1(just1)[1:] % ("np_src_zs_clip_window[-6:, :]",
            np_src_zs_clip_window[-6:, :])

    np_src_zs_clip_diff = np_src_zs_clip[1:, :] - np_src_zs_clip[:-1, :]
    np_src_zs_clip_diff_window = np_src_zs_clip_diff[np_indys_midp_iwindow]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_clip_diff_window.shape",
            np_src_zs_clip_diff_window.shape)
        print fmt1(just1)[1:] % ("np_src_zs_clip_diff_window[:6, :]",
            np_src_zs_clip_diff_window[:6, :])
        print fmt1(just1)[1:] % ("np_src_zs_clip_diff_window[-6:, :]",
            np_src_zs_clip_diff_window[-6:, :])

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    np_src_zs_fltr = np.apply_along_axis(
        lambda x: ndi.gaussian_filter(ndi.median_filter(
            x, median_filter_size), gaussian_filter_sigma),
        0, np_src_zs_clip)

    np_src_zs_fltr_window = (
        np_src_zs_fltr[indy_window_beg:indy_window_end + 1, :])
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_fltr_window.shape",
            np_src_zs_fltr_window.shape)
        print fmt1(just1)[1:] % ("np_src_zs_fltr_window[:6, :]",
            np_src_zs_fltr_window[:6, :])
        print fmt1(just1)[1:] % ("np_src_zs_fltr_window[-6:, :]",
            np_src_zs_fltr_window[-6:, :])

    np_src_zs_fltr_diff = np_src_zs_fltr[1:, :] - np_src_zs_fltr[:-1, :]
    np_src_zs_fltr_diff_window = np_src_zs_fltr_diff[np_indys_midp_iwindow]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_zs_fltr_diff_window.shape",
            np_src_zs_fltr_diff_window.shape)
        print fmt1(just1)[1:] % ("np_src_zs_fltr_diff_window[:6, :]",
            np_src_zs_fltr_diff_window[:6, :])
        print fmt1(just1)[1:] % ("np_src_zs_fltr_diff_window[-6:, :]",
            np_src_zs_fltr_diff_window[-6:, :])

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_indxs.shape", np_indxs.shape)
        print fmt0(just1)[1:] % ("np_indxs.dtype", np_indxs.dtype)
        print fmt0(just1)[1:] % ("np_indxs[:10]", np_indxs[:10])
        print fmt0(just1)[1:] % ("np_indxs[-10:]", np_indxs[-10:])

    if 1 and prt:
        print fmt0(just1)[0:] % ("indx_window_beg", indx_window_beg)
        print fmt0(just1)[1:] % ("indx_window_end", indx_window_end)

    np_indxs_window = np_indxs[indx_window_beg:indx_window_end + 1]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_indxs_window.shape",
            np_indxs_window.shape)
        print fmt0(just1)[1:] % ("np_indxs_window[:9]", np_indxs_window[:9])
        print fmt0(just1)[1:] % ("np_indxs_window[-9:]", np_indxs_window[-9:])

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if 1 and make_plot:
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        fig = plt.gcf()
        fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
        fig.suptitle('Laser Measurement Profiles (Y, Z):\n%s' %
            os.path.join(ngn.job_dir, ngn.job_zs_csv))
        fig.subplots_adjust(
            left=0.15,  # 0.125 default
            bottom=0.1,  # 0.1 default
            right=0.9,  # 0.9 default
            top=0.83,  # 0.9 default
            wspace=0.2,  # 0.2 default
            hspace=0.3  # 0.2 default
        ) if 1 else None
        gridspec = [2, 1]
        gs = mpl.gridspec.GridSpec(*gridspec)
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        #x1 = plt.gca()
        ax1 = plt.subplot(gs[0])  # xs
        #x1.set_xlabel('Y-Coordinate')
        ax1.set_ylabel('Z-Coordinate')
        ax1.xaxis.grid(True)
        ax1.set_title("%s\n%s" % (
            "NaN Thresholds:  Floor = %s,  Ceiling = %s%s" % (
                nan_threshold_floor if nan_threshold_floor is None
                else "%.3f" % nan_threshold_floor,
                nan_threshold_ceiling if nan_threshold_ceiling is None
                else "%.3f" % nan_threshold_ceiling,
                '' if nan_threshold_floor is None or
                nan_threshold_ceiling is None else ",  Range = %.3f" %
                (nan_threshold_ceiling - nan_threshold_floor)
            ),
            "Median Filter Size = %i,  Gaussian_Filter_Sigma = %.2f" % (
                median_filter_size, gaussian_filter_sigma
            ),
        ), fontsize=11.0)  # if np_src_xs_zs_indy_all_nnan else None
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        nan_threshold_ceiling_ = np.nan if nan_threshold_ceiling is None else (
            nan_threshold_ceiling)
        nan_threshold_floor_ = np.nan if nan_threshold_floor is None else (
            nan_threshold_floor)
        ax1_ylim_zs = np.concatenate((np_src_zs_fltr_window.flatten(),
            np.array([nan_threshold_ceiling_, nan_threshold_floor_])))
        ax1_ylim_zs_max = np.nanmax(ax1_ylim_zs)
        ax1_ylim_zs_min = np.nanmin(ax1_ylim_zs)
        ax1_ylim_zs_mid = (ax1_ylim_zs_max + ax1_ylim_zs_min) / 2.
        ax1_ylim_zs_hrg = (ax1_ylim_zs_max - ax1_ylim_zs_min) / 2.
        ax1_ylim_zs_mlt = 1.3
        ax1_set_ylim_min = ax1_ylim_zs_mid - ax1_ylim_zs_hrg * ax1_ylim_zs_mlt
        ax1_set_ylim_max = ax1_ylim_zs_mid + ax1_ylim_zs_hrg * ax1_ylim_zs_mlt
        if 1 and prt:
            print fmt0(just1)[0:] % ("nan_threshold_ceiling_",
                nan_threshold_ceiling_)
            print fmt0(just1)[1:] % ("ax1_ylim_zs_max", ax1_ylim_zs_max)
        if 1 and prt:
            print fmt0(just1)[0:] % ("nan_threshold_floor_",
                nan_threshold_floor_)
            print fmt0(just1)[1:] % ("ax1_ylim_zs_min", ax1_ylim_zs_min)
        if 1 and prt:
            print fmt0(just1)[0:] % ("ax1_ylim_zs_mid", ax1_ylim_zs_mid)
            print fmt0(just1)[1:] % ("ax1_ylim_zs_hrg", ax1_ylim_zs_hrg)
            print fmt0(just1)[1:] % ("ax1_ylim_zs_mlt", ax1_ylim_zs_mlt)
        if 1 and prt:
            print fmt0(just1)[0:] % ("ax1_set_ylim_min", ax1_set_ylim_min)
            print fmt0(just1)[1:] % ("ax1_set_ylim_max", ax1_set_ylim_max)
        None if 1 else sys.exit()
        #
        ax1.set_ylim((ax1_set_ylim_min, ax1_set_ylim_max))
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax1.axhline(
            y=nan_threshold_floor, xmin=0.02, xmax=0.98, c='y',
            ls='dashed', lw=2.
        ) if 1 and nan_threshold_floor is not None else None
        ax1.axhline(
            y=nan_threshold_ceiling, xmin=0.02, xmax=0.98, c='y',
            ls='dashed', lw=2.
        ) if 1 and nan_threshold_ceiling is not None else None
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        ax2 = plt.subplot(gs[1], sharex=ax1)  # dzdxs
        ax2.set_xlabel('Y-Coordinate')
        ax2.set_ylabel('dZ/dY')
        ax2_y_max_abs = (
            np.nanmax(np.abs(np_src_zs_fltr_diff_window[:, np_indxs_window])))
        ax2.set_title('dZdYs:  %s%s' % (
            "Max. Magnitude = %.4f" % ax2_y_max_abs,
            "" if dzdys_threshold is None else
            ",  Threshold = +- %.4f (%.1f%%)" %
            (dzdys_threshold, dzdys_threshold / ax2_y_max_abs * 100.),
        ), fontsize=11.)
        ax2.xaxis.grid(True)
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax2_y_max_abs = 0.001 if ax2_y_max_abs < 0.001 else ax2_y_max_abs
        ax2.set_ylim((
            -ax2_y_max_abs * 1.1, ax2_y_max_abs * 1.1
        )) if 1 else None
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax2_x_max_abs = np.nanmax(np.abs(np_indys_window))
        ax2.set_xlim((
            -ax2_x_max_abs * 0.05, ax2_x_max_abs * 1.05
        )) if 1 else None
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        ax2.axhline(y=-dzdys_threshold, xmin=0.02, xmax=0.98, c='y',
            ls='dashed', lw=2.) if dzdys_threshold is not None else None
        ax2.axhline(y=dzdys_threshold, xmin=0.02, xmax=0.98, c='y',
            ls='dashed', lw=2.) if dzdys_threshold is not None else None
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        for i, indx in enumerate(np_indxs_window):
            #f 1 and prt:
            #   print fmt0(just1)[0:] % ("indx", indx)
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            ax1.plot(np_indys_window, np_src_zs_window[:, indx],
                'c,-', mec='none') if 0 else None
            #ax1.plot(np_indys_window, np_src_zs_clip_window[:, indx],
            #    'c,-', mec='none')
            ax1.plot(np_indys_window, np_src_zs_fltr_window[:, indx],
                'b,-', mec='none')
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            ax2.plot(np_indys_midp_window, np_src_zs_diff_window[:, indx],
                'c,-', mec='none') if 0 else None
            #ax2.plot(
            #    np_indys_midp_window, np_src_zs_clip_diff_window[:, indx],
            #    'c,-', mec='none')
            ax2.plot(np_indys_midp_window, np_src_zs_fltr_diff_window[:, indx],
                'b,-', mec='none')
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            #f i >= 4:
            #   break
        # : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : :
        png = ngn.job_zs_csv.replace(".csv",
            "__preview_%s_meast_profiles.png" % np_src_zs_str)
        png_abspath = os.path.join(ngn.preview_absdir, png)
        if 1 and prt:
            print fmt0(just1)[0:] % ("ngn.preview_absdir", ngn.preview_absdir)
            print fmt0(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
            print fmt0(just1)[1:] % ("png", png)
        print fmt1(just1)[0:] % ("png_abspath", png_abspath)
        fig.savefig(png_abspath)
        None if 1 else plt.show()
        plt.close()
        None if 1 else sys.exit()
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def get_cnrc_laser_profiles_dataset(job_xs_csv_abspath, job_zs_csv_abspath):
    """
    Returns Pandas DataFrames containing laser measurement profiles dataset.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'get_cnrc_laser_profiles_dataset'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 0 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("job_zs_csv_abspath", job_zs_csv_abspath)

    pd_src = pd.read_csv(job_zs_csv_abspath, header=None, index_col=False)
    if 1 and prt:
        print fmt0(just1)[0:] % ("(init) pd_src.shape", pd_src.shape)
        idx_max = 9
        print fmt1(just1)[1:] % ("(init) pd_src.iloc[:%i, :%i]" %
            (idx_max, idx_max), pd_src.iloc[:idx_max, :idx_max])
        print fmt1(just1)[1:] % ("(init) pd_src.iloc[%i:, :%i]" %
            (-idx_max, idx_max), pd_src.iloc[-idx_max:, :idx_max])
        print fmt1(just1)[1:] % ("(init) pd_src.iloc[%i:, %i:]" %
            (-idx_max, -idx_max), pd_src.iloc[-idx_max:, -idx_max:])

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # get the "xs" and "zs" in separate Pandas DataFrames

    pd_src_xs = pd_src.iloc[1::2, 1:].copy()
    pd_src_xs.reset_index(drop=True, inplace=True)
    if 1 and prt:
        print fmt0(just1)[0:] % ("pd_src_xs.shape", pd_src_xs.shape)
        idx_max = 5
        print fmt1(just1)[1:] % ("pd_src_xs.iloc[:%i, :%i]" %
            (idx_max, idx_max), pd_src_xs.iloc[:idx_max, :idx_max])
        #rint fmt1(just1)[1:] % ("pd_src_xs.iloc[%i:, :%i]" %
        #   (-idx_max, idx_max), pd_src_xs.iloc[-idx_max:, :idx_max])
        print fmt1(just1)[1:] % ("pd_src_xs.iloc[%i:, %i:]" %
            (-idx_max, -idx_max), pd_src_xs.iloc[-idx_max:, -idx_max:])

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    pd_src_zs = pd_src.iloc[2::2, 1:].copy()
    pd_src_zs.reset_index(drop=True, inplace=True)
    if 1 and prt:
        print fmt0(just1)[0:] % ("pd_src_zs.shape", pd_src_zs.shape)
        idx_max = 5
        print fmt1(just1)[1:] % ("pd_src_zs.iloc[:%i, :%i]" %
            (idx_max, idx_max), pd_src_zs.iloc[:idx_max, :idx_max])
        #rint fmt1(just1)[1:] % ("pd_src_zs.iloc[%i:, :%i]" %
        #   (-idx_max, idx_max), pd_src_zs.iloc[-idx_max:, :idx_max])
        print fmt1(just1)[1:] % ("pd_src_zs.iloc[%i:, %i:]" %
            (-idx_max, -idx_max), pd_src_zs.iloc[-idx_max:, -idx_max:])

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return pd_src_xs, pd_src_zs


def get_laser_profiles_dataset(job_xs_csv_abspath, job_zs_csv_abspath):
    """
    Returns Pandas DataFrames containing laser measurement profiles dataset.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'get_laser_profiles_dataset'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 0 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("job_zs_csv_abspath", job_zs_csv_abspath)

    pd_src = pd.read_csv(job_zs_csv_abspath, header=0)
    pd_src = pd_src.iloc[:, 1:]
    if 0 and prt:
        hd, tl = (4, 4) if 1 else (20, 20)
        print fmt1(just1)[0:] % ("pd_src.shape", pd_src.shape)
        print fmt1(just1)[1:] % ("pd_src.head(%i)" % hd, pd_src.head(hd))
        print fmt1(just1)[1:] % ("pd_src.tail(%i)" % tl, pd_src.tail(tl))

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # get the "xs" and "zs" in separate Pandas DataFrames

    pd_src_zs = pd_src.copy()
    if 0 and prt:
        hd, tl = (4, 4) if 1 else (20, 20)
        print fmt1(just1)[0:] % ("pd_src_zs.shape", pd_src_zs.shape)
        print fmt1(just1)[1:] % ("pd_src_zs.head(%i)" % hd, pd_src_zs.head(hd))
        print fmt1(just1)[1:] % ("pd_src_zs.tail(%i)" % tl, pd_src_zs.tail(tl))

    None if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    np_src_xs = np.zeros(pd_src_zs.shape)
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs.shape", np_src_xs.shape)
        print fmt1(just1)[1:] % ("(init) np_src_xs", np_src_xs)

    np_src_xs_row = pd_src_zs.columns.values.astype(np.float)
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_row.shape", np_src_xs_row.shape)
        print fmt1(just1)[1:] % ("np_src_xs_row", np_src_xs_row)

    if 0:  # check for duplicate x values
        np_src_xs_row_uniqs = np.unique(np_src_xs_row)
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_src_xs_row_uniqs.shape",
                np_src_xs_row_uniqs.shape)
            print fmt1(just1)[1:] % ("np_src_xs_row_uniqs",
                np_src_xs_row_uniqs)
            print fmt1(just1)[1:] % (
                "np_src_xs_row.shape[0] == np_src_xs_row_uniqs.shape[0]",
                np_src_xs_row.shape[0] == np_src_xs_row_uniqs.shape[0])
        None if 0 else sys.exit()

    if 0:  # calculate the mean x differences
        np_src_xs_row_diff = np.diff(np_src_xs_row)
        np_src_xs_row_diff_mean = np.mean(np.diff(np_src_xs_row))
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_src_xs_row_diff.shape",
                np_src_xs_row_diff.shape)
            print fmt1(just1)[1:] % ("np_src_xs_row_diff",
                np_src_xs_row_diff)
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_src_xs_row_diff_mean",
                np_src_xs_row_diff_mean)
        None if 0 else sys.exit()

    np_src_xs = np_src_xs + np_src_xs_row
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs.shape", np_src_xs.shape)
        print fmt1(just1)[1:] % ("(updt) np_src_xs", np_src_xs)

    np_src_xs_uniqs = np.unique(np_src_xs)
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs_uniqs.shape",
            np_src_xs_uniqs.shape)
        print fmt1(just1)[1:] % ("np_src_xs_uniqs", np_src_xs_uniqs)

    pd_src_xs = pd_src.copy()
    pd_src_xs.iloc[:, :] = np_src_xs
    if 1 and prt:
        hd, tl = (4, 4) if 1 else (20, 20)
        print fmt1(just1)[0:] % ("pd_src_xs.shape", pd_src_xs.shape)
        print fmt1(just1)[1:] % ("pd_src_xs.head(%i)" % hd, pd_src_xs.head(hd))
        print fmt1(just1)[1:] % ("pd_src_xs.tail(%i)" % tl, pd_src_xs.tail(tl))

    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return pd_src_xs, pd_src_zs


def get_dataset(job_xs_csv_abspath, job_zs_csv_abspath):
    """
    Returns Pandas DataFrames containing laser measurement profiles dataset.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'get_dataset'

    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    if 'cnrc_r_and_d' == ngn.sensor.lower():
        if 1 and prt:
            print fmt0(just2)[0:] % (
                "using ... def get_\cnrc_laser_profiles_dataset",
                'to read ... "%s"' % ngn.job_zs_csv_abspath)

        # this gets cnrc laser profile measurements in their "native"
        # (given) file format
        pd_src_xs, pd_src_zs = get_cnrc_laser_profiles_dataset(
            ngn.job_xs_csv_abspath, ngn.job_zs_csv_abspath)

    #lif 'microepsilon' in sensor.lower():
    #   if 1 and prt:
    #       print fmt0(just2)[0:] % (
    #           "using ... def (*** *** *** TBD *** *** ***)",
    #           'to read ... "%s"' % ngn.job_zs_csv_abspath)
    #   None if 0 else sys.exit()

    elif 'cnrc' == ngn.sensor.lower() or 'keyence' == ngn.sensor.lower():
        if 1 and prt:
            print fmt0(just2)[0:] % (
                "using ... def get_laser_profiles_dataset",
                'to read ... "%s"' % ngn.job_zs_csv_abspath)

        pd_src_xs, pd_src_zs = get_laser_profiles_dataset(
            ngn.job_xs_csv_abspath, ngn.job_zs_csv_abspath)

    else:
        assert 0, "\n    sensor '%s' is unknown" % ngn.sensor

    if 1 and prt:
        print fmt0(just2)[0:] % ("pd_src_xs.shape", pd_src_xs.shape)
        print fmt0(just2)[1:] % ("pd_src_zs.shape", pd_src_zs.shape)

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    np_src_xs = pd_src_xs.values
    np_src_zs = pd_src_zs.values

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return np_src_xs, np_src_zs


###def autothreshold_nan_values(np_src_xs, np_src_zs):
### """
### Returns 'NaN' value floor and ceiling threshold values.
### """
### just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
### prt = False if 1 else True
### prt_ = prt
### mult_str = '--- '
### def_str = 'autothreshold_nan_values'
### if prt_:
###     print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
### #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

### prt = True if 1 and prt_ else False

### #== === === === === === === === === === === === === === === === === === ===

### #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
### # quick stats

### if 1 and prt:
###     print fmt0(just2)[0:] % ("np_src_xs.shape", np_src_xs.shape)
###     print fmt0(just2)[1:] % ("np_src_xs.iloc: [0, 0], [0, -1]",
###         np.array([np_src_xs[0, 0], np_src_xs[0, -1]]))

### np_src_zs_nanmax = np.nanmax(np_src_zs)
### np_src_zs_nanmean = np.nanmean(np_src_zs)
### np_src_zs_nanmin = np.nanmin(np_src_zs)
### if 1:
###     print fmt0(just2)[0:] % (
###         "[np_src_zs_nanmin, np_src_zs_nanmean, np_src_zs_nanmax]",
###         np.array([np_src_zs_nanmin, np_src_zs_nanmean, np_src_zs_nanmax]))

### np_src_zs_nanmax_nanmean = np_src_zs_nanmax - np_src_zs_nanmean
### np_src_zs_nanmin_nanmean = np_src_zs_nanmin - np_src_zs_nanmean
### if 1 and prt:
###     print fmt0(just2)[0:] % (
###         "[np_src_zs_nanmin_nanmean, np_src_zs_nanmax_nanmean]",
###         np.array([np_src_zs_nanmin_nanmean, np_src_zs_nanmax_nanmean]))

### np_src_zs_1nanstd = np.nanstd(np_src_zs)
### np_src_zs_2nanstd = 2. * np_src_zs_1nanstd
### np_src_zs_3nanstd = 3. * np_src_zs_1nanstd
### np_src_zs_5nanstd = 5. * np_src_zs_1nanstd
### np_src_zs_nanrng = np.min(np.abs(
###     [np_src_zs_nanmin_nanmean, np_src_zs_nanmax_nanmean]))
### if 1:
###     print fmt0(just2)[0:] % (
###         "[np_src_zs_1nanstd]", np.array([np_src_zs_1nanstd]))
###     print fmt0(just2)[1:] % (
###         "[np_src_zs_2nanstd]", np.array([np_src_zs_2nanstd]))
###     print fmt0(just2)[1:] % (
###         "[np_src_zs_3nanstd]", np.array([np_src_zs_3nanstd]))
###     print fmt0(just2)[1:] % (
###         "[np_src_zs_5nanstd]", np.array([np_src_zs_5nanstd]))
###     print fmt0(just2)[1:] % (
###         "[np_src_zs_nanrng]", np.array([np_src_zs_nanrng]))

### zs_rng = np_src_zs_nanrng
### zs_rng = zs_rng if 0 else np_src_zs_5nanstd
### zs_rng = zs_rng if 0 else np_src_zs_3nanstd
### zs_rng = zs_rng if 0 else np_src_zs_2nanstd
### zs_rng = zs_rng if 0 else np_src_zs_1nanstd

### #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

### np_src_zs_nanmean_m1std = np_src_zs_nanmean - np_src_zs_1nanstd
### np_src_zs_nanmean_p1std = np_src_zs_nanmean + np_src_zs_1nanstd
### #
### np_src_zs_nanmean_m2std = np_src_zs_nanmean - np_src_zs_2nanstd
### np_src_zs_nanmean_p2std = np_src_zs_nanmean + np_src_zs_2nanstd
### #
### np_src_zs_nanmean_m3std = np_src_zs_nanmean - np_src_zs_3nanstd
### np_src_zs_nanmean_p3std = np_src_zs_nanmean + np_src_zs_3nanstd
### #
### np_src_zs_nanmean_m5std = np_src_zs_nanmean - np_src_zs_5nanstd
### np_src_zs_nanmean_p5std = np_src_zs_nanmean + np_src_zs_5nanstd
### #
### np_src_zs_nanmean_mnanrng = np_src_zs_nanmean - np_src_zs_nanrng
### np_src_zs_nanmean_pnanrng = np_src_zs_nanmean + np_src_zs_nanrng
### if 1:
###     print fmt0(just1)[0:] % (
###         "[np_src_zs_nanmean_m1std, np_src_zs_nanmean_p1std]",
###         np.array([np_src_zs_nanmean_m1std, np_src_zs_nanmean_p1std]))
###     print fmt0(just1)[1:] % (
###         "[np_src_zs_nanmean_m2std, np_src_zs_nanmean_p2std]",
###         np.array([np_src_zs_nanmean_m2std, np_src_zs_nanmean_p2std]))
###     print fmt0(just1)[1:] % (
###         "[np_src_zs_nanmean_m3std, np_src_zs_nanmean_p3std]",
###         np.array([np_src_zs_nanmean_m3std, np_src_zs_nanmean_p3std]))
###     print fmt0(just1)[1:] % (
###         "[np_src_zs_nanmean_m5std, np_src_zs_nanmean_p5std]",
###         np.array([np_src_zs_nanmean_m5std, np_src_zs_nanmean_p5std]))
###     print fmt0(just1)[1:] % (
###         "[np_src_zs_nanmean_mnanrng, np_src_zs_nanmean__pnanrng]",
###         np.array([np_src_zs_nanmean_mnanrng,
###             np_src_zs_nanmean_pnanrng]))

### nan_value_floor_threshold = np_src_zs_nanmean - zs_rng
### nan_value_ceiling_threshold = np_src_zs_nanmean + zs_rng
### if 1 and prt:
###     print fmt0(just1)[0:] % (
###         "[nan_value_floor_threshold, nan_value_ceiling_threshold]",
###         np.array([nan_value_floor_threshold, nan_value_ceiling_threshold]))

### #== === === === === === === === === === === === === === === === === === ===

### #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
### if prt_:
###     print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
### None if 1 else sys.exit()
### return nan_value_floor_threshold, nan_value_ceiling_threshold


def fix_nan_values(np_src_xs, np_src_zs, nan_value_floor_threshold,
nan_value_ceiling_threshold, sensor):
    """
    Returns laser profiles & their first diff's with "true" (numpy) NaN values.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'fix_nan_values'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===
    # "fix" NaN values

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # before assigning NaN values ...
    if 0:
        prt = True if 1 and prt_ else False

        np_src_zs_nanmax = np.nanmax(np_src_zs)
        np_src_zs_nanmean = np.nanmean(np_src_zs)
        np_src_zs_nanmin = np.nanmin(np_src_zs)
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_src_zs_nanmax", np_src_zs_nanmax)
            print fmt0(just1)[1:] % ("np_src_zs_nanmean", np_src_zs_nanmean)
            print fmt0(just1)[1:] % ("np_src_zs_nanmin", np_src_zs_nanmin)

        np_src_zs_uniqs = np.unique(np_src_zs)
        if 1 and prt:
            print fmt0(just1)[0:] % (
                "np_src_zs_uniqs.shape", np_src_zs_uniqs.shape)
            print fmt1(just1)[1:] % ("np_src_zs_uniqs", np_src_zs_uniqs)

        None if 0 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if 1 and prt:
        print fmt0(just1)[0:] % ("sensor", sensor)
        print fmt0(just1)[1:] % ("nan_value_floor_threshold",
            nan_value_floor_threshold)
        print fmt0(just1)[1:] % ("nan_value_ceiling_threshold",
            nan_value_ceiling_threshold)

    np_hasnan_zs = np_src_zs.copy()

    np_hasnan_zs_mask_nans = (
        (np_hasnan_zs < nan_value_floor_threshold) |
        (np_hasnan_zs > nan_value_ceiling_threshold)
    )
    if 1 and prt:
        print fmt1(just1)[1:] % ("np.sort(np.unique(np_src_zs))",
            np.sort(np.unique(np_src_zs)))
    None if 1 else sys.exit()

    np_hasnan_zs[np_hasnan_zs_mask_nans] = np.nan
    if 0 and prt:
        print fmt1(just1)[0:] % (
            "np_hasnan_zs[np_hasnan_zs_mask_nans]",
            np_hasnan_zs[np_hasnan_zs_mask_nans])
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "np_hasnan_zs_mask_nans.shape", np_hasnan_zs_mask_nans.shape)
        print fmt0(just1)[1:] % (
            "np_hasnan_zs_mask_nans.size", np_hasnan_zs_mask_nans.size)
        print fmt0(just1)[1:] % ("np.sum(np_hasnan_zs_mask_nans)",
            np.sum(np_hasnan_zs_mask_nans))

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # quick profile plot
    if 0:
        meast_idx = 0
        xs = np_src_xs[meast_idx, :]
        zs_src = np_src_zs[meast_idx, :]
        zs_hasnan = np_hasnan_zs[meast_idx, :]
        #
        plt.suptitle('%s\nProfile %i of %i Profiles' % (
            os.path.join(ngn.job_dir, ngn.job_zs_csv), (meast_idx + 1),
            len(np_src_zs)))
        fig = plt.gcf()
        fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
        ax = plt.gca()
        #
        ax.axhline(y=nan_value_floor_threshold, xmin=0.01, xmax=0.99,
            c='y', ls='dashed', lw=2.)
        #
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Z Coordinate')
        ax.plot(xs, zs_src, ',-')
        ax.plot(xs, zs_hasnan, ',-')
        #
        #ig.savefig('fig1.png', bbox_inches='tight')
        None if 0 else plt.show()
        plt.close()
        None if 0 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return np_hasnan_zs_mask_nans, np_hasnan_zs


def imshow(image, title=None, cmap='gray', xticklabel=True, yticklabel=True,
xlabel=None, ylabel=None, colorbar=False, png=None):
    """
    Returns a matplotlib plot window with the given image.
    """
    plt.close('all')  # to avoid ttk "nasty-gram"
    #fig = plt.gcf()
    ax = plt.gca()
    #f not ticklabels:
    #   ax.axes.xaxis.set_ticklabels([])
    #   ax.axes.yaxis.set_ticklabels([])
    None if xticklabel is True else ax.axes.xaxis.set_ticklabels([])
    None if yticklabel is True else ax.axes.yaxis.set_ticklabels([])
    None if xlabel is None else ax.set_xlabel(xlabel)
    None if ylabel is None else ax.set_ylabel(ylabel)
    #
    None if title is None else plt.title(title)
    plt.imshow(image).set_cmap(cmap)
    #plt.colorbar() if colorbar is True else None
    plt.colorbar(orientation="horizontal", ) if colorbar is True else None
    plt.tight_layout() if 1 and colorbar is False else None
    if png is not None:
        png_absdir, png_filename = os.path.split(png)
        #rint fmt0(ngn.just1)[0:] % ("png", png)
        #rint fmt0(ngn.just1)[0:] % ("png_absdir", png_absdir)
        #rint fmt0(ngn.just1)[0:] % ("png_filename", png_filename)
        None if os.path.exists(png_absdir) else os.makedirs(png_absdir)
        plt.savefig(png)
    else:
        plt.show()
    plt.close()


def make_laser_profiles_image_plot(np_hasnan_zs, title=None):
    """
    Makes an image plot of all (unfiltered) laser measurement profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_laser_profiles_image_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    # "fix" the np.nan values to a chosen value for the image plot
    image = np_hasnan_zs.copy()
    image[np.isnan(image)] = np.nanmax(np_hasnan_zs)
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "np.sum(np.isnan(image))", np.sum(np.isnan(image)))
        print fmt0(just1)[1:] % (
            "np.nanmin(image)", np.nanmin(image))
        print fmt0(just1)[1:] % (
            "np.nanmax(image)", np.nanmax(image))

    img_png = ngn.job_zs_csv.replace('z', '').replace('.txt', '').replace(
        '.csv', '__laser_profiles_image.png')
    if 0 and prt:
        print fmt0(just1)[0:] % ("img_png", img_png)
    None if 1 else sys.exit()

    png_abspath = os.path.join(ngn.job_absdir, img_png)
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
        print fmt1(just1)[1:] % ("img_png", img_png)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)

    title = (
        title if title is not None else
        'Image Map of Z-coordinate Values\n%s' % ngn.job_zs_csv
    )
    cmap = 'jet'
    colorbar = True if 1 else False
    xlabel = 'Measurement Index (Y)'
    ylabel = 'Profile Index (X)'
    png = png_abspath if 1 else None
    imshow(image.T, title=title, cmap=cmap, xticklabel=True,
        yticklabel=True, xlabel=xlabel, ylabel=ylabel,
        colorbar=colorbar, png=png)

    if ngn.write_to_results_dir:
        png_abspath_results = os.path.join(ngn.results_absdir, img_png)
        if 0 and prt:
            print fmt1(just1)[0:] % ("ngn.results_absdir", ngn.results_absdir)
            print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
            print fmt1(just1)[1:] % ("img_png", img_png)
        print fmt1(just1)[0:] % ("png_abspath_results", png_abspath_results)
        shutil.copy(png_abspath, png_abspath_results)

    #f ngn.add_pyapp_version:
    #   img_png_pyapp_ver = img_png.replace(
    #       '.png', '_py%.3i.png' % ngn.version_py)
    #   if 0 and prt:
    #       print fmt0(just1)[0:] % ("img_png", img_png)
    #       print fmt0(just1)[1:] % ("img_png_pyapp_ver", img_png_pyapp_ver)
    #   png_abspath_pyapp_ver = os.path.join(ngn.job_absdir, img_png_pyapp_ver)
    #   shutil.copy(png_abspath, png_abspath_pyapp_ver)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_laser_profiles_image_plot_with_rois(np_hasnan_zs, title=None):
    """
    Makes an image plot of all (unfiltered) laser measurement profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_laser_profiles_image_plot_with_rois'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    # "fix" the np.nan values to a chosen value for the image plot
    image = np_hasnan_zs.copy()
    image_nanmin = np.nanmin(image)
    image_nanmax = np.nanmax(image)
    image[np.isnan(image)] = image_nanmax
    if 1 and prt:
        print fmt0(just1)[0:] % ("np.sum(np.isnan(image))",
            np.sum(np.isnan(image)))
        print fmt0(just1)[1:] % ("image_nanmin", image_nanmin)
        print fmt0(just1)[1:] % ("image_nanmax", image_nanmax)

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # "draw" the regions of interest boundaries (tow edge xref indx)

    line_width = 6
    line_width = line_width if line_width > 1 else 2
    line_width_half = line_width / 2  # integer division
    if 1 and prt:
        print fmt0(just1)[0:] % ('image.shape', image.shape)
    if 1 and prt:
        print fmt0(just1)[0:] % ('line_width', line_width)
        print fmt0(just1)[1:] % ('line_width_half', line_width_half)

    for i, tow_edge_xref_idx in enumerate(ngn.tow_edge_xref_idxs):
        idx0 = tow_edge_xref_idx - line_width_half
        idx0 = idx0 if idx0 >= 0 else 0
        idx1 = tow_edge_xref_idx + line_width_half
        idx1 = idx1 if idx1 < image.shape[1] else image.shape[1] - 1
        image[:, idx0:idx1 + 1] = image_nanmax
        #f 1 and prt:
        #   if i == 0:
        #       print
        #   print fmt0(just1)[1:] % (
        #       '[i, tow_edge_xref_idx, line_width_half idx0, idx1]',
        #       ["%.2i" % i, "%.4i" % tow_edge_xref_idx,
        #           line_width_half, "%.4i" % idx0, "%.4i" % idx1])

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    img_png = ngn.job_zs_csv.replace('z', '').replace('.txt', '').replace(
        '.csv', '__laser_profiles_image_with_tow_rois.png')
    if 0 and prt:
        print fmt0(just1)[0:] % ("img_png", img_png)
    None if 1 else sys.exit()

    png_abspath = os.path.join(ngn.job_absdir, img_png)
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
        print fmt1(just1)[1:] % ("img_png", img_png)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)

    title = (
        title if title is not None else
        'Image Map of Z-coordinate Values with Tow ROIs\n%s' % ngn.job_zs_csv
    )
    cmap = 'jet'
    colorbar = True if 1 else False
    xlabel = 'Measurement Index (Y)'
    ylabel = 'Profile Index (X)'
    png = png_abspath if 1 else None
    imshow(image.T, title=title, cmap=cmap, xticklabel=True,
        yticklabel=True, xlabel=xlabel, ylabel=ylabel,
        colorbar=colorbar, png=png)

    if ngn.write_to_results_dir:
        png_abspath_results = os.path.join(ngn.results_absdir, img_png)
        if 0 and prt:
            print fmt1(just1)[0:] % ("ngn.results_absdir", ngn.results_absdir)
            print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
            print fmt1(just1)[1:] % ("img_png", img_png)
        print fmt1(just1)[0:] % ("png_abspath_results", png_abspath_results)
        shutil.copy(png_abspath, png_abspath_results)

    #f ngn.add_pyapp_version:
    #   img_png_pyapp_ver = img_png.replace(
    #       '.png', '_py%.3i.png' % ngn.version_py)
    #   if 0 and prt:
    #       print fmt0(just1)[0:] % ("img_png", img_png)
    #       print fmt0(just1)[1:] % ("img_png_pyapp_ver", img_png_pyapp_ver)
    #   png_abspath_pyapp_ver = os.path.join(ngn.job_absdir, img_png_pyapp_ver)
    #   shutil.copy(png_abspath, png_abspath_pyapp_ver)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_laser_profiles_overlay_plot(np_src_xs, np_hasnan_zs):
    """
    Makes an overlay plot of all (unfiltered) laser measurement column
    profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_laser_profiles_overlay_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 0 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    profile_measts, profile_points = np_hasnan_zs.shape
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_hasnan_zs.shape", np_hasnan_zs.shape)
        print fmt0(just1)[1:] % ("profile_measts", profile_measts)
        print fmt0(just1)[1:] % ("profile_points", profile_points)

    #fig = plt.gcf()
    #fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
    ax = plt.gca()
    ax.set_title("Laser Profiles Overlay (%i Meas'ts, %i Points Each)\n%s" % (
        profile_measts, profile_points, ngn.job_zs_csv))
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Z-coordinate')
    #
#   ax.set_xlim((-ngn.np_src_xs_mgn_half, ngn.np_src_xs_mgn_half))
#   ax.set_ylim(ngn.np_hasnan_zs_min_mean_max_mgn) if 0 else None
#   ax.set_ylim(ngn.np_hasnan_zs_min_mid_max_mgn) if 1 else None
    #
#   ax.axhline(y=ngn.np_hasnan_zs_max, xmin=0.01, xmax=0.99,
#       c='y', ls='dashed', lw=2.)
#   ax.text(ngn.np_src_xs_mid, ngn.np_hasnan_zs_max,
#       "Max. Value: %.3f" % ngn.np_hasnan_zs_max, ha='center', va='bottom')
    #
#   ax.axhline(y=ngn.np_hasnan_zs_min, xmin=0.01, xmax=0.99,
#       c='y', ls='dashed', lw=2.)
#   ax.text(ngn.np_src_xs_mid, ngn.np_hasnan_zs_min,
#       "Min. Value: %.3f" % ngn.np_hasnan_zs_min, ha='center', va='top')
    #
    for indy in xrange(profile_measts):
        xs = np_src_xs[indy, :]
        zs = np_hasnan_zs[indy, :]
        ax.plot(xs, zs, ',-')
    #
    png_abspath = os.path.join(ngn.results_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '')
        .replace('.csv', '__001_laser_profiles_overlay.png'))
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.results_absdir", ngn.results_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    #
    png = png_abspath if 1 else None
    if png is not None:
        plt.savefig(png)
    else:
        plt.show()
    plt.close()
    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_laser_measts_overlay_plot(np_hasnan_zs):
    """
    Makes an overlay plot of all (unfiltered) laser profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_laser_measts_overlay_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    profile_measts, profile_points = np_hasnan_zs.shape
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_hasnan_zs.shape", np_hasnan_zs.shape)
        print fmt0(just1)[1:] % ("profile_measts", profile_measts)
        print fmt0(just1)[1:] % ("profile_points", profile_points)

    #fig = plt.gcf()
    #fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
    ax = plt.gca()
    ax.set_title(
        "Laser Measurements Overlay (%i Meas'ts, %i Points Each)\n%s" % (
            profile_measts, profile_points, ngn.job_zs_csv))
    ax.set_xlabel('Y-coordinate (Profile Measurement Index)')
    ax.set_ylabel('Z-coordinate')
    #
#   ax.set_xlim(ngn.np_src_ys_min_max_mgn)
#   ax.set_ylim(ngn.np_hasnan_zs_min_mean_max_mgn) if 0 else None
#   ax.set_ylim(ngn.np_hasnan_zs_min_mid_max_mgn) if 1 else None
    #
#   ax.axhline(y=ngn.np_hasnan_zs_max, xmin=0.01, xmax=0.99,
#       c='y', ls='dashed', lw=2.)
#   ax.text(ngn.np_src_ys_mid, ngn.np_hasnan_zs_max,
#       "Max. Value: %.3f" % ngn.np_hasnan_zs_max, ha='center', va='bottom')
    #
#   ax.axhline(y=ngn.np_hasnan_zs_min, xmin=0.01, xmax=0.99,
#       c='y', ls='dashed', lw=2.)
#   ax.text(ngn.np_src_ys_mid, ngn.np_hasnan_zs_min,
#       "Min. Value: %.3f" % ngn.np_hasnan_zs_min, ha='center', va='top')
    #
    ys = np.arange(profile_measts)
    if 1 and prt:
        print fmt1(just1)[0:] % ("ys", ys)
    None if 1 else sys.exit()

    for indx in xrange(profile_points):
        #ys = np_src_ys[:, indx]
        zs = np_hasnan_zs[:, indx]
        ax.plot(ys, zs, ',-')
    #
    png_abspath = os.path.join(ngn.results_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '')
        .replace('.csv', '__002_laser_measts_overlay.png'))
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.results_absdir", ngn.results_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    #
    png = png_abspath if 1 else None
    if png is not None:
        plt.savefig(png)
    else:
        plt.show()
    plt.close()
    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_laser_meast_zs_histogram_plot(np_hasnan_zs):
    """
    Makes a histogram plot of all (unfiltered) z-coordinate dataset values.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_laser_meast_zs_histogram_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 0 and prt_ else False

    profile_values = np_hasnan_zs.size
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_hasnan_zs.size", np_hasnan_zs.size)

    np_zs = np_hasnan_zs.copy().flatten()
    np_zs_mask_isnan = np.isnan(np_zs)
    np_zs_nnan = np_zs[~ np_zs_mask_isnan]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_zs.size", np_zs.size)
        print fmt0(just1)[1:] % (
            "np.sum(np_zs_mask_isnan)", np.sum(np_zs_mask_isnan))
        print fmt0(just1)[1:] % ("np_zs_nnan.size", np_zs_nnan.size)
        None if 1 else sys.exit()

    decimals = 2
    np_zs_nnan_rnd = np.round(np_zs_nnan, decimals=decimals)
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "np_zs_nnan_rnd.shape", np_zs_nnan_rnd.shape)
        print fmt0(just1)[1:] % ("np_zs_nnan_rnd.size", np_zs_nnan_rnd.size)
        #rint fmt1(just1)[1:] % ("np_zs_nnan_rnd", np_zs_nnan_rnd)
        #rint fmt1(just1)[1:] % (
        #   "np.sort(np_zs_nnan_rnd)", np.sort(np_zs_nnan_rnd))
        None if 1 else sys.exit()

    bins, bincounts = np.unique(np_zs_nnan_rnd, return_counts=True)
    if 1 and prt:
        print fmt0(just1)[0:] % ("bins.size", bins.size)
        #rint fmt1(just1)[1:] % ("bins", bins)
        None if 1 else sys.exit()
    if 0 and prt:
        print fmt0(just1)[1:] % ("bincounts.size", bincounts.size)
        print fmt1(just1)[1:] % ("bincounts", bincounts)
        print fmt1(just1)[1:] % ("np.sort(bincounts)", np.sort(bincounts))
        None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    width = 0.008
    color = None
    ax_get_ylim_rng_mult1 = 0.95
    ax_get_ylim_rng_mult2 = 0.90
    axvline_ymin = 0.01
    axvline_ymax = 0.99

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    ax = plt.gca()
    ax.set_title(
        "Histogram of Measurement Z-Coordinates (%i Values)\n%s" % (
            profile_values, ngn.job_zs_csv))
    ax.set_xlabel('Measurement Z-coordinate Value')
    ax.set_ylabel('Counts By Measurement Z-coordinate Value')
    #
    #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    #
    ax.bar(bins, bincounts, width=width, color=color, alpha=0.8)
    #
    #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

#   ax.set_xlim(ngn.np_hasnan_zs_min_mid_max_mgn)
    #
    ax_get_ylim = ax.get_ylim()
    ax_get_ylim_rng = ax_get_ylim[1] - ax_get_ylim[0]
    ax_text_y1 = ax_get_ylim[0] + ax_get_ylim_rng_mult1 * ax_get_ylim_rng
    ax_text_y2 = ax_get_ylim[0] + ax_get_ylim_rng_mult2 * ax_get_ylim_rng
    #
#   ax.axvline(x=ngn.np_hasnan_zs_min, ymin=axvline_ymin, ymax=axvline_ymax,
#       c='y', ls='dashed')
#   ax.text(ngn.np_hasnan_zs_min, ax_text_y1,
#       "Min.: %.3f" % ngn.np_hasnan_zs_min, ha='left', va='center')
    #
#   ax.axvline(x=ngn.np_hasnan_zs_mid, ymin=axvline_ymin, ymax=axvline_ymax,
#       c='y', ls='dashed')
#   ax.text(ngn.np_hasnan_zs_mid, ax_text_y1,
#       "Mid.: %.3f" % ngn.np_hasnan_zs_mid, ha='center', va='center')
    #
#   ax.axvline(x=ngn.np_hasnan_zs_max, ymin=axvline_ymin, ymax=axvline_ymax,
#       c='y', ls='dashed')
#   ax.text(ngn.np_hasnan_zs_max, ax_text_y1,
#       "Max.: %.3f" % ngn.np_hasnan_zs_max, ha='right', va='center')
    ##
#   ax.axvline(x=ngn.np_hasnan_zs_mean, ymin=axvline_ymin, ymax=axvline_ymax,
#       c='m', ls='solid')
#   ax.text(ngn.np_hasnan_zs_mean, ax_text_y2,
#       "Mean: %.3f" % ngn.np_hasnan_zs_mean, ha='center', va='center')

    ### (below) keep for reference --------------------------------------------
    #np_zs_nnan_max = np.max(np_zs_nnan)
    #np_zs_nnan_min = np.min(np_zs_nnan)
    #label = 'max.: %.3f    mean: %.3f    min.: %.3f    range.: %.3f' % (
    #    np_zs_nnan_max, np.mean(np_zs_nnan), np_zs_nnan_min,
    #    np_zs_nnan_max - np_zs_nnan_min)
    #ax.text(0.02, 0.95, label, transform=ax.transAxes,
    #    fontsize=12, fontweight='normal' if 1 else 'bold', ha='left')
    ### (above) keep for reference --------------------------------------------

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    plt.tight_layout() if 0 else None
    #
    png_abspath = os.path.join(ngn.results_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '')
        .replace('.csv', '__003_histogram_measts_z-coord.png'))
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.results_absdir", ngn.results_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    png = png_abspath if 1 else None
    if png is not None:
        plt.savefig(png)
    else:
        plt.show()
    plt.close()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_laser_measts_overlay_and_zs_histogram_plot(np_hasnan_zs):
    """
    Makes an overlay plot of all (unfiltered) laser profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_laser_measts_overlay_and_zs_histogram_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    profile_measts, profile_points = np_hasnan_zs.shape
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_hasnan_zs.shape", np_hasnan_zs.shape)
        print fmt0(just1)[1:] % ("profile_measts", profile_measts)
        print fmt0(just1)[1:] % ("profile_points", profile_points)

    fig = plt.gcf()
    fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
    fig.suptitle(
        "Laser Measurements Overlay (%i Meas'ts, %i Points Each)\n%s" % (
            profile_measts, profile_points, ngn.job_zs_csv))
    gridspec = [1, 2]
    gs = mpl.gridspec.GridSpec(*gridspec)
    #
    labelsize = 9.0
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1 = plt.subplot(gs[0])  # xs
    ax1.set_xlabel("Y-coordinate Value (Meas't Index)")
    ax1.set_ylabel('Z-coordinate Value')
    ax1.xaxis.set_tick_params(labelsize=labelsize)
    ax1.yaxis.set_tick_params(labelsize=labelsize)
    #
    ys = np.arange(profile_measts)
    if 1 and prt:
        print fmt1(just1)[0:] % ("ys", ys)
    None if 1 else sys.exit()

    ax1.set_xlim((ys[0] - 0.5, ys[-1] + 0.5))

    for indx in xrange(profile_points):
        zs = np_hasnan_zs[:, indx]
        ax1.plot(ys, zs, ',-')
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax2 = plt.subplot(gs[1], sharey=ax1)
    #x2.set_ylabel('Z-coordinate Value')
    ax2.set_xlabel('Z-coordinate Value Counts')
    ax2.xaxis.set_tick_params(labelsize=labelsize)
    ax2.yaxis.set_tick_params(labelsize=labelsize)
    #
    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
    profile_values = np_hasnan_zs.size
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_hasnan_zs.size", np_hasnan_zs.size)

    np_zs = np_hasnan_zs.copy().flatten()
    np_zs_mask_isnan = np.isnan(np_zs)
    np_zs_nnan = np_zs[~ np_zs_mask_isnan]
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_zs.size", np_zs.size)
        print fmt0(just1)[1:] % (
            "np.sum(np_zs_mask_isnan)", np.sum(np_zs_mask_isnan))
        print fmt0(just1)[1:] % ("np_zs_nnan.size", np_zs_nnan.size)
        None if 1 else sys.exit()

    decimals = 2
    np_zs_nnan_rnd = np.round(np_zs_nnan, decimals=decimals)
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "np_zs_nnan_rnd.shape", np_zs_nnan_rnd.shape)
        print fmt0(just1)[1:] % ("np_zs_nnan_rnd.size", np_zs_nnan_rnd.size)
        #rint fmt1(just1)[1:] % ("np_zs_nnan_rnd", np_zs_nnan_rnd)
        #rint fmt1(just1)[1:] % (
        #   "np.sort(np_zs_nnan_rnd)", np.sort(np_zs_nnan_rnd))
        None if 1 else sys.exit()

    bins, bincounts = np.unique(np_zs_nnan_rnd, return_counts=True)
    if 1 and prt:
        print fmt0(just1)[0:] % ("bins.size", bins.size)
        #rint fmt1(just1)[1:] % ("bins", bins)
        None if 1 else sys.exit()
    if 0 and prt:
        print fmt0(just1)[1:] % ("bincounts.size", bincounts.size)
        print fmt1(just1)[1:] % ("bincounts", bincounts)
        print fmt1(just1)[1:] % ("np.sort(bincounts)", np.sort(bincounts))
        None if 1 else sys.exit()
    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
    ax2.barh(bins, bincounts, height=0.008, align='center', color='None',
        alpha=0.8)
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    #
    png_abspath = os.path.join(ngn.results_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '')
        .replace('.csv', '__002_ovelay_and_histogram_measts_z-coord.png'))
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.results_absdir", ngn.results_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    png = png_abspath if 1 else None
    if png is not None:
        plt.savefig(png)
    else:
        plt.show()
    plt.close()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


###def make_laser_profiles_image_zs_not_isnan_plot(np_hasnan_zs):
### """
### Makes an image plot of all (unfiltered) "not isnan" z-coordinate values.
### """
### just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
### prt = False if 1 else True
### prt_ = prt
### mult_str = '--- '
### def_str = 'make_laser_profiles_image_zs_not_isnan_plot'
### if prt_:
###     print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
### #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

### prt = True if 0 and prt_ else False

### #== === === === === === === === === === === === === === === === === === ===

### profile_values = np_hasnan_zs.size
### if 1 and prt:
###     print fmt0(just1)[0:] % ("np_hasnan_zs.size", np_hasnan_zs.size)

### #== === === === === === === === === === === === === === === === === === ===

### np_hasnan_zs_not_isnan = ~ np.isnan(np_hasnan_zs)
### profile_not_isnans = np.sum(np_hasnan_zs_not_isnan)
### profile_isnans = np.sum(~ np_hasnan_zs_not_isnan)
### if 1 and prt:
###     print fmt0(just1)[0:] % (
###         "np_hasnan_zs_not_isnan.size", np_hasnan_zs_not_isnan.size)
###     print fmt0(just1)[1:] % (
###         "profile_not_isnans", profile_not_isnans)
###     print fmt0(just1)[1:] % (
###         "profile_isnans", profile_isnans)
### None if 1 else sys.exit()

### image = np_hasnan_zs_not_isnan
### #
### png_abspath = os.path.join(ngn.results_absdir,
###     ngn.job_zs_csv.replace('z', '').replace('.txt', '')
###     .replace('.csv', '__004_meast_zs_not_isnan_img.png'))
### if 0 and prt:
###     print fmt1(just1)[0:] % (
###         "ngn.results_absdir", ngn.results_absdir)
###     print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
### print fmt1(just1)[0:] % ("png_abspath", png_abspath)

### None if 1 else sys.exit()

### title = 'Not Is "NaN" Values Image (%i Values, %i Is "NaN" Values)\n%s' % (
###     profile_values, profile_isnans, ngn.job_zs_csv)
### cmap = 'gray'
### colorbar = True if 0 else False
### xlabel = 'Profile Index'
### ylabel = 'Measurement Index'
### png = png_abspath if 1 else None
### imshow(image, title=title, cmap=cmap, xticklabel=True,
###     yticklabel=True, xlabel=xlabel, ylabel=ylabel,
###     colorbar=colorbar, png=png)

### #== === === === === === === === === === === === === === === === === === ===

### #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
### if prt_:
###     print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
### None if 1 else sys.exit()


###def make_laser_profiles_image_cols_with_zs_isnan_plot(np_hasnan_zs):
### """
### Makes an image plot of all (unfiltered) laser measurement profile columns
### with a z-coordinate "isnan" value or not.
### """
### just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
### prt = False if 1 else True
### prt_ = prt
### mult_str = '--- '
### def_str = 'make_laser_profiles_image_cols_with_zs_isnan_plot'
### if prt_:
###     print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
### #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

### prt = True if 0 and prt_ else False

### #== === === === === === === === === === === === === === === === === === ===

### profile_values = np_hasnan_zs.size
### if 1 and prt:
###     print fmt0(just1)[0:] % ("np_hasnan_zs.shape", np_hasnan_zs.shape)
###     print fmt0(just1)[1:] % ("np_hasnan_zs.size", np_hasnan_zs.size)

### #== === === === === === === === === === === === === === === === === === ===

### np_hasnan_zs_isnan = np.isnan(np_hasnan_zs)
### np_hasnan_zs_isnan_sum_axis0 = np.sum(np_hasnan_zs_isnan, axis=0)
### profile_cols = np_hasnan_zs_isnan_sum_axis0.size
### np_hasnan_zs_isnan_sum_axis0_gt_0 = (
###     np_hasnan_zs_isnan_sum_axis0 > 0)
### profile_cols_with_isnan = np.sum(np_hasnan_zs_isnan_sum_axis0_gt_0)
### if 1 and prt:
###     print fmt0(just1)[0:] % ("np_hasnan_zs_isnan.shape",
###         np_hasnan_zs_isnan.shape)
###     print fmt0(just1)[1:] % ("np_hasnan_zs_isnan_sum_axis0.shape",
###         np_hasnan_zs_isnan_sum_axis0.shape)
###     print fmt0(just1)[1:] % ("profile_cols", profile_cols)
###     print fmt0(just1)[1:] % ("np_hasnan_zs_isnan_sum_axis0_gt_0.shape",
###         np_hasnan_zs_isnan_sum_axis0_gt_0.shape)
###     #rint fmt1(just1)[1:] % ("np_hasnan_zs_isnan_sum_axis0_gt_0",
###     #   np_hasnan_zs_isnan_sum_axis0_gt_0)
###     print fmt0(just1)[1:] % ("profile_cols_with_isnan",
###         profile_cols_with_isnan)

### #== === === === === === === === === === === === === === === === === === ===

### image_rows = 400
### image = np.array([list(np_hasnan_zs_isnan_sum_axis0_gt_0)] * image_rows)
### if 0 and prt:
###     print fmt0(just1)[0:] % ("image.shape", image.shape)
###     None if 0 else sys.exit()
### #
### png_abspath = os.path.join(ngn.results_absdir,
###     ngn.job_zs_csv.replace('z', '').replace('.txt', '')
###     .replace('.csv', '__005_cols_with meast_zs_isnan_img.png'))
### if 0 and prt:
###     print fmt1(just1)[0:] % (
###         "ngn.results_absdir", ngn.results_absdir)
###     print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
### print fmt1(just1)[0:] % ("png_abspath", png_abspath)

### None if 1 else sys.exit()

### title = 'Columns With An Is "NaN" Value Image (%s, %s)\n%s' % (
###     '%i Col\'s' % profile_cols,
###     '%i  Is "NaN" Col\'s' % profile_cols_with_isnan, ngn.job_zs_csv)
### cmap = 'gray'
### colorbar = True if 0 else False
### xlabel = 'Profile Index'
### ylabel = 'Measurement Index' if 0 else None
### png = png_abspath if 1 else None
### imshow(image, title=title, cmap=cmap, xticklabel=True,
###     yticklabel=False, xlabel=xlabel, ylabel=ylabel,
###     colorbar=colorbar, png=png)

### #== === === === === === === === === === === === === === === === === === ===

### #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
### if prt_:
###     print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
### None if 1 else sys.exit()


def make_gallery00_profile_plot(number_of_profiles, indy,
profile_xs, profile_dxs):
    """
    Makes a plot of x-coordinate values and differences for a laser profile.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_gallery00_profile_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    profile_is = np.arange(len(profile_xs))
    profile_dis = np.arange(len(profile_xs) - 1)

    if 1 and prt:
        print fmt0(just1)[0:] % ("profile_is.shspe", profile_is.shape)
        print fmt1(just1)[1:] % ("profile_xs",
            ["%11i" % v for v in profile_is])
    if 1 and prt:
        print fmt0(just1)[1:] % ("profile_xs.shspe", profile_xs.shape)
        print fmt1(just1)[1:] % (
            "profile_xs", ["%11.3f" % v for v in profile_xs])
        #rint fmt1(just1)[1:] % ("np.unique(profile_xs)",
        #   np.unique(profile_xs))
    #
    if 1 and prt:
        print fmt0(just1)[0:] % ("profile_dis.shspe", profile_dis.shape)
    if 1 and prt:
        print fmt0(just1)[1:] % ("profile_dxs.shspe", profile_dxs.shape)
        #rint fmt1(just1)[1:] % ("profile_dxs",
        #   ["%11.3f" % v for v in profile_dxs])
        print fmt0(just1)[1:] % (
            "np.unique(np.round(profile_dxs, decimals=3))",
            np.unique(np.round(profile_dxs, decimals=3)))

    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    #
    title_fontsize = 12
    text_fontsize = 10
    ax1_get_ylim_rng_mult1 = 0.95
    ax1_get_ylim_rng_mult2 = 0.87
    ax1_get_ylim_rng_mult3 = 0.08
    axvline_ymin = 0.01
    axvline_ymax = 0.99
    axhline_xmin = 0.01
    axhline_xmax = 0.99
    ms_data = 2.
    lw_data = 0.5
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    #
    fig = plt.gcf()
    fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
    #fig.suptitle("%s:\n%s, %s" % ("Xs for Laser Profile %i of %i" %
    #    ((indy + 1), number_of_profiles), ngn.job_zs_csv, ngn.job_id))
    fig.suptitle("%s:\n%s" % ("Xs for Laser Profile %i of %i" %
        ((indy + 1), number_of_profiles), ngn.job_xs_csv))
    gridspec = [2, 1]
    gs = mpl.gridspec.GridSpec(*gridspec)
    ax1 = plt.subplot(gs[0])  # xs
    ax2 = plt.subplot(gs[1], sharex=ax1)  # dzdxs
    #
    #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    #
    ax1.set_title('Xs Profile', y=0.87, fontsize=title_fontsize)
    #x1.set_xlabel('Profile Index')
    ax1.set_ylabel('X-Coordinate')
    #
    profile_xs_nanmax = np.nanmax(profile_xs)
    profile_xs_nanmin = np.nanmin(profile_xs)
    profile_xs_nanmid = (profile_xs_nanmax + profile_xs_nanmin) / 2.
    profile_xs_nanrng_half = (profile_xs_nanmax - profile_xs_nanmin) / 2.
    ax1_set_ylim_max = profile_xs_nanmid + profile_xs_nanrng_half * 1.5
    ax1_set_ylim_min = profile_xs_nanmid - profile_xs_nanrng_half * 1.5
    ax1.set_ylim((ax1_set_ylim_min, ax1_set_ylim_max))
    #
    ax1.plot(profile_is, profile_xs, 'co-', mec='none', ms=ms_data, lw=lw_data,
        label='profile_xs')
    #
    ax1.legend(
        loc=8,
        ncol=4,
        numpoints=1,
        markerscale=1.,
        prop={'size': 9.2, 'weight': 'bold'}
    ) if 1 else None
    #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    #
    ax2.set_title('Xs First Differences Profile', y=0.87,
        fontsize=title_fontsize)
    ax2.set_xlabel('Profile Index')
    ax2.set_ylabel('X-Coordinate First Difference')
    #
    profile_is_nanmax = np.nanmax(profile_is)
    profile_is_nanmin = np.nanmin(profile_is)
    profile_is_nanmid = (profile_is_nanmax + profile_is_nanmin) / 2.
    profile_is_nanrng_half = (profile_is_nanmax - profile_is_nanmin) / 2.
    ax2_set_xlim_max = profile_is_nanmid + profile_is_nanrng_half * 1.1
    ax2_set_xlim_min = profile_is_nanmid - profile_is_nanrng_half * 1.1
    ax2.set_xlim((ax2_set_xlim_min, ax2_set_xlim_max))
    #
    profile_dxs_nanmax = np.nanmax(profile_dxs)
    #rofile_dxs_nanmin = np.nanmin(profile_dxs)
    profile_dxs_nanmin = -profile_dxs_nanmax
    profile_dxs_nanmid = (profile_dxs_nanmax + profile_dxs_nanmin) / 2.
    profile_dxs_nanrng_half = (profile_dxs_nanmax - profile_dxs_nanmin) / 2.
    ax2_set_ylim_max = profile_dxs_nanmid + profile_dxs_nanrng_half * 2.0
    #x2_set_ylim_min = profile_dxs_nanmid - profile_dxs_nanrng_half * 1.1
    ax2_set_ylim_min = -ax2_set_ylim_max
    ax2.set_ylim((ax2_set_ylim_min, ax2_set_ylim_max))
    #
    ax2.axhline(y=0., xmin=0.01, xmax=0.99, c='r', ls='dotted', lw=2.)
    #
    ax2.plot(profile_dis, np.round(profile_dxs, decimals=3), 'co-', mec='none',
        ms=ms_data, lw=lw_data, label='np.round(profile_dxs, decimals=3)')
    #
    ax2.legend(
        loc=8,
        ncol=5,
        numpoints=1,
        markerscale=1.,
        prop={'size': 8.0, 'weight': 'bold'}
    ) if 1 else None
    #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
    #
    png_abspath = os.path.join(ngn.gallery00_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '')
        .replace('.csv', '_meast_%.5i.png' % (indy + 1)))
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.gallery00_absdir", ngn.gallery00_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    #
    png = png_abspath if 1 else None
    if png is not None:
        None if os.path.isdir(ngn.gallery00_absdir) else (
            os.makedirs(ngn.gallery00_absdir))
        plt.savefig(png)
    else:
        plt.show()
    plt.close()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def calculate_profile_quality_metrics(profile_zs_mask_nans):
    """
    Returns tuple containing laser profile quality metric values.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'calculate_profile_quality_metrics'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        #rint fmt0(just1)[0:] % ("profile_xs.shape", profile_xs.shape)
        #rint fmt0(just1)[1:] % ("profile_zs.shape", profile_zs.shape)
        print fmt0(just1)[0:] % (
            "profile_zs_mask_nans.shape", profile_zs_mask_nans.shape)
        print fmt1(just1)[1:] % (
            "np.where(profile_zs_mask_nans)", np.where(profile_zs_mask_nans))
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "np.sum(~ profile_zs_mask_nans)", np.sum(~ profile_zs_mask_nans))

    nnan_zs_where = np.where(~ profile_zs_mask_nans)[0]
    zs_idx_nnan_lf = nnan_zs_where[0]
    zs_idx_nnan_rt = nnan_zs_where[-1]
    if 1 and prt:
        print fmt0(just1)[1:] % ("zs_idx_nnan_lf", zs_idx_nnan_lf)
        print fmt0(just1)[1:] % ("zs_idx_nnan_rt", zs_idx_nnan_rt)

    None if 1 else sys.exit()

    pts_fov = profile_zs_mask_nans.shape[0]
    pts_roi = zs_idx_nnan_rt - zs_idx_nnan_lf + 1
    pts_value = (~ profile_zs_mask_nans).sum()
    pts_drop = pts_roi - pts_value
    if 1 and prt:
        print fmt0(just1)[0:] % ("pts_fov", pts_fov)
        print fmt0(just1)[1:] % ("pts_value", pts_value)
        print fmt0(just1)[1:] % ("pts_roi", pts_roi)
        print fmt0(just1)[1:] % ("pts_drop", pts_drop)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return (nnan_zs_where, zs_idx_nnan_lf, zs_idx_nnan_rt,
        pts_fov, pts_roi, pts_value, pts_drop)

#ZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ
#ZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ
# (below) defs ... for analyzing tow ends placement


def lsbf_line_slope_and_intercept(xs, ys):
    """
    Return the slope, m, and the intercept, c, given x and y values, xs & ys.

    xs, ys -- Numpy ndarrays with the same shape.
    """
    # ys.T = Ap, where A = [[xs.T 1]] and p = [[m], [c]].
    A = np.vstack([xs, np.ones(len(xs))]).T
    m, c = np.linalg.lstsq(A, ys)[0]
    #rint m, c
    return m, c


def make_gallery03_tow_ends_placement_plot_xz(number_of_profiles,
indy, window_indy, pd_src_us_tow_diff_row, tow_diff_names,
window_xs, window_ys, window_zs_ntrend0, window_zs_fltr0_ntrend0,
window_zs_fltr0_ntrend0_edges_start, window_ys_fltr0_ntrend0_edges_start,
window_zs_fltr0_ntrend0_edges_stop, window_ys_fltr0_ntrend0_edges_stop,
window_zs_fltr0_ntrend0_edges_stop_rois,
window_zs_fltr0_ntrend0_edges_start_rois,
np_roi_tow_xs_center, np_roi_tow_ys_mean_meast_idx, np_roi_tow_starts_stops):
    """
    Makes front- & top-view plots of laser & (ROI) edge profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_gallery03_tow_ends_placement_plot_xz'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    prt = True if 1 and prt_ else False

    target_u_sensor = pd_src_us_tow_diff_row['U-Sensor']
    if 0 and prt:
        print fmt0(just1)[0:] % ("target_u_sensor", target_u_sensor)

    #== === === === === === === === === === === === === === === === === === ===

    axhline_xmin = axvline_ymin = 0.01
    axhline_xmax = axvline_ymax = 0.99

    suptitle = "%s:\n%s" % ("Laser Profiles (X, Z) %s of %i" % (
        window_indy + 1, number_of_profiles), ngn.job_zs_csv)
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("suptitle", suptitle)

    ax1_title = pd_src_us_tow_diff_row.to_string(
        index=False).replace('  ', '    ')
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff_row",
    #       pd_src_us_tow_diff_row)
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("ax1_title", ax1_title)

    ax2_title_fmt = 'Tow End Target(s):  Measurement ID %i, U-Sensor %.3f'
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("ax2_title", ax2_title)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    #
    fig = plt.gcf()
    fig.suptitle(suptitle)
    fig.set_size_inches(16, 12, forward=True)  # default is (8, 6)
    gs = mpl.gridspec.GridSpec(2, 1)
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1 = plt.subplot(gs[0])
    ax1.set_title(ax1_title)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Z Coordinate')

    ax1_ylim_mgn = 1.50
    nanmax = np.nanmax(window_zs_fltr0_ntrend0)
    nanmin = np.nanmin(window_zs_fltr0_ntrend0)
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ax1_ylim_mgn
    ax1.set_ylim((nanmid - nanmgn, nanmid + nanmgn))
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("[nanmin, nanmax]", [nanmin, nanmax])
    #   print fmt0(just1)[1:] % ("ax1.get_ylim()", ax1.get_ylim())

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    for win_row_idx in xrange(window_xs.shape[0]):
        ax1.plot(
            window_xs[win_row_idx, :],
            window_zs_ntrend0[win_row_idx, :],
            ',-',
            color='gray',
            mec='none', lw=2
        ) if 1 else None

    for win_row_idx in xrange(window_xs.shape[0]):
        ax1.plot(
            window_xs[win_row_idx, :],
            window_zs_fltr0_ntrend0[win_row_idx, :],
            ',-',
            #color='gray',
            mec='none', lw=2
        ) if 1 else None

    for win_row_idx in xrange(window_xs.shape[0]):
        ax1.plot(
            window_xs[win_row_idx, :],
            window_zs_fltr0_ntrend0_edges_stop[win_row_idx, :],
            ',-',
            color='pink',
            mec='none', ms=48., lw=16.
        ) if 1 else None
    for win_row_idx in xrange(window_xs.shape[0]):
        ax1.plot(
            window_xs[win_row_idx, :],
            window_zs_fltr0_ntrend0_edges_start[win_row_idx, :],
            ',-',
            color='lightgreen',
            mec='none', ms=48., lw=16.
        ) if 1 else None

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    for i1, tow_edge_xref in enumerate(ngn.tow_edge_xrefs):
        ax1_axvline_ls = ('dashed'
            if i1 == 0 or (i1 + 1) == len(ngn.tow_edge_xrefs)
            else 'dotted')
        ax1.axvline(x=tow_edge_xref, ymin=axvline_ymin,
            ymax=axvline_ymax, c='y', ls=ax1_axvline_ls, lw=2.)

    trans1 = mpl.transforms.blended_transform_factory(
        ax1.transData, ax1.transAxes)

    zipped = zip(ngn.tow_ids[1:-1], ngn.tow_center_xrefs[1:-1])
    for i11, (tid, tcx) in enumerate(zipped):
        ax1.text(tcx, 0.94 if i11 % 2 else 0.99, "tow\n%.2i" % tid,
            color='black', fontsize=8, fontweight='bold',
            ha='center', va='top', transform=trans1)

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.set_title(ax2_title_fmt % (indy + 1, target_u_sensor))
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate (Measurement Index)')

    ax2_ylim_mgn = 1.40
    nanmax = np.nanmax(window_ys)
    nanmin = np.nanmin(window_ys)
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ax2_ylim_mgn
    ax2.set_ylim((nanmid - nanmgn, nanmid + nanmgn))
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("[nanmin, nanmax]", [nanmin, nanmax])
    #   print fmt0(just1)[1:] % ("ax2.get_ylim()", ax2.get_ylim())

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    ax2.plot(
        [np.nanmin(window_xs), np.nanmax(window_xs)],
        [indy, indy],
        '-',
        color='yellow',
        mec='none', lw=8.,
        label="tow end(s) target location"
    ) if 1 else None

    for win_col_idx in range(window_ys.shape[1]):
        ax2.plot(
            window_xs[:, win_col_idx],
            window_ys[:, win_col_idx],
            #indow_ys[:, win_col_idx] + 1,
            ',',
            color='gray',
            mec='none', lw=2
        ) if 1 else None

    for win_col_idx in range(window_ys.shape[1]):
        ax2.plot(
            window_xs[:, win_col_idx],
            window_ys_fltr0_ntrend0_edges_start[:, win_col_idx],
            ',-',
            color='lightgreen',
            mec='none',  # ms=16., lw=2.
            label="tow start edge(s)" if win_col_idx == 0 else None
        ) if 1 else None
    for win_col_idx in range(window_ys.shape[1]):
        ax2.plot(
            window_xs[:, win_col_idx],
            window_ys_fltr0_ntrend0_edges_stop[:, win_col_idx],
            ',-',
            color='pink',
            mec='none',  # ms=16., lw=2.
            label="tow stop edge(s)" if win_col_idx == 0 else None
        ) if 1 else None

    np_roi_tow_start_mask = np_roi_tow_starts_stops == 1
    np_roi_tow_ys_mean_meast_idx_start = np_roi_tow_ys_mean_meast_idx.copy()
    np_roi_tow_ys_mean_meast_idx_start[~ np_roi_tow_start_mask] = np.nan
    ax2.plot(
        np_roi_tow_xs_center,
        np_roi_tow_ys_mean_meast_idx_start,
        'o',
        color='green',
        #mec='none',
        ms=8.,
        # lw=8.
        label="tow end start location(s)"
    ) if 1 else None

    np_roi_tow_stop_mask = np_roi_tow_starts_stops == -1
    np_roi_tow_ys_mean_meast_idx_stop = np_roi_tow_ys_mean_meast_idx.copy()
    np_roi_tow_ys_mean_meast_idx_stop[~ np_roi_tow_stop_mask] = np.nan
    ax2.plot(
        np_roi_tow_xs_center,
        np_roi_tow_ys_mean_meast_idx_stop,
        'o',
        color='red',
        #mec='none',
        ms=8.,
        # lw=8.
        label="tow end stop location(s)"
    ) if 1 else None

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    for i2, strip_edge_xref in enumerate(ngn.tow_edge_xrefs):
        ax2_axvline_ls = ('dashed'
            if i2 == 0 or (i2 + 1) == len(ngn.tow_edge_xrefs)
            else 'dotted')
        ax2.axvline(x=strip_edge_xref, ymin=axvline_ymin,
            ymax=axvline_ymax, c='y', ls=ax2_axvline_ls, lw=2.)

    trans2 = mpl.transforms.blended_transform_factory(
        ax2.transData, ax2.transAxes)

    zipped = zip(ngn.tow_ids[1:-1], ngn.tow_center_xrefs[1:-1])
    for i22, (tid, tcx) in enumerate(zipped):
        ax2.text(tcx, 0.94 if i22 % 2 else 0.99, "tow\n%.2i" % tid,
            color='black', fontsize=8, fontweight='bold',
            ha='center', va='top', transform=trans2)

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    ax2.legend(
        loc=8,
        ncol=6,
        numpoints=1,
        markerscale=1.,
        prop={'size': 9.2, 'weight': 'bold'}
    ) if 1 else None

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    png_abspath = os.path.join(ngn.gallery03_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '')
        .replace('.csv', '_meast_ref_%.5i_xz.png' % (indy + 1)))
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("ngn.gallery03_absdir",
    #       ngn.gallery03_absdir)
    #   print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    #
    png = png_abspath if 1 else None
    if png is not None:
        None if os.path.isdir(ngn.gallery03_absdir) else (
            os.makedirs(ngn.gallery03_absdir))
        plt.savefig(png)
    else:
        plt.show()
    plt.close()
    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_gallery03_tow_ends_placement_plot_yz_start(number_of_profiles,
indy, window_indy, pd_src_us_tow_diff_row, tow_diff_names, window_ys,
window_zs_fltr0_ntrend0, window_ys_midp, window_dzdys_fltr0_ntrend0,
window_dzdys_fltr0_ntrend0_ge0, window_dzdys_fltr0_ntrend0_edges_start,
window_zs_fltr0_ntrend0_edges_start, dzdys_threshold):
    """
    Makes side-view plots of laser profiles & their first differences with
    their corresponding positive edge profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_gallery03_tow_ends_placement_plot_yz_start'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    target_u_sensor = pd_src_us_tow_diff_row['U-Sensor']
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("target_u_sensor", target_u_sensor)

    #== === === === === === === === === === === === === === === === === === ===

    axhline_xmin = axvline_ymin = 0.01
    axhline_xmax = axvline_ymax = 0.99

    suptitle = "%s:\n%s" % ("Laser Profiles (X, Z) %s of %i" % (
        window_indy + 1, number_of_profiles), ngn.job_zs_csv)
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("suptitle", suptitle)

    ax1_title = pd_src_us_tow_diff_row.to_string(
        index=False).replace('  ', '    ')
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff_row",
    #       pd_src_us_tow_diff_row)
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("ax1_title", ax1_title)

    ax2_title_fmt = 'Tow End%s Target(s):  Measurement ID %i, U-Sensor %.3f'
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("ax2_title", ax2_title)

    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    fig = plt.gcf()
    fig.suptitle(suptitle)
    fig.set_size_inches(16, 12, forward=True)  # default is (8, 6)
    gs = mpl.gridspec.GridSpec(2, 1)

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1 = plt.subplot(gs[0])
    ax1.set_title(ax1_title)
    ax1.set_xlabel('Y Coordinate')
    ax1.set_ylabel('Z Coordinate')

    ax1.set_xlim((window_ys[0, 0] - 1, window_ys[-1, 0] + 1))
    ylim_mgn = 1.10
    nanmax = np.nanmax(window_zs_fltr0_ntrend0)
    nanmin = np.nanmin(window_zs_fltr0_ntrend0)
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ylim_mgn
    ax1.set_ylim((nanmid - nanmgn, nanmid + nanmgn))
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("[nanmin, nanmax]", [nanmin, nanmax])
    #   print fmt0(just1)[1:] % ("ax1.get_ylim()", ax1.get_ylim())

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    for win_col_idx in range(window_ys.shape[1]):
        ax1.plot(
            window_ys[:, win_col_idx],
            window_zs_fltr0_ntrend0[:, win_col_idx],
            ',-',
            #color='gray',
            mec='none', lw=2
        ) if 1 else None

    for win_col_idx in range(window_ys.shape[1]):
        ax1.plot(
            window_ys[:, win_col_idx],
            window_zs_fltr0_ntrend0_edges_start[:, win_col_idx],
            'o-',
            color='lightgreen',
            mec='none', ms=16., lw=4.
        ) if 1 else None

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.set_title(ax2_title_fmt % (' Start', indy + 1, target_u_sensor))
    ax2.set_xlabel('Y Coordinate')
    ax2.set_ylabel('dZ/dy Coordinate')

    #x1.set_xlim((window_ys[0, 0] - 1, window_ys[-1, 0] + 1))
    ylim_mgn = 1.20
    nanmax = np.nanmax(np.abs(window_dzdys_fltr0_ntrend0))
    nanmin = -nanmax
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ylim_mgn
    ax2.set_ylim((nanmid - nanmgn, nanmid + nanmgn))

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    for win_col_idx in range(window_ys.shape[1]):
        ax2.plot(
            window_ys_midp[:, win_col_idx],
            window_dzdys_fltr0_ntrend0[:, win_col_idx],
            ',-',
            color='gray',
            mec='none', lw=2
        ) if 1 else None

    for win_col_idx in range(window_ys.shape[1]):
        ax2.plot(
            window_ys_midp[:, win_col_idx],
            window_dzdys_fltr0_ntrend0_ge0[:, win_col_idx],
            ',-',
            #color='gray',
            mec='none', lw=2
        ) if 1 else None

    ax2.plot(
        window_ys_midp,
        window_dzdys_fltr0_ntrend0_edges_start,
        'o',
        color='lightgreen',
        mec='none', ms=16.,
    ) if 1 else None

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    ax2.axhline(y=dzdys_threshold,
        xmin=axhline_xmin, xmax=axhline_xmax,
        c='y', ls='dashed', lw=2.)
    ax2.text(0.05, 0.95,
        "tow start(s):  >=  %.3f dzdy threshold" % dzdys_threshold,
        transform=ax2.transAxes, fontsize=14,  # fontweight='bold',
        va='center')

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    png_abspath = os.path.join(ngn.gallery03_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '')
        .replace('.csv', '_meast_ref_%.5i_yz_start.png' % (indy + 1)))
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("ngn.gallery03_absdir",
    #       ngn.gallery03_absdir)
    #   print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    #
    png = png_abspath if 1 else None
    if png is not None:
        None if os.path.isdir(ngn.gallery03_absdir) else (
            os.makedirs(ngn.gallery03_absdir))
        plt.savefig(png)
    else:
        plt.show()
    plt.close()
    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_gallery03_tow_ends_placement_plot_yz_stop(number_of_profiles,
indy, window_indy, pd_src_us_tow_diff_row, tow_diff_names, window_ys,
window_zs_fltr0_ntrend0, window_ys_midp, window_dzdys_fltr0_ntrend0,
window_dzdys_fltr0_ntrend0_le0, window_dzdys_fltr0_ntrend0_edges_stop,
window_zs_fltr0_ntrend0_edges_stop, dzdys_threshold):
    """
    Makes side-view plots of laser profiles & their first differences with
    their corresponding negative edge profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_gallery03_tow_ends_placement_plot_yz_stop'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    target_u_sensor = pd_src_us_tow_diff_row['U-Sensor']
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("target_u_sensor", target_u_sensor)

    #== === === === === === === === === === === === === === === === === === ===

    axhline_xmin = axvline_ymin = 0.01
    axhline_xmax = axvline_ymax = 0.99

    suptitle = "%s:\n%s" % ("Laser Profiles (X, Z) %s of %i" % (
        window_indy + 1, number_of_profiles), ngn.job_zs_csv)
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("suptitle", suptitle)

    ax1_title = pd_src_us_tow_diff_row.to_string(
        index=False).replace('  ', '    ')
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff_row",
    #       pd_src_us_tow_diff_row)
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("ax1_title", ax1_title)

    ax2_title_fmt = 'Tow End%s Target(s):  Measurement ID %i, U-Sensor %.3f'
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("ax2_title", ax2_title)

    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    fig = plt.gcf()
    fig.suptitle(suptitle)
    fig.set_size_inches(16, 12, forward=True)  # default is (8, 6)
    gs = mpl.gridspec.GridSpec(2, 1)

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1 = plt.subplot(gs[0])
    ax1.set_title(ax1_title)
    ax1.set_xlabel('Y Coordinate')
    ax1.set_ylabel('Z Coordinate')

    ax1.set_xlim((window_ys[0, 0] - 1, window_ys[-1, 0] + 1))

    ylim_mgn = 1.10
    nanmax = np.nanmax(window_zs_fltr0_ntrend0)
    nanmin = np.nanmin(window_zs_fltr0_ntrend0)
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ylim_mgn
    ax1.set_ylim((nanmid - nanmgn, nanmid + nanmgn))
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("[nanmin, nanmax]", [nanmin, nanmax])
    #   print fmt0(just1)[1:] % ("ax1.get_ylim()", ax1.get_ylim())

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    for win_col_idx in range(window_ys.shape[1]):
        ax1.plot(
            window_ys[:, win_col_idx],
            window_zs_fltr0_ntrend0[:, win_col_idx],
            ',-',
            #color='gray',
            mec='none', lw=2
        ) if 1 else None

    for win_col_idx in range(window_ys.shape[1]):
        ax1.plot(
            window_ys[:, win_col_idx],
            window_zs_fltr0_ntrend0_edges_stop[:, win_col_idx],
            'o-',
            color='pink',
            mec='none', ms=16., lw=4.
        ) if 1 else None

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.set_title(ax2_title_fmt % (' Stop', indy + 1, target_u_sensor))
    ax2.set_xlabel('Y Coordinate')
    ax2.set_ylabel('dZ/dy Coordinate')

    #x1.set_xlim((window_ys[0, 0] - 1, window_ys[-1, 0] + 1))
    ylim_mgn = 1.20
    nanmax = np.nanmax(np.abs(window_dzdys_fltr0_ntrend0))
    nanmin = -nanmax
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ylim_mgn
    ax2.set_ylim((nanmid - nanmgn, nanmid + nanmgn))

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    for win_col_idx in range(window_ys.shape[1]):
        ax2.plot(
            window_ys_midp[:, win_col_idx],
            window_dzdys_fltr0_ntrend0[:, win_col_idx],
            ',-',
            color='gray',
            mec='none', lw=2
        ) if 1 else None

    for win_col_idx in range(window_ys.shape[1]):
        ax2.plot(
            window_ys_midp[:, win_col_idx],
            window_dzdys_fltr0_ntrend0_le0[:, win_col_idx],
            ',-',
            #color='gray',
            mec='none', lw=2
        ) if 1 else None

    ax2.plot(
        window_ys_midp,
        window_dzdys_fltr0_ntrend0_edges_stop,
        'o',
        color='pink',
        mec='none', ms=16.,
    ) if 1 else None

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    ax2.axhline(y=-dzdys_threshold,
        xmin=axhline_xmin, xmax=axhline_xmax,
        c='y', ls='dashed', lw=2.)
    ax2.text(0.05, 0.05,
        "tow stop(s):  <=  %.3f dzdy threshold" % -dzdys_threshold,
        transform=ax2.transAxes, fontsize=14,  # fontweight='bold',
        va='center')

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    png_abspath = os.path.join(ngn.gallery03_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '')
        .replace('.csv', '_meast_ref_%.5i_yz_stop.png' % (indy + 1)))
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("ngn.gallery03_absdir",
    #       ngn.gallery03_absdir)
    #   print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    #
    png = png_abspath if 1 else None
    if png is not None:
        None if os.path.isdir(ngn.gallery03_absdir) else (
            os.makedirs(ngn.gallery03_absdir))
        plt.savefig(png)
    else:
        plt.show()
    plt.close()
    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_gallery04_tow_end_event_window_plot(
    i, indy, np_hasnan_zs, window_xs, window_zs_fltr0,
    window_zs_bins, window_zs_bincounts_fltrd,
    window_zs_bincounts_2lbls_mask,
    bincounts_max0_z, bincounts_max0,
    bincounts_max1_z, bincounts_max1,
    bincounts_mid_z, bincounts_mid,
):
    #"""
    #Makes an overlay plot of all (unfiltered) laser profiles.
    #"""
    """
    Makes histogram plots of profiles in a tow end event window.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_gallery04_tow_end_event_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    fig = plt.gcf()
    fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
    #fig.suptitle(
    #    "Laser Measurements Overlay (%i Meas'ts, %i Points Each)\n%s" % (
    #        profile_measts, profile_points, ngn.job_zs_csv))
    gridspec = [1, 2]
    gs = mpl.gridspec.GridSpec(*gridspec)
    #
    #labelsize = 10.0
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1 = plt.subplot(gs[0])
    ax1.set_ylabel('Z-coordinate Value')
    ax1.set_xlabel('Z-coordinate Value')
    #ax1.xaxis.set_tick_params(labelsize=labelsize)
    #ax1.yaxis.set_tick_params(labelsize=labelsize)
    #
    for iy in xrange(window_zs_fltr0.shape[0]):
        xs = window_xs[iy, :]
        zs = window_zs_fltr0[iy, :]
        ax1.plot(xs, zs, ',-')
    #
    ax1.axhline(y=bincounts_max0_z,
        xmin=0.01, xmax=0.99, c='m', ls='dashed', lw=3.)
    ax1.axhline(y=bincounts_mid_z,
        xmin=0.01, xmax=0.99, c='m', ls='dashed', lw=3.)
    ax1.axhline(y=bincounts_max1_z,
        xmin=0.01, xmax=0.99, c='m', ls='dashed', lw=3.)
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax2 = plt.subplot(gs[1], sharey=ax1)
    ax2.set_title("bincounts: %s, %s, max1/max0 = %.3f" % (
        'max0 = %i' % bincounts_max0,
        'max1 = %i' % bincounts_max1,
        (1.0 * bincounts_max1 / bincounts_max0)),
        fontsize=9.0)
    #x2.set_ylabel('Z-coordinate Value')
    ax2.set_xlabel('Z-coordinate Value Counts')
    #ax2.xaxis.set_tick_params(labelsize=labelsize)
    #ax2.yaxis.set_tick_params(labelsize=labelsize)
    #
    ax2.set_xlim((0.0, np.max(window_zs_bincounts_fltrd) * 1.3))
    ax2.set_ylim((np.nanmin(np_hasnan_zs), np.nanmax(np_hasnan_zs)))
    #
    ax2.axvline(x=bincounts_max0,
        ymin=0.01, ymax=0.99, c='y', ls='dashed', lw=2.)
    ax2.axvline(x=bincounts_max1,
        ymin=0.01, ymax=0.99, c='y', ls='dashed', lw=2.)
    #
    ax2.plot([0.0, bincounts_max0],
        [bincounts_max0_z, bincounts_max0_z], 'm--', lw=3.)
    ax2.plot([0.0, bincounts_max1],
        [bincounts_max1_z, bincounts_max1_z], 'm--', lw=3.)
    ax2.plot([0.0, bincounts_mid],
        [bincounts_mid_z, bincounts_mid_z], 'm--', lw=3.)
    #
    ax2.text(bincounts_max0, bincounts_max0_z,
        'Max0_Z:\n%.3f' % bincounts_max0_z,
        fontsize=9, fontweight='bold', ha='left', va='center')
    ax2.text(bincounts_max1, bincounts_max1_z,
        'Max1_Z:\n%.3f' % bincounts_max1_z,
        fontsize=9, fontweight='bold', ha='left', va='center')
    ax2.text(bincounts_mid, bincounts_mid_z,
        'Mid_Z:\n%.3f' % bincounts_mid_z,
        fontsize=10, fontweight='bold', ha='left', va='center')
    #
    ax2.plot(window_zs_bincounts_fltrd, window_zs_bins, 'o-', mec='none')
    ax2.plot(window_zs_bincounts_fltrd[window_zs_bincounts_2lbls_mask],
        window_zs_bins[window_zs_bincounts_2lbls_mask], 'ro', mec='none', ms=6)
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    tow_event_type = ''
    tow_event_type = '_start' if tow_event_type > 0 else tow_event_type
    tow_event_type = '_stop' if tow_event_type < 0 else tow_event_type
    png_abspath = os.path.join(ngn.gallery04_absdir,
        ngn.job_zs_csv.replace('.txt', '').replace(
            '.csv', '__tow_end_event_%.4i_meast_%.5i__window.png' % (
                (i + 1), (indy + 1))))
    if 1 and prt:
        print fmt0(just1)[0:] % ("ngn.gallery04_absdir", ngn.gallery04_absdir)
        print fmt0(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    png = png_abspath if 1 else None
    if png is not None:
        None if os.path.isdir(ngn.gallery04_absdir) else (
            os.makedirs(ngn.gallery04_absdir))
        plt.savefig(png)
    else:
        plt.show()
    plt.close()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_gallery05_tow_end_event_window_roi_plot(
    i, indy, tow_roi_id, tow_diff_roi_event,
    #window_xs, window_ys, window_zs_fltr0,
    #window_zs_bins, window_zs_bincounts, window_zs_bincounts_fltrd,
    #window_zs_bincounts_fltrd_max_z, window_zs_bincounts_fltrd_max,
):
    """
    Makes a overlay & histogram plots of profiles in a tow end event window
    roi.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_gallery05_tow_end_event_window_roi_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    fig = plt.gcf()
    fig.set_size_inches(12, 6, forward=True)  # default is (8, 6)
    #fig.suptitle(
    #    "Laser Measurements Overlay (%i Meas'ts, %i Points Each)\n%s" % (
    #        profile_measts, profile_points, ngn.job_zs_csv))
    gridspec = [1, 2]
    gs = mpl.gridspec.GridSpec(*gridspec)
    #
    #labelsize = 9.0
    lw_max_z = 6
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1 = plt.subplot(gs[0])  # xs vs zs
    ax1.set_xlabel("X-coordinate Value")
    ax1.set_ylabel('Z-coordinate Value')
#   #
#   for iy in xrange(window_zs_fltr0.shape[0]):
#       xs = window_xs[iy, :]
#       zs = window_zs_fltr0[iy, :]
#       ax1.plot(xs, zs, ',-')
#   #
#   ax1.plot([np.nanmin(window_xs), np.nanmax(window_xs)],
#       [window_zs_bincounts_fltrd_max_z, window_zs_bincounts_fltrd_max_z],
#       ',--', color='black', lw=lw_max_z)
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax2 = plt.subplot(gs[1], sharey=ax1)  # ys vs zs
    ax2.set_xlabel("Y-coordinate Value (Meas't Index)")
    #x2.set_ylabel('Z-coordinate Value')
    #ax2.xaxis.set_tick_params(labelsize=labelsize)
    #ax2.yaxis.set_tick_params(labelsize=labelsize)
#   #
#   #ax2.set_xlim((ys[0] - 0.5, ys[-1] + 0.5))
#   #
#   for ix in xrange(window_zs_fltr0.shape[1]):
#       ys = window_ys[:, ix]
#       zs = window_zs_fltr0[:, ix]
#       ax2.plot(ys, zs, ',-')
#   #
#   ax2.plot([np.nanmin(window_ys), np.nanmax(window_ys)],
#       [window_zs_bincounts_fltrd_max_z, window_zs_bincounts_fltrd_max_z],
#       ',--', color='black', lw=lw_max_z)
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
#   ax3 = plt.subplot(gs[2], sharey=ax1) # zs count vs zs uniq
    #x3.set_ylabel('Z-coordinate Value')
#   ax3.set_xlabel('Z-coordinate Value Counts')
    #ax3.xaxis.set_tick_params(labelsize=labelsize)
    #ax3.yaxis.set_tick_params(labelsize=labelsize)
    #
#   ax3.barh(window_zs_bins, window_zs_bincounts,
#       height=0.009, align='center', color='cyan', alpha=0.3)
#   ax3.barh(window_zs_bins, window_zs_bincounts_fltrd,
#       height=0.003, align='center', color='blue', alpha=1.0)
#   #
#   ax3.plot([0.0, window_zs_bincounts_fltrd_max],
#       [window_zs_bincounts_fltrd_max_z, window_zs_bincounts_fltrd_max_z],
#       ',--', color='black', lw=lw_max_z)
#   #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    tow_event_type = ''
    tow_event_type = '_start' if tow_diff_roi_event > 0 else tow_event_type
    tow_event_type = '_stop' if tow_diff_roi_event < 0 else tow_event_type
    png_abspath = os.path.join(ngn.gallery05_absdir,
        ngn.job_zs_csv.replace('.txt', '').replace(
            '.csv', '__tow_end_event_%.4i_meast_%.5i_tow_%.2i%s.png' % (
                (i + 1), (indy + 1), tow_roi_id, tow_event_type)))
    if 1 and prt:
        print fmt0(just1)[0:] % ("ngn.gallery05_absdir", ngn.gallery05_absdir)
        print fmt0(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    png = png_abspath if 1 else None
    if png is not None:
        None if os.path.isdir(ngn.gallery05_absdir) else (
            os.makedirs(ngn.gallery05_absdir))
        plt.savefig(png)
    else:
        plt.show()
    plt.close()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


# def make_gallery04_tow_end_event_plot(
# indy, tow_roi_id, tow_diff_roi_event,
# roi_event_xs, roi_event_ys, roi_event_zs,
# roi_event_zs_bins, roi_event_zs_bincounts,
# ):
##  #"""
##  #Makes an overlay plot of all (unfiltered) laser profiles.
##  #"""
##  """
##  Makes laser measurement profiles & histogram plots for a tow end event.
##  """
##  just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
##  prt = False if 0 else True
##  prt_ = prt
##  mult_str = '--- '
##  def_str = 'make_gallery04_tow_end_event_plot'
##  if prt_:
##      print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
##  #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

##  prt = True if 1 and prt_ else False

##  #== === === === === === === === === === === === === === === === === === ===

#   profile_measts, profile_points = np_hasnan_zs.shape
#   if 1 and prt:
#       print fmt0(just1)[0:] % ("np_hasnan_zs.shape", np_hasnan_zs.shape)
#       print fmt0(just1)[1:] % ("profile_measts", profile_measts)
#       print fmt0(just1)[1:] % ("profile_points", profile_points)

##  fig = plt.gcf()
##  fig.set_size_inches(16, 6, forward=True)  # default is (8, 6)
#   fig.suptitle(
#       "Laser Measurements Overlay (%i Meas'ts, %i Points Each)\n%s" % (
#           profile_measts, profile_points, ngn.job_zs_csv))
##  gridspec = [1, 3]
##  gs = mpl.gridspec.GridSpec(*gridspec)
#   #
#   labelsize = 9.0
##  #
##  #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##  ax1 = plt.subplot(gs[0])  # xs vs zs
##  ax1.set_xlabel("X-coordinate Value")
##  ax1.set_ylabel('Z-coordinate Value')
##  #
##  for iy in xrange(roi_event_zs.shape[0]):
##      xs = roi_event_xs[iy, :]
##      zs = roi_event_zs[iy, :]
##      ax1.plot(xs, zs, ',-')
##  #
##  #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##  ax2 = plt.subplot(gs[1], sharey=ax1)  # ys vs zs
##  ax2.set_xlabel("Y-coordinate Value (Meas't Index)")
##  #x2.set_ylabel('Z-coordinate Value')
#   ax2.xaxis.set_tick_params(labelsize=labelsize)
#   ax2.yaxis.set_tick_params(labelsize=labelsize)
#   #
#   ys = np.arange(profile_measts)
#   if 1 and prt:
#       print fmt1(just1)[0:] % ("ys", ys)
#   None if 1 else sys.exit()

#   ax2.set_xlim((ys[0] - 0.5, ys[-1] + 0.5))

#   for indx in xrange(profile_points):
#       zs = np_hasnan_zs[:, indx]
#       ax2.plot(ys, zs, ',-')
##  for ix in xrange(roi_event_zs.shape[1]):
##      ys = roi_event_ys[:, ix]
##      zs = roi_event_zs[:, ix]
##      ax2.plot(ys, zs, ',-')
##  #
##  #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##  ax3 = plt.subplot(gs[2], sharey=ax1) # zs count vs zs uniq
##  #x2.set_ylabel('Z-coordinate Value')
##  ax3.set_xlabel('Z-coordinate Value Counts')
#   ax3.xaxis.set_tick_params(labelsize=labelsize)
#   ax3.yaxis.set_tick_params(labelsize=labelsize)
#   #
#   #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
#   profile_values = np_hasnan_zs.size
#   if 1 and prt:
#       print fmt0(just1)[0:] % ("np_hasnan_zs.size", np_hasnan_zs.size)

#   np_zs = np_hasnan_zs.copy().flatten()
#   np_zs_mask_isnan = np.isnan(np_zs)
#   np_zs_nnan = np_zs[~ np_zs_mask_isnan]
#   if 1 and prt:
#       print fmt0(just1)[0:] % ("np_zs.size", np_zs.size)
#       print fmt0(just1)[1:] % (
#           "np.sum(np_zs_mask_isnan)", np.sum(np_zs_mask_isnan))
#       print fmt0(just1)[1:] % ("np_zs_nnan.size", np_zs_nnan.size)
#       None if 1 else sys.exit()

#   decimals = 2
#   np_zs_nnan_rnd = np.round(np_zs_nnan, decimals=decimals)
#   if 1 and prt:
#       print fmt0(just1)[0:] % (
#           "np_zs_nnan_rnd.shape", np_zs_nnan_rnd.shape)
#       print fmt0(just1)[1:] % ("np_zs_nnan_rnd.size", np_zs_nnan_rnd.size)
#       #rint fmt1(just1)[1:] % ("np_zs_nnan_rnd", np_zs_nnan_rnd)
#       #rint fmt1(just1)[1:] % (
#       #   "np.sort(np_zs_nnan_rnd)", np.sort(np_zs_nnan_rnd))
#       None if 1 else sys.exit()

#   bins, bincounts = np.unique(np_zs_nnan_rnd, return_counts=True)
#   if 1 and prt:
#       print fmt0(just1)[0:] % ("bins.size", bins.size)
#       #rint fmt1(just1)[1:] % ("bins", bins)
#       None if 1 else sys.exit()
#   if 0 and prt:
#       print fmt0(just1)[1:] % ("bincounts.size", bincounts.size)
#       print fmt1(just1)[1:] % ("bincounts", bincounts)
#       print fmt1(just1)[1:] % ("np.sort(bincounts)", np.sort(bincounts))
#       None if 1 else sys.exit()
#   #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
##  ax3.barh(roi_event_zs_bins, roi_event_zs_bincounts,
##      height=0.008, align='center', color='None', alpha=0.8) if 0 else None
##  #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##  tow_event_type = ''
##  tow_event_type = '_start' if tow_event_type > 0  else tow_event_type
##  tow_event_type = '_stop' if tow_event_type < 0  else tow_event_type
##  png_abspath = os.path.join(ngn.gallery04_absdir,
##      ngn.job_zs_csv.replace('.txt', '').replace(
##          '.csv', '__tow_end_event_meast_%.5i_tow_%.2i%s.png' % (
##              (indy + 1), tow_roi_id, tow_event_type)))
##  if 1 and prt:
##      print fmt0(just1)[0:] % ("ngn.gallery04_absdir", ngn.gallery04_absdir)
##      print fmt0(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
##  print fmt1(just1)[0:] % ("png_abspath", png_abspath)
##  png = png_abspath if 1 else None
##  if png is not None:
##      None if os.path.isdir(ngn.gallery04_absdir) else (
##          os.makedirs(ngn.gallery04_absdir))
##      plt.savefig(png)
##  else:
##      plt.show()
##  plt.close()

##  #== === === === === === === === === === === === === === === === === === ===

##  #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
##  if prt_:
##      print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
##  None if 1 else sys.exit()


def analyze_tow_ends_placements(pd_results_ends, dzdys_threshold,
pd_src_us_tow_present, pd_src_us_tow_diff, tow_diff_names, np_src_xs,
np_hasnan_zs, np_hasnan_zs_mask_nans):
    """
    Returns a Pandas DataFrame containing tow ends placement results.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    prt__ = False if 0 else True  # def print switch
    mult_str = '--- '
    def_str = 'analyze_tow_ends_placements'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #f 1 and prt:
    #   rw, cl = len(pd_src_us_tow_present), 13
    #   print fmt1(just1)[0:] % (
    #       "pd_src_us_tow_present.iloc[:%s,:%i]" % (rw, cl),
    #       pd_src_us_tow_present.iloc[:rw, :cl])
    #one if 1 else sys.exit()
    #
    #f 1 and prt:
    #   rw, cl = len(pd_src_us_tow_diff), 13
    #   print fmt1(just1)[0:] % ("pd_results_ends.iloc[:%s,:%i]" % (rw, cl),
    #       pd_results_ends.iloc[:rw, :cl])
    #one if 1 else sys.exit()

    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff", pd_src_us_tow_diff)
    #one if 1 else sys.exit()

    #f 1:
    #   image = np_hasnan_zs.copy()
    #   image[np.isnan(image)] = np.nanmin(np_hasnan_zs)
    #   imshow(
    #       image=image.T,
    #       cmap='gray' if 0 else 'jet',
    #       colorbar=True if 1 else False,
    #   )
    #   None if 0 else sys.exit()

    #f 1:
    #   # shows the regions of interest and not (left & right)
    #   np_src_xs_nanmin = np.nanmin(np_src_xs)
    #   np_src_xs_nanmax = np.nanmax(np_src_xs)
    #   np_src_xs_mask_lf = np_src_xs < ngn.tow_edge_xrefs[0]
    #   np_src_xs_mask_rt = np_src_xs > ngn.tow_edge_xrefs[-1]
    #   image = np_src_xs.copy()
    #   image[np_src_xs_mask_lf] = np_src_xs_nanmax
    #   image[np_src_xs_mask_rt] = np_src_xs_nanmin
    #   imshow(
    #       image=image.T,
    #       cmap='gray' if 0 else 'jet',
    #       colorbar=True if 1 else False,
    #   )
    #   None if 0 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #half_window_pts = 5
    half_window_pts = ngn.half_window_pts
    window_start = -half_window_pts
    window_stop = half_window_pts
    window_num = 2 * half_window_pts + 1
    window_idxs = (
        np.linspace(window_start, window_stop, window_num).astype(np.int))
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("half_window_pts", half_window_pts)
    #   print fmt0(just1)[1:] % ("window_idxs", window_idxs)
    #one if 1 else sys.exit()

    #zs_median_filter_size = 11
    zs_median_filter_size = ngn.tow_ends_analysis_zs_median_filter_size
    #zs_gaussian_filter_sigma = 1.2
    zs_gaussian_filter_sigma = ngn.tow_ends_analysis_zs_gaussian_filter_sigma
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("zs_median_filter_size",
    #       zs_median_filter_size)
    #   print fmt0(just1)[1:] % ("zs_gaussian_filter_sigma",
    #       zs_gaussian_filter_sigma)
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    #f 1 and prt:
    #   print
    #   print fmt0(just1)[1:] % ("ngn.tow_ids.shape", ngn.tow_ids.shape)
    #   print fmt0(just1)[1:] % ("ngn.tow_ids", ngn.tow_ids)
    #   #
    #   print fmt0(just1)[1:] % ("ngn.tow_center_xrefs.shape",
    #       ngn.tow_center_xrefs.shape)
    #   print fmt0(just1)[1:] % ("ngn.tow_center_xrefs",
    #       ngn.tow_center_xrefs)
    #   #
    #   print fmt0(just1)[1:] % ("ngn.tow_edge_xrefs.shape",
    #       ngn.tow_edge_xrefs.shape)
    #   print fmt0(just1)[1:] % ("ngn.tow_edge_xrefs",
    #       ngn.tow_edge_xrefs)

    roi_tow_ids = ngn.tow_ids[1:-1]
    roi_tow_center_xrefs = ngn.tow_center_xrefs[1:-1]
    roi_tow_edge_xrefs_lf = ngn.tow_edge_xrefs[:-1]
    roi_tow_edge_xrefs_rt = ngn.tow_edge_xrefs[1:]
    #f 1 and prt:
    #   print
    #   print fmt0(just1)[1:] % ("roi_tow_ids.shape", roi_tow_ids.shape)
    #   print fmt0(just1)[1:] % ("roi_tow_ids", roi_tow_ids)
    #   #:
    #   print fmt0(just1)[1:] % ("roi_tow_center_xrefs.shape",
    #       roi_tow_center_xrefs.shape)
    #   print fmt0(just1)[1:] % ("roi_tow_center_xrefs", roi_tow_center_xrefs)
    #   #
    #   print fmt0(just1)[1:] % ("roi_tow_edge_xrefs_lf.shape",
    #       roi_tow_edge_xrefs_lf.shape)
    #   print fmt0(just1)[1:] % ("roi_tow_edge_xrefs_lf",
    #       roi_tow_edge_xrefs_lf)
    #   #
    #   print fmt0(just1)[1:] % ("roi_tow_edge_xrefs_rt.shape",
    #       roi_tow_edge_xrefs_rt.shape)
    #   print fmt0(just1)[1:] % ("roi_tow_edge_xrefs_rt",
    #       roi_tow_edge_xrefs_rt)
    #
    #f 1 and prt:
    #   print fmt1(just0)[0:] % (
    #       "np.vstack((roi_tow_ids, roi_tow_center_xrefs, " +
    #       "roi_tow_edge_xrefs_lf, roi_tow_edge_xrefs_rt))",
    #       np.vstack((roi_tow_ids, roi_tow_center_xrefs,
    #       roi_tow_edge_xrefs_lf, roi_tow_edge_xrefs_rt)))
    #one if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("ngn.tow_ids", ngn.tow_ids)

    tow_ids_ascend = (np.diff(ngn.tow_ids) == 1).all()
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("ngn.tow_ids", ngn.tow_ids)
    #   #rint fmt0(just1)[1:] % ("(np.diff(ngn.tow_ids) == 1).all()",
    #   #   (np.diff(ngn.tow_ids) == 1).all())
    #   print fmt0(just1)[1:] % ("tow_ids_ascend", tow_ids_ascend)

    tow_diff_ints = np.array([
        np.int(s.replace('t', '').replace('d', '')) for s in tow_diff_names
    ])
    tow_diff_ints_ascend = (np.diff(tow_diff_ints) == 1).all()
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("tow_diff_names", tow_diff_names)
    #   print fmt0(just1)[1:] % ("tow_diff_ints", tow_diff_ints)
    #   #rint fmt0(just1)[1:] % ("(np.diff(tow_diff_ints) == 1).all()",
    #   #   (np.diff(tow_diff_ints) == 1).all())
    #   print fmt0(just1)[1:] % ("tow_diff_ints_ascend", tow_diff_ints_ascend)
    #f 1 and prt:
    #   print fmt0(just1)[1:] % ("not tow_ids_ascend is tow_diff_ints_ascend",
    #       not tow_ids_ascend is tow_diff_ints_ascend)

    if tow_ids_ascend is not tow_diff_ints_ascend:
        tow_diff_names = tow_diff_names[::-1]
    if 'LD90_LB65536_DF101p4-Located-20170217' in ngn.job_us_csv_abspath:
        # the reverse of what it "should" be ...
        tow_diff_names = tow_diff_names[::-1]
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("tow_diff_names", tow_diff_names)
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    np_src_ys = np.array([
        np.arange(np_src_xs.shape[0]), ] * np_src_xs.shape[1]).transpose()
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("np_src_ys.shape", np_src_ys.shape)
    #   print fmt0(just1)[1:] % ("np_src_xs.shape", np_src_xs.shape)
    #   print fmt0(just1)[1:] % ("np_hasnan_zs.shape", np_hasnan_zs.shape)
    #   print fmt0(just1)[1:] % ("np_hasnan_zs_mask_nans.shape",
    #       np_hasnan_zs_mask_nans.shape)

    np_src_xs_dzdy = 0.5 * (np_src_xs[1:, :] + np_src_xs[:-1, :])
    np_src_ys_midp = 0.5 * (np_src_ys[1:, :] + np_src_ys[:-1, :])
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("np_src_xs_dzdy.shape",
    #       np_src_xs_dzdy.shape)
    #   #rint fmt1(just1)[1:] % ("np_src_xs_dzdy[:9, :9]",
    #   #   np_src_xs_dzdy[:9, :9])
    #   #
    #   print fmt0(just1)[1:] % ("np_src_ys_midp.shape",
    #       np_src_ys_midp.shape)
    #   #rint fmt1(just1)[1:] % ("np_src_ys_midp[:9, :9]",
    #   #   np_src_ys_midp[:9, :9])
    #   #
    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    # this refers to dataset "CNRC20170216_scan_parallel_base_layer_part3"
    i_first = 0  # devt case, Tows all add
    #i_first = i_first if 0 else 1  # devt case, Tow 1 drop
    #i_first = i_first if 0 else 2  # devt case, Tow 2 drop
    #i_first = i_first if 0 else 3  # devt case, Tow 3 drop
    #i_first = i_first if 0 else 4  # devt case, Tow 4 drop
    #i_first = i_first if 0 else 5  # devt case, Tow 5 drop
    #i_first = i_first if 0 else 6  # devt case, Tow 6 drop
    #i_first = i_first if 0 else 7  # devt case, Tow 7 drop
    #i_first = i_first if 0 else 8  # devt case, Tow 8 drop
    #i_first = i_first if 0 else 9  # devt case, Tow 9 drop
    #i_first = i_first if 0 else 10  # devt case, Tow 10 drop
    #i_first = i_first if 0 else 11  # devt case, Tow 11 drop
    #i_first = i_first if 0 else 12  # devt case, Tow 12 drop
    #i_first = i_first if 0 else 13  # devt case, Tow 13 drop
    #i_first = i_first if 0 else 14  # devt case, Tow 14 drop
    #i_first = i_first if 0 else 15  # devt case, Tow 15 drop
    #i_first = i_first if 0 else 16  # devt case, Tow 16 drop

    for i, indy in enumerate(pd_src_us_tow_diff['meast_idx']):
        meast_id = indy + 1
        if i < i_first:
            continue
        start_mult = 50
        if 1 and prt:
            if i == 0:
                print
            #rint "\n%sanalyze this event %s" % ('::: ', '::: ' * start_mult)
            print fmt0(just1)[1:] % ("[i, indy]", [i, indy])

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # get the data for the current analysis window

        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "get the data for the current analysis window",
        #       "get the data for the current analysis window")

        pd_src_us_tow_diff_row_values = (
            pd_src_us_tow_diff.ix[i, tow_diff_names].values)
        #f 0 and prt:
        #   # note: this is ... pd_src_us_tow_diff_row...
        #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff.iloc[[i], :]",
        #       pd_src_us_tow_diff.iloc[[i], :])
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("tow_diff_names", tow_diff_names)
        #   print fmt0(just1)[1:] % ("pd_src_us_tow_diff_row_values",
        #       pd_src_us_tow_diff_row_values)
        #one if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        window_indy = window_idxs + indy
        window_xs = np_src_xs[window_indy, :]
        window_ys = np_src_ys[window_indy, :]
        window_zs = np_hasnan_zs[window_indy, :]
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("window_idxs", window_idxs)
        #   print fmt0(just1)[1:] % ("indy", indy)
        #   print fmt0(just1)[1:] % ("window_indy", window_indy)
        #   #
        #   print fmt0(just1)[1:] % ("window_xs.shape", window_xs.shape)
        #   print fmt0(just1)[1:] % ("window_ys.shape", window_ys.shape)
        #   print fmt0(just1)[1:] % ("window_zs.shape",window_zs.shape)
        #one if 1 else sys.exit()

        #f 1 and prt:
        #   print fmt1(just1)[0:] % (
        #       "pd_src_us_tow_present.ix[window_indy, 'U-Sensor']",
        #       pd_src_us_tow_present.ix[window_indy, 'U-Sensor'])
        #one if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        window_xs_dzdy = np_src_xs_dzdy[window_indy[:-1], :]
        window_ys_midp = np_src_ys_midp[window_indy[:-1], :]
        #f 1 and prt:
        #   #rint
        #   print fmt0(just1)[1:] % ("window_xs_dzdy.shape",
        #       window_xs_dzdy.shape)
        #   #rint fmt1(just1)[1:] % ("window_xs[:, :9]", window_xs[:, :9])
        #   #rint fmt1(just1)[1:] % ("window_xs_dzdy[:, :9]",
        #   #   window_xs_dzdy[:, :9])
        #   #
        #   print fmt0(just1)[1:] % ("window_ys_midp.shape",
        #       window_ys_midp.shape)
        #   #rint fmt1(just1)[1:] % ("window_ys[:, :9]", window_ys[:, :9])
        #   #rint fmt1(just1)[1:] % ("window_ys_midp[:, :9]",
        #   #   window_ys_midp[:, :9])
        #one if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # filter each laser measurement profile (along the axis=0 direction)

        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "filter each laser measurement profile",
        #       "filter each laser measurement profile")

        window_zs_mask_nnans = ~ np_hasnan_zs_mask_nans[window_indy, :]
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("window_zs_mask_nnans.shape",
        #       window_zs_mask_nnans.shape)

        window_zs_fltr0 = window_zs.copy()
        for win_row_idx in xrange(len(window_idxs)):
            nnan_zs = window_zs[
                win_row_idx, window_zs_mask_nnans[win_row_idx, :]]
            fltr_zs = ndi.gaussian_filter(
                ndi.median_filter(nnan_zs, zs_median_filter_size),
                zs_gaussian_filter_sigma)
            window_zs_fltr0[
                win_row_idx, window_zs_mask_nnans[win_row_idx, :]] = fltr_zs
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("window_zs_fltr0.shape",
        #       window_zs_fltr0.shape)
        #one if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # detrend all laser measurement profiles (along the axis=0 direction)

        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "detrend all laser measurement profiles",
        #       "detrend all laser measurement profiles")

        nnan_zs_mask = window_zs_mask_nnans.ravel()
        nnan_xs = window_xs.ravel()[nnan_zs_mask]
        nnan_zs_fltr = window_zs_fltr0.ravel()[nnan_zs_mask]
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("nnan_zs_mask.shape",
        #       nnan_zs_mask.shape)
        #   print fmt0(just1)[1:] % ("nnan_xs.shape", nnan_xs.shape)
        #   print fmt0(just1)[1:] % ("nnan_zs_fltr.shape", nnan_zs_fltr.shape)

        m0, c0 = lsbf_line_slope_and_intercept(nnan_xs, nnan_zs_fltr)
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("[m0, c0]", [m0, c0])

        window_zs_lsbf0 = m0 * window_xs + c0
        window_zs_fltr0_ntrend0 = window_zs_fltr0 - window_zs_lsbf0
        #f 1 and prt:
        #   print fmt0(just1)[1:] % ("window_xs.shape", window_xs.shape)
        #   print fmt0(just1)[1:] % ("window_zs_lsbf0.shape",
        #       window_zs_lsbf0.shape)
        #   print fmt0(just1)[1:] % ("window_zs_fltr0.shape",
        #       window_zs_fltr0.shape)
        #   print fmt0(just1)[1:] % ("window_zs_fltr0_ntrend0.shape",
        #       window_zs_fltr0_ntrend0.shape)
        #one if 1 else sys.exit()

        #f 1:
        #   imshow(
        #       image=window_zs_fltr0_ntrend0 if 1 else window_zs_ntrend0,
        #       cmap='gray' if 0 else 'jet',
        #       colorbar=True if 1 else False,
        #   )
        #   None if 0 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # calculate first differences for each z versus y profile (along the
        # axis=1 direction) in the measurements window.
        #
        # note:  These calculation are applied only z versys y profiles
        #        having no NaN values. Otherwise, all z values for the profiles
        #        (with at least one NaN value) are set to np.nan.
        #
        # note: All y-intervals between points are assumed equal.

        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "calculate first differences (dzdy)", "")

        window_dzdys_fltr0_ntrend0 = (
            np.full(window_zs_fltr0_ntrend0[1:, :].shape, np.nan))
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("window_zs_fltr0_ntrend0.shape",
        #       window_zs_fltr0_ntrend0.shape)
        #   print fmt0(just1)[1:] % ("window_dzdys_fltr0_ntrend0.shape",
        #       window_dzdys_fltr0_ntrend0.shape)

        for win_col_idx in xrange(window_zs_fltr0_ntrend0.shape[1]):
            #f 1 and prt:
            #   print fmt0(just1)[1:] % ("win_col_idx", win_col_idx)

            if np.isnan(window_zs_fltr0_ntrend0[:, win_col_idx]).any():
                window_zs_fltr0_ntrend0[:, win_col_idx] = np.nan
                #f 1 and prt:
                #   #rint fmt0(just1)[1:] % (
                #   #   "(init) window_zs_fltr0_ntrend0[:, win_col_idx]",
                #   #   window_zs_fltr0_ntrend0[:, win_col_idx])
                #   print fmt0(just1)[1:] % (
                #       "", " ... not all values ... (has at least one NaN)")
                #   print fmt0(just1)[1:] % (
                #       "(updt) window_zs_fltr0_ntrend0[:, win_col_idx]",
                #       window_zs_fltr0_ntrend0[:, win_col_idx])
                #one if 0 else sys.exit()
            else:
                window_dzdys_fltr0_ntrend0[:, win_col_idx] = (
                    window_zs_fltr0_ntrend0[1:, win_col_idx] -
                    window_zs_fltr0_ntrend0[:-1, win_col_idx])
                #f 1 and prt:
                #   print fmt0(just1)[1:] % (
                #       "window_zs_fltr0_ntrend0[:, win_col_idx]",
                #       window_zs_fltr0_ntrend0[:, win_col_idx])
                #   print fmt0(just1)[1:] % (
                #       "", " ... has all values ... (has no NaN values)")
                #   print fmt0(just1)[1:] % (
                #       "window_zs_fltr0_ntrend0[1:, win_col_idx]",
                #       window_zs_fltr0_ntrend0[1:, win_col_idx])
                #   print fmt0(just1)[1:] % (
                #       "window_zs_fltr0_ntrend0[:-1, win_col_idx]",
                #       window_zs_fltr0_ntrend0[:-1, win_col_idx])
                #   print fmt0(just1)[1:] % (
                #       "window_dzdys_fltr0_ntrend0[:, win_col_idx]",
                #       window_dzdys_fltr0_ntrend0[:, win_col_idx])
                #one if 0 else sys.exit()
            #f win_col_idx >= 20:
            #   break
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        #one if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # get the mask and data for dzdys and yz & zs corresponding to dzdys
        # large enough to be tow (strip) edges

        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "get the tow start edges & tow stop edges dzdy, ys & zs data",
        #       "get the tow start edges & tow stop edges dzdy, ys & zs data")

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        window_dzdys_fltr0_ntrend0_mask_edges_start = (
            window_dzdys_fltr0_ntrend0 >= dzdys_threshold)
        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "window_dzdys_fltr0_ntrend0_mask_edges_start.shape",
        #       window_dzdys_fltr0_ntrend0_mask_edges_start.shape)

        window_ys_fltr0_ntrend0_mask_edges_start = (
            np.full((window_ys.shape), False).astype(np.bool))
        window_ys_fltr0_ntrend0_mask_edges_start[:-1] = (
            window_ys_fltr0_ntrend0_mask_edges_start[:-1] |
            window_dzdys_fltr0_ntrend0_mask_edges_start)
        window_ys_fltr0_ntrend0_mask_edges_start[1:] = (
            window_ys_fltr0_ntrend0_mask_edges_start[1:] |
            window_dzdys_fltr0_ntrend0_mask_edges_start)
        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "window_ys_fltr0_ntrend0_mask_edges_start.shape",
        #       window_ys_fltr0_ntrend0_mask_edges_start.shape)
        #   #rint fmt1(just1)[1:] % (
        #   #   "window_ys_fltr0_ntrend0_mask_edges_start[:, 0: 21]",
        #   #   window_ys_fltr0_ntrend0_mask_edges_start[:, 0: 21])
        #   #rint fmt1(just1)[1:] % (
        #   #   "window_ys_fltr0_ntrend0_mask_edges_start[:, 510: 531]",
        #   #   window_ys_fltr0_ntrend0_mask_edges_start[:, 510: 531])

        window_ys_fltr0_ntrend0_edges_start = window_ys.copy().astype(np.float)
        window_ys_fltr0_ntrend0_edges_start[
            ~ window_ys_fltr0_ntrend0_mask_edges_start
        ] = np.nan
        #f 1 and prt:
        #   print fmt0(just1)[1:] % (
        #       "window_ys_fltr0_ntrend0_edges_start.shape",
        #       window_ys_fltr0_ntrend0_edges_start.shape)
        #   #rint fmt1(just1)[1:] % (
        #   #   "window_ys_fltr0_ntrend0_edges_start[:, 0: 21]",
        #   #   window_ys_fltr0_ntrend0_edges_start[:, 0: 21])
        #   #rint fmt1(just1)[1:] % (
        #   #   "window_ys_fltr0_ntrend0_edges_start[:, 510: 531]",
        #   #   window_ys_fltr0_ntrend0_edges_start[:, 510: 531])

        #one if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        window_dzdys_fltr0_ntrend0_mask_edges_stop = (
            window_dzdys_fltr0_ntrend0 <= -dzdys_threshold)
        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "window_dzdys_fltr0_ntrend0_mask_edges_stop.shape",
        #       window_dzdys_fltr0_ntrend0_mask_edges_stop.shape)

        window_ys_fltr0_ntrend0_mask_edges_stop = (
            np.full((window_ys.shape), False).astype(np.bool))
        window_ys_fltr0_ntrend0_mask_edges_stop[:-1] = (
            window_ys_fltr0_ntrend0_mask_edges_stop[:-1] |
            window_dzdys_fltr0_ntrend0_mask_edges_stop)
        window_ys_fltr0_ntrend0_mask_edges_stop[1:] = (
            window_ys_fltr0_ntrend0_mask_edges_stop[1:] |
            window_dzdys_fltr0_ntrend0_mask_edges_stop)
        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "window_ys_fltr0_ntrend0_mask_edges_stop.shape",
        #       window_ys_fltr0_ntrend0_mask_edges_stop.shape)
        #   #rint fmt1(just1)[1:] % (
        #   #   "window_ys_fltr0_ntrend0_mask_edges_stop[:, 0: 21]",
        #   #   window_ys_fltr0_ntrend0_mask_edges_stop[:, 0: 21])
        #   #rint fmt1(just1)[1:] % (
        #   #   "window_ys_fltr0_ntrend0_mask_edges_stop[:, 510: 531]",
        #   #   window_ys_fltr0_ntrend0_mask_edges_stop[:, 510: 531])

        window_ys_fltr0_ntrend0_edges_stop = window_ys.copy().astype(np.float)
        window_ys_fltr0_ntrend0_edges_stop[
            ~ window_ys_fltr0_ntrend0_mask_edges_stop
        ] = np.nan
        #f 1 and prt:
        #   print fmt0(just1)[1:] % (
        #       "window_ys_fltr0_ntrend0_edges_stop.shape",
        #       window_ys_fltr0_ntrend0_edges_stop.shape)
        #   #rint fmt1(just1)[1:] % (
        #   #   "window_ys_fltr0_ntrend0_edges_stop[:, 0: 21]",
        #   #   window_ys_fltr0_ntrend0_edges_stop[:, 0: 21])
        #   #rint fmt1(just1)[1:] % (
        #   #   "window_ys_fltr0_ntrend0_edges_stop[:, 510: 531]",
        #   #   window_ys_fltr0_ntrend0_edges_stop[:, 510: 531])

        #one if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # get the mask(s) for tow (strip) edges only the region(s) of interest

        #f 1 and prt:
        #   print fmt0(just1)[0:] % (
        #       "get the mask(s) for tow (strip) regions of interest", "")

        window_xs_dzdy_mean0 = window_xs_dzdy.mean(axis=0)
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("window_xs.shape",
        #       window_xs.shape)
        #   #rint fmt1(just1)[1:] % ("window_xs[:, :9]", window_xs[:, :9])
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("window_xs_dzdy.shape",
        #       window_xs_dzdy.shape)
        #   #rint fmt1(just1)[1:] % ("window_xs_dzdy[:, :9]",
        #   #   window_xs_dzdy[:, :9])
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("window_xs_dzdy_mean0.shape",
        #       window_xs_dzdy_mean0.shape)
        #   #rint fmt1(just1)[1:] % ("window_xs_dzdy_mean0[:9]",
        #   #   window_xs_dzdy_mean0[:9])
        #one if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        if ngn.write_to_results_dir:
            # this is for plotting
            rois_mask = np.full(
                window_xs_dzdy_mean0.shape, False).astype(np.bool)
            #f 1 and prt:
            #   print fmt0(just2)[0:] % (
            #       "(init) [rois_mask .dtype, .shape, .sum()]",
            #       [rois_mask.dtype, rois_mask.shape, rois_mask.sum()])
            #one if 1 else sys.exit()

            # this is for plotting
            roi_tow_xs_center = []
            roi_tow_ys_mean_meast_idx = []
            roi_tow_starts_stops = []

        for roi_idx, roi_tow_id in enumerate(roi_tow_ids):

            pd_src_us_tow_diff_row_val = (
                pd_src_us_tow_diff_row_values[roi_idx])

            if pd_src_us_tow_diff_row_val == 0:
                # this is not a region of interest in the field of view
                #f 1 and prt:
                #   print fmt0(just2)[1:] % (
                #       "this is not a region of interest (in the FOV)",
                #       "continue")
                continue

            tow_diff_name = tow_diff_names[roi_idx]
            #f 1 and prt:
            #   #rint "\n%s" % ('KKK ' * 40)
            #   print fmt0(just2)[0:] % ("[roi_idx, roi_tow_id, " +
            #       "tow_diff_name, pd_src_us_tow_diff_row_val]",
            #       [roi_idx, roi_tow_id, tow_diff_name,
            #       pd_src_us_tow_diff_row_val])
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            roi_tow_center_xref = roi_tow_center_xrefs[roi_idx]
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("roi_tow_center_xref",
            #       roi_tow_center_xref)
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            roi_tow_edge_xref_lf = roi_tow_edge_xrefs_lf[roi_idx]
            roi_tow_edge_xref_rt = roi_tow_edge_xrefs_rt[roi_idx]
            #f 1 and prt:
            #   print fmt0(just2)[1:] % (
            #       "[roi_tow_edge_xref_lf, roi_tow_edge_xref_rt]",
            #       np.array([roi_tow_edge_xref_lf, roi_tow_edge_xref_rt]))
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            roi_mask = ((window_xs_dzdy_mean0 >= roi_tow_edge_xref_lf) &
                (window_xs_dzdy_mean0 <= roi_tow_edge_xref_rt))
            window_xs_roi_mask = np.array([roi_mask, ] * window_xs.shape[0])
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

            if pd_src_us_tow_diff_row_val == 1:
                window_ys_fltr0_ntrend0_edges_start_roi = (
                    window_ys_fltr0_ntrend0_edges_start.copy())
                window_ys_fltr0_ntrend0_edges_start_roi[
                    ~ window_xs_roi_mask] = np.nan
                window_ys_fltr0_ntrend0_edges_start_roi_mean = (
                    np.nanmean(window_ys_fltr0_ntrend0_edges_start_roi))
                tow_diff_num_i_value = (
                    window_ys_fltr0_ntrend0_edges_start_roi_mean)
                #
                window_ys_fltr0_ntrend0_mask_edges = (
                    window_ys_fltr0_ntrend0_mask_edges_start)

            if pd_src_us_tow_diff_row_val == -1:
                window_ys_fltr0_ntrend0_edges_stop_roi = (
                    window_ys_fltr0_ntrend0_edges_stop.copy())
                window_ys_fltr0_ntrend0_edges_stop_roi[
                    ~ window_xs_roi_mask] = np.nan
                window_ys_fltr0_ntrend0_edges_stop_roi_mean = (
                    np.nanmean(window_ys_fltr0_ntrend0_edges_stop_roi))
                tow_diff_num_i_value = (
                    window_ys_fltr0_ntrend0_edges_stop_roi_mean)
                #
                window_ys_fltr0_ntrend0_mask_edges = (
                    window_ys_fltr0_ntrend0_mask_edges_stop)

            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

            tow_diff_num_i_value_us = np.interp(tow_diff_num_i_value,
                window_indy, pd_src_us_tow_present.ix[window_indy, 'U-Sensor'],
                left=np.nan, right=np.nan)
            #f 1 and prt:
            #   print fmt0(just1)[0:] % ("tow_diff_num_i_value",
            #       tow_diff_num_i_value)
            #f 1 and prt:
            #   print fmt0(just1)[0:] % ("tow_diff_num_i_value_us",
            #       tow_diff_num_i_value_us)

            if ngn.write_to_results_dir:
                # this is for plotting
                rois_mask = (rois_mask | roi_mask)
                roi_tow_xs_center.append(roi_tow_center_xref)
                roi_tow_ys_mean_meast_idx.append(tow_diff_num_i_value)
                roi_tow_starts_stops.append(pd_src_us_tow_diff_row_val)

            #f 0 and prt:
            #   rw, cl = len(pd_src_us_tow_diff), 13
            #   print fmt1(just1)[0:] % ("pd_results_ends.iloc[:%s,:%i]" % (
            #       rw, cl), pd_results_ends.iloc[:rw, :cl])

            tow_diff_num = tow_diff_names[roi_idx].replace('d', '')
            tow_diff_num_i = tow_diff_num + 'i'
            tow_diff_num_us = tow_diff_num + 'us'
            tow_diff_num_xc = tow_diff_num + 'xc'

            tow_diff_cols = [tow_diff_num_i, tow_diff_num_us, tow_diff_num_xc]
            pd_results_ends.ix[[i], tow_diff_cols] = [
                tow_diff_num_i_value,
                tow_diff_num_i_value_us,
                roi_tow_center_xref,
            ]
            tow_diff_cols = ['ProfileID', 'MeastID',
                'TowPresentBits_Tow32toTow01', 'U-Sensor',
                tow_diff_num_i, tow_diff_num_us, tow_diff_num_xc]
            #f 1 and prt:
            #   print fmt1(just1)[0:] % (
            #       "pd_results_ends.ix[[i], tow_diff_cols]",
            #       pd_results_ends.ix[[i], tow_diff_cols])
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            #f roi_idx >= 1:
            #  break
            #one if 0 else sys.exit()

        #== === === === === === === === === === === === === === === === === ===

        if 1 and ngn.write_to_results_dir and ngn.make_gallery03_plots:
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

            number_of_profiles = np_hasnan_zs.shape[0]
            #f 1 and prt:
            #   print fmt0(just1)[0:] % ("number_of_profiles",
            #       number_of_profiles)

            cols = ['U-Sensor', 'MeastID'] + tow_diff_names
            pd_src_us_tow_diff_row = pd_src_us_tow_diff.ix[[i], cols]
            #f 1 and prt:
            #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff",
            #       pd_src_us_tow_diff)
            #f 1 and prt:
            #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff_row",
            #       pd_src_us_tow_diff_row)

            window_zs_ntrend0 = window_zs - window_zs_lsbf0
            #f 1 and prt:
            #   print fmt0(just1)[1:] % ("window_zs_ntrend0.shape",
            #       window_zs_ntrend0.shape)

            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

            window_xs_rois_mask = (
                np.array([rois_mask, ] * window_xs.shape[0]))
            #f 1 and prt:
            #   print fmt0(just1)[0:] % ("window_xs_rois_mask.shape",
            #       window_xs_rois_mask.shape)

            window_zs_fltr0_ntrend0_edges_start = (
                window_zs_fltr0_ntrend0.copy())
            window_zs_fltr0_ntrend0_edges_start[
                ~ window_ys_fltr0_ntrend0_mask_edges_start
            ] = np.nan
            #f 1 and prt:
            #   print fmt0(just1)[1:] % (
            #       "window_zs_fltr0_ntrend0_edges_start.shape",
            #       window_zs_fltr0_ntrend0_edges_start.shape)

            window_zs_fltr0_ntrend0_edges_start_rois = (
                window_zs_fltr0_ntrend0_edges_start.copy())
            window_zs_fltr0_ntrend0_edges_start_rois[
                ~ window_xs_rois_mask] = np.nan

            window_zs_fltr0_ntrend0_edges_stop = window_zs_fltr0_ntrend0.copy()
            window_zs_fltr0_ntrend0_edges_stop[
                ~ window_ys_fltr0_ntrend0_mask_edges_stop
            ] = np.nan
            #f 1 and prt:
            #   print fmt0(just1)[1:] % (
            #       "window_zs_fltr0_ntrend0_edges_stop.shape",
            #       window_zs_fltr0_ntrend0_edges_stop.shape)

            window_zs_fltr0_ntrend0_edges_stop_rois = (
                window_zs_fltr0_ntrend0_edges_stop.copy())
            window_zs_fltr0_ntrend0_edges_stop_rois[
                ~ window_xs_rois_mask] = np.nan

            #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            np_roi_tow_xs_center = np.array(roi_tow_xs_center)
            np_roi_tow_ys_mean_meast_idx = np.array(roi_tow_ys_mean_meast_idx)
            np_roi_tow_starts_stops = np.array(roi_tow_starts_stops)

            make_gallery03_tow_ends_placement_plot_xz(
                number_of_profiles, indy, window_indy,
                pd_src_us_tow_diff_row, tow_diff_names, window_xs, window_ys,
                window_zs_ntrend0, window_zs_fltr0_ntrend0,
                window_zs_fltr0_ntrend0_edges_start,
                window_ys_fltr0_ntrend0_edges_start,
                window_zs_fltr0_ntrend0_edges_stop,
                window_ys_fltr0_ntrend0_edges_stop,
                window_zs_fltr0_ntrend0_edges_stop_rois,
                window_zs_fltr0_ntrend0_edges_start_rois,
                np_roi_tow_xs_center, np_roi_tow_ys_mean_meast_idx,
                np_roi_tow_starts_stops,
            ) if 1 else None
            #one if 0 else sys.exit()

            #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            window_dzdys_fltr0_ntrend0_mask_ge0 = (
                window_dzdys_fltr0_ntrend0 > 0.)
            window_dzdys_fltr0_ntrend0_ge0 = window_dzdys_fltr0_ntrend0.copy()
            window_dzdys_fltr0_ntrend0_ge0[
                ~ window_dzdys_fltr0_ntrend0_mask_ge0] = 0.
            #f 1 and prt:
            #   print fmt0(just1)[0:] % (
            #       "window_dzdys_fltr0_ntrend0_ge0.shape",
            #       window_dzdys_fltr0_ntrend0_ge0.shape)

            # window_dzdys_fltr0_ntrend0_mask_edges_start
            window_dzdys_fltr0_ntrend0_edges_start = (
                window_dzdys_fltr0_ntrend0.copy())
            window_dzdys_fltr0_ntrend0_edges_start[
                ~ window_dzdys_fltr0_ntrend0_mask_edges_start
            ] = np.nan
            #f 1 and prt:
            #   print fmt0(just1)[0:] % (
            #       "window_dzdys_fltr0_ntrend0_edges_start.shape",
            #       window_dzdys_fltr0_ntrend0_edges_start.shape)
            #f 0 and prt:
            #   print fmt1(just1)[1:] % (
            #       "window_dzdys_fltr0_ntrend0_edges_start[:, 0: 21]",
            #       window_dzdys_fltr0_ntrend0_edges_start[:, 0: 21])
            #   print fmt1(just1)[1:] % (
            #       "window_dzdys_fltr0_ntrend0_edges_start[:, 500: 541]",
            #       window_dzdys_fltr0_ntrend0_edges_start[:, 510: 531])

            make_gallery03_tow_ends_placement_plot_yz_start(
                number_of_profiles, indy, window_indy,
                pd_src_us_tow_diff_row, tow_diff_names, window_ys,
                window_zs_fltr0_ntrend0, window_ys_midp,
                window_dzdys_fltr0_ntrend0, window_dzdys_fltr0_ntrend0_ge0,
                window_dzdys_fltr0_ntrend0_edges_start,
                window_zs_fltr0_ntrend0_edges_start, dzdys_threshold,
            ) if 1 else None
            #one if 0 else sys.exit()

            #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

            window_dzdys_fltr0_ntrend0_mask_le0 = (
                window_dzdys_fltr0_ntrend0 < 0.)
            window_dzdys_fltr0_ntrend0_le0 = window_dzdys_fltr0_ntrend0.copy()
            window_dzdys_fltr0_ntrend0_le0[
                ~ window_dzdys_fltr0_ntrend0_mask_le0] = 0.
            #f 1 and prt:
            #   print fmt0(just1)[0:] % (
            #       "window_dzdys_fltr0_ntrend0_le0.shape",
            #       window_dzdys_fltr0_ntrend0_le0.shape)

            # window_dzdys_fltr0_ntrend0_mask_edges_stop
            window_dzdys_fltr0_ntrend0_edges_stop = (
                window_dzdys_fltr0_ntrend0.copy())
            window_dzdys_fltr0_ntrend0_edges_stop[
                ~ window_dzdys_fltr0_ntrend0_mask_edges_stop
            ] = np.nan
            #f 1 and prt:
            #   print fmt0(just1)[0:] % (
            #       "window_dzdys_fltr0_ntrend0_edges_stop.shape",
            #       window_dzdys_fltr0_ntrend0_edges_stop.shape)
            #f 0 and prt:
            #   print fmt1(just1)[1:] % (
            #       "window_dzdys_fltr0_ntrend0_edges_stop[:, 0: 21]",
            #       window_dzdys_fltr0_ntrend0_edges_stop[:, 0: 21])
            #   print fmt1(just1)[1:] % (
            #       "window_dzdys_fltr0_ntrend0_edges_stop[:, 500: 541]",
            #       window_dzdys_fltr0_ntrend0_edges_stop[:, 510: 531])

            make_gallery03_tow_ends_placement_plot_yz_stop(
                number_of_profiles, indy, window_indy,
                pd_src_us_tow_diff_row, tow_diff_names, window_ys,
                window_zs_fltr0_ntrend0, window_ys_midp,
                window_dzdys_fltr0_ntrend0, window_dzdys_fltr0_ntrend0_le0,
                window_dzdys_fltr0_ntrend0_edges_stop,
                window_zs_fltr0_ntrend0_edges_stop, dzdys_threshold,
            ) if 1 else None
            None if 1 else sys.exit()

            #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        None if 1 else sys.exit()
        #break
        #f meast_id >= 286:
        #   break

    #f 1 and prt:
    #   rw, cl = len(pd_src_us_tow_diff), 21
    #   print fmt1(just1)[0:] % ("pd_results_ends.iloc[:%s,:%i]" % (
    #       rw, cl), pd_results_ends.iloc[:rw, :cl])

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return pd_results_ends


#zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz
#zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz
# (below) defs ... for analyzing tow ends placement2

#
### ### ### ###
#


def analyze_tow_ends_placements2(pd_results_ends, dzdys_threshold,
pd_src_us_tow_present, pd_src_us_tow_diff, tow_diff_names, np_src_xs,
np_hasnan_zs, np_hasnan_zs_mask_nans):
    """
    Returns a Pandas DataFrame containing tow ends placement results.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    prt__ = False if 0 else True  # def print switch
    mult_str = '--- '
    def_str = 'analyze_tow_ends_placements2'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    np_src_ys = np.array([
        np.arange(np_src_xs.shape[0]), ] * np_src_xs.shape[1]).transpose()
    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs.shape", np_src_xs.shape)
        print fmt0(just1)[1:] % ("np_hasnan_zs.shape", np_hasnan_zs.shape)
        print fmt0(just1)[1:] % ("np_hasnan_zs_mask_nans.shape",
            np_hasnan_zs_mask_nans.shape)
        print fmt0(just1)[1:] % ("np_src_ys.shape", np_src_ys.shape)

    if 0:
        image = np_hasnan_zs.copy()
        image[np.isnan(image)] = np.nanmin(np_hasnan_zs)
        imshow(
            image=image.T,
            cmap='gray' if 0 else 'jet',
            colorbar=True if 1 else False,
        )
        None if 0 else sys.exit()

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if 1 and prt:
        print fmt0(just1)[0:] % ('pd_src_us_tow_diff.shape',
            pd_src_us_tow_diff.shape)
    if 0 and prt:
        rw, cl = len(pd_src_us_tow_diff) if 0 else 4, 13
        print fmt1(just1)[1:] % ("pd_results_ends.iloc[:%s,:%i]" % (rw, cl),
            pd_results_ends.iloc[:rw, :cl])
        None if 1 else sys.exit()

    meast_idxs = pd_src_us_tow_diff['meast_idx'].values
    if 1 or prt:
        print fmt0(just1)[0:] % ("meast_idxs.shape", meast_idxs.shape)
        print fmt1(just1)[1:] % ("meast_idxs", meast_idxs)

    #f 1 or prt:
    #   print fmt0(just1)[0:] % ('pd_src_us_tow_present.shape',
    #       pd_src_us_tow_present.shape)
    #f 0 and prt:
    #   rw, cl = len(pd_src_us_tow_present) if 0 else 4, 13
    #   print fmt1(just1)[1:] % (
    #       "pd_src_us_tow_present.iloc[:%s,:%i]" % (rw, cl),
    #       pd_src_us_tow_present.iloc[:rw, :cl])
    #   None if 1 else sys.exit()

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    half_window_pts = ngn.half_window_pts
    window_offset_pts = ngn.window_offset_pts
    if 1 or prt:
        print fmt0(just1)[0:] % ("ngn.half_window_pts", ngn.half_window_pts)
        print fmt0(just1)[1:] % ("ngn.window_offset_pts",
            ngn.window_offset_pts)

    window_start = -half_window_pts
    window_stop = half_window_pts
    window_num = 2 * half_window_pts + 1
    window_idxs = (
        np.linspace(window_start, window_stop, window_num).astype(np.int))
    if 1 or prt:
        print fmt0(just1)[1:] % ("(init) window_idxs", window_idxs)

    window_idxs += window_offset_pts
    if 1 or prt:
        print fmt0(just1)[1:] % ("(ofst) window_idxs", window_idxs)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    zs_median_filter_size = ngn.tow_ends_analysis_zs_median_filter_size
    zs_gaussian_filter_sigma = ngn.tow_ends_analysis_zs_gaussian_filter_sigma
    if 1 or prt:
        print fmt0(just1)[0:] % ("zs_median_filter_size",
            zs_median_filter_size)
        print fmt0(just1)[1:] % ("zs_gaussian_filter_sigma",
            zs_gaussian_filter_sigma)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if 0 and prt:
        print fmt0(just1)[0:] % ('ngn.tow_ids.shape', ngn.tow_ids.shape)
        print fmt1(just1)[1:] % ('ngn.tow_ids', ngn.tow_ids)

    tow_ids_within_course = ngn.tow_ids[1:-1]
    if 1 or prt:
        print fmt0(just1)[0:] % ('tow_ids_within_course.shape',
            tow_ids_within_course.shape)
        print fmt1(just1)[1:] % ('tow_ids_within_course',
            tow_ids_within_course)

    if 0 and prt:
        print fmt0(just1)[0:] % ('np.array(ngn.tow_edge_xref_idxs).shape',
            np.array(ngn.tow_edge_xref_idxs).shape)
        print fmt1(just1)[1:] % ('ngn.tow_edge_xref_idxs',
            ngn.tow_edge_xref_idxs)

    tow_edge_xref_idxs0 = np.array(ngn.tow_edge_xref_idxs[:-1])
    tow_edge_xref_idxs1 = np.array(ngn.tow_edge_xref_idxs[1:])
    if 1 or prt:
        print fmt0(just1)[0:] % (
            '[tow_edge_xref_idxs0.shape, tow_edge_xref_idxs1.shape]',
            [tow_edge_xref_idxs0.shape, tow_edge_xref_idxs1.shape])
        print fmt1(just1)[1:] % ('tow_edge_xref_idxs0',
            tow_edge_xref_idxs0)
        print fmt1(just1)[1:] % ('tow_edge_xref_idxs1',
            tow_edge_xref_idxs1)

#   if 1 or prt:
#       print fmt0(just1)[0:] % ('ngn.tow_edge_xrefs.shape',
#           ngn.tow_edge_xrefs.shape)
#       print fmt1(just1)[1:] % ('ngn.tow_edge_xrefs', ngn.tow_edge_xrefs)
#
#   tow_edge_xrefs0 = np.array(ngn.tow_edge_xrefs[:-1])
#   tow_edge_xrefs1 = np.array(ngn.tow_edge_xrefs[1:])
#   if 1 or prt:
#       print fmt0(just1)[0:] % (
#           '[tow_edge_xrefs0.shape, tow_edge_xrefs1.shape]',
#           [tow_edge_xrefs0.shape, tow_edge_xrefs1.shape])
#       print fmt1(just1)[1:] % ('tow_edge_xrefs0', tow_edge_xrefs0)
#       print fmt1(just1)[1:] % ('tow_edge_xrefs1', tow_edge_xrefs1)

#   if 1 or prt:
#       print fmt0(just1)[0:] % ('ngn.tow_edge_ids.shape',
#           ngn.tow_edge_ids.shape)
#       print fmt1(just1)[1:] % ('ngn.tow_edge_ids', ngn.tow_edge_ids)
#
#   tow_edge_ids0 = np.array(ngn.tow_edge_ids[:-1])
#   tow_edge_ids1 = np.array(ngn.tow_edge_ids[1:])
#   if 1 or prt:
#       print fmt0(just1)[0:] % (
#           '[tow_edge_ids0.shape, tow_edge_ids1.shape]',
#           [tow_edge_ids0.shape, tow_edge_ids1.shape])
#       print fmt1(just1)[1:] % ('tow_edge_ids0', tow_edge_ids0)
#       print fmt1(just1)[1:] % ('tow_edge_ids1', tow_edge_ids1)

##  #f 1 and prt:
##  #   print fmt0(just1)[1:] % ("ngn.tow_center_xrefs.shape",
##  #       ngn.tow_center_xrefs.shape)
##  #   print fmt0(just1)[1:] % ("ngn.tow_center_xrefs",
##  #       ngn.tow_center_xrefs)
##
##  roi_tow_center_xrefs = ngn.tow_center_xrefs[1:-1]
##  #f 1 and prt:
##  #   print
##  #   print fmt0(just1)[1:] % ("roi_tow_center_xrefs.shape",
##  #       roi_tow_center_xrefs.shape)
##  #   print fmt0(just1)[1:] % ("roi_tow_center_xrefs", roi_tow_center_xrefs)
##  #
##  #f 1 and prt:
##  #   print fmt1(just0)[0:] % (
##  #       "np.vstack((roi_tow_ids, roi_tow_center_xrefs, " +
##  #       "roi_tow_edge_xrefs_lf, roi_tow_edge_xrefs_rt))",
##  #       np.vstack((roi_tow_ids, roi_tow_center_xrefs,
##  #       roi_tow_edge_xrefs_lf, roi_tow_edge_xrefs_rt)))
##  #one if 1 else sys.exit()

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    tow_diff_names_ascending = sorted(tow_diff_names[:])
    if 1 and prt:
        print fmt1(just1)[0:] % ("(init) tow_diff_names", tow_diff_names)
        print fmt1(just1)[1:] % ("(updt) tow_diff_names_ascending",
            tow_diff_names_ascending)

    pd_tow_diff_values_ascending = (
        pd_src_us_tow_diff.ix[:, tow_diff_names_ascending])
    if 1 and prt:
        print fmt0(just1)[0:] % ("pd_tow_diff_values_ascending.shape",
            pd_tow_diff_values_ascending.shape)
    if 1 and prt:
        #rint fmt1(just1)[1:] % ("pd_tow_diff_values_ascending",
        #   pd_tow_diff_values_ascending)
        print fmt1(just1)[1:] % ("pd_tow_diff_values_ascending.head()",
            pd_tow_diff_values_ascending.head())
        #rint fmt1(just1)[1:] % ("pd_tow_diff_values_ascending.tail()",
        #   pd_tow_diff_values_ascending.tail())

    np_tow_diff_values_ascending = pd_tow_diff_values_ascending.values
    if 1 or prt:
        print fmt0(just1)[0:] % ("np_tow_diff_values_ascending.shape",
            np_tow_diff_values_ascending.shape)
        print fmt1(just1)[1:] % ("np_tow_diff_values_ascending[:4, :]",
            np_tow_diff_values_ascending[:4, :])

    None if 1 else sys.exit()

##  #== === === === === === === === === === === === === === === === === === ===
##
##  np_src_xs_dzdy = 0.5 * (np_src_xs[1:, :] + np_src_xs[:-1, :])
##  np_src_ys_midp = 0.5 * (np_src_ys[1:, :] + np_src_ys[:-1, :])
##  #f 1 and prt:
##  #   print fmt0(just1)[0:] % ("np_src_xs_dzdy.shape",
##  #       np_src_xs_dzdy.shape)
##  #   #rint fmt1(just1)[1:] % ("np_src_xs_dzdy[:9, :9]",
##  #   #   np_src_xs_dzdy[:9, :9])
##  #   #
##  #   print fmt0(just1)[1:] % ("np_src_ys_midp.shape",
##  #       np_src_ys_midp.shape)
##  #   #rint fmt1(just1)[1:] % ("np_src_ys_midp[:9, :9]",
##  #   #   np_src_ys_midp[:9, :9])
##  #   #
##  #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    # this refers to dataset "CNRC20170216_scan_parallel_base_layer_part3"
    i_first = 0  # devt case, Tows all add
    #i_first = i_first if 0 else 1  # devt case, Tow 1 drop
    #i_first = i_first if 0 else 2  # devt case, Tow 2 drop
    #i_first = i_first if 0 else 3  # devt case, Tow 3 drop
    #i_first = i_first if 0 else 4  # devt case, Tow 4 drop
    #i_first = i_first if 0 else 5  # devt case, Tow 5 drop
    #i_first = i_first if 0 else 6  # devt case, Tow 6 drop
    #i_first = i_first if 0 else 7  # devt case, Tow 7 drop
    #i_first = i_first if 0 else 8  # devt case, Tow 8 drop
    #i_first = i_first if 0 else 9  # devt case, Tow 9 drop
    #i_first = i_first if 0 else 10  # devt case, Tow 10 drop
    #i_first = i_first if 0 else 11  # devt case, Tow 11 drop
    #i_first = i_first if 0 else 12  # devt case, Tow 12 drop
    #i_first = i_first if 0 else 13  # devt case, Tow 13 drop
    #i_first = i_first if 0 else 14  # devt case, Tow 14 drop
    #i_first = i_first if 0 else 15  # devt case, Tow 15 drop
    #i_first = i_first if 0 else 16  # devt case, Tow 16 drop

    i_first = 12

    start_mult = 50
    for i, indy in enumerate(meast_idxs):
        meast_id = indy + 1
        if i < i_first:
            continue
        if 1 or prt:
            if i == 0:
                print
            print "\n%sanalyze this event %s" % ('::: ', '::: ' * start_mult)
            print fmt0(just1)[1:] % ("[i, indy (== meast_idx), meast_id]",
                [i, indy, meast_id])

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
        # get current analysis window data

        window_indy = window_idxs + indy
        window_xs = np_src_xs[window_indy, :]
        window_ys = np_src_ys[window_indy, :]
        window_zs = np_hasnan_zs[window_indy, :]
        if 1 and prt:
            print fmt0(just1)[0:] % ("window_idxs", window_idxs)
            print fmt0(just1)[1:] % ("indy", indy)
            print fmt0(just1)[1:] % ("window_indy", window_indy)
            #
            print fmt0(just1)[1:] % ("window_xs.shape", window_xs.shape)
            print fmt0(just1)[1:] % ("window_ys.shape", window_ys.shape)
            print fmt0(just1)[1:] % ("window_zs.shape", window_zs.shape)

        None if 1 else sys.exit()

        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
        # filter each laser measurement profile (along the axis=0 direction)

        if 1 and prt:
            print fmt0(just1)[0:] % (
                "filter each laser measurement profile",
                "filter each laser measurement profile")

        window_zs_mask_nnans = ~ np_hasnan_zs_mask_nans[window_indy, :]
        if 1 and prt:
            print fmt0(just1)[0:] % ("window_zs_mask_nnans.shape",
                window_zs_mask_nnans.shape)

        window_zs_fltr0 = window_zs.copy()
        for win_row_idx in xrange(len(window_idxs)):
            nnan_zs = window_zs[
                win_row_idx, window_zs_mask_nnans[win_row_idx, :]]
            fltr_zs = ndi.gaussian_filter(
                ndi.median_filter(nnan_zs, zs_median_filter_size),
                zs_gaussian_filter_sigma)
            window_zs_fltr0[
                win_row_idx, window_zs_mask_nnans[win_row_idx, :]] = fltr_zs
        if 1 and prt:
            print fmt0(just1)[1:] % ("window_zs.shape", window_zs.shape)
            print fmt0(just1)[1:] % ("window_zs_fltr0.shape",
                window_zs_fltr0.shape)

        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        #indow_zs_flatten = window_zs_fltr0_ntrend0.copy().flatten()
        window_zs_flatten = window_zs_fltr0.copy().flatten()
        if 1 and prt:
            print fmt0(just1)[0:] % ("window_zs_flatten.shape",
                window_zs_flatten.shape)
            #rint fmt1(just1)[1:] % ("window_zs_flatten",
            #   window_zs_flatten)

        window_zs_flatten_nnan = (
            window_zs_flatten[~ np.isnan(window_zs_flatten)])
        if 1 and prt:
            #rint
            print fmt0(just1)[1:] % ("window_zs_flatten_nnan.shape",
                window_zs_flatten_nnan.shape)
            #rint fmt1(just1)[1:] % ("window_zs_flatten_nnan",
            #   window_zs_flatten_nnan)

        decimals = 2
        window_zs_flatten_nnan_rnd = (
            np.round(window_zs_flatten_nnan, decimals=decimals))
        if 1 and prt:
            #rint
            print fmt0(just1)[1:] % ("window_zs_flatten_nnan_rnd.shape",
                window_zs_flatten_nnan_rnd.shape)
            #rint fmt1(just1)[1:] % ("window_zs_flatten_nnan_rnd",
            #   window_zs_flatten_nnan_rnd)

        None if 1 else sys.exit()

        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

        window_zs_bins, window_zs_bincounts = (
            np.unique(window_zs_flatten_nnan_rnd, return_counts=True))
        if 0 and prt:
            print fmt0(just1)[0:] % ("window_zs_bins.shape",
                window_zs_bins.shape)
            print fmt1(just1)[1:] % ("(init) window_zs_bins",
                window_zs_bins)
        if 0 and prt:
            print fmt0(just1)[1:] % ("(init) window_zs_bincounts.shape",
                window_zs_bincounts.shape)
            print fmt1(just1)[1:] % ("window_zs_bincounts",
                window_zs_bincounts)

        # put bins in descending order with corresponding bincounts
        if np.diff(window_zs_bins).all() > 0.:
            window_zs_bins = window_zs_bins[::-1]
            window_zs_bincounts = window_zs_bincounts[::-1]
        if 1 and prt:
            print fmt0(just1)[0:] % ("window_zs_bins.shape",
                window_zs_bins.shape)
            print fmt1(just1)[1:] % ("(updt) window_zs_bins",
                window_zs_bins)
        if 1 and prt:
            print fmt0(just1)[1:] % ("(updt) window_zs_bincounts.shape",
                window_zs_bincounts.shape)
            print fmt1(just1)[1:] % ("window_zs_bincounts",
                window_zs_bincounts)

        window_zs_bincounts_fltrd = ndi.gaussian_filter(
            ndi.median_filter(
                window_zs_bincounts, size=5, mode='constant', cval=0.0
            ), sigma=1.2
        )
        if 1 and prt:
            print fmt0(just1)[1:] % ("window_zs_bincounts_fltrd.shape",
                window_zs_bincounts_fltrd.shape)
            print fmt1(just1)[1:] % ("window_zs_bincounts_fltrd",
                window_zs_bincounts_fltrd)

        None if 1 else sys.exit()

        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

        bincounts_uniqs = np.unique(window_zs_bincounts_fltrd)
        bincounts_uniqs = bincounts_uniqs[::-1]
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("bincounts_uniqs.shape",
        #       bincounts_uniqs.shape)
        #   print fmt1(just1)[1:] % ("bincounts_uniqs",
        #       bincounts_uniqs)

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

###     bincounts_max0 = bincounts_uniqs[0]
###     bincounts_max0_mask = window_zs_bincounts_fltrd == bincounts_max0
###     bincounts_max0_where = np.where(bincounts_max0_mask)[0]
###     bincounts_max0_z = np.mean(window_zs_bins[bincounts_max0_where])
###     if 1 and prt:
###         print fmt0(just1)[0:] % ("bincounts_max0", bincounts_max0)
###         print fmt0(just1)[1:] % ("bincounts_max0_mask.shape",
###             bincounts_max0_mask.shape)
###         print fmt0(just1)[1:] % ("bincounts_max0_where",
###             bincounts_max0_where)
###         print fmt0(just1)[1:] % ("bincounts_max0_z", bincounts_max0_z)

        None if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

###     sys_exit_flag = False

        print "\nzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
###     bincounts_2lbls_uniq = None
###     bincounts_2lbls_mask = None
###     bincounts_2lbls_labels = None
###     #
###     bincounts_3lbls_uniq = None
###     bincounts_3lbls_mask = None
###     bincounts_3lbls_labels = None
###
###     bincounts_max1_z = np.nan
###     bincounts_max2_z = np.nan
###     bincounts_max3_z = np.nan
###     bincounts_max4_z = np.nan

        def _my_func2(
            bincounts_max_mask,
            window_zs_bins,

        ):
            """
            """
            bincounts_max_where = np.where(bincounts_max_mask)[0]
            bincounts_max_z = np.mean(window_zs_bins[bincounts_max_where])
            return pd.Series({
                'bincounts_max_where': bincounts_max_where,
                'bincounts_max_z': bincounts_max_z,
            })

        print "==========================================================="
        bincounts_maxs = []
        bincounts_uniq_wheres = []
#       bincounts_max_masks = []
#       bincounts_max_zs = []
        for bincounts_uniq in bincounts_uniqs:
            #rint "-----------------------------------------------------------"
            if 0 and prt:
                print fmt0(just1)[1:] % ("bincounts_uniq", bincounts_uniq)

            ### ### ###
            # (below) devt: ... change ... window_zs_bincounts_fltrd ...
            if len(bincounts_maxs) == 3 and bincounts_uniq == 40:  # for devt
                window_zs_bincounts_fltrd[-5] = 40
                #f 1 and prt:
                #   print fmt1(just1)[1:] % ("window_zs_bincounts_fltrd",
                #       window_zs_bincounts_fltrd)
                #ys.exit()
            # (above) devt: ... change ... window_zs_bincounts_fltrd ...
            ### ### ###

            bincounts_fltrd_ge_uniq_mask = (
                window_zs_bincounts_fltrd >= bincounts_uniq)

            bincounts_fltrd_ge_bincounts_uniq_lbls, _ = (
                ndi.label(bincounts_fltrd_ge_uniq_mask))
            bincounts_fltrd_ge_bincounts_uniq_lbls_max = (
                np.max(bincounts_fltrd_ge_bincounts_uniq_lbls))
            #f 1 and prt:
            #   print fmt0(just1)[0:] % ("bincounts_fltrd_ge_uniq_mask.shape",
            #       bincounts_fltrd_ge_uniq_mask.shape)
            #   print fmt0(just1)[1:] % ("bincounts_fltrd_ge_uniq_mask",
            #       bincounts_fltrd_ge_uniq_mask.astype(np.int))
            #f 1 and prt:
            #   #rint fmt0(just1)[1:] % (
            #   #   "bincounts_fltrd_ge_bincounts_uniq_lbls.shape",
            #   #   bincounts_fltrd_ge_bincounts_uniq_lbls.shape)
            if 0 and prt:
                print fmt0(just1)[1:] % (
                    "bincounts_fltrd_ge_bincounts_uniq_lbls",
                    bincounts_fltrd_ge_bincounts_uniq_lbls)
            #f 1 and prt:
            #   print fmt0(just1)[1:] % (
            #       "bincounts_fltrd_ge_bincounts_uniq_lbls_max",
            #       bincounts_fltrd_ge_bincounts_uniq_lbls_max)

            ### ### ###
            # (below) devt: ... change ... window_zs_bincounts_fltrd ...
            #f len(bincounts_maxs) == 3 and bincounts_uniq == 40:  # for devt
            #   if 1 and prt:
            #       print fmt0(just1)[1:] % ("bincounts_maxs", bincounts_maxs)
            #   if 1 and prt:
            #       print fmt0(just1)[1:] % ("bincounts_uniq", bincounts_uniq)
            #   if 1 and prt:
            #       print fmt1(just1)[1:] % ("bincounts_fltrd_ge_uniq_mask",
            #           bincounts_fltrd_ge_uniq_mask.astype(np.int))
            #   if 1 and prt:
            #       print fmt1(just1)[1:] % (
            #           "bincounts_fltrd_ge_bincounts_uniq_lbls",
            #           bincounts_fltrd_ge_bincounts_uniq_lbls)
            #   sys.exit()
            # (above) devt: ... change ... window_zs_bincounts_fltrd ...
            ### ### ###

            #
            ###
            #
            ilast = len(bincounts_maxs)

            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            if (bincounts_fltrd_ge_bincounts_uniq_lbls_max == 1 and
            len(bincounts_maxs) == 0):
                print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                if 1 and prt:
                    print fmt0(just1)[1:] % ("ilast", ilast)

                #
                ### ### ###
                #
                bincounts_max = bincounts_uniq
                bincounts_maxs.append(bincounts_max)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "(max%i) bincounts_max (bincounts_uniq)" % ilast,
                        bincounts_max)
                    #rint fmt0(just1)[1:] % (
                    #   "(max%i) bincounts_maxs[-1]" % ilast,
                    #   bincounts_maxs[-1])
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_maxs",
                        bincounts_maxs)

                #
                ### ### ###
                #
                bincounts_max_where = (
                    np.where(window_zs_bincounts_fltrd == bincounts_max)[0])
                bincounts_uniq_wheres.append(bincounts_max_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "bincounts_max", bincounts_max)
                if 1 and prt:
                    #rint fmt0(just1)[0:] % ("window_zs_bincounts_fltrd.shape",
                    #   window_zs_bincounts_fltrd.shape)
                    print fmt0(just1)[1:] % ("window_zs_bincounts_fltrd",
                        window_zs_bincounts_fltrd)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max_where",
                        bincounts_max_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_uniq_wheres",
                        bincounts_uniq_wheres)

                print "*** " * 20
                ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

                if 1 and prt:
                    #rint fmt0(just1)[1:] % (
                    #   "bincounts_fltrd_ge_bincounts_uniq_lbls.shape",
                    #   bincounts_fltrd_ge_bincounts_uniq_lbls.shape)
                    print fmt0(just1)[1:] % (
                        "bincounts_fltrd_ge_bincounts_uniq_lbls",
                        bincounts_fltrd_ge_bincounts_uniq_lbls)

                bincounts_max0_lbl = (
                    bincounts_fltrd_ge_bincounts_uniq_lbls[
                        bincounts_uniq_wheres[0]]
                )[0]
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max0_lbl",
                        bincounts_max0_lbl)

                bincounts_max_lbls = np.array([
                    bincounts_max0_lbl,
                ])
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max_lbls",
                        bincounts_max_lbls)

                ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
                print "*** " * 20

#%          #   bincounts_uniq_where = (
#%          #       np.where(bincounts_uniqs == bincounts_uniq)[0])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq_where",
#%          #           bincounts_uniq_where)

#%

#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq", bincounts_max)
#%          #   if 1 and prt:
#%          #       #rint fmt0(just1)[0:] % ("bincounts_uniqs.shape",
#%          #       #   bincounts_uniqs.shape)
#%          #       print fmt0(just1)[1:] % ("bincounts_uniqs",
#%          #           bincounts_uniqs)

#%          #   bincounts_uniq_where = (
#%          #       np.where(bincounts_uniqs == bincounts_uniq)[0])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq_where",
#%          #           bincounts_uniq_where)

#%          #   bincounts_maxs.append(bincounts_max)
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % (
#%          #           "(max%i) bincounts_max (bincounts_uniq)" % ilast,
#%          #           bincounts_max)
#%          #       #rint fmt0(just1)[1:] % (
#%          #       #   "(max%i) bincounts_maxs[-1]" % ilast,
#%          #       #   bincounts_maxs[-1])
#%          #   if 1 and prt:
#%          #       #rint fmt0(just1)[0:] % ("bincounts_uniqs.shape",
#%          #       #   bincounts_uniqs.shape)
#%          #       print fmt0(just1)[1:] % ("bincounts_uniqs",
#%          #           bincounts_uniqs)

#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % ("bincounts_maxs", bincounts_maxs)
#%  #
#%  #           bincounts_max_lbls = np.arange(
#%  #               1, bincounts_fltrd_ge_bincounts_uniq_lbls_max + 1)
#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % ("bincounts_max_lbls",
#%  #                   bincounts_max_lbls)
#%  #
#%
#%  #
#%  #           print "*** " * 20
#%  #           ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#%  #
#%  #       #   if 1 and prt:
#%  #       #       #rint fmt0(just1)[0:] % ("bincounts_uniq_wheres",
#%  #       #       #   bincounts_uniq_wheres)
#%  #
#%  #           bincounts_uniq_where_lbls = np.array([0])
#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % (
#%  #                   "(devt) bincounts_uniq_where_lbls",
#%  #                   bincounts_uniq_where_lbls)
#%  #
#%  #           bincounts_max_lbls_in1d_not_where_lbls = np.in1d(
#%  #               bincounts_max_lbls, bincounts_uniq_where_lbls, invert=True)
#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % (
#%  #                   "bincounts_max_lbls_in1d_not_where_lbls",
#%  #                   bincounts_max_lbls_in1d_not_where_lbls)

#               bincounts_max_lbl = (
#                   np.where(bincounts_max_lbls[bincounts_max_lbls])[0])
#               if 1 and prt:
#                   print fmt0(just1)[1:] % ("bincounts_max_lbl",
#                       bincounts_max_lbl)

#%          #   bincounts_uniq_where_lbl = (
#%          #       np.where(bincounts_max_lbls[
#%          #           bincounts_max_lbls_in1d_not_where_lbls])[0])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq_where_lbl",
#%          #           bincounts_uniq_where_lbl)

#%

#%          #   bincounts_uniq_where_lbl = (
#%          #       np.where(bincounts_max_lbls[
#%          #           bincounts_max_lbls_in1d_not_where_lbls])[0])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq_where_lbl",
#%          #           bincounts_uniq_where_lbl)

#%

#%              ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#%          #   print "*** " * 20
#%

#%          #   bincounts_max_lbl0 = 0
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_max_lbl0",
#%          #           bincounts_max_lbl0)

#%          #   bincounts_uniq_where_lbls = np.array([bincounts_max_lbl0])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq_where_lbls",
#%          #           bincounts_uniq_where_lbls)

#%          #   bincounts_max_lbls_in1d_where_lbls = np.in1d(
#%          #       bincounts_max_lbls, bincounts_uniq_where_lbls, invert=True)
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % (
#%          #           "bincounts_max_lbls_in1d_where_lbls",
#%          #           bincounts_max_lbls_in1d_where_lbls)

#               print "*** " * 20
#
#               bincounts_max_mask = (
#                   window_zs_bincounts_fltrd == bincounts_uniq)
#               #
#               bincounts_max_masks.append(bincounts_max_mask)
#               if 1 and prt:
#                   print fmt0(just1)[0:] % (
#                       "len(bincounts_max_masks)", len(bincounts_max_masks))
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "(max%i) bincounts_max_masks[-1]" % ps1['ilast'],
#                       bincounts_max_masks[-1].astype(np.int))
#
#               print "*** " * 20

#%

#%          #   bincounts_max_lbl = (
#%          #       bincounts_max_lbls[bincounts_max_lbls_in1d_where_lbls][0])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_max_lbl",
#%          #           bincounts_max_lbl)

#%          #   bincounts_max_mask = (
#%          #       bincounts_fltrd_ge_bincounts_uniq_lbls ==
#%                  bincounts_max_lbl)
#%          #   bincounts_max_masks.append(bincounts_max_mask)
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % (
#%          #           "(max%i) bincounts_max_mask" % ps1['ilast'],
#%          #           bincounts_max_mask.astype(np.int))
#%                  #rint fmt0(just1)[1:] % (
#%                  #   "(max%i) bincounts_max_masks[-1]" % ps1['ilast'],
#%                  #   bincounts_max_masks[-1].astype(np.int))

#%              ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#%          #   print "*** " * 20
#%          #   print "*** " * 20

#%          #   ps2 = _my_func2(
#%          #       bincounts_max_mask,
#%          #       window_zs_bins,
#%          #
#%          #   )

#%          #   bincounts_uniq_wheres.append(ps2['bincounts_uniq_where'])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq_wheres",
#%          #           bincounts_uniq_wheres)
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % (
#%          #           "(max%i) bincounts_uniq_wheres[-1]" % ps1['ilast'],
#%          #           bincounts_uniq_wheres[-1])

#%          #   bincounts_max_zs.append(ps2['bincounts_max_z'])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_max_zs",
#%          #           np.array(bincounts_max_zs))
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % (
#%          #           "(max%i) bincounts_max_zs[-1]" % ps1['ilast'],
#%          #           bincounts_max_zs[-1])

                print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                None if 1 else sys.exit()
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            # bincounts_fltrd_ge_bincounts_uniq_lbls
            if (bincounts_fltrd_ge_bincounts_uniq_lbls_max == 2 and
            len(bincounts_maxs) == 1):
                print "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                if 1 and prt:
                    print fmt0(just1)[1:] % ("ilast", ilast)

                #
                ### ### ###
                #
                bincounts_max = bincounts_uniq
                bincounts_maxs.append(bincounts_max)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "(max%i) bincounts_max (bincounts_uniq)" % ilast,
                        bincounts_max)
                    #rint fmt0(just1)[1:] % (
                    #   "(max%i) bincounts_maxs[-1]" % ilast,
                    #   bincounts_maxs[-1])
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_maxs",
                        bincounts_maxs)

                #
                ### ### ###
                #
                bincounts_max_where = (
                    np.where(window_zs_bincounts_fltrd == bincounts_max)[0])
                bincounts_uniq_wheres.append(bincounts_max_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "bincounts_max", bincounts_max)
                if 1 and prt:
                    #rint fmt0(just1)[0:] % ("window_zs_bincounts_fltrd.shape",
                    #   window_zs_bincounts_fltrd.shape)
                    print fmt0(just1)[1:] % ("window_zs_bincounts_fltrd",
                        window_zs_bincounts_fltrd)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max_where",
                        bincounts_max_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_uniq_wheres",
                        bincounts_uniq_wheres)

                print "*** " * 20
                ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

                if 1 and prt:
                    #rint fmt0(just1)[1:] % (
                    #   "bincounts_fltrd_ge_bincounts_uniq_lbls.shape",
                    #   bincounts_fltrd_ge_bincounts_uniq_lbls.shape)
                    print fmt0(just1)[1:] % (
                        "bincounts_fltrd_ge_bincounts_uniq_lbls",
                        bincounts_fltrd_ge_bincounts_uniq_lbls)

                bincounts_max0_lbl = (
                    bincounts_fltrd_ge_bincounts_uniq_lbls[
                        bincounts_uniq_wheres[0]]
                )[0]
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max0_lbl",
                        bincounts_max0_lbl)

                bincounts_max1_lbl = (
                    bincounts_fltrd_ge_bincounts_uniq_lbls[
                        bincounts_uniq_wheres[1]]
                )[0]
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max1_lbl",
                        bincounts_max1_lbl)

                bincounts_max_lbls = np.array([
                    bincounts_max0_lbl,
                    bincounts_max1_lbl,
                ])
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max_lbls",
                        bincounts_max_lbls)

                ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
                print "*** " * 20

##              bincounts_maxs.append(bincounts_max)
##              if 1 and prt:
##                  print fmt0(just1)[1:] % ("last", ilast)
##              if 1 and prt:
##                  print fmt0(just1)[1:] % (
##                      "(max%i) bincounts_max (bincounts_uniq)" % ilast,
##                      bincounts_max)
##                  #rint fmt0(just1)[1:] % (
##                  #   "(max%i) bincounts_maxs[-1]" % ilast,
##                  #   bincounts_maxs[-1])

#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq", bincounts_max)
#%          #   if 1 and prt:
#%          #       #rint fmt0(just1)[0:] % ("bincounts_uniqs.shape",
#%          #       #   bincounts_uniqs.shape)
#%          #       print fmt0(just1)[1:] % ("bincounts_uniqs",
#%          #           bincounts_uniqs)

#%          #   bincounts_uniq_where = (
#%          #       np.where(bincounts_uniqs == bincounts_uniq)[0])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq_where",
#%          #           bincounts_uniq_where)

#%          #   bincounts_maxs.append(bincounts_max)
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % (
#%          #           "(max%i) bincounts_max (bincounts_uniq)" % ilast,
#%          #           bincounts_max)
#%          #       #rint fmt0(just1)[1:] % (
#%          #       #   "(max%i) bincounts_maxs[-1]" % ilast,
#%          #       #   bincounts_maxs[-1])
#%          #   if 1 and prt:
#%          #       #rint fmt0(just1)[0:] % ("bincounts_uniqs.shape",
#%          #       #   bincounts_uniqs.shape)
#%          #       print fmt0(just1)[1:] % ("bincounts_uniqs",
#%          #           bincounts_uniqs)

#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % ("bincounts_maxs", bincounts_maxs)
#%  #
#%  #           bincounts_max_lbls = np.arange(
#%  #               1, bincounts_fltrd_ge_bincounts_uniq_lbls_max + 1)
#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % ("bincounts_max_lbls",
#%  #                   bincounts_max_lbls)
#%  #
#%  #           if 1 and prt:
#%  #               #rint fmt0(just1)[1:] % (
#%  #               #   "bincounts_fltrd_ge_bincounts_uniq_lbls.shape",
#%  #               #   bincounts_fltrd_ge_bincounts_uniq_lbls.shape)
#%  #               print fmt0(just1)[1:] % (
#%  #                   "bincounts_fltrd_ge_bincounts_uniq_lbls",
#%  #                   bincounts_fltrd_ge_bincounts_uniq_lbls)
#%  #
#%  #
#%  #           print "*** " * 20
#%  #           ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

#%          ##  max0_idx = 0
#%          ##  if 1 and prt:
#%          ##      print fmt0(just1)[1:] % (
#%          ##          "(max%i) bincounts_uniq_wheres[%i]" %
#%          ##          (max0_idx, max0_idx), bincounts_uniq_wheres[max0_idx])
#%          ##  bincounts_max_lbl0 = bincounts_fltrd_ge_bincounts_uniq_lbls[
#%          ##      bincounts_uniq_wheres[max0_idx]]
#%          ##  if 1 and prt:
#%          ##      print fmt0(just1)[1:] % ("bincounts_max_lbl0",
#%          ##          bincounts_max_lbl0)

#%          ### ### ###

#%          #   if 1 and prt:
#%          #       #rint fmt0(just1)[0:] % ("bincounts_uniq_wheres",
#%          #       #   bincounts_uniq_wheres)

#               bincounts_uniq_where_lbls = np.array([0])
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "(devt) bincounts_uniq_where_lbls",
#                       bincounts_uniq_where_lbls)

#               bincounts_max_lbls_in1d_not_where_lbls = np.in1d(
#                   bincounts_max_lbls, bincounts_uniq_where_lbls, invert=True)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "bincounts_max_lbls_in1d_not_where_lbls",
#                       bincounts_max_lbls_in1d_not_where_lbls)

#%              ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#%          #   print "*** " * 20
#%          #   print "*** " * 20

##              bincounts_uniq_wheres.append(
##                  np.where(bincounts_max_masks[-1])[0])
##              if 1 and prt:
##                  print fmt0(just1)[1:] % (
##                      "bincounts_uniq_wheres", bincounts_uniq_wheres)
##                  print fmt0(just1)[1:] % (
##                      "(max%i) bincounts_uniq_wheres[-1]" % ps1['ilast'],
##                      bincounts_uniq_wheres[-1])
##
##              bincounts_max0_where = bincounts_uniq_wheres[-1]
##              bincounts_max0_lbl = bincounts_fltrd_ge_bincounts_uniq_lbls[
##                  bincounts_uniq_wheres[-1]][0]
##              bincounts_max_lbl = 2 if bincounts_max0_lbl == 1 else 1
##              #
##              bincounts_max_mask = (
##                  bincounts_fltrd_ge_bincounts_uniq_lbls ==
##                  bincounts_max_lbl)
##      #       if 1 and prt:
##      #           print fmt0(just1)[1:] % (
##      #               "len(bincounts_max_masks)", len(bincounts_max_masks))
##
##              bincounts_max_masks.append(bincounts_max_mask)
##              if 1 and prt:
##                  print fmt0(just1)[1:] % (
##                      "(max%i) bincounts_max_masks[-1]" % ps1['ilast'],
##                      bincounts_max_masks[-1].astype(np.int))
##
##                  np.where(bincounts_max_masks[-1])[0])
##              bincounts_max_zs.append(
##                  np.mean(window_zs_bins[bincounts_uniq_wheres[-1]]))
##              if 1 and prt:
##                  print fmt0(just1)[1:] % (
##                      "(max%i) bincounts_max_masks[-1].shape" % ilast,
##                      bincounts_max_masks[-1].shape)
##                  print fmt0(just1)[1:] % (
##                      "(max%i) bincounts_max_zs[-1]" % ilast,
##                      bincounts_max_zs[-1])
##
##              bincounts_max_lbls = np.arange(
##                  1, bincounts_fltrd_ge_bincounts_uniq_lbls_max + 1)
##              if 1 and prt:
##                  print fmt0(just1)[0:] % (
##                      "bincounts_max_lbls", bincounts_max_lbls)
                print
                print "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
                None if 1 else sys.exit()
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            if (bincounts_fltrd_ge_bincounts_uniq_lbls_max == 3 and
            len(bincounts_maxs) == 2):
                print "ccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                if 1 and prt:
                    print fmt0(just1)[1:] % ("ilast", ilast)

                #
                ### ### ###
                #
                bincounts_max = bincounts_uniq
                bincounts_maxs.append(bincounts_max)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "(max%i) bincounts_max (bincounts_uniq)" % ilast,
                        bincounts_max)
                    #rint fmt0(just1)[1:] % (
                    #   "(max%i) bincounts_maxs[-1]" % ilast,
                    #   bincounts_maxs[-1])
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_maxs",
                        bincounts_maxs)

                #
                ### ### ###
                #
                bincounts_max_where = (
                    np.where(window_zs_bincounts_fltrd == bincounts_max)[0])
                bincounts_uniq_wheres.append(bincounts_max_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "bincounts_max", bincounts_max)
                if 1 and prt:
                    #rint fmt0(just1)[0:] % ("window_zs_bincounts_fltrd.shape",
                    #   window_zs_bincounts_fltrd.shape)
                    print fmt0(just1)[1:] % ("window_zs_bincounts_fltrd",
                        window_zs_bincounts_fltrd)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max_where",
                        bincounts_max_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_uniq_wheres",
                        bincounts_uniq_wheres)

                print "*** " * 20
                ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

                if 1 and prt:
                    #rint fmt0(just1)[1:] % (
                    #   "bincounts_fltrd_ge_bincounts_uniq_lbls.shape",
                    #   bincounts_fltrd_ge_bincounts_uniq_lbls.shape)
                    print fmt0(just1)[1:] % (
                        "bincounts_fltrd_ge_bincounts_uniq_lbls",
                        bincounts_fltrd_ge_bincounts_uniq_lbls)

                bincounts_max0_lbl = (
                    bincounts_fltrd_ge_bincounts_uniq_lbls[
                        bincounts_uniq_wheres[0]]
                )[0]
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max0_lbl",
                        bincounts_max0_lbl)

                bincounts_max1_lbl = (
                    bincounts_fltrd_ge_bincounts_uniq_lbls[
                        bincounts_uniq_wheres[1]]
                )[0]
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max1_lbl",
                        bincounts_max1_lbl)

                bincounts_max2_lbl = (
                    bincounts_fltrd_ge_bincounts_uniq_lbls[
                        bincounts_uniq_wheres[2]]
                )[0]
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max2_lbl",
                        bincounts_max2_lbl)

                bincounts_max_lbls = np.array([
                    bincounts_max0_lbl,
                    bincounts_max1_lbl,
                    bincounts_max2_lbl,
                ])
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max_lbls",
                        bincounts_max_lbls)

                ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
                print "*** " * 20

##              bincounts_maxs.append(bincounts_max)
##              if 1 and prt:
##                  print fmt0(just1)[1:] % ("last", ilast)
##              if 1 and prt:
##                  print fmt0(just1)[1:] % (
##                      "(max%i) bincounts_max (bincounts_uniq)" % ilast,
##                      bincounts_max)
##                  #rint fmt0(just1)[1:] % (
##                  #   "(max%i) bincounts_maxs[-1]" % ilast,
##                  #   bincounts_maxs[-1])

#%      #       if 1 and prt:
#%      #           print fmt0(just1)[1:] % ("bincounts_uniq", bincounts_max)
#%      #       if 1 and prt:
#%      #           #rint fmt0(just1)[0:] % ("bincounts_uniqs.shape",
#%      #           #   bincounts_uniqs.shape)
#%      #           print fmt0(just1)[1:] % ("bincounts_uniqs",
#%      #               bincounts_uniqs)

#%          #   bincounts_uniq_where = (
#%          #       np.where(bincounts_uniqs == bincounts_uniq)[0])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq_where",
#%          #           bincounts_uniq_where)
#%          #
#%          #   bincounts_maxs.append(bincounts_max)
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % (
#%          #           "(max%i) bincounts_max (bincounts_uniq)" % ilast,
#%          #           bincounts_max)
#%          #       #rint fmt0(just1)[1:] % (
#%          #       #   "(max%i) bincounts_maxs[-1]" % ilast,
#%          #       #   bincounts_maxs[-1])
#%          #   if 1 and prt:
#%          #       #rint fmt0(just1)[0:] % ("bincounts_uniqs.shape",
#%          #       #   bincounts_uniqs.shape)
#%          #       print fmt0(just1)[1:] % ("bincounts_uniqs",
#%          #           bincounts_uniqs)

#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % ("bincounts_maxs", bincounts_maxs)
#%  #
#%  #           bincounts_max_lbls = np.arange(
#%  #               1, bincounts_fltrd_ge_bincounts_uniq_lbls_max + 1)
#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % ("bincounts_max_lbls",
#%  #                   bincounts_max_lbls)
#%  #
#%  #           if 1 and prt:
#%  #               #rint fmt0(just1)[1:] % (
#%  #               #   "bincounts_fltrd_ge_bincounts_uniq_lbls.shape",
#%  #               #   bincounts_fltrd_ge_bincounts_uniq_lbls.shape)
#%  #               print fmt0(just1)[1:] % (
#%  #                   "bincounts_fltrd_ge_bincounts_uniq_lbls",
#%  #                   bincounts_fltrd_ge_bincounts_uniq_lbls)
#%  #
#%  #
#%  #           print "*** " * 20
#%  #           ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

#               if 1 and prt:
#                   print fmt1(just1)[0:] % (
#                       "bincounts_fltrd_ge_bincounts_uniq_lbls",
#                       bincounts_fltrd_ge_bincounts_uniq_lbls)
#
#               bincounts_max_lbls = np.arange(
#                   1, bincounts_fltrd_ge_bincounts_uniq_lbls_max + 1)
#               if 1 and prt:
#                   print fmt0(just1)[0:] % ("bincounts_max_lbls",
#                       bincounts_max_lbls)
#
#               bincounts_uniq_where_lbls = []
#               for bincounts_uniq_where in bincounts_uniq_wheres:
#                   bincounts_uniq_where_lbls.append(
#                       bincounts_fltrd_ge_bincounts_uniq_lbls[
#                           bincounts_uniq_where][0])
#                   #f 1 and prt:
#                   #   print fmt0(just1)[0:] % (
#                   #       "... [bincounts_uniq_where, " +
#                   #       "bincounts_uniq_where_lbls[-1]]",
#                   #       [bincounts_uniq_where,
#                   #           bincounts_uniq_where_lbls[-1]])
#               if 1 and prt:
#                   print fmt0(just1)[0:] % ("bincounts_uniq_wheres",
#                       bincounts_uniq_wheres)
#                   print fmt0(just1)[1:] % ("bincounts_uniq_where_lbls",
#                       bincounts_uniq_where_lbls)
#
#               bincounts_max_lbls_in1d_where_lbls = np.in1d(
#                   bincounts_max_lbls, bincounts_uniq_where_lbls, invert=True)
#               bincounts_max_lbl = (
#                   bincounts_max_lbls[bincounts_max_lbls_in1d_where_lbls][0])
#               if 1 and prt:
#                   print fmt0(just1)[0:] % (
#                       "bincounts_max_lbls_in1d_where_lbls",
#                       bincounts_max_lbls_in1d_where_lbls)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % ("bincounts_max_lbl",
#                       bincounts_max_lbl)

#%              ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#%          #   print "*** " * 20
#%          #   print "*** " * 20

#               bincounts_max_masks.append(
#                   bincounts_fltrd_ge_bincounts_uniq_lbls ==
#                   bincounts_max_lbl)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "(max%i) bincounts_max_masks[-1]" % ilast,
#                       bincounts_max_masks[-1].astype(np.int))

#               bincounts_uniq_wheres.append(
#                   np.where(bincounts_max_masks[-1])[0])
#               bincounts_max_zs.append(
#                   np.mean(window_zs_bins[bincounts_uniq_wheres[-1]]))
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "(max%i) bincounts_max_masks[-1].shape" % ilast,
#                       bincounts_max_masks[-1].shape)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "(max%i) bincounts_uniq_wheres[-1]" % ilast,
#                       bincounts_uniq_wheres[-1])
#                   print fmt0(just1)[1:] % (
#                       "(max%i) bincounts_max_zs[-1]" % ilast,
#                       bincounts_max_zs[-1])

                print
                print "ccccccccccccccccccccccccccccccccccccccccccccccccccccc"
                None if 1 else sys.exit()
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
            if (bincounts_fltrd_ge_bincounts_uniq_lbls_max == 4 and
            len(bincounts_maxs) == 3):
                print "ddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                if 1 and prt:
                    print fmt0(just1)[1:] % ("ilast", ilast)

                #
                ### ### ###
                #
                bincounts_max = bincounts_uniq
                bincounts_maxs.append(bincounts_max)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "(max%i) bincounts_max (bincounts_uniq)" % ilast,
                        bincounts_max)
                    #rint fmt0(just1)[1:] % (
                    #   "(max%i) bincounts_maxs[-1]" % ilast,
                    #   bincounts_maxs[-1])
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_maxs",
                        bincounts_maxs)

                #
                ### ### ###
                #
                bincounts_max_where = (
                    np.where(window_zs_bincounts_fltrd == bincounts_max)[0])
                bincounts_uniq_wheres.append(bincounts_max_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % (
                        "bincounts_max", bincounts_max)
                if 1 and prt:
                    #rint fmt0(just1)[0:] % ("window_zs_bincounts_fltrd.shape",
                    #   window_zs_bincounts_fltrd.shape)
                    print fmt0(just1)[1:] % ("window_zs_bincounts_fltrd",
                        window_zs_bincounts_fltrd)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_max_where",
                        bincounts_max_where)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_uniq_wheres",
                        bincounts_uniq_wheres)

                print "*** " * 20
                ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

                if 1 and prt:
                    #rint fmt0(just1)[1:] % (
                    #   "bincounts_fltrd_ge_bincounts_uniq_lbls.shape",
                    #   bincounts_fltrd_ge_bincounts_uniq_lbls.shape)
                    print fmt0(just1)[1:] % (
                        "bincounts_fltrd_ge_bincounts_uniq_lbls",
                        bincounts_fltrd_ge_bincounts_uniq_lbls)
                if 1 and prt:
                    print fmt0(just1)[1:] % ("bincounts_uniq_wheres",
                        bincounts_uniq_wheres)

                bincounts_max_lbls = np.array([]).astype(np.int)
                if 1 and prt:
                    print fmt0(just1)[0:] % ("(init) bincounts_max_lbls",
                        bincounts_max_lbls)

                for idx, bincounts_max in enumerate(bincounts_maxs):
                    if 1 and prt:
                        print fmt0(just1)[0:] % ("idx", idx)
                    if 1 and prt:
                        print fmt0(just1)[1:] % ("bincounts_max",
                            bincounts_max)

#%              ### ### ###

#%              #   if 1 and prt:
#%              #       print fmt0(just1)[1:] % ("(bfor) bincounts_max_lbls",
#%              #           bincounts_max_lbls)

#%              #   bincounts_max_where_lbls = (
#%              #       bincounts_fltrd_ge_bincounts_uniq_lbls[
#%              #           bincounts_uniq_wheres[idx]])
#%              #   if 1 and prt:
#%              #       print fmt0(just1)[1:] % ("bincounts_uniq_wheres[idx]",
#%              #           bincounts_uniq_wheres[idx])
#%              #   if 1 and prt:
#%              #       print fmt0(just1)[1:] % ("bincounts_max_where_lbls",
#%              #           bincounts_max_where_lbls)

#%      #       bincounts_max_lbls_in1d_max0_lbls = np.in1d(
#%      #           bincounts_max0_lbls, bincounts_max_lbls, invert=True)
#               bincounts_max_lbl = (
#                   bincounts_max_lbls[bincounts_max_lbls_in1d_where_lbls][0])
#%      #       if 1 and prt:
#%      #           print fmt0(just1)[0:] % (
#%      #               "bincounts_max_lbls_in1d_max0_lbls",
#%      #               bincounts_max_lbls_in1d_max0_lbls)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % ("bincounts_max_lbl",
#                       bincounts_max_lbl)

#%      #       bincounts_max_lbls = (
#%      #           np.append(bincounts_max_lbls, bincounts_max0_lbls))
#%      #       if 1 and prt:
#%      #           print fmt0(just1)[1:] % ("(max0) bincounts_max_lbls",
#%      #               bincounts_max_lbls)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "[bincounts_uniq_wheres[0], bincounts_max0_lbl]",
#                       [bincounts_uniq_wheres[0], bincounts_max0_lbl])

#%              #ys.exit()
#%              ### ### ###

#%      #       if 1 and prt:
#%      #           print fmt0(just1)[0:] % ("(bfor) bincounts_max_lbls",
#%      #               bincounts_max_lbls)

#%      #       bincounts_max1_lbls = (
#%      #           bincounts_fltrd_ge_bincounts_uniq_lbls[
#%      #               bincounts_uniq_wheres[1]])
#%      #       if 1 and prt:
#%      #           print fmt0(just1)[1:] % ("bincounts_max1_lbls",
#%      #               bincounts_max1_lbls)

#%      #       bincounts_max_lbls_in1d_max1_lbls = np.in1d(
#%      #           bincounts_max1_lbls, bincounts_max_lbls, invert=True)
#               bincounts_max_lbl = (
#                   bincounts_max_lbls[bincounts_max_lbls_in1d_where_lbls][0])
#%      #       if 1 and prt:
#%      #           print fmt0(just1)[0:] % (
#%      #               "bincounts_max_lbls_in1d_max1_lbls",
#%      #               bincounts_max_lbls_in1d_max1_lbls)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % ("bincounts_max_lbl",
#                       bincounts_max_lbl)

#%      #       bincounts_max_lbls = (
#%      #           np.append(bincounts_max_lbls, bincounts_max1_lbls))
#%      #       if 1 and prt:
#%      #           print fmt0(just1)[1:] % ("(max1) bincounts_max_lbls",
#%      #               bincounts_max_lbls)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "[bincounts_uniq_wheres[1], bincounts_max1_lbl]",
#                       [bincounts_uniq_wheres[1], bincounts_max1_lbl])

#%              #ys.exit()
#%              ### ### ###

#%      #       if 1 and prt:
#%      #           print fmt0(just1)[0:] % ("(bfor) bincounts_max_lbls",
#%      #               bincounts_max_lbls)

#%      #       bincounts_max2_lbls = (
#%      #           bincounts_fltrd_ge_bincounts_uniq_lbls[
#%      #               bincounts_uniq_wheres[2]])
#%      #       if 1 and prt:
#%      #           print fmt0(just1)[1:] % ("bincounts_max2_lbls",
#%      #               bincounts_max2_lbls)

#%      #       bincounts_max_lbls_in1d_max2_lbls = np.in1d(
#%      #           bincounts_max2_lbls, bincounts_max_lbls, invert=True)
#               bincounts_max_lbl = (
#                   bincounts_max_lbls[bincounts_max_lbls_in1d_where_lbls][0])
#%      #       if 1 and prt:
#%      #           print fmt0(just1)[0:] % (
#%      #               "bincounts_max_lbls_in1d_max2_lbls",
#%      #               bincounts_max_lbls_in1d_max2_lbls)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % ("bincounts_max_lbl",
#                       bincounts_max_lbl)

#%  #           bincounts_max_lbls = (
#%  #               np.append(bincounts_max_lbls, bincounts_max2_lbls))
#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % ("(max2) bincounts_max_lbls",
#%  #                   bincounts_max_lbls)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "[bincounts_uniq_wheres[2], bincounts_max2_lbl]",
#                       [bincounts_uniq_wheres[2], bincounts_max2_lbl])

#%              #ys.exit()
#% #            ### ### ###
#% #
#% #            if 1 and prt:
#% #                print fmt0(just1)[0:] % ("(bfor) bincounts_max_lbls",
#% #                    bincounts_max_lbls)
#% #
#% #            bincounts_max3_lbls = (
#% #                bincounts_fltrd_ge_bincounts_uniq_lbls[
#% #                    bincounts_uniq_wheres[3]])
#% #            if 1 and prt:
#% #                print fmt0(just1)[1:] % ("bincounts_max3_lbls",
#% #                    bincounts_max3_lbls)
#% #
#% #
#% #
#% #            bincounts_max_lbls = (
#% #                np.append(bincounts_max_lbls, bincounts_max3_lbls))
#% #            if 1 and prt:
#% #                print fmt0(just1)[1:] % ("(max3) bincounts_max_lbls",
#% #                    bincounts_max_lbls)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "[bincounts_uniq_wheres[3], bincounts_max3_lbl]",
#                       [bincounts_uniq_wheres[3], bincounts_max3_lbl])

                sys.exit()
                ### ### ###

#%          #   bincounts_max_lbls = np.array([
#%          #       bincounts_max0_lbl,
#%          #       bincounts_max1_lbl,
#%          #       bincounts_max2_lbl,
#%          #       bincounts_max3_lbl,
#%          #   ])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_max_lbls",
#%          #           bincounts_max_lbls)

#%              ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#%              print "*** " * 20

##              bincounts_maxs.append(bincounts_max)
##              if 1 and prt:
##                  print fmt0(just1)[1:] % ("last", ilast)
##              if 1 and prt:
##                  print fmt0(just1)[1:] % (
##                      "(max%i) bincounts_max (bincounts_uniq)" % ilast,
##                      bincounts_max)
##                  #rint fmt0(just1)[1:] % (
##                  #   "(max%i) bincounts_maxs[-1]" % ilast,
##                  #   bincounts_maxs[-1])

#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq", bincounts_max)
#%          #   if 1 and prt:
#%          #       #rint fmt0(just1)[0:] % ("bincounts_uniqs.shape",
#%          #       #   bincounts_uniqs.shape)
#%          #       print fmt0(just1)[1:] % ("bincounts_uniqs",
#%          #           bincounts_uniqs)

#%          #   bincounts_uniq_where = (
#%          #       np.where(bincounts_uniqs == bincounts_uniq)[0])
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % ("bincounts_uniq_where",
#%          #           bincounts_uniq_where)

#%          #   bincounts_maxs.append(bincounts_max)
#%          #   if 1 and prt:
#%          #       print fmt0(just1)[1:] % (
#%          #           "(max%i) bincounts_max (bincounts_uniq)" % ilast,
#%          #           bincounts_max)
#%          #       #rint fmt0(just1)[1:] % (
#%          #       #   "(max%i) bincounts_maxs[-1]" % ilast,
#%          #       #   bincounts_maxs[-1])
#%          #   if 1 and prt:
#%          #       #rint fmt0(just1)[0:] % ("bincounts_uniqs.shape",
#%          #       #   bincounts_uniqs.shape)
#%          #       print fmt0(just1)[1:] % ("bincounts_uniqs",
#%          #           bincounts_uniqs)

#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % ("bincounts_maxs", bincounts_maxs)
#%  #
#%  #           bincounts_max_lbls = np.arange(
#%  #               1, bincounts_fltrd_ge_bincounts_uniq_lbls_max + 1)
#%  #           if 1 and prt:
#%  #               print fmt0(just1)[1:] % ("bincounts_max_lbls",
#%  #                   bincounts_max_lbls)
#%  #
#%  #           if 1 and prt:
#%  #               #rint fmt0(just1)[1:] % (
#%  #               #   "bincounts_fltrd_ge_bincounts_uniq_lbls.shape",
#%  #               #   bincounts_fltrd_ge_bincounts_uniq_lbls.shape)
#%  #               print fmt0(just1)[1:] % (
#%  #                   "bincounts_fltrd_ge_bincounts_uniq_lbls",
#%  #                   bincounts_fltrd_ge_bincounts_uniq_lbls)
#%  #
#%  #
#%  #           print "*** " * 20
#%  #           ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

#               if 1 and prt:
#                   print fmt1(just1)[0:] % (
#                       "bincounts_fltrd_ge_bincounts_uniq_lbls",
#                       bincounts_fltrd_ge_bincounts_uniq_lbls)
#
#               bincounts_max_lbls = np.arange(
#                   1, bincounts_fltrd_ge_bincounts_uniq_lbls_max + 1)
#               if 1 and prt:
#                   print fmt0(just1)[0:] % ("bincounts_max_lbls",
#                       bincounts_max_lbls)
#
#               bincounts_uniq_where_lbls = []
#               for bincounts_uniq_where in bincounts_uniq_wheres:
#                   bincounts_uniq_where_lbls.append(
#                       bincounts_fltrd_ge_bincounts_uniq_lbls[
#                           bincounts_uniq_where][0])
#                   #f 1 and prt:
#                   #   print fmt0(just1)[0:] % (
#                   #       "... [bincounts_uniq_where, " +
#                   #       "bincounts_uniq_where_lbls[-1]]",
#                   #       [bincounts_uniq_where,
#                   #           bincounts_uniq_where_lbls[-1]])
#               if 1 and prt:
#                   print fmt0(just1)[0:] % ("bincounts_uniq_wheres",
#                       bincounts_uniq_wheres)
#                   print fmt0(just1)[1:] % ("bincounts_uniq_where_lbls",
#                       bincounts_uniq_where_lbls)
#
#               bincounts_max_lbls_in1d_where_lbls = np.in1d(
#                   bincounts_max_lbls, bincounts_uniq_where_lbls, invert=True)
#               bincounts_max_lbl = (
#                   bincounts_max_lbls[bincounts_max_lbls_in1d_where_lbls][0])
#               if 1 and prt:
#                   print fmt0(just1)[0:] % (
#                       "bincounts_max_lbls_in1d_where_lbls",
#                       bincounts_max_lbls_in1d_where_lbls)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % ("bincounts_max_lbl",
#                       bincounts_max_lbl)

#%              ### ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##
#%          #   print "*** " * 20
#%          #   print "*** " * 20

#               bincounts_max_masks.append(
#                   bincounts_fltrd_ge_bincounts_uniq_lbls ==
#                   bincounts_max_lbl)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "(max%i) bincounts_max_masks[-1]" % ilast,
#                       bincounts_max_masks[-1].astype(np.int))

#               bincounts_uniq_wheres.append(
#                   np.where(bincounts_max_masks[-1])[0])
#               bincounts_max_zs.append(
#                   np.mean(window_zs_bins[bincounts_uniq_wheres[-1]]))
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "(max%i) bincounts_max_masks[-1].shape" % ilast,
#                       bincounts_max_masks[-1].shape)
#               if 1 and prt:
#                   print fmt0(just1)[1:] % (
#                       "(max%i) bincounts_uniq_wheres[-1]" % ilast,
#                       bincounts_uniq_wheres[-1])
#                   print fmt0(just1)[1:] % (
#                       "(max%i) bincounts_max_zs[-1]" % ilast,
#                       bincounts_max_zs[-1])

                print
                print "ddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                None if 1 else sys.exit()
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        print "-----------------------------------------------------------"
        print "==========================================================="

#       if 1 and prt:
#           print fmt0(just1)[0:] % ("bincounts_maxs", bincounts_maxs)
#%  #   if 1 and prt:
#%  #       print fmt1(just1)[0:] % ("bincounts_max_masks",
#%  #           prt_list([a.astype(np.int) for a in bincounts_max_masks]))
#%  #   if 1 and prt:
#%  #       print fmt1(just1)[0:] % ("bincounts_uniq_wheres",
#%  #           prt_list(bincounts_uniq_wheres))

        None if 0 else sys.exit()

##      print fmt0(just1)[1:] % ("(last) [bincounts_uniq, " +
##          "bincounts_fltrd_ge_uniq_labels_max, " +
##          "2lbls_uniq, 3lbls_uniq,]",
##          [
##              '%5i' % bincounts_uniq,
##              bincounts_fltrd_ge_uniq_labels_max,
##              '%5s' % bincounts_2lbls_uniq,
##              '%5s' % bincounts_3lbls_uniq,
##          ])
##
##      if 1 and prt:
##          print fmt0(just1)[0:] % ("(max1) bincounts_2lbls_uniq",
##              bincounts_2lbls_uniq)
##      if 1 and prt:
##          print fmt0(just1)[1:] % ("bincounts_2lbls_mask.shape",
##              bincounts_2lbls_mask.shape)
##          print fmt1(just1)[1:] % ("bincounts_2lbls_mask",
##              bincounts_2lbls_mask.astype(np.int))
##          print fmt0(just1)[1:] % ("bincounts_2lbls_labels.shape",
##              bincounts_2lbls_labels.shape)
##          print fmt1(just1)[1:] % ("bincounts_2lbls_labels",
##              bincounts_2lbls_labels)
##
##      if 1 and prt:
##          print fmt0(just1)[0:] % ("(max2) bincounts_3lbls_uniq",
##              bincounts_3lbls_uniq)
##          if bincounts_3lbls_uniq is not None:
##              print fmt0(just1)[0:] % (
##                  "bincounts_3lbls_mask.shape",
##                  bincounts_3lbls_mask.shape)
##              print fmt1(just1)[1:] % ("bincounts_3lbls_mask",
##                  bincounts_3lbls_mask.astype(np.int))
##              print fmt0(just1)[1:] % (
##                  "bincounts_3lbls_labels.shape",
##                  bincounts_3lbls_labels.shape)
##              print fmt1(just1)[1:] % ("bincounts_3lbls_labels",
##                  bincounts_3lbls_labels)
##
##  #   print "\nzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
##  #   None if 1 else sys.exit()
##      #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
##
##      bincounts_max1_where = np.where(
##          bincounts_fltrd == bincounts_max1)[0]
##      if 1 and prt:
##          print fmt0(just1)[0:] % ("bincounts_max1",
##              bincounts_max1)
##          print fmt0(just1)[1:] % ("bincounts_max1_where",
##              bincounts_max1_where)
##
### ### ### ### ### ### ### ### ### ### ### ### ### ###
##      if len(bincounts_max1_where) > 1:
##          bincounts_max0_label = (
##              bincounts_2lbls_labels[
##                  bincounts_max0_where])
##          if 1 and prt:
##              print fmt0(just1)[0:] % ("bincounts_max0_label",
##                  bincounts_max0_label)
##
##          bincounts_max0_label_mask = (
##              bincounts_2lbls_labels !=
##              bincounts_max0_label)
##          if 1 and prt:
##              print fmt0(just1)[0:] % (
##                  "bincounts_max0_label_mask.shape",
##                  bincounts_max0_label_mask.shape)
##              print fmt1(just1)[1:] % (
##                  "bincounts_max0_label_mask",
##                  bincounts_max0_label_mask.astype(np.int))
##
##          bincounts_max1_label_mask = (
##              bincounts_2lbls_mask &
##              bincounts_max0_label_mask)
##          if 1 and prt:
##              print fmt1(just1)[1:] % (
##                  "bincounts_2lbls_mask",
##                  bincounts_2lbls_mask.astype(np.int))
##              print fmt1(just1)[1:] % (
##                  "bincounts_max0_label_mask",
##                  bincounts_max0_label_mask.astype(np.int))
##              print fmt1(just1)[1:] % (
##                  "bincounts_max1_label_mask",
##                  bincounts_max1_label_mask.astype(np.int))
##
##          bincounts_max1_where = (
##              np.where(bincounts_max1_label_mask)[0])
##          if 1 and prt:
##              print fmt0(just1)[0:] % ("bincounts_max1",
##                  bincounts_max1)
##              print fmt0(just1)[1:] % (
##                  "bincounts_max1_label_mask.shape",
##                  bincounts_max1_label_mask.shape)
##              print fmt0(just1)[1:] % ("bincounts_max1_where",
##                  bincounts_max1_where)
##
##      None if 1 else sys.exit()

###     #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
###
###     # to make "max0 z" always greater than "max1 z"
###     if bincounts_max0_z < bincounts_max1_z:
###         #
###         bincounts_max0_z, bincounts_max1_z = (
###             bincounts_max1_z, bincounts_max0_z)
###         #
###         bincounts_max0, bincounts_max1 = (
###             bincounts_max1, bincounts_max0)
###
        if 1 and prt:
            print fmt0(just1)[0:] % (
                "[bincounts_max0_z, bincounts_max0]",
                [bincounts_max0_z, bincounts_max0])
        if 1 and prt:
            print fmt0(just1)[1:] % (
                "[bincounts_max1_z, bincounts_max1]",
                [bincounts_max1_z, bincounts_max1])

        bincounts_mid_z = (
            (bincounts_max0_z + bincounts_max1_z) / 2.)
        bincounts_mid = (
            (bincounts_max0 + bincounts_max1) / 2.)

        None if 1 else sys.exit()
        print "\nzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"
        None if 1 else sys.exit()

        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

        if 1 and ngn.make_gallery04_plots:
            make_gallery04_tow_end_event_window_plot(
                i, indy, np_hasnan_zs, window_xs, window_zs_fltr0,
                window_zs_bins, window_zs_bincounts_fltrd,
                bincounts_2lbls_mask,
                bincounts_max0_z, bincounts_max0,
                bincounts_max1_z, bincounts_max1,
                bincounts_mid_z, bincounts_mid,
            )

        None if 1 else sys.exit()

        #f sys_exit_flag:
        if 1:
            print "... sys.exit() ... yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy"
            sys.exit()

##      #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##      #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
##
##      window_xs_dzdy = np_src_xs_dzdy[window_indy[:-1], :]
##      window_ys_midp = np_src_ys_midp[window_indy[:-1], :]
##      #f 1 and prt:
##      #   #rint
##      #   print fmt0(just1)[1:] % ("window_xs_dzdy.shape",
##      #       window_xs_dzdy.shape)
##      #   #rint fmt1(just1)[1:] % ("window_xs[:, :9]", window_xs[:, :9])
##      #   #rint fmt1(just1)[1:] % ("window_xs_dzdy[:, :9]",
##      #   #   window_xs_dzdy[:, :9])
##      #   #
##      #   print fmt0(just1)[1:] % ("window_ys_midp.shape",
##      #       window_ys_midp.shape)
##      #   #rint fmt1(just1)[1:] % ("window_ys[:, :9]", window_ys[:, :9])
##      #   #rint fmt1(just1)[1:] % ("window_ys_midp[:, :9]",
##      #   #   window_ys_midp[:, :9])
##      #one if 1 else sys.exit()
##
##      #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
##      #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
##      # calculate first differences for each z versus y profile (along the
##      # axis=1 direction) in the measurements window.
##      #
##      # note:  These calculation are applied only z versys y profiles
##      #        having no NaN values. Otherwise, all z values for the profiles
##      #        (with at least one NaN value) are set to np.nan.
##      #
##      # note: All y-intervals between points are assumed equal.
##
##      #f 1 and prt:
##      #   print fmt0(just1)[0:] % (
##      #       "calculate first differences (dzdy)", "")
##
##      window_dzdys_fltr0_ntrend0 = (
##          np.full(window_zs_fltr0_ntrend0[1:, :].shape, np.nan))
##      #f 1 and prt:
##      #   print fmt0(just1)[0:] % ("window_zs_fltr0_ntrend0.shape",
##      #       window_zs_fltr0_ntrend0.shape)
##      #   print fmt0(just1)[1:] % ("window_dzdys_fltr0_ntrend0.shape",
##      #       window_dzdys_fltr0_ntrend0.shape)
##
##      for win_col_idx in xrange(window_zs_fltr0_ntrend0.shape[1]):
##          #f 1 and prt:
##          #   print fmt0(just1)[1:] % ("win_col_idx", win_col_idx)

#       #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
#
#       if 0 and prt:
#           print fmt1(just1)[0:] % (
#               "pd_src_us_tow_present.ix[window_indy, 'U-Sensor']",
#               pd_src_us_tow_present.ix[window_indy, 'U-Sensor'])
#
#       None if 1 else sys.exit()

#       #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
#
#       window_zs_bins_midp = (window_zs_bins[1:] + window_zs_bins[:-1]) / 2.
#       if 1 and prt:
#           print fmt0(just1)[0:] % ("window_zs_bins_midp.shape",
#               window_zs_bins_midp.shape)
#           print fmt1(just1)[1:] % ("window_zs_bins_midp",
#               window_zs_bins_midp)
#
#       window_zs_bincounts_fltrd_diff = np.diff(window_zs_bincounts_fltrd)
#       if 1 and prt:
#           print fmt0(just1)[1:] % ("window_zs_bincounts_fltrd_diff.shape",
#               window_zs_bincounts_fltrd_diff.shape)
#           print fmt1(just1)[1:] % ("window_zs_bincounts_fltrd_diff",
#               window_zs_bincounts_fltrd_diff)
#
#       None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        np_tow_diff_rois_event = np_tow_diff_values_ascending[i]
        if 1 and prt:
            print fmt0(just1)[0:] % ("np_tow_diff_rois_event.shape",
                np_tow_diff_rois_event.shape)
            print fmt0(just1)[1:] % ("np_tow_diff_rois_event",
                np_tow_diff_rois_event)

        None if 1 else sys.exit()

#%     #for j, tow_diff_roi_event in enumerate(np_tow_diff_rois_event):
#%     #    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
#%     #    meast_id = indy + 1
#%     #    if tow_diff_roi_event == 0:
#%     #        # there is not tow start/stop event
#%     #        continue
#%     #    if 1 and prt:
#%     #        #f j == 0:
#%     #        #   print
#%     #        print "\n%sanalyze this roi event %s" % (
#%     #             '... ', '... ' * start_mult)
#%     #        print fmt0(just1)[1:] % (
#%     #            "[i, indy, meast_id, j, tow_diff_roi_event]",
#%     #            [i, indy, meast_id, j, tow_diff_roi_event])

#%          #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

#%     #    tow_roi_id = tow_ids_within_course[j]
#%     #    if 1 or prt:
#%     #        print fmt0(just1)[1:] % ('tow_roi_id', tow_roi_id)

#%          #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

#%     #    if 1 and ngn.make_gallery05_plots:
#%     #        make_gallery05_tow_end_event_window_roi_plot(
#%     #            i, indy, tow_roi_id, tow_diff_roi_event,
#      #            window_xs, window_ys, window_zs_fltr0,
#      #            window_zs_bins, window_zs_bincounts,
#      #            window_zs_bincounts_fltrd,
#      #            window_zs_bincounts_fltrd_max_z,
#      #            window_zs_bincounts_fltrd_max,
#%     #        )
#      #
#      #    None if 1 else sys.exit()
#%     #    break

#%          #ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
#%          #ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
#%          #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

#%          #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
# #
# #         # get the indices inside the roi tow edge indices
# #         tow_diff_roi_idxs = (
# #             np.arange(tow_edge_xref_idxs0[j] + 1, tow_edge_xref_idxs1[j]))
# #         if 1 or prt:
# #             print fmt0(just1)[0:] % (('[tow_edge_xref_idxs0[j == %i]' % j +
# #                 ', tow_edge_xref_idxs1[j == %i]]') % j,
# #                 [tow_edge_xref_idxs0[j], tow_edge_xref_idxs1[j]])
# #             print fmt0(just1)[1:] % ('tow_diff_roi_idxs.shape',
# #                 tow_diff_roi_idxs.shape)
# #             print fmt1(just1)[1:] % ('tow_diff_roi_idxs',
# #                 tow_diff_roi_idxs)
# #
# #         #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
# #
# #         roi_event_xs = window_xs[:, tow_diff_roi_idxs]
# #         roi_event_ys = window_ys[:, tow_diff_roi_idxs]
# #         roi_event_zs = window_zs_fltr0_ntrend0[:, tow_diff_roi_idxs].copy()
# #         if 1 or prt:
# #             print fmt0(just1)[0:] % ('roi_event_xs.shape',
# #                 roi_event_xs.shape)
# #             print fmt0(just1)[1:] % ('roi_event_ys.shape',
# #                 roi_event_ys.shape)
# #             print fmt0(just1)[1:] % ('roi_event_zs.shape',
# #                 roi_event_zs.shape)
# #
##          #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

##          roi_event_zs_flatten = roi_event_zs.copy().flatten()
##          if 1 and prt:
##              print fmt0(just1)[0:] % ("roi_event_zs_flatten.shape",
##                  roi_event_zs_flatten.shape)
##              print fmt1(just1)[1:] % ("roi_event_zs_flatten",
##                  roi_event_zs_flatten)
##
##          roi_event_zs_flatten_nnan = (
##              roi_event_zs_flatten[~ np.isnan(roi_event_zs_flatten)])
##          if 1 and prt:
##              print fmt0(just1)[0:] % ("roi_event_zs_flatten_nnan.shape",
##                  roi_event_zs_flatten_nnan.shape)
##              print fmt1(just1)[1:] % ("roi_event_zs_flatten_nnan",
##                  roi_event_zs_flatten_nnan)

##          decimals = 2
##          roi_event_zs_flatten_nnan_rnd = (
##              np.round(roi_event_zs_flatten_nnan, decimals=decimals))
##          if 1 and prt:
##              print fmt0(just1)[0:] % ("roi_event_zs_flatten_nnan_rnd.shape",
##                  roi_event_zs_flatten_nnan_rnd.shape)
##              print fmt1(just1)[1:] % ("roi_event_zs_flatten_nnan_rnd",
##                  roi_event_zs_flatten_nnan_rnd)

##          roi_event_zs_bins, roi_event_zs_bincounts = (
##              np.unique(roi_event_zs_flatten_nnan_rnd, return_counts=True))
##          if 1 and prt:
##              print fmt0(just1)[0:] % ("roi_event_zs_bins.shape",
##                  roi_event_zs_bins.shape)
##              print fmt1(just1)[1:] % ("roi_event_zs_bins",
##                  roi_event_zs_bins)
##              None if 1 else sys.exit()
##          if 1 and prt:
##              print fmt0(just1)[1:] % ("roi_event_zs_bincounts.shape",
##                  roi_event_zs_bincounts.shape)
##              print fmt1(just1)[1:] % ("roi_event_zs_bincounts",
##                  roi_event_zs_bincounts)
##              print fmt1(just1)[1:] % ("np.sort(roi_event_zs_bincounts)",
##                  np.sort(roi_event_zs_bincounts))
##              None if 1 else sys.exit()

##          #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
# #
# #         #
# #         ### ### ###
# #         #
# #
# #         #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
# #
# #     #   if 1 or ngn.make_gallery04_plots:
# #     #       make_gallery04_tow_end_event_plot(
# #     #           indy, tow_roi_id, tow_diff_roi_event,
# #     #           roi_event_xs, roi_event_ys, roi_event_zs,
# #     #           roi_event_zs_bins, roi_event_zs_bincounts,
# #     #   )
# #         print "\nmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"
# #         #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
# #         #break
# #
# #     None if 1 else sys.exit()
# #     #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
# #
# #     #ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
# #     #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
# #
# #     #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
# #     #ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

# # #   for win_col_idx in xrange(np_hasnan_zs.shape[1]):
# # #       if 1 and prt:
# # #           if win_col_idx == 0:
# # #               print
# # #           print fmt0(just1)[1:] % ("win_col_idx", win_col_idx)
# #

# #     #ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
# #     #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

#           if np.isnan(window_zs_fltr0_ntrend0[:, win_col_idx]).any():
#               window_zs_fltr0_ntrend0[:, win_col_idx] = np.nan
#               #f 1 and prt:
#               #   #rint fmt0(just1)[1:] % (
#               #   #   "(init) window_zs_fltr0_ntrend0[:, win_col_idx]",
#               #   #   window_zs_fltr0_ntrend0[:, win_col_idx])
#               #   print fmt0(just1)[1:] % (
#               #       "", " ... not all values ... (has at least one NaN)")
#               #   print fmt0(just1)[1:] % (
#               #       "(updt) window_zs_fltr0_ntrend0[:, win_col_idx]",
#               #       window_zs_fltr0_ntrend0[:, win_col_idx])
#               #one if 0 else sys.exit()
#           else:
#               window_dzdys_fltr0_ntrend0[:, win_col_idx] = (
#                   window_zs_fltr0_ntrend0[1:, win_col_idx] -
#                   window_zs_fltr0_ntrend0[:-1, win_col_idx])
#               #f 1 and prt:
#               #   print fmt0(just1)[1:] % (
#               #       "window_zs_fltr0_ntrend0[:, win_col_idx]",
#               #       window_zs_fltr0_ntrend0[:, win_col_idx])
#               #   print fmt0(just1)[1:] % (
#               #       "", " ... has all values ... (has no NaN values)")
#               #   print fmt0(just1)[1:] % (
#               #       "window_zs_fltr0_ntrend0[1:, win_col_idx]",
#               #       window_zs_fltr0_ntrend0[1:, win_col_idx])
#               #   print fmt0(just1)[1:] % (
#               #       "window_zs_fltr0_ntrend0[:-1, win_col_idx]",
#               #       window_zs_fltr0_ntrend0[:-1, win_col_idx])
#               #   print fmt0(just1)[1:] % (
#               #       "window_dzdys_fltr0_ntrend0[:, win_col_idx]",
#               #       window_dzdys_fltr0_ntrend0[:, win_col_idx])
#               #one if 0 else sys.exit()
#           #f win_col_idx >= 20:
#           #   break
#           #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
#       #one if 1 else sys.exit()

#       #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
#       # get the mask and data for dzdys and yz & zs corresponding to dzdys
#       # large enough to be tow (strip) edges

#       #f 1 and prt:
#       #   print fmt0(just1)[0:] % (
#       #       "get the tow start edges & tow stop edges dzdy, ys & zs data",
#       #       "get the tow start edges & tow stop edges dzdy, ys & zs data")

#       #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

#       window_dzdys_fltr0_ntrend0_mask_edges_start = (
#           window_dzdys_fltr0_ntrend0 >= dzdys_threshold)
#       #f 1 and prt:
#       #   print fmt0(just1)[0:] % (
#       #       "window_dzdys_fltr0_ntrend0_mask_edges_start.shape",
#       #       window_dzdys_fltr0_ntrend0_mask_edges_start.shape)

#       window_ys_fltr0_ntrend0_mask_edges_start = (
#           np.full((window_ys.shape), False).astype(np.bool))
#       window_ys_fltr0_ntrend0_mask_edges_start[:-1] = (
#           window_ys_fltr0_ntrend0_mask_edges_start[:-1] |
#           window_dzdys_fltr0_ntrend0_mask_edges_start)
#       window_ys_fltr0_ntrend0_mask_edges_start[1:] = (
#           window_ys_fltr0_ntrend0_mask_edges_start[1:] |
#           window_dzdys_fltr0_ntrend0_mask_edges_start)
#       #f 1 and prt:
#       #   print fmt0(just1)[0:] % (
#       #       "window_ys_fltr0_ntrend0_mask_edges_start.shape",
#       #       window_ys_fltr0_ntrend0_mask_edges_start.shape)
#       #   #rint fmt1(just1)[1:] % (
#       #   #   "window_ys_fltr0_ntrend0_mask_edges_start[:, 0: 21]",
#       #   #   window_ys_fltr0_ntrend0_mask_edges_start[:, 0: 21])
#       #   #rint fmt1(just1)[1:] % (
#       #   #   "window_ys_fltr0_ntrend0_mask_edges_start[:, 510: 531]",
#       #   #   window_ys_fltr0_ntrend0_mask_edges_start[:, 510: 531])

#       window_ys_fltr0_ntrend0_edges_start = window_ys.copy().astype(np.float)
#       window_ys_fltr0_ntrend0_edges_start[
#           ~ window_ys_fltr0_ntrend0_mask_edges_start
#       ] = np.nan
#       #f 1 and prt:
#       #   print fmt0(just1)[1:] % (
#       #       "window_ys_fltr0_ntrend0_edges_start.shape",
#       #       window_ys_fltr0_ntrend0_edges_start.shape)
#       #   #rint fmt1(just1)[1:] % (
#       #   #   "window_ys_fltr0_ntrend0_edges_start[:, 0: 21]",
#       #   #   window_ys_fltr0_ntrend0_edges_start[:, 0: 21])
#       #   #rint fmt1(just1)[1:] % (
#       #   #   "window_ys_fltr0_ntrend0_edges_start[:, 510: 531]",
#       #   #   window_ys_fltr0_ntrend0_edges_start[:, 510: 531])

#       #one if 1 else sys.exit()

#       #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

#       window_dzdys_fltr0_ntrend0_mask_edges_stop = (
#           window_dzdys_fltr0_ntrend0 <= -dzdys_threshold)
#       #f 1 and prt:
#       #   print fmt0(just1)[0:] % (
#       #       "window_dzdys_fltr0_ntrend0_mask_edges_stop.shape",
#       #       window_dzdys_fltr0_ntrend0_mask_edges_stop.shape)

#       window_ys_fltr0_ntrend0_mask_edges_stop = (
#           np.full((window_ys.shape), False).astype(np.bool))
#       window_ys_fltr0_ntrend0_mask_edges_stop[:-1] = (
#           window_ys_fltr0_ntrend0_mask_edges_stop[:-1] |
#           window_dzdys_fltr0_ntrend0_mask_edges_stop)
#       window_ys_fltr0_ntrend0_mask_edges_stop[1:] = (
#           window_ys_fltr0_ntrend0_mask_edges_stop[1:] |
#           window_dzdys_fltr0_ntrend0_mask_edges_stop)
#       #f 1 and prt:
#       #   print fmt0(just1)[0:] % (
#       #       "window_ys_fltr0_ntrend0_mask_edges_stop.shape",
#       #       window_ys_fltr0_ntrend0_mask_edges_stop.shape)
#       #   #rint fmt1(just1)[1:] % (
#       #   #   "window_ys_fltr0_ntrend0_mask_edges_stop[:, 0: 21]",
#       #   #   window_ys_fltr0_ntrend0_mask_edges_stop[:, 0: 21])
#       #   #rint fmt1(just1)[1:] % (
#       #   #   "window_ys_fltr0_ntrend0_mask_edges_stop[:, 510: 531]",
#       #   #   window_ys_fltr0_ntrend0_mask_edges_stop[:, 510: 531])

#       window_ys_fltr0_ntrend0_edges_stop = window_ys.copy().astype(np.float)
#       window_ys_fltr0_ntrend0_edges_stop[
#           ~ window_ys_fltr0_ntrend0_mask_edges_stop
#       ] = np.nan
#       #f 1 and prt:
#       #   print fmt0(just1)[1:] % (
#       #       "window_ys_fltr0_ntrend0_edges_stop.shape",
#       #       window_ys_fltr0_ntrend0_edges_stop.shape)
#       #   #rint fmt1(just1)[1:] % (
#       #   #   "window_ys_fltr0_ntrend0_edges_stop[:, 0: 21]",
#       #   #   window_ys_fltr0_ntrend0_edges_stop[:, 0: 21])
#       #   #rint fmt1(just1)[1:] % (
#       #   #   "window_ys_fltr0_ntrend0_edges_stop[:, 510: 531]",
#       #   #   window_ys_fltr0_ntrend0_edges_stop[:, 510: 531])

#       #one if 1 else sys.exit()

#       #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
#       # get the mask(s) for tow (strip) edges only the region(s) of interest

#       #f 1 and prt:
#       #   print fmt0(just1)[0:] % (
#       #       "get the mask(s) for tow (strip) regions of interest", "")

#       window_xs_dzdy_mean0 = window_xs_dzdy.mean(axis=0)
#       #f 1 and prt:
#       #   print fmt0(just1)[0:] % ("window_xs.shape",
#       #       window_xs.shape)
#       #   #rint fmt1(just1)[1:] % ("window_xs[:, :9]", window_xs[:, :9])
#       #f 1 and prt:
#       #   print fmt0(just1)[0:] % ("window_xs_dzdy.shape",
#       #       window_xs_dzdy.shape)
#       #   #rint fmt1(just1)[1:] % ("window_xs_dzdy[:, :9]",
#       #   #   window_xs_dzdy[:, :9])
#       #f 1 and prt:
#       #   print fmt0(just1)[0:] % ("window_xs_dzdy_mean0.shape",
#       #       window_xs_dzdy_mean0.shape)
#       #   #rint fmt1(just1)[1:] % ("window_xs_dzdy_mean0[:9]",
#       #   #   window_xs_dzdy_mean0[:9])
#       #one if 1 else sys.exit()

#       #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

#       if ngn.write_to_results_dir:
#           # this is for plotting
#           rois_mask = np.full(
#               window_xs_dzdy_mean0.shape, False).astype(np.bool)
#           #f 1 and prt:
#           #   print fmt0(just2)[0:] % (
#           #       "(init) [rois_mask .dtype, .shape, .sum()]",
#           #       [rois_mask.dtype, rois_mask.shape, rois_mask.sum()])
#           #one if 1 else sys.exit()

#           # this is for plotting
#           roi_tow_xs_center = []
#           roi_tow_ys_mean_meast_idx = []
#           roi_tow_starts_stops = []

#       for roi_idx, roi_tow_id in enumerate(roi_tow_ids):

#           pd_src_us_tow_diff_row_val = (
#               pd_src_us_tow_diff_row_values[roi_idx])

#           if pd_src_us_tow_diff_row_val == 0:
#               # this is not a region of interest in the field of view
#               #f 1 and prt:
#               #   print fmt0(just2)[1:] % (
#               #       "this is not a region of interest (in the FOV)",
#               #       "continue")
#               continue

#           tow_diff_name = tow_diff_names[roi_idx]
#           #f 1 and prt:
#           #   #rint "\n%s" % ('KKK ' * 40)
#           #   print fmt0(just2)[0:] % ("[roi_idx, roi_tow_id, " +
#           #       "tow_diff_name, pd_src_us_tow_diff_row_val]",
#           #       [roi_idx, roi_tow_id, tow_diff_name,
#           #       pd_src_us_tow_diff_row_val])
#           #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
#           roi_tow_center_xref = roi_tow_center_xrefs[roi_idx]
#           #f 1 and prt:
#           #   print fmt0(just2)[1:] % ("roi_tow_center_xref",
#           #       roi_tow_center_xref)
#           #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
#           roi_tow_edge_xref_lf = roi_tow_edge_xrefs_lf[roi_idx]
#           roi_tow_edge_xref_rt = roi_tow_edge_xrefs_rt[roi_idx]
#           #f 1 and prt:
#           #   print fmt0(just2)[1:] % (
#           #       "[roi_tow_edge_xref_lf, roi_tow_edge_xref_rt]",
#           #       np.array([roi_tow_edge_xref_lf, roi_tow_edge_xref_rt]))
#           #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
#           roi_mask = ((window_xs_dzdy_mean0 >= roi_tow_edge_xref_lf) &
#               (window_xs_dzdy_mean0 <= roi_tow_edge_xref_rt))
#           window_xs_roi_mask = np.array([roi_mask, ] * window_xs.shape[0])
#           #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

#           if pd_src_us_tow_diff_row_val == 1:
#               window_ys_fltr0_ntrend0_edges_start_roi = (
#                   window_ys_fltr0_ntrend0_edges_start.copy())
#               window_ys_fltr0_ntrend0_edges_start_roi[
#                   ~ window_xs_roi_mask] = np.nan
#               window_ys_fltr0_ntrend0_edges_start_roi_mean = (
#                   np.nanmean(window_ys_fltr0_ntrend0_edges_start_roi))
#               tow_diff_num_i_value = (
#                   window_ys_fltr0_ntrend0_edges_start_roi_mean)
#               #
#               window_ys_fltr0_ntrend0_mask_edges = (
#                   window_ys_fltr0_ntrend0_mask_edges_start)

#           if pd_src_us_tow_diff_row_val == -1:
#               window_ys_fltr0_ntrend0_edges_stop_roi = (
#                   window_ys_fltr0_ntrend0_edges_stop.copy())
#               window_ys_fltr0_ntrend0_edges_stop_roi[
#                   ~ window_xs_roi_mask] = np.nan
#               window_ys_fltr0_ntrend0_edges_stop_roi_mean = (
#                   np.nanmean(window_ys_fltr0_ntrend0_edges_stop_roi))
#               tow_diff_num_i_value = (
#                   window_ys_fltr0_ntrend0_edges_stop_roi_mean)
#               #
#               window_ys_fltr0_ntrend0_mask_edges = (
#                   window_ys_fltr0_ntrend0_mask_edges_stop)

#           #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

#           tow_diff_num_i_value_us = np.interp(tow_diff_num_i_value,
#               window_indy, pd_src_us_tow_present.ix[window_indy, 'U-Sensor'],
#               left=np.nan, right=np.nan)
#           #f 1 and prt:
#           #   print fmt0(just1)[0:] % ("tow_diff_num_i_value",
#           #       tow_diff_num_i_value)
#           #f 1 and prt:
#           #   print fmt0(just1)[0:] % ("tow_diff_num_i_value_us",
#           #       tow_diff_num_i_value_us)

#           if ngn.write_to_results_dir:
#               # this is for plotting
#               rois_mask = (rois_mask | roi_mask)
#               roi_tow_xs_center.append(roi_tow_center_xref)
#               roi_tow_ys_mean_meast_idx.append(tow_diff_num_i_value)
#               roi_tow_starts_stops.append(pd_src_us_tow_diff_row_val)

#           #f 0 and prt:
#           #   rw, cl = len(pd_src_us_tow_diff), 13
#           #   print fmt1(just1)[0:] % ("pd_results_ends.iloc[:%s,:%i]" % (
#           #       rw, cl), pd_results_ends.iloc[:rw, :cl])

#           tow_diff_num = tow_diff_names[roi_idx].replace('d', '')
#           tow_diff_num_i = tow_diff_num + 'i'
#           tow_diff_num_us = tow_diff_num + 'us'
#           tow_diff_num_xc = tow_diff_num + 'xc'

#           tow_diff_cols = [tow_diff_num_i, tow_diff_num_us, tow_diff_num_xc]
#           pd_results_ends.ix[[i], tow_diff_cols] = [
#               tow_diff_num_i_value,
#               tow_diff_num_i_value_us,
#               roi_tow_center_xref,
#           ]
#           tow_diff_cols = ['ProfileID', 'MeastID',
#               'TowPresentBits_Tow32toTow01', 'U-Sensor',
#               tow_diff_num_i, tow_diff_num_us, tow_diff_num_xc]
#           #f 1 and prt:
#           #   print fmt1(just1)[0:] % (
#           #       "pd_results_ends.ix[[i], tow_diff_cols]",
#           #       pd_results_ends.ix[[i], tow_diff_cols])
#           #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
#           #f roi_idx >= 1:
#           #  break
#           #one if 0 else sys.exit()

#       #== === === === === === === === === === === === === === === === === ===

#       if 1 and ngn.write_to_results_dir and ngn.make_gallery03_plots:
#           #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

#           number_of_profiles = np_hasnan_zs.shape[0]
#           #f 1 and prt:
#           #   print fmt0(just1)[0:] % ("number_of_profiles",
#           #       number_of_profiles)

#           cols = ['U-Sensor', 'MeastID'] + tow_diff_names
#           pd_src_us_tow_diff_row = pd_src_us_tow_diff.ix[[i], cols]
#           #f 1 and prt:
#           #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff",
#           #       pd_src_us_tow_diff)
#           #f 1 and prt:
#           #   print fmt1(just1)[0:] % ("pd_src_us_tow_diff_row",
#           #       pd_src_us_tow_diff_row)

#           window_zs_ntrend0 = window_zs - window_zs_lsbf0
#           #f 1 and prt:
#           #   print fmt0(just1)[1:] % ("window_zs_ntrend0.shape",
#           #       window_zs_ntrend0.shape)

#           #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

#           window_xs_rois_mask = (
#               np.array([rois_mask, ] * window_xs.shape[0]))
#           #f 1 and prt:
#           #   print fmt0(just1)[0:] % ("window_xs_rois_mask.shape",
#           #       window_xs_rois_mask.shape)

#           window_zs_fltr0_ntrend0_edges_start = (
#               window_zs_fltr0_ntrend0.copy())
#           window_zs_fltr0_ntrend0_edges_start[
#               ~ window_ys_fltr0_ntrend0_mask_edges_start
#           ] = np.nan
#           #f 1 and prt:
#           #   print fmt0(just1)[1:] % (
#           #       "window_zs_fltr0_ntrend0_edges_start.shape",
#           #       window_zs_fltr0_ntrend0_edges_start.shape)

#           window_zs_fltr0_ntrend0_edges_start_rois = (
#               window_zs_fltr0_ntrend0_edges_start.copy())
#           window_zs_fltr0_ntrend0_edges_start_rois[
#               ~ window_xs_rois_mask] = np.nan

#           window_zs_fltr0_ntrend0_edges_stop = window_zs_fltr0_ntrend0.copy()
#           window_zs_fltr0_ntrend0_edges_stop[
#               ~ window_ys_fltr0_ntrend0_mask_edges_stop
#           ] = np.nan
#           #f 1 and prt:
#           #   print fmt0(just1)[1:] % (
#           #       "window_zs_fltr0_ntrend0_edges_stop.shape",
#           #       window_zs_fltr0_ntrend0_edges_stop.shape)

#           window_zs_fltr0_ntrend0_edges_stop_rois = (
#               window_zs_fltr0_ntrend0_edges_stop.copy())
#           window_zs_fltr0_ntrend0_edges_stop_rois[
#               ~ window_xs_rois_mask] = np.nan

#           #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

#           np_roi_tow_xs_center = np.array(roi_tow_xs_center)
#           np_roi_tow_ys_mean_meast_idx = np.array(roi_tow_ys_mean_meast_idx)
#           np_roi_tow_starts_stops = np.array(roi_tow_starts_stops)

#           make_gallery03_tow_ends_placement_plot_xz(
#               number_of_profiles, indy, window_indy,
#               pd_src_us_tow_diff_row, tow_diff_names, window_xs, window_ys,
#               window_zs_ntrend0, window_zs_fltr0_ntrend0,
#               window_zs_fltr0_ntrend0_edges_start,
#               window_ys_fltr0_ntrend0_edges_start,
#               window_zs_fltr0_ntrend0_edges_stop,
#               window_ys_fltr0_ntrend0_edges_stop,
#               window_zs_fltr0_ntrend0_edges_stop_rois,
#               window_zs_fltr0_ntrend0_edges_start_rois,
#               np_roi_tow_xs_center, np_roi_tow_ys_mean_meast_idx,
#               np_roi_tow_starts_stops,
#           ) if 1 else None
#           #one if 0 else sys.exit()

#           #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

#           window_dzdys_fltr0_ntrend0_mask_ge0 = (
#               window_dzdys_fltr0_ntrend0 > 0.)
#           window_dzdys_fltr0_ntrend0_ge0 = window_dzdys_fltr0_ntrend0.copy()
#           window_dzdys_fltr0_ntrend0_ge0[
#               ~ window_dzdys_fltr0_ntrend0_mask_ge0] = 0.
#           #f 1 and prt:
#           #   print fmt0(just1)[0:] % (
#           #       "window_dzdys_fltr0_ntrend0_ge0.shape",
#           #       window_dzdys_fltr0_ntrend0_ge0.shape)

#           # window_dzdys_fltr0_ntrend0_mask_edges_start
#           window_dzdys_fltr0_ntrend0_edges_start = (
#               window_dzdys_fltr0_ntrend0.copy())
#           window_dzdys_fltr0_ntrend0_edges_start[
#               ~ window_dzdys_fltr0_ntrend0_mask_edges_start
#           ] = np.nan
#           #f 1 and prt:
#           #   print fmt0(just1)[0:] % (
#           #       "window_dzdys_fltr0_ntrend0_edges_start.shape",
#           #       window_dzdys_fltr0_ntrend0_edges_start.shape)
#           #f 0 and prt:
#           #   print fmt1(just1)[1:] % (
#           #       "window_dzdys_fltr0_ntrend0_edges_start[:, 0: 21]",
#           #       window_dzdys_fltr0_ntrend0_edges_start[:, 0: 21])
#           #   print fmt1(just1)[1:] % (
#           #       "window_dzdys_fltr0_ntrend0_edges_start[:, 500: 541]",
#           #       window_dzdys_fltr0_ntrend0_edges_start[:, 510: 531])

#           make_gallery03_tow_ends_placement_plot_yz_start(
#               number_of_profiles, indy, window_indy,
#               pd_src_us_tow_diff_row, tow_diff_names, window_ys,
#               window_zs_fltr0_ntrend0, window_ys_midp,
#               window_dzdys_fltr0_ntrend0, window_dzdys_fltr0_ntrend0_ge0,
#               window_dzdys_fltr0_ntrend0_edges_start,
#               window_zs_fltr0_ntrend0_edges_start, dzdys_threshold,
#           ) if 1 else None
#           #one if 0 else sys.exit()

#           #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

#           window_dzdys_fltr0_ntrend0_mask_le0 = (
#               window_dzdys_fltr0_ntrend0 < 0.)
#           window_dzdys_fltr0_ntrend0_le0 = window_dzdys_fltr0_ntrend0.copy()
#           window_dzdys_fltr0_ntrend0_le0[
#               ~ window_dzdys_fltr0_ntrend0_mask_le0] = 0.
#           #f 1 and prt:
#           #   print fmt0(just1)[0:] % (
#           #       "window_dzdys_fltr0_ntrend0_le0.shape",
#           #       window_dzdys_fltr0_ntrend0_le0.shape)

#           # window_dzdys_fltr0_ntrend0_mask_edges_stop
#           window_dzdys_fltr0_ntrend0_edges_stop = (
#               window_dzdys_fltr0_ntrend0.copy())
#           window_dzdys_fltr0_ntrend0_edges_stop[
#               ~ window_dzdys_fltr0_ntrend0_mask_edges_stop
#           ] = np.nan
#           #f 1 and prt:
#           #   print fmt0(just1)[0:] % (
#           #       "window_dzdys_fltr0_ntrend0_edges_stop.shape",
#           #       window_dzdys_fltr0_ntrend0_edges_stop.shape)
#           #f 0 and prt:
#           #   print fmt1(just1)[1:] % (
#           #       "window_dzdys_fltr0_ntrend0_edges_stop[:, 0: 21]",
#           #       window_dzdys_fltr0_ntrend0_edges_stop[:, 0: 21])
#           #   print fmt1(just1)[1:] % (
#           #       "window_dzdys_fltr0_ntrend0_edges_stop[:, 500: 541]",
#           #       window_dzdys_fltr0_ntrend0_edges_stop[:, 510: 531])

#           make_gallery03_tow_ends_placement_plot_yz_stop(
#               number_of_profiles, indy, window_indy,
#               pd_src_us_tow_diff_row, tow_diff_names, window_ys,
#               window_zs_fltr0_ntrend0, window_ys_midp,
#               window_dzdys_fltr0_ntrend0, window_dzdys_fltr0_ntrend0_le0,
#               window_dzdys_fltr0_ntrend0_edges_stop,
#               window_zs_fltr0_ntrend0_edges_stop, dzdys_threshold,
#           ) if 1 else None
#           None if 1 else sys.exit()

#%          #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        None if 1 else sys.exit()
        #break
        #f meast_id >= 286:
        #   break

    #f 1 and prt:
    #   rw, cl = len(pd_src_us_tow_diff), 21
    #   print fmt1(just1)[0:] % ("pd_results_ends.iloc[:%s,:%i]" % (
    #       rw, cl), pd_results_ends.iloc[:rw, :cl])

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return pd_results_ends

# (above) defs ... for analyzing tow ends placement2
#zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz
#zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz

# (above) defs ... for analyzing tow ends placement
#ZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ
#ZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ
# (below) defs ... for analyzing gaps between tows


def segment_edges_and_flats(number_of_profiles, indy, fltr_dzdxs,
dzdxs_threshold, nnan_xs_midp, nnan_xs):
    """
    Returns a Pandas Series containing masks and labels of laser profile
    edges and flats.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    prt__ = False if 1 else True  # def print switch
    mult_str = '--- '
    def_str = 'segment_edges_and_flats'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    #f 1 and prt:
    #   print "\n%s" % ('::::' * 40)

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

##  fltr_dzdxs_mask_edges_pos = fltr_dzdxs >= dzdxs_threshold
##  # to increase the edges ...
##  fltr_dzdxs_mask_edges_pos = ndi.binary_dilation(fltr_dzdxs_mask_edges_pos)
##  #f 1 and prt:
##  #   print fmt0(just2)[0:] % ("fltr_dzdxs_mask_edges_pos.shape",
##  #       fltr_dzdxs_mask_edges_pos.shape)
##  if 1 and prt:
##      print fmt0(just2)[0:] % ("np.where(fltr_dzdxs_mask_edges_pos)[0]",
##          np.where(fltr_dzdxs_mask_edges_pos)[0])

    fltr_dzdxs_mask_ge0 = fltr_dzdxs >= 0.
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_dzdxs_mask_ge0.shape",
    #       fltr_dzdxs_mask_ge0.shape)
    #   print fmt0(just2)[1:] % ("np.where(fltr_dzdxs_mask_ge0)[0].shape",
    #       np.where(fltr_dzdxs_mask_ge0)[0].shape)

    fltr_dzdxs_mask_edges_pos = fltr_dzdxs >= dzdxs_threshold
    # to increase the edges ...
    fltr_dzdxs_mask_edges_pos = ndi.binary_dilation(fltr_dzdxs_mask_edges_pos)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("(init) fltr_dzdxs_mask_edges_pos.shape",
    #       fltr_dzdxs_mask_edges_pos.shape)
    #   print fmt0(just2)[1:] % (
    #       "(init) np.where(fltr_dzdxs_mask_edges_pos)[0].shape",
    #       np.where(fltr_dzdxs_mask_edges_pos)[0].shape)

    fltr_dzdxs_mask_edges_pos = (
        fltr_dzdxs_mask_edges_pos & fltr_dzdxs_mask_ge0)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("(updt) fltr_dzdxs_mask_edges_pos.shape",
    #       fltr_dzdxs_mask_edges_pos.shape)
    #   print fmt0(just2)[1:] % (
    #       "(updt) np.where(fltr_dzdxs_mask_edges_pos)[0].shape",
    #       np.where(fltr_dzdxs_mask_edges_pos)[0].shape)

    #one if 1 else sys.exit()

    fltr_dzdxs_label_edges_pos, _ = ndi.label(fltr_dzdxs_mask_edges_pos)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("fltr_dzdxs_label_edges_pos.shape",
    #       fltr_dzdxs_label_edges_pos.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("np.unique(fltr_dzdxs_label_edges_pos)",
    #       np.unique(fltr_dzdxs_label_edges_pos))

    #f 1 and prt__:
    #   print fmt0(just2)[0:] % ("dzdxs edges pos", "")
    #   for uniq_label in np.unique(fltr_dzdxs_label_edges_pos):
    #       if uniq_label > 0:
    #           fltr_dzdxs_uniq_label_edges_pos_where = np.where(
    #               fltr_dzdxs_label_edges_pos == uniq_label)[0]
    #           fltr_dzdxs_uniq_label_edges_pos_xs_midp = nnan_xs_midp[
    #               fltr_dzdxs_uniq_label_edges_pos_where]
    #           fltr_dzdxs_uniq_label_edges_pos_xs_midp_mid = (
    #               fltr_dzdxs_uniq_label_edges_pos_xs_midp[0] +
    #               fltr_dzdxs_uniq_label_edges_pos_xs_midp[-1]) / 2.
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... where" %
    #               uniq_label, fltr_dzdxs_uniq_label_edges_pos_where)
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... xs_midp" %
    #               uniq_label, fltr_dzdxs_uniq_label_edges_pos_xs_midp)
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... xs_midp_mid" %
    #               uniq_label, fltr_dzdxs_uniq_label_edges_pos_xs_midp_mid)
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

##  fltr_dzdxs_mask_edges_neg = fltr_dzdxs <= -dzdxs_threshold
##  # to increase the edges ...
##  fltr_dzdxs_mask_edges_neg = ndi.binary_dilation(fltr_dzdxs_mask_edges_neg)
##  #f 1 and prt:
##  #   print fmt0(just2)[0:] % ("fltr_dzdxs_mask_edges_neg.shape",
##  #       fltr_dzdxs_mask_edges_neg.shape)
##  if 1 and prt:
##      print fmt0(just2)[0:] % ("np.where(fltr_dzdxs_mask_edges_neg)[0]",
##          np.where(fltr_dzdxs_mask_edges_neg)[0])

    fltr_dzdxs_mask_le0 = fltr_dzdxs <= 0.
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_dzdxs_mask_le0.shape",
    #       fltr_dzdxs_mask_le0.shape)
    #   print fmt0(just2)[1:] % ("np.where(fltr_dzdxs_mask_le0)[0].shape",
    #       np.where(fltr_dzdxs_mask_le0)[0].shape)

    fltr_dzdxs_mask_edges_neg = fltr_dzdxs <= -dzdxs_threshold
    # to increase the edges ...
    fltr_dzdxs_mask_edges_neg = ndi.binary_dilation(fltr_dzdxs_mask_edges_neg)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("(init) fltr_dzdxs_mask_edges_neg.shape",
    #       fltr_dzdxs_mask_edges_neg.shape)
    #   print fmt0(just2)[1:] % (
    #       "(init) np.where(fltr_dzdxs_mask_edges_neg)[0].shape",
    #       np.where(fltr_dzdxs_mask_edges_neg)[0].shape)

    fltr_dzdxs_mask_edges_neg = (
        fltr_dzdxs_mask_edges_neg & fltr_dzdxs_mask_le0)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("(updt) fltr_dzdxs_mask_edges_neg.shape",
    #       fltr_dzdxs_mask_edges_neg.shape)
    #   print fmt0(just2)[1:] % (
    #       "(updt) np.where(fltr_dzdxs_mask_edges_neg)[0].shape",
    #       np.where(fltr_dzdxs_mask_edges_neg)[0].shape)

    #one if 0 else sys.exit()

    fltr_dzdxs_label_edges_neg, _ = ndi.label(fltr_dzdxs_mask_edges_neg)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("fltr_dzdxs_label_edges_neg.shape",
    #       fltr_dzdxs_label_edges_neg.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("np.unique(fltr_dzdxs_label_edges_neg)",
    #       np.unique(fltr_dzdxs_label_edges_neg))

    #f 1 and prt__:
    #   print fmt0(just2)[0:] % ("dzdxs edges neg", "")
    #   for uniq_label in np.unique(fltr_dzdxs_label_edges_neg):
    #       if uniq_label > 0:
    #           fltr_dzdxs_uniq_label_edges_neg_where = np.where(
    #               fltr_dzdxs_label_edges_neg == uniq_label)[0]
    #           fltr_dzdxs_uniq_label_edges_neg_xs_midp = nnan_xs_midp[
    #               fltr_dzdxs_uniq_label_edges_neg_where]
    #           fltr_dzdxs_uniq_label_edges_neg_xs_midp_mid = (
    #               fltr_dzdxs_uniq_label_edges_neg_xs_midp[0] +
    #               fltr_dzdxs_uniq_label_edges_neg_xs_midp[-1]) / 2.
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... where" %
    #               uniq_label, fltr_dzdxs_uniq_label_edges_neg_where)
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... xs_midp" %
    #               uniq_label, fltr_dzdxs_uniq_label_edges_neg_xs_midp)
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... xs_midp_mid" %
    #               uniq_label, fltr_dzdxs_uniq_label_edges_neg_xs_midp_mid)
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    fltr_dzdxs_mask_edges = (
        fltr_dzdxs_mask_edges_pos | fltr_dzdxs_mask_edges_neg)
    #
    fltr_dzdxs_label_edges, _ = ndi.label(fltr_dzdxs_mask_edges)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_dzdxs_mask_edges.shape",
    #       fltr_dzdxs_mask_edges.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("fltr_dzdxs_label_edges.shape",
    #       fltr_dzdxs_label_edges.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("np.unique(fltr_dzdxs_label_edges)",
    #       np.unique(fltr_dzdxs_label_edges))

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    fltr_dzdxs_mask_flats = ~ fltr_dzdxs_mask_edges
    #
    fltr_dzdxs_label_flats, _ = ndi.label(fltr_dzdxs_mask_flats)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_dzdxs_mask_flats.shape",
    #       fltr_dzdxs_mask_flats.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("fltr_dzdxs_label_flats.shape",
    #       fltr_dzdxs_label_flats.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("np.unique(fltr_dzdxs_label_flats)",
    #       np.unique(fltr_dzdxs_label_flats))
    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #f 1 and prt:
    #   print "\n%s" % ('....' * 40)

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    fltr_zs_mask_edges_pos = np.zeros(fltr_dzdxs.shape[0] + 1).astype(np.bool)
    fltr_zs_mask_edges_pos[:-1] = (
        fltr_zs_mask_edges_pos[:-1] | fltr_dzdxs_mask_edges_pos)
    fltr_zs_mask_edges_pos[1:] = (
        fltr_zs_mask_edges_pos[1:] | fltr_dzdxs_mask_edges_pos)
    #
    fltr_zs_label_edges_pos, _ = ndi.label(fltr_zs_mask_edges_pos)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_zs_mask_edges_pos.shape",
    #       fltr_zs_mask_edges_pos.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("fltr_zs_label_edges_pos.shape",
    #       fltr_zs_label_edges_pos.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("np.unique(fltr_zs_label_edges_pos)",
    #       np.unique(fltr_zs_label_edges_pos))

    #f 1 and prt__:
    #   print fmt0(just2)[0:] % ("zs edges pos", "")
    #   for uniq_label in np.unique(fltr_zs_label_edges_pos):
    #       if uniq_label > 0:
    #           fltr_zs_uniq_label_edges_pos_where = np.where(
    #               fltr_zs_label_edges_pos == uniq_label)[0]
    #           fltr_zs_uniq_label_edges_pos_xs = nnan_xs[
    #               fltr_zs_uniq_label_edges_pos_where]
    #           fltr_zs_uniq_label_edges_pos_xs_mid = (
    #               fltr_zs_uniq_label_edges_pos_xs[0] +
    #               fltr_zs_uniq_label_edges_pos_xs[-1]) / 2.
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... where" %
    #               uniq_label, fltr_zs_uniq_label_edges_pos_where)
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... xs" %
    #               uniq_label, fltr_zs_uniq_label_edges_pos_xs)
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... xs_mid" %
    #               uniq_label, fltr_zs_uniq_label_edges_pos_xs_mid)
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    fltr_zs_mask_edges_neg = np.zeros(fltr_dzdxs.shape[0] + 1).astype(np.bool)
    fltr_zs_mask_edges_neg[:-1] = (
        fltr_zs_mask_edges_neg[:-1] | fltr_dzdxs_mask_edges_neg)
    fltr_zs_mask_edges_neg[1:] = (
        fltr_zs_mask_edges_neg[1:] | fltr_dzdxs_mask_edges_neg)
    #
    fltr_zs_label_edges_neg, _ = ndi.label(fltr_zs_mask_edges_neg)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_zs_mask_edges_neg.shape",
    #       fltr_zs_mask_edges_neg.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("fltr_zs_label_edges_neg.shape",
    #       fltr_zs_label_edges_neg.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("np.unique(fltr_zs_label_edges_neg)",
    #       np.unique(fltr_zs_label_edges_neg))

    #f 1 and prt__:
    #   print fmt0(just2)[0:] % ("zs edges neg", "")
    #   for uniq_label in np.unique(fltr_zs_label_edges_neg):
    #       if uniq_label > 0:
    #           fltr_zs_uniq_label_edges_neg_where = np.where(
    #               fltr_zs_label_edges_neg == uniq_label)[0]
    #           fltr_zs_uniq_label_edges_neg_xs = nnan_xs[
    #               fltr_zs_uniq_label_edges_neg_where]
    #           fltr_zs_uniq_label_edges_neg_xs_mid = (
    #               fltr_zs_uniq_label_edges_neg_xs[0] +
    #               fltr_zs_uniq_label_edges_neg_xs[-1]) / 2.
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... where" %
    #               uniq_label, fltr_zs_uniq_label_edges_neg_where)
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... xs" %
    #               uniq_label, fltr_zs_uniq_label_edges_neg_xs)
    #           print fmt0(just2)[1:] % ("uniq_label %2i ... xs_mid" %
    #               uniq_label, fltr_zs_uniq_label_edges_neg_xs_mid)
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    fltr_zs_mask_edges = np.zeros(fltr_dzdxs.shape[0] + 1).astype(np.bool)
    fltr_zs_mask_edges[:-1] = (fltr_zs_mask_edges[:-1] | fltr_dzdxs_mask_edges)
    fltr_zs_mask_edges[1:] = (fltr_zs_mask_edges[1:] | fltr_dzdxs_mask_edges)
    #
    fltr_zs_label_edges, _ = ndi.label(fltr_zs_mask_edges)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_zs_mask_edges.shape",
    #       fltr_zs_mask_edges.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("fltr_zs_label_edges.shape",
    #       fltr_zs_label_edges.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("np.unique(fltr_zs_label_edges)",
    #       np.unique(fltr_zs_label_edges))
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    fltr_zs_mask_flats = np.zeros(fltr_dzdxs.shape[0] + 1).astype(np.bool)
    fltr_zs_mask_flats[:-1] = (fltr_zs_mask_flats[:-1] | fltr_dzdxs_mask_flats)
    fltr_zs_mask_flats[1:] = (fltr_zs_mask_flats[1:] | fltr_dzdxs_mask_flats)
    #
    fltr_zs_label_flats, _ = ndi.label(fltr_zs_mask_flats)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_zs_mask_flats.shape",
    #       fltr_zs_mask_flats.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("fltr_zs_label_flats.shape",
    #       fltr_zs_label_flats.shape)
    #f 1 and prt:
    #   print fmt0(just2)[1:] % ("np.unique(fltr_zs_label_flats)",
    #       np.unique(fltr_zs_label_flats))
    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #f 1 and prt:
    #   print "\n%s" % ('::::' * 40)

    segment_edges_and_flats_ps = pd.Series({
        #
        'fltr_dzdxs_mask_edges_pos': fltr_dzdxs_mask_edges_pos,
        'fltr_dzdxs_mask_edges_neg': fltr_dzdxs_mask_edges_neg,
        'fltr_dzdxs_mask_edges': fltr_dzdxs_mask_edges,
        'fltr_dzdxs_mask_flats': fltr_dzdxs_mask_flats,
        #
        'fltr_dzdxs_label_edges_pos': fltr_dzdxs_label_edges_pos,
        'fltr_dzdxs_label_edges_neg': fltr_dzdxs_label_edges_neg,
        'fltr_dzdxs_label_edges': fltr_dzdxs_label_edges,
        'fltr_dzdxs_label_flats': fltr_dzdxs_label_flats,
        #
        'fltr_zs_mask_edges_pos': fltr_zs_mask_edges_pos,
        'fltr_zs_mask_edges_neg': fltr_zs_mask_edges_neg,
        'fltr_zs_mask_edges': fltr_zs_mask_edges,
        'fltr_zs_mask_flats': fltr_zs_mask_flats,
        #
        'fltr_zs_label_edges_pos': fltr_zs_label_edges_pos,
        'fltr_zs_label_edges_neg': fltr_zs_label_edges_neg,
        'fltr_zs_label_edges': fltr_zs_label_edges,
        'fltr_zs_label_flats': fltr_zs_label_flats,
    })

    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("sorted(segment_edges_and_flats_ps.keys())",
    #       prt_list(sorted(segment_edges_and_flats_ps.keys()), 0))
    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #f 1 and indy == 11:  # quick plot ...
    if 0:  # quick plot ...
        if 1 and prt__:
            print fmt0(just2)[0:] % ("nnan_xs.shape", nnan_xs.shape)
            print fmt0(just2)[1:] % ("fltr_zs_mask_edges_pos.shape",
                fltr_zs_mask_edges_pos.shape)
            print fmt0(just2)[1:] % ("fltr_zs_mask_edges_neg.shape",
                fltr_zs_mask_edges_neg.shape)
        if 1 and prt__:
            print fmt0(just2)[0:] % ("nnan_xs_midp.shape", nnan_xs_midp.shape)
            print fmt0(just2)[1:] % ("fltr_dzdxs_mask_edges_pos.shape",
                fltr_dzdxs_mask_edges_pos.shape)
            print fmt0(just2)[1:] % ("fltr_dzdxs_mask_edges_neg.shape",
                fltr_dzdxs_mask_edges_neg.shape)
        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
        axvline_ymin = 0.01
        axvline_ymax = 0.99
        fig = plt.gcf()
        fig.suptitle("%s:\n%s" % ("Laser Profile (X, Z) %i of %i" %
            ((indy + 1), number_of_profiles), ngn.job_zs_csv))
        fig.set_size_inches(16, 12, forward=True)  # default is (8, 6)
        gs = mpl.gridspec.GridSpec(2, 1)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        #x1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Z Coordinate')
        ax1_set_ylim = 1.6
        ax1.set_ylim((-ax1_set_ylim, ax1_set_ylim))
        #
        # fltr_zs_arange = np.arange(fltr_dzdxs.shape[0] + 1) + 1.
        # #
        # ax1.plot(fltr_zs_arange, 1. * fltr_zs_mask_edges_pos,
        #     'r,-', mec='none', label="fltr_zs_mask_edges_pos")
        # ax1.plot(fltr_zs_arange, -1. * fltr_zs_mask_edges_neg,
        #     'm,-', mec='none', label="fltr_zs_mask_edges_neg")
        # ax1.plot(fltr_zs_arange, 0. * fltr_zs_arange,
        #     'c,-', mec='none')
        #
        ax1.plot(nnan_xs, 1. * fltr_zs_mask_edges_pos,
            'r,-', mec='none', label="fltr_zs_mask_edges_pos")
        ax1.plot(nnan_xs, -1. * fltr_zs_mask_edges_neg,
            'm,-', mec='none', label="fltr_zs_mask_edges_neg")
        ax1.plot(nnan_xs, 0. * nnan_xs,
            'c,-', mec='none')
        #
        for i, tow_center_xref in enumerate(ngn.tow_center_xrefs):
            #
            ax1_axvline_ls = (
                'dashed' if i == 0 or (i + 1) == len(ngn.tow_center_xrefs)
                else 'dotted')
            ax1.axvline(x=tow_center_xref, ymin=axvline_ymin,
                ymax=axvline_ymax, c='y', ls=ax1_axvline_ls, lw=2.)
            #x1.text(tow_center_xref, ax1_text_y1, 'Strip\n%i' % i,
            #   fontsize=text_fontsize, fontweight='bold',
            #   ha='center', va='center')
        #
        ax1.legend(
            loc=8,
            ncol=2,
            numpoints=1,
            markerscale=1.,
            prop={'size': 9.2, 'weight': 'bold'}
        ) if 1 else None
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('dZdX Coordinate')
        ax2_set_ylim = 1.6
        ax2.set_ylim((-ax2_set_ylim, ax2_set_ylim))
        #
        # fltr_dzdxs_arange = np.arange(fltr_dzdxs.shape[0]) + 1.
        # #
        # ax2.plot(fltr_dzdxs_arange, 1. * fltr_dzdxs_mask_edges_pos,
        #     'r,-', mec='none', label="fltr_dzdxs_mask_edges_pos")
        # ax2.plot(fltr_dzdxs_arange, -1. * fltr_dzdxs_mask_edges_neg,
        #     'm,-', mec='none', label="fltr_dzdxs_mask_edges_neg")
        # ax2.plot(fltr_dzdxs_arange, 0. * fltr_dzdxs_arange,
        #     'c,-', mec='none')
        #
        ax2.plot(nnan_xs_midp, 1. * fltr_dzdxs_mask_edges_pos,
            'r,-', mec='none', label="fltr_dzdxs_mask_edges_pos")
        ax2.plot(nnan_xs_midp, -1. * fltr_dzdxs_mask_edges_neg,
            'm,-', mec='none', label="fltr_dzdxs_mask_edges_neg")
        ax2.plot(nnan_xs_midp, 0. * nnan_xs_midp,
            'c,-', mec='none')
        #
        for i, tow_center_xref in enumerate(ngn.tow_center_xrefs):
            #
            ax2_axvline_ls = (
                'dashed' if i == 0 or (i + 1) == len(ngn.tow_center_xrefs)
                else 'dotted')
            ax2.axvline(x=tow_center_xref, ymin=axvline_ymin,
                ymax=axvline_ymax, c='y', ls=ax2_axvline_ls, lw=2.)
            #x2.text(tow_center_xref, ax2_text_y1, 'Strip\n%i' % i,
            #   fontsize=text_fontsize, fontweight='bold',
            #   ha='center', va='center')
        #
        ax2.legend(
            loc=8,
            ncol=2,
            numpoints=1,
            markerscale=1.,
            prop={'size': 9.2, 'weight': 'bold'}
        ) if 1 else None
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        plt.show()
        plt.close()
        None if 0 else sys.exit()
        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return segment_edges_and_flats_ps


def make_gallery02_roi_profile_plot(indy, gap_idx, gap_id, nnan_xs_roi,
fltr_zs_roi, fltr_zs_roi_edges_neg, fltr_zs_roi_edges_pos, fltr_zs_roi_flats,
nnan_xs_midp_roi, fltr_dzdxs_roi,
fltr_dzdxs_roi_edges_neg, fltr_dzdxs_roi_edges_pos, fltr_dzdxs_roi_flats,
ps_gap):
    """
    Makes a plot of z-coordinate values and differences for a laser profile
    region of interest.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    #rt = True if 1 and gap_ps.gap_id == (7, 8) else False
    prt_ = prt
    prt__ = False if 1 else True  # def print switch
    mult_str = '--- '
    def_str = 'make_gallery02_profile_plot'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fig = plt.gcf()
    fig.set_size_inches(8, 6, forward=True)  # default is (8, 6)
    fig.suptitle("%s, %s Region of Interest Profile (X, Z)" % (
        "Measurement %i" % (indy + 1), 'Gap %s' % gap_id), fontsize=11)
    gs = mpl.gridspec.GridSpec(2, 1)

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1 = plt.subplot(gs[0])
    ax1.set_title(ngn.job_zs_csv, y=1.00, fontsize=11)
    #x1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Z Coordinate')
    ax1.xaxis.grid(True)

    ax1_ylim_mgn_min = 1.8
    ax1_ylim_mgn_max = 1.8
    nanmax = np.nanmax(fltr_zs_roi)
    nanmin = np.nanmin(fltr_zs_roi)
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn_min = nanrng * ax1_ylim_mgn_min
    nanmgn_max = nanrng * ax1_ylim_mgn_max
    ax1.set_ylim((nanmid - nanmgn_min, nanmid + nanmgn_max))
    if 0 and prt:
        print fmt0(just1)[0:] % ("[nanmin, nanmax]", [nanmin, nanmax])
        print fmt0(just1)[1:] % ("ax1.get_ylim()", ax1.get_ylim())

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    ms_roi_flats = 8.
    ax1.plot(
        nnan_xs_roi, fltr_zs_roi_flats,
        'co', mec='none', ms=ms_roi_flats, label="roi zs flat(s)"
    )
    ax1.plot(
        nnan_xs_roi, fltr_zs_roi,
        'b-', mec='none', label="roi zs profile"
    )
    ax1.plot(
        nnan_xs_roi, fltr_zs_roi_edges_neg,
        'mo-', mec='none', label="roi zs neg edge(s)"
    )
    ax1.plot(
        nnan_xs_roi, fltr_zs_roi_edges_pos,
        'ro-', mec='none', label="roi zs pos edge(s)"
    )

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    ha_edge_neg, ha_edge_pos = 'right', 'left'  # default alignments (for gap)
    ha_edge_neg_xofst, ha_edge_pos_xofst = -0.15, 0.15
    if (not np.isnan(ps_gap.edge_neg_x_midp) and
    not np.isnan(ps_gap.edge_pos_x_midp) and
    ps_gap.edge_pos_x_midp < ps_gap.edge_neg_x_midp):
        ha_edge_neg, ha_edge_pos = 'left', 'right'  # alignments for lap
        ha_edge_neg_xofst, ha_edge_pos_xofst = 0.15, -0.15

    ax1.plot(
        [ps_gap.edge_neg_x_midp], [ps_gap.edge_neg_z],
        'o', ms=10., mew=2., mfc='none',
    ) if ~ np.isnan(ps_gap.edge_neg_dzdx) else None
    ax1.text(
        ps_gap.edge_neg_x_midp + ha_edge_neg_xofst, ps_gap.edge_neg_z,
        'Z (mm): %.3f' % ps_gap.edge_neg_z,
        fontsize=8., fontweight='bold', ha=ha_edge_neg, va='center'
    ) if ~ np.isnan(ps_gap.edge_neg_dzdx) else None
    #
    ax1.plot(
        [ps_gap.edge_pos_x_midp], [ps_gap.edge_pos_z],
        'o', ms=10., mew=2., mfc='none',
    ) if ~ np.isnan(ps_gap.edge_pos_dzdx) else None
    ax1.text(
        ps_gap.edge_pos_x_midp + ha_edge_pos_xofst, ps_gap.edge_pos_z,
        'Z (mm): %.3f' % ps_gap.edge_pos_z,
        fontsize=8., fontweight='bold', ha=ha_edge_pos, va='center'
    ) if ~ np.isnan(ps_gap.edge_pos_dzdx) else None

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
    # constructions for a gap ...

    if not np.isnan(ps_gap.gap_width) and ps_gap.gap_width > 0.:

        ax1.plot(
            [ps_gap.edge_neg_x_midp, ps_gap.edge_pos_x_midp],
            [ps_gap.edge_neg_z, ps_gap.edge_pos_z],
            'o-', color='gray', mec='none',
            #  ms=10., mew=2., mfc='none',
        )
        ax1.text(
            ps_gap.gap_flat_x, ps_gap.gap_center_z + 0.15 * nanmgn_max,
            'Gap Width (mm): %.3f' % ps_gap.gap_width,
            fontsize=8., fontweight='bold', ha='center', va='bottom'
        )

        ax1.plot(
            [ps_gap.gap_flat_x], [ps_gap.gap_flat_z],
            'o', ms=10., mew=2., mfc='none',
        )
        ax1.text(
            ps_gap.gap_flat_x, ps_gap.gap_flat_z - 0.10 * nanmgn_min,
            'X, Z (mm): %.3f, %.3f' % (ps_gap.gap_flat_x, ps_gap.gap_flat_z),
            fontsize=8., fontweight='bold', ha='center', va='top'
        )
        #f 1 and prt:
        #   print fmt1(just1)[0:] % ("ps_gap", ps_gap)

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
    # constructions for a lap ...

    elif not np.isnan(ps_gap.gap_width) and ps_gap.gap_width < 0.:

        if ps_gap.edge_pos_dzdx > -ps_gap.edge_neg_dzdx:
            ax1_str_lap = 'Right Tow Over Left Tow'
        elif ps_gap.edge_pos_dzdx < -ps_gap.edge_neg_dzdx:
            ax1_str_lap = 'Left Tow Over Right Tow'
        else:
            ax1_str_lap = None

        ax1_trans1 = mpl.transforms.blended_transform_factory(
            ax1.transAxes, ax1.transAxes)
        ax1.text(
            0.5, 0.95,
            ax1_str_lap,
            transform=ax1_trans1,
            fontsize=8., fontweight='bold', ha='center', va='top',
        ) if ax1_str_lap is not None else None

        ax1.plot(
            [ps_gap.edge_neg_x_midp, ps_gap.edge_pos_x_midp],
            [ps_gap.edge_neg_z, ps_gap.edge_pos_z],
            'o-', color='gray', mec='none',
            #  ms=10., mew=2., mfc='none',
        )
        ax1.text(
            ps_gap.gap_flat_x, ps_gap.gap_center_z - 0.15 * nanmgn_max,
            'Gap Width (mm): %.3f' % ps_gap.gap_width,
            fontsize=8., fontweight='bold', ha='center', va='bottom'
        )

        ax1.plot(
            [ps_gap.gap_flat_x], [ps_gap.gap_flat_z],
            'o', ms=10., mew=2., mfc='none',
        )
        ax1.text(
            ps_gap.gap_flat_x, ps_gap.gap_flat_z + 0.10 * nanmgn_min,
            'X, Z (mm): %.3f, %.3f' % (ps_gap.gap_flat_x, ps_gap.gap_flat_z),
            fontsize=8., fontweight='bold', ha='center', va='top'
        )
        #f 1 and prt:
        #   print fmt1(just1)[0:] % ("ps_gap", ps_gap)

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    ax1.legend(
        loc=8,
        ncol=4,
        numpoints=1,
        markerscale=1.,
        prop={'size': 7.0, 'weight': 'bold'}
    ) if 1 else None
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.set_title('dZdX Thresholds (mm):  +- %.3f' % ngn.dzdxs_threshold,
        y=1.00, fontsize=11)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('dZdX Coordinate')
    ax2.xaxis.grid(True)

    ax2_xlim_mgn = 1.10
    nanmax = np.nanmax(nnan_xs_roi)
    nanmin = np.nanmin(nnan_xs_roi)
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ax2_xlim_mgn
    ax2.set_xlim((nanmid - nanmgn, nanmid + nanmgn))
    if 0 and prt:
        print fmt0(just1)[0:] % ("[nanmin, nanmax]", [nanmin, nanmax])
        print fmt0(just1)[1:] % ("ax2.get_xlim()", ax2.get_xlim())

    ax2_ylim_mgn = 2.7
    nanmax = np.nanmax(np.append(np.abs(fltr_dzdxs_roi), ngn.dzdxs_threshold))
    nanmin = -nanmax
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ax2_ylim_mgn
    ax2.set_ylim((nanmid - nanmgn, nanmid + nanmgn))
    if 0 and prt:
        print fmt0(just1)[0:] % ("[nanmin, nanmax]", [nanmin, nanmax])
        print fmt0(just1)[1:] % ("ax2.get_ylim()", ax2.get_ylim())

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    axhline_xmin, axhline_xmax = 0.05, 0.95
    ax2.axhline(y=ngn.dzdxs_threshold,
        xmin=axhline_xmin, xmax=axhline_xmax, c='k', ls='dotted', lw=2.)
    ax2.axhline(y=-ngn.dzdxs_threshold,
        xmin=axhline_xmin, xmax=axhline_xmax, c='k', ls='dotted', lw=2.)

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    ax2.plot(
        nnan_xs_midp_roi, fltr_dzdxs_roi_flats,
        'co', mec='none', ms=ms_roi_flats, label="roi dzdxs flat(s)"
    )
    ax2.plot(
        nnan_xs_midp_roi, fltr_dzdxs_roi,
        'b-', mec='none', label="roi dzdxs profile"
    )
    ax2.plot(
        nnan_xs_midp_roi, fltr_dzdxs_roi_edges_neg,
        'mo-', mec='none', label="roi dzdxs neg edge(s)"
    )
    ax2.plot(
        nnan_xs_midp_roi, fltr_dzdxs_roi_edges_pos,
        'ro-', mec='none', label="roi dzdxs pos edge(s)"
    )

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

    ax2.plot(
        [ps_gap.edge_neg_x_midp], [ps_gap.edge_neg_dzdx],
        'o', ms=10., mew=2., mfc='none',
    ) if ~ np.isnan(ps_gap.edge_neg_dzdx) else None
    ax2.text(
        ps_gap.edge_neg_x_midp + 0.15, ps_gap.edge_neg_dzdx,
        'dZdX (mm): %.3f\nX (mm): %.3f\nNnan Idx: %i\nProfile Idx: %i' % (
            ps_gap.edge_neg_dzdx, ps_gap.edge_neg_x_midp,
            ps_gap.edge_neg_x_midp_nnan_idx, ps_gap.edge_neg_x_midp_idx),
        fontsize=8., fontweight='bold', ha='left', va='top'
    ) if ~ np.isnan(ps_gap.edge_neg_dzdx) else None
    #
    ax2.plot(
        [ps_gap.edge_pos_x_midp], [ps_gap.edge_pos_dzdx],
        'o', ms=10., mew=2., mfc='none',
    ) if ~ np.isnan(ps_gap.edge_pos_dzdx) else None
    ax2.text(
        ps_gap.edge_pos_x_midp - 0.15, ps_gap.edge_pos_dzdx,
        'dZdX (mm): %.3f\nX (mm): %.3f\nNnan Idx: %i\nProfile Idx: %i' % (
            ps_gap.edge_pos_dzdx, ps_gap.edge_pos_x_midp,
            ps_gap.edge_pos_x_midp_nnan_idx, ps_gap.edge_pos_x_midp_idx),
        fontsize=8., fontweight='bold', ha='right', va='bottom'
    ) if ~ np.isnan(ps_gap.edge_pos_dzdx) else None

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    ax2.legend(
        loc=8,
        ncol=4,
        numpoints=1,
        markerscale=1.,
        prop={'size': 7.0, 'weight': 'bold'}
    ) if 1 else None
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    png_abspath = os.path.join(ngn.gallery02_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '').replace('.csv',
            '_meast_%.5i_gap_id_%s.png' % ((indy + 1), gap_id)))
    if 1 and prt:
        print fmt1(just1)[0:] % ("ngn.gallery02_absdir", ngn.gallery02_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    #
    png = png_abspath if 1 else None
    if png is not None:
        None if os.path.isdir(ngn.gallery02_absdir) else (
            os.makedirs(ngn.gallery02_absdir))
        plt.savefig(png)
    else:
        plt.show()
    plt.close()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


#ef find_nearest_value(array, value):
#   """
#   Returns the value in a Numpy array nearest a given value.
#   """
#   # http://stackoverflow.com/questions/2566412/ ...
#   # ... find-nearest-value-in-numpy-array
#   idx = (np.abs(array - value)).argmin()
#   return array[idx]


def find_nearest_value_index(array, value):
    """
    Returns the index(es) of values in a Numpy array nearest a given value.
    """
    # https://stackoverflow.com/questions/8914491/...
    # ...finding-the-nearest-value-and-return-the-index-of-array-in-python
    return (np.abs(array - value)).argmin()


def initialize_gap_rois_analyzed_counts():
    """
    Returns a set of counters set to zero.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    prt__ = False if 1 else True  # def print switch
    mult_str = '--- '
    def_str = 'initialize_gap_rois_analyzed_counts'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    ngn.count_analyzed_measts = 0
    ngn.count_analyzed_rois = 0
    #
    ngn.count_analyzed_rois_0_edge_neg = 0
    ngn.count_analyzed_rois_1_edge_neg = 0
    ngn.count_analyzed_rois_2_edge_neg = 0
    ngn.count_analyzed_rois_3_edge_neg = 0
    ngn.count_analyzed_rois_4_edge_neg = 0
    ngn.count_analyzed_rois_else_edge_neg = 0
    #
    ngn.count_analyzed_rois_0_edge_pos = 0
    ngn.count_analyzed_rois_1_edge_pos = 0
    ngn.count_analyzed_rois_2_edge_pos = 0
    ngn.count_analyzed_rois_3_edge_pos = 0
    ngn.count_analyzed_rois_4_edge_pos = 0
    ngn.count_analyzed_rois_else_edge_pos = 0
    #
    ngn.count_analyzed_rois_edges_none = 0
    ngn.count_analyzed_rois_edges_neg_only = 0
    ngn.count_analyzed_rois_edges_pos_only = 0
    ngn.count_analyzed_rois_edges_neg_pos = 0
    #
    ngn.count_analyzed_rois_edges_1_neg_lf_1_pos_rt = 0
    ngn.count_analyzed_rois_edges_1_pos_lf_1_neg_rt = 0
    ngn.count_analyzed_rois_edges_neg_pos_else = 0

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def increment_gap_rois_analyzed_counts(
    fltr_dzdxs_label_edges_neg_roi_uniqs_gt0,
    count_label_edges_neg, count_label_edges_neg_roi, gap_edge_neg_x_midp,
    fltr_dzdxs_label_edges_pos_roi_uniqs_gt0,
    count_label_edges_pos, count_label_edges_pos_roi, gap_edge_pos_x_midp,
):
    """
    Returns a set of conditionally incremented counters.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    prt__ = False if 1 else True  # def print switch
    mult_str = '--- '
    def_str = 'increment_gap_rois_analyzed_counts'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #== === === === === === === === === === === === === === === === === === ===

    ngn.count_analyzed_rois += 1

    #== === === === === === === === === === === === === === === === === === ===

    if len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 0:
        ngn.count_analyzed_rois_0_edge_neg += 1
    elif len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 1:
        ngn.count_analyzed_rois_1_edge_neg += 1
    elif len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 2:
        ngn.count_analyzed_rois_2_edge_neg += 1
    elif len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 3:
        ngn.count_analyzed_rois_3_edge_neg += 1
    elif len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 4:
        ngn.count_analyzed_rois_4_edge_neg += 1
    else:
        ngn.count_analyzed_rois_else_edge_neg += 1

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 0:
        ngn.count_analyzed_rois_0_edge_pos += 1
    elif len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 1:
        ngn.count_analyzed_rois_1_edge_pos += 1
    elif len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 2:
        ngn.count_analyzed_rois_2_edge_pos += 1
    elif len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 3:
        ngn.count_analyzed_rois_3_edge_pos += 1
    elif len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 4:
        ngn.count_analyzed_rois_4_edge_pos += 1
    else:
        ngn.count_analyzed_rois_else_edge_pos += 1

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if (len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 0 and
    len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 0):
        ngn.count_analyzed_rois_edges_none += 1

    elif (len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) > 0 and
    len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 0):
        ngn.count_analyzed_rois_edges_neg_only += 1

    elif (len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 0 and
    len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) > 0):
        ngn.count_analyzed_rois_edges_pos_only += 1

    elif (len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) > 0 and
    len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) > 0):
        ngn.count_analyzed_rois_edges_neg_pos += 1

    #== === === === === === === === === === === === === === === === === === ===

    if (len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) > 0 and
    len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) > 0):

        if (
            len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 1 and
            count_label_edges_neg == count_label_edges_neg_roi and
            len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 1 and
            count_label_edges_pos == count_label_edges_pos_roi and
            gap_edge_neg_x_midp < gap_edge_pos_x_midp
        ):
            ngn.count_analyzed_rois_edges_1_neg_lf_1_pos_rt += 1
        #
        elif (
            len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 1 and
            count_label_edges_neg == count_label_edges_neg_roi and
            len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 1 and
            count_label_edges_pos == count_label_edges_pos_roi and
            gap_edge_neg_x_midp > gap_edge_pos_x_midp
        ):
            ngn.count_analyzed_rois_edges_1_pos_lf_1_neg_rt += 1
        #
        else:
            ngn.count_analyzed_rois_edges_neg_pos_else += 1

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def report_gap_rois_analyzed_counts():
    """
    Returns a listing of ounters for gap rois analyzed.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    prt__ = False if 1 else True  # def print switch
    mult_str = '--- '
    def_str = 'report_gap_rois_analyzed_counts'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #== === === === === === === === === === === === === === === === === === ===

    if 1 or prt:
        print fmt0(just2)[0:] % ("ngn.count_analyzed_measts",
            ngn.count_analyzed_measts)
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois",
            ngn.count_analyzed_rois)

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    count_analyzed_rois_sum_edge_neg = (
        ngn.count_analyzed_rois_0_edge_neg +
        ngn.count_analyzed_rois_1_edge_neg +
        ngn.count_analyzed_rois_2_edge_neg +
        ngn.count_analyzed_rois_3_edge_neg +
        ngn.count_analyzed_rois_4_edge_neg +
        ngn.count_analyzed_rois_else_edge_neg
    )

    if count_analyzed_rois_sum_edge_neg > 0:
        _sum_edge_neg_pct = "%5.1f pct" % (
            count_analyzed_rois_sum_edge_neg * 100. /
            count_analyzed_rois_sum_edge_neg)
        _0_edge_neg_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_0_edge_neg * 100. /
            count_analyzed_rois_sum_edge_neg)
        _1_edge_neg_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_1_edge_neg * 100. /
            count_analyzed_rois_sum_edge_neg)
        _2_edge_neg_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_2_edge_neg * 100. /
            count_analyzed_rois_sum_edge_neg)
        _3_edge_neg_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_3_edge_neg * 100. /
            count_analyzed_rois_sum_edge_neg)
        _4_edge_neg_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_4_edge_neg * 100. /
            count_analyzed_rois_sum_edge_neg)
        _else_edge_neg_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_else_edge_neg * 100. /
            count_analyzed_rois_sum_edge_neg)
    else:
        _sum_edge_neg_pct = "na pct"
        _0_edge_neg_pct = "na pct"
        _1_edge_neg_pct = "na pct"
        _2_edge_neg_pct = "na pct"
        _3_edge_neg_pct = "na pct"
        _4_edge_neg_pct = "na pct"
        _else_edge_neg_pct = "na pct"

    if 1 or prt:
        print fmt0(just2)[0:] % ("ngn.count_analyzed_rois_sum_edge_neg",
            [count_analyzed_rois_sum_edge_neg, _sum_edge_neg_pct])
        print fmt0(just2)[0:] % ("ngn.count_analyzed_rois_0_edge_neg",
            [ngn.count_analyzed_rois_0_edge_neg, _0_edge_neg_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_1_edge_neg",
            [ngn.count_analyzed_rois_1_edge_neg, _1_edge_neg_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_2_edge_neg",
            [ngn.count_analyzed_rois_2_edge_neg, _2_edge_neg_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_3_edge_neg",
            [ngn.count_analyzed_rois_3_edge_neg, _3_edge_neg_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_4_edge_neg",
            [ngn.count_analyzed_rois_4_edge_neg, _4_edge_neg_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_else_edge_neg",
            [ngn.count_analyzed_rois_else_edge_neg, _else_edge_neg_pct])

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    count_analyzed_rois_sum_edge_pos = (
        ngn.count_analyzed_rois_0_edge_pos +
        ngn.count_analyzed_rois_1_edge_pos +
        ngn.count_analyzed_rois_2_edge_pos +
        ngn.count_analyzed_rois_3_edge_pos +
        ngn.count_analyzed_rois_4_edge_pos +
        ngn.count_analyzed_rois_else_edge_pos
    )

    if count_analyzed_rois_sum_edge_pos > 0:
        _sum_edge_pos_pct = "%5.1f pct" % (
            count_analyzed_rois_sum_edge_pos * 100. /
            count_analyzed_rois_sum_edge_pos)
        _0_edge_pos_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_0_edge_pos * 100. /
            count_analyzed_rois_sum_edge_pos)
        _1_edge_pos_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_1_edge_pos * 100. /
            count_analyzed_rois_sum_edge_pos)
        _2_edge_pos_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_2_edge_pos * 100. /
            count_analyzed_rois_sum_edge_pos)
        _3_edge_pos_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_3_edge_pos * 100. /
            count_analyzed_rois_sum_edge_pos)
        _4_edge_pos_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_4_edge_pos * 100. /
            count_analyzed_rois_sum_edge_pos)
        _else_edge_pos_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_else_edge_pos * 100. /
            count_analyzed_rois_sum_edge_pos)
    else:
        _sum_edge_pos_pct = "na pct"
        _0_edge_pos_pct = "na pct"
        _1_edge_pos_pct = "na pct"
        _2_edge_pos_pct = "na pct"
        _3_edge_pos_pct = "na pct"
        _4_edge_pos_pct = "na pct"
        _else_edge_pos_pct = "na pct"

    if 1 or prt:
        print fmt0(just2)[0:] % ("ngn.count_analyzed_rois_sum_edge_pos",
            [count_analyzed_rois_sum_edge_pos, _sum_edge_pos_pct])
        print fmt0(just2)[0:] % ("ngn.count_analyzed_rois_0_edge_pos",
            [ngn.count_analyzed_rois_0_edge_pos, _0_edge_pos_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_1_edge_pos",
            [ngn.count_analyzed_rois_1_edge_pos, _1_edge_pos_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_2_edge_pos",
            [ngn.count_analyzed_rois_2_edge_pos, _2_edge_pos_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_3_edge_pos",
            [ngn.count_analyzed_rois_3_edge_pos, _3_edge_pos_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_4_edge_pos",
            [ngn.count_analyzed_rois_4_edge_pos, _4_edge_pos_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_else_edge_pos",
            [ngn.count_analyzed_rois_else_edge_pos, _else_edge_pos_pct])

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    count_analyzed_rois_sum_edges = (
        ngn.count_analyzed_rois_edges_none +
        ngn.count_analyzed_rois_edges_neg_only +
        ngn.count_analyzed_rois_edges_pos_only +
        ngn.count_analyzed_rois_edges_neg_pos
    )

    if count_analyzed_rois_sum_edges > 0:
        _sum_edges_pct = "%5.1f pct" % (
            count_analyzed_rois_sum_edges * 100. /
            count_analyzed_rois_sum_edges)
        _edges_none_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_edges_none * 100. /
            count_analyzed_rois_sum_edges)
        _edges_neg_only_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_edges_neg_only * 100. /
            count_analyzed_rois_sum_edges)
        _edges_pos_only_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_edges_pos_only * 100. /
            count_analyzed_rois_sum_edges)
        _edges_neg_pos_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_edges_neg_pos * 100. /
            count_analyzed_rois_sum_edges)

    else:
        _sum_edges_pct = "na pct"
        _edges_none_pct = "na pct"
        _edges_neg_only_pct = "na pct"
        _edges_pos_only_pct = "na pct"
        _edges_neg_pos_pct = "na pct"

    if 1 or prt:
        print fmt0(just2)[0:] % ("count_analyzed_rois_sum_edges",
            [count_analyzed_rois_sum_edges, _sum_edges_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_edges_none",
            [ngn.count_analyzed_rois_edges_none, _edges_none_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_edges_neg_only",
            [ngn.count_analyzed_rois_edges_neg_only, _edges_neg_only_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_edges_pos_only",
            [ngn.count_analyzed_rois_edges_pos_only, _edges_pos_only_pct])
        print fmt0(just2)[1:] % ("ngn.count_analyzed_rois_edges_neg_pos",
            [ngn.count_analyzed_rois_edges_neg_pos, _edges_neg_pos_pct])

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    count_analyzed_rois_sum_edges_neg_pos = (
        ngn.count_analyzed_rois_edges_1_neg_lf_1_pos_rt +
        ngn.count_analyzed_rois_edges_1_pos_lf_1_neg_rt +
        ngn.count_analyzed_rois_edges_neg_pos_else
    )

    if count_analyzed_rois_sum_edges_neg_pos > 0:
        _sum_edges_neg_pos_pct = "%5.1f pct" % (
            count_analyzed_rois_sum_edges_neg_pos * 100. /
            count_analyzed_rois_sum_edges_neg_pos)
        _edges_1_neg_lf_1_pos_rt_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_edges_1_neg_lf_1_pos_rt * 100. /
            count_analyzed_rois_sum_edges_neg_pos)
        _edges_1_pos_lf_1_neg_rt_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_edges_1_pos_lf_1_neg_rt * 100. /
            count_analyzed_rois_sum_edges_neg_pos)
        _edges_neg_pos_else_pct = "%5.1f pct" % (
            ngn.count_analyzed_rois_edges_neg_pos_else * 100. /
            count_analyzed_rois_sum_edges_neg_pos)

    else:
        _sum_edges_neg_pos_pct = "na pct"
        _edges_1_neg_lf_1_pos_rt_pct = "na pct"
        _edges_1_pos_lf_1_neg_rt_pct = "na pct"
        _edges_neg_pos_else_pct = "na pct"

    if 1 or prt:
        print fmt0(just2)[0:] % ("count_analyzed_rois_sum_edges_neg_pos",
            [count_analyzed_rois_sum_edges_neg_pos, _sum_edges_neg_pos_pct])
        print fmt0(just2)[1:] % (
            "ngn.count_analyzed_rois_edges_1_neg_lf_1_pos_rt",
            [ngn.count_analyzed_rois_edges_1_neg_lf_1_pos_rt,
            _edges_1_neg_lf_1_pos_rt_pct])
        print fmt0(just2)[1:] % (
            "ngn.count_analyzed_rois_edges_1_pos_lf_1_neg_rt",
            [ngn.count_analyzed_rois_edges_1_pos_lf_1_neg_rt,
            _edges_1_pos_lf_1_neg_rt_pct])
        print fmt0(just2)[1:] % (
            "ngn.count_analyzed_rois_edges_neg_pos_else",
            [ngn.count_analyzed_rois_edges_neg_pos_else,
            _edges_neg_pos_else_pct])

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def identify_roi_profile_features(indy, profile_xs_nnan_where, profile_xs,
nnan_xs, fltr_zs, nnan_xs_midp, fltr_dzdxs, segm_ps,
meast_id, gap_idx, roi_tow_id_lf, roi_tow_id_rt, gap_id,
roi_tow_edge_xref_lf, roi_tow_edge_xref_rt):
    """
    Returns a Pandas Series containing gap analysis results for this ROI.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    prt__ = False if 1 else True  # def print switch
    mult_str = '--- '
    def_str = 'identify_roi_profile_features'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #f 1 and prt:
    #   print fmt0(just1)[1:] % ("indy", indy)
    #   print fmt0(just1)[1:] % ("gap_idx", gap_idx)
    #   print fmt0(just1)[1:] % ("gap_id", gap_id)
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs.shape", nnan_xs.shape)
    #f 0 and prt:
    #   print fmt1(just1)[0:] % ("sorted(segm_ps.keys())",
    #       prt_list(sorted(segm_ps.keys()), 0))
    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    ps_gap = ngn.ps_gap_init.copy()
    ps_gap.gap_tows = (roi_tow_id_lf, roi_tow_id_rt)
    ps_gap.gap_id = gap_id

    # required initialization
    #
    count_label_edges_neg = np.nan
    count_label_edges_neg_roi = np.nan
    gap_edge_neg_x_midp = np.nan
    #
    count_label_edges_pos = np.nan
    count_label_edges_pos_roi = np.nan
    gap_edge_pos_x_midp = np.nan

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("profile_xs.shape", profile_xs.shape)
    #   print fmt0(just2)[1:] % ("profile_xs_nnan_where.shape",
    #       profile_xs_nnan_where.shape)
    #   print fmt0(just2)[1:] % ("nnan_xs.shape", nnan_xs.shape)

    nnan_xs_roi_mask = ((nnan_xs >= roi_tow_edge_xref_lf) &
        (nnan_xs <= roi_tow_edge_xref_rt))
    nnan_xs_roi_where = np.where(nnan_xs_roi_mask)[0]
    nnan_xs_roi = nnan_xs[nnan_xs_roi_mask].copy()
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("nnan_xs_roi_mask.shape",
    #       nnan_xs_roi_mask.shape)
    #   print fmt0(just2)[1:] % ("np.sum(nnan_xs_roi_mask)",
    #       np.sum(nnan_xs_roi_mask))
    #   print fmt0(just2)[1:] % ("nnan_xs_roi_where.shape",
    #       nnan_xs_roi_where.shape)
    #   #rint fmt1(just2)[1:] % ("nnan_xs_roi_where",
    #   #   nnan_xs_roi_where)
    #   print fmt0(just2)[1:] % ("nnan_xs_roi.shape",
    #       nnan_xs_roi.shape)

    fltr_zs_roi = fltr_zs[nnan_xs_roi_mask].copy()
    #
    fltr_zs_mask_edges_neg_roi = (
        segm_ps.fltr_zs_mask_edges_neg[nnan_xs_roi_mask])
    fltr_zs_roi_edges_neg = fltr_zs_roi.copy()
    fltr_zs_roi_edges_neg[~ fltr_zs_mask_edges_neg_roi] = np.nan
    #
    fltr_zs_mask_edges_pos_roi = (
        segm_ps.fltr_zs_mask_edges_pos[nnan_xs_roi_mask])
    fltr_zs_roi_edges_pos = fltr_zs_roi.copy()
    fltr_zs_roi_edges_pos[~ fltr_zs_mask_edges_pos_roi] = np.nan
    #
    fltr_zs_mask_flats_roi = (
        segm_ps.fltr_zs_mask_flats[nnan_xs_roi_mask])
    fltr_zs_roi_flats = fltr_zs_roi.copy()
    fltr_zs_roi_flats[~ fltr_zs_mask_flats_roi] = np.nan
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_zs_roi.shape",
    #       fltr_zs_roi.shape)
    #   print fmt0(just2)[1:] % ("fltr_zs_roi_edges_neg.shape",
    #       fltr_zs_roi_edges_neg.shape)
    #   print fmt0(just2)[1:] % ("fltr_zs_roi_edges_pos.shape",
    #       fltr_zs_roi_edges_pos.shape)
    #   print fmt0(just2)[1:] % ("fltr_zs_roi_flats.shape",
    #       fltr_zs_roi_flats.shape)

    #f gap_idx >= 0:
    #   None if 0 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    nnan_xs_midp_roi_where = nnan_xs_roi_where[:-1]
    nnan_xs_midp_roi_mask = np.zeros(nnan_xs_midp.shape).astype(np.bool)
    nnan_xs_midp_roi_mask[nnan_xs_midp_roi_where] = True
    nnan_xs_midp_roi = nnan_xs_midp[nnan_xs_midp_roi_mask].copy()
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("nnan_xs_midp.shape", nnan_xs_midp.shape)
    #   print fmt0(just2)[1:] % ("nnan_xs_midp_roi_mask.shape",
    #       nnan_xs_midp_roi_mask.shape)
    #   print fmt0(just2)[1:] % ("np.sum(nnan_xs_midp_roi_mask)",
    #       np.sum(nnan_xs_midp_roi_mask))
    #   print fmt0(just2)[1:] % ("nnan_xs_midp_roi_where.shape",
    #       nnan_xs_midp_roi_where.shape)
    #   #rint fmt1(just2)[1:] % ("nnan_xs_midp_roi_where",
    #   #   nnan_xs_midp_roi_where)
    #   print fmt0(just2)[1:] % ("nnan_xs_midp_roi.shape",
    #       nnan_xs_midp_roi.shape)

    #f 0 and prt:
    #   print fmt0(just0)[0:] % ("len(nnan_xs_midp_roi)",
    #       len(nnan_xs_midp_roi))
    #   print fmt1(just0)[1:] % (
    #       "np.vstack((nnan_xs_roi[:-1], nnan_xs_midp_roi))",
    #       np.vstack((nnan_xs_roi[:-1], nnan_xs_midp_roi)))

    #f 1 and prt:
    #   print fmt1(just2)[0:] % ("nnan_xs_midp_roi_where",
    #       nnan_xs_midp_roi_where)
    #f 1 and prt:
    #   print fmt1(just2)[1:] % ("nnan_xs_roi_where",
    #       nnan_xs_roi_where)]

### #ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
### # (below) "hacks" for datasets
### #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
### if ('EvenTow2_LD60_LB35000_DF71.csv' in ngn.job_zs_csv and
### (indy == 272 or indy == 409 or indy == 509)):
###     # encountered an "infinite" dzdx value result that raised an exception.
###     # Below is a fix (hack) to avoid this.
###     #
###     #rint "\n... (below) hack hack hack hack hack hack hack hack hack hack"
###
###     fltr_dzdxs_mask_isfinite = np.isfinite(fltr_dzdxs)
###     #f 1 or  prt:
###     #   print fmt0(just2)[0:] % ("fltr_dzdxs.shape", fltr_dzdxs.shape)
###     #   print fmt0(just2)[1:] % ("fltr_dzdxs_mask_isfinite.shape",
###     #       fltr_dzdxs_mask_isfinite.shape)
###     #   print fmt0(just2)[1:] % ("np.sum(fltr_dzdxs_mask_isfinite)",
###     #       np.sum(fltr_dzdxs_mask_isfinite))
###
###     fltr_dzdxs[~ fltr_dzdxs_mask_isfinite] = np.nan
###
###     #rint "\n... (above) hack hack hack hack hack hack hack hack hack hack"
###     #one if 1 else sys.exit()
### #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
### if ('EvenTow1_LD60_LB35000_DF71.csv' in ngn.job_zs_csv):
###     # encountered an "infinite" dzdx value result that raised an exception.
###     # Below is a fix (hack) to avoid this.
###     #
###     #rint "\n... (below) hack hack hack hack hack hack hack hack hack hack"
###
###     fltr_dzdxs_mask_isfinite = np.isfinite(fltr_dzdxs)
###     #f 1 or  prt:
###     #   print fmt0(just2)[0:] % ("fltr_dzdxs.shape", fltr_dzdxs.shape)
###     #   print fmt0(just2)[1:] % ("fltr_dzdxs_mask_isfinite.shape",
###     #       fltr_dzdxs_mask_isfinite.shape)
###     #   print fmt0(just2)[1:] % ("np.sum(fltr_dzdxs_mask_isfinite)",
###     #       np.sum(fltr_dzdxs_mask_isfinite))
###
###     fltr_dzdxs[~ fltr_dzdxs_mask_isfinite] = np.nan
###
###     #rint "\n... (above) hack hack hack hack hack hack hack hack hack hack"
###     #one if 1 else sys.exit()
### #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
### if ('Scan parallel base layer part3_LD90_LB65536_DF101p4.csv' in
### ngn.job_zs_csv):
###     # encountered an "infinite" dzdx value result that raised an exception.
###     # Below is a fix (hack) to avoid this.
###     #
###     #rint "\n... (below) hack hack hack hack hack hack hack hack hack hack"
###
###     fltr_dzdxs_mask_isfinite = np.isfinite(fltr_dzdxs)
###     #f 1 or  prt:
###     #   print fmt0(just2)[0:] % ("fltr_dzdxs.shape", fltr_dzdxs.shape)
###     #   print fmt0(just2)[1:] % ("fltr_dzdxs_mask_isfinite.shape",
###     #       fltr_dzdxs_mask_isfinite.shape)
###     #   print fmt0(just2)[1:] % ("np.sum(fltr_dzdxs_mask_isfinite)",
###     #       np.sum(fltr_dzdxs_mask_isfinite))
###
###     fltr_dzdxs[~ fltr_dzdxs_mask_isfinite] = np.nan
###
###     #rint "\n... (above) hack hack hack hack hack hack hack hack hack hack"
###     #one if 1 else sys.exit()
### #zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
### # (above) "hacks" for datasets
### #ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ

    fltr_dzdxs_roi = fltr_dzdxs[nnan_xs_midp_roi_mask].copy()
    #
    fltr_dzdxs_mask_edges_neg_roi = (
        segm_ps.fltr_dzdxs_mask_edges_neg[nnan_xs_midp_roi_mask])
    fltr_dzdxs_roi_edges_neg = fltr_dzdxs_roi.copy()
    fltr_dzdxs_roi_edges_neg[~ fltr_dzdxs_mask_edges_neg_roi] = np.nan
    #
    fltr_dzdxs_mask_edges_pos_roi = (
        segm_ps.fltr_dzdxs_mask_edges_pos[nnan_xs_midp_roi_mask])
    fltr_dzdxs_roi_edges_pos = fltr_dzdxs_roi.copy()
    fltr_dzdxs_roi_edges_pos[~ fltr_dzdxs_mask_edges_pos_roi] = np.nan
    #
    fltr_dzdxs_mask_flats_roi = (
        segm_ps.fltr_dzdxs_mask_flats[nnan_xs_midp_roi_mask])
    fltr_dzdxs_roi_flats = fltr_dzdxs_roi.copy()
    fltr_dzdxs_roi_flats[~ fltr_dzdxs_mask_flats_roi] = np.nan
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_dzdxs_roi.shape",
    #       fltr_dzdxs_roi.shape)
    #   print fmt0(just2)[1:] % ("fltr_dzdxs_roi_edges_neg.shape",
    #       fltr_dzdxs_roi_edges_neg.shape)
    #   print fmt0(just2)[1:] % ("fltr_dzdxs_roi_edges_pos.shape",
    #       fltr_dzdxs_roi_edges_pos.shape)
    #   print fmt0(just2)[1:] % ("fltr_dzdxs_roi_flats.shape",
    #       fltr_dzdxs_roi_flats.shape)

    #one if 0 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #f 1 and prt:
    #   print "\n%s" % ('..  ' * 40)
    #   print fmt0(just2)[1:] % ("find the tow neg edge in the roi",
    #       "find the tow neg edge in the roi")

    fltr_dzdxs_label_edges_neg = segm_ps.fltr_dzdxs_label_edges_neg
    fltr_dzdxs_label_edges_neg_roi = (
        fltr_dzdxs_label_edges_neg[nnan_xs_midp_roi_mask])
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_dzdxs_label_edges_neg_roi.shape",
    #       fltr_dzdxs_label_edges_neg_roi.shape)
    #   print fmt1(just2)[1:] % ("fltr_dzdxs_label_edges_neg_roi",
    #       fltr_dzdxs_label_edges_neg_roi)

    fltr_dzdxs_label_edges_neg_roi_uniqs = np.unique(
        fltr_dzdxs_label_edges_neg_roi)
    fltr_dzdxs_label_edges_neg_roi_uniqs_gt0 = (
        fltr_dzdxs_label_edges_neg_roi_uniqs[
            fltr_dzdxs_label_edges_neg_roi_uniqs > 0])
    #f 1 and prt:
    #   #rint fmt0(just2)[0:] % ("fltr_dzdxs_label_edges_neg_roi_uniqs",
    #   #   fltr_dzdxs_label_edges_neg_roi_uniqs)
    #   print fmt0(just2)[0:] % (
    #       "fltr_dzdxs_label_edges_neg_roi_uniqs_gt0",
    #       fltr_dzdxs_label_edges_neg_roi_uniqs_gt0)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % (
    #       "len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 1",
    #       len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 1)

    if len(fltr_dzdxs_label_edges_neg_roi_uniqs_gt0) == 1:
        # there is only one neg edge in the ROI ...
        # ... now, check if the edge is fully contained within the ROI

        label_edges_neg_roi = fltr_dzdxs_label_edges_neg_roi_uniqs_gt0[0]
        #
        count_label_edges_neg = np.sum(
            fltr_dzdxs_label_edges_neg == label_edges_neg_roi)
        #
        count_label_edges_neg_roi = np.sum(
            fltr_dzdxs_label_edges_neg_roi == label_edges_neg_roi)
        #f 1 and prt:
        #   print fmt0(just2)[0:] % ("label_edges_neg_roi",
        #       label_edges_neg_roi)
        #   print fmt0(just2)[1:] % ("count_label_edges_neg",
        #       count_label_edges_neg)
        #   print fmt0(just2)[1:] % ("count_label_edges_neg_roi",
        #       count_label_edges_neg_roi)
        #f 1 and prt:
        #   print fmt0(just2)[0:] % (
        #       "count_label_edges_neg == count_label_edges_neg_roi",
        #       count_label_edges_neg == count_label_edges_neg_roi)

        if count_label_edges_neg == count_label_edges_neg_roi:
            #f 1 and prt:
            #   print fmt0(just2)[0:] % (
            #       "a tow neg edge is (considered) found",
            #       "a tow neg edge is (considered) found")

            fltr_dzdxs_where_edge_neg_roi = (
                np.where(fltr_dzdxs_mask_edges_neg_roi)[0])
            nnan_xs_midp_roi_where_edge_neg = (
                nnan_xs_midp_roi_where[fltr_dzdxs_where_edge_neg_roi])
            #f 1 and prt:
            #   print fmt0(just2)[0:] % (
            #       "fltr_dzdxs_where_edge_neg_roi.shape",
            #       fltr_dzdxs_where_edge_neg_roi.shape)
            #   print fmt0(just2)[1:] % ("fltr_dzdxs_where_edge_neg_roi",
            #       fltr_dzdxs_where_edge_neg_roi)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % (
            #       "nnan_xs_midp_roi_where_edge_neg",
            #       nnan_xs_midp_roi_where_edge_neg)

            fltr_dzdxs_roi_edge_neg = (
                fltr_dzdxs_roi_edges_neg[fltr_dzdxs_mask_edges_neg_roi])
            fltr_dzdxs_roi_edge_neg_min = np.nanmin(fltr_dzdxs_roi_edge_neg)
            fltr_dzdxs_roi_edge_neg_min_mask = (
                fltr_dzdxs_roi_edge_neg == fltr_dzdxs_roi_edge_neg_min)
            fltr_dzdxs_roi_edge_neg_min_where = np.where(
                fltr_dzdxs_roi_edge_neg_min_mask)[0]
            nnan_xs_midp_roi_where_edge_neg_min = (
                nnan_xs_midp_roi_where_edge_neg[
                    fltr_dzdxs_roi_edge_neg_min_where])
            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("fltr_dzdxs_roi_edge_neg",
            #       fltr_dzdxs_roi_edge_neg)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("fltr_dzdxs_roi_edge_neg_min",
            #       fltr_dzdxs_roi_edge_neg_min)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % (
            #       "fltr_dzdxs_roi_edge_neg_min_mask.astype(np.int)",
            #       fltr_dzdxs_roi_edge_neg_min_mask.astype(np.int))
            #f 1 and prt:
            #   print fmt0(just2)[1:] % (
            #       "fltr_dzdxs_roi_edge_neg_min_where",
            #       fltr_dzdxs_roi_edge_neg_min_where)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % (
            #       "nnan_xs_midp_roi_where_edge_neg_min",
            #       nnan_xs_midp_roi_where_edge_neg_min)

            nnan_xs_midp_roi_where_edge_neg_min_dzdxs = (
                fltr_dzdxs[nnan_xs_midp_roi_where_edge_neg_min])
            gap_edge_neg_dzdx = np.mean(
                nnan_xs_midp_roi_where_edge_neg_min_dzdxs)
            #f 1 and prt:
            #   print fmt0(just2)[0:] % (
            #       "nnan_xs_midp_roi_where_edge_neg_min_dzdxs",
            #       nnan_xs_midp_roi_where_edge_neg_min_dzdxs)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("gap_edge_neg_dzdx",
            #       gap_edge_neg_dzdx)

            nnan_xs_midp_roi_where_edge_neg_min_xs_midp = (
                nnan_xs_midp[nnan_xs_midp_roi_where_edge_neg_min])
            gap_edge_neg_x_midp = np.mean(
                nnan_xs_midp_roi_where_edge_neg_min_xs_midp)
            gap_edge_neg_x_midp_nnan_idx = find_nearest_value_index(
                nnan_xs_midp, gap_edge_neg_x_midp)
            gap_edge_neg_x_midp_idx = (
                profile_xs_nnan_where[gap_edge_neg_x_midp_nnan_idx])
            #f 1 and prt:
            #   print fmt0(just2)[0:] % (
            #       "nnan_xs_midp_roi_where_edge_neg_min_xs_midp",
            #       nnan_xs_midp_roi_where_edge_neg_min_xs_midp)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("gap_edge_neg_x_midp",
            #       gap_edge_neg_x_midp)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("gap_edge_neg_x_midp_nnan_idx",
            #       gap_edge_neg_x_midp_nnan_idx)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("gap_edge_neg_x_midp_idx",
            #       gap_edge_neg_x_midp_idx)

            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("nnan_xs_roi.shape",
            #       nnan_xs_roi.shape)
            #   #rint fmt1(just2)[1:] % ("nnan_xs_roi", nnan_xs_roi)
            #   print fmt0(just2)[1:] % ("fltr_zs_roi.shape",
            #       nnan_xs_roi.shape)
            #   #rint fmt1(just2)[1:] % ("fltr_zs_roi", fltr_zs_roi)

            array_x = nnan_xs_roi.copy()
            array_y = fltr_zs_roi.copy()
            #f 1 and prt:
            #   print fmt1(just2)[0:] % ("(init) array_x", array_x)
            #   print fmt1(just2)[1:] % ("(init) array_y", array_y)

            if not np.all(np.diff(array_x) > 0):
                # array_x values must be at least in sort order
                array_x_argsort = np.argsort(array_x)
                #rray_x_argsort = (
                #   array_x_argsort if 0 else array_x_argsort[::-1])
                array_x = array_x[array_x_argsort]
                array_y = array_y[array_x_argsort]
                #f 1 and prt:
                #   print fmt1(just2)[0:] % ("array_x_argsort",
                #       array_x_argsort)
                #   print fmt1(just2)[1:] % ("(updt) array_x", array_x)
                #   print fmt1(just2)[1:] % ("(updt) array_y", array_y)

            if np.all(np.diff(array_x) < 0):
                # np.interp requires ascending array_x values
                array_x = array_x[::-1]
                array_y = array_y[::-1]
                #f 1 and prt:
                #   print fmt1(just2)[0:] % ("(updt) array_x", array_x)
                #   print fmt1(just2)[1:] % ("(updt) array_y", array_y)

            value_xs = gap_edge_neg_x_midp
            value_ys = np.interp(
                value_xs, array_x, array_y, left=np.nan, right=np.nan)
            gap_edge_neg_z = value_ys
            #f 1 and prt:
            #   print fmt1(just2)[0:] % ("(interp) array_x", array_x)
            #   print fmt1(just2)[1:] % ("(interp) array_y", array_y)
            #   print fmt0(just2)[0:] % ("value_xs", value_xs)
            #   print fmt0(just2)[1:] % ("value_ys", value_ys)
            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("gap_edge_neg_z", gap_edge_neg_z)

            #one if 1 else sys.exit()

            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

            ps_gap.edge_neg_x_midp_nnan_idx = gap_edge_neg_x_midp_nnan_idx
            ps_gap.edge_neg_x_midp_idx = gap_edge_neg_x_midp_idx
            ps_gap.edge_neg_dzdx = gap_edge_neg_dzdx
            ps_gap.edge_neg_x_midp = gap_edge_neg_x_midp
            ps_gap.edge_neg_z = gap_edge_neg_z
            #f 1 and prt:
            #   print fmt1(just0)[0:] % ("ps_gap", ps_gap)

    #f 0 and gap_idx >= 1:
    #   None if 0 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    #f 1 and prt:
    #   print "\n%s" % ('..  ' * 40)
    #   print fmt0(just2)[1:] % ("find the tow pos edge in the roi",
    #       "find the tow pos edge in the roi")

    fltr_dzdxs_label_edges_pos = segm_ps.fltr_dzdxs_label_edges_pos
    fltr_dzdxs_label_edges_pos_roi = (
        fltr_dzdxs_label_edges_pos[nnan_xs_midp_roi_mask])
    #f 1 and prt:
    #   print fmt0(just2)[0:] % ("fltr_dzdxs_label_edges_pos_roi.shape",
    #       fltr_dzdxs_label_edges_pos_roi.shape)
    #   print fmt1(just2)[1:] % ("fltr_dzdxs_label_edges_pos_roi",
    #       fltr_dzdxs_label_edges_pos_roi)

    fltr_dzdxs_label_edges_pos_roi_uniqs = np.unique(
        fltr_dzdxs_label_edges_pos_roi)
    fltr_dzdxs_label_edges_pos_roi_uniqs_gt0 = (
        fltr_dzdxs_label_edges_pos_roi_uniqs[
            fltr_dzdxs_label_edges_pos_roi_uniqs > 0])
    #f 1 and prt:
    #   #rint fmt0(just2)[0:] % ("fltr_dzdxs_label_edges_pos_roi_uniqs",
    #   #   fltr_dzdxs_label_edges_pos_roi_uniqs)
    #   print fmt0(just2)[0:] % (
    #       "fltr_dzdxs_label_edges_pos_roi_uniqs_gt0",
    #       fltr_dzdxs_label_edges_pos_roi_uniqs_gt0)
    #f 1 and prt:
    #   print fmt0(just2)[0:] % (
    #       "len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 1",
    #       len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 1)

    if len(fltr_dzdxs_label_edges_pos_roi_uniqs_gt0) == 1:
        # there is only one pos edge in the ROI ...
        # ... now, check if the edge is fully contained within the ROI

        label_edges_pos_roi = fltr_dzdxs_label_edges_pos_roi_uniqs_gt0[0]
        #
        count_label_edges_pos = np.sum(
            fltr_dzdxs_label_edges_pos == label_edges_pos_roi)
        #
        count_label_edges_pos_roi = np.sum(
            fltr_dzdxs_label_edges_pos_roi == label_edges_pos_roi)
        #f 1 and prt:
        #   print fmt0(just2)[0:] % ("label_edges_pos_roi",
        #       label_edges_pos_roi)
        #   print fmt0(just2)[1:] % ("count_label_edges_pos",
        #       count_label_edges_pos)
        #   print fmt0(just2)[1:] % ("count_label_edges_pos_roi",
        #       count_label_edges_pos_roi)
        #f 1 and prt:
        #   print fmt0(just2)[0:] % (
        #       "count_label_edges_pos == count_label_edges_pos_roi",
        #       count_label_edges_pos == count_label_edges_pos_roi)

        if count_label_edges_pos == count_label_edges_pos_roi:
            #f 1 and prt:
            #   print fmt0(just2)[0:] % (
            #       "a tow pos edge is (considered) found",
            #       "a tow pos edge is (considered) found")

            fltr_dzdxs_where_edge_pos_roi = (
                np.where(fltr_dzdxs_mask_edges_pos_roi)[0])
            nnan_xs_midp_roi_where_edge_pos = (
                nnan_xs_midp_roi_where[fltr_dzdxs_where_edge_pos_roi])
            #f 1 and prt:
            #   print fmt0(just2)[0:] % (
            #       "fltr_dzdxs_where_edge_pos_roi.shape",
            #       fltr_dzdxs_where_edge_pos_roi.shape)
            #   print fmt0(just2)[1:] % ("fltr_dzdxs_where_edge_pos_roi",
            #       fltr_dzdxs_where_edge_pos_roi)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % (
            #       "nnan_xs_midp_roi_where_edge_pos",
            #       nnan_xs_midp_roi_where_edge_pos)

            fltr_dzdxs_roi_edge_pos = (
                fltr_dzdxs_roi_edges_pos[fltr_dzdxs_mask_edges_pos_roi])
            fltr_dzdxs_roi_edge_pos_max = np.nanmax(fltr_dzdxs_roi_edge_pos)
            fltr_dzdxs_roi_edge_pos_max_mask = (
                fltr_dzdxs_roi_edge_pos == fltr_dzdxs_roi_edge_pos_max)
            fltr_dzdxs_roi_edge_pos_max_where = np.where(
                fltr_dzdxs_roi_edge_pos_max_mask)[0]
            nnan_xs_midp_roi_where_edge_pos_max = (
                nnan_xs_midp_roi_where_edge_pos[
                    fltr_dzdxs_roi_edge_pos_max_where])
            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("fltr_dzdxs_roi_edge_pos",
            #       fltr_dzdxs_roi_edge_pos)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("fltr_dzdxs_roi_edge_pos_max",
            #       fltr_dzdxs_roi_edge_pos_max)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % (
            #       "fltr_dzdxs_roi_edge_pos_max_mask.astype(np.int)",
            #       fltr_dzdxs_roi_edge_pos_max_mask.astype(np.int))
            #f 1 and prt:
            #   print fmt0(just2)[1:] % (
            #       "fltr_dzdxs_roi_edge_pos_max_where",
            #       fltr_dzdxs_roi_edge_pos_max_where)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % (
            #       "nnan_xs_midp_roi_where_edge_pos_max",
            #       nnan_xs_midp_roi_where_edge_pos_max)

            nnan_xs_midp_roi_where_edge_pos_max_dzdxs = (
                fltr_dzdxs[nnan_xs_midp_roi_where_edge_pos_max])
            gap_edge_pos_dzdx = np.mean(
                nnan_xs_midp_roi_where_edge_pos_max_dzdxs)
            #f 1 and prt:
            #   print fmt0(just2)[0:] % (
            #       "nnan_xs_midp_roi_where_edge_pos_max_dzdxs",
            #       nnan_xs_midp_roi_where_edge_pos_max_dzdxs)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("gap_edge_pos_dzdx",
            #       gap_edge_pos_dzdx)

            nnan_xs_midp_roi_where_edge_pos_max_xs_midp = (
                nnan_xs_midp[nnan_xs_midp_roi_where_edge_pos_max])
            gap_edge_pos_x_midp = np.mean(
                nnan_xs_midp_roi_where_edge_pos_max_xs_midp)
            gap_edge_pos_x_midp_nnan_idx = find_nearest_value_index(
                nnan_xs_midp, gap_edge_pos_x_midp)
            gap_edge_pos_x_midp_idx = (
                profile_xs_nnan_where[gap_edge_pos_x_midp_nnan_idx])
            #f 1 and prt:
            #   print fmt0(just2)[0:] % (
            #       "nnan_xs_midp_roi_where_edge_pos_max_xs_midp",
            #       nnan_xs_midp_roi_where_edge_pos_max_xs_midp)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("gap_edge_pos_x_midp",
            #       gap_edge_pos_x_midp)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("gap_edge_pos_x_midp_nnan_idx",
            #       gap_edge_pos_x_midp_nnan_idx)
            #f 1 and prt:
            #   print fmt0(just2)[1:] % ("gap_edge_pos_x_midp_idx",
            #       gap_edge_pos_x_midp_idx)

            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("nnan_xs_roi.shape",
            #       nnan_xs_roi.shape)
            #   #rint fmt1(just2)[1:] % ("nnan_xs_roi", nnan_xs_roi)
            #   print fmt0(just2)[1:] % ("fltr_zs_roi.shape",
            #       nnan_xs_roi.shape)
            #   #rint fmt1(just2)[1:] % ("fltr_zs_roi", fltr_zs_roi)

            array_x = nnan_xs_roi.copy()
            array_y = fltr_zs_roi.copy()
            #f 1 and prt:
            #   print fmt1(just2)[0:] % ("(init) array_x", array_x)
            #   print fmt1(just2)[1:] % ("(init) array_y", array_y)

            if not np.all(np.diff(array_x) > 0):
                # array_x values must be at least in sort order
                array_x_argsort = np.argsort(array_x)
                #rray_x_argsort = (
                #   array_x_argsort if 0 else array_x_argsort[::-1])
                array_x = array_x[array_x_argsort]
                array_y = array_y[array_x_argsort]
                #f 1 and prt:
                #   print fmt1(just2)[0:] % ("array_x_argsort",
                #       array_x_argsort)
                #   print fmt1(just2)[1:] % ("(updt) array_x", array_x)
                #   print fmt1(just2)[1:] % ("(updt) array_y", array_y)

            if np.all(np.diff(array_x) < 0):
                # np.interp requires ascending array_x values
                array_x = array_x[::-1]
                array_y = array_y[::-1]
                #f 1 and prt:
                #   print fmt1(just2)[0:] % ("(updt) array_x", array_x)
                #   print fmt1(just2)[1:] % ("(updt) array_y", array_y)

            value_xs = gap_edge_pos_x_midp
            value_ys = np.interp(
                value_xs, array_x, array_y, left=np.nan, right=np.nan)
            gap_edge_pos_z = value_ys
            #f 1 and prt:
            #   print fmt1(just2)[0:] % ("(interp) array_x", array_x)
            #   print fmt1(just2)[1:] % ("(interp) array_y", array_y)
            #   print fmt0(just2)[0:] % ("value_xs", value_xs)
            #   print fmt0(just2)[1:] % ("value_ys", value_ys)
            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("gap_edge_pos_z", gap_edge_pos_z)

            #one if 1 else sys.exit()

            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

            ps_gap.edge_pos_x_midp_nnan_idx = gap_edge_pos_x_midp_nnan_idx
            ps_gap.edge_pos_x_midp_idx = gap_edge_pos_x_midp_idx
            ps_gap.edge_pos_dzdx = gap_edge_pos_dzdx
            ps_gap.edge_pos_x_midp = gap_edge_pos_x_midp
            ps_gap.edge_pos_z = gap_edge_pos_z
            #f 1 and prt:
            #   print fmt1(just0)[0:] % ("ps_gap", ps_gap)

    #f 0 and gap_idx >= 1:
    #   None if 0 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    if not np.isnan(ps_gap.edge_neg_z) and not np.isnan(ps_gap.edge_pos_z):
        # this is a gap/lap that can be evaluated

        fltr_zs_mask_edges_neg_where = (
            np.where(segm_ps.fltr_zs_mask_edges_neg & nnan_xs_roi_mask)[0])
        #f 1 and prt:
        #   print
        #   print fmt0(just2)[1:] % ("fltr_zs.shape", fltr_zs.shape)
        #   print fmt0(just2)[1:] % ("nnan_xs_roi_mask.shape",
        #       nnan_xs_roi_mask.shape)
        #   print fmt0(just2)[1:] % ("segm_ps.fltr_zs_mask_edges_neg.shape",
        #       segm_ps.fltr_zs_mask_edges_neg.shape)
        #   print fmt0(just2)[1:] % ("fltr_zs_roi.shape", fltr_zs_roi.shape)
        #   print fmt0(just2)[1:] % ("fltr_zs_mask_edges_neg_where.shape",
        #       fltr_zs_mask_edges_neg_where.shape)
        #   print fmt0(just2)[1:] % ("fltr_zs_mask_edges_neg_where",
        #       fltr_zs_mask_edges_neg_where)

        fltr_zs_mask_edges_pos_where = (
            np.where(segm_ps.fltr_zs_mask_edges_pos & nnan_xs_roi_mask)[0])
        #f 1 and prt:
        #   print
        #   print fmt0(just2)[1:] % ("fltr_zs.shape", fltr_zs.shape)
        #   print fmt0(just2)[1:] % ("nnan_xs_roi_mask.shape",
        #       nnan_xs_roi_mask.shape)
        #   print fmt0(just2)[1:] % ("segm_ps.fltr_zs_mask_edges_neg.shape",
        #       segm_ps.fltr_zs_mask_edges_neg.shape)
        #   print fmt0(just2)[1:] % ("fltr_zs_roi.shape", fltr_zs_roi.shape)
        #   print fmt0(just2)[1:] % ("fltr_zs_mask_edges_pos_where.shape",
        #       fltr_zs_mask_edges_pos_where.shape)
        #   print fmt0(just2)[1:] % ("fltr_zs_mask_edges_pos_where",
        #       fltr_zs_mask_edges_pos_where)

        #rint "\n...fltr_zs_mask_edges_neg_where", fltr_zs_mask_edges_neg_where
        #rint "\n...fltr_zs_mask_edges_pos_where", fltr_zs_mask_edges_pos_where

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        if (fltr_zs_mask_edges_neg_where[-1] <=
        fltr_zs_mask_edges_pos_where[0]):
            #f 1 and prt:
            #   print fmt0(just0)[1:] % ("this is (probably) a GAP",
            #       "this is (probably) a GAP")

            if (fltr_zs_mask_edges_neg_where[-1] <
            fltr_zs_mask_edges_pos_where[0]):
                fltr_zs_flat_x_lf = nnan_xs[fltr_zs_mask_edges_neg_where[-1]]
                fltr_zs_flat_x_rt = nnan_xs[fltr_zs_mask_edges_pos_where[0]]
                fltr_zs_flat_x_cn = (
                    (fltr_zs_flat_x_lf + fltr_zs_flat_x_rt) / 2.)
                flat_x_nnan_idx = find_nearest_value_index(
                    nnan_xs, fltr_zs_flat_x_cn)
            else:
                flat_x_nnan_idx = fltr_zs_mask_edges_pos_where[0]

            flat_x_idx = profile_xs_nnan_where[flat_x_nnan_idx]
            flat_x = nnan_xs[flat_x_nnan_idx]
            flat_z = fltr_zs[flat_x_nnan_idx]
            #f 1 and prt:
            #   print
            #   print fmt0(just2)[1:] % ("fltr_zs_flat_x_lf",
            #       fltr_zs_flat_x_lf)
            #   print fmt0(just2)[1:] % ("fltr_zs_flat_x_rt",
            #       fltr_zs_flat_x_rt)
            #   print fmt0(just2)[1:] % ("fltr_zs_flat_x_cn",
            #       fltr_zs_flat_x_cn)
            #   print fmt0(just2)[1:] % ("flat_x_nnan_idx",
            #       flat_x_nnan_idx)
            #   print fmt0(just2)[1:] % ("flat_x_idx", flat_x_idx)
            #   print fmt0(just2)[1:] % ("flat_x", flat_x)
            #   print fmt0(just2)[1:] % ("flat_z", flat_z)

            gap_center_z = np.interp(flat_x,
                np.array([ps_gap.edge_neg_x_midp, ps_gap.edge_pos_x_midp]),
                np.array([ps_gap.edge_neg_z, ps_gap.edge_pos_z]),
                left=np.nan, right=np.nan)
            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("flat_x", flat_x)
            #   print fmt0(just2)[0:] % ("gap_center_z", gap_center_z)

            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("flat_z", flat_z)
            #   print fmt0(just2)[1:] % ("gap_center_z", gap_center_z)
            #   print fmt0(just2)[1:] % ("flat_z < gap_center_z",
            #       flat_z < gap_center_z)

            if flat_z < gap_center_z:

                ps_gap.gap_flat_x_nnan_idx = flat_x_nnan_idx
                ps_gap.gap_flat_x_idx = flat_x_idx
                ps_gap.gap_flat_x = flat_x
                ps_gap.gap_flat_z = flat_z
                ps_gap.gap_center_z = gap_center_z
                ps_gap.gap_width = (
                    ps_gap.edge_pos_x_midp - ps_gap.edge_neg_x_midp)

                #f 1 or prt:
                #   print fmt1(just0)[0:] % ("ps_gap", ps_gap)
                #one if 1 else sys.exit()
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        #one if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        if (fltr_zs_mask_edges_pos_where[-1] <=
        fltr_zs_mask_edges_neg_where[0]):
            #f 1 and prt:
            #   print fmt0(just0)[1:] % ("this is (probably) a LAP",
            #       "this is (probably) a LAP")

            if (fltr_zs_mask_edges_pos_where[-1] <
            fltr_zs_mask_edges_neg_where[0]):
                fltr_zs_flat_x_lf = nnan_xs[fltr_zs_mask_edges_pos_where[-1]]
                fltr_zs_flat_x_rt = nnan_xs[fltr_zs_mask_edges_neg_where[0]]
                fltr_zs_flat_x_cn = (
                    (fltr_zs_flat_x_lf + fltr_zs_flat_x_rt) / 2.)
                flat_x_nnan_idx = find_nearest_value_index(
                    nnan_xs, fltr_zs_flat_x_cn)
            else:
                flat_x_nnan_idx = fltr_zs_mask_edges_neg_where[0]

            flat_x_idx = profile_xs_nnan_where[flat_x_nnan_idx]
            flat_x = nnan_xs[flat_x_nnan_idx]
            flat_z = fltr_zs[flat_x_nnan_idx]
            #f 1 and prt:
            #   print
            #   print fmt0(just2)[1:] % ("fltr_zs_flat_x_lf",
            #       fltr_zs_flat_x_lf)
            #   print fmt0(just2)[1:] % ("fltr_zs_flat_x_rt",
            #       fltr_zs_flat_x_rt)
            #   print fmt0(just2)[1:] % ("fltr_zs_flat_x_cn",
            #       fltr_zs_flat_x_cn)
            #   print fmt0(just2)[1:] % ("flat_x_nnan_idx",
            #       flat_x_nnan_idx)
            #   print fmt0(just2)[1:] % ("flat_x_idx", flat_x_idx)
            #   print fmt0(just2)[1:] % ("flat_x", flat_x)
            #   print fmt0(just2)[1:] % ("flat_z", flat_z)

            gap_center_z = np.interp(flat_x,
                np.array([ps_gap.edge_pos_x_midp, ps_gap.edge_neg_x_midp]),
                np.array([ps_gap.edge_pos_z, ps_gap.edge_neg_z]),
                left=np.nan, right=np.nan)
            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("flat_x", flat_x)
            #   print fmt0(just2)[0:] % ("gap_center_z", gap_center_z)

            #f 1 and prt:
            #   print fmt0(just2)[0:] % ("flat_z", flat_z)
            #   print fmt0(just2)[1:] % ("gap_center_z", gap_center_z)
            #   print fmt0(just2)[1:] % ("flat_z < gap_center_z",
            #       flat_z < gap_center_z)

            if flat_z > gap_center_z:

                ps_gap.gap_flat_x_nnan_idx = flat_x_nnan_idx
                ps_gap.gap_flat_x_idx = flat_x_idx
                ps_gap.gap_flat_x = flat_x
                ps_gap.gap_flat_z = flat_z
                ps_gap.gap_center_z = gap_center_z
                ps_gap.gap_width = (
                    ps_gap.edge_pos_x_midp - ps_gap.edge_neg_x_midp)

                #f 1 or prt:
                #   print fmt1(just0)[0:] % ("ps_gap", ps_gap)
                #one if 1 else sys.exit()
            #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #f 1 or prt:
    #   print fmt1(just0)[0:] % ("ps_gap", ps_gap)
    #f 1 or prt:
    #   print fmt0(just0)[0:] % ("ps_gap.edge_pos_dzdx", ps_gap.edge_pos_dzdx)
    #f 1 or prt:
    #   print fmt0(just0)[0:] % ("np.isfinite(ps_gap.edge_pos_dzdx)",
    #       np.isfinite(ps_gap.edge_pos_dzdx))

    if 1 and ngn.write_to_results_dir and ngn.make_gallery02_plots:

        make_gallery02_roi_profile_plot(
            indy, gap_idx, gap_id, nnan_xs_roi, fltr_zs_roi,
            fltr_zs_roi_edges_neg, fltr_zs_roi_edges_pos,
            fltr_zs_roi_flats,
            nnan_xs_midp_roi, fltr_dzdxs_roi,
            fltr_dzdxs_roi_edges_neg, fltr_dzdxs_roi_edges_pos,
            fltr_dzdxs_roi_flats, ps_gap
        )

    #one if 0 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if ngn.count_gap_rois_analyzed:
        increment_gap_rois_analyzed_counts(
            fltr_dzdxs_label_edges_neg_roi_uniqs_gt0,
            count_label_edges_neg, count_label_edges_neg_roi,
            gap_edge_neg_x_midp,
            fltr_dzdxs_label_edges_pos_roi_uniqs_gt0,
            count_label_edges_pos, count_label_edges_pos_roi,
            gap_edge_pos_x_midp)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return ps_gap


def identify_profile_features_in_rois(indy, profile_xs_nnan_where, profile_xs,
nnan_xs, fltr_zs, nnan_xs_midp, fltr_dzdxs, segm_ps, gap_present_bits=None):
    """
    Returns ...
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    prt__ = False if 1 else True  # def print switch
    mult_str = '--- '
    def_str = 'identify_profile_features_in_rois'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("indy", indy)
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs.shape", nnan_xs.shape)
    #f 0 and prt:
    #   print fmt1(just1)[0:] % ("sorted(segm_ps.keys())",
    #       prt_list(sorted(segm_ps.keys()), 0))
    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #f 1 and prt:
    #   print fmt0(just0)[0:] % ("len(ngn.tow_ids)", len(ngn.tow_ids))
    #   print fmt1(just0)[1:] % (
    #       "np.vstack((ngn.tow_ids, ngn.tow_center_xrefs))",
    #       np.vstack((ngn.tow_ids, ngn.tow_center_xrefs)))

    roi_tow_ids_lf = ngn.tow_ids[:-1]
    roi_tow_ids_rt = ngn.tow_ids[1:]
    roi_tow_edge_xrefs_lf = ngn.tow_center_xrefs[:-1]
    roi_tow_edge_xrefs_rt = ngn.tow_center_xrefs[1:]
    #f 1 and prt:
    #   print fmt0(just0)[0:] % ("len(ngn.tow_edge_ids)",
    #       len(ngn.tow_edge_ids))
    #   print fmt1(just0)[1:] % ("np.vstack((roi_tow_ids_lf, roi_tow_ids_rt," +
    #       " roi_tow_edge_xrefs_lf, roi_tow_edge_xrefs_rt))",
    #       np.vstack((roi_tow_ids_lf, roi_tow_ids_rt, roi_tow_edge_xrefs_lf,
    #           roi_tow_edge_xrefs_rt)))

    #f 1 and prt:
    #   print fmt0(just0)[0:] % ("gap_present_bits", gap_present_bits)
    #one if 0 else sys.exit()

    if gap_present_bits is None:
        # this is a "dumb" gap job ...
        gap_present_bits0 = '0' * len(ngn.tow_edge_ids)
        gap_present_bits1 = '1' * len(ngn.tow_edge_ids)
        if 1 and prt:
            print fmt0(just0)[0:] % ("gap_present_bits0", gap_present_bits0)
            print fmt0(just0)[1:] % ("list(gap_present_bits0)",
                list(gap_present_bits0))
        if 1 and prt:
            print fmt0(just0)[0:] % ("gap_present_bits1", gap_present_bits1)
            print fmt0(just0)[1:] % ("list(gap_present_bits1)",
                list(gap_present_bits1))

        gap_present_bits = gap_present_bits1
        if 1 and ('CNRC20170522_06_PlyDrops' in ngn.job_dir or
        'CNRC20170519_30ThouLap' in ngn.job_dir):
            gap_present_bits = (
                gap_present_bits0[:9] +
                gap_present_bits1[:15] +
                gap_present_bits0[:9]
            )
            #f 1 and prt:
            #   print fmt0(just0)[0:] % ("len(gap_present_bits)",
            #       len(gap_present_bits))
            #   print fmt0(just0)[1:] % ("gap_present_bits",
            #       gap_present_bits)

    if 0:
        # for development ...
        gap_present_bits_idx = np.arange(len(ngn.tow_edge_ids))
        #
        gap_present_bits0 = '0' * len(ngn.tow_edge_ids)
        gap_present_bits_list = list(gap_present_bits0)
        gap_idx_to_do = 17
        gap_present_bits_list[gap_idx_to_do] = '1'
        gap_present_bits_arr = np.array(gap_present_bits_list).astype(np.int)
        gap_present_bits = ''.join(gap_present_bits_list)
        if 1 and prt:
            print fmt1(just0)[0:] % (
                "np.vstack([gap_present_bits_idx, gap_present_bits_arr])",
                np.vstack([gap_present_bits_idx, gap_present_bits_arr]))
        if 1 and prt:
            print fmt0(just0)[0:] % ("gap_present_bits",
                gap_present_bits)
        None if 1 else sys.exit()

    ps_gaps = []
    for gap_idx, (roi_tow_id_lf, roi_tow_id_rt, roi_tow_edge_xref_lf,
        roi_tow_edge_xref_rt, gap_present_bit) in enumerate(zip(
            roi_tow_ids_lf, roi_tow_ids_rt,
            roi_tow_edge_xrefs_lf, roi_tow_edge_xrefs_rt,
            list(gap_present_bits))):
        if 1 and prt and gap_idx == 0:
            print

        if gap_present_bit != '1':
            continue

        meast_id = indy + 1
        gap_id = "%.2i%.2i" % (roi_tow_id_lf, roi_tow_id_rt)
        if 1 and prt:
            print "%s" % ('. . ' * ngn.mult)
            print fmt0(just0)[0:] % ("[indy, meast_id, gap_idx, " +
                "roi_tow_id_lf, roi_tow_id_rt, gap_id, gap_present_bit]",
                [indy, meast_id, gap_idx, roi_tow_id_lf, roi_tow_id_rt,
                    gap_id, gap_present_bit])

        ps_gap = identify_roi_profile_features(indy, profile_xs_nnan_where,
            profile_xs, nnan_xs, fltr_zs, nnan_xs_midp, fltr_dzdxs, segm_ps,
            meast_id, gap_idx, roi_tow_id_lf, roi_tow_id_rt, gap_id,
            roi_tow_edge_xref_lf, roi_tow_edge_xref_rt)
        ps_gaps.append(ps_gap)
        #f 1 or  prt:
        #   print fmt1(just0)[0:] % ("ps_gap", ps_gap)

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        #f 1 and prt:
        #   print "%s" % ('....' * 40)
        #f 0 and gap_idx >= gap_idx_to_do:
        #   break
        #break

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    #f 0 and prt:
    #   for gap_ps in ps_gaps:
    #       print fmt1(just0)[0:] % ("gap_ps", gap_ps)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return ps_gaps


def make_gallery01_profile_plot(number_of_profiles, indy, pts_fov, pts_roi,
pts_value, pts_drop, zs_median_filter_size, zs_gaussian_filter_sigma,
nnan_xs, nnan_zs, fltr_zs, nnan_xs_midp, nnan_dzdxs, fltr_dzdxs, segm_ps):
    """
    Makes a plot of z-coordinate values and differences for a laser profile.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_gallery01_profile_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 0 and prt_ else False

    fltr_zs_mask_edges_pos = segm_ps.fltr_zs_mask_edges_pos
    fltr_zs_mask_edges_neg = segm_ps.fltr_zs_mask_edges_neg
    fltr_dzdxs_mask_edges_pos = segm_ps.fltr_dzdxs_mask_edges_pos
    fltr_dzdxs_mask_edges_neg = segm_ps.fltr_dzdxs_mask_edges_neg

    nnan_xs_max = np.nanmax(nnan_xs)
    nnan_xs_min = np.nanmin(nnan_xs)
    nnan_xs_mid = (nnan_xs_max + nnan_xs_min) / 2.

    fltr_zs_max = np.nanmax(fltr_zs)
    fltr_zs_min = np.nanmin(fltr_zs)
    fltr_zs_mid = (fltr_zs_max + fltr_zs_min) / 2.

    #== === === === === === === === === === === === === === === === === === ===
    title_fontsize = 12
    text_fontsize = 10
    text_fontsize2 = 8
    ax1_get_ylim_rng_mult1 = 0.95
    ax1_get_ylim_rng_mult2 = 0.87
    ax1_get_ylim_rng_mult3 = 0.08
    axvline_ymin = 0.01
    axvline_ymax = 0.99
    axhline_xmin = 0.01
    axhline_xmax = 0.99
    ms_data = 2.
    lw_data = 0.5
    ax_text_xref1 = 0.02
    ax_text_yref1 = 0.1
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    fig = plt.gcf()
    fig.set_size_inches(16, 12, forward=True)  # default is (8, 6)
    fig.suptitle("%s:\n%s, %s" % ("Laser Profile (X, Z) %i of %i" %
        ((indy + 1), number_of_profiles), ngn.job_zs_csv, ngn.job_id))
    gridspec = [2, 1]
    gs = mpl.gridspec.GridSpec(*gridspec)
    #
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax1 = plt.subplot(gs[0])  # zs
    ax1.set_title('Given & Flitered "Not NaN" Profiles  (%s, %s, %s, %s)' % (
        "Point Counts: FOV: %i" % pts_fov, "ROI: %i" % pts_roi,
        "Values: %i" % pts_value, "Drops: %i" % pts_drop),
        fontsize=title_fontsize)
    #x1.set_xlabel('X-Coordinate (Profile Centered on 0.0)')
    ax1.set_ylabel('Z-Coordinate')

    ax1_ylim_mgn = 1.70
    nanmax = np.nanmax(fltr_zs)
    nanmin = np.nanmin(fltr_zs)
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ax1_ylim_mgn
    ax1.set_ylim((nanmid - nanmgn, nanmid + nanmgn))
    if 0 and prt:
        print fmt0(just1)[0:] % ("[nanmin, nanmax]", [nanmin, nanmax])
        print fmt0(just1)[1:] % ("ax1.get_ylim()", ax1.get_ylim())

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    ax1.plot(nnan_xs, nnan_zs,
        'co-', mec='none', ms=ms_data, lw=lw_data,
        label='Data')

    ax1.plot(nnan_xs, fltr_zs,
        'bo-', mec='none', ms=ms_data, lw=lw_data,
        label='%s & %s' % (
            'Median Filtered (size: %i pts)' % zs_median_filter_size,
            'Gaussian Filtered (sigma: %.1f)' % zs_gaussian_filter_sigma))

    fltr_zs_edges_pos = fltr_zs.copy()
    fltr_zs_edges_pos[~ fltr_zs_mask_edges_pos] = np.nan
    ax1.plot(nnan_xs, fltr_zs_edges_pos, 'ro-', mec='none',
        lw=1., label="positive edge points")

    fltr_zs_edges_neg = fltr_zs.copy()
    fltr_zs_edges_neg[~ fltr_zs_mask_edges_neg] = np.nan
    ax1.plot(nnan_xs, fltr_zs_edges_neg, 'mo-', mec='none',
        lw=1., label="negative edge points")

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    for i, tow_center_xref in enumerate(ngn.tow_center_xrefs):
        ax1_axvline_ls = (
            'dashed' if i == 0 or (i + 1) == len(ngn.tow_center_xrefs)
            else 'dotted')
        ax1.axvline(x=tow_center_xref, ymin=axvline_ymin,
            ymax=axvline_ymax, c='y', ls=ax1_axvline_ls, lw=2.)

    trans1 = mpl.transforms.blended_transform_factory(
        ax1.transData, ax1.transAxes)

    zipped = zip(ngn.tow_ids[1:-1], ngn.tow_center_xrefs[1:-1])
    for i11, (tid, tcx) in enumerate(zipped):
        ax1.text(tcx, 0.94 if i11 % 2 else 0.99, "tow\n%.2i" % tid,
            color='black', fontsize=text_fontsize2, fontweight='bold',
            ha='center', va='top', transform=trans1)

    ax1.axvline(x=nnan_xs_min,
        ymin=axvline_ymin, ymax=axvline_ymax, c='k', ls='dashed')
    ax1.text(nnan_xs_min, ax_text_yref1,
        'X Min. (mm):\n%.3f' % nnan_xs_min,
        fontsize=text_fontsize, fontweight='bold', ha='center', va='bottom',
        transform=trans1)

    ax1.axvline(x=nnan_xs_mid,
        ymin=axvline_ymin, ymax=axvline_ymax, c='k', ls='dashed')
    ax1.text(nnan_xs_mid, ax_text_yref1,
        'X Mid. (mm):\n%.3f' % nnan_xs_mid,
        fontsize=text_fontsize, fontweight='bold', ha='center', va='bottom',
        transform=trans1)

    ax1.axvline(x=nnan_xs_max,
        ymin=axvline_ymin, ymax=axvline_ymax, c='k', ls='dashed')
    ax1.text(nnan_xs_max, ax_text_yref1,
        'X Max. (mm):\n%.3f' % nnan_xs_max,
        fontsize=text_fontsize, fontweight='bold', ha='center', va='bottom',
        transform=trans1)

    trans1 = mpl.transforms.blended_transform_factory(
        ax1.transAxes, ax1.transData)

    ax_text_xref1 = nnan_xs_min
    ax1.axhline(y=fltr_zs_min,
        xmin=axhline_xmin, xmax=axhline_xmax, c='k', ls='dashed', lw=2.)
    ax1.text(ax_text_xref1, fltr_zs_min,
        'Z Min. (mm):\n%.3f' % fltr_zs_min,
        fontsize=text_fontsize, fontweight='bold', ha='right', va='bottom')
    #
    ax1.axhline(y=fltr_zs_mid,
        xmin=axhline_xmin, xmax=axhline_xmax, c='k', ls='dashed', lw=2.)
    ax1.text(ax_text_xref1, fltr_zs_mid,
        'Z Mid. (mm):\n%.3f' % fltr_zs_mid,
        fontsize=text_fontsize, fontweight='bold', ha='right', va='bottom')
    #
    ax1.axhline(y=fltr_zs_max,
        xmin=axhline_xmin, xmax=axhline_xmax, c='k', ls='dashed', lw=2.)
    ax1.text(ax_text_xref1, fltr_zs_max,
        'Z Max. (mm):\n%.3f' % fltr_zs_max,
        fontsize=text_fontsize, fontweight='bold', ha='right', va='bottom')

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
    ax1.legend(
        loc=8,
        ncol=4,
        numpoints=1,
        markerscale=1.,
        prop={'size': 9.2, 'weight': 'bold'}
    ) if 1 else None
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    ax2 = plt.subplot(gs[1], sharex=ax1)  # dzdxs
    ax2.set_title('First Difference of Given & Flitered "Not NaN" Profiles',
        fontsize=title_fontsize)
    ax2.set_xlabel('X-Coordinate (Profile Centered on 0.0)')
    ax2.set_ylabel('Z-Coordinate First Difference')

    nnan_xs_and_tow_center_xrefs = (
        np.concatenate((nnan_xs, ngn.tow_center_xrefs)))
    ax2_set_xlim_max = np.nanmax(np.abs(nnan_xs_and_tow_center_xrefs)) * 1.25
    ax2_set_xlim_min = -ax2_set_xlim_max
    ax2.set_xlim((ax2_set_xlim_min, ax2_set_xlim_max))

    ax2_ylim_mgn = 3.0
    nanmax = ngn.dzdxs_threshold
    #anmin = np.nanmin(fltr_dzdxs)
    nanmin = -nanmax
    nanmid = (nanmax + nanmin) / 2.
    nanrng = (nanmax - nanmin) / 2.
    nanmgn = nanrng * ax2_ylim_mgn
    ax2.set_ylim((nanmid - nanmgn, nanmid + nanmgn))
    if 0 and prt:
        print fmt0(just1)[0:] % ("[nanmin, nanmax]", [nanmin, nanmax])
        print fmt0(just1)[1:] % ("ax2.get_ylim()", ax2.get_ylim())

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    ax2.plot(nnan_xs_midp, nnan_dzdxs,
        'co-', mec='none', ms=ms_data, lw=lw_data,
        label='"Not NaN" Data') if 0 else None
    ax2.plot(nnan_xs_midp, fltr_dzdxs,
        'bo-', mec='none', ms=ms_data, lw=lw_data,
        label='Not NaN" Data %s & %s' % (
            'Median Filtered (size: %i pts)' % zs_median_filter_size,
            'Gaussian Filtered (sigma: %.1f)' % zs_gaussian_filter_sigma))

    fltr_dzdxs_edges_pos = fltr_dzdxs.copy()
    fltr_dzdxs_edges_pos[~ fltr_dzdxs_mask_edges_pos] = np.nan
    ax2.plot(nnan_xs_midp, fltr_dzdxs_edges_pos, 'ro-', mec='none',
        lw=1., label="positive edges")

    fltr_dzdxs_edges_neg = fltr_dzdxs.copy()
    fltr_dzdxs_edges_neg[~ fltr_dzdxs_mask_edges_neg] = np.nan
    ax2.plot(nnan_xs_midp, fltr_dzdxs_edges_neg, 'mo-', mec='none',
        lw=1., label="negative edges")

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

    for i, tow_center_xref in enumerate(ngn.tow_center_xrefs):
        ax2_axvline_ls = (
            'dashed' if i == 0 or (i + 1) == len(ngn.tow_center_xrefs)
            else 'dotted')
        ax2.axvline(x=tow_center_xref, ymin=axvline_ymin,
            ymax=axvline_ymax, c='y', ls=ax2_axvline_ls, lw=2.)

    ax2.axvline(x=nnan_xs_min,
        ymin=axvline_ymin, ymax=axvline_ymax, c='k', ls='dashed')
    #
    ax2.axvline(x=nnan_xs_mid,
        ymin=axvline_ymin, ymax=axvline_ymax, c='k', ls='dashed')
    #
    ax2.axvline(x=nnan_xs_max,
        ymin=axvline_ymin, ymax=axvline_ymax, c='k', ls='dashed')

    trans2 = mpl.transforms.blended_transform_factory(
        ax2.transData, ax2.transAxes)

    zipped = zip(ngn.tow_ids[1:-1], ngn.tow_center_xrefs[1:-1])
    for i22, (tid, tcx) in enumerate(zipped):
        ax2.text(tcx, 0.94 if i22 % 2 else 0.99, "tow\n%.2i" % tid,
            color='black', fontsize=text_fontsize2, fontweight='bold',
            ha='center', va='top', transform=trans2)

    ax_text_xref2 = ngn.tow_center_xrefs[-1]
    ax2.axhline(y=-ngn.dzdxs_threshold,
        xmin=axhline_xmin, xmax=axhline_xmax, c='k', ls='dotted', lw=2.)
    ax2.text(ax_text_xref2, -ngn.dzdxs_threshold,
        '-dZdX Threshold (mm):  %.3f' % -ngn.dzdxs_threshold,
        fontsize=text_fontsize2, fontweight='bold', ha='right', va='bottom')
    #
    ax2.axhline(y=ngn.dzdxs_threshold,
        xmin=axhline_xmin, xmax=axhline_xmax, c='k', ls='dotted', lw=2.)
    ax2.text(ax_text_xref2, ngn.dzdxs_threshold,
        'dZdX Threshold (mm):  %.3f' % ngn.dzdxs_threshold,
        fontsize=text_fontsize2, fontweight='bold', ha='right', va='top')

    #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
    ax2.legend(
        loc=8,
        ncol=5,
        numpoints=1,
        markerscale=1.,
        prop={'size': 8.0, 'weight': 'bold'}
    ) if 1 else None
    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    png_abspath = os.path.join(ngn.gallery01_absdir,
        ngn.job_zs_csv.replace('z', '').replace('.txt', '')
        .replace('.csv', '_meast_%.5i.png' % (indy + 1)))
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.gallery01_absdir", ngn.gallery01_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    #
    png = png_abspath if 1 else None
    if png is not None:
        None if os.path.isdir(ngn.gallery01_absdir) else (
            os.makedirs(ngn.gallery01_absdir))
        plt.savefig(png)
    else:
        plt.show()
    plt.close()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


#zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz
# (below) defs ... for analyzing gaps between tows using "smart gap" analysis


def analyze_laser_measurement_profile(number_of_profiles, indy,
profile_xs, profile_zs, profile_zs_mask_nans, results_gap_ps,
gap_present_bits):
    """
    Returns a Pandas Series containing gap results for one laser profile..
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt__ = False if 1 else True  # def print switch
    prt_ = prt
    mult_str = '--- '
    def_str = 'analyze_laser_measurement_profile_legacy_gap_analysis'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===
    # generate the Not "NaN"s profiles (nnan) dataset
    #
    #   note:
    #     the not "NaN" profile requires a series of points with unique and
    #     monotonically increasing x values

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    profile_zs_mask_nnans = ~ profile_zs_mask_nans
    profile_zs_mask_nnans_where = np.where(profile_zs_mask_nnans)[0]
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("profile_zs_mask_nnans.shape",
    #       profile_zs_mask_nnans.shape)
    #   print fmt0(just1)[1:] % ("np.sum(profile_zs_mask_nnans)",
    #       np.sum(profile_zs_mask_nnans))
    #   #rint fmt1(just1)[1:] % ("list(profile_zs_mask_nnans_where)",
    #   #   list(profile_zs_mask_nnans_where))
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    nnan_xs = profile_xs[~ profile_zs_mask_nans]
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs.shape", nnan_xs.shape)

    ### profile_xs monotonic "check"
    #
    # make gallery 00 profile plot
    if 1 and ngn.write_to_results_dir and ngn.make_gallery00_plots:
        profile_dxs = profile_xs[1:] - profile_xs[:-1]
        make_gallery00_profile_plot(
            number_of_profiles, indy, profile_xs, profile_dxs)
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    profile_xs_nnan_where = np.where(~ profile_zs_mask_nans)[0]
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("profile_xs.shape", profile_xs.shape)
    #   print fmt0(just1)[1:] % ("profile_zs.shape", profile_zs.shape)
    #   print fmt0(just1)[1:] % ("profile_zs_mask_nans.shape",
    #       profile_zs_mask_nans.shape)
    #   print fmt0(just1)[1:] % ("np.sum(~ profile_zs_mask_nans)",
    #       np.sum(~ profile_zs_mask_nans))
    #   print fmt0(just1)[1:] % ("profile_xs_nnan_where.shape",
    #       profile_xs_nnan_where.shape)
    #   print fmt0(just1)[1:] % ("profile_xs_nnan_where[:9],...,[-9:]",
    #       [profile_xs_nnan_where[:9], "...", profile_xs_nnan_where[-9:]])
    #f 0 and prt:
    #   nnan_xs_where = np.where(np.ones(nnan_xs.shape).astype(np.bool))[0]
    #   print fmt0(just1)[1:] % ("nnan_xs_where.shape", nnan_xs_where.shape)
    #   print fmt0(just1)[1:] % ("nnan_xs_where[:9],...,[-9:]",
    #       [nnan_xs_where[:9], "...", nnan_xs_where[-9:]])
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # remove 'NaN' values from the measurement profile
    # and calculate profile first differences & statistics

    nnan_zs = profile_zs[~ profile_zs_mask_nans]
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_zs.shape", nnan_zs.shape)
    #one if 1 else sys.exit()

    nnan_xs_midp = (nnan_xs[1:] + nnan_xs[:-1]) / 2.
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs_midp.shape", nnan_xs_midp.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_xs_midp", nnan_xs_midp)
    #one if 1 else sys.exit()

    nnan_xs_diff = (nnan_xs[1:] - nnan_xs[:-1])
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs_diff.shape", nnan_xs_diff.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_xs_diff", nnan_xs_diff)
    #   print fmt1(just1)[1:] % ("nnan_xs_diff[:6]", nnan_xs_diff[:6])

    #f 1 and prt:
    #   print fmt0(just1)[0:] % (
    #       "np.all(nnan_xs_diff > 0.)", np.all(nnan_xs_diff > 0.))
    #ssert np.all(nnan_xs_diff > 0.)

    nnan_zs_diff = (nnan_zs[1:] - nnan_zs[:-1])
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("nnan_zs_diff.shape", nnan_zs_diff.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_zs_diff", nnan_zs_diff)
    #   print fmt1(just1)[1:] % ("nnan_zs_diff[:6]", nnan_zs_diff[:6])

    nnan_dzdxs = nnan_zs_diff / nnan_xs_diff
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("nnan_dzdxs.shape", nnan_dzdxs.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_dzdxs", nnan_dzdxs)
    #   print fmt1(just1)[1:] % ("nnan_dzdxs[:6]", nnan_dzdxs[:6])

    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs_diff.shape", nnan_xs_diff.shape)
    #   print fmt0(just1)[0:] % ("nnan_zs_diff.shape", nnan_zs_diff.shape)
    #   print fmt0(just1)[0:] % ("nnan_dzdxs.shape", nnan_dzdxs.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_dzdxs", nnan_dzdxs)
    #   print fmt0(just1)[1:] % ("nnan_dzdxs[:6]", nnan_dzdxs[:6])
    #one if 1 else sys.exit()

    if 0:
        # zs resolution estimate
        if 1 and prt:
            print fmt1(just2)[0:] % ("nnan_zs", nnan_zs)

        nnan_zs_uniq = np.sort(np.unique(nnan_zs))
        if 1 and prt:
            print fmt1(just2)[0:] % ("nnan_zs_uniq", nnan_zs_uniq)

        nnan_zs_uniq_diff = nnan_zs_uniq[1:] - nnan_zs_uniq[:-1]
        if 1 and prt:
            print fmt1(just2)[0:] % ("nnan_zs_uniq_diff", nnan_zs_uniq_diff)

        nnan_zs_uniq_diff_uniq = np.sort(np.unique(nnan_zs_uniq_diff))
        if 1 and prt:
            print fmt1(just2)[0:] % ("nnan_zs_uniq_diff_uniq",
                nnan_zs_uniq_diff_uniq)

        nnan_zs_uniq_diff_uniq_mean = np.mean(nnan_zs_uniq_diff_uniq)
        if 1 and prt:
            print fmt0(just2)[0:] % (
                "nnan_zs_uniq_diff_uniq_mean", nnan_zs_uniq_diff_uniq_mean)

        None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    # assess profile 'quality'

    (nnan_zs_where, zs_idx_nnan_lf, zs_idx_nnan_rt,
        pts_fov, pts_roi, pts_value, pts_drop) = (
            calculate_profile_quality_metrics(profile_zs_mask_nans))
    #f 0 and prt:
    #   print fmt0(just1)[0:] % (
    #       "nnan_zs_where.shape", nnan_zs_where.shape)
    #   print fmt0(just1)[1:] % ("zs_idx_nnan_lf", zs_idx_nnan_lf)
    #   print fmt0(just1)[1:] % ("zs_idx_nnan_rt", zs_idx_nnan_rt)
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("pts_fov", pts_fov)
    #   print fmt0(just1)[1:] % ("pts_value", pts_value)
    #   print fmt0(just1)[1:] % ("pts_roi", pts_roi)
    #   print fmt0(just1)[1:] % ("pts_drop", pts_drop)
    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    # generate the filtered profiles (fltr) dataset

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # apply a median filter & a Gaussian filter ...
    # ... to the (nnan) profile to suppress noise for edge detection

    ### ### The median filter size chosen is ...
    #
    # the default value chosen
    #zs_median_filter_size = 1  # no effect (default)
    #zs_median_filter_size = zs_median_filter_size if 1 else 3  # "gentlest"
    #zs_median_filter_size = zs_median_filter_size if 1 else 5
    #zs_median_filter_size = zs_median_filter_size if 1 else 7
    #zs_median_filter_size = zs_median_filter_size if 1 else 9
    #zs_median_filter_size = zs_median_filter_size if 0 else 11
    #
    ### the value chosen from the parameters configuration file
    zs_median_filter_size = ngn.zs_median_filter_size
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("zs_median_filter_size",
    #       zs_median_filter_size)

    ### ### The gaussian filter size is chosen to be ...
    #
    # the default value chosen
    #                                                            pts > 0.0
    #zs_gaussian_filter_sigma = 0.0  # no effect (default)              # 1
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 1 else 0.3  # 3
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 1 else 0.5  # 5
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 1 else 0.8  # 7
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 1 else 1.0  # 9
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 0 else 1.2
    #
    ### the value chosen from the parameters configuration file
    zs_gaussian_filter_sigma = ngn.zs_gaussian_filter_sigma
    #f 1 and prt:
    #   print fmt0(just1)[1:] % ("zs_gaussian_filter_sigma",
    #       zs_gaussian_filter_sigma)

    fltr_zs = ndi.gaussian_filter(ndi.median_filter(
        nnan_zs, zs_median_filter_size), zs_gaussian_filter_sigma)

    #one if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
    # perform first difference calculations

    fltr_xs_diff = (nnan_xs[1:] - nnan_xs[:-1])
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("fltr_xs_diff.shape", fltr_xs_diff.shape)
    #   #rint fmt1(just1)[1:] % ("fltr_xs_diff", fltr_xs_diff)
    #   #rint fmt0(just1)[1:] % ("fltr_xs_diff[:6]", fltr_xs_diff[:6])
    #   print fmt1(just1)[0:] % ("fltr_xs_diff", fltr_xs_diff)

    fltr_zs_diff = (fltr_zs[1:] - fltr_zs[:-1])
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("fltr_zs_diff.shape", fltr_zs_diff.shape)
    #   #rint fmt1(just1)[1:] % ("fltr_zs_diff", fltr_zs_diff)
    #   print fmt0(just1)[1:] % ("fltr_zs_diff[:6]", fltr_zs_diff[:6])

    fltr_dzdxs = fltr_zs_diff / fltr_xs_diff
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("fltr_dzdxs.shape", fltr_dzdxs.shape)
    #   #rint fmt1(just1)[1:] % ("fltr_dzdxs", fltr_dzdxs)
    #   print fmt0(just1)[1:] % ("fltr_dzdxs[:6]", fltr_dzdxs[:6])

    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    # identify gap features in the measurement profile

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # segment edges and flats in the (filtered) laser measurement profile

    prt = True if 1 and prt_ else False

    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("ngn.dzdxs_threshold", ngn.dzdxs_threshold)

    segm_ps = segment_edges_and_flats(number_of_profiles, indy,
        fltr_dzdxs, ngn.dzdxs_threshold, nnan_xs_midp, nnan_xs)
    #f 0 and prt:
    #   print fmt1(just1)[0:] % ("sorted(segm_ps.keys())",
    #       prt_list(sorted(segm_ps.keys()), 0))

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # make gallery 01 profile plot

    if 1 and ngn.write_to_results_dir and ngn.make_gallery01_plots:

        make_gallery01_profile_plot(
            number_of_profiles, indy,
            pts_fov, pts_roi, pts_value, pts_drop,
            zs_median_filter_size, zs_gaussian_filter_sigma,
            nnan_xs, nnan_zs, fltr_zs,
            nnan_xs_midp, nnan_dzdxs, fltr_dzdxs,
            segm_ps,
        )
        None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # identify gap (& strip) features in each region of interest

    prt = True if 1 and prt_ else False

    ps_gaps = identify_profile_features_in_rois(indy,
        profile_xs_nnan_where, profile_xs, nnan_xs, fltr_zs,
        nnan_xs_midp, fltr_dzdxs, segm_ps, gap_present_bits)
    if 0 and prt:
        print
        for ps_gap in ps_gaps:
            print fmt0(just1)[1:] % ("dict(ps_gap)", dict(ps_gap))
        None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    # post results

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # post laser measurement profile (raw) edges to the edges image map

#   if 0 and prt:
#       print fmt0(just1)[0:] % ("profile_zs_mask_nnans.shape",
#           profile_zs_mask_nnans.shape)
#       print fmt0(just1)[1:] % ("np.sum(profile_zs_mask_nnans)",
#           np.sum(profile_zs_mask_nnans))
#       print fmt1(just1)[1:] % ("list(profile_zs_mask_nnans_where)",
#           list(profile_zs_mask_nnans_where))
#
#   #fltr_zs_mask_edges = segm_ps.fltr_zs_mask_edges
#   if 0 and prt:
#       print fmt0(just1)[0:] % ("segm_ps.fltr_zs_mask_edges.shape",
#           segm_ps.fltr_zs_mask_edges.shape)
#       print fmt0(just1)[1:] % ("np.sum(segm_ps.fltr_zs_mask_edges)",
#           np.sum(segm_ps.fltr_zs_mask_edges))
#
    profile_zs_mask_nnans_where_edges = (
        profile_zs_mask_nnans_where[segm_ps.fltr_zs_mask_edges])
    if 0 and prt:
        print fmt0(just1)[0:] % ("profile_zs_mask_nnans_where_edges.shape",
            profile_zs_mask_nnans_where_edges.shape)
        print fmt1(just1)[1:] % ("list(profile_zs_mask_nnans_where_edges)",
            list(profile_zs_mask_nnans_where_edges))

    ngn.np_edges_image[indy, profile_zs_mask_nnans_where_edges] = True
    if 0 and prt:
        print fmt0(just1)[0:] % ("indy", indy)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # post (tabular) gap results

    results_gap_ps = post_gap_results_legacy_gap_analysis(
        zs_idx_nnan_lf, zs_idx_nnan_rt, pts_fov, pts_roi, pts_value, pts_drop,
        ps_gaps, results_gap_ps)
    #f 0 and prt:
    #   print fmt1(just1)[0:] % ("(updt) results_gap_ps[:20]",
    #       results_gap_ps[:20])
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("results_gap_ps", results_gap_ps)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return results_gap_ps


def analyze_tow_gaps(np_src_xs, np_hasnan_zs, np_hasnan_zs_mask_nans,
pd_src_us_gap_any, pd_results_gap):
    """
    Returns a Pandas Series containing gap results for one laser profile..
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt__ = False if 1 else True  # def print switch
    prt_ = prt
    mult_str = '--- '
    def_str = 'analyze_tow_gaps'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #== === === === === === === === === === === === === === === === === === ===
    # (below) analyze gaps for each laser measurement profile

    prt = True if 1 and prt_ else False

    if 1 and prt:
        print fmt0(just1)[0:] % ("gap(/butt/lap) algorithm", "begin")
    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if 1 and prt:
        print fmt0(just1)[0:] % ("np_src_xs.shape", np_src_xs.shape)
        print fmt0(just1)[1:] % ("np_hasnan_zs.shape", np_hasnan_zs.shape)
        print fmt0(just1)[1:] % ("np_hasnan_zs_mask_nans.shape",
            np_hasnan_zs_mask_nans.shape)

    if 1 and prt:
        print fmt0(just1)[0:] % ("pd_src_us_gap_any.shape",
            pd_src_us_gap_any.shape)

    smart_gap_laser_profiles_skipped = (
        np_src_xs.shape[0] - pd_src_us_gap_any.shape[0])
    if 1 and prt:
        print fmt0(just1)[0:] % ("smart_gap_laser_profiles_skipped",
            smart_gap_laser_profiles_skipped)

    if 1 and prt:
        #rint fmt1(just1)[0:] % ("pd_src_us_gap_any.columns",
        #   pd_src_us_gap_any.columns)
        hd, tl = (4, 4) if 1 else (10, 10)
        print fmt1(just1)[0:] % (
            "pd_src_us_gap_any.head(%i)" % hd,
            pd_src_us_gap_any.head(hd))
        print fmt1(just1)[0:] % (
            "pd_src_us_gap_any.tail(%i)" % tl,
            pd_src_us_gap_any.tail(tl))
    if 0 and prt:
        print fmt1(just1)[0:] % ("pd_src_us_gap_any", pd_src_us_gap_any)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    number_of_profiles = np_src_xs.shape[0]
    if 1 and prt:
        print fmt0(just1)[0:] % ("number_of_profiles",
            number_of_profiles)

    number_of_any_gap_profiles = pd_src_us_gap_any.shape[0]
    if 1 and prt:
        print fmt0(just1)[0:] % ("number_of_any_gap_profiles",
            number_of_any_gap_profiles)

    iany_last = number_of_any_gap_profiles - 1  # default
    #iany_last = iany_last if 1 else 0
    #iany_last = iany_last if 1 else 1
    #iany_last = iany_last if 1 else 2
    #iany_last = iany_last if 1 else 9
    #iany_last = iany_last if 1 else 19
    #iany_last = iany_last if 1 else 120
    #iany_last = iany_last if 1 else 167
    #iany_last = iany_last if 1 else 189
    #
    iany_first = 0  # default
    #iany_first = iany_first if 1 else 168
    #iany_first = iany_first if 1 else 190
    #iany_first = iany_first if 1 else 167
    #iany_first = iany_first if 1 else 189
    iany_first = iany_first if 1 else 9999999999  # make same as iany_last
    #
    iany_first = iany_first if iany_first <= iany_last else iany_last
    if 1 or prt:
        print fmt0(just1)[0:] % (
            "[0, iany_first, iany_last, number_of_any_gap_profiles - 1]",
            [0, iany_first, iany_last, number_of_any_gap_profiles - 1])
    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    print "\n... 'smart' gap profiles analysis begin %s" % ('... ' * ngn.mult)

    if ngn.count_gap_rois_analyzed:
        initialize_gap_rois_analyzed_counts()
    if 0 and prt:
        print fmt0(just2)[0:] % ("ngn.count_gap_rois_analyzed",
            ngn.count_gap_rois_analyzed)

    ps_results_gap = pd.Series(np.full((len(ngn.results_fields_gap)), np.nan),
        index=ngn.results_fields_gap)
    if 1 and prt:
        print fmt0(just1)[0:] % ("ps_results_gap.shape",
            ps_results_gap.shape)
        print fmt1(just1)[1:] % ("ps_results_gap", ps_results_gap)

    None if 1 else sys.exit()

    for row in pd_src_us_gap_any.itertuples():
        iany = row[0]
        #f 0 and prt:
        #   print fmt1(just1)[0:] % ("row", row)
        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
        if iany < iany_first:
            continue

        #(profile_id, meast_id, u_sensor, gap_present_bits_gap3233_to_gap0001,
        #    meast_idx) = row[1:]
        gap_present_bits_gap3233_to_gap0001, meast_idx = row[-2:]
        # put in laser profile (ascending gap id) order ...
        gap_present_bits = gap_present_bits_gap3233_to_gap0001[::-1]
        indy = meast_idx

        #rint "\n... ... ... ... ... ... ... ... %s" % ('... ' * ngn.mult)
        if (iany % 100 == 0):
            print fmt0(just1)[1:] % (
                "[iany, number_of_any_gap_profiles, indy, number_of_profiles]",
                [iany, number_of_any_gap_profiles, indy, number_of_profiles])
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("[iany, profile_id, meast_id, u_sensor," +
        #       " gap_present_bits, meast_idx]", [iany, profile_id, meast_id,
        #       u_sensor, gap_present_bits, meast_idx])
        #f 1 and prt:
        #   print fmt0(just1)[1:] % ("gap_present_bits", gap_present_bits)
        #one if 0 else sys.exit()

        try:
            results_gap_ps = analyze_laser_measurement_profile(
                number_of_profiles, indy, np_src_xs[indy, :],
                np_hasnan_zs[indy, :], np_hasnan_zs_mask_nans[indy, :],
                ps_results_gap.copy(), gap_present_bits)

            pd_results_gap.loc[indy, :] = results_gap_ps.values
            #f 1:
            #   #rint fmt1(just1)[0:] % ("results_gap_ps", results_gap_ps)
            #   #rint fmt1(just1)[0:] % (
            #   #   "pd_results_gap.loc[indy]", pd_results_gap.loc[indy])
            #   print fmt1(just1)[0:] % (
            #       "pd_results_gap.loc[[indy]]", pd_results_gap.loc[[indy]])
            #   None if 1 else sys.exit()

        except Exception, err:
            #print 'print_exc():'
            #traceback.print_exc(file=sys.stdout)
            print "\n... iany: %i ... err: %s\n" % (iany, err)

        finally:
            # this gets executed regardless ...
            None if 1 else sys.exit()

        if ngn.count_gap_rois_analyzed:
            ngn.count_analyzed_measts += 1

        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
        if 1 and iany >= iany_last:
            break
        #break

    if ngn.count_gap_rois_analyzed:
        report_gap_rois_analyzed_counts()

    print "\n... 'smart' gap profiles analysis end %s" % ('... ' * ngn.mult)

    # (above) analyze gaps for each laser measurement profile
    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return pd_results_gap


# (above) defs ... for analyzing gaps between tows using "smart gap" analysis
#zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz
# (below) defs ... for analyzing gaps between tows using "dump gap" analysis


def post_gap_results_legacy_gap_analysis(zs_idx_nnan_lf, zs_idx_nnan_rt,
pts_fov, pts_roi, pts_value, pts_drop, ps_gaps, results_gap_ps):
    """
    Updates the Pandas Series containing all gap results for a laser profile
    using each Pandas Series containing results for a gap region of interest.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt__ = False if 1 else True  # def print switch
    prt_ = prt
    mult_str = '--- '
    def_str = 'post_gap_results'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 0 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    results_gap_ps['zs_idx_nnan_lf'] = zs_idx_nnan_lf
    results_gap_ps['zs_idx_nnan_rt'] = zs_idx_nnan_rt
    results_gap_ps['pts_fov'] = pts_fov
    results_gap_ps['pts_roi'] = pts_roi
    results_gap_ps['pts_value'] = pts_value
    results_gap_ps['pts_drop'] = pts_drop

    for idx, ps_gap in enumerate(ps_gaps):
        gap_id = ps_gap.gap_id
        #f 1 and prt:
        #   print fmt0(just1)[0:] % ("[idx, ps_gap.gap_id]",
        #       [idx, ps_gap.gap_id])
        #   print fmt1(just1)[1:] % ("ps_gap", ps_gap)

        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::

        ### Gap0001Present nan

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        if ~ np.isnan(ps_gap.edge_neg_x_midp_nnan_idx):

            results_gap_ps['Gap%sEdgeLfNnanIdx' % gap_id] = (
                ps_gap['edge_neg_x_midp_nnan_idx'])

            results_gap_ps['Gap%sEdgeLfIdx' % gap_id] = (
                ps_gap['edge_neg_x_midp_idx'])

            results_gap_ps['Gap%sEdgeLfX' % gap_id] = (
                ps_gap['edge_neg_x_midp'])

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        if ~ np.isnan(ps_gap.edge_pos_x_midp_nnan_idx):

            results_gap_ps['Gap%sEdgeRtNnanIdx' % gap_id] = (
                ps_gap['edge_pos_x_midp_nnan_idx'])

            results_gap_ps['Gap%sEdgeRtIdx' % gap_id] = (
                ps_gap['edge_pos_x_midp_idx'])

            results_gap_ps['Gap%sEdgeRtX' % gap_id] = (
                ps_gap['edge_pos_x_midp'])

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        if ~ np.isnan(ps_gap.gap_width):

            results_gap_ps['Gap%sWidth' % gap_id] = (
                ps_gap['gap_width'])

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        #results_gap_ps['Gap%sClass' % gap_id] = (
        #    ps_gap['gap_class'])

        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
        #break

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    #f 0 and prt:
    #   print fmt1(just1)[0:] % ("results_gap_ps", results_gap_ps)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return results_gap_ps


def analyze_laser_measurement_profile_legacy_gap_analysis(number_of_profiles,
indy, profile_xs, profile_zs, profile_zs_mask_nans, results_gap_ps):
    """
    Returns a Pandas Series containing gap results for one laser profile..
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt__ = False if 1 else True  # def print switch
    prt_ = prt
    mult_str = '--- '
    def_str = 'analyze_laser_measurement_profile_legacy_gap_analysis'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===
    # generate the Not "NaN"s profiles (nnan) dataset
    #
    #   note:
    #     the not "NaN" profile requires a series of points with unique and
    #     monotonically increasing x values

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    profile_zs_mask_nnans = ~ profile_zs_mask_nans
    profile_zs_mask_nnans_where = np.where(profile_zs_mask_nnans)[0]
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("profile_zs_mask_nnans.shape",
    #       profile_zs_mask_nnans.shape)
    #   print fmt0(just1)[1:] % ("np.sum(profile_zs_mask_nnans)",
    #       np.sum(profile_zs_mask_nnans))
    #   #rint fmt1(just1)[1:] % ("list(profile_zs_mask_nnans_where)",
    #   #   list(profile_zs_mask_nnans_where))
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    nnan_xs = profile_xs[~ profile_zs_mask_nans]
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs.shape", nnan_xs.shape)

    ### profile_xs monotonic "check"
    #
    # make gallery 00 profile plot
    if 1 and ngn.write_to_results_dir and ngn.make_gallery00_plots:
        profile_dxs = profile_xs[1:] - profile_xs[:-1]
        make_gallery00_profile_plot(
            number_of_profiles, indy, profile_xs, profile_dxs)
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    profile_xs_nnan_where = np.where(~ profile_zs_mask_nans)[0]
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("profile_xs.shape", profile_xs.shape)
    #   print fmt0(just1)[1:] % ("profile_zs.shape", profile_zs.shape)
    #   print fmt0(just1)[1:] % ("profile_zs_mask_nans.shape",
    #       profile_zs_mask_nans.shape)
    #   print fmt0(just1)[1:] % ("np.sum(~ profile_zs_mask_nans)",
    #       np.sum(~ profile_zs_mask_nans))
    #   print fmt0(just1)[1:] % ("profile_xs_nnan_where.shape",
    #       profile_xs_nnan_where.shape)
    #   print fmt0(just1)[1:] % ("profile_xs_nnan_where[:9],...,[-9:]",
    #       [profile_xs_nnan_where[:9], "...", profile_xs_nnan_where[-9:]])
    #f 0 and prt:
    #   nnan_xs_where = np.where(np.ones(nnan_xs.shape).astype(np.bool))[0]
    #   print fmt0(just1)[1:] % ("nnan_xs_where.shape", nnan_xs_where.shape)
    #   print fmt0(just1)[1:] % ("nnan_xs_where[:9],...,[-9:]",
    #       [nnan_xs_where[:9], "...", nnan_xs_where[-9:]])
    #one if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # remove 'NaN' values from the measurement profile
    # and calculate profile first differences & statistics

    nnan_zs = profile_zs[~ profile_zs_mask_nans]
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_zs.shape", nnan_zs.shape)
    #one if 1 else sys.exit()

    nnan_xs_midp = (nnan_xs[1:] + nnan_xs[:-1]) / 2.
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs_midp.shape", nnan_xs_midp.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_xs_midp", nnan_xs_midp)
    #one if 1 else sys.exit()

    nnan_xs_diff = (nnan_xs[1:] - nnan_xs[:-1])
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs_diff.shape", nnan_xs_diff.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_xs_diff", nnan_xs_diff)
    #   print fmt1(just1)[1:] % ("nnan_xs_diff[:6]", nnan_xs_diff[:6])

    #f 1 and prt:
    #   print fmt0(just1)[0:] % (
    #       "np.all(nnan_xs_diff > 0.)", np.all(nnan_xs_diff > 0.))
    #ssert np.all(nnan_xs_diff > 0.)

    nnan_zs_diff = (nnan_zs[1:] - nnan_zs[:-1])
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("nnan_zs_diff.shape", nnan_zs_diff.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_zs_diff", nnan_zs_diff)
    #   print fmt1(just1)[1:] % ("nnan_zs_diff[:6]", nnan_zs_diff[:6])

    nnan_dzdxs = nnan_zs_diff / nnan_xs_diff
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("nnan_dzdxs.shape", nnan_dzdxs.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_dzdxs", nnan_dzdxs)
    #   print fmt1(just1)[1:] % ("nnan_dzdxs[:6]", nnan_dzdxs[:6])

    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("nnan_xs_diff.shape", nnan_xs_diff.shape)
    #   print fmt0(just1)[0:] % ("nnan_zs_diff.shape", nnan_zs_diff.shape)
    #   print fmt0(just1)[0:] % ("nnan_dzdxs.shape", nnan_dzdxs.shape)
    #   #rint fmt1(just1)[1:] % ("nnan_dzdxs", nnan_dzdxs)
    #   print fmt0(just1)[1:] % ("nnan_dzdxs[:6]", nnan_dzdxs[:6])
    #one if 1 else sys.exit()

    if 0:
        # zs resolution estimate
        if 1 and prt:
            print fmt1(just2)[0:] % ("nnan_zs", nnan_zs)

        nnan_zs_uniq = np.sort(np.unique(nnan_zs))
        if 1 and prt:
            print fmt1(just2)[0:] % ("nnan_zs_uniq", nnan_zs_uniq)

        nnan_zs_uniq_diff = nnan_zs_uniq[1:] - nnan_zs_uniq[:-1]
        if 1 and prt:
            print fmt1(just2)[0:] % ("nnan_zs_uniq_diff", nnan_zs_uniq_diff)

        nnan_zs_uniq_diff_uniq = np.sort(np.unique(nnan_zs_uniq_diff))
        if 1 and prt:
            print fmt1(just2)[0:] % ("nnan_zs_uniq_diff_uniq",
                nnan_zs_uniq_diff_uniq)

        nnan_zs_uniq_diff_uniq_mean = np.mean(nnan_zs_uniq_diff_uniq)
        if 1 and prt:
            print fmt0(just2)[0:] % (
                "nnan_zs_uniq_diff_uniq_mean", nnan_zs_uniq_diff_uniq_mean)

        None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    # assess profile 'quality'

    (nnan_zs_where, zs_idx_nnan_lf, zs_idx_nnan_rt,
        pts_fov, pts_roi, pts_value, pts_drop) = (
            calculate_profile_quality_metrics(profile_zs_mask_nans))
    #f 0 and prt:
    #   print fmt0(just1)[0:] % (
    #       "nnan_zs_where.shape", nnan_zs_where.shape)
    #   print fmt0(just1)[1:] % ("zs_idx_nnan_lf", zs_idx_nnan_lf)
    #   print fmt0(just1)[1:] % ("zs_idx_nnan_rt", zs_idx_nnan_rt)
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("pts_fov", pts_fov)
    #   print fmt0(just1)[1:] % ("pts_value", pts_value)
    #   print fmt0(just1)[1:] % ("pts_roi", pts_roi)
    #   print fmt0(just1)[1:] % ("pts_drop", pts_drop)
    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    # generate the filtered profiles (fltr) dataset

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # apply a median filter & a Gaussian filter ...
    # ... to the (nnan) profile to suppress noise for edge detection

    ### ### The median filter size chosen is ...
    #
    # the default value chosen
    #zs_median_filter_size = 1  # no effect (default)
    #zs_median_filter_size = zs_median_filter_size if 1 else 3  # "gentlest"
    #zs_median_filter_size = zs_median_filter_size if 1 else 5
    #zs_median_filter_size = zs_median_filter_size if 1 else 7
    #zs_median_filter_size = zs_median_filter_size if 1 else 9
    #zs_median_filter_size = zs_median_filter_size if 0 else 11
    #
    ### the value chosen from the parameters configuration file
    zs_median_filter_size = ngn.zs_median_filter_size
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("zs_median_filter_size",
    #       zs_median_filter_size)

    ### ### The gaussian filter size is chosen to be ...
    #
    # the default value chosen
    #                                                            pts > 0.0
    #zs_gaussian_filter_sigma = 0.0  # no effect (default)              # 1
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 1 else 0.3  # 3
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 1 else 0.5  # 5
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 1 else 0.8  # 7
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 1 else 1.0  # 9
    #zs_gaussian_filter_sigma = zs_gaussian_filter_sigma if 0 else 1.2
    #
    ### the value chosen from the parameters configuration file
    zs_gaussian_filter_sigma = ngn.zs_gaussian_filter_sigma
    #f 1 and prt:
    #   print fmt0(just1)[1:] % ("zs_gaussian_filter_sigma",
    #       zs_gaussian_filter_sigma)

    fltr_zs = ndi.gaussian_filter(ndi.median_filter(
        nnan_zs, zs_median_filter_size), zs_gaussian_filter_sigma)

    #one if 1 else sys.exit()

    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
    # perform first difference calculations

    fltr_xs_diff = (nnan_xs[1:] - nnan_xs[:-1])
    #f 1 and prt:
    #   print fmt0(just1)[0:] % ("fltr_xs_diff.shape", fltr_xs_diff.shape)
    #   #rint fmt1(just1)[1:] % ("fltr_xs_diff", fltr_xs_diff)
    #   #rint fmt0(just1)[1:] % ("fltr_xs_diff[:6]", fltr_xs_diff[:6])
    #   print fmt1(just1)[0:] % ("fltr_xs_diff", fltr_xs_diff)

    fltr_zs_diff = (fltr_zs[1:] - fltr_zs[:-1])
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("fltr_zs_diff.shape", fltr_zs_diff.shape)
    #   #rint fmt1(just1)[1:] % ("fltr_zs_diff", fltr_zs_diff)
    #   print fmt0(just1)[1:] % ("fltr_zs_diff[:6]", fltr_zs_diff[:6])

    fltr_dzdxs = fltr_zs_diff / fltr_xs_diff
    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("fltr_dzdxs.shape", fltr_dzdxs.shape)
    #   #rint fmt1(just1)[1:] % ("fltr_dzdxs", fltr_dzdxs)
    #   print fmt0(just1)[1:] % ("fltr_dzdxs[:6]", fltr_dzdxs[:6])

    #one if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    # identify gap features in the measurement profile

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # segment edges and flats in the (filtered) laser measurement profile

    prt = True if 1 and prt_ else False

    #f 0 and prt:
    #   print fmt0(just1)[0:] % ("ngn.dzdxs_threshold", ngn.dzdxs_threshold)

    segm_ps = segment_edges_and_flats(number_of_profiles, indy,
        fltr_dzdxs, ngn.dzdxs_threshold, nnan_xs_midp, nnan_xs)
    #f 0 and prt:
    #   print fmt1(just1)[0:] % ("sorted(segm_ps.keys())",
    #       prt_list(sorted(segm_ps.keys()), 0))

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # make gallery 01 profile plot

    if 1 and ngn.write_to_results_dir and ngn.make_gallery01_plots:

        make_gallery01_profile_plot(
            number_of_profiles, indy,
            pts_fov, pts_roi, pts_value, pts_drop,
            zs_median_filter_size, zs_gaussian_filter_sigma,
            nnan_xs, nnan_zs, fltr_zs,
            nnan_xs_midp, nnan_dzdxs, fltr_dzdxs,
            segm_ps,
        )
        None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # identify gap (& strip) features in each region of interest

    prt = True if 1 and prt_ else False

    ps_gaps = identify_profile_features_in_rois(indy,
        profile_xs_nnan_where, profile_xs, nnan_xs, fltr_zs,
        nnan_xs_midp, fltr_dzdxs, segm_ps)
    if 0 and prt:
        print
        for ps_gap in ps_gaps:
            print fmt0(just1)[1:] % ("dict(ps_gap)", dict(ps_gap))
        None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    # post results

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # post laser measurement profile (raw) edges to the edges image map

    #if 0 and prt:
    #    print fmt0(just1)[0:] % ("profile_zs_mask_nnans.shape",
    #        profile_zs_mask_nnans.shape)
    #    print fmt0(just1)[1:] % ("np.sum(profile_zs_mask_nnans)",
    #        np.sum(profile_zs_mask_nnans))
    #    print fmt1(just1)[1:] % ("list(profile_zs_mask_nnans_where)",
    #        list(profile_zs_mask_nnans_where))
    #
    ##fltr_zs_mask_edges = segm_ps.fltr_zs_mask_edges
    #if 0 and prt:
    #    print fmt0(just1)[0:] % ("segm_ps.fltr_zs_mask_edges.shape",
    #        segm_ps.fltr_zs_mask_edges.shape)
    #    print fmt0(just1)[1:] % ("np.sum(segm_ps.fltr_zs_mask_edges)",
    #        np.sum(segm_ps.fltr_zs_mask_edges))

    profile_zs_mask_nnans_where_edges = (
        profile_zs_mask_nnans_where[segm_ps.fltr_zs_mask_edges])
    if 0 and prt:
        print fmt0(just1)[0:] % ("profile_zs_mask_nnans_where_edges.shape",
            profile_zs_mask_nnans_where_edges.shape)
        print fmt1(just1)[1:] % ("list(profile_zs_mask_nnans_where_edges)",
            list(profile_zs_mask_nnans_where_edges))

    ngn.np_edges_image[indy, profile_zs_mask_nnans_where_edges] = True
    if 0 and prt:
        print fmt0(just1)[0:] % ("indy", indy)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # post (tabular) gap results

    results_gap_ps = post_gap_results_legacy_gap_analysis(
        zs_idx_nnan_lf, zs_idx_nnan_rt, pts_fov, pts_roi, pts_value, pts_drop,
        ps_gaps, results_gap_ps)
    #f 0 and prt:
    #   print fmt1(just1)[0:] % ("(updt) results_gap_ps[:20]",
    #       results_gap_ps[:20])
    #f 1 and prt:
    #   print fmt1(just1)[0:] % ("results_gap_ps", results_gap_ps)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return results_gap_ps


def analyze_tow_gaps_legacy_gap_analysis(np_src_xs, np_hasnan_zs,
np_hasnan_zs_mask_nans, pd_results_gap):
    """
    Returns a Pandas DataFrame containing gap results for one laser profile..
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt__ = False if 1 else True  # def print switch
    prt_ = prt
    mult_str = '--- '
    def_str = 'analyze_tow_gaps_legacy_gap_analysis'
    if prt_ or prt__:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    #== === === === === === === === === === === === === === === === === === ===
    # (below) analyze gaps for each laser measurement profile

    if 1 and prt:
        print fmt0(just1)[0:] % ("gap(/butt/lap) algorithm", "begin")

    None if 1 else sys.exit()

    prt = True if 1 and prt_ else False

    number_of_profiles = np_hasnan_zs.shape[0]
    if 1 and prt:
        print fmt0(just1)[0:] % ("number_of_profiles", number_of_profiles)

    indy_last = number_of_profiles - 1  # default
    #ndy_last = indy_last if 1 else 0
    #ndy_last = indy_last if 1 else 1
    #ndy_last = indy_last if 1 else 9
    #ndy_last = indy_last if 1 else 19
    #ndy_last = indy_last if 1 else 60
    #ndy_last = indy_last if 1 else 49
    #ndy_last = indy_last if 1 else 120
    #ndy_last = indy_last if 1 else 167
    #ndy_last = indy_last if 1 else 179  # 20170519_06_PlyDrops
    #ndy_last = indy_last if 1 else 189
    #ndy_last = indy_last if 1 else 192
    #ndy_last = indy_last if 1 else 199
    #ndy_last = indy_last if 0 else 209
    #ndy_last = indy_last if 1 else 212
    #ndy_last = indy_last if 0 else 219
    #ndy_last = indy_last if 1 else 272
    #ndy_last = indy_last if 0 else 299
    #ndy_last = indy_last if 1 else 399  # 20170519_06_PlyDrops
    #ndy_last = indy_last if 0 else 399  # 30ThouLap
    #ndy_last = indy_last if 0 else 499
    #ndy_last = indy_last if 0 else 949
    #ndy_last = indy_last if 0 else 999
    #ndy_last = indy_last if 0 else 1599
    #
    indy_first = 0  # default
    #ndy_first = indy_first if 0 else 1
    #ndy_first = indy_first if 0 else 22
    #ndy_first = indy_first if 0 else 60
    #ndy_first = indy_first if 0 else 80
    #ndy_first = indy_first if 1 else 168
    #ndy_first = indy_first if 1 else 190
    #ndy_first = indy_first if 1 else 167
    #ndy_first = indy_first if 1 else 189
    #ndy_first = indy_first if 0 else 199
    #ndy_first = indy_first if 1 else 272
    #ndy_first = indy_first if 0 else 299
    #ndy_first = indy_first if 1 else 409
    #ndy_first = indy_first if 1 else 509
    #ndy_first = indy_first if 0 else 1400
    #
    indy_first = indy_first if 1 else 9999999999  # make same as indy_last
    if 1 or prt:
        print fmt0(just1)[0:] % ("indy_first", indy_first)
    #
### #f 1 and 'CNRC20170522_06_PlyDrops' in ngn.job_dir:
### #   if indy_first < 180:
### #       indy_first = 180  # let this be the first measurement index
### #f 1 and 'CNRC20170522_30ThouLap' in ngn.job_dir:
### #   if indy_first < 39:
### #       indy_first = 39  # let this be the first measurement index
    #
    indy_first = indy_first if indy_first <= indy_last else indy_last
    if 1 or prt:
        print fmt0(just1)[0:] % (
            "[0, indy_first, indy_last, number_of_profiles - 1]",
            [0, indy_first, indy_last, number_of_profiles - 1])

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    print "\n... 'dumb' gap profiles analysis begin %s" % ('... ' * ngn.mult)

    if ngn.count_gap_rois_analyzed:
        initialize_gap_rois_analyzed_counts()
    if 0 and prt:
        print fmt0(just2)[0:] % ("ngn.count_gap_rois_analyzed",
            ngn.count_gap_rois_analyzed)

    ps_results_gap = pd.Series(np.full((len(ngn.results_fields_gap)), np.nan),
        index=ngn.results_fields_gap)
    if 1 and prt:
        print fmt0(just1)[0:] % ("ps_results_gap.shape",
            ps_results_gap.shape)
        print fmt1(just1)[1:] % ("ps_results_gap", ps_results_gap)

    None if 1 else sys.exit()

    for indy in xrange(number_of_profiles):
        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
        if indy < indy_first:
            continue

        #rint "\n... ... ... ... ... ... ... ... %s" % ('... ' * ngn.mult)
        if (indy % 100 == 0):
            print fmt0(just1)[1:] % (
                "[indy, number_of_profiles]", [indy, number_of_profiles])
        try:
            results_gap_ps = (
                analyze_laser_measurement_profile_legacy_gap_analysis(
                    number_of_profiles, indy, np_src_xs[indy, :],
                    np_hasnan_zs[indy, :], np_hasnan_zs_mask_nans[indy, :],
                    ps_results_gap.copy()))

            pd_results_gap.loc[indy, :] = results_gap_ps.values
            #f 1:
            #   #rint fmt1(just1)[0:] % ("results_gap_ps", results_gap_ps)
            #   #rint fmt1(just1)[0:] % (
            #   #   "pd_results_gap.loc[indy]", pd_results_gap.loc[indy])
            #   print fmt1(just1)[0:] % (
            #       "pd_results_gap.loc[[indy]]", pd_results_gap.loc[[indy]])
            #   None if 1 else sys.exit()

        except Exception, err:
            #print 'print_exc():'
            #traceback.print_exc(file=sys.stdout)
            print "\n... indy: %i ... err: %s\n" % (indy, err)

        finally:
            # this gets executed regardless ...
            None if 1 else sys.exit()

        if ngn.count_gap_rois_analyzed:
            ngn.count_analyzed_measts += 1

        #:: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: ::: :::
        if 1 and indy >= indy_last:
            break
        #break

    if ngn.count_gap_rois_analyzed:
        report_gap_rois_analyzed_counts()

    print "\n... 'dumb' gap profiles analysis end %s" % ('... ' * ngn.mult)

    # (above) analyze gaps for each laser measurement profile
    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_ or prt__:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()
    return pd_results_gap


# (above) defs ... for analyzing gaps between tows using "dump gap" analysis
#zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz


def make_laser_profile_edges_image_plot(ngn, np_edges_image, title=None):
    """
    Makes an image plot of all (unfiltered) laser measurement profile edges.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_laser_profile_edges_image_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 0 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    image = np_edges_image.copy().astype(np.int)
    # presentation "fix" ...
    image[0, 0] = 2
    # (below) placement of course "boundary" lines
    #image[0::250, :] = 2
    #image[1::250, :] = 2
    #image[2::250, :] = 2
    #image[3::250, :] = 2
    #image[4::250, :] = 2
    #
    #
    img_png = ngn.job_zs_csv.replace('z', '').replace('.txt', '').replace(
        '.csv', '__laser_profile_edges_image.png')
    if 0 and prt:
        print fmt0(just1)[0:] % ("img_png", img_png)
    None if 1 else sys.exit()

    png_abspath = os.path.join(ngn.job_absdir, img_png)
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
        print fmt1(just1)[1:] % ("img_png", img_png)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)

    title = (
        title if title is not None else
        'Image Map of Laser Profile Edges\n%s' % ngn.job_zs_csv
    )
    cmap = 'gray' if 0 else 'jet'
    colorbar = True if 0 else False
    xlabel = 'Measurement Index (Y)'
    ylabel = 'Profile Index (X)'
    png = png_abspath if 1 else None
    imshow(image.T, title=title, cmap=cmap, xticklabel=True,
        yticklabel=True, xlabel=xlabel, ylabel=ylabel,
        colorbar=colorbar, png=png)

    if ngn.write_to_results_dir:
        png_abspath_results = os.path.join(ngn.results_absdir, img_png)
        if 0 and prt:
            print fmt1(just1)[0:] % ("ngn.results_absdir", ngn.results_absdir)
            print fmt1(just1)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
            print fmt1(just1)[1:] % ("img_png", img_png)
        print fmt1(just1)[0:] % ("png_abspath_results", png_abspath_results)
        shutil.copy(png_abspath, png_abspath_results)

    #f ngn.add_pyapp_version:
    #   img_png_pyapp_ver = img_png.replace(
    #       '.png', '_py%.3i.png' % ngn.version_py)
    #   if 0 and prt:
    #       print fmt0(just1)[0:] % ("img_png", img_png)
    #       print fmt0(just1)[1:] % ("img_png_pyapp_ver", img_png_pyapp_ver)
    #   png_abspath_pyapp_ver = os.path.join(ngn.job_absdir, img_png_pyapp_ver)
    #   shutil.copy(png_abspath, png_abspath_pyapp_ver)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


def make_gap_results_plot(ngn, pd_results_gap):
    """
    Makes a plot of gap results, left and right gap edge position profiles
    and gap width profiles.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 1 else True
    prt_ = prt
    mult_str = '--- '
    def_str = 'make_gap_results_plot'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    if 1 and prt:
        print fmt0(just1)[0:] % (
            "pd_results_gap.shape", pd_results_gap.shape)
        #rint fmt1(just1)[1:] % ("pd_results_gap", pd_results_gap)
        print fmt1(just1)[1:] % (
            "pd_results_gap.head()", pd_results_gap.head())
        print fmt1(just1)[1:] % (
            "pd_results_gap.tail()", pd_results_gap.tail())
        print fmt1(just1)[1:] % (
            "pd_results_gap.columns", pd_results_gap.columns)

    winctr_xrefs = ngn.tow_center_xrefs
    if 1 and prt:
        print fmt0(just1)[0:] % ("winctr_xrefs", winctr_xrefs)

    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===
    # plot ... gap edges & width profiles

    title_fontsize = 12
    text_fontsize = 9
    axhline_xmin = 0.01
    axhline_xmax = 0.99
    #
    gap_colors = [
        # http://matplotlib.org/examples/color/named_colors.html
        # index:
        #  0   1    2    3    4    5    6            7       8
        #  9  10   11   12   13   14   15           16      17
        # 18  19   20   21   22   23   24           25      26
        # 27  28   29   30   31   32   33           34      35
        # count:
        #  1   2    3    4    5    6    7            8       9
        # 10  11   12   13   14   15   16           17      18
        # 19  20   21   22   23   24   25           26      27
        # 28  29   30   31   32   33   34           35      36
        'b', 'g', 'r', 'c', 'm', 'y', 'orangered', 'teal', 'gray',
        'b', 'g', 'r', 'c', 'm', 'y', 'orangered', 'teal', 'gray',
        'b', 'g', 'r', 'c', 'm', 'y', 'orangered', 'teal', 'gray',
        'b', 'g', 'r', 'c', 'm', 'y', 'orangered', 'teal', 'gray',
    ]
    # ... for tows in a course
    #
    ### meast_nums = np.arange(len(pd_results_gap)) + 1.
    ### meast_nums_mid = 0.5 * (meast_nums[-1] + meast_nums[0])
    ### meast_nums_rng_half = 0.5 * (meast_nums[-1] - meast_nums[0])
    ### if 1 and prt:
    ###     print fmt0(just1)[0:] % (
    ###         "(old) meast_nums.shape", meast_nums.shape)
    ###     print fmt1(just1)[1:] % (
    ###         "(old) meast_nums", meast_nums)
    meast_nums = pd_results_gap.index.values + 1.
    meast_nums_mid = 0.5 * (meast_nums[-1] + 0.)
    meast_nums_rng_half = 0.5 * (meast_nums[-1] - 0.)
    if 0 and prt:
        print fmt0(just1)[0:] % (
            "(new) meast_nums.shape", meast_nums.shape)
        print fmt1(just1)[1:] % (
            "(new) meast_nums", meast_nums)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    fig = plt.gcf()
    fig.set_size_inches(16, 12, forward=True)  # default is (8, 6)
    fig.suptitle("%s:\n%s" % ("Gap Measurement Results", os.path.join(
        ngn.job_dir, ngn.job_zs_csv)))
    gs = mpl.gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)
    #
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # (below) ax1
    ax1.set_title("X-coordinate Positions of Gap/Lap Edges")

    #x1.set_xlabel('Measurement Number')
    #x1.set_xlim(job_ps.ax_set_xlim)
    ax1.set_ylabel('Gap/Lap Edges X-coordinate (mm)')

    ax1_trans1 = mpl.transforms.blended_transform_factory(
        ax1.transAxes, ax1.transData)

    for i, winctr_yref in enumerate(winctr_xrefs):
        #
        ax1_axhline_ls = (
            'dashed' if i == 0 or (i + 1) == len(winctr_xrefs) else 'dotted')
        ax1.axhline(y=winctr_yref, xmin=axhline_xmin, xmax=axhline_xmax,
            c='y', ls=ax1_axhline_ls, lw=2.)
        ax1.text(0.01, winctr_yref, 'Tow %.2i' % i,
            transform=ax1_trans1,
            fontsize=text_fontsize, fontweight='bold',
            ha='left', va='center')

    for tow_num1 in xrange(1, len(winctr_xrefs)):
        tow_num0 = tow_num1 - 1
        gap_id = "%.2i%.2i" % (tow_num0, tow_num1)
        #f 1 and prt:
        #   print fmt0(just2)[0:] % (
        #       "[tow_num0, tow_num1, gap_id]",
        #       [tow_num0, tow_num1, gap_id])
        #
        ys = pd_results_gap['Gap%sEdgeLfX' % gap_id]
        ax1.plot(meast_nums, ys, ',-', color=gap_colors[tow_num0],
            label='Gap %s Edges' % gap_id)
        #
        ys = pd_results_gap['Gap%sEdgeRtX' % gap_id]
        ax1.plot(meast_nums, ys, ',-', color=gap_colors[tow_num0],
            label=None)

    #x1_get_ylim_min, ax1_get_ylim_max = ax1.get_ylim()
    ax1_get_ylim_max = np.max(np.abs(winctr_xrefs))
    ax1_get_ylim_min = -ax1_get_ylim_max
    ax1.set_ylim((ax1_get_ylim_max * 1.1, ax1_get_ylim_min * 1.1))

    ax1.legend(
        loc=8,
        ncol=len(winctr_xrefs) - 1,
        numpoints=1,
        markerscale=1.,
        prop={'size': 9.2, 'weight': 'bold'}
    ) if 0 else None

    # (above) ax1
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    # (below) ax2
    ax2.set_title("Gap/Lap Width Between Gap/Lap Edges")

    ax2.set_xlabel('Measurement Number')
    #x2.set_xlim(job_ps.ax_set_xlim)
    ax2.set_ylabel('Gap/Lap Width (mm)')

    ax2_set_xlim_max = meast_nums_mid + meast_nums_rng_half * 1.15
    ax2_set_xlim_min = meast_nums_mid - meast_nums_rng_half * 1.15
    ax2.set_xlim((ax2_set_xlim_min, ax2_set_xlim_max))

    ax2.axhline(y=0., xmin=0.02, xmax=0.98, c='gray', ls='dotted', lw=2.)

    max_abs_ys = []
    for tow_num1 in xrange(1, len(winctr_xrefs)):
        tow_num0 = tow_num1 - 1
        gap_id = "%.2i%.2i" % (tow_num0, tow_num1)
        #f 1 and prt:
        #   print fmt0(just2)[0:] % (
        #       "[tow_num0, tow_num1, gap_id]",
        #       [tow_num0, tow_num1, gap_id])
        #
        ys = pd_results_gap['Gap%sWidth' % gap_id].astype(np.float).values
        ax2.plot(meast_nums, ys, ',-', color=gap_colors[tow_num0],
            label='Gap %s Width' % gap_id)
        if np.sum(~ np.isnan(ys)) > 0:
            ys_abs_max = np.nanmax(np.abs(ys))
            max_abs_ys.append(np.nanmax(np.abs(ys)))
            #rint "... np.sum(~ np.isnan(ys)) > 0", np.sum(~ np.isnan(ys)) > 0
            #rint "\n... ys.dtype", ys.dtype
            #rint "... ys.shape", ys.shape
            #rint "... ys", ys
            #rint "... ys_abs_max", ys_abs_max
            #rint "\n... max_abs_ys\n", max_abs_ys
    if len(max_abs_ys) > 0:
        ax2_get_ylim_max_abs = np.nanmax(max_abs_ys)
        ax2.set_ylim((
            -ax2_get_ylim_max_abs * 1.05,
            ax2_get_ylim_max_abs * 1.05
        ))
    ax2_trans1 = mpl.transforms.blended_transform_factory(
        ax2.transAxes, ax2.transAxes)
    ax2.text(0.01, 0.80, 'Gap Region\n(Width > 0.)',
        transform=ax2_trans1, rotation=90,
        fontsize=text_fontsize, fontweight='bold',
        ha='left', va='center')
    ax2.text(0.01, 0.20, 'Lap Region\n(Width < 0.)',
        transform=ax2_trans1, rotation=90,
        fontsize=text_fontsize, fontweight='bold',
        ha='left', va='center')

    if 1 and 'CNRC20170522_30ThouLap' in ngn.job_dir:
        _30thou_gaplap = 0.030 * 25.4
        ax2_trans2 = mpl.transforms.blended_transform_factory(
            ax2.transAxes, ax2.transData)
        #
        ax2.axhline(y=_30thou_gaplap, xmin=0.02, xmax=0.98,
            c='gray', ls='dashed', lw=2.)
        ax2.text(0.99, _30thou_gaplap,
            'Gap Target (mm): %.3f' % _30thou_gaplap,
            transform=ax2_trans2, rotation=90,
            fontsize=text_fontsize, fontweight='bold',
            ha='right', va='bottom')
        #
        ax2.axhline(y=-_30thou_gaplap, xmin=0.02, xmax=0.98,
            c='gray', ls='dashed', lw=2.)
        ax2.text(0.99, -_30thou_gaplap,
            'Lap Target (mm): %.3f' % -_30thou_gaplap,
            transform=ax2_trans2, rotation=90,
            fontsize=text_fontsize, fontweight='bold',
            ha='right', va='top')

    ax2.legend(
        loc=8,
        ncol=len(winctr_xrefs) - 1,
        numpoints=1,
        markerscale=1.,
        prop={'size': 9.2, 'weight': 'bold'}
    ) if 0 else None

    # (above) ax2
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    results_gap_png = ngn.job_zs_csv.replace('z', '').replace(
        '.txt', '').replace('.csv', '__results_gap.png')

    png_abspath = os.path.join(ngn.job_absdir, results_gap_png)
    if 0 and prt:
        print fmt1(just1)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        print fmt1(just1)[1:] % ("png", png)
    print fmt1(just1)[0:] % ("png_abspath", png_abspath)
    #
    png = png_abspath if 1 else None
    if png is not None:
        plt.savefig(png)
    else:
        plt.show()
    plt.close()
    None if 1 else sys.exit()

    if 1 and ngn.write_to_results_dir:
        png_results_dir_abspath = os.path.join(
            ngn.results_absdir, results_gap_png)
        if 1 or prt:
            print fmt1(just1)[0:] % ("png_results_dir_abspath",
                png_results_dir_abspath)

        shutil.copy(png_abspath, png_results_dir_abspath)

    #f 1 and ngn.add_pyapp_version:
    #   png_pyapp_ver = results_gap_png.replace(
    #       '.png', '_py%.3i.png' % ngn.version_py)
    #   png_pyapp_ver_abspath = os.path.join(ngn.job_absdir, png_pyapp_ver)
    #   if 1 or prt:
    #       print fmt1(just1)[0:] % ("png_pyapp_ver_abspath",
    #           png_pyapp_ver_abspath)
    #
    #   shutil.copy(png_abspath, png_pyapp_ver_abspath)

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)
    None if 1 else sys.exit()


# (above) defs ... for analyzing gaps between tows
#ZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ ZZZ


def process_job_preview():
    """
    Executes this module as a Python (preview) script.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    mult_str = '=== '
    def_str = 'process_job_preview'

    epoch_secs = time.time()
    if prt_:
        print fmt0(just1)[0:] % ('(beg) def %s' % def_str, ngn.in_)
        print fmt0(just1)[1:] % ('execution started',
            make_timestamp(ngn, epoch_secs))
        print "%s" % (mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # get the "laser profiles" dataset

    np_src_xs, np_src_zs = get_dataset(
        ngn.job_xs_csv_abspath, ngn.job_zs_csv_abspath)
    if 1 and prt:
        print fmt0(just2)[0:] % ("(init) np_src_xs.shape", np_src_xs.shape)
        print fmt0(just2)[1:] % ("(init) np_src_zs.shape", np_src_zs.shape)

    if 1 and prt:
        print "\n%s" % ('*** ' * ngn.stars_mult)
    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    if 1:  #
        print fmt0(just2)[0:] % ("*** quick stats ***", "*** quick stats ***")

        np_src_xs_col__0_nanmean = np.nanmean(np_src_xs[:, 0])
        np_src_xs_col_m1_nanmean = np.nanmean(np_src_xs[:, -1])
        print fmt0(just2)[0:] % (
            "[np_src_xs_col__0_nanmean, np_src_xs_col_m1_nanmean]",
            np.array([np_src_xs_col__0_nanmean, np_src_xs_col_m1_nanmean]))

        np_src_zs_nanmean = np.nanmean(np_src_zs)
        np_src_zs_nanstd = np.nanstd(np_src_zs)
        np_src_zs_nanmean_m3std = np_src_zs_nanmean - np_src_zs_nanstd * 3.
        np_src_zs_nanmean_p3std = np_src_zs_nanmean + np_src_zs_nanstd * 3.
        np_src_zs_nanmean_rng3std = (
            np_src_zs_nanmean_p3std - np_src_zs_nanmean_m3std)
        print fmt0(just2)[0:] % ("[np_src_zs_nanmean, np_src_zs_nanstd]",
            np.array([np_src_zs_nanmean, np_src_zs_nanstd]))
        print fmt0(just2)[1:] % (
            "[np_src_zs_nanmean_m3std, np_src_zs_nanmean_p3std]",
            np.array([np_src_zs_nanmean_m3std, np_src_zs_nanmean_p3std]))
        print fmt0(just2)[1:] % ("[np_src_zs_nanmean_rng3std]",
            np.array([np_src_zs_nanmean_rng3std]))

        print fmt0(just2)[0:] % ("np.isnan(np_src_zs).any()",
            np.isnan(np_src_zs).any())

        if 1 and prt:
            print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # "preview" the dataset "xs"

    preview_laser_profiles_dataset_xs(
        np_src_xs, ngn.job_xs_csv
    ) if 0 else None

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # "preview" the dataset "zs" & image map

    if 1 and prt:
        print fmt0(just2)[0:] % ("ngn.job_dir", ngn.job_dir)
    None if 1 else sys.exit()

    #zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz
    # (below) parameter inputs

    nan_threshold_floor, nan_threshold_ceiling = None, None  # default
    #
    # (below)  CNRC20170620_PlyDrops_XZZZZ_format
    #an_threshold_floor = nan_threshold_floor if 1 else -76.594
    #an_threshold_ceiling = nan_threshold_ceiling if 1 else -74.886
    #
    # (below) CNRC20170621_PlyDrops_V2_XZZZZ_format
    ##nan_threshold_floor = nan_threshold_floor if 1 else -76.594
    ##nan_threshold_floor = nan_threshold_floor if 1 else -76.836
    #an_threshold_floor = nan_threshold_floor if 1 else -76.303
    ##nan_threshold_ceiling = nan_threshold_ceiling if 1 else -74.886
    ##nan_threshold_ceiling = nan_threshold_ceiling if 1 else -74.230
    #an_threshold_ceiling = nan_threshold_ceiling if 1 else -75.014
    #
    if 'CNRC20170623_PlyDrops_V3_XZZZZ_format' == ngn.job_dir:
        nan_threshold_floor = -76.303
        nan_threshold_ceiling = -75.014
    if 'CNRC20170624_17_05_08_PlyDrops_Analytics2' == ngn.job_dir:
        nan_threshold_floor = -76.260
        nan_threshold_ceiling = -74.925
    if 'CNRC20170624_17_05_09_PlyDrops_Analytics2' == ngn.job_dir:
        nan_threshold_floor = -75.7
        nan_threshold_floor = -75.66
        nan_threshold_ceiling = -74.8
        nan_threshold_ceiling = -74.79
    if 'CNRC20170711_PlyDrops1_Dry_600ipm' == ngn.job_dir:
        nan_threshold_floor = -76.550
        nan_threshold_ceiling = -75.580
    if 'CNRC20170711_PlyDrops2_Fresh_600ipm' == ngn.job_dir:
        nan_threshold_floor = -75.900
        nan_threshold_ceiling = -75.050
    if 'CNRC20170724_PlyDrops2_PH1_PS12' == ngn.job_dir:
        nan_threshold_floor = -77.850
        nan_threshold_ceiling = -76.700
    if 'CNRC20170724_PlyDrops2_PH1_PS13' == ngn.job_dir:
        nan_threshold_floor = -77.850
        nan_threshold_ceiling = -76.600
    if 'CNRC20170724_PlyDrops2_PH1_PS16' == ngn.job_dir:
        nan_threshold_floor = -77.750
        nan_threshold_ceiling = -76.550
    if 'PH5_PS26_20170725190342_Keyence' == ngn.job_dir:
        nan_threshold_floor = 10
        nan_threshold_ceiling = 12
    if 'PH5_PS29_20170725194318_Keyence' == ngn.job_dir:
        nan_threshold_floor = 10
        nan_threshold_ceiling = 12

    count_threshold_floor = 2
    #ount_threshold_floor = None  # default
    #ount_threshold_floor = count_threshold_floor if 0 else 2.

    # (above) parameter inputs
    #zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz

    if 1 and prt:
        print fmt0(just2)[0:] % ("nan_threshold_floor",
            nan_threshold_floor)
        print fmt0(just2)[1:] % ("nan_threshold_ceiling",
            nan_threshold_ceiling)
        print fmt0(just2)[1:] % ("count_threshold_floor",
            count_threshold_floor)
    None if 1 else sys.exit()

    preview_laser_profiles_dataset_zs(
        np_src_zs, ngn.job_zs_csv,
        nan_threshold_ceiling, nan_threshold_floor, count_threshold_floor
    ) if 1 else None

    preview_laser_profile_zs_image(
        np_src_zs, ngn.job_zs_csv, nan_threshold_ceiling, nan_threshold_floor
    ) if 1 else None

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # "preview" the dataset profiles (for establishing alignment parameters)

    #zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz
    # (below) parameter inputs

    course_edge_roi_lf, course_edge_roi_rt = None, None

    if ('CNRC20170623_PlyDrops_V3_XZZZZ_format' == ngn.job_dir or
    'CNRC20170624_17_05_08_PlyDrops_Analytics2' == ngn.job_dir or
    'CNRC20170624_17_05_09_PlyDrops_Analytics2' == ngn.job_dir):
        indy_beg, indy_end = 200, 300
        #
        course_edge_roi_lf = (-25.500, -23.500)
        course_edge_roi_rt = (25.000, 27.000)

    if ('CNRC20170711_PlyDrops1_Dry_600ipm' == ngn.job_dir or
    'CNRC20170711_PlyDrops2_Fresh_600ipm' == ngn.job_dir):
        indy_beg, indy_end = 200, 250
        #
        course_edge_roi_lf = (-25.500, -23.500)
        course_edge_roi_rt = (25.000, 27.000)

    if 'CNRC20170724_PlyDrops2_PH1_PS12' == ngn.job_dir:
        indy_beg, indy_end = 350, 450
        #
        course_edge_roi_lf = (-25.900, -23.900)
        course_edge_roi_rt = (26.000, 28.000)

    if 'CNRC20170724_PlyDrops2_PH1_PS13' == ngn.job_dir:
        indy_beg, indy_end = 310, 410
        #
        course_edge_roi_lf = (-25.900, -23.900)
        course_edge_roi_rt = (26.000, 28.000)

    if 'CNRC20170724_PlyDrops2_PH1_PS16' == ngn.job_dir:
        indy_beg, indy_end = 260, 360
        #
        course_edge_roi_lf = (-25.900, -23.900)
        course_edge_roi_rt = (26.000, 28.000)

    if 'PH5_PS26_20170725190342_Keyence' == ngn.job_dir:
        indy_beg, indy_end = 260, 360
        #
        course_edge_roi_lf = (-25.900, -23.900)
        course_edge_roi_rt = (26.000, 28.000)
    
    if 'PH5_PS29_20170725194318_Keyence' == ngn.job_dir:
        indy_beg, indy_end = 260, 360
        #
        course_edge_roi_lf = (-25.900, -23.900)
        course_edge_roi_rt = (26.000, 28.000)

    median_filter_size = 3
    gaussian_filter_sigma = 0.8

    dzdxs_threshold = 0.32

    course_tows = 16
    tow_width_nominal_mm = 3.175

    # (above) parameter inputs
    #zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz

    if 1 and prt:
        print fmt0(just2)[0:] % ("[indy_beg, indy_end]", [indy_beg, indy_end])
        print fmt0(just2)[1:] % ("course_edge_roi_lf", course_edge_roi_lf)
        print fmt0(just2)[1:] % ("course_edge_roi_rt", course_edge_roi_rt)
        print fmt0(just2)[1:] % ("median_filter_size", median_filter_size)
        print fmt0(just2)[1:] % ("gaussian_filter_sigma",
            gaussian_filter_sigma)
        print fmt0(just2)[1:] % ("dzdxs_threshold", dzdxs_threshold)
        print fmt0(just2)[1:] % ("course_tows", course_tows)
        print fmt0(just2)[1:] % ("tow_width_nominal_mm", tow_width_nominal_mm)
    None if 1 else sys.exit()

    fix_plot_z_max = None if 1 else -74.8
    fix_plot_z_min = None if 1 else -75.6
    fix_plot_dz_max = None if 1 else 3.0
    fix_plot_dz_min = None if 1 else -3.0
    #
    indys = np.arange(indy_beg, indy_end)
    if 1 and prt:
        print fmt0(just2)[0:] % ("indys.shape", indys.shape)
        #rint fmt1(just2)[1:] % ("indys", indys)

    course_edge_roi_lf_xs = []
    course_edge_roi_rt_xs = []
    for i, indy in enumerate(indys):
        if 0 and prt:
            if i == 0:
                print
        if 0 and prt:
            print fmt0(just1)[0:] % ("indy", indy)
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        ps = preview_laser_profile(
            indy, np_src_xs, np_src_zs,
            nan_threshold_ceiling, nan_threshold_floor,
            median_filter_size, gaussian_filter_sigma, dzdxs_threshold,
            course_edge_roi_lf, course_edge_roi_rt,
            make_plot=False if 0 else True,
            fix_plot_z_max=fix_plot_z_max,
            fix_plot_z_min=fix_plot_z_min,
            fix_plot_dz_max=fix_plot_dz_max,
            fix_plot_dz_min=fix_plot_dz_min,
        )
        if 0 and prt:
            print fmt1(just1)[0:] % ("(indy: %5i) ps" % indy, ps)
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        None if np.isnan(ps.course_edge_roi_lf_x) else (
            course_edge_roi_lf_xs.append(ps.course_edge_roi_lf_x))
        None if np.isnan(ps.course_edge_roi_rt_x) else (
            course_edge_roi_rt_xs.append(ps.course_edge_roi_rt_x))
        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
        #break
    #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...
    if 0 and prt:
        print fmt1(just1)[0:] % ("(finl) course_edge_roi_lf_xs",
            course_edge_roi_lf_xs)
        print fmt1(just1)[1:] % ("(finl) course_edge_roi_rt_xs",
            course_edge_roi_rt_xs)

    None if 1 else sys.exit()

    np_course_edge_roi_lf_xs = np.array(course_edge_roi_lf_xs)
    course_edge_roi_lf_x_mean = np.mean(np_course_edge_roi_lf_xs)
    course_edge_roi_lf_x_3std = np.std(np_course_edge_roi_lf_xs) * 3.
    if 1 and prt:
        print fmt0(just1)[0:] % ("course_edge_roi_lf_x_mean",
            course_edge_roi_lf_x_mean)
        print fmt0(just1)[1:] % ("course_edge_roi_lf_x_3std",
            course_edge_roi_lf_x_3std)
        #rint fmt1(just1)[1:] % ("np_course_edge_roi_lf_xs",
        #   np_course_edge_roi_lf_xs)

    np_course_edge_roi_rt_xs = np.array(course_edge_roi_rt_xs)
    course_edge_roi_rt_x_mean = np.mean(np_course_edge_roi_rt_xs)
    course_edge_roi_rt_x_3std = np.std(np_course_edge_roi_rt_xs) * 3.
    if 1 and prt:
        print fmt0(just1)[0:] % ("course_edge_roi_rt_x_mean",
            course_edge_roi_rt_x_mean)
        print fmt0(just1)[1:] % ("course_edge_roi_rt_x_3std",
            course_edge_roi_rt_x_3std)
        #rint fmt1(just1)[1:] % ("np_course_edge_roi_rt_xs",
        #   np_course_edge_roi_rt_xs)

    course_width_between_edges = (
        course_edge_roi_rt_x_mean - course_edge_roi_lf_x_mean)
    tow_lane_width = course_width_between_edges / course_tows
    tow_gap_width = tow_lane_width - tow_width_nominal_mm
    course_width_x_mid = (
        (course_edge_roi_rt_x_mean + course_edge_roi_lf_x_mean) / 2.)
    if 1 and prt:
        print fmt0(just1)[0:] % ("course_width_between_edges",
            course_width_between_edges)
        print fmt0(just1)[1:] % ("course_tows", course_tows)
        print fmt0(just1)[1:] % ("tow_lane_width", tow_lane_width)
    if 1 and prt:
        print fmt0(just1)[0:] % (
            "*** *** *** tow_width_nominal_mm *** *** ***",
            tow_width_nominal_mm)
        print fmt0(just1)[1:] % (
            "*** *** *** tow_gap_width *** *** ***",
            tow_gap_width)
        print fmt0(just1)[1:] % (
            "*** *** *** course_width_x_mid *** *** ***",
            course_width_x_mid)

    # calculational check
    course_tow_edges_arange = np.arange(course_tows + 1)
    course_tow_edges = (course_tow_edges_arange * tow_lane_width -
        course_width_between_edges / 2. + course_width_x_mid)
    if 1 and prt:
        print fmt0(just1)[0:] % ("*** calculational check ***",
            "*** calculational check ***")
        print fmt0(just1)[1:] % ("course_tow_edges_arange",
            course_tow_edges_arange)
        print fmt0(just1)[1:] % ("course_tow_edges",
            course_tow_edges)

    None if 0 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # "preview" the dataset "dzdys" & image map

    #zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz
    # (below) parameter inputs

    # selected rows (indy) & columns (indx):
    indy_window_beg, indy_window_end = 0, np_src_zs.shape[0]  # default (all)
    indx_window_beg, indx_window_end = 0, np_src_zs.shape[1]  # default (all)

    if ('CNRC20170623_PlyDrops_V3_XZZZZ_format' == ngn.job_dir or
    'CNRC20170624_17_05_08_PlyDrops_Analytics2' == ngn.job_dir or
    'CNRC20170624_17_05_09_PlyDrops_Analytics2' == ngn.job_dir):
        indx_window_beg, indx_window_end = 400, 420
        #
        dzdys_threshold = 0.0783 * 0.5  # 0.03915

    if ('CNRC20170711_PlyDrops1_Dry_600ipm' == ngn.job_dir or
    'CNRC20170711_PlyDrops2_Fresh_600ipm' == ngn.job_dir):
        indx_window_beg, indx_window_end = 400, 420
        #
        dzdys_threshold = 0.1840 * 0.45  # 0.0828

    if ('CNRC20170724_PlyDrops2_PH1_PS12' == ngn.job_dir or
    'CNRC20170724_PlyDrops2_PH1_PS13' == ngn.job_dir):
        indx_window_beg, indx_window_end = 400, 420
        #
        dzdys_threshold = 0.0783 * 0.5  # 0.03915

    if 'CNRC20170724_PlyDrops2_PH1_PS16' == ngn.job_dir:
        indx_window_beg, indx_window_end = 400, 420
        #
        dzdys_threshold = 0.0783 * 0.5  # 0.03915

    # (above) parameter inputs
    #zz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz zzz

    preview_laser_profile_image_masked(
        np_src_zs, ngn.job_zs_csv,
        indy_window_beg, indy_window_end,
        indx_window_beg, indx_window_end,
        nan_threshold_ceiling, nan_threshold_floor,
    ) if 1 else None

    if 1:
        np_indys = np.arange(np_src_zs.shape[0])
        np_indxs = np.arange(np_src_zs.shape[1])

        preview_laser_meast_profiles(
            np_indys, np_indxs, np_src_zs,
            indy_window_beg, indy_window_end,
            indx_window_beg, indx_window_end,
            nan_threshold_ceiling, nan_threshold_floor,
            median_filter_size, gaussian_filter_sigma,
            dzdys_threshold,
            make_plot=False if 0 else True
        )

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s" % (mult_str * ngn.mult)
        print fmt0(just1)[1:] % ('execution completed',
            make_elapsed_time(ngn, epoch_secs))
        print fmt0(just1)[1:] % ('(end) def %s' % def_str, ngn.in_)


def process_job():
    """
    Executes this module as a Python script.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    mult_str = '=== '
    def_str = 'process_job'

    epoch_secs = time.time()
    if prt_:
        print fmt0(just1)[0:] % ('(beg) def %s' % def_str, ngn.in_)
        print fmt0(just1)[1:] % ('execution started',
            make_timestamp(ngn, epoch_secs))
        print "%s" % (mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---

    prt = True if 1 and prt_ else False

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print "\n%s" % ('*** ' * ngn.stars_mult)
    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # get the "laser profiles" dataset

    np_src_xs, np_src_zs = get_dataset(
        ngn.job_xs_csv_abspath, ngn.job_zs_csv_abspath)
    if 1 and prt:
        print fmt0(just2)[0:] % ("np_src_xs.shape", np_src_xs.shape)
        print fmt0(just2)[1:] % ("np_src_zs.shape", np_src_zs.shape)
    if 1 and prt:
        cdx_last = 20
        print fmt1(just2)[0:] % ("np_src_xs[:, :%i]" % cdx_last,
            np_src_xs[:, :cdx_last])

    if 1 and prt:
        print "\n%s" % ('*** ' * ngn.stars_mult)
    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # "wrangle" the dataset

##  if ngn.autothreshold_nan_values:
##      ngn.nan_value_floor_threshold, ngn.nan_value_ceiling_threshold = (
##          autothreshold_nan_values(np_src_xs, np_src_zs))
##      print fmt0(just1)[0:] % ("[ngn.nan_value_floor_threshold, " +
##          "ngn.nan_value_ceiling_threshold]",
##          np.array([ngn.nan_value_floor_threshold,
##              ngn.nan_value_ceiling_threshold]))
##
##  None if 1 else sys.exit()

    np_hasnan_zs_mask_nans, np_hasnan_zs = fix_nan_values(
        np_src_xs, np_src_zs,
        ngn.nan_value_floor_threshold, ngn.nan_value_ceiling_threshold,
        ngn.sensor)
    if 1 and prt:
        print fmt0(just2)[0:] % ("np_hasnan_zs_mask_nans.shape",
            np_hasnan_zs_mask_nans.shape)
        print fmt0(just2)[1:] % ("np_hasnan_zs.shape", np_hasnan_zs.shape)
    if 1 and prt:
        cdx_last = 20
        print fmt1(just2)[0:] % ("np_hasnan_zs[:, :%i]" % cdx_last,
            np_hasnan_zs[:, :cdx_last])

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # get the regions of interest (roi), tow edge xref indices

    ## this code assumes that all rows have identical values
    #np_src_xs0 = np_src_xs[0, :]
    ## this code assumes that rows do not have identical values
    np_src_xs0 = np.mean(np_src_xs, axis=0)
    if 1 or prt:
        print fmt0(just1)[0:] % ('np_src_xs0.shape', np_src_xs0.shape)
        #rint fmt1(just1)[1:] % ('np_src_xs0', np_src_xs0)
    if 0 and prt:
        print fmt0(just1)[1:] % ('np.allclose(np_src_xs0, np_src_xs[0, :]',
            np.allclose(np_src_xs0, np_src_xs[0, :]))

    ngn.tow_edge_xref_idxs = []
    for i, (tow_edge_id,
        tow_edge_xref) in enumerate(
            zip(ngn.tow_edge_ids, ngn.tow_edge_xrefs)):
        tow_edge_xref_idx = (
            find_nearest_value_index(np_src_xs0, tow_edge_xref))
        ngn.tow_edge_xref_idxs.append(tow_edge_xref_idx)
        #f 1 or prt:
        #   if i == 0:
        #       print
        #   print fmt0(just1)[1:] % (
        #       '[i, tow_edge_id, tow_edge_xref, tow_edge_xref_idx]',
        #       ["%.2i" % i, "%.4i" % tow_edge_id, "%7.3f" % tow_edge_xref,
        #           "%.4i" % tow_edge_xref_idx])
    if 1 or prt:
        print fmt1(just1)[1:] % ('ngn.tow_edge_ids', ngn.tow_edge_ids)
        print fmt1(just1)[1:] % ('ngn.tow_edge_xrefs', ngn.tow_edge_xrefs)
        print fmt1(just1)[1:] % ('ngn.tow_edge_xref_idxs',
            ngn.tow_edge_xref_idxs)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    # these are 'diagnostic' plots ...
    if 1 and ngn.make_survey_plots:
        img = np_hasnan_zs
        make_laser_profiles_image_plot(img) if 1 else None
        make_laser_profiles_image_plot_with_rois(img) if 1 else None
        #
        make_laser_profiles_overlay_plot(
            np_src_xs, np_hasnan_zs) if 1 else None
        ### make_laser_measts_overlay_plot(
        ###     np_hasnan_zs) if 1 else None
        ### make_laser_meast_zs_histogram_plot(
        ###     np_hasnan_zs, orientation='horizontal') if 1 else None
        make_laser_measts_overlay_and_zs_histogram_plot(
            np_hasnan_zs) if 1 else None
        ### make_laser_profiles_image_zs_not_isnan_plot(
        ###     np_hasnan_zs) if 1 else None
        ### make_laser_profiles_image_cols_with_zs_isnan_plot(
        ###     np_hasnan_zs) if 1 else None
        None if 1 else sys.exit()

    if 1 and prt:
        print "\n%s" % ('*** ' * ngn.stars_mult)
    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    if 1 and prt:
        print fmt0(just1)[0:] % ("ngn.job_us_csv_abspath",
            ngn.job_us_csv_abspath)

    pd_src_us = pd.read_csv(ngn.job_us_csv_abspath, index_col=False)
    if 1 and prt:
        print fmt0(just1)[0:] % ("pd_src_us.shape", pd_src_us.shape)
        #rint fmt1(just1)[1:] % ("pd_src_us", pd_src_us)
        hd, tl = 4, 4
        print fmt1(just1)[1:] % ("pd_src_us.head(%i)" % hd,
            pd_src_us.head(hd))
        print fmt1(just1)[1:] % ("pd_src_us.tail(%i)" % tl,
            pd_src_us.tail(tl))

    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    print "\n%s" % ('*** ' * ngn.stars_mult)
    print "%s" % ('TOW ' * ngn.stars_mult)
    None if 1 else sys.exit()

    if ngn.skip_tow_ends_analysis:
        if 1 and prt:
            print fmt0(just2)[0:] % ("ngn.skip_tow_ends_analysis",
                ngn.skip_tow_ends_analysis)
    else:
        #== === === === === === === === === === === === === === === === === ===

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # get the "located tow ends events" datasets

        (pd_src_us_tow_present, pd_src_us_tow_diff, tow_key_names,
            tow_diff_names) = get_located_tow_end_events_dataset(
                pd_src_us)

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        if 1 or prt:
            print fmt0(just1)[0:] % ("pd_src_us_tow_present.shape",
                pd_src_us_tow_present.shape)
            #rint fmt1(just1)[1:] % ("pd_src_us_tow_present",
            #   pd_src_us_tow_present)
        ##f 0 and prt:
        ##   hd, tl = (4, 4) if 1 else (98, 98)
        ##   print fmt1(just1)[0:] % ("pd_src_us_tow_present.head(%i)" % hd,
        ##       pd_src_us_tow_present.head(hd))
        ##   print fmt1(just1)[1:] % ("pd_src_us_tow_present.tail(%i)" % tl,
        ##       pd_src_us_tow_present.tail(tl))

        job_zs_csv_pd_src_us_tow_present = (
            ngn.job_zs_csv.replace('.csv', '___us_tow_present.csv'))
        job_zs_csv_pd_src_us_tow_present_abspath = os.path.join(
            ngn.job_absdir, job_zs_csv_pd_src_us_tow_present)
        #f 1 and prt:
        #   print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        #   print fmt0(just2)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
        #   print fmt0(just2)[1:] % ("job_zs_csv_pd_src_us_tow_present",
        #       job_zs_csv_pd_src_us_tow_present)
        print fmt0(just2)[0:] % ("job_zs_csv_pd_src_us_tow_present_abspath",
                job_zs_csv_pd_src_us_tow_present_abspath)

        pd_src_us_tow_present.to_csv(job_zs_csv_pd_src_us_tow_present_abspath,
            index=False, na_rep='NaN')

        if ngn.write_to_results_dir:
            if 0 and prt:
                print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
                print fmt0(just2)[1:] % ("ngn.results_absdir",
                    ngn.results_absdir)

            job_zs_csv_pd_src_us_tow_present_abspath_results = os.path.join(
                ngn.results_absdir, job_zs_csv_pd_src_us_tow_present)
            if 1 or prt:
                #rint fmt0(just2)[0:] % (
                #       "job_zs_csv_pd_src_us_tow_present_abspath",
                #       job_zs_csv_pd_src_us_tow_present_abspath)
                print fmt0(just2)[0:] % (
                    "job_zs_csv_pd_src_us_tow_present_abspath_results",
                    job_zs_csv_pd_src_us_tow_present_abspath_results)

            shutil.copy(job_zs_csv_pd_src_us_tow_present_abspath,
                job_zs_csv_pd_src_us_tow_present_abspath_results)

            None if 1 else sys.exit()

        #f ngn.add_pyapp_version:
        #   job_zs_csv_pd_src_us_tow_present_pyapp_ver = (
        #       job_zs_csv_pd_src_us_tow_present.replace(
        #           '.csv', '_py%.3i.csv' % ngn.version_py))
        #   job_zs_csv_pd_src_us_tow_present_pyapp_ver_abspath = os.path.join(
        #       ngn.job_absdir, job_zs_csv_pd_src_us_tow_present_pyapp_ver)
        #   shutil.copy(job_zs_csv_pd_src_us_tow_present_abspath,
        #       job_zs_csv_pd_src_us_tow_present_pyapp_ver_abspath)
        #   #f 1 and prt:
        #   #   print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        #   #   print fmt0(just2)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
        #   #   print fmt0(just2)[1:] % ("job_zs_csv_pd_src_us_tow_present",
        #   #       job_zs_csv_pd_src_us_tow_present)
        #   #f 1:
        #   #   print fmt0(just2)[0:] % (
        #   #       "job_zs_csv_pd_src_us_tow_present_pyapp_ver",
        #   #       job_zs_csv_pd_src_us_tow_present_pyapp_ver)

        if 1 and prt:
            print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        if 1 or prt:
            print fmt1(just1)[0:] % ("tow_key_names", tow_key_names)
            print fmt1(just1)[1:] % ("tow_diff_names", tow_diff_names)

        if 1 or prt:
            print fmt0(just1)[0:] % ("pd_src_us_tow_diff.shape",
                pd_src_us_tow_diff.shape)
            #rint fmt1(just1)[1:] % ("pd_src_us_tow_diff", pd_src_us_tow_diff)
        #if 0 or prt:
        #    print fmt1(just1)[0:] % (
        #        "pd_src_us_tow_diff[tow_key_names + tow_diff_names]",
        #        pd_src_us_tow_diff[tow_key_names + tow_diff_names])
        #if 1 or prt:
        #    print fmt1(just1)[0:] % (
        #        "pd_src_us_tow_diff[tow_key_names + tow_diff_names].head()",
        #        pd_src_us_tow_diff[tow_key_names + tow_diff_names].head())
        #    print fmt1(just1)[1:] % (
        #        "pd_src_us_tow_diff[tow_key_names + tow_diff_names].tail()",
        #        pd_src_us_tow_diff[tow_key_names + tow_diff_names].tail())

        job_zs_csv_pd_src_us_tow_diff = (
            ngn.job_zs_csv.replace('.csv', '___us_tow_diff.csv'))
        job_zs_csv_pd_src_us_tow_diff_abspath = os.path.join(
            ngn.job_absdir, job_zs_csv_pd_src_us_tow_diff)
        #f 1 and prt:
        #   print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        #   print fmt0(just2)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
        #   print fmt0(just2)[1:] % ("job_zs_csv_pd_src_us_tow_diff",
        #       job_zs_csv_pd_src_us_tow_diff)
        print fmt0(just2)[0:] % ("job_zs_csv_pd_src_us_tow_diff_abspath",
            job_zs_csv_pd_src_us_tow_diff_abspath)

        pd_src_us_tow_diff.to_csv(job_zs_csv_pd_src_us_tow_diff_abspath,
            index=False, na_rep='NaN')

        if ngn.write_to_results_dir:
            if 0 and prt:
                print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
                print fmt0(just2)[1:] % ("ngn.results_absdir",
                    ngn.results_absdir)

            job_zs_csv_pd_src_us_tow_diff_abspath_results = os.path.join(
                ngn.results_absdir, job_zs_csv_pd_src_us_tow_diff)
            if 1 or prt:
                #rint fmt0(just2)[0:] % (
                #       "job_zs_csv_pd_src_us_tow_diff_abspath",
                #       job_zs_csv_pd_src_us_tow_diff_abspath)
                print fmt0(just2)[0:] % (
                    "job_zs_csv_pd_src_us_tow_diff_abspath_results",
                    job_zs_csv_pd_src_us_tow_diff_abspath_results)

            shutil.copy(job_zs_csv_pd_src_us_tow_diff_abspath,
                job_zs_csv_pd_src_us_tow_diff_abspath_results)

            None if 1 else sys.exit()

        #f ngn.add_pyapp_version:
        #   job_zs_csv_pd_src_us_tow_diff_pyapp_ver = (
        #       job_zs_csv_pd_src_us_tow_diff.replace(
        #           '.csv', '_py%.3i.csv' % ngn.version_py))
        #   job_zs_csv_pd_src_us_tow_diff_pyapp_ver_abspath = os.path.join(
        #       ngn.job_absdir, job_zs_csv_pd_src_us_tow_diff_pyapp_ver)
        #   shutil.copy(job_zs_csv_pd_src_us_tow_diff_abspath,
        #       job_zs_csv_pd_src_us_tow_diff_pyapp_ver_abspath)
        #   #f 1 and prt:
        #   #   print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        #   #   print fmt0(just2)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
        #   #   print fmt0(just2)[1:] % ("job_zs_csv_pd_src_us_tow_diff",
        #   #       job_zs_csv_pd_src_us_tow_diff)
        #   #f 1:
        #   #   print fmt0(just2)[0:] % (
        #   #       "job_zs_csv_pd_src_us_tow_diff_pyapp_ver",
        #   #       job_zs_csv_pd_src_us_tow_diff_pyapp_ver)

        if 1 and prt:
            print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # initialize the tow ends results table (Pandas DataFrame)

        if 0 and prt:
            print fmt0(just1)[0:] % ("pd_src_us_tow_diff.shape",
                pd_src_us_tow_diff.shape)

        pd_results_ends = pd.DataFrame(np.full(
            (len(pd_src_us_tow_diff), len(ngn.results_fields_ends)), np.nan),
            columns=ngn.results_fields_ends)
        pd_results_ends[ngn.results_fields_ends_cols_to_copy] = (
            pd_src_us_tow_diff[ngn.results_fields_ends_cols_to_copy])
        pd_results_ends[tow_diff_names] = pd_src_us_tow_diff[tow_diff_names]

        if 1 or prt:
            print fmt0(just1)[0:] % ("pd_results_ends.shape",
                pd_results_ends.shape)
            print fmt1(just1)[1:] % ("pd_results_ends.columns.values",
                pd_results_ends.columns.values)
        if 1 and prt:
            cnum = 40
            hd, tl = (4, 4) if 1 else (98, 98)
            print fmt1(just1)[0:] % ("pd_results_ends[:, :%s].head(%i)" %
                (cnum, hd), pd_results_ends.iloc[:, :cnum].head(hd))
            print fmt1(just1)[1:] % ("pd_results_ends[:, :%s].tail(%i)" %
                (cnum, tl), pd_results_ends.iloc[:, :cnum].tail(tl))
        if 0 and prt:
            cnum = 40
            print fmt1(just1)[1:] % ("pd_results_ends[:, :%s]",
                pd_results_ends.iloc[:, :cnum])

        print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        if 1 and prt:
            print "\n... analyze_tow_ends_placements ... started"

        pd_results_ends = analyze_tow_ends_placements(
            pd_results_ends, ngn.dzdys_threshold, pd_src_us_tow_present,
            pd_src_us_tow_diff, tow_diff_names, np_src_xs, np_hasnan_zs,
            np_hasnan_zs_mask_nans
        ) if 1 else pd_results_ends

#       pd_results_ends = analyze_tow_ends_placements2(
#           pd_results_ends, ngn.dzdys_threshold, pd_src_us_tow_present,
#           pd_src_us_tow_diff, tow_diff_names, np_src_xs, np_hasnan_zs,
#           np_hasnan_zs_mask_nans
#       ) if 1 else pd_results_ends

        if 1 and prt:
            print "\n... analyze_tow_ends_placements ... completed"
        if 0 and prt:
            print fmt0(just1)[0:] % ("pd_results_ends.shape",
                pd_results_ends.shape)
            print fmt1(just1)[0:] % ("pd_results_ends.iloc[:, :24]",
                pd_results_ends.iloc[:, :24])

        None if 1 else sys.exit()

        print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # write the tabular tow end placement results file

        results_ends_csv_abspath = os.path.join(
            ngn.job_absdir, ngn.results_ends_csv)
        if 0 and prt:
            print fmt0(just1)[0:] % ("ngn.job_absdir", ngn.job_absdir)
            print fmt0(just1)[1:] % ("ngn.results_dir", ngn.results_dir)
            print fmt0(just1)[1:] % ("ngn.results_ends_csv",
                ngn.results_ends_csv)
            print fmt0(just1)[1:] % ("ngn.results_ends_csv_pyapp_ver",
                ngn.results_ends_csv_pyapp_ver)
        if 1 and prt:
            print fmt1(just1)[0:] % ("results_ends_csv_abspath",
                results_ends_csv_abspath)

        pd_results_ends.to_csv(
            results_ends_csv_abspath, index=False, na_rep='NaN')

        if ngn.write_to_results_dir:
            if 1 and prt:
                print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
                print fmt0(just2)[1:] % ("ngn.results_absdir",
                    ngn.results_absdir)

            results_ends_csv_abspath_results = os.path.join(
                ngn.results_absdir, ngn.results_ends_csv)
            if 1 or prt:
                #rint fmt0(just2)[0:] % ("results_ends_csv_abspath",
                #       results_ends_csv_abspath)
                print fmt1(just2)[0:] % ("results_ends_csv_abspath_results",
                        results_ends_csv_abspath_results)

            shutil.copy(
                results_ends_csv_abspath, results_ends_csv_abspath_results)

            None if 1 else sys.exit()

        #f ngn.add_pyapp_version:
        #   results_ends_csv_pyapp_ver_abspath = os.path.join(
        #       ngn.job_absdir, ngn.results_ends_csv_pyapp_ver)
        #   if 1 and prt:
        #       print fmt1(just1)[0:] % ("results_ends_csv_pyapp_ver_abspath",
        #           results_ends_csv_pyapp_ver_abspath)
        #
        #   shutil.copy(results_ends_csv_abspath,
        #       results_ends_csv_pyapp_ver_abspath)

        #== === === === === === === === === === === === === === === === === ===

    print "\n%s" % ('TOW ' * ngn.stars_mult)
    print "%s" % ('*** ' * ngn.stars_mult)
    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    print "\n%s" % ('*** ' * ngn.stars_mult)
    print "%s" % ('GAP ' * ngn.stars_mult)
    None if 1 else sys.exit()

    if ngn.skip_tow_gaps_analysis:
        if 1 and prt:
            print fmt0(just2)[0:] % ("ngn.skip_tow_gaps_analysis",
                ngn.skip_tow_gaps_analysis)
    else:
        #== === === === === === === === === === === === === === === === === ===

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # get the "located gap events" datasets

        pd_src_us_gap_present, pd_src_us_gap_any = (
            get_located_tow_gap_events_dataset(ngn, pd_src_us))
        #
        if 1 and prt:
            print fmt0(just1)[0:] % ("pd_src_us_gap_present.shape",
                pd_src_us_gap_present.shape)
        if 0 and prt:
            hd, tl = (4, 4) if 1 else (98, 98)
            print fmt1(just1)[0:] % ("pd_src_us_gap_present.head(%i)" %
                hd, pd_src_us_gap_present.head(hd))
            print fmt1(just1)[1:] % ("pd_src_us_gap_present.tail(%i)" %
                tl, pd_src_us_gap_present.tail(tl))
        #
        if 1 and prt:
            print fmt0(just1)[0:] % ("pd_src_us_gap_any.shape",
                pd_src_us_gap_any.shape)
        if 0 and prt:
            hd, tl = (4, 4) if 1 else (10, 10)
            print fmt1(just1)[0:] % ("pd_src_us_gap_any.head(%i)" % hd,
                pd_src_us_gap_any.head(hd))
            print fmt1(just1)[0:] % ("pd_src_us_gap_any.tail(%i)" % tl,
                pd_src_us_gap_any.tail(tl))

        gap_any_names = list(pd_src_us_gap_any.columns)
        gap_present_names = [s for s in pd_src_us_gap_present.columns
            if s not in gap_any_names]
        if 0 and prt:
            print fmt1(just1)[0:] % ("gap_any_names", gap_any_names)
            print fmt1(just1)[1:] % ("gap_present_names", gap_present_names)

        None if 1 else sys.exit()

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        job_zs_csv_pd_src_us_gap_any = (
            ngn.job_zs_csv.replace('.csv', '___us_gap_any.csv'))
        job_zs_csv_pd_src_us_gap_any_abspath = os.path.join(
            ngn.job_absdir, job_zs_csv_pd_src_us_gap_any)
        pd_src_us_gap_any.to_csv(
            job_zs_csv_pd_src_us_gap_any_abspath, index=False, na_rep='NaN')
        #f 1 and prt:
        #   print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        #   print fmt0(just2)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
        #   print fmt0(just2)[1:] % ("job_zs_csv_pd_src_us_gap_any",
        #       job_zs_csv_pd_src_us_gap_any)
        if 1:
            print fmt1(just2)[0:] % ("job_zs_csv_pd_src_us_gap_any_abspath",
                job_zs_csv_pd_src_us_gap_any_abspath)

        if ngn.write_to_results_dir:
            if 0 and prt:
                print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
                print fmt0(just2)[1:] % ("ngn.results_absdir",
                    ngn.results_absdir)

            job_zs_csv_pd_src_us_gap_any_abspath_results = (
                os.path.join(ngn.results_absdir, job_zs_csv_pd_src_us_gap_any))
            if 1 or prt:
                #rint fmt0(just2)[0:] % (
                #       "job_zs_csv_pd_src_us_gap_any_abspath",
                #       job_zs_csv_pd_src_us_gap_any_abspath)
                print fmt1(just2)[0:] % (
                    "job_zs_csv_pd_src_us_gap_any_abspath_results",
                    job_zs_csv_pd_src_us_gap_any_abspath_results)

            shutil.copy(job_zs_csv_pd_src_us_gap_any_abspath,
                job_zs_csv_pd_src_us_gap_any_abspath_results)

        #f ngn.add_pyapp_version:
        #
        #   job_zs_csv_pd_src_us_gap_any_pyapp_ver = (
        #       job_zs_csv_pd_src_us_gap_any.replace(
        #           '.csv', '_py%.3i.csv' % ngn.version_py))
        #   job_zs_csv_pd_src_us_gap_any_pyapp_ver_abspath = os.path.join(
        #       ngn.job_absdir, job_zs_csv_pd_src_us_gap_any_pyapp_ver)
        #   shutil.copy(job_zs_csv_pd_src_us_gap_any_abspath,
        #       job_zs_csv_pd_src_us_gap_any_pyapp_ver_abspath)
        #   #f 1 and prt:
        #   #   print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
        #   #   print fmt0(just2)[1:] % ("ngn.job_zs_csv", ngn.job_zs_csv)
        #   #   print fmt0(just2)[1:] % ("job_zs_csv_pd_src_us_gap_any",
        #   #       job_zs_csv_pd_src_us_gap_any)
        #   #f 1:
        #   #   print fmt0(just2)[0:] % (
        #   #       "job_zs_csv_pd_src_us_gap_any_pyapp_ver",
        #   #       job_zs_csv_pd_src_us_gap_any_pyapp_ver)

        if 1 and prt:
            print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # initialize the tow edges image (2-D Numpy array)

        ngn.np_edges_image = np.zeros(np_src_zs.shape).astype(np.bool)
        if 1 and prt:
            print fmt0(just1)[0:] % ("(init) ngn.np_edges_image.shape",
                ngn.np_edges_image.shape)

        if 0:  # quickplot
            image = ngn.np_edges_image.copy()
            # presentation "fix" ...
            image[0, 0] = image[0, 0] if np.any(image) else True
            #
            png = None
            title = None
            cmap = 'gray' if 1 else 'jet'
            colorbar = True if 0 else False
            xlabel = 'Profile Index'
            ylabel = 'Measurement Index'
            imshow(image.T, title=title, cmap=cmap,
                xticklabel=True, yticklabel=True, xlabel=xlabel, ylabel=ylabel,
                colorbar=colorbar, png=png) if 1 else None
            None if 0 else sys.exit()

        print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        if 1 and prt:
            print fmt0(just1)[0:] % ("ngn.legacy_gap_analysis",
                ngn.legacy_gap_analysis)

        if 1 and prt:
            print "\n... analyze_tow_gaps ... started"

        # initialize the tow gap results table (Pandas DataFrame)
        #  without any rows (yet)
        pd_results_gap = pd.DataFrame(columns=ngn.results_fields_gap)

        # this is "dumb" gap algo ...
        pd_results_gap = analyze_tow_gaps_legacy_gap_analysis(
            np_src_xs, np_hasnan_zs, np_hasnan_zs_mask_nans, pd_results_gap
        ) if 1 and ngn.legacy_gap_analysis else pd_results_gap

        # this is "smart" gap algo ...
        pd_results_gap = analyze_tow_gaps(
            np_src_xs, np_hasnan_zs, np_hasnan_zs_mask_nans, pd_src_us_gap_any,
            pd_results_gap
        ) if 1 and not ngn.legacy_gap_analysis else pd_results_gap

        if 1 and prt:
            print "\n... analyze_tow_gaps ... completed"

        if 1 and prt:
            print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # write the tow edges image (2-D Numpy array)

        if 1 and prt:
            print fmt0(just1)[0:] % ("(updt) ngn.np_edges_image.shape",
                ngn.np_edges_image.shape)

        if 0:  # quickplot
            image = ngn.np_edges_image.copy()
            # presentation "fix" ...
            image[0, 0] = image[0, 0] if np.any(image) else True
            #
            png = None
            title = None
            cmap = 'gray' if 1 else 'jet'
            colorbar = True if 0 else False
            xlabel = 'Profile Index'
            ylabel = 'Measurement Index'
            imshow(image.T, title=title, cmap=cmap,
                xticklabel=True, yticklabel=True, xlabel=xlabel, ylabel=ylabel,
                colorbar=colorbar, png=png) if 1 else None
            None if 0 else sys.exit()

        make_laser_profile_edges_image_plot(
            ngn, ngn.np_edges_image) if 1 else None
        #one if 1 else sys.exit()

        print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # write the tabular tow gap results file

        ### finalize the tow gap results table (Pandas DataFrame)
        #
        pd_results_gap.loc[:, ngn.results_fields_gaps_cols_to_copy] = (
            pd_src_us_gap_present.loc[
                pd_results_gap.index, ngn.results_fields_gaps_cols_to_copy])
        pd_results_gap.ix[:, gap_present_names] = (
            pd_src_us_gap_present.loc[pd_results_gap.index, gap_present_names])

        if 0 and prt:
            hd = len(pd_results_gap)
            print fmt1(just1)[0:] % (
                "pd_src_us_gap_present.loc[pd_results_gap.index, " +
                "ngn.results_fields_gaps_cols_to_copy].head(%i)" % hd,
                pd_src_us_gap_present.loc[pd_results_gap.index,
                    ngn.results_fields_gaps_cols_to_copy].head(hd))
        if 0 and prt:
            print fmt1(just1)[0:] % ("pd_results_tow_gap.loc[:, " +
                "ngn.results_fields_gaps_cols_to_copy]",
                pd_results_gap.loc[:, ngn.results_fields_gaps_cols_to_copy])
        if 1 or prt:
            print fmt0(just1)[0:] % ("pd_results_gap.shape",
                pd_results_gap.shape)
        if 0 and prt:
            print fmt1(just1)[0:] % ("pd_results_tow_gap.iloc[:, :24].head()",
                pd_results_gap.iloc[:, :24].head())
            print fmt1(just1)[0:] % ("pd_results_tow_gap.iloc[:, :24].tail()",
                pd_results_gap.iloc[:, :24].tail())

        #.. ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...

        results_gap_csv_abspath = os.path.join(
            ngn.job_absdir, ngn.results_gap_csv)
        if 0 and prt:
            print fmt0(just1)[0:] % ("ngn.job_absdir", ngn.job_absdir)
            print fmt0(just1)[1:] % ("ngn.results_dir", ngn.results_dir)
            print fmt0(just1)[1:] % ("ngn.results_gap_csv",
                ngn.results_gap_csv)
            print fmt0(just1)[1:] % ("ngn.results_gap_csv_pyapp_ver",
                ngn.results_gap_csv_pyapp_ver)
        if 1 and prt:
            print fmt1(just1)[0:] % ("results_gap_csv_abspath",
                results_gap_csv_abspath)

        pd_results_gap.to_csv(
            results_gap_csv_abspath, index=False, na_rep='NaN') if 1 else None

        if ngn.write_to_results_dir:
            if 0 and prt:
                print fmt0(just2)[0:] % ("ngn.job_absdir", ngn.job_absdir)
                print fmt0(just2)[1:] % ("ngn.results_absdir",
                    ngn.results_absdir)

            results_gap_csv_abspath_results = (
                os.path.join(ngn.results_absdir, ngn.results_gap_csv))
            if 1 or prt:
                #rint fmt0(just2)[0:] % ("results_gap_csv_abspath",
                #       results_gap_csv_abspath)
                print fmt1(just2)[0:] % ("results_gap_csv_abspath_results",
                        results_gap_csv_abspath_results)

            shutil.copy(results_gap_csv_abspath,
                results_gap_csv_abspath_results)

        #f 1 and ngn.add_pyapp_version:
        #   results_gap_csv_pyapp_ver_abspath = os.path.join(
        #       ngn.job_absdir, ngn.results_gap_csv_pyapp_ver)
        #   if 1 and prt:
        #       print fmt1(just1)[0:] % ("results_gap_csv_pyapp_ver_abspath",
        #           results_gap_csv_pyapp_ver_abspath)
        #
        #   shutil.copy(results_gap_csv_abspath,
        #       results_gap_csv_pyapp_ver_abspath)

        print "\n%s" % ('*** ' * ngn.stars_mult)
        None if 1 else sys.exit()

        #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        # write the gap results plot

        make_gap_results_plot(ngn, pd_results_gap) if 1 else None
        #one if 1 else sys.exit()

        #== === === === === === === === === === === === === === === === === ===

    print "\n%s" % ('GAP ' * ngn.stars_mult)
    print "%s" % ('*** ' * ngn.stars_mult)
    None if 1 else sys.exit()

    #== === === === === === === === === === === === === === === === === === ===

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s" % (mult_str * ngn.mult)
        print fmt0(just1)[1:] % ('execution completed',
            make_elapsed_time(ngn, epoch_secs))
        print fmt0(just1)[1:] % ('(end) def %s' % def_str, ngn.in_)


# (above) 'Application' functions
#==.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===
# (below) python script entry point


def parse_argv():
    """
    Provides an API for executing this module as a program.
    """
    just0, just1, just2 = ngn.just0, ngn.just1, ngn.just2
    prt = False if 0 else True
    prt_ = prt
    mult_str = '=== '
    def_str = 'parse_argv'
    if prt_:
        print "\n... (beg) def %s ...\n%s" % (def_str, mult_str * ngn.mult)
    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    # argparse reference:
    #   https://mkaz.com/2014/07/26/python-argparse-cookbook/

    prt = True and prt_ if 1 else False

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # instantiate the argparse object

    argparser = argparse.ArgumentParser(description=' '.join([
        "evaluates ends and gaps of fiber placements",
        "given one fabrication pass of laser measurement profiles",
    ]))

    ### ### required 'flag' arguements
    #
    argparser.add_argument('--job_xs_csv_abspath', required=True,
        type=str, help="absolute path to the laser measurements xs file")
    #
    argparser.add_argument('--job_zs_csv_abspath', required=True,
        type=str, help="absolute path to the laser measurements zs file")
    #
    argparser.add_argument('--job_us_csv_abspath', required=True,
        type=str, help="absolute path to the located tow ends & gaps file")
    #
    argparser.add_argument('--job_config_txt', required=True,
        type=str, help="the job configuration file name")

    ### ### boolean 'flag' arguements
    #
    argparser.add_argument('--preview', action='store_true',
        default=False,
        help='execution of dataset diagnostics for setting job parameters')
    #
    argparser.add_argument('--legacy_gap_analysis', action='store_true',
        default=False,
        help='execution using a "dumb" gaps analysis algorithm')
    #
    argparser.add_argument('--add_pyapp_version', action='store_true',
        default=False, help="add Python application version to results files")
    #
    argparser.add_argument('--write_to_results_dir', action='store_true',
        default=False, help="write outputs to the results directory")
    #
    argparser.add_argument('--count_gap_rois_analyzed', action='store_true',
        default=False, help='count gap regions of interest analyzed')
    #
    argparser.add_argument('--autothreshold_nan_values', action='store_true',
        default=False, help='automatically set "NaN" value thresholds')
    #
    ## plotting options:
    #
    argparser.add_argument('--make_survey_plots', action='store_true',
        default=False, help="write overlay plots to results directory")
    #
    argparser.add_argument('--make_gallery00_plots', action='store_true',
        default=False, help="write gallery00 plots to results directory")
    #
    argparser.add_argument('--make_gallery01_plots', action='store_true',
        default=False, help="write gallery01 plots to results directory")
    #
    argparser.add_argument('--make_gallery02_plots', action='store_true',
        default=False, help="write gallery02 plots to results directory")
    #
    argparser.add_argument('--make_gallery03_plots', action='store_true',
        default=False, help="write gallery03 plots to results directory")
    #
    argparser.add_argument('--make_gallery04_plots', action='store_true',
        default=False, help="write gallery04 plots to results directory")
    #
    argparser.add_argument('--make_gallery05_plots', action='store_true',
        default=False, help="write gallery05 plots to results directory")
    #
    ## analysis options:
    #
    argparser.add_argument('--skip_tow_ends_analysis', action='store_true',
        default=False, help="do not execute tow ends analysis")
    #
    argparser.add_argument('--skip_tow_gaps_analysis', action='store_true',
        default=False, help="do not execute tow gaps analysis")

    args_parse = argparser.parse_args()
    if 1 and prt:
        #rint fmt1(just0)[0:] % ("args_parse", args_parse)
        print fmt1(just0)[0:] % ("prt_dict(vars(args_parse))",
            prt_dict(vars(args_parse), 30))
    #f 1 and prt:
    #   print fmt0(just0)[0:] % ("args_parse.job_us_csv_abspath is None",
    #       args_parse.job_us_csv_abspath is None)

    None if 1 else sys.exit()

    #-- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # process the job

    ngn.job_init(args_parse)

    # (line match)
    # (line match)
    process_job_preview() if args_parse.preview else process_job()

    #--.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---
    if prt_:
        print "\n%s\n... (end) def %s ..." % (mult_str * ngn.mult, def_str)


# (above) python script entry point
#==.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===.===
if __name__ == '__main__':

    try:
        exit_code = 0
        parse_argv()

    except SystemExit:
        # may be used during development
        exit_code = -1

    except Exception, err:
        exit_code = 1
        print 'print_exc():'
        traceback.print_exc(file=sys.stdout)

    finally:
        # this gets executed regardless ...
        exit_str = "exit_code: %s" % exit_code
        print "\n%s" % exit_str
        exit_code_filename = 'z__exit_code.txt'
        with open(exit_code_filename, 'w', 0) as f:
            f.write(exit_str)
