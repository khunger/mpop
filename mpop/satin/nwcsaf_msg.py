#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2012, 2014.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Marco Sassi <> for CRR, PC (partly), SPhR, PRPh, CRPh
#   Jörg Asmus <joerg.asmus@dwd.de> for CRR, PC (partly), SPhR, PRPh, CRPh

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Plugin for reading NWCSAF MSG products hdf files.
"""
import ConfigParser
import os.path
from mpop import CONFIG_PATH
import mpop.channel
import numpy as np
import pyresample.utils

import glob
from mpop.utils import get_logger
from mpop.projector import get_area_def

LOG = get_logger('satin/nwcsaf_msg')
COMPRESS_LVL = 6


def pcs_def_from_region(region):
    items = region.proj_dict.items()
    return ' '.join([t[0] + '=' + t[1] for t in items])


def _get_area_extent(cfac, lfac, coff, loff, numcols, numlines):
    """Get the area extent from msg parameters.
    """

    xur = (numcols - coff) * 2 ** 16 / (cfac * 1.0)
    xur = np.deg2rad(xur) * 35785831.0
    xll = (-1 - coff) * 2 ** 16 / (cfac * 1.0)
    xll = np.deg2rad(xll) * 35785831.0
    xres = (xur - xll) / numcols
    xur, xll = xur - xres / 2, xll + xres / 2
    yll = (numlines - loff) * 2 ** 16 / (-lfac * 1.0)
    yll = np.deg2rad(yll) * 35785831.0
    yur = (-1 - loff) * 2 ** 16 / (-lfac * 1.0)
    yur = np.deg2rad(yur) * 35785831.0
    yres = (yur - yll) / numlines
    yll, yur = yll + yres / 2, yur - yres / 2
    return xll, yll, xur, yur


def get_area_extent(filename):
    """Get the area extent of the data in *filename*.
    """
    import h5py
    h5f = h5py.File(filename, 'r')
    aex = _get_area_extent(h5f.attrs["CFAC"],
                           h5f.attrs["LFAC"],
                           h5f.attrs["COFF"],
                           h5f.attrs["LOFF"],
                           h5f.attrs["NC"],
                           h5f.attrs["NL"])
    h5f.close()
    return aex


def _get_palette(h5f, dsname):
    try:
        p = h5f[dsname].attrs['PALETTE']
        return h5f[p].value
    except KeyError:
        return None

class MsgCloudTypeData(object):

    """NWCSAF/MSG Cloud Type data layer
    """

    def __init__(self):
        self.data = None
        self.scaling_factor = 1
        self.offset = 0
        self.num_of_lines = 0
        self.num_of_columns = 0
        self.product = ""
        self.id = ""


class MsgCloudType(mpop.channel.GenericChannel):

    """NWCSAF/MSG Cloud Type data structure as retrieved from HDF5
    file. Resolution sets the nominal resolution of the data.
    """

    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "CloudType")
        self.filled = False
        self.name = "CloudType"
        self.package = ""
        self.saf = ""
        self.product_name = ""
        self.num_of_columns = 0
        self.num_of_lines = 0
        self.projection_name = ""
        self.pcs_def = ""
        self.xscale = 0
        self.yscale = 0
        self.ll_lon = 0.0
        self.ll_lat = 0.0
        self.ur_lon = 0.0
        self.ur_lat = 0.0
        self.region_name = ""
        self.cfac = 0
        self.lfac = 0
        self.coff = 0
        self.loff = 0
        self.nb_param = 0
        self.gp_sc_id = 0
        self.image_acquisition_time = 0
        self.spectral_channel_id = 0
        self.nominal_product_time = 0
        self.sgs_product_quality = 0
        self.sgs_product_completeness = 0
        self.product_algorithm_version = ""
        self.cloudtype = None
        self.processing_flags = None
        self.cloudphase = None
        self.shape = None
        self.satid = ""
        self.qc_straylight = -1
        self.cloudtype_palette = None
        self.cloudphase_palette = None

    def __str__(self):
        return ("'%s: shape %s, resolution %sm'" %
                (self.name,
                 self.cloudtype.shape,
                 self.resolution))

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

# ------------------------------------------------------------------
    def read(self, filename, calibrate=True):
        """Reader for the NWCSAF/MSG cloudtype. Use *filename* to read data.
        """
        import h5py

        self.cloudtype = MsgCloudTypeData()
        self.processing_flags = MsgCloudTypeData()
        self.cloudphase = MsgCloudTypeData()

        LOG.debug("Filename = <" + str(filename) + ">")
        h5f = h5py.File(filename, 'r')
        # pylint: disable-msg=W0212
        self.package = h5f.attrs["PACKAGE"]
        self.saf = h5f.attrs["SAF"]
        self.product_name = h5f.attrs["PRODUCT_NAME"]
        self.num_of_columns = h5f.attrs["NC"]
        self.num_of_lines = h5f.attrs["NL"]
        self.projection_name = h5f.attrs["PROJECTION_NAME"]
        self.region_name = h5f.attrs["REGION_NAME"]
        self.cfac = h5f.attrs["CFAC"]
        self.lfac = h5f.attrs["LFAC"]
        self.coff = h5f.attrs["COFF"]
        self.loff = h5f.attrs["LOFF"]
        self.nb_param = h5f.attrs["NB_PARAMETERS"]
        self.gp_sc_id = h5f.attrs["GP_SC_ID"]
        self.image_acquisition_time = h5f.attrs["IMAGE_ACQUISITION_TIME"]
        self.spectral_channel_id = h5f.attrs["SPECTRAL_CHANNEL_ID"]
        self.nominal_product_time = h5f.attrs["NOMINAL_PRODUCT_TIME"]
        self.sgs_product_quality = h5f.attrs["SGS_PRODUCT_QUALITY"]
        self.sgs_product_completeness = h5f.attrs["SGS_PRODUCT_COMPLETENESS"]
        self.product_algorithm_version = h5f.attrs["PRODUCT_ALGORITHM_VERSION"]
        # pylint: enable-msg=W0212
        # ------------------------

        # The cloudtype data
        h5d = h5f['CT']
        self.cloudtype.data = h5d[:, :]
        self.cloudtype.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.cloudtype.offset = h5d.attrs["OFFSET"]
        self.cloudtype.num_of_lines = h5d.attrs["N_LINES"]
        self.cloudtype.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.cloudtype.num_of_lines,
                      self.cloudtype.num_of_columns)
        self.cloudtype.product = h5d.attrs["PRODUCT"]
        self.cloudtype.id = h5d.attrs["ID"]
        self.cloudtype_palette = _get_palette(h5f, 'CT')
        # ------------------------

        # The cloud phase data
        h5d = h5f['CT_PHASE']
        self.cloudphase.data = h5d[:, :]
        self.cloudphase.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.cloudphase.offset = h5d.attrs["OFFSET"]
        self.cloudphase.num_of_lines = h5d.attrs["N_LINES"]
        self.cloudphase.num_of_columns = h5d.attrs["N_COLS"]
        self.cloudphase.product = h5d.attrs["PRODUCT"]
        self.cloudphase.id = h5d.attrs["ID"]
        self.cloudphase_palette = _get_palette(h5f, 'CT_PHASE')

        # ------------------------

        # The cloudtype processing/quality flags
        h5d = h5f['CT_QUALITY']
        self.processing_flags.data = h5d[:, :]
        self.processing_flags.scaling_factor = \
            h5d.attrs["SCALING_FACTOR"]
        self.processing_flags.offset = h5d.attrs["OFFSET"]
        self.processing_flags.num_of_lines = h5d.attrs["N_LINES"]
        self.processing_flags.num_of_columns = h5d.attrs["N_COLS"]
        self.processing_flags.product = h5d.attrs["PRODUCT"]
        self.processing_flags.id = h5d.attrs["ID"]
        # ------------------------

        h5f.close()

        self.cloudtype = self.cloudtype.data
        self.cloudphase = self.cloudphase.data
        self.processing_flags = self.processing_flags.data

        self.area = get_area_from_file(filename)

        self.filled = True

    def save(self, filename):
        """Save the current cloudtype object to hdf *filename*, in pps format.
        """
        import h5py
        ctype = self.convert2pps()
        LOG.info("Saving CType hdf file...")
        ctype.save(filename)
        h5f = h5py.File(filename, mode="a")
        h5f.attrs["straylight_contaminated"] = self.qc_straylight
        h5f.close()
        LOG.info("Saving CType hdf file done !")

    def project(self, coverage):
        """Remaps the NWCSAF/MSG Cloud Type to cartographic map-projection on
        area give by a pre-registered area-id. Faster version of msg_remap!
        """
        LOG.info("Projecting channel %s..." % (self.name))

        region = coverage.out_area
        dest_area = region.area_id

        retv = MsgCloudType()

        retv.name = self.name
        retv.package = self.package
        retv.saf = self.saf
        retv.product_name = self.product_name
        retv.region_name = dest_area
        retv.cfac = self.cfac
        retv.lfac = self.lfac
        retv.coff = self.coff
        retv.loff = self.loff
        retv.nb_param = self.nb_param
        retv.gp_sc_id = self.gp_sc_id
        retv.image_acquisition_time = self.image_acquisition_time
        retv.spectral_channel_id = self.spectral_channel_id
        retv.nominal_product_time = self.nominal_product_time
        retv.sgs_product_quality = self.sgs_product_quality
        retv.sgs_product_completeness = self.sgs_product_completeness
        retv.product_algorithm_version = self.product_algorithm_version

        retv.cloudtype = coverage.project_array(self.cloudtype)
        retv.cloudtype_palette = self.cloudtype_palette

        retv.cloudphase = coverage.project_array(self.cloudphase)
        retv.cloudphase_palette = self.cloudphase_palette

        retv.processing_flags = \
            coverage.project_array(self.processing_flags)

        retv.qc_straylight = self.qc_straylight
        retv.region_name = dest_area
        retv.area = region
        retv.projection_name = region.proj_id

        retv.pcs_def = pcs_def_from_region(region)

        retv.num_of_columns = region.x_size
        retv.num_of_lines = region.y_size
        retv.xscale = region.pixel_size_x
        retv.yscale = region.pixel_size_y

        import pyproj
        prj = pyproj.Proj(region.proj4_string)
        aex = region.area_extent
        lonur, latur = prj(aex[2], aex[3], inverse=True)
        lonll, latll = prj(aex[0], aex[1], inverse=True)
        retv.ll_lon = lonll
        retv.ll_lat = latll
        retv.ur_lon = lonur
        retv.ur_lat = latur

        self.shape = region.shape

        retv.filled = True
        retv.resolution = self.resolution

        return retv

# is it necessary?

#    def convert2nordrad(self):
#        return NordRadCType(self)


class MsgCTTHData(object):

    """CTTH data object.
    """

    def __init__(self):
        self.data = None
        self.scaling_factor = 1
        self.offset = 0
        self.num_of_lines = 0
        self.num_of_columns = 0
        self.product = ""
        self.id = ""


class MsgCTTH(mpop.channel.GenericChannel):

    """CTTH channel.
    """

    def __init__(self, resolution=None):
        mpop.channel.GenericChannel.__init__(self, "CTTH")
        self.filled = False
        self.name = "CTTH"
        self.resolution = resolution
        self.package = ""
        self.saf = ""
        self.product_name = ""
        self.num_of_columns = 0
        self.num_of_lines = 0
        self.projection_name = ""
        self.region_name = ""
        self.cfac = 0
        self.lfac = 0
        self.coff = 0
        self.loff = 0
        self.nb_param = 0
        self.gp_sc_id = 0
        self.image_acquisition_time = 0
        self.spectral_channel_id = 0
        self.nominal_product_time = 0
        self.sgs_product_quality = 0
        self.sgs_product_completeness = 0
        self.product_algorithm_version = ""
        self.cloudiness = None  # Effective cloudiness
        self.processing_flags = None
        self.height = None
        self.temperature = None
        self.pressure = None
        self.satid = ""

    def __str__(self):
        return ("'%s: shape %s, resolution %sm'" %
                (self.name,
                 self.shape,
                 self.resolution))

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

    def read(self, filename, calibrate=True):
        import h5py

        self.cloudiness = MsgCTTHData()  # Effective cloudiness
        self.temperature = MsgCTTHData()
        self.height = MsgCTTHData()
        self.pressure = MsgCTTHData()
        self.processing_flags = MsgCTTHData()

        h5f = h5py.File(filename, 'r')

        # The header
        # pylint: disable-msg=W0212
        self.package = h5f.attrs["PACKAGE"]
        self.saf = h5f.attrs["SAF"]
        self.product_name = h5f.attrs["PRODUCT_NAME"]
        self.num_of_columns = h5f.attrs["NC"]
        self.num_of_lines = h5f.attrs["NL"]
        self.projection_name = h5f.attrs["PROJECTION_NAME"]
        self.region_name = h5f.attrs["REGION_NAME"]
        self.cfac = h5f.attrs["CFAC"]
        self.lfac = h5f.attrs["LFAC"]
        self.coff = h5f.attrs["COFF"]
        self.loff = h5f.attrs["LOFF"]
        self.nb_param = h5f.attrs["NB_PARAMETERS"]
        self.gp_sc_id = h5f.attrs["GP_SC_ID"]
        self.image_acquisition_time = h5f.attrs["IMAGE_ACQUISITION_TIME"]
        self.spectral_channel_id = h5f.attrs["SPECTRAL_CHANNEL_ID"]
        self.nominal_product_time = h5f.attrs["NOMINAL_PRODUCT_TIME"]
        self.sgs_product_quality = h5f.attrs["SGS_PRODUCT_QUALITY"]
        self.sgs_product_completeness = h5f.attrs["SGS_PRODUCT_COMPLETENESS"]
        self.product_algorithm_version = h5f.attrs["PRODUCT_ALGORITHM_VERSION"]
        # pylint: enable-msg=W0212
        # ------------------------

        # The CTTH cloudiness data
        h5d = h5f['CTTH_EFFECT']
        self.cloudiness.data = h5d[:, :]
        self.cloudiness.scaling_factor = \
            h5d.attrs["SCALING_FACTOR"]
        self.cloudiness.offset = h5d.attrs["OFFSET"]
        self.cloudiness.num_of_lines = h5d.attrs["N_LINES"]
        self.cloudiness.num_of_columns = h5d.attrs["N_COLS"]
        self.cloudiness.product = h5d.attrs["PRODUCT"]
        self.cloudiness.id = h5d.attrs["ID"]

        self.cloudiness.data = np.ma.masked_equal(self.cloudiness.data, 255)
        self.cloudiness = np.ma.masked_equal(self.cloudiness.data, 0)
        self.cloudiness_palette = _get_palette(h5f, 'CTTH_EFFECT')

        # ------------------------

        # The CTTH temperature data
        h5d = h5f['CTTH_TEMPER']
        self.temperature.data = h5d[:, :]
        self.temperature.scaling_factor = \
            h5d.attrs["SCALING_FACTOR"]
        self.temperature.offset = h5d.attrs["OFFSET"]
        self.temperature.num_of_lines = h5d.attrs["N_LINES"]
        self.shape = (self.temperature.num_of_lines,
                      self.temperature.num_of_columns)
        self.temperature.num_of_columns = h5d.attrs["N_COLS"]
        self.temperature.product = h5d.attrs["PRODUCT"]
        self.temperature.id = h5d.attrs["ID"]

        self.temperature.data = np.ma.masked_equal(self.temperature.data, 0)
        if calibrate:
            self.temperature = (self.temperature.data *
                                self.temperature.scaling_factor +
                                self.temperature.offset)
        else:
            self.temperature = self.temperature.data
        self.temperature_palette = _get_palette(h5f, 'CTTH_TEMPER') / 255.0

        # ------------------------

        # The CTTH pressure data
        h5d = h5f['CTTH_PRESS']
        self.pressure.data = h5d[:, :]
        self.pressure.scaling_factor = \
            h5d.attrs["SCALING_FACTOR"]
        self.pressure.offset = h5d.attrs["OFFSET"]
        self.pressure.num_of_lines = h5d.attrs["N_LINES"]
        self.pressure.num_of_columns = h5d.attrs["N_COLS"]
        self.pressure.product = h5d.attrs["PRODUCT"]
        self.pressure.id = h5d.attrs["ID"]

        self.pressure.data = np.ma.masked_equal(self.pressure.data, 255)
        self.pressure.data = np.ma.masked_equal(self.pressure.data, 0)
        if calibrate:
            self.pressure = (self.pressure.data *
                             self.pressure.scaling_factor +
                             self.pressure.offset)
        else:
            self.pressure = self.pressure.data
        self.pressure_palette = _get_palette(h5f, 'CTTH_PRESS') / 255.0

        # ------------------------

        # The CTTH height data
        h5d = h5f['CTTH_HEIGHT']
        self.height.data = h5d[:, :]
        self.height.scaling_factor = \
            h5d.attrs["SCALING_FACTOR"]
        self.height.offset = h5d.attrs["OFFSET"]
        self.height.num_of_lines = h5d.attrs["N_LINES"]
        self.height.num_of_columns = h5d.attrs["N_COLS"]
        self.height.product = h5d.attrs["PRODUCT"]
        self.height.id = h5d.attrs["ID"]

        self.height.data = np.ma.masked_equal(self.height.data, 255)
        self.height.data = np.ma.masked_equal(self.height.data, 0)
        if calibrate:
            self.height = (self.height.data *
                           self.height.scaling_factor +
                           self.height.offset)
        else:
            self.height = self.height.data
        self.height_palette = _get_palette(h5f, 'CTTH_HEIGHT') / 255.0

        # ------------------------

        # The CTTH processing/quality flags
        h5d = h5f['CTTH_QUALITY']
        self.processing_flags.data = h5d[:, :]
        self.processing_flags.scaling_factor = \
            h5d.attrs["SCALING_FACTOR"]
        self.processing_flags.offset = h5d.attrs["OFFSET"]
        self.processing_flags.num_of_lines = \
            h5d.attrs["N_LINES"]
        self.processing_flags.num_of_columns = \
            h5d.attrs["N_COLS"]
        self.processing_flags.product = h5d.attrs["PRODUCT"]
        self.processing_flags.id = h5d.attrs["ID"]

        self.processing_flags = \
            np.ma.masked_equal(self.processing_flags.data, 0)

        h5f.close()

        self.shape = self.height.shape

        self.area = get_area_from_file(filename)

        self.filled = True

    def save(self, filename):
        """Save the current CTTH channel to HDF5 format.
        """
        ctth = self.convert2pps()
        LOG.info("Saving CTTH hdf file...")
        ctth.save(filename)
        LOG.info("Saving CTTH hdf file done !")

    def project(self, coverage):
        """Project the current CTTH channel along the *coverage*
        """
        dest_area = coverage.out_area
        dest_area_id = dest_area.area_id

        retv = MsgCTTH()

        retv.temperature = coverage.project_array(self.temperature)
        retv.height = coverage.project_array(self.height)
        retv.pressure = coverage.project_array(self.pressure)
        retv.cloudiness = coverage.project_array(self.cloudiness)
        retv.processing_flags = \
            coverage.project_array(self.processing_flags)

        retv.area = dest_area
        retv.region_name = dest_area_id
        retv.projection_name = dest_area.proj_id
        retv.num_of_columns = dest_area.x_size
        retv.num_of_lines = dest_area.y_size

        retv.shape = dest_area.shape

        retv.name = self.name
        retv.resolution = self.resolution
        retv.filled = True

        return retv

# ----------------------------------------


class MsgPCData(object):

    """NWCSAF/MSG Precipitating Clouds data layer
    """

    def __init__(self):
        self.data = None
        self.scaling_factor = 1
        self.offset = 0
        self.num_of_lines = 0
        self.num_of_columns = 0
        self.product = ""
        self.id = ""


class MsgPC(mpop.channel.GenericChannel):

    """NWCSAF/MSG Precipitating Clouds data structure as retrieved from HDF5
    file. Resolution sets the nominal resolution of the data.
    """

    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "PC")
        self.filled = False
        self.name = "PC"
        self.package = ""
        self.saf = ""
        self.product_name = ""
        self.num_of_columns = 0
        self.num_of_lines = 0
        self.projection_name = ""
        self.pcs_def = ""
        self.xscale = 0
        self.yscale = 0
        self.ll_lon = 0.0
        self.ll_lat = 0.0
        self.ur_lon = 0.0
        self.ur_lat = 0.0
        self.region_name = ""
        self.cfac = 0
        self.lfac = 0
        self.coff = 0
        self.loff = 0
        self.nb_param = 0
        self.gp_sc_id = 0
        self.image_acquisition_time = 0
        self.spectral_channel_id = 0
        self.nominal_product_time = 0
        self.sgs_product_quality = 0
        self.sgs_product_completeness = 0
        self.product_algorithm_version = ""
        self.probability_1 = None
        self.processing_flags = None
        self.shape = None
        self.satid = ""
        self.qc_straylight = -1

    def __str__(self):
        return ("'%s: shape %s, resolution %sm'" %
                (self.name,
                 self.probability_1.shape,
                 self.resolution))

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

# ------------------------------------------------------------------
    def read(self, filename, calibrate=True):
        """Reader for the NWCSAF/MSG precipitating clouds. Use *filename* to read data.
        """
        import h5py

        self.probability_1 = MsgPCData()
        self.processing_flags = MsgPCData()

        h5f = h5py.File(filename, 'r')
        # pylint: disable-msg=W0212
        self.package = h5f.attrs["PACKAGE"]
        self.saf = h5f.attrs["SAF"]
        self.product_name = h5f.attrs["PRODUCT_NAME"]
        self.num_of_columns = h5f.attrs["NC"]
        self.num_of_lines = h5f.attrs["NL"]
        self.projection_name = h5f.attrs["PROJECTION_NAME"]
        self.region_name = h5f.attrs["REGION_NAME"]
        self.cfac = h5f.attrs["CFAC"]
        self.lfac = h5f.attrs["LFAC"]
        self.coff = h5f.attrs["COFF"]
        self.loff = h5f.attrs["LOFF"]
        self.nb_param = h5f.attrs["NB_PARAMETERS"]
        self.gp_sc_id = h5f.attrs["GP_SC_ID"]
        self.image_acquisition_time = h5f.attrs["IMAGE_ACQUISITION_TIME"]
        self.spectral_channel_id = h5f.attrs["SPECTRAL_CHANNEL_ID"]
        self.nominal_product_time = h5f.attrs["NOMINAL_PRODUCT_TIME"]
        self.sgs_product_quality = h5f.attrs["SGS_PRODUCT_QUALITY"]
        self.sgs_product_completeness = h5f.attrs["SGS_PRODUCT_COMPLETENESS"]
        self.product_algorithm_version = h5f.attrs["PRODUCT_ALGORITHM_VERSION"]
        # pylint: enable-msg=W0212
        # ------------------------

        # The precipitating clouds data
        h5d = h5f['PC_PROB1']
        self.probability_1.data = h5d[:, :]
        self.probability_1.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.probability_1.offset = h5d.attrs["OFFSET"]
        self.probability_1.num_of_lines = h5d.attrs["N_LINES"]
        self.probability_1.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.probability_1.num_of_lines,
                      self.probability_1.num_of_columns)
        self.probability_1.product = h5d.attrs["PRODUCT"]
        self.probability_1.id = h5d.attrs["ID"]
        self.probability_1.data = np.ma.masked_equal(
            self.probability_1.data, 0)
        if calibrate:
            self.probability_1 = (self.probability_1.data *
                                  self.probability_1.scaling_factor +
                                  self.probability_1.offset)
        else:
            self.probability_1 = self.probability_1.data
        self.probability_1_palette = _get_palette(h5f, 'PC_PROB1')

        # ------------------------

        # The cloudtype processing/quality flags
        h5d = h5f['PC_QUALITY']
        self.processing_flags.data = h5d[:, :]
        self.processing_flags.scaling_factor = \
            h5d.attrs["SCALING_FACTOR"]
        self.processing_flags.offset = h5d.attrs["OFFSET"]
        self.processing_flags.num_of_lines = h5d.attrs["N_LINES"]
        self.processing_flags.num_of_columns = h5d.attrs["N_COLS"]
        self.processing_flags.product = h5d.attrs["PRODUCT"]
        self.processing_flags.id = h5d.attrs["ID"]
        self.processing_flags = np.ma.masked_equal(
            self.processing_flags.data, 0)

        # ------------------------
        h5f.close()

        self.area = get_area_from_file(filename)

        self.filled = True

# ------------------------------------------------------------------


def get_bit_from_flags(arr, nbit):
    """I don't know what this function does.
    """
    res = np.bitwise_and(np.right_shift(arr, nbit), 1)
    return res.astype('b')





# NEU Anfang NEW Beginn

# ----------------------------------------

class MsgCRRData(object):

    """NWCSAF/MSG Convective Rain Rate data layer
    """

    def __init__(self):
        self.data = None
        self.scaling_factor = 1
        self.offset = 0
        self.num_of_lines = 0
        self.num_of_columns = 0
        self.product = ""
        self.id = ""


class MsgCRR(mpop.channel.GenericChannel):

    """NWCSAF/MSG Convective Rain Rate data structure as retrieved from HDF5
    file. Resolution sets the nominal resolution of the data.
    """

    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "CRR")
        self.filled = False
        self.name = "CRR"
#       self.resolution = resolution
        self.package = ""
        self.saf = ""
        self.product_name = ""
        self.num_of_columns = 0
        self.num_of_lines = 0
        self.projection_name = ""
        self.pcs_def = ""
        self.xscale = 0
        self.yscale = 0
        self.ll_lon = 0.0
        self.ll_lat = 0.0
        self.ur_lon = 0.0
        self.ur_lat = 0.0
        self.region_name = ""
        self.cfac = 0
        self.lfac = 0
        self.coff = 0
        self.loff = 0
        self.nb_param = 0
        self.gp_sc_id = 0
        self.image_acquisition_time = 0
        self.spectral_channel_id = 0
        self.nominal_product_time = 0
        self.sgs_product_quality = 0
        self.sgs_product_completeness = 0
        self.product_algorithm_version = ""
        self.crr = None
        self.crr_accum = None
        self.crr_intensity = None
        self.crr_quality = None
        self.crr_dataflag = None
        self.processing_flags = None
        self.shape = None
        self.satid = ""
        self.qc_straylight = -1
        self.crr_palette = None
        self.crr_accum_palette = None
        self.crr_intensity_palette = None
        self.crr_quality_palette = None

    def __str__(self):
        return ("'%s: shape %s, resolution %sm'" %
                (self.name,
                 self.crr.shape,
                 self.resolution))

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

# ------------------------------------------------------------------
    def read(self, filename, calibrate=True):
        """Reader for the . Use *filename* to read data.
        """
        import h5py

        self.crr = MsgCRRData()
        self.crr_accum = MsgCRRData()
        self.crr_intensity = MsgCRRData()
        self.crr_quality = MsgCRRData()
        self.processing_flags = MsgCRRData()

        LOG.debug("Filename = <" + str(filename) + ">")
        h5f = h5py.File(filename, 'r')
        # pylint: disable-msg=W0212
        self.package = h5f.attrs["PACKAGE"]
        self.saf = h5f.attrs["SAF"]
        self.product_name = h5f.attrs["PRODUCT_NAME"]
        self.num_of_columns = h5f.attrs["NC"]
        self.num_of_lines = h5f.attrs["NL"]
        self.projection_name = h5f.attrs["PROJECTION_NAME"]
        self.region_name = h5f.attrs["REGION_NAME"]
        self.cfac = h5f.attrs["CFAC"]
        self.lfac = h5f.attrs["LFAC"]
        self.coff = h5f.attrs["COFF"]
        self.loff = h5f.attrs["LOFF"]
        self.nb_param = h5f.attrs["NB_PARAMETERS"]
        self.gp_sc_id = h5f.attrs["GP_SC_ID"]
        self.image_acquisition_time = h5f.attrs["IMAGE_ACQUISITION_TIME"]
        self.spectral_channel_id = h5f.attrs["SPECTRAL_CHANNEL_ID"]
        self.nominal_product_time = h5f.attrs["NOMINAL_PRODUCT_TIME"]
        self.sgs_product_quality = h5f.attrs["SGS_PRODUCT_QUALITY"]
        self.sgs_product_completeness = h5f.attrs["SGS_PRODUCT_COMPLETENESS"]
        self.product_algorithm_version = h5f.attrs["PRODUCT_ALGORITHM_VERSION"]
        # pylint: enable-msg=W0212
        # ------------------------

        # The CRR data
        h5d = h5f['CRR']
        self.crr.data = h5d[:, :]
        self.crr.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.crr.offset = h5d.attrs["OFFSET"]
        self.crr.num_of_lines = h5d.attrs["N_LINES"]
        self.crr.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.crr.num_of_lines,
                      self.crr.num_of_columns)
        self.crr.product = h5d.attrs["PRODUCT"]
        self.crr.id = h5d.attrs["ID"]
        if calibrate:
            self.crr = (self.crr.data *
                                  self.crr.scaling_factor +
                                  self.crr.offset)
        else:
            self.crr = self.crr.data
        self.crr_palette = _get_palette(h5f, 'CRR') / 255.0

        # ------------------------

        # The CRR ACCUM data
        h5d = h5f['CRR_ACCUM']
        self.crr_accum.data = h5d[:, :]
        self.crr_accum.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.crr_accum.offset = h5d.attrs["OFFSET"]
        self.crr_accum.num_of_lines = h5d.attrs["N_LINES"]
        self.crr_accum.num_of_columns = h5d.attrs["N_COLS"]
        self.crr_accum.product = h5d.attrs["PRODUCT"]
        self.crr_accum.id = h5d.attrs["ID"]
        if calibrate:
            self.crr_accum = (self.crr_accum.data *
                                  self.crr_accum.scaling_factor +
                                  self.crr_accum.offset)
        else:
            self.crr_accum = self.crr_accum.data
        self.crr_accum_palette = _get_palette(h5f, 'CRR_ACCUM') / 255.0

        # ------------------------

        # The CRR Intensity data
        h5d = h5f['CRR_INTENSITY']
        self.crr_intensity.data = h5d[:, :]
        self.crr_intensity.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.crr_intensity.offset = h5d.attrs["OFFSET"]
        self.crr_intensity.num_of_lines = h5d.attrs["N_LINES"]
        self.crr_intensity.num_of_columns = h5d.attrs["N_COLS"]
        self.crr_intensity.product = h5d.attrs["PRODUCT"]
        self.crr_intensity.id = h5d.attrs["ID"]
        if calibrate:
            self.crr_intensity = (self.crr_intensity.data *
                                  self.crr_intensity.scaling_factor +
                                  self.crr_intensity.offset)
        else:
            self.crr_intensity = self.crr_intensity.data
        self.crr_intensity_palette = _get_palette(h5f, 'CRR_INTENSITY') / 255.0

        # ------------------------

        # The CRR quality data
        h5d = h5f['CRR_QUALITY']
        self.crr_quality.data = h5d[:, :]
        self.crr_quality.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.crr_quality.offset = h5d.attrs["OFFSET"]
        self.crr_quality.num_of_lines = h5d.attrs["N_LINES"]
        self.crr_quality.num_of_columns = h5d.attrs["N_COLS"]
        self.crr_quality.product = h5d.attrs["PRODUCT"]
        self.crr_quality.id = h5d.attrs["ID"]
        if calibrate:
            self.crr_quality = (self.crr_quality.data *
                                  self.crr_quality.scaling_factor +
                                  self.crr_quality.offset)
        else:
            self.crr_quality = self.crr_quality.data
        self.crr_quality_palette = _get_palette(h5f, 'CRR_QUALITY')
        # ------------------------

        h5f.close()

        #self.crr = self.crr.data
        #self.crr_accum = self.crr_accum.data
        #self.crr_intensity = self.crr_intensity.data
        #self.crr_quality = self.crr_quality.data
        #self.processing_flags = self.processing_flags.data

        self.area = get_area_from_file(filename)

        self.filled = True

    def save(self, filename):
        """Save the current cloudtype object to hdf *filename*, in pps format.
        """
        import h5py
        ctype = self.convert2pps()
        LOG.info("Saving CRR hdf file...")
        ctype.save(filename)
        h5f = h5py.File(filename, mode="a")
        h5f.attrs["straylight_contaminated"] = self.qc_straylight
        h5f.close()
        LOG.info("Saving CRR hdf file done !")

    def project(self, coverage):
        """Remaps the NWCSAF/MSG CRR to cartographic map-projection on
        area give by a pre-registered area-id. Faster version of msg_remap!
        """
        LOG.info("Projecting channel %s..." % (self.name))

        region = coverage.out_area
        dest_area = region.area_id

        retv = MsgCRR()

        retv.name = self.name
        retv.package = self.package
        retv.saf = self.saf
        retv.product_name = self.product_name
        retv.region_name = dest_area
        retv.cfac = self.cfac
        retv.lfac = self.lfac
        retv.coff = self.coff
        retv.loff = self.loff
        retv.nb_param = self.nb_param
        retv.gp_sc_id = self.gp_sc_id
        retv.image_acquisition_time = self.image_acquisition_time
        retv.spectral_channel_id = self.spectral_channel_id
        retv.nominal_product_time = self.nominal_product_time
        retv.sgs_product_quality = self.sgs_product_quality
        retv.sgs_product_completeness = self.sgs_product_completeness
        retv.product_algorithm_version = self.product_algorithm_version

        retv.crr = coverage.project_array(self.crr)
        retv.crr_palette = self.crr_palette

        retv.crr_accum = coverage.project_array(self.crr_accum)
        retv.crr_accum_palette = self.crr_accum_palette

        retv.crr_intensity = coverage.project_array(self.crr_intensity)
        retv.crr_intensity_palette = self.crr_intensity_palette

        retv.crr_quality = coverage.project_array(self.crr_quality)
        retv.crr_quality_palette = self.crr_quality_palette

        retv.processing_flags = \
            coverage.project_array(self.processing_flags)

        retv.qc_straylight = self.qc_straylight
        retv.region_name = dest_area
        retv.area = region
        retv.projection_name = region.proj_id

        retv.pcs_def = pcs_def_from_region(region)

        retv.num_of_columns = region.x_size
        retv.num_of_lines = region.y_size
        retv.xscale = region.pixel_size_x
        retv.yscale = region.pixel_size_y

        import pyproj
        prj = pyproj.Proj(region.proj4_string)
        aex = region.area_extent
        lonur, latur = prj(aex[2], aex[3], inverse=True)
        lonll, latll = prj(aex[0], aex[1], inverse=True)
        retv.ll_lon = lonll
        retv.ll_lat = latll
        retv.ur_lon = lonur
        retv.ur_lat = latur

        self.shape = region.shape

        retv.filled = True
        retv.resolution = self.resolution

        return retv


#    def convert2nordrad(self):
#        return NordRadCType(self)

class MsgSPhRData(object):

    """NWCSAF/MSG SPhR data layer
    """

    def __init__(self):
        self.data = None
        self.scaling_factor = 1
        self.offset = 0
        self.num_of_lines = 0
        self.num_of_columns = 0
        self.product = ""
        self.id = ""


class MsgSPhR(mpop.channel.GenericChannel):

    """NWCSAF/MSG SPhR data structure as retrieved from HDF5
    file. Resolution sets the nominal resolution of the data.
    Palette now missing
    """

    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "SPhR")
        self.filled = False
        self.name = "SPhR"
#       self.resolution = resolution
        self.package = ""
        self.saf = ""
        self.product_name = ""
        self.num_of_columns = 0
        self.num_of_lines = 0
        self.projection_name = ""
        self.pcs_def = ""
        self.xscale = 0
        self.yscale = 0
        self.ll_lon = 0.0
        self.ll_lat = 0.0
        self.ur_lon = 0.0
        self.ur_lat = 0.0
        self.region_name = ""
        self.cfac = 0
        self.lfac = 0
        self.coff = 0
        self.loff = 0
        self.nb_param = 0
        self.gp_sc_id = 0
        self.image_acquisition_time = 0
        self.spectral_channel_id = 0
        self.nominal_product_time = 0
        self.sgs_product_quality = 0
        self.sgs_product_completeness = 0
        self.product_algorithm_version = ""
        self.sphr = None
        self.sphr_bl = None
        self.sphr_cape = None
        self.sphr_diffbl = None
        self.sphr_diffhl = None
        self.sphr_diffki = None
        self.sphr_diffli = None
        self.sphr_diffml = None
        self.sphr_diffshw = None
        self.sphr_difftpw = None
        self.sphr_hl = None
        self.sphr_ki = None
        self.sphr_li = None
        self.sphr_ml = None
        self.sphr_quality = None
        self.sphr_sflag = None
        self.sphr_shw = None
        self.sphr_tpw = None
        self.processing_flags = None
        self.shape = None
        self.satid = ""
        self.qc_straylight = -1
        self.sphr = None
        self.sphr_bl_palette = None
        self.sphr_cape_palette = None
        self.sphr_diffbl_palette = None
        self.sphr_diffhl_palette = None
        self.sphr_diffki_palette = None
        self.sphr_diffli_palette = None
        self.sphr_diffml_palette = None
        self.sphr_diffshw_palette = None
        self.sphr_difftpw_palette = None
        self.sphr_hl_palette = None
        self.sphr_ki_palette = None
        self.sphr_li_palette = None
        self.sphr_ml_palette = None
        self.sphr_quality_palette = None
        self.sphr_sflag_palette = None
        self.sphr_shw_palette = None
        self.sphr_tpw_palette = None
        
    def __str__(self):
        return ("'%s: shape %s, resolution %sm'" %
                (self.name,
                 self.sphr_bl.shape,
                 self.resolution))

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

# ------------------------------------------------------------------
    def read(self, filename, calibrate=True):
        """Reader for the . Use *filename* to read data.
        """
        import h5py

# Erste Zeile notwendig?
        self.sphr = MsgSPhRData()
        self.sphr_bl = MsgSPhRData()
        self.sphr_cape = MsgSPhRData()
        self.sphr_diffbl = MsgSPhRData()
        self.sphr_diffhl = MsgSPhRData()
        self.sphr_diffki = MsgSPhRData()
        self.sphr_diffli = MsgSPhRData()
        self.sphr_diffml = MsgSPhRData()
        self.sphr_diffshw = MsgSPhRData()
        self.sphr_difftpw = MsgSPhRData()
        self.sphr_hl = MsgSPhRData()
        self.sphr_ki = MsgSPhRData()
        self.sphr_li = MsgSPhRData()
        self.sphr_ml = MsgSPhRData()
        self.sphr_quality = MsgSPhRData()
        self.sphr_sflag = MsgSPhRData()
        self.sphr_shw = MsgSPhRData()
        self.sphr_tpw = MsgSPhRData()

        self.processing_flags = MsgSPhRData()

        LOG.debug("Filename = <" + str(filename) + ">")
        h5f = h5py.File(filename, 'r')
        # pylint: disable-msg=W0212
        self.package = h5f.attrs["PACKAGE"]
        self.saf = h5f.attrs["SAF"]
        self.product_name = h5f.attrs["PRODUCT_NAME"]
        self.num_of_columns = h5f.attrs["NC"]
        self.num_of_lines = h5f.attrs["NL"]
        self.projection_name = h5f.attrs["PROJECTION_NAME"]
        self.region_name = h5f.attrs["REGION_NAME"]
        self.cfac = h5f.attrs["CFAC"]
        self.lfac = h5f.attrs["LFAC"]
        self.coff = h5f.attrs["COFF"]
        self.loff = h5f.attrs["LOFF"]
        self.nb_param = h5f.attrs["NB_PARAMETERS"]
        self.gp_sc_id = h5f.attrs["GP_SC_ID"]
        self.image_acquisition_time = h5f.attrs["IMAGE_ACQUISITION_TIME"]
        self.spectral_channel_id = h5f.attrs["SPECTRAL_CHANNEL_ID"]
        self.nominal_product_time = h5f.attrs["NOMINAL_PRODUCT_TIME"]
        self.sgs_product_quality = h5f.attrs["SGS_PRODUCT_QUALITY"]
        self.sgs_product_completeness = h5f.attrs["SGS_PRODUCT_COMPLETENESS"]
        self.product_algorithm_version = h5f.attrs["PRODUCT_ALGORITHM_VERSION"]
        # pylint: enable-msg=W0212
        # ------------------------

        # The SPhR BL data
        h5d = h5f['SPhR_BL']
        self.sphr_bl.data = h5d[:, :]
        self.sphr_bl.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_bl.offset = h5d.attrs["OFFSET"]
        self.sphr_bl.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_bl.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_bl.num_of_lines,
                      self.sphr_bl.num_of_columns)
        self.sphr_bl.product = h5d.attrs["PRODUCT"]
        self.sphr_bl.id = h5d.attrs["ID"]
#        self.sphr_bl.data = self.sphr_bl.raw * self.sphr_bl.scaling_factor  + self.sphr_bl.offset
        if calibrate:
            self.sphr_bl = (self.sphr_bl.data *
                                  self.sphr_bl.scaling_factor +
                                  self.sphr_bl.offset)
        else:
            self.sphr_bl = self.sphr_bl.data
        self.sphr_bl_palette = _get_palette(h5f, 'SPhR_BL') / 255.0

        # The SPhR Cape data
        h5d = h5f['SPhR_CAPE']
        self.sphr_cape.data = h5d[:, :]
        self.sphr_cape.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_cape.offset = h5d.attrs["OFFSET"]
        self.sphr_cape.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_cape.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_cape.num_of_lines,
                      self.sphr_cape.num_of_columns)
        self.sphr_cape.product = h5d.attrs["PRODUCT"]

        self.sphr_cape.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_cape = (self.sphr_cape.data *
                                  self.sphr_cape.scaling_factor +
                                  self.sphr_cape.offset)
        else:
            self.sphr = self.sphr.data
        #self.sphr_cape_palette = _get_palette(h5f, 'SPhR_CAPE') / 255.0

        # The SPhR DIFFBL data
        h5d = h5f['SPhR_DIFFBL']
        self.sphr_diffbl.data = h5d[:, :]
        self.sphr_diffbl.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_diffbl.offset = h5d.attrs["OFFSET"]
        self.sphr_diffbl.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_diffbl.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_diffbl.num_of_lines,
                      self.sphr_diffbl.num_of_columns)
        self.sphr_diffbl.product = h5d.attrs["PRODUCT"]
        self.sphr_diffbl.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_diffbl = (self.sphr_diffbl.data *
                                  self.sphr_diffbl.scaling_factor +
                                  self.sphr_diffbl.offset)
        else:
            self.sphr_diffbl = self.sphr_diffbl.data
        self.sphr_diffbl_palette = _get_palette(h5f, 'SPhR_DIFFBL') / 255.0

        # The SPhR DIFFHL data
        h5d = h5f['SPhR_DIFFHL']
        self.sphr_diffhl.data = h5d[:, :]
        self.sphr_diffhl.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_diffhl.offset = h5d.attrs["OFFSET"]
        self.sphr_diffhl.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_diffhl.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_diffhl.num_of_lines,
                      self.sphr_diffhl.num_of_columns)
        self.sphr_diffhl.product = h5d.attrs["PRODUCT"]
        self.sphr_diffhl.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_diffhl = (self.sphr_diffhl.data *
                                  self.sphr_diffhl.scaling_factor +
                                  self.sphr_diffhl.offset)
        else:
            self.sphr_diffhl = self.sphr_diffhl.data
        self.sphr_diffhl_palette = _get_palette(h5f, 'SPhR_DIFFHL') / 255.0

        # The SPhR DIFFKI data
        h5d = h5f['SPhR_DIFFKI']
        self.sphr_diffki.data = h5d[:, :]
        self.sphr_diffki.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_diffki.offset = h5d.attrs["OFFSET"]
        self.sphr_diffki.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_diffki.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_diffki.num_of_lines,
                      self.sphr_diffki.num_of_columns)
        self.sphr_diffki.product = h5d.attrs["PRODUCT"]
        self.sphr_diffki.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_diffki = (self.sphr_diffki.data *
                                  self.sphr_diffki.scaling_factor +
                                  self.sphr_diffki.offset)
        else:
            self.sphr_diffki = self.sphr_diffki.data
        self.sphr_diffki_palette = _get_palette(h5f, 'SPhR_DIFFKI') / 255.0

        # The SPhR DIFFLI data
        h5d = h5f['SPhR_DIFFLI']
        self.sphr_diffli.data = h5d[:, :]
        self.sphr_diffli.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_diffli.offset = h5d.attrs["OFFSET"]
        self.sphr_diffli.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_diffli.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_diffli.num_of_lines,
                      self.sphr_diffli.num_of_columns)
        self.sphr_diffli.product = h5d.attrs["PRODUCT"]
        self.sphr_diffli.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_diffli = (self.sphr_diffli.data *
                                  self.sphr_diffli.scaling_factor +
                                  self.sphr_diffli.offset)
        else:
            self.sphr_diffli= self.sphr_diffli.data
        self.sphr_diffli_palette = _get_palette(h5f, 'SPhR_DIFFLI') / 255.0

        # The SPhR DIFFML data
        h5d = h5f['SPhR_DIFFML']
        self.sphr_diffml.data = h5d[:, :]
        self.sphr_diffml.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_diffml.offset = h5d.attrs["OFFSET"]
        self.sphr_diffml.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_diffml.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_diffml.num_of_lines,
                      self.sphr_diffml.num_of_columns)
        self.sphr_diffml.product = h5d.attrs["PRODUCT"]
        self.sphr_diffml.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_diffml = (self.sphr_diffml.data *
                                  self.sphr_diffml.scaling_factor +
                                  self.sphr_diffml.offset)
        else:
            self.sphr_diffml = self.sphr_diffml.data
        self.sphr_diffml_palette = _get_palette(h5f, 'SPhR_DIFFML') / 255.0

        # The SPhR DIFFSHW data
        h5d = h5f['SPhR_DIFFSHW']
        self.sphr_diffshw.data = h5d[:, :]
        self.sphr_diffshw.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_diffshw.offset = h5d.attrs["OFFSET"]
        self.sphr_diffshw.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_diffshw.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_diffshw.num_of_lines,
                      self.sphr_diffshw.num_of_columns)
        self.sphr_diffshw.product = h5d.attrs["PRODUCT"]
        self.sphr_diffshw.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_diffshw = (self.sphr_diffshw.data *
                                  self.sphr_diffshw.scaling_factor +
                                  self.sphr_diffshw.offset)
        else:
            self.sphr_diffshw = self.sphr_diffshw.data
        self.sphr_diffshw_palette = _get_palette(h5f, 'SPhR_DIFFSHW') / 255.0

        # The SPhR DIFFTPW data
        h5d = h5f['SPhR_DIFFTPW']
        self.sphr_difftpw.data = h5d[:, :]
        self.sphr_difftpw.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_difftpw.offset = h5d.attrs["OFFSET"]
        self.sphr_difftpw.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_difftpw.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_difftpw.num_of_lines,
                      self.sphr_difftpw.num_of_columns)
        self.sphr_difftpw.product = h5d.attrs["PRODUCT"]
        self.sphr_difftpw.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_difftpw = (self.sphr_difftpw.data *
                                  self.sphr_difftpw.scaling_factor +
                                  self.sphr_difftpw.offset)
        else:
            self.sphr_difftpw = self.sphr_difftpw.data
        self.sphr_difftpw_palette = _get_palette(h5f, 'SPhR_DIFFTPW') / 255.0

        # The SPhR HL data
        h5d = h5f['SPhR_HL']
        self.sphr_hl.data = h5d[:, :]
        self.sphr_hl.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_hl.offset = h5d.attrs["OFFSET"]
        self.sphr_hl.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_hl.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_hl.num_of_lines,
                      self.sphr_hl.num_of_columns)
        self.sphr_hl.product = h5d.attrs["PRODUCT"]
        self.sphr_hl.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_hl = (self.sphr_hl.data *
                                  self.sphr_hl.scaling_factor +
                                  self.sphr_hl.offset)
        else:
            self.sphr_hl = self.sphr_hl.data
        self.sphr_hl_palette = _get_palette(h5f, 'SPhR_HL') / 255.0

        # The SPhR KI data
        h5d = h5f['SPhR_KI']
        self.sphr_ki.data = h5d[:, :]
        self.sphr_ki.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_ki.offset = h5d.attrs["OFFSET"]
        self.sphr_ki.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_ki.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_ki.num_of_lines,
                      self.sphr_ki.num_of_columns)
        self.sphr_ki.product = h5d.attrs["PRODUCT"]
        self.sphr_ki.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_ki = (self.sphr_ki.data *
                                  self.sphr_ki.scaling_factor +
                                  self.sphr_ki.offset)
        else:
            self.sphr_ki = self.sphr_ki.data
        self.sphr_ki_palette = _get_palette(h5f, 'SPhR_KI') / 255.0

        # The SPhR LI data
        h5d = h5f['SPhR_LI']
        self.sphr_li.data = h5d[:, :]
        self.sphr_li.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_li.offset = h5d.attrs["OFFSET"]
        self.sphr_li.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_li.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_li.num_of_lines,
                      self.sphr_li.num_of_columns)
        self.sphr_li.product = h5d.attrs["PRODUCT"]
        self.sphr_li.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_li = (self.sphr_li.data *
                                  self.sphr_li.scaling_factor +
                                  self.sphr_li.offset)
        else:
            self.sphr_li = self.sphr_li.data
        self.sphr_li_palette = _get_palette(h5f, 'SPhR_LI') / 255.0

        # The SPhR ML data
        h5d = h5f['SPhR_ML']
        self.sphr_ml.data = h5d[:, :]
        self.sphr_ml.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_ml.offset = h5d.attrs["OFFSET"]
        self.sphr_ml.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_ml.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_ml.num_of_lines,
                      self.sphr_ml.num_of_columns)
        self.sphr_ml.product = h5d.attrs["PRODUCT"]
        self.sphr_ml.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_ml = (self.sphr_ml.data *
                                  self.sphr_ml.scaling_factor +
                                  self.sphr_ml.offset)
        else:
            self.sphr_ml = self.sphr_ml.data
        self.sphr_ml_palette = _get_palette(h5f, 'SPhR_ML') / 255.0

        # The SPhR QUALITY data
        h5d = h5f['SPhR_QUALITY']
        self.sphr_quality.data = h5d[:, :]
        self.sphr_quality.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_quality.offset = h5d.attrs["OFFSET"]
        self.sphr_quality.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_quality.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_quality.num_of_lines,
                      self.sphr_quality.num_of_columns)
        self.sphr_quality.product = h5d.attrs["PRODUCT"]
        self.sphr_quality.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_quality = (self.sphr_quality.data *
                                  self.sphr_quality.scaling_factor +
                                  self.sphr_quality.offset)
        else:
            self.sphr_quality = self.sphr_quality.data

        # The SPhR SFLAG data
        h5d = h5f['SPhR_SFLAG']
        self.sphr_sflag.data = h5d[:, :]
        self.sphr_sflag.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_sflag.offset = h5d.attrs["OFFSET"]
        self.sphr_sflag.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_sflag.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_sflag.num_of_lines,
                      self.sphr_sflag.num_of_columns)
        self.sphr_sflag.product = h5d.attrs["PRODUCT"]
        self.sphr_sflag.id = h5d.attrs["ID"]

        # The SPhR SHW data
        h5d = h5f['SPhR_SHW']
        self.sphr_shw.data = h5d[:, :]
        self.sphr_shw.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_shw.offset = h5d.attrs["OFFSET"]
        self.sphr_shw.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_shw.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_shw.num_of_lines,
                      self.sphr_shw.num_of_columns)
        self.sphr_shw.product = h5d.attrs["PRODUCT"]
        self.sphr_shw.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_shw = (self.sphr_shw.data *
                                  self.sphr_shw.scaling_factor +
                                  self.sphr_shw.offset)
        else:
            self.sphr_shw = self.sphr_shw.data
        self.sphr_shw_palette = _get_palette(h5f, 'SPhR_SHW') / 255.0

        # The SPhR TPW data
        h5d = h5f['SPhR_TPW']
        self.sphr_tpw.data = h5d[:, :]
        self.sphr_tpw.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.sphr_tpw.offset = h5d.attrs["OFFSET"]
        self.sphr_tpw.num_of_lines = h5d.attrs["N_LINES"]
        self.sphr_tpw.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.sphr_tpw.num_of_lines,
                      self.sphr_tpw.num_of_columns)
        self.sphr_tpw.product = h5d.attrs["PRODUCT"]
        self.sphr_tpw.id = h5d.attrs["ID"]
        if calibrate:
            self.sphr_tpw = (self.sphr_tpw.data *
                                  self.sphr_tpw.scaling_factor +
                                  self.sphr_tpw.offset)
        else:
            self.sphr_tpw = self.sphr_tpw.data
        self.sphr_tpw_palette = _get_palette(h5f, 'SPhR_TPW') / 255.0

        # ------------------------

        h5f.close()

        self.processing_flags = self.processing_flags.data

        self.area = get_area_from_file(filename)

        self.filled = True

    def project(self, coverage):
        """Remaps the NWCSAF/MSG SPhR to cartographic map-projection on
        area give by a pre-registered area-id. Faster version of msg_remap!
        """
        LOG.info("Projecting channel %s..." % (self.name))

        region = coverage.out_area
        dest_area = region.area_id

        retv = MsgSPhR()

        retv.name = self.name
        retv.package = self.package
        retv.saf = self.saf
        retv.product_name = self.product_name
        retv.region_name = dest_area
        retv.cfac = self.cfac
        retv.lfac = self.lfac
        retv.coff = self.coff
        retv.loff = self.loff
        retv.nb_param = self.nb_param
        retv.gp_sc_id = self.gp_sc_id
        retv.image_acquisition_time = self.image_acquisition_time
        retv.spectral_channel_id = self.spectral_channel_id
        retv.nominal_product_time = self.nominal_product_time
        retv.sgs_product_quality = self.sgs_product_quality
        retv.sgs_product_completeness = self.sgs_product_completeness
        retv.product_algorithm_version = self.product_algorithm_version

        retv.sphr_bl = coverage.project_array(self.crr)
        retv.sphr_bl = self.sphr_bl_palette

        retv.processing_flags = \
            coverage.project_array(self.processing_flags)

        retv.qc_straylight = self.qc_straylight
        retv.region_name = dest_area
        retv.area = region
        retv.projection_name = region.proj_id

        retv.pcs_def = pcs_def_from_region(region)

        retv.num_of_columns = region.x_size
        retv.num_of_lines = region.y_size
        retv.xscale = region.pixel_size_x
        retv.yscale = region.pixel_size_y

        import pyproj
        prj = pyproj.Proj(region.proj4_string)
        aex = region.area_extent
        lonur, latur = prj(aex[2], aex[3], inverse=True)
        lonll, latll = prj(aex[0], aex[1], inverse=True)
        retv.ll_lon = lonll
        retv.ll_lat = latll
        retv.ur_lon = lonur
        retv.ur_lat = latur

        self.shape = region.shape

        retv.filled = True
        retv.resolution = self.resolution

        return retv


# PGE 14

class MsgPCPhData(object):

    """NWCSAF/MSG PCPh layer
    """

    def __init__(self):
        self.data = None
        self.scaling_factor = 1
        self.offset = 0
        self.num_of_lines = 0
        self.num_of_columns = 0
        self.product = ""
        self.id = ""


class MsgPCPh(mpop.channel.GenericChannel):

    """NWCSAF/MSG PCPh data structure as retrieved from HDF5
    file. Resolution sets the nominal resolution of the data.
    Palette now missing
    """

    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "PCPh")
        self.filled = False
        self.name = "PCPh"
#       self.resolution = resolution
        self.package = ""
        self.saf = ""
        self.product_name = ""
        self.num_of_columns = 0
        self.num_of_lines = 0
        self.projection_name = ""
        self.pcs_def = ""
        self.xscale = 0
        self.yscale = 0
        self.ll_lon = 0.0
        self.ll_lat = 0.0
        self.ur_lon = 0.0
        self.ur_lat = 0.0
        self.region_name = ""
        self.cfac = 0
        self.lfac = 0
        self.coff = 0
        self.loff = 0
        self.nb_param = 0
        self.gp_sc_id = 0
        self.image_acquisition_time = 0
        self.spectral_channel_id = 0
        self.nominal_product_time = 0
        self.sgs_product_quality = 0
        self.sgs_product_completeness = 0
        self.product_algorithm_version = ""
        self.pcph = None
        self.pcph_pc = None
        self.pcph_quality = None
        self.pcph_dataflag = None
        self.processing_flags = None
        self.shape = None
        self.satid = ""
        self.qc_straylight = -1
        self.pcph = None
        self.pcph_pc_palette = None
        self.pcph_quality_palette = None
        self.pcph_sflag_palette = None
        
    def __str__(self):
        return ("'%s: shape %s, resolution %sm'" %
                (self.name,
                 self.pcph_pc.shape,
                 self.resolution))

    def is_loaded(self): 
        """Tells if the channel contains loaded data.
        """
        return self.filled

# ------------------------------------------------------------------
    def read(self, filename, calibrate=True):
        """Reader for the . Use *filename* to read data.
        """
        import h5py

# Erste Zeile notwendig?
        self.pcph = MsgPCPhData()
        self.pcph_pc = MsgPCPhData()
        self.pcph_quality = MsgPCPhData()
        self.pcph_dataflag = MsgPCPhData()

        self.processing_flags = MsgPCPhData()

        LOG.debug("Filename = <" + str(filename) + ">")
        h5f = h5py.File(filename, 'r')
        # pylint: disable-msg=W0212
        self.package = h5f.attrs["PACKAGE"]
        self.saf = h5f.attrs["SAF"]
        self.product_name = h5f.attrs["PRODUCT_NAME"]
        self.num_of_columns = h5f.attrs["NC"]
        self.num_of_lines = h5f.attrs["NL"]
        self.projection_name = h5f.attrs["PROJECTION_NAME"]
        self.region_name = h5f.attrs["REGION_NAME"]
        self.cfac = h5f.attrs["CFAC"]
        self.lfac = h5f.attrs["LFAC"]
        self.coff = h5f.attrs["COFF"]
        self.loff = h5f.attrs["LOFF"]
        self.nb_param = h5f.attrs["NB_PARAMETERS"]
        self.gp_sc_id = h5f.attrs["GP_SC_ID"]
        self.image_acquisition_time = h5f.attrs["IMAGE_ACQUISITION_TIME"]
        self.spectral_channel_id = h5f.attrs["SPECTRAL_CHANNEL_ID"]
        self.nominal_product_time = h5f.attrs["NOMINAL_PRODUCT_TIME"]
        self.sgs_product_quality = h5f.attrs["SGS_PRODUCT_QUALITY"]
        self.sgs_product_completeness = h5f.attrs["SGS_PRODUCT_COMPLETENESS"]
        self.product_algorithm_version = h5f.attrs["PRODUCT_ALGORITHM_VERSION"]
        # pylint: enable-msg=W0212
        # ------------------------

        # The PPh PC data
        h5d = h5f['PCPh_PC']
        self.pcph_pc.data = h5d[:, :]
        self.pcph_pc.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.pcph_pc.offset = h5d.attrs["OFFSET"]
        self.pcph_pc.num_of_lines = h5d.attrs["N_LINES"]
        self.pcph_pc.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.pcph_pc.num_of_lines,
                      self.pcph_pc.num_of_columns)
        self.pcph_pc.product = h5d.attrs["PRODUCT"]

        self.pcph_pc.id = h5d.attrs["ID"]
        if calibrate:
            self.pcph_pc = (self.pcph_pc.data *
                                  self.pcph_pc.scaling_factor +
                                  self.pcph_pc.offset)
        else:
            self.pcph_pc = self.pcph_pc.data
        self.pcph_pc_palette = _get_palette(h5f, 'PCPh_PC') / 255.0

        # The PPh QUALITY data
        h5d = h5f['PCPh_QUALITY']
        self.pcph_quality.data = h5d[:, :]
        self.pcph_quality.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.pcph_quality.offset = h5d.attrs["OFFSET"]
        self.pcph_quality.num_of_lines = h5d.attrs["N_LINES"]
        self.pcph_quality.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.pcph_quality.num_of_lines,
                      self.pcph_quality.num_of_columns)
        self.pcph_quality.product = h5d.attrs["PRODUCT"]
        self.pcph_quality.id = h5d.attrs["ID"]

        # The PPh DATA FLAG data
        h5d = h5f['PCPh_DATAFLAG']
        self.pcph_dataflag.data = h5d[:, :]
        self.pcph_dataflag.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.pcph_dataflag.offset = h5d.attrs["OFFSET"]
        self.pcph_dataflag.num_of_lines = h5d.attrs["N_LINES"]
        self.pcph_dataflag.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.pcph_dataflag.num_of_lines,
                      self.pcph_dataflag.num_of_columns)
        self.pcph_dataflag.product = h5d.attrs["PRODUCT"]
        self.pcph_dataflag.id = h5d.attrs["ID"]

        # ------------------------

        h5f.close()

        self.processing_flags = self.processing_flags.data

        self.area = get_area_from_file(filename)

        self.filled = True

    def project(self, coverage):
        """Remaps the NWCSAF/MSG PCPh to cartographic map-projection on
        area give by a pre-registered area-id. Faster version of msg_remap!
        """
        LOG.info("Projecting channel %s..." % (self.name))

        region = coverage.out_area
        dest_area = region.area_id

        retv = MsgPCPh()

        retv.name = self.name
        retv.package = self.package
        retv.saf = self.saf
        retv.product_name = self.product_name
        retv.region_name = dest_area
        retv.cfac = self.cfac
        retv.lfac = self.lfac
        retv.coff = self.coff
        retv.loff = self.loff
        retv.nb_param = self.nb_param
        retv.gp_sc_id = self.gp_sc_id
        retv.image_acquisition_time = self.image_acquisition_time
        retv.spectral_channel_id = self.spectral_channel_id
        retv.nominal_product_time = self.nominal_product_time
        retv.sgs_product_quality = self.sgs_product_quality
        retv.sgs_product_completeness = self.sgs_product_completeness
        retv.product_algorithm_version = self.product_algorithm_version

        retv.pcph_bl = coverage.project_array(self.crr)
        retv.pcph_bl = self.pcph_bl_palette

        retv.processing_flags = \
            coverage.project_array(self.processing_flags)

        retv.qc_straylight = self.qc_straylight
        retv.region_name = dest_area
        retv.area = region
        retv.projection_name = region.proj_id

        retv.pcs_def = pcs_def_from_region(region)

        retv.num_of_columns = region.x_size
        retv.num_of_lines = region.y_size
        retv.xscale = region.pixel_size_x
        retv.yscale = region.pixel_size_y

        import pyproj
        prj = pyproj.Proj(region.proj4_string)
        aex = region.area_extent
        lonur, latur = prj(aex[2], aex[3], inverse=True)
        lonll, latll = prj(aex[0], aex[1], inverse=True)
        retv.ll_lon = lonll
        retv.ll_lat = latll
        retv.ur_lon = lonur
        retv.ur_lat = latur

        self.shape = region.shape

        retv.filled = True
        retv.resolution = self.resolution

        return retv

class MsgCRPhData(object):

    """NWCSAF/MSG CRPh layer
    """

    def __init__(self):
        self.data = None
        self.scaling_factor = 1
        self.offset = 0
        self.num_of_lines = 0
        self.num_of_columns = 0
        self.product = ""
        self.id = ""


class MsgCRPh(mpop.channel.GenericChannel):

    """NWCSAF/MSG CRPh data structure as retrieved from HDF5
    file. Resolution sets the nominal resolution of the data.
    Palette now missing
    """

    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "CRPh")
        self.filled = False
        self.name = "CRPh"
#       self.resolution = resolution
        self.package = ""
        self.saf = ""
        self.product_name = ""
        self.num_of_columns = 0
        self.num_of_lines = 0
        self.projection_name = ""
        self.pcs_def = ""
        self.xscale = 0
        self.yscale = 0
        self.ll_lon = 0.0
        self.ll_lat = 0.0
        self.ur_lon = 0.0
        self.ur_lat = 0.0
        self.region_name = ""
        self.cfac = 0
        self.lfac = 0
        self.coff = 0
        self.loff = 0
        self.nb_param = 0
        self.gp_sc_id = 0
        self.image_acquisition_time = 0
        self.spectral_channel_id = 0
        self.nominal_product_time = 0
        self.sgs_product_quality = 0
        self.sgs_product_completeness = 0
        self.product_algorithm_version = ""
        self.crph = None
        self.crph_crr = None
        self.crph_accum = None
        self.crph_IQF = None
        self.crph_quality = None
        self.crph_dataflag = None
        self.processing_flags = None
        self.shape = None
        self.satid = ""
        self.qc_straylight = -1
        self.crph = None
        self.crph_pc_palette = None
        self.crph_quality_palette = None
        self.crph_sflag_palette = None
        
    def __str__(self):
        return ("'%s: shape %s, resolution %sm'" %
                (self.name,
                 self.crph_crr.shape,
                 self.resolution))

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

# ------------------------------------------------------------------
    def read(self, filename, calibrate=True):
        """Reader for the . Use *filename* to read data.
        """
        import h5py

# Erste Zeile notwendig?
        self.crph = MsgCRPhData()
        self.crph_crr = MsgCRPhData()
        self.crph_accum = MsgCRPhData()
        self.crph_iqf = MsgCRPhData()
        self.crph_quality = MsgCRPhData()
        self.crph_dataflag = MsgCRPhData()

        self.processing_flags = MsgCRPhData()

        LOG.debug("Filename = <" + str(filename) + ">")
        h5f = h5py.File(filename, 'r')
        # pylint: disable-msg=W0212
        self.package = h5f.attrs["PACKAGE"]
        self.saf = h5f.attrs["SAF"]
        self.product_name = h5f.attrs["PRODUCT_NAME"]
        self.num_of_columns = h5f.attrs["NC"]
        self.num_of_lines = h5f.attrs["NL"]
        self.projection_name = h5f.attrs["PROJECTION_NAME"]
        self.region_name = h5f.attrs["REGION_NAME"]
        self.cfac = h5f.attrs["CFAC"]
        self.lfac = h5f.attrs["LFAC"]
        self.coff = h5f.attrs["COFF"]
        self.loff = h5f.attrs["LOFF"]
        self.nb_param = h5f.attrs["NB_PARAMETERS"]
        self.gp_sc_id = h5f.attrs["GP_SC_ID"]
        self.image_acquisition_time = h5f.attrs["IMAGE_ACQUISITION_TIME"]
        self.spectral_channel_id = h5f.attrs["SPECTRAL_CHANNEL_ID"]
        self.nominal_product_time = h5f.attrs["NOMINAL_PRODUCT_TIME"]
        self.sgs_product_quality = h5f.attrs["SGS_PRODUCT_QUALITY"]
        self.sgs_product_completeness = h5f.attrs["SGS_PRODUCT_COMPLETENESS"]
        self.product_algorithm_version = h5f.attrs["PRODUCT_ALGORITHM_VERSION"]
        # pylint: enable-msg=W0212
        # ------------------------

        # The CRPh CRR data
        h5d = h5f['CRPh_CRR']
        self.crph_crr.data = h5d[:, :]
        self.crph_crr.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.crph_crr.offset = h5d.attrs["OFFSET"]
        self.crph_crr.num_of_lines = h5d.attrs["N_LINES"]
        self.crph_crr.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.crph_crr.num_of_lines,
                      self.crph_crr.num_of_columns)
        self.crph_crr.product = h5d.attrs["PRODUCT"]
        self.crph_crr.id = h5d.attrs["ID"]
        if calibrate:
            self.crph_crr = (self.crph_crr.data *
                                  self.crph_crr.scaling_factor +
                                  self.crph_crr.offset)
        else:
            self.crph_crr = self.crph_crr.data
        self.crph_crr_palette = _get_palette(h5f, 'CRPh_CRR') / 255.0

        # The CRPh ACCUM data
        h5d = h5f['CRPh_ACUM']
        self.crph_accum.data = h5d[:, :]
        self.crph_accum.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.crph_accum.offset = h5d.attrs["OFFSET"]
        self.crph_accum.num_of_lines = h5d.attrs["N_LINES"]
        self.crph_accum.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.crph_accum.num_of_lines,
                      self.crph_accum.num_of_columns)
        self.crph_accum.product = h5d.attrs["PRODUCT"]
        self.crph_accum.id = h5d.attrs["ID"]
        if calibrate:
            self.crph_accum = (self.crph_accum.data *
                                  self.crph_accum.scaling_factor +
                                  self.crph_accum.offset)
        else:
            self.crph_accum = self.crph_accum.data
        self.crph_accum_palette = _get_palette(h5f, 'CRPh_ACUM') / 255.0


        # The CRPH IQF data
        h5d = h5f['CRPh_DATAFLAG']
        self.crph_iqf.data = h5d[:, :]
        self.crph_iqf.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.crph_iqf.offset = h5d.attrs["OFFSET"]
        self.crph_iqf.num_of_lines = h5d.attrs["N_LINES"]
        self.crph_iqf.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.crph_iqf.num_of_lines,
                      self.crph_iqf.num_of_columns)
        self.crph_iqf.product = h5d.attrs["PRODUCT"]
        self.crph_iqf.id = h5d.attrs["ID"]
        
        # The PPh QUALITY data
        h5d = h5f['CRPh_QUALITY']
        self.crph_quality.data = h5d[:, :]
        self.crph_quality.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.crph_quality.offset = h5d.attrs["OFFSET"]
        self.crph_quality.num_of_lines = h5d.attrs["N_LINES"]
        self.crph_quality.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.crph_quality.num_of_lines,
                      self.crph_quality.num_of_columns)
        self.crph_quality.product = h5d.attrs["PRODUCT"]
        self.crph_quality.id = h5d.attrs["ID"]

        # The PPh DATA FLAG data
        h5d = h5f['CRPh_DATAFLAG']
        self.crph_dataflag.data = h5d[:, :]
        self.crph_dataflag.scaling_factor = h5d.attrs["SCALING_FACTOR"]
        self.crph_dataflag.offset = h5d.attrs["OFFSET"]
        self.crph_dataflag.num_of_lines = h5d.attrs["N_LINES"]
        self.crph_dataflag.num_of_columns = h5d.attrs["N_COLS"]
        self.shape = (self.crph_dataflag.num_of_lines,
                      self.crph_dataflag.num_of_columns)
        self.crph_dataflag.product = h5d.attrs["PRODUCT"]
        self.crph_dataflag.id = h5d.attrs["ID"]

        # ------------------------

        h5f.close()

        self.processing_flags = self.processing_flags.data

        self.area = get_area_from_file(filename)

        self.filled = True

    def project(self, coverage):
        """Remaps the NWCSAF/MSG CRPh to cartographic map-projection on
        area give by a pre-registered area-id. Faster version of msg_remap!
        """
        LOG.info("Projecting channel %s..." % (self.name))

        region = coverage.out_area
        dest_area = region.area_id

        retv = MsgCRPh()

        retv.name = self.name
        retv.package = self.package
        retv.saf = self.saf
        retv.product_name = self.product_name
        retv.region_name = dest_area
        retv.cfac = self.cfac
        retv.lfac = self.lfac
        retv.coff = self.coff
        retv.loff = self.loff
        retv.nb_param = self.nb_param
        retv.gp_sc_id = self.gp_sc_id
        retv.image_acquisition_time = self.image_acquisition_time
        retv.spectral_channel_id = self.spectral_channel_id
        retv.nominal_product_time = self.nominal_product_time
        retv.sgs_product_quality = self.sgs_product_quality
        retv.sgs_product_completeness = self.sgs_product_completeness
        retv.product_algorithm_version = self.product_algorithm_version

        retv.crph_bl = coverage.project_array(self.crr)
        retv.crph_bl = self.crph_bl_palette

        retv.processing_flags = \
            coverage.project_array(self.processing_flags)

        retv.qc_straylight = self.qc_straylight
        retv.region_name = dest_area
        retv.area = region
        retv.projection_name = region.proj_id

        retv.pcs_def = pcs_def_from_region(region)

        retv.num_of_columns = region.x_size
        retv.num_of_lines = region.y_size
        retv.xscale = region.pixel_size_x
        retv.yscale = region.pixel_size_y

        import pyproj
        prj = pyproj.Proj(region.proj4_string)
        aex = region.area_extent
        lonur, latur = prj(aex[2], aex[3], inverse=True)
        lonll, latll = prj(aex[0], aex[1], inverse=True)
        retv.ll_lon = lonll
        retv.ll_lat = latll
        retv.ur_lon = lonur
        retv.ur_lat = latur

        self.shape = region.shape

        retv.filled = True
        retv.resolution = self.resolution

        return retv



""" NEU ENDE """



class NordRadCType(object):

    """Wrapper aroud the msg_ctype channel.
    """

    def __init__(self, ctype_instance):
        self.ctype = ctype_instance
        self.datestr = ctype_instance.image_acquisition_time

    def save(self, filename):
        """Save the current instance to nordrad hdf format.
        """
        import _pyhl
        status = 1

        msgctype = self.ctype

        node_list = _pyhl.nodelist()

        # What
        node = _pyhl.node(_pyhl.GROUP_ID, "/what")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/object")
        node.setScalarValue(-1, "IMAGE", "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/sets")
        node.setScalarValue(-1, 1, "int", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/version")
        node.setScalarValue(-1, "H5rad 1.2", "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/date")
        yyyymmdd = self.datestr[0:8]
        hourminsec = self.datestr[8:12] + '00'
        node.setScalarValue(-1, yyyymmdd, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/what/time")
        node.setScalarValue(-1, hourminsec, "string", -1)
        node_list.addNode(node)

        # Where
        node = _pyhl.node(_pyhl.GROUP_ID, "/where")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/projdef")
        node.setScalarValue(-1, msgctype.area.proj4_string, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/xsize")
        node.setScalarValue(-1, msgctype.num_of_columns, "int", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/ysize")
        node.setScalarValue(-1, msgctype.num_of_lines, "int", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/xscale")
        node.setScalarValue(-1, msgctype.xscale, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/yscale")
        node.setScalarValue(-1, msgctype.yscale, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/LL_lon")
        node.setScalarValue(-1, msgctype.ll_lon, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/LL_lat")
        node.setScalarValue(-1, msgctype.ll_lat, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/UR_lon")
        node.setScalarValue(-1, msgctype.ur_lon, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/where/UR_lat")
        node.setScalarValue(-1, msgctype.ur_lat, "float", -1)
        node_list.addNode(node)

        # How
        node = _pyhl.node(_pyhl.GROUP_ID, "/how")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/how/area")
        node.setScalarValue(-1, msgctype.region_name, "string", -1)
        node_list.addNode(node)

        # image1
        node = _pyhl.node(_pyhl.GROUP_ID, "/image1")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.DATASET_ID, "/image1/data")
        node.setArrayValue(1, [msgctype.num_of_columns, msgctype.num_of_lines],
                           msgctype.cloudtype.astype('B'), "uchar", -1)
        node_list.addNode(node)

        node = _pyhl.node(_pyhl.GROUP_ID, "/image1/what")
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/product")
        # We should eventually try to use the msg-parameters "package",
        #"product_algorithm_version", and "product_name":
        node.setScalarValue(1, 'MSGCT', "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/prodpar")
        node.setScalarValue(1, 0.0, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/quantity")
        node.setScalarValue(1, "ct", "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/startdate")
        node.setScalarValue(-1, yyyymmdd, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/starttime")
        node.setScalarValue(-1, hourminsec, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/enddate")
        node.setScalarValue(-1, yyyymmdd, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/endtime")
        node.setScalarValue(-1, hourminsec, "string", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/gain")
        node.setScalarValue(-1, 1.0, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/offset")
        node.setScalarValue(-1, 0.0, "float", -1)
        node_list.addNode(node)
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/nodata")
        node.setScalarValue(-1, 0.0, "float", -1)
        node_list.addNode(node)
        # What we call missingdata in PPS:
        node = _pyhl.node(_pyhl.ATTRIBUTE_ID, "/image1/what/undetect")
        node.setScalarValue(-1, 20.0, "float", -1)
        node_list.addNode(node)

        node_list.write(filename, COMPRESS_LVL)

        return status

MSG_PGE_EXTENTIONS = ["PLAX.CTTH.0.h5", "PLAX.CLIM.0.h5", "h5"]


def get_best_product(filename, area_extent):
    """Get the best of the available products for the *filename* template.
    """

    for ext in MSG_PGE_EXTENTIONS:
        match_str = filename + "." + ext
        LOG.debug("glob-string for filename: " + str(match_str))
        flist = glob.glob(match_str)
        if len(flist) == 0:
            LOG.warning("No matching .%s input MSG file."
                        % ext)
        else:
            # File found:
            if area_extent is None:
                LOG.warning("Didn't specify an area, taking " + flist[0])
                return flist[0]
            for fname in flist:
                aex = get_area_extent(fname)
                #import pdb
                # pdb.set_trace()
                if np.all(np.max(np.abs(np.array(aex) -
                                        np.array(area_extent))) < 1000):
                    LOG.info("MSG file found: %s" % fname)
                    return fname
            LOG.info("Did not find any MSG file for specified area")


def get_best_products(filename, area_extent):
    """Get the best of the available products for the *filename* template.
    """

    filenames = []

    for ext in MSG_PGE_EXTENTIONS:
        match_str = filename + "." + ext
        LOG.debug('Match string = ' + str(match_str))
        flist = glob.glob(match_str)
        if len(flist) == 0:
            LOG.warning("No matching .%s input MSG file."
                        % ext)
        else:
            # File found:
            if area_extent is None:
                LOG.warning("Didn't specify an area, taking " + flist[0])
                filenames.append(flist[0])
            else:
                found = False
                for fname in flist:
                    aex = get_area_extent(fname)
                    if np.all(np.max(np.abs(np.array(aex) -
                                            np.array(area_extent))) < 1000):
                        found = True
                        LOG.info("MSG file found: %s" % fname)
                        filenames.append(fname)
                    if not found:
                        LOG.info(
                            "Did not find any MSG file for specified area")
    LOG.debug("Sorted filenames: %s", str(sorted(filenames)))
    return sorted(filenames)


def get_area_from_file(filename):
    """Get the area from the h5 file.
    """
    from pyresample.geometry import AreaDefinition
    import h5py

    aex = get_area_extent(filename)
    h5f = h5py.File(filename, 'r')
    pname = h5f.attrs["PROJECTION_NAME"]
    proj = {}
    if pname.startswith("GEOS"):
        proj["proj"] = "geos"
        proj["a"] = "6378169.0"
        proj["b"] = "6356583.8"
        proj["h"] = "35785831.0"
        proj["lon_0"] = str(float(pname.split("<")[1][:-1]))
    else:
        raise NotImplementedError("Only geos projection supported yet.")
    area_def = AreaDefinition(h5f.attrs["REGION_NAME"],
                              h5f.attrs["REGION_NAME"],
                              pname,
                              proj,
                              int(h5f.attrs["NC"]),
                              int(h5f.attrs["NL"]),
                              aex)
    h5f.close()
    return area_def


def _get_sat_id(scene):
    satid = scene.satname.capitalize()
    if scene.number != '':
        ct_chan.satid = (ct_chan.satid + 
                         str(int(scene.number)).rjust(2))
    return satid
    
    
def load(scene, **kwargs):
    """Load data into the *channels*. *Channels* is a list or a tuple
    containing channels we will load data into. If None, all channels are
    loaded.
    """

    area_extent = kwargs.get("area_extent")
    calibrate = kwargs.get("calibrate", True)

    print area_extent
    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, scene.fullname + ".cfg"))
    directory = conf.get(scene.instrument_name + "-level3",
                         "dir",
                         raw=True)
    filename = conf.get(scene.instrument_name + "-level3", "filename",
                        raw=True)
    pathname = os.path.join(directory, filename)

    LOG.debug("Inside load: " + str(scene.channels_to_load))

    if "CTTH" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    % {"number": "03",
                       "product": "CTTH_"})
        ct_chan = MsgCTTH()
        ct_chan.read(get_best_product(filename, area_extent),calibrate)
        ct_chan.satid = _get_sat_id(scene)
        ct_chan.resolution = ct_chan.area.pixel_size_x
        scene.channels.append(ct_chan)

    if "CloudType" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    % {"product": "CT___"})
        products = get_best_products(filename, area_extent)

        ct_chan = MsgCloudType()
        ct_chan.read(products[-1])
        LOG.debug("Uncorrected file: %s", products[-1])
        ct_chan.name = "CloudType"
        ct_chan.satid = _get_sat_id(scene)
        ct_chan.resolution = ct_chan.area.pixel_size_x
        scene.channels.append(ct_chan)
    if "CloudType_plax" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    % {"number": "02",
                       "product": "CT___"})
        products = get_best_products(filename, area_extent)
        ct_chan_plax = MsgCloudType()
        LOG.debug("Parallax corrected file: %s", products[0])
        ct_chan_plax.read(products[0])
        ct_chan_plax.name = "CloudType_plax"
        ct_chan_plax.satid = _get_sat_id(scene)
        ct_chan_plax.resolution = ct_chan_plax.area.pixel_size_x
        scene.channels.append(ct_chan_plax)
    if "CRR" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    % {"number": "05",
                       "product": "CRR__"})
        ct_chan = MsgCRR()
        ct_chan.read(get_best_product(filename, area_extent),calibrate)
        ct_chan.name = "CRR"
        ct_chan.satid = _get_sat_id(scene)
        ct_chan.resolution = ct_chan.area.pixel_size_x
        scene.channels.append(ct_chan)
    if "PC" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    % {"number": "04",
                       "product": "PC___"})
        ct_chan = MsgPC()
        ct_chan.read(get_best_product(filename, area_extent),calibrate)
        ct_chan.name = "PC"
        ct_chan.satid = _get_sat_id(scene)
        ct_chan.resolution = ct_chan.area.pixel_size_x
        scene.channels.append(ct_chan)
    if "SPhR" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    % {"number": "13",
                       "product": "SPhR_"})
        ct_chan = MsgSPhR()
        ct_chan.read(get_best_product(filename, area_extent),calibrate)
        ct_chan.name = "SPhR"
        ct_chan.satid = _get_sat_id(scene)
        ct_chan.resolution = ct_chan.area.pixel_size_x
        scene.channels.append(ct_chan)
    if "PCPh" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    % {"number": "14",
                       "product": "PCPh_"})
        ct_chan = MsgPCPh()
        ct_chan.read(get_best_product(filename, area_extent),calibrate)
        ct_chan.name = "PCPh"
        ct_chan.satid = _get_sat_id(scene)
        ct_chan.resolution = ct_chan.area.pixel_size_x
        scene.channels.append(ct_chan)
    if "CRPh" in scene.channels_to_load:
        filename = (scene.time_slot.strftime(pathname)
                    % {"number": "14",
                       "product": "CRPh_"})
        ct_chan = MsgCRPh()
        ct_chan.read(get_best_product(filename, area_extent),calibrate)
        ct_chan.name = "CRPh"
        ct_chan.satid = _get_sat_id(scene)
        ct_chan.resolution = ct_chan.area.pixel_size_x
        scene.channels.append(ct_chan)

    
    LOG.info("Loading channels done.")

