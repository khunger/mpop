# -*- coding: utf-8 -*-
# Copyright (c) 2014, 2015
#
# Author(s):
#
#   Panu Lahtinen <pnuu+git@iki.fi>
#
# This file is part of mpop.
#
# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

'''Helper functions for eg. performing Sun zenith angle correction.
'''

import numpy as np


def sunzen_corr_cos(data, cos_zen, limit=80.):
    '''Perform Sun zenith angle correction to the given *data* using
    cosine of the zenith angle (*cos_zen*).  The correction is limited
    to *limit* degrees (default: 80.0 degrees).  For larger zenith
    angles, the correction is the same as at the *limit*.  Both *data*
    and *cos_zen* are given as 2-dimensional Numpy arrays or Numpy
    MaskedArrays, and they should have equal shapes.
    '''

    # Convert the zenith angle limit to cosine of zenith angle
    cos_limit = np.cos(np.radians(limit))

    # Cosine correction
    lim_y, lim_x = np.where(cos_zen > cos_limit)
    data[lim_y, lim_x] /= cos_zen[lim_y, lim_x]
    # Use constant value (the limit) for larger zenith
    # angles
    lim_y, lim_x = np.where(cos_zen <= cos_limit)
    data[lim_y, lim_x] /= cos_limit

    return data


def viewzen_corr(data, view_zen):
    """Apply atmospheric correction on the given *data* using the
    specified satellite zenith angles (*view_zen*). Both input data
    are given as 2-dimensional Numpy (masked) arrays, and they should
    have equal shapes.
    The *data* array will be changed in place and has to be copied before.
    """
    def ratio(value, v_null, v_ref):
        return (value - v_null)/(v_ref - v_null)

    def tau0(t):
        T_0 = 210.0
        T_REF = 320.0
        TAU_REF = 9.85
        return (1 + TAU_REF)**ratio(t, T_0, T_REF) - 1

    def tau(t):
        T_0 = 170.0
        T_REF = 295.0
        TAU_REF = 1.0
        M = 4
        return TAU_REF*ratio(t, T_0, T_REF)**M

    def delta(z):
        Z_0 = 0.0
        Z_REF = 70.0
        DELTA_REF = 6.2
        return (1 + DELTA_REF)**ratio(z, Z_0, Z_REF) - 1

    y0, x0 = np.ma.where(view_zen == 0)
    data[y0, x0] += tau0(data[y0, x0])

    y, x = np.ma.where((view_zen > 0) & (view_zen < 90) & (~data.mask))
    data[y, x] += tau(data[y, x])*delta(view_zen[y, x])
