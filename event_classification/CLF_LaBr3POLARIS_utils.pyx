#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#!/usr/bin/python

# 210721 - Added to force use of python3 verion of Cython
#import include
# cython: language_level=3

"""
Description:
Contains functions that will be used with processPolarisData.py for Compton camera event processing.

NOTES:
- written using python v2.7.15 / updated to also work in python 3.6.2

Code borrowed heavily from Dennis Mackin <dsmackin@mdanderson.org>
"""
__author__ = "Steve Peterson <steve.peterson@uct.ac.za>"
__date__ = "November 01, 2018"
__version__ = "$Revision: 4.0.0$"

#------------------------------------------------------------------
# PYTHON IMPORT STATEMENTS
#------------------------------------------------------------------

import sys, os
import numpy
import numpy as np
from math import sin, cos, pi, log, floor
import pandas
import cProfile
import re
import matplotlib
import matplotlib.pyplot as plt
import pylab


#------------------------------------------------------------------
# CONSTANTS
#------------------------------------------------------------------
MeCsq = 510.9989461  # electron mass in energy units (keV)

#------------------------------------------------------------------
# FUNCTIONS
#------------------------------------------------------------------
def physical_energy_ordering_double(E1, E2):

    #  check if energy ordering of double scatter event produces physical scatter angle
    E0 = E1 + E2

    if numpy.abs(1 + MeCsq * ( 1.0/(E0) - 1.0/(E0 - E1) ) ) < 1:
        return 1
    else:
        return 0

#checks for unphysical events (theta1 = nan) from double scatter event data
#   - input format (scatters): eng1[0], x1[1], y1[2], z1[3], eng2[4], x2[5], y2[6], z2[7] (original code)
#   - input format for new code: 'time[0]', 'energyPOLARIS[1]', 'x[2]', 'y[3]', 'z[4]', 'energyL0[5]', 'Window_ID[6]', 'TimeDiff[7]', 'TimeDiffCum[8]', 'theta1[9]'
def filtering_unphysical_double_scatters(scatters):

    print('-------------- UNFILTERED (filtering_unphysical_double_scatters) -------------- \n', scatters)

    scatters_physical = []
    count_both, count_one, count_flip, count_none = [0] * 4

    #  loop through list of events
    for index in range( len(scatters) ):
        #  check original energy ordering
        order1 = physical_energy_ordering_double(scatters[index][1], scatters[index][5]) # new code would be scatters[index][1], scatters[index][5]
        #  check flipped energy ordering
        order2 = physical_energy_ordering_double(scatters[index][5], scatters[index][1]) # new code would be scatters[index][5], scatters[index][1]

        #  storing appropriate events into final output array: scatters_physical
        if (order1 == 1 & order2 == 1):
            scatters_physical.append(scatters[index])
            count_both += 1
        elif (order1 == 1):
            scatters_physical.append(scatters[index]) 
            count_one += 1
        elif (order2 == 1):
            SE = scatters[index] 
            flipped_scatter = numpy.array((SE[0],SE[5], SE[2], SE[3], SE[4], SE[1], SE[6], SE[7]))
            scatters_physical.append(flipped_scatter)
            count_one += 1; count_flip += 1
        else:
            count_none += 1

    #  print counts to screen
    print ('    - results of event ordering -> number of physical events returned: {} | both work: {} | only one order works: {} | order flipped: {} | neither work: {}'.format(len(scatters_physical), count_both, count_one, count_flip, count_none))
    #print('-------------- PHYSICAL SCATTERS FILTERED (filtering_unphysical_double_scatters) -------------- \n', scatters_physical)
    #  return physical events
    #    vstack takes list of numpy arrays and converts into single numpy array
    return numpy.vstack(scatters_physical)



#  function used by compton_line_filtering() to calculate the energy of the first scatter
def calculate_expected_first_scatter_energy(E0 , theta):

    #  re-arrangement for Compton scatter equation to solve for E1
    alpha = E0 / MeCsq
    beta = alpha * ( 1 - numpy.cos(theta) ) 
    return E0 * beta / ( 1 + beta )


#  filtering events using Compton Line Filtering (based on expected gamma energies, input variable: CL_energies)
#   - input format (scatters): eng1[0], x1[1], y1[2], z1[3], eng2[4], x2[5], y2[6], z2[7]
# - input format for new code: 'time[0]', 'energyPOLARIS[1]', 'x[2]', 'y[3]', 'z[4]', 'energyL0[5]', 'Window_ID[6]', 'TimeDiff[7]', 'TimeDiffCum[8]', 'theta1[9]'
#   - returns data in same format
def compton_line_filtering(scatters, CL_range, CL_energies, No_CLF):

    scatters_filtered = []
    count_compton = [0] * len(CL_energies) 
    
    # Loop through list of events
    for index in range(len(scatters)):

        # Use the first and second energies to calculate E0 and theta1
        E1 = scatters[index][1]
        E2 = scatters[index][5] 
        E0 = E1 + E2
        theta1 = numpy.arccos(1 + MeCsq * ((1.0 / E0) - (1.0 / (E0 - E1))))
        
        No_CLF[index][0] = E0
        No_CLF[index][1] = E1
        No_CLF[index][2] = theta1

        scatters[index][7] = theta1
        
        # Loop through expected gamma energies
        for i, e in enumerate(CL_energies):
            # Calculate range of Compton values for given total energy and first scatter angle
            expected_E1 = calculate_expected_first_scatter_energy(e, theta1)
            minE1 = (CL_range[0] * expected_E1)
            maxE1 = (CL_range[1] * expected_E1)
            print('expected_E1: ', expected_E1, 'theta1: ', theta1, 'E1: ', E1, 'minE1: ', minE1, 'maxE1: ', maxE1)
            # Check if energy falls within expected range
            if minE1 < E1 < maxE1:
                # Add filtered data to output array
                scatters_filtered.append(scatters[index])
                count_compton[i] += 1

    # Total up number of Compton filtered events
    total_compton = numpy.sum(count_compton)

    #  print results to screen
    print ('    - results of Compton filtering -> number of filtered events returned: {} -> counts: {} by energy {}, respectively'.format(total_compton, count_compton, CL_energies))
    #  return filtered events
    #    vstack takes list of numpy arrays and converts into single numpy array
    #print('-------------- COMPTON FILTERED (compton_line_filtering) -------------- \n', scatters_filtered)
    return numpy.vstack(scatters_filtered), numpy.vstack(No_CLF)




