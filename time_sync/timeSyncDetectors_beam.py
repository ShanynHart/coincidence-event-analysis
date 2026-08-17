################################################################################
#                               to run:                                        #
# python3 timeSyncDetectors.py /directory/to/rootfile/  (R**_rawData.root)   
# eg. timesyncLaBrPOLARIS % python3 timeSyncDetectors.py ~/Documents/PhD/exp/2022/220615/analysis/angle/water/run27/ R63_sorted.root (obtained from sort1labr.C then sort2labr.C then sort3labr.C)
__author__ = "Shanyn Hart"
__date__ = "2023-05-26"
__version__ = "1.0"

######################## CODE UPDATES: ########################
# 2022-08-22: Created script to run all detectors for all runs
# 2022-08-22: Added in all relevant transformations for each run
# 2022-08-22: Added in all relevant detectors for each run
# 2023-03-30: Changed LaBr3 TTree branch names to correspond to new RXX.root file
#             which has been updated to only use insync events with RF for background reduction.
# 2023-05-26: All LaBr3:Ce detectors share a tree. Code updated to reflect this.
# 2023-05-26: Code improved to run with functions                                
################################################################################

#------------------------------------------------------------------
# python import statements
#------------------------------------------------------------------
from cmath import nan
from ctypes import sizeof
from tokenize import Double
from unittest import skip
from venv import create
import ROOT as root
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import uproot
import csv as csv
import io as io
import sys, os
from math import sin, cos, pi, log, floor
import cProfile
import re
import pylab
import pyroot as pr
import decimal
import time as time
import fastparquet
import pyarrow
#os.environ["OPENBLAS_NUM_THREADS"] = "200"
import pyximport; pyximport.install(setup_args={"include_dirs":np.get_include()}, language_level=3)
run_dir = os.getcwd()
LaBrPOLARIS_utils =  run_dir + "/LaBrPOLARIS_utils.pyx"
import LaBrPOLARIS_utils as utils #type: ignore
from ydata_profiling import ProfileReport

# run this script in multiple threads
os.environ["NUMEXPR_MAX_THREADS"] = "16"

pd.options.mode.chained_assignment = None  # default='warn'

#_______________________________________________________________________________________________
# ________________  RELEVANT TRANSFORMATIONS FOR D1 FOR EACH RUN 15/06/2022 ____________________ 
#_______________________________________________________________________________________________
transformation_run1 = [90.0, 0.0, 0.0, -11.0, 362.0, 0.0]
transformation_run2 = [90.0, 0.0, 0.0, -11.0, 362.0, 0.0]
transformation_run3 = [90.0, 0.0, 0.0, -11.0, 362.0, 0.0]
transformation_run4 = [90.0, 0.0, 0.0, -11.0, 362.0, 0.0]
transformation_run5 = [90.0, 0.0, 0.0, -11.0, 362.0, 0.0]
transformation_run6 = [90.0, 0.0, 0.0, -11.0, 362.0, 0.0]
transformation_run7 = [90.0, 0.0, 0.0, -11.0, 362.0, 0.0]
transformation_run8 = [90.0, 0.0, 0.0, -11.0, 362.0, 0.0]
transformation_run9 = [90.0, 0.0, 0.0, -11.0, 192.0, 0.0]
transformation_run10 = [90.0, 0.0, 0.0, -11.0, 192.0, 0.0]
transformation_run11 = [90.0, 0.0, 0.0, -11.0, 192.0, 0.0]
transformation_run12 = [0.0, 0.0, 0.0, -11.0, 0.0, -192.0]
transformation_run13 = [0.0, 0.0, 0.0, -11.0, 0.0, -192.0]
transformation_run14 = [0.0, 0.0, 0.0, -11.0, 0.0, -192.0]
transformation_run15 = [0.0, 0.0, 0.0, -11.0, 0.0, -192.0]
transformation_run16 = [45.0, 0.0, 0.0, -11.0, 135.8, -135.8]
transformation_run17 = [45.0, 0.0, 0.0, -11.0, 135.8, -135.8]
transformation_run18 = [45.0, 0.0, 0.0, -11.0, 213.5, -213.5]
transformation_run19 = [45.0, 0.0, 0.0, -11.0, 213.5, -213.5]
transformation_run20 = [90.0, 0.0, 0.0, -11.0, 362.0, 0.0]
transformation_run21 = [90.0, 0.0, 0.0, -11.0, 122.0, 0.0]
transformation_run22 = [90.0, 0.0, 0.0, -11.0, 122.0, 0.0]
transformation_run23 = [90.0, 0.0, 0.0, -11.0, 122.0, 0.0]
transformation_run24 = [90.0, 0.0, 0.0, -11.0, 122.0, 0.0]
transformation_run25 = [0.0, 0.0, 0.0, -11.0, 0.0, -202.0]
transformation_run26 = [0.0, 0.0, 0.0, -11.0, 0.0, -202.0]
transformation_run27 = [0.0, 0.0, 0.0, -11.0, 0.0, -202.0]
transformation_run28 = [270.0, 0.0, 0.0, -11.0, -122.0, 0.0]
transformation_run29 = [270.0, 0.0, 0.0, -11.0, -122.0, 0.0]
transformation_run30 = [270.0, 0.0, 0.0, -11.0, -122.0, 0.0]
transformation_run31 = [270.0, 0.0, 0.0, -11.0, -122.0, 0.0]
transformation_run32 = [270.0, 0.0, 0.0, -11.0, -122.0, 0.0]
transformation_run33 = [270.0, 0.0, 0.0, -11.0, -122.0, 0.0]
transformation_run34 = [270.0, 0.0, 0.0, -11.0, -102.0, 0.0]
transformation_run35 = [270.0, 0.0, 0.0, -11.0, -102.0, 0.0]
transformation_run36 = [270.0, 0.0, 0.0, -11.0, -102.0, 0.0]
transformation_run37 = [45.0, 0.0, 0.0, -11.0, 235.6, -235.6]
transformation_run38 = [45.0, 0.0, 0.0, -11.0, 235.6, -235.6]
transformation_run39 = [45.0, 0.0, 0.0, -11.0, 235.6, -235.6]
transformation_run40 = [45.0, 0.0, 0.0, -11.0, 235.6, -235.6]
transformation_run41 = [315.0, 0.0, 0.0, -11.0, -235.6, -235.6]
transformation_run42 = [315.0, 0.0, 0.0, -11.0, -235.6, -235.6]
transformation_run43 = [315.0, 0.0, 0.0, -11.0, -235.6, -235.6]

#make a dictionary of the transformations (automates the process for each run)
transformation_dict = {1: transformation_run1, 2: transformation_run2, 3: transformation_run3, 4: transformation_run4, 5: transformation_run5, 6: transformation_run6, 7: transformation_run7, 8: transformation_run8, 9: transformation_run9, 10: transformation_run10, 11: transformation_run11, 12: transformation_run12, 13: transformation_run13, 14: transformation_run14, 15: transformation_run15, 16: transformation_run16, 17: transformation_run17, 18: transformation_run18, 19: transformation_run19, 20: transformation_run20, 21: transformation_run21, 22: transformation_run22, 23: transformation_run23, 24: transformation_run24, 25: transformation_run25, 26: transformation_run26, 27: transformation_run27, 28: transformation_run28, 29: transformation_run29, 30: transformation_run30, 31: transformation_run31, 32: transformation_run32, 33: transformation_run33, 34: transformation_run34, 35: transformation_run35, 36: transformation_run36, 37: transformation_run37, 38: transformation_run38, 39: transformation_run39, 40: transformation_run40, 41: transformation_run41, 42: transformation_run42, 43: transformation_run43}


#_______________________________________________________________________________________________
# ________________  RELEVANT TRANSFORMATIONS FOR D1 FOR EACH RUN 28/02/2023 ____________________ 
#_______________________________________________________________________________________________




#_______________________________________________________________________________________________________________________________________________________________________________________
def read_polaris_data(data_dir):
    print('\033[91m' +'________________________________________') 
    print('\033[91m' +'_____________ READ IN POLARIS DATA _____________')
    print('\033[91m' +'________________________________________', '\n')

    data_filename = 'mod51.txt'
    run_num = int(re.search('run(\d+)', data_dir).group(1))

    start_time = time.time()

    print('-------------->> Run number: ', run_num)

    mod_event_data = pd.read_csv(data_dir + data_filename, sep='	', header=None)
    mod_event_data.columns = ['scatters', 'x', 'y', 'z', 'energy', 'time']
    mod_event_data = mod_event_data[(mod_event_data['scatters'] == 122) | (mod_event_data['scatters'] == 1)]
    mod_event_data_df = pd.DataFrame(mod_event_data)

    evts_all = len(mod_event_data.index)
    time_total = (mod_event_data.iloc[-1]['time'] - mod_event_data.iloc[0]['time']) * 1e-8
    evt_rate_all = evts_all / time_total
    POLARIS_time_diff = np.diff(mod_event_data['time']*10)

    print('\033[91m' +'{:30} {:<10d}'.format('\n nEvents POLARIS: ', evts_all),
          '{:30} {:<10.1f}'.format('\n Total time of Run POLARIS (minutes): ', time_total/60),
          '{:30} {:<10.1f}'.format('\n Event rate POLARIS (events/sec): ', evt_rate_all), '\n\n')

    if data_filename == 'mod51.txt':
        det_index = 0
    mod_event_data_df['detector'] = det_index
    mod_event_data_df = mod_event_data_df[['detector', 'scatters', 'energy', 'x', 'y', 'z', 'time']]
    mod_event_data_df['scatters'] = mod_event_data_df['scatters'].astype(int)
    mod_event_data_df = mod_event_data_df.reset_index(drop=True)
    mod_event_data_df['time'] = mod_event_data_df['time'].apply(lambda x: int(x) if not np.isnan(x) else x)

    print('\033[91m' +'Dataframe with POLARIS data created. \n', mod_event_data_df)

    return run_num, mod_event_data_df, mod_event_data

def apply_coordinate_transformations(run_num,mod_event_data_df):
    print('\033[92m' +'________________________________________')
    print('\033[92m' +'Applying transformations to convert data into isocentric coordinate system...')
    print('\033[92m' +'________________________________________', '\n')
    transformation_matrices = {}  # Creating transformation matrices
    transformation_matrices[0] = utils.get_transformation_matrix_array(transformation_dict[run_num])
    mod_event_data_df.drop(['detector'], inplace=True, axis=1)
    mod_event_data_df = mod_event_data_df[['scatters', 'energy', 'time', 'x', 'y', 'z']]
    mod_event_data_df['energy'] = mod_event_data_df['energy'] / 1e3  # convert to keV
    mod_event_data_df['x'] = mod_event_data_df['x'] / 1e3  # convert to mm
    mod_event_data_df['y'] = mod_event_data_df['y'] / 1e3  # convert to mm
    mod_event_data_df['z'] = mod_event_data_df['z'] / 1e3  # convert to mm
    mod_event_data_df['time'] = mod_event_data_df['time']*10  # convert to seconds

    print('\033[92m' +'Dataframe with POLARIS data created and coordinates transformed. \n', mod_event_data_df)

    return mod_event_data_df

def find_POLARIS_sync_pulses(mod_event_data, mod_event_data_df, data_dir):
    print('\033[93m' +'________________________________________') 
    print ('\033[93m' +'_____________ FIND SYNC PULSES IN POLARIS DATAFRAME _____________ ')
    print('\033[93m' +'________________________________________', '\n')
    mod_sync_pulses = mod_event_data.loc[mod_event_data['scatters'] == 122]
    mod_sync_pulses.drop(['z', 'energy', 'time', 'detector'], inplace=True, axis=1)
    mod_sync_pulses.columns = ['sync_flag', 'sync_index', 'sync_timestamp']
    mod_sync_pulses.reset_index(drop=True, inplace=True)
    mod_sync_pulses['sync_timestamp'] = mod_sync_pulses['sync_timestamp']*10
    mod_sync_pulses['sync_energy'] = mod_event_data['energy'].iloc[mod_sync_pulses['sync_index'] + 1]
    mod_sync_pulses_df = pd.DataFrame(mod_sync_pulses)
    mod_sync_pulses_df.reset_index(drop=True, inplace=True)
    #mod_sync_pulses_df.to_parquet(data_dir + 'syncPulsesPOLARIS.parquet')

    firstValidSyncPOLARIS = mod_sync_pulses_df.loc[mod_sync_pulses_df['sync_energy'].first_valid_index()]['sync_index']
    firstValidSyncPOLARIS_timestamp = mod_sync_pulses_df.loc[firstValidSyncPOLARIS - 1]['sync_timestamp']
    mod_sync_pulses_df = mod_sync_pulses_df.loc[mod_sync_pulses_df['sync_timestamp'] >= firstValidSyncPOLARIS_timestamp]
    mod_sync_pulses_df.reset_index(drop=True, inplace=True)

    print('\033[93m' +'____________  Sync Pulses POLARIS  ____________ \n', mod_sync_pulses_df)

    sync_time_diff_POLARIS = 200000006 - (mod_sync_pulses_df['sync_timestamp']).diff()
    mod_sync_pulses_df['sync_time_diff_POLARIS'] = sync_time_diff_POLARIS
    sync_time_diff_POLARIS = sync_time_diff_POLARIS[1:]
    print('\033[93m' +'First Valid Timestamp: ', firstValidSyncPOLARIS_timestamp,
          '\n First Valid Index: ', firstValidSyncPOLARIS,
          '\n Time Diff Between POLARIS Sync Pulses: \n', sync_time_diff_POLARIS.astype(float).round(17).astype(str),
          ' \n Average POLARIS sync difference: {} s +- {} s'.format(np.mean(mod_sync_pulses_df['sync_time_diff_POLARIS']),
                                                                       np.sqrt(
                                                                           np.var(mod_sync_pulses_df['sync_time_diff_POLARIS']))),
          '\n\n')

    mod_event_data_df = mod_event_data_df.loc[mod_event_data_df['time'] >= firstValidSyncPOLARIS_timestamp]
    mod_event_data_df.reset_index(drop=True, inplace=True)

    print('\033[93m' +'____________ POLARIS df  ____________ \n', mod_event_data_df)
    return mod_sync_pulses_df,firstValidSyncPOLARIS,mod_event_data_df

def read_LaBr_data(data_dir):
    print('\033[94m' + '________________________________________') 
    print('\033[94m' + '_____________ READ IN LaBr3:Ce DATA _____________')
    print('\033[94m' + '________________________________________', '\n')
    

    LaBr_file = sys.argv[2]
    LaBr_data = uproot.open(data_dir + LaBr_file)

    LaBr_tree = LaBr_data["LaBrData"]
    fastEnergyPOLARIS = "slowEPOLARIS"  
    fastTimePOLARIS = "slowTPOLARIS"
    fastEnergyL0 = "energyFastL0"
    fastTimeLaBr0 = "timeFL0" 
    slowEnergyLaBr0 = "slowECalibL0"
    slowTimeLaBr0 = "timeSL0"
    fastEnergyL1 = "energyFastL1"
    fastTimeLaBr1 = "timeFL1"
    slowEnergyLaBr1 = "slowECalibL1"
    slowTimeLaBr1 = "timeSL1"
    fastEnergyL2 = "energyFastL2"
    fastTimeLaBr2 = "timeFL2"
    slowEnergyLaBr2 = "slowECalibL2"
    slowTimeLaBr2 = "timeSL2"
    fastEnergyL3 = "energyFastL3"
    fastTimeLaBr3 = "timeFL3"
    slowEnergyLaBr3 = "slowECalibL3"
    slowTimeLaBr3 = "timeSL3"

    fastEnergyPOLARIS = LaBr_tree[fastEnergyPOLARIS].array()
    fastTimePOLARIS = LaBr_tree[fastTimePOLARIS].array()
    fastEnergyL0 = LaBr_tree[fastEnergyL0].array()
    fastTimeLaBr0 = LaBr_tree[fastTimeLaBr0].array()
    slowEnergyLaBr0 = LaBr_tree[slowEnergyLaBr0].array()
    slowTimeLaBr0 = LaBr_tree[slowTimeLaBr0].array()
    fastEnergyL1 = LaBr_tree[fastEnergyL1].array()
    fastTimeLaBr1 = LaBr_tree[fastTimeLaBr1].array()
    slowEnergyLaBr1 = LaBr_tree[slowEnergyLaBr1].array()
    slowTimeLaBr1 = LaBr_tree[slowTimeLaBr1].array()
    fastEnergyL2 = LaBr_tree[fastEnergyL2].array()
    fastTimeLaBr2 = LaBr_tree[fastTimeLaBr2].array()
    slowEnergyLaBr2 = LaBr_tree[slowEnergyLaBr2].array()
    slowTimeLaBr2 = LaBr_tree[slowTimeLaBr2].array()
    fastEnergyL3 = LaBr_tree[fastEnergyL3].array()
    fastTimeLaBr3 = LaBr_tree[fastTimeLaBr3].array()
    slowEnergyLaBr3 = LaBr_tree[slowEnergyLaBr3].array()
    slowTimeLaBr3 = LaBr_tree[slowTimeLaBr3].array()

    LaBr_data_df = pd.DataFrame({'fastEnergyPOLARIS': fastEnergyPOLARIS, 'fastTimePOLARIS': fastTimePOLARIS, 'fastEnergyL0': fastEnergyL0,
                                 'fastTimeLaBr0': fastTimeLaBr0, 'slowEnergyLaBr0': slowEnergyLaBr0, 'slowTimeLaBr0': slowTimeLaBr0, 'fastEnergyL1': fastEnergyL1,
                                 'fastTimeLaBr1': fastTimeLaBr1, 'slowEnergyLaBr1': slowEnergyLaBr1, 'slowTimeLaBr1': slowTimeLaBr1, 'fastEnergyL2': fastEnergyL2,
                                 'fastTimeLaBr2': fastTimeLaBr2, 'slowEnergyLaBr2': slowEnergyLaBr2, 'slowTimeLaBr2': slowTimeLaBr2, 'fastEnergyL3': fastEnergyL3,
                                 'fastTimeLaBr3': fastTimeLaBr3, 'slowEnergyLaBr3': slowEnergyLaBr3, 'slowTimeLaBr3': slowTimeLaBr3})
    LaBr_data_df.reset_index(drop=True, inplace=True)
    #divide 'fastTimePOLARIS', 'fastTimeLaBr0', 'fastTimeLaBr1', 'fastTimeLaBr2', 'fastTimeLaBr3', 'slowTimeLaBr0', 'slowTimeLaBr1', 'slowTimeLaBr2', 'slowTimeLaBr3' by 100
    LaBr_data_df['fastTimePOLARIS'] = LaBr_data_df['fastTimePOLARIS']
    LaBr_data_df['fastTimeLaBr0'] = LaBr_data_df['fastTimeLaBr0']/10
    LaBr_data_df['fastTimeLaBr1'] = LaBr_data_df['fastTimeLaBr1']/10
    LaBr_data_df['fastTimeLaBr2'] = LaBr_data_df['fastTimeLaBr2']/10
    LaBr_data_df['fastTimeLaBr3'] = LaBr_data_df['fastTimeLaBr3']/10
    LaBr_data_df['slowTimeLaBr0'] = LaBr_data_df['slowTimeLaBr0']/10
    LaBr_data_df['slowTimeLaBr1'] = LaBr_data_df['slowTimeLaBr1']/10
    LaBr_data_df['slowTimeLaBr2'] = LaBr_data_df['slowTimeLaBr2']/10
    LaBr_data_df['slowTimeLaBr3'] = LaBr_data_df['slowTimeLaBr3']/10

    LaBr_data_df = LaBr_data_df.sort_values(by=['fastTimePOLARIS', 'fastTimeLaBr0', 'fastTimeLaBr1', 'fastTimeLaBr2', 'fastTimeLaBr3', 'slowTimeLaBr0', 'slowTimeLaBr1', 'slowTimeLaBr2', 'slowTimeLaBr3'], ascending=True)
    LaBr_data_df.reset_index(drop=True, inplace=True)
    print('\033[94m' +'____________ LaBr3:Ce df  ____________ \n', LaBr_data_df)

    # pull out first_time_L0 as the first non-zero time in the fastTimeLaBr0 column
    first_time_L0 = LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr0'] != 0.0].iloc[0]['fastTimeLaBr0']
    # pull out last_time_L0 as the last non-zero time in the fastTimeLaBr0 column
    last_time_L0 = LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr0'] != 0.0].iloc[-1]['fastTimeLaBr0']

    print('Time duration LaBr0: ', (last_time_L0 - first_time_L0)*1e-9/60, 'minutes')

    fast_time_dff = LaBr_data_df['fastTimeLaBr0'].diff()
    slow_time_dff = LaBr_data_df['slowTimeLaBr0'].diff()

    # drop slowtime columns 
    LaBr_data_df = LaBr_data_df.drop(['slowTimeLaBr0', 'slowTimeLaBr1', 'slowTimeLaBr2', 'slowTimeLaBr3'], axis=1)
    LaBr_data_df = LaBr_data_df.drop(['fastEnergyL0', 'fastEnergyL1', 'fastEnergyL2', 'fastEnergyL3'], axis=1)

    # plot a scatter of energyPOLARIS vs fastTimePOLARIS
    # fig = plt.figure()
    # plt.scatter(LaBr_data_df['fastTimePOLARIS'], LaBr_data_df['fastEnergyPOLARIS'], s=1)
    # plt.xlabel('Time (ns)', fontsize=25)
    # plt.ylabel('Energy (keV)', fontsize=25)
    # plt.title('Energy vs Time for POLARIS')
    # plt.show()

    return LaBr_data_df, fast_time_dff, slow_time_dff

def find_LaBr3_sync_pulses(firstValidSyncPOLARIS,LaBr_data_df,mod_sync_pulses_df):
    print('\033[95m' + '__________________________________________________________________')
    print ('\033[95m' + '_____________ FIND SYNC PULSES IN LaBr3:Ce DATAFRAME _____________ ')
    print('\033[95m' + '__________________________________________________________________', '\n')

    sync_pulses_LaBr_df = pd.DataFrame(LaBr_data_df)
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.loc[sync_pulses_LaBr_df['fastTimePOLARIS'] != 0.0]
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.loc[sync_pulses_LaBr_df['fastEnergyPOLARIS'] != 0.0]
    #sync_pulses_LaBr_df['fastEnergyPOLARIS'] = sync_pulses_LaBr_df['fastEnergyPOLARIS'].astype(int)
    #sync_pulses_LaBr_df['fastTimePOLARIS'] = sync_pulses_LaBr_df['fastTimePOLARIS'].astype(int)
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.loc[(sync_pulses_LaBr_df['fastEnergyPOLARIS'].isin(sync_pulses_LaBr_df['fastEnergyPOLARIS'].unique())) & (sync_pulses_LaBr_df['fastTimePOLARIS'].isin(sync_pulses_LaBr_df['fastTimePOLARIS'].unique()))]
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.loc[sync_pulses_LaBr_df['fastTimePOLARIS'].isin(sync_pulses_LaBr_df['fastTimePOLARIS'].unique())]
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.loc[sync_pulses_LaBr_df['fastEnergyPOLARIS'].isin(sync_pulses_LaBr_df['fastEnergyPOLARIS'].unique())]
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.drop_duplicates(subset = ['fastEnergyPOLARIS'], keep = 'first')
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.drop_duplicates(subset = ['fastTimePOLARIS'], keep = 'first')
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.drop_duplicates(subset = ['fastEnergyPOLARIS', 'fastTimePOLARIS'], keep = 'first')

    sync_pulses_LaBr_df['synctimediff'] = sync_pulses_LaBr_df['fastTimePOLARIS'].diff()
    print(sync_pulses_LaBr_df['synctimediff']/200000006)
    #sync_pulses_LaBr_df = sync_pulses_LaBr_df.loc[sync_pulses_LaBr_df['synctimediff']/200000006 >0.187]

    # fig = plt.figure()
    # plt.plot(sync_pulses_LaBr_df['synctimediff']/200000006)
    # plt.xlabel('Sync Pulse Time diff ratio', fontsize=25)
    # plt.ylabel('Counts', fontsize=25)
    # plt.title('Time Difference Between Consecutive Sync Pulses')
    # plt.show()
    # sync_pulses_LaBr_df = sync_pulses_LaBr_df.loc[(sync_pulses_LaBr_df['synctimediff']/200000006 >0.9)  & (sync_pulses_LaBr_df['synctimediff']/200000006 <1.6)]
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.drop(['synctimediff'], axis=1)

    print("time duration sync pulses", (sync_pulses_LaBr_df['fastTimePOLARIS'].iloc[-1] - sync_pulses_LaBr_df['fastTimePOLARIS'].iloc[0])*1e-9/60, "minutes")

    print('\033[95m' +'Number of LaBr sync pulses\n', len(sync_pulses_LaBr_df))
    print('\033[95m' +'Discrepancy between number of sync pulses LaBr3:Ce and POLARIS\n', len(sync_pulses_LaBr_df)-len(mod_sync_pulses_df))

    print("time duration sync pulses after dropping ", (sync_pulses_LaBr_df['fastTimePOLARIS'].iloc[-1] - sync_pulses_LaBr_df['fastTimePOLARIS'].iloc[0])*1e-9/60, "minutes")

    sync_pulses_LaBr_df.insert(0, 'sync_flag', 122)
    syncTimeDiff = sync_pulses_LaBr_df['fastTimePOLARIS'].diff()
    sync_pulses_LaBr_df['syncTimeDiff'] = syncTimeDiff
    print ('\033[95m' + 'syncTimeDiff \n', syncTimeDiff/200000006)
    firstValidSyncPOLARIS = int(firstValidSyncPOLARIS)
    # cut firstValidSyncPOLARIS-1 sync pulses from the beginning of the dataframe
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.iloc[firstValidSyncPOLARIS-1:]
    firstValidSyncLaBr = sync_pulses_LaBr_df['fastTimePOLARIS'].iloc[0]
    excess_LaBr_sync_pulses = len(sync_pulses_LaBr_df)-len(mod_sync_pulses_df)
    sync_pulses_LaBr_df = sync_pulses_LaBr_df.iloc[:len(sync_pulses_LaBr_df) - excess_LaBr_sync_pulses]  # drop excess sync pulses from the end of the dataframe
    print('\033[95m' +'Number of LaBr sync pulses after dropping excess\n', len(sync_pulses_LaBr_df), 
          "\n sync pulses df: \n", sync_pulses_LaBr_df)
    sync_pulses_LaBr_df = sync_pulses_LaBr_df[['sync_flag', 'syncTimeDiff', 'fastTimePOLARIS', 'fastEnergyPOLARIS', 'fastTimeLaBr0', 'slowEnergyLaBr0', 'fastTimeLaBr1', 'slowEnergyLaBr1', 'fastTimeLaBr2', 'slowEnergyLaBr2', 'fastTimeLaBr3', 'slowEnergyLaBr3']]

    return sync_pulses_LaBr_df,firstValidSyncLaBr
    
def time_walk_correction(data_dir,mod_event_data_df,LaBr_data_df,sync_pulses_LaBr_df,firstValidSyncLaBr,mod_sync_pulses_df, fast_time_dff, slow_time_dff):  
    
    polaris_time_dff = mod_event_data_df['time'].diff()
    polaris_time_dff = pd.DataFrame(polaris_time_dff/1000000) # microseconds
    slow_time_dff = pd.DataFrame(slow_time_dff/1000000) # microseconds
    fast_time_dff = pd.DataFrame(fast_time_dff/1000000) # microseconds
    # drop the first row of each dataframe
    polaris_time_dff = polaris_time_dff.iloc[1:]
    slow_time_dff = slow_time_dff.iloc[1:]
    fast_time_dff = fast_time_dff.iloc[1:]

    print(polaris_time_dff, slow_time_dff, fast_time_dff)

    fig = plt.figure()
    plt.plot(polaris_time_dff, label = 'POLARIS')
    plt.xlabel('Time difference ($\mu$s)', fontsize = 25)
    plt.ylabel('Counts', fontsize = 25)
    plt.title('Time Difference Between Consecutive Events')
    plt.legend(loc = 'upper right', fontsize = 30)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
    plt.tick_params(axis = 'both', which = 'minor', labelsize = 20)

    fig2 = plt.figure()
    plt.plot(slow_time_dff, label = 'LaBr$_{3}$ Slow Signal')
    plt.xlabel('Time difference ($\mu$s)', fontsize = 25)
    plt.ylabel('Counts', fontsize = 25)
    plt.title('Time Difference Between Consecutive Events')
    plt.legend(loc = 'upper right', fontsize = 30)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
    plt.tick_params(axis = 'both', which = 'minor', labelsize = 20)

    fig3 = plt.figure()
    plt.plot(fast_time_dff, label = 'LaBr$_{3}$ Fast Signal')
    plt.xlabel('Time difference ($\mu$s)', fontsize = 25)
    plt.ylabel('Counts', fontsize = 25)
    plt.title('Time Difference Between Consecutive Events')
    plt.legend(loc = 'upper right', fontsize = 30)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 20)
    plt.tick_params(axis = 'both', which = 'minor', labelsize = 20)



    # xxx = root.TCanvas("xxx", "Time Diff Cons", 800, 600)
    # xxx.cd()
    # h1 = root.TH1F("h1", "POLARIS", 100000, 0, 10000)
    # h1.GetXaxis().SetTitle("Time Difference (us)")
    # h1.GetYaxis().SetTitle("Counts")
    # h1.SetLineColor(root.kBlue)
    # h1.SetLineWidth(2)
    # h1.SetStats(0)
    # for i in polaris_time_dff['time']:
    #     h1.Fill(i)
    # h2 = root.TH1F("h2", "LaBr_{3}:Ce Slow Signal", 100000, 0, 10000)
    # for i in slow_time_dff['slowTimeLaBr0']:
    #     h2.Fill(i)
    # h2.SetLineColor(root.kRed)
    # h2.SetLineWidth(2)
    # h2.SetStats(0)
    # h3 = root.TH1F("h3", "LaBr_{3}:Ce Fast Signal", 100000, 0, 10000)
    # for i in fast_time_dff['fastTimeLaBr0']:
    #     h3.Fill(i)
    # h3.SetLineColor(root.kGreen)
    # h3.SetLineWidth(2)
    # h3.SetStats(0)
    # h1.Draw()
    # h2.Draw("same")
    # h3.Draw("same")
    # legend = root.TLegend(0.7, 0.7, 0.9, 0.9)
    # legend.AddEntry(h1, "POLARIS", "l")
    # legend.AddEntry(h2, "LaBr_{3}(ce) Slow Signal", "l")
    # legend.AddEntry(h3, "LaBr_{3}(ce) Fast Signal", "l")
    # legend.Draw()
    # xxx.Draw()
    # xxx.SaveAs(data_dir + "plots/time_diff_cons.root")



    
    print('\033[96m' + '________________________________________') 
    print('\033[96m' + '_____________ TIME WALK CORRECTION _____________')
    print('\033[96m' + '________________________________________', '\n')  
    LaBr_data_df.insert(0, 'sync_flag', 0)
    LaBr_data_df.insert(1, 'syncTimeDiff', 0) 
    LaBr_data_df.reset_index(drop = True, inplace = True)
    LaBr_data_df = pd.concat([LaBr_data_df, sync_pulses_LaBr_df], ignore_index = False)
    LaBr_data_df.sort_values(by = ['fastTimePOLARIS', 'fastTimeLaBr0', 'fastTimeLaBr1', 'fastTimeLaBr2', 'fastTimeLaBr3'], inplace = True)
    LaBr_data_df.reset_index(drop = True, inplace = True)

    sync_pulses_LaBr_df = sync_pulses_LaBr_df.drop(['sync_flag', 'syncTimeDiff', 'fastEnergyPOLARIS', 'fastTimeLaBr0', 'slowEnergyLaBr0', 'fastTimeLaBr1', 'slowEnergyLaBr1', 'fastTimeLaBr2', 'slowEnergyLaBr2', 'fastTimeLaBr3', 'slowEnergyLaBr3'], axis = 1)
    sync_pulses_LaBr_df.reset_index(drop = True, inplace = True)    
    sync_pulses_LaBr_df = pd.concat([sync_pulses_LaBr_df['fastTimePOLARIS'], mod_sync_pulses_df['sync_timestamp']], axis = 1)    
    sync_pulses_LaBr_df.columns = ['LaBr3_time', 'POLARIS_time']
    sync_pulses_LaBr_df['index'] = LaBr_data_df[LaBr_data_df['sync_flag'] == 122].index 
    sync_pulses_LaBr_df.set_index('index', inplace=True)

    print('\033[96m' +'Number of sync pulses in LaBr3:Ce data: ', len(LaBr_data_df[LaBr_data_df['sync_flag'] == 122]))
    print('\033[96m' +'____________ LaBr3:Ce df  ____________ \n', LaBr_data_df[LaBr_data_df['sync_flag'] == 122])
   
    # LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr0'] = ((LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr0'] / LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 122, 'syncTimeDiff'].iloc[-1]))* 200000006 # scale the fastTimeLaBr0 column by the last syncTimeDiff value
    # LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr1'] = ((LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr1'] / LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 122, 'syncTimeDiff'].iloc[-1]))* 200000006 # scale the fastTimeLaBr1 column by the last syncTimeDiff value
    # LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr2'] = ((LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr2'] / LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 122, 'syncTimeDiff'].iloc[-1]))* 200000006 # scale the fastTimeLaBr2 column by the last syncTimeDiff value
    # LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr3'] = ((LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr3'] / LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 122, 'syncTimeDiff'].iloc[-1]))* 200000006 # scale the fastTimeLaBr3 column by the last syncTimeDiff value
    LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr0'] = ((LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr0'] / LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 122, 'syncTimeDiff'].iloc[-1]))* 200000006 # scale the fastTimeLaBr0 column by the last syncTimeDiff value
    LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr1'] = ((LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr1'] / LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 122, 'syncTimeDiff'].iloc[-1]))* 200000006 # scale the fastTimeLaBr1 column by the last syncTimeDiff value
    LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr2'] = ((LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr2'] / LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 122, 'syncTimeDiff'].iloc[-1]))* 200000006 # scale the fastTimeLaBr2 column by the last syncTimeDiff value
    LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr3'] = ((LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 0, 'fastTimeLaBr3'] / LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 122, 'syncTimeDiff'].iloc[-1]))* 200000006 # scale the fastTimeLaBr3 column by the last syncTimeDiff value
    LaBr_data_df.reset_index(drop = True, inplace = True)

    LaBr_data_df['sync_flag'] = LaBr_data_df['sync_flag'].astype(int)
    LaBr_data_df['timeDiffL0'] = LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr0'] > 0, 'fastTimeLaBr0'].diff()
    LaBr_data_df['timeDiffL1'] = LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr1'] > 0, 'fastTimeLaBr1'].diff()
    LaBr_data_df['timeDiffL2'] = LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr2'] > 0, 'fastTimeLaBr2'].diff()
    LaBr_data_df['timeDiffL3'] = LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr3'] > 0, 'fastTimeLaBr3'].diff()
    LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr0'] <= 0, 'timeDiffL0'] = 0
    LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr1'] <= 0, 'timeDiffL1'] = 0
    LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr2'] <= 0, 'timeDiffL2'] = 0
    LaBr_data_df.loc[LaBr_data_df['fastTimeLaBr3'] <= 0, 'timeDiffL3'] = 0

    print('\033[96m' +'____________ LaBr3:Ce df  ____________ \n', LaBr_data_df[['timeDiffL0', 'timeDiffL1', 'timeDiffL2', 'timeDiffL3']])

    index_list = LaBr_data_df[LaBr_data_df['sync_flag'] == 122].index.to_list()
    len_index_list = len(index_list)

    print('\033[96m' +'____________ sync_pulses_LaBr_df ____________ \n', sync_pulses_LaBr_df)

    for index in sync_pulses_LaBr_df.index:
        LaBr_data_df.loc[index, 'fastTimeLaBr0'] = sync_pulses_LaBr_df.loc[index,'POLARIS_time'] # replace fastTimeLaBr0 with POLARIS_Sync_Time
        LaBr_data_df.loc[index, 'fastTimeLaBr1'] = sync_pulses_LaBr_df.loc[index,'POLARIS_time'] # replace fastTimeLaBr1 with POLARIS_Sync_Time
        LaBr_data_df.loc[index, 'fastTimeLaBr2'] = sync_pulses_LaBr_df.loc[index,'POLARIS_time'] # replace fastTimeLaBr2 with POLARIS_Sync_Time
        LaBr_data_df.loc[index, 'fastTimeLaBr3'] = sync_pulses_LaBr_df.loc[index,'POLARIS_time'] # replace fastTimeLaBr3 with POLARIS_Sync_Time

    for i in range(len_index_list):
        if i < len_index_list - 1:
            LaBr_data_df.loc[index_list[i]:index_list[i+1]-1, 'fastTimeLaBr0'] = LaBr_data_df.loc[index_list[i], 'fastTimeLaBr0'] + LaBr_data_df.loc[index_list[i]:index_list[i+1]-1, 'timeDiffL0'].cumsum() # 
            LaBr_data_df.loc[index_list[i]:index_list[i+1]-1, 'fastTimeLaBr1'] = LaBr_data_df.loc[index_list[i], 'fastTimeLaBr1'] + LaBr_data_df.loc[index_list[i]:index_list[i+1]-1, 'timeDiffL1'].cumsum()
            LaBr_data_df.loc[index_list[i]:index_list[i+1]-1, 'fastTimeLaBr2'] = LaBr_data_df.loc[index_list[i], 'fastTimeLaBr2'] + LaBr_data_df.loc[index_list[i]:index_list[i+1]-1, 'timeDiffL2'].cumsum()
            LaBr_data_df.loc[index_list[i]:index_list[i+1]-1, 'fastTimeLaBr3'] = LaBr_data_df.loc[index_list[i], 'fastTimeLaBr3'] + LaBr_data_df.loc[index_list[i]:index_list[i+1]-1, 'timeDiffL3'].cumsum()
        else:
            LaBr_data_df.loc[index_list[i]:, 'fastTimeLaBr0'] = LaBr_data_df.loc[index_list[i], 'fastTimeLaBr0'] + LaBr_data_df.loc[index_list[i]:, 'timeDiffL0'].cumsum()
            LaBr_data_df.loc[index_list[i]:, 'fastTimeLaBr1'] = LaBr_data_df.loc[index_list[i], 'fastTimeLaBr1'] + LaBr_data_df.loc[index_list[i]:, 'timeDiffL1'].cumsum()
            LaBr_data_df.loc[index_list[i]:, 'fastTimeLaBr2'] = LaBr_data_df.loc[index_list[i], 'fastTimeLaBr2'] + LaBr_data_df.loc[index_list[i]:, 'timeDiffL2'].cumsum()
            LaBr_data_df.loc[index_list[i]:, 'fastTimeLaBr3'] = LaBr_data_df.loc[index_list[i], 'fastTimeLaBr3'] + LaBr_data_df.loc[index_list[i]:, 'timeDiffL3'].cumsum()

    print('\033[96m' +'Ratio of sync pulses in LaBr_data_df/sync_pulses_LaBr_df \n', len(LaBr_data_df.loc[LaBr_data_df['sync_flag'] == 122])/len(sync_pulses_LaBr_df))  
    

    LaBr_data_df.reset_index(drop = True, inplace = True)
    LaBr_data_df = LaBr_data_df.loc[LaBr_data_df.index <= sync_pulses_LaBr_df.index[-1]] # remove all data after the last sync pulse 
    LaBr_data_df = LaBr_data_df.loc[LaBr_data_df.index >= sync_pulses_LaBr_df.index[0]] # remove all data before the first sync pulse
    LaBr_data_df = LaBr_data_df.loc[LaBr_data_df['sync_flag'] != 122]
    LaBr_data_df.drop(['sync_flag', 'syncTimeDiff', 'timeDiffL0', 'timeDiffL1', 'timeDiffL2', 'timeDiffL3', 'fastTimePOLARIS', 'fastEnergyPOLARIS'], axis = 1, inplace = True)
    LaBr_data_df.reset_index(drop=True, inplace=True)
    print('\033[96m' +'____________ LaBr3:Ce df time walk corrected ____________ \n', LaBr_data_df)

    return LaBr_data_df

def merge_two_detector_dataframes(data_dir,mod_event_data_df,LaBr_data_df):
    print('\033[89m' + '________________________________________') 
    print('\033[89m' + '_____________ MERGE POLARIS AND LaBr3:Ce DATAFRAMES _____________')
    print('\033[89m' + '________________________________________', '\n')

    mod_event_data_df = mod_event_data_df.loc[mod_event_data_df['scatters'] != 122]
    mod_event_data_df.reset_index(drop = True, inplace = True)
    mod_event_data_df.drop(['scatters'], axis = 1, inplace = True)
    mod_event_data_df.rename(columns = {'energy': 'energyPOLARIS'}, inplace = True)
    mod_event_data_df.sort_values(by = ['time'], inplace = True)
    mod_event_data_df.reset_index(drop = True, inplace = True)

    df0 = LaBr_data_df[['fastTimeLaBr0', 'slowEnergyLaBr0']]
    df0.columns = ['time', 'energyL0']
    df0['energyL1'] = np.nan
    df0['energyL2'] = np.nan
    df0['energyL3'] = np.nan
    df0 = df0[['time', 'energyL0', 'energyL1', 'energyL2', 'energyL3']]
    df0.sort_values(by = ['time'], inplace = True)
    df1 = LaBr_data_df[['fastTimeLaBr1', 'slowEnergyLaBr1']]
    df1.columns = ['time', 'energyL1']
    df1['energyL0'] = np.nan
    df1['energyL2'] = np.nan
    df1['energyL3'] = np.nan
    df1 = df1[['time', 'energyL0', 'energyL1', 'energyL2', 'energyL3']]
    df1.sort_values(by = ['time'], inplace = True)
    df2 = LaBr_data_df[['fastTimeLaBr2', 'slowEnergyLaBr2']]
    df2.columns = ['time', 'energyL2']
    df2['energyL0'] = np.nan
    df2['energyL1'] = np.nan
    df2['energyL3'] = np.nan
    df2 = df2[['time', 'energyL0', 'energyL1', 'energyL2', 'energyL3']]
    df2.sort_values(by = ['time'], inplace = True)
    df3 = LaBr_data_df[['fastTimeLaBr3', 'slowEnergyLaBr3']]
    df3.columns = ['time', 'energyL3']
    df3['energyL0'] = np.nan
    df3['energyL1'] = np.nan
    df3['energyL2'] = np.nan
    df3 = df3[['time', 'energyL0', 'energyL1', 'energyL2', 'energyL3']]
    df3.sort_values(by = ['time'], inplace = True)

    print('\033[89m' +'____________ df0  ____________ \n', df0)
    print('\033[89m' +'____________ df1  ____________ \n', df1)
    print('\033[89m' +'____________ df2  ____________ \n', df2)
    print('\033[89m' +'____________ df3  ____________ \n', df3)

    df_final = pd.concat([df0, df1, df2, df3], ignore_index = True, axis=0) # the dataframes are concatenated vertically

    df_final = df_final[df_final['time'] > 1e9]
    df_final = df_final[df_final['time'] < 1e16]
    df_final.reset_index(drop = True, inplace = True)
    df_final.sort_values(by = ['time'], inplace = True)
    df_final.reset_index(drop = True, inplace = True)
    print('\033[89m' +'____________ df_final  ____________ \n', df_final)

    df_final['energyPOLARIS'] = np.nan
    df_final['x'] = np.nan
    df_final['y'] = np.nan
    df_final['z'] = np.nan
    df_final= df_final[['time', 'energyPOLARIS' , 'energyL0', 'energyL1', 'energyL2', 'energyL3', 'x', 'y', 'z']] # rearrange the columns
    mod_event_data_df = mod_event_data_df[['time', 'energyPOLARIS' , 'x', 'y', 'z']]
    print('\033[89m' +'____________ POLARIS df  ____________ \n', mod_event_data_df)
    print('\033[89m' +'____________ new LaBr3:Ce df  ____________ \n', df_final)
    mod_event_data_df['energyL0'] = np.nan
    mod_event_data_df['energyL1'] = np.nan
    mod_event_data_df['energyL2'] = np.nan
    mod_event_data_df['energyL3'] = np.nan
    mod_event_data_df = mod_event_data_df[['time', 'energyPOLARIS' , 'energyL0', 'energyL1', 'energyL2', 'energyL3', 'x', 'y', 'z']]
    # divide time by 10
    mod_event_data_df['time'] = mod_event_data_df['time'] #ns
    df_final['time'] = df_final['time'] #ns 

    # subtract 300 us from mod_event_data_df['time']
    mod_event_data_df['time'] = mod_event_data_df['time'] #ns 
    #merge the two dataframes
    merged_df = pd.concat([df_final, mod_event_data_df], ignore_index = True, axis=0) # the dataframes are concatenated vertically
    merged_df.sort_values(by = ['time'], inplace = True)
    merged_df.reset_index(drop = True, inplace = True)
    merged_df.to_parquet(data_dir + 'merged_df.parquet', engine = 'pyarrow')
    # columns = ['time', 'energyPOLARIS', 'energyL0', 'energyL1', 'energyL2', 'energyL3', 'x', 'y', 'z']

    print('\033[89m' +'____________ merged_df  ____________ \n', merged_df)

    run_time_exp = (merged_df.iloc[-1]['time'] - merged_df.iloc[0]['time']) * 1e-9
    run_time_exp = run_time_exp / 60
    print('\033[89m' +'Run time of experiment: {} minutes'.format(run_time_exp)) 

    profile = ProfileReport(merged_df, title="Profiling Report")
    import subprocess
    profile.to_file("profiling_report.html")
    html_file = 'profiling_report.html'
    subprocess.call(['cp', html_file, data_dir])

    return merged_df







# ______________________________________________________________________________________________________________________
def main():
    data_dir = sys.argv[1]
    run_num, mod_event_data_df,mod_event_data = read_polaris_data(data_dir)
    LaBr_data_df,fast_time_dff, slow_time_dff = read_LaBr_data(data_dir) 

    mod_event_data_df = apply_coordinate_transformations(run_num, mod_event_data_df)
    mod_sync_pulses_df,firstValidSyncPOLARIS,mod_event_data_df = find_POLARIS_sync_pulses(mod_event_data,mod_event_data_df, data_dir)
    sync_pulses_LaBr_df,firstValidSyncLaBr = find_LaBr3_sync_pulses(firstValidSyncPOLARIS,LaBr_data_df,mod_sync_pulses_df)
    LaBr_data_df = time_walk_correction(data_dir, mod_event_data_df, LaBr_data_df, sync_pulses_LaBr_df,firstValidSyncLaBr,mod_sync_pulses_df,fast_time_dff, slow_time_dff)
    merged_df = merge_two_detector_dataframes(data_dir,mod_event_data_df,LaBr_data_df)


if __name__ == '__main__':
    main()

