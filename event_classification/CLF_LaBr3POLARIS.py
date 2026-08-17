################################################################################
#                               to run:                                        #
# python3 CLF_LaBrPOLARIS.py /directory/to/merged_df/runX_mergedLaBrPOLARISdata.parquet            #
__author__ = "Shanyn Hart"
__date__ = "2022-08-24"
__version__ = "1.0"
                                            #
################################################################################
#----
# PYTHON IMPORT STATEMENTS
#----
from ctypes import sizeof
from operator import index
from tokenize import Double
from unittest import skip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re, sys, os
import time as time
#os.environ["OPENBLAS_NUM_THREADS"] = "300"
import cython as cy
# if changes are made to CLF_LaBr3POLARIS_utils.pyx, you must run the following command in the terminal:
# python3 setup.py build_ext --inplace
import CLF_LaBr3POLARIS_utils as utils #type: ignore


start_time = time.time()

#----
# ********************* READ IN FILE ******************************
#----
data_dir = sys.argv[1]

run_num = str(re.search('run(\d+)', data_dir).group(1))
data_df = pd.read_parquet(data_dir + 'run'+run_num+'_mergedLaBrPOLARISdata.parquet')
run_num = int(run_num)
data_df = data_df[['detectorID','energy', 'time','x', 'y', 'z']]
data_df = data_df.loc[(data_df['detectorID'] == 5) | (data_df['detectorID'] == 0)]
data_df.reset_index(inplace=True, drop=True)

# ______________________________________________________

save_dir = data_dir + '/plots/' + 'CLFplots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print ('Creating directory: {}'.format(save_dir))
# ______________________________________________________

# Find the indices of rows where detectorID == 5
# detector_5_indices = data_df[data_df['detectorID'] == 5].index

# # Calculate the number of detectorID == 0 events between each detectorID == 5 event
# event_counts = detector_5_indices.to_series().diff() - 1
# event_counts = event_counts[event_counts.notnull()].astype(int).tolist()

# Create a histogram
# plt.hist(event_counts, bins=20, edgecolor='k')
# plt.xlabel('No. LaBr3 events between POLARIS events')
# plt.ylabel('Frequency')
# plt.title('Histogram of the number of LaBr3 events between POLARIS events')
# plt.grid(True)
# plt.xticks(np.arange(0, max(event_counts)+1, 1.0))
# plt.savefig(save_dir + 'LaBr3_events_between_POLARIS_events' + '.png')
# plt.show()

# print the ratio of the number of events in the LaBr3 detector to the number of events in the POLARIS detector
print('Ratio percentage of the number of events in the LaBr3 detector to POLARIS detector: ', (len(data_df.loc[data_df['detectorID'] == 0])/len(data_df.loc[data_df['detectorID'] == 5]) * 100), '%')

#data_df['time'] = data_df['time']*10e-9 # seconds

# plt.figure(figsize=(30, 15))
# plt.hist(data_df['energyPOLARIS'], bins=5500, range=(0, 5500), label='POLARIS', histtype='step', color = 'royalblue')
# plt.hist(data_df['energyL0'], bins=5500, range=(0, 5500), label='LaBr3', histtype='step', color='darkorange')
# plt.xlabel('Energy (keV)', fontsize=30)
# plt.yscale('log')
# plt.ylabel('Counts', fontsize=30)
# plt.legend(loc = 'upper right', fontsize=30)
# #make the tick labels larger
# plt.tick_params(axis='both', which='major', labelsize=30)
# plt.grid(axis='both', linestyle='--', linewidth=0.5)
# plt.savefig(save_dir + 'POLARIS_energyL0_Histograms_BEFORECOINCIDENCE' + '.png', dpi=300)

#______________________________________________________
#Plot energy for POLARIS and LaBr3

plt.figure(figsize=(30, 15))
plt.hist(data_df.loc[data_df['detectorID'] == 5, 'energy'], bins=1500, range=(0, 1500), label='POLARIS', histtype='step', color = 'royalblue')
plt.hist(data_df.loc[data_df['detectorID'] == 0, 'energy'], bins=1500, range=(0, 1500), label='LaBr3', histtype='step', color='darkorange')
plt.xlabel('Energy (keV)', fontsize=30)
plt.ylabel('Counts', fontsize=30)
plt.legend(loc = 'upper right', fontsize=30)
#make the tick labels larger
plt.tick_params(axis='both', which='major', labelsize=30)
plt.grid(axis='both', linestyle='--', linewidth=0.5)
plt.yscale('log')
plt.savefig(save_dir + 'POLARIS_energyL0_Histograms_BEFORECOINCIDENCE' + '.png', dpi=300)

#______________________________________________________
# ********************* COINCIDENCE ******************************
data_df.loc[data_df['detectorID'] == 5, 'time'] = data_df.loc[data_df['detectorID'] == 5, 'time']# add 300 microseconds to the time of the POLARIS event
data_df = data_df.sort_values(by=['time'], ascending=True, ignore_index=True)
data_df.reset_index(inplace=True, drop=True)

data_df['Window_ID'] = np.nan
# every time we have an entry for POLARIS, we set the Window_ID to 1. It is zero for LaBr3.
data_df.loc[(data_df['detectorID'] == 5), 'Window_ID'] = 1 
# We then assign each POLARIS event a unique Window_ID by cumulatively summing the Window_ID column
data_df['Window_ID'] = data_df['Window_ID'].cumsum()
# We now back fill nan values with the previous Window_ID value so that the LaBr3 events have the same Window_ID as the POLARIS event in a coincidence window
data_df['Window_ID'] = data_df['Window_ID'].fillna(method='bfill')


# We then take the difference in time between each event backwards in time
data_df['TimeDiff'] = (data_df['time'] - data_df.shift(1)['time'])
# We then multiply the TimeDiff by -1 so that the time difference is not negative
# data_df['TimeDiff'] = data_df['TimeDiff'] * -1
# We then let the time difference of POLARIS be zero since it is the first event in the coincidence window
data_df.loc[data_df['detectorID'] == 5, 'TimeDiff'] = 0
# We then take the cumulative sum of the time difference in the coincidence window to see the time difference between the POLARIS event and the LaBr3 event
data_df['TimeDiffCum'] = data_df.groupby('Window_ID')['TimeDiff'].transform(lambda x: x.cumsum())

#______________________________________________________
rows_only_POLARIS = int(len(data_df.loc[data_df['detectorID'] == 5]))
#______________________________________________________
data_df.rename(columns={'energy':'energyPOLARIS'}, inplace=True)
data_df['energyL0'] = np.nan
data_df['energyL0'] = data_df.loc[data_df['detectorID'] == 0, 'energyPOLARIS']
data_df['energyL0'] = data_df['energyL0'].fillna(method='bfill')
data_df = data_df.loc[data_df['detectorID'] == 5]

data_df = data_df.drop(columns=['detectorID'])
data_df = data_df.dropna(subset=['energyPOLARIS', 'energyL0'])
# drop rows where detectorID == 0

Coincidence_df = data_df.copy(deep = True)
Coincidence_df.reset_index(inplace=True, drop=True)
Coincidence_df['theta1']=0
# reorder columns to be time, energyPOLARIS, x, y, z, energyL0, Window_ID, TimeDiff, TimeDiffCum, theta1
Coincidence_df = Coincidence_df[['time', 'energyPOLARIS', 'x', 'y', 'z', 'energyL0', 'Window_ID', 'TimeDiff', 'TimeDiffCum', 'theta1']]
Coincidence_df.reset_index(inplace=True, drop=True)

#______________________________________________________

Coincidence_df = Coincidence_df.to_numpy()
print('Coincidence_df : \n', Coincidence_df)

#----
# ********************* TAKEN FROM PROF. PETERSON polarisPGI_processing.py ******************************
#----
'''
- APPLY_COMPTON_LINE_FILTERING           = apply Compton line filtering (checks if E1 and theta1 are consistent with Compton formula, default is False)
  applying Compton line filtering (checks if E1 and theta1 are consistent with Compton formula)
   - use COMPTON_LINE_RANGE (accepted energy range) & COMPTON_LINE_ENERGIES (expected gamma energies) global variables
   - modified code from Matt Leigh's Filter.py

- REMOVE_UNPHYSICAL_EVENTS               = remove unphysical events, i.e. Compton scatter angle == nan (checks both ordering of two scatters and flips if original ordering in unphysical, default is False)
- COMPTON_LINE_RANGE                     = sets accepted energy range (min / max values (percentage) for Compton line filtering)
- COMPTON_LINE_ENERGIES                  = sets expected gamma energies to use for filtering (can take multiple arguments, see list of options below)
    Possible gamma energies for Compton Line Filtering
      - Oxygen Prompts: 2.742, 5.240, 6.129, 6.916, 7.116 MeV
      - Carbon: 4.444 MeV
      - Nitrogen: 1.635, 2.313, 5.269, 5.298 MeV
      - Boron: 0.718 MeV
      - Cobalt-60: 1.173, 1.332 MeV
      - Cesium-137: 0.6617  MeV
      - Sodium-22 : 0.5110, 1.274 MeV
'''

#----
# ********************* PARAMETERS / SETTINGS ******************************
#----
REMOVE_UNPHYSICAL_EVENTS = True
APPLY_COMPTON_LINE_FILTERING = True
COMPTON_LINE_RANGE = np.array( [0.8, 1.2] )     # min / max values (percentage) for Compton line filtering

# 28-02-2023 SOURCE DATA DICTIONARY ( to map run numbers to Compton line energies)

run_compton_lines = {
    1: [],  # Run 1 - No Compton lines
    2: [661.7],  # Run 2 - Cesium-137
    3: [1173, 1332],  # Run 3 - Cobalt-60
    4: [511, 1274],  # Run 4 - Sodium-22
    5: [1173, 1332, 511, 1274, 661.7],  # Run 5 - Cobalt-60, Sodium-22, Cesium-137
    6: [1173, 1332, 511, 1274, 661.7],  # Run 6 - Cobalt-60, Sodium-22, Cesium-137
    7: [511, 1274],  # Run 7 - Sodium-22
    8: [511, 1274],  # Run 8 - Sodium-22
    9: [511, 1274],  # Run 9 - Sodium-22
    10: [511, 1274],  # Run 10 - Sodium-22
    11: [511, 1274],  # Run 11 - Sodium-22
    12: [661.7, 1173, 1332],  # Run 12 - Cesium-137, Cobalt-60
    13: [511, 1274],  # Run 13 - Sodium-22
    14: [511, 1274]   # Run 14 - Sodium-22
}
COMPTON_LINE_ENERGIES = run_compton_lines.get(run_num, []) # sets expected gamma energies to use for filtering (can take multiple arguments, see list of options below)
Na22_only_runs = [4, 7, 8, 9, 10, 11, 13, 14] # runs with only Na22 source
#---- 
# ********************* COMPTON LINE FILTERING ******************************
#----
### FILTERING DETECTOR DATA ###

print ('\n--- Filtering Detector Data ---')

#  removing unphysical events (Compton scatter angle == nan)
#   - checks both ordering of two scatters and flips if original ordering in unphysical
#   - modified code from Matt Leigh's Filter.py & CPUFunctions.cu
if REMOVE_UNPHYSICAL_EVENTS:
    print ('  Removing unphysical events (double scatters) . . .')

    print ('   - Filtering {} | total events checked: {}'.format('Coincidence_df', len(Coincidence_df)))
    Coincidence_df = utils.filtering_unphysical_double_scatters(Coincidence_df)

    # count number of physical double scatters
    num_ds_pe = len(Coincidence_df)

#  applying Compton line filtering (checks if E1 and theta1 are consistent with Compton formula)
#   - use COMPTON_LINE_RANGE (accepted energy range) & COMPTON_LINE_ENERGIES (expected gamma energies) global variables
#   - modified code from Matt Leigh's Filter.py
if APPLY_COMPTON_LINE_FILTERING:
    print ('\n  Applying Compton line filtering / Energies (MeV): {} / Range of values: {}'.format(COMPTON_LINE_ENERGIES, COMPTON_LINE_RANGE))

    print ('   - Filtering {} | total events checked: {}'.format('Coincidence_df', len(Coincidence_df)))
    BeforeCLF_df = pd.DataFrame(np.zeros((len(Coincidence_df), 3)), columns=['E0_No_CLF', 'E1_No_CLF', 'theta1_No_CLF']) # create empty dataframe to store E0, E1, theta1 before CLF
    BeforeCLF_df = BeforeCLF_df.to_numpy()
    Coincidence_df, BeforeCLF_df = utils.compton_line_filtering(Coincidence_df, COMPTON_LINE_RANGE, COMPTON_LINE_ENERGIES, BeforeCLF_df)
    
    # count number of Compton line filtered double scatters
    num_ds_cl = len(Coincidence_df)
Coincidence_df = pd.DataFrame(Coincidence_df)

print('Coincidence_df : \n', Coincidence_df)
# columns are time, energyPOLARIS, energyL0, x, y, z, Window_ID, TimeDiff, TimeDiffCum, energyPOLARIS, theta1
Coincidence_df.rename(columns={0:'time', 1: 'energyPOLARIS', 2:'x', 3:'y',  4:'z',  5:'energyL0', 6:'Window', 7:'TimeDiff',  8:'TimeDiff_CumSum',  9:'theta1' }, inplace=True)
# starting from the top of the dataframe, group the first row containing the same energyL0 value
#Coincidence_df = Coincidence_df.drop_duplicates(subset=['energyPOLARIS'], keep='first')
Coincidence_df['theta1'] = np.degrees(Coincidence_df['theta1'])
Coincidence_df.reset_index(inplace=True, drop=True)
BeforeCLF_df = pd.DataFrame(BeforeCLF_df)
BeforeCLF_df.rename(columns={0:'E0_No_CLF', 1:'E1_No_CLF', 2:'theta1_No_CLF'}, inplace=True)
BeforeCLF_df['theta1_No_CLF'] = np.degrees(BeforeCLF_df['theta1_No_CLF'])
BeforeCLF_df['E2_No_CLF'] = BeforeCLF_df['E0_No_CLF'] - BeforeCLF_df['E1_No_CLF']
print('BeforeCLF_df: \n', BeforeCLF_df)


#______________________________________________________
#----
# ********************* PLOT DATA ******************************
#----
# BEFORE CLF PLOTS
# ----

plt.figure(figsize = (30, 15))
plt.hist2d(BeforeCLF_df['theta1_No_CLF'], BeforeCLF_df['E1_No_CLF'], bins=150, range=[[0, 180], [0, 1500]], cmap='jet', vmin=0.01, vmax = 1.1)
plt.xlim(0, 180)
plt.xlabel('$\\theta$ (deg)', fontsize=30)
plt.ylabel('Energy (keV)', fontsize=30)
plt.title('POLARIS Energy vs $\\theta$ before CLF', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=30)
cbar = plt.colorbar()
plt.savefig(save_dir + 'E1_vs_Theta_beforeCLF_hist2D' + '.png')
plt.show()

counts = BeforeCLF_df.groupby(['theta1_No_CLF', 'E1_No_CLF']).size().reset_index(name = 'counts')
plt.figure(figsize = (30, 15))
plt.scatter(counts['theta1_No_CLF'], counts['E1_No_CLF'] , c = counts['counts'] , cmap = 'jet',vmin=0.01,vmax = 1.1, s=0.5)
plt.xlim(0, 180)
plt.xlabel('$\\theta$ (deg)', fontsize=30)
plt.ylabel('Energy (keV)', fontsize=30)
plt.title('POLARIS Energy vs $\\theta$ before CLF', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=30)
cbar = plt.colorbar()
plt.savefig(save_dir + 'E1_vs_Theta_beforeCLF' + '.png')
plt.show()

# plot the above figure but for Coincidence_df
counts = Coincidence_df.groupby(['theta1', 'energyPOLARIS']).size().reset_index(name = 'counts')
plt.figure(figsize = (30, 15))
plt.scatter(counts['theta1'], counts['energyPOLARIS'] , c = counts['counts'] , cmap = 'jet',vmin=0.01,vmax = 2, s=0.5)
plt.xlim(0, 180)
plt.xlabel('$\\theta$ (deg)', fontsize=30)
plt.ylabel('Energy (keV)', fontsize=30)
plt.title('POLARIS Energy vs $\\theta$ after CLF', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=30)
cbar = plt.colorbar()
plt.savefig(save_dir + 'E1_vs_Theta_afterCLF' + '.png')
plt.show()


# plot a scatter of ploaris energy vs labr energy
counts = BeforeCLF_df.groupby(['E1_No_CLF', 'E2_No_CLF']).size().reset_index(name = 'counts')
plt.figure(figsize = (30, 15))
plt.scatter(counts['E1_No_CLF'], counts['E2_No_CLF'] , c = counts['counts'] , cmap = 'jet',vmin=0.01,vmax=1.1, s=0.5)
plt.ylim(0, 1500)
plt.ylabel('LaBr3 Energy (keV)', fontsize=30)
plt.xlabel('POLARIS Energy (keV)', fontsize=30)
plt.title('POLARIS Energy vs LaBr3 Energy', fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=30)
cbar = plt.colorbar()
plt.savefig(save_dir + 'E1_E2_coincidence_countsscatter_afterCLF' + '.png')
plt.show()


# ______________________________________________________________
#----
# AFTER CLF PLOTS
#----


# plot a scatter of ploaris energy vs labr energy
counts = Coincidence_df.groupby(['energyPOLARIS', 'energyL0']).size().reset_index(name = 'counts')
plt.figure(figsize = (8, 6))
plt.hist2d(Coincidence_df['energyPOLARIS'], Coincidence_df['energyL0'], bins=150, range=[[0, 1500], [0, 1500]], cmap='jet', vmin=0.01)
plt.xlim(0, 1500)
plt.ylim(0, 1500)
plt.xlabel('Scatterer Energy (keV)', fontsize=20)
plt.ylabel('AbsorberBr3 Energy (keV)', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(save_dir + 'E1_E2_coincidence_hist2D_afterCLF' + '.png')
plt.show()



# plot a scatter of ploaris energy vs labr energy
counts = Coincidence_df.groupby(['energyPOLARIS', 'energyL0']).size().reset_index(name = 'counts')
plt.figure(figsize = (8, 6))
plt.hist2d(Coincidence_df['theta1'], Coincidence_df['energyPOLARIS'], bins=150, range=[[0, 180], [0, 1500]], cmap='jet', vmin=0.01)
plt.xlim(0, 180)
plt.ylim(0, 1500)
plt.xlabel('$\\theta$ (deg)', fontsize=20)
plt.ylabel('Scatterer Energy (keV)', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.savefig(save_dir + 'E1_E2_coincidence_hist2D' + '.png')
plt.show()

#######
# fig, ax = plt.subplots(1, 1, figsize = (20,10))
# ax.scatter( Coincidence_df['theta1'], Coincidence_df['energyPOLARIS'], c=Coincidence_df['energyPOLARIS'], cmap='jet', vmin=0.1, vmax = 3, s=0.5)
# ax.set_ylabel('E1 (keV)')
# ax=plt.gca()
# #PCM=ax.get_children()[2]
# #plt.colorbar(PCM, ax=ax)
# ax.set_xlabel('$\\theta$ (rad)')
# ax.set_title('E1 vs $\\theta$')
# ax.grid(axis='both', linestyle='--', linewidth=0.5)
# # plt.show()
# plt.savefig(save_dir + 'E1_vs_theta' + '.png')


#######
# create a df with a column of the sum of Coincidence_df['energyPOLARIS'] and Coincidence_df['energyL0']. This is called E0
# E0 = Coincidence_df['energyPOLARIS'] + Coincidence_df['energyL0']
# E0 = pd.DataFrame(E0)
# plt.figure(figsize = (30,15))
# plt.hist(E0[0], bins=1500, range=(0, 1500), label='E0', color='royalblue', histtype='step', linewidth=1.5)
# plt.xlabel('Energy (keV)', fontsize=30)
# plt.ylabel('Counts', fontsize=30)
# plt.title('E0 after CLF', fontsize=30)
# plt.grid(axis='both', linestyle='--', linewidth=0.5)
# plt.legend(loc = 'upper right', fontsize=30)
# plt.tick_params(axis='both', which='major', labelsize=30)
# plt.savefig(save_dir + 'E0_after_CLF' + '.png')
# # plt.show()
# print(' PLOT SAVED: ' + save_dir + 'E0_after_CLF' + '.png')

#######
# Coincidence_df.to_csv(save_dir + 'Coincidence_df_CLF' + '.txt', index = False, sep = '\t')
# BeforeCLF_df.to_csv(save_dir + 'BeforeCLF_df' + '.txt', index = False, sep = '\t')

# Calculate the time in minutes of the script execution
end_time = time.time()
time_taken = end_time - start_time
print('Time taken to execute the script: ', time_taken/60, ' minutes')


