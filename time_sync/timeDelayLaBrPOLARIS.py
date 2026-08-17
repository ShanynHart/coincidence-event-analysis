import pandas as pd
import sys 
import pyarrow.parquet as pq


data_dir = sys.argv[1]
data_dir = data_dir + '/'
df = pd.read_parquet(data_dir+'timeSyncedPOLARISAllLaBr27.parquet')

# Filter the DataFrame to include only POLARIS and LaBr 2 detectors
df = df[df['Detector'].isin(['POLARIS', 'LaBr 2'])]

# Filter the DataFrame to include only events within 1% of 511 keV
df = df[(df['Energy'] >= 511 * 0.99) & (df['Energy'] <= 511 * 1.01)]

# Group the DataFrame by Detector and sort by Time
grouped = df.groupby('Detector').apply(lambda x: x.sort_values('Time'))

# Iterate through the grouped DataFrame to find the delay in the electronics of POLARIS
prev_event = None
for index, row in grouped.iterrows():
    if row['Detector'] == 'LaBr 2':
        # Found a 511 keV event from LaBr 2, look for the next one from POLARIS
        for _, next_row in grouped.loc[index+1:].iterrows(): #
            if next_row['Detector'] == 'POLARIS' and \
               next_row['Energy'] >= 511 * 0.99 and \
               next_row['Energy'] <= 511 * 1.01:
                # Found a 511 keV event from POLARIS, calculate the delay in electronics
                delay = next_row['Time'] - row['Time']
                if prev_event is not None:
                    # Calculate the time difference between this and the previous back-to-back event
                    time_diff = row['Time'] - prev_event['Time']
                    print(f"Delay in POLARIS electronics: {delay}, time difference: {time_diff}")
                prev_event = row
                break
