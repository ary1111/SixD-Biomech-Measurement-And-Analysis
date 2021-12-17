# Author: Adam Ryason (Graduate Student, RPI)
# Date: 2/1/2020

import pandas as pd
import numpy as np

# Import the CSV
data = pd.read_csv(r'TestKinematicForceTorqueData.csv')

# Create a DF of the voltage columns
df = pd.DataFrame(data, columns = ['V0','V1','V2','V3','V4','V5'])

# Define the ATI Voltage-Force transformation matrix
# Serial="FT29052", BodyStle="Nano25"
V_2_FT = ([0.13458,-0.08264,-0.48198,13.28748,0.53062,-13.38537],
    [0.78230,-15.82432,0.16722,7.69609,-0.42362,7.79557],
    [25.94063,-0.71219,26.23009,-0.71410,26.45688,-0.57775],
    [0.01013,-0.13356,0.26102,0.06023,-0.25933,0.07035],
    [-0.29491,0.00828,0.16093,-0.11569,0.13367,0.11145],
    [0.00703,-0.12520,0.00378,-0.12086,0.00056,-0.12100])

# Create a list of the calculated forces
force = []
for index, row in df.iterrows():
    force.append(np.matmul(V_2_FT,row.to_numpy()))

# Convert the list to a DF for further manipulation and export to CSV
df1 = pd.DataFrame(force)
df1.to_csv("output.csv")