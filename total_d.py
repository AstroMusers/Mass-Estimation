import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
from pathlib import Path
from tdpy.verbosity import print
from pairwise_model import adjacent_period_spacings

plt.rcParams.update({'font.size': 8})

filename = Path(__file__).with_name("only_transit_data_28062023.csv")
print(f"Reading from {filename}...")
datafile = pd.read_csv(filename, skiprows=202)

##### total D data #####

pair_positions, d_values = adjacent_period_spacings(datafile)
passed = len(datafile) - 1 - len(pair_positions)
d_data = pd.Series(d_values)
wanted_data = d_data[d_data.between(0, 2, inclusive="neither")]
w = np.log10(wanted_data)

output_filename = Path("logD.csv")
print(f"Writing to {output_filename}...")
w.to_csv(output_filename, index=False, header=True)

print("Mean of logD :", np.mean(w))
print("Standard Deviation of logD :", np.std(w))

phi_D = st.norm.cdf((np.log10(2) - np.mean(w)) / np.std(w))
print("Phi value of D is:", phi_D)

res2 = st.shapiro(w)
print("Normality test 2 result for dataset is", res2.statistic)

##### end #####

##### tess D data #####

_, tess_d_values = adjacent_period_spacings(
    datafile, instrument="TESS CCD Array"
)
d1_data = pd.Series(tess_d_values)

wanted_data1 = d1_data[d1_data > 0]
w1 = np.log10(wanted_data1)

print("Mean of logD for Kepler-Tess :", np.mean(w1))
print("Standard Deviation of logD for Kepler-Tess :", np.std(w1))

phi_D1 = st.norm.cdf((np.log10(2) - np.mean(w1)) / np.std(w1))
print("Phi value of D for Kepler-Tess is:", phi_D1)

res2 = st.shapiro(w1)
print("Normality test 2 result for Kepler-Tess is", res2.statistic)

##### end #####

##### kepler D data #####

_, kepler_d_values = adjacent_period_spacings(
    datafile, instrument="Kepler CCD Array"
)
d2_data = pd.Series(kepler_d_values)

wanted_data2 = d2_data[d2_data > 0]
w2 = np.log10(wanted_data2)

print("Mean of logD for Kepler :", np.mean(w2))
print("Standard Deviation of logD for Kepler :", np.std(w2))

phi_D2 = st.norm.cdf((np.log10(2) - np.mean(w2)) / np.std(w2))
print("Phi value of D for Kepler is:", phi_D2)

res2 = st.shapiro(w2)
print("Normality test 2 result for Kepler-Tess is", res2.statistic)

##### plots #####

plt.figure("logD", figsize=(7.2, 4.5))
plt.subplot(1,3,1)
plt.hist(w, bins=50, histtype="step", color="Red")
plt.xlabel("log$_{10}$D")
plt.ylabel("Number of Exoplanet Pairs")
plt.xlim(left=-1.1)
plt.xticks()
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.subplot(1,3,2)
plt.hist(w1, bins=50, histtype="step", color="Blue")
plt.xlabel("log$_{10}$D")
plt.xticks()
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.subplot(1,3,3)
plt.hist(w2, bins=50, histtype="step", color="Green")
plt.xlabel("log$_{10}$D")
plt.xlim(left=-1.1)
plt.xticks()
plt.tick_params(top=True, bottom=True, left=True, right=True, direction="in", which="minor", length=3)
plt.minorticks_on()
plt.tight_layout()
plt.show()

print(f"{passed} lines in the a_total.csv has been passed")

##### end #####