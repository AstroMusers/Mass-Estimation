import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st

plt.rcParams.update({'font.size': 8})

filename = "only_transit_data_28062023.csv"
datafile = pd.read_csv(filename, skiprows=177)

##### total D data #####

d_list = []
passed = 0
number_of_planets = 0
for i in range(len(datafile)-1):
    number_of_planets += 1
    if datafile["hostname"][i] == datafile["hostname"][i+1]:
        P_i = datafile["pl_orbper"][i+1]/datafile["pl_orbper"][i]
        D = 2*(((P_i**(2/3)) - 1)/((P_i**(2/3))+1))
        d_list.append(D)
    else:
        passed += 1
d_data = pd.Series(d_list)
wanted_data = d_data[d_data.between(0, 2, inclusive="neither")]
w = np.log10(wanted_data)

w.to_csv("logD.csv", index=False, header=True)

print("Mean of logD :", np.mean(w))
print("Standard Deviation of logD :", np.std(w))

phi_D = st.norm.cdf((np.log10(2) - np.mean(w)) / np.std(w))
print("Phi value of D is:", phi_D)

res2 = st.shapiro(w)
print("Normality test 2 result for dataset is", res2.statistic)

##### end #####

##### tess D data #####

d1_list = []
for i in range(0,len(datafile)-1):
    if datafile["disc_instrument"][i] == "TESS CCD Array":
        if datafile["hostname"][i] == datafile["hostname"][i+1]:
            P_i = datafile["pl_orbper"][i+1]/datafile["pl_orbper"][i]
            D = 2*((P_i**(2/3) - 1)/(P_i**(2/3)+1))
            d1_list.append(D)
d1_data = pd.Series(d1_list)

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

d2_list = []
for i in range(0,len(datafile)-1):
    if datafile["disc_instrument"][i] == "Kepler CCD Array":
        if datafile["hostname"][i] == datafile["hostname"][i+1]:
            P_i = datafile["pl_orbper"][i+1]/datafile["pl_orbper"][i]
            D = 2*((P_i**(2/3) - 1)/(P_i**(2/3)+1))
            d2_list.append(D)
d2_data = pd.Series(d2_list)

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