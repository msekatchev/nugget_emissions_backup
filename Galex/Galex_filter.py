import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from scipy import integrate

filter_min = 1300
filter_max = 1700

with open("Results2.csv") as file:
    reader = csv.DictReader(file)
    x = []
    y = []
    for row in reader:
        x.append(float(row['X']))
        y.append(float(row['Y']))
        #print(row['X'],row['Y'])

def gaussian(x,a,b,c):
    return a * np.exp(-(x-b)**2/(2*c)**2)


x = np.array(x)
y = np.array(y)
p_initial = (0.79,1513,121)
bandwidth_filter_x = (x>filter_min) & (x<filter_max)
parameters = curve_fit(gaussian, x[bandwidth_filter_x], y[bandwidth_filter_x], p0=p_initial)[0]
print("unnormalized parameters: ", parameters)

bandwidth_integral = integrate.quad(gaussian, filter_min,filter_max,args=(parameters[0],parameters[1],parameters[2]))[0]
print("bandwidth integral is: ", bandwidth_integral)

print("do I divide the amplitude parameter", parameters[0], "by this value to normalize?")



plt.figure(dpi=250, figsize=(6,3))

plt.plot(x,y)
xdata = np.arange(1200,1800,3)
bandwidth_filter_x_data = (xdata>filter_min) & (xdata<filter_max)

plt.plot(xdata[bandwidth_filter_x_data],gaussian(xdata[bandwidth_filter_x_data],*p_initial), "--",color = "blue", label="initial")
plt.plot(xdata[bandwidth_filter_x_data],gaussian(xdata[bandwidth_filter_x_data],*parameters), color="red",label="unnormalized fit")
plt.plot([filter_min,filter_min],[0,1], color="black", label="bandwidth")
plt.plot([filter_max,filter_max],[0,1], color="black")

plt.legend()
plt.ylim(0,1)

plt.savefig("GALEX_filter.png")

plt.show()
