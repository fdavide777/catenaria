import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

dati = np.loadtxt(r'C:/Users/Davide/Desktop/Catenaria.csv',delimiter = ',')

px = dati[:,0]
py = -dati[:,1]
sigma_px = np.full(px.shape, 0.5)
sigma_py = np.full(py.shape, 0.5)


def catanaio(x, a, c, x0):
    return c + a * np.cosh((x - x0) / a)


#plt.show()
#plt.savefig("fit.png", dpi=300)

popt,pcov = curve_fit(catanaio, px, py, p0=(126, 126, 126))
a_hat, c_hat, x0_hat = popt

sigma_a, sigma_c, sigma_x0 = pcov
print(popt,pcov)

r = py - catanaio(px, a_hat, c_hat, x0_hat)
r = r
/sigma_py

xx = np.linspace(0, 1200, 1000)
yy = catanaio(xx, popt[0], popt[1], popt[2])

plt.figure('supercatanazzo')
plt.axes([0.25,0.2,0.7,0.7])
plt.scatter(px, py, marker='o', color ='chartreuse', label='Dati raccolti' )
plt.errorbar(px, py, sigma_py, sigma_px, fmt="|", label='Barre di errore', color = 'green')
plt.plot(xx, yy, '-b', label='Fit')
plt.xlim([0,1500])
plt.ylim([-1200,0])
plt.xlabel(r"x [px]")
plt.ylabel(r"y [py]")
plt.legend()
plt.grid(which='both', ls='dashed', color='gray')

plt.figure('supercatanazzo_residui')
plt.errorbar(px, r, 1, fmt='.', label = 'Scarti ed errore di misura')
plt.axhline(0,color="black")
plt.xlabel(r"x [px]")
plt.ylabel(r"Residui")
plt.legend()
plt.grid(which='both', ls='dashed', color='gray')
plt.show()







