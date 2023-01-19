import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import lambertw
import scipy.constants as const
from pvlib.pvsystem import i_from_v

class cell:
    def __init__(self):
        return

    def load_eqe(self, dir_f):
        f = os.path.join(os.getcwd(),'EQE',dir_f+'.csv')
        df = pd.read_csv(f)
        wavelength = df['Wavelength'].to_numpy()
        eqe = df['EQE'].to_numpy()
        self.eqe = [wavelength, eqe]
        return

    def load_sepectrum(self, dir_f, scaling=1):
        f = os.path.join(os.getcwd(), 'Spectrums', dir_f+'.csv')
        df = pd.read_csv(f)
        wavelength = df['Wavelength'].to_numpy()
        intensity = df['Intensity'].to_numpy()
        self.spectrum = [wavelength, intensity * scaling]
        return

    def load_jv(self, dir_f):
        f = os.path.join(os.getcwd(), 'JV', dir_f+'.csv')
        df = pd.read_csv(f)
        voltage = df['Voltage'].to_numpy()
        current = df['Current'].to_numpy()
        self.jv = [voltage, current]
        return

    def calculate_resistance(self, slope_frac=0.05):
        x = self.jv[0]
        y = -self.jv[1]
        length = len(x)
        values = int(length * slope_frac)
        start = np.arange(0,values)
        end = np.arange(length-values,length)
        self.shunt_resistance = np.average(np.diff(y[start]) / np.diff(x[start]))
        self.serise_resistance = np.average(np.diff(y[end]) / np.diff(x[end]))
        return

    def load_dark_jv(self, dir_f):
        f = os.path.join(os.getcwd(), 'Dark JV', dir_f+'.csv')
        df = pd.read_csv(f)
        voltage = df['Voltage'].to_numpy()
        current = df['Current'].to_numpy()
        self.dark_jv = [voltage, current]
        return

    def calculate_local_ideality(self):
        y = self.dark_jv[1]
        self.dark_satuartion = self.dark_jv[1][0]
        y = np.log(y)
        y = np.diff(y)/np.diff(self.dark_jv[0])
        y = const.e / (y * const.k * 273.15)
        y = np.where(y > 4, np.nan, y)
        y = np.where(y < 0, np.nan, y)
        not_nans = np.argwhere(np.isnan(y) == False).ravel()
        self.local_ideality = [self.dark_jv[0], y]
        self.average_ideality = np.average(y[not_nans])
        return

    def calculate_photogenerated(self):
        eqe_W_min = self.eqe[0][0]
        eqe_W_max = self.eqe[0][-1]

        spectrum_W_min = self.spectrum[0][0]
        spectrum_W_max = self.spectrum[0][-1]

        if eqe_W_min >= spectrum_W_min:
            W_min = eqe_W_min
        else:
            W_min = spectrum_W_min
        if eqe_W_max <= spectrum_W_max:
            W_max = eqe_W_max
        else:
            W_max = spectrum_W_max

        eqe_arg_W_min = np.argwhere(self.eqe[0][:] == W_min)[0][0]
        eqe_arg_W_max = np.argwhere(self.eqe[0][:] == W_max)[0][0]

        eqe = self.eqe[:][eqe_arg_W_min:eqe_arg_W_max]

        spectrum_arg_W_min = np.argwhere(self.spectrum[0][:] == W_min)[0][0]
        spectrum_arg_W_max = np.argwhere(self.spectrum[0][:] == W_max)[0][0]

        spectrum = [self.spectrum[0][spectrum_arg_W_min:spectrum_arg_W_max], self.spectrum[1][spectrum_arg_W_min:spectrum_arg_W_max]]

        if len(eqe[0]) < len(spectrum[0]):
            x = np.linspace(np.min(eqe[0]), np.max(eqe[0]), len(spectrum))
            y = np.interp(x, eqe[0], eqe[1])
            eqe = [x, y]
        elif len(spectrum[0]) < len(eqe[0]):
            x = np.linspace(np.min(spectrum[0]), np.max(spectrum[0]), len(eqe))
            y = np.interp(x, spectrum[0], spectrum[1])
            spectrum = [x, y]

        I = np.zeros(len(eqe[0]))
        for j in range(len(eqe[0])):
            I[j] = eqe[1][j] * spectrum[1][j]
        I = np.cumsum(I)
        I = I * (const.e / (const.h * const.c))
        I = I[-1]/10000
        self.photogenerated = I

        return

    def equivilent_cuircuit_jv(self, temperature = 273.15):
        v = np.linspace(-2,2,1000)
        vt = (const.k * temperature) / const.e
        A = ((self.photogenerated + self.dark_satuartion) - v / self.shunt_resistance) / (1 + self.serise_resistance/self.shunt_resistance)
        B = (self.average_ideality * vt) / self.serise_resistance
        C = (self.dark_satuartion * self.serise_resistance) / (self.average_ideality * vt * (1+ self.serise_resistance/self.shunt_resistance))
        D1 = v/self.average_ideality * vt
        D2 = 1 - (self.serise_resistance/(self.serise_resistance + self.shunt_resistance))
        D3 = ((self.photogenerated + self.dark_satuartion) * self.serise_resistance) / (self.average_ideality * vt * (1 + (self.serise_resistance/self.shunt_resistance)))
        D = np.exp((D1*D2) + D3)
        lamb = lambertw(C * D)
        i = A - (B*lamb)
        print(B)
        return v,i

    def equivilent_cuircuit_jv_2(self, temperature=273.15):
        v = np.linspace(-2, 2, 1000)
        nNsVth = self.average_ideality * 1 * (const.k * temperature/const.e)
        i = i_from_v(self.shunt_resistance, self.serise_resistance, nNsVth, v, self.dark_satuartion, self.photogenerated)
        return v, i



Perc = cell()
Perc.load_eqe('PERC')
Perc.load_jv('PERC')
Perc.calculate_resistance()
Perc.load_dark_jv('PERC')
Perc.calculate_local_ideality()
Perc.load_sepectrum('AM1.5G')
Perc.calculate_photogenerated()
v,i = Perc.equivilent_cuircuit_jv()
plt.plot(v,i)
v,i = Perc.equivilent_cuircuit_jv_2()
plt.plot(v,i)
plt.show()

