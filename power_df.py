'''
Script para calcular el auto power spectrum de las partículas de DM a traveś de un density
field.
También se puede calcular el power spectrum de la DM sin utilizar partículas, hay un
ejemplo en otro script
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from nbodykit.source.catalog import CSVCatalog
from nbodykit.lab import ArrayCatalog, FFTPower
from nbodykit.lab import BigFileMesh

# Definimos los parámetros de la simulación
Lbox = 1000.
Ngrid=2048
Verbose=False
kf = 2*np.pi/Lbox

# Importamos los density fields pregenerados para la materia oscura y calculamos su Pk
f_particles="/home/adrian/Notebooks/PNG_UNITsim/data/unitsim_fnl0_1_032_density"
particles = BigFileMesh(f_particles,dataset='Field')
results = FFTPower(particles,mode='1d',dk=kf,kmin=kf)

Pk = results.power
Pk_corr = Pk['power'].real - Pk.attrs['shotnoise']

# Exportamos los valores obtenidos
np.savetxt('fichero-de-salida',np.column_stack((Pk['k'], Pk_corr)) )
