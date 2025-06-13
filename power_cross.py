'''
Script para calcular el cross power spectrum. Utilizamos para la DM un density field. 
Se puede hacer también sin este fichero (hay un ejemplo en otro script)
'''

import numpy as np 
from nbodykit.lab import ArrayCatalog, FFTPower
from nbodykit.lab import BigFileMesh


# Parámetros de la simulación
BoxSize = 1000
Ngrid = 512
kf = 2*np.pi/BoxSize

Verbose=False

# Fichero de entrada de las partículas de DM
f_particles="/home/adrian/Notebooks/PNG_UNITsim/data/unitsim_fnl0_1_032_density"

# Número de halos por bin
N = np.loadtxt("ruta-al-fichero-de-N")

# Archivo bin sobredensidad
bin_sobdens = "ruta-al-bin-de-sobredensidad"

# Archivo para guardar la salida
salida = "fichero-de-salida"


# Cargamos el fichero de las partículas como un mesh utilizando BigFileMesh
particles = BigFileMesh(f_particles,dataset='Field')


# Para cada bin, cargamos los halos guardados y calculamos el cross power spectrum
for i in range(len(N)):
    bin_sob = np.genfromtxt(bin_sobdens, delimiter=" ", names=True)
    x = bin_sob['x']
    y = bin_sob['y']
    z = bin_sob['z']
    posiciones = np.array([x,y,z]).T
    catalog = ArrayCatalog({'Position': posiciones})
    mesh = catalog.to_mesh(resampler='cic', compensated=True, Nmesh=Ngrid, BoxSize=BoxSize)
    r_cross = FFTPower(mesh, second=particles, mode='1d', dk=kf, kmin=kf)
    Pk_cross = r_cross.power
    Pk_cross_corr = Pk_cross['power'].real - Pk_cross.attrs['shotnoise']

    # Guardamos los datos en ficheros para utilizarlos más adelante
    np.savetxt(salida, np.column_stack((Pk_cross['k'], Pk_cross_corr)), header='k P')