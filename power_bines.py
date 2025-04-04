import matplotlib.pyplot as plt 
import numpy as np 
from nbodykit.lab import ArrayCatalog, FFTPower
import pandas as pd


# Parámetros de la simulación
BoxSize = 1000
Ngrid = 512
kf = 2*np.pi/BoxSize

Verbose=False

# Fichero de entrada
datos = np.genfromtxt("/home/anguren/celia/full_simulation/sobredensidad_halos_corrected.csv", delimiter=" ", skip_header=0, names=True) # Sobredensidad

# Fichero para guardar los límites de los bines
ruta_bins = "/home/anguren/celia/full_simulation/bins_lims.txt"

# Ficheros de salida
salida = f"/home/anguren/celia/full_simulation/bines/bin_sobdens_{i}.txt" # Archivo para guardar los bines en sobredensidad
salida_N = "/home/anguren/celia/full_simulation/Nvalue.txt" # Archivo para guardar el número de halos por bin
ruta_directorio = "/home/anguren/celia/full_simulation/power_bin" # Carpeta en la que se guardan los valores del power spectrum



# Cargamos los datos obtenidos para la sobredensidad
x, y, z, sobredensidad = datos['x'], datos['y'], datos['z'], datos['sobredensidad']

# Definimos unos diccionarios para el power spectrum y las k de los halos, nos servirán para guardar los datos de cada bin de forma cómoda
P_k = {}
k_halos = {}

# Definimos los bines para que todos tengan el mismo número de halos
df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'sobredensidad': sobredensidad})
df['bin'], bin_edges = pd.qcut(df['sobredensidad'], q=50, labels=False, retbins=True)

# Guardamos los límites de los bines en un archivo, para así tenerlos a mano para otros códigos
bin_limits = np.column_stack((bin_edges[:-1], bin_edges[1:]))  
np.savetxt(ruta_bins, bin_limits, delimiter=" ", header="limite_inferior limite_superior")
print(bin_limits) # Comprobamos que los límites guardados tengan sentido

# Creamos la variable N, que luego nos servirá para guardar el número de halos por bin
N = np.zeros(len(bin_limits[:,0]))

# Creamos un diccionario para guardar los bines de halos. Estos nos serán útiles en otros códigos
selection = {}

for i in range(50):
    selection[f"selec_{i}"] = df.index[df['bin'] == i].to_numpy() # Definimos el bin i como los halos que entran dentro del bin i definido con pandas
    N[i] = len(selection[f"selec_{i}"]) # Definimos el valor de N para ese bin

    bin_sobredens = df.iloc[selection[f"selec_{i}"]][['x', 'y', 'z', 'sobredensidad']].to_numpy()
    np.savetxt(salida, bin_sobredens, header="x y z sobredensidad") # Guardamos el bin de sobredensidad para futuro uso

np.savetxt(salida_N, N) # Guardamos N para luego calcular el shotnoise


# Definimos una función para calcular el power spectrum de los halos mediante FFTPower
def calcular_espectro(indices, x, y, z, Ngrid, BoxSize):
    posiciones_bin = np.array([x[indices], y[indices], z[indices]]).T

    catalog = ArrayCatalog({'Position': posiciones_bin})
    mesh = catalog.to_mesh(resampler='cic', compensated=True, Nmesh=Ngrid, BoxSize=BoxSize)

    r = FFTPower(mesh, mode='1d', dk=kf, kmin=kf)
    Pk = r.power
    Pk_corr = Pk['power'].real - Pk.attrs['shotnoise']

    return Pk['k'], Pk_corr


# Aplicamos la función a los bines anteriormente definidos
for key, indices in selection.items():
    k_halos[key], P_k[key] = calcular_espectro(indices, x, y, z, Ngrid, BoxSize)
    plt.loglog(k_halos[key], P_k[key], 'o')
    
    # Guardamos los datos de k y Pk
    nombre_archivo_Pk = f"Pk_{key}.txt"
    ruta_archivo_Pk = f"{ruta_directorio}/{nombre_archivo_Pk}"
    np.savetxt(ruta_archivo_Pk, np.column_stack((k_halos[key], P_k[key])), delimiter=" ", header="k P")