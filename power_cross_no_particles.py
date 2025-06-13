'''
Script para calcular el cross power spectrum en caso de no tener un density field 
para la DM
'''

import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib
matplotlib.use('Agg')
from nbodykit.source.catalog import CSVCatalog
from nbodykit.lab import ArrayCatalog, FFTPower
from nbodykit.lab import BigFileMesh
import pandas as pd

# Parámetros de la simulación
BoxSize = 3000
Ngrid = 512
kf = 2*np.pi/BoxSize

Verbose=False

# Ruta al fichero de las partículas de DM
ruta = "/home/adrian/MN5/UNIT3_4096_fnlm20/DM_PARTICLES/dm_particles_0.5_081"

# Línea de corte. En el fichero utilizado dicha linea contiene un error, y hay que excluirla
linea_corte = 77214769

# Tamaño de chunk, para procesar el archivo por trozos
chunksize = 1000000  # Ajustable según memoria disponible
iterador = pd.read_csv(
    ruta,
    delim_whitespace=True,
    header=None,
    usecols=[0, 1, 2],
    names=['x', 'y', 'z'],
    chunksize=chunksize
)

lista1 = []
lista2 = []
contador = 0

# Comprobación por trozos de que no haya ninguna linea en el archivo que cree conflictos
for trozo in iterador:
    filas_trozo = len(trozo)
    if contador + filas_trozo < linea_corte:
        # Todo el trozo cabe antes de la línea de corte
        lista1.append(trozo)
    else:
        # Aquí es donde ocurre la división dentro de este trozo
        indice_split = linea_corte - contador
        if indice_split > 0:
            lista1.append(trozo.iloc[:indice_split])
        lista2.append(trozo.iloc[indice_split:])
        # El resto de trozos (si quedan) van directo a lista2
        break
    contador += filas_trozo

# Añadimos los trozos restantes (si los hay) a lista2
for trozo in iterador:
    lista2.append(trozo)

# Concatenamos
df_sec1 = pd.concat(lista1, ignore_index=True)
df_sec2 = pd.concat(lista2, ignore_index=True)

df_all = pd.concat([df_sec1, df_sec2], ignore_index=True)
df_all = df_all.dropna(subset=['x', 'y', 'z'])
positions_dm = df_all[['x', 'y', 'z']].to_numpy()

'''
ruta = "/home/adrian/MN5/UNIT3_4096_fnlm20/DM_PARTICLES/dm_particles_0.5_103"
df = pd.read_csv(ruta, delim_whitespace=True, header=None, usecols=[0, 1, 2], names=['x', 'y', 'z'])
positions_dm = df[['x', 'y', 'z']].to_numpy()
assert np.isfinite(positions_dm).all(), "¡Cuidado! 'posiciones' contiene NaNs o Inf"
'''

N = np.loadtxt("ruta-al-fichero-de-N")

for i in range(len(N)):

    bin_sob = np.genfromtxt("ruta-al-bin", delimiter=" ", names=True)
    x = bin_sob['x']
    y = bin_sob['y']
    z = bin_sob['z']
    posiciones = np.array([x,y,z]).T

    assert np.isfinite(posiciones).all(), "¡Cuidado! 'positions_dm' contiene NaNs o Inf"
     
    catalog = ArrayCatalog({'Position': posiciones})
    catalog_dm = ArrayCatalog({'Position': positions_dm})
    mesh = catalog.to_mesh(resampler='cic', compensated=True, Nmesh=Ngrid, BoxSize=BoxSize)
    mesh_dm = catalog_dm.to_mesh(resampler='cic', compensated=True, Nmesh=Ngrid, BoxSize=BoxSize)
    r_cross = FFTPower(mesh, second=mesh_dm, mode='1d', dk=kf, kmin=kf)
    Pk_cross = r_cross.power
    Pk_cross_corr = Pk_cross['power'].real #- Pk_cross.attrs['shotnoise'] # type: ignore
    

    np.savetxt("fichero-de-salida", np.column_stack((Pk_cross['k'], Pk_cross_corr)), header='k P')