'''
Script para calcular el auto power spectrum de la DM en caso de no disponer de density field
'''

import numpy as np
import pandas as pd
from nbodykit.lab import ArrayCatalog, FFTPower
import matplotlib
matplotlib.use('Agg')

# Parámetros
Lbox = 3000.
Ngrid = 512
kf = 2 * np.pi / Lbox

# Ruta al fichero de partículas
ruta = "/home/adrian/MN5/UNIT3_4096_fnlm20/DM_PARTICLES/dm_particles_0.5_103"

# Línea de corte en el fichero. Dicha linea contiene un error y hay que excluirla
linea_corte = 77214769

# Procesamos el archivo por trozos
chunksize = 1000000  # Ajustable según memoria disponible


# Comprobación de errores en el archivo

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

print(f"Primer DataFrame: {len(df_sec1)} filas (de 1 a {linea_corte})")
print(f"Segundo DataFrame: {len(df_sec2)} filas (desde {linea_corte+1} en adelante)")


df_all = pd.concat([df_sec1, df_sec2], ignore_index=True)
df_all = df_all.dropna(subset=['x', 'y', 'z'])
positions = df_all[['x', 'y', 'z']].to_numpy()

n_valid = len(df_all)

assert positions.shape == (n_valid, 3), (
    f"Error de dimensión: esperaba {(n_valid, 3)}, "
    f"pero positions tiene {positions.shape}"
)

nan_mask = np.isnan(positions)
if np.any(nan_mask):
    nan_counts = np.sum(nan_mask, axis=0)
    print(f"Valores NaN encontrados: x={nan_counts[0]}, "
          f"y={nan_counts[1]}, z={nan_counts[2]}")
else:
    print("Ningún valor NaN en posiciones.")


inf_mask = np.isinf(positions)
if np.any(inf_mask):
    inf_counts = np.sum(inf_mask, axis=0)
    print(f"Valores infinitos encontrados: x={inf_counts[0]}, "
          f"y={inf_counts[1]}, z={inf_counts[2]}")
else:
    print("Ningún valor infinito en posiciones.")


# Cálculo del power spectrum para DM

catalog = ArrayCatalog({'Position': positions})
mesh    = catalog.to_mesh(resampler='cic', compensated=True,
                          Nmesh=Ngrid, BoxSize=Lbox)


results = FFTPower(mesh, mode='1d', dk=kf, kmin=kf)

Pk      = results.power
Pk_corr = Pk['power'].real - Pk.attrs['shotnoise']



np.savetxt(
    'fichero-de-salida',
    np.column_stack((Pk['k'], Pk_corr)),
    header="k    Pk"
)
