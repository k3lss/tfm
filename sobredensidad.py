import numpy as np
from scipy.spatial import cKDTree
import pandas as pd

# Definimos los parámetros a utilizar para calcular la sobredensidad. Rc puede cambiarse.
Rc = 8.0      # Radio de corte para vecinos cercanos. Es el radio de la esfera dentro de la cual consideramos los vecinos cercanos a un halo.
L = 1000.0    # Ramaño de la simulación (LBox)

# Definimos el fichero de entrada
data = pd.read_hdf('/home/adrian/Notebooks/PNG_UNITsim/data/PNG_UNITsim_N4096_fnl100_z1_032.h5', key='rockstar_catalog')

# Definimos el fichero de salida
output = "/home/anguren/celia/full_sim_fnl100/R8_fnl100_sobredensidad.csv"

# Cargamos los datos de la simulación
main_halos=True
x, y, z, main_halos, masas = data['X'].to_numpy(), data['Y'].to_numpy(), data['Z'].to_numpy(), data['PID'].to_numpy(), data['M200c'].to_numpy()

# Hacemos un corte en masas. En nuestro caso, tomamos los main halos que tengan masa >= 2e10
seleccion_masas = np.where((masas >= 2e10) & (main_halos==-1))[0]
x = x[seleccion_masas]
y = y[seleccion_masas]
z = z[seleccion_masas]

# Print de las coordenadas minima y máxima para x, y, z; queremos comprobar que no se salgan de la caja
print("x min:", np.min(x), "x max:", np.max(x))
print("y min:", np.min(y), "y max:", np.max(y))
print("z min:", np.min(z), "z max:", np.max(z))

# Imponemos una corrección a las coordenadas en caso de que estas esten fuera de la caja
x = np.mod(x, L)
y = np.mod(y, L)
z = np.mod(z, L)
# Comprobamos la corrección
print("coordenadas min tras correccion:", "x:", np.min(x), "y:", np.min(y), "z:", np.min(z))
print("coordenadas max tras correccion:", "x:", np.max(x), "y:", np.max(y), "z:", np.max(z))


# Construimos un tree con las coordenadas. Este ya incluye condiciones de contorno periódicas.
tree = cKDTree(np.column_stack((x, y, z)), boxsize=L)


# Definimos nuestra función para encontrar vecinos cercanos mediante el KDTree. Lo hacemos en tandas, ya que si no ocuparía demasiada memoria
def compute_neighbors_batch(start, end):
    indices_list = tree.query_ball_point(np.column_stack((x[start:end], y[start:end], z[start:end])), Rc, p=2)  # Para los halos dentro del batch, calculamos el número de halos que se encuentran en la esfera centrada en el halo y de radio Rc
    return [len(indices) - 1 for indices in indices_list]  # No contamos el propio halo

# Establecemos un tamaño de batch, para no llenar la memoria
batch_size = 50000
n_halos = np.zeros(len(x), dtype=int)

# Aplicamos la función a cada barch
for i in range(0, len(x), batch_size):
    print(f"Processing batch {i} to {min(i + batch_size, len(x))}...")
    n_halos[i:i + batch_size] = compute_neighbors_batch(i, min(i + batch_size, len(x)))

# Calculamos la media de vecinos cercanos y con ello la sobredensidad
N_mean = np.mean(n_halos)
sobredensidad = (n_halos - N_mean) / N_mean 

# Visualizamos algunos datos de sobredensidad y la forma de la matriz de datos guardados para comprobar que todo va bien
print("Valores de sobredensidad:", sobredensidad[:10])
print("Forma de la matriz guardada:", np.column_stack((x, y, z, sobredensidad)).shape)

# Guardamos los datos en un fichero
np.savetxt(output, np.column_stack((x, y, z, sobredensidad)), delimiter=" ", header="x y z sobredensidad")