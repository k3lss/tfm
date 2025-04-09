import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path


LBox = 1000
V=LBox**3
dk = 2*np.pi/LBox

#cargamos los valores de N para la densidad
N = np.loadtxt("/home/anguren/celia/full_simulation/Nvalue_R20.txt")
n=N/V

# cargamos los límites de los bines
ruta_bins = "/home/anguren/celia/full_simulation/bins_lims_R20.txt"
sobredens_bins = np.genfromtxt(ruta_bins, delimiter=" ", skip_header=0, names=True)
#checkeamos que los límites estén bien, por si acaso
print(sobredens_bins)


archivo_salida = "/home/anguren/celia/full_simulation/bias_R20/bias_df.txt"
#las columnas van a ser: sobredens_low | sobredens_high | bias_auto_df | error_auto_df | bias_cross_df | error_cross_df
bias_salida = np.zeros((len(N),6))

bias_salida[:,0] = sobredens_bins['limite_inferior'] # type: ignore
bias_salida[:,1] = sobredens_bins['limite_superior'] # type: ignore


b_low_auto= -0  # el bias auto no puede ser negativo
b_low_cross = -15
b_high = 100 # valor maximo para el calculo del bias

# definimos las funciones que vamos a utilizar
def chi2(P, param, Pm, s, value):
    if value == 1:
        # Caso auto: b²
        theta = param
        return np.sum(((P - theta * Pm) ** 2) / s ** 2)
    elif value == 0:
        # Caso cross: b
        b = param
        return np.sum(((P - b * Pm) ** 2) / s ** 2)
    else:
        raise ValueError("value debe ser 0 (cross) o 1 (auto)")

def resultbias(P, Pm, s, value, b_low, b_high):
    if value not in [0, 1]:
        raise ValueError("value debe ser 0 (cross) o 1 (auto)")

    # Caso cross
    if value == 0:
        # Minimizar Chi² en b
        resultado = minimize_scalar(
            lambda b: chi2(P, b, Pm, s, 0),
            bounds=(b_low, b_high),
            method='bounded'
        )
        b_min = resultado.x # type: ignore
        chi2_min = resultado.fun # type: ignore

        # Generar puntos alrededor del mínimo
        b_vals = np.linspace(b_min - 0.5, b_min + 0.5, 200)
        chi2_vals = [chi2(P, b, Pm, s, 0) for b in b_vals]

        # Encontrar cruces con Chi²_min + 1
        b_low_interp = np.interp(chi2_min + 1, chi2_vals[::-1], b_vals[::-1])
        b_up_interp = np.interp(chi2_min + 1, chi2_vals, b_vals)
        inc = abs(b_up_interp - b_low_interp) / 2

    # Caso auto (modelo cuadrático en b)
    else:
        # Reparametrizar a theta = b²
        theta_low = max(0.0, b_low**2)  # Theta no puede ser negativo
        theta_high = b_high**2

        # Minimizar Chi² en theta
        resultado_theta = minimize_scalar(
            lambda theta: chi2(P, theta, Pm, s, 1),
            bounds=(theta_low, theta_high),
            method='bounded'
        )
        theta_min = resultado_theta.x # type: ignore # type: ignore
        chi2_min = resultado_theta.fun # type: ignore # type: ignore
        b_min = np.sqrt(theta_min)

        # Generar puntos alrededor del mínimo en theta
        theta_vals = np.linspace(theta_min - 0.5, theta_min + 0.5, 200)
        chi2_theta_vals = [chi2(P, t, Pm, s, 1) for t in theta_vals]

        # Calcular incertidumbre en theta
        theta_low_interp = np.interp(chi2_min + 1, chi2_theta_vals[::-1], theta_vals[::-1])
        theta_up_interp = np.interp(chi2_min + 1, chi2_theta_vals, theta_vals)
        delta_theta = (theta_up_interp - theta_low_interp) / 2

        # Propagación de errores: theta = b² -> delta_b = delta_theta / (2*b_min)
        inc = abs(delta_theta) / (2 * b_min) if b_min != 0 else np.inf

    return b_min, inc

##############
#CÁLCULO BIAS#
##############

k_auto_df = np.loadtxt("/home/anguren/celia/power_df/Pk_df.txt")[:,0]
P_auto_df = np.loadtxt("/home/anguren/celia/power_df/Pk_df.txt")[:,1]

for i in range(len(N)):
    ruta_halo = f"/home/anguren/celia/full_simulation/power_bin_R20/Pk_selec_{i}.txt"
    ruta_cross = f"/home/anguren/celia/full_simulation/power_cross_R20/Pk_cross_df_{i}.txt"
    datos = np.genfromtxt(ruta_halo, delimiter=" ", names=True)
    datos_cross = np.genfromtxt(ruta_cross, delimiter=" ", names=True)
    k_hal, P_hal = datos['k'], datos['P'] # type: ignore
    k_cross_df, P_cross_df = datos_cross['k'], datos_cross['P'] # type: ignore
    
    selection = np.where((k_hal < 0.1) & (k_cross_df < 0.1))[0]
    
    k_hal = k_hal[selection]
    P_hal = P_hal[selection]

    k_auto_df = k_auto_df[selection]
    P_auto_df = P_auto_df[selection]

    k_cross_df = k_cross_df[selection]
    P_cross_df = P_cross_df[selection]


    error = np.sqrt( (P_hal+1/n[i])**2 *2*(2*np.pi)**3/(4*np.pi*k_hal**2*dk*V) )
    error_cross = np.sqrt( ((P_hal+1/n[i])*P_auto_df + (P_cross_df+1/n[i])**2) *(2*np.pi)**3/(4*np.pi*k_cross_df**2*dk*V) )

    bias_salida[i][2], bias_salida[i][3] = resultbias(P_hal, P_auto_df, error, 1, b_low_auto, b_high)
    bias_salida[i][4], bias_salida[i][5] = resultbias(P_cross_df, P_auto_df, error_cross, 0, b_low_cross, b_high)
     

    #carpeta para guardar los plots
    carpeta = Path(f"/home/anguren/celia/full_simulation/bias_R20/plots_powers/bin_{i}")
    carpeta.mkdir(parents=True, exist_ok=True)


    # plot auto

    plt.figure(figsize=(8,6))
    plt.loglog(k_hal, P_hal, 'ro', label='halo')
    plt.loglog(k_auto_df, bias_salida[i][2]**2 * P_auto_df, 'bo', label='df_auto * bias^2')

    # Asegurar que los errores no sean negativos
    lower_auto = np.maximum(P_hal - error, 1e-10)  # Evitar valores <=0
    upper_auto = P_hal + error
    plt.fill_between(k_hal, lower_auto, upper_auto, facecolor='red', alpha=0.4)

    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]", fontsize=14)
    plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]", fontsize=14)
    plt.legend(loc='upper right')
    plt.title(f'Bin {i} auto, b = {bias_salida[i][2]:.3f}', fontsize=16) 
    plt.savefig(carpeta / f"auto_bin_{i}.png", dpi=300)
    plt.close()


    # plot cross
    # Usar escala lineal si hay valores negativos
    use_log = np.all(P_cross_df > 0) and np.all(bias_salida[i][4] * P_auto_df > 0)

    if use_log:
        plt.loglog(k_cross_df, P_cross_df, 'ro', label='halo')
        plt.loglog(k_auto_df, bias_salida[i][4] * P_auto_df, 'bo', label='df_cross * bias')
    else:
        plt.plot(k_cross_df, P_cross_df, 'ro', label='halo')
        plt.plot(k_auto_df, bias_salida[i][4] * P_auto_df, 'bo', label='df_cross * bias')
        plt.yscale('symlog' if np.any(P_cross_df <= 0) else 'linear')  # Escala symlog si hay negativos

    plt.fill_between(k_cross_df, P_cross_df - error_cross, P_cross_df + error_cross, facecolor='red', alpha=0.4)

    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]", fontsize=14)
    plt.ylabel(r"$P(k)$ [$h^{-3}\mathrm{Mpc}^3$]", fontsize=14)
    plt.legend(loc='upper right')
    plt.title(f'Bin {i} cross, b = {bias_salida[i][4]:.3f}', fontsize=16) 
    plt.savefig(carpeta / f"cross_bin_{i}.png", dpi=300)
    plt.close()


np.savetxt(archivo_salida, bias_salida, fmt=['%.5f', '%.5f' , '%.5f', '%.5f' , '%.5f' , '%.5f'], header='sobredens_low sobredens_high bias_auto_df error_auto_df bias_cross_df error_cross_df' )
