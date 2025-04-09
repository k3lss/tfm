import numpy as np
import camb
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# Definimos nuestras funciones

def get_camb_results(
        zout, nonlinear=False, accuracy_boost=1.,kmax=100.,
        path_ini='/home/adrian/utilities/params_Planck15Table4LastColumn.ini'):
    """
    Reads the cosmological parameters from path_ini and uses CAMB for computing
    the matter power spectrum

    Parameters
    ----------
    zout : float
        Redshift at which the matter power spectrum is computed.
    nonlinear : boolean, optional
        Flag which determines if we want the linear power spectrum (False) or 
        the nonlinear one (True). The default is False.
    accuracy_boost : float, optional
        Determines the accuracy which with the power spectrum is computed by 
        CAMB. A larger value results in a more precise Pk, but also it takes 
        more time the computation. The default is 1.0.
    kmax : float, optional
        kmax at which the power spectrum is computed. The default is 100.
    path_ini : string, optional
        Sets the path to the parameter file of CAMB. It contains all the 
        cosmological parameters needed for the computation of the matter power 
        spectrum. The default is 'params_Planck15Table4LastColumn.ini'.

    Returns
    -------
    pk.P : class
        It is an interpolator of the matter power spectrum for the cosmology
        provided in path_ini. PK.P(zout,kh) gives you the power spectrum at 
        z=zout, kh = k/h.

    """
    params = camb.read_ini(path_ini)
    params.set_matter_power(redshifts=[zout],kmax=kmax)
    params.set_accuracy(AccuracyBoost=accuracy_boost)

    results = camb.get_results(params)
    
    return results


zout = 1.032
results = get_camb_results(zout)
transf = results.get_matter_transfer_data()
k_camb = np.array(transf.transfer_data[0,:,0])
tk_camb = transf.transfer_data[1,:,0]

def get_tk(k):
    return sp.interpolate.interp1d(k_camb,tk_camb)(k)/tk_camb[0] #Normalised to T(k->0) = 1

Pk =  results.get_matter_power_interpolator(nonlinear=False)


def growth_factor(z,omega_m0=0.3089):
    """
    Assuming a flat LambdaCDM cosmology, computes the growth factor at a 
    redshift z, given the actual matter density parameter omega_m0, 
    normalised at D(z=0)=1.
    
    Parameters
    ----------
    z : float
        Redshift at which we want the growth factor.
    omega_m0 : float, optional
        Matter density parameter at z=0. The default is 0.3089.

    Returns
    -------
    dz : float
        The growth factor at the redshift given by z.
    """

    def get_delta(z,omega_m0):
        a = 1/(z+1) #Scale factor
        w = -1 #EoS for the Dark Energy. w=-1 for a cosmological constant
        omega_m = (omega_m0*a**(-3))/(omega_m0*a**(-3)+(1-omega_m0))
        
        delta =  a*sp.special.hyp2f1( (w-1)/(2*w) , -1/(3*w) ,
                                1- 5/(6*w),1 - 1/omega_m )  #See 1105.4825

        return delta

    num = get_delta(z,omega_m0=omega_m0)
    den = get_delta(0,omega_m0=omega_m0)
    dz = num/den
    return dz



def m_sdb(k,z=1.032,omega_m=0.3089,norm=1.275):
    """
    Solves the Poisson Equation in Fourier Space, given the k modes, the 
    redshift, the cosmology (in terms of the transfer function given in path),
    and the normalization of the growth factor (we are using the convention 
    that D(z=0)=1, but this expresion requires the D(z)=1 at matter-radiation
    equality.) 
    \delta(k,z) = M(k,z) * \phi(k,z)  
    This M_sdb can is used for computing the scale-dependent bias of the 
    power spectrum.

    Parameters
    ----------
    k : numpy ndarray
        Values at which we want to compute M(k,z).
    z : float, optional
        Redshift at which we want to compute M(k,z). The default is 0.9872.
    omega_m : float, optional
        The cosmological Omega_matter parameter at z=0. The default is 0.3089.
    norm : float, optional
        Normalization of the growth factor. The function I am using for 
        computing the growth factor is normalized in such way that D(z=0)=1.
        However, the convetion of the rest of the parameters assume that the 
        growth factor should be normalized at D(z_eq)=1. Then we have to 
        correct this. The default is 1.275.

    Returns
    -------
    m : numpy ndarray
        Computes M(k,z) at the desired k-values and redshift.

    """
    
    tk = get_tk(k)
    dz = growth_factor(z)
    c_kms = sp.constants.c/1000.
   
    num = 3*omega_m*norm*100**2 #The term 100**2 comes from (H_0/h)**2 
    den = 2*k**2 *tk *dz *c_kms**2
    m = num/den
    return m


def get_scale_dependent_bias(k,b1,bphi,fnl=100,z=1.032,omega_m=0.3089,norm=1.275):
    """
    Computes the scale-dependent bias induced by fNL at first order in 
    the bias expansion.

    Parameters
    ----------
    k : numpy ndarray
        The k values at which we want the power spectrum of galaxies/halos.
    pk : numpy ndarray
        The matter power spectrum from which we want to derive the power 
        spectrum of halos.
    b1 : float
        Linear bias parameter.
    fnl : float
        f_NL parameter.
    z : float, optional
        Redshift at which we want to compute M(k,z). The default is 0.9753.
    omega_m : float, optional
        The cosmological Omega_matter parameter at z=0. The default is 0.3089.
    norm : float, optional
        Normalization of the growth factor. The function I am using for 
        computing the growth factor is normalized in such way that D(z=0)=1.
        However, the convetion of the rest of the parameters assume that the 
        growth factor should be normalized at D(z_eq)=1. Then we have to 
        correct this. The default is 1.275.

    Returns
    -------
    scale_dependent_bias : numpy ndarray
        Array containing the scale-dependent bias computed at the k-values
        given by k.

    """
    #delta_c = 1.686
    #bphi = 2*delta_c*(b1-1)  # <--This is the universality relation. You may want to change to 2*delta_c*(b1-p) with p=1.6 or other values. 
    scale_dependent_bias = b1 + bphi*fnl*m_sdb(k,z=z,omega_m=omega_m,norm=norm)
    return scale_dependent_bias



def get_var_Pk(k,Pk,ngal,BoxSize,dk=None):
    """
    Computes the theoretical variance of the Power Spectrum

    Parameters
    ----------
    k : numpy ndarray
        k-bins at which the power spectrum is cumputed.
    Pk : numpy ndarray
        The power spectrum of which we want to estimate its variance, computed
        over the k-bins given by k.
    ngal : float
        Number of galaxies/halos/dm-particles used for obtaining the power
        spectrum.
    BoxSize : float
        Size of the box.
    dk : float, optional
        Witdth of the k-bins. If nothing is passed as input, the used value is 
        the lowest k resolved in the box. The default is None.

    Returns
    -------
    var : numpy ndarray
        Estimated variance (\sig^2) of the Pk for each k-bin.

    """
    V = BoxSize**3
    n = ngal/V
    if dk is None:
        dk = 2*np.pi/BoxSize
    var = (Pk+1/n)**2 * 4*np.pi**2 / (V*k**2 *dk)
    return var


def my_model(matter,b1,bphi,As,n):
    k = matter[:,0]
    pkmm = matter[:,1]
    bias = get_scale_dependent_bias(k,b1,bphi)
    return pkmm * bias**2 + As/n


# Cargamos nuestros datos para realizar los ajustes

Nvalue = np.loadtxt('/home/anguren/celia/full_sim_fnl100/Nvalue_R8.txt')


# Generamos arrays vacios para almacenar los parámetros del ajuste P_hh = (b1+bphi*fnl*M)^2*P_mm + As/n

b1=np.zeros_like(Nvalue) 
b1_error=np.zeros_like(Nvalue)
bphi=np.zeros_like(Nvalue)
bphi_error=np.zeros_like(Nvalue)
As=np.zeros_like(Nvalue)
As_error=np.zeros_like(Nvalue)

kmax = 0.1 #Mpc/h
Lbox = 1000. #Mpc/h

for i in range(len(Nvalue)):
    data_halos = np.loadtxt(f'/home/anguren/celia/full_sim_fnl100/power_bin_R8/Pk_selec_{i}.txt')
    nhalos = Nvalue[i]
    n_shotnoise = nhalos/(Lbox**3)
    k_halos = data_halos[:,0]
    tmp_pk_halos = data_halos[:,1]
    
    mask = k_halos < kmax
    ksel = k_halos[mask]
    pk_halos = tmp_pk_halos[mask]
    var_pk = get_var_Pk(ksel,pk_halos,nhalos,BoxSize=Lbox)

    '''
    pkmm = Pk.P(zout,ksel) #Or use here the Pkmm measured directly from the simulation.
                       #(In this case take care of interpolating the Pkmm to the k-values of the halos)
    '''

    pkmm = np.loadtxt("/home/anguren/celia/power_df/Pk_df.txt")[mask][:,1]
    matter = np.column_stack([ksel,pkmm])

    p0 = [2.5, 0, 1]  # Solo para b1, bphi, As
    popt, pcov = curve_fit(
        lambda matter, b1, bphi, As: my_model(matter, b1, bphi, As, n_shotnoise),
        matter,
        pk_halos,
        p0=p0,
        sigma=np.sqrt(var_pk),
        absolute_sigma=True
    )

    print(f'Best fit parameters for bin {i}:')
    print('b1 = ',popt[0],'+/-',np.sqrt(pcov[0,0]))
    print('bphi = ',popt[1],'+/-',np.sqrt(pcov[1,1]))
    print('As = ',popt[2],'+/-',np.sqrt(pcov[2,2]))

    b1[i]=popt[0]
    b1_error[i]=np.sqrt(pcov[0,0])
    bphi[i]=popt[1]
    bphi_error[i]=np.sqrt(pcov[1,1])
    As[i]=popt[2]
    As_error[i]=np.sqrt(pcov[2,2])

    # Creamos una carpeta para guardar los plots
    carpeta = Path(f"/home/anguren/celia/full_sim_fnl100/bias_R8/plots")
    carpeta.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.errorbar(ksel,pk_halos,yerr=np.sqrt(var_pk),fmt='.',capsize=3,label='PNG-UNITsim')
    plt.plot(ksel, my_model(matter, *popt, n_shotnoise), label='Best fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$k$ [h/Mpc]',fontsize=14)
    plt.ylabel(r'$P(k)$ [Mpc/h]$^3$',fontsize=14)
    plt.title(f'b1={b1[i]:.3f}, bphi={bphi[i]:.3f}')
    plt.legend()
    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)
    plt.savefig(carpeta / f"bin_{i}.png", dpi=300)
    plt.close()




b1=np.array(b1).reshape(-1,1)
b1_error=np.array(b1_error).reshape(-1,1)
bphi=np.array(bphi).reshape(-1,1)
bphi_error=np.array(bphi_error).reshape(-1,1)
As = np.array(As).reshape(-1,1)
As_error = np.array(As_error).reshape(-1,1)
output = np.concatenate((b1, b1_error, bphi, bphi_error, As, As_error), axis=1)
np.savetxt('/home/anguren/celia/full_sim_fnl100/bphi_ajuste.txt', output, header='b1 b1_error bphi bphi_error As As_error')
