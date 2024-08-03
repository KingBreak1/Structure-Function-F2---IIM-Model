from multiprocessing import Pool
import numpy as np
from scipy.special import kv
import cmath
from scipy.integrate import nquad
from time import perf_counter
import pandas as pd
import matplotlib.pyplot as plt
import locale
import warnings

warnings.filterwarnings("ignore")


Q2 = 22  # virtuality ( ENTER DESIRED VALUE )

 # constants taken from https://arxiv.org/pdf/1307.0825.pdf (and references)

alpha_elm = 1.0/137.0  # Fine-structure constant
Nc = 3.0  # number of colors
N0 = 0.7  # initial dipole scattering amplitude
sigma0 = (24.064/0.3894)  # free parameter
lanbda = 0.227  # free parameter
x0 = (2.22)*((10)**(-5))  # free parameter
gama = 0.719 # saddle point near the saturation line (considering heavy quarks)
kappa = 9.9 # value from the LO BFKL kernel
Q0 = x0**(lanbda/2)  # saturation scale at initial conditions

alpha = -((N0*gama)**2.0)/(((1.0-N0)**2.0)*(np.log(1.0-N0)))  # IIM model alpha factor
beta = 0.5*((1.0-N0)**(-(1-N0)/(N0*gama)))  # IIM model beta factor


def calculate(parameter_list):  # Calculation of F2

    ie, im, qs, kly = parameter_list # list of parameter for the calculation (charge, mass, qs, kly)

    def Qf2(z):
        return z*(1-z)*Q2+(im)**2

    def K_0(r, z):  # modified bessel function second kind, order 0
        return (kv(0, r*cmath.sqrt(Qf2(z))))**2

    def K_1(r, z):  # modified bessel function second kind, order 1
        return (kv(1, r*cmath.sqrt(Qf2(z))))**2

    def psi_t(r, z):  # squared modulus of the transverse photon wavefunction
        return (((ie)**2)*(alpha_elm*Nc/((2)*(np.pi)**2))*(((z**2)+((1-z)**2))*(Qf2(z))*(K_1(r, z)) + (im**2)*(K_0(r, z))))

    def psi_l(r, z):  # squared modulus of the longitudinal photon wavefunction
        return (((ie)**2)*(alpha_elm*Nc/((2)*(np.pi)**2))*(4)*(Q2)*(z**2)*((1-z)**2)*(K_0(r, z)))

    def Sig_dip1(r):  # dipole-target cross section using IIM model for r*qs <= 2
        return (sigma0)*(N0*((r*qs)/2.0)**(2.0*(gama+((np.log(2.0/(r*qs)))/kly))))

    def Sig_dip2(r):  # dipole-target cross section using IIM model for r*qs > 2
        return (sigma0)*(1.0 - np.exp(-alpha*(np.log(beta*(r*qs)))**2))

    def sigmat(r, z):  # function to be integrated for the transversal cross section calculation
        return 2*np.pi*r*psi_t(r, z)*(Sig_dip1(r) if ((r*qs) <= 2) else Sig_dip2(r))

    def sigmal(r, z):  # function to be integrated for the longitudinal cross section calculation
        return 2*np.pi*r*psi_l(r, z)*(Sig_dip1(r) if ((r*qs) <= 2) else Sig_dip2(r))

    options = {'limit': 1000000}

    sig_t, erro1 = nquad(sigmat, [[0, np.inf], [0, 1]], opts=[options, options])  # Transversal cross section

    sig_l, erro2 = nquad(sigmal, [[0, np.inf], [0, 1]], opts=[options, options])  # Longitudinal cross section

    f2 = (Q2/((4*np.pi**2)*alpha_elm))*(sig_t + sig_l)  # structure function F2

    return ((ie, im, qs, kly), f2)


def main():
    start = perf_counter()
    print('>>> Starting calculation')


    df = pd.read_excel('experimental_data.xlsx', header=None,index_col=False, names=['Q2', 'x', 'sigma', 'erro']) # reads the excel document labeled "experimental_data.xlsx" that contains all the experimental information for x < 10^-2


    XA = df.loc[df['Q2'] == Q2, 'x'].to_list()  # lists the experimental values of x that correspond to the selected virtuality
    SA = df.loc[df['Q2'] == Q2, 'sigma'].to_list()  # lists the experimental values of the total cross section that correspond to the selected virtuality
    EA = df.loc[df['Q2'] == Q2, 'erro'].to_list() # lists the values of the experimental error that correspond to the selected virtuality


    if Q2 in df['Q2'].to_list():

        if len(XA)==1:
            values_x = [eval(XA[0])-eval(XA[0])*0.1,eval(XA[0]),eval(XA[0])+eval(XA[0])*0.1]
            list_x = values_x
            quantity = len(list_x)
        else:
            lower_limit_Y = -np.log(eval(XA[-1]))
            upper_limit_Y = -np.log(eval(XA[0]))

            values_x = np.exp(-np.linspace(lower_limit_Y, upper_limit_Y, num=10))
            list_x = list(values_x)  
            quantity = len(list_x)  
            
    else: 
        values_x = np.exp(-np.linspace(-np.log(10**-2), -np.log(10**-6), num=10))
        list_x = list(values_x)  
        quantity = len(list_x)  

    # constants taken from https://arxiv.org/pdf/1307.0825.pdf (and references)

    light_quarks_m = [(10**(-2)), (10**(-2)), (10**(-2))] # light-quark masses (up, down, strange, respectively)
    light_quarks_e = [2/3, -1/3, -1/3] # light-quark charges (up, down, strange, respectively)
    charm_quarks_m = [1.27]  # charm quark mass
    charm_quarks_e = [2/3]  # charm quark charge

    light_quarks_qs = []  # values of saturation scale (light-quarks)
    light_quarks_kly = []  # values of kappa*lambda*rapidity (light-quarks)
    for x in values_x:
        qs_light = np.sqrt(Q0*Q0*(np.exp(lanbda*(np.log(1.0/x)))))
        light_quarks_qs.append(qs_light)

        kly_light = kappa*lanbda*(np.log(1.0/x))
        light_quarks_kly.append(kly_light)

    XP = []  # values of Bjornken x for heavy quarks (charm quark)
    for x in values_x:
        y = x*(1+(4*(charm_quarks_m[0])**2)/Q2)
        XP.append(y)

    charm_quarks_qs = []  # values of saturation scale using XP (charm quark)
    charm_quarks_kly = [] # values of kappa*lambda*rapidity using XP (charm quark)
    for n in range(len(XP)):
        qs_charm = np.sqrt(Q0*Q0*(np.exp(lanbda*(np.log(1.0/XP[n])))))
        charm_quarks_qs.append(qs_charm)

        kly_charm = kappa*lanbda*(np.log(1.0/XP[n]))
        charm_quarks_kly.append(kly_charm)

   # Dictionary with keys as quark types and values as the parameters associated with those quarks
    quark_types = {'Light': [light_quarks_e, light_quarks_m, light_quarks_qs, light_quarks_kly], 'Charm': [charm_quarks_e, charm_quarks_m, charm_quarks_qs, charm_quarks_kly]}

    parameters = {}
    for type, value in quark_types.items():  # run through the quark types and their parameter values
        e = value[0]  # filter values of charge
        m = value[1]  # filter values of mass
        qs = value[2]  # filter values of qs
        kly = value[3]  # filter values of kly

        for e_m in zip(e, m):
            for qs_kly in zip(qs, kly):
                key_results = (e_m[0], e_m[1], qs_kly[0], qs_kly[1])
                if(key_results not in parameters.keys()):  # filter only distinct parameter sets
                    parameters[key_results] = [e_m[0], e_m[1], qs_kly[0], qs_kly[1]]

    calc_results = []
    with Pool() as pool:
        calc_results = pool.map(calculate, parameters.values()) # do the calculation with parallelism

    indexed_results = {}
    all_results = []
    for nr, tr in enumerate(calc_results):
        indexed_results[tr[0]] = tr[1] # select the result of F2 based on the parameters
    for type, value in quark_types.items():
        e = value[0]
        m = value[1]
        qs = value[2]
        kly = value[3]

        for e_m in zip(e, m):
            for qs_kly in zip(qs, kly):
                key_results = (e_m[0], e_m[1], qs_kly[0], qs_kly[1])
                all_results.append(indexed_results[key_results]) # run through all parameter sets (even the repeated ones) and put the values of F2 corresponding to those parameters in a list (if there are repeated parameter sets, the values of F2 corresponding to those will also be repeated)
                                                                 


    splited_all_results = np.array_split(all_results, len(all_results)/quantity) # split the list of all values of F2 between the quark flavours

    list_F2 = [up + down + strange + charm for up, down, strange, charm in zip(splited_all_results[0], splited_all_results[1], splited_all_results[2], splited_all_results[3])]  # total F2 list (theoretical)

    list_x1 = []  # values of Bjorken x (experimental)
    list_F21 = []  # total structure function (experimental)
    list_err1 = []  # values of the experimental error

    plt.figure()

    for j in range(len(XA)):

        list_x1.append(eval(XA[j]))

        sig1 = (SA[j]) * (1/389)  # CONVERTS FROM MICROBARN TO GeV^-2
        F21 = (Q2 / (4 * np.pi ** 2 * alpha_elm)) * sig1
        list_F21.append(F21)

        err1 = (EA[j]) * (1/389)  # CONVERTS FROM MICROBARN TO GeV^-2
        F2_err1 = (Q2 / (4 * np.pi ** 2 * alpha_elm)) * err1
        list_err1.append(F2_err1)

    locale.setlocale(locale.LC_NUMERIC, "de_DE")
    plt.rcdefaults()
    plt.rcParams['axes.formatter.use_locale'] = True
    plt.plot(list_x, list_F2)
    plt.plot(list_x1, list_F21, '.', color='black')
    plt.errorbar(list_x1, list_F21, yerr=list_err1, fmt='|k')
    plt.xscale('log')
    plt.title(f"$Q^2={Q2:.3f}".replace('.', ',') + " \\, \\mathrm{GeV^2}$", fontsize=14)
    plt.ylabel('$F_2$', fontsize=14)
    plt.xlabel('x', fontsize=14)
    plt.grid(linestyle='-')

    print('>>> Finishing, total time spent: %s' % ((perf_counter() - start)))

    plt.show()

if __name__ == "__main__":
    main()
