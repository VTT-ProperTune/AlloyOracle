import pandas as pd
import numpy as np
import json 
import os

# Function to calculate yield strength and VEC  

def CalculateYieldStrength(x):
    config = {}
    config['element_labels'] = ["Cr",
                                "Al",
                                "Fe",
                                "Ni",
                                "Ti",
                                "V",
                                "Mo",
                                "W",
                                "Mn",
                                "Si",
                                "Co"]
    
    config['yield_strength_temperatures'] = [0, 298, 1100]
    
    # Load element properties to calculate metrics
    script_name = os.path.realpath(__file__).split('/')[-1]
    script_path = os.path.realpath(__file__).split(script_name)[0]
    filepath = os.path.join( script_path, 'element_properties.json')
    with open(filepath, 'r') as json_file:
        element_properties = json.load(json_file)
    
    config['n_valence'] = element_properties['n_valence']
    config['molar_mass'] = element_properties['molar_mass']
    config['mass_density'] = element_properties['mass_density']
    config['elastics'] = element_properties['elastics']
    config['alats_1x1x1'] = element_properties['alats_1x1x1']

    x = x/100  # scale concentrations to range 0...1
    elems = config['element_labels']
    elast = config['elastics']

    """ 1. rule of mixture (weighted sum) volume of each composition (V in the model) """
    n_valence = config['n_valence']
    VEC_rom = []
    for i_comp in range(x.shape[0]):
        temp_VEC = sum([x[i_comp, i_elem] * n_valence[elems[i_elem]]
                       for i_elem in range(len(elems))])
        VEC_rom.append(temp_VEC)

    df = pd.DataFrame(VEC_rom, columns=["VEC_m"])


    # Work with lattice constants instead of the volumes
    alats_1x1x1 = config['alats_1x1x1']

    alat_rom = []
    for i_comp in range(x.shape[0]):
        temp_alat = sum([x[i_comp, i_elem] * alats_1x1x1[elems[i_elem]]
                       for i_elem in range(len(elems))])
        alat_rom.append(temp_alat)

    df["alat_rom(Å)"] = alat_rom 


    df["Vbar_rom(Å)"] = 0.5*(df["alat_rom(Å)"]**3) # Per atom

    """ 2. burgers vector """
    # Burgers vector for each composition
    df["burgers_v(Å)"] = np.sqrt(3)*df["alat_rom(Å)"]/2

    # Misfit volume per atom
    for key in alats_1x1x1.keys():
        df[f"dV_{key}"] = 0.5*(alats_1x1x1[key]**3 - df["alat_rom(Å)"]**3)

    """Calculate the rule of mixture weighted sum elastic constants for each composition in the df """

    C11_rom, C12_rom, C44_rom = [], [], []

    for i_comp in range(x.shape[0]):
        C11_rom.append(sum([x[i_comp, i_elem] * elast[elems[i_elem]][0]
                       for i_elem in range(len(elems))]))
        C12_rom.append(sum([x[i_comp, i_elem] * elast[elems[i_elem]][1]
                       for i_elem in range(len(elems))]))
        C44_rom.append(sum([x[i_comp, i_elem] * elast[elems[i_elem]][2]
                       for i_elem in range(len(elems))]))

    df['C11(GPa)'] = C11_rom
    df['C12(GPa)'] = C12_rom
    df['C44(GPa)'] = C44_rom

    # print(df["C11(GPa)"].head())

    # Bulk modulus
    df["B(GPa)"] = (df["C11(GPa)"] + 2*df["C12(GPa)"]) / 3
    
    # Shear modulus
    df["mu(GPa)"] = np.sqrt(1/2*df["C44(GPa)"]
                            * (df["C11(GPa)"] - df["C12(GPa)"]))
    # Poisson ratio
    df["nu"] = (3*df["B(GPa)"] - 2*df["mu(GPa)"]) / \
        (2*(3*df["B(GPa)"] + df["mu(GPa)"]))

    # Calculate the yield strength model
    """
    Zero temperature strength
    - Eq (1) in https://doi.org/10.1016/j.actamat.2022.118132 written as tau = a*b*c
    """

    a = 0.040*(1/12)**(-1/3)*df["mu(GPa)"]*1e9  # (Pa)
    b = ((1+df["nu"])/(1-df["nu"]))**(4/3)  # (1)

    # Sum of the concentrations times corresponding misfit volumes squared
    c_sum = sum([x[:, i_elem] * df['dV_{}'.format(elems[i_elem])]
                ** 2 for i_elem in range(len(elems))])

    c = ((c_sum)/(df["burgers_v(Å)"]**6))**(2/3)

    ys_0 = a*b*c  # Pa

    df["ys_0(GPa)"] = ys_0/1e9  # GPa

    # Calculate also the misfit parameter delta
    misfit_delta = (c_sum**(1/2)) / (3*0.5*(df["alat_rom(Å)"]**3)) # Factor 0.5 for volume per atom !
    df["misfit_delta"] = misfit_delta*100 # if > 3.5 -> Edge

    """
    Zero temperature energy barrier:
    Eq. (2) in https://doi.org/10.1016/j.actamat.2022.118132 written as: Eb = h*j*k
    """

    h = 2.00*(1/12)**(1/3)*(df["mu(GPa)"]*1E9) * \
        (df["burgers_v(Å)"]*1E-10)**3  # N*m = J

    j = ((1+df["nu"])/(1-df["nu"]))**(2/3)  # Dimensionless

    k_sum = sum([x[:, i_elem] * ((df['dV_{}'.format(elems[i_elem])]*1E-30)**2)
                for i_elem in range(len(elems))])  # Å^6

    k = ((k_sum)/((df["burgers_v(Å)"]*1E-10)**6))**(1/3)  # Dimensionless

    Eb = h*j*k  # Joules

    df["Eb_0(J)"] = Eb
    # df["Eb_0(J)"]*(6.241509*1E18) ## eV

    """ Solve the temperature dependent yield strength """
    ys_ref = {}
    for T in config['yield_strength_temperatures']:
        ys_T = ys(T, df["ys_0(GPa)"], df["Eb_0(J)"])
        df[f"ys_{int(T)}K"] = ys_T.values
        ys_ref[f"ys_{int(T)}K"] = ys_T.values

    return ys_ref, df

###############################################################################
def ys(T, ys0, Eb, eps0=1E4, eps=1E-3):
    """Temperature dependent polycrystal yield strength in GPa"""
    taylorfactor = 3.067
    kB = 1.380649*1E-23  # J/K
    prefactor = -1/0.55
    fraction = (kB*T)/(Eb)
    return taylorfactor*ys0*np.exp(prefactor*((fraction)*np.log(eps0/eps))**0.91)

if __name__ == '__main__':
   x = [50,5,10,10,5,20,0,0,0,0,0]
   print("x:", x)
   print("shape(x):", np.shape(x))
   x = np.array(x).reshape(1,-1)
   print("x:", x)
   print("shape(x):", np.shape(x))
   y_s, ys_data = CalculateYieldStrength(x) #, config)

   strength_label = 'ys_298K'
   for label in ['VEC_m', strength_label, "alat_rom(Å)"]:
       print( np.min( ys_data[label]), "<=", label, "<=", np.max(ys_data[label]) )
