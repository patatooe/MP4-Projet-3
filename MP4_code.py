# PHS3903 - Projet de simulation
# Mini-devoir 1

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats
from scipy.optimize import curve_fit

# Paramètres physiques du problème
g = 9.81     # Champ gravitationnel (m²/s)
m = 0.10    # Masse du pendule (kg)
L = 0.50    # Longueur du câble (m)
beta = 0.4  # Constante d'amortissement (1/s)

# Conditions initiales
theta_0 = (np.pi)/6         # Position initiale (rad)
omega_0 = 5               # Vitesse inititale (rad/s)

# Paramètres généraux de simulation
tf = 10             # Temps final (s)
dt_0 = 0.008        # Pas de temps le plus élevé (s) (initialement 0.01418 mais change pour 0.008 afin qu'il soit un multiple de 10)
# dtemps = np.arange(0,tf,dt_0)   #liste de temps
dtemps = np.linspace(0,tf,1250)   #liste de temps ou dernier t = 10
print(dtemps[-1])

#Initiation de la methode numerique
theta_1 = theta_0 + (1-(beta*dt_0)/2)*omega_0*dt_0 - ((g/(2*L))*dt_0**2)*np.sin(theta_0) #Definition de theta_1
theta_list = [theta_0, theta_1]
for i in range(len(dtemps)-2): #Iteration de la methode
    theta_n = (4*theta_list[i+1]-(2-beta*dt_0)*theta_list[i]-(2*g*(dt_0**2)/L)*np.sin(theta_list[i+1]))/(2+beta*dt_0)
    theta_list.append(theta_n)

#Representation graphique de theta en fonction du temps (0 a 10s avec des pas de 0.008s)
plt.plot(dtemps, theta_list, label="dt = 0.01418")
plt.xlabel("temps (s)")
plt.ylabel("Angle avec la verticale (rad)")
plt.title("Position angulaire du pendule par rapport à la verticale en fonction du temps")
plt.legend()
plt.show()

#Differents pas de temps
dt = [dt_0, dt_0/2, dt_0/4, dt_0/8, dt_0/16]
Er = []
theta_final = []

for t in range(len(dt)):
    tf = 10                       # Temps final (s)
    dt_0 = dt[t]                  # Pas de temps le plus élevé (s)
    dtemps = np.arange(0,tf,dt_0)
    
    #Methode similaire a celle expliquee plus haut
    theta_1 = theta_0 + (1-(beta*dt_0)/2)*omega_0*dt_0 - ((g/(2*L))*dt_0**2)*np.sin(theta_0)
    theta_list = [theta_0, theta_1]
    for i in range(len(dtemps)-2):
        theta_n = (4*theta_list[i+1]-(2-beta*dt_0)*theta_list[i]-(2*g*(dt_0**2)/L)*np.sin(theta_list[i+1]))/(2+beta*dt_0)
        theta_list.append(theta_n)
    print("Position finale du pendule pour" , dt[t], " = ", theta_list[-1], "rad")
    theta_final.append(theta_list[-1])

#Definition de l'erreur 
for e in range(4):
    Er.append(np.abs(theta_final[e]-theta_final[e+1]))
Er = np.array(Er)
dt = np.array([dt_0, dt_0/2, dt_0/4, dt_0/8])

#Fit lineaire afin de savoir l'ordre de convergence (pente du graphique)
def f(x,a,b):
    return a*x+b
popt, cov = curve_fit(f, np.log(dt), np.log(Er))
print(popt[0])
#Deuxieme methode afin de s'assurer que la pente du graphique est bel et bien valide
m = (np.log(Er[3]) - np.log(Er[2]))/(np.log(dt[3]) - np.log(dt[2]))
print(m, 1/m)

#Representation de l'erreur en fonction du pas de temps en echelle logarithmique
plt.plot((dt), (Er))
plt.xlabel("pas de temps (s)")
plt.ylabel("Erreur sur la position finale(rad)")
plt.title("Erreur sur la position finale en fonction du pas de temps")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()

plt.plot(dt, f(dt,popt[0],popt[1]))
plt.show()


#Code suggere sur moodle
# Boucle sur le nombre de simulations
M = 5                                       # Nombre de simulations
dt_val = [1, 2, 3, 4, 5]                    # Vecteur des pas de temps pour chaque simulation
theta_f = np.zeros(M)                        # Vecteur des positions finales pour chaque simulation

for m in range(0,M):
# Paramètres spécifiques de la simulation
    dt = dt_val[m]               # Pas de temps de la simulation
    N = 100                      # Nombre d'itérations (conseil : s'assurer que dt soit un multiple entier de tf)

# Initialisation
    t = np.arange(0, tf + dt, dt)  # Vecteur des valeurs t_n
    theta = np.zeros(N + 1)  # Vecteur des valeurs theta_n
    theta[0] = 0
    theta[1] = 0

# Exécution
    for n in range(2, N + 1):
        theta[n] = theta[n-1]

    theta_f[m] = theta[-1]  # Position au temps final tf
