# PHS3903 - Projet de simulation
# Mini-devoir 1

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import stats

# Paramètres physiques du problème
g = 9.81     # Champ gravitationnel (m²/s)
m = 0.10    # Masse du pendule (kg)
L = 0.50    # Longueur du câble (m)
beta = 0.4  # Constante d'amortissement (1/s)

# Conditions initiales
theta_0 = np.pi/6         # Position initiale (rad)
omega_0 = 5               # Vitesse inititale (rad/s)

# Paramètres généraux de simulation
tf = 10            # Temps final (s)
dt_0 = 0.01418       # Pas de temps le plus élevé (s)
dtemps = np.arange(0,tf,dt_0)

theta_1 = theta_0 + (1-(beta*dt_0)/2)*omega_0*dt_0 - ((g/(2*L))*dt_0**2)*np.sin(theta_0)
theta_list = []
theta_list.append(theta_0)
theta_list.append(theta_1)
for i in range(len(dtemps)-2):
    theta_n = (4*theta_list[i+1]-(2-beta*dt_0)*theta_list[i]-(2*g*(dt_0**2)/L)*np.sin(theta_list[i+1]))/(2+beta*dt_0)
    theta_list.append(theta_n)


plt.plot(dtemps, theta_list, label="dt = 0.01418")
plt.xlabel("temps (s)")
plt.ylabel("Angle avec la verticale (rad)")
plt.title("Position angulaire du pendule par rapport à la verticale en fonction du temps")
plt.legend()
plt.show()


dt = [dt_0, dt_0/2, dt_0/4, dt_0/8, dt_0/16]
Er = []

for t in range(len(dt)):

    tf = 10                       # Temps final (s)
    dt_0 = dt[t]             # Pas de temps le plus élevé (s)
    dtemps = np.arange(0,tf,dt_0)

    theta_1 = theta_0 + (1-(beta*dt_0)/2)*omega_0*dt_0 - ((g/(2*L))*dt_0**2)*np.sin(theta_0)
    theta_list = []
    theta_list.append(theta_0)
    theta_list.append(theta_1)
    for i in range(len(dtemps)-2):
        theta_n = (4*theta_list[i+1]-(2-beta*dt_0)*theta_list[i]-(2*g*(dt_0**2)/L)*np.sin(theta_list[i+1]))/(2+beta*dt_0)
        theta_list.append(theta_n)
    print("Position finale du pendule pour" , dt[t], " = ", theta_list[-1], "rad")
        
for e in range(4):
    Er.append(np.abs(theta_list[e+1]-theta_list[e]))



dt = [dt_0, dt_0/2, dt_0/4, dt_0/8]

# Ordre de convergence avec la pente du graphique en échelle log

m = (np.log(Er[3]) - np.log(Er[2]))/(np.log(dt[3]) - np.log(dt[2]))
print(m, 1/m)

# plt.plot(dt, dt_2, label='2')
# plt.plot(dt, dt_3, label = '3')
plt.plot(dt, Er)
plt.xlabel("pas de temps (s)")
plt.ylabel("AErreur sur la position finale(rad)")
plt.title("Erreur sur la position finale en fonction du pas de temps")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()

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
