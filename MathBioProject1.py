import numpy as np
from math import exp
import matplotlib.pyplot as plt
from scipy.integrate import odeint

C = 1
I = 10.0

E_K = -90
E_L = -80
E_Na = 60

V_Na = -20
V_K = -25
V_L = -50

k_K = 5
k_Na = 15

g_L = 8
g_Na = 20
g_K = 10


'''
Construct the HH model

C,
I, 
E_K,
E_l,
E_Na,
V_Na,
V_k,
V_l,
k_K,
k_Na
'''




def m_inf(V):
    return 1/(1+np.exp((V_K - V)/k_K))


def n_inf(V):
    return 1/(1+np.exp((V_Na - V)/k_Na))


def v_nullcline(V):
    return (I - g_L*(V - E_L) -  g_Na * m_inf(V) * (V - E_Na))/(g_K*(V - E_K))


x = np.arange(-85, 85)
plt.plot(x, n_inf(x), 'r')
plt.plot(x, v_nullcline(x), 'b')
plt.show()



