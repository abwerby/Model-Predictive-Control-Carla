import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize, differential_evolution


class state:
    def __init__(self, X=0, Y=0, TH=0, V=0, CTE=0, ETH=0):
        self.x = X
        self.y = Y
        self.th = TH
        self.v = V
        self.cte = CTE
        self.eth = ETH

class inputs:
    def __init__(self, steer_angle=0, accelartion=0):
        self.steer_angle = steer_angle
        self.accelartion = accelartion

def model(inputs, init_state, coff, dt = 0.1, L = 3):

    final_state = state()

    ## find the final satte after dt of time ###
    final_state.x  = init_state.x  + init_state.v*np.cos(init_state.th)*dt
    final_state.y  = init_state.y  + init_state.v*np.sin(init_state.th)*dt
    final_state.th = init_state.th + (init_state.v/L)*inputs.steer_angle*dt
    final_state.v  = init_state.v  + inputs.accelartion*dt

    th_des = np.arctan(coff[2] + 2*coff[1]*init_state.x + 3*coff[0]*init_state.x**2)
    final_state.cte = np.polyval(coff,init_state.x) - init_state.y + (init_state.v*np.sin(init_state.eth)*dt)
    final_state.eth = init_state.th - th_des + ((init_state.v/L)*inputs.steer_angle*dt)

    return final_state




