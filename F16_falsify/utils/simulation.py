from F16_falsify.utils import loader

from numpy import pi, deg2rad
import numpy as np

from CoRec.envs.F16.AeroBenchVVPython.code.RunF16Sim import RunF16Sim
from CoRec.envs.F16.AeroBenchVVPython.code.PassFailAutomaton import FlightLimitsPFA, FlightLimits
from CoRec.envs.F16.AeroBenchVVPython.code.CtrlLimits import CtrlLimits
from CoRec.envs.F16.AeroBenchVVPython.code.Autopilot import GcasAutopilot
from CoRec.envs.F16.AeroBenchVVPython.code.controlledF16 import controlledF16


class NeuralNetLLC:
    """
        Neural Network Controller Interface
    """
    def __init__(self, ctrl_limits, neural_net):
        """
        :param ctrl_limits: Control limitation
        :param neural_net: Stable Baselines Controller
        """
        self.nn = neural_net

        # equilibrium points from BuildLqrControllers.py
        self.xequil = np.array([502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 1000.0, 9.05666543872074], dtype=float).transpose()
        self.uequil = np.array(
            [0.13946204864060271, -0.7495784725828754, 0.0, 0.0], dtype=float).transpose()

        self.ctrlLimits = ctrl_limits

    def get_u_deg(self, u_ref, f16_state):
        # Calculate perturbation from trim state
        x_delta = f16_state.copy()
        x_delta[:len(self.xequil)] -= self.xequil

        # Implement LQR Feedback Control
        # Reorder states to match controller:
        # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
        x_ctrl = np.array([x_delta[i]
                           for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=float)

        # Initialize control vectors
        u_deg = np.zeros((4,))  # throt, ele, ail, rud

        # Calculate control using LQR gains
        u_deg[1:4] = self.nn.predict(x_ctrl, deterministic=True)[0]  # Full Control

        # Set throttle as directed from output of getOuterLoopCtrl(...)
        u_deg[0] = u_ref[3]

        # Add in equilibrium control
        u_deg[0:4] += self.uequil

        # Limit controls to saturation limits
        ctrlLimits = self.ctrlLimits

        # Limit throttle from 0 to 1
        u_deg[0] = max(min(u_deg[0], ctrlLimits.ThrottleMax),
                       ctrlLimits.ThrottleMin)

        # Limit elevator from -25 to 25 deg
        u_deg[1] = max(min(u_deg[1], ctrlLimits.ElevatorMaxDeg),
                       ctrlLimits.ElevatorMinDeg)

        # Limit aileron from -21.5 to 21.5 deg
        u_deg[2] = max(min(u_deg[2], ctrlLimits.AileronMaxDeg),
                       ctrlLimits.AileronMinDeg)

        # Limit rudder from -30 to 30 deg
        u_deg[3] = max(min(u_deg[3], ctrlLimits.RudderMaxDeg),
                       ctrlLimits.RudderMinDeg)

        return x_ctrl, u_deg

    def get_num_integrators(self):
        return 3

    def get_integrator_derivatives(self, t, x_f16, u_ref, x_ctrl, Nz, Ny):
        """
        get the derivatives of the integrators in the low-level controller
        """

        # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
        ps = x_ctrl[4] * np.cos(x_ctrl[0]) + x_ctrl[5] * np.sin(x_ctrl[0])

        # Calculate (side force + yaw rate) term
        Ny_r = Ny + x_ctrl[5]

        return [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]


def gcas_simulation(initial_state, initial_time):
    """
    F16 GCAS simulation
    :param initial_state: The initial state
    :param initial_time: The initial time
    :return: Simulation results
    """
    flight_limits = FlightLimits()
    ctrl_limits = CtrlLimits()
    nn_llc = loader.get_policy("ddpg")
    llc = NeuralNetLLC(ctrl_limits, nn_llc)
    ap = GcasAutopilot(llc.xequil, llc.uequil, flight_limits, ctrl_limits)
    pass_fail = FlightLimitsPFA(flight_limits)
    pass_fail.break_on_error = False

    # Select Desired F-16 Plant
    f16_plant = 'morelli'  # 'stevens' or 'morelli'

    tMax = 15  # simulation time

    xcg_mult = 1.0  # center of gravity multiplier

    val = 1.0  # other aerodynmic coefficient multipliers
    cxt_mult = val
    cyt_mult = val
    czt_mult = val
    clt_mult = val
    cmt_mult = val
    cnt_mult = val

    multipliers = (xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult)

    der_func = lambda t, y: controlledF16(t, y, f16_plant, ap, llc, multipliers=multipliers)[0]

    passed, times, states, modes, ps_list, Nz_list, u_list = RunF16Sim(initial_state,
                                                                       tMax, der_func, f16_plant, ap, llc,
                                                                       pass_fail, multipliers=multipliers,
                                                                       initial_time=initial_time,
                                                                       reset_full_state=True)

    return passed, times, states, modes, ps_list, Nz_list, u_list


def initial_space():
    """
    Self-defined initial space
    :return: lower and upper boundary of initial space box
    """
    # Initial Conditions
    power = 9  # Power

    # Default alpha & beta
    alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)

    # Initial Attitude
    alt = 3600
    Vt = 540  # Pass at Vtg = 540;    Fail at Vtg = 550;
    phi = (pi / 2) * 0.5  # Roll angle from wings level (rad)
    theta = (-pi / 2) * 0.8  # Pitch angle from nose level (rad)
    psi = -pi / 4  # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    center = np.array([Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power, 0, 0, 0])
    state_delta = np.abs(center) * 0.02
    initial_state_low = center - state_delta
    initial_state_high = center + state_delta

    return initial_state_low, initial_state_high


if __name__ == '__main__':
    l, h = initial_space()
    initial_state = (np.random.uniform(l, h))
    passed, times, states, modes, ps_list, Nz_list, u_list = gcas_simulation(initial_state, 0)
    print(passed)
