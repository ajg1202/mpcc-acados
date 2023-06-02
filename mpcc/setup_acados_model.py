import json

from acados_template import AcadosModel
from casadi import *


def export_mpcc_vehicle_model(type):

    model = AcadosModel()
    model_bound = types.SimpleNamespace()

    model.name = 'mpcc_' + type + '_vehicle_ode'

    ### States
    X     = MX.sym('X')     # x coordinate in global frame [m]
    Y     = MX.sym('Y')     # y coordinate in global frame [m]
    phi   = MX.sym('phi')   # yaw [rad]
    v_x   = MX.sym('v_x')   # longitudinal velocity [m/s]
    v_y   = MX.sym('v_y')   # lateral velocity [m/s]
    omega = MX.sym('omega') # yaw rate [rad/s]
    s     = MX.sym('s')     # progress [m]
    F_rx  = MX.sym('F_rx')  # longitudinal force on rear tire [N]
    delta = MX.sym('delta') # steering angle [rad]
    model.x = vertcat(X, Y, phi, v_x, v_y, omega, s, F_rx, delta)

    # Inputs
    ds     = MX.sym('ds')       # progress velocity input [m/s]
    dF_rx  = MX.sym('dF_rx')    # derivative of tire force [N/s]
    ddelta = MX.sym('ddelta')   # steering speed [rad/s]
    model.u = vertcat(ds, dF_rx, ddelta)

    ### Time Derivatives of States
    dX     = MX.sym('dX')
    dY     = MX.sym('dY')
    dphi   = MX.sym('dphi')
    dv_x   = MX.sym('dv_x')
    dv_y   = MX.sym('dv_y')
    domega = MX.sym('domega')
    model.xdot = vertcat(dX, dY, dphi, dv_x, dv_y, domega, ds, dF_rx, ddelta)

    ### Parameters
    s_ref     = MX.sym('s_ref')     # progress of last path [m]
    s_X_ref   = MX.sym('s_X_ref')   # global x coordinate of last path [m]
    s_Y_ref   = MX.sym('s_Y_ref')   # global y coordinate of last path [m]
    s_phi_ref = MX.sym('s_phi_ref') # orientation of last path [rad]
    model.p = vertcat(s_ref, s_X_ref, s_Y_ref, s_phi_ref)

    ### Dynamics
    # physical constants
    with open('model_params.json', 'r') as json_file:
        param = json.load(json_file)
    m   = param['m']    # vehicle overall mass [kg]
    l_f = param['l_f']  # distance CoG to front axle [m]
    l_r = param['l_r']  # distance CoG to rear axle [m]
    I_z = param['I_z']  # vehicle overall inertia [kgm^2]
    # Pacejka magic formula coefficients
    B_f   = param['B_f']
    C_f   = param['C_f']
    D_f   = param['D_f']
    B_r   = param['B_r']
    C_r   = param['C_r']
    D_r   = param['D_r']

    # Pacejka magic formula
    alpha_f = -atan2(omega*l_f + v_y, v_x) + delta  # front side slip [rad]
    alpha_r =  atan2(omega*l_r - v_y, v_x)          # rear side slip [rad]
    F_fy = D_f*sin(C_f*atan(B_f*alpha_f))   # lateral force on front tire [N]
    F_ry = D_r*sin(C_r*atan(B_r*alpha_r))   # lateral force on rear tire [N]

    # equations of motion
    if type == 'dynamic':
        model.f_expl_expr = vertcat(
            v_x*cos(phi) - v_y*sin(phi),
            v_x*sin(phi) + v_y*cos(phi),
            omega,
            (F_rx - F_fy*sin(delta))/m + v_y*omega,
            (F_ry + F_fy*cos(delta))/m - v_x*omega,
            (F_fy*l_f*cos(delta) - F_ry*l_r)/I_z,
            ds,
            dF_rx,
            ddelta
            )
        model.f_impl_expr = model.xdot - model.f_expl_expr

    elif type == 'kinematic':
        model.f_expl_expr = vertcat(
            v_x*cos(phi) - v_y*sin(phi),
            v_x*sin(phi) + v_y*cos(phi),
            omega,
            F_rx/m
            (ddelta*v_x + delta*F_rx/m)/(l_f+l_r)*l_r,
            (ddelta*v_x + delta*F_rx/m)/(l_f+l_r),
            ds,
            dF_rx,
            ddelta
            )
        model.f_impl_expr = model.xdot - model.f_expl_expr

    else:
        raise ValueError("Invalid type. Only 'dynamic' or 'kinematic' are aceepted.")


    ### External Cost Expression
    # weight
    Q = np.diag([1.0, 1.0])
    R = np.diag([0.0, 0.0])
    q_s = 1.0

    # linearized point on path according to last path
    X_lin = s_X_ref + (s - s_ref)*cos(s_phi_ref)
    Y_lin = s_Y_ref + (s - s_ref)*sin(s_phi_ref)

    # contouring and lag error
    e_c = sin(s_phi_ref)*(X_lin - X) - cos(s_phi_ref)*(Y_lin - Y)
    e_l = cos(s_phi_ref)*(X_lin - X) + sin(s_phi_ref)*(Y_lin - Y)

    e = vertcat(e_c, e_l)
    u = vertcat(dF_rx, ddelta)
    model.cost_expr_ext_cost = e.T @ Q @ e \
                             + u.T @ R @ u \
                             - q_s * ds
    
    ### Bounds
    with open('model_bounds.json', 'omega') as json_file:
        bound = json.load(json_file)
    # state bounds
    model_bound.v_x_min   = bound['v_x_min']
    model_bound.v_x_max   = bound['v_x_max']
    model_bound.v_y_min   = bound['v_y_min']
    model_bound.v_y_max   = bound['v_y_max']
    model_bound.omega_min = bound['omega_min']
    model_bound.omega_max = bound['omega_max']
    model_bound.F_rx_min  = bound['F_rx_min']
    model_bound.F_rx_max  = bound['F_rx_max']
    model_bound.delta_min = bound['delta_min']
    model_bound.delta_max = bound['delta_max']
    # input bounds
    model_bound.ds_min     = bound['ds_min']
    model_bound.ds_max     = bound['ds_max']
    model_bound.dF_xr_min  = bound['dF_xr_min']
    model_bound.dF_xr_max  = bound['dF_xr_max']
    model_bound.ddelta_min = bound['ddelta_min']
    model_bound.ddelta_max = bound['ddelta_max']

    return model, model_bound
