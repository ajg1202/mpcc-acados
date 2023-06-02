import numpy as np
import matplotlib.pyplot as plt

from track import get_reference_lane, get_track_bound, parameterize_track
from setup_acados_solver import export_acados_solver
from plot_utils import plot_track


def main():

    ### Hyperparameters
    N = 50
    N_sim = 1000
    tf = 1.0 # [s]

    ### Track
    s_lane, lane_center, lane_inner, lane_outer = get_reference_lane('track.json')

    bound_inner, bound_outer = get_track_bound(lane_inner, lane_outer)
    
    cs_center, cs_inner, cs_outer = parameterize_track(s_lane, lane_center, bound_inner, bound_outer)
    cs_d_center = cs_center.derivative()

    ### Acados Solvers
    acados_ocp_solver_dyn, acados_sim_solver_dyn = export_acados_solver(N, tf, 'dynamic')
    acados_ocp_solver_kin, acados_sim_solver_kin = export_acados_solver(N, tf, 'kinematic')

    nx = acados_ocp_solver_kin.acados_ocp.dims.nx
    nu = acados_ocp_solver_kin.acados_ocp.dims.nu

    ### Initial Conditions
    x_sol = np.zeros((N, nx))
    u_sol = np.zeros((N, nu))

    s       = np.zeros((N, ))
    s_pos   = cs_center(s)
    s_d_pos = cs_d_center(s)

    s_X   = s_pos[:, 0]
    s_Y   = s_pos[:, 1]
    s_phi = np.arctan2(s_d_pos[:, 1], s_d_pos[:, 0])

    x_sol[:, 0]  = s
    x_sol[:, 1]  = s_X
    x_sol[:, 2]  = s_Y
    x_sol[:, 3]  = s_phi
    x_sol[:, 4:] = np.zeros((N, nx-4))

    x_0 = x_sol[0, :]

    ### Simuation
    X_sim = np.zeros((N_sim, nx))
    for t in range(N_sim):

        # initialize optimal control problem
        for j in range(N):
            # # stage-wise polytopic soft constraints
            # # gl <= Cx + Du + Jsg*sg <= gu
            # acados_ocp_solver_kin.constraints_set(i, 'lg', lg[i])
            # acados_ocp_solver_kin.constraints_set(i, 'ug', ug[i])
            # acados_ocp_solver_kin.constraints_set(i, 'C', C[i])
            # acados_ocp_solver_kin.constraints_set(i, 'D', D[i])
        
            # set stages for linearized error
            p = np.array([s[j], s_X[j], s_Y[j], s_phi[j]])
            acados_ocp_solver_kin.set(j, 'p', p)
        # duplicate last stage of last solution for last stage
        acados_ocp_solver_kin.set(N, 'p', p)

        # update initial condition
        acados_ocp_solver_kin.set(0, 'lbx', x_0)
        acados_ocp_solver_kin.set(0, 'ubx', x_0)

        # solve OCP
        status_ocp = acados_ocp_solver_kin.solve()

        # extract solution
        for k in range(N):
            u_sol[k] = acados_ocp_solver_kin.get(k, 'u')
            x_sol[k] = acados_ocp_solver_kin.get(k+1, 'x')
        
        # calculate linearization points based on last solution
        s       = x_sol[:, 0]
        s_pos   = cs_center(s)
        s_d_pos = cs_d_center(s)

        s_X   = s_pos[:, 0]
        s_Y   = s_pos[:, 1]
        s_phi = np.arctan2(s_d_pos[:, 1], s_d_pos[:, 0])

        # simulate system
        acados_sim_solver_kin.set('x', x_0)
        acados_sim_solver_kin.set('u', u_sol[0])

        status_sim = acados_sim_solver_kin.solve()

        x_0 = acados_sim_solver_kin.get('x')
        X_sim[t] = x_0

    plot_track(lane_center, lane_inner, lane_outer)


if __name__ == "__main__":
    main()
