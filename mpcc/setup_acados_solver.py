import numpy as np

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from setup_acados_model import export_mpcc_vehicle_model


def export_acados_solver(N, tf, type):
    
    ocp = AcadosOcp()

    model, model_bound = export_mpcc_vehicle_model(type)
    ocp.model = model

    ### Dimension
    ocp.dims.N = N
    ocp.dims.nx = model.x.size()[0]
    ocp.dims.nu = model.u.size()[0]
    ocp.dims.np = model.p.size()[0]
    
    ### Cost
    ocp.cost.cost_type = 'EXTERNAL' # declared in model.cost_expr_ext_cost

    ### constraints
    # state constraints : lbx <= Jbx*x <= ubx
    ocp.constraints.lbx = np.array([model_bound.v_x_min,
                                    model_bound.v_y_min,
                                    model_bound.omega_min,
                                    model_bound.F_rx_min,
                                    model_bound.delta_min])
    ocp.constraints.ubx = np.array([model_bound.v_x_max,
                                    model_bound.v_y_max,
                                    model_bound.omega_max,
                                    model_bound.F_rx_max,
                                    model_bound.delta_max])
    ocp.constraints.idxbx = np.array([3, 4, 5, 7, 8])
    ocp.dims.nbx = ocp.constraints.idxbx.size
    
    # input constraints : lbu <= Jbu*u <= ubu
    ocp.constraints.lbu = np.array([model_bound.ds_min,
                                    model_bound.dF_rx_min,
                                    model_bound.ddelta_min])
    ocp.constraints.ubu = np.array([model_bound.ds_max,
                                    model_bound.dF_rx_max,
                                    model_bound.ddelta_max])
    ocp.constraints.idxbu = np.arange(ocp.dims.nu)
    ocp.dims.nbu = ocp.constraints.idxbu.size

    # # general polytopic constraints : lg <= C*x + D*u + Jsg*sg <= ug
    # ocp.constraints.lg  = np.array([0.0])
    # ocp.constraints.ug  = np.array([0.0])
    # ocp.constraints.lsg = np.array([0.0])
    # ocp.constraints.usg = np.array([0.0])
    # ocp.constraints.C   = np.ones((1, 2))
    # ocp.constraints.D   = np.zeros((1, 2))
    # ocp.dims.ng = 1
    # ocp.dims.ns = ocp.dims.ng

    ### Initialization
    ocp.constraints.x0 = np.zeros(ocp.dims.nx)
    ocp.parameter_values = np.zeros(ocp.dims.np)

    ### Solver options
    ocp.solver_options.hessian_approx = 'EXACT'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_step_length = 0.05
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.tf = tf
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.regularize_method = 'CONVEXIFY'

    acados_ocp_solver = AcadosOcpSolver(ocp, json_file='acados_ocp' + type + '.json')
    acados_sim_solver = AcadosSimSolver(ocp, json_file='acados_ocp' + type + '.json')

    return acados_ocp_solver, acados_sim_solver
