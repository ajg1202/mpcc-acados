import numpy as np
import json
from scipy.interpolate import CubicSpline


def get_reference_lane(file_name):

    with open(file_name, 'r') as json_file:
        track = json.load(json_file)

        lane_s = np.array(track['s'])
        lane_center = np.array(track['center'])
        lane_inner  = np.array(track['inner'])
        lane_outer  = np.array(track['outer'])

    return lane_s, lane_center, lane_inner, lane_outer


def get_track_bound(lane_inner, lane_outer, PADDING=0.75):
    ### add safety margin to borders
    v_norm = lane_outer - lane_inner
    v_unit_norm = v_norm / np.linalg.norm(v_norm, axis=1)[:, None]

    bound_inner = lane_inner + PADDING * v_unit_norm
    bound_outer = lane_outer - PADDING * v_unit_norm

    return bound_inner, bound_outer


def parameterize_track(lane_s, lane_center, bound_inner, bound_outer):
    # ### approximate the arc-lenth of the center lane by piecewise linearization 
    # lane_ds = np.hypot(np.diff(lane_center[:, 0]), np.diff(lane_center[:, 1]))
    # lane_s  = np.append(0.0, np.cumsum(ds))
    
    cs_center = CubicSpline(lane_s, lane_center, bc_type='periodic')
    cs_inner  = CubicSpline(lane_s, bound_inner, bc_type='periodic')
    cs_outer  = CubicSpline(lane_s, bound_outer, bc_type='periodic')

    return cs_center, cs_inner, cs_outer
