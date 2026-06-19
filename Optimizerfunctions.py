import sklearn.gaussian_process as gp 
import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal import Dict, open_docs
from qiskit_metal.toolbox_metal import math_and_overrides
from qiskit_metal.qlibrary.core import QComponent
from collections import OrderedDict
from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.terminations.short_to_ground import ShortToGround
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.core import QComponent
from HCres import HCresonator
import numpy as np
import matplotlib.pyplot as plt
from SQDMetal.PALACE.Eigenmode_Simulation import PALACE_Eigenmode_Simulation
from SQDMetal.Utilities.Materials import MaterialInterface
from SQDMetal.PALACE.Capacitance_Simulation import PALACE_Capacitance_Simulation
import pandas as pd 
import os 
import json 
import pickle
from datetime import datetime, timezone

from skopt import Optimizer 
from skopt.space import Real, Integer

import csv


LOG = '/home/kit/LakeResearch/Palace/HighCapLowGeoInductDesigns/BayseanOpt/optlog.csv'
CHECKPOINT = '/home/kit/LakeResearch/Palace/HighCapLowGeoInductDesigns/BayseanOpt/optimizer.pkl'

SPACE = [
    Real(100, 310, name = "res_width"),
    Real(100, 310, name = "finger_length"),
    Integer(0, 150, name = "fingers_n"),
    Real(2, 6, name = "finger_width"),
    Real(2, 3, name = "finger_gap"),
    Real(1, 10, name = "geo_ind_length"),
    Real(1, 10, name = "geo_ind_thick"),
    Real(10, 80, name = "metal_width"),
    Real(400, 700, name = "res_height"),
    Real(300, 650, name ="cap_section_height"), 
    Real(25, 30, name = 'gap')
]

PARAM_NAMES = [d.name for d in SPACE]

def run_simulation(params):
    design = designs.DesignPlanar({}, overwrite_enabled=True)

    # Set up chip dimensions
    design.chips.main.size.size_x = '600um'
    design.chips.main.size.size_y = '800um'
    design.chips.main.size.size_z = '-200um'
    design.chips.main.size.center_x = '0um'
    design.chips.main.size.center_y = f"-{params['res_height']/2}um"

    res = HCresonator(design=design, name = 'hcres1', options = Dict(
        res_width = f"{params['res_width']}um", 
        finger_length = f"{params['finger_length']}um",
        fingers_n = f'{int(params["fingers_n"])}',
        finger_width = f"{params['finger_width']}um", 
        finger_gap = f"{params['finger_gap']}um",
        geo_ind_length = f"{params['geo_ind_length']}um", 
        geo_ind_thick = f"{params['geo_ind_thick']}um",
        metal_width = f"{params['metal_width']}um", 
        res_height = f"{params['res_height']}um",
        cap_section_height = f"{params['cap_section_height']}um", 
        layer = '1', 
        gap = f"{params['gap']}um"
    ))

    #Eigenmode Simulation Options
    user_defined_options = {
                    "mesh_refinement":  0,                             #refines mesh in PALACE - essetially divides every mesh element in half
                    "dielectric_material": "silicon",                  #choose dielectric material - 'silicon' or 'sapphire'
                    "starting_freq": 8e9,                              #starting frequency in Hz
                    "number_of_freqs": 1,                              #number of eigenmodes to find
                    "solns_to_save": 1,                                #number of electromagnetic field visualizations to save
                    "solver_order": 1,                                 #increasing solver order increases accuracy of simulation, but significantly increases sim time
                    "solver_tol": 1.0e-6,                              #error residual tolerance foriterative solver
                    "solver_maxits": 200,                              #number of solver iterations
                    "fillet_resolution":4,                            #number of vertices per quarter turn on a filleted path
                    "palace_dir":"~/LakeResearch/Palace/build/bin/palace",#"PATH/TO/PALACE/BINARY",
                    "num_cpus": 1  ,                                  #number of cpus to use in the simulation
                    
                    }

    #Creat the Palace Eigenmode simulation
    eigen_sim = PALACE_Eigenmode_Simulation(name ='HCres_EigenSim_demo',                              #name of simulation
                                            metal_design = design,                                      #feed in qiskit metal design
                                            sim_parent_directory = "/home/kit/LakeResearch/Palace/HighCapLowGeoInductDesigns/BayseanOpt/",            #choose directory where mesh file, config file and HPC batch file will be saved
                                            mode = 'simPC',                                               #choose simulation mode 'HPC' or 'simPC'
                                            meshing = 'GMSH',                                         #choose meshing 'GMSH' or 'COMSOL'
                                            user_options = user_defined_options,                        #provide options chosen above
                                            create_files = True)                                        #create mesh, config and HPC batch files

    # Setting Kinetic inductance
    eigen_sim.add_kinetic_inductance(190e-12)

    #Add in metals from layer 1 of the design file
    eigen_sim.add_metallic(1)

    #Add in ground plane for simulation
    eigen_sim.add_ground_plane()

    #Fine mesh
    eigen_sim.fine_mesh_components(['hcres1'], min_size=10e-7, max_size=200e-6, taper_dist_min=2e-7, metals_only=False)

    #Sets up the lossy interfaces for MA, SA and MS interfaces
    eigen_sim.setup_EPR_interfaces(metal_air=MaterialInterface('Aluminium-Vacuum'), substrate_air=MaterialInterface('Silicon-Vacuum'), substrate_metal=MaterialInterface('Silicon-Aluminium'))

    eigen_sim.prepare_simulation()

    eigen_sim.run()

    cap_sim = PALACE_Capacitance_Simulation(name ='HCres_cap_demo',                              #name of simulation
                                        metal_design = design,                                      #feed in qiskit metal design
                                        sim_parent_directory = "/home/kit/LakeResearch/Palace/HighCapLowGeoInductDesigns/BayseanOpt/",            #choose directory where mesh file, config file and HPC batch file will be saved
                                        mode = 'simPC',                                               #choose simulation mode 'HPC' or 'simPC'
                                        meshing = 'GMSH',                                         #choose meshing 'GMSH' or 'COMSOL'
                                        user_options = user_defined_options,                        #provide options chosen above
                                        create_files = True)  
    
    cap_sim.add_metallic(1)

    cap_sim.add_ground_plane()

    cap_sim.fine_mesh_components(['hcres1'], min_size=10e-7, max_size=200e-6, taper_dist_min=2e-7, metals_only=False)

    cap_sim.prepare_simulation()

    cap_sim.run()

    eigen_sim2 = PALACE_Eigenmode_Simulation(name ='HCres_wo_KI2_demo',                              #name of simulation
                                            metal_design = design,                                      #feed in qiskit metal design
                                            sim_parent_directory = "/home/kit/LakeResearch/Palace/HighCapLowGeoInductDesigns/BayseanOpt/",            #choose directory where mesh file, config file and HPC batch file will be saved
                                            mode = 'simPC',                                               #choose simulation mode 'HPC' or 'simPC'
                                            meshing = 'GMSH',                                         #choose meshing 'GMSH' or 'COMSOL'
                                            user_options = user_defined_options,                        #provide options chosen above
                                            create_files = True)                                        #create mesh, config and HPC batch files



    #Add in metals from layer 1 of the design file
    eigen_sim2.add_metallic(1)

    #Add in ground plane for simulation
    eigen_sim2.add_ground_plane()

    #Fine mesh
    eigen_sim2.fine_mesh_components(['hcres1'], min_size=10e-7, max_size=200e-6, taper_dist_min=2e-7, metals_only=False)

    #Sets up the lossy interfaces for MA, SA and MS interfaces
    eigen_sim2.setup_EPR_interfaces(metal_air=MaterialInterface('Aluminium-Vacuum'), substrate_air=MaterialInterface('Silicon-Vacuum'), substrate_metal=MaterialInterface('Silicon-Aluminium'))

    eigen_sim2.prepare_simulation()

    eigen_sim2.run()

    freq_wKI = pd.read_csv("/home/kit/LakeResearch/Palace/HighCapLowGeoInductDesigns/BayseanOpt/HCres_EigenSim_demo/outputFiles/eig.csv").iloc[0,1] * 1e9
    cap_total = pd.read_csv("/home/kit/LakeResearch/Palace/HighCapLowGeoInductDesigns/BayseanOpt/HCres_cap_demo/outputFiles/terminal-C.csv").iloc[1,1] + pd.read_csv("/home/kit/LakeResearch/Palace/HighCapLowGeoInductDesigns/BayseanOpt/HCres_cap_demo/outputFiles/terminal-C.csv").iloc[1,2]
    freq_woKI = pd.read_csv("/home/kit/LakeResearch/Palace/HighCapLowGeoInductDesigns/BayseanOpt/HCres_wo_KI2_demo/outputFiles/eig.csv").iloc[0,1] * 1e9
    L_geo = (1/cap_total) * (1/(2 * np.pi *freq_woKI))**2 
    ratio = (freq_woKI/freq_wKI)**2 - 1 
    L_kin = ratio * L_geo
    eta_k = L_kin / (L_kin + L_geo)
    L_tot = L_geo + L_kin 
    Z = np.sqrt(L_tot/cap_total)


    I_crit = 65e-6 # From High efficiency NbN nanowire superconducting single photon detectors fabricated on MgO substrates from a low temperature process 
    mu0 = 4 * np.pi * 1e-7
    # Assuming a solenoid is applying the field: 
    #n = 1e4 # Typical value according to google in units 1/m
    B = 10e-3
    area = res.get_area_of_res()
    # Assuming a uniform field applied across the resonator: 
    area = area * (1e-3)**2
    flux_ex = B*area 

    x_sweep = np.linspace(-float(params['geo_ind_length']/2)*1e-6, float(params['geo_ind_length']/2)*1e-6, 100)
    y_sweep = np.linspace(-float(params['res_height'] + params['geo_ind_thick'] + 1)*1e-6, -float(params['res_height'] + params['geo_ind_thick'])*1e-6, 100)
    z_sweep = 0


    B_array = np.zeros((np.shape(x_sweep)[0], np.shape(y_sweep)[0]))

    for x in range(len(x_sweep)):
        for y in range(len(y_sweep)): 
            r0 = np.array([x_sweep[x],y_sweep[y],0])
            B_z = res.biot_savart(r0 = r0, N = 10000)[2]
            B_array[x,y] = B_z

    g = 16.6 
    mub = 9.274e-24
    Phi_c = (g * mub * B_array)/2

    coupling_normalized = eta_k * (flux_ex / (I_crit * (L_geo + L_kin)) ** 2 ) * Phi_c
    coupling = freq_wKI * coupling_normalized

    return coupling.max(), freq_wKI, Z, {'Capacitance': cap_total, 'Ind_total': L_tot, 'eta_k': eta_k,
                                         'L_geo': L_geo, 'L_kin': L_kin }

def objective(coupling, freq, Z, lambda_f: float = 0.5, lambda_z: float = 0.5): 
    return coupling*1e-3 - lambda_f *( (freq - 9*1e9)/(9e9) )**2 - lambda_z * ((Z - 50) / 50) **2

def load_create_optimizer(): 
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f: 
            opt = pickle.load(f)
        return opt 
    
    return Optimizer(
        dimensions = SPACE, 
        base_estimator = 'GP', 
        acq_func = "EI",
        n_initial_points = 10, 
        random_state = 22
    )

def load(csv_path, lambda_f = 0.5, lambda_z = 0.5):
    df = pd.read_csv(csv_path)

    xs, ys = [], []
    for _, row in df.iterrows():
        x = [row[name] for name in PARAM_NAMES]
        if "objective" in df.columns and not pd.isna(row["objective"]):
            score = row["objective"]
        else:
            score = objective(row["coupling_max"], row["freq_wKI"], row["Z"],
                                       lambda_f, lambda_z)
        xs.append(x)
        ys.append(-score)  
    return xs, ys




def log_evals(params, coupling, score, freq, Z, extra, error = None):

    row = {
        'time': datetime.now(timezone.utc).isoformat(), 
        **params, 
        'objective': score, 
        'coupling': coupling,
        'frequency': freq, 
        'Impedance': Z, 
        **extra, 
        'error': error,
    }
    write_header = not os.path.exists(LOG)
    with open(LOG, 'a', newline='') as f: 
        writer = csv.DictWriter(f, fieldnames = row.keys())
        if write_header: 
            writer.writeheader()
        writer.writerow(row)
    

def evaluate(x): 

    params = dict(zip(PARAM_NAMES, x))

    try: 
        coupling, freq, Z, extra = run_simulation(params)
        score = objective(coupling, freq, Z)
        log_evals(params, coupling, score, freq, Z, extra)

        return -score
    except Exception as e: 
        log_evals(params,None, None, None, None, {}, error = str(e))
        return 1e9
    
def main(n_calls = 50, prior_csv = None): 
    print("Starting Optimization")
    opt = load_create_optimizer()

    if len(opt.Xi) == 0 and prior_csv: 
        x, y = load(prior_csv)

        opt.tell(x,y)

    for i in range(len(opt.Xi), len(opt.Xi) + n_calls):
        print(f'Iteration {i} of simulation') 
        x = opt.ask()
        y = evaluate(x)
        opt.tell(x,y)

        with open(CHECKPOINT, 'wb') as f: 
            pickle.dump(opt,f)
    print("Finished :)")

    return opt

