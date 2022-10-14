import numpy as np
from qocag.models.close_system.optimization import grape_schroedinger_discrete
from qocag.costs.targetstateinfidelity import TargetStateInfidelity
from qocag import ControlVariation,generate_save_file_path,ControlBandwidthMax
from qocag import LBFGSB,Adam
total_time_steps=100
#Toltal number of descretized time pieces
cost1=TargetStateInfidelity(target_states=np.array([0,1]))
cost2=ControlVariation(control_num=1,total_time_steps=100,cost_multiplier=0.1,order=1)
cost3=ControlVariation(control_num=1,total_time_steps=100,cost_multiplier=0.1,order=2)
cost4=ControlBandwidthMax(control_num=1,total_time_steps=100,cost_multiplier=0.02,evolution_time=10,max_bandwidths=np.array([1.1]))
costs=[cost1,cost3]
#Target state is |1>
total_time=10
#Evolution time is 10 ns
H0=np.array([[-1,0],[0,1]])*2*np.pi/2
#Qubit frequency is 1GHZ
H_controls=[np.array([[0,1],[1,0]])]
#Control Hamiltonian is sigma_x
initial_states=np.array([1,0])
#Initial state is |0>
save_file_path=generate_save_file_path("qubit01bc","./out")
# result=np.load("./out/00000_qubit01.npy",allow_pickle=True).item()
# initial_control=result["control_iter"][-1]
def impose_bc(controls):
    controls[0][0]=0
    controls[0][-1]=0
    return controls
initial_control=0.001*np.ones((1,total_time_steps))
result=grape_schroedinger_discrete(total_time_steps,
                                costs, total_time, H0, H_controls,
                                initial_states,max_iteration_num=1000,
                                optimizer=Adam(), mode='AD', tol=1e-15,initial_controls=initial_control,save_file_path=save_file_path,
                                   impose_control_conditions=impose_bc)
