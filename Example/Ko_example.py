import numpy as np
from qocag import KOTargetStateInfidelity
from qocag import ControlVariation,generate_save_file_path,ControlBandwidthMax
from qocag import LBFGSB,Adam
from qocag import grape_keldysh_discrete
total_time_steps=400
#Toltal number of descretized time pieces
target_states=np.array([0,1])
cost=KOTargetStateInfidelity(target_states=target_states)
costs=[cost]
#Target state is |1>
total_time=10
#Evolution time is 10 ns
H0=np.array([[1,0],[0,-1]])*2*np.pi/2
#Qubit frequency is 1GHZ
H_controls=[np.array([[0,1],[1,0]])]
#Control Hamiltonian is sigma_x
initial_states=np.array([1,0])
#Initial state is |0>
# result=np.load("./out/00000_qubit01.npy",allow_pickle=True).item()
# initial_control=result["control_iter"][-1]
times = np.linspace(0, total_time, total_time_steps+1)
times=np.delete(times, [len(times) - 1])
initial_control=(np.pi/total_time)*np.array([np.cos(2*np.pi*times)])
def S_B(w):
    if w>0:
        return 1e2*w
    else:
        return 0
result=grape_keldysh_discrete(total_time_steps,
                                costs, total_time, H0, H_controls,
                                initial_states,H_controls[0],max_iteration_num=1,
                                optimizer=Adam(),initial_controls=initial_control,noise_spectrum=S_B)