import numpy as np
from qocag import grape_schroedinger_discrete
from qocag import TargetStateInfidelity,Robustness
from qocag import ControlVariation,generate_save_file_path,ControlBandwidthMax
from qocag import LBFGSB,Adam
total_time_steps=100
#Toltal number of descretized time pieces
target_states=np.array( [0,1])
cost1=TargetStateInfidelity(target_states=target_states)
fluc_para=np.linspace(-10,10,21)*2*np.pi*1e-3
fluc_oper=np.array([[1,0],[0,-1]])
robustness=[fluc_para,fluc_oper]
costs=[cost1]
#Target state is |1>
total_time=100
#Evolution time is 10 ns
H0=0
#Qubit frequency is 1GHZ
H_controls=[np.array([[0,1],[1,0]])]
#Control Hamiltonian is sigma_x
initial_states=np.array([1,0])
#Initial state is |0>
# result=np.load("./out/00000_qubit01.npy",allow_pickle=True).item()
# initial_control=result["control_iter"][-1]
times = np.linspace(0, total_time, total_time_steps+1)
times=np.delete(times, [len(times) - 1])
initial_control=(np.pi/total_time/2)*np.array([np.ones(len(times))])
result1=grape_schroedinger_discrete(total_time_steps,
                                costs, total_time, H0, H_controls,
                                initial_states,max_iteration_num=100,
                                optimizer=Adam(),initial_controls=initial_control,mode="AD",robustness=robustness)