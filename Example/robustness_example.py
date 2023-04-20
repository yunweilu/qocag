import numpy as np
from qocag import grape_schroedinger_discrete
from qutip import *
from qocag import TargetStateInfidelity,Robustness,ForbidStates
from qocag import ControlBandwidthMax
from qocag import Adam,LBFGSB
def Rx(matrix,angle):
    matrix[0,0]=np.cos(angle/2)
    matrix[0,1]=-1j*np.sin(angle/2)
    matrix[1,0]=-1j*np.sin(angle/2)
    matrix[1,1]=np.cos(angle/2)
    return matrix
N_q = 5
a_q = destroy(N_q)
n_q = a_q.dag() * a_q
x_q = a_q + a_q.dag()
w_q = 0
k_q = -200e-3 * 2*np.pi  # anharmonicity/2
cross = 0e-3 * 2*np.pi
# without -1/2, strange result when change time origin
H0 = (w_q + cross) * (n_q ) + 1/2*k_q * a_q.dag()**2 * a_q**2
Hcx=a_q+a_q.dag()
H0=H0.data.toarray()
Hcx=Hcx.data.toarray()
H_controls=[Hcx]
total_time_steps=10
#Toltal number of descretized time pieces
target_states=np.zeros([N_q,N_q],dtype=complex)
angle= np.pi
target_states=Rx(target_states,angle)
total_time=10
cost1=TargetStateInfidelity(target_states=target_states,subspace_dim=2)
fluc_para=[np.arange(-10,11)*2*np.pi*1e-3]
def fluc_oper(para):
    return n_q.data.toarray()*para
robustness=[fluc_para,fluc_oper]
robust_operator=np.array([n_q.data.toarray()])
cost2=Robustness(robust_operator,cost_multiplier=[0.2e-1,0.1e-1,0e-1],order=3)
f1=np.zeros((N_q,N_q))
f1[4,0]=1
f2=np.zeros((N_q,N_q))
f2[4,1]=1
cost4=ForbidStates(np.array([f1,f2]),cost_multiplier=1)
costs=[cost1,cost4,cost2]

#Target state is |1>

#Evolution time is 10 ns
#Qubit frequency is 1GHZ
#Control Hamiltonian is sigma_x
initial_states=np.identity(N_q)
def impose_bc(controls):
    controls[0][0]=0
    controls[0][-1]=0
    return controls

times = np.linspace(0, total_time, total_time_steps+1)
times=np.delete(times, [len(times) - 1])
def first_order(times,angle,):
    total_time=times[-1]+times[-1]-times[-2]
    a=-2.159224
    control=angle/2*np.ones(len(times))+(a-angle/2)*np.cos(2*np.pi/total_time*times)-a*np.cos(4*np.pi/total_time*times)
    return control/total_time
initial_controlx=angle/2/total_time*np.ones(len(times))
initial_controly=1e-2*np.ones(len(times))
initial_control=[initial_controlx]

result1=grape_schroedinger_discrete(total_time_steps,
                                costs, total_time, H0, H_controls,
                                initial_states,max_iteration_num=1,
                                optimizer=Adam(),initial_controls=initial_control,mode="AD")
print(result1.control_iter[-1])