import numpy as np
from qoc_ag.models.close_system.optimization import grape_schroedinger_discrete
from qoc_ag.costs.targetstateinfidelity import TargetStateInfidelity
from qoc_ag.optimizers.adam import Adam
total_time_steps=10
#Toltal number of descretized time pieces
costs=[TargetStateInfidelity(target_states=np.array([0,1]))]
#Target state is |1>
total_time=10
#Evolution time is 10 ns
H0=np.array([[-1,0],[0,1]])/2
#Qubit frequency is 1GHZ
H_controls=[np.array([[0,1],[1,0]])]
#Control Hamiltonian is sigma_x
initial_states=np.array([1,0])
#Initial state is |0>

grape_schroedinger_discrete(total_time_steps,
                                costs, total_time, H0, H_controls,
                                initial_states,
                                optimizer=Adam(), mode='AG', tol=1e-15)
