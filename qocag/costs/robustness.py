import autograd.numpy as anp
from qocag.functions.common import conjugate_transpose_ad
import numpy as np
class Robustness():
    name = "Robustness"
    requires_step_evaluation = False

    def __init__(self, robust_operator: np.ndarray, cost_multiplier :float = 1.,order : int = 1) -> None:
        self.type = "robust"
        self.cost_multiplier = cost_multiplier
        self.robust_operator = robust_operator
        self.order = order

    def format(self, control_num, total_time_steps):
        """
        Will get shape of cost values and gradients.
        For this cost function, we store the values at each time step.
        We store gradients for each target state, control and time step.
        The reason is that we evolve each state seperately, so we get each cost value
        and sum over them after evolution. Please check the formula in the paper.
        Parameters
        ----------
        cost_multiplier:
            Weight factor of the cost function; expected < 1
        total_time_steps:
            Number of total time steps
        """
        self.total_time_steps = total_time_steps
        self.cost_format = (1)

    def commutator(self,A,B):
        return anp.matmul(A,B)-anp.matmul(B,A)

    def norm_frob(self,A):
        A = A - A[0][0]*np.identity(len(A))
        norm = anp.abs(A[1][1])**2
        for i in range(2):
            for j in range(i+1,len(A)):
                norm += anp.abs(A[i][j])**2
        return anp.sqrt(norm)

    def cost(self, states,deltat,n,cost_value,total_time,control_H=None,control=None) -> np.ndarray:
        """
        Compute the cost. The cost==the overlap of each evolved state and its target state.

        Parameters
        ----------
        forward_state:
            Evolving state in the forward evolution.
        mode:
            The way of getting gradients. "AD" or "AG"
        backward_state:
            Target states
        cost_value:
            Cost values that have shape self.cost_format
        time_step:
            Toltal number of time steps
        """
        states=anp.transpose(states)
        self.robust_tilda=[]
        cost1=0
        cost2=0
        cost3=0
        if n==0:
            self.A1=[]
            self.cost1 = np.zeros(len(self.robust_operator))
            for i in range(len(self.robust_operator)):
                self.A1.append(anp.zeros_like(self.robust_operator[0]))
            if self.order >= 2:
                self.A2=[]
                for i in range(len(self.robust_operator)):
                    self.A2.append([])
                    for j in range(len(self.robust_operator)):
                        self.A2[i].append(anp.zeros((len(self.robust_operator[0]),len(self.robust_operator[0]))))
            if self.order >=3:
                self.A3=0*states
            self.cost2 = np.zeros((len(self.robust_operator),len(self.robust_operator)))

        for i in range(len(self.robust_operator)):
            states_dag=conjugate_transpose_ad(states)
            self.robust_tilda.append(anp.matmul(states_dag,anp.matmul(self.robust_operator[i],states)))
            self.A1[i]=self.A1[i]+deltat*self.robust_tilda[i]/total_time


        if self.order >= 2:
            for i in range(len(self.robust_operator)):
                for j in range(len(self.robust_operator)):
                    self.A2[i][j]+=deltat/2*self.commutator(self.A1[i],self.robust_tilda[j])/total_time
        if self.order >=3:
            self.A3 +=deltat/2*self.commutator(self.A2[0][0],self.robust_tilda[0])/total_time
        if cost_value==True:
            total_cost = 0
            for i in range(len(self.robust_tilda)):
                print(self.norm_frob(self.A1[i]))
                cost1 += self.norm_frob(self.A1[i])

                # total_cost += (anp.linalg.norm(self.A1[i]-self.A1[i][0][0]*np.identity(len(self.A1[i])))/total_time)**2

            if self.order >= 2:

                for i in range(len(self.robust_tilda)):
                    for j in range(len(self.robust_tilda)):
                        print(self.norm_frob(self.A2[i][j]))
                        cost2 +=self.norm_frob(self.A2[i][j])
            if self.order >= 3:
                print((self.norm_frob(self.A3)))
                cost3 +=self.norm_frob(self.A3)
            cost = np.array([cost1, cost2, cost3])
            for i in range(len(cost)):
                total_cost += cost[i]* self.cost_multiplier[i]
            self.cost_value=total_cost
            return total_cost
        else:
            self.cost_value=0.
            return 0


