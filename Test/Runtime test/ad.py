import numpy as np
from scipy.sparse import kron,dia_matrix,identity,csc_matrix
from qoc_ag.models.close_system.optimization import grape_schroedinger_discrete
from qoc_ag.costs import TargetStateInfidelity
def harmonic(H_size):
    diagnol = np.arange(H_size)
    up_diagnol = np.sqrt(diagnol)
    low_diagnol = np.sqrt(np.arange(1, H_size + 1))
    a= dia_matrix(([ up_diagnol], [ 1]), shape=(H_size, H_size)).tocsc()
    a_dag=dia_matrix(([ low_diagnol], [ -1]), shape=(H_size, H_size)).tocsc()
    return a_dag,a
def get_control(N):
    sigmap, sigmam = harmonic(2)
    sigmap=sigmap
    sigmam=sigmam
    sigmax=sigmap+sigmam
    sigmay=-1j*sigmap+1j*sigmam
    control=[]
    if N==1:
        control.append(kron(sigmax, identity(2 ** (N - 1))))
        control.append(kron(sigmay, identity(2 ** (N - 1))))
        return control
    else:
        a=identity(2**(N-1))
        control.append(kron(sigmax,a,format="csc"))
        control.append(kron(sigmay, identity(2 ** (N - 1)),format="csc"))
        for i in range(1,N-1):
            control.append(kron(kron(identity(2**i),sigmax), identity(2 ** (N - 1-i)),format="csc"))
            control.append(kron(kron(identity(2 ** i), sigmay), identity(2 ** (N - 1 - i)),format="csc"))
        control.append(kron(identity(2**(N-1)),sigmax,format="csc"))
        control.append(kron(identity(2**(N-1)),sigmay,format="csc"))
    return control
def get_int(N):
    sigmap, sigmam = harmonic(2)
    sigmaz=sigmap.dot(sigmam)
    H0=0
    SIGMAZ=kron(sigmaz,sigmaz)
    H0=H0+kron(SIGMAZ,identity(2**(N-2)))+kron(identity(2**(N-2)),SIGMAZ)
    for i in range(1,N-2):
        H0=H0+kron(kron(identity(2**i),SIGMAZ),identity(2 ** (N - 2 - i)))
    return H0

def Had(d,n):
    omega=np.exp(2j*np.pi/d)
    Had = 1/np.sqrt(d) * np.array([[((omega) ** (i*j))
                                      for i in range(d)]
                                     for j in range(d)])
    Had_gat=Had
    for i in range(n-1):
        Had_gat=np.kron(Had_gat,Had)
    return Had_gat

def control_H(control,H_control):
    H=0
    for i in range(len(control)):
        H=H+control[i]*H_control[i]
    return H
def get_initial(N):
    state=[]
    for i in range(2**N):
        s=np.zeros((2 ** N))
        s[i]=1
        state.append(s)
    return np.array(state)

def simulation(q_number):
    H0=csc_matrix(get_int(q_number)).toarray()
    H_controls=get_control(q_number)
    for i,control in enumerate(H_controls):
        H_controls[i]=np.array(control.toarray())
    initial_states=get_initial(q_number)
    Target=Had(2,q_number)
    total_time=2*q_number
    total_time_steps=20*q_number
    costs = [TargetStateInfidelity(Target,cost_multiplier=1)]

    grape_schroedinger_discrete(total_time_steps,
                                    costs, total_time, H0, H_controls,
                                    initial_states,
                                    mode='AG', tol=1e-3)


simulation(7)
