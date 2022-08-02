"""
lbfgsb.py - a module to expose the L-BFGS-B optimization algorithm
"""

from scipy.optimize import minimize
import numpy as np
class LBFGSB(object):
    """
    The L-BFGS-B optimizer.

    Fields: none
    """

    def __init__(self):
        """
        See class docstring for argument information.
        """
        super().__init__()


    def run(self, function, iteration_count, 
            initial_params, jacobian, args=(), **kwargs):
        """
        Run the L-BFGS-B method.

        Args:
        args :: any - a tuple of arguments to pass to the function
            and jacobian
        function :: any -> float
            - the function to minimize
        iteration_count :: int - how many iterations to perform
        initial_params :: numpy.ndarray - the initial optimization values
        jacobian :: any -> numpy.ndarray - a function that returns the jacobian
            of `function`

        Returns:
        result :: scipy.optimize.OptimizeResult
        """
        # Disregard termination conditions.
        function_ = lambda *args, **kwargs: function(*args, **kwargs)[0]
        jacobian_ = lambda *args, **kwargs: np.array(jacobian(*args, **kwargs)[0])
        # NOTE: min_error termination not implemented
        options = {
            "maxiter": iteration_count,
        }
        
# ##############################################################################
# ################## plot the current status of optimization ###################
        from IPython.display import clear_output, display
        import matplotlib.pyplot as plt
        import qutip as qt

        # plot setting
        detail_update_period = 100
        state_evolution_samples = 4

        # prepare parameters
        hamiltonian = qt.Qobj(kwargs["hamiltonian"])
        H_controls = kwargs["H_controls"]
        time_step_interval = kwargs["time_step_interval"]
        try:
            dims = kwargs["dims"]
        except KeyError:
            dims = hamiltonian.dims[0]

        target_states = kwargs["target_states"]
        init_states = kwargs["init_states"]
        try: 
            tol = kwargs["target_tol"]
        except KeyError:
            tol = 1e-10

        pulse_num = len(H_controls)
        target_num = init_states.shape[0]
        subsys_num = len(dims)

        cmap = plt.cm.get_cmap("hsv", target_num + 1)

        # prepare target
        target_n_basis = []
        qt_init = [qt.Qobj(init, dims=[dims, list(np.ones_like(dims))]) for init in init_states]
        qt_target = [qt.Qobj(tgt, dims=[dims, list(np.ones_like(dims))]) for tgt in target_states]
        for j in range(subsys_num):
            for i in range(target_num):
                if subsys_num == 1:
                    np_target = qt_target[i].full().reshape(-1)
                    target_n_basis.append(np_target.conj() * np_target)
                else:
                    target_n_basis.append(np.diag(qt.ptrace(qt_target[i], j).full()))

        # figure
        fig = plt.figure(figsize=(10, 4))
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 2, 3)
        ax2 = fig.add_subplot(2, 2, 4)

        # record
        error_record = []
        special_step = []
        special_fid_record = []
        
        def plot_status(x):
            clear_output(wait=True)

            current_error = function_(x, *args)
            error_record.append(current_error)

            pulse = x.reshape(pulse_num, -1)

            # plot pulse
            ax0.clear()
            for i in range(pulse_num):
                ax0.plot(pulse[i], label=f"pulse {i}")
            ax0.set_title(f"Current error: {current_error:.14f}")
            ax0.legend()

            # plot error
            plot_round_length = 5
            while plot_round_length < len(error_record):
                plot_round_length = int(plot_round_length * 1.5)
            ax1.clear()
            ax1.plot(range(len(error_record)), error_record, label="Error")
            ax1.set_xlim(-0.02 * plot_round_length, 1.02 * plot_round_length)
            ax1.set_ylim(-0.02, 1.02)

            # add special fid plot to ax1 and final states in ax2
            if (len(error_record) - 1) % detail_update_period == 0:
                # calculate state transfer (ax1 and ax2)
                special_step.append(len(error_record) - 1)
                ax2.clear()

                qt_operators = [qt.Qobj(op) for op in H_controls]

                current_states = qt.Qobj(init_states.copy().transpose()) # Now they are column vectors
                sample_time_steps = np.round(np.linspace(0, pulse.shape[1] - 1, state_evolution_samples + 1)).astype(int)
                remain_sample_steps = sample_time_steps[1:].copy()
                sampled_states = []
                for t_i in range(pulse.shape[1]):
                    hamil = hamiltonian.copy()
                    for i in range(pulse_num):
                        hamil = hamil + pulse[i, t_i] * qt_operators[i]

                    current_states = (-1j * time_step_interval * hamil).expm() * current_states
                
                    if t_i == remain_sample_steps[0]:
                        sampled_states.append(current_states.full().copy())
                        remain_sample_steps = np.delete(remain_sample_steps, [0])

                target_fid = [np.abs((qt_target[i].dag() * sampled_states[-1][:, i:i+1])[0, 0]) for i in range(len(qt_init))]

                # append fidelity evolution
                special_fid_record.append(target_fid)
                fid_record_for_plot = np.array(special_fid_record)

                # plot states in n basis
                for sample_i in range(state_evolution_samples):
                    result_n_basis = []
                    
                    if sample_i != state_evolution_samples-1:
                        scatter_alpha = 0.3
                        scatter_size = 5
                    else:
                        scatter_alpha = 0.3
                        scatter_size = 50

                    for j in range(subsys_num):

                        for i in range(target_num):

                            idx = j * target_num + i
                            plot_x = np.arange(len(target_n_basis[idx])) + np.sum(dims[:j])
                            color = cmap(i)

                            if subsys_num == 1:
                                np_result = sampled_states[sample_i][:, i]
                                result_n_basis.append(np_result.conj() * np_result)
                            else:
                                qt_target_result = qt.Qobj(sampled_states[sample_i][:, i], dims=[dims, np.ones_like(dims)])
                                result_n_basis.append(np.diag(qt.ptrace(qt_target_result, j).full()))

                            if sample_i == state_evolution_samples-1:
                                ax2.plot(plot_x, target_n_basis[idx].real, ls='--', color=color, alpha=0.7)

                            if j == 0 and sample_i == state_evolution_samples-1:
                                scatter_label = f"target state {i}"
                            else:
                                scatter_label = None
 
                            ax2.scatter(plot_x, result_n_basis[idx].real, color=color, label=scatter_label, alpha=scatter_alpha, s=scatter_size)
                ax2.legend()

            fid_record_for_plot = np.array(special_fid_record)
            for i in range(target_num):
                ax1.plot(special_step, fid_record_for_plot[:, i], label=f"Fid {i}")
            ax1.legend()

            plt.tight_layout()
            display(plt.gcf())


        def print_error(x):
            print(function_(x, *args))
        
        

# ##############################################################################
        minimized_pulse = minimize(function_, initial_params, args=args,
                        method="L-BFGS-B", jac=jacobian_,
                        options=options, callback=plot_status, tol=tol).x

        plt.close()


        return minimized_pulse
