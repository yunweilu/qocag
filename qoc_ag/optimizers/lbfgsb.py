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

        ######################### plot the current status of optimization #########################
        from IPython.display import clear_output, display
        import matplotlib.pyplot as plt
        import qutip as qt

        # prepare parameters
        hamiltonian = qt.Qobj(kwargs["hamiltonian"])
        H_controls = kwargs["H_controls"]
        time_step_interval = kwargs["time_step_interval"]
        dims = kwargs["dims"]
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
        fig = plt.figure(figsize=(20, 8))
        ax0 = fig.add_subplot(2, 1, 1)
        ax1 = fig.add_subplot(2, 2, 3)
        ax2 = fig.add_subplot(2, 2, 4)

        # record
        error_record = []
        special_step = []
        special_fid_record = []
        
        def plot_status(x):
            clear_output(wait=True)

            current_error = function_(x, args[0])
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

            if (len(error_record) - 1) % 10 == 0:
                # calculate state transfer (ax1 and ax2)
                special_step.append(len(error_record) - 1)
                ax2.clear()

                qt_operators = [qt.Qobj(op) for op in H_controls]

                unitary = 1
                for t_i in range(pulse.shape[1]):
                    hamil = hamiltonian.copy()
                    for i in range(pulse_num):
                        hamil = hamil + pulse[i, t_i] * qt_operators[i]

                    unitary = (-1j * time_step_interval * hamil).expm() * unitary

                result_unitary = qt.Qobj(unitary, dims=[dims, dims])

                target_results = []
                target_fid = []

                for i in range(len(qt_init)):
                    target_results.append(result_unitary * qt_init[i])
                    target_fid.append(np.abs((target_results[-1].dag() * qt_target[i]).data[0, 0]))

                # append fidelity evolution
                special_fid_record.append(target_fid)
                fid_record_for_plot = np.array(special_fid_record)

                # plot states in n basis
                result_n_basis = []
                for j in range(subsys_num):
                    for i in range(target_num):
                        idx = j * target_num + i
                        plot_x = np.arange(len(target_n_basis[idx])) + np.sum(dims[:j])
                        color = cmap(i)
                        if subsys_num == 1:
                            np_result = target_results[i].full().reshape(-1)
                            result_n_basis.append(np_result.conj() * np_result)
                        else:
                            result_n_basis.append(np.diag(qt.ptrace(target_results[i], j).full()))

                        ax2.plot(plot_x, target_n_basis[idx].real, ls='--', color=color)
                        if j == 0:
                            ax2.scatter(plot_x, result_n_basis[idx].real, color=color, label=f"target state {i}")
                        else:
                            ax2.scatter(plot_x, result_n_basis[idx].real, color=color)
                ax2.legend()

            fid_record_for_plot = np.array(special_fid_record)
            for i in range(target_num):
                ax1.plot(special_step, fid_record_for_plot[:, i], label=f"Fid {i}")
            ax1.legend()

            plt.tight_layout()
            display(plt.gcf())


        def print_error(x):
            print(function_(x, args[0]))

        minimized_pulse = minimize(function_, initial_params, args=args,
                        method="L-BFGS-B", jac=jacobian_,
                        options=options, callback=plot_status, tol=tol).x
        plt.close()
        return minimized_pulse
