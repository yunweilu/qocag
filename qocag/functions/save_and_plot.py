import os
import matplotlib.pyplot as plt
import matplotlib.animation
from numpy.fft import rfft, rfftfreq
import numpy as np
from matplotlib.pyplot import figure

def print_grads(iteration,cost_value,grads,cost_set):
    grads_norm = np.linalg.norm(grads)
    output="{:^6d} | {:^1.8e} |".format(iteration, cost_value,)
    cost_len=len(cost_set)
    for i in range(cost_len):
        output+="  {:^1.8e}  |".format(cost_set[i])
    output += "  {:^1.8e}  ".format(grads_norm)
    print(output)

def print_heading(cost_len):
    output="iter   |   total error  |"
    dash="========================="
    for i in range(cost_len):
        output+=("       cost"+str(i)+"      |")
        dash+="================="
    output+=("   grads_l2  ")
    dash += "======================="
    print(output)
    print(dash)


def generate_save_file_path(save_file_name, save_path):
    """
    Create the full path to a h5 file using the base name
    save_file_name in the path save_path. File name conflicts are avoided
    by appending a numeric prefix to the file name. This method assumes
    that all objects in save_path that contain _{save_file_name}.h5
    are created with this convention. The save path will be created
    if it does not already exist.

    Args:
    save_file_name :: str - the prefix of the
    save_path :: str -

    Returns:
    save_file_path :: str - the full path to the save file
    """
    # Ensure the path exists.
    os.makedirs(save_path, exist_ok=True)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for file_name in os.listdir(save_path):
        if ("_{}.npy".format(save_file_name)) in file_name:
            max_numeric_prefix = max(int(file_name.split("_")[0]),
                                     max_numeric_prefix)
    # ENDFOR
    save_file_name_augmented = ("{:05d}_{}.npy"
                                "".format(max_numeric_prefix + 1,
                                          save_file_name))

    return os.path.join(save_path, save_file_name_augmented)

def control_ani(result):
    figure(figsize=(10,10), dpi=80)
    fig, (ax_fft, ax_con) = plt.subplots(2, 1, )
    plt.rcParams['figure.figsize'] = [10, 10]
    controls = result["control_iter"]
    times = result["times"]
    if len(controls)<=100:
        frame_num=10
    else:
        frame_num=100
    def animate(l):
        control_num = len(controls[0])
        gap=np.ceil(len(controls)/frame_num)
        ax_con.clear()
        ax_fft.clear()
        for j in range(control_num):
            i=gap*(l)
            if i>len(controls):
                i=-1
            i=np.int(i)
            ax_con.plot(times, controls[i][j], label=str(j))
            dt = (times[1] - times[0])
            fourier = np.abs(rfft(controls[i][j]))
            freq = rfftfreq(len(times), dt)
            max_index = np.argpartition(abs(fourier), -3)[-3:]
            ax_fft.stem(freq, fourier, markerfmt=' ', label='')
            for index in max_index:
                ax_fft.annotate('{}'.format(freq[index]), xy=(freq[index], fourier[index]))
        ax_con.legend(bbox_to_anchor=(0.2, 2.3))
        ax_con.set_xlabel('time')
        ax_con.set_ylabel('control amplitude')
        ax_fft.set_ylabel("FFT Amplitude")
        ax_fft.set_xlabel("frequency")

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=frame_num,interval=10)
    return ani
