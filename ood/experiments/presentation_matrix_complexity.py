import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

matrix_sizes = [10, 100, 500, 1000, 2000, 3000, 5000, 10000]
# matrix_sizes = [10, 100, 500, 1000, 2000]
times = []

markersize_x0 = 10
markersize_trajs = 0.4
fontsize_labels = 28
matplotlib.rc('xtick', labelsize=fontsize_labels)
matplotlib.rc('ytick', labelsize=fontsize_labels)
matplotlib.rc('text', usetex=True)
matplotlib.rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
plt.rc('legend',fontsize=fontsize_labels-2)

for size in matrix_sizes:
    # Repeat the inversion 5 times for each matrix size
    times_for_size = []
    for _ in range(5):
        # Generate a random matrix, make it symmetric and positive definite
        matrix = np.random.rand(size, size)
        matrix = matrix @ matrix.T  # Make the matrix symmetric and positive definite

        start_time = time.time()  # Start the timer

        # Compute the inverse using np.linalg.inv
        inverse = np.linalg.inv(matrix)

        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        times_for_size.append(elapsed_time)

    # Record the times for this matrix size
    times.append(times_for_size)

# Compute the mean and standard deviation for each matrix size
times_mean = [np.mean(t) for t in times]
times_std = [np.std(t) for t in times]

# Plotting the results
hdl_fig_ker, hdl_splots_illus = plt.subplots(1,1,figsize=(12,6),sharex=False)
hdl_splots_illus.plot(matrix_sizes, times_mean, 'o-', label='Mean')
hdl_splots_illus.fill_between(matrix_sizes, np.subtract(times_mean, times_std), np.add(times_mean, times_std), color='b', alpha=.1, label='Std dev')
hdl_splots_illus.set_xlabel(r'Matrix Size (N)',fontsize=fontsize_labels)
hdl_splots_illus.set_ylabel(r'Wall Clock Time (seconds)',fontsize=fontsize_labels)
hdl_splots_illus.set_title(r'Matrix Inversion Time',fontsize=fontsize_labels)
hdl_splots_illus.grid(True)
hdl_splots_illus.legend()
plt.show(block=False)


savefig = True
path2folder = "/Users/alonrot/work/meetings/presentation_2023_june/pics"
if savefig:
    path2save_fig = "{0:s}/compu_time_matrix_inversion.png".format(path2folder)
    print("Saving fig at {0:s} ...".format(path2save_fig))
    hdl_fig_ker.savefig(path2save_fig,bbox_inches='tight',dpi=300,transparent=True)
    print("Done saving fig!")
else:
    plt.show(block=True)
