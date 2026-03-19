import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from statsmodels.tsa.stattools import acf
from mpl_toolkits.mplot3d import Axes3D

from system import *
from events import *

#print(simulation.snapshots)
def drawAllGraphs(snapshots, lambdas):
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    times = [snap[0] for snap in snapshots]
    mean_durations = [snap[2] for snap in snapshots]
    queues_lengths = [snap[3] for snap in snapshots]
    mean_queue_lenghs = [snap[4] for snap in snapshots]
    mean_wait_times = [snap[5] for snap in snapshots]

    plt.figure()
    for thread_idx in range(NUMBER_OF_THREADS):
        mean_state_durations = [d[thread_idx] for d in mean_durations]
        plt.plot(times, mean_state_durations, marker='.', linestyle='-', label=f"Thread {thread_idx+1}", gid=f"Thread_{thread_idx+1}")
    plt.xlabel("Time")
    plt.ylabel("Mean State Duration")
    plt.title("Time vs Mean State Duration")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "time_mean_duration.svg"), dpi=600)
    plt.close()


    plt.figure(figsize=(30, 8))
    for thread_idx in range(NUMBER_OF_THREADS):
        thread_lengths = [q[thread_idx] for q in queues_lengths]

        plt.plot(times, thread_lengths, marker='.', linestyle='-', label=f"Thread {thread_idx+1}, Lambda = {lambdas[thread_idx]}", gid=f"Thread_{thread_idx+1}")
    plt.xlabel("Time")
    plt.ylabel("Queue Length")
    plt.title(f"Time vs Queue Length per Thread. SPEED = {CONNECTION_SPEED}pkg/s")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "time_queue_per_thread.svg"), dpi=600)
    plt.close()

    plt.figure(figsize=(30, 8))
    for thread_idx in range(NUMBER_OF_THREADS):
        mean_lengths = [mq[thread_idx] for mq in mean_queue_lenghs]
        
        plt.plot(times, mean_lengths, marker='.', linestyle='-', label=f"Thread {thread_idx+1}, Lambda = {lambdas[thread_idx]}", gid=f"Thread_{thread_idx+1}")
    plt.xlabel("Time")
    plt.ylabel("Mean Queue Length")
    plt.title(f"Time vs Mean Queue Length per Thread. SPEED = {CONNECTION_SPEED}pkg/s")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mean_queue_length_by_thread.svg"), dpi=600)
    plt.close()

    plt.figure(figsize=(30, 8))
    for thread_idx in range(NUMBER_OF_THREADS):
        mean_wait_times_by_thread = [mt[thread_idx] for mt in mean_wait_times]
        
        plt.plot(times, mean_wait_times_by_thread, marker='.', linestyle='-', label=f"Thread {thread_idx+1}, Lambda = {lambdas[thread_idx]}", gid=f"Thread_{thread_idx+1}")
    plt.xlabel("Time")
    plt.ylabel("Mean Wait Times")
    plt.title(f"Time vs Mean Wait Times per Thread. SPEED = {CONNECTION_SPEED}pkg/s")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mean_wait_times_by_thread.svg"), dpi=600)
    plt.close()

def isOverloaded(queues_lengths, lag, threshold):
    isOverloaded = []
    for thread_idx in range(NUMBER_OF_THREADS):
        thread_lengths = [q[thread_idx] for q in queues_lengths]
        acf_values = acf(thread_lengths, nlags=lag)
        isOverloaded.append(acf_values[-1]>threshold)
        #print(isOverloaded[-1])
    return any(isOverloaded)


def generateLambdaMesh(maxLambda, step):
    lambdas = []
    for l1 in np.arange(0+step, maxLambda, step):
        for l2 in np.arange(0+step, maxLambda, step):
            for l3 in np.arange(0+step, maxLambda, step):
                lambdas.append((l1,l2,l3))
    return lambdas

def run_simulation(lambdas):
    simulation = System()

    t = np.zeros(NUMBER_OF_THREADS)
    while True:
        array_tau = np.random.uniform(size=NUMBER_OF_THREADS)
        array_tau = -np.log(array_tau)/lambdas
        t += array_tau
        if np.all(t > T_END):
            break
        for i in range(NUMBER_OF_THREADS):
            if t[i]<= T_END:
                random_size = np.random.uniform()
                if random_size < PROBABILITY_OF_SMALL_GROUP:
                    simulation.schedule_event(IncomeEvent(t[i],SIZE_OF_SMALL_GROUP,i))
                else:
                    simulation.schedule_event(IncomeEvent(t[i],SIZE_OF_BIG_GROUP,i))


    simulation.run()
    #drawAllGraphs(simulation.snapshots, lambdas)

    queues_lengths = [snap[3] for snap in simulation.snapshots]

    tail = int(len(queues_lengths)*0.5)

    return (isOverloaded(queues_lengths[-tail:], 50, 0.7), lambdas)

def draw_lambdas_vs_overload():
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    with open('lambdas_vs_overload.pkl', 'rb') as f:
        results = pickle.load(f)
    is_overloaded = np.array([item[0] for item in results])
    coordinates = np.array([item[1] for item in results])

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]

    colors = np.where(is_overloaded, 'r', 'g')
    alpha = np.where(is_overloaded, 0.5, 1)
    sizes = np.where(is_overloaded, 2, 10)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x,y,z,c=colors, s=sizes, alpha=alpha)
    ax.set_xlabel('Lambda 1')
    ax.set_ylabel('Lambda 2')
    ax.set_zlabel('Lambda 3')
    ax.set_title("Lambdas vs overload")

    ax.view_init(elev=30, azim=60)
    plt.savefig(os.path.join(output_dir, "lambdas_vs_overload.svg"), dpi=600)

    # fig = plt.figure(figsize=(15, 15))
    # plt.scatter(x,y,c=colors, s=5, alpha=0.6)
    # plt.xlabel('Lambda 1')
    # plt.ylabel('Lambda 2')
    # plt.savefig(os.path.join(output_dir, "lambda1_2_vs_overload.svg"), dpi=600)

    # fig = plt.figure(figsize=(15, 15))
    # plt.scatter(x,z,c=colors, s=5, alpha=0.6)
    # plt.xlabel('Lambda 1')
    # plt.ylabel('Lambda 3')
    # plt.savefig(os.path.join(output_dir, "lambda1_3_vs_overload.svg"), dpi=600)

    # fig = plt.figure(figsize=(15, 15))
    # plt.scatter(y,z,c=colors, s=5, alpha=0.6)
    # plt.xlabel('Lambda 1')
    # plt.ylabel('Lambda 2')
    # plt.savefig(os.path.join(output_dir, "lambda1_2_vs_overload.svg"), dpi=600)

    x_min, y_min, z_min = np.min(coordinates,axis=0)
    eps = 1e-5
    fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize= (18,5))
    mask_z=np.abs(z-z_min)<eps
    ax1.scatter(x[mask_z],y[mask_z],c=colors[mask_z], s=10)
    ax1.set_title(f'Срез X-Y (Z={z_min:.2f})')
    ax1.set_xlabel('Lambda 1')
    ax1.set_ylabel('Lambda 2')

    mask_y=np.abs(y-y_min)<eps
    ax2.scatter(x[mask_y],z[mask_y],c=colors[mask_y], s=10)
    ax2.set_title(f'Срез X-Z (Y={y_min:.2f})')
    ax2.set_xlabel('Lambda 1')
    ax2.set_ylabel('Lambda 3')

    mask_x=np.abs(x-x_min)<eps
    ax3.scatter(y[mask_x],z[mask_x],c=colors[mask_x], s=10)
    ax3.set_title(f'Срез Y-Z (X={x_min:.2f})')
    ax3.set_xlabel('Lambda 2')
    ax3.set_ylabel('Lambda 3')


    plt.tight_layout()
    plt.savefig('progections.svg', dpi=600)
    plt.close()

if __name__=='__main__':

    # ARRAY_OF_LAMBDAS = generateLambdaMesh(4, 0.33)
    # results = []
    
    # with ProcessPoolExecutor(max_workers=10) as executor:
    #     for res in executor.map(run_simulation, ARRAY_OF_LAMBDAS):
    #         results.append(res)
    # with open('lambdas_vs_overload.pkl', 'wb') as f:
    #     pickle.dump(results, f)
    
    #tmp = run_simulation((10,10,10))
    #print(tmp[0])
    draw_lambdas_vs_overload()

    
    