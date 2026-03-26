import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
#import plotly.graph_objects as go
import pandas as pd

from functools import partial
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

def generateEpsilonMesh(Eps_max, step):
    epsilons = []
    for e1 in np.arange(0, Eps_max, step):
        for e2 in np.arange(0, Eps_max, step):
            for e3 in np.arange(0, Eps_max, step):
                epsilons.append((e1,e2,e3))
    return epsilons

def run_simulation_for_stationary_mode(lambdas, epsilons):
    simulation = System(epsilons)

    t = np.zeros(NUMBER_OF_THREADS)
    while True: #Income generation
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

def run_simulation_for_epsilon_optimisation(lambdas, epsilons):
    simulation = System(epsilons)

    t = np.zeros(NUMBER_OF_THREADS)
    while True: #Income generation
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

    mean_wait_times_by_threads_last = simulation.snapshots[-1][5]

    mean_wait_time = sum(mean_wait_times_by_threads_last)/len(mean_wait_times_by_threads_last)

    return (mean_wait_time, mean_wait_times_by_threads_last, epsilons)

def draw_lambdas_vs_overload():
    with open('lambdas_vs_overload.pkl', 'rb') as f:
        results = pickle.load(f)

    df = pd.DataFrame({
        'x': [item[1][0] for item in results],
        'y': [item[1][1] for item in results],
        'z': [item[1][2] for item in results],
        'isOverloaded': [item[0] for  item in results]
    })

    df['color_rgba'] = np.where(df['isOverloaded'], 'rgba(255,0,0,0)', 'rgba(0,128,0,1.0)')
    df['text'] = np.where(df['isOverloaded'], 'Overloaded', 'Not overloaded')
    

    fig = go.Figure(data=[go.Scatter3d(
        x=df['x'],
        y=df['y'],
        z=df['z'],
        mode='markers',
        marker=dict(
            size=3,
            color=df['color_rgba'],
            # opacity=df['opacity'],
            line=dict(width=0)
        ),
        text=df['text'],
        hoverinfo='text'
    )])

    fig.update_layout(
        title='Lambdas vs Overload',
        scene=dict(
            xaxis_title='Lambda 1',
            yaxis_title='Lambda 2',
            zaxis_title='Lambda 3',
        ),
        margin=dict(l=0, r=0, b=0,  t=40)
    )

    html_filename = 'interactive_3_lambdas_vs_overload.html'
    fig.write_html(html_filename)

def anylize_epsilon_optimization(filename):
    with open(filename, 'rb') as f:
        results = pickle.load(f)

    general_mean_wait_times = [r[0] for r in results]
    min_general_wait_time = min(general_mean_wait_times)
    index_of_min_general_wait_time = general_mean_wait_times.index(min_general_wait_time)

    print(f"All threads general optimal:\nGeneral Mean wait time: {results[index_of_min_general_wait_time][0]},\nMean wait times by threads:{results[index_of_min_general_wait_time][1]},\nEpsilons:{results[index_of_min_general_wait_time][2]}\n")

    mean_wait_times_by_threads = [r[1] for r in results]
    for thread_idx in range(NUMBER_OF_THREADS):
        mean_wait_times = [mt[thread_idx] for mt in mean_wait_times_by_threads]

        min_wait_time = min(mean_wait_times)
        index_of_min_wait_time = mean_wait_times.index(min_wait_time)

        print(f"Thread {thread_idx} marginal optimal:\nGeneral mean wait time:{results[index_of_min_wait_time][0]}\nMin wait time by threads: {results[index_of_min_wait_time][1]},\nEpsilons:{results[index_of_min_wait_time][2]}\n")


if __name__=='__main__':

    # ARRAY_OF_LAMBDAS = generateLambdaMesh(4, 0.1)
    # results = []
    
    # with ProcessPoolExecutor(max_workers=15) as executor:
    #     for res in executor.map(run_simulation, ARRAY_OF_LAMBDAS):
    #         results.append(res)
    # with open('lambdas_vs_overload.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    # ARRAY_OF_EPSILONS = generateEpsilonMesh(T_MAX*10, 1)
    # results = []

    # run_with_configured_lambdas = partial(run_simulation_for_epsilon_optimisation, (3,5,1))
    
    # with ProcessPoolExecutor(max_workers=12) as executor:
    #     for res in executor.map(run_with_configured_lambdas, ARRAY_OF_EPSILONS):
    #         results.append(res)
    # with open('epsilons_vs_mean_times.pkl', 'wb') as f:
    #     pickle.dump(results, f)

    anylize_epsilon_optimization('epsilons_vs_mean_times_small_lambdas.pkl')
    print('-'*50)
    anylize_epsilon_optimization('epsilons_vs_mean_times_HighSpeed_mid_lambdas.pkl')
    
    #tmp = run_simulation((10,10,10))
    #print(tmp[0])
    #draw_lambdas_vs_overload()

    
    