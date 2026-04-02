import numpy as np
import heapq

from config import *
from events import StateChangeEvent

class RingBuffer:
    def __init__(self, size):
        self.end = 0
        self.start = 0
        self.max_size = size
        self.size = 0
        self.data = np.zeros(shape=size)
    def append(self, obj):
        self.data[self.end] = obj
        self.end = (self.end + 1) % self.max_size
        if self.start==self.end:
            raise RuntimeError("RingBuffer overflow!")
       
        self.size += 1
    def get(self):
        self.start = (self.start + 1) % self.max_size
        self.size -= 1
        return self.data[self.start-1]
    def is_empty(self):
        return self.start == self.end
    def length(self):
        return self.size

class System:
    def __init__(self, additions, TMax):
        self.events = []
        self.schedule_event(StateChangeEvent(additions[0]/CONNECTION_SPEED+1e-5, (0,1))) #first state change scheduled!

        self.threads_queues = [RingBuffer(20000) for n in range(NUMBER_OF_THREADS)]
        self.system_time = 0
        self.system_state = (0,0)
        self.system_is_free = True
        self.redline = 0

        self.snapshots = []
        mean_times_by_thread = [0]*NUMBER_OF_THREADS
        mean_times_by_thread[0] = additions[0]/CONNECTION_SPEED
        mean_lengths_by_thread = [0]*NUMBER_OF_THREADS
        self.snapshots.append((0, (0,0), mean_times_by_thread, [0]*NUMBER_OF_THREADS, mean_lengths_by_thread, [0]*NUMBER_OF_THREADS))
        self.last_mean_state_duration = [0]*NUMBER_OF_THREADS
        self.number_of_measurements_times = [0]*NUMBER_OF_THREADS
        self.number_of_measurements_lengths = [0]*NUMBER_OF_THREADS
        self.last_mean_state_duration[0] = additions[0]/CONNECTION_SPEED
        self.number_of_measurements_times[0] = 1
        self.number_of_measurements_lengths[0] = 1

        self.last_areas = [0]*NUMBER_OF_THREADS

        self.last_mean_wait_time = [0]*NUMBER_OF_THREADS

        self.additions = additions
        self.TMax = TMax
        self.number_of_served_calls = [0]*NUMBER_OF_THREADS

    def schedule_event(self, event):
        heapq.heappush(self.events, event)

    def run(self):
        while self.system_time < T_END:
            event = heapq.heappop(self.events)
            event.handle(self)

    def length_of_all_queues(self):
        length = []
        for n in range(NUMBER_OF_THREADS):
            length.append(self.threads_queues[n].length())
        return length
    
    def take_snapshot(self, update_mean_wait_time = False, wait_time=0, state_duration=0, update_mean_state_duration=False):
        if update_mean_state_duration and update_mean_wait_time:
            thread_number = self.system_state[0]
            mean_state_duration = (self.number_of_measurements_times[thread_number] * self.last_mean_state_duration[thread_number] + state_duration)/(self.number_of_measurements_times[thread_number]+1)
            self.number_of_measurements_times[thread_number] += 1
            self.last_mean_state_duration[thread_number] = mean_state_duration
            dt = self.system_time - self.snapshots[-1][0]
            self.last_areas = [area + length*dt for area, length in zip(self.last_areas, self.length_of_all_queues())]
            mean_wait_time = (self.number_of_measurements_lengths[thread_number]* self.last_mean_wait_time[thread_number]+wait_time)/(self.number_of_measurements_lengths[thread_number]+1)
            self.number_of_measurements_lengths[thread_number]+=1
            self.last_mean_wait_time[thread_number] = mean_wait_time
            self.snapshots.append(
                (self.system_time,
                self.system_state,
                self.last_mean_state_duration.copy(),
                self.length_of_all_queues(),
                [area/self.system_time for area in self.last_areas],
                self.last_mean_wait_time.copy()
                )
                )
        elif not(update_mean_state_duration) and update_mean_wait_time:
            thread_number = self.system_state[0]
            dt = self.system_time - self.snapshots[-1][0]
            self.last_areas = [area + length*dt for area, length in zip(self.last_areas, self.length_of_all_queues())]
            mean_wait_time = (self.number_of_measurements_lengths[thread_number]* self.last_mean_wait_time[thread_number]+wait_time)/(self.number_of_measurements_lengths[thread_number]+1)
            self.number_of_measurements_lengths[thread_number]+=1
            self.last_mean_wait_time[thread_number] = mean_wait_time
            self.snapshots.append(
                (self.system_time,
                self.system_state,
                self.last_mean_state_duration.copy(),
                self.length_of_all_queues(),
                [area/self.system_time for area in self.last_areas],
                self.last_mean_wait_time.copy()
                )
                )
        elif update_mean_state_duration and not(update_mean_wait_time):
            thread_number = self.system_state[0]
            mean_state_duration = (self.number_of_measurements_times[thread_number] * self.last_mean_state_duration[thread_number] + state_duration)/(self.number_of_measurements_times[thread_number]+1)
            self.number_of_measurements_times[thread_number] += 1
            self.last_mean_state_duration[thread_number] = mean_state_duration
            dt = self.system_time - self.snapshots[-1][0]
            self.last_areas = [area + length*dt for area, length in zip(self.last_areas, self.length_of_all_queues())]
            self.snapshots.append(
                (self.system_time,
                self.system_state,
                self.last_mean_state_duration.copy(),
                self.length_of_all_queues(),
                [area/self.system_time for area in self.last_areas],
                self.last_mean_wait_time.copy()
                )
                )
        else:
            dt = self.system_time - self.snapshots[-1][0]
            self.last_areas = [area + length*dt for area, length in zip(self.last_areas, self.length_of_all_queues())]
            self.snapshots.append(
                (self.system_time,
                self.system_state,
                self.last_mean_state_duration.copy(),
                self.length_of_all_queues(),
                [area/self.system_time for area in self.last_areas],
                self.last_mean_wait_time.copy()
                )
                )