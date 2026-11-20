import numpy as np
import heapq
import datasketches

from config import *
from events import StateChangeEvent, Event
from dataclasses import dataclass, field
from numpy.typing import NDArray

@dataclass
class Snapshots:
    capacity:int
    number_of_threads: int

    index:int = field(default=0, init=False)

    times: NDArray[np.float64] = field(init=False)
    states: NDArray[np.int64] = field(init=False)
    queue_lengths: NDArray[np.int64] = field(init=False)
    total_wait_times: NDArray[np.float64] = field(init=False)
    cumulative_served_clients: NDArray[np.int64] = field(init=False)

    def __post_init__(self):
        self.times = np.empty(self.capacity, dtype=np.float64)
        self.states = np.empty((self.capacity, 2), dtype=np.int64)

        self.queue_lengths = np.empty(
            (self.capacity, self.number_of_threads), dtype=np.int64
        )
        self.total_wait_times = np.empty(
            (self.capacity, self.number_of_threads), dtype=np.float64
        )
        self.cumulative_served_clients = np.empty(
            (self.capacity, self.number_of_threads), dtype=np.int64
        )
        

    def _grow(self) -> None:
        new_capacity = self.capacity * 2

        self.times = self._resize(self.times, new_capacity)
        self.states = self._resize(self.states, new_capacity)

        self.queue_lengths = self._resize(
            self.queue_lengths, new_capacity
        )
        self.total_wait_times = self._resize(
            self.total_wait_times, new_capacity
        )
        self.cumulative_served_clients = self._resize(
            self.cumulative_served_clients, new_capacity
        )

        self.capacity = new_capacity


    def _resize(self, array, new_capacity) -> NDArray:
        new_shape = (new_capacity, *array.shape[1:])
        new_array = np.empty(new_shape, dtype=array.dtype)
        new_array[:self.index] = array[:self.index]
        return new_array

    def push(
        self,
        times: float,
        states: tuple[int, int],
        queue_lengths: list[int],
        total_wait_times: list[float],
        cumulative_served_clients: list[float]
    ) -> None:
        if self.index >= self.capacity:
            self._grow()

        i = self.index

        self.times[i] = times
        self.states[i] = states
        self.queue_lengths[i] = queue_lengths
        self.total_wait_times[i] = total_wait_times
        self.cumulative_served_clients[i] = cumulative_served_clients

        self.index += 1

    @property
    def size(self) -> int:
        return self.index


class RingBuffer:
    def __init__(self, max_size, actual_size=0):
        
        self.start = 0
        self.max_size = max_size
        self.size = actual_size
        self.data = np.zeros(shape=max_size)
        self.end = (actual_size ) % self.max_size
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
    def __init__(self, additions: tuple[float, ...], TMax: float, expected_capacity: int):
        self.additions: tuple[float, ...] = additions
        self.TMax: float = TMax

        self.events: list[Event] = []
        self.schedule_event(StateChangeEvent(additions[0]/CONNECTION_SPEED+1e-5, (0,1))) #first state change scheduled!

        self.threads_queues = [RingBuffer(50000) for _ in range(NUMBER_OF_THREADS)]
        self.system_time = 0
        self.system_state = (0,0)
        self.system_is_free = True
        self.redline = 0

        self.snapshots: Snapshots = Snapshots(expected_capacity, NUMBER_OF_THREADS)
        self.snapshots.push(
            times=0,
            states=(0,0),
            queue_lengths= [0]*NUMBER_OF_THREADS,
            total_wait_times= [0]*NUMBER_OF_THREADS,
            cumulative_served_clients= [0]*NUMBER_OF_THREADS
        )

        self.kll = datasketches.kll_doubles_sketch() # type: ignore[attr-defined]

    def schedule_event(self, event: Event) -> None:
        heapq.heappush(self.events, event)

    def run(self):
        while self.system_time < T_END:
            event = heapq.heappop(self.events)
            event.handle(self)

    def length_of_all_queues(self) -> list[int]:
        length = []
        for n in range(NUMBER_OF_THREADS):
            length.append(self.threads_queues[n].length())
        return length

    def register_wait_time(self, wait_time:float=0.0) -> None:
        self.kll.update(wait_time)
    
    def take_snapshot(self, wait_time=0):
        # if update_mean_state_duration and update_mean_wait_time:
            
            #mean_state_duration = (self.number_of_measurements_times[thread_number] * self.snapshots.mean_state_duration[-1][thread_number] + state_duration)/(self.number_of_measurements_times[thread_number]+1)
            #self.last_mean_state_duration[thread_number] = mean_state_duration
            # dt = self.system_time - self.snapshots.times[-1]
            # self.last_areas = [area + length*dt for area, length in zip(self.last_areas, self.length_of_all_queues())]
            # mean_wait_time = (self.number_of_measurements_lengths[thread_number]* self.last_mean_wait_time[thread_number]+wait_time)/(self.number_of_measurements_lengths[thread_number]+1)
            # self.number_of_measurements_lengths[thread_number]+=1
            # self.last_mean_wait_time[thread_number] = mean_wait_time

            # self.snapshots.append(
            #     (self.system_time,
            #     self.system_state,
            #     self.last_mean_state_duration.copy(),
            #     self.length_of_all_queues(),
            #     [area/self.system_time for area in self.last_areas],
            #     self.last_mean_wait_time.copy()
            #     )
            # )
            thread_number = self.system_state[0]
            previous = self.snapshots.size-1
            
            total_wait_times = self.snapshots.total_wait_times[previous].copy()
            total_wait_times[thread_number] += wait_time

            cumulative_served_clients = self.snapshots.cumulative_served_clients[previous].copy()
            cumulative_served_clients[thread_number] += 1 

            self.snapshots.push(
                            times= self.system_time,
                            states= self.system_state,
                            queue_lengths= self.length_of_all_queues(),
                            total_wait_times= total_wait_times, 
                            cumulative_served_clients= cumulative_served_clients
                        )

            

        # elif not(update_mean_state_duration) and update_mean_wait_time:
        #     thread_number = self.system_state[0]
        #     dt = self.system_time - self.snapshots[-1][0]
        #     self.last_areas = [area + length*dt for area, length in zip(self.last_areas, self.length_of_all_queues())]
        #     mean_wait_time = (self.number_of_measurements_lengths[thread_number]* self.last_mean_wait_time[thread_number]+wait_time)/(self.number_of_measurements_lengths[thread_number]+1)
        #     self.number_of_measurements_lengths[thread_number]+=1
        #     self.last_mean_wait_time[thread_number] = mean_wait_time
        #     self.snapshots.append(
        #         (self.system_time,
        #         self.system_state,
        #         self.last_mean_state_duration.copy(),
        #         self.length_of_all_queues(),
        #         [area/self.system_time for area in self.last_areas],
        #         self.last_mean_wait_time.copy()
        #         )
        #         )
            
        #     self.kll.update(wait_time)
           
                
        # elif update_mean_state_duration and not(update_mean_wait_time):
        #     thread_number = self.system_state[0]
        #     mean_state_duration = (self.number_of_measurements_times[thread_number] * self.last_mean_state_duration[thread_number] + state_duration)/(self.number_of_measurements_times[thread_number]+1)
        #     self.number_of_measurements_times[thread_number] += 1
        #     self.last_mean_state_duration[thread_number] = mean_state_duration
        #     dt = self.system_time - self.snapshots[-1][0]
        #     self.last_areas = [area + length*dt for area, length in zip(self.last_areas, self.length_of_all_queues())]
        #     self.snapshots.append(
        #         (self.system_time,
        #         self.system_state,
        #         self.last_mean_state_duration.copy(),
        #         self.length_of_all_queues(),
        #         [area/self.system_time for area in self.last_areas],
        #         self.last_mean_wait_time.copy()
        #         )
        #         )
        # else:
        #     dt = self.system_time - self.snapshots[-1][0]
        #     self.last_areas = [area + length*dt for area, length in zip(self.last_areas, self.length_of_all_queues())]
        #     self.snapshots.append(
        #         (self.system_time,
        #         self.system_state,
        #         self.last_mean_state_duration.copy(),
        #         self.length_of_all_queues(),
        #         [area/self.system_time for area in self.last_areas],
        #         self.last_mean_wait_time.copy()
        #         )
        #         )

    def get_data(self) -> Snapshots:
        return self.snapshots