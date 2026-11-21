import numpy as np
import heapq
import datasketches

from config import *
from events import *
from dataclasses import dataclass, field
from numpy.typing import NDArray

class RingBufferOverflowError(RuntimeError):
    pass

class SimulationError(RuntimeError):
    pass

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
        queue_lengths: NDArray[np.int64],
        total_wait_times: NDArray[np.float64],
        cumulative_served_clients: NDArray[np.int64]
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
            raise RingBufferOverflowError("RingBuffer overflow!")
       
        self.size += 1
    def get(self):
        self.start = (self.start + 1) % self.max_size
        self.size -= 1
        return self.data[self.start-1]
    def is_empty(self):
        return self.start == self.end
    def length(self):
        return self.size

SERVING = 0 
TUNING = 1

class System:
    def __init__(self, additions: tuple[float, ...], TMax: float, expected_capacity: int):
        self.additions: tuple[float, ...] = additions
        self.TMax: float = TMax

        self.events: list[Event] = []
        self.schedule_event(StateChangeEvent(additions[0]/CONNECTION_SPEED+1e-5, 0, (0,1))) #first state change scheduled!

        self.system_time: float = 0
        self.system_state: tuple[int, int] = (0,0)
        self.threads_queues: list[RingBuffer] = [RingBuffer(50000) for _ in range(NUMBER_OF_THREADS)]
        self.total_wait_times: NDArray[np.float64] = np.zeros(NUMBER_OF_THREADS)
        self.cumulative_served_clients: NDArray[np.int64] = np.zeros(NUMBER_OF_THREADS, dtype=np.int64)

        self.system_is_free = True
        self.redline = additions[0]/CONNECTION_SPEED+1e-5 + T_TUNE

        self.snapshots: Snapshots = Snapshots(expected_capacity, NUMBER_OF_THREADS)
        self.snapshots.push(
            times=self.system_time,
            states=self.system_state,
            queue_lengths= self.length_of_all_queues(),
            total_wait_times= self.total_wait_times,
            cumulative_served_clients= self.cumulative_served_clients
        )

        self.kll = datasketches.kll_doubles_sketch() # type: ignore[attr-defined]

    def schedule_event(self, event: Event) -> None:
        heapq.heappush(self.events, event)

    def run(self):
        while self.events and self.system_time < T_END:
            event: Event = heapq.heappop(self.events)

            if event.time < self.system_time:
                print("sys_time: ", self.system_time)
                print("event: ", event)
                raise SimulationError("Time tried to go backwards!")

            self.system_time = event.time

            if isinstance(event, IncomeEvent):
                self.handle_income_event(event)
            elif isinstance(event, EndOfServingEvent):
                self.handle_end_of_service_event()
            elif isinstance(event, StateChangeEvent):
                self.handle_change_state_event(event.next_state)
            else:
                print(event)
                raise SimulationError("Unknown event type!")

            if self.system_state[1] == SERVING and self.system_is_free and not(self.threads_queues[self.system_state[0]].is_empty()) and (self.system_time + T_SERVE < self.redline):
                self.handle_start_of_service()
        
        while self.events:
            event: Event = heapq.heappop(self.events)
            if not(isinstance(event, StateChangeEvent)):
                print(event)
                print(self.events)
                raise SimulationError("Simulation ended when there were unhadled events!")

    def handle_start_of_service(self):
        thread_number = self.system_state[0]

        time_of_arrival = self.threads_queues[thread_number].get()
        wait_time = self.system_time - time_of_arrival

        self.schedule_event(EndOfServingEvent(self.system_time + T_SERVE, thread_number))
        self.system_is_free = False
        self.total_wait_times[thread_number] += wait_time
        self.cumulative_served_clients[thread_number] += 1

        self.kll.update(wait_time)
        self.take_snapshot()

    def handle_income_event(self, event: IncomeEvent):
        thread_number = event.thread_number
        for _ in range(event.number_of_calls):
            self.threads_queues[thread_number].append(self.system_time)
        self.take_snapshot()

    def handle_end_of_service_event(self):
        self.system_is_free = True
        self.take_snapshot()

    def handle_change_state_event(self, next_state: tuple[int,int]):
        self.system_state = next_state

        thread_number = self.system_state[0]

        if self.system_state[1] == TUNING:
            new_thread_number = (thread_number+1)%NUMBER_OF_THREADS
            self.schedule_event(StateChangeEvent(self.system_time+T_TUNE, new_thread_number, (new_thread_number, 0)))

            try:
                state_duration = min((self.threads_queues[thread_number].length()+self.additions[thread_number])/CONNECTION_SPEED, self.TMax)
            except TypeError:
                print((self.threads_queues[thread_number].length()+self.additions[thread_number])/CONNECTION_SPEED, self.TMax)
                raise
            except IndexError:
                print(thread_number)
                print(self.additions[thread_number])
                print(self.threads_queues[thread_number])
                raise

            self.redline = min(self.system_time + T_TUNE + state_duration + T_TUNE, T_END)
            self.schedule_event(StateChangeEvent(self.system_time+T_TUNE+state_duration, new_thread_number, (new_thread_number, 1)))
        else:
            if not(self.system_is_free):
                print("sys_time: ", self.system_time, "sys_state: ", self.system_state)
                raise SimulationError("When switched to serving new thread there was a client being served!")
        self.take_snapshot()
        
        

    def length_of_all_queues(self) -> NDArray[np.int64]:
        length: NDArray[np.int64] = np.zeros(NUMBER_OF_THREADS, dtype=np.int64)
        for n in range(NUMBER_OF_THREADS):
            length[n] = (self.threads_queues[n].length())
        return length

    def register_start_of_serving(self, thread_number:int, wait_time: float) -> None:
        self.total_wait_times[thread_number] += wait_time
        self.cumulative_served_clients[thread_number] += 1

        self.kll.update(wait_time)

    
    def take_snapshot(self):
        self.snapshots.push(
            times= self.system_time,
            states= self.system_state,
            queue_lengths= self.length_of_all_queues(),
            total_wait_times= self.total_wait_times, 
            cumulative_served_clients= self.cumulative_served_clients
        )

    def get_data(self) -> Snapshots:
        if self.events:
            print(self.events)
            raise SimulationError("Simulation ended when there were unhadled events!")
        return self.snapshots