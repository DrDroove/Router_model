from abc import ABC, abstractmethod

from config import *

class Event(ABC):
    def __init__(self, time):
        self.time = time
    def __lt__(self, other):
        return self.time < other.time
    @abstractmethod
    def handle(self, simulation):
        pass

class StateChangeEvent(Event):
    def __init__(self, time, next_state):
        super().__init__(time)
        self.next_state = next_state
    def handle(self, simulation):
        simulation.system_time = self.time
        simulation.system_state = self.next_state
        simulation.system_is_free = True

        

        if simulation.system_state[1] == 1:

            simulation.schedule_event(StateChangeEvent(self.time+T_TUNE, ((simulation.system_state[0]+1)%NUMBER_OF_THREADS,0)))

            simulation.take_snapshot()

        else:
            state_duration = min((simulation.threads_queues[simulation.system_state[0]].length()+ADDITION)/CONNECTION_SPEED, T_MAX)

            simulation.redline = self.time + state_duration + T_TUNE
            simulation.schedule_event(StateChangeEvent(
                self.time+state_duration, 
                (simulation.system_state[0],1)))
            
            if not(simulation.threads_queues[simulation.system_state[0]].is_empty()) and (self.time+T_SERVE<simulation.redline):
                wait_time = simulation.system_time - simulation.threads_queues[simulation.system_state[0]].get()
                simulation.schedule_event(EndOfServingEvent(self.time + T_SERVE, simulation.system_state[0]))
                simulation.take_snapshot(update_mean_wait_time=True, wait_time=wait_time,state_duration = state_duration, update_mean_state_duration =True)
            else:
                simulation.take_snapshot(state_duration = state_duration, update_mean_state_duration =True)

class IncomeEvent(Event):
    def __init__(self, time, number_of_calls, thread_number):
       super().__init__(time)
       self.number_of_calls = number_of_calls
       self.thread_number = thread_number
       self.updates_mean_state_duration = False
    def handle(self, simulation):
        simulation.system_time = self.time
        
        if (self.thread_number == simulation.system_state[0]) and (simulation.system_state[1] == 0) and simulation.system_is_free and (self.time+T_SERVE<simulation.redline):
            simulation.schedule_event(EndOfServingEvent(self.time + T_SERVE, self.thread_number))
            simulation.system_is_free = False
            simulation.take_snapshot(update_mean_wait_time=True, wait_time=0)
            for i in range(self.number_of_calls-1):
                simulation.threads_queues[self.thread_number].append(self.time)
        else:
            simulation.take_snapshot()
            for i in range(self.number_of_calls):
                simulation.threads_queues[self.thread_number].append(self.time)



class EndOfServingEvent(Event):
    def __init__(self, time, thread_number):
        super().__init__(time)
        self.thread_number = thread_number
        self.updates_mean_state_duration = False
    def handle(self, simulation):
        simulation.system_time = self.time
       
        if (self.thread_number == simulation.system_state[0]) and (simulation.system_state[1] == 0) and not(simulation.threads_queues[self.thread_number].is_empty()) and (self.time+T_SERVE<simulation.redline):
            time_of_arrival = simulation.threads_queues[self.thread_number].get()
            wait_time = simulation.system_time - time_of_arrival
            simulation.take_snapshot(update_mean_wait_time=True, wait_time=wait_time)
            simulation.schedule_event(EndOfServingEvent(self.time + T_SERVE, self.thread_number))
        elif self.thread_number != simulation.system_state[0]:
            raise RuntimeError("Thread overloop!")
        else:
            simulation.take_snapshot()
            simulation.system_is_free = True
            return