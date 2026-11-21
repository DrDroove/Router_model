from abc import ABC
from dataclasses import dataclass, field

@dataclass
class Event(ABC):
    time: float
    thread_number: int
    priority: int
    def __lt__(self, other: "Event"):
        return (self.time, self.priority) < (other.time, other.priority)

@dataclass
class StateChangeEvent(Event):
    next_state: tuple[int, int]
    priority: int = field(default= 2, init=False)

@dataclass
class IncomeEvent(Event):
    number_of_calls: int
    priority: int = field(default= 1, init=False)

@dataclass
class EndOfServingEvent(Event):
    priority: int = field(default= 0, init=False)
    