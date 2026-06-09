import logging
import os

from dataclasses import dataclass, field
from enum import IntEnum

# from instance import Instance
# from simulator import clock, schedule_event, cancel_event, reschedule_event

# >
from vidur.entities.replica import Replica

class ProcessorType(IntEnum):
   DEFAULT = 0  # Default processor type
   CPU = 1  # CPU processor type
   GPU = 2  # GPU processor type


@dataclass(kw_only=True)
class Processor():
   """
    Processor is the lowest-level processing unit that can execute computations (tasks).
    Multiple Processors form a Server and may be connected via Interconnects.
    For example, both CPU and GPU are different types of Processors.
    
    Each Processor can only belong to one Server.
    A Processor can ultimately run multiple Instances/tasks.

    Attributes:
        processor_type (ProcessorType): The type of the processor.
        memory_size (float): The memory size of the processor.
        memory_used (float): The memory used by the processor.
        server (Server): The Server to which the processor belongs.
        instances (list[Instance]): Instances running on this processor.
        interconnects (list[Link]): Peers directly connected to this processor.
   """
   processor_type: ProcessorType = None 
   name: str = None # Processor name
   server: 'Server' = None  # Parent server
   memory_size: int = 0 
   memory_used: int = 0  
   _memory_used: int = 0 # Internal used memory
   power: float = 0.  # Power consumption
   _power: float = 0.  # Internal power consumption
   #instances: list[Instance] = field(default_factory=list)  # List of running instances
   instances: list[Replica] = field(default_factory=list) 
   interconnects: list['Link'] = field(default_factory=list)  

   @property
   def server(self):
       return self._server  # Return parent server

   @server.setter
   def server(self, server):
       if type(server) is property:
           server = None  # If server is a property type, set to None
       self._server = server  # Set parent server

   @property
   def memory_used(self):
       return self._memory_used  # Return used memory

   @memory_used.setter
   def memory_used(self, memory_used):
       if type(memory_used) is property:
           memory_used = 0  # If used memory is a property type, set to 0
       if memory_used < 0:
           raise ValueError("Memory cannot be negative")  # Memory cannot be negative
       # If memory overflow, record instance details
       if memory_used > self.memory_size:
           if os.path.exists("oom.csv") is False:
               with open("oom.csv", "w", encoding="UTF-8") as f:
                   fields = ["time", 
                             "instance_name",  
                             "instance_id",  
                             "memory_used",  
                             "processor_memory",  
                             "pending_queue_length"]  
                   f.write(",".join(fields) + "\n")  # Write CSV header
           with open("oom.csv", "a", encoding="UTF-8") as f:
               instance = self.instances[0]
               csv_entry = []
               csv_entry.append(clock())   # Current time
               csv_entry.append(instance.name) 
               csv_entry.append(instance.instance_id)  
               csv_entry.append(memory_used) 
               csv_entry.append(self.memory_size)  
               csv_entry.append(len(instance.pending_queue))  
               f.write(",".join(map(str, csv_entry)) + "\n")  # Write to CSV file
           # Raise memory overflow error
           #raise ValueError("Out of memory")
       self._memory_used = memory_used  # Set used memory

   @property
   def memory_free(self):
       return self.memory_size - self.memory_used  # Return free memory

   @property
   def power(self):
       return self._power  # Return power consumption

   @power.setter
   def power(self, power):
       if type(power) is property:
           power = 0.  # If power is a property type, set to 0
       if power < 0:
           raise ValueError("Power cannot be negative")  # Power cannot be negative
       self._power = power  # Set power consumption

   @property
   def peers(self):
       pass  # Return peers

@dataclass(kw_only=True)
class CPU(Processor):
   processor_type: ProcessorType = ProcessorType.CPU  # Processor type is CPU

@dataclass(kw_only=True)
class GPU(Processor):
   processor_type: ProcessorType = ProcessorType.GPU  # Processor type is GPU

if __name__ == "__main__":
   pass  # Main entry point