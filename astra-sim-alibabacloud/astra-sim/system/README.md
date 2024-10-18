This layer sits on top of the network simulator (e.g., gem5-garnet or NS3) and is responsible for executing and managing collective communication algorithms.
It maanages scheduling of the workload over the network, compute (and/or memory) simulators via APIs.
The APIs are defined in AstraNetworkAPI.hh, AstraComputeAPI.hh, and AstraMemoryAPI.hh
