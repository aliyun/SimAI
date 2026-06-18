
UNISON for ns-3
================================

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10077300.svg)](https://doi.org/10.5281/zenodo.10077300)

A fast and user-transparent parallel simulator implementation for ns-3.
More information about UNISON can be found in our EuroSys '24 paper (coming soon).

## Getting Started

The quickest way to get started is to type the command

```shell
./ns3 configure --enable-mtp --enable-examples
```

> The build profile is set to default (which uses `-O2 -g` compiler flags) in this case.
> If you want to get `-O3` optimized build and discard all log outputs, please add `-d optimized` arguments.

The `--enable-mtp` option will enable multi-threaded parallelization.
You can verify UNISON is enabled by checking whether `Multithreaded Simulation : ON` appears in the optional feature list.

Now, let's build and run a DCTCP example with default sequential simulation and parallel simulation (using 4 threads) respectively:

```shell
./ns3 build dctcp-example dctcp-example-mtp
time ./ns3 run dctcp-example
time ./ns3 run dctcp-example-mtp
```

The simulation should finish in 4-5 minutes for `dctcp-example` and 1-2 minutes for `dctcp-example-mtp`, depending on your hardware and your build profile.
The output in `*.dat` should be in accordance with the comments in the source file.

The speedup of UNISON is more significant for larger topologies and traffic volumes.
If you are interested in using it to simulate topologies like fat-tree, BCube and 2D-torus, please refer to [Running Evaluations](#running-evaluations).

## Speedup Your Existing Code

To understand how UNISON affects your model code, let's find the differences between two versions of the source files of the above example:

```shell
diff examples/tcp/dctcp-example.cc examples/mtp/dctcp-example-mtp.cc
```

It turns out that to bring UNISON to existing model code, all you need to do is to include the `ns3/mtp-interface.h` header file and add the following line at the beginning of the `main` function:

```c++
MtpInterface::Enable (numberOfThreads);
```

The parameter `numberOfThreads` is optional.
If it is omitted, the number of threads is automatically chosen and will not exceed the maximum number of available hardware threads on your system. If you want to enable UNISON for distributed simulation on existing MPI programs for further speedup, place the above line before MPI initialization.

UNISON resolved a lot of thread-safety issues with ns-3's architecture.
You don't need to consider these issues on your own for most of the time, except if you have custom global statistics other than the built-in flow-monitor.
In the latter case, if multiple nodes can access your global statistics, you can replace them with atomic variables via `std::atomic<>`.
For complex custom data structures, you can create critical sections by adding

```c++
MtpInterface::CriticalSection cs;
```

at the beginning of your methods.

## Running Evaluations

To evaluate UNISON, please checkout to [unison-evaluations](https://github.com/NASA-NJU/UNISON-for-ns-3/tree/unison-evaluations) branch.
In this branch, you can find various topology models in the `scratch` folder.
There are a lot of parameters you can set for each topology.
We provided a utility script `exp.py` to compare these simulators and parameters.
We also provided `process.py` to convert these raw experiment data to CSV files suitable for ploting.
Please see the [README in that branch](https://github.com/NASA-NJU/UNISON-for-ns-3/tree/unison-evaluations) for more details.

## Module Documentation

### 1. Overview

UNISON for ns-3 is mainly implemented in the `mtp` module (located at `src/mtp/*`), which stands for multi-threaded parallelization.
This module contains three parts: A parallel simulator implementation `multithreaded-simulator-impl`, an interface to users `mtp-interface`, and `logical-process` to represent LPs in terms of parallel simulation.

All LPs and threads are stored in the `mtp-interface`.
It controls the simulation progress, schedules LPs to threads and manages the lifecycles of LPs and threads.
The interface also provides some methods and options for users to tweak the simulation.

Each LP's logic is implemented in `logical-process`. It contains most of the methods of the default sequential simulator plus some auxiliary methods for parallel simulation.

The simulator implementation `multithreaded-simulator-impl` is a derived class from the base simulator.
It converts calls to the base simulator into calls to logical processes based on the context of the current thread.
It also provides a partition method for automatic fine-grained topology partition.

For distributed simulation with MPI, we added `hybrid-simulator-impl` in the `mpi` module (located at `src/mpi/model/hybrid-simulator-impl*`).
This simulator uses both `mtp-interface` and `mpi-interface` to coordinate local LPs and global MPI communications.
We also modified the module to make it locally thread-safe.

### 2. Modifications to ns-3 Architecture

In addition to the `mtp` and `mpi` modules, we also modified the following part of the ns-3 architecture to make it thread-safe, also with some bug fixing for ns-3.

Modifications to the build system to provide `--enable-mtp` option to enable/disable UNISON:

```
ns3                                                |   2 +
CMakeLists.txt                                     |   1 +
build-support/custom-modules/ns3-configtable.cmake |   3 +
build-support/macros-and-definitions.cmake         |  10 +
```

Modifications to the `core` module to make reference counting thread-safe:

```
src/core/CMakeLists.txt                            |   1 +
src/core/model/atomic-counter.h                    |  49 ++
src/core/model/hash.h                              |  16 +
src/core/model/object.cc                           |   2 +
src/core/model/simple-ref-count.h                  |  11 +-
```

Modifications to the `network` module to make packets thread-safe:

```
src/network/model/buffer.cc                        |  15 +-
src/network/model/buffer.h                         |   7 +
src/network/model/byte-tag-list.cc                 |  14 +-
src/network/model/node.cc                          |   7 +
src/network/model/node.h                           |   7 +
src/network/model/packet-metadata.cc               |  10 +-
src/network/model/packet-metadata.h                |  11 +-
src/network/model/packet-tag-list.h                |  11 +-
src/network/model/socket.cc                        |   6 +
src/network/utils/simple-net-device.cc             |   4 +
```

Modifications to the `internet` module to make it thread-safe, plus adding per-flow ECMP routing and increasing # of available ports to support a large number of flows:

```
src/internet/model/global-route-manager-impl.cc    |   2 +
src/internet/model/ipv4-end-point-demux.cc         |   2 +-
src/internet/model/ipv4-global-routing.cc          |  31 +-
src/internet/model/ipv4-global-routing.h           |   5 +-
src/internet/model/ipv4-packet-info-tag.cc         |   2 +
src/internet/model/ipv6-end-point-demux.cc         |   4 +-
src/internet/model/ipv6-packet-info-tag.cc         |   2 +
src/internet/model/tcp-option.cc                   |   2 +-
```

Modifications to the `flow-monitor` module to make it thread-safe:

```
src/flow-monitor/model/flow-monitor.cc             |  42 +
src/flow-monitor/model/flow-monitor.h              |   4 +
src/flow-monitor/model/ipv4-flow-classifier.cc     |  12 +
src/flow-monitor/model/ipv4-flow-classifier.h      |   4 +
src/flow-monitor/model/ipv4-flow-probe.cc          |   2 +
src/flow-monitor/model/ipv6-flow-classifier.cc     |  12 +
src/flow-monitor/model/ipv6-flow-classifier.h      |   4 +
src/flow-monitor/model/ipv6-flow-probe.cc          |   2 +
```

Modifications to the `nix-vector-routing` module to make it thread-safe:

```
src/nix-vector-routing/model/nix-vector-routing.cc |  92 ++
src/nix-vector-routing/model/nix-vector-routing.h  |   8 +
```

### 3. Logging

The reason behind UNISON's fast speed is that it divides the network into multiple logical processes (LPs) with fine granularity and schedules them dynamically.
To get to know more details of such workflow, you can enable the following log component:

```c++
LogComponentEnable ("LogicalProcess", LOG_LEVEL_INFO);
LogComponentEnable ("MultithreadedSimulatorImpl", LOG_LEVEL_INFO);
```

### 4. Advanced Options

These options can be modified at the beginning of the `main` function using the native config syntax of ns-3.

You can also change the default maximum number of threads by setting

```c++
Config::SetDefault ("ns3::MultithreadedSimulatorImpl::MaxThreads", UintegerValue (8));
Config::SetDefault ("ns3::HybridSimulatorImpl::MaxThreads", UintegerValue (8));
```

The automatic partition will cut off stateless links whose delay is above the threshold.
The threshold is automatically calculated based on the delay of every link.
If you are not satisfied with the partition results, you can set a custom threshold by setting

```c++
Config::SetDefault ("ns3::MultithreadedSimulatorImpl::MinLookahead", TimeValue (NanoSeconds (500));
Config::SetDefault ("ns3::HybridSimulatorImpl::MinLookahead", TimeValue (NanoSeconds (500));
```

The scheduling method determines the priority (estimated completion time of the next round) of each logical process.
There are five available options:

- `ByExecutionTime`: LPs with higher execution time of the last round will have higher priority.
- `ByPendingEventCount`: LPs with more pending events of this round will have higher priority.
- `ByEventCount`: LPs with more pending events of this round will have higher priority.
- `BySimulationTime`: LPs with larger current clock time will have higher priority.
- `None`: Do not schedule. The partition's priority is based on their ID.

Many experiments show that the first one usually leads to better performance.
However, you can still choose one according to your taste by setting

```c++
GlobalValue::Bind ("PartitionSchedulingMethod", StringValue ("ByExecutionTime"));
```

By default, the scheduling period is 2 when the number of partitions is less than 16, 3 when it is less than 256, 4 when it is less than 4096, etc.
Since more partitions lead to more scheduling costs.
You can also set how frequently scheduling occurs by setting

```c++
GlobalValue::Bind ("PartitionSchedulingPeriod", UintegerValue (4));
```

## Links

If you find the code useful, please consider citing our paper (coming soon).
Below are some links that may also be helpful to you:

- [ns-3 Tutorial](https://www.nsnam.org/docs/tutorial/html/index.html)
- [ns-3 Model Library](https://www.nsnam.org/docs/models/html/index.html)
- [ns-3 Manual](https://www.nsnam.org/docs/manual/html/index.html)
