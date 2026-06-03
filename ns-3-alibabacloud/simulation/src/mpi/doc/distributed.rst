.. include:: replace.txt

MPI for Distributed Simulation
------------------------------

Parallel and distributed discrete event simulation allows the execution of a
single simulation program on multiple processors. By splitting up the simulation
into logical processes, LPs, each LP can be executed by a different processor.
This simulation methodology enables very large-scale simulations by leveraging
increased processing power and memory availability. In order to ensure proper
execution of a distributed simulation, message passing between LPs is required.
To support distributed simulation in |ns3|, the standard Message Passing
Interface (MPI) is used, along with a new distributed simulator class.
Currently, dividing a simulation for distributed purposes in |ns3| can only occur
across point-to-point links.

.. _current-implementation-details:

Current Implementation Details
******************************

During the course of a distributed simulation, many packets must cross simulator
boundaries. In other words, a packet that originated on one LP is destined for a
different LP, and in order to make this transition, a message containing the
packet contents must be sent to the remote LP. Upon receiving this message, the
remote LP can rebuild the packet and proceed as normal. The process of sending
an receiving messages between LPs is handled easily by the new MPI interface in
|ns3|.

Along with simple message passing between LPs, a distributed simulator is used
on each LP to determine which events to process. It is important to process
events in time-stamped order to ensure proper simulation execution. If a LP
receives a message containing an event from the past, clearly this is an issue,
since this event could change other events which have already been executed. To
address this problem, two conservative synchronization algorithm with lookahead are
used in |ns3|. For more information on different synchronization approaches and
parallel and distributed simulation in general, please refer to "Parallel and
Distributed Simulation Systems" by Richard Fujimoto.   

The default parallel synchronization strategy implemented in the
DistributedSimulatorImpl class is based on a globally synchronized
algorithm using an MPI collective operation to synchronize simulation
time across all LPs.  A second synchronization strategy based on local
communication and null messages is implemented in the
NullMessageSimulatorImpl class, For the null message strategy the
global all to all gather is not required; LPs only need to
communication with LPs that have shared point-to-point links.  The
algorithm to use is controlled by which the |ns3| global value
SimulatorImplementationType.

The best algorithm to use is dependent on the communication and event
scheduling pattern for the application.  In general, null message
synchronization algorithms will scale better due to local
communication scaling better than a global all-to-all gather that is
required by DistributedSimulatorImpl.  There are two known cases where
the global synchronization performs better.  The first is when most
LPs have point-to-point link with most other LPs, in other words the
LPs are nearly fully connected.  In this case the null message
algorithm will generate more message passing traffic than the
all-to-all gather.  A second case where the global all-to-all gather
is more efficient is when there are long periods of simulation time
when no events are occurring.  The all-to-all gather algorithm is able
to quickly determine then next event time globally.  The nearest
neighbor behavior of the null message algorithm will require more
communications to propagate that knowledge; each LP is only aware of
neighbor next event times.


Remote point-to-point links
+++++++++++++++++++++++++++

As described in the introduction, dividing a simulation for distributed purposes
in |ns3| currently can only occur across point-to-point links; therefore, the
idea of remote point-to-point links is very important for distributed simulation
in |ns3|. When a point-to-point link is installed, connecting two nodes, the
point-to-point helper checks the system id, or rank, of both nodes. The rank
should be assigned during node creation for distributed simulation and is
intended to signify on which LP a node belongs. If the two nodes are on the same
rank, a regular point-to-point link is created. If, however, the two nodes are
on different ranks, then these nodes are intended for different LPs, and a
remote point-to-point link is used. If a packet is to be sent across a remote
point-to-point link, MPI is used to send the message to the remote LP.

Distributing the topology
+++++++++++++++++++++++++

Currently, the full topology is created on each rank, regardless of the
individual node system ids. Only the applications are specific to a rank. For
example, consider node 1 on LP 1 and node 2 on LP 2, with a traffic generator on
node 1. Both node 1 and node 2 will be created on both LP1 and LP2; however, the
traffic generator will only be installed on LP1. While this is not optimal for
memory efficiency, it does simplify routing, since all current routing
implementations in |ns3| will work with distributed simulation.

Running Distributed Simulations
*******************************

Prerequisites
+++++++++++++
.. highlight:: bash

Ensure that MPI is installed, as well as mpic++. In Ubuntu repositories, 
these are openmpi-bin, openmpi-common, openmpi-doc, libopenmpi-dev. In 
Fedora, these are openmpi and openmpi-devel.

Note: 

There is a conflict on some Fedora systems between libotf and openmpi. A 
possible "quick-fix" is to yum remove libotf before installing openmpi. 
This will remove conflict, but it will also remove emacs. Alternatively, 
these steps could be followed to resolve the conflict:

    1) Rename the tiny otfdump which emacs says it needs::

         $ mv /usr/bin/otfdump /usr/bin/otfdump.emacs-version

    2) Manually resolve openmpi dependencies::

         $ sudo yum install libgfortran libtorque numactl

    3) Download rpm packages: 

       .. sourcecode:: text

         openmpi-1.3.1-1.fc11.i586.rpm
         openmpi-devel-1.3.1-1.fc11.i586.rpm
         openmpi-libs-1.3.1-1.fc11.i586.rpm
         openmpi-vt-1.3.1-1.fc11.i586.rpm

       from http://mirrors.kernel.org/fedora/releases/11/Everything/i386/os/Packages/

    4) Force the packages in::

         $ sudo rpm -ivh --force \
         openmpi-1.3.1-1.fc11.i586.rpm \
         openmpi-libs-1.3.1-1.fc11.i586.rpm \
         openmpi-devel-1.3.1-1.fc11.i586.rpm \
         openmpi-vt-1.3.1-1.fc11.i586.rpm

Also, it may be necessary to add the openmpi bin directory to PATH in order to
execute mpic++ and mpirun from the command line. Alternatively, the full path to
these executables can be used. Finally, if openmpi complains about the inability
to open shared libraries, such as libmpi_cxx.so.0, it may be necessary to add
the openmpi lib directory to LD_LIBRARY_PATH.

Here is an example of setting up PATH and LD_LIBRARY_PATH using a bash shell:

    * For a 32-bit Linux distribution::
         
         $ export PATH=$PATH:/usr/lib/openmpi/bin
         $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/openmpi/lib      
    
     For a 64-bit Linux distribution::
     
         $ export PATH=$PATH:/usr/lib64/openmpi/bin
         $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/openmpi/lib          

These lines can be added into ~/.bash_profile or ~/.bashrc to avoid having to
retype them when a new shell is opened.

Note 2:  There is a separate issue on recent Fedora distributions, which is
that the libraries are built with AVX instructions.  On older machines or
some virtual machines, this results in an illegal instruction 
being thrown.  This is not an |ns3| issue, a simple MPI test case will also
fail.  The AVX instructions are being called during initialization.

The symptom of this is that attempts to run an ns-3 MPI program will fail
with the error: `terminated with signal SIGILL`.  To check if this is the
problem, run::

    $ grep avx /proc/cpuinfo

and it will not return anything if AVX is not present.

If AVX is not supported, it is recommended to switch to a different MPI
implementation such as MPICH::

    $ dnf remove openmpi openmpi-devel
    $ dnf install mpich mpich-devel environment-modules
    $ module load mpi/mpich-x86_64

Building and Running Examples
+++++++++++++++++++++++++++++

If you already built |ns3| without MPI enabled, you must re-build::

    $ ./ns3 distclean

Configure |ns3| with the --enable-mpi option::

    $ ./ns3 configure -d debug --enable-examples --enable-tests --enable-mpi

Ensure that MPI is enabled by checking the optional features shown from the
output of configure.

Next, build |ns3|::

    $ ./ns3

After building |ns3| with mpi enabled, the example programs are now
ready to run with `mpiexec`.  It is advised to avoid running ns3 directly
with `mpiexec`; two options that should be more robust are to either use
the `--command-template` way of running the mpiexec program, or to use
`./ns3 shell` and run the executables directly on the command line.
Here are a few examples (from the root |ns3| directory)::

    $ ./ns3 run simple-distributed --command-template="mpiexec -np 2 %s"
    $ ./ns3 run nms-p2p-nix-distributed --command-template="mpiexec -np 2 -machinefile mpihosts %s --nix=0"
            
An example using the null message synchronization algorithm::

    $ ./ns3 run simple-distributed --command-template="mpiexec -np 2 %s --nullmsg"

The np switch is the number of logical processors to use. The machinefile switch
is which machines to use. In order to use machinefile, the target file must
exist (in this case mpihosts). This can simply contain something like:

.. sourcecode:: text

    localhost
    localhost
    localhost
    ...

Or if you have a cluster of machines, you can name them.

The other alternative to `command-template` is to use `./ns3 shell`.  Here
are the equivalent examples to the above (assuming optimized build profile)::

    $ ./ns3 shell
    $ cd build/src/mpi/examples
    $ mpiexec -np 2 ns3-dev-simple-distributed-optimized
    $ mpiexec -np 2 -machinefile mpihosts ns3-dev-nms-p2p-nix-distributed-optimized --nix=0
    $ mpiexec -np 2 ns3-dev-simple-distributed-optimized --nullmsg

Setting synchronization algorithm to use
++++++++++++++++++++++++++++++++++++++++

The global value SimulatorImplementationType is used to set the
synchronization algorithm to use.  This value must be set before the
MpiInterface::Enable method is invoked if the default
DistributedSimulatorImpl is not used.  Here is an example code snippet
showing how to add a command line argument to control the
synchronization algorithm choice:::

  cmd.AddValue ("nullmsg", "Enable the use of null-message synchronization", nullmsg);
  if(nullmsg) 
    {
      GlobalValue::Bind ("SimulatorImplementationType",
                         StringValue ("ns3::NullMessageSimulatorImpl"));
    } 
  else 
    {
      GlobalValue::Bind ("SimulatorImplementationType",
                         StringValue ("ns3::DistributedSimulatorImpl"));
    }

  // Enable parallel simulator with the command line arguments
  MpiInterface::Enable (&argc, &argv);



Creating custom topologies
++++++++++++++++++++++++++
.. highlight:: cpp

The example programs in src/mpi/examples give a good idea of how to create different
topologies for distributed simulation. The main points are assigning system ids
to individual nodes, creating point-to-point links where the simulation should
be divided, and installing applications only on the LP associated with the
target node.

Assigning system ids to nodes is simple and can be handled two different ways.
First, a NodeContainer can be used to create the nodes and assign system ids::

    NodeContainer nodes;
    nodes.Create (5, 1); // Creates 5 nodes with system id 1.

Alternatively, nodes can be created individually, assigned system ids, and added
to a NodeContainer. This is useful if a NodeContainer holds nodes with different
system ids::

    NodeContainer nodes;
    Ptr<Node> node1 = CreateObject<Node> (0); // Create node1 with system id 0
    Ptr<Node> node2 = CreateObject<Node> (1); // Create node2 with system id 1
    nodes.Add (node1);
    nodes.Add (node2);

Next, where the simulation is divided is determined by the placement of 
point-to-point links. If a point-to-point link is created between two 
nodes with different system ids, a remote point-to-point link is created, 
as described in :ref:`current-implementation-details`.

Finally, installing applications only on the LP associated with the target node
is very important. For example, if a traffic generator is to be placed on node
0, which is on LP0, only LP0 should install this application.  This is easily
accomplished by first checking the simulator system id, and ensuring that it
matches the system id of the target node before installing the application.

Tracing During Distributed Simulations
**************************************

Depending on the system id (rank) of the simulator, the information traced will
be different, since traffic originating on one simulator is not seen by another
simulator until it reaches nodes specific to that simulator. The easiest way to
keep track of different traces is to just name the trace files or pcaps
differently, based on the system id of the simulator. For example, something
like this should work well, assuming all of these local variables were
previously defined::

    if (MpiInterface::GetSystemId () == 0)
      {
        pointToPoint.EnablePcapAll ("distributed-rank0");
        phy.EnablePcap ("distributed-rank0", apDevices.Get (0));
        csma.EnablePcap ("distributed-rank0", csmaDevices.Get (0), true);
      }
    else if (MpiInterface::GetSystemId () == 1)
      {
        pointToPoint.EnablePcapAll ("distributed-rank1");
        phy.EnablePcap ("distributed-rank1", apDevices.Get (0));
        csma.EnablePcap ("distributed-rank1", csmaDevices.Get (0), true);
      }
