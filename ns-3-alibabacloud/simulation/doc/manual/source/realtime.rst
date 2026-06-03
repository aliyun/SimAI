.. include:: replace.txt
.. highlight:: cpp

RealTime
--------

|ns3| has been designed for integration into testbed and virtual machine
environments. To integrate with real network stacks and emit/consume packets, a
real-time scheduler is needed to try to lock the simulation clock with the
hardware clock. We describe here a component of this: the RealTime scheduler. 

The purpose of the realtime scheduler is to cause the progression of the
simulation clock to occur synchronously with respect to some external time base.
Without the presence of an external time base (wall clock), simulation time
jumps instantly from one simulated time to the next.

Behavior
********

When using a non-realtime scheduler (the default in |ns3|), the simulator
advances the simulation time to the next scheduled event. During event
execution, simulation time is frozen. With the realtime scheduler, the behavior
is similar from the perspective of simulation models (i.e., simulation time is
frozen during event execution), but between events, the simulator will attempt
to keep the simulation clock aligned with the machine clock.

When an event is finished executing, and the scheduler moves to the next event,
the scheduler compares the next event execution time with the machine clock.  If
the next event is scheduled for a future time, the simulator sleeps until that
realtime is reached and then executes the next event.

It may happen that, due to the processing inherent in the execution of
simulation events, that the simulator cannot keep up with realtime.  In such a
case, it is up to the user configuration what to do. There are two |ns3|
attributes that govern the behavior. The first is
``ns3::RealTimeSimulatorImpl::SynchronizationMode``. The two entries possible
for this attribute are ``BestEffort`` (the default) or ``HardLimit``. In
"BestEffort" mode, the simulator will just try to catch up to realtime by
executing events until it reaches a point where the next event is in the
(realtime) future, or else the simulation ends. In BestEffort mode, then, it is
possible for the simulation to consume more time than the wall clock time. The
other option "HardLimit" will cause the simulation to abort if the tolerance
threshold is exceeded.  This attribute is
``ns3::RealTimeSimulatorImpl::HardLimit`` and the default is 0.1 seconds.   

A different mode of operation is one in which simulated time is **not** frozen
during an event execution. This mode of realtime simulation was implemented but
removed from the |ns3| tree because of questions of whether it would be useful.
If users are interested in a realtime simulator for which simulation time does
not freeze during event execution (i.e., every call to ``Simulator::Now()``
returns the current wall clock time, not the time at which the event started
executing), please contact the ns-developers mailing list.

Usage
*****

The usage of the realtime simulator is straightforward, from a scripting
perspective.  Users just need to set the attribute 
``SimulatorImplementationType`` to the Realtime simulator, such as follows: ::

  GlobalValue::Bind ("SimulatorImplementationType",
    StringValue ("ns3::RealtimeSimulatorImpl"));

There is a script in ``examples/realtime/realtime-udp-echo.cc`` that
has an example of how to configure the realtime behavior.  Try:

.. sourcecode:: bash

    $ ./ns3 run realtime-udp-echo

Whether the simulator will work in a best effort or hard limit policy fashion is
governed by the attributes explained in the previous section.

Implementation
**************

The implementation is contained in the following files:

* ``src/core/model/realtime-simulator-impl.{cc,h}``
* ``src/core/model/wall-clock-synchronizer.{cc,h}``

In order to create a realtime scheduler, to a first approximation you just want
to cause simulation time jumps to consume real time. We propose doing this using
a combination of sleep- and busy- waits. Sleep-waits cause the calling process
(thread) to yield the processor for some amount of time. Even though this
specified amount of time can be passed to nanosecond resolution, it is actually
converted to an OS-specific granularity. In Linux, the granularity is called a
Jiffy. Typically this resolution is insufficient for our needs (on the order of
a ten milliseconds), so we round down and sleep for some smaller number of
Jiffies. The process is then awakened after the specified number of Jiffies has
passed. At this time, we have some residual time to wait. This time is generally
smaller than the minimum sleep time, so we busy-wait for the remainder of the
time. This means that the thread just sits in a for loop consuming cycles until
the desired time arrives. After the combination of sleep- and busy-waits, the
elapsed realtime (wall) clock should agree with the simulation time of the next
event and the simulation proceeds. 
