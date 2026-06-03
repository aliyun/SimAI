.. include:: replace.txt
.. highlight:: cpp

.. heading hierarchy:
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)
   ~~~~~~~~~~~~~ Sub-paragraph (no number)

Probes
******

This section details the functionalities provided by the Probe class
to an |ns3| simulation, and gives examples on how to code them in a
program. This section is meant for users interested in developing
simulations with the |ns3| tools and using the Data Collection
Framework, of which the Probe class is a part, to generate data output
with their simulation's results.

Probe Overview
==============

A Probe object is supposed to be connected to a variable from the 
simulation whose values throughout the experiment are relevant to the user. 
The Probe will record what were values assumed by the variable throughout 
the simulation and pass such data to another member of the Data Collection 
Framework.  While it is out of this section's scope to discuss what happens 
after the Probe produces its output, it is sufficient to say that, by the 
end of the simulation, the user will have detailed information about what 
values were stored inside the variable being probed during the simulation.

Typically, a Probe is connected to an |ns3| trace source.  In this manner,
whenever the trace source exports a new value, the Probe consumes the
value (and exports it downstream to another object via its own trace source).

The Probe can be thought of as kind of a filter on trace sources.  The 
main reasons for possibly hooking to a Probe rather than directly to a 
trace source are as follows:

* Probes may be dynamically turned on and off during the simulation
  with calls to ``Enable()`` and ``Disable()``.  For example, the
  outputting of data may be turned off during the simulation warmup
  phase.
* Probes may perform operations on the data to extract values from more
  complicated structures; for instance, outputting the packet size value
  from a received ns3::Packet.
* Probes register a name in the ns3::Config namespace (using
  ``Names::Add ()``) so that other objects may refer to them.
* Probes provide a static method that allows one to manipulate a Probe
  by name, such as what is done in ns2measure [Cic06]_

  ::

    Stat::put ("my_metric", ID, sample);

  The ns-3 equivalent of the above ns2measure code is, e.g.

  ::

    DoubleProbe::SetValueByPath ("/path/to/probe", sample);

Creation
########

Note that a Probe base class object can not be created because it
is an abstract base class, i.e. it has pure virtual functions that
have not been implemented.  An object of type DoubleProbe,
which is a subclass of the Probe class, will be created here to
show what needs to be done.

One declares a DoubleProbe in dynamic memory by using the smart pointer class 
(Ptr<T>). To create a DoubleProbe in dynamic memory with smart pointers, one
just needs to call the |ns3| method ``CreateObject()``:

::

  Ptr<DoubleProbe> myprobe = CreateObject<DoubleProbe> ();

The declaration above creates DoubleProbes using the default values for its 
attributes.  There are four attributes in the DoubleProbe class; two in the 
base class object DataCollectionObject, and two in the Probe base class:

* "Name" (DataCollectionObject), a StringValue
* "Enabled" (DataCollectionObject), a BooleanValue
* "Start" (Probe), a TimeValue
* "Stop" (Probe), a TimeValue

One can set such attributes at object creation by using the following 
method:

::

  Ptr<DoubleProbe> myprobe = CreateObjectWithAttributes<DoubleProbe> (
      "Name", StringValue ("myprobe"), 
      "Enabled", BooleanValue (false),
      "Start", TimeValue (Seconds (100.0)),
      "Stop", TimeValue (Seconds (1000.0)));

Start and Stop are Time variables which determine the interval of action 
of the Probe. The Probe will only output data if the current time of the 
Simulation is inside of that interval.  The special time value of 0 seconds
for Stop will disable this attribute (i.e. keep the Probe on for the whole
simulation).  Enabled is a flag that turns the
Probe on or off, and must be set to true for the Probe to export data.
The Name is the object's name in the DCF framework.

Importing and exporting data
############################

|ns3| trace sources are strongly typed, so the mechanisms for hooking
Probes to a trace source and for exporting data belong to its
subclasses.  For instance, the default distribution of |ns3| provides
a class DoubleProbe that is designed to hook to a trace source
exporting a double value.  We'll next detail the operation of the
DoubleProbe, and then discuss how other Probe classes may be defined
by the user.

DoubleProbe Overview
====================

The DoubleProbe connects to a double-valued |ns3| trace source, and itself
exports a different double-valued |ns3| trace source.

The following code, drawn from
``src/stats/examples/double-probe-example.cc``, shows the basic
operations of plumbing the DoubleProbe into a simulation, where it is
probing a Counter exported by an emitter object (class Emitter).

::

  Ptr<Emitter> emitter = CreateObject<Emitter> ();  
  Names::Add ("/Names/Emitter", emitter);
  ...

  Ptr<DoubleProbe> probe1 = CreateObject<DoubleProbe> ();

  // Connect the probe to the emitter's Counter
  bool connected = probe1->ConnectByObject ("Counter", emitter);

The following code is probing the same Counter exported by the same
emitter object.  This DoubleProbe, however, is using a path in the
configuration namespace to make the connection.  Note that the emitter
registered itself in the configuration namespace after it was created;
otherwise, the ConnectByPath would not work.

::

  Ptr<DoubleProbe> probe2 = CreateObject<DoubleProbe> ();

  // Note, no return value is checked here
  probe2->ConnectByPath ("/Names/Emitter/Counter");

The next DoubleProbe shown that is shown below will have its value set using
its path in the configuration namespace.  Note that this time the
DoubleProbe registered itself in the configuration namespace after it was
created.

::

  Ptr<DoubleProbe> probe3 = CreateObject<DoubleProbe> ();
  probe3->SetName ("StaticallyAccessedProbe");

  // We must add it to the config database
  Names::Add ("/Names/Probes", probe3->GetName (), probe3);

The emitter's Count() function is now able to set the value for this DoubleProbe as follows: 

::

  void
  Emitter::Count (void)
  {
    ...
    m_counter += 1.0;
    DoubleProbe::SetValueByPath ("/Names/StaticallyAccessedProbe", m_counter);
    ...
  }

The above example shows how the code calling the Probe does not have to
have an explicit reference to the Probe, but can direct the value
setting through the Config namespace.  This is similar in functionality
to the `Stat::Put` method introduced by ns2measure paper [Cic06]_, and allows
users to temporarily insert Probe statements like `printf` statements within 
existing |ns3| models.  Note that in order to be able to use the DoubleProbe in this example like this, 2 things were necessary:

1. the stats module header file was included in the example .cc file 
2. the example was made dependent on the stats module in its CMakeLists.txt file.

Analogous things need to be done in order to add other Probes in other places in the |ns3| code base.

The values for the DoubleProbe can also be set using the function
DoubleProbe::SetValue(), while the values for the DoubleProbe can be gotten
using the function DoubleProbe::GetValue().

The DoubleProbe exports double values in its "Output" trace source;
a downstream object can hook a trace sink (NotifyViaProbe) to this as follows:
 
::

  connected = probe1->TraceConnect ("Output", probe1->GetName (), MakeCallback (&NotifyViaProbe));


Other probes
============

Besides the DoubleProbe, the following Probes are also available:

- Uinteger8Probe connects to an |ns3| trace source exporting an uint8_t.
- Uinteger16Probe connects to an |ns3| trace source exporting an uint16_t.
- Uinteger32Probe connects to an |ns3| trace source exporting an uint32_t.
- PacketProbe connects to an |ns3| trace source exporting a packet.
- ApplicationPacketProbe connects to an |ns3| trace source exporting a packet and a socket address.
- Ipv4PacketProbe connects to an |ns3| trace source exporting a packet, an IPv4 object, and an interface.

Creating new Probe types
========================

To create a new Probe type, you need to perform the following steps:

- Be sure that your new Probe class is derived from the Probe base class.
- Be sure that the pure virtual functions that your new Probe class
  inherits from the Probe base class are implemented.
- Find an existing Probe class that uses a trace source that is
  closest in type to the type of trace source your Probe will be
  using.
- Copy that existing Probe class's header file (.h) and implementation
  file (.cc) to two new files with names matching your new Probe.
- Replace the types, arguments, and variables in the copied files with
  the appropriate type for your Probe.
- Make necessary modifications to make the code compile and to make it
  behave as you would like.

Examples
========

Two examples will be discussed in detail here:

- Double Probe Example
- IPv4 Packet Plot Example

Double Probe Example
####################

The double probe example has been discussed previously.  The example
program can be found in
``src/stats/examples/double-probe-example.cc``.  To
summarize what occurs in this program, there is an emitter that
exports a counter that increments according to a Poisson process.  In
particular, two ways of emitting data are shown:

1. through a traced variable hooked to one Probe: 

   ::
  
     TracedValue<double> m_counter;  // normally this would be integer type
  
2. through a counter whose value is posted to a second Probe, referenced by its name in the Config system:

  ::

    void
    Emitter::Count (void)
    {
      NS_LOG_FUNCTION (this);
      NS_LOG_DEBUG ("Counting at " << Simulator::Now ().GetSeconds ());
      m_counter += 1.0;
      DoubleProbe::SetValueByPath ("/Names/StaticallyAccessedProbe", m_counter);
      Simulator::Schedule (Seconds (m_var->GetValue ()), &Emitter::Count, this);
    }

Let's look at the Probe more carefully.  Probes can receive their values
in a multiple ways:

1.  by the Probe accessing the trace source directly and connecting
    a trace sink to it
2.  by the Probe accessing the trace source through the config namespace 
    and connecting a trace sink to it
3.  by the calling code explicitly calling the Probe's `SetValue()` method
4.  by the calling code explicitly calling `SetValueByPath ("/path/through/Config/namespace", ...)`

The first two techniques are expected to be the most common.  Also in the 
example, the hooking of a normal callback function is shown,
as is typically done in |ns3|.  This callback function is not associated
with a Probe object.  We'll call this case 0) below.

::

  // This is a function to test hooking a raw function to the trace source
  void 
  NotifyViaTraceSource (std::string context, double oldVal, double newVal)
  {
    NS_LOG_DEBUG ("context: " << context << " old " << oldVal << " new " << newVal);
  }
  
First, the emitter needs to be setup:

::

    Ptr<Emitter> emitter = CreateObject<Emitter> ();
    Names::Add ("/Names/Emitter", emitter);

    // The Emitter object is not associated with an ns-3 node, so
    // it won't get started automatically, so we need to do this ourselves
    Simulator::Schedule (Seconds (0.0), &Emitter::Start, emitter);

The various DoubleProbes interact with the emitter in the example as
shown below.

Case 0):

  ::

    // The below shows typical functionality without a probe
    // (connect a sink function to a trace source)
    //
    connected = emitter->TraceConnect ("Counter", "sample context", MakeCallback (&NotifyViaTraceSource));
    NS_ASSERT_MSG (connected, "Trace source not connected");
  

case 1):

  ::

    // 
    // Probe1 will be hooked directly to the Emitter trace source object
    //
  
    // probe1 will be hooked to the Emitter trace source
    Ptr<DoubleProbe> probe1 = CreateObject<DoubleProbe> ();
    // the probe's name can serve as its context in the tracing
    probe1->SetName ("ObjectProbe");
  
    // Connect the probe to the emitter's Counter
    connected = probe1->ConnectByObject ("Counter", emitter);
    NS_ASSERT_MSG (connected, "Trace source not connected to probe1");

case 2):

  ::

    // 
    // Probe2 will be hooked to the Emitter trace source object by 
    // accessing it by path name in the Config database
    //
  
    // Create another similar probe; this will hook up via a Config path
    Ptr<DoubleProbe> probe2 = CreateObject<DoubleProbe> ();
    probe2->SetName ("PathProbe");
  
    // Note, no return value is checked here
    probe2->ConnectByPath ("/Names/Emitter/Counter");
  
case 4) (case 3 is not shown in this example):

  ::

    // 
    // Probe3 will be called by the emitter directly through the 
    // static method SetValueByPath().  
    //
    Ptr<DoubleProbe> probe3 = CreateObject<DoubleProbe> ();
    probe3->SetName ("StaticallyAccessedProbe");
    // We must add it to the config database
    Names::Add ("/Names/Probes", probe3->GetName (), probe3);
  
And finally, the example shows how the probes can be hooked to
generate output:

  ::

    // The probe itself should generate output.  The context that we provide
    // to this probe (in this case, the probe name) will help to disambiguate
    // the source of the trace
    connected = probe3->TraceConnect ("Output",
                                      "/Names/Probes/StaticallyAccessedProbe/Output",
                                      MakeCallback (&NotifyViaProbe));
    NS_ASSERT_MSG (connected, "Trace source not .. connected to probe3 Output");
  
The following callback is hooked to the Probe in this example for 
illustrative purposes; normally, the Probe would be hooked to a 
Collector object.

::

  // This is a function to test hooking it to the probe output
  void 
  NotifyViaProbe (std::string context, double oldVal, double newVal)
  {
    NS_LOG_DEBUG ("context: " << context << " old " << oldVal << " new " << newVal);
  }


IPv4 Packet Plot Example
########################

The IPv4 packet plot example is based on the fifth.cc example from the |ns3| 
Tutorial.  It can be found in 
``src/stats/examples/ipv4-packet-plot-example.cc``.

.. sourcecode:: text

          node 0                 node 1
    +----------------+    +----------------+
    |    ns-3 TCP    |    |    ns-3 TCP    |
    +----------------+    +----------------+
    |    10.1.1.1    |    |    10.1.1.2    |
    +----------------+    +----------------+
    | point-to-point |    | point-to-point |
    +----------------+    +----------------+
            |                     |
            +---------------------+


We'll just look at the Probe, as it illustrates that Probes may also
unpack values from structures (in this case, packets) and report
those values as trace source outputs, rather than just passing through
the same type of data.

There are other aspects of this example that will be explained later in
the documentation.  The two types of data that are exported are the packet
itself (`Output`) and a count of the number of bytes in the packet
(`OutputBytes`).

::

    TypeId
    Ipv4PacketProbe::GetTypeId ()
    {
      static TypeId tid = TypeId ("ns3::Ipv4PacketProbe")
        .SetParent<Probe> ()
        .AddConstructor<Ipv4PacketProbe> ()
        .AddTraceSource ( "Output",
                          "The packet plus its IPv4 object and interface that serve as the output for this probe",
                          MakeTraceSourceAccessor (&Ipv4PacketProbe::m_output))
        .AddTraceSource ( "OutputBytes",
                          "The number of bytes in the packet",
                          MakeTraceSourceAccessor (&Ipv4PacketProbe::m_outputBytes))
      ;
      return tid;
    }


When the Probe's trace sink gets a packet, if the Probe is enabled, then
it will output the packet on its `Output` trace source, but it will also
output the number of bytes on the `OutputBytes` trace source.

::

    void
    Ipv4PacketProbe::TraceSink (Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t interface)
    {
      NS_LOG_FUNCTION (this << packet << ipv4 << interface);
      if (IsEnabled ())
        {
          m_packet    = packet;
          m_ipv4      = ipv4;
          m_interface = interface;
          m_output (packet, ipv4, interface);
    
          uint32_t packetSizeNew = packet->GetSize ();
          m_outputBytes (m_packetSizeOld, packetSizeNew);
          m_packetSizeOld = packetSizeNew;
        }
    }


References
==========

.. [Cic06] Claudio Cicconetti, Enzo Mingozzi, Giovanni Stea, "An Integrated 
    Framework for Enabling Effective Data Collection and Statistical 
    Analysis with ns2, Workshop on ns-2 (WNS2), Pisa, Italy, October 2006.
