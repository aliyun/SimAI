.. include:: replace.txt
.. highlight:: cpp
.. role:: raw-role(raw)
   :format: html latex

Data Collection
---------------

Our final tutorial chapter introduces some components that were added
to |ns3| in version 3.18, and that are still under development.  This
tutorial section is also a work-in-progress.

Motivation
**********

One of the main points of running simulations is to generate output data, 
either for research purposes or simply to learn about the system.
In the previous chapter, we introduced the tracing subsystem and
the example ``sixth.cc``. from which PCAP or ASCII trace files are 
generated.  These traces are valuable for data analysis using a
variety of external tools, and for many users, such output data is
a preferred means of gathering data (for analysis by external tools).

However, there are also use cases for more than trace file generation,
including the following:

* generation of data that does not map well to PCAP or ASCII traces, such
  as non-packet data (e.g. protocol state machine transitions),
* large simulations for which the disk I/O requirements for generating
  trace files is prohibitive or cumbersome, and
* the need for *online*  data reduction or computation, during the course
  of the simulation.  A good example of this is to define a termination
  condition for the simulation, to tell it when to stop when it has 
  received enough data to form a narrow-enough confidence interval around
  the estimate of some parameter.

The |ns3| data collection framework is designed to provide these
additional capabilities beyond trace-based output.  We recommend
that the reader interested in this topic consult the |ns3| Manual
for a more detailed treatment of this framework; here, we summarize
with an example program some of the developing capabilities. 

Example Code
************

The tutorial example ``examples/tutorial/seventh.cc`` resembles the
``sixth.cc`` example we previously reviewed, except for a few changes.
First, it has been enabled for IPv6 support with a command-line option:

::   

  CommandLine cmd;
  cmd.AddValue ("useIpv6", "Use Ipv6", useV6);
  cmd.Parse (argc, argv);

If the user specifies ``useIpv6``, option, the program will be run
using IPv6 instead of IPv4.  The ``help`` option, available on all |ns3| 
programs that support the CommandLine object as shown above, can
be invoked as follows (please note the use of double quotes):

::

  ./ns3 run "seventh --help"

which produces:

::

  ns3-dev-seventh-debug [Program Arguments] [General Arguments]
  
  Program Arguments:
      --useIpv6:  Use Ipv6 [false]
  
  General Arguments:
      --PrintGlobals:              Print the list of globals.
      --PrintGroups:               Print the list of groups.
      --PrintGroup=[group]:        Print all TypeIds of group.
      --PrintTypeIds:              Print all TypeIds.
      --PrintAttributes=[typeid]:  Print all attributes of typeid.
      --PrintHelp:                 Print this help message.

This default (use of IPv4, since useIpv6 is false) can be changed by 
toggling the boolean value as follows:

::

  ./ns3 run "seventh --useIpv6=1"

and have a look at the pcap generated, such as with ``tcpdump``:

::

  tcpdump -r seventh.pcap -nn -tt

This has been a short digression into IPv6 support and the command line,
which was also introduced earlier in this tutorial.  For a dedicated 
example of command line usage, please see 
``src/core/examples/command-line-example.cc``.

Now back to data collection.  In the ``examples/tutorial/`` directory, 
type the following command: ``diff -u sixth.cc seventh.cc``, and examine
some of the new lines of this diff:

.. sourcecode:: diff

  +  std::string probeType;
  +  std::string tracePath;
  +  if (useV6 == false)
  +    {
     ...
  +      probeType = "ns3::Ipv4PacketProbe";
  +      tracePath = "/NodeList/*/$ns3::Ipv4L3Protocol/Tx";
  +    }
  +  else
  +    {
     ...
  +      probeType = "ns3::Ipv6PacketProbe";
  +      tracePath = "/NodeList/*/$ns3::Ipv6L3Protocol/Tx";
  +    }
   ...
  +   // Use GnuplotHelper to plot the packet byte count over time
  +   GnuplotHelper plotHelper;
  + 
  +   // Configure the plot.  The first argument is the file name prefix
  +   // for the output files generated.  The second, third, and fourth
  +   // arguments are, respectively, the plot title, x-axis, and y-axis labels
  +   plotHelper.ConfigurePlot ("seventh-packet-byte-count",
  +                             "Packet Byte Count vs. Time",
  +                             "Time (Seconds)",
  +                             "Packet Byte Count");
  + 
  +   // Specify the probe type, trace source path (in configuration namespace), and
  +   // probe output trace source ("OutputBytes") to plot.  The fourth argument
  +   // specifies the name of the data series label on the plot.  The last
  +   // argument formats the plot by specifying where the key should be placed.
  +   plotHelper.PlotProbe (probeType,
  +                         tracePath,
  +                         "OutputBytes",
  +                         "Packet Byte Count",
  +                         GnuplotAggregator::KEY_BELOW);
  + 
  +   // Use FileHelper to write out the packet byte count over time
  +   FileHelper fileHelper;
  + 
  +   // Configure the file to be written, and the formatting of output data.
  +   fileHelper.ConfigureFile ("seventh-packet-byte-count",
  +                             FileAggregator::FORMATTED);
  + 
  +   // Set the labels for this formatted output file.
  +   fileHelper.Set2dFormat ("Time (Seconds) = %.3e\tPacket Byte Count = %.0f");
  + 
  +   // Specify the probe type, probe path (in configuration namespace), and
  +   // probe output trace source ("OutputBytes") to write.
  +   fileHelper.WriteProbe (probeType,
  +                          tracePath,
  +                          "OutputBytes");
  + 
      Simulator::Stop (Seconds (20));
      Simulator::Run ();
      Simulator::Destroy ();
  

The careful reader will have noticed, when testing the IPv6 command
line attribute above, that ``seventh.cc`` had created a number of new output files:

::

  seventh-packet-byte-count-0.txt
  seventh-packet-byte-count-1.txt
  seventh-packet-byte-count.dat
  seventh-packet-byte-count.plt
  seventh-packet-byte-count.png
  seventh-packet-byte-count.sh

These were created by the additional statements introduced above; in
particular, by a GnuplotHelper and a FileHelper.  This data was produced
by hooking the data collection components to |ns3| trace sources, and
marshaling the data into a formatted ``gnuplot`` and into a formatted
text file.  In the next sections, we'll review each of these.

.. _GnuPlotHelper:

GnuplotHelper
*************

The GnuplotHelper is an |ns3| helper object aimed at the production of
``gnuplot`` plots with as few statements as possible, for common cases.
It hooks |ns3| trace sources with data types supported by the 
data collection system.  Not all |ns3| trace sources data types are 
supported, but many of the common trace types are, including TracedValues 
with plain old data (POD) types.

Let's look at the output produced by this helper:

::
  
  seventh-packet-byte-count.dat
  seventh-packet-byte-count.plt
  seventh-packet-byte-count.sh

The first is a gnuplot data file with a series of space-delimited 
timestamps and packet byte counts.  We'll cover how this particular
data output was configured below, but let's continue with the output
files.  The file ``seventh-packet-byte-count.plt`` is a gnuplot plot file,
that can be opened from within gnuplot.  Readers who understand gnuplot
syntax can see that this will produce a formatted output PNG file named
``seventh-packet-byte-count.png``.   Finally, a small shell script
``seventh-packet-byte-count.sh`` runs this plot file through gnuplot
to produce the desired PNG (which can be viewed in an image editor); that
is, the command:

:: 

  sh seventh-packet-byte-count.sh

will yield ``seventh-packet-byte-count.png``.  Why wasn't this PNG
produced in the first place?  The answer is that by providing the 
plt file, the user can hand-configure the result if desired, before
producing the PNG.

The PNG image title states that this plot is a plot of 
"Packet Byte Count vs. Time", and that it is plotting the probed data
corresponding to the trace source path:

.. sourcecode:: text

  /NodeList/*/$ns3::Ipv6L3Protocol/Tx

Note the wild-card in the trace path.  In summary, what this plot is 
capturing is the plot of packet bytes observed at the transmit trace
source of the Ipv6L3Protocol object; largely 596-byte TCP segments
in one direction, and 60-byte TCP acks in the other (two node
trace sources were matched by this trace source).

How was this configured?  A few statements need to be provided.  First,
the GnuplotHelper object must be declared and configured:

::
  
  +  // Use GnuplotHelper to plot the packet byte count over time
  +  GnuplotHelper plotHelper;
  +
  +  // Configure the plot.  The first argument is the file name prefix
  +  // for the output files generated.  The second, third, and fourth
  +  // arguments are, respectively, the plot title, x-axis, and y-axis labels
  +  plotHelper.ConfigurePlot ("seventh-packet-byte-count",
  +                            "Packet Byte Count vs. Time",
  +                            "Time (Seconds)",
  +                            "Packet Byte Count");


To this point, an empty plot has been configured.  The filename prefix
is the first argument, the plot title is the second, the x-axis label
the third, and the y-axis label the fourth argument.

The next step is to configure the data, and here is where the trace
source is hooked.  First, note above in the program we declared a few 
variables for later use:

::

  +  std::string probeType;
  +  std::string tracePath;
  +  probeType = "ns3::Ipv6PacketProbe";
  +  tracePath = "/NodeList/*/$ns3::Ipv6L3Protocol/Tx";

We use them here:

::
   
  +  // Specify the probe type, trace source path (in configuration namespace), and
  +  // probe output trace source ("OutputBytes") to plot.  The fourth argument
  +  // specifies the name of the data series label on the plot.  The last
  +  // argument formats the plot by specifying where the key should be placed.
  +  plotHelper.PlotProbe (probeType,
  +                        tracePath,
  +                        "OutputBytes",
  +                        "Packet Byte Count",
  +                        GnuplotAggregator::KEY_BELOW);

The first two arguments are the name of the probe type and the trace source path.
These two are probably the hardest to determine when you try to use
this framework to plot other traces.  The probe trace here is the ``Tx``
trace source of class ``Ipv6L3Protocol``.  When we examine this class
implementation (``src/internet/model/ipv6-l3-protocol.cc``) we can
observe:

::

    .AddTraceSource ("Tx", "Send IPv6 packet to outgoing interface.",
                     MakeTraceSourceAccessor (&Ipv6L3Protocol::m_txTrace))

This says that ``Tx`` is a name for variable ``m_txTrace``, which has
a declaration of:

::

  /**
   * \brief Callback to trace TX (transmission) packets.
   */
  TracedCallback<Ptr<const Packet>, Ptr<Ipv6>, uint32_t> m_txTrace;

It turns out that this specific trace source signature is supported
by a Probe class (what we need here) of class Ipv6PacketProbe.
See the files ``src/internet/model/ipv6-packet-probe.{h,cc}``.

So, in the PlotProbe statement above, we see that the statement is
hooking the trace source (identified by path string) with a matching
|ns3| Probe type of ``Ipv6PacketProbe``.  If we did not support
this probe type (matching trace source signature), we could have not
used this statement (although some more complicated lower-level statements
could have been used, as described in the manual).

The Ipv6PacketProbe exports, itself, some trace sources that extract
the data out of the probed Packet object:

::
  
  TypeId
  Ipv6PacketProbe::GetTypeId ()
  {
    static TypeId tid = TypeId ("ns3::Ipv6PacketProbe")
      .SetParent<Probe> ()
      .SetGroupName ("Stats")
      .AddConstructor<Ipv6PacketProbe> ()
      .AddTraceSource ( "Output",
                        "The packet plus its IPv6 object and interface that serve as the output for this probe",
                        MakeTraceSourceAccessor (&Ipv6PacketProbe::m_output))
      .AddTraceSource ( "OutputBytes",
                        "The number of bytes in the packet",
                        MakeTraceSourceAccessor (&Ipv6PacketProbe::m_outputBytes))
    ;
    return tid;
  }
  

The third argument of our PlotProbe statement specifies that we are
interested in the number of bytes in this packet; specifically, the
"OutputBytes" trace source of Ipv6PacketProbe.
Finally, the last two arguments of the statement provide the plot
legend for this data series ("Packet Byte Count"), and an optional
gnuplot formatting statement (GnuplotAggregator::KEY_BELOW) that we want 
the plot key to be inserted below the plot.  Other options include
NO_KEY, KEY_INSIDE, and KEY_ABOVE.


Supported Trace Types
*********************

The following traced values are supported with Probes as of this writing:

  +------------------+-------------------+------------------------------------+
  | TracedValue type |   Probe type      |  File                              |
  +==================+=========+=========+====================================+
  | double           | DoubleProbe       | stats/model/double-probe.h         |
  +------------------+-------------------+------------------------------------+
  | uint8_t          | Uinteger8Probe    | stats/model/uinteger-8-probe.h     |
  +------------------+-------------------+------------------------------------+
  | uint16_t         | Uinteger16Probe   | stats/model/uinteger-16-probe.h    |
  +------------------+-------------------+------------------------------------+
  | uint32_t         | Uinteger32Probe   | stats/model/uinteger-32-probe.h    |
  +------------------+-------------------+------------------------------------+
  | bool             | BooleanProbe      | stats/model/uinteger-16-probe.h    |
  +------------------+-------------------+------------------------------------+
  | ns3::Time        | TimeProbe         | stats/model/time-probe.h           |
  +------------------+-------------------+------------------------------------+

The following TraceSource types are supported by Probes as of this writing:

  +------------------------------------------+------------------------+---------------+----------------------------------------------------+
  | TracedSource type                        |   Probe type           | Probe outputs |  File                                              |
  +==========================================+========================+===============+====================================================+
  | Ptr<const Packet>                        | PacketProbe            | OutputBytes   | network/utils/packet-probe.h                       |
  +------------------------------------------+------------------------+---------------+----------------------------------------------------+
  | Ptr<const Packet>, Ptr<Ipv4>, uint32_t   | Ipv4PacketProbe        | OutputBytes   | internet/model/ipv4-packet-probe.h                 |
  +------------------------------------------+------------------------+---------------+----------------------------------------------------+
  | Ptr<const Packet>, Ptr<Ipv6>, uint32_t   | Ipv6PacketProbe        | OutputBytes   | internet/model/ipv6-packet-probe.h                 |
  +------------------------------------------+------------------------+---------------+----------------------------------------------------+
  | Ptr<const Packet>, Ptr<Ipv6>, uint32_t   | Ipv6PacketProbe        | OutputBytes   | internet/model/ipv6-packet-probe.h                 |
  +------------------------------------------+------------------------+---------------+----------------------------------------------------+
  | Ptr<const Packet>, const Address&        | ApplicationPacketProbe | OutputBytes   | applications/model/application-packet-probe.h      |
  +------------------------------------------+------------------------+---------------+----------------------------------------------------+

As can be seen, only a few trace sources are supported, and they are all 
oriented towards outputting the Packet size (in bytes).  However,
most of the fundamental data types available as TracedValues can be
supported with these helpers.

FileHelper
**********

The FileHelper class is just a variation of the previous GnuplotHelper
example.  The example program provides formatted output of the
same timestamped data, such as follows:

::
  
  Time (Seconds) = 9.312e+00	Packet Byte Count = 596
  Time (Seconds) = 9.312e+00	Packet Byte Count = 564

Two files are provided, one for node "0" and one for node "1" as can
be seen in the filenames.  Let's look at the code piece-by-piece:

::

  +   // Use FileHelper to write out the packet byte count over time
  +   FileHelper fileHelper;
  + 
  +   // Configure the file to be written, and the formatting of output data.
  +   fileHelper.ConfigureFile ("seventh-packet-byte-count",
  +                             FileAggregator::FORMATTED);

The file helper file prefix is the first argument, and a format specifier
is next.
Some other options for formatting include SPACE_SEPARATED, COMMA_SEPARATED,
and TAB_SEPARATED.  Users are able to change the formatting (if
FORMATTED is specified) with a format string such as follows:  

::

  + 
  +   // Set the labels for this formatted output file.
  +   fileHelper.Set2dFormat ("Time (Seconds) = %.3e\tPacket Byte Count = %.0f");

Finally, the trace source of interest must be hooked.  Again, the probeType and
tracePath variables in this example are used, and the probe's output
trace source "OutputBytes" is hooked:

::

  + 
  +   // Specify the probe type, trace source path (in configuration namespace), and
  +   // probe output trace source ("OutputBytes") to write.
  +   fileHelper.WriteProbe (probeType,
  +                          tracePath,
  +                          "OutputBytes");
  + 

The wildcard fields in this trace source specifier match two trace sources.
Unlike the GnuplotHelper example, in which two data series were overlaid
on the same plot, here, two separate files are written to disk. 

Summary
*******

Data collection support is new as of ns-3.18, and basic support for
providing time series output has been added.  The basic pattern described
above may be replicated within the scope of support of the existing
probes and trace sources.  More capabilities including statistics
processing will be added in future releases. 

