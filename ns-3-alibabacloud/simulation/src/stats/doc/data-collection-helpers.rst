.. include:: replace.txt
.. highlight:: cpp

.. heading hierarchy:
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)
   ~~~~~~~~~~~~~ Sub-paragraph (no number)

Data Collection Helpers
***********************

The full flexibility of the data collection framework is provided by
the interconnection of probes, collectors, and aggregators.  Performing
all of these interconnections leads to many configuration statements
in user programs.  For ease of use, some of the most common operations
can be combined and encapsulated in helper functions.  In addition,
some statements involving |ns3| trace sources do not have Python
bindings, due to limitations in the bindings.

Data Collection Helpers Overview
================================

In this section, we provide an overview of some helper classes that
have been created to ease the configuration of the data collection
framework for some common use cases.  The helpers allow users to form
common operations with only a few statements in their C++ or Python
programs.  But, this ease of use comes at the cost of significantly
less flexibility than low-level configuration can provide, and the
need to explicitly code support for new Probe types into the helpers
(to work around an issue described below). 

The emphasis on the current helpers is to marshal data out of |ns3|
trace sources into gnuplot plots or text files, without a high degree
of output customization or statistical processing (initially).  Also,
the use is constrained to the available probe types in |ns3|.  Later
sections of this documentation will go into more detail about creating
new Probe types, as well as details about hooking together Probes,
Collectors, and Aggregators in custom arrangements.

To date, two Data Collection helpers have been implemented:

- GnuplotHelper
- FileHelper

GnuplotHelper
=============

The GnuplotHelper is a helper class for producing output files used to
make gnuplots.  The overall goal is to provide the ability for users
to quickly make plots from data exported in |ns3| trace sources.  By
default, a minimal amount of data transformation is performed; the
objective is to generate plots with as few (default) configuration
statements as possible.

GnuplotHelper Overview
######################

The GnuplotHelper will create 3 different files at the end of the
simulation:

- A space separated gnuplot data file
- A gnuplot control file
- A shell script to generate the gnuplot

There are two configuration statements that are needed to produce plots.
The first statement configures the plot (filename, title, legends, and 
output type, where the output type defaults to PNG if unspecified):

::

   void ConfigurePlot (const std::string &outputFileNameWithoutExtension,
                       const std::string &title,
                       const std::string &xLegend,
                       const std::string &yLegend,
                       const std::string &terminalType = ".png");

The second statement hooks the trace source of interest:

::

   void PlotProbe (const std::string &typeId,
                   const std::string &path,
                   const std::string &probeTraceSource,
                   const std::string &title);

The arguments are as follows:

* typeId:  The |ns3| TypeId of the Probe
* path:  The path in the |ns3| configuration namespace to one or more trace sources 
* probeTraceSource:  Which output of the probe (itself a trace source) should be plotted
* title:  The title to associate with the dataset(s) (in the gnuplot legend)

A variant on the PlotProbe above is to specify a fifth optional argument
that controls where in the plot the key (legend) is placed.

A fully worked example (from ``seventh.cc``) is shown below:

::

   // Create the gnuplot helper.
   GnuplotHelper plotHelper;
 
   // Configure the plot.
   // Configure the plot.  The first argument is the file name prefix
   // for the output files generated.  The second, third, and fourth
   // arguments are, respectively, the plot title, x-axis, and y-axis labels
   plotHelper.ConfigurePlot ("seventh-packet-byte-count",
                             "Packet Byte Count vs. Time",
                             "Time (Seconds)",
                             "Packet Byte Count",
                             "png");

   // Specify the probe type, trace source path (in configuration namespace), and
   // probe output trace source ("OutputBytes") to plot.  The fourth argument
   // specifies the name of the data series label on the plot.  The last
   // argument formats the plot by specifying where the key should be placed.
   plotHelper.PlotProbe (probeType,
                         tracePath,
                         "OutputBytes",
                         "Packet Byte Count",
                         GnuplotAggregator::KEY_BELOW);

In this example, the ``probeType`` and ``tracePath`` are as follows (for IPv4):

::

   probeType = "ns3::Ipv4PacketProbe";
   tracePath = "/NodeList/*/$ns3::Ipv4L3Protocol/Tx";

The probeType is a key parameter for this helper to work.  This TypeId
must be registered in the system, and the signature on the Probe's trace
sink must match that of the trace source it is being hooked to.  Probe
types are pre-defined for a number of data types corresponding to |ns3|
traced values, and for a few other trace source signatures such as the 
'Tx' trace source of ``ns3::Ipv4L3Protocol`` class.

Note that the trace source path specified may contain wildcards.  
In this case, multiple
datasets are plotted on one plot; one for each matched path.

The main output produced will be three files:

::

   seventh-packet-byte-count.dat
   seventh-packet-byte-count.plt
   seventh-packet-byte-count.sh

At this point, users can either hand edit the .plt file for further
customizations, or just run it through gnuplot.  Running 
`sh seventh-packet-byte-count.sh` simply runs the plot through gnuplot,
as shown below. 

.. _seventh-packet-byte-count:

.. figure:: figures/seventh-packet-byte-count.png

  2-D Gnuplot Created by seventh.cc Example.

It can be seen that the key elements (legend, title, legend placement,
xlabel, ylabel, and path for the data) are all placed on the plot.
Since there were two matches to the configuration path provided, the
two data series are shown:

* Packet Byte Count-0 corresponds to /NodeList/0/$ns3::Ipv4L3Protocol/Tx
* Packet Byte Count-1 corresponds to /NodeList/1/$ns3::Ipv4L3Protocol/Tx

GnuplotHelper ConfigurePlot
###########################

The GnuplotHelper's ``ConfigurePlot()`` function can be used 
to configure plots.

It has the following prototype:

::

  void ConfigurePlot (const std::string &outputFileNameWithoutExtension,
                      const std::string &title,
                      const std::string &xLegend,
                      const std::string &yLegend,
                      const std::string &terminalType = ".png");

It has the following arguments:

  +--------------------------------+------------------------------+
  | Argument                       | Description                  |
  +================================+==============================+
  | outputFileNameWithoutExtension | Name of gnuplot related files|
  |                                | to write with no extension.  |
  +--------------------------------+------------------------------+
  | title                          | Plot title string to use for |
  |                                | this plot.                   |
  +--------------------------------+------------------------------+
  | xLegend                        | The legend for the x         |
  |                                | horizontal axis.             |
  +--------------------------------+------------------------------+
  | yLegend                        | The legend for the y         |
  |                                | vertical axis.               |
  +--------------------------------+------------------------------+
  | terminalType                   | Terminal type setting string |
  |                                | for output.  The default     |
  |                                | terminal type is "png".      |
  +--------------------------------+------------------------------+
  
The GnuplotHelper's ``ConfigurePlot()`` function configures plot 
related parameters for this gnuplot helper so
that it will create a space separated gnuplot data file named
outputFileNameWithoutExtension + ".dat", a gnuplot control file
named outputFileNameWithoutExtension + ".plt", and a shell script
to generate the gnuplot named outputFileNameWithoutExtension +
".sh".

An example of how to use this function can be seen in the 
``seventh.cc`` code described above where it was used as follows:

::

  plotHelper.ConfigurePlot ("seventh-packet-byte-count",
                            "Packet Byte Count vs. Time",
                            "Time (Seconds)",
                            "Packet Byte Count",
                            "png");

GnuplotHelper PlotProbe
#######################

The GnuplotHelper's ``PlotProbe()`` function can be used 
to plot values generated by probes.

It has the following prototype:

::

  void PlotProbe (const std::string &typeId,
                  const std::string &path,
                  const std::string &probeTraceSource,
                  const std::string &title,
                  enum GnuplotAggregator::KeyLocation keyLocation = GnuplotAggregator::KEY_INSIDE);

It has the following arguments:

  +------------------+------------------------------+
  | Argument         | Description                  |
  +==================+==============================+
  | typeId           | The type ID for the probe    |
  |                  | created by this helper.      |
  +------------------+------------------------------+
  | path             | Config path to access the    |
  |                  | trace source.                |
  +------------------+------------------------------+
  | probeTraceSource | The probe trace source to    |
  |                  | access.                      |
  +------------------+------------------------------+
  | title            | The title to be associated   |
  |                  | to this dataset              |
  +------------------+------------------------------+
  | keyLocation      | The location of the key in   |
  |                  | the plot.  The default       |
  |                  | location is inside.          |
  +------------------+------------------------------+
  
The GnuplotHelper's ``PlotProbe()`` function 
plots a dataset generated by hooking the |ns3| trace source with a
probe created by the helper, and then plotting the values from the 
probeTraceSource. 
The dataset will have the provided title, and will consist of 
the 'newValue' at each timestamp.

If the config path has more than one match in the system because 
there is a wildcard, then one dataset for each match will
be plotted.  The dataset titles will be suffixed with the matched
characters for each of the wildcards in the config path,
separated by spaces.  For example, if the proposed dataset title
is the string "bytes", and there are two wildcards in the path,
then dataset titles like "bytes-0 0" or "bytes-12 9" will be
possible as labels for the datasets that are plotted.

An example of how to use this function can be seen in the 
``seventh.cc`` code described above where it was used (with
variable substitution) as follows:

::

  plotHelper.PlotProbe ("ns3::Ipv4PacketProbe",
                        "/NodeList/*/$ns3::Ipv4L3Protocol/Tx",
                        "OutputBytes",
                        "Packet Byte Count",
                        GnuplotAggregator::KEY_BELOW);

Other Examples
##############

Gnuplot Helper Example
~~~~~~~~~~~~~~~~~~~~~~

A slightly simpler example than the ``seventh.cc`` example can be 
found in ``src/stats/examples/gnuplot-helper-example.cc``.  The 
following 2-D gnuplot was created using the example.

.. _gnuplot-helper-example:

.. figure:: figures/gnuplot-helper-example.png

  2-D Gnuplot Created by gnuplot-helper-example.cc Example.

In this example, there is an Emitter object that increments 
its counter according to a Poisson process and then emits the counter's 
value as a trace source.

::

  Ptr<Emitter> emitter = CreateObject<Emitter> ();
  Names::Add ("/Names/Emitter", emitter);

Note that because there are no wildcards in the path 
used below, only 1 datastream was drawn in the plot.  
This single datastream in the plot is simply labeled 
"Emitter Count", with no extra suffixes like one would 
see if there were wildcards in the path.

::

  // Create the gnuplot helper.
  GnuplotHelper plotHelper;

  // Configure the plot.
  plotHelper.ConfigurePlot ("gnuplot-helper-example",
                            "Emitter Counts vs. Time",
                            "Time (Seconds)",
                            "Emitter Count",
                            "png");

  // Plot the values generated by the probe.  The path that we provide
  // helps to disambiguate the source of the trace.
  plotHelper.PlotProbe ("ns3::Uinteger32Probe",
                        "/Names/Emitter/Counter",
                        "Output",
                        "Emitter Count",
                        GnuplotAggregator::KEY_INSIDE);

FileHelper
==========

The FileHelper is a helper class used to put data values into a file.
The overall goal is to provide the ability for users
to quickly make formatted text files from data exported in |ns3| 
trace sources.  By default, a minimal amount of data transformation is 
performed; the objective is to generate files with as few (default) 
configuration statements as possible.

FileHelper Overview
###################

The FileHelper will create 1 or more text files at the end of the
simulation.

The FileHelper can create 4 different types of text files:

- Formatted
- Space separated (the default)
- Comma separated
- Tab separated

Formatted files use C-style format strings and the sprintf() function
to print their values in the file being written.

The following text file with 2 columns of formatted values named 
``seventh-packet-byte-count-0.txt`` was created using more new 
code that was added to the original |ns3| Tutorial example's code.  
Only the first 10 lines of this file are shown here for brevity.

.. sourcecode:: text

  Time (Seconds) = 1.000e+00	Packet Byte Count = 40
  Time (Seconds) = 1.004e+00	Packet Byte Count = 40
  Time (Seconds) = 1.004e+00	Packet Byte Count = 576
  Time (Seconds) = 1.009e+00	Packet Byte Count = 576
  Time (Seconds) = 1.009e+00	Packet Byte Count = 576
  Time (Seconds) = 1.015e+00	Packet Byte Count = 512
  Time (Seconds) = 1.017e+00	Packet Byte Count = 576
  Time (Seconds) = 1.017e+00	Packet Byte Count = 544
  Time (Seconds) = 1.025e+00	Packet Byte Count = 576
  Time (Seconds) = 1.025e+00	Packet Byte Count = 544
  
  ...

The following different text file with 2 columns of formatted 
values named ``seventh-packet-byte-count-1.txt`` was also 
created using the same new code that was added to the original 
|ns3| Tutorial example's code.  Only the first 10 lines of this 
file are shown here for brevity.

.. sourcecode:: text

  Time (Seconds) = 1.002e+00	Packet Byte Count = 40
  Time (Seconds) = 1.007e+00	Packet Byte Count = 40
  Time (Seconds) = 1.013e+00	Packet Byte Count = 40
  Time (Seconds) = 1.020e+00	Packet Byte Count = 40
  Time (Seconds) = 1.028e+00	Packet Byte Count = 40
  Time (Seconds) = 1.036e+00	Packet Byte Count = 40
  Time (Seconds) = 1.045e+00	Packet Byte Count = 40
  Time (Seconds) = 1.053e+00	Packet Byte Count = 40
  Time (Seconds) = 1.061e+00	Packet Byte Count = 40
  Time (Seconds) = 1.069e+00	Packet Byte Count = 40
  
  ...

The new code that was added to produce the two text files is below.  
More details about this API will be covered in a later section. 

Note that because there were 2 matches for the wildcard in the path, 
2 separate text files were created.  The first text file, which is 
named "seventh-packet-byte-count-0.txt", corresponds to the 
wildcard match with the "*" replaced with "0".  The second text file, 
which is named "seventh-packet-byte-count-1.txt", corresponds to 
the wildcard match with the "*" replaced with "1".  Also, note that 
the function call to ``WriteProbe()`` will give an error message if 
there are no matches for a path that contains wildcards.

::

   // Create the file helper.
   FileHelper fileHelper;
 
   // Configure the file to be written.
   fileHelper.ConfigureFile ("seventh-packet-byte-count",
                             FileAggregator::FORMATTED);

   // Set the labels for this formatted output file.
   fileHelper.Set2dFormat ("Time (Seconds) = %.3e\tPacket Byte Count = %.0f");

   // Write the values generated by the probe.
   fileHelper.WriteProbe ("ns3::Ipv4PacketProbe",
                          "/NodeList/*/$ns3::Ipv4L3Protocol/Tx",
                          "OutputBytes");

FileHelper ConfigureFile
########################

The FileHelper's ``ConfigureFile()`` function can be used 
to configure text files.

It has the following prototype:

::

  void ConfigureFile (const std::string &outputFileNameWithoutExtension,
                      enum FileAggregator::FileType fileType = FileAggregator::SPACE_SEPARATED);

It has the following arguments:

  +--------------------------------+------------------------------+
  | Argument                       | Description                  |
  +================================+==============================+
  | outputFileNameWithoutExtension | Name of output file to write |
  |                                | with no extension.           |
  +--------------------------------+------------------------------+
  | fileType                       | Type of file to write.  The  | 
  |                                | default type of file is space| 
  |                                | separated.                   |
  +--------------------------------+------------------------------+
  
The FileHelper's ``ConfigureFile()`` function configures text file 
related parameters for the file helper so that
it will create a file named outputFileNameWithoutExtension plus
possible extra information from wildcard matches plus ".txt" with
values printed as specified by fileType.  The default file type
is space-separated.

An example of how to use this function can be seen in the 
``seventh.cc`` code described above where it was used as follows:

::

  fileHelper.ConfigureFile ("seventh-packet-byte-count",
                            FileAggregator::FORMATTED);

FileHelper WriteProbe
#####################

The FileHelper's ``WriteProbe()`` function can be used 
to write values generated by probes to text files.

It has the following prototype:

::

  void WriteProbe (const std::string &typeId,
                   const std::string &path,
                   const std::string &probeTraceSource);

It has the following arguments:

  +------------------+------------------------------+
  | Argument         | Description                  |
  +==================+==============================+
  | typeId           | The type ID for the probe    |
  |                  | to be created.               |
  +------------------+------------------------------+
  | path             | Config path to access the    |
  |                  | trace source.                |
  +------------------+------------------------------+
  | probeTraceSource | The probe trace source to    |
  |                  | access.                      |
  +------------------+------------------------------+
  
The FileHelper's ``WriteProbe()`` function
creates output text files generated by hooking the ns-3 trace source
with a probe created by the helper, and then writing the values from the
probeTraceSource. The output file names will have the text stored
in the member variable  m_outputFileNameWithoutExtension plus ".txt", 
and will consist of the 'newValue' at each timestamp.

If the config path has more than one match in the system because
there is a wildcard, then one output file for each match
will be created.  The output file names will contain the text in
m_outputFileNameWithoutExtension plus the matched characters for
each of the wildcards in the config path, separated by dashes,
plus ".txt".  For example, if the value in
m_outputFileNameWithoutExtension is the string
"packet-byte-count", and there are two wildcards in the path,
then output file names like "packet-byte-count-0-0.txt" or
"packet-byte-count-12-9.txt" will be possible as names for the
files that will be created.

An example of how to use this function can be seen in the 
``seventh.cc`` code described above where it was used as follows:

::

  fileHelper.WriteProbe ("ns3::Ipv4PacketProbe",
                         "/NodeList/*/$ns3::Ipv4L3Protocol/Tx",
                         "OutputBytes");

Other Examples
##############

File Helper Example
~~~~~~~~~~~~~~~~~~~

A slightly simpler example than the ``seventh.cc`` example can be 
found in ``src/stats/examples/file-helper-example.cc``.  
This example only uses the FileHelper.

The following text file with 2 columns of formatted values named 
``file-helper-example.txt`` was created using the example.  
Only the first 10 lines of this file are shown here for brevity.

.. sourcecode:: text

  Time (Seconds) = 0.203  Count = 1
  Time (Seconds) = 0.702  Count = 2
  Time (Seconds) = 1.404  Count = 3
  Time (Seconds) = 2.368  Count = 4
  Time (Seconds) = 3.364  Count = 5
  Time (Seconds) = 3.579  Count = 6
  Time (Seconds) = 5.873  Count = 7
  Time (Seconds) = 6.410  Count = 8
  Time (Seconds) = 6.472  Count = 9
  ...

In this example, there is an Emitter object that increments 
its counter according to a Poisson process and then emits the counter's 
value as a trace source.

::

  Ptr<Emitter> emitter = CreateObject<Emitter> ();
  Names::Add ("/Names/Emitter", emitter);

Note that because there are no wildcards in the path 
used below, only 1 text file was created.  
This single text file is simply named 
"file-helper-example.txt", with no extra suffixes like 
you would see if there were wildcards in the path.

::

  // Create the file helper.
  FileHelper fileHelper;

  // Configure the file to be written.
  fileHelper.ConfigureFile ("file-helper-example",
                            FileAggregator::FORMATTED);

  // Set the labels for this formatted output file.
  fileHelper.Set2dFormat ("Time (Seconds) = %.3e\tCount = %.0f");

  // Write the values generated by the probe.  The path that we
  // provide helps to disambiguate the source of the trace.
  fileHelper.WriteProbe ("ns3::Uinteger32Probe",
                         "/Names/Emitter/Counter",
                         "Output");

Scope and Limitations
=====================

Currently, only these Probes have been implemented and connected
to the GnuplotHelper and to the FileHelper:

- BooleanProbe
- DoubleProbe
- Uinteger8Probe
- Uinteger16Probe
- Uinteger32Probe
- TimeProbe
- PacketProbe
- ApplicationPacketProbe
- Ipv4PacketProbe

These Probes, therefore, are the only TypeIds available to be used 
in ``PlotProbe()`` and ``WriteProbe()``.

In the next few sections, we cover each of the fundamental object
types (Probe, Collector, and Aggregator) in more detail, and show
how they can be connected together using lower-level API.
