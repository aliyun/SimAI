.. include:: replace.txt
.. highlight:: cpp


Tweaking
--------

.. _UsingLogging:

Using the Logging Module
************************

We have already taken a brief look at the |ns3| logging module while
going over the ``first.cc`` script.  We will now take a closer look and 
see what kind of use-cases the logging subsystem was designed to cover.

Logging Overview
++++++++++++++++
Many large systems support some kind of message logging facility, and 
|ns3| is not an exception.  In some cases, only error messages are 
logged to the "operator console" (which is typically ``stderr`` in Unix-
based systems).  In other systems, warning messages may be output as well as 
more detailed informational messages.  In some cases, logging facilities are 
used to output debug messages which can quickly turn the output into a blur.

|ns3| takes the view that all of these verbosity levels are useful 
and we provide a selectable, multi-level approach to message logging.  Logging
can be disabled completely, enabled on a component-by-component basis, or
enabled globally; and it provides selectable verbosity levels.  The 
|ns3| log module provides a straightforward, relatively easy to use
way to get useful information out of your simulation.

You should understand that we do provide a general purpose mechanism --- 
tracing --- to get data out of your models which should be preferred for 
simulation output (see the tutorial section Using the Tracing System for
more details on our tracing system).  Logging should be preferred for 
debugging information, warnings, error messages, or any time you want to 
easily get a quick message out of your scripts or models.

There are currently seven levels of log messages of increasing verbosity
defined in the system.  

* LOG_ERROR --- Log error messages (associated macro: NS_LOG_ERROR);
* LOG_WARN --- Log warning messages (associated macro: NS_LOG_WARN);
* LOG_DEBUG --- Log relatively rare, ad-hoc debugging messages
  (associated macro: NS_LOG_DEBUG);
* LOG_INFO --- Log informational messages about program progress
  (associated macro: NS_LOG_INFO);
* LOG_FUNCTION --- Log a message describing each function called
  (two associated macros: NS_LOG_FUNCTION, used for member functions,
  and NS_LOG_FUNCTION_NOARGS, used for static functions);
* LOG_LOGIC -- Log messages describing logical flow within a function
  (associated macro: NS_LOG_LOGIC);
* LOG_ALL --- Log everything mentioned above (no associated macro).

For each LOG_TYPE there is also LOG_LEVEL_TYPE that, if used, enables
logging of all the levels above it in addition to it's level.  (As a
consequence of this, LOG_ERROR and LOG_LEVEL_ERROR and also LOG_ALL
and LOG_LEVEL_ALL are functionally equivalent.)  For example,
enabling LOG_INFO will only enable messages provided by NS_LOG_INFO macro,
while enabling LOG_LEVEL_INFO will also enable messages provided by
NS_LOG_DEBUG, NS_LOG_WARN and NS_LOG_ERROR macros.  

We also provide an unconditional logging macro that is always displayed,
irrespective of logging levels or component selection.

* NS_LOG_UNCOND -- Log the associated message unconditionally (no associated
  log level).

Each level can be requested singly or cumulatively; and logging can be set 
up using a shell environment variable (NS_LOG) or by logging system function 
call.  As was seen earlier in the tutorial, the logging system has Doxygen 
documentation and now would be a good time to peruse the Logging Module 
documentation if you have not done so.

Now that you have read the documentation in great detail, let's use some of
that knowledge to get some interesting information out of the 
``scratch/myfirst.cc`` example script you have already built.

Enabling Logging
++++++++++++++++
Let's use the NS_LOG environment variable to turn on some more logging, but
first, just to get our bearings, go ahead and run the last script just as you 
did previously,

.. sourcecode:: bash

  $ ./ns3 run scratch/myfirst

You should see the now familiar output of the first |ns3| example
program

.. sourcecode:: bash

  $ Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.413s)
  Sent 1024 bytes to 10.1.1.2
  Received 1024 bytes from 10.1.1.1
  Received 1024 bytes from 10.1.1.2

It turns out that the "Sent" and "Received" messages you see above are
actually logging messages from the ``UdpEchoClientApplication`` and 
``UdpEchoServerApplication``.  We can ask the client application, for 
example, to print more information by setting its logging level via the 
NS_LOG environment variable.  

I am going to assume from here on that you are using an sh-like shell that uses 
the"VARIABLE=value" syntax.  If you are using a csh-like shell, then you 
will have to convert my examples to the "setenv VARIABLE value" syntax 
required by those shells.

Right now, the UDP echo client application is responding to the following line
of code in ``scratch/myfirst.cc``,

::

  LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);

This line of code enables the ``LOG_LEVEL_INFO`` level of logging.  When 
we pass a logging level flag, we are actually enabling the given level and
all lower levels.  In this case, we have enabled ``NS_LOG_INFO``,
``NS_LOG_DEBUG``, ``NS_LOG_WARN`` and ``NS_LOG_ERROR``.  We can
increase the logging level and get more information without changing the
script and recompiling by setting the NS_LOG environment variable like this:

.. sourcecode:: bash

  $ export NS_LOG=UdpEchoClientApplication=level_all

This sets the shell environment variable ``NS_LOG`` to the string,

.. sourcecode:: bash

  UdpEchoClientApplication=level_all

The left hand side of the assignment is the name of the logging component we
want to set, and the right hand side is the flag we want to use.  In this case,
we are going to turn on all of the debugging levels for the application.  If
you run the script with NS_LOG set this way, the |ns3| logging 
system will pick up the change and you should see the following output:

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.404s)
  UdpEchoClientApplication:UdpEchoClient()
  UdpEchoClientApplication:SetDataSize(1024)
  UdpEchoClientApplication:StartApplication()
  UdpEchoClientApplication:ScheduleTransmit()
  UdpEchoClientApplication:Send()
  Sent 1024 bytes to 10.1.1.2
  Received 1024 bytes from 10.1.1.1
  UdpEchoClientApplication:HandleRead(0x6241e0, 0x624a20)
  Received 1024 bytes from 10.1.1.2
  UdpEchoClientApplication:StopApplication()
  UdpEchoClientApplication:DoDispose()
  UdpEchoClientApplication:~UdpEchoClient()

The additional debug information provided by the application is from
the NS_LOG_FUNCTION level.  This shows every time a function in the application
is called during script execution.  Generally, use of (at least)
NS_LOG_FUNCTION(this) in member functions is preferred. Use
NS_LOG_FUNCTION_NOARGS() only in static functions.  Note, however, that there
are no requirements in the |ns3| system that models must support any particular
logging  functionality.  The decision regarding how much information is logged
is left to the individual model developer.  In the case of the echo
applications, a good deal of log output is available.

You can now see a log of the function calls that were made to the application.
If you look closely you will notice a single colon between the string 
``UdpEchoClientApplication`` and the method name where you might have 
expected a C++ scope operator (``::``).  This is intentional.  

The name is not actually a class name, it is a logging component name.  When 
there is a one-to-one correspondence between a source file and a class, this 
will generally be the class name but you should understand that it is not 
actually a class name, and there is a single colon there instead of a double
colon to remind you in a relatively subtle way to conceptually separate the 
logging component name from the class name.

It turns out that in some cases, it can be hard to determine which method
actually generates a log message.  If you look in the text above, you may
wonder where the string "``Received 1024 bytes from 10.1.1.2``" comes
from.  You can resolve this by OR'ing the ``prefix_func`` level into the
``NS_LOG`` environment variable.  Try doing the following,

.. sourcecode:: bash

  $ export 'NS_LOG=UdpEchoClientApplication=level_all|prefix_func'

Note that the quotes are required since the vertical bar we use to indicate an
OR operation is also a Unix pipe connector.

Now, if you run the script you will see that the logging system makes sure 
that every message from the given log component is prefixed with the component
name.

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.417s)
  UdpEchoClientApplication:UdpEchoClient()
  UdpEchoClientApplication:SetDataSize(1024)
  UdpEchoClientApplication:StartApplication()
  UdpEchoClientApplication:ScheduleTransmit()
  UdpEchoClientApplication:Send()
  UdpEchoClientApplication:Send(): Sent 1024 bytes to 10.1.1.2
  Received 1024 bytes from 10.1.1.1
  UdpEchoClientApplication:HandleRead(0x6241e0, 0x624a20)
  UdpEchoClientApplication:HandleRead(): Received 1024 bytes from 10.1.1.2
  UdpEchoClientApplication:StopApplication()
  UdpEchoClientApplication:DoDispose()
  UdpEchoClientApplication:~UdpEchoClient()

You can now see all of the messages coming from the UDP echo client application
are identified as such.  The message "Received 1024 bytes from 10.1.1.2" is
now clearly identified as coming from the echo client application.  The 
remaining message must be coming from the UDP echo server application.  We 
can enable that component by entering a colon separated list of components in
the NS_LOG environment variable.

.. sourcecode:: bash

  $ export 'NS_LOG=UdpEchoClientApplication=level_all|prefix_func:
                 UdpEchoServerApplication=level_all|prefix_func'

Warning:  You will need to remove the newline after the ``:`` in the
example text above which is only there for document formatting purposes.

Now, if you run the script you will see all of the log messages from both the
echo client and server applications.  You may see that this can be very useful
in debugging problems.

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.406s)
  UdpEchoServerApplication:UdpEchoServer()
  UdpEchoClientApplication:UdpEchoClient()
  UdpEchoClientApplication:SetDataSize(1024)
  UdpEchoServerApplication:StartApplication()
  UdpEchoClientApplication:StartApplication()
  UdpEchoClientApplication:ScheduleTransmit()
  UdpEchoClientApplication:Send()
  UdpEchoClientApplication:Send(): Sent 1024 bytes to 10.1.1.2
  UdpEchoServerApplication:HandleRead(): Received 1024 bytes from 10.1.1.1
  UdpEchoServerApplication:HandleRead(): Echoing packet
  UdpEchoClientApplication:HandleRead(0x624920, 0x625160)
  UdpEchoClientApplication:HandleRead(): Received 1024 bytes from 10.1.1.2
  UdpEchoServerApplication:StopApplication()
  UdpEchoClientApplication:StopApplication()
  UdpEchoClientApplication:DoDispose()
  UdpEchoServerApplication:DoDispose()
  UdpEchoClientApplication:~UdpEchoClient()
  UdpEchoServerApplication:~UdpEchoServer()

It is also sometimes useful to be able to see the simulation time at which a
log message is generated.  You can do this by ORing in the prefix_time bit.

.. sourcecode:: bash

  $ export 'NS_LOG=UdpEchoClientApplication=level_all|prefix_func|prefix_time:
                 UdpEchoServerApplication=level_all|prefix_func|prefix_time'

Again, you will have to remove the newline above.  If you run the script now,
you should see the following output:

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.418s)
  0s UdpEchoServerApplication:UdpEchoServer()
  0s UdpEchoClientApplication:UdpEchoClient()
  0s UdpEchoClientApplication:SetDataSize(1024)
  1s UdpEchoServerApplication:StartApplication()
  2s UdpEchoClientApplication:StartApplication()
  2s UdpEchoClientApplication:ScheduleTransmit()
  2s UdpEchoClientApplication:Send()
  2s UdpEchoClientApplication:Send(): Sent 1024 bytes to 10.1.1.2
  2.00369s UdpEchoServerApplication:HandleRead(): Received 1024 bytes from 10.1.1.1
  2.00369s UdpEchoServerApplication:HandleRead(): Echoing packet
  2.00737s UdpEchoClientApplication:HandleRead(0x624290, 0x624ad0)
  2.00737s UdpEchoClientApplication:HandleRead(): Received 1024 bytes from 10.1.1.2
  10s UdpEchoServerApplication:StopApplication()
  10s UdpEchoClientApplication:StopApplication()
  UdpEchoClientApplication:DoDispose()
  UdpEchoServerApplication:DoDispose()
  UdpEchoClientApplication:~UdpEchoClient()
  UdpEchoServerApplication:~UdpEchoServer()

You can see that the constructor for the UdpEchoServer was called at a 
simulation time of 0 seconds.  This is actually happening before the 
simulation starts, but the time is displayed as zero seconds.  The same is true
for the UdpEchoClient constructor message.

Recall that the ``scratch/first.cc`` script started the echo server 
application at one second into the simulation.  You can now see that the 
``StartApplication`` method of the server is, in fact, called at one second.
You can also see that the echo client application is started at a simulation 
time of two seconds as we requested in the script.

You can now follow the progress of the simulation from the 
``ScheduleTransmit`` call in the client that calls ``Send`` to the 
``HandleRead`` callback in the echo server application.  Note that the 
elapsed time for the packet to be sent across the point-to-point link is 3.69
milliseconds.  You see the echo server logging a message telling you that it
has echoed the packet and then, after another channel delay, you see the echo
client receive the echoed packet in its ``HandleRead`` method.

There is a lot that is happening under the covers in this simulation that you
are not seeing as well.  You can very easily follow the entire process by
turning on all of the logging components in the system.  Try setting the 
``NS_LOG`` variable to the following,

.. sourcecode:: bash

  $ export 'NS_LOG=*=level_all|prefix_func|prefix_time'

The asterisk above is the logging component wildcard.  This will turn on all 
of the logging in all of the components used in the simulation.  I won't 
reproduce the output here (as of this writing it produces 1265 lines of output
for the single packet echo) but you can redirect this information into a file 
and look through it with your favorite editor if you like,

.. sourcecode:: bash

  $ ./ns3 run scratch/myfirst > log.out 2>&1

I personally use this extremely verbose version of logging when I am presented 
with a problem and I have no idea where things are going wrong.  I can follow the 
progress of the code quite easily without having to set breakpoints and step 
through code in a debugger.  I can just edit up the output in my favorite editor
and search around for things I expect, and see things happening that I don't 
expect.  When I have a general idea about what is going wrong, I transition into
a debugger for a fine-grained examination of the problem.  This kind of output 
can be especially useful when your script does something completely unexpected.
If you are stepping using a debugger you may miss an unexpected excursion 
completely.  Logging the excursion makes it quickly visible.

Adding Logging to your Code
+++++++++++++++++++++++++++
You can add new logging to your simulations by making calls to the log 
component via several macros.  Let's do so in the ``myfirst.cc`` script we
have in the ``scratch`` directory.

Recall that we have defined a logging component in that script:

::

  NS_LOG_COMPONENT_DEFINE ("FirstScriptExample");

You now know that you can enable all of the logging for this component by
setting the ``NS_LOG`` environment variable to the various levels.  Let's
go ahead and add some logging to the script.  The macro used to add an 
informational level log message is ``NS_LOG_INFO``.  Go ahead and add one 
(just before we start creating the nodes) that tells you that the script is 
"Creating Topology."  This is done as in this code snippet,

Open ``scratch/myfirst.cc`` in your favorite editor and add the line,

::

  NS_LOG_INFO ("Creating Topology");

right before the lines,

::

  NodeContainer nodes;
  nodes.Create (2);

Now build the script using ns3 and clear the ``NS_LOG`` variable to turn
off the torrent of logging we previously enabled:

.. sourcecode:: bash

  $ ./ns3
  $ export NS_LOG=

Now, if you run the script, 

.. sourcecode:: bash

  $ ./ns3 run scratch/myfirst

you will ``not`` see your new message since its associated logging 
component (``FirstScriptExample``) has not been enabled.  In order to see your
message you will have to enable the ``FirstScriptExample`` logging component
with a level greater than or equal to ``NS_LOG_INFO``.  If you just want to 
see this particular level of logging, you can enable it by,

.. sourcecode:: bash

  $ export NS_LOG=FirstScriptExample=info

If you now run the script you will see your new "Creating Topology" log
message,

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.404s)
  Creating Topology
  Sent 1024 bytes to 10.1.1.2
  Received 1024 bytes from 10.1.1.1
  Received 1024 bytes from 10.1.1.2

Using Command Line Arguments
****************************

.. _Attribute:

Overriding Default Attributes
+++++++++++++++++++++++++++++
Another way you can change how |ns3| scripts behave without editing
and building is via *command line arguments.*  We provide a mechanism to 
parse command line arguments and automatically set local and global variables
based on those arguments.

The first step in using the command line argument system is to declare the
command line parser.  This is done quite simply (in your main program) as
in the following code,

::

  int
  main (int argc, char *argv[])
  {
    ...  

    CommandLine cmd;
    cmd.Parse (argc, argv);

    ...
  }

This simple two line snippet is actually very useful by itself.  It opens the
door to the |ns3| global variable and ``Attribute`` systems.  Go 
ahead and add that two lines of code to the ``scratch/myfirst.cc`` script at
the start of ``main``.  Go ahead and build the script and run it, but ask 
the script for help in the following way,

.. sourcecode:: bash

  $ ./ns3 run "scratch/myfirst --PrintHelp"

This will ask ns3 to run the ``scratch/myfirst`` script and pass the command
line argument ``--PrintHelp`` to the script.  The quotes are required to 
sort out which program gets which argument.  The command line parser will
now see the ``--PrintHelp`` argument and respond with,

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.413s)
  TcpL4Protocol:TcpStateMachine()
  CommandLine:HandleArgument(): Handle arg name=PrintHelp value=
  --PrintHelp: Print this help message.
  --PrintGroups: Print the list of groups.
  --PrintTypeIds: Print all TypeIds.
  --PrintGroup=[group]: Print all TypeIds of group.
  --PrintAttributes=[typeid]: Print all attributes of typeid.
  --PrintGlobals: Print the list of globals.

Let's focus on the ``--PrintAttributes`` option.  We have already hinted
at the |ns3| ``Attribute`` system while walking through the 
``first.cc`` script.  We looked at the following lines of code,

::

    PointToPointHelper pointToPoint;
    pointToPoint.SetDeviceAttribute ("DataRate", StringValue ("5Mbps"));
    pointToPoint.SetChannelAttribute ("Delay", StringValue ("2ms"));

and mentioned that ``DataRate`` was actually an ``Attribute`` of the 
``PointToPointNetDevice``.  Let's use the command line argument parser
to take a look at the ``Attributes`` of the PointToPointNetDevice.  The help
listing says that we should provide a ``TypeId``.  This corresponds to the
class name of the class to which the ``Attributes`` belong.  In this case it
will be ``ns3::PointToPointNetDevice``.  Let's go ahead and type in,

.. sourcecode:: bash

  $ ./ns3 run "scratch/myfirst --PrintAttributes=ns3::PointToPointNetDevice"

The system will print out all of the ``Attributes`` of this kind of net device.
Among the ``Attributes`` you will see listed is,

.. sourcecode:: bash

  --ns3::PointToPointNetDevice::DataRate=[32768bps]:
    The default data rate for point to point links

This is the default value that will be used when a ``PointToPointNetDevice``
is created in the system.  We overrode this default with the ``Attribute``
setting in the ``PointToPointHelper`` above.  Let's use the default values 
for the point-to-point devices and channels by deleting the 
``SetDeviceAttribute`` call and the ``SetChannelAttribute`` call from 
the ``myfirst.cc`` we have in the scratch directory.

Your script should now just declare the ``PointToPointHelper`` and not do 
any ``set`` operations as in the following example,

::

  ...

  NodeContainer nodes;
  nodes.Create (2);

  PointToPointHelper pointToPoint;

  NetDeviceContainer devices;
  devices = pointToPoint.Install (nodes);

  ...

Go ahead and build the new script with ns3 (``./ns3``) and let's go back
and enable some logging from the UDP echo server application and turn on the 
time prefix.

.. sourcecode:: bash

  $ export 'NS_LOG=UdpEchoServerApplication=level_all|prefix_time'

If you run the script, you should now see the following output,

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.405s)
  0s UdpEchoServerApplication:UdpEchoServer()
  1s UdpEchoServerApplication:StartApplication()
  Sent 1024 bytes to 10.1.1.2
  2.25732s Received 1024 bytes from 10.1.1.1
  2.25732s Echoing packet
  Received 1024 bytes from 10.1.1.2
  10s UdpEchoServerApplication:StopApplication()
  UdpEchoServerApplication:DoDispose()
  UdpEchoServerApplication:~UdpEchoServer()

Recall that the last time we looked at the simulation time at which the packet
was received by the echo server, it was at 2.00369 seconds.

.. sourcecode:: bash

  2.00369s UdpEchoServerApplication:HandleRead(): Received 1024 bytes from 10.1.1.1

Now it is receiving the packet at 2.25732 seconds.  This is because we just dropped
the data rate of the ``PointToPointNetDevice`` down to its default of 
32768 bits per second from five megabits per second.

If we were to provide a new ``DataRate`` using the command line, we could 
speed our simulation up again.  We do this in the following way, according to
the formula implied by the help item:

.. sourcecode:: bash

  $ ./ns3 run "scratch/myfirst --ns3::PointToPointNetDevice::DataRate=5Mbps"

This will set the default value of the ``DataRate`` ``Attribute`` back to 
five megabits per second.  Are you surprised by the result?  It turns out that
in order to get the original behavior of the script back, we will have to set 
the speed-of-light delay of the channel as well.  We can ask the command line 
system to print out the ``Attributes`` of the channel just like we did for
the net device:

.. sourcecode:: bash

  $ ./ns3 run "scratch/myfirst --PrintAttributes=ns3::PointToPointChannel"

We discover the ``Delay`` ``Attribute`` of the channel is set in the following
way:

.. sourcecode:: bash

  --ns3::PointToPointChannel::Delay=[0ns]:
    Transmission delay through the channel

We can then set both of these default values through the command line system,

.. sourcecode:: bash

  $ ./ns3 run "scratch/myfirst
    --ns3::PointToPointNetDevice::DataRate=5Mbps
    --ns3::PointToPointChannel::Delay=2ms"

in which case we recover the timing we had when we explicitly set the
``DataRate`` and ``Delay`` in the script:

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.417s)
  0s UdpEchoServerApplication:UdpEchoServer()
  1s UdpEchoServerApplication:StartApplication()
  Sent 1024 bytes to 10.1.1.2
  2.00369s Received 1024 bytes from 10.1.1.1
  2.00369s Echoing packet
  Received 1024 bytes from 10.1.1.2
  10s UdpEchoServerApplication:StopApplication()
  UdpEchoServerApplication:DoDispose()
  UdpEchoServerApplication:~UdpEchoServer()

Note that the packet is again received by the server at 2.00369 seconds.  We 
could actually set any of the ``Attributes`` used in the script in this way.
In particular we could set the ``UdpEchoClient Attribute MaxPackets`` 
to some other value than one.

How would you go about that?  Give it a try.  Remember you have to comment 
out the place we override the default ``Attribute`` and explicitly set 
``MaxPackets`` in the script.  Then you have to rebuild the script.  You 
will also have to find the syntax for actually setting the new default attribute
value using the command line help facility.  Once you have this figured out 
you should be able to control the number of packets echoed from the command 
line.  Since we're nice folks, we'll tell you that your command line should 
end up looking something like,

.. sourcecode:: bash

  $ ./ns3 run "scratch/myfirst
    --ns3::PointToPointNetDevice::DataRate=5Mbps 
    --ns3::PointToPointChannel::Delay=2ms 
    --ns3::UdpEchoClient::MaxPackets=2"

A natural question to arise at this point is how to learn about the existence
of all of these attributes.  Again, the command line help facility has
a feature for this.  If we ask for command line help we should see:

.. sourcecode:: bash

  $ ./ns3 run "scratch/myfirst --PrintHelp"
  myfirst [Program Arguments] [General Arguments]

  General Arguments:
    --PrintGlobals:              Print the list of globals.
    --PrintGroups:               Print the list of groups.
    --PrintGroup=[group]:        Print all TypeIds of group.
    --PrintTypeIds:              Print all TypeIds.
    --PrintAttributes=[typeid]:  Print all attributes of typeid.
    --PrintHelp:                 Print this help message.

If you select the "PrintGroups" argument, you should see a list of all
registered TypeId groups.  The group names are aligned with the module names
in the source directory (although with a leading capital letter).  Printing 
out all of the information at once would be too much, so a further filter 
is available to print information on a per-group basis.   So, focusing
again on the point-to-point module:

.. sourcecode:: bash

  ./ns3 run "scratch/myfirst --PrintGroup=PointToPoint"
  TypeIds in group PointToPoint:
    ns3::PointToPointChannel
    ns3::PointToPointNetDevice
    ns3::PointToPointRemoteChannel
    ns3::PppHeader

and from here, one can find the possible TypeId names to search for
attributes, such as in the ``--PrintAttributes=ns3::PointToPointChannel``
example shown above.

Another way to find out about attributes is through the ns-3 Doxygen; there
is a page that lists out all of the registered attributes in the simulator.

Hooking Your Own Values
+++++++++++++++++++++++
You can also add your own hooks to the command line system.  This is done
quite simply by using the ``AddValue`` method to the command line parser.

Let's use this facility to specify the number of packets to echo in a 
completely different way.  Let's add a local variable called ``nPackets``
to the ``main`` function.  We'll initialize it to one to match our previous 
default behavior.  To allow the command line parser to change this value, we
need to hook the value into the parser.  We do this by adding a call to 
``AddValue``.  Go ahead and change the ``scratch/myfirst.cc`` script to
start with the following code,

::

  int
  main (int argc, char *argv[])
  {
    uint32_t nPackets = 1;

    CommandLine cmd;
    cmd.AddValue("nPackets", "Number of packets to echo", nPackets);
    cmd.Parse (argc, argv);

    ...

Scroll down to the point in the script where we set the ``MaxPackets``
``Attribute`` and change it so that it is set to the variable ``nPackets``
instead of the constant ``1`` as is shown below.

::

  echoClient.SetAttribute ("MaxPackets", UintegerValue (nPackets));

Now if you run the script and provide the ``--PrintHelp`` argument, you 
should see your new ``User Argument`` listed in the help display.

Try,

.. sourcecode:: bash

  $ ./ns3 run "scratch/myfirst --PrintHelp"

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.403s)
  --PrintHelp: Print this help message.
  --PrintGroups: Print the list of groups.
  --PrintTypeIds: Print all TypeIds.
  --PrintGroup=[group]: Print all TypeIds of group.
  --PrintAttributes=[typeid]: Print all attributes of typeid.
  --PrintGlobals: Print the list of globals.
  User Arguments:
      --nPackets: Number of packets to echo

If you want to specify the number of packets to echo, you can now do so by
setting the ``--nPackets`` argument in the command line,

.. sourcecode:: bash

  $ ./ns3 run "scratch/myfirst --nPackets=2"

You should now see

.. sourcecode:: bash

  Waf: Entering directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  Waf: Leaving directory `/home/craigdo/repos/ns-3-allinone/ns-3-dev/build'
  'build' finished successfully (0.404s)
  0s UdpEchoServerApplication:UdpEchoServer()
  1s UdpEchoServerApplication:StartApplication()
  Sent 1024 bytes to 10.1.1.2
  2.25732s Received 1024 bytes from 10.1.1.1
  2.25732s Echoing packet
  Received 1024 bytes from 10.1.1.2
  Sent 1024 bytes to 10.1.1.2
  3.25732s Received 1024 bytes from 10.1.1.1
  3.25732s Echoing packet
  Received 1024 bytes from 10.1.1.2
  10s UdpEchoServerApplication:StopApplication()
  UdpEchoServerApplication:DoDispose()
  UdpEchoServerApplication:~UdpEchoServer()

You have now echoed two packets.  Pretty easy, isn't it?

You can see that if you are an |ns3| user, you can use the command 
line argument system to control global values and ``Attributes``.  If you are
a model author, you can add new ``Attributes`` to your ``Objects`` and 
they will automatically be available for setting by your users through the
command line system.  If you are a script author, you can add new variables to 
your scripts and hook them into the command line system quite painlessly.

.. _UsingTracingSystem:

Using the Tracing System
************************

The whole point of simulation is to generate output for further study, and 
the |ns3| tracing system is a primary mechanism for this.  Since 
|ns3| is a C++ program, standard facilities for generating output 
from C++ programs could be used:  

::

  #include <iostream>
  ...
  int main ()
  {
    ...
    std::cout << "The value of x is " << x << std::endl;
    ...
  } 

You could even use the logging module to add a little structure to your 
solution.  There are many well-known problems generated by such approaches
and so we have provided a generic event tracing subsystem to address the 
issues we thought were important.

The basic goals of the |ns3| tracing system are:

* For basic tasks, the tracing system should allow the user to generate 
  standard tracing for popular tracing sources, and to customize which objects
  generate the tracing;
* Intermediate users must be able to extend the tracing system to modify
  the output format generated, or to insert new tracing sources, without 
  modifying the core of the simulator;
* Advanced users can modify the simulator core to add new tracing sources
  and sinks.

The |ns3| tracing system is built on the concepts of independent 
tracing sources and tracing sinks, and a uniform mechanism for connecting
sources to sinks.  Trace sources are entities that can signal events that
happen in a simulation and provide access to interesting underlying data. 
For example, a trace source could indicate when a packet is received by a net
device and provide access to the packet contents for interested trace sinks.

Trace sources are not useful by themselves, they must be "connected" to
other pieces of code that actually do something useful with the information 
provided by the sink.  Trace sinks are consumers of the events and data
provided by the trace sources.  For example, one could create a trace sink 
that would (when connected to the trace source of the previous example) print 
out interesting parts of the received packet.

The rationale for this explicit division is to allow users to attach new
types of sinks to existing tracing sources, without requiring editing and 
recompilation of the core of the simulator.  Thus, in the example above, 
a user could define a new tracing sink in her script and attach it to an 
existing tracing source defined in the simulation core by editing only the 
user script.

In this tutorial, we will walk through some pre-defined sources and sinks and
show how they may be customized with little user effort.  See the ns-3 manual
or how-to sections for information on advanced tracing configuration including
extending the tracing namespace and creating new tracing sources.

ASCII Tracing
+++++++++++++
|ns3| provides helper functionality that wraps the low-level tracing
system to help you with the details involved in configuring some easily 
understood packet traces.  If you enable this functionality, you will see
output in a ASCII files --- thus the name.  For those familiar with 
|ns2| output, this type of trace is analogous to the ``out.tr``
generated by many scripts.

Let's just jump right in and add some ASCII tracing output to our 
``scratch/myfirst.cc`` script.  Right before the call to 
``Simulator::Run ()``, add the following lines of code:

::

  AsciiTraceHelper ascii;
  pointToPoint.EnableAsciiAll (ascii.CreateFileStream ("myfirst.tr"));

Like in many other |ns3| idioms, this code uses a  helper object to 
help create ASCII traces.  The second line contains two nested method calls.  
The "inside" method, ``CreateFileStream()`` uses an unnamed object idiom
to create a file stream object on the stack (without an object  name) and pass
it down to the called method.  We'll go into this more in the future, but all
you have to know at this point is that you are creating an object representing
a file named "myfirst.tr" and passing it into ``ns-3``.  You are telling 
``ns-3`` to deal with the lifetime issues of the created object and also to 
deal with problems caused by a little-known (intentional) limitation of C++ 
ofstream objects relating to copy constructors.

The outside call, to ``EnableAsciiAll()``, tells the helper that you 
want to enable ASCII tracing on all point-to-point devices in your simulation; 
and you want the (provided) trace sinks to write out information about packet 
movement in ASCII format.

For those familiar with |ns2|, the traced events are equivalent to 
the popular trace points that log "+", "-", "d", and "r" events.

You can now build the script and run it from the command line:

.. sourcecode:: bash

  $ ./ns3 run scratch/myfirst

Just as you have seen many times before, you will see some messages from ns3 and then
"'build' finished successfully" with some number of messages from 
the running program.  

When it ran, the program will have created a file named ``myfirst.tr``.  
Because of the way that ns3 works, the file is not created in the local
directory, it is created at the top-level directory of the repository by 
default.  If you want to control where the traces are saved you can use the 
``--cwd`` option of ns3 to specify this.  We have not done so, thus we
need to change into the top level directory of our repo and take a look at 
the ASCII trace file ``myfirst.tr`` in your favorite editor.

Parsing Ascii Traces
~~~~~~~~~~~~~~~~~~~~
There's a lot of information there in a pretty dense form, but the first thing
to notice is that there are a number of distinct lines in this file.  It may
be difficult to see this clearly unless you widen your window considerably.

Each line in the file corresponds to a *trace event*.  In this case
we are tracing events on the *transmit queue* present in every 
point-to-point net device in the simulation.  The transmit queue is a queue 
through which every packet destined for a point-to-point channel must pass.
Note that each line in the trace file begins with a lone character (has a 
space after it).  This character will have the following meaning:

* ``+``: An enqueue operation occurred on the device queue;
* ``-``: A dequeue operation occurred on the device queue;
* ``d``: A packet was dropped, typically because the queue was full;
* ``r``: A packet was received by the net device.

Let's take a more detailed view of the first line in the trace file.  I'll 
break it down into sections (indented for clarity) with a reference
number on the left side:

.. sourcecode:: text
  :linenos:

  + 
  2 
  /NodeList/0/DeviceList/0/$ns3::PointToPointNetDevice/TxQueue/Enqueue 
  ns3::PppHeader (
    Point-to-Point Protocol: IP (0x0021)) 
    ns3::Ipv4Header (
      tos 0x0 ttl 64 id 0 protocol 17 offset 0 flags [none] 
      length: 1052 10.1.1.1 > 10.1.1.2)
      ns3::UdpHeader (
        length: 1032 49153 > 9) 
        Payload (size=1024)

The first section of this expanded trace event (reference number 0) is the 
operation.  We have a ``+`` character, so this corresponds to an
*enqueue* operation on the transmit queue.  The second section (reference 1)
is the simulation time expressed in seconds.  You may recall that we asked the 
``UdpEchoClientApplication`` to start sending packets at two seconds.  Here
we see confirmation that this is, indeed, happening.

The next section of the example trace (reference 2) tell us which trace source
originated this event (expressed in the tracing namespace).  You can think
of the tracing namespace somewhat like you would a filesystem namespace.  The 
root of the namespace is the ``NodeList``.  This corresponds to a container
managed in the |ns3| core code that contains all of the nodes that are
created in a script.  Just as a filesystem may have directories under the 
root, we may have node numbers in the ``NodeList``.  The string 
``/NodeList/0`` therefore refers to the zeroth node in the ``NodeList``
which we typically think of as "node 0".  In each node there is a list of 
devices that have been installed.  This list appears next in the namespace.
You can see that this trace event comes from ``DeviceList/0`` which is the 
zeroth device installed in the node. 

The next string, ``$ns3::PointToPointNetDevice`` tells you what kind of 
device is in the zeroth position of the device list for node zero.
Recall that the operation ``+`` found at reference 00 meant that an enqueue 
operation happened on the transmit queue of the device.  This is reflected in 
the final segments of the "trace path" which are ``TxQueue/Enqueue``.

The remaining sections in the trace should be fairly intuitive.  References 3-4
indicate that the packet is encapsulated in the point-to-point protocol.  
References 5-7 show that the packet has an IP version four header and has
originated from IP address 10.1.1.1 and is destined for 10.1.1.2.  References
8-9 show that this packet has a UDP header and, finally, reference 10 shows
that the payload is the expected 1024 bytes.

The next line in the trace file shows the same packet being dequeued from the
transmit queue on the same node. 

The Third line in the trace file shows the packet being received by the net
device on the node with the echo server. I have reproduced that event below.

.. sourcecode:: text
  :linenos:

  r 
  2.25732 
  /NodeList/1/DeviceList/0/$ns3::PointToPointNetDevice/MacRx 
    ns3::Ipv4Header (
      tos 0x0 ttl 64 id 0 protocol 17 offset 0 flags [none]
      length: 1052 10.1.1.1 > 10.1.1.2)
      ns3::UdpHeader (
        length: 1032 49153 > 9) 
        Payload (size=1024)

Notice that the trace operation is now ``r`` and the simulation time has
increased to 2.25732 seconds.  If you have been following the tutorial steps
closely this means that you have left the ``DataRate`` of the net devices
and the channel ``Delay`` set to their default values.  This time should 
be familiar as you have seen it before in a previous section.

The trace source namespace entry (reference 02) has changed to reflect that
this event is coming from node 1 (``/NodeList/1``) and the packet reception
trace source (``/MacRx``).  It should be quite easy for you to follow the 
progress of the packet through the topology by looking at the rest of the 
traces in the file.

PCAP Tracing
++++++++++++
The |ns3| device helpers can also be used to create trace files in the
``.pcap`` format.  The acronym pcap (usually written in lower case) stands
for packet capture, and is actually an API that includes the 
definition of a ``.pcap`` file format.  The most popular program that can
read and display this format is Wireshark (formerly called Ethereal).
However, there are many traffic trace analyzers that use this packet format.
We encourage users to exploit the many tools available for analyzing pcap
traces.  In this tutorial, we concentrate on viewing pcap traces with tcpdump.

The code used to enable pcap tracing is a one-liner.  

::

  pointToPoint.EnablePcapAll ("myfirst");

Go ahead and insert this line of code after the ASCII tracing code we just 
added to ``scratch/myfirst.cc``.  Notice that we only passed the string
"myfirst," and not "myfirst.pcap" or something similar.  This is because the 
parameter is a prefix, not a complete file name.  The helper will actually 
create a trace file for every point-to-point device in the simulation.  The 
file names will be built using the prefix, the node number, the device number
and a ".pcap" suffix.

In our example script, we will eventually see files named "myfirst-0-0.pcap" 
and "myfirst-1-0.pcap" which are the pcap traces for node 0-device 0 and 
node 1-device 0, respectively.

Once you have added the line of code to enable pcap tracing, you can run the
script in the usual way:

.. sourcecode:: bash

  $ ./ns3 run scratch/myfirst

If you look at the top level directory of your distribution, you should now
see three log files:  ``myfirst.tr`` is the ASCII trace file we have 
previously examined.  ``myfirst-0-0.pcap`` and ``myfirst-1-0.pcap``
are the new pcap files we just generated.  

Reading output with tcpdump
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The easiest thing to do at this point will be to use ``tcpdump`` to look
at the ``pcap`` files.  

.. sourcecode:: bash

  $ tcpdump -nn -tt -r myfirst-0-0.pcap
  reading from file myfirst-0-0.pcap, link-type PPP (PPP)
  2.000000 IP 10.1.1.1.49153 > 10.1.1.2.9: UDP, length 1024
  2.514648 IP 10.1.1.2.9 > 10.1.1.1.49153: UDP, length 1024

  tcpdump -nn -tt -r myfirst-1-0.pcap
  reading from file myfirst-1-0.pcap, link-type PPP (PPP)
  2.257324 IP 10.1.1.1.49153 > 10.1.1.2.9: UDP, length 1024
  2.257324 IP 10.1.1.2.9 > 10.1.1.1.49153: UDP, length 1024

You can see in the dump of ``myfirst-0-0.pcap`` (the client device) that the 
echo packet is sent at 2 seconds into the simulation.  If you look at the
second dump (``myfirst-1-0.pcap``) you can see that packet being received
at 2.257324 seconds.  You see the packet being echoed back at 2.257324 seconds
in the second dump, and finally, you see the packet being received back at 
the client in the first dump at 2.514648 seconds.

Reading output with Wireshark
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you are unfamiliar with Wireshark, there is a web site available from which
you can download programs and documentation:  http://www.wireshark.org/.

Wireshark is a graphical user interface which can be used for displaying these
trace files.  If you have Wireshark available, you can open each of the trace
files and display the contents as if you had captured the packets using a
*packet sniffer*.
