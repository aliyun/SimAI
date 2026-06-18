.. include:: replace.txt
.. highlight:: bash

.. Section Separators:
   ----
   ****
   ++++
   ====
   ~~~~

.. _Utilities:

Utilities
---------

Print-introspected-doxygen
**************************

`print-introspected-doxygen` is used to generate doxygen documentation
using various TypeIds defined throughout the |ns3| source code.
The tool returns the various config paths, attributes, trace sources,
etc. for the various files in |ns3|.

Invocation
++++++++++

This tool is run automatically by the build system when generating
the Doxygen API docs, so you don't normally have to run it by hand.

However, since it does give a fair bit of information about TypeIds
it can be useful to run from the command line and
search for specific information.

To run it, simply open terminal and type

.. sourcecode:: bash

    $ ./ns3 run print-introspected-doxygen

This will give all the output, formatted for Doxygen, which can be viewed
in a text editor.

One way to use this is to capture it to a file::

    $ ./ns3 run print-introspected-doxygen > doc.html

Some users might prefer to use tools like grep
to locate the required piece of information from the documentation
instead of using an editor. For such uses-cases and more,
`print-introspected-doxygen` can return plain text::

    $ ./ns3 run "print-introspected-doxygen --output-text"

(Note the quotes around the inner command and options.)

    $ ./ns3 run "print-introspected-doxygen --output-text" | grep "hello"

This will output the following::

    * HelloInterval: HELLO messages emission interval.
    * DeletePeriod: DeletePeriod is intended to provide an upper bound on the time for which an upstream node A can have a neighbor B as an active next hop for destination D, while B has invalidated the route to D. = 5 * max (HelloInterval, ActiveRouteTimeout)
    * AllowedHelloLoss: Number of hello messages which may be loss for valid link.
    * EnableHello: Indicates whether a hello messages enable.
    * HelloInterval: HELLO messages emission interval.
    * HelloInterval: HELLO messages emission interval.
    * DeletePeriod: DeletePeriod is intended to provide an upper bound on the time for which an upstream node A can have a neighbor B as an active next hop for destination D, while B has invalidated the route to D. = 5 * max (HelloInterval, ActiveRouteTimeout)
    * AllowedHelloLoss: Number of hello messages which may be loss for valid link.
    * EnableHello: Indicates whether a hello messages enable.
    * HelloInterval: HELLO messages emission interval.


    
Bench-simulator
***************

This tool is used to benchmark the scheduler algorithms used in |ns3|.

Command-line Arguments
++++++++++++++++++++++

.. sourcecode:: bash

    $ ./ns3 run "bench-simulator --help"

    Program Options:
	--cal:    use CalendarSheduler [false]
	--heap:   use HeapScheduler [false]
	--list:   use ListSheduler [false]
	--map:    use MapScheduler (default) [true]
	--debug:  enable debugging output [false]
	--pop:    event population size (default 1E5) [100000]
	--total:  total number of events to run (default 1E6) [1000000]
	--runs:   number of runs (default 1) [1]
	--file:   file of relative event times []
	--prec:   printed output precision [6]

You can change the Scheduler being benchmarked by passing
the appropriate flags, for example if you want to 
benchmark the CalendarScheduler pass `--cal` to the program.

The default total number of events, runs or population size
can be overridden by passing `--total=value`, `--runs=value`  
and `--pop=value` respectively. 

If you want to use event distribution which is stored in a file,
you can pass the file option by `--file=FILE_NAME`. 

`--prec` can be used to change the output precision value and
`--debug` as the name suggests enables debugging. 

Invocation
++++++++++

To run it, simply open the terminal and type

.. sourcecode:: bash

    $ ./ns3 run bench-simulator
    
It will show something like this depending upon the scheduler being benchmarked::

    ns3-dev-bench-simulator-debug:
    ns3-dev-bench-simulator-debug: scheduler: ns3::MapScheduler
    ns3-dev-bench-simulator-debug: population: 100000
    ns3-dev-bench-simulator-debug: total events: 1000000
    ns3-dev-bench-simulator-debug: runs: 1
    ns3-dev-bench-simulator-debug: using default exponential distribution

    Run        Inititialization:                   Simulation:
		Time (s)    Rate (ev/s) Per (s/ev)  Time (s)    Rate (ev/s) Per (s/ev)
    ----------- ----------- ----------- ----------- ----------- ----------- -----------
    (prime)     0.4         250000      4e-06       1.84        543478      1.84e-06
    0           0.15        666667      1.5e-06     1.86        537634      1.86e-06


Suppose we had to benchmark `CalendarScheduler` instead, we would have written

.. sourcecode:: bash

    $ ./ns3 run "bench-simulator --cal"
 
And the output would look something like this::

    ns3-dev-bench-simulator-debug:
    ns3-dev-bench-simulator-debug: scheduler: ns3::CalendarScheduler
    ns3-dev-bench-simulator-debug: population: 100000
    ns3-dev-bench-simulator-debug: total events: 1000000
    ns3-dev-bench-simulator-debug: runs: 1
    ns3-dev-bench-simulator-debug: using default exponential distribution

    Run        Inititialization:                   Simulation:
		Time (s)    Rate (ev/s) Per (s/ev)  Time (s)    Rate (ev/s) Per (s/ev)
    ----------- ----------- ----------- ----------- ----------- ----------- -----------
    (prime)     1.19        84033.6     1.19e-05    32.03       31220.7     3.203e-05
    0           0.99        101010      9.9e-06     31.22       32030.7     3.122e-05
    ```
