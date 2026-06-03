.. include:: replace.txt

.. heading hierarchy:
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)
   ~~~~~~~~~~~~~ Sub-paragraph (no number)

Adaptors
********

This section details the functionalities provided by the Adaptor
class to an |ns3| simulation. This section is meant for users
interested in developing simulations with the |ns3| tools and using
the Data Collection Framework, of which the Adaptor class is a
part, to generate data output with their simulation's results.

Note:  the term 'adaptor' may also be spelled 'adapter'; we chose
the spelling aligned with the C++ standard.

Adaptor Overview
================

An Adaptor is used to make connections between different types of DCF
objects. 

To date, one Adaptor has been implemented:

- TimeSeriesAdaptor

Time Series Adaptor
===================

The TimeSeriesAdaptor lets Probes connect directly to Aggregators
without needing any Collector in between.

Both of the implemented DCF helpers utilize TimeSeriesAdaptors
in order to take probed values of different types and output the
current time plus the value with both converted to doubles.

The role of the TimeSeriesAdaptor class is that of an adaptor, which
takes raw-valued probe data of different types and outputs a tuple of
two double values.  The first is a timestamp, which may be set to
different resolutions (e.g. Seconds, Milliseconds, etc.) in the future
but which is presently hardcoded to Seconds.
The second is the conversion of a non-double
value to a double value (possibly with loss of precision).

