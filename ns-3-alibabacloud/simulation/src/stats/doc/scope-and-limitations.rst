.. include:: replace.txt

.. heading hierarchy:
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

Scope/Limitations
*****************

This section discusses the scope and limitations of the Data
Collection Framework.

Currently, only these Probes have been implemented in DCF:

- BooleanProbe
- DoubleProbe
- Uinteger8Probe
- Uinteger16Probe
- Uinteger32Probe
- TimeProbe
- PacketProbe
- ApplicationPacketProbe
- Ipv4PacketProbe

Currently, no Collectors are available in the DCF, although a 
BasicStatsCollector is under development.

Currently, only these Aggregators have been implemented in DCF:

- GnuplotAggregator
- FileAggregator

Currently, only this Adaptor has been implemented in DCF:

Time-Series Adaptor.

Future Work
###########

This section discusses the future work to be done on the Data
Collection Framework.

Here are some things that still need to be done:

- Hook up more trace sources in |ns3| code to get more values out of the simulator.
- Implement more types of Probes than there currently are.
- Implement more than just the single current 2-D Collector, BasicStatsCollector.
- Implement more Aggregators.
- Implement more than just Adaptors.

