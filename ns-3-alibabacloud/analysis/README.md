# Analysis
This folder includes code and scripts for analysis.

## FCT analysis
`fct_analysis.py` is used to analyze fct. It reads multiple fct files (simulation's output), and prints data that can produce figures like Figure 11 (a) and (c) in [HPCC paper](https://liyuliang001.github.io/publications/hpcc.pdf).

Usage: please check `python fct_analysis.py -h` and read line 20-26 in `fct_analysis.py`

## Trace reader
`trace_reader` is used to parse the .tr files output by the simulation.

### Usage: 
1. `make trace_reader`

2. `./trace_reader <.tr file> [filter_expr]`. The filter_expr is used to filter events. For example, `time > 2000010000` will display only events after 2000010000, `sip=0x0b000101&dip=0x0b000201` will display only events with sip=0x0b000101 and dip=0x0b000201. Feel free to play with it (we may come up with more detailed descriptions in the future. For now, please read trace_filter.hpp for more details).

### Output:
Each line is like:

`2000055540 n:338 4:3 100608 Enqu ecn:0 0b00d101 0b012301 10000 100 U 161000 0 3 1048(1000)`

It means: at time 2000055540ns, at node 338, port 4, queue #3, the queue length is 100608B, and a packet is enqueued; the packet does not have ECN marked, is from 11.0.209.1:10000 to 11.1.35.1:100, is a data packet (U), sequence number 161000, tx timestamp 0, priority group 3, packet size 1048B, payload 1000B.

There are other types of packets. Please refer to print_trace() in utils.hpp for details.
