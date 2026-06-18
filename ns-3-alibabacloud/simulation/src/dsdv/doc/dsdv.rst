.. include:: replace.txt

DSDV Routing
------------

Destination-Sequenced Distance Vector (DSDV) routing protocol is a pro-active, table-driven routing protocol
for MANETs developed by Charles E. Perkins and Pravin Bhagwat in 1994. It uses the hop count as metric in route
selection.

This model was developed by 
`the ResiliNets research group <http://www.ittc.ku.edu/resilinets>`_
at the University of Kansas.  A paper on this model exists at
`this URL <https://wiki.ittc.ku.edu/resilinets/ResiliNets_Publications#.E2.80.9CDestination-Sequenced_Distance_Vector_.28DSDV.29_Routing_Protocol_Implementation_in_ns-3.E2.80.9D>`_. 

DSDV Routing Overview
*********************

DSDV Routing Table: Every node will maintain a table listing all the other nodes it has known either directly
or through some neighbors. Every node has a single entry in the routing table. The entry will have information
about the node's IP address, last known sequence number and the hop count to reach that node. Along with these
details the table also keeps track of the nexthop neighbor to reach the destination node, the timestamp of the last
update received for that node.

The DSDV update message consists of three fields, Destination Address, Sequence Number and Hop Count.

Each node uses 2 mechanisms to send out the DSDV updates. They are,

1. Periodic Updates
    Periodic updates are sent out after every m_periodicUpdateInterval(default:15s). In this update the node broadcasts
    out its entire routing table.
2. Trigger Updates
    Trigger Updates are small updates in-between the periodic updates. These updates are sent out whenever a node
    receives a DSDV packet that caused a change in its routing table. The original paper did not clearly mention
    when for what change in the table should a DSDV update be sent out. The current implementation sends out an update
    irrespective of the change in the routing table.

The updates are accepted based on the metric for a particular node. The first factor determining the acceptance of
an update is the sequence number. It has to accept the update if the sequence number of the update message is higher
irrespective of the metric. If the update with same sequence number is received, then the update with least metric
(hopCount) is given precedence.

In highly mobile scenarios, there is a high chance of route fluctuations, thus we have the concept of weighted
settling time where an update with change in metric will not be advertised to neighbors. The node waits for
the settling time to make sure that it did not receive the update from its old neighbor before sending out
that update.

The current implementation covers all the above features of DSDV. The current implementation also has a request queue
to buffer packets that have no routes to destination. The default is set to buffer up to 5 packets per destination.

References
**********

Link to the Paper: http://portal.acm.org/citation.cfm?doid=190314.190336
