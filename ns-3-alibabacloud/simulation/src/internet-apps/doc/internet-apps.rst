Internet Applications Module Documentation
------------------------------------------

.. include:: replace.txt
.. highlight:: cpp

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

The goal of this module is to hold all the Internet-specific applications,
and most notably some very specific applications (e.g., ping) or daemons (e.g., radvd).  Other non-Internet-specific applications such as packet generators
are contained in other modules.

The source code for the new module lives in the directory ``src/internet-apps``.

Each application has its own goals, limitations and scope, which are briefly explained
in the following.

All the applications are extensively used in the top-level ``examples`` 
directories. The users are encouraged to check the scripts therein to have a 
clear overview of the various options and usage tricks.


V4Ping
******

This app mimics a "ping" (ICMP Echo) using IPv4. The application allows the 
following attributes to be set:

* Remote address
* Verbose mode
* Packet size (default 56 bytes)
* Packet interval  (default 1 second)

Moreover, the user can access the measured RTT value (as a Traced Source).

Ping6
*****

This app mimics a "ping" (ICMP Echo) using IPv6. The application allows the 
following attributes to be set:

* Remote address
* Local address (sender address)
* Packet size (default 56 bytes)
* Packet interval  (default 1 second)
* Max number of packets to send

Radvd
*****

This app mimics a "RADVD" daemon. I.e., the daemon responsible for IPv6 routers 
advertisements. All the IPv6 routers should have a RADVD daemon installed.

The configuration of the Radvd application mimics the one of the radvd Linux program.

DHCPv4
******

The |ns3| implementation of Dynamic Host Configuration Protocol (DHCP)
follows the specifications of :rfc:`2131` and :rfc:`2132`.

The source code for DHCP is located in ``src/internet-apps/model`` and consists of the 
following 6 files:

* dhcp-server.h,
* dhcp-server.cc,
* dhcp-client.h,
* dhcp-client.cc,
* dhcp-header.h and
* dhcp-header.cc

Helpers
=======

The following two files have been added to ``src/internet-apps/helper`` for DHCP: 

* dhcp-helper.h and 
* dhcp-helper.cc

Tests
=====
The tests for DHCP can be found at ``src/internet-apps/test/dhcp-test.cc``

Examples
========
The examples for DHCP can be found at ``src/internet-apps/examples/dhcp-example.cc``


Scope and Limitations
=====================

The server should be provided with a network address, mask and a range of address
for the pool. One client application can be installed on only one netdevice in a
node, and can configure address for only that netdevice.

The following five basic DHCP messages are supported: 

* DHCP DISCOVER
* DHCP OFFER
* DHCP REQUEST
* DHCP ACK
* DHCP NACK

Also, the following eight options of BootP are supported:

* 1 (Mask)
* 50 (Requested Address)
* 51 (Address Lease Time)
* 53 (DHCP message type)
* 54 (DHCP server identifier)
* 58 (Address renew time)
* 59 (Address rebind time)
* 255 (end)

The client identifier option (61) can be implemented in near future.

In the current implementation, a DHCP client can obtain IPv4 address dynamically 
from the DHCP server, and can renew it within a lease time period.

Multiple DHCP servers can be configured, but the implementation does not support
the use of a DHCP Relay yet.

