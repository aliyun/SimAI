.. include:: replace.txt

3GPP HTTP applications
----------------------

Model Description
*****************

The model is a part of the applications library. The HTTP model is based on a commonly 
used 3GPP model in standardization `[4]`_. 

Design
======

This traffic generator simulates web browsing traffic using the Hypertext
Transfer Protocol (HTTP). It consists of one or more ``ThreeGppHttpClient``
applications which connect to a ``ThreeGppHttpServer`` application. The client
models a web browser which requests web pages to the server. The server
is then responsible to serve the web pages as requested. Please refer to
``ThreeGppHttpClientHelper`` and ``ThreeGppHttpServerHelper`` for usage instructions.

Technically speaking, the client transmits *request objects* to demand a
service from the server. Depending on the type of request received, the
server transmits either:

  - a *main object*, i.e., the HTML file of the web page; or
  - an *embedded object*, e.g., an image referenced by the HTML file.

The main and embedded object sizes are illustrated in figures :ref:`fig-http-main-object-size`
and :ref:`fig-http-embedded-object-size`.


.. _fig-http-main-object-size:

.. figure:: figures/http-main-object-size.*
   :figwidth: 15cm

   3GPP HTTP main object size histogram

.. _fig-http-embedded-object-size:

.. figure:: figures/http-embedded-object-size.*
   :figwidth: 15cm

   3GPP HTTP embedded object size histogram

\

A major portion of the traffic pattern is *reading time*, which does not
generate any traffic. Because of this, one may need to simulate a good
number of clients and/or sufficiently long simulation duration in order to
generate any significant traffic in the system. Reading time is illustrated in 
:ref:`fig-http-reading-time`.

.. _fig-http-reading-time:

.. figure:: figures/http-reading-time.*
   :figwidth: 15cm

   3GPP HTTP reading time histogram


3GPP HTTP server description
############################

3GPP HTTP server is a model application which simulates the traffic of a web server. This
application works in conjunction with ``ThreeGppHttpClient`` applications.

The application works by responding to requests. Each request is a small
packet of data which contains ``ThreeGppHttpHeader``. The value of the *content type*
field of the header determines the type of object that the client is
requesting. The possible type is either a *main object* or an *embedded object*.

The application is responsible to generate the right type of object and send
it back to the client. The size of each object to be sent is randomly
determined (see ``ThreeGppHttpVariables``). Each object may be sent as multiple packets
due to limited socket buffer space.

To assist with the transmission, the application maintains several instances
of ``ThreeGppHttpServerTxBuffer``. Each instance keeps track of the object type to be
served and the number of bytes left to be sent.

The application accepts connection request from clients. Every connection is
kept open until the client disconnects.

Maximum transmission unit (MTU) size is configurable in ``ThreeGppHttpServer`` or in 
``ThreeGppHttpVariables``. By default, the low variant is 536 bytes and high variant is 1460 bytes. 
The default values are set with the intention of having a TCP header (size of which is 40 bytes) added 
in the packet in such way that lower layers can avoid splitting packets. The change of MTU sizes 
affects all TCP sockets after the server application has started. It is mainly visible in sizes of 
packets received by ``ThreeGppHttpClient`` applications. 

3GPP HTTP client description
############################

3GPP HTTP client is a model application which simulates the traffic of a web browser. This
application works in conjunction with an ThreeGppHttpServer application.

In summary, the application works as follows.

1. Upon start, it opens a connection to the destination web server
   (ThreeGppHttpServer).
2. After the connection is established, the application immediately requests
   a *main object* from the server by sending a request packet.
3. After receiving a main object (which can take some time if it consists of
   several packets), the application "parses" the main object. Parsing time 
   is illustrated in figure :ref:`fig-http-parsing-time`.
4. The parsing takes a short time (randomly determined) to determine the
   number of *embedded objects* (also randomly determined) in the web page. 
   Number of embedded object is illustrated in :ref:`fig-http-num-of-embedded-objects`.
   
    * If at least one embedded object is determined, the application requests
       the first embedded object from the server. The request for the next
       embedded object follows after the previous embedded object has been
       completely received.
    * If there is no more embedded object to request, the application enters
       the *reading time*.
       
5. Reading time is a long delay (again, randomly determined) where the
   application does not induce any network traffic, thus simulating the user
   reading the downloaded web page.
6. After the reading time is finished, the process repeats to step #2.

.. _fig-http-parsing-time:

.. figure:: figures/http-parsing-time.*
   :figwidth: 15cm

   3GPP HTTP parsing time histogram

.. _fig-http-num-of-embedded-objects: 

.. figure:: figures/http-num-of-embedded-objects.*
   :figwidth: 15cm

   3GPP HTTP number of embedded objects histogram

The client models HTTP *persistent connection*, i.e., HTTP 1.1, where the
connection to the server is maintained and used for transmitting and receiving
all objects.

Each request by default has a constant size of 350 bytes. A ``ThreeGppHttpHeader``
is attached to each request packet. The header contains information
such as the content type requested (either main object or embedded object)
and the timestamp when the packet is transmitted (which will be used to
compute the delay and RTT of the packet).


References
==========

Many aspects of the traffic are randomly determined by ``ThreeGppHttpVariables``. 
A separate instance of this object is used by the HTTP server and client applications. 
These characteristics are based on a legacy 3GPP specification. The description
can be found in the following references:

\

.. _`[1]`:

[1] 3GPP TR 25.892, "Feasibility Study for Orthogonal Frequency Division Multiplexing (OFDM) for UTRAN enhancement"

\ 

.. _`[2]`:

[2] IEEE 802.16m, "Evaluation Methodology Document (EMD)", IEEE 802.16m-08/004r5, July 2008.

\

.. _`[3]`:

[3] NGMN Alliance, "NGMN Radio Access Performance Evaluation Methodology", v1.0, January 2008.

\

.. _`[4]`:

[4] 3GPP2-TSGC5, "HTTP, FTP and TCP models for 1xEV-DV simulations", 2001.

\

Usage
*****

The three-gpp-http-example can be referenced to see basic usage of the HTTP applications. 
In summary, using the ``ThreeGppHttpServerHelper`` and ``ThreeGppHttpClientHelper`` allow the 
user to easily install ``ThreeGppHttpServer`` and ``ThreeGppHttpClient`` applications to nodes.
The helper objects can be used to configure attribute values for the client
and server objects, but not for the ``ThreeGppHttpVariables`` object. Configuration of variables 
is done by modifying attributes of ``ThreeGppHttpVariables``, which should be done prior to helpers 
installing applications to nodes. 

The client and server provide a number of ns-3 trace sources such as
"Tx", "Rx", "RxDelay", and "StateTransition" on the server side, and a large
number on the client side ("ConnectionEstablished",
"ConnectionClosed","TxMainObjectRequest", "TxEmbeddedObjectRequest",
"RxMainObjectPacket", "RxMainObject", "RxEmbeddedObjectPacket",
"RxEmbeddedObject", "Rx", "RxDelay", "RxRtt", "StateTransition"). 


Building the 3GPP HTTP applications 
===================================

Building the applications does not require any special steps to be taken. It suffices to enable 
the applications module. 

Examples
========

For an example demonstrating HTTP applications
run::

  $ ./ns3 run 'three-gpp-http-example'

By default, the example will print out the web page requests of the client and responses of the 
server and client receiving content packets by using LOG_INFO of ``ThreeGppHttpServer`` and ``ThreeGppHttpClient``. 

Tests
=====

For testing HTTP applications, three-gpp-http-client-server-test is provided. Run::

  $ ./test.py -s three-gpp-http-client-server-test
  
The test consists of simple Internet nodes having HTTP server and client applications installed. 
Multiple variant scenarios are tested: delay is 3ms, 30ms or 300ms, bit error rate 0 or 5.0*10^(-6), 
MTU size 536 or 1460 bytes and either IPV4 or IPV6 is used. A simulation with each combination of 
these parameters is run multiple times to verify functionality with different random variables. 

Test cases themselves are rather simple: test verifies that HTTP object packet bytes sent match 
total bytes received by the client, and that ``ThreeGppHttpHeader`` matches the expected packet.



