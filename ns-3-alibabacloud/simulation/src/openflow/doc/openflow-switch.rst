.. include:: replace.txt
.. highlight:: bash

OpenFlow switch support
-----------------------

ns-3 simulations can use OpenFlow switches (McKeown et al. [1]_), 
widely used in research.  OpenFlow switches are configurable via the 
OpenFlow API, and also have an MPLS extension for quality-of-service and 
service-level-agreement support. By extending these capabilities to ns-3 
for a simulated OpenFlow switch that is both configurable and can use 
the MPLS extension, ns-3 simulations can accurately simulate many different 
switches.


The OpenFlow software implementation distribution is hereby referred to as 
the OFSID. This is a demonstration of running OpenFlow in software that 
the OpenFlow research group has made available. There is also an OFSID 
that Ericsson researchers created to add MPLS capabilities; this is the 
OFSID currently used with ns-3. The design will allow the users to, 
with minimal effort, switch in a different OFSID that may include more 
efficient code than a previous OFSID.


Model Description
*****************

The model relies on building an external OpenFlow switch library (OFSID), 
and then building some ns-3 wrappers that call out to the library.  
The source code for the ns-3 wrappers lives in the directory 
``src/openflow/model``.

Design
======

The OpenFlow module presents a OpenFlowSwitchNetDevice and a OpenFlowSwitchHelper for 
installing it on nodes. Like the Bridge module, it takes a collection of 
NetDevices to set up as ports, and it acts as the intermediary between 
them, receiving a packet on one port and forwarding it on another, or all 
but the received port when flooding. Like an OpenFlow switch, it maintains 
a configurable flow table that can match packets by their headers and do 
different actions with the packet based on how it matches. The module's 
understanding of OpenFlow configuration messages are kept the same format 
as a real OpenFlow-compatible switch, so users testing Controllers via 
ns-3 won't have to rewrite their Controller to work on real 
OpenFlow-compatible switches. 

The ns-3 OpenFlow switch device models an OpenFlow-enabled switch. It is designed to
express basic use of the OpenFlow protocol, with the maintaining of a virtual
Flow Table and TCAM to provide OpenFlow-like results.

The functionality comes down to the Controllers, which send messages to the
switch that configure its flows, producing different effects. Controllers can
be added by the user, under the ofi namespace extending ofi::Controller. To 
demonstrate this, a DropController, which creates flows for ignoring every single
packet, and LearningController, which effectively makes the switch a more complicated
BridgeNetDevice. A user versed in a standard OFSID, and/or OF protocol, can write
virtual controllers to create switches of all kinds of types.

OpenFlow switch Model
#####################

The OpenFlow switch device behaves somewhat according to the diagram setup as a classical OFSID
switch, with a few modifications made for a proper simulation environment.

Normal OF-enabled Switch:

.. sourcecode:: text
  
  | Secure Channel                  | <--OF Protocol--> | Controller is external |
  | Hardware or Software Flow Table |
 

ns-3 OF-enabled Switch (module):

.. sourcecode:: text
  
  | m_controller->ReceiveFromSwitch() | <--OF Protocol--> | Controller is internal |
  | Software Flow Table, virtual TCAM |


In essence, there are two differences: 

1) No SSL, Embedded Controller: Instead of a secure channel and connecting to an
outside location for the Controller program/machine, we currently only allow a
Controller extended from ofi::Controller, an extension of an ns3::Object. This
means ns-3 programmers cannot model the SSL part of the interface or possibility
of network failure. The connection to the OpenFlowSwitch is local and there aren't any
reasons for the channel/connection to break down. <<This difference may be an
option in the future. Using EmuNetDevices, it should be possible to engage an
external Controller program/machine, and thus work with controllers designed
outside of the ns-3 environment, that simply use the proper OF protocol when
communicating messages to the switch through a tap device.>>

2) Virtual Flow Table, TCAM: Typical OF-enabled switches are implemented on a hardware
TCAM. The OFSID we turn into a library includes a modelled software TCAM, that produces
the same results as a hardware TCAM. We include an attribute FlowTableLookupDelay, which
allows a simple delay of using the TCAM to be modelled. We don't endeavor to make this
delay more complicated, based on the tasks we are running on the TCAM, that is a possible
future improvement.

The OpenFlowSwitch network device is aimed to model an OpenFlow switch, with a TCAM and a connection
to a controller program. With some tweaking, it can model every switch type, per OpenFlow's
extensibility. It outsources the complexity of the switch ports to NetDevices of the user's choosing.
It should be noted that these NetDevices must behave like practical switch ports, i.e. a Mac Address
is assigned, and nothing more. It also must support a SendFrom function so that
the OpenFlowSwitch can forward across that port.

Scope and Limitations
=====================

All MPLS capabilities are implemented on the OFSID side in the OpenFlowSwitchNetDevice,
but ns-3-mpls hasn't been integrated, so ns-3 has no way to pass in
proper MPLS packets to the OpenFlowSwitch. If it did, one would only need to make
BufferFromPacket pick up the MplsLabelStack or whatever the MPLS header 
is called on the Packet, and build the MPLS header into the ofpbuf.

Future Work
===========


References
==========

.. [1] McKeown, N.; Anderson, T.; Balakrishan, H.; Parulkar, G.; Peterson, L.; Rexford, J.; Shenker, S.; Turner, J.; OpenFlow: enabling innovation in campus networks, ACM SIGCOMM Computer Communication Review, Vol. 38, Issue 2, April 2008.

Usage
*****

The OFSID requires libxml2 (for MPLS FIB xml file parsing), and libdl (for address fault checking).

Building OFSID
==============

In order to use the OpenFlowSwitch module, you must create and link the
OFSID (OpenFlow Software Implementation Distribution) to ns-3.
To do this:

1. Obtain the OFSID code.
   An ns-3 specific OFSID branch is provided to ensure
   operation with ns-3. Use mercurial to download this branch and ns3 to build
   the library::

     $ hg clone http://code.nsnam.org/openflow
     $ cd openflow

   From the "openflow" directory, run::
     
     $ ./waf configure
     $ ./waf build

2. Your OFSID is now built into a libopenflow.a library!
   To link to an ns-3 build with this OpenFlow switch module, run from the ns-3-dev
   (or whatever you have named your distribution)::

     $ ./ns3 configure --enable-examples --enable-tests --with-openflow=path/to/openflow

3. Under ``---- Summary of optional NS-3 features:`` you should see:

   .. sourcecode:: text
  
     "NS-3 OpenFlow Integration     : enabled"

   indicating the library has been linked to ns-3. Run::

     $ ./ns3 build

to build ns-3 and activate the OpenFlowSwitch module in ns-3.

Examples
========
For an example demonstrating its use in a simple learning controller/switch, 
run::

  $ ./ns3 run openflow-switch

To see it in detailed logging, run::

  $ ./ns3 run "openflow-switch -v"


Helpers
=======

Attributes
==========

The SwitchNetDevice provides following Attributes:

- FlowTableLookUpDelay:      This time gets run off the clock when making a lookup in our Flow Table.
- Flags:                     OpenFlow specific configuration flags. They are defined in the ofp_config_flags enum. Choices include:
                             OFPC_SEND_FLOW_EXP (Switch notifies controller when a flow has expired),
                             OFPC_FRAG_NORMAL (Match fragment against Flow table),
                             OFPC_FRAG_DROP (Drop fragments),
                             OFPC_FRAG_REASM (Reassemble only if OFPC_IP_REASM set, which is currently impossible,
                             because switch implementation does not support IP reassembly)
                             OFPC_FRAG_MASK (Mask Fragments)
- FlowTableMissSendLength:   When the packet doesn't match in our Flow Table, and we forward to the controller,
                             this sets # of bytes forwarded (packet is not forwarded in its entirety, unless specified).

.. note::

    TODO

Tracing
=======

.. note::

    TODO

Logging
=======

.. note::

    TODO

Caveats
=======

.. note::

    TODO

Validation
**********

This model has one test suite which can be run as follows::

  $ ./test.py --suite=openflow

