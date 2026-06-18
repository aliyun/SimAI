.. include:: replace.txt
.. highlight:: cpp
.. highlight:: bash

PageBreak

Low-Rate Wireless Personal Area Network (LR-WPAN)
-------------------------------------------------

This chapter describes the implementation of |ns3| models for the
low-rate, wireless personal area network (LR-WPAN) as specified by
IEEE standard 802.15.4 (2006).

Model Description
*****************

The source code for the lr-wpan module lives in the directory ``src/lr-wpan``.

Design
======

The model design closely follows the standard from an architectural standpoint.

.. _fig-lr-wpan-arch:

.. figure:: figures/lr-wpan-arch.*

    Architecture and scope of lr-wpan models

The grey areas in the figure (adapted from Fig 3. of IEEE Std. 802.15.4-2006)
show the scope of the model.

The Spectrum NetDevice from Nicola Baldo is the basis for the implementation.

The implementation also plans to borrow from the ns-2 models developed by 
Zheng and Lee in the future.

APIs
####

The APIs closely follow the standard, adapted for |ns3| naming conventions
and idioms.  The APIs are organized around the concept of service primitives
as shown in the following figure adapted from Figure 14 of 
IEEE Std. 802.15.4-2006.

.. _fig-lr-wpan-primitives:

.. figure:: figures/lr-wpan-primitives.*

    Service primitives

The APIs are organized around four conceptual services and service access
points (SAP):

* MAC data service (MCPS)
* MAC management service  (MLME)
* PHY data service (PD)
* PHY management service (PLME)

In general, primitives are standardized as follows (e.g. Sec 7.1.1.1.1
of IEEE 802.15.4-2006):::

  MCPS-DATA.request      (
                          SrcAddrMode,
                          DstAddrMode,
                          DstPANId,
                          DstAddr,
                          msduLength,
                          msdu,
                          msduHandle,
                          TxOptions,
                          SecurityLevel,
                          KeyIdMode,
                          KeySource,
                          KeyIndex
                          )

This maps to |ns3| classes and methods such as:::

  struct McpsDataRequestParameters
  {
    uint8_t m_srcAddrMode;
    uint8_t m_dstAddrMode;
    ...
  };

  void
  LrWpanMac::McpsDataRequest (McpsDataRequestParameters params)
  {
  ...
  }

MAC
###

The MAC at present implements the unslotted CSMA/CA variant, without beaconing.
Currently there is no support for coordinators and the relevant APIs.

The implemented MAC is similar to Contiki's NullMAC, i.e., a MAC without sleep
features. The radio is assumed to be always active (receiving or transmitting),
of completely shut down. Frame reception is not disabled while performing the
CCA.

The main API supported is the data transfer API
(McpsDataRequest/Indication/Confirm).  CSMA/CA according to Stc 802.15.4-2006,
section 7.5.1.4 is supported. Frame reception and rejection according to
Std 802.15.4-2006, section 7.5.6.2 is supported, including acknowledgements.
Only short addressing completely implemented. Various trace sources are
supported, and trace sources can be hooked to sinks.

PHY
###

The physical layer components consist of a Phy model, an error rate model, 
and a loss model.  The error rate model presently models the error rate 
for IEEE 802.15.4 2.4 GHz AWGN channel for OQPSK; the model description can 
be found in IEEE Std 802.15.4-2006, section E.4.1.7.   The Phy model is 
based on SpectrumPhy and it follows specification described in section 6 
of IEEE Std 802.15.4-2006. It models PHY service specifications, PPDU 
formats, PHY constants and PIB attributes. It currently only supports 
the transmit power spectral density mask specified in 2.4 GHz per section 
6.5.3.1. The noise power density assumes uniformly distributed thermal 
noise across the frequency bands. The loss model can fully utilize all 
existing simple (non-spectrum phy) loss models. The Phy model uses 
the existing single spectrum channel model.
The physical layer is modeled on packet level, that is, no preamble/SFD
detection is done. Packet reception will be started with the first bit of the
preamble (which is not modeled), if the SNR is more than -5 dB, see IEEE
Std 802.15.4-2006, appendix E, Figure E.2. Reception of the packet will finish
after the packet was completely transmitted. Other packets arriving during
reception will add up to the interference/noise.

Currently the receiver sensitivity is set to a fixed value of -106.58 dBm. This
corresponds to a packet error rate of 1% for 20 byte reference packets for this
signal power, according to IEEE Std 802.15.4-2006, section 6.1.7. In the future
we will provide support for changing the sensitivity to different values. 

.. _fig-802-15-4-per-sens:

.. figure:: figures/802-15-4-per-sens.*

    Packet error rate vs. signal power


NetDevice
#########

Although it is expected that other technology profiles (such as 
6LoWPAN and ZigBee) will write their own NetDevice classes, a basic
LrWpanNetDevice is provided, which encapsulates the common operations
of creating a generic LrWpan device and hooking things together.

MAC addresses
+++++++++++++

Contrary to other technologies, a IEEE 802.15.4 has 2 different kind of addresses:

* Long addresses (64 bits)
* Short addresses (16 bits)

The 64-bit addresses are unique worldwide, and set by the device vendor (in a real device).
The 16-bit addresses are not guaranteed to be unique, and they are typically either assigned 
during the devices deployment, or assigned dynamically during the device bootstrap.

In |ns3| the device bootstrap is not (yet) present. Hence, both addresses are set when the
device is created.

The other relavant "address" to consider is the PanId (16 bits), which represents the PAN
the device is attached to.

Due to the limited number of available bytes in a packet, IEEE 802.15.4 tries to use short 
addresses instead of long addresses, even though the two might be used at the same time.

For the sake of communicating with the upper layers, and in particular to generate auto-configured
IPv6 addresses, each NetDevice must identify itself with a MAC address. The MAC addresses are
also used during packet reception, so it is important to use them consistently.

Focusing on IPv6 Stateless address autoconfiguration (SLAAC), there are two relevant RFCs to
consider: RFC 4944 and RFC 6282, and the two differ on how to build the IPv6 address given
the NetDevice address.

RFC 4944 mandates that the IID part of the IPv6 address is calculated as ``YYYY:00ff:fe00:XXXX``,
while RFC 6282 mandates that the IID part of the IPv6 address is calculated as ``0000:00ff:fe00:XXXX``
where ``XXXX`` is the device short address, and ``YYYY`` is the PanId.
In both cases the U/L bit must be set to local, so in the RFC 4944 the PanId might have one bit flipped.

In order to facilitate interoperability, and to avoid unwanted module dependencies, the |ns3| 
implementation moves the IID calculation in the ``LrWpanNetDevice::GetAddress ()``, which will
return an ``Address`` formatted properly, i.e.:

* The Long address (a ``Mac64Address``) if the Short address has not been set, or
* A properly formatted 48-bit pseudo-address (a ``Mac48Address``) if the short address has been set.

The 48-bit pseudo-address is generated according to either RFC 4944 or RFC 6282 depending on the
configuration of an Attribute (``PseudoMacAddressMode``).

The default is to use RFC 6282 style addresses.

Note that, on reception, a packet might contain either a short or a long address. This is reflected
in the upper-layer notification callback, which can contain either the pseudo-address (48 bits) or 
the long address (64 bit) of the sender.

Note also that RFC 4944 or RFC 6282 are the RFCs defining the IPv6 address compression formats
(HC1 and IPHC respectively). It is defintely not a good idea to either mix devices using different
pseudo-address format or compression types in the same network. This point is further discussed
in the ``sixlowpan`` module documentation.


Scope and Limitations
=====================

Future versions of this document will contain a PICS proforma similar to
Appendix D of IEEE 802.15.4-2006.  The current emphasis is on the 
unslotted mode of 802.15.4 operation for use in Zigbee, and the scope
is limited to enabling a single mode (CSMA/CA) with basic data transfer
capabilities. Association with PAN coordinators is not yet supported, nor the
use of extended addressing. Interference is modeled as AWGN but this is
currently not thoroughly tested.

The NetDevice Tx queue is not limited, i.e., packets are never dropped
due to queue becoming full. They may be dropped due to excessive transmission 
retries or channel access failure.

References
==========

* Wireless Medium Access Control (MAC) and Physical Layer (PHY) Specifications for Low-Rate Wireless Personal Area Networks (WPANs), IEEE Computer Society, IEEE Std 802.15.4-2006, 8 September 2006.
* J. Zheng and Myung J. Lee, "A comprehensive performance study of IEEE 802.15.4," Sensor Network Operations, IEEE Press, Wiley Interscience, Chapter 4, pp. 218-237, 2006.

Usage
*****

Enabling lr-wpan
================

Add ``lr-wpan`` to the list of modules built with |ns3|.

Helper
======

The helper is patterned after other device helpers.  In particular,
tracing (ascii and pcap) is enabled similarly, and enabling of all
lr-wpan log components is performed similarly.  Use of the helper
is exemplified in ``examples/lr-wpan-data.cc``.  For ascii tracing,
the transmit and receive traces are hooked at the Mac layer.

The default propagation loss model added to the channel, when this helper
is used, is the LogDistancePropagationLossModel with default parameters.

Examples
========

The following examples have been written, which can be found in ``src/lr-wpan/examples/``:

* ``lr-wpan-data.cc``:  A simple example showing end-to-end data transfer.
* ``lr-wpan-error-distance-plot.cc``:  An example to plot variations of the packet success ratio as a function of distance.
* ``lr-wpan-error-model-plot.cc``:  An example to test the phy.
* ``lr-wpan-packet-print.cc``:  An example to print out the MAC header fields.
* ``lr-wpan-phy-test.cc``:  An example to test the phy.

In particular, the module enables a very simplified end-to-end data
transfer scenario, implemented in ``lr-wpan-data.cc``.  The figure
shows a sequence of events that are triggered when the MAC receives
a DataRequest from the higher layer.  It invokes a Clear Channel
Assessment (CCA) from the PHY, and if successful, sends the frame
down to the PHY where it is transmitted over the channel and results
in a DataIndication on the peer node.
  
.. _fig-lr-wpan-data:

.. figure:: figures/lr-wpan-data-example.*

    Data example for simple LR-WPAN data transfer end-to-end

The example ``lr-wpan-error-distance-plot.cc`` plots the packet success
ratio (PSR) as a function of distance, using the default LogDistance
propagation loss model and the 802.15.4 error model.  The channel (default 11),
packet size (default 20 bytes) and transmit power (default 0 dBm) can be
varied by command line arguments.  The program outputs a file named
``802.15.4-psr-distance.plt``.  Loading this file into gnuplot yields
a file ``802.15.4-psr-distance.eps``, which can be converted to pdf or
other formats.  The default output is shown below. 

.. _fig-802-15-4-psr-distance:

.. figure:: figures/802-15-4-psr-distance.*

    Default output of the program ``lr-wpan-error-distance-plot.cc``

Tests
=====

The following tests have been written, which can be found in ``src/lr-wpan/tests/``:

* ``lr-wpan-ack-test.cc``:  Check that acknowledgments are being used and issued in the correct order.
* ``lr-wpan-collision-test.cc``:  Test correct reception of packets with interference and collisions.
* ``lr-wpan-error-model-test.cc``:  Check that the error model gives predictable values.
* ``lr-wpan-packet-test.cc``:  Test the 802.15.4 MAC header/trailer classes
* ``lr-wpan-pd-plme-sap-test.cc``:  Test the PLME and PD SAP per IEEE 802.15.4
* ``lr-wpan-spectrum-value-helper-test.cc``:  Test that the conversion between power (expressed as a scalar quantity) and spectral power, and back again, falls within a 25% tolerance across the range of possible channels and input powers.

Validation
**********

The model has not been validated against real hardware.  The error model
has been validated against the data in IEEE Std 802.15.4-2006, 
section E.4.1.7 (Figure E.2). The MAC behavior (CSMA backoff) has been 
validated by hand against expected behavior.  The below plot is an example 
of the error model validation and can be reproduced by running
``lr-wpan-error-model-plot.cc``:

.. _fig-802-15-4-ber:

.. figure:: figures/802-15-4-ber.*

    Default output of the program ``lr-wpan-error-model-plot.cc`` 


