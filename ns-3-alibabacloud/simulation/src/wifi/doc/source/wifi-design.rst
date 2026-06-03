.. include:: replace.txt
.. highlight:: cpp

++++++++++++++++++++
Design Documentation
++++++++++++++++++++


|ns3| nodes can contain a collection of NetDevice objects, much like an actual
computer contains separate interface cards for Ethernet, Wifi, Bluetooth, etc.
This chapter describes the |ns3| WifiNetDevice and related models. By adding
WifiNetDevice objects to |ns3| nodes, one can create models of 802.11-based
infrastructure and ad hoc networks.

Overview of the model
*********************

The WifiNetDevice models a wireless network interface controller based
on the IEEE 802.11 standard [ieee80211]_. We will go into more detail below but in brief,
|ns3| provides models for these aspects of 802.11:

* basic 802.11 DCF with **infrastructure** and **adhoc** modes
* **802.11a**, **802.11b**, **802.11g**, **802.11n** (both 2.4 and 5 GHz bands), **802.11ac** and **802.11ax** (2.4, 5 and 6 GHz bands) physical layers
* **MSDU aggregation** and **MPDU aggregation** extensions of 802.11n, and both can be combined together (two-level aggregation)
* 802.11ax **DL OFDMA** and **UL OFDMA** (including support for the MU EDCA Parameter Set)
* QoS-based EDCA and queueing extensions of **802.11e**
* the ability to use different propagation loss models and propagation delay models,
  please see the chapter on :ref:`Propagation` for more detail
* packet error models and frame detection models that have been validated
  against link simulations and other references
* various rate control algorithms including **Aarf, Arf, Cara, Onoe, Rraa,
  ConstantRate, Minstrel and Minstrel-HT**
* 802.11s (mesh), described in another chapter
* 802.11p and WAVE (vehicular), described in another chapter

The set of 802.11 models provided in |ns3| attempts to provide an accurate
MAC-level implementation of the 802.11 specification and to provide a
packet-level abstraction of the PHY-level for different PHYs, corresponding to
802.11a/b/e/g/n/ac/ax specifications.

In |ns3|, nodes can have multiple WifiNetDevices on separate channels, and the
WifiNetDevice can coexist with other device types.
With the use of the **SpectrumWifiPhy** framework, one can also build scenarios
involving cross-channel interference or multiple wireless technologies on
a single channel.

The source code for the WifiNetDevice and its models lives in the directory
``src/wifi``.

The implementation is modular and provides roughly three sublayers of models:

* the **PHY layer models**: they model amendment-specific and common
  PHY layer operations and functions.
* the so-called **MAC low models**: they model functions such as medium
  access (DCF and EDCA), frame protection (RTS/CTS) and acknowledgment (ACK/BlockAck).
  In |ns3|, the lower-level MAC is comprised of a **Frame Exchange Manager** hierarchy,
  a **Channel Access Manager** and a **MAC middle** entity.
* the so-called **MAC high models**: they implement non-time-critical processes
  in Wifi such as the MAC-level beacon generation, probing, and association
  state machines, and a set of **Rate control algorithms**.  In the literature,
  this sublayer is sometimes called the **upper MAC** and consists of more
  software-oriented implementations vs. time-critical hardware implementations.

Next, we provide an design overview of each layer, shown in
Figure :ref:`wifi-architecture`.

.. _wifi-architecture:

.. figure:: figures/WifiArchitecture.*

   *WifiNetDevice architecture*

MAC high models
===============

There are presently three **MAC high models** that provide for the three
(non-mesh; the mesh equivalent, which is a sibling of these with common
parent ``ns3::WifiMac``, is not discussed here) Wi-Fi topological
elements - Access Point (AP) (``ns3::ApWifiMac``),
non-AP Station (STA) (``ns3::StaWifiMac``), and STA in an Independent
Basic Service Set (IBSS) - also commonly referred to as an ad hoc
network (``ns3::AdhocWifiMac``).

The simplest of these is ``ns3::AdhocWifiMac``, which implements a
Wi-Fi MAC that does not perform any kind of beacon generation,
probing, or association. The ``ns3::StaWifiMac`` class implements
an active probing and association state machine that handles automatic
re-association whenever too many beacons are missed. Finally,
``ns3::ApWifiMac`` implements an AP that generates periodic
beacons, and that accepts every attempt to associate.

These three MAC high models share a common parent in
``ns3::WifiMac``, which exposes, among other MAC
configuration, an attribute ``QosSupported`` that allows
configuration of 802.11e/WMM-style QoS support.

There are also several **rate control algorithms** that can be used by the
MAC low layer.  A complete list of available rate control algorithms is
provided in a separate section.

MAC low layer
==============

The **MAC low layer** is split into three main components:

#. ``ns3::FrameExchangeManager`` a class hierarchy which implement the frame exchange
   sequences introduced by the supported IEEE 802.11 amendments. It also handles
   frame aggregation, frame retransmissions, protection and acknowledgment.
#. ``ns3::ChannelAccessManager`` which implements the DCF and EDCAF
   functions.
#. ``ns3::Txop`` and ``ns3::QosTxop`` which handle the packet queue.
   The ``ns3::Txop`` object is used by high MACs that are not QoS-enabled,
   and for transmission of frames (e.g., of type Management)
   that the standard says should access the medium using the DCF.
   ``ns3::QosTxop`` is used by QoS-enabled high MACs.

PHY layer models
================

In short, the physical layer models are mainly responsible for modeling
the reception of packets and for tracking energy consumption.  There
are typically three main components to packet reception:

* each packet received is probabilistically evaluated for successful or
  failed reception.  The probability depends on the modulation, on
  the signal to noise (and interference) ratio for the packet, and on
  the state of the physical layer (e.g. reception is not possible while
  transmission or sleeping is taking place);
* an object exists to track (bookkeeping) all received signals so that
  the correct interference power for each packet can be computed when
  a reception decision has to be made; and
* one or more error models corresponding to the modulation and standard
  are used to look up probability of successful reception.

|ns3| offers users a choice between two physical layer models, with a
base interface defined in the ``ns3::WifiPhy`` class.  The YansWifiPhy
class implements a simple physical layer model, which is described
in a paper entitled
`Yet Another Network Simulator <https://dl.acm.org/doi/pdf/10.1145/1190455.1190467?download=true>`_
The acronym *Yans* derives from this paper title.  The SpectrumWifiPhy
class is a more advanced implementation based on the Spectrum framework
used for other |ns3| wireless models.  Spectrum allows a fine-grained
frequency decomposition of the signal, and permits scenarios to
include multiple technologies coexisting on the same channel.

Scope and Limitations
*********************

The IEEE 802.11 standard [ieee80211]_ is a large specification,
and not all aspects are covered by |ns3|; the documentation of |ns3|'s
conformance by itself would lead to a very long document.  This section
attempts to summarize compliance with the standard and with behavior
found in practice.

The physical layer and channel models operate on a per-packet basis, with
no frequency-selective propagation nor interference effects when using
the default YansWifiPhy model.  Directional antennas are also not
supported at this time.  For additive white Gaussian noise (AWGN)
scenarios, or wideband interference scenarios, performance is governed
by the application of analytical models (based on modulation and factors
such as channel width) to the received signal-to-noise ratio, where noise
combines the effect of thermal noise and of interference from other Wi-Fi
packets.  Interference from other wireless technologies is only modeled
when the SpectrumWifiPhy is used.
The following details pertain to the physical layer and channel models:

* 802.11ax MU-RTS/CTS is not yet supported
* 802.11ac/ax MU-MIMO is not supported, and no more than 4 antennas can be configured
* 802.11n/ac/ax beamforming is not supported
* 802.11n RIFS is not supported
* 802.11 PCF/HCF/HCCA are not implemented
* Channel Switch Announcement is not supported
* Authentication and encryption are missing
* Processing delays are not modeled
* Channel bonding implementation only supports the use of the configured channel width
  and does not perform CCA on secondary channels
* Cases where RTS/CTS and ACK are transmitted using HT/VHT/HE formats are not supported
* Energy consumption model does not consider MIMO

At the MAC layer, most of the main functions found in deployed Wi-Fi
equipment for 802.11a/b/e/g/n/ac/ax are implemented, but there are scattered instances
where some limitations in the models exist. Support for 802.11n, ac and ax is evolving.

Some implementation choices that are not imposed by the standard are listed below:

* BSSBasicRateSet for 802.11b has been assumed to be 1-2 Mbit/s
* BSSBasicRateSet for 802.11a/g has been assumed to be 6-12-24 Mbit/s
* OperationalRateSet is assumed to contain all mandatory rates (see
  `issue 183 <https://gitlab.com/nsnam/ns-3-dev/-/issues/183>`_)
* The wifi manager always selects the lowest basic rate for management frames.

Design Details
**************

The remainder of this section is devoted to more in-depth design descriptions
of some of the Wi-Fi models.  Users interested in skipping to the section
on usage of the wifi module (:ref:`User Documentation<sec-wifi-user-doc>`) may do so at this point.
We organize these more detailed sections from the bottom-up, in terms of
layering, by describing the channel and PHY models first, followed by
the MAC models.

We focus first on the choice between physical layer frameworks.  |ns3|
contains support for a Wi-Fi-only physical layer model called YansWifiPhy
that offers no frequency-level decomposition of the signal.  For simulations
that involve only Wi-Fi signals on the Wi-Fi channel, and that do not
involve frequency-dependent propagation loss or fading models, the default
YansWifiPhy framework is a suitable choice.  For simulations involving
mixed technologies on the same channel, or frequency dependent effects,
the SpectrumWifiPhy is more appropriate.  The two frameworks are very
similarly configured.

The SpectrumWifiPhy framework uses the :ref:`sec-spectrum-module` channel
framework.

The YansWifiChannel is the only concrete channel model class in
the |ns3| wifi module.  The
``ns3::YansWifiChannel`` implementation uses the propagation loss and
delay models provided within the |ns3| :ref:`Propagation` module.
In particular, a number of propagation models can be added (chained together,
if multiple loss models are added) to the channel object, and a propagation
delay model also added. Packets sent from a ``ns3::YansWifiPhy`` object
onto the channel with a particular signal power, are copied to all of the
other ``ns3::YansWifiPhy`` objects after the signal power is reduced due
to the propagation loss model(s), and after a delay corresponding to
transmission (serialization) delay and propagation delay due to
any channel propagation delay model (typically due to speed-of-light
delay between the positions of the devices).

Only objects of ``ns3::YansWifiPhy`` may be attached to a
``ns3::YansWifiChannel``; therefore, objects modeling other
(interfering) technologies such as LTE are not allowed. Furthermore,
packets from different channels do not interact; if a channel is logically
configured for e.g. channels 5 and 6, the packets do not cause
adjacent channel interference (even if their channel numbers overlap).

WifiPhy and related models
==========================

The ``ns3::WifiPhy`` is an abstract base class representing the 802.11
physical layer functions.  Packets passed to this object (via a
``Send()`` method) are sent over a channel object, and
upon reception, the receiving PHY object decides (based on signal power
and interference) whether the packet was successful or not.  This class
also provides a number of callbacks for notifications of physical layer
events, exposes a notion of a state machine that can be monitored for
MAC-level processes such as carrier sense, and handles sleep/wake/off models
and energy consumption.  The ``ns3::WifiPhy`` hooks to the ``ns3::FrameExchangeManager``
object in the WifiNetDevice.

There are currently two implementations of the ``WifiPhy``: the
``ns3::YansWifiPhy`` and the ``ns3::SpectrumWifiPhy``.  They each work in
conjunction with five other objects:

* **PhyEntity**: Contains the amendment-specific part of the PHY processing
* **WifiPpdu**: Models the amendment-specific PHY protocol data unit (PPDU)
* **WifiPhyStateHelper**:  Maintains the PHY state machine
* **InterferenceHelper**:  Tracks all packets observed on the channel
* **ErrorModel**:  Computes a probability of error for a given SNR

PhyEntity
##################################

A bit of background
-------------------

Some restructuring of ``ns3::WifiPhy`` and ``ns3::WifiMode`` (among others) was necessary
considering the size and complexity of the corresponding files.
In addition, adding and maintaining new PHY amendments had become a complex
task (especially those implemented inside other modules, e.g. DMG).
The adopted solution was to have ``PhyEntity`` classes that contain the "clause"
specific (i.e. HT/VHT/HE etc) parts of the PHY process.

The notion of "PHY entity" is in the standard at the beginning of each PHY
layer description clause, e.g. section 21.1.1 of IEEE 802.11-2016:

::
Clause 21 specifies the **PHY entity** for a very high throughput (VHT) orthogonal
frequency division multiplexing (OFDM) system.

*Note that there is already such a name inside the wave module
(e.g. ``WaveNetDevice::AddPhy``) to designate the WifiPhys on each 11p channel,
but the wording is only used within the classes and there is no file using
that name, so no ambiguity in using the name for 802.11 amendments.*

Architecture
-------------------

The abstract base class ``ns3::PhyEntity`` enables to have a unique set of APIs
to be used by each PHY entity, corresponding to the different amendments of
the IEEE 802.11 standard. The currently implemented PHY entities are:

* ``ns3::DsssPhy``: PHY entity for DSSS and HR/DSSS (11b)
* ``ns3::OfdmPhy``: PHY entity for OFDM (11a and 11p)
* ``ns3::ErpOfdmPhy``: PHY entity for ERP-OFDM (11g)
* ``ns3::HtPhy``: PHY entity for HT (11n)
* ``ns3::VhtPhy``: PHY entity for VHT (11ac)
* ``ns3::HePhy``: PHY entity for HE (11ax)

Their inheritance diagram is given in Figure :ref:`phyentity-hierarchy` and
closely follows the standard's logic, e.g. section 21.1.1 of IEEE 802.11-2016:

::
The VHT PHY is **based** on the HT PHY defined in Clause 19, which **in turn**
is **based** on the OFDM PHY defined in Clause 17.

.. _phyentity-hierarchy:

.. figure:: figures/PhyEntityHierarchy.*

   *PhyEntity hierarchy*

Such an architecture enables to handle the following operations in an amendment-
specific manner:

* ``WifiMode`` handling and data/PHY rate computation,
* PPDU field size and duration computation, and
* Transmit and receive paths.

WifiPpdu
##################################

In the same vein as ``PhyEntity``, the ``ns3::WifiPpdu`` base class has been
specialized into the following amendment-specific PPDUs:

* ``ns3::DsssPpdu``: PPDU for DSSS and HR/DSSS (11b)
* ``ns3::OfdmPpdu``: PPDU for OFDM (11a and 11p)
* ``ns3::ErpOfdmPpdu``: PPDU for ERP-OFDM (11g)
* ``ns3::HtPpdu``: PPDU for HT (11n)
* ``ns3::VhtPpdu``: PPDU for VHT (11ac)
* ``ns3::HePpdu``: PPDU for HE (11ax)

Their inheritance diagram is given in Figure :ref:`wifippdu-hierarchy` and
closely follows the standard's logic, e.g. section 21.3.8.1 of IEEE 802.11-2016:

::
To maintain compatibility with non-VHT STAs, specific non-VHT fields are defined
that can be received by non-VHT STAs compliant with **Clause 17** [OFDM] or **Clause 19** [HT].

.. _wifippdu-hierarchy:

.. figure:: figures/WifiPpduHierarchy.*

   *WifiPpdu hierarchy*

YansWifiPhy and WifiPhyStateHelper
##################################

Class ``ns3::YansWifiPhy`` is responsible for taking packets passed to
it from the MAC (the ``ns3::FrameExchangeManager`` object) and sending them onto the
``ns3::YansWifiChannel`` to which it is attached.  It is also responsible
to receive packets from that channel, and, if reception is deemed to have
been successful, to pass them up to the MAC.

The energy of the signal intended to be received is
calculated from the transmission power and adjusted based on the Tx gain
of the transmitter, Rx gain of the receiver, and any path loss propagation
model in effect.

Class ``ns3::WifiPhyStateHelper`` manages the state machine of the PHY
layer, and allows other objects to hook as *listeners* to monitor PHY
state.  The main use of listeners is for the MAC layer to know when
the PHY is busy or not (for transmission and collision avoidance).

The PHY layer can be in one of seven states:

#. TX: the PHY is currently transmitting a signal on behalf of its associated
   MAC
#. RX: the PHY is synchronized on a signal and is waiting until it has received
   its last bit to forward it to the MAC.
#. IDLE: the PHY is not in the TX, RX, or CCA_BUSY states.
#. CCA_BUSY: the PHY is not in TX or RX state but the measured energy is higher than the energy detection threshold.
#. SWITCHING: the PHY is switching channels.
#. SLEEP: the PHY is in a power save mode and cannot send nor receive frames.
#. OFF: the PHY is powered off and cannot send nor receive frames.

Packet reception works as follows.  For ``YansWifiPhy``, most of the logic
is implemented in the ``WifiPhy`` base class.  The ``YansWifiChannel`` calls
``WifiPhy::StartReceivePreamble ()``. The latter calls
``PhyEntity::StartReceivePreamble ()`` of the appropriate PHY entity
to start packet reception, but first
there is a check of the packet's notional signal power level against a
threshold value stored in the attribute ``WifiPhy::RxSensitivity``.  Any
packet with a power lower than RxSensitivity will be dropped with no
further processing.  The default value is -101 dBm, which is the thermal
noise floor for 20 MHz signal at room temperature.  The purpose of this
attribute is two-fold:  1) very weak signals that will not affect the
outcome will otherwise consume simulation memory and event processing, so
they are discarded, and 2) this value can be adjusted upwards to function as
a basic carrier sense threshold limitation for experiments involving
spatial reuse considerations.  Users are cautioned about the behavior of
raising this threshold; namely, that all packets with power below this
threshold will be discarded upon reception.

In ``StartReceivePreamble ()``, the packet is immediately added
to the interference helper for signal-to-noise
tracking, and then further reception steps are decided upon the state of
the PHY.  In the case that the PHY is transmitting, for instance, the
packet will be dropped.  If the PHY is IDLE, or if the PHY is receiving and
an optional FrameCaptureModel is being used (and the packet is within
the capture window), then ``PhyEntity::StartPreambleDetectionPeriod ()`` is called next.

The ``PhyEntity::StartPreambleDetectionPeriod ()`` will typically schedule an event,
``PhyEntity::EndPreambleDetectionPeriod ()``, to occur at
the notional end of the first OFDM symbol, to check whether the preamble
has been detected.  As of revisions to the model in ns-3.30, any state
machine transitions from IDLE state are suppressed until after the preamble
detection event.

The ``PhyEntity::EndPreambleDetectionPeriod ()`` method will check, with a preamble detection
model, whether the signal is strong enough to be received, and if so,
an event ``PhyEntity::EndReceiveField ()`` is scheduled for the end of the
preamble and the PHY is put into the CCA_BUSY state. Currently, there is only a
simple threshold-based preamble detection model in ns-3,
called ``ThresholdPreambleDetectionModel``.  If there is no preamble detection
model, the preamble is assumed to have been detected.
It is important to note that, starting with the ns-3.30 release, the default
in the WifiPhyHelper is to add the ``ThresholdPreambleDetectionModel`` with
a threshold RSSI of -82 dBm, and a threshold SNR of 4 dB.  Both the RSSI
and SNR must be above these respective values for the preamble to be
successfully detected.  The default sensitivity has been reduced in ns-3.30
compared with that of previous releases, so some packet receptions that were
previously successful will now fail on this check.  More details on the
modeling behind this change are provided in [lanante2019]_.

The ``PhyEntity::EndReceiveField ()`` method will check the correct reception
of the current preamble and header field and, if so, calls ``PhyEntity::StartReceiveField ()``
for the next field,
otherwise the reception is aborted and PHY is put either in IDLE state or in CCA_BUSY state,
depending on whether the measured energy is higher than the energy detection threshold.

The next event at ``PhyEntity::StartReceiveField ()`` checks, using the interference
helper and error model, whether the header was successfully decoded, and if so,
a ``PhyRxPayloadBegin`` callback (equivalent to the PHY-RXSTART primitive)
is triggered. The PHY header is often transmitted
at a lower modulation rate than is the payload. The portion of the packet
corresponding to the PHY header is evaluated for probability of error
based on the observed SNR.  The InterferenceHelper object returns a value
for "probability of error (PER)" for this header based on the SNR that has
been tracked by the InterferenceHelper.  The ``PhyEntity`` then draws
a random number from a uniform distribution and compares it against the
PER and decides success or failure.

This is iteratively performed up to the beginning of the data field
upon which ``PhyEntity::StartReceivePayload ()`` is called.

Even if packet objects received by the PHY are not part of the reception
process, they are tracked by the InterferenceHelper object for purposes
of SINR computation and making clear channel assessment decisions.
If, in the course of reception, a packet is errored or dropped due to
the PHY being in a state in which it cannot receive a packet, the packet
is added to the interference helper, and the aggregate of the energy of
all such signals is compared against an energy detection threshold to
determine whether the PHY should enter a CCA_BUSY state.
The ``WifiPhy::CcaEdThreshold`` attribute
corresponds to what the standard calls the "ED threshold" for CCA Mode 1.
In section 16.4.8.5 in the 802.11-2012 standard: "CCA Mode 1: Energy above
threshold. CCA shall report a busy medium upon detection of any energy above
the ED threshold." By default, this value is set to the -62 dBm level specified
in the standard for 20 MHz channels. When using ``YansWifiPhy``, there are no
non-Wi-Fi signals, so it is unlikely that this attribute would play much of a
role in Yans wifi models if left at the default value, but if there is a strong
Wi-Fi signal that is not otherwise being received by the model, it has
the possibility to raise the CCA_BUSY while the overall energy exceeds
this threshold.

The above describes the case in which the packet is a single MPDU.  For
more recent Wi-Fi standards using MPDU aggregation, ``StartReceivePayload``
schedules an event for reception of each individual MPDU (``ScheduleEndOfMpdus``),
which then forwards each MPDU as they arrive up to FrameExchangeManager, if the
reception of the MPDU has been successful. Once the A-MPDU reception is finished,
FrameExchangeManager is also notified about the amount of successfully received MPDUs.

InterferenceHelper
##################

The InterferenceHelper is an object that tracks all incoming packets and
calculates probability of error values for packets being received, and
also evaluates whether energy on the channel rises above the CCA
threshold.

The basic operation of probability of error calculations is shown in Figure
:ref:`snir`.  Packets are represented as bits (not symbols) in the |ns3|
model, and the InterferenceHelper breaks the packet into one or more
"chunks", each with a different signal to noise (and interference) ratio
(SNIR).  Each chunk is separately evaluated by asking for the probability
of error for a given number of bits from the error model in use.  The
InterferenceHelper builds an aggregate "probability of error" value
based on these chunks and their duration, and returns this back to
the ``WifiPhy`` for a reception decision.

.. _snir:

.. figure:: figures/snir.*

   *SNIR function over time*

From the SNIR function we can derive the Bit Error Rate (BER) and Packet
Error Rate (PER) for
the modulation and coding scheme being used for the transmission.

If MIMO is used and the number of spatial streams is lower than the number
of active antennas at the receiver, then a gain is applied to the calculated
SNIR as follows (since STBC is not used):

.. math::

  gain (dB) = 10 \log(\frac{RX \ antennas}{spatial \ streams})

Having more TX antennas can be safely ignored for AWGN. The resulting gain is:

::

  antennas   NSS    gain
  2 x 1       1     0 dB
  1 x 2       1     3 dB
  2 x 2       1     3 dB
  3 x 3       1   4.8 dB
  3 x 3       2   1.8 dB
  3 x 3       3     0 dB
  4 x 4       1     6 dB
  4 x 4       2     3 dB
  4 x 4       3   1.2 dB
  4 x 4       4     0 dB
  ...

ErrorRateModel
##############

|ns3| makes a packet error or success decision based on the input received
SNR of a frame and based on any possible interfering frames that may overlap
in time; i.e. based on the signal-to-noise (plus interference) ratio, or
SINR.  The relationship between packet error ratio (PER) and SINR in |ns3|
is defined by the ``ns3::ErrorRateModel``, of which there are several.
The PER is a function of the frame's modulation and coding (MCS), its SINR,
and the specific ErrorRateModel configured for the MCS.

|ns3| has updated its default ErrorRateModel over time.  The current
(as of ns-3.33 release) model for recent OFDM-based standards (i.e.,
802.11n/ac/ax), is the ``ns3::TableBasedErrorRateModel``.  The default
for 802.11a/g is the ``ns3::YansErrorRateModel``, and the default for
802.11b is the ``ns3::DsssErrorRateModel``.  The error rate model for
recent standards was updated during the ns-3.33 release cycle (previously,
it was the ``ns3::NistErrorRateModel``).

The error models are described in more detail in outside references.  The
current OFDM model is based on work published in [patidar2017]_, using
link simulations results from the MATLAB WLAN Toolbox, and validated against
IEEE TGn results [erceg2004]_.  For publications related to other error models,
please refer to [pei80211ofdm]_, [pei80211b]_, [lacage2006yans]_, [Haccoun]_,
[hepner2015]_ and [Frenger]_ for a detailed description of the legacy PER models.

The current |ns3| error rate models are for additive white gaussian
noise channels (AWGN) only; any potential frequency-selective fading
effects are not modeled.

In summary, there are four error models:

#. ``ns3::TableBasedErrorRateModel``: for OFDM modes and reuses
   ``ns3::DsssErrorRateModel`` for 802.11b modes.
   This is the default for 802.11n/ac/ax.
#. ``ns3::YansErrorRateModel``: for OFDM modes and reuses
   ``ns3::DsssErrorRateModel`` for 802.11b modes.
   This is the default for 802.11a/g.
#. ``ns3::DsssErrorRateModel``:  contains models for 802.11b modes.  The
   802.11b 1 Mbps and 2 Mbps error models are based on classical modulation
   analysis.  If GNU Scientific Library (GSL) is installed, the 5.5 Mbps
   and 11 Mbps from [pursley2009]_ are used for CCK modulation;
   otherwise, results from a backup MATLAB-based CCK model are used.
#. ``ns3::NistErrorRateModel``: for OFDM modes and reuses
   ``ns3::DsssErrorRateModel`` for 802.11b modes.

Users may select either NIST, YANS or Table-based models for OFDM,
and DSSS will be used in either case for 802.11b.  The NIST model was
a long-standing default in ns-3 (through release 3.32).

TableBasedErrorRateModel
########################

The ``ns3::TableBasedErrorRateModel`` has been recently added and is now the |ns3| default
for 802.11n/ac/ax, while ``ns3::YansErrorRateModel`` is the |ns3| default for 802.11a/g.

Unlike analytical error models based on error bounds, ``ns3::TableBasedErrorRateModel`` contains
end-to-end link simulation tables (PER vs SNR) for AWGN channels. Since it is infeasible to generate
such look-up tables for all desired packet sizes and input SNRs, we adopt the recommendation of IEEE P802.11 TGax [porat2016]_ that proposed
estimating PER for any desired packet length using BCC FEC encoding by extrapolating the results from two reference lengths:
32 (all lengths less than 400) bytes and 1458 (all lengths greater or equal to 400) bytes respectively.
In case of LDPC FEC encoding, IEEE P802.11 TGax recommends the use of a single reference length.
Hence, we provide two tables for BCC and one table for LDPC that are generated using a reliable and publicly
available commercial link simulator (MATLAB WLAN Toolbox) for each modulation and coding scheme.
Note that BCC tables are limited to MCS 9. For higher MCSs, the models fall back to the use of the YANS analytical model.

The validation scenario is set as follows:

#. Ideal channel and perfect channel estimation.
#. Perfect packet synchronization and detection.
#. Phase tracking, phase correction, phase noise, carrier frequency offset, power amplifier non-linearities etc. are not considered.

Several packets are simulated across the link to obtain PER, the number of packets needed to reliably
estimate a PER value is computed using the consideration that the ratio of the estimation error to the
true value should be within 10 % with probability 0.95.
For each SNR value, simulations were run until a total of 40000 packets were simulated.

The obtained results are very close to TGax curves as shown in Figure
:ref:`default-table-based-error-model-validation`

.. _default-table-based-error-model-validation:

.. figure:: figures/default-table-based-error-model-validation.*
   :scale: 75%

   *Comparison of table-based OFDM Error Model with TGax results.*

Legacy ErrorRateModels
######################

The original error rate model was called the ``ns3::YansErrorRateModel`` and
was based on analytical results.  For 802.11b modulations, the 1 Mbps mode
is based on DBPSK. BER is from equation 5.2-69 from [proakis2001]_.
The 2 Mbps model is based on DQPSK. Equation 8 of [ferrari2004]_.
More details are provided in [lacage2006yans]_.

The ``ns3::NistErrorRateModel`` was later added.
The model was largely aligned with the previous ``ns3::YansErrorRateModel``
for DSSS modulations 1 Mbps and 2 Mbps, but the 5.5 Mbps and 11 Mbps models
were re-based on equations (17) and (18) from [pursley2009]_.
For OFDM modulations, newer results were
obtained based on work previously done at NIST [miller2003]_.  The results
were also compared against the CMU wireless network emulator, and details
of the validation are provided in [pei80211ofdm]_.  Since OFDM modes use
hard-decision of punctured codes, the coded BER is calculated using
Chernoff bounds [hepner2015]_.

The 802.11b model was split from the OFDM model when the NIST error rate
model was added, into a new model called DsssErrorRateModel.

Furthermore, the 5.5 Mbps and 11 Mbps models for 802.11b rely on library
methods implemented in the GNU Scientific Library (GSL).  The ns3 build
system tries to detect whether the host platform has GSL installed; if so,
it compiles in the newer models from [pursley2009]_ for 5.5 Mbps and 11 Mbps;
if not, it uses a backup model derived from MATLAB simulations.

The error curves for analytical models are shown to diverge from link simulation results for higher MCS in
Figure :ref:`error-models-comparison`. This prompted the move to a new error
model based on link simulations (the default TableBasedErrorRateModel, which
provides curves close to those depicted by the TGn dashed line).

.. _error-models-comparison:

.. figure:: figures/error-models-comparison.*

  *YANS and NIST error model comparison with TGn results*

SpectrumWifiPhy
###############

This section describes the implementation of the ``SpectrumWifiPhy``
class that can be found in ``src/wifi/model/spectrum-wifi-phy.{cc,h}``.

The implementation also makes use of additional classes found in the
same directory:

* ``wifi-spectrum-phy-interface.{cc,h}``
* ``wifi-spectrum-signal-parameters.{cc,h}``

and classes found in the spectrum module:

* ``wifi-spectrum-value-helper.{cc,h}``

The current ``SpectrumWifiPhy`` class
reuses the existing interference manager and error rate models originally
built for ``YansWifiPhy``, but allows, as a first step, foreign (non Wi-Fi)
signals to be treated as additive noise.

Two main changes were needed to adapt the Spectrum framework to Wi-Fi.
First, the physical layer must send signals compatible with the
Spectrum channel framework, and in particular, the
``MultiModelSpectrumChannel`` that allows signals from different
technologies to coexist.  Second, the InterferenceHelper must be
extended to support the insertion of non-Wi-Fi signals and to
add their received power to the noise, in the same way that
unintended Wi-Fi signals (perhaps from a different SSID or arriving
late from a hidden node) are added to the noise.

Unlike ``YansWifiPhy``, where there are no foreign signals, CCA_BUSY state
will be raised for foreign signals that are higher than CcaEdThreshold
(see section 16.4.8.5 in the 802.11-2012 standard for definition of
CCA Mode 1).  The attribute ``WifiPhy::CcaEdThreshold`` therefore
potentially plays a larger role in this model than in the ``YansWifiPhy``
model.

To support the Spectrum channel, the ``YansWifiPhy`` transmit and receive methods
were adapted to use the Spectrum channel API.  This required developing
a few ``SpectrumModel``-related classes.  The class
``WifiSpectrumValueHelper`` is used to create Wi-Fi signals with the
spectrum framework and spread their energy across the bands. The
spectrum is sub-divided into sub-bands (the width of an OFDM
subcarrier, which depends on the technology). The power allocated to a particular channel
is spread across the sub-bands roughly according to how power would
be allocated to sub-carriers. Adjacent channels are models by the use of
OFDM transmit spectrum masks as defined in the standards.

To support an easier user configuration experience, the existing
YansWifi helper classes (in ``src/wifi/helper``) were copied and
adapted to provide equivalent SpectrumWifi helper classes.

Finally, for reasons related to avoiding C++ multiple inheritance
issues, a small forwarding class called ``WifiSpectrumPhyInterface``
was inserted as a shim between the ``SpectrumWifiPhy`` and the
Spectrum channel.  The ``WifiSpectrumPhyInterface`` calls a different
``SpectrumWifiPhy::StartRx ()`` method to start the reception process.
This method performs the check of the signal power against the
``WifiPhy::RxSensitivity`` attribute and discards weak signals, and
also checks if the signal is a Wi-Fi signal; non-Wi-Fi signals are added
to the InterferenceHelper and can raise CCA_BUSY but are not further processed
in the reception chain.   After this point, valid Wi-Fi signals cause
``WifiPhy::StartReceivePreamble`` to be called, and the processing continues
as described above.

The MAC model
=============

Infrastructure association
##########################

Association in infrastructure mode is a high-level MAC function.
Either active probing or passive scanning is used (default is passive scan).
At the start of the simulation, Wi-Fi network devices configured as
STA will attempt to scan the channel. Depends on whether passive or active
scanning is selected, STA will attempt to gather beacons, or send a probe
request and gather probe responses until the respective timeout occurs. The
end result will be a list of candidate AP to associate to. STA will then try
to associate to the best AP (i.e., best SNR).

If association is rejected by the AP for some reason, the STA will try to
associate to the next best AP until the candidate list is exhausted which
then sends STA to 'REFUSED' state. If this occurs, the simulation user will
need to force reassociation retry in some way, perhaps by changing
configuration (i.e. the STA will not persistently try to associate upon a
refusal).

When associated, if the configuration is changed by the simulation user,
the STA will try to reassociate with the existing AP.

If the number of missed beacons exceeds the threshold, the STA will notify
the rest of the device that the link is down (association is lost) and
restart the scanning process. Note that this can also happen when an
association request fails without explicit refusal (i.e., the AP fails to
respond to association request).

Roaming
#######

Roaming at layer-2 (i.e. a STA migrates its association from one AP to
another) is not presently supported. Because of that, the Min/Max channel
dwelling time implementation as described by the IEEE 802.11 standard
[ieee80211]_ is also omitted, since it is only meaningful on the context
of channel roaming.

Channel access
##############

The 802.11 Distributed Coordination Function is used to calculate when to grant
access to the transmission medium. While implementing the DCF would have been
particularly easy if we had used a recurring timer that expired every slot, we
chose to use the method described in [ji2004sslswn]_
where the backoff timer duration is lazily calculated whenever needed since it
is claimed to have much better performance than the simpler recurring timer
solution.

The DCF basic access is described in section 10.3.4.2 of [ieee80211-2016]_.

*  “A STA may transmit an MPDU when it is operating under the DCF access method
   [..] when the STA determines that the medium is idle when a frame is queued
   for transmission, and remains idle for a period of a DIFS, or an EIFS
   (10.3.2.3.7) from the end of the immediately preceding medium-busy event,
   whichever is the greater, and the backoff timer is zero. Otherwise the random
   backoff procedure described in 10.3.4.3 shall be followed."

Thus, a station is allowed not to invoke the backoff procedure if all of the
following conditions are met:

*  the medium is idle when a frame is queued for transmission
*  the medium remains idle until the most recent of these two events: a DIFS
   from the time when the frame is queued for transmission; an EIFS from the
   end of the immediately preceding medium-busy event (associated with the
   reception of an erroneous frame)
*  the backoff timer is zero

The backoff procedure of DCF is described in section 10.3.4.3 of [ieee80211-2016]_.

*  “A STA shall invoke the backoff procedure to transfer a frame
   when finding the medium busy as indicated by either the physical or
   virtual CS mechanism.”
*  “A backoff procedure shall be performed immediately after the end of
   every transmission with the More Fragments bit set to 0 of an MPDU of
   type Data, Management, or Control with subtype PS-Poll, even if no
   additional transmissions are currently queued.”

The EDCA backoff procedure is slightly different than the DCF backoff procedure
and is described in section 10.22.2.2 of [ieee80211-2016]_. The backoff procedure
shall be invoked by an EDCAF when any of the following events occur:

*  a frame is "queued for transmission such that one of the transmit queues
   associated with that AC has now become non-empty and any other transmit queues
   associated with that AC are empty; the medium is busy on the primary channel"
*  "The transmission of the MPDU in the final PPDU transmitted by the TXOP holder
   during the TXOP for that AC has completed and the TXNAV timer has expired, and
   the AC was a primary AC"
*  "The transmission of an MPDU in the initial PPDU of a TXOP fails [..] and the
   AC was a primary AC"
*  "The transmission attempt collides internally with another EDCAF of an AC that
   has higher priority"
*  (optionally) "The transmission by the TXOP holder of an MPDU in a non-initial
   PPDU of a TXOP fails"

Additionally, section 10.22.2.4 of [ieee80211-2016]_ introduces the notion of
slot boundary, which basically occurs following SIFS + AIFSN * slotTime of idle
medium after the last busy medium that was the result of a reception of a frame
with a correct FCS or following EIFS - DIFS + AIFSN * slotTime + SIFS of idle
medium after the last indicated busy medium that was the result of a frame reception
that has resulted in FCS error, or following a slotTime of idle medium occurring
immediately after any of these conditions.

On these specific slot boundaries, each EDCAF shall make a determination to perform
one and only one of the following functions:

*  Decrement the backoff timer.
*  Initiate the transmission of a frame exchange sequence.
*  Invoke the backoff procedure due to an internal collision.
*  Do nothing.

Thus, if an EDCAF decrements its backoff timer on a given slot boundary and, as
a result, the backoff timer has a zero value, the EDCAF cannot immediately
transmit, but it has to wait for another slotTime of idle medium before transmission
can start.

The higher-level MAC functions are implemented in a set of other C++ classes and
deal with:

* packet fragmentation and defragmentation,
* use of the RTS/CTS protocol,
* rate control algorithm,
* connection and disconnection to and from an Access Point,
* the MAC transmission queue,
* beacon generation,
* MSDU aggregation,
* etc.

Frame Exchange Managers
#######################
As the IEEE 802.11 standard evolves, more and more features are added and it is
more and more difficult to have a single component handling all of the allowed
frame exchange sequences. A hierarchy of FrameExchangeManager classes has been
introduced to make the code clean and scalable, while avoiding code duplication.
Each FrameExchangeManager class handles the frame exchange sequences introduced
by a given amendment. The FrameExchangeManager hierarchy is depicted in Figure
:ref:`fem-hierarchy`.

.. _fem-hierarchy:

.. figure:: figures/FemHierarchy.*

   *FrameExchangeManager hierarchy*

The features supported by every FrameExchangeManager class are as follows:

* ``FrameExchangeManager`` is the base class. It handles the basic sequences
  for non-QoS stations: MPDU followed by Normal Ack, RTS/CTS and CTS-to-self,
  NAV setting and resetting, MPDU fragmentation
* ``QosFrameExchangeManager`` adds TXOP support: multiple protection setting,
  TXOP truncation via CF-End, TXOP recovery, ignore NAV when responding to an
  RTS sent by the TXOP holder
* ``HtFrameExchangeManager`` adds support for Block Ack (compressed variant),
  A-MSDU and A-MPDU aggregation, Implicit Block Ack Request policy
* ``VhtFrameExchangeManager`` adds support for S-MPDUs
* ``HeFrameExchangeManager`` adds support for the transmission and reception of
  multi-user frames via DL OFDMA and UL OFDMA, as detailed below.

.. _wifi-mu-ack-sequences:

Multi-user transmissions
########################

Since the introduction of the IEEE 802.11ax amendment, multi-user (MU) transmissions are
possible, both in downlink (DL) and uplink (UL), by using OFDMA and/or MU-MIMO. Currently,
ns-3 only supports multi-user transmissions via OFDMA. Three acknowledgment sequences are
implemented for DL OFDMA.

The first acknowledgment sequence is made of multiple BlockAckRequest/BlockAck frames sent
as single-user frames, as shown in Figure :ref:`fig-ack-su-format-80211ax`.

.. _fig-ack-su-format-80211ax:

.. figure:: figures/ack-su-format.*
   :align: center

   Acknowledgment of DL MU frames in single-user format

For the second acknowledgment sequence, an MU-BAR Trigger Frame is sent (as a single-user
frame) to solicit BlockAck responses sent in TB PPDUs, as shown in Figure :ref:`fig-mu-bar-80211ax`.

.. _fig-mu-bar-80211ax:

.. figure:: figures/mu-bar.*
   :align: center

   Acknowledgment of DL MU frames via MU-BAR Trigger Frame sent as single-user frame

For the third acknowledgment sequence, an MU-BAR Trigger Frame is aggregated to every PSDU
included in the DL MU PPDU and the BlockAck responses are sent in TB PPDUs, as shown in
Figure :ref:`fig-aggr-mu-bar-80211ax`.

.. _fig-aggr-mu-bar-80211ax:

.. figure:: figures/aggr-mu-bar.*
   :align: center

   Acknowledgment of DL MU frames via aggregated MU-BAR Trigger Frames

For UL OFDMA, both BSRP Trigger Frames and Basic Trigger Frames are supported, as shown in
Figure :ref:`fig-ul-ofdma-80211ax`. A BSRP Trigger Frame is sent by an AP to solicit stations
to send QoS Null frames containing Buffer Status Reports. A Basic Trigger Frame is sent by an AP
to solicit stations to send data frames in TB PPDUs, which are acknowledged by the AP via a
Multi-STA BlockAck frame.

.. _fig-ul-ofdma-80211ax:

.. figure:: figures/ul-ofdma.*
   :align: center

   Acknowledgment of DL MU frames via aggregated MU-BAR Trigger Frames

Multi-User Scheduler
####################

A new component, named **MultiUserScheduler**, is in charge of determining what frame exchange
sequence the aggregated AP has to perform when gaining a TXOP (DL OFDMA, UL OFDMA or BSRP Trigger
Frame), along with the information needed to perform the selected frame exchange sequence (e.g.,
the set of PSDUs to send in case of DL OFDMA). ``MultiUserScheduler`` is an abstract base class.
Currently, the only available subclass is **RrMultiUserScheduler**. By default, no multi-user
scheduler is aggregated to an AP (hence, OFDMA is not enabled).

Round-robin Multi-User Scheduler
################################
The Round-robin Multi-User Scheduler dynamically assigns a priority to each station to ensure
airtime fairness in the selection of stations for DL multi-user transmissions. The ``NStations``
attribute enables to set the maximum number of stations that can be the recipients of a DL
multi-user frame. Therefore, every time an HE AP accesses the channel to transmit a DL
multi-user frame, the scheduler determines the number of stations the AP has frames to send
to (capped at the value specified through the mentioned attribute) and attempts to allocate
equal sized RUs to as many such stations as possible without leaving RUs of the same size
unused. For instance, if the channel bandwidth is 40 MHz and the determined number of stations
is 5, the first 4 stations (in order of priority) are allocated a 106-tone RU each (if 52-tone
RUs were allocated, we would have three 52-tone RUs unused). If central 26-tone RUs can be
allocated (as determined by the ``UseCentral26TonesRus`` attribute), possible stations that
have not been allocated an RU are assigned one of such 26-tone RU. In the previous example,
the fifth station would have been allocated one of the two available central 26-tone RUs.

When UL OFDMA is enabled (via the ``EnableUlOfdma`` attribute), every DL OFDMA frame exchange
is followed by an UL OFDMA frame exchange involving the same set of stations and the same RU
allocation as the preceding DL multi-user frame. The transmission of a BSRP Trigger Frame can
optionally (depending on the value of the ``EnableBsrp`` attribute) precede the transmission
of a Basic Trigger Frame in order for the AP to collect information about the buffer status
of the stations.

Ack manager
###########

Since the introduction of the IEEE 802.11e amendment, multiple acknowledgment policies
are available, which are coded in the Ack Policy subfield in the QoS Control field of
QoS Data frames (see Section 9.2.4.5.4 of the IEEE 802.11-2016 standard). For instance,
an A-MPDU can be sent with the *Normal Ack or Implicit Block Ack Request* policy, in which
case the receiver replies with a Normal Ack or a Block Ack depending on whether the A-MPDU
contains a single MPDU or multiple MPDUs, or with the *Block Ack* policy, in which case
the receiver waits to receive a Block Ack Request in the future to which it replies with
a Block Ack.

``WifiAckManager`` is the abstract base class introduced to provide an interface
for multiple ack managers. Currently, the default ack manager is
the ``WifiDefaultAckManager``.

WifiDefaultAckManager
#####################

The ``WifiDefaultAckManager`` allows to determine which acknowledgment policy
to use depending on the value of its attributes:

* ``UseExplicitBar``: used to determine the ack policy to use when a response is needed from
  the recipient and the current transmission includes multiple frames (A-MPDU) or there are
  frames transmitted previously for which an acknowledgment is needed. If this attribute is
  true, the *Block Ack* policy is used. Otherwise, the *Implicit Block Ack Request* policy is used.
* ``BaThreshold``: used to determine when the originator of a Block Ack agreement needs to
  request a response from the recipient. A value of zero means that a response is requested
  at every frame transmission. Otherwise, a non-zero value (less than or equal to 1) means
  that a response is requested upon transmission of a frame whose sequence number is distant
  at least BaThreshold multiplied by the transmit window size from the starting sequence
  number of the transmit window.
* ``DlMuAckSequenceType``: used to select the acknowledgment sequence for DL MU frames
  (acknowledgment in single-user format, acknowledgment via MU-BAR Trigger Frame sent as
  single-user frame, or acknowledgment via MU-BAR Trigger Frames aggregated to the data
  frames).

Protection manager
##################

The protection manager is in charge of determining the protection mechanism to use,
if any, when sending a frame.

``WifiProtectionManager`` is the abstract base class introduced to provide an interface
for multiple protection managers. Currently, the default protection manager is
the ``WifiDefaultProtectionManager``.

WifiDefaultProtectionManager
############################

The ``WifiDefaultProtectionManager`` selects a protection mechanism based on the
information provided by the remote station manager.

Rate control algorithms
#######################

Multiple rate control algorithms are available in |ns3|.
Some rate control algorithms are modeled after real algorithms used in real devices;
others are found in literature.
The following rate control algorithms can be used by the MAC low layer:

Algorithms found in real devices:

* ``ArfWifiManager``
* ``OnoeWifiManager``
* ``ConstantRateWifiManager``
* ``MinstrelWifiManager``
* ``MinstrelHtWifiManager``

Algorithms in literature:

* ``IdealWifiManager``  (default for ``WifiHelper``)
* ``AarfWifiManager`` [lacage2004aarfamrr]_
* ``AmrrWifiManager`` [lacage2004aarfamrr]_
* ``CaraWifiManager`` [kim2006cara]_
* ``RraaWifiManager`` [wong2006rraa]_
* ``AarfcdWifiManager`` [maguolo2008aarfcd]_
* ``ParfWifiManager`` [akella2007parf]_
* ``AparfWifiManager`` [chevillat2005aparf]_
* ``ThompsonSamplingWifiManager`` [krotov2020rate]_

ConstantRateWifiManager
#######################

The constant rate control algorithm always uses the same
transmission mode for every packet. Users can set a desired
'DataMode' for all 'unicast' packets and 'ControlMode' for all
'request' control packets (e.g. RTS).

To specify different data mode for non-unicast packets, users
must set the 'NonUnicastMode' attribute of the
WifiRemoteStationManager.  Otherwise, WifiRemoteStationManager
will use a mode with the lowest rate for non-unicast packets.

The 802.11 standard is quite clear on the rules for selection
of transmission parameters for control response frames (e.g.
CTS and ACK).  |ns3| follows the standard and selects the rate
of control response frames from the set of basic rates or
mandatory rates. This means that control response frames may
be sent using different rate even though the ConstantRateWifiManager
is used.  The ControlMode attribute of the ConstantRateWifiManager
is used for RTS frames only.  The rate of CTS and ACK frames are
selected according to the 802.11 standard.  However, users can still
manually add WifiMode to the basic rate set that will allow control
response frames to be sent at other rates.  Please consult the
`project wiki <https://www.nsnam.org/wiki/HOWTO_add_basic_rates_to_802.11>`_ on how to do this.

Available attributes:

* DataMode (default WifiMode::OfdmRate6Mbps): specify a mode for
  all non-unicast packets
* ControlMode (default WifiMode::OfdmRate6Mbps): specify a mode for
  all 'request' control packets

IdealWifiManager
################

The ideal rate control algorithm selects the best
mode according to the SNR of the previous packet sent.
Consider node *A* sending a unicast packet to node *B*.
When *B* successfully receives the packet sent from *A*,
*B* records the SNR of the received packet into a ``ns3::SnrTag``
and adds the tag to an ACK back to *A*.
By doing this, *A* is able to learn the SNR of the packet sent to *B*
using an out-of-band mechanism (thus the name 'ideal').
*A* then uses the SNR to select a transmission mode based
on a set of SNR thresholds, which was built from a target BER and
mode-specific SNR/BER curves.

Available attribute:

* BerThreshold (default 1e-6): The maximum Bit Error Rate
  that is used to calculate the SNR threshold for each mode.

Note that the BerThreshold has to be low enough to select a robust enough MCS
(or mode) for a given SNR value, without being too restrictive on the target BER.
Indeed we had noticed that the previous default value (i.e. 1e-5) led to the
selection of HE MCS-11 which resulted in high PER.
With this new default value (i.e. 1e-6), a HE STA moving away from a HE AP has
smooth throughput decrease (whereas with 1e-5, better performance was seen further
away, which is not "ideal").

ThompsonSamplingWifiManager
###########################

Thompson Sampling (TS) is a classical solution to the Multi-Armed
Bandit problem.  `ThompsonSamplingWifiManager` implements a rate
control algorithm based on TS with the goal of providing a simple
statistics-based algorithm with a low number of parameters.

The algorithm maintains the number of successful transmissions
:math:`\alpha_i` and the number of unsuccessful transmissions
:math:`\beta_i` for each MCS :math:`i`, both of which are initially
set to zero.

To select MCS for a data frame, the algorithm draws a sample frame
success rate :math:`q_i` from the beta distribution with shape
parameters :math:`(1 + \alpha_i, 1 + \beta_i)` for each MCS and then
selects MCS with the highest expected throughput calculated as the
sample frame success rate multiplied by MCS rate.

To account for changing channel conditions, exponential decay is
applied to :math:`\alpha_i` and :math:`\beta_i`. The rate of
exponential decay is controlled with the `Decay` attribute which is
the inverse of the time constant. Default value of 1 Hz results in
using exponential window with the time constant of 1 second.  Setting
this value to zero effectively disables exponential decay and can be
used in static scenarios.

Control frames are always transmitted using the most robust MCS,
except when the standard specifies otherwise, such as for ACK frames.

As the main goal of this algorithm is to provide a stable baseline, it
does not take into account backoff overhead, inter-frame spaces and
aggregation for MCS rate calculation. For an example of a more complex
statistics-based rate control algorithm used in real devices, consider
Minstrel-HT described below.

MinstrelWifiManager
###################

The minstrel rate control algorithm is a rate control algorithm originated from
madwifi project. It is currently the default rate control algorithm of the Linux kernel.

Minstrel keeps track of the probability of successfully sending a frame of each available rate.
Minstrel then calculates the expected throughput by multiplying the probability with the rate.
This approach is chosen to make sure that lower rates are not selected in favor of the higher
rates (since lower rates are more likely to have higher probability).

In minstrel, roughly 10 percent of transmissions are sent at the so-called lookaround rate.
The goal of the lookaround rate is to force minstrel to try higher rate than the currently used rate.

For a more detailed information about minstrel, see [linuxminstrel]_.

MinstrelHtWifiManager
#####################

This is the extension of minstrel for 802.11n/ac/ax.

802.11ax OBSS PD spatial reuse
##############################

802.11ax mode supports OBSS PD spatial reuse feature.
OBSS PD stands for Overlapping Basic Service Set Preamble-Detection.
OBSS PD is an 802.11ax specific feature that allows a STA, under specific conditions,
to ignore an inter-BSS PPDU.

OBSS PD Algorithm
#################

``ObssPdAlgorithm`` is the base class of OBSS PD algorithms.
It implements the common functionalities. First, it makes sure the necessary callbacks are setup.
Second, when a PHY reset is requested by the algorithm, it performs the computation to determine the TX power
restrictions and informs the PHY object.

The PHY keeps tracks of incoming requests from the MAC to get access to the channel.
If a request is received and if PHY reset(s) indicating TX power limitations occured
before a packet was transmitted, the next packet to be transmitted will be sent with
a reduced power. Otherwise, no TX power restrictions will be applied.

Constant OBSS PD Algorithm
##########################

Constant OBSS PD algorithm is a simple OBSS PD algorithm implemented in the ``ConstantObssPdAlgorithm`` class.

Once a HE preamble and its header have been received by the PHY, ``ConstantObssPdAlgorithm::
ReceiveHeSig`` is triggered.
The algorithm then checks whether this is an OBSS frame by comparing its own BSS color with the BSS color of the received preamble.
If this is an OBSS frame, it compares the received RSSI with its configured OBSS PD level value. The PHY then gets reset to IDLE
state in case the received RSSI is lower than that constant OBSS PD level value, and is informed about a TX power restrictions.

Note: since our model is based on a single threshold, the PHY only supports one restricted power level.

Modifying Wifi model
####################

Modifying the default wifi model is one of the common tasks when performing research.
We provide an overview of how to make changes to the default wifi model in this section.
Depending on your goal, the common tasks are (in no particular order):

* Creating or modifying the default Wi-Fi frames/headers by making changes to ``wifi-mac-header.*``.
* MAC low modification. For example, handling new/modified control frames (think RTS/CTS/ACK/Block ACK),
  making changes to two-way transaction/four-way transaction.  Users usually make changes to
  ``frame-exchange-manager.*`` or its subclasses to accomplish this.
  Handling of control frames is performed in ``FrameExchangeManager::ReceiveMpdu``.
* MAC high modification. For example, handling new management frames (think beacon/probe),
  beacon/probe generation.  Users usually make changes to ``wifi-mac.*``,``sta-wifi-mac.*``, ``ap-wifi-mac.*``, or ``adhoc-wifi-mac.*`` to accomplish this.
* Wi-Fi queue management.  The files ``txop.*`` and ``qos-txop.*`` are of interest for this task.
* Channel access management.  Users should modify the files ``channel-access-manager.*``, which grant access to
  ``Txop`` and ``QosTxop``.
* Fragmentation and RTS threholds are handled by Wi-Fi remote station manager.  Note that Wi-Fi remote
  station manager simply indicates if fragmentation and RTS are needed.  Fragmentation is handled by
  ``Txop`` or ``QosTxop`` while RTS/CTS transaction is handled by ``FrameExchangeManager``.
* Modifying or creating new rate control algorithms can be done by creating a new child class of Wi-Fi remote
  station manager or modifying the existing ones.
