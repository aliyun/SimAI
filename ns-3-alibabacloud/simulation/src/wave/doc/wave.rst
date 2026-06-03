WAVE models
-----------
WAVE is a system architecture for wireless-based vehicular communications, 
specified by the IEEE.  This chapter documents available models for WAVE
within |ns3|.  The focus is on the MAC layer and MAC extension layer
defined by [ieee80211p]_ and [ieee1609dot4]_.

.. include:: replace.txt

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)

Model Description
*****************

WAVE is an overall system architecture for vehicular communications.
The standards for specifying WAVE include a set of extensions to the IEEE
802.11 standard, found in IEEE Std 802.11p-2010 [ieee80211p]_, and
the IEEE 1609 standard set, consisting of four documents:  
resource manager:  IEEE 1609.1 [ieee1609dot1]_,
security services:  IEEE 1609.2 [ieee1609dot2]_,
network and transport layer services:  IEEE 1609.3 [ieee1609dot3]_,
and multi-channel coordination:  IEEE 1609.4 [ieee1609dot4]_.
Additionally, SAE standard J2735 [saej2735]_ describes a Dedicated
Short Range Communications (DSRC) application message set that allows
applications to transmit information using WAVE. 

In |ns3|, the focus of the ``wave`` module is on both the MAC layer and the 
multi-channel coordination layer.
The key design aspect of 802.11p-compilant MAC layer is that they allow
communications outside the context of a basic service set (BSS).
The literature uses the acronym OCB to denote "outside the context
of a BSS", and the class ``ns3::OcbWifiMac`` models this in |ns3|.
This MAC does not require any association between devices (similar to an 
adhoc WiFi MAC). Many management frames will not be used, but when used, the BSSID field 
needs to be set to a wildcard BSSID value. Management information is 
transmitted by what is called a vendor specific action (VSA) frame. With these 
changes, the packet transmissions (for a moving vehicle) can be fast with 
small delay in the MAC layer. Users can create IEEE802.11p-compliant device 
(the object of the class ``ns3::WifiNetDevice`` associating with 
``ns3::OcbWifiMac``) .

The key design aspect of the WAVE-compilant MAC layer (including 802.11p MAC 
layer and 1609.4 MAC extension layer) is that, based on OCB features, they 
provide devices with the capability of switching between control and service channels, using a single radio or using multiple radios. 
Therefore devices can communicate with others in single or multiple 
channels, which can 
support both safety related and non-safety related service for vehicular environments. 

At the physical layer, the biggest difference is the use of the 5.9 GHz band 
with a channel bandwidth of 10 MHz.  These physical
layer changes can make the wireless signal relatively more stable,
without degrading throughput too much (ranging from 3 Mbps to 27 Mbps).

The source code for the WAVE MAC models lives in the directory
``src/wave``.

For better modeling WAVE and VANET, the WAVE models for high layers 
(mainly [ieee1609dot3]_ ) are planned for a later patch.

Design
======

In |ns3|, support for 802.11p involves the MAC and PHY layers.
To use an 802.11p NetDevice, ``ns3::Wifi80211pHelper`` is suggested.

In |ns3|, support for WAVE involves the MAC, its MAC extension and PHY layers.
To use a WAVE NetDevice, ``ns3::WaveHelper`` is suggested.

MAC layer
#########

The classes used to model the MAC layer are ``ns3::OrganizationIdentifier``,
``ns3::VendorSpecificActionHeader`` and ``ns3::OcbWifiMac``.

The OrganizationIdentifier and VendorSpecificActionHeader are used to support 
the sending of a Vendor Specific Action frame.

OcbWifiMac is very similar to AdhocWifiMac, with some modifications. 
The |ns3| AdhocWifiMac class is implemented very close to the 802.11p OCB 
mode rather than a real 802.11 ad-hoc mode. The AdhocWifiMac has no BSS 
context that is defined in 802.11 standard, so it will not take time to 
send beacons and to authenticate, making its behavior similar to that
of an OcbWifiMac.

1. SetBssid, GetBssid, SetSsid, GetSsid

  These methods are related to 802.11 BSS context, and are unused in the OCB context.

2. SetLinkUpCallback, SetLinkDownCallback
 
   WAVE device can send packets directly, so the WiFi link is never down.

3. SendVsc, AddReceiveVscCallback

   WAVE management information shall be sent by vendor specific action frames, 
   sent by the upper layer 1609.4 standard as WSA
   (WAVE Service Advertisement) packets or other vendor specific information.

4. SendTimingAdvertisement (not implemented)

   Although Timing Advertisement is very important and specifically defined in 
   802.11p standard, it is not useful in a simulation environment. 
   Every node in |ns3| vehicular simulation is assumed to be already time 
   synchronized (perhaps by GPS).

5. ConfigureEdca

   This method will allow the user to set EDCA parameters of WAVE channels 
   including CCH ans SCHs. And the OcbWifiMac itself also uses this method 
   to configure default 802.11p EDCA parameters.

6. Wildcard BSSID

   The Wildcard BSSID is set to "ff:ff:ff:ff:ff:ff".
   As defined in IEEE 802.11-2007, a wildcard BSSID shall not be used in the
   BSSID field except for management frames of subtype probe request. But Adhoc 
   mode of |ns3| simplifies this mechanism:  when stations receive packets, 
   they will be forwarded up to the higher layer, regardless of BSSID. 
   This process is very close 
   to OCB mode as defined in 802.11p-2010, in which stations use the wildcard 
   BSSID to allow the higher layer of other stations to hear directly.

7. Enqueue, Receive

   The most important methods are send and receive methods. According to the 
   standard, we should filter the frames that are not permitted. Thus here we 
   just identify the frames we care about; the other frames will be discarded.

MAC extension layer
###################

Although 1609.4 is still in the MAC layer, the implementation 
approach for |ns3| does not do much modification in the
source code of the wifi module. Instead, if some feature is related 
to wifi MAC classes, then a relevant subclass is defined; if some 
feature has no relation to wifi MAC classes, then a new class 
will be defined. This approach was selected to be non-intrusive to the 
|ns3| wifi module. All of these classes will be hosted in a 'container'
class called ``ns3:: WaveNetDevice``. This class is a subclass inherting 
from ``ns3::NetDeivce``, composed of the objects of 
``ns3::ChannelScheduler``, ``ns3::ChannelManager``, 
``ns3::ChannelCoordinator`` and ``ns3::VsaManager`` 
classes to provide the features described in 1609.4 while still 
containing the objects of ``ns3::OcbWifiMac`` and ``ns3::WifiPhy`` 
classes.  Moreover, ``ns3::OcbWifiMac`` class is further extended with 
support for IEEE 1609.4 associating with ``ns3::HigherLayerTxVectorTag`` 
and ``ns3::WaveMacLow``. The main work of the WaveNetDevice is to create 
objects, configure, check arguments and provide new APIs for multiple 
channel operation as follows:

1. AddMac, GetMac and GetMacs

  Different from ``ns3::WifiNetDevice``, the WAVE device will have 
  multiple internal MAC entities rather than a single one. Each MAC 
  entity is used to support each WAVE channel. Thus, when 
  devices switch from the current channel to the next channel in different 
  channel intervals, the packets in the internal queue will not be
  flushed and the current MAC entity will perform a suspend operation 
  until woken up in next appropriate channel interval.

2. AddPhy, GetPhy and GetPhys

  Also in contrast to ``ns3::WifiNetDevice``,  the WAVE device here 
  can allow more than one PHY entity, which permits the use cases of
  of single-PHY devices or multiple-PHY devices.

3. SetChannelScheduler and GetChannelScheduler

  How to deal with multiple MAC entities and PHY entities to assign 
  channel access for different requests from higher layer? IEEE 
  1609.4 [ieee1609dot4]_ does not seem to give a very clear and detailed 
  mechanism, deferring to the implementor. In this model, the class 
  ``ns3::ChannelScheduler`` provides the API and delegates to the subclasses
  to implement the virtual methods. In the current implementation, the default 
  assignment mechanism for channel access, 
  called ``ns3::DefaultChannelScheduler``, gives a simple answer that only
  deals with multiple channel operation in the context of a single-PHY device. 
  If users define their own different assignment mechanisms such as in the 
  context of two PHY entities, they can easily reuse models using AddPhy and 
  SetChannelScheduler methods to import a new assignment mechanism.

4. SetChannelManager and GetChannelManager

  class ``ns3::ChannelManager`` is a WAVE channel set which contains 
  valid WAVE channel numbers. Moreover, the tx information in this channel 
  set such as data rate and tx power level is used for transmitting management frames.

5. SetVsaManager and GetVsaManager

  class ``ns3::VsaManager`` is used to deal with sending and receiving 
  VSA frames. According to different request parameters from the higher layer, 
  this class may transmit VSA frames repeatedly in the appropriate channel 
  number and channel interval.

6. SetChannelCoordinator and GetChannelCoordinator

  class ``ns3::ChannelCoordinator`` is used to deal with channel coordination. 
  The WAVE device can be aware of the channel interval at the current time or 
  in the future.  It can also notify listeners about incoming channel coordination 
  events. Generally this class is used in the case of assigning alternating CCH and SCH access.

7. StartSch and StopSch

  In contrast to the basic 802.11p device that allow transmission packets 
  immediately after 
  the device is created, the WAVE device should assign channel access 
  for sending packets.  This method will call class ``ns3::ChannelScheduler`` 
  to assign radio resources for the relevant channel.

8. ChangeAddress

  The WAVE device can support a change of address after devices are already 
  initialized, which will cause all of MAC entities reset their status.

9. CancelTx

  The WAVE device can support a request to cancel all transmissions associated 
  with the particular category and channel number, which will reset the 
  particular interval queue and drop all of the queued packets in this queue.

10. RegisterTxProfile and DeleteTxProfile

  After channel access is assigned, we still cannot send IP-based 
  (or other protocol) packets by the Send () method. A tx profile should 
  be registered to specify tx parameters before transmission. 

11. StartVsa, StopVsa and SetWaveVsaCallback

  These methods will call an object from class ``ns3::VsaManager`` to send 
  and receive VSA frames.  Generally these methods are used by IEEE 1609.3 
  for WSA management information.

12. SendX

  After channel access is assigned, we can send WSMP (or other protocol) 
  packets via the SendX () method. We should specify the tx parameters for 
  each packet, e.g. the channel number for transmit.

13. Send and SetReceiveCallback

  This method is the abstract method defined in the parent class 
  ``ns3::NetDevice``, defined to allow the sending of IP-based packets. 
  The channel access should be already assigned and tx profile should 
  be registered, otherwise incoming packets from the higher layer will be 
  discarded. No matter whether packets are sent by Send method or SendX 
  method, the received packets will be only be delivered to the higher layer 
  by the registered ReceiveCallback.

14. other methods from its parent class ``ns3::NetDevice``

  These methods are implemented very similar to the code in ``ns3::WifiNetDevice``.

In the above numbered list, we can categorize the methods into three types:
the first type, from 1 to 6 and also 14, is the configuration for modeling and 
creating a WAVE device; the second type, from 7 to 11, is 
the management plane of the standard; and the third type, 12 and 13, 
is the data plane of the standard.

Channel coordination

The class ``ns3::ChannelCoordinator`` defines the CCH Interval, SCH Interval and GuardInteval. Users can be aware of which interval the current time or 
future time will be in. If channel access mode is assigned to 
alternating CCH and SCH access, 
channel interval events will be notified repeatedly for class 
``ns3::ChannelCoordinator`` to switch channels.  Current default values are 
for CCHI with 50ms interval, SCHI with 50ms interval, and GuardI with 4ms interval. Users can change these values by configuring the class attributes. 

Channel routing

Channel routing service means different transmission approaches for WSMP data, 
IP datagram and management information.
For WSMP data, the SendX () method implements the service primitive 
MA-UNITDATAX, and users can 
set transmission parameters for each individual packet. The parameters include 
channel number, priority,
data rate and tx power level (expiration time is not supported now). 
For IP datagrams, the Send () method is a virtual method from ``ns3::NetDevice`` that implements the service primitive MA-UNITDATA.
Users should insert QoS tags into packets themselves if they want to use QoS. 
Moreover, a tx profile should be registered
before the Send method is called for transmit; the profile contains SCH number, data rate, power level and adaptable mode. 
For management information, StartVsa method implements the service primitive 
MLMEX-VSA. The tx information is already configured 
in ``ns3::ChannelManager``, including data rate, power level and adaptable mode.

Channel access assignment

The channel access assignment is done by class ``ns3::ChannelScheduler`` to assign ContinuousAccess, ExtendedAccess 
and AlternatingAccess. Besides that, immediate access is achieved by enabling 
the "immediate" parameter, in which case
the request channel will be switched to immediately.  However this class is a 
virtual parent class.  The current module provides a
subclass ``ns3::DefaultChannelScheduler`` to assign channel access in the context of a single-PHY device. In this subclass, if the channel 
access is already assigned for another request, the next coming request will 
fail until the previous channel access is released.
Users can implement different assignment mechanisms to deal with multiple MAC entities and multiple PHY entities by 
inheriting from parent class ``ns3::ChannelScheduler``.
An important point is that channel access should be assigned before sending
routing packets, otherwise the packets will be discard.

Vendor Specific Action frames

When users want to send VSA repeatedly by calling WaveNetDevice::StartVsa, VSA will be sent repeatedly by 
``ns3::VsaManager``. It is worth noting that if the peer MAC address is a unicast address, the VSA can only 
be transmitted once even there is a repeat request. The tx parameters for VSA management frames can be obtained from the ``ns3::ChannelManager``.

User priority and Multi-channel synchronization

Since wifi module has already implemented a QoS mechanism, the wave module 
reuses the mechanism; VSA frames will be assigned the default value with the 
highest AC according to the standard.
Multiple-channel synchronization is very important in practice for devices 
without a local timing source. 
However, in simulation, every node is supposed to have the same system clock, which could be provided by GPS devices in a real environment, so this feature is not modelled in |ns3|.
During the guard interval, the device can only be in receive state, except 
for the switch state when the device does channel switching operation.

PHY layer
#########

No modification or extension is made to the |ns3| PHY layer
corresponding to this model.
In the 802.11p standard, the PHY layer wireless technology is still 80211a OFDM with a 10MHz channel width,
so Wifi80211pHelper will only allow the user to set the standard 
to WIFI_PHY_STANDARD_80211_10MHZ or WIFI_PHY_STANDARD_80211_20MHZ, 
while WaveHelper will only support WIFI_PHY_STANDARD_80211_10MHZ.
The maximum station transmit power and maximum permitted EIRP defined in 
802.11p is larger 
than that of WiFi, so transmit range can normally become longer than the
usual WiFi.  However, this feature will 
not be implemented. Users who want to obtain longer range should configure 
attributes "TxPowerStart", 
"TxPowerEnd" and "TxPowerLevels" of the YansWifiPhy class by themselves.

Scope and Limitations
=====================

1. Does the model involve vehicular mobility of some sort?

Vehicular networks involve not only communication protocols, but 
also a communication environment including vehicular mobility and 
propagation loss models. Because of specific features of the latter, 
the protocols need to change. The MAC layer model in this project just 
adapts MAC changes to vehicular environment. However this model 
does not involve any vehicular mobility with time limit. While users 
should understand that vehicular mobility is out of scope for the 
current WAVE module, they can use any mobility model in |ns3|.  For
example, users may use a ``ns3::RandomWaypointMobilityModel`` (RWP)
for node mobilty or may generate ns-2-style playback files using 
other third-party tools and then playback those mobility traces 
using ``ns3::Ns2MobilityHelper``.

2. Does this model use different propagation models?

Referring to the first issue, some more realistic propagation loss
models for vehicualr environment are suggested and welcome.  Some 
existing propagation los models in |ns3| are also suitable.  
Normally, users can use Friis, Two-Ray Ground, and Nakagami models.
The ``ns3::VanetRoutingExample`` example defaults to Two-Ray 
Ground propagation loss with no additional fading, although adding 
stochastic Nakagami-m fading is parametrically supported.

3. Are there any vehicular application models to drive the code?

About vehicular application models, SAE J2375 depends on WAVE 
architecture and is an application message set in US.  Cooperative Awareness
Messages (CAM) and Decentralized Environment Notification Messages (DENM) can
be sent Europe between network and application layer, and is 
very close to being an application model. The BSM in J2375 [saej2735] and CAM
send alert messages that every vehicle node will sent periodically about 
its status information to cooperate with others. The 
``ns3::VanetRoutingExample`` example sets up a network of (vehicular)
nodes that each broadcast BSMs at regular intervals and may additionally
attempt to route non-BSM data through the network using select IP-based
routing protocols.

5. Why are there two kinds of NetDevice helpers?

The current module provides two helpers to create two kinds of NetDevice. 
The first is an object of WifiNetDevice (802.11p device) which mainly 
contains class ``ns3::OcbWifiMac`` to enable OCB mode. The second 
is an object of WaveNetDevice (WAVE device) which contains additional 
classes ``ns3::ChannelScheduler``, ``ns3::ChannelManager``, 
``ns3::ChannelCoordinator`` and ``ns3::VsaManager`` to support 
multi-channel operation mode.  The reason to provide a special 802.11p 
device helper is that, considering the fact that many researchers are 
interested in routing protocols or other aspects of vehicular environment in
a single channel context, they need neither multi-channel operation nor 
WAVE architectures. 
Besides that, the European standard may also reuse an 802.11p device in an 
modified ITS-G5 implementation (maybe called ItsG5NetDevice).  Hence,
the model supports configuration of both types of devices.

References
==========

.. [ieee80211p] IEEE Std 802.11p-2010 "IEEE Standard for Information technology-- Local and metropolitan area networks-- Specific requirements-- Part 11: Wireless LAN Medium Access Control (MAC) and Physical Layer (PHY) Specifications Amendment 6: Wireless Access in Vehicular Environments"

.. [ieee1609dot1] IEEE Std 1609.1-2010 "IEEE Standard for Wireless Access in Vehicular Environments (WAVE) - Resource Manager, 2010"

.. [ieee1609dot2] IEEE Std 1609.2-2010 "IEEE Standard for Wireless Access in Vehicular Environments (WAVE) - Security Services for Applications and Management Messages, 2010"

.. [ieee1609dot3] IEEE Std 1609.3-2010 "IEEE Standard for Wireless Access in Vehicular Environments (WAVE) - Networking Services, 2010"

.. [ieee1609dot4] IEEE Std 1609.4-2010 "IEEE Standard for Wireless Access in Vehicular Environments (WAVE) - Multi-Channel Operation, 2010"

.. [saej2735] SAE Std J2735 "J2735 dedicated short range communications (DSRC) message set dictionary. 2009"

Usage
*****

Helpers
=======

The helpers include a) lower-level MAC and PHY channel helpers and 
b) higher-level application helpers that handle the sending and receiving
of the Basic Safety Message (BSM).

The lower-level helpers include ``ns3::YansWavePhyHelper``, ``ns3::NqosWaveMacHelper``, ``ns3::QosWaveMacHelper``, 
``ns3::Wifi80211pHelper`` and ``ns3::WaveHelper``. 

Wifi80211pHelper is used to create 
802.11p devices that follow the 802.11p-2010 standard. WaveHelper is 
used to create WAVE devices that follow both 802.11p-2010 and 1609.4-2010 
standards which are the MAC and PHY layers of the WAVE architecture. 

The relation of ``ns3::NqosWaveMacHelper``, ``ns3::QosWaveMacHelper`` and 
``ns3::Wifi80211pHelper`` is described as below:

::

    WifiHelper ------------use-------------->   WifiMacHelper
        ^                                        ^         ^
        |                                        |         |
        |                                        |         |
      inherit                                inherit      inherit
        |                                        |         |
    Wifi80211pHelper ------use----->  QosWaveMacHelper or NqosWaveHelper
 
From the above diagram, there are two Mac helper classes that both
inherit from the WifiMacHelper; when the WAVE module was originally
written, there were specialized versions (QoS and Nqos) of WifiMacHelper 
that have since been removed from the Wifi codebase, but the distinction 
remains in the WAVE helpers.  The functions of WiFi 802.11p device can be achieved by 
WaveNetDevice's ContinuousAccess assignment, Wifi80211pHelper is recommended
if there is no need for multiple channel operation.
Usage is as follows:

::

    NodeContainer nodes;
    NetDeviceContainer devices;
    nodes.Create (2);
    YansWifiPhyHelper wifiPhy = YansWifiPhyHelper::Default ();
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
    wifiPhy.SetChannel (wifiChannel.Create ());
    NqosWave80211pMacHelper wifi80211pMac = NqosWaveMacHelper::Default();
    Wifi80211pHelper 80211pHelper = Wifi80211pHelper::Default ();
    devices = 80211pHelper.Install (wifiPhy, wifi80211pMac, nodes);
    
The relation of  ``ns3::YansWavePhyHelper``, ``ns3::QosWaveMacHelper`` and ``ns3::WaveHelper`` 
is described as below:
 
::
  
                                            WifiMacHelper
                                                  ^
                                                  | 
                                                inherit
                                                  | 
    WaveHelper -------- only use --------> QosWaveMacHelper
      
From the above diagram, WaveHelper is not the subclass of WifiHelper and should only 
use QosWaveMacHelper because WAVE MAC layer is based on QoS mechanism. But 
the WaveHelper is recommended if there is a need for multiple channel operation.
Usage is as follows:

::

    NodeContainer nodes;
    NetDeviceContainer devices;
    nodes.Create (2);
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default ();
    YansWavePhyHelper wavePhy =  YansWavePhyHelper::Default ();
    wavePhy.SetChannel (wifiChannel.Create ());
    QosWaveMacHelper waveMac = QosWaveMacHelper::Default ();
    WaveHelper waveHelper = WaveHelper::Default ();
    devices = waveHelper.Install (wavePhy, waveMac, nodes);

The higher-level helpers include ``ns3::WaveBsmStats`` and ``ns3::WaveBsmHelper``.

WaveBsmStats is used to collect and manage statistics, such as packet and byte
counts and Packet Delivery Ratio (PDR), regarding the sending 
and receiving of WAVE BSM packets.  WaveBsmHelper is used by applications that
wish to send and receive BSMs.

The relation of ``ns3::WaveBsmHelper`` and ``WaveBsmStats`` is described
below:

::

    <Your Vanet Routing Application> ----use----> WaveBsmHelper ----use----> WaveBsmStats

From <Your Vanet Routing Application>, usage is as follows:

    // declare WAVE BSM helper instance
    WaveBsmHelper m_waveBsmHelper;

    // the following are passed to the WaveBsmHelpe::Install()
    // method, and they are thus assumed to be created and 
    // initialized themselves, based on the user's
    // simulation setup criteria.
    // container of network node
    NodeContainer m_adhocTxNodes;
    // (transmitting) devices (1 per node)
    NetDeviceContainer m_adhocTxDevices;
    // IPv4 interfaces (1 per device)
    Ipv4InterfaceContainer m_adhocTxInterfaces;
    // total simulation time (in seconds)
    double m_TotalSimTime;
    // WAVE BSM broadcast interval.  E.g., 100ms = 0.1 seconds
    double m_waveInterval; // seconds
    // time-synchronization accuracy of GPS devices.  E.g., +/- 40ns
    double m_gpsAccuracyNs;
    // array of distances (m) at which safety PDR shall be determined, 
    // e.g. 50m, 100m, 200m, 300m, 400m, 500m, 600m, 800m, 1000m, and 1500m
    std::vector <double> m_txSafetyRanges;
    // used to get consistent random numbers across scenarios
    int64_t m_streamIndex;

    m_waveBsmHelper.Install (m_adhocTxNodes,
                           m_adhocTxDevices,
                           m_adhocTxInterfaces,
                           Seconds(m_TotalSimTime),
                           m_wavePacketSize,
                           Seconds(m_waveInterval),
                           // convert GPS accuracy, in ns, to Time
                           Seconds(m_gpsAccuracyNs / 1000000.0),
                           m_txSafetyRanges);

    // fix random number streams
    m_streamIndex += m_waveBsmHelper.AssignStreams (m_streamIndex); 

Example usages of BSM statistics are as follows:

    // Get the cumulative PDR of the first safety Tx range (i.e, 50m in the 
    // m_txSafetyRanges example above).
    double bsm_pdr1 = m_waveBsmHelper.GetWaveBsmStats ()->GetBsmPdr (1);

    // total WAVE BSM bytes sent
    uint32_t cumulativeWaveBsmBytes = m_waveBsmHelper.GetWaveBsmStats ()->GetTxByteCount ();

    // get number of WAVE BSM packets sent
    int wavePktsSent = m_waveBsmHelper.GetWaveBsmStats ()->GetTxPktCount ();

    // get number of WAVE BSM packets received
    int wavePktsReceived = m_waveBsmHelper.GetWaveBsmStats ()->GetRxPktCount ();

    // reset count of WAVE BSM packets received
    m_waveBsmHelper.GetWaveBsmStats ()->SetRxPktCount (0);

    // reset count of WAVE BSM packets sent
    m_waveBsmHelper.GetWaveBsmStats ()->SetTxPktCount (0);

    // indicate that a node (nodeId) is moving.  (set to 0 to "stop" node)
    WaveBsmHelper::GetNodesMoving()[nodeId] = 1;

APIs
====

MAC layer
#########
The 802.11p device can allow the upper layer to send different information
over Vendor Specific Action management frames by using different
OrganizationIdentifier fields to identify differences.

1. create some Node objects and WifiNetDevice objects, e.g. one sender and one receiver.

2. receiver defines an OrganizationIdentifier

  ::

   uint8_t oi_bytes[5] = {0x00, 0x50, 0xC2, 0x4A, 0x40};
   OrganizationIdentifier oi(oi_bytes,5);

3. receiver defines a Callback for the defined OrganizationIdentifier

  ::

    VscCallback vsccall = MakeCallback (&VsaExample::GetWsaAndOi, this);

4. receiver registers this identifier and function

  ::

      Ptr<WifiNetDevice> device1 = DynamicCast<WifiNetDevice>(nodes.Get (i)->GetDevice (0));
      Ptr<OcbWifiMac> ocb1 = DynamicCast<OcbWifiMac>(device->GetMac ());
      ocb1->AddReceiveVscCallback (oi, vsccall);

5. sender transmits management information over VSA frames

  ::

      Ptr<Packet> vsc = Create<Packet> ();
      ocb2->SendVsc (vsc, Mac48Address::GetBroadcast (), m_16093oi);

6. then registered callbacks in the receiver will be called.

MAC extension layer
###################

The WAVE devices allow the upper layer to route packets in different control 
approaches.  However dedicated APIs and invocation sequences should be 
followed; otherwise, the packets may be discarded by devices.

1. create some Node objects and WaveNetDevice objects by helpers, e.g. one sender and one receiver.

2. receiver registers the receive callback if WSMP and IP-based packets are supposed to be received.

  ::

    // the class ``ns3::WaveNetDeviceExample``here will has a receive method "Receive" to be registered.
    receiver->SetReceiveCallback (MakeCallback (&WaveNetDeviceExample::Receive, this));
    
3. receiver registers the receive callback if WSA frames are supposed to be received.

  ::

    // the class ``ns3::WaveNetDeviceExample``here will has a receive method "ReceiveVsa" to be registered.
    receiver->SetWaveVsaCallback (MakeCallback  (&WaveNetDeviceExample::ReceiveVsa, this));
      
4. sender and receiver assign channel access by StartSch method.

  ::

    // in this case that alternating access with non-immediate mode is assigned for sender and receiver devices.
    const SchInfo schInfo = SchInfo (SCH1, false, EXTENDED_ALTERNATING);
    Simulator::Schedule (Seconds (0.0), &WaveNetDevice::StartSch, sender, schInfo);
    Simulator::Schedule (Seconds (0.0), &WaveNetDevice::StartSch, receiver, schInfo);
 
  or

  ::

    // in this case that continuous access with immediate mode is assigned for sender and receiver devices.
    const SchInfo schInfo = SchInfo (SCH1, true, EXTENDED_CONTINUOUS);
    Simulator::Schedule (Seconds (0.0), &WaveNetDevice::StartSch, sender, schInfo);
    Simulator::Schedule (Seconds (0.0), &WaveNetDevice::StartSch, receiver, schInfo)
    
  or

  ::

    // in this case that extended access with non-immediate mode is assigned for sender and receiver devices.
    const SchInfo schInfo = SchInfo (SCH1, false, 100);
    Simulator::Schedule (Seconds (0.0), &WaveNetDevice::StartSch, sender, schInfo);
    Simulator::Schedule (Seconds (0.0), &WaveNetDevice::StartSch, receiver, schInfo)

5. sender registers a tx profile if IP-based packets are planned to be transmitted

  ::

    // the IP-based packets will be transmitted in SCH1 with 6Mbps and 4 txPowerLevel with adaptable mode.
    const TxProfile txProfile = TxProfile (SCH1, true, 4, WifiMode("OfdmRate6MbpsBW10MHz"));
    Simulator::Schedule (Seconds (2.0), &WaveNetDevice::RegisterTxProfile, sender, txProfile);
    
6. sender transmits  WSMP packets by SendX method.

  ::

    // the data rate and txPowerLevel is controlled by the high layer which are 6Mbps and 0 level here.
    const TxInfo txInfo = TxInfo (CCH, 7, WifiMode("OfdmRate6MbpsBW10MHz"),  0);
    // this packet will contain WSMP header when IEEE 1609.3 model is implemented
    const static uint16_t WSMP_PROT_NUMBER = 0x88DC;
    Ptr<Packet> wsaPacket  = Create<Packet> (100);
    const Address dest = receiver->GetAddress ();  
    Simulator::Schedule (Seconds (2.0),  &WaveNetDevice::SendX, sender, wsaPacket, dest, WSMP_PROT_NUMBER, txInfo);
 
  or

  ::

    // the data rate and txPowerLevel is controlled by the MAC layer which are decided by WifiRemoteStationManager
    const TxInfo txInfo = TxInfo (CCH, 7, WifiMode(),  8);
    // this packet will contain WSMP header when IEEE 1609.3 model is implemented
    const static uint16_t WSMP_PROT_NUMBER = 0x88DC;
    Ptr<Packet> wsaPacket  = Create<Packet> (100);
    const Address dest = receiver->GetAddress ();  
    Simulator::Schedule (Seconds (2.0),  &WaveNetDevice::SendX, sender, wsaPacket, dest, WSMP_PROT_NUMBER, txInfo);

7. sender transmits IP-based packets by Send method.
  
  ::
  
    const static uint16_t IPv6_PROT_NUMBER = 0x86DD;
    Ptr<Packet> packet  = Create<Packet> (100);
    const Address dest = receiver->GetAddress ();
    Simulator::Schedule (Seconds (2.0),  &WaveNetDevice::Send, sender, packet, dest, IPv6_PROT_NUMBER);
 
8. send transmits WSA frames repeatedly by StartVsa method.
 
  ::
   
     // this packet will contain WSA management information when IEEE 1609.3 model is implemented
    Ptr<Packet> wsaPacket = Create<Packet> (100);
    Mac48Address dest = Mac48Address::GetBroadcast ();
    const VsaInfo vsaInfo = VsaInfo (dest, OrganizationIdentifier (), 0, wsaPacket, SCH1, 100, VSA_TRANSMIT_IN_BOTHI);
    Simulator::Schedule (Seconds (2.0), &WaveNetDevice::StartVsa, sender, vsaInfo);
    
9. sender stops WSA frames repeatedly transmit by StopVsa method.

  ::
  
    Simulator::Schedule (Seconds (3.0), &WaveNetDevice::StopVsa, sender, SCH1);

10. sender and receiver release assigned channel access by StopSch method.

  ::

    Simulator::Schedule (Seconds (4.0), &WaveNetDevice::StopSch, sender, SCH1);
    Simulator::Schedule (Seconds (4.0), &WaveNetDevice::StopSch, receiver, SCH1);
    
11. sender or receiver changes current MAC address by ChangeAddress method.

  ::

    Address newAddress = Mac48Address::Allocate ();
    Simulator::Schedule (Seconds (4.0), &WaveNetDevice::ChangeAddress, sender, newAddress);

12. sender cancels all transmissions with the particular category and channel number by CancelTx method.
 
  ::
  
    Simulator::Schedule (Seconds (4.0), &WaveNetDevice::CancelTx, sender, CCH,  AC_BE);

For transmitting and receiving these packets successfully, 
the normal and appropriate invocation procedures should be performed.

(a) For WSMP, channel access should be assigned for transmit and receive. 
The channel access release operation may be optional if there is no need for 
transmission in another channel.

::

    StartSch -------------> SendX / ReceiveCallback -------------->  StopSch

(b) For IP, a tx profile should be registered before transmit and receive
operations. The delete operation of tx profile may be 
optional if there is no need for transmission with other tx parameters. 
The channel access assignment and release optional usage is the same with 
WSMP here.

::

    StartSch -------------> RegisterTxProfile ----------> Send / ReceiveCallback -------------->  DeleteTxProfile -------------> StopSch

(c) For WSA, StartVsa is called to transmit while StopVsa is an optional 
operation for canceling repeat transmit. The channel 
access assignment and release optional usage is also the same with WSMP here. 
To receive VSA, WaveVsaCallback should 
be registered; otherwise, the received VSA frames will be discard by 
the MAC extension layer and not delivered to the higher layer.

::

    StartSch -------------> StartVsa / WaveVsaCallback -------------->  StopVsa ---------------> StopSch

(d) Here an important point is that if the higher layer wants to transmit 
these packets in a control channel (the channel 178), 
there will be no need to request for CCH by the StartSch method, which means 
that StartSch can be optional or should be avoided 
here. The reason is that the default continuous CCH access has been assigned automatically after WAVE devices are created and initialized. 
Therefore, if calling StartSch and StopSch method with CCH as a parameter, 
the request will be discarded by devices and the method will return false to
indicate failure.

Attributes
==========

The channel interval duration's default value is defined in the standard. 
However, the current implementation allows users to configure these 
attributes with other values. These attributes are included in the class
``ns3::ChannelCoodinator`` with config paths shown in the below. The method 
IsValidConfig is suggested to test whether new configuration follows the 
standard.

::

/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/ChannelCoordinator/$ns3::ChannelCoordinator/CchInterval
/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/ChannelCoordinator/$ns3::ChannelCoordinator/SchInterval
/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/ChannelCoordinator/$ns3::ChannelCoordinator/GuardInterval

The ``ns3::WaveNetDevice`` is a wrapper class that contains those classes to support for multiple channel 
operation. To set or get the pointers of those objects, users can also 
use them by config paths shown in the below.

::

/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/Mtu
/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/Channel
/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/PhyEntities
/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/MacEntities
/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/ChannelScheduler
/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/ChannelManager
/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/ChannelCoordinator
/NodeList/[i]/DeviceList/[i]/$ns3::WaveNetDevice/VsaManager
 
Output
======

For the 802.11p device, current classes provide output of the same type as WiFi devices;
namely, ASCII and pcap traces, and logging output.  The 802.11p logging
components can be enabled globally via the call to

::
  
  Wifi80211pHelper::EnableLogComponents ();
  
For the WAVE device, current classes provide output of the same type as WiFi 
devices; namely, ASCII and pcap traces, and logging output. The WAVE logging
components can be enabled globally via the call to

::
  
  WaveHelper::EnableLogComponents ();


Advanced Usage
==============

Advanced WaveHelper configuration
#################################

If users can make sure in which channel this WAVE device will work, 
they can set specific channel numbers to save resources of unused channels .
Usage is as follows:

::

    // in this case, the MAC entities for SCH2 to SCH6 will not be created
    WaveHelper helper = WaveHelper::Default ();
    uint32_t channels[] = {CCH, SCH1};
    std::vector<uint32_t> channelsVector (channels, channels + 2);
    helper.CreateMacForChannel (channelsVector);

If users can create other channel access assignment mechanism, e.g. 
in the context of more PHY entities, which may be called 
"ns3::AnotherScheduler", they can use this helper to create WAVE devices 
with new assignment mechanisms.  Usage is as follows:

::

    WaveHelper helper = WaveHelper::Default ();
    helper.helper.CreateMacForChannel (ChannelManager::GetWaveChannels ());    // create all 7 MAC entities for WAVE
    helper.CreatePhys (2);        // or other number which should be less than 7
    helper.SetChannelScheduler ("ns3::AnotherScheduler");    // The AnotherScheduler should be implemented by users.
    helper.SetRemoteStationManager ("ns3::ConstantRateWifiManager");    // or other  rate control algorithms
    
Examples
========

A basic example exists called ``wave-simple-80211p.cc``.
This example shows basic construction of an 802.11p node.  Two nodes
are constructed with 802.11p devices, and by default, one node sends a single
packet to another node (the number of packets and interval between
them can be configured by command-line arguments).  The example shows
typical usage of the helper classes for this mode of WiFi.

Another example exists called ``wave-simple-device.cc``. This 
example shows how to create WAVE devices by helpers and the routing service 
for different packets.
After WAVE devices are configured and created by helpers, these packets are 
transmitted in different approaches.

Another example exists called ``vanet-routing-compare.cc``. This
example shows how to create mobility nodes in a VANET scenario and
send Basic Safety Message (BSM) packets are regular intervals and/or 
additional data traffic to be routed between nodes.  BSMs are transmitted
assuming the WAVE Short Message Protocol (WSMP), whereas non-BSM data 
packets are relayed by using one of several different IP-based routing
protocols (e.g., AODV, OLSR, DSDV, or DSR).

Troubleshooting
===============

To be defined.

Validation
**********

A test suite named ``wifi-80211p-ocb`` is defined.  This test
case consists of a stationary node and a mobile node.  The mobile
node moves towards the stationary mode, and time points are checked
at which time the physical layer starts to receive packets (and
whether the MAC becomes associated, if applicable).  The same physical
experiment is repeated for normal WiFi NetDevices in AP/STA mode, in
Adhoc mode, and the new OCB mode.

Another test suite named ``wave-mac-extension`` is defined. This test suite 
has four test cases, including ``channel-coordination``, ``channel-routing``, 
``channel-access`` and ``annex-c``. The first case is to test channel 
coordination 
feature. The second case is to test channel routing for three types of packets.
The third case is to test four channel access assignments. And the fourth case 
is to test the implemented feature described in the Annex C of the standard.
It is worth noting that the  ``channel-routing`` and ``channel-access`` test 
cases are both in the context of single-PHY device, which depends on the 
default channel 
access assignment mechanism ``ns3:DefaultChannelScheduler``, thus they may not
be suitable for testing when other channel access assignment mechanisms are 
used.  Although they are test cases, they are also good examples to show 
usage.

The ``ns3::VanetRoutingExample`` example was studied using mobility trace
files in the Raleigh, NC USA area generated using Simulation for Urban 
Mobility (SUMO).  Three environments were studied:  a) an open highway
scenario, b) a residential neighborhood scenario, and c) and urban downtown
scenario.  For each environment, a constant number of 50-750 vehicles was
maintained for 2000 simulation seconds (> 30 minutes).  The mobility trace
file were played back using ``ns3::Ns2MobilityHelper``.  All vehicular nodes
transmitted a 200-byte BSM at 10 Hz and the PDR was determined for 
transmission ranges of 50-1500m.  No additional non-BSM data was injected /
routed through the network.  The default propagation loss model used
was Two-Ray Ground.  Different fading / shadowing models were evaluated,
including a) no fading, b) stochastic Nakagami-m fading, and c) an 
obstacle shadowing model (to be contributed to |ns3|).  30 trials of each
scenario were run in the North Carolina State University (NCSU) High 
Performance Computing (HPC) center, with each trial requiring from 
8 hours to 6 days of CPU time to complete.  Preliminary results were
presented at the PhD Forum, 22nd IEEE International Conference on 
Network Protocols (ICNP), October 24, 2014, Research Triangle Park, NC.
See:  http://www4.ncsu.edu/~scarpen/Research_files/Final-PHD_Forum_SE_Carpenter_2014.pdf
