.. include:: replace.txt
.. highlight:: cpp

.. _sec-wifi-user-doc:

++++++++++++++++++
User Documentation
++++++++++++++++++

Using the WifiNetDevice
***********************

The modularity provided by the implementation makes low-level configuration of
the WifiNetDevice powerful but complex. For this reason, we provide some helper
classes to perform common operations in a simple matter, and leverage the |ns3|
attribute system to allow users to control the parameterization of the underlying
models.

Users who use the low-level |ns3| API and who wish to add a WifiNetDevice to
their node must create an instance of a WifiNetDevice, plus a number of
constituent objects, and bind them together appropriately (the WifiNetDevice is
very modular in this regard, for future extensibility). At the low-level API,
this can be done with about 20 lines of code (see ``ns3::WifiHelper::Install``,
and ``ns3::YansWifiPhyHelper::Create``). They also must create, at some point, a
Channel, which also contains a number of constituent objects (see
``ns3::YansWifiChannelHelper::Create``).

However, a few helpers are available for users to add these devices and channels
with only a few lines of code, if they are willing to use defaults, and the
helpers provide additional API to allow the passing of attribute values to
change default values.  Commonly used attribute values are listed in the
Attributes section.  The scripts in ``examples/wireless`` can be browsed to
see how this is done.  Next, we describe the common steps to create a WifiNetDevice
from the bottom layer (Channel) up to the device layer (WifiNetDevice).

To create a WifiNetDevice, users need to follow these steps:

* Decide on which physical layer framework, the ``SpectrumWifiPhy`` or
  ``YansWifiPhy``, to use.  This will affect which Channel and Phy type to use.
* Configure the Channel: Channel takes care of getting signal
  from one device to other devices on the same Wi-Fi channel.
  The main configurations of WifiChannel are propagation loss model and propagation delay model.
* Configure the WifiPhy: WifiPhy takes care of actually sending and receiving wireless
  signal from Channel.  Here, WifiPhy decides whether each frame will be successfully
  decoded or not depending on the received signal strength and noise.  Thus, the main
  configuration of WifiPhy is the error rate model, which is the one that actually
  calculates the probability of successfully decoding the frame based on the signal.
* Configure WifiMac: this step is more related to the architecture and device level.
  The users configure the wifi architecture (i.e. ad-hoc or ap-sta) and whether QoS (802.11e),
  HT (802.11n) and/or VHT (802.11ac) and/or HE (802.11ax) features are supported or not.
* Create WifiDevice: at this step, users configure the desired wifi standard
  (e.g. **802.11b**, **802.11g**, **802.11a**, **802.11n**, **802.11ac** or **802.11ax**) and rate control algorithm.
* Configure mobility: finally, a mobility model is (usually) required before WifiNetDevice
  can be used; even if the devices are stationary, their relative positions
  are needed for propagation loss calculations.

The following sample code illustrates a typical configuration using mostly
default values in the simulator, and infrastructure mode::

  NodeContainer wifiStaNode;
  wifiStaNode.Create (10);   // Create 10 station node objects
  NodeContainer wifiApNode;
  wifiApNode.Create (1);   // Create 1 access point node object

  // Create a channel helper and phy helper, and then create the channel
  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy = YansWifiPhyHelper::Default ();
  phy.SetChannel (channel.Create ());

  // Create a WifiMacHelper, which is reused across STA and AP configurations
  WifiMacHelper mac;

  // Create a WifiHelper, which will use the above helpers to create
  // and install Wifi devices.  Configure a Wifi standard to use, which
  // will align various parameters in the Phy and Mac to standard defaults.
  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  // Declare NetDeviceContainers to hold the container returned by the helper
  NetDeviceContainer wifiStaDevices;
  NetDeviceContainer wifiApDevice;

  // Perform the installation
  mac.SetType ("ns3::StaWifiMac");
  wifiStaDevices = wifi.Install (phy, mac, wifiStaNodes);
  mac.SetType ("ns3::ApWifiMac");
  wifiApDevice = wifi.Install (phy, mac, wifiApNode);

At this point, the 11 nodes have Wi-Fi devices configured, attached to a
common channel.  The rest of this section describes how additional
configuration may be performed.

YansWifiChannelHelper
=====================

The YansWifiChannelHelper has an unusual name. Readers may wonder why it is
named this way. The reference is to the `yans simulator
<https://dl.acm.org/doi/pdf/10.1145/1190455.1190467?download=true>`_ from which this model is taken. The
helper can be used to create a YansWifiChannel with a default PropagationLoss and
PropagationDelay model.

Users will typically type code such as::

  YansWifiChannelHelper wifiChannelHelper = YansWifiChannelHelper::Default ();
  Ptr<Channel> wifiChannel = wifiChannelHelper.Create ();

to get the defaults.  Specifically, the default is a channel model with a
propagation delay equal to a constant, the speed of light (``ns3::ConstantSpeedPropagationDelayModel``),
and a propagation loss based on a default log distance model (``ns3::LogDistancePropagationLossModel``), using a default exponent of 3.
Please note that the default log distance model is configured with a reference
loss of 46.6777 dB at reference distance of 1m.  The reference loss of 46.6777 dB
was calculated using Friis propagation loss model at 5.15 GHz.  The reference loss
must be changed if **802.11b**, **802.11g**, **802.11n** (at 2.4 GHz) or **802.11ax** (at 2.4 GHz) are used since they operate at 2.4 Ghz.

Note the distinction above in creating a helper object vs. an actual simulation
object.  In |ns3|, helper objects (used at the helper API only) are created on
the stack (they could also be created with operator new and later deleted).
However, the actual |ns3| objects typically inherit from ``class ns3::Object``
and are assigned to a smart pointer.  See the chapter in the |ns3| manual for
a discussion of the |ns3| object model, if you are not familiar with it.

The following two methods are useful when configuring YansWifiChannelHelper:

* ``YansWifiChannelHelper::AddPropagationLoss`` adds a PropagationLossModel; if one or more PropagationLossModels already exist, the new model is chained to the end
* ``YansWifiChannelHelper::SetPropagationDelay`` sets a PropagationDelayModel (not chainable)

YansWifiPhyHelper
=================

Physical devices (base class ``ns3::WifiPhy``) connect to ``ns3::YansWifiChannel`` models in
|ns3|.  We need to create WifiPhy objects appropriate for the YansWifiChannel; here
the ``YansWifiPhyHelper`` will do the work.

The YansWifiPhyHelper class configures an object factory to create instances of
a ``YansWifiPhy`` and adds some other objects to it, including possibly a
supplemental ErrorRateModel and a pointer to a MobilityModel. The user code is
typically::

  YansWifiPhyHelper wifiPhyHelper;
  wifiPhyHelper.SetChannel (wifiChannel);

The default YansWifiPhyHelper is configured with TableBasedErrorRateModel
(``ns3::TableBasedErrorRateModel``). You can change the error rate model by
calling the ``YansWifiPhyHelper::SetErrorRateModel`` method.

Optionally, if pcap tracing is needed, a user may use the following
command to enable pcap tracing::

  YansWifiPhyHelper::SetPcapDataLinkType (enum SupportedPcapDataLinkTypes dlt)

|ns3| supports RadioTap and Prism tracing extensions for 802.11.

Note that we haven't actually created any WifiPhy objects yet; we've just
prepared the YansWifiPhyHelper by telling it which channel it is connected to.
The Phy objects are created in the next step.

In order to enable 802.11n/ac/ax MIMO, the number of antennas as well as the number of supported spatial streams need to be configured.
For example, this code enables MIMO with 2 antennas and 2 spatial streams::

 wifiPhyHelper.Set ("Antennas", UintegerValue (2));
 wifiPhyHelper.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
 wifiPhyHelper.Set ("MaxSupportedRxSpatialStreams", UintegerValue (2));

It is also possible to configure less streams than the number of antennas in order to benefit from diversity gain, and to define different MIMO capabilities for downlink and uplink.
For example, this code configures a node with 3 antennas that supports 2 spatial streams in downstream and 1 spatial stream in upstream::

 wifiPhyHelper.Set ("Antennas", UintegerValue (3));
 wifiPhyHelper.Set ("MaxSupportedTxSpatialStreams", UintegerValue (2));
 wifiPhyHelper.Set ("MaxSupportedRxSpatialStreams", UintegerValue (1));

802.11n PHY layer can support both 20 (default) or 40 MHz channel width, and 802.11ac/ax PHY layer can use either 20, 40, 80 (default) or 160 MHz channel width.  See below for further documentation on setting the frequency, channel width, and channel number.

::

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ac);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue ("VhtMcs9"),
                                "ControlMode", StringValue ("VhtMcs0"));

  //Install PHY and MAC
  Ssid ssid = Ssid ("ns3-wifi");
  WifiMacHelper mac;

  mac.SetType ("ns3::StaWifiMac",
	       "Ssid", SsidValue (ssid),
	       "ActiveProbing", BooleanValue (false));

  NetDeviceContainer staDevice;
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  mac.SetType ("ns3::ApWifiMac",
	       "Ssid", SsidValue (ssid));

  NetDeviceContainer apDevice;
  apDevice = wifi.Install (phy, mac, wifiApNode);

.. _Channel-settings:

Channel, frequency, channel width, and band configuration
=========================================================

There is a unique ``ns3::WifiPhy`` attribute, named ``ChannelSettings``, that
enables to set channel number, channel width, frequency band and primary20 index
all together, in order to eliminate the possibility of inconsistent settings.
The ``ChannelSettings`` attribute can be set in a number of ways (see below) by
providing either a StringValue object or a TupleValue object:

* Defining a StringValue object to set the ``ChannelSettings`` attribute

::

  StringValue value ("{38, 40, BAND_5GHZ, 0}"));

* Defining a TupleValue object to set the ``ChannelSettings`` attribute

::

  TupleValue<UintegerValue, UintegerValue, EnumValue, UintegerValue> value;
  value.Set (WifiPhy::ChannelTuple {38, 40, WIFI_PHY_BAND_5GHZ, 0});

In both cases, the operating channel will be channel 38 in the 5 GHz band, which
has a width of 40 MHz, and the primary20 channel will be the 20 MHz subchannel
with the lowest center frequency (index 0).

The operating channel settings can then be configured in a number of ways:

* by setting global configuration default; e.g.

::

  Config::SetDefault ("ns3::WifiPhy::ChannelSettings", StringValue ("{38, 40, BAND_5GHZ, 0}"));

* by setting an attribute value in the helper; e.g.

::

  TupleValue<UintegerValue, UintegerValue, EnumValue, UintegerValue> value;
  value.Set (WifiPhy::ChannelTuple {38, 40, WIFI_PHY_BAND_5GHZ, 0});
  YansWifiPhyHelper wifiPhyHelper = YansWifiPhyHelper::Default ();
  wifiPhyHelper.Set ("ChannelSettings", value);

* by setting the WifiHelper::SetStandard (enum WifiStandard) method; and

* by performing post-installation configuration of the option, either
  via a Ptr to the WifiPhy object, or through the Config namespace; e.g.:

::

  Config::Set ("/NodeList/0/DeviceList/*/$ns3::WifiNetDevice/Phy/$ns3::WifiPhy/ChannelSettings",
               StringValue ("{38, 40, BAND_5GHZ, 0}"));

This section provides guidance on how to properly configure these settings.

WifiHelper::SetStandard()
+++++++++++++++++++++++++

``WifiHelper::SetStandard ()`` is a method required to set various parameters
in the Mac and Phy to standard values, but also to check that the channel
settings as described above are allowed. For instance, a channel in the 2.4 GHz
band cannot be configured if the standard is 802.11ac, or a channel in the 6 GHz
band can only be configured if the standard is 802.11ax (or beyond).

The following values for WifiStandard are defined in
``src/wifi/model/wifi-standards.h``:

::

  WIFI_STANDARD_80211a,
  WIFI_STANDARD_80211b,
  WIFI_STANDARD_80211g,
  WIFI_STANDARD_80211p,
  WIFI_STANDARD_80211n,
  WIFI_STANDARD_80211ac,
  WIFI_STANDARD_80211ax

By default, the WifiHelper (the typical use case for WifiPhy creation) will
configure the WIFI_STANDARD_80211ax standard by default.  Other values
for standards should be passed explicitly to the WifiHelper object.

If user has not already configured ChannelSettings when SetStandard is called,
the user obtains default values, as described next.

Default settings for the operating channel
++++++++++++++++++++++++++++++++++++++++++

Not all the parameters in the channel settings have to be set to a valid value,
but they can be left unspecified, in which case default values are substituted
as soon as the WifiStandard is set. Here are the rules (applied in the given order):

* If the band is unspecified (i.e., it is set to WIFI_PHY_BAND_UNSPECIFIED or
  "BAND_UNSPECIFIED"), the default band for the configured standard is set
  (5 GHz band for 802.11{a, ac, ax, p} and 2.4 GHz band for all the others).

* If both the channel width and the channel number are unspecified (i.e., they
  are set to zero), the default channel width for the configured standard and
  band is set (22 MHz for 802.11b, 10 MHz for 802.11p, 80 MHz for 802.11ac and
  for 802.11ax if the band is 5 GHz, and 20 MHz for all other cases).

* If the channel width is unspecified but the channel number is valid, the settings
  are valid only if there exists a unique channel with the given number for the
  configured standard and band, in which case the channel width is set to the width
  of such unique channel. Otherwise, the simulation aborts.

* If the channel number is unspecified (i.e., it is set to zero), the default
  channel number for the configured standard, band and channel width is used
  (the default channel number is the first one in the list of channels that can
  be used with the configured standard, band and channel width)

Following are a few examples to clarify these rules:

::

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ac);
  YansWifiPhyHelper phyHelper;
  phyHelper.Set ("ChannelSettings", StringValue ("{58, 0, WIFI_PHY_BAND_5GHZ, 0}"));
  // channel width unspecified
  // -> it is set to 80 MHz (width of channel 58)

::

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  YansWifiPhyHelper phyHelper;
  phyHelper.Set ("ChannelSettings", StringValue ("{0, 40, WIFI_PHY_BAND_5GHZ, 0}"));
  // channel number unspecified
  // -> it is set to channel 38 (first 40 MHz channel in the 5GHz band)

::

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211ax);
  YansWifiPhyHelper phyHelper;
  phyHelper.Set ("ChannelSettings", StringValue ("{0, 0, WIFI_PHY_BAND_2_4GHZ, 0}"));
  // both channel number and width unspecified
  // -> width set to 20 MHz (default width for 802.11ax in the 2.4 GHZ band)
  // -> channel number set to 1 (first 20 MHz channel in the 2.4 GHz band)

::

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211a);
  YansWifiPhyHelper phyHelper;
  phyHelper.Set ("ChannelSettings", StringValue ("{0, 0, WIFI_PHY_BAND_UNSPECIFIED, 0}"));
  // band, channel number and width unspecified
  // -> band is set to WIFI_PHY_BAND_5GHZ (default band for 802.11a)
  // -> width set to 20 MHz (default width for 802.11a in the 5 GHZ band)
  // -> channel number set to 36 (first 20 MHz channel in the 5 GHz band)


The default value for the ChannelSettings attribute leaves all the parameters
unspecified, except for the primary20 index, which is equal to zero.

WifiPhy::Frequency
++++++++++++++++++

The configured WifiPhy channel center frequency can be got via the attribute
``Frequency`` in the class ``WifiPhy``.  It is expressed in units of MHz.

Note that this is a change in definition from ns-3.25 and earlier releases,
where this attribute referred to the start of the overall frequency band
on which the channel resides, not the specific channel center frequency.

WifiPhy::ChannelWidth
+++++++++++++++++++++

The configured WifiPhy channel width can be got via the attribute ``ChannelWidth``
in the class ``WifiPhy``.  It is expressed in units of MHz.

WifiPhy::ChannelNumber
++++++++++++++++++++++

Several channel numbers are defined and well-known in practice.  However,
valid channel numbers vary by geographical region around the world, and
there is some overlap between the different standards.

The configured WifiPhy channel number can be got via the attribute ``ChannelNumber``
in the class ``WifiPhy``.

In |ns3|, a ChannelNumber may be defined or unknown.  These terms
are not found in the code; they are just used to describe behavior herein.

If a ChannelNumber is defined, it means that WifiPhy has stored a
map of ChannelNumber to the center frequency and channel width commonly
known for that channel in practice.  For example:

* Channel 1, when IEEE 802.11b is configured, corresponds to a channel
  width of 22 MHz and a center frequency of 2412 MHz.

* Channel 36, when IEEE 802.11n is configured at 5 GHz, corresponds to
  a channel width of 20 MHz and a center frequency of 5180 MHz.

The following channel numbers are well-defined for 2.4 GHz standards:

* channels 1-14 with ChannelWidth of 22 MHz for 802.11b
* channels 1-14 with ChannelWidth of 20 MHz for 802.11n-2.4GHz and 802.11g

The following channel numbers are well-defined for 5 GHz standards:

.. table:: 5 GHz channel numbers
   :widths: 20 70

   +------------------+-------------------------------------------+
   | ``ChannelWidth`` | ``ChannelNumber``                         |
   +------------------+-------------------------------------------+
   | 20 MHz           | 36, 40, 44, 48, 52, 56, 60, 64, 100,      |
   |                  | 104, 108, 112, 116, 120, 124,             |
   |                  | 128, 132, 136, 140, 144,                  |
   |                  | 149, 153, 161, 165, 169                   |
   +------------------+-------------------------------------------+
   | 40 MHz           | 38, 46, 54, 62, 102, 110, 118, 126,       |
   |                  | 134, 142, 151, 159                        |
   +------------------+-------------------------------------------+
   | 80 MHz           | 42, 58, 106, 122, 138, 155                |
   +------------------+-------------------------------------------+
   | 160 MHz          | 50, 114                                   |
   +------------------+-------------------------------------------+
   | 10 MHz (802.11p) | 172, 174, 176, 178, 180, 182, 184         |
   +------------------+-------------------------------------------+
   | 5 MHz (802.11p)  | 171, 173, 175, 177, 179, 181, 183         |
   +------------------+-------------------------------------------+


The following channel numbers are well-defined for 6 GHz standards (802.11ax only):

.. table:: 6 GHz channel numbers
   :widths: 20 70

   +------------------+-------------------------------------------+
   | ``ChannelWidth`` | ``ChannelNumber``                         |
   +------------------+-------------------------------------------+
   | 20 MHz           | 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41,  |
   |                  | 45, 49, 53, 57, 61, 65, 69, 73, 77, 81,   |
   |                  | 85, 89, 93, 97, 101, 105, 109, 113, 117,  |
   |                  | 121, 125, 129, 133, 137, 141, 145, 149,   |
   |                  | 153, 157, 161, 165, 169, 173, 177, 181,   |
   |                  | 185, 189, 193, 197, 201, 205, 209, 213,   |
   |                  | 217, 221, 225, 229, 233                   |
   +------------------+-------------------------------------------+
   | 40 MHz           | 3, 11, 19, 27, 35, 43, 51, 59, 67, 75,    |
   |                  | 83, 91, 99, 107, 115, 123, 131, 139, 147, |
   |                  | 155, 163, 171, 179, 187, 195, 203, 211,   |
   |                  | 219, 227                                  |
   +------------------+-------------------------------------------+
   | 80 MHz           | 7, 23, 39, 55, 71, 87, 103, 119, 135,     |
   |                  | 151, 167, 183, 199, 215                   |
   +------------------+-------------------------------------------+
   | 160 MHz          | 15, 47, 79, 111, 143, 175, 207            |
   +------------------+-------------------------------------------+


The channel number may be set either before or after creation of the
WifiPhy object.

If an unknown channel number (other than zero) is configured, the
simulator will exit with an error; for instance, such as:

::

  Ptr<WifiPhy> wifiPhy = ...;
  wifiPhy->SetAttribute ("ChannelSettings", StringValue ("{1321, 20, BAND_5GHZ, 0}"));

The known channel numbers are defined in the implementation file
``src/wifi/model/wifi-phy-operating-channel.cc``. Of course, this file may be edited
by users to extend to additional channel numbers.

If a known channel number is configured against an incorrect value
of the WifiPhyStandard, the simulator will exit with an error; for instance,
such as:

::

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  ...
  Ptr<WifiPhy> wifiPhy = ...;
  wifiPhy->SetAttribute ("ChannelSettings", StringValue ("{14, 20, BAND_5GHZ, 0}"));

In the above, while channel number 14 is well-defined in practice for 802.11b
only, it is for 2.4 GHz band, not 5 GHz band.

WifiPhy::Primary20MHzIndex
++++++++++++++++++++++++++

The configured WifiPhy primary 20MHz channel index can be got via the attribute
``Primary20MHzIndex`` in the class ``WifiPhy``.

Order of operation issues
+++++++++++++++++++++++++

Channel settings can be configured either before or after the wifi standard.
If the channel settings are configured before the wifi standard, the channel
settings are stored and applied when the wifi standard is configured.
Otherwise, they are applied immediately.

The wifi standard can be configured only once, i.e., it is not possible to
change standard during a simulation. It is instead possible to change the
channel settings at any time.


SpectrumWifiPhyHelper
=====================

The API for this helper closely tracks the API of the YansWifiPhyHelper,
with the exception that a channel of type ``ns3::SpectrumChannel`` instead
of type ``ns3::YansWifiChannel`` must be used with it.

WifiMacHelper
=============

The next step is to configure the MAC model. We use WifiMacHelper to accomplish this.
WifiMacHelper takes care of both the MAC low model and MAC high model, and configures an object factory to create instances of a ``ns3::WifiMac``.
It is used to configure MAC parameters like type of MAC, and to select whether 802.11/WMM-style QoS and/or 802.11n-style High Throughput (HT)
and/or 802.11ac-style Very High Throughput (VHT) support and/or 802.11ax-style High Efficiency (HE) support are/is required.

By default, it creates an ad-hoc MAC instance that does not have 802.11e/WMM-style QoS nor 802.11n-style High Throughput (HT)
nor 802.11ac-style Very High Throughput (VHT) nor 802.11ax-style High Efficiency (HE) support enabled.

For example the following user code configures a non-QoS and non-HT/non-VHT/non-HE MAC that
will be a non-AP STA in an infrastructure network where the AP has SSID ``ns-3-ssid``::

    WifiMacHelper wifiMacHelper;
    Ssid ssid = Ssid ("ns-3-ssid");
    wifiMacHelper.SetType ("ns3::StaWifiMac",
                          "Ssid", SsidValue (ssid),
                          "ActiveProbing", BooleanValue (false));

The following code shows how to create an AP with QoS enabled::

  WifiMacHelper wifiMacHelper;
  wifiMacHelper.SetType ("ns3::ApWifiMac",
                         "Ssid", SsidValue (ssid),
                         "QosSupported", BooleanValue (true),
                         "BeaconGeneration", BooleanValue (true),
                         "BeaconInterval", TimeValue (Seconds (2.5)));

To create ad-hoc MAC instances, simply use ``ns3::AdhocWifiMac`` instead of ``ns3::StaWifiMac`` or ``ns3::ApWifiMac``.

With QoS-enabled MAC models it is possible to work with traffic belonging to
four different Access Categories (ACs): **AC_VO** for voice traffic,
**AC_VI** for video traffic, **AC_BE** for best-effort
traffic and **AC_BK** for background traffic.

When selecting **802.11n** as the desired wifi standard, both 802.11e/WMM-style QoS and 802.11n-style High Throughput (HT) support gets enabled.
Similarly when selecting **802.11ac** as the desired wifi standard, 802.11e/WMM-style QoS, 802.11n-style High Throughput (HT) and 802.11ac-style Very High Throughput (VHT)
support gets enabled. And when selecting **802.11ax** as the desired wifi standard, 802.11e/WMM-style QoS, 802.11n-style High Throughput (HT),
802.11ac-style Very High Throughput (VHT) and 802.11ax-style High Efficiency (HE) support gets enabled.

For MAC instances that have QoS support enabled, the ``ns3::WifiMacHelper`` can be also used to set:

* block ack threshold (number of packets for which block ack mechanism should be used);
* block ack inactivity timeout.

For example the following user code configures a MAC that will be a non-AP STA with QoS enabled and a block ack threshold for AC_BE set to 2 packets,
in an infrastructure network where the AP has SSID ``ns-3-ssid``::

    WifiMacHelper wifiMacHelper;
    Ssid ssid = Ssid ("ns-3-ssid");
    wifiMacHelper.SetType ("ns3::StaWifiMac",
                          "Ssid", SsidValue (ssid),
                          "QosSupported", BooleanValue (true),
                          "BE_BlockAckThreshold", UintegerValue (2),
                          "ActiveProbing", BooleanValue (false));

For MAC instances that have 802.11n-style High Throughput (HT) and/or 802.11ac-style Very High Throughput (VHT) and/or 802.11ax-style High Efficiency (HE) support enabled,
the ``ns3::WifiMacHelper`` can be also used to set:

* MSDU aggregation parameters for a particular Access Category (AC) in order to use 802.11n/ac A-MSDU feature;
* MPDU aggregation parameters for a particular Access Category (AC) in order to use 802.11n/ac A-MPDU feature.

By default, MSDU aggregation feature is disabled for all ACs and MPDU aggregation is enabled for AC_VI and AC_BE, with a maximum aggregation size of 65535 bytes.

For example the following user code configures a MAC that will be a non-AP STA with HT and QoS enabled, MPDU aggregation enabled for AC_VO with a maximum aggregation size of 65535 bytes, and MSDU aggregation enabled for AC_BE with a maximum aggregation size of 7935 bytes,
in an infrastructure network where the AP has SSID ``ns-3-ssid``::

    WifiHelper wifi;
    wifi.SetStandard (WIFI_STANDARD_80211n);

    WifiMacHelper wifiMacHelper;
    Ssid ssid = Ssid ("ns-3-ssid");
    wifiMacHelper.SetType ("ns3::StaWifiMac",
                          "Ssid", SsidValue (ssid),
                          "VO_MaxAmpduSize", UintegerValue (65535),
                          "BE_MaxAmsduSize", UintegerValue (7935),
                          "ActiveProbing", BooleanValue (false));

802.11ax APs support sending multi-user frames via DL OFDMA and UL OFDMA if a Multi-User Scheduler is
aggregated to the wifi MAC (by default no scheduler is aggregated). WifiMacHelper enables to aggregate
a Multi-User Scheduler to an AP and set its parameters::

    WifiMacHelper wifiMacHelper;
    wifiMacHelper.SetMultiUserScheduler ("ns3::RrMultiUserScheduler",
                                        "EnableUlOfdma", BooleanValue (true),
                                        "EnableBsrp", BooleanValue (false));

The Ack Manager is in charge of selecting the acknowledgment method among the three
available methods (see section :ref:`wifi-mu-ack-sequences` ). The default ack manager
enables to select the acknowledgment method, e.g.::

    Config::SetDefault ("ns3::WifiDefaultAckManager::DlMuAckSequenceType",
                       EnumValue (WifiAcknowledgment::DL_MU_AGGREGATE_TF));

Selection of the Access Category (AC)
+++++++++++++++++++++++++++++++++++++

Since ns-3.26, the QosTag is no longer used to assign a user priority to an MSDU.
Instead, the selection of the Access Category (AC) for an MSDU is based on the
value of the DS field in the IP header of the packet (ToS field in case of IPv4,
Traffic Class field in case of IPv6). Details on how to set the ToS field of IPv4
packets are given in the :ref:`Type-of-service` section of the documentation. In
summary, users can create an address of type :cpp:class:`ns3::InetSocketAddress`
with the desired type of service value and pass it to the application helpers::

    InetSocketAddress destAddress (ipv4Address, udpPort);
    destAddress.SetTos (tos);
    OnOffHelper onoff ("ns3::UdpSocketFactory", destAddress);

Mapping the values of the DS field onto user priorities is performed similarly to the
Linux mac80211 subsystem. Basically, the :cpp:func:`ns3::WifiNetDevice::SelectQueue()`
method sets the user priority (UP) of an MSDU to the three most significant
bits of the DS field. The Access Category is then determined based on the user priority
according to the following table:

===  ===============
UP   Access Category
===  ===============
 7     AC_VO
 6     AC_VO
 5     AC_VI
 4     AC_VI
 3     AC_BE
 0     AC_BE
 2     AC_BK
 1     AC_BK
===  ===============

TOS and DSCP values map onto user priorities and access categories according
to the following table.

============  ============  ==  ===============
DiffServ PHB  TOS (binary)  UP  Access Category
============  ============  ==  ===============
EF            101110xx      5   AC_VI
AF11          001010xx      1   AC_BK
AF21          010010xx      2   AC_BK
AF31          011010xx      3   AC_BE
AF41          100010xx      4   AC_VI
AF12          001100xx      1   AC_BK
AF22          010100xx      2   AC_BK
AF32          011100xx      3   AC_BE
AF42          100100xx      4   AC_VI
AF13          001110xx      1   AC_BK
AF23          010110xx      2   AC_BK
AF33          011110xx      3   AC_BE
AF43          100110xx      4   AC_VI
CS0           000000xx      0   AC_BE
CS1           001000xx      1   AC_BK
CS2           010000xx      2   AC_BK
CS3           011000xx      3   AC_BE
CS4           100000xx      4   AC_VI
CS5           101000xx      5   AC_VI
CS6           110000xx      6   AC_VO
CS7           111000xx      7   AC_VO
============  ============  ==  ===============

So, for example,::

    destAddress.SetTos (0xc0);

will map to CS6, User Priority 6, and Access Category AC_VO.
Also, the ns3-wifi-ac-mapping test suite (defined in
src/test/ns3wifi/wifi-ac-mapping-test-suite.cc) can provide additional
useful information.

Note that :cpp:func:`ns3::WifiNetDevice::SelectQueue()` also sets the packet
priority to the user priority, thus overwriting the value determined by the
socket priority (users can read :ref:`Socket-options` for details on how to
set the packet priority). Also, given that the Traffic Control layer calls
:cpp:func:`ns3::WifiNetDevice::SelectQueue()` before enqueuing the packet
into a queue disc, it turns out that queuing disciplines (such as
PfifoFastQueueDisc) that classifies packets based on their priority will
use the user priority instead of the socket priority.

WifiHelper
==========

We're now ready to create WifiNetDevices. First, let's create
a WifiHelper with default settings::

  WifiHelper wifiHelper;

What does this do?  It sets the default wifi standard to **802.11a** and sets the RemoteStationManager to
``ns3::ArfWifiManager``.  You can change the RemoteStationManager by calling the
``WifiHelper::SetRemoteStationManager`` method. To change the wifi standard, call the
``WifiHelper::SetStandard`` method with the desired standard.

Now, let's use the wifiPhyHelper and wifiMacHelper created above to install WifiNetDevices
on a set of nodes in a NodeContainer "c"::

  NetDeviceContainer wifiContainer = WifiHelper::Install (wifiPhyHelper, wifiMacHelper, c);

This creates the WifiNetDevice which includes also a WifiRemoteStationManager, a
WifiMac, and a WifiPhy (connected to the matching Channel).

The ``WifiHelper::SetStandard`` method sets various default timing parameters as defined in the selected standard version, overwriting values that may exist or have been previously configured.
In order to change parameters that are overwritten by ``WifiHelper::SetStandard``, this should be done post-install using ``Config::Set``::

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager", "DataMode", StringValue("HtMcs7"), "ControlMode", StringValue("HtMcs0"));

  //Install PHY and MAC
  Ssid ssid = Ssid ("ns3-wifi");

  WifiMacHelper mac;
  mac.SetType ("ns3::StaWifiMac",
  "Ssid", SsidValue (ssid),
  "ActiveProbing", BooleanValue (false));

  NetDeviceContainer staDevice;
  staDevice = wifi.Install (phy, mac, wifiStaNode);

  mac.SetType ("ns3::ApWifiMac",
  "Ssid", SsidValue (ssid));

  NetDeviceContainer apDevice;
  apDevice = wifi.Install (phy, mac, wifiApNode);

  //Once install is done, we overwrite the standard timing values
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/Slot", TimeValue (MicroSeconds (slot)));
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/Sifs", TimeValue (MicroSeconds (sifs)));
  Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/Pifs", TimeValue (MicroSeconds (pifs)));

The WifiHelper can be used to set the attributes of the default ack policy selector
(``ConstantWifiAckPolicySelector``) or to select a different (user provided) ack
policy selector, for each of the available Access Categories. As an example, the
following code can be used to set the BaThreshold attribute of the default ack
policy selector associated with BE AC to 0.5::

  WifiHelper wifi;
  wifi.SetAckPolicySelectorForAc (AC_BE, "ns3::ConstantWifiAckPolicySelector",
                                  "BaThreshold", DoubleValue (0.5));

The WifiHelper is also used to configure OBSS PD spatial reuse for 802.11ax.
The following lines configure a WifiHelper to support OBSS PD spatial reuse
using the ``ConstantObssPdAlgorithm`` with a threshold set to -72 dBm::

  WifiHelper wifi;
  wifi.SetObssPdAlgorithm ("ns3::ConstantObssPdAlgorithm",
                           "ObssPdLevel", DoubleValue (-72.0));

There are many other |ns3| attributes that can be set on the above helpers to
deviate from the default behavior; the example scripts show how to do some of
this reconfiguration.

HT configuration
================

HT is an acronym for High Throughput, a term synonymous with the IEEE 802.11n
standard.  Once the ``ns3::WifiHelper::Install`` has been called and the
user sets the standard to a variant that supports HT capabilities (802.11n,
802.11ac, or 802.11ax), an HT configuration object will automatically be
created for the device.  The configuration object is used to store and
manage HT-specific attributes.

802.11n/ac PHY layer can use either long (800 ns) or short (400 ns) OFDM guard intervals. To configure this parameter for a given device, the following lines of code could be used (in this example, it enables the support of a short guard interval for the first station)::

 Ptr<NetDevice> nd = wifiStaDevices.Get (0);
 Ptr<WifiNetDevice> wnd = nd->GetObject<WifiNetDevice> ();
 Ptr<HtConfiguration> htConfiguration = wnd->GetHtConfiguration ();
 htConfiguration->SetShortGuardIntervalSupported (true);

It is also possible to configure HT-specific attributes using ``Config::Set``.
The following line of code enables the support of a short guard interval for all stations:

 Config::Set ("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/ShortGuardIntervalSupported", BooleanValue (true));

VHT configuration
=================

IEEE 802.11ac devices are also known as supporting Very High Throughput (VHT).  Once the ``ns3::WifiHelper::Install`` has been called and either the 802.11ac
or 802.11ax 5 GHz standards are configured, a VHT configuration object will be
automatically created to manage VHT-specific attributes.

As of ns-3.29, however, there are no VHT-specific configuration items to
manage; therefore, this object is a placeholder for future growth.

HE configuration
================

IEEE 802.11ax is also known as High Efficiency (HE).  Once the ``ns3::WifiHelper::Install`` has been called and IEEE 802.11ax configured as the standard, an
HE configuration object will automatically be created to manage HE-specific
attributes for 802.11ax devices.

802.11ax PHY layer can use either 3200 ns, 1600 ns or 800 ns OFDM guard intervals. To configure this parameter, the following lines of code could be used (in this example, it enables the support of 1600 ns guard interval), such as in this example code snippet::

 Ptr<NetDevice> nd = wifiStaDevices.Get (0);
 Ptr<WifiNetDevice> wnd = nd->GetObject<WifiNetDevice> ();
 Ptr<HeConfiguration> heConfiguration = wnd->GetHeConfiguration ();
 heConfiguration->SetGuardInterval (NanoSeconds (1600));

 802.11ax allows extended compressed Block ACKs containing a 256-bits bitmap, making possible transmissions of A-MPDUs containing up to 256 MPDUs,
 depending on the negotiated buffer size. In order to configure the buffer size of an 802.11ax device, the following line of code could be used::

 heConfiguration->SetMpduBufferSize (256);

 For transmitting large MPDUs, it might also be needed to increase the maximum aggregation size (see above).

Mobility configuration
======================

Finally, a mobility model must be configured on each node with Wi-Fi device.
Mobility model is used for calculating propagation loss and propagation delay.
Two examples are provided in the next section.
Users are referred to the chapter on :ref:`Mobility` module for detailed information.

Example configuration
=====================

We provide two typical examples of how a user might configure a Wi-Fi network --
one example with an ad-hoc network and one example with an infrastructure network.
The two examples were modified from the two examples in the ``examples/wireless`` folder
(``wifi-simple-adhoc.cc`` and ``wifi-simple-infra.cc``).
Users are encouraged to see examples in the ``examples/wireless`` folder.

AdHoc WifiNetDevice configuration
+++++++++++++++++++++++++++++++++

In this example, we create two ad-hoc nodes equipped with 802.11a Wi-Fi devices.
We use the ``ns3::ConstantSpeedPropagationDelayModel`` as the propagation delay model and
``ns3::LogDistancePropagationLossModel`` with the exponent of 3.0 as the propagation loss model.
Both devices are configured with ``ConstantRateWifiManager`` at the fixed rate of 12Mbps.
Finally, we manually place them by using the ``ns3::ListPositionAllocator``::

  std::string phyMode ("OfdmRate12Mbps");

  NodeContainer c;
  c.Create (2);

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211a);

  YansWifiPhyHelper wifiPhy =  YansWifiPhyHelper::Default ();
  // ns-3 supports RadioTap and Prism tracing extensions for 802.11
  wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);

  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                  "Exponent", DoubleValue (3.0));
  wifiPhy.SetChannel (wifiChannel.Create ());

  // Add a non-QoS upper mac, and disable rate control (i.e. ConstantRateWifiManager)
  WifiMacHelper wifiMac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode",StringValue (phyMode),
                                "ControlMode",StringValue (phyMode));
  // Set it to adhoc mode
  wifiMac.SetType ("ns3::AdhocWifiMac");
  NetDeviceContainer devices = wifi.Install (wifiPhy, wifiMac, c);

  // Configure mobility
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (c);

  // other set up (e.g. InternetStack, Application)

Infrastructure (access point and clients) WifiNetDevice configuration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is a typical example of how a user might configure an access point and a set of clients.
In this example, we create one access point and two clients.
Each node is equipped with 802.11b Wi-Fi device::

  std::string phyMode ("DsssRate1Mbps");

  NodeContainer ap;
  ap.Create (1);
  NodeContainer sta;
  sta.Create (2);

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211b);

  YansWifiPhyHelper wifiPhy =  YansWifiPhyHelper::Default ();
  // ns-3 supports RadioTap and Prism tracing extensions for 802.11
  wifiPhy.SetPcapDataLinkType (WifiPhyHelper::DLT_IEEE802_11_RADIO);

  YansWifiChannelHelper wifiChannel;
  // reference loss must be changed since 802.11b is operating at 2.4GHz
  wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
  wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel",
                                  "Exponent", DoubleValue (3.0),
				  "ReferenceLoss", DoubleValue (40.0459));
  wifiPhy.SetChannel (wifiChannel.Create ());

  // Add a non-QoS upper mac, and disable rate control
  WifiMacHelper wifiMac;
  wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
                                "DataMode",StringValue (phyMode),
                                "ControlMode",StringValue (phyMode));

  // Setup the rest of the upper mac
  Ssid ssid = Ssid ("wifi-default");
  // setup ap.
  wifiMac.SetType ("ns3::ApWifiMac",
                   "Ssid", SsidValue (ssid));
  NetDeviceContainer apDevice = wifi.Install (wifiPhy, wifiMac, ap);
  NetDeviceContainer devices = apDevice;

  // setup sta.
  wifiMac.SetType ("ns3::StaWifiMac",
                   "Ssid", SsidValue (ssid),
                   "ActiveProbing", BooleanValue (false));
  NetDeviceContainer staDevice = wifi.Install (wifiPhy, wifiMac, sta);
  devices.Add (staDevice);

  // Configure mobility
  MobilityHelper mobility;
  Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator> ();
  positionAlloc->Add (Vector (0.0, 0.0, 0.0));
  positionAlloc->Add (Vector (5.0, 0.0, 0.0));
  positionAlloc->Add (Vector (0.0, 5.0, 0.0));
  mobility.SetPositionAllocator (positionAlloc);
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (ap);
  mobility.Install (sta);

  // other set up (e.g. InternetStack, Application)
