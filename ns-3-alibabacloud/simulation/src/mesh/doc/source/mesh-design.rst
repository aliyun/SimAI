.. include:: replace.txt

.. heading hierarchy:
   ------------- Document Title
   ************* Chapter (#)
   ============= Section (#.#)
   ############# Subsection (#.#.#)
   ~~~~~~~~~~~~~ Paragraph (no number)

Design Documentation
********************

Overview
========

The |ns3| `mesh` module extends the |ns3| `wifi` module to provide mesh
networking capabilities according to the IEEE 802.11s standard [ieee80211s]_.

The basic purpose of IEEE 802.11s is to define a mode of operation for 
Wi-Fi that permits frames to be forwarded over multiple radio hops 
transparent to higher layer protocols such as IP.  To accomplish this,
mesh-capable stations form a `Mesh Basic Service Set` (MBSS) by running
a pair-wise peering protocol to establish forwarding associations, and
by running a routing protocol to find paths through the network.  A
special gateway device called a `mesh gate` allows a MBSS to interconnect
with a Distribution System (DS).

The basic enhancements defined by IEEE 802.11s include:

* discovery services
* peering management
* security 
* beaconing and synchronization
* the Mesh Coordination Function (MCF)
* power management
* channel switching
* extended frame formats
* path selection and forwarding
* interworking (proxy mesh gateways)
* intra-mesh congestion control, and
* emergency service support.

The |ns3| models implement only a subset of the above service extensions,
focusing mainly on those items related to peering and routing/forwarding
of data frames through the mesh.

The Mesh NetDevice based on 802.11s D3.0 draft standard
was added in *ns-3.6* and includes the Mesh Peering Management Protocol and
HWMP (routing) protocol implementations. An overview
presentation by Kirill Andreev was published at the Workshop on ns-3
in 2009 [And09]_.  An overview paper is available at [And10]_.

As of ns-3.23 release, the model has been updated to the 802.11s-2012
standard [ieee80211s]_ with regard to packet formats, based on the 
contribution in [Hep15]_.

These changes include:

* Category codes and the categories compliant to IEEE-802.11-2012 Table 8-38—Category values.  
* Information Elements (An adjustment of the element ID values was needed according to Table 8-54 of IEEE-802.11-2012).
* Mesh Peering Management element format changed according to IEEE-802.11-2012 Figure 8-370.
* Mesh Configuration element format changed according to IEEE-802.11-2012 Figure 8-363.
* PERR element format changed according to IEEE-802.11-2012 Figure 8-394.

With these changes the messages of the Peering Management Protocol and Hybrid Wireless Mesh Protocol will be transmitted compliant to IEEE802.11-2012 and the resulting pcap trace files can be analyzed by Wireshark.

The multi-interface mesh points are supported as an extension of IEEE draft version 3.0. Note that corresponding |ns3| mesh device helper creates a single interface station by default.

Overview of IEEE 802.11s
########################

The implementation of the 802.11s extension consists of two main parts: the Peer Management Protocol (PMP) and Hybrid Wireless Mesh Protocol (HWMP).

The tasks of the peer management protocol are the following:

* opening links, detecting beacons, and starting peer link finite state machine, and
* closing peer links due to transmission failures or beacon loss.

If a peer link between the sender and receiver does not exist, a frame will be 
dropped. So, the plug-in to the peer management protocol (PMP) is the first 
in the list of ``ns3::MeshWifiInterfaceMacPlugins`` to be used.

Peer management protocol
~~~~~~~~~~~~~~~~~~~~~~~~

The peer management protocol consists of three main parts:

* the protocol itself, ``ns3::dot11s::PeerManagementProtocol``, which keeps all active peer links on interfaces, handles all changes of their states and notifies the routing protocol about link failures.
* the MAC plug-in, ``ns3::dot11s::PeerManagementProtocolMac``, which drops frames if there is no peer link, and peeks all needed information from management frames and information elements from beacons.
* the peer link, ``ns3::dot11s::PeerLink``, which keeps finite state machine of each peer link, keeps beacon loss counter and counter of successive transmission failures.

The procedure of closing a peer link is not described in detail in the 
standard, so in the model the link may be closed by: 

* beacon loss (see an appropriate attribute of ns3::dot11s::PeerLink class)
* transmission failure – when a predefined number of successive packets have failed to transmit, the link will be closed.

The peer management protocol is also responsible for beacon collision avoidance, because it keeps beacon timing elements from all neighbours. Note that the PeerManagementProtocol is not attached to the MeshPointDevice as a routing protocol, but the structure is similar: the upper tier of the protocol is 
``ns3::dot11s::PeerManagementProtocol`` and its plug-in is 
``ns3::dot11s::PeerManagementProtocolMac``.

Hybrid Wireless Mesh Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HWMP is implemented in both modes, reactive and proactive, although path maintenance is not implemented (so active routes may time out and need to be rebuilt, causing packet loss). Also the model implements an ability to transmit broadcast data and management frames as unicasts (see appropriate attributes). This feature is disabled at a station when the number of neighbors of the station is more than a threshold value.

Forwarding delay
~~~~~~~~~~~~~~~~

Previous versions of this model have had issues with collisions due to
the lack of a model for forwarding delay, and lack of clarity about whether
backoff should be invoked when forwarding a broadcast frame.  These issues
become problematic for mesh, because in a topology in which multiple nodes
within radio range of the next hop node decide to forward a received
frame at the same time, they will repeatedly collide (if no backoff is
triggered).  Past contributors have argued for the triggering of backoff
when forwarding a frame (e.g., [Hep16]_), but current wifi module maintainers
concluded that the standard does not call for backoff in this case.  
This was also privately confirmed with a Wi-Fi vendor.
`Issue 478 <https://gitlab.com/nsnam/ns-3-dev/-/issues/478>`_ in the GitLab.com
tracker has more discussion on this point.

In practice, it is assumed that collisions can be avoided due to the fact
that each mesh node will take slightly different times to process and
forward the frame, and one will go first and trigger a channel busy detection
on the other nodes.  To accomplish this in ns-3, we
must include a model for forwarding delay that includes some randomness.

The class :cpp:class:`ns3::MeshPointDevice` is responsible for forwarding
unicast frames, and the class :cpp:class:`ns3::dot11s::HwmpProtocol` is
responsible for forwarding management frames when HWMP is used.
:cpp:class:`ns3::MeshPointDevice` has an attribute
called `ForwardingDelay` that configures a random variable (units of
microseconds) from which a forwarding delay value is drawn for each frame
forwarding event.  The default configuration of this attribute
is a uniform random variable between 300 and 400 microseconds.  The mean
value was chosen based on the measurement results reported in [Hep16]_ which
measured and derived an average 350 microsecond delay in forwarding frames
on real mesh devices.  The range of this variable is somewhat arbitrary and
determined by some simulation testing to provide a generally low
probability of collision; the 100 microsecond range is roughly 11 slot times.
The HWMP protocol can also access this random variable to create forwarding
delay for the forwarding of management frames.
Users may substitute other random variable configurations as desired.

Scope and Limitations
=====================

Supported features
##################

* Peering Management Protocol (PMP), including link close heuristics and beacon collision avoidance.
* Hybrid Wireless Mesh Protocol (HWMP), including proactive and reactive modes, unicast/broadcast propagation of management traffic, multi-radio extensions.
* 802.11e compatible airtime link metric.

Verification
############

* Comes with the custom Wireshark dissector.
* Linux kernel mac80211 layer compatible message formats.

Unsupported features
####################

* Mesh Coordinated Channel Access (MCCA).
* Internetworking: mesh access point and mesh portal.
* Security.
* Power save.
* Path maintenance (sending PREQ proactively before a path expires)
* Though multi-radio operation is supported, no channel assignment protocol is proposed for now. (Correct channel switching is not implemented)

Models yet to be created
########################

* Mesh access point (QoS + non-QoS?)
* Mesh portal (QoS + non-QoS?)

Open issues
###########

Users should be aware that the mesh module has not been actively maintained
for several years and that there may be some performance and
standards-alignment issues with the current code.  Below is a listing of
possible (confirmed and unconfirmed) issues.

A bug was previously reported in the Wi-Fi module that manifests itself as performance
degradation in large mesh networks, due to incorrect duplicate frame
detection for QoS data frames (https://www.nsnam.org/bugzilla/show_bug.cgi?id=2326).

Mesh does not work for 802.11n/ac/ax stations (https://gitlab.com/nsnam/ns-3-dev/-/issues/176).

Mesh PCAP is not decoded properly by Wireshark (https://www.nsnam.org/bugzilla/show_bug.cgi?id=2880).

Energy module can not be used on mesh devices (https://www.nsnam.org/bugzilla/show_bug.cgi?id=2265).

IE11S_MESH_PEERING_PROTOCOL_VERSION should be removed as per standard. 
Protocol ID should actually be part of the Mesh Peering Management IE (https://www.nsnam.org/bugzilla/show_bug.cgi?id=2600).

MeshInformationElementVector printing error (https://www.nsnam.org/bugzilla/show_bug.cgi?id=2728).

Mesh is not compatible with IPv6 (https://www.nsnam.org/bugzilla/show_bug.cgi?id=2881).

Mesh is forwarding multicast frames as unicast rather than as group-addressed frames (https://gitlab.com/nsnam/ns-3-dev/-/issues/485).

Mesh group addresses are not being set correctly for multicast frames (https://gitlab.com/nsnam/ns-3-dev/-/issues/476).
