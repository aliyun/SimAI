ns-3 RELEASE NOTES
==================

This file contains ns-3 release notes (most recent releases first).

All of the ns-3 documentation is accessible from the ns-3 website:
http://www.nsnam.org including tutorials: http://www.nsnam.org/tutorials.html

Consult the file CHANGES.html for more detailed information about changed
API and behavior across ns-3 releases.

Release 3.36.1
--------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.36.1.tar.bz2

### Supported platforms

The supported platforms are the same as for the ns-3.36 release.  Linux
systems based on musl libc may also work with this release, although this
has not been completely tested.

### New user-visible features

- (core) Remove PTHREAD-dependent classes (`ns-3::SystemMutex`, `ns-3::CriticalSection`, `ns-3::SystemThread` and `ns-3::SystemCondition`) and replace them with C++ STL libraries, respectively `std::mutex`, `std::thread` and `std::condition_variable`.

### Bugs fixed

- Various small changes were made to suppress new g++-12 compiler warnings
- (build) - Fixed issue in which localized changes to modules or scratch directory were causing a global reconfiguration and rebuild
- (build) - The optimization and debug compiler flags for the 'debug' build profile were reverted to the Waf settings, to improve the debugging experience
- (build) - .ns3rc file parsing was fixed to support multiline lists and to skip comments
- (build) - Allow module examples in subdirectories other than 'examples'
- (build) - Report more supported features in the configuration message
- (build) - Fix and improve generated pkgconfig files
- (build) - Let ns3 change the configured build profile and C++ standard
- (energy) #658 - Update energy source after updating the current drain
- (openflow) - Update documentation and add compilation flag required by Clang
- (sixlowpan) #659 - Fix UDP_NHC port decompression
- (stats) - Fix missing macro definition (STATS_HAS_SQLITE3) for sqlite3 support
- (wifi) - Check that PHY band is allowed before setting channel
- (wifi) - Fix computation of maximum allowed channel width
- (wifi) - Fix VHT Operation Channel Width encoding
- (wifi) - Always add PPDU received with unsupported modulation as interference
- (wifi) #639 - Remove FCS before deserializing Wi-Fi frames
- (wifi) - Do not change protection when aggregating MSDU to DL MU PPDU
- (wifi) - IdealWifiManager was not using the appropriate default mode

Release 3.36
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.36.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements (Note: not all ns-3 features are available on all systems):
- g++-8 or later, or LLVM/clang++-6 or later
- Python 3.6 or later
- (macOS only) Xcode 11 or later

Python API scanning only works for Python versions 3.6 through 3.8, due to
an upstream toolchain limitation.

This release has discontinued support for g++-7 compilers.

### New user-visible features

- (build system) The ns-3 build system has been changed to CMake from Waf.  A wrapper script called 'ns3' provides a Waf-like command-line API.
- (build system) The default build profile has been changed from 'debug' to a new 'default'.  Two key differences are that the new default has optimizations enabled (-O2 vs. previous -O0), and the -Werror flag is disabled.  Select the 'debug' profile to disable optimizations and enable warnings as errors.
- (core) An attribute value type for a std::tuple has been added
- (core) Simulation events can be created from lambda expressions
- (core) TypeId constructors now accept std::string
- (fd-net-device) Support for PlanetLab helper was removed.
- (lte) Handover now works with carrier aggregation configurations.
- (mesh) The mesh model now includes a randomized per-hop forwarding delay.
- (spectrum) ThreeGppSpectrumPropagationLossModel and ThreeGppChannelModel now support multiple PhasedArrayModel instances per device. This feature can be used to implement MIMO.
- (spectrum) A new phased array spectrum propagation loss model, called PhasedArraySpectrumPropagationLossModel, has been added to better support different 4G/5G MIMO models that can exploit the multiple subarray concept.
- (spectrum) ThreeGppSpectrumPropagationLossModel and ThreeGppChannelModel now support multiple PhasedArrayModel instances per device.
- (spectrum) SpectrumPhy now also supports the new antenna arrays in addition to the previous antenna types.
- (tcp) Support for Network Simulation Cradle was removed.
- (traffic-control) Flow control features can be disabled from device helpers.
- (wifi) The default Wi-Fi standard has been upgraded from 802.11a to 802.11ax.
- (wifi) The default Wi-Fi rate control has been changed from ArfWifiManager to IdealWifiManager.
- (wifi) The framework for consistently configuring operating channel, standard, band, channel width, and primary channel was changed.

### Bugs fixed

- (build): #461 - Avoid period character in executable names
- (core) #141 - Print parent attributes from CommandLine --PrintAttributes
- (core) #487 - Refactor double to int64x64 conversion test for platforms with less than 64 bit mantissa
- (internet) #431 - ICMPv6 error messages carry packet metadata
- (internet) - Fix RIPng address selection for intermediate nodes
- (internet) - Fix source address selection for localhost
- (internet) - Read correct bits when deserializing IPv6 flow label
- (internet) -  Fix errors in IPv6 Path MTU discovery.
- (internet-apps) #576 - Fix V4TraceRoute crash at application stop
- (lte) #592 - Update lte-rrc-protocol-ideal.cc to solve SIB2 repeatedly reception issue
- (mesh) #472 - mesh.cc example distance too large for preamble detect
- (mesh) #478 - Persistent collisions of forwarded broadcast frames
- (mpi) #607 - Call CommandLine correctly to generate docs
- (nix-vector) - Remove unused BuildNixVectorLocal () function.
- (nix-vector) #466 - Handle loopback IPv4/IPv6 addresses
- (nix-vector) #567 - Rebuild NixVector if topology changes
- (spectrum) Fix condition for channel matrix update in ThreeGppChannelModel (left and right operand were pointing to the same object)
- (spectrum) Assign stream to random variable for Doppler term in ThreeGppChannelModel (moved from ThreeGppSpectrumPropagationLossModel)
- (tcp) #531 - TcpWestwood divide-by-zero and floating point issues
- (traffic-control) !844 - Fix value of MaxBurstAllowance in PIE
- (wifi) - Mark MPDU as acknowledged before removing from the in-flight queue
- (wifi) - Fix issue with setting of QosSupported attributes
- (wifi) - Fix issue with installing QosTxop objects
- (wifi) - Initialize WifiUlMuMultiStaBa as MULTI_STA
- (wifi) - Fix misinterpretation of 802.11ax specs
- (wifi) - Restore configuration of CWmin and CWmax for 802.11b
- (wifi) - Set wifi-spatial-reuse.cc example to use 20 MHz BW
- (wifi) - Fix assert condition in HeFEM::ReceiveMpdu()
- (wifi) - Workaround to avoid a crash when expecting a TB PPDU and receiving a pre-11ax PPDU
- (wifi) - Fix Duration/ID setting when TXOP Limit is exceeded
- (wifi) - Enable protection for DL MU PPDUs including one PSDU, but disable for more than one PSDU until MU-RTS is implemented
- (wifi) - Fix L-SIG length computation for HE TB PPDUs
- (wifi) - Clear PSDU map of HE FEM when needed
- (wifi) #459 - Fix HeCapabilities IEEE 802.11ax-compliance
- (wifi) #467 - Fix recording of failed association process
- (wifi) #468 - Fix wrong txDuration for trigger frame
- (wifi) #473 - Discard preamble state for truncated TX or device in OFF
- (wifi) #475 - Assert when sending OFDMA DL to STAs with different TIDs
- (wifi) #480 - 2.4GHz remote station manager internal collision issue
- (wifi) #483 - Cancel EndOfMpdu events when cancelling all events
- (wifi) #500 - Clear PHY RX event if switching channel in IDLE and CCA_BUSY state
- (wifi) #507 - Set CS Required field according to 802.11ax specs
- (wifi) #600 - Fix HT Operation Primary Channel setting
- (wifi) #609 - STA checks that a Multi-STA BA is sent by the AP it is associated with

Release 3.35
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.35.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements (Note: not all ns-3 features are available on all systems):
- g++-7 or later, or LLVM/clang++-8 or later
- Python 3.6 or later
- (macOS only) Xcode 11 or later

Python API scanning only works for Python versions 3.6 through 3.8, due to
an upstream toolchain limitation.

This release has been tested on the following systems:
- Fedora 34 with g++-11.2.1 and clang++-12.0.1; Python 3.9.7
- Fedora 33 with g++-10.3.1 and clang++-11.0.0; Python 3.9.6
- Ubuntu 21.04 with g++-11.1.and clang++-12.0.0; Python 3.9.5
- Ubuntu 20.04 with g++-9.3.0 and clang++-10.0.0; Python 3.8.10
- Ubuntu 18.04 with g++-7.5.0 and clang++-6.0.0; Python 3.6.9
- macOS 10.15.7 (Catalina) with Xcode 12.4 (Apple clang-1200.0.32.29, Python 3.9.1)

This release has discontinued support for g++-5 and g++-6 compilers, and
for macOS Xcode 10.

### New user-visible features

ns-3 has switched to the C++17 standard by default, although the more
advanced constructs (beyond C++11) are generally avoided in the header files,
to stay compatible with the Python API scanning framework.

- (antenna) Extend UniformPlanarArray to have a configurable polarization slant angle
- (buildings) Add example program demonstrating group mobility with the buildings-aware random walk mobility model, and a script to animate the group motion.
- (config-store) OBSOLETE Attributes are not anymore saved.
- (config-store) New ConfigStore::SaveDeprecated Attribute (default false) to avoid to save DEPRECATED attributes.
- (core) Add TracedCallback::IsEmpty to know if a TracedCallback has any callback associated.
- (core) Add std::hash<ns3::Ptr<T>> to allow seamless use of Ptr as key in unordered_maps.
- (core) Print available values on EnumChecker::GetValue () error
- (internet) Added getters and setters for source and destination in Ipv6Header consistent with Ipv4Header naming.
- (internet) New functions SetAddress () and GetAddress () are added, corresponding to SetLocal () and GetLocal () respectively in Ipv4InterfaceAddress to keep consistency with Ipv6InterfaceAddress.
- (nix-vector-routing) Support for IPv6 Nix-Vector routing. Existing nix-simple and nms-p2p-nix examples are available in IPv6 as well.
- (nix-vector-routing) ipv4-nix-vector-helper.h has been deprecated in favour of nix-vector-helper.h.
- (nix-vector-routing) ipv4-nix-vector-routing.h has been deprecated in favour of nix-vector-routing.h.
- (internet) Added new function IsInSameSubnet () in Ipv4InterfaceAddress similar to Ipv6InterfaceAddress::IsInSameSubnet (). It checks if the two Ipv4Address are in the same subnet.
- (nix-vector-routing) Nix-Vector routing supports topologies with multiple WiFi networks using the same WiFi channel object.
- (mobility) Add GroupMobilityHelper and demonstrate its use with a new program that configures the Reference Point Group Mobility model for a number of nodes.
- (wifi) Add support for 11ax MU EDCA Parameter Set.
- (wifi) MPDUs discarded because they became "old" (from the Block Ack agreement point of view) are now traced using the DroppedMpdu trace source of RegularWifiMac and with the WIFI_MAC_DROP_QOS_OLD_PACKET reason.

### Bugs fixed

- (applications) #449, #453 - add correct receiving address to PacketSink traces for UDP.
- (config-store) #443 - do not save obsolete attributes
- (energy) #423 - SimpleDeviceEnergyModel calculation of consumed energy
- (fd-net-device) #441 - Fix double dequeue on device stop
- (internet) - Ipv4L3Protocol duplicate detection now accounts for sent packets, not just forwards.
- (lte) !703 - Fix global-buffer-overflow when nprb is equal to zero
- (mobility) - Initialize HierarchicalMobilityModel and avoid course change stream corruption
- (nix-vector-routing) #450 - Avoid recalculation if path doesn't exist
- (tcp) #407 - Align TcpCubic::GetSsThresh() with Linux
- (tcp) #410 - Fix TcpWestwood bandwidth estimation bug
- (traffic-control) #258 - Avoid overflow, divide-by-zero in TbfQueueDisc
- (traffic-control) !706 - Fix FqPIE qdisc not using attributes
- (wifi) - Always track interference when not locking on PPDU

Release 3.34
-------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.34.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements (Note: not all ns-3 features are available on all systems):
- g++-7 or later, or LLVM/clang++-8 or later
- Python 3.6 or later
- (macOS only) Xcode 10.1 or later

This release has been tested on the following systems:
- Fedora 33 with g++-10.2.1 and clang++-11.0.0; Python 3.9.2
- Ubuntu 21.04 with g++-11.1.0 and clang++-12.0.0; Python 3.9.5
- Ubuntu 20.04 with g++-9.3.0 and Python 3.8.5
- Ubuntu 18.04 with g++-7.5.0 and Python 3.6.9
- macOS 10.15.7 (Catalina) with Xcode 12.4 (Apple clang-1200.0.32.29, Python 3.9.1)
- macOS 10.13.6 (High Sierra) with Xcode 10.1 (Apple clang-1000.11.45.5, Python 3.7.1)

### New user-visible features

- (antenna) Added PhasedArrayModel, providing a flexible interface for modeling a number of Phase Antenna Array (PAA) models
- (antenna) Replaced the ThreeGppAntennaArrayModel with a UniformPlanarArray model, extending the new PhaseAdrrayModel
- (antenna) Improved the Angles class to be more robust and user-friendly.
- (antenna) AntennaModel child classes have been extended to produce 3D radiation patterns
- (internet) Support for IPv6 stateless address auto-configuration (SLAAC).
- (network) Improved support for bit fields in header serialization/deserialization.
- (network) Added support for DLT_LORATAP DataLinkType to PCAP files
- (network) More arithmetic operators are provided for DataRate objects
- (nix-vector) Nix-Vector routing supports multiple interface addresse and can print out routing paths.
- (olsr) Add support for printing OLSR headers
- (sixlowpan) Added support for stateful (i.e., context-based) RFC6282 compression.
- (tcp) TCP CUBIC is now the default TCP congestion control, replacing NewReno.
- (tcp) A BBRv1 congestion control model has been added.
- (traffic-control) Added FqCobalt queue disc with L4S features and set associative hash.
- (traffic-control) Added FqPIE queue disc with L4S mode.
- (wifi) Add support for 802.11ax DL and UL OFDMA, and a round-robin multi-user scheduler
- (wifi) Added 802.11ax support to MinstrelHt rate control algorithm.
- (wifi) The PHY layer has been refactored: the amendment-specific logic has been ported to PhyEntity classes and WifiPpdu classes
- (wifi) The MAC layer has been refactored. The MacLow class has been replaced by a hierarchy of FrameExchangeManager classes, each adding support for the frame exchange sequences introduced by a given amendment.
- (wifi) Add ThompsonSamplingWifiManager rate control algorithm.
- (wifi) Holland PHY configuration has been removed from the model
- (wifi) HT Greenfield (HT_GF) preamble support has been removed from the model
- (wifi) Added the ability to configure the primary 20 MHz channel for 802.11 devices operating on channels of width greater than 20 MHz
- (wifi) Some wifi/src/model files were moved to the relevant subfolders (non-ht, ht, vht, he, and rate-control)
- (wifi) Stations perform TXOP recovery if the transmission of a non-initial MPDU in a TXOP fails
- (wifi) Stations keep track of the TXOP holder and ignore the NAV when they receive an RTS frame from the TXOP holder
- (wifi) The TxOkHeader and TxErrHeader trace sources of RegularWifiMac have been obsoleted and replaced by trace sources that better capture the result of a transmission (AckedMpdu, NAckedMpdu, DroppedMpdu, MpduResponseTimeout and PsduResponseTimeout)
- (wifi) Bianchi example program extended to support 802.11n/ac/ax rates
- (wifi) ErrorRateModel API extended to support link-to-system models

### Bugs fixed

- (antenna) Fix random angle generation for the 3gpp channel model.
- (build) #52 - Fix and standardize Boost detection
- (build) #352 - Handle multiple attempts to configure C++ standard
- (core) #349 - Warn about Config::SetDefault() of deprecated attribute
- (internet) #385 - IPv6 NA are now sent directly by the NetDevice
- (lr-wpan) #382 - Fix Tx power calculation from PIB attribute
- (lte) #285 - Check the RandomAccess status of an RNTI to whom the preamble was assigned
- (lte) #322 - Add Channel pointer value type attribute to REM helper
- (lte) #354 - Use uint16_t for srsIndex instead of uint8_t
- (lte) #395 - Add transmission buffer size for RLC Acknowledged Mode
- (lte) !555 - Remove assert upon RAR timeout in RLF example and..
- (lte) !605 - Make sure RNTI exists before removing UE context
- (lte) - Set UeRrcSapProvider only if RNTI exists at the eNB
- (nix-vector) #393 - Avoid changing routing-dependent members
- (sixlowpan) #358 - Use correct fragment size when using mesh routing.
- (tcp) #374 - Assume TCP RTT is one microseconds if it measures as zero
- (uan) #129 - EndTx moved to PhyListener
- (uan) #293 - Fixed integer truncation in window computation
- (wave) Fix compilation and runtime issues with vanet-routing-compare.cc
- (wifi) #418 - Fix HE A-MPDU Max Size Upper Bound
- (wifi) - Compare received power per MHz to normalized RX sensitivity

Release 3.33
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.33.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements (Note: not all ns-3 features are available on all systems):
- g++-5.4 or later, or clang++-8 or later
- Python 3.5 or later
- (macOS only) Xcode 10.1 or later

This release has been tested on the following systems:
- Arch Linux with g++-10.2.0 and clang++ versions 8-11, Python 3.9.1
- Fedora 33 with g++-10.2.1 and clang++-11.0.0; Python 3.9.1
- Ubuntu 20.04 with g++-9.3.0 and Python 3.8.5
- Ubuntu 18.04 with g++-7.5.0 and Python 3.6.9
- Ubuntu 16.04.6 with g++-5.4.0 and Python 3.5.2
- Linux Mint 20 Ulyana with g++-9.3.0 and Python 3.8.5
- macOS 11.1 (Big Sur) with Xcode 12.3 (Apple clang-1200.0.32.28), Python 3.8.2 (system) and Python 3.9.1 (Homebrew)
- macOS 10.15.7 (Catalina) with Xcode 12 (Apple clang-1200.0.31.1, Python 3.9.1)
- macOS 10.13.6 (High Sierra) with Xcode 10.1 (Apple clang-1000.11.45.5, Python 3.7.1)

### New user-visible features

- (core) A new Length class has been added to represent lengths with an explicit unit, rather than the use of raw numbers (ints, doubles) with implicit length.
- (core) A flexible CsvReader class has been introduced to allow users to read in csv- or tab-delimited data.
- (lr-wpan) Added support for handling differences between RFC4944- and RFC6282-style IPv6 address expansion
- (mobility) The ListPositionAllocator can now input positions from a csv file.
- (mpi) A new Enable call was added that takes a user supplied MPI_Communicator, which enables (optional) partitioning of the MPI processes.  See !389.
- (propagation) A channel condition model for vehicular scenarios was added.  See !435
- (tcp) A model for TCP CUBIC congestion control has been added.
- (tcp) TCP handling of ECN has been improved with the addition of a Congestion Window Reduced state
- (tcp) TCP DCTCP includes a trace source for tracing the alpha value
- (wifi) A new table-based error model based on link simulations has been
  added (TableBasedErrorRateModel) and made the default for 802.11n/ac/ax simulations.
- (wifi) Add error table support for LDPC FEC encoding (BCC or LDPC may now be selected)
- (wifi) Add PHY layer support for High Efficiency Multi Users [HE-MU] access, including High Efficiency Resource Units (HE RU) and PCAP support
- (wifi) Add option to configure the OFDM transmit spectrum mask

### Bugs fixed

- (applications) #444 - Report Tx bytes from UdpClient
- (core) #265 - Time is not rounded when created from an int64x64_t
- (core) #277 - Make Time::As() print with autoscaled unit
- (core) #302 - Time is not rounded when created from an int64x64_t
- (internet) #272 - Add missing data to Ipv[4,6]PacketInfoTag
- (internet) !440 - Loopback IPv4 address should have HOST scope
- (lte) #273 - When removing LC, remove also its BSR
- (lr-wpan) #314 - Multicast/broadcast must not generate ACKs
- (mobility) #276 - Make EnableAsciiAll connect in a failsafe way
- (tcp) TcpCongestionOps objects can now be accessed by config paths
- (tcp) #783 - Update previousLost and priorInFlight on reception of ACK
- (wifi) #284 - Fix memory consumption issue for saturated devices
- (wifi) Fix Minstrel HT statistics update window duration
- (wifi) Fix incorrect calculations for 1024-QAM when using NistErrorRateModel
- (wifi) Fix maximum channel number that can be selected for the PHY
- (wifi) Set BER threshold for Ideal rate manager to 1e-6 to avoid high PER for high MCS values
- (wifi) Reduce precision to fix infinite loop when Ideal rate manager as used with Yans error rate model
- (wifi) Report reception of BAR to WifiRemoteStationManager
- (wifi) TXOP durations were not always being traced

Release 3.32
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.32.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements (Note:  not all ns-3 features available on all systems):
- g++-4.9 or later
- Xcode 10.1 or later
- Python 3.5 or later

Starting with ns-3.30, Python 3 has been the only supported Python version,
and scripts default to Python 3 usage.

This release has been tested on the following systems:
- Fedora 32 with g++-10.2.1 and Python 3.8.5
- Fedora 32 with clang-10.0.0 and Python 3.8.5
- Ubuntu 20.04 with g++-9.3.0 and Python 3.8.2
- Ubuntu 18.04 with g++-7.5.0 and Python 3.6.9
- Ubuntu 16.04.6 with g++-5.4.0 and Python 3.5.2
- macOS 10.15.7 (Catalina) with Xcode 12
- macOS 10.13.6 (High Sierra) with Xcode 10.1

### New user-visible features

- (build system) Added "--enable-asserts" and "--enable-logs" to waf configure,
  to selectively enable asserts and/or logs in release and optimized builds.
- (build system) The "--enable-build-version" option has been added to waf
  configure, allowing users to embed version information in the libraries
- (build system) Added support for EditorConfig
- (config-store) Improved the RawTextConfig output (see !93)
- (core) Added capability to configure STL pair and containers as attributes
- (core) The default timestamp print format for logging statements has been
  changed to use the As() method that prepends +/- and appends the time unit
- (fd-net-device) Added two enhanced emulation devices based on netmap and DPDK
- (mobility) Add CartesianToGeographic coordinate conversion capability
- (network) Packet tags are now serialized and deserialized for distributed
  simulations
- (network) Added LollipopCounter, a sequence number counter type
- (olsr) Some internal OLSR API has been opened up to enable cross-layer access
- (tcp) Support for dynamic pacing in TCP.
- (tcp) Added TcpLinuxReno congestion control (aligns with Linux 'reno'
- (tests) test.py text output now distinguishes between TestSuites and TestCases
  congestion control).
- (traffic-control) Added, to PIE queue disc, queue delay calculation using
  timestamp feature (Linux default behavior), cap drop adjustment feature
  (Section 5.5 of RFC 8033), ECN (Section 5.1 of RFC 8033) and derandomization
  feature (Section 5.4 of RFC 8033).
- (traffic-control) Add support for L4S mode to CoDel and FqCoDel queue discs
- (traffic-control) Add Active/Inactive feature to PIE queue disc.
- (traffic-control) Add ECN and L4S support for CoDel, FqCoDel, Cobalt queues
- (wifi) Add 6 GHz band support for 802.11ax
- (wifi) RIFS support has been removed from the model

### Bugs fixed

- (applications) Handle possible partial socket sends (during emulation)
- (applications) Remove stray exit call from OnOffApplication
- (internet-apps) Fixes DHCP behaviour when interface goes down and back up
- (lte) #224 - The scheduled SendMeasurementReport() event doesn't cancel after link failure detection
- (mesh) - Enable possible use with SpectrumWifiPhy
- (network) #264 - Rounding is more accurate for DataRate transmission times
- (network) !424 - Fixed instantaneous send behavior in SimpleNetDevice when queue is empty
- (tcp) !368 - Fix assert in tx-buffer when loss rate high and w/o SACK
- (tcp) !367 - Stop to merge items whose m_lost values are different
- (tcp) - Fix TcpRateOps to update m_ackElapsed like Linux
- (traffic-control) #225 - Do not ECN mark packet twice within a CoDel queue
- (wifi) - Fix radiotap header for received 802.11ax PPDUs
- (wifi) - Do not send VHT capabilities when operating in 2.4 GHz band
- (wifi) #252 - Correctly handle multicast frames

Release 3.31
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.31.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements (Note:  not all ns-3 features available on all systems):
- g++-4.9 or later
- Xcode 10.1 or later
- Python 3.5 or later

Starting with ns-3.30, Python 3 has been the only supported Python version,
and scripts default to Python 3 usage.

This release has been tested on the following systems:
- Fedora 32 with g++-10.1.1 and Python 3.8.3
- Fedora 32 with clang-10.0.0 and Python 3.8.3
- Fedora 29 with g++-8.3.1 and Python 3.7.5
- Ubuntu 20.04 with g++-9.3.0 and Python 3.8.2
- Ubuntu 18.04 with g++-7.5.0 and Python 3.6.9
- Ubuntu 16.04.6 with g++-5.4.0 and Python 3.5.2
- macOS 10.15.3 (Catalina) with Xcode 11.3
- macOS 10.13.6 (High Sierra) with Xcode 10.1

### New user-visible features

- (antenna) Model for antenna arrays based on 3GPP model (part of Integration of the 3GPP TR 38.901 fast fading model)
- (buildings) Random walk mobility model that does not allow nodes to enter buildings
- (core) The CommandLine facility can now add the Usage message to the Doxygen for the program
- (internet) TCP DCTCP model
- (internet) An option to enable IPv4 hash-based multicast duplicate packet
  detection (DPD) based on RFC 6621
- (lte) Trace sources for uplink PSD and RBs
- (propagation) 3GPP TR 38.901 pathloss and channel condition models
- (spectrum) Matrix-based channel model base class to support antenna arrays
- (spectrum) 3GPP pathloss, channel and fast fading models
- (stats) A new interface to SQLite
- (tests) Capability to run an example program as a regression test
- (traffic-control) Support for the Cake set-associative hash to the FqCoDel queue disc
- (traffic-control) Support for ECN marking to CoDel and FqCoDel queue discs
- (wifi) A Bianchi (saturation) example with a comparable MATLAB model
- (wifi) The idealized wifi rate control includes better support for different MIMO modes and channel widths

### Bugs fixed

If available, the numbers below make reference to a Bugzilla bug number, GitLab
issue number (prefixed by '#'), or GitLab merge request number (prefixed by '!')
- (build system) #119 - Waf --lcov-report option was broken
- (buildings) #80 - Update indoor/outdoor status of a moving node
- (core) Bug 2725 - EmpiricalRandomVariable should not interpolate
- (documentation) Bug 2636 - Add to doxygen a list of all registered TypeIds
- (internet) - Fix Ipv6Prefix length calc and add explicit prefix length constructors
- (internet) - IPv6 Duplicate Address Detection (DAD) must use Solicited-Node multicast address
- (internet) #111 - Rip and RipNg don't receive unicast packets
- (internet:tcp) - call NotifyConnectionFailed on SYN timeout
- (internet:tcp) #179 - Avoid incorrect exit of LAST-ACK state
- (internet:tcp) #182 - Fix SACK list cleaning in TcpOptionSack::Deserialize()
- (internet:tcp) !156 - Enable entering CA_RECOVERY despite large sequence number increase
- (internet:tcp) !157 - Prevent sending outside receiver window
- (lr-wpan) #212 - Rx callback must have both source mac and pan id
- (lr-wpan) !283 - Beacon collisions and timings
- (lr-wpan) !326 - Inactive Periods Queue fix
- (lte) #106 - Inconsistent imsi representation in LTE module
- (lte) #196 - Remove LteSecondaryCellSelectionTestSuite from LTE module
- (lte) #221 - Stop T310 upon receving handover command
- (network) #216 - Correct the return value for PacketSocket::SendTo()
- (network) !239 - Refactor DelayJitterEstimation to more closely follow RFC 1889 and RFC 3550
- (propagation) !269 - Fix Okumura-Hata propagation loss model for frequency > 1.5 Ghz and medium or small city
- (wifi) - Keep the medium busy if reception is aborted ahead of scheduled time
- (wifi) - Fix frame capture when signals arrive at the exact same time and add additional tests to verify these cases
- (wifi) - Fix MCS selection in ideal rate manager for cases with unbalanced MIMO settings and/or RX diversity
- (wifi) - Fix SNR computations for MIMO
- (wifi) - Fix ReportAmpduTxStatus called two times when BAR is explicitely sent upon missed BACK
- (wifi)  A zero value for the backoff timer might be discarded and a new value
  might be generated by an erroneous call to NotifyCollision().
- (wifi) Bug 1909, Issue #41 - Implementation of ACK timeout
- (wifi) Bug 2385, Issue #178 - IdealWifiManager can pick invalid WifiTxVector under certain MIMO conditions
- (wifi) Bug 2928 - BlockAckManager::NeedBarRetransmission returns "true" infinitely
- (wifi) Bug 2985 - PhyTxEnd tracecallback not implemented
- (wifi) Bug 3011, Issue #172 - MIMO error rate model is incorrect
- (wifi) #22 - Station long retry counter is incremented twice if BlockAck was not received
- (wifi) #40 - IdealWifiManager not working if stations are moving
- (wifi) #84 - Wi-Fi removing wrong header due to copy-paste error
- (wifi) #94 - Retransmitted frames not marked as retransmits
- (wifi) #107 - Frame capture model works incorrectly if two signals arrive at the same time
- (wifi) #110 - PCAP trace file gives unexpected results
- (wifi) #116 - Incorrect state when receiving A-MPDU in WifiPhy
- (wifi) #165 - Notify station manager when a BlockAck has not been received after the transmission of an A-MPDU
- (wifi) #169 - Ideal rate manager does not work when non best-effort traffic is used
- (wifi) #211 - Remove MSDUs with expired lifetime when (re)starting channel access

### Known issues

In general, known issues are tracked on the project tracker available
at https://gitlab.com/nsnam/ns-3-dev/issues

Release 3.30.1
--------------

Release 3.30.1 is a maintenance release that fixes the following issues from
the ns-3.30 release:

- fix issue with Block Ack transmit window leading to low throughput or
  connection dropping
- fix invalid restriction on Wi-Fi VHT Capabilities MaxMpduLength field
- support for Apple clang version 11.0.0 (macOS Catalina preview release)
- fix the handling of ICMP time exceeded responses
- fix syntax error in Wi-Fi-based Tap Bridge Python example

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.30.1.tar.bz2

### Supported platforms

The list of supported platforms includes those listed for ns-3.30 plus
- macOS 10.15 preview (Catalina) with Apple clang version 11.0.0

### New user-visible features

Features are identical to release 3.30.

### Bugs fixed

- Issue #62 - ICMP does not handle ICMPv4 TIME_EXCEEDED responses
- Issue #77 - Wi-Fi VHT capabilities MaxMpduLength had invalid restriction
- Issue #79 - Wi-Fi connection drop or low throughput due to Block Ack transmit window
- Example program tap-wifi-virtual-machine.py had a syntax error
- Issue #80 - Indoor/outdoor status is not updated when the node moves

Release 3.30
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.30.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements (Note:  not all features available on all platforms):
- g++-4.9 or later
- Xcode 10.1 or later
- Python 3.5 or later

Starting with ns-3.30, Python 3 will be the only supported Python version,
and scripts will default to Python 3 usage.  Python 2.7 may continue to work
in the near term.

This release has been tested on the following systems:
- Fedora 30 with g++-9.1.1
- Ubuntu 19.04 with g++-8.3.0
- Ubuntu 18.04 with g++-7.4.0
- Ubuntu 16.04.6 with g++-5.4.0
- Linux Mint 19 Tara with g++-7.3.0
- macOS 10.14.6 (Mojave) with Xcode 10.3
- macOS 10.13.6 (High Sierra) with Xcode 10.1

### New user-visible features

- (core) ShowProgress object can be used to report on simulation execution
- (wifi) Preamble detection can now be modelled and is enabled by default.
- (wifi) 802.11ax spatial reuse is now supported.
- (lte) Radio Link Failure (RLF) functionality is now supported.
- (lte) LTE/EPC model enhanced with new features: SGW, PGW, MME are full nodes; new S5 interface between SGW and PGW, allows simulations with multiple SGW/PGW
- (lte) LTE eNB RRC extended to support improved S1 signalling model
- (lte) Backhaul links can now use any link technology, not just point-to-point
- (traffic-control) Cobalt queue disc model has been added.

### Bugs fixed

During this release cycle, the project transitioned from a Bugzilla issue
tracker to a GitLab.com issue tracker.  Bugs fixed from the Bugzilla tracker
include:

- Bug 2339 - lte: Use only the active RBs for uplink PSD computation
- Bug 2460 - wifi: Refactor detection thresholds in WifiPhy and add a new sensitivity threshold to throw away weak signals
- Bug 2470 - wifi: Handshake to setup the Block Ack Agreement is not protected
- Bug 2637 - build: macOS SDK headers in non-standard place
- Bug 2860 - mobility: Set Z coordinate for position-allocation classes
- Bug 2861 - lte: Forward TXOPs to the RLCs of SCCs
- Bug 2893 - lte: GetPgw in helper should be const
- Bug 2979 - lte: Remove unused code from UE RRC and UE CCM
- Bug 2980 - lte: Reset MAC and PHY of secondary carriers upon handover
- Bug 2982 - lte: Configure UE MAC SAP LteMacSapUser properly
- Bug 2988 - lte:  Correction in L3 filtering formula used for UL power control
- Bug 2990 - internet: IPv6 packet shouldn't be marked with SocketIpTtlTag
- Bug 2992 - lte: Send method of the LteUeNetDevice doesn't use protocolNumber parameter
- Bug 2993 - build: Waf --enable-sudo option broken
- Bug 2997 - lte: EpcTft::PacketFilter::Matches does not use ipv6 address to match an IP packet
- Bug 3007 - build: Boost library configuration in contrib
- Bug 3027 - lte: S1 signalling is done before RRC connection establishment is finished

Bugs fixed from the GitLab.com issue tracker include:

- Issue #11 - mobility: Rectangle::GetClosestSide returns the correct side also for positions outside the rectangle
- Issue #14 - build: Check file timestamps instead of hashes for changes
- Issue #15 - build: Fix build erroneous order constraint warnings
- Issue #27 - wifi: Re-compute the A-MPDU after an RTS/CTS exchange fails
- Issue #28 - wifi: wifi error model: coded bits vs. data bits
- Issue #30 - wifi: Sequence number wrap around in MacLow::ReceiveMpdu
- Issue #32 - wifi: Example "rate-adaptation-distance.cc" fails.
- Issue #33 - wifi: Issues with QosTxop::StartNextPacket
- Issue #35 - wifi: Unexpected EDCA performance under virtual collisions for 802.11n
- Issue #42 - wifi: Check configured capture window for capture
- Issue #43 - core: Check if ObjectFactory has been configured
- Issue #45 - wifi: Wi-Fi transmits frames outside BlockAck window
- Issue #46 - internet: Export headers for transport protocol modularity
- Issue #47 - core: Return non-const reference when a const Ptr is dereferenced
- Issue #48 - wifi: Incorrect sequence comparison in BlockAckManager::NeedBarRetransmission
- Issue #53 - lte: Token Bank Fair Queue Scheduler is creating a transmit opportunity of 0 bytes
- Issue #54 - lte: RadioBearerStats are not correctly connected
- Issue #70 - visualizer: Update configuration of GooCanvas.CanvasEllipse
- Merge request !19 - wifi: Rework MSDU and MPDU aggregation code
- Merge request !22 - spectrum: Fix order of PHYs stored in MultiModelSpectrumChannel
- Merge request !30 - lte: Fix PDU delivering at the edge of reordering window
- Merge request !43 - traffic-control: Install FqCoDelQueueDisc by default
- Merge request !55 - lte: Reset preamble transmission count before starting NC based RACH
- Merge request !65 - tcp: Call BytesInFlight method before processing ack
- Merge request !67 - wifi: Pass number of data bits instead of coded bits to error models
- Merge request !85 - uan: Fix MAC address variable shadowing in subclasses

Other issues fixed:

- core: Replace int64_t operator/(Time,Time) with exact int64x64_t version, add int64x64_t scaling.
- core: Refactor LogTimePrinter, LogNodePrinter
- energy: Update remaining energy even if simulation has finished
- flow-monitor:  Fix FlowMonitor::Stop () method
- mobility: Allow AssignStreams after construction for SteadyStateRandomWaypointMobilityModel
- lte: Add attribute to configure the periodicity of DL CQI reports
- lte: Fix type of them_rapIdRntiMap key
- lte: Various updates of the ComponentCarrier subclasses to accommodate NR
- lte: Fix issue with use of packet tag in LTE PDCP and RLC by using byte tag
- visualizer:  Fix various issues with Python 3 and X server compatibility
- wifi: Fix A-MPDU reception logic
- wifi: Fix noise interference calculation
- wifi: Do not allow S-MPDU transmission as long as ADDBA handshake not established
- wifi: Fix frequencies for channels 54 and 126
- wifi: Fix rate used to send Block Ack Request frames
- wifi: Check that SSID in probe requests is either broadcast or corresponds to AP SSID
- wifi: Do not call ReportDataFailed if retransmission not needed
- wifi: Fix duplicate MPDUs sent in an A-MPDU when explicit BARs are disabled
- wifi: Fix issues when MSDU lifetime is expired
- wifi: Enforce capture window duration in frame capture model, and fix capture model for A-MPDU
- wifi: Correct handling of timeout situations during block ack agreement setup
- wifi: Return correct serialized size for AmpduTag
- wifi: Fix incorrect channel width when HT, VHT, or HE STA received legacy PPDU

### Known issues

In general, known issues are tracked on the project tracker available
at https://gitlab.com/nsnam/ns-3-dev/issues

Release 3.29
------------

This release is dedicated to the memory of project co-founder George Riley.

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.29.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements (Note:  not all features available on all platforms):
- g++-4.9 or later
- Apple LLVM version 7.0.2 or later
- clang-3.3 or later

In addition, Python 2.7 (Python 2 series) or Python 3.4-3.7 (Python 3 series)

This release has been tested on the following platforms:
- Ubuntu 18.04 (64 bit) with g++-7.3.0 and Python 2.7.15
- Ubuntu 16.04 (64 bit) with g++-5.4.0 and Python 2.7.12/3.5.2
- Fedora Core 28 (64 bit) with g++-8.1.1 and Python 2.7.15/3.7.0
- Fedora Core 26 (64 bit) with g++-7.3.1 and Python 2.7.14/3.6.5
- macOS High Sierra 10.13.5 with Xcode 9.4.1, Apple LLVM version 9.1.0, Python 2.7.10

### New user-visible features

- (buildings) A new position allocator has been added to the buildings module.
  The allocator places nodes randomly but in a manner that rejects positions
  that are located within buildings defined in the scenario.
- (applications) Add 3GPP HTTP model
- (traffic-control) Add priority queue disc (PrioQueueDisc)
- (tcp) Added PRR as recovery algorithm
- (wifi) Add a new trace source to StaWifiMac to trace beacon arrivals
- (network) Add a method to allow random variable-based jitter to be added
  to the start times of applications in a container.
- (network) Add a method to check whether a node with a given ID is within
  a NodeContainer.
- (spectrum) A new trace source named "Gain" is added to the SpectrumChannel class.
  This trace is fired whenever a new path loss value is calculated. It exports pointers
  to the mobility model of the transmitter and the receiver, Tx antenna gain, Rx antenna gain,
  propagation gain and the pathloss value.
- (wifi) It is now possible to transmit A-MPDUs containing up to 256 MPDUs (802.11ax only)

### Bugs fixed

- Bug 2299 - uan: WOSS integration missing patch
- Bug 2399 - wifi: Improve scanning procedure of StaWifiMac
- Bug 2451 - bindings: (partial fix) generating ILP32 bindings from LP64
- Bug 2461 - core: CommandLine should handle non-option arguments
- Bug 2772 - bridge:  Bridge doesn't learn from packets sent to it
- Bug 2801 - fd-net-device: FdNetDevice device MTU is not set correctly
- Bug 2819 - traffic-control: FqCoDel handling of non-IP packets
- Bug 2891 - netanim:  dumbbell-animation breaks when RightCount > LeftCount
- Bug 2901 - core: Add CommandLine::Parse (const std::vector<std::string>> args)
- Bug 2902 - build: missing modules cause Waf to run indefinitely
- Bug 2908 - build:  Check if pybindgen version exists before use
- Bug 2911 - aodv:  Binary exponential backoff can become unlimited
- Bug 2914 - tcp: Adv Win resilience to SequenceNumber wrap-around
- Bug 2915 - build: Enforce gcc minimum version
- Bug 2920 - wifi: Default MaxSlrc and fragmentation threshold values differ from standard
- Bug 2921 - tcp: Add min_cwnd variable to LEDBAT
- Bug 2923 - tap-bridge: UseLocal mode uses wrong MAC address
- Bug 2924 - traffic-control:documentation about Peek/Dequeue usage
- Bug 2925 - wifi: MinstrelHt provides strange results at low SNR with A-MPDU enabled
- Bug 2926 - wifi: SSRC and SLRC mechanism not fully aligned to the standard
- Bug 2927 - lte:  Valgrind error in lena-x2-handover example
- Bug 2931 - traffic-control: QueueDisc drops the CE marked packets
- Bug 2936 - Disable -Wparentheses with GTK+ and gcc-8
- Bug 2938 - build: add --disable-werror flag to Waf
- Bug 2940 - traffic-control: QueueDisc SojournTime should be TracedCallback
- Bug 2941 - wifi: Order bit of Frame control field of WifiMacHeader not correctly set for some frames
- Bug 2948 - network: SetPriority does not support value 7
- Bug 2949 - test: Exit test.py if incorrect fullness option used
- Bug 2964 - wifi: IdealWifiManager keeps using the same MCS when all transmissions fail
- Bug 2974 - lte: Inconsistent declaration of IPV6 methods

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.28
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.28.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
compiler requirements (other compilers supporting C++11 may also work):
- g++-4.9 or later
- Apple LLVM version 7.0.2 or later
- clang-3.3 or later
In addition, a Python 2.7 (Python 2 series) or Python 3.4 (Python 3 series)
or later is required.

This release has been tested on the following platforms:
- Ubuntu 17.10 (64 bit) with g++-7.2.0 and Python 2.7.14
- Ubuntu 16.04 (64 bit) with g++-5.4.0 and Python 2.7.12
- Fedora Core 27 (64 bit) with g++-7.3.1 and Python 2.7.14
- Fedora Core 26 (64 bit) with g++-7.3.1 and g++-7.1.1, and Python 2.7.14
- CentOS/RHEL 7.4 (64-bit) with g++-6.3.1 and Python 3.5.1
  - Note: the default g++ version (4.8) must be upgraded
- CentOS/RHEL 6.8 (64-bit) with g++-4.9.2 and Python 2.7.13
  - Note: the default g++ version (4.4) must be upgraded
  - Note: the default Python version (2.6) must be upgraded
- OS X Sierra 10.12.6 with Xcode 9.2, Apple LLVM version 9.0.0, Python 2.7.13
- OS X Yosemite 10.10.5 with Xcode 7.2.1, Apple LLVM vers. 7.0.2, Python 2.7.10

### New user-visible features

- (tcp) Added TCP-LP as a congestion control module
- (tcp) When TCP SACK is enabled, the third rule defined in RFC 6675 regarding the next segment to transmit is followed
- (tcp) Separated the CongestionWindow trace into a CongestionWindow trace and a CongestionWindowInflated trace, where the latter tracks window inflation during loss recovery as defined by the Reno algorithm
- (tcp) Implemented the core functionality of TCP Pacing.
- (lr-wpan) Extended addressing mode is now supported.
- (internet) Ipv[4,6]AddressGenerator can now check if an address or a network is already allocated.
- (internet) Ipv6AddressHelper has more pedantic checks, but more than one such helper can now be used in a program.
- (lte) UEs can now use IPv6 to send and receive traffic.
- (uan) The UAN module now supports an IP stack
- (uan) Added some examples for running raw, IPv4, IPv6, and 6LoWPAN over UAN
- (traffic-control) Added a FIFO queue disc (FifoQueueDisc) and the Token Bucket Filter (TbfQueueDisc).

### Bugs fixed

- Bug 1745 - There can be only one Ipv6AddressHelper in a script
- Bug 1783 - tcp: Experiencing drops during fast recovery causes TCP's congestion window to blow up. Not anymore.
- Bug 2107 - lte: Enable PCAP for S1 and X2 point-to-point links
- Bug 2152 - lte: Fix uplink HARQ retx. out of synch at the MAC layer
- Bug 2151 - lte: Generate correct redundancy version in uplink HARQ
- Bug 2277 - lte: EpcTftClassifier::Classify blindly assumes that a packet has a L4 header
- Bug 2505 - network:  Avoid asserts in Header/Trailer deserialization
- Bug 2653 - tcp: Avoid saving smaller TS in case of packet reordering
- Bug 2656 - wifi: Minstrel and MinstrelHt provide different results for 802.11a/b/g
- Bug 2754 - lte: Incorrect arguments passed to Asn1Header::SerializeInteger
- Bug 2764 - wifi: WifiSpectrumModelId doesn't distinguish 11ax from legacy
- Bug 2766 - core: Modify logging for int64x64 to avoid stack overflow
- Bug 2768 - lte: LteUeNetDevice has a null MAC address
- Bug 2774 - wifi: Wifi Transmitter Sends ADDBA Request Continuously Even After Transmitting Delba Frame
- Bug 2791 - wifi: Interference Helper gives negative trace value for Interference Power
- Bug 2807 - energy: GetTotalEnergyConsumption is not updated correctly
- Bug 2809 - wifi: Wifi doesn't fully stop when energy is depleted
- Bug 2813 - wifi: OFDM 10 MHz and 5 MHz PHYs do not have correct subcarrier frequency spacings
- Bug 2820 - wifi: segmentation fault when Rrpaa wifi manager is used
- Bug 2821 - lte: Set correct limit for UL retx. in Round Robin scheduler
- Bug 2824 - ICMP opcode fr fragment timeout drop is wrong
- Bug 2826 - wifi: Management frames should be sent using the lowest basic rate
- Bug 2827 - wifi: Active scanning not working with 802.11n/ac/ax
- Bug 2828 - OLSR simple P2P example produces wrong results
- Bug 2831 - wifi: runtime channel width switch has no effect
- Bug 2832 - tcp: Partial ACK triggering repeated retransmission
- Bug 2836 - wifi: Missing VHT information in radiotap header when A-MPDU is used
- Bug 2838 - wifi: ht-wifi-network crashes with RTS/CTS enabled and frequency set to 2.4GHz
- Bug 2840 - lte: Wrong configuration of eNBs and UEs
- Bug 2843 - spectrum, wifi: Incorrect channel width and center frequency provided for non-HT PPDUs when building SpectralDensity
- Bug 2848 - wifi: Association ID not correctly set upon association
- Bug 2849 - lte: Received RLC and PDCP PDUs are missing in the stats files
- Bug 2854 - wifi: Fix 802.11ax channel width capabilities
- Bug 2856 - wifi: Correct bugs and improve logging in MinstrelHt
- Bug 2859 - tcp: calculation of BytesInFlight during partial recovery is fixed
- Bug 2866 - tcp: Removed VS warnings on TcpSocketBase
- Bug 2872 - csma: Packet copy in CsmaChannel breaks netanim tracing
- Bug 2873 - tcp: Potential SIGFPE in TcpYeah
- Bug 2874 - tcp: Corrected interpretation of RFC 6675 for IsLost check
- Bug 2875 - tcp: Use Rule3 of RFC6675 only with SACK enabled
- Bug 2876 - tcp: Do not shrink adv window after we send the FIN
- Bug 2877 - wifi: Wrong data types for CWmin and CWmax
- Bug 2885 - lte: Tests fail for CQA and PSS testsuites
- Bug 2890 - internet: Icmpv4 "ECHO" is defined in system-wide headers
- Bug 2898 - lte: Coding issue in function RrcAsn1Header::DeserializeRachConfigCommon
- Bug 2899 - lte: Inconsistent type for variable cellForWhichToReportCGI
- wifi:  Fix wrong guard band for SpectrumWifiPhy DSSS/HR-DSSS

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.27
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.27.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements (Note:  not all features available on all platforms):
- g++-4.9 or later (Note:  this is an upgraded requirement from ns-3.26)
- Apple LLVM version 7.0.2 or later
- clang-3.3 or later
- Python 2.7 (Python 2 series) or Python 3.4 (Python 3 series) or later

This release has been tested on the following platforms:
- Ubuntu 17.04 (64 bit) with g++-6.3.0
- Ubuntu 16.04 (64 bit) with g++-5.4.0
- Ubuntu 14.04.5 (64 bit) with g++-4.9.4
  - Note: the default g++ version (4.8) must be upgraded
- Fedora Core 26 (64 bit) with g++-7.1.1
- Fedora Core 22 (64 bit) with g++-5.3.1
- CentOS/RHEL 7.4 (64-bit) with g++-6.3.1
  - Note: the default g++ version (4.8) must be upgraded
- CentOS/RHEL 6.8 (64-bit) with g++-4.9.2
  - Note: the default g++ version (4.4) must be upgraded
  - Note: the default Python version (2.6) must be upgraded
- OS X Sierra 10.12.6 with Xcode 9.0 and Apple LLVM version 9.0.0
- OS X El Capitan 10.11.6 with Xcode 8.2.1 and Apple LLVM version 8.0.0
- OS X Yosemite 10.10.5 with Xcode 7.2.1 and Apple LLVM version 7.0.2
- Windows 10 Linux Subsystem with g++-5.4.0

### New user-visible features

- (build) Modules can also be located in contrib/ directory (as well as src/)
- (core) Logging timestamp precision has been improved
- (lte) LTE carrier aggregation feature according to 3GPP Release 10 is now supported.
- (traffic-control) CsmaNetDevice, SimpleNetDevice and WifiNetDevice support flow control.
- (traffic-control) Added support for queue discs to mark packets, and ECN support has been added to RED
- (traffic-control) A multi-queue aware queue disc modelled after Linux mq
- (traffic-control) Added support for detailed statistics output of queue discs
- (traffic-control) Added support for tracing queue sojourn times
- (tcp) The SACK option and the RFC 6675 loss recovery algorithm are now supported.
- (tcp) A LEDBAT model has been added.
- (wifi) 802.11ax High Efficiency (HE) physical layer modes are now supported.
- (wifi) A new RRPAA rate control mechanism has been added.
- (wifi) A new trace source for TXOP duration is now supported
- (wifi) Frame capture effect can now be modelled
- (uan) World Ocean System Simulation (WOSS) integration, and new PER models

### Bugs fixed

- Bug 1034 - wifi: No trace source for packet dropping from WifiMacQueue
- Bug 2007 - uan: Remove deprecation on SetRxThresholdDb
- Bug 2048 - lte: Fix scheduler resource allocation when CQI==0 (out of rante)
- Bug 2133 - tcp: Avoid assert when receiving data in FIN_WAIT_1 or _2 states
- Bug 2214 - tcp: Use of ScheduleNow only in selected part of the code
- Bug 2221 - network: Remove constraint on size of ns3::Packet Tag objects
- Bug 2256 - tcp: bytes in flight now updated in TcpTxBuffer
- Bug 2263 - tcp: Support processing of multiple TCP options
- Bug 2285 - tcp: Loss of ack of SYN can cause improper connection setup
- Bug 2367 - wifi: BlockAckManager does not remove iterators to freed items
- Bug 2450 - propagation: LogDistancePropagationLossModel is not continuous
- Bug 2456 - lte: MemberCschedSapProvider and Member CschedSapProvider templates
- Bug 2464 - spectrum: Store spectrum conversion matrix in CSR format
- Bug 2467 - spectrum: Do not schedule StartRx for receivers on other models
- Bug 2471 - wifi: unable to disable Block Ack agreement for 802.11n
- Bug 2477 - wifi: DCF manager assert
- Bug 2485 - traffic-control: Check queue full before checking RED thresholds
- Bug 2491 - wifi: replace DcfManager::MY_DEBUG
- Bug 2492 - uan: Make use of RxGain attribute in UanPhyGen class
- Bug 2511 - wifi: HT Greenfield is not working
- Bug 2512 - traffic-control: Byte mode not configured correctly for RED test
- Bug 2513 - core: ParetoRandomVariable needs a "scale", not a "mean" attribute
- Bug 2518 - build: Suppress printing of list of modules for Python programs
- Bug 2519 - lte: m_windowOffsetsMap.end () dereference
- Bug 2520 - examples: TCP variant not configured in wifi-tcp.cc
- Bug 2521 - internet: Include ipv6-option.h in wscript
- Bug 2523 - lte: Generate the RLC PDU properly
- Bug 2527 - internet: PrintRoutingTable adds an optional Time::Units parameter
- Bug 2528 - wifi: 802.11n RIFS cannot be enabled
- Bug 2529 - wifi: Missing trace for Block ACK timeout or missing MPDUs
- Bug 2530 - aodv: Rename aodv::SetBalckListTimeout to aodv::SetBlackListTimeout
- Bug 2531 - lte: Set HARQ feedback default value to avoid false retrasmissions
- Bug 2532 - wifi: Inconsistencies between 802.11n MCS and NSS value in TXVECTOR
- Bug 2533 - wifi: Provide better 802.11n/ac PHY abstraction for SIMO/MISO/MIMO
- Bug 2534 - traffic-control: ARED and PIE examples should return 1 upon exit
- Bug 2535 - utils: memory leak in bench-simulator.cc
- Bug 2536 - wimax: fixed dead assignment and potential memory leak in wimax
- Bug 2537 - traffic-control: Fix dead assignment on CoDel::DoDequeue
- Bug 2538 - tap-bridge: fixed dead assignment on tap-bridge
- Bug 2540 - mesh: fixed dead assignment on mesh/ie-dot11s-perr
- Bug 2541 - wifi: preamble not assigned correctly
- Bug 2542 - wifi: dead assignment on wifi mac-low
- Bug 2543 - lte: Fix when EstimateUlSinr returns NO_SINR
- Bug 2545 - flow-monitor: Optimized build fails for flow-monitor
- Bug 2547 - tcp: dead assignments on various tcp congestion controls
- Bug 2551 - wifi: wifi preamble should be part of TXVECTOR
- Bug 2552 - wifi: Remove code duplication between Yans and Spectrum
- Bug 2554 - wave: Fix static analysis warning
- Bug 2556 - csma, point-to-point: Avoid modification of transmitted packets
- Bug 2557 - point-to-point:  Fix dead assignments in queue-discs-benchmark
- Bug 2559 - tcp: TCP advertised window still incorrect
- Bug 2561 - tcp: TCP can not use Connect after a BindToNetDevice
- Bug 2563 - bindings: pybindgen version check to >= minimum version required
- Bug 2564 - wifi: Simulation crashes when CtsTimeout triggered for A-MPDU
- Bug 2565 - tcp: Do not wait for 2*MSL to notify socket close
- Bug 2566 - wifi: BlockAckManager::GetNRetryNeededPackets missing some packets
- Bug 2568 - examples: Avoid enumerating TCP variants
- Bug 2577 - wifi: crashes when A-MPDU and multiple TOS used, upon CTS timeout
- Bug 2578 - wifi: Unexpected assert "Internal collision but no packet in queue"
- Bug 2584 - wifi: MacLow triggers StartNext even if there is no TXOP
- Bug 2587 - tcp: Avoid overflow in htcp.cc
- Bug 2590 - traffic-control: Minor enhancements in red-queue-disc{.h, .cc}
- Bug 2591 - wifi: 802.11e Block Ack cannot be enabled on HT/VHT stations
- Bug 2594 - wifi: vht-wifi-network very low throughtput at MCS 6, 160 MHz, SGI
- Bug 2596 - network: EthernetTrailer::GetFcs() should be const
- Bug 2601 - wifi: HT stations should use 40 MHz width if configured  80 or 160z
- Bug 2604 - wifi: QosData frames separation with Block Ack enabled
- Bug 2605 - wifi: HT/VHT station sending to legacy results in null throughput
- Bug 2606 - wifi: Ideal rate manager for HT/VHT station to a legacy AP
- Bug 2607 - wifi: Correct overflow in channel width loop in Minstrel HT
- Bug 2613 - tcp: MaxRxSequence () is sometimes too large
- Bug 2614 - internet: RIP header version should be set to 2
- Bug 2615 - wifi: When the TXOP limit is nonzero, fragment in some cases
- Bug 2617 - traffic-control: PfifoFastQueueDisc::Peek all bands
- Bug 2621 - wifi:  Fix mismatch between log message and action
- Bug 2624 - lte: Inconsistent coding of type and naming
- Bug 2627 - internet: Ipv6RawSocket does not honor the bound interface
- Bug 2628 - wifi: Simulation crashes because of an out of range TID
- Bug 2629 - wifi: Assert failure in MinstrelHtWifiManager::GetLowestIndex
- Bug 2632 - lte: Build error due to uninitialized variable in rem helper
- Bug 2646 - lte: lena-simple-eps-emu example will never work
- Bug 2647 - wifi: ideal-wifi-manager-example crashes when NSS > 1
- Bug 2649 - tcp: Disabling TCP SACK doesn't fall back to NewReno
- Bug 2654 - wifi: Rraa wifi manager cannot be used with 802.11b
- Bug 2655 - wifi: A-MPDU simulation using TCP sometimes fails
- Bug 2658 - core: Avoid unexpected TracedValue type conversions
- Bug 2660 - traffic-control: TrafficControlHelper::Default configuration
- Bug 2661 - wave: Fix outdated class references in documentation
- Bug 2665 - internet: Ipv4RawSocket can not send to broadcast or multicast
- Bug 2666 - lte: Remove deprecated variables/config paths
- Bug 2668 - various: Make template classes use NS_LOG_* macros
- Bug 2671 - internet: ArpCache::Entry::SetMacAddress is misspelled
- Bug 2673 - wifi: run-time channel switch does not update WiFi spectrum model
- Bug 2675 - traffic-control: Use correct string value in PIE example
- Bug 2716 - tcp: TCP socket ignoring FIN flag when in FIN_WAIT state
- Bug 2717 - internet: Fix mask for Ipv4RoutingTableEntry::CreateDefaultRoute
- Bug 2722 - wifi: 802.11g sends DSSS spectrum signals inappropriately
- Bug 2731 - lte: Allow retranmission in RLC AM when PR expires, vtS overflows
- Bug 2732 - applications: BulkSend Tx trace is fired too much
- Bug 2733 - wifi: Ideal wifi manager cannot handle NSS higher than 1
- Bug 2741 - internet: IPv4 fragmentation fails on last fragment fragmentation
- Bug 2744 - wifi: 802.11n/ac with RTS/CTS crashes for a large number of nodes
- Bug 2751 - traffic-control: QueueDisc::Enqueue() order of operations
- Bug 2756 - internet: Ipv4AddressGenerator underestimates available addresses
- Bug 2757 - wifi: 802.11n/ac/ax maximum TXOP is not properly enforced
- Bug 2758 - internet: IPv4 sockets bound to unicast receiving subnet broadcast
- Bug 2759 - internet: Packet conversion from broadcast to subnet-directed
- Bug 2760 - internet: OLSR uses unicast-bound sockets to receive broadcasts
- Bug 2761 - wifi: Packet has no Traffic ID for CTS frames when A-MPDU is used
- Bug 2762 - internet: BindToNetDevice behaviour is not coherent with Linux
- Bug 2763 - wifi: management packets in PCAPs are marked as wrong by Wireshark
- Bug 2769 - tcp: Set ssthresh correctly upon RTO
- Bug 2771 - wifi: IdealWifiManager problems under different number of antennas
- Bug 2776 - wifi: Fix segmentation fault in MacLow
- Bug 2777 - lte: Save bearer info in m_drbMap, needed for handover request
- Bug 2781 - tcp: Do not go beyond limit in TcpTxBuffer
- Bug 2783 - wifi: MonitorSnifferRx SNR is invariant to RxNoiseFigure
- Bug 2786 - lte:  Remove unimplemented declaration
- Bug 2796 - tcp: NSC TCP transmission flow control not working

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.26
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.26.tar.bz2

### Supported platforms

This release is intended to work on systems with the following minimal
requirements:
- g++-4.8 or later
- clang-3.3 or later
- Python 2.7 (Python 2 series)
- Python 3.4 or later (Python 3 series)

This release has been tested on the following platforms:
- Ubuntu 16.04 (64 bit) with g++-5.4.0
- Ubuntu 14.04 (64 bit) with g++-4.8.2
- Fedora Core 24 (64 bit) with g++-6.1.1
- Fedora Core 22 (32 bit) with g++-5.3.1
- Fedora Core 21 (64 bit) with g++-4.9.2
- CentOS/RHEL 6.7 (64-bit) with g++-4.9.2
- OS X El Capitan 10.11.4 with Xcode 7.3.1 and Apple LLVM version 7.3.0
- OS X El Capitan 10.11.3 with Xcode 7.2.1 and Apple LLVM version 7.0.2

### New user-visible features

- (aodv) The node search radius is increased progressively (as per standard).
- (build system) New --check-config option to waf to reprint the summary
  of optional features which are configured.
- (build system) C++11 features are now supported in the codebase
- (core) DES Metrics support; see the API docs for class DesMetrics.
- (internet) Added TCP Vegas, Scalable, Veno, Illinois, Bic, YeAH, and
  H-TCP congestion control algorithms
- (internet) Add a TOS field to the InetSocketAddress class
- (network) SocketAddressTag has been removed from the codebase.
  Users can use RecvFrom (for UDP) or GetPeerName (for TCP) instead.
- (traffic control) Added the FQ-CoDel and PIE queue disc models
- (traffic control) Introduce Byte Queue Limits (BQL)
- (wifi) A new SpectrumWifiPhy physical layer model, making use of the
  ns-3 spectrum framework, has been introduced.  The current version of
  this model matches the API and behavior of YansWifiPhy closely, but
  over time is expected to support a different PHY abstraction and error
  models.
- (wifi) IEEE 802.11e TXOP limits are now enforced for different access
  categories when a QoS MAC is configured.

### Bugs fixed

- Bug 231 - Avoid adding SocketAddressTag (at all)
- Bug 1579 - Final fix for segfault caused by EDCA fragmentation
- Bug 1797 - Rate adaptation algorithms don't work with new 802.11n/ac standards
- Bug 1911 - AODV doesn't work with more than one NetDevice
- Bug 1939 - Aggregating the same object to two nodes produce unexpected results
- Bug 1977 - v4Ping verbose output when not explicitly stopped
- Bug 2057 - ARP and Ndisc caches should be updated by receiving valid L3 packets
- Bug 2069 - Avoid integer overflow in ByteTagList
- Bug 2102 - Ipv4GlobalRouting confused by bridged interfaces
- Bug 2123 - AODV doesn't use / honor IP TTL in the RREQ / RREP
- Bug 2149 - support deprecating attributes and trace sources
- Bug 2222 - incorrect EDCA behavior in case of internal collision
- Bug 2270 - Support -std=c++11 by default
- Bug 2306 - Raise DCF IsBusy() condition for CCA busy
- Bug 2307 - Get rid of m_receivedAtLeastOneMpdu flag
- Bug 2308 - PacketTag instead of ByteTag in LTE PDCP/RLC
- Bug 2344 - Attribute names can't have spaces
- Bug 2346 - sixlowpan CompressionThreshold limit can be violated
- Bug 2347 - LrWpan Ascii traces are hooked to the wrong traces.
- Bug 2351 - Fix symbol rate calculation for channels with 10 and 5 MHz width
- Bug 2352 - Add 'drop oldest' drop policy to WifiMacQueue
- Bug 2369 - Generate backoff if DCF requests access during AIFS
- Bug 2375 - Flowmonitor parse crashes when no pkt is received
- Bug 2376 - New A-MPDU might not be received if MPDUs of a previous A-MPDU are lost
- Bug 2377 - SocketIpTosTag and SocketIpv6TclassTag may be added twice in UDP
- Bug 2378 - A-MPDU variables are not correctly reset in YansWifiPhy when a PLCP preamble/header failed
- Bug 2379 - BlockAckRequest should not be part of single-TID A-MPDUs
- Bug 2380 - EdcaTxopN fails to retrieve TID for ADDBA requests, ADDBA responses and DELBA frames
- Bug 2383 - IPv4 header for reassembled packets reports a wrong payload size
- Bug 2386 - Move UanPhyGen out of Tx mode when energy depleted
- Bug 2387 - Fix waf --check-profile command
- Bug 2390 - WaypointMobilityModel::AddWaypoint lazy notify schedules an event using absolute time (should be relative time)
- Bug 2392 - SYN segment advertised window should not be scaled
- Bug 2395 - Upgrade pybindgen for Python >= 3.5
- Bug 2396 - move utility functions to WifiPhy
- Bug 2397 - add backoff and cw tracing to EDCA
- Bug 2398 - add SNR tag to beacons and probe responses
- Bug 2401 - Deserialize correctly the len of options in TcpHeader
- Bug 2402 - IPv4 Interface forwarding state is not honored
- Bug 2406 - Poor 802.11g performance in ad-hoc mode
- Bug 2408 - Simulation fails when 802.11n/ac is running with HT Minstrel and pcap enabled
- Bug 2412 - align WifiPhy frequency and channel number
- Bug 2414 - UdpSocket doesn't call NotifyConnectionFailed
- Bug 2419 - BsmApplication should use RecvFrom and not SocketAddressTag
- Bug 2420 - Remove code duplication between Wifi and Wave modules
- Bug 2421 - Forbid negative RemainingAmpduDuration for the last A-MPDU
- Bug 2425 - UdpSocketImpl simplification
- Bug 2427 - 802.11ac selects wrong slot duration
- Bug 2433 - Bic does not copy private parameters
- Bug 2436 - Do not send block ack if received A-MPDU not addressed to station
- Bug 2438 - Routing protocols should stop processing packets coming from a non-forwarding interface
- Bug 2439 - SixLowPan Compression kind need to be casted to int in the Print function
- Bug 2440 - SocketIpTosTag might be added twice if a packet is sent multiple times
- Bug 2442 - ConfigStore crash if object container has null item
- Bug 2443 - Increase application counters to accommodate Gbps transfer rates
- Bug 2444 - PcapHelper DataLinkType enum should have a name
- Bug 2445 - Out of bounds array access in Minstrel HT rate control
- Bug 2446 - Fix comma delimiter instead of pipe delimiter in Attributes
- Bug 2448 - Run examples in traffic control test suite
- Bug 2452 - Add Object::IsInitialized method
- Bug 2454 - DsrRouting::NotifyDataReceipt is also triggered for wifi management packets
- Bug 2461 - TCP BIC should copy private members
- Bug 2468 - Simulation with A-MPDU enabled hangs when fragmentation threshold is smaller than MSDU size
- Bug 2469 - send Block Ack Request upon short/long retry failures
- Bug 2474 - UdpEchoClient does not call Connect with addresses of type Inet[6]SocketAddress
- Bug 2477 - Partial fix for wifi asserts when transmit cancels a reception
- Bug 2479 - Flow monitor does not a have a "DROP_QUEUE_DISC" drop reason
- Bug 2480 - Use SQLite prepared statements
- Bug 2481 - Missing or incorrect group names
- Bug 2484 - Corrected the exit from CA_LOSS state in TCP
- Bug 2486 - NextTxSequence was not traced back from TCB
- Bug 2488 - Error in UanPdp::SumTapsFromMaxNc
- Bug 2496 - BSS membership selector in SupportedRates
- Bug 2500 - Ipv[4,6]RawSocket ignores IpTtl tag
- Bug 2507 - ConfigStore RawTextConfigLoad doesn't reset stream error state
- Bug 2508 - Duplicate of bug 2507
- Bug 2514 - The maximum transmission time for VHT A-MPDUs is not correct
- No BugId - Arp logging: label with request or reply properly

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.25
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.25.tar.bz2

### Supported platforms

This release has been tested on the following platforms:
- Ubuntu 15.10 (64 bit) with g++-5.2.1
- Ubuntu 14.04 (64 bit) with g++-4.8.2
- Fedora Core 23 (64 bit) with g++-5.3.1
- Fedora Core 22 (32 bit) with g++-5.1.1
- Fedora Core 21 (64 bit) with g++-4.9.2
- Arch Linux (64 bit) with g++-5.3.0 and clang++-3.7.1
- CentOS/RHEL 6.7 (64-bit) with g++-4.4.7
- OS X El Capitan 10.11.4 with Xcode 7.3 and Apple LLVM version 7.3.0
- OS X El Capitan 10.11.3 with Xcode 7.2.1 and Apple LLVM version 7.0.2
- OS X Yosemite 10.10.5 with Xcode 7.2.1 and Apple LLVM version 7.0.2
- FreeBSD 10.2-RELEASE (64 bit) with clang-3.4.1

### New user-visible features

- (internet) The native TCP implementation has been refactored, for the
  following improvements:
  - congestion control has been aligned with the Linux architecture,
    to facilitate easier coding of new congestion control algorithms
    or porting implementations from the Linux kernel,
  - Fast Retransmit and Fast Recovery algorithms, compliant to RFCs, are
    shared by all TCP variants, unlike previous releases.
  - TCP Hybla and TCP HighSpeed variants have been added.

- (routing) The PrintRoutingTable* functions now print the following information
  each time they are called:
  - the node ID,
  - the global time,
  - the local time (allowing for future introduction of per-node clocks),
  - the routing protocol type

- (traffic control) A traffic control sublayer has been added with the aim of
  introducing an equivalent of the Linux Traffic +Control infrastructure into
  ns-3. This layer sits in between the NetDevices (L2) and the network
  protocol (e.g. IP). It is in charge of processing packets and performing
  actions on them: scheduling, dropping, marking, policing, etc.  Initial
  support for flow control has been added to the PointToPointNetDevice;
  backward-compatible support for other device types is provided.

- (traffic control) A three band priority queue modelled after Linux
  pfifo_fast queueing discipline has been added.

- (traffic control) Adaptive RED extensions have been added to the RED model

- (wifi) A number of changes related to improving support for 802.11n/ac
  models have been introduced.
  - Two rate controls compatible with High Throughput (HT) Wi-Fi modes
    (e.g. 802.11n/ac), Ideal and MinstrelHT, have been added
  - Support for multiple spatial streams (MIMO) using existing SISO
    error models
  - The WiFi helper API for the WifiMac sublayer, and the configuration
    of HT modes and parameters, has been simplified
  - Better support for backward compatibility with legacy modes

- (wifi) 802.11g was made backward compatible with 802.11b networks

- (internet) L4 protocols (e.g., TCP, UDP, ICMP, etc.) can be demultiplexed
  by IP according to the incoming interface. In other words, it is now
  possible to use specialized L4 protocols according to the interface.

- (internet) Ipv6Address::IsAllHostsMulticast() is now deprecated.
  FF02::3 is not in the official reserved IPv6 multicast address list.

- (internet) Ipv6Address::IsAll[Nodes,Routers]Multicast() now checks the
  address scope beyond the simple link-local. Nodes are checked for
  Interface-Local, Link-Local and Realm-Local, Routers for the above plus
  Site-Local.

- (internet) Ipv6 routing protocols must now *not* forward packets to upper
  layers unless for extremey specific cases. The Ipv6L3protocol handles
  almost all the packets directed to the host.

- (internet) Ipv6 can now reject packets directed to an address not configured
  on the interface they are received from (Strong End System Model, RFC 1222).

- (internet) UDP and Ipv6 RAW sockets can now join Ipv6 multicast groups.
  Incoming packets will be filtered according to if there's a socket listening
  to that group. Source filtering is left to the application.

- (network) ns-3 is now capable of serializing cooked (SLL) headers. This is
  used in DCE to allow the generation of pcap directly readable by wireshark.

- (internet) It is now possible to set custom values for RipNg Link Down
  (standard is 16).

- (internet) permanent (static) NDISC entries can be created for IPv6

- (internet) IPv4 RIPv2 protocol model is now available.

### Bugs fixed

- Bug 1132 - useless for loops in block-ack-test-suite.cc
- Bug 1205 - EDCA is incorrectly modelled as DCF
- Bug 1571 - TCP zero-window and flow control window updates by the receiver
- Bug 1631 - Acoustic modem Energy consumption calc is wrong
- Bug 1761 - Rounding with olsr::EmfToSeconds
- Bug 1783 - BytesInFlight value fixed
- Bug 1954 - Serialized size of wifi-net-device differ for TX and RX trace
- Bug 1999 - PointToPointRemoteChannel invokes PointToPointChannel constructor
- Bug 2003 - Missing DSSS short PLCP preamble
- Bug 2041 - TCP RTO needs unit tests
- Bug 2068 - Timestamp option conforms to RFC 7323
- Bug 2102 - Make global routing robust to bridged links
- Bug 2116 - refactoring aggregation API
- Bug 2120 - 802.11g networks are not compatible with 802.11b clients
- Bug 2141 - TCP DataSent callback now correctly notifies sent data, without missing bytes in particular conditions
- Bug 2150 - The TCP sender keeps retransmitting and does not terminate the connection after some retries.
- Bug 2159 - TCP advertises wrong receive window
- Bug 2176 - Building IPv4 address from char* doesn't look reliable
- Bug 2183 - LiIonEnergySourceHelper is not in the energy wscript
- Bug 2184 - Integer overflow in Wi-Fi MacLow; remove dependency between tx and rx reference number field in MPDU status
- Bug 2185 - WiFi MacLow may respond to errored frames that it should ignore
- Bug 2193 - Simulation fails when transmitting very small MPDU subframes
- Bug 2195 - Udp[*]Client can't send packets to broadcast address
- Bug 2201 - Simulation fails when active probing is used in 802.11n/ac with channel bonding enabled
- Bug 2206 - Split internet-apps from applications
- Bug 2207 - Print node ID and time when printing routing tables
- Bug 2208 - Interface index based L4 protocols
- Bug 2210 - set Order field to 0 for HT/VHT
- Bug 2211 - Ipv{4,6}EndPoint can cause memory corruption
- Bug 2213 - Inconsistencies may exist between the selected WifiMacHelper and the chosen 802.11 version
- Bug 2219 - SixLowPanNetDevice hangs trying to decode a IPv6 Fragment extension header
- Bug 2224 - scope of GetAmpduExist() in EdcaTxopN
- Bug 2233 - Implement RFC 1222 - Strong End System Model
- Bug 2234 - Ipv6L3Protocol should trash multicast packets not interesting for the node
- Bug 2238 - Ipv6 routing reorganization
- Bug 2242 - Mobility of both sender PHY and receiver PHY set to sender mobility in lr-wpan-phy-test.cc example.
- Bug 2243 - TCP Socket Fork() fails to copy some parameters, causing connections to close prematurely on retransmit
- Bug 2246 - Some DSR LogComponents and classes are not defined in a unique way.
- Bug 2247 - Disabled Fast retransmit after an RTO
- Bug 2249 - LTE perl speed tests possibly broken
- Bug 2252 - Nix needs StaticRouting to work
- Bug 2254 - Ipv[4,6]RawSocket can return the wrong number of bytes sent.
- Bug 2255 - Ipv6RawSocket does not call data sent callbacks.
- Bug 2257 - Ipv[4,6]InterfaceContainer::Add are not consistent
- Bug 2258 - TcpSocketBase updated options and RTT for out of range segments.
- Bug 2259 - GSL not successfully enabled for Wi-Fi DSSS error rate model
- Bug 2262 - Corrected initial slow start threshold value
- Bug 2266 - NixRouting info are not copied during a packet fragmentation.
- Bug 2267 - Wrong channel bandwidth value in pcap files
- Bug 2271 - Decision to enable RTS/CTS handshake for A-MPDUs should be based on the comparison between RtsCtsThreshold and the total A-MPDU size
- Bug 2272 - SixLowPan NetDevice can not send uncompressed packets larger than 802.15.4 MTU
- Bug 2273 - WeakEs model should be enforced in Static and Global routing
- Bug 2275 - AdhocWifiMac not compatible with 802.11n/ac
- Bug 2279 - Ipv[4,6]L3Protocol::GetInterfaceForDevice speedup.
- Bug 2283 - Add capability to use pcap trace files with nanosecond precision timestamps
- Bug 2286 - Support for SLL header (de)serialization.
- Bug 2287 - FlowMonitor is confused by IP over IP.
- Bug 2288 - Ipv4 broadcast and multicast packets are replicated on all the interfaces.
- Bug 2289 - RIP v2 implementation
- Bug 2290 - Cannot run click examples from test.py
- Bug 2292 - Uninitialized variables since commit 7c60a9f8f271
- Bug 2293 - Red Queue Estimator spins when trying to compute queue average size under long idle times.
- Bug 2297 - 802.11n/ac networks are not compatible with legacy clients
- Bug 2302 - Fixing RTT calculation inside TCP Socket
- Bug 2303 - WifiMacQueue::GetSize should cleanup queue beforehand
- Bug 2304 - DSCP Set/Get are using 8 bit values instead of 6
- Bug 2307 - m_receivedAtLeastOneMpdu is not always reset
- Bug 2311 - save Ht/Vht capabilities info on a per-station basis
- Bug 2313 - Assert failed when using aggregation and RTS/CTS
- Bug 2315 - Problem when BACK Request is part of an A-MPDU
- Bug 2316 - MacLow shall use a single TXVECTOR for all MPDUs belonging to a same A-MPDU
- Bug 2317 - Get the non-HT reference rate of a HT MCS
- Bug 2318 - MPDU Aggregation fails with TCP
- Bug 2319 - BlockAckTimeout value is too low for 802.11n operating at 2.4 GHz
- Bug 2320 - Add method to check whether a WifiTxVector contains a valid combination of WifiMode, number of spatial streams and channel width
- Bug 2321 - Wifi rate managers should not be triggered for each MPDU when A-MPDU is used
- Bug 2327 - CWmin value selection for 802.11g is not compliant with the standard rules
- Bug 2328 - changes to antenna configuration should update HT MCS set
- Bug 2331 - Wrong mobility model in vanet-routing-compare example
- Bug 2332 - Assert failed when using aggregation and two pairs of AP-STA
- Bug 2337 - WifiPhy uses channel width in Hz iso MHz
- Bug 2338 - WifiMode selected for control frames may sometimes be invalid

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.24.1
--------------

Release 3.24.1 is a maintenance release that fixes three issues:
- support for Xcode 7 clang compiler version
- detection of Python bindings location
- compilation problem on 32-bit gcc-4.4 machine

The issues prevented selected systems from being able to successfully
build ns-3, and prevented the released version from being able to
detect the presence of pybindgen used for Python bindings.

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.24.1.tar.bz2

### Supported platforms

This release has been tested on the following platforms:
- Ubuntu 15.04 (64 bit) with g++-4.9.2
- Fedora Core 22 (32 bit) with g++-5.1.1
- Fedora Core 21 (64 bit) with g++-4.9.2
- Ubuntu 14.04 (64 bit) with g++-4.8.2
- Ubuntu 12.04.4 (64 bit) with g++-4.6.3
- CentOS/RHEL 6.7 (64-bit) with g++-4.4.7
- OS X Yosemite 10.10.5 with Xcode 7 and Apple LLVM version 7.0.0
- OS X Yosemite 10.10.5 with Xcode 6.4 and Apple LLVM version 6.1.0
- FreeBSD 10.2-RELEASE (64 bit) with clang-3.4.1

### New user-visible features

Features are identical to release 3.24.

### Bugs fixed

- Bug 2181 - Xcode 7.0 compiler warnings trigger build failure
- Bug 2182 - pybindgen required version is now a string literal
- Bug 2184 - integer overflow in MacLow

Release 3.24
------------

### Availability

This release is available from:
https://www.nsnam.org/release/ns-allinone-3.24.tar.bz2

### Supported platforms

This release has been tested on the following platforms:
- Ubuntu 15.04 (64 bit) with g++-4.9.2
- Fedora Core 22 (32 bit) with g++-5.1.1
- Fedora Core 21 (64 bit) with g++-4.9.2
- Ubuntu 14.04 (64 bit) with g++-4.8.2
- Ubuntu 12.04.4 (32/64 bit) with g++-4.6.3
- CentOS/RHEL 6.7 (64-bit) with g++-4.4.7
- OS X Yosemite 10.10.4 with Xcode 6.4 and Apple LLVM version 6.1.0
- FreeBSD 10.1-RELEASE (64 bit) with clang-3.4.1

### New user-visible features

- (build system) Python 3 is now supported by waf and test runner programs
- (wifi) 802.11ac Very High Throughput (VHT) physical layer modes are now
  supported.
- (internet) permanent (static) ARP entries can be created for IPv4
- (network) SimpleChannel allows per-NetDevice blacklists, in order to do hidden terminal testcases.
- (documentation) TracedCallback function signatures now documented in Doxygen

### Bugs fixed

- Bug 730 - Enabling fragmentation at run-time breaks simulation
- Bug 1249 - doxygen comments on device-level SetMobility ()
- Bug 1482 - mesh module was producing fatal error with ASCII and netanim traces
- Bug 1557 - errors in function OpenFlowSwitchNetDevice::BufferFromPacket()
- Bug 1681 - m_lastNavStart is assigned twice continuously
- Bug 1736 - default dot11EDCATableMSDULifetime
- Bug 1901 - mesh module producing malformed pcaps
- Bug 1929 - TcpL4Protocol::Send must indicate the source address to routing (if known)
- Bug 1982 - AODV and mesh modules use random variables before seeds can be set
- Bug 2058 - TCP window update can shrink left edge of window
- Bug 2063 - FdNetDevice::SendFrom assert does not account for headers correctly
- Bug 2082 - Empirical RV fails if value provided in CDF function is negative
- Bug 2093 - MultiModelSpectrumChannel::GetDevice only works for 0-th index
- Bug 2096 - Wrong pcap information when MPDU aggregation is used
- Bug 2104 - Sequence Number passed to QosUtilsMapSeqControlToUniqueInteger instead of Sequence Control
- Bug 2108 - Erroneous implementation in InterferenceHelper
- Bug 2119 - Fix memory leaks in FdNetDevice test due to non-executed events
- Bug 2124 - UdpSocketImpl::ShutdownRecv doesn't stop the Ipv[4,6]EndPointDemux
- Bug 2125 - subclasses must chain up to base class NotifyNewAggregate
- Bug 2126 - LrWpanNetDevice silently accepts no mobility on the node
- Bug 2127 - Scheduler benchmark utility works in first run only
- Bug 2130 - Allow SimpleChannel to simulate hidden terminals
- Bug 2132 - TcpL4Protocol sends a bad-formatted RST packet
- Bug 2135 - TCP doesn't honor the socket's output interface
- Bug 2136 - The usage of tid in wifi and wave module shall be "if (tid > 7)" rather than "if (tid >= 7)"
- Bug 2138 - SimpleNetDevice could send overlapped packets
- Bug 2143 - maybe-uninitialized warning
- Bug 2146 - bug in WAVE channel coordination
- Bug 2148 - Ipv6Interface::SetUp doesn't re-create the Link-Local addresses
- Bug 2153 - Incorrect power limits in wifi power control algorithms
- Bug 2154 - Incorrect power calculation in wifi power adaptation examples
- Bug 2156 - Duplicate packets when using two level aggregation
- Bug 2157 - OpenFlowSwitchNetDevice::BufferFromPacket setting eth_type incorrectly
- Bug 2158 - OpenflowSwitchNetDevice flooding bug
- Bug 2165 - server replies to SYN (with option TS) with a SYNACK (with option TS) also if the attribute Timestamp is false
- Bug 2166 - SteadyStateRandomWaypointMobilityModel directs the node to the ground
- Bug 2170 - AnimationInterface outputs improperly formed XML
- Bug 2171 - power-adaptation-distance.cc:202:22: error: call to 'pow' is ambiguous
- Bug 2173 - WifiInformationElement::DeserializeIfPresent attempts to read beyond the end of buffer
- Bug 2174 - m_rWnd not updated when segments without ACK are received
- Bug 2177 - Ipv6ExtensionFragmentHeader length is initialized to a wrong value.

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.23
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.23.tar.bz2

### Supported platforms

This release has been tested on the following platforms:
- Ubuntu 15.04 (64 bit) with g++-4.9.2
- Fedora Core 21 (64 bit) with g++-4.9.2
- Ubuntu 14.04 (64 bit) with g++-4.8.2
- Ubuntu 12.04.4 (32/64 bit) with g++-4.6.3
- CentOS/RHEL 6.6 (64-bit) with g++-4.4.7
- OS X Yosemite 10.10.3 with Xcode 6.3.1 and Apple LLVM version 6.1.0
- FreeBSD 10.1-RELEASE (64 bit) with clang-3.4.1

### New user-visible features

- (network) CalculateTxTime has been declared obsolete.
  CalculateBytesTxTime and CalculateBitsTxTime are to be used instead.
  The return value is a Time, instead of a double.
- (mobility) Classes added for converting geographic to cartesian coordinates.
- (spectrum) TvSpectrumTransmitter models added to create television
  transmitter(s) that transmit PSD spectrums customized by attributes such
  as modulation type, power, antenna type, channel frequency, etc.
- (wifi) Several model improvements have been made.  Access points now
  establish BSSBasicRateSet for control frame transmissions.  PLCP header
  and payload reception have been decoupled for improved PHY modeling
  accuracy.  RTS/CTS with A-MPDU is now fully supported, and 802.11n
  stations now support two-level aggregation.

### Bugs fixed

- Bug 1757 - RLC AM not using NACK_SN
- Bug 1974 - CalculateTxTime should return a Time, not a double
- Bug 1982 - AODV and mesh use random variables before seed can be set
- Bug 2004 - Wrong timeout calculation for 802.11n in WifiMac
- Bug 2008 - (lr-wpan) Crash if ending rx while status change is in progress
- Bug 2025 - (lr-wpan) Changing the channel doesn't affect the Tx params
- Bug 2034 - (lr-wpan) CSMA-CA BackoffPeriod is too short
- Bug 2054 - TCP window update can shrink the Left Edge of the window
- Bug 2066 - A-MPDU reception should check for successful preamble
- Bug 2067 - TCP performance when advertised window exceeds sender buffer size
- Bug 2070 - Wrong report of Packets and Bytes stored in CoDeL
- Bug 2073 - NDisc cache entries update timer might be stuck in a loop
- Bug 2075 - A-MPDU using RTS/CTS behaves wrongly when MaxSsrc is reached
- Bug 2076 - TCP MinRTO Attribute is not actually used
- Bug 2077 - Icmpv6L4Protocol::HandleDestinationUnreachable must check the packet size, not its serialized size
- Bug 2079 - mcs variable in ht-wifi-network example is confusing
- Bug 2082 - Empirical RV fails if the value provided in the CDF member function is negative
- Bug 2083 - BridgeNetDevice may send corrupted packets
- Bug 2087 - Waf fails to build ns-3 if the path contains accented characters
- Bug 2090 - (rip-ng) Routes may be added twice on interface activation
- Bug 2093 - MultiModelSpectrumChannel::GetDevice only works for 0-th index
- Bug 2095 - (wimax) Wrong values in default-traces.h for 16 QAM 3/4
- Bug 2097 - ACKs should be sent using legacy rates and preambles in 802.11n
- Bug 2103 - Ipv[4,6]RoutingHelper::PrintRoutingTableAll[At,Every] hangs if a node doesn't have IP

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.22
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.22.tar.bz2

### Supported platforms

This release has been tested on the following platforms:
- Fedora Core 21 (64 bit) with g++-4.9.2
- Ubuntu 14.10 (32 bit) with g++-4.9.1
- Ubuntu 14.04 (64 bit) with g++-4.8.2
- Ubuntu 12.04.4 (32/64 bit) with g++-4.6.3
- CentOS/RHEL 6.6 (64-bit) with g++-4.4.7
- OS X Yosemite 10.10 with Xcode 6.1.1 and clang-600.0.56
- FreeBSD 10.1-RELEASE (64 bit) with clang-3.4.1

### New user-visible features

- (wifi) Support for MPDU aggregation has been added to the wifi model,
  as well as a number of related example programs.
- (wifi) Added two combined power and rate control mechanisms to the
  wifi module.  The new mechanisms, PARF and APARF, are the first in the
  wifi module to jointly control transmission power and data rate.  Two use
  case examples (PowerAdaptationDistance and PowerAdaptationInterference)
  and a test case have also been added.
- (lte) In previous releases of the LTE module, the bearer release
  functionality was only partially supported. As an enhancement, a
  complete release bearer procedure is now implemented, which can be
  invoked by calling the new helper method
  LteHelper::DeActivateDedicatedEpsBearer(). The related output can be
  seen through the stats collected at different layers like PDCP, RLC,
  MAC, PHY. To support this implementation, an example and test suite is
  added within the LTE module examples and tests folder.
- (wave) Additional support has been added for WiFi-based vehicular networks,
  including the channel-access coordination features of IEEE 1609.4 and a
  comprehensive VANET routing example that includes a Basic Safety
  Message (BSM) packet generator application and associated statistics
  counters.  Together, these allow users to evaluate the performance effects
  in a VANET of varying transmission and channel properties (e.g., packet
  rate, message size, transmit power, propagation loss model, impact of
  routing protocol traffic, etc.).
- (internet) It is now possible to print the Neighbor Cache (ARP and NDISC)
  by using the RoutingProtocolHelper
- (stats) A new TimeProbe class has been added to hook the data collection
  framework to traced values emitting Time objects
- (documentation) the callback function signatures for all TraceSources
  is documented in Doxygen
- (utils) print-introspected-doxygen.cc has had several enhancements;
  use -h to read the usage message for details.
- (core) TracedValue and TracedCallback function signatures are now documented,
  which required changing the TypeId::AddTraceSource API.

### Bugs fixed

- Bug 1405 - RttEstimator improvements
- Bug 1551 - NS_LOG_COMPONENT_DEFINE inside or outside of ns3 namespace?
- Bug 1726 - WiFi Minstrel rate control algorithm doesn't save state
- Bug 1734 - TcpSocketBase produces spurious delayed ACKs
- Bug 1758 - Yans and Nist error rate models for 5/6 code rate 802.11n HT
- Bug 1770 - mesh test and example crash for 32-bit optimized builds
- Bug 1774 - compute signal power around channel, not across whole band, and fix LrWpanHelper to add a default PropagationDelayModel
- Bug 1791 - TCP Endpoint never deallocates when closing
- Bug 1801 - Setting Wi-Fi timing parameters through WifiMac attributes (documentation fix)
- Bug 1906 - 802.11n PHY configuration for 2.4GHz and 5GHz devices
- Bug 1957 - UdpSocketImpl is stuck after a Close()
- Bug 1968 - Missing supported 802.11n HT rates in the WifiPhy constructor
- Bug 1969 - Support short guard interval durations in 802.11n
- Bug 1970 - Missing NotifyAckTimeoutStartNow in MacLow::StartDataTxTimers
- Bug 1971 - 802.11n at 2.4 GHz should include a signal extension duration
- Bug 1972 - CommandLine duplicate argument handling: documentation updated.
- Bug 1983 - FlowMonitor returns containers copies instead of references.
- Bug 1986 - test result divergence for lte-frequency-reuse test
- Bug 1991 - PcapFileWrapper::CaptureSize attribute (snaplen) has no effect.
- Bug 1995 - avoid multiple definitions of PI
- Bug 1996 - RSRQ calculation: misleading variable names
- Bug 1997 - Fix PlotProbe() documentation and usage for GnuplotHelper and FileHelper
- Bug 2002 - Hardcoded include paths cause breakage
- Bug 2011 - Default Speed attribute in ConstantSpeedPropagationDelayModel
- Bug 2016 - Radvd do not consider the SendAdvert option and don't reply to RS
- Bug 2020 - Erroneous MCS field in 802.11n PCAP files
- Bug 2021 - Missing const qualifier in TopologyReader::Link::Attributes{Begin,End}()
- Bug 2026 - 802.11n Ness parameter badly set for data frames
- Bug 2027 - Calculation of HT training symbol duration does not accurately follow 802.11n standard
- Bug 2028 - remove unnecessary Time to double conversions in Wifi models
- Bug 2029 - new CQI generation approach fix
- Bug 2030 - provide default values for WifiTxVector
- Bug 2037 - HT capabilities may print bogus chars
- Bug 2038 - Stop method does not stop next wave in WaveformGenerator
- Bug 2042 - LTE a3-rsrp-handover-algorithm.cc:  error: overflow in implicit constant conversion
- Bug 2043 - print-introspected-doxygen crashes when some modules are disabled
- Bug 2044 - Buffer::Iterator::ReadNtohU16() and ReadNtohU32() not implemented correctly
- Bug 2045 - Missing NS_OBJECT_ENSURE_REGISTERED in TcpTxBuffer and TcpRxBuffer
- Bug 2046 - set Block Ack timeout when SetStandard is called
- Bug 2047 - Ipv6EndPointDemux::Lookup may crash
- Bug 2049 - CQI feedback should always use the same calculation method
- Bug 2053 - In tcp-socket-base.cc, NotifyDataSent incorrectly called with retransmits
- Bug 2055 - TCP TxBuffer and RxBuffer traces don't work
- Nix-vector routing implementation now uses a lazy flush mechanism,
  which dramatically speeds up the creation of large topologies.

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.21
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.21.tar.bz2

### Supported platforms

- Fedora Core 20 (32/64 bit) with g++-4.8.2
- Ubuntu 14.04 (32/64 bit) with g++-4.8.2
- Ubuntu 12.04.4 (64 bit) with g++-4.6.3
- Ubuntu 10.04.4 LTS (64 bit) with g++-4.4.3
- CentOS/RHEL 6.5 (64-bit) with g++-4.4.7
- OS X Mavericks 10.9 with Xcode 5.1.1 and clang-503.0.40
- FreeBSD 9.2-RELEASE (64 bit) with clang-3.3

### New user-visible features

- The LTE module now supports the transport of the S1-U, X2-U and X2-C
  interfaces over emulated links via the new helper class EmuEpcHelper.
- CommandLine can now provide a shorthand argument name for any
  Attribute.
- Implemented support for Frequency Reuse algorithms in LTE module, as the
  outcome of GSoC 2014 project.
  The project also includes several sub-features, such as:
  - implementation of Downlink Power Control
  - implementation of Uplink Power Control
  - new DL-CQI generation approach, which increases throughput if FR algorithms
    are used
  - seven options of Frequency Reuse algorithms: LteFrNoOpAlgorithm,
    LteFrHardAlgorithm, LteFrStrictAlgorithm, LteFrSoftAlgorithm,
    LteFfrSoftAlgorithm, LteFfrEnhancedAlgorithm, LteFfrDistributedAlgorithm
  - updated RadioEnvironmentMapHelper. Now RadioEnvironmentMap can be generated
    for Data or Control channel and for specified RbId, what is helpful when
    using FR algorithms
- Added a CoDel queue model.  CoDel queues measure and control the queue
  traversal delay.  The ns-3 implementation is a port of the Linux
  implementation.
- Added support for TCP timestamp and window scale options, and added
  ability to trace the TCP slow start threshold value.
- SimpleNetDevice and SimpleChannel (used for adding basic link effects
  for testing of higher-layer protocols) have been extended to support
  the option of broadcast or PointToPoint link semantics.  The bandwidth
  and link delay can be constrained, and it uses an output queue.
- SimpleNetDevice and SimpleChannel can be installed in a node through
  a new helper: SimpleNetDeviceHelper.
- Implemented new PacketSocketServer and PacketSocketClient applications.
  The primary use is in tests, to avoid using the ones from the
  application module that also bring in a dependency on the internet module.

### Bugs fixed

- Bug 1673 - Config::Set/Connect does not search for attributes in parent class
- Bug 1762 - UE stuck in IDLE_CONNECTING because RRC CONN REQ is not transmitted
- Bug 1811 - basic traffic generator for network module
- Bug 1824 - L4 protocol sockets should support BindToNetDevice over IPv6
- Bug 1831 - TcpSocket SlowStartThreshold is not a TraceSource
- Bug 1851 - WifiRadioEnergyModel energy consumption values are taken from a 802.15.4 chip
- Bug 1854 - std::out_of_range Problem
- Bug 1858 - wireless examples not correctly recording packet reception
- Bug 1860 - TCP needs the Window Scale option
- Bug 1893 - issue in DoSchedUlTriggerReq with harq
- Bug 1911 - AODV cannot work on nodes with more than one netdevice
- Bug 1921 - Icmpv6L4Protocol::ForgeEchoRequest returns a malformed packet
- Bug 1930 - Use of invalid reference in OLSR RemoveLinkTuple
- Bug 1932 - NdiscCache entry is not failsafe on double neighbor probing.
- Bug 1937 - FlowMonitor fails to track multiplexed packets
- Bug 1942 - refactoring of lte-sinr-chunk-processor -> lte-chunk-processor
- Bug 1943 - Waveform generator signal duration calc error
- Bug 1951 - AODV does not update nexthop for 1-hop nodes
- Bug 1955 - The IPv4 identification field should be unique per (source, destination, protocol) tuple
- Bug 1960 - Wrong information on index range, about Node::GetDevice
- Bug 1961 - planetlab-tap-creator "variable set but not used"
- Bug 1963 - AODV can tag the same packet twice (and raise an assert)
- Bug 1964 - Integer overflow on UniformRandomVariable::GetInteger()
- Bug 1967 - LL Multicast is not compressed in the right way in IPHC
- Bug 1981 - PyViz shell not compatible with ipython >= 0.11

### Known issues

- Bug 1770 - The mesh module will crash if used for g++ version >= 4.8.1
in optimized mode, on a 32-bit Linux machine.  Lowering the optimization
level to -O1 in this case can be used as a workaround.

Release 3.20
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.20.tar.bz2

### Supported platforms

- Fedora Core 20 (32/64 bit) with g++-4.8.2
- Ubuntu 14.04 (32/64 bit) with g++-4.8.2
- Ubuntu 12.04.4 (64 bit) with g++-4.6.3
- Ubuntu 10.04.4 LTS (64 bit) with g++-4.4.3
- CentOS/RHEL 6.5 (64-bit) with g++-4.4.7
- OS X Mavericks 10.9 with Xcode 5.1.1 and clang-503.0.40
- FreeBSD 9.2-RELEASE (64 bit) with clang-3.3

### New user-visible features

- A new LrWpan model, providing initial support for IEEE 802.15.4 networks
- A new IPv6 routing protocol has been added: RIPng. This protocol is
  an Interior Gateway Protocol and it is available in the Internet module.
- A new LTE MAC downlink scheduling algorithm named Channel and QoS Aware (CQA)
  Scheduler is provided by the new ``ns3::CqaFfMacScheduler`` object.
- The Internet FlowMonitor can now track IPv6 packets.
- FlowMonitor no longer tracks multicast/broadcast packets, reflecting
  its original design.
- FlowMonitor "SerializeToXml" functions are now directly available
  from the helper.
- The SixLowPan model can now use uncompressed IPv6 headers. An option to
  define the minimum compressed packet size has been added.
- Simplify output of Times in a specific unit; see Time::As ()
- Ipv6Extension::m_dropTrace has been removed. Ipv6L3Protocol::m_dropTrace
  is now fired when appropriate.
- IPv4 identification field value is now dependent on the protocol field.
- Fixes to support Python >= 3.3 in ns3 Python bindings
- Enable selection of high precision int64x64_t implementation
  at configure time, for debugging purposes.

### Bugs fixed

- Bug 1276 - optimize NistErrorRateModel
- Bug 1294 - New PeekU8 () and Read (Buffer::Iterator start, uint32_t size) methods in Buffer::Iterator
- Bug 1443 - MinDistance replaced by MinLoss in FriisPropagationLossModel, to
  better handle conditions outside of the assumed far field region.
- Bug 1653 - Extension of CommandLine interface: restored operator << (CommandLine)
- Bug 1717 - Detect unsettable attributes
- Bug 1730 - no model library documentation for spectrum module
- Bug 1739 - The endpoint is not deallocated for UDP sockets
- Bug 1786 - os << int64x64_t prints un-normalized fractional values
- Bug 1787 - Runtime error when using AnimationInterface::EnablePacketMetadata() to fetch metadata of CSMA packet
- Bug 1792 - Parameter logger constructor
- Bug 1808 - FlowMon relies on IPv4's Identification field to trace packets
- Bug 1817 - IPv4 Identification field should consider protocol as well.
- Bug 1818 - FlowMonitor needs IPv6 support
- Bug 1820 - models library doc: make should not rm -rf figures
- Bug 1821 - Setting an interface to Down state will cause various asserts in IPv6
- Bug 1829 - Multiple TCP socket entries
- Bug 1837 - AODV crashes when using multiple interfaces
- Bug 1838 - FlowMonitorHelper must not be copied.
- Bug 1841 - FlowMonitor fails to install if IPv4 is not installed in the node
- Bug 1842 - FlowMonitor SerializeToXml<Something> should be called by the helper
- Bug 1843 - IPv6 extensions dropped packets do not fire L3 drop trace
- Bug 1845 - FlowMonitor should discard any broadcast/multicast packet
- Bug 1846 - IPv6 should send Destination Unreachable if no route is available
- Bug 1850 - TCP NewReno loss behavior
- Bug 1852 - cairo-wideint-private.h error cannot find definitions for fixed-width integral types
- Bug 1853 - NS_LOG_FUNCTION broken on OSX 10.9
- Bug 1855 - SixLowPanNetDevice is not correctly indexed
- Bug 1857 - Detect location of installed boost libraries
- Bug 1862 - NS_LOG="Time=*|prefix_time" causes stack overflow
- Bug 1868 - Optimized builds are sensitive to -fstrict-overflow
- Bug 1870 - Remove unnecessary AsInt functions
- Bug 1872 - Inside RREQ processing, in case of IP duplication, packet dropped instead of being forwarded
- Bug 1873 - Energy source checked to be aggregated to the node
- Bug 1874 - Ipv4L3Protocol::ProcessFragment: addressCombination and idProto identifiers not properly computed
- Bug 1876 - enable OLSR HNA table access
- Bug 1877 - constructors missing for PropagationLossModels
- Bug 1882 - int64x64 tests trigger valgrind bug
- Bug 1883 - IPv6 don't consider the prefix and network when choosing output address
- Bug 1885 - WifiSpectrumValue5MhzFactory::CreateRfFilter does not align with the used 5Mhz SpectrumModel
- Bug 1887 - Point-to-point traces should contain PPP headers
- Bug 1888 - COST231 propagation loss model: corrections
- Bug 1889 - PointToPointNetDevice: In some cases MacTxDrop trace is not called
- Bug 1890 - UdpClientTrace: MPEG frame size is squeezed into (insufficient) 16 bit integer
- Bug 1891 - UdpSocketImpl::GetSockName doesn't return the IPv6 address
- Bug 1894 - CqaFfMacScheduler needs an update
- Bug 1895 - IP header Source Address changed while forwarding RREQ
- Bug 1900 - Avoid floating point differences across platforms in test outputs
- Bug 1903 - Namespace usage in olsr-state.cc/h
- Bug 1907 - Add IsSupportedMcs method in YansWifiPhy
- Bug 1912 - Avoid multiple Wifi MacTxMiddle instances
- Bug 1913 - Avoid crash in Wifi BlockAckManager::GetNextPacket()
- Bug 1915 - BRITE channel delay is rounded to an integer
- Bug 1916 - RandomWalk2dMobilityMode default "Bounds" attribute is not a rectangle
- Bug 1919 - Strip trailing semi-colons from mobility trace files
- Bug 1920 - Remove DSR attributes so file can be re-loaded by config-store
- Bug 1922 - WAVE GetSsid should not be fatal
- Bug 1923 - Setting Active Probing to false in Wifi Sta has no effect
- Bug 1924 - sensing radius and CCA

### Known issues

- Bug 1770 - The mesh module will crash if used for g++ version >= 4.8.1
in optimized mode, on a 32-bit Linux machine.  Lowering the optimization
level to -O1 in this case can be used as a workaround.

Release 3.19
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.19.tar.bz2

### Supported platforms

These platforms have been tested; others may work also:
- Fedora Core 20 (32 bit) with g++-4.8.2
- Fedora Core 19 (32/64 bit) with g++-4.8.1
- Ubuntu 13.10 (64 bit) with g++-4.8.1
- Ubuntu 12.04.3 (32/64 bit) with g++-4.6.3
- Ubuntu 10.04.4 LTS (64 bit) with g++-4.4.3
- OS X Mavericks 10.9 with Xcode 5.0.1 and clang-500.2.79
- OS X Mountain Lion 10.8.5 with Xcode 5 and g++-4.2.1
- FreeBSD 9.2-RELEASE (64 bit) with clang-3.3

### New user-visible features

- Extension to UE measurements and improved handover algorithm models in LTE
  module, as the outcome of GSoC 2013 project. The project also includes several
  sub-features, such as:
  - implementation of System Information Block Type 1 (SIB1);
  - a new option for automatic UE attachment using Idle mode cell selection
    procedure;
  - improved configurability of UE measurements; and
  - two options of handover algorithms for enabling automatic handover trigger
    in LTE simulation: A2-A4-RSRQ and strongest cell (A3-RSRP).

- A new FixedRoomPositionAllocator has been added to the buildings
  module. It allows one to generate a random position uniformly
  distributed in the volume of a chosen room inside a chosen building.

- A new attribute ns3::LteRlcAm::TxOpportunityForRetxAlwaysBigEnough
  allows to overcome the lack for re-segmentation in the RLC AM
  implementation by assuming that the size of a TxOpportunity is
  always big enough for the RLC AM PDU to be retransmitted.

- After some profiling, the code of LteMiErrorModel has been optimized
  for speed, resulting in a significantly lower execution time of the
  LTE model when used with the error model enabled.

- A new WiFi extension for vehicular simulation support is available in
  the src/wave directory.  The current code represents an interim capability
  to realize an IEEE 802.11p-compliant device, but without the WAVE
  extensions (which are planned for a later patch).  The WaveNetDevice
  modelled herein enforces that a WAVE-compliant physical layer (at 5.9 GHz)
  is selected, and does not require any association between devices (similar
  to an adhoc WiFi MAC), but is otherwise similar (at this time) to a
  WifiNetDevice.  WAVE capabililties of switching between control and
  service channels, or using multiple radios, are not yet modelled.

- A new IPv6 over Low power Wireless Personal Area Networks (6LoWPAN) model
  is available.  Using ns-3's naming convention, the acronym is expanded
  to SixLowPanNetDevice.   The SixLowPanNetDevice is able to act as a
  shim between IPv6 and a NetDevice, compressing IPv6 headers according
  to RFCs 4944 and 6262.  The SixLowPanNetDevice is meant to be used over
  IEEE 802.15.4 NetDevices, but it can be used on other NetDevices as
  well (see the manual for full details).  This model precedes the
  general availability of an 802.15.4 model, so must be run in conjunction
  with a wired NetDevice model for now, or with experimental versions of
  802.15.4 models.

- It is now possible to use Ipv6PacketInfoTag from UDP applications in the
  same way as with Ipv4PacketInfoTag. See Doxygen for current limitations in
  using Ipv[4,6]PacketInfoTag to set IP properties.

- Ipv[4,6]Interfaces not respecting the minimum MTU requirements (68 octects
  for IPv4 and 1280 octects for IPv6) will be automatically set as Down.

- IPv6 addresses and routing tables are printed in a more conventional way,
  closely matching the Linux "route -A inet6" command.

- Additional time units (Year, Day, Hour, Minute) were added to the time
  value class that represents simulation time; the largest unit prior to
  this addition was Second.

- A new parallel scheduling algorithm based on null messages, a common
  parallel DES scheduling algorithm, has been added.  The null message
  scheduler has better scaling properties when running on some scenarios
  with large numbers of nodes since it does not require a global
  communication.

### Bugs fixed

- Bug 1496 - Option to print log level in NS_LOG messages, and documentation.
- Bug 1592 - Parsing bug in FlowMonitor example script
- Bug 1756 - RLC AM Mode State Variable Bug
- Bug 1763 - Message 3 should be sent using the UL GRANT in the RAR
- Bug 1778 - Implement TapBridge::IsLinkUp() function
- Bug 1777 - Implement the more direct way of "using" configuration of existing tap interface
- Bug 1776 - Improve CRC performance for CsmaNetDevice in emulation modes
- Bug 1788 - unused private field warning
- Bug 1789 - missing test condition for sigma in buildings-shadowing-test
- Bug 1796 - Ipv6PacketInfoTag is not filled by UdpSocketImpl::ForwardUp6
- Bug 1798 - Changing the rate of onOffApplication might stop transmission
- Bug 1802 - FlowMon header deserialization problem with IPv4 fragments
- Bug 1803 - Lookup /NodeList/4/DeviceList/0/LteEnbRrc/UeMap/0 got no matches
- Bug 1807 - Multiple bugs in Ipv4L3Protocol::LocalDeliver
- Bug 1810 - IP packets can be sent on NetDevices not respecting the minimum MTU requirements
- Bug 1814 - IPv6 Packet with length not multiple of 8 bytes are fragmented incorrectly.
- Bug 1815 - Python bindings compilation with clang compiler toolchain
- Bug 1816 - IPv4 fragmentation loses Packet tags
- Bug 1877 - constructor missing for <something>PropagationLossModels

Release 3.18.2
--------------

ns-allinone-3.18.2 was released to include a bake configuration file update
for Direct Code Execution.  The ns-3 code in this release was unchanged
from that of ns-3.18.1.

Release 3.18.1
--------------

This release is mainly to provide updated compiler support (clang/LLVM)
and fix the Python API scanning facility.  A few additional bug fixes
and new features are described below.

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.18.1.tar.bz2

### Supported platforms

These platforms have been tested; others may work also:
- Fedora Core 19 (32/64 bit) with g++-4.8.1
- Ubuntu 13.10 (64 bit) with g++-4.8.1
- Ubuntu 12.04.3 (32/64 bit) with g++-4.6.3
- Ubuntu 10.04.4 LTS (64 bit) with g++-4.4.3
- OS X Mavericks 10.9 with Xcode 5.0.1 and clang-500.2.79
- OS X Mountain Lion 10.8.5 with Xcode 5 and g++-4.2.1
- FreeBSD 9.2-RELEASE (64 bit) with clang-3.3

### New user-visible features

- It is now possible to randomize the time of the first beacon from an
  access point.  Use an attribute "EnableBeaconJitter" to enable/disable
  this feature.
- A new FixedRoomPositionAllocator helper class is available; it
  allows one to generate a random position uniformly distributed in the
  volume of a chosen room inside a chosen building.
- Logging wildcards:  allow "***" as synonym for "*=**" to turn on all logging.
- The log component list ("NS_LOG=print-list") is now printed alphabetically.

### Bugs fixed

- Bug 1779 - NS_UNUSED_GLOBAL not working in attribute test class declaration
- Bug 1766 - Fixes to wifi-hidden-terminal.cc example
- Bug 1722 - Avoid transmitting beacons concurrently
- Bug 1691 - RTS/CTS NAV reset prematurely
- Bug 1622 - Avoid waf hanging during apiscan
- Bug 1616 - WifiPhyStateHelper reports false CCA_BUSY times at State trace source
- Bug 1552 - Storing log name inside LogComponent class (NS_LOG) as std::string
- Bug 1011 - assert failed. file=../src/devices/wifi/dcf-manager.cc
- bug 945 - remove deprecated IEEE 802.11p code from wifi module
- Fix aliasing bug in optimized static builds
- Fix memory leak due to circular reference in MPI module
- Make wifi tests more robust to random variable perturbations
- Fix Time class doxygen
- Fix compilation with Clang 3.2 and newer versions, including Apple Xcode 5
- Miscellaneous NetAnim fixes

Release 3.18
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.18.tar.bz2

### Supported platforms

These platforms have been tested; others may work also:
- Fedora Core 18 (32/64 bit) with g++-4.7.2
- Fedora Core 17 (32/64 bit) with g++-4.7.0
- Ubuntu 13.04 (32/64 bit) with g++-4.7.3
- Ubuntu 12.04 (32/64 bit) with g++-4.6.3
- Ubuntu 10.04.4 LTS (64 bit) with g++-4.4.3
- OS X Mountain Lion 10.8.3 with g++-4.2.1
- FreeBSD 9.1-RELEASE (64 bit) with g++-4.2.1

### New user-visible features

- Time attributes can now be bounded.  See attribute-test-suite.cc for an
  example.
- Data collection components have been added to the stats module.  These
  components can be used to generate time series data in files and plots.
- IPv6 address class 2001:db8::/32 is now dropped by routers (RFC 3849).
- New generic hash function interface.  Two hash functions are provided:
  murmur3 (default), and the venerable FNV1a.  See the Hash Functions
  section in the Manual.
- New Mac16Address has been added, Mac64Address is now in-line with
  Mac48Address and all the three can be used in IPv6 autoconfigure.
- Previously, the use of Building models was limited to the use of the
  companion BuildingsMobilityModel. Now, any MobilityModel can be
  used with Building models.
- The latest LTE module code by the LENA project has been merged,
  including the following new features:
  - PHY support for UE measurements (RSRP and RSRQ)
  - RRC support for UE measurements (configuration, execution, reporting)
  - Automatic Handover trigger based on RRC UE measurement reports
- IPv6 can now detect and use Path-MTU. See
  examples/ipv6/fragmentation-ipv6-two-MTU.cc for an example.
- Radvd application have a new Helper. See the updated
  examples/ipv6/radvd.cc for an example.
- 11n- It is now possible to create a high throughput (HT) node that used the new 11n data rates and preambles.
- It is now possible to request printing command line arguments to the
  desired output stream using PrintHelp or operator <<

### Bugs fixed

- Bug  760 - IP address removal can be painful
- Bug 1190 - Suppress hello if bcast was sent within the last hello interval
- Bug 1296 - Enhancement in Ipv[4,6]RoutingHelper
- Bug 1390 - ICMPv6 Redirect are handled correctly only for /64 networks
- Bug 1522 - Hidden node scenario leads to ARP failure
- Bug 1584 - Old Association Request Timeouts are not canceled
- Bug 1629 - Make AODV Default to Disable Hello
- Bug 1643 - NdiscCache creation and existence checks
- Bug 1646 - ICMPv6 Redirect are sent from global address instead of link-local
- Bug 1662 - m_type not set for Ipv6OptionRouterAlertHeader
- Bug 1678 - C++11 compliance problem with std::pair"
- Bug 1682 - ./waf crashes on FC10
- Bug 1683 - IPv6 autoconfigured don't use *infinite* lifetimes
- Bug 1669 - ns-3 should support binding two and three (possibly more) arguments
- Bug 1675 - Throughput computation error in Wireless examples
- Bug 1687 - wscript features report doesn't respect NOCOLOR
- Bug 1688 - Routers should advertise themselves from the link-local address
- Bug 1689 - IPv6 shouldn't add a default gateway without checking the Router lifetime
- Bug 1690 - missing header files from wifi wscript
- Bug 1697 - ICMPv6 Redirect trigger contains multiple bugs
- Bug 1698 - mobility.SetPositionAllocator misses prefix "ns3::"
- Bug 1700 - Ipv6RawSocket does not honor the bound address when sending packets
- Bug 1701 - Ipv6StaticRouting: the source address should match the destination scope
- Bug 1702 - Ipv6InterfaceContainer::SetRouter should not always add the router as the default router.
- Bug 1703 - Nodes don't react to a DAD
- Bug 1712 - The IP (v4 and v6) forwarding needs a test
- Bug 1718 - Ipv4StaticRouting log component is misspelled
- Bug 1720 - IPv6 Fragmentation cause crashes
- Bug 1721 - Path MTU isn't handled properly
- Bug 1723 - name clash in ipv4-header.h with <termios.h>
- Bug 1727 - Ping6 should use a proper source address
- Bug 1728 - Radvd application is missing an Helper
- Bug 1731 - lte-phy-error-model passes unexpectedly
- Bug 1738 - strict aliasing compiler bug
- Bug 1742 - IPv6 HbH and Dst Extension Header size is not correctly calculated
- Bug 1752 - RadvdInterface m_defaultLifeTime is set to milliseconds instead of seconds
- Bug 1753 - Halting Issue with DistributedSimulatorImpl
- Bug 1754 - Missing GIL lock in generated callback destructor

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.17
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.17.tar.bz2

### Supported platforms

These platforms have been tested; others may work also:
- Fedora Core 18 (32/64 bit) with g++-4.7.2
- Fedora Core 17 (32/64 bit) with g++-4.7.0
- Ubuntu 13.04 (32/64 bit) with g++-4.7.3
- Ubuntu 12.10 (32/64 bit) with g++-4.6.3
- Ubuntu 12.04 (32/64 bit) with g++-4.6.3
- Ubuntu 10.04.4 LTS (64 bit) with g++-4.4.3
- OS X Mountain Lion 10.8.3 with g++-4.2.1
- FreeBSD 9.1-RELEASE (64 bit) with g++-4.2.1

### New user-visible features

- new TCP Westwood and Westwood+ models
- new FdNetDevice model and associated helpers.  The FdNetDevice is able
  to read and write from a file descriptor.  Various helpers are provided
  to associate this descriptor with underlying devices or sockets on the
  host operating system, including a packet socket for emulation, and
  tap devices including a version specialized for use on PlanetLab.
- ns-3-click: it's now possible to (i) have Click pull random numbers from
  ns-3 and (ii) have ns-3 set "defines" in Click via the simulation file
  (see src/click/examples/nsclick-defines.cc).
- Waf shipped with ns-3 has been upgraded to version 1.7.10 and custom
  pkg-config generator has been replaced by Waf's builtin tool.
- create-module.py script has been updated to work with waf 1.7 and support
  for creating modules with names containing dashes has been added.
- the M5 release of the LTE module by the LENA project has been
  merged; please see src/lte/RELEASE_NOTES for more detailed info

### Bugs fixed

- bug 1256 - Unnecessary SND.NXT advance, missing ACK for Out of Order segments
- bug 1318 - Ipv6L3Protocol::LocalDeliver can get stuck in an infinite loop
- bug 1409 - Add an attribute "SystemId" to configure the ID for MPI
- bug 1421 - Frequency dependent propagation loss models need uniform Frequency / Lambda attribute
- bug 1434 - DSR throughput not comparable to other protocols for manet example
- bug 1502 - Shutdown on tcp socket seems to misbehave
- bug 1503 - BlockAckManager infine loop
- bug 1517 - Waf clean/distclean doesn't remove the doc/html directory
- bug 1540 - Waf not finding click libraries
- bug 1549 - Test for NS_LOG
- bug 1556 - Uses of htonl making OpenFlow's match field error
- bug 1563 - Reduce valgrind test scope
- bug 1564 - Packet meta data isn't shown in dumbbell-animation.xml
- bug 1566 - WiFi SNR tag improvements
- bug 1568 - Deserialized addresses are implicitly marked as Mac48Address
- bug 1569 - droptail_vs_red example doesn't run
- bug 1570 - Valgrind errors in new test examples
- bug 1574 - Node color overwritten, by mobility updates in netanim
- bug 1575 - Invert the y-axis in netanim
- bug 1576 - Frequency units HERTZ and MEGAHERTZ mix up
- bug 1577 - Typo in ascii picture in example aodv script
- bug 1579 - edca-txop-n fragmentation causes segfault
- bug 1582 - IPv6 raw socket return value is not like Linux socket
- bug 1585 - Length field of A-MSDU subframe header endianness
- bug 1586 - Building documentation fails if make runs in parallel
- bug 1588 - UdpEchoServer::HandleRead logs fail when using Ipv6
- bug 1589 - Bake - support pre-2.7 version of python
- bug 1590 - Bake - more autotools version support
- bug 1595 - Function declarations without implementations cause problems with dsr module's python bindings
- bug 1596 - Inet TopologyReader is skipping one link and duplicating another one
- bug 1600 - Icmpv6OptionLinkLayerAddress can only carry 48 bit addresses correctly
- bug 1601 - RttEstimator doesn't set the m_currentEstimatedRtt to m_initialEstimatedRtt on creation
- bug 1602 - waf build can break due to file collisions in higher-level directory
- bug 1603 - random-variable-stream-helper - this unavalable for static member functions
- bug 1607 - OnOffApplication over TCP with IPv6 support
- bug 1608 - DSR Network ACK is not handled correctly
- bug 1609 - Route Request table is needed
- bug 1612 - pyviz (visualizer) will not be installed
- bug 1613 - Can't build ns-3-dev with g++ 4.7.2
- bug 1615 - Adjusting OLSR HelloInterval Attribute results in no links
- bug 1618 - bake.py not detecting install of libxml2-dev on ubuntu
- bug 1623 - pybindgen rev809 is not able to build after Ubuntu 1210
- bug 1625 - ns-3-dev fails to build on Debian wheezy amd64
- bug 1626 - ipv6-only network can't use UDP or TCP
- bug 1632 - Prepend bake build directory to the guessed locations
             that waf will look to find libraries
- bug 1633 - Bake - should not report that it is downloading qt4 when it is already installed
- bug 1635 - Small bug without Simulator::Destroy()
- bug 1636 - Compilation error flagged as unmet dependency
- bug 1637 - Bake calling apt-get for unpriviledged user
- bug 1639 - bake.py support for linux mint
- bug 1640 - bake needs to test for g++
- bug 1641 - bake reports autotools dependency, but needs automake
- bug 1661 - Variable ub1 defined but not used in ipv6-address.cc

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.16
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.16.tar.bz2

### Supported platforms

- Fedora Core 17 (32/64 bit) with g++-4.7.0
- Fedora Core 15 (64 bit) with g++-4.6.3
- Ubuntu 12.10 (32/64 bit) with g++-4.6.3
- Ubuntu 12.04 (32/64 bit) with g++-4.6.3
- Ubuntu 10.04.4 LTS (64 bit) with g++-4.4.3
- OS X Mountain Lion 10.8.3 with g++-4.2.1
- FreeBSD 8.2 (32 bit) with g++-4.2.1
- Cygwin 1.7.9-1 with g++-4.5.3

### New user-visible features

- Support several new LTE MAC schedulers developed in GSoC 2012 project.
  Those schedulers include FD-MT, TD-MT, TTA, FD-BET, TD-BET, FD-TBFQ,
  TD-TBFQ, PSS. Here, FD and TD mean frequency domain and time domain
  respectively.
- It's now possible to use distcc when building ns-3 by using
  $ CXX='distcc g++' ./waf configure
  $ ./waf build
- Support topology generation with the (external) BRITE topology generation
  tool

### Bugs fixed

- bug  555 - DCF immediate access
- bug  976 - wifi-wired-bridging regression test fails because of
             rounding errors in mobility model
- bug 1055:  Wrong UAN's Thorp absorption loss model formula
- bug 1284 - ./test.py performance tests do not exist
- bug 1324 - ns3tcp-cwnd-test-suite.cc test failing
- bug 1340 - ns3tcp-loss-suite.cc tests failing
- bug 1342 - ns3tcp-cwnd-test-suite.cc tests failing
- bug 1345 - ns-3 builds with clang compiler >= 3.0
- bug 1359 - TCP cannot receive ICMP
- bug 1361 - Set TOS from Application level (sockets)
- bug 1463 - ./test.py --example should automatically detect the path
- bug 1479 - When the Ipv4RawSocket "IpHeaderInclude" Attribute set true,
             Ip Checksum error
- bug 1485 - Setting global properties in config file (raw text) does
             not have any effect
- bug 1493 - test.py --list should show the test type default tip
- bug 1493 - test.py --list should show the test type
- bug 1494 - test.py --constrain doesn't work
- bug 1495 - test.py claims test passed when no test was run
- bug 1506:  TCP data segment piggybacking acknowledgment
- bug 1509 - RttEstimator returns invalid value in presence of errors
- bug 1516 - GtkConfigStore does not show read-only attributes
- bug 1520 - Config paths not fully documented (partial fix)
- bug 1524 - Fragmentation Threshold equals to Packet Size at MAC Layer
             (Data + IP + UDP) crash the reception
- bug 1525 - Linker error with mpi on Mac 10.8
- bug 1526:  Wrong calculation of pathloss within
             UanPropModelThorp::GetPathLossDb
- bug 1527 - Ipv4RawSocket's BindToNetDevice not working
- bug 1528 - BindToNetDevice not working for IPv6 sockets
- bug 1531 - Crash when using NS_LOG in destructors of static objects
- bug 1532 - unimplemented LTE Scheduler methods

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.15
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.15.tar.bz2

### Supported platforms

- Fedora Core 17 (32/64 bit) with g++-4.7.0
- Fedora Core 15 (64 bit) with g++-4.6.3
- Ubuntu 12.04 (32/64 bit) with g++-4.6.3
- Ubuntu 10.04.4 LTS (64 bit) with g++-4.4.3
- OS X Mountain Lion 10.7.4 with g++-4.2.1
- OS X Snow Leopard 10.6.8 with g++-4.2.1
- FreeBSD 8.2 (32 bit) with g++-4.2.1
- Cygwin 1.7.9-1 with g++-4.5.3

### New user-visible features

- A new random variable base class called 'RandomVariableStream', has been
  introduced.  This base class derives from ns3::Object, unlike the current
  'RandomVariable' class which is a special type of object in the ns-3
  system to date.  By making this class derive from ns3::Object, it can be
  handled with the Ptr class, can carry attributes, and can have its parameters
  and initial state saved in the config-store subsystem.  A new attribute
  called "Stream" has been introduced for this class, to allow users to
  better control the assignment of underlying pseudo-random sequences to
  the RandomVariableStream objects (bug 101).

### Bugs fixed

 - bug 101 - random variable initialization
 - bug 1256 - Unnecessary SND.NXT advance, missing ACK for Out of Order segments
 - bug 1308 - Nix-vector; do not process packets to self
 - bug 1386 - assert if PositionAllocator not provided to
              RandomWaypointMobilityModel
 - bug 1399 - TCP not backing off retransmissions properly
 - bug 1441 - IPv4 header length handling
 - bug 1464 - ConfigStore Save + Load => Could not set default value
                  for ns3::UdpSocketImpl::IcmpCallback
 - bug 1470 - define default parameter for constructor in .h file
 - bug 1473 - GetQueue/SetQueue missing from Csma python bindings
 - bug 1475 - "test.py --list" should sort the output
 - bug 1476 - TestCase::GetName () should not be private
 - bug 1477 - Click doesn't work with raw sockets
 - bug 1481 - Openflow example doesn't run
 - bug 1486 - Segfaults in nix-vector-routing with bridges
 - bug 1489 - Jakes Log component is missing
 - fixed bug with multiple LCs in UL BSR
 - fixed bug with lena-dual-stripe crashes with --homeUesHomeEnbRatio=2

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.14.1
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.14.1.tar.bz2

### Bugs fixed

This hotfix release contains a fix for the PyViz visualizer and makes it
easier to add PyViz support to examples; otherwise it is the same as the
ns-3.14 release.

Release 3.14
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.14.tar.bz2

### Supported platforms

ns-3.14 has been tested on the following platforms.  Not all features are
available on all platforms; check the Installation page on the project wiki.

- Fedora Core 17 (32/64 bit) with g++-4.7.0
- Fedora Core 16 (32/64 bit) with g++-4.6.3
- Fedora Core 15 (64 bit) with g++-4.6.3
- Ubuntu 12.04 (32/64 bit) with g++-4.6.3
- Ubuntu 11.10 (32 bit) with g++-4.6.1
- Ubuntu 10.04.4 LTS (64 bit) with g++-4.4.3
- OS X Lion 10.7.4 with g++-4.2.1
- OS X Snow Leopard 10.6.8 with g++-4.2.1
- FreeBSD 8.2 (32 bit) with g++-4.2.1
- Cygwin 1.7.9-1 with g++-4.5.3

### New user-visible features

- Transport protocol implementations (TCP, UDP) have been refactored to
  support also IPv6 connections. Dual-stacked IPv6 sockets are implemented.
  An IPv6 socket can accept an IPv4 connection, returning the sender's
  address as an IPv4-mapped address (IPV6_V6ONLY socket option is not
  implemented).
- The LTE code from the LENA project has been merged, bringing in a
  significant redesign of the LTE module as well as many new features.
- An antenna module is now included, which includes different
  radiation pattern models. See the corresponding new section of the
  ns-3 models library documentation for details.
- A new buildings module is introduced which allows to model the
  presence of buildings in a wireless network topology.
- New propagation models (OkumuraHata, ITU-R P.1411, ITU-R P.1238)
  have been added
- The Dynamic Source Routing (DSR) MANET routing protocol for IPv4 was added.
- A Random Early Detection (RED) queue model has been added.
- Ipv6RoutingHelper is now in-line with Ipv4RoutingHelper concerning the RT
  print functions. Various minor changes were made in Ipv6RoutingProtocol and
  derived classes to make this possible.
- New "SendIcmpv6Redirect" attribute (and getter/setter functions) to
  Ipv6L3Protocol. The behavior is similar to Linux's conf "send_redirects",
  i.e., enable/disable the ICMPv6 Redirect sending.
- Longer and more descriptive names are used for error units in RateErrorModel
  class and queue mode in Queue class. Attributes in those classes are also
  changed for consistency. See API documentation for details.
- The netanim animator is now bundled with the release.

### Bugs fixed

 - bug 603 - Simulator::Next is useless
 - bug 631 - RealtimeSimulatorImpl does not handle Ctrl-C with python bindings
 - bug 962 - list of paths to reach objects contains bogus entries
 - bug 1000 - Make RealtimeSimulatorImpl last until stop
 - bug 1053 - Need better error diagnostics in ns2-mobility-trace example
 - bug 1109 - Point out the effects of ArpCache::PendingQueueSize
 - bug 1179 - Add LTE trace hooks for supporting visualizer
 - bug 1180 - LTE: Add TX/RX trace support
 - bug 1273 - Better error message on missing Python development files
 - bug 1282 - Sleep state and CCA
 - bug 1283 - PacketSocket::SendTo should return number of bytes
 - bug 1302 - Flow monitor bug
 - bug 1304 - Tag information changed after transmission
 - bug 1313 - Stddev (average.h) returning NaN
 - bug 1318 - Asserts for IPv6 malformed packets
 - bug 1319 - Fix Ipv6RawSocketImpl Icmpv6 filter
 - bug 1325 - Python examples get valgrind errors when build subdirectories
              are specified in waf
 - bug 1339 - test.py uses the old name for the test runner
 - bug 1348 - python bindings: deprecated conversion from string constant
              to ‘char*’
 - bug 1349 - TypeId.LookupAttributeByName Python bindings missing
 - bug 1350 - Simulator.ScheduleWithContext Python binding missing
 - bug 1351 and 1333 - TCP not able to take RTT samples on long delay network
 - bug 1352 - Fixed MapAttributes, previously was mapped to a vector in
              ObjectPtrContainer and (and key information was dropped).
              Now the container is a map.
 - bug 1354 - Building scratch subdir programs is broken
 - bug 1355 - Visualizer dependencies not detected at buildtime
 - bug 1357 - IPv6 fragmentation fails due to checks about malformed extensions
 - bug 1362 - ICMPv6 does not forward ICMPs to upper layers (and minor
              fixes to ICMPv6)
 - bug 1376 - wrong wind variable's name
 - bug 1378 - UdpEchoClient::SetFill () does not set packet size correctly
 - bug 1379 - lte-multiple-flows example crashes
 - bug 1380 - lte-phy-uplink example crashes
 - bug 1391 - .ns3rc does not allow comments as expected
 - bug 1392 - Modules built report does not clarify C++ or Python
 - bug 1393 - IPv6 Routing Helper RT Print functions
 - bug 1395 - AODV DeferredRouteOutputTag missing constructor
 - bug 1396 - ARP with hardware addresses longer than 6 bytes
 - bug 1399 - TCP not backing off retransmissions properly
 - bug 1404 - Bound user input in tutorial third.cc program
 - bug 1406 - waf exits with maximum recursion depth exceeded
 - bug 1415 - examples-to-run.py doesn't work with command line arguments
 - bug 1420 - no python bindings for csma-layout
 - bug 1441 - IPv4 header length handling

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.13
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.13.tar.bz2

### Supported platforms

ns-3.13 has been tested on the following platforms.  Not all features are
available on all platforms; check the Installation page on the project wiki.

- Ubuntu 11.10 (32 bit) with g++-4.5.2
- Ubuntu 11.04 (32/64 bit) with g++-4.5.2
- Ubuntu 10.04.3 LTS (64 bit) with g++-4.4.3, g++-3.4.6
- OS X Lion with g++-4.2.1
- OS X Snow Leopard with g++-4.2.1
- Fedora Core 16 (32/64 bit) with g++-4.6.2
  - however, g++-3.4.6 fails on Fedora 16 i686
- Fedora Core 14 (64 bit) with g++-4.5.1
- FreeBSD 9.0-RC1 (AMD64) with g++-4.2.1

### New user-visible features

- IPv6 address generator with support for duplicate address detection

### Bugs fixed

- bug 962 - list of paths to reach objects contains bogus entries
- bug 1001 - Buffer::CopyData() doesn't return the number of bytes copied
- bug 1010 - Uan model sleep patch
- bug 1020 - Wrong usage of the originator sequence number in HWMP PREP
- bug 1021 - Beacon collision avoidance in Mesh module works incorrectly
- bug 1039 - Nagle's algorithm in TCP
- bug 1055 - Wrong UAN's Thorp absorption loss model formula
- bug 1059 - Unable to load trace files created from SUMO and TraNS Lite
- bug 1112 - Advance m_nextTxSequence upon retransmit after RTO
- bug 1137 - mpi module is hard-coded for openmpi
- bug 1166 - IPV4 TCP failed to send a RST when connect arrives before listen
- bug 1186 - Ipv4Header lacks DSCP and ECN
- bug 1204 - Can't Parse Time +100000000.0ns
- bug 1219 - Coding style of ns2-mobility-helper-test-suite.cc is fixed
- bug 1257 - waf install __init__ Python files even with --disable-python
- bug 1263 - waf configure fails on FreeBSD 9.0-BETA2 amd64
- bug 1266 - gdb cannot be loaded
- bug 1227 - Spurious RTO due to low min RTO
- bug 1229 - Multiplication overflow in WaypointMobilityModel::EndMobility
- bug 1242 - m_lastRtt in tcp-socket-base.cc not implemented
- bug 1256 - TCP unnecessary snd.nxt advance
- bug 1265 - Make ns-3 directory "movable"
- bug 1269 - sqlite3 not found on FreeBSD
- bug 1270 - "Checking boost includes" weirdness
- bug 1278 - Ipv4ClickRouting::HandleScheduleFromClick bug
- bug 1281 - Checksum not calculated when doing IP fragmentation
- bug 1285 - IPv6 Localhost is marked as GLOBAL instead of HOST
- bug 1290 - buffer-test.cc gets a valgrind error
- bug 1295 - Missing const qualifiers in TopologyRead
- bug 1299 - EnableAsciiIpv4All tracing doesn't show transmitted/recvd packets
- bug 1300 - HalfDuplexIdealPhy notify SpectrumInterference of AbortRx
- bug 1301 - Ns2MobilityHelper causes Node GetPosition() to return NaN
- bug 1305 - do not list modules built upon exiting waf shell
- bug 1312: TopologyRead Assert condition fix
- IPv4 packets double fragmentation was broken
- Fix wifi-clear-channel-cmu.cc example
- NetAnim: fix for bcast packet reuse
- Missing PropagationLossModel.CalcRxPower in Python bindings
- Corrected compilation behavior in Ubuntu 11.10 due to ldd behavior change
- Added required PTHREAD dependency to RT library check.

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.12.1
--------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.12.1.tar.bz2

### Bugs fixed

This release fixes PyViz visualizer's python bindings; otherwise, it
is the same as the ns-3.12 release.

Release 3.12
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.12.tar.bz2

### Supported platforms

ns-3.12 has been tested on the following platforms.  Not all features are
available on all platforms; check the Installation page on the project wiki.

- Fedora Core 15 (32/64 bit) with g++-4.6.0
- Ubuntu 11.04 (32/64 bit) with g++-4.5.2
- OS X Lion with g++-4.2.1
- Fedora Core 14 (64 bit) with g++-4.3.4, g++-4.5.3, g++-4.4.6
- Fedora Core 12 (64 bit) with g++-4.4.4
- OS X Snow Leopard with g++-4.2.1
- Ubuntu 10.04.3 LTS (64 bit) with g++-4.4.3, g++-3.4.6

### New user-visible features

- SpectrumChannel models now support the usage of single-frequency
  propagation loss models based on the PropagationLossModel
  class. These model can be used in conjunction with
  the frequency-dependent propagation loss model based on the
  SpectrumPropagationLossModel class already supported by
  SpectrumChannel.

- Extend the attribute system extend the attribute system to store map
  container objects in addition to vector containers.  The ObjectMap
  and related classes are patterned after ObjectVector.

- Support for IPv4 fragmentation has been added.

- Significant performance improvement for AODV in dense mobile scenarios
  due to a fixed bug in RERR processing.

- WiFi traces that were named promiscuous are now renamed monitor, and
  a separate flag for setting the interface to promiscuous mode has been
  introduced.  This disambiguates monitor mode from promiscuous mode.

- Monolithic Python bindings have been removed

### Bugs fixed

- bug 845  - fix ConfigStore output for changing default values
- bug 1010 - Uan model Sleep patch
- bug 1033 - Mesh airtime-metric fixed
- bug 1043 - lte-spectrum-value-helper.cc file swapped uplink/downlink freq.
- bug 1050 - ISO C++ forbids variable length array; remove all instances
- bug 1057 - LTE bearers differentiation issue
- bug 1082 - Check that log component names are correct when taken from NS_LOG
- bug 1100 - IPv6 fragmentation enhancements
- bug 1102 - IPv4 header fragment offset field not set correctly
- bug 1123 - class MinMaxAvgTotalCalculator does not implement all methods
- bug 1134 - Fix LiIonEnergySource attribute spelling for InitialCellVoltage
- bug 1139 - Broadcast packets issue in uan cw mac
- bug 1141 - MeshWifiInterfaceMac adds supported modes to wrong remote station
- bug 1163 - Ipv4EndPointDemux::AllocateEphemeralPort forget to increment port
- bug 1164 - IPV4 TCP Bind already used port failed, but without setting errno
- bug 1168 - BridgeNetDevice: default expiration time not realistic
- bug 1174 - Ns-3 does not generate static libraries
- bug 1175 - shared libraries are not versioned
- bug 1193 - AODV Hello timers scheduling skewed when RREP, RREQ are processed
- bug 1194 - AODV Incorrect RERR processing
- bug 1201 - UDP sockets need to set connected state before notifying the ConnectionSucceeded callback
- bug 1208 - Spectrum module example causing valgrind errors
- bug 1209 - ns-3-click fails to build on Fedora 15 when tests are enabled
- bug 1220 - FdReader always stops with NS_FATAL_ERROR
- bug 1222 - Incomplete PointToPointNetDevice Python bindings
- bug 1224 - ns-3-allinone fails to compile on OS X Lion
- bug 1226 - Incomplete YansWifiChannel Python bindings
- bug 1234 - GetAttributeChecker is not defined anymore but still used
- bug 1239 - Add assertion that reference count never overflows
- bug 1244 - Seg Faults in optimized builds due to missing Logging components
- Fix ARP issue in nsclick-ip-router.click and static route in nsclick-routing-node2.click

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.11
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.11.tar.bz2

### Supported platforms

ns-3.11 has been tested on the following platforms.  Not all features are
available on all platforms; check the Installation page on the project wiki.

- Linux x86_64 Ubuntu 11.04
  - g++-4.5.2
- Linux i686 Ubuntu 11.04
  - g++-4.5.2, g++-4.4.5
- Linux x86_64 Fedora Core 15
  - g++-4.6.0
- Linux x86_64 Fedora Core 14
  - g++-4.5.3, g++-4.4.6, g++-4.3.4
- Linux x86_64 Fedora Core 10
  - g++-3.4.6, 4.0.4, 4.1.2, 4.2.4, 4.3.2, 4.4.0
- OS X Snow Leopard
  - g++-4.2.1

### New user-visible features

- The build system has been modularized, and the source code reorganized,
  to allow for modular libraries instead of a single monolithic ns-3
  library.   User programs now link a number of smaller, per-module
  libraries depending on the dependencies expressed to the build system.
  Source code is now being maintained in individual modules with
  consistent directory structures.

- Python bindings have also been modularized, and the bindings are now
  generated into a 'ns' namespace instead of 'ns3' for the old
  (monolithic) bindings.  By default the bindings are now modular, with
  a backward compatibility layer for older programs.

- By default, example and test programs are disabled from the build.
  Users can enable them via the use of a waf command at configure time,
  or automatically through the use of a new .ns3rc file.

- int64x64_t is a new type which allows portable and easy to write arithmetic
  calculations that require a high degree of fractional precision.

- An interface to the Click Modular Router and an Ipv4ClickRouting
  class has been added, to allow a node to use Click for external routing.

- An interface to an OpenFlow software implementation distribution has
  been added to allow the simulation of OpenFlow switches in ns-3.

- ns-3 coding style requirements have been applied to the entire coding
  base, resulting in a large number of whitespace changes.  The coding
  style has been automatically enforced by running a style checking
  program (utils/check-style.py) that uses the uncrustify (v. 0.58)
  program to apply changes.

- Some documentation reorganization to split documentation between
  the ns-3 manual (primarily focusing on the ns-3 core) and a new
  "model library" document has been started.

### Bugs fixed

The following lists many of the bugs fixed or small feature additions
since ns-3.10, in many cases referencing the Bugzilla bug number.

- bugfix: CsmaNetDevice ErrorModel not discarding packet
- bugfix: do not call RouteOutput() twice
- Workaround gccxml bug scanning Seconds(0) as default value
- Copy a bug fix from ns3modulescan.py into ns3modulescan-modular.py
- Modular Python bindings work (many bug fixes, more modules tested)
- Modular bindings: bug fixes, more verbose scanning
- ./waf --apiscan: fix bug in detection of whether the per-module
- Remove the 'Modules to build' debug print
- CsmaNetDevice ReceiveErrorModel was not dropping the packet
- Several "set but not used" bugs for newer compilers quick-fixed

- bug 445 - Is the class name Scalar in nstime.h appropriate?
- bug 699 - TestCase::DoRun probably should not return a bool
- bug 823 - Need finer-grained control over what goes into libns3
- bug 957 - Issue with test.py
- bug 1017 - node --> internet-stack --> node
- bug 1018 - mobility --> helper --> mobility circular dependency
- bug 1019 - common --> node --> common circular dependency
- bug 1038 - Time::Get*Seconds () return signed integer while actually returning unsigned.
- bug 1040 - Olsr and multiple interfaces
- bug 1042 - AODV RERR implosion (missing RERR_RATELIMIT)
- bug 1044 - Seconds (1e-9) creates Time that is not IsPositive ()
- bug 1047 - Multicast routes on nodes with >16 interfaces
- bug 1048 - suggested MatrixPropagationLossModel::SetLoss() API change
- bug 1049 - Issues with OLSR HNA messages sent by multiple gateways
- bug 1052 - olsr::RoutingProtocol::LinkTupleUpdated incorrect detection of asymmetric link with multiple interface nodes
- bug 1054 - ipv6 InternetStackHelper EnablePcapIpv6All() broken
- bug 1056 - CSMA: padding not handled correctly for LLC encapsulation
- bug 1058 - InternetStackHelper pitfall: calling Install before adding routing protocols
- bug 1060 - Setting NqStaWifiMac Active Probing true, crashes simulation
- bug 1061 - Title level inconsistent in click.rst
- bug 1062 - rescale-pdf not working
- bug 1064 - Correct Friis propagation loss equation in spectrum module
- bug 1066 - Set Radiotap modulation type flags correctly
- bug 1069 - ApWifiMac unduly calls RegularWifiMac::Receive and crashes
- bug 1070 - csma/ipv6 examples duplicate NS_LOG_COMPONENT_DEFINE
- bug 1072 - crash upon call to Packet::AddAtEnd of fragmented packets
- bug 1075 - Python examples fail when static built is enabled
- bug 1076 - Waf gives an error if you enable only a single module
- bug 1078 - PacketTagList::Add performs NS_ASSERT counting one byte too few
- bug 1079 - MPI code doesn't compile
- bug 1081 - wutils.py refers to the old directory for visualizer module
- bug 1083 - scratch doesn't work when only some modules are enabled
- bug 1085 - Build failed: list index out of range
- bug 1086 - sanity check for not calling Object::Dispose () twice
- bug 1087 - Call MPI_Finalize to exit MPI environment
- bug 1088 - Code in scratch directory assumes all modules are enabled
- bug 1089 - lwip build issue
- bug 1090 - WifiMacQueue is not accessible through the attribute path
- bug 1092 - Problem building FreeBSD stack with g++ 4.5.2 (Ubuntu)
- bug 1094 - Object::GetObject upon dlopen
- bug 1097 - AODV routing entry set to be VALID mistakenly.
- bug 1098 - when to show users which modules are enabled?
- bug 1101 - Sqlite stats disappeared from configuration file src/wscript
- bug 1103 - Useless assignment in omnet-data-output.cc
- bug 1105 - Move topology helpers into separate per-device modules
- bug 1106 - Remove "CanvasLocation" dependencies in p2p helpers
- bug 1108 - test.py -e option needs repair
- bug 1110 - Sort alphabetically and display "Modules built" in columns
- bug 1117 - bindings failures on selected platforms
- bug 1122 - spectrum-value test should check tolerance on a per-value basis
- bug 1125 - ./waf --enable-modules fails
- bug 1131 - Bug in Ipv4L3Protocol::RemoveAddress()
- bug 1133 - DSDV: Possible bug in settling time calc
- bug 1142 - Wrong 802.11p Slot time
- bug 1146 - help string for test.py with disabled tests

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Release 3.10
------------

### Availability

This release is available from:
http://www.nsnam.org/release/ns-allinone-3.10.tar.bz2

### Supported platforms

ns-3.10 has been tested on the following platforms.  Not all features are
available on all platforms; check the Installation page on the project wiki.

- Linux i686 Ubuntu 10.10
  - g++-4.4.5
- Linux x86_64 Ubuntu 8.04.4
  - g++-3.4.6, g++-4.2.4
- Linux x86_64 Fedora Core 12
  - g++-4.4.4
- Linux x86_64 Fedora Core 10
  - g++-3.4.6, 4.0.4, 4.1.2, 4.2.4, 4.3.4, 4.4.0
- OS X Snow Leopard
  - g++-4.2.1
- OS X powerpc Leopard
  - g++-4.0, g++-4.2

### New user-visible features

- NS-3 PyViz, a live simulation visualizer, has been added.  The
  visualizer interacts with a running simulation, such that it uses no
  trace files. It can be most useful for debugging purposes, i.e. to
  figure out if mobility models are what you expect, where packets are
  being dropped, etc. There is also a builtin interactive python console
  that can be used to debug the state of the running objects. Although
  it is mostly written in Python, it works both with Python and pure C++
  simulations.

- Wi-Fi MAC high classes have been reorganised in attempt to
  consolidate shared functionality into a single class. This new
  class is RegularWifiMac, and it derives from the abstract
  WifiMac, and is parent of AdhocWifiMac, StaWifiMac, ApWifiMac,
  and MeshWifiInterfaceMac. The QoS and non-QoS class variants are
  no longer, with a RegularWifiMac attribute "QosSupported"
  allowing selection between these two modes of
  operation.

- Support for IEEE 802.11g (specifically, rates with Modulation
  Class ERP-OFDM) has been added to the wifi module. The new
  WifiModes have names of the form ErpOfdmRatexxMbps, where xx is
  the rate in Mbps (one of 6, 9, 12, 18, 24, 36, 48, 54), and a
  WifiPhyStandard enumeration WIFI_PHY_STANDARD_80211g has been
  added to allow configuration of a Wi-Fi STA supporting ERP-OFDM,
  HR/DSSS, and DSSS rates (which some call an "802.11b/g" station).

- Two attributes were added to WaypointMobilityModel.  The first
  "LazyNotify" (default false), if enabled to true, will suppress
  course change notifications if there are no position updates.
  The second, "InitialPositionIsWaypoint" (default false), if enabled,
  will allow an initial call to SetPosition () (before any waypoints
  have been added) to be treated as the first waypoint.

- Regression testing is now completely performed within test.py
  rather than a separate "./waf --regression" that consults
  reference traces in the ns-3.x-ref-traces directory.  In most
  cases, the example programs whose traces were included in
  ns-3.x-ref-traces have a corresponding test called from the
  test.py program.  It is still possible to write trace-based
  regression tests but they are now called from the test.py
  program.

- New BulkSendApplication sends data as fast as possible up to
  MaxBytes or unlimited if MaxBytes is zero.  Think OnOff, but
  without the "off" and without the variable data rate. This
  application only works with SOCK_STREAM and SOCK_SEQPACKET
  sockets, for example TCP sockets and not UDP sockets.

- Extensions to the energy models:  1) a new Rakhmatov Vrudhula
  non-linear battery model, 2) additional support for modeling
  energy consumption in WiFi devices, 3) an example for how to add
  energy models to a WiFi-based simulation (in examples/energy/
  directory).

- New methods to print IPv4 routing tables to an output stream.

- A replacement implementation for TCP for IPv4.  TcpSocketBase
  now replaces TcpSocketImpl.  There are subclasses defined for
  TCP Tahoe, Reno, NewReno, and the original RFC793 without
  congestion control.

- Energy support for UAN module and Li-Ion energy model.
  Now each network node can have an associated energy source
  from which it consumes energy. The user can specify a custom
  energy model for the underwater transducer, with default values
  set for WHOI acoustic transducer. Li-Ion battery model added
  into energy models folder.

- Destination-Sequenced Distance Vector (DSDV) routing protocol
  is a proactive, table-driven routing protocol for MANETs
  developed by Charles E. Perkins and Pravin Bhagwat in 1994.
  This implementation is for IPv4 routing and was contributed
  by ResilNets Research Group.

- A novel model to simulate LTE networks has been added.  It focuses
  mainly on modeling the E-UTRA part of the system, with a particular
  attention on the aspects related to the channel, PHY and MAC layers.
  The most important features available at this moment are (i) a basic
  implementation of both the UE and the eNB devices, (ii) RRC and
  entities for both the UE and the eNB, (iii) an Adaptive Modulation
  and Coding (AMC) scheme for the downlink, (iv) the management of the
  data radio bearers (with their QoS parameters), the MAC queues and
  the RLC instances, (v) Channel Quality Indicator (CQI) management,
  (vi) support for both uplink and downlik packet scheduling,
  (vii) a PHY layer model with Resource Block level granularity, and
  (viii) a channel model with the outdoor E-UTRAN propagation loss
  model.

- Project documentation has been converted from GNU Texinfo to Sphinx.

### Bugs fixed

The following lists many of the bugs fixed or small feature additions
since ns-3.9, in many cases referencing the Bugzilla bug number

- bug 824 - TCP should implement FastRecovery by default
- bug 852 - Add support for 802.11g devices
- bug 892 - WaypointMobilityModel incompatible with MobilityHelper::Install
- bug 893 - Lazy CourseChange notification for WaypointMobilityModel
- bug 903 - TapBridge does not shut down properly
- bug 953 - WiMAX channel scanning overflow
- bug 967 - Need to decouple Ipv4L4Protocols from Ipv4L3Protocol
- bug 978 - Consolidate Wi-Fi MAC high functionality
- bug 979 - Multi-octet fields in Wi-Fi headers have wrong endianness
- bug 981 - use a cache to keep track of received MPDUs under block ack
- bug 983 - handle correctly ADDBA response action frames in QadhocWifiMac
- bug 984 - EmuNetDevice should use DIX encapsulation by default
- bug 985 - WiMAX Invalid management message type on wimax-simple
- bug 988 - MacRxMiddle::SequenceControlSmaller method
- bug 990 - Error with an NS_ASSERT at uan-phy-gen.cc
- bug 991 - InterferenceHelper assertion failure
- bug 992 - Unusual (possibly incorrect assertion) in Packet::Deserialize
- bug 993 - MinstrelWifiManager::UpdateStats useless if clause
- bug 994 - PointToPointGridHelper useless if clauses
- bug 995 - Useless (possibly incorrect) comparison of unsigned int
- bug 1004 - module header not rebuilt
- bug 1005 - GetSocket() methods for OnOffApplication and PacketSink
- bug 1009 - decouple m_finSequence (TcpSocketImpl) from tx sequence numbers
- bug 1012 - UAN Throp propagation model bug
- bug 1015 - GetChannelFrequencyMhz() does not match with standard
- bug 1022 - inappropriate ASSERT in tcp-socket-impl.cc
- bug 1025 - wimax-ipv4 script exists with signal SIGSEGV when nbSS>20
- bug 1027 - RocketfuelTopologyReader is not working at all
- bug 1029 - v4Ping application endian issues
- bug 1030 - routing/aodv example fixed
- bug 1031 - Wifi hidden terminal example does not work
- bug 1032 - Unable to specify multiple Compiler/Linker flags

### Known issues

In general, known issues are tracked on the project tracker available
at http://www.nsnam.org/bugzilla/

Windows XP 32 bit Cygwin 1.7.7 does not seem to work because of a
problem (reported elsewhere) in linking large libraries.

Release 3.9
-----------

### Availability

This release is immediately available from:
http://www.nsnam.org/release/ns-allinone-3.9.tar.bz2

### Supported platforms

ns-3.9 has been tested on the following platforms:
- Ubuntu 10.04 LTS 32/64-bit gcc-4.4.3
- Fedora Core 12 64-bit gcc-4.4.3
- Ubuntu 9.10 32-bit gcc-4.4.1
- Ubuntu 8.04 LTS 64-bit gcc-4.2.4
- Fedora Core 5 32-bit gcc-4.1.1
- Mac OS X Leopard powerpc gcc-4.0.1
- Windows Cygwin 1.7.5

Not all ns-3 options are available on all platforms; consult the
wiki for more information:
http://www.nsnam.org/wiki/Installation

### New user-visible features

- A reorganisation of the construction of Wi-Fi transmit rates has
  been undertaken with the aim of simplifying the task of
  supporting further IEEE 802.11 PHYs. This work has been completed
  under the auspices of Bug 871. A consequence of this work is that
  WifiModes have a new (hopefully future-proof) naming, and
  simulations which specify rates by name will need updating. The
  list of new rate names is given in the error message produced
  when an attempt is made to use any invalid rate name (including
  those that were previously valid).

- A new OFDM error rate model for WiFi (NistErrorRateModel); this model
  has been validated in clear-channel testbed tests.  For 802.11b, it
  uses the same underlying model as the YansErrorRateModel, but it differs
  from YansErrorRateModel for OFDM modes (802.11a/g).  The default
  YansWifiPhyHelper has been changed to use this model.

- A new NS-2 mobility trace reader supports BonnMotion, SUMO, TraNS,
  etc. traces.

- An energy model for nodes and devices, including an energy source
  model and device energy models allowing energy-aware devices
  to notify the energy source about energy consumption.

- Rocketfuel topology dataset support for existing topology reader
  Rocketfuel project:
  http://www.cs.washington.edu/research/networking/rocketfuel/

- Underwater Acoustic Network (UAN) model, split in to three parts, the
  channel, PHY, and MAC models to enable researchers to model a variety
  of underwater network scenarios.

- the Spectrum framework, which provides support for spectrum-aware
  Channel and PHY layer modeling. It features a set of classes that
  allows the mathematical representation of frequency dependent
  phenomena. Some example channel, propagation and device
  implementations that make use of this framework are also
  included.

### Bugs fixed

The following lists many of the bugs fixed or small feature additions
since ns-3.8, in many cases referencing the Bugzilla bug number

- IPv4 global routing code; stub networks were assumed to be /24
  and added as /24 networks even if the mask is actually different
- OLSR was generating messages on non-OLSR interfaces
- ICMPv4 objects not properly registered; affected serialization code
- bug 231 - SocketAddressTag needs to be removed from a packet before
            forwarding the packet to the user
- bug 385 - Add a generic "sequence number" class
- bug 473 - [PATCH] Alternative ns-2 trace reader
- bug 665 - Need Tutorial Sectino on Config Path and Tracing Use
- bug 671 - add packet-info-tag.cc for IP_PKTINFO/IPV6_PKTINFO
- bug 702 - Global routing doesn't respond to interface events
- bug 799 - Interference helper is too slow
- bug 800 - Erroneous packet sender while using aloha or cw mac in uan module
- bug 802 - Minstrel algorithm causes segmentation fault
- bug 826 - Using uint64_t instead of Time in DcfManager
- bug 828 - PacketSocket::Close does not unregister protocol handler
- bug 842 - ns-3-dev crashes using block acks
- bug 843 - Most wifi examples change BeaconInterval to unrealistic values
- bug 844 - YansWifiPhy::GetPowerDbm off-by-one problem when calculating Tx
            power
- bug 846 - packet.cc triggers deprecated warning
- bug 853 - Rates for Wi-Fi control responses are incorrectly selected
- bug 854 - Support DROP_QUEUE reason-code in Ipv4FlowProbe
- bug 858 - support MSG_PEEK in IPv4/IPv6 raw socket
- bug 861 - Forwarding drops (due to no route found) were not being logged
            in IPv4 or IPv6 ascii traces
- bug 869 - suggested test framework enhancements
- bug 871 - naming for WifiPhyStandard
- bug 873 - Queue occupancy counter not decremented in WifiMacQueue::Remove()
- bug 874 - wrong modulation type is selected in the forwardBurst method
- bug 875 - "frame includes FCS" flag should be set in Radiotap frame header
- bug 879 - source address selection for AODV using DeferredRouteRequest
- bug 881 - Reorganise to allow wider use of WifiInformationElement
- bug 890 - several rate adaptation algorithms not tested by test.py
- bug 900:  RawTextConfigLoad::Default does not load configurations
- bug 901 - Optimize Mac48Address < != and ==
- bug 902 - TCP: handle out-of-order packets during connection shutdown
- bug 905 - WimaxNetDevice loses packet uid, tags, and memory optimization
- bug 906 - NSC TCP socket fork did not copy txbuffersize over
- bug 908 - test.py should exit if waf dies
- bug 910 - Change Wi-Fi "AccessClass" to something closer to the standard
- bug 911 - IPv4/v6 multicast forwarding not going to all output interfaces
- bug 913 - Queue Enqueue/Drop trace sources behavior unintuitive
- bug 916 - EnableAsciiAll ("prefix") does not work for YansWifiPhyHelper
- bug 918 - samples/main-packet-header.cc is broken
- bug 919 - minstrel does not pass valgrind tests
- bug 921 - Inconsistent declaration of class/struct Object in object.h
- bug 922 - Inconsistent declaration of class/struct in
            wifi-remote-station-manager.h
- bug 923 - Inconsistent declaration of class/struct in mac-low.h
- bug 924 - Inconsistent declaration of class/struct in <xxx>-wifi-managers
- bug 925 - Various IPv6 cc files trigger deprecated warning
- bug 926 - olsr handling of multicast packets
- bug 927 - SimpleOfdmWimaxChannel RxPower computation
- bug 930 - examples/topology-read/topology-example-sim.cc uses variable
            length array
- bug 931 - Abnormal exit reports SIGSEGV on failure
- bug 932 - Support IP_HDRINCL option for Ipv4RawSocket
- bug 933 - Flushing ostream and files on abnormal program exit
            (ASSERT, ABORT and FATAL_ERROR)
- bug 936 - Waf build error for python bindings on "AccessClass"
- bug 937 - bugs in ns-3 wimax
- bug 939 - EmuNetDevice uses too much memory when reading packet bursts
- bug 940 - AODV fails to set up a correct path
- bug 941 - Wifi Dcf attributes not reachable through configuration namespace
- bug 943 - Add a SO_BROADCAST socket option
- bug 944 - change default ErrorRateModel from YansErrorRateModel to
            NistErrorRateModel
- bug 946 - Rocketfuel topology dataset support for topology reader
- bug 949 - Node::NonPromiscReceiveFromDevice reports a meaningless
            destination address to user callbacks
- bug 950 - PointToPointNetDevice says promisc support is not implemented
            but it is
- bug 955 - Install NSC for NS3 in a Itanium Architecture (ia64) with Linux 2.6.16.6
- bug 956 - Bindings failure in core (traced-value) with older gcc
- bug 958 - WiFi uses wrong default values for MTU, FragmentationThreshold
            and RtsCtsThreshold
- bug 959 - Simulation never finishes when using RealtimeSimulatorImpl
- bug 960 - V4Ping does not generate checksum
- bug 961 - Problem with MPI activation
- bug 964 - AODV does not work with host addressed interfaces
- bug 965 - Problem building /src/core/callback.cc
- bug 966 - AODV originates new RREQ on each packet, when doesn't have
            valid route
- bug 969 - No SocketList is present in UdpL4Protocol class
- bug 971 - fix AODV header Print functions
- bug 972 - [flow monitor] assertion fails in pv4-flow-probe.cc

### Known issues


Release 3.8
-----------

### Availability

This release is immediately available from:
http://www.nsnam.org/release/ns-allinone-3.8.tar.bz2

### Supported platforms

ns-3.8 has been tested on the following platforms:
  - linux x86 gcc 4.4.0, 4.3.2, 4.2, 4.1.1, 4.1 and 3.4.6 (debug and optimized)
  - linux x86_64 gcc 4.4.0, 4.3.2, 4.2.4, 4.2.3, 4.2.1, 4.1.3, 3.4.6 (debug and optimized)
  - MacOS X ppc gcc 4.0.x and 4.2.x (debug and optimized)
  - cygwin gcc 4.3.4 (debug and optimized)

Not all ns-3 options are available on all platforms; consult the
wiki for more information:
http://www.nsnam.org/wiki/Installation

### New user-visible features

- WiMAX net device: Allow to simulated IEEE 802.16 point to multi-point based networks

- Distributed simulation for point-to-point networks using the Message
  Passing Interface (MPI) standard.

- Matrix propagation loss model uses a two-dimensional matrix of path loss indexed
  by source and destination nodes.

- Topology read system: Allows quick and easy creation of large topologies by reading
  Inet or Orbis files

- Gauss-Markov mobility model: Adds 3-d adaptation of Gauss-Markov mobility model which
  has both memory and variability

- Steady-state random waypoint mobility model: Based on random waypoint mobility (RWM) model for case when
  speed, pause and position are uniformly distributed random variables. However, initial values of these
  parameters are not from uniform distribution but from stationary distribution of RWM model.

- Two-ray ground propagation loss model: Calculates the crossover distance under which Friis is used.
  The antenna height is set to the nodes z coordinate, but can be added to using the model parameter
  SetHeightAboveZ, which will affect ALL stations


### API changes from ns-3.7

API changes for this release are documented in the file CHANGES.html.

### Bugs fixed

The following lists many of the bugs that were fixed since ns-3.7, in
many cases referencing the Bugzilla bug number
- bug 155 - std::ostream & os" parameters not Python friendly
- bug 184 - GtkConfigStore do not support ConfigureDefault
- bug 407 - OLSR is missing HNA support
- bug 414 - No ReceiveErrorModel in SimpleNetDevice
- bug 602 - WifiRemoteStation lacks information about the access class of outgoing packets
- bug 622 - [PATCH] Friendly names for pcap traces
- bug 683 - Helper methods for pcap tracing with explicit filenames
- bug 706 - Backoff counting when starting NS.
- bug 720 - TapBridge creation fails from a script outside the ns3 tree
- bug 731 - Send function in point-to-point-net-device fails to check the return value of the Dequeue function
- bug 747 - Listening TCP socket closes on RST
- bug 748 - Cloned TCP socket uses wrong source address
- bug 772 - AODV is unable to correctly buffer packets waiting for route reply
- bug 777 - AODV ignores specified outgoing interface in RouteOutput()
- bug 778 - OLSR ignores specified outgoing interface in RouteOutput()
- bug 787 - Addition of Two Ray Ground model to propagation loss model and tests
- bug 788 - OLSR_NEIGH_HOLD_TIME should be 3 times OLSR_REFRESH_INTERVAL
- bug 789 - [PATCH] Globalrouting externalroutes to use the new GetRootExitDirections()
- bug 794 - Ipv4Mask constructor for "/yy"-notation is wrong
- bug 796 - TCP bug in ns-3-dev branch : Crash detected during retesting of Chord on ns-3-dev branch
- bug 797 - Enhancements to src/core/random-variable.cc/h
- bug 801 - ns-3.7 and SVN not coexisting nicely
- bug 802 - Minstrel algorithm causes segmentation fault
- bug 804 - null-pointer references in 3.7 internet stack
- bug 806 - TCP doesn't work over a CSMA link
- bug 807 - ns2-mobility-helper.cc: node id parsed wrong
- bug 809 - Missing Python binding for Ipv4GlobalRouting::GetRoute
- bug 810 - In TCP, Socket::GetSockName() does not return the local socket address
- bug 812 - Assert when getting socket in RecvReply for AODV
- bug 813 - Nqos AP sends packet to non associated STA
- bug 814 - Function logging causing assert in wireless examples
- bug 815 - waf shell file descriptor leak
- bug 816 - tap-creator deadlock when python bindings enabled
- bug 817 - Pareto rng constructors using scale and shape instead of mean and shape
- bug 818 - TCP Socket implementation does not set ACK flag on retransmits
- bug 819 - Build break when gtk not installed
- bug 820 - Bad things happen in test.py when logging is enabled
- bug 821 - AODV asserts with function logging enabled
- bug 822 - Move Mtu attribute from NetDevice base class to subclasses
- bug 825 - UDP-Client-server's packet loss counter not properly reset
- bug 828 - PacketSocket::Close does not unregister protocol handler
- bug 829 - TCP unbound memory problem (pending data)
- bug 833 - OnOffApplication with PacketSocket: sniffs all traffic
- bug 834 - Incorrect signature of Ipv4FlowProbe::DropLogger
- bug 835 - Unlimited receive queues in sockets == evil
- bug 836 - Delay is incremented over time with BsUplinkSchedulerSimple and BsUplinkSchedulerRtps
- bug 838 - ns-3 can't compile on MacOS with 32bit processor
- bug 839 - TestSuite wimax-ss-mac-layer crashes on Darwin 9.8.0 Power Macintosh
- bug 840 - BS scheduler does not support fragmentation for UGS flows
- bug 841 - Multicast transmission breaks with QoS Wifi
- bug 844 - YansWifiPhy::GetPowerDbm off-by-one problem when calculating Tx power
- bug 847 - Segfaults on BaseStationNetDevice with OnOffApplication and rtPS sched
- bug 849 - stray patch files in lwip directory
- bug 850 - Ipv4GlobalRouting::LookupGlobal bug
- bug 855 - waf dies badly when switching from debug to optimized build or vice versa
- bug 856 - initialize vbl
- bug 857 - Link-Local Multicast handle in Ipv4 Output processing
- bug 859 - Output interface estimation for the source address bound socket in IPv4 Raw socket
- bug 860 - waf sometimes dies while executing ns3header or gen_ns3_module_header tasks in case of
            parallel jobs
- bug 862 - NotifyInterfaceUp() Adds network route even when netmask is /32
- bug 863 - Wrong Scalar arithmetic
- bug 864 - Invalid return value in UdpSocketImpl::Send and Ipv4RawSocketImpl::Send
- bug 865 - Ipv4RawSocketImpl::RecvFrom does not return from address all the time.
- bug 866 - WiMAX mobility models not aggregated to Node
- bug 867 - Minor bug in Ipv4L3Protocol::Send()
- bug 868 - invalid packet size after Ipv4L3Protocol::Send
- bug 872 - ns3::PcapFileWrapper::Write explodes stack
- bug 873 - Queue occupancy counter not decremented in WifiMacQueue::Remove()
- bug 876 - Tcp socket does not handle ShutdownRecv correctly
- bug 877 - python bindings broken with multiple inheritance ?
- bug 880 - Node sending a packet to itself via 127.0.0.1 aborts
- bug 885 - Error in Ascii tracing in Python examples
- bug 888 - Writing ascii trace to additional tests fails
- bug 891 - WiMAX device helper does not include propagation loss model by default
- bug 894 - ./waf --run error message upon segfault
- bug 895 - SimpleOfdmWimaxPhy SNR computation
- bug 899 - EmuNetDevice::SetPromiscReceiveCallback not implemented

### Known issues

ns-3 builds have been known to fail on the following platforms:
- gcc 3.3 and earlier
- optimized builds on gcc 3.4.4 and 3.4.5
- optimized builds on linux x86 gcc 4.0.x


Release 3.7
-----------

### Availability

This release is immediately available from:
http://www.nsnam.org/release/ns-allinone-3.7.tar.bz2

### Supported platforms

ns-3.7 has been tested on the following platforms:
- linux x86 gcc 4.4.0, 4.3.2, 4.2, 4.1.1, 4.1 and 3.4.6 (debug and optimized)
- linux x86_64 gcc 4.4.0, 4.3.2, 4.2.4, 4.2.3, 4.2.1, 4.1.3, 3.4.6 (debug and optimized)
- MacOS X ppc gcc 4.0.x and 4.2.x (debug and optimized)
- cygwin gcc 3.4.4 (debug only), gcc 4.3.2 (debug and optimized)

### Unofficially supported platform

- mingw gcc 3.4.5 (debug only)

Not all ns-3 options are available on all platforms; consult the
wiki for more information:
http://www.nsnam.org/wiki/Installation

### New user-visible features

- Ad hoc On-Demand Distance Vector (AODV) routing model (RFC 3561)
- IPv6 extensions support to add IPv6 extensions and options.  Two
  examples (fragmentation and loose routing) are available.
- NetAnim interface: Provides an interface to the Qt-based NetAnim
  animator, which supports static, point-to-point topology-based
  packet animations.
- New topology helpers have been introduced
  - PointToPointDumbbellHelper
  - PointToPointGridHelper
  - PointToPointStarHelper
  - CsmaStarHelper
- Equal-cost multipath for global routing: Enables quagga's equal
  cost multipath for Ipv4GlobalRouting, and adds an attribute that
  can enable it with random packet distribution policy across
  equal cost routes.
- Binding sockets to devices:  A method analogous to a SO_BINDTODEVICE
  socket option has been introduced to class Socket
- Object::DoStart: Users who need to complete their object setup at
  the start of a simulation can override this virtual method, perform their
  adhoc setup, and then, must chain up to their parent.
- Ipv4::IsDestinationAddress method added to support checks of whether a
  destination address should be accepted as one of the host's own
  addresses.
- UniformDiscPositionAllocator added; distributes uniformly the nodes
  within a disc of given radius.
- ChannelNumber attribute added to YansWifiPhy. Now it is possible to
  setup wifi channel using WifiPhyHelper::Set() method.
- WaypointMobilityModel provides a method to add mobility as a set of
  (time, position) pairs
- 802.11p WiFi standards
- UDP Client/Server application
- Support transactions in the SQLite output interface, making it usable for larger amounts of data

### API changes from ns-3.6

API changes for this release are documented in the file CHANGES.html.

### Bugs fixed

The following lists many of the bugs that were fixed since ns-3.6, in
many cases referencing the Bugzilla bug number

- bug 752: Object::DoStart is not executed for objects created at t > 0
- bug 767: Incorrect modulation for 802.11a modes
- bug 725: wifi fragmentation and RTS cannot be used at the same time
- bug 782: CreateTap () requires IP address in modes other than
  CONFIGURE_LOCAL.
- bug 769: Queue::GetTotalReceived{Bytes,Packets}() broken
- bug 738 ReceiveErrorModel called too late
- Fix NSC improper response to FIN
- Fixed bug in serialization of PbbAddressBlock.
- Fix bug 780 (problem in RoutingTableComputation with asymmetric links),
  while adding debugging methods to OLSR.
- bug 759: Ipv6 uses wrong outgoing interface.
- bug 770: IPv6 size calculation for unknown options is wrong.
- bug 771: Radvd does not set ttl value.
- Fix bug 606:  Arp depends on IP routing system
- pad out CSMA payloads to 46 bytes if needed
- Drop CSMA packets with CRC errors, rescan, dox tweaks
- Add FCS capability to CSMA
- Mesh:Dot11s: fixed airtime metric
- Get emu working again:  Add Dix/Llc option, add and use contextual
  realtime schedule ops, don't refcount realtime simulator impl
- bug 695 - DcfManager::UpdateBackoff () uses slow HighPrecision::Div()
- bug 674 - EIFS is not handled correctly in DcfManager::GetAccessGrantStart
- bug 739 -  OLSR: Strange HTime value in HELLO messages
- bug 746 -  UDP source address is not set to bound address
- bug 735 Update Olsr for local delivery
- bug 740 OLSR MprCompute () works wrong: fixed
- bug 729 Enable IPv6 over PPP.
- bug 645: fixes for opening stats file with OMNeT++
- bug 689: default energy detection and CCA thresholds are changed to be
  more realistic.
- bug 733: OLSR MPR Computation give incorrect result
- Mesh: HWMP: fixed proactive routes
- Mesh: fixed FLAME PATH_UPDATE procedure, fixed mesh.cc

### Known issues

ns-3 builds have been known to fail on the following platforms:
- gcc 3.3 and earlier
- optimized builds on gcc 3.4.4 and 3.4.5
- optimized builds on linux x86 gcc 4.0.x


Release 3.6
-----------

### Availability

This release is immediately available from:
http://www.nsnam.org/release/ns-allinone-3.6.tar.bz2

### Supported platforms

ns-3.6 has been tested on the following platforms:
- linux x86 gcc 4.4.1, 4.2, 4.1, and, 3.4.6.
- linux x86_64 gcc 4.4.0, 4.3.2, 4.2.3, 4.2.1, 4.1.3, 3.4.6
- MacOS X ppc and x86 (gcc 4.0.x and 4.2.x)
- cygwin gcc 3.4.4 (debug only), gcc 4.3.2 (debug and optimized)

Not all ns-3 options are available on all platforms; consult the
wiki for more information:
http://www.nsnam.org/wiki/Installation

### New user-visible features

- 802.11 models:
  - Add an implementation of the minstrel rate control algorithm
    (Duy Nguyen for gsoc)
  - AthstatsHelper: enables the wifi device to produce periodic
    reports similar to the ones generated by madwifi's
    athstats tool (Nicola Baldo)
  - 10MHz and 5MHz channel width supported by 802.11a model
    (Ramon Bauza and Kirill Andreev)
  - Channel switching support. YansWifiPhy can now switch among
    different channels (Ramon Bauza and Pavel Boyko)

- IPv6 models:
  - IPv6 interface;
  - IPv6 layer;
  - IPv6 raw socket;
  - Static IPv6 routing;
  - ICMPv6 layer;
  - Some ICMPv6 error messages (destination unreachable, ...);
  - Neighbor Discovery Protocol (NS/NA, RS/RA, redirection);
  - Ping6 application (send Echo request);
  - Radvd application (send RA);
  - Examples (ping6, simple-routing-ping6, radvd, radvd-two-prefix,
    icmpv6-redirect).

- Wireless Mesh Networking models:
  - General multi-interface mesh stack infrastructure (devices/mesh module).
  - IEEE 802.11s (Draft 3.0) model including Peering Management Protocol and HWMP.
  - Forwarding Layer for Meshing (FLAME) protocol.

- Nix-vector routing:
  - Ipv4NixVectorHelper
  - Examples (nix-simple, nms-p2p-nix)

- New Test Framework
  - Use test.py instead of ./waf check or ./waf --regression
  - Previous unit tests have been ported to new framework.
  - Examples are tested for run-ability.

- A new Flow Monitor module
  - To very easily measure flow metrics in a simulation
  - No need to use trace callbacks or parsing trace files

### API changes from ns-3.5

API changes for this release are documented in the file CHANGES.html.

### Known issues

ns-3.6 build is known to fail on the following platforms:
- gcc 3.3 and earlier
- optimized builds on gcc 3.4.4 and 3.4.5
- optimized builds on linux x86 gcc 4.0.x

Release 3.5
-----------

### Availability

This release is immediately available from:
http://www.nsnam.org/release/ns-allinone-3.5.tar.bz2

### Supported platforms

ns-3.5 has been tested on the following platforms:
- linux x86 gcc 4.2, 4.1, and, 3.4.6.
- linux x86_64 gcc 4.4.0, 4.3.2, 4.2.3, 4.2.1, 4.1.3, 3.4.6
- MacOS X ppc and x86 (gcc 4.0.x and 4.2.x)
- cygwin gcc 3.4.4 (debug only), gcc 4.3.2 (debug and optimized)
- mingw gcc 3.4.5 (debug only)

Not all ns-3 options are available on all platforms; consult the
wiki for more information:
http://www.nsnam.org/wiki/Installation

### New user-visible features

- 802.11 MAC:
  - EDCA multi-qos-class support (Mirko Banchi)
  - 802.11n initial support for A-MSDU frame aggregation (Mirko Banchi)
  - aarf-cd and cara rate control algorithms (Federico Maguolo)
- 802.11 PHY:
  - 802.11b PHY support (Gary Pei)
  - Nakagami propagation loss model (Timo Bingmann)
  - radiotap and prism headers for PCAP output (Nicola Baldo)
- RandomVariable:
  - GammaVariable and ErlangVariable (Timo Bingmann)
  - ZipfVariable (Francesco Malandrino)
- UdpEcho: allows setting the payload of echo packets
- Ipv4/Ipv4RoutingProtocol: allow fine-grained control over
  routing policy and arbitrary composition of routing protocols

### API changes from ns-3.4

API changes for this release are documented in the file CHANGES.html.  The
internal API and composition of the IPv4 stack underwent significant
refactoring in this release cycle.

### Known issues

ns-3 build is known to fail on the following platforms:
- gcc 3.3 and earlier
- optimized builds on gcc 3.4.4 and 3.4.5
- optimized builds on linux x86 gcc 4.0.x

### Future releases

Our next release, which is expected to happen in 2 to 4 months from now, will
feature the merging of some of our projects currently in development including
fuller IPv6 support, some smaller features such as a new Global ARP
package, and possibly a new Testing and Validation suite.

Release 3.4
-----------

### Availability

This release is immediately available from:
http://www.nsnam.org/release/ns-allinone-3.4.tar.bz2

### Supported platforms

ns-3.4 has been tested on the following platforms:
- linux x86 gcc 4.2, 4.1, and, 3.4.6.
- linux x86_64 gcc 4.3.2, 4.2.3, 4.2.1, 4.1.3, 3.4.6
- MacOS X ppc and x86
- cygwin gcc 3.4.4 (debug only)

Not all ns-3 options are available on all platforms; consult the
wiki for more information:
http://www.nsnam.org/wiki/Installation

### New user-visible features

- Wifi models: Timo Bingman contributed a ThreeLogDistance and a Nakagami
  propagation loss model based on the ns-2 models. Fabian Mauchle contributed
  multicast support.
- Object Name Service:  A facility allowing ns-3 Objects to be assigned
  names has been added.
- Tap Bridge:  A second option for integrating ns-3 with real-world hosts
  has been added.  This allows for real hosts to talk over ns-3 net devices
  and simulated networks.
- A new build option (ns-3-allinone) has been provided to make it easier for
  users to download and build commonly used ns-3 configurations.
- The ns-3 calendar queue scheduler has been ported to ns-3.
- XML support has been added to the ConfigStore.

### API changes from ns-3.3

API changes for this release are documented in the file CHANGES.html

### Known issues

ns-3 build is known to fail on the following platforms:
- gcc 3.3 and earlier
- optimized builds on gcc 3.4.4 and 3.4.5
- optimized builds on linux x86 gcc 4.0.x
- optimized builds on Ubuntu 8.10 alpha 5 x86 gcc4.3.2
- MinGW

The IPv4 API defined in src/node/ipv4.h is expected to undergo major changes
in preparation of the merge of the IPv6 API and implementation.

### Future releases

Our next release, which is expected to happen in 2 to 4 months from now, will
feature the merging of some of our projects currently in development including
fuller IPv6 support, and IPv4 and routing protocol refactoring, and some smaller
features such as a new Global ARP package and possibly a new Testing and Validation
suite,

Release 3.3
-----------

### Availability

This release is immediately available from:
http://www.nsnam.org/release/ns-3.3.tar.bz2

### Supported platforms

ns-3.3 has been tested on the following platforms:
- linux x86 gcc 4.2, 4.1, and, 3.4.6.
- linux x86_64 gcc 4.3.2, 4.2.3, 4.2.1, 4.1.3, 3.4.6
- MacOS X ppc and x86
- cygwin gcc 3.4.4 (debug only)

Not all ns-3 options are available on all platforms; consult the
wiki for more information:
http://www.nsnam.org/wiki/Installation

### New user-visible features

- Emulated Net Device
  A new net device has been added as enabling technology for ns-3 emulation
  scenarios.  See src/devices/emu and examples/emu-udp-echo.cc for details.
- ICMP Support
  Support for several ICMP messages has been added to ns-3.  See
  src/internet-stack/icmpv4.h for details.
- IPv6 Address Support
  New clases to support IPv6 addresses has been added to the system.  This
  is enabling technology for fuller IPv6 support scheduled for ns-3.4.
- A flow-id tag has been added to the contributed code section
- Star topologies can be created from the topology helper functions
- The global routing code has been made dynamic (not just limited to
  (pre-simulation computation) and supports stub network interfaces and
  bridge net devices
- A MatchContainer has been added to the configuration subsystem

### API changes from ns-3.2

API changes for this release are documented in the file CHANGES.html

### Known issues

ns-3 build is known to fail on the following platforms:
- gcc 3.3 and earlier
- optimized builds on gcc 3.4.4 and 3.4.5
- optimized builds on linux x86 gcc 4.0.x
- optimized builds on Ubuntu 8.10 alpha 5 x86 gcc4.3.2
- MinGW

The IPv4 API defined in src/node/ipv4.h is expected to undergo major changes
in preparation of the merge of the IPv6 API and implementation.

### Future releases

Our next release, which is expected to happen in 2 to 4 months from now, will
feature the merging of some of our projects currently in development including
fuller IPv6 support, and IPv4 and routing protocol refactoring, and some smaller
features such as an object naming facility and a new Global ARP package.

Release 3.2
-----------

### Availability

This release is immediately available from:
http://www.nsnam.org/release/ns-3.2.tar.bz2

### Supported platforms

ns-3.2 has been tested on the following platforms:
- linux x86 gcc 4.2, 4.1, and, 3.4.6.
- linux x86_64 gcc 4.3.2, 4.2.3, 4.2.1, 4.1.3, 3.4.6
- MacOS X ppc and x86
- cygwin gcc 3.4.4 (debug only)

Not all ns-3 options are available on all platforms; consult the
wiki for more information:
http://www.nsnam.org/wiki/Installation

### New user-visible features

- Learning bridge (IEEE 802.1D)
  It is now possible to bridge together multiple layer 2 devices to
  create larger layer 2 networks. The Wifi and Csma models support
  this new mode of operation. (contributed by Gustavo Carneiro)

- Python bindings
  It is now possible to write simulation scripts in python using our
  python bindings (contributed by Gustavo Carneiro).

- Real-time simulator
  It is now possible to run simulations synchronized on the real-world
  wall-clock time (contributed by Craig Dowell).

- Network Simulation Cradle
  It is now possible to use the Network Simulation Cradle
  (http://www.wand.net.nz/~stj2/nsc/) in ns-3 and run simulations
  using various versions of kernel TCP network stacks. (contributed
  by Florian Westphal as part of his Google Summer of Code work)

- A statistics framework
  Joseph Kopena contributed a statistics framework which can be used
  keep track of simulation data in persistent storage across multiple
  runs (database and ascii file backends are available).
  More information on the wiki:
  http://www.nsnam.org/wiki/Statistical_Framework_for_Network_Simulation

### API changes from ns-3.1

API changes for this release are documented in the file CHANGES.html

### Known issues

ns-3 build is known to fail on the following platforms:
- gcc 3.3 and earlier
- optimized builds on gcc 3.4.4 and 3.4.5
- optimized builds on linux x86 gcc 4.0.x
- optimized builds on Ubuntu 8.10 alpha 5 x86 gcc4.3.2
- MinGW

The IPv4 API defined in src/node/ipv4.h is expected to undergo major changes
in preparation of the merge of the IPv6 API and implementation.

### Future releases

Our next release, which is expected to happen in 2 to 4 months from now, will
feature the merging of some of our projects currently in development: IPv6,
emulation, and synchronous posix sockets.

Release 3.1 (2008/06/30)
------------------------

### The first ns-3 stable release

This release is immediately available from:
http://www.nsnam.org/release/ns-3.1.tar.bz2

We dedicate this initial ns-3 release to our late contributor and friend,
Federico Maguolo.

### What is ns-3 ?

ns-3 is a new discrete-event network simulator designed for supporting network
research and education. ns-3 features a solid, well documented C++ core and
models for TCP/IP (IPv4), several link types including WiFi, and mobility
models.

ns-3 is an open source project released under the GNU GPLv2 license which
allows anyone to use ns-3 without having to pay any license fee or royalties.
ns-3 is actively seeking new contributors to extend the range of supported
models and/or to maintain existing models.

### Where to get more information about ns-3

All the ns-3 documentation, is accessible from the ns-3 website:
http://www.nsnam.org

Including, tutorials:
http://www.nsnam.org/tutorials.html

### Supported platforms

ns-3 is regularly tested on the following platforms:
- linux x86 gcc 4.2, 4.1, and, 3.4.6.
- linux x86_64 gcc 4.1.3, 4.2.1, 3.4.6
- MacOS X ppc and x86
- cygwin gcc 3.4.4 (debug only)

### Known issues

ns-3 is known to fail on the following platforms:
- gcc 3.3 and earlier
- optimized builds on gcc 3.4.4 and 3.4.5
- optimized builds on linux x86 gcc 4.0.x

The IPv4 API defined in src/node/ipv4.h is expected to undergo major changes
in preparation of the merge of the IPv6 API and implementation.

### Future releases

Our next release, which is expected to happen in 2 to 4 months from now, will
feature the merging of some of our projects currently in development: python
scripting, IPv6, emulation, a statistics framework and synchronous posix
sockets.

Release 3.0.13 (2008/06/02)
---------------------------

- point to point links generate ppp pcap traces
- point to point links support asymmetrical data rates.
- generate doxygen documentation for all attributes and trace sources
- add ConfigStore and GtkConfigStore to contrib module
- socket API now support tx and rx buffers: implemented for UDP and TCP
- ARP cache now supports per-entry pending queues
- lots of bugfixes and implementation and API cleanups

Warning: among API changes in this release, Application::Start and
Application::Stop now interprets the time argument as a relative
instead of absolute simulation time, to align with how Simulator::Schedule
behaves.  Any code that calls these APIs in the middle of the simulation
will need to be adapted.

The API of Simulator::StopAt (time) has also changed.  Now it is
called Simulator::Stop (time), and takes a relative time, instead of
absolute.

Release 3.0.12 (2008/04/07)
---------------------------

- Add Attribute support to the TypeId metadata system and add
  attribute support to all in-tree models
- Add a mid-level helper API to build simulation topologies
  and perform simple ascii and pcap link-level tracing.
- Large amount of structural changes to accommodate the needs
  of the upcoming python bindings
- new rate control algorithm (RRAA) and a jakes propagation model
  in the 802.11 model (Federico Maguolo).
- regression test framework added
- TCP delayed acknowledgements and multitasking server

Release 3.0.11 (2008/02/15)
---------------------------

- Initial port of GTNetS TCP implementation (initial version
  that does not support multitasking or delayed acknowledgments yet,
  but supports a reliable stream service)
- Changes to the ns-3 object model to create a TypeId-based
  metadata system
- lots of bug fixes throughout the system
- tutorial updates

Release 3.0.10 (2008/01/15)
---------------------------

- Add tutorial document content;
- Valgrind option for "waf" tool;
- Doxygen organization changes.

Release 3.0.9 (2007/12/15)
--------------------------

- A 802.11 model ported from Yans. This model supports:
  - a rather extensive PHY model
  - log-distance and friis propagation model
  - a simple set of rate control algorithm
    (ARF, Ideal, AARF, constant-rate)
  - adhoc and infrastructure mode (beacon+assoc)
- Use smart pointer for Packet in the APIs
- A new contrib directory with helper classes:
  EventGarbageCollector, Gnuplot
- Tracing support for Applications
- many bugs fixed

Release 3.0.8 (2007/11/15)
--------------------------

- A simple error model
- Source files for ns-3 tutorial

Release 3.0.7 (2007/10/15)
--------------------------

- OLSR routing protocol
- A timer class
- Additional mobility models (random waypoint, random 2D walk)
- A mobility visualization tool

Release 3.0.6 (2007/09/15)
--------------------------

- Static multicast IPv4 routing
- Logging overhaul (NS_LOG macros)
- Refactoring of tracing subsystem
- Tutorial document started

Release 3.0.5 (2007/08/15)
--------------------------

- Refactoring to support win32-based unix environments (Cygwin, mingw)
- "Packet socket" for allowing applications to access NetDevices directly
- Generalized, polymorphic Address class
- Add CSMA NetDevice model (from Emmanuelle Laprise)
- Modularize IPv4 routing support (from Gustavo Carneiro)
- Add mobility framework and basic mobility models
- Global unicast centralized routing

Release 3.0.4 (2007/07/15)
--------------------------

- Enable waf as the default build system.
- Per-packet metadata:  a system to track which headers and trailers
  are added to a packet
- Simplifications to point-to-point devices and channel

Release 3.0.3 (2007/06/15)
--------------------------

- Enable Waf for release tarballs: users can now build ns-3
  with the "waf" tool. See doc/build-waf.txt.
- Add support for variable time precision: it is now possible
  to run a simulation with an accuracy which is higher or lower
  than a nanosecond: seconds, milliseconds, microseconds,
  femtoseconds and picoseconds are supported.
- Optimize and rework the COM framework, solidify the component
  manager
- Many small API cleanups

Release 3.0.2 (2007/05/18)
--------------------------

- Implement a new memory management infrastructure based
  on reference counting and smart pointers (the latter being
  optional)
- Implement a COM-like framework with support for QueryInterface
  to provide object extensibility
- Add support for a BSD-style socket API for user applications

Release 3.0.1 (2007/03/31)
--------------------------

- First public release; not yet pre-alpha.
- Simple UDP-based simulation script (examples/simple-p2p.cc)
