DPDK NetDevice
--------------
.. include:: replace.txt
.. highlight:: cpp

.. heading hierarchy:
   ------------- Chapter
   ************* Section (#.#)
   ============= Subsection (#.#.#)
   ############# Paragraph (no number)


Data Plane Development Kit (DPDK) is a library hosted by The Linux Foundation
to accelerate packet processing workloads (https://www.dpdk.org/).

The ``DpdkNetDevice`` class provides the implementation of a network device which uses DPDK’s fast packet processing abilities and bypasses the kernel. This class is included in the ``src/fd-net-device model``. The ``DpdkNetDevice`` class inherits the ``FdNetDevice`` class and overrides the functions which are required by |ns3| to interact with DPDK environment.

The ``DpdkNetDevice`` for |ns3| [Patel2019]_ was developed by Harsh Patel,
Hrishikesh Hiraskar and Mohit P. Tahiliani. They were supported by Intel
Technology India Pvt. Ltd., Bangalore for this work.

.. [Patel2019] Harsh Patel, Hrishikesh Hiraskar, Mohit P. Tahiliani, "Extending Network Emulation Support in ns-3 using DPDK", Proceedings of the 2019 Workshop on ns-3, ACM, Pages 17-24, (https://dl.acm.org/doi/abs/10.1145/3321349.3321358)

Model Description
*****************

``DpdkNetDevice`` is a network device which provides network emulation capabilities i.e. to allow simulated nodes to interact with real hosts and vice versa. The main feature of the ``DpdkNetDevice`` is that is uses the Environment Abstraction Layer (EAL) provided by DPDK to perform fast packet processing. EAL hides the device specific attributes from the applications and provides an interface via which the applications can interact directly with the Network Interface Card (NIC). This allows |ns3| to send/receive packets directly to/from the NIC without the kernel involvement.

Design
======

``DpdkNetDevice`` is designed to act as an interface between |ns3| and DPDK environment. There are 3 main phases in the life cycle of ``DpdkNetDevice``:

* Initialization
* Packet Transfer - Read and Write
* Termination

Initialization
##############

``DpdkNetDeviceHelper`` model is responsible for the initialization of ``DpdkNetDevice``. After this, the EAL is initialized, a memory pool is allocated, access to the Ethernet port is obtained and it is initialized, reception (Rx) and transmission (Tx) queues are set up on the port, Rx and Tx buffers are set up and LaunchCore method is called which will launch the ``HandleRx`` method to handle reading of packets in burst.

Packet Transfer
###############

DPDK interacts with packet in the form of mbuf, a data structure provided by it, while |ns3| interacts with packets in the form of raw buffer. The packet transfer functions take care of converting DPDK mbufs to |ns3| buffers. The functions are read and write.

* Read: ``HandleRx`` method takes care of reading the packets from NIC and transferring them to |ns3| Internet Stack. This function is called by ``LaunchCore`` method which is launched during initialization. It continuously polls the NIC using DPDK API for packets to read. It reads the mbuf packets in burst from NIC Rx ring, which are placed into Rx buffer upon read. For each mbuf packet in Rx buffer, it then converts it to |ns3| raw buffer and then forwards the packet to |ns3| Internet Stack.
* Write: ``Write`` method handles transmission of packets. |ns3| provides this packet in the form of a buffer, which is converted to packet mbuf and then placed in the Tx buffer. These packets are then transferred to NIC Tx ring when the Tx buffer is full, from where they will be transmitted by the NIC. However, there might be a scenario where there are not enough packets to fill the Tx buffer. This will lead to stale packet mbufs in buffer. In such cases, the ``Write`` function schedules a manual flush of these stale packet mbufs to NIC Tx ring, which will occur upon a certain timeout period. The default value of this timeout is set to ``2 ms``.

Termination
###########

When |ns3| is done using ``DpdkNetDevice``, the ``DpdkNetDevice`` will stop polling for Rx, free the allocated mbuf packets and then the mbuf pool. Lastly, it will stop the Ethernet device and close the port.


Scope and Limitations
=====================

The current implementation supports only one NIC to be bound to DPDK with single  Rx and Tx on the NIC. This can be extended to support multiple NICs and multiple Rx/Tx queues simultaneously. Currently there is no support for Jumbo frames, which can be added. Offloading, scheduling features can also be added. Flow control and support for qdisc can be added to provide a more extensive model for network testing.


DPDK Installation
*****************

This section contains information on downloading DPDK source code and setting up DPDK for ``DpdkNetDevice`` to work.

Is my NIC supported by DPDK?
============================

Check `Supported Devices <https://core.dpdk.org/supported/>`_.

Not supported? Use Virtual Machine instead
==========================================

Install `Oracle VM VirtualBox <https://www.virtualbox.org/>`_. Create a new VM and install Ubuntu on it. Open settings, create a network adapter with following configuration:

* Attached to: Bridged Adapter
* Name: The host network device you want to use
* In Advanced
    * Adapter Type: Intel PRO/1000 MT Server (82545EM) or any other DPDK supported NIC
    * Promiscuous Mode: Allow All
    * Select Cable Connected


Then rest of the steps are same as follows.

DPDK can be installed in 2 ways:

* Install DPDK on Ubuntu
* Compile DPDK from source

Install DPDK on Ubuntu
======================

To install DPDK on Ubuntu, run the following command:

.. sourcecode:: text

 apt-get install dpdk dpdk-dev libdpdk-dev dpdk-igb-uio-dkms

Ubuntu 20.04 has packaged DPDK v19.11 LTS which is tested with this module and DpdkNetDevice will only be enabled if this version is available.

Compile from Source
===================

To compile DPDK from source, you need to perform the following 4 steps:

1. Download the source
######################

Visit the `DPDK Downloads <https://core.dpdk.org/download/>`_ page to download the latest stable source. (This module has been tested with version 19.11 LTS and DpdkNetDevice will only be enabled if this version is available.)

2. Configure DPDK as a shared library
#####################################

In the DPDK directory, edit the ``config/common_base`` file to change the following line to compile DPDK as a shared library:

.. sourcecode:: text

 # Compile to share library
 CONFIG_RTE_BUILD_SHARED_LIB=y

3. Install the source
#####################

Refer to `Installation <https://doc.dpdk.org/guides/linux_gsg/build_dpdk.html>`_ for detailed instructions.

For a 64 bit linux machine with gcc, run:

.. sourcecode:: text

 make install T=x86_64-native-linuxapp-gcc DESTDIR=install

4. Export DPDK Environment variables
####################################

Export the following environment variables:

* RTE_SDK as the your DPDK source folder.
* RTE_TARGET as the build target directory.


For example:

.. sourcecode:: text

 export RTE_SDK=/home/username/dpdk/dpdk-stable-19.11.1
 export RTE_TARGET=x86_64-native-linuxapp-gcc

(Note: In case DPDK is moved, ns-3 needs to be reconfigured using ``./ns3 configure [options]``)

It is advisable that you export these variables in ``.bashrc`` or similar for reusability.

Load DPDK Drivers to kernel
===========================

Execute the following:

.. sourcecode:: text

 sudo modprobe uio_pci_generic
 sudo modprobe uio
 sudo modprobe vfio-pci

 sudo modprobe igb_uio # for ubuntu package
 # OR
 sudo insmod $RTE_SDK/$RTE_TARGET/kmod/igb_uio.ko # for dpdk source


These should be done every time you reboot your system.

Configure hugepages
===================

Refer `System Requirements <https://doc.dpdk.org/guides/linux_gsg/sys_reqs.html>`_ for detailed instructions.

To allocate hugepages at runtime, write a value such as '256' to the following:

.. sourcecode:: text

 echo 256 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages

To allocate hugepages at boot time, edit ``/etc/default/grub``, and following to ``GRUB_CMDLINE_LINUX_DEFAULT``:

.. sourcecode:: text

 hugepages=256

We suggest minimum of number of ``256`` to run our applications. (This is to test an application run at 1 Gbps on a 1 Gbps NIC.) You can use any number of hugepages based on your system capacity and application requirements.

Then update the grub configurations using:

.. sourcecode:: text

 sudo update-grub

OR

.. sourcecode:: text

 sudo update-grub2


You will need to reboot your system in order to see these changes.

To check allocation of hugepages, run:

.. sourcecode:: text

 cat /proc/meminfo | grep HugePages


You will see the number of hugepages allocated, they should be equal to the number you used above.

Once the hugepage memory is reserved (at either runtime or boot time), 
to make the memory available for DPDK use, perform the following steps:

.. sourcecode:: text

 sudo mkdir /mnt/huge
 sudo mount -t hugetlbfs nodev /mnt/huge

The mount point can be made permanent across reboots, by adding the following line to the ``/etc/fstab`` file:

.. sourcecode:: text

 nodev /mnt/huge hugetlbfs defaults 0 0


Usage
*****

The status of DPDK support is shown in the output of ``./ns3 configure``.  If
it is found, a user should see:

.. sourcecode:: text

  DPDK NetDevice                : enabled

``DpdkNetDeviceHelper`` class supports the configuration of ``DpdkNetDevice``.

.. sourcecode:: text

 +----------------------+
 |         host 1       |
 +----------------------+
 |   ns-3 simulation    |
 +----------------------+
 |       ns-3 Node      |
 |  +----------------+  |
 |  |    ns-3 TCP    |  |
 |  +----------------+  |
 |  |    ns-3 IP     |  |
 |  +----------------+  |
 |  |  DpdkNetDevice |  |
 |  |    10.1.1.1    |  |
 |  +----------------+  |
 |  |   raw socket   |  |
 |--+----------------+--|
 |       | eth0 |       |
 +-------+------+-------+

         10.1.1.11

             |
             +-------------- ( Internet ) ----


Initialization of DPDK driver requires initialization of EAL. EAL requires PMD (Poll Mode Driver) Library for using NIC. DPDK supports multiple Poll Mode Drivers and you can use one that works for your NIC. PMD Library can be set via ``DpdkNetDeviceHelper::SetPmdLibrary``, as follows:

.. sourcecode:: text

 DpdkNetDeviceHelper* dpdk = new DpdkNetDeviceHelper ();
 dpdk->SetPmdLibrary("librte_pmd_e1000.so");

Also, NIC should be bound to DPDK Driver in order to be used with EAL. The default driver used is ``uio_pci_generic`` which supports most of the NICs. You can change it using ``DpdkNetDeviceHelper::SetDpdkDriver``, as follows:

.. sourcecode:: text

 DpdkNetDeviceHelper* dpdk = new DpdkNetDeviceHelper ();
 dpdk->SetDpdkDriver("igb_uio");

Attributes
==========

The ``DpdkNetDevice`` provides a number of attributes:

* ``TxTimeout`` - The time to wait before transmitting burst from Tx Buffer (in us). (default - ``2000``) This attribute is only used to flush out buffer in case it is not filled. This attribute can be decrease for low data rate traffic. For high data rate traffic, this attribute needs no change.
* ``MaxRxBurst`` - Size of Rx Burst. (default - ``64``) This attribute can be increased for higher data rates.
* ``MaxTxBurst`` - Size of Tx Burst. (default - ``64``) This attribute can be increased for higher data rates.
* ``MempoolCacheSize`` - Size of mempool cache. (default - ``256``) This attribute can be increased for higher data rates.
* ``NbRxDesc`` - Number of Rx descriptors. (default - ``1024``) This attribute can be increased for higher data rates.
* ``NbTxDesc`` - Number of Tx descriptors. (default - ``1024``) This attribute can be increased for higher data rates.

Note: Default values work well with 1Gbps traffic.

Output
======

As ``DpdkNetDevice`` is inherited from ``FdNetDevice``, all the output methods provided by ``FdNetDevice`` can be used directly.

Examples
========

The following examples are provided:

* ``fd-emu-ping.cc``: This example can be configured to use the ``DpdkNetDevice`` to send ICMP traffic bypassing the kernel over a real channel.
* ``fd-emu-onoff.cc``: This example can be configured to measure the throughput of the ``DpdkNetDevice`` by sending traffic from the simulated node to a real device using the ``ns3::OnOffApplication`` while leveraging DPDK’s fast packet processing abilities. This is achieved by saturating the channel with TCP/UDP traffic.
