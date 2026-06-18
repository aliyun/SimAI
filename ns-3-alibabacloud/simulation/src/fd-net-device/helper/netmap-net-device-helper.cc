/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017 Universita' degli Studi di Napoli Federico II
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Pasquale Imputato <p.imputato@gmail.com>
 */

#include "netmap-net-device-helper.h"
#include "encode-decode.h"

#include "ns3/abort.h"
#include "ns3/config.h"
#include "ns3/fd-net-device.h"
#include "ns3/log.h"
#include "ns3/names.h"
#include "ns3/object-factory.h"
#include "ns3/packet.h"
#include "ns3/simulator.h"
#include "ns3/trace-helper.h"
#include "ns3/netmap-net-device.h"
#include "ns3/uinteger.h"

#include <arpa/inet.h>
#include <errno.h>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <net/ethernet.h>
#include <net/if.h>
#include <netinet/in.h>

#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <string>

#include <net/netmap_user.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NetmapNetDeviceHelper");

#define EMU_MAGIC 65867

NetmapNetDeviceHelper::NetmapNetDeviceHelper ()
{
  m_deviceName = "undefined";
  SetTypeId ("ns3::NetmapNetDevice");
}

std::string
NetmapNetDeviceHelper::GetDeviceName (void)
{
  return m_deviceName;
}

void
NetmapNetDeviceHelper::SetDeviceName (std::string deviceName)
{
  m_deviceName = deviceName;
}

Ptr<NetDevice>
NetmapNetDeviceHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<NetDevice> d = FdNetDeviceHelper::InstallPriv (node);
  Ptr<FdNetDevice> device = d->GetObject<FdNetDevice> ();

  SetDeviceAttributes (device);

  int fd = CreateFileDescriptor ();
  Ptr<NetmapNetDevice> netmapDevice = DynamicCast<NetmapNetDevice> (device);
  SwitchInNetmapMode (fd, netmapDevice);

  // Aggregate NetDeviceQueueInterface object
  Ptr<NetDeviceQueueInterface> ndqi = CreateObjectWithAttributes<NetDeviceQueueInterface> (
      "TxQueuesType", TypeIdValue (NetDeviceQueueLock::GetTypeId ()),
      "NTxQueues", UintegerValue (1));

  device->AggregateObject (ndqi);
  netmapDevice->SetNetDeviceQueue (ndqi->GetTxQueue (0));

  return device;
}

void
NetmapNetDeviceHelper::SetDeviceAttributes (Ptr<FdNetDevice> device) const
{

  if (m_deviceName == "undefined")
    {
      NS_FATAL_ERROR ("NetmapNetDeviceHelper::SetFileDescriptor (): m_deviceName is not set");
    }

  //
  // Call out to a separate process running as suid root in order to get a raw
  // socket.  We do this to avoid having the entire simulation running as root.
  //
  int fd = socket (PF_INET, SOCK_DGRAM, 0);

  //
  // Figure out which interface index corresponds to the device name in the corresponding attribute.
  //
  struct ifreq ifr;
  bzero (&ifr, sizeof(ifr));
  strncpy ((char *)ifr.ifr_name, m_deviceName.c_str (), IFNAMSIZ - 1);

  NS_LOG_LOGIC ("Getting interface index");
  int32_t rc = ioctl (fd, SIOCGIFINDEX, &ifr);
  if (rc == -1)
    {
      NS_FATAL_ERROR ("NetmapNetDeviceHelper::SetFileDescriptor (): Can't get interface index");
    }

  rc = ioctl (fd, SIOCGIFFLAGS, &ifr);
  if (rc == -1)
    {
      NS_FATAL_ERROR ("NetmapNetDeviceHelper::SetFileDescriptor (): Can't get interface flags");
    }

  //
  // This device only works if the underlying interface is up in promiscuous
  // mode.  We could have turned it on in the socket creator, but the situation
  // is that we expect these devices to be used in conjunction with virtual
  // machines with connected host-only (simulated) networks, or in a testbed.
  // There is a lot of setup and configuration happening outside of this one
  // issue, and we expect that configuration to include choosing a valid
  // interface (e.g, "ath1"), ensuring that the device supports promiscuous
  // mode, and placing it in promiscuous mode.  We just make sure of the
  // end result.
  //
  if ((ifr.ifr_flags & IFF_PROMISC) == 0)
    {
      NS_FATAL_ERROR ("NetmapNetDeviceHelper::SetFileDescriptor (): "
                      << m_deviceName.c_str ()
                      << " is not in promiscuous mode. Please config the interface in promiscuous "
                      "mode before to run the simulation.");
    }

  if ((ifr.ifr_flags & IFF_BROADCAST) != IFF_BROADCAST)
    {
      // We default m_isBroadcast to true but turn it off here if not
      // supported, because in the common case, overlying IP code will
      // assert during configuration time if this is false, before this
      // method has a chance to set it during runtime
      device->SetIsBroadcast (false);
    }

  if ((ifr.ifr_flags & IFF_MULTICAST) == IFF_MULTICAST)
    {
      // This one is OK to enable at runtime
      device->SetIsMulticast (true);
    }

  // Set the MTU of the device to the mtu of the associated network interface
//   struct ifreq ifr2;
//
//   bzero (&ifr2, sizeof (ifr2));
//   strcpy (ifr2.ifr_name, m_deviceName.c_str ());

//   int32_t mtufd = socket (PF_INET, SOCK_DGRAM, IPPROTO_IP);

  rc = ioctl (fd, SIOCGIFMTU, &ifr);
  if (rc == -1)
    {
      NS_FATAL_ERROR ("FdNetDevice::SetFileDescriptor (): Can't ioctl SIOCGIFMTU");
    }

  NS_LOG_DEBUG ("Device MTU " << ifr.ifr_mtu);
  device->SetMtu (ifr.ifr_mtu);

  close (fd);

}

int
NetmapNetDeviceHelper::CreateFileDescriptor (void) const
{
  NS_LOG_FUNCTION (this);

  //
  // We want to create a raw socket for our net device.  Unfortunately for us
  // you have to have root privileges to do that.  Instead of running the
  // entire simulation as root, we decided to make a small program who's whole
  // reason for being is to run as suid root and create a raw socket.  We're
  // going to fork and exec that program soon, but we need to have a socket
  // to talk to it with.  So we create a local interprocess (Unix) socket
  // for that purpose.
  //
  int sock = socket (PF_UNIX, SOCK_DGRAM, 0);
  if (sock == -1)
    {
      NS_FATAL_ERROR ("NetmapNetDeviceHelper::CreateFileDescriptor(): Unix socket creation error, errno = " << strerror (errno));
    }

  //
  // Bind to that socket and let the kernel allocate an endpoint
  //
  struct sockaddr_un un;
  memset (&un, 0, sizeof (un));
  un.sun_family = AF_UNIX;
  int status = bind (sock, (struct sockaddr*)&un, sizeof (sa_family_t));
  if (status == -1)
    {
      NS_FATAL_ERROR ("NetmapNetDeviceHelper::CreateFileDescriptor(): Could not bind(): errno = " << strerror (errno));
    }

  NS_LOG_INFO ("Created Unix socket");
  NS_LOG_INFO ("sun_family = " << un.sun_family);
  NS_LOG_INFO ("sun_path = " << un.sun_path);

  //
  // We have a socket here, but we want to get it there -- to the program we're
  // going to exec.  What we'll do is to do a getsockname and then encode the
  // resulting address information as a string, and then send the string to the
  // program as an argument.  So we need to get the sock name.
  //
  socklen_t len = sizeof (un);
  status = getsockname (sock, (struct sockaddr*)&un, &len);
  if (status == -1)
    {
      NS_FATAL_ERROR ("NetmapNetDeviceHelper::CreateFileDescriptor(): Could not getsockname(): errno = " << strerror (errno));
    }

  //
  // Now encode that socket name (family and path) as a string of hex digits
  //
  std::string path = BufferToString ((uint8_t *)&un, len);
  NS_LOG_INFO ("Encoded Unix socket as \"" << path << "\"");
  //
  // Fork and exec the process to create our socket.  If we're us (the parent)
  // we wait for the child (the socket creator) to complete and read the
  // socket it created using the ancillary data mechanism.
  //
  // Tom Goff reports the possiblility of a deadlock when trying to acquire the
  // python GIL here.  He says that this might be due to trying to access Python
  // objects after fork() without calling PyOS_AfterFork() to properly reset
  // Python state (including the GIL).  There is no code to cause the problem
  // here in emu, but this was visible in similar code in tap-bridge.
  //
  pid_t pid = ::fork ();
  if (pid == 0)
    {
      NS_LOG_DEBUG ("Child process");

      //
      // build a command line argument from the encoded endpoint string that
      // the socket creation process will use to figure out how to respond to
      // the (now) parent process.
      //
      std::ostringstream oss;

      oss << "-p" << path;



      NS_LOG_INFO ("Parameters set to \"" << oss.str () << "\"");

      //
      // Execute the socket creation process image.
      //
      status = ::execlp (NETMAP_DEV_CREATOR,
                         NETMAP_DEV_CREATOR,                            // argv[0] (filename)
                         oss.str ().c_str (),                           // argv[1] (-p<path?
                         (char *)NULL);

      //
      // If the execlp successfully completes, it never returns.  If it returns it failed or the OS is
      // broken.  In either case, we bail.
      //
      NS_FATAL_ERROR ("NetmapNetDeviceHelper::CreateFileDescriptor(): Back from execlp(), status = " <<
                      status << ", errno = " << ::strerror (errno));
    }
  else
    {
      NS_LOG_DEBUG ("Parent process");
      //
      // We're the process running the emu net device.  We need to wait for the
      // socket creator process to finish its job.
      //
      int st;
      pid_t waited = waitpid (pid, &st, 0);
      if (waited == -1)
        {
          NS_FATAL_ERROR ("NetmapNetDeviceHelper::CreateFileDescriptor(): waitpid() fails, errno = " << strerror (errno));
        }
      NS_ASSERT_MSG (pid == waited, "NetmapNetDeviceHelper::CreateFileDescriptor(): pid mismatch");

      //
      // Check to see if the socket creator exited normally and then take a
      // look at the exit code.  If it bailed, so should we.  If it didn't
      // even exit normally, we bail too.
      //
      if (WIFEXITED (st))
        {
          int exitStatus = WEXITSTATUS (st);
          if (exitStatus != 0)
            {
              NS_FATAL_ERROR ("NetmapNetDeviceHelper::CreateFileDescriptor(): socket creator exited normally with status " << exitStatus);
            }
        }
      else
        {
          NS_FATAL_ERROR ("NetmapNetDeviceHelper::CreateFileDescriptor(): socket creator exited abnormally");
        }

      //
      // At this point, the socket creator has run successfully and should
      // have created our raw socket and sent it back to the socket address
      // we provided.  Our socket should be waiting on the Unix socket.  We've
      // got to do a bunch of grunto work to get at it, though.
      //
      // The struct iovec below is part of a scatter-gather list.  It describes a
      // buffer.  In this case, it describes a buffer (an integer) that will
      // get the data that comes back from the socket creator process.  It will
      // be a magic number that we use as a consistency/sanity check.
      //
      struct iovec iov;
      uint32_t magic;
      iov.iov_base = &magic;
      iov.iov_len = sizeof(magic);

      //
      // The CMSG macros you'll see below are used to create and access control
      // messages (which is another name for ancillary data).  The ancillary
      // data is made up of pairs of struct cmsghdr structures and associated
      // data arrays.
      //
      // First, we're going to allocate a buffer on the stack to receive our
      // data array (that contains the socket).  Sometimes you'll see this called
      // an "ancillary element" but the msghdr uses the control message terminology
      // so we call it "control."
      //
      size_t msg_size = sizeof(int);
      char control[CMSG_SPACE (msg_size)];

      //
      // There is a msghdr that is used to minimize the number of parameters
      // passed to recvmsg (which we will use to receive our ancillary data).
      // This structure uses terminology corresponding to control messages, so
      // you'll see msg_control, which is the pointer to the ancillary data and
      // controller which is the size of the ancillary data array.
      //
      // So, initialize the message header that describes the ancillary/control
      // data we expect to receive and point it to buffer.
      //
      struct msghdr msg;
      msg.msg_name = 0;
      msg.msg_namelen = 0;
      msg.msg_iov = &iov;
      msg.msg_iovlen = 1;
      msg.msg_control = control;
      msg.msg_controllen = sizeof (control);
      msg.msg_flags = 0;

      //
      // Now we can actually receive the interesting bits from the socket
      // creator process.
      //
      ssize_t bytesRead = recvmsg (sock, &msg, 0);
      if (bytesRead != sizeof(int))
        {
          NS_FATAL_ERROR ("NetmapNetDeviceHelper::CreateFileDescriptor(): Wrong byte count from socket creator");
        }

      //
      // There may be a number of message headers/ancillary data arrays coming in.
      // Let's look for the one with a type SCM_RIGHTS which indicates it' the
      // one we're interested in.
      //
      struct cmsghdr *cmsg;
      for (cmsg = CMSG_FIRSTHDR (&msg); cmsg != NULL; cmsg = CMSG_NXTHDR (&msg, cmsg))
        {
          if (cmsg->cmsg_level == SOL_SOCKET
              && cmsg->cmsg_type == SCM_RIGHTS)
            {
              //
              // This is the type of message we want.  Check to see if the magic
              // number is correct and then pull out the socket we care about if
              // it matches
              //
              if (magic == EMU_MAGIC)
                {
                  NS_LOG_INFO ("Got SCM_RIGHTS with correct magic " << magic);
                  int *rawSocket = (int*)CMSG_DATA (cmsg);
                  NS_LOG_INFO ("Got the socket from the socket creator = " << *rawSocket);
                  return *rawSocket;
                }
              else
                {
                  NS_LOG_INFO ("Got SCM_RIGHTS, but with bad magic " << magic);
                }
            }
        }
      NS_FATAL_ERROR ("Did not get the raw socket from the socket creator");
    }
}

void
NetmapNetDeviceHelper::SwitchInNetmapMode (int fd, Ptr<NetmapNetDevice> device) const
{
  NS_LOG_FUNCTION (this << fd << device);
  NS_ASSERT (device);

  if (m_deviceName == "undefined")
    {
      NS_FATAL_ERROR ("NetmapNetDevice: m_deviceName is not set");
    }

  if (fd == -1)
    {
      NS_FATAL_ERROR ("NetmapNetDevice: fd is not set");
    }

  struct nmreq nmr;
  memset (&nmr, 0, sizeof (nmr));

  nmr.nr_version = NETMAP_API;

  // setting the interface name in the netmap request
  strncpy (nmr.nr_name, m_deviceName.c_str (), m_deviceName.length ());

  // switch the interface in netmap mode
  int code = ioctl (fd, NIOCREGIF, &nmr);
  if (code == -1)
    {
      NS_FATAL_ERROR ("Switching failed");
    }

  // memory mapping
  uint8_t *memory = (uint8_t *) mmap (0, nmr.nr_memsize, PROT_WRITE | PROT_READ,
                                      MAP_SHARED, fd, 0);

  if (memory == MAP_FAILED)
    {
      NS_FATAL_ERROR ("Memory mapping failed");
    }

  // getting the base struct of the interface in netmap mode
  struct netmap_if *nifp = NETMAP_IF (memory, nmr.nr_offset);

  if (!nifp)
    {
      NS_FATAL_ERROR ("Failed getting the base struct of the interface in netmap mode");
    }

  device->SetNetmapInterfaceRepresentation (nifp);
  device->SetTxRingsInfo (nifp->ni_tx_rings, nmr.nr_tx_slots);
  device->SetRxRingsInfo (nifp->ni_rx_rings, nmr.nr_rx_slots);

  device->SetFileDescriptor (fd);

}

} // namespace ns3
