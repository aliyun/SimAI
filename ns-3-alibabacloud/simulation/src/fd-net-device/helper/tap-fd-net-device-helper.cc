/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 INRIA, 2012 University of Washington
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
 */

#include "tap-fd-net-device-helper.h"
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
#include "ns3/internet-module.h"

#include <arpa/inet.h>
#include <errno.h>
#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>
#include <net/ethernet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netpacket/packet.h>
#include <linux/if_tun.h>

#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include <string>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TapFdNetDeviceHelper");

#define TAP_MAGIC 95549

TapFdNetDeviceHelper::TapFdNetDeviceHelper ()
{
  m_deviceName = "";
  m_modePi = false;
  m_tapIp4 = Ipv4Address::GetZero ();
  m_tapMask4 = Ipv4Mask::GetZero ();
  m_tapIp6 = Ipv6Address::GetZero ();
  m_tapPrefix6 = 64;
  m_tapMac = Mac48Address::Allocate ();
}

void
TapFdNetDeviceHelper::SetModePi (bool modePi)
{
  m_modePi = modePi;
}

void
TapFdNetDeviceHelper::SetTapIpv4Address (Ipv4Address address)
{
  m_tapIp4 = address;
}

void
TapFdNetDeviceHelper::SetTapIpv4Mask (Ipv4Mask mask)
{
  m_tapMask4 = mask;
}

void
TapFdNetDeviceHelper::SetTapIpv6Address (Ipv6Address address)
{
  m_tapIp6 = address;
}

void
TapFdNetDeviceHelper::SetTapIpv6Prefix (int prefix)
{
  m_tapPrefix6 = prefix;
}

void
TapFdNetDeviceHelper::SetTapMacAddress (Mac48Address mac)
{
  m_tapMac = mac;
}

Ptr<NetDevice>
TapFdNetDeviceHelper::InstallPriv (Ptr<Node> node) const
{
  Ptr<NetDevice> d = FdNetDeviceHelper::InstallPriv (node);
  Ptr<FdNetDevice> device = d->GetObject<FdNetDevice> ();

  //
  // We need to explicitly set the encapsulation mode for the traffic
  // traversing the TAP device, so the FdNetDevice is able to know
  // how to treat the traffic in a way that in compatible with the
  // TAP device.
  //
  if (m_modePi)
    {
      device->SetEncapsulationMode (FdNetDevice::DIXPI);
    }

  SetFileDescriptor (device);
  return device;
}

void
TapFdNetDeviceHelper::SetFileDescriptor (Ptr<FdNetDevice> device) const
{
  NS_LOG_LOGIC ("Creating TAP device");

  //
  // Call out to a separate process running as suid root in order to create a
  // TAP device.  We do this to avoid having the entire simulation running as root.
  //
  int fd = CreateFileDescriptor ();
  device->SetFileDescriptor (fd);
}

int
TapFdNetDeviceHelper::CreateFileDescriptor (void) const
{
  NS_LOG_FUNCTION (this);

  //
  // We're going to fork and exec that program soon, but first we need to have
  // a socket to talk to it with.  So we create a local interprocess (Unix)
  // socket for that purpose.
  //
  int sock = socket (PF_UNIX, SOCK_DGRAM, 0);
  NS_ABORT_MSG_IF (sock == -1, "TapFdNetDeviceHelper::CreateFileDescriptor(): Unix socket creation error, errno = " << strerror (errno));

  //
  // Bind to that socket and let the kernel allocate an endpoint
  //
  struct sockaddr_un un;
  memset (&un, 0, sizeof (un));
  un.sun_family = AF_UNIX;
  int status = bind (sock, (struct sockaddr*)&un, sizeof (sa_family_t));
  NS_ABORT_MSG_IF (status == -1, "TapFdNetDeviceHelper::CreateFileDescriptor(): Could not bind(): errno = " << strerror (errno));
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
  NS_ABORT_MSG_IF (status == -1, "TapFdNetDeviceHelper::CreateFileDescriptor(): Could not getsockname(): errno = " << strerror (errno));

  //
  // Now encode that socket name (family and path) as a string of hex digits
  //
  std::string path = BufferToString ((uint8_t *)&un, len);
  NS_LOG_INFO ("Encoded Unix socket as \"" << path << "\"");

  //
  // Fork and exec the process to create our socket.  If we're us (the parent)
  // we wait for the child (the creator) to complete and read the socket it
  // created and passed back using the ancillary data mechanism.
  //
  pid_t pid = ::fork ();
  if (pid == 0)
    {
      NS_LOG_DEBUG ("Child process");

      //
      // build a command line argument from the encoded endpoint string that
      // the socket creation process will use to figure out how to respond to
      // the (now) parent process.  We're going to have to give this program
      // quite a bit of information.
      //
      // -d<device-name> The name of the tap device we want to create;
      // -m<MAC-address> The MAC-48 address to assign to the new tap device;
      // -i<IPv4-address> The IP v4 address to assign to the new tap device;
      // -I<IPv6-address> The IP v6 address to assign to the new tap device;
      // -n<network-IPv4-mask> The network IPv4 mask to assign to the new tap device;
      // -N<network-IPv6-mask> The network IPv6 mask to assign to the new tap device;
      // -t Set the IFF_TAP flag
      // -h Set the IFF_NO_PI flag
      // -p<path> the path to the unix socket described above.
      //
      // Example tap-creator -dnewdev -i1.2.3.1 -m08:00:2e:00:01:23 -n255.255.255.0 -t -h -pblah
      //

      //
      // The device-name is something we may want the system to make up in
      // every case.  We also rely on it being configured via an Attribute
      // through the helper.  By default, it is set to the empty string
      // which tells the system to make up a device name such as "tap123".
      //
      std::ostringstream ossDeviceName;
      if (m_deviceName != "")
        {
          ossDeviceName << "-d" << m_deviceName;
        }

      std::ostringstream ossMac;
      ossMac << "-m" << m_tapMac;

      std::ostringstream ossIp4;
      if (m_tapIp4 != Ipv4Address::GetZero ())
        {
          ossIp4 << "-i" << m_tapIp4;
        }

      std::ostringstream ossIp6;
      if (m_tapIp6 != Ipv6Address::GetZero ())
        {
          ossIp6 << "-I" << m_tapIp6;
        }

      std::ostringstream ossNetmask4;
      if (m_tapMask4 != Ipv4Mask::GetZero () )
        {
          ossNetmask4 << "-n" << m_tapMask4;
        }

      std::ostringstream ossPrefix6;
      ossPrefix6 << "-P" << m_tapPrefix6;

      std::ostringstream ossMode;
      ossMode << "-t";

      std::ostringstream ossPI;
      if (m_modePi)
        {
          ossPI << "-h";
        }

      std::ostringstream ossPath;
      ossPath << "-p" << path;

      //
      // Execute the socket creation process image.
      //
      status = ::execlp (TAP_DEV_CREATOR,
                         TAP_DEV_CREATOR,                       // argv[0] (filename)
                         ossDeviceName.str ().c_str (),     // argv[1] (-d<device name>)
                         ossMac.str ().c_str (),            // argv[2] (-m<MAC address>
                         ossIp4.str ().c_str (),            // argv[3] (-i<IP v4 address>)
                         ossIp6.str ().c_str (),            // argv[4] (-I<IP v6 address>)
                         ossNetmask4.str ().c_str (),       // argv[5] (-n<IP v4 net mask>)
                         ossPrefix6.str ().c_str (),        // argv[6] (-P<IP v6 prefix>)
                         ossMode.str ().c_str (),           // argv[7] (-t <tap>)
                         ossPI.str ().c_str (),             // argv[8] (-h <pi>)
                         ossPath.str ().c_str (),           // argv[9] (-p<path>)
                         (char *)NULL);

      //
      // If the execlp successfully completes, it never returns.  If it returns it failed or the OS is
      // broken.  In either case, we bail.
      //
      NS_FATAL_ERROR ("TapFdNetDeviceHelper::CreateFileDescriptor(): Back from execlp(), status = " <<
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
      NS_ABORT_MSG_IF (waited == -1, "TapFdNetDeviceHelper::CreateFileDescriptor(): waitpid() fails, errno = " << strerror (errno));
      NS_ASSERT_MSG (pid == waited, "TapFdNetDeviceHelper::CreateFileDescriptor(): pid mismatch");

      //
      // Check to see if the socket creator exited normally and then take a
      // look at the exit code.  If it bailed, so should we.  If it didn't
      // even exit normally, we bail too.
      //
      if (WIFEXITED (st))
        {
          int exitStatus = WEXITSTATUS (st);
          NS_ABORT_MSG_IF (exitStatus != 0,
                           "TapFdNetDeviceHelper::CreateFileDescriptor(): socket creator exited normally with status " << exitStatus);
        }
      else
        {
          NS_FATAL_ERROR ("TapFdNetDeviceHelper::CreateFileDescriptor(): socket creator exited abnormally");
        }

      //
      // At this point, the socket creator has run successfully and should
      // have created our tap device, initialized it with the information we
      // passed and sent it back to the socket address we provided.  A socket
      // (fd) we can use to talk to this tap device should be waiting on the
      // Unix socket we set up to receive information back from the creator
      // program.  We've got to do a bunch of grunt work to get at it, though.
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
      // an "ancillary element" but the msghdr uses the control message termimology
      // so we call it "control."
      //
      size_t msg_size = sizeof(int);
      char control[CMSG_SPACE (msg_size)];

      //
      // There is a msghdr that is used to minimize the number of parameters
      // passed to recvmsg (which we will use to receive our ancillary data).
      // This structure uses terminology corresponding to control messages, so
      // you'll see msg_control, which is the pointer to the ancillary data and
      // controllen which is the size of the ancillary data array.
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
      // Now we can actually receive the interesting bits from the tap
      // creator process.  Lots of pain to get four bytes.
      //
      ssize_t bytesRead = recvmsg (sock, &msg, 0);
      NS_ABORT_MSG_IF (bytesRead != sizeof(int), "TapFdNetDeviceHelper::CreateFileDescriptor(): Wrong byte count from socket creator");

      //
      // There may be a number of message headers/ancillary data arrays coming in.
      // Let's look for the one with a type SCM_RIGHTS which indicates it's the
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
              if (magic == TAP_MAGIC)
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
  NS_FATAL_ERROR ("Should be unreachable");
  return 0; // Silence compiler warning about lack of return value
}

} // namespace ns3


