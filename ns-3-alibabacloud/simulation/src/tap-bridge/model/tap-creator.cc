/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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

#include <unistd.h>
#include <stdint.h>
#include <string>
#include <cstring> // for strerror
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <cerrno>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <net/if.h>
#include <linux/if_tun.h>
#include <net/route.h>
#include <netinet/in.h>

#include "tap-encode-decode.h"

#define TAP_MAGIC 95549

static int gVerbose = 0; // Set to true to turn on logging messages.

#define LOG(msg) \
  if (gVerbose) \
    { \
      std::cout << __FUNCTION__ << "(): " << msg << std::endl;   \
    }

#define ABORT(msg, printErrno) \
  std::cout << __FILE__ << ": fatal error at line " << __LINE__ << ": " << __FUNCTION__ << "(): " << msg << std::endl; \
  if (printErrno) \
    { \
      std::cout << "    errno = " << errno << " (" << std::strerror (errno) << ")" << std::endl; \
    } \
  std::exit (-1); 

#define ABORT_IF(cond, msg, printErrno) \
  if (cond) \
    { \
      ABORT (msg, printErrno); \
    }

//
// Lots of the following helper code taken from corresponding functions in src/node.
//
#define ASCII_DOT (0x2e)
#define ASCII_ZERO (0x30)
#define ASCII_a (0x41)
#define ASCII_z (0x5a)
#define ASCII_A (0x61)
#define ASCII_Z (0x7a)
#define ASCII_COLON (0x3a)
#define ASCII_ZERO (0x30)

static char
AsciiToLowCase (char c)
{
  if (c >= ASCII_a && c <= ASCII_z) {
      return c;
    } else if (c >= ASCII_A && c <= ASCII_Z) {
      return c + (ASCII_a - ASCII_A);
    } else {
      return c;
    }
}

static uint32_t 
AsciiToIpv4 (const char *address)
{
  uint32_t host = 0;
  while (true) {
      uint8_t byte = 0;
      while (*address != ASCII_DOT &&
             *address != 0) {
          byte *= 10;
          byte += *address - ASCII_ZERO;
          address++;
        }
      host <<= 8;
      host |= byte;
      if (*address == 0) {
          break;
        }
      address++;
    }
  return host;
}

static void 
AsciiToMac48 (const char *str, uint8_t addr[6])
{
  int i = 0;
  while (*str != 0 && i < 6) 
    {
      uint8_t byte = 0;
      while (*str != ASCII_COLON && *str != 0) 
        {
          byte <<= 4;
          char low = AsciiToLowCase (*str);
          if (low >= ASCII_a)
            {
              byte |= low - ASCII_a + 10;
            }
          else
            {
              byte |= low - ASCII_ZERO;
            }
          str++;
        }
      addr[i] = byte;
      i++;
      if (*str == 0) 
        {
          break;
        }
      str++;
    }
}

static sockaddr
CreateInetAddress (uint32_t networkOrder)
{
  union {
    struct sockaddr any_socket;
    struct sockaddr_in si;
  } s;
  s.si.sin_family = AF_INET;
  s.si.sin_port = 0; // unused
  s.si.sin_addr.s_addr = htonl (networkOrder);
  return s.any_socket;
}

static void
SendSocket (const char *path, int fd)
{
  //
  // Open a Unix (local interprocess) socket to call back to the tap bridge
  //
  LOG ("Create Unix socket");
  int sock = socket (PF_UNIX, SOCK_DGRAM, 0);
  ABORT_IF (sock == -1, "Unable to open socket", 1);

  //
  // We have this string called path, which is really a hex representation
  // of the endpoint that the tap bridge created.  It used a forward encoding
  // method (TapBufferToString) to take the sockaddr_un it made and passed 
  // the resulting string to us.  So we need to take the inverse method
  // (TapStringToBuffer) and build the same sockaddr_un over here.
  //
  socklen_t clientAddrLen;
  struct sockaddr_un clientAddr;

  LOG ("Decode address " << path);
  bool rc = ns3::TapStringToBuffer (path, (uint8_t *)&clientAddr, &clientAddrLen);
  ABORT_IF (rc == false, "Unable to decode path", 0);

  LOG ("Connect");
  int status = connect (sock, (struct sockaddr*)&clientAddr, clientAddrLen);
  ABORT_IF (status == -1, "Unable to connect to tap bridge", 1);

  LOG ("Connected");

  //
  // This is arcane enough that a few words are worthwhile to explain what's 
  // going on here.
  //
  // The interesting information (the socket FD) is going to go back to the
  // tap bridge as an integer of ancillary data.  Ancillary data is bits 
  // that are not a part a socket payload (out-of-band data).  We're also 
  // going to send one integer back.  It's just initialized to a magic number
  // we use to make sure that the tap bridge is talking to the tap socket 
  // creator and not some other creator process (emu, specifically)
  //
  // The struct iovec below is part of a scatter-gather list.  It describes a
  // buffer.  In this case, it describes a buffer (an integer) containing the
  // data that we're going to send back to the tap bridge (that magic number).
  // 
  struct iovec iov;
  uint32_t magic = TAP_MAGIC;
  iov.iov_base = &magic;
  iov.iov_len = sizeof(magic);

  //
  // The CMSG macros you'll see below are used to create and access control 
  // messages (which is another name for ancillary data).  The ancillary 
  // data is made up of pairs of struct cmsghdr structures and associated
  // data arrays.
  // 
  // First, we're going to allocate a buffer on the stack to contain our 
  // data array (that contains the socket).  Sometimes you'll see this called
  // an "ancillary element" but the msghdr uses the control message termimology
  // so we call it "control."
  //
  size_t msg_size = sizeof(int);
  char control[CMSG_SPACE (msg_size)];

  //
  // There is a msghdr that is used to minimize the number of parameters
  // passed to sendmsg (which we will use to send our ancillary data).  This
  // structure uses terminology corresponding to control messages, so you'll
  // see msg_control, which is the pointer to the ancillary data and controllen
  // which is the size of the ancillary data array.
  //
  // So, initialize the message header that describes our ancillary/control data
  // and point it to the control message/ancillary data we just allocated space
  // for.
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
  // A cmsghdr contains a length field that is the length of the header and
  // the data.  It has a cmsg_level field corresponding to the originating 
  // protocol.  This takes values which are legal levels for getsockopt and
  // setsockopt (here SOL_SOCKET).  We're going to use the SCM_RIGHTS type of 
  // cmsg, that indicates that the ancillary data array contains access rights 
  // that we are sending back to the tap bridge.
  //
  // We have to put together the first (and only) cmsghdr that will describe
  // the whole package we're sending.
  //
  struct cmsghdr *cmsg;
  cmsg = CMSG_FIRSTHDR (&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN (msg_size);
  //
  // We also have to update the controllen in case other stuff is actually
  // in there we may not be aware of (due to macros).
  //
  msg.msg_controllen = cmsg->cmsg_len;

  //
  // Finally, we get a pointer to the start of the ancillary data array and
  // put our file descriptor in.
  //
  int *fdptr = (int*)(CMSG_DATA (cmsg));
  *fdptr = fd; // 

  //
  // Actually send the file descriptor back to the tap bridge.
  //
  ssize_t len = sendmsg (sock, &msg, 0);
  ABORT_IF (len == -1, "Could not send socket back to tap bridge", 1);

  LOG ("sendmsg complete");
}

static int
CreateTap (const char *dev, const char *gw, const char *ip, const char *mac, const char *mode, const char *netmask)
{
  //
  // Creation and management of Tap devices is done via the tun device
  //
  int tap = open ("/dev/net/tun", O_RDWR);
  ABORT_IF (tap == -1, "Could not open /dev/net/tun", true);

  //
  // Allocate a tap device, making sure that it will not send the tun_pi header.
  // If we provide a null name to the ifr.ifr_name, we tell the kernel to pick
  // a name for us (i.e., tapn where n = 0..255.
  //
  // If the device does not already exist, the system will create one.
  //
  struct ifreq ifr;
  ifr.ifr_flags = IFF_TAP | IFF_NO_PI;
  strcpy (ifr.ifr_name, dev);
  int status = ioctl (tap, TUNSETIFF, (void *) &ifr);
  ABORT_IF (status == -1, "Could not allocate tap device", true);

  std::string tapDeviceName = (char *)ifr.ifr_name;
  LOG ("Allocated TAP device " << tapDeviceName);

  //
  // Operating mode "2" corresponds to USE_LOCAL and "3" to USE_BRIDGE mode.
  // This means that we expect that the user will have named, created and 
  // configured a network tap that we are just going to use.  So don't mess 
  // up his hard work by changing anything, just return the tap fd.
  //
  if (strcmp (mode, "2") == 0 || strcmp (mode, "3") == 0)
    {
      LOG ("Returning precreated tap ");
      return tap;
    }

  //
  // Set the hardware (MAC) address of the new device
  //
  ifr.ifr_hwaddr.sa_family = 1; // this is ARPHRD_ETHER from if_arp.h
  AsciiToMac48 (mac, (uint8_t*)ifr.ifr_hwaddr.sa_data);
  status = ioctl (tap, SIOCSIFHWADDR, &ifr);
  ABORT_IF (status == -1, "Could not set MAC address", true);
  LOG ("Set device MAC address to " << mac);

  int fd = socket (AF_INET, SOCK_DGRAM, 0);

  //
  // Bring the interface up.
  //
  status = ioctl (fd, SIOCGIFFLAGS, &ifr);
  ABORT_IF (status == -1, "Could not get flags for interface", true);
  ifr.ifr_flags |= IFF_UP | IFF_RUNNING;
  status = ioctl (fd, SIOCSIFFLAGS, &ifr);
  ABORT_IF (status == -1, "Could not bring interface up", true);
  LOG ("Device is up");

  //
  // Set the IP address of the new interface/device.
  //
  ifr.ifr_addr = CreateInetAddress (AsciiToIpv4 (ip));
  status = ioctl (fd, SIOCSIFADDR, &ifr);
  ABORT_IF (status == -1, "Could not set IP address", true);
  LOG ("Set device IP address to " << ip);

  //
  // Set the net mask of the new interface/device
  //
  ifr.ifr_netmask = CreateInetAddress (AsciiToIpv4 (netmask));
  status = ioctl (fd, SIOCSIFNETMASK, &ifr);
  ABORT_IF (status == -1, "Could not set net mask", true);
  LOG ("Set device Net Mask to " << netmask);

  return tap;
}

int
main (int argc, char *argv[])
{
  int c;
  char *dev = (char *)"";
  char *gw = NULL;
  char *ip = NULL;
  char *mac = NULL;
  char *netmask = NULL;
  char *operatingMode = NULL;
  char *path = NULL;

  opterr = 0;

  while ((c = getopt (argc, argv, "vd:g:i:m:n:o:p:")) != -1)
    {
      switch (c)
        {
        case 'd':
          dev = optarg;           // name of the new tap device
          break;
        case 'g':
          gw = optarg;            // gateway address for the new device
          break;
        case 'i':
          ip = optarg;            // ip address of the new device
          break;
        case 'm':
          mac = optarg;           // mac address of the new device
          break;
        case 'n':
          netmask = optarg;       // net mask for the new device
          break;
        case 'o':
          operatingMode = optarg; // operating mode of tap bridge
          break;
        case 'p':
          path = optarg;          // path back to the tap bridge
          break;
        case 'v':
          gVerbose = true;
          break;
        }
    }

  //
  // We have got to be able to coordinate the name of the tap device we are
  // going to create and or open with the device that an external Linux host
  // will use.  If this name is provided we use it.  If not we let the system
  // create the device for us.  This name is given in dev
  //
  LOG ("Provided Device Name is \"" << dev << "\"");

  //
  // We have got to be able to provide a gateway to the external Linux host 
  // so it can talk to the ns-3 network.  This ip address is provided in 
  // gw.
  //
  ABORT_IF (gw == NULL, "Gateway Address is a required argument", 0);
  LOG ("Provided Gateway Address is \"" << gw << "\"");

  //
  // We have got to be able to assign an IP address to the tap device we are
  // allocating.  This address is allocated in the simulation and assigned to
  // the tap bridge.  This address is given in ip.
  //
  ABORT_IF (ip == NULL, "IP Address is a required argument", 0);
  LOG ("Provided IP Address is \"" << ip << "\"");

  //
  // We have got to be able to assign a Mac address to the tap device we are
  // allocating.  This address is allocated in the simulation and assigned to
  // the bridged device.  This allows packets addressed to the bridged device
  // to appear in the Linux host as if they were received there.
  //
  ABORT_IF (mac == NULL, "MAC Address is a required argument", 0);
  LOG ("Provided MAC Address is \"" << mac << "\"");

  //
  // We have got to be able to assign a net mask to the tap device we are
  // allocating.  This mask is allocated in the simulation and given to
  // the bridged device.
  //
  ABORT_IF (netmask == NULL, "Net Mask is a required argument", 0);
  LOG ("Provided Net Mask is \"" << netmask << "\"");

  //
  // We have got to know whether or not to create the TAP.
  //
  ABORT_IF (operatingMode == NULL, "Operating Mode is a required argument", 0);
  LOG ("Provided Operating Mode is \"" << operatingMode << "\"");

  //
  // This program is spawned by a tap bridge running in a simulation.  It
  // wants to create a socket as described below.  We are going to do the
  // work here since we're running suid root.  Once we create the socket,
  // we have to send it back to the tap bridge.  We do that over a Unix
  // (local interprocess) socket.  The tap bridge created a socket to 
  // listen for our response on, and it is expected to have encoded the address
  // information as a string and to have passed that string as an argument to
  // us.  We see it here as the "path" string.  We can't do anything useful 
  // unless we have that string.
  //
  ABORT_IF (path == NULL, "path is a required argument", 0);
  LOG ("Provided path is \"" << path << "\"");

  //
  // The whole reason for all of the hoops we went through to call out to this
  // program will pay off here.  We created this program to run as suid root
  // in order to keep the main simulation program from having to be run with
  // root privileges.  We need root privileges to be able to futz with the 
  // Tap device underlying all of this.  So all of these hoops are to allow 
  // us to exeucte the following code:
  //
  LOG ("Creating Tap");
  int sock = CreateTap (dev, gw, ip, mac, operatingMode, netmask);
  ABORT_IF (sock == -1, "main(): Unable to create tap socket", 1);

  //
  // Send the socket back to the tap net device so it can go about its business
  //
  SendSocket (path, sock);

  return 0;
}
