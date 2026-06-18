/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) University of Washington
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
#include <string>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <errno.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/ioctl.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "creator-utils.h"
#include "encode-decode.h"

namespace ns3 {

/// Flag to enable / disable verbose log mode
int gVerbose = 0;

/**
 * \brief Send the file descriptor back to the code that invoked the creation.
 *
 * \param path The socket address information from the Unix socket we use
 * to send the created socket back to.
 * \param fd The file descriptor we're going to send.
 * \param magic_number A verification number to verify the caller is talking to the
 * right process.
 */
void
SendSocket (const char *path, int fd, const int magic_number)
{
  //
  // Open a Unix (local interprocess) socket to call back to the emu net
  // device.
  //
  LOG ("Create Unix socket");
  int sock = socket (PF_UNIX, SOCK_DGRAM, 0);
  ABORT_IF (sock == -1, "Unable to open socket", 1);

  //
  // We have this string called path, which is really a hex representation
  // of the endpoint that the net device created.  It used a forward encoding
  // method (BufferToString) to take the sockaddr_un it made and passed
  // the resulting string to us.  So we need to take the inverse method
  // (StringToBuffer) and build the same sockaddr_un over here.
  //
  socklen_t clientAddrLen;
  struct sockaddr_un clientAddr;

  LOG ("Decode address " << path);
  bool rc = ns3::StringToBuffer (path, (uint8_t *)&clientAddr, &clientAddrLen);
  ABORT_IF (rc == false, "Unable to decode path", 0);

  LOG ("Connect");
  int status = connect (sock, (struct sockaddr*)&clientAddr, clientAddrLen);
  ABORT_IF (status == -1, "Unable to connect to emu device", 1);

  LOG ("Connected");

  //
  // This is arcane enough that a few words are worthwhile to explain what's
  // going on here.
  //
  // The interesting information (the socket FD) is going to go back to the
  // fd net device as an integer of ancillary data.  Ancillary data is bits
  // that are not a part a socket payload (out-of-band data).  We're also
  // going to send one integer back.  It's just initialized to a magic number
  // we use to make sure that the fd device is talking to the emu socket
  // creator and not some other creator process.
  //
  // The struct iovec below is part of a scatter-gather list.  It describes a
  // buffer.  In this case, it describes a buffer (an integer) containing the
  // data that we're going to send back to the emu net device (that magic
  // number).
  //
  struct iovec iov;
  uint32_t magic = magic_number;
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
  // that we are sending back to the emu net device.
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
  // Actually send the file descriptor back to the emulated net device.
  //
  ssize_t len = sendmsg (sock, &msg, 0);
  ABORT_IF (len == -1, "Could not send socket back to emu net device", 1);

  LOG ("sendmsg complete");
}

} // namespace ns3
