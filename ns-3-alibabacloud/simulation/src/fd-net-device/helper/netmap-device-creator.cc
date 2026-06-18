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
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/ioctl.h>
#include <net/ethernet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "creator-utils.h"

#define EMU_MAGIC 65867

using namespace ns3;

int
main (int argc, char *argv[])
{
  int c;
  char *path = NULL;

  opterr = 0;

  while ((c = getopt (argc, argv, "vp:")) != -1)
    {
      switch (c)
        {
          case 'v':
            gVerbose = true;
            break;
          case 'p':
            path = optarg;
            break;
        }
    }

  //
  // This program is spawned by an emu net device running in a simulation.  It
  // wants to create a raw socket as described below.  We are going to do the
  // work here since we're running suid root.  Once we create the raw socket,
  // we have to send it back to the emu net device.  We do that over a Unix
  // (local interprocess) socket.  The emu net device created a socket to
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
  // root privileges.  We need root privileges to be able to open a raw socket
  // though.  So all of these hoops are to allow us to execute the following
  // single line of code:
  //
  LOG ("Creating netmap fd");
  int sock = open ("/dev/netmap", O_RDWR);
  ABORT_IF (sock == -1, "CreateSocket(): Unable to open netmap fd", 1);

  //
  // Send the socket back to the emu net device so it can go about its business
  //
  SendSocket (path, sock, EMU_MAGIC);

  return 0;
}

