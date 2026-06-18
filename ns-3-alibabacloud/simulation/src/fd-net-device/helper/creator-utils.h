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

#ifndef CREATOR_UTILS_H
#define CREATOR_UTILS_H

#include <unistd.h>
#include <cstring>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdlib.h>
#include <errno.h>

#include <sys/socket.h>

namespace ns3 {

extern int gVerbose;

#define LOG(msg) \
  if (gVerbose) \
    { \
      std::cout << __FUNCTION__ << "(): " << msg << std::endl;   \
    }

#define ABORT(msg, printErrno) \
  std::cout << __FILE__ << ": fatal error at line " << __LINE__ << ": " << __FUNCTION__ << "(): " << msg << std::endl; \
  if (printErrno) \
    { \
      std::cout << "    errno = " << errno << " (" << strerror (errno) << ")" << std::endl; \
    } \
  exit (-1);

#define ABORT_IF(cond, msg, printErrno) \
  if (cond) \
    { \
      ABORT (msg, printErrno); \
    }

/**
 * \ingroup fd-net-device
 * \brief Send the file descriptor back to the code that invoked the creation.
 *
 * \param path The socket address information from the Unix socket we use
 * to send the created socket back to.
 * \param fd The file descriptor we're going to send.
 * \param magic_number A verification number to verify the caller is talking to the
 * right process.
 */
void SendSocket (const char *path, int fd, const int magic_number);

} // namespace ns3

#endif /* CREATOR_UTILS_DEVICE_H */

