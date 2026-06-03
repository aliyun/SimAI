/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */

/*
 * Copyright (c) 2010 The Boeing Company
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
 * Author: Tom Goff <thomas.goff@boeing.com>
 */

#include <sys/select.h>

#include <cerrno>
#include <cstring>
#include <unistd.h>  // close()
#include <fcntl.h>
#include <thread>

#include "log.h"
#include "fatal-error.h"
#include "simple-ref-count.h"
#include "simulator.h"

#include "unix-fd-reader.h"

/**
 * \file
 * \ingroup system
 * ns3::FdReader implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("FdReader");

FdReader::FdReader ()
  : m_fd (-1), m_readCallback (0), m_stop (false),
    m_destroyEvent ()
{
  NS_LOG_FUNCTION (this);
  m_evpipe[0] = -1;
  m_evpipe[1] = -1;
}

FdReader::~FdReader ()
{
  NS_LOG_FUNCTION (this);
  Stop ();
}

void FdReader::Start (int fd, Callback<void, uint8_t *, ssize_t> readCallback)
{
  NS_LOG_FUNCTION (this << fd << &readCallback);
  int tmp;

  NS_ASSERT_MSG (!m_readThread.joinable(), "read thread already exists");

  // create a pipe for inter-thread event notification
  tmp = pipe (m_evpipe);
  if (tmp == -1)
    {
      NS_FATAL_ERROR ("pipe() failed: " << std::strerror (errno));
    }

  // make the read end non-blocking
  tmp = fcntl (m_evpipe[0], F_GETFL);
  if (tmp == -1)
    {
      NS_FATAL_ERROR ("fcntl() failed: " << std::strerror (errno));
    }
  if (fcntl (m_evpipe[0], F_SETFL, tmp | O_NONBLOCK) == -1)
    {
      NS_FATAL_ERROR ("fcntl() failed: " << std::strerror (errno));
    }

  m_fd = fd;
  m_readCallback = readCallback;

  //
  // We're going to spin up a thread soon, so we need to make sure we have
  // a way to tear down that thread when the simulation stops.  Do this by
  // scheduling a "destroy time" method to make sure the thread exits before
  // proceeding.
  //
  if (!m_destroyEvent.IsRunning ())
    {
      // hold a reference to ensure that this object is not
      // deallocated before the destroy-time event fires
      this->Ref ();
      m_destroyEvent =
        Simulator::ScheduleDestroy (&FdReader::DestroyEvent, this);
    }

  //
  // Now spin up a thread to read from the fd
  //
  NS_LOG_LOGIC ("Spinning up read thread");

  m_readThread = std::thread (&FdReader::Run, this);
}

void FdReader::DestroyEvent (void)
{
  NS_LOG_FUNCTION (this);
  Stop ();
  this->Unref ();
}

void FdReader::Stop (void)
{
  NS_LOG_FUNCTION (this);
  m_stop = true;

  // signal the read thread
  if (m_evpipe[1] != -1)
    {
      char zero = 0;
      ssize_t len = write (m_evpipe[1], &zero, sizeof (zero));
      if (len != sizeof (zero))
        {
          NS_LOG_WARN ("incomplete write(): " << std::strerror (errno));
        }
    }

  // join the read thread
  if (m_readThread.joinable ())
    {
      m_readThread.join ();
    }

  // close the write end of the event pipe
  if (m_evpipe[1] != -1)
    {
      close (m_evpipe[1]);
      m_evpipe[1] = -1;
    }

  // close the read end of the event pipe
  if (m_evpipe[0] != -1)
    {
      close (m_evpipe[0]);
      m_evpipe[0] = -1;
    }

  // reset everything else
  m_fd = -1;
  m_readCallback.Nullify ();
  m_stop = false;
}

// This runs in a separate thread
void FdReader::Run (void)
{
  NS_LOG_FUNCTION (this);
  int nfds;
  fd_set rfds;

  nfds = (m_fd > m_evpipe[0] ? m_fd : m_evpipe[0]) + 1;

  FD_ZERO (&rfds);
  FD_SET (m_fd, &rfds);
  FD_SET (m_evpipe[0], &rfds);

  for (;;)
    {
      int r;
      fd_set readfds = rfds;

      r = select (nfds, &readfds, NULL, NULL, NULL);
      if (r == -1 && errno != EINTR)
        {
          NS_FATAL_ERROR ("select() failed: " << std::strerror (errno));
        }

      if (FD_ISSET (m_evpipe[0], &readfds))
        {
          // drain the event pipe
          for (;;)
            {
              char buf[1024];
              ssize_t len = read (m_evpipe[0], buf, sizeof (buf));
              if (len == 0)
                {
                  NS_FATAL_ERROR ("event pipe closed");
                }
              if (len < 0)
                {
                  if (errno == EAGAIN || errno == EINTR || errno == EWOULDBLOCK)
                    {
                      break;
                    }
                  else
                    {
                      NS_FATAL_ERROR ("read() failed: " << std::strerror (errno));
                    }
                }
            }
        }

      if (m_stop)
        {
          // this thread is done
          break;
        }

      if (FD_ISSET (m_fd, &readfds))
        {
          struct FdReader::Data data = DoRead ();
          // reading stops when m_len is zero
          if (data.m_len == 0)
            {
              break;
            }
          // the callback is only called when m_len is positive (data
          // is ignored if m_len is negative)
          else if (data.m_len > 0)
            {
              m_readCallback (data.m_buf, data.m_len);
            }
        }
    }
}

} // namespace ns3
