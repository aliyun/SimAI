/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2019 Universita' degli Studi di Napoli Federico II
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
 * Author: Stefano Avallone <stavallo@unina.it>
 */

#include "ns3/log.h"
#include "block-ack-window.h"
#include "wifi-utils.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("BlockAckWindow");

BlockAckWindow::BlockAckWindow ()
  : m_winStart (0),
    m_head (0)
{
}

void
BlockAckWindow::Init (uint16_t winStart, uint16_t winSize)
{
  NS_LOG_FUNCTION (this << winStart << winSize);
  m_winStart = winStart;
  m_window.assign (winSize, false);
  m_head = 0;
}

void
BlockAckWindow::Reset (uint16_t winStart)
{
  Init (winStart, m_window.size ());
}

uint16_t
BlockAckWindow::GetWinStart (void) const
{
  return m_winStart;
}

uint16_t
BlockAckWindow::GetWinEnd (void) const
{
  return (m_winStart + m_window.size () - 1) % SEQNO_SPACE_SIZE;
}

std::size_t
BlockAckWindow::GetWinSize (void) const
{
  return m_window.size ();
}

std::vector<bool>::reference
BlockAckWindow::At (std::size_t distance)
{
  NS_ASSERT (distance < m_window.size ());

  return m_window.at ((m_head + distance) % m_window.size ());
}

std::vector<bool>::const_reference
BlockAckWindow::At (std::size_t distance) const
{
  NS_ASSERT (distance < m_window.size ());

  return m_window.at ((m_head + distance) % m_window.size ());
}

void
BlockAckWindow::Advance (std::size_t count)
{
  NS_LOG_FUNCTION (this << count);

  if (count >= m_window.size ())
    {
      Reset ((m_winStart + count) % SEQNO_SPACE_SIZE);
      return;
    }

  for (std::size_t i = 0; i < count; i++)
    {
      m_window[m_head] = false;
      m_head = (m_head + 1) % m_window.size ();
    }
  m_winStart = (m_winStart + count) % SEQNO_SPACE_SIZE;
}

} //namespace ns3
