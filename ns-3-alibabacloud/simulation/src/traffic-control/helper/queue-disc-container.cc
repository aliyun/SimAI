/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Universita' degli Studi di Napoli Federico II
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

#include "queue-disc-container.h"

namespace ns3 {

QueueDiscContainer::QueueDiscContainer ()
{
}

QueueDiscContainer::QueueDiscContainer (Ptr<QueueDisc> qDisc)
{
  m_queueDiscs.push_back (qDisc);
}

QueueDiscContainer::ConstIterator
QueueDiscContainer::Begin (void) const
{
  return m_queueDiscs.begin ();
}

QueueDiscContainer::ConstIterator
QueueDiscContainer::End (void) const
{
  return m_queueDiscs.end ();
}

std::size_t
QueueDiscContainer::GetN (void) const
{
  return m_queueDiscs.size ();
}

Ptr<QueueDisc>
QueueDiscContainer::Get (std::size_t i) const
{
  return m_queueDiscs[i];
}

void
QueueDiscContainer::Add (QueueDiscContainer other)
{
  for (ConstIterator i = other.Begin (); i != other.End (); i++)
    {
      m_queueDiscs.push_back (*i);
    }
}

void
QueueDiscContainer::Add (Ptr<QueueDisc> qDisc)
{
  m_queueDiscs.push_back (qDisc);
}

} // namespace ns3
