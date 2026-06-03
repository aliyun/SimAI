/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Yufei Cheng
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
 * Author: Yufei Cheng   <yfcheng@ittc.ku.edu>
 *
 * James P.G. Sterbenz <jpgs@ittc.ku.edu>, director
 * ResiliNets Research Group  http://wiki.ittc.ku.edu/resilinets
 * Information and Telecommunication Technology Center (ITTC)
 * and Department of Electrical Engineering and Computer Science
 * The University of Kansas Lawrence, KS USA.
 *
 * Work supported in part by NSF FIND (Future Internet Design) Program
 * under grant CNS-0626918 (Postmodern Internet Architecture),
 * NSF grant CNS-1050226 (Multilayer Network Resilience Analysis and Experimentation on GENI),
 * US Department of Defense (DoD), and ITTC at The University of Kansas.
 */

#include "dsr-gratuitous-reply-table.h"
#include "ns3/log.h"
#include <algorithm>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DsrGraReplyTable");

namespace dsr {

NS_OBJECT_ENSURE_REGISTERED (DsrGraReply);

TypeId DsrGraReply::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::dsr::DsrGraReply")
    .SetParent<Object> ()
    .SetGroupName ("Dsr")
    .AddConstructor<DsrGraReply> ()
  ;
  return tid;
}

DsrGraReply::DsrGraReply ()
{
}

DsrGraReply::~DsrGraReply ()
{
  NS_LOG_FUNCTION_NOARGS ();
}

bool
DsrGraReply::FindAndUpdate (Ipv4Address replyTo, Ipv4Address replyFrom, Time gratReplyHoldoff)
{
  Purge ();  // purge the gratuitous reply table
  for (std::vector<GraReplyEntry>::iterator i = m_graReply.begin ();
       i != m_graReply.end (); ++i)
    {
      if ((i->m_replyTo == replyTo) && (i->m_hearFrom == replyFrom))
        {
          NS_LOG_DEBUG ("Update the reply to ip address if found the gratuitous reply entry");
          i->m_gratReplyHoldoff = std::max (gratReplyHoldoff + Simulator::Now (), i->m_gratReplyHoldoff);
          return true;
        }
    }
  return false;
}

bool
DsrGraReply::AddEntry (GraReplyEntry & graTableEntry)
{
  m_graReply.push_back (graTableEntry);
  return true;
}

void
DsrGraReply::Purge ()
{
  /*
   * Purge the expired gratuitous reply entries
   */
  m_graReply.erase (remove_if (m_graReply.begin (), m_graReply.end (),
                               IsExpired ()), m_graReply.end ());
}

} // namespace dsr
} // namespace ns3
