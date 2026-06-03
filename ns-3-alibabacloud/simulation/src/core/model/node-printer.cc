/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Lawrence Livermore National Laboratory
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
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

#include "log.h"
#include "node-printer.h"
#include "simulator.h"  // GetContext()

#include <iomanip>

/**
 * \file
 * \ingroup simulator
 * ns3::DefaultNodePrinter implementation.
 */

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("NodePrinter");

void
DefaultNodePrinter (std::ostream &os)
{
  if (Simulator::GetContext () == Simulator::NO_CONTEXT)
    {
      os << "-1";
    }
  else
    {
      os << Simulator::GetContext ();
    }
}


} // namespace ns3

