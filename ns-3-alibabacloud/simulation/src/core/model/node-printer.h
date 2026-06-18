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
#ifndef NODE_PRINTER_H
#define NODE_PRINTER_H

#include <ostream>

/**
 * \file
 * \ingroup logging
 * Declaration of ns3::NodePrinter function pointer type
 * and ns3::DefaultNodePrinter function.
 */

namespace ns3 {

/**
 * Function signature for prepending the node id
 * to a log message.
 *
 * \param [in,out] os The output stream to print on.
 */
typedef void (*NodePrinter)(std::ostream &os);

/**
 * \ingroup logging
 * Default node id printer implementation.
 *
 * \param [in,out] os The output stream to print the node id on.
 */
void DefaultNodePrinter (std::ostream &os);

} // namespace ns3

#endif /* NODE_H */
