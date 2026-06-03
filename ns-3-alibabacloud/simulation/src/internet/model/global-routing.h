/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright 2008 University of Washington
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
 */

/**
 * \ingroup ipv4Routing
 * \defgroup globalrouting Global Routing.
 *
 * Performs pre-simulation static route computation
 * on a layer-3 IPv4 topology.
 *
 * \section model Model
 *
 * ns-3 global routing performs pre-simulation static route computation
 * on a layer-3 IPv4 topology.  The user API from the script level is
 * fairly minimal; once a topology has been constructed and addresses
 * assigned, the user may call ns3::GlobalRouteManager::PopulateRoutingTables()
 * and the simulator will initialize the routing database and set up
 * static unicast forwarding tables for each node.
 *
 * The model assumes that all nodes on an ns-3 channel are reachable to
 * one another, regardless of whether the nodes can use the channel
 * successfully (in the case of wireless).  Therefore, this model
 * should typically be used only on wired topologies.  Layer-2 bridge
 * devices are supported.  API does not yet exist to control the subset 
 * of a topology to which this global static routing is applied.
 * 
 * If the topology changes during the simulation, by default, routing
 * will not adjust.  There are two ways to make it adjust.
 * - Set the attribute Ipv4GlobalRouting::RespondToInterfaceEvents to true 
 * - Manually call the sequence of GlobalRouteManager methods to delete global
 *   routes, build global routing database, and initialize routes.
 *   There is a helper method that encapsulates this 
 *   (Ipv4GlobalRoutingHelper::RecomputeRoutingTables())
 * 
 * \section api API and Usage
 * 
 * Users must include ns3/global-route-manager.h header file.  After the
 * IPv4 topology has been built and addresses assigned, users call
 * ns3::GlobalRouteManager::PopulateRoutingTables (), prior to the 
 * ns3::Simulator::Run() call.
 *
 * There are two attributes of Ipv4GlobalRouting that govern behavior.
 * - Ipv4GlobalRouting::RandomEcmpRouting
 * - Ipv4GlobalRouting::RespondToInterfaceEvents
 *
 * \section impl Implementation
 *
 * A singleton object, ns3::GlobalRouteManager, builds a global routing
 * database of information about the topology, and executes a Dijkstra
 * Shortest Path First (SPF) algorithm on the topology for each node, and
 * stores the computed routes in each node's IPv4 forwarding table by
 * making use of the routing API in class ns3::Ipv4.
 *
 * The nodes that export data are those that have had an ns3::GlobalRouter
 * object aggregated to them.  The ns3::GlobalRouter can be thought of
 * as a per-node agent that exports topology information to the 
 * ns3::GlobalRouteManager.  When it comes time to build the global
 * routing database, the list of nodes is iterated and each node with
 * an ns3::GlobalRouter object is asked to export routing information
 * concerning the links to which it is attached.
 *
 * The format of the data exported conforms to the OSPFv2 standard
 * \RFC{2328}.  In particular, the
 * information is exported in the form of ns3::GlobalLSA objects that
 * semantically match the Link State Advertisements of OSPF.
 *
 * By using a standard data format for reporting topology, existing 
 * OSPF route computation code can be reused, and that is what is done
 * by the ns3::GlobalRouteManager.  The main computation functions are
 * ported from the quagga routing suite (http://www.quagga.net).
 * 
 */
