/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2006,2007 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */

/**
 * @defgroup mobility Mobility
 */

/**
 * \addtogroup mobility Mobility
 *
 * The mobility support includes:
 *  - a set of mobility models which are used to track and maintain
 *    the "current" cartesian position and speed of an object.
 *
 *  - a "course change notifier" trace source which can be used to register
 *    listeners to the course changes of a mobility model
 * 
 *  - a number of helper classes which are used to place nodes and setup 
 *    mobility models (including parsers for some mobility definition formats).
 */
