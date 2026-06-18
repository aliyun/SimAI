/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2015 Lawrence Livermore National Laboratory
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
 * Authors: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 */

/* 
 * This is an example included in the doxygen-generated documentation
 * for NS_DEPRECATED.
 */

// [doxygen snippet]
/*
 * Do something useful.
 *
 * \deprecated This method will go away in future versions of ns-3.
 * See instead TheNewWay()
 */
NS_DEPRECATED_3_XX ("see TheNewWay")
void SomethingUseful ();
/*
 * Do something more useful.
 */
void TheNewWay ();
// [doxygen snippet]
