/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011, 2012 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 * Author: Marco Miozzo  <marco.miozzo@cttc.es>
 *         Nicola Baldo <nbaldo@cttc.es>
 * 
 */

#ifndef PROPAGATION_ENVIRONMENT_H
#define PROPAGATION_ENVIRONMENT_H


namespace ns3 {


/**
 * \ingroup propagation
 *
 * The type of propagation environment
 * 
 */
enum EnvironmentType
  {
    UrbanEnvironment, SubUrbanEnvironment, OpenAreasEnvironment
  };


/**
 * \ingroup propagation
 *
 * The size of the city in which propagation takes place
 * 
 */
enum CitySize
  {
    SmallCity, MediumCity, LargeCity
  };


} // namespace ns3


#endif // PROPAGATION_ENVIRONMENT_H

