/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 University of Washington
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
 * Author: Leonard Tracy <lentracy@gmail.com>
 */

#ifndef UAN_PROP_MODEL_IDEAL_H
#define UAN_PROP_MODEL_IDEAL_H

#include "uan-prop-model.h"
#include "ns3/mobility-model.h"
#include "ns3/nstime.h"

namespace ns3 {

/**
 * \ingroup uan
 *
 * Ideal propagation model (no pathloss, impulse PDP).
 */
class UanPropModelIdeal : public UanPropModel
{
public:
  /** Default constructor. */
  UanPropModelIdeal ();
  /** Destructor */
  virtual ~UanPropModelIdeal ();

  /**
   * Register this type.
   * \return The object TypeId.
   */
  static TypeId GetTypeId (void);

  // Inherited methods
  virtual double GetPathLossDb (Ptr<MobilityModel> a, Ptr<MobilityModel> b, UanTxMode mode);
  virtual UanPdp GetPdp (Ptr<MobilityModel> a, Ptr<MobilityModel> b, UanTxMode mode);
  virtual Time GetDelay (Ptr<MobilityModel> a, Ptr<MobilityModel> b, UanTxMode mode);

};  // class UanPropModelIdeal

} // namespace ns3

#endif /* UAN_PROP_MODEL_IDEAL_H */
