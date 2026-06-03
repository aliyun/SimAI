/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2009 CTTC
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
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef HALF_DUPLEX_IDEAL_PHY_SPECTRUM_PARAMETERS_H
#define HALF_DUPLEX_IDEAL_PHY_SPECTRUM_PARAMETERS_H


#include <ns3/spectrum-signal-parameters.h>

namespace ns3 {

class Packet;

/**
 * \ingroup spectrum
 *
 * Signal parameters for HalfDuplexIdealPhy
 */
struct HalfDuplexIdealPhySignalParameters : public SpectrumSignalParameters
{

  // inherited from SpectrumSignalParameters
  virtual Ptr<SpectrumSignalParameters> Copy ();

  /**
   * default constructor
   */
  HalfDuplexIdealPhySignalParameters ();

  /**
   * copy constructor
   * \param p object to copy
   */
  HalfDuplexIdealPhySignalParameters (const HalfDuplexIdealPhySignalParameters& p);

  /**
   * The data packet being transmitted with this signal
   */
  Ptr<Packet> data;
};

}  // namespace ns3


#endif /* HALF_DUPLEX_IDEAL_PHY_SPECTRUM_PARAMETERS_H */
