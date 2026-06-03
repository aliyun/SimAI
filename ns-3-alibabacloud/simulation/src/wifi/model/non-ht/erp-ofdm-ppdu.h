/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2020 Orange Labs
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
 * Author: Rediet <getachew.redieteab@orange.com>
 *         Muhammad Iqbal Rochman <muhiqbalcr@uchicago.edu>
 */

#ifndef ERP_OFDM_PPDU_H
#define ERP_OFDM_PPDU_H

#include "ofdm-ppdu.h"

/**
 * \file
 * \ingroup wifi
 * Declaration of ns3::ErpOfdmPpdu class.
 */

namespace ns3 {

class WifiPsdu;

/**
 * \brief ERP-OFDM PPDU (11g)
 * \ingroup wifi
 *
 * ErpOfdmPpdu stores a preamble, PHY headers and a PSDU of a PPDU with non-HT header,
 * i.e., PPDU that uses ERP-OFDM modulation.
 */
class ErpOfdmPpdu : public OfdmPpdu
{
public:
  /**
   * Create an ERP-OFDM PPDU.
   *
   * \param psdu the PHY payload (PSDU)
   * \param txVector the TXVECTOR that was used for this PPDU
   * \param band the WifiPhyBand used for the transmission of this PPDU
   * \param uid the unique ID of this PPDU
   */
  ErpOfdmPpdu (Ptr<const WifiPsdu> psdu, const WifiTxVector& txVector,
               WifiPhyBand band, uint64_t uid);
  /**
   * Destructor for ErpOfdmPpdu.
   */
  virtual ~ErpOfdmPpdu ();

  Ptr<WifiPpdu> Copy (void) const override;

private:
  WifiTxVector DoGetTxVector (void) const override;
}; //class ErpOfdmPpdu

} //namespace ns3

#endif /* ERP_OFDM_PPDU_H */
