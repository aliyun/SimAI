/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
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
 */

#ifndef LTE_VENDOR_SPECIFIC_PARAMETERS
#define LTE_VENDOR_SPECIFIC_PARAMETERS

#include <ns3/ff-mac-common.h>


#define SRS_CQI_RNTI_VSP 1


namespace ns3 {

/**
* \brief Define the RNTI that has generated the 
*/
class SrsCqiRntiVsp : public VendorSpecificValue
{
  public:
  /**
   * \brief SRS CQI RNTI VSP
   *
   * \param rnti the RNTI 
   */
  SrsCqiRntiVsp (uint16_t rnti);
  virtual ~SrsCqiRntiVsp ();
  
  /**
   * \brief Get RNTI function
   *
   * \returns the RNTI 
   */
  uint16_t GetRnti ();
  
  private:
  uint16_t m_rnti; ///< the rnti
};



}; // namespace ns3

#endif /* LTE_VENDOR_SPECIFIC_PARAMETERS */

