/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2010 TELEMATICS LAB, DEE - Politecnico di Bari
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
 * Original Author: Giuseppe Piro  <g.piro@poliba.it>
 * Modified by:     Nicola Baldo   <nbaldo@cttc.es>
 * Modified by:     Marco Miozzo   <mmiozzo@cttc.es>
 */

#ifndef AMCMODULE_H
#define AMCMODULE_H

#include <vector>
#include <ns3/ptr.h>
#include <ns3/object.h>

namespace ns3 {

class SpectrumValue;

/**
 * \ingroup lte
 * Implements the Adaptive Modulation And Coding Scheme. As proposed in 3GPP
 * TSG-RAN WG1 [R1-081483 Conveying MCS and TB size via PDCCH]
 * (http://www.3gpp.org/ftp/tsg_ran/WG1_RL1/TSGR1_52b/Docs/R1-081483.zip).
 */
class LteAmc : public Object
{

public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  
  LteAmc ();
  virtual ~LteAmc();
  
  /// Types of AMC model.
  enum AmcModel
    {
      /**
       * \details
       * An AMC model based on Piro, G.; Grieco, L.A; Boggia, G.; Camarda, P.,
       * "A two-level scheduling algorithm for QoS support in the downlink of
       * LTE cellular networks," _Wireless Conference (EW), 2010 European_,
       * pp.246,253, 12-15 April 2010.
       */
      PiroEW2010,
      /**
       * An AMC model based on 10% of BER according to LteMiErrorModel.
       */
      MiErrorModel
    };
  
  /**
   * \brief Get the Modulation and Coding Scheme for
   * a CQI value
   * \param cqi the cqi value
   * \return the MCS value
   */
  int GetMcsFromCqi (int cqi);

  /**
  * \brief Get the Transport Block Size for a selected MCS and number of PRB (table 7.1.7.2.1-1 of 36.213)
  * \param mcs the MCS index
  * \param nprb the no. of PRB
  * \return the Transport Block Size in bits
  */
  int GetDlTbSizeFromMcs (int mcs, int nprb);

  /**
   * \brief Get the Transport Block Size for a selected MCS and number of PRB (table 8.6.1-1 of 36.213)
   * \param mcs the MCS index
   * \param nprb the no. of PRB
   * \return the Transport Block Size in bits
   */
  int GetUlTbSizeFromMcs (int mcs, int nprb);

  /**
   * \brief Get the spectral efficiency value associated
   * to the received CQI
   * \param cqi the cqi value
   * \return the spectral efficiency in (bit/s)/Hz
   */
  double GetSpectralEfficiencyFromCqi (int cqi);

  /**
   * \brief Create a message with CQI feedback
   * \param sinr the SpectrumValue vector of SINR for evaluating the CQI
   * \param rbgSize size of RB group (in RBs) for evaluating subband/wideband CQI
   * \return a vector of CQI feedbacks
   */
  std::vector<int> CreateCqiFeedbacks (const SpectrumValue& sinr,
                                                  uint8_t rbgSize = 0);

  /**
   * \brief Get a proper CQI for the spectral efficiency value.
   * In order to assure a lower block error rate, the AMC chooses the lower CQI value
   * for a given spectral efficiency
   * \param s the spectral efficiency
   * \return the CQI value
   */
  int GetCqiFromSpectralEfficiency (double s);
  
private:
  
  /**
   * The `Ber` attribute.
   *
   * The requested BER in assigning MCS (default is 0.00005).
   */
  double m_ber;

  /**
   * The `AmcModel` attribute.
   *
   * AMC model used to assign CQI.
   */
  AmcModel m_amcModel;

}; // end of `class LteAmc`


}

#endif /* AMCMODULE_H */
