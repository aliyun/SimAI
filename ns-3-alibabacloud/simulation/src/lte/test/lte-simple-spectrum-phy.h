/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Piotr Gawlowicz
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
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 *
 */

#ifndef LTE_SIMPLE_SPECTRUM_PHY_H
#define LTE_SIMPLE_SPECTRUM_PHY_H

#include <ns3/event-id.h>
#include <ns3/spectrum-value.h>
#include <ns3/mobility-model.h>
#include <ns3/net-device.h>
#include <ns3/spectrum-phy.h>
#include <ns3/spectrum-channel.h>
#include <ns3/traced-callback.h>

namespace ns3 {

/**
 * \ingroup lte
 *
 * The LteSimpleSpectrumPhy models the physical layer of LTE
 * This class is used to test Frequency Reuse Algorithms,
 * it allow to get SpectrumValue from channel and pass it to
 * test script by trace mechanism.
 * When m_cellId is 0, all received signals will be traced,
 * if m_cellId > 0, only signals from specified Cell will be traced.
 *
 */

class LteSimpleSpectrumPhy : public SpectrumPhy
{

public:
  LteSimpleSpectrumPhy ();
  virtual ~LteSimpleSpectrumPhy ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  // inherited from Object
  virtual void DoDispose ();

  // inherited from SpectrumPhy
  void SetChannel (Ptr<SpectrumChannel> c);
  void SetMobility (Ptr<MobilityModel> m);
  void SetDevice (Ptr<NetDevice> d);
  Ptr<MobilityModel> GetMobility () const;
  Ptr<NetDevice> GetDevice () const;
  Ptr<const SpectrumModel> GetRxSpectrumModel () const;
  Ptr<Object> GetAntenna () const;
  void StartRx (Ptr<SpectrumSignalParameters> params);

  /**
   * \brief Set receive spectrum model.
   * \param model the spectrum model
   */
  void SetRxSpectrumModel (Ptr<const SpectrumModel> model);

  /**
   * \brief Set cell ID.
   * \param cellId the cell ID
   */
  void SetCellId (uint16_t cellId);

private:
  Ptr<MobilityModel> m_mobility; ///< the mobility model
  Ptr<AntennaModel> m_antenna; ///< the antenna model
  Ptr<NetDevice> m_device; ///< the device
  Ptr<SpectrumChannel> m_channel; ///< the channel
  Ptr<const SpectrumModel> m_rxSpectrumModel; ///< the spectrum model

  uint16_t m_cellId; ///< the cell ID

  TracedCallback< Ptr<const SpectrumValue> > m_rxStart; ///< receive start trace callback function
};






}

#endif /* LTE_SIMPLE_SPECTRUM_PHY_H */
