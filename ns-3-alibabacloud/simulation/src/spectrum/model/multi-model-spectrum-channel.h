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


#ifndef MULTI_MODEL_SPECTRUM_CHANNEL_H
#define MULTI_MODEL_SPECTRUM_CHANNEL_H

#include <ns3/spectrum-value.h>
#include <ns3/spectrum-converter.h>
#include <ns3/spectrum-channel.h>
#include <ns3/spectrum-propagation-loss-model.h>
#include <ns3/propagation-delay-model.h>
#include <map>
#include <set>

namespace ns3 {


/**
 * \ingroup spectrum
 * Container: SpectrumModelUid_t, SpectrumConverter
 */
typedef std::map<SpectrumModelUid_t, SpectrumConverter> SpectrumConverterMap_t;

/**
 * \ingroup spectrum
 * The Tx spectrum model information. This class is used to convert
 * one spectrum model into another one.
 */
class TxSpectrumModelInfo
{
public:
  /**
   * Constructor.
   * \param txSpectrumModel the Tx Spectrum model.
   */
  TxSpectrumModelInfo (Ptr<const SpectrumModel> txSpectrumModel);

  Ptr<const SpectrumModel> m_txSpectrumModel;     //!< Tx Spectrum model.
  SpectrumConverterMap_t m_spectrumConverterMap;  //!< Spectrum converter.
};


/**
 * \ingroup spectrum
 * Container: SpectrumModelUid_t, TxSpectrumModelInfo
 */
typedef std::map<SpectrumModelUid_t, TxSpectrumModelInfo> TxSpectrumModelInfoMap_t;


/**
 * \ingroup spectrum
 * The Rx spectrum model information. This class is used to convert
 * one spectrum model into another one.
 */
class RxSpectrumModelInfo
{
public:
  /**
   * Constructor.
   * \param rxSpectrumModel the Rx Spectrum model.
   */
  RxSpectrumModelInfo (Ptr<const SpectrumModel> rxSpectrumModel);

  Ptr<const SpectrumModel> m_rxSpectrumModel;  //!< Rx Spectrum model.
  std::vector<Ptr<SpectrumPhy> > m_rxPhys;     //!< Container of the Rx Spectrum phy objects.
};

/**
 * \ingroup spectrum
 * Container: SpectrumModelUid_t, RxSpectrumModelInfo
 */
typedef std::map<SpectrumModelUid_t, RxSpectrumModelInfo> RxSpectrumModelInfoMap_t;


/**
 * \ingroup spectrum
 *
 * This SpectrumChannel implementation can handle the presence of
 * SpectrumPhy instances which can use
 * different spectrum models, i.e.,  different SpectrumModel. 
 *
 * \note It is allowed for a receiving SpectrumPhy to switch to a
 * different SpectrumModel during the simulation. The requirement
 * for this to work is that, after the SpectrumPhy switched its
 * SpectrumModel,  MultiModelSpectrumChannel::AddRx () is
 * called again passing the pointer to that SpectrumPhy.
 */
class MultiModelSpectrumChannel : public SpectrumChannel
{

public:
  MultiModelSpectrumChannel ();

  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  // inherited from SpectrumChannel
  virtual void RemoveRx (Ptr<SpectrumPhy> phy);
  virtual void AddRx (Ptr<SpectrumPhy> phy);
  virtual void StartTx (Ptr<SpectrumSignalParameters> params);


  // inherited from Channel
  virtual std::size_t GetNDevices (void) const;
  virtual Ptr<NetDevice> GetDevice (std::size_t i) const;


protected:
  void DoDispose ();

private:
  /**
   * This method checks if m_rxSpectrumModelInfoMap contains an entry
   * for the given TX SpectrumModel. If such entry exists, it returns
   * an iterator pointing to it. If not, it creates a new entry in
   * m_txSpectrumMpodelInfoMap, and returns an iterator to it.
   *
   * \param txSpectrumModel The TX SpectrumModel  being considered
   *
   * \return An iterator pointing to the corresponding entry in m_txSpectrumModelInfoMap
   */
  TxSpectrumModelInfoMap_t::const_iterator FindAndEventuallyAddTxSpectrumModel (Ptr<const SpectrumModel> txSpectrumModel);

  /**
   * Used internally to reschedule transmission after the propagation delay.
   *
   * \param params The signal parameters.
   * \param receiver A pointer to the receiver SpectrumPhy.
   */
  virtual void StartRx (Ptr<SpectrumSignalParameters> params, Ptr<SpectrumPhy> receiver);

  /**
   * Data structure holding, for each TX SpectrumModel,  all the
   * converters to any RX SpectrumModel, and all the corresponding
   * SpectrumPhy instances.
   *
   */
  TxSpectrumModelInfoMap_t m_txSpectrumModelInfoMap;


  /**
   * Data structure holding, for each RX spectrum model, all the
   * corresponding SpectrumPhy instances.
   */
  RxSpectrumModelInfoMap_t m_rxSpectrumModelInfoMap;

  /**
   * Number of devices connected to the channel.
   */
  std::size_t m_numDevices;

};



}



#endif /* MULTI_MODEL_SPECTRUM_CHANNEL_H */
