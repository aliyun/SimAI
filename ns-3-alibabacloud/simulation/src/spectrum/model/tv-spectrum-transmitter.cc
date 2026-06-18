/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 University of Washington
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
 * Author: Benjamin Cizdziel <ben.cizdziel@gmail.com>
 */

#include <ns3/log.h>
#include <ns3/simulator.h>
#include <ns3/enum.h>
#include <ns3/uinteger.h>
#include <ns3/double.h>
#include <ns3/integer.h>
#include <ns3/string.h>
#include <ns3/pointer.h>
#include <ns3/isotropic-antenna-model.h>
#include <ns3/antenna-model.h>
#include <cmath>
#include "tv-spectrum-transmitter.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TvSpectrumTransmitter");

NS_OBJECT_ENSURE_REGISTERED (TvSpectrumTransmitter);

TvSpectrumTransmitter::TvSpectrumTransmitter ()
  : m_mobility (0),
    m_antenna (CreateObject<IsotropicAntennaModel> ()),
    m_netDevice (0),
    m_channel (0),
    m_tvType (TVTYPE_8VSB),
    m_startFrequency (500e6),
    m_channelBandwidth (6e6),
    m_basePsd (20),
    m_txPsd (0),
    m_startingTime (Seconds (0)),
    m_transmitDuration (Seconds (0.2)),
    m_active (false)
{
  NS_LOG_FUNCTION (this);
}

TvSpectrumTransmitter::~TvSpectrumTransmitter ()
{
  m_mobility = 0;
  m_antenna = 0;
  m_netDevice = 0;
  m_channel = 0;
  m_txPsd = 0;
  NS_LOG_FUNCTION (this);
}

TypeId
TvSpectrumTransmitter::GetTypeId(void)
{
  static TypeId tid = TypeId("ns3::TvSpectrumTransmitter")
    .SetParent<SpectrumPhy> ()
    .SetGroupName ("Spectrum")
    .AddConstructor<TvSpectrumTransmitter> ()
    .AddAttribute ("TvType",
                   "The type of TV transmitter/modulation to be used.",
                   EnumValue (TvSpectrumTransmitter::TVTYPE_8VSB),
                   MakeEnumAccessor (&TvSpectrumTransmitter::m_tvType),
                   MakeEnumChecker (TvSpectrumTransmitter::TVTYPE_8VSB, "8vsb",
                                    TvSpectrumTransmitter::TVTYPE_COFDM, "cofdm",
                                    TvSpectrumTransmitter::TVTYPE_ANALOG, "analog"))
    .AddAttribute ("StartFrequency",
                   "The lower end frequency (in Hz) of the TV transmitter's "
                   "signal. Must be greater than or equal to 0.",
                   DoubleValue (500e6),
                   MakeDoubleAccessor (&TvSpectrumTransmitter::m_startFrequency),
                   MakeDoubleChecker<double> (0, std::numeric_limits<double>::max ()))
    .AddAttribute ("ChannelBandwidth",
                   "The bandwidth (in Hz) of the TV transmitter's signal. Must "
                   "be greater than or equal to 0.",
                   DoubleValue (6e6),
                   MakeDoubleAccessor (&TvSpectrumTransmitter::m_channelBandwidth),
                   MakeDoubleChecker<double> (0, std::numeric_limits<double>::max ()))
    .AddAttribute ("BasePsd",
                   "The base power spectral density (in dBm/Hz) of the TV "
                   "transmitter's transmitted spectrum. Base PSD is the "
                   "maximum PSD of the spectrum excluding pilots. For analog "
                   "and COFDM transmitters this is the maximum PSD, but for "
                   "8-VSB transmitters this is the maximum PSD of the main "
                   "signal spectrum (flat-top segment) since the pilot "
                   "actually has the maximum PSD overall.",
                   DoubleValue (20),
                   MakeDoubleAccessor (&TvSpectrumTransmitter::m_basePsd),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Antenna",
                   "The AntennaModel to be used. Allows classes inherited "
                   "from ns3::AntennaModel. Defaults to ns3::IsotropicAntennaModel.",
                   StringValue ("ns3::IsotropicAntennaModel"),
                   MakePointerAccessor (&TvSpectrumTransmitter::m_antenna),
                   MakePointerChecker<AntennaModel> ())
    .AddAttribute ("StartingTime",
                   "The time point after the simulation begins in which the TV "
                   "transmitter will begin transmitting.",
                   TimeValue (Seconds (0)),
                   MakeTimeAccessor (&TvSpectrumTransmitter::m_startingTime),
                   MakeTimeChecker ())
    .AddAttribute ("TransmitDuration",
                   "The duration of time that the TV transmitter will transmit for.",
                   TimeValue (Seconds (0.2)),
                   MakeTimeAccessor (&TvSpectrumTransmitter::m_transmitDuration),
                   MakeTimeChecker ())
  ;
  return tid;
}

void
TvSpectrumTransmitter::SetChannel (Ptr<SpectrumChannel> c)
{
  NS_LOG_FUNCTION (this << c);
  m_channel = c;
}

void
TvSpectrumTransmitter::SetMobility (Ptr<MobilityModel> m)
{
  NS_LOG_FUNCTION (this << m);
  m_mobility = m;
}

void
TvSpectrumTransmitter::SetDevice (Ptr<NetDevice> d)
{
  NS_LOG_FUNCTION (this << d);
  m_netDevice = d;
}

Ptr<MobilityModel>
TvSpectrumTransmitter::GetMobility () const
{
  NS_LOG_FUNCTION (this);
  return m_mobility;
}

Ptr<NetDevice>
TvSpectrumTransmitter::GetDevice () const
{
  NS_LOG_FUNCTION (this);
  return m_netDevice;
}

Ptr<const SpectrumModel>
TvSpectrumTransmitter::GetRxSpectrumModel () const
{
  NS_LOG_FUNCTION (this);
  return 0;
}

Ptr<Object>
TvSpectrumTransmitter::GetAntenna () const
{
  NS_LOG_FUNCTION (this);
  return m_antenna;
}

void
TvSpectrumTransmitter::StartRx (Ptr<SpectrumSignalParameters> params)
{
  NS_LOG_FUNCTION (this << params);
}


Ptr<SpectrumChannel>
TvSpectrumTransmitter::GetChannel () const
{
  NS_LOG_FUNCTION (this);
  return m_channel;
}

/// Used as key for map containing created spectrum models
struct TvSpectrumModelId
{
  /**
   * Constructor
   * \param stFreq Start frequency [Hz]
   * \param bwidth Bandwidth [Hz]
   */
  TvSpectrumModelId (double stFreq, double bwidth);
  double startFrequency;  //!< Start frequency [Hz]
  double bandwidth;       //!< Bandwidth [Hz]
};

TvSpectrumModelId::TvSpectrumModelId (double stFreq, double bwidth)
  : startFrequency (stFreq), 
    bandwidth (bwidth)
{
}

/**
 * Minus-than operator
 * \param a left operand
 * \param b right operand
 * \returns true if the left operand has a lower starting frequency
 *          or a smaller bandwidth (if both have the same starting freq.)
 */
bool
operator < (const TvSpectrumModelId& a, const TvSpectrumModelId& b)
{
  return ( (a.startFrequency < b.startFrequency) || 
           ( (a.startFrequency == b.startFrequency) && (a.bandwidth < b.bandwidth) ) );
}

/// Stores created spectrum models
static std::map<TvSpectrumModelId, Ptr<SpectrumModel> > g_tvSpectrumModelMap;

/** 
 * 8-VSB PSD approximated from Figure 3 of the following article:
 * Baron, Stanley. "First-Hand:Digital Television: The Digital Terrestrial 
 * Television Broadcasting (DTTB) Standard." IEEE Global History Network.
 * <http://www.ieeeghn.org/wiki/index.php/First-Hand:Digital_Television:_The_
 * Digital_Terrestrial_Television_Broadcasting_(DTTB)_Standard>.
 *
 * COFDM PSD approximated from Figure 12 (8k mode) of the following article:
 * Kopp, Carlo. "High Definition Television." High Definition Television. Air 
 * Power Australia. <http://www.ausairpower.net/AC-1100.html>.
 *
 * Analog PSD approximated from Figure 4 of the following paper:
 * Stephen Shellhammer, Ahmed Sadek, and Wenyi Zhang. "Technical Challenges for 
 * Cognitive Radio in the TV White Space Spectrum." Qualcomm Incorporated.
 */
void
TvSpectrumTransmitter::CreateTvPsd ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_channelBandwidth != 0);
  Ptr<SpectrumModel> model;
  TvSpectrumModelId key (m_startFrequency, m_channelBandwidth);
  std::map<TvSpectrumModelId, Ptr<SpectrumModel> >::iterator iter = g_tvSpectrumModelMap.find (key);
  if (iter != g_tvSpectrumModelMap.end ())
    {
      model = iter->second; //set SpectrumModel to previously created one
    }
  else // no previously created SpectrumModel with same frequency and bandwidth
    {
      Bands bands;
      double halfSubBand = 0.5 * (m_channelBandwidth / 100);
      for (double fl = m_startFrequency - halfSubBand; fl <= (m_startFrequency - 
           halfSubBand) + m_channelBandwidth; fl += m_channelBandwidth / 100)
        {
          BandInfo bi;
          bi.fl = fl;
          bi.fc = fl + halfSubBand;
          bi.fh = fl + (2 * halfSubBand);
          bands.push_back (bi);
        }
      model = Create<SpectrumModel> (bands);
      g_tvSpectrumModelMap.insert (std::pair<TvSpectrumModelId, Ptr<SpectrumModel> > (key, model));
    }
  Ptr<SpectrumValue> psd = Create<SpectrumValue> (model);
  double basePsdWattsHz = pow (10.0, (m_basePsd - 30) / 10.0); //convert dBm to W/Hz
  switch (m_tvType)
    {
      case TVTYPE_8VSB:
        {
          for (int i = 0; i <= 100; i++)
            {     
              switch (i)
                {  
                  case 0: 
                  case 100: 
                    (*psd) [i] = 0.015 * basePsdWattsHz;
                    break;
                  case 1: 
                  case 99: 
                    (*psd) [i] = 0.019 * basePsdWattsHz;
                    break;
                  case 2: 
                  case 98: 
                    (*psd) [i] = 0.034 * basePsdWattsHz;
                    break;
                  case 3: 
                  case 97: 
                    (*psd) [i] = 0.116 * basePsdWattsHz;
                    break;
                  case 4: 
                  case 96: 
                    (*psd) [i] = 0.309 * basePsdWattsHz;
                    break;
                  case 5: 
                    (*psd) [i] = (0.502 * basePsdWattsHz) + (21.577 * basePsdWattsHz); //pilot
                    break;
                  case 6: 
                  case 94: 
                    (*psd) [i] = 0.696 * basePsdWattsHz;
                    break;
                  case 7: 
                  case 93: 
                    (*psd) [i] = 0.913 * basePsdWattsHz;
                    break;
                  case 8: 
                  case 92: 
                    (*psd) [i] = 0.978 * basePsdWattsHz;
                    break;
                  case 9:
                  case 91:  
                    (*psd) [i] = 0.990 * basePsdWattsHz;
                    break;
                  case 95: 
                    (*psd) [i] = 0.502 * basePsdWattsHz;
                    break;
                  default: 
                    (*psd) [i] = basePsdWattsHz;
                    break;
                }  
            }
          break;
        }
      case TVTYPE_COFDM:
        {
          for (int i = 0; i <= 100; i++)
            {
              switch (i)
                {
                  case 0: 
                  case 100: 
                    (*psd) [i] = 1.52e-4 * basePsdWattsHz;
                    break;
                  case 1: 
                  case 99: 
                    (*psd) [i] = 2.93e-4 * basePsdWattsHz;
                    break;
                  case 2: 
                  case 98: 
                    (*psd) [i] = 8.26e-4 * basePsdWattsHz;
                    break;
                  case 3:
                  case 97:  
                    (*psd) [i] = 0.0927 * basePsdWattsHz;
                    break;
                  default: 
                    (*psd) [i] = basePsdWattsHz;
                    break;
                }
            }
          break;
        }
      case TVTYPE_ANALOG:
        {
          for (int i = 0; i <= 100; i++)
            {
              switch (i)
                {
                  case 0:
                  case 1:
                  case 2:
                  case 3:
                    (*psd) [i] = 27.07946e-08 * basePsdWattsHz;
                    break;
                  case 4:
                  case 5:
                  case 6:
                    (*psd) [i] = 2.51189e-07 * basePsdWattsHz;
                    break;
                  case 7:
                  case 8:
                  case 9:
                    (*psd) [i] = 1e-06 * basePsdWattsHz;
                    break;
                  case 10:
                  case 11:
                  case 12:
                    (*psd) [i] = 2.39883e-06 * basePsdWattsHz;
                    break;
                  case 13:
                  case 14:
                  case 15:
                    (*psd) [i] = 5.62341e-06 * basePsdWattsHz;
                    break;
                  case 16:
                  case 17:
                  case 18:
                    (*psd) [i] = 6.68344e-06 * basePsdWattsHz;
                    break;
                  case 19:
                  case 20:
                  case 21:
                    (*psd) [i] = 1.25893e-05 * basePsdWattsHz;
                    break;
                  case 22:
                  case 23:
                  case 24:
                    (*psd) [i] = 3.16228e-05 * basePsdWattsHz;
                    break;
                  case 25:
                    (*psd) [i] = 0.000158489 * basePsdWattsHz;
                    break;
                  case 26:
                    (*psd) [i] = basePsdWattsHz;
                    break;
                  case 27:
                    (*psd) [i] = 7.49894e-05 * basePsdWattsHz;
                    break;
                  case 28:
                  case 29:
                  case 30:
                    (*psd) [i] = 2.37137e-05 * basePsdWattsHz;
                    break;
                  case 31:
                  case 32:
                  case 33:
                    (*psd) [i] = 1.14815e-05 * basePsdWattsHz;
                    break;
                  case 34:
                  case 35:
                  case 36:
                    (*psd) [i] = 7.49894e-06 * basePsdWattsHz;
                    break;
                  case 37:
                  case 38:
                  case 39:
                    (*psd) [i] = 5.62341e-06 * basePsdWattsHz;
                    break;
                  case 40:
                  case 41:
                  case 42:
                    (*psd) [i] = 4.21697e-06 * basePsdWattsHz;
                    break;
                  case 43:
                  case 44:
                  case 45:
                    (*psd) [i] = 3.16228e-06 * basePsdWattsHz;
                    break;
                  case 46:
                  case 47:
                  case 48:
                    (*psd) [i] = 1.99526e-06 * basePsdWattsHz;
                    break;
                  case 49:
                  case 50:
                  case 51:
                    (*psd) [i] = 1.25893e-06 * basePsdWattsHz;
                    break;
                  case 52:
                  case 53:
                  case 54:
                    (*psd) [i] = 8.41395e-07 * basePsdWattsHz;
                    break;
                  case 55:
                  case 56:
                  case 57:
                    (*psd) [i] = 6.30957e-07 * basePsdWattsHz;
                    break;
                  case 58:
                  case 59:
                  case 60:
                    (*psd) [i] = 5.88844e-07 * basePsdWattsHz;
                    break;
                  case 61:
                  case 62:
                  case 63:
                    (*psd) [i] = 5.62341e-07 * basePsdWattsHz;
                    break;
                  case 64:
                  case 65:
                  case 66:
                    (*psd) [i] = 5.30884e-07 * basePsdWattsHz;
                    break;
                  case 67:
                  case 68:
                  case 69:
                    (*psd) [i] = 5.01187e-07 * basePsdWattsHz;
                    break;
                  case 70:
                  case 71:
                  case 72:
                    (*psd) [i] = 5.30884e-07 * basePsdWattsHz;
                    break;
                  case 73:
                  case 74:
                  case 75:
                    (*psd) [i] = 7.49894e-07 * basePsdWattsHz;
                    break;
                  case 76:
                  case 77:
                  case 78:
                    (*psd) [i] = 1.77828e-06 * basePsdWattsHz;
                    break;
                  case 79:
                    (*psd) [i] = 5.62341e-06 * basePsdWattsHz;
                    break;
                  case 80:
                    (*psd) [i] = 0.000177828 * basePsdWattsHz;
                    break;
                  case 81:
                    (*psd) [i] = 4.21697e-06 * basePsdWattsHz;
                    break;
                  case 82:
                  case 83:
                  case 84:
                    (*psd) [i] = 3.16228e-06 * basePsdWattsHz;
                    break;
                  case 85:
                  case 86:
                  case 87:
                    (*psd) [i] = 3.16228e-06 * basePsdWattsHz;
                    break;
                  case 88:
                  case 89:
                  case 90:
                    (*psd) [i] = 4.73151e-06 * basePsdWattsHz;
                    break;
                  case 91:
                  case 92:
                  case 93:
                    (*psd) [i] = 7.49894e-06 * basePsdWattsHz;
                    break;
                  case 94:
                    (*psd) [i] = 7.49894e-05 * basePsdWattsHz;
                    break;
                  case 95:
                    (*psd) [i] = 0.1 * basePsdWattsHz;
                    break;
                  case 96:
                    (*psd) [i] = 7.49894e-05 * basePsdWattsHz;
                    break;
                  case 97:
                  case 98:
                  case 99:
                  case 100:
                    (*psd) [i] = 1.77828e-06 * basePsdWattsHz;
                    break;
                }
            }
          break;
        }
      default:
        {
          NS_LOG_ERROR ("no valid TvType selected");
          break;
        }
    }
  m_txPsd = psd;
}

Ptr<SpectrumValue>
TvSpectrumTransmitter::GetTxPsd () const
{
  NS_LOG_FUNCTION (this);
  return m_txPsd;
}

void
TvSpectrumTransmitter::SetupTx ()
{
  NS_LOG_FUNCTION (this);
  NS_ASSERT (m_txPsd);
  Ptr<SpectrumSignalParameters> signal = Create<SpectrumSignalParameters> ();
  signal->duration = m_transmitDuration;
  signal->psd = m_txPsd;
  signal->txPhy = GetObject<SpectrumPhy> ();
  signal->txAntenna = m_antenna;
  m_channel->StartTx (signal);
}

void
TvSpectrumTransmitter::Start ()
{
  NS_LOG_FUNCTION (this);
  if (!m_active)
    {
      NS_LOG_LOGIC ("starting TV transmitter");
      m_active = true;
      Simulator::Schedule (m_startingTime, &TvSpectrumTransmitter::SetupTx, this);
    }
}

void
TvSpectrumTransmitter::Stop ()
{
  NS_LOG_FUNCTION (this);
  m_active = false;
}

} // namespace ns3

