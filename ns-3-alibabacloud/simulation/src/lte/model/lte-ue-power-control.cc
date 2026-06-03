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

#include "lte-ue-power-control.h"
#include <ns3/log.h>
#include <ns3/boolean.h>
#include <ns3/double.h>
#include <ns3/integer.h>
#include <ns3/math.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LteUePowerControl");

NS_OBJECT_ENSURE_REGISTERED (LteUePowerControl);

LteUePowerControl::LteUePowerControl ()
{
  NS_LOG_FUNCTION (this);
  m_deltaTF = 0;
  m_fc = 0;
  m_pathLoss = 100; //initial value
  m_curPuschTxPower = 10;
  m_curPucchTxPower = 10;
  m_curSrsTxPower = 10;
  m_txPower = 10;

  m_cellId = 0;
  m_rnti = 0;

  m_M_Pusch = 0;
  m_rsrpSet = false;
  m_pcRsrpFilterCoefficient = 4; //Default value similar to the eNB (see lte-enb-rrc.cc)
}

LteUePowerControl::~LteUePowerControl ()
{
  NS_LOG_FUNCTION (this);
}

void
LteUePowerControl::DoInitialize ()
{
  NS_LOG_FUNCTION (this);
}

void
LteUePowerControl::DoDispose ()
{
  NS_LOG_FUNCTION (this);
}

TypeId
LteUePowerControl::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::LteUePowerControl")
    .SetParent<Object> ()
    .SetGroupName("Lte")
    .AddConstructor<LteUePowerControl> ()
    .AddAttribute ("ClosedLoop",
                   "If true Closed Loop mode will be active, otherwise Open Loop",
                   BooleanValue (true),
                   MakeBooleanAccessor (&LteUePowerControl::m_closedLoop),
                   MakeBooleanChecker ())
    .AddAttribute ("AccumulationEnabled",
                   "If true TCP accumulation mode will be active, otherwise absolute mode will be active",
                   BooleanValue (true),
                   MakeBooleanAccessor (&LteUePowerControl::m_accumulationEnabled),
                   MakeBooleanChecker ())
    .AddAttribute ("Alpha",
                   "Value of Alpha parameter",
                   DoubleValue (1.0),
                   MakeDoubleAccessor (&LteUePowerControl::SetAlpha),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Pcmax",
                   "Max Transmission power in dBm, Default value 23 dBm"
                   "TS36.101 section 6.2.3",
                   DoubleValue (23.0),
                   MakeDoubleAccessor (&LteUePowerControl::m_Pcmax),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("Pcmin",
                   "Min Transmission power in dBm, Default value -40 dBm"
                   "TS36.101 section 6.2.3",
                   DoubleValue (-40),
                   MakeDoubleAccessor (&LteUePowerControl::m_Pcmin),
                   MakeDoubleChecker<double> ())
    .AddAttribute ("PoNominalPusch",
                   "P_O_NOMINAL_PUSCH   INT (-126 ... 24), Default value -80",
                   IntegerValue (-80),
                   MakeIntegerAccessor (&LteUePowerControl::SetPoNominalPusch),
                   MakeIntegerChecker<int16_t> ())
    .AddAttribute ("PoUePusch",
                   "P_O_UE_PUSCH   INT(-8...7), Default value 0",
                   IntegerValue (0),
                   MakeIntegerAccessor (&LteUePowerControl::SetPoUePusch),
                   MakeIntegerChecker<int16_t> ())
    .AddAttribute ("PsrsOffset",
                   "P_SRS_OFFSET   INT(0...15), Default value 7",
                   IntegerValue (7),
                   MakeIntegerAccessor (&LteUePowerControl::m_PsrsOffset),
                   MakeIntegerChecker<int16_t> ())
    .AddTraceSource ("ReportPuschTxPower",
                     "Report PUSCH TxPower in dBm",
                     MakeTraceSourceAccessor (&LteUePowerControl::m_reportPuschTxPower),
                     "ns3::LteUePowerControl::TxPowerTracedCallback")
    .AddTraceSource ("ReportPucchTxPower",
                     "Report PUCCH TxPower in dBm",
                     MakeTraceSourceAccessor (&LteUePowerControl::m_reportPucchTxPower),
                     "ns3::LteUePowerControl::TxPowerTracedCallback")
    .AddTraceSource ("ReportSrsTxPower",
                     "Report SRS TxPower in dBm",
                     MakeTraceSourceAccessor (&LteUePowerControl::m_reportSrsTxPower),
                     "ns3::LteUePowerControl::TxPowerTracedCallback")
  ;
  return tid;
}

void
LteUePowerControl::SetPcmax (double value)
{
  NS_LOG_FUNCTION (this);
  m_Pcmax = value;
}

double
LteUePowerControl::GetPcmax ()
{
  NS_LOG_FUNCTION (this);
  return m_Pcmax;
}

void
LteUePowerControl::SetTxPower (double value)
{
  NS_LOG_FUNCTION (this);
  m_txPower = value;
  m_curPuschTxPower = value;
  m_curPucchTxPower = value;
  m_curSrsTxPower = value;
}

void
LteUePowerControl::ConfigureReferenceSignalPower (int8_t referenceSignalPower)
{
  NS_LOG_FUNCTION (this);
  m_referenceSignalPower = referenceSignalPower;
}

void
LteUePowerControl::SetCellId (uint16_t cellId)
{
  NS_LOG_FUNCTION (this);
  m_cellId = cellId;
}
void
LteUePowerControl::SetRnti (uint16_t rnti)
{
  NS_LOG_FUNCTION (this);
  m_rnti = rnti;
}

void
LteUePowerControl::SetPoNominalPusch (int16_t value)
{
  NS_LOG_FUNCTION (this);

  if (m_PoNominalPusch.empty ())
    {
      m_PoNominalPusch.push_back (value);
      m_PoNominalPusch.push_back (value);
      m_PoNominalPusch.push_back (value);
    }
  else
    {
      m_PoNominalPusch[0] = value;
      m_PoNominalPusch[1] = value;
      m_PoNominalPusch[2] = value;
    }

}
void
LteUePowerControl::SetPoUePusch (int16_t value)
{
  NS_LOG_FUNCTION (this);
  if (m_PoUePusch.empty ())
    {
      m_PoUePusch.push_back (value);
      m_PoUePusch.push_back (value);
      m_PoUePusch.push_back (0);
    }
  else
    {
      m_PoUePusch[0] = value;
      m_PoUePusch[1] = value;
      m_PoUePusch[2] = 0;
    }
}
void
LteUePowerControl::SetAlpha (double value)
{
  NS_LOG_FUNCTION (this);

  uint32_t temp = value * 10;
  switch (temp)
    {
    case 0:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 10:
      break;
    default:
      NS_FATAL_ERROR ("Unexpected Alpha value");
    }

  if (m_alpha.empty ())
    {
      m_alpha.push_back (value);
      m_alpha.push_back (value);
      m_alpha.push_back (0);
    }
  else
    {
      m_alpha[0] = value;
      m_alpha[1] = value;
      m_alpha[2] = 1;
    }

}

void
LteUePowerControl::SetRsrp (double value)
{
  NS_LOG_FUNCTION (this);

  if (!m_rsrpSet)
    {
      m_rsrp = value;
      m_rsrpSet = true;
      return;
    }

  double alphaRsrp = std::pow (0.5, m_pcRsrpFilterCoefficient / 4.0);
  m_rsrp = (1 - alphaRsrp) * m_rsrp + alphaRsrp * value;

  m_pathLoss = m_referenceSignalPower - m_rsrp;
}

void
LteUePowerControl::SetRsrpFilterCoefficient (uint8_t rsrpFilterCoefficient)
{
  NS_LOG_FUNCTION (this);
  m_pcRsrpFilterCoefficient = rsrpFilterCoefficient;
}

void
LteUePowerControl::ReportTpc (uint8_t tpc)
{
  NS_LOG_FUNCTION (this);

  int delta = 0;
  if (m_accumulationEnabled)
    {
      switch (tpc)
        {
        case 0:
          delta = -1;
          break;
        case 1:
          delta = 0;
          break;
        case 2:
          delta = 1;
          break;
        case 3:
          delta = 3;
          break;
        default:
          NS_FATAL_ERROR ("Unexpected TPC value");
        }
    }
  else
    {
      switch (tpc)
        {
        case 0:
          delta = -4;
          break;
        case 1:
          delta = -1;
          break;
        case 2:
          delta = 1;
          break;
        case 3:
          delta = 4;
          break;
        default:
          NS_FATAL_ERROR ("Unexpected TPC value");
        }
    }

  m_deltaPusch.push_back (delta);

  if (m_closedLoop)
    {
      if (m_accumulationEnabled)
        {
          if (m_deltaPusch.size () == 4)
            {
              if ((m_curPuschTxPower <= m_Pcmin && m_deltaPusch.at (0) < 0)
                  || (m_curPuschTxPower >= m_Pcmax && m_deltaPusch.at (0) > 0))
                {
                  //TPC commands for serving cell shall not be accumulated
                  m_deltaPusch.erase (m_deltaPusch.begin ());
                }
              else
                {
                  m_fc = m_fc + m_deltaPusch.at (0);
                  m_deltaPusch.erase (m_deltaPusch.begin ());
                }
            }
          else
            {
              m_fc = 0;
            }
        }
      else
        {
          m_fc = m_deltaPusch.at (0);
          m_deltaPusch.erase (m_deltaPusch.begin ());
        }
    }
  else
    {
      m_fc = 0;
    }

  NS_LOG_INFO ("ReportTpc: " << (int)tpc << " delta: " << delta << " Fc: " << m_fc);
}

void
LteUePowerControl::SetSubChannelMask (std::vector <int> mask)
{
  NS_LOG_FUNCTION (this);
  m_M_Pusch = mask.size ();
}

void
LteUePowerControl::CalculatePuschTxPower ()
{
  NS_LOG_FUNCTION (this);
  int32_t j = 1;
  int32_t PoPusch = m_PoNominalPusch[j] + m_PoUePusch[j];

  NS_LOG_INFO ("RB: " << m_M_Pusch << " m_PoPusch: " << PoPusch
                      << " Alpha: " << m_alpha[j] << " PathLoss: " << m_pathLoss
                      << " deltaTF: " << m_deltaTF << " fc: " << m_fc);

  if ( m_M_Pusch > 0 )
    {
      m_curPuschTxPower = 10 * log10 (1.0 * m_M_Pusch) + PoPusch + m_alpha[j] * m_pathLoss + m_deltaTF + m_fc;
      m_M_Pusch = 0;
    }
  else
    {
      m_curPuschTxPower = PoPusch + m_alpha[j] * m_pathLoss + m_fc;
    }

  NS_LOG_INFO ("CalcPower: " << m_curPuschTxPower << " MinPower: " << m_Pcmin << " MaxPower:" << m_Pcmax);

  m_curPuschTxPower = m_curPuschTxPower > m_Pcmin ? m_curPuschTxPower : m_Pcmin;
  m_curPuschTxPower = m_Pcmax < m_curPuschTxPower ? m_Pcmax : m_curPuschTxPower;
  NS_LOG_INFO ("PuschTxPower: " << m_curPuschTxPower);
}

void
LteUePowerControl::CalculatePucchTxPower ()
{
  NS_LOG_FUNCTION (this);
  m_curPucchTxPower = m_curPuschTxPower;
  NS_LOG_INFO ("PucchTxPower: " << m_curPucchTxPower);
}

void
LteUePowerControl::CalculateSrsTxPower ()
{
  NS_LOG_FUNCTION (this);
  int32_t j = 1;
  int32_t PoPusch = m_PoNominalPusch[j] + m_PoUePusch[j];

  NS_LOG_INFO ("RB: " << m_M_Pusch << " m_PoPusch: " << PoPusch
                      << " Alpha: " << m_alpha[j] << " PathLoss: " << m_pathLoss
                      << " deltaTF: " << m_deltaTF << " fc: " << m_fc);


  double pSrsOffsetValue = -10.5 + m_PsrsOffset * 1.5;

  m_curSrsTxPower = pSrsOffsetValue + 10 * log10 (m_srsBandwidth) + PoPusch + m_alpha[j] * m_pathLoss + m_fc;

  NS_LOG_INFO ("CalcPower: " << m_curSrsTxPower << " MinPower: " << m_Pcmin << " MaxPower:" << m_Pcmax);

  m_curSrsTxPower = m_curSrsTxPower > m_Pcmin ? m_curSrsTxPower : m_Pcmin;
  m_curSrsTxPower = m_Pcmax < m_curSrsTxPower ? m_Pcmax : m_curSrsTxPower;
  NS_LOG_INFO ("SrsTxPower: " << m_curSrsTxPower);
}


double
LteUePowerControl::GetPuschTxPower (std::vector <int> dlRb)
{
  NS_LOG_FUNCTION (this);

  m_M_Pusch = dlRb.size ();
  CalculatePuschTxPower ();

  m_reportPuschTxPower (m_cellId, m_rnti, m_curPuschTxPower);

  return m_curPuschTxPower;
}

double
LteUePowerControl::GetPucchTxPower (std::vector <int> dlRb)
{
  NS_LOG_FUNCTION (this);

  CalculatePucchTxPower ();

  m_reportPucchTxPower (m_cellId, m_rnti, m_curPucchTxPower);

  return m_curPucchTxPower;
}

double
LteUePowerControl::GetSrsTxPower (std::vector <int> dlRb)
{
  NS_LOG_FUNCTION (this);

  m_srsBandwidth = dlRb.size ();
  CalculateSrsTxPower ();

  m_reportSrsTxPower (m_cellId, m_rnti, m_curSrsTxPower);

  return m_curSrsTxPower;
}

} // namespace ns3
