/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2011 CTTC
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
 * Modified by Marco Miozzo <mmiozzo@cttc.es> (add data and ctrl diversity)
 */

#include "ns3/log.h"
#include "wifi-ppdu.h"
#include "wifi-spectrum-signal-parameters.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("WifiSpectrumSignalParameters");

WifiSpectrumSignalParameters::WifiSpectrumSignalParameters ()
  : txCenterFreq (0)
{
  NS_LOG_FUNCTION (this);
}

WifiSpectrumSignalParameters::WifiSpectrumSignalParameters (const WifiSpectrumSignalParameters& p)
  : SpectrumSignalParameters (p)
{
  NS_LOG_FUNCTION (this << &p);
  ppdu = p.ppdu;
  txCenterFreq = p.txCenterFreq;
}

Ptr<SpectrumSignalParameters>
WifiSpectrumSignalParameters::Copy ()
{
  NS_LOG_FUNCTION (this);
  // Ideally we would use:
  //   return Copy<WifiSpectrumSignalParameters> (*this);
  // but for some reason it doesn't work. Another alternative is
  //   return Copy<WifiSpectrumSignalParameters> (this);
  // but it causes a double creation of the object, hence it is less efficient.
  // The solution below is copied from the implementation of Copy<> (Ptr<>) in ptr.h
  Ptr<WifiSpectrumSignalParameters> wssp (new WifiSpectrumSignalParameters (*this), false);
  return wssp;
}


} // namespace ns3
