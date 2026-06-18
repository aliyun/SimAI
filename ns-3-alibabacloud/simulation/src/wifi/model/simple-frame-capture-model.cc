/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2017
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
 * Author: Sébastien Deronne <sebastien.deronne@gmail.com>
 */

#include "ns3/simulator.h"
#include "ns3/log.h"
#include "ns3/double.h"
#include "simple-frame-capture-model.h"
#include "wifi-utils.h"
#include "interference-helper.h"

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SimpleFrameCaptureModel");

NS_OBJECT_ENSURE_REGISTERED (SimpleFrameCaptureModel);

TypeId
SimpleFrameCaptureModel::GetTypeId (void)
{
  static TypeId tid = TypeId ("ns3::SimpleFrameCaptureModel")
    .SetParent<FrameCaptureModel> ()
    .SetGroupName ("Wifi")
    .AddConstructor<SimpleFrameCaptureModel> ()
    .AddAttribute ("Margin",
                   "Reception is switched if the newly arrived frame has a power higher than "
                   "this value above the frame currently being received (expressed in dB).",
                   DoubleValue (5),
                   MakeDoubleAccessor (&SimpleFrameCaptureModel::GetMargin,
                                       &SimpleFrameCaptureModel::SetMargin),
                   MakeDoubleChecker<double> ())
  ;
  return tid;
}

SimpleFrameCaptureModel::SimpleFrameCaptureModel ()
{
  NS_LOG_FUNCTION (this);
}

SimpleFrameCaptureModel::~SimpleFrameCaptureModel ()
{
  NS_LOG_FUNCTION (this);
}

void
SimpleFrameCaptureModel::SetMargin (double margin)
{
  NS_LOG_FUNCTION (this << margin);
  m_margin = margin;
}

double
SimpleFrameCaptureModel::GetMargin (void) const
{
  return m_margin;
}

bool
SimpleFrameCaptureModel::CaptureNewFrame (Ptr<Event> currentEvent, Ptr<Event> newEvent) const
{
  NS_LOG_FUNCTION (this);
  if ((WToDbm (currentEvent->GetRxPowerW ()) + GetMargin ()) < WToDbm (newEvent->GetRxPowerW ())
      && (IsInCaptureWindow (currentEvent->GetStartTime ())))
    {
      return true;
    }
  return false;
}

} //namespace ns3
