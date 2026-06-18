/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2013 Fraunhofer FKIE
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
 * Author:
 *  Sascha Alexander Jopen <jopen@cs.uni-bonn.de>
 */
#include "lr-wpan-interference-helper.h"
#include <ns3/spectrum-value.h>
#include <ns3/spectrum-model.h>
#include <ns3/log.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("LrWpanInterferenceHelper");

LrWpanInterferenceHelper::LrWpanInterferenceHelper (Ptr<const SpectrumModel> spectrumModel)
  : m_spectrumModel (spectrumModel),
    m_dirty (false)
{
  m_signal = Create<SpectrumValue> (m_spectrumModel);
}

LrWpanInterferenceHelper::~LrWpanInterferenceHelper (void)
{
  m_spectrumModel = 0;
  m_signal = 0;
  m_signals.clear ();
}

bool
LrWpanInterferenceHelper::AddSignal (Ptr<const SpectrumValue> signal)
{
  NS_LOG_FUNCTION (this << signal);

  bool result = false;

  if (signal->GetSpectrumModel () == m_spectrumModel)
    {
      result = m_signals.insert (signal).second;
      if (result && !m_dirty)
        {
          *m_signal += *signal;
        }
    }
  return result;
}

bool
LrWpanInterferenceHelper::RemoveSignal (Ptr<const SpectrumValue> signal)
{
  NS_LOG_FUNCTION (this << signal);

  bool result = false;

  if (signal->GetSpectrumModel () == m_spectrumModel)
    {
      result = (m_signals.erase (signal) == 1);
      if (result)
        {
          m_dirty = true;
        }
    }
  return result;
}

void
LrWpanInterferenceHelper::ClearSignals (void)
{
  NS_LOG_FUNCTION (this);

  m_signals.clear ();
  m_dirty = true;
}

Ptr<SpectrumValue>
LrWpanInterferenceHelper::GetSignalPsd (void) const
{
  NS_LOG_FUNCTION (this);

  if (m_dirty)
    {
      // Sum up the current interference PSD.
      std::set<Ptr<const SpectrumValue> >::const_iterator it;
      m_signal = Create<SpectrumValue> (m_spectrumModel);
      for (it = m_signals.begin (); it != m_signals.end (); ++it)
        {
          *m_signal += *(*it);
        }
      m_dirty = false;
    }

  return m_signal->Copy ();
}

}
