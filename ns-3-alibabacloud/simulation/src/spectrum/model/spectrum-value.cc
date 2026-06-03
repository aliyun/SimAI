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

#include <ns3/spectrum-value.h>
#include <ns3/math.h>
#include <ns3/log.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("SpectrumValue");

SpectrumValue::SpectrumValue ()
{
}

SpectrumValue::SpectrumValue (Ptr<const SpectrumModel> sof)
  : m_spectrumModel (sof),
    m_values (sof->GetNumBands ())
{

}

double&
SpectrumValue::operator[] (size_t index)
{
  return m_values.at (index);
}

const double&
SpectrumValue::operator[] (size_t index) const
{
  return m_values.at (index);
}


SpectrumModelUid_t
SpectrumValue::GetSpectrumModelUid () const
{
  return m_spectrumModel->GetUid ();
}


Ptr<const SpectrumModel>
SpectrumValue::GetSpectrumModel () const
{
  return m_spectrumModel;
}


Values::const_iterator
SpectrumValue::ConstValuesBegin () const
{
  return m_values.begin ();
}

Values::const_iterator
SpectrumValue::ConstValuesEnd () const
{
  return m_values.end ();
}


Values::iterator
SpectrumValue::ValuesBegin ()
{
  return m_values.begin ();
}

Values::iterator
SpectrumValue::ValuesEnd ()
{
  return m_values.end ();
}

Bands::const_iterator
SpectrumValue::ConstBandsBegin () const
{
  return m_spectrumModel->Begin ();
}

Bands::const_iterator
SpectrumValue::ConstBandsEnd () const
{
  return m_spectrumModel->End ();
}


void
SpectrumValue::Add (const SpectrumValue& x)
{
  Values::iterator it1 = m_values.begin ();
  Values::const_iterator it2 = x.m_values.begin ();

  NS_ASSERT (m_spectrumModel == x.m_spectrumModel);
  NS_ASSERT (m_values.size () == x.m_values.size ());

  while (it1 != m_values.end ())
    {
      *it1 += *it2;
      ++it1;
      ++it2;
    }
}


void
SpectrumValue::Add (double s)
{
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 += s;
      ++it1;
    }
}



void
SpectrumValue::Subtract (const SpectrumValue& x)
{
  Values::iterator it1 = m_values.begin ();
  Values::const_iterator it2 = x.m_values.begin ();

  NS_ASSERT (m_spectrumModel == x.m_spectrumModel);
  NS_ASSERT (m_values.size () == x.m_values.size ());

  while (it1 != m_values.end ())
    {
      *it1 -= *it2;
      ++it1;
      ++it2;
    }
}


void
SpectrumValue::Subtract (double s)
{
  Add (-s);
}



void
SpectrumValue::Multiply (const SpectrumValue& x)
{
  Values::iterator it1 = m_values.begin ();
  Values::const_iterator it2 = x.m_values.begin ();

  NS_ASSERT (m_spectrumModel == x.m_spectrumModel);
  NS_ASSERT (m_values.size () == x.m_values.size ());

  while (it1 != m_values.end ())
    {
      *it1 *= *it2;
      ++it1;
      ++it2;
    }
}


void
SpectrumValue::Multiply (double s)
{
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 *= s;
      ++it1;
    }
}




void
SpectrumValue::Divide (const SpectrumValue& x)
{
  Values::iterator it1 = m_values.begin ();
  Values::const_iterator it2 = x.m_values.begin ();

  NS_ASSERT (m_spectrumModel == x.m_spectrumModel);
  NS_ASSERT (m_values.size () == x.m_values.size ());

  while (it1 != m_values.end ())
    {
      *it1 /= *it2;
      ++it1;
      ++it2;
    }
}


void
SpectrumValue::Divide (double s)
{
  NS_LOG_FUNCTION (this << s);
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 /= s;
      ++it1;
    }
}




void
SpectrumValue::ChangeSign ()
{
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 = -(*it1);
      ++it1;
    }
}


void
SpectrumValue::ShiftLeft (int n)
{
  int i = 0;
  while (i < (int) m_values.size () - n)
    {
      m_values.at (i) = m_values.at (i + n);
      i++;
    }
  while (i < (int)m_values.size ())
    {
      m_values.at (i) = 0;
      i++;
    }
}


void
SpectrumValue::ShiftRight (int n)
{
  int i = m_values.size () - 1;
  while (i - n >= 0)
    {
      m_values.at (i) = m_values.at (i - n);
      i = i - 1;
    }
  while (i >= 0)
    {
      m_values.at (i) = 0;
      --i;
    }
}



void
SpectrumValue::Pow (double exp)
{
  NS_LOG_FUNCTION (this << exp);
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 = std::pow (*it1, exp);
      ++it1;
    }
}


void
SpectrumValue::Exp (double base)
{
  NS_LOG_FUNCTION (this << base);
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 = std::pow (base, *it1);
      ++it1;
    }
}


void
SpectrumValue::Log10 ()
{
  NS_LOG_FUNCTION (this);
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 = std::log10 (*it1);
      ++it1;
    }
}

void
SpectrumValue::Log2 ()
{
  NS_LOG_FUNCTION (this);
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 = log2 (*it1);
      ++it1;
    }
}


void
SpectrumValue::Log ()
{
  NS_LOG_FUNCTION (this);
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 = std::log (*it1);
      ++it1;
    }
}

double
Norm (const SpectrumValue& x)
{
  double s = 0;
  Values::const_iterator it1 = x.ConstValuesBegin ();
  while (it1 != x.ConstValuesEnd ())
    {
      s += (*it1) * (*it1);
      ++it1;
    }
  return std::sqrt (s);
}


double
Sum (const SpectrumValue& x)
{
  double s = 0;
  Values::const_iterator it1 = x.ConstValuesBegin ();
  while (it1 != x.ConstValuesEnd ())
    {
      s += (*it1);
      ++it1;
    }
  return s;
}



double
Prod (const SpectrumValue& x)
{
  double s = 0;
  Values::const_iterator it1 = x.ConstValuesBegin ();
  while (it1 != x.ConstValuesEnd ())
    {
      s *= (*it1);
      ++it1;
    }
  return s;
}

double
Integral (const SpectrumValue& arg)
{
  double i = 0;
  Values::const_iterator vit = arg.ConstValuesBegin ();
  Bands::const_iterator bit = arg.ConstBandsBegin ();
  while (vit != arg.ConstValuesEnd ())
    {
      NS_ASSERT (bit != arg.ConstBandsEnd ());
      i += (*vit) * (bit->fh - bit->fl);
      ++vit;
      ++bit;
    }
  NS_ASSERT (bit == arg.ConstBandsEnd ());
  return i;
}



Ptr<SpectrumValue>
SpectrumValue::Copy () const
{
  Ptr<SpectrumValue> p = Create<SpectrumValue> (m_spectrumModel);
  *p = *this;
  return p;

  //  return Copy<SpectrumValue> (*this)
}


/**
 * \brief Output stream operator
 * \param os output stream
 * \param pvf the SpectrumValue to print
 * \return an output stream
 */
std::ostream&
operator << (std::ostream& os, const SpectrumValue& pvf)
{
  Values::const_iterator it1 = pvf.ConstValuesBegin ();
  while (it1 != pvf.ConstValuesEnd ())
    {
      os << *it1 << " ";
      ++it1;
    }
  os << std::endl;
  return os;
}



SpectrumValue
operator+ (const SpectrumValue& lhs, const SpectrumValue& rhs)
{
  SpectrumValue res = lhs;
  res.Add (rhs);
  return res;
}


SpectrumValue
operator+ (const SpectrumValue& lhs, double rhs)
{
  SpectrumValue res = lhs;
  res.Add (rhs);
  return res;
}


SpectrumValue
operator+ (double lhs, const SpectrumValue& rhs)
{
  SpectrumValue res = rhs;
  res.Add (lhs);
  return res;
}


SpectrumValue
operator- (const SpectrumValue& lhs, const SpectrumValue& rhs)
{
  SpectrumValue res = rhs;
  res.ChangeSign ();
  res.Add (lhs);
  return res;
}



SpectrumValue
operator- (const SpectrumValue& lhs, double rhs)
{
  SpectrumValue res = lhs;
  res.Subtract (rhs);
  return res;
}


SpectrumValue
operator- (double lhs, const SpectrumValue& rhs)
{
  SpectrumValue res = rhs;
  res.Subtract (lhs);
  return res;
}

SpectrumValue
operator* (const SpectrumValue& lhs, const SpectrumValue& rhs)
{
  SpectrumValue res = lhs;
  res.Multiply (rhs);
  return res;
}


SpectrumValue
operator* (const SpectrumValue& lhs, double rhs)
{
  SpectrumValue res = lhs;
  res.Multiply (rhs);
  return res;
}


SpectrumValue
operator* (double lhs, const SpectrumValue& rhs)
{
  SpectrumValue res = rhs;
  res.Multiply (lhs);
  return res;
}


SpectrumValue
operator/ (const SpectrumValue& lhs, const SpectrumValue& rhs)
{
  SpectrumValue res = lhs;
  res.Divide (rhs);
  return res;
}


SpectrumValue
operator/ (const SpectrumValue& lhs, double rhs)
{
  SpectrumValue res = lhs;
  res.Divide (rhs);
  return res;
}


SpectrumValue
operator/ (double lhs, const SpectrumValue& rhs)
{
  SpectrumValue res = rhs;
  res.Divide (lhs);
  return res;
}


SpectrumValue
operator+ (const SpectrumValue& rhs)
{
  return rhs;
}

SpectrumValue
operator- (const SpectrumValue& rhs)
{
  SpectrumValue res = rhs;
  res.ChangeSign ();
  return res;
}


SpectrumValue
Pow (double lhs, const SpectrumValue& rhs)
{
  SpectrumValue res = rhs;
  res.Exp (lhs);
  return res;
}


SpectrumValue
Pow (const SpectrumValue& lhs, double rhs)
{
  SpectrumValue res = lhs;
  res.Pow (rhs);
  return res;
}


SpectrumValue
Log10 (const SpectrumValue& arg)
{
  SpectrumValue res = arg;
  res.Log10 ();
  return res;
}

SpectrumValue
Log2 (const SpectrumValue& arg)
{
  SpectrumValue res = arg;
  res.Log2 ();
  return res;
}

SpectrumValue
Log (const SpectrumValue& arg)
{
  SpectrumValue res = arg;
  res.Log ();
  return res;
}

SpectrumValue&
SpectrumValue::operator+= (const SpectrumValue& rhs)
{
  Add (rhs);
  return *this;
}

SpectrumValue&
SpectrumValue::operator-= (const SpectrumValue& rhs)
{
  Subtract (rhs);
  return *this;
}

SpectrumValue&
SpectrumValue::operator*= (const SpectrumValue& rhs)
{
  Multiply (rhs);
  return *this;
}

SpectrumValue&
SpectrumValue::operator/= (const SpectrumValue& rhs)
{
  Divide (rhs);
  return *this;
}


SpectrumValue&
SpectrumValue::operator+= (double rhs)
{
  Add (rhs);
  return *this;
}

SpectrumValue&
SpectrumValue::operator-= (double rhs)
{
  Subtract (rhs);
  return *this;
}

SpectrumValue&
SpectrumValue::operator*= (double rhs)
{
  Multiply (rhs);
  return *this;
}

SpectrumValue&
SpectrumValue::operator/= (double rhs)
{
  Divide (rhs);
  return *this;
}


SpectrumValue&
SpectrumValue::operator= (double rhs)
{
  Values::iterator it1 = m_values.begin ();

  while (it1 != m_values.end ())
    {
      *it1 = rhs;
      ++it1;
    }
  return *this;
}



SpectrumValue
SpectrumValue::operator<< (int n) const
{
  SpectrumValue res = *this;
  res.ShiftLeft (n);
  return res;
}

SpectrumValue
SpectrumValue::operator>> (int n) const
{
  SpectrumValue res = *this;
  res.ShiftRight (n);
  return res;
}

uint32_t
SpectrumValue::GetValuesN () const
{
  return m_values.size ();
}

const double &
SpectrumValue::ValuesAt (uint32_t pos) const
{
  return m_values.at (pos);
}

} // namespace ns3

