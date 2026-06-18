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

#ifndef SPECTRUM_VALUE_H
#define SPECTRUM_VALUE_H

#include <ns3/ptr.h>
#include <ns3/simple-ref-count.h>
#include <ns3/spectrum-model.h>
#include <ostream>
#include <vector>

namespace ns3 {


/// Container for element values
typedef std::vector<double> Values;

/**
 * \ingroup spectrum
 *
 * \brief Set of values corresponding to a given SpectrumModel
 *
 * This class implements a Function Space which can represent any
 * function \f$ g: F \in {\sf
 * R\hspace*{-0.9ex}\rule{0.15ex}{1.5ex}\hspace*{0.9ex}}^N \rightarrow {\sf R\hspace*{-0.9ex}\rule{0.15ex}{1.5ex}\hspace*{0.9ex}}  \f$
 *
 * Every instance of this class represent a particular function \f$ g(F) \f$.
 * The domain of the function space, i.e., \f$ F \f$, is implemented by Bands.
 * The codomain of the function space is implemented by Values.
 *
 * To every possible value of \f$ F\f$ corresponds a different Function
 * Space.
 * Mathematical operations are defined in this Function Space; these
 * operations are implemented by means of operator overloading.
 *
 * The intended use of this class is to represent frequency-dependent
 * things, such as power spectral densities, frequency-dependent
 * propagation losses, spectral masks, etc.
 */
class SpectrumValue : public SimpleRefCount<SpectrumValue>
{
public:
  /**
   * @brief SpectrumValue constructor
   *
   * @param sm pointer to the SpectrumModel which implements the set of frequencies to which the values
   * will be referring.
   *
   * @warning the intended use if that sm points to a static object
   * which will be there for the whole simulation. This is reasonable
   * since the set of frequencies which are to be used in the
   * simulation is normally known before the simulation starts. Make
   * sure that the SpectrumModel instance which sm points to has already been
   * initialized by the time this method is invoked. The main reason is
   * that if you initialize the SpectrumModel instance afterwards, and
   * the memory for the underlying std::vector gets reallocated, then
   * sm will not be a valid reference anymore. Another reason is that
   * m_values could end up having the wrong size.
   */
  SpectrumValue (Ptr<const SpectrumModel> sm);


  SpectrumValue ();


  /**
   * Access value at given frequency index
   *
   * @param index the given frequency index
   *
   * @return reference to the value
   */
  double& operator[] (size_t index);

  /**
   * Access value at given frequency index
   *
   * @param index the given frequency index
   *
   * @return const reference to the value
   */
  const double& operator[] (size_t index) const;


  /**
   *
   * @return the uid of the embedded SpectrumModel
   */
  SpectrumModelUid_t GetSpectrumModelUid () const;


  /**
   *
   * @return the  embedded SpectrumModel
   */
  Ptr<const SpectrumModel> GetSpectrumModel () const;


  /**
   *
   *
   * @return a const iterator pointing to the beginning of the embedded Bands
   */
  Bands::const_iterator ConstBandsBegin () const;

  /**
   *
   *
   * @return a const iterator pointing to the end of the embedded Bands
   */
  Bands::const_iterator ConstBandsEnd () const;


  /**
   *
   *
   * @return a const iterator pointing to the beginning of the embedded Values
   */
  Values::const_iterator ConstValuesBegin () const;

  /**
   *
   *
   * @return a const iterator pointing to the end of the embedded Values
   */
  Values::const_iterator ConstValuesEnd () const;

  /**
   *
   *
   * @return an iterator pointing to the beginning of the embedded Values
   */
  Values::iterator ValuesBegin ();

  /**
   *
   *
   * @return an iterator pointing to the end of the embedded Values
   */
  Values::iterator ValuesEnd ();

  /**
   * \brief Get the number of values stored in the array
   * \return the values array size
   */
  uint32_t GetValuesN () const;

  /**
   * \brief Get the value element at the position
   * \param pos position
   * \return the value element in that position (with bounds checking)
   */
  const double & ValuesAt (uint32_t pos) const;

  /**
   *  addition operator
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs + rhs
   */
  friend SpectrumValue operator+ (const SpectrumValue& lhs, const SpectrumValue& rhs);


  /**
   *  addition operator
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs + rhs
   */
  friend SpectrumValue operator+ (const SpectrumValue& lhs, double rhs);

  /**
   *  addition operator
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs + rhs
   */
  friend SpectrumValue operator+ (double lhs, const SpectrumValue& rhs);


  /**
   *  subtraction operator
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs - rhs
   */
  friend SpectrumValue operator- (const SpectrumValue& lhs, const SpectrumValue& rhs);

  /**
   *  subtraction operator
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs - rhs
   */
  friend SpectrumValue operator- (const SpectrumValue& lhs, double rhs);

  /**
   *  subtraction operator
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs - rhs
   */
  friend SpectrumValue operator- (double lhs, const SpectrumValue& rhs);

  /**
   *  multiplication component-by-component (Schur product)
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs * rhs
   */
  friend SpectrumValue operator* (const SpectrumValue& lhs, const SpectrumValue& rhs);

  /**
   *  multiplication by a scalar
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs * rhs
   */
  friend SpectrumValue operator* (const SpectrumValue& lhs, double rhs);

  /**
   *  multiplication of a scalar
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs * rhs
   */
  friend SpectrumValue operator* (double lhs, const SpectrumValue& rhs);

  /**
   *  division component-by-component
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of lhs / rhs
   */
  friend SpectrumValue operator/ (const SpectrumValue& lhs, const SpectrumValue& rhs);

  /**
   * division by a scalar
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of *this / rhs
   */
  friend SpectrumValue operator/ (const SpectrumValue& lhs, double rhs);

  /**
   * division of a scalar
   *
   * @param lhs Left Hand Side of the operator
   * @param rhs Right Hand Side of the operator
   *
   * @return the value of *this / rhs
   */
  friend SpectrumValue operator/ (double lhs, const SpectrumValue& rhs);

  /**
   * unary plus operator
   *
   * @param rhs Right Hand Side of the operator
   * @return the value of *this
   */
  friend SpectrumValue operator+ (const SpectrumValue& rhs);

  /**
   * unary minus operator
   *
   * @param rhs Right Hand Side of the operator
   * @return the value of - *this
   */
  friend SpectrumValue operator- (const SpectrumValue& rhs);


  /**
   * left shift operator
   *
   * @param n position to shift
   *
   * @return the value of *this left shifted by n positions. In other
   * words, the function of the set of frequencies represented by
   * *this is left-shifted in frequency by n positions.
   */
  SpectrumValue operator<< (int n) const;

  /**
    * right shift operator
    *
    * @param n position to shift
    *
    * @return the value of *this right shifted by n positions. In other
    * words, the function of the set of frequencies represented by
    * *this is left-shifted in frequency by n positions.
    */
  SpectrumValue operator>> (int n) const;



  /**
   * Add the Right Hand Side of the operator to *this, component by component
   *
   * @param rhs the Right Hand Side
   *
   * @return a reference to *this
   */
  SpectrumValue& operator+= (const SpectrumValue& rhs);

  /**
   * Subtract the Right Hand Side of the operator from *this, component by component
   *
   * @param rhs the Right Hand Side
   *
   * @return a reference to *this
   */
  SpectrumValue& operator-= (const SpectrumValue& rhs);

  /**
   * Multiply *this by the Right Hand Side of the operator, component by component
   *
   * @param rhs the Right Hand Side
   *
   * @return  a reference to *this
   */
  SpectrumValue& operator*= (const SpectrumValue& rhs);

  /**
   * Divide *this by the Right Hand Side of the operator, component by component
   *
   * @param rhs the Right Hand Side
   *
   * @return  a reference to *this
   */
  SpectrumValue& operator/= (const SpectrumValue& rhs);

  /**
   * Add the value of the Right Hand Side of the operator to all
   * components of *this
   *
   * @param rhs the Right Hand Side
   *
   * @return a reference to *this
   */
  SpectrumValue& operator+= (double rhs);

  /**
   * Subtract the value of the Right Hand Side of the operator from all
   * components of *this
   *
   * @param rhs the Right Hand Side
   *
   * @return a reference to *this
   */
  SpectrumValue& operator-= (double rhs);

  /**
   * Multiply every component of *this by the value of the Right Hand
   * Side of the operator
   *
   * @param rhs the Right Hand Side
   *
   * @return  a reference to *this
   */
  SpectrumValue& operator*= (double rhs);

  /**
   * Divide every component of *this by the value of the Right Hand
   * Side of the operator
   *
   * @param rhs the Right Hand Side
   *
   * @return  a reference to *this
   */
  SpectrumValue& operator/= (double rhs);


  /**
   * Assign each component of *this to the value of the Right Hand
   * Side of the operator
   *
   * @param rhs
   *
   * @return
   */
  SpectrumValue& operator= (double rhs);



  /**
   *
   * @param x the operand
   *
   * @return the euclidean norm, i.e., the sum of the squares of all
   * the values in x
   */
  friend double Norm (const SpectrumValue& x);


  /**
   *
   * @param x the operand
   *
   * @return the sum of all
   * the values in x
   */
  friend double Sum (const SpectrumValue& x);


  /**
   * @param x the operand
   *
   * @return the product of all
   * the values in x
   */
  friend double Prod (const SpectrumValue& x);


  /**
   *
   *
   * @param lhs the base
   * @param rhs the exponent
   *
   * @return each value in base raised to the exponent
   */
  friend SpectrumValue Pow (const SpectrumValue& lhs, double rhs);


  /**
   *
   * @param lhs the base
   * @param rhs the exponent
   *
   * @return the value in base raised to each value in the exponent
   */
  friend SpectrumValue Pow (double lhs, const SpectrumValue& rhs);

  /**
   *
   *
   * @param arg the argument
   *
   * @return the logarithm in base 10 of all values in the argument
   */
  friend SpectrumValue Log10 (const SpectrumValue&  arg);


  /**
   *
   *
   * @param arg the argument
   *
   * @return the logarithm in base 2 of all values in the argument
   */
  friend SpectrumValue Log2 (const SpectrumValue&  arg);

  /**
   *
   *
   * @param arg the argument
   *
   * @return the logarithm in base e of all values in the argument
   */
  friend SpectrumValue Log (const SpectrumValue&  arg);

  /**
   *
   *
   * @param arg the argument
   *
   * @return the value of the integral \f$\int_F g(f) df  \f$
   */
  friend double Integral (const SpectrumValue&  arg);

  /**
   *
   * @return a Ptr to a copy of this instance
   */
  Ptr<SpectrumValue> Copy () const;

  /**
   *  TracedCallback signature for SpectrumValue.
   *
   * \param [in] value Value of the traced variable.
   * \deprecated The non-const \c Ptr<SpectrumPhy> argument
   * is deprecated and will be changed to \c Ptr<const SpectrumPhy>
   * in a future release.
   */
  typedef void (* TracedCallback)(Ptr<SpectrumValue> value);


private:
  /**
   * Add a SpectrumValue (element to element addition)
   * \param x SpectrumValue
   */
  void Add (const SpectrumValue& x);
  /**
   * Add a flat value to all the current elements
   * \param s flat value
   */
  void Add (double s);
  /**
   * Subtracts a SpectrumValue (element by element subtraction)
   * \param x SpectrumValue
   */
  void Subtract (const SpectrumValue& x);
  /**
   * Subtracts a flat value to all the current elements
   * \param s flat value
   */
  void Subtract (double s);
  /**
   * Multiplies for a SpectrumValue (element to element multiplication)
   * \param x SpectrumValue
   */
  void Multiply (const SpectrumValue& x);
  /**
   * Multiplies for a flat value to all the current elements
   * \param s flat value
   */
  void Multiply (double s);
  /**
   * Divides by a SpectrumValue (element to element division)
   * \param x SpectrumValue
   */
  void Divide (const SpectrumValue& x);
  /**
   * Divides by a flat value to all the current elements
   * \param s flat value
   */
  void Divide (double s);
  /**
   * Change the values sign
   */
  void ChangeSign ();
  /**
   * Shift the values to the left
   * \param n number of positions to shift
   */
  void ShiftLeft (int n);
  /**
   * Shift the values to the right
   * \param n number of positions to shift
   */
  void ShiftRight (int n);
  /**
   * Modifies each element so that it each element is raised to the exponent
   *
   * \param exp the exponent
   */
  void Pow (double exp);
  /**
   * Modifies each element so that it is
   * the base raised to each element value
   *
   * \param base the base
   */
  void Exp (double base);
  /**
   * Applies a Log10 to each the elements
   */
  void Log10 ();
  /**
   * Applies a Log2 to each the elements
   */
  void Log2 ();
  /**
   * Applies a Log to each the elements
   */
  void Log ();

  Ptr<const SpectrumModel> m_spectrumModel; //!< The spectrum model


  /**
   * Set of values which implement the codomain of the functions in
   * the Function Space defined by SpectrumValue. There is no restriction
   * on what these values represent (a transmission power density, a
   * propagation loss, etc.).
   *
   */
  Values m_values;


};

std::ostream& operator << (std::ostream& os, const SpectrumValue& pvf);

double Norm (const SpectrumValue& x);
double Sum (const SpectrumValue& x);
double Prod (const SpectrumValue& x);
SpectrumValue Pow (const SpectrumValue& lhs, double rhs);
SpectrumValue Pow (double lhs, const SpectrumValue& rhs);
SpectrumValue Log10 (const SpectrumValue& arg);
SpectrumValue Log2 (const SpectrumValue& arg);
SpectrumValue Log (const SpectrumValue& arg);
double Integral (const SpectrumValue& arg);


} // namespace ns3

#endif /* SPECTRUM_VALUE_H */
