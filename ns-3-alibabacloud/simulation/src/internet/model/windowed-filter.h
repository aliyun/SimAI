/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 *  Copyright (c) 2016 The Chromium Authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *    * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 *  Note: This code is modified to work under ns-3's environment.
 *  Modified by: Vivek Jain <jain.vivek.anand@gmail.com>
 *               Viyom Mittal <viyommittal@gmail.com>
 *               Mohit P. Tahiliani <tahiliani@nitk.edu.in>
 */

// Implements Kathleen Nichols' algorithm for tracking the minimum (or maximum)
// estimate of a stream of samples over some fixed time interval. (E.g.,
// the minimum RTT over the past five minutes.) The algorithm keeps track of
// the best, second best, and third best min (or max) estimates, maintaining an
// invariant that the measurement time of the n'th best >= n-1'th best.
// The algorithm works as follows. On a reset, all three estimates are set to
// the same sample. The second best estimate is then recorded in the second
// quarter of the window, and a third best estimate is recorded in the second
// half of the window, bounding the worst case error when the true min is
// monotonically increasing (or true max is monotonically decreasing) over the
// window.
//
// A new best sample replaces all three estimates, since the new best is lower
// (or higher) than everything else in the window and it is the most recent.
// The window thus effectively gets reset on every new min. The same property
// holds true for second best and third best estimates. Specifically, when a
// sample arrives that is better than the second best but not better than the
// best, it replaces the second and third best estimates but not the best
// estimate. Similarly, a sample that is better than the third best estimate
// but not the other estimates replaces only the third best estimate.
//
// Finally, when the best expires, it is replaced by the second best, which in
// turn is replaced by the third best. The newest sample replaces the third
// best.

#ifndef WINDOWED_FILTER_H_
#define WINDOWED_FILTER_H_

namespace ns3 {
/**
 * \brief Compares two values
 * \param T  type of the measurement that is being filtered.
 */
template <class T>
struct MinFilter
{
public:
  /**
   * \brief Compares two values
   *
   * \param lhs left hand value
   * \param rhs right hand value
   * \return returns true if the first is less than or equal to the second.
   */
  bool operator() (const T& lhs, const T& rhs) const
  {
    if (rhs == 0 || lhs == 0)
      {
        return false;
      }
    return lhs <= rhs;
  }
};
/**
 * \brief Compares two values
 * \param T  type of the measurement that is being filtered.
 */
template <class T>
struct MaxFilter
{
public:
  /**
   * \brief Compares two values
   *
   * \param lhs left hand value
   * \param rhs right hand value
   * \return returns true if the first is greater than or equal to the second.
   */
  bool operator() (const T& lhs, const T& rhs) const
  {
    if (rhs == 0 || lhs == 0)
      {
        return false;
      }
    return lhs >= rhs;
  }
};
/**
 * \brief Construct a windowed filter
 *
 * Use the following to construct a windowed filter object of type T.
 * For example, a min filter using QuicTime as the time type:
 *      WindowedFilter<T, MinFilter<T>, QuicTime, QuicTime::Delta> ObjectName;
 * max filter using 64-bit integers as the time type:
 *      WindowedFilter<T, MaxFilter<T>, uint64_t, int64_t> ObjectName;
 *
 * \param T -- type of the measurement that is being filtered.
 * \param Compare -- MinFilter<T> or MaxFilter<T>, depending on the type of filter desired.
 * \param TimeT -- the type used to represent timestamps.
 * \param TimeDeltaT -- the type used to represent continuous time intervals between
 * two timestamps.  Has to be the type of (a - b) if both |a| and |b| are
 * of type TimeT.
 */
template <class T, class Compare, typename TimeT, typename TimeDeltaT>
class WindowedFilter
{
public:
  /**
   * \brief contructor
   */
  WindowedFilter ()
  {
  }

  /**
   * \brief contructor
   * \param windowLength is the period after which a best estimate expires.
   * \param zeroValue is used as the uninitialized value for objects of T. Importantly,
   * zeroValue should be an invalid value for a true sample.
   * \param zeroTime is the time of instance record time.
   */
  WindowedFilter (TimeDeltaT windowLength, T zeroValue, TimeT zeroTime)
    : m_windowLength (windowLength),
      m_zeroValue (zeroValue),
      m_samples
  {
    Sample (m_zeroValue, zeroTime),
    Sample (m_zeroValue, zeroTime),
    Sample (m_zeroValue, zeroTime)
  } {}
  /**
   * \brief Changes the window length.  Does not update any current samples.
   * \param windowLength is the period after which a best estimate expires.
   */
  void SetWindowLength (TimeDeltaT windowLength)
  {
    m_windowLength = windowLength;
  }
  /**
   * \brief Updates best estimates with |sample|, and expires and updates best
   * estimates as necessary.
   *
   * \param new_sample update new sample.
   * \param new_time record time of the new sample.
   */
  void Update (T new_sample, TimeT new_time)
  {
    // Reset all estimates if they have not yet been initialized, if new sample
    // is a new best, or if the newest recorded estimate is too old.
    if (m_samples[0].sample == m_zeroValue
        || Compare () (new_sample, m_samples[0].sample)
        || new_time - m_samples[2].time > m_windowLength)
      {
        Reset (new_sample, new_time);
        return;
      }
    if (Compare () (new_sample, m_samples[1].sample))
      {
        m_samples[1] = Sample (new_sample, new_time);
        m_samples[2] = m_samples[1];
      }
    else if (Compare () (new_sample, m_samples[2].sample))
      {
        m_samples[2] = Sample (new_sample, new_time);
      }
    // Expire and update estimates as necessary.
    if (new_time - m_samples[0].time > m_windowLength)
      {
        // The best estimate hasn't been updated for an entire window, so promote
        // second and third best estimates.
        m_samples[0] = m_samples[1];
        m_samples[1] = m_samples[2];
        m_samples[2] = Sample (new_sample, new_time);
        // Need to iterate one more time. Check if the new best estimate is
        // outside the window as well, since it may also have been recorded a
        // long time ago. Don't need to iterate once more since we cover that
        // case at the beginning of the method.
        if (new_time - m_samples[0].time > m_windowLength)
          {
            m_samples[0] = m_samples[1];
            m_samples[1] = m_samples[2];
          }
        return;
      }
    if (m_samples[1].sample == m_samples[0].sample
        && new_time - m_samples[1].time > m_windowLength >> 2)
      {
        // A quarter of the window has passed without a better sample, so the
        // second-best estimate is taken from the second quarter of the window.
        m_samples[2] = m_samples[1] = Sample (new_sample, new_time);
        return;
      }
    if (m_samples[2].sample == m_samples[1].sample
        && new_time - m_samples[2].time > m_windowLength >> 1)
      {
        // We've passed a half of the window without a better estimate, so take
        // a third-best estimate from the second half of the window.
        m_samples[2] = Sample (new_sample, new_time);
      }
  }

  /**
   * \brief Resets all estimates to new sample.
   * \param new_sample update new sample.
   * \param new_time record time of the new sample.
   */
  void Reset (T new_sample, TimeT new_time)
  {
    m_samples[0] = m_samples[1] = m_samples[2] = Sample (new_sample, new_time);
  }

  /**
   * \brief Returns Max/Min value so far among the windowed samples.
   * \return returns Best (max/min) value so far among the windowed samples.
   */
  T GetBest () const
  {
    return m_samples[0].sample;
  }

  /**
   * \brief Returns second Max/Min value so far among the windowed samples.
   * \return returns second Best (max/min) value so far among the windowed samples.
   */
  T GetSecondBest () const
  {
    return m_samples[1].sample;
  }

  /**
   * \brief Returns third Max/Min value so far among the windowed samples.
   * \return returns third Best (max/min) value so far among the windowed samples.
   */
  T GetThirdBest () const
  {
    return m_samples[2].sample;
  }

  /**
   * \brief sample.
   */
  struct Sample
  {
    T sample; //!< recorded sample.
    TimeT time; //!< time when the sample was recorded.
    /**
     * \brief constructor
     */
    Sample ()
    {
    }

    /**
     * \brief constructor
     * \param init_sample value of sample.
     * \param init_time time when the sample was recorded..
     */
    Sample (T init_sample, TimeT init_time)
      : sample (init_sample),
        time (init_time)
    {
    }
  };

  TimeDeltaT m_windowLength;    //!< Time length of window.
  T          m_zeroValue;       //!< Uninitialized value of T.
  Sample     m_samples[3];      //!< Best estimate is element 0.
};

}  // namespace ns3
#endif  // WINDOWED_FILTER_H_
