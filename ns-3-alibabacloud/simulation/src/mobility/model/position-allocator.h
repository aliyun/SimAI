/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2007 INRIA
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
 * Author: Mathieu Lacage <mathieu.lacage@sophia.inria.fr>
 */
#ifndef POSITION_ALLOCATOR_H
#define POSITION_ALLOCATOR_H

#include "ns3/object.h"
#include "ns3/random-variable-stream.h"
#include "ns3/vector.h"

namespace ns3 {

/**
 * \ingroup mobility
 * \brief Allocate a set of positions. The allocation strategy is implemented in subclasses.
 *
 * This is a pure abstract base class.
 */
class PositionAllocator : public Object
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  PositionAllocator ();
  virtual ~PositionAllocator ();
  /**
   * \return the next chosen position.
   *
   * This method _must_ be implement in subclasses.
   */
  virtual Vector GetNext (void) const = 0;
  /**
   * Assign a fixed random variable stream number to the random variables
   * used by this model. Return the number of streams (possibly zero) that
   * have been assigned.
   *
   * This method _must_ be implement in subclasses.
   *
   * \param stream first stream index to use
   * \return the number of stream indices assigned by this model
   */
  virtual int64_t AssignStreams (int64_t stream) = 0;
};

/**
 * \ingroup mobility
 * \brief Allocate positions from a deterministic list specified by the user.
 *
 * The first call to ListPositionAllocator::GetNext  will return the
 * first element of the list, the second call, the second element, and so on.
 */
class ListPositionAllocator : public PositionAllocator
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  ListPositionAllocator ();

  /**
   * \brief Add a position to the list of positions
   * \param v the position to append at the end of the list of positions to return from GetNext.
   */
  void Add (Vector v);

  /**
   * \brief Add the positions listed in a file.
   * The file should be a simple text file, with one position per line,
   * either X and Y, or X, Y and Z, in meters.  The delimiter can
   * be any character, such as ',' or '\\t'; the default is a comma ','.
   *
   * The file is read using CsvReader, which explains how comments
   * and whitespace are handled.
   *
   * \param [in] filePath The path to the input file.
   * \param [in] defaultZ The default Z value to use when reading files
   *             with only X and Y positions.
   * \param [in] delimiter The delimiter character; see CsvReader.
   */
  void Add (const std::string filePath,
            double defaultZ = 0,
            char delimiter = ',');

  /**
   * Return the number of positions stored.  Note that this will not change
   * based on calling GetNext(), as the number of positions is not altered
   * by calling GetNext ().
   *
   * \return the number of positions stored
   */
  uint32_t GetSize (void) const;
  virtual Vector GetNext (void) const;
  virtual int64_t AssignStreams (int64_t stream);

private:
  std::vector<Vector> m_positions;  //!< vector of positions
  mutable std::vector<Vector>::const_iterator m_current; //!< vector iterator
};

/**
 * \ingroup mobility
 * \brief Allocate positions on a rectangular 2d grid.
 */
class GridPositionAllocator : public PositionAllocator
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Determine whether positions are allocated row first or column first.
   */
  enum LayoutType
  {
    /**
     * In row-first mode, positions are allocated on the first row until
     * N positions have been allocated. Then, the second row located a yMin + yDelta
     * is used to allocate positions.
     */
    ROW_FIRST,
    /**
     * In column-first mode, positions are allocated on the first column until
     * N positions have been allocated. Then, the second column located a xMin + xDelta
     * is used to allocate positions.
     */
    COLUMN_FIRST
  };

  GridPositionAllocator ();

  /**
   * \param xMin the x coordinate where layout will start.
   */
  void SetMinX (double xMin);
  /**
   * \param yMin the y coordinate where layout will start
   */
  void SetMinY (double yMin);
  /**
   * \param z   the Z coordinate of all the positions allocated
   */
  void SetZ (double z);
  /**
   * \param deltaX the x interval between two x-consecutive positions.
   */
  void SetDeltaX (double deltaX);
  /**
   * \param deltaY the y interval between two y-consecutive positions.
   */
  void SetDeltaY (double deltaY);
  /**
   * \param n the number of positions allocated on each row (or each column)
   *        before switching to the next column (or row).
   */
  void SetN (uint32_t n);
  /**
   * \param layoutType the type of layout to use (row first or column first).
   */
  void SetLayoutType (enum LayoutType layoutType);

  /**
   * \return the x coordinate of the first allocated position.
   */
  double GetMinX (void) const;
  /**
   * \return the y coordinate of the first allocated position.
   */
  double GetMinY (void) const;
  /**
   * \return the x interval between two consecutive x-positions.
   */
  double GetDeltaX (void) const;
  /**
   * \return the y interval between two consecutive y-positions.
   */
  double GetDeltaY (void) const;
  /**
   * \return the number of positions to allocate on each row or each column.
   */
  uint32_t GetN (void) const;
  /**
   * \return the currently-selected layout type.
   */
  enum LayoutType GetLayoutType (void) const;


  virtual Vector GetNext (void) const;
  virtual int64_t AssignStreams (int64_t stream);

private:
  mutable uint32_t m_current; //!< currently position
  enum LayoutType m_layoutType;  //!< currently selected layout type
  double m_xMin; //!< minimum boundary on x positions
  double m_yMin; //!< minimum boundary on y positions
  double m_z; //!< z coordinate of all the positions generated
  uint32_t m_n;  //!< number of positions to allocate on each row or column
  double m_deltaX; //!< x interval between two consecutive x positions
  double m_deltaY; //!< y interval between two consecutive y positions
};

/**
 * \ingroup mobility
 * \brief Allocate random positions within a rectangle according to a pair of random variables.
 */
class RandomRectanglePositionAllocator : public PositionAllocator
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  RandomRectanglePositionAllocator ();
  virtual ~RandomRectanglePositionAllocator ();

  /**
   * \brief Set the random variable stream object that generates x-positions
   * \param x pointer to a RandomVariableStream object
   */
  void SetX (Ptr<RandomVariableStream> x);
  /**
   * \brief Set the random variable stream object that generates y-positions
   * \param y pointer to a RandomVariableStream object
   */
  void SetY (Ptr<RandomVariableStream> y);
  /**
   * \param z   the Z coordinate of all the positions allocated
   */
  void SetZ (double z);

  virtual Vector GetNext (void) const;
  virtual int64_t AssignStreams (int64_t stream);

private:
  Ptr<RandomVariableStream> m_x; //!< pointer to x's random variable stream
  Ptr<RandomVariableStream> m_y; //!< pointer to y's random variable stream
  double m_z; //!< z coordinate of all the positions generated
};

/**
 * \ingroup mobility
 * \brief Allocate random positions within a 3D box according to a set of three random variables.
 */
class RandomBoxPositionAllocator : public PositionAllocator
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  RandomBoxPositionAllocator ();
  virtual ~RandomBoxPositionAllocator ();

  /**
   * \brief Set the random variable stream object that generates x-positions
   * \param x pointer to a RandomVariableStream object
   */
  void SetX (Ptr<RandomVariableStream> x);
  /**
   * \brief Set the random variable stream object that generates y-positions
   * \param y pointer to a RandomVariableStream object
   */
  void SetY (Ptr<RandomVariableStream> y);
  /**
   * \brief Set the random variable stream object that generates z-positions
   * \param z pointer to a RandomVariableStream object
   */
  void SetZ (Ptr<RandomVariableStream> z);

  virtual Vector GetNext (void) const;
  virtual int64_t AssignStreams (int64_t stream);

private:
  Ptr<RandomVariableStream> m_x; //!< pointer to x's random variable stream
  Ptr<RandomVariableStream> m_y; //!< pointer to y's random variable stream
  Ptr<RandomVariableStream> m_z; //!< pointer to z's random variable stream
};

/**
 * \ingroup mobility
 * \brief Allocate random positions within a disc according to
 * a given distribution for the polar coordinates of each node
 * with respect to the provided center of the disc.
 *
 * \note With the default uniform distribution over \f$2 \pi\f$ in \c theta and a
 * uniform distribution for \c rho this position allocator will *not*
 * uniformly populate the disc.  The radial distribution will be proportional
 * to \f$\frac{1}{r^2}\f$.
 *
 * To get a uniform distribution over a circle use the UniformDiscPositionAllocator.
 */
class RandomDiscPositionAllocator : public PositionAllocator
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  RandomDiscPositionAllocator ();
  virtual ~RandomDiscPositionAllocator ();

  /**
   * \brief Set the random variable that generates position angle, in radians.
   * \param theta Random variable that represents the angle in radians of a position in a random disc.
   */
  void SetTheta (Ptr<RandomVariableStream> theta);
  /**
   * \brief Set the random variable that generates position radius, in meters
   * \param rho Random variable that represents the radius of a position, in meters, in a random disc.
   */
  void SetRho (Ptr<RandomVariableStream> rho);
  /**
   * \param x  the X coordinate of the center of the disc
   */
  void SetX (double x);
  /**
   * \param y   the Y coordinate of the center of the disc
   */
  void SetY (double y);
  /**
   * \param z   the Z coordinate of all the positions allocated
   */
  void SetZ (double z);

  virtual Vector GetNext (void) const;
  virtual int64_t AssignStreams (int64_t stream);

private:
  Ptr<RandomVariableStream> m_theta; //!< pointer to theta's random variable stream
  Ptr<RandomVariableStream> m_rho; //!< pointer to rho's random variable stream
  double m_x; //!< x coordinate of center of disc
  double m_y; //!< y coordinate of center of disc
  double m_z;  //!< z coordinate of the disc
};


/**
 * \ingroup mobility
 * \brief Allocate the positions uniformly (with constant density) randomly within a disc.
 *
 * UniformDiscPositionAllocator allocates the positions randomly within a disc \f$ D \f$ lying on the
 * plane \f$ z\f$ and having center at coordinates \f$ (x,y,z) \f$
 * and radius \f$ \rho \f$. The random positions are chosen such that,
 * for any subset \f$ S \subset D \f$, the expected value of the
 * fraction of points which fall into \f$ S \subset D \f$ corresponds
 * to \f$ \frac{|S|}{|D|} \f$, i.e., to the ratio of the area of the
 * subset to the area of the whole disc.
 *
 * \note using UniformDiscPositionAllocator is not equivalent to using
 * a RandomDiscPositionAllocator with a uniformly-distributed radius,
 * since doing that would results in a point distribution which is
 * more dense towards the center of the disc.
 */
class UniformDiscPositionAllocator : public PositionAllocator
{
public:
  /**
   * Register this type with the TypeId system.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);
  UniformDiscPositionAllocator ();
  virtual ~UniformDiscPositionAllocator ();

  /**
   * \param rho the value of the radius of the disc
   */
  void SetRho (double rho);

  /**
   * \param x  the X coordinate of the center of the disc
   */
  void SetX (double x);

  /**
   * \param y   the Y coordinate of the center of the disc
   */
  void SetY (double y);

  /**
   * \param z   the Z coordinate of all the positions allocated
   */
  void SetZ (double z);

  virtual Vector GetNext (void) const;
  virtual int64_t AssignStreams (int64_t stream);

private:
  Ptr<UniformRandomVariable> m_rv;  //!< pointer to uniform random variable
  double m_rho; //!< value of the radius of the disc
  double m_x;  //!< x coordinate of center of disc
  double m_y;  //!< y coordinate of center of disc
  double m_z;  //!< z coordinate of the disc
};

} // namespace ns3

#endif /* RANDOM_POSITION_H */
