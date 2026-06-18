/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2012 Lawrence Livermore National Laboratory
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
 * Author: Peter D. Barnes, Jr. <pdbarnes@llnl.gov>
 *
 * This copyright notice applies strictly to the wrapper material.
 *
 * The FNV source code itself is in the public domain.  The FNV source
 * code sections are marked by
 *   // Begin <fnv-file> ---->
 * and
 *   // End <fnv-file>   ---->
 * comments.
 *
 * Code changes from the FNV distribution are marked with `//PDB'
 * In addition comment blocks have been converted to Doxygen format.
 */

#include <sys/types.h>
#include <stdlib.h>

#include "log.h"
#include "hash-fnv.h"

/**
 * \file
 * \ingroup hash
 * \brief ns3::Hash::Function::Fnv1a implementation.
 */


namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("Hash-Fnv");

namespace Hash {

namespace Function {

/** FNV hash implementation details. */
namespace Fnv1aImplementation {

/*************************************************
 **  class FnvHashImplementation
 ************************************************/

/**
 * \ingroup hash
 * \defgroup hash_fnv FNV Hash Implementation
 */
/**@{*/

extern "C" {

// Changes from FNV distribution are marked with `//PDB'
//

/* Begin fnv.h -------------- *NS_CHECK_STYLE_OFF* ----> */

/*
 * fnv - Fowler/Noll/Vo- hash code
 *
 * @(#) $Revision: 5.4 $
 * @(#) $Id: fnv.h,v 5.4 2009/07/30 22:49:13 chongo Exp $
 * @(#) $Source: /usr/local/src/cmd/fnv/RCS/fnv.h,v $
 *
 ***
 *
 * Fowler/Noll/Vo- hash
 *
 * The basis of this hash algorithm was taken from an idea sent
 * as reviewer comments to the IEEE POSIX P1003.2 committee by:
 *
 *      Phong Vo (http://www.research.att.com/info/kpv/)
 *      Glenn Fowler (http://www.research.att.com/~gsf/)
 *
 * In a subsequent ballot round:
 *
 *      Landon Curt Noll (http://www.isthe.com/chongo/)
 *
 * improved on their algorithm.  Some people tried this hash
 * and found that it worked rather well.  In an EMail message
 * to Landon, they named it the ``Fowler/Noll/Vo'' or FNV hash.
 *
 * FNV hashes are designed to be fast while maintaining a low
 * collision rate. The FNV speed allows one to quickly hash lots
 * of data while maintaining a reasonable collision rate.  See:
 *
 *      http://www.isthe.com/chongo/tech/comp/fnv/index.html
 *
 * for more details as well as other forms of the FNV hash.
 *
 ***
 *
 * NOTE: The FNV-0 historic hash is not recommended.  One should use
 *	 the FNV-1 hash instead.
 *
 * To use the 32 bit FNV-0 historic hash, pass FNV0_32_INIT as the
 * Fnv32_t hashval argument to fnv_32_buf() or fnv_32_str().
 *
 * To use the 64 bit FNV-0 historic hash, pass FNV0_64_INIT as the
 * Fnv64_t hashval argument to fnv_64_buf() or fnv_64_str().
 *
 * To use the recommended 32 bit FNV-1 hash, pass FNV1_32_INIT as the
 * Fnv32_t hashval argument to fnv_32_buf() or fnv_32_str().
 *
 * To use the recommended 64 bit FNV-1 hash, pass FNV1_64_INIT as the
 * Fnv64_t hashval argument to fnv_64_buf() or fnv_64_str().
 *
 * To use the recommended 32 bit FNV-1a hash, pass FNV1_32A_INIT as the
 * Fnv32_t hashval argument to fnv_32a_buf() or fnv_32a_str().
 *
 * To use the recommended 64 bit FNV-1a hash, pass FNV1A_64_INIT as the
 * Fnv64_t hashval argument to fnv_64a_buf() or fnv_64a_str().
 *
 ***
 *
 * Please do not copyright this code.  This code is in the public domain.
 *
 * LANDON CURT NOLL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
 * EVENT SHALL LANDON CURT NOLL BE LIABLE FOR ANY SPECIAL, INDIRECT OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
 * USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 *
 * By:
 *	chongo <Landon Curt Noll> /\oo/\
 *      http://www.isthe.com/chongo/
 *
 * Share and Enjoy!	:-)
 */

#if !defined(__FNV_H__)
/** Include guard from the original fnv.h. */
#define __FNV_H__


//#include <sys/types.h>  //PDB

#define FNV_VERSION "5.0.2"	/**< @(#) FNV Version */


/**
 * 32 bit FNV-0 hash type
 */
typedef uint32_t Fnv32_t;  //PDB


/**
 * 32 bit FNV-0 zero initial basis
 *
 * This historic hash is not recommended.  One should use
 * the FNV-1 hash and initial basis instead.
 */
// Use fully qualified type so this define works outside this scope //PDB
#define FNV0_32_INIT ((Fnv1aImplementation::Fnv32_t)0)


/**
 * 32 bit FNV-1 and FNV-1a non-zero initial basis
 *
 * The FNV-1 initial basis is the FNV-0 hash of the following 32 octets:
 *
 *              chongo <Landon Curt Noll> /\../\
 *
 * \note The \'s above are not back-slashing escape characters.
 * They are literal ASCII  backslash 0x5c characters.
 *
 * \note The FNV-1a initial basis is the same value as FNV-1 by definition.
 */
// Use fully qualified type so this define works outside this scope //PDB
#define FNV1_32_INIT ((Fnv1aImplementation::Fnv32_t)0x811c9dc5)
/** \copydoc FNV1_32_INIT */
#define FNV1_32A_INIT FNV1_32_INIT


/**
 * Determine how 64 bit unsigned values are represented
 */
//#include "longlong.h"  //PDB - assume `unsigned long long' is 64 bit
#define HAVE_64BIT_LONG_LONG  



/** 
 * 64 bit FNV-0 hash
 */
#if defined(HAVE_64BIT_LONG_LONG)
typedef uint64_t Fnv64_t;  //PDB
#else /* HAVE_64BIT_LONG_LONG */
typedef struct {
    uint32_t w32[2]; /* w32[0] is low order, w32[1] is high order word */  //PDB
} Fnv64_t;
#endif /* HAVE_64BIT_LONG_LONG */


/**
 * 64 bit FNV-0 zero initial basis
 *
 * This historic hash is not recommended.  One should use
 * the FNV-1 hash and initial basis instead.
 */
// Use fully qualified type so this define works outside this scope //PDB
#if defined(HAVE_64BIT_LONG_LONG)
#define FNV0_64_INIT ((Fnv1aImplementation::Fnv64_t)0)
#else /* HAVE_64BIT_LONG_LONG */
extern const Fnv64_t fnv0_64_init;
#define FNV0_64_INIT (Fnv1aImplementation::fnv0_64_init)
#endif /* HAVE_64BIT_LONG_LONG */


/**
 * 64 bit FNV-1 non-zero initial basis
 *
 * The FNV-1 initial basis is the FNV-0 hash of the following 32 octets:
 * 
 *              chongo <Landon Curt Noll> /\../\
 *
 * \note The \'s above are not back-slashing escape characters.
 * They are literal ASCII  backslash 0x5c characters.
 *
 * \note The FNV-1a initial basis is the same value as FNV-1 by definition.
 */
#if defined(HAVE_64BIT_LONG_LONG)
#define FNV1_64_INIT ((Fnv1aImplementation::Fnv64_t)0xcbf29ce484222325ULL)
/** \copydoc FNV1_64_INIT */
#define FNV1A_64_INIT FNV1_64_INIT
#else /* HAVE_64BIT_LONG_LONG */
extern const fnv1_64_init;
extern const Fnv64_t fnv1a_64_init;
#define FNV1_64_INIT (fnv1_64_init)
/** \copydoc FNV1_64_INIT */
#define FNV1A_64_INIT (fnv1a_64_init)
#endif /* HAVE_64BIT_LONG_LONG */


/**
 * FNV hash types
 */
enum fnv_type {
    FNV_NONE = 0,	/**< invalid FNV hash type */
    FNV0_32 = 1,	/**< FNV-0 32 bit hash */
    FNV1_32 = 2,	/**< FNV-1 32 bit hash */
    FNV1a_32 = 3,	/**< FNV-1a 32 bit hash */
    FNV0_64 = 4,	/**< FNV-0 64 bit hash */
    FNV1_64 = 5,	/**< FNV-1 64 bit hash */
    FNV1a_64 = 6,	/**< FNV-1a 64 bit hash */
};

//PDB  test vector declarations deleted

/*
 * external functions  //PDB converted to forward declarations
 */
/** \copydoc fnv_32a_buf() */
/* extern */ Fnv32_t fnv_32_buf(void *buf, size_t len, Fnv32_t hval);
/** \copydoc fnv_32a_str() */
/* extern */ Fnv32_t fnv_32_str(char *str, Fnv32_t hval);

/* hash_32a.c */
/* extern */ Fnv32_t fnv_32a_buf(void *buf, size_t len, Fnv32_t hashval);
/* extern */ Fnv32_t fnv_32a_str(char *buf, Fnv32_t hashval);

/* hash_64.c */
/** \copydoc fnv_64a_buf() */
/* extern */ Fnv64_t fnv_64_buf(void *buf, size_t len, Fnv64_t hval);
/** \copydoc fnv_64a_str() */
/* extern */ Fnv64_t fnv_64_str(char *str, Fnv64_t hval);

/* hash_64a.c */
/* extern */ Fnv64_t fnv_64a_buf(void *buf, size_t len, Fnv64_t hashval);
/* extern */ Fnv64_t fnv_64a_str(char *buf, Fnv64_t hashval);

//PDB  test vector declarations deleted


#endif /* __FNV_H__ */

/* End fnv.h ---------------- *NS_CHECK_STYLE_ON* -----> */

/* Begin hash_32a.c --------- *NS_CHECK_STYLE_OFF* ----> */

/*
 * hash_32 - 32 bit Fowler/Noll/Vo FNV-1a hash code
 *
 * @(#) $Revision: 5.1 $
 * @(#) $Id: hash_32a.c,v 5.1 2009/06/30 09:13:32 chongo Exp $
 * @(#) $Source: /usr/local/src/cmd/fnv/RCS/hash_32a.c,v $
 *
 ***
 *
 * Fowler/Noll/Vo hash
 *
 * The basis of this hash algorithm was taken from an idea sent
 * as reviewer comments to the IEEE POSIX P1003.2 committee by:
 *
 *      Phong Vo (http://www.research.att.com/info/kpv/)
 *      Glenn Fowler (http://www.research.att.com/~gsf/)
 *
 * In a subsequent ballot round:
 *
 *      Landon Curt Noll (http://www.isthe.com/chongo/)
 *
 * improved on their algorithm.  Some people tried this hash
 * and found that it worked rather well.  In an EMail message
 * to Landon, they named it the ``Fowler/Noll/Vo'' or FNV hash.
 *
 * FNV hashes are designed to be fast while maintaining a low
 * collision rate. The FNV speed allows one to quickly hash lots
 * of data while maintaining a reasonable collision rate.  See:
 *
 *      http://www.isthe.com/chongo/tech/comp/fnv/index.html
 *
 * for more details as well as other forms of the FNV hash.
 ***
 *
 * To use the recommended 32 bit FNV-1a hash, pass FNV1_32A_INIT as the
 * Fnv32_t hashval argument to fnv_32a_buf() or fnv_32a_str().
 *
 ***
 *
 * Please do not copyright this code.  This code is in the public domain.
 *
 * LANDON CURT NOLL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
 * EVENT SHALL LANDON CURT NOLL BE LIABLE FOR ANY SPECIAL, INDIRECT OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
 * USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 *
 * By:
 *	chongo <Landon Curt Noll> /\oo/\
 *      http://www.isthe.com/chongo/
 *
 * Share and Enjoy!	:-)
 */

//#include <stdlib.h>    //PDB
//#include "fnv.h"       //PDB


/**
 * 32 bit magic FNV-1a prime
 */
#define FNV_32_PRIME ((Fnv1aImplementation::Fnv32_t)0x01000193)


/**
 * fnv_32a_buf - perform a 32 bit Fowler/Noll/Vo FNV-1a hash on a buffer
 *
 * input:
 * \param  [in] buf	start of buffer to hash
 * \param  [in] len	length of buffer in octets
 * \param  [in] hval	previous hash value or 0 if first call
 *
 * \returns		32 bit hash as a static hash type.
 *
 * \note To use the recommended 32 bit FNV-1a hash, use FNV1_32A_INIT as the
 * 	 hval arg on the first call to either fnv_32a_buf() or fnv_32a_str().
 */
Fnv32_t
fnv_32a_buf(void *buf, size_t len, Fnv32_t hval)
{
    unsigned char *bp = (unsigned char *)buf;	/* start of buffer */
    unsigned char *be = bp + len;		/* beyond end of buffer */

    /*
     * FNV-1a hash each octet in the buffer
     */
    while (bp < be) {

	/* xor the bottom with the current octet */
	hval ^= (Fnv32_t)*bp++;

	/* multiply by the 32 bit FNV magic prime mod 2^32 */
#if defined(NO_FNV_GCC_OPTIMIZATION)
	hval *= FNV_32_PRIME;
#else
	hval += (hval<<1) + (hval<<4) + (hval<<7) + (hval<<8) + (hval<<24);
#endif
    }

    /* return our new hash value */
    return hval;
}


/**
 * fnv_32a_str - perform a 32 bit Fowler/Noll/Vo FNV-1a hash on a string
 *
 * input:
 * \param  [in] str	string to hash
 * \param  [in] hval	previous hash value or 0 if first call
 *
 * \returns		32 bit hash as a static hash type
 *
 * \note To use the recommended 32 bit FNV-1a hash, use FNV1_32A_INIT as the
 *  	 hval arg on the first call to either fnv_32a_buf() or fnv_32a_str().
 */
Fnv32_t
fnv_32a_str(char *str, Fnv32_t hval)
{
    unsigned char *s = (unsigned char *)str;	/* unsigned string */

    /*
     * FNV-1a hash each octet in the buffer
     */
    while (*s) {

	/* xor the bottom with the current octet */
	hval ^= (Fnv32_t)*s++;

	/* multiply by the 32 bit FNV magic prime mod 2^32 */
#if defined(NO_FNV_GCC_OPTIMIZATION)
	hval *= FNV_32_PRIME;
#else
	hval += (hval<<1) + (hval<<4) + (hval<<7) + (hval<<8) + (hval<<24);
#endif
    }

    /* return our new hash value */
    return hval;
}

/* End hash_32a.c ----------- *NS_CHECK_STYLE_ON* -----> */

/* Begin hash_64a.c --------- *NS_CHECK_STYLE_OFF* ----> */

/*
 * hash_64 - 64 bit Fowler/Noll/Vo-0 FNV-1a hash code
 *
 * @(#) $Revision: 5.1 $
 * @(#) $Id: hash_64a.c,v 5.1 2009/06/30 09:01:38 chongo Exp $
 * @(#) $Source: /usr/local/src/cmd/fnv/RCS/hash_64a.c,v $
 *
 ***
 *
 * Fowler/Noll/Vo hash
 *
 * The basis of this hash algorithm was taken from an idea sent
 * as reviewer comments to the IEEE POSIX P1003.2 committee by:
 *
 *      Phong Vo (http://www.research.att.com/info/kpv/)
 *      Glenn Fowler (http://www.research.att.com/~gsf/)
 *
 * In a subsequent ballot round:
 *
 *      Landon Curt Noll (http://www.isthe.com/chongo/)
 *
 * improved on their algorithm.  Some people tried this hash
 * and found that it worked rather well.  In an EMail message
 * to Landon, they named it the ``Fowler/Noll/Vo'' or FNV hash.
 *
 * FNV hashes are designed to be fast while maintaining a low
 * collision rate. The FNV speed allows one to quickly hash lots
 * of data while maintaining a reasonable collision rate.  See:
 *
 *      http://www.isthe.com/chongo/tech/comp/fnv/index.html
 *
 * for more details as well as other forms of the FNV hash.
 *
 ***
 *
 * To use the recommended 64 bit FNV-1a hash, pass FNV1A_64_INIT as the
 * Fnv64_t hashval argument to fnv_64a_buf() or fnv_64a_str().
 *
 ***
 *
 * Please do not copyright this code.  This code is in the public domain.
 *
 * LANDON CURT NOLL DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 * INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
 * EVENT SHALL LANDON CURT NOLL BE LIABLE FOR ANY SPECIAL, INDIRECT OR
 * CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
 * USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE.
 *
 * By:
 *	chongo <Landon Curt Noll> /\oo/\
 *      http://www.isthe.com/chongo/
 *
 * Share and Enjoy!	:-)
 */

//#include <stdlib.h>    //PDB
//#include "fnv.h"       //PDB


/**
 * FNV-1a defines the initial basis to be non-zero
 */
#if !defined(HAVE_64BIT_LONG_LONG)
const Fnv64_t fnv1a_64_init = { 0x84222325, 0xcbf29ce4 };
#endif /* ! HAVE_64BIT_LONG_LONG */


/**
 * 64 bit magic FNV-1a prime
 */
/**@{*/
#if defined(HAVE_64BIT_LONG_LONG)
#define FNV_64_PRIME ((Fnv1aImplementation::Fnv64_t)0x100000001b3ULL)
#else /* HAVE_64BIT_LONG_LONG */
#define FNV_64_PRIME_LOW ((unsigned long)0x1b3)	/* lower bits of FNV prime */
#define FNV_64_PRIME_SHIFT (8)		/* top FNV prime shift above 2^32 */
#endif /* HAVE_64BIT_LONG_LONG */
/**@}*/


/**
 * fnv_64a_buf - perform a 64 bit Fowler/Noll/Vo FNV-1a hash on a buffer
 *
 * input:
 * \param  [in] buf	start of buffer to hash
 * \param  [in] len	length of buffer in octets
 * \param  [in] hval	previous hash value or 0 if first call
 *
 * \returns		64 bit hash as a static hash type
 *
 * \note To use the recommended 64 bit FNV-1a hash, use FNV1A_64_INIT as the
 * 	 hval arg on the first call to either fnv_64a_buf() or fnv_64a_str().
 */
Fnv64_t
fnv_64a_buf(void *buf, size_t len, Fnv64_t hval)
{
    unsigned char *bp = (unsigned char *)buf;	/* start of buffer */
    unsigned char *be = bp + len;		/* beyond end of buffer */

#if defined(HAVE_64BIT_LONG_LONG)
    /*
     * FNV-1a hash each octet of the buffer
     */
    while (bp < be) {

	/* xor the bottom with the current octet */
	hval ^= (Fnv64_t)*bp++;

	/* multiply by the 64 bit FNV magic prime mod 2^64 */
#if defined(NO_FNV_GCC_OPTIMIZATION)
	hval *= FNV_64_PRIME;
#else /* NO_FNV_GCC_OPTIMIZATION */
	hval += (hval << 1) + (hval << 4) + (hval << 5) +
		(hval << 7) + (hval << 8) + (hval << 40);
#endif /* NO_FNV_GCC_OPTIMIZATION */
    }

#else /* HAVE_64BIT_LONG_LONG */

    unsigned long val[4];			/* hash value in base 2^16 */
    unsigned long tmp[4];			/* tmp 64 bit value */

    /*
     * Convert Fnv64_t hval into a base 2^16 array
     */
    val[0] = hval.w32[0];
    val[1] = (val[0] >> 16);
    val[0] &= 0xffff;
    val[2] = hval.w32[1];
    val[3] = (val[2] >> 16);
    val[2] &= 0xffff;

    /*
     * FNV-1a hash each octet of the buffer
     */
    while (bp < be) {

	/* xor the bottom with the current octet */
	val[0] ^= (unsigned long)*bp++;

	/*
	 * multiply by the 64 bit FNV magic prime mod 2^64
	 *
	 * Using 0x100000001b3 we have the following digits base 2^16:
	 *
	 *	0x0	0x100	0x0	0x1b3
	 *
	 * which is the same as:
	 *
	 *	0x0	1<<FNV_64_PRIME_SHIFT	0x0	FNV_64_PRIME_LOW
	 */
	/* multiply by the lowest order digit base 2^16 */
	tmp[0] = val[0] * FNV_64_PRIME_LOW;
	tmp[1] = val[1] * FNV_64_PRIME_LOW;
	tmp[2] = val[2] * FNV_64_PRIME_LOW;
	tmp[3] = val[3] * FNV_64_PRIME_LOW;
	/* multiply by the other non-zero digit */
	tmp[2] += val[0] << FNV_64_PRIME_SHIFT;	/* tmp[2] += val[0] * 0x100 */
	tmp[3] += val[1] << FNV_64_PRIME_SHIFT;	/* tmp[3] += val[1] * 0x100 */
	/* propagate carries */
	tmp[1] += (tmp[0] >> 16);
	val[0] = tmp[0] & 0xffff;
	tmp[2] += (tmp[1] >> 16);
	val[1] = tmp[1] & 0xffff;
	val[3] = tmp[3] + (tmp[2] >> 16);
	val[2] = tmp[2] & 0xffff;
	/*
	 * Doing a val[3] &= 0xffff; is not really needed since it simply
	 * removes multiples of 2^64.  We can discard these excess bits
	 * outside of the loop when we convert to Fnv64_t.
	 */
    }

    /*
     * Convert base 2^16 array back into an Fnv64_t
     */
    hval.w32[1] = ((val[3]<<16) | val[2]);
    hval.w32[0] = ((val[1]<<16) | val[0]);

#endif /* HAVE_64BIT_LONG_LONG */

    /* return our new hash value */
    return hval;
}


/**
 * fnv_64a_str - perform a 64 bit Fowler/Noll/Vo FNV-1a hash on a buffer
 *
 * input:
 * \param  [in] str	string to hash
 * \param  [in] hval	previous hash value or 0 if first call
 *
 * \returns		64 bit hash as a static hash type
 *
 * \note To use the recommended 64 bit FNV-1a hash, use FNV1A_64_INIT as the
 * 	 hval arg on the first call to either fnv_64a_buf() or fnv_64a_str().
 */
Fnv64_t
fnv_64a_str(char *str, Fnv64_t hval)
{
    unsigned char *s = (unsigned char *)str;	/* unsigned string */

#if defined(HAVE_64BIT_LONG_LONG)

    /*
     * FNV-1a hash each octet of the string
     */
    while (*s) {

	/* xor the bottom with the current octet */
	hval ^= (Fnv64_t)*s++;

	/* multiply by the 64 bit FNV magic prime mod 2^64 */
#if defined(NO_FNV_GCC_OPTIMIZATION)
	hval *= FNV_64_PRIME;
#else /* NO_FNV_GCC_OPTIMIZATION */
	hval += (hval << 1) + (hval << 4) + (hval << 5) +
		(hval << 7) + (hval << 8) + (hval << 40);
#endif /* NO_FNV_GCC_OPTIMIZATION */
    }

#else /* !HAVE_64BIT_LONG_LONG */

    unsigned long val[4];	/* hash value in base 2^16 */
    unsigned long tmp[4];	/* tmp 64 bit value */

    /*
     * Convert Fnv64_t hval into a base 2^16 array
     */
    val[0] = hval.w32[0];
    val[1] = (val[0] >> 16);
    val[0] &= 0xffff;
    val[2] = hval.w32[1];
    val[3] = (val[2] >> 16);
    val[2] &= 0xffff;

    /*
     * FNV-1a hash each octet of the string
     */
    while (*s) {

	/* xor the bottom with the current octet */

	/*
	 * multiply by the 64 bit FNV magic prime mod 2^64
	 *
	 * Using 1099511628211, we have the following digits base 2^16:
	 *
	 *	0x0	0x100	0x0	0x1b3
	 *
	 * which is the same as:
	 *
	 *	0x0	1<<FNV_64_PRIME_SHIFT	0x0	FNV_64_PRIME_LOW
	 */
	/* multiply by the lowest order digit base 2^16 */
	tmp[0] = val[0] * FNV_64_PRIME_LOW;
	tmp[1] = val[1] * FNV_64_PRIME_LOW;
	tmp[2] = val[2] * FNV_64_PRIME_LOW;
	tmp[3] = val[3] * FNV_64_PRIME_LOW;
	/* multiply by the other non-zero digit */
	tmp[2] += val[0] << FNV_64_PRIME_SHIFT;	/* tmp[2] += val[0] * 0x100 */
	tmp[3] += val[1] << FNV_64_PRIME_SHIFT;	/* tmp[3] += val[1] * 0x100 */
	/* propagate carries */
	tmp[1] += (tmp[0] >> 16);
	val[0] = tmp[0] & 0xffff;
	tmp[2] += (tmp[1] >> 16);
	val[1] = tmp[1] & 0xffff;
	val[3] = tmp[3] + (tmp[2] >> 16);
	val[2] = tmp[2] & 0xffff;
	/*
	 * Doing a val[3] &= 0xffff; is not really needed since it simply
	 * removes multiples of 2^64.  We can discard these excess bits
	 * outside of the loop when we convert to Fnv64_t.
	 */
	val[0] ^= (unsigned long)(*s++);
    }

    /*
     * Convert base 2^16 array back into an Fnv64_t
     */
    hval.w32[1] = ((val[3]<<16) | val[2]);
    hval.w32[0] = ((val[1]<<16) | val[0]);

#endif /* !HAVE_64BIT_LONG_LONG */

    /* return our new hash value */
    return hval;
}

/* End hash_64a.c ----------- *NS_CHECK_STYLE_ON* -----> */

}  /* extern "C" */

//-----------------------------------------------------------------------------

/**@}*/  // \defgroup hash_fnv

}  // namespace Fnv1aImplementation


Fnv1a::Fnv1a ()
{
  clear ();
}

uint32_t
Fnv1a::GetHash32  (const char * buffer, const std::size_t size)
{
  m_hash32 =
    Fnv1aImplementation::fnv_32a_buf ((void *)buffer, size, m_hash32);
  return m_hash32;
}

uint64_t
Fnv1a::GetHash64  (const char * buffer, const std::size_t size)
{
  m_hash64 =
    Fnv1aImplementation::fnv_64a_buf ((void *)buffer, size, m_hash64);
  return m_hash64;
}

void
Fnv1a::clear (void)
{
  m_hash32 = FNV1_32A_INIT;
  m_hash64 = FNV1A_64_INIT;
}

}  // namespace Function

}  // namespace Hash

}  // namespace ns3
