/*

   timing_of.c
   Calculates timing given struct rusage.

   $Id: timing_of.c,v 1.3 2002/07/12 20:48:49 eschrich Exp $
   Steven Eschrich

   Copyright (C) 2002 University of South Florida

   This program is free software; you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published by the
   Free Software Foundation; either version 2 of the License, or (at
   your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/

#include <stdlib.h>
#include <time.h>
#include <sys/times.h>
#include <sys/resource.h>

/* Get time, in seconds */
double *timing_of(struct rusage start, struct rusage stop)
{
  long totaluMicroseconds, totalsMicroseconds;
  static double totalTime[2];

  totaluMicroseconds = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;
  totalsMicroseconds = stop.ru_stime.tv_usec - start.ru_stime.tv_usec;

  /* If we need to borrow, do so */
  if ( totaluMicroseconds < 0 ) {
    stop.ru_utime.tv_sec--;
    totaluMicroseconds=1000000L + stop.ru_utime.tv_usec - start.ru_utime.tv_usec;
  }

  if ( totalsMicroseconds < 0 ) {
    stop.ru_stime.tv_sec--;
    totalsMicroseconds=1000000L + stop.ru_stime.tv_usec - start.ru_stime.tv_usec;
  }

  totalTime[0] = (stop.ru_utime.tv_sec - start.ru_utime.tv_sec) + 0.000001 * totaluMicroseconds;
  totalTime[1] = (stop.ru_stime.tv_sec - start.ru_stime.tv_sec) + 0.000001 * totalsMicroseconds;

  return totalTime;
}
