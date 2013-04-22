#ifndef __UTILS_H_
#define __UTILS_H_
/*
   utils.h
   $Id: utils.h,v 1.3 2002/07/12 20:48:49 eschrich Exp $
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

/*
   General utilities implemented as macros. You can include this file
   either from it's original location or copy it to the local
   directory. It requires various other include files, which you
   should include prior to this one (such as stdio, stdlib, etc.) They
   are only needed if you use a particular macro.

   This file almost exclusively requires the use of GNU C extensions
   and GNU CPP extensions, although there is a warning if you do not
   use them. Don't expect any of these macros to work if you don't use
   gcc.


   Type Description
   Data type Boolean={True,False} values.


   Macro Descriptions

   General Functions
   -----------------
   STREQ(a,b)
      returns true if strings a and b are equal
   chomp(p)
      removes the last char in string p (by replacing it with a 0)
   min(x,y)
      safely returns min of x and y. Defined for types in which >,<
      are defined.
   max(x,y)
      safely returns max of x and y. Defined for types in which >,<
      are defined.
   push(array,item,array_count++)
      push item onto the end of array (you increment array_count). You
      must start using with array=NULL, array_count=0; then the storage
      is dynamically allocated in chunks when neccessary.
   pop(array,array_count--)
      pop item off of end of array (you decrement array_count). Does not
      free storage, only provide a convenient method of taking last
      element off.


   Output Functions
   ----------------
   info(format_string, arg, arg, ... )
      printf wrapper, prints to stderr
   warn(format_string, arg, arg, ...)
      Printf-like, prints to stderr;includes source file and line in message
   debug(level, format_string, arg, arg, ...)
      Given a global variable debug_level, prints out message if
         level < debug_level
   die(format_string, arg, arg, ...)
      Printf-like, prints error with source file/line and exits

   Function Wrappers
   -----------------
   CALLOC(n,size)
      calloc wrapper, exits on error
   MALLOC(size)
      malloc wrapper, exits on error
   REALLOC(ptr, size)
      realloc wrapper, exits on error
   FOPEN(file, mode)
      fopen wrapper, exits on error

*/

void exit();

#ifndef STREQ
 #define STREQ(a,b)   ( !strcmp(a,b) )
#endif
#ifndef MAXBUF
 #define MAXBUF       1024
#endif


/* Variable length argument macros are available only to GNU C */
#ifdef __GNUC__
 /* info() - */
 #ifndef info
    #define info(msg,args...) fprintf(stderr, msg, ##args)
 #else
    #error "info previously defined, not re-defined"
 #endif

 /* warn() - info with source file/line number */
 #ifndef warn
    #define warn(msg,args...)  \
                  fprintf(stderr, "%s:%d " msg, __FILE__, __LINE__ , ##args)
 #else
   #error "warn previously defined, not re-defined"
 #endif

 /* debug(level, printf). Output msg when debug_level >= level */
 #ifndef debug
    #define debug(level,msg,args...) \
       do { \
          if ( debug_level >= level ) \
           fprintf(stderr, "%s:%d " msg, __FILE__, __LINE__ , ##args); \
       } while (0)
 #else
   #error "debug previously defined, not re-defined"
 #endif

 /* die() - a perlism, printf macro that also dies */
 #ifndef die
    #define die(msg,args...) \
       do { \
	  fprintf(stderr, "%s:%d " msg "\n", __FILE__ , __LINE__ , ##args); \
          exit(1); \
       } while (0)
 #else
   #error "die previously defined, not re-defined"
 #endif

 #ifndef MALLOC
    #define MALLOC(size) \
         ({ \
            register void *__p; \
            if ( (__p=(void *)malloc(size)) == 0 ) { \
	      fprintf(stderr, "%s:%d error allocating memory\n", \
                                                       __FILE__, __LINE__); \
              exit(1); \
            } \
            __p; \
	 })
 #else
   #error "MALLOC previously defined, not re-defined"
 #endif

 #ifndef CALLOC
    #define CALLOC(n,size) \
         ({ \
            register void *__p; \
            if ( (__p=(void *)calloc(n,size)) == 0 ) { \
	      fprintf(stderr, "%s:%d error allocating memory\n", \
                                                       __FILE__, __LINE__); \
              exit(1); \
            } \
            __p; \
	 })
 #else
   #error "CALLOC previously defined, not re-defined"
 #endif

 #ifndef REALLOC
    #define REALLOC(q, n) \
         ({ \
            register void *__p; \
            if ( (__p=(void *)realloc(q,n)) == 0 ) { \
	      fprintf(stderr, "%s:%d error allocating memory\n", \
                                                       __FILE__, __LINE__); \
              exit(1); \
            } \
            __p;  \
	 })
 #else
   #error "RECALLOC previously defined, not re-defined"
 #endif



 #ifndef FOPEN
    #define FOPEN(filename, mode) \
       ({ \
          FILE *__fp; \
          char *__fstr=filename, *__mstr=mode; \
          if ( (__fp=fopen(__fstr,__mstr)) == 0 ) { \
            fprintf(stderr,"%s:%d\n Error opening %s for mode %s\n", \
	                                     __FILE__, __LINE__, \
		                             __fstr, __mstr); \
            exit(1); \
          } \
          __fp; \
       })

 #else
   #error "FOPEN previously defined, not re-defined"
 #endif

 /* The min and max macros are from the GNU cpp info pages */
 #ifndef min
   #define min(X, Y)                     \
     ({ typeof (X) __x = (X), __y = (Y);   \
        (__x < __y) ? __x : __y; })
 #else
   #error "min previously defined, not re-defined"
 #endif

 #ifndef max
   #define max(X, Y)                     \
     ({ typeof (X) __x = (X), __y = (Y);   \
        (__x > __y) ? __x : __y; })
 #else
   #error "max previously defined, not re-defined"
 #endif

 #ifndef chomp
   #define chomp(q) \
      do { \
         register char *__p=q; \
         register int __plen=strlen(__p); \
         if ( __p != 0 && __plen > 0 ) \
	   __p[__plen-1]=0; \
      } while(0)
 #else
   #error "chomp previously defined, not re-defined"
 #endif

 #ifndef push
   #define REALLOC_INCR 100
   #define push(array, element, array_count) \
      do { \
         register int __tmp_count=array_count; \
         if ( __tmp_count % REALLOC_INCR == 0 ) \
             array=(typeof(array))realloc(array, \
                                            (__tmp_count+REALLOC_INCR)* \
                                              sizeof(typeof(*array))); \
         if ( ! array ) { \
            fprintf(stderr,"Memory allocation failure\n"); \
            exit(1); \
         }\
         array[__tmp_count]=element; \
      } while (0)
 #else
   #error "push previously defined, not re-defined"
 #endif

 #ifndef pop
    #define pop(array,array_count) \
        ({ \
          register int __tmp_count=array_count; \
          typeof(*array) __retval=0; \
          if ( __tmp_count-1 >= 0 ) \
             __retval=array[__tmp_count-1]; \
          __retval; \
        })
 #else
   #error "pop previously defined, not re-defined"
 #endif


#else
 #error "Some macros not included since GCC is not being used."
#endif


#ifndef _BOOLEAN_TYPE_
#define _BOOLEAN_TYPE_
typedef enum {False, True} Boolean;
#endif

#endif


