//go:build ignore

#define _UCRT
#define __CRT__NO_INLINE

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wchar.h>
#include <io.h>
#include <process.h>
#include <float.h>
#include <sys/types.h>
#include <sys/timeb.h>
#include <locale.h>
#include <malloc.h>

int __cdecl _mkdir(const char *_Path);

int __cdecl _fstat64(int _FileDes,struct _stat64 *_Stat);
int __cdecl _stat64(const char *_Name,struct _stat64 *_Stat);
int __cdecl _stat64i32(const char *_Name,struct _stat64i32 *_Stat);
