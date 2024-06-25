// Code generated for windows/amd64 by 'ccgo --cpp=/usr/bin/x86_64-w64-mingw32-gcc --goos=windows --package-name libc --prefix-external=X --prefix-field=F --prefix-static-internal=_ --prefix-static-none=_ --prefix-tagged-struct=T --prefix-tagged-union=T --prefix-typename=T --winapi=stdlib.h --winapi=string.h --winapi=time.h --winapi=wchar.h -build-lines=  -eval-all-macros -hide __acrt_iob_func -hide __mingw_strtod -hide __mingw_vfwprintf -hide __mingw_vfwscanf -hide __mingw_vsnwprintf -hide __mingw_vswscanf -hide _byteswap_uint64 -hide _byteswap_ulong -hide _errno -hide _exit -hide _gmtime64 -hide _localtime64 -hide _mktime64 -hide _set_abort_behavior -hide _snwprintf -hide _strdup -hide _stricmp -hide _strnicmp -hide _time64 -hide _vsnwprintf -hide _wcsicmp -hide _wcsnicmp -hide _wgetenv -hide _wopen -hide _wputenv -hide _wtoi -hide _wunlink -hide abort -hide abs -hide atexit -hide atof -hide atoi -hide atol -hide bsearch -hide calloc -hide div -hide exit -hide free -hide getenv -hide labs -hide ldiv -hide llabs -hide lldiv -hide malloc -hide mblen -hide mbstowcs -hide mbtowc -hide memchr -hide memcmp -hide memcpy -hide memmove -hide memset -hide perror -hide putenv -hide qsort -hide rand -hide realloc -hide strcasecmp -hide strcat -hide strchr -hide strcmp -hide strcpy -hide strcspn -hide strdup -hide strerror -hide strftime -hide strlen -hide strncmp -hide strncpy -hide strpbrk -hide strrchr -hide strspn -hide strstr -hide strtol -hide strtoul -hide strtoull -hide system -hide tzset -hide wcrtomb -hide wcschr -hide wcscmp -hide wcscpy -hide wcsicmp -hide wcslen -hide wcsncmp -hide wcsrtombs -hide wcstombs -hide wctomb -import syscall -o msvcrt_windows_amd64.go --prefix-macro m libmsvcrt.c', DO NOT EDIT.

package libc

import (
	"reflect"
	"unsafe"

	"syscall"
)

var (
	_ reflect.Type
	_ unsafe.Pointer
)

const mCLK_TCK = 1000
const mCLOCKS_PER_SEC = 1000
const mCLOCK_MONOTONIC = 1
const mCLOCK_PROCESS_CPUTIME_ID = 2
const mCLOCK_REALTIME = 0
const mCLOCK_REALTIME_COARSE = 4
const mCLOCK_THREAD_CPUTIME_ID = 3
const mE2BIG = 7
const mEACCES = 13
const mEADDRINUSE = 100
const mEADDRNOTAVAIL = 101
const mEAFNOSUPPORT = 102
const mEAGAIN = 11
const mEALREADY = 103
const mEBADF = 9
const mEBADMSG = 104
const mEBUSY = 16
const mECANCELED = 105
const mECHILD = 10
const mECONNABORTED = 106
const mECONNREFUSED = 107
const mECONNRESET = 108
const mEDEADLK = 36
const mEDEADLOCK = 36
const mEDESTADDRREQ = 109
const mEDOM = 33
const mEEXIST = 17
const mEFAULT = 14
const mEFBIG = 27
const mEHOSTUNREACH = 110
const mEIDRM = 111
const mEILSEQ = 42
const mEINPROGRESS = 112
const mEINTR = 4
const mEINVAL = 22
const mEIO = 5
const mEISCONN = 113
const mEISDIR = 21
const mELOOP = 114
const mEMFILE = 24
const mEMLINK = 31
const mEMSGSIZE = 115
const mENAMETOOLONG = 38
const mENETDOWN = 116
const mENETRESET = 117
const mENETUNREACH = 118
const mENFILE = 23
const mENOBUFS = 119
const mENODATA = 120
const mENODEV = 19
const mENOENT = 2
const mENOEXEC = 8
const mENOFILE = 2
const mENOLCK = 39
const mENOLINK = 121
const mENOMEM = 12
const mENOMSG = 122
const mENOPROTOOPT = 123
const mENOSPC = 28
const mENOSR = 124
const mENOSTR = 125
const mENOSYS = 40
const mENOTCONN = 126
const mENOTDIR = 20
const mENOTEMPTY = 41
const mENOTRECOVERABLE = 127
const mENOTSOCK = 128
const mENOTSUP = 129
const mENOTTY = 25
const mENXIO = 6
const mEOPNOTSUPP = 130
const mEOVERFLOW = 132
const mEOWNERDEAD = 133
const mEPERM = 1
const mEPIPE = 32
const mEPROTO = 134
const mEPROTONOSUPPORT = 135
const mEPROTOTYPE = 136
const mERANGE = 34
const mEROFS = 30
const mESPIPE = 29
const mESRCH = 3
const mETIME = 137
const mETIMEDOUT = 138
const mETXTBSY = 139
const mEWOULDBLOCK = 140
const mEXDEV = 18
const mEXIT_FAILURE = 1
const mEXIT_SUCCESS = 0
const mMB_CUR_MAX = 0
const mMB_LEN_MAX = 5
const mMINGW_HAS_DDK_H = 1
const mMINGW_HAS_SECURE_API = 1
const mPATH_MAX = 260
const mRAND_MAX = 32767
const mSIZE_MAX = 18446744073709551615
const mSSIZE_MAX = 9223372036854775807
const mSTRUNCATE = 80
const mTIMER_ABSTIME = 1
const mUNALIGNED = 0
const mUSE___UUIDOF = 0
const mWCHAR_MAX = 65535
const mWCHAR_MIN = 0
const mWIN32 = 1
const mWIN64 = 1
const mWINNT = 1
const m_ALLOCA_S_HEAP_MARKER = 56797
const m_ALLOCA_S_MARKER_SIZE = 16
const m_ALLOCA_S_STACK_MARKER = 52428
const m_ALLOCA_S_THRESHOLD = 1024
const m_ALPHA = 259
const m_ANONYMOUS_STRUCT = 0
const m_ANONYMOUS_UNION = 0
const m_ARGMAX = 100
const m_BLANK = 64
const m_CALL_REPORTFAULT = 2
const m_CONTROL = 32
const m_CRTIMP2 = "_CRTIMP"
const m_CRTIMP_ALTERNATIVE = "_CRTIMP"
const m_CRTIMP_NOIA64 = "_CRTIMP"
const m_CRTIMP_PURE = "_CRTIMP"
const m_CRT_INTERNAL_LOCAL_PRINTF_OPTIONS = 4
const m_CRT_INTERNAL_LOCAL_SCANF_OPTIONS = 2
const m_CRT_INTERNAL_PRINTF_LEGACY_MSVCRT_COMPATIBILITY = 8
const m_CRT_INTERNAL_PRINTF_LEGACY_THREE_DIGIT_EXPONENTS = 16
const m_CRT_INTERNAL_PRINTF_LEGACY_VSPRINTF_NULL_TERMINATION = 1
const m_CRT_INTERNAL_PRINTF_LEGACY_WIDE_SPECIFIERS = 4
const m_CRT_INTERNAL_PRINTF_STANDARD_SNPRINTF_BEHAVIOR = 2
const m_CRT_INTERNAL_SCANF_LEGACY_MSVCRT_COMPATIBILITY = 4
const m_CRT_INTERNAL_SCANF_LEGACY_WIDE_SPECIFIERS = 2
const m_CRT_INTERNAL_SCANF_SECURECRT = 1
const m_CRT_SECURE_CPP_NOTHROW = 0
const m_CVTBUFSIZE = 349
const m_DIGIT = 4
const m_FREEENTRY = 0
const m_HEAPBADBEGIN = -3
const m_HEAPBADNODE = -4
const m_HEAPBADPTR = -6
const m_HEAPEMPTY = -1
const m_HEAPEND = -5
const m_HEAPOK = -2
const m_HEAP_MAXREQ = 18446744073709551584
const m_HEX = 128
const m_I16_MAX = 32767
const m_I16_MIN = -32768
const m_I32_MAX = 2147483647
const m_I32_MIN = -2147483648
const m_I64_MAX = 9223372036854775807
const m_I64_MIN = -9223372036854775808
const m_I8_MAX = 127
const m_I8_MIN = -128
const m_INTEGRAL_MAX_BITS = 64
const m_LEADBYTE = 32768
const m_LOWER = 2
const m_MAX_DIR = 256
const m_MAX_DRIVE = 3
const m_MAX_ENV = 32767
const m_MAX_EXT = 256
const m_MAX_FNAME = 256
const m_MAX_PATH = 260
const m_MAX_WAIT_MALLOC_CRT = 60000
const m_MCRTIMP = "_CRTIMP"
const m_MRTIMP2 = "_CRTIMP"
const m_M_AMD64 = 100
const m_M_X64 = 100
const m_NLSCMPERROR = 2147483647
const m_OUT_TO_DEFAULT = 0
const m_OUT_TO_MSGBOX = 2
const m_OUT_TO_STDERR = 1
const m_POSIX_CPUTIME = 200809
const m_POSIX_MONOTONIC_CLOCK = 200809
const m_POSIX_THREAD_CPUTIME = 200809
const m_POSIX_TIMERS = 200809
const m_PUNCT = 16
const m_REPORT_ERRMODE = 3
const m_SECURECRT_FILL_BUFFER_PATTERN = 253
const m_SPACE = 8
const m_TRUNCATE = -1
const m_UI16_MAX = 65535
const m_UI32_MAX = 4294967295
const m_UI64_MAX = 18446744073709551615
const m_UI8_MAX = 255
const m_UPPER = 1
const m_USEDENTRY = 1
const m_WConst_return = 0
const m_WIN32 = 1
const m_WIN32_WINNT = 2560
const m_WIN64 = 1
const m_WRITE_ABORT_MSG = 1
const m__ATOMIC_ACQUIRE = 2
const m__ATOMIC_ACQ_REL = 4
const m__ATOMIC_CONSUME = 1
const m__ATOMIC_HLE_ACQUIRE = 65536
const m__ATOMIC_HLE_RELEASE = 131072
const m__ATOMIC_RELAXED = 0
const m__ATOMIC_RELEASE = 3
const m__ATOMIC_SEQ_CST = 5
const m__BIGGEST_ALIGNMENT__ = 16
const m__BYTE_ORDER__ = 1234
const m__C89_NAMELESS = 0
const m__CCGO__ = 1
const m__CHAR_BIT__ = 8
const m__CRTDECL = "__cdecl"
const m__DBL_DECIMAL_DIG__ = 17
const m__DBL_DIG__ = 15
const m__DBL_HAS_DENORM__ = 1
const m__DBL_HAS_INFINITY__ = 1
const m__DBL_HAS_QUIET_NAN__ = 1
const m__DBL_IS_IEC_60559__ = 2
const m__DBL_MANT_DIG__ = 53
const m__DBL_MAX_10_EXP__ = 308
const m__DBL_MAX_EXP__ = 1024
const m__DBL_MIN_10_EXP__ = -307
const m__DBL_MIN_EXP__ = -1021
const m__DEC128_EPSILON__ = 0
const m__DEC128_MANT_DIG__ = 34
const m__DEC128_MAX_EXP__ = 6145
const m__DEC128_MAX__ = 0
const m__DEC128_MIN_EXP__ = -6142
const m__DEC128_MIN__ = 0
const m__DEC128_SUBNORMAL_MIN__ = 0
const m__DEC32_EPSILON__ = 0
const m__DEC32_MANT_DIG__ = 7
const m__DEC32_MAX_EXP__ = 97
const m__DEC32_MAX__ = 0
const m__DEC32_MIN_EXP__ = -94
const m__DEC32_MIN__ = 0
const m__DEC32_SUBNORMAL_MIN__ = 0
const m__DEC64_EPSILON__ = 0
const m__DEC64_MANT_DIG__ = 16
const m__DEC64_MAX_EXP__ = 385
const m__DEC64_MAX__ = 0
const m__DEC64_MIN_EXP__ = -382
const m__DEC64_MIN__ = 0
const m__DEC64_SUBNORMAL_MIN__ = 0
const m__DECIMAL_BID_FORMAT__ = 1
const m__DECIMAL_DIG__ = 17
const m__DEC_EVAL_METHOD__ = 2
const m__FINITE_MATH_ONLY__ = 0
const m__FLOAT_WORD_ORDER__ = 1234
const m__FLT128_DECIMAL_DIG__ = 36
const m__FLT128_DENORM_MIN__ = 0
const m__FLT128_DIG__ = 33
const m__FLT128_EPSILON__ = 0
const m__FLT128_HAS_DENORM__ = 1
const m__FLT128_HAS_INFINITY__ = 1
const m__FLT128_HAS_QUIET_NAN__ = 1
const m__FLT128_IS_IEC_60559__ = 2
const m__FLT128_MANT_DIG__ = 113
const m__FLT128_MAX_10_EXP__ = 4932
const m__FLT128_MAX_EXP__ = 16384
const m__FLT128_MAX__ = 0
const m__FLT128_MIN_10_EXP__ = -4931
const m__FLT128_MIN_EXP__ = -16381
const m__FLT128_MIN__ = 0
const m__FLT128_NORM_MAX__ = 0
const m__FLT32X_DECIMAL_DIG__ = 17
const m__FLT32X_DENORM_MIN__ = 0
const m__FLT32X_DIG__ = 15
const m__FLT32X_EPSILON__ = 0
const m__FLT32X_HAS_DENORM__ = 1
const m__FLT32X_HAS_INFINITY__ = 1
const m__FLT32X_HAS_QUIET_NAN__ = 1
const m__FLT32X_IS_IEC_60559__ = 2
const m__FLT32X_MANT_DIG__ = 53
const m__FLT32X_MAX_10_EXP__ = 308
const m__FLT32X_MAX_EXP__ = 1024
const m__FLT32X_MAX__ = 0
const m__FLT32X_MIN_10_EXP__ = -307
const m__FLT32X_MIN_EXP__ = -1021
const m__FLT32X_MIN__ = 0
const m__FLT32X_NORM_MAX__ = 0
const m__FLT32_DECIMAL_DIG__ = 9
const m__FLT32_DENORM_MIN__ = 0
const m__FLT32_DIG__ = 6
const m__FLT32_EPSILON__ = 0
const m__FLT32_HAS_DENORM__ = 1
const m__FLT32_HAS_INFINITY__ = 1
const m__FLT32_HAS_QUIET_NAN__ = 1
const m__FLT32_IS_IEC_60559__ = 2
const m__FLT32_MANT_DIG__ = 24
const m__FLT32_MAX_10_EXP__ = 38
const m__FLT32_MAX_EXP__ = 128
const m__FLT32_MAX__ = 0
const m__FLT32_MIN_10_EXP__ = -37
const m__FLT32_MIN_EXP__ = -125
const m__FLT32_MIN__ = 0
const m__FLT32_NORM_MAX__ = 0
const m__FLT64X_DECIMAL_DIG__ = 36
const m__FLT64X_DENORM_MIN__ = 0
const m__FLT64X_DIG__ = 33
const m__FLT64X_EPSILON__ = 0
const m__FLT64X_HAS_DENORM__ = 1
const m__FLT64X_HAS_INFINITY__ = 1
const m__FLT64X_HAS_QUIET_NAN__ = 1
const m__FLT64X_IS_IEC_60559__ = 2
const m__FLT64X_MANT_DIG__ = 113
const m__FLT64X_MAX_10_EXP__ = 4932
const m__FLT64X_MAX_EXP__ = 16384
const m__FLT64X_MAX__ = 0
const m__FLT64X_MIN_10_EXP__ = -4931
const m__FLT64X_MIN_EXP__ = -16381
const m__FLT64X_MIN__ = 0
const m__FLT64X_NORM_MAX__ = 0
const m__FLT64_DECIMAL_DIG__ = 17
const m__FLT64_DENORM_MIN__ = 0
const m__FLT64_DIG__ = 15
const m__FLT64_EPSILON__ = 0
const m__FLT64_HAS_DENORM__ = 1
const m__FLT64_HAS_INFINITY__ = 1
const m__FLT64_HAS_QUIET_NAN__ = 1
const m__FLT64_IS_IEC_60559__ = 2
const m__FLT64_MANT_DIG__ = 53
const m__FLT64_MAX_10_EXP__ = 308
const m__FLT64_MAX_EXP__ = 1024
const m__FLT64_MAX__ = 0
const m__FLT64_MIN_10_EXP__ = -307
const m__FLT64_MIN_EXP__ = -1021
const m__FLT64_MIN__ = 0
const m__FLT64_NORM_MAX__ = 0
const m__FLT_DECIMAL_DIG__ = 9
const m__FLT_DENORM_MIN__ = 0
const m__FLT_DIG__ = 6
const m__FLT_EPSILON__ = 0
const m__FLT_EVAL_METHOD_TS_18661_3__ = 2
const m__FLT_EVAL_METHOD__ = 2
const m__FLT_HAS_DENORM__ = 1
const m__FLT_HAS_INFINITY__ = 1
const m__FLT_HAS_QUIET_NAN__ = 1
const m__FLT_IS_IEC_60559__ = 2
const m__FLT_MANT_DIG__ = 24
const m__FLT_MAX_10_EXP__ = 38
const m__FLT_MAX_EXP__ = 128
const m__FLT_MAX__ = 0
const m__FLT_MIN_10_EXP__ = -37
const m__FLT_MIN_EXP__ = -125
const m__FLT_MIN__ = 0
const m__FLT_NORM_MAX__ = 0
const m__FLT_RADIX__ = 2
const m__FUNCTION__ = 0
const m__FXSR__ = 1
const m__GCC_ASM_FLAG_OUTPUTS__ = 1
const m__GCC_ATOMIC_BOOL_LOCK_FREE = 2
const m__GCC_ATOMIC_CHAR16_T_LOCK_FREE = 2
const m__GCC_ATOMIC_CHAR32_T_LOCK_FREE = 2
const m__GCC_ATOMIC_CHAR_LOCK_FREE = 2
const m__GCC_ATOMIC_INT_LOCK_FREE = 2
const m__GCC_ATOMIC_LLONG_LOCK_FREE = 2
const m__GCC_ATOMIC_LONG_LOCK_FREE = 2
const m__GCC_ATOMIC_POINTER_LOCK_FREE = 2
const m__GCC_ATOMIC_SHORT_LOCK_FREE = 2
const m__GCC_ATOMIC_TEST_AND_SET_TRUEVAL = 1
const m__GCC_ATOMIC_WCHAR_T_LOCK_FREE = 2
const m__GCC_CONSTRUCTIVE_SIZE = 64
const m__GCC_DESTRUCTIVE_SIZE = 64
const m__GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 = 1
const m__GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 = 1
const m__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 = 1
const m__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 = 1
const m__GCC_IEC_559 = 2
const m__GCC_IEC_559_COMPLEX = 2
const m__GNUC_EXECUTION_CHARSET_NAME = "UTF-8"
const m__GNUC_MINOR__ = 0
const m__GNUC_PATCHLEVEL__ = 0
const m__GNUC_STDC_INLINE__ = 1
const m__GNUC_WIDE_EXECUTION_CHARSET_NAME = "UTF-16LE"
const m__GNUC__ = 12
const m__GNU_EXTENSION = 0
const m__GOT_SECURE_LIB__ = 200411
const m__GXX_ABI_VERSION = 1017
const m__GXX_MERGED_TYPEINFO_NAMES = 0
const m__GXX_TYPEINFO_EQUALITY_INLINE = 0
const m__HAVE_SPECULATION_SAFE_VALUE = 1
const m__INT16_MAX__ = 32767
const m__INT32_MAX__ = 2147483647
const m__INT32_TYPE__ = 0
const m__INT64_MAX__ = 9223372036854775807
const m__INT8_MAX__ = 127
const m__INTMAX_MAX__ = 9223372036854775807
const m__INTMAX_WIDTH__ = 64
const m__INTPTR_MAX__ = 9223372036854775807
const m__INTPTR_WIDTH__ = 64
const m__INT_FAST16_MAX__ = 32767
const m__INT_FAST16_WIDTH__ = 16
const m__INT_FAST32_MAX__ = 2147483647
const m__INT_FAST32_TYPE__ = 0
const m__INT_FAST32_WIDTH__ = 32
const m__INT_FAST64_MAX__ = 9223372036854775807
const m__INT_FAST64_WIDTH__ = 64
const m__INT_FAST8_MAX__ = 127
const m__INT_FAST8_WIDTH__ = 8
const m__INT_LEAST16_MAX__ = 32767
const m__INT_LEAST16_WIDTH__ = 16
const m__INT_LEAST32_MAX__ = 2147483647
const m__INT_LEAST32_TYPE__ = 0
const m__INT_LEAST32_WIDTH__ = 32
const m__INT_LEAST64_MAX__ = 9223372036854775807
const m__INT_LEAST64_WIDTH__ = 64
const m__INT_LEAST8_MAX__ = 127
const m__INT_LEAST8_WIDTH__ = 8
const m__INT_MAX__ = 2147483647
const m__INT_WIDTH__ = 32
const m__LDBL_DECIMAL_DIG__ = 17
const m__LDBL_DENORM_MIN__ = 0
const m__LDBL_DIG__ = 15
const m__LDBL_EPSILON__ = 0
const m__LDBL_HAS_DENORM__ = 1
const m__LDBL_HAS_INFINITY__ = 1
const m__LDBL_HAS_QUIET_NAN__ = 1
const m__LDBL_IS_IEC_60559__ = 2
const m__LDBL_MANT_DIG__ = 53
const m__LDBL_MAX_10_EXP__ = 308
const m__LDBL_MAX_EXP__ = 1024
const m__LDBL_MAX__ = 0
const m__LDBL_MIN_10_EXP__ = -307
const m__LDBL_MIN_EXP__ = -1021
const m__LDBL_MIN__ = 0
const m__LDBL_NORM_MAX__ = 0
const m__LONG32 = 0
const m__LONG_DOUBLE_64__ = 1
const m__LONG_LONG_MAX__ = 9223372036854775807
const m__LONG_LONG_WIDTH__ = 64
const m__LONG_MAX__ = 2147483647
const m__LONG_WIDTH__ = 32
const m__MINGW32_MAJOR_VERSION = 3
const m__MINGW32_MINOR_VERSION = 11
const m__MINGW32__ = 1
const m__MINGW64_VERSION_BUGFIX = 0
const m__MINGW64_VERSION_MAJOR = 10
const m__MINGW64_VERSION_MINOR = 0
const m__MINGW64_VERSION_RC = 0
const m__MINGW64_VERSION_STATE = "alpha"
const m__MINGW64__ = 1
const m__MINGW_ATTRIB_DEPRECATED_MSVC2005 = 0
const m__MINGW_ATTRIB_DEPRECATED_SEC_WARN = 0
const m__MINGW_DEBUGBREAK_IMPL = 1
const m__MINGW_FORTIFY_LEVEL = 0
const m__MINGW_FORTIFY_VA_ARG = 0
const m__MINGW_GCC_VERSION = 120000
const m__MINGW_HAVE_ANSI_C99_PRINTF = 1
const m__MINGW_HAVE_ANSI_C99_SCANF = 1
const m__MINGW_HAVE_WIDE_C99_PRINTF = 1
const m__MINGW_HAVE_WIDE_C99_SCANF = 1
const m__MINGW_MSVC2005_DEPREC_STR = "This POSIX function is deprecated beginning in Visual C++ 2005, use _CRT_NONSTDC_NO_DEPRECATE to disable deprecation"
const m__MINGW_SEC_WARN_STR = "This function or variable may be unsafe, use _CRT_SECURE_NO_WARNINGS to disable deprecation"
const m__MINGW_USE_UNDERSCORE_PREFIX = 0
const m__MSVCRT_VERSION__ = 1792
const m__MSVCRT__ = 1
const m__NO_INLINE__ = 1
const m__ORDER_BIG_ENDIAN__ = 4321
const m__ORDER_LITTLE_ENDIAN__ = 1234
const m__ORDER_PDP_ENDIAN__ = 3412
const m__PCTYPE_FUNC = 0
const m__PIC__ = 1
const m__PRAGMA_REDEFINE_EXTNAME = 1
const m__PRETTY_FUNCTION__ = 0
const m__PTRDIFF_MAX__ = 9223372036854775807
const m__PTRDIFF_WIDTH__ = 64
const m__SCHAR_MAX__ = 127
const m__SCHAR_WIDTH__ = 8
const m__SEG_FS = 1
const m__SEG_GS = 1
const m__SEH__ = 1
const m__SHRT_MAX__ = 32767
const m__SHRT_WIDTH__ = 16
const m__SIG_ATOMIC_MAX__ = 2147483647
const m__SIG_ATOMIC_MIN__ = -2147483648
const m__SIG_ATOMIC_TYPE__ = 0
const m__SIG_ATOMIC_WIDTH__ = 32
const m__SIZEOF_DOUBLE__ = 8
const m__SIZEOF_FLOAT128__ = 16
const m__SIZEOF_FLOAT80__ = 16
const m__SIZEOF_FLOAT__ = 4
const m__SIZEOF_INT128__ = 16
const m__SIZEOF_INT__ = 4
const m__SIZEOF_LONG_DOUBLE__ = 8
const m__SIZEOF_LONG_LONG__ = 8
const m__SIZEOF_LONG__ = 4
const m__SIZEOF_POINTER__ = 8
const m__SIZEOF_PTRDIFF_T__ = 8
const m__SIZEOF_SHORT__ = 2
const m__SIZEOF_SIZE_T__ = 8
const m__SIZEOF_WCHAR_T__ = 2
const m__SIZEOF_WINT_T__ = 2
const m__SIZE_MAX__ = 18446744073709551615
const m__SIZE_WIDTH__ = 64
const m__STDC_HOSTED__ = 1
const m__STDC_SECURE_LIB__ = 200411
const m__STDC_UTF_16__ = 1
const m__STDC_UTF_32__ = 1
const m__STDC_VERSION__ = 201710
const m__STDC__ = 1
const m__UINT16_MAX__ = 65535
const m__UINT32_MAX__ = 4294967295
const m__UINT64_MAX__ = 18446744073709551615
const m__UINT8_MAX__ = 255
const m__UINTMAX_MAX__ = 18446744073709551615
const m__UINTPTR_MAX__ = 18446744073709551615
const m__UINT_FAST16_MAX__ = 65535
const m__UINT_FAST32_MAX__ = 4294967295
const m__UINT_FAST64_MAX__ = 18446744073709551615
const m__UINT_FAST8_MAX__ = 255
const m__UINT_LEAST16_MAX__ = 65535
const m__UINT_LEAST32_MAX__ = 4294967295
const m__UINT_LEAST64_MAX__ = 18446744073709551615
const m__UINT_LEAST8_MAX__ = 255
const m__USE_MINGW_ANSI_STDIO = 1
const m__USE_MINGW_STRTOX = 1
const m__VERSION__ = "12-win32"
const m__WCHAR_MAX__ = 65535
const m__WCHAR_MIN__ = 0
const m__WCHAR_WIDTH__ = 16
const m__WIN32 = 1
const m__WIN32__ = 1
const m__WIN64 = 1
const m__WIN64__ = 1
const m__WINNT = 1
const m__WINNT__ = 1
const m__WINT_MAX__ = 65535
const m__WINT_MIN__ = 0
const m__WINT_WIDTH__ = 16
const m__amd64 = 1
const m__amd64__ = 1
const m__argc = 0
const m__argv = 0
const m__clockid_t_defined = 1
const m__code_model_medium__ = 1
const m__int16 = 0
const m__int32 = 0
const m__int8 = 0
const m__k8 = 1
const m__k8__ = 1
const m__mb_cur_max = 0
const m__mingw_bos_ovr = "__mingw_ovr"
const m__pic__ = 1
const m__stat64 = 0
const m__wargv = 0
const m__x86_64 = 1
const m__x86_64__ = 1
const m_doserrno = 0
const m_environ = 0
const m_fmode = 0
const m_fstat = 0
const m_fstati64 = 0
const m_ftime = 0
const m_ftime_s = 0
const m_inline = 0
const m_iob = 0
const m_osplatform = 0
const m_osver = 0
const m_pctype = 0
const m_pgmptr = 0
const m_pwctype = 0
const m_stat = 0
const m_stati64 = 0
const m_timeb = 0
const m_wctype = 0
const m_wenviron = 0
const m_wfinddata_t = 0
const m_wfinddatai64_t = 0
const m_wfindfirst = 0
const m_wfindfirsti64 = 0
const m_wfindnext = 0
const m_wfindnexti64 = 0
const m_winmajor = 0
const m_winminor = 0
const m_winver = 0
const m_wpgmptr = 0
const m_wstat = 0
const m_wstati64 = 0
const menviron = 0
const merrno = 0
const mfstat64 = 0
const monexit_t = 0
const mstat64 = 0
const mstderr = 0
const mstdin = 0
const mstdout = 0
const mstrcasecmp = 0
const mstrncasecmp = 0
const msys_errlist = 0
const msys_nerr = 0
const mwcswcs = 0
const mwpopen = 0

type T__builtin_va_list = uintptr

type T__predefined_size_t = uint64

type T__predefined_wchar_t = uint16

type T__predefined_ptrdiff_t = int64

type T__gnuc_va_list = uintptr

type Tva_list = uintptr

type Tsize_t = uint64

type Tssize_t = int64

type Trsize_t = uint64

type Tintptr_t = int64

type Tuintptr_t = uint64

type Tptrdiff_t = int64

type Twchar_t = uint16

type Twint_t = uint16

type Twctype_t = uint16

type Terrno_t = int32

type T__time32_t = int32

type T__time64_t = int64

type Ttime_t = int64

type Tthreadlocaleinfostruct = struct {
	Frefcount      int32
	Flc_codepage   uint32
	Flc_collate_cp uint32
	Flc_handle     [6]uint32
	Flc_id         [6]TLC_ID
	Flc_category   [6]struct {
		Flocale    uintptr
		Fwlocale   uintptr
		Frefcount  uintptr
		Fwrefcount uintptr
	}
	Flc_clike            int32
	Fmb_cur_max          int32
	Flconv_intl_refcount uintptr
	Flconv_num_refcount  uintptr
	Flconv_mon_refcount  uintptr
	Flconv               uintptr
	Fctype1_refcount     uintptr
	Fctype1              uintptr
	Fpctype              uintptr
	Fpclmap              uintptr
	Fpcumap              uintptr
	Flc_time_curr        uintptr
}

type Tpthreadlocinfo = uintptr

type Tpthreadmbcinfo = uintptr

type T_locale_tstruct = struct {
	Flocinfo Tpthreadlocinfo
	Fmbcinfo Tpthreadmbcinfo
}

type Tlocaleinfo_struct = T_locale_tstruct

type T_locale_t = uintptr

type TLC_ID = struct {
	FwLanguage uint16
	FwCountry  uint16
	FwCodePage uint16
}

type TtagLC_ID = TLC_ID

type TLPLC_ID = uintptr

type Tthreadlocinfo = struct {
	Frefcount      int32
	Flc_codepage   uint32
	Flc_collate_cp uint32
	Flc_handle     [6]uint32
	Flc_id         [6]TLC_ID
	Flc_category   [6]struct {
		Flocale    uintptr
		Fwlocale   uintptr
		Frefcount  uintptr
		Fwrefcount uintptr
	}
	Flc_clike            int32
	Fmb_cur_max          int32
	Flconv_intl_refcount uintptr
	Flconv_num_refcount  uintptr
	Flconv_mon_refcount  uintptr
	Flconv               uintptr
	Fctype1_refcount     uintptr
	Fctype1              uintptr
	Fpctype              uintptr
	Fpclmap              uintptr
	Fpcumap              uintptr
	Flc_time_curr        uintptr
}

type T_iobuf = struct {
	F_ptr      uintptr
	F_cnt      int32
	F_base     uintptr
	F_flag     int32
	F_file     int32
	F_charbuf  int32
	F_bufsiz   int32
	F_tmpfname uintptr
}

type TFILE = struct {
	F_ptr      uintptr
	F_cnt      int32
	F_base     uintptr
	F_flag     int32
	F_file     int32
	F_charbuf  int32
	F_bufsiz   int32
	F_tmpfname uintptr
}

var proc__iob_func = modcrt.NewProc(GoString(__ccgo_ts))

// __attribute__ ((__dllimport__)) FILE * __attribute__((__cdecl__)) __iob_func(void);
func X__iob_func(tls *TLS) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc__iob_func.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

type T_fsize_t = uint32

type T_wfinddata32_t = struct {
	Fattrib      uint32
	Ftime_create T__time32_t
	Ftime_access T__time32_t
	Ftime_write  T__time32_t
	Fsize        T_fsize_t
	Fname        [260]Twchar_t
}

type T_wfinddata32i64_t = struct {
	Fattrib      uint32
	Ftime_create T__time32_t
	Ftime_access T__time32_t
	Ftime_write  T__time32_t
	Fsize        int64
	Fname        [260]Twchar_t
}

type T_wfinddata64i32_t = struct {
	Fattrib      uint32
	Ftime_create T__time64_t
	Ftime_access T__time64_t
	Ftime_write  T__time64_t
	Fsize        T_fsize_t
	Fname        [260]Twchar_t
}

type T_wfinddata64_t = struct {
	Fattrib      uint32
	Ftime_create T__time64_t
	Ftime_access T__time64_t
	Ftime_write  T__time64_t
	Fsize        int64
	Fname        [260]Twchar_t
}

var prociswalpha = modcrt.NewProc(GoString(__ccgo_ts + 11))

// int __attribute__((__cdecl__)) iswalpha(wint_t _C);
func Xiswalpha(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswalpha.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswalpha_l = modcrt.NewProc(GoString(__ccgo_ts + 20))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswalpha_l(wint_t _C,_locale_t _Locale);
func X_iswalpha_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswalpha_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswupper = modcrt.NewProc(GoString(__ccgo_ts + 32))

// int __attribute__((__cdecl__)) iswupper(wint_t _C);
func Xiswupper(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswupper.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswupper_l = modcrt.NewProc(GoString(__ccgo_ts + 41))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswupper_l(wint_t _C,_locale_t _Locale);
func X_iswupper_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswupper_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswlower = modcrt.NewProc(GoString(__ccgo_ts + 53))

// int __attribute__((__cdecl__)) iswlower(wint_t _C);
func Xiswlower(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswlower.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswlower_l = modcrt.NewProc(GoString(__ccgo_ts + 62))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswlower_l(wint_t _C,_locale_t _Locale);
func X_iswlower_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswlower_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswdigit = modcrt.NewProc(GoString(__ccgo_ts + 74))

// int __attribute__((__cdecl__)) iswdigit(wint_t _C);
func Xiswdigit(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswdigit.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswdigit_l = modcrt.NewProc(GoString(__ccgo_ts + 83))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswdigit_l(wint_t _C,_locale_t _Locale);
func X_iswdigit_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswdigit_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswxdigit = modcrt.NewProc(GoString(__ccgo_ts + 95))

// int __attribute__((__cdecl__)) iswxdigit(wint_t _C);
func Xiswxdigit(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswxdigit.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswxdigit_l = modcrt.NewProc(GoString(__ccgo_ts + 105))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswxdigit_l(wint_t _C,_locale_t _Locale);
func X_iswxdigit_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswxdigit_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswspace = modcrt.NewProc(GoString(__ccgo_ts + 118))

// int __attribute__((__cdecl__)) iswspace(wint_t _C);
func Xiswspace(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswspace.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswspace_l = modcrt.NewProc(GoString(__ccgo_ts + 127))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswspace_l(wint_t _C,_locale_t _Locale);
func X_iswspace_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswspace_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswpunct = modcrt.NewProc(GoString(__ccgo_ts + 139))

// int __attribute__((__cdecl__)) iswpunct(wint_t _C);
func Xiswpunct(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswpunct.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswpunct_l = modcrt.NewProc(GoString(__ccgo_ts + 148))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswpunct_l(wint_t _C,_locale_t _Locale);
func X_iswpunct_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswpunct_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswalnum = modcrt.NewProc(GoString(__ccgo_ts + 160))

// int __attribute__((__cdecl__)) iswalnum(wint_t _C);
func Xiswalnum(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswalnum.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswalnum_l = modcrt.NewProc(GoString(__ccgo_ts + 169))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswalnum_l(wint_t _C,_locale_t _Locale);
func X_iswalnum_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswalnum_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswprint = modcrt.NewProc(GoString(__ccgo_ts + 181))

// int __attribute__((__cdecl__)) iswprint(wint_t _C);
func Xiswprint(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswprint.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswprint_l = modcrt.NewProc(GoString(__ccgo_ts + 190))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswprint_l(wint_t _C,_locale_t _Locale);
func X_iswprint_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswprint_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswgraph = modcrt.NewProc(GoString(__ccgo_ts + 202))

// int __attribute__((__cdecl__)) iswgraph(wint_t _C);
func Xiswgraph(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswgraph.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswgraph_l = modcrt.NewProc(GoString(__ccgo_ts + 211))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswgraph_l(wint_t _C,_locale_t _Locale);
func X_iswgraph_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswgraph_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswcntrl = modcrt.NewProc(GoString(__ccgo_ts + 223))

// int __attribute__((__cdecl__)) iswcntrl(wint_t _C);
func Xiswcntrl(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswcntrl.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_iswcntrl_l = modcrt.NewProc(GoString(__ccgo_ts + 232))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _iswcntrl_l(wint_t _C,_locale_t _Locale);
func X_iswcntrl_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_iswcntrl_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswascii = modcrt.NewProc(GoString(__ccgo_ts + 244))

// int __attribute__((__cdecl__)) iswascii(wint_t _C);
func Xiswascii(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswascii.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procisleadbyte = modcrt.NewProc(GoString(__ccgo_ts + 253))

// int __attribute__((__cdecl__)) isleadbyte(int _C);
func Xisleadbyte(tls *TLS, __C int32) (r int32) {
	r0, _, err := syscall.SyscallN(procisleadbyte.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_isleadbyte_l = modcrt.NewProc(GoString(__ccgo_ts + 264))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _isleadbyte_l(int _C,_locale_t _Locale);
func X_isleadbyte_l(tls *TLS, __C int32, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_isleadbyte_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proctowupper = modcrt.NewProc(GoString(__ccgo_ts + 278))

// wint_t __attribute__((__cdecl__)) towupper(wint_t _C);
func Xtowupper(tls *TLS, __C Twint_t) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proctowupper.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var proc_towupper_l = modcrt.NewProc(GoString(__ccgo_ts + 287))

// __attribute__ ((__dllimport__)) wint_t __attribute__((__cdecl__)) _towupper_l(wint_t _C,_locale_t _Locale);
func X_towupper_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proc_towupper_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var proctowlower = modcrt.NewProc(GoString(__ccgo_ts + 299))

// wint_t __attribute__((__cdecl__)) towlower(wint_t _C);
func Xtowlower(tls *TLS, __C Twint_t) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proctowlower.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var proc_towlower_l = modcrt.NewProc(GoString(__ccgo_ts + 308))

// __attribute__ ((__dllimport__)) wint_t __attribute__((__cdecl__)) _towlower_l(wint_t _C,_locale_t _Locale);
func X_towlower_l(tls *TLS, __C Twint_t, __Locale T_locale_t) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proc_towlower_l.Addr(), uintptr(__C), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var prociswctype = modcrt.NewProc(GoString(__ccgo_ts + 320))

// int __attribute__((__cdecl__)) iswctype(wint_t _C,wctype_t _Type);
func Xiswctype(tls *TLS, __C Twint_t, __Type Twctype_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswctype.Addr(), uintptr(__C), uintptr(__Type))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procis_wctype = modcrt.NewProc(GoString(__ccgo_ts + 329))

// int __attribute__((__cdecl__)) is_wctype(wint_t _C,wctype_t _Type);
func Xis_wctype(tls *TLS, __C Twint_t, __Type Twctype_t) (r int32) {
	r0, _, err := syscall.SyscallN(procis_wctype.Addr(), uintptr(__C), uintptr(__Type))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var prociswblank = modcrt.NewProc(GoString(__ccgo_ts + 339))

// int __attribute__((__cdecl__)) iswblank(wint_t _C);
func Xiswblank(tls *TLS, __C Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(prociswblank.Addr(), uintptr(__C))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wgetcwd = modcrt.NewProc(GoString(__ccgo_ts + 348))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wgetcwd(wchar_t *_DstBuf,int _SizeInWords);
func X_wgetcwd(tls *TLS, __DstBuf uintptr, __SizeInWords int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wgetcwd.Addr(), __DstBuf, uintptr(__SizeInWords))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wgetdcwd = modcrt.NewProc(GoString(__ccgo_ts + 357))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wgetdcwd(int _Drive,wchar_t *_DstBuf,int _SizeInWords);
func X_wgetdcwd(tls *TLS, __Drive int32, __DstBuf uintptr, __SizeInWords int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wgetdcwd.Addr(), uintptr(__Drive), __DstBuf, uintptr(__SizeInWords))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wchdir = modcrt.NewProc(GoString(__ccgo_ts + 367))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wchdir(const wchar_t *_Path);
func X_wchdir(tls *TLS, __Path uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wchdir.Addr(), __Path)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wmkdir = modcrt.NewProc(GoString(__ccgo_ts + 375))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wmkdir(const wchar_t *_Path);
func X_wmkdir(tls *TLS, __Path uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wmkdir.Addr(), __Path)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wrmdir = modcrt.NewProc(GoString(__ccgo_ts + 383))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wrmdir(const wchar_t *_Path);
func X_wrmdir(tls *TLS, __Path uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wrmdir.Addr(), __Path)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_waccess = modcrt.NewProc(GoString(__ccgo_ts + 391))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _waccess(const wchar_t *_Filename,int _AccessMode);
func X_waccess(tls *TLS, __Filename uintptr, __AccessMode int32) (r int32) {
	r0, _, err := syscall.SyscallN(proc_waccess.Addr(), __Filename, uintptr(__AccessMode))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wchmod = modcrt.NewProc(GoString(__ccgo_ts + 400))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wchmod(const wchar_t *_Filename,int _Mode);
func X_wchmod(tls *TLS, __Filename uintptr, __Mode int32) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wchmod.Addr(), __Filename, uintptr(__Mode))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcreat = modcrt.NewProc(GoString(__ccgo_ts + 408))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcreat(const wchar_t *_Filename,int _PermissionMode);
func X_wcreat(tls *TLS, __Filename uintptr, __PermissionMode int32) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcreat.Addr(), __Filename, uintptr(__PermissionMode))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wfindfirst32 = modcrt.NewProc(GoString(__ccgo_ts + 416))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wfindfirst32(const wchar_t *_Filename,struct _wfinddata32_t *_FindData);
func X_wfindfirst32(tls *TLS, __Filename uintptr, __FindData uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wfindfirst32.Addr(), __Filename, __FindData)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wfindnext32 = modcrt.NewProc(GoString(__ccgo_ts + 430))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wfindnext32(intptr_t _FindHandle,struct _wfinddata32_t *_FindData);
func X_wfindnext32(tls *TLS, __FindHandle Tintptr_t, __FindData uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wfindnext32.Addr(), uintptr(__FindHandle), __FindData)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wrename = modcrt.NewProc(GoString(__ccgo_ts + 443))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wrename(const wchar_t *_OldFilename,const wchar_t *_NewFilename);
func X_wrename(tls *TLS, __OldFilename uintptr, __NewFilename uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wrename.Addr(), __OldFilename, __NewFilename)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wmktemp = modcrt.NewProc(GoString(__ccgo_ts + 452))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wmktemp(wchar_t *_TemplateName);
func X_wmktemp(tls *TLS, __TemplateName uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wmktemp.Addr(), __TemplateName)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wfindfirst32i64 = modcrt.NewProc(GoString(__ccgo_ts + 461))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wfindfirst32i64(const wchar_t *_Filename,struct _wfinddata32i64_t *_FindData);
func X_wfindfirst32i64(tls *TLS, __Filename uintptr, __FindData uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wfindfirst32i64.Addr(), __Filename, __FindData)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wfindfirst64i32 = modcrt.NewProc(GoString(__ccgo_ts + 478))

// intptr_t __attribute__((__cdecl__)) _wfindfirst64i32(const wchar_t *_Filename,struct _wfinddata64i32_t *_FindData);
func X_wfindfirst64i32(tls *TLS, __Filename uintptr, __FindData uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wfindfirst64i32.Addr(), __Filename, __FindData)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wfindfirst64 = modcrt.NewProc(GoString(__ccgo_ts + 495))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wfindfirst64(const wchar_t *_Filename,struct _wfinddata64_t *_FindData);
func X_wfindfirst64(tls *TLS, __Filename uintptr, __FindData uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wfindfirst64.Addr(), __Filename, __FindData)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wfindnext32i64 = modcrt.NewProc(GoString(__ccgo_ts + 509))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wfindnext32i64(intptr_t _FindHandle,struct _wfinddata32i64_t *_FindData);
func X_wfindnext32i64(tls *TLS, __FindHandle Tintptr_t, __FindData uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wfindnext32i64.Addr(), uintptr(__FindHandle), __FindData)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wfindnext64i32 = modcrt.NewProc(GoString(__ccgo_ts + 525))

// int __attribute__((__cdecl__)) _wfindnext64i32(intptr_t _FindHandle,struct _wfinddata64i32_t *_FindData);
func X_wfindnext64i32(tls *TLS, __FindHandle Tintptr_t, __FindData uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wfindnext64i32.Addr(), uintptr(__FindHandle), __FindData)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wfindnext64 = modcrt.NewProc(GoString(__ccgo_ts + 541))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wfindnext64(intptr_t _FindHandle,struct _wfinddata64_t *_FindData);
func X_wfindnext64(tls *TLS, __FindHandle Tintptr_t, __FindData uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wfindnext64.Addr(), uintptr(__FindHandle), __FindData)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wsopen_s = modcrt.NewProc(GoString(__ccgo_ts + 554))

// __attribute__ ((__dllimport__)) errno_t __attribute__((__cdecl__)) _wsopen_s(int *_FileHandle,const wchar_t *_Filename,int _OpenFlag,int _ShareFlag,int _PermissionFlag);
func X_wsopen_s(tls *TLS, __FileHandle uintptr, __Filename uintptr, __OpenFlag int32, __ShareFlag int32, __PermissionFlag int32) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_wsopen_s.Addr(), __FileHandle, __Filename, uintptr(__OpenFlag), uintptr(__ShareFlag), uintptr(__PermissionFlag))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_wsopen = modcrt.NewProc(GoString(__ccgo_ts + 564))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wsopen(const wchar_t *_Filename,int _OpenFlag,int _ShareFlag,...);
func X_wsopen(tls *TLS, __Filename uintptr, __OpenFlag int32, __ShareFlag int32, va_list uintptr) (r int32) {
	panic(651)
}

var proc_wsetlocale = modcrt.NewProc(GoString(__ccgo_ts + 572))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wsetlocale(int _Category,const wchar_t *_Locale);
func X_wsetlocale(tls *TLS, __Category int32, __Locale uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wsetlocale.Addr(), uintptr(__Category), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wexecl = modcrt.NewProc(GoString(__ccgo_ts + 584))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wexecl(const wchar_t *_Filename,const wchar_t *_ArgList,...);
func X_wexecl(tls *TLS, __Filename uintptr, __ArgList uintptr, va_list uintptr) (r Tintptr_t) {
	panic(651)
}

var proc_wexecle = modcrt.NewProc(GoString(__ccgo_ts + 592))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wexecle(const wchar_t *_Filename,const wchar_t *_ArgList,...);
func X_wexecle(tls *TLS, __Filename uintptr, __ArgList uintptr, va_list uintptr) (r Tintptr_t) {
	panic(651)
}

var proc_wexeclp = modcrt.NewProc(GoString(__ccgo_ts + 601))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wexeclp(const wchar_t *_Filename,const wchar_t *_ArgList,...);
func X_wexeclp(tls *TLS, __Filename uintptr, __ArgList uintptr, va_list uintptr) (r Tintptr_t) {
	panic(651)
}

var proc_wexeclpe = modcrt.NewProc(GoString(__ccgo_ts + 610))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wexeclpe(const wchar_t *_Filename,const wchar_t *_ArgList,...);
func X_wexeclpe(tls *TLS, __Filename uintptr, __ArgList uintptr, va_list uintptr) (r Tintptr_t) {
	panic(651)
}

var proc_wexecv = modcrt.NewProc(GoString(__ccgo_ts + 620))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wexecv(const wchar_t *_Filename,const wchar_t *const *_ArgList);
func X_wexecv(tls *TLS, __Filename uintptr, __ArgList uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wexecv.Addr(), __Filename, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wexecve = modcrt.NewProc(GoString(__ccgo_ts + 628))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wexecve(const wchar_t *_Filename,const wchar_t *const *_ArgList,const wchar_t *const *_Env);
func X_wexecve(tls *TLS, __Filename uintptr, __ArgList uintptr, __Env uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wexecve.Addr(), __Filename, __ArgList, __Env)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wexecvp = modcrt.NewProc(GoString(__ccgo_ts + 637))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wexecvp(const wchar_t *_Filename,const wchar_t *const *_ArgList);
func X_wexecvp(tls *TLS, __Filename uintptr, __ArgList uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wexecvp.Addr(), __Filename, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wexecvpe = modcrt.NewProc(GoString(__ccgo_ts + 646))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wexecvpe(const wchar_t *_Filename,const wchar_t *const *_ArgList,const wchar_t *const *_Env);
func X_wexecvpe(tls *TLS, __Filename uintptr, __ArgList uintptr, __Env uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wexecvpe.Addr(), __Filename, __ArgList, __Env)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wspawnl = modcrt.NewProc(GoString(__ccgo_ts + 656))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wspawnl(int _Mode,const wchar_t *_Filename,const wchar_t *_ArgList,...);
func X_wspawnl(tls *TLS, __Mode int32, __Filename uintptr, __ArgList uintptr, va_list uintptr) (r Tintptr_t) {
	panic(651)
}

var proc_wspawnle = modcrt.NewProc(GoString(__ccgo_ts + 665))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wspawnle(int _Mode,const wchar_t *_Filename,const wchar_t *_ArgList,...);
func X_wspawnle(tls *TLS, __Mode int32, __Filename uintptr, __ArgList uintptr, va_list uintptr) (r Tintptr_t) {
	panic(651)
}

var proc_wspawnlp = modcrt.NewProc(GoString(__ccgo_ts + 675))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wspawnlp(int _Mode,const wchar_t *_Filename,const wchar_t *_ArgList,...);
func X_wspawnlp(tls *TLS, __Mode int32, __Filename uintptr, __ArgList uintptr, va_list uintptr) (r Tintptr_t) {
	panic(651)
}

var proc_wspawnlpe = modcrt.NewProc(GoString(__ccgo_ts + 685))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wspawnlpe(int _Mode,const wchar_t *_Filename,const wchar_t *_ArgList,...);
func X_wspawnlpe(tls *TLS, __Mode int32, __Filename uintptr, __ArgList uintptr, va_list uintptr) (r Tintptr_t) {
	panic(651)
}

var proc_wspawnv = modcrt.NewProc(GoString(__ccgo_ts + 696))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wspawnv(int _Mode,const wchar_t *_Filename,const wchar_t *const *_ArgList);
func X_wspawnv(tls *TLS, __Mode int32, __Filename uintptr, __ArgList uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wspawnv.Addr(), uintptr(__Mode), __Filename, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wspawnve = modcrt.NewProc(GoString(__ccgo_ts + 705))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wspawnve(int _Mode,const wchar_t *_Filename,const wchar_t *const *_ArgList,const wchar_t *const *_Env);
func X_wspawnve(tls *TLS, __Mode int32, __Filename uintptr, __ArgList uintptr, __Env uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wspawnve.Addr(), uintptr(__Mode), __Filename, __ArgList, __Env)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wspawnvp = modcrt.NewProc(GoString(__ccgo_ts + 715))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wspawnvp(int _Mode,const wchar_t *_Filename,const wchar_t *const *_ArgList);
func X_wspawnvp(tls *TLS, __Mode int32, __Filename uintptr, __ArgList uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wspawnvp.Addr(), uintptr(__Mode), __Filename, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wspawnvpe = modcrt.NewProc(GoString(__ccgo_ts + 725))

// __attribute__ ((__dllimport__)) intptr_t __attribute__((__cdecl__)) _wspawnvpe(int _Mode,const wchar_t *_Filename,const wchar_t *const *_ArgList,const wchar_t *const *_Env);
func X_wspawnvpe(tls *TLS, __Mode int32, __Filename uintptr, __ArgList uintptr, __Env uintptr) (r Tintptr_t) {
	r0, _, err := syscall.SyscallN(proc_wspawnvpe.Addr(), uintptr(__Mode), __Filename, __ArgList, __Env)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tintptr_t(r0)
}

var proc_wsystem = modcrt.NewProc(GoString(__ccgo_ts + 736))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wsystem(const wchar_t *_Command);
func X_wsystem(tls *TLS, __Command uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wsystem.Addr(), __Command)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

type T_ino_t = uint16

type Tino_t = uint16

type T_dev_t = uint32

type Tdev_t = uint32

type T_off_t = int32

type Toff32_t = int32

type T_off64_t = int64

type Toff64_t = int64

type Toff_t = int32

type T_stat32 = struct {
	Fst_dev   T_dev_t
	Fst_ino   T_ino_t
	Fst_mode  uint16
	Fst_nlink int16
	Fst_uid   int16
	Fst_gid   int16
	Fst_rdev  T_dev_t
	Fst_size  T_off_t
	Fst_atime T__time32_t
	Fst_mtime T__time32_t
	Fst_ctime T__time32_t
}

type Tstat = struct {
	Fst_dev   T_dev_t
	Fst_ino   T_ino_t
	Fst_mode  uint16
	Fst_nlink int16
	Fst_uid   int16
	Fst_gid   int16
	Fst_rdev  T_dev_t
	Fst_size  T_off_t
	Fst_atime Ttime_t
	Fst_mtime Ttime_t
	Fst_ctime Ttime_t
}

type T_stat32i64 = struct {
	Fst_dev   T_dev_t
	Fst_ino   T_ino_t
	Fst_mode  uint16
	Fst_nlink int16
	Fst_uid   int16
	Fst_gid   int16
	Fst_rdev  T_dev_t
	Fst_size  int64
	Fst_atime T__time32_t
	Fst_mtime T__time32_t
	Fst_ctime T__time32_t
}

type T_stat64i32 = struct {
	Fst_dev   T_dev_t
	Fst_ino   T_ino_t
	Fst_mode  uint16
	Fst_nlink int16
	Fst_uid   int16
	Fst_gid   int16
	Fst_rdev  T_dev_t
	Fst_size  T_off_t
	Fst_atime T__time64_t
	Fst_mtime T__time64_t
	Fst_ctime T__time64_t
}

type T_stat64 = struct {
	Fst_dev   T_dev_t
	Fst_ino   T_ino_t
	Fst_mode  uint16
	Fst_nlink int16
	Fst_uid   int16
	Fst_gid   int16
	Fst_rdev  T_dev_t
	Fst_size  int64
	Fst_atime T__time64_t
	Fst_mtime T__time64_t
	Fst_ctime T__time64_t
}

var proc_wstat32 = modcrt.NewProc(GoString(__ccgo_ts + 745))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wstat32(const wchar_t *_Name,struct _stat32 *_Stat);
func X_wstat32(tls *TLS, __Name uintptr, __Stat uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wstat32.Addr(), __Name, __Stat)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wstat32i64 = modcrt.NewProc(GoString(__ccgo_ts + 754))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wstat32i64(const wchar_t *_Name,struct _stat32i64 *_Stat);
func X_wstat32i64(tls *TLS, __Name uintptr, __Stat uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wstat32i64.Addr(), __Name, __Stat)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wstat64i32 = modcrt.NewProc(GoString(__ccgo_ts + 766))

// int __attribute__((__cdecl__)) _wstat64i32(const wchar_t *_Name,struct _stat64i32 *_Stat);
func X_wstat64i32(tls *TLS, __Name uintptr, __Stat uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wstat64i32.Addr(), __Name, __Stat)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wstat64 = modcrt.NewProc(GoString(__ccgo_ts + 778))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wstat64(const wchar_t *_Name,struct _stat64 *_Stat);
func X_wstat64(tls *TLS, __Name uintptr, __Stat uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wstat64.Addr(), __Name, __Stat)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_cgetws = modcrt.NewProc(GoString(__ccgo_ts + 787))

// __attribute__ ((__dllimport__)) wchar_t *_cgetws(wchar_t *_Buffer);
func X_cgetws(tls *TLS, __Buffer uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_cgetws.Addr(), __Buffer)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_getwch = modcrt.NewProc(GoString(__ccgo_ts + 795))

// __attribute__ ((__dllimport__)) wint_t __attribute__((__cdecl__)) _getwch(void);
func X_getwch(tls *TLS) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proc_getwch.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var proc_getwche = modcrt.NewProc(GoString(__ccgo_ts + 803))

// __attribute__ ((__dllimport__)) wint_t __attribute__((__cdecl__)) _getwche(void);
func X_getwche(tls *TLS) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proc_getwche.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var proc_putwch = modcrt.NewProc(GoString(__ccgo_ts + 812))

// __attribute__ ((__dllimport__)) wint_t __attribute__((__cdecl__)) _putwch(wchar_t _WCh);
func X_putwch(tls *TLS, __WCh Twchar_t) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proc_putwch.Addr(), uintptr(__WCh))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var proc_ungetwch = modcrt.NewProc(GoString(__ccgo_ts + 820))

// __attribute__ ((__dllimport__)) wint_t __attribute__((__cdecl__)) _ungetwch(wint_t _WCh);
func X_ungetwch(tls *TLS, __WCh Twint_t) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proc_ungetwch.Addr(), uintptr(__WCh))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var proc_cputws = modcrt.NewProc(GoString(__ccgo_ts + 830))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _cputws(const wchar_t *_String);
func X_cputws(tls *TLS, __String uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_cputws.Addr(), __String)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_cwprintf = modcrt.NewProc(GoString(__ccgo_ts + 838))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _cwprintf(const wchar_t * __restrict__ _Format,...);
func X_cwprintf(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_cwscanf = modcrt.NewProc(GoString(__ccgo_ts + 848))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _cwscanf(const wchar_t * __restrict__ _Format,...);
func X_cwscanf(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_cwscanf_l = modcrt.NewProc(GoString(__ccgo_ts + 857))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _cwscanf_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_cwscanf_l(tls *TLS, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vcwprintf = modcrt.NewProc(GoString(__ccgo_ts + 868))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vcwprintf(const wchar_t * __restrict__ _Format,va_list _ArgList);
func X_vcwprintf(tls *TLS, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vcwprintf.Addr(), __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_cwprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 879))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _cwprintf_p(const wchar_t * __restrict__ _Format,...);
func X_cwprintf_p(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vcwprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 891))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vcwprintf_p(const wchar_t * __restrict__ _Format,va_list _ArgList);
func X_vcwprintf_p(tls *TLS, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vcwprintf_p.Addr(), __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_cwprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 904))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _cwprintf_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_cwprintf_l(tls *TLS, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vcwprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 916))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vcwprintf_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vcwprintf_l(tls *TLS, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vcwprintf_l.Addr(), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_cwprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 929))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _cwprintf_p_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_cwprintf_p_l(tls *TLS, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vcwprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 943))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vcwprintf_p_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vcwprintf_p_l(tls *TLS, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vcwprintf_p_l.Addr(), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__mingw_swscanf = modcrt.NewProc(GoString(__ccgo_ts + 958))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __mingw_swscanf(const wchar_t * __restrict__ _Src,const wchar_t * __restrict__ _Format,...);
func X__mingw_swscanf(tls *TLS, __Src uintptr, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__mingw_wscanf = modcrt.NewProc(GoString(__ccgo_ts + 974))

// __attribute__ ((__nonnull__ (1))) int __attribute__((__cdecl__)) __mingw_wscanf(const wchar_t * __restrict__ _Format,...);
func X__mingw_wscanf(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__mingw_vwscanf = modcrt.NewProc(GoString(__ccgo_ts + 989))

// __attribute__ ((__nonnull__ (1))) int __attribute__((__cdecl__)) __mingw_vwscanf(const wchar_t * __restrict__ Format, va_list argp);
func X__mingw_vwscanf(tls *TLS, _Format uintptr, _argp Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc__mingw_vwscanf.Addr(), _Format, _argp)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__mingw_fwscanf = modcrt.NewProc(GoString(__ccgo_ts + 1005))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __mingw_fwscanf(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,...);
func X__mingw_fwscanf(tls *TLS, __File uintptr, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__mingw_fwprintf = modcrt.NewProc(GoString(__ccgo_ts + 1021))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __mingw_fwprintf(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,...);
func X__mingw_fwprintf(tls *TLS, __File uintptr, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__mingw_wprintf = modcrt.NewProc(GoString(__ccgo_ts + 1038))

// __attribute__ ((__nonnull__ (1))) int __attribute__((__cdecl__)) __mingw_wprintf(const wchar_t * __restrict__ _Format,...);
func X__mingw_wprintf(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__mingw_vwprintf = modcrt.NewProc(GoString(__ccgo_ts + 1054))

// __attribute__ ((__nonnull__ (1))) int __attribute__((__cdecl__)) __mingw_vwprintf(const wchar_t * __restrict__ _Format,va_list _ArgList);
func X__mingw_vwprintf(tls *TLS, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc__mingw_vwprintf.Addr(), __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__mingw_snwprintf = modcrt.NewProc(GoString(__ccgo_ts + 1071))

// __attribute__ ((__nonnull__ (3))) int __attribute__((__cdecl__)) __mingw_snwprintf (wchar_t * __restrict__ s, size_t n, const wchar_t * __restrict__ format, ...);
func X__mingw_snwprintf(tls *TLS, _s uintptr, _n Tsize_t, _format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__mingw_swprintf = modcrt.NewProc(GoString(__ccgo_ts + 1089))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __mingw_swprintf(wchar_t * __restrict__ , const wchar_t * __restrict__ , ...);
func X__mingw_swprintf(tls *TLS, _0 uintptr, _1 uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__mingw_vswprintf = modcrt.NewProc(GoString(__ccgo_ts + 1106))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __mingw_vswprintf(wchar_t * __restrict__ , const wchar_t * __restrict__ ,va_list);
func X__mingw_vswprintf(tls *TLS, _0 uintptr, _1 uintptr, _2 Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc__mingw_vswprintf.Addr(), _0, _1, _2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__ms_swscanf = modcrt.NewProc(GoString(__ccgo_ts + 1124))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __ms_swscanf(const wchar_t * __restrict__ _Src,const wchar_t * __restrict__ _Format,...);
func X__ms_swscanf(tls *TLS, __Src uintptr, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__ms_wscanf = modcrt.NewProc(GoString(__ccgo_ts + 1137))

// __attribute__ ((__nonnull__ (1))) int __attribute__((__cdecl__)) __ms_wscanf(const wchar_t * __restrict__ _Format,...);
func X__ms_wscanf(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__ms_fwscanf = modcrt.NewProc(GoString(__ccgo_ts + 1149))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __ms_fwscanf(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,...);
func X__ms_fwscanf(tls *TLS, __File uintptr, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__ms_fwprintf = modcrt.NewProc(GoString(__ccgo_ts + 1162))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __ms_fwprintf(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,...);
func X__ms_fwprintf(tls *TLS, __File uintptr, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__ms_wprintf = modcrt.NewProc(GoString(__ccgo_ts + 1176))

// __attribute__ ((__nonnull__ (1))) int __attribute__((__cdecl__)) __ms_wprintf(const wchar_t * __restrict__ _Format,...);
func X__ms_wprintf(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__ms_vfwprintf = modcrt.NewProc(GoString(__ccgo_ts + 1189))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __ms_vfwprintf(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,va_list _ArgList);
func X__ms_vfwprintf(tls *TLS, __File uintptr, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc__ms_vfwprintf.Addr(), __File, __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__ms_vwprintf = modcrt.NewProc(GoString(__ccgo_ts + 1204))

// __attribute__ ((__nonnull__ (1))) int __attribute__((__cdecl__)) __ms_vwprintf(const wchar_t * __restrict__ _Format,va_list _ArgList);
func X__ms_vwprintf(tls *TLS, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc__ms_vwprintf.Addr(), __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__ms_swprintf = modcrt.NewProc(GoString(__ccgo_ts + 1218))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __ms_swprintf(wchar_t * __restrict__ , const wchar_t * __restrict__ , ...);
func X__ms_swprintf(tls *TLS, _0 uintptr, _1 uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc__ms_vswprintf = modcrt.NewProc(GoString(__ccgo_ts + 1232))

// __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __ms_vswprintf(wchar_t * __restrict__ , const wchar_t * __restrict__ ,va_list);
func X__ms_vswprintf(tls *TLS, _0 uintptr, _1 uintptr, _2 Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc__ms_vswprintf.Addr(), _0, _1, _2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wfsopen = modcrt.NewProc(GoString(__ccgo_ts + 1247))

// __attribute__ ((__dllimport__)) FILE * __attribute__((__cdecl__)) _wfsopen(const wchar_t *_Filename,const wchar_t *_Mode,int _ShFlag);
func X_wfsopen(tls *TLS, __Filename uintptr, __Mode uintptr, __ShFlag int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wfsopen.Addr(), __Filename, __Mode, uintptr(__ShFlag))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procfgetwc = modcrt.NewProc(GoString(__ccgo_ts + 1256))

// wint_t __attribute__((__cdecl__)) fgetwc(FILE *_File);
func Xfgetwc(tls *TLS, __File uintptr) (r Twint_t) {
	r0, _, err := syscall.SyscallN(procfgetwc.Addr(), __File)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var proc_fgetwchar = modcrt.NewProc(GoString(__ccgo_ts + 1263))

// __attribute__ ((__dllimport__)) wint_t __attribute__((__cdecl__)) _fgetwchar(void);
func X_fgetwchar(tls *TLS) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proc_fgetwchar.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var procfputwc = modcrt.NewProc(GoString(__ccgo_ts + 1274))

// wint_t __attribute__((__cdecl__)) fputwc(wchar_t _Ch,FILE *_File);
func Xfputwc(tls *TLS, __Ch Twchar_t, __File uintptr) (r Twint_t) {
	r0, _, err := syscall.SyscallN(procfputwc.Addr(), uintptr(__Ch), __File)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var proc_fputwchar = modcrt.NewProc(GoString(__ccgo_ts + 1281))

// __attribute__ ((__dllimport__)) wint_t __attribute__((__cdecl__)) _fputwchar(wchar_t _Ch);
func X_fputwchar(tls *TLS, __Ch Twchar_t) (r Twint_t) {
	r0, _, err := syscall.SyscallN(proc_fputwchar.Addr(), uintptr(__Ch))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var procgetwc = modcrt.NewProc(GoString(__ccgo_ts + 1292))

// wint_t __attribute__((__cdecl__)) getwc(FILE *_File);
func Xgetwc(tls *TLS, __File uintptr) (r Twint_t) {
	r0, _, err := syscall.SyscallN(procgetwc.Addr(), __File)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var procgetwchar = modcrt.NewProc(GoString(__ccgo_ts + 1298))

// wint_t __attribute__((__cdecl__)) getwchar(void);
func Xgetwchar(tls *TLS) (r Twint_t) {
	r0, _, err := syscall.SyscallN(procgetwchar.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var procputwc = modcrt.NewProc(GoString(__ccgo_ts + 1307))

// wint_t __attribute__((__cdecl__)) putwc(wchar_t _Ch,FILE *_File);
func Xputwc(tls *TLS, __Ch Twchar_t, __File uintptr) (r Twint_t) {
	r0, _, err := syscall.SyscallN(procputwc.Addr(), uintptr(__Ch), __File)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var procputwchar = modcrt.NewProc(GoString(__ccgo_ts + 1313))

// wint_t __attribute__((__cdecl__)) putwchar(wchar_t _Ch);
func Xputwchar(tls *TLS, __Ch Twchar_t) (r Twint_t) {
	r0, _, err := syscall.SyscallN(procputwchar.Addr(), uintptr(__Ch))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var procungetwc = modcrt.NewProc(GoString(__ccgo_ts + 1322))

// wint_t __attribute__((__cdecl__)) ungetwc(wint_t _Ch,FILE *_File);
func Xungetwc(tls *TLS, __Ch Twint_t, __File uintptr) (r Twint_t) {
	r0, _, err := syscall.SyscallN(procungetwc.Addr(), uintptr(__Ch), __File)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var procfgetws = modcrt.NewProc(GoString(__ccgo_ts + 1330))

// wchar_t * __attribute__((__cdecl__)) fgetws(wchar_t * __restrict__ _Dst,int _SizeInWords,FILE * __restrict__ _File);
func Xfgetws(tls *TLS, __Dst uintptr, __SizeInWords int32, __File uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procfgetws.Addr(), __Dst, uintptr(__SizeInWords), __File)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procfputws = modcrt.NewProc(GoString(__ccgo_ts + 1337))

// int __attribute__((__cdecl__)) fputws(const wchar_t * __restrict__ _Str,FILE * __restrict__ _File);
func Xfputws(tls *TLS, __Str uintptr, __File uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(procfputws.Addr(), __Str, __File)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_getws = modcrt.NewProc(GoString(__ccgo_ts + 1344))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _getws(wchar_t *_String);
func X_getws(tls *TLS, __String uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_getws.Addr(), __String)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_putws = modcrt.NewProc(GoString(__ccgo_ts + 1351))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _putws(const wchar_t *_Str);
func X_putws(tls *TLS, __Str uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_putws.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_scwprintf = modcrt.NewProc(GoString(__ccgo_ts + 1358))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _scwprintf(const wchar_t * __restrict__ _Format,...);
func X_scwprintf(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_swprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1369))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _swprintf_l(wchar_t * __restrict__ ,size_t _SizeInWords,const wchar_t * __restrict__ _Format,_locale_t _Locale,... );
func X_swprintf_l(tls *TLS, _0 uintptr, __SizeInWords Tsize_t, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_swprintf_c = modcrt.NewProc(GoString(__ccgo_ts + 1381))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _swprintf_c(wchar_t * __restrict__ _DstBuf,size_t _SizeInWords,const wchar_t * __restrict__ _Format,...);
func X_swprintf_c(tls *TLS, __DstBuf uintptr, __SizeInWords Tsize_t, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vswprintf_c = modcrt.NewProc(GoString(__ccgo_ts + 1393))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vswprintf_c(wchar_t * __restrict__ _DstBuf,size_t _SizeInWords,const wchar_t * __restrict__ _Format,va_list _ArgList);
func X_vswprintf_c(tls *TLS, __DstBuf uintptr, __SizeInWords Tsize_t, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vswprintf_c.Addr(), __DstBuf, uintptr(__SizeInWords), __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_fwprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 1406))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _fwprintf_p(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,...);
func X_fwprintf_p(tls *TLS, __File uintptr, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_wprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 1418))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wprintf_p(const wchar_t * __restrict__ _Format,...);
func X_wprintf_p(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vfwprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 1429))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vfwprintf_p(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,va_list _ArgList);
func X_vfwprintf_p(tls *TLS, __File uintptr, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vfwprintf_p.Addr(), __File, __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_vwprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 1442))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vwprintf_p(const wchar_t * __restrict__ _Format,va_list _ArgList);
func X_vwprintf_p(tls *TLS, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vwprintf_p.Addr(), __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_swprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 1454))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _swprintf_p(wchar_t * __restrict__ _DstBuf,size_t _MaxCount,const wchar_t * __restrict__ _Format,...);
func X_swprintf_p(tls *TLS, __DstBuf uintptr, __MaxCount Tsize_t, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vswprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 1466))

// __attribute__((dllimport)) int __attribute__((__cdecl__)) _vswprintf_p(wchar_t * __restrict__ _DstBuf,size_t _MaxCount,const wchar_t * __restrict__ _Format,va_list _ArgList);
func X_vswprintf_p(tls *TLS, __DstBuf uintptr, __MaxCount Tsize_t, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vswprintf_p.Addr(), __DstBuf, uintptr(__MaxCount), __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_scwprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 1479))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _scwprintf_p(const wchar_t * __restrict__ _Format,...);
func X_scwprintf_p(tls *TLS, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vscwprintf_p = modcrt.NewProc(GoString(__ccgo_ts + 1492))

// __attribute__((dllimport)) int __attribute__((__cdecl__)) _vscwprintf_p(const wchar_t * __restrict__ _Format,va_list _ArgList);
func X_vscwprintf_p(tls *TLS, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vscwprintf_p.Addr(), __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1506))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wprintf_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_wprintf_l(tls *TLS, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_wprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 1517))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wprintf_p_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_wprintf_p_l(tls *TLS, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vwprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1530))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vwprintf_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vwprintf_l(tls *TLS, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vwprintf_l.Addr(), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_vwprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 1542))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vwprintf_p_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vwprintf_p_l(tls *TLS, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vwprintf_p_l.Addr(), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_fwprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1556))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _fwprintf_l(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_fwprintf_l(tls *TLS, __File uintptr, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_fwprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 1568))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _fwprintf_p_l(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_fwprintf_p_l(tls *TLS, __File uintptr, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vfwprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1582))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vfwprintf_l(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vfwprintf_l(tls *TLS, __File uintptr, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vfwprintf_l.Addr(), __File, __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_vfwprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 1595))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vfwprintf_p_l(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vfwprintf_p_l(tls *TLS, __File uintptr, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vfwprintf_p_l.Addr(), __File, __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_swprintf_c_l = modcrt.NewProc(GoString(__ccgo_ts + 1610))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _swprintf_c_l(wchar_t * __restrict__ _DstBuf,size_t _MaxCount,const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_swprintf_c_l(tls *TLS, __DstBuf uintptr, __MaxCount Tsize_t, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_swprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 1624))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _swprintf_p_l(wchar_t * __restrict__ _DstBuf,size_t _MaxCount,const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_swprintf_p_l(tls *TLS, __DstBuf uintptr, __MaxCount Tsize_t, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vswprintf_c_l = modcrt.NewProc(GoString(__ccgo_ts + 1638))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vswprintf_c_l(wchar_t * __restrict__ _DstBuf,size_t _MaxCount,const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vswprintf_c_l(tls *TLS, __DstBuf uintptr, __MaxCount Tsize_t, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vswprintf_c_l.Addr(), __DstBuf, uintptr(__MaxCount), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_vswprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 1653))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vswprintf_p_l(wchar_t * __restrict__ _DstBuf,size_t _MaxCount,const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vswprintf_p_l(tls *TLS, __DstBuf uintptr, __MaxCount Tsize_t, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vswprintf_p_l.Addr(), __DstBuf, uintptr(__MaxCount), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_scwprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1668))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _scwprintf_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_scwprintf_l(tls *TLS, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_scwprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 1681))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _scwprintf_p_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_scwprintf_p_l(tls *TLS, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vscwprintf_p_l = modcrt.NewProc(GoString(__ccgo_ts + 1696))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vscwprintf_p_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vscwprintf_p_l(tls *TLS, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vscwprintf_p_l.Addr(), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_snwprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1712))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _snwprintf_l(wchar_t * __restrict__ _DstBuf,size_t _MaxCount,const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_snwprintf_l(tls *TLS, __DstBuf uintptr, __MaxCount Tsize_t, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vsnwprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1725))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vsnwprintf_l(wchar_t * __restrict__ _DstBuf,size_t _MaxCount,const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vsnwprintf_l(tls *TLS, __DstBuf uintptr, __MaxCount Tsize_t, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vsnwprintf_l.Addr(), __DstBuf, uintptr(__MaxCount), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_swprintf = modcrt.NewProc(GoString(__ccgo_ts + 1739))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _swprintf(wchar_t * __restrict__ _Dest,const wchar_t * __restrict__ _Format,...);
func X_swprintf(tls *TLS, __Dest uintptr, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vswprintf = modcrt.NewProc(GoString(__ccgo_ts + 1749))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vswprintf(wchar_t * __restrict__ _Dest,const wchar_t * __restrict__ _Format,va_list _Args);
func X_vswprintf(tls *TLS, __Dest uintptr, __Format uintptr, __Args Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vswprintf.Addr(), __Dest, __Format, __Args)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__swprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1760))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) __swprintf_l(wchar_t * __restrict__ _Dest,const wchar_t * __restrict__ _Format,_locale_t _Plocinfo,...);
func X__swprintf_l(tls *TLS, __Dest uintptr, __Format uintptr, __Plocinfo T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_vswprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1773))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vswprintf_l(wchar_t * __restrict__ _Dest,size_t _MaxCount,const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vswprintf_l(tls *TLS, __Dest uintptr, __MaxCount Tsize_t, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vswprintf_l.Addr(), __Dest, uintptr(__MaxCount), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__vswprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1786))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) __vswprintf_l(wchar_t * __restrict__ _Dest,const wchar_t * __restrict__ _Format,_locale_t _Plocinfo,va_list _Args);
func X__vswprintf_l(tls *TLS, __Dest uintptr, __Format uintptr, __Plocinfo T_locale_t, __Args Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc__vswprintf_l.Addr(), __Dest, __Format, __Plocinfo, __Args)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wtempnam = modcrt.NewProc(GoString(__ccgo_ts + 1800))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wtempnam(const wchar_t *_Directory,const wchar_t *_FilePrefix);
func X_wtempnam(tls *TLS, __Directory uintptr, __FilePrefix uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wtempnam.Addr(), __Directory, __FilePrefix)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_vscwprintf = modcrt.NewProc(GoString(__ccgo_ts + 1810))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vscwprintf(const wchar_t * __restrict__ _Format,va_list _ArgList);
func X_vscwprintf(tls *TLS, __Format uintptr, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vscwprintf.Addr(), __Format, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_vscwprintf_l = modcrt.NewProc(GoString(__ccgo_ts + 1822))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _vscwprintf_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,va_list _ArgList);
func X_vscwprintf_l(tls *TLS, __Format uintptr, __Locale T_locale_t, __ArgList Tva_list) (r int32) {
	r0, _, err := syscall.SyscallN(proc_vscwprintf_l.Addr(), __Format, __Locale, __ArgList)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_fwscanf_l = modcrt.NewProc(GoString(__ccgo_ts + 1836))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _fwscanf_l(FILE * __restrict__ _File,const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_fwscanf_l(tls *TLS, __File uintptr, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_swscanf_l = modcrt.NewProc(GoString(__ccgo_ts + 1847))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _swscanf_l(const wchar_t * __restrict__ _Src,const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_swscanf_l(tls *TLS, __Src uintptr, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_snwscanf = modcrt.NewProc(GoString(__ccgo_ts + 1858))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _snwscanf(const wchar_t * __restrict__ _Src,size_t _MaxCount,const wchar_t * __restrict__ _Format,...);
func X_snwscanf(tls *TLS, __Src uintptr, __MaxCount Tsize_t, __Format uintptr, va_list uintptr) (r int32) {
	panic(651)
}

var proc_snwscanf_l = modcrt.NewProc(GoString(__ccgo_ts + 1868))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _snwscanf_l(const wchar_t * __restrict__ _Src,size_t _MaxCount,const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_snwscanf_l(tls *TLS, __Src uintptr, __MaxCount Tsize_t, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_wscanf_l = modcrt.NewProc(GoString(__ccgo_ts + 1880))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wscanf_l(const wchar_t * __restrict__ _Format,_locale_t _Locale,...);
func X_wscanf_l(tls *TLS, __Format uintptr, __Locale T_locale_t, va_list uintptr) (r int32) {
	panic(651)
}

var proc_wfdopen = modcrt.NewProc(GoString(__ccgo_ts + 1890))

// __attribute__ ((__dllimport__)) FILE * __attribute__((__cdecl__)) _wfdopen(int _FileHandle ,const wchar_t *_Mode);
func X_wfdopen(tls *TLS, __FileHandle int32, __Mode uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wfdopen.Addr(), uintptr(__FileHandle), __Mode)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wfopen = modcrt.NewProc(GoString(__ccgo_ts + 1899))

// __attribute__ ((__dllimport__)) FILE * __attribute__((__cdecl__)) _wfopen(const wchar_t * __restrict__ _Filename,const wchar_t * __restrict__ _Mode);
func X_wfopen(tls *TLS, __Filename uintptr, __Mode uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wfopen.Addr(), __Filename, __Mode)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wfreopen = modcrt.NewProc(GoString(__ccgo_ts + 1907))

// __attribute__ ((__dllimport__)) FILE * __attribute__((__cdecl__)) _wfreopen(const wchar_t * __restrict__ _Filename,const wchar_t * __restrict__ _Mode,FILE * __restrict__ _OldFile);
func X_wfreopen(tls *TLS, __Filename uintptr, __Mode uintptr, __OldFile uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wfreopen.Addr(), __Filename, __Mode, __OldFile)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wperror = modcrt.NewProc(GoString(__ccgo_ts + 1917))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _wperror(const wchar_t *_ErrMsg);
func X_wperror(tls *TLS, __ErrMsg uintptr) {
	_, _, err := syscall.SyscallN(proc_wperror.Addr(), __ErrMsg)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_wpopen = modcrt.NewProc(GoString(__ccgo_ts + 1926))

// __attribute__ ((__dllimport__)) FILE * __attribute__((__cdecl__)) _wpopen(const wchar_t *_Command,const wchar_t *_Mode);
func X_wpopen(tls *TLS, __Command uintptr, __Mode uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wpopen.Addr(), __Command, __Mode)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wremove = modcrt.NewProc(GoString(__ccgo_ts + 1934))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wremove(const wchar_t *_Filename);
func X_wremove(tls *TLS, __Filename uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wremove.Addr(), __Filename)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wtmpnam = modcrt.NewProc(GoString(__ccgo_ts + 1943))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wtmpnam(wchar_t *_Buffer);
func X_wtmpnam(tls *TLS, __Buffer uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wtmpnam.Addr(), __Buffer)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_itow = modcrt.NewProc(GoString(__ccgo_ts + 1952))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _itow(int _Value,wchar_t *_Dest,int _Radix);
func X_itow(tls *TLS, __Value int32, __Dest uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_itow.Addr(), uintptr(__Value), __Dest, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_ltow = modcrt.NewProc(GoString(__ccgo_ts + 1958))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _ltow(long _Value,wchar_t *_Dest,int _Radix);
func X_ltow(tls *TLS, __Value int32, __Dest uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_ltow.Addr(), uintptr(__Value), __Dest, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_ultow = modcrt.NewProc(GoString(__ccgo_ts + 1964))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _ultow(unsigned long _Value,wchar_t *_Dest,int _Radix);
func X_ultow(tls *TLS, __Value uint32, __Dest uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_ultow.Addr(), uintptr(__Value), __Dest, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcstod_l = modcrt.NewProc(GoString(__ccgo_ts + 1971))

// __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _wcstod_l(const wchar_t * __restrict__ _Str,wchar_t ** __restrict__ _EndPtr,_locale_t _Locale);
func X_wcstod_l(tls *TLS, __Str uintptr, __EndPtr uintptr, __Locale T_locale_t) (r float64) {
	r0, _, err := syscall.SyscallN(proc_wcstod_l.Addr(), __Str, __EndPtr, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc__mingw_wcstod = modcrt.NewProc(GoString(__ccgo_ts + 1981))

// double __attribute__((__cdecl__)) __mingw_wcstod(const wchar_t * __restrict__ _Str,wchar_t ** __restrict__ _EndPtr);
func X__mingw_wcstod(tls *TLS, __Str uintptr, __EndPtr uintptr) (r float64) {
	r0, _, err := syscall.SyscallN(proc__mingw_wcstod.Addr(), __Str, __EndPtr)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc__mingw_wcstof = modcrt.NewProc(GoString(__ccgo_ts + 1996))

// float __attribute__((__cdecl__)) __mingw_wcstof(const wchar_t * __restrict__ nptr, wchar_t ** __restrict__ endptr);
func X__mingw_wcstof(tls *TLS, _nptr uintptr, _endptr uintptr) (r float32) {
	r0, _, err := syscall.SyscallN(proc__mingw_wcstof.Addr(), _nptr, _endptr)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float32(r0)
}

var proc__mingw_wcstold = modcrt.NewProc(GoString(__ccgo_ts + 2011))

// long double __attribute__((__cdecl__)) __mingw_wcstold(const wchar_t * __restrict__, wchar_t ** __restrict__);
func X__mingw_wcstold(tls *TLS, _0 uintptr, _1 uintptr) (r float64) {
	r0, _, err := syscall.SyscallN(proc__mingw_wcstold.Addr(), _0, _1)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var procwcstold = modcrt.NewProc(GoString(__ccgo_ts + 2027))

// long double __attribute__((__cdecl__)) wcstold (const wchar_t * __restrict__, wchar_t ** __restrict__);
func Xwcstold(tls *TLS, _0 uintptr, _1 uintptr) (r float64) {
	r0, _, err := syscall.SyscallN(procwcstold.Addr(), _0, _1)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var procwcstol = modcrt.NewProc(GoString(__ccgo_ts + 2035))

// long __attribute__((__cdecl__)) wcstol(const wchar_t * __restrict__ _Str,wchar_t ** __restrict__ _EndPtr,int _Radix);
func Xwcstol(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32) (r int32) {
	r0, _, err := syscall.SyscallN(procwcstol.Addr(), __Str, __EndPtr, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcstol_l = modcrt.NewProc(GoString(__ccgo_ts + 2042))

// __attribute__ ((__dllimport__)) long __attribute__((__cdecl__)) _wcstol_l(const wchar_t * __restrict__ _Str,wchar_t ** __restrict__ _EndPtr,int _Radix,_locale_t _Locale);
func X_wcstol_l(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcstol_l.Addr(), __Str, __EndPtr, uintptr(__Radix), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procwcstoul = modcrt.NewProc(GoString(__ccgo_ts + 2052))

// unsigned long __attribute__((__cdecl__)) wcstoul(const wchar_t * __restrict__ _Str,wchar_t ** __restrict__ _EndPtr,int _Radix);
func Xwcstoul(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32) (r uint32) {
	r0, _, err := syscall.SyscallN(procwcstoul.Addr(), __Str, __EndPtr, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint32(r0)
}

var proc_wcstoul_l = modcrt.NewProc(GoString(__ccgo_ts + 2060))

// __attribute__ ((__dllimport__)) unsigned long __attribute__((__cdecl__)) _wcstoul_l(const wchar_t * __restrict__ _Str,wchar_t ** __restrict__ _EndPtr,int _Radix,_locale_t _Locale);
func X_wcstoul_l(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32, __Locale T_locale_t) (r uint32) {
	r0, _, err := syscall.SyscallN(proc_wcstoul_l.Addr(), __Str, __EndPtr, uintptr(__Radix), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint32(r0)
}

var proc_wtof = modcrt.NewProc(GoString(__ccgo_ts + 2071))

// __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _wtof(const wchar_t *_Str);
func X_wtof(tls *TLS, __Str uintptr) (r float64) {
	r0, _, err := syscall.SyscallN(proc_wtof.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc_wtof_l = modcrt.NewProc(GoString(__ccgo_ts + 2077))

// __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _wtof_l(const wchar_t *_Str,_locale_t _Locale);
func X_wtof_l(tls *TLS, __Str uintptr, __Locale T_locale_t) (r float64) {
	r0, _, err := syscall.SyscallN(proc_wtof_l.Addr(), __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc_wtoi_l = modcrt.NewProc(GoString(__ccgo_ts + 2085))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wtoi_l(const wchar_t *_Str,_locale_t _Locale);
func X_wtoi_l(tls *TLS, __Str uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wtoi_l.Addr(), __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wtol = modcrt.NewProc(GoString(__ccgo_ts + 2093))

// __attribute__ ((__dllimport__)) long __attribute__((__cdecl__)) _wtol(const wchar_t *_Str);
func X_wtol(tls *TLS, __Str uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wtol.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wtol_l = modcrt.NewProc(GoString(__ccgo_ts + 2099))

// __attribute__ ((__dllimport__)) long __attribute__((__cdecl__)) _wtol_l(const wchar_t *_Str,_locale_t _Locale);
func X_wtol_l(tls *TLS, __Str uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wtol_l.Addr(), __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_i64tow = modcrt.NewProc(GoString(__ccgo_ts + 2107))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _i64tow( long long _Val,wchar_t *_DstBuf,int _Radix);
func X_i64tow(tls *TLS, __Val int64, __DstBuf uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_i64tow.Addr(), uintptr(__Val), __DstBuf, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_ui64tow = modcrt.NewProc(GoString(__ccgo_ts + 2115))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _ui64tow(unsigned long long _Val,wchar_t *_DstBuf,int _Radix);
func X_ui64tow(tls *TLS, __Val uint64, __DstBuf uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_ui64tow.Addr(), uintptr(__Val), __DstBuf, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wtoi64 = modcrt.NewProc(GoString(__ccgo_ts + 2124))

// __attribute__ ((__dllimport__)) long long __attribute__((__cdecl__)) _wtoi64(const wchar_t *_Str);
func X_wtoi64(tls *TLS, __Str uintptr) (r int64) {
	r0, _, err := syscall.SyscallN(proc_wtoi64.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proc_wtoi64_l = modcrt.NewProc(GoString(__ccgo_ts + 2132))

// __attribute__ ((__dllimport__)) long long __attribute__((__cdecl__)) _wtoi64_l(const wchar_t *_Str,_locale_t _Locale);
func X_wtoi64_l(tls *TLS, __Str uintptr, __Locale T_locale_t) (r int64) {
	r0, _, err := syscall.SyscallN(proc_wtoi64_l.Addr(), __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proc_wcstoi64 = modcrt.NewProc(GoString(__ccgo_ts + 2142))

// __attribute__ ((__dllimport__)) long long __attribute__((__cdecl__)) _wcstoi64(const wchar_t *_Str,wchar_t **_EndPtr,int _Radix);
func X_wcstoi64(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32) (r int64) {
	r0, _, err := syscall.SyscallN(proc_wcstoi64.Addr(), __Str, __EndPtr, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proc_wcstoi64_l = modcrt.NewProc(GoString(__ccgo_ts + 2152))

// __attribute__ ((__dllimport__)) long long __attribute__((__cdecl__)) _wcstoi64_l(const wchar_t *_Str,wchar_t **_EndPtr,int _Radix,_locale_t _Locale);
func X_wcstoi64_l(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32, __Locale T_locale_t) (r int64) {
	r0, _, err := syscall.SyscallN(proc_wcstoi64_l.Addr(), __Str, __EndPtr, uintptr(__Radix), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proc_wcstoui64 = modcrt.NewProc(GoString(__ccgo_ts + 2164))

// __attribute__ ((__dllimport__)) unsigned long long __attribute__((__cdecl__)) _wcstoui64(const wchar_t *_Str,wchar_t **_EndPtr,int _Radix);
func X_wcstoui64(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32) (r uint64) {
	r0, _, err := syscall.SyscallN(proc_wcstoui64.Addr(), __Str, __EndPtr, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint64(r0)
}

var proc_wcstoui64_l = modcrt.NewProc(GoString(__ccgo_ts + 2175))

// __attribute__ ((__dllimport__)) unsigned long long __attribute__((__cdecl__)) _wcstoui64_l(const wchar_t *_Str,wchar_t **_EndPtr,int _Radix,_locale_t _Locale);
func X_wcstoui64_l(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32, __Locale T_locale_t) (r uint64) {
	r0, _, err := syscall.SyscallN(proc_wcstoui64_l.Addr(), __Str, __EndPtr, uintptr(__Radix), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint64(r0)
}

var proc_wfullpath = modcrt.NewProc(GoString(__ccgo_ts + 2188))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wfullpath(wchar_t *_FullPath,const wchar_t *_Path,size_t _SizeInWords);
func X_wfullpath(tls *TLS, __FullPath uintptr, __Path uintptr, __SizeInWords Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wfullpath.Addr(), __FullPath, __Path, uintptr(__SizeInWords))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wmakepath = modcrt.NewProc(GoString(__ccgo_ts + 2199))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _wmakepath(wchar_t *_ResultPath,const wchar_t *_Drive,const wchar_t *_Dir,const wchar_t *_Filename,const wchar_t *_Ext);
func X_wmakepath(tls *TLS, __ResultPath uintptr, __Drive uintptr, __Dir uintptr, __Filename uintptr, __Ext uintptr) {
	_, _, err := syscall.SyscallN(proc_wmakepath.Addr(), __ResultPath, __Drive, __Dir, __Filename, __Ext)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_wsearchenv = modcrt.NewProc(GoString(__ccgo_ts + 2210))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _wsearchenv(const wchar_t *_Filename,const wchar_t *_EnvVar,wchar_t *_ResultPath);
func X_wsearchenv(tls *TLS, __Filename uintptr, __EnvVar uintptr, __ResultPath uintptr) {
	_, _, err := syscall.SyscallN(proc_wsearchenv.Addr(), __Filename, __EnvVar, __ResultPath)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_wsplitpath = modcrt.NewProc(GoString(__ccgo_ts + 2222))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _wsplitpath(const wchar_t *_FullPath,wchar_t *_Drive,wchar_t *_Dir,wchar_t *_Filename,wchar_t *_Ext);
func X_wsplitpath(tls *TLS, __FullPath uintptr, __Drive uintptr, __Dir uintptr, __Filename uintptr, __Ext uintptr) {
	_, _, err := syscall.SyscallN(proc_wsplitpath.Addr(), __FullPath, __Drive, __Dir, __Filename, __Ext)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_wcsdup = modcrt.NewProc(GoString(__ccgo_ts + 2234))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wcsdup(const wchar_t *_Str);
func X_wcsdup(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcsdup.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcscat = modcrt.NewProc(GoString(__ccgo_ts + 2242))

// wchar_t * __attribute__((__cdecl__)) wcscat(wchar_t * __restrict__ _Dest,const wchar_t * __restrict__ _Source);
func Xwcscat(tls *TLS, __Dest uintptr, __Source uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcscat.Addr(), __Dest, __Source)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcscspn = modcrt.NewProc(GoString(__ccgo_ts + 2249))

// size_t __attribute__((__cdecl__)) wcscspn(const wchar_t *_Str,const wchar_t *_Control);
func Xwcscspn(tls *TLS, __Str uintptr, __Control uintptr) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procwcscspn.Addr(), __Str, __Control)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var procwcsnlen = modcrt.NewProc(GoString(__ccgo_ts + 2257))

// size_t __attribute__((__cdecl__)) wcsnlen(const wchar_t *_Src,size_t _MaxCount);
func Xwcsnlen(tls *TLS, __Src uintptr, __MaxCount Tsize_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procwcsnlen.Addr(), __Src, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var procwcsncat = modcrt.NewProc(GoString(__ccgo_ts + 2265))

// wchar_t * __attribute__((__cdecl__)) wcsncat(wchar_t * __restrict__ _Dest,const wchar_t * __restrict__ _Source,size_t _Count);
func Xwcsncat(tls *TLS, __Dest uintptr, __Source uintptr, __Count Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcsncat.Addr(), __Dest, __Source, uintptr(__Count))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcsncpy = modcrt.NewProc(GoString(__ccgo_ts + 2273))

// wchar_t * __attribute__((__cdecl__)) wcsncpy(wchar_t * __restrict__ _Dest,const wchar_t * __restrict__ _Source,size_t _Count);
func Xwcsncpy(tls *TLS, __Dest uintptr, __Source uintptr, __Count Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcsncpy.Addr(), __Dest, __Source, uintptr(__Count))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcsncpy_l = modcrt.NewProc(GoString(__ccgo_ts + 2281))

// wchar_t * __attribute__((__cdecl__)) _wcsncpy_l(wchar_t * __restrict__ _Dest,const wchar_t * __restrict__ _Source,size_t _Count,_locale_t _Locale);
func X_wcsncpy_l(tls *TLS, __Dest uintptr, __Source uintptr, __Count Tsize_t, __Locale T_locale_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcsncpy_l.Addr(), __Dest, __Source, uintptr(__Count), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcspbrk = modcrt.NewProc(GoString(__ccgo_ts + 2292))

// wchar_t * __attribute__((__cdecl__)) wcspbrk(const wchar_t *_Str,const wchar_t *_Control);
func Xwcspbrk(tls *TLS, __Str uintptr, __Control uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcspbrk.Addr(), __Str, __Control)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcsrchr = modcrt.NewProc(GoString(__ccgo_ts + 2300))

// wchar_t * __attribute__((__cdecl__)) wcsrchr(const wchar_t *_Str,wchar_t _Ch);
func Xwcsrchr(tls *TLS, __Str uintptr, __Ch Twchar_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcsrchr.Addr(), __Str, uintptr(__Ch))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcsspn = modcrt.NewProc(GoString(__ccgo_ts + 2308))

// size_t __attribute__((__cdecl__)) wcsspn(const wchar_t *_Str,const wchar_t *_Control);
func Xwcsspn(tls *TLS, __Str uintptr, __Control uintptr) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procwcsspn.Addr(), __Str, __Control)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var procwcsstr = modcrt.NewProc(GoString(__ccgo_ts + 2315))

// wchar_t * __attribute__((__cdecl__)) wcsstr(const wchar_t *_Str,const wchar_t *_SubStr);
func Xwcsstr(tls *TLS, __Str uintptr, __SubStr uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcsstr.Addr(), __Str, __SubStr)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcstok = modcrt.NewProc(GoString(__ccgo_ts + 2322))

// wchar_t * __attribute__((__cdecl__)) wcstok(wchar_t * __restrict__ _Str,const wchar_t * __restrict__ _Delim);
func Xwcstok(tls *TLS, __Str uintptr, __Delim uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcstok.Addr(), __Str, __Delim)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcserror = modcrt.NewProc(GoString(__ccgo_ts + 2329))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wcserror(int _ErrNum);
func X_wcserror(tls *TLS, __ErrNum int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcserror.Addr(), uintptr(__ErrNum))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc__wcserror = modcrt.NewProc(GoString(__ccgo_ts + 2339))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) __wcserror(const wchar_t *_Str);
func X__wcserror(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc__wcserror.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcsicmp_l = modcrt.NewProc(GoString(__ccgo_ts + 2350))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcsicmp_l(const wchar_t *_Str1,const wchar_t *_Str2,_locale_t _Locale);
func X_wcsicmp_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcsicmp_l.Addr(), __Str1, __Str2, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcsnicmp_l = modcrt.NewProc(GoString(__ccgo_ts + 2361))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcsnicmp_l(const wchar_t *_Str1,const wchar_t *_Str2,size_t _MaxCount,_locale_t _Locale);
func X_wcsnicmp_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcsnicmp_l.Addr(), __Str1, __Str2, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcsnset = modcrt.NewProc(GoString(__ccgo_ts + 2373))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wcsnset(wchar_t *_Str,wchar_t _Val,size_t _MaxCount);
func X_wcsnset(tls *TLS, __Str uintptr, __Val Twchar_t, __MaxCount Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcsnset.Addr(), __Str, uintptr(__Val), uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcsrev = modcrt.NewProc(GoString(__ccgo_ts + 2382))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wcsrev(wchar_t *_Str);
func X_wcsrev(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcsrev.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcsset = modcrt.NewProc(GoString(__ccgo_ts + 2390))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wcsset(wchar_t *_Str,wchar_t _Val);
func X_wcsset(tls *TLS, __Str uintptr, __Val Twchar_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcsset.Addr(), __Str, uintptr(__Val))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcslwr = modcrt.NewProc(GoString(__ccgo_ts + 2398))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wcslwr(wchar_t *_String);
func X_wcslwr(tls *TLS, __String uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcslwr.Addr(), __String)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcslwr_l = modcrt.NewProc(GoString(__ccgo_ts + 2406))

// __attribute__ ((__dllimport__)) wchar_t *_wcslwr_l(wchar_t *_String,_locale_t _Locale);
func X_wcslwr_l(tls *TLS, __String uintptr, __Locale T_locale_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcslwr_l.Addr(), __String, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcsupr = modcrt.NewProc(GoString(__ccgo_ts + 2416))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wcsupr(wchar_t *_String);
func X_wcsupr(tls *TLS, __String uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcsupr.Addr(), __String)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wcsupr_l = modcrt.NewProc(GoString(__ccgo_ts + 2424))

// __attribute__ ((__dllimport__)) wchar_t *_wcsupr_l(wchar_t *_String,_locale_t _Locale);
func X_wcsupr_l(tls *TLS, __String uintptr, __Locale T_locale_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wcsupr_l.Addr(), __String, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcsxfrm = modcrt.NewProc(GoString(__ccgo_ts + 2434))

// size_t __attribute__((__cdecl__)) wcsxfrm(wchar_t * __restrict__ _Dst,const wchar_t * __restrict__ _Src,size_t _MaxCount);
func Xwcsxfrm(tls *TLS, __Dst uintptr, __Src uintptr, __MaxCount Tsize_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procwcsxfrm.Addr(), __Dst, __Src, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_wcsxfrm_l = modcrt.NewProc(GoString(__ccgo_ts + 2442))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _wcsxfrm_l(wchar_t * __restrict__ _Dst,const wchar_t * __restrict__ _Src,size_t _MaxCount,_locale_t _Locale);
func X_wcsxfrm_l(tls *TLS, __Dst uintptr, __Src uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_wcsxfrm_l.Addr(), __Dst, __Src, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var procwcscoll = modcrt.NewProc(GoString(__ccgo_ts + 2453))

// int __attribute__((__cdecl__)) wcscoll(const wchar_t *_Str1,const wchar_t *_Str2);
func Xwcscoll(tls *TLS, __Str1 uintptr, __Str2 uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(procwcscoll.Addr(), __Str1, __Str2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcscoll_l = modcrt.NewProc(GoString(__ccgo_ts + 2461))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcscoll_l(const wchar_t *_Str1,const wchar_t *_Str2,_locale_t _Locale);
func X_wcscoll_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcscoll_l.Addr(), __Str1, __Str2, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcsicoll = modcrt.NewProc(GoString(__ccgo_ts + 2472))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcsicoll(const wchar_t *_Str1,const wchar_t *_Str2);
func X_wcsicoll(tls *TLS, __Str1 uintptr, __Str2 uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcsicoll.Addr(), __Str1, __Str2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcsicoll_l = modcrt.NewProc(GoString(__ccgo_ts + 2482))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcsicoll_l(const wchar_t *_Str1,const wchar_t *_Str2,_locale_t _Locale);
func X_wcsicoll_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcsicoll_l.Addr(), __Str1, __Str2, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcsncoll = modcrt.NewProc(GoString(__ccgo_ts + 2494))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcsncoll(const wchar_t *_Str1,const wchar_t *_Str2,size_t _MaxCount);
func X_wcsncoll(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcsncoll.Addr(), __Str1, __Str2, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcsncoll_l = modcrt.NewProc(GoString(__ccgo_ts + 2504))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcsncoll_l(const wchar_t *_Str1,const wchar_t *_Str2,size_t _MaxCount,_locale_t _Locale);
func X_wcsncoll_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcsncoll_l.Addr(), __Str1, __Str2, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcsnicoll = modcrt.NewProc(GoString(__ccgo_ts + 2516))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcsnicoll(const wchar_t *_Str1,const wchar_t *_Str2,size_t _MaxCount);
func X_wcsnicoll(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcsnicoll.Addr(), __Str1, __Str2, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcsnicoll_l = modcrt.NewProc(GoString(__ccgo_ts + 2527))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wcsnicoll_l(const wchar_t *_Str1,const wchar_t *_Str2,size_t _MaxCount,_locale_t _Locale);
func X_wcsnicoll_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wcsnicoll_l.Addr(), __Str1, __Str2, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procwcsdup = modcrt.NewProc(GoString(__ccgo_ts + 2540))

// wchar_t * __attribute__((__cdecl__)) wcsdup(const wchar_t *_Str);
func Xwcsdup(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcsdup.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcsnicmp = modcrt.NewProc(GoString(__ccgo_ts + 2547))

// int __attribute__((__cdecl__)) wcsnicmp(const wchar_t *_Str1,const wchar_t *_Str2,size_t _MaxCount);
func Xwcsnicmp(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(procwcsnicmp.Addr(), __Str1, __Str2, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procwcsnset = modcrt.NewProc(GoString(__ccgo_ts + 2556))

// wchar_t * __attribute__((__cdecl__)) wcsnset(wchar_t *_Str,wchar_t _Val,size_t _MaxCount);
func Xwcsnset(tls *TLS, __Str uintptr, __Val Twchar_t, __MaxCount Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcsnset.Addr(), __Str, uintptr(__Val), uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcsrev = modcrt.NewProc(GoString(__ccgo_ts + 2564))

// wchar_t * __attribute__((__cdecl__)) wcsrev(wchar_t *_Str);
func Xwcsrev(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcsrev.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcsset = modcrt.NewProc(GoString(__ccgo_ts + 2571))

// wchar_t * __attribute__((__cdecl__)) wcsset(wchar_t *_Str,wchar_t _Val);
func Xwcsset(tls *TLS, __Str uintptr, __Val Twchar_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcsset.Addr(), __Str, uintptr(__Val))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcslwr = modcrt.NewProc(GoString(__ccgo_ts + 2578))

// wchar_t * __attribute__((__cdecl__)) wcslwr(wchar_t *_Str);
func Xwcslwr(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcslwr.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcsupr = modcrt.NewProc(GoString(__ccgo_ts + 2585))

// wchar_t * __attribute__((__cdecl__)) wcsupr(wchar_t *_Str);
func Xwcsupr(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwcsupr.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwcsicoll = modcrt.NewProc(GoString(__ccgo_ts + 2592))

// int __attribute__((__cdecl__)) wcsicoll(const wchar_t *_Str1,const wchar_t *_Str2);
func Xwcsicoll(tls *TLS, __Str1 uintptr, __Str2 uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(procwcsicoll.Addr(), __Str1, __Str2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

type Ttm = struct {
	Ftm_sec   int32
	Ftm_min   int32
	Ftm_hour  int32
	Ftm_mday  int32
	Ftm_mon   int32
	Ftm_year  int32
	Ftm_wday  int32
	Ftm_yday  int32
	Ftm_isdst int32
}

var proc_wasctime = modcrt.NewProc(GoString(__ccgo_ts + 2601))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wasctime(const struct tm *_Tm);
func X_wasctime(tls *TLS, __Tm uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wasctime.Addr(), __Tm)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wasctime_s = modcrt.NewProc(GoString(__ccgo_ts + 2611))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _wasctime_s (wchar_t *_Buf,size_t _SizeInWords,const struct tm *_Tm);
func X_wasctime_s(tls *TLS, __Buf uintptr, __SizeInWords Tsize_t, __Tm uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_wasctime_s.Addr(), __Buf, uintptr(__SizeInWords), __Tm)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_wctime32 = modcrt.NewProc(GoString(__ccgo_ts + 2623))

// wchar_t * __attribute__((__cdecl__)) _wctime32(const __time32_t *_Time);
func X_wctime32(tls *TLS, __Time uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wctime32.Addr(), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wctime32_s = modcrt.NewProc(GoString(__ccgo_ts + 2633))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _wctime32_s (wchar_t *_Buf,size_t _SizeInWords,const __time32_t *_Time);
func X_wctime32_s(tls *TLS, __Buf uintptr, __SizeInWords Tsize_t, __Time uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_wctime32_s.Addr(), __Buf, uintptr(__SizeInWords), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var procwcsftime = modcrt.NewProc(GoString(__ccgo_ts + 2645))

// size_t __attribute__((__cdecl__)) wcsftime(wchar_t * __restrict__ _Buf,size_t _SizeInWords,const wchar_t * __restrict__ _Format,const struct tm * __restrict__ _Tm);
func Xwcsftime(tls *TLS, __Buf uintptr, __SizeInWords Tsize_t, __Format uintptr, __Tm uintptr) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procwcsftime.Addr(), __Buf, uintptr(__SizeInWords), __Format, __Tm)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_wcsftime_l = modcrt.NewProc(GoString(__ccgo_ts + 2654))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _wcsftime_l(wchar_t * __restrict__ _Buf,size_t _SizeInWords,const wchar_t * __restrict__ _Format,const struct tm * __restrict__ _Tm,_locale_t _Locale);
func X_wcsftime_l(tls *TLS, __Buf uintptr, __SizeInWords Tsize_t, __Format uintptr, __Tm uintptr, __Locale T_locale_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_wcsftime_l.Addr(), __Buf, uintptr(__SizeInWords), __Format, __Tm, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_wstrdate = modcrt.NewProc(GoString(__ccgo_ts + 2666))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wstrdate(wchar_t *_Buffer);
func X_wstrdate(tls *TLS, __Buffer uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wstrdate.Addr(), __Buffer)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wstrdate_s = modcrt.NewProc(GoString(__ccgo_ts + 2676))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _wstrdate_s (wchar_t *_Buf,size_t _SizeInWords);
func X_wstrdate_s(tls *TLS, __Buf uintptr, __SizeInWords Tsize_t) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_wstrdate_s.Addr(), __Buf, uintptr(__SizeInWords))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_wstrtime = modcrt.NewProc(GoString(__ccgo_ts + 2688))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wstrtime(wchar_t *_Buffer);
func X_wstrtime(tls *TLS, __Buffer uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wstrtime.Addr(), __Buffer)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wstrtime_s = modcrt.NewProc(GoString(__ccgo_ts + 2698))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _wstrtime_s (wchar_t *_Buf,size_t _SizeInWords);
func X_wstrtime_s(tls *TLS, __Buf uintptr, __SizeInWords Tsize_t) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_wstrtime_s.Addr(), __Buf, uintptr(__SizeInWords))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_wctime64 = modcrt.NewProc(GoString(__ccgo_ts + 2710))

// __attribute__ ((__dllimport__)) wchar_t * __attribute__((__cdecl__)) _wctime64(const __time64_t *_Time);
func X_wctime64(tls *TLS, __Time uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wctime64.Addr(), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wctime64_s = modcrt.NewProc(GoString(__ccgo_ts + 2720))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _wctime64_s (wchar_t *_Buf,size_t _SizeInWords,const __time64_t *_Time);
func X_wctime64_s(tls *TLS, __Buf uintptr, __SizeInWords Tsize_t, __Time uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_wctime64_s.Addr(), __Buf, uintptr(__SizeInWords), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_wctime = modcrt.NewProc(GoString(__ccgo_ts + 2732))

// wchar_t * __attribute__((__cdecl__)) _wctime(const time_t *_Time);
func X_wctime(tls *TLS, __Time uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_wctime.Addr(), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wctime_s = modcrt.NewProc(GoString(__ccgo_ts + 2740))

// errno_t __attribute__((__cdecl__)) _wctime_s(wchar_t *, size_t, const time_t *);
func X_wctime_s(tls *TLS, _0 uintptr, _1 Tsize_t, _2 uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_wctime_s.Addr(), _0, uintptr(_1), _2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

type Tmbstate_t = int32

type T_Wint_t = uint16

var procbtowc = modcrt.NewProc(GoString(__ccgo_ts + 2750))

// wint_t __attribute__((__cdecl__)) btowc(int);
func Xbtowc(tls *TLS, _0 int32) (r Twint_t) {
	r0, _, err := syscall.SyscallN(procbtowc.Addr(), uintptr(_0))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Twint_t(r0)
}

var procmbrlen = modcrt.NewProc(GoString(__ccgo_ts + 2756))

// size_t __attribute__((__cdecl__)) mbrlen(const char * __restrict__ _Ch,size_t _SizeInBytes,mbstate_t * __restrict__ _State);
func Xmbrlen(tls *TLS, __Ch uintptr, __SizeInBytes Tsize_t, __State uintptr) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procmbrlen.Addr(), __Ch, uintptr(__SizeInBytes), __State)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var procmbrtowc = modcrt.NewProc(GoString(__ccgo_ts + 2763))

// size_t __attribute__((__cdecl__)) mbrtowc(wchar_t * __restrict__ _DstCh,const char * __restrict__ _SrcCh,size_t _SizeInBytes,mbstate_t * __restrict__ _State);
func Xmbrtowc(tls *TLS, __DstCh uintptr, __SrcCh uintptr, __SizeInBytes Tsize_t, __State uintptr) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procmbrtowc.Addr(), __DstCh, __SrcCh, uintptr(__SizeInBytes), __State)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var procmbsrtowcs = modcrt.NewProc(GoString(__ccgo_ts + 2771))

// size_t __attribute__((__cdecl__)) mbsrtowcs(wchar_t * __restrict__ _Dest,const char ** __restrict__ _PSrc,size_t _Count,mbstate_t * __restrict__ _State);
func Xmbsrtowcs(tls *TLS, __Dest uintptr, __PSrc uintptr, __Count Tsize_t, __State uintptr) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procmbsrtowcs.Addr(), __Dest, __PSrc, uintptr(__Count), __State)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var procwctob = modcrt.NewProc(GoString(__ccgo_ts + 2781))

// int __attribute__((__cdecl__)) wctob(wint_t _WCh);
func Xwctob(tls *TLS, __WCh Twint_t) (r int32) {
	r0, _, err := syscall.SyscallN(procwctob.Addr(), uintptr(__WCh))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procwmemset = modcrt.NewProc(GoString(__ccgo_ts + 2787))

// wchar_t * __attribute__((__cdecl__)) wmemset(wchar_t *s, wchar_t c, size_t n);
func Xwmemset(tls *TLS, _s uintptr, _c Twchar_t, _n Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwmemset.Addr(), _s, uintptr(_c), uintptr(_n))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwmemchr = modcrt.NewProc(GoString(__ccgo_ts + 2795))

// wchar_t * __attribute__((__cdecl__)) wmemchr(const wchar_t *s, wchar_t c, size_t n);
func Xwmemchr(tls *TLS, _s uintptr, _c Twchar_t, _n Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwmemchr.Addr(), _s, uintptr(_c), uintptr(_n))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwmemcmp = modcrt.NewProc(GoString(__ccgo_ts + 2803))

// int __attribute__((__cdecl__)) wmemcmp(const wchar_t *s1, const wchar_t *s2,size_t n);
func Xwmemcmp(tls *TLS, _s1 uintptr, _s2 uintptr, _n Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(procwmemcmp.Addr(), _s1, _s2, uintptr(_n))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procwmemcpy = modcrt.NewProc(GoString(__ccgo_ts + 2811))

// wchar_t * __attribute__((__cdecl__)) wmemcpy(wchar_t * __restrict__ s1,const wchar_t * __restrict__ s2,size_t n);
func Xwmemcpy(tls *TLS, _s1 uintptr, _s2 uintptr, _n Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwmemcpy.Addr(), _s1, _s2, uintptr(_n))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwmempcpy = modcrt.NewProc(GoString(__ccgo_ts + 2819))

// wchar_t * __attribute__((__cdecl__)) wmempcpy (wchar_t *_Dst, const wchar_t *_Src, size_t _Size);
func Xwmempcpy(tls *TLS, __Dst uintptr, __Src uintptr, __Size Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwmempcpy.Addr(), __Dst, __Src, uintptr(__Size))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procwmemmove = modcrt.NewProc(GoString(__ccgo_ts + 2828))

// wchar_t * __attribute__((__cdecl__)) wmemmove(wchar_t *s1, const wchar_t *s2, size_t n);
func Xwmemmove(tls *TLS, _s1 uintptr, _s2 uintptr, _n Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procwmemmove.Addr(), _s1, _s2, uintptr(_n))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procfwide = modcrt.NewProc(GoString(__ccgo_ts + 2837))

// int __attribute__((__cdecl__)) fwide(FILE *stream,int mode);
func Xfwide(tls *TLS, _stream uintptr, _mode int32) (r int32) {
	r0, _, err := syscall.SyscallN(procfwide.Addr(), _stream, uintptr(_mode))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procmbsinit = modcrt.NewProc(GoString(__ccgo_ts + 2843))

// int __attribute__((__cdecl__)) mbsinit(const mbstate_t *ps);
func Xmbsinit(tls *TLS, _ps uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(procmbsinit.Addr(), _ps)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procwcstoll = modcrt.NewProc(GoString(__ccgo_ts + 2851))

// long long __attribute__((__cdecl__)) wcstoll(const wchar_t * __restrict__ nptr,wchar_t ** __restrict__ endptr, int base);
func Xwcstoll(tls *TLS, _nptr uintptr, _endptr uintptr, _base int32) (r int64) {
	r0, _, err := syscall.SyscallN(procwcstoll.Addr(), _nptr, _endptr, uintptr(_base))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var procwcstoull = modcrt.NewProc(GoString(__ccgo_ts + 2859))

// unsigned long long __attribute__((__cdecl__)) wcstoull(const wchar_t * __restrict__ nptr,wchar_t ** __restrict__ endptr, int base);
func Xwcstoull(tls *TLS, _nptr uintptr, _endptr uintptr, _base int32) (r uint64) {
	r0, _, err := syscall.SyscallN(procwcstoull.Addr(), _nptr, _endptr, uintptr(_base))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint64(r0)
}

var proc__mingw_str_wide_utf8 = modcrt.NewProc(GoString(__ccgo_ts + 2868))

// int __attribute__((__cdecl__)) __mingw_str_wide_utf8 (const wchar_t * const wptr, char **mbptr, size_t * buflen);
func X__mingw_str_wide_utf8(tls *TLS, _wptr uintptr, _mbptr uintptr, _buflen uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc__mingw_str_wide_utf8.Addr(), _wptr, _mbptr, _buflen)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__mingw_str_utf8_wide = modcrt.NewProc(GoString(__ccgo_ts + 2890))

// int __attribute__((__cdecl__)) __mingw_str_utf8_wide (const char *const mbptr, wchar_t ** wptr, size_t * buflen);
func X__mingw_str_utf8_wide(tls *TLS, _mbptr uintptr, _wptr uintptr, _buflen uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc__mingw_str_utf8_wide.Addr(), _mbptr, _wptr, _buflen)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc__mingw_str_free = modcrt.NewProc(GoString(__ccgo_ts + 2912))

// void __attribute__((__cdecl__)) __mingw_str_free(void *ptr);
func X__mingw_str_free(tls *TLS, _ptr uintptr) {
	_, _, err := syscall.SyscallN(proc__mingw_str_free.Addr(), _ptr)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_memccpy = modcrt.NewProc(GoString(__ccgo_ts + 2929))

// __attribute__ ((__dllimport__)) void * __attribute__((__cdecl__)) _memccpy(void *_Dst,const void *_Src,int _Val,size_t _MaxCount);
func X_memccpy(tls *TLS, __Dst uintptr, __Src uintptr, __Val int32, __MaxCount Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_memccpy.Addr(), __Dst, __Src, uintptr(__Val), uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_memicmp = modcrt.NewProc(GoString(__ccgo_ts + 2938))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _memicmp(const void *_Buf1,const void *_Buf2,size_t _Size);
func X_memicmp(tls *TLS, __Buf1 uintptr, __Buf2 uintptr, __Size Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_memicmp.Addr(), __Buf1, __Buf2, uintptr(__Size))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_memicmp_l = modcrt.NewProc(GoString(__ccgo_ts + 2947))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _memicmp_l(const void *_Buf1,const void *_Buf2,size_t _Size,_locale_t _Locale);
func X_memicmp_l(tls *TLS, __Buf1 uintptr, __Buf2 uintptr, __Size Tsize_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_memicmp_l.Addr(), __Buf1, __Buf2, uintptr(__Size), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procmemcpy_s = modcrt.NewProc(GoString(__ccgo_ts + 2958))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) memcpy_s (void *_dest,size_t _numberOfElements,const void *_src,size_t _count);
func Xmemcpy_s(tls *TLS, __dest uintptr, __numberOfElements Tsize_t, __src uintptr, __count Tsize_t) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(procmemcpy_s.Addr(), __dest, uintptr(__numberOfElements), __src, uintptr(__count))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var procmempcpy = modcrt.NewProc(GoString(__ccgo_ts + 2967))

// void * __attribute__((__cdecl__)) mempcpy (void *_Dst, const void *_Src, size_t _Size);
func Xmempcpy(tls *TLS, __Dst uintptr, __Src uintptr, __Size Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procmempcpy.Addr(), __Dst, __Src, uintptr(__Size))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procmemccpy = modcrt.NewProc(GoString(__ccgo_ts + 2975))

// void * __attribute__((__cdecl__)) memccpy(void *_Dst,const void *_Src,int _Val,size_t _Size);
func Xmemccpy(tls *TLS, __Dst uintptr, __Src uintptr, __Val int32, __Size Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procmemccpy.Addr(), __Dst, __Src, uintptr(__Val), uintptr(__Size))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procmemicmp = modcrt.NewProc(GoString(__ccgo_ts + 2983))

// int __attribute__((__cdecl__)) memicmp(const void *_Buf1,const void *_Buf2,size_t _Size);
func Xmemicmp(tls *TLS, __Buf1 uintptr, __Buf2 uintptr, __Size Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(procmemicmp.Addr(), __Buf1, __Buf2, uintptr(__Size))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_strset = modcrt.NewProc(GoString(__ccgo_ts + 2991))

// char * __attribute__((__cdecl__)) _strset(char *_Str,int _Val);
func X_strset(tls *TLS, __Str uintptr, __Val int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strset.Addr(), __Str, uintptr(__Val))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_strset_l = modcrt.NewProc(GoString(__ccgo_ts + 2999))

// char * __attribute__((__cdecl__)) _strset_l(char *_Str,int _Val,_locale_t _Locale);
func X_strset_l(tls *TLS, __Str uintptr, __Val int32, __Locale T_locale_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strset_l.Addr(), __Str, uintptr(__Val), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrnlen = modcrt.NewProc(GoString(__ccgo_ts + 3009))

// size_t __attribute__((__cdecl__)) strnlen(const char *_Str,size_t _MaxCount);
func Xstrnlen(tls *TLS, __Str uintptr, __MaxCount Tsize_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procstrnlen.Addr(), __Str, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_strcmpi = modcrt.NewProc(GoString(__ccgo_ts + 3017))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _strcmpi(const char *_Str1,const char *_Str2);
func X_strcmpi(tls *TLS, __Str1 uintptr, __Str2 uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_strcmpi.Addr(), __Str1, __Str2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_stricmp_l = modcrt.NewProc(GoString(__ccgo_ts + 3026))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _stricmp_l(const char *_Str1,const char *_Str2,_locale_t _Locale);
func X_stricmp_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_stricmp_l.Addr(), __Str1, __Str2, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procstrcoll = modcrt.NewProc(GoString(__ccgo_ts + 3037))

// int __attribute__((__cdecl__)) strcoll(const char *_Str1,const char *_Str2);
func Xstrcoll(tls *TLS, __Str1 uintptr, __Str2 uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(procstrcoll.Addr(), __Str1, __Str2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_strcoll_l = modcrt.NewProc(GoString(__ccgo_ts + 3045))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _strcoll_l(const char *_Str1,const char *_Str2,_locale_t _Locale);
func X_strcoll_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_strcoll_l.Addr(), __Str1, __Str2, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_stricoll = modcrt.NewProc(GoString(__ccgo_ts + 3056))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _stricoll(const char *_Str1,const char *_Str2);
func X_stricoll(tls *TLS, __Str1 uintptr, __Str2 uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_stricoll.Addr(), __Str1, __Str2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_stricoll_l = modcrt.NewProc(GoString(__ccgo_ts + 3066))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _stricoll_l(const char *_Str1,const char *_Str2,_locale_t _Locale);
func X_stricoll_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_stricoll_l.Addr(), __Str1, __Str2, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_strncoll = modcrt.NewProc(GoString(__ccgo_ts + 3078))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _strncoll (const char *_Str1,const char *_Str2,size_t _MaxCount);
func X_strncoll(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_strncoll.Addr(), __Str1, __Str2, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_strncoll_l = modcrt.NewProc(GoString(__ccgo_ts + 3088))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _strncoll_l(const char *_Str1,const char *_Str2,size_t _MaxCount,_locale_t _Locale);
func X_strncoll_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_strncoll_l.Addr(), __Str1, __Str2, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_strnicoll = modcrt.NewProc(GoString(__ccgo_ts + 3100))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _strnicoll (const char *_Str1,const char *_Str2,size_t _MaxCount);
func X_strnicoll(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_strnicoll.Addr(), __Str1, __Str2, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_strnicoll_l = modcrt.NewProc(GoString(__ccgo_ts + 3111))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _strnicoll_l(const char *_Str1,const char *_Str2,size_t _MaxCount,_locale_t _Locale);
func X_strnicoll_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_strnicoll_l.Addr(), __Str1, __Str2, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_strerror = modcrt.NewProc(GoString(__ccgo_ts + 3124))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _strerror(const char *_ErrMsg);
func X_strerror(tls *TLS, __ErrMsg uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strerror.Addr(), __ErrMsg)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_strlwr = modcrt.NewProc(GoString(__ccgo_ts + 3134))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _strlwr(char *_String);
func X_strlwr(tls *TLS, __String uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strlwr.Addr(), __String)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrlwr_l = modcrt.NewProc(GoString(__ccgo_ts + 3142))

// char *strlwr_l(char *_String,_locale_t _Locale);
func Xstrlwr_l(tls *TLS, __String uintptr, __Locale T_locale_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procstrlwr_l.Addr(), __String, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrncat = modcrt.NewProc(GoString(__ccgo_ts + 3151))

// char * __attribute__((__cdecl__)) strncat(char * __restrict__ _Dest,const char * __restrict__ _Source,size_t _Count);
func Xstrncat(tls *TLS, __Dest uintptr, __Source uintptr, __Count Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procstrncat.Addr(), __Dest, __Source, uintptr(__Count))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_strnicmp_l = modcrt.NewProc(GoString(__ccgo_ts + 3159))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _strnicmp_l(const char *_Str1,const char *_Str2,size_t _MaxCount,_locale_t _Locale);
func X_strnicmp_l(tls *TLS, __Str1 uintptr, __Str2 uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_strnicmp_l.Addr(), __Str1, __Str2, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_strnset = modcrt.NewProc(GoString(__ccgo_ts + 3171))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _strnset(char *_Str,int _Val,size_t _MaxCount);
func X_strnset(tls *TLS, __Str uintptr, __Val int32, __MaxCount Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strnset.Addr(), __Str, uintptr(__Val), uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_strnset_l = modcrt.NewProc(GoString(__ccgo_ts + 3180))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _strnset_l(char *str,int c,size_t count,_locale_t _Locale);
func X_strnset_l(tls *TLS, _str uintptr, _c int32, _count Tsize_t, __Locale T_locale_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strnset_l.Addr(), _str, uintptr(_c), uintptr(_count), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_strrev = modcrt.NewProc(GoString(__ccgo_ts + 3191))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _strrev(char *_Str);
func X_strrev(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strrev.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrtok = modcrt.NewProc(GoString(__ccgo_ts + 3199))

// char * __attribute__((__cdecl__)) strtok(char * __restrict__ _Str,const char * __restrict__ _Delim);
func Xstrtok(tls *TLS, __Str uintptr, __Delim uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procstrtok.Addr(), __Str, __Delim)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrtok_r = modcrt.NewProc(GoString(__ccgo_ts + 3206))

// char *strtok_r(char * __restrict__ _Str, const char * __restrict__ _Delim, char ** __restrict__ __last);
func Xstrtok_r(tls *TLS, __Str uintptr, __Delim uintptr, ___last uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procstrtok_r.Addr(), __Str, __Delim, ___last)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_strupr = modcrt.NewProc(GoString(__ccgo_ts + 3215))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _strupr(char *_String);
func X_strupr(tls *TLS, __String uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strupr.Addr(), __String)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_strupr_l = modcrt.NewProc(GoString(__ccgo_ts + 3223))

// __attribute__ ((__dllimport__)) char *_strupr_l(char *_String,_locale_t _Locale);
func X_strupr_l(tls *TLS, __String uintptr, __Locale T_locale_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strupr_l.Addr(), __String, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrxfrm = modcrt.NewProc(GoString(__ccgo_ts + 3233))

// size_t __attribute__((__cdecl__)) strxfrm(char * __restrict__ _Dst,const char * __restrict__ _Src,size_t _MaxCount);
func Xstrxfrm(tls *TLS, __Dst uintptr, __Src uintptr, __MaxCount Tsize_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(procstrxfrm.Addr(), __Dst, __Src, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_strxfrm_l = modcrt.NewProc(GoString(__ccgo_ts + 3241))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _strxfrm_l(char * __restrict__ _Dst,const char * __restrict__ _Src,size_t _MaxCount,_locale_t _Locale);
func X_strxfrm_l(tls *TLS, __Dst uintptr, __Src uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_strxfrm_l.Addr(), __Dst, __Src, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var procstrcmpi = modcrt.NewProc(GoString(__ccgo_ts + 3252))

// int __attribute__((__cdecl__)) strcmpi(const char *_Str1,const char *_Str2);
func Xstrcmpi(tls *TLS, __Str1 uintptr, __Str2 uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(procstrcmpi.Addr(), __Str1, __Str2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procstricmp = modcrt.NewProc(GoString(__ccgo_ts + 3260))

// int __attribute__((__cdecl__)) stricmp(const char *_Str1,const char *_Str2);
func Xstricmp(tls *TLS, __Str1 uintptr, __Str2 uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(procstricmp.Addr(), __Str1, __Str2)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procstrlwr = modcrt.NewProc(GoString(__ccgo_ts + 3268))

// char * __attribute__((__cdecl__)) strlwr(char *_Str);
func Xstrlwr(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procstrlwr.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrnicmp = modcrt.NewProc(GoString(__ccgo_ts + 3275))

// int __attribute__((__cdecl__)) strnicmp(const char *_Str1,const char *_Str,size_t _MaxCount);
func Xstrnicmp(tls *TLS, __Str1 uintptr, __Str uintptr, __MaxCount Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(procstrnicmp.Addr(), __Str1, __Str, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procstrncasecmp = modcrt.NewProc(GoString(__ccgo_ts + 3284))

// int __attribute__((__cdecl__)) strncasecmp (const char *, const char *, size_t);
func Xstrncasecmp(tls *TLS, _0 uintptr, _1 uintptr, _2 Tsize_t) (r int32) {
	r0, _, err := syscall.SyscallN(procstrncasecmp.Addr(), _0, _1, uintptr(_2))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procstrnset = modcrt.NewProc(GoString(__ccgo_ts + 3296))

// char * __attribute__((__cdecl__)) strnset(char *_Str,int _Val,size_t _MaxCount);
func Xstrnset(tls *TLS, __Str uintptr, __Val int32, __MaxCount Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(procstrnset.Addr(), __Str, uintptr(__Val), uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrrev = modcrt.NewProc(GoString(__ccgo_ts + 3304))

// char * __attribute__((__cdecl__)) strrev(char *_Str);
func Xstrrev(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procstrrev.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrset = modcrt.NewProc(GoString(__ccgo_ts + 3311))

// char * __attribute__((__cdecl__)) strset(char *_Str,int _Val);
func Xstrset(tls *TLS, __Str uintptr, __Val int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(procstrset.Addr(), __Str, uintptr(__Val))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procstrupr = modcrt.NewProc(GoString(__ccgo_ts + 3318))

// char * __attribute__((__cdecl__)) strupr(char *_Str);
func Xstrupr(tls *TLS, __Str uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procstrupr.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

type T__timeb32 = struct {
	Ftime     T__time32_t
	Fmillitm  uint16
	Ftimezone int16
	Fdstflag  int16
}

type Ttimeb = struct {
	Ftime     Ttime_t
	Fmillitm  uint16
	Ftimezone int16
	Fdstflag  int16
}

type T__timeb64 = struct {
	Ftime     T__time64_t
	Fmillitm  uint16
	Ftimezone int16
	Fdstflag  int16
}

type T_timespec32 = struct {
	Ftv_sec  T__time32_t
	Ftv_nsec int32
}

type T_timespec64 = struct {
	Ftv_sec  T__time64_t
	Ftv_nsec int32
}

type Ttimespec = struct {
	Ftv_sec  Ttime_t
	Ftv_nsec int32
}

type Titimerspec = struct {
	Fit_interval Ttimespec
	Fit_value    Ttimespec
}

type Tclock_t = int32

var proc_get_daylight = modcrt.NewProc(GoString(__ccgo_ts + 3325))

// __attribute__ ((__dllimport__)) errno_t __attribute__((__cdecl__)) _get_daylight(int *_Daylight);
func X_get_daylight(tls *TLS, __Daylight uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_daylight.Addr(), __Daylight)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_dstbias = modcrt.NewProc(GoString(__ccgo_ts + 3339))

// __attribute__ ((__dllimport__)) errno_t __attribute__((__cdecl__)) _get_dstbias(long *_Daylight_savings_bias);
func X_get_dstbias(tls *TLS, __Daylight_savings_bias uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_dstbias.Addr(), __Daylight_savings_bias)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_timezone = modcrt.NewProc(GoString(__ccgo_ts + 3352))

// __attribute__ ((__dllimport__)) errno_t __attribute__((__cdecl__)) _get_timezone(long *_Timezone);
func X_get_timezone(tls *TLS, __Timezone uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_timezone.Addr(), __Timezone)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_tzname = modcrt.NewProc(GoString(__ccgo_ts + 3366))

// __attribute__ ((__dllimport__)) errno_t __attribute__((__cdecl__)) _get_tzname(size_t *_ReturnValue,char *_Buffer,size_t _SizeInBytes,int _Index);
func X_get_tzname(tls *TLS, __ReturnValue uintptr, __Buffer uintptr, __SizeInBytes Tsize_t, __Index int32) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_tzname.Addr(), __ReturnValue, __Buffer, uintptr(__SizeInBytes), uintptr(__Index))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var procasctime = modcrt.NewProc(GoString(__ccgo_ts + 3378))

// char * __attribute__((__cdecl__)) asctime(const struct tm *_Tm);
func Xasctime(tls *TLS, __Tm uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procasctime.Addr(), __Tm)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procasctime_s = modcrt.NewProc(GoString(__ccgo_ts + 3386))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) asctime_s (char *_Buf,size_t _SizeInWords,const struct tm *_Tm);
func Xasctime_s(tls *TLS, __Buf uintptr, __SizeInWords Tsize_t, __Tm uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(procasctime_s.Addr(), __Buf, uintptr(__SizeInWords), __Tm)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_ctime32 = modcrt.NewProc(GoString(__ccgo_ts + 3396))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _ctime32(const __time32_t *_Time);
func X_ctime32(tls *TLS, __Time uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_ctime32.Addr(), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_ctime32_s = modcrt.NewProc(GoString(__ccgo_ts + 3405))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _ctime32_s (char *_Buf,size_t _SizeInBytes,const __time32_t *_Time);
func X_ctime32_s(tls *TLS, __Buf uintptr, __SizeInBytes Tsize_t, __Time uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_ctime32_s.Addr(), __Buf, uintptr(__SizeInBytes), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var procclock = modcrt.NewProc(GoString(__ccgo_ts + 3416))

// clock_t __attribute__((__cdecl__)) clock(void);
func Xclock(tls *TLS) (r Tclock_t) {
	r0, _, err := syscall.SyscallN(procclock.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tclock_t(r0)
}

var proc_difftime32 = modcrt.NewProc(GoString(__ccgo_ts + 3422))

// __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _difftime32(__time32_t _Time1,__time32_t _Time2);
func X_difftime32(tls *TLS, __Time1 T__time32_t, __Time2 T__time32_t) (r float64) {
	r0, _, err := syscall.SyscallN(proc_difftime32.Addr(), uintptr(__Time1), uintptr(__Time2))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc_gmtime32 = modcrt.NewProc(GoString(__ccgo_ts + 3434))

// __attribute__ ((__dllimport__)) struct tm * __attribute__((__cdecl__)) _gmtime32(const __time32_t *_Time);
func X_gmtime32(tls *TLS, __Time uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_gmtime32.Addr(), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_gmtime32_s = modcrt.NewProc(GoString(__ccgo_ts + 3444))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _gmtime32_s (struct tm *_Tm,const __time32_t *_Time);
func X_gmtime32_s(tls *TLS, __Tm uintptr, __Time uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_gmtime32_s.Addr(), __Tm, __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_localtime32 = modcrt.NewProc(GoString(__ccgo_ts + 3456))

// __attribute__ ((__dllimport__)) struct tm * __attribute__((__cdecl__)) _localtime32(const __time32_t *_Time);
func X_localtime32(tls *TLS, __Time uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_localtime32.Addr(), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_localtime32_s = modcrt.NewProc(GoString(__ccgo_ts + 3469))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _localtime32_s (struct tm *_Tm,const __time32_t *_Time);
func X_localtime32_s(tls *TLS, __Tm uintptr, __Time uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_localtime32_s.Addr(), __Tm, __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_strftime_l = modcrt.NewProc(GoString(__ccgo_ts + 3484))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _strftime_l(char * __restrict__ _Buf,size_t _Max_size,const char * __restrict__ _Format,const struct tm * __restrict__ _Tm,_locale_t _Locale);
func X_strftime_l(tls *TLS, __Buf uintptr, __Max_size Tsize_t, __Format uintptr, __Tm uintptr, __Locale T_locale_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_strftime_l.Addr(), __Buf, uintptr(__Max_size), __Format, __Tm, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_strdate = modcrt.NewProc(GoString(__ccgo_ts + 3496))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _strdate(char *_Buffer);
func X_strdate(tls *TLS, __Buffer uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strdate.Addr(), __Buffer)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_strdate_s = modcrt.NewProc(GoString(__ccgo_ts + 3505))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _strdate_s (char *_Buf,size_t _SizeInBytes);
func X_strdate_s(tls *TLS, __Buf uintptr, __SizeInBytes Tsize_t) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_strdate_s.Addr(), __Buf, uintptr(__SizeInBytes))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_strtime = modcrt.NewProc(GoString(__ccgo_ts + 3516))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _strtime(char *_Buffer);
func X_strtime(tls *TLS, __Buffer uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_strtime.Addr(), __Buffer)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_strtime_s = modcrt.NewProc(GoString(__ccgo_ts + 3525))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _strtime_s (char *_Buf ,size_t _SizeInBytes);
func X_strtime_s(tls *TLS, __Buf uintptr, __SizeInBytes Tsize_t) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_strtime_s.Addr(), __Buf, uintptr(__SizeInBytes))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_time32 = modcrt.NewProc(GoString(__ccgo_ts + 3536))

// __attribute__ ((__dllimport__)) __time32_t __attribute__((__cdecl__)) _time32(__time32_t *_Time);
func X_time32(tls *TLS, __Time uintptr) (r T__time32_t) {
	r0, _, err := syscall.SyscallN(proc_time32.Addr(), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T__time32_t(r0)
}

var proc_mktime32 = modcrt.NewProc(GoString(__ccgo_ts + 3544))

// __attribute__ ((__dllimport__)) __time32_t __attribute__((__cdecl__)) _mktime32(struct tm *_Tm);
func X_mktime32(tls *TLS, __Tm uintptr) (r T__time32_t) {
	r0, _, err := syscall.SyscallN(proc_mktime32.Addr(), __Tm)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T__time32_t(r0)
}

var proc_mkgmtime32 = modcrt.NewProc(GoString(__ccgo_ts + 3554))

// __attribute__ ((__dllimport__)) __time32_t __attribute__((__cdecl__)) _mkgmtime32(struct tm *_Tm);
func X_mkgmtime32(tls *TLS, __Tm uintptr) (r T__time32_t) {
	r0, _, err := syscall.SyscallN(proc_mkgmtime32.Addr(), __Tm)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T__time32_t(r0)
}

var proc_tzset = modcrt.NewProc(GoString(__ccgo_ts + 3566))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _tzset(void);
func X_tzset(tls *TLS) {
	_, _, err := syscall.SyscallN(proc_tzset.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_difftime64 = modcrt.NewProc(GoString(__ccgo_ts + 3573))

// __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _difftime64(__time64_t _Time1,__time64_t _Time2);
func X_difftime64(tls *TLS, __Time1 T__time64_t, __Time2 T__time64_t) (r float64) {
	r0, _, err := syscall.SyscallN(proc_difftime64.Addr(), uintptr(__Time1), uintptr(__Time2))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc_ctime64 = modcrt.NewProc(GoString(__ccgo_ts + 3585))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _ctime64(const __time64_t *_Time);
func X_ctime64(tls *TLS, __Time uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_ctime64.Addr(), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_ctime64_s = modcrt.NewProc(GoString(__ccgo_ts + 3594))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _ctime64_s (char *_Buf,size_t _SizeInBytes,const __time64_t *_Time);
func X_ctime64_s(tls *TLS, __Buf uintptr, __SizeInBytes Tsize_t, __Time uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_ctime64_s.Addr(), __Buf, uintptr(__SizeInBytes), __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_gmtime64_s = modcrt.NewProc(GoString(__ccgo_ts + 3605))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _gmtime64_s (struct tm *_Tm,const __time64_t *_Time);
func X_gmtime64_s(tls *TLS, __Tm uintptr, __Time uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_gmtime64_s.Addr(), __Tm, __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_localtime64_s = modcrt.NewProc(GoString(__ccgo_ts + 3617))

// __attribute__((dllimport)) errno_t __attribute__((__cdecl__)) _localtime64_s (struct tm *_Tm,const __time64_t *_Time);
func X_localtime64_s(tls *TLS, __Tm uintptr, __Time uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_localtime64_s.Addr(), __Tm, __Time)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_mkgmtime64 = modcrt.NewProc(GoString(__ccgo_ts + 3632))

// __attribute__ ((__dllimport__)) __time64_t __attribute__((__cdecl__)) _mkgmtime64(struct tm *_Tm);
func X_mkgmtime64(tls *TLS, __Tm uintptr) (r T__time64_t) {
	r0, _, err := syscall.SyscallN(proc_mkgmtime64.Addr(), __Tm)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T__time64_t(r0)
}

var proc_getsystime = modcrt.NewProc(GoString(__ccgo_ts + 3644))

// unsigned __attribute__((__cdecl__)) _getsystime(struct tm *_Tm);
func X_getsystime(tls *TLS, __Tm uintptr) (r uint32) {
	r0, _, err := syscall.SyscallN(proc_getsystime.Addr(), __Tm)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint32(r0)
}

var proc_setsystime = modcrt.NewProc(GoString(__ccgo_ts + 3656))

// unsigned __attribute__((__cdecl__)) _setsystime(struct tm *_Tm,unsigned _MilliSec);
func X_setsystime(tls *TLS, __Tm uintptr, __MilliSec uint32) (r uint32) {
	r0, _, err := syscall.SyscallN(proc_setsystime.Addr(), __Tm, uintptr(__MilliSec))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint32(r0)
}

type Ttimeval = struct {
	Ftv_sec  int32
	Ftv_usec int32
}

type Ttimezone = struct {
	Ftz_minuteswest int32
	Ftz_dsttime     int32
}

var procmingw_gettimeofday = modcrt.NewProc(GoString(__ccgo_ts + 3668))

// extern int __attribute__((__cdecl__)) mingw_gettimeofday (struct timeval *p, struct timezone *z);
func Xmingw_gettimeofday(tls *TLS, _p uintptr, _z uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(procmingw_gettimeofday.Addr(), _p, _z)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

type Tclockid_t = int32

type T_onexit_t = uintptr

type Tdiv_t = struct {
	Fquot int32
	Frem  int32
}

type T_div_t = Tdiv_t

type Tldiv_t = struct {
	Fquot int32
	Frem  int32
}

type T_ldiv_t = Tldiv_t

type T_LDOUBLE = struct {
	Fld [10]uint8
}

type T_CRT_DOUBLE = struct {
	Fx float64
}

type T_CRT_FLOAT = struct {
	Ff float32
}

type T_LONGDOUBLE = struct {
	Fx float64
}

type T_LDBL12 = struct {
	Fld12 [12]uint8
}

var proc___mb_cur_max_func = modcrt.NewProc(GoString(__ccgo_ts + 3687))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) ___mb_cur_max_func(void);
func X___mb_cur_max_func(tls *TLS) (r int32) {
	r0, _, err := syscall.SyscallN(proc___mb_cur_max_func.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

type T_purecall_handler = uintptr

var proc_set_purecall_handler = modcrt.NewProc(GoString(__ccgo_ts + 3706))

// __attribute__ ((__dllimport__)) _purecall_handler __attribute__((__cdecl__)) _set_purecall_handler(_purecall_handler _Handler);
func X_set_purecall_handler(tls *TLS, __Handler T_purecall_handler) (r T_purecall_handler) {
	r0, _, err := syscall.SyscallN(proc_set_purecall_handler.Addr(), __Handler)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T_purecall_handler(r0)
}

var proc_get_purecall_handler = modcrt.NewProc(GoString(__ccgo_ts + 3728))

// __attribute__ ((__dllimport__)) _purecall_handler __attribute__((__cdecl__)) _get_purecall_handler(void);
func X_get_purecall_handler(tls *TLS) (r T_purecall_handler) {
	r0, _, err := syscall.SyscallN(proc_get_purecall_handler.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T_purecall_handler(r0)
}

type T_invalid_parameter_handler = uintptr

var proc_set_invalid_parameter_handler = modcrt.NewProc(GoString(__ccgo_ts + 3750))

// __attribute__ ((__dllimport__)) _invalid_parameter_handler __attribute__((__cdecl__)) _set_invalid_parameter_handler(_invalid_parameter_handler _Handler);
func X_set_invalid_parameter_handler(tls *TLS, __Handler T_invalid_parameter_handler) (r T_invalid_parameter_handler) {
	r0, _, err := syscall.SyscallN(proc_set_invalid_parameter_handler.Addr(), __Handler)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T_invalid_parameter_handler(r0)
}

var proc_get_invalid_parameter_handler = modcrt.NewProc(GoString(__ccgo_ts + 3781))

// __attribute__ ((__dllimport__)) _invalid_parameter_handler __attribute__((__cdecl__)) _get_invalid_parameter_handler(void);
func X_get_invalid_parameter_handler(tls *TLS) (r T_invalid_parameter_handler) {
	r0, _, err := syscall.SyscallN(proc_get_invalid_parameter_handler.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T_invalid_parameter_handler(r0)
}

var proc_set_errno = modcrt.NewProc(GoString(__ccgo_ts + 3812))

// errno_t __attribute__((__cdecl__)) _set_errno(int _Value);
func X_set_errno(tls *TLS, __Value int32) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_set_errno.Addr(), uintptr(__Value))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_errno = modcrt.NewProc(GoString(__ccgo_ts + 3823))

// errno_t __attribute__((__cdecl__)) _get_errno(int *_Value);
func X_get_errno(tls *TLS, __Value uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_errno.Addr(), __Value)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc__doserrno = modcrt.NewProc(GoString(__ccgo_ts + 3834))

// __attribute__ ((__dllimport__)) unsigned long * __attribute__((__cdecl__)) __doserrno(void);
func X__doserrno(tls *TLS) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc__doserrno.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_set_doserrno = modcrt.NewProc(GoString(__ccgo_ts + 3845))

// errno_t __attribute__((__cdecl__)) _set_doserrno(unsigned long _Value);
func X_set_doserrno(tls *TLS, __Value uint32) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_set_doserrno.Addr(), uintptr(__Value))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_doserrno = modcrt.NewProc(GoString(__ccgo_ts + 3859))

// errno_t __attribute__((__cdecl__)) _get_doserrno(unsigned long *_Value);
func X_get_doserrno(tls *TLS, __Value uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_doserrno.Addr(), __Value)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc__p___argv = modcrt.NewProc(GoString(__ccgo_ts + 3873))

// __attribute__ ((__dllimport__)) char *** __attribute__((__cdecl__)) __p___argv(void);
func X__p___argv(tls *TLS) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc__p___argv.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc__p__fmode = modcrt.NewProc(GoString(__ccgo_ts + 3884))

// __attribute__ ((__dllimport__)) int * __attribute__((__cdecl__)) __p__fmode(void);
func X__p__fmode(tls *TLS) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc__p__fmode.Addr())
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_get_pgmptr = modcrt.NewProc(GoString(__ccgo_ts + 3895))

// errno_t __attribute__((__cdecl__)) _get_pgmptr(char **_Value);
func X_get_pgmptr(tls *TLS, __Value uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_pgmptr.Addr(), __Value)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_wpgmptr = modcrt.NewProc(GoString(__ccgo_ts + 3907))

// errno_t __attribute__((__cdecl__)) _get_wpgmptr(wchar_t **_Value);
func X_get_wpgmptr(tls *TLS, __Value uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_wpgmptr.Addr(), __Value)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_set_fmode = modcrt.NewProc(GoString(__ccgo_ts + 3920))

// __attribute__ ((__dllimport__)) errno_t __attribute__((__cdecl__)) _set_fmode(int _Mode);
func X_set_fmode(tls *TLS, __Mode int32) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_set_fmode.Addr(), uintptr(__Mode))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_fmode = modcrt.NewProc(GoString(__ccgo_ts + 3931))

// __attribute__ ((__dllimport__)) errno_t __attribute__((__cdecl__)) _get_fmode(int *_PMode);
func X_get_fmode(tls *TLS, __PMode uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_fmode.Addr(), __PMode)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_osplatform = modcrt.NewProc(GoString(__ccgo_ts + 3942))

// errno_t __attribute__((__cdecl__)) _get_osplatform(unsigned int *_Value);
func X_get_osplatform(tls *TLS, __Value uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_osplatform.Addr(), __Value)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_osver = modcrt.NewProc(GoString(__ccgo_ts + 3958))

// errno_t __attribute__((__cdecl__)) _get_osver(unsigned int *_Value);
func X_get_osver(tls *TLS, __Value uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_osver.Addr(), __Value)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_winver = modcrt.NewProc(GoString(__ccgo_ts + 3969))

// errno_t __attribute__((__cdecl__)) _get_winver(unsigned int *_Value);
func X_get_winver(tls *TLS, __Value uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_winver.Addr(), __Value)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_winmajor = modcrt.NewProc(GoString(__ccgo_ts + 3981))

// errno_t __attribute__((__cdecl__)) _get_winmajor(unsigned int *_Value);
func X_get_winmajor(tls *TLS, __Value uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_winmajor.Addr(), __Value)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_get_winminor = modcrt.NewProc(GoString(__ccgo_ts + 3995))

// errno_t __attribute__((__cdecl__)) _get_winminor(unsigned int *_Value);
func X_get_winminor(tls *TLS, __Value uintptr) (r Terrno_t) {
	r0, _, err := syscall.SyscallN(proc_get_winminor.Addr(), __Value)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Terrno_t(r0)
}

var proc_Exit = modcrt.NewProc(GoString(__ccgo_ts + 4009))

// void __attribute__((__cdecl__)) _Exit(int) __attribute__ ((__noreturn__));
func X_Exit(tls *TLS, _0 int32) {
	_, _, err := syscall.SyscallN(proc_Exit.Addr(), uintptr(_0))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_abs64 = modcrt.NewProc(GoString(__ccgo_ts + 4015))

// long long __attribute__((__cdecl__)) _abs64( long long);
func X_abs64(tls *TLS, _x int64) (r int64) {
	r0, _, err := syscall.SyscallN(proc_abs64.Addr(), uintptr(_x))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proc_atof_l = modcrt.NewProc(GoString(__ccgo_ts + 4022))

// double __attribute__((__cdecl__)) _atof_l(const char *_String,_locale_t _Locale);
func X_atof_l(tls *TLS, __String uintptr, __Locale T_locale_t) (r float64) {
	r0, _, err := syscall.SyscallN(proc_atof_l.Addr(), __String, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc_atoi_l = modcrt.NewProc(GoString(__ccgo_ts + 4030))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _atoi_l(const char *_Str,_locale_t _Locale);
func X_atoi_l(tls *TLS, __Str uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_atoi_l.Addr(), __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_atol_l = modcrt.NewProc(GoString(__ccgo_ts + 4038))

// __attribute__ ((__dllimport__)) long __attribute__((__cdecl__)) _atol_l(const char *_Str,_locale_t _Locale);
func X_atol_l(tls *TLS, __Str uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_atol_l.Addr(), __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_byteswap_ushort = modcrt.NewProc(GoString(__ccgo_ts + 4046))

// unsigned short __attribute__((__cdecl__)) _byteswap_ushort(unsigned short _Short);
func X_byteswap_ushort(tls *TLS, __Short uint16) (r uint16) {
	r0, _, err := syscall.SyscallN(proc_byteswap_ushort.Addr(), uintptr(__Short))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint16(r0)
}

var proc_itoa = modcrt.NewProc(GoString(__ccgo_ts + 4063))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _itoa(int _Value,char *_Dest,int _Radix);
func X_itoa(tls *TLS, __Value int32, __Dest uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_itoa.Addr(), uintptr(__Value), __Dest, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_i64toa = modcrt.NewProc(GoString(__ccgo_ts + 4069))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _i64toa( long long _Val,char *_DstBuf,int _Radix);
func X_i64toa(tls *TLS, __Val int64, __DstBuf uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_i64toa.Addr(), uintptr(__Val), __DstBuf, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_ui64toa = modcrt.NewProc(GoString(__ccgo_ts + 4077))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _ui64toa(unsigned long long _Val,char *_DstBuf,int _Radix);
func X_ui64toa(tls *TLS, __Val uint64, __DstBuf uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_ui64toa.Addr(), uintptr(__Val), __DstBuf, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_atoi64 = modcrt.NewProc(GoString(__ccgo_ts + 4086))

// __attribute__ ((__dllimport__)) long long __attribute__((__cdecl__)) _atoi64(const char *_String);
func X_atoi64(tls *TLS, __String uintptr) (r int64) {
	r0, _, err := syscall.SyscallN(proc_atoi64.Addr(), __String)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proc_atoi64_l = modcrt.NewProc(GoString(__ccgo_ts + 4094))

// __attribute__ ((__dllimport__)) long long __attribute__((__cdecl__)) _atoi64_l(const char *_String,_locale_t _Locale);
func X_atoi64_l(tls *TLS, __String uintptr, __Locale T_locale_t) (r int64) {
	r0, _, err := syscall.SyscallN(proc_atoi64_l.Addr(), __String, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proc_strtoi64 = modcrt.NewProc(GoString(__ccgo_ts + 4104))

// __attribute__ ((__dllimport__)) long long __attribute__((__cdecl__)) _strtoi64(const char *_String,char **_EndPtr,int _Radix);
func X_strtoi64(tls *TLS, __String uintptr, __EndPtr uintptr, __Radix int32) (r int64) {
	r0, _, err := syscall.SyscallN(proc_strtoi64.Addr(), __String, __EndPtr, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proc_strtoi64_l = modcrt.NewProc(GoString(__ccgo_ts + 4114))

// __attribute__ ((__dllimport__)) long long __attribute__((__cdecl__)) _strtoi64_l(const char *_String,char **_EndPtr,int _Radix,_locale_t _Locale);
func X_strtoi64_l(tls *TLS, __String uintptr, __EndPtr uintptr, __Radix int32, __Locale T_locale_t) (r int64) {
	r0, _, err := syscall.SyscallN(proc_strtoi64_l.Addr(), __String, __EndPtr, uintptr(__Radix), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proc_strtoui64 = modcrt.NewProc(GoString(__ccgo_ts + 4126))

// __attribute__ ((__dllimport__)) unsigned long long __attribute__((__cdecl__)) _strtoui64(const char *_String,char **_EndPtr,int _Radix);
func X_strtoui64(tls *TLS, __String uintptr, __EndPtr uintptr, __Radix int32) (r uint64) {
	r0, _, err := syscall.SyscallN(proc_strtoui64.Addr(), __String, __EndPtr, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint64(r0)
}

var proc_strtoui64_l = modcrt.NewProc(GoString(__ccgo_ts + 4137))

// __attribute__ ((__dllimport__)) unsigned long long __attribute__((__cdecl__)) _strtoui64_l(const char *_String,char **_EndPtr,int _Radix,_locale_t _Locale);
func X_strtoui64_l(tls *TLS, __String uintptr, __EndPtr uintptr, __Radix int32, __Locale T_locale_t) (r uint64) {
	r0, _, err := syscall.SyscallN(proc_strtoui64_l.Addr(), __String, __EndPtr, uintptr(__Radix), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint64(r0)
}

var proc_ltoa = modcrt.NewProc(GoString(__ccgo_ts + 4150))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _ltoa(long _Value,char *_Dest,int _Radix);
func X_ltoa(tls *TLS, __Value int32, __Dest uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_ltoa.Addr(), uintptr(__Value), __Dest, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_mblen_l = modcrt.NewProc(GoString(__ccgo_ts + 4156))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _mblen_l(const char *_Ch,size_t _MaxCount,_locale_t _Locale);
func X_mblen_l(tls *TLS, __Ch uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_mblen_l.Addr(), __Ch, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_mbstrlen = modcrt.NewProc(GoString(__ccgo_ts + 4165))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _mbstrlen(const char *_Str);
func X_mbstrlen(tls *TLS, __Str uintptr) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_mbstrlen.Addr(), __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_mbstrlen_l = modcrt.NewProc(GoString(__ccgo_ts + 4175))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _mbstrlen_l(const char *_Str,_locale_t _Locale);
func X_mbstrlen_l(tls *TLS, __Str uintptr, __Locale T_locale_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_mbstrlen_l.Addr(), __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_mbstrnlen = modcrt.NewProc(GoString(__ccgo_ts + 4187))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _mbstrnlen(const char *_Str,size_t _MaxCount);
func X_mbstrnlen(tls *TLS, __Str uintptr, __MaxCount Tsize_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_mbstrnlen.Addr(), __Str, uintptr(__MaxCount))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_mbstrnlen_l = modcrt.NewProc(GoString(__ccgo_ts + 4198))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _mbstrnlen_l(const char *_Str,size_t _MaxCount,_locale_t _Locale);
func X_mbstrnlen_l(tls *TLS, __Str uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_mbstrnlen_l.Addr(), __Str, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_mbtowc_l = modcrt.NewProc(GoString(__ccgo_ts + 4211))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _mbtowc_l(wchar_t * __restrict__ _DstCh,const char * __restrict__ _SrcCh,size_t _SrcSizeInBytes,_locale_t _Locale);
func X_mbtowc_l(tls *TLS, __DstCh uintptr, __SrcCh uintptr, __SrcSizeInBytes Tsize_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_mbtowc_l.Addr(), __DstCh, __SrcCh, uintptr(__SrcSizeInBytes), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_mbstowcs_l = modcrt.NewProc(GoString(__ccgo_ts + 4221))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _mbstowcs_l(wchar_t * __restrict__ _Dest,const char * __restrict__ _Source,size_t _MaxCount,_locale_t _Locale);
func X_mbstowcs_l(tls *TLS, __Dest uintptr, __Source uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_mbstowcs_l.Addr(), __Dest, __Source, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var procmkstemp = modcrt.NewProc(GoString(__ccgo_ts + 4233))

// int __attribute__((__cdecl__)) mkstemp(char *template_name);
func Xmkstemp(tls *TLS, _template_name uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(procmkstemp.Addr(), _template_name)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_set_error_mode = modcrt.NewProc(GoString(__ccgo_ts + 4241))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _set_error_mode(int _Mode);
func X_set_error_mode(tls *TLS, __Mode int32) (r int32) {
	r0, _, err := syscall.SyscallN(proc_set_error_mode.Addr(), uintptr(__Mode))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var procsrand = modcrt.NewProc(GoString(__ccgo_ts + 4257))

// void __attribute__((__cdecl__)) srand(unsigned int _Seed);
func Xsrand(tls *TLS, __Seed uint32) {
	_, _, err := syscall.SyscallN(procsrand.Addr(), uintptr(__Seed))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var procstrtold = modcrt.NewProc(GoString(__ccgo_ts + 4263))

// long double __attribute__((__cdecl__)) __attribute__ ((__nothrow__)) strtold(const char * __restrict__ , char ** __restrict__ );
func Xstrtold(tls *TLS, _0 uintptr, _1 uintptr) (r float64) {
	r0, _, err := syscall.SyscallN(procstrtold.Addr(), _0, _1)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc__strtod = modcrt.NewProc(GoString(__ccgo_ts + 4271))

// extern double __attribute__((__cdecl__)) __attribute__ ((__nothrow__)) __strtod (const char * __restrict__ , char ** __restrict__);
func X__strtod(tls *TLS, _0 uintptr, _1 uintptr) (r float64) {
	r0, _, err := syscall.SyscallN(proc__strtod.Addr(), _0, _1)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc__mingw_strtof = modcrt.NewProc(GoString(__ccgo_ts + 4280))

// float __attribute__((__cdecl__)) __mingw_strtof (const char * __restrict__, char ** __restrict__);
func X__mingw_strtof(tls *TLS, _0 uintptr, _1 uintptr) (r float32) {
	r0, _, err := syscall.SyscallN(proc__mingw_strtof.Addr(), _0, _1)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float32(r0)
}

var proc__mingw_strtold = modcrt.NewProc(GoString(__ccgo_ts + 4295))

// long double __attribute__((__cdecl__)) __mingw_strtold(const char * __restrict__, char ** __restrict__);
func X__mingw_strtold(tls *TLS, _0 uintptr, _1 uintptr) (r float64) {
	r0, _, err := syscall.SyscallN(proc__mingw_strtold.Addr(), _0, _1)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc_strtod_l = modcrt.NewProc(GoString(__ccgo_ts + 4311))

// __attribute__ ((__dllimport__)) double __attribute__((__cdecl__)) _strtod_l(const char * __restrict__ _Str,char ** __restrict__ _EndPtr,_locale_t _Locale);
func X_strtod_l(tls *TLS, __Str uintptr, __EndPtr uintptr, __Locale T_locale_t) (r float64) {
	r0, _, err := syscall.SyscallN(proc_strtod_l.Addr(), __Str, __EndPtr, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return float64(r0)
}

var proc_strtol_l = modcrt.NewProc(GoString(__ccgo_ts + 4321))

// __attribute__ ((__dllimport__)) long __attribute__((__cdecl__)) _strtol_l(const char * __restrict__ _Str,char ** __restrict__ _EndPtr,int _Radix,_locale_t _Locale);
func X_strtol_l(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_strtol_l.Addr(), __Str, __EndPtr, uintptr(__Radix), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_strtoul_l = modcrt.NewProc(GoString(__ccgo_ts + 4331))

// __attribute__ ((__dllimport__)) unsigned long __attribute__((__cdecl__)) _strtoul_l(const char * __restrict__ _Str,char ** __restrict__ _EndPtr,int _Radix,_locale_t _Locale);
func X_strtoul_l(tls *TLS, __Str uintptr, __EndPtr uintptr, __Radix int32, __Locale T_locale_t) (r uint32) {
	r0, _, err := syscall.SyscallN(proc_strtoul_l.Addr(), __Str, __EndPtr, uintptr(__Radix), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint32(r0)
}

var proc_ultoa = modcrt.NewProc(GoString(__ccgo_ts + 4342))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _ultoa(unsigned long _Value,char *_Dest,int _Radix);
func X_ultoa(tls *TLS, __Value uint32, __Dest uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_ultoa.Addr(), uintptr(__Value), __Dest, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_wctomb_l = modcrt.NewProc(GoString(__ccgo_ts + 4349))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _wctomb_l(char *_MbCh,wchar_t _WCh,_locale_t _Locale);
func X_wctomb_l(tls *TLS, __MbCh uintptr, __WCh Twchar_t, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_wctomb_l.Addr(), __MbCh, uintptr(__WCh), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_wcstombs_l = modcrt.NewProc(GoString(__ccgo_ts + 4359))

// __attribute__ ((__dllimport__)) size_t __attribute__((__cdecl__)) _wcstombs_l(char * __restrict__ _Dest,const wchar_t * __restrict__ _Source,size_t _MaxCount,_locale_t _Locale);
func X_wcstombs_l(tls *TLS, __Dest uintptr, __Source uintptr, __MaxCount Tsize_t, __Locale T_locale_t) (r Tsize_t) {
	r0, _, err := syscall.SyscallN(proc_wcstombs_l.Addr(), __Dest, __Source, uintptr(__MaxCount), __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return Tsize_t(r0)
}

var proc_recalloc = modcrt.NewProc(GoString(__ccgo_ts + 4371))

// __attribute__ ((__dllimport__)) void * __attribute__((__cdecl__)) _recalloc(void *_Memory,size_t _Count,size_t _Size);
func X_recalloc(tls *TLS, __Memory uintptr, __Count Tsize_t, __Size Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_recalloc.Addr(), __Memory, uintptr(__Count), uintptr(__Size))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_aligned_free = modcrt.NewProc(GoString(__ccgo_ts + 4381))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _aligned_free(void *_Memory);
func X_aligned_free(tls *TLS, __Memory uintptr) {
	_, _, err := syscall.SyscallN(proc_aligned_free.Addr(), __Memory)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_aligned_malloc = modcrt.NewProc(GoString(__ccgo_ts + 4395))

// __attribute__ ((__dllimport__)) void * __attribute__((__cdecl__)) _aligned_malloc(size_t _Size,size_t _Alignment);
func X_aligned_malloc(tls *TLS, __Size Tsize_t, __Alignment Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_aligned_malloc.Addr(), uintptr(__Size), uintptr(__Alignment))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_aligned_offset_malloc = modcrt.NewProc(GoString(__ccgo_ts + 4411))

// __attribute__ ((__dllimport__)) void * __attribute__((__cdecl__)) _aligned_offset_malloc(size_t _Size,size_t _Alignment,size_t _Offset);
func X_aligned_offset_malloc(tls *TLS, __Size Tsize_t, __Alignment Tsize_t, __Offset Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_aligned_offset_malloc.Addr(), uintptr(__Size), uintptr(__Alignment), uintptr(__Offset))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_aligned_realloc = modcrt.NewProc(GoString(__ccgo_ts + 4434))

// __attribute__ ((__dllimport__)) void * __attribute__((__cdecl__)) _aligned_realloc(void *_Memory,size_t _Size,size_t _Alignment);
func X_aligned_realloc(tls *TLS, __Memory uintptr, __Size Tsize_t, __Alignment Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_aligned_realloc.Addr(), __Memory, uintptr(__Size), uintptr(__Alignment))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_aligned_recalloc = modcrt.NewProc(GoString(__ccgo_ts + 4451))

// __attribute__ ((__dllimport__)) void * __attribute__((__cdecl__)) _aligned_recalloc(void *_Memory,size_t _Count,size_t _Size,size_t _Alignment);
func X_aligned_recalloc(tls *TLS, __Memory uintptr, __Count Tsize_t, __Size Tsize_t, __Alignment Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_aligned_recalloc.Addr(), __Memory, uintptr(__Count), uintptr(__Size), uintptr(__Alignment))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_aligned_offset_realloc = modcrt.NewProc(GoString(__ccgo_ts + 4469))

// __attribute__ ((__dllimport__)) void * __attribute__((__cdecl__)) _aligned_offset_realloc(void *_Memory,size_t _Size,size_t _Alignment,size_t _Offset);
func X_aligned_offset_realloc(tls *TLS, __Memory uintptr, __Size Tsize_t, __Alignment Tsize_t, __Offset Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_aligned_offset_realloc.Addr(), __Memory, uintptr(__Size), uintptr(__Alignment), uintptr(__Offset))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_aligned_offset_recalloc = modcrt.NewProc(GoString(__ccgo_ts + 4493))

// __attribute__ ((__dllimport__)) void * __attribute__((__cdecl__)) _aligned_offset_recalloc(void *_Memory,size_t _Count,size_t _Size,size_t _Alignment,size_t _Offset);
func X_aligned_offset_recalloc(tls *TLS, __Memory uintptr, __Count Tsize_t, __Size Tsize_t, __Alignment Tsize_t, __Offset Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_aligned_offset_recalloc.Addr(), __Memory, uintptr(__Count), uintptr(__Size), uintptr(__Alignment), uintptr(__Offset))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_putenv = modcrt.NewProc(GoString(__ccgo_ts + 4518))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _putenv(const char *_EnvString);
func X_putenv(tls *TLS, __EnvString uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_putenv.Addr(), __EnvString)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_fullpath = modcrt.NewProc(GoString(__ccgo_ts + 4526))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _fullpath(char *_FullPath,const char *_Path,size_t _SizeInBytes);
func X_fullpath(tls *TLS, __FullPath uintptr, __Path uintptr, __SizeInBytes Tsize_t) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_fullpath.Addr(), __FullPath, __Path, uintptr(__SizeInBytes))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_ecvt = modcrt.NewProc(GoString(__ccgo_ts + 4536))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _ecvt(double _Val,int _NumOfDigits,int *_PtDec,int *_PtSign);
func X_ecvt(tls *TLS, __Val float64, __NumOfDigits int32, __PtDec uintptr, __PtSign uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_ecvt.Addr(), uintptr(__Val), uintptr(__NumOfDigits), __PtDec, __PtSign)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_fcvt = modcrt.NewProc(GoString(__ccgo_ts + 4542))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _fcvt(double _Val,int _NumOfDec,int *_PtDec,int *_PtSign);
func X_fcvt(tls *TLS, __Val float64, __NumOfDec int32, __PtDec uintptr, __PtSign uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_fcvt.Addr(), uintptr(__Val), uintptr(__NumOfDec), __PtDec, __PtSign)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_gcvt = modcrt.NewProc(GoString(__ccgo_ts + 4548))

// __attribute__ ((__dllimport__)) char * __attribute__((__cdecl__)) _gcvt(double _Val,int _NumOfDigits,char *_DstBuf);
func X_gcvt(tls *TLS, __Val float64, __NumOfDigits int32, __DstBuf uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(proc_gcvt.Addr(), uintptr(__Val), uintptr(__NumOfDigits), __DstBuf)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proc_atodbl = modcrt.NewProc(GoString(__ccgo_ts + 4554))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _atodbl(_CRT_DOUBLE *_Result,char *_Str);
func X_atodbl(tls *TLS, __Result uintptr, __Str uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_atodbl.Addr(), __Result, __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_atoldbl = modcrt.NewProc(GoString(__ccgo_ts + 4562))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _atoldbl(_LDOUBLE *_Result,char *_Str);
func X_atoldbl(tls *TLS, __Result uintptr, __Str uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_atoldbl.Addr(), __Result, __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_atoflt = modcrt.NewProc(GoString(__ccgo_ts + 4571))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _atoflt(_CRT_FLOAT *_Result,char *_Str);
func X_atoflt(tls *TLS, __Result uintptr, __Str uintptr) (r int32) {
	r0, _, err := syscall.SyscallN(proc_atoflt.Addr(), __Result, __Str)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_atodbl_l = modcrt.NewProc(GoString(__ccgo_ts + 4579))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _atodbl_l(_CRT_DOUBLE *_Result,char *_Str,_locale_t _Locale);
func X_atodbl_l(tls *TLS, __Result uintptr, __Str uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_atodbl_l.Addr(), __Result, __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_atoldbl_l = modcrt.NewProc(GoString(__ccgo_ts + 4589))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _atoldbl_l(_LDOUBLE *_Result,char *_Str,_locale_t _Locale);
func X_atoldbl_l(tls *TLS, __Result uintptr, __Str uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_atoldbl_l.Addr(), __Result, __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_atoflt_l = modcrt.NewProc(GoString(__ccgo_ts + 4600))

// __attribute__ ((__dllimport__)) int __attribute__((__cdecl__)) _atoflt_l(_CRT_FLOAT *_Result,char *_Str,_locale_t _Locale);
func X_atoflt_l(tls *TLS, __Result uintptr, __Str uintptr, __Locale T_locale_t) (r int32) {
	r0, _, err := syscall.SyscallN(proc_atoflt_l.Addr(), __Result, __Str, __Locale)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int32(r0)
}

var proc_lrotl = modcrt.NewProc(GoString(__ccgo_ts + 4610))

// unsigned long __attribute__((__cdecl__)) _lrotl(unsigned long,int);
func X_lrotl(tls *TLS, _0 uint32, _1 int32) (r uint32) {
	r0, _, err := syscall.SyscallN(proc_lrotl.Addr(), uintptr(_0), uintptr(_1))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint32(r0)
}

var proc_lrotr = modcrt.NewProc(GoString(__ccgo_ts + 4617))

// unsigned long __attribute__((__cdecl__)) _lrotr(unsigned long,int);
func X_lrotr(tls *TLS, _0 uint32, _1 int32) (r uint32) {
	r0, _, err := syscall.SyscallN(proc_lrotr.Addr(), uintptr(_0), uintptr(_1))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint32(r0)
}

var proc_makepath = modcrt.NewProc(GoString(__ccgo_ts + 4624))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _makepath(char *_Path,const char *_Drive,const char *_Dir,const char *_Filename,const char *_Ext);
func X_makepath(tls *TLS, __Path uintptr, __Drive uintptr, __Dir uintptr, __Filename uintptr, __Ext uintptr) {
	_, _, err := syscall.SyscallN(proc_makepath.Addr(), __Path, __Drive, __Dir, __Filename, __Ext)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_onexit = modcrt.NewProc(GoString(__ccgo_ts + 4634))

// _onexit_t __attribute__((__cdecl__)) _onexit(_onexit_t _Func);
func X_onexit(tls *TLS, __Func T_onexit_t) (r T_onexit_t) {
	r0, _, err := syscall.SyscallN(proc_onexit.Addr(), __Func)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T_onexit_t(r0)
}

var proc_rotl64 = modcrt.NewProc(GoString(__ccgo_ts + 4642))

// unsigned long long __attribute__((__cdecl__)) _rotl64(unsigned long long _Val,int _Shift);
func X_rotl64(tls *TLS, __Val uint64, __Shift int32) (r uint64) {
	r0, _, err := syscall.SyscallN(proc_rotl64.Addr(), uintptr(__Val), uintptr(__Shift))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint64(r0)
}

var proc_rotr64 = modcrt.NewProc(GoString(__ccgo_ts + 4650))

// unsigned long long __attribute__((__cdecl__)) _rotr64(unsigned long long Value,int Shift);
func X_rotr64(tls *TLS, _Value uint64, _Shift int32) (r uint64) {
	r0, _, err := syscall.SyscallN(proc_rotr64.Addr(), uintptr(_Value), uintptr(_Shift))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint64(r0)
}

var proc_rotr = modcrt.NewProc(GoString(__ccgo_ts + 4658))

// unsigned int __attribute__((__cdecl__)) _rotr(unsigned int _Val,int _Shift);
func X_rotr(tls *TLS, __Val uint32, __Shift int32) (r uint32) {
	r0, _, err := syscall.SyscallN(proc_rotr.Addr(), uintptr(__Val), uintptr(__Shift))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint32(r0)
}

var proc_rotl = modcrt.NewProc(GoString(__ccgo_ts + 4664))

// unsigned int __attribute__((__cdecl__)) _rotl(unsigned int _Val,int _Shift);
func X_rotl(tls *TLS, __Val uint32, __Shift int32) (r uint32) {
	r0, _, err := syscall.SyscallN(proc_rotl.Addr(), uintptr(__Val), uintptr(__Shift))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uint32(r0)
}

var proc_searchenv = modcrt.NewProc(GoString(__ccgo_ts + 4670))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _searchenv(const char *_Filename,const char *_EnvVar,char *_ResultPath);
func X_searchenv(tls *TLS, __Filename uintptr, __EnvVar uintptr, __ResultPath uintptr) {
	_, _, err := syscall.SyscallN(proc_searchenv.Addr(), __Filename, __EnvVar, __ResultPath)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_splitpath = modcrt.NewProc(GoString(__ccgo_ts + 4681))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _splitpath(const char *_FullPath,char *_Drive,char *_Dir,char *_Filename,char *_Ext);
func X_splitpath(tls *TLS, __FullPath uintptr, __Drive uintptr, __Dir uintptr, __Filename uintptr, __Ext uintptr) {
	_, _, err := syscall.SyscallN(proc_splitpath.Addr(), __FullPath, __Drive, __Dir, __Filename, __Ext)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_swab = modcrt.NewProc(GoString(__ccgo_ts + 4692))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _swab(char *_Buf1,char *_Buf2,int _SizeInBytes);
func X_swab(tls *TLS, __Buf1 uintptr, __Buf2 uintptr, __SizeInBytes int32) {
	_, _, err := syscall.SyscallN(proc_swab.Addr(), __Buf1, __Buf2, uintptr(__SizeInBytes))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_beep = modcrt.NewProc(GoString(__ccgo_ts + 4698))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _beep(unsigned _Frequency,unsigned _Duration) __attribute__ ((__deprecated__));
func X_beep(tls *TLS, __Frequency uint32, __Duration uint32) {
	_, _, err := syscall.SyscallN(proc_beep.Addr(), uintptr(__Frequency), uintptr(__Duration))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_seterrormode = modcrt.NewProc(GoString(__ccgo_ts + 4704))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _seterrormode(int _Mode) __attribute__ ((__deprecated__));
func X_seterrormode(tls *TLS, __Mode int32) {
	_, _, err := syscall.SyscallN(proc_seterrormode.Addr(), uintptr(__Mode))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var proc_sleep = modcrt.NewProc(GoString(__ccgo_ts + 4718))

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _sleep(unsigned long _Duration) __attribute__ ((__deprecated__));
func X_sleep(tls *TLS, __Duration uint32) {
	_, _, err := syscall.SyscallN(proc_sleep.Addr(), uintptr(__Duration))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var procecvt = modcrt.NewProc(GoString(__ccgo_ts + 4725))

// char * __attribute__((__cdecl__)) ecvt(double _Val,int _NumOfDigits,int *_PtDec,int *_PtSign);
func Xecvt(tls *TLS, __Val float64, __NumOfDigits int32, __PtDec uintptr, __PtSign uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procecvt.Addr(), uintptr(__Val), uintptr(__NumOfDigits), __PtDec, __PtSign)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procfcvt = modcrt.NewProc(GoString(__ccgo_ts + 4730))

// char * __attribute__((__cdecl__)) fcvt(double _Val,int _NumOfDec,int *_PtDec,int *_PtSign);
func Xfcvt(tls *TLS, __Val float64, __NumOfDec int32, __PtDec uintptr, __PtSign uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procfcvt.Addr(), uintptr(__Val), uintptr(__NumOfDec), __PtDec, __PtSign)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procgcvt = modcrt.NewProc(GoString(__ccgo_ts + 4735))

// char * __attribute__((__cdecl__)) gcvt(double _Val,int _NumOfDigits,char *_DstBuf);
func Xgcvt(tls *TLS, __Val float64, __NumOfDigits int32, __DstBuf uintptr) (r uintptr) {
	r0, _, err := syscall.SyscallN(procgcvt.Addr(), uintptr(__Val), uintptr(__NumOfDigits), __DstBuf)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procitoa = modcrt.NewProc(GoString(__ccgo_ts + 4740))

// char * __attribute__((__cdecl__)) itoa(int _Val,char *_DstBuf,int _Radix);
func Xitoa(tls *TLS, __Val int32, __DstBuf uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(procitoa.Addr(), uintptr(__Val), __DstBuf, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procltoa = modcrt.NewProc(GoString(__ccgo_ts + 4745))

// char * __attribute__((__cdecl__)) ltoa(long _Val,char *_DstBuf,int _Radix);
func Xltoa(tls *TLS, __Val int32, __DstBuf uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(procltoa.Addr(), uintptr(__Val), __DstBuf, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var procswab = modcrt.NewProc(GoString(__ccgo_ts + 4750))

// void __attribute__((__cdecl__)) swab(char *_Buf1,char *_Buf2,int _SizeInBytes);
func Xswab(tls *TLS, __Buf1 uintptr, __Buf2 uintptr, __SizeInBytes int32) {
	_, _, err := syscall.SyscallN(procswab.Addr(), __Buf1, __Buf2, uintptr(__SizeInBytes))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
}

var procultoa = modcrt.NewProc(GoString(__ccgo_ts + 4755))

// char * __attribute__((__cdecl__)) ultoa(unsigned long _Val,char *_Dstbuf,int _Radix);
func Xultoa(tls *TLS, __Val uint32, __Dstbuf uintptr, __Radix int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(procultoa.Addr(), uintptr(__Val), __Dstbuf, uintptr(__Radix))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proconexit = modcrt.NewProc(GoString(__ccgo_ts + 4761))

// _onexit_t __attribute__((__cdecl__)) onexit( _onexit_t _Func);
func Xonexit(tls *TLS, __Func T_onexit_t) (r T_onexit_t) {
	r0, _, err := syscall.SyscallN(proconexit.Addr(), __Func)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return T_onexit_t(r0)
}

type Tlldiv_t = struct {
	Fquot int64
	Frem  int64
}

var procstrtoll = modcrt.NewProc(GoString(__ccgo_ts + 4768))

// long long __attribute__((__cdecl__)) strtoll(const char * __restrict__, char ** __restrict, int);
func Xstrtoll(tls *TLS, _0 uintptr, _1 uintptr, _2 int32) (r int64) {
	r0, _, err := syscall.SyscallN(procstrtoll.Addr(), _0, _1, uintptr(_2))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var procatoll = modcrt.NewProc(GoString(__ccgo_ts + 4776))

// long long __attribute__((__cdecl__)) atoll (const char *);
func Xatoll(tls *TLS, _0 uintptr) (r int64) {
	r0, _, err := syscall.SyscallN(procatoll.Addr(), _0)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var procwtoll = modcrt.NewProc(GoString(__ccgo_ts + 4782))

// long long __attribute__((__cdecl__)) wtoll (const wchar_t *);
func Xwtoll(tls *TLS, _0 uintptr) (r int64) {
	r0, _, err := syscall.SyscallN(procwtoll.Addr(), _0)
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return int64(r0)
}

var proclltoa = modcrt.NewProc(GoString(__ccgo_ts + 4788))

// char * __attribute__((__cdecl__)) lltoa (long long, char *, int);
func Xlltoa(tls *TLS, _0 int64, _1 uintptr, _2 int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proclltoa.Addr(), uintptr(_0), _1, uintptr(_2))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proculltoa = modcrt.NewProc(GoString(__ccgo_ts + 4794))

// char * __attribute__((__cdecl__)) ulltoa (unsigned long long , char *, int);
func Xulltoa(tls *TLS, _0 uint64, _1 uintptr, _2 int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proculltoa.Addr(), uintptr(_0), _1, uintptr(_2))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proclltow = modcrt.NewProc(GoString(__ccgo_ts + 4801))

// wchar_t * __attribute__((__cdecl__)) lltow (long long, wchar_t *, int);
func Xlltow(tls *TLS, _0 int64, _1 uintptr, _2 int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proclltow.Addr(), uintptr(_0), _1, uintptr(_2))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

var proculltow = modcrt.NewProc(GoString(__ccgo_ts + 4807))

// wchar_t * __attribute__((__cdecl__)) ulltow (unsigned long long, wchar_t *, int);
func Xulltow(tls *TLS, _0 uint64, _1 uintptr, _2 int32) (r uintptr) {
	r0, _, err := syscall.SyscallN(proculltow.Addr(), uintptr(_0), _1, uintptr(_2))
	if err != 0 {
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(err)
	}
	return uintptr(r0)
}

type T_HEAPINFO = struct {
	F_pentry  uintptr
	F_size    Tsize_t
	F_useflag int32
}

type T_heapinfo = T_HEAPINFO

var __ccgo_ts = (*reflect.StringHeader)(unsafe.Pointer(&__ccgo_ts1)).Data

var __ccgo_ts1 = "__iob_func\x00iswalpha\x00_iswalpha_l\x00iswupper\x00_iswupper_l\x00iswlower\x00_iswlower_l\x00iswdigit\x00_iswdigit_l\x00iswxdigit\x00_iswxdigit_l\x00iswspace\x00_iswspace_l\x00iswpunct\x00_iswpunct_l\x00iswalnum\x00_iswalnum_l\x00iswprint\x00_iswprint_l\x00iswgraph\x00_iswgraph_l\x00iswcntrl\x00_iswcntrl_l\x00iswascii\x00isleadbyte\x00_isleadbyte_l\x00towupper\x00_towupper_l\x00towlower\x00_towlower_l\x00iswctype\x00is_wctype\x00iswblank\x00_wgetcwd\x00_wgetdcwd\x00_wchdir\x00_wmkdir\x00_wrmdir\x00_waccess\x00_wchmod\x00_wcreat\x00_wfindfirst32\x00_wfindnext32\x00_wrename\x00_wmktemp\x00_wfindfirst32i64\x00_wfindfirst64i32\x00_wfindfirst64\x00_wfindnext32i64\x00_wfindnext64i32\x00_wfindnext64\x00_wsopen_s\x00_wsopen\x00_wsetlocale\x00_wexecl\x00_wexecle\x00_wexeclp\x00_wexeclpe\x00_wexecv\x00_wexecve\x00_wexecvp\x00_wexecvpe\x00_wspawnl\x00_wspawnle\x00_wspawnlp\x00_wspawnlpe\x00_wspawnv\x00_wspawnve\x00_wspawnvp\x00_wspawnvpe\x00_wsystem\x00_wstat32\x00_wstat32i64\x00_wstat64i32\x00_wstat64\x00_cgetws\x00_getwch\x00_getwche\x00_putwch\x00_ungetwch\x00_cputws\x00_cwprintf\x00_cwscanf\x00_cwscanf_l\x00_vcwprintf\x00_cwprintf_p\x00_vcwprintf_p\x00_cwprintf_l\x00_vcwprintf_l\x00_cwprintf_p_l\x00_vcwprintf_p_l\x00__mingw_swscanf\x00__mingw_wscanf\x00__mingw_vwscanf\x00__mingw_fwscanf\x00__mingw_fwprintf\x00__mingw_wprintf\x00__mingw_vwprintf\x00__mingw_snwprintf\x00__mingw_swprintf\x00__mingw_vswprintf\x00__ms_swscanf\x00__ms_wscanf\x00__ms_fwscanf\x00__ms_fwprintf\x00__ms_wprintf\x00__ms_vfwprintf\x00__ms_vwprintf\x00__ms_swprintf\x00__ms_vswprintf\x00_wfsopen\x00fgetwc\x00_fgetwchar\x00fputwc\x00_fputwchar\x00getwc\x00getwchar\x00putwc\x00putwchar\x00ungetwc\x00fgetws\x00fputws\x00_getws\x00_putws\x00_scwprintf\x00_swprintf_l\x00_swprintf_c\x00_vswprintf_c\x00_fwprintf_p\x00_wprintf_p\x00_vfwprintf_p\x00_vwprintf_p\x00_swprintf_p\x00_vswprintf_p\x00_scwprintf_p\x00_vscwprintf_p\x00_wprintf_l\x00_wprintf_p_l\x00_vwprintf_l\x00_vwprintf_p_l\x00_fwprintf_l\x00_fwprintf_p_l\x00_vfwprintf_l\x00_vfwprintf_p_l\x00_swprintf_c_l\x00_swprintf_p_l\x00_vswprintf_c_l\x00_vswprintf_p_l\x00_scwprintf_l\x00_scwprintf_p_l\x00_vscwprintf_p_l\x00_snwprintf_l\x00_vsnwprintf_l\x00_swprintf\x00_vswprintf\x00__swprintf_l\x00_vswprintf_l\x00__vswprintf_l\x00_wtempnam\x00_vscwprintf\x00_vscwprintf_l\x00_fwscanf_l\x00_swscanf_l\x00_snwscanf\x00_snwscanf_l\x00_wscanf_l\x00_wfdopen\x00_wfopen\x00_wfreopen\x00_wperror\x00_wpopen\x00_wremove\x00_wtmpnam\x00_itow\x00_ltow\x00_ultow\x00_wcstod_l\x00__mingw_wcstod\x00__mingw_wcstof\x00__mingw_wcstold\x00wcstold\x00wcstol\x00_wcstol_l\x00wcstoul\x00_wcstoul_l\x00_wtof\x00_wtof_l\x00_wtoi_l\x00_wtol\x00_wtol_l\x00_i64tow\x00_ui64tow\x00_wtoi64\x00_wtoi64_l\x00_wcstoi64\x00_wcstoi64_l\x00_wcstoui64\x00_wcstoui64_l\x00_wfullpath\x00_wmakepath\x00_wsearchenv\x00_wsplitpath\x00_wcsdup\x00wcscat\x00wcscspn\x00wcsnlen\x00wcsncat\x00wcsncpy\x00_wcsncpy_l\x00wcspbrk\x00wcsrchr\x00wcsspn\x00wcsstr\x00wcstok\x00_wcserror\x00__wcserror\x00_wcsicmp_l\x00_wcsnicmp_l\x00_wcsnset\x00_wcsrev\x00_wcsset\x00_wcslwr\x00_wcslwr_l\x00_wcsupr\x00_wcsupr_l\x00wcsxfrm\x00_wcsxfrm_l\x00wcscoll\x00_wcscoll_l\x00_wcsicoll\x00_wcsicoll_l\x00_wcsncoll\x00_wcsncoll_l\x00_wcsnicoll\x00_wcsnicoll_l\x00wcsdup\x00wcsnicmp\x00wcsnset\x00wcsrev\x00wcsset\x00wcslwr\x00wcsupr\x00wcsicoll\x00_wasctime\x00_wasctime_s\x00_wctime32\x00_wctime32_s\x00wcsftime\x00_wcsftime_l\x00_wstrdate\x00_wstrdate_s\x00_wstrtime\x00_wstrtime_s\x00_wctime64\x00_wctime64_s\x00_wctime\x00_wctime_s\x00btowc\x00mbrlen\x00mbrtowc\x00mbsrtowcs\x00wctob\x00wmemset\x00wmemchr\x00wmemcmp\x00wmemcpy\x00wmempcpy\x00wmemmove\x00fwide\x00mbsinit\x00wcstoll\x00wcstoull\x00__mingw_str_wide_utf8\x00__mingw_str_utf8_wide\x00__mingw_str_free\x00_memccpy\x00_memicmp\x00_memicmp_l\x00memcpy_s\x00mempcpy\x00memccpy\x00memicmp\x00_strset\x00_strset_l\x00strnlen\x00_strcmpi\x00_stricmp_l\x00strcoll\x00_strcoll_l\x00_stricoll\x00_stricoll_l\x00_strncoll\x00_strncoll_l\x00_strnicoll\x00_strnicoll_l\x00_strerror\x00_strlwr\x00strlwr_l\x00strncat\x00_strnicmp_l\x00_strnset\x00_strnset_l\x00_strrev\x00strtok\x00strtok_r\x00_strupr\x00_strupr_l\x00strxfrm\x00_strxfrm_l\x00strcmpi\x00stricmp\x00strlwr\x00strnicmp\x00strncasecmp\x00strnset\x00strrev\x00strset\x00strupr\x00_get_daylight\x00_get_dstbias\x00_get_timezone\x00_get_tzname\x00asctime\x00asctime_s\x00_ctime32\x00_ctime32_s\x00clock\x00_difftime32\x00_gmtime32\x00_gmtime32_s\x00_localtime32\x00_localtime32_s\x00_strftime_l\x00_strdate\x00_strdate_s\x00_strtime\x00_strtime_s\x00_time32\x00_mktime32\x00_mkgmtime32\x00_tzset\x00_difftime64\x00_ctime64\x00_ctime64_s\x00_gmtime64_s\x00_localtime64_s\x00_mkgmtime64\x00_getsystime\x00_setsystime\x00mingw_gettimeofday\x00___mb_cur_max_func\x00_set_purecall_handler\x00_get_purecall_handler\x00_set_invalid_parameter_handler\x00_get_invalid_parameter_handler\x00_set_errno\x00_get_errno\x00__doserrno\x00_set_doserrno\x00_get_doserrno\x00__p___argv\x00__p__fmode\x00_get_pgmptr\x00_get_wpgmptr\x00_set_fmode\x00_get_fmode\x00_get_osplatform\x00_get_osver\x00_get_winver\x00_get_winmajor\x00_get_winminor\x00_Exit\x00_abs64\x00_atof_l\x00_atoi_l\x00_atol_l\x00_byteswap_ushort\x00_itoa\x00_i64toa\x00_ui64toa\x00_atoi64\x00_atoi64_l\x00_strtoi64\x00_strtoi64_l\x00_strtoui64\x00_strtoui64_l\x00_ltoa\x00_mblen_l\x00_mbstrlen\x00_mbstrlen_l\x00_mbstrnlen\x00_mbstrnlen_l\x00_mbtowc_l\x00_mbstowcs_l\x00mkstemp\x00_set_error_mode\x00srand\x00strtold\x00__strtod\x00__mingw_strtof\x00__mingw_strtold\x00_strtod_l\x00_strtol_l\x00_strtoul_l\x00_ultoa\x00_wctomb_l\x00_wcstombs_l\x00_recalloc\x00_aligned_free\x00_aligned_malloc\x00_aligned_offset_malloc\x00_aligned_realloc\x00_aligned_recalloc\x00_aligned_offset_realloc\x00_aligned_offset_recalloc\x00_putenv\x00_fullpath\x00_ecvt\x00_fcvt\x00_gcvt\x00_atodbl\x00_atoldbl\x00_atoflt\x00_atodbl_l\x00_atoldbl_l\x00_atoflt_l\x00_lrotl\x00_lrotr\x00_makepath\x00_onexit\x00_rotl64\x00_rotr64\x00_rotr\x00_rotl\x00_searchenv\x00_splitpath\x00_swab\x00_beep\x00_seterrormode\x00_sleep\x00ecvt\x00fcvt\x00gcvt\x00itoa\x00ltoa\x00swab\x00ultoa\x00onexit\x00strtoll\x00atoll\x00wtoll\x00lltoa\x00ulltoa\x00lltow\x00ulltow\x00"
