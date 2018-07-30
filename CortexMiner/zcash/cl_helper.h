
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>


void debug(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
}

void warn(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

void fatal(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    exit(1);
}

uint64_t parse_num(char *str)
{
    char *endptr;
    uint64_t n;
    n = strtoul(str, &endptr, 0);
    if (endptr == str || *endptr)
        fatal("'%s' is not a valid number\n", str);
    return n;
}

void hexdump(uint8_t *a, uint32_t a_len)
{
    for (uint32_t i = 0; i < a_len; i++)
        fprintf(stderr, "%02x", a[i]);
}

char *s_hexdump(const void *_a, uint32_t a_len)
{
    const uint8_t *a = (const uint8_t*)_a;
    static char buf[4096];
    uint32_t i;
    for (i = 0; i < a_len && i + 2 < sizeof(buf); i++)
        sprintf(buf + i * 2, "%02x", a[i]);
    buf[i * 2] = 0;
    return buf;
}

uint8_t hex2val(const char *base, size_t off)
{
    const char c = base[off];
    if (c >= '0' && c <= '9')
        return c - '0';
    else if (c >= 'a' && c <= 'f')
        return 10 + c - 'a';
    else if (c >= 'A' && c <= 'F')
        return 10 + c - 'A';
    fatal("Invalid hex char at offset %zd: ...%c...\n", off, c);
    return 0;
}

uint64_t now(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 * 1000 + tv.tv_usec;
}

void show_time(uint64_t t0)
{
    uint64_t t1;
    t1 = now();
    fprintf(stderr, "Elapsed time: %.1f msec\n", (t1 - t0) / 1e3);
}

// set file I/O block mode
void set_blocking_mode(int fd, int block)
{
    int f;
    
    // get file state descriptor 
    if (-1 == (f = fcntl(fd, F_GETFL)))
        fatal("fcntl F_GETFL: %s\n", strerror(errno));

    // set file lock status
    if (-1 == fcntl(fd, F_SETFL, block ? (f & ~O_NONBLOCK) : (f | O_NONBLOCK)))
        fatal("fcntl F_SETFL: %s\n", strerror(errno));
}


// "/dev/urandom" is a pseudo random stream provided by unix system
// as a device

void randomize(void *p, ssize_t l)
{
    const char *fname = "/dev/urandom";
    int fd;
    ssize_t ret;
    if (-1 == (fd = open(fname, O_RDONLY)))
        fatal("open %s: %s\n", fname, strerror(errno));
    if (-1 == (ret = read(fd, p, l)))
        fatal("read %s: %s\n", fname, strerror(errno));
    if (ret != l)
        fatal("%s: short read %d bytes out of %d\n", fname, ret, l);
    if (-1 == close(fd))
        fatal("close %s: %s\n", fname, strerror(errno));
}

void * memrchr(const void* s, int c, size_t n)
{
    const unsigned char *cp;

    if (n != 0)
    {
        cp = (unsigned char *)s + n;
        do
        {
            if (*(--cp) == (unsigned char)c)
                return (void *)cp;
        } while (--n != 0);
    }
    return (void *)0;
}