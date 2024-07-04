# Copyright 2024 The Libc Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

.PHONY:	all build_all_targets check clean download edit editor generate dev membrk-test test work xtest short-test xlibc libc-test surface

SHELL=/bin/bash -o pipefail	

DIR = /tmp/libc
TAR = musl-7ada6dde6f9dc6a2836c3d92c2f762d35fd229e0.tar.gz
URL = https://git.musl-libc.org/cgit/musl/snapshot/$(TAR)
UCRT_386 = libc_windows_386.go
UCRT_AMD64 = libc_windows_amd64.go
UCRT_ARM64 = libc_windows_arm64.go

all: editor
	golint 2>&1
	staticcheck 2>&1

build_all_targets:
	./build_all_targets.sh
	echo done

clean:
	rm -f log-* cpu.test mem.test *.out
	git clean -fd
	find testdata/nsz.repo.hu/ -name \*.go -delete
	make -C testdata/nsz.repo.hu/libc-test/ cleanall
	go clean

check:
	staticcheck 2>&1 | grep -v U1000

download:
	@if [ ! -f $(TAR) ]; then wget $(URL) ; fi

edit:
	@touch log
	@if [ -f "Session.vim" ]; then novim -S & else novim -p Makefile all_windows_test.go generator.go libc.go libc_windows*.go & fi

editor:
	gofmt -l -s -w *.go 2>&1 | tee log-editor
	go test -c -o /dev/null 2>&1 | tee -a log-editor
	go install -v  2>&1 | tee -a log-editor
	go build -o /dev/null generator*.go

ucrt:
	make ucrt_amd64 ucrt_arm64 ucrt_386
	go build -v ./... 2>&1 | tee -a log-generate
	GOOS=darwin go build -v ./... 2>&1 | tee -a log-generate
	git status

ucrt_amd64:
	echo -n > log-generate
	( ccgo -v4 \
		--cpp=$(shell which x86_64-w64-mingw32-gcc) \
		--goos=windows \
		--goarch=amd64 \
		--package-name libc \
		--prefix-external=X \
		--prefix-field=F \
		--prefix-static-internal=_ \
		--prefix-static-none=_ \
		--prefix-tagged-struct=T \
		--prefix-tagged-union=T \
		--prefix-typename=T \
		--winapi-test panic \
		--winapi=ctype.h \
		--winapi=float.h \
		--winapi=io.h \
		--winapi=libucrt.c \
		--winapi=locale.h \
		--winapi=malloc.h \
		--winapi=math.h \
		--winapi=process.h \
		--winapi=types.h \
		--winapi=stat.h \
		--winapi=stdio.h \
		--winapi=stdlib.h \
		--winapi=string.h \
		--winapi=time.h \
		--winapi=timeb.h \
		--winapi=wchar.h \
		--winapi=winbase.h \
		-build-lines=" " \
		-eval-all-macros \
		-hide __acrt_iob_func \
		-hide __create_locale \
		-hide __free_locale \
		-hide __get_current_locale \
		-hide __iob_func \
		-hide __lock_fhandle \
		-hide __sep__ \
		-hide __updatetlocinfo \
		-hide __updatetmbcinfo \
		-hide _beginthread \
		-hide _beginthreadex \
		-hide _endthreadex \
		-hide _errno \
		-hide _filbuf \
		-hide _flsbuf \
		-hide _get_amblksiz \
		-hide _get_osplatform \
		-hide _get_osver \
		-hide _get_output_format \
		-hide _get_sbh_threshold \
		-hide _get_winmajor \
		-hide _get_winminor \
		-hide _get_winver \
		-hide _heapadd \
		-hide _heapset \
		-hide _heapused \
		-hide _matherr \
		-hide _onexit \
		-hide _set_amblksiz \
		-hide _set_malloc_crt_max_wait \
		-hide _set_output_format \
		-hide _set_sbh_threshold \
		-hide _strcmpi \
		-hide _strnset_l \
		-hide _strset_l \
		-hide _unlock_fhandle \
		-hide _wcsncpy_l \
		-hide _wctime \
		-hide _wctime_s \
		-hide _wgetdcwd_nolock \
		-hide _wgetenv \
		-hide _wputenv \
		-hide access \
		-hide at_quick_exit \
		-hide atexit \
		-hide chdir \
		-hide exit \
		-hide lldiv \
		-hide qsort \
		-hide chmod \
		-hide chsize \
		-hide close \
		-hide creat \
		-hide cwait \
		-hide dup \
		-hide dup2 \
		-hide eof \
		-hide execv \
		-hide execve \
		-hide execvp \
		-hide execvpe \
		-hide fcloseall \
		-hide fdopen \
		-hide fgetchar \
		-hide fgetpos64 \
		-hide filelength \
		-hide fileno \
		-hide flushall \
		-hide fopen64 \
		-hide fpreset \
		-hide fputchar \
		-hide fsetpos64 \
		-hide ftime \
		-hide fwide \
		-hide getcwd \
		-hide getpid \
		-hide getw \
		-hide isatty \
		-hide itoa \
		-hide lltoa \
		-hide lltow \
		-hide locking \
		-hide lseek \
		-hide lseek64 \
		-hide ltoa \
		-hide memccpy \
		-hide memicmp \
		-hide mempcpy \
		-hide mkdir \
		-hide mkstemp \
		-hide mktemp \
		-hide onexit \
		-hide putenv \
		-hide putw \
		-hide read \
		-hide rmdir \
		-hide rmtmp \
		-hide setmode \
		-hide spawnv \
		-hide spawnve \
		-hide spawnvp \
		-hide spawnvpe \
		-hide strcasecmp \
		-hide strcmpi \
		-hide strdup \
		-hide stricmp \
		-hide strlwr \
		-hide strlwr_l \
		-hide strncasecmp \
		-hide strnicmp \
		-hide strnset \
		-hide strrev \
		-hide strset \
		-hide strtok_r \
		-hide strupr \
		-hide swab \
		-hide tell \
		-hide tempnam \
		-hide tzset \
		-hide ulltoa \
		-hide ulltow \
		-hide ultoa \
		-hide umask \
		-hide unlink \
		-hide wcsdup \
		-hide wcsicmp \
		-hide wcsicoll \
		-hide wcslwr \
		-hide wcsnicmp \
		-hide wcsnset \
		-hide wcsrev \
		-hide wcsset \
		-hide wcsupr \
		-hide wmemchr \
		-hide wmemcmp \
		-hide wmemcpy \
		-hide wmemmove \
		-hide wmempcpy \
		-hide wmemset \
		-hide write \
		-hide wtoll \
		-ignore-link-errors \
		-import syscall \
		-keep-strings \
		-o $(UCRT_AMD64) \
		libucrt.c \
		|| true ) 2>&1 | tee -a log-generate
	sed -i '/"modernc.org\/libc"/d' $(UCRT_AMD64)
	sed -i 's/\<libc\>\.//g' $(UCRT_AMD64)
	GOOS=windows GOARCH=amd64 go build -v ./... 2>&1 | tee -a log-generate

ucrt_arm64:
	echo -n > log-generate
	( ccgo -v4 \
		--cpp=$(shell which x86_64-w64-mingw32-gcc) \
		--goos=windows \
		--goarch=amd64 \
		--package-name libc \
		--prefix-external=X \
		--prefix-field=F \
		--prefix-static-internal=_ \
		--prefix-static-none=_ \
		--prefix-tagged-struct=T \
		--prefix-tagged-union=T \
		--prefix-typename=T \
		--winapi-test panic \
		--winapi=ctype.h \
		--winapi=float.h \
		--winapi=io.h \
		--winapi=libucrt.c \
		--winapi=locale.h \
		--winapi=malloc.h \
		--winapi=math.h \
		--winapi=process.h \
		--winapi=types.h \
		--winapi=stat.h \
		--winapi=stdio.h \
		--winapi=stdlib.h \
		--winapi=string.h \
		--winapi=time.h \
		--winapi=timeb.h \
		--winapi=wchar.h \
		--winapi=winbase.h \
		-build-lines=" " \
		-eval-all-macros \
		-hide __acrt_iob_func \
		-hide _errno \
		-hide _wgetenv \
		-hide _wputenv \
		-hide exit \
		-hide lldiv \
		-hide qsort \
		-hide __sep__ \
		-hide __create_locale \
		-hide __free_locale \
		-hide __get_current_locale \
		-hide __iob_func \
		-hide __lock_fhandle \
		-hide __updatetlocinfo \
		-hide __updatetmbcinfo \
		-hide _beginthread \
		-hide _beginthreadex \
		-hide _endthreadex \
		-hide _filbuf \
		-hide _flsbuf \
		-hide _get_amblksiz \
		-hide _get_osplatform \
		-hide _get_osver \
		-hide _get_output_format \
		-hide _get_sbh_threshold \
		-hide _get_winmajor \
		-hide _get_winminor \
		-hide _get_winver \
		-hide _heapadd \
		-hide _heapset \
		-hide _heapused \
		-hide _matherr \
		-hide _onexit \
		-hide _set_amblksiz \
		-hide _set_malloc_crt_max_wait \
		-hide _set_output_format \
		-hide _set_sbh_threshold \
		-hide _strcmpi \
		-hide _strnset_l \
		-hide _strset_l \
		-hide _unlock_fhandle \
		-hide _wcsncpy_l \
		-hide _wctime \
		-hide _wctime_s \
		-hide _wgetdcwd_nolock \
		-hide access \
		-hide at_quick_exit \
		-hide atexit \
		-hide chdir \
		-hide chmod \
		-hide chsize \
		-hide close \
		-hide creat \
		-hide cwait \
		-hide dup \
		-hide dup2 \
		-hide eof \
		-hide execv \
		-hide execve \
		-hide execvp \
		-hide execvpe \
		-hide fcloseall \
		-hide fdopen \
		-hide fgetchar \
		-hide fgetpos64 \
		-hide filelength \
		-hide fileno \
		-hide flushall \
		-hide fopen64 \
		-hide fpreset \
		-hide fputchar \
		-hide fsetpos64 \
		-hide ftime \
		-hide fwide \
		-hide getcwd \
		-hide getpid \
		-hide getw \
		-hide isatty \
		-hide itoa \
		-hide lltoa \
		-hide lltow \
		-hide locking \
		-hide lseek \
		-hide lseek64 \
		-hide ltoa \
		-hide memccpy \
		-hide memicmp \
		-hide mempcpy \
		-hide mkdir \
		-hide mkstemp \
		-hide mktemp \
		-hide onexit \
		-hide putenv \
		-hide putw \
		-hide read \
		-hide rmdir \
		-hide rmtmp \
		-hide setmode \
		-hide spawnv \
		-hide spawnve \
		-hide spawnvp \
		-hide spawnvpe \
		-hide strcasecmp \
		-hide strcmpi \
		-hide strdup \
		-hide stricmp \
		-hide strlwr \
		-hide strlwr_l \
		-hide strncasecmp \
		-hide strnicmp \
		-hide strnset \
		-hide strrev \
		-hide strset \
		-hide strtok_r \
		-hide strupr \
		-hide swab \
		-hide tell \
		-hide tempnam \
		-hide tzset \
		-hide ulltoa \
		-hide ulltow \
		-hide ultoa \
		-hide umask \
		-hide unlink \
		-hide wcsdup \
		-hide wcsicmp \
		-hide wcsicoll \
		-hide wcslwr \
		-hide wcsnicmp \
		-hide wcsnset \
		-hide wcsrev \
		-hide wcsset \
		-hide wcsupr \
		-hide wmemchr \
		-hide wmemcmp \
		-hide wmemcpy \
		-hide wmemmove \
		-hide wmempcpy \
		-hide wmemset \
		-hide write \
		-hide wtoll \
		-ignore-link-errors \
		-import syscall \
		-keep-strings \
		-o $(UCRT_ARM64) \
		libucrt.c \
		|| true ) 2>&1 | tee -a log-generate
	sed -i '/"modernc.org\/libc"/d' $(UCRT_ARM64)
	sed -i 's/\<libc\>\.//g' $(UCRT_ARM64)
	GOOS=windows GOARCH=arm64 go build -v ./... 2>&1 | tee -a log-generate

ucrt_386:
	echo -n > log-generate
	( ccgo -v4 \
		--cpp=$(shell which i686-w64-mingw32-gcc) \
		--goos=windows \
		--goarch=386 \
		--package-name libc \
		--prefix-external=X \
		--prefix-field=F \
		--prefix-static-internal=_ \
		--prefix-static-none=_ \
		--prefix-tagged-struct=T \
		--prefix-tagged-union=T \
		--prefix-typename=T \
		--winapi-test panic \
		--winapi=ctype.h \
		--winapi=float.h \
		--winapi=io.h \
		--winapi=libucrt.c \
		--winapi=locale.h \
		--winapi=malloc.h \
		--winapi=math.h \
		--winapi=process.h \
		--winapi=types.h \
		--winapi=stat.h \
		--winapi=stdio.h \
		--winapi=stdlib.h \
		--winapi=string.h \
		--winapi=time.h \
		--winapi=timeb.h \
		--winapi=wchar.h \
		--winapi=winbase.h \
		-build-lines=" " \
		-eval-all-macros \
		-hide __acrt_iob_func \
		-hide _errno \
		-hide _wgetenv \
		-hide _wputenv \
		-hide exit \
		-hide lldiv \
		-hide qsort \
		-hide __sep__ \
		-hide __create_locale \
		-hide __free_locale \
		-hide __get_current_locale \
		-hide __lock_fhandle \
		-hide __updatetlocinfo \
		-hide __updatetmbcinfo \
		-hide _beginthread \
		-hide _beginthreadex \
		-hide _endthreadex \
		-hide _filbuf \
		-hide _flsbuf \
		-hide _get_amblksiz \
		-hide _get_osplatform \
		-hide _get_osver \
		-hide _get_output_format \
		-hide _get_sbh_threshold \
		-hide _get_winmajor \
		-hide _get_winminor \
		-hide _get_winver \
		-hide _heapadd \
		-hide _heapset \
		-hide _heapused \
		-hide _matherr \
		-hide _onexit \
		-hide _set_amblksiz \
		-hide _set_malloc_crt_max_wait \
		-hide _set_output_format \
		-hide _set_sbh_threshold \
		-hide _strcmpi \
		-hide _strnset_l \
		-hide _strset_l \
		-hide _unlock_fhandle \
		-hide _wcsncpy_l \
		-hide _wctime \
		-hide _wctime_s \
		-hide _wgetdcwd_nolock \
		-hide access \
		-hide at_quick_exit \
		-hide atexit \
		-hide chdir \
		-hide chmod \
		-hide chsize \
		-hide close \
		-hide creat \
		-hide cwait \
		-hide dup \
		-hide dup2 \
		-hide eof \
		-hide execv \
		-hide execve \
		-hide execvp \
		-hide execvpe \
		-hide fcloseall \
		-hide fdopen \
		-hide fgetchar \
		-hide fgetpos64 \
		-hide filelength \
		-hide fileno \
		-hide flushall \
		-hide fopen64 \
		-hide fpreset \
		-hide fputchar \
		-hide fsetpos64 \
		-hide ftime \
		-hide fwide \
		-hide getcwd \
		-hide getpid \
		-hide getw \
		-hide isatty \
		-hide itoa \
		-hide lltoa \
		-hide lltow \
		-hide locking \
		-hide lseek \
		-hide lseek64 \
		-hide ltoa \
		-hide memccpy \
		-hide memicmp \
		-hide mempcpy \
		-hide mkdir \
		-hide mkstemp \
		-hide mktemp \
		-hide onexit \
		-hide putenv \
		-hide putw \
		-hide read \
		-hide rmdir \
		-hide rmtmp \
		-hide setmode \
		-hide spawnv \
		-hide spawnve \
		-hide spawnvp \
		-hide spawnvpe \
		-hide strcasecmp \
		-hide strcmpi \
		-hide strdup \
		-hide stricmp \
		-hide strlwr \
		-hide strlwr_l \
		-hide strncasecmp \
		-hide strnicmp \
		-hide strnset \
		-hide strrev \
		-hide strset \
		-hide strtok_r \
		-hide strupr \
		-hide swab \
		-hide tell \
		-hide tempnam \
		-hide tzset \
		-hide ulltoa \
		-hide ulltow \
		-hide ultoa \
		-hide umask \
		-hide unlink \
		-hide wcsdup \
		-hide wcsicmp \
		-hide wcsicoll \
		-hide wcslwr \
		-hide wcsnicmp \
		-hide wcsnset \
		-hide wcsrev \
		-hide wcsset \
		-hide wcsupr \
		-hide wmemchr \
		-hide wmemcmp \
		-hide wmemcpy \
		-hide wmemmove \
		-hide wmempcpy \
		-hide wmemset \
		-hide write \
		-hide wtoll \
		-ignore-link-errors \
		-import syscall \
		-keep-strings \
		-o $(UCRT_386) \
		libucrt.c \
		|| true ) 2>&1 | tee -a log-generate
	sed -i '/"modernc.org\/libc"/d' $(UCRT_386)
	sed -i 's/\<libc\>\.//g' $(UCRT_386)
	GOOS=windows GOARCH=386 go build -v ./... 2>&1 | tee -a log-generate

generate: download
	mkdir -p $(DIR) || true
	rm -rf $(DIR)/*
	GO_GENERATE_DIR=$(DIR) go run generator*.go 2>&1 | tee log-generate
	go build -v
	# go install github.com/mdempsky/unconvert@latest
	go build -v 2>&1 | tee -a log-generate
	go test -v -short -count=1 ./... | tee -a log-generate
	git status | tee -a log-generate
	grep 'TRC\|TODO\|ERRORF\|FAIL' log-generate || true

dev: download
	mkdir -p $(DIR) || true
	rm -rf $(DIR)/*
	echo -n > /tmp/ccgo.log
	GO_GENERATE_DIR=$(DIR) GO_GENERATE_DEV=1 go run -tags=ccgo.dmesg,ccgo.assert generator*.go 2>&1 | tee log-generate
	go build -v | tee -a log-generate
	go test -v -short -count=1 ./... | tee -a log-generate
	git status | tee -a log-generate
	grep 'TRC\|TODO\|ERRORF\|FAIL' log-generate || true
	grep 'TRC\|TODO\|ERRORF\|FAIL' /tmp/ccgo.log || true

membrk-test:
	echo -n > /tmp/ccgo.log
	touch log-test
	cp log-test log-test0
	go test -v -timeout 24h -count=1 -tags=libc.membrk 2>&1 | tee log-test
	grep -a 'TRC\|TODO\|ERRORF\|FAIL' log-test || true 2>&1 | tee -a log-test

test:
	echo -n > /tmp/ccgo.log
	touch log-test
	cp log-test log-test0
	go test -v -timeout 24h -count=1 2>&1 | tee log-test

short-test:
	echo -n > /tmp/ccgo.log
	touch log-test
	cp log-test log-test0
	go test -v -timeout 24h -count=1 -short 2>&1 | tee log-test
	grep -a 'TRC\|TODO\|ERRORF\|FAIL' log-test || true 2>&1 | tee -a log-test

xlibc:
	echo -n > /tmp/ccgo.log
	touch log-test
	cp log-test log-test0
	go test -v -timeout 24h -count=1 -tags=ccgo.dmesg,ccgo.assert 2>&1 -run TestLibc | tee log-test
	grep -a 'TRC\|TODO\|ERRORF\|FAIL' log-test || true 2>&1 | tee -a log-test

xpthread:
	echo -n > /tmp/ccgo.log
	touch log-test
	cp log-test log-test0
	go test -v -timeout 24h -count=1 2>&1 -run TestLibc -re pthread | tee log-test
	grep -a 'TRC\|TODO\|ERRORF\|FAIL' log-test || true 2>&1 | tee -a log-test

libc-test:
	echo -n > /tmp/ccgo.log
	touch log-test
	cp log-test log-test0
	go test -v -timeout 24h -count=1 2>&1 -run TestLibc | tee log-test
	# grep -a 'TRC\|TODO\|ERRORF\|FAIL' log-test || true 2>&1 | tee -a log-test
	grep -o 'undefined: \<.*\>' log-test | sort -u

xtest:
	echo -n > /tmp/ccgo.log
	touch log-test
	cp log-test log-test0
	go test -v -timeout 24h -count=1 -tags=ccgo.dmesg,ccgo.assert 2>&1 | tee log-test
	grep -a 'TRC\|TODO\|ERRORF\|FAIL' log-test || true 2>&1 | tee -a log-test

work:
	rm -f go.work*
	go work init
	go work use .
	go work use ../ccgo/v4
	go work use ../ccgo/v3
	go work use ../cc/v4

surface:
	surface > surface.new
	surface surface.old surface.new > log-todo-surface || true
