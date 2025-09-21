#!/bin/bash
#
#
#
#
source env-debug
#
#
#
# --enable-tap-tests


cd postgresql-12.5

make distclean

./configure --prefix=$PGHOME --disable-tap-tests --enable-debug --disable-coverage CCFLAG="-g -O0" CC='gcc'

make -j 4 world                 #

make install-world

cd ..

echo 'Run Sucessful.'



#   --enable-profiling      build with profiling enabled
#  --enable-coverage       build with coverage testing instrumentation
#  --enable-dtrace         build with DTrace support
#  --enable-tap-tests      enable TAP tests (requires Perl and IPC::Run)
