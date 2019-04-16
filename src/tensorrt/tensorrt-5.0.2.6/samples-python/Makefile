SHELL=/bin/bash -o pipefail
TARGET?=$(shell uname -m)
LIBDIR?=lib
VERBOSE?=0
ifeq ($(VERBOSE), 1)
AT=
else
AT=@
endif
TRIPLE=x86_64-linux
ifeq ($(TARGET), aarch64)
TRIPLE=aarch64-linux
endif
ifeq ($(TARGET), qnx)
TRIPLE=qnx-linux
endif
ifeq ($(TARGET), android32)
TRIPLE=arm-linux-androideabi
endif
ifeq ($(TARGET), android64)
TRIPLE=aarch64-linux-android
endif
export TARGET
export VERBOSE
export LIBDIR
export TRIPLE
samples=getDigits sampleCharRNN sampleFasterRCNN sampleGoogleNet sampleINT8 sampleINT8API sampleMLP sampleMNIST sampleMNISTAPI sampleMovieLens sampleNMT sampleOnnxMNIST samplePlugin sampleSSD sampleUffMNIST sampleUffSSD trtexec

# sampleMovieLensMPS should only be compiled for Linux targets.
# sample uses Linux specific shared memory and IPC libraries.
ifeq ($(TARGET),x86_64)
	samples += sampleMovieLensMPS
endif

.PHONY: all clean help test test_debug test_release
all:
	$(AT)$(foreach sample,$(samples), $(MAKE) -C $(sample) &&) :

clean:
	$(AT)$(foreach sample,$(samples), $(MAKE) clean -C $(sample) &&) :

test: test_debug test_release

test_debug:
	$(AT)$(foreach sample,$(samples), $(MAKE) test_debug -C $(sample) &&) :

test_release:
	$(AT)$(foreach sample,$(samples), $(MAKE) test_release -C $(sample) &&) :

help:
	$(AT)echo "Sample building help menu."
	$(AT)echo "Samples:"
	$(AT)$(foreach sample,$(samples), echo "\t$(sample)" &&) :
	$(AT)echo "\nCommands:"
	$(AT)echo "\tall - build all samples."
	$(AT)echo "\tclean - clean all samples."
	$(AT)echo "\ttest[_debug|_release] - test all/release/debug samples."
	$(AT)echo "\nVariables:"
	$(AT)echo "\tTARGET - Specify the target to build for."
	$(AT)echo "\tVERBOSE - Specify verbose output."
	$(AT)echo "\tCUDA_INSTALL_DIR - Directory where cuda installs to."
