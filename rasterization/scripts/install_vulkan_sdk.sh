#!/bin/bash
# helper script to clone, build and install the Vulkan SDK on headless CI systems
# currently outputs into VULKAN_SDK in the current directory
# 2021.02 humbletim -- released under the MIT license

# MIT License
# 
# Copyright (c) 2021 humbletim
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# usage: ./install_vulkan_sdk.sh <SDK_release_version | sdk-x.y.z>
#    eg: ./install_vulkan_sdk.sh 1.2.162.1 # branch resolved via service
#    eg: ./install_vulkan_sdk.sh sdk-1.2.162 # branch used directly

# log messages will be printed to STDERR (>&2)
# these sourcable environment variables will be printed to STDOUT on success:
#   VULKAN_SDK=...
#   VULKAN_SDK_VERSION=...

set -e

VK_VERSION=${1:-latest}

os=unknown
build_dir=$(realpath ./third_party/VULKAN_SDK)
case `uname -s` in
  Darwin) echo "TODO=Darwin" ;  exit 5 ;;
  Linux)
    os=linux
    ;;
  MINGW*)
    os=windows
    CC=cl.exe
    CXX=cl.exe
    PreferredToolArchitecture=x64
    unset TEMP
    unset TMP
    ;;
esac
echo os=$os >&2
echo build_dir=$build_dir >&2

# resolve latest into an actual SDK release number (currently only used for troubleshooting / debug output)
echo "using specified branch/tag name as-is: $VK_VERSION" >&2
BRANCH=$VK_VERSION

MAKEFLAGS=-j2

mkdir -p $build_dir/_build
pushd $build_dir/_build >&2

  git clone --single-branch --depth=1 --branch="$BRANCH" https://github.com/KhronosGroup/Vulkan-Headers.git >&2
  pushd Vulkan-Headers >&2
    cmake -DCMAKE_INSTALL_PREFIX=$build_dir -DCMAKE_BUILD_TYPE=Release . >&2
    cmake --build . --config Release >&2
    cmake --install . >&2
  popd >&2

  git clone --single-branch --depth=1 --branch="$BRANCH" https://github.com/KhronosGroup/Vulkan-Loader.git >&2
  pushd Vulkan-Loader >&2
    cmake -DVULKAN_HEADERS_INSTALL_DIR=$build_dir -DCMAKE_INSTALL_PREFIX=$build_dir -DCMAKE_BUILD_TYPE=Release . >&2
    cmake --build . --config Release >&2
    cmake --install . >&2
  popd >&2
popd >&2

echo "" >&2

# export these so that "sourcing" this file directly also works
export VULKAN_SDK_VERSION=$BRANCH
export VULKAN_SDK=$build_dir

# also print to STDOUT for eval'ing or appending to $GITHUB_ENV:
echo VULKAN_SDK_VERSION=$VULKAN_SDK_VERSION
echo VULKAN_SDK=$VULKAN_SDK

# cleanup _build artifacts which are no longer needed after cmake --installs above
rm -rf $build_dir/_build >&2

# echo "VULKAN_SDK/" >&2
# ls VULKAN_SDK >&2
du -hs $build_dir >&2
