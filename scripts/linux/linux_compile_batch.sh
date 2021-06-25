#!/bin/bash

INSTALL_DIR=visionEngine-ubuntu-1804
BUILD_DIR=build-ubuntu-1804

echo 'Build ubuntu 1804!'
sh compile_ubuntu_1804.sh
mkdir -p $INSTALL_DIR
cp -r $BUILD_DIR/include $INSTALL_DIR
cp -r $BUILD_DIR/lib $INSTALL_DIR
cp -r $BUILD_DIR/bin $INSTALL_DIR
rm -rf $BUILD_DIR