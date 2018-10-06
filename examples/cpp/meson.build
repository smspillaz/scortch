# /examples/meson.build
#
# Meson build file for C++ examples
#
# Copyright (C) 2018 Sam Spilsbury.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

example = executable(
  'predict-words',
  ['predict-words.cpp'],
  dependencies: [
    glib,
    gobject,
    aten,
    caffe2,
    caffe2_gpu,
    caffe2_observers,
    caffe2_module_test_dynamic,
    caffe2_detectron_ops_gpu,
    torch
  ],
  include_directories: [ torch_includes ]
)
