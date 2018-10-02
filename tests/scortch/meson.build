# /tests/scortch/meson.build
#
# Meson build file for scortch library tests.
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

scortch_test_sources = [
  'local-tensor-test.cpp',
]

glib = dependency('glib-2.0')
gobject = dependency('gobject-2.0')

scortch_test_executable = executable(
  'scortch_test',
  scortch_test_sources,
  dependencies: [
    gtest_dep,
    gtest_main_dep,
    gmock_dep,
    glib,
    gobject,
    scortch_dep
  ],
  include_directories: [ scortch_inc, tests_inc ]
)

test('scortch_test', scortch_test_executable)
