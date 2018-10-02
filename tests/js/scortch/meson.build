# /tests/js/scortch/meson.build
#
# Meson build file for the scortch library javascript tests.
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

javascript_tests = [
  'testLocalTensor.js'
]

jasmine = find_program('jasmine', required: false)
include_path = '@0@:@1@'.format(meson.source_root(), meson.build_root())
built_library_path = join_paths(meson.build_root(), meson.project_name())

tests_environment = environment()
tests_environment.set('GJS_PATH', include_path)
tests_environment.prepend('GI_TYPELIB_PATH', built_library_path)
tests_environment.prepend('LD_LIBRARY_PATH', built_library_path)

if jasmine.found()
  foreach test_file : javascript_tests
    test(test_file,
         jasmine,
         args: [join_paths(meson.current_source_dir(), test_file)],
         env: tests_environment)
  endforeach
endif
