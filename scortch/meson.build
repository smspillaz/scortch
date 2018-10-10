# /scortch/meson.build
#
# Build the libscortch library.
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

api_version = '0'

scortch_toplevel_headers = files([
  'local-tensor.h',
  'scortch-errors.h'
])
scortch_introspectable_sources = files([
  'local-tensor.cpp',
  'scortch-errors.cpp'
])
scortch_private_headers = files([
])
scortch_private_sources = files([
])

scortch_headers_subdir = 'scortch'

install_headers(scortch_toplevel_headers, subdir: scortch_headers_subdir)

scortch_sources = scortch_introspectable_sources + scortch_private_sources

glib = dependency('glib-2.0')
gobject = dependency('gobject-2.0')

scortch_lib = shared_library(
  'scortch',
  scortch_sources,
  soversion: api_version,
  install: true,
  include_directories: [ scortch_inc, torch_inc ],
  dependencies: [
    aten,
    c10,
    caffe2,
    caffe2_detectron_ops_gpu,
    caffe2_gpu,
    caffe2_module_test_dynamic,
    caffe2_observers,
    glib,
    gobject,
    shm,
    torch
  ]
)

scortch_dep = declare_dependency(
  link_with: scortch_lib,
  include_directories: [ scortch_inc ],
)

introspection_sources = [ scortch_introspectable_sources, scortch_toplevel_headers ]

gnome = import('gnome')
gnome.generate_gir(
  scortch_lib,
  extra_args: ['--warn-all', '--warn-error'],
  identifier_prefix: 'Scortch',
  include_directories: scortch_inc,
  includes: ['GLib-2.0', 'GObject-2.0'],
  install: true,
  namespace: 'Scortch',
  nsversion: api_version,
  sources: introspection_sources,
  symbol_prefix: 'scortch'
)

pkg = import('pkgconfig')
pkg.generate(
  description: 'GObject bindings to PyTorch C++ API',
  name: 'libscortch',
  filebase: 'libscortch-' + api_version,
  version: meson.project_version(),
  libraries: scortch_lib,
  install_dir: join_paths(get_option('libdir'), 'pkgconfig')
)
