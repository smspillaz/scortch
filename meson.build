# /meson.build
#
# Toplevel meson build file for scortch.
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

project('scortch', 'cpp',
        version: '0.0.0',
        default_options : ['cpp_std=c++17'],
        license: 'LGPL2+',
        meson_version: '>= 0.46.0')
cpp_compiler = meson.get_compiler('cpp')

scortch_inc = include_directories('.')

python = import('python')
gnome = import('gnome')

glib = dependency('glib-2.0')
gobject = dependency('gobject-2.0')

py_installation_path = get_option('python_installation')

if py_installation_path == ''
  py_installation_path = python.find_installation('python3').get_install_dir()
endif

torch_dir = join_paths([py_installation_path,
                        'torch'])
torch_lib_include_path = join_paths([py_installation_path,
                                     'torch',
                                     'lib',
                                     'include',
                                     'torch',
                                     'csrc',
                                     'api',
                                     'include'])
torch_include_path = join_paths([py_installation_path,
                                 'torch',
                                 'lib',
                                 'include'])
py_include_path = join_paths([py_installation_path,
                              '..',
                              '..',
                              '..',
                              'include',
                              'python3.6m'])

torch_includes = include_directories(torch_lib_include_path,
                                     torch_include_path,
                                     py_include_path)
torch_lib_path = join_paths([torch_dir, 'lib'])
torch = cpp_compiler.find_library('libtorch',
                                  dirs: torch_lib_path)
aten = cpp_compiler.find_library('libATen',
                                 dirs: torch_lib_path)
caffe2 = cpp_compiler.find_library('libcaffe2',
                                   dirs: torch_lib_path)
caffe2_gpu = cpp_compiler.find_library('libcaffe2_gpu',
                                       dirs: torch_lib_path)
caffe2_observers = cpp_compiler.find_library('libcaffe2_observers',
                                             dirs: torch_lib_path)
caffe2_module_test_dynamic = cpp_compiler.find_library('libcaffe2_module_test_dynamic',
                                                       dirs: torch_lib_path)
caffe2_detectron_ops_gpu = cpp_compiler.find_library('libcaffe2_detectron_ops_gpu',
                                                     dirs: torch_lib_path)

subdir('scortch')
subdir('examples')