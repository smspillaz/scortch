/*
 * /scortch/local-tensor.cpp
 *
 * GObject Binding to the Tensor Object, the foundation
 * of tensor operations in PyTorch. C header file.
 *
 * Copyright (C) 2018 Sam Spilsbury.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#pragma once

#include <glib.h>
#include <glib-object.h>

G_BEGIN_DECLS

#define SCORTCH_TYPE_LOCAL_TENSOR scortch_local_tensor_get_type ()
G_DECLARE_FINAL_TYPE (ScortchLocalTensor, scortch_local_tensor, SCORTCH, LOCAL_TENSOR, GObject)

GVariant * scortch_local_tensor_get_dimensions (ScortchLocalTensor *local_tensor);
void scortch_local_tensor_set_dimensions (ScortchLocalTensor *local_tensor,
                                          GVariant           *dimensions);

ScortchLocalTensor * scortch_local_tensor_new (void);

G_END_DECLS
