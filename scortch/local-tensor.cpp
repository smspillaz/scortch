/*
 * /scortch/local-tensor.cpp
 *
 * GObject Binding to the Tensor Object, the foundation
 * of tensor operations in PyTorch. C++ source file.
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

#include <glib-object.h>
#include <glib.h>
#include <gobject/gobject.h>

#include <torch/torch.h>
#include <torch/script.h>

#include <scortch/local-tensor.h>

struct _ScortchLocalTensor
{
  GObject parent_instance;
};

typedef struct _ScortchLocalTensorPrivate {
  torch::Tensor *tensor;
} ScortchLocalTensorPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (ScortchLocalTensor, scortch_local_tensor, G_TYPE_OBJECT);

namespace
{
  template <typename T>
  void safe_delete (T *t)
  {
    delete t;
  }
}

static void
scortch_local_tensor_finalize (GObject *object)
{
  ScortchLocalTensor *local_tensor = SCORTCH_LOCAL_TENSOR (object);
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  g_clear_pointer (&priv->tensor, (GDestroyNotify) safe_delete <torch::Tensor>);
}

static void
scortch_local_tensor_class_init (ScortchLocalTensorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->finalize = scortch_local_tensor_finalize;
}

static void
scortch_local_tensor_init (ScortchLocalTensor *local_tensor)
{
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  priv->tensor = new torch::Tensor ();
}

ScortchLocalTensor *
scortch_local_tensor_new (void)
{
  return static_cast <ScortchLocalTensor *> (g_object_new (SCORTCH_TYPE_LOCAL_TENSOR, NULL));
}
