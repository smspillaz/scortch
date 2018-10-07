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

#include <vector>

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

  GArray *dimension_list; /* element-type: gint64 */
} ScortchLocalTensorPrivate;

enum {
  PROP_0,
  PROP_DIMENSIONS,
  PROP_N
};

G_DEFINE_TYPE_WITH_PRIVATE (ScortchLocalTensor, scortch_local_tensor, G_TYPE_OBJECT);

namespace
{
  template <typename T>
  void safe_delete (T *t)
  {
    delete t;
  }

  /* XXX: Its not entirely clear to me why,
   *      but if we return an IntList here, we crash
   *      because at::List doesn't make a copy of the underlying
   *      memory, and the constructor does not take an rvalue
   *      reference, so the move never happens. */
  std::vector<int64_t> int_list_from_g_array (GArray *array)
  {
    std::vector <int64_t> vec;
    vec.reserve (array->len);

    for (size_t i = 0; i < array->len; ++i)
      vec.push_back (g_array_index (array, int64_t, i));

    return vec;
  }

  GArray * single_dimensional_empty_tensor ()
  {
    GArray *array = g_array_sized_new (true, true, sizeof (int64_t), 1);

    int64_t value = 1;
    g_array_append_val (array, value);
    return array;
  }
}

/**
 * scortch_local_tensor_get_dimensions:
 * @local_tensor: A #ScortchLocalTensor
 *
 * Get the dimensionality of the tensor in the form of an array
 * of integer values.
 *
 * Arrays can be N-dimensional, as indicated by the number of
 * elements in the array. For instance, a Tensor with dimension
 * [3, 4, 5] has 3 rows, 4 columns and 5 stacks.
 *
 * Returns: (transfer none) (element-type gint64): A #GArray of integer
 *          values representing the dimensionality of the array.
 */
GArray *
scortch_local_tensor_get_dimensions (ScortchLocalTensor *local_tensor)
{
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  return priv->dimension_list;
}

/**
 * scortch_local_tensor_set_dimensions:
 * @local_tensor: A #ScortchLocalTensor
 * @dimensions: (element-type gint64): A #GArray of integer values
 *              representing the dimensionality of the array.
 *
 * Set the dimensionality of the tensor in the form of an array
 * of integer values. If the change in dimensionality results
 * in fewer total array cells than before, then the Tensor
 * data will be truncated. If the change in dimensionality results
 * in more total array cells than before, then the Tensor
 * data will be padded at the end with uninitialized data.
 *
 * Arrays can be N-dimensional, as indicated by the number of
 * elements in the array. For instance, a Tensor with dimension
 * [3, 4, 5] has 3 rows, 4 columns and 5 stacks.
 */
void
scortch_local_tensor_set_dimensions (ScortchLocalTensor *local_tensor,
                                     GArray             *dimensionality)
{
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  g_clear_pointer (&priv->dimension_list, (GDestroyNotify) g_array_unref);
  priv->dimension_list = dimensionality != nullptr ?
    g_array_ref (dimensionality) : single_dimensional_empty_tensor ();
  priv->tensor->resize_ (torch::IntList (int_list_from_g_array (priv->dimension_list)));
}

static void
scortch_local_tensor_get_property (GObject    *object,
                                   guint       prop_id,
                                   GValue     *value,
                                   GParamSpec *pspec)
{
  ScortchLocalTensor *local_tensor = SCORTCH_LOCAL_TENSOR (object);

  switch (prop_id)
    {
      case PROP_DIMENSIONS:
        g_value_set_boxed (value, scortch_local_tensor_get_dimensions (local_tensor));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
scortch_local_tensor_set_property (GObject      *object,
                                   guint         prop_id,
                                   const GValue *value,
                                   GParamSpec   *pspec)
{
  ScortchLocalTensor *local_tensor = SCORTCH_LOCAL_TENSOR (object);

  switch (prop_id)
    {
      case PROP_DIMENSIONS:
        scortch_local_tensor_set_dimensions (local_tensor,
                                             static_cast <GArray *> (g_value_get_boxed (value)));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
scortch_local_tensor_finalize (GObject *object)
{
  ScortchLocalTensor *local_tensor = SCORTCH_LOCAL_TENSOR (object);
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  g_clear_pointer (&priv->tensor, (GDestroyNotify) safe_delete <torch::Tensor>);
  g_clear_pointer (&priv->dimension_list, (GDestroyNotify) g_array_unref);
}

static void
scortch_local_tensor_class_init (ScortchLocalTensorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = scortch_local_tensor_get_property;
  object_class->set_property = scortch_local_tensor_set_property;
  object_class->finalize = scortch_local_tensor_finalize;

  /**
   * ScortchLocalTensor:dimensions: (type GArray(gint64))
   *
   * The dimensions of the tensor.
   *
   * Arrays can be N-dimensional, as indicated by the number of
   * elements in the array. For instance, a Tensor with dimension
   * [3, 4, 5] has 3 rows, 4 columns and 5 stacks.
   */
  g_object_class_install_property (object_class,
                                   PROP_DIMENSIONS,
                                   g_param_spec_boxed ("dimensions",
                                                       "Dimensions",
                                                       "Dimensions of the Tensor",
                                                       G_TYPE_ARRAY,
                                                       G_PARAM_READWRITE));
}

static void
scortch_local_tensor_init (ScortchLocalTensor *local_tensor)
{
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  priv->dimension_list = priv->dimension_list != nullptr ?
    priv->dimension_list : single_dimensional_empty_tensor ();
  priv->tensor = new torch::Tensor (torch::zeros (torch::IntList (int_list_from_g_array (priv->dimension_list))));
}

ScortchLocalTensor *
scortch_local_tensor_new (void)
{
  return static_cast <ScortchLocalTensor *> (g_object_new (SCORTCH_TYPE_LOCAL_TENSOR, NULL));
}
