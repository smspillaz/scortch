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

#include <algorithm>
#include <functional>
#include <vector>

#include <glib-object.h>
#include <glib.h>
#include <gobject/gobject.h>

#include <torch/torch.h>

#include <scortch/local-tensor.h>
#include <scortch/scortch-errors.h>

struct _ScortchLocalTensor
{
  GObject parent_instance;
};

typedef struct _ScortchLocalTensorPrivate {
  torch::Tensor *tensor;

  GVariant *dimension_list; /* signature: ax */
  GVariant *construction_data_variant; /* signature: av */
} ScortchLocalTensorPrivate;

enum {
  PROP_0,
  PROP_DIMENSIONS,
  PROP_DATA,
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
   *      but if we return an IntArrayRef here, we crash
   *      because at::List doesn't make a copy of the underlying
   *      memory, and the constructor does not take an rvalue
   *      reference, so the move never happens. */
  std::vector<int64_t> int_list_from_g_variant (GVariant *variant)
  {
    std::vector <int64_t> vec;
    size_t n_elements;
    int64_t const *fixed_array =
      static_cast <int64_t const *> (g_variant_get_fixed_array (variant,
                                                                &n_elements,
                                                                sizeof (int64_t)));
    vec.reserve (n_elements);

    for (size_t i = 0; i < n_elements; ++i)
      vec.push_back (fixed_array[i]);

    return vec;
  }

  GVariant * single_dimensional_empty_tensor ()
  {
    g_auto(GVariantBuilder) builder = G_VARIANT_BUILDER_INIT (G_VARIANT_TYPE ("ax"));

    g_variant_builder_init (&builder, G_VARIANT_TYPE ("ax"));
    g_variant_builder_add (&builder, "x", 0);

    return g_variant_builder_end (&builder);
  }

  class InvalidVariantTypeError : public std::logic_error
  {
    public:
      InvalidVariantTypeError (GVariantType const *variant_type) :
        std::logic_error::logic_error (InvalidVariantTypeError::format_error (variant_type))
      {
      }

    private:
      static inline std::string format_error (GVariantType const *variant_type)
      {
        std::stringstream ss;
        ss << "Cannot convert GVariantType "
           << g_variant_type_peek_string (variant_type)
           << " to GVariantType";
        return ss.str ();
      }
  };

  class InvalidScalarTypeError : public std::logic_error
  {
    public:
      InvalidScalarTypeError (caffe2::TypeMeta const &scalar_type) :
        std::logic_error::logic_error (InvalidScalarTypeError::format_error (scalar_type))
      {
      }

    private:
      static inline std::string format_error (caffe2::TypeMeta const &scalar_type)
      {
        std::stringstream ss;
        ss << "Cannot handle scalar type " << scalar_type;
        return ss.str ();
      }
  };

  GVariantType const * scalar_type_to_g_variant_type (caffe2::TypeMeta scalar_type)
  {
    /* XXX: We do not support float tensors
     *      at the moment as GVariant doesn't
     *      support floats. */
    if (scalar_type == torch::kFloat64) {
      return G_VARIANT_TYPE_DOUBLE;
    } else if (scalar_type == torch::kInt64) {
      return G_VARIANT_TYPE_INT64;
    } else {
      throw InvalidScalarTypeError (scalar_type);
    }
  }

  size_t scalar_type_to_element_size (caffe2::TypeMeta scalar_type)
  {
    if (scalar_type == torch::kFloat64) {
      return sizeof (double);
    } else if (scalar_type == torch::kInt64) {
      return sizeof (int64_t);
    } else {
      throw InvalidScalarTypeError (scalar_type);
    }
  }

  template <typename T>
  std::vector <T>
  append_to_vector (std::vector <T> &&vec, T &&v)
  {
    std::vector vec_out (vec);
    vec_out.push_back(v);
    return vec_out;
  }

  std::tuple <GVariantType const *, std::vector <int64_t>> ascertain_underlying_type_and_dimensions (GVariant *array_variant)
  {
    GVariantType const *array_variant_type = G_VARIANT_TYPE (g_variant_get_type_string (array_variant));
    if (!g_variant_type_equal (array_variant_type, G_VARIANT_TYPE ("av")))
      {
        g_assert (g_variant_type_is_container (array_variant_type));
        return std::make_tuple (array_variant_type,
                                append_to_vector (std::vector <int64_t> (),
                                                  static_cast <int64_t> (g_variant_n_children (array_variant))));
      }

    g_autoptr(GVariant) child_variant = g_variant_ref_sink (g_variant_get_child_value (array_variant, 0));
    g_autoptr(GVariant) child_array = g_variant_ref_sink (g_variant_get_variant (child_variant));

    GVariantType const *variant_type;
    std::vector <int64_t> dimension_vec;

    std::tie (variant_type, dimension_vec) = ascertain_underlying_type_and_dimensions (child_array);

    return std::make_tuple (variant_type,
                            append_to_vector (std::move (dimension_vec),
                                              static_cast <int64_t> (g_variant_n_children (array_variant))));
  }

  unsigned int set_error_from_exception (std::exception const  &exception,
                                         GQuark                 domain,
                                         ScortchError           code,
                                         GError               **error)
  {
    g_set_error (error,
                 domain,
                 code,
                 "%s",
                 exception.what ());
    return 0;
  }

  template <typename T>
  void iterate_and_assign_to_tensor (torch::Tensor &tensor,
                                     const char    *type_string,
                                     GVariant      *array_variant)
  {
    T scalar;
    GVariantIter iter;
    size_t       count = 0;

    g_variant_iter_init (&iter, array_variant);

    while (g_variant_iter_next (&iter, type_string, &scalar))
      {
        tensor[count++] = scalar;
      }
  }

  void set_tensor_data_from_nested_variant_arrays (torch::Tensor       &tensor,
                                                   GVariant            *array_variant,
                                                   GVariantType  const *underlying_type)
  {
    /* Base case */
    if (!g_variant_is_of_type (array_variant, G_VARIANT_TYPE ("av")))
      {
        g_assert (g_variant_type_is_container (G_VARIANT_TYPE (g_variant_get_type_string (array_variant))));
        g_assert (g_variant_is_of_type (array_variant, underlying_type));

        const char   *type_string = (g_variant_type_peek_string (underlying_type) + 1);

        if (g_variant_type_equal (underlying_type, G_VARIANT_TYPE ("ad")))
          {
            iterate_and_assign_to_tensor <double> (tensor, type_string, array_variant);
          }
        else if (g_variant_type_equal (underlying_type, G_VARIANT_TYPE ("ax")))
          {
            iterate_and_assign_to_tensor <int64_t> (tensor, type_string, array_variant);
          }
        else
          {
            throw InvalidVariantTypeError (underlying_type);
          }

        return;
      }

    /* Recursive case */
    GVariantIter iter;
    GVariant     *unowned_child_array;
    size_t       tensor_index = 0;

    g_variant_iter_init (&iter, array_variant);
    while (g_variant_iter_next (&iter, "v", &unowned_child_array))
      {
        g_autoptr(GVariant) child_array = g_variant_ref_sink (unowned_child_array);

        torch::Tensor child_tensor (tensor[tensor_index]);
        set_tensor_data_from_nested_variant_arrays (child_tensor, child_array, underlying_type);

        ++tensor_index;
      }
  }

  torch::Tensor new_tensor_from_nested_gvariants (GVariant *array_variant)
  {
    GVariantType const *underlying_type;
    std::vector <int64_t> dimensions;

    std::tie (underlying_type, dimensions) = ascertain_underlying_type_and_dimensions (array_variant);
    std::reverse (dimensions.begin (), dimensions.end ());

    torch::Tensor tensor = torch::zeros (torch::IntArrayRef (dimensions)).cpu ();
    set_tensor_data_from_nested_variant_arrays (tensor, array_variant, underlying_type);

    return tensor;
  }

  GVariant * serialize_tensor_data_to_nested_gvariants (at::Tensor const &tensor)
  {
    /* Base case, only a single dimension left */
    if (tensor.dim () == 1)
      {
        return g_variant_new_fixed_array (scalar_type_to_g_variant_type (tensor.dtype ()),
                                          tensor.data_ptr (),
                                          tensor.sizes ()[0],
                                          scalar_type_to_element_size (tensor.dtype ()));
      }

    /* Recursive case: Build a new array-of-variants
     * by looping through the current dimension and
     * getting arrays from that. */
    g_auto(GVariantBuilder) builder = G_VARIANT_BUILDER_INIT (G_VARIANT_TYPE ("av"));
    g_variant_builder_init (&builder, G_VARIANT_TYPE ("av"));

    for (size_t i = 0; i < tensor.sizes ()[0]; ++i)
      {
        g_variant_builder_add (&builder,
                               "v",
                               serialize_tensor_data_to_nested_gvariants (tensor[i]));
      }

    return g_variant_builder_end (&builder);
  }

  template <typename Func, typename... Args>
  typename std::result_of <Func(Args..., GError **)>::type
  call_and_warn_about_gerror(const char *operation, Func &&f, Args&& ...args)
  {
    GError *error = nullptr;

    auto result = f(args..., &error);

    if (error != nullptr)
      {
        g_warning ("Could not %s: %s", operation, error->message);
        return reinterpret_cast <decltype(result)> (0);
      }

    return result;
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
 * Returns: (transfer none): A #GVariant of integer
 *          values representing the dimensionality of the array.
 */
GVariant *
scortch_local_tensor_get_dimensions (ScortchLocalTensor *local_tensor)
{
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  return priv->dimension_list;
}

/**
 * scortch_local_tensor_set_dimensions:
 * @local_tensor: A #ScortchLocalTensor
 * @dimensions: A #GVariant of integer values
 *               representing the dimensionality of the array.
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
                                     GVariant           *dimensionality)
{
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  g_clear_pointer (&priv->dimension_list, (GDestroyNotify) g_variant_unref);
  priv->dimension_list = dimensionality != nullptr ?
    g_variant_ref (dimensionality) : single_dimensional_empty_tensor ();

  /* We can't set the dimensions until the underlying tensor is constructed */
  if (priv->tensor != nullptr)
    {
      priv->tensor->resize_ (torch::IntArrayRef (int_list_from_g_variant (priv->dimension_list)));
    }
}

/**
 * scortch_local_tensor_get_data:
 * @local_tensor: A tensor to get the data for.
 *
 * Return the underlying data for a tensor as an array of variants
 * (av), where each variant in the array is itself an array
 * array of variants or an array of a particular datatype
 * (d|x|f).
 *
 * The level of nesting of array-variants corresponds to
 * the number of dimensions in the tensor. For instance, a 2D
 * tensor will have an array of arrays of (d|x|f). It is the programmer's
 * responsibility to ensure that the returned variant is decoded
 * properly, both in terms of its nesting and its underlying
 * datatype.
 *
 * Note that calling this function will cause PyTorch to
 * copy data from GPU memory into CPU memory, so it should
 * be used seldomly.
 *
 * Returns: (transfer full): A floating reference to a new
 *          #GVariant containing the tensor data.
 */
GVariant *
scortch_local_tensor_get_data (ScortchLocalTensor  *local_tensor,
                               GError             **error)
{
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  if (priv->tensor == nullptr)
    return g_variant_ref (priv->construction_data_variant);

  try
    {
      return serialize_tensor_data_to_nested_gvariants (*priv->tensor);
    }
  catch (InvalidScalarTypeError const &e)
    {
      return reinterpret_cast <GVariant *> (set_error_from_exception (e,
                                                                      SCORTCH_ERROR,
                                                                      SCORTCH_ERROR_INVALID_DATA_TYPE,
                                                                      error));
    }
}

/**
 * scortch_local_tensor_set_data:
 * @local_tensor: A tensor to set the data on
 * @data: (transfer none): A #GVariant of type "av" containing
 *        an array of variants according to the schema
 *        specified in %scortch_local_tensor_get_data.
 *
 * The tensor will be automatically resized and adopt
 * the dimensionality of the nested array of variants. It
 * is the programmer's responsibility to ensure that
 * sub-array sizes are consistent between sub-arrays
 * of the same dimension and that the underlying datatype
 * is consistent between all sub-arrays.
 *
 * PyTorch will likely copy the contents of the array
 * either into CPU memory or GPU memory as a result of
 * calling this function, so it should be used seldomly.
 */
gboolean
scortch_local_tensor_set_data (ScortchLocalTensor  *local_tensor,
                               GVariant            *data,
                               GError            **error)
{
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  /* %NULL data is ignored */
  if (data == nullptr)
    return TRUE;

  if (priv->tensor != nullptr)
    {
      try
        {
          priv->tensor->set_data (new_tensor_from_nested_gvariants (data));
        }
      catch (InvalidVariantTypeError &e)
        {
          return (gboolean) (set_error_from_exception (e,
                                                       SCORTCH_ERROR,
                                                       SCORTCH_ERROR_INVALID_DATA_TYPE,
                                                       error));
        }
    }
  else
    {
      priv->construction_data_variant = g_variant_ref_sink (data);
    }

  return TRUE;
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
        g_value_set_variant (value, scortch_local_tensor_get_dimensions (local_tensor));
        break;
      case PROP_DATA:
        /* XXX: It seems that clang can't deduce the type of a function
         * pointer at the moment, so we work around that by calling through
         * a lambda instead */
        g_value_set_variant (value,
                             call_and_warn_about_gerror ("get 'data' property",
                                                         [](ScortchLocalTensor  *local_tensor,
                                                            GError             **error) -> decltype(auto) {
                                                           return scortch_local_tensor_get_data (local_tensor,
                                                                                                 error);
                                                         },
                                                         local_tensor));
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
                                             g_value_get_variant (value));
        break;
      case PROP_DATA:
        call_and_warn_about_gerror ("set 'data' property",
                                    [](ScortchLocalTensor  *local_tensor,
                                       GVariant            *data,
                                       GError             **error) -> decltype(auto) {
                                      return scortch_local_tensor_set_data (local_tensor,
                                                                            data,
                                                                            error);
                                    },
                                    local_tensor,
                                    g_value_get_variant (value));
        break;
      default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
        break;
    }
}

static void
scortch_local_tensor_constructed (GObject *object)
{
  ScortchLocalTensor *local_tensor = SCORTCH_LOCAL_TENSOR (object);
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  priv->tensor = new torch::Tensor (torch::zeros (torch::IntArrayRef (int_list_from_g_variant (priv->dimension_list))));

  /* We need to wait until we have the tensor to set
   * its data from the construction parameters. */
  if (priv->construction_data_variant != nullptr)
    {
      call_and_warn_about_gerror ("set 'data' property on construction",
                                  [](ScortchLocalTensor  *local_tensor,
                                      GVariant            *data,
                                      GError             **error) -> decltype(auto) {
                                    return scortch_local_tensor_set_data (local_tensor,
                                                                          data,
                                                                          error);
                                  },
                                  local_tensor,
                                  priv->construction_data_variant);
      g_clear_pointer (&priv->construction_data_variant, (GDestroyNotify) g_variant_unref);
    }

  G_OBJECT_CLASS (scortch_local_tensor_parent_class)->constructed (object);
}

static void
scortch_local_tensor_finalize (GObject *object)
{
  ScortchLocalTensor *local_tensor = SCORTCH_LOCAL_TENSOR (object);
  ScortchLocalTensorPrivate *priv =
    static_cast <ScortchLocalTensorPrivate *> (scortch_local_tensor_get_instance_private (local_tensor));

  g_clear_pointer (&priv->tensor, (GDestroyNotify) safe_delete <torch::Tensor>);
  g_clear_pointer (&priv->dimension_list, (GDestroyNotify) g_variant_unref);
  g_clear_pointer (&priv->construction_data_variant, (GDestroyNotify) g_variant_unref);

  G_OBJECT_CLASS (scortch_local_tensor_parent_class)->finalize (object);
}

static void
scortch_local_tensor_class_init (ScortchLocalTensorClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->constructed = scortch_local_tensor_constructed;
  object_class->get_property = scortch_local_tensor_get_property;
  object_class->set_property = scortch_local_tensor_set_property;
  object_class->finalize = scortch_local_tensor_finalize;

  /**
   * ScortchLocalTensor:dimensions:
   *
   * The dimensions of the tensor.
   *
   * Arrays can be N-dimensional, as indicated by the number of
   * elements in the array. For instance, a Tensor with dimension
   * [3, 4, 5] has 3 rows, 4 columns and 5 stacks.
   */
  g_object_class_install_property (object_class,
                                   PROP_DIMENSIONS,
                                   g_param_spec_variant ("dimensions",
                                                         "Dimensions",
                                                         "Dimensions of the Tensor",
                                                         G_VARIANT_TYPE ("ax"),
                                                         single_dimensional_empty_tensor (),
                                                         static_cast <GParamFlags> (G_PARAM_READWRITE |
                                                                                    G_PARAM_CONSTRUCT)));

  /**
   * ScortchLocalTensor:data:
   *
   * The data of the tensor as a nested array of arrays of variants,
   * with the leaf variants being arrays of concrete types.
   *
   * When set, the tensor will be automatically resized and adopt
   * the dimensionality of the nested array of variants. It
   * is the programmer's responsibility to ensure that
   * sub-array sizes are consistent between sub-arrays
   * of the same dimension and that the underlying datatype
   * is consistent between all sub-arrays. Error cannot be thrown
   * from accessing properties, so if an error occurs %NULL will
   * be returned and an error message printed to the standard out. If
   * you need to handle errors, use %scortch_local_tensor_set_data instead.
   *
   * PyTorch will likely copy the contents of the array
   * either into CPU memory or GPU memory as a result of
   * calling this function, so this property should be used
   * seldomly.
   */
  g_object_class_install_property (object_class,
                                   PROP_DATA,
                                   g_param_spec_variant ("data",
                                                         "Data",
                                                         "Data of the Tensor",
                                                         G_VARIANT_TYPE ("av"),
                                                         nullptr,
                                                         static_cast <GParamFlags> (G_PARAM_READWRITE |
                                                                                    G_PARAM_CONSTRUCT)));
}

static void
scortch_local_tensor_init (ScortchLocalTensor *local_tensor)
{
}

ScortchLocalTensor *
scortch_local_tensor_new (void)
{
  return static_cast <ScortchLocalTensor *> (g_object_new (SCORTCH_TYPE_LOCAL_TENSOR, NULL));
}
