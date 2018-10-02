/*
 * /tests/scortch/local-tensor-test.cpp
 *
 * Tests for the GObject Binding to the Tensor Object.
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

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <scortch/local-tensor.h>

using ::testing::ElementsAre;
using ::testing::IsNull;
using ::testing::Not;

namespace {
  TEST (ScortchLocalTensor, construct) {
    g_autoptr(ScortchLocalTensor) tensor = scortch_local_tensor_new ();

    EXPECT_THAT (tensor, Not(IsNull()));
  }

  TEST (ScortchLocalTensor, initial_size) {
    g_autoptr(ScortchLocalTensor) tensor = scortch_local_tensor_new ();

    size_t n_dimensions;
    int64_t const *dimensions =
      static_cast <int64_t const *> (g_variant_get_fixed_array (scortch_local_tensor_get_dimensions (tensor),
                                                                &n_dimensions,
                                                                sizeof (int64_t)));
    EXPECT_THAT (std::vector <int64_t> (dimensions, dimensions + n_dimensions),
                 ElementsAre (0));
  }

  TEST (ScortchLocalTensor, resize) {
    g_autoptr(ScortchLocalTensor) tensor = scortch_local_tensor_new ();
    g_autoptr(GArray) array = g_array_sized_new (true, true, sizeof (int64_t), 1);

    int64_t value = 2;
    g_array_append_val (array, value);

    g_autoptr(GVariant) array_variant =
      g_variant_new_fixed_array (G_VARIANT_TYPE ("x"),
                                 static_cast <gconstpointer> (array->data),
                                 array->len,
                                 sizeof (int64_t));

    scortch_local_tensor_set_dimensions (tensor, array_variant);

    size_t n_dimensions;
    int64_t const *dimensions =
      static_cast <int64_t const *> (g_variant_get_fixed_array (scortch_local_tensor_get_dimensions (tensor),
                                                                &n_dimensions,
                                                                sizeof (int64_t)));
    EXPECT_THAT (std::vector <int64_t> (dimensions, dimensions + n_dimensions),
                 ElementsAre (2));
  }
}
