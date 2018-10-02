/*
 * /tests/js/scortch/testLocalTensor.js
 *
 * Tests for the JavaScript Binding to the Tensor Object.
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

const { Scortch } = imports.gi;

describe('ScortchLocalTensor', function() {
  it('can be constructed', function() {
    let local_tensor = new Scortch.LocalTensor({});
  });

  it('can be constructed with a dimension', function() {
    let local_tensor = new Scortch.LocalTensor({ dimensions: [2] });
  });

  it('can be resized', function() {
    let local_tensor = new Scortch.LocalTensor({});

    local_tensor.resize([2]);
  });
});
