// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

// Radius of the multiresolution prediction (1: third order, 2: fifth order).
#ifndef EULER_PREDICTION_RADIUS
#define EULER_PREDICTION_RADIUS 1
#endif

template <std::size_t Dim>
struct config
{
    static constexpr std::size_t dim = Dim;
    using Config                     = samurai::mesh_config<dim, EULER_PREDICTION_RADIUS>;
    using mesh_t                     = samurai::MRMesh<Config>;
    using field_t                    = samurai::VectorField<mesh_t, double, dim + 2>;
};
