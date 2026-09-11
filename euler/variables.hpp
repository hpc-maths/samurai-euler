// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "eos.hpp"

template <std::size_t Dim>
struct EulerLayout
{
    static constexpr std::size_t rho  = 0;
    static constexpr std::size_t rhoE = 1;

    static constexpr std::size_t mom(std::size_t d)
    {
        assert(d < Dim);
        return 2 + d;
    }

    static constexpr std::size_t size = 2 + Dim;
};

template <std::size_t Dim>
struct PrimState
{
    double rho;
    double p;
    xt::xtensor_fixed<double, xt::xshape<Dim>> v;
};

template <std::size_t Dim, class Eos>
auto cons2prim(const xt::xtensor_fixed<double, xt::xshape<EulerLayout<Dim>::size>>& conserved, const Eos& eos)
{
    using EulerConsVar = EulerLayout<Dim>;

    PrimState<Dim> primitives;
    primitives.rho = conserved[EulerConsVar::rho];
    auto e         = conserved[EulerConsVar::rhoE] / conserved[EulerConsVar::rho];
    for (std::size_t d = 0; d < Dim; ++d)
    {
        primitives.v[d] = conserved[EulerConsVar::mom(d)] / conserved[EulerConsVar::rho];
        e -= 0.5 * (primitives.v[d] * primitives.v[d]);
    }
    primitives.p = eos.p(primitives.rho, e);
    return primitives;
}

template <std::size_t Dim, class Eos>
auto prim2cons(const PrimState<Dim>& primitives, const Eos& eos)
{
    using EulerConsVar = EulerLayout<Dim>;

    xt::xtensor_fixed<double, xt::xshape<EulerConsVar::size>> conserved;

    conserved[EulerConsVar::rho]  = primitives.rho;
    auto e                        = eos.e(primitives.rho, primitives.p);
    conserved[EulerConsVar::rhoE] = e * conserved[EulerConsVar::rho];
    for (std::size_t d = 0; d < Dim; ++d)
    {
        conserved[EulerConsVar::mom(d)] = primitives.v[d] * conserved[EulerConsVar::rho];
        conserved[EulerConsVar::rhoE] += 0.5 * primitives.v[d] * primitives.v[d] * conserved[EulerConsVar::rho];
    }
    return conserved;
}
