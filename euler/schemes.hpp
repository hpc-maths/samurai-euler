// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "eos.hpp"
#include "schemes/hll.hpp"
#include "schemes/hllc.hpp"
#include "schemes/rusanov.hpp"

template <class Field, class Eos>
auto get_fv_scheme(const std::string& scheme, const Eos& eos)
{
    if (scheme == "rusanov")
    {
        return make_euler_rusanov<Field>(eos);
    }
    else if (scheme == "hll")
    {
        return make_euler_hll<Field>(eos);
    }
    else if (scheme == "hllc")
    {
        return make_euler_hllc<Field>(eos);
    }
    else
    {
        throw std::runtime_error("Unknown scheme: " + scheme);
    }
}