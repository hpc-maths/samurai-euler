// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "double_mach.hpp"
#include "free_stream.hpp"
#include "isentropic_vortex.hpp"
#include "kelvin_helmholtz.hpp"
#include "riemann_2d.hpp"
#include "sedov_blast.hpp"
#include "sod.hpp"

namespace test_case
{
    // The list of available test cases, and the dimension each one supports.
    // Call this once, before the command line is parsed, so that `--test-case`
    // can check its argument against the registry.
    template <class Field>
    void register_all()
    {
        // Dimension agnostic.
        free_stream::register_me<Field>();
        sedov_blast::register_me<Field>();

        // Two-dimensional only.
        if constexpr (Field::dim == 2)
        {
            double_mach_reflection::register_me();
            isentropic_vortex::register_me();
            kelvin_helmholtz::register_me();
            riemann_2d::register_me();
            sod::register_me();
        }
    }
}
