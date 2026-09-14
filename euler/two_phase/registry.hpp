// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../config.hpp"
#include "../init/registry.hpp"
#include "eos.hpp"

// =============================================================================
//  Registering a two-phase test case
// -----------------------------------------------------------------------------
//  The registry of euler/init/registry.hpp is already templated on the field and
//  on the equation of state, which is all that has to change here: a two-phase
//  state has dim + 4 components instead of dim + 2, and its thermodynamics takes
//  a volume fraction. What is left is the field type and the registration macro.
//
//      REGISTER_TWO_PHASE_CASE(water_air_shock_tube, test_case::water_air, 1, 2)
//
//  reads exactly like its monofluid counterpart and registers the case for each
//  of the dimensions listed.
// =============================================================================

namespace two_phase
{
    template <std::size_t Dim>
    struct config
    {
        static constexpr std::size_t dim = Dim;
        using mesh_t                     = typename ::config<Dim>::mesh_t;
        using field_t                    = samurai::VectorField<mesh_t, double, Dim + 4>;
    };

    template <std::size_t Dim>
    using registry_t = test_case::TestCaseRegistry<typename config<Dim>::field_t, EOS::Mixture>;

    template <std::size_t Dim>
    using test_case_t = test_case::TestCase<typename config<Dim>::field_t, EOS::Mixture>;

    template <std::size_t... Dims, class MakeDefinition>
    bool register_for_dims(const std::string& name, MakeDefinition make)
    {
        (registry_t<Dims>::instance().register_test_case(name, make.template operator()<typename config<Dims>::field_t>()), ...);
        return true;
    }
}

#define REGISTER_TWO_PHASE_CASE(NAME, NS, ...)                                                                                   \
    namespace                                                                                                                    \
    {                                                                                                                            \
        const bool registered_two_phase_##NAME = ::two_phase::register_for_dims<__VA_ARGS__>(#NAME,                              \
                                                                                             []<class Field>()                   \
                                                                                             {                                   \
                                                                                                 return NS::definition<Field>(); \
                                                                                             });                                 \
    }
