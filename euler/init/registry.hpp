// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <samurai/box.hpp>

#include "../config.hpp"
#include "../eos.hpp"

// =============================================================================
//  Test case registry
// -----------------------------------------------------------------------------
//  A test case is the four things main() needs in order to set a simulation up
//  and that main() cannot guess: where the domain is, what the initial state is,
//  how the boundaries behave, and which gas is being modelled.
//
//  The registry is templated on the *field* rather than fixed to 2D, so the same
//  case can serve euler_2d and euler_3d when its definition is dimension
//  agnostic (Sedov, a free stream, a Riemann problem). Cases that only make
//  sense in one dimension simply register for that one.
//
//  It is templated on the equation of state as well. Unlike the flux kernel,
//  which is generic, the registry has to type-erase the initial state and the
//  boundary conditions behind a std::function, so their signature must name one
//  concrete state law. Every monofluid case is an ideal gas, hence the default;
//  a two-phase model would instantiate its own registry on StiffenedGas.
//
//  Registration is explicit: each case exposes `register_me<Field>()` and
//  `register_all<Field>()` in cases.hpp lists them. That costs one line per case
//  compared to a self-registering macro, and buys a single readable place where
//  the available cases — and the dimension each supports — can be read off.
// =============================================================================

namespace test_case
{
    template <class Field>
    using BoxFunc = std::function<samurai::Box<double, Field::dim>()>;

    template <class Field, class Eos>
    using InitFunc = std::function<void(Field&, const typename Field::cell_t&, const Eos&)>;

    template <class Field, class Eos>
    using BCFunc = std::function<void(Field&, double&, const Eos&)>;

    template <class Field, class Eos = EOS::IdealGas>
    struct TestCase
    {
        BoxFunc<Field> box;
        InitFunc<Field, Eos> init;
        BCFunc<Field, Eos> bc;

        // Default gas for this case. `--gamma` on the command line overrides it.
        Eos eos = {};

        // Periodicity per axis. Set on the mesh, not on the field: samurai wraps
        // the ghost update rather than attaching a boundary condition.
        std::array<bool, Field::dim> periodic = {};
    };

    template <class Field, class Eos = EOS::IdealGas>
    class TestCaseRegistry
    {
      public:

        using test_case_t = TestCase<Field, Eos>;

        static TestCaseRegistry& instance()
        {
            static TestCaseRegistry registry;
            return registry;
        }

        void register_test_case(const std::string& name, test_case_t test_case)
        {
            test_cases_[name] = std::move(test_case);
        }

        const test_case_t& get(const std::string& name) const
        {
            auto it = test_cases_.find(name);
            if (it == test_cases_.end())
            {
                throw std::runtime_error("Test case '" + name + "' not found");
            }
            return it->second;
        }

        std::vector<std::string> available_test_cases() const
        {
            std::vector<std::string> names;
            names.reserve(test_cases_.size());
            for (const auto& [name, _] : test_cases_)
            {
                names.push_back(name);
            }
            return names;
        }

      private:

        std::map<std::string, test_case_t> test_cases_;
    };

    template <class Field, class Eos = EOS::IdealGas>
    void register_test_case(const std::string& name, TestCase<Field, Eos> test_case)
    {
        TestCaseRegistry<Field, Eos>::instance().register_test_case(name, std::move(test_case));
    }
}
