// Copyright 2025 the samurai team
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <type_traits>

#include <samurai/samurai_config.hpp>

// Forces the inlining of the small functions the flux calls for every face.
// Recent versions of samurai define the macro; this is the fallback for the others.
#ifndef SAMURAI_ALWAYS_INLINE
#if defined(_MSC_VER)
#define SAMURAI_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define SAMURAI_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define SAMURAI_ALWAYS_INLINE inline
#endif
#endif
