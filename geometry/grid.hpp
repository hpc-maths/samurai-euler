#pragma once

#include <cmath>
#include <deque>

// constexpr double bmin = -.5;
// constexpr double bmax =  .5;

using point_t = xt::xtensor_fixed<double, xt::xshape<2>>;
using MapFunction = std::function<point_t(const point_t&)>;

point_t radial_map(const point_t& p)
{
    constexpr double epsilon = 1e-10;
    const double ksi = std::max(std::abs(p[0]), std::abs(p[1]));
    const double r   = std::hypot(p[0], p[1]);
    return p * ksi / std::max(r, epsilon);
}

point_t cartesian_map(const point_t& p)
{
    return p;
}

//

double cross(const point_t& a, const point_t& b)
{
    return a[0] * b[1] - a[1] * b[0];
}

// Flux of the velocity through the segment [a, b], with the normal on the
// right of a -> b. The velocity is linear, so the midpoint rule is exact.
point_t oriented_surface(const point_t& a, const point_t& b)
{
    const point_t tj = b - a;
    return {tj[1], -tj[0]};
}

auto get_norm_unit(const point_t& p) {
  const double len = std::hypot(p[0], p[1]);
  return std::make_pair(len, point_t(p / len));
}


// Square of parametric cells [i0, i0 + n) x [j0, j0 + n) of the geometry grid
struct Box
{
    long i0;
    long j0;
    long n;
};

// Piecewise bilinear map of the parametric grid of a given level onto the
// annulus. Vertex (i, j) is the parametric point (-2 + i h, -2 + j h) and
// the hole (-1, 1)^2 is [N / 4, 3 N / 4]^2 in indices. Without the hole, the
// parametric domain is the whole square and the map sends it onto the disc
// of radius 2.
class PolysquareMap
{
  public:

    PolysquareMap(std::size_t level, double h, bool hole, const MapFunction& map_func)
        : m_level(level)
        , m_hole(hole)
        , m_h(h)
        , m_n(std::lround(4. / h))
        , m_x(static_cast<std::size_t>((m_n + 1) * (m_n + 1)), point_t{std::nan(""), std::nan("")})
        , m_func(map_func)
    {
        if (m_n < 4 || m_n % 4 != 0)
        {
            throw std::runtime_error("the geometry grid must align with the hole");
        }
    }

    static PolysquareMap analytic(std::size_t level, double h, bool hole, const MapFunction& map_func)
    {
        PolysquareMap map(level, h, hole, map_func);
        for (long j = 0; j <= map.m_n; ++j)
        {
            for (long i = 0; i <= map.m_n; ++i)
            {
                if (map.vertex_in_domain(i, j))
                {
                    map.vertex(i, j) = map.m_func({map.coordinate(i), map.coordinate(j)});
                }
            }
        }
        return map;
    }

    // Same map on the grid of the next level: bilinear interpolation, then
    // the new boundary vertices are pushed back onto their circle.
    PolysquareMap refined() const
    {
        PolysquareMap fine(m_level + 1, 0.5 * m_h, m_hole, m_func);
        for (long j = 0; j <= fine.m_n; ++j)
        {
            for (long i = 0; i <= fine.m_n; ++i)
            {
                if (!fine.vertex_in_domain(i, j))
                {
                    continue;
                }
                const long ic = i / 2;
                const long jc = j / 2;
                const long di = i % 2;
                const long dj = j % 2;
                point_t x     = {0, 0};
                for (long b = 0; b <= dj; ++b)
                {
                    for (long a = 0; a <= di; ++a)
                    {
                        const double w = (di ? 0.5 : 1.) * (dj ? 0.5 : 1.);
                        x[0] += w * vertex(ic + a, jc + b)[0];
                        x[1] += w * vertex(ic + a, jc + b)[1];
                    }
                }
                if (const double r = fine.boundary_radius(i, j); r > 0)
                {
                    const double norm = std::hypot(x[0], x[1]);
                    x                 = {r * x[0] / norm, r * x[1] / norm};
                }
                fine.vertex(i, j) = x;
            }
        }
        return fine;
    }

    std::size_t level() const
    {
        return m_level;
    }

    double step() const
    {
        return m_h;
    }

    long size() const
    {
        return m_n;
    }

    double coordinate(long i) const
    {
        return -2. + static_cast<double>(i) * m_h;
    }

    long index(double s) const
    {
        return std::lround((s + 2.) / m_h);
    }

    const point_t& vertex(long i, long j) const
    {
        return m_x[static_cast<std::size_t>(j * (m_n + 1) + i)];
    }

    point_t& vertex(long i, long j)
    {
        return m_x[static_cast<std::size_t>(j * (m_n + 1) + i)];
    }

    bool vertex_in_domain(long i, long j) const
    {
        const long a = m_n / 4;
        const long b = 3 * m_n / 4;
        return i >= 0 && i <= m_n && j >= 0 && j <= m_n && !(m_hole && i > a && i < b && j > a && j < b);
    }

    bool quad_in_domain(long i, long j) const
    {
        const long a = m_n / 4;
        const long b = 3 * m_n / 4;
        return i >= 0 && i < m_n && j >= 0 && j < m_n && !(m_hole && i >= a && i < b && j >= a && j < b);
    }

    // Radius of the circle carrying a boundary vertex, 0 inside the domain
    double boundary_radius(long i, long j) const
    {
        if (i == 0 || i == m_n || j == 0 || j == m_n)
        {
            return 2.;
        }
        const long a = m_n / 4;
        const long b = 3 * m_n / 4;
        if (m_hole && i >= a && i <= b && j >= a && j <= b)
        {
            return 1.;
        }
        return 0.;
    }

    // The corners of the polysquare stay at their analytic position
    bool is_corner(long i, long j) const
    {
        auto extreme = [this](long k)
        {
            return k == 0 || k == m_n;
        };
        auto inner = [this](long k)
        {
            return k == m_n / 4 || k == 3 * m_n / 4;
        };
        return (extreme(i) && extreme(j)) || (m_hole && inner(i) && inner(j));
    }

    point_t operator()(double s, double t) const
    {
        const double fs = (s + 2.) / m_h;
        const double ft = (t + 2.) / m_h;
        const long i    = std::clamp(static_cast<long>(std::floor(fs)), 0L, m_n - 1);
        const long j    = std::clamp(static_cast<long>(std::floor(ft)), 0L, m_n - 1);
        const double u  = fs - static_cast<double>(i);
        const double v  = ft - static_cast<double>(j);
        const auto& a   = vertex(i, j);
        const auto& b   = vertex(i + 1, j);
        const auto& c   = vertex(i + 1, j + 1);
        const auto& d   = vertex(i, j + 1);
        return {(1 - u) * (1 - v) * a[0] + u * (1 - v) * b[0] + u * v * c[0] + (1 - u) * v * d[0],
                (1 - u) * (1 - v) * a[1] + u * (1 - v) * b[1] + u * v * c[1] + (1 - u) * v * d[1]};
    }

    // Boundary of a box, counterclockwise
    std::vector<point_t> polygon(const Box& box) const
    {
        std::vector<point_t> poly;
        poly.reserve(static_cast<std::size_t>(4 * box.n));
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(vertex(box.i0 + k, box.j0));
        }
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(vertex(box.i0 + box.n, box.j0 + k));
        }
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(vertex(box.i0 + box.n - k, box.j0 + box.n));
        }
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(vertex(box.i0, box.j0 + box.n - k));
        }
        return poly;
    }

    // Physical polygon of a box, counterclockwise in the parametric order of
    // polygon(): the image of the box by the map if it lies in the domain,
    // otherwise the geometry of a ghost cell.
    //
    // A ghost cell is the isometric image of the cell of the domain that
    // mirrors it (see ghost_image):
    //  - across one wall, its reflection across the line of the wall face of
    //    its column, so that the face it shares with the domain is exactly
    //    the wall face;
    //  - at a corner of the polysquare, where it is mirrored across two
    //    walls, its point reflection through the corner. The two parametric
    //    reflections keep the orientation, a physical reflection would turn
    //    the cell over.
    // The ghost then has the shape and the Jacobian of its mirror, and its
    // centre is the physical mirror image of the centre of the mirror cell.
    // Its geometry depends only on the box, never on the direction from which
    // it is reached: the ghost of the corner of the hole is the same cell
    // whether it is the +x neighbour of one cell or the +y neighbour of
    // another. Ghost cells near a corner can overlap: the parametric angle
    // around an outer corner is 90 degrees inside and 270 outside, the
    // physical one 180 on each side.
    std::vector<point_t> physical_polygon(const Box& box) const
    {
        if (quad_in_domain(box.i0, box.j0))
        {
            return polygon(box);
        }
        const auto img = ghost_image(box);
        auto mirror    = [&](long i, long j) -> point_t
        {
            const point_t& x = vertex(img.reflect_i ? 2 * img.wall_i - i : i, img.reflect_j ? 2 * img.wall_j - j : j);
            if (img.reflect_i && img.reflect_j)
            {
                return 2 * img.corner - x;
            }
            const double d = (x[0] - img.origin[0]) * img.normal[0] + (x[1] - img.origin[1]) * img.normal[1];
            return x - 2 * d * img.normal;
        };
        std::vector<point_t> poly;
        poly.reserve(static_cast<std::size_t>(4 * box.n));
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(mirror(box.i0 + k, box.j0));
        }
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(mirror(box.i0 + box.n, box.j0 + k));
        }
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(mirror(box.i0 + box.n - k, box.j0 + box.n));
        }
        for (long k = 0; k < box.n; ++k)
        {
            poly.push_back(mirror(box.i0, box.j0 + box.n - k));
        }
        return poly;
    }

    // Outward unit normal of the boundary edge (i, j) - (i1, j1), pointing out
    // of the quad of the domain that has this edge; {0, 0} if the edge is not
    // on the wall (both quads, or none, in the domain).
    point_t wall_edge_normal(long i, long j, long i1, long j1) const
    {
        const long di  = i1 - i;
        const long dj  = j1 - j;
        const long qi0 = std::min(i, i1) - (dj != 0 ? 1 : 0);
        const long qj0 = std::min(j, j1) - (di != 0 ? 1 : 0);
        const long qi1 = std::min(i, i1);
        const long qj1 = std::min(j, j1);
        const bool in0 = quad_in_domain(qi0, qj0);
        if (in0 == quad_in_domain(qi1, qj1))
        {
            return {0, 0};
        }
        const long qi       = in0 ? qi0 : qi1;
        const long qj       = in0 ? qj0 : qj1;
        const point_t inner = 0.25 * (vertex(qi, qj) + vertex(qi + 1, qj) + vertex(qi + 1, qj + 1) + vertex(qi, qj + 1));
        const point_t t     = vertex(i1, j1) - vertex(i, j);
        point_t n           = point_t{t[1], -t[0]} / std::hypot(t[0], t[1]);
        const point_t mid   = 0.5 * (vertex(i, j) + vertex(i1, j1));
        if ((mid[0] - inner[0]) * n[0] + (mid[1] - inner[1]) * n[1] < 0)
        {
            n = -n;
        }
        return n;
    }

    // Signed curvature of the wall at the wall vertex (i, j): the inverse
    // radius of the circle through the vertex and its two neighbours along
    // the wall, positive when the wall is concave seen from the fluid (the
    // outer circle), negative when it is convex (the inner circle). The
    // neighbours along the wall are found by the boundary edges, so the
    // corners of the polysquare need no special case.
    double wall_curvature(long i, long j) const
    {
        std::array<point_t, 2> nb;
        point_t n_out   = {0, 0};
        std::size_t cnt = 0;
        const std::array<std::pair<long, long>, 4> dirs = {
            {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
        };
        for (const auto& [di, dj] : dirs)
        {
            if (!vertex_in_domain(i + di, j + dj))
            {
                continue;
            }
            const point_t ne = wall_edge_normal(i, j, i + di, j + dj);
            if (ne[0] == 0 && ne[1] == 0)
            {
                continue;
            }
            if (cnt < 2)
            {
                nb[cnt] = vertex(i + di, j + dj);
            }
            ++cnt;
            n_out += ne;
        }
        if (cnt != 2)
        {
            return 0.;
        }
        const point_t& a = nb[0];
        const point_t& b = vertex(i, j);
        const point_t& c = nb[1];
        // circumcentre of a, b, c
        const double d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
        if (std::abs(d) < 1e-300)
        {
            return 0.;
        }
        const double a2      = a[0] * a[0] + a[1] * a[1];
        const double b2      = b[0] * b[0] + b[1] * b[1];
        const double c2      = c[0] * c[0] + c[1] * c[1];
        const point_t centre = {(a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d,
                                (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d};
        const double radius  = std::hypot(b[0] - centre[0], b[1] - centre[1]);
        const double side    = (centre[0] - b[0]) * n_out[0] + (centre[1] - b[1]) * n_out[1];
        return side < 0 ? 1. / radius : -1. / radius;
    }

    // Mean curvature of the wall over the wall face of a box in direction d
    // (0: the face along j at index i_wall, 1: the face along i at j_wall)
    double face_curvature(long i0, long j0, long n, std::size_t d) const
    {
        double kappa = 0;
        for (long k = 0; k <= n; ++k)
        {
            kappa += d == 0 ? wall_curvature(i0, j0 + k) : wall_curvature(i0 + k, j0);
        }
        return kappa / static_cast<double>(n + 1);
    }

    // Curvature of the wall seen by a ghost box: along the wall face of its
    // column for a single reflection, at the corner for a point reflection.
    double ghost_curvature(const Box& box) const
    {
        const auto img = ghost_image(box);
        if (img.reflect_i && img.reflect_j)
        {
            return wall_curvature(img.wall_i, img.wall_j);
        }
        return img.reflect_i ? face_curvature(img.wall_i, box.j0, box.n, 0) : face_curvature(box.i0, img.wall_j, box.n, 1);
    }

    // The box itself if it lies in the domain, otherwise the cell of the
    // domain that mirrors it
    Box mirror_box(const Box& box) const
    {
        if (quad_in_domain(box.i0, box.j0))
        {
            return box;
        }
        const auto img = ghost_image(box);
        return {img.reflect_i ? 2 * img.wall_i - box.i0 - box.n : box.i0, img.reflect_j ? 2 * img.wall_j - box.j0 - box.n : box.j0, box.n};
    }

    // How a ghost box is mirrored into the domain: which indices are
    // reflected and across which walls, and the physical isometry.
    struct GhostImage
    {
        bool reflect_i = false;
        bool reflect_j = false;
        long wall_i    = 0;
        long wall_j    = 0;
        point_t origin = {0, 0}; // a point of the wall line (single reflection)
        point_t normal = {0, 0}; // unit normal of the wall line (single reflection)
        point_t corner = {0, 0}; // centre of the point reflection (corner)
    };

    // Outside the polysquare, each index is reflected across the side it
    // overshoots, both at a corner. Inside the hole, the box is reflected
    // across its nearest wall, and across both walls when they are equally
    // near, as on the diagonals of the hole. One reflection must bring the box
    // into the domain.
    GhostImage ghost_image(const Box& box) const
    {
        const long a = m_n / 4;
        const long b = 3 * m_n / 4;
        GhostImage img;
        if (box.i0 < 0 || box.i0 >= m_n || box.j0 < 0 || box.j0 >= m_n)
        {
            auto outer = [&](long i0, bool& reflect, long& wall)
            {
                if (i0 < 0 || i0 >= m_n)
                {
                    reflect = true;
                    wall    = i0 < 0 ? 0 : m_n;
                }
            };
            outer(box.i0, img.reflect_i, img.wall_i);
            outer(box.j0, img.reflect_j, img.wall_j);
        }
        else
        {
            // inside the hole
            const long di_low  = box.i0 - a;
            const long di_high = b - box.i0 - box.n;
            const long dj_low  = box.j0 - a;
            const long dj_high = b - box.j0 - box.n;
            const long di      = std::min(di_low, di_high);
            const long dj      = std::min(dj_low, dj_high);
            img.reflect_i      = di <= dj;
            img.reflect_j      = dj <= di;
            img.wall_i         = di_low <= di_high ? a : b;
            img.wall_j         = dj_low <= dj_high ? a : b;
        }
        const long mi0 = img.reflect_i ? 2 * img.wall_i - box.i0 - box.n : box.i0;
        const long mj0 = img.reflect_j ? 2 * img.wall_j - box.j0 - box.n : box.j0;
        if (!quad_in_domain(mi0, mj0))
        {
            throw std::runtime_error(fmt::format("ghost box ({}, {}, {}) too far from the domain to mirror it", box.i0, box.j0, box.n));
        }
        if (img.reflect_i && img.reflect_j)
        {
            img.corner = vertex(img.wall_i, img.wall_j);
            return img;
        }
        // the wall face of the column (or row) of the box
        const point_t p0 = img.reflect_i ? vertex(img.wall_i, box.j0) : vertex(box.i0, img.wall_j);
        const point_t p1 = img.reflect_i ? vertex(img.wall_i, box.j0 + box.n) : vertex(box.i0 + box.n, img.wall_j);
        const point_t t  = p1 - p0;
        img.origin       = p0;
        img.normal       = point_t{-t[1], t[0]} / std::hypot(t[0], t[1]);
        return img;
    }

    // Largest interior angle of the quads of the geometry grid in a box, in degrees
    double max_angle(const Box& box) const
    {
        double angle = 0;
        for (long j = box.j0; j < box.j0 + box.n; ++j)
        {
            for (long i = box.i0; i < box.i0 + box.n; ++i)
            {
                for (const auto& [a, b] : corner_edges(i, j))
                {
                    angle = std::max(angle, std::atan2(cross(a, b), a[0] * b[0] + a[1] * b[1]));
                }
            }
        }
        return angle * 180. / std::numbers::pi;
    }

    // Edges leaving each corner of quad (i, j), counterclockwise then clockwise.
    // [a b] / h is the Jacobian of the corner up to a rotation of the
    // parametric frame.
    std::array<std::pair<point_t, point_t>, 4> corner_edges(long i, long j) const
    {
        const std::array<point_t, 4> p = {vertex(i, j), vertex(i + 1, j), vertex(i + 1, j + 1), vertex(i, j + 1)};
        std::array<std::pair<point_t, point_t>, 4> edges;
        for (std::size_t c = 0; c < 4; ++c)
        {
            const auto& x    = p[c];
            const auto& next = p[(c + 1) % 4];
            const auto& prev = p[(c + 3) % 4];
            edges[c]         = {
                {next[0] - x[0], next[1] - x[1]},
                {prev[0] - x[0], prev[1] - x[1]}
            };
        }
        return edges;
    }

  private:

    std::size_t m_level;

    bool m_hole;
    double m_h;
    long m_n;
    std::vector<point_t> m_x;
  public:
    MapFunction m_func;
};

// Garanzha's foldover-free energy of the map, summed over the corners of the
// quads: f(J) = (1 - theta) tr(J^T J) / (2 det J) + theta (det J / V + V / det J) / 2,
// with V = |annulus| / |polysquare| = pi / 4. It is infinite as soon as a
// corner folds, so a descent started from a valid map stays valid.
class MapEnergy
{
  public:

    MapEnergy(PolysquareMap& map, double theta)
        : m_map(map)
        , m_theta(theta)
        , m_first(static_cast<std::size_t>((map.size() + 1) * (map.size() + 1)), -1)
    {
        const long n = map.size();
        for (long j = 0; j <= n; ++j)
        {
            for (long i = 0; i <= n; ++i)
            {
                if (!map.vertex_in_domain(i, j) || map.is_corner(i, j))
                {
                    continue;
                }
                m_first[flat(i, j)] = m_size;
                m_size += (map.boundary_radius(i, j) > 0) ? 1 : 2;
            }
        }
    }

    std::size_t size() const
    {
        return static_cast<std::size_t>(m_size);
    }

    // Free interior vertices carry (x, y), sliding boundary vertices their angle
    std::vector<double> dofs() const
    {
        std::vector<double> x(size());
        for_each_free_vertex(
            [&](long i, long j, std::size_t k, double r)
            {
                const auto& p = m_map.vertex(i, j);
                if (r > 0)
                {
                    x[k] = std::atan2(p[1], p[0]);
                }
                else
                {
                    x[k]     = p[0];
                    x[k + 1] = p[1];
                }
            });
        return x;
    }

    void set_dofs(const std::vector<double>& x)
    {
        for_each_free_vertex(
            [&](long i, long j, std::size_t k, double r)
            {
                m_map.vertex(i, j) = (r > 0) ? point_t{r * std::cos(x[k]), r * std::sin(x[k])} : point_t{x[k], x[k + 1]};
            });
    }

    // Energy of the map for the given dofs and its gradient
    double operator()(const std::vector<double>& x, std::vector<double>& grad)
    {
        set_dofs(x);
        const long n   = m_map.size();
        const double h = m_map.step();
        const double V = std::numbers::pi / 4;
        std::vector<point_t> g(m_first.size(), point_t{0, 0});
        double energy = 0;
        for (long j = 0; j < n; ++j)
        {
            for (long i = 0; i < n; ++i)
            {
                if (!m_map.quad_in_domain(i, j))
                {
                    continue;
                }
                const std::array<std::pair<long, long>, 4> ids = {
                    {{i, j}, {i + 1, j}, {i + 1, j + 1}, {i, j + 1}}
                };
                const auto edges = m_map.corner_edges(i, j);
                for (std::size_t c = 0; c < 4; ++c)
                {
                    const auto& [a, b] = edges[c];
                    const double det   = cross(a, b) / (h * h);
                    if (det <= 0)
                    {
                        return std::numeric_limits<double>::infinity();
                    }
                    const double tr = (a[0] * a[0] + a[1] * a[1] + b[0] * b[0] + b[1] * b[1]) / (h * h);
                    energy += (1 - m_theta) * tr / (2 * det) + m_theta * (det / V + V / det) / 2;

                    // derivatives with respect to tr and det, then to a and b
                    const double f_tr  = (1 - m_theta) / (2 * det);
                    const double f_det = -(1 - m_theta) * tr / (2 * det * det) + m_theta * (1 / V - V / (det * det)) / 2;
                    const point_t ga   = {(2 * f_tr * a[0] + f_det * b[1]) / (h * h), (2 * f_tr * a[1] - f_det * b[0]) / (h * h)};
                    const point_t gb   = {(2 * f_tr * b[0] - f_det * a[1]) / (h * h), (2 * f_tr * b[1] + f_det * a[0]) / (h * h)};
                    auto add           = [&](std::pair<long, long> id, const point_t& v, double sign)
                    {
                        auto& gi = g[flat(id.first, id.second)];
                        gi[0] += sign * v[0];
                        gi[1] += sign * v[1];
                    };
                    add(ids[(c + 1) % 4], ga, 1);
                    add(ids[(c + 3) % 4], gb, 1);
                    add(ids[c], ga, -1);
                    add(ids[c], gb, -1);
                }
            }
        }
        grad.assign(size(), 0.);
        for_each_free_vertex(
            [&](long i, long j, std::size_t k, double r)
            {
                const auto& gi = g[flat(i, j)];
                if (r > 0)
                {
                    const double angle = x[k];
                    grad[k]            = r * (-std::sin(angle) * gi[0] + std::cos(angle) * gi[1]);
                }
                else
                {
                    grad[k]     = gi[0];
                    grad[k + 1] = gi[1];
                }
            });
        return energy;
    }

  private:

    std::size_t flat(long i, long j) const
    {
        return static_cast<std::size_t>(j * (m_map.size() + 1) + i);
    }

    template <class Func>
    void for_each_free_vertex(Func&& f) const
    {
        const long n = m_map.size();
        for (long j = 0; j <= n; ++j)
        {
            for (long i = 0; i <= n; ++i)
            {
                if (const long k = m_first[flat(i, j)]; k >= 0)
                {
                    f(i, j, static_cast<std::size_t>(k), m_map.boundary_radius(i, j));
                }
            }
        }
    }

    PolysquareMap& m_map;
    double m_theta;
    std::vector<long> m_first; // first dof of each vertex, -1 if it has none
    long m_size = 0;
};

// L-BFGS with a backtracking line search. The steps are limited to a
// fraction of the grid step, and a step that folds a corner has an infinite
// energy and is shortened like any other rejected step.
std::size_t minimize(MapEnergy& energy, double max_step, std::size_t max_iterations)
{
    constexpr std::size_t memory = 10;
    auto dot                     = [](const std::vector<double>& a, const std::vector<double>& b)
    {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.);
    };

    auto x = energy.dofs();
    std::vector<double> g;
    double e = energy(x, g);
    if (!std::isfinite(e))
    {
        throw std::runtime_error("the initial map folds");
    }
    std::deque<std::pair<std::vector<double>, std::vector<double>>> history; // (s, y)
    std::vector<double> x_new(x.size());
    std::vector<double> g_new;
    std::size_t stalled = 0;
    std::size_t it      = 0;
    for (; it < max_iterations && stalled < 10; ++it)
    {
        // two-loop recursion: d = -H g
        std::vector<double> d(g);
        std::vector<double> alpha(history.size());
        for (std::size_t k = history.size(); k-- > 0;)
        {
            const auto& [s, y] = history[k];
            alpha[k]           = dot(s, d) / dot(y, s);
            std::transform(d.begin(),
                           d.end(),
                           y.begin(),
                           d.begin(),
                           [&](double di, double yi)
                           {
                               return di - alpha[k] * yi;
                           });
        }
        if (!history.empty())
        {
            const auto& [s, y] = history.back();
            const double gamma = dot(s, y) / dot(y, y);
            std::transform(d.begin(),
                           d.end(),
                           d.begin(),
                           [&](double di)
                           {
                               return gamma * di;
                           });
        }
        for (std::size_t k = 0; k < history.size(); ++k)
        {
            const auto& [s, y] = history[k];
            const double beta  = dot(y, d) / dot(y, s);
            std::transform(d.begin(),
                           d.end(),
                           s.begin(),
                           d.begin(),
                           [&](double di, double si)
                           {
                               return di + (alpha[k] - beta) * si;
                           });
        }
        std::transform(d.begin(),
                       d.end(),
                       d.begin(),
                       [](double di)
                       {
                           return -di;
                       });
        double slope = dot(g, d);
        if (slope >= 0)
        {
            history.clear();
            std::transform(g.begin(),
                           g.end(),
                           d.begin(),
                           [](double gi)
                           {
                               return -gi;
                           });
            slope = dot(g, d);
        }

        const double d_max = std::abs(*std::max_element(d.begin(),
                                                        d.end(),
                                                        [](double a, double b)
                                                        {
                                                            return std::abs(a) < std::abs(b);
                                                        }));
        double step        = std::min(1., max_step / d_max);
        double e_new       = std::numeric_limits<double>::infinity();
        for (int k = 0; k < 60; ++k, step *= 0.5)
        {
            std::transform(x.begin(),
                           x.end(),
                           d.begin(),
                           x_new.begin(),
                           [&](double xi, double di)
                           {
                               return xi + step * di;
                           });
            e_new = energy(x_new, g_new);
            if (e_new <= e + 1e-4 * step * slope)
            {
                break;
            }
        }
        if (!(e_new <= e))
        {
            break;
        }
        stalled = (e - e_new <= 1e-9 * e) ? stalled + 1 : 0;

        std::vector<double> s(x.size());
        std::vector<double> y(x.size());
        std::transform(x_new.begin(), x_new.end(), x.begin(), s.begin(), std::minus<>());
        std::transform(g_new.begin(), g_new.end(), g.begin(), y.begin(), std::minus<>());
        if (dot(s, y) > 1e-14 * std::sqrt(dot(s, s) * dot(y, y)))
        {
            history.emplace_back(std::move(s), std::move(y));
            if (history.size() > memory)
            {
                history.pop_front();
            }
        }
        std::swap(x, x_new);
        std::swap(g, g_new);
        e = e_new;
    }
    energy.set_dofs(x);
    return it;
}

struct MapQuality
{
    double max_angle  = 0; // degrees
    double min_det    = std::numeric_limits<double>::max();
    double max_det    = 0;
    double min_sigma  = std::numeric_limits<double>::max();
    double max_sigma  = 0;
    double max_aspect = 0; // sigma_max / sigma_min
};

// Corner Jacobians of all the quads of the geometry grid
MapQuality map_quality(const PolysquareMap& map)
{
    MapQuality q;
    const double h = map.step();
    for (long j = 0; j < map.size(); ++j)
    {
        for (long i = 0; i < map.size(); ++i)
        {
            if (!map.quad_in_domain(i, j))
            {
                continue;
            }
            for (const auto& [a, b] : map.corner_edges(i, j))
            {
                const double det = cross(a, b) / (h * h);
                const double tr  = (a[0] * a[0] + a[1] * a[1] + b[0] * b[0] + b[1] * b[1]) / (h * h);
                const double rad = std::sqrt(std::max(0., tr * tr - 4 * det * det));
                const double s1  = std::sqrt(0.5 * (tr + rad));
                const double s2  = std::sqrt(std::max(0., 0.5 * (tr - rad)));
                q.max_angle      = std::max(q.max_angle, std::atan2(cross(a, b), a[0] * b[0] + a[1] * b[1]) * 180. / std::numbers::pi);
                q.min_det        = std::min(q.min_det, det);
                q.max_det        = std::max(q.max_det, det);
                q.min_sigma      = std::min(q.min_sigma, s2);
                q.max_sigma      = std::max(q.max_sigma, s1);
                q.max_aspect     = std::max(q.max_aspect, s1 / s2);
            }
        }
    }
    return q;
}

// Analytic map on the grid of the given level, optionally optimized level by
// level from a coarse grid: each level starts from the optimized map of the
// previous one, so that the fine levels only need a local smoothing.
PolysquareMap build_map(std::size_t level, double h, bool hole, bool optimize, double theta, std::size_t max_iterations, const MapFunction& map_func)
{
    if (!optimize)
    {
        return PolysquareMap::analytic(level, h, hole, map_func);
    }
    const std::size_t start = std::min(level, std::size_t{2});
    auto map                = PolysquareMap::analytic(start, h * static_cast<double>(std::size_t{1} << (level - start)), hole, map_func);
    while (true)
    {
        MapEnergy energy(map, theta);
        const auto t0                               = std::chrono::steady_clock::now();
        const auto iterations                       = minimize(energy, 0.25 * map.step(), max_iterations);
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - t0;
        fmt::print("map level {}: {} L-BFGS iterations, {:.2f} s\n", map.level(), iterations, elapsed.count());
        if (map.level() == level)
        {
            return map;
        }
        map = map.refined();
    }
}

// Highest geometry grid of a cell, at the level of the map
template <class Cell>
Box cell_box(const PolysquareMap& map, const Cell& cell)
{
    return {map.index(cell.corner(0)), map.index(cell.corner(1)), 1L << (map.level() - cell.level)};
}

// Physical polygon of a cell, ghost cells included
template <class Cell>
std::vector<point_t> cell_polygon(const PolysquareMap& map, const Cell& cell)
{
    return map.physical_polygon(cell_box(map, cell));
}

double polygon_area(const std::vector<point_t>& poly)
{
    double area = 0;
    for (std::size_t k = 0; k < poly.size(); ++k)
    {
        area += cross(poly[k], poly[(k + 1) % poly.size()]);
    }
    return 0.5 * area;
}

template <class Cell>
double physical_area(const PolysquareMap& map, const Cell& cell)
{
    return polygon_area(cell_polygon(map, cell));
}

// Mean Jacobian |K| / h^2 of a parametric cell. A ghost cell is an
// isometric image of its mirror, so it has the Jacobian of its mirror.
template <class Cell>
double mean_jacobian(const PolysquareMap& map, const Cell& cell)
{
    return std::abs(polygon_area(cell_polygon(map, cell))) / (cell.length * cell.length);
}

// Number of cells of the level of `box` between the box and the domain: 0
// for a cell of the domain, 1 for the first layer of ghosts, and so on.
inline long ghost_layer(const PolysquareMap& map, const Box& box)
{
    const long N = map.size();
    if (map.quad_in_domain(box.i0, box.j0))
    {
        return 0;
    }
    auto outside = [&](long i0)
    {
        return i0 < 0 ? -i0 / box.n : (i0 >= N ? (i0 - N) / box.n + 1 : 0);
    };
    if (const long outer = std::max(outside(box.i0), outside(box.j0)); outer > 0)
    {
        return outer;
    }
    // inside the hole: distance to its nearest wall
    const long a = N / 4;
    const long b = 3 * N / 4;
    return std::min({(box.i0 - a) / box.n + 1, (b - box.i0 - box.n) / box.n + 1, (box.j0 - a) / box.n + 1, (b - box.j0 - box.n) / box.n + 1});
}

// Centroid of a polygon (its vertices in order)
inline point_t polygon_centroid(const std::vector<point_t>& poly)
{
    double a = 0;
    point_t c = {0, 0};
    for (std::size_t k = 0; k < poly.size(); ++k)
    {
        const auto& p = poly[k];
        const auto& q = poly[(k + 1) % poly.size()];
        const double w = cross(p, q);
        a += w;
        c += w * (p + q);
    }
    return c / (3 * a);
}
