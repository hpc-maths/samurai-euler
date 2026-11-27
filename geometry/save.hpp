#pragma once

#include <samurai/io/hdf5.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

#include "grid.hpp"
#include "../euler/variables.hpp"

namespace fs = std::filesystem;

// VTK file of the physical cells with u, its error against the exact
// solution, the level, the mean Jacobian and the largest angle of the quads
// of the geometry grid inside the cell. Each cell is the polygon used by
// the scheme, so coarse cells follow the curved image of the grid lines.
template <class Field>
void save_physical(const fs::path& file, const Field& q, const PolysquareMap& map, EOS::IdealGas eos)
{
    std::vector<point_t> points;
    std::vector<std::size_t> offsets = {0};
    // std::vector<double> error;
    std::vector<double> jacobian;
    std::vector<double> max_angle;
    std::vector<std::size_t> level;

    std::vector<double> rho;
    std::vector<double> prs;
    std::vector<double> u;
    std::vector<double> v;

    samurai::for_each_cell(q.mesh(),
                           [&](const auto& cell)
                           {
                               auto poly = cell_polygon(map, cell);
                               points.insert(points.end(), poly.begin(), poly.end());
                               offsets.push_back(points.size());
                               const double jac = mean_jacobian(map, cell);
                            //    auto c           = cell.center();
                              //  error.push_back(u.back() - exact(map(c(0), c(1))));
                               jacobian.push_back(jac);
                               max_angle.push_back(map.max_angle(cell_box(map, cell)));
                               level.push_back(cell.level);

                               const PrimState<2> prim = cons2prim<2>(q[cell] / jac, eos); 
                               rho.push_back(prim.rho);
                               prs.push_back(prim.p);
                               u.push_back(prim.v[0]);
                               v.push_back(prim.v[1]);
                           });

    // VTK XML unstructured grid with the arrays appended as raw binary
    const std::size_t n = u.size();
    std::vector<float> coords;
    coords.reserve(3 * points.size());
    for (const auto& p : points)
    {
        coords.insert(coords.end(), {static_cast<float>(p[0]), static_cast<float>(p[1]), 0.f});
    }
    std::vector<std::int32_t> connectivity(points.size());
    std::iota(connectivity.begin(), connectivity.end(), 0);
    std::vector<std::int32_t> cell_ends(offsets.begin() + 1, offsets.end());
    std::vector<std::uint8_t> types(n, 7); // VTK_POLYGON
    auto to_float = [](const auto& values)
    {
        return std::vector<float>(values.begin(), values.end());
    };
    std::vector<std::int32_t> levels(level.begin(), level.end());

    std::string header;
    std::vector<std::pair<const char*, std::size_t>> blocks; // data, size in bytes
    std::size_t offset = 0;
    auto add_array     = [&](const std::string& attributes, const auto& values)
    {
        using value_t = typename std::decay_t<decltype(values)>::value_type;
        header += fmt::format("<DataArray {} format=\"appended\" offset=\"{}\"/>\n", attributes, offset);
        blocks.emplace_back(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(value_t));
        offset += sizeof(std::uint64_t) + values.size() * sizeof(value_t);
    };

    auto rho_f      = to_float(rho);
    auto prs_f      = to_float(prs);
    auto u_f        = to_float(u);
    auto v_f        = to_float(v);

    // auto error_f    = to_float(error);
    auto jacobian_f = to_float(jacobian);
    auto angle_f    = to_float(max_angle);
    header += fmt::format("<Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n<Points>\n", points.size(), n);
    add_array("type=\"Float32\" NumberOfComponents=\"3\"", coords);
    header += "</Points>\n<Cells>\n";
    add_array("type=\"Int32\" Name=\"connectivity\"", connectivity);
    add_array("type=\"Int32\" Name=\"offsets\"", cell_ends);
    add_array("type=\"UInt8\" Name=\"types\"", types);
    header += "</Cells>\n<CellData Scalars=\"PrimState\">\n";

    add_array("type=\"Float32\" Name=\"rho\"", rho_f);
    add_array("type=\"Float32\" Name=\"p\"", prs_f);
    add_array("type=\"Float32\" Name=\"u\"", u_f);
    add_array("type=\"Float32\" Name=\"v\"", v_f);

    add_array("type=\"Float32\" Name=\"jacobian\"", jacobian_f);
    add_array("type=\"Float32\" Name=\"max_angle\"", angle_f);
    add_array("type=\"Int32\" Name=\"level\"", levels);
    header += "</CellData>\n</Piece>\n";

    std::ofstream out(file, std::ios::binary);
    out << "<?xml version=\"1.0\"?>\n";
    out << fmt::format("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"{}\" header_type=\"UInt64\">\n",
                       std::endian::native == std::endian::little ? "LittleEndian" : "BigEndian");
    out << "<UnstructuredGrid>\n" << header << "</UnstructuredGrid>\n<AppendedData encoding=\"raw\">\n_";
    for (const auto& [data, size] : blocks)
    {
        const std::uint64_t nbytes = size;
        out.write(reinterpret_cast<const char*>(&nbytes), sizeof(nbytes));
        out.write(data, static_cast<std::streamsize>(size));
    }
    out << "\n</AppendedData>\n</VTKFile>\n";
}

// VTK file of every cell of the reference mesh, the cells of the domain and
// their ghosts, each drawn with its physical polygon from the extended map.
// u holds the physical state, ghosts included. Fields: ghost (0 or 1), its
// layer, the level, the Jacobian of the extended map, the Jacobian given by
// the mirror cell (what mean_jacobian uses), the signed physical area (a
// folded ghost would have a negative one) and the primitive state.
template <class Field>
void save_cells_and_ghosts(const fs::path& file, const Field& u, const PolysquareMap& map, EOS::IdealGas eos)
{
    using mesh_t = typename Field::mesh_t;
    const auto& mesh = u.mesh();

    std::vector<point_t> points;
    std::vector<std::size_t> offsets = {0};
    std::vector<std::int32_t> ghost, layer, level;
    std::vector<double> jacobian, jacobian_mirror, area, rho, prs, vx, vy;

    auto add = [&](const auto& cell, long l)
    {
        auto poly = cell_polygon(map, cell);
        points.insert(points.end(), poly.begin(), poly.end());
        offsets.push_back(points.size());
        ghost.push_back(l > 0 ? 1 : 0);
        layer.push_back(static_cast<std::int32_t>(l));
        level.push_back(static_cast<std::int32_t>(cell.level));
        const double signed_area = polygon_area(poly);
        area.push_back(signed_area);
        jacobian.push_back(std::abs(signed_area) / (cell.length * cell.length));
        jacobian_mirror.push_back(std::abs(polygon_area(map.polygon(map.mirror_box(cell_box(map, cell))))) / (cell.length * cell.length));
        const PrimState<2> prim = cons2prim<2>(u[cell], eos);
        rho.push_back(prim.rho);
        prs.push_back(prim.p);
        vx.push_back(prim.v[0]);
        vy.push_back(prim.v[1]);
    };
    // the leaves of the domain, then every ghost of the reference mesh
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               add(cell, 0);
                           });
    samurai::for_each_cell(mesh[mesh_t::mesh_id_t::reference],
                           [&](const auto& cell)
                           {
                               if (const long l = ghost_layer(map, cell_box(map, cell)); l > 0)
                               {
                                   add(cell, l);
                               }
                           });

    const std::size_t n = ghost.size();
    std::vector<float> coords;
    coords.reserve(3 * points.size());
    for (const auto& p : points)
    {
        coords.insert(coords.end(), {static_cast<float>(p[0]), static_cast<float>(p[1]), 0.f});
    }
    std::vector<std::int32_t> connectivity(points.size());
    std::iota(connectivity.begin(), connectivity.end(), 0);
    std::vector<std::int32_t> cell_ends(offsets.begin() + 1, offsets.end());
    std::vector<std::uint8_t> types(n, 7); // VTK_POLYGON
    auto to_float = [](const auto& values)
    {
        return std::vector<float>(values.begin(), values.end());
    };

    std::string header;
    std::vector<std::pair<const char*, std::size_t>> blocks;
    std::size_t offset = 0;
    auto add_array     = [&](const std::string& attributes, const auto& values)
    {
        using value_t = typename std::decay_t<decltype(values)>::value_type;
        header += fmt::format("<DataArray {} format=\"appended\" offset=\"{}\"/>\n", attributes, offset);
        blocks.emplace_back(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(value_t));
        offset += sizeof(std::uint64_t) + values.size() * sizeof(value_t);
    };
    auto jac_f = to_float(jacobian), jacm_f = to_float(jacobian_mirror), area_f = to_float(area);
    auto rho_f = to_float(rho), prs_f = to_float(prs), vx_f = to_float(vx), vy_f = to_float(vy);

    header += fmt::format("<Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n<Points>\n", points.size(), n);
    add_array("type=\"Float32\" NumberOfComponents=\"3\"", coords);
    header += "</Points>\n<Cells>\n";
    add_array("type=\"Int32\" Name=\"connectivity\"", connectivity);
    add_array("type=\"Int32\" Name=\"offsets\"", cell_ends);
    add_array("type=\"UInt8\" Name=\"types\"", types);
    header += "</Cells>\n<CellData Scalars=\"ghost\">\n";
    add_array("type=\"Int32\" Name=\"ghost\"", ghost);
    add_array("type=\"Int32\" Name=\"layer\"", layer);
    add_array("type=\"Int32\" Name=\"level\"", level);
    add_array("type=\"Float32\" Name=\"jacobian\"", jac_f);
    add_array("type=\"Float32\" Name=\"jacobian_mirror\"", jacm_f);
    add_array("type=\"Float32\" Name=\"signed_area\"", area_f);
    add_array("type=\"Float32\" Name=\"rho\"", rho_f);
    add_array("type=\"Float32\" Name=\"p\"", prs_f);
    add_array("type=\"Float32\" Name=\"u\"", vx_f);
    add_array("type=\"Float32\" Name=\"v\"", vy_f);
    header += "</CellData>\n</Piece>\n";

    std::ofstream out(file, std::ios::binary);
    out << "<?xml version=\"1.0\"?>\n";
    out << fmt::format("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"{}\" header_type=\"UInt64\">\n",
                       std::endian::native == std::endian::little ? "LittleEndian" : "BigEndian");
    out << "<UnstructuredGrid>\n" << header << "</UnstructuredGrid>\n<AppendedData encoding=\"raw\">\n_";
    for (const auto& [data, size] : blocks)
    {
        const std::uint64_t nbytes = size;
        out.write(reinterpret_cast<const char*>(&nbytes), sizeof(nbytes));
        out.write(data, static_cast<std::streamsize>(size));
    }
    out << "\n</AppendedData>\n</VTKFile>\n";
}

// ParaView collection that gives each VTK file its physical time
void save_collection(const fs::path& file, const std::vector<std::pair<double, std::string>>& steps)
{
    std::ofstream out(file);
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n";
    for (const auto& [time, name] : steps)
    {
        out << fmt::format("<DataSet timestep=\"{}\" file=\"{}\"/>\n", time, name);
    }
    out << "</Collection>\n</VTKFile>\n";
}