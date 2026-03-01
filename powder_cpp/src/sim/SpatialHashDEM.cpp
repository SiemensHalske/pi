#include "powder/sim/SpatialHashDEM.hpp"

#include <cmath>

namespace powder::sim {

namespace {

[[nodiscard]] std::pair<int, int> key_for(float x, float y, float h) {
  const int ix = static_cast<int>(std::floor(x / h));
  const int iy = static_cast<int>(std::floor(y / h));
  return {ix, iy};
}

}  // namespace

void SpatialHash2D::build(const DEMParticleSoA& particles) {
  buckets_.clear();
  for (std::size_t i = 0; i < particles.x.size(); ++i) {
    if (!particles.is_alive(i)) {
      continue;
    }
    buckets_[key_for(particles.x[i], particles.y[i], cell_size_)].push_back(i);
  }
}

std::vector<DEMContact> SpatialHash2D::generate_contacts(const DEMParticleSoA& particles) const {
  static constexpr std::pair<int, int> offsets[5] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {-1, 1}};

  std::vector<DEMContact> contacts;
  for (const auto& [key, ids] : buckets_) {
    for (const auto& off : offsets) {
      const Key nb{key.first + off.first, key.second + off.second};
      const auto it = buckets_.find(nb);
      if (it == buckets_.end()) {
        continue;
      }

      const auto& jids = it->second;
      for (std::size_t ai = 0; ai < ids.size(); ++ai) {
        const auto a = ids[ai];
        const std::size_t bj0 = (&ids == &jids) ? ai + 1U : 0U;
        for (std::size_t bj = bj0; bj < jids.size(); ++bj) {
          const auto b = jids[bj];
          if (!particles.is_alive(a) || !particles.is_alive(b)) {
            continue;
          }
          const float dx = particles.x[b] - particles.x[a];
          const float dy = particles.y[b] - particles.y[a];
          const float rr = particles.radius[a] + particles.radius[b];
          const float d2 = dx * dx + dy * dy;
          if (d2 >= rr * rr || d2 <= 1.0e-20F) {
            continue;
          }
          const float d = std::sqrt(d2);
          DEMContact c{};
          c.a = a;
          c.b = b;
          c.nx = dx / d;
          c.ny = dy / d;
          c.penetration = rr - d;
          contacts.push_back(c);
        }
      }
    }
  }
  return contacts;
}

void solve_contacts_penalty_dashpot(DEMParticleSoA& particles, const std::vector<DEMContact>& contacts,
                                    float dt, const ContactModelConfig& cfg) {
  for (const auto& c : contacts) {
    const auto a = c.a;
    const auto b = c.b;
    if (!particles.is_alive(a) || !particles.is_alive(b)) {
      continue;
    }

    const float rvx = particles.vx[b] - particles.vx[a];
    const float rvy = particles.vy[b] - particles.vy[a];
    const float vn = rvx * c.nx + rvy * c.ny;

    float fn = cfg.k_normal * c.penetration - cfg.c_damping * vn;
    if (fn < 0.0F) {
      fn = 0.0F;
    }

    const float tx = -c.ny;
    const float ty = c.nx;
    const float vt = rvx * tx + rvy * ty;
    const float max_ft = cfg.friction_mu * fn;
    float ft = -vt * cfg.c_damping;
    if (ft > max_ft) {
      ft = max_ft;
    }
    if (ft < -max_ft) {
      ft = -max_ft;
    }

    const float fx = fn * c.nx + ft * tx;
    const float fy = fn * c.ny + ft * ty;

    const float inv_ma = particles.mass[a] > 0.0F ? 1.0F / particles.mass[a] : 0.0F;
    const float inv_mb = particles.mass[b] > 0.0F ? 1.0F / particles.mass[b] : 0.0F;

    particles.vx[a] -= fx * inv_ma * dt;
    particles.vy[a] -= fy * inv_ma * dt;
    particles.vx[b] += fx * inv_mb * dt;
    particles.vy[b] += fy * inv_mb * dt;

    const float inv_ia = particles.inertia[a] > 0.0F ? 1.0F / particles.inertia[a] : 0.0F;
    const float inv_ib = particles.inertia[b] > 0.0F ? 1.0F / particles.inertia[b] : 0.0F;
    particles.omega[a] -= ft * particles.radius[a] * inv_ia * dt;
    particles.omega[b] += ft * particles.radius[b] * inv_ib * dt;
  }
}

}  // namespace powder::sim
