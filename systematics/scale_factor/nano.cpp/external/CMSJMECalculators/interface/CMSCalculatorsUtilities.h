#ifndef CMSCALCULATORS_UTILITIES_H
#define CMSCALCULATORS_UTILITIES_H
#include <map>
#include <ROOT/RVec.hxx>
#include <variant>
#include <correction.h>

//#define BAMBOO_JME_DEBUG // uncomment to debug

#ifdef BAMBOO_JME_DEBUG
#define LogDebug std::cout
#else
#define LogDebug if (false) std::cout
#endif

namespace rdfhelpers {

class ModifiedPtMCollection { // map of variation collections
public:
  using compv_t = ROOT::VecOps::RVec<float>;

  ModifiedPtMCollection() = default;
  ModifiedPtMCollection(std::size_t n, const compv_t& pt, const compv_t& mass)
    : m_pt(n, pt), m_mass(n, mass) {}

  std::size_t size() const { return m_pt.size(); }

  const compv_t& pt(std::size_t i) const { return m_pt[i]; }
  const compv_t& mass(std::size_t i) const { return m_mass[i]; }

  void set(std::size_t i, const compv_t& pt, const compv_t& mass) {
    m_pt[i] = pt;
    m_mass[i] = mass;
  }
  void set(std::size_t i, compv_t&& pt, compv_t&& mass) {
    m_pt[i] = std::move(pt);
    m_mass[i] = std::move(mass);
  }
private:
  std::vector<compv_t> m_pt;
  std::vector<compv_t> m_mass;
};


class ModifiedPtMMsdCollection { // for fat jets
public:
  using compv_t = ROOT::VecOps::RVec<float>;

  ModifiedPtMMsdCollection() = default;
  ModifiedPtMMsdCollection(std::size_t n, const compv_t& pt, const compv_t& mass, const compv_t& msd)
    : m_pt(n, pt), m_mass(n, mass), m_msd(n, msd) {}

  std::size_t size() const { return m_pt.size(); }
  std::size_t sizeM() const { return m_mass.size(); }

  const compv_t& pt(std::size_t i) const { return m_pt[i]; }
  const compv_t& mass(std::size_t i) const { return m_mass[i]; }
  const compv_t& msoftdrop(std::size_t i) const { return m_msd[i]; }

  void set(std::size_t i, const compv_t& pt, const compv_t& mass, const compv_t& msd) {
    m_pt[i] = pt;
    m_mass[i] = mass;
    m_msd[i] = msd;
  }

  void set(std::size_t i, compv_t&& pt, compv_t&& mass, compv_t&& msd) {
    m_pt[i] = std::move(pt);
    m_mass[i] = std::move(mass);
    m_msd[i] = std::move(msd);
  }

private:
  std::vector<compv_t> m_pt;
  std::vector<compv_t> m_mass;
  std::vector<compv_t> m_msd;
};

class ModifiedPtCollection { // map of variation collections
public:
  using compv_t = ROOT::VecOps::RVec<float>;

  ModifiedPtCollection() = default;
  ModifiedPtCollection(std::size_t n, const compv_t& pt)
    : m_pt(n, pt) {}

  std::size_t size() const { return m_pt.size(); }

  const compv_t& pt(std::size_t i) const { return m_pt[i]; }

  void set(std::size_t i, const compv_t& pt) {
    m_pt[i] = pt;
  }
  void set(std::size_t i, compv_t&& pt) {
    m_pt[i] = std::move(pt);
  }

private:
  std::vector<compv_t> m_pt;
};

class ModifiedMET {
public:
  using compv_t = ROOT::VecOps::RVec<double>;

  ModifiedMET() = default;
  // initialize with the nominal value for all variations
  ModifiedMET(std::size_t n, double px_nom, double py_nom)
    : m_px(n, px_nom), m_py(n, py_nom) {}

  std::size_t size() const { return m_px.size(); }
  const compv_t& px() const { return m_px; }
  const compv_t& py() const { return m_py; }
  double px (std::size_t i) const { return m_px[i]; }
  double py (std::size_t i) const { return m_py[i]; }
  double pt (std::size_t i) const { return std::sqrt(m_px[i]*m_px[i]+m_py[i]*m_py[i]); }
  double phi(std::size_t i) const { return std::atan2(m_py[i], m_px[i]); }

  void setXY(std::size_t i, double dpx, double dpy) {
    m_px[i] = dpx;
    m_py[i] = dpy;
  }
  void addR_proj(std::size_t i, double cosphi, double sinphi, double dp) {
    m_px[i] += dp*cosphi;
    m_py[i] += dp*sinphi;
  }

private:
  compv_t m_px;
  compv_t m_py;
};
}

#endif
