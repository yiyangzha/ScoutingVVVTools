// combine.C
//
// Build CMS combine datacards from qcd_est.py output and run Significance +
// AsymptoticLimits on the Asimov data (sum of the process rates). Each input
// ROOT file is one channel with its matching BDT output directory; channels
// are concatenated with combineCards.py. Every channel card is a one-bin-per-SR
// shape card whose histogram errors carry the MC statistics (sum w^2, test
// split) for autoMCStats. The lnN nuisances are the per-sample-ratio ones
// (theory, pileup, JES, JER, JMS, JMR), lumi and trigger on the MC processes,
// and the per-channel ABCD non-closure and B/C/D statistics on the ABCD QCD
// prediction; see README.md. Stored ROOT covariance blocks can optionally be
// injected as eigen-decomposed Gaussian shape nuisances instead of autoMCStats.
//
// Invocation follows the other C++ tools: the binary reads its config from
// $COMBINE_CONFIG_PATH (or ./config.json). Any command-line arguments are
// ignored.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TKey.h>
#include <TMatrixDSym.h>
#include <TMatrixDSymEigen.h>
#include <TTree.h>
#include <TVectorD.h>

#include "../src/simple_json.h"

namespace fs = std::filesystem;
using JsonValue = simple_json::Value;

namespace {

const char* kAppConfigPath = "./config.json";
const char* kAppConfigEnvVar = "COMBINE_CONFIG_PATH";
std::string timestamp() {
    time_t now = time(nullptr);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    return std::string(buf);
}

void logMessage(const std::string& msg) {
    std::cout << "[" << timestamp() << "] " << msg << std::endl;
}

std::string resolveConfigPath(const char* preferredPath, const char* envVar) {
    if (envVar != nullptr) {
        const char* envPath = getenv(envVar);
        if (envPath != nullptr && *envPath != '\0') {
            if (fs::exists(envPath)) {
                return envPath;
            }
            throw std::runtime_error(
                std::string("Cannot find config file from environment variable ") +
                envVar + ": " + envPath);
        }
    }
    if (fs::exists(preferredPath)) {
        return preferredPath;
    }
    throw std::runtime_error(std::string("Cannot find config file: ") + preferredPath);
}

std::string resolveReferencedPath(const std::string& base, const std::string& target) {
    if (target.empty()) return target;
    fs::path p(target);
    if (p.is_absolute()) return fs::weakly_canonical(p).string();
    fs::path baseDir = fs::path(base).parent_path();
    return fs::weakly_canonical(baseDir / p).string();
}

std::string resolveReferencedPathFromDir(const std::string& base_dir,
                                         const std::string& target) {
    if (target.empty()) return target;
    fs::path p(target);
    if (p.is_absolute()) return fs::weakly_canonical(p).string();
    return fs::weakly_canonical(fs::path(base_dir) / p).string();
}

std::string slugify(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        } else {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.front() == '_') out.erase(out.begin());
    while (!out.empty() && out.back() == '_') out.pop_back();
    return out;
}

std::string shellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

// -------------------- Systematic ratio inputs --------------------
// Up/down yield ratios of one MC sample in one channel (tree), inclusive and
// per signal region (keyed by bin_index), from the *_syst_yields.json outputs
// of the systematic scripts.
struct RatioPair {
    double up = 1.0;
    double down = 1.0;
};
struct RatioSet {
    RatioPair inclusive;
    std::map<int, RatioPair> regions;
};
// sample -> channel -> ratios of one nuisance.
using SampleRatios = std::map<std::string, std::map<std::string, RatioSet>>;

// lnN nuisances built from per-sample yield ratios: the process kappa in each
// SR is the yield-weighted mean of its samples' ratios.
struct RatioNuisanceSpec {
    const char* nuisance;    // datacard row name (and enabled_nuisances name)
    const char* config_key;  // config.json key of the *_syst_yields.json path
    const char* producer;    // what writes that file
    const char* up_key;      // JSON field of the up ratio
    const char* down_key;    // JSON field of the down ratio
    bool theory;             // only samples with has_theory_weights carry it
};
const RatioNuisanceSpec kRatioNuisances[] = {
    {"theory_pdf",    "theory_syst_json", "mode 8 (theory_syst.py)",  "pdf_up",    "pdf_down",    true},
    {"theory_scale",  "theory_syst_json", "mode 8 (theory_syst.py)",  "scale_up",  "scale_down",  true},
    {"theory_ps_isr", "theory_syst_json", "mode 8 (theory_syst.py)",  "ps_isr_up", "ps_isr_down", true},
    {"theory_ps_fsr", "theory_syst_json", "mode 8 (theory_syst.py)",  "ps_fsr_up", "ps_fsr_down", true},
    {"pileup",        "pileup_syst_json", "mode 9 (pileup_syst.py)",  "pu_up",     "pu_down",     false},
    {"jes",           "jes_syst_json",    "mode 11 (jes_syst.py)",    "jes_up",    "jes_down",    false},
    {"jer",           "jer_syst_json",    "mode 12 (jer_syst.py)",    "jer_up",    "jer_down",    false},
    {"jms",           "jms_syst_json",    "mode 13 (jms_syst.py)",    "jms_up",    "jms_down",    false},
    {"jmr",           "jmr_syst_json",    "mode 14 (jmr_syst.py)",    "jmr_up",    "jmr_down",    false},
};
// The other nuisances: lumi and trigger (flat lnN on the MC processes), mcstat
// (autoMCStats), and the per-channel ABCD rows abcd_nonclosure_<channel> and
// abcd_mcstat_<channel>.
const char* const kOtherNuisances[] = {"lumi", "trigger", "mcstat", "abcd_nonclosure", "abcd_mcstat"};
// Process name of the ABCD QCD prediction (distinct from an MC-true QCD class
// process bkg_<class>).
const char* kAbcdQcdProcess = "bkg_qcd_abcd";

// -------------------- Config --------------------
struct ChannelSpec {
    std::string name;
    std::string root_file;
    std::string bdt_root;
};

struct AppConfig {
    std::vector<ChannelSpec> channels;
    std::string output_dir;
    std::string combine_cmd = "combine";
    std::string combine_cards_cmd = "combineCards.py";
    double eigen_rel_cutoff = 1e-10;
    bool use_root_covariance = false;
    bool rescale_shape_modes_to_positive = true;
    bool keep_work = true;
    std::string work_dir;  // resolved under output_dir
    // Fractional luminosity and trigger uncertainties, symmetric lnN on every
    // MC process (not on the ABCD QCD prediction).
    double lumi_unc = 0.0;
    double trigger_unc = 0.0;
    // autoMCStats threshold on the effective number of MC events per bin.
    int mc_stat_threshold = 10;
    // Per-sample ratios of each enabled kRatioNuisances nuisance.
    std::map<std::string, SampleRatios> ratios;
    // Nuisance names to include in datacards.  Empty set = all enabled (default).
    std::set<std::string> enabled_nuisances;
};

// Returns true when a nuisance should be emitted: always when the enabled set
// is empty (= all enabled), otherwise only when listed.
inline bool nuisanceEnabled(const std::set<std::string>& enabled,
                            const std::string& name) {
    return enabled.empty() || enabled.count(name) > 0;
}

RatioPair readRatioPair(const JsonValue& node, const RatioNuisanceSpec& spec) {
    RatioPair out;
    out.up = static_cast<double>(node.at(spec.up_key).asNumber());
    out.down = static_cast<double>(node.at(spec.down_key).asNumber());
    return out;
}

// Reads the per-sample, per-channel, per-SR ratios of one nuisance.
SampleRatios loadSampleRatios(const JsonValue& payload, const RatioNuisanceSpec& spec,
                              const std::string& path) {
    SampleRatios out;
    for (const auto& sample_kv : payload.asObject()) {
        for (const auto& tree_kv : sample_kv.second.asObject()) {
            const JsonValue& v = tree_kv.second;
            if (!v.contains("regions")) {
                throw std::runtime_error(
                    std::string(path) + ": sample '" + sample_kv.first + "' tree '" + tree_kv.first +
                    "' has no per-signal-region ratios; configure bdt_root and signal_region_csv "
                    "for " + spec.producer);
            }
            RatioSet rs;
            rs.inclusive = readRatioPair(v, spec);
            for (const auto& reg_kv : v.at("regions").asObject()) {
                rs.regions[std::stoi(reg_kv.first)] = readRatioPair(reg_kv.second, spec);
            }
            out[sample_kv.first][tree_kv.first] = std::move(rs);
        }
    }
    return out;
}

AppConfig loadAppConfig() {
    const std::string path = resolveConfigPath(kAppConfigPath, kAppConfigEnvVar);
    const std::string abs = fs::weakly_canonical(path).string();
    const JsonValue payload = simple_json::parseFile(abs);

    AppConfig cfg;
    if (!payload.contains("channels")) {
        throw std::runtime_error("config.json missing 'channels' array");
    }
    for (const auto& item : payload.at("channels").asArray()) {
        ChannelSpec ch;
        ch.name = item.at("name").asString();
        ch.root_file = resolveReferencedPath(abs, item.at("root_file").asString());
        if (!item.contains("bdt_root")) {
            throw std::runtime_error(
                "config.json channel '" + ch.name + "' missing required 'bdt_root'");
        }
        ch.bdt_root = resolveReferencedPath(abs, item.at("bdt_root").asString());
        cfg.channels.push_back(std::move(ch));
    }
    if (cfg.channels.empty()) {
        throw std::runtime_error("config.json 'channels' is empty");
    }

    cfg.output_dir = resolveReferencedPath(
        abs, payload.getStringOr("output_dir", "./output"));
    cfg.combine_cmd = payload.getStringOr("combine_cmd", "combine");
    cfg.combine_cards_cmd = payload.getStringOr("combine_cards_cmd", "combineCards.py");
    cfg.eigen_rel_cutoff = static_cast<double>(
        payload.getNumberOr("eigen_rel_cutoff", 1e-10L));
    cfg.use_root_covariance = payload.getBoolOr("use_root_covariance", false);
    cfg.rescale_shape_modes_to_positive =
        payload.getBoolOr("rescale_shape_modes_to_positive", true);
    cfg.keep_work = payload.getBoolOr("keep_work", true);
    cfg.work_dir = (fs::path(cfg.output_dir) / "work").string();
    cfg.lumi_unc = static_cast<double>(payload.getNumberOr("lumi_unc", 0.0L));
    cfg.trigger_unc = static_cast<double>(payload.getNumberOr("trigger_unc", 0.0L));
    cfg.mc_stat_threshold = payload.getIntOr("mc_stat_threshold", cfg.mc_stat_threshold);

    // Enabled-nuisances filter: empty list = all enabled.
    if (payload.contains("enabled_nuisances")) {
        std::set<std::string> known(std::begin(kOtherNuisances), std::end(kOtherNuisances));
        for (const auto& spec : kRatioNuisances) known.insert(spec.nuisance);
        for (const auto& item : payload.at("enabled_nuisances").asArray()) {
            const std::string name = item.asString();
            if (!known.count(name)) {
                throw std::runtime_error("enabled_nuisances lists unknown nuisance '" + name + "'");
            }
            cfg.enabled_nuisances.insert(name);
        }
    }

    // Every enabled nuisance needs its input.
    std::map<std::string, JsonValue> parsed;  // path -> payload
    for (const auto& spec : kRatioNuisances) {
        if (!nuisanceEnabled(cfg.enabled_nuisances, spec.nuisance)) continue;
        if (!payload.contains(spec.config_key)) {
            throw std::runtime_error(std::string("nuisance '") + spec.nuisance + "' is enabled but " +
                                     spec.config_key + " is not set in config.json");
        }
        const std::string ratio_path =
            resolveReferencedPath(abs, payload.at(spec.config_key).asString());
        if (!fs::exists(ratio_path)) {
            throw std::runtime_error(std::string("nuisance '") + spec.nuisance + "' is enabled but " +
                                     ratio_path + " does not exist; run " + spec.producer);
        }
        auto it = parsed.find(ratio_path);
        if (it == parsed.end()) {
            it = parsed.emplace(ratio_path, simple_json::parseFile(ratio_path)).first;
            logMessage("Loaded syst yields: " + ratio_path);
        }
        cfg.ratios[spec.nuisance] = loadSampleRatios(it->second, spec, ratio_path);
    }
    if (nuisanceEnabled(cfg.enabled_nuisances, "lumi") && !(cfg.lumi_unc > 0.0)) {
        throw std::runtime_error("nuisance 'lumi' is enabled but lumi_unc is not positive");
    }
    if (nuisanceEnabled(cfg.enabled_nuisances, "trigger") && !(cfg.trigger_unc > 0.0)) {
        throw std::runtime_error("nuisance 'trigger' is enabled but trigger_unc is not positive");
    }
    if (nuisanceEnabled(cfg.enabled_nuisances, "mcstat")) {
        if (cfg.mc_stat_threshold < 0) {
            throw std::runtime_error("mc_stat_threshold must be >= 0");
        }
        if (cfg.use_root_covariance) {
            throw std::runtime_error(
                "use_root_covariance and the mcstat nuisance (autoMCStats) both model the "
                "statistical uncertainty of the predictions; disable one of them");
        }
    }
    return cfg;
}

// -------------------- Sample / class registry --------------------
struct SampleInfo {
    std::string name;
    bool is_MC = false;
    bool is_signal = false;
    bool has_theory_weights = false;
};

struct ClassRegistry {
    std::vector<std::string> class_order;                           // BDT class order
    std::map<std::string, std::vector<std::string>> class_members;  // class -> samples
    std::unordered_map<std::string, std::string> sample_to_class;
    std::set<std::string> signal_classes;
    std::vector<std::string> signal_samples;
    std::vector<std::string> qcd_classes;
    std::set<std::string> qcd_class_set;
    std::unordered_map<std::string, SampleInfo> samples;
};

ClassRegistry loadRegistryFromBdtRoot(const std::string& bdt_root,
                                      const std::string& label) {
    ClassRegistry reg;

    const std::string bdt_config_path =
        fs::weakly_canonical(fs::path(bdt_root) / "config.json").string();
    JsonValue bdtJson = simple_json::parseFile(bdt_config_path);

    const std::string sample_cfg_path = resolveReferencedPathFromDir(
        fs::path(bdt_root).parent_path().string(),
        bdtJson.at("sample_config").asString());
    JsonValue sampleJson = simple_json::parseFile(sample_cfg_path);
    for (const auto& node : sampleJson.at("sample").asArray()) {
        SampleInfo info;
        info.name = node.at("name").asString();
        info.is_MC = node.at("is_MC").asBool();
        info.is_signal = node.at("is_signal").asBool();
        info.has_theory_weights = node.getBoolOr("has_theory_weights", false);
        reg.samples[info.name] = info;
    }

    if (!bdtJson.contains("class_groups")) {
        throw std::runtime_error(label + " bdt_root/config.json missing 'class_groups'");
    }
    for (const auto& kv : bdtJson.at("class_groups").asObject()) {
        reg.class_order.push_back(kv.first);
        std::vector<std::string> members;
        bool all_signal = true;
        bool any = false;
        for (const auto& item : kv.second.asArray()) {
            const std::string s = item.asString();
            members.push_back(s);
            reg.sample_to_class[s] = kv.first;
            auto it = reg.samples.find(s);
            if (it == reg.samples.end()) {
                throw std::runtime_error(
                    label + " class_groups references unknown sample: " + s);
            }
            if (!it->second.is_signal) all_signal = false;
            any = true;
        }
        reg.class_members[kv.first] = std::move(members);
        if (any && all_signal) reg.signal_classes.insert(kv.first);
        if (slugify(kv.first).find("qcd") != std::string::npos) {
            reg.qcd_classes.push_back(kv.first);
            reg.qcd_class_set.insert(kv.first);
        }
    }
    if (reg.qcd_classes.empty()) {
        throw std::runtime_error(label + " class_groups must contain at least one QCD class");
    }

    for (const auto& c : reg.class_order) {
        for (const auto& s : reg.class_members.at(c)) {
            if (reg.samples.at(s).is_signal) reg.signal_samples.push_back(s);
        }
    }
    return reg;
}

void ensureRegistryCompatible(const ClassRegistry& reference,
                              const ClassRegistry& candidate,
                              const std::string& reference_label,
                              const std::string& candidate_label) {
    const std::string detail =
        candidate_label + " BDT registry differs from " + reference_label +
        "; combine.C requires all channel bdt_root configs to share the same "
        "class_groups and signal/QCD sample definitions";

    if (candidate.class_order != reference.class_order) {
        throw std::runtime_error(detail + " (class order mismatch)");
    }
    if (candidate.class_members != reference.class_members) {
        throw std::runtime_error(detail + " (class_groups membership mismatch)");
    }
    if (candidate.signal_classes != reference.signal_classes) {
        throw std::runtime_error(detail + " (signal class mismatch)");
    }
    if (candidate.signal_samples != reference.signal_samples) {
        throw std::runtime_error(detail + " (signal sample mismatch)");
    }
    if (candidate.qcd_classes != reference.qcd_classes) {
        throw std::runtime_error(detail + " (QCD class mismatch)");
    }
}

ClassRegistry loadRegistry(const AppConfig& cfg) {
    const ChannelSpec& first = cfg.channels.front();
    const std::string reference_label = "channel '" + first.name + "'";
    ClassRegistry reg = loadRegistryFromBdtRoot(first.bdt_root, reference_label);

    for (size_t i = 1; i < cfg.channels.size(); ++i) {
        const ChannelSpec& ch = cfg.channels[i];
        const std::string candidate_label = "channel '" + ch.name + "'";
        ClassRegistry candidate = loadRegistryFromBdtRoot(ch.bdt_root, candidate_label);
        ensureRegistryCompatible(reg, candidate, reference_label, candidate_label);
    }
    return reg;
}

// -------------------- ROOT reading --------------------
struct YieldCov {
    std::vector<double> yields;
    std::vector<double> mc_stat_vars;  // sum w^2 of the MC per SR (sr<id>/mc_stat_error^2)
    TMatrixDSym cov;                // n x n
    YieldCov() = default;
    explicit YieldCov(int n) : yields(n, 0.0), mc_stat_vars(n, 0.0), cov(n) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) cov(i, j) = 0.0;
    }
    YieldCov(const YieldCov&) = default;
    YieldCov(YieldCov&&) noexcept = default;
    YieldCov& operator=(const YieldCov& other) {
        if (this == &other) return *this;
        yields = other.yields;
        mc_stat_vars = other.mc_stat_vars;
        cov.ResizeTo(other.cov);
        cov = other.cov;
        return *this;
    }
    YieldCov& operator=(YieldCov&& other) noexcept {
        if (this == &other) return *this;
        yields = std::move(other.yields);
        mc_stat_vars = std::move(other.mc_stat_vars);
        cov.ResizeTo(other.cov);
        cov = other.cov;
        return *this;
    }
    int n() const { return static_cast<int>(yields.size()); }
};

// ABCD scale factors of qcd_est.py (metadata/abcd_closure).
struct AbcdClosure {
    double final_scale = 1.0;       // k = pred_union / A_union
    double pred_union = 0.0;
    double pred_union_error = 0.0;  // MC statistics of the B/C/D regions
};

struct ChannelData {
    std::string name;
    int n_sr = 0;
    std::vector<int> sr_ids;
    std::map<std::string, YieldCov> sample;       // per MC sample (MC true)
    std::map<std::string, YieldCov> group;        // per BDT class (MC true)
    YieldCov qcd_predict;                         // merged ABCD QCD prediction
    AbcdClosure abcd;
};

double readOneBinHist(TFile& f, const std::string& path) {
    TH1* h = dynamic_cast<TH1*>(f.Get(path.c_str()));
    if (h == nullptr) {
        throw std::runtime_error("Missing one-bin histogram '" + path + "' in " + f.GetName());
    }
    if (h->GetNbinsX() != 1) {
        throw std::runtime_error("Histogram '" + path + "' in " + f.GetName() +
                                 " must have exactly one bin");
    }
    return h->GetBinContent(1);
}

std::vector<int> defaultSignalRegionIds(int n) {
    std::vector<int> out;
    out.reserve(std::max(n, 0));
    for (int i = 0; i < n; ++i) out.push_back(i + 1);
    return out;
}

std::vector<int> readSignalRegionIds(TFile& f) {
    TTree* t = dynamic_cast<TTree*>(f.Get("metadata/signal_regions"));
    if (t == nullptr) return {};

    int bin_index = 0;
    if (t->GetBranch("bin_index") == nullptr) {
        throw std::runtime_error(
            "metadata/signal_regions in " + std::string(f.GetName()) +
            " is missing required branch 'bin_index'");
    }
    t->SetBranchAddress("bin_index", &bin_index);

    std::vector<int> ids;
    std::set<int> seen;
    const Long64_t n = t->GetEntries();
    if (n <= 0) {
        throw std::runtime_error(
            "metadata/signal_regions in " + std::string(f.GetName()) +
            " must contain at least one row");
    }
    ids.reserve(static_cast<size_t>(n));
    for (Long64_t i = 0; i < n; ++i) {
        t->GetEntry(i);
        if (bin_index <= 0) {
            throw std::runtime_error(
                "metadata/signal_regions in " + std::string(f.GetName()) +
                " contains a non-positive bin_index");
        }
        if (!seen.insert(bin_index).second) {
            throw std::runtime_error(
                "metadata/signal_regions in " + std::string(f.GetName()) +
                " contains duplicate bin_index=" + std::to_string(bin_index));
        }
        ids.push_back(bin_index);
    }
    return ids;
}

std::string formatSignalRegionIds(const std::vector<int>& ids) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) os << ",";
        os << ids[i];
    }
    os << "]";
    return os.str();
}

YieldCov readYieldCov(TFile& f, const std::string& prefix,
                      const std::vector<int>& signal_region_ids,
                      bool read_mc_stat) {
    // TFile owns the returned histograms; just copy the contents out.
    TH2* h2 = dynamic_cast<TH2*>(f.Get((prefix + "/covariance_total").c_str()));
    if (h2 == nullptr) {
        throw std::runtime_error("Missing '" + prefix + "/covariance_total' in " + f.GetName());
    }
    const int n = h2->GetNbinsX();
    if (n <= 0 || h2->GetNbinsY() != n) {
        throw std::runtime_error("Covariance size mismatch for " + prefix);
    }
    const std::vector<int> sr_ids =
        signal_region_ids.empty() ? defaultSignalRegionIds(n) : signal_region_ids;
    if (static_cast<int>(sr_ids.size()) != n) {
        throw std::runtime_error(
            "Signal-region metadata size mismatch for " + prefix +
            " in " + f.GetName() + ": metadata has " +
            std::to_string(sr_ids.size()) + " entries but covariance has " +
            std::to_string(n) + " bins");
    }
    YieldCov out(n);
    for (int i = 0; i < n; ++i) {
        const std::string sr_prefix = prefix + "/sr" + std::to_string(sr_ids[i]);
        out.yields[i] = readOneBinHist(f, sr_prefix + "/yield");
        readOneBinHist(f, sr_prefix + "/stat_error");
        readOneBinHist(f, sr_prefix + "/scale_error");
        if (read_mc_stat) {
            const double err = readOneBinHist(f, sr_prefix + "/mc_stat_error");
            out.mc_stat_vars[i] = err * err;
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out.cov(i, j) = h2->GetBinContent(i + 1, j + 1);
        }
    }
    return out;
}

// List the subdirectory names directly under a TDirectory prefix.
std::vector<std::string> listSubdirs(TFile& f, const std::string& prefix) {
    std::vector<std::string> out;
    TDirectory* dir = f.GetDirectory(prefix.c_str());
    if (dir == nullptr) return out;
    TIter next(dir->GetListOfKeys());
    std::set<std::string> seen;
    while (TKey* key = static_cast<TKey*>(next())) {
        const std::string cls = key->GetClassName();
        if (cls != "TDirectory" && cls != "TDirectoryFile") continue;
        const std::string name = key->GetName();
        if (seen.insert(name).second) out.push_back(name);
    }
    std::sort(out.begin(), out.end());
    return out;
}

AbcdClosure readAbcdClosure(TFile& f) {
    TTree* t = dynamic_cast<TTree*>(f.Get("metadata/abcd_closure"));
    if (t == nullptr || t->GetEntries() != 1) {
        throw std::runtime_error(
            "metadata/abcd_closure (one row) not found in " + std::string(f.GetName()) +
            "; re-run mode 5 (qcd_est.py) for the ABCD nuisances");
    }
    AbcdClosure out;
    t->SetBranchAddress("final_scale", &out.final_scale);
    t->SetBranchAddress("pred_union", &out.pred_union);
    t->SetBranchAddress("pred_union_error", &out.pred_union_error);
    t->GetEntry(0);
    t->ResetBranchAddresses();
    if (!(out.pred_union > 0.0) || !std::isfinite(out.final_scale) || !(out.pred_union_error >= 0.0)) {
        throw std::runtime_error("Invalid metadata/abcd_closure in " + std::string(f.GetName()));
    }
    return out;
}

// read_mc_stat / read_abcd: the inputs of the mcstat and ABCD nuisances.
ChannelData loadChannel(const ChannelSpec& spec, bool read_mc_stat, bool read_abcd) {
    logMessage("Reading channel '" + spec.name + "' from " + spec.root_file);
    if (!fs::exists(spec.root_file)) {
        throw std::runtime_error("Channel ROOT file not found: " + spec.root_file);
    }
    TFile* f = TFile::Open(spec.root_file.c_str(), "READ");
    if (f == nullptr || f->IsZombie()) {
        if (f != nullptr) delete f;
        throw std::runtime_error("Cannot open ROOT file: " + spec.root_file);
    }
    ChannelData data;
    data.name = spec.name;
    const std::vector<int> metadata_sr_ids = readSignalRegionIds(*f);

    // Samples
    const std::vector<std::string> sample_names = listSubdirs(*f, "samples");
    if (sample_names.empty()) {
        throw std::runtime_error("No samples/ entries in " + spec.root_file);
    }
    for (const auto& s : sample_names) {
        auto inserted = data.sample.emplace(
            s, readYieldCov(*f, "samples/" + s, metadata_sr_ids, read_mc_stat));
        if (!inserted.second) {
            throw std::runtime_error("Duplicate sample '" + s + "' in " + spec.root_file);
        }
    }
    data.n_sr = data.sample.begin()->second.n();
    data.sr_ids =
        metadata_sr_ids.empty() ? defaultSignalRegionIds(data.n_sr) : metadata_sr_ids;
    for (const auto& kv : data.sample) {
        if (kv.second.n() != data.n_sr) {
            throw std::runtime_error("Inconsistent SR count for sample " + kv.first);
        }
    }

    const std::vector<std::string> group_names = listSubdirs(*f, "groups");
    if (group_names.empty()) {
        throw std::runtime_error("No groups/ entries in " + spec.root_file);
    }
    for (const auto& g : group_names) {
        const std::string key = slugify(g);
        auto inserted = data.group.emplace(
            key, readYieldCov(*f, "groups/" + g, data.sr_ids, read_mc_stat));
        if (!inserted.second) {
            throw std::runtime_error(
                "Duplicate groups/ entries after case-insensitive matching: '" + g +
                "' collides with another group in " + spec.root_file);
        }
    }
    for (const auto& kv : data.group) {
        if (kv.second.n() != data.n_sr) {
            throw std::runtime_error("Inconsistent SR count for group " + kv.first);
        }
    }

    data.qcd_predict = readYieldCov(*f, "qcd_predict", data.sr_ids, read_mc_stat);
    if (data.qcd_predict.n() != data.n_sr) {
        throw std::runtime_error("qcd_predict SR count mismatch for " + spec.name);
    }
    if (read_abcd) {
        data.abcd = readAbcdClosure(*f);
    }

    logMessage("  channel '" + data.name + "': n_sr=" + std::to_string(data.n_sr) +
               ", sr_ids=" + formatSignalRegionIds(data.sr_ids) +
               ", samples=" + std::to_string(data.sample.size()) +
               ", groups=" + std::to_string(data.group.size()));
    f->Close();
    delete f;
    return data;
}

// -------------------- Scenario --------------------
struct Scenario {
    std::string scope;          // combined / class / sample
    std::string name;           // row identifier
    std::set<std::string> signal_samples;  // samples treated as signal
};

std::vector<Scenario> buildScenarios(const ClassRegistry& reg) {
    std::vector<Scenario> out;

    Scenario comb;
    comb.scope = "combined";
    comb.name = "combined";
    for (const auto& s : reg.signal_samples) comb.signal_samples.insert(s);
    out.push_back(std::move(comb));

    // signal classes in registry order
    for (const auto& cls : reg.class_order) {
        if (!reg.signal_classes.count(cls)) continue;
        Scenario sc;
        sc.scope = "class";
        sc.name = cls;
        for (const auto& s : reg.class_members.at(cls)) sc.signal_samples.insert(s);
        out.push_back(std::move(sc));
    }
    // signal samples
    for (const auto& s : reg.signal_samples) {
        Scenario sc;
        sc.scope = "sample";
        sc.name = s;
        sc.signal_samples = {s};
        out.push_back(std::move(sc));
    }
    return out;
}

// -------------------- Process construction --------------------
struct Process {
    std::string name;            // e.g. "signal", "bkg_vh", "bkg_qcd_abcd"
    std::vector<double> yields;
    std::vector<double> mc_stat_vars;  // sum w^2 per SR (autoMCStats bin errors)
    TMatrixDSym cov;
    // MC samples whose yields make up this process; empty for the ABCD QCD
    // prediction, which takes the ABCD nuisances instead of the MC-based ones.
    std::vector<std::string> samples;
    bool abcd = false;
    Process() : cov(1) {}
};

bool isQcdClass(const ClassRegistry& reg, const std::string& class_name) {
    return reg.qcd_class_set.count(class_name) != 0u;
}

void addYieldCov(Process& proc, const YieldCov& src) {
    if (proc.yields.empty()) {
        proc.yields = std::vector<double>(src.n(), 0.0);
        proc.mc_stat_vars = std::vector<double>(src.n(), 0.0);
        proc.cov.ResizeTo(src.n(), src.n());
        for (int i = 0; i < src.n(); ++i)
            for (int j = 0; j < src.n(); ++j) proc.cov(i, j) = 0.0;
    }
    const int n = static_cast<int>(proc.yields.size());
    if (src.n() != n) throw std::runtime_error("SR size mismatch in addYieldCov");
    for (int i = 0; i < n; ++i) proc.yields[i] += src.yields[i];
    for (int i = 0; i < n; ++i) proc.mc_stat_vars[i] += src.mc_stat_vars[i];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) proc.cov(i, j) += src.cov(i, j);
}

void validateYieldCov(const YieldCov& yc, const std::string& label) {
    if (yc.n() <= 0) {
        throw std::runtime_error("Empty yield/cov bundle for " + label);
    }
    for (int i = 0; i < yc.n(); ++i) {
        const double y = yc.yields[i];
        if (!std::isfinite(y)) {
            throw std::runtime_error("Non-finite yield in " + label);
        }
        if (y < 0.0) {
            throw std::runtime_error("Negative nominal yield in " + label);
        }
        for (int j = 0; j < yc.n(); ++j) {
            const double c = yc.cov(i, j);
            if (!std::isfinite(c)) {
                throw std::runtime_error("Non-finite covariance in " + label);
            }
        }
        if (yc.cov(i, i) < 0.0) {
            throw std::runtime_error("Negative covariance diagonal in " + label);
        }
    }
}

const YieldCov& getRequiredYieldCov(const std::map<std::string, YieldCov>& bundles,
                                    const std::string& name,
                                    const std::string& bundle_kind,
                                    const std::string& channel_name) {
    auto it = bundles.find(name);
    if (it == bundles.end()) {
        throw std::runtime_error(
            "Missing required " + bundle_kind + " '" + name +
            "' in channel '" + channel_name + "'");
    }
    validateYieldCov(it->second, bundle_kind + " '" + name + "' in channel '" + channel_name + "'");
    return it->second;
}

const YieldCov& getRequiredGroupYieldCov(const ChannelData& ch,
                                         const std::string& class_name) {
    const std::string lookup = slugify(class_name);
    auto it = ch.group.find(lookup);
    if (it == ch.group.end()) {
        throw std::runtime_error(
            "Missing required group '" + class_name + "' (matched as '" + lookup +
            "') in channel '" + ch.name + "'");
    }
    validateYieldCov(
        it->second,
        "group '" + class_name + "' (matched as '" + lookup + "') in channel '" + ch.name + "'");
    return it->second;
}

Process makeProcessFromYieldCov(const std::string& name, const YieldCov& yc,
                                const std::string& label,
                                std::vector<std::string> samples, bool abcd = false) {
    validateYieldCov(yc, label);
    Process out;
    out.name = name;
    out.yields = yc.yields;
    out.mc_stat_vars = yc.mc_stat_vars;
    out.samples = std::move(samples);
    out.abcd = abcd;
    out.cov.ResizeTo(yc.n(), yc.n());
    for (int i = 0; i < yc.n(); ++i) {
        for (int j = 0; j < yc.n(); ++j) {
            out.cov(i, j) = yc.cov(i, j);
        }
    }
    return out;
}

void appendProcess(std::vector<Process>& processes, std::set<std::string>& names,
                   Process proc, const std::string& label) {
    if (!names.insert(proc.name).second) {
        throw std::runtime_error("Duplicate process name '" + proc.name + "' for " + label);
    }
    processes.push_back(std::move(proc));
}

std::vector<Process> buildGroupProcesses(const ChannelData& ch, const ClassRegistry& reg,
                                         const Scenario& sc, bool use_abcd) {
    std::vector<Process> out;
    std::set<std::string> process_names;
    bool qcd_predict_added = false;

    Process sig;
    sig.name = "signal";

    if (sc.scope == "combined") {
        for (const auto& cls : reg.class_order) {
            if (!reg.signal_classes.count(cls)) continue;
            const YieldCov& yc = getRequiredGroupYieldCov(ch, cls);
            addYieldCov(sig, yc);
            const auto& members = reg.class_members.at(cls);
            sig.samples.insert(sig.samples.end(), members.begin(), members.end());
        }
    } else if (sc.scope == "class") {
        if (!reg.signal_classes.count(sc.name)) {
            throw std::runtime_error("Scenario class '" + sc.name + "' is not a signal class");
        }
        const YieldCov& yc = getRequiredGroupYieldCov(ch, sc.name);
        addYieldCov(sig, yc);
        sig.samples = reg.class_members.at(sc.name);
    } else {
        throw std::runtime_error("Group-based builder cannot handle scenario scope '" + sc.scope + "'");
    }

    if (sig.yields.empty()) {
        throw std::runtime_error(
            "Signal is empty in grouped scenario '" + sc.scope + "/" + sc.name +
            "' for channel '" + ch.name + "'");
    }
    appendProcess(out, process_names, std::move(sig),
                  "grouped scenario '" + sc.scope + "/" + sc.name +
                  "' in channel '" + ch.name + "'");

    for (const auto& cls : reg.class_order) {
        if (sc.scope == "combined" && reg.signal_classes.count(cls)) continue;
        if (sc.scope == "class" && cls == sc.name) continue;

        if (use_abcd && isQcdClass(reg, cls)) {
            if (!qcd_predict_added) {
                appendProcess(out, process_names,
                              makeProcessFromYieldCov(
                                  kAbcdQcdProcess,
                                  ch.qcd_predict,
                                  "qcd_predict in channel '" + ch.name + "'",
                                  {}, /*abcd=*/true),
                              "grouped scenario '" + sc.scope + "/" + sc.name +
                              "' in channel '" + ch.name + "'");
                qcd_predict_added = true;
            }
            continue;
        }

        const YieldCov& yc = getRequiredGroupYieldCov(ch, cls);
        appendProcess(out, process_names,
                      makeProcessFromYieldCov(
                          "bkg_" + slugify(cls),
                          yc,
                          "group '" + cls + "' in channel '" + ch.name + "'",
                          reg.class_members.at(cls)),
                      "grouped scenario '" + sc.scope + "/" + sc.name +
                      "' in channel '" + ch.name + "'");
    }
    return out;
}

std::vector<Process> buildSampleProcesses(const ChannelData& ch, const ClassRegistry& reg,
                                          const Scenario& sc, bool use_abcd) {
    if (sc.scope != "sample" || sc.signal_samples.size() != 1u) {
        throw std::runtime_error("Sample-based builder requires exactly one signal sample");
    }

    std::vector<Process> out;
    std::set<std::string> process_names;

    const std::string signal_sample = *sc.signal_samples.begin();
    const YieldCov& sig_yc = getRequiredYieldCov(ch.sample, signal_sample, "sample", ch.name);
    appendProcess(out, process_names,
                  makeProcessFromYieldCov(
                      "signal",
                      sig_yc,
                      "signal sample '" + signal_sample + "' in channel '" + ch.name + "'",
                      {signal_sample}),
                  "sample scenario '" + signal_sample + "' in channel '" + ch.name + "'");

    bool qcd_predict_added = false;
    for (const auto& cls : reg.class_order) {
        if (use_abcd && isQcdClass(reg, cls)) {
            if (!qcd_predict_added) {
                appendProcess(out, process_names,
                              makeProcessFromYieldCov(
                                  kAbcdQcdProcess,
                                  ch.qcd_predict,
                                  "qcd_predict in channel '" + ch.name + "'",
                                  {}, /*abcd=*/true),
                              "sample scenario '" + signal_sample + "' in channel '" + ch.name + "'");
                qcd_predict_added = true;
            }
            continue;
        }

        for (const auto& sample_name : reg.class_members.at(cls)) {
            if (sample_name == signal_sample) continue;
            auto it = ch.sample.find(sample_name);
            if (it == ch.sample.end()) continue;
            appendProcess(out, process_names,
                          makeProcessFromYieldCov(
                              "bkg_" + slugify(sample_name),
                              it->second,
                              "sample '" + sample_name + "' in channel '" + ch.name + "'",
                              {sample_name}),
                          "sample scenario '" + signal_sample + "' in channel '" + ch.name + "'");
        }
    }
    return out;
}

std::vector<Process> buildProcesses(const ChannelData& ch, const ClassRegistry& reg,
                                    const Scenario& sc, bool use_abcd) {
    if (sc.scope == "sample") {
        return buildSampleProcesses(ch, reg, sc, use_abcd);
    }
    return buildGroupProcesses(ch, reg, sc, use_abcd);
}

void validateChannelAgainstRegistry(const ChannelData& ch, const ClassRegistry& reg) {
    for (const auto& cls : reg.class_order) {
        getRequiredGroupYieldCov(ch, cls);
    }
    for (const auto& sample_name : reg.signal_samples) {
        getRequiredYieldCov(ch.sample, sample_name, "signal sample", ch.name);
    }
    validateYieldCov(ch.qcd_predict, "qcd_predict in channel '" + ch.name + "'");
}

// -------------------- Covariance eigen-decomposition --------------------
struct EigenMode {
    double scale;              // sqrt(lambda_k)
    std::vector<double> v;     // eigenvector entries
    double template_scale = 1.0;  // a in the datacard explanation
};

std::vector<EigenMode> decomposeCov(const TMatrixDSym& cov, double rel_cutoff) {
    const int n = cov.GetNrows();
    std::vector<EigenMode> modes;
    if (n == 0) return modes;

    // Find max diagonal to set absolute cutoff.
    double max_diag = 0.0;
    for (int i = 0; i < n; ++i) max_diag = std::max(max_diag, cov(i, i));
    if (max_diag <= 0.0) return modes;
    const double cutoff = rel_cutoff * max_diag;

    // ROOT's TMatrixDSymEigen produces eigenvalues/eigenvectors.
    TMatrixDSym sym(cov);
    TMatrixDSymEigen eig(sym);
    const TVectorD evals = eig.GetEigenValues();
    const TMatrixD evecs = eig.GetEigenVectors();

    for (int k = 0; k < n; ++k) {
        const double lam = evals(k);
        if (lam < -cutoff) {
            throw std::runtime_error(
                "Covariance matrix has a negative eigenvalue (" + std::to_string(lam) +
                "), cannot represent it as Gaussian shape nuisances");
        }
        if (lam <= cutoff) continue;
        EigenMode m;
        m.scale = std::sqrt(lam);
        m.v.resize(n);
        for (int i = 0; i < n; ++i) m.v[i] = evecs(i, k);
        modes.push_back(std::move(m));
    }
    return modes;
}

std::string formatDouble(double value) {
    std::ostringstream os;
    os << std::setprecision(17) << value;
    return os.str();
}

double modeIntegralDelta(const EigenMode& mode) {
    double sum = 0.0;
    for (double x : mode.v) sum += mode.scale * mode.template_scale * x;
    return sum;
}

double processIntegral(const std::vector<double>& values) {
    double sum = 0.0;
    for (double x : values) sum += x;
    return sum;
}

bool isExactlyZeroProcess(const Process& proc) {
    for (double y : proc.yields) {
        if (y != 0.0) return false;
    }
    const int n = proc.cov.GetNrows();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < proc.cov.GetNcols(); ++j) {
            if (proc.cov(i, j) != 0.0) return false;
        }
    }
    return true;
}

std::vector<Process> maybeDropZeroBackgroundProcesses(const AppConfig& cfg,
                                                      std::vector<Process> processes,
                                                      const Scenario& sc,
                                                      const std::string& channel_name) {
    if (!cfg.rescale_shape_modes_to_positive) return processes;

    std::vector<Process> kept;
    kept.reserve(processes.size());
    for (auto& proc : processes) {
        if (proc.name == "signal") {
            kept.push_back(std::move(proc));
            continue;
        }

        if (isExactlyZeroProcess(proc)) {
            logMessage("WARNING: Dropping zero-yield zero-covariance background process: "
                       "channel=" + channel_name + " scenario=" + sc.scope + "/" + sc.name +
                       " process=" + proc.name);
            continue;
        }
        kept.push_back(std::move(proc));
    }
    return kept;
}

const Process& getRequiredProcess(const std::vector<Process>& processes,
                                  const std::string& name,
                                  const Scenario& sc,
                                  const std::string& channel_name) {
    for (const auto& proc : processes) {
        if (proc.name == name) return proc;
    }
    throw std::runtime_error("Missing process '" + name + "' for scenario '" + sc.scope +
                             "/" + sc.name + "' in channel '" + channel_name + "'");
}

double computePositiveTemplateScaleLimit(const Process& proc, const EigenMode& mode,
                                         std::string& limiting_reason) {
    const double nominal_integral = processIntegral(proc.yields);
    double max_a = std::numeric_limits<double>::infinity();
    bool has_bound = false;
    limiting_reason = "none";

    auto tighten = [&](double bound, const std::string& reason) {
        has_bound = true;
        if (bound < max_a) {
            max_a = bound;
            limiting_reason = reason;
        }
    };

    for (size_t i = 0; i < proc.yields.size(); ++i) {
        const double y = proc.yields[i];
        const double delta = mode.scale * mode.v[i];
        if (delta > 0.0) {
            tighten(y / delta, "down_bin_sr" + std::to_string(i + 1));
        } else if (delta < 0.0) {
            tighten(y / (-delta), "up_bin_sr" + std::to_string(i + 1));
        }
    }

    const double integral_delta = modeIntegralDelta(mode);
    if (integral_delta > 0.0) {
        tighten(nominal_integral / integral_delta, "down_integral");
    } else if (integral_delta < 0.0) {
        tighten(nominal_integral / (-integral_delta), "up_integral");
    }

    if (!has_bound) return std::numeric_limits<double>::infinity();
    return max_a;
}

void regularizeModeTemplateScale(const AppConfig& cfg, const Process& proc,
                                 EigenMode& mode, const std::string& channel_name,
                                 const std::string& nuisance_name) {
    std::string limiting_reason;
    const double max_a = computePositiveTemplateScaleLimit(proc, mode, limiting_reason);
    if (!std::isfinite(max_a)) {
        throw std::runtime_error("Non-finite template scale bound for process '" + proc.name +
                                 "' in channel '" + channel_name + "'");
    }
    if (max_a <= 0.0) {
        throw std::runtime_error(
            "No positive template scale keeps process '" + proc.name + "' nuisance '" +
            nuisance_name + "' strictly positive in channel '" + channel_name + "'");
    }
    if (max_a > 1.0) return;
    if (!cfg.rescale_shape_modes_to_positive) {
        throw std::runtime_error(
            "Shape nuisance '" + nuisance_name + "' for process '" + proc.name +
            "' in channel '" + channel_name +
            "' needs template-step rescaling to stay positive, but "
            "rescale_shape_modes_to_positive=false");
    }

    const double a_used = std::min(1.0, max_a * 0.999999);
    if (!(a_used > 0.0)) {
        throw std::runtime_error(
            "Failed to build a positive template scale for process '" + proc.name +
            "' nuisance '" + nuisance_name + "' in channel '" + channel_name + "'");
    }
    mode.template_scale = a_used;

    const double nominal_integral = processIntegral(proc.yields);
    double up_integral = 0.0;
    double down_integral = 0.0;
    for (size_t i = 0; i < proc.yields.size(); ++i) {
        const double delta = mode.scale * mode.template_scale * mode.v[i];
        up_integral += proc.yields[i] + delta;
        down_integral += proc.yields[i] - delta;
    }

    logMessage("WARNING: Rescaled shape nuisance to keep templates positive: channel=" +
               channel_name + " process=" + proc.name + " syst=" + nuisance_name +
               " a_max=" + formatDouble(max_a) +
               " a_used=" + formatDouble(mode.template_scale) +
               " shape_effect=" + formatDouble(1.0 / mode.template_scale) +
               " nominal_integral=" + formatDouble(nominal_integral) +
               " up_integral=" + formatDouble(up_integral) +
               " down_integral=" + formatDouble(down_integral) +
               " limiting_reason=" + limiting_reason);
}

// -------------------- Shape ROOT + datacard --------------------
struct PerChannelCard {
    std::string name;              // channel name
    std::string datacard_path;
    int n_sr = 0;
    std::vector<int> sr_ids;
    std::vector<Process> processes;
    // For each process: eigen modes; nuisance name convention below.
    std::vector<std::vector<EigenMode>> modes;
    std::vector<double> data_obs;
};

void validateProcessShapes(const AppConfig& cfg, const Process& proc,
                           const std::vector<EigenMode>& modes,
                           const std::string& channel_name) {
    const int n = static_cast<int>(proc.yields.size());
    for (int i = 0; i < n; ++i) {
        const double y = proc.yields[i];
        if (!std::isfinite(y)) {
            throw std::runtime_error("Non-finite nominal yield for process '" + proc.name +
                                     "' in channel '" + channel_name + "'");
        }
        if (y < 0.0) {
            throw std::runtime_error("Negative nominal yield for process '" + proc.name +
                                     "' in channel '" + channel_name + "'");
        }
    }
    for (size_t k = 0; k < modes.size(); ++k) {
        double up_integral = 0.0;
        double down_integral = 0.0;
        for (int i = 0; i < n; ++i) {
            const double delta = modes[k].scale * modes[k].template_scale * modes[k].v[i];
            const double up = proc.yields[i] + delta;
            const double down = proc.yields[i] - delta;
            if (!std::isfinite(up) || !std::isfinite(down)) {
                throw std::runtime_error("Non-finite shape variation for process '" + proc.name +
                                         "' in channel '" + channel_name + "'");
            }
            if (up < 0.0 || down < 0.0) {
                throw std::runtime_error(
                    "Negative shape variation for process '" + proc.name +
                    "' in channel '" + channel_name +
                    "'; refusing to modify the input yields/covariance");
            }
            if (!cfg.rescale_shape_modes_to_positive && (up == 0.0 || down == 0.0)) {
                throw std::runtime_error(
                    "Zero-valued shape variation for process '" + proc.name +
                    "' in channel '" + channel_name +
                    "'; set rescale_shape_modes_to_positive=true to allow zero bins "
                    "while still enforcing positive template norms");
            }
            up_integral += up;
            down_integral += down;
        }
        if (!(up_integral > 0.0) || !(down_integral > 0.0)) {
            throw std::runtime_error(
                "Non-positive template integral for process '" + proc.name +
                "' in channel '" + channel_name +
                "'; combine shape interpolation requires strictly positive norms");
        }
    }
}

std::string srBinName(const std::string& channel_name, int sr_id) {
    return channel_name + "_sr" + std::to_string(sr_id);
}

void writeChannelShape(const AppConfig& cfg, const PerChannelCard& pc,
                       const std::string& shape_path) {
    if (static_cast<int>(pc.sr_ids.size()) != pc.n_sr) {
        throw std::runtime_error("Signal-region id count mismatch in channel '" + pc.name + "'");
    }
    TFile* f = TFile::Open(shape_path.c_str(), "RECREATE");
    if (f == nullptr || f->IsZombie()) {
        if (f != nullptr) delete f;
        throw std::runtime_error("Cannot create shape file: " + shape_path);
    }

    // The nominal process histograms carry the MC statistics (sqrt(sum w^2)) as
    // bin errors for autoMCStats; all other uncertainties live in nuisances.
    const bool mc_stat = nuisanceEnabled(cfg.enabled_nuisances, "mcstat");
    auto makeHist = [&](TDirectory* dir, const std::string& hname, double value, double error) {
        TH1D* h = new TH1D(hname.c_str(), hname.c_str(), 1, 0.0, 1.0);
        h->SetDirectory(dir);
        h->SetBinContent(1, value);
        h->SetBinError(1, error);
        return h;
    };

    for (int i = 0; i < pc.n_sr; ++i) {
        if (!std::isfinite(pc.data_obs[i]) || pc.data_obs[i] < 0.0) {
            throw std::runtime_error("Invalid data_obs content in channel '" + pc.name + "'");
        }
    }
    for (size_t p = 0; p < pc.processes.size(); ++p) {
        validateProcessShapes(cfg, pc.processes[p], pc.modes[p], pc.name);
    }

    for (int sr = 0; sr < pc.n_sr; ++sr) {
        TDirectory* sr_dir = f->mkdir(srBinName(pc.name, pc.sr_ids[sr]).c_str());
        sr_dir->cd();
        makeHist(sr_dir, "data_obs", pc.data_obs[sr], 0.0);

        for (size_t p = 0; p < pc.processes.size(); ++p) {
            const Process& proc = pc.processes[p];
            makeHist(sr_dir, proc.name, proc.yields[sr],
                     mc_stat ? std::sqrt(std::max(proc.mc_stat_vars[sr], 0.0)) : 0.0);

            const auto& modes = pc.modes[p];
            for (size_t k = 0; k < modes.size(); ++k) {
                const double d = modes[k].scale * modes[k].template_scale * modes[k].v[sr];
                const std::string nuis = "cov_" + pc.name + "_" + proc.name +
                                         "_eig" + std::to_string(k);
                makeHist(sr_dir, proc.name + "_" + nuis + "Up", proc.yields[sr] + d, 0.0);
                makeHist(sr_dir, proc.name + "_" + nuis + "Down", proc.yields[sr] - d, 0.0);
            }
        }
    }
    f->Write();
    f->Close();
    delete f;
}

// -------------------- lnN nuisance rows --------------------

// Effect of one lnN nuisance on one process in one SR.
struct Kappa {
    double down = 1.0;
    double up = 1.0;
    bool active = false;
    bool symmetric = false;  // written as the single value up
};

// Yield of one MC sample in one SR (0 when qcd_est.py wrote no entry: the
// sample has no test-split event in the channel).
double sampleYield(const ChannelData& ch, const std::string& sample, int sr) {
    auto it = ch.sample.find(sample);
    return it == ch.sample.end() ? 0.0 : it->second.yields[sr];
}

// Kappa of one process in one SR for a per-sample-ratio nuisance: the mean of
// its samples' ratios weighted by their SR yields (the process yield is their
// sum). Under the theory nuisances a sample without theory weights has ratio 1.
// Inactive for a process without MC yield in the SR.
Kappa ratioKappa(const RatioNuisanceSpec& spec, const SampleRatios& ratios,
                 const Process& proc, const ChannelData& ch, int sr,
                 const ClassRegistry& reg) {
    Kappa k;
    const int sr_id = ch.sr_ids[sr];
    double total = 0.0, up = 0.0, down = 0.0;
    for (const auto& sample : proc.samples) {
        const double y = sampleYield(ch, sample, sr);
        if (y <= 0.0) continue;
        total += y;
        if (spec.theory && !reg.samples.at(sample).has_theory_weights) {
            up += y;
            down += y;
            continue;
        }
        const RatioPair* r = nullptr;
        auto s_it = ratios.find(sample);
        if (s_it != ratios.end()) {
            auto c_it = s_it->second.find(ch.name);
            if (c_it != s_it->second.end()) {
                auto r_it = c_it->second.regions.find(sr_id);
                if (r_it != c_it->second.regions.end()) r = &r_it->second;
            }
        }
        if (r == nullptr) {
            throw std::runtime_error(
                std::string("nuisance '") + spec.nuisance + "' has no ratio for sample '" + sample +
                "' in channel '" + ch.name + "' SR " + std::to_string(sr_id) +
                ", where the sample has a nonzero yield; re-run " + spec.producer);
        }
        up += y * r->up;
        down += y * r->down;
    }
    if (total <= 0.0) return k;
    k.up = up / total;
    k.down = down / total;
    k.active = true;
    if (!(k.up > 0.0) || !(k.down > 0.0) || !std::isfinite(k.up) || !std::isfinite(k.down)) {
        std::ostringstream os;
        os << "nuisance '" << spec.nuisance << "' gives kappa down/up = " << k.down << "/" << k.up
           << " for process '" << proc.name << "' in channel '" << ch.name << "' SR " << sr_id
           << "; a lnN kappa must be positive (the variation removes the whole yield)";
        throw std::runtime_error(os.str());
    }
    return k;
}

std::string formatKappa(double kappa) {
    std::ostringstream os;
    os << std::setprecision(6) << std::fixed << kappa;
    return os.str();
}

// Writes one lnN row, kappas[sr][process]; an asymmetric entry is written as
// kappa_down/kappa_up (combine's order). No row when no entry is active.
void writeLnNRow(std::ofstream& ofs, const std::string& name,
                 const std::vector<std::vector<Kappa>>& kappas) {
    bool any_active = false;
    for (const auto& row : kappas)
        for (const auto& k : row) any_active = any_active || k.active;
    if (!any_active) return;
    ofs << name << " lnN";
    for (const auto& row : kappas) {
        for (const auto& k : row) {
            if (!k.active) {
                ofs << " -";
            } else if (k.symmetric) {
                ofs << " " << formatKappa(k.up);
            } else {
                ofs << " " << formatKappa(k.down) << "/" << formatKappa(k.up);
            }
        }
    }
    ofs << "\n";
}

// All lnN rows of one channel card. The MC-based nuisances act on the MC
// processes and are correlated across processes, SRs, and channels; the ABCD
// QCD prediction takes the per-channel ABCD rows instead: the non-closure
// |k - 1| (k = final_scale) and the MC statistics of the B/C/D regions, the
// SR-correlated part of its statistics (the per-SR part is in autoMCStats).
void writeLnNRows(std::ofstream& ofs, const AppConfig& cfg, const PerChannelCard& pc,
                  const ChannelData& ch, const ClassRegistry& reg) {
    const size_t n_proc = pc.processes.size();
    for (const auto& spec : kRatioNuisances) {
        auto r_it = cfg.ratios.find(spec.nuisance);
        if (r_it == cfg.ratios.end()) continue;  // not enabled
        std::vector<std::vector<Kappa>> kappas(pc.n_sr, std::vector<Kappa>(n_proc));
        for (int sr = 0; sr < pc.n_sr; ++sr) {
            for (size_t p = 0; p < n_proc; ++p) {
                if (!pc.processes[p].abcd) {
                    kappas[sr][p] = ratioKappa(spec, r_it->second, pc.processes[p], ch, sr, reg);
                }
            }
        }
        writeLnNRow(ofs, spec.nuisance, kappas);
    }

    const auto writeFlat = [&](const std::string& name, double kappa, bool on_abcd) {
        std::vector<std::vector<Kappa>> kappas(pc.n_sr, std::vector<Kappa>(n_proc));
        for (auto& row : kappas) {
            for (size_t p = 0; p < n_proc; ++p) {
                if (pc.processes[p].abcd == on_abcd) row[p] = {kappa, kappa, true, true};
            }
        }
        writeLnNRow(ofs, name, kappas);
    };
    if (nuisanceEnabled(cfg.enabled_nuisances, "lumi")) {
        writeFlat("lumi", 1.0 + cfg.lumi_unc, /*on_abcd=*/false);
    }
    if (nuisanceEnabled(cfg.enabled_nuisances, "trigger")) {
        writeFlat("trigger", 1.0 + cfg.trigger_unc, /*on_abcd=*/false);
    }
    if (nuisanceEnabled(cfg.enabled_nuisances, "abcd_nonclosure")) {
        writeFlat("abcd_nonclosure_" + pc.name, 1.0 + std::fabs(ch.abcd.final_scale - 1.0),
                  /*on_abcd=*/true);
    }
    if (nuisanceEnabled(cfg.enabled_nuisances, "abcd_mcstat")) {
        writeFlat("abcd_mcstat_" + pc.name, 1.0 + ch.abcd.pred_union_error / ch.abcd.pred_union,
                  /*on_abcd=*/true);
    }
}

void writeChannelDatacard(const AppConfig& cfg, const PerChannelCard& pc,
                          const std::string& shape_file, const ChannelData& ch,
                          const ClassRegistry& reg) {
    std::ofstream ofs(pc.datacard_path);
    if (!ofs) {
        throw std::runtime_error("Cannot write datacard: " + pc.datacard_path);
    }
    if (static_cast<int>(pc.sr_ids.size()) != pc.n_sr) {
        throw std::runtime_error("Signal-region id count mismatch in channel '" + pc.name + "'");
    }
    ofs << "# Auto-generated by combine.C\n";
    ofs << "imax " << pc.n_sr << "\njmax *\nkmax *\n";
    ofs << "----------\n";
    // One-bin shapes per SR: the nominal histograms carry the MC statistics
    // for autoMCStats; the optional covariance modes carry their variations.
    ofs << "shapes * * " << shape_file
        << " $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC\n";
    ofs << "----------\n";
    ofs << "bin";
    for (int sr = 0; sr < pc.n_sr; ++sr) {
        ofs << " " << srBinName(pc.name, pc.sr_ids[sr]);
    }
    ofs << "\n";
    ofs << "observation";
    for (int sr = 0; sr < pc.n_sr; ++sr) {
        ofs << " " << formatDouble(pc.data_obs[sr]);
    }
    ofs << "\n";
    ofs << "----------\n";

    // bin row
    ofs << "bin";
    for (int sr = 0; sr < pc.n_sr; ++sr) {
        for (size_t p = 0; p < pc.processes.size(); ++p) {
            ofs << " " << srBinName(pc.name, pc.sr_ids[sr]);
        }
    }
    ofs << "\n";
    // process names
    ofs << "process";
    for (int sr = 0; sr < pc.n_sr; ++sr) {
        for (const auto& pr : pc.processes) ofs << " " << pr.name;
    }
    ofs << "\n";
    // process indices (signal=0, backgrounds=1,2,...)
    ofs << "process";
    for (int sr = 0; sr < pc.n_sr; ++sr) {
        for (size_t p = 0; p < pc.processes.size(); ++p) ofs << " " << p;
    }
    ofs << "\n";
    // Explicit rates, equal to the shape integrals (combine checks them): a
    // process without yield in an SR has rate 0 and drops out of that bin.
    ofs << "rate";
    for (int sr = 0; sr < pc.n_sr; ++sr) {
        for (size_t p = 0; p < pc.processes.size(); ++p) {
            ofs << " " << formatDouble(pc.processes[p].yields[sr]);
        }
    }
    ofs << "\n";
    ofs << "----------\n";

    if (cfg.use_root_covariance) {
        // Shape nuisances: one correlated row per process eigenmode. The row is
        // active for that process in every SR and inactive for all other processes.
        for (size_t p = 0; p < pc.processes.size(); ++p) {
            const auto& proc = pc.processes[p];
            for (size_t k = 0; k < pc.modes[p].size(); ++k) {
                const std::string nuis = "cov_" + pc.name + "_" + proc.name +
                                         "_eig" + std::to_string(k);
                ofs << nuis << " shape";
                for (int sr = 0; sr < pc.n_sr; ++sr) {
                    for (size_t q = 0; q < pc.processes.size(); ++q) {
                        ofs << " "
                            << (q == p ? formatDouble(1.0 / pc.modes[p][k].template_scale) : "-");
                    }
                }
                ofs << "\n";
            }
        }
    }

    writeLnNRows(ofs, cfg, pc, ch, reg);

    // Bin-wise MC statistics (Barlow-Beeston-lite) from the histogram errors;
    // include-signal = 0 still models the signal statistics.
    if (nuisanceEnabled(cfg.enabled_nuisances, "mcstat")) {
        ofs << "* autoMCStats " << cfg.mc_stat_threshold << " 0 1\n";
    }
}

// -------------------- Running combine --------------------
struct CombineOutput {
    // Significance: single value
    double significance = 0.0;
    // AsymptoticLimits: quantiles
    double exp_2p5 = 0.0, exp_16 = 0.0, exp_50 = 0.0, exp_84 = 0.0, exp_97p5 = 0.0;
};

void runShell(const std::string& cmd) {
    logMessage("$ " + cmd);
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        throw std::runtime_error("Command failed (rc=" + std::to_string(rc) +
                                 "): " + cmd);
    }
}

double readSingleLimit(const std::string& root_path) {
    TFile* f = TFile::Open(root_path.c_str(), "READ");
    if (f == nullptr || f->IsZombie()) {
        if (f != nullptr) delete f;
        throw std::runtime_error("Cannot open combine output: " + root_path);
    }
    TTree* t = dynamic_cast<TTree*>(f->Get("limit"));
    if (t == nullptr) {
        delete f;
        throw std::runtime_error("Missing TTree 'limit' in " + root_path);
    }
    double limit = 0.0;
    t->SetBranchAddress("limit", &limit);
    if (t->GetEntries() < 1) {
        delete f;
        throw std::runtime_error("Empty 'limit' tree in " + root_path);
    }
    t->GetEntry(0);
    f->Close();
    delete f;
    return limit;
}

bool tryReadLimitQuantiles(const std::string& root_path, CombineOutput& out,
                           std::string& failure_reason) {
    TFile* f = TFile::Open(root_path.c_str(), "READ");
    if (f == nullptr || f->IsZombie()) {
        if (f != nullptr) delete f;
        failure_reason = "Cannot open combine output: " + root_path;
        return false;
    }
    TTree* t = dynamic_cast<TTree*>(f->Get("limit"));
    if (t == nullptr) {
        delete f;
        failure_reason = "Missing TTree 'limit' in " + root_path;
        return false;
    }
    double limit = 0.0;
    float quantile = 0.0;
    t->SetBranchAddress("limit", &limit);
    t->SetBranchAddress("quantileExpected", &quantile);
    const Long64_t n = t->GetEntries();
    bool got2p5 = false, got16 = false, got50 = false, got84 = false, got975 = false;
    for (Long64_t i = 0; i < n; ++i) {
        t->GetEntry(i);
        const float q = quantile;
        auto close = [&](float target) { return std::fabs(q - target) < 5e-3f; };
        if (close(0.025f)) { out.exp_2p5 = limit; got2p5 = true; }
        else if (close(0.16f)) { out.exp_16 = limit; got16 = true; }
        else if (close(0.5f)) { out.exp_50 = limit; got50 = true; }
        else if (close(0.84f)) { out.exp_84 = limit; got84 = true; }
        else if (close(0.975f)) { out.exp_97p5 = limit; got975 = true; }
    }
    f->Close();
    delete f;
    if (!(got2p5 && got16 && got50 && got84 && got975)) {
        failure_reason = "Missing expected limit quantiles in " + root_path;
        return false;
    }
    return true;
}

void setInfiniteExpectedLimits(CombineOutput& out) {
    const double inf = std::numeric_limits<double>::infinity();
    out.exp_2p5 = inf;
    out.exp_16 = inf;
    out.exp_50 = inf;
    out.exp_84 = inf;
    out.exp_97p5 = inf;
}

std::string csvDouble(double value) {
    if (std::isinf(value)) return "inf";
    return formatDouble(value);
}

// -------------------- Driver --------------------
void buildAndRun(const AppConfig& cfg, const ClassRegistry& reg,
                 const std::vector<ChannelData>& channels,
                 const Scenario& sc, bool use_abcd,
                 const std::string& tag_prefix,
                 CombineOutput& sig_out, CombineOutput& lim_out) {
    const std::string mode_tag = use_abcd ? "abcd" : "mc";
    const std::string scope_tag = slugify(sc.scope);
    const std::string name_tag = slugify(sc.name);
    const std::string prefix_tag = tag_prefix.empty() ? "" : slugify(tag_prefix) + "_";
    const std::string tag = prefix_tag + mode_tag + "_" + scope_tag + "_" + name_tag;
    const fs::path work = fs::path(cfg.work_dir) / tag;
    fs::create_directories(work);

    // Build per-channel cards. Shape files are only needed when optional ROOT
    // covariance nuisances are enabled.
    std::vector<std::string> card_tokens;  // for combineCards.py
    std::vector<std::string> skipped_zero_signal_channels;
    for (const auto& ch : channels) {
        PerChannelCard pc;
        pc.name = ch.name;
        pc.n_sr = ch.n_sr;
        pc.sr_ids = ch.sr_ids;
        pc.processes = buildProcesses(ch, reg, sc, use_abcd);
        pc.processes = maybeDropZeroBackgroundProcesses(cfg, std::move(pc.processes), sc, ch.name);
        const Process& signal_proc = getRequiredProcess(pc.processes, "signal", sc, ch.name);
        if (isExactlyZeroProcess(signal_proc)) {
            if (sc.scope == "sample" || channels.size() == 1u) {
                logMessage("WARNING: Dropping zero-yield zero-covariance signal channel from "
                           "scenario: channel=" + ch.name +
                           " scenario=" + sc.scope + "/" + sc.name +
                           " qcd_mode=" + mode_tag);
                skipped_zero_signal_channels.push_back(ch.name);
                continue;
            }
            throw std::runtime_error(
                "Signal process is identically zero for scenario '" + sc.scope + "/" +
                sc.name + "' in channel '" + ch.name + "'");
        }
        pc.modes.reserve(pc.processes.size());
        for (const auto& p : pc.processes) {
            if (cfg.use_root_covariance) {
                pc.modes.push_back(decomposeCov(p.cov, cfg.eigen_rel_cutoff));
            } else {
                pc.modes.emplace_back();
            }
            for (size_t k = 0; k < pc.modes.back().size(); ++k) {
                const std::string nuis = "cov_" + ch.name + "_" + p.name +
                                         "_eig" + std::to_string(k);
                regularizeModeTemplateScale(
                    cfg, p, pc.modes.back()[k], ch.name, nuis);
            }
        }
        // Asimov data_obs: sum of all process rates (signal+background).
        pc.data_obs.assign(pc.n_sr, 0.0);
        for (const auto& p : pc.processes) {
            for (int i = 0; i < pc.n_sr; ++i) pc.data_obs[i] += p.yields[i];
        }

        const std::string shape_path = (work / ("shapes_" + ch.name + ".root")).string();
        writeChannelShape(cfg, pc, shape_path);

        pc.datacard_path = (work / ("card_" + ch.name + ".txt")).string();
        // Bin names are globally unique (<channel>_sr<N>), so labels are not
        // needed and channel names stay identical to the shape directories.
        writeChannelDatacard(cfg, pc, shape_path, ch, reg);
        card_tokens.push_back(pc.datacard_path);
    }

    if (card_tokens.empty()) {
        if (sc.scope == "sample" || channels.size() == 1u) {
            std::ostringstream channels_os;
            for (size_t i = 0; i < skipped_zero_signal_channels.size(); ++i) {
                if (i) channels_os << ",";
                channels_os << skipped_zero_signal_channels[i];
            }
            logMessage("WARNING: Signal scenario is identically zero in all usable channels; "
                       "storing significance=0 and infinite expected limits: scenario=" +
                       sc.scope + "/" + sc.name +
                       " qcd_mode=" + mode_tag +
                       " skipped_channels=" + channels_os.str());
            sig_out.significance = 0.0;
            setInfiniteExpectedLimits(lim_out);
            return;
        }
        throw std::runtime_error("No usable channels remain for scenario '" + sc.scope + "/" +
                                 sc.name + "'");
    }

    // Combine per-channel cards. Run from the work directory so combineCards.py
    // produces shape paths resolvable from there, and combine picks up the same
    // cwd.
    {
        std::ostringstream os;
        os << "cd " << shellQuote(work.string()) << " && "
           << cfg.combine_cards_cmd;
        for (const auto& tok : card_tokens) os << " " << shellQuote(tok);
        os << " > " << shellQuote(std::string("datacard.txt"));
        runShell(os.str());
    }

    const std::string combine_name_sig = "sig_" + tag;
    const std::string combine_name_lim = "lim_" + tag;

    // Significance
    {
        std::ostringstream os;
        os << "cd " << shellQuote(work.string()) << " && "
           << cfg.combine_cmd
           << " -M Significance -t -1 --expectSignal 1 -m 120 "
           << " -n " << shellQuote(combine_name_sig) << " "
           << shellQuote("datacard.txt");
        runShell(os.str());
    }
    const std::string sig_root =
        (work / ("higgsCombine" + combine_name_sig +
                 ".Significance.mH120.root"))
            .string();
    sig_out.significance = readSingleLimit(sig_root);

    // AsymptoticLimits
    {
        std::ostringstream os;
        os << "cd " << shellQuote(work.string()) << " && "
           << cfg.combine_cmd
           << " -M AsymptoticLimits -t -1 -m 120 --run expected "
           << " -n " << shellQuote(combine_name_lim) << " "
           << shellQuote("datacard.txt");
        runShell(os.str());
    }
    const std::string lim_root =
        (work / ("higgsCombine" + combine_name_lim +
                 ".AsymptoticLimits.mH120.root"))
            .string();
    std::string limit_parse_failure;
    if (!tryReadLimitQuantiles(lim_root, lim_out, limit_parse_failure)) {
        if (limit_parse_failure.find("Missing expected limit quantiles") != std::string::npos) {
            logMessage("WARNING: AsymptoticLimits output is missing expected quantiles; "
                       "storing significance=0 and infinite expected limits: scenario=" +
                       sc.scope + "/" + sc.name +
                       " qcd_mode=" + mode_tag +
                       " previous_significance=" + formatDouble(sig_out.significance) +
                       " reason=" + limit_parse_failure);
            sig_out.significance = 0.0;
            setInfiniteExpectedLimits(lim_out);
        } else {
            throw std::runtime_error(limit_parse_failure);
        }
    }
}

void writeSignificanceCsv(const std::string& path,
                          const std::vector<Scenario>& scenarios,
                          const std::vector<CombineOutput>& results) {
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error("Cannot write " + path);
    ofs << "scope,name,significance\n";
    for (size_t i = 0; i < scenarios.size(); ++i) {
        ofs << scenarios[i].scope << "," << scenarios[i].name << ","
            << results[i].significance << "\n";
    }
    logMessage("Wrote " + path);
}

void writeLimitsCsv(const std::string& path,
                    const std::vector<Scenario>& scenarios,
                    const std::vector<CombineOutput>& results) {
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error("Cannot write " + path);
    ofs << "scope,name,exp_2p5,exp_16,exp_50,exp_84,exp_97p5\n";
    for (size_t i = 0; i < scenarios.size(); ++i) {
        ofs << scenarios[i].scope << "," << scenarios[i].name << ","
            << csvDouble(results[i].exp_2p5) << ","
            << csvDouble(results[i].exp_16) << ","
            << csvDouble(results[i].exp_50) << ","
            << csvDouble(results[i].exp_84) << ","
            << csvDouble(results[i].exp_97p5) << "\n";
    }
    logMessage("Wrote " + path);
}

void writeChannelSignificanceCsv(const std::string& path,
                                 const std::vector<std::string>& channel_names,
                                 const std::vector<Scenario>& scenarios,
                                 const std::vector<CombineOutput>& results) {
    if (channel_names.size() != scenarios.size() || scenarios.size() != results.size()) {
        throw std::runtime_error("Channel significance CSV row size mismatch");
    }
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error("Cannot write " + path);
    ofs << "channel,scope,name,significance\n";
    for (size_t i = 0; i < scenarios.size(); ++i) {
        ofs << channel_names[i] << "," << scenarios[i].scope << "," << scenarios[i].name << ","
            << results[i].significance << "\n";
    }
    logMessage("Wrote " + path);
}

void writeChannelLimitsCsv(const std::string& path,
                           const std::vector<std::string>& channel_names,
                           const std::vector<Scenario>& scenarios,
                           const std::vector<CombineOutput>& results) {
    if (channel_names.size() != scenarios.size() || scenarios.size() != results.size()) {
        throw std::runtime_error("Channel limits CSV row size mismatch");
    }
    std::ofstream ofs(path);
    if (!ofs) throw std::runtime_error("Cannot write " + path);
    ofs << "channel,scope,name,exp_2p5,exp_16,exp_50,exp_84,exp_97p5\n";
    for (size_t i = 0; i < scenarios.size(); ++i) {
        ofs << channel_names[i] << "," << scenarios[i].scope << "," << scenarios[i].name << ","
            << csvDouble(results[i].exp_2p5) << ","
            << csvDouble(results[i].exp_16) << ","
            << csvDouble(results[i].exp_50) << ","
            << csvDouble(results[i].exp_84) << ","
            << csvDouble(results[i].exp_97p5) << "\n";
    }
    logMessage("Wrote " + path);
}

int runMain() {
    AppConfig cfg = loadAppConfig();
    fs::create_directories(cfg.output_dir);
    fs::create_directories(cfg.work_dir);
    logMessage("combine.C: output_dir=" + cfg.output_dir);
    logMessage("combine.C: work_dir=" + cfg.work_dir);
    logMessage(std::string("combine.C: root covariance nuisances=") +
               (cfg.use_root_covariance ? "enabled" : "disabled"));
    {
        std::ostringstream msg;
        msg << "combine.C: lumi_unc=" << cfg.lumi_unc << ", trigger_unc=" << cfg.trigger_unc
            << ", mc_stat_threshold=" << cfg.mc_stat_threshold << ", ratio nuisances loaded:";
        for (const auto& kv : cfg.ratios) msg << " " << kv.first;
        logMessage(msg.str());
    }
    if (!cfg.enabled_nuisances.empty()) {
        std::string list;
        for (const auto& n : cfg.enabled_nuisances) {
            if (!list.empty()) list += ", ";
            list += n;
        }
        logMessage("combine.C: enabled_nuisances filter: {" + list + "}");
    }
    for (const auto& ch : cfg.channels) {
        logMessage("combine.C: channel=" + ch.name +
                   " root_file=" + ch.root_file +
                   " bdt_root=" + ch.bdt_root);
    }

    ClassRegistry reg = loadRegistry(cfg);
    logMessage("Loaded registry: classes=" + std::to_string(reg.class_order.size()) +
               ", signal_classes=" + std::to_string(reg.signal_classes.size()) +
               ", signal_samples=" + std::to_string(reg.signal_samples.size()) +
               ", qcd_classes=" + std::to_string(reg.qcd_classes.size()));

    const bool read_mc_stat = nuisanceEnabled(cfg.enabled_nuisances, "mcstat");
    const bool read_abcd = nuisanceEnabled(cfg.enabled_nuisances, "abcd_nonclosure") ||
                           nuisanceEnabled(cfg.enabled_nuisances, "abcd_mcstat");
    std::vector<ChannelData> channels;
    for (const auto& ch : cfg.channels) {
        channels.push_back(loadChannel(ch, read_mc_stat, read_abcd));
        validateChannelAgainstRegistry(channels.back(), reg);
    }

    std::vector<Scenario> scenarios = buildScenarios(reg);
    logMessage("Scenarios: " + std::to_string(scenarios.size()));

    std::vector<CombineOutput> sig_mc(scenarios.size());
    std::vector<CombineOutput> lim_mc(scenarios.size());
    std::vector<CombineOutput> sig_abcd(scenarios.size());
    std::vector<CombineOutput> lim_abcd(scenarios.size());

    for (size_t i = 0; i < scenarios.size(); ++i) {
        const auto& sc = scenarios[i];
        logMessage("=== Scenario: scope=" + sc.scope + " name=" + sc.name + " (MC true QCD) ===");
        buildAndRun(cfg, reg, channels, sc, /*use_abcd=*/false, "", sig_mc[i], lim_mc[i]);
        logMessage("=== Scenario: scope=" + sc.scope + " name=" + sc.name + " (ABCD QCD) ===");
        buildAndRun(cfg, reg, channels, sc, /*use_abcd=*/true, "", sig_abcd[i], lim_abcd[i]);
    }

    std::vector<std::string> channel_result_names;
    std::vector<Scenario> channel_result_scenarios;
    std::vector<CombineOutput> sig_channel_mc;
    std::vector<CombineOutput> lim_channel_mc;
    std::vector<CombineOutput> sig_channel_abcd;
    std::vector<CombineOutput> lim_channel_abcd;

    for (const auto& ch : channels) {
        std::vector<ChannelData> single_channel;
        single_channel.push_back(ch);
        const std::string tag_prefix = "channel_" + ch.name;
        for (const auto& sc : scenarios) {
            channel_result_names.push_back(ch.name);
            channel_result_scenarios.push_back(sc);
            sig_channel_mc.emplace_back();
            lim_channel_mc.emplace_back();
            sig_channel_abcd.emplace_back();
            lim_channel_abcd.emplace_back();
            const size_t idx = sig_channel_mc.size() - 1;

            logMessage("=== Channel scenario: channel=" + ch.name +
                       " scope=" + sc.scope + " name=" + sc.name +
                       " (MC true QCD) ===");
            buildAndRun(
                cfg,
                reg,
                single_channel,
                sc,
                /*use_abcd=*/false,
                tag_prefix,
                sig_channel_mc[idx],
                lim_channel_mc[idx]);
            logMessage("=== Channel scenario: channel=" + ch.name +
                       " scope=" + sc.scope + " name=" + sc.name +
                       " (ABCD QCD) ===");
            buildAndRun(
                cfg,
                reg,
                single_channel,
                sc,
                /*use_abcd=*/true,
                tag_prefix,
                sig_channel_abcd[idx],
                lim_channel_abcd[idx]);
        }
    }

    writeSignificanceCsv((fs::path(cfg.output_dir) / "significance.csv").string(),
                         scenarios, sig_mc);
    writeLimitsCsv((fs::path(cfg.output_dir) / "limits.csv").string(),
                   scenarios, lim_mc);
    writeSignificanceCsv((fs::path(cfg.output_dir) / "significance_abcd_mc.csv").string(),
                         scenarios, sig_abcd);
    writeLimitsCsv((fs::path(cfg.output_dir) / "limits_abcd_mc.csv").string(),
                   scenarios, lim_abcd);
    writeChannelSignificanceCsv(
        (fs::path(cfg.output_dir) / "significance_by_channel.csv").string(),
        channel_result_names,
        channel_result_scenarios,
        sig_channel_mc);
    writeChannelLimitsCsv(
        (fs::path(cfg.output_dir) / "limits_by_channel.csv").string(),
        channel_result_names,
        channel_result_scenarios,
        lim_channel_mc);
    writeChannelSignificanceCsv(
        (fs::path(cfg.output_dir) / "significance_by_channel_abcd_mc.csv").string(),
        channel_result_names,
        channel_result_scenarios,
        sig_channel_abcd);
    writeChannelLimitsCsv(
        (fs::path(cfg.output_dir) / "limits_by_channel_abcd_mc.csv").string(),
        channel_result_names,
        channel_result_scenarios,
        lim_channel_abcd);

    if (!cfg.keep_work) {
        std::error_code ec;
        fs::remove_all(cfg.work_dir, ec);
    }
    logMessage("combine.C done");
    return 0;
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    try {
        return runMain();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
