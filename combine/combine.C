// combine.C
//
// Build CMS combine datacards from qcd_est.py output and run Significance +
// AsymptoticLimits. Each input ROOT file is one channel with its matching
// BDT output directory; channels are concatenated with combineCards.py.
// By default, statistical uncertainty comes from combine's binned Poisson
// likelihood. Stored ROOT covariance blocks can optionally be injected as
// extra eigen-decomposed Gaussian shape nuisances.
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
};

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
    return cfg;
}

// -------------------- Sample / class registry --------------------
struct SampleInfo {
    std::string name;
    bool is_MC = false;
    bool is_signal = false;
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
    TMatrixDSym cov;                // n x n
    YieldCov() = default;
    explicit YieldCov(int n) : yields(n, 0.0), cov(n) {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) cov(i, j) = 0.0;
    }
    YieldCov(const YieldCov&) = default;
    YieldCov(YieldCov&&) noexcept = default;
    YieldCov& operator=(const YieldCov& other) {
        if (this == &other) return *this;
        yields = other.yields;
        cov.ResizeTo(other.cov);
        cov = other.cov;
        return *this;
    }
    YieldCov& operator=(YieldCov&& other) noexcept {
        if (this == &other) return *this;
        yields = std::move(other.yields);
        cov.ResizeTo(other.cov);
        cov = other.cov;
        return *this;
    }
    int n() const { return static_cast<int>(yields.size()); }
};

struct ChannelData {
    std::string name;
    int n_sr = 0;
    std::map<std::string, YieldCov> sample;       // per MC sample (MC true)
    std::map<std::string, YieldCov> group;        // per BDT class (MC true)
    YieldCov qcd_predict;                         // merged ABCD QCD prediction
};

YieldCov readYieldCov(TFile& f, const std::string& prefix) {
    // TFile owns the returned histograms; just copy the contents out.
    TH1* h = dynamic_cast<TH1*>(f.Get((prefix + "/yield").c_str()));
    TH2* h2 = dynamic_cast<TH2*>(f.Get((prefix + "/covariance_total").c_str()));
    if (h == nullptr || h2 == nullptr) {
        throw std::runtime_error("Missing '" + prefix + "/yield' or '" + prefix +
                                 "/covariance_total' in " + f.GetName());
    }
    const int n = h->GetNbinsX();
    if (h2->GetNbinsX() != n || h2->GetNbinsY() != n) {
        throw std::runtime_error("Covariance size mismatch for " + prefix);
    }
    YieldCov out(n);
    for (int i = 0; i < n; ++i) out.yields[i] = h->GetBinContent(i + 1);
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

ChannelData loadChannel(const ChannelSpec& spec) {
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

    // Samples
    const std::vector<std::string> sample_names = listSubdirs(*f, "samples");
    if (sample_names.empty()) {
        throw std::runtime_error("No samples/ entries in " + spec.root_file);
    }
    for (const auto& s : sample_names) {
        auto inserted = data.sample.emplace(s, readYieldCov(*f, "samples/" + s));
        if (!inserted.second) {
            throw std::runtime_error("Duplicate sample '" + s + "' in " + spec.root_file);
        }
    }
    data.n_sr = data.sample.begin()->second.n();
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
        auto inserted = data.group.emplace(key, readYieldCov(*f, "groups/" + g));
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

    data.qcd_predict = readYieldCov(*f, "qcd_predict");
    if (data.qcd_predict.n() != data.n_sr) {
        throw std::runtime_error("qcd_predict SR count mismatch for " + spec.name);
    }

    logMessage("  channel '" + data.name + "': n_sr=" + std::to_string(data.n_sr) +
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
    std::string name;            // e.g. "signal", "bkg_vh", "bkg_qcd"
    std::vector<double> yields;
    TMatrixDSym cov;
    Process() : cov(1) {}
};

bool isQcdClass(const ClassRegistry& reg, const std::string& class_name) {
    return reg.qcd_class_set.count(class_name) != 0u;
}

void addYieldCov(std::vector<double>& yields, TMatrixDSym& cov, const YieldCov& src) {
    if (yields.empty()) {
        yields = std::vector<double>(src.n(), 0.0);
        cov.ResizeTo(src.n(), src.n());
        for (int i = 0; i < src.n(); ++i)
            for (int j = 0; j < src.n(); ++j) cov(i, j) = 0.0;
    }
    const int n = static_cast<int>(yields.size());
    if (src.n() != n) throw std::runtime_error("SR size mismatch in addYieldCov");
    for (int i = 0; i < n; ++i) yields[i] += src.yields[i];
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) cov(i, j) += src.cov(i, j);
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
                                const std::string& label) {
    validateYieldCov(yc, label);
    Process out;
    out.name = name;
    out.yields = yc.yields;
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
            addYieldCov(sig.yields, sig.cov, yc);
        }
    } else if (sc.scope == "class") {
        if (!reg.signal_classes.count(sc.name)) {
            throw std::runtime_error("Scenario class '" + sc.name + "' is not a signal class");
        }
        const YieldCov& yc = getRequiredGroupYieldCov(ch, sc.name);
        addYieldCov(sig.yields, sig.cov, yc);
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
                                  "bkg_qcd",
                                  ch.qcd_predict,
                                  "qcd_predict in channel '" + ch.name + "'"),
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
                          "group '" + cls + "' in channel '" + ch.name + "'"),
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
                      "signal sample '" + signal_sample + "' in channel '" + ch.name + "'"),
                  "sample scenario '" + signal_sample + "' in channel '" + ch.name + "'");

    bool qcd_predict_added = false;
    for (const auto& cls : reg.class_order) {
        if (use_abcd && isQcdClass(reg, cls)) {
            if (!qcd_predict_added) {
                appendProcess(out, process_names,
                              makeProcessFromYieldCov(
                                  "bkg_qcd",
                                  ch.qcd_predict,
                                  "qcd_predict in channel '" + ch.name + "'"),
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
                              "sample '" + sample_name + "' in channel '" + ch.name + "'"),
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

void writeChannelShape(const AppConfig& cfg, const PerChannelCard& pc,
                       const std::string& shape_path) {
    TFile* f = TFile::Open(shape_path.c_str(), "RECREATE");
    if (f == nullptr || f->IsZombie()) {
        if (f != nullptr) delete f;
        throw std::runtime_error("Cannot create shape file: " + shape_path);
    }
    TDirectory* ch_dir = f->mkdir(pc.name.c_str());
    ch_dir->cd();

    auto makeHist = [&](const std::string& hname,
                        const std::vector<double>& vals) {
        TH1D* h = new TH1D(hname.c_str(), hname.c_str(), pc.n_sr, 0.0,
                           static_cast<double>(pc.n_sr));
        h->SetDirectory(ch_dir);
        for (int i = 0; i < pc.n_sr; ++i) {
            h->SetBinContent(i + 1, vals[i]);
            h->SetBinError(i + 1, 0.0);  // uncertainties live in nuisances
        }
        return h;
    };

    for (int i = 0; i < pc.n_sr; ++i) {
        if (!std::isfinite(pc.data_obs[i]) || pc.data_obs[i] < 0.0) {
            throw std::runtime_error("Invalid data_obs content in channel '" + pc.name + "'");
        }
    }
    makeHist("data_obs", pc.data_obs);

    for (size_t p = 0; p < pc.processes.size(); ++p) {
        const Process& proc = pc.processes[p];
        validateProcessShapes(cfg, proc, pc.modes[p], pc.name);
        makeHist(proc.name, proc.yields);

        const auto& modes = pc.modes[p];
        for (size_t k = 0; k < modes.size(); ++k) {
            std::vector<double> up(pc.n_sr), down(pc.n_sr);
            for (int i = 0; i < pc.n_sr; ++i) {
                const double d = modes[k].scale * modes[k].template_scale * modes[k].v[i];
                up[i] = proc.yields[i] + d;
                down[i] = proc.yields[i] - d;
            }
            const std::string nuis = "cov_" + pc.name + "_" + proc.name +
                                     "_eig" + std::to_string(k);
            makeHist(proc.name + "_" + nuis + "Up", up);
            makeHist(proc.name + "_" + nuis + "Down", down);
        }
    }
    f->Write();
    f->Close();
    delete f;
}

void writeChannelDatacard(const PerChannelCard& pc, const std::string& shape_file) {
    std::ofstream ofs(pc.datacard_path);
    if (!ofs) {
        throw std::runtime_error("Cannot write datacard: " + pc.datacard_path);
    }
    ofs << "# Auto-generated by combine.C\n";
    ofs << "imax 1\njmax *\nkmax *\n";
    ofs << "----------\n";
    ofs << "shapes * " << pc.name << " " << shape_file
        << " $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC\n";
    ofs << "----------\n";
    ofs << "bin " << pc.name << "\n";
    ofs << "observation -1\n";
    ofs << "----------\n";

    // bin row
    ofs << "bin";
    for (size_t p = 0; p < pc.processes.size(); ++p) ofs << " " << pc.name;
    ofs << "\n";
    // process names
    ofs << "process";
    for (const auto& pr : pc.processes) ofs << " " << pr.name;
    ofs << "\n";
    // process indices (signal=0, backgrounds=1,2,...)
    ofs << "process";
    for (size_t p = 0; p < pc.processes.size(); ++p) ofs << " " << p;
    ofs << "\n";
    // rates (-1 => from histogram integral)
    ofs << "rate";
    for (size_t p = 0; p < pc.processes.size(); ++p) ofs << " -1";
    ofs << "\n";
    ofs << "----------\n";

    // shape nuisances: one per process per eigen mode, listed in one row each.
    for (size_t p = 0; p < pc.processes.size(); ++p) {
        const auto& proc = pc.processes[p];
        for (size_t k = 0; k < pc.modes[p].size(); ++k) {
            const std::string nuis = "cov_" + pc.name + "_" + proc.name +
                                     "_eig" + std::to_string(k);
            ofs << nuis << " shape";
            for (size_t q = 0; q < pc.processes.size(); ++q) {
                ofs << " " << (q == p ? formatDouble(1.0 / pc.modes[p][k].template_scale) : "-");
            }
            ofs << "\n";
        }
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
                 CombineOutput& sig_out, CombineOutput& lim_out) {
    const std::string mode_tag = use_abcd ? "abcd" : "mc";
    const std::string scope_tag = slugify(sc.scope);
    const std::string name_tag = slugify(sc.name);
    const std::string tag = mode_tag + "_" + scope_tag + "_" + name_tag;
    const fs::path work = fs::path(cfg.work_dir) / tag;
    fs::create_directories(work);

    // Build per-channel cards + one shape file per channel.
    std::vector<std::string> card_tokens;  // for combineCards.py: chname=path
    std::vector<std::string> skipped_zero_signal_channels;
    for (const auto& ch : channels) {
        PerChannelCard pc;
        pc.name = ch.name;
        pc.n_sr = ch.n_sr;
        pc.processes = buildProcesses(ch, reg, sc, use_abcd);
        pc.processes = maybeDropZeroBackgroundProcesses(cfg, std::move(pc.processes), sc, ch.name);
        const Process& signal_proc = getRequiredProcess(pc.processes, "signal", sc, ch.name);
        if (isExactlyZeroProcess(signal_proc)) {
            if (sc.scope == "sample") {
                logMessage("WARNING: Dropping zero-yield zero-covariance signal channel from "
                           "sample scenario: channel=" + ch.name +
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
        // Asimov data_obs: sum of all processes (signal+background).
        pc.data_obs.assign(pc.n_sr, 0.0);
        for (const auto& p : pc.processes) {
            for (int i = 0; i < pc.n_sr; ++i) pc.data_obs[i] += p.yields[i];
        }

        const std::string shape_path =
            (work / ("shapes_" + ch.name + ".root")).string();
        writeChannelShape(cfg, pc, shape_path);

        pc.datacard_path = (work / ("card_" + ch.name + ".txt")).string();
        // Use absolute shape paths so combineCards.py preserves them verbatim.
        writeChannelDatacard(pc, shape_path);
        card_tokens.push_back(ch.name + "=" + pc.datacard_path);
    }

    if (card_tokens.empty()) {
        if (sc.scope == "sample") {
            std::ostringstream channels_os;
            for (size_t i = 0; i < skipped_zero_signal_channels.size(); ++i) {
                if (i) channels_os << ",";
                channels_os << skipped_zero_signal_channels[i];
            }
            logMessage("WARNING: Signal sample scenario is identically zero in all channels; "
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

int runMain() {
    AppConfig cfg = loadAppConfig();
    fs::create_directories(cfg.output_dir);
    fs::create_directories(cfg.work_dir);
    logMessage("combine.C: output_dir=" + cfg.output_dir);
    logMessage("combine.C: work_dir=" + cfg.work_dir);
    logMessage(std::string("combine.C: root covariance nuisances=") +
               (cfg.use_root_covariance ? "enabled" : "disabled"));
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

    std::vector<ChannelData> channels;
    for (const auto& ch : cfg.channels) {
        channels.push_back(loadChannel(ch));
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
        buildAndRun(cfg, reg, channels, sc, /*use_abcd=*/false, sig_mc[i], lim_mc[i]);
        logMessage("=== Scenario: scope=" + sc.scope + " name=" + sc.name + " (ABCD QCD) ===");
        buildAndRun(cfg, reg, channels, sc, /*use_abcd=*/true, sig_abcd[i], lim_abcd[i]);
    }

    writeSignificanceCsv((fs::path(cfg.output_dir) / "significance.csv").string(),
                         scenarios, sig_mc);
    writeLimitsCsv((fs::path(cfg.output_dir) / "limits.csv").string(),
                   scenarios, lim_mc);
    writeSignificanceCsv((fs::path(cfg.output_dir) / "significance_abcd_mc.csv").string(),
                         scenarios, sig_abcd);
    writeLimitsCsv((fs::path(cfg.output_dir) / "limits_abcd_mc.csv").string(),
                   scenarios, lim_abcd);

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
