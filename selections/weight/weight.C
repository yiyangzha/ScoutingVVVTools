// Summary: Build pileup reweight CSV files from MC Pileup_nTrueInt and three reference data pileup histograms.
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <TAxis.h>
#include <TFile.h>
#include <TH1.h>
#include <TSystem.h>
#include <TTree.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

namespace {

const char* kConfigPath = "weight/config.json";
const char* kConfigEnvVar = "WEIGHT_CONFIG_PATH";

struct SampleRuleConfig {
    vector<string> containsAny;
    string category;
    string inputRootKey;
    int sampleType = 0;
    bool hasSampleType = false;
    bool isSignal = false;
    bool hasIsSignal = false;
};

struct AppConfig {
    string treeName = "Events";
    string runSample;
    string pileupBranch = "Pileup_nTrueInt";
    string pileupHistogram = "pileup";
    unordered_map<string, string> pileupDataFiles;
    unordered_map<string, string> inputRoots;
    string outputRoot;
    string inputPattern;
    string outputPattern;
    string defaultCategory = "bkg";
    string defaultInputRootKey = "mc";
    int defaultSampleType = -1;
    bool defaultIsSignal = false;
    vector<SampleRuleConfig> sampleRules;
};

struct SampleMeta {
    string sample;
    string category;
    string inputRootKey;
    string inputRoot;
    string inputFileName;
    string outputFileName;
    int sampleType = -1;
    bool isSignal = false;

    bool isMC() const {
        return category != "data";
    }
};

struct WeightTriplet {
    double nominal = 0.;
    double low = 0.;
    double high = 0.;
};

string replaceAll(string text, const string& from, const string& to) {
    if (from.empty()) {
        return text;
    }

    size_t pos = 0;
    while ((pos = text.find(from, pos)) != string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

string applyTemplate(string text, const unordered_map<string, string>& values) {
    for (const auto& item : values) {
        text = replaceAll(text, "{" + item.first + "}", item.second);
    }
    return text;
}

string resolveConfigPath(const char* preferredPath, const char* envVar) {
    const char* envPath = gSystem->Getenv(envVar);
    if (envPath != nullptr && *envPath != '\0') {
        if (!gSystem->AccessPathName(envPath)) {
            return envPath;
        }
        throw runtime_error(string("Cannot find config file from environment variable ") + envVar + ": " + envPath);
    }

    if (!gSystem->AccessPathName(preferredPath)) {
        return preferredPath;
    }

    const string fallback = gSystem->BaseName(preferredPath);
    if (!gSystem->AccessPathName(fallback.c_str())) {
        return fallback;
    }

    throw runtime_error("Cannot find config file: " + string(preferredPath));
}

json loadJson(const char* path, const char* envVar) {
    const string resolved = resolveConfigPath(path, envVar);
    ifstream fin(resolved);
    if (!fin) {
        throw runtime_error("Failed to open JSON file: " + resolved);
    }

    json payload;
    fin >> payload;
    return payload;
}

bool matchesRule(const string& sample, const SampleRuleConfig& rule) {
    for (const auto& token : rule.containsAny) {
        if (sample.find(token) != string::npos) {
            return true;
        }
    }
    return false;
}

AppConfig loadAppConfig() {
    const json payload = loadJson(kConfigPath, kConfigEnvVar);

    AppConfig config;
    config.treeName = payload.value("tree_name", "Events");
    config.runSample = payload.value("run_sample", "");
    config.pileupBranch = payload.value("pileup_branch", "Pileup_nTrueInt");
    config.pileupHistogram = payload.value("pileup_histogram", "pileup");

    if (payload.contains("pileup_data_files")) {
        for (auto it = payload.at("pileup_data_files").begin(); it != payload.at("pileup_data_files").end(); ++it) {
            config.pileupDataFiles[it.key()] = it.value().get<string>();
        }
    }

    if (payload.contains("input_roots")) {
        for (auto it = payload.at("input_roots").begin(); it != payload.at("input_roots").end(); ++it) {
            config.inputRoots[it.key()] = it.value().get<string>();
        }
    }

    config.outputRoot = payload.at("output_root").get<string>();
    config.inputPattern = payload.at("input_pattern").get<string>();
    config.outputPattern = payload.at("output_pattern").get<string>();

    if (payload.contains("defaults")) {
        const auto& defaults = payload.at("defaults");
        config.defaultCategory = defaults.value("category", config.defaultCategory);
        config.defaultInputRootKey = defaults.value("input_root_key", config.defaultInputRootKey);
        config.defaultSampleType = defaults.value("sample_type", config.defaultSampleType);
        config.defaultIsSignal = defaults.value("is_signal", config.defaultIsSignal);
    }

    if (payload.contains("sample_rules")) {
        for (const auto& node : payload.at("sample_rules")) {
            SampleRuleConfig rule;
            rule.containsAny = node.at("contains_any").get<vector<string>>();
            if (node.contains("category")) {
                rule.category = node.at("category").get<string>();
            }
            if (node.contains("input_root_key")) {
                rule.inputRootKey = node.at("input_root_key").get<string>();
            }
            if (node.contains("sample_type")) {
                rule.sampleType = node.at("sample_type").get<int>();
                rule.hasSampleType = true;
            }
            if (node.contains("is_signal")) {
                rule.isSignal = node.at("is_signal").get<bool>();
                rule.hasIsSignal = true;
            }
            config.sampleRules.push_back(std::move(rule));
        }
    }

    return config;
}

string resolveRequestedSample(const char* typeArg, const AppConfig& appConfig) {
    if (typeArg != nullptr && *typeArg != '\0') {
        return typeArg;
    }
    if (!appConfig.runSample.empty()) {
        return appConfig.runSample;
    }
    throw runtime_error("No sample specified. Pass typeArg explicitly or set run_sample in weight/config.json.");
}

SampleMeta resolveSampleMeta(const string& sample, const AppConfig& appConfig) {
    SampleMeta meta;
    meta.sample = sample;
    meta.category = appConfig.defaultCategory;
    meta.inputRootKey = appConfig.defaultInputRootKey;
    meta.sampleType = appConfig.defaultSampleType;
    meta.isSignal = appConfig.defaultIsSignal;

    for (const auto& rule : appConfig.sampleRules) {
        if (!matchesRule(sample, rule)) {
            continue;
        }
        if (!rule.category.empty()) {
            meta.category = rule.category;
        }
        if (!rule.inputRootKey.empty()) {
            meta.inputRootKey = rule.inputRootKey;
        }
        if (rule.hasSampleType) {
            meta.sampleType = rule.sampleType;
        }
        if (rule.hasIsSignal) {
            meta.isSignal = rule.isSignal;
        } else {
            meta.isSignal = (meta.category == "signal");
        }
        break;
    }

    const auto inputRootIt = appConfig.inputRoots.find(meta.inputRootKey);
    if (inputRootIt == appConfig.inputRoots.end()) {
        throw runtime_error("Unknown input root key: " + meta.inputRootKey);
    }

    meta.inputRoot = inputRootIt->second;
    unordered_map<string, string> templateValues;
    templateValues["sample"] = meta.sample;
    templateValues["category"] = meta.category;
    templateValues["input_root"] = meta.inputRoot;
    templateValues["output_root"] = appConfig.outputRoot;

    meta.inputFileName = applyTemplate(appConfig.inputPattern, templateValues);
    meta.outputFileName = applyTemplate(appConfig.outputPattern, templateValues);
    return meta;
}

string getRequiredDataFile(const AppConfig& appConfig, const string& key) {
    const auto it = appConfig.pileupDataFiles.find(key);
    if (it == appConfig.pileupDataFiles.end() || it->second.empty()) {
        throw runtime_error("Missing pileup_data_files." + key + " in weight/config.json");
    }
    return it->second;
}

unique_ptr<TH1> cloneHistogram(const TH1* source, const string& name, bool reset) {
    TH1* clone = dynamic_cast<TH1*>(source->Clone(name.c_str()));
    if (!clone) {
        throw runtime_error("Failed to clone histogram: " + string(source->GetName()));
    }
    clone->SetDirectory(nullptr);
    if (reset) {
        clone->Reset("ICES");
    }
    return unique_ptr<TH1>(clone);
}

unique_ptr<TH1> loadHistogram(const string& filePath, const string& histName, const string& cloneName) {
    unique_ptr<TFile> file(TFile::Open(filePath.c_str(), "READ"));
    if (!file || file->IsZombie()) {
        throw runtime_error("Cannot open pileup ROOT file: " + filePath);
    }

    TH1* hist = dynamic_cast<TH1*>(file->Get(histName.c_str()));
    if (!hist) {
        throw runtime_error("Cannot find histogram " + histName + " in " + filePath);
    }

    return cloneHistogram(hist, cloneName, false);
}

void ensureSameBinning(const TH1* reference, const TH1* other, const string& label) {
    if (reference->GetNbinsX() != other->GetNbinsX()) {
        throw runtime_error("Pileup histogram bin count mismatch for " + label);
    }

    const TAxis* refAxis = reference->GetXaxis();
    const TAxis* otherAxis = other->GetXaxis();
    for (int bin = 1; bin <= reference->GetNbinsX() + 1; ++bin) {
        const double refEdge = refAxis->GetBinLowEdge(bin);
        const double otherEdge = otherAxis->GetBinLowEdge(bin);
        if (fabs(refEdge - otherEdge) > 1e-9) {
            throw runtime_error("Pileup histogram bin edge mismatch for " + label);
        }
    }
}

double visibleIntegral(const TH1* hist) {
    return hist->Integral(1, hist->GetNbinsX());
}

void normalizeHistogram(TH1* hist, const string& label) {
    const double integral = visibleIntegral(hist);
    if (integral <= 0.) {
        throw runtime_error("Histogram has non-positive visible integral: " + label);
    }
    hist->Scale(1. / integral);
}

WeightTriplet computeWeights(const TH1* dataNominal,
                             const TH1* dataLow,
                             const TH1* dataHigh,
                             const TH1* mc,
                             int bin) {
    const double mcValue = mc->GetBinContent(bin);
    if (mcValue <= 0.) {
        return {};
    }

    WeightTriplet out;
    out.nominal = dataNominal->GetBinContent(bin) / mcValue;
    out.low = dataLow->GetBinContent(bin) / mcValue;
    out.high = dataHigh->GetBinContent(bin) / mcValue;
    return out;
}

void writeCsv(const string& outputPath,
              const TH1* reference,
              const TH1* dataNominal,
              const TH1* dataLow,
              const TH1* dataHigh,
              const TH1* mc) {
    gSystem->mkdir(gSystem->DirName(outputPath.c_str()), true);

    ofstream fout(outputPath);
    if (!fout) {
        throw runtime_error("Cannot open output CSV: " + outputPath);
    }

    fout << "bin_low,bin_max,weight,weight_low,weight_high\n";
    fout << fixed << setprecision(10);
    for (int bin = 1; bin <= reference->GetNbinsX(); ++bin) {
        const WeightTriplet weights = computeWeights(dataNominal, dataLow, dataHigh, mc, bin);
        fout << reference->GetXaxis()->GetBinLowEdge(bin) << ","
             << reference->GetXaxis()->GetBinUpEdge(bin) << ","
             << weights.nominal << ","
             << weights.low << ","
             << weights.high << "\n";
    }
}

}  // namespace

void weight(const char* typeArg = nullptr) {
    AppConfig appConfig;
    try {
        appConfig = loadAppConfig();
    } catch (const exception& ex) {
        cerr << "Configuration error: " << ex.what() << endl;
        return;
    }

    string sample;
    try {
        sample = resolveRequestedSample(typeArg, appConfig);
    } catch (const exception& ex) {
        cerr << "Sample selection error: " << ex.what() << endl;
        return;
    }

    cout << "Running weight with sample = " << sample << endl;

    SampleMeta sampleMeta;
    try {
        sampleMeta = resolveSampleMeta(sample, appConfig);
    } catch (const exception& ex) {
        cerr << "Sample resolution error: " << ex.what() << endl;
        return;
    }

    if (!sampleMeta.isMC()) {
        cerr << "weight.C only supports MC samples, but got category = " << sampleMeta.category << endl;
        return;
    }

    unique_ptr<TH1> dataNominal;
    unique_ptr<TH1> dataLow;
    unique_ptr<TH1> dataHigh;
    try {
        dataNominal = loadHistogram(getRequiredDataFile(appConfig, "nominal"), appConfig.pileupHistogram, "pileup_data_nominal");
        dataLow = loadHistogram(getRequiredDataFile(appConfig, "low"), appConfig.pileupHistogram, "pileup_data_low");
        dataHigh = loadHistogram(getRequiredDataFile(appConfig, "high"), appConfig.pileupHistogram, "pileup_data_high");
        ensureSameBinning(dataNominal.get(), dataLow.get(), "low");
        ensureSameBinning(dataNominal.get(), dataHigh.get(), "high");
    } catch (const exception& ex) {
        cerr << "Pileup histogram error: " << ex.what() << endl;
        return;
    }

    unique_ptr<TH1> mcHist = cloneHistogram(dataNominal.get(), "pileup_mc", true);

    unique_ptr<TFile> inputFile(TFile::Open(sampleMeta.inputFileName.c_str(), "READ"));
    if (!inputFile || inputFile->IsZombie()) {
        cerr << "Error opening input file " << sampleMeta.inputFileName << endl;
        return;
    }

    TTree* tree = static_cast<TTree*>(inputFile->Get(appConfig.treeName.c_str()));
    if (!tree) {
        cerr << "Error: tree " << appConfig.treeName << " not found in " << sampleMeta.inputFileName << endl;
        return;
    }

    if (!tree->GetBranch(appConfig.pileupBranch.c_str())) {
        cerr << "Error: branch " << appConfig.pileupBranch << " not found in " << sampleMeta.inputFileName << endl;
        return;
    }

    Float_t pileupValue = 0.f;
    tree->SetBranchAddress(appConfig.pileupBranch.c_str(), &pileupValue);

    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t entry = 0; entry < nEntries; ++entry) {
        if (entry % 500000 == 0) {
            cout << "Processing entry " << entry << " / " << nEntries << endl;
        }
        tree->GetEntry(entry);
        mcHist->Fill(pileupValue);
    }

    try {
        normalizeHistogram(dataNominal.get(), "data nominal");
        normalizeHistogram(dataLow.get(), "data low");
        normalizeHistogram(dataHigh.get(), "data high");
        normalizeHistogram(mcHist.get(), "mc");
        writeCsv(sampleMeta.outputFileName, dataNominal.get(), dataNominal.get(), dataLow.get(), dataHigh.get(), mcHist.get());
    } catch (const exception& ex) {
        cerr << "Output error: " << ex.what() << endl;
        return;
    }

    cout << "Wrote pileup weights to " << sampleMeta.outputFileName << endl;
}
