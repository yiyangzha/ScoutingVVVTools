// Summary: Build pileup reweight CSV files from MC Pileup_nTrueInt and three reference data pileup histograms.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <TAxis.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TFile.h>
#include <TH1.h>
#include <TROOT.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TTree.h>

#include "../src/simple_json.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
namespace fs = std::filesystem;
using JsonValue = simple_json::Value;

namespace {

const char* kConfigPath = "./config.json";
const char* kConfigEnvVar = "WEIGHT_CONFIG_PATH";
const char* kRemotePrefix = "root://cms-xrd-global.cern.ch/";

struct SampleRuleConfig {
    vector<string> containsAny;
    string category;
    vector<string> paths;
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
    string outputRoot;
    string outputPattern;
    int maxThreads = 12;
    string defaultCategory = "bkg";
    int defaultSampleType = -1;
    bool defaultIsSignal = false;
    vector<SampleRuleConfig> sampleRules;
};

struct SampleMeta {
    string sample;
    string category;
    vector<string> inputPaths;
    string outputFileName;
    int sampleType = -1;
    bool isSignal = false;
    size_t remoteSourceCount = 0;

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

bool endsWith(const string& text, const string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

string resolveConfigPath(const char* preferredPath, const char* envVar) {
    const char* envPath = getenv(envVar);
    if (envPath != nullptr && *envPath != '\0') {
        if (fs::exists(envPath)) {
            return envPath;
        }
        throw runtime_error(string("Cannot find config file from environment variable ") + envVar + ": " + envPath);
    }

    if (fs::exists(preferredPath)) {
        return preferredPath;
    }

    const string fallback = fs::path(preferredPath).filename().string();
    if (fs::exists(fallback)) {
        return fallback;
    }

    throw runtime_error("Cannot find config file: " + string(preferredPath));
}

JsonValue loadJson(const char* path, const char* envVar) {
    const string resolved = resolveConfigPath(path, envVar);
    return simple_json::parseFile(resolved);
}

bool matchesRule(const string& sample, const SampleRuleConfig& rule) {
    for (const auto& token : rule.containsAny) {
        if (sample.find(token) != string::npos) {
            return true;
        }
    }
    return false;
}

vector<string> getStringListOrScalar(const JsonValue& node, const string& key) {
    const JsonValue* child = node.find(key);
    if (child == nullptr || child->isNull()) {
        return {};
    }
    if (child->isString()) {
        return {child->asString()};
    }
    if (child->isArray()) {
        return child->toStringArray();
    }
    throw runtime_error("JSON key '" + key + "' must be a string or array of strings.");
}

string formatInputSources(const vector<string>& sources) {
    if (sources.empty()) {
        return "";
    }
    if (sources.size() == 1) {
        return sources.front();
    }

    ostringstream ss;
    for (size_t index = 0; index < sources.size(); ++index) {
        if (index != 0) {
            ss << ", ";
        }
        ss << sources[index];
    }
    return ss.str();
}

AppConfig loadAppConfig() {
    const JsonValue payload = loadJson(kConfigPath, kConfigEnvVar);

    AppConfig config;
    config.treeName = payload.getStringOr("tree_name", "Events");
    config.runSample = payload.getStringOr("run_sample", "");
    config.pileupBranch = payload.getStringOr("pileup_branch", "Pileup_nTrueInt");
    config.pileupHistogram = payload.getStringOr("pileup_histogram", "pileup");
    config.outputRoot = payload.at("output_root").asString();
    config.outputPattern = payload.at("output_pattern").asString();
    config.maxThreads = payload.getIntOr("max_threads", 12);

    if (payload.contains("pileup_data_files")) {
        for (const auto& item : payload.at("pileup_data_files").asObject()) {
            config.pileupDataFiles[item.first] = item.second.asString();
        }
    }

    if (payload.contains("defaults")) {
        const auto& defaults = payload.at("defaults");
        config.defaultCategory = defaults.getStringOr("category", config.defaultCategory);
        config.defaultSampleType = defaults.getIntOr("sample_type", config.defaultSampleType);
        config.defaultIsSignal = defaults.getBoolOr("is_signal", config.defaultIsSignal);
    }

    if (payload.contains("sample_rules")) {
        for (const auto& node : payload.at("sample_rules").asArray()) {
            SampleRuleConfig rule;
            rule.containsAny = node.at("contains_any").toStringArray();
            rule.category = node.getStringOr("category", "");
            rule.paths = getStringListOrScalar(node, "path");
            if (node.contains("sample_type")) {
                rule.sampleType = node.at("sample_type").asInt();
                rule.hasSampleType = true;
            }
            if (node.contains("is_signal")) {
                rule.isSignal = node.at("is_signal").asBool();
                rule.hasIsSignal = true;
            }
            config.sampleRules.push_back(std::move(rule));
        }
    }

    return config;
}

string resolveRequestedSample(int argc, char** argv, const AppConfig& appConfig) {
    if (argc >= 2 && argv[1] != nullptr && *argv[1] != '\0') {
        return argv[1];
    }
    if (!appConfig.runSample.empty()) {
        return appConfig.runSample;
    }
    throw runtime_error("No sample specified. Pass sample as argv[1] or set run_sample in ./config.json.");
}

bool isCmsDatasetPath(const string& path) {
    if (path.empty() || path[0] != '/' || endsWith(path, ".root")) {
        return false;
    }

    size_t parts = 0;
    string token;
    stringstream ss(path);
    while (getline(ss, token, '/')) {
        if (!token.empty()) {
            ++parts;
        }
    }
    return parts == 3;
}

bool isUserDataset(const string& path) {
    return endsWith(path, "/USER");
}

string runCommand(const string& command) {
    unique_ptr<FILE, int(*)(FILE*)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        throw runtime_error("Failed to run command: " + command);
    }

    string output;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        output += buffer;
    }

    const int status = pclose(pipe.release());
    if (status != 0) {
        throw runtime_error("Command failed (" + to_string(status) + "): " + command + "\n" + output);
    }
    return output;
}

vector<string> splitLines(const string& text) {
    vector<string> out;
    string line;
    stringstream ss(text);
    while (getline(ss, line)) {
        if (!line.empty()) {
            out.push_back(line);
        }
    }
    return out;
}

vector<string> listRemoteRootFiles(const string& datasetPath) {
    string query = "file dataset=" + datasetPath;
    if (isUserDataset(datasetPath)) {
        query += " instance=prod/phys03";
    }

    const string command = "dasgoclient -query=\"" + query + "\" 2>&1";
    vector<string> lines = splitLines(runCommand(command));
    vector<string> files;
    files.reserve(lines.size());
    for (const auto& line : lines) {
        if (endsWith(line, ".root")) {
            files.push_back(string(kRemotePrefix) + line);
        }
    }
    sort(files.begin(), files.end());
    return files;
}

vector<string> listLocalRootFiles(const string& inputPath) {
    vector<string> files;
    const fs::path path(inputPath);

    if (!fs::exists(path)) {
        throw runtime_error("Local input path does not exist: " + inputPath);
    }

    if (fs::is_regular_file(path)) {
        if (!endsWith(path.string(), ".root")) {
            throw runtime_error("Local input file is not a ROOT file: " + inputPath);
        }
        files.push_back(fs::absolute(path).string());
        return files;
    }

    if (!fs::is_directory(path)) {
        throw runtime_error("Unsupported local input path: " + inputPath);
    }

    for (const auto& entry : fs::recursive_directory_iterator(path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const string filePath = entry.path().string();
        if (endsWith(filePath, ".root")) {
            files.push_back(fs::absolute(entry.path()).string());
        }
    }

    sort(files.begin(), files.end());
    return files;
}

vector<string> discoverInputFiles(SampleMeta& sampleMeta) {
    sampleMeta.remoteSourceCount = 0;
    vector<string> files;
    for (const auto& inputPath : sampleMeta.inputPaths) {
        const bool isRemoteDataset = isCmsDatasetPath(inputPath);
        if (isRemoteDataset) {
            ++sampleMeta.remoteSourceCount;
        }

        vector<string> sourceFiles = isRemoteDataset ? listRemoteRootFiles(inputPath)
                                                     : listLocalRootFiles(inputPath);
        files.insert(files.end(), sourceFiles.begin(), sourceFiles.end());
    }

    sort(files.begin(), files.end());
    files.erase(unique(files.begin(), files.end()), files.end());
    if (files.empty()) {
        throw runtime_error("No ROOT files found for sample " + sampleMeta.sample +
                            " from configured path(s): " + formatInputSources(sampleMeta.inputPaths));
    }
    return files;
}

SampleMeta resolveSampleMeta(const string& sample, const AppConfig& appConfig) {
    SampleMeta meta;
    meta.sample = sample;
    meta.category = appConfig.defaultCategory;
    meta.sampleType = appConfig.defaultSampleType;
    meta.isSignal = appConfig.defaultIsSignal;

    vector<string> pathTemplates;
    for (const auto& rule : appConfig.sampleRules) {
        if (!matchesRule(sample, rule)) {
            continue;
        }
        if (!rule.category.empty()) {
            meta.category = rule.category;
        }
        if (!rule.paths.empty()) {
            pathTemplates = rule.paths;
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

    if (pathTemplates.empty()) {
        throw runtime_error("No input path configured for sample: " + sample);
    }

    unordered_map<string, string> templateValues;
    templateValues["sample"] = meta.sample;
    templateValues["category"] = meta.category;
    templateValues["output_root"] = appConfig.outputRoot;

    for (const auto& pathTemplate : pathTemplates) {
        const string resolvedPath = applyTemplate(pathTemplate, templateValues);
        if (find(meta.inputPaths.begin(), meta.inputPaths.end(), resolvedPath) == meta.inputPaths.end()) {
            meta.inputPaths.push_back(resolvedPath);
        }
    }
    if (meta.inputPaths.empty()) {
        throw runtime_error("No input path configured for sample: " + sample);
    }
    meta.outputFileName = applyTemplate(appConfig.outputPattern, templateValues);
    return meta;
}

string getRequiredDataFile(const AppConfig& appConfig, const string& key) {
    const auto it = appConfig.pileupDataFiles.find(key);
    if (it == appConfig.pileupDataFiles.end() || it->second.empty()) {
        throw runtime_error("Missing pileup_data_files." + key + " in ./config.json");
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
    fs::create_directories(fs::path(outputPath).parent_path());

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

string makePdfPath(const string& csvPath) {
    fs::path path(csvPath);
    path.replace_extension(".pdf");
    return path.string();
}

double histogramMaxWithError(const TH1* hist) {
    double maxValue = 0.;
    for (int bin = 1; bin <= hist->GetNbinsX(); ++bin) {
        maxValue = max(maxValue, hist->GetBinContent(bin) + hist->GetBinError(bin));
    }
    return maxValue;
}

void configureLineStyle(TH1* hist,
                        int color,
                        int lineStyle,
                        int lineWidth,
                        const char* xTitle,
                        const char* yTitle) {
    hist->SetTitle("");
    hist->SetStats(0);
    hist->SetLineColor(color);
    hist->SetMarkerColor(color);
    hist->SetMarkerSize(0.);
    hist->SetLineStyle(lineStyle);
    hist->SetLineWidth(lineWidth);
    hist->GetXaxis()->SetTitle(xTitle);
    hist->GetYaxis()->SetTitle(yTitle);
    hist->GetXaxis()->SetTitleSize(0.05);
    hist->GetYaxis()->SetTitleSize(0.05);
    hist->GetXaxis()->SetLabelSize(0.042);
    hist->GetYaxis()->SetLabelSize(0.042);
    hist->GetYaxis()->SetTitleOffset(1.15);
}

void configureBandStyle(TH1* hist, int color) {
    hist->SetStats(0);
    hist->SetLineColorAlpha(color, 0.);
    hist->SetMarkerColorAlpha(color, 0.);
    hist->SetMarkerSize(0.);
    hist->SetFillStyle(1001);
    hist->SetFillColorAlpha(color, 0.3);
}

void applyBinWeights(TH1* target, const TH1* weightsSource, const TH1* mcReference) {
    for (int bin = 1; bin <= target->GetNbinsX(); ++bin) {
        const double mcValue = mcReference->GetBinContent(bin);
        const double mcError = mcReference->GetBinError(bin);
        double weight = 0.;
        if (mcValue > 0.) {
            weight = weightsSource->GetBinContent(bin) / mcValue;
        }
        target->SetBinContent(bin, target->GetBinContent(bin) * weight);
        target->SetBinError(bin, target->GetBinError(bin) * weight);
        if (mcValue <= 0. && mcError > 0.) {
            target->SetBinError(bin, 0.);
        }
    }
}

void writePileupPdf(const string& outputPath,
                    const TH1* dataNominal,
                    const TH1* dataLow,
                    const TH1* dataHigh,
                    const TH1* mc) {
    fs::create_directories(fs::path(outputPath).parent_path());

    unique_ptr<TH1> drawDataNominal = cloneHistogram(dataNominal, "pileup_draw_data_nominal", false);
    unique_ptr<TH1> drawDataLow = cloneHistogram(dataLow, "pileup_draw_data_low", false);
    unique_ptr<TH1> drawDataHigh = cloneHistogram(dataHigh, "pileup_draw_data_high", false);
    unique_ptr<TH1> drawMc = cloneHistogram(mc, "pileup_draw_mc", false);
    unique_ptr<TH1> drawMcWeightedNominal = cloneHistogram(mc, "pileup_draw_mc_weighted_nominal", false);
    unique_ptr<TH1> drawMcWeightedLow = cloneHistogram(mc, "pileup_draw_mc_weighted_low", false);
    unique_ptr<TH1> drawMcWeightedHigh = cloneHistogram(mc, "pileup_draw_mc_weighted_high", false);
    unique_ptr<TH1> bandDataNominal = cloneHistogram(dataNominal, "pileup_band_data_nominal", false);
    unique_ptr<TH1> bandDataLow = cloneHistogram(dataLow, "pileup_band_data_low", false);
    unique_ptr<TH1> bandDataHigh = cloneHistogram(dataHigh, "pileup_band_data_high", false);
    unique_ptr<TH1> bandMc = cloneHistogram(mc, "pileup_band_mc", false);

    const int nominalColor = TColor::GetColor("#d62728");
    const int lightRedColor = TColor::GetColor("#f4a3a8");
    const int darkRedColor = TColor::GetColor("#7f0000");
    const int mcColor = kBlack;

    applyBinWeights(drawMcWeightedNominal.get(), dataNominal, mc);
    applyBinWeights(drawMcWeightedLow.get(), dataLow, mc);
    applyBinWeights(drawMcWeightedHigh.get(), dataHigh, mc);

    configureLineStyle(drawDataNominal.get(), nominalColor, 1, 3, "pileup", "A.U.");
    configureLineStyle(drawDataLow.get(), lightRedColor, 1, 3, "pileup", "A.U.");
    configureLineStyle(drawDataHigh.get(), darkRedColor, 1, 3, "pileup", "A.U.");
    configureLineStyle(drawMc.get(), mcColor, 1, 3, "pileup", "A.U.");
    configureLineStyle(drawMcWeightedNominal.get(), nominalColor, 2, 3, "pileup", "A.U.");
    configureLineStyle(drawMcWeightedLow.get(), lightRedColor, 2, 3, "pileup", "A.U.");
    configureLineStyle(drawMcWeightedHigh.get(), darkRedColor, 2, 3, "pileup", "A.U.");

    configureBandStyle(bandDataNominal.get(), nominalColor);
    configureBandStyle(bandDataLow.get(), lightRedColor);
    configureBandStyle(bandDataHigh.get(), darkRedColor);
    configureBandStyle(bandMc.get(), mcColor);

    double maxValue = 0.;
    maxValue = max(maxValue, histogramMaxWithError(drawDataNominal.get()));
    maxValue = max(maxValue, histogramMaxWithError(drawDataLow.get()));
    maxValue = max(maxValue, histogramMaxWithError(drawDataHigh.get()));
    maxValue = max(maxValue, histogramMaxWithError(drawMc.get()));
    maxValue = max(maxValue, histogramMaxWithError(drawMcWeightedNominal.get()));
    maxValue = max(maxValue, histogramMaxWithError(drawMcWeightedLow.get()));
    maxValue = max(maxValue, histogramMaxWithError(drawMcWeightedHigh.get()));
    drawDataNominal->SetMaximum(maxValue * 1.25);
    drawDataNominal->SetMinimum(0.);

    TCanvas canvas("pileup_canvas", "", 900, 700);
    canvas.SetTicks(1, 1);
    canvas.SetLeftMargin(0.125);
    canvas.SetBottomMargin(0.12);
    canvas.SetRightMargin(0.04);
    canvas.SetTopMargin(0.04);

    gStyle->SetOptStat(0);
    gStyle->SetErrorX(0.);

    drawDataNominal->Draw("HIST");
    bandDataNominal->Draw("E2 SAME");
    bandDataLow->Draw("E2 SAME");
    bandDataHigh->Draw("E2 SAME");
    bandMc->Draw("E2 SAME");
    drawDataNominal->Draw("HIST SAME");
    drawDataLow->Draw("HIST SAME");
    drawDataHigh->Draw("HIST SAME");
    drawMc->Draw("HIST SAME");
    drawMcWeightedNominal->Draw("HIST SAME");
    drawMcWeightedLow->Draw("HIST SAME");
    drawMcWeightedHigh->Draw("HIST SAME");

    TLegend legend(0.15, 0.55, 0.50, 0.90);
    legend.SetBorderSize(0);
    legend.SetFillStyle(0);
    legend.SetTextSize(0.038);
    legend.AddEntry(drawDataNominal.get(), "Data nominal", "l");
    legend.AddEntry(drawDataLow.get(), "Data low", "l");
    legend.AddEntry(drawDataHigh.get(), "Data high", "l");
    legend.AddEntry(drawMc.get(), "MC", "l");
    legend.AddEntry(drawMcWeightedNominal.get(), "MC x weight", "l");
    legend.AddEntry(drawMcWeightedLow.get(), "MC x weight low", "l");
    legend.AddEntry(drawMcWeightedHigh.get(), "MC x weight high", "l");
    legend.Draw();

    canvas.SaveAs(outputPath.c_str());
}

int determineThreadCount(int configuredThreads, size_t workItems) {
    int threads = max(1, configuredThreads);
#ifdef _OPENMP
    threads = min(threads, omp_get_max_threads());
#else
    threads = 1;
#endif
    if (workItems > 0) {
        threads = min<int>(threads, static_cast<int>(workItems));
    }
    return max(1, threads);
}

void printFileProgress(const string& sample, size_t done, size_t total) {
    ostringstream ss;
    const double percent = (total == 0) ? 100. : (100.0 * static_cast<double>(done) / static_cast<double>(total));
    ss << "\r[" << sample << "] files " << done << "/" << total
       << " (" << fixed << setprecision(1) << percent << "%)";
    cout << ss.str() << flush;
    if (done >= total) {
        cout << '\n';
    }
}

void processInputFile(const string& inputFileName,
                      const AppConfig& appConfig,
                      TH1* localHist) {
    unique_ptr<TFile> inputFile(TFile::Open(inputFileName.c_str(), "READ"));
    if (!inputFile || inputFile->IsZombie()) {
        throw runtime_error("Error opening input file " + inputFileName);
    }

    TTree* tree = static_cast<TTree*>(inputFile->Get(appConfig.treeName.c_str()));
    if (!tree) {
        throw runtime_error("Tree " + appConfig.treeName + " not found in " + inputFileName);
    }
    if (!tree->GetBranch(appConfig.pileupBranch.c_str())) {
        throw runtime_error("Branch " + appConfig.pileupBranch + " not found in " + inputFileName);
    }

    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus(appConfig.pileupBranch.c_str(), 1);
    tree->SetCacheSize(50 * 1024 * 1024);
    tree->AddBranchToCache(appConfig.pileupBranch.c_str(), true);

    Float_t pileupValue = 0.f;
    tree->SetBranchAddress(appConfig.pileupBranch.c_str(), &pileupValue);

    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t entry = 0; entry < nEntries; ++entry) {
        tree->GetEntry(entry);
        localHist->Fill(pileupValue);
    }
}

}  // namespace

int main(int argc, char** argv) {
    TH1::AddDirectory(false);

    AppConfig appConfig;
    try {
        appConfig = loadAppConfig();
    } catch (const exception& ex) {
        cerr << "Configuration error: " << ex.what() << endl;
        return 1;
    }

    string sample;
    try {
        sample = resolveRequestedSample(argc, argv, appConfig);
    } catch (const exception& ex) {
        cerr << "Sample selection error: " << ex.what() << endl;
        return 1;
    }

    SampleMeta sampleMeta;
    try {
        sampleMeta = resolveSampleMeta(sample, appConfig);
    } catch (const exception& ex) {
        cerr << "Sample resolution error: " << ex.what() << endl;
        return 1;
    }

    if (!sampleMeta.isMC()) {
        cerr << "weight only supports MC samples, but got category = " << sampleMeta.category << endl;
        return 1;
    }

    vector<string> inputFiles;
    try {
        inputFiles = discoverInputFiles(sampleMeta);
    } catch (const exception& ex) {
        cerr << "Input discovery error: " << ex.what() << endl;
        return 1;
    }

    cout << "Running weight for sample = " << sample
         << ", files = " << inputFiles.size();
    if (sampleMeta.inputPaths.size() == 1) {
        cout << ", source = " << sampleMeta.inputPaths.front()
             << (sampleMeta.remoteSourceCount == 1 ? " [dataset]" : " [local]");
    } else {
        cout << ", sources = " << sampleMeta.inputPaths.size()
             << " (dataset = " << sampleMeta.remoteSourceCount
             << ", local = " << (sampleMeta.inputPaths.size() - sampleMeta.remoteSourceCount) << ")";
    }
    cout << endl;

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
        return 1;
    }

    unique_ptr<TH1> mcHist = cloneHistogram(dataNominal.get(), "pileup_mc", true);
    const int threadCount = determineThreadCount(appConfig.maxThreads, inputFiles.size());

#ifdef _OPENMP
    if (threadCount > 1) {
        ROOT::EnableThreadSafety();
    }
#endif

    cout << "Thread mode: ";
#ifdef _OPENMP
    cout << "OpenMP";
#else
    cout << "serial";
#endif
    cout << ", threads = " << threadCount << endl;

    atomic<size_t> processedFiles{0};
    atomic<bool> failed{false};
    vector<string> errors;

#pragma omp parallel num_threads(threadCount) if(threadCount > 1)
    {
        unique_ptr<TH1> localHist = cloneHistogram(dataNominal.get(), "pileup_mc_local", true);

#pragma omp for schedule(dynamic)
        for (int index = 0; index < static_cast<int>(inputFiles.size()); ++index) {
            if (failed.load()) {
                continue;
            }

            try {
                processInputFile(inputFiles[index], appConfig, localHist.get());
                const size_t done = processedFiles.fetch_add(1) + 1;
#pragma omp critical(weight_progress)
                printFileProgress(sampleMeta.sample, done, inputFiles.size());
            } catch (const exception& ex) {
                failed.store(true);
#pragma omp critical(weight_error)
                errors.push_back(ex.what());
            }
        }

#pragma omp critical(weight_merge)
        mcHist->Add(localHist.get());
    }

    if (!errors.empty()) {
        cerr << "Runtime error: " << errors.front() << endl;
        return 1;
    }

    try {
        normalizeHistogram(dataNominal.get(), "data nominal");
        normalizeHistogram(dataLow.get(), "data low");
        normalizeHistogram(dataHigh.get(), "data high");
        normalizeHistogram(mcHist.get(), "mc");
        writeCsv(sampleMeta.outputFileName, dataNominal.get(), dataNominal.get(), dataLow.get(), dataHigh.get(), mcHist.get());
        writePileupPdf(makePdfPath(sampleMeta.outputFileName), dataNominal.get(), dataLow.get(), dataHigh.get(), mcHist.get());
    } catch (const exception& ex) {
        cerr << "Output error: " << ex.what() << endl;
        return 1;
    }

    cout << "Wrote pileup weights to " << sampleMeta.outputFileName << endl;
    cout << "Wrote pileup plot to " << makePdfPath(sampleMeta.outputFileName) << endl;
    return 0;
}
