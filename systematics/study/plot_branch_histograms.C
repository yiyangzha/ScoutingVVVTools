// Summary: Draw configured scalar or C-array ROOT branches from ROOT files.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include "TBranch.h"
#include "TCanvas.h"
#include "TChain.h"
#include "TFile.h"
#include "TH1D.h"
#include "TLeaf.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TTree.h"

using namespace std;
namespace fs = std::filesystem;

namespace {

static const vector<string> INPUT_FILES = {
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_1.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_2.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_3.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_4.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_5.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_6.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_7.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_8.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_9.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_10.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_11.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_12.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_13.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_14.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I_15.root",
    "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/2024I.root"
};

static const string TREE_NAME = "Events";
static const string OUTPUT_DIR = "systematics/";
static const string OUTPUT_ROOT_FILE = "histograms.root";
static const string XROOTD_REDIRECTOR = "root://cms-xrd-global.cern.ch/";
static const long long REMOTE_DATASET_FILE_NUMBER_MIN = 4820;
static const long long REMOTE_DATASET_FILE_NUMBER_MAX = 5400;
static const int DEFAULT_BINS = 100;
static const double LOG_AXIS_MIN = 0.1;

struct BranchPlotConfig {
    string name;
    double xmin = 0.;
    double xmax = 1.;
    bool logx = false;
    int bins = DEFAULT_BINS;
};

static const vector<BranchPlotConfig> BRANCHES = {
    {"ScoutingFatPFJetRecluster_pt", 100, 2000., true}
};

enum class NumericType {
    Float,
    Double,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    Long64,
    ULong64,
    Char,
    UChar,
    Bool
};

struct NumericBuffer {
    NumericType type = NumericType::Float;
    size_t size = 1;

    vector<Float_t> floats;
    vector<Double_t> doubles;
    vector<Short_t> shorts;
    vector<UShort_t> ushorts;
    vector<Int_t> ints;
    vector<UInt_t> uints;
    vector<Long_t> longs;
    vector<ULong_t> ulongs;
    vector<Long64_t> long64s;
    vector<ULong64_t> ulong64s;
    vector<Char_t> chars;
    vector<UChar_t> uchars;
    unique_ptr<Bool_t[]> bools;

    void allocate(NumericType t, size_t n) {
        type = t;
        size = max<size_t>(1, n);

        floats.clear();
        doubles.clear();
        shorts.clear();
        ushorts.clear();
        ints.clear();
        uints.clear();
        longs.clear();
        ulongs.clear();
        long64s.clear();
        ulong64s.clear();
        chars.clear();
        uchars.clear();
        bools.reset();

        if (type == NumericType::Float) {
            floats.assign(size, 0.f);
        } else if (type == NumericType::Double) {
            doubles.assign(size, 0.);
        } else if (type == NumericType::Short) {
            shorts.assign(size, 0);
        } else if (type == NumericType::UShort) {
            ushorts.assign(size, 0);
        } else if (type == NumericType::Int) {
            ints.assign(size, 0);
        } else if (type == NumericType::UInt) {
            uints.assign(size, 0);
        } else if (type == NumericType::Long) {
            longs.assign(size, 0);
        } else if (type == NumericType::ULong) {
            ulongs.assign(size, 0);
        } else if (type == NumericType::Long64) {
            long64s.assign(size, 0);
        } else if (type == NumericType::ULong64) {
            ulong64s.assign(size, 0);
        } else if (type == NumericType::Char) {
            chars.assign(size, 0);
        } else if (type == NumericType::UChar) {
            uchars.assign(size, 0);
        } else {
            bools.reset(new Bool_t[size]());
        }
    }

    void* address() {
        if (type == NumericType::Float) {
            return floats.data();
        }
        if (type == NumericType::Double) {
            return doubles.data();
        }
        if (type == NumericType::Short) {
            return shorts.data();
        }
        if (type == NumericType::UShort) {
            return ushorts.data();
        }
        if (type == NumericType::Int) {
            return ints.data();
        }
        if (type == NumericType::UInt) {
            return uints.data();
        }
        if (type == NumericType::Long) {
            return longs.data();
        }
        if (type == NumericType::ULong) {
            return ulongs.data();
        }
        if (type == NumericType::Long64) {
            return long64s.data();
        }
        if (type == NumericType::ULong64) {
            return ulong64s.data();
        }
        if (type == NumericType::Char) {
            return chars.data();
        }
        if (type == NumericType::UChar) {
            return uchars.data();
        }
        return bools.get();
    }

    double valueAt(size_t index) const {
        if (index >= size) {
            throw runtime_error("Numeric buffer index is out of range");
        }

        if (type == NumericType::Float) {
            return floats[index];
        }
        if (type == NumericType::Double) {
            return doubles[index];
        }
        if (type == NumericType::Short) {
            return shorts[index];
        }
        if (type == NumericType::UShort) {
            return ushorts[index];
        }
        if (type == NumericType::Int) {
            return ints[index];
        }
        if (type == NumericType::UInt) {
            return uints[index];
        }
        if (type == NumericType::Long) {
            return static_cast<double>(longs[index]);
        }
        if (type == NumericType::ULong) {
            return static_cast<double>(ulongs[index]);
        }
        if (type == NumericType::Long64) {
            return static_cast<double>(long64s[index]);
        }
        if (type == NumericType::ULong64) {
            return static_cast<double>(ulong64s[index]);
        }
        if (type == NumericType::Char) {
            return chars[index];
        }
        if (type == NumericType::UChar) {
            return uchars[index];
        }
        return bools[index] ? 1. : 0.;
    }
};

struct BoundBranch {
    string name;
    string rootTypeName;
    NumericType type = NumericType::Float;
    bool isArray = false;
    string countBranch;
    size_t arraySize = 1;
    NumericBuffer buffer;

    void allocate() {
        buffer.allocate(type, isArray ? arraySize : 1);
    }

    void bind(TTree& tree) {
        if (tree.SetBranchAddress(name.c_str(), buffer.address()) < 0) {
            throw runtime_error("Failed to bind branch: " + name);
        }
    }

    double scalarValue() const {
        return buffer.valueAt(0);
    }

    Long64_t scalarAsLength() const {
        const double value = scalarValue();
        if (!isfinite(value) || value <= 0.) {
            return 0;
        }
        return static_cast<Long64_t>(value);
    }

    double firstArrayValue() const {
        return buffer.valueAt(0);
    }
};

struct PlotRuntime {
    BranchPlotConfig config;
    BoundBranch* valueBranch = nullptr;
    BoundBranch* countBranch = nullptr;
    unique_ptr<TH1D> hist;
};

bool endsWith(const string& text, const string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool startsWith(const string& text, const string& prefix) {
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

string stripQueryString(const string& path) {
    const size_t pos = path.find('?');
    return pos == string::npos ? path : path.substr(0, pos);
}

string baseName(const string& path) {
    const string cleanPath = stripQueryString(path);
    const size_t slash = cleanPath.find_last_of('/');
    if (slash == string::npos) {
        return cleanPath;
    }
    return cleanPath.substr(slash + 1);
}

string fileStem(const string& path) {
    string name = baseName(path);
    if (endsWith(name, ".root")) {
        name.resize(name.size() - 5);
    }
    return name.empty() ? "input" : name;
}

bool isRemoteRootFile(const string& path) {
    return startsWith(path, "root://") && endsWith(stripQueryString(path), ".root");
}

bool isCmsStoreRootFile(const string& path) {
    return startsWith(path, "/store/") && endsWith(stripQueryString(path), ".root");
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
    vector<string> lines;
    string line;
    stringstream ss(text);
    while (getline(ss, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    return lines;
}

bool remoteDatasetFilePassesFilter(const string& rootFilePath) {
    const string name = baseName(rootFilePath);
    if (!endsWith(name, ".root")) {
        return false;
    }

    const size_t numberEnd = name.size() - 5;
    size_t numberBegin = numberEnd;
    while (numberBegin > 0 && isdigit(static_cast<unsigned char>(name[numberBegin - 1]))) {
        --numberBegin;
    }
    if (numberBegin == numberEnd) {
        return false;
    }

    long long fileNumber = 0;
    try {
        fileNumber = stoll(name.substr(numberBegin, numberEnd - numberBegin));
    } catch (const exception&) {
        return false;
    }

    return fileNumber >= REMOTE_DATASET_FILE_NUMBER_MIN &&
           fileNumber <= REMOTE_DATASET_FILE_NUMBER_MAX;
}

vector<string> listRemoteDatasetRootFiles(const string& datasetPath) {
    string query = "file dataset=" + datasetPath;
    if (isUserDataset(datasetPath)) {
        query += " instance=prod/phys03";
    }

    const string command = "dasgoclient -query=\"" + query + "\" 2>&1";
    const vector<string> lines = splitLines(runCommand(command));

    vector<string> files;
    files.reserve(lines.size());
    for (const string& line : lines) {
        if (!endsWith(line, ".root")) {
            continue;
        }
        if (!remoteDatasetFilePassesFilter(line)) {
            continue;
        }
        files.push_back(XROOTD_REDIRECTOR + line);
    }

    sort(files.begin(), files.end());
    files.erase(unique(files.begin(), files.end()), files.end());
    return files;
}

vector<string> listLocalRootFiles(const string& inputPath) {
    const fs::path path(inputPath);
    if (!fs::exists(path)) {
        throw runtime_error("Local input path does not exist: " + inputPath);
    }

    vector<string> files;
    if (fs::is_regular_file(path)) {
        if (!endsWith(path.string(), ".root")) {
            throw runtime_error("Local input file is not a ROOT file: " + inputPath);
        }
        files.push_back(fs::absolute(path).lexically_normal().string());
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
            files.push_back(fs::absolute(entry.path()).lexically_normal().string());
        }
    }

    sort(files.begin(), files.end());
    files.erase(unique(files.begin(), files.end()), files.end());
    return files;
}

vector<string> resolveInputRootFiles() {
    vector<string> files;

    for (const string& inputPath : INPUT_FILES) {
        vector<string> sourceFiles;
        if (isRemoteRootFile(inputPath)) {
            sourceFiles.push_back(inputPath);
        } else if (isCmsStoreRootFile(inputPath)) {
            sourceFiles.push_back(XROOTD_REDIRECTOR + inputPath);
        } else if (fs::exists(fs::path(inputPath))) {
            sourceFiles = listLocalRootFiles(inputPath);
        } else if (isCmsDatasetPath(inputPath)) {
            sourceFiles = listRemoteDatasetRootFiles(inputPath);
        } else {
            sourceFiles = listLocalRootFiles(inputPath);
        }

        files.insert(files.end(), sourceFiles.begin(), sourceFiles.end());
    }

    sort(files.begin(), files.end());
    files.erase(unique(files.begin(), files.end()), files.end());
    if (files.empty()) {
        throw runtime_error("No ROOT files found from INPUT_FILES");
    }
    return files;
}

string stripSpaces(string text) {
    text.erase(remove_if(text.begin(), text.end(),
                         [](unsigned char c) { return isspace(c); }),
               text.end());
    return text;
}

string lower(string text) {
    transform(text.begin(), text.end(), text.begin(),
              [](unsigned char c) { return static_cast<char>(tolower(c)); });
    return text;
}

NumericType parseNumericType(const string& rootTypeName, const string& branchName) {
    const string t = lower(stripSpaces(rootTypeName));

    if (t == "float_t" || t == "float" || t == "float16_t") {
        return NumericType::Float;
    }
    if (t == "double_t" || t == "double" || t == "double32_t") {
        return NumericType::Double;
    }
    if (t == "short_t" || t == "short") {
        return NumericType::Short;
    }
    if (t == "ushort_t" || t == "unsignedshort") {
        return NumericType::UShort;
    }
    if (t == "int_t" || t == "int") {
        return NumericType::Int;
    }
    if (t == "uint_t" || t == "unsignedint") {
        return NumericType::UInt;
    }
    if (t == "long_t" || t == "long") {
        return NumericType::Long;
    }
    if (t == "ulong_t" || t == "unsignedlong") {
        return NumericType::ULong;
    }
    if (t == "long64_t" || t == "longlong_t" || t == "longlong") {
        return NumericType::Long64;
    }
    if (t == "ulong64_t" || t == "ulonglong_t" || t == "unsignedlonglong") {
        return NumericType::ULong64;
    }
    if (t == "char_t" || t == "char" || t == "byte_t") {
        return NumericType::Char;
    }
    if (t == "uchar_t" || t == "unsignedchar" || t == "ubyte_t") {
        return NumericType::UChar;
    }
    if (t == "bool_t" || t == "bool") {
        return NumericType::Bool;
    }

    throw runtime_error("Unsupported numeric type '" + rootTypeName +
                        "' for branch '" + branchName + "'");
}

TLeaf* primaryLeaf(TBranch* branch, const string& branchName) {
    TLeaf* leaf = branch->GetLeaf(branchName.c_str());
    if (leaf != nullptr) {
        return leaf;
    }

    const auto* leaves = branch->GetListOfLeaves();
    if (leaves == nullptr || leaves->GetEntries() != 1) {
        throw runtime_error("Branch '" + branchName +
                            "' does not have exactly one numeric leaf");
    }
    return static_cast<TLeaf*>(leaves->At(0));
}

string makeUniquePath(const string& requestedPath) {
    if (gSystem->AccessPathName(requestedPath.c_str())) {
        return requestedPath;
    }

    const size_t slash = requestedPath.find_last_of('/');
    const size_t dot = requestedPath.find_last_of('.');
    const bool hasExtension = (dot != string::npos && (slash == string::npos || dot > slash));
    const string stem = hasExtension ? requestedPath.substr(0, dot) : requestedPath;
    const string ext = hasExtension ? requestedPath.substr(dot) : "";

    for (int index = 1; index < 10000; ++index) {
        const string candidate = stem + "_" + to_string(index) + ext;
        if (gSystem->AccessPathName(candidate.c_str())) {
            return candidate;
        }
    }

    throw runtime_error("Could not find a free output path for " + requestedPath);
}

string safeName(string text) {
    for (char& c : text) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (!isalnum(uc) && c != '_') {
            c = '_';
        }
    }
    return text;
}

string sourceTag(const string& inputFile) {
    return safeName(fileStem(inputFile));
}

double effectiveXMin(const BranchPlotConfig& config) {
    return config.logx ? max(config.xmin, LOG_AXIS_MIN) : config.xmin;
}

void validatePlotConfig(const BranchPlotConfig& config) {
    if (config.name.empty()) {
        throw runtime_error("Configured branch name is empty");
    }
    if (config.bins <= 0) {
        throw runtime_error("Branch '" + config.name + "' has non-positive bin count");
    }
    const double xmin = effectiveXMin(config);
    if (!(config.xmax > xmin)) {
        throw runtime_error("Branch '" + config.name + "' has invalid histogram range");
    }
}

BoundBranch inspectBranch(TTree& tree, const string& branchName) {
    TBranch* branch = tree.GetBranch(branchName.c_str());
    if (branch == nullptr) {
        throw runtime_error("Missing branch: " + branchName);
    }

    const char* className = branch->GetClassName();
    if (className != nullptr && className[0] != '\0') {
        throw runtime_error("Branch '" + branchName +
                            "' is an object/vector branch, not a C array or scalar branch");
    }

    TLeaf* leaf = primaryLeaf(branch, branchName);
    const string typeName = leaf->GetTypeName();
    TLeaf* countLeaf = leaf->GetLeafCount();

    BoundBranch out;
    out.name = branchName;
    out.rootTypeName = typeName;
    out.type = parseNumericType(typeName, branchName);
    out.isArray = (countLeaf != nullptr || leaf->GetLenStatic() > 1);

    if (countLeaf != nullptr) {
        out.countBranch = countLeaf->GetName();
        Long64_t observedMax = static_cast<Long64_t>(ceil(tree.GetMaximum(out.countBranch.c_str())));
        if (observedMax < 0) {
            observedMax = 0;
        }
        out.arraySize = max<Long64_t>(1, observedMax);
    } else if (out.isArray) {
        out.arraySize = max(1, leaf->GetLenStatic());
    }

    return out;
}

BoundBranch& ensureBoundBranch(TTree& tree,
                               map<string, BoundBranch>& boundBranches,
                               const string& branchName) {
    auto it = boundBranches.find(branchName);
    if (it != boundBranches.end()) {
        return it->second;
    }

    BoundBranch branch = inspectBranch(tree, branchName);
    auto inserted = boundBranches.emplace(branchName, std::move(branch));
    return inserted.first->second;
}

vector<PlotRuntime> buildPlots(TTree& tree, map<string, BoundBranch>& boundBranches) {
    vector<PlotRuntime> plots;
    plots.reserve(BRANCHES.size());

    for (const BranchPlotConfig& config : BRANCHES) {
        validatePlotConfig(config);

        BoundBranch& valueBranch = ensureBoundBranch(tree, boundBranches, config.name);
        BoundBranch* countBranch = nullptr;
        if (valueBranch.isArray && !valueBranch.countBranch.empty()) {
            countBranch = &ensureBoundBranch(tree, boundBranches, valueBranch.countBranch);
            if (countBranch->isArray) {
                throw runtime_error("Count branch '" + valueBranch.countBranch +
                                    "' for array branch '" + valueBranch.name +
                                    "' is not scalar");
            }
        }

        const string histName = "h_" + safeName(config.name);
        const double xmin = effectiveXMin(config);
        unique_ptr<TH1D> hist(new TH1D(histName.c_str(), "", config.bins, xmin, config.xmax));
        hist->SetDirectory(nullptr);
        hist->SetLineWidth(2);
        hist->SetTitle((config.name + ";" + config.name + ";Events").c_str());

        plots.push_back(PlotRuntime{config, &valueBranch, countBranch, std::move(hist)});
    }

    return plots;
}

void configureBranches(TTree& tree, map<string, BoundBranch>& boundBranches) {
    tree.SetBranchStatus("*", 0);
    for (const auto& item : boundBranches) {
        tree.SetBranchStatus(item.first.c_str(), 1);
    }
    tree.SetCacheSize(50 * 1024 * 1024);
    for (const auto& item : boundBranches) {
        tree.AddBranchToCache(item.first.c_str(), true);
    }

    for (auto& item : boundBranches) {
        item.second.allocate();
        item.second.bind(tree);
    }
}

void fillPlots(TTree& tree, vector<PlotRuntime>& plots, const string& inputFile) {
    const Long64_t nEntries = tree.GetEntries();
    cout << "Total entries in " << inputFile << " = " << nEntries << endl;

    for (Long64_t entry = 0; entry < nEntries; ++entry) {
        tree.GetEntry(entry);

        if (entry % 1000000 == 0) {
            const double percent = (nEntries > 0) ? 100. * static_cast<double>(entry + 1) / nEntries : 100.;
            cout << "\rProcessing entry " << (entry + 1) << " / " << nEntries
                 << " (" << percent << "%)" << flush;
        }

        for (PlotRuntime& plot : plots) {
            double value = 0.;
            if (plot.valueBranch->isArray) {
                const Long64_t count = plot.countBranch != nullptr
                    ? plot.countBranch->scalarAsLength()
                    : static_cast<Long64_t>(plot.valueBranch->arraySize);
                if (count <= 0) {
                    continue;
                }
                value = plot.valueBranch->firstArrayValue();
            } else {
                value = plot.valueBranch->scalarValue();
            }

            if (isfinite(value)) {
                plot.hist->Fill(value);
            }
        }
    }

    if (nEntries > 0) {
        cout << endl;
    }
}

void savePlots(const vector<PlotRuntime>& plots, const string& inputFile) {
    gSystem->mkdir(OUTPUT_DIR.c_str(), kTRUE);

    const string tag = sourceTag(inputFile);
    const string rootPath = makeUniquePath(OUTPUT_DIR + "/" + tag + "_" + OUTPUT_ROOT_FILE);
    TFile output(rootPath.c_str(), "CREATE");
    if (output.IsZombie()) {
        throw runtime_error("Failed to create output ROOT file: " + rootPath);
    }

    for (const PlotRuntime& plot : plots) {
        plot.hist->Write();
    }
    output.Close();

    for (const PlotRuntime& plot : plots) {
        const string branchTag = safeName(plot.config.name);
        TCanvas canvas(("c_" + tag + "_" + branchTag).c_str(), "", 800, 700);
        canvas.SetMargin(0.12, 0.04, 0.12, 0.08);
        canvas.SetLogy(true);
        if (plot.config.logx) {
            canvas.SetLogx(true);
        }

        plot.hist->SetMinimum(LOG_AXIS_MIN);
        if (plot.hist->GetMaximum() > 0.) {
            plot.hist->SetMaximum(plot.hist->GetMaximum() * 10.);
        } else {
            plot.hist->SetMaximum(1.);
        }
        plot.hist->Draw("hist");

        const string pdfPath = makeUniquePath(OUTPUT_DIR + "/" + tag + "_" + branchTag + ".pdf");
        canvas.SaveAs(pdfPath.c_str());
        cout << "Wrote " << pdfPath << endl;
    }

    cout << "Wrote " << rootPath << endl;
}

void printBranchSummary(const vector<PlotRuntime>& plots) {
    for (const PlotRuntime& plot : plots) {
        const BoundBranch& branch = *plot.valueBranch;
        cout << "Branch " << branch.name << ": type = " << branch.rootTypeName;
        if (branch.isArray) {
            cout << ", C array";
            if (!branch.countBranch.empty()) {
                cout << ", count branch = " << branch.countBranch;
            }
            cout << ", buffer size = " << branch.arraySize
                 << ", filling index 0";
        } else {
            cout << ", scalar";
        }
        cout << endl;
    }
}

void processRootFile(const string& inputFile) {
    cout << "[FILE] " << inputFile << endl;

    TChain chain(TREE_NAME.c_str());
    if (chain.Add(inputFile.c_str()) <= 0) {
        throw runtime_error("Failed to add ROOT file to TChain: " + inputFile);
    }

    if (chain.GetEntries() > 0) {
        chain.LoadTree(0);
    }

    map<string, BoundBranch> boundBranches;
    vector<PlotRuntime> plots = buildPlots(chain, boundBranches);
    configureBranches(chain, boundBranches);
    printBranchSummary(plots);
    fillPlots(chain, plots, inputFile);
    savePlots(plots, inputFile);

    chain.ResetBranchAddresses();
}

}  // namespace

int plot_branch_histograms() {
    try {
        if (INPUT_FILES.empty()) {
            throw runtime_error("INPUT_FILES is empty");
        }
        if (BRANCHES.empty()) {
            throw runtime_error("BRANCHES is empty");
        }
        if (REMOTE_DATASET_FILE_NUMBER_MIN > REMOTE_DATASET_FILE_NUMBER_MAX) {
            throw runtime_error("REMOTE_DATASET_FILE_NUMBER_MIN is larger than REMOTE_DATASET_FILE_NUMBER_MAX");
        }

        gROOT->SetBatch(kTRUE);
        gStyle->SetOptStat(0);

        const vector<string> rootFiles = resolveInputRootFiles();
        cout << "Resolved ROOT files = " << rootFiles.size() << endl;
        cout << "Remote dataset file-number filter = "
             << REMOTE_DATASET_FILE_NUMBER_MIN << "-"
             << REMOTE_DATASET_FILE_NUMBER_MAX << endl;

        for (const string& inputFile : rootFiles) {
            processRootFile(inputFile);
        }
    } catch (const exception& ex) {
        cerr << "plot_branch_histograms error: " << ex.what() << endl;
        return 1;
    }

    return 0;
}

#ifndef __CLING__
int main() {
    return plot_branch_histograms();
}
#endif
