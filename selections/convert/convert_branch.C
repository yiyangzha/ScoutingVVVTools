// Summary: Skim events and convert branches for BDT training with JSON-driven sample config, string formulas, and precompiled selections.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

#include <TBranch.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TList.h>
#include <TLorentzVector.h>
#include <TROOT.h>
#include <TTree.h>

#include "../../src/simple_json.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
namespace fs = std::filesystem;
using JsonValue = simple_json::Value;

namespace {

// Thrown when a single input file is malformed or has an incompatible schema.
// The file is skipped and processing continues with the next file.
struct SkippableFileError : std::runtime_error {
    using runtime_error::runtime_error;
};

const float def = -99.f;
const double kMissingDistance = -999.;
const double kLargeDistance = 999.;
const char* kRemotePrefix = "root://cms-xrd-global.cern.ch/";

const char* kAppConfigPath = "./config.json";
const char* kBranchConfigPath = "./branch.json";
const char* kSelectionConfigPath = "./selection.json";
const char* kAppConfigEnvVar = "CONVERT_CONFIG_PATH";
const char* kSuccessfulBatchesEnvVar = "CONVERT_SUCCESSFUL_BATCHES";
const char* kDeferFinalMergeEnvVar = "CONVERT_DEFER_FINAL_MERGE";
const char* kGoldenJsonEnvVar = "CONVERT_GOLDEN_JSON";
const char* kDefaultSampleConfigPath = "../../src/sample.json";
const int kRemoteInputOpenRetries = 5;
const unsigned int kRemoteInputRetrySleepSeconds = 5;

enum class DataType {
    Float,
    Short,
    Int,
    UInt,
    UChar,
    Bool,
    Long64,
    ULong64,
};

struct Expression;
using ExprPtr = shared_ptr<Expression>;

enum class ExprKind {
    Number,
    Identifier,
    Unary,
    Binary,
    Call,
    Index,
    Member,
};

struct Expression {
    ExprKind kind = ExprKind::Number;
    long double number = 0.;
    string text;
    ExprPtr lhs;
    ExprPtr rhs;
    vector<ExprPtr> args;
};

struct SortRule {
    string text;
    ExprPtr expr;
    bool descending = true;
};

struct RuntimeCollectionConfig {
    string name;
    string source;
    vector<string> merge;
    string selectionText = "1";
    ExprPtr selectionExpr;
    string dedupCollection;
    string dedupText;
    ExprPtr dedupExpr;
    string sortText;
    SortRule sortRule;
};

struct SelectionConfig {
    string eventPreselectionText = "1";
    ExprPtr eventPreselection;
    vector<string> collectionOrder;
    unordered_map<string, RuntimeCollectionConfig> collections;
    unordered_map<string, string> treeSelectionText;
    unordered_map<string, ExprPtr> treeSelections;
};

struct ScalarInputConfig {
    string name;
    string branch;
    DataType type = DataType::Int;
    bool onlyMC = false;
    bool bound = false;
    Short_t shortValue = 0;
    Int_t intValue = 0;
    UInt_t uintValue = 0;
    Float_t floatValue = 0.f;
    UChar_t ucharValue = 0;
    Bool_t boolValue = false;
    Long64_t long64Value = 0;
    ULong64_t ulong64Value = 0;

    void bind(TTree* tree, bool isMC) {
        if (onlyMC && !isMC) {
            bound = false;
            return;
        }
        if (!tree->GetBranch(branch.c_str())) {
            throw SkippableFileError("Missing scalar branch: " + branch);
        }

        if (type == DataType::Float) {
            tree->SetBranchAddress(branch.c_str(), &floatValue);
        } else if (type == DataType::Short) {
            tree->SetBranchAddress(branch.c_str(), &shortValue);
        } else if (type == DataType::Int) {
            tree->SetBranchAddress(branch.c_str(), &intValue);
        } else if (type == DataType::UInt) {
            tree->SetBranchAddress(branch.c_str(), &uintValue);
        } else if (type == DataType::UChar) {
            tree->SetBranchAddress(branch.c_str(), &ucharValue);
        } else if (type == DataType::Bool) {
            tree->SetBranchAddress(branch.c_str(), &boolValue);
        } else if (type == DataType::Long64) {
            tree->SetBranchAddress(branch.c_str(), &long64Value);
        } else {
            tree->SetBranchAddress(branch.c_str(), &ulong64Value);
        }
        bound = true;
    }

    long double numericValue() const {
        if (type == DataType::Float) {
            return floatValue;
        }
        if (type == DataType::Short) {
            return shortValue;
        }
        if (type == DataType::Int) {
            return intValue;
        }
        if (type == DataType::UInt) {
            return uintValue;
        }
        if (type == DataType::UChar) {
            return ucharValue;
        }
        if (type == DataType::Bool) {
            return boolValue ? 1. : 0.;
        }
        if (type == DataType::Long64) {
            return static_cast<long double>(long64Value);
        }
        return static_cast<long double>(ulong64Value);
    }
};

struct ArrayInputConfig {
    string name;
    string branch;
    DataType type = DataType::Float;
    bool onlyMC = false;
    int maxSize = 0;
    bool bound = false;
    vector<Float_t> floatValues;
    vector<Short_t> shortValues;
    vector<Int_t> intValues;
    vector<UInt_t> uintValues;
    vector<UChar_t> ucharValues;
    vector<UChar_t> boolValues;
    vector<Long64_t> long64Values;
    vector<ULong64_t> ulong64Values;

    void initBuffer() {
        if (type == DataType::Float) {
            floatValues.assign(maxSize, 0.f);
        } else if (type == DataType::Short) {
            shortValues.assign(maxSize, 0);
        } else if (type == DataType::Int) {
            intValues.assign(maxSize, 0);
        } else if (type == DataType::UInt) {
            uintValues.assign(maxSize, 0);
        } else if (type == DataType::UChar) {
            ucharValues.assign(maxSize, 0);
        } else if (type == DataType::Bool) {
            boolValues.assign(maxSize, 0);
        } else if (type == DataType::Long64) {
            long64Values.assign(maxSize, 0);
        } else {
            ulong64Values.assign(maxSize, 0);
        }
    }

    void ensureBufferSize(int size) {
        const int newSize = max(1, size);
        if (newSize == maxSize) {
            return;
        }
        maxSize = newSize;
        initBuffer();
    }

    void bind(TTree* tree, bool isMC) {
        if (onlyMC && !isMC) {
            bound = false;
            return;
        }
        if (!tree->GetBranch(branch.c_str())) {
            throw SkippableFileError("Missing array branch: " + branch);
        }

        if (type == DataType::Float) {
            tree->SetBranchAddress(branch.c_str(), floatValues.data());
        } else if (type == DataType::Short) {
            tree->SetBranchAddress(branch.c_str(), shortValues.data());
        } else if (type == DataType::Int) {
            tree->SetBranchAddress(branch.c_str(), intValues.data());
        } else if (type == DataType::UInt) {
            tree->SetBranchAddress(branch.c_str(), uintValues.data());
        } else if (type == DataType::UChar) {
            tree->SetBranchAddress(branch.c_str(), ucharValues.data());
        } else if (type == DataType::Bool) {
            tree->SetBranchAddress(branch.c_str(), boolValues.data());
        } else if (type == DataType::Long64) {
            tree->SetBranchAddress(branch.c_str(), long64Values.data());
        } else {
            tree->SetBranchAddress(branch.c_str(), ulong64Values.data());
        }
        bound = true;
    }

    float valueAt(int index) const {
        if (type == DataType::Float) {
            return floatValues[index];
        }
        if (type == DataType::Short) {
            return shortValues[index];
        }
        if (type == DataType::Int) {
            return intValues[index];
        }
        if (type == DataType::UInt) {
            return uintValues[index];
        }
        if (type == DataType::UChar) {
            return ucharValues[index];
        }
        if (type == DataType::Bool) {
            return boolValues[index] ? 1.f : 0.f;
        }
        if (type == DataType::Long64) {
            return static_cast<float>(long64Values[index]);
        }
        return static_cast<float>(ulong64Values[index]);
    }
};

struct InputCollectionConfig {
    string name;
    string sizeName;
    int maxSize = 0;
    string ptField;
    string etaField;
    string phiField;
    string massField;
    float defaultMass = 0.f;
    int ptIndex = -1;
    int etaIndex = -1;
    int phiIndex = -1;
    int massIndex = -1;
    vector<ArrayInputConfig> fields;
};

struct OutputScalarConfig {
    string name;
    DataType type = DataType::Float;
    bool onlyMC = false;
    string formulaText;
    ExprPtr formula;
    string collection;
    int slots = 0;
};

struct TreeConfig {
    string name;
    string title;
    string selection;
    vector<OutputScalarConfig> regularScalars;
    vector<OutputScalarConfig> extremaScalars;
};

struct BranchConfig {
    vector<ScalarInputConfig> scalars;
    vector<InputCollectionConfig> collections;
    vector<TreeConfig> trees;
};

struct ObjectSchema {
    vector<string> fields;
    unordered_map<string, size_t> indexByName;
};

struct RuntimeObject {
    vector<float> values;
    TLorentzVector p4;
};

struct RuntimeCollection {
    string name;
    ObjectSchema schema;
    vector<RuntimeObject> objects;
};

struct OutputBranchRuntime {
    string name;
    DataType type = DataType::Float;
    const OutputScalarConfig* sourceConfig = nullptr;
    int slotIndex = -1;
    Float_t floatValue = def;
    Int_t intValue = 0;
    UInt_t uintValue = 0;
    Bool_t boolValue = false;
    Long64_t long64Value = 0;
    ULong64_t ulong64Value = 0;
};

// Input buffers for reading LHE/PS theory weight branches from NANOAOD.
struct TheoryWeightBufs {
    static constexpr int kMaxPdf   = 200;
    static constexpr int kMaxScale =  20;
    static constexpr int kMaxPS    =  10;
    float genWeight              = 1.f;
    int   nLHEPdfWeight          = 0;
    float LHEPdfWeight[kMaxPdf]  = {};
    int   nLHEScaleWeight        = 0;
    float LHEScaleWeight[kMaxScale] = {};
    int   nPSWeight              = 0;
    float PSWeight[kMaxPS]       = {};
};

// Fixed-size output arrays written as branches to the converted ROOT trees.
struct TheoryOutBufs {
    static constexpr int kNPdf     = 101;
    static constexpr int kNAlphaS  =   2;
    static constexpr int kNScale   =   9;
    static constexpr int kNPS      =   4;
    float genWeight                      = 1.f;
    float LHEPdfWeight[kNPdf]            = {};
    float LHEPdfWeightAlphaS[kNAlphaS]   = {1.f, 1.f};
    float LHEScaleWeight[kNScale]        = {};
    float PSWeight[kNPS]                 = {};
};

struct OutputTreeState {
    TreeConfig config;
    TTree* tree = nullptr;
    vector<OutputBranchRuntime> branches;
    unordered_map<string, size_t> branchIndexByName;
    TheoryOutBufs theoryOutBuf;
    bool hasTheoryBranches = false;
};

struct ThreadConvertResult {
    vector<OutputTreeState> outputTrees;
    TFile* tempFile = nullptr;
    string tempFilePath;
};

struct SampleRuleConfig {
    string name;
    vector<string> paths;
    int sampleId = -1;
    bool isMC = true;
    bool isSignal = false;
    bool hasTheoryWeights = false;
    double xsection = -1.;
    double lumi = -1.;
};

struct AppConfig {
    string treeName = "Events";
    string configPath;
    string configDir;
    string outputRoot;
    string outputPattern;
    string runSample;
    string lumiMaskPath;
    string sampleConfigPath = kDefaultSampleConfigPath;
    int maxThreads = 12;
    double maxOutputFileSizeGB = 5.;
    bool resumeSuccessfulBatches = true;
    vector<SampleRuleConfig> sampleRules;
    string puWeightPathPattern;
};

struct BatchRequest {
    bool printBatchCount = false;
    bool mergeSuccessfulBatches = false;
    bool singleBatch = false;
    size_t batchIndex = 0;
};

struct BatchTempCollection {
    vector<string> paths;
    Long64_t rawEntries = 0;
    size_t skipped = 0;
};

struct PileupBin {
    float binLow = 0.f;
    float binHigh = 0.f;
    float weight = 1.f;
    float weightLow = 1.f;
    float weightHigh = 1.f;
};

struct SampleMeta {
    string sample;
    vector<string> inputPaths;
    string outputFileName;
    int sampleId = -1;
    bool isMC = true;
    bool isSignal = false;
    bool hasTheoryWeights = false;
    double xsection = -1.;
    double lumi = -1.;
    size_t remoteSourceCount = 0;

    string sampleGroup() const {
        if (!isMC) {
            return "data";
        }
        return isSignal ? "signal" : "bkg";
    }
};

struct LumiRange {
    UInt_t first = 0;
    UInt_t last = 0;
};

struct LumiMaskRun {
    UInt_t run = 0;
    vector<LumiRange> ranges;
};

struct LumiMask {
    vector<LumiMaskRun> runs;

    bool contains(UInt_t run, UInt_t lumi) const {
        const auto runIt = lower_bound(runs.begin(), runs.end(), run,
                                       [](const LumiMaskRun& item, UInt_t value) {
                                           return item.run < value;
                                       });
        if (runIt == runs.end() || runIt->run != run) {
            return false;
        }

        const auto& ranges = runIt->ranges;
        const auto rangeIt = upper_bound(ranges.begin(), ranges.end(), lumi,
                                         [](UInt_t value, const LumiRange& range) {
                                             return value < range.first;
                                         });
        if (rangeIt == ranges.begin()) {
            return false;
        }
        const LumiRange& candidate = *(rangeIt - 1);
        return lumi <= candidate.last;
    }
};

struct EvalContext {
    const unordered_map<string, long double>* vars = nullptr;
    const unordered_map<string, RuntimeCollection>* collections = nullptr;
    const unordered_map<string, RuntimeCollection>* inputCollections = nullptr;
    const unordered_map<string, const ScalarInputConfig*>* rawScalars = nullptr;
    const RuntimeCollection* currentCollection = nullptr;
    const RuntimeObject* currentObject = nullptr;
    const RuntimeCollection* otherCollection = nullptr;
    const RuntimeObject* otherObject = nullptr;
};

struct Value {
    enum class Kind {
        Number,
        ObjectRef,
        CollectionRef,
        P4,
    };

    Kind kind = Kind::Number;
    long double number = 0.;
    const RuntimeCollection* collection = nullptr;
    const RuntimeObject* object = nullptr;
    TLorentzVector p4;
};

vector<string> getStringListOrScalar(const JsonValue& node, const string& key);
string formatInputSources(const vector<string>& sources);

class ExpressionParser {
public:
    explicit ExpressionParser(string text) : text_(std::move(text)) {}

    ExprPtr parse() {
        ExprPtr expr = parseLogicalOr();
        skipWhitespace();
        if (pos_ != text_.size()) {
            throw runtime_error("Unexpected token in expression: " + text_.substr(pos_));
        }
        return expr;
    }

private:
    string text_;
    size_t pos_ = 0;

    void skipWhitespace() {
        while (pos_ < text_.size() && isspace(static_cast<unsigned char>(text_[pos_]))) {
            ++pos_;
        }
    }

    bool match(const string& token) {
        skipWhitespace();
        if (text_.compare(pos_, token.size(), token) == 0) {
            pos_ += token.size();
            return true;
        }
        return false;
    }

    void expect(const string& token) {
        if (!match(token)) {
            throw runtime_error("Expected token '" + token + "' in expression: " + text_);
        }
    }

    ExprPtr makeNumber(long double value) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Number;
        node->number = value;
        return node;
    }

    ExprPtr makeIdentifier(const string& name) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Identifier;
        node->text = name;
        return node;
    }

    ExprPtr makeUnary(const string& op, ExprPtr arg) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Unary;
        node->text = op;
        node->lhs = std::move(arg);
        return node;
    }

    ExprPtr makeBinary(const string& op, ExprPtr lhs, ExprPtr rhs) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Binary;
        node->text = op;
        node->lhs = std::move(lhs);
        node->rhs = std::move(rhs);
        return node;
    }

    ExprPtr makeCall(const string& name, vector<ExprPtr> args) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Call;
        node->text = name;
        node->args = std::move(args);
        return node;
    }

    ExprPtr makeIndex(ExprPtr base, ExprPtr index) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Index;
        node->lhs = std::move(base);
        node->rhs = std::move(index);
        return node;
    }

    ExprPtr makeMember(ExprPtr base, const string& member) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Member;
        node->lhs = std::move(base);
        node->text = member;
        return node;
    }

    string parseIdentifierText() {
        skipWhitespace();
        if (pos_ >= text_.size() || !(isalpha(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '_')) {
            throw runtime_error("Expected identifier in expression: " + text_);
        }

        const size_t begin = pos_;
        ++pos_;
        while (pos_ < text_.size()) {
            const unsigned char ch = static_cast<unsigned char>(text_[pos_]);
            if (isalnum(ch) || ch == '_') {
                ++pos_;
                continue;
            }
            break;
        }
        return text_.substr(begin, pos_ - begin);
    }

    ExprPtr parseNumberLiteral() {
        skipWhitespace();
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        const long double value = strtold(begin, &end);
        if (end == begin) {
            throw runtime_error("Expected numeric literal in expression: " + text_);
        }
        pos_ += static_cast<size_t>(end - begin);
        return makeNumber(value);
    }

    vector<ExprPtr> parseArgumentList() {
        vector<ExprPtr> args;
        skipWhitespace();
        if (match(")")) {
            return args;
        }

        while (true) {
            args.push_back(parseLogicalOr());
            skipWhitespace();
            if (match(")")) {
                break;
            }
            expect(",");
        }
        return args;
    }

    ExprPtr parsePrimary() {
        skipWhitespace();
        if (pos_ >= text_.size()) {
            throw runtime_error("Unexpected end of expression: " + text_);
        }

        if (match("(")) {
            ExprPtr expr = parseLogicalOr();
            expect(")");
            return expr;
        }

        const unsigned char ch = static_cast<unsigned char>(text_[pos_]);
        if (isdigit(ch) || text_[pos_] == '.') {
            return parseNumberLiteral();
        }

        const string identifier = parseIdentifierText();
        skipWhitespace();
        if (match("(")) {
            return makeCall(identifier, parseArgumentList());
        }
        return makeIdentifier(identifier);
    }

    ExprPtr parsePostfix() {
        ExprPtr expr = parsePrimary();
        while (true) {
            skipWhitespace();
            if (match("[")) {
                ExprPtr index = parseLogicalOr();
                expect("]");
                expr = makeIndex(expr, index);
                continue;
            }
            if (match(".")) {
                expr = makeMember(expr, parseIdentifierText());
                continue;
            }
            break;
        }
        return expr;
    }

    ExprPtr parseUnary() {
        skipWhitespace();
        if (match("+")) {
            return makeUnary("+", parseUnary());
        }
        if (match("-")) {
            return makeUnary("-", parseUnary());
        }
        if (match("!")) {
            return makeUnary("!", parseUnary());
        }
        return parsePostfix();
    }

    ExprPtr parseMultiplicative() {
        ExprPtr expr = parseUnary();
        while (true) {
            if (match("*")) {
                expr = makeBinary("*", expr, parseUnary());
            } else if (match("/")) {
                expr = makeBinary("/", expr, parseUnary());
            } else {
                break;
            }
        }
        return expr;
    }

    ExprPtr parseAdditive() {
        ExprPtr expr = parseMultiplicative();
        while (true) {
            if (match("+")) {
                expr = makeBinary("+", expr, parseMultiplicative());
            } else if (match("-")) {
                expr = makeBinary("-", expr, parseMultiplicative());
            } else {
                break;
            }
        }
        return expr;
    }

    ExprPtr parseRelational() {
        ExprPtr expr = parseAdditive();
        while (true) {
            if (match("<=")) {
                expr = makeBinary("<=", expr, parseAdditive());
            } else if (match(">=")) {
                expr = makeBinary(">=", expr, parseAdditive());
            } else if (match("<")) {
                expr = makeBinary("<", expr, parseAdditive());
            } else if (match(">")) {
                expr = makeBinary(">", expr, parseAdditive());
            } else {
                break;
            }
        }
        return expr;
    }

    ExprPtr parseEquality() {
        ExprPtr expr = parseRelational();
        while (true) {
            if (match("==")) {
                expr = makeBinary("==", expr, parseRelational());
            } else if (match("!=")) {
                expr = makeBinary("!=", expr, parseRelational());
            } else {
                break;
            }
        }
        return expr;
    }

    ExprPtr parseLogicalAnd() {
        ExprPtr expr = parseEquality();
        while (match("&&")) {
            expr = makeBinary("&&", expr, parseEquality());
        }
        return expr;
    }

    ExprPtr parseLogicalOr() {
        ExprPtr expr = parseLogicalAnd();
        while (match("||")) {
            expr = makeBinary("||", expr, parseLogicalAnd());
        }
        return expr;
    }
};

bool endsWith(const string& text, const string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool startsWith(const string& text, const string& prefix) {
    return text.size() >= prefix.size() &&
           text.compare(0, prefix.size(), prefix) == 0;
}

size_t skipWhitespace(const string& text, size_t pos) {
    while (pos < text.size() && isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
    return pos;
}

size_t findMatchingJsonDelimiter(const string& text, size_t openPos, char openChar, char closeChar) {
    if (openPos >= text.size() || text[openPos] != openChar) {
        throw runtime_error("Invalid JSON delimiter search");
    }

    size_t pos = openPos;
    int depth = 0;
    bool inString = false;
    bool escape = false;
    while (pos < text.size()) {
        const char ch = text[pos];
        if (inString) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                inString = false;
            }
            ++pos;
            continue;
        }

        if (ch == '"') {
            inString = true;
        } else if (ch == openChar) {
            ++depth;
        } else if (ch == closeChar) {
            --depth;
            if (depth == 0) {
                return pos;
            }
        }
        ++pos;
    }

    throw runtime_error("Unmatched JSON delimiter while editing sample config");
}

DataType parseDataType(const string& text) {
    if (text == "F") {
        return DataType::Float;
    }
    if (text == "S") {
        return DataType::Short;
    }
    if (text == "I") {
        return DataType::Int;
    }
    if (text == "UI") {
        return DataType::UInt;
    }
    if (text == "b") {
        return DataType::UChar;
    }
    if (text == "O") {
        return DataType::Bool;
    }
    if (text == "L64") {
        return DataType::Long64;
    }
    if (text == "UL64") {
        return DataType::ULong64;
    }
    throw runtime_error("Unsupported data type: " + text);
}

char outputLeafCode(DataType type) {
    if (type == DataType::Float) {
        return 'F';
    }
    if (type == DataType::Short) {
        return 'S';
    }
    if (type == DataType::Int) {
        return 'I';
    }
    if (type == DataType::UInt) {
        return 'i';
    }
    if (type == DataType::Bool) {
        return 'O';
    }
    if (type == DataType::Long64) {
        return 'L';
    }
    if (type == DataType::ULong64) {
        return 'l';
    }
    throw runtime_error("Unsupported output type for tree branch");
}

string resolveConfigPath(const char* preferredPath, const char* envVar = nullptr) {
    if (envVar != nullptr) {
        const char* envPath = getenv(envVar);
        if (envPath != nullptr && *envPath != '\0') {
            if (fs::exists(envPath)) {
                return envPath;
            }
            throw runtime_error(string("Cannot find config file from environment variable ") + envVar + ": " + envPath);
        }
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

JsonValue loadJson(const char* path, const char* envVar = nullptr) {
    const string resolved = resolveConfigPath(path, envVar);
    return simple_json::parseFile(resolved);
}

JsonValue loadJsonPath(const string& path) {
    return simple_json::parseFile(path);
}

string resolveReferencedPath(const string& baseConfigPath, const string& targetPath) {
    if (targetPath.empty()) {
        return targetPath;
    }

    fs::path basePath(baseConfigPath);
    if (!basePath.is_absolute()) {
        basePath = fs::absolute(basePath);
    }

    const fs::path path(targetPath);
    if (path.is_absolute()) {
        return path.lexically_normal().string();
    }

    return (basePath.parent_path() / path).lexically_normal().string();
}

string resolveConfiguredPathPattern(const string& baseConfigPath, const string& pathPattern) {
    if (pathPattern.empty() || pathPattern.find("{output_root}") != string::npos) {
        return pathPattern;
    }
    return resolveReferencedPath(baseConfigPath, pathPattern);
}

string normalizeOutputPath(const AppConfig& appConfig, const string& outputPath) {
    const fs::path path(outputPath);
    if (path.is_absolute()) {
        return path.lexically_normal().string();
    }
    return (fs::path(appConfig.configDir) / path).lexically_normal().string();
}

UInt_t parseUInt32Text(const string& text, const string& context) {
    size_t pos = 0;
    unsigned long long value = 0;
    try {
        value = stoull(text, &pos, 10);
    } catch (const exception&) {
        throw runtime_error("Invalid unsigned integer for " + context + ": " + text);
    }
    if (pos != text.size() || value > numeric_limits<UInt_t>::max()) {
        throw runtime_error("Invalid unsigned integer for " + context + ": " + text);
    }
    return static_cast<UInt_t>(value);
}

UInt_t parseUInt32Json(const JsonValue& value, const string& context) {
    const long double number = value.asNumber();
    if (number < 0 || number > numeric_limits<UInt_t>::max() || floor(number) != number) {
        throw runtime_error("Invalid unsigned integer for " + context);
    }
    return static_cast<UInt_t>(number);
}

LumiMask loadLumiMask(const string& path) {
    const JsonValue payload = loadJsonPath(path);
    const auto& object = payload.asObject();

    LumiMask mask;
    mask.runs.reserve(object.size());
    for (const auto& item : object) {
        LumiMaskRun runConfig;
        runConfig.run = parseUInt32Text(item.first, "lumi mask run");

        const auto& ranges = item.second.asArray();
        runConfig.ranges.reserve(ranges.size());
        for (const auto& rangeNode : ranges) {
            const auto& range = rangeNode.asArray();
            if (range.size() != 2) {
                throw runtime_error("Each lumi mask range must contain exactly two values for run " +
                                    item.first);
            }

            LumiRange lumiRange;
            lumiRange.first = parseUInt32Json(range[0], "lumi mask range start");
            lumiRange.last = parseUInt32Json(range[1], "lumi mask range end");
            if (lumiRange.last < lumiRange.first) {
                throw runtime_error("Lumi mask range end is smaller than start for run " + item.first);
            }
            runConfig.ranges.push_back(lumiRange);
        }
        sort(runConfig.ranges.begin(), runConfig.ranges.end(),
             [](const LumiRange& lhs, const LumiRange& rhs) {
                 return lhs.first < rhs.first;
             });
        mask.runs.push_back(std::move(runConfig));
    }

    sort(mask.runs.begin(), mask.runs.end(),
         [](const LumiMaskRun& lhs, const LumiMaskRun& rhs) {
             return lhs.run < rhs.run;
         });
    return mask;
}

ExprPtr compileExpression(const string& text) {
    return ExpressionParser(text).parse();
}

SortRule parseSortRule(const string& text) {
    SortRule rule;
    rule.text = text;
    string trimmed = text;
    auto trim = [](string value) {
        const auto begin = value.find_first_not_of(" \t\r\n");
        if (begin == string::npos) {
            return string();
        }
        const auto end = value.find_last_not_of(" \t\r\n");
        return value.substr(begin, end - begin + 1);
    };
    trimmed = trim(trimmed);

    const auto lastSpace = trimmed.find_last_of(" \t");
    if (lastSpace != string::npos) {
        const string maybeOrder = trim(trimmed.substr(lastSpace + 1));
        if (maybeOrder == "asc" || maybeOrder == "desc") {
            rule.descending = (maybeOrder != "asc");
            trimmed = trim(trimmed.substr(0, lastSpace));
        }
    }

    if (trimmed.empty()) {
        trimmed = "1";
    }
    rule.expr = compileExpression(trimmed);
    return rule;
}

AppConfig loadAppConfig() {
    const string appConfigPath = resolveConfigPath(kAppConfigPath, kAppConfigEnvVar);
    const JsonValue payload = simple_json::parseFile(appConfigPath);

    AppConfig config;
    config.configPath = fs::absolute(fs::path(appConfigPath)).lexically_normal().string();
    config.configDir = fs::path(config.configPath).parent_path().string();
    config.treeName = payload.getStringOr("tree_name", "Events");
    config.runSample = payload.getStringOr("run_sample", "");
    config.outputRoot = resolveReferencedPath(config.configPath, payload.at("output_root").asString());
    config.outputPattern = payload.at("output_pattern").asString();
    config.lumiMaskPath = resolveReferencedPath(config.configPath, payload.getStringOr("lumi_mask", ""));
    config.sampleConfigPath = resolveReferencedPath(
        config.configPath, payload.getStringOr("sample_config", kDefaultSampleConfigPath));
    config.maxThreads = payload.getIntOr("max_threads", 12);
    config.maxOutputFileSizeGB = static_cast<double>(payload.getNumberOr("max_output_file_size_gb", 5.));
    config.resumeSuccessfulBatches = payload.getBoolOr("resume_successful_batches", true);

    const JsonValue samplePayload = simple_json::parseFile(config.sampleConfigPath);
    if (samplePayload.contains("sample")) {
        for (const auto& node : samplePayload.at("sample").asArray()) {
            SampleRuleConfig rule;
            rule.name = node.at("name").asString();
            rule.paths = getStringListOrScalar(node, "path");
            rule.sampleId = node.at("sample_ID").asInt();
            rule.isMC = node.at("is_MC").asBool();
            rule.isSignal = node.at("is_signal").asBool();
            rule.hasTheoryWeights = node.contains("has_theory_weights") && node.at("has_theory_weights").asBool();
            rule.xsection = static_cast<double>(node.getNumberOr("xsection", -1.));
            rule.lumi = static_cast<double>(node.getNumberOr("lumi", -1.));
            config.sampleRules.push_back(std::move(rule));
        }
    }

    config.puWeightPathPattern = resolveConfiguredPathPattern(
        config.configPath, payload.getStringOr("pu_weight_path", ""));
    return config;
}

void writeSampleRawEntries(const string& sampleConfigPath,
                           const string& sampleName,
                           Long64_t rawEntries) {
    ifstream fin(sampleConfigPath);
    if (!fin) {
        throw runtime_error("Cannot open sample config for raw_entries update: " + sampleConfigPath);
    }

    const string content((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
    const size_t sampleKeyPos = content.find("\"sample\"");
    if (sampleKeyPos == string::npos) {
        throw runtime_error("Cannot find 'sample' array in sample config: " + sampleConfigPath);
    }

    const size_t colonPos = content.find(':', sampleKeyPos);
    if (colonPos == string::npos) {
        throw runtime_error("Malformed 'sample' entry in sample config: " + sampleConfigPath);
    }

    const size_t arrayPos = skipWhitespace(content, colonPos + 1);
    if (arrayPos >= content.size() || content[arrayPos] != '[') {
        throw runtime_error("Expected sample array in sample config: " + sampleConfigPath);
    }

    const size_t arrayEnd = findMatchingJsonDelimiter(content, arrayPos, '[', ']');
    const regex namePattern("\"name\"\\s*:\\s*\"([^\"]+)\"");
    const regex rawEntriesPattern("\"raw_entries\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
    string updated = content;
    bool foundSample = false;

    size_t pos = arrayPos + 1;
    while (pos < arrayEnd) {
        pos = skipWhitespace(content, pos);
        if (pos >= arrayEnd) {
            break;
        }
        if (content[pos] == ',') {
            ++pos;
            continue;
        }
        if (content[pos] != '{') {
            throw runtime_error("Expected sample object in sample config: " + sampleConfigPath);
        }

        const size_t objectEnd = findMatchingJsonDelimiter(content, pos, '{', '}');
        const string objectText = content.substr(pos, objectEnd - pos + 1);
        smatch nameMatch;
        if (regex_search(objectText, nameMatch, namePattern) && nameMatch.size() >= 2 &&
            nameMatch[1].str() == sampleName) {
            smatch rawEntriesMatch;
            if (!regex_search(objectText, rawEntriesMatch, rawEntriesPattern) || rawEntriesMatch.size() < 2) {
                throw runtime_error("Cannot find raw_entries for sample '" + sampleName +
                                    "' in sample config: " + sampleConfigPath);
            }

            const size_t replacePos = pos + static_cast<size_t>(rawEntriesMatch.position(1));
            const size_t replaceLen = rawEntriesMatch.length(1);
            updated.replace(replacePos, replaceLen, to_string(static_cast<long long>(rawEntries)));
            foundSample = true;
            break;
        }
        pos = objectEnd + 1;
    }

    if (!foundSample) {
        throw runtime_error("Cannot find sample '" + sampleName + "' in sample config: " + sampleConfigPath);
    }

    const fs::path targetPath(sampleConfigPath);
    const fs::path tempPath = targetPath.string() + ".tmp";
    ofstream fout(tempPath);
    if (!fout) {
        throw runtime_error("Cannot write temporary sample config file: " + tempPath.string());
    }
    fout << updated;
    fout.close();
    if (!fout) {
        throw runtime_error("Failed writing sample config file: " + tempPath.string());
    }

    std::error_code ec;
    fs::rename(tempPath, targetPath, ec);
    if (ec) {
        throw runtime_error("Failed to replace sample config file '" + targetPath.string() +
                            "': " + ec.message());
    }
}

OutputScalarConfig parseOutputScalar(const JsonValue& node) {
    OutputScalarConfig config;
    config.name = node.at("name").asString();
    config.type = parseDataType(node.at("type").asString());
    config.onlyMC = node.getBoolOr("onlyMC", false);
    config.formulaText = node.at("formula").asString();
    config.formula = compileExpression(config.formulaText);
    config.collection = node.getStringOr("collection", "");
    config.slots = node.getIntOr("slots", 0);
    if (!config.collection.empty() && config.slots <= 0) {
        throw runtime_error("Output scalar with collection must define slots: " + config.name);
    }
    return config;
}

vector<OutputScalarConfig> parseOutputScalarGroup(const JsonValue& node, const string& key) {
    vector<OutputScalarConfig> out;
    if (!node.contains(key)) {
        return out;
    }
    for (const auto& item : node.at(key).asArray()) {
        out.push_back(parseOutputScalar(item));
    }
    return out;
}

void finalizeInputCollection(InputCollectionConfig& collection) {
    for (size_t index = 0; index < collection.fields.size(); ++index) {
        const string& name = collection.fields[index].name;
        if (name == collection.ptField) {
            collection.ptIndex = static_cast<int>(index);
        }
        if (name == collection.etaField) {
            collection.etaIndex = static_cast<int>(index);
        }
        if (name == collection.phiField) {
            collection.phiIndex = static_cast<int>(index);
        }
        if (!collection.massField.empty() && name == collection.massField) {
            collection.massIndex = static_cast<int>(index);
        }
    }

    if (collection.ptIndex < 0 || collection.etaIndex < 0 || collection.phiIndex < 0) {
        throw runtime_error("Missing pt/eta/phi field in input collection: " + collection.name);
    }
}

BranchConfig loadBranchConfig(const AppConfig& appConfig) {
    const JsonValue payload = loadJsonPath(resolveReferencedPath(appConfig.configPath, kBranchConfigPath));

    BranchConfig config;
    for (const auto& node : payload.at("input").at("scalars").asArray()) {
        ScalarInputConfig scalar;
        scalar.name = node.at("name").asString();
        scalar.branch = node.getStringOr("branch", scalar.name);
        scalar.type = parseDataType(node.at("type").asString());
        scalar.onlyMC = node.getBoolOr("onlyMC", false);
        config.scalars.push_back(std::move(scalar));
    }

    for (const auto& node : payload.at("input").at("collections").asArray()) {
        InputCollectionConfig collection;
        collection.name = node.at("name").asString();
        collection.sizeName = node.at("size").asString();
        collection.maxSize = node.at("max_size").asInt();
        if (node.contains("p4")) {
            const auto& p4 = node.at("p4");
            collection.ptField = p4.at("pt").asString();
            collection.etaField = p4.at("eta").asString();
            collection.phiField = p4.at("phi").asString();
            collection.massField = p4.getStringOr("mass", "");
            collection.defaultMass = static_cast<float>(p4.getNumberOr("default_mass", 0.));
        }
        for (const auto& fieldNode : node.at("fields").asArray()) {
            ArrayInputConfig field;
            field.name = fieldNode.at("name").asString();
            field.branch = fieldNode.getStringOr("branch", field.name);
            field.type = parseDataType(fieldNode.at("type").asString());
            field.onlyMC = fieldNode.getBoolOr("onlyMC", false);
            field.maxSize = collection.maxSize;
            field.initBuffer();
            collection.fields.push_back(std::move(field));
        }
        finalizeInputCollection(collection);
        config.collections.push_back(std::move(collection));
    }

    const auto& output = payload.at("output");
    for (const auto& node : output.at("trees").asArray()) {
        TreeConfig treeConfig;
        treeConfig.name = node.at("name").asString();
        treeConfig.title = node.at("title").asString();
        treeConfig.selection = node.at("selection").asString();
        if (node.contains("scalars")) {
            const auto& scalarNode = node.at("scalars");
            treeConfig.regularScalars = parseOutputScalarGroup(scalarNode, "regular");
            treeConfig.extremaScalars = parseOutputScalarGroup(scalarNode, "extrema");
        }
        config.trees.push_back(std::move(treeConfig));
    }

    return config;
}

SelectionConfig loadSelectionConfig(const AppConfig& appConfig) {
    const JsonValue payload = loadJsonPath(resolveReferencedPath(appConfig.configPath, kSelectionConfigPath));

    SelectionConfig config;
    config.eventPreselectionText = payload.getStringOr("event_preselection", "1");
    config.eventPreselection = compileExpression(config.eventPreselectionText);

    for (const auto& node : payload.at("collections").asArray()) {
        RuntimeCollectionConfig collection;
        collection.name = node.at("name").asString();
        collection.source = node.getStringOr("source", "");
        if (node.contains("merge")) {
            collection.merge = node.at("merge").toStringArray();
        }
        collection.selectionText = node.getStringOr("selection", "1");
        collection.selectionExpr = compileExpression(collection.selectionText);
        collection.dedupCollection = node.getStringOr("deduplicate_against", "");
        collection.dedupText = node.getStringOr("deduplicate", "");
        if (!collection.dedupText.empty()) {
            collection.dedupExpr = compileExpression(collection.dedupText);
        }
        collection.sortText = node.getStringOr("sort", "");
        if (!collection.sortText.empty()) {
            collection.sortRule = parseSortRule(collection.sortText);
        }
        config.collectionOrder.push_back(collection.name);
        config.collections[collection.name] = std::move(collection);
    }

    if (payload.contains("tree_selection")) {
        for (const auto& item : payload.at("tree_selection").asObject()) {
            config.treeSelectionText[item.first] = item.second.asString();
            config.treeSelections[item.first] = compileExpression(item.second.asString());
        }
    }

    return config;
}

ObjectSchema makeSchema(const vector<string>& fields) {
    ObjectSchema schema;
    schema.fields = fields;
    for (size_t index = 0; index < fields.size(); ++index) {
        schema.indexByName[fields[index]] = index;
    }
    return schema;
}

ObjectSchema makeSchemaFromCollection(const InputCollectionConfig& collection) {
    vector<string> fields;
    fields.reserve(collection.fields.size());
    ObjectSchema schema;
    schema.fields.reserve(collection.fields.size());
    schema.indexByName.reserve(collection.fields.size() * 2);
    for (const auto& field : collection.fields) {
        fields.push_back(field.name);
        schema.fields.push_back(field.name);
        schema.indexByName[field.name] = schema.fields.size() - 1;
        if (!field.branch.empty()) {
            schema.indexByName[field.branch] = schema.fields.size() - 1;
        }
    }
    return schema;
}

bool hasObjectField(const RuntimeCollection& collection, const string& fieldName) {
    return collection.schema.indexByName.find(fieldName) != collection.schema.indexByName.end();
}

float getObjectField(const RuntimeCollection& collection, const RuntimeObject& object, const string& fieldName, float defaultValue = def) {
    const auto it = collection.schema.indexByName.find(fieldName);
    if (it == collection.schema.indexByName.end()) {
        return defaultValue;
    }
    return object.values[it->second];
}

RuntimeObject remapObject(const RuntimeCollection& source, const RuntimeObject& sourceObject, const ObjectSchema& targetSchema) {
    RuntimeObject out;
    out.values.assign(targetSchema.fields.size(), def);
    out.p4 = sourceObject.p4;
    for (size_t index = 0; index < targetSchema.fields.size(); ++index) {
        out.values[index] = getObjectField(source, sourceObject, targetSchema.fields[index], def);
    }
    return out;
}

RuntimeCollection mergeCollections(const string& name, const vector<const RuntimeCollection*>& collections) {
    vector<string> mergedFields;
    unordered_map<string, bool> seen;
    for (const auto* collection : collections) {
        if (!collection) {
            continue;
        }
        for (const auto& field : collection->schema.fields) {
            if (!seen[field]) {
                seen[field] = true;
                mergedFields.push_back(field);
            }
        }
    }

    RuntimeCollection merged;
    merged.name = name;
    merged.schema = makeSchema(mergedFields);

    for (const auto* collection : collections) {
        if (!collection) {
            continue;
        }
        for (const auto& object : collection->objects) {
            merged.objects.push_back(remapObject(*collection, object, merged.schema));
        }
    }

    return merged;
}

vector<PileupBin> loadPileupWeights(const string& path) {
    ifstream fin(path);
    if (!fin) {
        throw runtime_error("Cannot open pileup weight CSV: " + path);
    }
    vector<PileupBin> bins;
    string line;
    bool firstLine = true;
    while (getline(fin, line)) {
        if (firstLine) {
            firstLine = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }
        istringstream ss(line);
        string tok;
        PileupBin bin;
        int col = 0;
        while (getline(ss, tok, ',')) {
            switch (col) {
                case 0: bin.binLow = stof(tok); break;
                case 1: bin.binHigh = stof(tok); break;
                case 2: bin.weight = stof(tok); break;
                case 3: bin.weightLow = stof(tok); break;
                case 4: bin.weightHigh = stof(tok); break;
            }
            ++col;
        }
        if (col >= 5) {
            bins.push_back(bin);
        }
    }
    return bins;
}

long double lookupPileupWeight(const vector<PileupBin>& bins, float pu, int col) {
    for (const auto& bin : bins) {
        if (pu >= bin.binLow && pu < bin.binHigh) {
            if (col == 0) {
                return static_cast<long double>(bin.weight);
            }
            if (col == 1) {
                return static_cast<long double>(bin.weightLow);
            }
            return static_cast<long double>(bin.weightHigh);
        }
    }
    if (!bins.empty() && pu == bins.back().binHigh) {
        const auto& last = bins.back();
        if (col == 0) {
            return static_cast<long double>(last.weight);
        }
        if (col == 1) {
            return static_cast<long double>(last.weightLow);
        }
        return static_cast<long double>(last.weightHigh);
    }
    return 1.0L;
}

// Enable and bind theory weight branches on an input TTree. Silently skips missing branches.
void activateTheoryInputBranches(TTree* tree, TheoryWeightBufs& buf) {
    const auto tryBind = [&](const char* name, void* addr) {
        if (tree->GetBranch(name) != nullptr) {
            tree->SetBranchStatus(name, 1);
            tree->SetBranchAddress(name, addr);
        }
    };
    tryBind("genWeight",       &buf.genWeight);
    tryBind("nLHEPdfWeight",   &buf.nLHEPdfWeight);
    tryBind("LHEPdfWeight",     buf.LHEPdfWeight);
    tryBind("nLHEScaleWeight", &buf.nLHEScaleWeight);
    tryBind("LHEScaleWeight",   buf.LHEScaleWeight);
    tryBind("nPSWeight",       &buf.nPSWeight);
    tryBind("PSWeight",         buf.PSWeight);
}

// Create fixed-size array branches on an output tree, pointed at treeState.theoryOutBuf.
void setupTheoryOutputBranches(OutputTreeState& treeState) {
    treeState.tree->Branch("genWeight",      &treeState.theoryOutBuf.genWeight,       "genWeight/F");
    treeState.tree->Branch("LHEPdfWeight",    treeState.theoryOutBuf.LHEPdfWeight,    "LHEPdfWeight[101]/F");
    treeState.tree->Branch("LHEPdfWeightAlphaS", treeState.theoryOutBuf.LHEPdfWeightAlphaS, "LHEPdfWeightAlphaS[2]/F");
    treeState.tree->Branch("LHEScaleWeight",  treeState.theoryOutBuf.LHEScaleWeight,  "LHEScaleWeight[9]/F");
    treeState.tree->Branch("PSWeight",        treeState.theoryOutBuf.PSWeight,        "PSWeight[4]/F");
    treeState.hasTheoryBranches = true;
}

// Copy input theory weight buffers to an output struct, padding unused slots with 1.0.
void copyTheoryWeights(const TheoryWeightBufs& src, TheoryOutBufs& dst) {
    dst.genWeight = src.genWeight;
    const int nPdf = min(src.nLHEPdfWeight, TheoryOutBufs::kNPdf);
    for (int i = 0; i < nPdf; ++i) dst.LHEPdfWeight[i] = src.LHEPdfWeight[i];
    for (int i = nPdf; i < TheoryOutBufs::kNPdf; ++i) dst.LHEPdfWeight[i] = 1.f;
    // alpha_s variations: for NNPDF31_*_hessian_pdfas (LHA 306000) the two
    // alpha_s members (306101/306102) follow the central + 100 Hessian members
    // at source indices 101 and 102. Stored in a dedicated branch because
    // LHEPdfWeight keeps only the 101 PDF members. Defaults to 1.0 (no
    // variation) when the source set has no alpha_s members.
    for (int i = 0; i < TheoryOutBufs::kNAlphaS; ++i) {
        const int srcIdx = TheoryOutBufs::kNPdf + i;   // source indices 101, 102
        dst.LHEPdfWeightAlphaS[i] =
            (srcIdx < src.nLHEPdfWeight) ? src.LHEPdfWeight[srcIdx] : 1.f;
    }
    const int nScale = min(src.nLHEScaleWeight, TheoryOutBufs::kNScale);
    for (int i = 0; i < nScale; ++i) dst.LHEScaleWeight[i] = src.LHEScaleWeight[i];
    for (int i = nScale; i < TheoryOutBufs::kNScale; ++i) dst.LHEScaleWeight[i] = 1.f;
    const int nPS = min(src.nPSWeight, TheoryOutBufs::kNPS);
    for (int i = 0; i < nPS; ++i) dst.PSWeight[i] = src.PSWeight[i];
    for (int i = nPS; i < TheoryOutBufs::kNPS; ++i) dst.PSWeight[i] = 1.f;
}

unordered_map<string, long double> buildRawScalarValues(const BranchConfig& branchConfig,
                                                        const SampleMeta& sampleMeta,
                                                        const vector<PileupBin>* pileupWeights = nullptr,
                                                        const TheoryWeightBufs* theoryBufs = nullptr) {
    unordered_map<string, long double> values;
    values.reserve(branchConfig.scalars.size() + 8);
    for (const auto& scalar : branchConfig.scalars) {
        values[scalar.name] = scalar.numericValue();
    }
    values["sample_ID"] = sampleMeta.sampleId;
    values["is_MC"] = sampleMeta.isMC ? 1. : 0.;
    values["is_signal"] = sampleMeta.isSignal ? 1. : 0.;
    values["xsection"] = sampleMeta.xsection;
    values["lumi"] = sampleMeta.lumi;
    if (sampleMeta.isMC) {
        const auto puIt = values.find("Pileup_nTrueInt");
        const float puValue = (puIt != values.end()) ? static_cast<float>(puIt->second) : 0.f;
        if (pileupWeights != nullptr && !pileupWeights->empty()) {
            values["weight_pu"] = lookupPileupWeight(*pileupWeights, puValue, 0);
            values["weight_pu_down"] = lookupPileupWeight(*pileupWeights, puValue, 1);
            values["weight_pu_up"] = lookupPileupWeight(*pileupWeights, puValue, 2);
        } else {
            values["weight_pu"] = 1.;
            values["weight_pu_down"] = 1.;
            values["weight_pu_up"] = 1.;
        }
        values["genWeight"] = theoryBufs ? static_cast<long double>(theoryBufs->genWeight) : 1.L;
    }
    return values;
}

TLorentzVector buildObjectP4(const InputCollectionConfig& config, int index) {
    TLorentzVector vector;
    const float pt = config.fields[config.ptIndex].valueAt(index);
    const float eta = config.fields[config.etaIndex].valueAt(index);
    const float phi = config.fields[config.phiIndex].valueAt(index);
    const float mass = (config.massIndex >= 0) ? config.fields[config.massIndex].valueAt(index) : config.defaultMass;
    vector.SetPtEtaPhiM(pt, eta, phi, mass);
    return vector;
}

RuntimeCollection buildInputCollection(const InputCollectionConfig& config,
                                       const unordered_map<string, long double>& rawVars) {
    const auto sizeIt = rawVars.find(config.sizeName);
    if (sizeIt == rawVars.end()) {
        throw runtime_error("Input collection size not found: " + config.sizeName);
    }

    RuntimeCollection collection;
    collection.name = config.name;
    collection.schema = makeSchemaFromCollection(config);

    const int size = min(static_cast<int>(sizeIt->second), config.maxSize);
    collection.objects.reserve(size);
    for (int index = 0; index < size; ++index) {
        RuntimeObject object;
        object.values.reserve(collection.schema.fields.size());
        for (const auto& field : config.fields) {
            object.values.push_back(field.valueAt(index));
        }
        object.p4 = buildObjectP4(config, index);
        collection.objects.push_back(std::move(object));
    }

    return collection;
}

const RuntimeCollection* findCollection(const EvalContext& context, const string& name) {
    if (context.collections) {
        const auto it = context.collections->find(name);
        if (it != context.collections->end()) {
            return &it->second;
        }
    }
    if (context.inputCollections) {
        const auto it = context.inputCollections->find(name);
        if (it != context.inputCollections->end()) {
            return &it->second;
        }
    }
    return nullptr;
}

Value makeNumberValue(long double value) {
    Value out;
    out.kind = Value::Kind::Number;
    out.number = value;
    return out;
}

Value makeObjectValue(const RuntimeCollection* collection, const RuntimeObject* object) {
    Value out;
    out.kind = Value::Kind::ObjectRef;
    out.collection = collection;
    out.object = object;
    return out;
}

Value makeCollectionValue(const RuntimeCollection* collection) {
    Value out;
    out.kind = Value::Kind::CollectionRef;
    out.collection = collection;
    return out;
}

Value makeP4Value(const TLorentzVector& p4) {
    Value out;
    out.kind = Value::Kind::P4;
    out.p4 = p4;
    return out;
}

long double toNumber(const Value& value) {
    if (value.kind != Value::Kind::Number) {
        throw runtime_error("Numeric value expected in expression");
    }
    return value.number;
}

TLorentzVector toP4(const Value& value) {
    if (value.kind == Value::Kind::P4) {
        return value.p4;
    }
    if (value.kind == Value::Kind::ObjectRef) {
        return value.object->p4;
    }
    throw runtime_error("Object or p4 expected in expression");
}

const RuntimeCollection* toCollection(const Value& value) {
    if (value.kind != Value::Kind::CollectionRef || !value.collection) {
        throw runtime_error("Collection expected in expression");
    }
    return value.collection;
}

bool truthy(const Value& value) {
    if (value.kind == Value::Kind::Number) {
        return value.number != 0.;
    }
    if (value.kind == Value::Kind::ObjectRef) {
        return value.object != nullptr;
    }
    if (value.kind == Value::Kind::CollectionRef) {
        return value.collection != nullptr;
    }
    return true;
}

double pairMetric(const string& metric, const TLorentzVector& lhs, const TLorentzVector& rhs) {
    if (metric == "deltaR") {
        return lhs.DeltaR(rhs);
    }
    if (metric == "deltaPhi") {
        return lhs.DeltaPhi(rhs);
    }
    throw runtime_error("Unsupported metric: " + metric);
}

Value evalExpression(const ExprPtr& expr, const EvalContext& context);

long double evalNumber(const ExprPtr& expr, const EvalContext& context) {
    return toNumber(evalExpression(expr, context));
}

// Eigenvalues (descending: l1 >= l2 >= l3) of the real symmetric 3x3 matrix
// [[a, d, e], [d, b, f], [e, f, c]] via the analytic trigonometric method
// (Smith 1961) -- avoids pulling in a matrix-eigen dependency.
void symmetricEigenvalues3(double a, double b, double c, double d, double e, double f,
                           double& l1, double& l2, double& l3) {
    const double p1 = d * d + e * e + f * f;
    if (p1 <= 1e-18) {  // already diagonal
        double v[3] = {a, b, c};
        std::sort(v, v + 3);
        l1 = v[2]; l2 = v[1]; l3 = v[0];
        return;
    }
    const double q = (a + b + c) / 3.0;
    const double p2 = (a - q) * (a - q) + (b - q) * (b - q) + (c - q) * (c - q) + 2.0 * p1;
    const double p = std::sqrt(p2 / 6.0);
    const double b11 = (a - q) / p, b22 = (b - q) / p, b33 = (c - q) / p;
    const double b12 = d / p, b13 = e / p, b23 = f / p;
    const double detB = b11 * (b22 * b33 - b23 * b23)
                      - b12 * (b12 * b33 - b23 * b13)
                      + b13 * (b12 * b23 - b22 * b13);
    double r = detB / 2.0;
    if (r <= -1.0) r = -1.0; else if (r >= 1.0) r = 1.0;
    const double phi = std::acos(r) / 3.0;
    const double kPi = 3.14159265358979323846;
    l1 = q + 2.0 * p * std::cos(phi);
    l3 = q + 2.0 * p * std::cos(phi + 2.0 * kPi / 3.0);
    l2 = 3.0 * q - l1 - l3;
}

// Event-shape variables from the normalized momentum (sphericity) tensor built
// over the objects given as arguments.  Each argument may be a collection (all of
// its objects are pooled) or a single object / p4 (e.g. ak8[0], ak4[1]), so the
// caller can choose exactly which objects define the system per category:
//   S^{ab} = sum_i p_i^a p_i^b / sum_i |p_i|^2 ,  eigenvalues l1 >= l2 >= l3 (sum = 1)
//   sphericity = 1.5*(l2 + l3),  aplanarity = 1.5*l3,  planarity = l2 - l3
// Returns -1 when fewer than 2 objects with non-zero momentum are present.
double evalEventShape(const string& op, const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.empty()) {
        throw runtime_error(op + " requires at least one object or collection argument");
    }
    double sxx = 0, syy = 0, szz = 0, sxy = 0, sxz = 0, syz = 0, norm = 0;
    int n = 0;
    auto addP4 = [&](const TLorentzVector& p) {
        const double px = p.Px(), py = p.Py(), pz = p.Pz();
        const double p2 = px * px + py * py + pz * pz;
        if (p2 <= 0.0) return;
        sxx += px * px; syy += py * py; szz += pz * pz;
        sxy += px * py; sxz += px * pz; syz += py * pz;
        norm += p2;
        ++n;
    };
    for (const auto& arg : args) {
        const Value value = evalExpression(arg, context);
        if (value.kind == Value::Kind::CollectionRef) {
            for (const auto& object : toCollection(value)->objects) addP4(object.p4);
        } else {
            addP4(toP4(value));
        }
    }
    if (n < 2 || norm <= 0.0) return -1.0;
    sxx /= norm; syy /= norm; szz /= norm; sxy /= norm; sxz /= norm; syz /= norm;
    double l1, l2, l3;
    symmetricEigenvalues3(sxx, syy, szz, sxy, sxz, syz, l1, l2, l3);
    if (l3 < 0.0) l3 = 0.0;  // guard tiny negative eigenvalue from round-off
    if (op == "sphericity") return 1.5 * (l2 + l3);
    if (op == "aplanarity") return 1.5 * l3;
    if (op == "planarity") return l2 - l3;
    throw runtime_error("Unsupported event-shape: " + op);
}

Value evalAggregation(const string& op,
                      const vector<ExprPtr>& args,
                      const EvalContext& context) {
    if (args.size() < 2) {
        throw runtime_error(op + " requires at least 2 arguments");
    }

    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    if (op == "sum") {
        long double total = 0.;
        for (const auto& object : collection->objects) {
            EvalContext loop = context;
            loop.currentCollection = collection;
            loop.currentObject = &object;
            total += evalNumber(args[1], loop);
        }
        return makeNumberValue(total);
    }

    const long double defaultValue = (args.size() >= 3) ? evalNumber(args[2], context) : def;
    bool found = false;
    long double best = defaultValue;
    for (const auto& object : collection->objects) {
        EvalContext loop = context;
        loop.currentCollection = collection;
        loop.currentObject = &object;
        const long double value = evalNumber(args[1], loop);
        if (!found || (op == "max_value" && value > best) || (op == "min_value" && value < best)) {
            best = value;
            found = true;
        }
    }
    return makeNumberValue(found ? best : defaultValue);
}

Value evalNthMaxValue(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 3) {
        throw runtime_error("nth_max_value requires collection, expression, and rank arguments");
    }

    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const int rank = static_cast<int>(llround(evalNumber(args[2], context)));
    if (rank < 1) {
        throw runtime_error("nth_max_value rank must be >= 1");
    }
    const long double defaultValue = (args.size() >= 4) ? evalNumber(args[3], context) : def;
    if (static_cast<int>(collection->objects.size()) < rank) {
        return makeNumberValue(defaultValue);
    }

    vector<long double> values;
    values.reserve(collection->objects.size());
    for (const auto& object : collection->objects) {
        EvalContext loop = context;
        loop.currentCollection = collection;
        loop.currentObject = &object;
        values.push_back(evalNumber(args[1], loop));
    }

    const auto nth = values.begin() + (rank - 1);
    nth_element(values.begin(), nth, values.end(), greater<long double>());
    return makeNumberValue(*nth);
}

Value evalValueAtMax(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 3) {
        throw runtime_error("value_at_max requires collection, key expression, and value expression arguments");
    }

    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const long double defaultValue = (args.size() >= 4) ? evalNumber(args[3], context) : def;
    bool found = false;
    long double bestKey = defaultValue;
    long double bestValue = defaultValue;
    for (const auto& object : collection->objects) {
        EvalContext loop = context;
        loop.currentCollection = collection;
        loop.currentObject = &object;
        const long double key = evalNumber(args[1], loop);
        if (!found || key > bestKey) {
            bestKey = key;
            bestValue = evalNumber(args[2], loop);
            found = true;
        }
    }
    return makeNumberValue(found ? bestValue : defaultValue);
}

Value evalValueAtNthMax(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 4) {
        throw runtime_error("value_at_nth_max requires collection, key expression, value expression, and rank arguments");
    }

    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const int rank = static_cast<int>(llround(evalNumber(args[3], context)));
    if (rank < 1) {
        throw runtime_error("value_at_nth_max rank must be >= 1");
    }
    const long double defaultValue = (args.size() >= 5) ? evalNumber(args[4], context) : def;
    if (static_cast<int>(collection->objects.size()) < rank) {
        return makeNumberValue(defaultValue);
    }

    vector<pair<long double, long double>> keyedValues;
    keyedValues.reserve(collection->objects.size());
    for (const auto& object : collection->objects) {
        EvalContext loop = context;
        loop.currentCollection = collection;
        loop.currentObject = &object;
        keyedValues.emplace_back(evalNumber(args[1], loop), evalNumber(args[2], loop));
    }

    const auto nth = keyedValues.begin() + (rank - 1);
    nth_element(
        keyedValues.begin(),
        nth,
        keyedValues.end(),
        [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first;
        }
    );
    return makeNumberValue(nth->second);
}

Value evalPairwiseMetric(const string& op,
                         const vector<ExprPtr>& args,
                         const EvalContext& context) {
    if (args.empty()) {
        throw runtime_error(op + " requires a collection argument");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const int limit = (args.size() >= 2) ? static_cast<int>(llround(evalNumber(args[1], context)))
                                         : static_cast<int>(collection->objects.size());
    const int count = min(limit, static_cast<int>(collection->objects.size()));
    if (count < 2) {
        return makeNumberValue(kMissingDistance);
    }

    const bool takeMin = (op.find("_min_") != string::npos);
    const string metric = (op.find("deltaPhi") != string::npos) ? "deltaPhi" : "deltaR";
    bool first = true;
    double best = 0.;
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            const double value = pairMetric(metric, collection->objects[i].p4, collection->objects[j].p4);
            if (first || (takeMin && value < best) || (!takeMin && value > best)) {
                best = value;
                first = false;
            }
        }
    }
    return makeNumberValue(first ? kMissingDistance : best);
}

Value evalClosestMetric(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 2) {
        throw runtime_error("closest_deltaR requires a collection and at least one reference");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    if (collection->objects.empty()) {
        return makeNumberValue(kMissingDistance);
    }

    vector<TLorentzVector> refs;
    refs.reserve(args.size() - 1);
    for (size_t index = 1; index < args.size(); ++index) {
        refs.push_back(toP4(evalExpression(args[index], context)));
    }
    if (refs.empty()) {
        return makeNumberValue(kMissingDistance);
    }

    bool found = false;
    double best = 0.;
    for (const auto& object : collection->objects) {
        for (const auto& ref : refs) {
            const double value = object.p4.DeltaR(ref);
            if (!found || value < best) {
                best = value;
                found = true;
            }
        }
    }
    return makeNumberValue(found ? best : kMissingDistance);
}

Value evalMinDeltaR(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 2) {
        throw runtime_error("min_deltaR requires an object and a collection");
    }

    const TLorentzVector objectP4 = toP4(evalExpression(args[0], context));
    const RuntimeCollection* collection = toCollection(evalExpression(args[1], context));
    const int limit = (args.size() >= 3) ? static_cast<int>(llround(evalNumber(args[2], context)))
                                         : static_cast<int>(collection->objects.size());
    const int count = min(limit, static_cast<int>(collection->objects.size()));
    if (count < 1) {
        return makeNumberValue(kLargeDistance);
    }

    double best = numeric_limits<double>::max();
    for (int index = 0; index < count; ++index) {
        best = min(best, objectP4.DeltaR(collection->objects[index].p4));
    }
    return makeNumberValue(best);
}

// Among the reference objects (args[1..]), pick the one closest to args[0] in deltaR
// and return the deltaPhi between args[0] and that closest reference. Used to attach,
// per AK4 jet, the deltaPhi to the closest signal AK8 jet.
Value evalDeltaPhiAtMinDeltaR(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 2) {
        throw runtime_error("deltaPhi_at_min_deltaR requires an object and at least one reference");
    }

    const TLorentzVector objectP4 = toP4(evalExpression(args[0], context));

    bool found = false;
    double bestDeltaR = numeric_limits<double>::max();
    double bestDeltaPhi = kMissingDistance;
    for (size_t index = 1; index < args.size(); ++index) {
        const TLorentzVector ref = toP4(evalExpression(args[index], context));
        const double dr = objectP4.DeltaR(ref);
        if (!found || dr < bestDeltaR) {
            bestDeltaR = dr;
            bestDeltaPhi = objectP4.DeltaPhi(ref);
            found = true;
        }
    }
    return makeNumberValue(found ? bestDeltaPhi : kMissingDistance);
}

Value evalCall(const ExprPtr& expr, const EvalContext& context) {
    const string& op = expr->text;
    const auto& args = expr->args;

    if (op == "abs") {
        return makeNumberValue(fabsl(evalNumber(args.at(0), context)));
    }
    if (op == "sqrt") {
        return makeNumberValue(sqrtl(evalNumber(args.at(0), context)));
    }
    if (op == "cos") {
        return makeNumberValue(cosl(evalNumber(args.at(0), context)));
    }
    if (op == "sin") {
        return makeNumberValue(sinl(evalNumber(args.at(0), context)));
    }
    if (op == "pow") {
        return makeNumberValue(powl(evalNumber(args.at(0), context), evalNumber(args.at(1), context)));
    }
    if (op == "min") {
        long double best = 0.;
        bool first = true;
        for (const auto& arg : args) {
            const long double value = evalNumber(arg, context);
            if (first || value < best) {
                best = value;
                first = false;
            }
        }
        return makeNumberValue(best);
    }
    if (op == "max") {
        long double best = 0.;
        bool first = true;
        for (const auto& arg : args) {
            const long double value = evalNumber(arg, context);
            if (first || value > best) {
                best = value;
                first = false;
            }
        }
        return makeNumberValue(best);
    }
    if (op == "safe_div") {
        const long double numerator = evalNumber(args.at(0), context);
        const long double denominator = evalNumber(args.at(1), context);
        const long double fallback = (args.size() >= 3) ? evalNumber(args.at(2), context) : 0.;
        if (denominator == 0.) {
            return makeNumberValue(fallback);
        }
        return makeNumberValue(numerator / denominator);
    }
    if (op == "first_valid") {
        for (const auto& arg : args) {
            const long double value = evalNumber(arg, context);
            if (fabsl(value - def) > 1e-9L) {
                return makeNumberValue(value);
            }
        }
        return makeNumberValue(def);
    }
    if (op == "size") {
        return makeNumberValue(static_cast<long double>(toCollection(evalExpression(args.at(0), context))->objects.size()));
    }
    if (op == "sum" || op == "max_value" || op == "min_value") {
        return evalAggregation(op, args, context);
    }
    if (op == "sphericity" || op == "aplanarity" || op == "planarity") {
        return makeNumberValue(static_cast<long double>(evalEventShape(op, args, context)));
    }
    if (op == "nth_max_value") {
        return evalNthMaxValue(args, context);
    }
    if (op == "value_at_max") {
        return evalValueAtMax(args, context);
    }
    if (op == "value_at_nth_max") {
        return evalValueAtNthMax(args, context);
    }
    if (op == "mass") {
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).M());
    }
    if (op == "pt") {
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).Pt());
    }
    if (op == "eta") {
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).Eta());
    }
    if (op == "phi") {
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).Phi());
    }
    if (op == "deltaR") {
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).DeltaR(toP4(evalExpression(args.at(1), context))));
    }
    if (op == "deltaPhi") {
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).DeltaPhi(toP4(evalExpression(args.at(1), context))));
    }
    if (op == "relPtDiff") {
        const double pt1 = toP4(evalExpression(args.at(0), context)).Pt();
        const double pt2 = toP4(evalExpression(args.at(1), context)).Pt();
        if (pt1 == 0.) {
            return makeNumberValue(kLargeDistance);
        }
        return makeNumberValue(fabs(pt1 - pt2) / pt1);
    }
    if (op == "pair_min_deltaR" || op == "pair_max_deltaR" || op == "pair_min_deltaPhi" || op == "pair_max_deltaPhi") {
        return evalPairwiseMetric(op, args, context);
    }
    if (op == "closest_deltaR") {
        return evalClosestMetric(args, context);
    }
    if (op == "min_deltaR") {
        return evalMinDeltaR(args, context);
    }
    if (op == "deltaPhi_at_min_deltaR") {
        return evalDeltaPhiAtMinDeltaR(args, context);
    }

    throw runtime_error("Unsupported function in expression: " + op);
}

Value evalExpression(const ExprPtr& expr, const EvalContext& context) {
    if (!expr) {
        throw runtime_error("Null expression");
    }

    if (expr->kind == ExprKind::Number) {
        return makeNumberValue(expr->number);
    }
    if (expr->kind == ExprKind::Identifier) {
        if (expr->text == "true") {
            return makeNumberValue(1.);
        }
        if (expr->text == "false") {
            return makeNumberValue(0.);
        }
        if (expr->text == "self") {
            if (!context.currentCollection || !context.currentObject) {
                throw runtime_error("self used without current object");
            }
            return makeObjectValue(context.currentCollection, context.currentObject);
        }
        if (expr->text == "other") {
            if (!context.otherCollection || !context.otherObject) {
                throw runtime_error("other used without comparison object");
            }
            return makeObjectValue(context.otherCollection, context.otherObject);
        }
        if (context.currentCollection && context.currentObject && hasObjectField(*context.currentCollection, expr->text)) {
            return makeNumberValue(getObjectField(*context.currentCollection, *context.currentObject, expr->text, def));
        }
        if (context.vars && context.vars->count(expr->text)) {
            return makeNumberValue(context.vars->at(expr->text));
        }
        if (context.rawScalars && context.rawScalars->count(expr->text)) {
            return makeNumberValue(context.rawScalars->at(expr->text)->numericValue());
        }
        const RuntimeCollection* collection = findCollection(context, expr->text);
        if (collection) {
            return makeCollectionValue(collection);
        }
        if (context.currentCollection) {
            ostringstream ss;
            ss << "Unknown identifier in expression: " << expr->text
               << " (current collection: " << context.currentCollection->name << ")";
            throw runtime_error(ss.str());
        }
        throw runtime_error("Unknown identifier in expression: " + expr->text);
    }
    if (expr->kind == ExprKind::Unary) {
        const Value value = evalExpression(expr->lhs, context);
        if (expr->text == "+") {
            return makeNumberValue(+toNumber(value));
        }
        if (expr->text == "-") {
            return makeNumberValue(-toNumber(value));
        }
        if (expr->text == "!") {
            return makeNumberValue(truthy(value) ? 0. : 1.);
        }
        throw runtime_error("Unsupported unary operator: " + expr->text);
    }
    if (expr->kind == ExprKind::Binary) {
        if (expr->text == "&&") {
            return makeNumberValue((truthy(evalExpression(expr->lhs, context)) && truthy(evalExpression(expr->rhs, context))) ? 1. : 0.);
        }
        if (expr->text == "||") {
            return makeNumberValue((truthy(evalExpression(expr->lhs, context)) || truthy(evalExpression(expr->rhs, context))) ? 1. : 0.);
        }

        const Value lhs = evalExpression(expr->lhs, context);
        const Value rhs = evalExpression(expr->rhs, context);

        if (expr->text == "+" || expr->text == "-") {
            const bool lhsP4 = (lhs.kind == Value::Kind::ObjectRef || lhs.kind == Value::Kind::P4);
            const bool rhsP4 = (rhs.kind == Value::Kind::ObjectRef || rhs.kind == Value::Kind::P4);
            if (lhsP4 || rhsP4) {
                TLorentzVector total = toP4(lhs);
                if (expr->text == "+") {
                    total += toP4(rhs);
                } else {
                    total -= toP4(rhs);
                }
                return makeP4Value(total);
            }
        }

        const long double leftNumber = toNumber(lhs);
        const long double rightNumber = toNumber(rhs);
        if (expr->text == "+") {
            return makeNumberValue(leftNumber + rightNumber);
        }
        if (expr->text == "-") {
            return makeNumberValue(leftNumber - rightNumber);
        }
        if (expr->text == "*") {
            return makeNumberValue(leftNumber * rightNumber);
        }
        if (expr->text == "/") {
            return makeNumberValue(leftNumber / rightNumber);
        }
        if (expr->text == "<") {
            return makeNumberValue(leftNumber < rightNumber ? 1. : 0.);
        }
        if (expr->text == "<=") {
            return makeNumberValue(leftNumber <= rightNumber ? 1. : 0.);
        }
        if (expr->text == ">") {
            return makeNumberValue(leftNumber > rightNumber ? 1. : 0.);
        }
        if (expr->text == ">=") {
            return makeNumberValue(leftNumber >= rightNumber ? 1. : 0.);
        }
        if (expr->text == "==") {
            return makeNumberValue(leftNumber == rightNumber ? 1. : 0.);
        }
        if (expr->text == "!=") {
            return makeNumberValue(leftNumber != rightNumber ? 1. : 0.);
        }
        throw runtime_error("Unsupported binary operator: " + expr->text);
    }
    if (expr->kind == ExprKind::Call) {
        return evalCall(expr, context);
    }
    if (expr->kind == ExprKind::Index) {
        const RuntimeCollection* collection = toCollection(evalExpression(expr->lhs, context));
        const int index = static_cast<int>(llround(evalNumber(expr->rhs, context)));
        if (index < 0 || index >= static_cast<int>(collection->objects.size())) {
            throw runtime_error("Collection index out of range in expression");
        }
        return makeObjectValue(collection, &collection->objects[index]);
    }
    if (expr->kind == ExprKind::Member) {
        const Value base = evalExpression(expr->lhs, context);
        if (base.kind != Value::Kind::ObjectRef || !base.collection || !base.object) {
            throw runtime_error("Member access requires an object in expression");
        }
        return makeNumberValue(getObjectField(*base.collection, *base.object, expr->text, def));
    }

    throw runtime_error("Unsupported expression kind");
}

bool evaluateCondition(const ExprPtr& expr, const EvalContext& context) {
    return truthy(evalExpression(expr, context));
}

RuntimeCollection applySelection(const RuntimeCollection& source,
                                 const ExprPtr& expr,
                                 const EvalContext& baseContext) {
    RuntimeCollection out;
    out.name = source.name;
    out.schema = source.schema;
    out.objects.reserve(source.objects.size());

    for (const auto& object : source.objects) {
        EvalContext context = baseContext;
        context.currentCollection = &source;
        context.currentObject = &object;
        if (evaluateCondition(expr, context)) {
            out.objects.push_back(object);
        }
    }
    return out;
}

RuntimeCollection applyDeduplication(const RuntimeCollection& source,
                                     const RuntimeCollection& reference,
                                     const ExprPtr& expr,
                                     const EvalContext& baseContext) {
    RuntimeCollection out;
    out.name = source.name;
    out.schema = source.schema;
    out.objects.reserve(source.objects.size());

    for (const auto& object : source.objects) {
        bool duplicate = false;
        for (const auto& referenceObject : reference.objects) {
            EvalContext context = baseContext;
            context.currentCollection = &source;
            context.currentObject = &object;
            context.otherCollection = &reference;
            context.otherObject = &referenceObject;
            if (evaluateCondition(expr, context)) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            out.objects.push_back(object);
        }
    }

    return out;
}

void sortCollection(RuntimeCollection& collection,
                    const SortRule& rule,
                    const EvalContext& baseContext) {
    if (!rule.expr) {
        return;
    }

    stable_sort(collection.objects.begin(), collection.objects.end(),
                [&](const RuntimeObject& lhs, const RuntimeObject& rhs) {
                    EvalContext leftContext = baseContext;
                    leftContext.currentCollection = &collection;
                    leftContext.currentObject = &lhs;
                    EvalContext rightContext = baseContext;
                    rightContext.currentCollection = &collection;
                    rightContext.currentObject = &rhs;
                    const long double leftValue = evalNumber(rule.expr, leftContext);
                    const long double rightValue = evalNumber(rule.expr, rightContext);
                    if (leftValue == rightValue) {
                        return false;
                    }
                    return rule.descending ? (leftValue > rightValue) : (leftValue < rightValue);
                });
}

const RuntimeCollection& buildRuntimeCollection(const string& name,
                                                const SelectionConfig& selectionConfig,
                                                const unordered_map<string, RuntimeCollection>& inputCollections,
                                                unordered_map<string, RuntimeCollection>& builtCollections,
                                                unordered_set<string>& activeCollections,
                                                const unordered_map<string, long double>& baseVars,
                                                const unordered_map<string, const ScalarInputConfig*>& rawScalars) {
    const auto builtIt = builtCollections.find(name);
    if (builtIt != builtCollections.end()) {
        return builtIt->second;
    }

    if (activeCollections.count(name)) {
        throw runtime_error("Collection dependency cycle detected at: " + name);
    }

    const auto configIt = selectionConfig.collections.find(name);
    if (configIt == selectionConfig.collections.end()) {
        throw runtime_error("Unknown runtime collection: " + name);
    }

    activeCollections.insert(name);
    const RuntimeCollectionConfig& config = configIt->second;

    RuntimeCollection current;
    if (!config.source.empty()) {
        const auto inputIt = inputCollections.find(config.source);
        if (inputIt == inputCollections.end()) {
            throw runtime_error("Unknown input collection source: " + config.source);
        }
        current = inputIt->second;
        current.name = config.name;
    } else if (!config.merge.empty()) {
        vector<const RuntimeCollection*> sources;
        sources.reserve(config.merge.size());
        for (const auto& childName : config.merge) {
            sources.push_back(&buildRuntimeCollection(childName, selectionConfig, inputCollections, builtCollections, activeCollections, baseVars, rawScalars));
        }
        current = mergeCollections(config.name, sources);
    } else {
        throw runtime_error("Runtime collection must define source or merge: " + config.name);
    }

    EvalContext context;
    context.vars = &baseVars;
    context.collections = &builtCollections;
    context.inputCollections = &inputCollections;
    context.rawScalars = &rawScalars;

    if (config.selectionExpr) {
        current = applySelection(current, config.selectionExpr, context);
    }

    if (config.dedupExpr && !config.dedupCollection.empty()) {
        const RuntimeCollection& reference = buildRuntimeCollection(config.dedupCollection, selectionConfig, inputCollections, builtCollections, activeCollections, baseVars, rawScalars);
        current = applyDeduplication(current, reference, config.dedupExpr, context);
    }

    if (config.sortRule.expr) {
        sortCollection(current, config.sortRule, context);
    }

    activeCollections.erase(name);
    builtCollections[name] = std::move(current);
    return builtCollections.at(name);
}

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

bool matchesRule(const string& sample, const SampleRuleConfig& rule) {
    return sample == rule.name;
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

Long64_t outputSizeLimitBytes(double maxOutputFileSizeGB) {
    if (maxOutputFileSizeGB <= 0.) {
        return 0;
    }
    constexpr long double kBytesPerGiB = 1024.0L * 1024.0L * 1024.0L;
    return static_cast<Long64_t>(maxOutputFileSizeGB * kBytesPerGiB);
}

Long64_t estimateTreeBytes(const TTree* tree) {
    Long64_t bytes = tree->GetZipBytes();
    if (bytes <= 0) {
        bytes = tree->GetTotBytes();
    }
    if (bytes <= 0) {
        bytes = max<Long64_t>(tree->GetEntries(), 1);
    }
    return bytes;
}

fs::path makeSplitOutputPath(const fs::path& basePath, size_t index) {
    const string stem = basePath.stem().string();
    const string extension = basePath.has_extension() ? basePath.extension().string() : ".root";
    return basePath.parent_path() / (stem + "_" + to_string(index) + extension);
}

fs::path makeBatchTempOutputDir(const AppConfig& appConfig, const SampleMeta& sampleMeta) {
    return fs::path(appConfig.outputRoot) / (sampleMeta.sampleGroup() + "_tmp");
}

fs::path makeBatchTempOutputPath(const AppConfig& appConfig,
                                 const SampleMeta& sampleMeta,
                                 size_t batchIndex) {
    return makeBatchTempOutputDir(appConfig, sampleMeta) /
           (sampleMeta.sample + "_" + to_string(batchIndex) + ".root");
}

fs::path makeBatchRawEntriesPath(const fs::path& batchOutputPath) {
    return fs::path(batchOutputPath.string() + ".raw_entries");
}

void writeBatchRawEntries(const fs::path& batchOutputPath, Long64_t rawEntries) {
    const fs::path path = makeBatchRawEntriesPath(batchOutputPath);
    ofstream fout(path);
    if (!fout) {
        throw runtime_error("Cannot write batch raw_entries file: " + path.string());
    }
    fout << rawEntries << '\n';
    fout.close();
    if (!fout) {
        throw runtime_error("Failed writing batch raw_entries file: " + path.string());
    }
}

Long64_t readBatchRawEntries(const fs::path& batchOutputPath) {
    const fs::path path = makeBatchRawEntriesPath(batchOutputPath);
    ifstream fin(path);
    if (!fin) {
        throw runtime_error("Missing batch raw_entries file: " + path.string());
    }
    Long64_t rawEntries = 0;
    fin >> rawEntries;
    if (!fin || rawEntries < 0) {
        throw runtime_error("Invalid batch raw_entries file: " + path.string());
    }
    return rawEntries;
}

bool validateBatchTempOutput(const fs::path& batchOutputPath,
                             const vector<TreeConfig>& treeConfigs,
                             Long64_t& rawEntries,
                             string& reason) {
    const fs::path rawEntriesPath = makeBatchRawEntriesPath(batchOutputPath);
    if (!fs::exists(batchOutputPath)) {
        reason = "missing ROOT output";
        return false;
    }
    if (!fs::exists(rawEntriesPath)) {
        reason = "missing raw_entries";
        return false;
    }

    try {
        rawEntries = readBatchRawEntries(batchOutputPath);
    } catch (const exception& ex) {
        reason = ex.what();
        return false;
    }

    unique_ptr<TFile> file(TFile::Open(batchOutputPath.string().c_str(), "READ"));
    if (!file || file->IsZombie()) {
        reason = "cannot open ROOT output";
        return false;
    }

    for (const auto& treeConfig : treeConfigs) {
        TTree* tree = dynamic_cast<TTree*>(file->Get(treeConfig.name.c_str()));
        if (tree == nullptr) {
            reason = "missing tree " + treeConfig.name;
            return false;
        }
        (void)tree->GetEntries();
    }

    reason.clear();
    return true;
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
    for (const auto& rule : appConfig.sampleRules) {
        if (!matchesRule(sample, rule)) {
            continue;
        }
        SampleMeta meta;
        meta.sample = sample;
        meta.sampleId = rule.sampleId;
        meta.isMC = rule.isMC;
        meta.isSignal = rule.isSignal;
        meta.hasTheoryWeights = rule.hasTheoryWeights;
        meta.xsection = rule.xsection;
        meta.lumi = rule.lumi;

        if (rule.paths.empty()) {
            throw runtime_error("No input path configured for sample: " + sample);
        }

        unordered_map<string, string> templateValues;
        templateValues["sample"] = meta.sample;
        templateValues["sample_group"] = meta.sampleGroup();
        templateValues["output_root"] = appConfig.outputRoot;

        for (const auto& pathTemplate : rule.paths) {
            const string resolvedPath = applyTemplate(pathTemplate, templateValues);
            if (find(meta.inputPaths.begin(), meta.inputPaths.end(), resolvedPath) == meta.inputPaths.end()) {
                meta.inputPaths.push_back(resolvedPath);
            }
        }
        if (meta.inputPaths.empty()) {
            throw runtime_error("No input path configured for sample: " + sample);
        }
        meta.outputFileName = normalizeOutputPath(
            appConfig, applyTemplate(appConfig.outputPattern, templateValues));
        return meta;
    }

    throw runtime_error("No sample named '" + sample + "' found in " + appConfig.sampleConfigPath);
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

bool parseNonNegativeIndex(const string& text, size_t& value) {
    if (text.empty()) {
        return false;
    }
    for (char c : text) {
        if (!isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    try {
        value = static_cast<size_t>(stoull(text));
    } catch (const exception&) {
        return false;
    }
    return true;
}

BatchRequest resolveBatchRequest(int argc, char** argv) {
    BatchRequest request;
    if (argc <= 2) {
        return request;
    }
    if (argc > 3) {
        throw runtime_error("Usage: convert_branch <sample> [batch_index|--batch-count|--merge-successful-batches]");
    }

    const string arg = argv[2] == nullptr ? "" : argv[2];
    if (arg == "--batch-count") {
        request.printBatchCount = true;
        return request;
    }
    if (arg == "--merge-successful-batches") {
        request.mergeSuccessfulBatches = true;
        return request;
    }

    size_t batchIndex = 0;
    if (!parseNonNegativeIndex(arg, batchIndex)) {
        throw runtime_error("Invalid batch argument '" + arg +
                            "'. Use a non-negative batch index, --batch-count, or --merge-successful-batches.");
    }
    request.singleBatch = true;
    request.batchIndex = batchIndex;
    return request;
}

vector<size_t> parseSuccessfulBatchIndicesFromEnv(size_t nBatches, bool& restrictedToEnv) {
    restrictedToEnv = false;
    const char* envValue = getenv(kSuccessfulBatchesEnvVar);
    if (envValue == nullptr) {
        return {};
    }

    restrictedToEnv = true;
    vector<size_t> indices;
    string text(envValue);
    size_t begin = 0;
    while (begin <= text.size()) {
        const size_t comma = text.find(',', begin);
        string token = text.substr(begin,
                                   comma == string::npos ? string::npos : comma - begin);
        token.erase(remove_if(token.begin(), token.end(),
                              [](unsigned char ch) { return isspace(ch); }),
                    token.end());
        if (!token.empty()) {
            size_t batchIndex = 0;
            if (!parseNonNegativeIndex(token, batchIndex) || batchIndex >= nBatches) {
                throw runtime_error(string("Invalid ") + kSuccessfulBatchesEnvVar +
                                    " entry: " + token);
            }
            indices.push_back(batchIndex);
        }
        if (comma == string::npos) {
            break;
        }
        begin = comma + 1;
    }

    sort(indices.begin(), indices.end());
    indices.erase(unique(indices.begin(), indices.end()), indices.end());
    return indices;
}

vector<size_t> resolveBatchIndicesForFinalMerge(size_t nBatches,
                                                const BatchRequest& batchRequest) {
    bool restrictedToEnv = false;
    vector<size_t> indices = parseSuccessfulBatchIndicesFromEnv(nBatches, restrictedToEnv);
    if (!restrictedToEnv) {
        indices.reserve(nBatches);
        for (size_t batchIndex = 0; batchIndex < nBatches; ++batchIndex) {
            indices.push_back(batchIndex);
        }
        return indices;
    }

    if (batchRequest.singleBatch && batchRequest.batchIndex + 1 == nBatches) {
        indices.push_back(batchRequest.batchIndex);
        sort(indices.begin(), indices.end());
        indices.erase(unique(indices.begin(), indices.end()), indices.end());
    }
    return indices;
}

bool finalMergeDeferredByEnv() {
    const char* envValue = getenv(kDeferFinalMergeEnvVar);
    return envValue != nullptr && string(envValue) != "0";
}

string resolvePileupWeightPath(const AppConfig& appConfig, const SampleMeta& sampleMeta) {
    unordered_map<string, string> templateValues;
    templateValues["sample"] = sampleMeta.sample;
    templateValues["sample_group"] = sampleMeta.sampleGroup();
    templateValues["output_root"] = appConfig.outputRoot;
    return normalizeOutputPath(appConfig, applyTemplate(appConfig.puWeightPathPattern, templateValues));
}

size_t countOutputGroupBranches(const vector<OutputScalarConfig>& configs, bool isMC) {
    size_t count = 0;
    for (const auto& config : configs) {
        if (config.onlyMC && !isMC) {
            continue;
        }
        count += config.collection.empty() ? 1u : static_cast<size_t>(config.slots);
    }
    return count;
}

void appendOutputBranch(OutputTreeState& treeState,
                        const OutputScalarConfig& config,
                        const string& branchName,
                        int slotIndex) {
    treeState.branches.emplace_back();
    OutputBranchRuntime& branch = treeState.branches.back();
    branch.name = branchName;
    branch.type = config.type;
    branch.sourceConfig = &config;
    branch.slotIndex = slotIndex;

    const string leafList = branchName + "/" + string(1, outputLeafCode(config.type));
    if (config.type == DataType::Float) {
        treeState.tree->Branch(branchName.c_str(), &branch.floatValue, leafList.c_str());
    } else if (config.type == DataType::Int) {
        treeState.tree->Branch(branchName.c_str(), &branch.intValue, leafList.c_str());
    } else if (config.type == DataType::UInt) {
        treeState.tree->Branch(branchName.c_str(), &branch.uintValue, leafList.c_str());
    } else if (config.type == DataType::Bool) {
        treeState.tree->Branch(branchName.c_str(), &branch.boolValue, leafList.c_str());
    } else if (config.type == DataType::Long64) {
        treeState.tree->Branch(branchName.c_str(), &branch.long64Value, leafList.c_str());
    } else if (config.type == DataType::ULong64) {
        treeState.tree->Branch(branchName.c_str(), &branch.ulong64Value, leafList.c_str());
    } else {
        throw runtime_error("Unsupported output branch type for booking: " + branchName);
    }
    treeState.branchIndexByName[branchName] = treeState.branches.size() - 1;
}

void bookOutputGroup(OutputTreeState& treeState,
                     const vector<OutputScalarConfig>& configs,
                     bool isMC) {
    for (const auto& config : configs) {
        if (config.onlyMC && !isMC) {
            continue;
        }
        if (!config.collection.empty()) {
            for (int slot = 0; slot < config.slots; ++slot) {
                appendOutputBranch(treeState, config, config.name + "_" + to_string(slot + 1), slot);
            }
        } else {
            appendOutputBranch(treeState, config, config.name, -1);
        }
    }
}

void bookTreeBranches(OutputTreeState& treeState, bool isMC, TDirectory* directory) {
    const size_t totalBranches = countOutputGroupBranches(treeState.config.regularScalars, isMC) +
                                 countOutputGroupBranches(treeState.config.extremaScalars, isMC);
    if (directory != nullptr) {
        directory->cd();
    }
    treeState.tree = new TTree(treeState.config.name.c_str(), treeState.config.title.c_str());
    if (directory != nullptr) {
        treeState.tree->SetDirectory(directory);
    }
    treeState.tree->SetAutoSave(64LL * 1024LL * 1024LL);
    treeState.branches.reserve(totalBranches);
    treeState.branchIndexByName.reserve(totalBranches);
    bookOutputGroup(treeState, treeState.config.regularScalars, isMC);
    bookOutputGroup(treeState, treeState.config.extremaScalars, isMC);
}

vector<OutputTreeState> makeOutputTrees(const BranchConfig& branchConfig, bool isMC, TDirectory* directory) {
    vector<OutputTreeState> outputTrees;
    outputTrees.reserve(branchConfig.trees.size());
    for (const auto& treeConfig : branchConfig.trees) {
        outputTrees.emplace_back();
        outputTrees.back().config = treeConfig;
        bookTreeBranches(outputTrees.back(), isMC, directory);
    }
    return outputTrees;
}

void destroyOutputTrees(vector<OutputTreeState>& outputTrees, bool deleteTrees) {
    for (auto& treeState : outputTrees) {
        if (deleteTrees && treeState.tree != nullptr) {
            treeState.tree->SetDirectory(nullptr);
            delete treeState.tree;
        }
        treeState.tree = nullptr;
    }
}

string makeThreadTempFilePath(const fs::path& tempDir,
                              const string& sample,
                              size_t batchIndex,
                              int threadIndex) {
    const string fileName = "convert_" + sample + "_batch_" + to_string(batchIndex) +
                            "_" + to_string(static_cast<long long>(getpid())) +
                            "_" + to_string(threadIndex) + ".root";
    // Prefer local node scratch over NFS to avoid ESTALE errors during auto-save
    // flushes on long-running jobs. Fall back to tempDir (NFS) if $TMPDIR is unset.
    const char* scratch = getenv("TMPDIR");
    if (scratch != nullptr && *scratch != '\0' && fs::is_directory(scratch)) {
        return (fs::path(scratch) / fileName).string();
    }
    return (tempDir / fileName).string();
}

void initializeThreadResult(ThreadConvertResult& result,
                            const BranchConfig& branchConfig,
                            bool isMC,
                            const string& sample,
                            const fs::path& tempDir,
                            size_t batchIndex,
                            int threadIndex) {
    result.tempFilePath = makeThreadTempFilePath(tempDir, sample, batchIndex, threadIndex);
    result.tempFile = TFile::Open(result.tempFilePath.c_str(), "RECREATE");
    if (!result.tempFile || result.tempFile->IsZombie()) {
        throw runtime_error("Error opening temporary output file " + result.tempFilePath);
    }
    result.outputTrees = makeOutputTrees(branchConfig, isMC, result.tempFile);
}

void cleanupThreadResult(ThreadConvertResult& result) {
    destroyOutputTrees(result.outputTrees, result.tempFile == nullptr);
    if (result.tempFile != nullptr) {
        result.tempFile->Close();
        delete result.tempFile;
        result.tempFile = nullptr;
    }
    if (!result.tempFilePath.empty()) {
        std::error_code ec;
        fs::remove(result.tempFilePath, ec);
        result.tempFilePath.clear();
    }
}

void resetBranchValue(OutputBranchRuntime& branch) {
    if (branch.type == DataType::Float) {
        branch.floatValue = def;
    } else if (branch.type == DataType::Int) {
        branch.intValue = 0;
    } else if (branch.type == DataType::UInt) {
        branch.uintValue = 0;
    } else if (branch.type == DataType::Bool) {
        branch.boolValue = false;
    } else if (branch.type == DataType::Long64) {
        branch.long64Value = 0;
    } else if (branch.type == DataType::ULong64) {
        branch.ulong64Value = 0;
    }
}

void resetTreeValues(OutputTreeState& treeState) {
    for (auto& branch : treeState.branches) {
        resetBranchValue(branch);
    }
}

bool isRawScalarIdentity(const ExprPtr& expr,
                         const unordered_map<string, const ScalarInputConfig*>& rawScalars,
                         const ScalarInputConfig*& scalar) {
    if (!expr || expr->kind != ExprKind::Identifier) {
        return false;
    }
    const auto it = rawScalars.find(expr->text);
    if (it == rawScalars.end()) {
        return false;
    }
    scalar = it->second;
    return true;
}

void assignExactScalar(OutputBranchRuntime& branch, const ScalarInputConfig& scalar) {
    if (branch.type == DataType::Float) {
        branch.floatValue = static_cast<Float_t>(scalar.numericValue());
    } else if (branch.type == DataType::Int) {
        branch.intValue = static_cast<Int_t>(scalar.numericValue());
    } else if (branch.type == DataType::UInt) {
        branch.uintValue = static_cast<UInt_t>(scalar.numericValue());
    } else if (branch.type == DataType::Bool) {
        branch.boolValue = (scalar.numericValue() != 0.);
    } else if (branch.type == DataType::Long64) {
        branch.long64Value = (scalar.type == DataType::Long64) ? scalar.long64Value
                                                               : static_cast<Long64_t>(scalar.numericValue());
    } else if (branch.type == DataType::ULong64) {
        branch.ulong64Value = (scalar.type == DataType::ULong64) ? scalar.ulong64Value
                                                                 : static_cast<ULong64_t>(scalar.numericValue());
    } else {
        throw runtime_error("Unsupported exact scalar output assignment");
    }
}

void assignNumericValue(OutputBranchRuntime& branch, long double value) {
    if (branch.type == DataType::Float) {
        branch.floatValue = static_cast<Float_t>(value);
    } else if (branch.type == DataType::Int) {
        branch.intValue = static_cast<Int_t>(value);
    } else if (branch.type == DataType::UInt) {
        branch.uintValue = static_cast<UInt_t>(value);
    } else if (branch.type == DataType::Bool) {
        branch.boolValue = (value != 0.);
    } else if (branch.type == DataType::Long64) {
        branch.long64Value = static_cast<Long64_t>(value);
    } else if (branch.type == DataType::ULong64) {
        branch.ulong64Value = static_cast<ULong64_t>(value);
    } else {
        throw runtime_error("Unsupported numeric output assignment");
    }
}

void fillOutputGroup(const vector<OutputScalarConfig>& configs,
                     OutputTreeState& treeState,
                     unordered_map<string, long double>& vars,
                     const unordered_map<string, RuntimeCollection>& collections,
                     const unordered_map<string, RuntimeCollection>& inputCollections,
                     const unordered_map<string, const ScalarInputConfig*>& rawScalars,
                     bool isMC) {
    for (const auto& config : configs) {
        if (config.onlyMC && !isMC) {
            continue;
        }

        if (config.collection.empty()) {
            EvalContext context;
            context.vars = &vars;
            context.collections = &collections;
            context.inputCollections = &inputCollections;
            context.rawScalars = &rawScalars;

            const auto branchIt = treeState.branchIndexByName.find(config.name);
            if (branchIt == treeState.branchIndexByName.end()) {
                continue;
            }

            OutputBranchRuntime& branch = treeState.branches[branchIt->second];
            const ScalarInputConfig* scalar = nullptr;
            if (isRawScalarIdentity(config.formula, rawScalars, scalar) && scalar != nullptr) {
                assignExactScalar(branch, *scalar);
                vars[config.name] = scalar->numericValue();
            } else {
                const long double value = evalNumber(config.formula, context);
                assignNumericValue(branch, value);
                vars[config.name] = value;
            }
            continue;
        }

        const auto collectionIt = collections.find(config.collection);
        if (collectionIt == collections.end()) {
            throw runtime_error("Unknown output collection: " + config.collection);
        }
        const RuntimeCollection& collection = collectionIt->second;
        for (int slot = 0; slot < config.slots; ++slot) {
            if (slot >= static_cast<int>(collection.objects.size())) {
                continue;
            }
            EvalContext context;
            context.vars = &vars;
            context.collections = &collections;
            context.inputCollections = &inputCollections;
            context.rawScalars = &rawScalars;
            context.currentCollection = &collection;
            context.currentObject = &collection.objects[slot];

            const string branchName = config.name + "_" + to_string(slot + 1);
            const auto branchIt = treeState.branchIndexByName.find(branchName);
            if (branchIt == treeState.branchIndexByName.end()) {
                continue;
            }
            OutputBranchRuntime& branch = treeState.branches[branchIt->second];
            assignNumericValue(branch, evalNumber(config.formula, context));
        }
    }
}

void fillOutputTree(OutputTreeState& treeState,
                    const unordered_map<string, RuntimeCollection>& collections,
                    const unordered_map<string, RuntimeCollection>& inputCollections,
                    const unordered_map<string, long double>& baseVars,
                    const unordered_map<string, const ScalarInputConfig*>& rawScalars,
                    bool isMC) {
    resetTreeValues(treeState);

    unordered_map<string, long double> vars = baseVars;
    fillOutputGroup(treeState.config.regularScalars, treeState, vars, collections, inputCollections, rawScalars, isMC);
    fillOutputGroup(treeState.config.extremaScalars, treeState, vars, collections, inputCollections, rawScalars, isMC);

    treeState.tree->Fill();
}

unordered_map<string, string> buildScalarBranchMap(const BranchConfig& branchConfig) {
    unordered_map<string, string> branches;
    branches.reserve(branchConfig.scalars.size());
    for (const auto& scalar : branchConfig.scalars) {
        branches[scalar.name] = scalar.branch;
    }
    return branches;
}

void configureActiveBranches(TTree* tree, const BranchConfig& branchConfig, bool isMC) {
    tree->SetBranchStatus("*", 0);
    unordered_set<string> activeBranches;
    activeBranches.reserve(branchConfig.scalars.size() + branchConfig.collections.size() * 8);
    const auto scalarBranchMap = buildScalarBranchMap(branchConfig);

    for (const auto& scalar : branchConfig.scalars) {
        if (scalar.onlyMC && !isMC) {
            continue;
        }
        activeBranches.insert(scalar.branch);
    }

    for (const auto& collection : branchConfig.collections) {
        const auto sizeIt = scalarBranchMap.find(collection.sizeName);
        activeBranches.insert(sizeIt != scalarBranchMap.end() ? sizeIt->second : collection.sizeName);
        for (const auto& field : collection.fields) {
            if (field.onlyMC && !isMC) {
                continue;
            }
            activeBranches.insert(field.branch);
        }
    }

    for (const auto& branch : activeBranches) {
        tree->SetBranchStatus(branch.c_str(), 1);
    }
    tree->SetCacheSize(50 * 1024 * 1024);
    for (const auto& branch : activeBranches) {
        tree->AddBranchToCache(branch.c_str(), true);
    }
}

unordered_map<string, const ScalarInputConfig*> bindInputBranches(TTree* tree,
                                                                  BranchConfig& branchConfig,
                                                                  bool isMC) {
    unordered_map<string, const ScalarInputConfig*> rawScalarByName;
    rawScalarByName.reserve(branchConfig.scalars.size());
    for (auto& scalar : branchConfig.scalars) {
        scalar.bind(tree, isMC);
        rawScalarByName[scalar.name] = &scalar;
    }
    for (auto& collection : branchConfig.collections) {
        for (auto& field : collection.fields) {
            field.bind(tree, isMC);
        }
    }
    return rawScalarByName;
}

void ensureCollectionBufferCapacities(TTree* tree, BranchConfig& branchConfig, bool isMC) {
    const auto scalarBranchMap = buildScalarBranchMap(branchConfig);
    for (auto& collection : branchConfig.collections) {
        const auto sizeIt = scalarBranchMap.find(collection.sizeName);
        const string sizeBranch = (sizeIt != scalarBranchMap.end()) ? sizeIt->second : collection.sizeName;
        Long64_t observedMax = static_cast<Long64_t>(llround(tree->GetMaximum(sizeBranch.c_str())));
        if (observedMax < 0) {
            observedMax = 0;
        }
        const int bindSize = max(collection.maxSize, static_cast<int>(observedMax));
        for (auto& field : collection.fields) {
            if (field.onlyMC && !isMC) {
                continue;
            }
            field.ensureBufferSize(bindSize);
        }
    }
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

// Persist each thread's filled trees to its temp ROOT file and close the file.
// The temp file on disk is kept so the streaming writer below can re-open it
// read-only; cleanupThreadResult() unlinks the file later.
vector<string> finalizeThreadTempFiles(vector<ThreadConvertResult>& threadResults) {
    vector<string> paths;
    paths.reserve(threadResults.size());
    for (auto& result : threadResults) {
        if (result.tempFile == nullptr) {
            continue;
        }
        result.tempFile->cd();
        for (auto& treeState : result.outputTrees) {
            if (treeState.tree != nullptr) {
                treeState.tree->Write("", TObject::kOverwrite);
            }
        }
        result.tempFile->Close();
        delete result.tempFile;
        result.tempFile = nullptr;
        // TFile::Close() deletes the TTree objects owned by the file, so the
        // cached pointers in outputTrees are now dangling — null them out.
        for (auto& treeState : result.outputTrees) {
            treeState.tree = nullptr;
        }
        if (!result.tempFilePath.empty()) {
            paths.push_back(result.tempFilePath);
        }
    }
    return paths;
}

// Stream the combined per-thread trees into one or more output files. Each
// chunk is a fresh TFile with both fat2/fat3 trees cloned from a TChain over
// the thread temp files, so ROOT keeps the clone's branch addresses synced as
// the chain advances across file boundaries. Basket size is fixed and
// auto-flush is disabled so that OptimizeBaskets cannot inflate a basket past
// ROOT's 1GB TBufferFile serialization limit (the failure mode observed on the
// large QCD samples when TTree::MergeTrees was used instead).
vector<string> writeOutputFilesStreaming(const fs::path& baseOutputPath,
                                         const vector<string>& threadTempPaths,
                                         const vector<TreeConfig>& treeConfigs,
                                         Long64_t maxOutputBytes) {
    const size_t nTrees = treeConfigs.size();

    // Sum per-tree entry counts and estimated bytes across all thread temp files.
    vector<Long64_t> treeEntries(nTrees, 0);
    Long64_t estimatedTotalBytes = 0;
    for (const auto& p : threadTempPaths) {
        unique_ptr<TFile> f(TFile::Open(p.c_str(), "READ"));
        if (!f || f->IsZombie()) {
            continue;
        }
        for (size_t i = 0; i < nTrees; ++i) {
            TTree* t = dynamic_cast<TTree*>(f->Get(treeConfigs[i].name.c_str()));
            if (t == nullptr) {
                continue;
            }
            treeEntries[i] += t->GetEntries();
            estimatedTotalBytes += estimateTreeBytes(t);
        }
    }

    // Decide how many chunks to write. Stay safely under the configured GB
    // limit (~10% margin) so per-file on-disk size comes close to but does
    // not exceed the limit.
    size_t nChunks = 1;
    if (maxOutputBytes > 0 && estimatedTotalBytes > maxOutputBytes) {
        const Long64_t target = max<Long64_t>(1, static_cast<Long64_t>(maxOutputBytes * 0.9));
        nChunks = static_cast<size_t>((estimatedTotalBytes + target - 1) / target);
        if (nChunks < 2) {
            nChunks = 2;
        }
    }
    const bool split = (nChunks > 1);

    vector<string> writtenFiles;
    writtenFiles.reserve(nChunks);

    for (size_t chunkIdx = 0; chunkIdx < nChunks; ++chunkIdx) {
        const fs::path outPath = split
            ? makeSplitOutputPath(baseOutputPath, chunkIdx)
            : baseOutputPath;
        unique_ptr<TFile> outFile(TFile::Open(outPath.c_str(), "RECREATE"));
        if (!outFile || outFile->IsZombie()) {
            throw runtime_error("Error opening output file " + outPath.string());
        }

        for (size_t i = 0; i < nTrees; ++i) {
            const Long64_t totalE = treeEntries[i];
            const Long64_t first = (totalE * static_cast<Long64_t>(chunkIdx)) /
                                   static_cast<Long64_t>(nChunks);
            const Long64_t last = (totalE * static_cast<Long64_t>(chunkIdx + 1)) /
                                  static_cast<Long64_t>(nChunks);
            const Long64_t expectedEntries = last - first;

            auto configureOutTree = [&](TTree& outTree) {
                outTree.SetDirectory(outFile.get());
                outTree.SetNameTitle(treeConfigs[i].name.c_str(),
                                     treeConfigs[i].title.c_str());
                // Explicit small basket + disabled auto-flush prevents ROOT's
                // OptimizeBaskets from growing a single basket's uncompressed size
                // past its 1GB TBufferFile serialization cap on highly-compressible
                // branches (e.g. constant sample_ID / is_MC flags).
                outTree.SetBasketSize("*", 32000);
                outTree.SetAutoFlush(0);
                outTree.SetAutoSave(0);
            };

            outFile->cd();
            TTree* outTree = nullptr;
            if (expectedEntries > 0) {
                TChain chain(treeConfigs[i].name.c_str());
                for (const auto& p : threadTempPaths) {
                    chain.Add(p.c_str());
                }
                if (chain.LoadTree(first) < 0) {
                    throw runtime_error("Failed to load entry " + to_string(first) +
                                        " for tree " + treeConfigs[i].name);
                }
                outTree = chain.CloneTree(0);
                if (outTree == nullptr) {
                    throw runtime_error("Failed to clone output tree from chain for " +
                                        treeConfigs[i].name);
                }
                configureOutTree(*outTree);
                Long64_t written = 0;
                for (Long64_t e = first; e < last; ++e) {
                    if (chain.GetEntry(e) <= 0) {
                        throw runtime_error("Failed to read entry " + to_string(e) +
                                            " for tree " + treeConfigs[i].name);
                    }
                    outTree->Fill();
                    ++written;
                }
                if (written != expectedEntries) {
                    throw runtime_error("Output chunk entry count mismatch for tree " +
                                        treeConfigs[i].name + ": expected " +
                                        to_string(expectedEntries) + ", got " +
                                        to_string(written));
                }
                outFile->cd();
                outTree->Write("", TObject::kOverwrite);
                chain.ResetBranchAddresses();
                outTree->ResetBranchAddresses();
                continue;
            }

            // No entries land in this chunk for this tree, but still create an
            // empty tree so every output file keeps the full fat2/fat3 layout.
            unique_ptr<TFile> structureFile;
            TTree* structureSrc = nullptr;
            for (const auto& p : threadTempPaths) {
                unique_ptr<TFile> f(TFile::Open(p.c_str(), "READ"));
                if (!f || f->IsZombie()) {
                    continue;
                }
                TTree* t = dynamic_cast<TTree*>(f->Get(treeConfigs[i].name.c_str()));
                if (t != nullptr) {
                    structureFile = std::move(f);
                    structureSrc = t;
                    break;
                }
            }
            if (structureSrc != nullptr) {
                outTree = structureSrc->CloneTree(0);
            }
            if (outTree == nullptr) {
                outTree = new TTree(treeConfigs[i].name.c_str(),
                                    treeConfigs[i].title.c_str());
            }
            configureOutTree(*outTree);

            outFile->cd();
            outTree->Write("", TObject::kOverwrite);
            outTree->ResetBranchAddresses();
        }
        outFile->Close();
        writtenFiles.push_back(outPath.string());
    }

    return writtenFiles;
}

unique_ptr<TFile> openInputFileWithRetry(const string& inputFileName) {
    const bool remoteInput = startsWith(inputFileName, "root://");
    const int maxRetries = remoteInput ? kRemoteInputOpenRetries : 0;

    for (int retry = 0; retry <= maxRetries; ++retry) {
        unique_ptr<TFile> inputFile(TFile::Open(inputFileName.c_str(), "READ"));
        if (inputFile && !inputFile->IsZombie()) {
            return inputFile;
        }

        if (retry < maxRetries) {
            cerr << "Warning: failed to open remote input file " << inputFileName
                 << "; retry " << (retry + 1) << "/" << maxRetries
                 << " after " << kRemoteInputRetrySleepSeconds << " seconds" << endl;
            sleep(kRemoteInputRetrySleepSeconds);
        }
    }

    throw runtime_error("Error opening input file " + inputFileName);
}

Long64_t processInputFile(const string& inputFileName,
                          const AppConfig& appConfig,
                          const SelectionConfig& selectionConfig,
                          const SampleMeta& sampleMeta,
                          const vector<PileupBin>& pileupWeights,
                          const LumiMask* lumiMask,
                          BranchConfig& branchConfig,
                          vector<OutputTreeState>& outputTrees) {
    unique_ptr<TFile> inputFile;
    try {
        inputFile = openInputFileWithRetry(inputFileName);
    } catch (const runtime_error& ex) {
        throw SkippableFileError(ex.what());
    }

    TTree* tree = static_cast<TTree*>(inputFile->Get(appConfig.treeName.c_str()));
    if (!tree) {
        throw SkippableFileError("Tree " + appConfig.treeName + " not found in " + inputFileName);
    }

    const Long64_t nEntries = tree->GetEntries();
    if (nEntries == 0) {
        return 0;
    }

    configureActiveBranches(tree, branchConfig, sampleMeta.isMC);
    ensureCollectionBufferCapacities(tree, branchConfig, sampleMeta.isMC);
    unordered_map<string, const ScalarInputConfig*> rawScalarByName = bindInputBranches(tree, branchConfig, sampleMeta.isMC);

    TheoryWeightBufs theoryInBuf;
    if (sampleMeta.hasTheoryWeights) {
        activateTheoryInputBranches(tree, theoryInBuf);
    }

    const bool applyLumiMask = (!sampleMeta.isMC && lumiMask != nullptr);
    const ScalarInputConfig* runScalar = nullptr;
    const ScalarInputConfig* lumiScalar = nullptr;
    TBranch* runBranch = nullptr;
    TBranch* lumiBranch = nullptr;
    if (applyLumiMask) {
        const auto runIt = rawScalarByName.find("run");
        const auto lumiIt = rawScalarByName.find("luminosityBlock");
        if (runIt == rawScalarByName.end() || lumiIt == rawScalarByName.end()) {
            throw runtime_error("Data lumi mask requires input scalars 'run' and 'luminosityBlock'");
        }
        runScalar = runIt->second;
        lumiScalar = lumiIt->second;
        runBranch = tree->GetBranch(runScalar->branch.c_str());
        lumiBranch = tree->GetBranch(lumiScalar->branch.c_str());
        if (runBranch == nullptr || lumiBranch == nullptr) {
            throw runtime_error("Data lumi mask requires input branches '" +
                                runScalar->branch + "' and '" + lumiScalar->branch + "'");
        }
    }

    Long64_t rawEntries = applyLumiMask ? 0 : nEntries;
    for (Long64_t entry = 0; entry < nEntries; ++entry) {
        if (applyLumiMask) {
            if (runBranch->GetEntry(entry) < 0 || lumiBranch->GetEntry(entry) < 0) {
                throw runtime_error("Failed to read run/luminosityBlock for lumi mask");
            }
            const UInt_t runValue = static_cast<UInt_t>(runScalar->numericValue());
            const UInt_t lumiValue = static_cast<UInt_t>(lumiScalar->numericValue());
            if (!lumiMask->contains(runValue, lumiValue)) {
                continue;
            }
            ++rawEntries;
        }

        tree->GetEntry(entry);

        const TheoryWeightBufs* theoryBufsPtr = sampleMeta.hasTheoryWeights ? &theoryInBuf : nullptr;
        unordered_map<string, long double> baseVars = buildRawScalarValues(branchConfig, sampleMeta, &pileupWeights, theoryBufsPtr);

        EvalContext preContext;
        preContext.vars = &baseVars;
        preContext.rawScalars = &rawScalarByName;
        if (!evaluateCondition(selectionConfig.eventPreselection, preContext)) {
            continue;
        }

        unordered_map<string, RuntimeCollection> inputCollections;
        inputCollections.reserve(branchConfig.collections.size());
        for (const auto& inputConfig : branchConfig.collections) {
            inputCollections[inputConfig.name] = buildInputCollection(inputConfig, baseVars);
        }

        unordered_map<string, RuntimeCollection> runtimeCollections;
        runtimeCollections.reserve(selectionConfig.collectionOrder.size());
        unordered_set<string> activeCollections;
        activeCollections.reserve(selectionConfig.collectionOrder.size());
        for (const auto& name : selectionConfig.collectionOrder) {
            buildRuntimeCollection(name, selectionConfig, inputCollections, runtimeCollections, activeCollections, baseVars, rawScalarByName);
        }

        for (auto& treeState : outputTrees) {
            const auto cutIt = selectionConfig.treeSelections.find(treeState.config.selection);
            if (cutIt == selectionConfig.treeSelections.end()) {
                throw runtime_error("Missing tree selection: " + treeState.config.selection);
            }

            EvalContext treeContext;
            treeContext.vars = &baseVars;
            treeContext.collections = &runtimeCollections;
            treeContext.inputCollections = &inputCollections;
            treeContext.rawScalars = &rawScalarByName;
            if (!evaluateCondition(cutIt->second, treeContext)) {
                continue;
            }

            if (treeState.hasTheoryBranches) {
                copyTheoryWeights(theoryInBuf, treeState.theoryOutBuf);
            }
            fillOutputTree(treeState, runtimeCollections, inputCollections, baseVars, rawScalarByName, sampleMeta.isMC);
        }
    }
    return rawEntries;
}

vector<string> processInputBatchToTempFile(const vector<string>& batchInputFiles,
                                           size_t batchIndex,
                                           int threadCount,
                                           const fs::path& batchOutputPath,
                                           const AppConfig& appConfig,
                                           const SelectionConfig& selectionConfig,
                                           const SampleMeta& sampleMeta,
                                           const vector<PileupBin>& pileupWeights,
                                           const LumiMask* lumiMask,
                                           const BranchConfig& branchConfig,
                                           atomic<size_t>& processedFiles,
                                           size_t totalFiles,
                                           atomic<Long64_t>& batchRawEntries) {
    if (batchInputFiles.empty()) {
        throw runtime_error("Empty input batch for sample " + sampleMeta.sample);
    }

    const fs::path tempDir = batchOutputPath.parent_path();
    if (!tempDir.empty()) {
        fs::create_directories(tempDir);
    }

    vector<ThreadConvertResult> threadResults(threadCount);
    try {
        for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
            initializeThreadResult(threadResults[threadIndex],
                                   branchConfig,
                                   sampleMeta.isMC,
                                   sampleMeta.sample,
                                   tempDir,
                                   batchIndex,
                                   threadIndex);
        }
    } catch (const exception& ex) {
        for (auto& result : threadResults) {
            cleanupThreadResult(result);
        }
        throw runtime_error("Temporary output initialization error: " + string(ex.what()));
    }

    if (sampleMeta.hasTheoryWeights) {
        for (auto& result : threadResults) {
            for (auto& treeState : result.outputTrees) {
                setupTheoryOutputBranches(treeState);
            }
        }
    }

    vector<BranchConfig> threadConfigs(threadCount, branchConfig);
    atomic<bool> failed{false};
    vector<string> errors;

#pragma omp parallel num_threads(threadCount) if(threadCount > 1)
    {
        const int tid =
#ifdef _OPENMP
            omp_get_thread_num();
#else
            0;
#endif

#pragma omp for schedule(dynamic)
        for (int index = 0; index < static_cast<int>(batchInputFiles.size()); ++index) {
            if (failed.load()) {
                continue;
            }

            try {
                const Long64_t fileEntries = processInputFile(batchInputFiles[index],
                                                              appConfig,
                                                              selectionConfig,
                                                              sampleMeta,
                                                              pileupWeights,
                                                              lumiMask,
                                                              threadConfigs[tid],
                                                              threadResults[tid].outputTrees);
                batchRawEntries.fetch_add(fileEntries);
                const size_t done = processedFiles.fetch_add(1) + 1;
#pragma omp critical(convert_progress)
                printFileProgress(sampleMeta.sample, done, totalFiles);
            } catch (const SkippableFileError& ex) {
                const size_t done = processedFiles.fetch_add(1) + 1;
#pragma omp critical(convert_progress)
                {
                    cerr << "\nWarning: skipping " << batchInputFiles[index]
                         << ": " << ex.what() << '\n';
                    printFileProgress(sampleMeta.sample, done, totalFiles);
                }
            } catch (const exception& ex) {
                failed.store(true);
#pragma omp critical(convert_error)
                errors.push_back("Input ROOT file " + batchInputFiles[index] + ": " + ex.what());
            }
        }
    }

    if (!errors.empty()) {
        for (auto& result : threadResults) {
            cleanupThreadResult(result);
        }
        throw runtime_error(errors.front());
    }

    // Detect NFS stale-handle or other write failures before attempting to read
    // the temp files back. ROOT sets TFile::kWriteError when a flush/write fails.
    for (int threadIndex = 0; threadIndex < threadCount; ++threadIndex) {
        TFile* f = threadResults[threadIndex].tempFile;
        if (f != nullptr && f->TestBit(TFile::kWriteError)) {
            for (auto& result : threadResults) {
                cleanupThreadResult(result);
            }
            throw runtime_error("Thread " + to_string(threadIndex) +
                                " temp file write error (possibly NFS stale handle): " +
                                threadResults[threadIndex].tempFilePath);
        }
    }

    try {
        const vector<string> threadTempPaths = finalizeThreadTempFiles(threadResults);
        const vector<string> writtenFiles = writeOutputFilesStreaming(batchOutputPath,
                                                                      threadTempPaths,
                                                                      branchConfig.trees,
                                                                      0);
        for (auto& result : threadResults) {
            cleanupThreadResult(result);
        }
        return writtenFiles;
    } catch (...) {
        for (auto& result : threadResults) {
            cleanupThreadResult(result);
        }
        throw;
    }
}

BatchTempCollection collectSuccessfulBatchTempFiles(const AppConfig& appConfig,
                                                    const SampleMeta& sampleMeta,
                                                    const BranchConfig& branchConfig,
                                                    const vector<size_t>& batchIndices,
                                                    size_t nBatches) {
    BatchTempCollection collection;
    collection.paths.reserve(batchIndices.size());
    for (const size_t batchIndex : batchIndices) {
        const fs::path batchOutputPath = makeBatchTempOutputPath(appConfig, sampleMeta, batchIndex);
        Long64_t batchRawEntries = 0;
        string invalidReason;
        if (!validateBatchTempOutput(batchOutputPath,
                                     branchConfig.trees,
                                     batchRawEntries,
                                     invalidReason)) {
            ++collection.skipped;
            cerr << "Warning: skipping incomplete batch " << (batchIndex + 1)
                 << "/" << nBatches << " for sample = " << sampleMeta.sample
                 << ": " << invalidReason << endl;
            continue;
        }
        collection.rawEntries += batchRawEntries;
        collection.paths.push_back(batchOutputPath.string());
    }

    if (collection.paths.empty()) {
        throw runtime_error("No successful temporary batch outputs found for sample " +
                            sampleMeta.sample);
    }
    return collection;
}

int finalizeSuccessfulBatches(const AppConfig& appConfig,
                              const SampleMeta& sampleMeta,
                              const BranchConfig& branchConfig,
                              const vector<size_t>& batchIndices,
                              size_t nBatches) {
    BatchTempCollection batchFiles;
    try {
        batchFiles = collectSuccessfulBatchTempFiles(appConfig, sampleMeta, branchConfig, batchIndices, nBatches);
        writeSampleRawEntries(appConfig.sampleConfigPath, sampleMeta.sample, batchFiles.rawEntries);
        cout << "Updated raw_entries in " << appConfig.sampleConfigPath
             << " for sample = " << sampleMeta.sample
             << ", tree = " << appConfig.treeName
             << ", raw_entries = " << batchFiles.rawEntries << endl;
    } catch (const exception& ex) {
        cerr << "raw_entries update error: " << ex.what() << endl;
        return 1;
    }

    try {
        const fs::path outputPath(sampleMeta.outputFileName);
        if (!outputPath.parent_path().empty()) {
            fs::create_directories(outputPath.parent_path());
        }

        const Long64_t maxOutputBytes = outputSizeLimitBytes(appConfig.maxOutputFileSizeGB);
        cout << "Merging " << batchFiles.paths.size()
             << " successful temporary batch file" << (batchFiles.paths.size() == 1 ? "" : "s")
             << " out of " << nBatches << endl;
        const vector<string> writtenFiles = writeOutputFilesStreaming(outputPath,
                                                                      batchFiles.paths,
                                                                      branchConfig.trees,
                                                                      maxOutputBytes);
        if (writtenFiles.size() <= 1) {
            cout << "Wrote output file: "
                 << (writtenFiles.empty() ? sampleMeta.outputFileName : writtenFiles.front())
                 << endl;
        } else {
            cout << "Wrote output files:";
            for (const auto& fileName : writtenFiles) {
                cout << ' ' << fileName;
            }
            cout << endl;
        }
    } catch (const exception& ex) {
        cerr << "Output error: " << ex.what() << endl;
        return 1;
    }

    if (batchFiles.skipped > 0) {
        cerr << "Warning: skipped " << batchFiles.skipped
             << " incomplete selected batch"
             << (batchFiles.skipped == 1 ? "" : "es")
             << " while finalizing sample = " << sampleMeta.sample << endl;
    }
    return 0;
}

size_t inferNBatchesFromTempDir(const fs::path& batchTempDir, const string& sampleName) {
    if (!fs::is_directory(batchTempDir)) {
        return 0;
    }
    const string prefix = sampleName + "_";
    const string suffix = ".root";
    size_t maxIndex = 0;
    bool found = false;
    for (const auto& entry : fs::directory_iterator(batchTempDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const string name = entry.path().filename().string();
        if (name.size() <= prefix.size() + suffix.size()) {
            continue;
        }
        if (name.substr(0, prefix.size()) != prefix) {
            continue;
        }
        if (name.substr(name.size() - suffix.size()) != suffix) {
            continue;
        }
        const string indexStr = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        size_t batchIndex = 0;
        if (!parseNonNegativeIndex(indexStr, batchIndex)) {
            continue;
        }
        if (!found || batchIndex > maxIndex) {
            maxIndex = batchIndex;
            found = true;
        }
    }
    return found ? maxIndex + 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    TH1::AddDirectory(false);

    AppConfig appConfig;
    BranchConfig branchConfig;
    SelectionConfig selectionConfig;
    try {
        appConfig = loadAppConfig();
        branchConfig = loadBranchConfig(appConfig);
        selectionConfig = loadSelectionConfig(appConfig);
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

    BatchRequest batchRequest;
    try {
        batchRequest = resolveBatchRequest(argc, argv);
    } catch (const exception& ex) {
        cerr << "Batch selection error: " << ex.what() << endl;
        return 1;
    }

    SampleMeta sampleMeta;
    try {
        sampleMeta = resolveSampleMeta(sample, appConfig);
    } catch (const exception& ex) {
        cerr << "Sample resolution error: " << ex.what() << endl;
        return 1;
    }

    if (batchRequest.mergeSuccessfulBatches) {
        const fs::path batchTempDir = makeBatchTempOutputDir(appConfig, sampleMeta);
        const size_t nBatches = inferNBatchesFromTempDir(batchTempDir, sampleMeta.sample);
        cout << "Running convert_branch for sample = " << sample
             << ", merge successful batches" << endl;
        cout << "Batch mode: " << nBatches
             << " batch" << (nBatches == 1 ? "" : "es")
             << ", temporary output = " << batchTempDir.string() << endl;
        vector<size_t> batchIndices;
        try {
            batchIndices = resolveBatchIndicesForFinalMerge(nBatches, batchRequest);
        } catch (const exception& ex) {
            cerr << "Batch selection error: " << ex.what() << endl;
            return 1;
        }
        return finalizeSuccessfulBatches(appConfig, sampleMeta, branchConfig, batchIndices, nBatches);
    }

    vector<string> inputFiles;
    try {
        inputFiles = discoverInputFiles(sampleMeta);
    } catch (const exception& ex) {
        cerr << "Input discovery error: " << ex.what() << endl;
        return 1;
    }

    const int threadCount = determineThreadCount(appConfig.maxThreads, inputFiles.size());
    size_t batchSize;
    {
        const char* fpbEnv = getenv("CONVERT_FILES_PER_BATCH");
        size_t fpbOverride = 0;
        if (fpbEnv != nullptr && *fpbEnv != '\0') {
            try { fpbOverride = static_cast<size_t>(stoull(fpbEnv)); } catch (...) {}
        }
        batchSize = fpbOverride > 0 ? fpbOverride
                                    : max<size_t>(1, static_cast<size_t>(threadCount) * 32);
    }
    const size_t nBatches = (inputFiles.size() + batchSize - 1) / batchSize;
    if (batchRequest.printBatchCount) {
        cout << nBatches << endl;
        return 0;
    }
    if (batchRequest.singleBatch && batchRequest.batchIndex >= nBatches) {
        cerr << "Batch selection error: requested batch " << batchRequest.batchIndex
             << " but sample has " << nBatches << " batch"
             << (nBatches == 1 ? "" : "es") << endl;
        return 1;
    }

    cout << "Running convert_branch for sample = " << sample
         << ", files = " << inputFiles.size();
    if (batchRequest.singleBatch) {
        cout << ", batch = " << (batchRequest.batchIndex + 1) << "/" << nBatches;
    }
    if (sampleMeta.inputPaths.size() == 1) {
        cout << ", source = " << sampleMeta.inputPaths.front()
             << (sampleMeta.remoteSourceCount == 1 ? " [dataset]" : " [local]");
    } else {
        cout << ", sources = " << sampleMeta.inputPaths.size()
             << " (dataset = " << sampleMeta.remoteSourceCount
             << ", local = " << (sampleMeta.inputPaths.size() - sampleMeta.remoteSourceCount) << ")";
    }
    cout << endl;

    const fs::path batchTempDir = makeBatchTempOutputDir(appConfig, sampleMeta);

    vector<PileupBin> pileupWeights;
    if (sampleMeta.isMC && !appConfig.puWeightPathPattern.empty()) {
        try {
            const string puWeightPath = resolvePileupWeightPath(appConfig, sampleMeta);
            pileupWeights = loadPileupWeights(puWeightPath);
            cout << "Loaded pileup weights from: " << puWeightPath << endl;
        } catch (const exception& ex) {
            cerr << "Pileup weight error: " << ex.what() << endl;
            return 1;
        }
    }

    unique_ptr<LumiMask> lumiMask;
    if (!sampleMeta.isMC && !appConfig.lumiMaskPath.empty()) {
        try {
            lumiMask = make_unique<LumiMask>(loadLumiMask(appConfig.lumiMaskPath));
            cout << "Loaded lumi mask from: " << appConfig.lumiMaskPath
                 << " (" << lumiMask->runs.size() << " runs)" << endl;
        } catch (const exception& ex) {
            cerr << "Lumi mask error: " << ex.what() << endl;
            return 1;
        }
    }

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

    cout << "Batch mode: " << nBatches
         << " batch" << (nBatches == 1 ? "" : "es")
         << ", max files per batch = " << batchSize
         << ", temporary output = " << batchTempDir.string() << endl;

    const size_t firstBatchIndex = batchRequest.singleBatch ? batchRequest.batchIndex : 0;
    const size_t lastBatchIndexExclusive = batchRequest.singleBatch ? (batchRequest.batchIndex + 1) : nBatches;
    atomic<size_t> processedFiles{firstBatchIndex * batchSize};

    for (size_t batchIndex = firstBatchIndex; batchIndex < lastBatchIndexExclusive; ++batchIndex) {
        const size_t begin = batchIndex * batchSize;
        const size_t end = min(inputFiles.size(), begin + batchSize);
        const auto batchBegin = inputFiles.begin() + static_cast<vector<string>::difference_type>(begin);
        const auto batchEnd = inputFiles.begin() + static_cast<vector<string>::difference_type>(end);
        vector<string> batchInputFiles(batchBegin, batchEnd);
        const int batchThreadCount = determineThreadCount(appConfig.maxThreads, batchInputFiles.size());
        const fs::path batchOutputPath = makeBatchTempOutputPath(appConfig, sampleMeta, batchIndex);
        bool batchAlreadyComplete = false;
        if (appConfig.resumeSuccessfulBatches && batchRequest.singleBatch) {
            Long64_t existingRawEntries = 0;
            string invalidReason;
            if (validateBatchTempOutput(batchOutputPath,
                                        branchConfig.trees,
                                        existingRawEntries,
                                        invalidReason)) {
                cout << "Skipping completed batch " << (batchIndex + 1) << "/" << nBatches
                     << ": found valid existing temporary batch file "
                     << batchOutputPath.string()
                     << " with raw_entries = " << existingRawEntries << endl;
                batchAlreadyComplete = true;
            }
            if (!batchAlreadyComplete &&
                (fs::exists(batchOutputPath) || fs::exists(makeBatchRawEntriesPath(batchOutputPath)))) {
                cerr << "Warning: resume check will rerun batch " << (batchIndex + 1)
                     << "/" << nBatches << " for sample = " << sampleMeta.sample
                     << ": existing temporary output is incomplete (" << invalidReason << ")" << endl;
            }
        }
        if (batchAlreadyComplete) {
            continue;
        }
        atomic<Long64_t> batchRawEntries{0};

        cout << "Processing batch " << (batchIndex + 1) << "/" << nBatches
             << ": files " << (begin + 1) << "-" << end
             << " -> " << batchOutputPath.string()
             << " using " << batchThreadCount << " thread"
             << (batchThreadCount == 1 ? "" : "s") << endl;

        try {
            const vector<string> writtenBatchFiles = processInputBatchToTempFile(batchInputFiles,
                                                                                 batchIndex,
                                                                                 batchThreadCount,
                                                                                 batchOutputPath,
                                                                                 appConfig,
                                                                                 selectionConfig,
                                                                                 sampleMeta,
                                                                                 pileupWeights,
                                                                                 lumiMask.get(),
                                                                                 branchConfig,
                                                                                 processedFiles,
                                                                                 inputFiles.size(),
                                                                                 batchRawEntries);
            if (writtenBatchFiles.size() != 1) {
                throw runtime_error("Expected one temporary batch file, got " +
                                    to_string(writtenBatchFiles.size()));
            }
            writeBatchRawEntries(batchOutputPath, batchRawEntries.load());
            cout << "Wrote temporary batch file: " << writtenBatchFiles.front() << endl;
        } catch (const exception& ex) {
            cerr << "Runtime error: " << ex.what() << endl;
            return 1;
        }
    }

    const bool deferFinalMerge = batchRequest.singleBatch && finalMergeDeferredByEnv();
    if (batchRequest.singleBatch &&
        (batchRequest.batchIndex + 1 < nBatches || deferFinalMerge)) {
        cout << "Batch " << (batchRequest.batchIndex + 1) << "/" << nBatches
             << " complete; final merge will run after the batch loop" << endl;
        return 0;
    }

    vector<size_t> batchIndices;
    try {
        batchIndices = resolveBatchIndicesForFinalMerge(nBatches, batchRequest);
    } catch (const exception& ex) {
        cerr << "Batch selection error: " << ex.what() << endl;
        return 1;
    }
    return finalizeSuccessfulBatches(appConfig, sampleMeta, branchConfig, batchIndices, nBatches);
}
