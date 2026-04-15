// Summary: Skim events and convert branches for BDT training with JSON-driven sample config, string formulas, and precompiled selections.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <TFile.h>
#include <TH1.h>
#include <TList.h>
#include <TLorentzVector.h>
#include <TROOT.h>
#include <TTree.h>

#include "../src/simple_json.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
namespace fs = std::filesystem;
using JsonValue = simple_json::Value;

namespace {

const float def = -99.f;
const double kMissingDistance = -999.;
const double kLargeDistance = 999.;
const char* kRemotePrefix = "root://cms-xrd-global.cern.ch/";

const char* kAppConfigPath = "./config.json";
const char* kBranchConfigPath = "./branch.json";
const char* kSelectionConfigPath = "./selection.json";
const char* kAppConfigEnvVar = "CONVERT_CONFIG_PATH";

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
            throw runtime_error("Missing scalar branch: " + branch);
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

    void bind(TTree* tree, bool isMC) {
        if (onlyMC && !isMC) {
            bound = false;
            return;
        }
        if (!tree->GetBranch(branch.c_str())) {
            throw runtime_error("Missing array branch: " + branch);
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

struct OutputTreeState {
    TreeConfig config;
    TTree* tree = nullptr;
    vector<OutputBranchRuntime> branches;
    unordered_map<string, size_t> branchIndexByName;
};

struct ThreadConvertResult {
    vector<OutputTreeState> outputTrees;
};

struct SampleRuleConfig {
    vector<string> containsAny;
    string category;
    string path;
    int sampleType = 0;
    bool hasSampleType = false;
    bool isSignal = false;
    bool hasIsSignal = false;
};

struct AppConfig {
    string treeName = "Events";
    string outputRoot;
    string outputPattern;
    string runSample;
    int maxThreads = 12;
    double maxOutputFileSizeGB = 5.;
    string defaultCategory = "bkg";
    int defaultSampleType = -1;
    bool defaultIsSignal = false;
    vector<SampleRuleConfig> sampleRules;
    string puWeightPathPattern;
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
    string category;
    string inputPath;
    string outputFileName;
    int sampleType = -1;
    bool isSignal = false;
    bool isRemoteDataset = false;

    bool isMC() const {
        return category != "data";
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
    const JsonValue payload = loadJson(kAppConfigPath, kAppConfigEnvVar);

    AppConfig config;
    config.treeName = payload.getStringOr("tree_name", "Events");
    config.runSample = payload.getStringOr("run_sample", "");
    config.outputRoot = payload.at("output_root").asString();
    config.outputPattern = payload.at("output_pattern").asString();
    config.maxThreads = payload.getIntOr("max_threads", 12);
    config.maxOutputFileSizeGB = static_cast<double>(payload.getNumberOr("max_output_file_size_gb", 5.));

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
            rule.path = node.getStringOr("path", "");
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

    config.puWeightPathPattern = payload.getStringOr("pu_weight_path", "");
    return config;
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

BranchConfig loadBranchConfig() {
    const JsonValue payload = loadJson(kBranchConfigPath);

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

SelectionConfig loadSelectionConfig() {
    const JsonValue payload = loadJson(kSelectionConfigPath);

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

unordered_map<string, long double> buildRawScalarValues(const BranchConfig& branchConfig,
                                                        const SampleMeta& sampleMeta,
                                                        const vector<PileupBin>* pileupWeights = nullptr) {
    unordered_map<string, long double> values;
    values.reserve(branchConfig.scalars.size() + 5);
    for (const auto& scalar : branchConfig.scalars) {
        values[scalar.name] = scalar.numericValue();
    }
    values["sample_type"] = sampleMeta.sampleType;
    values["is_signal"] = sampleMeta.isSignal ? 1. : 0.;
    if (sampleMeta.isMC()) {
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

Value evalCall(const ExprPtr& expr, const EvalContext& context) {
    const string& op = expr->text;
    const auto& args = expr->args;

    if (op == "abs") {
        return makeNumberValue(fabsl(evalNumber(args.at(0), context)));
    }
    if (op == "sqrt") {
        return makeNumberValue(sqrtl(evalNumber(args.at(0), context)));
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
    for (const auto& token : rule.containsAny) {
        if (sample.find(token) != string::npos) {
            return true;
        }
    }
    return false;
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

bool hasRemainingEntries(const vector<Long64_t>& currentEntries,
                         const vector<Long64_t>& totalEntries) {
    for (size_t index = 0; index < totalEntries.size(); ++index) {
        if (currentEntries[index] < totalEntries[index]) {
            return true;
        }
    }
    return false;
}

void writeSingleOutputFile(const string& outputFileName,
                           const vector<unique_ptr<TTree>>& mergedTrees) {
    unique_ptr<TFile> outFile(TFile::Open(outputFileName.c_str(), "RECREATE"));
    if (!outFile || outFile->IsZombie()) {
        throw runtime_error("Error opening output file " + outputFileName);
    }

    outFile->cd();
    for (const auto& mergedTree : mergedTrees) {
        mergedTree->Write();
    }
    outFile->Close();
}

vector<string> writeSplitOutputFiles(const fs::path& baseOutputPath,
                                     const vector<unique_ptr<TTree>>& mergedTrees,
                                     Long64_t maxOutputBytes) {
    const Long64_t targetBytes = max<Long64_t>(1, static_cast<Long64_t>(maxOutputBytes * 0.90));
    vector<Long64_t> totalEntries;
    vector<Long64_t> currentEntries;
    vector<double> bytesPerEntry;
    totalEntries.reserve(mergedTrees.size());
    currentEntries.assign(mergedTrees.size(), 0);
    bytesPerEntry.reserve(mergedTrees.size());

    for (const auto& mergedTree : mergedTrees) {
        const Long64_t entries = mergedTree->GetEntries();
        totalEntries.push_back(entries);
        const Long64_t estimatedBytes = estimateTreeBytes(mergedTree.get());
        const double perEntry = (entries > 0) ? static_cast<double>(estimatedBytes) / static_cast<double>(entries) : 1.0;
        bytesPerEntry.push_back(max(perEntry, 1.0));
    }

    vector<string> writtenFiles;
    size_t fileIndex = 0;
    while (hasRemainingEntries(currentEntries, totalEntries)) {
        long double remainingBytes = 0.;
        for (size_t treeIndex = 0; treeIndex < mergedTrees.size(); ++treeIndex) {
            remainingBytes += static_cast<long double>(totalEntries[treeIndex] - currentEntries[treeIndex]) * bytesPerEntry[treeIndex];
        }

        long double ratio = 1.;
        if (remainingBytes > static_cast<long double>(targetBytes)) {
            ratio = static_cast<long double>(targetBytes) / remainingBytes;
        }

        const fs::path chunkPath = makeSplitOutputPath(baseOutputPath, fileIndex++);
        unique_ptr<TFile> outFile(TFile::Open(chunkPath.c_str(), "RECREATE"));
        if (!outFile || outFile->IsZombie()) {
            throw runtime_error("Error opening output file " + chunkPath.string());
        }

        bool wroteAnyTree = false;
        for (size_t treeIndex = 0; treeIndex < mergedTrees.size(); ++treeIndex) {
            const Long64_t entriesLeft = totalEntries[treeIndex] - currentEntries[treeIndex];
            if (entriesLeft <= 0) {
                continue;
            }

            Long64_t chunkEntries = entriesLeft;
            if (ratio < 1.) {
                chunkEntries = static_cast<Long64_t>(floor(static_cast<long double>(entriesLeft) * ratio));
            }
            if (chunkEntries <= 0) {
                continue;
            }

            unique_ptr<TTree> chunkTree(mergedTrees[treeIndex]->CopyTree("", "", chunkEntries, currentEntries[treeIndex]));
            if (!chunkTree) {
                throw runtime_error("Failed to split output tree: " + string(mergedTrees[treeIndex]->GetName()));
            }

            outFile->cd();
            chunkTree->Write();
            currentEntries[treeIndex] += chunkEntries;
            wroteAnyTree = true;
        }

        if (!wroteAnyTree) {
            for (size_t treeIndex = 0; treeIndex < mergedTrees.size(); ++treeIndex) {
                const Long64_t entriesLeft = totalEntries[treeIndex] - currentEntries[treeIndex];
                if (entriesLeft <= 0) {
                    continue;
                }
                unique_ptr<TTree> chunkTree(mergedTrees[treeIndex]->CopyTree("", "", 1, currentEntries[treeIndex]));
                if (!chunkTree) {
                    throw runtime_error("Failed to split output tree: " + string(mergedTrees[treeIndex]->GetName()));
                }

                outFile->cd();
                chunkTree->Write();
                currentEntries[treeIndex] += 1;
                wroteAnyTree = true;
                break;
            }
        }

        if (!wroteAnyTree) {
            throw runtime_error("Unable to write any trees while splitting output file " + baseOutputPath.string());
        }

        outFile->Close();
        writtenFiles.push_back(chunkPath.string());
    }

    return writtenFiles;
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
    sampleMeta.isRemoteDataset = isCmsDatasetPath(sampleMeta.inputPath);
    vector<string> files = sampleMeta.isRemoteDataset ? listRemoteRootFiles(sampleMeta.inputPath)
                                                      : listLocalRootFiles(sampleMeta.inputPath);
    if (files.empty()) {
        throw runtime_error("No ROOT files found for sample " + sampleMeta.sample + " from path " + sampleMeta.inputPath);
    }
    return files;
}

SampleMeta resolveSampleMeta(const string& sample, const AppConfig& appConfig) {
    SampleMeta meta;
    meta.sample = sample;
    meta.category = appConfig.defaultCategory;
    meta.sampleType = appConfig.defaultSampleType;
    meta.isSignal = appConfig.defaultIsSignal;

    string pathTemplate;
    for (const auto& rule : appConfig.sampleRules) {
        if (!matchesRule(sample, rule)) {
            continue;
        }
        if (!rule.category.empty()) {
            meta.category = rule.category;
        }
        if (!rule.path.empty()) {
            pathTemplate = rule.path;
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

    if (pathTemplate.empty()) {
        throw runtime_error("No input path configured for sample: " + sample);
    }

    unordered_map<string, string> templateValues;
    templateValues["sample"] = meta.sample;
    templateValues["category"] = meta.category;
    templateValues["output_root"] = appConfig.outputRoot;

    meta.inputPath = applyTemplate(pathTemplate, templateValues);
    meta.outputFileName = applyTemplate(appConfig.outputPattern, templateValues);
    return meta;
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

string resolvePileupWeightPath(const AppConfig& appConfig, const SampleMeta& sampleMeta) {
    unordered_map<string, string> templateValues;
    templateValues["sample"] = sampleMeta.sample;
    templateValues["category"] = sampleMeta.category;
    templateValues["output_root"] = appConfig.outputRoot;
    return applyTemplate(appConfig.puWeightPathPattern, templateValues);
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

void bookTreeBranches(OutputTreeState& treeState, bool isMC) {
    const size_t totalBranches = countOutputGroupBranches(treeState.config.regularScalars, isMC) +
                                 countOutputGroupBranches(treeState.config.extremaScalars, isMC);
    treeState.tree = new TTree(treeState.config.name.c_str(), treeState.config.title.c_str());
    treeState.branches.reserve(totalBranches);
    treeState.branchIndexByName.reserve(totalBranches);
    bookOutputGroup(treeState, treeState.config.regularScalars, isMC);
    bookOutputGroup(treeState, treeState.config.extremaScalars, isMC);
}

vector<OutputTreeState> makeOutputTrees(const BranchConfig& branchConfig, bool isMC) {
    vector<OutputTreeState> outputTrees;
    outputTrees.reserve(branchConfig.trees.size());
    for (const auto& treeConfig : branchConfig.trees) {
        outputTrees.emplace_back();
        outputTrees.back().config = treeConfig;
        bookTreeBranches(outputTrees.back(), isMC);
    }
    return outputTrees;
}

void destroyOutputTrees(vector<OutputTreeState>& outputTrees) {
    for (auto& treeState : outputTrees) {
        delete treeState.tree;
        treeState.tree = nullptr;
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

TTree* mergeOutputTree(const vector<ThreadConvertResult>& results,
                       size_t treeIndex,
                       const TreeConfig& treeConfig,
                       bool isMC) {
    TList list;
    list.SetOwner(false);
    bool hasEntries = false;
    for (const auto& result : results) {
        if (treeIndex >= result.outputTrees.size() || result.outputTrees[treeIndex].tree == nullptr) {
            continue;
        }
        list.Add(result.outputTrees[treeIndex].tree);
        hasEntries = hasEntries || (result.outputTrees[treeIndex].tree->GetEntries() > 0);
    }

    TTree* merged = nullptr;
    if (hasEntries) {
        merged = TTree::MergeTrees(&list);
    }
    if (merged == nullptr) {
        OutputTreeState emptyState;
        emptyState.config = treeConfig;
        bookTreeBranches(emptyState, isMC);
        merged = emptyState.tree;
        emptyState.tree = nullptr;
    }
    merged->SetNameTitle(treeConfig.name.c_str(), treeConfig.title.c_str());
    return merged;
}

void processInputFile(const string& inputFileName,
                      const AppConfig& appConfig,
                      const SelectionConfig& selectionConfig,
                      const SampleMeta& sampleMeta,
                      const vector<PileupBin>& pileupWeights,
                      BranchConfig& branchConfig,
                      vector<OutputTreeState>& outputTrees) {
    unique_ptr<TFile> inputFile(TFile::Open(inputFileName.c_str(), "READ"));
    if (!inputFile || inputFile->IsZombie()) {
        throw runtime_error("Error opening input file " + inputFileName);
    }

    TTree* tree = static_cast<TTree*>(inputFile->Get(appConfig.treeName.c_str()));
    if (!tree) {
        throw runtime_error("Tree " + appConfig.treeName + " not found in " + inputFileName);
    }

    configureActiveBranches(tree, branchConfig, sampleMeta.isMC());
    unordered_map<string, const ScalarInputConfig*> rawScalarByName = bindInputBranches(tree, branchConfig, sampleMeta.isMC());

    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t entry = 0; entry < nEntries; ++entry) {
        tree->GetEntry(entry);

        unordered_map<string, long double> baseVars = buildRawScalarValues(branchConfig, sampleMeta, &pileupWeights);

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

            fillOutputTree(treeState, runtimeCollections, inputCollections, baseVars, rawScalarByName, sampleMeta.isMC());
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    TH1::AddDirectory(false);

    AppConfig appConfig;
    BranchConfig branchConfig;
    SelectionConfig selectionConfig;
    try {
        appConfig = loadAppConfig();
        branchConfig = loadBranchConfig();
        selectionConfig = loadSelectionConfig();
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

    vector<string> inputFiles;
    try {
        inputFiles = discoverInputFiles(sampleMeta);
    } catch (const exception& ex) {
        cerr << "Input discovery error: " << ex.what() << endl;
        return 1;
    }

    cout << "Running convert_branch for sample = " << sample
         << ", files = " << inputFiles.size()
         << ", source = " << sampleMeta.inputPath
         << (sampleMeta.isRemoteDataset ? " [dataset]" : " [local]") << endl;

    vector<PileupBin> pileupWeights;
    if (sampleMeta.isMC() && !appConfig.puWeightPathPattern.empty()) {
        try {
            const string puWeightPath = resolvePileupWeightPath(appConfig, sampleMeta);
            pileupWeights = loadPileupWeights(puWeightPath);
            cout << "Loaded pileup weights from: " << puWeightPath << endl;
        } catch (const exception& ex) {
            cerr << "Pileup weight error: " << ex.what() << endl;
            return 1;
        }
    }

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

    vector<ThreadConvertResult> threadResults(threadCount);
    for (auto& result : threadResults) {
        result.outputTrees = makeOutputTrees(branchConfig, sampleMeta.isMC());
    }
    vector<BranchConfig> threadConfigs(threadCount, branchConfig);

    atomic<size_t> processedFiles{0};
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
        for (int index = 0; index < static_cast<int>(inputFiles.size()); ++index) {
            if (failed.load()) {
                continue;
            }

            try {
                processInputFile(inputFiles[index],
                                 appConfig,
                                 selectionConfig,
                                 sampleMeta,
                                 pileupWeights,
                                 threadConfigs[tid],
                                 threadResults[tid].outputTrees);
                const size_t done = processedFiles.fetch_add(1) + 1;
#pragma omp critical(convert_progress)
                printFileProgress(sampleMeta.sample, done, inputFiles.size());
            } catch (const exception& ex) {
                failed.store(true);
#pragma omp critical(convert_error)
                errors.push_back(ex.what());
            }
        }
    }

    if (!errors.empty()) {
        cerr << "Runtime error: " << errors.front() << endl;
        for (auto& result : threadResults) {
            destroyOutputTrees(result.outputTrees);
        }
        return 1;
    }

    try {
        const fs::path outputPath(sampleMeta.outputFileName);
        if (!outputPath.parent_path().empty()) {
            fs::create_directories(outputPath.parent_path());
        }
        vector<unique_ptr<TTree>> mergedTrees;
        mergedTrees.reserve(branchConfig.trees.size());
        for (size_t treeIndex = 0; treeIndex < branchConfig.trees.size(); ++treeIndex) {
            mergedTrees.emplace_back(mergeOutputTree(threadResults, treeIndex, branchConfig.trees[treeIndex], sampleMeta.isMC()));
        }

        const Long64_t maxOutputBytes = outputSizeLimitBytes(appConfig.maxOutputFileSizeGB);
        Long64_t estimatedTotalBytes = 0;
        for (const auto& mergedTree : mergedTrees) {
            estimatedTotalBytes += estimateTreeBytes(mergedTree.get());
        }

        if (maxOutputBytes > 0 && estimatedTotalBytes > maxOutputBytes) {
            const vector<string> writtenFiles = writeSplitOutputFiles(outputPath, mergedTrees, maxOutputBytes);
            cout << "Wrote output files:";
            for (const auto& fileName : writtenFiles) {
                cout << ' ' << fileName;
            }
            cout << endl;
        } else {
            writeSingleOutputFile(sampleMeta.outputFileName, mergedTrees);
            cout << "Wrote output file: " << sampleMeta.outputFileName << endl;
        }
    } catch (const exception& ex) {
        cerr << "Output error: " << ex.what() << endl;
        for (auto& result : threadResults) {
            destroyOutputTrees(result.outputTrees);
        }
        return 1;
    }

    for (auto& result : threadResults) {
        destroyOutputTrees(result.outputTrees);
    }
    return 0;
}
