// Summary: Skim events and convert branches for BDT training with JSON-driven sample config, string formulas, and precompiled selections.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <TBranch.h>
#include <TFile.h>
#include <TFileMerger.h>
#include <TH1.h>
#include <TList.h>
#include <TLorentzVector.h>
#include <TROOT.h>
#include <TTree.h>

#include "../../src/simple_json.h"
#include "correction.h"

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
const double kNominalWMass = 80.4;
const double kNominalZMass = 91.2;
// Data-driven scales (IQR/1.349 robust sigma, from the true same-W AK4 dijet pair
// distributions in fat2 www MC) used to combine the mass and deltaR terms of the
// "combined_wz_dr" resolved-dijet pairing criterion into a unitless chi2-like score.
const double kWZMassSigma = 12.7;
const double kPairDrSigma = 0.60;
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
// Set to 1 (run.py does this for a new mode-0 submission) to re-query DAS and overwrite the
// per-sample input file-list snapshot; otherwise an existing snapshot is reused.
const char* kRefreshFileListEnvVar = "CONVERT_REFRESH_FILE_LIST";
// ZSTD level 5 for every ROOT file convert writes (thread temps, batch and final outputs),
// so the batch and final merges can copy the compressed baskets unchanged (fast merge).
const int kOutputCompression = 505;

// Jet/MET corrections (see JetPtCorrector).
const char* kAk4JetCollection = "ScoutingPFJetRecluster2";
const char* kAk8JetCollection = "ScoutingFatPFJetRecluster";
const char* kMetPtScalar = "ScoutingMET_pt";
const char* kMetPhiScalar = "ScoutingMET_phi";
const double kTwoPi = 6.28318530717958647692;
// Type-1 MET propagates AK4 jets whose muon-subtracted, fully corrected pT is
// above this threshold and whose EM energy fraction is below the maximum.
const double kType1JetPtThreshold = 15.0;
const double kType1MaxEmFraction = 0.9;
// The JER PtResolution Rho axis ends at 52.05 and returns 1.0 (100% resolution)
// outside it, so rho is clamped below that edge for the resolution lookup.
const double kJerRhoUpperEdge = 52.05;
// Event rho recovered from the AK4 production JEC (JetPtCorrector::recoverRho).
const double kRhoRecoveryMax = 70.0;              // L1FastJet clamps rho to [0, 70]
const double kRhoRecoveryTolerance = 0.5;         // GeV, allowed spread between jets
const double kRhoRecoveryMinFactor = 1e-3;        // jets at the L1FastJet floor carry no rho
const double kRhoRecoveryMinSensitivity = 1e-4;   // min relative JEC change over the rho range
const double kRhoSolverTolerance = 1e-10;         // |JEC - stored factor| at the rho solution
const int kRhoSolverMaxIterations = 60;
const double kJecBinEdgeProbe = 1e-3;             // eta/phi shift that reaches a neighbouring JEC bin
const double kJecBinProbeRho = 30.0;              // GeV, rho at which neighbouring bins are compared
const vector<string> kJetVariationNames = {"jes_up", "jes_down", "jer_up", "jer_down",
                                           "jms_up", "jms_down", "jmr_up", "jmr_down"};

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

// Operator / function / reserved-identifier code, resolved once from Expression::text at parse
// time so evaluation dispatches on an enum instead of comparing strings for every node.
enum class Op {
    None,
    Plus, Minus, Not, Mul, Div, Lt, Le, Gt, Ge, Eq, Ne, And, Or,
    True, False, Self, Other,
    Abs, Sqrt, Cos, Sin, Pow, Min, Max, SafeDiv, FirstValid, Size,
    Sum, MaxValue, MinValue, Sphericity, Aplanarity, Planarity,
    NthMaxValue, ValueAtMax, ValueAtNthMax, ValueAt,
    FirstAncestorIndex, FirstNonQgAncestorIndex, FirstBosonAncestorIndex,
    CountHadronicTauFromWz, CountLeptonicTauFromWz,
    PairP4MinDr, PairP4ClosestWzMass, PairIndexMinDr, PairIndexClosestWzMass,
    PairP4CombinedWzDr, PairIndexCombinedWzDr, PairP4SfosZMass, PairIndexSfosZMass,
    Mass, Pt, Eta, Phi, DeltaR, DeltaPhi, RelPtDiff,
    PairMinDeltaR, PairMaxDeltaR, PairMinDeltaPhi, PairMaxDeltaPhi,
    ClosestDeltaR, MinDeltaR, MaxRatioWithinDr, DeltaPhiAtMinDeltaR,
};

struct Expression {
    ExprKind kind = ExprKind::Number;
    Op op = Op::None;
    long double number = 0.;
    string text;
    ExprPtr lhs;
    ExprPtr rhs;
    vector<ExprPtr> args;
    // Identifier nodes, filled once by resolveEngineSymbols: the event-variable slot, the
    // runtime/input collection slots (-1: no such name) and, per ObjectSchema::id, the field index
    // (-1: not a field of that schema).
    bool resolved = false;
    int varSlot = -1;
    int runtimeSlot = -1;
    int inputSlot = -1;
    vector<int> fieldIndex;
};

Op operatorOp(const string& text) {
    static const unordered_map<string, Op> ops = {
        {"+", Op::Plus}, {"-", Op::Minus}, {"!", Op::Not}, {"*", Op::Mul}, {"/", Op::Div},
        {"<", Op::Lt}, {"<=", Op::Le}, {">", Op::Gt}, {">=", Op::Ge}, {"==", Op::Eq},
        {"!=", Op::Ne}, {"&&", Op::And}, {"||", Op::Or},
    };
    const auto it = ops.find(text);
    return (it != ops.end()) ? it->second : Op::None;
}

Op identifierOp(const string& text) {
    if (text == "true") return Op::True;
    if (text == "false") return Op::False;
    if (text == "self") return Op::Self;
    if (text == "other") return Op::Other;
    return Op::None;
}

// Unknown names map to Op::None, which evalCall rejects as an unsupported function.
Op functionOp(const string& text) {
    static const unordered_map<string, Op> ops = {
        {"abs", Op::Abs}, {"sqrt", Op::Sqrt}, {"cos", Op::Cos}, {"sin", Op::Sin}, {"pow", Op::Pow},
        {"min", Op::Min}, {"max", Op::Max}, {"safe_div", Op::SafeDiv}, {"first_valid", Op::FirstValid},
        {"size", Op::Size}, {"sum", Op::Sum}, {"max_value", Op::MaxValue}, {"min_value", Op::MinValue},
        {"sphericity", Op::Sphericity}, {"aplanarity", Op::Aplanarity}, {"planarity", Op::Planarity},
        {"nth_max_value", Op::NthMaxValue}, {"value_at_max", Op::ValueAtMax},
        {"value_at_nth_max", Op::ValueAtNthMax}, {"value_at", Op::ValueAt},
        {"first_ancestor_index", Op::FirstAncestorIndex},
        {"first_nonqg_ancestor_index", Op::FirstNonQgAncestorIndex},
        {"first_boson_ancestor_index", Op::FirstBosonAncestorIndex},
        {"count_hadronic_tau_from_wz", Op::CountHadronicTauFromWz},
        {"count_leptonic_tau_from_wz", Op::CountLeptonicTauFromWz},
        {"pair_p4_min_dr", Op::PairP4MinDr}, {"pair_p4_closest_wz_mass", Op::PairP4ClosestWzMass},
        {"pair_index_min_dr", Op::PairIndexMinDr}, {"pair_index_closest_wz_mass", Op::PairIndexClosestWzMass},
        {"pair_p4_combined_wz_dr", Op::PairP4CombinedWzDr}, {"pair_index_combined_wz_dr", Op::PairIndexCombinedWzDr},
        {"pair_p4_sfos_z_mass", Op::PairP4SfosZMass}, {"pair_index_sfos_z_mass", Op::PairIndexSfosZMass},
        {"mass", Op::Mass}, {"pt", Op::Pt}, {"eta", Op::Eta}, {"phi", Op::Phi},
        {"deltaR", Op::DeltaR}, {"deltaPhi", Op::DeltaPhi}, {"relPtDiff", Op::RelPtDiff},
        {"pair_min_deltaR", Op::PairMinDeltaR}, {"pair_max_deltaR", Op::PairMaxDeltaR},
        {"pair_min_deltaPhi", Op::PairMinDeltaPhi}, {"pair_max_deltaPhi", Op::PairMaxDeltaPhi},
        {"closest_deltaR", Op::ClosestDeltaR}, {"min_deltaR", Op::MinDeltaR},
        {"max_ratio_within_dr", Op::MaxRatioWithinDr}, {"deltaPhi_at_min_deltaR", Op::DeltaPhiAtMinDeltaR},
    };
    const auto it = ops.find(text);
    return (it != ops.end()) ? it->second : Op::None;
}

struct SortRule {
    string text;
    ExprPtr expr;
    bool descending = true;
};

struct ObjectSchema;

struct RuntimeCollectionConfig {
    string name;
    string source;
    vector<string> merge;
    // Merge collections only (resolveEngineSymbols): the merged field schema and, per merged
    // child, the child field index of every merged field (-1: the child lacks it -> def).
    shared_ptr<const ObjectSchema> mergedSchema;
    vector<vector<int>> mergeFieldMaps;
    // Resolved slots (resolveEngineSymbols): input collection of `source`, runtime collections
    // of `merge` and of `deduplicate_against` (-1: unset).
    int sourceSlot = -1;
    vector<int> mergeSlots;
    int dedupSlot = -1;
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
    // Indexed by runtime collection slot; a repeated name replaces the earlier definition.
    vector<RuntimeCollectionConfig> collections;
    unordered_map<string, int> collectionSlotByName;
    vector<int> buildOrder;  // slots of collectionOrder
    unordered_map<string, string> treeSelectionText;
    unordered_map<string, ExprPtr> treeSelections;
};

struct ScalarInputConfig {
    string name;
    string branch;
    DataType type = DataType::Int;
    bool onlyMC = false;
    bool optional = false;
    bool bound = false;
    int varSlot = -1;
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
            if (optional) {
                bound = false;
                return;
            }
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
    bool optional = false;
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
            if (optional) {
                bound = false;
                return;
            }
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
    // Set by resolveEngineSymbols: the field schema shared by every event's collection and the
    // event-variable slot of sizeName.
    shared_ptr<const ObjectSchema> schema;
    int sizeSlot = -1;
};

struct OutputScalarConfig {
    string name;
    DataType type = DataType::Float;
    bool onlyMC = false;
    string formulaText;
    ExprPtr formula;
    string collection;
    int slots = 0;
    // Set by resolveEngineSymbols: scalar outputs store their value in event-variable varSlot
    // (later formulas of the tree can read it); a formula that is just an input scalar's name is
    // copied exactly from branchConfig.scalars[exactScalarIndex]; collectionSlot is the runtime
    // collection of slot outputs (-1: unknown).
    int varSlot = -1;
    int exactScalarIndex = -1;
    int collectionSlot = -1;
};

struct TreeConfig {
    string name;
    string title;
    string selection;
    vector<OutputScalarConfig> regularScalars;
    vector<OutputScalarConfig> extremaScalars;
    // Jet-correction variation filled into this tree: empty for the nominal
    // trees, otherwise the systematic of the <tree>__<variation> tree.
    string variation;
    // Output branches booked for a variation tree; empty books every branch.
    unordered_set<string> keptBranches;
    // Scalar outputs a variation tree evaluates: the ones its kept branches
    // read, directly or through other scalars (nominal trees evaluate all).
    unordered_set<string> neededScalars;
};

// Slot of every name an expression can read as an event variable (input scalars, sample
// metadata, MC weights, scalar output names), built by resolveEngineSymbols; -1: not a variable.
struct EventVarLayout {
    unordered_map<string, int> slotByName;
    int sampleId = -1;
    int isMC = -1;
    int isSignal = -1;
    int xsection = -1;
    int lumi = -1;
    int weightPu = -1;
    int weightPuDown = -1;
    int weightPuUp = -1;
    int genWeight = -1;
    int puTrueInt = -1;
    int run = -1;
    int luminosityBlock = -1;
};

struct BranchConfig {
    vector<ScalarInputConfig> scalars;
    vector<InputCollectionConfig> collections;
    vector<TreeConfig> trees;
    EventVarLayout varLayout;
};

struct ObjectSchema {
    int id = -1;  // index into Expression::fieldIndex
    vector<string> fields;
    unordered_map<string, size_t> indexByName;
};

struct RuntimeObject {
    vector<float> values;
    TLorentzVector p4;
};

struct RuntimeCollection {
    string name;
    // Shared, immutable: the schema depends only on the configuration, not on the event.
    shared_ptr<const ObjectSchema> schema;
    vector<RuntimeObject> objects;
};

// The event variables, indexed by EventVarLayout slot. defined[slot] == 0: the name has no value
// (yet), e.g. a scalar output not computed so far in the current tree.
struct EventVars {
    vector<long double> values;
    vector<unsigned char> defined;

    void reset(size_t size) {
        values.assign(size, 0.L);
        defined.assign(size, 0);
    }
    void set(int slot, long double value) {
        values[slot] = value;
        defined[slot] = 1;
    }
    bool has(int slot) const {
        return slot >= 0 && defined[slot] != 0;
    }
};

// The event's collections: input collections by input slot (branch.json order) and runtime
// collections by runtime slot; built/active mark the runtime collections built so far / in
// progress (dependency-cycle check).
struct EventCollections {
    vector<RuntimeCollection> inputs;
    vector<RuntimeCollection> runtime;
    vector<unsigned char> built;
    vector<unsigned char> active;
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

// Input buffers for genWeight (all MC) and the LHE/PS theory weight arrays (theory samples).
// The array buffers are sized per file from the largest stored count before binding.
struct TheoryWeightBufs {
    float genWeight              = 1.f;
    int   nLHEPdfWeight          = 0;
    vector<float> LHEPdfWeight;
    int   nLHEScaleWeight        = 0;
    vector<float> LHEScaleWeight;
    int   nPSWeight              = 0;
    vector<float> PSWeight;
};

// Fixed-size output arrays written as branches to the converted ROOT trees.
struct TheoryOutBufs {
    static constexpr int kNPdf     = 101;
    static constexpr int kNAlphaS  =   2;
    static constexpr int kNScale   =   9;
    static constexpr int kNPS      =   4;
    // Number of weights stored in the source NanoAOD (provenance, e.g. 101 = no alpha_s
    // members, 8 = scale set without the nominal entry).
    Int_t nLHEPdfWeight                  = 0;
    Int_t nLHEScaleWeight                = 0;
    Int_t nPSWeight                      = 0;
    float genWeight                      = 1.f;
    float LHEPdfWeight[kNPdf]            = {};
    float LHEPdfWeightAlphaS[kNAlphaS]   = {1.f, 1.f};
    float LHEScaleWeight[kNScale]        = {};
    float PSWeight[kNPS]                 = {};
};

// Branch index of an output a variation tree does not book.
const size_t kNoBranch = numeric_limits<size_t>::max();

struct OutputTreeState {
    TreeConfig config;
    TTree* tree = nullptr;
    vector<OutputBranchRuntime> branches;
    unordered_map<string, size_t> branchIndexByName;
    // Index in branches of every (config, slot), parallel to config.regularScalars /
    // config.extremaScalars (empty for configs not booked).
    vector<vector<size_t>> regularBranches;
    vector<vector<size_t>> extremaBranches;
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

// AK4/AK8 jet energy corrections, Type-1 MET, and the jet systematic
// variations (correctionlib), configured by the convert config's
// jet_pt_correction block; see JetPtCorrector for the correction chain.
struct JetPtCorrectionConfig {
    bool enabled = false;
    // "hlt_jec": official Winter24HLT MC-truth JEC on the raw jets (default).
    // "scouting_to_offline": legacy ad-hoc scouting->offline response SF from
    // corrections_file on the stored jets.
    string nominalCorrection = "hlt_jec";
    string jecAk4File;           // Winter24HLT jetHLT_jerc.json.gz (AK4PFHLT)
    string jecAk8File;           // Winter24HLT fatJetHLT_jerc.json.gz (AK8PFHLT)
    string jecAk4Name = "Winter24HLT_V1_MC_L1L2L3Res_AK4PFHLT";
    string jecAk8Name = "Winter24HLT_V1_MC_L1L2L3Res_AK8PFHLT";
    string jecAk4L1Name = "Winter24HLT_V1_MC_L1FastJet_AK4PFHLT";
    string jecAk8L1Name = "Winter24HLT_V1_MC_L1FastJet_AK8PFHLT";
    string correctionsFile;      // scoutingPUPPI_corrections.json.gz (scouting_to_offline)
    double ak4TagThreshold = 0.5;
    double ak8TagThreshold = 0.5;
    string jesJerFile;           // JME-POG jet_jerc.json.gz (JER resolution + SF)
    string jerSmearFile;         // JME-POG jer_smear.json.gz (JERSmear tool)
    string jerResolutionName = "Summer23BPixPrompt23_RunD_JRV1_MC_PtResolution_AK4PFPuppi";
    string jerScaleFactorName = "Summer23BPixPrompt23_RunD_JRV1_MC_ScaleFactor_AK4PFPuppi";
    // Flat per-jet JES uncertainty (fraction) of jes_up/jes_down.
    double jesShift = 0.10;
    // JMS/JMR of the MC AK8 soft-drop mass from selections/jms_jmr/.
    bool applyJmsJmr = false;
    string jmsJmrResultsFile;
    // Systematic variations written as <tree>__<variation> trees (MC only),
    // each keeping the per-tree branch list of variationBranches.
    vector<string> variations;
    unordered_map<string, vector<string>> variationBranches;
    // Validation only: fill the nominal trees with this configuration.
    string debugNominalConfiguration = "nominal";
};

// Parameters of one correction configuration (the nominal corrections or one
// systematic variation of them); every configuration is evaluated per event.
struct CorrectionConfiguration {
    // Variation written by this configuration: empty for the nominal trees,
    // otherwise the <tree>__<variation> suffix.
    string variation;
    double jesFactor = 1.;
    int jerSfIndex = 0;          // JER scale factor: 0 nom, 1 up, 2 down
    double jms = 1.;
    double jmr = 1.;
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
    bool updateRawEntries = true;
    vector<SampleRuleConfig> sampleRules;
    string puWeightPathPattern;
    JetPtCorrectionConfig jetPtCorrection;
};

struct BatchRequest {
    bool printBatchCount = false;
    bool mergeSuccessfulBatches = false;
    bool updateGenWeightMean = false;
    bool singleBatch = false;
    size_t batchIndex = 0;
};

struct BatchTempCollection {
    vector<string> paths;
    Long64_t rawEntries = 0;
    long double sumWeightPu = 0.L;
    long double sumWeightPuUp = 0.L;
    long double sumWeightPuDown = 0.L;
    // MC: genWeight sums over every processed generated event (before any selection), the
    // denominators of the signed, absolutely normalized MC event weights downstream.
    long double sumGenWeight = 0.L;
    long double sumGenWeightPu = 0.L;
    long double sumGenWeightPuUp = 0.L;
    long double sumGenWeightPuDown = 0.L;
    vector<string> skippedFiles;
    set<pair<UInt_t, UInt_t>> lumis;
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
    const EventVars* vars = nullptr;
    const EventCollections* collections = nullptr;
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
    // Kind::P4 only. TLorentzVector is a TObject; holding one inline made every (mostly numeric)
    // Value large and expensive to construct, which was a measurable part of the event loop.
    shared_ptr<const TLorentzVector> p4;
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
        node->op = identifierOp(name);
        node->text = name;
        return node;
    }

    ExprPtr makeUnary(const string& op, ExprPtr arg) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Unary;
        node->op = operatorOp(op);
        node->text = op;
        node->lhs = std::move(arg);
        return node;
    }

    ExprPtr makeBinary(const string& op, ExprPtr lhs, ExprPtr rhs) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Binary;
        node->op = operatorOp(op);
        node->text = op;
        node->lhs = std::move(lhs);
        node->rhs = std::move(rhs);
        return node;
    }

    ExprPtr makeCall(const string& name, vector<ExprPtr> args) {
        auto node = make_shared<Expression>();
        node->kind = ExprKind::Call;
        node->op = functionOp(name);
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

void validateJetPtCorrectionConfig(const JetPtCorrectionConfig& jpc) {
    const bool hltJec = (jpc.nominalCorrection == "hlt_jec");
    if (!hltJec && jpc.nominalCorrection != "scouting_to_offline") {
        throw runtime_error("jet_pt_correction.nominal_correction must be 'hlt_jec' or "
                            "'scouting_to_offline', got '" + jpc.nominalCorrection + "'");
    }
    // The AK4 production JEC provides the event rho in both modes.
    if (jpc.jecAk4File.empty()) {
        throw runtime_error("jet_pt_correction.jec_ak4_file is required");
    }
    if (hltJec && jpc.jecAk8File.empty()) {
        throw runtime_error("jet_pt_correction.jec_ak8_file is required for nominal_correction = hlt_jec");
    }
    if (!hltJec && jpc.correctionsFile.empty()) {
        throw runtime_error("jet_pt_correction.corrections_file is required for "
                            "nominal_correction = scouting_to_offline");
    }
    if (jpc.jesJerFile.empty() || jpc.jerSmearFile.empty()) {
        throw runtime_error("jet_pt_correction.jes_jer_file and jer_smear_file are required "
                            "(JER smearing of the MC jets)");
    }
    if (jpc.applyJmsJmr && jpc.jmsJmrResultsFile.empty()) {
        throw runtime_error("jet_pt_correction.apply_jms_jmr requires jms_jmr_results_file");
    }
    const auto checkVariation = [&](const string& name, const string& key) {
        if (find(kJetVariationNames.begin(), kJetVariationNames.end(), name) == kJetVariationNames.end()) {
            throw runtime_error(key + " has unknown variation '" + name + "'");
        }
        if (!jpc.applyJmsJmr && (startsWith(name, "jms_") || startsWith(name, "jmr_"))) {
            throw runtime_error(key + " variation '" + name + "' requires apply_jms_jmr = true");
        }
    };
    unordered_set<string> seen;
    for (const auto& variation : jpc.variations) {
        checkVariation(variation, "jet_pt_correction.variations");
        if (!seen.insert(variation).second) {
            throw runtime_error("jet_pt_correction.variations lists '" + variation + "' twice");
        }
    }
    if (jpc.debugNominalConfiguration != "nominal") {
        checkVariation(jpc.debugNominalConfiguration, "jet_pt_correction.debug_nominal_configuration");
    }
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
    config.updateRawEntries = payload.getBoolOr("update_raw_entries", true);

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
    if (payload.contains("jet_pt_correction")) {
        const JsonValue& jc = payload.at("jet_pt_correction");
        // Keys of the removed one-variation-per-conversion scheme.
        for (const char* removedKey : {"variation", "jes_unc_name", "jer_rho_fallback"}) {
            if (jc.contains(removedKey)) {
                throw runtime_error(string("jet_pt_correction.") + removedKey +
                                    " was removed: one conversion now writes every variation "
                                    "(jet_pt_correction.variations, see README)");
            }
        }
        JetPtCorrectionConfig& jpc = config.jetPtCorrection;
        jpc.enabled = jc.getBoolOr("enabled", true);
        jpc.nominalCorrection = jc.getStringOr("nominal_correction", jpc.nominalCorrection);
        jpc.jecAk4File = resolveReferencedPath(config.configPath,
                                                jc.getStringOr("jec_ak4_file", ""));
        jpc.jecAk8File = resolveReferencedPath(config.configPath,
                                                jc.getStringOr("jec_ak8_file", ""));
        jpc.jecAk4Name = jc.getStringOr("jec_ak4_name", jpc.jecAk4Name);
        jpc.jecAk8Name = jc.getStringOr("jec_ak8_name", jpc.jecAk8Name);
        jpc.jecAk4L1Name = jc.getStringOr("jec_ak4_l1_name", jpc.jecAk4L1Name);
        jpc.jecAk8L1Name = jc.getStringOr("jec_ak8_l1_name", jpc.jecAk8L1Name);
        jpc.correctionsFile = resolveReferencedPath(config.configPath,
                                                     jc.getStringOr("corrections_file", ""));
        jpc.ak4TagThreshold = static_cast<double>(jc.getNumberOr("ak4_tag_threshold", jpc.ak4TagThreshold));
        jpc.ak8TagThreshold = static_cast<double>(jc.getNumberOr("ak8_tag_threshold", jpc.ak8TagThreshold));
        jpc.jesJerFile = resolveReferencedPath(config.configPath,
                                                jc.getStringOr("jes_jer_file", ""));
        jpc.jerSmearFile = resolveReferencedPath(config.configPath,
                                                  jc.getStringOr("jer_smear_file", ""));
        jpc.jerResolutionName = jc.getStringOr("jer_resolution_name", jpc.jerResolutionName);
        jpc.jerScaleFactorName = jc.getStringOr("jer_scale_factor_name", jpc.jerScaleFactorName);
        jpc.jesShift = static_cast<double>(jc.getNumberOr("jes_shift", jpc.jesShift));
        jpc.applyJmsJmr = jc.getBoolOr("apply_jms_jmr", jpc.applyJmsJmr);
        jpc.jmsJmrResultsFile = resolveReferencedPath(config.configPath,
                                                       jc.getStringOr("jms_jmr_results_file", ""));
        if (jc.contains("variations")) {
            jpc.variations = jc.at("variations").toStringArray();
        }
        if (jc.contains("variation_branches")) {
            for (const auto& item : jc.at("variation_branches").asObject()) {
                jpc.variationBranches[item.first] = item.second.toStringArray();
            }
        }
        jpc.debugNominalConfiguration = jc.getStringOr("debug_nominal_configuration",
                                                       jpc.debugNominalConfiguration);
        if (jpc.enabled) {
            validateJetPtCorrectionConfig(jpc);
        }
    }
    return config;
}

// Set a numeric field of one sample object in sample.json: replace the value when the
// key exists, otherwise insert it on a new line after "raw_entries" (same indentation).
// Holds an exclusive flock on <sample.json>.lock and writes through a per-process
// temporary file, so concurrent per-sample jobs cannot lose each other's updates.
void writeSampleNumericField(const string& sampleConfigPath,
                             const string& sampleName,
                             const string& key,
                             const string& valueText) {
    const string lockPath = sampleConfigPath + ".lock";
    const int lockFd = open(lockPath.c_str(), O_CREAT | O_RDWR, 0644);
    if (lockFd < 0 || flock(lockFd, LOCK_EX) != 0) {
        if (lockFd >= 0) {
            close(lockFd);
        }
        throw runtime_error("Cannot lock sample config via " + lockPath);
    }
    try {
        ifstream fin(sampleConfigPath);
        if (!fin) {
            throw runtime_error("Cannot open sample config for " + key + " update: " + sampleConfigPath);
        }
        const string content((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
        fin.close();

        const size_t sampleKeyPos = content.find("\"sample\"");
        const size_t colonPos = (sampleKeyPos == string::npos) ? string::npos : content.find(':', sampleKeyPos);
        const size_t arrayPos = (colonPos == string::npos) ? string::npos : skipWhitespace(content, colonPos + 1);
        if (arrayPos == string::npos || arrayPos >= content.size() || content[arrayPos] != '[') {
            throw runtime_error("Cannot find 'sample' array in sample config: " + sampleConfigPath);
        }
        const size_t arrayEnd = findMatchingJsonDelimiter(content, arrayPos, '[', ']');
        const regex namePattern("\"name\"\\s*:\\s*\"([^\"]+)\"");
        const regex keyPattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]*)?(?:[eE][-+]?[0-9]+)?)");
        const regex rawEntriesPattern("\"raw_entries\"\\s*:\\s*-?[0-9]+(?:\\.[0-9]+)?");
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
                smatch keyMatch;
                smatch rawMatch;
                if (regex_search(objectText, keyMatch, keyPattern) && keyMatch.size() >= 2) {
                    updated.replace(pos + static_cast<size_t>(keyMatch.position(1)), keyMatch.length(1), valueText);
                } else if (regex_search(objectText, rawMatch, rawEntriesPattern)) {
                    const size_t keyStart = pos + static_cast<size_t>(rawMatch.position(0));
                    const size_t lineStart = content.rfind('\n', keyStart);
                    const string indent = (lineStart == string::npos)
                        ? string() : content.substr(lineStart + 1, keyStart - lineStart - 1);
                    const size_t insertPos = keyStart + static_cast<size_t>(rawMatch.length(0));
                    updated.insert(insertPos, ",\n" + indent + "\"" + key + "\": " + valueText);
                } else {
                    throw runtime_error("Cannot find raw_entries for sample '" + sampleName +
                                        "' in sample config: " + sampleConfigPath);
                }
                foundSample = true;
                break;
            }
            pos = objectEnd + 1;
        }
        if (!foundSample) {
            throw runtime_error("Cannot find sample '" + sampleName + "' in sample config: " + sampleConfigPath);
        }

        const fs::path targetPath(sampleConfigPath);
        const fs::path tempPath = targetPath.string() + ".tmp." + to_string(static_cast<long long>(getpid()));
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
    } catch (...) {
        flock(lockFd, LOCK_UN);
        close(lockFd);
        throw;
    }
    flock(lockFd, LOCK_UN);
    close(lockFd);
}

// raw_entries goes through the same locked writer, so concurrent merges (and
// --update-genweight-mean jobs) cannot lose each other's sample.json updates.
void writeSampleRawEntries(const string& sampleConfigPath,
                           const string& sampleName,
                           Long64_t rawEntries) {
    writeSampleNumericField(sampleConfigPath, sampleName, "raw_entries",
                            to_string(static_cast<long long>(rawEntries)));
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
        scalar.optional = node.getBoolOr("optional", false);
        config.scalars.push_back(std::move(scalar));
    }

    for (const auto& node : payload.at("input").at("collections").asArray()) {
        InputCollectionConfig collection;
        collection.name = node.at("name").asString();
        collection.sizeName = node.at("size").asString();
        collection.maxSize = node.at("max_size").asInt();
        const bool collectionOptional = node.getBoolOr("optional", false);
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
            field.optional = fieldNode.getBoolOr("optional", collectionOptional);
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
        const auto inserted = config.collectionSlotByName.emplace(collection.name, static_cast<int>(config.collections.size()));
        if (inserted.second) {
            config.collections.push_back(std::move(collection));
        } else {
            config.collections[inserted.first->second] = std::move(collection);
        }
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

float getObjectField(const RuntimeCollection& collection, const RuntimeObject& object, const string& fieldName, float defaultValue = def) {
    const auto it = collection.schema->indexByName.find(fieldName);
    if (it == collection.schema->indexByName.end()) {
        return defaultValue;
    }
    return object.values[it->second];
}

// fieldMap[k]: index in sourceObject.values of target field k, or -1 if the source lacks it.
RuntimeObject remapObject(const RuntimeObject& sourceObject, const vector<int>& fieldMap) {
    RuntimeObject out;
    out.values.assign(fieldMap.size(), def);
    out.p4 = sourceObject.p4;
    for (size_t index = 0; index < fieldMap.size(); ++index) {
        if (fieldMap[index] >= 0) {
            out.values[index] = sourceObject.values[fieldMap[index]];
        }
    }
    return out;
}

RuntimeCollection mergeCollections(const RuntimeCollectionConfig& config, const vector<const RuntimeCollection*>& collections) {
    RuntimeCollection merged;
    merged.name = config.name;
    merged.schema = config.mergedSchema;

    for (size_t child = 0; child < collections.size(); ++child) {
        for (const auto& object : collections[child]->objects) {
            merged.objects.push_back(remapObject(object, config.mergeFieldMaps[child]));
        }
    }

    return merged;
}

// Startup name resolution for the expression engine. The field schemas (with their ids), the
// event-variable slots and the collection slots depend only on the configuration, so every
// identifier is resolved here once and the event loop indexes arrays instead of hashing names.
// The merged field list of a merge collection keeps the first-occurrence order over its
// children's fields. Unknown sources/children/dedup references and merge cycles are reported here.
void resolveEngineSymbols(SelectionConfig& selectionConfig, BranchConfig& branchConfig) {
    vector<const ObjectSchema*> schemas;
    const auto registerSchema = [&](ObjectSchema schema) {
        schema.id = static_cast<int>(schemas.size());
        auto shared = make_shared<const ObjectSchema>(std::move(schema));
        schemas.push_back(shared.get());
        return shared;
    };
    const auto findSlot = [](const unordered_map<string, int>& slots, const string& name) {
        const auto it = slots.find(name);
        return (it != slots.end()) ? it->second : -1;
    };

    // Input collections (a repeated name resolves to the last one, as with the former name map).
    unordered_map<string, int> inputSlotByName;
    for (size_t slot = 0; slot < branchConfig.collections.size(); ++slot) {
        InputCollectionConfig& input = branchConfig.collections[slot];
        input.schema = registerSchema(makeSchemaFromCollection(input));
        inputSlotByName[input.name] = static_cast<int>(slot);
    }

    // Runtime collections.
    const auto runtimeSlotOf = [&](const string& name) {
        const int slot = findSlot(selectionConfig.collectionSlotByName, name);
        if (slot < 0) {
            throw runtime_error("Unknown runtime collection: " + name);
        }
        return slot;
    };
    vector<shared_ptr<const ObjectSchema>> runtimeSchemas(selectionConfig.collections.size());
    vector<unsigned char> active(selectionConfig.collections.size(), 0);
    function<shared_ptr<const ObjectSchema>(int)> resolveCollection = [&](int slot) {
        if (runtimeSchemas[slot]) {
            return runtimeSchemas[slot];
        }
        RuntimeCollectionConfig& config = selectionConfig.collections[slot];
        if (!config.source.empty()) {
            config.sourceSlot = findSlot(inputSlotByName, config.source);
            if (config.sourceSlot < 0) {
                throw runtime_error("Unknown input collection source: " + config.source);
            }
            runtimeSchemas[slot] = branchConfig.collections[config.sourceSlot].schema;
        } else if (!config.merge.empty()) {
            if (active[slot]) {
                throw runtime_error("Collection dependency cycle detected at: " + config.name);
            }
            active[slot] = 1;
            vector<shared_ptr<const ObjectSchema>> children;
            vector<string> mergedFields;
            unordered_set<string> seen;
            config.mergeSlots.clear();
            for (const auto& childName : config.merge) {
                config.mergeSlots.push_back(runtimeSlotOf(childName));
                children.push_back(resolveCollection(config.mergeSlots.back()));
                for (const auto& field : children.back()->fields) {
                    if (seen.insert(field).second) {
                        mergedFields.push_back(field);
                    }
                }
            }
            active[slot] = 0;
            config.mergedSchema = registerSchema(makeSchema(mergedFields));
            config.mergeFieldMaps.clear();
            for (const auto& child : children) {
                vector<int> fieldMap(mergedFields.size(), -1);
                for (size_t index = 0; index < mergedFields.size(); ++index) {
                    const auto it = child->indexByName.find(mergedFields[index]);
                    if (it != child->indexByName.end()) {
                        fieldMap[index] = static_cast<int>(it->second);
                    }
                }
                config.mergeFieldMaps.push_back(std::move(fieldMap));
            }
            runtimeSchemas[slot] = config.mergedSchema;
        } else {
            throw runtime_error("Runtime collection must define source or merge: " + config.name);
        }
        return runtimeSchemas[slot];
    };
    for (size_t slot = 0; slot < selectionConfig.collections.size(); ++slot) {
        resolveCollection(static_cast<int>(slot));
        RuntimeCollectionConfig& config = selectionConfig.collections[slot];
        if (config.dedupExpr && !config.dedupCollection.empty()) {
            config.dedupSlot = runtimeSlotOf(config.dedupCollection);
        }
    }
    selectionConfig.buildOrder.clear();
    for (const auto& name : selectionConfig.collectionOrder) {
        selectionConfig.buildOrder.push_back(runtimeSlotOf(name));
    }

    // Event variables: the input scalars, the sample metadata, the MC weights and every scalar
    // output name (written while the tree is filled, readable by the later formulas).
    EventVarLayout& layout = branchConfig.varLayout;
    layout = EventVarLayout();
    const auto addVar = [&](const string& name) {
        return layout.slotByName.emplace(name, static_cast<int>(layout.slotByName.size())).first->second;
    };
    unordered_map<string, int> scalarIndexByName;
    for (size_t index = 0; index < branchConfig.scalars.size(); ++index) {
        branchConfig.scalars[index].varSlot = addVar(branchConfig.scalars[index].name);
        scalarIndexByName[branchConfig.scalars[index].name] = static_cast<int>(index);
    }
    layout.sampleId = addVar("sample_ID");
    layout.isMC = addVar("is_MC");
    layout.isSignal = addVar("is_signal");
    layout.xsection = addVar("xsection");
    layout.lumi = addVar("lumi");
    layout.weightPu = addVar("weight_pu");
    layout.weightPuDown = addVar("weight_pu_down");
    layout.weightPuUp = addVar("weight_pu_up");
    layout.genWeight = addVar("genWeight");
    for (auto& tree : branchConfig.trees) {
        for (auto* group : {&tree.regularScalars, &tree.extremaScalars}) {
            for (auto& config : *group) {
                if (config.collection.empty()) {
                    config.varSlot = addVar(config.name);
                }
            }
        }
    }
    layout.puTrueInt = findSlot(layout.slotByName, "Pileup_nTrueInt");
    layout.run = findSlot(layout.slotByName, "run");
    layout.luminosityBlock = findSlot(layout.slotByName, "luminosityBlock");
    for (auto& input : branchConfig.collections) {
        input.sizeSlot = findSlot(layout.slotByName, input.sizeName);
    }

    // Identifiers.
    function<void(const ExprPtr&)> annotate = [&](const ExprPtr& expr) {
        if (!expr) {
            return;
        }
        if (expr->kind == ExprKind::Identifier) {
            expr->varSlot = findSlot(layout.slotByName, expr->text);
            expr->runtimeSlot = findSlot(selectionConfig.collectionSlotByName, expr->text);
            expr->inputSlot = findSlot(inputSlotByName, expr->text);
            expr->fieldIndex.assign(schemas.size(), -1);
            for (const ObjectSchema* schema : schemas) {
                const auto it = schema->indexByName.find(expr->text);
                if (it != schema->indexByName.end()) {
                    expr->fieldIndex[schema->id] = static_cast<int>(it->second);
                }
            }
            expr->resolved = true;
        }
        annotate(expr->lhs);
        annotate(expr->rhs);
        for (const auto& arg : expr->args) {
            annotate(arg);
        }
    };
    annotate(selectionConfig.eventPreselection);
    for (auto& config : selectionConfig.collections) {
        annotate(config.selectionExpr);
        annotate(config.dedupExpr);
        annotate(config.sortRule.expr);
    }
    for (auto& item : selectionConfig.treeSelections) {
        annotate(item.second);
    }
    for (auto& tree : branchConfig.trees) {
        for (auto* group : {&tree.regularScalars, &tree.extremaScalars}) {
            for (auto& config : *group) {
                annotate(config.formula);
                if (!config.collection.empty()) {
                    config.collectionSlot = findSlot(selectionConfig.collectionSlotByName, config.collection);
                } else if (config.formula && config.formula->kind == ExprKind::Identifier) {
                    config.exactScalarIndex = findSlot(scalarIndexByName, config.formula->text);
                }
            }
        }
    }
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
    // Outside the pileup histogram range the data profile carries no probability, so the
    // event gets weight 0 (the old 1.0 gave full weight to a region data does not populate).
    return 0.0L;
}

// Bind genWeight on an MC input tree (every MC sample, not only theory samples: downstream
// event weights use its sign/magnitude). A missing branch is an error.
void bindGenWeight(TTree* tree, TheoryWeightBufs& buf) {
    if (tree->GetBranch("genWeight") == nullptr) {
        throw runtime_error("MC input tree has no genWeight branch");
    }
    tree->SetBranchStatus("genWeight", 1);
    tree->SetBranchAddress("genWeight", &buf.genWeight);
    tree->AddBranchToCache("genWeight", true);
}

// Enable and bind the theory weight arrays on an input TTree. Silently skips missing
// branches. Each array buffer is sized from the largest count stored in this file, so
// samples with more weights than usual (e.g. 44 PS weights) cannot overflow it.
void activateTheoryInputBranches(TTree* tree, TheoryWeightBufs& buf) {
    const auto bindArray = [&](const char* countName, int* countAddr,
                               const char* arrayName, vector<float>& values, Long64_t minSize) {
        if (tree->GetBranch(countName) == nullptr || tree->GetBranch(arrayName) == nullptr) {
            return;
        }
        // The count branch must be enabled before GetMaximum (a disabled branch reads as 0).
        tree->SetBranchStatus(countName, 1);
        tree->SetBranchStatus(arrayName, 1);
        const Long64_t maxCount = max<Long64_t>(minSize, llround(tree->GetMaximum(countName)));
        values.assign(static_cast<size_t>(maxCount), 1.f);
        tree->SetBranchAddress(countName, countAddr);
        tree->SetBranchAddress(arrayName, values.data());
        tree->AddBranchToCache(countName, true);
        tree->AddBranchToCache(arrayName, true);
    };
    // Lower bounds = the previous fixed buffer sizes.
    bindArray("nLHEPdfWeight",   &buf.nLHEPdfWeight,   "LHEPdfWeight",   buf.LHEPdfWeight,   200);
    bindArray("nLHEScaleWeight", &buf.nLHEScaleWeight, "LHEScaleWeight", buf.LHEScaleWeight, 20);
    bindArray("nPSWeight",       &buf.nPSWeight,       "PSWeight",       buf.PSWeight,       10);
}

// Create fixed-size array branches on an output tree, pointed at treeState.theoryOutBuf.
// genWeight itself is written by branch.json for every MC tree; it is only added here when a
// tree does not define it, so no tree ends up with two branches named genWeight.
void setupTheoryOutputBranches(OutputTreeState& treeState) {
    if (treeState.tree->GetBranch("genWeight") == nullptr) {
        treeState.tree->Branch("genWeight",  &treeState.theoryOutBuf.genWeight,       "genWeight/F");
    }
    treeState.tree->Branch("nLHEPdfWeight",   &treeState.theoryOutBuf.nLHEPdfWeight,   "nLHEPdfWeight/I");
    treeState.tree->Branch("nLHEScaleWeight", &treeState.theoryOutBuf.nLHEScaleWeight, "nLHEScaleWeight/I");
    treeState.tree->Branch("nPSWeight",       &treeState.theoryOutBuf.nPSWeight,       "nPSWeight/I");
    treeState.tree->Branch("LHEPdfWeight",    treeState.theoryOutBuf.LHEPdfWeight,    "LHEPdfWeight[101]/F");
    treeState.tree->Branch("LHEPdfWeightAlphaS", treeState.theoryOutBuf.LHEPdfWeightAlphaS, "LHEPdfWeightAlphaS[2]/F");
    treeState.tree->Branch("LHEScaleWeight",  treeState.theoryOutBuf.LHEScaleWeight,  "LHEScaleWeight[9]/F");
    treeState.tree->Branch("PSWeight",        treeState.theoryOutBuf.PSWeight,        "PSWeight[4]/F");
    treeState.hasTheoryBranches = true;
}

// Copy input theory weight buffers to an output struct, padding unused slots with 1.0.
// An 8-entry scale set (nominal omitted, NanoAOD order (muR,muF) without (1,1)) is stored in
// the standard 9-entry layout with 1.0 inserted at the nominal index 4.
void copyTheoryWeights(const TheoryWeightBufs& src, TheoryOutBufs& dst) {
    dst.genWeight = src.genWeight;
    dst.nLHEPdfWeight = src.nLHEPdfWeight;
    dst.nLHEScaleWeight = src.nLHEScaleWeight;
    dst.nPSWeight = src.nPSWeight;
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
    if (src.nLHEScaleWeight == TheoryOutBufs::kNScale - 1) {
        for (int i = 0; i < 4; ++i) dst.LHEScaleWeight[i] = src.LHEScaleWeight[i];
        dst.LHEScaleWeight[4] = 1.f;
        for (int i = 4; i < 8; ++i) dst.LHEScaleWeight[i + 1] = src.LHEScaleWeight[i];
    } else {
        const int nScale = min(src.nLHEScaleWeight, TheoryOutBufs::kNScale);
        for (int i = 0; i < nScale; ++i) dst.LHEScaleWeight[i] = src.LHEScaleWeight[i];
        for (int i = nScale; i < TheoryOutBufs::kNScale; ++i) dst.LHEScaleWeight[i] = 1.f;
    }
    const int nPS = min(src.nPSWeight, TheoryOutBufs::kNPS);
    for (int i = 0; i < nPS; ++i) dst.PSWeight[i] = src.PSWeight[i];
    for (int i = nPS; i < TheoryOutBufs::kNPS; ++i) dst.PSWeight[i] = 1.f;
}

// The event variables every expression can read: the input scalars, the sample metadata and,
// for MC, the pileup weights and genWeight.
void fillEventVars(EventVars& vars,
                   const BranchConfig& branchConfig,
                   const SampleMeta& sampleMeta,
                   const vector<PileupBin>* pileupWeights = nullptr,
                   const TheoryWeightBufs* theoryBufs = nullptr) {
    const EventVarLayout& layout = branchConfig.varLayout;
    vars.reset(layout.slotByName.size());
    for (const auto& scalar : branchConfig.scalars) {
        vars.set(scalar.varSlot, scalar.numericValue());
    }
    vars.set(layout.sampleId, sampleMeta.sampleId);
    vars.set(layout.isMC, sampleMeta.isMC ? 1. : 0.);
    vars.set(layout.isSignal, sampleMeta.isSignal ? 1. : 0.);
    vars.set(layout.xsection, sampleMeta.xsection);
    vars.set(layout.lumi, sampleMeta.lumi);
    if (sampleMeta.isMC) {
        const float puValue = vars.has(layout.puTrueInt) ? static_cast<float>(vars.values[layout.puTrueInt]) : 0.f;
        if (pileupWeights != nullptr && !pileupWeights->empty()) {
            vars.set(layout.weightPu, lookupPileupWeight(*pileupWeights, puValue, 0));
            vars.set(layout.weightPuDown, lookupPileupWeight(*pileupWeights, puValue, 1));
            vars.set(layout.weightPuUp, lookupPileupWeight(*pileupWeights, puValue, 2));
        } else {
            vars.set(layout.weightPu, 1.);
            vars.set(layout.weightPuDown, 1.);
            vars.set(layout.weightPuUp, 1.);
        }
        // theoryBufs holds the bound genWeight for every MC sample (see bindGenWeight).
        vars.set(layout.genWeight, theoryBufs ? static_cast<long double>(theoryBufs->genWeight) : 1.L);
    }
}

// The event-variable slots an expression reads.
void collectVarSlots(const ExprPtr& expr, set<int>& slots) {
    if (!expr) {
        return;
    }
    if (expr->kind == ExprKind::Identifier && expr->varSlot >= 0) {
        slots.insert(expr->varSlot);
    }
    collectVarSlots(expr->lhs, slots);
    collectVarSlots(expr->rhs, slots);
    for (const auto& arg : expr->args) {
        collectVarSlots(arg, slots);
    }
}

long double requireEventVar(const EventVars& vars, int slot, const char* name) {
    if (!vars.has(slot)) {
        throw runtime_error(string("Event variable not available: ") + name);
    }
    return vars.values[slot];
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

RuntimeCollection buildInputCollection(const InputCollectionConfig& config, const EventVars& vars) {
    if (!vars.has(config.sizeSlot)) {
        throw runtime_error("Input collection size not found: " + config.sizeName);
    }

    RuntimeCollection collection;
    collection.name = config.name;
    collection.schema = config.schema;

    const int size = min(static_cast<int>(vars.values[config.sizeSlot]), config.maxSize);
    collection.objects.reserve(size);
    for (int index = 0; index < size; ++index) {
        RuntimeObject object;
        object.values.reserve(config.fields.size());
        for (const auto& field : config.fields) {
            object.values.push_back(field.valueAt(index));
        }
        object.p4 = buildObjectP4(config, index);
        collection.objects.push_back(std::move(object));
    }

    return collection;
}

// A runtime collection already built in this event, else an input collection of that name.
const RuntimeCollection* findCollection(const EvalContext& context, const Expression& identifier) {
    if (!context.collections) {
        return nullptr;
    }
    if (identifier.runtimeSlot >= 0 && context.collections->built[identifier.runtimeSlot]) {
        return &context.collections->runtime[identifier.runtimeSlot];
    }
    if (identifier.inputSlot >= 0) {
        return &context.collections->inputs[identifier.inputSlot];
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
    out.p4 = make_shared<const TLorentzVector>(p4);
    return out;
}

long double toNumber(const Value& value) {
    if (value.kind != Value::Kind::Number) {
        throw runtime_error("Numeric value expected in expression");
    }
    return value.number;
}

// The reference points into value (or the object it refers to): use it within the lifetime of value.
const TLorentzVector& toP4(const Value& value) {
    if (value.kind == Value::Kind::P4) {
        return *value.p4;
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

double pairMetric(bool deltaPhi, const TLorentzVector& lhs, const TLorentzVector& rhs) {
    if (deltaPhi) {
        // |dphi|: pair_min/max_deltaPhi summarise the smallest/largest angular separation;
        // the signed TLorentzVector::DeltaPhi made them the most negative/positive value.
        return fabs(lhs.DeltaPhi(rhs));
    }
    return lhs.DeltaR(rhs);
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
double evalEventShape(Op op, const string& name, const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.empty()) {
        throw runtime_error(name + " requires at least one object or collection argument");
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
    if (op == Op::Sphericity) return 1.5 * (l2 + l3);
    if (op == Op::Aplanarity) return 1.5 * l3;
    if (op == Op::Planarity) return l2 - l3;
    throw runtime_error("Unsupported event-shape: " + name);
}

Value evalAggregation(Op op,
                      const string& name,
                      const vector<ExprPtr>& args,
                      const EvalContext& context) {
    if (args.size() < 2) {
        throw runtime_error(name + " requires at least 2 arguments");
    }

    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    if (op == Op::Sum) {
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
        if (!found || (op == Op::MaxValue && value > best) || (op == Op::MinValue && value < best)) {
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

// value_at(collection, index_expr, value_expr [, default]): evaluate value_expr
// in the context of collection[index], where index is the (rounded) result of
// index_expr in the CURRENT context. Returns default (or `def`) when the index
// is out of range -- e.g. an unmatched gen index of -1, or one beyond the
// collection's max_size. Used for cross-collection gen matching, e.g.
// value_at(GenPart, ScoutingMuonVtx_genPartIdx, GenPart_pdgId, 0).
Value evalValueAt(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 3) {
        throw runtime_error("value_at requires collection, index expression, and value expression arguments");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const long double defaultValue = (args.size() >= 4) ? evalNumber(args[3], context) : def;
    const int index = static_cast<int>(llround(evalNumber(args[1], context)));
    if (index < 0 || index >= static_cast<int>(collection->objects.size())) {
        return makeNumberValue(defaultValue);
    }
    EvalContext loop = context;
    loop.currentCollection = collection;
    loop.currentObject = &collection->objects[index];
    return makeNumberValue(evalNumber(args[2], loop));
}

// first_ancestor_index(collection, index_expr, pdgid_field, mother_index_field [, default]):
// Starting at collection[index_expr], walk the mother chain (mother_index_field) while the
// mother's pdgid_field is IDENTICAL to the starting object's own pdgid_field -- i.e. skip
// PYTHIA/parton-shower "self-copy" bookkeeping entries inserted whenever a particle radiates
// (e.g. a muon's immediate GenPart mother is often a pre-FSR copy of the same muon) -- and
// return the index of the first ancestor whose pdgId differs. Returns default (or `def`) if
// index_expr is out of range, or if the chain runs off the end (mother index out of range)
// before a differing pdgId is found. Typical use: pair with value_at to resolve the true
// production vertex of a gen-matched lepton, e.g.
//   value_at(GenPart, first_ancestor_index(GenPart, ScoutingMuonVtx_genPartIdx, GenPart_pdgId,
//                                           GenPart_genPartIdxMother, -1), GenPart_pdgId, 0)
Value evalFirstAncestorIndex(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 4) {
        throw runtime_error(
            "first_ancestor_index requires collection, index expression, pdgId field "
            "expression, and mother-index field expression (plus optional default)");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const long double defaultValue = (args.size() >= 5) ? evalNumber(args[4], context) : def;
    const int size = static_cast<int>(collection->objects.size());
    const int startIndex = static_cast<int>(llround(evalNumber(args[1], context)));
    if (startIndex < 0 || startIndex >= size) {
        return makeNumberValue(defaultValue);
    }

    auto fieldAt = [&](int index, const ExprPtr& fieldExpr) -> long double {
        EvalContext loop = context;
        loop.currentCollection = collection;
        loop.currentObject = &collection->objects[index];
        return evalNumber(fieldExpr, loop);
    };

    const long double startPdgId = fieldAt(startIndex, args[2]);
    int current = startIndex;
    // Cap the walk so a malformed/cyclic mother chain can't hang the conversion.
    for (int step = 0; step <= size; ++step) {
        const int motherIndex = static_cast<int>(llround(fieldAt(current, args[3])));
        if (motherIndex < 0 || motherIndex >= size) {
            return makeNumberValue(defaultValue);
        }
        if (fieldAt(motherIndex, args[2]) != startPdgId) {
            return makeNumberValue(static_cast<long double>(motherIndex));
        }
        current = motherIndex;
    }
    return makeNumberValue(defaultValue);
}

// first_nonqg_ancestor_index(collection, index_expr, pdgid_field, mother_index_field [, default]):
// Starting at collection[index_expr], walk the mother chain (mother_index_field) while the
// CURRENT node's pdgid_field is a quark (|pdgId| in 1..6) or gluon (pdgId == 21) -- i.e. keep
// climbing past intermediate partons and parton-shower radiation (self-copies, gluon splittings,
// additional emissions) -- and return the index of the first ancestor whose pdgId is NOT a quark
// or gluon. This differs from first_ancestor_index (which only skips same-pdgId self-copies): it
// also skips through genuinely different quark/gluon flavors introduced by radiation, so it can
// trace a jet's matched parton back to the hard-process resonance (e.g. a W boson) even when
// there were intermediate FSR emissions. Returns default (or `def`) if index_expr is out of
// range, or if the chain runs off the end before a non-quark/gluon ancestor is found.
// Typical use: pair with value_at to find the hard-process origin of a gen-matched jet, e.g.
//   value_at(GenPart, first_nonqg_ancestor_index(GenPart, ScoutingFatPFJetRecluster_genPartIdx,
//                                                 GenPart_pdgId, GenPart_genPartIdxMother, -1),
//            GenPart_pdgId, 0)
bool isQuarkOrGluonPdgId(long double pdgId) {
    const long long absPdgId = llabs(static_cast<long long>(llround(pdgId)));
    return (absPdgId >= 1 && absPdgId <= 6) || absPdgId == 21;
}

Value evalFirstNonQuarkGluonAncestorIndex(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 4) {
        throw runtime_error(
            "first_nonqg_ancestor_index requires collection, index expression, pdgId field "
            "expression, and mother-index field expression (plus optional default)");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const long double defaultValue = (args.size() >= 5) ? evalNumber(args[4], context) : def;
    const int size = static_cast<int>(collection->objects.size());
    const int startIndex = static_cast<int>(llround(evalNumber(args[1], context)));
    if (startIndex < 0 || startIndex >= size) {
        return makeNumberValue(defaultValue);
    }

    auto fieldAt = [&](int index, const ExprPtr& fieldExpr) -> long double {
        EvalContext loop = context;
        loop.currentCollection = collection;
        loop.currentObject = &collection->objects[index];
        return evalNumber(fieldExpr, loop);
    };

    if (!isQuarkOrGluonPdgId(fieldAt(startIndex, args[2]))) {
        return makeNumberValue(static_cast<long double>(startIndex));
    }

    int current = startIndex;
    // Cap the walk so a malformed/cyclic mother chain can't hang the conversion.
    for (int step = 0; step <= size; ++step) {
        const int motherIndex = static_cast<int>(llround(fieldAt(current, args[3])));
        if (motherIndex < 0 || motherIndex >= size) {
            return makeNumberValue(defaultValue);
        }
        if (!isQuarkOrGluonPdgId(fieldAt(motherIndex, args[2]))) {
            return makeNumberValue(static_cast<long double>(motherIndex));
        }
        current = motherIndex;
    }
    return makeNumberValue(defaultValue);
}

// first_boson_ancestor_index(collection, index_expr, pdgid_field, mother_index_field
//                             [, default]):
// Starting at collection[index_expr], walk the mother chain (mother_index_field) looking
// for the first ancestor (including the starting particle itself) that is a hard-process
// boson -- W (|pdgId| == 24), Z (|pdgId| == 23), or Higgs (|pdgId| == 25) -- i.e. the
// actual resonance that produced the jet, rather than any intermediate quark/gluon/
// lepton/photon/hadron bookkeeping in between. This is more general than
// first_nonqg_ancestor_index, which stops at the first non-quark/gluon ancestor and so
// can be fooled by a lepton/photon/hadron sitting between the jet's matched particle and
// the real boson further up the chain (the failure mode seen with looser/generic
// genPartIdx matching schemes, and also why checking the generator's own "isHardProcess"
// status flag doesn't work here: that flag is set on the boson's own decay products too,
// so it triggers immediately at the starting particle without ever climbing to the
// boson). first_boson_ancestor_index instead keeps climbing through ANY particle type
// until a genuine boson is found.
//
// Once a boson ancestor is found, the walk continues climbing through same-pdgId
// self-copies (mother has the IDENTICAL pdgId -- the same physical particle at an
// earlier point in its shower history, before it radiated/copied itself) to canonicalize
// on the EARLIEST recorded copy of that boson. This matters under loose/generic
// genPartIdx matching schemes where a jet's matched particle can be the boson itself
// rather than one of its quark daughters: without climbing to the earliest copy, two
// jets that both trace back to the very same physical boson -- one via a quark
// descendant (which lands on the LAST copy, immediately before the 2-body decay) and one
// matched directly to an early copy of the boson -- would resolve to two different
// GenPart indices, silently breaking any downstream index-equality "same boson" check.
//
// If the chain runs off the end (or hits the walk cap) before a boson is found, returns
// the terminal (last valid) ancestor instead of `default`, so the caller can still
// inspect its pdgId (e.g. a leftover quark/gluon signals ISR/extra QCD radiation with no
// W/Z in its history). Returns default only if index_expr itself is out of range.
// Typical use: pair with value_at to identify the hard-process origin of a gen-matched
// jet, e.g.
//   value_at(GenPart, first_boson_ancestor_index(GenPart, ScoutingFatPFJetRecluster_genPartIdx,
//                                                 GenPart_pdgId, GenPart_genPartIdxMother, -1),
//            GenPart_pdgId, 0)
bool isHardProcessBosonPdgId(long double pdgId) {
    const long long absPdgId = llabs(static_cast<long long>(llround(pdgId)));
    return absPdgId == 23 || absPdgId == 24 || absPdgId == 25;
}

Value evalFirstBosonAncestorIndex(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 4) {
        throw runtime_error(
            "first_boson_ancestor_index requires collection, index expression, pdgId field "
            "expression, and mother-index field expression (plus optional default)");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const long double defaultValue = (args.size() >= 5) ? evalNumber(args[4], context) : def;
    const int size = static_cast<int>(collection->objects.size());
    const int startIndex = static_cast<int>(llround(evalNumber(args[1], context)));
    if (startIndex < 0 || startIndex >= size) {
        return makeNumberValue(defaultValue);
    }

    auto fieldAt = [&](int index, const ExprPtr& fieldExpr) -> long double {
        EvalContext loop = context;
        loop.currentCollection = collection;
        loop.currentObject = &collection->objects[index];
        return evalNumber(fieldExpr, loop);
    };

    // Given the index of a confirmed boson ancestor, keep climbing through same-pdgId
    // self-copy mothers to reach the earliest recorded copy of that same physical boson.
    auto climbToEarliestCopy = [&](int bosonIndex) -> int {
        const long double bosonPdgId = fieldAt(bosonIndex, args[2]);
        int earliest = bosonIndex;
        for (int step = 0; step <= size; ++step) {
            const int motherIndex = static_cast<int>(llround(fieldAt(earliest, args[3])));
            if (motherIndex < 0 || motherIndex >= size || fieldAt(motherIndex, args[2]) != bosonPdgId) {
                return earliest;
            }
            earliest = motherIndex;
        }
        return earliest;
    };

    if (isHardProcessBosonPdgId(fieldAt(startIndex, args[2]))) {
        return makeNumberValue(static_cast<long double>(climbToEarliestCopy(startIndex)));
    }

    int current = startIndex;
    // Cap the walk so a malformed/cyclic mother chain can't hang the conversion.
    for (int step = 0; step <= size; ++step) {
        const int motherIndex = static_cast<int>(llround(fieldAt(current, args[3])));
        if (motherIndex < 0 || motherIndex >= size) {
            // Chain end before any boson ancestor was found: return the terminal
            // ancestor so the caller can still inspect its pdgId.
            return makeNumberValue(static_cast<long double>(current));
        }
        if (isHardProcessBosonPdgId(fieldAt(motherIndex, args[2]))) {
            return makeNumberValue(static_cast<long double>(climbToEarliestCopy(motherIndex)));
        }
        current = motherIndex;
    }
    // Cap exceeded: return the last visited (non-boson) ancestor.
    return makeNumberValue(static_cast<long double>(current));
}

// count_hadronic_tau_from_wz(collection, pdgid_field, mother_index_field) /
// count_leptonic_tau_from_wz(collection, pdgid_field, mother_index_field):
// Count hard-process W/Z bosons whose decay proceeds via a tau (|pdgId| == 15) that is
// itself a DIRECT daughter of the boson, split by how that tau subsequently decays.
//
// This exists because nGenHadronicWZ (wz_had_quarks/2) only counts a boson as hadronic
// when its direct daughter is a quark -- a W/Z -> tau -> hadrons + nu chain is always
// tagged "leptonic" there, even though ~65% of tau decays are hadronic (a narrow,
// jet-like visible signature with no electron/muon at all). Since any e/mu-based lepton
// veto and any jet-based BDT can only ever see the tau's decay PRODUCTS -- never the
// intermediate tau itself -- a hadronically-decaying tau is functionally invisible to
// both, and physically indistinguishable from a genuine light-quark jet. This builtin
// exposes that population so its effect on the nGenHadronicWZ-based purity metric can be
// quantified directly, rather than assumed.
//
// For each candidate tau, first walk DOWN through same-pdgId self-copy daughters (the
// tau equivalent of first_boson_ancestor_index's upward same-pdgId climb: a radiated/
// copied tau appears as its own daughter with identical pdgId before it actually decays)
// to reach the final copy, then classify that copy's own daughters: presence of an
// electron/muon (|pdgId| in {11,13}) means a leptonic tau decay, its absence (given the
// tau did decay at all) means hadronic.
Value evalCountTauDecayFromBoson(Op op, const string& name, const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 3) {
        throw runtime_error(name + " requires collection, pdgId field, and mother-index field expressions");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const int size = static_cast<int>(collection->objects.size());

    auto fieldAt = [&](int index, const ExprPtr& fieldExpr) -> long double {
        EvalContext loop = context;
        loop.currentCollection = collection;
        loop.currentObject = &collection->objects[index];
        return evalNumber(fieldExpr, loop);
    };
    auto pdgIdAt = [&](int index) -> long long { return llabs(llround(fieldAt(index, args[1]))); };
    auto motherIndexAt = [&](int index) -> int { return static_cast<int>(llround(fieldAt(index, args[2]))); };

    long long count = 0;
    for (int i = 0; i < size; ++i) {
        if (pdgIdAt(i) != 15) continue;
        const int motherIdx = motherIndexAt(i);
        if (motherIdx < 0 || motherIdx >= size) continue;
        const long long motherPdgId = pdgIdAt(motherIdx);
        if (motherPdgId != 24 && motherPdgId != 23) continue;
        // i is a "root" tau: the first recorded copy directly attributed to the W/Z.

        int current = i;
        bool hasLeptonDaughter = false;
        bool decayed = false;
        for (int step = 0; step <= size; ++step) {
            int nextTauCopy = -1;
            bool anyDaughter = false;
            bool leptonDaughterHere = false;
            for (int j = 0; j < size; ++j) {
                if (motherIndexAt(j) != current) continue;
                anyDaughter = true;
                const long long daughterPdgId = pdgIdAt(j);
                if (daughterPdgId == 15 && nextTauCopy < 0) nextTauCopy = j;
                if (daughterPdgId == 11 || daughterPdgId == 13) leptonDaughterHere = true;
            }
            if (nextTauCopy >= 0) {
                current = nextTauCopy;
                continue;  // still a self-copy; keep climbing down to the final tau
            }
            if (anyDaughter) {
                hasLeptonDaughter = leptonDaughterHere;
                decayed = true;
            }
            break;
        }
        if (!decayed) continue;  // undecayed tau in the record (shouldn't happen); skip

        if (op == Op::CountLeptonicTauFromWz) {
            if (hasLeptonDaughter) ++count;
        } else {
            if (!hasLeptonDaughter) ++count;
        }
    }
    return makeNumberValue(static_cast<long double>(count));
}

Value evalPairwiseMetric(Op op,
                         const string& name,
                         const vector<ExprPtr>& args,
                         const EvalContext& context) {
    if (args.empty()) {
        throw runtime_error(name + " requires a collection argument");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const int limit = (args.size() >= 2) ? static_cast<int>(llround(evalNumber(args[1], context)))
                                         : static_cast<int>(collection->objects.size());
    const int count = min(limit, static_cast<int>(collection->objects.size()));
    if (count < 2) {
        return makeNumberValue(kMissingDistance);
    }

    const bool takeMin = (op == Op::PairMinDeltaR || op == Op::PairMinDeltaPhi);
    const bool deltaPhi = (op == Op::PairMinDeltaPhi || op == Op::PairMaxDeltaPhi);
    bool first = true;
    double best = 0.;
    for (int i = 0; i < count; ++i) {
        for (int j = i + 1; j < count; ++j) {
            const double value = pairMetric(deltaPhi, collection->objects[i].p4, collection->objects[j].p4);
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

// max_ratio_within_dr(anchor, collection, dr_threshold [, default]): among
// collection members within dr_threshold (deltaR) of anchor, the maximum
// ratio of a candidate's pT to the anchor's pT; `default` if none qualify.
// The anchor is evaluated ONCE in the caller's context before the loop --
// unlike max_value()/min_value() (evalAggregation), which rebind
// currentObject/self to each candidate while looping, so an outer "self"
// cannot be referenced from inside their loop body. Following min_deltaR's
// convention (self passed explicitly as args[0]) sidesteps that: this can be
// called as max_ratio_within_dr(self, ak4_nocuts, 0.8, default) from within
// a per-AK8-slot formula to get the nearby-AK4/AK8 pT ratio feature used by
// the ISR-vs-W discriminant score.
Value evalMaxRatioWithinDr(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 3) {
        throw runtime_error("max_ratio_within_dr requires anchor, collection, and dr_threshold arguments");
    }
    const TLorentzVector anchorP4 = toP4(evalExpression(args[0], context));
    const RuntimeCollection* collection = toCollection(evalExpression(args[1], context));
    const double drThreshold = static_cast<double>(evalNumber(args[2], context));
    const long double defaultValue = (args.size() >= 4) ? evalNumber(args[3], context) : def;

    const double anchorPt = anchorP4.Pt();
    bool found = false;
    long double best = defaultValue;
    for (const auto& object : collection->objects) {
        if (anchorP4.DeltaR(object.p4) >= drThreshold) continue;
        const long double ratio = (anchorPt > 0.0) ? static_cast<long double>(object.p4.Pt() / anchorPt) : 0.0L;
        if (!found || ratio > best) {
            best = ratio;
            found = true;
        }
    }
    return makeNumberValue(found ? best : defaultValue);
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

// Resolved-dijet pair selection: among all 2-object pairs (i < j, i,j < limit) in a
// collection, find the pair that best matches a criterion. Two criteria are supported:
//   "min_dr"          -- minimise deltaR(i, j) (closeby-jets heuristic)
//   "closest_wz_mass" -- minimise |m(i+j) - mW| or |m(i+j) - mZ|, whichever is closer
// Shared by both the P4-returning (pair_p4_*) and index-returning (pair_index_*)
// builtins below, so the two families can never disagree about which pair won.
enum class PairCriterion {
    MinDr,
    ClosestWzMass,
    CombinedWzDr,
};

struct BestJetPairResult {
    bool found = false;
    int i = -1;
    int j = -1;
    TLorentzVector sum;
};

BestJetPairResult findBestJetPair(const RuntimeCollection* collection, PairCriterion criterion, int limit) {
    BestJetPairResult result;
    const int count = min(limit, static_cast<int>(collection->objects.size()));
    if (count < 2) {
        return result;
    }
    double bestMetric = 0.;
    for (int i = 0; i < count; ++i) {
        const TLorentzVector& p4i = collection->objects[i].p4;
        for (int j = i + 1; j < count; ++j) {
            const TLorentzVector& p4j = collection->objects[j].p4;
            double metric;
            if (criterion == PairCriterion::MinDr) {
                metric = p4i.DeltaR(p4j);
            } else if (criterion == PairCriterion::ClosestWzMass) {
                const double m = (p4i + p4j).M();
                metric = min(fabs(m - kNominalWMass), fabs(m - kNominalZMass));
            } else {  // "combined_wz_dr": chi2-like combination of both terms
                const double m = (p4i + p4j).M();
                const double massTerm = min(fabs(m - kNominalWMass), fabs(m - kNominalZMass)) / kWZMassSigma;
                const double drTerm = p4i.DeltaR(p4j) / kPairDrSigma;
                metric = massTerm * massTerm + drTerm * drTerm;
            }
            if (!result.found || metric < bestMetric) {
                bestMetric = metric;
                result.found = true;
                result.i = i;
                result.j = j;
                result.sum = p4i + p4j;
            }
        }
    }
    return result;
}

// Charge/flavor of a single collection object, resolved through whichever of the three
// lepton sources (ScoutingElectron / ScoutingMuonVtx / ScoutingMuonNoVtx) it was merged
// from -- mirrors the first_valid(...) idiom used throughout branch.json (each source's
// pt defaults to `def` = -99 on an object that didn't come from it).
struct LeptonChargeFlavor {
    bool valid = false;
    bool isElectron = false;
    int charge = 0;
};

LeptonChargeFlavor getLeptonChargeFlavor(const RuntimeCollection& collection, const RuntimeObject& object) {
    LeptonChargeFlavor result;
    if (getObjectField(collection, object, "ScoutingElectron_pt", def) > -90.f) {
        result.valid = true;
        result.isElectron = true;
        result.charge = static_cast<int>(llround(getObjectField(collection, object, "ScoutingElectron_bestTrack_charge", def)));
        return result;
    }
    if (getObjectField(collection, object, "ScoutingMuonVtx_pt", def) > -90.f) {
        result.valid = true;
        result.isElectron = false;
        result.charge = static_cast<int>(llround(getObjectField(collection, object, "ScoutingMuonVtx_charge", def)));
        return result;
    }
    if (getObjectField(collection, object, "ScoutingMuonNoVtx_pt", def) > -90.f) {
        result.valid = true;
        result.isElectron = false;
        result.charge = static_cast<int>(llround(getObjectField(collection, object, "ScoutingMuonNoVtx_charge", def)));
    }
    return result;
}

// Same-flavor opposite-sign (SFOS) lepton pair selection: among all 2-object pairs
// (i < j, i,j < limit) in a collection, find the SFOS pair whose invariant mass is
// closest to the Z mass. Pairs that aren't same-flavor-opposite-sign are skipped
// entirely (not merely disfavoured), so "not found" means no SFOS pair exists.
BestJetPairResult findBestSFOSPair(const RuntimeCollection* collection, int limit) {
    BestJetPairResult result;
    const int count = min(limit, static_cast<int>(collection->objects.size()));
    double bestMetric = 0.;
    for (int i = 0; i < count; ++i) {
        const LeptonChargeFlavor infoI = getLeptonChargeFlavor(*collection, collection->objects[i]);
        if (!infoI.valid) continue;
        for (int j = i + 1; j < count; ++j) {
            const LeptonChargeFlavor infoJ = getLeptonChargeFlavor(*collection, collection->objects[j]);
            if (!infoJ.valid) continue;
            if (infoI.isElectron != infoJ.isElectron) continue;
            if (infoI.charge * infoJ.charge >= 0) continue;
            const TLorentzVector sum = collection->objects[i].p4 + collection->objects[j].p4;
            const double metric = fabs(sum.M() - kNominalZMass);
            if (!result.found || metric < bestMetric) {
                bestMetric = metric;
                result.found = true;
                result.i = i;
                result.j = j;
                result.sum = sum;
            }
        }
    }
    return result;
}

// pair_p4_min_dr(collection [, limit]) / pair_p4_closest_wz_mass(collection [, limit]):
// returns the summed 4-vector of the winning pair (zero vector if fewer than 2 objects),
// meant to be composed with mass()/pt()/eta()/phi(), e.g.
//   mass(pair_p4_min_dr(ak4, 4))
Value evalPairP4Selection(PairCriterion criterion, const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.empty()) {
        throw runtime_error("pair_p4_* requires a collection argument");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const int limit = (args.size() >= 2) ? static_cast<int>(llround(evalNumber(args[1], context)))
                                         : static_cast<int>(collection->objects.size());
    const BestJetPairResult best = findBestJetPair(collection, criterion, limit);
    return makeP4Value(best.found ? best.sum : TLorentzVector());
}

// pair_index_min_dr(collection, slot [, limit [, default]]) /
// pair_index_closest_wz_mass(collection, slot [, limit [, default]]):
// returns the winning pair's collection index for the requested slot (1 or 2; since
// collections are pT-sorted, slot 1 is the higher-pT member), or `default` (or `def`)
// if fewer than 2 objects are available. Lets downstream formulas (or offline analysis)
// identify exactly which two jets were picked, e.g. to cross-check against gen truth.
Value evalPairIndexSelection(PairCriterion criterion, const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 2) {
        throw runtime_error("pair_index_* requires a collection and a slot (1 or 2) argument");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const int slot = static_cast<int>(llround(evalNumber(args[1], context)));
    const int limit = (args.size() >= 3) ? static_cast<int>(llround(evalNumber(args[2], context)))
                                         : static_cast<int>(collection->objects.size());
    const long double defaultValue = (args.size() >= 4) ? evalNumber(args[3], context) : def;
    const BestJetPairResult best = findBestJetPair(collection, criterion, limit);
    if (!best.found) {
        return makeNumberValue(defaultValue);
    }
    if (slot == 1) {
        return makeNumberValue(static_cast<long double>(best.i));
    }
    if (slot == 2) {
        return makeNumberValue(static_cast<long double>(best.j));
    }
    throw runtime_error("pair_index_* slot must be 1 or 2");
}

// pair_p4_sfos_z_mass(collection [, limit]) / pair_index_sfos_z_mass(collection, slot
// [, limit [, default]]): same call shape as pair_p4_combined_wz_dr / pair_index_combined_wz_dr,
// but selects the same-flavor opposite-sign pair closest to the Z mass instead of a plain
// jet pair. "not found" (index default, zero-vector p4) means no SFOS pair exists.
Value evalPairP4SFOS(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.empty()) {
        throw runtime_error("pair_p4_sfos_z_mass requires a collection argument");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const int limit = (args.size() >= 2) ? static_cast<int>(llround(evalNumber(args[1], context)))
                                         : static_cast<int>(collection->objects.size());
    const BestJetPairResult best = findBestSFOSPair(collection, limit);
    return makeP4Value(best.found ? best.sum : TLorentzVector());
}

Value evalPairIndexSFOS(const vector<ExprPtr>& args, const EvalContext& context) {
    if (args.size() < 2) {
        throw runtime_error("pair_index_sfos_z_mass requires a collection and a slot (1 or 2) argument");
    }
    const RuntimeCollection* collection = toCollection(evalExpression(args[0], context));
    const int slot = static_cast<int>(llround(evalNumber(args[1], context)));
    const int limit = (args.size() >= 3) ? static_cast<int>(llround(evalNumber(args[2], context)))
                                         : static_cast<int>(collection->objects.size());
    const long double defaultValue = (args.size() >= 4) ? evalNumber(args[3], context) : def;
    const BestJetPairResult best = findBestSFOSPair(collection, limit);
    if (!best.found) {
        return makeNumberValue(defaultValue);
    }
    if (slot == 1) {
        return makeNumberValue(static_cast<long double>(best.i));
    }
    if (slot == 2) {
        return makeNumberValue(static_cast<long double>(best.j));
    }
    throw runtime_error("pair_index_sfos_z_mass slot must be 1 or 2");
}

Value evalCall(const ExprPtr& expr, const EvalContext& context) {
    const string& op = expr->text;
    const auto& args = expr->args;

    switch (expr->op) {
    case Op::Abs:
        return makeNumberValue(fabsl(evalNumber(args.at(0), context)));
    case Op::Sqrt:
        return makeNumberValue(sqrtl(evalNumber(args.at(0), context)));
    case Op::Cos:
        return makeNumberValue(cosl(evalNumber(args.at(0), context)));
    case Op::Sin:
        return makeNumberValue(sinl(evalNumber(args.at(0), context)));
    case Op::Pow:
        return makeNumberValue(powl(evalNumber(args.at(0), context), evalNumber(args.at(1), context)));
    case Op::Min: {
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
    case Op::Max: {
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
    case Op::SafeDiv: {
        const long double numerator = evalNumber(args.at(0), context);
        const long double denominator = evalNumber(args.at(1), context);
        const long double fallback = (args.size() >= 3) ? evalNumber(args.at(2), context) : 0.;
        if (denominator == 0.) {
            return makeNumberValue(fallback);
        }
        return makeNumberValue(numerator / denominator);
    }
    case Op::FirstValid:
        for (const auto& arg : args) {
            const long double value = evalNumber(arg, context);
            if (fabsl(value - def) > 1e-9L) {
                return makeNumberValue(value);
            }
        }
        return makeNumberValue(def);
    case Op::Size:
        return makeNumberValue(static_cast<long double>(toCollection(evalExpression(args.at(0), context))->objects.size()));
    case Op::Sum:
    case Op::MaxValue:
    case Op::MinValue:
        return evalAggregation(expr->op, op, args, context);
    case Op::Sphericity:
    case Op::Aplanarity:
    case Op::Planarity:
        return makeNumberValue(static_cast<long double>(evalEventShape(expr->op, op, args, context)));
    case Op::NthMaxValue:
        return evalNthMaxValue(args, context);
    case Op::ValueAtMax:
        return evalValueAtMax(args, context);
    case Op::ValueAtNthMax:
        return evalValueAtNthMax(args, context);
    case Op::ValueAt:
        return evalValueAt(args, context);
    case Op::FirstAncestorIndex:
        return evalFirstAncestorIndex(args, context);
    case Op::FirstNonQgAncestorIndex:
        return evalFirstNonQuarkGluonAncestorIndex(args, context);
    case Op::FirstBosonAncestorIndex:
        return evalFirstBosonAncestorIndex(args, context);
    case Op::CountHadronicTauFromWz:
    case Op::CountLeptonicTauFromWz:
        return evalCountTauDecayFromBoson(expr->op, op, args, context);
    case Op::PairP4MinDr:
        return evalPairP4Selection(PairCriterion::MinDr, args, context);
    case Op::PairP4ClosestWzMass:
        return evalPairP4Selection(PairCriterion::ClosestWzMass, args, context);
    case Op::PairIndexMinDr:
        return evalPairIndexSelection(PairCriterion::MinDr, args, context);
    case Op::PairIndexClosestWzMass:
        return evalPairIndexSelection(PairCriterion::ClosestWzMass, args, context);
    case Op::PairP4CombinedWzDr:
        return evalPairP4Selection(PairCriterion::CombinedWzDr, args, context);
    case Op::PairIndexCombinedWzDr:
        return evalPairIndexSelection(PairCriterion::CombinedWzDr, args, context);
    case Op::PairP4SfosZMass:
        return evalPairP4SFOS(args, context);
    case Op::PairIndexSfosZMass:
        return evalPairIndexSFOS(args, context);
    case Op::Mass:
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).M());
    case Op::Pt:
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).Pt());
    case Op::Eta:
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).Eta());
    case Op::Phi:
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).Phi());
    case Op::DeltaR:
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).DeltaR(toP4(evalExpression(args.at(1), context))));
    case Op::DeltaPhi:
        return makeNumberValue(toP4(evalExpression(args.at(0), context)).DeltaPhi(toP4(evalExpression(args.at(1), context))));
    case Op::RelPtDiff: {
        const double pt1 = toP4(evalExpression(args.at(0), context)).Pt();
        const double pt2 = toP4(evalExpression(args.at(1), context)).Pt();
        if (pt1 == 0.) {
            return makeNumberValue(kLargeDistance);
        }
        return makeNumberValue(fabs(pt1 - pt2) / pt1);
    }
    case Op::PairMinDeltaR:
    case Op::PairMaxDeltaR:
    case Op::PairMinDeltaPhi:
    case Op::PairMaxDeltaPhi:
        return evalPairwiseMetric(expr->op, op, args, context);
    case Op::ClosestDeltaR:
        return evalClosestMetric(args, context);
    case Op::MinDeltaR:
        return evalMinDeltaR(args, context);
    case Op::MaxRatioWithinDr:
        return evalMaxRatioWithinDr(args, context);
    case Op::DeltaPhiAtMinDeltaR:
        return evalDeltaPhiAtMinDeltaR(args, context);
    default:
        break;
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
        switch (expr->op) {
        case Op::True:
            return makeNumberValue(1.);
        case Op::False:
            return makeNumberValue(0.);
        case Op::Self:
            if (!context.currentCollection || !context.currentObject) {
                throw runtime_error("self used without current object");
            }
            return makeObjectValue(context.currentCollection, context.currentObject);
        case Op::Other:
            if (!context.otherCollection || !context.otherObject) {
                throw runtime_error("other used without comparison object");
            }
            return makeObjectValue(context.otherCollection, context.otherObject);
        default:
            break;
        }
        // Lookup order: field of the current object, event variable, collection. (Every input
        // scalar is an event variable, so the former raw-scalar fallback could never be reached.)
        if (!expr->resolved) {
            throw runtime_error("Internal error: unresolved identifier in expression: " + expr->text);
        }
        if (context.currentCollection && context.currentObject) {
            const int field = expr->fieldIndex[context.currentCollection->schema->id];
            if (field >= 0) {
                return makeNumberValue(context.currentObject->values[field]);
            }
        }
        if (context.vars && context.vars->has(expr->varSlot)) {
            return makeNumberValue(context.vars->values[expr->varSlot]);
        }
        const RuntimeCollection* collection = findCollection(context, *expr);
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
        switch (expr->op) {
        case Op::Plus:
            return makeNumberValue(+toNumber(value));
        case Op::Minus:
            return makeNumberValue(-toNumber(value));
        case Op::Not:
            return makeNumberValue(truthy(value) ? 0. : 1.);
        default:
            break;
        }
        throw runtime_error("Unsupported unary operator: " + expr->text);
    }
    if (expr->kind == ExprKind::Binary) {
        if (expr->op == Op::And) {
            return makeNumberValue((truthy(evalExpression(expr->lhs, context)) && truthy(evalExpression(expr->rhs, context))) ? 1. : 0.);
        }
        if (expr->op == Op::Or) {
            return makeNumberValue((truthy(evalExpression(expr->lhs, context)) || truthy(evalExpression(expr->rhs, context))) ? 1. : 0.);
        }

        const Value lhs = evalExpression(expr->lhs, context);
        const Value rhs = evalExpression(expr->rhs, context);

        if (expr->op == Op::Plus || expr->op == Op::Minus) {
            const bool lhsP4 = (lhs.kind == Value::Kind::ObjectRef || lhs.kind == Value::Kind::P4);
            const bool rhsP4 = (rhs.kind == Value::Kind::ObjectRef || rhs.kind == Value::Kind::P4);
            if (lhsP4 || rhsP4) {
                TLorentzVector total = toP4(lhs);
                if (expr->op == Op::Plus) {
                    total += toP4(rhs);
                } else {
                    total -= toP4(rhs);
                }
                return makeP4Value(total);
            }
        }

        const long double leftNumber = toNumber(lhs);
        const long double rightNumber = toNumber(rhs);
        switch (expr->op) {
        case Op::Plus:
            return makeNumberValue(leftNumber + rightNumber);
        case Op::Minus:
            return makeNumberValue(leftNumber - rightNumber);
        case Op::Mul:
            return makeNumberValue(leftNumber * rightNumber);
        case Op::Div:
            return makeNumberValue(leftNumber / rightNumber);
        case Op::Lt:
            return makeNumberValue(leftNumber < rightNumber ? 1. : 0.);
        case Op::Le:
            return makeNumberValue(leftNumber <= rightNumber ? 1. : 0.);
        case Op::Gt:
            return makeNumberValue(leftNumber > rightNumber ? 1. : 0.);
        case Op::Ge:
            return makeNumberValue(leftNumber >= rightNumber ? 1. : 0.);
        case Op::Eq:
            return makeNumberValue(leftNumber == rightNumber ? 1. : 0.);
        case Op::Ne:
            return makeNumberValue(leftNumber != rightNumber ? 1. : 0.);
        default:
            break;
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

const RuntimeCollection& buildRuntimeCollection(int slot,
                                                const SelectionConfig& selectionConfig,
                                                EventCollections& collections,
                                                const EventVars& baseVars) {
    if (collections.built[slot]) {
        return collections.runtime[slot];
    }
    const RuntimeCollectionConfig& config = selectionConfig.collections[slot];
    if (collections.active[slot]) {
        throw runtime_error("Collection dependency cycle detected at: " + config.name);
    }
    collections.active[slot] = 1;

    EvalContext context;
    context.vars = &baseVars;
    context.collections = &collections;

    // A sourced collection is selected straight from the input collection (same schema), so the
    // input is not copied first. Unknown sources/children were rejected by resolveEngineSymbols.
    RuntimeCollection current;
    if (config.sourceSlot >= 0) {
        const RuntimeCollection& input = collections.inputs[config.sourceSlot];
        current = config.selectionExpr ? applySelection(input, config.selectionExpr, context) : input;
        current.name = config.name;
    } else {
        vector<const RuntimeCollection*> sources;
        sources.reserve(config.mergeSlots.size());
        for (const int child : config.mergeSlots) {
            sources.push_back(&buildRuntimeCollection(child, selectionConfig, collections, baseVars));
        }
        current = mergeCollections(config, sources);
        if (config.selectionExpr) {
            current = applySelection(current, config.selectionExpr, context);
        }
    }

    if (config.dedupSlot >= 0) {
        const RuntimeCollection& reference = buildRuntimeCollection(config.dedupSlot, selectionConfig, collections, baseVars);
        current = applyDeduplication(current, reference, config.dedupExpr, context);
    }

    if (config.sortRule.expr) {
        sortCollection(current, config.sortRule, context);
    }

    collections.active[slot] = 0;
    collections.runtime[slot] = std::move(current);
    collections.built[slot] = 1;
    return collections.runtime[slot];
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

// -------------------- Batch provenance --------------------
uint64_t fnv1a64(const string& data, uint64_t hash = 1469598103934665603ULL) {
    for (const unsigned char c : data) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return hash;
}

string hashHex(uint64_t hash) {
    ostringstream os;
    os << hex << setw(16) << setfill('0') << hash;
    return os.str();
}

string readFileBytes(const string& path) {
    ifstream fin(path, ios::binary);
    if (!fin) {
        throw runtime_error("Cannot read " + path + " for the batch provenance hash");
    }
    return string((istreambuf_iterator<char>(fin)), istreambuf_iterator<char>());
}

// Identity of a batch's input slice (ordered file names).
string hashFileList(const vector<string>& files) {
    uint64_t hash = 1469598103934665603ULL;
    for (const auto& file : files) {
        hash = fnv1a64(file + "\n", hash);
    }
    return hashHex(hash);
}

// Everything besides the input files that changes a batch output: the converter binary,
// branch.json / selection.json, the input tree name, the pileup weights (MC) or lumi mask
// (data) in use, and the sample metadata written into the trees. A batch whose stored hash
// differs is rerun on resume and rejected at merge time.
string computeConversionConfigHash(const AppConfig& appConfig,
                                   const SampleMeta& sampleMeta,
                                   const string& puWeightPath) {
    uint64_t hash = 1469598103934665603ULL;
    const auto addFile = [&](const string& label, const string& path) {
        hash = fnv1a64(label + "\n", hash);
        hash = fnv1a64(readFileBytes(path), hash);
    };
    addFile("binary", "/proc/self/exe");
    addFile("branch", resolveReferencedPath(appConfig.configPath, kBranchConfigPath));
    addFile("selection", resolveReferencedPath(appConfig.configPath, kSelectionConfigPath));
    if (!puWeightPath.empty()) {
        addFile("pileup", puWeightPath);
    }
    if (!sampleMeta.isMC && !appConfig.lumiMaskPath.empty()) {
        addFile("lumi_mask", appConfig.lumiMaskPath);
    }
    // Jet corrections: their settings and every correction / JMS-JMR input file.
    const JetPtCorrectionConfig& jpc = appConfig.jetPtCorrection;
    if (jpc.enabled) {
        ostringstream jec;
        jec << setprecision(17) << jpc.nominalCorrection << '|' << jpc.jecAk4Name << '|' << jpc.jecAk8Name << '|'
            << jpc.jecAk4L1Name << '|' << jpc.jecAk8L1Name << '|' << jpc.ak4TagThreshold << '|'
            << jpc.ak8TagThreshold << '|' << jpc.jerResolutionName << '|' << jpc.jerScaleFactorName << '|'
            << jpc.jesShift << '|' << jpc.applyJmsJmr << '|' << jpc.debugNominalConfiguration;
        for (const auto& variation : jpc.variations) {
            jec << "|variation=" << variation;
        }
        const map<string, vector<string>> variationBranches(jpc.variationBranches.begin(),
                                                            jpc.variationBranches.end());
        for (const auto& item : variationBranches) {
            jec << "|" << item.first << ":";
            for (const auto& branch : item.second) {
                jec << branch << ",";
            }
        }
        hash = fnv1a64("jet_pt_correction\n" + jec.str(), hash);
        const vector<pair<string, string>> files = {
            {"jec_ak4", jpc.jecAk4File}, {"jec_ak8", jpc.jecAk8File}, {"corrections", jpc.correctionsFile},
            {"jes_jer", jpc.jesJerFile}, {"jer_smear", jpc.jerSmearFile}, {"jms_jmr", jpc.jmsJmrResultsFile},
        };
        for (const auto& file : files) {
            if (!file.second.empty()) {
                addFile(file.first, file.second);
            }
        }
    }
    ostringstream meta;
    meta << setprecision(17) << appConfig.treeName << '|' << sampleMeta.sample << '|'
         << sampleMeta.sampleId << '|' << sampleMeta.isMC << '|' << sampleMeta.isSignal << '|'
         << sampleMeta.hasTheoryWeights << '|' << sampleMeta.xsection << '|' << sampleMeta.lumi;
    hash = fnv1a64(meta.str(), hash);
    return hashHex(hash);
}

// Batch completion record, written last (atomically) after the batch ROOT file and the other
// sidecars, so a batch without a matching .meta is never treated as complete.
struct BatchMeta {
    size_t nFiles = 0;
    string filesHash;
    string configHash;
    Long64_t rawEntries = 0;
    long double sumWeightPu = 0.L;
    long double sumWeightPuUp = 0.L;
    long double sumWeightPuDown = 0.L;
    // MC: genWeight sums over every processed generated event (before any selection), the
    // denominators of the signed, absolutely normalized MC event weights downstream.
    long double sumGenWeight = 0.L;
    long double sumGenWeightPu = 0.L;
    long double sumGenWeightPuUp = 0.L;
    long double sumGenWeightPuDown = 0.L;
    vector<string> skippedFiles;
};

fs::path makeBatchMetaPath(const fs::path& batchOutputPath) {
    return fs::path(batchOutputPath.string() + ".meta");
}

fs::path makeBatchLumisPath(const fs::path& batchOutputPath) {
    return fs::path(batchOutputPath.string() + ".lumis");
}

void writeTextFileAtomically(const fs::path& path, const string& content) {
    const fs::path tempPath = path.string() + ".tmp." + to_string(static_cast<long long>(getpid()));
    {
        ofstream fout(tempPath);
        if (!fout) {
            throw runtime_error("Cannot write " + tempPath.string());
        }
        fout << content;
        fout.close();
        if (!fout) {
            throw runtime_error("Failed writing " + tempPath.string());
        }
    }
    std::error_code ec;
    fs::rename(tempPath, path, ec);
    if (ec) {
        throw runtime_error("Failed to move " + tempPath.string() + " to " + path.string() + ": " + ec.message());
    }
}

void writeBatchMeta(const fs::path& batchOutputPath, const BatchMeta& meta) {
    ostringstream os;
    os << setprecision(21);
    os << "format=1\n"
       << "n_files=" << meta.nFiles << "\n"
       << "files_hash=" << meta.filesHash << "\n"
       << "config_hash=" << meta.configHash << "\n"
       << "raw_entries=" << meta.rawEntries << "\n"
       << "sum_weight_pu=" << meta.sumWeightPu << "\n"
       << "sum_weight_pu_up=" << meta.sumWeightPuUp << "\n"
       << "sum_weight_pu_down=" << meta.sumWeightPuDown << "\n"
       << "sum_genweight=" << meta.sumGenWeight << "\n"
       << "sum_genweight_pu=" << meta.sumGenWeightPu << "\n"
       << "sum_genweight_pu_up=" << meta.sumGenWeightPuUp << "\n"
       << "sum_genweight_pu_down=" << meta.sumGenWeightPuDown << "\n";
    for (const auto& file : meta.skippedFiles) {
        os << "skipped_file=" << file << "\n";
    }
    writeTextFileAtomically(makeBatchMetaPath(batchOutputPath), os.str());
}

bool readBatchMeta(const fs::path& batchOutputPath, BatchMeta& meta, string& reason) {
    ifstream fin(makeBatchMetaPath(batchOutputPath));
    if (!fin) {
        reason = "missing batch .meta (incomplete batch or written by an older convert_branch)";
        return false;
    }
    meta = BatchMeta();
    set<string> seen;
    string line;
    try {
        while (getline(fin, line)) {
            const size_t eq = line.find('=');
            if (eq == string::npos) {
                continue;
            }
            const string key = line.substr(0, eq);
            const string value = line.substr(eq + 1);
            seen.insert(key);
            if (key == "n_files") meta.nFiles = static_cast<size_t>(stoull(value));
            else if (key == "files_hash") meta.filesHash = value;
            else if (key == "config_hash") meta.configHash = value;
            else if (key == "raw_entries") meta.rawEntries = stoll(value);
            else if (key == "sum_weight_pu") meta.sumWeightPu = stold(value);
            else if (key == "sum_weight_pu_up") meta.sumWeightPuUp = stold(value);
            else if (key == "sum_weight_pu_down") meta.sumWeightPuDown = stold(value);
            else if (key == "sum_genweight") meta.sumGenWeight = stold(value);
            else if (key == "sum_genweight_pu") meta.sumGenWeightPu = stold(value);
            else if (key == "sum_genweight_pu_up") meta.sumGenWeightPuUp = stold(value);
            else if (key == "sum_genweight_pu_down") meta.sumGenWeightPuDown = stold(value);
            else if (key == "skipped_file") meta.skippedFiles.push_back(value);
        }
    } catch (const exception& ex) {
        reason = string("unreadable batch .meta: ") + ex.what();
        return false;
    }
    for (const char* key : {"format", "n_files", "files_hash", "config_hash", "raw_entries"}) {
        if (seen.count(key) == 0u) {
            reason = string("batch .meta lacks '") + key + "'";
            return false;
        }
    }
    return true;
}

void writeBatchLumis(const fs::path& batchOutputPath, const set<pair<UInt_t, UInt_t>>& lumis) {
    ostringstream os;
    for (const auto& lumi : lumis) {
        os << lumi.first << ' ' << lumi.second << '\n';
    }
    writeTextFileAtomically(makeBatchLumisPath(batchOutputPath), os.str());
}

void readBatchLumis(const fs::path& batchOutputPath, set<pair<UInt_t, UInt_t>>& lumis) {
    const fs::path path = makeBatchLumisPath(batchOutputPath);
    ifstream fin(path);
    if (!fin) {
        throw runtime_error("Missing batch lumi list " + path.string());
    }
    UInt_t run = 0;
    UInt_t lumi = 0;
    while (fin >> run >> lumi) {
        lumis.emplace(run, lumi);
    }
    if (!fin.eof()) {
        throw runtime_error("Malformed batch lumi list " + path.string());
    }
}

// Golden-JSON layout ({"run": [[first, last], ...]}), usable directly with brilcalc -i.
void writeProcessedLumiJson(const fs::path& path, const set<pair<UInt_t, UInt_t>>& lumis) {
    map<UInt_t, vector<pair<UInt_t, UInt_t>>> ranges;
    for (const auto& item : lumis) {
        auto& runRanges = ranges[item.first];
        if (!runRanges.empty() && item.second == runRanges.back().second + 1) {
            runRanges.back().second = item.second;
        } else {
            runRanges.emplace_back(item.second, item.second);
        }
    }
    ostringstream os;
    os << "{";
    bool firstRun = true;
    for (const auto& run : ranges) {
        os << (firstRun ? "\n" : ",\n") << "  \"" << run.first << "\": [";
        firstRun = false;
        for (size_t i = 0; i < run.second.size(); ++i) {
            os << (i ? ", " : "") << "[" << run.second[i].first << ", " << run.second[i].second << "]";
        }
        os << "]";
    }
    os << "\n}\n";
    writeTextFileAtomically(path, os.str());
}

// A batch output counts as complete only when its ROOT file opens with every configured tree,
// the .raw_entries and .meta sidecars agree, the .meta matches the expected input slice and
// conversion configuration, and (for data) the processed-lumi list exists.
bool validateBatchTempOutput(const fs::path& batchOutputPath,
                             const vector<TreeConfig>& treeConfigs,
                             const string& expectedFilesHash,
                             const string& expectedConfigHash,
                             bool isMC,
                             BatchMeta& meta,
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

    Long64_t rawEntries = 0;
    try {
        rawEntries = readBatchRawEntries(batchOutputPath);
    } catch (const exception& ex) {
        reason = ex.what();
        return false;
    }
    if (!readBatchMeta(batchOutputPath, meta, reason)) {
        return false;
    }
    if (meta.rawEntries != rawEntries) {
        reason = "batch .meta and .raw_entries disagree";
        return false;
    }
    if (meta.filesHash != expectedFilesHash) {
        reason = "batch was produced from a different input-file slice";
        return false;
    }
    if (meta.configHash != expectedConfigHash) {
        reason = "batch was produced with a different converter binary or configuration";
        return false;
    }
    if (!isMC && !fs::exists(makeBatchLumisPath(batchOutputPath))) {
        reason = "missing processed-lumi list";
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

fs::path makeFileListSnapshotPath(const AppConfig& appConfig, const SampleMeta& sampleMeta) {
    return makeBatchTempOutputDir(appConfig, sampleMeta) / (sampleMeta.sample + ".files");
}

// The sorted input file list is discovered once and stored as {group}_tmp/{sample}.files;
// every batch job and the merge read that snapshot, so the file-to-batch mapping cannot shift
// when a DAS dataset grows while jobs run. The snapshot is (re)written when it does not exist
// or when CONVERT_REFRESH_FILE_LIST=1 (run.py sets it for a new mode-0 submission).
vector<string> resolveInputFiles(const AppConfig& appConfig, SampleMeta& sampleMeta) {
    const fs::path snapshotPath = makeFileListSnapshotPath(appConfig, sampleMeta);
    const char* refreshEnv = getenv(kRefreshFileListEnvVar);
    const bool refresh = refreshEnv != nullptr && string(refreshEnv) == "1";
    if (!refresh && fs::exists(snapshotPath)) {
        sampleMeta.remoteSourceCount = static_cast<size_t>(
            count_if(sampleMeta.inputPaths.begin(), sampleMeta.inputPaths.end(), isCmsDatasetPath));
        ifstream fin(snapshotPath);
        if (!fin) {
            throw runtime_error("Cannot read input file-list snapshot " + snapshotPath.string());
        }
        vector<string> files;
        string line;
        while (getline(fin, line)) {
            if (!line.empty()) {
                files.push_back(line);
            }
        }
        if (files.empty()) {
            throw runtime_error("Input file-list snapshot is empty: " + snapshotPath.string());
        }
        // stderr: --batch-count prints only the count on stdout.
        cerr << "Using input file-list snapshot " << snapshotPath.string()
             << " (" << files.size() << " files)" << endl;
        return files;
    }

    vector<string> files = discoverInputFiles(sampleMeta);
    fs::create_directories(snapshotPath.parent_path());
    ostringstream content;
    for (const auto& file : files) {
        content << file << '\n';
    }
    writeTextFileAtomically(snapshotPath, content.str());
    cerr << "Wrote input file-list snapshot " << snapshotPath.string()
         << " (" << files.size() << " files)" << endl;
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
        throw runtime_error("Usage: convert_branch <sample> "
                            "[batch_index|--batch-count|--merge-successful-batches|--update-genweight-mean]");
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
    if (arg == "--update-genweight-mean") {
        request.updateGenWeightMean = true;
        return request;
    }

    size_t batchIndex = 0;
    if (!parseNonNegativeIndex(arg, batchIndex)) {
        throw runtime_error("Invalid batch argument '" + arg +
                            "'. Use a non-negative batch index, --batch-count, --merge-successful-batches, "
                            "or --update-genweight-mean.");
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
    const unordered_set<string>& kept = treeState.config.keptBranches;
    const auto isKept = [&](const string& branchName) {
        return kept.empty() || kept.count(branchName) > 0;
    };
    for (const auto& config : configs) {
        if (config.onlyMC && !isMC) {
            continue;
        }
        if (!config.collection.empty()) {
            for (int slot = 0; slot < config.slots; ++slot) {
                const string branchName = config.name + "_" + to_string(slot + 1);
                if (isKept(branchName)) {
                    appendOutputBranch(treeState, config, branchName, slot);
                }
            }
        } else if (isKept(config.name)) {
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

    // Resolve every (config, slot) to its branch once; the lookup by name keeps the former
    // per-event semantics (a repeated branch name maps to the last booked branch). A variation
    // tree books only its kept branches; the others get kNoBranch.
    const auto branchIndex = [&](const string& branchName) {
        const auto it = treeState.branchIndexByName.find(branchName);
        if (it != treeState.branchIndexByName.end()) {
            return it->second;
        }
        if (treeState.config.keptBranches.empty()) {
            throw runtime_error("Output branch not booked: " + branchName);
        }
        return kNoBranch;
    };
    const auto indexGroup = [&](const vector<OutputScalarConfig>& configs) {
        vector<vector<size_t>> indices(configs.size());
        for (size_t index = 0; index < configs.size(); ++index) {
            const OutputScalarConfig& config = configs[index];
            if (config.onlyMC && !isMC) {
                continue;
            }
            if (config.collection.empty()) {
                indices[index].push_back(branchIndex(config.name));
            } else {
                for (int slot = 0; slot < config.slots; ++slot) {
                    indices[index].push_back(branchIndex(config.name + "_" + to_string(slot + 1)));
                }
            }
        }
        return indices;
    };
    treeState.regularBranches = indexGroup(treeState.config.regularScalars);
    treeState.extremaBranches = indexGroup(treeState.config.extremaScalars);
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
    result.tempFile = TFile::Open(result.tempFilePath.c_str(), "RECREATE", "", kOutputCompression);
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

// branchIndices: OutputTreeState::regularBranches / extremaBranches for configs.
void fillOutputGroup(const vector<OutputScalarConfig>& configs,
                     const vector<vector<size_t>>& branchIndices,
                     OutputTreeState& treeState,
                     EventVars& vars,
                     const EventCollections& collections,
                     const vector<ScalarInputConfig>& scalars,
                     bool isMC) {
    for (size_t index = 0; index < configs.size(); ++index) {
        const OutputScalarConfig& config = configs[index];
        if (config.onlyMC && !isMC) {
            continue;
        }

        if (config.collection.empty()) {
            // A variation tree evaluates only the scalars its kept branches read; a needed
            // scalar that is not kept is evaluated but not booked.
            if (!treeState.config.keptBranches.empty() && treeState.config.neededScalars.count(config.name) == 0) {
                continue;
            }
            EvalContext context;
            context.vars = &vars;
            context.collections = &collections;

            const size_t branchIndex = branchIndices[index][0];
            OutputBranchRuntime* branch = (branchIndex != kNoBranch) ? &treeState.branches[branchIndex] : nullptr;
            if (config.exactScalarIndex >= 0) {
                const ScalarInputConfig& scalar = scalars[config.exactScalarIndex];
                if (branch != nullptr) {
                    assignExactScalar(*branch, scalar);
                }
                vars.set(config.varSlot, scalar.numericValue());
            } else {
                const long double value = evalNumber(config.formula, context);
                if (branch != nullptr) {
                    assignNumericValue(*branch, value);
                }
                vars.set(config.varSlot, value);
            }
            continue;
        }

        if (config.collectionSlot < 0 || !collections.built[config.collectionSlot]) {
            throw runtime_error("Unknown output collection: " + config.collection);
        }
        const RuntimeCollection& collection = collections.runtime[config.collectionSlot];
        for (int slot = 0; slot < config.slots; ++slot) {
            if (slot >= static_cast<int>(collection.objects.size()) || branchIndices[index][slot] == kNoBranch) {
                continue;
            }
            EvalContext context;
            context.vars = &vars;
            context.collections = &collections;
            context.currentCollection = &collection;
            context.currentObject = &collection.objects[slot];

            OutputBranchRuntime& branch = treeState.branches[branchIndices[index][slot]];
            assignNumericValue(branch, evalNumber(config.formula, context));
        }
    }
}

// vars: scratch copy of baseVars that receives the tree's scalar outputs.
void fillOutputTree(OutputTreeState& treeState,
                    const EventCollections& collections,
                    const EventVars& baseVars,
                    EventVars& vars,
                    const vector<ScalarInputConfig>& scalars,
                    bool isMC) {
    resetTreeValues(treeState);

    vars = baseVars;
    fillOutputGroup(treeState.config.regularScalars, treeState.regularBranches, treeState, vars, collections, scalars, isMC);
    fillOutputGroup(treeState.config.extremaScalars, treeState.extremaBranches, treeState, vars, collections, scalars, isMC);

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
        if (tree->GetBranch(branch.c_str())) {
            tree->SetBranchStatus(branch.c_str(), 1);
        }
    }
    tree->SetCacheSize(50 * 1024 * 1024);
    for (const auto& branch : activeBranches) {
        if (tree->GetBranch(branch.c_str())) {
            tree->AddBranchToCache(branch.c_str(), true);
        }
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
        Long64_t observedMax = 0;
        if (tree->GetBranch(sizeBranch.c_str())) {
            observedMax = static_cast<Long64_t>(llround(tree->GetMaximum(sizeBranch.c_str())));
            if (observedMax < 0) {
                observedMax = 0;
            }
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

// Persist each thread's filled trees to its temp ROOT file and close the file. A write error
// (e.g. a full node-local $TMPDIR) fails the batch instead of silently dropping that thread's
// entries. The temp file on disk is kept so the batch merge can read it; cleanupThreadResult()
// unlinks it later.
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
        const bool writeError = result.tempFile->TestBit(TFile::kWriteError);
        delete result.tempFile;
        result.tempFile = nullptr;
        // TFile::Close() deletes the TTree objects owned by the file, so the
        // cached pointers in outputTrees are now dangling — null them out.
        for (auto& treeState : result.outputTrees) {
            treeState.tree = nullptr;
        }
        if (writeError) {
            throw runtime_error("Write error on thread temp file " + result.tempFilePath);
        }
        if (!result.tempFilePath.empty()) {
            paths.push_back(result.tempFilePath);
        }
    }
    return paths;
}

// Merge ROOT files holding the configured trees into one output by copying the compressed
// baskets (TFileMerger fast mode; every convert output uses kOutputCompression), writing to a
// hidden temporary name, checking each tree's entry count against the inputs, then renaming.
// Unreadable inputs, missing trees or a failed merge throw instead of being skipped. Fast
// merging never re-optimises baskets, so it cannot hit the 1 GB TBufferFile limit that the old
// TTree::MergeTrees path did.
void fastMergeRootFiles(const vector<string>& inputPaths,
                        const fs::path& outputPath,
                        const vector<TreeConfig>& treeConfigs) {
    if (inputPaths.empty()) {
        throw runtime_error("No inputs to merge into " + outputPath.string());
    }
    vector<Long64_t> expectedEntries(treeConfigs.size(), 0);
    for (const auto& input : inputPaths) {
        unique_ptr<TFile> file(TFile::Open(input.c_str(), "READ"));
        if (!file || file->IsZombie()) {
            throw runtime_error("Cannot open merge input " + input);
        }
        for (size_t i = 0; i < treeConfigs.size(); ++i) {
            TTree* tree = dynamic_cast<TTree*>(file->Get(treeConfigs[i].name.c_str()));
            if (tree == nullptr) {
                throw runtime_error("Merge input " + input + " lacks tree " + treeConfigs[i].name);
            }
            expectedEntries[i] += tree->GetEntries();
        }
    }

    const fs::path tempPath = outputPath.parent_path() /
        ("." + outputPath.filename().string() + ".tmp_" + to_string(static_cast<long long>(getpid())));
    try {
        {
            TFileMerger merger(kFALSE, kFALSE);
            merger.SetPrintLevel(0);
            merger.SetFastMethod(kTRUE);
            if (!merger.OutputFile(tempPath.string().c_str(), "RECREATE", kOutputCompression)) {
                throw runtime_error("Cannot create merge output " + tempPath.string());
            }
            for (const auto& input : inputPaths) {
                if (!merger.AddFile(input.c_str(), kFALSE)) {
                    throw runtime_error("Cannot add merge input " + input);
                }
            }
            if (!merger.Merge()) {
                throw runtime_error("TFileMerger failed for " + outputPath.string());
            }
        }
        unique_ptr<TFile> merged(TFile::Open(tempPath.string().c_str(), "READ"));
        if (!merged || merged->IsZombie()) {
            throw runtime_error("Cannot reopen merged file " + tempPath.string());
        }
        for (size_t i = 0; i < treeConfigs.size(); ++i) {
            TTree* tree = dynamic_cast<TTree*>(merged->Get(treeConfigs[i].name.c_str()));
            const Long64_t found = (tree != nullptr) ? tree->GetEntries() : -1;
            if (found != expectedEntries[i]) {
                throw runtime_error("Merged entry count mismatch for tree " + treeConfigs[i].name +
                                    " in " + outputPath.string() + ": expected " +
                                    to_string(expectedEntries[i]) + ", got " + to_string(found));
            }
        }
    } catch (...) {
        std::error_code ignored;
        fs::remove(tempPath, ignored);
        throw;
    }
    std::error_code ec;
    fs::rename(tempPath, outputPath, ec);
    if (ec) {
        throw runtime_error("Failed to move merged file to " + outputPath.string() + ": " + ec.message());
    }
}

// Remote opens are serialised: concurrent TFile::Open of root:// URLs from OpenMP threads
// races in TNetXNGFile::SetEnv (getenv/setenv) and segfaulted ~2% of the data batches.
mutex& remoteOpenMutex() {
    static mutex instance;
    return instance;
}

unique_ptr<TFile> openInputFileWithRetry(const string& inputFileName) {
    const bool remoteInput = startsWith(inputFileName, "root://");
    const int maxRetries = remoteInput ? kRemoteInputOpenRetries : 0;

    for (int retry = 0; retry <= maxRetries; ++retry) {
        unique_ptr<TFile> inputFile;
        if (remoteInput) {
            lock_guard<mutex> lock(remoteOpenMutex());
            inputFile.reset(TFile::Open(inputFileName.c_str(), "READ"));
        } else {
            inputFile.reset(TFile::Open(inputFileName.c_str(), "READ"));
        }
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

struct FileProcessResult {
    Long64_t rawEntries = 0;
    // MC: weight_pu sums over every entry of the file (all generated events, before any
    // selection) for the absolute pileup normalisation.
    long double sumWeightPu = 0.L;
    long double sumWeightPuUp = 0.L;
    long double sumWeightPuDown = 0.L;
    // MC: genWeight sums over every processed generated event (before any selection), the
    // denominators of the signed, absolutely normalized MC event weights downstream.
    long double sumGenWeight = 0.L;
    long double sumGenWeightPu = 0.L;
    long double sumGenWeightPuUp = 0.L;
    long double sumGenWeightPuDown = 0.L;
    // Data: processed (run, lumi) pairs passing the lumi mask.
    set<pair<UInt_t, UInt_t>> lumis;
};

// Jet energy corrections, Type-1 MET, and the systematic configurations of the
// single-pass conversion. For each configuration the corrected jets are written
// into the AK4/AK8 input buffers and the corrected MET into the MET input
// scalars, before the selections and outputs of that configuration read them.
//
// Nominal chain (nominal_correction = hlt_jec):
//  - The AK4 ScoutingPFJetRecluster2 inputs are stored with the production
//    Winter24HLT_V1 L1L2L3Res JEC (AK4PFHLT, true event rho) applied; raw =
//    stored * (1 - rawFactor) for pT and mass. The stored AK4 pT and mass are the
//    nominal JEC result.
//  - The event rho is not stored. recoverRho recovers it from the AK4 production
//    JEC and checks that every AK4 jet reproduces its stored factor with it.
//  - The AK8 ScoutingFatPFJetRecluster inputs are stored raw and get the official
//    Winter24HLT_V1 L1L2L3Res JEC (AK8PFHLT) with the recovered rho.
//  - MC jets, AK4 and AK8, are smeared with the offline AK4PFPuppi JER proxy
//    (stochastic JERSmear, GenPt = -1).
//  - The AK8 soft-drop mass takes every factor of the jet pT except L1FastJet
//    (grooming removes most of the pileup) and, for MC with apply_jms_jmr, the
//    JMS scale and the stochastic JMR smearing.
//  - MET is Type-1 corrected with the AK4 jets: the corrected minus the
//    L1FastJet-only muon-subtracted pT of the jets whose corrected (unsmeared)
//    muon-subtracted pT is above 15 GeV and whose EM fraction is below 0.9.
// Data get the JEC and the Type-1 MET only.
//
// Each systematic configuration (MC) changes one ingredient:
//  - jes_up/down scale every jet pT, mass, and soft-drop mass by 1 +- jes_shift
//    after the JER smearing, and the MET through Type-1;
//  - jer_up/down smear with the JER scale factor up/down and the same random
//    numbers (JERSmear hashes JetPt, JetEta, Rho, and EventID, which are the
//    same in every configuration);
//  - jms_up/down and jmr_up/down move JMS and JMR by their uncertainties.
struct JetCorrectionInputs {
    InputCollectionConfig* ak4 = nullptr;
    InputCollectionConfig* ak8 = nullptr;
    ArrayInputConfig* ak4Pt = nullptr;
    ArrayInputConfig* ak4Eta = nullptr;
    ArrayInputConfig* ak4Phi = nullptr;
    ArrayInputConfig* ak4Mass = nullptr;
    ArrayInputConfig* ak4Area = nullptr;
    ArrayInputConfig* ak4RawFactor = nullptr;
    ArrayInputConfig* ak4MuEF = nullptr;
    ArrayInputConfig* ak4ChEmEF = nullptr;
    ArrayInputConfig* ak4NeEmEF = nullptr;
    ArrayInputConfig* ak8Pt = nullptr;
    ArrayInputConfig* ak8Eta = nullptr;
    ArrayInputConfig* ak8Phi = nullptr;
    ArrayInputConfig* ak8Mass = nullptr;
    ArrayInputConfig* ak8Msoftdrop = nullptr;
    ArrayInputConfig* ak8Area = nullptr;
    // scouting_to_offline categories: parton flavour for MC, tagger scores for data.
    const ArrayInputConfig* ak4Flavour = nullptr;
    const ArrayInputConfig* ak4Tag = nullptr;
    const ArrayInputConfig* ak8Flavour = nullptr;
    const ArrayInputConfig* ak8TagXbb = nullptr;
    const ArrayInputConfig* ak8TagQcd = nullptr;
    ScalarInputConfig* metPt = nullptr;
    ScalarInputConfig* metPhi = nullptr;
    const ScalarInputConfig* run = nullptr;
    const ScalarInputConfig* luminosityBlock = nullptr;
    const ScalarInputConfig* event = nullptr;
};

// Configuration-independent quantities of one jet in one event.
struct CorrectedJet {
    double rawPt = 0.;
    double rawMass = 0.;
    double rawMsoftdrop = 0.;           // AK8
    double eta = 0.;
    double phi = 0.;
    double area = 0.;
    double storedFactor = 1.;           // stored / raw pT: the AK4 production JEC
    double jecFactor = 1.;              // nominal jet energy correction
    double l1Factor = 1.;               // L1FastJet part of the official JEC
    double msoftdropFactor = 1.;        // nominal soft-drop mass correction (AK8)
    double jerSmear[3] = {1., 1., 1.};  // JER smearing for scale factor nom / up / down
    double jmrRandom = 0.;              // standard-normal number of the JMR smearing (AK8)
    double rawPtNoMuon = 0.;            // raw pT without the muon energy (AK4)
    bool type1 = false;                 // AK4 jet propagated to the Type-1 MET
};

struct JetCorrectionEventState {
    vector<CorrectedJet> ak4;
    vector<CorrectedJet> ak8;
    double rho = 0.;
    double rawMetPx = 0.;
    double rawMetPy = 0.;
};

uint64_t splitMix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

uint64_t floatBits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

// Standard-normal number fixed by the event and the stored jet, so every
// configuration of an event smears the jet with the same random number.
double jetGaussian(ULong64_t eventId, float pt, float eta, float phi) {
    uint64_t hash = splitMix64(static_cast<uint64_t>(eventId));
    hash = splitMix64(hash ^ floatBits(pt));
    hash = splitMix64(hash ^ ((floatBits(eta) << 32) | floatBits(phi)));
    const uint64_t hash2 = splitMix64(hash);
    const double kInverse2Pow53 = 1.0 / 9007199254740992.0;
    const double u1 = (static_cast<double>(hash >> 11) + 0.5) * kInverse2Pow53;
    const double u2 = (static_cast<double>(hash2 >> 11) + 0.5) * kInverse2Pow53;
    return sqrt(-2.0 * log(u1)) * cos(kTwoPi * u2);
}

double medianOf(vector<double> values) {
    sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return (values.size() % 2 == 1) ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

void requireCorrectionInputs(const vector<correction::Variable>& inputs,
                             const vector<string>& expected,
                             const string& name) {
    bool matches = (inputs.size() == expected.size());
    for (size_t index = 0; matches && index < inputs.size(); ++index) {
        matches = (inputs[index].name() == expected[index]);
    }
    if (!matches) {
        ostringstream ss;
        ss << "Correction " << name << " does not have the expected inputs (";
        for (size_t index = 0; index < expected.size(); ++index) {
            ss << (index == 0 ? "" : ", ") << expected[index];
        }
        ss << ")";
        throw runtime_error(ss.str());
    }
}

ArrayInputConfig* findCollectionField(InputCollectionConfig& collection, const string& suffix) {
    const string name = collection.name + "_" + suffix;
    for (auto& field : collection.fields) {
        if (field.name == name) {
            return &field;
        }
    }
    return nullptr;
}

ArrayInputConfig& requireFloatField(InputCollectionConfig& collection, const string& suffix) {
    ArrayInputConfig* field = findCollectionField(collection, suffix);
    if (field == nullptr || field->type != DataType::Float || field->onlyMC || field->optional) {
        throw runtime_error("jet_pt_correction requires the float input field " + collection.name + "_" +
                            suffix + " (neither onlyMC nor optional) in branch.json");
    }
    return *field;
}

ScalarInputConfig& requireInputScalar(BranchConfig& branchConfig, const string& name, DataType type) {
    for (auto& scalar : branchConfig.scalars) {
        if (scalar.name == name && scalar.type == type && !scalar.onlyMC && !scalar.optional) {
            return scalar;
        }
    }
    throw runtime_error("jet_pt_correction requires the input scalar " + name +
                        " with its NanoAOD type (neither onlyMC nor optional) in branch.json");
}

int correctedCollectionSize(const InputCollectionConfig& collection, const EventVars& vars) {
    if (!vars.has(collection.sizeSlot)) {
        throw runtime_error("Input collection size not found: " + collection.sizeName);
    }
    return max(0, min(static_cast<int>(vars.values[collection.sizeSlot]), collection.maxSize));
}

bool referencesIdentifier(const ExprPtr& expr, const string& name) {
    if (!expr) {
        return false;
    }
    if (expr->kind == ExprKind::Identifier && expr->text == name) {
        return true;
    }
    if (referencesIdentifier(expr->lhs, name) || referencesIdentifier(expr->rhs, name)) {
        return true;
    }
    for (const auto& arg : expr->args) {
        if (referencesIdentifier(arg, name)) {
            return true;
        }
    }
    return false;
}

class JetPtCorrector {
public:
    void initialize(const JetPtCorrectionConfig& cfg) {
        cfg_ = cfg;
        hltJec_ = (cfg_.nominalCorrection == "hlt_jec");
        mcConfigurations_.assign(1, CorrectionConfiguration());
        dataConfigurations_.assign(1, CorrectionConfiguration());
        if (!cfg_.enabled) {
            return;
        }

        const vector<string> jecInputs = {"JetA", "JetEta", "JetPhi", "JetPt", "Rho"};
        jecAk4Set_ = correction::CorrectionSet::from_file(cfg_.jecAk4File);
        jecAk4_ = jecAk4Set_->compound().at(cfg_.jecAk4Name);
        jecAk4L1_ = jecAk4Set_->at(cfg_.jecAk4L1Name);
        requireCorrectionInputs(jecAk4_->inputs(), jecInputs, cfg_.jecAk4Name);
        requireCorrectionInputs(jecAk4L1_->inputs(), jecInputs, cfg_.jecAk4L1Name);
        if (hltJec_) {
            jecAk8Set_ = correction::CorrectionSet::from_file(cfg_.jecAk8File);
            jecAk8_ = jecAk8Set_->compound().at(cfg_.jecAk8Name);
            jecAk8L1_ = jecAk8Set_->at(cfg_.jecAk8L1Name);
            requireCorrectionInputs(jecAk8_->inputs(), jecInputs, cfg_.jecAk8Name);
            requireCorrectionInputs(jecAk8L1_->inputs(), jecInputs, cfg_.jecAk8L1Name);
        } else {
            offlineSet_ = correction::CorrectionSet::from_file(cfg_.correctionsFile);
            ak4OfflineMc_ = offlineSet_->at("AK4_plain_MC");
            ak4OfflineData_ = offlineSet_->at("AK4_plain_Data2024");
            ak8OfflineMc_ = offlineSet_->at("AK8_MC");
            ak8OfflineData_ = offlineSet_->at("AK8_Data2024");
        }

        jesJerSet_ = correction::CorrectionSet::from_file(cfg_.jesJerFile);
        jerResolution_ = jesJerSet_->at(cfg_.jerResolutionName);
        jerScaleFactor_ = jesJerSet_->at(cfg_.jerScaleFactorName);
        requireCorrectionInputs(jerResolution_->inputs(), {"JetEta", "JetPt", "Rho"}, cfg_.jerResolutionName);
        requireCorrectionInputs(jerScaleFactor_->inputs(), {"JetEta", "JetPt", "systematic"},
                                cfg_.jerScaleFactorName);
        jerSmearSet_ = correction::CorrectionSet::from_file(cfg_.jerSmearFile);
        jerSmear_ = jerSmearSet_->at("JERSmear");
        requireCorrectionInputs(jerSmear_->inputs(),
                                {"JetPt", "JetEta", "GenPt", "Rho", "EventID", "JER", "JERSF"}, "JERSmear");

        if (cfg_.applyJmsJmr) {
            const JsonValue results = loadJsonPath(cfg_.jmsJmrResultsFile);
            jms_ = static_cast<double>(results.at("JMS").asNumber());
            jmsErr_ = static_cast<double>(results.at("JMS_err").asNumber());
            jmr_ = static_cast<double>(results.at("JMR").asNumber());
            jmrErr_ = static_cast<double>(results.at("JMR_err").asNumber());
            // Relative soft-drop mass resolution of the MC W peak of the JMS/JMR fit.
            const JsonValue& mcPeak = results.at("pass").at("mc");
            jmrRelResolution_ = static_cast<double>(mcPeak.at("sigma").asNumber() / mcPeak.at("mu").asNumber());
        }

        mcConfigurations_.assign(1, makeConfiguration(cfg_.debugNominalConfiguration));
        mcConfigurations_.front().variation.clear();
        if (cfg_.debugNominalConfiguration == "nominal") {
            for (const auto& variation : cfg_.variations) {
                mcConfigurations_.push_back(makeConfiguration(variation));
            }
        }
    }

    bool enabled() const { return cfg_.enabled; }

    // The first configuration fills the nominal trees; each further one fills the
    // <tree>__<variation> trees of its variation.
    const vector<CorrectionConfiguration>& configurations(bool isMC) const {
        return isMC ? mcConfigurations_ : dataConfigurations_;
    }

    string describe(bool isMC) const {
        ostringstream ss;
        ss << "nominal_correction = " << cfg_.nominalCorrection << ", configurations =";
        for (const auto& configuration : configurations(isMC)) {
            ss << ' ' << (configuration.variation.empty() ? "nominal" : configuration.variation);
        }
        if (isMC && cfg_.debugNominalConfiguration != "nominal") {
            ss << " (validation: nominal trees filled with " << cfg_.debugNominalConfiguration << ")";
        }
        if (isMC) {
            ss << ", JES shift = " << cfg_.jesShift;
        }
        if (isMC && cfg_.applyJmsJmr) {
            ss << ", JMS = " << jms_ << " +- " << jmsErr_ << ", JMR = " << jmr_ << " +- " << jmrErr_
               << " (relative msoftdrop resolution " << jmrRelResolution_ << ")";
        }
        return ss.str();
    }

    JetCorrectionInputs resolveInputs(BranchConfig& branchConfig) const {
        JetCorrectionInputs in;
        for (auto& collection : branchConfig.collections) {
            if (collection.name == kAk4JetCollection) {
                in.ak4 = &collection;
            } else if (collection.name == kAk8JetCollection) {
                in.ak8 = &collection;
            }
        }
        if (in.ak4 == nullptr || in.ak8 == nullptr) {
            throw runtime_error(string("jet_pt_correction requires the input collections ") +
                                kAk4JetCollection + " and " + kAk8JetCollection + " in branch.json");
        }
        in.ak4Pt = &requireFloatField(*in.ak4, "pt");
        in.ak4Eta = &requireFloatField(*in.ak4, "eta");
        in.ak4Phi = &requireFloatField(*in.ak4, "phi");
        in.ak4Mass = &requireFloatField(*in.ak4, "mass");
        in.ak4Area = &requireFloatField(*in.ak4, "area");
        in.ak4RawFactor = &requireFloatField(*in.ak4, "rawFactor");
        in.ak4MuEF = &requireFloatField(*in.ak4, "muEF");
        in.ak4ChEmEF = &requireFloatField(*in.ak4, "chEmEF");
        in.ak4NeEmEF = &requireFloatField(*in.ak4, "neEmEF");
        in.ak8Pt = &requireFloatField(*in.ak8, "pt");
        in.ak8Eta = &requireFloatField(*in.ak8, "eta");
        in.ak8Phi = &requireFloatField(*in.ak8, "phi");
        in.ak8Mass = &requireFloatField(*in.ak8, "mass");
        in.ak8Msoftdrop = &requireFloatField(*in.ak8, "msoftdrop");
        in.ak8Area = &requireFloatField(*in.ak8, "area");
        if (!hltJec_) {
            in.ak4Flavour = findCollectionField(*in.ak4, "partonFlavour");
            in.ak4Tag = findCollectionField(*in.ak4, "scoutUParT_probb");
            in.ak8Flavour = findCollectionField(*in.ak8, "partonFlavour");
            in.ak8TagXbb = findCollectionField(*in.ak8, "scoutGlobalParT_prob_Xbb");
            in.ak8TagQcd = findCollectionField(*in.ak8, "scoutGlobalParT_prob_QCD");
        }
        in.metPt = &requireInputScalar(branchConfig, kMetPtScalar, DataType::Float);
        in.metPhi = &requireInputScalar(branchConfig, kMetPhiScalar, DataType::Float);
        in.run = &requireInputScalar(branchConfig, "run", DataType::UInt);
        in.luminosityBlock = &requireInputScalar(branchConfig, "luminosityBlock", DataType::UInt);
        in.event = &requireInputScalar(branchConfig, "event", DataType::ULong64);
        return in;
    }

    // Reads the stored values of the current entry, before any configuration
    // overwrites them, and evaluates everything the configurations share.
    void prepareEvent(const JetCorrectionInputs& in,
                      const EventVars& vars,
                      bool isMC,
                      JetCorrectionEventState& state) const {
        const int nAk4 = correctedCollectionSize(*in.ak4, vars);
        const int nAk8 = correctedCollectionSize(*in.ak8, vars);
        state.ak4.assign(nAk4, CorrectedJet());
        state.ak8.assign(nAk8, CorrectedJet());
        for (int i = 0; i < nAk4; ++i) {
            CorrectedJet& jet = state.ak4[i];
            const double rawScale = 1. - in.ak4RawFactor->floatValues[i];
            jet.rawPt = in.ak4Pt->floatValues[i] * rawScale;
            jet.rawMass = in.ak4Mass->floatValues[i] * rawScale;
            jet.eta = in.ak4Eta->floatValues[i];
            jet.phi = in.ak4Phi->floatValues[i];
            jet.area = in.ak4Area->floatValues[i];
            jet.storedFactor = 1. / rawScale;
        }
        for (int i = 0; i < nAk8; ++i) {
            CorrectedJet& jet = state.ak8[i];
            jet.rawPt = in.ak8Pt->floatValues[i];
            jet.rawMass = in.ak8Mass->floatValues[i];
            jet.rawMsoftdrop = in.ak8Msoftdrop->floatValues[i];
            jet.eta = in.ak8Eta->floatValues[i];
            jet.phi = in.ak8Phi->floatValues[i];
            jet.area = in.ak8Area->floatValues[i];
        }

        // rho enters only through the jets.
        state.rho = (nAk4 + nAk8 > 0) ? recoverRho(in, state.ak4) : 0.;
        const double jerRho = min(state.rho, nextafter(kJerRhoUpperEdge, 0.));
        const ULong64_t eventId = in.event->ulong64Value;
        // JERSmear's EventID input is an int (entropy of its hash only).
        const int eventSeed = static_cast<int>(eventId & 0x7FFFFFFFULL);

        for (int i = 0; i < nAk4; ++i) {
            CorrectedJet& jet = state.ak4[i];
            if (!(jet.rawPt > 0.)) {
                continue;
            }
            jet.l1Factor = jecAk4L1_->evaluate({jet.area, jet.eta, jet.phi, jet.rawPt, state.rho});
            jet.jecFactor = hltJec_ ? jet.storedFactor
                                    : offlineResponseFactor(in, false, isMC, i) * jet.storedFactor;
            jet.rawPtNoMuon = jet.rawPt * (1. - in.ak4MuEF->floatValues[i]);
            const double emFraction = in.ak4ChEmEF->floatValues[i] + in.ak4NeEmEF->floatValues[i];
            jet.type1 = (jet.rawPtNoMuon * jet.jecFactor > kType1JetPtThreshold) &&
                        (emFraction < kType1MaxEmFraction);
            if (isMC) {
                fillJerSmear(jet, state.rho, jerRho, eventSeed);
            }
        }

        for (int i = 0; i < nAk8; ++i) {
            CorrectedJet& jet = state.ak8[i];
            if (!(jet.rawPt > 0.)) {
                continue;
            }
            if (hltJec_) {
                const vector<correction::Variable::Type> jecArgs = {jet.area, jet.eta, jet.phi, jet.rawPt, state.rho};
                jet.jecFactor = jecAk8_->evaluate(jecArgs);
                jet.l1Factor = jecAk8L1_->evaluate(jecArgs);
                jet.msoftdropFactor = jet.jecFactor / jet.l1Factor;
            } else {
                jet.jecFactor = offlineResponseFactor(in, true, isMC, i);
                jet.msoftdropFactor = jet.jecFactor;
            }
            if (isMC) {
                fillJerSmear(jet, state.rho, jerRho, eventSeed);
                jet.jmrRandom = jetGaussian(eventId, in.ak8Pt->floatValues[i], in.ak8Eta->floatValues[i],
                                            in.ak8Phi->floatValues[i]);
            }
        }

        const double rawMetPt = in.metPt->floatValue;
        const double rawMetPhi = in.metPhi->floatValue;
        state.rawMetPx = rawMetPt * cos(rawMetPhi);
        state.rawMetPy = rawMetPt * sin(rawMetPhi);
    }

    // Writes the jets and the MET of one configuration into the input buffers,
    // the MET input scalars, and vars.
    void applyConfiguration(const JetCorrectionInputs& in,
                            const JetCorrectionEventState& state,
                            const CorrectionConfiguration& configuration,
                            bool isMC,
                            EventVars& vars) const {
        double type1Px = 0.;
        double type1Py = 0.;
        for (size_t i = 0; i < state.ak4.size(); ++i) {
            const CorrectedJet& jet = state.ak4[i];
            const double factor = jet.jecFactor * variationFactor(jet, configuration, isMC);
            in.ak4Pt->floatValues[i] = static_cast<Float_t>(jet.rawPt * factor);
            in.ak4Mass->floatValues[i] = static_cast<Float_t>(jet.rawMass * factor);
            if (jet.type1) {
                const double shift = jet.rawPtNoMuon * (factor - jet.l1Factor);
                type1Px += shift * cos(jet.phi);
                type1Py += shift * sin(jet.phi);
            }
        }
        for (size_t i = 0; i < state.ak8.size(); ++i) {
            const CorrectedJet& jet = state.ak8[i];
            const double variation = variationFactor(jet, configuration, isMC);
            double msoftdropScale = jet.msoftdropFactor * variation;
            if (isMC && cfg_.applyJmsJmr) {
                const double jmrWidth =
                    sqrt(max(configuration.jmr * configuration.jmr - 1., 0.)) * jmrRelResolution_;
                msoftdropScale *= configuration.jms * (1. + jet.jmrRandom * jmrWidth);
            }
            in.ak8Pt->floatValues[i] = static_cast<Float_t>(jet.rawPt * jet.jecFactor * variation);
            in.ak8Mass->floatValues[i] = static_cast<Float_t>(jet.rawMass * jet.jecFactor * variation);
            in.ak8Msoftdrop->floatValues[i] = static_cast<Float_t>(jet.rawMsoftdrop * msoftdropScale);
        }
        const double metPx = state.rawMetPx - type1Px;
        const double metPy = state.rawMetPy - type1Py;
        in.metPt->floatValue = static_cast<Float_t>(hypot(metPx, metPy));
        in.metPhi->floatValue = static_cast<Float_t>(atan2(metPy, metPx));
        vars.set(in.metPt->varSlot, in.metPt->floatValue);
        vars.set(in.metPhi->varSlot, in.metPhi->floatValue);
    }

private:
    CorrectionConfiguration makeConfiguration(const string& name) const {
        CorrectionConfiguration configuration;
        configuration.variation = (name == "nominal") ? "" : name;
        if (cfg_.applyJmsJmr) {
            configuration.jms = jms_;
            configuration.jmr = jmr_;
        }
        if (name == "jes_up") {
            configuration.jesFactor = 1. + cfg_.jesShift;
        } else if (name == "jes_down") {
            configuration.jesFactor = 1. - cfg_.jesShift;
        } else if (name == "jer_up") {
            configuration.jerSfIndex = 1;
        } else if (name == "jer_down") {
            configuration.jerSfIndex = 2;
        } else if (name == "jms_up") {
            configuration.jms = jms_ + jmsErr_;
        } else if (name == "jms_down") {
            configuration.jms = jms_ - jmsErr_;
        } else if (name == "jmr_up") {
            configuration.jmr = jmr_ + jmrErr_;
        } else if (name == "jmr_down") {
            configuration.jmr = jmr_ - jmrErr_;
        }
        return configuration;
    }

    static double variationFactor(const CorrectedJet& jet,
                                  const CorrectionConfiguration& configuration,
                                  bool isMC) {
        return isMC ? jet.jerSmear[configuration.jerSfIndex] * configuration.jesFactor : 1.;
    }

    void fillJerSmear(CorrectedJet& jet, double rho, double jerRho, int eventSeed) const {
        const double pt = jet.rawPt * jet.jecFactor;
        const double resolution = jerResolution_->evaluate({jet.eta, pt, jerRho});
        const char* const systematics[3] = {"nom", "up", "down"};
        for (int k = 0; k < 3; ++k) {
            const double scaleFactor = jerScaleFactor_->evaluate({jet.eta, pt, string(systematics[k])});
            jet.jerSmear[k] = jerSmear_->evaluate({pt, jet.eta, -1.0, rho, eventSeed, resolution, scaleFactor});
        }
    }

    // Legacy scouting->offline response SF (scouting_to_offline) of the stored
    // jet, with a b/light (MC) or btag/nobtag (data) category.
    double offlineResponseFactor(const JetCorrectionInputs& in, bool isAK8, bool isMC, int index) const {
        const ArrayInputConfig& ptField = isAK8 ? *in.ak8Pt : *in.ak4Pt;
        const ArrayInputConfig& etaField = isAK8 ? *in.ak8Eta : *in.ak4Eta;
        const ArrayInputConfig* flavourField = isAK8 ? in.ak8Flavour : in.ak4Flavour;
        const ArrayInputConfig* tagField = isAK8 ? in.ak8TagXbb : in.ak4Tag;
        string category = "inclusive";
        if (isMC && flavourField != nullptr) {
            category = (std::abs(flavourField->valueAt(index) - 5.f) < 0.5f) ? "b" : "light";
        } else if (!isMC && tagField != nullptr) {
            float score = tagField->valueAt(index);
            if (isAK8 && in.ak8TagQcd != nullptr) {
                const float qcd = in.ak8TagQcd->valueAt(index);
                const float denom = score + qcd;
                score = (denom > 0.f) ? (score / denom) : 0.f;
            }
            const double tagThreshold = isAK8 ? cfg_.ak8TagThreshold : cfg_.ak4TagThreshold;
            category = (score >= static_cast<float>(tagThreshold)) ? "btag" : "nobtag";
        }
        const correction::Correction::Ref& correction =
            isMC ? (isAK8 ? ak8OfflineMc_ : ak4OfflineMc_) : (isAK8 ? ak8OfflineData_ : ak4OfflineData_);
        return correction->evaluate({category, static_cast<double>(etaField.valueAt(index)),
                                     static_cast<double>(ptField.valueAt(index))});
    }

    double ak4Jec(const CorrectedJet& jet, double eta, double phi, double rho) const {
        return jecAk4_->evaluate({jet.area, eta, phi, jet.rawPt, rho});
    }

    // The AK4 JEC as a function of rho, continued beyond the L1FastJet clamp
    // [0, kRhoRecoveryMax] by point reflection at the clamp edges: a stored factor
    // that the input precision puts just outside the clamped range then has a
    // solution at a distance from the edge instead of none.
    double extendedAk4Jec(const CorrectedJet& jet, double eta, double phi, double rho) const {
        if (rho > kRhoRecoveryMax) {
            return 2. * ak4Jec(jet, eta, phi, kRhoRecoveryMax) - ak4Jec(jet, eta, phi, 2. * kRhoRecoveryMax - rho);
        }
        if (rho < 0.) {
            return 2. * ak4Jec(jet, eta, phi, 0.) - ak4Jec(jet, eta, phi, -rho);
        }
        return ak4Jec(jet, eta, phi, rho);
    }

    // True when the JEC of the jet changes with rho by more than
    // kRhoRecoveryMinSensitivity across the clamp range, i.e. the stored factor
    // determines rho.
    bool carriesRho(const CorrectedJet& jet, double eta, double phi) const {
        return ak4Jec(jet, eta, phi, 0.) - ak4Jec(jet, eta, phi, kRhoRecoveryMax) >
               kRhoRecoveryMinSensitivity * jet.storedFactor;
    }

    // Solves JEC(rho) = stored factor, the JEC falling with rho, with the Illinois
    // (modified regula falsi) method on the clamp range widened by
    // kRhoRecoveryTolerance; the solution is clamped to [0, kRhoRecoveryMax].
    bool solveRho(const CorrectedJet& jet, double eta, double phi, double& rho) const {
        const auto residual = [&](double value) {
            return extendedAk4Jec(jet, eta, phi, value) - jet.storedFactor;
        };
        double low = -kRhoRecoveryTolerance;
        double high = kRhoRecoveryMax + kRhoRecoveryTolerance;
        double fLow = residual(low);
        double fHigh = residual(high);
        if (!(fLow >= 0. && fHigh <= 0. && fLow > fHigh)) {
            return false;
        }
        int side = 0;
        for (int iteration = 0; iteration < kRhoSolverMaxIterations; ++iteration) {
            rho = (low * fHigh - high * fLow) / (fHigh - fLow);
            const double fRho = residual(rho);
            if (fabs(fRho) <= kRhoSolverTolerance) {
                break;
            }
            if (fRho > 0.) {
                low = rho;
                fLow = fRho;
                if (side == 1) {
                    fHigh *= 0.5;
                }
                side = 1;
            } else {
                high = rho;
                fHigh = fRho;
                if (side == -1) {
                    fLow *= 0.5;
                }
                side = -1;
            }
        }
        rho = min(max(rho, 0.), kRhoRecoveryMax);
        return true;
    }

    // True when the jet's rho solution lies within kRhoRecoveryTolerance of rho:
    // its stored factor lies between the JEC at rho + tolerance and at rho -
    // tolerance. A jet that does not carry rho agrees with every rho.
    bool agreesWithRho(const CorrectedJet& jet, double rho) const {
        if (jet.storedFactor >= extendedAk4Jec(jet, jet.eta, jet.phi, rho + kRhoRecoveryTolerance) &&
            jet.storedFactor <= extendedAk4Jec(jet, jet.eta, jet.phi, rho - kRhoRecoveryTolerance)) {
            return true;
        }
        return !carriesRho(jet, jet.eta, jet.phi);
    }

    // Event rho from the AK4 production JEC: stored / raw = C(JetA, JetEta,
    // JetPhi, raw pT, rho) of the official compound. The first jet with a rho
    // solution gives the event rho when every other jet agrees with it;
    // otherwise consensusRho resolves the event from all jets.
    double recoverRho(const JetCorrectionInputs& in, const vector<CorrectedJet>& ak4) const {
        vector<const CorrectedJet*> usable;
        for (const auto& jet : ak4) {
            if (jet.rawPt > 0. && jet.storedFactor > kRhoRecoveryMinFactor) {
                usable.push_back(&jet);
            }
        }
        for (const CorrectedJet* jet : usable) {
            double rho = 0.;
            if (!carriesRho(*jet, jet->eta, jet->phi) || !solveRho(*jet, jet->eta, jet->phi, rho)) {
                continue;
            }
            const bool agree = all_of(usable.begin(), usable.end(), [&](const CorrectedJet* other) {
                return agreesWithRho(*other, rho);
            });
            if (agree) {
                return rho;
            }
            break;
        }
        return consensusRho(in, usable);
    }

    // Resolves the event rho when the first solution disagrees. A stored eta or
    // phi on a JEC bin edge can select a neighbouring bin of the production value,
    // so each jet that carries rho gives the solutions of its own bin and of the
    // bins a shift by kJecBinEdgeProbe in eta or phi reaches. The event rho is the
    // median of one solution per jet with all of them within
    // kRhoRecoveryTolerance of it.
    double consensusRho(const JetCorrectionInputs& in, const vector<const CorrectedJet*>& usable) const {
        const double shifts[5][2] = {{0., 0.},
                                     {kJecBinEdgeProbe, 0.},
                                     {-kJecBinEdgeProbe, 0.},
                                     {0., kJecBinEdgeProbe},
                                     {0., -kJecBinEdgeProbe}};
        vector<vector<double>> solutions;
        for (const CorrectedJet* jet : usable) {
            const double ownBin = ak4Jec(*jet, jet->eta, jet->phi, kJecBinProbeRho);
            vector<double> jetSolutions;
            bool carries = false;
            for (const auto& shift : shifts) {
                const double eta = jet->eta + shift[0];
                const double phi = jet->phi + shift[1];
                const bool ownPosition = (shift[0] == 0. && shift[1] == 0.);
                if (!ownPosition && ak4Jec(*jet, eta, phi, kJecBinProbeRho) == ownBin) {
                    continue;
                }
                if (!carriesRho(*jet, eta, phi)) {
                    continue;
                }
                carries = true;
                double rho = 0.;
                if (solveRho(*jet, eta, phi, rho)) {
                    jetSolutions.push_back(rho);
                }
            }
            if (!carries) {
                continue;
            }
            if (jetSolutions.empty()) {
                throw runtime_error("AK4 jet (pt = " + to_string(jet->rawPt * jet->storedFactor) +
                                    ", eta = " + to_string(jet->eta) + ", phi = " + to_string(jet->phi) +
                                    ") in event " + eventLabel(in) + " reproduces its stored JEC with no rho; "
                                    "the inputs were not corrected with " + cfg_.jecAk4Name);
            }
            solutions.push_back(std::move(jetSolutions));
        }
        if (solutions.empty()) {
            throw runtime_error("Cannot recover rho in event " + eventLabel(in) +
                                ": no AK4 jet carries the event rho in its production JEC");
        }
        vector<double> chosen(solutions.size());
        for (const auto& anchorSolutions : solutions) {
            for (const double anchor : anchorSolutions) {
                for (size_t j = 0; j < solutions.size(); ++j) {
                    chosen[j] = *min_element(solutions[j].begin(), solutions[j].end(),
                                             [&](double lhs, double rhs) {
                                                 return fabs(lhs - anchor) < fabs(rhs - anchor);
                                             });
                }
                const double rho = medianOf(chosen);
                if (all_of(chosen.begin(), chosen.end(),
                           [&](double value) { return fabs(value - rho) <= kRhoRecoveryTolerance; })) {
                    return rho;
                }
            }
        }
        throw runtime_error("AK4 jets disagree on the rho of their production JEC in event " + eventLabel(in) +
                            "; the inputs were not corrected with " + cfg_.jecAk4Name);
    }

    static string eventLabel(const JetCorrectionInputs& in) {
        return to_string(in.run->uintValue) + ":" + to_string(in.luminosityBlock->uintValue) + ":" +
               to_string(in.event->ulong64Value);
    }

    JetPtCorrectionConfig cfg_;
    bool hltJec_ = true;
    vector<CorrectionConfiguration> mcConfigurations_;
    vector<CorrectionConfiguration> dataConfigurations_;
    std::unique_ptr<correction::CorrectionSet> jecAk4Set_;
    std::unique_ptr<correction::CorrectionSet> jecAk8Set_;
    std::unique_ptr<correction::CorrectionSet> offlineSet_;
    std::unique_ptr<correction::CorrectionSet> jesJerSet_;
    std::unique_ptr<correction::CorrectionSet> jerSmearSet_;
    correction::CompoundCorrection::Ref jecAk4_, jecAk8_;
    correction::Correction::Ref jecAk4L1_, jecAk8L1_;
    correction::Correction::Ref ak4OfflineMc_, ak4OfflineData_, ak8OfflineMc_, ak8OfflineData_;
    correction::Correction::Ref jerResolution_, jerScaleFactor_, jerSmear_;
    double jms_ = 1.;
    double jmsErr_ = 0.;
    double jmr_ = 1.;
    double jmrErr_ = 0.;
    double jmrRelResolution_ = 0.;
};

void collectIdentifiers(const ExprPtr& expr, vector<string>& names) {
    if (!expr) {
        return;
    }
    if (expr->kind == ExprKind::Identifier) {
        names.push_back(expr->text);
    }
    collectIdentifiers(expr->lhs, names);
    collectIdentifiers(expr->rhs, names);
    for (const auto& arg : expr->args) {
        collectIdentifiers(arg, names);
    }
}

// The scalar outputs that the kept branches of a variation tree read: the kept
// scalars and every output scalar their formulas (and the formulas of kept
// collection slots) refer to through vars, transitively.
unordered_set<string> neededScalarOutputs(const TreeConfig& tree, const unordered_set<string>& kept) {
    unordered_map<string, const OutputScalarConfig*> scalarByName;
    for (const auto* group : {&tree.regularScalars, &tree.extremaScalars}) {
        for (const auto& config : *group) {
            if (config.collection.empty()) {
                scalarByName[config.name] = &config;
            }
        }
    }
    unordered_set<string> needed;
    vector<const OutputScalarConfig*> pending;
    const auto require = [&](const string& name) {
        const auto it = scalarByName.find(name);
        if (it != scalarByName.end() && needed.insert(name).second) {
            pending.push_back(it->second);
        }
    };
    const auto requireReferences = [&](const OutputScalarConfig& config) {
        vector<string> names;
        collectIdentifiers(config.formula, names);
        for (const auto& name : names) {
            require(name);
        }
    };
    for (const auto* group : {&tree.regularScalars, &tree.extremaScalars}) {
        for (const auto& config : *group) {
            if (config.collection.empty()) {
                if (kept.count(config.name) > 0) {
                    require(config.name);
                }
                continue;
            }
            for (int slot = 0; slot < config.slots; ++slot) {
                if (kept.count(config.name + "_" + to_string(slot + 1)) > 0) {
                    requireReferences(config);
                    break;
                }
            }
        }
    }
    while (!pending.empty()) {
        const OutputScalarConfig* config = pending.back();
        pending.pop_back();
        requireReferences(*config);
    }
    return needed;
}

unordered_set<string> outputBranchNames(const TreeConfig& tree, bool isMC) {
    unordered_set<string> names;
    for (const auto* group : {&tree.regularScalars, &tree.extremaScalars}) {
        for (const auto& config : *group) {
            if (config.onlyMC && !isMC) {
                continue;
            }
            if (config.collection.empty()) {
                names.insert(config.name);
                continue;
            }
            for (int slot = 0; slot < config.slots; ++slot) {
                names.insert(config.name + "_" + to_string(slot + 1));
            }
        }
    }
    return names;
}

// For MC, appends one <tree>__<variation> output tree per nominal tree and
// configured variation; each books only the branches listed for its nominal
// tree in jet_pt_correction.variation_branches.
void addVariationTrees(BranchConfig& branchConfig, const JetPtCorrectionConfig& cfg, bool isMC) {
    if (!cfg.enabled || !isMC || cfg.variations.empty() || cfg.debugNominalConfiguration != "nominal") {
        return;
    }
    unordered_set<string> nominalNames;
    for (const auto& tree : branchConfig.trees) {
        nominalNames.insert(tree.name);
    }
    for (const auto& item : cfg.variationBranches) {
        if (nominalNames.count(item.first) == 0) {
            throw runtime_error("jet_pt_correction.variation_branches names unknown output tree " + item.first);
        }
    }
    vector<TreeConfig> variationTrees;
    for (const auto& nominal : branchConfig.trees) {
        const auto listIt = cfg.variationBranches.find(nominal.name);
        if (listIt == cfg.variationBranches.end()) {
            throw runtime_error("jet_pt_correction.variation_branches has no branch list for tree " + nominal.name);
        }
        if (listIt->second.empty()) {
            throw runtime_error("jet_pt_correction.variation_branches[" + nominal.name + "] is empty");
        }
        const unordered_set<string> available = outputBranchNames(nominal, true);
        unordered_set<string> kept;
        for (const auto& name : listIt->second) {
            if (available.count(name) == 0) {
                throw runtime_error("jet_pt_correction.variation_branches[" + nominal.name + "] lists " + name +
                                    ", which is not an output branch of that tree");
            }
            kept.insert(name);
        }
        const unordered_set<string> needed = neededScalarOutputs(nominal, kept);
        for (const auto& variation : cfg.variations) {
            TreeConfig tree = nominal;
            tree.name = nominal.name + "__" + variation;
            tree.title = nominal.title + " [" + variation + "]";
            tree.variation = variation;
            tree.keptBranches = kept;
            tree.neededScalars = needed;
            variationTrees.push_back(std::move(tree));
        }
    }
    for (auto& tree : variationTrees) {
        branchConfig.trees.push_back(std::move(tree));
    }
}

// Runtime collections whose content depends on the jet corrections: a jet input
// source, a jet-dependent merge or deduplication partner, or an expression that
// reads a jet collection or field, a jet-dependent collection, or a corrected
// MET scalar. The other collections are built once per event and shared by
// every jet-correction configuration.
unordered_set<string> jetDependentCollections(const SelectionConfig& selectionConfig) {
    const string ak4Prefix = string(kAk4JetCollection) + "_";
    const string ak8Prefix = string(kAk8JetCollection) + "_";
    unordered_set<string> dependent;
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& name : selectionConfig.collectionOrder) {
            if (dependent.count(name) > 0) {
                continue;
            }
            const RuntimeCollectionConfig& config =
                selectionConfig.collections[selectionConfig.collectionSlotByName.at(name)];
            bool jetDependent = (config.source == kAk4JetCollection || config.source == kAk8JetCollection) ||
                                (!config.dedupCollection.empty() && dependent.count(config.dedupCollection) > 0);
            for (const auto& child : config.merge) {
                jetDependent = jetDependent || dependent.count(child) > 0;
            }
            vector<string> identifiers;
            collectIdentifiers(config.selectionExpr, identifiers);
            collectIdentifiers(config.dedupExpr, identifiers);
            collectIdentifiers(config.sortRule.expr, identifiers);
            for (const auto& identifier : identifiers) {
                jetDependent = jetDependent || dependent.count(identifier) > 0 ||
                               identifier == kAk4JetCollection || identifier == kAk8JetCollection ||
                               startsWith(identifier, ak4Prefix) || startsWith(identifier, ak8Prefix) ||
                               identifier == kMetPtScalar || identifier == kMetPhiScalar;
            }
            if (jetDependent) {
                dependent.insert(name);
                changed = true;
            }
        }
    }
    return dependent;
}

// The event preselection is evaluated once per event on the input scalars,
// before the jet corrections, so it must not read the MET scalars they change.
void requirePreselectionWithoutCorrectedInputs(const SelectionConfig& selectionConfig,
                                               const JetPtCorrectionConfig& cfg) {
    if (!cfg.enabled) {
        return;
    }
    for (const char* name : {kMetPtScalar, kMetPhiScalar}) {
        if (referencesIdentifier(selectionConfig.eventPreselection, name)) {
            throw runtime_error(string("event_preselection reads ") + name +
                                ", which the jet corrections change (Type-1 MET); cut on it in "
                                "tree_selection instead");
        }
    }
}

FileProcessResult processInputFile(const string& inputFileName,
                                   const AppConfig& appConfig,
                                   const SelectionConfig& selectionConfig,
                                   const SampleMeta& sampleMeta,
                                   const vector<PileupBin>& pileupWeights,
                                   const LumiMask* lumiMask,
                                   const JetPtCorrector& jetCorrector,
                                   BranchConfig& branchConfig,
                                   vector<OutputTreeState>& outputTrees) {
    FileProcessResult result;
    unique_ptr<TFile> inputFile;
    try {
        inputFile = openInputFileWithRetry(inputFileName);
    } catch (const runtime_error& ex) {
        throw SkippableFileError(ex.what());
    }

    // Data: every lumisection of this file passing the lumi mask counts as processed. They come
    // from the LuminosityBlocks tree, so lumisections without (selected) events are included;
    // files without that tree fall back to the (run, lumi) of their Events entries.
    bool lumisFromTree = false;
    if (!sampleMeta.isMC) {
        TTree* lumiTree = dynamic_cast<TTree*>(inputFile->Get("LuminosityBlocks"));
        if (lumiTree != nullptr && lumiTree->GetBranch("run") != nullptr &&
            lumiTree->GetBranch("luminosityBlock") != nullptr) {
            UInt_t lbRun = 0;
            UInt_t lbLumi = 0;
            lumiTree->SetBranchStatus("*", 0);
            lumiTree->SetBranchStatus("run", 1);
            lumiTree->SetBranchStatus("luminosityBlock", 1);
            if (lumiTree->SetBranchAddress("run", &lbRun) < 0 ||
                lumiTree->SetBranchAddress("luminosityBlock", &lbLumi) < 0) {
                throw runtime_error("Cannot bind LuminosityBlocks run/luminosityBlock in " + inputFileName);
            }
            const Long64_t nLumis = lumiTree->GetEntries();
            for (Long64_t i = 0; i < nLumis; ++i) {
                if (lumiTree->GetEntry(i) <= 0) {
                    throw runtime_error("Failed to read LuminosityBlocks entry " + to_string(i) +
                                        " in " + inputFileName);
                }
                if (lumiMask == nullptr || lumiMask->contains(lbRun, lbLumi)) {
                    result.lumis.emplace(lbRun, lbLumi);
                }
            }
            lumiTree->ResetBranchAddresses();
            lumisFromTree = true;
        }
    }

    TTree* tree = static_cast<TTree*>(inputFile->Get(appConfig.treeName.c_str()));
    if (!tree) {
        throw SkippableFileError("Tree " + appConfig.treeName + " not found in " + inputFileName);
    }

    const Long64_t nEntries = tree->GetEntries();
    if (nEntries == 0) {
        return result;
    }

    configureActiveBranches(tree, branchConfig, sampleMeta.isMC);
    ensureCollectionBufferCapacities(tree, branchConfig, sampleMeta.isMC);
    unordered_map<string, const ScalarInputConfig*> rawScalarByName = bindInputBranches(tree, branchConfig, sampleMeta.isMC);

    TheoryWeightBufs theoryInBuf;
    if (sampleMeta.isMC) {
        bindGenWeight(tree, theoryInBuf);
        if (sampleMeta.hasTheoryWeights) {
            activateTheoryInputBranches(tree, theoryInBuf);
        }
    }
    // Every branch is now registered with the TTreeCache: end the learning phase so the
    // remaining entries are prefetched instead of being read basket by basket.
    tree->StopCacheLearningPhase();

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

    // Per-file event state, reused across events.
    const EventVarLayout& varLayout = branchConfig.varLayout;

    // An entry is read completely only if it passes the event preselection. Before that, only
    // the branches of the scalars the preselection reads are read, plus what the all-event
    // bookkeeping below needs: Pileup_nTrueInt and genWeight (MC weight sums) and, for data without a
    // LuminosityBlocks tree, run/luminosityBlock (processed lumis). Rejected entries skip
    // decompressing and unpacking all other branches.
    set<int> preselectionSlots;
    collectVarSlots(selectionConfig.eventPreselection, preselectionSlots);
    if (sampleMeta.isMC) {
        preselectionSlots.insert(varLayout.puTrueInt);
        preselectionSlots.insert(varLayout.genWeight);
    } else if (!lumisFromTree) {
        preselectionSlots.insert(varLayout.run);
        preselectionSlots.insert(varLayout.luminosityBlock);
    }
    vector<TBranch*> preselectionBranches;
    for (const auto& scalar : branchConfig.scalars) {
        if (scalar.bound && preselectionSlots.count(scalar.varSlot)) {
            preselectionBranches.push_back(tree->GetBranch(scalar.branch.c_str()));
        }
    }
    if (sampleMeta.isMC && preselectionSlots.count(varLayout.genWeight)) {
        preselectionBranches.push_back(tree->GetBranch("genWeight"));
    }

    EventVars baseVars;
    EventVars treeVars;
    EventCollections collections;
    collections.inputs.resize(branchConfig.collections.size());
    collections.runtime.resize(selectionConfig.collections.size());
    vector<const ExprPtr*> treeCuts;
    for (const auto& treeState : outputTrees) {
        const auto cutIt = selectionConfig.treeSelections.find(treeState.config.selection);
        treeCuts.push_back(cutIt != selectionConfig.treeSelections.end() ? &cutIt->second : nullptr);
    }

    // Output trees (indices into outputTrees) filled by each jet-correction configuration: the
    // nominal one first, then one per variation (MC); without corrections only the nominal.
    const vector<CorrectionConfiguration>& configurations = jetCorrector.configurations(sampleMeta.isMC);
    vector<vector<size_t>> treesByConfiguration(configurations.size());
    for (size_t t = 0; t < outputTrees.size(); ++t) {
        const auto configurationIt = find_if(configurations.begin(), configurations.end(),
                                             [&](const CorrectionConfiguration& configuration) {
                                                 return configuration.variation == outputTrees[t].config.variation;
                                             });
        if (configurationIt == configurations.end()) {
            throw runtime_error("No jet correction configuration fills output tree " + outputTrees[t].config.name);
        }
        treesByConfiguration[configurationIt - configurations.begin()].push_back(t);
    }
    // With corrections, the AK4/AK8 input collections and the runtime collections that depend
    // on them (or on the corrected MET) are rebuilt for every configuration; the other runtime
    // collections are built once per event.
    JetCorrectionInputs correctionInputs;
    size_t ak4Input = 0;
    size_t ak8Input = 0;
    vector<unsigned char> jetDependent(selectionConfig.collections.size(), 0);
    if (jetCorrector.enabled()) {
        correctionInputs = jetCorrector.resolveInputs(branchConfig);
        ak4Input = static_cast<size_t>(correctionInputs.ak4 - branchConfig.collections.data());
        ak8Input = static_cast<size_t>(correctionInputs.ak8 - branchConfig.collections.data());
        for (const auto& name : jetDependentCollections(selectionConfig)) {
            jetDependent[selectionConfig.collectionSlotByName.at(name)] = 1;
        }
    }
    JetCorrectionEventState correctionState;
    EventVars configurationVars;

    vector<Long64_t> truncatedEvents(branchConfig.collections.size(), 0);
    result.rawEntries = applyLumiMask ? 0 : nEntries;
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
            ++result.rawEntries;
        }

        for (TBranch* branch : preselectionBranches) {
            if (branch->GetEntry(entry) < 0) {
                throw runtime_error("Failed to read branch " + string(branch->GetName()) + " of entry " +
                                    to_string(entry) + " of tree " + appConfig.treeName + " in " + inputFileName);
            }
        }

        const TheoryWeightBufs* theoryBufsPtr = sampleMeta.isMC ? &theoryInBuf : nullptr;
        fillEventVars(baseVars, branchConfig, sampleMeta, &pileupWeights, theoryBufsPtr);
        if (sampleMeta.isMC) {
            result.sumWeightPu += baseVars.values[varLayout.weightPu];
            result.sumWeightPuUp += baseVars.values[varLayout.weightPuUp];
            result.sumWeightPuDown += baseVars.values[varLayout.weightPuDown];
            const long double genWeight = baseVars.values[varLayout.genWeight];
            result.sumGenWeight += genWeight;
            result.sumGenWeightPu += genWeight * baseVars.values[varLayout.weightPu];
            result.sumGenWeightPuUp += genWeight * baseVars.values[varLayout.weightPuUp];
            result.sumGenWeightPuDown += genWeight * baseVars.values[varLayout.weightPuDown];
        } else if (!lumisFromTree) {
            result.lumis.emplace(static_cast<UInt_t>(requireEventVar(baseVars, varLayout.run, "run")),
                                 static_cast<UInt_t>(requireEventVar(baseVars, varLayout.luminosityBlock, "luminosityBlock")));
        }

        EvalContext preContext;
        preContext.vars = &baseVars;
        if (!evaluateCondition(selectionConfig.eventPreselection, preContext)) {
            continue;
        }

        if (tree->GetEntry(entry) <= 0) {
            throw runtime_error("Failed to read entry " + to_string(entry) + " of tree " +
                                appConfig.treeName + " in " + inputFileName);
        }
        fillEventVars(baseVars, branchConfig, sampleMeta, &pileupWeights, theoryBufsPtr);

        for (size_t c = 0; c < branchConfig.collections.size(); ++c) {
            const auto& inputConfig = branchConfig.collections[c];
            if (baseVars.has(inputConfig.sizeSlot) && baseVars.values[inputConfig.sizeSlot] > inputConfig.maxSize) {
                ++truncatedEvents[c];
            }
            collections.inputs[c] = buildInputCollection(inputConfig, baseVars);
        }

        collections.built.assign(selectionConfig.collections.size(), 0);
        collections.active.assign(selectionConfig.collections.size(), 0);
        if (jetCorrector.enabled()) {
            jetCorrector.prepareEvent(correctionInputs, baseVars, sampleMeta.isMC, correctionState);
        }
        for (const int slot : selectionConfig.buildOrder) {
            if (!jetDependent[slot]) {
                buildRuntimeCollection(slot, selectionConfig, collections, baseVars);
            }
        }

        for (size_t configurationIndex = 0; configurationIndex < configurations.size(); ++configurationIndex) {
            const EventVars* vars = &baseVars;
            if (jetCorrector.enabled()) {
                configurationVars = baseVars;
                jetCorrector.applyConfiguration(correctionInputs, correctionState,
                                                configurations[configurationIndex], sampleMeta.isMC,
                                                configurationVars);
                collections.inputs[ak4Input] = buildInputCollection(*correctionInputs.ak4, configurationVars);
                collections.inputs[ak8Input] = buildInputCollection(*correctionInputs.ak8, configurationVars);
                for (size_t slot = 0; slot < jetDependent.size(); ++slot) {
                    if (jetDependent[slot]) {
                        collections.built[slot] = 0;
                        collections.active[slot] = 0;
                    }
                }
                vars = &configurationVars;
            }
            for (const int slot : selectionConfig.buildOrder) {
                buildRuntimeCollection(slot, selectionConfig, collections, *vars);
            }

            for (const size_t t : treesByConfiguration[configurationIndex]) {
                OutputTreeState& treeState = outputTrees[t];
                if (treeCuts[t] == nullptr) {
                    throw runtime_error("Missing tree selection: " + treeState.config.selection);
                }

                EvalContext treeContext;
                treeContext.vars = vars;
                treeContext.collections = &collections;
                if (!evaluateCondition(*treeCuts[t], treeContext)) {
                    continue;
                }

                if (treeState.hasTheoryBranches) {
                    copyTheoryWeights(theoryInBuf, treeState.theoryOutBuf);
                }
                fillOutputTree(treeState, collections, *vars, treeVars, branchConfig.scalars, sampleMeta.isMC);
            }
        }
    }
    for (size_t c = 0; c < truncatedEvents.size(); ++c) {
        if (truncatedEvents[c] == 0) {
            continue;
        }
#pragma omp critical(convert_progress)
        cerr << "\nWarning: " << truncatedEvents[c] << " preselected events in " << inputFileName
             << " have more than max_size = " << branchConfig.collections[c].maxSize << " "
             << branchConfig.collections[c].name << " objects; only the first "
             << branchConfig.collections[c].maxSize << " are used" << endl;
    }
    return result;
}

// Runs one batch: every input file is converted by the OpenMP threads into per-thread temp
// files, which are then fast-merged into batchOutputPath. batchMeta accumulates raw_entries,
// the MC pileup-weight sums and (MC) skipped files; batchLumis the processed data lumis. A data
// file that cannot be opened/read fails the batch, because skipping it would lose luminosity.
void processInputBatchToTempFile(const vector<string>& batchInputFiles,
                                 size_t batchIndex,
                                 int threadCount,
                                 const fs::path& batchOutputPath,
                                 const AppConfig& appConfig,
                                 const SelectionConfig& selectionConfig,
                                 const SampleMeta& sampleMeta,
                                 const vector<PileupBin>& pileupWeights,
                                 const LumiMask* lumiMask,
                                 const JetPtCorrector& jetCorrector,
                                 const BranchConfig& branchConfig,
                                 atomic<size_t>& processedFiles,
                                 size_t totalFiles,
                                 BatchMeta& batchMeta,
                                 set<pair<UInt_t, UInt_t>>& batchLumis) {
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
                // Variation trees keep only their listed branches.
                if (treeState.config.variation.empty()) {
                    setupTheoryOutputBranches(treeState);
                }
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
                const FileProcessResult fileResult = processInputFile(batchInputFiles[index],
                                                                      appConfig,
                                                                      selectionConfig,
                                                                      sampleMeta,
                                                                      pileupWeights,
                                                                      lumiMask,
                                                                      jetCorrector,
                                                                      threadConfigs[tid],
                                                                      threadResults[tid].outputTrees);
#pragma omp critical(convert_accumulate)
                {
                    batchMeta.rawEntries += fileResult.rawEntries;
                    batchMeta.sumWeightPu += fileResult.sumWeightPu;
                    batchMeta.sumWeightPuUp += fileResult.sumWeightPuUp;
                    batchMeta.sumWeightPuDown += fileResult.sumWeightPuDown;
                    batchMeta.sumGenWeight += fileResult.sumGenWeight;
                    batchMeta.sumGenWeightPu += fileResult.sumGenWeightPu;
                    batchMeta.sumGenWeightPuUp += fileResult.sumGenWeightPuUp;
                    batchMeta.sumGenWeightPuDown += fileResult.sumGenWeightPuDown;
                    batchLumis.insert(fileResult.lumis.begin(), fileResult.lumis.end());
                }
                const size_t done = processedFiles.fetch_add(1) + 1;
#pragma omp critical(convert_progress)
                printFileProgress(sampleMeta.sample, done, totalFiles);
            } catch (const SkippableFileError& ex) {
                if (!sampleMeta.isMC) {
                    failed.store(true);
#pragma omp critical(convert_error)
                    errors.push_back("Input ROOT file " + batchInputFiles[index] +
                                     " cannot be processed and data files are never skipped: " + ex.what());
                } else {
                    const size_t done = processedFiles.fetch_add(1) + 1;
#pragma omp critical(convert_accumulate)
                    batchMeta.skippedFiles.push_back(batchInputFiles[index]);
#pragma omp critical(convert_progress)
                    {
                        cerr << "\nWarning: skipping " << batchInputFiles[index]
                             << ": " << ex.what() << '\n';
                        printFileProgress(sampleMeta.sample, done, totalFiles);
                    }
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
        fastMergeRootFiles(threadTempPaths, batchOutputPath, branchConfig.trees);
        for (auto& result : threadResults) {
            cleanupThreadResult(result);
        }
    } catch (...) {
        for (auto& result : threadResults) {
            cleanupThreadResult(result);
        }
        throw;
    }
}

vector<string> sliceBatchFiles(const vector<string>& inputFiles, size_t batchSize, size_t batchIndex) {
    const size_t begin = min(inputFiles.size(), batchIndex * batchSize);
    const size_t end = min(inputFiles.size(), begin + batchSize);
    return vector<string>(inputFiles.begin() + static_cast<vector<string>::difference_type>(begin),
                          inputFiles.begin() + static_cast<vector<string>::difference_type>(end));
}

BatchTempCollection collectSuccessfulBatchTempFiles(const AppConfig& appConfig,
                                                    const SampleMeta& sampleMeta,
                                                    const BranchConfig& branchConfig,
                                                    const vector<string>& inputFiles,
                                                    size_t batchSize,
                                                    const string& configHash,
                                                    const vector<size_t>& batchIndices,
                                                    size_t nBatches) {
    BatchTempCollection collection;
    collection.paths.reserve(batchIndices.size());
    for (const size_t batchIndex : batchIndices) {
        const fs::path batchOutputPath = makeBatchTempOutputPath(appConfig, sampleMeta, batchIndex);
        const string expectedFilesHash = hashFileList(sliceBatchFiles(inputFiles, batchSize, batchIndex));
        BatchMeta meta;
        string invalidReason;
        if (!validateBatchTempOutput(batchOutputPath,
                                     branchConfig.trees,
                                     expectedFilesHash,
                                     configHash,
                                     sampleMeta.isMC,
                                     meta,
                                     invalidReason)) {
            // A missing, incomplete or stale batch must fail the merge outright rather than
            // silently producing an under-counted (or mixed-configuration) output.
            throw runtime_error("Missing or incomplete batch " + to_string(batchIndex + 1) +
                                "/" + to_string(nBatches) + " for sample = " + sampleMeta.sample +
                                ": " + invalidReason);
        }
        collection.rawEntries += meta.rawEntries;
        collection.sumWeightPu += meta.sumWeightPu;
        collection.sumWeightPuUp += meta.sumWeightPuUp;
        collection.sumWeightPuDown += meta.sumWeightPuDown;
        collection.sumGenWeight += meta.sumGenWeight;
        collection.sumGenWeightPu += meta.sumGenWeightPu;
        collection.sumGenWeightPuUp += meta.sumGenWeightPuUp;
        collection.sumGenWeightPuDown += meta.sumGenWeightPuDown;
        collection.skippedFiles.insert(collection.skippedFiles.end(),
                                       meta.skippedFiles.begin(), meta.skippedFiles.end());
        if (!sampleMeta.isMC) {
            readBatchLumis(batchOutputPath, collection.lumis);
        }
        collection.paths.push_back(batchOutputPath.string());
    }

    if (collection.paths.empty()) {
        throw runtime_error("No successful temporary batch outputs found for sample " +
                            sampleMeta.sample);
    }
    return collection;
}

// True for "<stem><ext>" or "<stem>_<digits><ext>", the names downstream globs as this sample.
bool isSampleOutputName(const string& name, const string& stem, const string& extension) {
    if (name == stem + extension) {
        return true;
    }
    const string prefix = stem + "_";
    if (name.size() <= prefix.size() + extension.size() || name.compare(0, prefix.size(), prefix) != 0 ||
        name.compare(name.size() - extension.size(), extension.size(), extension) != 0) {
        return false;
    }
    const string middle = name.substr(prefix.size(), name.size() - prefix.size() - extension.size());
    return !middle.empty() && all_of(middle.begin(), middle.end(),
                                     [](unsigned char c) { return isdigit(c) != 0; });
}

int finalizeSuccessfulBatches(const AppConfig& appConfig,
                              const SampleMeta& sampleMeta,
                              const BranchConfig& branchConfig,
                              const vector<string>& inputFiles,
                              size_t batchSize,
                              const string& configHash,
                              const vector<size_t>& batchIndices,
                              size_t nBatches) {
    BatchTempCollection batchFiles;
    try {
        batchFiles = collectSuccessfulBatchTempFiles(appConfig, sampleMeta, branchConfig, inputFiles,
                                                     batchSize, configHash, batchIndices, nBatches);
    } catch (const exception& ex) {
        cerr << "Batch collection error: " << ex.what() << endl;
        return 1;
    }
    if (!batchFiles.skippedFiles.empty()) {
        cerr << "Warning: " << batchFiles.skippedFiles.size() << " MC input file"
             << (batchFiles.skippedFiles.size() == 1 ? " was" : "s were")
             << " skipped (unreadable); raw_entries counts only processed files, so the"
             << " normalisation stays consistent:";
        for (const auto& file : batchFiles.skippedFiles) {
            cerr << "\n  " << file;
        }
        cerr << endl;
    }

    const fs::path outputPath(sampleMeta.outputFileName);
    try {
        if (!outputPath.parent_path().empty()) {
            fs::create_directories(outputPath.parent_path());
        }

        // Group the batch files in order into outputs of at most max_output_file_size_gb
        // (fast merging keeps the compressed size, so the input file sizes add up).
        const Long64_t maxOutputBytes = outputSizeLimitBytes(appConfig.maxOutputFileSizeGB);
        vector<vector<string>> groups;
        Long64_t groupBytes = 0;
        for (const auto& path : batchFiles.paths) {
            const Long64_t bytes = static_cast<Long64_t>(fs::file_size(path));
            if (groups.empty() || (maxOutputBytes > 0 && !groups.back().empty() &&
                                   groupBytes + bytes > maxOutputBytes)) {
                groups.emplace_back();
                groupBytes = 0;
            }
            groups.back().push_back(path);
            groupBytes += bytes;
        }
        vector<fs::path> outputs;
        for (size_t k = 0; k < groups.size(); ++k) {
            outputs.push_back(groups.size() == 1 ? outputPath : makeSplitOutputPath(outputPath, k));
        }

        // Outputs of an earlier merge that this one would not overwrite (e.g. more chunks
        // before) would be globbed downstream as extra events: refuse instead of mixing them.
        const string stem = outputPath.stem().string();
        const string extension = outputPath.has_extension() ? outputPath.extension().string() : ".root";
        set<string> planned;
        for (const auto& output : outputs) {
            planned.insert(output.filename().string());
        }
        vector<string> stale;
        const fs::path outputDir = outputPath.parent_path().empty() ? fs::path(".") : outputPath.parent_path();
        for (const auto& entry : fs::directory_iterator(outputDir)) {
            const string name = entry.path().filename().string();
            if (isSampleOutputName(name, stem, extension) && planned.count(name) == 0u) {
                stale.push_back(entry.path().string());
            }
        }
        if (!stale.empty()) {
            ostringstream os;
            os << stale.size() << " existing output file(s) of sample " << sampleMeta.sample
               << " would not be overwritten by this merge (" << outputs.size()
               << " outputs) and would be read downstream as extra events; remove them and rerun"
               << " the merge:";
            for (const auto& path : stale) {
                os << ' ' << path;
            }
            throw runtime_error(os.str());
        }

        cout << "Merging " << batchFiles.paths.size()
             << " successful temporary batch file" << (batchFiles.paths.size() == 1 ? "" : "s")
             << " out of " << nBatches << " into " << outputs.size() << " output file"
             << (outputs.size() == 1 ? "" : "s") << endl;
        for (size_t k = 0; k < groups.size(); ++k) {
            fastMergeRootFiles(groups[k], outputs[k], branchConfig.trees);
            cout << "Wrote output file: " << outputs[k].string() << endl;
        }
    } catch (const exception& ex) {
        cerr << "Output error: " << ex.what() << endl;
        return 1;
    }

    // sample.json and the processed-lumi list are written only once every output exists.
    try {
        if (appConfig.updateRawEntries) {
            // Checked before any sample.json write, so a failure leaves the entry untouched.
            if (sampleMeta.isMC && batchFiles.rawEntries > 0 && batchFiles.sumGenWeightPu <= 0.L) {
                throw runtime_error("non-positive sum of genWeight * weight_pu over the processed events of " +
                                    sampleMeta.sample);
            }
            writeSampleRawEntries(appConfig.sampleConfigPath, sampleMeta.sample, batchFiles.rawEntries);
            cout << "Updated raw_entries in " << appConfig.sampleConfigPath
                 << " for sample = " << sampleMeta.sample
                 << ", tree = " << appConfig.treeName
                 << ", raw_entries = " << batchFiles.rawEntries << endl;
            if (sampleMeta.isMC && batchFiles.rawEntries > 0) {
                // Mean weight_pu (and up/down) over all generated events of the processed
                // files: the denominator of the absolute pileup normalisation downstream.
                const long double n = static_cast<long double>(batchFiles.rawEntries);
                const vector<pair<string, long double>> means = {
                    {"weight_pu_mean", batchFiles.sumWeightPu / n},
                    {"weight_pu_up_mean", batchFiles.sumWeightPuUp / n},
                    {"weight_pu_down_mean", batchFiles.sumWeightPuDown / n},
                };
                for (const auto& item : means) {
                    ostringstream value;
                    value << setprecision(17) << static_cast<double>(item.second);
                    writeSampleNumericField(appConfig.sampleConfigPath, sampleMeta.sample, item.first, value.str());
                    cout << "Updated " << item.first << " = " << value.str() << endl;
                }
                // Signed generator-weight sums over the same processed generated events: the
                // denominators S of the MC event weights L * xsection * genWeight * weight_pu / S.
                const vector<pair<string, long double>> sums = {
                    {"sum_genweight", batchFiles.sumGenWeight},
                    {"sum_genweight_pu", batchFiles.sumGenWeightPu},
                    {"sum_genweight_pu_up", batchFiles.sumGenWeightPuUp},
                    {"sum_genweight_pu_down", batchFiles.sumGenWeightPuDown},
                };
                for (const auto& item : sums) {
                    ostringstream value;
                    value << setprecision(17) << static_cast<double>(item.second);
                    writeSampleNumericField(appConfig.sampleConfigPath, sampleMeta.sample, item.first, value.str());
                    cout << "Updated " << item.first << " = " << value.str() << endl;
                }
            }
        } else {
            // update_raw_entries: false -- the processed entry count is not the normalisation
            // of this configuration, so sample.json is left untouched.
            cout << "Processed raw_entries = " << batchFiles.rawEntries
                 << " for sample = " << sampleMeta.sample
                 << ", tree = " << appConfig.treeName
                 << " (not written to " << appConfig.sampleConfigPath
                 << " -- update_raw_entries is false)" << endl;
        }
    } catch (const exception& ex) {
        cerr << "raw_entries update error: " << ex.what() << endl;
        return 1;
    }

    if (!sampleMeta.isMC) {
        try {
            const fs::path lumiPath = (outputPath.parent_path().empty() ? fs::path(".") : outputPath.parent_path()) /
                                      (sampleMeta.sample + "_processed_lumis.json");
            writeProcessedLumiJson(lumiPath, batchFiles.lumis);
            set<UInt_t> runs;
            for (const auto& lumi : batchFiles.lumis) {
                runs.insert(lumi.first);
            }
            cout << "Wrote processed-lumi JSON " << lumiPath.string() << " ("
                 << batchFiles.lumis.size() << " lumisections in " << runs.size()
                 << " runs); pass it to brilcalc -i for the luminosity" << endl;
        } catch (const exception& ex) {
            cerr << "Processed-lumi error: " << ex.what() << endl;
            return 1;
        }
    }

    return 0;
}

size_t computeBatchSize(const AppConfig& appConfig, size_t inputFileCount) {
    const int threadCount = determineThreadCount(appConfig.maxThreads, inputFileCount);
    const char* fpbEnv = getenv("CONVERT_FILES_PER_BATCH");
    size_t fpbOverride = 0;
    if (fpbEnv != nullptr && *fpbEnv != '\0') {
        try { fpbOverride = static_cast<size_t>(stoull(fpbEnv)); } catch (...) {}
    }
    return fpbOverride > 0 ? fpbOverride : max<size_t>(1, static_cast<size_t>(threadCount) * 32);
}

// The true expected batch count, derived from the current input file discovery -- NOT from
// whichever batch temp files happen to already exist on disk. Input samples are now produced
// with an upstream event filter, so a batch whose job never ran (no temp file at all) must be
// detected as missing rather than silently shrinking the batch count the merge expects.
size_t computeNBatches(const AppConfig& appConfig, size_t inputFileCount) {
    if (inputFileCount == 0) {
        return 0;
    }
    const size_t batchSize = computeBatchSize(appConfig, inputFileCount);
    return (inputFileCount + batchSize - 1) / batchSize;
}

}  // namespace

struct GenWeightSums {
    long double sumw = 0.L;
    long double count = 0.L;
    size_t filesUsed = 0;
    vector<string> skippedFiles;
};

// Sum genEventSumw / genEventCount over the Runs trees of the input files. Files are
// opened serially (concurrent remote TFile::Open is not thread-safe). A file that cannot
// be read after the usual retries is skipped and reported: the result is a mean, so a
// few missing files do not bias it.
GenWeightSums sumRunsGenWeights(const vector<string>& inputFiles) {
    GenWeightSums sums;
    for (size_t i = 0; i < inputFiles.size(); ++i) {
        const string& inputFileName = inputFiles[i];
        try {
            unique_ptr<TFile> inputFile = openInputFileWithRetry(inputFileName);
            TTree* runs = dynamic_cast<TTree*>(inputFile->Get("Runs"));
            if (runs == nullptr || runs->GetBranch("genEventSumw") == nullptr ||
                runs->GetBranch("genEventCount") == nullptr) {
                throw runtime_error("missing Runs tree or genEventSumw/genEventCount branch");
            }
            Double_t fileSumw = 0.;
            Long64_t fileCount = 0;
            runs->SetBranchStatus("*", 0);
            runs->SetBranchStatus("genEventSumw", 1);
            runs->SetBranchStatus("genEventCount", 1);
            runs->SetBranchAddress("genEventSumw", &fileSumw);
            runs->SetBranchAddress("genEventCount", &fileCount);
            long double sumw = 0.L;
            long double count = 0.L;
            for (Long64_t entry = 0; entry < runs->GetEntries(); ++entry) {
                if (runs->GetEntry(entry) <= 0) {
                    throw runtime_error("failed to read Runs entry " + to_string(entry));
                }
                sumw += fileSumw;
                count += fileCount;
            }
            runs->ResetBranchAddresses();
            sums.sumw += sumw;
            sums.count += count;
            ++sums.filesUsed;
        } catch (const exception& ex) {
            cerr << "Warning: skipping " << inputFileName << " for genweight_mean: " << ex.what() << endl;
            sums.skippedFiles.push_back(inputFileName);
        }
        if ((i + 1) % 100 == 0 || i + 1 == inputFiles.size()) {
            cout << "Runs trees read: " << (i + 1) << "/" << inputFiles.size() << endl;
        }
    }
    return sums;
}

// --update-genweight-mean: store genweight_mean = sum(genEventSumw) / sum(genEventCount)
// (the mean signed generator weight of the generated sample) in sample.json. Downstream
// event weights use genWeight / genweight_mean, so negative-weight events enter with a
// negative sign while the raw_entries-based normalisation keeps its meaning.
int updateGenWeightMean(const AppConfig& appConfig, SampleMeta& sampleMeta) {
    if (!sampleMeta.isMC) {
        cerr << "genweight_mean error: sample " << sampleMeta.sample << " is data" << endl;
        return 1;
    }
    vector<string> inputFiles;
    try {
        inputFiles = discoverInputFiles(sampleMeta);
    } catch (const exception& ex) {
        cerr << "Input discovery error: " << ex.what() << endl;
        return 1;
    }
    cout << "Running convert_branch for sample = " << sampleMeta.sample
         << ", update genweight_mean from " << inputFiles.size() << " Runs trees" << endl;

    const GenWeightSums sums = sumRunsGenWeights(inputFiles);
    if (sums.filesUsed == 0 || sums.count <= 0.L || sums.sumw <= 0.L) {
        cerr << "genweight_mean error: no usable Runs tree content for sample " << sampleMeta.sample
             << " (files used = " << sums.filesUsed << ", sumw = " << static_cast<double>(sums.sumw)
             << ", count = " << static_cast<double>(sums.count) << ")" << endl;
        return 1;
    }
    if (!sums.skippedFiles.empty()) {
        cerr << "Warning: genweight_mean for sample " << sampleMeta.sample << " uses " << sums.filesUsed
             << "/" << inputFiles.size() << " files (" << sums.skippedFiles.size() << " skipped)" << endl;
    }

    const double mean = static_cast<double>(sums.sumw / sums.count);
    ostringstream valueText;
    valueText << setprecision(17) << mean;
    try {
        writeSampleNumericField(appConfig.sampleConfigPath, sampleMeta.sample, "genweight_mean", valueText.str());
    } catch (const exception& ex) {
        cerr << "genweight_mean update error: " << ex.what() << endl;
        return 1;
    }
    cout << "Updated genweight_mean in " << appConfig.sampleConfigPath << " for sample = " << sampleMeta.sample
         << ": genweight_mean = " << valueText.str()
         << ", sum(genEventSumw) = " << setprecision(17) << static_cast<double>(sums.sumw)
         << ", sum(genEventCount) = " << static_cast<long long>(sums.count)
         << " (compare with raw_entries)" << endl;
    return 0;
}

int main(int argc, char** argv) {
    TH1::AddDirectory(false);

    AppConfig appConfig;
    BranchConfig branchConfig;
    SelectionConfig selectionConfig;
    try {
        appConfig = loadAppConfig();
        branchConfig = loadBranchConfig(appConfig);
        selectionConfig = loadSelectionConfig(appConfig);
        resolveEngineSymbols(selectionConfig, branchConfig);
        requirePreselectionWithoutCorrectedInputs(selectionConfig, appConfig.jetPtCorrection);
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

    if (batchRequest.updateGenWeightMean) {
        return updateGenWeightMean(appConfig, sampleMeta);
    }

    // The variation trees are part of the output layout of an MC sample, for the batch outputs
    // and for the final merge alike. They are copies of the nominal trees, whose expressions and
    // variable slots resolveEngineSymbols has already resolved.
    try {
        addVariationTrees(branchConfig, appConfig.jetPtCorrection, sampleMeta.isMC);
    } catch (const exception& ex) {
        cerr << "Configuration error: " << ex.what() << endl;
        return 1;
    }

    // Pileup weights in use (MC) and the conversion-configuration hash recorded in, and
    // checked against, every batch .meta.
    string puWeightPath;
    if (sampleMeta.isMC && !appConfig.puWeightPathPattern.empty()) {
        try {
            puWeightPath = resolvePileupWeightPath(appConfig, sampleMeta);
        } catch (const exception& ex) {
            cerr << "Pileup weight error: " << ex.what() << endl;
            return 1;
        }
    }

    if (batchRequest.mergeSuccessfulBatches) {
        const fs::path batchTempDir = makeBatchTempOutputDir(appConfig, sampleMeta);
        // The expected batch count and each batch's input slice come from the same file-list
        // snapshot the batch jobs used, rather than from whichever batch temp files happen to
        // exist -- a batch whose job never ran (no temp file, no sidecar) fails the merge.
        vector<string> inputFiles;
        string configHash;
        try {
            inputFiles = resolveInputFiles(appConfig, sampleMeta);
            configHash = computeConversionConfigHash(appConfig, sampleMeta, puWeightPath);
        } catch (const exception& ex) {
            cerr << "Input discovery error: " << ex.what() << endl;
            return 1;
        }
        const size_t batchSize = computeBatchSize(appConfig, inputFiles.size());
        const size_t nBatches = computeNBatches(appConfig, inputFiles.size());
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
        return finalizeSuccessfulBatches(appConfig, sampleMeta, branchConfig, inputFiles, batchSize,
                                         configHash, batchIndices, nBatches);
    }

    vector<string> inputFiles;
    try {
        inputFiles = resolveInputFiles(appConfig, sampleMeta);
    } catch (const exception& ex) {
        cerr << "Input discovery error: " << ex.what() << endl;
        return 1;
    }

    const int threadCount = determineThreadCount(appConfig.maxThreads, inputFiles.size());
    const size_t batchSize = computeBatchSize(appConfig, inputFiles.size());
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

    string configHash;
    try {
        configHash = computeConversionConfigHash(appConfig, sampleMeta, puWeightPath);
    } catch (const exception& ex) {
        cerr << "Configuration error: " << ex.what() << endl;
        return 1;
    }

    vector<PileupBin> pileupWeights;
    if (!puWeightPath.empty()) {
        try {
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

    JetPtCorrector jetCorrector;
    try {
        jetCorrector.initialize(appConfig.jetPtCorrection);
        if (jetCorrector.enabled()) {
            jetCorrector.resolveInputs(branchConfig);
            cout << "Jet corrections: " << jetCorrector.describe(sampleMeta.isMC) << endl;
            const unordered_set<string> jetDependent = jetDependentCollections(selectionConfig);
            cout << "Runtime collections built once per event (independent of the jet corrections):";
            for (const auto& name : selectionConfig.collectionOrder) {
                if (jetDependent.count(name) == 0) {
                    cout << ' ' << name;
                }
            }
            cout << endl;
        }
    } catch (const exception& ex) {
        cerr << "Jet pt correction error: " << ex.what() << endl;
        return 1;
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
        const string filesHash = hashFileList(batchInputFiles);
        bool batchAlreadyComplete = false;
        if (appConfig.resumeSuccessfulBatches && batchRequest.singleBatch) {
            BatchMeta existingMeta;
            string invalidReason;
            if (validateBatchTempOutput(batchOutputPath,
                                        branchConfig.trees,
                                        filesHash,
                                        configHash,
                                        sampleMeta.isMC,
                                        existingMeta,
                                        invalidReason)) {
                cout << "Skipping completed batch " << (batchIndex + 1) << "/" << nBatches
                     << ": found valid existing temporary batch file "
                     << batchOutputPath.string()
                     << " with raw_entries = " << existingMeta.rawEntries << endl;
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
        // Invalidate any completion record of an earlier attempt before rewriting the batch.
        {
            std::error_code ignored;
            fs::remove(makeBatchMetaPath(batchOutputPath), ignored);
        }
        BatchMeta batchMeta;
        batchMeta.nFiles = batchInputFiles.size();
        batchMeta.filesHash = filesHash;
        batchMeta.configHash = configHash;
        set<pair<UInt_t, UInt_t>> batchLumis;

        cout << "Processing batch " << (batchIndex + 1) << "/" << nBatches
             << ": files " << (begin + 1) << "-" << end
             << " -> " << batchOutputPath.string()
             << " using " << batchThreadCount << " thread"
             << (batchThreadCount == 1 ? "" : "s") << endl;

        try {
            processInputBatchToTempFile(batchInputFiles,
                                        batchIndex,
                                        batchThreadCount,
                                        batchOutputPath,
                                        appConfig,
                                        selectionConfig,
                                        sampleMeta,
                                        pileupWeights,
                                        lumiMask.get(),
                                        jetCorrector,
                                        branchConfig,
                                        processedFiles,
                                        inputFiles.size(),
                                        batchMeta,
                                        batchLumis);
            writeBatchRawEntries(batchOutputPath, batchMeta.rawEntries);
            if (!sampleMeta.isMC) {
                writeBatchLumis(batchOutputPath, batchLumis);
            }
            writeBatchMeta(batchOutputPath, batchMeta);  // completion marker, written last
            cout << "Wrote temporary batch file: " << batchOutputPath.string() << endl;
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
    return finalizeSuccessfulBatches(appConfig, sampleMeta, branchConfig, inputFiles, batchSize,
                                     configHash, batchIndices, nBatches);
}
