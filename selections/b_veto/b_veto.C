// Summary: Build AK4 b-tag histograms for working-point studies.
#include <TChain.h>
#include <TError.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TSystem.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// Configuration
static const size_t MAX_FILES_PER_SAMPLE = 1000;

static const int    NBINS = 50000;
static const double XMIN  = 0.0;
static const double XMAX  = 1.0;

static const std::string OUTFILE = "b_veto_hists.root";
static const std::string XROOTD_PREFIX = "root://cms-xrd-global.cern.ch/";

static const int MAXFAT = 16;
static const int MAXAK4 = 64;
static const int TOP3_NBINS = 110;
static const double TOP3_MIN = -0.1;
static const double TOP3_MAX = 1.0;
static const double TOP3_MISSING = -0.1;

static const Long64_t TREE_CACHE_SIZE = 100LL * 1024LL * 1024LL; // 100 MB

struct SampleInfo {
  std::string label;
  std::string dataset;
  bool isSignal = false;
};

static const std::vector<SampleInfo> SAMPLES = {
  {
    "TTbar",
    "/TTto4Q_TuneCP5_13p6TeV_powheg-pythia8/jschulte-ScoutingNanoUPart_v2-00000000000000000000000000000000/USER",
    false
  },
  {
    "WWW",
    "/WWW-4F_TuneCP5_13p6TeV_amcatnlo-pythia8/jschulte-ScoutingNanoUPart_v2-00000000000000000000000000000000/USER",
    true
  }
};

static const char* TREE_NAME = "Events";

static const char* BR_nFatJet      = "nScoutingFatPFJetRecluster";
static const char* BR_FatJet_pt    = "ScoutingFatPFJetRecluster_pt";
static const char* BR_FatJet_eta   = "ScoutingFatPFJetRecluster_eta";
static const char* BR_FatJet_phi   = "ScoutingFatPFJetRecluster_phi";

static const char* BR_nJet                  = "nScoutingPFJetRecluster2";
static const char* BR_Jet_pt                = "ScoutingPFJetRecluster2_pt";
static const char* BR_Jet_eta               = "ScoutingPFJetRecluster2_eta";
static const char* BR_Jet_phi               = "ScoutingPFJetRecluster2_phi";
static const char* BR_Jet_btag              = "ScoutingPFJetRecluster2_scoutUParT_probb";
static const char* BR_Jet_nConstituents     = "ScoutingPFJetRecluster2_nConstituents";
static const char* BR_Jet_neHEF             = "ScoutingPFJetRecluster2_neHEF";
static const char* BR_Jet_neEmEF            = "ScoutingPFJetRecluster2_neEmEF";
static const char* BR_Jet_chHadMultiplicity = "ScoutingPFJetRecluster2_chHadMultiplicity";
static const char* BR_Jet_muEF              = "ScoutingPFJetRecluster2_muEF";
static const char* BR_Jet_neHadMultiplicity = "ScoutingPFJetRecluster2_neHadMultiplicity";
static const char* BR_Jet_hadronFlavour     = "ScoutingPFJetRecluster2_hadronFlavour";

// Helpers

static inline std::string trim(std::string s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

static inline double deltaPhi(double a, double b) {
  double d = a - b;
  while (d >  M_PI) d -= 2.0 * M_PI;
  while (d <= -M_PI) d += 2.0 * M_PI;
  return d;
}

static inline double deltaR(double eta1, double phi1, double eta2, double phi2) {
  const double dphi = deltaPhi(phi1, phi2);
  const double deta = eta1 - eta2;
  return std::sqrt(deta * deta + dphi * dphi);
}

static inline std::string to_xrootd_path(std::string path) {
  path = trim(path);
  if (path.empty()) return path;
  if (path.rfind("root://", 0) == 0) return path;
  if (path.rfind("/store/", 0) == 0) return XROOTD_PREFIX + path;
  if (path.rfind("store/", 0) == 0) return XROOTD_PREFIX + "/" + path;
  return path;
}

static inline bool is_user_dataset(const std::string& dataset) {
  return dataset.size() >= 5 && dataset.rfind("/USER") == dataset.size() - 5;
}

static std::vector<std::string> collect_dataset_files(const std::string& dataset, size_t maxfiles) {
  std::vector<std::string> files;
  files.reserve(maxfiles);

  std::unordered_set<std::string> seen;
  seen.reserve(maxfiles * 2 + 1);

  std::string query = "file dataset=" + dataset;
  if (is_user_dataset(dataset)) {
    query += " instance=prod/phys03";
  }

  std::ostringstream cmdss;
  cmdss << "dasgoclient -query=\"" << query << "\" -limit=" << maxfiles;
  const std::string cmd = cmdss.str();

  std::cout << "  [INFO] DAS query: " << query << std::endl;

  const std::string output = gSystem->GetFromPipe(cmd.c_str()).Data();

  std::istringstream iss(output);
  std::string line;
  while (std::getline(iss, line)) {
    std::string path = to_xrootd_path(line);
    if (path.empty()) continue;
    if (!seen.insert(path).second) continue;
    files.push_back(std::move(path));
    if (files.size() >= maxfiles) break;
  }

  return files;
}

static inline bool pass_ak4_quality(float pt,
                                    float eta,
                                    int nConstituents,
                                    float neHEF,
                                    float neEmEF,
                                    int chHadMultiplicity,
                                    float muEF,
                                    int neHadMultiplicity) {
  const double abseta = std::abs(eta);
  if (pt <= 30.0 || abseta >= 2.5) return false;

  if (abseta < 2.6) {
    return nConstituents > 1 &&
           neHEF < 0.99 &&
           neEmEF < 0.90 &&
           chHadMultiplicity > 0 &&
           muEF < 0.80;
  }

  if (abseta < 2.7) {
    return neEmEF < 0.99 &&
           muEF < 0.80 &&
           neHadMultiplicity > 1;
  }

  if (abseta < 3.0) {
    return neEmEF < 0.99 &&
           neHadMultiplicity > 1;
  }

  return neEmEF < 0.20;
}

// Per-sample processing

struct SampleResult {
  SampleInfo sample;
  Long64_t rawEntries = 0;
  TH1D* hCat2 = nullptr;
  TH1D* hCat3 = nullptr;
  TH2D* hJetScoreVsFlavor = nullptr;
  TH3D* hTop3Cat2 = nullptr;
  TH3D* hTop3Cat3 = nullptr;
};

static bool process_one_sample(const SampleInfo& sample, SampleResult& out) {
  out.sample = sample;

  std::cout << "  [STEP] collect files..." << std::endl;
  size_t MAX_FILES_PER_SAMPLE_tmp = MAX_FILES_PER_SAMPLE;
  // Use 10% of the TTbar files.
  if (sample.label == "TTbar") {
    MAX_FILES_PER_SAMPLE_tmp = std::max<size_t>(1, MAX_FILES_PER_SAMPLE_tmp / 10);
    std::cout << "         (using " << MAX_FILES_PER_SAMPLE_tmp << " files for TTbar sample)" << std::endl;
  }
  std::vector<std::string> files = collect_dataset_files(sample.dataset, MAX_FILES_PER_SAMPLE_tmp);
  std::cout << "  [STEP] collected " << files.size() << " files" << std::endl;

  if (files.empty()) {
    std::cout << "  [ERROR] No files found from DAS for dataset: " << sample.dataset << std::endl;
    std::cout << "          Please confirm `dasgoclient` is available, the dataset path is valid," << std::endl;
    std::cout << "          and for /USER datasets the correct DBS instance is being used." << std::endl;
    return false;
  }

  std::cout << "  [STEP] first file = " << files.front() << std::endl;
  std::cout << "  [STEP] detected tree name = " << TREE_NAME << std::endl;

  std::cout << "  [STEP] build TChain..." << std::endl;
  TChain chain(TREE_NAME);
  chain.SetCacheSize(TREE_CACHE_SIZE);

  for (const auto& f : files) {
    chain.Add(f.c_str());
  }

  std::cout << "  [STEP] calling GetEntries()..." << std::endl;
  out.rawEntries = chain.GetEntries();
  std::cout << "  [STEP] GetEntries() done, rawEntries = " << out.rawEntries << std::endl;

  if (out.rawEntries <= 0) {
    std::cout << "  [WARN] Zero entries for sample " << sample.label << std::endl;
    chain.Reset();
    files.clear();
    files.shrink_to_fit();
    return false;
  }

  chain.SetBranchStatus("*", 0);
  chain.SetBranchStatus(BR_nFatJet,    1);
  chain.SetBranchStatus(BR_FatJet_pt,  1);
  chain.SetBranchStatus(BR_FatJet_eta, 1);
  chain.SetBranchStatus(BR_FatJet_phi, 1);

  chain.SetBranchStatus(BR_nJet,                  1);
  chain.SetBranchStatus(BR_Jet_pt,                1);
  chain.SetBranchStatus(BR_Jet_eta,               1);
  chain.SetBranchStatus(BR_Jet_phi,               1);
  chain.SetBranchStatus(BR_Jet_btag,              1);
  chain.SetBranchStatus(BR_Jet_nConstituents,     1);
  chain.SetBranchStatus(BR_Jet_neHEF,             1);
  chain.SetBranchStatus(BR_Jet_neEmEF,            1);
  chain.SetBranchStatus(BR_Jet_chHadMultiplicity, 1);
  chain.SetBranchStatus(BR_Jet_muEF,              1);
  chain.SetBranchStatus(BR_Jet_neHadMultiplicity, 1);
  chain.SetBranchStatus(BR_Jet_hadronFlavour,     1);

  chain.AddBranchToCache(BR_nFatJet,    true);
  chain.AddBranchToCache(BR_FatJet_pt,  true);
  chain.AddBranchToCache(BR_FatJet_eta, true);
  chain.AddBranchToCache(BR_FatJet_phi, true);

  chain.AddBranchToCache(BR_nJet,                  true);
  chain.AddBranchToCache(BR_Jet_pt,                true);
  chain.AddBranchToCache(BR_Jet_eta,               true);
  chain.AddBranchToCache(BR_Jet_phi,               true);
  chain.AddBranchToCache(BR_Jet_btag,              true);
  chain.AddBranchToCache(BR_Jet_nConstituents,     true);
  chain.AddBranchToCache(BR_Jet_neHEF,             true);
  chain.AddBranchToCache(BR_Jet_neEmEF,            true);
  chain.AddBranchToCache(BR_Jet_chHadMultiplicity, true);
  chain.AddBranchToCache(BR_Jet_muEF,              true);
  chain.AddBranchToCache(BR_Jet_neHadMultiplicity, true);
  chain.AddBranchToCache(BR_Jet_hadronFlavour,     true);

  Int_t   nFatJet = 0;
  Float_t FatJet_pt[MAXFAT]  = {0.0f};
  Float_t FatJet_eta[MAXFAT] = {0.0f};
  Float_t FatJet_phi[MAXFAT] = {0.0f};

  Int_t   nJet = 0;
  Float_t Jet_pt[MAXAK4]                = {0.0f};
  Float_t Jet_eta[MAXAK4]               = {0.0f};
  Float_t Jet_phi[MAXAK4]               = {0.0f};
  Float_t Jet_btag[MAXAK4]              = {0.0f};
  Int_t   Jet_nConstituents[MAXAK4]     = {0};
  Float_t Jet_neHEF[MAXAK4]             = {0.0f};
  Float_t Jet_neEmEF[MAXAK4]            = {0.0f};
  Short_t Jet_chHadMultiplicity[MAXAK4] = {0};
  Float_t Jet_muEF[MAXAK4]              = {0.0f};
  Int_t   Jet_neHadMultiplicity[MAXAK4] = {0};
  Int_t   Jet_hadronFlavour[MAXAK4]     = {0};

  chain.SetBranchAddress(BR_nFatJet,    &nFatJet);
  chain.SetBranchAddress(BR_FatJet_pt,  FatJet_pt);
  chain.SetBranchAddress(BR_FatJet_eta, FatJet_eta);
  chain.SetBranchAddress(BR_FatJet_phi, FatJet_phi);

  chain.SetBranchAddress(BR_nJet,                  &nJet);
  chain.SetBranchAddress(BR_Jet_pt,                Jet_pt);
  chain.SetBranchAddress(BR_Jet_eta,               Jet_eta);
  chain.SetBranchAddress(BR_Jet_phi,               Jet_phi);
  chain.SetBranchAddress(BR_Jet_btag,              Jet_btag);
  chain.SetBranchAddress(BR_Jet_nConstituents,     Jet_nConstituents);
  chain.SetBranchAddress(BR_Jet_neHEF,             Jet_neHEF);
  chain.SetBranchAddress(BR_Jet_neEmEF,            Jet_neEmEF);
  chain.SetBranchAddress(BR_Jet_chHadMultiplicity, Jet_chHadMultiplicity);
  chain.SetBranchAddress(BR_Jet_muEF,              Jet_muEF);
  chain.SetBranchAddress(BR_Jet_neHadMultiplicity, Jet_neHadMultiplicity);
  chain.SetBranchAddress(BR_Jet_hadronFlavour,     Jet_hadronFlavour);

  const std::string h2name = "h_minBtag_cat2_" + sample.label;
  const std::string h3name = "h_minBtag_cat3_" + sample.label;
  const std::string h2dname = "h2_probb_vs_hadronFlavour_" + sample.label;
  const std::string h3top2name = "h3_top3_probb_cat2_" + sample.label;
  const std::string h3top3name = "h3_top3_probb_cat3_" + sample.label;
  out.hCat2 = new TH1D(h2name.c_str(), (h2name + ";min Jet_btagUParTAK4B;Events").c_str(), NBINS, XMIN, XMAX);
  out.hCat3 = new TH1D(h3name.c_str(), (h3name + ";min Jet_btagUParTAK4B;Events").c_str(), NBINS, XMIN, XMAX);
  out.hJetScoreVsFlavor = new TH2D(
    h2dname.c_str(),
    (h2dname + ";ScoutingPFJetRecluster2_scoutUParT_probb;ScoutingPFJetRecluster2_hadronFlavour").c_str(),
    NBINS,
    XMIN,
    XMAX,
    26,
    -0.5,
    25.5
  );
  out.hTop3Cat2 = new TH3D(
    h3top2name.c_str(),
    (h3top2name + ";leading AK4 probb;subleading AK4 probb;third AK4 probb").c_str(),
    TOP3_NBINS,
    TOP3_MIN,
    TOP3_MAX,
    TOP3_NBINS,
    TOP3_MIN,
    TOP3_MAX,
    TOP3_NBINS,
    TOP3_MIN,
    TOP3_MAX
  );
  out.hTop3Cat3 = new TH3D(
    h3top3name.c_str(),
    (h3top3name + ";leading AK4 probb;subleading AK4 probb;third AK4 probb").c_str(),
    TOP3_NBINS,
    TOP3_MIN,
    TOP3_MAX,
    TOP3_NBINS,
    TOP3_MIN,
    TOP3_MAX,
    TOP3_NBINS,
    TOP3_MIN,
    TOP3_MAX
  );
  out.hCat2->SetDirectory(nullptr);
  out.hCat3->SetDirectory(nullptr);
  out.hJetScoreVsFlavor->SetDirectory(nullptr);
  out.hTop3Cat2->SetDirectory(nullptr);
  out.hTop3Cat3->SetDirectory(nullptr);
  out.hCat2->Sumw2();
  out.hCat3->Sumw2();
  out.hJetScoreVsFlavor->Sumw2();
  out.hTop3Cat2->Sumw2();
  out.hTop3Cat3->Sumw2();

  const Long64_t N = out.rawEntries;
  const int reportEvery = (N < 5e5) ? 50000 : 200000;

  Long64_t nPass2 = 0;
  Long64_t nPass3 = 0;

  int selectedAK4[MAXAK4];

  for (Long64_t i = 0; i < N; ++i) {
    chain.GetEntry(i);

    if (reportEvery > 0 && (i % reportEvery == 0)) {
      std::cout << "    processing " << sample.label << " : " << i << " / " << N << "\r" << std::flush;
    }

    const int nF = std::min(nFatJet, MAXFAT);
    const int nJ = std::min(nJet, MAXAK4);

    int nSelectedAK4 = 0;
    for (int j = 0; j < nJ; ++j) {
      if (!pass_ak4_quality(Jet_pt[j],
                            Jet_eta[j],
                            Jet_nConstituents[j],
                            Jet_neHEF[j],
                            Jet_neEmEF[j],
                            static_cast<int>(Jet_chHadMultiplicity[j]),
                            Jet_muEF[j],
                            Jet_neHadMultiplicity[j])) {
        continue;
      }
      selectedAK4[nSelectedAK4++] = j;
    }

    for (int ii = 0; ii < nSelectedAK4; ++ii) {
      const int idx = selectedAK4[ii];
      out.hJetScoreVsFlavor->Fill(Jet_btag[idx], Jet_hadronFlavour[idx]);
    }

    if (nF < 2) continue;

    bool pass_cat2 = false;
    bool pass_cat3 = false;
    int maxAK8 = 0;

    if (nF == 2) {
      if (!(FatJet_pt[0] > 180.0 && std::abs(FatJet_eta[0]) < 2.4 &&
            FatJet_pt[1] > 180.0 && std::abs(FatJet_eta[1]) < 2.4)) {
        continue;
      }
      pass_cat2 = true;
      maxAK8 = 2;
    } else {
      if (!(FatJet_pt[0] > 180.0 && std::abs(FatJet_eta[0]) < 2.4 &&
            FatJet_pt[1] > 180.0 && std::abs(FatJet_eta[1]) < 2.4 &&
            FatJet_pt[2] > 180.0 && std::abs(FatJet_eta[2]) < 2.4)) {
        continue;
      }
      pass_cat3 = true;
      maxAK8 = 3;
    }

    int nNonOverlap = 0;

    for (int ii = 0; ii < nSelectedAK4; ++ii) {
      const int idx = selectedAK4[ii];

      bool overlap = false;
      for (int k = 0; k < maxAK8; ++k) {
        if (deltaR(Jet_eta[idx], Jet_phi[idx], FatJet_eta[k], FatJet_phi[k]) < 0.8) {
          overlap = true;
          break;
        }
      }
      if (overlap) continue;

      ++nNonOverlap;
    }

    if (pass_cat2 && nNonOverlap < 2) continue;

    double topScores[3] = {TOP3_MISSING, TOP3_MISSING, TOP3_MISSING};
    if (nSelectedAK4 > 0) {
      std::vector<double> sortedScores;
      sortedScores.reserve(nSelectedAK4);
      for (int ii = 0; ii < nSelectedAK4; ++ii) {
        sortedScores.push_back(Jet_btag[selectedAK4[ii]]);
      }
      std::sort(sortedScores.begin(), sortedScores.end(), std::greater<double>());
      for (int ii = 0; ii < std::min<int>(3, sortedScores.size()); ++ii) {
        topScores[ii] = sortedScores[ii];
      }
    }

    if (pass_cat2) {
      out.hTop3Cat2->Fill(topScores[0], topScores[1], topScores[2]);
    }
    if (pass_cat3) {
      out.hTop3Cat3->Fill(topScores[0], topScores[1], topScores[2]);
    }

    if (nSelectedAK4 == 0) continue;

    double max_btag = 0;
    for (int ii = 0; ii < nSelectedAK4; ++ii) {
      const int idx = selectedAK4[ii];
      const double b = Jet_btag[idx];
      if (b > max_btag) max_btag = b;
    }

    if (pass_cat2) {
      out.hCat2->Fill(max_btag);
      ++nPass2;
    }
    if (pass_cat3) {
      out.hCat3->Fill(max_btag);
      ++nPass3;
    }
  }

  std::cout << "    processing " << sample.label << " : " << N << " / " << N << "\n";
  std::cout << "  [DONE] " << sample.label
            << " files=" << files.size()
            << " rawentry=" << out.rawEntries
            << " pass(cat2)=" << nPass2
            << " pass(cat3)=" << nPass3 << std::endl;

  chain.ResetBranchAddresses();
  chain.Reset();

  files.clear();
  files.shrink_to_fit();

  return true;
}

// Main

int b_veto() {
  gErrorIgnoreLevel = kWarning;

  std::cout << "[INFO] Processing " << SAMPLES.size() << " CMS datasets via DAS" << std::endl;

  std::vector<SampleResult> results;
  results.reserve(SAMPLES.size());

  for (size_t i = 0; i < SAMPLES.size(); ++i) {
    const auto& sample = SAMPLES[i];
    std::cout << "\n[START] Sample " << (i + 1) << "/" << SAMPLES.size()
              << " : " << sample.label << std::endl;
    std::cout << "  dataset = " << sample.dataset << std::endl;

    SampleResult r;
    if (process_one_sample(sample, r)) {
      results.push_back(r);
    } else {
      std::cout << "  [SKIP] " << sample.label << " due to errors or empty data." << std::endl;
    }
  }

  if (results.empty()) {
    std::cerr << "[FATAL] No valid samples processed. Exiting." << std::endl;
    return 2;
  }

  TH1D* hSig_cat2 = new TH1D("h_signal_cat2", "Signal (WWW, nFatJet==2);min Jet_btagUParTAK4B;Events", NBINS, XMIN, XMAX);
  TH1D* hSig_cat3 = new TH1D("h_signal_cat3", "Signal (WWW, nFatJet>2);min Jet_btagUParTAK4B;Events", NBINS, XMIN, XMAX);
  TH1D* hBkg_cat2 = new TH1D("h_background_cat2", "Background (TTbar, nFatJet==2);min Jet_btagUParTAK4B;Events", NBINS, XMIN, XMAX);
  TH1D* hBkg_cat3 = new TH1D("h_background_cat3", "Background (TTbar, nFatJet>2);min Jet_btagUParTAK4B;Events", NBINS, XMIN, XMAX);

  hSig_cat2->SetDirectory(nullptr);
  hSig_cat3->SetDirectory(nullptr);
  hBkg_cat2->SetDirectory(nullptr);
  hBkg_cat3->SetDirectory(nullptr);

  hSig_cat2->Sumw2();
  hSig_cat3->Sumw2();
  hBkg_cat2->Sumw2();
  hBkg_cat3->Sumw2();

  std::cout << "\n[AGGREGATE] Merge raw histograms into signal/background" << std::endl;
  for (auto& r : results) {
    std::cout << "  - " << std::setw(8) << std::left << r.sample.label
              << " raw=" << std::setw(12) << std::left << r.rawEntries
              << " integral(cat2)=" << std::setw(12) << std::left << r.hCat2->Integral()
              << " integral(cat3)=" << r.hCat3->Integral()
              << std::endl;

    if (r.sample.isSignal) {
      hSig_cat2->Add(r.hCat2);
      hSig_cat3->Add(r.hCat3);
    } else {
      hBkg_cat2->Add(r.hCat2);
      hBkg_cat3->Add(r.hCat3);
    }
  }

  std::unique_ptr<TFile> fout(TFile::Open(OUTFILE.c_str(), "RECREATE"));
  if (!fout || fout->IsZombie()) {
    std::cerr << "[FATAL] Cannot create output file: " << OUTFILE << std::endl;

    delete hSig_cat2;
    delete hSig_cat3;
    delete hBkg_cat2;
    delete hBkg_cat3;
    for (auto& r : results) {
      delete r.hCat2;
      delete r.hCat3;
      delete r.hJetScoreVsFlavor;
      delete r.hTop3Cat2;
      delete r.hTop3Cat3;
      r.hCat2 = nullptr;
      r.hCat3 = nullptr;
      r.hJetScoreVsFlavor = nullptr;
      r.hTop3Cat2 = nullptr;
      r.hTop3Cat3 = nullptr;
    }

    return 3;
  }

  std::cout << "\n[WRITE] Saving histograms to: " << OUTFILE << std::endl;
  fout->cd();
  hSig_cat2->Write();
  hSig_cat3->Write();
  hBkg_cat2->Write();
  hBkg_cat3->Write();
  for (auto& r : results) {
    r.hJetScoreVsFlavor->Write();
    r.hTop3Cat2->Write();
    r.hTop3Cat3->Write();
  }

  fout->mkdir("per_sample");
  fout->cd("per_sample");
  for (auto& r : results) {
    r.hCat2->Write();
    r.hCat3->Write();
  }

  fout->Write();
  fout->Close();
  fout.reset();

  for (auto& r : results) {
    delete r.hCat2;
    delete r.hCat3;
    delete r.hJetScoreVsFlavor;
    delete r.hTop3Cat2;
    delete r.hTop3Cat3;
    r.hCat2 = nullptr;
    r.hCat3 = nullptr;
    r.hJetScoreVsFlavor = nullptr;
    r.hTop3Cat2 = nullptr;
    r.hTop3Cat3 = nullptr;
  }

  delete hSig_cat2;
  delete hSig_cat3;
  delete hBkg_cat2;
  delete hBkg_cat3;

  std::cout << "[DONE] Wrote " << (4 + results.size() * 5)
            << " histograms. Bye.\n" << std::endl;

  return 0;
}
