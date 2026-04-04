// Summary: Build JetHT efficiency and distribution plots from scouting events.
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <iomanip>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TStyle.h"
#include "TLorentzVector.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TPad.h"

using namespace std;

// Valid types: "vh", "vvv", "www", "wwz", "2024*"
static const string TYPE = "2024F";
static const int N_DATA_FILES = 200;

static const string BASEDIR = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/MC/ScoutingNano/skim/v3/";
static const string XROOTD_REDIRECTOR = "root://cms-xrd-global.cern.ch/";

static const int MAX_FAT = 32;
static const int MAX_AK4 = 64;

struct EventVars {
    double sumFatPt = 0.0;
    double sumFatPtPlusNonOverlapAK4Pt = 0.0;
    double sumAK4HT = 0.0;
    int nNonOverlapAK4 = 0;
};

bool isRun2024Type(const string& type) {
    return (type.size() == 5 &&
            type.rfind("2024", 0) == 0 &&
            std::isalpha(static_cast<unsigned char>(type[4])));
}

string buildRun2024Dataset(const string& type) {
    return "/ScoutingPFRun3/Run" + type + "-ScoutNano-v1/NANOAOD";
}

vector<string> splitLines(const string& text) {
    vector<string> lines;
    std::istringstream iss(text);
    string line;
    while (std::getline(iss, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

vector<string> getDataFilesFromDAS(const string& type, int nDataFiles) {
    vector<string> files;
    const string dataset = buildRun2024Dataset(type);

    const string cmd = "dasgoclient -query=\"file dataset=" + dataset + "\"";
    cout << "Running DAS query: " << cmd << endl;

    TString output = gSystem->GetFromPipe(cmd.c_str());
    vector<string> rawFiles = splitLines(output.Data());

    if (rawFiles.empty()) {
        cerr << "ERROR: no files returned from DAS for dataset: " << dataset << endl;
        return files;
    }

    int nToTake = static_cast<int>(rawFiles.size());
    if (nDataFiles > 0) nToTake = std::min(nToTake, nDataFiles);

    for (int i = 0; i < nToTake; ++i) {
        files.push_back(XROOTD_REDIRECTOR + rawFiles[i]);
    }

    cout << "Dataset: " << dataset << endl;
    cout << "Total files returned by DAS = " << rawFiles.size() << endl;
    cout << "Files actually used         = " << files.size() << endl;

    return files;
}

vector<string> getInputFiles(const string& type, int nDataFiles) {
    if (type == "vvv") {
        return {
            BASEDIR + "www.root",
            BASEDIR + "wwz.root",
            BASEDIR + "wzz.root",
            BASEDIR + "zzz.root"
        };
    }

    if (type == "vh") {
        return {
            BASEDIR + "wminush.root",
            BASEDIR + "wplush.root",
            BASEDIR + "zh.root"
        };
    }

    if (isRun2024Type(type)) {
        return getDataFilesFromDAS(type, nDataFiles);
    }

    return { BASEDIR + type + ".root" };
}

double clampToHistRange(double x, const TH1D* h) {
    const double xmin = h->GetXaxis()->GetXmin();
    const double xmax = h->GetXaxis()->GetXmax();
    if (x < xmin) return xmin + 1e-6;
    if (x >= xmax) return xmax - 1e-6;
    return x;
}

double deltaR(double eta1, double phi1, double eta2, double phi2) {
    TLorentzVector v1, v2;
    v1.SetPtEtaPhiM(1.0, eta1, phi1, 0.0);
    v2.SetPtEtaPhiM(1.0, eta2, phi2, 0.0);
    return v1.DeltaR(v2);
}

bool allFatJetsPassPtEta(int nFat, const float* fat_pt, const float* fat_eta,
                         double ptCut = 180.0, double etaCut = 2.4) {
    for (int i = 0; i < nFat; ++i) {
        if (fat_pt[i] <= ptCut) return false;
        if (std::fabs(fat_eta[i]) >= etaCut) return false;
    }
    return true;
}

EventVars computeEventVars(
    int nFat,
    const float* fat_pt,
    const float* fat_eta,
    const float* fat_phi,
    int nAK4,
    const float* ak4_pt,
    const float* ak4_eta,
    const float* ak4_phi
) {
    EventVars out;

    for (int i = 0; i < nFat; ++i) {
        if (fat_pt[i] > 175.0 && std::fabs(fat_eta[i]) < 2.4) {
            out.sumFatPt += fat_pt[i];
        }
    }

    for (int j = 0; j < nAK4; ++j) {
        const bool passAK4Kinematics = (ak4_pt[j] > 30.0 && std::fabs(ak4_eta[j]) < 2.4);
        
        if (passAK4Kinematics) {
            out.sumAK4HT += ak4_pt[j];
        }

        bool overlap = false;
        for (int i = 0; i < nFat; ++i) {
            if (deltaR(fat_eta[i], fat_phi[i], ak4_eta[j], ak4_phi[j]) <= 0.8) {
                overlap = true;
                break;
            }
        }

        if (!overlap && passAK4Kinematics) {
            out.sumFatPtPlusNonOverlapAK4Pt += ak4_pt[j];
            out.nNonOverlapAK4++;
        }
    }

    out.sumFatPtPlusNonOverlapAK4Pt += out.sumFatPt;
    return out;
}

TH1D* buildEfficiencyHistogram(TH1D* h_num, TH1D* h_den, const string& name) {
    TH1D* h_eff = (TH1D*)h_num->Clone(name.c_str());
    h_eff->Reset();
    h_eff->Divide(h_num, h_den, 1.0, 1.0, "B");
    return h_eff;
}

void styleEfficiencyHistogram(TH1D* h, const string& title, const string& xTitle) {
    h->SetTitle((title + ";" + xTitle + ";Efficiency").c_str());
    h->SetMinimum(0.0);
    h->SetMaximum(1.05);
    h->SetLineWidth(2);
    h->SetMarkerStyle(20);
    h->SetMarkerSize(1.0);

    h->GetXaxis()->SetTitleSize(0.045);
    h->GetYaxis()->SetTitleSize(0.045);
    h->GetXaxis()->SetLabelSize(0.04);
    h->GetYaxis()->SetLabelSize(0.04);
}

void drawEfficiencyHistogram(TH1D* h_eff, const string& pdfName,
                             const string& title, const string& xTitle) {
    TCanvas c("c_eff", "c_eff", 800, 700);
    c.SetMargin(0.12, 0.04, 0.12, 0.08);

    styleEfficiencyHistogram(h_eff, title, xTitle);
    h_eff->Draw("E1");

    gPad->Modified();
    gPad->Update();

    c.SaveAs(pdfName.c_str());
}

void drawHistogram(TH1D* h, const string& pdfName,
                   const string& title, const string& xTitle) {
    TCanvas c("c_hist", "c_hist", 800, 700);
    c.SetMargin(0.12, 0.04, 0.12, 0.08);

    h->SetTitle((title + ";" + xTitle + ";Events").c_str());
    h->SetLineWidth(2);
    h->SetMinimum(0);
    h->GetXaxis()->SetTitleSize(0.045);
    h->GetYaxis()->SetTitleSize(0.045);
    h->GetXaxis()->SetLabelSize(0.04);
    h->GetYaxis()->SetLabelSize(0.04);
    h->Draw("hist");

    c.SaveAs(pdfName.c_str());
}

int efficiency(const string& type = TYPE, int nDataFiles = N_DATA_FILES) {
    const bool isData2024 = isRun2024Type(type);

    string outDir = "efficiency_" + type;
    gSystem->mkdir(outDir.c_str(), true);

    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);

    vector<string> inputFiles = getInputFiles(type, nDataFiles);
    if (inputFiles.empty()) {
        cerr << "ERROR: no input files found." << endl;
        return 1;
    }

    TChain chain("Events");
    for (size_t i = 0; i < inputFiles.size(); ++i) {
        cout << "Adding file: " << inputFiles[i] << endl;
        chain.Add(inputFiles[i].c_str());
    }

    // ---- Only enable needed branches ----
    chain.SetBranchStatus("*", 0);

    chain.SetBranchStatus("nScoutingFatPFJetRecluster", 1);
    chain.SetBranchStatus("ScoutingFatPFJetRecluster_pt", 1);
    chain.SetBranchStatus("ScoutingFatPFJetRecluster_eta", 1);
    chain.SetBranchStatus("ScoutingFatPFJetRecluster_phi", 1);

    chain.SetBranchStatus("nScoutingPFJetRecluster", 1);
    chain.SetBranchStatus("ScoutingPFJetRecluster_pt", 1);
    chain.SetBranchStatus("ScoutingPFJetRecluster_eta", 1);
    chain.SetBranchStatus("ScoutingPFJetRecluster_phi", 1);

    chain.SetBranchStatus("DST_PFScouting_JetHT", 1);

    if (isData2024) {
        chain.SetBranchStatus("DST_PFScouting_ZeroBias", 1);
    }

    // ---- Branch variables ----
    Int_t nFat = 0;
    Int_t nAK4 = 0;
    Bool_t DST_PFScouting_JetHT = false;
    Bool_t DST_PFScouting_ZeroBias = false;

    Float_t fat_pt[MAX_FAT]  = {0};
    Float_t fat_eta[MAX_FAT] = {0};
    Float_t fat_phi[MAX_FAT] = {0};

    Float_t ak4_pt[MAX_AK4]  = {0};
    Float_t ak4_eta[MAX_AK4] = {0};
    Float_t ak4_phi[MAX_AK4] = {0};

    chain.SetBranchAddress("nScoutingFatPFJetRecluster", &nFat);
    chain.SetBranchAddress("ScoutingFatPFJetRecluster_pt", fat_pt);
    chain.SetBranchAddress("ScoutingFatPFJetRecluster_eta", fat_eta);
    chain.SetBranchAddress("ScoutingFatPFJetRecluster_phi", fat_phi);

    chain.SetBranchAddress("nScoutingPFJetRecluster", &nAK4);
    chain.SetBranchAddress("ScoutingPFJetRecluster_pt", ak4_pt);
    chain.SetBranchAddress("ScoutingPFJetRecluster_eta", ak4_eta);
    chain.SetBranchAddress("ScoutingPFJetRecluster_phi", ak4_phi);

    chain.SetBranchAddress("DST_PFScouting_JetHT", &DST_PFScouting_JetHT);

    if (isData2024) {
        chain.SetBranchAddress("DST_PFScouting_ZeroBias", &DST_PFScouting_ZeroBias);
    }

    // ---- Histograms ----
    const int nbin_eff = 40;
    const double x1_min = 0.0, x1_max = 2500.0;
    const double x2_min = 0.0, x2_max = 4000.0;
    const double x3_min = 0.0, x3_max = 4000.0;

    TH1D* h_den_sumFat = new TH1D("h_den_sumFat", "", nbin_eff, x1_min, x1_max);
    TH1D* h_num_sumFat = new TH1D("h_num_sumFat", "", nbin_eff, x1_min, x1_max);

    TH1D* h_den_sumFatPlusAK4 = new TH1D("h_den_sumFatPlusAK4", "", nbin_eff, x2_min, x2_max);
    TH1D* h_num_sumFatPlusAK4 = new TH1D("h_num_sumFatPlusAK4", "", nbin_eff, x2_min, x2_max);

    TH1D* h_den_AK4HT = new TH1D("h_den_AK4HT", "", nbin_eff, x3_min, x3_max);
    TH1D* h_num_AK4HT = new TH1D("h_num_AK4HT", "", nbin_eff, x3_min, x3_max);

    TH1D* h_sumFat_nFat2 = new TH1D("h_sumFat_nFat2", "", 40, x1_min, x1_max);
    TH1D* h_sumFat_nFat3 = new TH1D("h_sumFat_nFat3", "", 40, x1_min, x1_max);

    TH1D* h_sumFatPlusAK4_nFat2 = new TH1D("h_sumFatPlusAK4_nFat2", "", 40, x2_min, x2_max);
    TH1D* h_sumFatPlusAK4_nFat3 = new TH1D("h_sumFatPlusAK4_nFat3", "", 40, x2_min, x2_max);

    TH1D* h_AK4HT_nFat2 = new TH1D("h_AK4HT_nFat2", "", 40, x3_min, x3_max);
    TH1D* h_AK4HT_nFat3 = new TH1D("h_AK4HT_nFat3", "", 40, x3_min, x3_max);

    Long64_t nEntries = chain.GetEntries();
    cout << "Total entries = " << nEntries << endl;

    for (Long64_t ievt = 0; ievt < nEntries; ++ievt)
    {
        chain.GetEntry(ievt);

        if (ievt % 1000000 == 0)
        {
            double frac = (nEntries > 0) ? (100.0 * (ievt + 1) / nEntries) : 100.0;
            cout << "\rProcessing entry " << (ievt + 1) << " / " << nEntries
                 << " (" << fixed << setprecision(2) << frac << "%)" << std::flush;
        }

        const int nFatUse = std::max(0, std::min((int)nFat, MAX_FAT));
        const int nAK4Use = std::max(0, std::min((int)nAK4, MAX_AK4));

        // ---- 2024 data mode: first require ZeroBias != 0 ----
        if (isData2024) {
            if (!DST_PFScouting_ZeroBias) continue;
        }

        EventVars vars = computeEventVars(
            nFatUse, fat_pt, fat_eta, fat_phi,
            nAK4Use, ak4_pt, ak4_eta, ak4_phi
        );

        // ---- Efficiency histograms ----
        {
            double x = clampToHistRange(vars.sumFatPt, h_den_sumFat);
            h_den_sumFat->Fill(x);
            if (DST_PFScouting_JetHT) {
                h_num_sumFat->Fill(x);
            }
        }

        {
            double x = clampToHistRange(vars.sumFatPtPlusNonOverlapAK4Pt, h_den_sumFatPlusAK4);
            h_den_sumFatPlusAK4->Fill(x);
            if (DST_PFScouting_JetHT) {
                h_num_sumFatPlusAK4->Fill(x);
            }
        }

        {
            double x = clampToHistRange(vars.sumAK4HT, h_den_AK4HT);
            h_den_AK4HT->Fill(x);
            if (DST_PFScouting_JetHT) {
                h_num_AK4HT->Fill(x);
            }
        }

        // ---- Distribution selections ----
        if (nFat == 2) {
            bool fat2_pass = allFatJetsPassPtEta(2, fat_pt, fat_eta, 180.0, 2.4);
            bool ak4_pass = (vars.nNonOverlapAK4 >= 2);

            if (fat2_pass && ak4_pass) {
                h_sumFat_nFat2->Fill(clampToHistRange(vars.sumFatPt, h_sumFat_nFat2));
                h_sumFatPlusAK4_nFat2->Fill(clampToHistRange(vars.sumFatPtPlusNonOverlapAK4Pt, h_sumFatPlusAK4_nFat2));
                h_AK4HT_nFat2->Fill(clampToHistRange(vars.sumAK4HT, h_AK4HT_nFat2));
            }
        }

        if (nFat >= 3) {
            bool fat3_pass = allFatJetsPassPtEta(3, fat_pt, fat_eta, 180.0, 2.4);

            if (fat3_pass) {
                h_sumFat_nFat3->Fill(clampToHistRange(vars.sumFatPt, h_sumFat_nFat3));
                h_sumFatPlusAK4_nFat3->Fill(clampToHistRange(vars.sumFatPtPlusNonOverlapAK4Pt, h_sumFatPlusAK4_nFat3));
                h_AK4HT_nFat3->Fill(clampToHistRange(vars.sumAK4HT, h_AK4HT_nFat3));
            }
        }
    }
    cout << endl;

    // ---- Build efficiency histograms from numerator / denominator ----
    TH1D* h_eff_sumFat = buildEfficiencyHistogram(h_num_sumFat, h_den_sumFat, "h_eff_sumFat");
    TH1D* h_eff_sumFatPlusAK4 = buildEfficiencyHistogram(h_num_sumFatPlusAK4, h_den_sumFatPlusAK4, "h_eff_sumFatPlusAK4");
    TH1D* h_eff_AK4HT = buildEfficiencyHistogram(h_num_AK4HT, h_den_AK4HT, "h_eff_AK4HT");

    // ---- Save ROOT file ----
    string outRoot = outDir + "/jetht_efficiency_and_distributions_" + type;
    if (isData2024 && nDataFiles > 0) {
        outRoot += "_" + to_string(nDataFiles) + "files";
    }
    outRoot += ".root";

    TFile fout(outRoot.c_str(), "RECREATE");

    h_den_sumFat->Write();
    h_num_sumFat->Write();
    h_den_sumFatPlusAK4->Write();
    h_num_sumFatPlusAK4->Write();
    h_den_AK4HT->Write();
    h_num_AK4HT->Write();

    h_eff_sumFat->Write();
    h_eff_sumFatPlusAK4->Write();
    h_eff_AK4HT->Write();

    h_sumFat_nFat2->Write();
    h_sumFat_nFat3->Write();
    h_sumFatPlusAK4_nFat2->Write();
    h_sumFatPlusAK4_nFat3->Write();
    h_AK4HT_nFat2->Write();
    h_AK4HT_nFat3->Write();

    fout.Close();

    // ---- Save PDFs ----
    drawEfficiencyHistogram(
        h_eff_sumFat,
        outDir + "/eff_AK8_HT_" + type + ".pdf",
        "DST_PFScouting_JetHT efficiency",
        "#sum p_{T}^{AK8} [GeV]"
    );

    drawEfficiencyHistogram(
        h_eff_sumFatPlusAK4,
        outDir + "/eff_AK8_AK4_HT_" + type + ".pdf",
        "DST_PFScouting_JetHT efficiency",
        "#sum p_{T}^{AK8} + #sum p_{T}^{AK4, #DeltaR>0.8} [GeV]"
    );

    drawEfficiencyHistogram(
        h_eff_AK4HT,
        outDir + "/eff_AK4_HT_" + type + ".pdf",
        "DST_PFScouting_JetHT efficiency",
        "#sum p_{T}^{AK4} [GeV]"
    );

    drawHistogram(
        h_sumFat_nFat2,
        outDir + "/dist_AK8_HT_nFat2_" + type + ".pdf",
        "2 AK8",
        "#sum p_{T}^{AK8} [GeV]"
    );

    drawHistogram(
        h_sumFat_nFat3,
        outDir + "/dist_AK8_HT_nFat3_" + type + ".pdf",
        ">=3 AK8",
        "#sum p_{T}^{AK8} [GeV]"
    );

    drawHistogram(
        h_sumFatPlusAK4_nFat2,
        outDir + "/dist_AK8_AK4_nFat2_" + type + ".pdf",
        "2 AK8",
        "#sum p_{T}^{AK8} + #sum p_{T}^{AK4, #DeltaR>0.8} [GeV]"
    );

    drawHistogram(
        h_sumFatPlusAK4_nFat3,
        outDir + "/dist_AK8_AK4_nFat3_" + type + ".pdf",
        ">=3 AK8",
        "#sum p_{T}^{AK8} + #sum p_{T}^{AK4, #DeltaR>0.8} [GeV]"
    );

    drawHistogram(
        h_AK4HT_nFat2,
        outDir + "/dist_AK4_HT_nFat2_" + type + ".pdf",
        "2 AK8",
        "#sum p_{T}^{AK4} [GeV]"
    );

    drawHistogram(
        h_AK4HT_nFat3,
        outDir + "/dist_AK4_HT_nFat3_" + type + ".pdf",
        ">=3 AK8",
        "#sum p_{T}^{AK4} [GeV]"
    );

    cout << "All done. Output written to: " << outDir << endl;
    return 0;
}
