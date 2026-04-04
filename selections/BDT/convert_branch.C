// Summary: Skim events and convert branches for BDT training.
#include <iostream>
#include <algorithm>
#include <TFile.h>
#include <TTree.h>
#include <TLorentzVector.h>
#include <TVector2.h>

using namespace std;

// Simple struct to hold leptons
struct Lepton {
    float pt;
    float eta;
    float phi;
    float ecalIso;
    float hcalIso;
};

const float def = -99.;

// --- constants for maximum sizes ---
const int MAX_ELE   = 30;
const int MAX_MU    = 30;
const int MAX_FAT8  = 20;
const int MAX_PF    = 30;
const int MAX_LEP   = MAX_MU * 2 + MAX_ELE;

void convert_branch(const char* typeArg) {
    
    string type(typeArg);
    string subdir = "bkg";
    if (type.find("www") != string::npos || type.find("wwz") != string::npos ||
        type.find("wzz") != string::npos || type.find("zzz") != string::npos ||
        type.find("zh") != string::npos  || type.find("wminush") != string::npos || 
        type.find("wplush") != string::npos) {
        subdir = "signal";
    }
    if (type.find("202") != string::npos) subdir = "data";
    cout << "Running convert_branch with type = " << type << endl;

    // Open input
    string inputFileName  = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/MC/ScoutingNano/skim/v3/" + type + ".root";
    if (type.find("202") != string::npos) inputFileName  = "/afs/ihep.ac.cn/users/y/yiyangzhao/Research/CMS_THU_Space/VVV/2024/skim/" + type + ".root";
    const string outputFileName = "dataset/" + subdir + "/" + type + ".root";
    TFile* inFile = TFile::Open(inputFileName.c_str(), "READ");
    if (!inFile || inFile->IsZombie()) {
        cerr << "Error opening input file " << inputFileName << endl;
        return;
    }
    TTree* tree = (TTree*)inFile->Get("Events");
    if (!tree) {
        cerr << "Error: Events tree not found in " << inputFileName << endl;
        inFile->Close();
        return;
    }

    // --- INPUT BRANCHES using C-arrays ---
    Int_t nFat8 = 0, nJet4 = 0, nEle = 0, nMuNoVtx = 0, nMuVtx = 0, nPho = 0, nPV = 0;
    tree->SetBranchAddress("nScoutingFatPFJetRecluster",  &nFat8);
    tree->SetBranchAddress("nScoutingPFJetRecluster",     &nJet4);
    tree->SetBranchAddress("nScoutingElectron",           &nEle);
    tree->SetBranchAddress("nScoutingMuonNoVtx",          &nMuNoVtx);
    tree->SetBranchAddress("nScoutingMuonVtx",            &nMuVtx);
    tree->SetBranchAddress("nScoutingPhoton",             &nPho);
    tree->SetBranchAddress("nScoutingPrimaryVertex",      &nPV);

    // electrons
    Float_t ele_pt[MAX_ELE], ele_ecalIso[MAX_ELE], ele_hcalIso[MAX_ELE], ele_eta[MAX_ELE] = {}, ele_phi[MAX_ELE] = {};
    tree->SetBranchAddress("ScoutingElectron_pt",      ele_pt);
    tree->SetBranchAddress("ScoutingElectron_eta",     ele_eta);
    tree->SetBranchAddress("ScoutingElectron_phi",     ele_phi);
    tree->SetBranchAddress("ScoutingElectron_ecalIso", ele_ecalIso);
    tree->SetBranchAddress("ScoutingElectron_hcalIso", ele_hcalIso);

    // muons no vtx
    Float_t muNV_pt[MAX_MU], muNV_ecalIso[MAX_MU], muNV_hcalIso[MAX_MU], muNV_eta[MAX_MU] = {}, muNV_phi[MAX_MU] = {};
    tree->SetBranchAddress("ScoutingMuonNoVtx_pt",      muNV_pt);
    tree->SetBranchAddress("ScoutingMuonNoVtx_eta",     muNV_eta);
    tree->SetBranchAddress("ScoutingMuonNoVtx_phi",     muNV_phi);
    tree->SetBranchAddress("ScoutingMuonNoVtx_ecalIso", muNV_ecalIso);
    tree->SetBranchAddress("ScoutingMuonNoVtx_hcalIso", muNV_hcalIso);

    // muons vtx
    Float_t muV_pt[MAX_MU], muV_ecalIso[MAX_MU], muV_hcalIso[MAX_MU], muV_eta[MAX_MU] = {}, muV_phi[MAX_MU] = {};
    tree->SetBranchAddress("ScoutingMuonVtx_pt",      muV_pt);
    tree->SetBranchAddress("ScoutingMuonVtx_eta",     muV_eta);
    tree->SetBranchAddress("ScoutingMuonVtx_phi",     muV_phi);
    tree->SetBranchAddress("ScoutingMuonVtx_ecalIso", muV_ecalIso);
    tree->SetBranchAddress("ScoutingMuonVtx_hcalIso", muV_hcalIso);

    // AK8 jets
    Float_t fat_pt[MAX_FAT8], fat_eta[MAX_FAT8], fat_phi[MAX_FAT8];
    Float_t fat_msoftdrop[MAX_FAT8], fat_mass[MAX_FAT8], fat_area[MAX_FAT8];
    Int_t fat_nCh[MAX_FAT8], fat_nEle[MAX_FAT8], fat_nMu[MAX_FAT8];
    Int_t fat_nNh[MAX_FAT8], fat_nPho[MAX_FAT8];
    Float_t fat_chEmEF[MAX_FAT8], fat_chHEF[MAX_FAT8];
    Float_t fat_hfEmEF[MAX_FAT8], fat_hfHEF[MAX_FAT8];
    Float_t fat_neEmEF[MAX_FAT8], fat_neHEF[MAX_FAT8], fat_muEF[MAX_FAT8];
    Float_t fat_n2b1[MAX_FAT8], fat_n3b1[MAX_FAT8];
    Float_t fat_tau1[MAX_FAT8], fat_tau2[MAX_FAT8], fat_tau3[MAX_FAT8], fat_tau4[MAX_FAT8];
    Float_t fat_Xud[MAX_FAT8], fat_Xcs[MAX_FAT8], fat_QCD[MAX_FAT8];
    UChar_t fat_nConst[MAX_FAT8];
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_pt",            fat_pt);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_eta",           fat_eta);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_phi",           fat_phi);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_msoftdrop",     fat_msoftdrop);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_mass",          fat_mass);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_nConstituents", fat_nConst);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_nCh",           fat_nCh);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_nElectrons",    fat_nEle);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_nMuons",        fat_nMu);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_nNh",           fat_nNh);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_nPhotons",      fat_nPho);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_area",          fat_area);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_chEmEF",        fat_chEmEF);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_chHEF",         fat_chHEF);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_hfEmEF",        fat_hfEmEF);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_hfHEF",         fat_hfHEF);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_neEmEF",        fat_neEmEF);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_neHEF",         fat_neHEF);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_muEF",          fat_muEF);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_n2b1",          fat_n2b1);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_n3b1",          fat_n3b1);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_tau1",          fat_tau1);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_tau2",          fat_tau2);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_tau3",          fat_tau3);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_tau4",          fat_tau4);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_scoutGlobalParT_prob_Xud",  fat_Xud);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_scoutGlobalParT_prob_Xcs",  fat_Xcs);
    tree->SetBranchAddress("ScoutingFatPFJetRecluster_scoutGlobalParT_prob_QCD",   fat_QCD);

    // AK4 jets
    Float_t pf_pt[MAX_PF], pf_eta[MAX_PF], pf_phi[MAX_PF], pf_mass[MAX_PF], pf_area[MAX_PF];
    Int_t pf_nCh[MAX_PF], pf_nEle[MAX_PF], pf_nMu[MAX_PF];
    Int_t pf_nNh[MAX_PF], pf_nPho[MAX_PF];
    Float_t pf_chEmEF[MAX_PF], pf_chHEF[MAX_PF];
    Float_t pf_hfEmEF[MAX_PF], pf_hfHEF[MAX_PF];
    Float_t pf_neEmEF[MAX_PF], pf_neHEF[MAX_PF], pf_muEF[MAX_PF];
    UChar_t pf_nConst[MAX_PF];
    tree->SetBranchAddress("ScoutingPFJetRecluster_pt",            pf_pt);
    tree->SetBranchAddress("ScoutingPFJetRecluster_eta",           pf_eta);
    tree->SetBranchAddress("ScoutingPFJetRecluster_phi",           pf_phi);
    tree->SetBranchAddress("ScoutingPFJetRecluster_mass",          pf_mass);
    tree->SetBranchAddress("ScoutingPFJetRecluster_nConstituents", pf_nConst);
    tree->SetBranchAddress("ScoutingPFJetRecluster_nCh",           pf_nCh);
    tree->SetBranchAddress("ScoutingPFJetRecluster_nElectrons",    pf_nEle);
    tree->SetBranchAddress("ScoutingPFJetRecluster_nMuons",        pf_nMu);
    tree->SetBranchAddress("ScoutingPFJetRecluster_nNh",           pf_nNh);
    tree->SetBranchAddress("ScoutingPFJetRecluster_nPhotons",      pf_nPho);
    tree->SetBranchAddress("ScoutingPFJetRecluster_area",          pf_area);
    tree->SetBranchAddress("ScoutingPFJetRecluster_chEmEF",        pf_chEmEF);
    tree->SetBranchAddress("ScoutingPFJetRecluster_chHEF",         pf_chHEF);
    tree->SetBranchAddress("ScoutingPFJetRecluster_hfEmEF",        pf_hfEmEF);
    tree->SetBranchAddress("ScoutingPFJetRecluster_hfHEF",         pf_hfHEF);
    tree->SetBranchAddress("ScoutingPFJetRecluster_neEmEF",        pf_neEmEF);
    tree->SetBranchAddress("ScoutingPFJetRecluster_neHEF",         pf_neHEF);
    tree->SetBranchAddress("ScoutingPFJetRecluster_muEF",          pf_muEF);

    // --- OUTPUT FILE & TREES ---
    TFile* outFile = TFile::Open(outputFileName.c_str(), "RECREATE");
    TTree* tree2 = new TTree("fat2", "2 AK8 + >=2 AK4");
    TTree* tree3 = new TTree("fat3", ">=3 AK8");

    // --- OUTPUT VARIABLES FOR tree2 ---
    Bool_t out2_isSignal = false;
    Int_t out2_type = 0;
    Int_t   out2_Nak8, out2_Nak4, out2_NPV, out2_Ne, out2_Nmu, out2_Npho;
    Float_t out2_HT;
    // AK8 up to 2
    Float_t out2_pt8_1, out2_pt8_2;
    Float_t out2_eta8_1, out2_eta8_2;
    Float_t out2_msd8_1, out2_msd8_2;
    Float_t out2_mr8_1,  out2_mr8_2;
    Float_t out2_nConst8_1, out2_nConst8_2;
    Float_t out2_nCh8_1,    out2_nCh8_2;
    Float_t out2_nEle8_1,   out2_nEle8_2;
    Float_t out2_nMu8_1,    out2_nMu8_2;
    Float_t out2_nNh8_1,    out2_nNh8_2;
    Float_t out2_nPho8_1,   out2_nPho8_2;
    Float_t out2_area8_1,   out2_area8_2;
    Float_t out2_chEmEF8_1, out2_chEmEF8_2;
    Float_t out2_chHEF8_1,  out2_chHEF8_2;
    Float_t out2_hfEmEF8_1, out2_hfEmEF8_2;
    Float_t out2_hfHEF8_1,  out2_hfHEF8_2;
    Float_t out2_neEmEF8_1, out2_neEmEF8_2;
    Float_t out2_neHEF8_1,  out2_neHEF8_2;
    Float_t out2_muEF8_1,   out2_muEF8_2;
    Float_t out2_n2b1_1,    out2_n2b1_2;
    Float_t out2_n3b1_1,    out2_n3b1_2;
    Float_t out2_tau21_1,    out2_tau21_2;
    Float_t out2_tau32_1,    out2_tau32_2;
    Float_t out2_WvsQCD_1, out2_WvsQCD_2;
    // AK4 up to 4
    Float_t out2_pt4_1,   out2_pt4_2,     out2_pt4_3,    out2_pt4_4;
    Float_t out2_eta4_1,  out2_eta4_2,   out2_eta4_3,   out2_eta4_4;
    Float_t out2_mPF4_1,  out2_mPF4_2,    out2_mPF4_3,   out2_mPF4_4;
    Float_t out2_nConst4_1,out2_nConst4_2,out2_nConst4_3,out2_nConst4_4;
    Float_t out2_nCh4_1,   out2_nCh4_2,   out2_nCh4_3,   out2_nCh4_4;
    Float_t out2_nEle4_1,  out2_nEle4_2,  out2_nEle4_3,  out2_nEle4_4;
    Float_t out2_nMu4_1,   out2_nMu4_2,   out2_nMu4_3,   out2_nMu4_4;
    Float_t out2_nNh4_1,   out2_nNh4_2,   out2_nNh4_3,   out2_nNh4_4;
    Float_t out2_nPho4_1,  out2_nPho4_2,  out2_nPho4_3,  out2_nPho4_4;
    Float_t out2_area4_1,  out2_area4_2,  out2_area4_3,  out2_area4_4;
    Float_t out2_chEmEF4_1,out2_chEmEF4_2,out2_chEmEF4_3,out2_chEmEF4_4;
    Float_t out2_chHEF4_1, out2_chHEF4_2, out2_chHEF4_3, out2_chHEF4_4;
    Float_t out2_hfEmEF4_1,out2_hfEmEF4_2,out2_hfEmEF4_3,out2_hfEmEF4_4;
    Float_t out2_hfHEF4_1, out2_hfHEF4_2, out2_hfHEF4_3, out2_hfHEF4_4;
    Float_t out2_neEmEF4_1,out2_neEmEF4_2,out2_neEmEF4_3,out2_neEmEF4_4;
    Float_t out2_neHEF4_1, out2_neHEF4_2, out2_neHEF4_3, out2_neHEF4_4;
    Float_t out2_muEF4_1,  out2_muEF4_2,  out2_muEF4_3,  out2_muEF4_4;
    // Angular/mass
    Float_t out2_dR8, out2_dPhi;
    Float_t out2_M8, out2_m1overM, out2_m2overM, out2_M84, out2_sphereM;
    Float_t out2_PT, out2_pt1overPT, out2_pt2overPT, out2_PToverptsum;
    Float_t out2_dR84_min, out2_dR44_min;
    Float_t out2_dR8L_min;
    // Lepton up to 3
    Float_t out2_ptL_1, out2_ptL_2, out2_ptL_3;
    Float_t out2_etaL_1, out2_etaL_2, out2_etaL_3;
    Float_t out2_phiL_1, out2_phiL_2, out2_phiL_3;
    Float_t out2_isoEcalL_1, out2_isoEcalL_2, out2_isoEcalL_3;
    Float_t out2_isoHcalL_1, out2_isoHcalL_2, out2_isoHcalL_3;

    // --- OUTPUT VARIABLES FOR tree3 ---
    Bool_t out3_isSignal = false;
    Int_t out3_type = 0;
    Int_t   out3_Nak8, out3_Nak4, out3_NPV, out3_Ne, out3_Nmu, out3_Npho;
    Float_t out3_HT;
    // AK8 up to 3
    Float_t out3_pt8_1, out3_pt8_2, out3_pt8_3;
    Float_t out3_eta8_1, out3_eta8_2, out3_eta8_3;
    Float_t out3_msd8_1, out3_msd8_2, out3_msd8_3;
    Float_t out3_mr8_1,  out3_mr8_2,  out3_mr8_3;
    Float_t out3_nConst8_1, out3_nConst8_2, out3_nConst8_3;
    Float_t out3_nCh8_1,    out3_nCh8_2,    out3_nCh8_3;
    Float_t out3_nEle8_1,   out3_nEle8_2,   out3_nEle8_3;
    Float_t out3_nMu8_1,    out3_nMu8_2,    out3_nMu8_3;
    Float_t out3_nNh8_1,    out3_nNh8_2,    out3_nNh8_3;
    Float_t out3_nPho8_1,   out3_nPho8_2,   out3_nPho8_3;
    Float_t out3_area8_1,   out3_area8_2,   out3_area8_3;
    Float_t out3_chEmEF8_1, out3_chEmEF8_2, out3_chEmEF8_3;
    Float_t out3_chHEF8_1,  out3_chHEF8_2,  out3_chHEF8_3;
    Float_t out3_hfEmEF8_1, out3_hfEmEF8_2, out3_hfEmEF8_3;
    Float_t out3_hfHEF8_1,  out3_hfHEF8_2,  out3_hfHEF8_3;
    Float_t out3_neEmEF8_1, out3_neEmEF8_2, out3_neEmEF8_3;
    Float_t out3_neHEF8_1,  out3_neHEF8_2,  out3_neHEF8_3;
    Float_t out3_muEF8_1,   out3_muEF8_2,   out3_muEF8_3;
    Float_t out3_n2b1_1,    out3_n2b1_2,    out3_n2b1_3;
    Float_t out3_n3b1_1,    out3_n3b1_2,    out3_n3b1_3;
    Float_t out3_tau21_1,    out3_tau21_2,    out3_tau21_3;
    Float_t out3_tau32_1,    out3_tau32_2,    out3_tau32_3;
    Float_t out3_WvsQCD_1, out3_WvsQCD_2, out3_WvsQCD_3;
    // Angular/mass
    Float_t out3_M, out3_m1overM, out3_m2overM, out3_m3overM, out3_sphereM;
    Float_t out3_PT, out3_pt1overPT, out3_pt2overPT, out3_pt3overPT, out3_PToverptsum;
    Float_t out3_dR_min, out3_dR_max, out3_dPhi_min, out3_dPhi_max, out3_dRL_min;
    // Leptons up to 3
    Float_t out3_ptL_1, out3_ptL_2, out3_ptL_3;
    Float_t out3_etaL_1, out3_etaL_2, out3_etaL_3;
    Float_t out3_phiL_1, out3_phiL_2, out3_phiL_3;
    Float_t out3_isoEcalL_1, out3_isoEcalL_2, out3_isoEcalL_3;
    Float_t out3_isoHcalL_1, out3_isoHcalL_2, out3_isoHcalL_3;

    // --- BRANCHES FOR tree2 ---
    tree2->Branch("isSignal", &out2_isSignal, "isSignal/O");
    tree2->Branch("type",     &out2_type,     "type/I");
    tree2->Branch("N_ak8",   &out2_Nak8,   "N_ak8/I");
    tree2->Branch("N_ak4",   &out2_Nak4,   "N_ak4/I");
    tree2->Branch("N_PV",    &out2_NPV,    "N_PV/I");
    tree2->Branch("N_e",     &out2_Ne,     "N_e/I");
    tree2->Branch("N_mu",    &out2_Nmu,    "N_mu/I");
    tree2->Branch("N_gamma", &out2_Npho,   "N_gamma/I");
    tree2->Branch("H_T",     &out2_HT,     "H_T/F");
    // AK8_1,2
    tree2->Branch("pt8_1",    &out2_pt8_1,    "pt8_1/F");
    tree2->Branch("pt8_2",    &out2_pt8_2,    "pt8_2/F");
    tree2->Branch("eta8_1",   &out2_eta8_1,   "eta8_1/F");
    tree2->Branch("eta8_2",   &out2_eta8_2,   "eta8_2/F");
    tree2->Branch("msd8_1",   &out2_msd8_1,   "msd8_1/F");
    tree2->Branch("msd8_2",   &out2_msd8_2,   "msd8_2/F");
    tree2->Branch("mr8_1",    &out2_mr8_1,    "mr8_1/F");
    tree2->Branch("mr8_2",    &out2_mr8_2,    "mr8_2/F");
    tree2->Branch("nConst8_1",&out2_nConst8_1,"nConst8_1/F");
    tree2->Branch("nConst8_2",&out2_nConst8_2,"nConst8_2/F");
    tree2->Branch("nCh8_1",   &out2_nCh8_1,   "nCh8_1/F");
    tree2->Branch("nCh8_2",   &out2_nCh8_2,   "nCh8_2/F");
    tree2->Branch("nEle8_1",  &out2_nEle8_1,  "nEle8_1/F");
    tree2->Branch("nEle8_2",  &out2_nEle8_2,  "nEle8_2/F");
    tree2->Branch("nMu8_1",   &out2_nMu8_1,   "nMu8_1/F");
    tree2->Branch("nMu8_2",   &out2_nMu8_2,   "nMu8_2/F");
    tree2->Branch("nNh8_1",   &out2_nNh8_1,   "nNh8_1/F");
    tree2->Branch("nNh8_2",   &out2_nNh8_2,   "nNh8_2/F");
    tree2->Branch("nPho8_1",  &out2_nPho8_1,  "nPho8_1/F");
    tree2->Branch("nPho8_2",  &out2_nPho8_2,  "nPho8_2/F");
    tree2->Branch("area8_1",  &out2_area8_1,  "area8_1/F");
    tree2->Branch("area8_2",  &out2_area8_2,  "area8_2/F");
    tree2->Branch("chEmEF8_1",&out2_chEmEF8_1,"chEmEF8_1/F");
    tree2->Branch("chEmEF8_2",&out2_chEmEF8_2,"chEmEF8_2/F");
    tree2->Branch("chHEF8_1", &out2_chHEF8_1, "chHEF8_1/F");
    tree2->Branch("chHEF8_2", &out2_chHEF8_2, "chHEF8_2/F");
    tree2->Branch("hfEmEF8_1",&out2_hfEmEF8_1,"hfEmEF8_1/F");
    tree2->Branch("hfEmEF8_2",&out2_hfEmEF8_2,"hfEmEF8_2/F");
    tree2->Branch("hfHEF8_1", &out2_hfHEF8_1, "hfHEF8_1/F");
    tree2->Branch("hfHEF8_2", &out2_hfHEF8_2, "hfHEF8_2/F");
    tree2->Branch("neEmEF8_1",&out2_neEmEF8_1,"neEmEF8_1/F");
    tree2->Branch("neEmEF8_2",&out2_neEmEF8_2,"neEmEF8_2/F");
    tree2->Branch("neHEF8_1", &out2_neHEF8_1, "neHEF8_1/F");
    tree2->Branch("neHEF8_2", &out2_neHEF8_2, "neHEF8_2/F");
    tree2->Branch("muEF8_1",  &out2_muEF8_1,  "muEF8_1/F");
    tree2->Branch("muEF8_2",  &out2_muEF8_2,  "muEF8_2/F");
    tree2->Branch("n2b1_1",   &out2_n2b1_1,   "n2b1_1/F");
    tree2->Branch("n2b1_2",   &out2_n2b1_2,   "n2b1_2/F");
    tree2->Branch("n3b1_1",   &out2_n3b1_1,   "n3b1_1/F");
    tree2->Branch("n3b1_2",   &out2_n3b1_2,   "n3b1_2/F");
    tree2->Branch("tau21_1",   &out2_tau21_1,   "tau21_1/F");
    tree2->Branch("tau21_2",   &out2_tau21_2,   "tau21_2/F");
    tree2->Branch("tau32_1",   &out2_tau32_1,   "tau32_1/F");
    tree2->Branch("tau32_2",   &out2_tau32_2,   "tau32_2/F");
    tree2->Branch("WvsQCD_1", &out2_WvsQCD_1, "WvsQCD_1/F");
    tree2->Branch("WvsQCD_2", &out2_WvsQCD_2, "WvsQCD_2/F");
    // AK4_1,2
    tree2->Branch("pt4_1",    &out2_pt4_1,    "pt4_1/F");
    tree2->Branch("pt4_2",    &out2_pt4_2,    "pt4_2/F");
    tree2->Branch("pt4_3",    &out2_pt4_3,    "pt4_3/F");
    tree2->Branch("pt4_4",    &out2_pt4_4,    "pt4_4/F");
    tree2->Branch("eta4_1",   &out2_eta4_1,   "eta4_1/F");
    tree2->Branch("eta4_2",   &out2_eta4_2,   "eta4_2/F");
    tree2->Branch("eta4_3",   &out2_eta4_3,   "eta4_3/F");
    tree2->Branch("eta4_4",   &out2_eta4_4,   "eta4_4/F");
    tree2->Branch("mPF4_1",   &out2_mPF4_1,   "mPF4_1/F");
    tree2->Branch("mPF4_2",   &out2_mPF4_2,   "mPF4_2/F");
    tree2->Branch("mPF4_3",   &out2_mPF4_3,   "mPF4_3/F");
    tree2->Branch("mPF4_4",   &out2_mPF4_4,   "mPF4_4/F");
    tree2->Branch("nConst4_1",&out2_nConst4_1,"nConst4_1/F");
    tree2->Branch("nConst4_2",&out2_nConst4_2,"nConst4_2/F");
    tree2->Branch("nConst4_3",&out2_nConst4_3,"nConst4_3/F");
    tree2->Branch("nConst4_4",&out2_nConst4_4,"nConst4_4/F");
    tree2->Branch("nCh4_1",   &out2_nCh4_1,   "nCh4_1/F");
    tree2->Branch("nCh4_2",   &out2_nCh4_2,   "nCh4_2/F");
    tree2->Branch("nCh4_3",   &out2_nCh4_3,   "nCh4_3/F");
    tree2->Branch("nCh4_4",   &out2_nCh4_4,   "nCh4_4/F");
    tree2->Branch("nEle4_1",  &out2_nEle4_1,  "nEle4_1/F");
    tree2->Branch("nEle4_2",  &out2_nEle4_2,  "nEle4_2/F");
    tree2->Branch("nEle4_3",  &out2_nEle4_3,  "nEle4_3/F");
    tree2->Branch("nEle4_4",  &out2_nEle4_4,  "nEle4_4/F");
    tree2->Branch("nMu4_1",   &out2_nMu4_1,   "nMu4_1/F");
    tree2->Branch("nMu4_2",   &out2_nMu4_2,   "nMu4_2/F");
    tree2->Branch("nMu4_3",   &out2_nMu4_3,   "nMu4_3/F");
    tree2->Branch("nMu4_4",   &out2_nMu4_4,   "nMu4_4/F");
    tree2->Branch("nNh4_1",   &out2_nNh4_1,   "nNh4_1/F");
    tree2->Branch("nNh4_2",   &out2_nNh4_2,   "nNh4_2/F");
    tree2->Branch("nNh4_3",   &out2_nNh4_3,   "nNh4_3/F");
    tree2->Branch("nNh4_4",   &out2_nNh4_4,   "nNh4_4/F");
    tree2->Branch("nPho4_1",  &out2_nPho4_1,  "nPho4_1/F");
    tree2->Branch("nPho4_2",  &out2_nPho4_2,  "nPho4_2/F");
    tree2->Branch("nPho4_3",  &out2_nPho4_3,  "nPho4_3/F");
    tree2->Branch("nPho4_4",  &out2_nPho4_4,  "nPho4_4/F");
    tree2->Branch("area4_1",  &out2_area4_1,  "area4_1/F");
    tree2->Branch("area4_2",  &out2_area4_2,  "area4_2/F");
    tree2->Branch("area4_3",  &out2_area4_3,  "area4_3/F");
    tree2->Branch("area4_4",  &out2_area4_4,  "area4_4/F");
    tree2->Branch("chEmEF4_1",&out2_chEmEF4_1,"chEmEF4_1/F");
    tree2->Branch("chEmEF4_2",&out2_chEmEF4_2,"chEmEF4_2/F");
    tree2->Branch("chEmEF4_3",&out2_chEmEF4_3,"chEmEF4_3/F");
    tree2->Branch("chEmEF4_4",&out2_chEmEF4_4,"chEmEF4_4/F");
    tree2->Branch("chHEF4_1", &out2_chHEF4_1, "chHEF4_1/F");
    tree2->Branch("chHEF4_2", &out2_chHEF4_2, "chHEF4_2/F");
    tree2->Branch("chHEF4_3", &out2_chHEF4_3, "chHEF4_3/F");
    tree2->Branch("chHEF4_4", &out2_chHEF4_4, "chHEF4_4/F");
    tree2->Branch("hfEmEF4_1",&out2_hfEmEF4_1,"hfEmEF4_1/F");
    tree2->Branch("hfEmEF4_2",&out2_hfEmEF4_2,"hfEmEF4_2/F");
    tree2->Branch("hfEmEF4_3", &out2_hfEmEF4_3, "hfEmEF4_3/F");
    tree2->Branch("hfEmEF4_4", &out2_hfEmEF4_4, "hfEmEF4_4/F");
    tree2->Branch("hfHEF4_1", &out2_hfHEF4_1, "hfHEF4_1/F");
    tree2->Branch("hfHEF4_2", &out2_hfHEF4_2, "hfHEF4_2/F");
    tree2->Branch("hfHEF4_3", &out2_hfHEF4_3, "hfHEF4_3/F");
    tree2->Branch("hfHEF4_4", &out2_hfHEF4_4, "hfHEF4_4/F");
    tree2->Branch("neEmEF4_1",&out2_neEmEF4_1,"neEmEF4_1/F");
    tree2->Branch("neEmEF4_2",&out2_neEmEF4_2,"neEmEF4_2/F");
    tree2->Branch("neEmEF4_3",&out2_neEmEF4_3,"neEmEF4_3/F");
    tree2->Branch("neEmEF4_4",&out2_neEmEF4_4,"neEmEF4_4/F");
    tree2->Branch("neHEF4_1", &out2_neHEF4_1, "neHEF4_1/F");
    tree2->Branch("neHEF4_2", &out2_neHEF4_2, "neHEF4_2/F");
    tree2->Branch("neHEF4_3", &out2_neHEF4_3, "neHEF4_3/F");
    tree2->Branch("neHEF4_4", &out2_neHEF4_4, "neHEF4_4/F");
    tree2->Branch("muEF4_1",  &out2_muEF4_1,  "muEF4_1/F");
    tree2->Branch("muEF4_2",  &out2_muEF4_2,  "muEF4_2/F");
    tree2->Branch("muEF4_3",  &out2_muEF4_3,  "muEF4_3/F");
    tree2->Branch("muEF4_4",  &out2_muEF4_4,  "muEF4_4/F");
    // angles & masses
    tree2->Branch("dR8",        &out2_dR8,        "dR8/F");
    tree2->Branch("dPhi",       &out2_dPhi,       "dPhi/F");
    tree2->Branch("M8",         &out2_M8,         "M8/F");
    tree2->Branch("m1overM",    &out2_m1overM,    "m1overM/F");
    tree2->Branch("m2overM",    &out2_m2overM,    "m2overM/F");
    tree2->Branch("M84",        &out2_M84,        "M84/F");
    tree2->Branch("sphereM",    &out2_sphereM,    "sphereM/F");
    tree2->Branch("PT",         &out2_PT,         "PT/F");
    tree2->Branch("pt1overPT",  &out2_pt1overPT,  "pt1overPT/F");
    tree2->Branch("pt2overPT",  &out2_pt2overPT,  "pt2overPT/F");
    tree2->Branch("PToverptsum",&out2_PToverptsum,"PToverptsum/F");
    tree2->Branch("dR84_min",   &out2_dR84_min,   "dR84_min/F");
    tree2->Branch("dR44_min",   &out2_dR44_min,   "dR44_min/F");
    tree2->Branch("dR8L_min",   &out2_dR8L_min,   "dR8L_min/F");
    // leptons
    tree2->Branch("ptL_1",      &out2_ptL_1,      "ptL_1/F");
    tree2->Branch("ptL_2",      &out2_ptL_2,      "ptL_2/F");
    tree2->Branch("ptL_3",      &out2_ptL_3,      "ptL_3/F");
    tree2->Branch("etaL_1",     &out2_etaL_1,     "etaL_1/F");
    tree2->Branch("etaL_2",     &out2_etaL_2,     "etaL_2/F");
    tree2->Branch("etaL_3",     &out2_etaL_3,     "etaL_3/F");
    tree2->Branch("phiL_1",     &out2_phiL_1,     "phiL_1/F");
    tree2->Branch("phiL_2",     &out2_phiL_2,     "phiL_2/F");
    tree2->Branch("phiL_3",     &out2_phiL_3,     "phiL_3/F");
    tree2->Branch("isoEcalL_1", &out2_isoEcalL_1, "isoEcalL_1/F");
    tree2->Branch("isoEcalL_2", &out2_isoEcalL_2, "isoEcalL_2/F");
    tree2->Branch("isoEcalL_3", &out2_isoEcalL_3, "isoEcalL_3/F");
    tree2->Branch("isoHcalL_1", &out2_isoHcalL_1, "isoHcalL_1/F");
    tree2->Branch("isoHcalL_2", &out2_isoHcalL_2, "isoHcalL_2/F");
    tree2->Branch("isoHcalL_3", &out2_isoHcalL_3, "isoHcalL_3/F");

    // --- BRANCHES FOR tree3 (same names, different variables) ---
    tree3->Branch("isSignal", &out3_isSignal, "isSignal/O");
    tree3->Branch("type",     &out3_type,     "type/I");
    tree3->Branch("N_ak8",   &out3_Nak8,   "N_ak8/I");
    tree3->Branch("N_ak4",   &out3_Nak4,   "N_ak4/I");
    tree3->Branch("N_PV",    &out3_NPV,    "N_PV/I");
    tree3->Branch("N_e",     &out3_Ne,     "N_e/I");
    tree3->Branch("N_mu",    &out3_Nmu,    "N_mu/I");
    tree3->Branch("N_gamma", &out3_Npho,   "N_gamma/I");
    tree3->Branch("H_T",     &out3_HT,     "H_T/F");
    // AK8_1,2,3
    tree3->Branch("pt8_1",    &out3_pt8_1,    "pt8_1/F");
    tree3->Branch("pt8_2",    &out3_pt8_2,    "pt8_2/F");
    tree3->Branch("pt8_3",    &out3_pt8_3,    "pt8_3/F");
    tree3->Branch("eta8_1",   &out3_eta8_1,   "eta8_1/F");
    tree3->Branch("eta8_2",   &out3_eta8_2,   "eta8_2/F");
    tree3->Branch("eta8_3",   &out3_eta8_3,   "eta8_3/F");
    tree3->Branch("msd8_1",   &out3_msd8_1,   "msd8_1/F");
    tree3->Branch("msd8_2",   &out3_msd8_2,   "msd8_2/F");
    tree3->Branch("msd8_3",   &out3_msd8_3,   "msd8_3/F");
    tree3->Branch("mr8_1",    &out3_mr8_1,    "mr8_1/F");
    tree3->Branch("mr8_2",    &out3_mr8_2,    "mr8_2/F");
    tree3->Branch("mr8_3",    &out3_mr8_3,    "mr8_3/F");
    tree3->Branch("nConst8_1",&out3_nConst8_1,"nConst8_1/F");
    tree3->Branch("nConst8_2",&out3_nConst8_2,"nConst8_2/F");
    tree3->Branch("nConst8_3",&out3_nConst8_3,"nConst8_3/F");
    tree3->Branch("nCh8_1",   &out3_nCh8_1,   "nCh8_1/F");
    tree3->Branch("nCh8_2",   &out3_nCh8_2,   "nCh8_2/F");
    tree3->Branch("nCh8_3",   &out3_nCh8_3,   "nCh8_3/F");
    tree3->Branch("nEle8_1",  &out3_nEle8_1,  "nEle8_1/F");
    tree3->Branch("nEle8_2",  &out3_nEle8_2,  "nEle8_2/F");
    tree3->Branch("nEle8_3",  &out3_nEle8_3,  "nEle8_3/F");
    tree3->Branch("nMu8_1",   &out3_nMu8_1,   "nMu8_1/F");
    tree3->Branch("nMu8_2",   &out3_nMu8_2,   "nMu8_2/F");
    tree3->Branch("nMu8_3",   &out3_nMu8_3,   "nMu8_3/F");
    tree3->Branch("nNh8_1",   &out3_nNh8_1,   "nNh8_1/F");
    tree3->Branch("nNh8_2",   &out3_nNh8_2,   "nNh8_2/F");
    tree3->Branch("nNh8_3",   &out3_nNh8_3,   "nNh8_3/F");
    tree3->Branch("nPho8_1",  &out3_nPho8_1,  "nPho8_1/F");
    tree3->Branch("nPho8_2",  &out3_nPho8_2,  "nPho8_2/F");
    tree3->Branch("nPho8_3",  &out3_nPho8_3,  "nPho8_3/F");
    tree3->Branch("area8_1",  &out3_area8_1,  "area8_1/F");
    tree3->Branch("area8_2",  &out3_area8_2,  "area8_2/F");
    tree3->Branch("area8_3",  &out3_area8_3,  "area8_3/F");
    tree3->Branch("chEmEF8_1",&out3_chEmEF8_1,"chEmEF8_1/F");
    tree3->Branch("chEmEF8_2",&out3_chEmEF8_2,"chEmEF8_2/F");
    tree3->Branch("chEmEF8_3",&out3_chEmEF8_3,"chEmEF8_3/F");
    tree3->Branch("chHEF8_1", &out3_chHEF8_1, "chHEF8_1/F");
    tree3->Branch("chHEF8_2", &out3_chHEF8_2, "chHEF8_2/F");
    tree3->Branch("chHEF8_3", &out3_chHEF8_3, "chHEF8_3/F");
    tree3->Branch("hfEmEF8_1",&out3_hfEmEF8_1,"hfEmEF8_1/F");
    tree3->Branch("hfEmEF8_2",&out3_hfEmEF8_2,"hfEmEF8_2/F");
    tree3->Branch("hfEmEF8_3",&out3_hfEmEF8_3,"hfEmEF8_3/F");
    tree3->Branch("hfHEF8_1", &out3_hfHEF8_1, "hfHEF8_1/F");
    tree3->Branch("hfHEF8_2", &out3_hfHEF8_2, "hfHEF8_2/F");
    tree3->Branch("hfHEF8_3", &out3_hfHEF8_3, "hfHEF8_3/F");
    tree3->Branch("neEmEF8_1",&out3_neEmEF8_1,"neEmEF8_1/F");
    tree3->Branch("neEmEF8_2",&out3_neEmEF8_2,"neEmEF8_2/F");
    tree3->Branch("neEmEF8_3",&out3_neEmEF8_3,"neEmEF8_3/F");
    tree3->Branch("neHEF8_1", &out3_neHEF8_1, "neHEF8_1/F");
    tree3->Branch("neHEF8_2", &out3_neHEF8_2, "neHEF8_2/F");
    tree3->Branch("neHEF8_3", &out3_neHEF8_3, "neHEF8_3/F");
    tree3->Branch("muEF8_1",  &out3_muEF8_1,  "muEF8_1/F");
    tree3->Branch("muEF8_2",  &out3_muEF8_2,  "muEF8_2/F");
    tree3->Branch("muEF8_3",  &out3_muEF8_3,  "muEF8_3/F");
    tree3->Branch("n2b1_1",   &out3_n2b1_1,   "n2b1_1/F");
    tree3->Branch("n2b1_2",   &out3_n2b1_2,   "n2b1_2/F");
    tree3->Branch("n2b1_3",   &out3_n2b1_3,   "n2b1_3/F");
    tree3->Branch("n3b1_1",   &out3_n3b1_1,   "n3b1_1/F");
    tree3->Branch("n3b1_2",   &out3_n3b1_2,   "n3b1_2/F");
    tree3->Branch("n3b1_3",   &out3_n3b1_3,   "n3b1_3/F");
    tree3->Branch("tau21_1",   &out3_tau21_1,   "tau21_1/F");
    tree3->Branch("tau21_2",   &out3_tau21_2,   "tau21_2/F");
    tree3->Branch("tau21_3",   &out3_tau21_3,   "tau21_3/F");
    tree3->Branch("tau32_1",   &out3_tau32_1,   "tau32_1/F");
    tree3->Branch("tau32_2",   &out3_tau32_2,   "tau32_2/F");
    tree3->Branch("tau32_3",   &out3_tau32_3,   "tau32_3/F");
    tree3->Branch("WvsQCD_1", &out3_WvsQCD_1, "WvsQCD_1/F");
    tree3->Branch("WvsQCD_2", &out3_WvsQCD_2, "WvsQCD_2/F");
    tree3->Branch("WvsQCD_3", &out3_WvsQCD_3, "WvsQCD_3/F");
    // angles & masses
    tree3->Branch("M",           &out3_M,         "M/F");
    tree3->Branch("m1overM",     &out3_m1overM,    "m1overM/F");
    tree3->Branch("m2overM",     &out3_m2overM,    "m2overM/F");
    tree3->Branch("m3overM",     &out3_m3overM,    "m3overM/F");
    tree3->Branch("sphereM",     &out3_sphereM,    "sphereM/F");
    tree3->Branch("PT",          &out3_PT,         "PT/F");
    tree3->Branch("pt1overPT",   &out3_pt1overPT,  "pt1overPT/F");
    tree3->Branch("pt2overPT",   &out3_pt2overPT,  "pt2overPT/F");
    tree3->Branch("pt3overPT",   &out3_pt3overPT,  "pt3overPT/F");
    tree3->Branch("PToverptsum", &out3_PToverptsum,"PToverptsum/F");
    tree3->Branch("dR_min",      &out3_dR_min,     "dR_min/F");
    tree3->Branch("dR_max",      &out3_dR_max,     "dR_max/F");
    tree3->Branch("dPhi_min",    &out3_dPhi_min,   "dPhi_min/F");
    tree3->Branch("dPhi_max",    &out3_dPhi_max,   "dPhi_max/F");
    tree3->Branch("dRL_min",     &out3_dRL_min, "dRL_min/F");
    // leptons
    tree3->Branch("ptL_1",      &out3_ptL_1,      "ptL_1/F");
    tree3->Branch("ptL_2",      &out3_ptL_2,      "ptL_2/F");
    tree3->Branch("ptL_3",      &out3_ptL_3,      "ptL_3/F");
    tree3->Branch("etaL_1",     &out3_etaL_1,     "etaL_1/F");
    tree3->Branch("etaL_2",     &out3_etaL_2,     "etaL_2/F");
    tree3->Branch("etaL_3",     &out3_etaL_3,     "etaL_3/F");
    tree3->Branch("phiL_1",     &out3_phiL_1,     "phiL_1/F");
    tree3->Branch("phiL_2",     &out3_phiL_2,     "phiL_2/F");
    tree3->Branch("phiL_3",     &out3_phiL_3,     "phiL_3/F");
    tree3->Branch("isoEcalL_1", &out3_isoEcalL_1, "isoEcalL_1/F");
    tree3->Branch("isoEcalL_2", &out3_isoEcalL_2, "isoEcalL_2/F");
    tree3->Branch("isoEcalL_3", &out3_isoEcalL_3, "isoEcalL_3/F");
    tree3->Branch("isoHcalL_1", &out3_isoHcalL_1, "isoHcalL_1/F");
    tree3->Branch("isoHcalL_2", &out3_isoHcalL_2, "isoHcalL_2/F");
    tree3->Branch("isoHcalL_3", &out3_isoHcalL_3, "isoHcalL_3/F");

    // --- EVENT LOOP ---
    out2_isSignal = outputFileName.find("signal") != std::string::npos;
    out3_isSignal = out2_isSignal;
    if (type.find("qcd") != std::string::npos)
    {
        out2_type = 10;
    }
    else if (type.find("tt_had") != std::string::npos)
    {
        out2_type = 11;
    }
    else if (type.find("tt_semilep") != std::string::npos)
    {
        out2_type = 12;
    }
    else if (type.find("ww") != std::string::npos && outputFileName.find("signal") == std::string::npos)
    {
        out2_type = 13;
    }
    else if (type.find("wz") != std::string::npos && outputFileName.find("signal") == std::string::npos)
    {
        out2_type = 14;
    }
    else if (type.find("zz") != std::string::npos && outputFileName.find("signal") == std::string::npos)
    {
        out2_type = 15;
    }
    else if (type.find("www") != std::string::npos)
    {
        out2_type = 0;
    }
    else if (type.find("wwz") != std::string::npos)
    {
        out2_type = 1;
    }
    else if (type.find("wzz") != std::string::npos)
    {
        out2_type = 2;
    }
    else if (type.find("zzz") != std::string::npos)
    {
        out2_type = 3;
    }
    else if (type.find("zh") != std::string::npos)
    {
        out2_type = 4;
    }
    else if (type.find("wplush") != std::string::npos || type.find("wminush") != std::string::npos)
    {
        out2_type = 5;
    }
    else if (type.find("202") != std::string::npos)
    {
        out2_type = -2;
    }
    else
    {
        out2_type = -1; // unknown type
    }
    out3_type = out2_type;
    std::cout << "Processing file: " << inputFileName << std::endl;
    Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        
        if (i % 200000 == 0) std::cout << "Processing entry " << i << " / " << nEntries << std::endl; 
        
        tree->GetEntry(i);
        if (nFat8 < 1) continue;

        // build lepton array without vectors
        Lepton leptons[MAX_LEP];
        int nLep = 0;
        int nMuon = 0;
        for (int e = 0; e < nEle && e < MAX_ELE; ++e)
        {
            leptons[e] = { ele_pt[e], ele_eta[e], ele_phi[e], ele_ecalIso[e], ele_hcalIso[e] };
            nLep++;
        }
        for (int m = 0; m < nMuVtx && m < MAX_MU && nLep < MAX_LEP; ++m)
        {
            leptons[nLep] = { muV_pt[m], muV_eta[m], muV_phi[m], muV_ecalIso[m], muV_hcalIso[m] };
            nLep++;
            nMuon++;
        }
        for (int m = 0; m < nMuNoVtx && m < MAX_MU && nLep < MAX_LEP; ++m)
        {
            bool isDuplicate = false;
            for (int j = 0; j < nMuVtx && j < MAX_MU; ++j) {
                if (fabs(muNV_pt[m] - muV_pt[j]) < 0.5) {
                    isDuplicate = true;
                    break;
                }
            }
            if (isDuplicate) continue;
            leptons[nLep] = { muNV_pt[m], muNV_eta[m], muNV_phi[m], muNV_ecalIso[m], muNV_hcalIso[m] };
            nLep++;
            nMuon++;
        }

        // 2fat common
        out2_Nak8 = nFat8;
        out2_Nak4 = nJet4;
        out2_NPV  = nPV;
        out2_Ne   = nEle;
        out2_Nmu  = nMuon;
        out2_Npho = nPho;
        out2_HT   = 0;
        for (int j = 0; j < nFat8 && j < MAX_FAT8; ++j) out2_HT += fat_pt[j];
        // 3fat common
        out3_Nak8 = nFat8;
        out3_Nak4 = nJet4;
        out3_NPV  = nPV;
        out3_Ne   = nEle;
        out3_Nmu  = nMuon;
        out3_Npho = nPho;
        out3_HT   = out2_HT;

        // leptons
        auto fillL = [&](int idx, float &pt, float& eta, float& phi, float &ecal, float &hcal){
            if (idx < nLep && idx < MAX_LEP) {
                pt   = leptons[idx].pt;
                eta  = leptons[idx].eta;
                phi  = leptons[idx].phi;
                ecal = leptons[idx].ecalIso;
                hcal = leptons[idx].hcalIso;
            } else {
                pt = eta = phi = ecal = hcal = def;
            }
        };
        fillL(0, out2_ptL_1, out2_etaL_1, out2_phiL_1, out2_isoEcalL_1, out2_isoHcalL_1);
        fillL(1, out2_ptL_2, out2_etaL_2, out2_phiL_2, out2_isoEcalL_2, out2_isoHcalL_2);
        fillL(2, out2_ptL_3, out2_etaL_3, out2_phiL_3, out2_isoEcalL_3, out2_isoHcalL_3);
        fillL(0, out3_ptL_1, out3_etaL_1, out3_phiL_1, out3_isoEcalL_1, out3_isoHcalL_1);
        fillL(1, out3_ptL_2, out3_etaL_2, out3_phiL_2, out3_isoEcalL_2, out3_isoHcalL_2);
        fillL(2, out3_ptL_3, out3_etaL_3, out3_phiL_3, out3_isoEcalL_3, out3_isoHcalL_3);


        // 2fat case
        int nOK=0;
        for(int j = 0 ; j < nJet4; ++j)
        {
            bool ov = false;
            for(int k = 0 ;k < min(nFat8, 2); ++k)
            {
                TLorentzVector ak8, ak4;
                ak8.SetPtEtaPhiM(fat_pt[k], fat_eta[k], fat_phi[k], fat_mass[k]);
                ak4.SetPtEtaPhiM(pf_pt[j], pf_eta[j], pf_phi[j], pf_mass[j]);
                if (ak8.DeltaR(ak4) < 0.8) 
                {
                    ov = true;
                    break;
                }
            }
            if(!ov) ++nOK;
        }
        if (nFat8 == 2 && nOK >= 2) {
            // AK8
            out2_pt8_1      = fat_pt[0];
            out2_pt8_2      = fat_pt[1];
            out2_eta8_1     = fat_eta[0];
            out2_eta8_2     = fat_eta[1];
            out2_msd8_1     = fat_msoftdrop[0];
            out2_msd8_2     = fat_msoftdrop[1];
            out2_mr8_1      = fat_mass[0];
            out2_mr8_2      = fat_mass[1];
            out2_nConst8_1 = fat_nConst[0];
            out2_nConst8_2 = fat_nConst[1];
            out2_nCh8_1    = fat_nCh[0];
            out2_nCh8_2    = fat_nCh[1];
            out2_nEle8_1  = fat_nEle[0];
            out2_nEle8_2  = fat_nEle[1];
            out2_nMu8_1   = fat_nMu[0];
            out2_nMu8_2   = fat_nMu[1];
            out2_nNh8_1   = fat_nNh[0];
            out2_nNh8_2   = fat_nNh[1];
            out2_nPho8_1  = fat_nPho[0];
            out2_nPho8_2  = fat_nPho[1];
            out2_area8_1  = fat_area[0];
            out2_area8_2  = fat_area[1];
            out2_chEmEF8_1= fat_chEmEF[0];
            out2_chEmEF8_2= fat_chEmEF[1];
            out2_chHEF8_1 = fat_chHEF[0];
            out2_chHEF8_2 = fat_chHEF[1];
            out2_hfEmEF8_1= fat_hfEmEF[0];
            out2_hfEmEF8_2= fat_hfEmEF[1];
            out2_hfHEF8_1 = fat_hfHEF[0];
            out2_hfHEF8_2 = fat_hfHEF[1];
            out2_neEmEF8_1= fat_neEmEF[0];
            out2_neEmEF8_2= fat_neEmEF[1];
            out2_neHEF8_1 = fat_neHEF[0];
            out2_neHEF8_2 = fat_neHEF[1];
            out2_muEF8_1  = fat_muEF[0];
            out2_muEF8_2  = fat_muEF[1];
            out2_n2b1_1   = fat_n2b1[0];
            out2_n2b1_2   = fat_n2b1[1];
            out2_n3b1_1   = fat_n3b1[0];
            out2_n3b1_2   = fat_n3b1[1];
            out2_tau21_1   = fat_tau1[0] == 0 ? 0 : fat_tau2[0] / fat_tau1[0];
            out2_tau21_2   = fat_tau1[1] == 0 ? 0 : fat_tau2[1] / fat_tau1[1];
            out2_tau32_1   = fat_tau2[0] == 0 ? 0 : fat_tau3[0] / fat_tau2[0];
            out2_tau32_2   = fat_tau2[1] == 0 ? 0 : fat_tau3[1] / fat_tau2[1];
            out2_WvsQCD_1 = (fat_Xud[0] + fat_Xcs[0] + fat_QCD[0]) == 0 ? 0 : (fat_Xud[0] + fat_Xcs[0]) / (fat_Xud[0] + fat_Xcs[0] + fat_QCD[0]);
            out2_WvsQCD_2 = (fat_Xud[1] + fat_Xcs[1] + fat_QCD[1]) == 0 ? 0 : (fat_Xud[1] + fat_Xcs[1]) / (fat_Xud[1] + fat_Xcs[1] + fat_QCD[1]);

            // AK4
            out2_pt4_1    = pf_pt[0];
            out2_pt4_2    = pf_pt[1];
            out2_eta4_1   = pf_eta[0];
            out2_eta4_2   = pf_eta[1];
            out2_mPF4_1   = pf_mass[0];
            out2_mPF4_2   = pf_mass[1];
            out2_nConst4_1= pf_nConst[0];
            out2_nConst4_2= pf_nConst[1];
            out2_nCh4_1   = pf_nCh[0];
            out2_nCh4_2   = pf_nCh[1];
            out2_nEle4_1  = pf_nEle[0];
            out2_nEle4_2  = pf_nEle[1];
            out2_nMu4_1   = pf_nMu[0];
            out2_nMu4_2   = pf_nMu[1];
            out2_nNh4_1   = pf_nNh[0];
            out2_nNh4_2   = pf_nNh[1];
            out2_nPho4_1  = pf_nPho[0];
            out2_nPho4_2  = pf_nPho[1];
            out2_area4_1  = pf_area[0];
            out2_area4_2  = pf_area[1];
            out2_chEmEF4_1= pf_chEmEF[0];
            out2_chEmEF4_2= pf_chEmEF[1];
            out2_chHEF4_1 = pf_chHEF[0];
            out2_chHEF4_2 = pf_chHEF[1];
            out2_hfEmEF4_1= pf_hfEmEF[0];
            out2_hfEmEF4_2= pf_hfEmEF[1];
            out2_hfHEF4_1 = pf_hfHEF[0];
            out2_hfHEF4_2 = pf_hfHEF[1];
            out2_neEmEF4_1= pf_neEmEF[0];
            out2_neEmEF4_2= pf_neEmEF[1];
            out2_neHEF4_1 = pf_neHEF[0];
            out2_neHEF4_2 = pf_neHEF[1];
            out2_muEF4_1  = pf_muEF[0];
            out2_muEF4_2  = pf_muEF[1];

            if (nJet4 >= 4)
            {
                out2_pt4_3    = pf_pt[2];
                out2_pt4_4    = pf_pt[3];
                out2_eta4_3   = pf_eta[2];
                out2_eta4_4   = pf_eta[3];
                out2_mPF4_3   = pf_mass[2];
                out2_mPF4_4   = pf_mass[3];
                out2_nConst4_3= pf_nConst[2];
                out2_nConst4_4= pf_nConst[3];
                out2_nCh4_3   = pf_nCh[2];
                out2_nCh4_4   = pf_nCh[3];
                out2_nEle4_3  = pf_nEle[2];
                out2_nEle4_4  = pf_nEle[3];
                out2_nMu4_3   = pf_nMu[2];
                out2_nMu4_4   = pf_nMu[3];
                out2_nNh4_3   = pf_nNh[2];
                out2_nNh4_4   = pf_nNh[3];
                out2_nPho4_3  = pf_nPho[2];
                out2_nPho4_4  = pf_nPho[3];
                out2_area4_3  = pf_area[2];
                out2_area4_4  = pf_area[3];
                out2_chEmEF4_3= pf_chEmEF[2];
                out2_chEmEF4_4= pf_chEmEF[3];
                out2_chHEF4_3 = pf_chHEF[2];
                out2_chHEF4_4 = pf_chHEF[3];
                out2_hfEmEF4_3= pf_hfEmEF[2];
                out2_hfEmEF4_4= pf_hfEmEF[3];
                out2_hfHEF4_3 = pf_hfHEF[2];
                out2_hfHEF4_4 = pf_hfHEF[3];
                out2_neEmEF4_3= pf_neEmEF[2];
                out2_neEmEF4_4= pf_neEmEF[3];
                out2_neHEF4_3 = pf_neHEF[2];
                out2_neHEF4_4 = pf_neHEF[3];
                out2_muEF4_3  = pf_muEF[2];
                out2_muEF4_4  = pf_muEF[3];
            } 
            else if(nJet4 == 3) 
            {
                out2_pt4_3    = pf_pt[2];
                out2_pt4_4    = def;
                out2_eta4_3   = pf_eta[2];
                out2_eta4_4   = def;
                out2_mPF4_3   = pf_mass[2];
                out2_mPF4_4   = def;
                out2_nConst4_3= pf_nConst[2];
                out2_nConst4_4= def;
                out2_nCh4_3   = pf_nCh[2];
                out2_nEle4_3  = pf_nEle[2];
                out2_nCh4_4   = def;
                out2_nMu4_3   = pf_nMu[2];
                out2_nMu4_4   = def;
                out2_nNh4_3   = pf_nNh[2];
                out2_nNh4_4   = def;
                out2_nPho4_3  = pf_nPho[2];
                out2_nPho4_4  = def;
                out2_area4_3  = pf_area[2];
                out2_area4_4  = def;
                out2_chEmEF4_3= pf_chEmEF[2];
                out2_chEmEF4_4= def;
                out2_chHEF4_3 = pf_chHEF[2];
                out2_chHEF4_4 = def;
                out2_hfEmEF4_3= pf_hfEmEF[2];
                out2_hfEmEF4_4= def;
                out2_hfHEF4_3 = pf_hfHEF[2];
                out2_hfHEF4_4 = def;
                out2_neEmEF4_3= pf_neEmEF[2];
                out2_neEmEF4_4= def;
                out2_neHEF4_3 = pf_neHEF[2];
                out2_neHEF4_4 = def;
                out2_muEF4_3  = pf_muEF[2];
                out2_muEF4_4  = def;
            }
            else 
            {
                out2_pt4_3    = def;
                out2_pt4_4    = def;
                out2_eta4_3   = def;
                out2_eta4_4   = def;
                out2_mPF4_3   = def;
                out2_mPF4_4   = def;
                out2_nConst4_3= def;
                out2_nConst4_4= def;
                out2_nCh4_3   = def;
                out2_nCh4_4   = def;
                out2_nEle4_3  = def;
                out2_nEle4_4  = def;
                out2_nMu4_3   = def;
                out2_nMu4_4   = def;
                out2_nNh4_3   = def;
                out2_nNh4_4   = def;
                out2_nPho4_3  = def;
                out2_nPho4_4  = def;
                out2_area4_3  = def;
                out2_area4_4  = def;
                out2_chEmEF4_3= def;
                out2_chEmEF4_4= def;
                out2_chHEF4_3 = def;
                out2_chHEF4_4 = def;
                out2_hfEmEF4_3= def;
                out2_hfEmEF4_4= def;
                out2_hfHEF4_3 = def;
                out2_hfHEF4_4 = def;
                out2_neEmEF4_3= def;
                out2_neEmEF4_4= def;
                out2_neHEF4_3 = def;
                out2_neHEF4_4 = def;
                out2_muEF4_3  = def;
                out2_muEF4_4  = def;
            }

            // angular & masses for AK8[0,1]
            TLorentzVector v8_0, v8_1, v4_0, v4_1;
            v8_0.SetPtEtaPhiM(fat_pt[0], fat_eta[0], fat_phi[0], fat_mass[0]);
            v8_1.SetPtEtaPhiM(fat_pt[1], fat_eta[1], fat_phi[1], fat_mass[1]);
            v4_0.SetPtEtaPhiM(pf_pt[0], pf_eta[0], pf_phi[0], pf_mass[0]);
            v4_1.SetPtEtaPhiM(pf_pt[1], pf_eta[1], pf_phi[1], pf_mass[1]);

            out2_dR8 = v8_0.DeltaR(v8_1);
            out2_dPhi = v8_0.DeltaPhi(v8_1);
            out2_M8 = (v8_0 + v8_1).M();
            out2_m1overM = v8_0.M() / out2_M8;
            out2_m2overM = v8_1.M() / out2_M8;
            out2_M84 = (v8_0 + v8_1 + v4_0 + v4_1).M();
            out2_PT = (v8_0 + v8_1 + v4_0 + v4_1).Pt();
            out2_pt1overPT = v8_0.Pt() / out2_PT;
            out2_pt2overPT = v8_1.Pt() / out2_PT;
            out2_PToverptsum = out2_PT / (v8_0.Pt() + v8_1.Pt() + v4_0.Pt() + v4_1.Pt());
            out2_sphereM = sqrt((fat_msoftdrop[0] - 85)*(fat_msoftdrop[0] - 85)  + (fat_msoftdrop[1] - 85)*(fat_msoftdrop[1] - 85));
            // Loop over AK4 jets and leptons to find min dR with AK8
            out2_dR84_min = 9999;
            out2_dR44_min = 9999;
            out2_dR8L_min = 9999;
            for (int j = 0; j < nJet4 && j < MAX_PF; ++j) {
                TLorentzVector ak4;
                ak4.SetPtEtaPhiM(pf_pt[j], pf_eta[j], pf_phi[j], pf_mass[j]);
                float dR = std::min(v8_0.DeltaR(ak4), v8_1.DeltaR(ak4));
                if (dR < out2_dR84_min) out2_dR84_min = dR;
                for (int k = j + 1; k < nJet4 && k < MAX_PF; ++k) {
                    TLorentzVector ak4_2;
                    ak4_2.SetPtEtaPhiM(pf_pt[k], pf_eta[k], pf_phi[k], pf_mass[k]);
                    float dR2 = ak4.DeltaR(ak4_2);
                    if (dR2 < out2_dR44_min) out2_dR44_min = dR2;
                }
            }

            for (int j = 0; j < nLep && j < MAX_LEP; ++j) {
                TLorentzVector lep;
                lep.SetPtEtaPhiM(leptons[j].pt, leptons[j].eta, leptons[j].phi, 0);
                float dR = std::min(v8_0.DeltaR(lep), v8_1.DeltaR(lep));
                if (dR < out2_dR8L_min) out2_dR8L_min = dR;
            }

            if (out2_dR84_min == 9999) out2_dR84_min = -999;
            if (out2_dR44_min == 9999) out2_dR44_min = -999;
            if (out2_dR8L_min == 9999) out2_dR8L_min = -999;

            tree2->Fill();
        }

        // 3fat case
        if (nFat8 >= 3) {
            // AK8 1,2,3
            out3_pt8_1      = fat_pt[0];
            out3_pt8_2      = fat_pt[1];
            out3_pt8_3      = fat_pt[2];
            out3_eta8_1     = fat_eta[0];
            out3_eta8_2     = fat_eta[1];
            out3_eta8_3     = fat_eta[2];
            out3_msd8_1     = fat_msoftdrop[0];
            out3_msd8_2     = fat_msoftdrop[1];
            out3_msd8_3     = fat_msoftdrop[2];
            out3_mr8_1      = fat_mass[0];
            out3_mr8_2      = fat_mass[1];
            out3_mr8_3      = fat_mass[2];
            out3_nConst8_1 = fat_nConst[0];
            out3_nConst8_2 = fat_nConst[1];
            out3_nConst8_3 = fat_nConst[2];
            out3_nCh8_1    = fat_nCh[0];
            out3_nCh8_2    = fat_nCh[1];
            out3_nCh8_3    = fat_nCh[2];
            out3_nEle8_1  = fat_nEle[0];
            out3_nEle8_2  = fat_nEle[1];
            out3_nEle8_3  = fat_nEle[2];
            out3_nMu8_1   = fat_nMu[0];
            out3_nMu8_2   = fat_nMu[1];
            out3_nMu8_3   = fat_nMu[2];
            out3_nNh8_1   = fat_nNh[0];
            out3_nNh8_2   = fat_nNh[1];
            out3_nNh8_3   = fat_nNh[2];
            out3_nPho8_1  = fat_nPho[0];
            out3_nPho8_2  = fat_nPho[1];
            out3_nPho8_3  = fat_nPho[2];
            out3_area8_1  = fat_area[0];
            out3_area8_2  = fat_area[1];
            out3_area8_3  = fat_area[2];
            out3_chEmEF8_1= fat_chEmEF[0];
            out3_chEmEF8_2= fat_chEmEF[1];
            out3_chEmEF8_3= fat_chEmEF[2];
            out3_chHEF8_1 = fat_chHEF[0];
            out3_chHEF8_2 = fat_chHEF[1];
            out3_chHEF8_3 = fat_chHEF[2];
            out3_hfEmEF8_1= fat_hfEmEF[0];
            out3_hfEmEF8_2= fat_hfEmEF[1];
            out3_hfEmEF8_3= fat_hfEmEF[2];
            out3_hfHEF8_1 = fat_hfHEF[0];
            out3_hfHEF8_2 = fat_hfHEF[1];
            out3_hfHEF8_3 = fat_hfHEF[2];
            out3_neEmEF8_1= fat_neEmEF[0];
            out3_neEmEF8_2= fat_neEmEF[1];
            out3_neEmEF8_3= fat_neEmEF[2];
            out3_neHEF8_1 = fat_neHEF[0];
            out3_neHEF8_2 = fat_neHEF[1];
            out3_neHEF8_3 = fat_neHEF[2];
            out3_muEF8_1  = fat_muEF[0];
            out3_muEF8_2  = fat_muEF[1];
            out3_muEF8_3  = fat_muEF[2];
            out3_n2b1_1   = fat_n2b1[0];
            out3_n2b1_2   = fat_n2b1[1];
            out3_n2b1_3   = fat_n2b1[2];
            out3_n3b1_1   = fat_n3b1[0];
            out3_n3b1_2   = fat_n3b1[1];
            out3_n3b1_3   = fat_n3b1[2];
            out3_tau21_1   = fat_tau1[0] == 0 ? 0 : fat_tau2[0] / fat_tau1[0];
            out3_tau21_2   = fat_tau1[1] == 0 ? 0 : fat_tau2[1] / fat_tau1[1];
            out3_tau21_3   = fat_tau1[2] == 0 ? 0 : fat_tau2[2] / fat_tau1[2];
            out3_tau32_1   = fat_tau2[0] == 0 ? 0 : fat_tau3[0] / fat_tau2[0];
            out3_tau32_2   = fat_tau2[1] == 0 ? 0 : fat_tau3[1] / fat_tau2[1];
            out3_tau32_3   = fat_tau2[2] == 0 ? 0 : fat_tau3[2] / fat_tau2[2];
            out3_WvsQCD_1 = (fat_Xud[0] + fat_Xcs[0] + fat_QCD[0]) == 0 ? 0 : (fat_Xud[0] + fat_Xcs[0]) / (fat_Xud[0] + fat_Xcs[0] + fat_QCD[0]);
            out3_WvsQCD_2 = (fat_Xud[1] + fat_Xcs[1] + fat_QCD[1]) == 0 ? 0 : (fat_Xud[1] + fat_Xcs[1]) / (fat_Xud[1] + fat_Xcs[1] + fat_QCD[1]);
            out3_WvsQCD_3 = (fat_Xud[2] + fat_Xcs[2] + fat_QCD[2]) == 0 ? 0 : (fat_Xud[2] + fat_Xcs[2]) / (fat_Xud[2] + fat_Xcs[2] + fat_QCD[2]);

            // angular & masses on AK8[0,1]
            TLorentzVector v8_0, v8_1, v8_2;
            v8_0.SetPtEtaPhiM(fat_pt[0], fat_eta[0], fat_phi[0], fat_mass[0]);
            v8_1.SetPtEtaPhiM(fat_pt[1], fat_eta[1], fat_phi[1], fat_mass[1]);
            v8_2.SetPtEtaPhiM(fat_pt[2], fat_eta[2], fat_phi[2], fat_mass[2]);

            out3_M = (v8_0 + v8_1 + v8_2).M();
            out3_m1overM = v8_0.M() / out3_M;
            out3_m2overM = v8_1.M() / out3_M;
            out3_m3overM = v8_2.M() / out3_M;
            out3_PT = (v8_0 + v8_1 + v8_2).Pt();
            out3_pt1overPT = v8_0.Pt() / out3_PT;
            out3_pt2overPT = v8_1.Pt() / out3_PT;
            out3_pt3overPT = v8_2.Pt() / out3_PT;
            out3_PToverptsum = out3_PT / (v8_0.Pt() + v8_1.Pt() + v8_2.Pt());
            out3_sphereM = sqrt((fat_msoftdrop[0] - 85)*(fat_msoftdrop[0] - 85)  + (fat_msoftdrop[1] - 85)*(fat_msoftdrop[1] - 85) + (fat_msoftdrop[2] - 85)*(fat_msoftdrop[2] - 85));
            
            out3_dR_min = std::min(std::min(v8_0.DeltaR(v8_1), v8_0.DeltaR(v8_2)), v8_1.DeltaR(v8_2));
            out3_dPhi_min = std::min(std::min(v8_0.DeltaPhi(v8_1), v8_0.DeltaPhi(v8_2)), v8_1.DeltaPhi(v8_2));
            out3_dR_max = std::max(std::max(v8_0.DeltaR(v8_1), v8_0.DeltaR(v8_2)), v8_1.DeltaR(v8_2));
            out3_dPhi_max = std::max(std::max(v8_0.DeltaPhi(v8_1), v8_0.DeltaPhi(v8_2)), v8_1.DeltaPhi(v8_2));

            out3_dRL_min = 9999;
            
            for (int j = 0; j < nLep && j < MAX_LEP; ++j) {
                TLorentzVector lep;
                lep.SetPtEtaPhiM(leptons[j].pt, leptons[j].eta, leptons[j].phi, 0);
                float dR = std::min(std::min(v8_0.DeltaR(lep), v8_1.DeltaR(lep)), v8_2.DeltaR(lep));
                if (dR < out3_dRL_min) out3_dRL_min = dR;
            }

            if (out3_dRL_min == 9999) out3_dRL_min = -999;

            tree3->Fill();
        }
    }

    // Write
    outFile->cd();
    tree2->Write();
    tree3->Write();
    outFile->Close();
    inFile->Close();
    std::cout << "Finished processing file: " << inputFileName << std::endl;
}
