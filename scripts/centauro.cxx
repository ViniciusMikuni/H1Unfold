#include <iostream>
#include <vector>

#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/Selector.hh"
#include "fastjet/contrib/Centauro.hh"
#include "fastjet/EECambridgePlugin.hh"

#include "TH1D.h"
#include "TFile.h"
#include "TMath.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include "TCanvas.h"

int main()
{
    TFile *breit_file = TFile::Open("./breit_particles.root");

    TTreeReader particle_tree("particles", breit_file);
    TTreeReaderArray<Double_t> px(particle_tree, "px");
    TTreeReaderArray<Double_t> py(particle_tree, "py");
    TTreeReaderArray<Double_t> pz(particle_tree, "pz");
    TTreeReaderArray<Double_t> energy(particle_tree, "energy");

    TH1D *h_mult = new TH1D("h_mult", "", 50, 0, 50);
    TH1D *h_eta = new TH1D("h_eta", "", 50, -5, 5);
    TH1D *h_pt = new TH1D("h_pt", "", 50, 0, 20);
    TH1D *h_phi = new TH1D("h_phi", "", 50, 0, 6.5);
    TH1D *h_z = new TH1D("h_z", "", 50, 0, 5);

    fastjet::contrib::CentauroPlugin *centauro_plugin = new fastjet::contrib::CentauroPlugin(.8);
    
    std::vector<Double_t> jet_eta;
    std::vector<Double_t> jet_px;
    std::vector<Double_t> jet_py;
    std::vector<Double_t> jet_pz;
    std::vector<Double_t> jet_E;
    std::vector<Double_t> jet_phi;
    std::vector<Double_t> jet_pt;
    

    TFile* file = TFile::Open("centauro_jets.root", "RECREATE");
    TTree* tree = new TTree("jets", "");

    tree->Branch("eta", &jet_eta);
    tree->Branch("px", &jet_px);
    tree->Branch("py", &jet_py);
    tree->Branch("pz", &jet_pz);
    tree->Branch("E", &jet_E);
    tree->Branch("pT", &jet_pt);
    tree->Branch("phi", &jet_phi);

    while (particle_tree.Next()) 
    {
        std::vector<fastjet::PseudoJet> particle_vector;
        for (int i = 0; i < px.GetSize(); i++)
        {
            fastjet::PseudoJet particle(px[i], py[i], pz[i], energy[i]);
            particle_vector.push_back(particle);
        }
        
        fastjet::JetDefinition jet_def(centauro_plugin);
        fastjet::ClusterSequence clust_seq(particle_vector, jet_def);

        std::vector<fastjet::PseudoJet> jets = clust_seq.inclusive_jets(0);

        std::vector<fastjet::PseudoJet> sortedJets = sorted_by_E(jets);
        jet_eta.clear();
        jet_px.clear();
        jet_py.clear();
        jet_pz.clear();
        jet_E.clear();
        jet_pt.clear();
        jet_phi.clear();
        
        h_mult->Fill(sortedJets.size());
        for (unsigned int i=0; i<sortedJets.size(); i++){
            const fastjet::PseudoJet &jet = sortedJets[i];
            jet_eta.push_back(jet.pseudorapidity());
            jet_px.push_back(jet.px());
            jet_py.push_back(jet.py());
            jet_pz.push_back(jet.pz());
            jet_E.push_back(jet.E());
            jet_pt.push_back(jet.pt());
            jet_phi.push_back(jet.phi());

            h_eta->Fill(jet.pseudorapidity());
            h_pt->Fill(jet.pt());
            h_phi->Fill(jet.phi_02pi());
        }
        tree->Fill();
    }

    tree->Write();
    file->Close();
    
    TCanvas *c_mult = new TCanvas();
    h_mult->Draw();
    h_mult->GetXaxis()->SetTitle("Num. jets in event");
    c_mult->SaveAs("mult.pdf");

    TCanvas *c_eta = new TCanvas();
    h_eta->Draw();
    h_eta->GetXaxis()->SetTitle("Jet pseudorapidity");
    c_eta->SaveAs("eta.pdf");

    TCanvas *c_pt = new TCanvas();
    h_pt->Draw();
    h_pt->GetXaxis()->SetTitle("Jet pT (GeV)");
    c_pt->SaveAs("pt.pdf");

    TCanvas *c_phi = new TCanvas();
    h_phi->Draw();
    h_phi->GetXaxis()->SetTitle("Jet phi (rad)");
    c_phi->SaveAs("phi.pdf");

    delete c_mult;
    delete c_eta;
    delete c_pt;
    delete c_phi;
    delete h_mult;
    delete h_eta;
    delete h_pt;
    delete h_phi;
}