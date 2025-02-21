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

int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " --input <input_file> --output <output_file>" << std::endl;
        return 1;
    }

    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) { // Ensure there's a value after the flag
            args[argv[i]] = argv[i + 1];
        }
    }

    if (args.find("--input") == args.end() || args.find("--output") == args.end()) {
        std::cerr << "Error: Both --input and --output arguments are required." << std::endl;
        return 1;
    }

    std::string inputFileName;
    if (args.find("--input") == args.end())
    {
        inputFileName = "breit_particles.root";
        std::cout<<"No input argument found. Input file is "<<inputFileName<<std::endl;
    }
    else
    {
        inputFileName = args["--input"];
        std::cout<<"Input file is "<<inputFileName<<std::endl;
    }

    std::string outputFileName;
    if (args.find("--output") == args.end())
    {
        outputFileName = "centuaro_jets.root";
        std::cout<<"No output argument found. Output file is "<<outputFileName<<std::endl;
    }
    else
    {
        outputFileName = args["--output"];
        std::cout<<"Output file is "<<outputFileName<<std::endl;
    }

    Double_t jet_radius;
    if (args.find("--jet_radius") == args.end())
    {
        jet_radius = 0.8;
        std::cout<<"No jet radius argument found. Jet radius set to "<<jet_radius<<std::endl;
    }
    else
    {
        jet_radius = std::stod(args["--jet_radius"]);
        std::cout<<" Jet radius set to "<<jet_radius<<std::endl;
    }


    TFile *breit_file = TFile::Open(inputFileName.c_str());

    TTreeReader particle_tree("particles", breit_file);
    TTreeReaderArray<Double_t> px(particle_tree, "px");
    TTreeReaderArray<Double_t> py(particle_tree, "py");
    TTreeReaderArray<Double_t> pz(particle_tree, "pz");
    TTreeReaderArray<Double_t> energy(particle_tree, "energy");

    fastjet::contrib::CentauroPlugin *centauro_plugin = new fastjet::contrib::CentauroPlugin(jet_radius);
    
    std::vector<Double_t> jet_eta;
    std::vector<Double_t> jet_px;
    std::vector<Double_t> jet_py;
    std::vector<Double_t> jet_pz;
    std::vector<Double_t> jet_E;
    std::vector<Double_t> jet_phi;
    std::vector<Double_t> jet_pt;
    

    TFile* file = TFile::Open(outputFileName.c_str(), "RECREATE");
    TTree* tree = new TTree("jets", "");

    tree->Branch("eta", &jet_eta);
    tree->Branch("px", &jet_px);
    tree->Branch("py", &jet_py);
    tree->Branch("pz", &jet_pz);
    tree->Branch("E", &jet_E);
    tree->Branch("pt", &jet_pt);
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
        // std::vector<fastjet::PseudoJet> jets;
        // for(auto& jet : all_jets)
        // {
            // jets.push_back(jet);
            // std::vector<fastjet::PseudoJet> constituents = jet.constituents();

            // if(constituents.size() > 0)
            // {
            //     jets.push_back(jet);
            // }
        // }

        std::vector<fastjet::PseudoJet> sortedJets = sorted_by_pt(jets);
        jet_eta.clear();
        jet_px.clear();
        jet_py.clear();
        jet_pz.clear();
        jet_E.clear();
        jet_pt.clear();
        jet_phi.clear();
        
        for (unsigned int i=0; i<sortedJets.size(); i++){
            const fastjet::PseudoJet &jet = sortedJets[i];
            jet_eta.push_back(jet.pseudorapidity());
            jet_px.push_back(jet.px());
            jet_py.push_back(jet.py());
            jet_pz.push_back(jet.pz());
            jet_E.push_back(jet.E());
            jet_pt.push_back(-1*jet.pt());
            jet_phi.push_back(jet.phi());
        }
        tree->Fill();
    }
    tree->Write();
    file->Close();
}