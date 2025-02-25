#include <iostream>
#include <vector>

#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/Selector.hh"
#include "fastjet/contrib/Centauro.hh"
#include "fastjet/EECambridgePlugin.hh"

#include "H5Cpp.h"

using namespace H5;

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

    double jet_radius;
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


    H5File file(inputFileName, H5F_ACC_RDONLY);

    DataSet dataset = file.openDataSet("breit_particles");
    fastjet::contrib::CentauroPlugin *centauro_plugin = new fastjet::contrib::CentauroPlugin(jet_radius);
    
    // Get dataspace and dimensions
    DataSpace dataspace = dataset.getSpace();
    hsize_t dims[3];  // Expecting a 3D dataset
    dataspace.getSimpleExtentDims(dims, nullptr);

    hsize_t num_events = dims[0];
    hsize_t num_particles = dims[1];


    // Vector to store the number of jets per event
    std::vector<size_t> jet_counts(num_events, 0);

    // Read each event and cluster jets
    hsize_t offset[3] = {0, 0, 0};
    hsize_t count[3] = {1, num_particles, 4};
    DataSpace memspace(2, &dims[1]);

    std::vector<std::vector<std::vector<double>>> jet_data(num_events); // 3D structure

    for (hsize_t i = 0; i < num_events; i++) {
        offset[0] = i;
        dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);

        // Read event particles
        std::vector<double> event_data(num_particles * 4);
        dataset.read(event_data.data(), PredType::NATIVE_DOUBLE, memspace, dataspace);

        // Convert to PseudoJets
        std::vector<fastjet::PseudoJet> particle_vector;
        for (size_t j = 0; j < num_particles; j++) {
            double px = event_data[j * 4 + 0];
            double py = event_data[j * 4 + 1];
            double pz = event_data[j * 4 + 2];
            double E = event_data[j * 4 + 3];
            if (E>0)
            {
                fastjet::PseudoJet particle(
                    px,
                    py,
                    pz,
                    E
                );
                particle_vector.push_back(particle);
            }
            
        }

        // Perform clustering
        fastjet::JetDefinition jet_def(centauro_plugin);
        fastjet::ClusterSequence clust_seq(particle_vector, jet_def);

        std::vector<fastjet::PseudoJet> jets = sorted_by_pt(clust_seq.inclusive_jets(0));

        // Store jet attributes
        std::vector<std::vector<double>> event_jets;
        for (const auto &jet : jets) {
            event_jets.push_back({
                jet.pt(), jet.eta(), jet.phi(), jet.E(),
                jet.px(), jet.py(), jet.pz()
            });
        }

        jet_counts[i] = event_jets.size();
        jet_data[i] = event_jets;
    }

    // Find the max number of jets across all events
    hsize_t max_jets = 0;
    for (size_t i = 0; i < num_events; i++) {
        max_jets = std::max(max_jets, static_cast<hsize_t>(jet_counts[i]));
    }

    // Prepare zero-padded 3D array for HDF5
    std::vector<double> flat_data(num_events * max_jets * 7, 0.0);
    for (size_t i = 0; i < num_events; i++) {
        for (size_t j = 0; j < jet_data[i].size(); j++) {
            for (size_t k = 0; k < 7; k++) {
                flat_data[i * max_jets * 7 + j * 7 + k] = jet_data[i][j][k];
            }
        }
    }

    H5File output_file(outputFileName, H5F_ACC_TRUNC);

    hsize_t output_dims[3] = {num_events, max_jets, 7};
    DataSpace output_dataspace(3, output_dims);

    DataSet output_dataset = output_file.createDataSet(
        "jets", PredType::NATIVE_DOUBLE, output_dataspace
    );

    // Write zero-padded 3D array to HDF5
    output_dataset.write(flat_data.data(), PredType::NATIVE_DOUBLE);
    
}