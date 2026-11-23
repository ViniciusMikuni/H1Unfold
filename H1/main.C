// (c) MPI 2020 for the benefit of H1

// stl includes
#include <iostream>

// ROOT includes
#include <TROOT.h>
#include <TBenchmark.h>

// H1 includes
#include "H1Steering/H1StdCmdLine.h"
#include "H1Steering/H1ErrorHandler.h"
#include "H1Steering/H1SteerManager.h"
#include "H1Analysis/H1AnalysisSteer.h"
#include "H1Skeleton/H1SteerTree.h"
#include "H1Skeleton/H1Tree.h"
#include "H1Calculator/H1Calculator.h"

// analysis class
#include "AnalysisEventShapes.h"

//TROOT EventShapes("EventShapes","Event shapes at low and high Q2");

// ___________________________________________________________________________ //
/// /////////////////////////////////////////////////////////////////////////////
///
///                              EventShapes
///               (c) 2020, MPI, D. Britzger, J. Hessler
///
///  EventShapes -f <steerfile> -o <outputfile> -n <maxevent> -c <chain> -s -g -y -r
///
/// /////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
   
   // Root options
   TH1::SetDefaultSumw2();
   gROOT->AddDirectory(false);
   TBenchmark* Mybench = new TBenchmark();
   Mybench->Start("Analysis time");
   H1ErrorHandler::Instance()->SetMaxCount(100); // set max count


   // parse the command line
   H1StdCmdLine opts;
   // ---- default options ---------------------- //
   // AddOption("help", 'h', fIsHelp);
   // AddOption("version", 'V', fIsVersion);
   // AddOption("verbose", 'v', fIsVerbose);
   // AddOption("file", 'f', fSteerFiles);
   // AddOption("output", 'o', fOutput);
   // AddOption("nevents", 'n', fNevents);
   // AddOption("Errors", 'E', fNerrors)
   // -------------------------------------------- //

   // ---- options for Event shapes
   // AddOption("chain",'c',cmd_chain);         -> select anaysis-chain (e.g. djangoh, data, etc...). Must match a name from the steer-file
   // AddOption("tree",'t',cmd_write_minitree); -> write mini-tree to file
   // AddOption("sys",'s',cmd_sys);             -> process events with a systematic shift
   // -------------------------------------------- //

   H1OptionString   cmd_chain("");
   opts.AddOption("chain",'c',cmd_chain);
   H1OptionBool     cmd_write_minitree(false);
   opts.AddOption("tree",'t',cmd_write_minitree);
   H1OptionInt      cmd_sys(-9999);
   opts.AddOption("sys",'s',cmd_sys);

   opts.Parse(&argc, argv);
   cout<<"Process chain:          " << cmd_chain<<endl;
   cout<<"Write mini tree:        " << cmd_write_minitree<<endl;
   cout<<"Apply systematic shift: " << cmd_sys<<endl;
   // -------------------------------------------- //


   // --- read main steering
   H1AnalysisSteer* AnaSteer = static_cast<H1AnalysisSteer*>
      (gH1SteerManager->GetSteer( H1AnalysisSteer::Class() , "EventShapes" ));
   if ( !AnaSteer ) {
      Error("main","Cannot open steering. Please pass steering file with flag -f <file.steer>.");
      exit(1);
   }
   //AnaSteer->Print();
   
   
   // --- define chain (e.g. "Django_8")
   TString chain = string(cmd_chain);
   if ( chain != "" ) 
      Info("main", "Taking chain from command-line. Chain: %s", chain.Data());
   else {
      if ( AnaSteer->GetChains()->GetEntries() != 1 ) { // check if only a single entry is given
         Error("main","Multiple chains specified in 'fChains', but processing of a single chain is only implemented.");
         exit(1);
      }
      chain = ((TObjString*)AnaSteer->GetChains()->At(0))->GetString();
      Info("main", "Taking chain from steering. Chain: %s", chain.Data());
   }
   //const TString      = gH1SteerManager->ReadString("Chain");    // -c <...> chain to be analyzed

   // --- read H1SteerTree
   cout<<"Reading H1SteerTree from steering file with name: "<<chain<<endl;
   H1SteerTree* SteerTree = (H1SteerTree*)gH1SteerManager->GetSteer( H1SteerTree::Class() , chain.Data() );
   if ( !SteerTree ) { Error("main","Cannot read H1SteerTree for chain '%s'.",chain.Data()); exit(1); }
   SteerTree->SetLoadHAT(true);   // default settings
   SteerTree->SetLoadMODS(true);  // default settings
   //SteerTree->Print();

   
   // --- Load HAT/MOD files
   cout << "Starting HAT selection:" << endl;
   H1Tree::Instance()->AddFilesFromSteer(SteerTree);
   H1Tree::Instance()->Open();          // this statement must be here!
   H1Tree::Instance()->SelectAll();     // ? from H1AnalysisChain.C
   H1Tree::Instance()->Reset();         // ? from H1AnalysisChain.C
   cout << "HAT selection finished." << endl;
   cout << "Start looping over the "
        << H1Tree::Instance()->GetEntries()
        << " selected events for runtype: "<< gH1Tree ->IsMC() << endl << endl;

   
   // --- initialize EventShape analysis object
   TFile file(opts.GetOutput(), "RECREATE"); // make TFile before makein TTree for minitree
   AnalysisEventShapes esanalysis(chain);
   esanalysis.SetSysShift(cmd_sys);
   // --- event loop
   int ievent = 0;
   while (H1Tree::Instance()->Next() ) {
      if ( ievent%AnaSteer->GetInterval() == 0 ) 
         cout<<"Processing event: "<< ievent <<endl;

      // --- initial settings for base class and analysis class
      if ( ievent == 0 ) { 
         H1Calculator::Instance();
         //H1Tree::Instance()->Reset();
         esanalysis.DoBaseInitialSettings();
         esanalysis.DoInitialSettings();
         if ( cmd_write_minitree ) {
            Info("main","Preparing to write out minitree.");
            file.mkdir(esanalysis.GetChainName().Data())->cd();
            esanalysis.InitMiniTree();
         }
      }

      // --- reset at beginning of event loop
      esanalysis.DoBaseReset();
      esanalysis.DoReset();


      // --- evaluate basic cuts (base class)
      bool GenBaseCuts = esanalysis.DoBasicCutsGen();
      GenBaseCuts &=     esanalysis.DoAnalysisCutsGen();
      if ( !GenBaseCuts ) continue;
      bool RecBaseCuts = esanalysis.DoBasicCutsRec();
      RecBaseCuts     &= esanalysis.DoAnalysisCutsRec();


      // fill cross-section-observables, control plots
      //  and cross section histograms
      esanalysis.DoCrossSectionObservablesGen();
      if ( RecBaseCuts ) esanalysis.DoCrossSectionObservablesRec();


      //esanalysis.DoCrossSectionsGenRec();
      if ( cmd_write_minitree){
	if(gH1Tree ->IsMC()) esanalysis.FillMiniTree();
        else if(RecBaseCuts) esanalysis.FillMiniTree();
      }

      // break event loop (if requested)
      if ( opts.IsMaxEvent(ievent+1) ) break; // if command-line option -n is set
      if ( AnaSteer->GetNEvents() > 0 && ievent+1 >= AnaSteer->GetNEvents() ) break; // if steering is set
      ievent++;

   } // *** end event loop ***

   // --- write histograms
   //TFile file(opts.GetOutput(), "RECREATE");
   file.mkdir(esanalysis.GetChainName().Data(),esanalysis.GetChainName().Data(),true)->cd();
   esanalysis.DoWriteHistograms(); 
   Info("main","Histograms written to file %s.",file.GetName());
   file.Write();	
   //if ( cmd_write_minitree ) {
      //file.cd(esanalysis.GetChainName().Data());
      //esanalysis.WriteMiniTree();
   //}
   file.Close();
   cout << "\nHistogram written to " << opts.GetOutput() << endl;

   Mybench->Stop("Analysis time");
   Mybench->Show("Analysis time");

   return 0;
}