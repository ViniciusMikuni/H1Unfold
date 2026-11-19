// (c) MPI 2020
#include "AnalysisBase.h"
 
// C++ includes

// Root includes
 
// H1 includes
#include "H1Tools/H1RunList.h"
#include "H1Tools/H1TrigPrescaleWeight.h"
#include "H1Geom/H1DetectorStatus.h"
#include "H1Analysis/H1AnalysisSteer.h"
#include "H1PhysUtils/H1MCGeneratorFinder.h"

#include "H1Skeleton/H1Tree.h"
#include "H1PhysUtils/H1BoostedJets.h"
#include "H1HadronicCalibration/H1HadronicCalibration.h"

#include "H1Calculator/H1Calculator.h"
#include "H1Calculator/H1CalcTrig.h"
#include "H1Calculator/H1CalcWeight.h"
#include "H1Calculator/H1CalcVertex.h"
#include "H1Calculator/H1CalcEvent.h"
#include "H1Calculator/H1CalcKine.h"
#include "H1Calculator/H1CalcElec.h"
#include "H1Calculator/H1CalcFs.h"
#include "H1Calculator/H1CalcHad.h"
#include "H1Calculator/H1CalcTrack.h"
#include "H1Calculator/H1CalcBgTiming.h"
#include "H1Calculator/H1CalcSystematic.h"

#include "H1Steering/H1SteerManager.h"
#include "H1Skeleton/H1SteerTree.h"
#include "H1Analysis/H1AnalysisChainSteer.h"

// local headers
#include "H2020HistManager.h"
#include "FidVolCut.h"          // JetsAtHighQ2
#include "JetTools.h"           // JetsAtHighQ2
#include "TDetectQedc.h"        // JetsAtHighQ2
#include "EventshapeTools.h"

using namespace std;

// #include "H1Calculator/H1CalcGenericInterface.h"
// using namespace H1CalcGenericInterface;


// _______________________________________________________ //
//! Constructor
AnalysisBase::AnalysisBase(TString chain) {
   fChain       = chain;
   fChainName   = fChain;
   fChainName.Resize(fChainName.First('_'));
}


// _______________________________________________________ //
AnalysisBase::~AnalysisBase() {

}


// _______________________________________________________ //
//!
//!  AnalysisBase::SetSysShift
//!
//!  Apply a systematic shift to the entire analysis
//!  Use:  
//!        0   =  no shift
//!        >0  =  systematic shift, as defined in H1CalcSystematic
//!
//!  See: https://www-h1.desy.de/icas/oop/roothtml/4.0.25/src/H1CalcSystematic.h.html#20
//!
//!  Note: this method is called in 'main' to set fSys
//!        and the for each event to re-initialize the
//!        systematic shift.
//!        It also serves for documentation.
//!  
//!  Useful: name        enum 
//!          eHadEn         0   (shift: 0.01)  "JES"         (clusters inside calibrated jets). Thesis R. Kogler & high-Q2 jets paper
//!          eHadEnCor      1   (shift: 0.01)  "RCES","HFS"  (clusters outside of jets)         Thesis R. Kogler & high-Q2 jets paper
//!          eElecEn        6   (shift 0.005)  "electron energy" arXiv:0904.3513: 0.5 [2000]
//!          eElecTh        9   (shift 0.0005) "electron theta"  arXiv:0904.3513: 0.2mrad, 0.09 [2000], 0.5mrad [low-Q2 FL]
//!
//! //fSystematics["ElEnUnc"] = make_pair(H1CalcSystematic::eElecEnUnc,0.005);
void AnalysisBase::SetSysShift(int sys) {
   
  if ( fSys != -9999 ) {
    gH1Calc->Systematic()->ShiftUp(fSys);
    if ( fSys == H1CalcSystematic::eHadEn )    H1BoostedJets::Instance()->SysShiftJetEnergy(0.01);
    if ( fSys == H1CalcSystematic::eHadEnCor ) H1BoostedJets::Instance()->SysShiftHFSEnergy(0.01);
  }
  // fThisSys = it->second.first;
  // gH1Calc->Systematic()->ShiftUp(it->second.first);
  // if ( it->second.first == H1CalcSystematic::eHadEn )        H1BoostedJets::Instance()->SysShiftJetEnergy(it->second.second);
  // if ( it->second.first == H1CalcSystematic::eHadEnCor )     H1BoostedJets::Instance()->SysShiftHFSEnergy(it->second.second);
      
  // gH1Calc->Systematic()->SetShift(H1CalcSystematic::eElecEn, 0.005);
  // gH1Calc->Systematic()->SetShift(H1CalcSystematic::eElecEnUnc, 0.005);
  //   gH1Calc->Systematic()->SetShift(H1CalcSystematic::eElecTh, 0.001);
      
  //   gH1Calc->Systematic()->SetShiftElecEnWheel(0, 0.005);
  //   gH1Calc->Systematic()->SetShiftElecEnWheel(1, 0.005);
  //   gH1Calc->Systematic()->SetShiftElecEnWheel(2, 0.005);
  //   gH1Calc->Systematic()->SetShiftElecEnWheel(3, 0.01);
  //   gH1Calc->Systematic()->SetShiftElecEnWheel(4, 0.01);

  //    gH1Calc->Systematic()->SetShiftElecTheta(0, 0.001);
  //    gH1Calc->Systematic()->SetShiftElecTheta(1, 0.001);
  //    gH1Calc->Systematic()->SetShiftElecTheta(2, 0.002);

  //    gH1Calc->Systematic()->SetShift(H1CalcSystematic::eHadEn, 0.01);

  fSys = sys; // set sys shift.
}


// _______________________________________________________ //
//!
//!  AnalysisBase::DoBaseInitialSettings
//!
//!  Initialize all quantities of the base class
//!  before the first event
//!  
void AnalysisBase::DoBaseInitialSettings() {
   
   // ------------------------------------------------------------------------ //
   // --- read steerings
   // ------------------------------------------------------------------------ //
   H1SteerTree*          SteerTree     = (H1SteerTree*)        gH1SteerManager->GetSteer( H1SteerTree::Class() , fChain.Data() );
   fLumiMC = SteerTree->GetLumi();

   //Info("AnalysisBase::InitialSettings","Reading H1AnalysisChainSteer from steering file with name: %s",ChainName);
   H1AnalysisChainSteer* SteerAnaChain = (H1AnalysisChainSteer*)gH1SteerManager->GetSteer( H1AnalysisChainSteer::Class() , fChainName.Data() );
   H1AnalysisSteer*      AnaSteer      = (H1AnalysisSteer*)     gH1SteerManager->GetSteer( H1AnalysisSteer::Class() , "EventShapes" );


   // ------------------------------------------------------------------------ //
   // --- Identify chain (MC,Data,NoRad,Bkg ... )
   // ------------------------------------------------------------------------ //
   fNoRadMC  =  false ; // Todo, set this from chain
   fGenOnly  =  false ; // todo, set from steering
   IsMC      =  gH1Calc->Event()->IsMC();
   IsBkgMC   =  IsMC && SteerAnaChain->GetChainType() == H1AnalysisChainSteer::eBackground;
   IsHQ      =  true  ; //Todo. not yet implemented
   
   // ------------------------------------------------------------------------ //
   // --- set run period 
   // ------------------------------------------------------------------------ //
   gH1Constants->SetConstants(gH1Calc->GetRunNumber()); // SetConstants takes either period or run
   int RunPeriod = gH1Constants->GetRunPeriod();




   if ( gH1Calc->GetRunNumber() >0 &&  gH1Calc->GetRunNumber()<=12 && RunPeriod == gH1Calc->GetRunNumber()) {
     Warning("AnalysisBase","RunPerdiod is %d. So RunNumber can be misinterpreted as RunPeriod. Changing to 0.",gH1Calc->GetRunNumber());
     RunPeriod=0;
   }
   Info("AnalysisBase","First RunNumber is %d, and corresponding RunPerdiod %d",gH1Calc->GetRunNumber(),RunPeriod);
   if ( RunPeriod < H1Constants::eEplus9900 ) {
     Info("AnalysisBase","RunPeriod not recognized from RunNumber %d",gH1Calc->GetRunNumber());
     TObjArray* runsel = AnaSteer->GetRunSelection();
     TObjString* run1= (TObjString*)runsel->First();
     TObjString* run2= (TObjString*)runsel->At(1);
     gH1Constants->SetConstants(run1->GetString().Atoi());
     RunPeriod = gH1Constants->GetRunPeriod();
     // test, if last run is consistent.
     gH1Constants->SetConstants(run2->GetString().Atoi());
     int RunPeriod2 = gH1Constants->GetRunPeriod();
     if ( RunPeriod != RunPeriod2 ) {
       Error("AnalysisBase::InitialSettings","Run Period not consistent from first and last run in Steering 'fRunSelection'.");
       exit(1);
     }
   }
   RunPeriod = gH1Constants->GetRunPeriod();

   cout << "Run Period                          = ";
   if      ( RunPeriod == H1Constants::eEplus9900 )  cout << "e+ 99/00" << endl; 
   else if ( RunPeriod == H1Constants::eEplus0304 )  cout << "e+ 03/04" << endl; 
   else if ( RunPeriod == H1Constants::eEminus0405 ) cout << "e- 04/05" << endl; 
   else if ( RunPeriod == H1Constants::eEminus06 )   cout << "e- 06" << endl;    
   else if ( RunPeriod == H1Constants::eEplus0607 )  cout << "e+ 06/07" << endl; 
   else {
      Error("AnalysisBase::InitialSettings","Unkown Run Period. Please correct the steering.");
      exit(1);
   }

   // --- reconstruction method
   H1Calculator::Instance()->Const()->SetKineRecMethod( AnaSteer->GetKineRecMethod() );

   // ------------------------------------------------------------------------ //
   //  Goodrun file
   // ------------------------------------------------------------------------ //
   TFile* GoodRunFile  = TFile::Open(AnaSteer->GetGoodRunFile(),"READ");       //  Preselected exper. data List for analyzed year
   if ( !GoodRunFile ) { Error("AnalysisBase::InitialSettings","Cannot open goodrun file %s",AnaSteer->GetGoodRunFile().Data()); exit(1);}
   fRunList    = (H1RunList*)        GoodRunFile->Get("H1RunList");
   fDectStatus = (H1DetectorStatus*) GoodRunFile->Get("MyDetectorStatus");
   H1ArrayF* zInfo = (H1ArrayF*)     GoodRunFile->Get("zInfo");
   if (!fRunList || !fDectStatus || !zInfo ) {
      Error("AnalysisBase::InitialSettings","Runlist or DetectorStatus could not been opened. Please check the steering.");
      exit(1);
   } 
   // --- set Vtx settings (for each event, or only once?)
   f_VtxZMin = zInfo->At(0) - zInfo->At(1); // needed?
   f_VtxZMax = zInfo->At(0) + zInfo->At(1); // needed?

   // --- some print-out
   cout << "Reading lumi file: " << "first run = "
        << fRunList ->FirstRun() << " last run = "
        << fRunList ->LastRun()  << " integrated luminosity = "
        << fRunList ->GetIntLumi()/1000.0 << " pb^-1 " << endl;

   fLumiData = fRunList->GetIntLumi()/1000.0;
   H1Calculator::Instance()->Const()->SetRunList(fRunList);
   
   // set lumi to CalcWeight
   if ( IsMC ) {
      H1Calculator::Instance()->Weight()->SetDataLumi(fLumiData);
      H1Calculator::Instance()->Weight()->SetMCLumi(fLumiMC);
      H1Calculator::Instance()->Weight()->ApplyLumiRatio(true); // fSteer->GetApplyLumiWeight()
      H1Calculator::Instance()->Weight()->ApplyVertexWeight(true);
   }

   /*
   // ------------------------------------------------------------------------ //
   // --- veto efficienies (low-Q2 analysis)
   // ------------------------------------------------------------------------ //
   //TString vetofile = "/afs/desy.de/user/b/britzger/xxl/LowQ2JetsAnalysis/jet/VetoFiles/s0veto.txt";
   TString vetofile = "/nfs/dust/h1/group/britzger/LowQ2_run/s0veto.txt";
   ReadVetoFile(vetofile);

   // --- use H1TrigPrescale
   H1TrigPrescaleWeight* psw = fRunList->GetTrigPrescaleWeight();
   psw->UseSubTrigger(0);
   if (fMyRunYear>0 ) psw->UseSubTrigger(1);
   if (fMyRunYear>0 ) psw->UseSubTrigger(2);
   psw->UseSubTrigger(3);
   psw->UseSubTrigger(61);
   psw->UseSubTrigger(67);
   psw->UseSubTrigger(76);
   */

   
   // cout<<"First run: "<<fRunList ->FirstRun()<<endl;
   // cout<<"Last  run: "<<fRunList ->LastRun()<<endl;
   //fRunSelection     = "444094,466997"; // e-06 running                                                                                                                                       

   // ------------------------------------------------------------------------ //
   // --- H1BoostedJets
   // ------------------------------------------------------------------------ //
   // --- reset and init kinematics
   // //---this requires H1Calc->Reset()
   // if (dist>c_ElPhotMerg){ // (~5 deg in theta and phi)
   // ....
   // ------------------------------------------------------------------------ //
   H1Calculator::Instance()->SetFinalStateToNC();
   H1Calculator::Instance()->Elec()->UseGenElecUncombined(false);
   H1BoostedJets::Instance()->UseGenElecGammaCombined(true);
   H1BoostedJets::Instance()->ResetSysShifts();
   H1Calculator::Instance() ->Reset(); 
   H1BoostedJets::Instance()->Reset();
   // ------------------------------------------------------------------------ //
   // H1BoostedJets::Instance()->ExcludeScatElecTracks(true, 0.2);
   // H1BoostedJets::Instance()->ExcludeFwdClusters(true, 3.2);
   // H1BoostedJets::Instance()->SetReferenceFrame(H1BoostedJets::eBreit);
   // H1BoostedJets::Instance()->UseFastJet();
   // ------------------------------------------------------------------------ //


   // ------------------------------------------------------------------------ //
   // --- The H1Calculator
   // ------------------------------------------------------------------------ //
   H1Calculator::Instance();
   // if H1HadronicCalibration::eHighPtJet: 
   H1HadronicCalibration::Instance()->SetCalibrationMethod(H1HadronicCalibration::eHighPtJet);
   H1Calculator::Instance()->CalibrateHadrooII();
   H1Calculator::Instance()->CalibrateLatestHadrooII();
   // --- HFS Settings
   H1Calculator::Instance()->Had()->DoNotUseIron();                  // Do not include Iron in HFS sum
   H1Calculator::Instance()->Had()->RejectTracksNearScatElec(0.2);   // exclude tracks close to the scattered electron
   
   // apply electron calibration locally:
   //gH1Calc->Elec()->CalibrateElectrons();
   //gH1Calc->Elec()->DoStackWiseCalibration(kTRUE);
   //gH1Calc->Elec()->DoZimpactWiseCalibration(kTRUE);
   //gH1Calc->Elec()->DoPhiWiseCalibration(kTRUE);
   //gH1Calc->Elec()->DoResolutionSmearing(kTRUE);
   //gH1Calc->Elec()->UseTrackDowngrade(kFALSE);     // downgrade tracks in MC?

   // --- Set Final State to NC
   H1Calculator::Instance()->SetFinalStateToNC();
   
   // --- vertex settings
   H1Calculator::Instance()->Vertex()->DoNotUseCIPForOptimalVertex();
   H1Calculator::Instance()->Vertex()->SetTrackClusterDistance(8.0); //  03.16
   SetSysShift(fSys);

   // -------------------------------------------------------------------
   // see: h1oo/H1Analysis/H1AnalysisChain.C::SteerCalculator()
   //      for many options...
   // H1Calculator::Instance() -> Weight() -> ApplyVertexWeight(true);        /// Applied ONLY for MC vs. H1Calc.
   // H1Calculator::Instance() -> Weight() -> SetZvtxMean(0.8);               /// 0.8 & 11.1 taken from Dja&Rap.log files
   //      gH1Calc -> Weight() -> SetZvtxSigma(11.1);             /// The same for both MC-s
   //      gH1Calc->Systematic()->SetSpaCalFlShifts();             /// For Setting SpaCal Systematics Shift


   // -------------------------------------------------------------------
   // --- fiducial volume cut
   fFidVolCut = new FidVolCut("FidVolCut_HERA2");
   
   // -------------------------------------------------------------------
   // --- Reweighting
   //if ( !fGenOnly ) InitReweighting();
   
}



// _______________________________________________________ //
//!
//!  AnalysisBase::DoBaseReset
//!
//!  Reset all quantities of the base class
//!  at the beginning of each new event
//!  
void AnalysisBase::DoBaseReset() {
   fBasicCutsRec = false;
   fBasicCutsGen = false;

   // --- reset h1oo
   gH1Calc->Reset();
   gH1Calc->Vertex()->SetPrimaryVertexType(H1CalcVertex::vtOptimalNC); // use optimal NC vertex
   H1BoostedJets::Instance()->Reset();
   H1BoostedJets::Instance()->ResetSysShifts();
   SetSysShift(fSys);
   static EventshapeTools ESTools;
   ESTools.ApplyNCTrackClusterWeight();

   static JetTools tools;
   if ( tools.GenElecPhotDist()  > 0.15 ) { // (~5 deg in theta and phi) 
      gH1Calc->Elec()->UseGenElecUncombined(true);
      H1BoostedJets::Instance()->UseGenElecGammaCombined(false);
      H1BoostedJets::Instance()->ExcludeGenGamma(true);
   } else {
      gH1Calc->Elec()->UseGenElecUncombined(false);
      H1BoostedJets::Instance()->UseGenElecGammaCombined(true);
   }

   
}


// _______________________________________________________ //
//!
//!  AnalysisBase::DoBasicRecCuts
//!
//!  Do _all_ basic cuts on rec-level for the analysis
//!  Store result in 
//!     bool AnalysisBase::fBasicCutsRec
//!
//!  Fulfill background cuts.
//!    Presently cuts implemented from
//!       - HighQ2: BackgroundRejector
//!       - HighQ2: DISSelector
//!  
bool AnalysisBase::DoBasicCutsRec() {

   // ---------------- Run list -------------------------
   // --- set return value
   //     and initialize with basic run selection cut
   
//insert q2-x plot
   H2020HistManager& hm       = HistMaster::Instance()->GetHistManager("DISEvent");
   auto betabins = H2020HistManager::MakeLogBinning(50, 0.001, 1.);
   hm.Get<TH2D>("13_1_before_cuts",";X_{es};Q2_{es} [GeV^2]", 50, -0.05, 1.05, 50, 5, 10000)  -> Fill(gH1Calc->Kine()->GetXes(), gH1Calc->Kine()->GetQ2es());
   hm.Get<TH2D>("13_1_before_cuts_lxy",";X_{es};Q2_{es} [GeV^2]", H2020HistManager::MakeLogBinning(50, 0.001, 1.), H2020HistManager::MakeLogBinning(50, 5, 10000)) -> Fill(gH1Calc->Kine()->GetXes(), gH1Calc->Kine()->GetQ2es());
   fBasicCutsRec   = fRunList ->IsSelected( H1Calculator::Instance()->GetRunNumber()) && fDectStatus->IsOn();

   
   // ---------------- electron cuts ---------------------
   // DB: from HQ-DISSelector
   // Electron : Scattered Electron in LAr with Energy > 11 GeV
   bool ElecCuts = true;
   ElecCuts   &=   gH1Calc->Elec()->GetType() == 1;            // AddRecCut(new H1CutDiscreteInt(Elec_Type, 1), "Elec_Type");     // 1 for LAr, 4 for SpaCal
   ElecCuts   &=   gH1Calc->Elec()->GetIsScatteredElectron();  // AddRecCut(new H1CutBool(Elec_IsScatteredElectron), "ScattElec");
   ElecCuts   &=   gH1Calc->Elec()->GetFirstElectron().E() >= 11.0; //8.0, tighter for DIS  // AddRecCut(new H1CutLorentz(Elec_FirstElectron, H1CutLorentz::E, 11.0, FLT_MAX), "Electron Energy");


  
   // ---------------- HFS cuts --------------------------
   // Empz
   bool HFSCuts = true;
   HFSCuts    &=  gH1Calc->Fs()->GetEmpz() > 45;               // cut harder than ELAN for better boost reconstruction
   HFSCuts    &=  gH1Calc->Fs()->GetEmpz() < 65;               // AddRecCut(new H1CutFloat(Fs_Empz,       45.0, 65.0), "E-pz");  
   

   // ---------------- Trigger ---------------------------
   // Trigger Cuts: Select St67 and St77 -> discard S77, only 0.01% of events triggered by it 
   // H1Cut* SubTrig67 = new H1CutBool(Trig_L1ac_idx, 67);
   // AddRecCut(new H1CutOr(new H1CutBool(Event_IsMC), SubTrig67), "SubTrigger S67");
   bool TrigCuts = ( gH1Calc->Trig()->GetL1ac(67) || IsMC );


   // ---------------- Vertex ----------------------------
   // vertex-Z (from DISSelector [implicitly]
   // H1Cut* FwdTheta   = new H1CutLorentz(Elec_FirstElectron, H1CutLorentz::Theta, -FLT_MAX, 30*DegToRad);
   // H1Cut* CJCVtx     = new H1CutBool(Vertex_IsVertexFound_idx, H1CalcVertex::vtCJC);
   // H1Cut* FwdRegion  = new H1CutAnd(FwdTheta, CJCVtx);
   // H1Cut* OptimalVtx = new H1CutBool(Vertex_IsVertexFound_idx, H1CalcVertex::vtOptimalNC);
   // AddRecCut(new H1CutOr(OptimalVtx, FwdRegion) , "NCOptimalVertex");
   const bool FwdTheta   = gH1Calc->Elec()->GetFirstElectron().Theta() < 30.*TMath::DegToRad();
   const bool CJCVtx     = gH1Calc->Vertex()->GetIsVertexFound(H1CalcVertex::vtCJC);
   const bool FwdRegion  = FwdTheta && CJCVtx;
   const bool OptimalVtx = gH1Calc->Vertex()->GetIsVertexFound(H1CalcVertex::vtOptimalNC);
   bool VtxCuts          =  ( OptimalVtx  || FwdRegion ); // "NCOptimalVertex"
   VtxCuts &=  fabs(gH1Calc->Vertex()->GetZ()) < 35.0; //H1CutFloat on variable - Vertex_Z in range -35  to  35
   

   // AddRecCut(new H1CutFloat(Vertex_Z, -35.0, 35.0), "Z-Vertex");
   VtxCuts &= gH1Calc->Vertex()->GetZ() > -35.0; // "Z-Vertex"
   VtxCuts &= gH1Calc->Vertex()->GetZ() <  35.0; // "Z-Vertex"
   

   // --- collect NC DIS cuts
   fBasicCutsRec &= ElecCuts;
   fBasicCutsRec &= HFSCuts;
   fBasicCutsRec &= TrigCuts;
   fBasicCutsRec &= VtxCuts;
   
   if ( !fBasicCutsRec ) return false;
   // ---- HQ-DISSelector end ---


   // ---------------- LAr fiducial volume cuts  --------------------
   // electron Zimpact Criteria
   // AddRecCut(new H1CutOr(new H1CutFloat(Elec_Zimpact, -190.0,  15.0),
   //                       new H1CutFloat(Elec_Zimpact,   25.0,  FLT_MAX)), "Electron Zimpact");
   const bool Zimpact1 = gH1Calc->Elec()->GetZimpact() > -190.0  && gH1Calc->Elec()->GetZimpact() < 15.0;
   const bool Zimpact2 = gH1Calc->Elec()->GetZimpact() > 25.0;
   const bool FidCuts1 = (Zimpact1 || Zimpact2); // Electron Zimpact

   // Remove Cracks
   // AddRecCut(new H1CutOr(new H1CutFloat(Elec_PhiOctant, 2.0*DegToRad, 43.0*DegToRad),
   //                       new H1CutFloat(Elec_Zimpact, 300.0, FLT_MAX)), "PhiCracks");
   const bool nocrack1 = gH1Calc->Elec()->GetPhiOctant() > 2.0*TMath::DegToRad()  && gH1Calc->Elec()->GetPhiOctant() < 43.0*TMath::DegToRad();
   const bool nocrack2 = gH1Calc->Elec()->GetZimpact() > 300.0;
   const bool FidCuts2 = ( nocrack1 || nocrack2 ); // PhiCracks

   // --- Apply Fiducial Volume to Get 100% of Trigger Efficiency
   //AddRecCut(new FidVolCut("FidVolCut_HERA2"), "Fiducial_Volume_0307");
   const bool FidCuts3 =  fFidVolCut->FiducialVolumeCut(); // Fiducial_Volume_0307

   fBasicCutsRec &= FidCuts1;
   fBasicCutsRec &= FidCuts2;
   fBasicCutsRec &= FidCuts3;
   if ( !fBasicCutsRec ) return false;



   // ---------------------- Background ------------------
   // From HQ background-rejector:
   bool BkgCuts = true;
   // this cut gets rid of most of the photoproduction background due to soft, wrongly identified electrons in the forward direction
   BkgCuts &= gH1Calc->Kine()->GetYe() >= 0.0; // >= than 0, since Ye can become zero, and this is accepted in JetsAtHighQ2
   BkgCuts &= gH1Calc->Kine()->GetYe() < 0.94; // Ye Rec Background Reduction

   // ---- colliding bunch 
   BkgCuts &= gH1Calc->Event()->GetBunchType() == 3 ; //  "Bunch Type" == 3;

   // cut out events with a wrongly charged scattered lepton -> don't cut them out: efficiency at high electron Pt unknown
   // /// ... AddRecCut(new H1CutInt(Elec_BeamChargeMatch, 1,  2), "Wrong Charge Electrons")
   
   // ---------------------- NON EP BACKGROUND FINDERS ---------------------------
   {
      const bool Finder0 =  gH1Calc->BgTiming()->GetIbg(0);
      const bool Finder1 =  gH1Calc->BgTiming()->GetIbg(1);
      const bool Finder5 =  gH1Calc->BgTiming()->GetIbg(5);
      const bool Finder6 =  gH1Calc->BgTiming()->GetIbg(6);
      const bool Finder7 =  gH1Calc->BgTiming()->GetIbg(7);
      // Finder 7 and  Pth/Pte < 0.1
      BkgCuts &= !( Finder7 && gH1Calc->Fs()->GetHadPtElecPtRatio() < 0.1); // BkgFinder1, Finder 7 and  Pth/Pte < 0.1
   
      // Finder 0 and 1 or by two finders out of 5-7 and  Pth/Pte >= 0.1
      const bool Pair1  = Finder0 && Finder1;
      const bool Pair2  = Finder5 && Finder6;
      const bool Pair3  = Finder5 && Finder7;
      const bool Pair4  = Finder6 && Finder7;
      const bool Pairs  = Pair1 || Pair2 || Pair3 || Pair4; // H1Cut* Pairs  = new H1CutOr(Pair1, Pair2, Pair3, Pair4);
      BkgCuts &=  !(Pairs && gH1Calc->Fs()->GetHadPtElecPtRatio() > 0.1 ); // BkgFinder2
   
      // Finder ( 5 || 6 ) && Pth/Pte < 0.5
      const bool Finders = Finder5 || Finder6;
      BkgCuts &=  !( Finders && gH1Calc->Fs()->GetHadPtElecPtRatio() < 0.5 ); // BkgFinder3
   }

   // ---------------- Anti QED Compton ------------------
   {
      // H1Cut* nEmParts = new H1CutInt(Elec_NEmParts, 2, INT_MAX);
      // H1Cut* Energy1  = new H1CutLorentz(Elec_FirstElectron, H1CutLorentz::E, 4, FLT_MAX);
      // H1Cut* Energy2  = new H1CutLorentz(Elec_SecondElectron, H1CutLorentz::E, 4, FLT_MAX);
      const TLorentzVector& Elec1 = gH1Calc->Elec()->GetFirstElectron();
      const TLorentzVector& Elec2 = gH1Calc->Elec()->GetSecondElectron();
      const bool nEmParts = gH1Calc->Elec()->GetNEmParts() >= 2 ;
      const bool Energy1  = Elec1.E()  > 4.0;
      const bool Energy2  = Elec2.E()  > 4.0;
   
      // H1Cut* EnergySum = new H1CutFunction("[0]+[1]", 18.0, FLT_MAX);
      // EnergySum->AddLorentzVariable(0, Elec_SecondElectron, H1CutLorentz::E);
      // EnergySum->AddLorentzVariable(1, Elec_FirstElectron, H1CutLorentz::E);
      const bool EnergySum = Elec1.E() + Elec2.E() > 18.0;

      // H1Cut* Acoplanarity = new H1CutFunction("-cos(TMath::Abs([0]-[1]))", 0.95, FLT_MAX);
      // Acoplanarity->AddLorentzVariable(0, Elec_SecondElectron, H1CutLorentz::Phi);
      // Acoplanarity->AddLorentzVariable(1, Elec_FirstElectron, H1CutLorentz::Phi);
      const bool Acoplanarity = -cos(TMath::Abs(Elec1.Phi()-Elec2.Phi())) > 0.95;

      // AddRecCut(new H1CutNot(new H1CutAnd(nEmParts, Energy1,
      //                                     Energy2, EnergySum, Acoplanarity)), "AntiCompton");
      BkgCuts &= !( nEmParts && Energy1 && Energy2 && EnergySum && Acoplanarity ); // AntiCompton
   }

   
   // --- apply background cuts
   fBasicCutsRec &= BkgCuts;

//x and q2 from calculator
   if(fBasicCutsRec){
      hm.Get<TH2D>("13_2_detector_cuts",";X_{es};Q2_{es} [GeV^2]", 50, -0.05, 1.05, 50, 5, 10000)  -> Fill(gH1Calc->Kine()->GetXes(), gH1Calc->Kine()->GetQ2es());
      hm.Get<TH2D>("13_2_detector_cuts_lxy",";X_{es};Q2_{es} [GeV^2]", H2020HistManager::MakeLogBinning(50, 0.001, 1.), H2020HistManager::MakeLogBinning(50, 5, 10000.)) -> Fill(gH1Calc->Kine()->GetXes(), gH1Calc->Kine()->GetQ2es());
   }


   return fBasicCutsRec;
}


// _______________________________________________________ //
//!
//!  AnalysisBase::DoBasicCutsGen
//!
//!  Do _all_ basic cuts on rec-level for the analysis
//!  Store result in 
//!     bool AnalysisBase::fBasicCutsGen
//!  
bool AnalysisBase::DoBasicCutsGen() {
   // (re)set return value
   fBasicCutsGen = true;
   if ( !IsMC ) return true; // data is always good!
   
   //  for Django and Rapgap (radiative MC's)
   // drop events with QEDC topology
   if (!fNoRadMC){
      if ( (H1Tree::Instance()->IsMC() == H1MCGeneratorFinder::kRAPGAP) ||
           (H1Tree::Instance()->IsMC() == H1MCGeneratorFinder::kDJANGO) ){
         static H1PartMCArrayPtr mcparts;
         TDetectQedc QEDCFinder(mcparts);
         fBasicCutsGen &= !QEDCFinder.IsQedcEvent();
      }
   }

   // todo
   // remove low-Q2 Django and rapgap events
   // ... these overlap with Pythia photoproduction (Q2~ 2-4GeV2)
   return fBasicCutsGen;
}



// _______________________________________________________ //
void   AnalysisBase::DoWriteHistograms(){
   Info("DoWriteHistograms","Write histograms to directory: %s",gDirectory->GetName());
   HistMaster::Instance()->WriteAll(gDirectory);
}


//!
//!  AnalysisBase::InitMiniTree
//!
//!  Initialize mini tree
//!  
void AnalysisBase::InitMiniTree() {
   fMiniTree = new TTree("minitree","minitree");
   TTree* T = fMiniTree; // rename

   T->Branch("wgt", &fTreeVar.event_weight, "event_weight/F");
   T->Branch("x", &fTreeVar.event_x, "event_x/F");
   T->Branch("y", &fTreeVar.event_y, "event_y/F");
   T->Branch("Q2", &fTreeVar.event_Q2, "event_Q2/F");
   T->Branch("gen_x", &fTreeVar.gen_event_x, "gen_event_x/F");
   T->Branch("gen_y", &fTreeVar.gen_event_y, "gen_event_y/F");
   T->Branch("gen_Q2", &fTreeVar.gen_event_Q2, "gen_event_Q2/F");

   T->Branch("tau1b", &fTreeVar.tau1b, "tau1b/F");
   T->Branch("gen_tau1b", &fTreeVar.gen_tau1b, "gen_tau1b/F");
   T->Branch("tauzQ", &fTreeVar.tauzQ, "tauzQ/F");
   T->Branch("gen_tauzQ", &fTreeVar.gen_tauzQ, "gen_tauzQ/F");

   T->Branch("vertex_z", &fTreeVar.vertex_z, "vertex_z/F");
   T->Branch("ptmiss", &fTreeVar.ptmiss, "ptmiss/F");
   T->Branch("pth", &fTreeVar.pth, "pth/F");
   T->Branch("ptratio_da", &fTreeVar.ptratio_da, "ptratio_da/F");
   T->Branch("ptratio_ele", &fTreeVar.ptratio_ele, "ptratio_ele/F");

   T->Branch("acoplanarity", &fTreeVar.acoplanarity, "acoplanarity/F");
   T->Branch("Empz", &fTreeVar.Empz, "Empz/F");
   T->Branch("e_px", &fTreeVar.e_px, "e_px/F");
   T->Branch("e_py", &fTreeVar.e_py, "e_py/F");
   T->Branch("e_pz",&fTreeVar.e_pz, "e_pz/F");

   T->Branch("gen_e_px", &fTreeVar.gene_px, "gene_px/F");
   T->Branch("gen_e_py", &fTreeVar.gene_py, "gene_py/F");
   T->Branch("gen_e_pz",&fTreeVar.gene_pz, "gene_pz/F");

   
   T->Branch("njet", &fTreeVar.njet, "njet/F");
   T->Branch("npart", &fTreeVar.npart, "npart/F");
   T->Branch("ngenjet", &fTreeVar.ngenjet, "ngenjet/F");
   T->Branch("ngenpart", &fTreeVar.ngenpart, "ngenpart/F");
   T->Branch("n_total",&fTreeVar.nconstituents);
   T->Branch("jet_pt", &fTreeVar.jet_pt);
   T->Branch("gen_jet_pt", &fTreeVar.gen_jet_pt);
   T->Branch("jet_phi", &fTreeVar.jet_phi);
   T->Branch("gen_jet_phi",&fTreeVar.gen_jet_phi);
   T->Branch("jet_eta", &fTreeVar.jet_eta);
   T->Branch("gen_jet_eta", &fTreeVar.gen_jet_eta);

   T->Branch("part_pt", &fTreeVar.part_pt);
   T->Branch("part_eta", &fTreeVar.part_eta);
   T->Branch("part_phi", &fTreeVar.part_phi);
   T->Branch("part_charge", &fTreeVar.part_charge);


   T->Branch("gen_part_pt", &fTreeVar.gen_part_pt);
   T->Branch("gen_part_eta", &fTreeVar.gen_part_eta);
   T->Branch("gen_part_phi", &fTreeVar.gen_part_phi);
   T->Branch("gen_part_charge", &fTreeVar.gen_part_charge);

   
   // //   qT and z, defined below:
   // TVector2 jet2(jet.px(),jet.py());
   // TVector2 electron2(ElecPx,ElecPy);
   // TVector2 qT = jet2 + electron2;
   // // z variable (Lorentz invariant).                      
   // double z = proton.Dot(jet_vec)/proton.Dot(virtual_photon);

}


// _______________________________________________________ //
//!
//!  AnalysisBase::InitMiniTree
//!
//!  Fill the minitree
//!  
void AnalysisBase::FillMiniTree() {
   fMiniTree->Fill();
}

// _______________________________________________________ //
//!
//!  AnalysisBase::InitMiniTree
//!
//!  Write mini tree to file
//!  
void AnalysisBase::WriteMiniTree() {
   fMiniTree->Write();
}
