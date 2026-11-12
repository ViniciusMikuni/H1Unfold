"""
Read H1 MC events from a ROOT file with pre-selected hadrons and scattered electron,
cluster into jets using FastJet kt algorithm (R=1.0, pT > 7 GeV), compute jet
observables, and save to a new ROOT file.

Input tree branches:
  weight, Q2, y, x,
  elec_pt, elec_E, elec_eta, elec_phi,
  nhadron, hadron_pt, hadron_E, hadron_eta, hadron_phi, hadron_charge
"""

import uproot
import numpy as np
import fastjet
import awkward as ak
import argparse

PROTON_BEAM = np.array([0.0, 0.0, 920.0, 920.0])  # px, py, pz, E


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Cluster jets from H1 MC DIS events and compute observables."
    )
    parser.add_argument("--input", required=True, help="Path to input ROOT file")
    parser.add_argument("--output", required=True, help="Path to output ROOT file")
    parser.add_argument("--tree", default="Tree", help="Name of input TTree")
    parser.add_argument("-R", type=float, default=1.0, help="Jet radius parameter")
    parser.add_argument("--ptmin", type=float, default=7.0, help="Min jet pT (lab frame)")
    parser.add_argument("--breit_ptmin", type=float, default=5.0, help="Min jet pT (Breit frame)")
    parser.add_argument("--no_breit", action="store_true", default=False,
                        help="Skip Breit-frame clustering (zjet_breit will be absent)")
    parser.add_argument("--num_events", default=None, type=int,
                        help="Number of events to process")
    return parser.parse_args()


def to_cartesian(pt, eta, phi, E=None):
    """Convert (pt, eta, phi) to (px, py, pz). If E is None, assumes massless."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    if E is None:
        E = np.sqrt(px**2 + py**2 + pz**2)
    return px, py, pz, E


def calculate_q(hfs_pz, hfs_energy, elec_px, elec_py, elec_pz, elec_E):
    """Compute virtual photon 4-momentum q = k - k' via the sigma method."""
    sigma_h = np.sum(hfs_energy - hfs_pz)
    elec_p = np.sqrt(elec_px**2 + elec_py**2 + elec_pz**2)
    elec_theta = np.arccos(np.clip(elec_pz / elec_p, -1, 1)) if elec_p > 0 else 0.0
    sigma_eprime = elec_E * (1 - np.cos(elec_theta))
    sigma_tot = sigma_h + sigma_eprime

    beam_pz = -sigma_tot / 2.0
    beam_E = sigma_tot / 2.0

    q = np.array([0.0 - elec_px,
                  0.0 - elec_py,
                  beam_pz - elec_pz,
                  beam_E  - elec_E])
    return q


def boost_particles_to_breit(px, py, pz, E, q):
    """Vectorized Breit-frame boost (one event at a time)."""
    qx, qy, qz, qE = q
    qpt = np.sqrt(qx * qx + qy * qy)

    qdotq = qE * qE - qx * qx - qy * qy - qz * qz
    Q = np.sqrt(-qdotq)
    Sigma = qE - qz

    Q2_over_sigma = (Q * Q) / Sigma
    boostvector_px = qx
    boostvector_py = qy
    boostvector_pz = qz + Q2_over_sigma
    boostvector_E  = qE + Q2_over_sigma

    xboost_px = qx / qpt
    xboost_py = qy / qpt
    xboost_pz = qpt / Sigma
    xboost_E  = qpt / Sigma

    yboost_px = -qy / qpt
    yboost_py =  qx / qpt

    boosted_px = xboost_E * E - xboost_px * px - xboost_py * py - xboost_pz * pz
    boosted_py = -yboost_px * px - yboost_py * py
    boosted_pz = (qE * E - qx * px - qy * py - qz * pz) / Q
    boosted_E  = (boostvector_E * E - boostvector_px * px
                  - boostvector_py * py - boostvector_pz * pz) / Q

    return boosted_px, boosted_py, boosted_pz, boosted_E


def calculate_breit_zjet(breit_jet, Q):
    n = np.array([0, 0, 1, 1], dtype=np.float64)
    numerator = n[3] * breit_jet.E() - n[2] * breit_jet.pz()
    return numerator / Q


def calculate_jet_features(jet, q, P_dot_q):
    """Compute substructure observables for a single jet."""
    tau_10 = 0.0
    for constituent in jet.constituents():
        dr = jet.delta_R(constituent)
        pt = constituent.pt()
        tau_10 += pt * dr

    jet_pt = jet.pt()
    log_tau_10 = np.log(tau_10 / jet_pt) if jet_pt > 0 and tau_10 > 0 else np.nan

    P_dot_jet = (PROTON_BEAM[3] * jet.E()
                 - PROTON_BEAM[0] * jet.px()
                 - PROTON_BEAM[1] * jet.py()
                 - PROTON_BEAM[2] * jet.pz())
    zjet = P_dot_jet / P_dot_q if P_dot_q != 0 else np.nan

    phi = (jet.phi() + np.pi) % (2 * np.pi) - np.pi
    return jet_pt, phi, log_tau_10, zjet


def make_pseudojets(px, py, pz, E):
    return [fastjet.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(E[i]))
            for i in range(len(px))]


def main():
    flags = parse_arguments()
    R = flags.R

    print(f"Opening {flags.input}")
    with uproot.open(flags.input) as f:
        t = f[flags.tree]
        branches = [
            "weight", "Q2", "y", "x",
            "elec_pt", "elec_E", "elec_eta", "elec_phi",
            "hadron_pt", "hadron_E", "hadron_eta", "hadron_phi",
        ]
        data = t.arrays(branches, library="ak", entry_stop=flags.num_events)

    n_events = len(data["Q2"])
    print(f"Read {n_events} events")

    jetdef = fastjet.JetDefinition(fastjet.kt_algorithm, R)

    do_breit = not flags.no_breit

    # Per-event output (jagged)
    out_jet_pt    = []
    out_jet_tau10 = []
    out_zjet      = []
    out_deltaphi  = []
    out_njets     = []
    if do_breit:
        out_breit_pt   = []
        out_zjet_breit = []

    # Event-level kinematics (one per event, kept in sync with jet lists)
    out_weight = []
    out_Q2     = []
    out_x      = []
    out_y      = []

    for i in range(n_events):
        if i % 100_000 == 0:
            print(f"  Event {i}/{n_events}")

        Q2_i = float(data["Q2"][i])
        x_i  = float(data["x"][i])
        y_i  = float(data["y"][i])
        w_i  = float(data["weight"][i])

        # Scattered electron
        elec_px, elec_py, elec_pz, elec_E = to_cartesian(
            float(data["elec_pt"][i]),
            float(data["elec_eta"][i]),
            float(data["elec_phi"][i]),
            float(data["elec_E"][i]),
        )
        elec_phi_val = float(data["elec_phi"][i])

        # Hadrons
        h_pt  = np.asarray(data["hadron_pt"][i],  dtype=np.float64)
        h_E   = np.asarray(data["hadron_E"][i],   dtype=np.float64)
        h_eta = np.asarray(data["hadron_eta"][i], dtype=np.float64)
        h_phi = np.asarray(data["hadron_phi"][i], dtype=np.float64)

        if len(h_pt) == 0:
            out_jet_pt.append([]);    out_jet_tau10.append([])
            out_zjet.append([]);      out_deltaphi.append([])
            out_njets.append(0)
            out_weight.append(w_i);  out_Q2.append(Q2_i)
            out_x.append(x_i);       out_y.append(y_i)
            if do_breit:
                out_breit_pt.append([]); out_zjet_breit.append([])
            continue

        hfs_px, hfs_py, hfs_pz, hfs_E = to_cartesian(h_pt, h_eta, h_phi, h_E)

        q = calculate_q(hfs_pz, hfs_E, elec_px, elec_py, elec_pz, elec_E)
        P_dot_q = PROTON_BEAM[3] * q[3] - PROTON_BEAM[2] * q[2]

        lab_pseudojets = make_pseudojets(hfs_px, hfs_py, hfs_pz, hfs_E)
        lab_cluster = fastjet.ClusterSequence(lab_pseudojets, jetdef)
        lab_jets = fastjet.sorted_by_pt(lab_cluster.inclusive_jets(ptmin=flags.ptmin))

        ev_pt = []; ev_tau10 = []; ev_zjet = []; ev_dphi = []
        for jet in lab_jets:
            jpt, jphi, ltau, zj = calculate_jet_features(jet, q, P_dot_q)
            delta_phi = np.abs(np.pi + jphi - elec_phi_val) % (2 * np.pi)
            ev_pt.append(jpt);    ev_tau10.append(ltau)
            ev_zjet.append(zj);   ev_dphi.append(delta_phi)

        out_jet_pt.append(ev_pt);    out_jet_tau10.append(ev_tau10)
        out_zjet.append(ev_zjet);    out_deltaphi.append(ev_dphi)
        out_njets.append(len(ev_pt))
        out_weight.append(w_i);  out_Q2.append(Q2_i)
        out_x.append(x_i);       out_y.append(y_i)

        if do_breit:
            breit_px, breit_py, breit_pz, breit_E = boost_particles_to_breit(
                hfs_px, hfs_py, hfs_pz, hfs_E, q)
            pseudojets_breit = make_pseudojets(breit_px, breit_py, breit_pz, breit_E)
            breit_cluster = fastjet.ClusterSequence(pseudojets_breit, jetdef)
            breit_jets = fastjet.sorted_by_pt(
                breit_cluster.inclusive_jets(ptmin=flags.breit_ptmin))
            ev_bpt = [np.sqrt(j.px()**2 + j.py()**2) for j in breit_jets]
            ev_bzjet = [calculate_breit_zjet(j, np.sqrt(Q2_i)) for j in breit_jets]
            out_breit_pt.append(ev_bpt)
            out_zjet_breit.append(ev_bzjet)

    print(f"Writing output to {flags.output}")
    event_tree = {
        "weight": np.array(out_weight, dtype=np.float32),
        "Q2":     np.array(out_Q2,     dtype=np.float32),
        "x":      np.array(out_x,      dtype=np.float32),
        "y":      np.array(out_y,      dtype=np.float32),
        "njets":  np.array(out_njets,  dtype=np.int32),
    }
    jet_tree = {
        "jet_pt":    out_jet_pt,
        "jet_tau10": out_jet_tau10,
        "zjet":      out_zjet,
        "deltaphi":  out_deltaphi,
    }
    if do_breit:
        jet_tree["jet_breit_pt"] = out_breit_pt
        jet_tree["zjet_breit"]   = out_zjet_breit

    with uproot.recreate(flags.output) as fout:
        fout["events"] = event_tree
        fout["jets"]   = jet_tree

    print("Done.")


if __name__ == "__main__":
    main()
