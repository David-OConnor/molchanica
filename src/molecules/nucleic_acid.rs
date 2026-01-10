#![allow(unused)]

//! For setting up and rendering nucleic acids: DNA and RNA.

// todo: Load Amber FF params for nucleic acids.

use std::{collections::HashMap, f64::consts::TAU, fmt::Display, io};

use bincode::{Decode, Encode};
use bio_files::{
    BondType, ResidueEnd, ResidueType,
    mol_templates::{TemplateData, load_templates},
};
use dynamics::{
    find_tetra_posit_final,
    params::{OL24_LIB, RNA_LIB},
};
use lin_alg::f64::{Quaternion, Vec3, Z_VEC};
use na_seq::{
    AminoAcid,
    Element::*,
    Nucleotide::{self, *},
    seq_complement,
};

use crate::{
    molecules::{
        Atom, Bond, MolGenericRef, MolGenericTrait, MolType, MoleculePeptide, Residue,
        build_adjacency_list, common::MoleculeCommon,
    },
    util::rotate_about_point,
};

const HELIX_TWIST: f64 = TAU / 10.0; // 36°
// const HELIX_RISE: f64 = 3.4; // Å per base (visual B-DNA-ish)
const HELIX_RADIUS: f64 = 10.0; // Å (tweak to taste)

const RISE: f64 = 3.4;
const TWIST: f64 = 36.0_f64.to_radians();

#[derive(Debug, Clone, Copy, PartialEq, Default, Encode, Decode)]
pub enum NucleicAcidType {
    #[default]
    Dna,
    Rna,
}

impl Display for NucleicAcidType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::Dna => "DNA",
            Self::Rna => "RNA",
        };
        write!(f, "{v}")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Encode, Decode)]
pub enum Strands {
    Single,
    #[default]
    Double,
}

impl Display for Strands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = match self {
            Self::Single => "Single",
            Self::Double => "Double",
        };
        write!(f, "{v}")?;
        Ok(())
    }
}

/// Returns an arbitrary nucleotide combination that codes for the AA in question.
fn nts_from_aa(aa: AminoAcid) -> [Nucleotide; 3] {
    match aa {
        AminoAcid::Arg => [A, G, G],
        AminoAcid::His => [C, A, C],
        AminoAcid::Lys => [A, A, G],
        AminoAcid::Asp => [G, A, C],
        AminoAcid::Glu => [G, A, G],
        AminoAcid::Ser => [T, C, A],
        AminoAcid::Thr => [A, C, A],
        AminoAcid::Asn => [A, A, C],
        AminoAcid::Gln => [C, A, G],
        AminoAcid::Cys => [T, G, C],
        AminoAcid::Sec => [A, A, A], // todo temp. Find it.,
        AminoAcid::Gly => [G, G, A],
        AminoAcid::Pro => [C, C, A],
        AminoAcid::Ala => [G, C, A],
        AminoAcid::Val => [G, T, A],
        AminoAcid::Ile => [A, T, A],
        AminoAcid::Leu => [C, T, A],
        AminoAcid::Met => [A, T, G],
        AminoAcid::Phe => [T, T, C],
        AminoAcid::Tyr => [T, A, C],
        AminoAcid::Trp => [T, G, G],
    }
}

/// Represents a nucleic acid as a collection of atoms and bonds. Omits mol-generic fields.
#[derive(Debug, Default, Clone)]
pub struct MoleculeNucleicAcid {
    pub na_type: NucleicAcidType,
    pub common: MoleculeCommon,
    pub residues: Vec<Residue>,
    pub seq: Vec<Nucleotide>,
    // pub bonds_hydrogen: Vec<HydrogenBond>,
    /// This is in the same vein as used by nucleic acid sequencing tools
    /// for example, things like promoters, operators, RNA bind site, etc.
    /// todo: Use the same struct as PlasCAD, or similar?
    pub features: Vec<(String, (usize, usize))>,
    //     pub common_name: String,
    //     /// We use residues to denote headgroups and chains.
    //     pub residues: Vec<Residue>,
}

/// Search our library, and choose the correct template for a given nucleic acid
/// in the chain.
fn find_template(
    nt: Nucleotide,
    na_type: NucleicAcidType,
    is_first: bool,
    is_last: bool,
    templates: &HashMap<String, TemplateData>,
) -> io::Result<&TemplateData> {
    let mut nt_str = nt.to_str_upper();

    // Note: We also have, for DNA, "neutral" templates that have an N suffix.
    // We have many other templates for RNA. I'm not sure what they're for. Structural?
    let ident = match na_type {
        NucleicAcidType::Dna => {
            if is_first && !is_last {
                format!("D{nt_str}5")
            } else if is_last && !is_first {
                format!("D{nt_str}3")
            } else {
                format!("D{nt_str}")
            }
        }
        NucleicAcidType::Rna => {
            if nt_str == "T" {
                nt_str = "U".to_string();
            }

            if is_first && !is_last {
                format!("{nt_str}5")
            } else if is_last && !is_first {
                format!("{nt_str}3")
            } else {
                nt_str
            }
        }
    };

    match templates.get(&ident) {
        Some(t) => Ok(t),
        None => Err(io::Error::other(format!(
            "Unable to find the template for ident {ident}"
        ))),
    }
}

// /// Rotate all atoms in a single residue so that its bases align to a plane, and are located
// /// at the correct spacing and relative positions.
// fn align_bases(atoms: &mut [Atom], nt: Nucleotide, tgt_base_norm: Vec3, template: &TemplateData) {
//     // todo: Delegate this to a fn as required.
//     let (base_n_name, plane_0_name, plane_1_name) = match nt {
//         A | G => ("N9", "C8", "C4"),
//         C | T => ("N1", "C6", "C2"),
//     };
//
//     let base_n = template.find_atom_by_name(base_n_name).unwrap().posit;
//     let plane_0 = template.find_atom_by_name(plane_0_name).unwrap().posit;
//     let plane_1 = template.find_atom_by_name(plane_1_name).unwrap().posit;
//
//     // These are arbitrary; choose any bonds in the base; they share the same plane.
//     let base_rot_axis = {
//         let base_anchor = template.find_atom_by_name("C1'").unwrap().posit;
//         (base_anchor - base_n).to_normalized()
//     };
//
//     let base_plane_norm = {
//         // todo: QC sign/direction on this
//         let plane_bond_0 = (base_n - plane_0).to_normalized();
//         let plane_bond_1 = (base_n - plane_1).to_normalized();
//         (plane_bond_0.cross(plane_bond_1)).to_normalized()
//     };
//
//     // todo: QC dir
//     // This is the shortest rotation to align the bases, but it's not along the bond in question...
//     let norm_rot = Quaternion::from_unit_vecs(tgt_base_norm, base_plane_norm);
//
//     for atom in atoms {
//     }
// }

/// Create the complementary residue for a double strand.
///
/// This works by:
/// 1. Calculating the geometry of the current strand's base (Strand A).
/// 2. Constructing the theoretical position of the complementary base (Strand B)
///    based on B-DNA parameters (C1'-C1' distance and dyad angles).
/// 3. Aligning the raw template for Strand B to this target position.
/// Create the complementary residue for a double strand.
///
/// This uses B-DNA dyad symmetry to:
/// 1. Mirror the orientation of the base so it faces "inward".
/// 2. Flip the normal so the backbone runs antiparallel and faces "outward".
fn create_paired_ds_segment(
    atoms_a: &[Atom],
    nt_a: Nucleotide,
    template_a: &TemplateData,
    template_b: &TemplateData,
) -> io::Result<(Vec<Atom>, Vec<Bond>, Residue, u32, u32)> {
    // --- 1. Get Geometry of Strand A ---
    let frame_a = BaseFrame::from_atoms(atoms_a, nt_a, template_a)
        .ok_or_else(|| io::Error::other("Could not extract BaseFrame from Strand A"))?;

    // --- 2. Construct Target Frame for Strand B (BASE-FIRST) ---

    // B-DNA-ish constants (still idealized)
    let dist_c1_c1 = 10.85;
    let bond_len_c1_n = 1.47;

    // Across-pair direction is approximately opposite of the sugar-attachment direction in-plane.
    // (C1'->N points "inward"; across to the other sugar goes outward.)
    let dir_c1a_to_c1b = (-frame_a.glyco_dir).to_normalized();
    let target_c1_b = frame_a.c1_prime + dir_c1a_to_c1b * dist_c1_c1;

    // Keep the base planes coplanar for realistic stacking.
    let target_normal_b = frame_a.normal;

    // Complement should present the opposite Watson–Crick edge direction so the WC edges meet.
    let target_wc_dir_b = (-frame_a.wc_dir).to_normalized();

    // Glycosidic direction for B should point from C1'_b toward the base interior (toward A)
    let target_glyco_dir_b = (-dir_c1a_to_c1b).to_normalized();
    let target_n_b = target_c1_b + target_glyco_dir_b * bond_len_c1_n;

    let target_frame_b = BaseFrame {
        c1_prime: target_c1_b,
        n_glyco: target_n_b,
        normal: target_normal_b,
        wc_dir: target_wc_dir_b,
        glyco_dir: target_glyco_dir_b,
    };

    // --- 3. Align Template B ---
    let nt_b = nt_a.complement();
    // Convert template atoms to a temporary Vec<Atom> for alignment calc
    let atoms_b_raw: Vec<Atom> = template_b.atoms.iter().map(|a| a.try_into().unwrap()).collect();

    let frame_b_local = BaseFrame::from_atoms(&atoms_b_raw, nt_b, template_b)
        .ok_or_else(|| io::Error::other("Could not extract BaseFrame from Template B"))?;

    let (rot, trans) = calculate_alignment(frame_b_local, target_frame_b);

    // --- 4. Generate Output ---
    let mut atoms_out = Vec::new();
    let mut local_to_global_sn = HashMap::new();
    let mut res_atom_sns = Vec::new();
    let mut res_atom_indices = Vec::new();
    let sn_offset = 20000; // Offset to distinguish Strand B from Strand A

    // Get Head (P) and Tail (O3') for backbone connectivity
    let (head_local_idx, tail_local_idx) = template_b.attach_points()?;
    let head_local_sn = template_b.atoms[head_local_idx.unwrap_or(0)].serial_number;
    let tail_local_sn = template_b.atoms[tail_local_idx.unwrap_or(0)].serial_number;

    let mut b_head_global_sn = 0;
    let mut b_tail_global_sn = 0;

    for (idx, atom_tmpl) in template_b.atoms.iter().enumerate() {
        let mut atom: Atom = atom_tmpl.try_into().unwrap();

        // Apply transformation: Rotate then Translate
        atom.posit = trans + rot.rotate_vec(atom.posit);
        atom.serial_number += sn_offset;

        if atom_tmpl.serial_number == head_local_sn { b_head_global_sn = atom.serial_number; }
        if atom_tmpl.serial_number == tail_local_sn { b_tail_global_sn = atom.serial_number; }

        local_to_global_sn.insert(atom_tmpl.serial_number, atom.serial_number);
        res_atom_sns.push(atom.serial_number);
        res_atom_indices.push(idx);
        atoms_out.push(atom);
    }

    let mut bonds_out = Vec::new();
    for bond_tmpl in &template_b.bonds {
        let mut b = Bond {
            atom_0: 0, // Resolved by caller using global indices
            atom_1: 0,
            atom_0_sn: *local_to_global_sn.get(&bond_tmpl.atom_0_sn).unwrap(),
            atom_1_sn: *local_to_global_sn.get(&bond_tmpl.atom_1_sn).unwrap(),
            bond_type: bond_tmpl.bond_type,
            is_backbone: true,
        };
        bonds_out.push(b);
    }

    let res = Residue {
        serial_number: 0, // Set by caller
        res_type: ResidueType::Other(format!("Nucleotide: {}", nt_b)),
        atom_sns: res_atom_sns,
        atoms: res_atom_indices,
        dihedral: None,
        end: ResidueEnd::Internal,
    };

    Ok((atoms_out, bonds_out, res, b_head_global_sn, b_tail_global_sn))
}

/// Build a single or double strand of DNA or RNA. If double-stranded, use the
/// base alignment geometry to define the helix shape.
fn build_strands(
    seq: &[Nucleotide],
    na_type: NucleicAcidType,
    posit_5p: Vec3,
    templates: &HashMap<String, TemplateData>,
    strands: Strands,
    helix_phase: f64,
    strand_label: &str,
) -> io::Result<(Vec<Atom>, Vec<Bond>, Vec<Residue>)> {
    let mut atoms_out = Vec::new();
    let mut bonds_out = Vec::new();
    let mut res_out = Vec::new();

    let mut res_i: usize = 0;

    let mut stack_ref_wc_dir: Option<Vec3> = None;

    // Trackers for Strand A backbone
    let mut prev_tail_sn: Option<u32> = None; // For A-strand P-O3' bond
    let mut prev_o3p = posit_5p;              // For A-strand positioning

    // Trackers for Strand B backbone (NEW)
    let mut prev_b_head_sn: Option<u32> = None; // For B-strand O3'-P bond

    for (i, &nt) in seq.iter().enumerate() {
        let is_first = i == 0;
        let is_last = i + 1 == seq.len();

        let template = find_template(nt, na_type, is_first, is_last, templates)?;
        let (head_local_i, tail_local_i) = template.attach_points()?;

        // --- PREP: Calculate Global Position for Strand A Head ---
        let head_i = match head_local_i {
            Some(i) => i,
            None => template.find_atom_i_by_name("P").unwrap_or(0),
        };

        let posit_head_local = template.atoms[head_i].posit;

        let posit_head_global = if is_first {
            prev_o3p
        } else {
            // Find tetrahedral position for Phosphate relative to previous O3'
            let o_0 = template.find_atom_by_name("OP1").unwrap().posit;
            let o_1 = template.find_atom_by_name("OP2").unwrap().posit;
            let o_2 = template.find_atom_by_name("O5'").unwrap().posit;

            let o_3p_posit = find_tetra_posit_final(posit_head_local, o_0, o_1, o_2);
            prev_o3p + o_3p_posit
        };

        let translation = posit_head_global - posit_head_local;

        // --- STEP 1: Create Strand A Segment (MOVED UP) ---
        // We must do this FIRST so `atoms_segment` is ready for Strand B logic.

        let mut res = Residue {
            serial_number: res_i as u32 + 1,
            res_type: ResidueType::Other(format!("Nucleotide: {nt}")),
            atom_sns: Vec::new(),
            atoms: Vec::new(),
            dihedral: None,
            end: if is_first { ResidueEnd::NTerminus } else if is_last { ResidueEnd::CTerminus } else { ResidueEnd::Internal },
        };

        let mut local_to_global_sn = HashMap::new();
        let mut atoms_segment = Vec::new(); // Important: Holds this residue's atoms for geometry calc

        // Capture tail info for Strand A backbone logic later
        let mut tail_global_pos: Option<Vec3> = None;
        let mut tail_global_sn: Option<u32> = None;

        // Identify the tail atom serial in the template to capture its global position
        let tail_template_sn = if !is_last {
            let t_i = tail_local_i.ok_or_else(|| io::Error::other("Missing tail attach point"))?;
            Some(template.atoms[t_i].serial_number)
        } else {
            None
        };

        for atom_template in &template.atoms {
            let mut atom: Atom = atom_template.try_into().unwrap();

            // 1. Assign Serial and Residue
            let new_sn = (atoms_out.len() as u32) + 1;
            atom.serial_number = new_sn;
            atom.residue = Some(res_i);

            // 2. Position Atom
            atom.posit += translation;
            // (Rotation logic would go here if you re-enable helix twist)

            // 3. Capture Tail Position (for next iteration of Strand A)
            if tail_template_sn == Some(atom_template.serial_number) {
                tail_global_pos = Some(atom.posit);
                tail_global_sn = Some(new_sn);
            }

            // 4. Store locally and globally
            res.atom_sns.push(new_sn);
            res.atoms.push(new_sn as usize - 1);
            local_to_global_sn.insert(atom_template.serial_number, new_sn);

            atoms_segment.push(atom.clone()); // <--- Crucial: Populates geometry for Step 2
            atoms_out.push(atom);
        }




        // --- BASE STACKING: twist residue i about its own base normal (through C1') ---
        // This makes stacking geometry driven by bases, not by backbone heuristics.

        let segment_start = atoms_out.len() - template.atoms.len();
        let segment_end = atoms_out.len();

        let frame_a_now = BaseFrame::from_atoms(&atoms_out[segment_start..segment_end], nt, template)
            .ok_or_else(|| io::Error::other("Could not extract BaseFrame from Strand A (post-place)"))?;

        if stack_ref_wc_dir.is_none() {
            stack_ref_wc_dir = Some(frame_a_now.wc_dir);
        }

        let desired_wc = {
            let ref_wc = stack_ref_wc_dir.unwrap();
            let q = Quaternion::from_axis_angle(frame_a_now.normal, helix_phase + (i as f64) * TWIST);
            q.rotate_vec(ref_wc)
        };

        let angle = signed_angle_around_axis(frame_a_now.normal, frame_a_now.wc_dir, desired_wc);
        let rot_stack = Quaternion::from_axis_angle(frame_a_now.normal, angle);

        for atom in atoms_out[segment_start..segment_end].iter_mut() {
            let rel = atom.posit - frame_a_now.c1_prime;
            atom.posit = frame_a_now.c1_prime + rot_stack.rotate_vec(rel);
        }

        // Refresh atoms_segment so Strand B pairing sees the *stacked* base geometry.
        atoms_segment = atoms_out[segment_start..segment_end].to_vec();

        // If you captured tail position earlier, recompute from the global atom SN after rotation.
        if let Some(sn) = tail_global_sn {
            if let Some(a) = atoms_out.iter().find(|a| a.serial_number == sn) {
                tail_global_pos = Some(a.posit);
            }
        }





        // Push the Residue for Strand A
        res_out.push(res);


        // --- STEP 2: Create Strand B Segment (Dependent on Step 1) ---
        if strands == Strands::Double {
            let template_complementary = find_template(nt.complement(), na_type, is_first, is_last, templates)?;

            // 2a. Generate the geometry and atoms for the complementary residue
            let (atoms_comp, mut bonds_comp, mut res_comp, b_head_sn, b_tail_sn) = create_paired_ds_segment(
                &atoms_segment,
                nt,
                template,
                &template_complementary
            )?;

            let current_comp_start_idx = atoms_out.len();
            let sn_offset = 20000; // Must match the offset in create_paired_ds_segment

            // 2b. Add Atoms to global list
            for mut atom in atoms_comp {
                // Ensure the atom knows it belongs to the residue we are about to push
                atom.residue = Some(res_out.len());
                atoms_out.push(atom);
            }

            // 2c. Fix internal bonds indices for B
            for bond in bonds_comp.iter_mut() {
                // Subtract offset to find original index in the template
                let orig_sn_0 = bond.atom_0_sn - sn_offset;
                let orig_sn_1 = bond.atom_1_sn - sn_offset;

                let local_idx_0 = template_complementary.find_atom_i_by_sn(orig_sn_0)
                    .ok_or_else(|| io::Error::other("SN not found in template B"))?;
                let local_idx_1 = template_complementary.find_atom_i_by_sn(orig_sn_1)
                    .ok_or_else(|| io::Error::other("SN not found in template B"))?;

                bond.atom_0 = current_comp_start_idx + local_idx_0;
                bond.atom_1 = current_comp_start_idx + local_idx_1;
                bonds_out.push(bond.clone());
            }

            // 2d. Fix Residue indicRes and push
            res_comp.serial_number = (seq.len() * 2 - i) as u32;
            res_comp.atoms = res_comp.atoms.iter().map(|x| x + current_comp_start_idx).collect();
            res_out.push(res_comp);

            // 2e. Fix backbone bond for B (antiparallel: 3' -> 5')
            if let Some(prev_head) = prev_b_head_sn {
                let orig_tail_sn = b_tail_sn - sn_offset;
                let tail_local_idx = template_complementary.find_atom_i_by_sn(orig_tail_sn).unwrap();
                let idx_tail = current_comp_start_idx + tail_local_idx;

                // Find the previous head in the global list
                let idx_prev_head = atoms_out.iter().position(|a| a.serial_number == prev_head)
                    .ok_or_else(|| io::Error::other("Previous B-head not found"))?;

                bonds_out.push(Bond {
                    atom_0: idx_tail,
                    atom_1: idx_prev_head,
                    atom_0_sn: b_tail_sn,
                    atom_1_sn: prev_head,
                    bond_type: BondType::Single,
                    is_backbone: true, // Uncomment if your Bond struct has this field
                });
            }

            // Update the tracker for the next residue in Strand B
            prev_b_head_sn = Some(b_head_sn);
        }

        // --- STEP 3: Add Bonds for Strand A ---

        // 3a. Intra-residue bonds (within the nucleotide)
        for bond_template in &template.bonds {
            let atom_0_sn = *local_to_global_sn.get(&bond_template.atom_0_sn).unwrap();
            let atom_1_sn = *local_to_global_sn.get(&bond_template.atom_1_sn).unwrap();

            let mut bond = bond_template.clone();
            bond.atom_0_sn = atom_0_sn;
            bond.atom_1_sn = atom_1_sn;
            // Note: Bond::from_generic likely scans `atoms_out` to find indices
            bonds_out.push(Bond::from_generic(&bond, &atoms_out)?);
        }

        // 3b. Inter-residue backbone bond (Prev O3' -> Curr P)
        if !is_first {
            let cur_head_sn = *local_to_global_sn.get(
                &template.atoms[head_i].serial_number
            ).unwrap();

            let prev_sn = prev_tail_sn.ok_or_else(|| io::Error::other("Missing prev tail sn"))?;

            let bond = bio_files::BondGeneric {
                atom_0_sn: prev_sn,
                atom_1_sn: cur_head_sn,
                bond_type: BondType::Single,
            };
            bonds_out.push(Bond::from_generic(&bond, &atoms_out)?);
        }

        // --- STEP 4: Update Iteration State ---
        res_i += 1; // Increment residue counter

        if !is_last {
            prev_o3p = tail_global_pos.ok_or_else(|| io::Error::other("Tail atom not captured"))?;
            prev_tail_sn = Some(tail_global_sn.ok_or_else(|| io::Error::other("Tail sn not captured"))?);
        }
    }

    Ok((atoms_out, bonds_out, res_out))
}

impl MoleculeNucleicAcid {
    /// Build a simple single-strand helix with a phosphate (P), sugar anchor (C4′ proxy),
    /// and a base anchor (N9 for purines, N1 for pyrimidines). Bonds:
    ///   P—S (intra), S—B (intra), and the inter-residue backbone S(i-1)—P(i).
    ///
    /// Geometry is **idealized B-DNA-like**: rise ~3.4 Å, twist 36°, with simple radial offsets.
    /// This is a minimal “it renders now” model you can extend with full atom templates later.
    /// Initializes a linear molecule.
    pub fn from_seq(
        seq: &[Nucleotide],
        na_type: NucleicAcidType,
        strands: Strands,
        posit_5p: Vec3,
        templates_dna: &HashMap<String, TemplateData>,
        templates_rna: &HashMap<String, TemplateData>,
    ) -> io::Result<Self> {
        let templates = match na_type {
            NucleicAcidType::Dna => templates_dna,
            NucleicAcidType::Rna => templates_rna,
        };

        let (mut atoms, mut bonds, mut residues) =
            build_strands(seq, na_type, posit_5p, templates, strands, 0., "strand_0")?;

        let adjacency_list = build_adjacency_list(&bonds, atoms.len());

        let mut metadata = HashMap::new();
        metadata.insert(
            "nucleic_acid_type".to_string(),
            match na_type {
                NucleicAcidType::Dna => "dna",
                NucleicAcidType::Rna => "rna",
            }
                .to_string(),
        );
        metadata.insert(
            "strands".to_string(),
            match strands {
                Strands::Single => "single",
                Strands::Double => "double",
            }
                .to_string(),
        );

        let atom_posits = atoms.iter().map(|a| a.posit).collect();

        let common = MoleculeCommon {
            ident: match (na_type, strands) {
                (NucleicAcidType::Dna, Strands::Single) => format!("DNA(ss) {}nt", seq.len()),
                (NucleicAcidType::Dna, Strands::Double) => format!("DNA(ds) {}nt", seq.len()),
                (NucleicAcidType::Rna, Strands::Single) => format!("RNA(ss) {}nt", seq.len()),
                (NucleicAcidType::Rna, Strands::Double) => format!("RNA(ds) {}nt", seq.len()),
            },
            atoms,
            bonds,
            adjacency_list,
            atom_posits,
            metadata,
            ..Default::default()
        };

        Ok(Self {
            na_type,
            common,
            residues,
            seq: seq.to_vec(),
            features: Vec::new(),
        })
    }

    /// This wrapper that extracts the AA sequence, then chooses a suitable DNA sequence.
    /// note that there are many possible combinations due to multiple codons corresponding
    /// to some AAs.
    pub fn from_peptide(
        peptide: &MoleculePeptide,
        na_type: NucleicAcidType,
        strands: Strands,
        posit_5p: Vec3,
        templates_dna: &HashMap<String, TemplateData>,
        templates_rna: &HashMap<String, TemplateData>,
    ) -> io::Result<Self> {
        let mut seq = Vec::with_capacity(&peptide.residues.len() * 3);
        for res in &peptide.residues {
            seq.push(A);
            seq.push(A);
            seq.push(A);
        }

        Self::from_seq(
            &seq,
            na_type,
            strands,
            posit_5p,
            templates_dna,
            templates_rna,
        )
    }
}


impl MolGenericTrait for MoleculeNucleicAcid {
    fn common(&self) -> &MoleculeCommon {
        &self.common
    }

    fn common_mut(&mut self) -> &mut MoleculeCommon {
        &mut self.common
    }

    fn to_ref(&self) -> MolGenericRef<'_> {
        MolGenericRef::NucleicAcid(self)
    }

    fn mol_type(&self) -> MolType {
        MolType::NucleicAcid
    }
}


/// Returns (DNA, RNA)
pub fn load_na_templates()
    -> io::Result<(HashMap<String, TemplateData>, HashMap<String, TemplateData>)> {
    let templates_dna = load_templates(OL24_LIB)?;
    let templates_rna = load_templates(RNA_LIB)?;

    Ok((templates_dna, templates_rna))
}

// ----- Helpers below for DS alignment. WIP

#[derive(Clone, Copy, Debug)]
struct BaseFrame {
    c1_prime: Vec3, // sugar anchor
    n_glyco: Vec3,  // N9 (purine) or N1 (pyrimidine)
    normal: Vec3,   // base-plane normal (unit)
    wc_dir: Vec3,   // in-plane direction pointing toward Watson–Crick edge (unit)
    glyco_dir: Vec3, // in-plane projection of (C1' -> N) (unit)
}

fn project_onto_plane(v: Vec3, n_unit: Vec3) -> Vec3 {
    v - n_unit * v.dot(n_unit)
}

impl BaseFrame {
    fn from_atoms(atoms: &[Atom], nt: Nucleotide, template: &TemplateData) -> Option<Self> {
        let (n_name, plane_0, plane_1, wc_name) = match nt {
            A | G => ("N9", "C8", "C4", "N1"),
            C | T => ("N1", "C6", "C2", "N3"),
            U => ("N1", "C6", "C2", "N3"),
        };

        let c1_idx = template.find_atom_i_by_name("C1'")?;
        let n_idx  = template.find_atom_i_by_name(n_name)?;
        let p0_idx = template.find_atom_i_by_name(plane_0)?;
        let p1_idx = template.find_atom_i_by_name(plane_1)?;
        let wc_idx = template.find_atom_i_by_name(wc_name)?;

        let c1_pos = atoms.get(c1_idx)?.posit;
        let n_pos  = atoms.get(n_idx)?.posit;
        let p0_pos = atoms.get(p0_idx)?.posit;
        let p1_pos = atoms.get(p1_idx)?.posit;
        let wc_pos = atoms.get(wc_idx)?.posit;

        // Base-plane normal (use two ring vectors anchored at glycosidic N)
        let v1 = (p0_pos - n_pos).to_normalized();
        let v2 = (p1_pos - n_pos).to_normalized();
        let mut normal = v1.cross(v2).to_normalized();

        // In-plane projected glycosidic direction (C1' -> N)
        let glyco_raw = n_pos - c1_pos;
        let mut glyco_dir = project_onto_plane(glyco_raw, normal);
        if glyco_dir.magnitude_squared() < 1.0e-8 {
            return None;
        }
        glyco_dir = glyco_dir.to_normalized();

        // In-plane Watson–Crick edge direction (N -> WC atom)
        let wc_raw = wc_pos - n_pos;
        let mut wc_dir = project_onto_plane(wc_raw, normal);
        if wc_dir.magnitude_squared() < 1.0e-8 {
            // fallback: pick something in-plane orthogonal to glyco_dir
            wc_dir = normal.cross(glyco_dir);
        }
        wc_dir = wc_dir.to_normalized();

        // Make handedness consistent (prevents random 180° flips when templates differ)
        if normal.dot(glyco_dir.cross(wc_dir)) < 0.0 {
            normal = -normal;
            wc_dir = -wc_dir;
        }

        Some(Self {
            c1_prime: c1_pos,
            n_glyco: n_pos,
            normal,
            wc_dir,
            glyco_dir,
        })
    }
}

fn signed_angle_around_axis(axis_unit: Vec3, from: Vec3, to: Vec3) -> f64 {
    let axis = axis_unit.to_normalized();

    let mut f = from - axis * from.dot(axis);
    let mut t = to   - axis * to.dot(axis);

    let f_norm = f.magnitude();
    let t_norm = t.magnitude();
    if f_norm < 1.0e-8 || t_norm < 1.0e-8 {
        return 0.0;
    }

    f = f / f_norm;
    t = t / t_norm;

    let sin = axis.dot(f.cross(t)) as f64;
    let cos = f.dot(t) as f64;
    sin.atan2(cos)
}



/// Calculate rigid transform mapping `src` base frame to `dst`, prioritizing:
/// 1) base-plane normal
/// 2) Watson–Crick in-plane direction
/// 3) anchor translation at C1'
fn calculate_alignment(src: BaseFrame, dst: BaseFrame) -> (Quaternion, Vec3) {
    // 1) Align normals
    let rot_normal = Quaternion::from_unit_vecs(src.normal, dst.normal);

    // 2) Rotate within the plane about the (already-aligned) normal to align WC edge direction
    let src_wc_rot = rot_normal.rotate_vec(src.wc_dir);
    let angle = signed_angle_around_axis(dst.normal, src_wc_rot, dst.wc_dir);
    let rot_plane = Quaternion::from_axis_angle(dst.normal, angle);

    let total_rot = rot_plane * rot_normal;

    // 3) Translate so rotated C1' lands exactly on target C1'
    let translation = dst.c1_prime - total_rot.rotate_vec(src.c1_prime);

    (total_rot, translation)
}