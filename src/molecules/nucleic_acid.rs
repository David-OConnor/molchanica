// todo: You may be nissing, on G, the H on H1 (WOuld be H1)

//! For setting up and rendering nucleic acids: DNA and RNA. This module loads atom positions
//! for each base from Amber templates, and positions atoms to be geometrically consistent,
//! and realistic.
//!
//! Ref pic: https://upload.wikimedia.org/wikipedia/commons/4/4c/DNA_Structure%2BKey%2BLabelled.pn_NoBB.png
//!
// todo: Load Amber FF params for nucleic acids.

use std::{collections::HashMap,fmt::Display, io,};
use std::f64::consts::TAU;
use bincode::{Decode, Encode};
use bio_files::{
    BondType, ResidueEnd, ResidueType,
    mol_templates::{TemplateData, load_templates},
};
use dynamics::{
    params::{OL24_LIB, RNA_LIB},
};
use lin_alg::f64::{Quaternion, Vec3, Y_VEC};
use na_seq::{
    AminoAcid,
    Element::*,
    Nucleotide::{self, *},
};

use crate::{
    molecules::{
        Atom, Bond, MolGenericRef, MolGenericTrait, MolType, MoleculePeptide, Residue,
        common::MoleculeCommon,
    },
};
use crate::util::rotate_atoms_about_point;

// Axial rise; height difference between two consecutive bases.
// This is a suitable default for B-DNA.
const RISE: f64 = 3.4;

// ~10.5 bp per turn, so ~34 Å per helical turn (10.5 × 3.4)
const TWIST: f64 = 34.0_f64.to_radians();

// Used for aligning bases with each other. These are distances between the heavy atoms; not the H.
// These are all div2. First listed name is for first listed NT. I.e. A: N1 to T: N3.
const H_BOND_AT_N1_N3_DIV2: f64 = 2.85 / 2.;
const H_BOND_AT_N6_O4_DIV2: f64 = 2.81 / 2.;
// todo: Update these A/R
const H_BOND_CG_N3_N1_DIV2: f64 = 2.83 / 2.;
const H_BOND_CG_N4_O6_DIV2: f64 = 2.71 / 2.;
const H_BOND_CG_O2_N2_DIV2: f64 = 2.82 / 2.;


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


fn atom_pos_from_name(atoms: &[Atom], name: &str) -> Vec3 {
    for atom in atoms {
        if atom.type_in_res_general.as_deref() == Some(name) {
            return atom.posit;
        }
    }
    panic!("Atom of name {} not found", name);
}

/// We use this for finding our pivot to align to the helix centroid (ish),
/// and rotate around for the twist. We pick the atom closest to the rotation
/// point, then offset it slightly.
fn base_center_ref(atoms_base: &[Atom], nt: Nucleotide) -> Vec3 {
    let (name_heavy, name_dir_ref, offset) = match nt {
        A => ("N1", "C4", -H_BOND_AT_N1_N3_DIV2), // The atom across the ring.
        T => ("N3", "H3", H_BOND_AT_N1_N3_DIV2),
        C => ("N3", "C6", -H_BOND_CG_N3_N1_DIV2), // The atom across the ring.
        G => ("N1", "H1", H_BOND_CG_N3_N1_DIV2),
    };
    let heavy = atom_pos_from_name(atoms_base, name_heavy);
    let dir_ref = atom_pos_from_name(atoms_base, name_dir_ref);

    let dir = (dir_ref - heavy).to_normalized();
    heavy + dir * offset
}

/// Find the normal vector to the plane of the base atoms.
fn find_base_plane_norm(template: &TemplateData) -> Vec3 {
    // These are arbitrary base atom labels shared by all NTs.
    let n1 = template.find_atom_by_name("N1").unwrap();
    let c2 = template.find_atom_by_name("C2").unwrap();
    let c3 = template.find_atom_by_name("C4").unwrap();

    let v1 = n1.posit - c2.posit;
    let v2 = c3.posit - c2.posit;
    v1.cross(v2).to_normalized()
}

/// Aligns bases using a set of geometric transformations. Places them in the way A-T, and C-G bases
/// are aligned  with H bonds. Returns the positions of the complementary base, with the alignment.
///
/// This positions relative to the or original base.
fn align_bases(
    template: &TemplateData,
    templ_comp: &TemplateData,
    // nt of strand A.
    nt: Nucleotide,
    na_type: NucleicAcidType,
    atoms_base: &[Atom],
    plane_norm: Vec3,
    helix_angle: f64,
) -> Vec<Vec3> {
    let nt_comp = nt.complement();

    let (mut atoms_base_comp, bonds_base_comp) = base_from_template(templ_comp, nt_comp, na_type);

    // An arbitrary anchor.
    // let n1 = template.find_atom_by_name("N1").unwrap();

    // Position the bases relative to each other, based on the pairing.

    // Rotate the complementary base plane onto the template base plane.
    {
        let plane_norm_compl = find_base_plane_norm(template);

        // The negative sign here is important; otherwise the planes will be in the opposite
        // direction relative to each other than they should be.
        let plane_aligner = Quaternion::from_unit_vecs(plane_norm_compl, -plane_norm);

        let n1_comp = templ_comp.find_atom_by_name("N1").unwrap(); // This anchor is arbitrary.
        rotate_atoms_about_point(&mut atoms_base_comp, n1_comp.posit, plane_aligner);
    }

    // Find these H-bond mid points, for one of the H bonds. Set their positions
    // equal to each other.
    let ctr_ref = base_center_ref(atoms_base, nt);
    let ctr_ref_comp = base_center_ref(&atoms_base_comp, nt_comp);

    // This shifts the (already plane-matched) complementary base so the H bonds are set up
    // between the two *center* donor/acceptor pairs. Also handles the plane-shift maneuver
    // along the helix axis.
    let offset = ctr_ref - ctr_ref_comp;
    for a in &mut atoms_base_comp {
        a.posit += offset;
    }

    // Now, rotate around this common H bond center, along the plane, until the rest of the
    // geometry lines up. (Most visibly noted by the 2 or 3 H bonds). This approach of choosing the rotation angle is inelegant, and
    // is hard-coded to the template values.
    let amt = match nt {
        A => 25.,
        T => -25.,
        C => -25.,
        G => 25.,
    }  * TAU / 32. + helix_angle;

    let aligner = Quaternion::from_axis_angle(plane_norm, amt);
    rotate_atoms_about_point(&mut atoms_base_comp, ctr_ref, aligner);

    atoms_base_comp.iter().map(|a| a.posit).collect()
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
) -> io::Result<(Vec<Atom>, Vec<Bond>, Vec<Residue>)> {
    let mut atoms_out = Vec::new();
    let mut bonds_out = Vec::new();
    let mut res_out = Vec::new();

    let mut res_i: usize = 0;
    let mut atom_sn_offset = 0;

    let helix_axis = Y_VEC;

    for (i_nt, &nt) in seq.iter().enumerate() {
        let is_first = i_nt == 0;
        let is_last = i_nt + 1 == seq.len();

        let mut res = Residue {
            serial_number: res_i as u32 + 1,
            res_type: ResidueType::Other(format!("Nucleotide: {nt}")),
            atom_sns: Vec::new(),
            atoms: Vec::new(),
            dihedral: None,
            end: if is_first {
                ResidueEnd::NTerminus
            } else if is_last {
                ResidueEnd::CTerminus
            } else {
                ResidueEnd::Internal
            },
        };

        let height_offset = RISE * i_nt as f64;
        let helix_angle = TWIST * i_nt as f64;

        let template = find_template(nt, na_type, is_first, is_last, templates)?;

        let (mut atoms_base, mut bonds_base) = base_from_template(template, nt, na_type);

        // Update SN.
        for atom in &mut atoms_base {
            atom.serial_number += atom_sn_offset;
            atom.residue = Some(res_out.len()); // Not -1; haven't added this residue yet.
        }
        for bond in &mut bonds_base {
            bond.atom_0_sn += atom_sn_offset;
            bond.atom_1_sn += atom_sn_offset;
        }

        atom_sn_offset += atoms_base.len() as u32;

        // Rotate the base atoms so their planes are aligned to the (arbitrarily-chosen) Y axis.
        {
            // For the A strand, as initially present in the template.
            let plane_norm = find_base_plane_norm(template);

            let plane_rotator = Quaternion::from_unit_vecs(plane_norm, helix_axis);

            let pivot = template.find_atom_by_name("N1").unwrap().posit;
            rotate_atoms_about_point(&mut atoms_base, pivot, plane_rotator);
        }

        // This is where we move the slide pivot to, and rotate around to position
        // helically. Located along the helix axis, at the correct height for this nt.
        // todo: This pivot is probably wrong. It should probably move around in a small
        // todo circle.
        let pivot = Vec3::new(0.0, height_offset, 0.0);
        // Move so that 1:  This base's plane is at the correct height. 2: Slide, along the plane axis,
        // so the base atoms are centered on the helix axis.
        {
            let posit_to_align = base_center_ref(&atoms_base, nt);
            let offset = pivot - posit_to_align;
            for atom in &mut atoms_base {
                atom.posit += offset;
            }
        }

        // Rotate, to form the helix, using the same pivot.
        {
            let helix_rot = Quaternion::from_axis_angle(helix_axis, helix_angle);
            rotate_atoms_about_point(&mut atoms_base, pivot, helix_rot);
        }

        // Populate our all-NT atoms and bonds with those of the base from this NT.
        for atom in &atoms_base {
            atoms_out.push(atom.clone());
            res.atoms.push(atoms_base.len() - 1);
            res.atom_sns.push(atom.serial_number);
        }
        for bond in bonds_base {
            bonds_out.push(bond);
        }


        // Position the opposite base.
        if strands == Strands::Double {
            let nt_comp = nt.complement();
            let template_comp = find_template(nt_comp, na_type, is_first, is_last, templates)?;
            let (mut atoms_comp, mut bonds_comp) = base_from_template(template_comp, nt_comp, na_type);

            // The plane norm is now the Y vec; align the second strand to it.
            // todo: You passing in `template` here is probably wrong, as you moved the positions!!
            // todo: Come back to this after your SS bases are correct.
            let posits_comp = align_bases(
                &template, &template_comp, nt, na_type, &atoms_base, helix_axis, helix_angle
            );

            let compl_sn_offset = 100_00; // todo for now.

            for (i, atom) in atoms_comp.iter_mut().enumerate() {
                atom.serial_number += atom_sn_offset + compl_sn_offset;
                atom.posit = posits_comp[i];
                atom.residue = Some(res_out.len())
            }

            for bond in &mut bonds_comp {
                bond.atom_0_sn += atom_sn_offset + compl_sn_offset;
                bond.atom_1_sn += atom_sn_offset + compl_sn_offset;
            }

            atom_sn_offset += atoms_comp.len() as u32;

            // We'll update atom indices at the end, synchronizing them to SN.
            for atom in atoms_comp {
                res.atoms.push(atoms_base.len() - 1);
                res.atom_sns.push(atom.serial_number);
                atoms_out.push(atom);
            }
            for bond in bonds_comp {
                bonds_out.push(bond);
            }
        }

        res_out.push(res);
    }

    // Now, update all bond indices based on serial numbers.
    for bond in &mut bonds_out {
        let atom_0 = atoms_out
            .iter()
            .position(|a| a.serial_number == bond.atom_0_sn)
            .unwrap();

        let atom_1 = atoms_out
            .iter()
            .position(|a| a.serial_number == bond.atom_1_sn)
            .unwrap();

        bond.atom_0 = atom_0;
        bond.atom_1 = atom_1;
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
            build_strands(seq, na_type, posit_5p, templates, strands, 0.)?;

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

        let ident = match (na_type, strands) {
            (NucleicAcidType::Dna, Strands::Single) => format!("DNA(ss) {}nt", seq.len()),
            (NucleicAcidType::Dna, Strands::Double) => format!("DNA(ds) {}nt", seq.len()),
            (NucleicAcidType::Rna, Strands::Single) => format!("RNA(ss) {}nt", seq.len()),
            (NucleicAcidType::Rna, Strands::Double) => format!("RNA(ds) {}nt", seq.len()),
        };

        let mut common = MoleculeCommon::new(ident, atoms, bonds, metadata, None);

        // todo: Put these in when ready.
        // NEXT_ATOM_SN.store(0, Ordering::Release);
        // common.reassign_sns();

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

/// Loads templates from Amber data built into the binary. Returns (DNA, RNA)
pub fn load_na_templates()
    -> io::Result<(HashMap<String, TemplateData>, HashMap<String, TemplateData>)> {
    let templates_dna = load_templates(OL24_LIB)?;
    let templates_rna = load_templates(RNA_LIB)?;

    Ok((templates_dna, templates_rna))
}

/// Extract the base atoms from a template. We use hard-coded atom indices for Amber.
/// todo: For RNA, substitute U for T!
pub fn base_from_template(
    template: &TemplateData,
    nt: Nucleotide,
    na_type: NucleicAcidType,
) -> (Vec<Atom>, Vec<Bond>) {
    // for at in template.atoms.iter() {
    //     println!("{nt} Template atom: {:?}", at.type_in_res_general);
    // }

    let (a_names, b_names) = match nt {
        A => {
            let a = vec![
                "N1", "N3", "N6", "N7", "N9", "C2", "C4", "C5", "C6", "C8", "H2", "H8", "H61",
                "H62",
            ];

            let b = vec![
                ("N1", "C2"),
                ("C2", "H2"),
                ("C2", "N3"),
                ("N3", "C4"),
                ("C4", "C5"),
                ("C5", "C6"),
                ("C6", "N1"),
                ("C6", "N6"),
                ("N6", "H61"),
                ("N6", "H62"),
                ("C4", "N9"),
                ("N9", "C8"),
                ("C8", "H8"),
                ("C8", "N7"),
                ("N7", "C5"),
            ];

            (a, b)
        }
        T => {
            // Note: Uracil is similar, but removes the C7 methyl group attached to C5. (Replaced with a single H5)
            let a = match na_type {
                NucleicAcidType::Dna => vec![
                    "N1", "N3", "C2", "C4", "C5", "C6", "C7", "O2", "O4", "H3", "H6", "H71", "H72",
                    "H73",
                ],
                NucleicAcidType::Rna => vec![
                    "N1", "N3", "C2", "C4", "C5", "C6", "O2", "O4", "H3", "H5", "H6",
                ]
            };

            let b = match na_type {
                NucleicAcidType::Dna => vec![
                    ("N1", "C2"),
                    ("C2", "O2"),
                    ("C2", "N3"),
                    ("N3", "H3"),
                    ("N3", "C4"),
                    ("C4", "O4"),
                    ("C4", "C5"),
                    ("C5", "C7"),
                    ("C7", "H71"),
                    ("C7", "H72"),
                    ("C7", "H73"),
                    ("C5", "C6"),
                    ("C6", "H6"),
                    ("C6", "N1"),
                ],
                NucleicAcidType::Rna => vec![
                    ("N1", "C2"),
                    ("C2", "O2"),
                    ("C2", "N3"),
                    ("N3", "H3"),
                    ("N3", "C4"),
                    ("C4", "O4"),
                    ("C4", "C5"),
                    ("C5", "H5"),
                    ("C5", "C6"),
                    ("C6", "H6"),
                    ("C6", "N1"),
                ],
            };

            (a, b)
        }
        C => {
            let a = vec![
                "N1", "N3", "N4", "C2", "C4", "C5", "C6", "O2", "H5", "H6", "H41", "H42",
            ];

            let b = vec![
                ("N1", "C2"),
                ("C2", "O2"),
                ("C2", "N3"),
                ("N3", "C4"),
                ("C4", "N4"),
                ("N4", "H41"),
                ("N4", "H42"),
                ("C4", "C5"),
                ("C5", "H5"),
                ("C5", "C6"),
                ("C6", "H6"),
                ("C6", "N1"),
            ];

            (a, b)
        }
        G => {
            let a = vec![
                "N1", "N2", "N3", "N7", "N9", "C2", "C4", "C5", "C6", "C8", "H1", "H8", "H21",
                "H22", "O6",
            ];

            let b = vec![
                ("N1", "C2"),
                ("C2", "N2"),
                ("N2", "H21"),
                ("N2", "H22"),
                ("C2", "N3"),
                ("N3", "C4"),
                ("C4", "C5"),
                ("C5", "C6"),
                ("C6", "O6"),
                ("C6", "N1"),
                ("N1", "H1"),
                //
                ("C4", "N9"),
                ("N9", "C8"),
                ("C8", "H8"),
                ("C8", "N7"),
                ("N7", "C5"),
            ];

            (a, b)
        }
    };

    // For the smalleset: C. DNA.
    let mut atoms = Vec::with_capacity(12);
    let mut bonds = Vec::with_capacity(12);

    for a in a_names {
        atoms.push(template.find_atom_by_name(a).unwrap().into());
    }

    for (a0, a1) in b_names {
        let atom_0_sn = template.find_atom_by_name(a0).unwrap().serial_number;
        let atom_1_sn = template.find_atom_by_name(a1).unwrap().serial_number;

        // We will fill out indices later.
        bonds.push(Bond {
            atom_0_sn,
            atom_1_sn,
            bond_type: BondType::Single,
            atom_0: 0,
            atom_1: 0,
            is_backbone: false,
        });
    }

    (atoms, bonds)
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
