//! Shared GHS displays for molecule properties and synthesis reaction cards.

use bio_apis::pubchem::SafetyData;
use egui::{ScrollArea, Ui, vec2};

pub(super) fn safety_summary(data: Option<&SafetyData>) -> String {
    match data {
        Some(data) => {
            let labels: Vec<_> = data
                .pictograms()
                .into_iter()
                .map(|(_, label)| label)
                .collect();
            if labels.is_empty() {
                "No pictograms reported".to_owned()
            } else {
                labels.join("; ")
            }
        }
        None => "GHS unavailable".to_owned(),
    }
}

pub(super) fn safety_badges(data: Option<&SafetyData>, ui: &mut Ui) {
    let Some(data) = data else {
        ui.weak("GHS unavailable")
            .on_hover_text("No GHS classification is available. Safety is unknown.");
        return;
    };
    pictograms(data, 22.0, false, ui);
}

/// White SVG symbols from bio_files are cached as egui textures under stable local URIs.
pub(super) fn pictograms(data: &SafetyData, size: f32, labels: bool, ui: &mut Ui) {
    egui_extras::install_image_loaders(ui.ctx());
    let pictograms = data.pictograms();
    if pictograms.is_empty() {
        ui.weak("No pictograms reported")
            .on_hover_ui(|ui| safety_details(data, ui));
    }
    if labels {
        for (code, label) in pictograms {
            ui.horizontal_wrapped(|ui| {
                pictogram(data, code, label, size, ui);
                ui.label(label);
            });
        }
        ui.hyperlink_to("PubChem GHS", &data.pubchem_url);
    } else {
        ui.horizontal_wrapped(|ui| {
            for (code, label) in pictograms {
                pictogram(data, code, label, size, ui);
            }
            ui.hyperlink_to("GHS", &data.pubchem_url)
                .on_hover_text(safety_summary(Some(data)));
        });
    }
}

fn pictogram(data: &SafetyData, code: &str, label: &str, size: f32, ui: &mut Ui) {
    let Some(bytes) = bio_files::pubchem::ghs_pictogram(code) else {
        return;
    };
    ui.add(
        egui::Image::from_bytes(format!("bytes://ghs-white/{code}.svg"), bytes)
            .fit_to_exact_size(vec2(size, size)),
    )
    .on_hover_ui(|ui| {
        ui.strong(format!("{code}: {label}"));
        safety_details(data, ui);
    });
}

pub(super) fn safety_details(data: &SafetyData, ui: &mut Ui) {
    ui.set_max_width(ui.available_width().min(480.0));
    ui.label(safety_summary(Some(data)));
    if let Some(signal) = &data.signal_word {
        ui.strong(format!("Signal word: {signal}"));
    }
    if let Some(date) = &data.retrieved_on {
        ui.weak(format!("PubChem retrieved: {date}"));
    }
    ui.weak(
        "Union of contributor reports; forms and mixtures may differ. \
        Material hazards do not quantify reaction risk. Check the supplier SDS.",
    );
    ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
        for statement in &data.hazard_statements {
            ui.label(statement);
        }
        ui.separator();
        for source in &data.sources {
            if source.url.is_empty() {
                ui.weak(&source.name);
            } else {
                ui.hyperlink_to(&source.name, &source.url);
            }
            if !source.subject.is_empty() {
                ui.weak(&source.subject);
            }
        }
    });
}
