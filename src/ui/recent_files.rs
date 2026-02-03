use egui::Color32;
pub const NUM_TO_SHOW: usize = 30;

// todo: Return a color too
pub fn recentness_descrip(age_min: i64) -> (String, Color32) {
    match age_min {
        0..=30 => ("Active".to_string(), Color32::GREEN),
        31..=480 => ("Today".to_string(), Color32::LIGHT_GREEN),
        481..=10_080 => ("Recent".to_string(), Color32::YELLOW),
        _ => ("Older".to_string(), Color32::GRAY),
    }
}
