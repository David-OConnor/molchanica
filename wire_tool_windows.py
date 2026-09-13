from pathlib import Path
root = Path(__file__).resolve().parent
for module, struct, fn, title, default, choices, flag in [
    ('structure_pred', 'StructurePredUi', 'structure_prediction_window', 'Structure prediction and co-folding', 'OpenDde', 'Tool::OpenDde, Tool::Boltz2, Tool::Chai1, Tool::EsmFold2', 'structure_pred'),
    ('sequence_pred', 'SequencePredUi', 'sequence_prediction_window', 'Sequence design and scoring', 'ProteinMpnn', 'Tool::ProteinMpnn, Tool::LigandMpnn', 'sequence_pred'),
    ('rfdiffusion3', 'Rfd3Ui', 'rfdiffusion3_window', 'RFdiffusion3 backbone generation', 'RfDiffusion3', 'Tool::RfDiffusion3', 'rfd3'),
]:
    callbacks = '''
impl StructurePredUi {
    // Sequence convenience actions still report completion through the main worker queue.
    pub(crate) fn finish_prediction(&mut self) {}
    pub(crate) fn mark_complete(&mut self, message: String) { self.message = Some(message); }
}
''' if module == 'structure_pred' else ''
    extra = 'message: Option<String>,' if module == 'structure_pred' else ''
    extra_init = 'message: None,' if module == 'structure_pred' else ''
    extra_draw = 'if let Some(message) = &state.ui.structure_pred.message { ui.label(message); }' if module == 'structure_pred' else ''
    code = f'''//! {title}, using the shared bio_tools form and runner.
use egui::Ui;
use graphics::{{EngineUpdates, Scene}};
use crate::{{external_tools::Tool, state::State, util::handle_err}};
use super::{{close_btn, tool_runner::ToolWindow}};

pub(crate) struct {struct} {{
    window: ToolWindow,
    {extra}
}}

impl Default for {struct} {{
    fn default() -> Self {{ Self {{ window: ToolWindow::new(Tool::{default}), {extra_init} }} }}
}}
{callbacks}
pub(in crate::ui) fn {fn}(state: &mut State, scene: &mut Scene, updates: &mut EngineUpdates, ui: &mut Ui) {{
    ui.horizontal(|ui| {{ ui.heading("{title}"); close_btn(ui, &mut state.ui.popup.{flag}); }});
    {extra_draw}
    let load = state.ui.{flag}.window.draw(&[{choices}], &state.peptides, ui);
    if let Some(path) = load && let Err(error) = state.open_file(&path, scene, updates) {{
        handle_err(&mut state.ui, error.to_string());
    }}
}}
'''
    (root / f'src/ui/popup/{module}.rs').write_text(code, encoding='utf-8')
p = root / 'src/ui/popup/mod.rs'
s = p.read_text(encoding='utf-8').replace('pub(crate) mod structure_pred;', 'pub(crate) mod structure_pred;\nmod tool_runner;')
s = s.replace('structure_pred::structure_prediction_window(state, ui)', 'structure_pred::structure_prediction_window(state, scene, updates, ui)')
s = s.replace('sequence_pred::sequence_prediction_window(state, ui)', 'sequence_pred::sequence_prediction_window(state, scene, updates, ui)')
p.write_text(s, encoding='utf-8')
print('Connected all three popups to the shared form and results browser.')
