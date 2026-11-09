//! Scrolled lists that share a page's spare height, so they grow with the window.

use egui::{ScrollArea, Ui};

/// Keeps rounding from showing the page's scroll bar when its contents just fit.
const PAGE_MARGIN: f32 = 1.;

/// Smaller changes to a list's height don't warrant another frame.
const HEIGHT_EPS: f32 = 0.5;

/// One list, as of the last frame.
#[derive(Clone, Copy)]
struct List {
    salt: &'static str,
    /// Height before any spare height is shared out.
    base: f32,
    content_height: f32,
    height: f32,
}

impl List {
    fn unmet(&self) -> f32 {
        self.content_height - self.height
    }
}

#[derive(Clone, Default)]
struct PageState {
    lists: Vec<List>,
    /// Height of everything on the page but its lists.
    fixed_height: f32,
}

/// Passed to a [`page`]'s contents, to show its lists.
pub(super) struct PageLists {
    allotted: Vec<List>,
    shown: Vec<List>,
    visible_height: f32,
}

impl PageLists {
    /// A scrolled list, `base` tall, or taller if the page has height to spare. Shrinks to short
    /// contents.
    pub(super) fn show<R>(
        &mut self,
        ui: &mut Ui,
        salt: &'static str,
        base: f32,
        add_contents: impl FnOnce(&mut Ui) -> R,
    ) -> R {
        let allotted = self
            .allotted
            .iter()
            .find(|list| list.salt == salt)
            .map_or(base, |list| list.height);

        let max_height = allotted.max(base);

        // egui would otherwise squeeze a list that's past the bottom of the page.
        let output = ScrollArea::vertical()
            .id_salt(salt)
            .max_height(max_height)
            .min_scrolled_height(max_height)
            .show(ui, add_contents);

        // `inner_rect` is from before the list shrinks to its contents.
        let content_height = output.content_size.y;
        let height = output.inner_rect.height().min(content_height);

        self.shown.push(List {
            salt,
            base,
            content_height,
            height,
        });

        // Clipped while in a collapsing header that's animating.
        let top = output.inner_rect.top();
        self.visible_height += (ui.clip_rect().bottom() - top).clamp(0., height);

        output.inner
    }
}

/// Set each list to its base height, or its content height if shorter, then share out what's left
/// of `budget` evenly. Lists that fill up pass the remainder to the others.
fn allot(lists: &mut [List], budget: f32) {
    for list in lists.iter_mut() {
        list.height = list.base.min(list.content_height);
    }

    let mut spare = budget - PAGE_MARGIN - lists.iter().map(|list| list.height).sum::<f32>();

    let mut order: Vec<usize> = (0..lists.len()).collect();
    order.sort_by(|&a, &b| lists[a].unmet().total_cmp(&lists[b].unmet()));

    for (done, &i) in order.iter().enumerate() {
        let share = (spare / (order.len() - done) as f32).max(0.);
        let growth = share.min(lists[i].unmet());

        lists[i].height += growth;
        spare -= growth;
    }
}

/// A page filling the height left in `ui`, and scrolled only if it doesn't fit. Its height doesn't
/// depend on its contents, so a list growing to fit the window can't in turn grow the window.
pub(super) fn page<R>(
    ui: &mut Ui,
    salt: &'static str,
    add_contents: impl FnOnce(&mut Ui, &mut PageLists) -> R,
) -> R {
    let id = ui.id().with(salt);
    let viewport_height = ui.available_height();

    let mut prev: PageState = ui.data(|data| data.get_temp(id)).unwrap_or_default();
    allot(&mut prev.lists, viewport_height - prev.fixed_height);

    let mut lists = PageLists {
        allotted: prev.lists,
        shown: Vec::new(),
        visible_height: 0.,
    };

    let output = ScrollArea::vertical()
        .id_salt(salt)
        .auto_shrink(false)
        .show(ui, |ui| add_contents(ui, &mut lists));

    let fixed_height = output.content_size.y - lists.visible_height;

    let mut next = lists.shown.clone();
    allot(&mut next, viewport_height - fixed_height);

    let settled = next
        .iter()
        .zip(&lists.shown)
        .all(|(next, shown)| (next.height - shown.height).abs() < HEIGHT_EPS);
    if !settled {
        ui.ctx().request_repaint();
    }

    ui.data_mut(|data| {
        data.insert_temp(
            id,
            PageState {
                lists: lists.shown,
                fixed_height,
            },
        )
    });

    output.inner
}
