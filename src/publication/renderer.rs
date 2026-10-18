//! Window-independent, instanced raster rendering. Supersampling includes text and annotations.

use std::{collections::BTreeMap, sync::mpsc};

use egui::{Align2, Color32, FontId, Pos2, Stroke};
use graphics::{Entity, Handedness, Scene};
use lin_alg::f32::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use super::{Figure, Measurement, Settings};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 3] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4];
const INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 9] = wgpu::vertex_attr_array![
    3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4,
    7 => Float32x4, 8 => Float32x4, 9 => Float32x4, 10 => Float32x4, 11 => Float32x4
];

fn floats(bytes: &mut Vec<u8>, values: impl IntoIterator<Item = f32>) {
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
}

fn instance(e: &Entity, bytes: &mut Vec<u8>) {
    let scale = e
        .scale_partial
        .unwrap_or(Vec3::new(e.scale, e.scale, e.scale));
    let pivot = e.pivot.unwrap_or(Vec3::new_zero());
    let model = Mat4::new_translation(e.position)
        * Mat4::new_translation(pivot)
        * e.orientation.to_matrix()
        * Mat4::new_translation(-pivot)
        * Mat4::new_scaler_partial(scale);
    let normal = e.orientation.to_matrix()
        * Mat4::new_scaler_partial(Vec3::new(1. / scale.x, 1. / scale.y, 1. / scale.z));
    bytes.extend_from_slice(&model.to_bytes());
    bytes.extend_from_slice(&normal.to_bytes());
    floats(bytes, [e.color.0, e.color.1, e.color.2, e.opacity]);
}

fn texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Figure target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

struct Batch {
    mesh: usize,
    first: u32,
    count: u32,
    transparent: bool,
}

pub(super) fn render(
    mut scene: Scene,
    settings: Settings,
    measurements: &[Measurement],
) -> Result<Figure, String> {
    settings.validate()?;
    if !scene.gaussians.is_empty() {
        return Err("Gaussian volume rendering is not supported for figures yet. Switch to a mesh representation first.".into());
    }
    let width = settings.width * settings.samples;
    let height = settings.height * settings.samples;
    scene.camera.aspect = width as f32 / height as f32;
    scene.camera.update_proj_mat();
    if measurements
        .iter()
        .any(|m| !scene.camera.in_view(m.a).0 || !scene.camera.in_view(m.b).0)
    {
        return Err("A distance endpoint is outside the figure. Reframe the scene or change the aspect ratio to include both endpoints.".into());
    }

    let gpu = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(gpu.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .map_err(|e| format!("No offscreen GPU adapter: {e}"))?;
    let max_dimension = adapter.limits().max_texture_dimension_2d;
    if width > max_dimension || height > max_dimension {
        return Err(format!(
            "GPU limit is {max_dimension} pixels per supersampled dimension. Reduce size or antialiasing."
        ));
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: wgpu::Limits {
            max_texture_dimension_2d: max_dimension,
            ..Default::default()
        },
        ..Default::default()
    }))
    .map_err(|e| format!("Could not initialize figure renderer: {e}"))?;

    let buffer = |label, contents: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage,
        })
    };
    let mut uniforms = Vec::new();
    uniforms
        .extend_from_slice(&(scene.camera.proj_mat.clone() * scene.camera.view_mat()).to_bytes());
    uniforms.extend_from_slice(&scene.camera.view_mat().to_bytes());
    floats(
        &mut uniforms,
        [
            settings.ambient,
            settings.key,
            settings.fill,
            settings.specular,
        ],
    );
    floats(
        &mut uniforms,
        [
            settings.light_angle.to_radians(),
            settings.occlusion,
            settings.outlines,
            height as f32 / 900.,
        ],
    );
    floats(&mut uniforms, [scene.camera.near, scene.camera.far, 0., 0.]);
    let globals = buffer(
        "Figure camera and lighting",
        &uniforms,
        wgpu::BufferUsages::UNIFORM,
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Figure shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("figure.wgsl").into()),
    });
    let layouts = [
        Some(wgpu::VertexBufferLayout {
            array_stride: 40,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: 144,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &INSTANCE_ATTRIBUTES,
        }),
    ];
    let make_pipeline = |transparent: bool, post: bool| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Figure pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(if post { "vs_post" } else { "vs_main" }),
                compilation_options: Default::default(),
                buffers: if post { &[] } else { &layouts },
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(if post { "fs_post" } else { "fs_main" }),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: (!post).then_some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                front_face: if scene.camera.handedness == Handedness::Right {
                    wgpu::FrontFace::Cw
                } else {
                    wgpu::FrontFace::Ccw
                },
                cull_mode: Some(wgpu::Face::Back).filter(|_| !post),
                ..Default::default()
            },
            depth_stencil: (!post).then_some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(!transparent),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    };
    let opaque_pipeline = make_pipeline(false, false);
    let transparent_pipeline = make_pipeline(true, false);
    let post_pipeline = make_pipeline(false, true);
    let bind_globals = |pipeline: &wgpu::RenderPipeline| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            }],
        })
    };
    let opaque_bind = bind_globals(&opaque_pipeline);
    let transparent_bind = bind_globals(&transparent_pipeline);

    let mut opaque = BTreeMap::<usize, Vec<&Entity>>::new();
    let mut transparent = Vec::new();
    let mesh_transparency: Vec<bool> = scene
        .meshes
        .iter()
        .map(|mesh| {
            mesh.vertices
                .iter()
                .any(|v| v.color.is_some_and(|c| c.3 > 0 && c.3 < 255))
        })
        .collect();
    for e in &scene.entities {
        if e.opacity <= 0. {
            continue;
        }
        if e.mesh >= scene.meshes.len() {
            return Err("Scene contains an invalid mesh reference.".into());
        }
        let scale = e
            .scale_partial
            .unwrap_or(Vec3::new(e.scale, e.scale, e.scale));
        if [scale.x, scale.y, scale.z]
            .iter()
            .any(|s| !s.is_finite() || s.abs() < 1e-8)
        {
            continue;
        }
        if e.opacity < 1. || mesh_transparency[e.mesh] {
            transparent.push(e);
        } else {
            opaque.entry(e.mesh).or_default().push(e);
        }
    }
    // Preserve depth order across mesh types for translucent surfaces and representation layers.
    let view_depth = |e: &Entity| {
        scene
            .camera
            .orientation
            .inverse()
            .rotate_vec(e.position - scene.camera.position)
            .z
    };
    transparent.sort_by(|a, b| view_depth(b).total_cmp(&view_depth(a)));
    let mut instance_bytes = Vec::new();
    let mut batches = Vec::new();
    for (mesh, entities) in opaque {
        let first = (instance_bytes.len() / 144) as u32;
        for e in &entities {
            instance(e, &mut instance_bytes);
        }
        batches.push(Batch {
            mesh,
            first,
            count: entities.len() as u32,
            transparent: false,
        });
    }
    for e in transparent {
        let first = (instance_bytes.len() / 144) as u32;
        instance(e, &mut instance_bytes);
        batches.push(Batch {
            mesh: e.mesh,
            first,
            count: 1,
            transparent: true,
        });
    }
    if instance_bytes.len() as u64 > device.limits().max_buffer_size {
        return Err("Scene is too large for the GPU instance buffer.".into());
    }
    // Empty scenes are useful for annotation-only panels; wgpu still needs a nonempty buffer.
    if instance_bytes.is_empty() {
        instance_bytes.resize(144, 0);
    }
    let instances = buffer(
        "Figure instances",
        &instance_bytes,
        wgpu::BufferUsages::VERTEX,
    );
    let mut meshes = BTreeMap::new();
    for batch in &batches {
        if meshes.contains_key(&batch.mesh) {
            continue;
        }
        let mesh = &scene.meshes[batch.mesh];
        if mesh.vertices.is_empty() || mesh.indices.is_empty() {
            continue;
        }
        let mut vertices = Vec::with_capacity(mesh.vertices.len() * 40);
        for v in &mesh.vertices {
            floats(&mut vertices, v.position);
            floats(&mut vertices, [v.normal.x, v.normal.y, v.normal.z]);
            let color = v.color.unwrap_or((0, 0, 0, 0));
            floats(
                &mut vertices,
                [color.0, color.1, color.2, color.3].map(|c| c as f32 / 255.),
            );
        }
        let mut indices = Vec::with_capacity(mesh.indices.len() * 4);
        for &index in &mesh.indices {
            if index >= mesh.vertices.len() || index > u32::MAX as usize {
                return Err("Invalid mesh index.".into());
            }
            indices.extend_from_slice(&(index as u32).to_ne_bytes());
        }
        if vertices.len().max(indices.len()) as u64 > device.limits().max_buffer_size {
            return Err("Scene mesh exceeds the GPU buffer limit.".into());
        }
        meshes.insert(
            batch.mesh,
            (
                buffer("Figure vertices", &vertices, wgpu::BufferUsages::VERTEX),
                buffer("Figure indices", &indices, wgpu::BufferUsages::INDEX),
                mesh.indices.len() as u32,
            ),
        );
    }

    let color = texture(&device, width, height, FORMAT);
    let color_view = color.create_view(&Default::default());
    let depth = texture(&device, width, height, wgpu::TextureFormat::Depth32Float);
    let depth_view = depth.create_view(&Default::default());
    let output = texture(&device, width, height, FORMAT);
    let output_view = output.create_view(&Default::default());
    let post_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &post_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: globals.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&color_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&depth_view),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let background = if settings.transparent {
        wgpu::Color::TRANSPARENT
    } else {
        wgpu::Color {
            r: settings.background[0] as f64,
            g: settings.background[1] as f64,
            b: settings.background[2] as f64,
            a: 1.,
        }
    };
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Figure geometry"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_vertex_buffer(1, instances.slice(..));
        for batch in &batches {
            let Some((vertices, indices, count)) = meshes.get(&batch.mesh) else {
                continue;
            };
            pass.set_pipeline(if batch.transparent {
                &transparent_pipeline
            } else {
                &opaque_pipeline
            });
            pass.set_bind_group(
                0,
                if batch.transparent {
                    &transparent_bind
                } else {
                    &opaque_bind
                },
                &[],
            );
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..*count, 0, batch.first..batch.first + batch.count);
        }
    }
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Figure depth shading"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &output_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&post_pipeline);
        pass.set_bind_group(0, &post_bind, &[]);
        pass.draw(0..3, 0..1);
    }
    draw_labels(
        &scene,
        &settings,
        measurements,
        &device,
        &queue,
        &mut encoder,
        &output_view,
    );

    let padded_row = (width * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Figure readback"),
        size: padded_row as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let (tx, rx) = mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
    let data = readback
        .slice(..)
        .get_mapped_range()
        .map_err(|e| e.to_string())?;
    let rgba = downsample(&data, padded_row as usize, &settings);
    drop(data);
    readback.unmap();
    Ok(Figure { settings, rgba })
}

fn draw_labels(
    scene: &Scene,
    settings: &Settings,
    measurements: &[Measurement],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
) {
    let width = settings.width * settings.samples;
    let height = settings.height * settings.samples;
    // A fixed logical panel height keeps typography proportional and independent of the window.
    let ppp = height as f32 / 900.;
    let logical_width = width as f32 / ppp;
    let ctx = egui::Context::default();
    ctx.set_pixels_per_point(ppp);
    let project = |p| {
        let (visible, ndc) = scene.camera.in_view(p);
        visible.then_some(Pos2::new(
            (ndc.0 + 1.) * 0.5 * logical_width,
            (1. - ndc.1) * 450.,
        ))
    };
    let color = Color32::from_rgb(
        settings.label_color[0],
        settings.label_color[1],
        settings.label_color[2],
    );
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                egui::vec2(logical_width, 900.),
            )),
            ..Default::default()
        },
        |ui| {
            let painter = ui.painter();
            if settings.labels {
                for e in &scene.entities {
                    if let Some(label) = &e.overlay_text
                        && let Some(pos) = project(e.position + Vec3::new(0., 0.05 * e.scale, 0.))
                    {
                        painter.text(
                            pos,
                            Align2::CENTER_BOTTOM,
                            &label.text,
                            FontId::new(
                                label.size * settings.label_scale,
                                label.font_family.clone(),
                            ),
                            color,
                        );
                    }
                }
            }
            for m in measurements {
                if let (Some(a), Some(b)) = (project(m.a), project(m.b)) {
                    let length = a.distance(b);
                    let steps = (length / 7.).ceil().max(1.) as usize;
                    for i in (0..steps).step_by(2) {
                        painter.line_segment(
                            [
                                a.lerp(b, i as f32 / steps as f32),
                                a.lerp(b, (i + 1).min(steps) as f32 / steps as f32),
                            ],
                            Stroke::new(1.5, color),
                        );
                    }
                    painter.text(
                        a.lerp(b, 0.5) - egui::vec2(0., 4.),
                        Align2::CENTER_BOTTOM,
                        format!("{:.3} Å", (m.b - m.a).magnitude()),
                        FontId::proportional(16. * settings.label_scale),
                        color,
                    );
                }
            }
        },
    );
    let primitives = ctx.tessellate(output.shapes, ppp);
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [width, height],
        pixels_per_point: ppp,
    };
    let mut renderer = egui_wgpu::Renderer::new(
        device,
        FORMAT,
        egui_wgpu::RendererOptions {
            dithering: false,
            ..Default::default()
        },
    );
    for (id, delta) in &output.textures_delta.set {
        for change in delta {
            renderer.update_texture(device, queue, *id, change);
        }
    }
    output.textures_delta.clear();
    let commands = renderer.update_buffers(device, queue, encoder, &primitives, &screen);
    queue.submit(commands);
    let mut pass = encoder
        .begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Figure labels and measurements"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        })
        .forget_lifetime();
    renderer.render(&mut pass, &primitives, &screen);
}

fn linear(v: u8) -> f32 {
    let v = v as f32 / 255.;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb(v: f32) -> u8 {
    let v = v.clamp(0., 1.);
    let v = if v <= 0.0031308 {
        12.92 * v
    } else {
        1.055 * v.powf(1. / 2.4) - 0.055
    };
    (v * 255.).round() as u8
}

fn downsample(data: &[u8], row: usize, settings: &Settings) -> Vec<u8> {
    use rayon::prelude::*;
    let mut rgba = vec![0; settings.width as usize * settings.height as usize * 4];
    let ss = settings.samples as usize;
    let lut: Vec<f32> = (0..=255).map(|v| linear(v as u8)).collect();
    rgba.par_chunks_exact_mut(4)
        .enumerate()
        .for_each(|(pixel, out)| {
            let x = pixel % settings.width as usize;
            let y = pixel / settings.width as usize;
            let mut sum = [0.; 4];
            for dy in 0..ss {
                for dx in 0..ss {
                    let i = (y * ss + dy) * row + (x * ss + dx) * 4;
                    for c in 0..3 {
                        sum[c] += lut[data[i + c] as usize];
                    }
                    sum[3] += data[i + 3] as f32 / 255.;
                }
            }
            // Average premultiplied linear color, then unpremultiply for straight-alpha PNG.
            // This avoids both dark and white fringes on transparent antialiased geometry.
            if sum[3] > 0. {
                for c in 0..3 {
                    out[c] = srgb(sum[c] / sum[3]);
                }
                out[3] = (sum[3] / (ss * ss) as f32 * 255.).round() as u8;
            }
        });
    rgba
}
