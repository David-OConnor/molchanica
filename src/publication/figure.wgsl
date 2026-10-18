struct Globals {
    projection_view: mat4x4<f32>,
    view: mat4x4<f32>,
    light: vec4<f32>, // ambient, key, fill, specular
    effects: vec4<f32>, // key angle, occlusion, outline width, resolution scale
    clip: vec4<f32>, // near, far
}
@group(0) @binding(0) var<uniform> g: Globals;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) model0: vec4<f32>,
    @location(4) model1: vec4<f32>,
    @location(5) model2: vec4<f32>,
    @location(6) model3: vec4<f32>,
    @location(7) normal0: vec4<f32>,
    @location(8) normal1: vec4<f32>,
    @location(9) normal2: vec4<f32>,
    @location(10) normal3: vec4<f32>,
    @location(11) tint: vec4<f32>,
}
struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec4<f32>,
}
@vertex fn vs_main(v: VertexIn) -> VertexOut {
    let model = mat4x4<f32>(v.model0, v.model1, v.model2, v.model3);
    let normal = mat4x4<f32>(v.normal0, v.normal1, v.normal2, v.normal3);
    var out: VertexOut;
    out.clip_position = g.projection_view * model * vec4<f32>(v.position, 1.0);
    out.normal = (g.view * normal * vec4<f32>(v.normal, 0.0)).xyz;
    out.color = vec4<f32>(select(v.tint.rgb, v.color.rgb, v.color.a > 0.0),
        v.tint.a * select(1.0, v.color.a, v.color.a > 0.0));
    return out;
}
@fragment fn fs_main(v: VertexOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let n = normalize(v.normal) * select(-1.0, 1.0, front);
    let key = normalize(vec3<f32>(sin(g.effects.x), 0.6, -cos(g.effects.x)));
    let fill = normalize(vec3<f32>(0.8, -0.3, -0.5));
    let diffuse = g.light.x + g.light.y * max(dot(n, key), 0.0)
        + g.light.z * max(dot(n, fill), 0.0);
    let half_dir = normalize(key + vec3<f32>(0.0, 0.0, -1.0));
    let highlight = g.light.w * pow(max(dot(n, half_dir), 0.0), 48.0);
    return vec4<f32>(clamp(v.color.rgb * diffuse + highlight, vec3<f32>(0.0), vec3<f32>(1.0)), v.color.a);
}

@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: texture_depth_2d;

@vertex fn vs_post(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}
fn linear_depth(d: f32) -> f32 {
    return g.clip.x * g.clip.y / (g.clip.y - d * (g.clip.y - g.clip.x));
}
@fragment fn fs_post(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(position.xy);
    let size = vec2<i32>(textureDimensions(scene_depth));
    let c = textureLoad(scene_color, p, 0);
    let raw = textureLoad(scene_depth, p, 0);
    if raw >= 1.0 { return c; }
    let depth = linear_depth(raw);
    let directions = array<vec2<f32>, 8>(
        vec2<f32>(1.0, 0.0), vec2<f32>(0.707, 0.707),
        vec2<f32>(0.0, 1.0), vec2<f32>(-0.707, 0.707),
        vec2<f32>(-1.0, 0.0), vec2<f32>(-0.707, -0.707),
        vec2<f32>(0.0, -1.0), vec2<f32>(0.707, -0.707));
    var occlusion = 0.0;
    var edge = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let q = clamp(p + vec2<i32>(round(directions[i] * 5.0 * g.effects.w)), vec2<i32>(0), size - 1);
        let neighbor = linear_depth(textureLoad(scene_depth, q, 0));
        let delta = depth - neighbor;
        // Bounded local contact shading; avoid dark halos from distant foreground objects.
        occlusion += smoothstep(0.1, 0.7, delta) * (1.0 - smoothstep(1.0, 4.0, delta));
        let e = clamp(p + vec2<i32>(round(directions[i] * g.effects.z * g.effects.w)), vec2<i32>(0), size - 1);
        let other = textureLoad(scene_depth, e, 0);
        edge = max(edge, select(0.0, 1.0, other >= 1.0 || abs(linear_depth(other) - depth) > 0.6));
    }
    let shading = (1.0 - g.effects.y * occlusion / 8.0)
        * (1.0 - select(0.0, edge * 0.85, g.effects.z > 0.0));
    return vec4<f32>(c.rgb * shading, c.a);
}
