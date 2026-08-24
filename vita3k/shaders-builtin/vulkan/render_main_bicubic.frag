#version 460 core

precision mediump float;
precision highp int;

layout(push_constant) uniform PC {
    float ndcX0;
    float ndcY0;
    float ndcX1;
    float ndcY1;
    int   useTexAlpha;
    float invSrcW;
    float invSrcH;
    float srcW;
    float srcH;
    int   effectId;   // now only used for CRT (2 = on, else off)
    float resW;       // declared but never referenced anywhere below -- dead field, same pattern as in present_mm.frag/present_mmpx.frag
    float sharpness;  // NOT CURRENTLY READ -- see SHARP_DEFAULT in main(); nothing
                       // writes this via vkCmdPushConstants, so it was unsafe to read
} pc;

layout(binding = 0) uniform mediump sampler2D texSampler;
layout(location = 0) in  highp vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

// --- EASU direction/anisotropy, ported from AMD FidelityFX FSR 1.0's
// FsrEasuSetF (ffx_fsr1.h, master branch -- same file this project vendors
// at Edengit/externals/FidelityFX-FSR/ffx-fsr/ffx_fsr1.h). This is the
// single 5-point '+' sample (A=up,B=left,C=center,D=right,E=down) that
// fsr.txt's FsrMobile also uses -- confirmed identical to the master
// source, just formatted differently (+(-x) vs -x). Deliberately NOT
// desktop EASU's 4-corner bilinear-blended direction estimate, since
// fsr.txt is the Mobile variant and that's what it uses.
//
// Sampled fresh at (uv, texel) rather than reused from the gather grid in
// main() below, so this doesn't depend on that grid's tap-to-pixel
// ordering at all -- getting that ordering wrong once already caused the
// diagonal-staircasing bug the `right` gather's FIX comment describes.
//
// Uses the r*0.5+g luma proxy from the article fsr.txt is sourced from
// (atyuwen, "Optimizing AMD FSR for Mobiles", section 5) -- not full RGB
// luma, but not this file's G-only shortcut either. Costs nothing extra:
// texture() already returns the full RGB, this just stops discarding .r.
struct EasuDir { vec2 dir; vec2 len2; float lob; float clp; float aniso; };

// Estimate the source-texel footprint of one destination pixel.
// The screen-filter host draws the fullscreen quad over the destination
// viewport while fragTexCoord spans the source image, so derivatives give us
// the actual source-pixel footprint without adding a Vita3K-specific uniform.
//
// footprint < 1: upscaling (one output pixel sees less than one source texel)
// footprint ~= 1: native scale
// footprint > 1: downscaling
vec2 sourceFootprint() {
    vec2 dx = abs(dFdx(fragTexCoord)) * vec2(pc.srcW, pc.srcH);
    vec2 dy = abs(dFdy(fragTexCoord)) * vec2(pc.srcW, pc.srcH);

    // Use the larger derivative rather than dx+dy. For a 2x upscale this
    // correctly reports ~0.5 source texel per output pixel, instead of 1.0.
    return max(max(dx, dy), vec2(1.0e-6));
}

float sourceUpscaleFactor(vec2 footprint) {
    float sx = 1.0 / max(footprint.x, 1.0e-6);
    float sy = 1.0 / max(footprint.y, 1.0e-6);
    // Use the geometric mean so a non-uniform scale (e.g. Vita 960x544 to
    // 1920x1080) is represented by one stable quality factor.
    return sqrt(max(sx * sy, 1.0));
}

// CONTENT-AWARE POST FX:
// Color processing below reuses EASU's existing center/neighborhood samples.
// No additional texture taps are introduced for shadow/detail awareness.
// This is specifically intended to prevent dark textured areas from losing
// contrast/detail after DLS + HDR + Natural while preserving those effects in
// ordinary midtone/bright regions.
//
// // Scale-aware EASU kernel width. At large upscales the reconstruction is
// allowed to use a slightly broader footprint; at 1:1 it falls back to the
// existing sharpness-controlled kernel. This is intentionally bounded.
float scaleAwareSpatialFactor(float sharpAmt, vec2 footprint) {
    float upscale = sourceUpscaleFactor(footprint);
    float lowResBoost = clamp((upscale - 1.0) * 0.075, 0.0, 0.12);
    return mix(0.40, 0.65, sharpAmt) * (1.0 - lowResBoost);
}

// A very small directional pre-resolve for low-resolution upscaling.
// EASU reconstructs the edge, while this 3-sample tangent resolve reduces
// the "stair-step" phase changes that remain on diagonals when the source
// rasterization is substantially below the display resolution.
//
// We only return a luminance correction. This avoids washing chroma across
// colored edges and keeps the operation compatible with the existing
// luminance-based EASU result.
float subpixelEdgeAA(vec2 uv, vec2 texel, EasuDir ed, float edgeStrength, float upscale) {
    if (upscale <= 1.15 || edgeStrength <= 0.02)
        return 0.0;

    // ed.dir follows the local gradient (across the edge); its perpendicular
    // is therefore along the edge, where sampling is useful for reducing
    // staircase phase changes without crossing the edge as aggressively.
    vec2 tangent = vec2(-ed.dir.y, ed.dir.x);

    // Keep the offsets subpixel at the source resolution. Stronger scaling
    // gets a little more coverage, but never more than 0.5 source texel.
    float radius = mix(0.20, 0.50, clamp((upscale - 1.0) / 2.0, 0.0, 1.0));
    vec2 off = tangent * texel * radius;

    vec3 p0 = texture(texSampler, uv - off).rgb;
    vec3 p1 = texture(texSampler, uv + off).rgb;

    float y0 = p0.r * 0.5 + p0.g;
    float y1 = p1.r * 0.5 + p1.g;
    float avgY = 0.5 * (y0 + y1);

    // The correction is deliberately small. It is an anti-aliasing resolve,
    // not another sharpening pass.
    float strength = 0.12 * clamp((upscale - 1.0) / 1.0, 0.0, 1.0);
    strength *= smoothstep(0.02, 0.18, edgeStrength);
    return (avgY) * strength;
}


EasuDir easuDirection(vec2 uv, vec2 texel) {
    vec3 sA = texture(texSampler, uv + vec2(0.0, -texel.y)).rgb;
    vec3 sB = texture(texSampler, uv + vec2(-texel.x, 0.0)).rgb;
    vec3 sC = texture(texSampler, uv).rgb;
    vec3 sD = texture(texSampler, uv + vec2( texel.x, 0.0)).rgb;
    vec3 sE = texture(texSampler, uv + vec2(0.0,  texel.y)).rgb;

    float lA = sA.r * 0.5 + sA.g;
    float lB = sB.r * 0.5 + sB.g;
    float lC = sC.r * 0.5 + sC.g;
    float lD = sD.r * 0.5 + sD.g;
    float lE = sE.r * 0.5 + sE.g;

    float dc = lD - lC, cb = lC - lB;
    float lenX = max(abs(dc), abs(cb));
    lenX = 1.0 / max(lenX, 1.0e-6);
    float dirX = lD - lB;
    lenX = clamp(abs(dirX) * lenX, 0.0, 1.0);
    lenX *= lenX;

    float ec = lE - lC, ca = lC - lA;
    float lenY = max(abs(ec), abs(ca));
    lenY = 1.0 / max(lenY, 1.0e-6);
    float dirY = lE - lA;
    lenY = clamp(abs(dirY) * lenY, 0.0, 1.0);
    lenY *= lenY;

    vec2 dir = vec2(dirX, dirY);
    float len = lenX + lenY;

    // NEW: structure-tensor anisotropy, computed from the same lenX/lenY
    // this function already derives for the EASU resample direction --
    // no extra taps. lenX/lenY are each in [0,1] and measure gradient
    // strength along one axis independently, before AMD's stretch/len2
    // blend below mixes them together (which loses this distinction).
    //
    // A real edge (sign lettering stroke, hair line, brick joint) has one
    // axis's gradient clearly dominant -- anisotropic, aniso near 1.
    // Isotropic noise (JPEG block noise, texture dither) has lenX~=lenY --
    // aniso near 0. This is what applyPostFX's detailMask alone can't
    // tell apart, since it only looks at a single scalar local-variation
    // number with no directional information.
    float aniso = clamp(abs(lenX - lenY) / max(lenX + lenY, 1.0e-6), 0.0, 1.0);

    // Normalize direction, guarding the near-zero case exactly like AMD's
    // `zro` branch (master uses 1/32768 here -- a normalization guard, not
    // the coarser 1/64 early-out fsr.txt's fused mobile pass uses to skip
    // its whole tap loop; we're not replicating that shortcut, we always
    // run the 12-tap loop below, so the tighter guard is the correct one).
    vec2 dir2 = dir * dir;
    float dirR = dir2.x + dir2.y;
    if (dirR < 1.0 / 32768.0) {
        dir = vec2(1.0, 0.0);
    } else {
        dir *= inversesqrt(dirR);
    }

    len *= 0.5;
    len *= len;
    float stretch = (dir.x * dir.x + dir.y * dir.y) / max(max(abs(dir.x), abs(dir.y)), 1.0e-6);
    vec2 len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    float lob = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * len;
    float clp = 1.0 / max(lob, 1.0e-6);

    EasuDir ed;
    ed.dir = dir;
    ed.len2 = len2;
    ed.lob = lob;
    ed.clp = clp;
    ed.aniso = aniso;
    return ed;
}

// weightY's spatial term is now the real EASU tap weight, ported verbatim
// from FsrEasuTapF (ffx_fsr1.h): rotate the tap offset into the gradient
// frame, scale anisotropically by len2, then run the two-window Lanczos-2
// approximation (wA, wB) AMD ships. This replaces the old isotropic
// (dx*dx+dy*dy) distance term entirely -- fastLanczos2 is gone, since
// nothing else in this file called it.
//
// The existing value/range term (clamp(abs(c)*std,0,1)) isn't part of
// stock EASU -- it's this file's own bilateral-style "down-weight taps
// whose value differs a lot from center" mechanism. Kept as a multiplier
// on top of the real EASU weight rather than dropped, so both mechanisms
// combine instead of one replacing the other.
//
// DEVIATION FROM AMD'S REFERENCE: `spatialFactor` (from sharpAmt, see
// SHARP_DEFAULT in main() -- not currently host-controlled, see comment
// there) is
// folded in as an extra scale on d2 before the clp clip, preserving this
// file's existing user-facing sharpness knob (higher sharpAmt -> narrower
// kernel), matching its old role under the previous isotropic weightY.
// AMD's own lob/clp tuning doesn't include this scale; flag it if the
// sharpness slider's feel changes noticeably from before.
vec2 weightY(float dx, float dy, float c, float std, float spatialFactor, EasuDir ed) {
    float vx = dx * ed.dir.x + dy * ed.dir.y;
    float vy = dx * (-ed.dir.y) + dy * ed.dir.x;
    vx *= ed.len2.x;
    vy *= ed.len2.y;
    float d2 = (vx * vx + vy * vy) * spatialFactor;
    d2 = min(d2, ed.clp);

    float wB = (2.0 / 5.0) * d2 - 1.0;
    float wA = ed.lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = (25.0 / 16.0) * wB - (25.0 / 16.0 - 1.0);
    float wEasu = wB * wA;

    float rangeSimilarity = 1.0 - clamp(abs(c) * std, 0.0, 1.0);
    float w = wEasu * rangeSimilarity;
    return vec2(w, w * c);
}

// --- RCAS (Robust Contrast Adaptive Sharpening), ported from AMD FidelityFX
// FSR 1.0 (see fsr.txt / FsrMobile's combined RCAS block). Replaces the old
// heuristic applyCAS(): same job (4-tap cardinal adaptive sharpen, called
// the same way from applyPostFX below) but AMD's actual limiter math instead
// of a flat "contrast range -> blend weight" heuristic.
//
// Per channel, RCAS looks at the min/max of the 4-neighbor ring and computes
// how far the center pixel sits from that ring in each direction (hitMin /
// hitMax). The worst-case (most restrictive) channel sets a single "lobe"
// value that is then used to blend center against the 4 neighbors -- lobe
// near 0 near strong edges (don't sharpen, avoid ringing/halos), lobe more
// negative in flat regions (sharpen more). FSR_RCAS_LIMIT bounds how far
// that can go, same constant AMD ships (0.25 - 1/16).
//
// `sharp` keeps the same contract applyPostFX already passes in (the
// edgeStrength-backed-off casSharp, ~0.3..1.0) and scales the lobe directly,
// same role the old "* 0.2" constant played.
vec3 applyRCAS(vec3 e, vec2 uv, float sharp) {
    vec2 texel = vec2(pc.invSrcW, pc.invSrcH);

    // Sample 4 cardinal neighbors (same taps/positions as the old CAS).
    vec3 b = texture(texSampler, uv + vec2( 0.0,    -texel.y)).rgb;
    vec3 d = texture(texSampler, uv + vec2(-texel.x,  0.0   )).rgb;
    vec3 f = texture(texSampler, uv + vec2( texel.x,  0.0   )).rgb;
    vec3 h = texture(texSampler, uv + vec2( 0.0,     texel.y)).rgb;

    vec3 mn4 = min(min(b, d), min(f, h));
    vec3 mx4 = max(max(b, d), max(f, h));

    // Limiters -- distance from center's ring extremes, high-precision rcp
    // per AMD's note (tonality shifts visibly if this rcp is too coarse).
    // Denominators are guarded the same way the rest of this file guards
    // rcps (max(x, 1e-6) / min(x, -1e-6)) rather than AMD's raw rcp.
    vec3 hitMin = mn4 / max(4.0 * mx4, 1.0e-6);
    vec3 hitMax = (1.0 - mx4) / min(4.0 * mn4 - 4.0, -1.0e-6);
    vec3 lobeRGB = max(-hitMin, hitMax);

    const float FSR_RCAS_LIMIT = 0.25 - (1.0 / 16.0);
    float lobe = max(-FSR_RCAS_LIMIT,
                      min(max(max(lobeRGB.r, lobeRGB.g), lobeRGB.b), 0.0)) * sharp;

    // Resolve: blend center against the ring using the shared lobe weight.
    float rcpL = 1.0 / (4.0 * lobe + 1.0);
    vec3 result = (lobe * (b + d + f + h) + e) * rcpL;
    return clamp(result, 0.0, 1.0);
}

// --- DLS handles ONLY color/contrast now -- RCAS handles sharpening ---
// MICRO-OPT: this used to also fetch a 4-tap cardinal blur (uv, texel, the
// four texture() calls below) to feed a sharpening term that was already
// zeroed out (SHARP = 0.0) -- the blur was computed and then never read.
// Dead code, dead fetches: removed. Output is bit-for-bit unchanged; this
// just drops 4 texture instructions per pixel that were doing nothing.
vec3 applyDLS(vec3 center) {
    // MAX_SHARP now strictly controls the saturation and contrast boost
    const float MAX_SHARP = 0.4;   
    float SAT = 1.0 + MAX_SHARP * 0.20;   // 1.08
    float CON = 1.0 + MAX_SHARP * 0.12;   // 1.048

    vec3 c = clamp((center - 0.5) * CON + 0.5, 0.0, 1.0);
    float gray = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(gray), c, SAT);
    return c;
}

vec3 applyCRT(vec3 center, vec2 uv) {
    const float CA = 1.0025;
    float r = texture(texSampler, (uv - 0.5) * CA + 0.5).r;
    float b = texture(texSampler, (uv - 0.5) / CA + 0.5).b;
    vec3 fc  = vec3(r, center.g, b);
    float sx = abs(sin(uv.x * 1024.0) * 0.5 * 0.125);
    float sy = abs(sin(uv.y * 1024.0) * 0.5 * 0.375);
    return mix(fc, vec3(0.0), sx + sy);
}

vec3 applyHDR(vec3 center, vec2 uv) {
    vec2 texel = vec2(pc.invSrcW, pc.invSrcH);
    const float r1 = 0.793, r2 = 0.870;
    vec3 b1 = vec3(0.0), b2 = vec3(0.0);
    // PERF: cut from 8 directions to 4 (diagonals only). Confirmed no mip
    // chain is available for this source (mipLevels = 1 on the backing
    // image), so the cheap mip-blur route isn't on the table -- this is
    // the next-best tap reduction. Weights doubled (0.005->0.010,
    // 0.010->0.020) so average bloom brightness stays roughly where it was
    // with 8 taps instead of dimming purely from the smaller sample count.
    // Trade-off: the glow is now 4-point-sampled instead of 8, so it's
    // slightly less isotropic -- worth an eyeball on a bright point-light
    // source specifically, since that's where under-sampling shows first.
    vec2 offs[4] = vec2[](
        vec2( 1.6264,  1.6264), vec2(-1.6264,  1.6264),
        vec2(-1.6264, -1.6264), vec2( 1.6264, -1.6264)
    );
    for (int i = 0; i < 4; i++) {
        b1 += texture(texSampler, uv + offs[i] * r1 * texel).rgb;
        b2 += texture(texSampler, uv + offs[i] * r2 * texel).rgb;
    }
    b1 *= 0.010; b2 *= 0.020;
    vec3 hdr = (center + (b2 - b1)) * (r2 - r1);
    return clamp(pow(abs(hdr + center), vec3(1.15)) + hdr, 0.0, 1.0); // 1.15 instead of 1.30
}

vec3 applyNatural(vec3 c) {
    mat3 toYIQ = mat3( 0.299,  0.596,  0.212,
                       0.587, -0.275, -0.523,
                       0.114, -0.321,  0.311);
    mat3 toRGB = mat3( 1.0,         1.0,         1.0,
                       0.95568806, -0.27158179, -1.10817732,
                       0.61985809, -0.64687381,  1.70506455);
    vec3 t = c * toYIQ;
    // Chroma multiplier lowered from 1.2 -> 1.10. applyDLS already adds
    // +8% saturation before this runs; stacking a second +20% boost on top
    // was exaggerating the color fringing left by EASU/RCAS on saturated
    // edges (turns a subtle ring into a visible colored halo).
    t = vec3(pow(t.r, 1.12), t.g * 1.10, t.b * 1.10);
    return clamp(t * toRGB, 0.0, 1.0);
}

// --- UPDATED POST-PROCESSING CHAIN ---
// edgeStrength in [0,1]: how much the directional resample in main() already
// corrected this pixel (0 = fast path / flat region, 1 = max correction).
// RCAS's neighbor taps sample the raw source texture rather than this
// already-corrected center pixel, so running RCAS at full strength on a pixel
// the NIS pass above already reconstructed risks stacking two sharpeners on
// the same edge (halo / over-sharpen). Back RCAS off proportionally instead.
// Content-aware post processing.
//
// IMPORTANT: this version deliberately adds NO texture taps.
// It reuses information already calculated by the EASU pass:
//   - centerG: source center luminance proxy
//   - localMean: mean of the already-gathered neighborhood
//   - edgeStrength: amount of EASU luminance correction
//   - aniso: structure-tensor anisotropy from easuDirection (0 = isotropic
//     / noise-like, 1 = coherent directional edge). Also free -- computed
//     from EasuDir's existing lenX/lenY, no new taps.
//
// The goal is to stop DLS/HDR/Natural from crushing fine information in
// dark regions while retaining the original look elsewhere.
void applyPostFX(inout vec3 rgb,
                 vec2 uv,
                 float edgeStrength,
                 float centerLuma,
                 float localMean,
                 float aniso) {

    // Dark-region detector. Keep the transition soft so there is no visible
    // brightness boundary around a shadow threshold.
    float shadowMask = 1.0 - smoothstep(0.055, 0.30, centerLuma);

    // Reuse the already-sampled neighborhood to estimate local variation.
    // This is intentionally cheap: no new texture() calls.
    float localVariation = abs(centerLuma - localMean);
    float detailMask = smoothstep(0.012, 0.075, localVariation);

    // If a region is both dark and locally detailed, preserve it strongly.
    // Flat dark regions receive much less protection.
    float darkDetailProtect = shadowMask * detailMask;

    // EASU edge correction is another useful signal. A strong reconstructed
    // edge in a dark region is exactly where we do not want post-processing
    // to erase the recovered texture.
    darkDetailProtect *= mix(0.65, 1.0, clamp(edgeStrength, 0.0, 1.0));

    // Protect the shadow/detail region, but don't completely disable the
    // user's post-processing. 0.72 means at maximum protection 72% of the
    // enhancement is backed off.
    float fxStrength = 1.0 - 0.72 * clamp(darkDetailProtect, 0.0, 1.0);

    // 1. RCAS: keep it active, but reduce it slightly in already-reconstructed
    // edges. This is the same zero-extra-tap RCAS path as before.
    // Ceiling lowered from 1.0 -> 0.75: RCAS sharpens flat regions harder
    // than edges by design, which is correct when the flatness hides real
    // detail but amplifies texture/JPEG micro-noise when it doesn't (e.g.
    // a flat, mid-bright sign board). 0.75 still sharpens genuine flat
    // detail, just not at full strength.
    float casSharp = mix(0.75, 0.3, clamp(edgeStrength, 0.0, 1.0));
    casSharp = mix(0.55, casSharp, fxStrength);

    // Additional guard: darkDetailProtect above only covers dark regions
    // (shadowMask gates on centerLuma < ~0.30), so a flat, NON-dark, noisy
    // surface gets none of that protection.
    //
    // flatMask alone (localVariation near 0) misses surfaces like a sign
    // board that have enough micro-variance to NOT read as "flat" --
    // detailMask picks them up as "detail" even though that variance has
    // no coherent direction (it's dither/JPEG noise, not a stroke or
    // edge). isotropicNoise catches exactly that case: high when there IS
    // local variation (detailMask) but it's NOT directional (low aniso).
    // Using max() means either signal alone is enough to trigger the
    // guard, so genuinely flat surfaces and noisy-but-textured surfaces
    // are both covered without double-counting where they overlap.
    float flatMask = 1.0 - detailMask;
    float isotropicNoise = detailMask * (1.0 - clamp(aniso, 0.0, 1.0));
    float noiseSuppress = max(flatMask, isotropicNoise);
    casSharp = mix(casSharp, casSharp * 0.6, noiseSuppress * (1.0 - shadowMask));

    rgb = applyRCAS(rgb, uv, casSharp);

    // Preserve the pre-color-processing result for a final shadow/detail
    // guard. This costs only registers/ALU, not texture bandwidth.
    vec3 beforeColorFX = rgb;

    // 2. DLS: contrast/saturation enhancement.
    vec3 dls = applyDLS(rgb);
    rgb = mix(rgb, dls, fxStrength);

    // 3. HDR: existing four-diagonal-tap implementation. We do not add taps.
    // In dark detailed areas its nonlinear response is backed off.
    vec3 hdr = applyHDR(rgb, uv);
    rgb = mix(rgb, hdr, fxStrength);

    // 4. Natural: existing YIQ/gamma-style processing, again attenuated in
    // dark detailed regions instead of globally disabling it.
    vec3 natural = applyNatural(rgb);
    rgb = mix(rgb, natural, fxStrength);

    // Final conservative luminance/detail guard.
    //
    // If the color pipeline has moved a dark detailed pixel substantially,
    // pull it partway back toward the pre-color-FX value. This is intentionally
    // a blend rather than an RGB rescale, so hue/saturation are not destabilized.
    float guard = 0.35 * clamp(darkDetailProtect, 0.0, 1.0);
    rgb = mix(rgb, beforeColorFX, guard);
}

void main() {
    highp vec2 step = vec2(pc.invSrcW, pc.invSrcH);

    vec4 center = textureLod(texSampler, fragTexCoord, 0.0);

    // FIXED: was (-0.5, +0.5) -- an asymmetric pixel-center offset that
    // shifted every vertical fractional phase (pl.y) by up to a full texel
    // relative to x, which reads exactly as diagonal staircasing. Standard
    // pixel-center alignment for a UV*srcRes coordinate is -0.5 on both axes.
    // CAVEAT: if this project deliberately compensates for a Y-flip upstream
    // (Vulkan's Y-down NDC vs. the ndcY0/ndcY1 handling in the vertex stage),
    // that compensation lived in this one line -- revert it and re-check the
    // vertex stage instead. I can't render this shader, so verify visually.
    highp vec2 imgCoord      = fragTexCoord * vec2(pc.srcW, pc.srcH) + vec2(-0.5, -0.5);
    highp vec2 imgCoordFloor = floor(imgCoord);
    highp vec2 baseUV        = imgCoordFloor * step;
    vec2  pl                 = imgCoord - imgCoordFloor;

    vec4 left = textureGather(texSampler, baseUV, 1);

    float centerG  = center.g;
    // NOTE: vote is G-channel-only (luma proxy). Isoluminant color edges --
    // similar brightness, different hue -- won't register here and will pass
    // through unresampled. Fixing that needs gathering R/B too and extending
    // the weighted resample below to all three channels, which roughly
    // triples the tap count; left out of this pass as a cost/benefit call --
    // worth doing separately if that's specifically what you're seeing.
    float edgeVote = abs(left.z - left.y) + abs(centerG - left.y) + abs(centerG - left.z);

    const float EDGE_THRESHOLD  = 12.0 / 255.0;
    const float EDGE_TAPER_BAND = 4.0 / 255.0; // soft transition width around
                                                // EDGE_THRESHOLD

    if (edgeVote <= EDGE_THRESHOLD - EDGE_TAPER_BAND) {
        // Well inside a flat region -- skip the directional resample entirely.
        vec3 rgb = center.rgb;
        // No neighborhood EASU data exists on this fast path, so use the
        // center itself as the local mean. This adds no taps.
        applyPostFX(rgb, fragTexCoord, 0.0, centerG, centerG, 0.0);
        outColor = vec4(rgb, (pc.useTexAlpha != 0) ? center.a : 1.0);
        return;
    }

    highp vec2 baseUV2 = baseUV + vec2(step.x, 0.0);

    // FIX: this previously added step.x a second time here, on top of the
    // shift already baked into baseUV2 -- so `right` sampled 2 texels over
    // from `left` instead of 1, leaving a whole column of source pixels never
    // sampled at all. The directional weight math below assumes a physically
    // contiguous grid at fixed integer offsets, so that gap corrupted the
    // horizontal/diagonal component of every resample -- the likely cause of
    // staircasing on diagonal and horizontal edges (most letterforms).
    vec4 right;
    right = textureGather(texSampler, baseUV2, 1);

    vec4 upDown;
    upDown.xy = textureGather(texSampler, baseUV + vec2(0.0, -step.y), 1).wz;
    upDown.zw = textureGather(texSampler, baseUV + vec2(0.0,  step.y), 1).yx;

    float mean = (left.y + left.z + right.x + right.w) * 0.25;
    left   -= vec4(mean);
    right  -= vec4(mean);
    upDown -= vec4(mean);

    float sum =
        abs(left.x)   + abs(left.y)   + abs(left.z)   + abs(left.w)   +
        abs(right.x)  + abs(right.y)  + abs(right.z)  + abs(right.w)  +
        abs(upDown.x) + abs(upDown.y) + abs(upDown.z) + abs(upDown.w);
    float std = 2.181818 / max(sum, 1.0e-6);

    // pc.sharpness is declared in the push constant block but nothing on
    // the host side currently calls vkCmdPushConstants (or equivalent) to
    // write it -- per the Vulkan spec, push constant memory has NO
    // guaranteed initial value until something writes to it, so reading it
    // unwritten means whatever was left over from a prior draw, or
    // uninitialized driver memory. clamp() on a resulting NaN is undefined
    // behavior in GLSL too, so this isn't a "gets a bland default" risk --
    // it can silently corrupt the sharpening chain (feeds spatialFactor,
    // edgeSharpness, maxDelta below).
    //
    // Using a fixed local default instead until that's wired up. To make
    // it host-controlled later: replace SHARP_DEFAULT below with
    // clamp(pc.sharpness, 0.0, 1.0) once vkCmdPushConstants actually writes
    // this field -- nothing else here needs to change.
    // Lowered from 0.5 -> 0.32. pc.sharpness is still unsafe to read here --
    // host-side code never calls vkCmdPushConstants for this field, so
    // reading it would pull uninitialized data. Actually wiring a
    // user-facing slider requires a host-side (C++) change outside this
    // file; until that lands, 0.32 trims EASU's edge-delta overshoot
    // (see edgeSharpness/maxDelta below) that was producing halo/ringing
    // on hard, saturated edges, while keeping most of the reconstruction
    // sharpness EASU is there for.
    const float SHARP_DEFAULT = 0.32;
    float sharpAmt = SHARP_DEFAULT;

    vec2 footprint = sourceFootprint();
    float spatialFactor = scaleAwareSpatialFactor(sharpAmt, footprint);

    // EASU direction/anisotropy, computed once per pixel and shared by all
    // 12 taps below (see easuDirection's doc comment above).
    EasuDir ed = easuDirection(fragTexCoord, step);

    vec2 aWY = weightY(pl.x,       pl.y + 1.0, upDown.x, std, spatialFactor, ed);
    aWY += weightY(pl.x - 1.0, pl.y + 1.0, upDown.y, std, spatialFactor, ed);
    aWY += weightY(pl.x - 1.0, pl.y - 2.0, upDown.z, std, spatialFactor, ed);
    aWY += weightY(pl.x,       pl.y - 2.0, upDown.w, std, spatialFactor, ed);
    aWY += weightY(pl.x + 1.0, pl.y - 1.0, left.x,   std, spatialFactor, ed);
    aWY += weightY(pl.x,       pl.y - 1.0, left.y,   std, spatialFactor, ed);
    aWY += weightY(pl.x,       pl.y,       left.z,   std, spatialFactor, ed);
    aWY += weightY(pl.x + 1.0, pl.y,       left.w,   std, spatialFactor, ed);
    aWY += weightY(pl.x - 1.0, pl.y - 1.0, right.x,  std, spatialFactor, ed);
    aWY += weightY(pl.x - 2.0, pl.y - 1.0, right.y,  std, spatialFactor, ed);
    aWY += weightY(pl.x - 2.0, pl.y,       right.z,  std, spatialFactor, ed);
    aWY += weightY(pl.x - 1.0, pl.y,       right.w,  std, spatialFactor, ed);

    float finalY = aWY.y / max(aWY.x, 1.0e-6);

    // Low-resolution edge resolve: use the reconstructed EASU edge strength
    // as the gate so flat areas remain untouched. The center value is added
    // outside this helper because the helper returns only the correction.
    float preAAEdge = abs(finalY - centerG);
    float upscale = sourceUpscaleFactor(footprint);
    finalY += subpixelEdgeAA(fragTexCoord, step, ed, preAAEdge, upscale);

    float maxY = max(max(left.y, left.z), max(right.x, right.w)) + mean;
    float minY = min(min(left.y, left.z), min(right.x, right.w)) + mean;

    // Capped from 2.0 -> 1.6. At 2.0x this term was overshooting hard,
    // saturated color edges (graffiti-on-concrete style content) and
    // producing a visible halo before RCAS even runs. 1.6 keeps most of
    // the perceived crispness on linework/text with less ringing headroom.
    float edgeSharpness = mix(1.0, 1.6, sharpAmt);
    finalY = clamp(edgeSharpness * finalY + mean, minY, maxY);

    // Capped from 40 -> 28. Limits the max luminance correction per pixel,
    // directly trimming halo intensity on the same hard-edge case above.
    float maxDelta = mix(16.0, 28.0, sharpAmt) / 255.0;
    float deltaY   = clamp(finalY - centerG, -maxDelta, maxDelta);

    // Taper the correction out near EDGE_THRESHOLD instead of the old hard
    // branch. Previously, a pixel just above the cutoff jumped straight to
    // full-strength correction while a pixel just below got none at all --
    // that step was visible as a faint seam tracing the threshold boundary.
    // Cost: pixels inside the taper band that used to take the fast path now
    // run the full resample above (yielding a small correction), in exchange
    // for removing that seam.
    float pathBlend = smoothstep(EDGE_THRESHOLD - EDGE_TAPER_BAND,
                                  EDGE_THRESHOLD + EDGE_TAPER_BAND, edgeVote);
    deltaY *= pathBlend;

    vec4 result;
    result.rgb = clamp(center.rgb + vec3(deltaY), 0.0, 1.0);
    result.a   = (pc.useTexAlpha != 0) ? center.a : 1.0;

    // How much correction this pixel already got, normalized to [0,1] -- fed
    // into applyPostFX to back RCAS off on pixels the resample above already
    // sharpened (see comment on applyPostFX).
    float edgeStrength = abs(deltaY) / max(maxDelta, 1.0e-6);

    // Keep RCAS useful after reconstruction, but avoid excessive sharpening
    // when the source is already near/native resolution. The helper still
    // caps the actual RCAS limiter, so this is a small quality bias only.
    float scaleSharp = 1.0 + clamp((upscale - 1.0) * 0.06, 0.0, 0.10);
    edgeStrength = clamp(edgeStrength / scaleSharp, 0.0, 1.0);

    // `mean` is already computed from the EASU gather neighborhood, so the
    // content-aware color guard costs no additional texture reads.
    applyPostFX(result.rgb, fragTexCoord, edgeStrength, centerG, mean, ed.aniso);

    outColor = result;
}
