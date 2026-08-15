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
    float sharpness;  // still used for edge enhancement, but NOT for DLS
} pc;

layout(binding = 0) uniform mediump sampler2D texSampler;
layout(location = 0) in  highp vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

float fastLanczos2(float x) {
    float wA = x - 4.0;
    float wB = x * wA - wA;
    wA *= wA;
    return wB * wA;
}

vec2 weightY(float dx, float dy, float c, float std, float spatialFactor) {
    float x = (dx * dx + dy * dy) * spatialFactor + clamp(abs(c) * std, 0.0, 1.0);
    float w = fastLanczos2(x);
    return vec2(w, w * c);
}

// --- NEW: Contrast Adaptive Sharpening (CAS) ---
vec3 applyCAS(vec3 center, vec2 uv, float sharp) {
    vec2 texel = vec2(pc.invSrcW, pc.invSrcH);
    
    // Sample 4 cardinal neighbors
    vec3 b = texture(texSampler, uv + vec2( 0.0,    -texel.y)).rgb;
    vec3 d = texture(texSampler, uv + vec2(-texel.x,  0.0   )).rgb;
    vec3 f = texture(texSampler, uv + vec2( texel.x,  0.0   )).rgb;
    vec3 h = texture(texSampler, uv + vec2( 0.0,     texel.y)).rgb;

    vec3 mn = min(min(b, d), min(f, h));
    vec3 mx = max(max(b, d), max(f, h));
    
    // Calculate local contrast
    vec3 range = mx - mn;
    float maxContrast = max(max(range.r, range.g), range.b);
    
    // Adaptive magic: If contrast is high (strong edge), weight drops to 0.
    // If contrast is low (flat), weight boosts to 1.0.
    float adaptiveWeight = clamp(1.0 - (maxContrast * 1.5), 0.0, 1.0);
    
    // Standard blur from 4 neighbors
    vec3 blur = (b + d + f + h) * 0.25;
    
    // Apply adaptive sharpening
    vec3 result = center + (center - blur) * (sharp * 0.2) * adaptiveWeight;
    return clamp(result, 0.0, 1.0);
}

// --- DLS handles ONLY color/contrast now -- CAS handles sharpening ---
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
    t = vec3(pow(t.r, 1.12), t.g * 1.2, t.b * 1.2);
    return clamp(t * toRGB, 0.0, 1.0);
}

// --- UPDATED POST-PROCESSING CHAIN ---
// edgeStrength in [0,1]: how much the directional resample in main() already
// corrected this pixel (0 = fast path / flat region, 1 = max correction).
// CAS's neighbor taps sample the raw source texture rather than this
// already-corrected center pixel, so running CAS at full strength on a pixel
// the NIS pass above already reconstructed risks stacking two sharpeners on
// the same edge (halo / over-sharpen). Back CAS off proportionally instead.
void applyPostFX(inout vec3 rgb, vec2 uv, float edgeStrength) {
    // 1. CAS handles adaptive sharpening without creating blocky halos.
    float casSharp = mix(1.0, 0.3, clamp(edgeStrength, 0.0, 1.0));
    rgb = applyCAS(rgb, uv, casSharp);     
    
    // 2. DLS handles color vibrancy and contrast (sharpening is zeroed inside).
    rgb = applyDLS(rgb);     
    
    // 3. HDR handles the glowing highlights.
    rgb = applyHDR(rgb, uv);          
    
    // 4. Natural is UNCOMMENTED by default, but optional. 
    // If you want the warm cinematic tone, remove the // below.
    rgb = applyNatural(rgb);          
    
    // 5. Optional CRT        
        // rgb = applyCRT(rgb, uv);
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
        applyPostFX(rgb, fragTexCoord, 0.0);
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

    // MICRO-OPT: clamp(pc.sharpness, 0, 1) was recomputed three times below
    // (here, edgeSharpness, maxDelta) for the same value each time. Hoisted.
    float sharpAmt = clamp(pc.sharpness, 0.0, 1.0);

    float spatialFactor = mix(0.40, 0.65, sharpAmt);

    vec2 aWY = weightY(pl.x,       pl.y + 1.0, upDown.x, std, spatialFactor);
    aWY += weightY(pl.x - 1.0, pl.y + 1.0, upDown.y, std, spatialFactor);
    aWY += weightY(pl.x - 1.0, pl.y - 2.0, upDown.z, std, spatialFactor);
    aWY += weightY(pl.x,       pl.y - 2.0, upDown.w, std, spatialFactor);
    aWY += weightY(pl.x + 1.0, pl.y - 1.0, left.x,   std, spatialFactor);
    aWY += weightY(pl.x,       pl.y - 1.0, left.y,   std, spatialFactor);
    aWY += weightY(pl.x,       pl.y,       left.z,   std, spatialFactor);
    aWY += weightY(pl.x + 1.0, pl.y,       left.w,   std, spatialFactor);
    aWY += weightY(pl.x - 1.0, pl.y - 1.0, right.x,  std, spatialFactor);
    aWY += weightY(pl.x - 2.0, pl.y - 1.0, right.y,  std, spatialFactor);
    aWY += weightY(pl.x - 2.0, pl.y,       right.z,  std, spatialFactor);
    aWY += weightY(pl.x - 1.0, pl.y,       right.w,  std, spatialFactor);

    float finalY = aWY.y / max(aWY.x, 1.0e-6);

    float maxY = max(max(left.y, left.z), max(right.x, right.w)) + mean;
    float minY = min(min(left.y, left.z), min(right.x, right.w)) + mean;

    float edgeSharpness = mix(1.0, 2.0, sharpAmt);
    finalY = clamp(edgeSharpness * finalY + mean, minY, maxY);

    float maxDelta = mix(16.0, 40.0, sharpAmt) / 255.0;
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
    // into applyPostFX to back CAS off on pixels the resample above already
    // sharpened (see comment on applyPostFX).
    float edgeStrength = abs(deltaY) / max(maxDelta, 1.0e-6);
    applyPostFX(result.rgb, fragTexCoord, edgeStrength);

    outColor = result;
}
