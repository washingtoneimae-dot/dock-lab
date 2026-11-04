/**
 * engine.js — WebGL2 rigid-pose docking engine.
 *
 * Pipeline per substep:
 *   1. force pass  (one fragment per ligand pose) samples the receptor voxel hash
 *   2. integrate   (one fragment per pose) updates position/quaternion + sleep
 *
 * The receptor is static. Each ligand pose is a rigid body; its atoms are the
 * particles shown in the scene. That keeps 60k atom evaluations per frame while
 * only integrating ~2k rigid poses.
 */

import {
  MAX_POSES, MAX_TORSIONS, makeInitialPoses, crystalPose, rmsdToCrystal
} from './target.js';

const TEX_W = 4096;
const LIG_TEX_W = 64;

// Keep the CPU-side "moving" count in sync with the shader's sleep test.
const SLEEP_VEL2 = 0.00025;

// ------------------------------------------------------------- math -------

function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0
  ]);
}

function lookAt(eye, center, up) {
  let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
  let zl = Math.hypot(zx, zy, zz) || 1;
  zx /= zl; zy /= zl; zz /= zl;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  let xl = Math.hypot(xx, xy, xz) || 1;
  xx /= xl; xy /= xl; xz /= xl;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1
  ]);
}

function multiply(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

// ---------------------------------------------------------- shader bits ---

const GLSL_ELEMENT = `
vec3 elementColor(float t) {
  int i = int(t + 0.5);
  if (i == 0) return vec3(0.88, 0.90, 0.94);
  if (i == 1) return vec3(0.74, 0.76, 0.80);
  if (i == 2) return vec3(0.36, 0.66, 1.00);
  if (i == 3) return vec3(1.00, 0.36, 0.42);
  if (i == 4) return vec3(1.00, 0.78, 0.30);
  if (i == 5) return vec3(1.00, 0.55, 0.20);
  if (i == 6) return vec3(0.55, 1.00, 0.78);
  if (i == 7) return vec3(0.50, 1.00, 0.55);
  if (i == 8) return vec3(0.85, 0.45, 0.35);
  if (i == 9) return vec3(0.70, 0.45, 0.85);
  if (i == 10) return vec3(0.50, 0.65, 0.85); // ZN
  if (i == 11) return vec3(0.20, 0.85, 0.35); // MG
  if (i == 12) return vec3(0.50, 0.55, 0.60); // CA
  if (i == 13) return vec3(0.65, 0.50, 0.75); // MN
  if (i == 14) return vec3(0.85, 0.50, 0.20); // FE
  if (i == 15) return vec3(0.78, 0.55, 0.30); // CU
  if (i == 16) return vec3(0.65, 0.40, 0.85); // NA
  if (i == 17) return vec3(0.55, 0.35, 0.75); // K
  return vec3(0.62, 0.68, 0.78);
}
`;

const GLSL_QUAT = `
mat3 quatToMat(vec4 q) {
  float x = q.x, y = q.y, z = q.z, w = q.w;
  return mat3(
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w),
    2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w),
    2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)
  );
}
vec4 quatMul(vec4 a, vec4 b) {
  return vec4(
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z
  );
}
float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}
`;

const GLSL_TORSION = `
uniform int uTorsionCount;
uniform vec4 uTorsionAxis[${MAX_TORSIONS}];
uniform vec4 uTorsionPivot[${MAX_TORSIONS}];

vec3 rotAbout(vec3 v, vec3 axis, float a) {
  float c = cos(a), s = sin(a);
  return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

mat3 rotAboutMat(vec3 axis, float a) {
  float c = cos(a), s = sin(a), ic = 1.0 - c;
  float x = axis.x, y = axis.y, z = axis.z;
  float xx = x * x * ic + c,     xy = x * y * ic - z * s, xz = x * z * ic + y * s;
  float yx = y * x * ic + z * s, yy = y * y * ic + c,     yz = y * z * ic - x * s;
  float zx = z * x * ic - y * s, zy = z * y * ic + x * s, zz = z * z * ic + c;
  return mat3(xx, yx, zx, xy, yy, zy, xz, yz, zz);
}

vec3 applyTorsions(vec3 rel0, int mask, float ang[${MAX_TORSIONS}]) {
  vec3 rel = rel0;
  mat3 Racc = mat3(1.0);
  vec3 Tacc = vec3(0.0);
  for (int k = 0; k < ${MAX_TORSIONS}; ++k) {
    if (k >= uTorsionCount) break;
    if ((mask & (1 << k)) == 0) continue;
    vec3 ax = Racc * uTorsionAxis[k].xyz;
    vec3 pv = Racc * uTorsionPivot[k].xyz + Tacc;
    float a = ang[k];
    rel = rotAbout(rel - pv, ax, a) + pv;
    mat3 Rk = rotAboutMat(ax, a);
    Tacc = Rk * (Tacc - pv) + pv;
    Racc = Rk * Racc;
  }
  return rel;
}
`;

const VS_QUAD = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
`;

const FS_FIELD = `#version 300 es
precision highp float;
precision highp int;

uniform sampler2D uRecPos;
uniform sampler2D uRecParam;
uniform sampler2D uCellMeta;
uniform sampler2D uCellAtoms;

uniform vec3 uGridMin;
uniform ivec3 uGridDims;
uniform float uCellSize;
uniform float uCutoff2;

uniform vec3 uFieldMin;
uniform ivec3 uFieldDims;
uniform float uFieldCell;

uniform float uProbeSigma;
uniform float uProbeEps;
uniform float uProbeQ;
uniform float uProbePolar;

uniform float uDielectric;
uniform float uLambda;
uniform float uKappa;
uniform float uMaxForce;

layout(location = 0) out vec4 outLJ;
layout(location = 1) out vec4 outES;
layout(location = 2) out vec4 outApolar;
layout(location = 3) out vec4 outPolar;

const float COULOMB = 332.0637;
const int TEX_W = 4096;

vec4 fetchTex(sampler2D t, int idx) {
  return texelFetch(t, ivec2(idx % TEX_W, idx / TEX_W), 0);
}

void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy);
  int ix = fc.x;
  int iy = fc.y % uFieldDims.y;
  int iz = fc.y / uFieldDims.y;
  vec3 x = uFieldMin + (vec3(float(ix), float(iy), float(iz)) + 0.5) * uFieldCell;

  vec3 ljF = vec3(0.0);
  float ljE = 0.0;
  vec3 esF = vec3(0.0);
  float esE = 0.0;
  vec3 apolF = vec3(0.0);
  float apolE = 0.0;
  vec3 polF = vec3(0.0);
  float polE = 0.0;

  ivec3 cell = ivec3(floor((x - uGridMin) / uCellSize));
  for (int dz = -1; dz <= 1; ++dz) {
    int cz = cell.z + dz;
    if (cz < 0 || cz >= uGridDims.z) continue;
    for (int dy = -1; dy <= 1; ++dy) {
      int cy = cell.y + dy;
      if (cy < 0 || cy >= uGridDims.y) continue;
      for (int dx = -1; dx <= 1; ++dx) {
        int cx = cell.x + dx;
        if (cx < 0 || cx >= uGridDims.x) continue;
        int ci = cx + cy * uGridDims.x + cz * uGridDims.x * uGridDims.y;
        vec4 meta = fetchTex(uCellMeta, ci);
        int start = int(meta.x + 0.5);
        int count = int(meta.y + 0.5);
        int kmax = min(count, 48);
        for (int k = 0; k < kmax; ++k) {
          int idx = int(fetchTex(uCellAtoms, start + k).r + 0.5);
          vec4 rp = fetchTex(uRecPos, idx);
          vec4 rpar = fetchTex(uRecParam, idx);

          vec3 d = x - rp.xyz;
          float r2 = dot(d, d);
          if (r2 > uCutoff2) continue;
          r2 = max(r2, 0.81);
          float r = sqrt(r2);
          float invR2 = 1.0 / r2;
          float invR = 1.0 / r;

          float sig = 0.5 * (uProbeSigma + rpar.x);
          float eps = sqrt(max(uProbeEps * rpar.y, 0.0));
          
          // Softened Lennard-Jones: avoids catastrophic r^-12 ejection of crystal contacts
          float rClamped = max(r, sig * 0.72);
          float sr2 = (sig * sig) / (rClamped * rClamped);
          float sr6 = sr2 * sr2 * sr2;
          float sr12 = sr6 * sr6;
          float lj = 4.0 * eps * (sr12 - sr6);
          float fLJ = 24.0 * eps * (2.0 * sr12 - sr6) / (rClamped * r);

          // Distance-dependent dielectric: eps -> 1 at contact, eps_s in bulk.
          float epsR = 1.0 + (uDielectric - 1.0) * (1.0 - exp(-r / uLambda));
          float deps = (uDielectric - 1.0) / uLambda * exp(-r / uLambda);
          // Debye-Hückel screening for a ~0.15 M ionic strength.
          float screen = exp(-uKappa * r);
          float qq = uProbeQ * rpar.z;
          float coul = COULOMB * qq * screen / (epsR * r);
          float fES = COULOMB * qq * screen / (epsR * r2 * r) *
                      (1.0 + r * deps / epsR + uKappa * r);

          float sw = 1.0 - smoothstep(uCutoff2 * 0.64, uCutoff2, r2);
          ljE += lj * sw;
          ljF += fLJ * sw * d;
          esE += coul * sw;
          esF += fES * sw * d;

          // AutoDock-style pairwise desolvation with explicit H-bonding & metal coordination
          float contact = exp(-r2 / 12.5);
          float polarPenalty = uProbePolar * min(rpar.w, 1.0);
          apolE -= contact;
          apolF -= (1.0 / 6.25) * contact * d;
          polE += 0.85 * contact * polarPenalty;
          polF += (0.85 / 6.25) * contact * polarPenalty * d;

          // Explicit Hydrogen Bonding (r ~ 2.85 Å) and Metal Coordination (r ~ 2.08 Å)
          if (uProbePolar > 0.5 && rpar.w > 0.5) {
            float rHB = 2.85;
            float wHB = 0.45;
            float depthHB = -3.8;
            if (rpar.w > 1.5) { // Metal ion coordination (e.g. catalytic Zinc in 1S63)
              rHB = 2.08;
              wHB = 0.35;
              depthHB = -7.5;
            }
            float dHB = (r - rHB) / wHB;
            float hbond = depthHB * exp(-dHB * dHB);
            float fHB = -(2.0 * dHB / (wHB * r)) * hbond;
            polE += hbond * sw;
            polF += fHB * sw * d;
          }
        }
      }
    }
  }

  float ljm = length(ljF);
  if (ljm > uMaxForce) ljF *= uMaxForce / ljm;
  float esm = length(esF);
  if (esm > uMaxForce) esF *= uMaxForce / esm;
  float am = length(apolF);
  if (am > uMaxForce) apolF *= uMaxForce / am;
  float pm = length(polF);
  if (pm > uMaxForce) polF *= uMaxForce / pm;

  outLJ = vec4(ljF, clamp(ljE, -1.0e5, 1.0e5));
  outES = vec4(esF, clamp(esE, -1.0e5, 1.0e5));
  outApolar = vec4(apolF, clamp(apolE, -1.0e5, 0.0));
  outPolar = vec4(polF, clamp(polE, -1.0e5, 1.0e5));
}
`;

const FS_COMBINE = `#version 300 es
precision highp float;
precision highp sampler2DArray;

uniform sampler2DArray uField;
uniform float uLJScale;
uniform float uEScale;
uniform float uBurial;
uniform float uSolvent;
uniform int uLayerBase;

layout(location = 0) out vec4 outComposite;

void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy);
  vec4 lj = texelFetch(uField, ivec3(fc, uLayerBase), 0);
  vec4 es = texelFetch(uField, ivec3(fc, uLayerBase + 1), 0);
  vec4 ap = texelFetch(uField, ivec3(fc, uLayerBase + 2), 0);
  vec4 po = texelFetch(uField, ivec3(fc, uLayerBase + 3), 0);
  outComposite = uLJScale * lj + uEScale * es + uBurial * ap + uSolvent * po;
}
`;

const FS_FORCE = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2DArray;

uniform sampler2D uPosePos;
uniform sampler2D uPoseQuat;
uniform sampler2D uPoseVel;
uniform sampler2D uPoseAng;
uniform sampler2D uPoseTorsA;
uniform sampler2D uPoseTorsB;
uniform sampler2D uLigOffset;
uniform sampler2D uLigParam;
uniform sampler2D uAtomMask;
uniform sampler2DArray uCompositeField;

uniform vec3 uFieldMin;
uniform ivec3 uFieldDims;
uniform float uFieldCell;
uniform vec3 uSearchCenter;
uniform float uSearchRadius;
uniform float uRestraintK;
uniform float uTemp;
uniform float uNoise;
uniform float uTorqueNoise;
uniform float uMaxForce;
uniform float uFrame;
uniform float uSeed;
uniform float uTorsionBarrier;
uniform float uTorsionPrior;
uniform float uTorsionDeadZone;
uniform int uLigCount;

layout(location = 0) out vec4 outForce;
layout(location = 1) out vec4 outTorque;
layout(location = 2) out vec4 outForceTorsA;
layout(location = 3) out vec4 outForceTorsB;

${GLSL_QUAT}
${GLSL_TORSION}

vec4 sampleFieldLayer(vec3 x, int layer) {
  vec3 g = clamp((x - uFieldMin) / uFieldCell - 0.5, vec3(0.0), vec3(uFieldDims - 2));
  ivec3 i0 = ivec3(floor(g));
  vec3 t = clamp(g - vec3(i0), 0.0, 1.0);
  int stride = uFieldDims.y;
  ivec3 p000 = ivec3(i0.x, i0.y + i0.z * stride, layer);
  ivec3 p100 = ivec3(i0.x + 1, i0.y + i0.z * stride, layer);
  ivec3 p010 = ivec3(i0.x, i0.y + 1 + i0.z * stride, layer);
  ivec3 p110 = ivec3(i0.x + 1, i0.y + 1 + i0.z * stride, layer);
  ivec3 p001 = ivec3(i0.x, i0.y + (i0.z + 1) * stride, layer);
  ivec3 p101 = ivec3(i0.x + 1, i0.y + (i0.z + 1) * stride, layer);
  ivec3 p011 = ivec3(i0.x, i0.y + 1 + (i0.z + 1) * stride, layer);
  ivec3 p111 = ivec3(i0.x + 1, i0.y + 1 + (i0.z + 1) * stride, layer);
  vec4 c000 = texelFetch(uCompositeField, p000, 0);
  vec4 c100 = texelFetch(uCompositeField, p100, 0);
  vec4 c010 = texelFetch(uCompositeField, p010, 0);
  vec4 c110 = texelFetch(uCompositeField, p110, 0);
  vec4 c001 = texelFetch(uCompositeField, p001, 0);
  vec4 c101 = texelFetch(uCompositeField, p101, 0);
  vec4 c011 = texelFetch(uCompositeField, p011, 0);
  vec4 c111 = texelFetch(uCompositeField, p111, 0);
  vec4 c00 = mix(c000, c100, t.x);
  vec4 c10 = mix(c010, c110, t.x);
  vec4 c01 = mix(c001, c101, t.x);
  vec4 c11 = mix(c011, c111, t.x);
  return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec4 p = texelFetch(uPosePos, coord, 0);
  vec4 q = texelFetch(uPoseQuat, coord, 0);
  vec4 v = texelFetch(uPoseVel, coord, 0);
  vec4 a = texelFetch(uPoseAng, coord, 0);
  vec4 tA = texelFetch(uPoseTorsA, coord, 0);
  vec4 tB = texelFetch(uPoseTorsB, coord, 0);

  if (a.w < 0.5) {
    outForce = vec4(0.0, 0.0, 0.0, p.w);
    outTorque = vec4(0.0);
    outForceTorsA = vec4(0.0);
    outForceTorsB = vec4(0.0);
    return;
  }

  float ang[${MAX_TORSIONS}];
  ang[0] = tA.x; ang[1] = tA.y; ang[2] = tA.z; ang[3] = tA.w;
  ang[4] = tB.x; ang[5] = tB.y; ang[6] = tB.z; ang[7] = tB.w;
  float tq[${MAX_TORSIONS}];
  for (int k = 0; k < ${MAX_TORSIONS}; ++k) tq[k] = 0.0;

  vec3 pos = p.xyz;
  mat3 R = quatToMat(q);
  mat3 Rt = transpose(R);
  vec3 F = vec3(0.0);
  vec3 T = vec3(0.0);
  float energy = 0.0;

  for (int i = 0; i < uLigCount; ++i) {
    vec4 lo = texelFetch(uLigOffset, ivec2(i, 0), 0);
    int mask = int(texelFetch(uAtomMask, ivec2(i, 0), 0).r + 0.5);

    vec3 rel = lo.xyz;
    mat3 Racc = mat3(1.0);
    vec3 Tacc = vec3(0.0);
    vec3 pv[${MAX_TORSIONS}];
    vec3 ax[${MAX_TORSIONS}];
    int kidx[${MAX_TORSIONS}];
    int nSt = 0;
    for (int k = 0; k < ${MAX_TORSIONS}; ++k) {
      if (k >= uTorsionCount) break;
      if ((mask & (1 << k)) == 0) continue;
      vec3 axk = Racc * uTorsionAxis[k].xyz;
      vec3 pvk = Racc * uTorsionPivot[k].xyz + Tacc;
      float ak = ang[k];
      rel = rotAbout(rel - pvk, axk, ak) + pvk;
      mat3 Rk = rotAboutMat(axk, ak);
      Tacc = Rk * (Tacc - pvk) + pvk;
      Racc = Rk * Racc;
      pv[nSt] = pvk; ax[nSt] = axk; kidx[nSt] = k; nSt++;
    }

    vec3 relWorld = R * rel;
    vec3 x = pos + relWorld;
    int probe = int(lo.w + 0.5);
    vec4 fE = sampleFieldLayer(x, probe);
    vec3 f = fE.xyz;
    energy += fE.w;
    F += f;
    T += cross(relWorld, f);

    // Generalized torsion force: project the atom force onto each axis.
    vec3 fLocal = Rt * f;
    for (int s = 0; s < nSt; ++s) {
      tq[kidx[s]] += dot(cross(rel - pv[s], fLocal), ax[s]);
    }
  }

  // A weak rotamer prior towards the input conformation: small librations are
  // free, but flipping to a different staggered rotamer must be paid for.
  for (int k = 0; k < ${MAX_TORSIONS}; ++k) {
    if (k >= uTorsionCount) break;
    float dev = max(abs(ang[k]) - uTorsionDeadZone, 0.0);
    float sgn = ang[k] < 0.0 ? -1.0 : 1.0;
    tq[k] -= 3.0 * uTorsionBarrier * sin(3.0 * ang[k]) + uTorsionPrior * dev * sgn;
    energy += uTorsionBarrier * (1.0 - cos(3.0 * ang[k])) + 0.5 * uTorsionPrior * dev * dev;
  }

  vec3 toC = pos - uSearchCenter;
  float dist = length(toC);
  if (dist > uSearchRadius) {
    float over = dist - uSearchRadius;
    F -= (uRestraintK * over / max(dist, 0.0001)) * toC;
    energy += 0.5 * uRestraintK * over * over;
  }

  float amp = sqrt(max(uTemp, 0.0) / 300.0);
  vec3 rnd = vec3(
    hash11(float(coord.x) * 1.31 + uFrame * 0.017 + uSeed),
    hash11(float(coord.x) * 2.17 + uFrame * 0.023 + uSeed + 7.1),
    hash11(float(coord.x) * 3.71 + uFrame * 0.031 + uSeed + 13.7)
  ) - 0.5;
  F += rnd * uNoise * amp;
  T += (vec3(
    hash11(float(coord.x) * 4.11 + uFrame * 0.019 + uSeed + 23.1),
    hash11(float(coord.x) * 5.37 + uFrame * 0.029 + uSeed + 31.7),
    hash11(float(coord.x) * 6.73 + uFrame * 0.041 + uSeed + 41.3)
  ) - 0.5) * uTorqueNoise * amp;

  float fm = length(F);
  if (fm > uMaxForce) F *= uMaxForce / fm;
  energy = clamp(energy, -1.0e6, 1.0e6);

  outForce = vec4(F, energy);
  outTorque = vec4(T, 0.0);
  outForceTorsA = vec4(tq[0], tq[1], tq[2], tq[3]);
  outForceTorsB = vec4(tq[4], tq[5], tq[6], tq[7]);
}
`;

const FS_INTEGRATE = `#version 300 es
precision highp float;
precision highp int;

uniform sampler2D uPosePos;
uniform sampler2D uPoseQuat;
uniform sampler2D uPoseVel;
uniform sampler2D uPoseAng;
uniform sampler2D uPoseTorsA;
uniform sampler2D uPoseTorsB;
uniform sampler2D uPoseTorsVelA;
uniform sampler2D uPoseTorsVelB;
uniform sampler2D uForce;
uniform sampler2D uTorque;
uniform sampler2D uForceTorsA;
uniform sampler2D uForceTorsB;

uniform float uDt;
uniform float uInvMass;
uniform float uInvInertia;
uniform float uAccelScale;
uniform float uAngAccelScale;
uniform float uDamping;
uniform float uMaxSpeed;
uniform float uMaxSpin;
uniform float uMaxForce;
uniform float uMaxTorque;
uniform float uTemp;
uniform float uNoiseVel;
uniform float uNoiseAng;
uniform float uSleepFrames;
uniform float uSleepVel2;
uniform float uSleepSpin2;
uniform float uSleepEps;
uniform float uWakeProb;
uniform float uUseSleep;
uniform float uReheat;
uniform float uFrame;
uniform float uSeed;
uniform int uTorsionCount;
uniform float uTorsionInvInertia[${MAX_TORSIONS}];
uniform float uTorsionAccelScale;
uniform float uTorsionDamping;
uniform float uMaxTorsionVel;
uniform vec3 uBestPos;
uniform vec4 uBestQuat;
uniform float uBestEnergy;
uniform float uHasBest;

layout(location = 0) out vec4 outPos;
layout(location = 1) out vec4 outQuat;
layout(location = 2) out vec4 outVel;
layout(location = 3) out vec4 outAng;
layout(location = 4) out vec4 outTorsA;
layout(location = 5) out vec4 outTorsB;
layout(location = 6) out vec4 outTorsVelA;
layout(location = 7) out vec4 outTorsVelB;

${GLSL_QUAT}

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec4 p = texelFetch(uPosePos, coord, 0);
  vec4 q = texelFetch(uPoseQuat, coord, 0);
  vec4 v = texelFetch(uPoseVel, coord, 0);
  vec4 a = texelFetch(uPoseAng, coord, 0);
  vec4 f = texelFetch(uForce, coord, 0);
  vec4 t = texelFetch(uTorque, coord, 0);
  vec4 tA = texelFetch(uPoseTorsA, coord, 0);
  vec4 tB = texelFetch(uPoseTorsB, coord, 0);
  vec4 tvA = texelFetch(uPoseTorsVelA, coord, 0);
  vec4 tvB = texelFetch(uPoseTorsVelB, coord, 0);
  vec4 tqA = texelFetch(uForceTorsA, coord, 0);
  vec4 tqB = texelFetch(uForceTorsB, coord, 0);

  float ang[${MAX_TORSIONS}];
  ang[0] = tA.x; ang[1] = tA.y; ang[2] = tA.z; ang[3] = tA.w;
  ang[4] = tB.x; ang[5] = tB.y; ang[6] = tB.z; ang[7] = tB.w;
  float tvel[${MAX_TORSIONS}];
  tvel[0] = tvA.x; tvel[1] = tvA.y; tvel[2] = tvA.z; tvel[3] = tvA.w;
  tvel[4] = tvB.x; tvel[5] = tvB.y; tvel[6] = tvB.z; tvel[7] = tvB.w;
  float tq[${MAX_TORSIONS}];
  tq[0] = tqA.x; tq[1] = tqA.y; tq[2] = tqA.z; tq[3] = tqA.w;
  tq[4] = tqB.x; tq[5] = tqB.y; tq[6] = tqB.z; tq[7] = tqB.w;

  float amp = sqrt(max(uTemp, 0.0) / 300.0);

  if (a.w < 0.5 && uReheat < 0.5) {
    if (uUseSleep < 0.5) {
      // Sleep was switched off while this pose was asleep: wake it and fall
      // through to the integration step so forces can move it again.
      a.w = 1.0;
      v.w = 0.0;
    } else if (uTemp > 1.0 && uHasBest > 0.5 && v.w > 40.0 && f.w > uBestEnergy + 15.0) {
      // Swarm Genetic Migration: if a pose is stuck in a high-energy trap (> +15 kcal/mol above best)
      // for more than 40 frames, migrate it to explore the winning pocket minimum!
      // Gated above 1 K so a quench can freeze high-energy local minima in place.
      float r1 = hash11(float(coord.x) * 2.31 + uFrame * 0.017 + uSeed + 13.0);
      float r2 = hash11(float(coord.x) * 3.73 + uFrame * 0.029 + uSeed + 29.0);
      float r3 = hash11(float(coord.x) * 4.97 + uFrame * 0.043 + uSeed + 41.0);
      p.xyz = uBestPos + (vec3(r1, r2, r3) - 0.5) * 1.5;
      q = normalize(uBestQuat + (vec4(
        hash11(float(coord.x) * 5.11 + uSeed + 53.0),
        hash11(float(coord.x) * 6.27 + uSeed + 67.0),
        hash11(float(coord.x) * 7.43 + uSeed + 79.0),
        hash11(float(coord.x) * 8.89 + uSeed + 97.0)
      ) - 0.5) * 0.25);
      v.xyz = vec3(0.0);
      a.xyz = vec3(0.0);
      v.w = 0.0;
      a.w = 1.0;
    } else {
      float wake = hash11(float(coord.x) * 1.73 + uFrame * 0.071 + uSeed);
      if (wake < uWakeProb) {
        a.w = 1.0;
        v.w = 0.0;
        v.xyz = (vec3(
          hash11(float(coord.x) * 2.03 + uSeed + 3.0),
          hash11(float(coord.x) * 3.17 + uSeed + 11.0),
          hash11(float(coord.x) * 4.31 + uSeed + 19.0)
        ) - 0.5) * 0.006 * amp;
        a.xyz = (vec3(
          hash11(float(coord.x) * 5.41 + uSeed + 29.0),
          hash11(float(coord.x) * 6.53 + uSeed + 37.0),
          hash11(float(coord.x) * 7.67 + uSeed + 43.0)
        ) - 0.5) * 0.003 * amp;
        for (int k = 0; k < ${MAX_TORSIONS}; ++k) {
          if (k >= uTorsionCount) break;
          tvel[k] = (hash11(float(coord.x) * (8.1 + float(k)) + uSeed + 47.0) - 0.5) * 0.004 * amp;
        }
      } else {
        outPos = p; outQuat = q; outVel = v; outAng = a;
        outTorsA = tA; outTorsB = tB; outTorsVelA = tvA; outTorsVelB = tvB;
        return;
      }
    }
  }

  if (uReheat > 0.5) {
    a.w = 1.0;
    v.w = 0.0;
    v.xyz += (vec3(
      hash11(float(coord.x) * 8.13 + uSeed + 53.0),
      hash11(float(coord.x) * 9.27 + uSeed + 61.0),
      hash11(float(coord.x) * 10.39 + uSeed + 71.0)
    ) - 0.5) * 0.02;
    a.xyz += (vec3(
      hash11(float(coord.x) * 11.43 + uSeed + 83.0),
      hash11(float(coord.x) * 12.57 + uSeed + 97.0),
      hash11(float(coord.x) * 13.69 + uSeed + 101.0)
    ) - 0.5) * 0.01;
    for (int k = 0; k < ${MAX_TORSIONS}; ++k) {
      if (k >= uTorsionCount) break;
      tvel[k] += (hash11(float(coord.x) * (14.1 + float(k)) + uSeed + 103.0) - 0.5) * 0.012;
    }
  }

  vec3 F = f.xyz;
  float fm = length(F);
  if (fm > uMaxForce) F *= uMaxForce / fm;
  vec3 T = t.xyz;
  float tm = length(T);
  if (tm > uMaxTorque) T *= uMaxTorque / tm;

  v.xyz += F * uInvMass * uAccelScale * uDt;
  v.xyz *= uDamping;
  float sp = length(v.xyz);
  if (sp > uMaxSpeed) v.xyz *= uMaxSpeed / sp;

  p.xyz += v.xyz * uDt;

  a.xyz += T * uInvInertia * uAngAccelScale * uDt;
  a.xyz *= uDamping;
  float spin = length(a.xyz);
  if (spin > uMaxSpin) a.xyz *= uMaxSpin / spin;

  vec4 wq = vec4(a.xyz * uDt * 0.5, 0.0);
  q = normalize(q + quatMul(wq, q));

  v.xyz += (vec3(
    hash11(float(coord.x) * 14.71 + uFrame * 0.013 + uSeed + 107.0),
    hash11(float(coord.x) * 15.83 + uFrame * 0.017 + uSeed + 109.0),
    hash11(float(coord.x) * 16.97 + uFrame * 0.019 + uSeed + 113.0)
  ) - 0.5) * uNoiseVel * amp;
  a.xyz += (vec3(
    hash11(float(coord.x) * 17.11 + uFrame * 0.023 + uSeed + 127.0),
    hash11(float(coord.x) * 18.23 + uFrame * 0.029 + uSeed + 131.0),
    hash11(float(coord.x) * 19.37 + uFrame * 0.031 + uSeed + 137.0)
  ) - 0.5) * uNoiseAng * amp;

  float torsSpin2 = 0.0;
  for (int k = 0; k < ${MAX_TORSIONS}; ++k) {
    if (k >= uTorsionCount) continue;
    tvel[k] += tq[k] * uTorsionInvInertia[k] * uTorsionAccelScale * uDt;
    tvel[k] *= uTorsionDamping;
    tvel[k] = clamp(tvel[k], -uMaxTorsionVel, uMaxTorsionVel);
    ang[k] += tvel[k] * uDt;
    ang[k] = mod(ang[k] + 3.14159265, 6.2831853) - 3.14159265;
    torsSpin2 += tvel[k] * tvel[k];
  }

  float speed2 = dot(v.xyz, v.xyz);
  float spin2 = dot(a.xyz, a.xyz) + torsSpin2;
  float dE = abs(f.w - p.w);
  float timer = v.w;
  if (uUseSleep > 0.5 && speed2 < uSleepVel2 && spin2 < uSleepSpin2 && dE < uSleepEps) {
    timer += 1.0;
  } else {
    timer = 0.0;
  }
  if (timer > uSleepFrames) {
    a.w = 0.0;
    v.xyz = vec3(0.0);
    a.xyz = vec3(0.0);
    for (int k = 0; k < ${MAX_TORSIONS}; ++k) tvel[k] = 0.0;
  }

  outPos = vec4(p.xyz, f.w);
  outQuat = q;
  outVel = vec4(v.xyz, timer);
  outAng = vec4(a.xyz, a.w);
  outTorsA = vec4(ang[0], ang[1], ang[2], ang[3]);
  outTorsB = vec4(ang[4], ang[5], ang[6], ang[7]);
  outTorsVelA = vec4(tvel[0], tvel[1], tvel[2], tvel[3]);
  outTorsVelB = vec4(tvel[4], tvel[5], tvel[6], tvel[7]);
}
`;

const VS_RECEPTOR = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aType;
uniform mat4 uViewProj;
uniform mat4 uView;
uniform float uPixelScale;
uniform float uSizeScale;
flat out float vType;
out float vDepth;
void main() {
  vec4 mv = uView * vec4(aPos, 1.0);
  gl_Position = uViewProj * vec4(aPos, 1.0);
  gl_PointSize = clamp(uSizeScale * 0.9 * uPixelScale / max(-mv.z, 1.0), 1.0, 42.0);
  vType = aType;
  vDepth = -mv.z;
}
`;

const FS_RECEPTOR = `#version 300 es
precision highp float;
uniform float uAlpha;
uniform float uSurface;
in float vDepth;
flat in float vType;
out vec4 outColor;
${GLSL_ELEMENT}
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float z = sqrt(max(0.0, 1.0 - r2));
  vec3 n = vec3(p.x, -p.y, z);
  float diff = 0.32 + 0.68 * max(dot(n, normalize(vec3(0.4, 0.7, 0.6))), 0.0);
  vec3 col = elementColor(vType) * diff;
  float alpha = uAlpha;
  if (uSurface > 0.5) alpha *= (1.0 - r2) * 0.9;
  outColor = vec4(col, alpha);
}
`;

const VS_SWARM = `#version 300 es
precision highp float;
precision highp int;
layout(location = 0) in vec3 aOffset;
layout(location = 1) in float aRadius;
layout(location = 2) in float aType;
uniform sampler2D uPosePos;
uniform sampler2D uPoseQuat;
uniform sampler2D uPoseVel;
uniform sampler2D uPoseAng;
uniform sampler2D uPoseTorsA;
uniform sampler2D uPoseTorsB;
uniform sampler2D uAtomMask;
uniform mat4 uViewProj;
uniform mat4 uView;
uniform float uPixelScale;
uniform float uSizeScale;
flat out float vType;
out float vEnergy;
out float vActive;
out float vDepth;
${GLSL_QUAT}
${GLSL_TORSION}
void main() {
  int id = gl_InstanceID;
  vec4 p = texelFetch(uPosePos, ivec2(id, 0), 0);
  vec4 q = texelFetch(uPoseQuat, ivec2(id, 0), 0);
  vec4 a = texelFetch(uPoseAng, ivec2(id, 0), 0);
  vec4 tA = texelFetch(uPoseTorsA, ivec2(id, 0), 0);
  vec4 tB = texelFetch(uPoseTorsB, ivec2(id, 0), 0);
  float ang[${MAX_TORSIONS}];
  ang[0] = tA.x; ang[1] = tA.y; ang[2] = tA.z; ang[3] = tA.w;
  ang[4] = tB.x; ang[5] = tB.y; ang[6] = tB.z; ang[7] = tB.w;
  int mask = int(texelFetch(uAtomMask, ivec2(gl_VertexID, 0), 0).r + 0.5);
  mat3 R = quatToMat(q);
  vec3 local = applyTorsions(aOffset, mask, ang);
  vec3 world = p.xyz + R * local;
  vec4 mv = uView * vec4(world, 1.0);
  gl_Position = uViewProj * vec4(world, 1.0);
  gl_PointSize = clamp(uSizeScale * aRadius * uPixelScale / max(-mv.z, 1.0), 1.0, 24.0);
  vType = aType;
  vEnergy = p.w;
  vActive = a.w;
  vDepth = -mv.z;
}
`;

const FS_SWARM = `#version 300 es
precision highp float;
uniform vec3 uLowColor;
uniform vec3 uHighColor;
uniform float uAlpha;
uniform float uEMin;
uniform float uEMax;
in float vEnergy;
in float vActive;
in float vDepth;
flat in float vType;
out vec4 outColor;
${GLSL_ELEMENT}
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float t = clamp((vEnergy - uEMin) / max(uEMax - uEMin, 1.0), 0.0, 1.0);
  vec3 col = mix(elementColor(vType), mix(uLowColor, uHighColor, t), 0.72);
  float alpha = uAlpha * (vActive > 0.5 ? 1.0 : 0.16) * (1.0 - r2);
  outColor = vec4(col, alpha);
}
`;

const VS_BEST = `#version 300 es
precision highp float;
precision highp int;
layout(location = 0) in vec3 aOffset;
layout(location = 1) in float aRadius;
layout(location = 2) in float aType;
uniform sampler2D uAtomMask;
uniform mat4 uViewProj;
uniform mat4 uView;
uniform vec3 uPos;
uniform vec4 uQuat;
uniform float uTorsionAngles[${MAX_TORSIONS}];
uniform float uPixelScale;
uniform float uSizeScale;
flat out float vType;
out float vDepth;
${GLSL_QUAT}
${GLSL_TORSION}
void main() {
  int mask = int(texelFetch(uAtomMask, ivec2(gl_VertexID, 0), 0).r + 0.5);
  mat3 R = quatToMat(uQuat);
  vec3 local = applyTorsions(aOffset, mask, uTorsionAngles);
  vec3 world = uPos + R * local;
  vec4 mv = uView * vec4(world, 1.0);
  gl_Position = uViewProj * vec4(world, 1.0);
  gl_PointSize = clamp(uSizeScale * aRadius * uPixelScale / max(-mv.z, 1.0), 2.0, 52.0);
  vType = aType;
  vDepth = -mv.z;
}
`;

const FS_BEST = `#version 300 es
precision highp float;
uniform float uAlpha;
in float vDepth;
flat in float vType;
out vec4 outColor;
${GLSL_ELEMENT}
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  float z = sqrt(max(0.0, 1.0 - r2));
  vec3 n = vec3(p.x, -p.y, z);
  float diff = 0.45 + 0.75 * max(dot(n, normalize(vec3(0.35, 0.75, 0.65))), 0.0);
  float rim = pow(1.0 - z, 2.0) * 0.55;
  vec3 col = elementColor(vType) * diff + vec3(0.35, 0.95, 0.55) * rim;
  outColor = vec4(col, uAlpha);
}
`;

const VS_LINE = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
void main() { gl_Position = uViewProj * vec4(aPos, 1.0); }
`;

const FS_LINE = `#version 300 es
precision highp float;
uniform vec3 uColor;
uniform float uAlpha;
out vec4 outColor;
void main() { outColor = vec4(uColor, uAlpha); }
`;

const VS_BOND = `#version 300 es
precision highp float;
precision highp int;
layout(location = 0) in float aIndexA;
layout(location = 1) in float aIndexB;
layout(location = 2) in float aEnd;
layout(location = 3) in float aSide;
uniform sampler2D uLigOffset;
uniform sampler2D uAtomMask;
uniform mat4 uViewProj;
uniform vec3 uPos;
uniform vec4 uQuat;
uniform vec2 uViewport;
uniform float uBondWidth;
uniform float uTorsionAngles[${MAX_TORSIONS}];
${GLSL_QUAT}
${GLSL_TORSION}
vec3 atomLocal(int idx) {
  vec4 lo = texelFetch(uLigOffset, ivec2(idx, 0), 0);
  int mask = int(texelFetch(uAtomMask, ivec2(idx, 0), 0).r + 0.5);
  return applyTorsions(lo.xyz, mask, uTorsionAngles);
}
void main() {
  mat3 R = quatToMat(uQuat);
  int ia = int(aIndexA + 0.5);
  int ib = int(aIndexB + 0.5);
  vec4 ca = uViewProj * vec4(uPos + R * atomLocal(ia), 1.0);
  vec4 cb = uViewProj * vec4(uPos + R * atomLocal(ib), 1.0);
  vec2 na = ca.xy / max(ca.w, 0.0001);
  vec2 nb = cb.xy / max(cb.w, 0.0001);
  vec2 dir = (nb - na) * uViewport;
  float dl = length(dir);
  if (dl < 0.0001) dir = vec2(1.0, 0.0); else dir /= dl;
  vec2 perp = vec2(-dir.y, dir.x) / uViewport;
  vec4 c = mix(ca, cb, aEnd);
  c.xy += perp * aSide * uBondWidth * c.w;
  gl_Position = c;
}
`;

const FS_BOND = `#version 300 es
precision highp float;
uniform vec3 uColor;
uniform float uAlpha;
out vec4 outColor;
void main() { outColor = vec4(uColor, uAlpha); }
`;

const VS_DOT = `#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
uniform mat4 uView;
uniform float uPixelScale;
uniform float uSizeScale;
uniform vec3 uColor;
out vec3 vColor;
void main() {
  vec4 mv = uView * vec4(aPos, 1.0);
  gl_Position = uViewProj * vec4(aPos, 1.0);
  gl_PointSize = clamp(uSizeScale * uPixelScale / max(-mv.z, 1.0), 1.0, 22.0);
  vColor = uColor;
}
`;

const FS_DOT = `#version 300 es
precision highp float;
uniform float uAlpha;
in vec3 vColor;
out vec4 outColor;
void main() {
  vec2 p = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(p, p);
  if (r2 > 1.0) discard;
  outColor = vec4(vColor, uAlpha * (1.0 - r2));
}
`;

// ------------------------------------------------------------- engine ----

export class DockEngine {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: true,
      depth: true,
      stencil: false,
      desynchronized: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false
    });
    if (!this.gl) throw new Error('WebGL2 is not available in this browser.');
    const gl = this.gl;

    this.floatColor = gl.getExtension('EXT_color_buffer_float');
    if (!this.floatColor) throw new Error('EXT_color_buffer_float is required for GPU physics.');

    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    this.renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    this.vendor = dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);

    this.opts = Object.assign({
      poseCount: 2000,
      substeps: 2,
      temperature: 330,
      ljScale: 1.0,
      esScale: 1.0,
      burial: 1.0,
      solvent: 1.0,
      torsions: true,
      torsionBarrier: 1.5,
      torsionPrior: 8.0,
      torsionDeadZone: 0.30,
      torsionAccelScale: 0.028,
      torsionDamping: 0.90,
      maxTorsionVel: 0.09,
      dielectric: 4.0,
      dielectricLambda: 2.0,
      screeningKappa: 0.12,
      damping: 0.925,
      accelScale: 0.032,
      angAccelScale: 0.055,
      searchScale: 0.85,
      fieldCell: 0.9,
      restraintK: 2.2,
      cutoff: 8.0,
      sleep: true,
      showReceptor: true,
      showSwarm: true,
      showBest: true,
      showCrystal: false,
      showLines: true,
      showSurface: true,
      autoRotate: true,
      renderScale: 1.0,
      seedFraction: 0.08
    }, opts);

    this.poseCount = clamp(Math.round(this.opts.poseCount), 64, MAX_POSES);
    this.seed = 0x1a2b3c4d;
    this.frame = 0;
    this.readEvery = 10;
    this.reheatFrames = 0;
    this.best = null;
    this.history = [];
    this.eMin = -20;
    this.eMax = 20;
    this.activeCount = 0;
    this.movingCount = 0;
    this.meanEnergy = 0;
    this.energyDelta = null;
    this.converged = false;
    this.stableSince = 0;
    this.fps = 0;
    this.renderScale = this.opts.renderScale;
    this.camera = {
      yaw: 0.72,
      pitch: -0.22,
      distance: 105,
      target: [0, 0, 0]
    };
    this.width = 1;
    this.height = 1;
    this.aspect = 1;
    this.pixelScale = 500;

    this.energyRead = new Float32Array(MAX_POSES * 4);
    this.activeEnergies = new Float32Array(MAX_POSES);
    this.prevEnergies = new Float32Array(MAX_POSES).fill(NaN);
    this.velRead = new Float32Array(MAX_POSES * 4);
    this.posRead = new Float32Array(MAX_POSES * 4);
    this.quatRead = new Float32Array(MAX_POSES * 4);
    this.angRead = new Float32Array(MAX_POSES * 4);
    this.torsARead = new Float32Array(MAX_POSES * 4);
    this.torsBRead = new Float32Array(MAX_POSES * 4);
    this.zeroTors = new Float32Array(MAX_TORSIONS);

    this._initGL();
    this._initPrograms();
    this._initQuad();
    this._initPoseTargets();
  }

  // ------------------------------------------------------------ setup ----

  _initGL() {
    const gl = this.gl;
    gl.disable(gl.CULL_FACE);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.clearColor(0.012, 0.017, 0.030, 1.0);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  }

  _compile(type, source, name) {
    const gl = this.gl;
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error(`${name} shader failed:\n${log}`);
    }
    return sh;
  }

  _program(vs, fs, name) {
    const gl = this.gl;
    const p = gl.createProgram();
    const v = this._compile(gl.VERTEX_SHADER, vs, name + ' vertex');
    const f = this._compile(gl.FRAGMENT_SHADER, fs, name + ' fragment');
    gl.attachShader(p, v);
    gl.attachShader(p, f);
    gl.linkProgram(p);
    gl.deleteShader(v);
    gl.deleteShader(f);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(p);
      gl.deleteProgram(p);
      throw new Error(`${name} link failed:\n${log}`);
    }
    const u = new Proxy({}, {
      get(target, key) { return key in target ? target[key] : null; }
    });
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(p, i);
      const base = info.name.replace(/\[0\]$/, '');
      u[base] = gl.getUniformLocation(p, info.name);
    }
    return { program: p, u };
  }

  _initPrograms() {
    this.progField = this._program(VS_QUAD, FS_FIELD, 'field');
    this.progCombine = this._program(VS_QUAD, FS_COMBINE, 'combine');
    this.progForce = this._program(VS_QUAD, FS_FORCE, 'force');
    this.progIntegrate = this._program(VS_QUAD, FS_INTEGRATE, 'integrate');
    this.progReceptor = this._program(VS_RECEPTOR, FS_RECEPTOR, 'receptor');
    this.progSwarm = this._program(VS_SWARM, FS_SWARM, 'swarm');
    this.progBest = this._program(VS_BEST, FS_BEST, 'best');
    this.progLine = this._program(VS_LINE, FS_LINE, 'line');
    this.progBond = this._program(VS_BOND, FS_BOND, 'bond');
    this.progDot = this._program(VS_DOT, FS_DOT, 'dot');
  }

  _initQuad() {
    const gl = this.gl;
    this.quadVBO = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 3, -1, -1, 3
    ]), gl.STATIC_DRAW);
    this.quadVAO = gl.createVertexArray();
    gl.bindVertexArray(this.quadVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  _initPoseTargets() {
    const gl = this.gl;
    this.poseTex = [];
    this.poseFBO = [];
    for (let i = 0; i < 2; i++) {
      const tex = [];
      for (let j = 0; j < 8; j++) tex.push(this._tex2D(MAX_POSES, 1, null));
      const fbo = this._fbo(tex);
      this.poseTex.push(tex);
      this.poseFBO.push(fbo);
    }
    this.forceTex = [];
    for (let j = 0; j < 4; j++) this.forceTex.push(this._tex2D(MAX_POSES, 1, null));
    this.forceFBO = this._fbo(this.forceTex);
    this.poseRead = 0;

    // Double-buffered PBOs for non-blocking asynchronous readback
    this.pbo = [gl.createBuffer(), gl.createBuffer()];
    const pboBytes = MAX_POSES * 4 * 4 * 7; // 7 float attachments
    for (let b = 0; b < 2; b++) {
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo[b]);
      gl.bufferData(gl.PIXEL_PACK_BUFFER, pboBytes, gl.STREAM_READ);
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.pboWrite = 0;
    this.pboReady = false;
  }

  _tex2D(w, h, data) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data || null);
    return t;
  }

  _fbo(texs) {
    const gl = this.gl;
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const buffers = [];
    for (let i = 0; i < texs.length; i++) {
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, texs[i], 0);
      buffers.push(gl.COLOR_ATTACHMENT0 + i);
    }
    gl.drawBuffers(buffers);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error('Incomplete float framebuffer: 0x' + status.toString(16));
    return fbo;
  }

  _buffer(data, target = this.gl.ARRAY_BUFFER) {
    const gl = this.gl;
    const b = gl.createBuffer();
    gl.bindBuffer(target, b);
    gl.bufferData(target, data, gl.STATIC_DRAW);
    return b;
  }

  // ----------------------------------------------------------- target ----

  setTarget(target) {
    this.target = target;
    const init = makeInitialPoses(target, this.poseCount, this.seed, this.opts.seedFraction);
    this.grid = init.grid;
    this._uploadTarget(target);
    this._buildField();
    this._uploadPoses(init);
    this._resetStats();
    this.camera.target = target.ligand.siteCenter.slice();
    this.camera.distance = Math.max(95, target.ligand.searchRadius * 7.0);
    this._packed = true;
  }

  _uploadTarget(target) {
    const gl = this.gl;
    if (this.recTex) this._disposeTargetGL();

    const recN = target.receptor.count;
    const recRows = Math.max(1, Math.ceil(recN / TEX_W));
    const recPosData = new Float32Array(TEX_W * recRows * 4);
    const recParData = new Float32Array(TEX_W * recRows * 4);
    for (let i = 0; i < recN; i++) {
      recPosData[i * 4] = target.receptor.pos[i * 3];
      recPosData[i * 4 + 1] = target.receptor.pos[i * 3 + 1];
      recPosData[i * 4 + 2] = target.receptor.pos[i * 3 + 2];
      recPosData[i * 4 + 3] = target.receptor.element[i];
      recParData[i * 4] = target.receptor.param[i * 4];
      recParData[i * 4 + 1] = target.receptor.param[i * 4 + 1];
      recParData[i * 4 + 2] = target.receptor.param[i * 4 + 2];
      // w was the unused vdW radius; reuse it as the polar/desolvation flag.
      recParData[i * 4 + 3] = (target.receptor.element[i] === 2 || target.receptor.element[i] === 3) ? 1 : 0;
    }
    this.recTex = this._tex2D(TEX_W, recRows, recPosData);
    this.recParTex = this._tex2D(TEX_W, recRows, recParData);

    const cellRows = Math.max(1, Math.ceil(this.grid.cellCount / TEX_W));
    const metaData = new Float32Array(TEX_W * cellRows * 4);
    metaData.set(this.grid.meta);
    this.cellMetaTex = this._tex2D(TEX_W, cellRows, metaData);

    const atomRows = Math.max(1, Math.ceil(recN / TEX_W));
    const atomData = new Float32Array(TEX_W * atomRows * 4);
    for (let i = 0; i < recN; i++) atomData[i * 4] = this.grid.indices[i];
    this.cellAtomTex = this._tex2D(TEX_W, atomRows, atomData);

    const ligN = target.ligand.count;
    const ligOffData = new Float32Array(LIG_TEX_W * 4);
    const ligParData = new Float32Array(LIG_TEX_W * 4);

    // One precomputed field layer triple per distinct ligand element type.
    const layerOf = new Int32Array(11).fill(-1);
    const probes = [];
    for (let i = 0; i < ligN; i++) {
      const el = target.ligand.element[i];
      if (layerOf[el] < 0) {
        layerOf[el] = probes.length;
        probes.push({
          sigma: target.ligand.param[i * 4],
          eps: target.ligand.param[i * 4 + 1],
          q: target.ligand.param[i * 4 + 2],
          polar: (el === 2 || el === 3) ? 1 : 0
        });
      }
    }
    this.fieldLayerOf = layerOf;
    this.fieldProbes = probes;
    this.fieldLayerCount = probes.length;

    for (let i = 0; i < ligN; i++) {
      ligOffData[i * 4] = target.ligand.offset[i * 3];
      ligOffData[i * 4 + 1] = target.ligand.offset[i * 3 + 1];
      ligOffData[i * 4 + 2] = target.ligand.offset[i * 3 + 2];
      ligOffData[i * 4 + 3] = layerOf[target.ligand.element[i]];
      ligParData[i * 4] = target.ligand.param[i * 4 + 1];
      ligParData[i * 4 + 1] = target.ligand.param[i * 4 + 2];
      ligParData[i * 4 + 2] = target.ligand.radius[i];
      ligParData[i * 4 + 3] = target.ligand.param[i * 4 + 3];
    }
    this.ligOffTex = this._tex2D(LIG_TEX_W, 1, ligOffData);
    this.ligParTex = this._tex2D(LIG_TEX_W, 1, ligParData);

    const maskData = new Float32Array(LIG_TEX_W * 4);
    const atomMask = target.ligand.torsions ? target.ligand.torsions.atomMask : null;
    if (atomMask) for (let i = 0; i < ligN; i++) maskData[i * 4] = atomMask[i];
    this.atomMaskTex = this._tex2D(LIG_TEX_W, 1, maskData);

    const tor = target.ligand.torsions;
    this.torsionAxisData = new Float32Array(MAX_TORSIONS * 4);
    this.torsionPivotData = new Float32Array(MAX_TORSIONS * 4);
    this.torsionInvInertiaData = new Float32Array(MAX_TORSIONS);
    if (tor) {
      for (let k = 0; k < tor.count; k++) {
        this.torsionAxisData[k * 4] = tor.axis[k * 3];
        this.torsionAxisData[k * 4 + 1] = tor.axis[k * 3 + 1];
        this.torsionAxisData[k * 4 + 2] = tor.axis[k * 3 + 2];
        this.torsionPivotData[k * 4] = tor.pivot[k * 3];
        this.torsionPivotData[k * 4 + 1] = tor.pivot[k * 3 + 1];
        this.torsionPivotData[k * 4 + 2] = tor.pivot[k * 3 + 2];
        this.torsionInvInertiaData[k] = tor.invInertia[k];
      }
    }
    this.torsionCount = tor ? tor.count : 0;

    // 3D interaction field atlas: one 2D layer per (type, component).
    // Height = ny * nz, so the whole volume is one tall 2D texture.
    const fieldCenter = target.ligand.siteCenter;
    const half = target.ligand.searchRadius + target.ligand.siteRadius + 2.5;
    let cell = clamp(this.opts.fieldCell != null ? this.opts.fieldCell : 1.0, 0.55, 1.6);
    let n = Math.ceil((2 * half) / cell);
    const maxN = Math.min(88, Math.floor(Math.sqrt(gl.getParameter(gl.MAX_TEXTURE_SIZE))));
    if (n > maxN) { cell = (2 * half) / maxN; n = maxN; }
    // One field layer per (element type, component): LJ, screened Coulomb,
    // apolar burial, polar desolvation.
    const fieldLayers = probes.length * 4;
    const maxVoxels = Math.floor(96e6 / (16 * fieldLayers));
    if (n * n * n > maxVoxels) {
      n = Math.max(24, Math.floor(Math.cbrt(maxVoxels)));
      cell = (2 * half) / n;
    }
    this.fieldCell = cell;
    this.fieldDims = [n, n, n];
    this.fieldMin = [fieldCenter[0] - half, fieldCenter[1] - half, fieldCenter[2] - half];
    if (fieldLayers > gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS)) {
      throw new Error('This ligand has too many element types for the interaction field.');
    }
    this.fieldTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.fieldTex);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA32F, n, n * n, fieldLayers, 0, gl.RGBA, gl.FLOAT, null);
    this.fieldFBO = gl.createFramebuffer();

    // Pre-baked composite field texture: 1 layer per ligand element probe
    this.compositeTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.compositeTex);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA32F, n, n * n, probes.length, 0, gl.RGBA, gl.FLOAT, null);
    this.compositeFBO = gl.createFramebuffer();

    // receptor point VBO: pos + element index
    const recVerts = new Float32Array(recN * 4);
    for (let i = 0; i < recN; i++) {
      recVerts[i * 4] = target.receptor.pos[i * 3];
      recVerts[i * 4 + 1] = target.receptor.pos[i * 3 + 1];
      recVerts[i * 4 + 2] = target.receptor.pos[i * 3 + 2];
      recVerts[i * 4 + 3] = target.receptor.element[i];
    }
    this.recVBO = this._buffer(recVerts);
    this.recVAO = this._vao(this.recVBO, [
      { loc: 0, size: 3, stride: 16, offset: 0 },
      { loc: 1, size: 1, stride: 16, offset: 12 }
    ]);

    // ligand base VBO: local offset + render radius + element
    const ligVerts = new Float32Array(ligN * 5);
    for (let i = 0; i < ligN; i++) {
      ligVerts[i * 5] = target.ligand.offset[i * 3];
      ligVerts[i * 5 + 1] = target.ligand.offset[i * 3 + 1];
      ligVerts[i * 5 + 2] = target.ligand.offset[i * 3 + 2];
      ligVerts[i * 5 + 3] = target.ligand.radius[i];
      ligVerts[i * 5 + 4] = target.ligand.element[i];
    }
    this.ligVBO = this._buffer(ligVerts);
    this.ligVAO = this._vao(this.ligVBO, [
      { loc: 0, size: 3, stride: 20, offset: 0 },
      { loc: 1, size: 1, stride: 20, offset: 12 },
      { loc: 2, size: 1, stride: 20, offset: 16 }
    ]);

    // CA trace
    const trace = target.receptor.trace;
    this.traceVBO = trace.length ? this._buffer(trace) : null;
    this.traceVAO = trace.length ? this._vao(this.traceVBO, [{ loc: 0, size: 3, stride: 12, offset: 0 }]) : null;
    this.caVBO = target.receptor.caCount ? this._buffer(target.receptor.caPos) : null;
    this.caVAO = target.receptor.caCount ? this._vao(this.caVBO, [{ loc: 0, size: 3, stride: 12, offset: 0 }]) : null;
    this.caCount = target.receptor.caCount || 0;

    // ligand bonds as screen-space quads; the vertex shader applies the same
    // torsion chain as the atoms, so the bond VBO only stores atom indices.
    const nb = target.ligand.bonds.length / 2;
    const bondData = new Float32Array(nb * 6 * 4);
    let o = 0;
    for (let b = 0; b < nb; b++) {
      const ia = target.ligand.bonds[b * 2];
      const ib = target.ligand.bonds[b * 2 + 1];
      const quad = [
        [0, -1], [0, 1], [1, 1],
        [0, -1], [1, 1], [1, -1]
      ];
      for (const q of quad) {
        bondData[o++] = ia;
        bondData[o++] = ib;
        bondData[o++] = q[0];
        bondData[o++] = q[1];
      }
    }
    this.bondVBO = this._buffer(bondData);
    this.bondVAO = this._vao(this.bondVBO, [
      { loc: 0, size: 1, stride: 16, offset: 0 },
      { loc: 1, size: 1, stride: 16, offset: 4 },
      { loc: 2, size: 1, stride: 16, offset: 8 },
      { loc: 3, size: 1, stride: 16, offset: 12 }
    ]);
    this.bondCount = nb * 6;

    // search-volume circles
    const sc = target.ligand.siteCenter;
    const rr = target.ligand.searchRadius * 0.72;
    const seg = 72;
    const circles = new Float32Array(3 * seg * 2 * 3);
    let k = 0;
    for (let axis = 0; axis < 3; axis++) {
      for (let s = 0; s < seg; s++) {
        for (let e = 0; e < 2; e++) {
          const a = ((s + e) / seg) * Math.PI * 2;
          const c = Math.cos(a) * rr, sn = Math.sin(a) * rr;
          let x = sc[0], y = sc[1], z = sc[2];
          if (axis === 0) { y += c; z += sn; }
          else if (axis === 1) { x += c; z += sn; }
          else { x += c; y += sn; }
          circles[k++] = x; circles[k++] = y; circles[k++] = z;
        }
      }
    }
    this.circleVBO = this._buffer(circles);
    this.circleVAO = this._vao(this.circleVBO, [{ loc: 0, size: 3, stride: 12, offset: 0 }]);
    this.circleCount = 3 * seg * 2;

    this.recCount = recN;
    this.ligCount = ligN;
  }

  _vao(buffer, attrs) {
    const gl = this.gl;
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    for (const a of attrs) {
      gl.enableVertexAttribArray(a.loc);
      gl.vertexAttribPointer(a.loc, a.size, gl.FLOAT, false, a.stride, a.offset);
    }
    gl.bindVertexArray(null);
    return vao;
  }

  _disposeTargetGL() {
    const gl = this.gl;
    const textures = [this.recTex, this.recParTex, this.cellMetaTex, this.cellAtomTex, this.ligOffTex, this.ligParTex, this.atomMaskTex, this.fieldTex, this.compositeTex];
    for (const t of textures) if (t) gl.deleteTexture(t);
    if (this.fieldFBO) gl.deleteFramebuffer(this.fieldFBO);
    if (this.compositeFBO) gl.deleteFramebuffer(this.compositeFBO);
    this.fieldTex = null;
    this.fieldFBO = null;
    this.compositeTex = null;
    this.compositeFBO = null;
    const buffers = [this.recVBO, this.ligVBO, this.traceVBO, this.caVBO, this.bondVBO, this.circleVBO];
    for (const b of buffers) if (b) gl.deleteBuffer(b);
    const vaos = [this.recVAO, this.ligVAO, this.traceVAO, this.caVAO, this.bondVAO, this.circleVAO];
    for (const v of vaos) if (v) gl.deleteVertexArray(v);
    this.recTex = null;
  }

  _uploadPoses(init) {
    const gl = this.gl;
    const srcs = [init.pos, init.quat, init.vel, init.ang,
      init.torsA, init.torsB, init.torsVelA, init.torsVelB];
    for (let b = 0; b < 2; b++) {
      for (let j = 0; j < 8; j++) {
        gl.bindTexture(gl.TEXTURE_2D, this.poseTex[b][j]);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, MAX_POSES, 1, gl.RGBA, gl.FLOAT, srcs[j]);
      }
    }
    this.poseRead = 0;
  }

  _resetStats() {
    this.best = null;
    this.history.length = 0;
    this.activeCount = 0;
    this.movingCount = 0;
    this.meanEnergy = 0;
    this.energyDelta = null;
    this.converged = false;
    this.stableSince = 0;
    this.prevEnergies.fill(NaN);
    this.eMin = -20;
    this.eMax = 20;
    this.pboReady = false;
    this.pboWrite = 0;
  }

  // ---------------------------------------------------------- controls ----

  setOptions(partial) {
    const before = this.poseCount;
    Object.assign(this.opts, partial);
    if (partial.temperature !== undefined || partial.sleep !== undefined) {
      this.converged = false;
      this.stableSince = 0;
    }
    if (partial.renderScale !== undefined) this.renderScale = partial.renderScale;
    if (partial.poseCount !== undefined) {
      const n = clamp(Math.round(partial.poseCount), 64, MAX_POSES);
      if (n !== before) this.setPoseCount(n);
    }
    if (partial.ljScale !== undefined || partial.esScale !== undefined ||
        partial.burial !== undefined || partial.solvent !== undefined) {
      this._updateCompositeField();
    }
  }

  setPoseCount(n) {
    this.poseCount = clamp(Math.round(n), 64, MAX_POSES);
    this.opts.poseCount = this.poseCount;
    if (this.target) this.reset(true);
  }

  reset(newSeed = true) {
    if (!this.target) return;
    if (newSeed) this.seed = (Math.imul(this.seed, 1664525) + 1013904223) >>> 0;
    const init = makeInitialPoses(this.target, this.poseCount, this.seed, this.opts.seedFraction);
    this.grid = init.grid;
    this._uploadPoses(init);
    this._resetStats();
    this.frame = 0;
  }

  reheat() {
    this.reheatFrames = 3;
  }

  focusBest() {
    if (!this.best) return;
    this.camera.target = this.best.pos.slice();
    this.camera.distance = Math.min(this.camera.distance, 62);
  }

  resetCamera() {
    if (!this.target) return;
    this.camera.target = this.target.ligand.siteCenter.slice();
    this.camera.distance = Math.max(95, this.target.ligand.searchRadius * 7.0);
    this.camera.yaw = 0.72;
    this.camera.pitch = -0.22;
  }

  resize(cssW, cssH, dpr = 1) {
    const scale = this.renderScale || 1;
    const w = Math.max(1, Math.round(cssW * dpr * scale));
    const h = Math.max(1, Math.round(cssH * dpr * scale));
    if (w === this.canvas.width && h === this.canvas.height) return;
    this.canvas.width = w;
    this.canvas.height = h;
    this.width = w;
    this.height = h;
    this.aspect = w / h;
    this.pixelScale = h / (2 * Math.tan((45 * Math.PI / 180) / 2));
  }

  // ------------------------------------------------------------- step ----

  step() {
    if (!this.target) return;
    const gl = this.gl;
    const o = this.opts;
    const substeps = clamp(Math.round(o.substeps), 1, 4);

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);

    for (let s = 0; s < substeps; s++) {
      this._forcePass();
      this._integratePass();
      this.frame++;
      if (this.reheatFrames > 0) this.reheatFrames--;
    }

    if (this.frame % this.readEvery === 0) this._readStats();
    gl.enable(gl.DEPTH_TEST);
  }

  _bindPoseTextures(prog, withForce = false) {
    const gl = this.gl;
    const u = prog.u;
    const poseNames = ['uPosePos', 'uPoseQuat', 'uPoseVel', 'uPoseAng',
      'uPoseTorsA', 'uPoseTorsB', 'uPoseTorsVelA', 'uPoseTorsVelB'];
    for (let i = 0; i < 8; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, this.poseTex[this.poseRead][i]);
      gl.uniform1i(u[poseNames[i]], i);
    }
    if (withForce) {
      const forceNames = ['uForce', 'uTorque', 'uForceTorsA', 'uForceTorsB'];
      for (let i = 0; i < 4; i++) {
        gl.activeTexture(gl.TEXTURE8 + i);
        gl.bindTexture(gl.TEXTURE_2D, this.forceTex[i]);
        gl.uniform1i(u[forceNames[i]], 8 + i);
      }
    }
  }

  _setTorsionUniforms(prog) {
    const gl = this.gl;
    const u = prog.u;
    const tor = this.target && this.target.ligand.torsions;
    const enabled = this.opts.torsions !== false && tor;
    gl.uniform1i(u.uTorsionCount, enabled ? tor.count : 0);
    if (u.uTorsionAxis) gl.uniform4fv(u.uTorsionAxis, this.torsionAxisData);
    if (u.uTorsionPivot) gl.uniform4fv(u.uTorsionPivot, this.torsionPivotData);
    if (u.uTorsionInvInertia) gl.uniform1fv(u.uTorsionInvInertia, this.torsionInvInertiaData);
  }

  _buildField() {
    const gl = this.gl;
    const p = this.progField;
    const g = this.grid;
    const [nx, ny, nz] = this.fieldDims;
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fieldFBO);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3]);
    gl.viewport(0, 0, nx, ny * nz);
    gl.useProgram(p.program);

    const bind = (loc, unit, tex) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    bind(p.u.uRecPos, 0, this.recTex);
    bind(p.u.uRecParam, 1, this.recParTex);
    bind(p.u.uCellMeta, 2, this.cellMetaTex);
    bind(p.u.uCellAtoms, 3, this.cellAtomTex);

    gl.uniform3fv(p.u.uGridMin, g.min);
    gl.uniform3i(p.u.uGridDims, g.dims[0], g.dims[1], g.dims[2]);
    gl.uniform1f(p.u.uCellSize, g.cellSize);
    gl.uniform1f(p.u.uCutoff2, this.opts.cutoff * this.opts.cutoff);
    gl.uniform3fv(p.u.uFieldMin, this.fieldMin);
    gl.uniform3i(p.u.uFieldDims, nx, ny, nz);
    gl.uniform1f(p.u.uFieldCell, this.fieldCell);
    gl.uniform1f(p.u.uMaxForce, 2800.0);
    gl.uniform1f(p.u.uDielectric, this.opts.dielectric != null ? this.opts.dielectric : 4.0);
    gl.uniform1f(p.u.uLambda, this.opts.dielectricLambda != null ? this.opts.dielectricLambda : 2.0);
    gl.uniform1f(p.u.uKappa, this.opts.screeningKappa != null ? this.opts.screeningKappa : 0.12);

    for (let layer = 0; layer < this.fieldLayerCount; layer++) {
      const probe = this.fieldProbes[layer];
      const base = layer * 4;
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, this.fieldTex, 0, base);
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, this.fieldTex, 0, base + 1);
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, this.fieldTex, 0, base + 2);
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, this.fieldTex, 0, base + 3);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Incomplete interaction field framebuffer: 0x' + status.toString(16));
      }
      gl.uniform1f(p.u.uProbeSigma, probe.sigma);
      gl.uniform1f(p.u.uProbeEps, probe.eps);
      gl.uniform1f(p.u.uProbeQ, probe.q);
      gl.uniform1f(p.u.uProbePolar, probe.polar);
      gl.bindVertexArray(this.quadVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
    this._updateCompositeField();
  }

  _updateCompositeField() {
    if (!this.target || !this.compositeTex || !this.fieldTex) return;
    const gl = this.gl;
    const p = this.progCombine;
    const o = this.opts;
    const [nx, ny, nz] = this.fieldDims;

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.compositeFBO);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, nx, ny * nz);
    gl.useProgram(p.program);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.fieldTex);
    gl.uniform1i(p.u.uField, 0);

    gl.uniform1f(p.u.uLJScale, o.ljScale != null ? o.ljScale : 1.0);
    gl.uniform1f(p.u.uEScale, o.esScale != null ? o.esScale : 1.0);
    gl.uniform1f(p.u.uBurial, o.burial != null ? o.burial : 0.0);
    gl.uniform1f(p.u.uSolvent, o.solvent != null ? o.solvent : 0.0);

    for (let layer = 0; layer < this.fieldLayerCount; layer++) {
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, this.compositeTex, 0, layer);
      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Incomplete composite field framebuffer: 0x' + status.toString(16));
      }
      gl.uniform1i(p.u.uLayerBase, layer * 4);
      gl.bindVertexArray(this.quadVAO);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.DEPTH_TEST);
  }

  _forcePass() {
    const gl = this.gl;
    const p = this.progForce;
    const o = this.opts;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.forceFBO);
    gl.viewport(0, 0, this.poseCount, 1);
    gl.useProgram(p.program);
    this._bindPoseTextures(p, false);

    const bind = (loc, unit, tex) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(loc, unit);
    };
    bind(p.u.uLigOffset, 8, this.ligOffTex);
    bind(p.u.uLigParam, 9, this.ligParTex);
    bind(p.u.uAtomMask, 10, this.atomMaskTex);
    gl.activeTexture(gl.TEXTURE11);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.compositeTex);
    gl.uniform1i(p.u.uCompositeField, 11);

    gl.uniform3fv(p.u.uFieldMin, this.fieldMin);
    gl.uniform3i(p.u.uFieldDims, this.fieldDims[0], this.fieldDims[1], this.fieldDims[2]);
    gl.uniform1f(p.u.uFieldCell, this.fieldCell);
    gl.uniform3fv(p.u.uSearchCenter, this.target.ligand.siteCenter);
    gl.uniform1f(p.u.uSearchRadius, this.target.ligand.searchRadius * o.searchScale);
    gl.uniform1f(p.u.uRestraintK, o.restraintK);
    gl.uniform1f(p.u.uTemp, o.temperature);
    gl.uniform1f(p.u.uNoise, 9.0);
    gl.uniform1f(p.u.uTorqueNoise, 1.4);
    gl.uniform1f(p.u.uMaxForce, 2800.0);
    gl.uniform1f(p.u.uFrame, this.frame);
    gl.uniform1f(p.u.uSeed, (this.seed % 10000) * 0.017);
    gl.uniform1f(p.u.uTorsionBarrier, o.torsionBarrier != null ? o.torsionBarrier : 1.5);
    gl.uniform1f(p.u.uTorsionPrior, o.torsionPrior != null ? o.torsionPrior : 8.0);
    gl.uniform1f(p.u.uTorsionDeadZone, o.torsionDeadZone != null ? o.torsionDeadZone : 0.30);
    gl.uniform1i(p.u.uLigCount, this.ligCount);
    this._setTorsionUniforms(p);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  _integratePass() {
    const gl = this.gl;
    const p = this.progIntegrate;
    const o = this.opts;
    const write = 1 - this.poseRead;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.poseFBO[write]);
    gl.viewport(0, 0, this.poseCount, 1);
    gl.useProgram(p.program);
    this._bindPoseTextures(p, true);

    gl.uniform1f(p.u.uDt, 1.0);
    gl.uniform1f(p.u.uInvMass, 1 / Math.max(this.target.mass, 1));
    gl.uniform1f(p.u.uInvInertia, 1 / Math.max(this.target.inertia, 1));
    gl.uniform1f(p.u.uAccelScale, o.accelScale);
    gl.uniform1f(p.u.uAngAccelScale, o.angAccelScale);
    gl.uniform1f(p.u.uDamping, o.damping);
    gl.uniform1f(p.u.uMaxSpeed, 0.55);
    gl.uniform1f(p.u.uMaxSpin, 0.22);
    gl.uniform1f(p.u.uMaxForce, 2800.0);
    gl.uniform1f(p.u.uMaxTorque, 1200.0);
    gl.uniform1f(p.u.uTemp, o.temperature);
    gl.uniform1f(p.u.uNoiseVel, 0.00055);
    gl.uniform1f(p.u.uNoiseAng, 0.00032);
    gl.uniform1f(p.u.uSleepFrames, 80.0);
    gl.uniform1f(p.u.uSleepVel2, SLEEP_VEL2);
    gl.uniform1f(p.u.uSleepSpin2, 0.00006);
    gl.uniform1f(p.u.uSleepEps, 0.08);
    gl.uniform1f(p.u.uWakeProb, o.sleep ? 0.00055 * Math.min(1, Math.max(0, o.temperature) / 300) : 0.0);
    gl.uniform1f(p.u.uUseSleep, o.sleep ? 1.0 : 0.0);
    gl.uniform1f(p.u.uReheat, this.reheatFrames > 0 ? 1.0 : 0.0);
    gl.uniform1f(p.u.uFrame, this.frame);
    gl.uniform1f(p.u.uSeed, (this.seed % 10000) * 0.017);
    gl.uniform1f(p.u.uTorsionAccelScale, o.torsionAccelScale != null ? o.torsionAccelScale : 0.028);
    gl.uniform1f(p.u.uTorsionDamping, o.torsionDamping != null ? o.torsionDamping : 0.90);
    gl.uniform1f(p.u.uMaxTorsionVel, o.maxTorsionVel != null ? o.maxTorsionVel : 0.09);
    if (this.best && Number.isFinite(this.best.energy)) {
      gl.uniform3fv(p.u.uBestPos, this.best.pos);
      gl.uniform4fv(p.u.uBestQuat, this.best.quat);
      gl.uniform1f(p.u.uBestEnergy, this.best.energy);
      gl.uniform1f(p.u.uHasBest, 1.0);
    } else {
      gl.uniform3f(p.u.uBestPos, 0, 0, 0);
      gl.uniform4f(p.u.uBestQuat, 0, 0, 0, 1);
      gl.uniform1f(p.u.uBestEnergy, 9999.0);
      gl.uniform1f(p.u.uHasBest, 0.0);
    }
    this._setTorsionUniforms(p);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    this.poseRead = write;
  }

  _readStats() {
    const gl = this.gl;
    const n = this.poseCount;
    const sliceBytes = MAX_POSES * 16;
    const writeIdx = this.pboWrite;
    const readIdx = 1 - writeIdx;

    if (this.pboReady) {
      // Non-blocking asynchronous read from previously queued PBO
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo[readIdx]);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.energyRead, 0, n * 4);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, sliceBytes, this.posRead, 0, n * 4);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, sliceBytes * 2, this.quatRead, 0, n * 4);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, sliceBytes * 3, this.angRead, 0, n * 4);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, sliceBytes * 4, this.torsARead, 0, n * 4);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, sliceBytes * 5, this.torsBRead, 0, n * 4);
      gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, sliceBytes * 6, this.velRead, 0, n * 4);
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      this._processStats(n);
    } else {
      // Synchronous bootstrap on initial frame so user sees metrics immediately
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.forceFBO);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, this.energyRead);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.poseFBO[this.poseRead]);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, this.posRead);
      gl.readBuffer(gl.COLOR_ATTACHMENT1);
      gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, this.quatRead);
      gl.readBuffer(gl.COLOR_ATTACHMENT2);
      gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, this.velRead);
      gl.readBuffer(gl.COLOR_ATTACHMENT3);
      gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, this.angRead);
      gl.readBuffer(gl.COLOR_ATTACHMENT4);
      gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, this.torsARead);
      gl.readBuffer(gl.COLOR_ATTACHMENT5);
      gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, this.torsBRead);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this._processStats(n);
    }

    // Queue asynchronous DMA transfer into active PBO for next read cycle
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbo[writeIdx]);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.forceFBO);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, 0);

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.poseFBO[this.poseRead]);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, sliceBytes);
    gl.readBuffer(gl.COLOR_ATTACHMENT1);
    gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, sliceBytes * 2);
    gl.readBuffer(gl.COLOR_ATTACHMENT3);
    gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, sliceBytes * 3);
    gl.readBuffer(gl.COLOR_ATTACHMENT4);
    gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, sliceBytes * 4);
    gl.readBuffer(gl.COLOR_ATTACHMENT5);
    gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, sliceBytes * 5);
    gl.readBuffer(gl.COLOR_ATTACHMENT2);
    gl.readPixels(0, 0, n, 1, gl.RGBA, gl.FLOAT, sliceBytes * 6);

    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    this.pboWrite = readIdx;
    this.pboReady = true;
  }

  _processStats(n) {
    let bestE = Infinity, bestI = -1;
    let active = 0, activeE = 0, moving = 0;
    let deltaSum = 0, deltaCount = 0;
    const prevE = this.prevEnergies;

    for (let i = 0; i < n; i++) {
      const e = this.energyRead[i * 4 + 3];
      if (Number.isFinite(e)) {
        if (e < bestE) { bestE = e; bestI = i; }
        if (Number.isFinite(prevE[i])) {
          deltaSum += Math.abs(e - prevE[i]);
          deltaCount++;
        }
        prevE[i] = e;
      } else {
        prevE[i] = NaN;
      }

      const vx = this.velRead[i * 4];
      const vy = this.velRead[i * 4 + 1];
      const vz = this.velRead[i * 4 + 2];
      if (vx * vx + vy * vy + vz * vz > SLEEP_VEL2) moving++;

      if (this.angRead[i * 4 + 3] > 0.5) {
        active++;
        if (Number.isFinite(e) && e < 500) this.activeEnergies[activeE++] = e;
      }
    }

    this.activeCount = active;
    this.movingCount = moving;
    this.energyDelta = deltaCount ? deltaSum / deltaCount : null;
    if (activeE) {
      const arr = this.activeEnergies.subarray(0, activeE);
      arr.sort();
      this.meanEnergy = arr[activeE >> 1];
    } else {
      this.meanEnergy = null;
    }

    // A quench is converged when the swarm is fully asleep, or when both the
    // mean |ΔE| and the number of moving poses have been flat for 2 seconds.
    const nowMs = performance.now();
    const quiet = Number.isFinite(this.energyDelta) && this.energyDelta < 0.05 && moving === 0;
    if (active === 0 && bestI >= 0) {
      this.converged = true;
      this.stableSince = nowMs;
    } else if (quiet) {
      if (!this.stableSince) this.stableSince = nowMs;
      if (nowMs - this.stableSince >= 2000) this.converged = true;
    } else {
      this.stableSince = 0;
      this.converged = false;
    }
    if (bestI >= 0) {
      const pos = [
        this.posRead[bestI * 4],
        this.posRead[bestI * 4 + 1],
        this.posRead[bestI * 4 + 2]
      ];
      const quat = [
        this.quatRead[bestI * 4],
        this.quatRead[bestI * 4 + 1],
        this.quatRead[bestI * 4 + 2],
        this.quatRead[bestI * 4 + 3]
      ];
      // Only report torsion angles when the GPU actually applies them; in rigid
      // mode the stored angles are stale and would corrupt the RMSD.
      const flexOn = this.opts.torsions !== false && this.target.ligand.torsions;
      const nTors = flexOn ? this.target.ligand.torsions.count : 0;
      const tors = new Float32Array(nTors);
      for (let k = 0; k < nTors; k++) {
        tors[k] = k < 4 ? this.torsARead[bestI * 4 + k] : this.torsBRead[bestI * 4 + k - 4];
      }
      const rmsd = rmsdToCrystal(this.target, pos, quat, tors);
      this.best = { index: bestI, energy: bestE, pos, quat, tors, rmsd };
      this.eMin = bestE;
      this.eMax = bestE + 42;
    }

    const now = performance.now();
    this.history.push({ t: now, best: bestE, mean: this.meanEnergy, active });
    if (this.history.length > 300) this.history.shift();

    if (this.onstats) {
      this.onstats({
        fps: this.fps,
        poses: n,
        atoms: (this.recCount || 0) + n * (this.ligCount || 0),
        active,
        awake: active,
        moving,
        energyDelta: this.energyDelta,
        converged: this.converged,
        best: Number.isFinite(bestE) ? bestE : null,
        mean: this.meanEnergy,
        rmsd: this.best ? this.best.rmsd : null,
        temperature: this.opts.temperature,
        history: this.history,
        bestPos: this.best ? this.best.pos : null,
        torsions: this.target.ligand.torsions ? this.target.ligand.torsions.count : 0,
        solvent: this.opts.solvent
      });
    }
  }

  // ------------------------------------------------------------ render ----

  _cameraMatrices() {
    const c = this.camera;
    const cp = Math.cos(c.pitch), sp = Math.sin(c.pitch);
    const eye = [
      c.target[0] + c.distance * cp * Math.sin(c.yaw),
      c.target[1] + c.distance * sp,
      c.target[2] + c.distance * cp * Math.cos(c.yaw)
    ];
    const view = lookAt(eye, c.target, [0, 1, 0]);
    const proj = perspective(45 * Math.PI / 180, this.aspect, 0.6, 600);
    return { view, viewProj: multiply(proj, view) };
  }

  render() {
    if (!this.target) return;
    const gl = this.gl;
    const o = this.opts;
    if (o.autoRotate) this.camera.yaw += 0.00055;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    const { view, viewProj } = this._cameraMatrices();

    if (o.showReceptor) this._drawReceptor(view, viewProj);

    // backbone trace + search volume
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    if (o.showLines) {
      if (this.caVAO) {
        this._drawDots(this.caVAO, this.caCount, view, viewProj, [0.62, 0.95, 1.0], 0.38, 1.5);
      }
      if (this.traceVAO) {
        this._drawLines(this.traceVAO, this.target.receptor.trace.length / 3, viewProj, [0.37, 0.91, 1.0], 0.10);
      }
      this._drawLines(this.circleVAO, this.circleCount, viewProj, [0.37, 0.91, 1.0], 0.14);
    }

    if (o.showSwarm) this._drawSwarm(view, viewProj);
    gl.depthMask(true);

    // best + crystal overlay: draw after clearing depth so the pose is legible
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    if (o.showCrystal) this._drawGhost(view, viewProj);
    if (o.showBest && this.best) this._drawBest(view, viewProj);

    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
  }

  _drawReceptor(view, viewProj) {
    const gl = this.gl;
    const p = this.progReceptor;
    gl.useProgram(p.program);
    gl.uniformMatrix4fv(p.u.uViewProj, false, viewProj);
    gl.uniformMatrix4fv(p.u.uView, false, view);
    gl.uniform1f(p.u.uPixelScale, this.pixelScale);

    gl.bindVertexArray(this.recVAO);

    // soft molecular volume
    if (this.opts.showSurface) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.depthMask(false);
      gl.uniform1f(p.u.uSurface, 1.0);
      gl.uniform1f(p.u.uAlpha, 0.055);
      gl.uniform1f(p.u.uSizeScale, 1.42);
      gl.drawArrays(gl.POINTS, 0, this.recCount);
    }

    // crisp atoms
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(true);
    gl.uniform1f(p.u.uSurface, 0.0);
    gl.uniform1f(p.u.uAlpha, 0.95);
    gl.uniform1f(p.u.uSizeScale, 0.86);
    gl.drawArrays(gl.POINTS, 0, this.recCount);

    gl.bindVertexArray(null);
    gl.depthMask(false);
  }

  _drawSwarm(view, viewProj) {
    const gl = this.gl;
    const p = this.progSwarm;
    gl.useProgram(p.program);
    this._bindPoseTextures(p, false);
    gl.activeTexture(gl.TEXTURE8);
    gl.bindTexture(gl.TEXTURE_2D, this.atomMaskTex);
    gl.uniform1i(p.u.uAtomMask, 8);
    this._setTorsionUniforms(p);
    gl.uniformMatrix4fv(p.u.uViewProj, false, viewProj);
    gl.uniformMatrix4fv(p.u.uView, false, view);
    gl.uniform1f(p.u.uPixelScale, this.pixelScale);
    gl.uniform1f(p.u.uSizeScale, 1.0);
    gl.uniform3f(p.u.uLowColor, 0.78, 1.0, 0.37);
    gl.uniform3f(p.u.uHighColor, 1.0, 0.25, 0.55);
    gl.uniform1f(p.u.uAlpha, 0.19);
    gl.uniform1f(p.u.uEMin, this.eMin);
    gl.uniform1f(p.u.uEMax, this.eMax);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.depthMask(false);
    gl.bindVertexArray(this.ligVAO);
    gl.drawArraysInstanced(gl.POINTS, 0, this.ligCount, this.poseCount);
    gl.bindVertexArray(null);
  }

  _drawBest(view, viewProj) {
    const gl = this.gl;
    const p = this.progBest;
    const b = this.best;
    gl.useProgram(p.program);
    this._setTorsionUniforms(p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atomMaskTex);
    gl.uniform1i(p.u.uAtomMask, 0);
    gl.uniform1fv(p.u.uTorsionAngles, b.tors.length ? b.tors : this.zeroTors);
    gl.uniformMatrix4fv(p.u.uViewProj, false, viewProj);
    gl.uniformMatrix4fv(p.u.uView, false, view);
    gl.uniform3fv(p.u.uPos, b.pos);
    gl.uniform4fv(p.u.uQuat, b.quat);
    gl.uniform1f(p.u.uPixelScale, this.pixelScale);
    gl.uniform1f(p.u.uSizeScale, 1.5);
    gl.uniform1f(p.u.uAlpha, 1.0);

    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    gl.bindVertexArray(this.ligVAO);
    gl.drawArrays(gl.POINTS, 0, this.ligCount);
    gl.bindVertexArray(null);

    // bonds
    if (this.bondCount) {
      const pb = this.progBond;
      gl.useProgram(pb.program);
      this._setTorsionUniforms(pb);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.ligOffTex);
      gl.uniform1i(pb.u.uLigOffset, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.atomMaskTex);
      gl.uniform1i(pb.u.uAtomMask, 1);
      gl.uniform1fv(pb.u.uTorsionAngles, b.tors.length ? b.tors : this.zeroTors);
      gl.uniformMatrix4fv(pb.u.uViewProj, false, viewProj);
      gl.uniform3fv(pb.u.uPos, b.pos);
      gl.uniform4fv(pb.u.uQuat, b.quat);
      gl.uniform2f(pb.u.uViewport, this.width, this.height);
      gl.uniform1f(pb.u.uBondWidth, Math.max(1.4, this.height / 700));
      gl.uniform3f(pb.u.uColor, 0.78, 1.0, 0.48);
      gl.uniform1f(pb.u.uAlpha, 0.82);
      gl.bindVertexArray(this.bondVAO);
      gl.drawArrays(gl.TRIANGLES, 0, this.bondCount);
      gl.bindVertexArray(null);
    }
  }

  _drawGhost(view, viewProj) {
    const gl = this.gl;
    const crystal = crystalPose(this.target);
    const p = this.progBest;
    gl.useProgram(p.program);
    this._setTorsionUniforms(p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.atomMaskTex);
    gl.uniform1i(p.u.uAtomMask, 0);
    gl.uniform1fv(p.u.uTorsionAngles, crystal.tors);
    gl.uniformMatrix4fv(p.u.uViewProj, false, viewProj);
    gl.uniformMatrix4fv(p.u.uView, false, view);
    gl.uniform3fv(p.u.uPos, crystal.pos);
    gl.uniform4fv(p.u.uQuat, crystal.quat);
    gl.uniform1f(p.u.uPixelScale, this.pixelScale);
    gl.uniform1f(p.u.uSizeScale, 0.72);
    gl.uniform1f(p.u.uAlpha, 0.20);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    gl.bindVertexArray(this.ligVAO);
    gl.drawArrays(gl.POINTS, 0, this.ligCount);
    gl.bindVertexArray(null);

    if (this.bondCount) {
      const pb = this.progBond;
      gl.useProgram(pb.program);
      this._setTorsionUniforms(pb);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.ligOffTex);
      gl.uniform1i(pb.u.uLigOffset, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.atomMaskTex);
      gl.uniform1i(pb.u.uAtomMask, 1);
      gl.uniform1fv(pb.u.uTorsionAngles, crystal.tors);
      gl.uniformMatrix4fv(pb.u.uViewProj, false, viewProj);
      gl.uniform3fv(pb.u.uPos, crystal.pos);
      gl.uniform4fv(pb.u.uQuat, crystal.quat);
      gl.uniform2f(pb.u.uViewport, this.width, this.height);
      gl.uniform1f(pb.u.uBondWidth, Math.max(1.0, this.height / 900));
      gl.uniform3f(pb.u.uColor, 0.65, 0.95, 1.0);
      gl.uniform1f(pb.u.uAlpha, 0.22);
      gl.bindVertexArray(this.bondVAO);
      gl.drawArrays(gl.TRIANGLES, 0, this.bondCount);
      gl.bindVertexArray(null);
    }
  }

  _drawDots(vao, count, view, viewProj, color, alpha, size) {
    if (!vao || !count) return;
    const gl = this.gl;
    const p = this.progDot;
    gl.useProgram(p.program);
    gl.uniformMatrix4fv(p.u.uViewProj, false, viewProj);
    gl.uniformMatrix4fv(p.u.uView, false, view);
    gl.uniform1f(p.u.uPixelScale, this.pixelScale);
    gl.uniform1f(p.u.uSizeScale, size);
    gl.uniform3fv(p.u.uColor, color);
    gl.uniform1f(p.u.uAlpha, alpha);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.bindVertexArray(null);
  }

  _drawLines(vao, count, viewProj, color, alpha) {
    if (!vao || !count) return;
    const gl = this.gl;
    const p = this.progLine;
    gl.useProgram(p.program);
    gl.uniformMatrix4fv(p.u.uViewProj, false, viewProj);
    gl.uniform3fv(p.u.uColor, color);
    gl.uniform1f(p.u.uAlpha, alpha);
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.LINES, 0, count);
    gl.bindVertexArray(null);
  }

  dispose() {
    const gl = this.gl;
    this._disposeTargetGL();
    for (const tex of this.forceTex) gl.deleteTexture(tex);
    gl.deleteFramebuffer(this.forceFBO);
    for (const arr of this.poseTex) for (const t of arr) gl.deleteTexture(t);
    for (const f of this.poseFBO) gl.deleteFramebuffer(f);
    for (const p of [this.progField, this.progCombine, this.progForce, this.progIntegrate, this.progReceptor, this.progSwarm, this.progBest, this.progLine, this.progBond, this.progDot]) {
      if (p && p.program) gl.deleteProgram(p.program);
    }
    if (this.pbo) {
      if (this.pbo[0]) gl.deleteBuffer(this.pbo[0]);
      if (this.pbo[1]) gl.deleteBuffer(this.pbo[1]);
    }
    gl.deleteBuffer(this.quadVBO);
    gl.deleteVertexArray(this.quadVAO);
  }
}
