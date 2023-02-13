/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
// Please make sure the shaker key in makeShaderKey in gpgpu_math.ts is well
// mapped if any shader source code is changed in this file.
import { backend_util, util } from '@tensorflow/tfjs-core';
const { getBroadcastDims } = backend_util;
import { getGlslDifferences } from './glsl_version';
import * as shader_util from './shader_compiler_util';
export function makeShader(inputsInfo, outputShape, program) {
    const prefixSnippets = [];
    inputsInfo.forEach(x => {
        const size = util.sizeFromShape(x.shapeInfo.logicalShape);
        // Snippet when we decided to upload the values as uniform.
        if (x.shapeInfo.isUniform) {
            prefixSnippets.push(`uniform float ${x.name}${size > 1 ? `[${size}]` : ''};`);
        }
        else {
            prefixSnippets.push(`uniform sampler2D ${x.name};`);
            prefixSnippets.push(`uniform int offset${x.name};`);
        }
        if (program.enableShapeUniforms) {
            const { uniformShape } = getUniformInfoFromShape(program.packedInputs, x.shapeInfo.logicalShape, x.shapeInfo.texShape);
            switch (uniformShape.length) {
                case 1:
                    prefixSnippets.push(`uniform int ${x.name}Shape;`);
                    break;
                case 2:
                    prefixSnippets.push(`uniform ivec2 ${x.name}Shape;`);
                    break;
                case 3:
                    prefixSnippets.push(`uniform ivec3 ${x.name}Shape;`);
                    break;
                case 4:
                    prefixSnippets.push(`uniform ivec4 ${x.name}Shape;`);
                    break;
                default:
                    break;
            }
            prefixSnippets.push(`uniform ivec2 ${x.name}TexShape;`);
        }
    });
    if (program.enableShapeUniforms) {
        switch (outputShape.logicalShape.length) {
            case 1:
                prefixSnippets.push(`uniform int outShape;`);
                break;
            case 2:
                prefixSnippets.push(`uniform ivec2 outShape;`);
                prefixSnippets.push(`uniform int outShapeStrides;`);
                break;
            case 3:
                prefixSnippets.push(`uniform ivec3 outShape;`);
                prefixSnippets.push(`uniform ivec2 outShapeStrides;`);
                break;
            case 4:
                prefixSnippets.push(`uniform ivec4 outShape;`);
                prefixSnippets.push(`uniform ivec3 outShapeStrides;`);
                break;
            default:
                break;
        }
        prefixSnippets.push(`uniform ivec2 outTexShape;`);
    }
    if (program.customUniforms) {
        program.customUniforms.forEach((d) => {
            prefixSnippets.push(`uniform ${d.type} ${d.name}${d.arrayIndex ? `[${d.arrayIndex}]` : ''};`);
        });
    }
    const inputPrefixSnippet = prefixSnippets.join('\n');
    const inputSamplingSnippet = inputsInfo
        .map(x => getInputSamplingSnippet(x, outputShape, program.packedInputs, program.enableShapeUniforms))
        .join('\n');
    const outTexShape = outputShape.texShape;
    const glsl = getGlslDifferences();
    const floatTextureSampleSnippet = getFloatTextureSampleSnippet(glsl);
    let outputSamplingSnippet;
    let floatTextureSetOutputSnippet;
    let shaderPrefix = getShaderPrefix(glsl);
    if (outputShape.isPacked) {
        outputSamplingSnippet = getPackedOutputSamplingSnippet(outputShape.logicalShape, outTexShape, program.enableShapeUniforms);
        floatTextureSetOutputSnippet = getFloatTextureSetRGBASnippet(glsl);
    }
    else {
        outputSamplingSnippet = getOutputSamplingSnippet(outputShape.logicalShape, outTexShape, program.enableShapeUniforms);
        floatTextureSetOutputSnippet = getFloatTextureSetRSnippet(glsl);
    }
    if (program.packedInputs) {
        shaderPrefix += SHADER_PACKED_PREFIX;
    }
    const source = [
        shaderPrefix, floatTextureSampleSnippet, floatTextureSetOutputSnippet,
        inputPrefixSnippet, outputSamplingSnippet, inputSamplingSnippet,
        program.userCode
    ].join('\n');
    return source;
}
function getSamplerFromInInfo(inInfo, enableShapeUniforms = false) {
    const shape = inInfo.shapeInfo.logicalShape;
    switch (shape.length) {
        case 0:
            return getSamplerScalar(inInfo, enableShapeUniforms);
        case 1:
            return getSampler1D(inInfo, enableShapeUniforms);
        case 2:
            return getSampler2D(inInfo, enableShapeUniforms);
        case 3:
            return getSampler3D(inInfo, enableShapeUniforms);
        case 4:
            return getSampler4D(inInfo, enableShapeUniforms);
        case 5:
            return getSampler5D(inInfo);
        case 6:
            return getSampler6D(inInfo);
        default:
            throw new Error(`${shape.length}-D input sampling` +
                ` is not yet supported`);
    }
}
function getPackedSamplerFromInInfo(inInfo, enableShapeUniforms) {
    const shape = inInfo.shapeInfo.logicalShape;
    switch (shape.length) {
        case 0:
            return getPackedSamplerScalar(inInfo);
        case 1:
            return getPackedSampler1D(inInfo, enableShapeUniforms);
        case 2:
            return getPackedSampler2D(inInfo, enableShapeUniforms);
        case 3:
            return getPackedSampler3D(inInfo, enableShapeUniforms);
        default:
            return getPackedSamplerND(inInfo, enableShapeUniforms);
    }
}
function getInputSamplingSnippet(inInfo, outShapeInfo, usesPackedTextures = false, enableShapeUniforms) {
    let res = '';
    if (usesPackedTextures) {
        res += getPackedSamplerFromInInfo(inInfo, enableShapeUniforms);
    }
    else {
        res += getSamplerFromInInfo(inInfo, enableShapeUniforms);
    }
    const inShape = inInfo.shapeInfo.logicalShape;
    const outShape = outShapeInfo.logicalShape;
    if (inShape.length <= outShape.length) {
        if (usesPackedTextures) {
            res += getPackedSamplerAtOutputCoords(inInfo, outShapeInfo);
        }
        else {
            res += getSamplerAtOutputCoords(inInfo, outShapeInfo);
        }
    }
    return res;
}
function getPackedOutputSamplingSnippet(outShape, outTexShape, enableShapeUniforms) {
    switch (outShape.length) {
        case 0:
            return getOutputScalarCoords();
        case 1:
            return getOutputPacked1DCoords(outShape, outTexShape, enableShapeUniforms);
        case 2:
            return getOutputPacked2DCoords(outShape, outTexShape, enableShapeUniforms);
        case 3:
            return getOutputPacked3DCoords(outShape, outTexShape, enableShapeUniforms);
        default:
            return getOutputPackedNDCoords(outShape, outTexShape, enableShapeUniforms);
    }
}
function getOutputSamplingSnippet(outShape, outTexShape, enableShapeUniforms) {
    switch (outShape.length) {
        case 0:
            return getOutputScalarCoords();
        case 1:
            return getOutput1DCoords(outShape, outTexShape, enableShapeUniforms);
        case 2:
            return getOutput2DCoords(outShape, outTexShape, enableShapeUniforms);
        case 3:
            return getOutput3DCoords(outShape, outTexShape, enableShapeUniforms);
        case 4:
            return getOutput4DCoords(outShape, outTexShape, enableShapeUniforms);
        case 5:
            return getOutput5DCoords(outShape, outTexShape);
        case 6:
            return getOutput6DCoords(outShape, outTexShape);
        default:
            throw new Error(`${outShape.length}-D output sampling is not yet supported`);
    }
}
function getFloatTextureSampleSnippet(glsl) {
    return `
    float sampleTexture(sampler2D textureSampler, vec2 uv) {
      return ${glsl.texture2D}(textureSampler, uv).r;
    }
  `;
}
function getFloatTextureSetRSnippet(glsl) {
    return `
    void setOutput(float val) {
      ${glsl.output} = vec4(val, 0, 0, 0);
    }
  `;
}
function getFloatTextureSetRGBASnippet(glsl) {
    return `
    void setOutput(vec4 val) {
      ${glsl.output} = val;
    }
  `;
}
function getShaderPrefix(glsl) {
    const SHADER_PREFIX = `${glsl.version}
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    ${glsl.varyingFs} vec2 resultUV;
    ${glsl.defineOutput}
    const vec2 halfCR = vec2(0.5, 0.5);

    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    uniform float NAN;
    ${glsl.defineSpecialNaN}
    ${glsl.defineSpecialInf}
    ${glsl.defineRound}

    int imod(int x, int y) {
      return x - y * (x / y);
    }

    int idiv(int a, int b, float sign) {
      int res = a / b;
      int mod = imod(a, b);
      if (sign < 0. && mod != 0) {
        res -= 1;
      }
      return res;
    }

    //Based on the work of Dave Hoskins
    //https://www.shadertoy.com/view/4djSRW
    #define HASHSCALE1 443.8975
    float random(float seed){
      vec2 p = resultUV * seed;
      vec3 p3  = fract(vec3(p.xyx) * HASHSCALE1);
      p3 += dot(p3, p3.yzx + 19.19);
      return fract((p3.x + p3.y) * p3.z);
    }

    ${SAMPLE_1D_SNIPPET}
    ${SAMPLE_2D_SNIPPET}
    ${SAMPLE_3D_SNIPPET}
  `;
    return SHADER_PREFIX;
}
const SAMPLE_1D_SNIPPET = `
vec2 uvFromFlat(int texNumR, int texNumC, int index) {
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
  int texelIndex = index / 2;
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;
const SAMPLE_2D_SNIPPET = `
vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
  int texNumC, int row, int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;
const SAMPLE_3D_SNIPPET = `
vec2 packedUVfrom3D(int texNumR, int texNumC,
    int texelsInBatch, int texelsInLogicalRow, int b,
    int row, int col) {
  int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = index / texNumC;
  int texC = index - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}
`;
const SHADER_PACKED_PREFIX = `
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  float getChannel(vec4 frag, int dim) {
    float modCoord = mod(float(dim), 2.);
    return modCoord == 0. ? frag.r : frag.g;
  }
`;
function getOutputScalarCoords() {
    return `
    int getOutputCoords() {
      return 0;
    }
  `;
}
function getOutputPacked1DCoords(shape, texShape, enableShapeUniforms) {
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    if (packedTexShape[0] === 1) {
        if (enableShapeUniforms) {
            return `
      int getOutputCoords() {
        return 2 * int(resultUV.x * ceil(float(outTexShape[1]) / 2.0));
      }
    `;
        }
        return `
      int getOutputCoords() {
        return 2 * int(resultUV.x * ${packedTexShape[1]}.0);
      }
    `;
    }
    if (packedTexShape[1] === 1) {
        if (enableShapeUniforms) {
            return `
      int getOutputCoords() {
        return 2 * int(resultUV.y * ceil(float(outTexShape[0]) / 2.0));
      }
    `;
        }
        return `
      int getOutputCoords() {
        return 2 * int(resultUV.y * ${packedTexShape[0]}.0);
      }
    `;
    }
    if (enableShapeUniforms) {
        return `
    int getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      return 2 * (resTexRC.x * packedTexShape[1] + resTexRC.y);
    }
  `;
    }
    return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      return 2 * (resTexRC.x * ${packedTexShape[1]} + resTexRC.y);
    }
  `;
}
function getOutput1DCoords(shape, texShape, enableShapeUniforms) {
    if (texShape[0] === 1) {
        if (enableShapeUniforms) {
            return `
      int getOutputCoords() {
        return int(resultUV.x * float(outTexShape[1]));
      }
    `;
        }
        return `
      int getOutputCoords() {
        return int(resultUV.x * ${texShape[1]}.0);
      }
    `;
    }
    if (texShape[1] === 1) {
        if (enableShapeUniforms) {
            return `
      int getOutputCoords() {
        return int(resultUV.y * float(outTexShape[0]));
      }
    `;
        }
        return `
      int getOutputCoords() {
        return int(resultUV.y * ${texShape[0]}.0);
      }
    `;
    }
    if (enableShapeUniforms) {
        return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      return resTexRC.x * outTexShape[1] + resTexRC.y;
    }
  `;
    }
    return `
    int getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      return resTexRC.x * ${texShape[1]} + resTexRC.y;
    }
  `;
}
function getOutputPacked3DCoords(shape, texShape, enableShapeUniforms) {
    if (enableShapeUniforms) {
        return `
    ivec3 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec3(b, r, c);
    }
  `;
    }
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    const texelsInLogicalRow = Math.ceil(shape[2] / 2);
    const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[1] / 2);
    return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec3(b, r, c);
    }
  `;
}
function getOutput3DCoords(shape, texShape, enableShapeUniforms) {
    if (enableShapeUniforms) {
        const coordsFromIndexSnippet = shader_util.getOutputLogicalCoordinatesFromFlatIndexByUniform(['r', 'c', 'd'], shape);
        return `
  ivec3 getOutputCoords() {
    ivec2 resTexRC = ivec2(resultUV.yx *
                           vec2(outTexShape[0], outTexShape[1]));
    int index = resTexRC.x * outTexShape[1] + resTexRC.y;
    ${coordsFromIndexSnippet}
    return ivec3(r, c, d);
  }
`;
    }
    const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);
    return `
    ivec3 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      ${coordsFromIndexSnippet}
      return ivec3(r, c, d);
    }
  `;
}
function getOutputPackedNDCoords(shape, texShape, enableShapeUniforms) {
    if (enableShapeUniforms) {
        // TODO: support 5d and 6d
        return `
    ivec4 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));
      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;

      int texelsInLogicalRow = int(ceil(float(outShape[3]) / 2.0));
      int texelsInBatch = texelsInLogicalRow * int(ceil(float(outShape[2]) / 2.0));
      int texelsInBatchN = texelsInBatch * outShape[1];

      int b2 = index / texelsInBatchN;
      index -= b2 * texelsInBatchN;

      int b = index / texelsInBatch;
      index -= b * texelsInBatch;

      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec4(b2, b, r, c);
    }
  `;
    }
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    const texelsInLogicalRow = Math.ceil(shape[shape.length - 1] / 2);
    const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[shape.length - 2] / 2);
    let texelsInBatchN = texelsInBatch;
    let batches = ``;
    let coords = 'b, r, c';
    for (let b = 2; b < shape.length - 1; b++) {
        texelsInBatchN *= shape[shape.length - b - 1];
        batches = `
      int b${b} = index / ${texelsInBatchN};
      index -= b${b} * ${texelsInBatchN};
    ` + batches;
        coords = `b${b}, ` + coords;
    }
    return `
    ivec${shape.length} getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

      ${batches}

      int b = index / ${texelsInBatch};
      index -= b * ${texelsInBatch};

      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec${shape.length}(${coords});
    }
  `;
}
function getOutput4DCoords(shape, texShape, enableShapeUniforms) {
    if (enableShapeUniforms) {
        const coordsFromIndexSnippet = shader_util.getOutputLogicalCoordinatesFromFlatIndexByUniform(['r', 'c', 'd', 'd2'], shape);
        return `
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      ${coordsFromIndexSnippet}
      return ivec4(r, c, d, d2);
    }
  `;
    }
    const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2'], shape);
    return `
    ivec4 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      ${coordsFromIndexSnippet}
      return ivec4(r, c, d, d2);
    }
  `;
}
function getOutput5DCoords(shape, texShape) {
    const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2', 'd3'], shape);
    return `
    ivec5 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx * vec2(${texShape[0]},
                             ${texShape[1]}));

      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      ${coordsFromIndexSnippet}

      ivec5 outShape = ivec5(r, c, d, d2, d3);
      return outShape;
    }
  `;
}
function getOutput6DCoords(shape, texShape) {
    const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd', 'd2', 'd3', 'd4'], shape);
    return `
    ivec6 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
        vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;

      ${coordsFromIndexSnippet}

      ivec6 result = ivec6(r, c, d, d2, d3, d4);
      return result;
    }
  `;
}
function getOutputPacked2DCoords(shape, texShape, enableShapeUniforms) {
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    if (util.arraysEqual(shape, texShape)) {
        if (enableShapeUniforms) {
            return `
      ivec2 getOutputCoords() {
        ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
        return 2 * ivec2(resultUV.yx * vec2(packedTexShape[0], packedTexShape[1]));
      }
    `;
        }
        return `
      ivec2 getOutputCoords() {
        return 2 * ivec2(resultUV.yx * vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
      }
    `;
    }
    // texels needed to accommodate a logical row
    const texelsInLogicalRow = Math.ceil(shape[1] / 2);
    /**
     * getOutputCoords
     *
     * resTexRC: The rows and columns of the texels. If you move over one
     * texel to the right in the packed texture, you are moving over one column
     * (not two).
     *
     * index: The texel index
     */
    if (enableShapeUniforms) {
        return `
    ivec2 getOutputCoords() {
      ivec2 packedTexShape = ivec2(ceil(float(outTexShape[0]) / 2.0), ceil(float(outTexShape[1]) / 2.0));
      int texelsInLogicalRow = int(ceil(float(outShape[1]) / 2.0));
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(packedTexShape[0], packedTexShape[1]));

      int index = resTexRC.x * packedTexShape[1] + resTexRC.y;
      int r = 2 * (index / texelsInLogicalRow);
      int c = imod(index, texelsInLogicalRow) * 2;

      return ivec2(r, c);
    }
  `;
    }
    return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${packedTexShape[0]}, ${packedTexShape[1]}));

      int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
      int r = 2 * (index / ${texelsInLogicalRow});
      int c = imod(index, ${texelsInLogicalRow}) * 2;

      return ivec2(r, c);
    }
  `;
}
function getOutput2DCoords(shape, texShape, enableShapeUniforms) {
    if (util.arraysEqual(shape, texShape)) {
        if (enableShapeUniforms) {
            return `
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(outTexShape[0], outTexShape[1]));
      }
    `;
        }
        return `
      ivec2 getOutputCoords() {
        return ivec2(resultUV.yx * vec2(${texShape[0]}, ${texShape[1]}));
      }
    `;
    }
    if (shape[1] === 1) {
        if (enableShapeUniforms) {
            return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(index, 0);
      }
    `;
        }
        return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(index, 0);
      }
    `;
    }
    if (shape[0] === 1) {
        if (enableShapeUniforms) {
            return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(outTexShape[0], outTexShape[1]));
        int index = resTexRC.x * outTexShape[1] + resTexRC.y;
        return ivec2(0, index);
      }
    `;
        }
        return `
      ivec2 getOutputCoords() {
        ivec2 resTexRC = ivec2(resultUV.yx *
                               vec2(${texShape[0]}, ${texShape[1]}));
        int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
        return ivec2(0, index);
      }
    `;
    }
    if (enableShapeUniforms) {
        return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(outTexShape[0], outTexShape[1]));
      int index = resTexRC.x * outTexShape[1] + resTexRC.y;
      int r = index / outShape[1];
      int c = index - r * outShape[1];
      return ivec2(r, c);
    }
  `;
    }
    return `
    ivec2 getOutputCoords() {
      ivec2 resTexRC = ivec2(resultUV.yx *
                             vec2(${texShape[0]}, ${texShape[1]}));
      int index = resTexRC.x * ${texShape[1]} + resTexRC.y;
      int r = index / ${shape[1]};
      int c = index - r * ${shape[1]};
      return ivec2(r, c);
    }
  `;
}
function getFlatOffsetUniformName(texName) {
    return `offset${texName}`;
}
function getPackedSamplerScalar(inputInfo) {
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const glsl = getGlslDifferences();
    return `
    vec4 ${funcName}() {
      return ${glsl.texture2D}(${texName}, halfCR);
    }
  `;
}
function getSamplerScalar(inputInfo, enableShapeUniforms) {
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    if (inputInfo.shapeInfo.isUniform) {
        return `float ${funcName}() {return ${texName};}`;
    }
    const [texNumR, texNumC] = inputInfo.shapeInfo.texShape;
    if (texNumR === 1 && texNumC === 1) {
        return `
      float ${funcName}() {
        return sampleTexture(${texName}, halfCR);
      }
    `;
    }
    const offset = getFlatOffsetUniformName(texName);
    if (enableShapeUniforms) {
        return `
    float ${funcName}() {
      vec2 uv = uvFromFlat(${texName}TexShape[0], ${texName}TexShape[1], ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
    }
    const [tNumR, tNumC] = inputInfo.shapeInfo.texShape;
    return `
    float ${funcName}() {
      vec2 uv = uvFromFlat(${tNumR}, ${tNumC}, ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
}
function getPackedSampler1D(inputInfo, enableShapeUniforms) {
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const texShape = inputInfo.shapeInfo.texShape;
    const glsl = getGlslDifferences();
    if (enableShapeUniforms) {
        return `
    vec4 ${funcName}(int index) {
      ivec2 packedTexShape = ivec2(ceil(float(${texName}TexShape[0]) / 2.0), ceil(float(${texName}TexShape[1]) / 2.0));
      vec2 uv = packedUVfrom1D(
        packedTexShape[0], packedTexShape[1], index);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
    }
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    return `
    vec4 ${funcName}(int index) {
      vec2 uv = packedUVfrom1D(
        ${packedTexShape[0]}, ${packedTexShape[1]}, index);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}
function getSampler1D(inputInfo, enableShapeUniforms) {
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    if (inputInfo.shapeInfo.isUniform) {
        // Uniform arrays will be less than 65505 (no risk of float16 overflow).
        return `
      float ${funcName}(int index) {
        ${getUniformSampler(inputInfo)}
      }
    `;
    }
    const texShape = inputInfo.shapeInfo.texShape;
    const tNumR = texShape[0];
    const tNumC = texShape[1];
    if (tNumC === 1 && tNumR === 1) {
        return `
      float ${funcName}(int index) {
        return sampleTexture(${texName}, halfCR);
      }
    `;
    }
    const offset = getFlatOffsetUniformName(texName);
    if (tNumC === 1) {
        if (enableShapeUniforms) {
            return `
      float ${funcName}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${offset}) + 0.5) / float(${texName}TexShape[0]));
        return sampleTexture(${texName}, uv);
      }
    `;
        }
        return `
      float ${funcName}(int index) {
        vec2 uv = vec2(0.5, (float(index + ${offset}) + 0.5) / ${tNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
    }
    if (tNumR === 1) {
        if (enableShapeUniforms) {
            return `
      float ${funcName}(int index) {
        vec2 uv = vec2((float(index + ${offset}) + 0.5) / float(${texName}TexShape[1]), 0.5);
        return sampleTexture(${texName}, uv);
      }
    `;
        }
        return `
      float ${funcName}(int index) {
        vec2 uv = vec2((float(index + ${offset}) + 0.5) / ${tNumC}.0, 0.5);
        return sampleTexture(${texName}, uv);
      }
    `;
    }
    if (enableShapeUniforms) {
        return `
    float ${funcName}(int index) {
      vec2 uv = uvFromFlat(${texName}TexShape[0], ${texName}TexShape[1], index + ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
    }
    return `
    float ${funcName}(int index) {
      vec2 uv = uvFromFlat(${tNumR}, ${tNumC}, index + ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
}
function getPackedSampler2D(inputInfo, enableShapeUniforms) {
    const shape = inputInfo.shapeInfo.logicalShape;
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const texShape = inputInfo.shapeInfo.texShape;
    const texNumR = texShape[0];
    const texNumC = texShape[1];
    const glsl = getGlslDifferences();
    if (texShape != null && util.arraysEqual(shape, texShape)) {
        if (enableShapeUniforms) {
            return `
      vec4 ${funcName}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${texName}TexShape[1], ${texName}TexShape[0]);

        return ${glsl.texture2D}(${texName}, uv);
      }
    `;
        }
        return `
      vec4 ${funcName}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);

        return ${glsl.texture2D}(${texName}, uv);
      }
    `;
    }
    if (enableShapeUniforms) {
        return `
    vec4 ${funcName}(int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${texName}TexShape[0]) / 2.0), ceil(float(${texName}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${texName}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom2D(valuesPerRow, packedTexShape[0], packedTexShape[1], row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
    }
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    const valuesPerRow = Math.ceil(shape[1] / 2);
    return `
    vec4 ${funcName}(int row, int col) {
      vec2 uv = packedUVfrom2D(${valuesPerRow}, ${packedTexShape[0]}, ${packedTexShape[1]}, row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}
function getSampler2D(inputInfo, enableShapeUniforms) {
    const shape = inputInfo.shapeInfo.logicalShape;
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const texShape = inputInfo.shapeInfo.texShape;
    if (texShape != null && util.arraysEqual(shape, texShape)) {
        if (enableShapeUniforms) {
            return `
      float ${funcName}(int row, int col) {
        vec2 uv = (vec2(col, row) + halfCR) / vec2(${texName}TexShape[1], ${texName}TexShape[0]);
        return sampleTexture(${texName}, uv);
      }
    `;
        }
        const texNumR = texShape[0];
        const texNumC = texShape[1];
        return `
    float ${funcName}(int row, int col) {
      vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
    }
    const { newShape, keptDims } = util.squeezeShape(shape);
    const squeezedShape = newShape;
    if (squeezedShape.length < shape.length) {
        const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
        const params = ['row', 'col'];
        return `
      ${getSamplerFromInInfo(newInputInfo, enableShapeUniforms)}
      float ${funcName}(int row, int col) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
    }
    if (inputInfo.shapeInfo.isUniform) {
        // Uniform arrays will be less than 65505 (no risk of float16 overflow).
        return `
      float ${funcName}(int row, int col) {
        int index = round(dot(vec2(row, col), vec2(${shape[1]}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
    }
    const texNumR = texShape[0];
    const texNumC = texShape[1];
    const offset = getFlatOffsetUniformName(texName);
    if (texNumC === 1) {
        // index is used directly as physical (no risk of float16 overflow).
        if (enableShapeUniforms) {
            return `
      float ${funcName}(int row, int col) {
        float index = dot(vec3(row, col, ${offset}), vec3(${texName}Shape[1], 1, 1));
        vec2 uv = vec2(0.5, (index + 0.5) / float(${texName}TexShape[0]));
        return sampleTexture(${texName}, uv);
      }
    `;
        }
        return `
    float ${funcName}(int row, int col) {
      float index = dot(vec3(row, col, ${offset}), vec3(${shape[1]}, 1, 1));
      vec2 uv = vec2(0.5, (index + 0.5) / ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
    }
    if (texNumR === 1) {
        // index is used directly as physical (no risk of float16 overflow).
        if (enableShapeUniforms) {
            return `
      float ${funcName}(int row, int col) {
        float index = dot(vec3(row, col, ${offset}), vec3(${texName}Shape[1], 1, 1));
        vec2 uv = vec2((index + 0.5) / float(${texName}TexShape[1]), 0.5);
        return sampleTexture(${texName}, uv);
      }
    `;
        }
        return `
    float ${funcName}(int row, int col) {
      float index = dot(vec3(row, col, ${offset}), vec3(${shape[1]}, 1, 1));
      vec2 uv = vec2((index + 0.5) / ${texNumC}.0, 0.5);
      return sampleTexture(${texName}, uv);
    }
  `;
    }
    if (enableShapeUniforms) {
        return `
      float ${funcName}(int row, int col) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${texName}Shape[1] + col + ${offset};
        vec2 uv = uvFromFlat(${texName}TexShape[0], ${texName}TexShape[1], index);
        return sampleTexture(${texName}, uv);
      }
    `;
    }
    return `
  float ${funcName}(int row, int col) {
    // Explicitly use integer operations as dot() only works on floats.
    int index = row * ${shape[1]} + col + ${offset};
    vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
    return sampleTexture(${texName}, uv);
  }
`;
}
function getPackedSampler3D(inputInfo, enableShapeUniforms) {
    const shape = inputInfo.shapeInfo.logicalShape;
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const texShape = inputInfo.shapeInfo.texShape;
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    if (shape[0] === 1) {
        const squeezedShape = shape.slice(1);
        const keptDims = [1, 2];
        const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
        const params = ['b', 'row', 'col'];
        return `
        ${getPackedSamplerFromInInfo(newInputInfo, enableShapeUniforms)}
        vec4 ${funcName}(int b, int row, int col) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
    }
    const glsl = getGlslDifferences();
    if (enableShapeUniforms) {
        return `
    vec4 ${funcName}(int b, int row, int col) {
      ivec2 packedTexShape = ivec2(ceil(float(${texName}TexShape[0]) / 2.0), ceil(float(${texName}TexShape[1]) / 2.0));
      int valuesPerRow = int(ceil(float(${texName}Shape[2]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${texName}Shape[1]) / 2.0));
      vec2 uv = packedUVfrom3D(
        packedTexShape[0], packedTexShape[1], texelsInBatch, valuesPerRow, b, row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
    }
    const texNumR = packedTexShape[0];
    const texNumC = packedTexShape[1];
    const valuesPerRow = Math.ceil(shape[2] / 2);
    const texelsInBatch = valuesPerRow * Math.ceil(shape[1] / 2);
    return `
    vec4 ${funcName}(int b, int row, int col) {
      vec2 uv = packedUVfrom3D(
        ${texNumR}, ${texNumC}, ${texelsInBatch}, ${valuesPerRow}, b, row, col);
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}
function getSampler3D(inputInfo, enableShapeUniforms) {
    const shape = inputInfo.shapeInfo.logicalShape;
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const stride0 = shape[1] * shape[2];
    const stride1 = shape[2];
    const { newShape, keptDims } = util.squeezeShape(shape);
    const squeezedShape = newShape;
    if (squeezedShape.length < shape.length) {
        const newInputInfo = squeezeInputInfo(inputInfo, squeezedShape);
        const params = ['row', 'col', 'depth'];
        return `
        ${getSamplerFromInInfo(newInputInfo, enableShapeUniforms)}
        float ${funcName}(int row, int col, int depth) {
          return ${funcName}(${getSqueezedParams(params, keptDims)});
        }
      `;
    }
    if (inputInfo.shapeInfo.isUniform) {
        // Uniform arrays will be less than 65505 (no risk of float16 overflow).
        return `
      float ${funcName}(int row, int col, int depth) {
        int index = round(dot(vec3(row, col, depth),
                          vec3(${stride0}, ${stride1}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
    }
    const texShape = inputInfo.shapeInfo.texShape;
    const texNumR = texShape[0];
    const texNumC = texShape[1];
    const flatOffset = inputInfo.shapeInfo.flatOffset;
    if (texNumC === stride0 && flatOffset == null) {
        // texC is used directly as physical (no risk of float16 overflow).
        if (enableShapeUniforms) {
            return `
      float ${funcName}(int row, int col, int depth) {
        int stride1 = ${texName}Shape[2];
        float texR = float(row);
        float texC = dot(vec2(col, depth), vec2(stride1, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texName}TexShape[1], ${texName}TexShape[0]);
        return sampleTexture(${texName}, uv);
      }
    `;
        }
        return `
        float ${funcName}(int row, int col, int depth) {
          float texR = float(row);
          float texC = dot(vec2(col, depth), vec2(${stride1}, 1));
          vec2 uv = (vec2(texC, texR) + halfCR) /
                     vec2(${texNumC}.0, ${texNumR}.0);
          return sampleTexture(${texName}, uv);
        }
      `;
    }
    if (texNumC === stride1 && flatOffset == null) {
        // texR is used directly as physical (no risk of float16 overflow).
        if (enableShapeUniforms) {
            return `
      float ${funcName}(int row, int col, int depth) {
        float texR = dot(vec2(row, col), vec2(${texName}Shape[1], 1));
        float texC = float(depth);
        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texName}TexShape[1], ${texName}TexShape[0]);
        return sampleTexture(${texName}, uv);
      }
    `;
        }
        return `
    float ${funcName}(int row, int col, int depth) {
      float texR = dot(vec2(row, col), vec2(${shape[1]}, 1));
      float texC = float(depth);
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);
      return sampleTexture(${texName}, uv);
    }
  `;
    }
    const offset = getFlatOffsetUniformName(texName);
    if (enableShapeUniforms) {
        return `
    float ${funcName}(int row, int col, int depth) {
      // Explicitly use integer operations as dot() only works on floats.
      int stride0 = ${texName}Shape[1] * ${texName}Shape[2];
      int stride1 = ${texName}Shape[2];
      int index = row * ${stride0} + col * ${stride1} + depth + ${offset};
      vec2 uv = uvFromFlat(${texName}TexShape[0], ${texName}TexShape[1], index);
      return sampleTexture(${texName}, uv);
    }
    `;
    }
    return `
      float ${funcName}(int row, int col, int depth) {
        // Explicitly use integer operations as dot() only works on floats.
        int index = row * ${stride0} + col * ${stride1} + depth + ${offset};
        vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
        return sampleTexture(${texName}, uv);
      }
  `;
}
function getPackedSamplerND(inputInfo, enableShapeUniforms) {
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const glsl = getGlslDifferences();
    if (enableShapeUniforms) {
        // TODO: support 5d and 6d
        return `
    vec4 ${funcName}(int b2, int b, int row, int col) {
      int valuesPerRow = int(ceil(float(${texName}Shape[3]) / 2.0));
      int texelsInBatch = valuesPerRow * int(ceil(float(${texName}Shape[2]) / 2.0));
      int index = b * texelsInBatch + (row / 2) * valuesPerRow + (col / 2);
      texelsInBatch *= ${texName}Shape[1];
      index = b2 * texelsInBatch + index;
      ivec2 packedTexShape = ivec2(ceil(float(${texName}TexShape[0]) / 2.0), ceil(float(${texName}TexShape[1]) / 2.0));
      int texR = index / packedTexShape[1];
      int texC = index - texR * packedTexShape[1];
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(packedTexShape[1], packedTexShape[0]); return ${glsl.texture2D}(${texName}, uv);
    }
  `;
    }
    const shape = inputInfo.shapeInfo.logicalShape;
    const rank = shape.length;
    const texShape = inputInfo.shapeInfo.texShape;
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    const texNumR = packedTexShape[0];
    const texNumC = packedTexShape[1];
    const valuesPerRow = Math.ceil(shape[rank - 1] / 2);
    let texelsInBatch = valuesPerRow * Math.ceil(shape[rank - 2] / 2);
    let params = `int b, int row, int col`;
    let index = `b * ${texelsInBatch} + (row / 2) * ${valuesPerRow} + (col / 2)`;
    for (let b = 2; b < rank - 1; b++) {
        params = `int b${b}, ` + params;
        texelsInBatch *= shape[rank - b - 1];
        index = `b${b} * ${texelsInBatch} + ` + index;
    }
    return `
    vec4 ${funcName}(${params}) {
      int index = ${index};
      int texR = index / ${texNumC};
      int texC = index - texR * ${texNumC};
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}, ${texNumR});
      return ${glsl.texture2D}(${texName}, uv);
    }
  `;
}
function getSampler4D(inputInfo, enableShapeUniforms) {
    const shape = inputInfo.shapeInfo.logicalShape;
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const stride2 = shape[3];
    const stride1 = shape[2] * stride2;
    const stride0 = shape[1] * stride1;
    const { newShape, keptDims } = util.squeezeShape(shape);
    if (newShape.length < shape.length) {
        const newInputInfo = squeezeInputInfo(inputInfo, newShape);
        const params = ['row', 'col', 'depth', 'depth2'];
        return `
      ${getSamplerFromInInfo(newInputInfo, enableShapeUniforms)}
      float ${funcName}(int row, int col, int depth, int depth2) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
    }
    if (inputInfo.shapeInfo.isUniform) {
        // Uniform arrays will be less than 65505 (no risk of float16 overflow).
        return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        int index = round(dot(vec4(row, col, depth, depth2),
                          vec4(${stride0}, ${stride1}, ${stride2}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
    }
    const flatOffset = inputInfo.shapeInfo.flatOffset;
    const texShape = inputInfo.shapeInfo.texShape;
    const texNumR = texShape[0];
    const texNumC = texShape[1];
    const stride2Str = `int stride2 = ${texName}Shape[3];`;
    const stride1Str = `int stride1 = ${texName}Shape[2] * stride2;`;
    const stride0Str = `int stride0 = ${texName}Shape[1] * stride1;`;
    if (texNumC === stride0 && flatOffset == null) {
        // texC is used directly as physical (no risk of float16 overflow).
        if (enableShapeUniforms) {
            return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        ${stride2Str}
        ${stride1Str}
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(stride1, stride2, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texName}TexShape[1], ${texName}TexShape[0]);
        return sampleTexture(${texName}, uv);
      }
    `;
        }
        return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float texR = float(row);
        float texC =
            dot(vec3(col, depth, depth2),
                vec3(${stride1}, ${stride2}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
    }
    if (texNumC === stride2 && flatOffset == null) {
        // texR is used directly as physical (no risk of float16 overflow).
        if (enableShapeUniforms) {
            return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${texName}Shape[1] * ${texName}Shape[2], ${texName}Shape[2], 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texName}TexShape[1], ${texName}TexShape[0]);
        return sampleTexture(${texName}, uv);
      }
    `;
        }
        return `
      float ${funcName}(int row, int col, int depth, int depth2) {
        float texR = dot(vec3(row, col, depth),
                         vec3(${shape[1] * shape[2]}, ${shape[2]}, 1));
        float texC = float(depth2);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
    }
    const offset = getFlatOffsetUniformName(texName);
    if (enableShapeUniforms) {
        return `
    float ${funcName}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      ${stride2Str}
      ${stride1Str}
      ${stride0Str}
      int index = row * stride0 + col * stride1 +
          depth * stride2 + depth2;
      vec2 uv = uvFromFlat(${texName}TexShape[0], ${texName}TexShape[1], index + ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
    }
    return `
    float ${funcName}(int row, int col, int depth, int depth2) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} +
          depth * ${stride2} + depth2;
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index + ${offset});
      return sampleTexture(${texName}, uv);
    }
  `;
}
function getSampler5D(inputInfo) {
    const shape = inputInfo.shapeInfo.logicalShape;
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const stride3 = shape[4];
    const stride2 = shape[3] * stride3;
    const stride1 = shape[2] * stride2;
    const stride0 = shape[1] * stride1;
    const { newShape, keptDims } = util.squeezeShape(shape);
    if (newShape.length < shape.length) {
        const newInputInfo = squeezeInputInfo(inputInfo, newShape);
        const params = ['row', 'col', 'depth', 'depth2', 'depth3'];
        return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
    }
    if (inputInfo.shapeInfo.isUniform) {
        // Uniform arrays will be less than 65505 (no risk of float16 overflow).
        return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        float index = dot(
          vec4(row, col, depth, depth2),
          vec4(${stride0}, ${stride1}, ${stride2}, ${stride3})) +
          depth3;
        ${getUniformSampler(inputInfo)}
      }
    `;
    }
    const flatOffset = inputInfo.shapeInfo.flatOffset;
    const texShape = inputInfo.shapeInfo.texShape;
    const texNumR = texShape[0];
    const texNumC = texShape[1];
    if (texNumC === stride0 && flatOffset == null) {
        // texC is used directly as physical (no risk of float16 overflow).
        return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
                         vec4(${stride1}, ${stride2}, ${stride3}, 1));
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
    }
    if (texNumC === stride3 && flatOffset == null) {
        // texR is used directly as physical (no risk of float16 overflow).
        return `
      float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
        float texR = dot(
          vec4(row, col, depth, depth2),
          vec4(${shape[1] * shape[2] * shape[3]},
               ${shape[2] * shape[3]}, ${shape[3]}, 1));
        int texC = depth3;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
    }
    const offset = getFlatOffsetUniformName(texName);
    return `
    float ${funcName}(int row, int col, int depth, int depth2, int depth3) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} + depth * ${stride2} +
          depth2 * ${stride3} + depth3 + ${offset};
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
      return sampleTexture(${texName}, uv);
    }
  `;
}
function getSampler6D(inputInfo) {
    const shape = inputInfo.shapeInfo.logicalShape;
    const texName = inputInfo.name;
    const funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    const { newShape, keptDims } = util.squeezeShape(shape);
    if (newShape.length < shape.length) {
        const newInputInfo = squeezeInputInfo(inputInfo, newShape);
        const params = ['row', 'col', 'depth', 'depth2', 'depth3', 'depth4'];
        return `
      ${getSamplerFromInInfo(newInputInfo)}
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        return ${funcName}(${getSqueezedParams(params, keptDims)});
      }
    `;
    }
    const stride4 = shape[5];
    const stride3 = shape[4] * stride4;
    const stride2 = shape[3] * stride3;
    const stride1 = shape[2] * stride2;
    const stride0 = shape[1] * stride1;
    if (inputInfo.shapeInfo.isUniform) {
        // Uniform arrays will be less than 65505 (no risk of float16 overflow).
        return `
      float ${funcName}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
        int index = round(dot(
          vec4(row, col, depth, depth2),
          vec4(${stride0}, ${stride1}, ${stride2}, ${stride3})) +
          dot(
            vec2(depth3, depth4),
            vec2(${stride4}, 1)));
        ${getUniformSampler(inputInfo)}
      }
    `;
    }
    const flatOffset = inputInfo.shapeInfo.flatOffset;
    const texShape = inputInfo.shapeInfo.texShape;
    const texNumR = texShape[0];
    const texNumC = texShape[1];
    if (texNumC === stride0 && flatOffset == null) {
        // texC is used directly as physical (no risk of float16 overflow).
        return `
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        int texR = row;
        float texC = dot(vec4(col, depth, depth2, depth3),
          vec4(${stride1}, ${stride2}, ${stride3}, ${stride4})) +
               float(depth4);
        vec2 uv = (vec2(texC, texR) + halfCR) /
                   vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
    }
    if (texNumC === stride4 && flatOffset == null) {
        // texR is used directly as physical (no risk of float16 overflow).
        return `
      float ${funcName}(int row, int col, int depth,
                    int depth2, int depth3, int depth4) {
        float texR = dot(vec4(row, col, depth, depth2),
          vec4(${shape[1] * shape[2] * shape[3] * shape[4]},
               ${shape[2] * shape[3] * shape[4]},
               ${shape[3] * shape[4]},
               ${shape[4]})) + float(depth3);
        int texC = depth4;
        vec2 uv = (vec2(texC, texR) + halfCR) /
                  vec2(${texNumC}.0, ${texNumR}.0);
        return sampleTexture(${texName}, uv);
      }
    `;
    }
    const offset = getFlatOffsetUniformName(texName);
    return `
    float ${funcName}(int row, int col, int depth,
                  int depth2, int depth3, int depth4) {
      // Explicitly use integer operations as dot() only works on floats.
      int index = row * ${stride0} + col * ${stride1} + depth * ${stride2} +
          depth2 * ${stride3} + depth3 * ${stride4} + depth4 + ${offset};
      vec2 uv = uvFromFlat(${texNumR}, ${texNumC}, index);
      return sampleTexture(${texName}, uv);
    }
  `;
}
function getUniformSampler(inputInfo) {
    const texName = inputInfo.name;
    const inSize = util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
    if (inSize < 2) {
        return `return ${texName};`;
    }
    return `
    for (int i = 0; i < ${inSize}; i++) {
      if (i == index) {
        return ${texName}[i];
      }
    }
  `;
}
function getPackedSamplerAtOutputCoords(inputInfo, outShapeInfo) {
    const texName = inputInfo.name;
    const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
    const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
    const inRank = inputInfo.shapeInfo.logicalShape.length;
    const outRank = outShapeInfo.logicalShape.length;
    const broadcastDims = getBroadcastDims(inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
    const type = getCoordsDataType(outRank);
    const rankDiff = outRank - inRank;
    let coordsSnippet;
    const fields = ['x', 'y', 'z', 'w', 'u', 'v'];
    if (inRank === 0) {
        coordsSnippet = '';
    }
    else if (outRank < 2 && broadcastDims.length >= 1) {
        coordsSnippet = 'coords = 0;';
    }
    else {
        coordsSnippet =
            broadcastDims.map(d => `coords.${fields[d + rankDiff]} = 0;`)
                .join('\n');
    }
    let unpackedCoordsSnippet = '';
    if (outRank < 2 && inRank > 0) {
        unpackedCoordsSnippet = 'coords';
    }
    else {
        unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
            .map((s, i) => `coords.${fields[i + rankDiff]}`)
            .join(', ');
    }
    let output = `return outputValue;`;
    const inSize = util.sizeFromShape(inputInfo.shapeInfo.logicalShape);
    const isInputScalar = inSize === 1;
    const outSize = util.sizeFromShape(outShapeInfo.logicalShape);
    const isOutputScalar = outSize === 1;
    if (inRank === 1 && !isInputScalar && !isOutputScalar) {
        output = `
      return vec4(outputValue.xy, outputValue.xy);
    `;
    }
    else if (isInputScalar && !isOutputScalar) {
        if (outRank === 1) {
            output = `
        return vec4(outputValue.x, outputValue.x, 0., 0.);
      `;
        }
        else {
            output = `
        return vec4(outputValue.x);
      `;
        }
    }
    else if (broadcastDims.length) {
        const rows = inRank - 2;
        const cols = inRank - 1;
        if (broadcastDims.indexOf(rows) > -1 && broadcastDims.indexOf(cols) > -1) {
            output = `return vec4(outputValue.x);`;
        }
        else if (broadcastDims.indexOf(rows) > -1) {
            output = `return vec4(outputValue.x, outputValue.y, ` +
                `outputValue.x, outputValue.y);`;
        }
        else if (broadcastDims.indexOf(cols) > -1) {
            output = `return vec4(outputValue.xx, outputValue.zz);`;
        }
    }
    return `
    vec4 ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      vec4 outputValue = get${texFuncSnippet}(${unpackedCoordsSnippet});
      ${output}
    }
  `;
}
function getSamplerAtOutputCoords(inputInfo, outShapeInfo) {
    const texName = inputInfo.name;
    const texFuncSnippet = texName.charAt(0).toUpperCase() + texName.slice(1);
    const funcName = 'get' + texFuncSnippet + 'AtOutCoords';
    const outTexShape = outShapeInfo.texShape;
    const inTexShape = inputInfo.shapeInfo.texShape;
    const inRank = inputInfo.shapeInfo.logicalShape.length;
    const outRank = outShapeInfo.logicalShape.length;
    if (!inputInfo.shapeInfo.isUniform && inRank === outRank &&
        inputInfo.shapeInfo.flatOffset == null &&
        util.arraysEqual(inTexShape, outTexShape)) {
        return `
      float ${funcName}() {
        return sampleTexture(${texName}, resultUV);
      }
    `;
    }
    const type = getCoordsDataType(outRank);
    const broadcastDims = getBroadcastDims(inputInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape);
    const rankDiff = outRank - inRank;
    let coordsSnippet;
    const fields = ['x', 'y', 'z', 'w', 'u', 'v'];
    if (inRank === 0) {
        coordsSnippet = '';
    }
    else if (outRank < 2 && broadcastDims.length >= 1) {
        coordsSnippet = 'coords = 0;';
    }
    else {
        coordsSnippet =
            broadcastDims.map(d => `coords.${fields[d + rankDiff]} = 0;`)
                .join('\n');
    }
    let unpackedCoordsSnippet = '';
    if (outRank < 2 && inRank > 0) {
        unpackedCoordsSnippet = 'coords';
    }
    else {
        unpackedCoordsSnippet = inputInfo.shapeInfo.logicalShape
            .map((s, i) => `coords.${fields[i + rankDiff]}`)
            .join(', ');
    }
    return `
    float ${funcName}() {
      ${type} coords = getOutputCoords();
      ${coordsSnippet}
      return get${texFuncSnippet}(${unpackedCoordsSnippet});
    }
  `;
}
export function getCoordsDataType(rank) {
    if (rank <= 1) {
        return 'int';
    }
    else if (rank === 2) {
        return 'ivec2';
    }
    else if (rank === 3) {
        return 'ivec3';
    }
    else if (rank === 4) {
        return 'ivec4';
    }
    else if (rank === 5) {
        return 'ivec5';
    }
    else if (rank === 6) {
        return 'ivec6';
    }
    else {
        throw Error(`GPU for rank ${rank} is not yet supported`);
    }
}
export function getUniformInfoFromShape(isPacked, shape, texShape) {
    const { newShape, keptDims } = util.squeezeShape(shape);
    const rank = shape.length;
    const useSqueezePackedShape = isPacked && rank === 3 && shape[0] === 1;
    const squeezeShape = useSqueezePackedShape ? shape.slice(1) : newShape;
    const useSqueezeShape = (!isPacked && rank > 1 && !util.arraysEqual(shape, texShape) &&
        newShape.length < rank) ||
        useSqueezePackedShape;
    const uniformShape = useSqueezeShape ? squeezeShape : shape;
    return { useSqueezeShape, uniformShape, keptDims };
}
/** Returns a new input info (a copy) that has a squeezed logical shape. */
export function squeezeInputInfo(inInfo, squeezedShape) {
    // Deep copy.
    const newInputInfo = JSON.parse(JSON.stringify(inInfo));
    newInputInfo.shapeInfo.logicalShape = squeezedShape;
    return newInputInfo;
}
function getSqueezedParams(params, keptDims) {
    return keptDims.map(d => params[d]).join(', ');
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2hhZGVyX2NvbXBpbGVyLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9zaGFkZXJfY29tcGlsZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsNEVBQTRFO0FBQzVFLDREQUE0RDtBQUU1RCxPQUFPLEVBQUMsWUFBWSxFQUFFLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ3pELE1BQU0sRUFBQyxnQkFBZ0IsRUFBQyxHQUFHLFlBQVksQ0FBQztBQUN4QyxPQUFPLEVBQUMsa0JBQWtCLEVBQU8sTUFBTSxnQkFBZ0IsQ0FBQztBQUN4RCxPQUFPLEtBQUssV0FBVyxNQUFNLHdCQUF3QixDQUFDO0FBMEJ0RCxNQUFNLFVBQVUsVUFBVSxDQUN0QixVQUF1QixFQUFFLFdBQXNCLEVBQy9DLE9BQXNCO0lBQ3hCLE1BQU0sY0FBYyxHQUFhLEVBQUUsQ0FBQztJQUNwQyxVQUFVLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFO1FBQ3JCLE1BQU0sSUFBSSxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLFNBQVMsQ0FBQyxZQUFZLENBQUMsQ0FBQztRQUUxRCwyREFBMkQ7UUFDM0QsSUFBSSxDQUFDLENBQUMsU0FBUyxDQUFDLFNBQVMsRUFBRTtZQUN6QixjQUFjLENBQUMsSUFBSSxDQUNmLGlCQUFpQixDQUFDLENBQUMsSUFBSSxHQUFHLElBQUksR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksSUFBSSxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUM7U0FDL0Q7YUFBTTtZQUNMLGNBQWMsQ0FBQyxJQUFJLENBQUMscUJBQXFCLENBQUMsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO1lBQ3BELGNBQWMsQ0FBQyxJQUFJLENBQUMscUJBQXFCLENBQUMsQ0FBQyxJQUFJLEdBQUcsQ0FB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SSxHQUFHLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsS0FBSyxFQUFFLFFBQVEsQ0FBQztRQUMzRCxRQUFRLENBQUMsTUFBTSxHQUFHLElBQUksQ0FBQztRQUN4QixxQkFBcUIsQ0FBQztJQUMxQixNQUFNLFlBQVksR0FBRyxlQUFlLENBQUMsQ0FBQyxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDO0lBQzVELE9BQU8sRUFBQyxlQUFlLEVBQUUsWUFBWSxFQUFFLFFBQVEsRUFBQyxDQUFDO0FBQ25ELENBQUM7QUFFRCwyRUFBMkU7QUFDM0UsTUFBTSxVQUFVLGdCQUFnQixDQUM1QixNQUFpQixFQUFFLGFBQXVCO0lBQzVDLGFBQWE7SUFDYixNQUFNLFlBQVksR0FBYyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQztJQUNuRSxZQUFZLENBQUMsU0FBUyxDQUFDLFlBQVksR0FBRyxhQUFhLENBQUM7SUFDcEQsT0FBTyxZQUFZLENBQUM7QUFDdEIsQ0FBQztBQUVELFNBQVMsaUJBQWlCLENBQUMsTUFBZ0IsRUFBRSxRQUFrQjtJQUM3RCxPQUFPLFFBQVEsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7QUFDakQsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE3IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLy8gUGxlYXNlIG1ha2Ugc3VyZSB0aGUgc2hha2VyIGtleSBpbiBtYWtlU2hhZGVyS2V5IGluIGdwZ3B1X21hdGgudHMgaXMgd2VsbFxuLy8gbWFwcGVkIGlmIGFueSBzaGFkZXIgc291cmNlIGNvZGUgaXMgY2hhbmdlZCBpbiB0aGlzIGZpbGUuXG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuY29uc3Qge2dldEJyb2FkY2FzdERpbXN9ID0gYmFja2VuZF91dGlsO1xuaW1wb3J0IHtnZXRHbHNsRGlmZmVyZW5jZXMsIEdMU0x9IGZyb20gJy4vZ2xzbF92ZXJzaW9uJztcbmltcG9ydCAqIGFzIHNoYWRlcl91dGlsIGZyb20gJy4vc2hhZGVyX2NvbXBpbGVyX3V0aWwnO1xuXG5leHBvcnQgdHlwZSBTaGFwZUluZm8gPSB7XG4gIGxvZ2ljYWxTaGFwZTogbnVtYmVyW10sXG4gIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLFxuICBpc1VuaWZvcm06IGJvb2xlYW4sXG4gIGlzUGFja2VkOiBib29sZWFuLFxuICBmbGF0T2Zmc2V0OiBudW1iZXJcbn07XG5cbmV4cG9ydCB0eXBlIElucHV0SW5mbyA9IHtcbiAgbmFtZTogc3RyaW5nLFxuICBzaGFwZUluZm86IFNoYXBlSW5mb1xufTtcblxuZXhwb3J0IHR5cGUgVW5pZm9ybVR5cGUgPVxuICAgICdmbG9hdCd8J3ZlYzInfCd2ZWMzJ3wndmVjNCd8J2ludCd8J2l2ZWMyJ3wnaXZlYzMnfCdpdmVjNCc7XG5cbmludGVyZmFjZSBQcm9ncmFtUGFyYW1zIHtcbiAgdXNlckNvZGU6IHN0cmluZztcbiAgZW5hYmxlU2hhcGVVbmlmb3Jtcz86IGJvb2xlYW47XG4gIHBhY2tlZElucHV0cz86IGJvb2xlYW47XG4gIGN1c3RvbVVuaWZvcm1zPzpcbiAgICAgIEFycmF5PHtuYW1lOiBzdHJpbmc7IGFycmF5SW5kZXg/OiBudW1iZXI7IHR5cGU6IFVuaWZvcm1UeXBlO30+O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbWFrZVNoYWRlcihcbiAgICBpbnB1dHNJbmZvOiBJbnB1dEluZm9bXSwgb3V0cHV0U2hhcGU6IFNoYXBlSW5mbyxcbiAgICBwcm9ncmFtOiBQcm9ncmFtUGFyYW1zKTogc3RyaW5nIHtcbiAgY29uc3QgcHJlZml4U25pcHBldHM6IHN0cmluZ1tdID0gW107XG4gIGlucHV0c0luZm8uZm9yRWFjaCh4ID0+IHtcbiAgICBjb25zdCBzaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHguc2hhcGVJbmZvLmxvZ2ljYWxTaGFwZSk7XG5cbiAgICAvLyBTbmlwcGV0IHdoZW4gd2UgZGVjaWRlZCB0byB1cGxvYWQgdGhlIHZhbHVlcyBhcyB1bmlmb3JtLlxuICAgIGlmICh4LnNoYXBlSW5mby5pc1VuaWZvcm0pIHtcbiAgICAgIHByZWZpeFNuaXBwZXRzLnB1c2goXG4gICAgICAgICAgYHVuaWZvcm0gZmxvYXQgJHt4Lm5hbWV9JHtzaXplID4gMSA/IGBbJHtzaXplfV1gIDogJyd9O2ApO1xuICAgIH0gZWxzZSB7XG4gICAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGB1bmlmb3JtIHNhbXBsZXIyRCAke3gubmFtZX07YCk7XG4gICAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGB1bmlmb3JtIGludCBvZmZzZXQke3gubmFtZX07YCk7XG4gICAgfVxuXG4gICAgaWYgKHByb2dyYW0uZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgICAgY29uc3Qge3VuaWZvcm1TaGFwZX0gPSBnZXRVbmlmb3JtSW5mb0Zyb21TaGFwZShcbiAgICAgICAgICBwcm9ncmFtLnBhY2tlZElucHV0cywgeC5zaGFwZUluZm8ubG9naWNhbFNoYXBlLCB4LnNoYXBlSW5mby50ZXhTaGFwZSk7XG4gICAgICBzd2l0Y2ggKHVuaWZvcm1TaGFwZS5sZW5ndGgpIHtcbiAgICAgICAgY2FzZSAxOlxuICAgICAgICAgIHByZWZpeFNuaXBwZXRzLnB1c2goYHVuaWZvcm0gaW50ICR7eC5uYW1lfVNoYXBlO2ApO1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICBjYXNlIDI6XG4gICAgICAgICAgcHJlZml4U25pcHBldHMucHVzaChgdW5pZm9ybSBpdmVjMiAke3gubmFtZX1TaGFwZTtgKTtcbiAgICAgICAgICBicmVhaztcbiAgICAgICAgY2FzZSAzOlxuICAgICAgICAgIHByZWZpeFNuaXBwZXRzLnB1c2goYHVuaWZvcm0gaXZlYzMgJHt4Lm5hbWV9U2hhcGU7YCk7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIGNhc2UgNDpcbiAgICAgICAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGB1bmlmb3JtIGl2ZWM0ICR7eC5uYW1lfVNoYXBlO2ApO1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICBkZWZhdWx0OlxuICAgICAgICAgIGJyZWFrO1xuICAgICAgfVxuICAgICAgcHJlZml4U25pcHBldHMucHVzaChgdW5pZm9ybSBpdmVjMiAke3gubmFtZX1UZXhTaGFwZTtgKTtcbiAgICB9XG4gIH0pO1xuXG4gIGlmIChwcm9ncmFtLmVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICBzd2l0Y2ggKG91dHB1dFNoYXBlLmxvZ2ljYWxTaGFwZS5sZW5ndGgpIHtcbiAgICAgIGNhc2UgMTpcbiAgICAgICAgcHJlZml4U25pcHBldHMucHVzaChgdW5pZm9ybSBpbnQgb3V0U2hhcGU7YCk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgY2FzZSAyOlxuICAgICAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGB1bmlmb3JtIGl2ZWMyIG91dFNoYXBlO2ApO1xuICAgICAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGB1bmlmb3JtIGludCBvdXRTaGFwZVN0cmlkZXM7YCk7XG4gICAgICAgIGJyZWFrO1xuICAgICAgY2FzZSAzOlxuICAgICAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGB1bmlmb3JtIGl2ZWMzIG91dFNoYXBlO2ApO1xuICAgICAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGB1bmlmb3JtIGl2ZWMyIG91dFNoYXBlU3RyaWRlcztgKTtcbiAgICAgICAgYnJlYWs7XG4gICAgICBjYXNlIDQ6XG4gICAgICAgIHByZWZpeFNuaXBwZXRzLnB1c2goYHVuaWZvcm0gaXZlYzQgb3V0U2hhcGU7YCk7XG4gICAgICAgIHByZWZpeFNuaXBwZXRzLnB1c2goYHVuaWZvcm0gaXZlYzMgb3V0U2hhcGVTdHJpZGVzO2ApO1xuICAgICAgICBicmVhaztcbiAgICAgIGRlZmF1bHQ6XG4gICAgICAgIGJyZWFrO1xuICAgIH1cbiAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGB1bmlmb3JtIGl2ZWMyIG91dFRleFNoYXBlO2ApO1xuICB9XG4gIGlmIChwcm9ncmFtLmN1c3RvbVVuaWZvcm1zKSB7XG4gICAgcHJvZ3JhbS5jdXN0b21Vbmlmb3Jtcy5mb3JFYWNoKChkKSA9PiB7XG4gICAgICBwcmVmaXhTbmlwcGV0cy5wdXNoKGB1bmlmb3JtICR7ZC50eXBlfSAke2QubmFtZX0ke1xuICAgICAgICAgIGQuYXJyYXlJbmRleCA/IGBbJHtkLmFycmF5SW5kZXh9XWAgOiAnJ307YCk7XG4gICAgfSk7XG4gIH1cbiAgY29uc3QgaW5wdXRQcmVmaXhTbmlwcGV0ID0gcHJlZml4U25pcHBldHMuam9pbignXFxuJyk7XG5cbiAgY29uc3QgaW5wdXRTYW1wbGluZ1NuaXBwZXQgPSBpbnB1dHNJbmZvXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC5tYXAoXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB4ID0+IGdldElucHV0U2FtcGxpbmdTbmlwcGV0KFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHgsIG91dHB1dFNoYXBlLCBwcm9ncmFtLnBhY2tlZElucHV0cyxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwcm9ncmFtLmVuYWJsZVNoYXBlVW5pZm9ybXMpKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAuam9pbignXFxuJyk7XG4gIGNvbnN0IG91dFRleFNoYXBlID0gb3V0cHV0U2hhcGUudGV4U2hhcGU7XG4gIGNvbnN0IGdsc2wgPSBnZXRHbHNsRGlmZmVyZW5jZXMoKTtcbiAgY29uc3QgZmxvYXRUZXh0dXJlU2FtcGxlU25pcHBldCA9IGdldEZsb2F0VGV4dHVyZVNhbXBsZVNuaXBwZXQoZ2xzbCk7XG4gIGxldCBvdXRwdXRTYW1wbGluZ1NuaXBwZXQ6IHN0cmluZztcbiAgbGV0IGZsb2F0VGV4dHVyZVNldE91dHB1dFNuaXBwZXQ6IHN0cmluZztcbiAgbGV0IHNoYWRlclByZWZpeCA9IGdldFNoYWRlclByZWZpeChnbHNsKTtcblxuICBpZiAob3V0cHV0U2hhcGUuaXNQYWNrZWQpIHtcbiAgICBvdXRwdXRTYW1wbGluZ1NuaXBwZXQgPSBnZXRQYWNrZWRPdXRwdXRTYW1wbGluZ1NuaXBwZXQoXG4gICAgICAgIG91dHB1dFNoYXBlLmxvZ2ljYWxTaGFwZSwgb3V0VGV4U2hhcGUsIHByb2dyYW0uZW5hYmxlU2hhcGVVbmlmb3Jtcyk7XG4gICAgZmxvYXRUZXh0dXJlU2V0T3V0cHV0U25pcHBldCA9IGdldEZsb2F0VGV4dHVyZVNldFJHQkFTbmlwcGV0KGdsc2wpO1xuICB9IGVsc2Uge1xuICAgIG91dHB1dFNhbXBsaW5nU25pcHBldCA9IGdldE91dHB1dFNhbXBsaW5nU25pcHBldChcbiAgICAgICAgb3V0cHV0U2hhcGUubG9naWNhbFNoYXBlLCBvdXRUZXhTaGFwZSwgcHJvZ3JhbS5lbmFibGVTaGFwZVVuaWZvcm1zKTtcbiAgICBmbG9hdFRleHR1cmVTZXRPdXRwdXRTbmlwcGV0ID0gZ2V0RmxvYXRUZXh0dXJlU2V0UlNuaXBwZXQoZ2xzbCk7XG4gIH1cblxuICBpZiAocHJvZ3JhbS5wYWNrZWRJbnB1dHMpIHtcbiAgICBzaGFkZXJQcmVmaXggKz0gU0hBREVSX1BBQ0tFRF9QUkVGSVg7XG4gIH1cblxuICBjb25zdCBzb3VyY2UgPSBbXG4gICAgc2hhZGVyUHJlZml4LCBmbG9hdFRleHR1cmVTYW1wbGVTbmlwcGV0LCBmbG9hdFRleHR1cmVTZXRPdXRwdXRTbmlwcGV0LFxuICAgIGlucHV0UHJlZml4U25pcHBldCwgb3V0cHV0U2FtcGxpbmdTbmlwcGV0LCBpbnB1dFNhbXBsaW5nU25pcHBldCxcbiAgICBwcm9ncmFtLnVzZXJDb2RlXG4gIF0uam9pbignXFxuJyk7XG4gIHJldHVybiBzb3VyY2U7XG59XG5cbmZ1bmN0aW9uIGdldFNhbXBsZXJGcm9tSW5JbmZvKFxuICAgIGluSW5mbzogSW5wdXRJbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zID0gZmFsc2UpOiBzdHJpbmcge1xuICBjb25zdCBzaGFwZSA9IGluSW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlO1xuICBzd2l0Y2ggKHNoYXBlLmxlbmd0aCkge1xuICAgIGNhc2UgMDpcbiAgICAgIHJldHVybiBnZXRTYW1wbGVyU2NhbGFyKGluSW5mbywgZW5hYmxlU2hhcGVVbmlmb3Jtcyk7XG4gICAgY2FzZSAxOlxuICAgICAgcmV0dXJuIGdldFNhbXBsZXIxRChpbkluZm8sIGVuYWJsZVNoYXBlVW5pZm9ybXMpO1xuICAgIGNhc2UgMjpcbiAgICAgIHJldHVybiBnZXRTYW1wbGVyMkQoaW5JbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zKTtcbiAgICBjYXNlIDM6XG4gICAgICByZXR1cm4gZ2V0U2FtcGxlcjNEKGluSW5mbywgZW5hYmxlU2hhcGVVbmlmb3Jtcyk7XG4gICAgY2FzZSA0OlxuICAgICAgcmV0dXJuIGdldFNhbXBsZXI0RChpbkluZm8sIGVuYWJsZVNoYXBlVW5pZm9ybXMpO1xuICAgIGNhc2UgNTpcbiAgICAgIHJldHVybiBnZXRTYW1wbGVyNUQoaW5JbmZvKTtcbiAgICBjYXNlIDY6XG4gICAgICByZXR1cm4gZ2V0U2FtcGxlcjZEKGluSW5mbyk7XG4gICAgZGVmYXVsdDpcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgJHtzaGFwZS5sZW5ndGh9LUQgaW5wdXQgc2FtcGxpbmdgICtcbiAgICAgICAgICBgIGlzIG5vdCB5ZXQgc3VwcG9ydGVkYCk7XG4gIH1cbn1cblxuZnVuY3Rpb24gZ2V0UGFja2VkU2FtcGxlckZyb21JbkluZm8oXG4gICAgaW5JbmZvOiBJbnB1dEluZm8sIGVuYWJsZVNoYXBlVW5pZm9ybXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBjb25zdCBzaGFwZSA9IGluSW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlO1xuICBzd2l0Y2ggKHNoYXBlLmxlbmd0aCkge1xuICAgIGNhc2UgMDpcbiAgICAgIHJldHVybiBnZXRQYWNrZWRTYW1wbGVyU2NhbGFyKGluSW5mbyk7XG4gICAgY2FzZSAxOlxuICAgICAgcmV0dXJuIGdldFBhY2tlZFNhbXBsZXIxRChpbkluZm8sIGVuYWJsZVNoYXBlVW5pZm9ybXMpO1xuICAgIGNhc2UgMjpcbiAgICAgIHJldHVybiBnZXRQYWNrZWRTYW1wbGVyMkQoaW5JbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zKTtcbiAgICBjYXNlIDM6XG4gICAgICByZXR1cm4gZ2V0UGFja2VkU2FtcGxlcjNEKGluSW5mbywgZW5hYmxlU2hhcGVVbmlmb3Jtcyk7XG4gICAgZGVmYXVsdDpcbiAgICAgIHJldHVybiBnZXRQYWNrZWRTYW1wbGVyTkQoaW5JbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zKTtcbiAgfVxufVxuXG5mdW5jdGlvbiBnZXRJbnB1dFNhbXBsaW5nU25pcHBldChcbiAgICBpbkluZm86IElucHV0SW5mbywgb3V0U2hhcGVJbmZvOiBTaGFwZUluZm8sIHVzZXNQYWNrZWRUZXh0dXJlcyA9IGZhbHNlLFxuICAgIGVuYWJsZVNoYXBlVW5pZm9ybXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBsZXQgcmVzID0gJyc7XG4gIGlmICh1c2VzUGFja2VkVGV4dHVyZXMpIHtcbiAgICByZXMgKz0gZ2V0UGFja2VkU2FtcGxlckZyb21JbkluZm8oaW5JbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zKTtcbiAgfSBlbHNlIHtcbiAgICByZXMgKz0gZ2V0U2FtcGxlckZyb21JbkluZm8oaW5JbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zKTtcbiAgfVxuXG4gIGNvbnN0IGluU2hhcGUgPSBpbkluZm8uc2hhcGVJbmZvLmxvZ2ljYWxTaGFwZTtcbiAgY29uc3Qgb3V0U2hhcGUgPSBvdXRTaGFwZUluZm8ubG9naWNhbFNoYXBlO1xuICBpZiAoaW5TaGFwZS5sZW5ndGggPD0gb3V0U2hhcGUubGVuZ3RoKSB7XG4gICAgaWYgKHVzZXNQYWNrZWRUZXh0dXJlcykge1xuICAgICAgcmVzICs9IGdldFBhY2tlZFNhbXBsZXJBdE91dHB1dENvb3JkcyhpbkluZm8sIG91dFNoYXBlSW5mbyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJlcyArPSBnZXRTYW1wbGVyQXRPdXRwdXRDb29yZHMoaW5JbmZvLCBvdXRTaGFwZUluZm8pO1xuICAgIH1cbiAgfVxuICByZXR1cm4gcmVzO1xufVxuXG5mdW5jdGlvbiBnZXRQYWNrZWRPdXRwdXRTYW1wbGluZ1NuaXBwZXQoXG4gICAgb3V0U2hhcGU6IG51bWJlcltdLCBvdXRUZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSxcbiAgICBlbmFibGVTaGFwZVVuaWZvcm1zOiBib29sZWFuKTogc3RyaW5nIHtcbiAgc3dpdGNoIChvdXRTaGFwZS5sZW5ndGgpIHtcbiAgICBjYXNlIDA6XG4gICAgICByZXR1cm4gZ2V0T3V0cHV0U2NhbGFyQ29vcmRzKCk7XG4gICAgY2FzZSAxOlxuICAgICAgcmV0dXJuIGdldE91dHB1dFBhY2tlZDFEQ29vcmRzKFxuICAgICAgICAgIG91dFNoYXBlIGFzIFtudW1iZXJdLCBvdXRUZXhTaGFwZSwgZW5hYmxlU2hhcGVVbmlmb3Jtcyk7XG4gICAgY2FzZSAyOlxuICAgICAgcmV0dXJuIGdldE91dHB1dFBhY2tlZDJEQ29vcmRzKFxuICAgICAgICAgIG91dFNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl0sIG91dFRleFNoYXBlLCBlbmFibGVTaGFwZVVuaWZvcm1zKTtcbiAgICBjYXNlIDM6XG4gICAgICByZXR1cm4gZ2V0T3V0cHV0UGFja2VkM0RDb29yZHMoXG4gICAgICAgICAgb3V0U2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBvdXRUZXhTaGFwZSxcbiAgICAgICAgICBlbmFibGVTaGFwZVVuaWZvcm1zKTtcbiAgICBkZWZhdWx0OlxuICAgICAgcmV0dXJuIGdldE91dHB1dFBhY2tlZE5EQ29vcmRzKFxuICAgICAgICAgIG91dFNoYXBlLCBvdXRUZXhTaGFwZSwgZW5hYmxlU2hhcGVVbmlmb3Jtcyk7XG4gIH1cbn1cblxuZnVuY3Rpb24gZ2V0T3V0cHV0U2FtcGxpbmdTbmlwcGV0KFxuICAgIG91dFNoYXBlOiBudW1iZXJbXSwgb3V0VGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgZW5hYmxlU2hhcGVVbmlmb3JtczogYm9vbGVhbik6IHN0cmluZyB7XG4gIHN3aXRjaCAob3V0U2hhcGUubGVuZ3RoKSB7XG4gICAgY2FzZSAwOlxuICAgICAgcmV0dXJuIGdldE91dHB1dFNjYWxhckNvb3JkcygpO1xuICAgIGNhc2UgMTpcbiAgICAgIHJldHVybiBnZXRPdXRwdXQxRENvb3JkcyhcbiAgICAgICAgICBvdXRTaGFwZSBhcyBbbnVtYmVyXSwgb3V0VGV4U2hhcGUsIGVuYWJsZVNoYXBlVW5pZm9ybXMpO1xuICAgIGNhc2UgMjpcbiAgICAgIHJldHVybiBnZXRPdXRwdXQyRENvb3JkcyhcbiAgICAgICAgICBvdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXJdLCBvdXRUZXhTaGFwZSwgZW5hYmxlU2hhcGVVbmlmb3Jtcyk7XG4gICAgY2FzZSAzOlxuICAgICAgcmV0dXJuIGdldE91dHB1dDNEQ29vcmRzKFxuICAgICAgICAgIG91dFNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgb3V0VGV4U2hhcGUsXG4gICAgICAgICAgZW5hYmxlU2hhcGVVbmlmb3Jtcyk7XG4gICAgY2FzZSA0OlxuICAgICAgcmV0dXJuIGdldE91dHB1dDREQ29vcmRzKFxuICAgICAgICAgIG91dFNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLCBvdXRUZXhTaGFwZSxcbiAgICAgICAgICBlbmFibGVTaGFwZVVuaWZvcm1zKTtcbiAgICBjYXNlIDU6XG4gICAgICByZXR1cm4gZ2V0T3V0cHV0NURDb29yZHMoXG4gICAgICAgICAgb3V0U2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgb3V0VGV4U2hhcGUpO1xuICAgIGNhc2UgNjpcbiAgICAgIHJldHVybiBnZXRPdXRwdXQ2RENvb3JkcyhcbiAgICAgICAgICBvdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICAgICAgb3V0VGV4U2hhcGUpO1xuICAgIGRlZmF1bHQ6XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYCR7b3V0U2hhcGUubGVuZ3RofS1EIG91dHB1dCBzYW1wbGluZyBpcyBub3QgeWV0IHN1cHBvcnRlZGApO1xuICB9XG59XG5cbmZ1bmN0aW9uIGdldEZsb2F0VGV4dHVyZVNhbXBsZVNuaXBwZXQoZ2xzbDogR0xTTCk6IHN0cmluZyB7XG4gIHJldHVybiBgXG4gICAgZmxvYXQgc2FtcGxlVGV4dHVyZShzYW1wbGVyMkQgdGV4dHVyZVNhbXBsZXIsIHZlYzIgdXYpIHtcbiAgICAgIHJldHVybiAke2dsc2wudGV4dHVyZTJEfSh0ZXh0dXJlU2FtcGxlciwgdXYpLnI7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRGbG9hdFRleHR1cmVTZXRSU25pcHBldChnbHNsOiBHTFNMKTogc3RyaW5nIHtcbiAgcmV0dXJuIGBcbiAgICB2b2lkIHNldE91dHB1dChmbG9hdCB2YWwpIHtcbiAgICAgICR7Z2xzbC5vdXRwdXR9ID0gdmVjNCh2YWwsIDAsIDAsIDApO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0RmxvYXRUZXh0dXJlU2V0UkdCQVNuaXBwZXQoZ2xzbDogR0xTTCk6IHN0cmluZyB7XG4gIHJldHVybiBgXG4gICAgdm9pZCBzZXRPdXRwdXQodmVjNCB2YWwpIHtcbiAgICAgICR7Z2xzbC5vdXRwdXR9ID0gdmFsO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0U2hhZGVyUHJlZml4KGdsc2w6IEdMU0wpOiBzdHJpbmcge1xuICBjb25zdCBTSEFERVJfUFJFRklYID0gYCR7Z2xzbC52ZXJzaW9ufVxuICAgIHByZWNpc2lvbiBoaWdocCBmbG9hdDtcbiAgICBwcmVjaXNpb24gaGlnaHAgaW50O1xuICAgIHByZWNpc2lvbiBoaWdocCBzYW1wbGVyMkQ7XG4gICAgJHtnbHNsLnZhcnlpbmdGc30gdmVjMiByZXN1bHRVVjtcbiAgICAke2dsc2wuZGVmaW5lT3V0cHV0fVxuICAgIGNvbnN0IHZlYzIgaGFsZkNSID0gdmVjMigwLjUsIDAuNSk7XG5cbiAgICBzdHJ1Y3QgaXZlYzVcbiAgICB7XG4gICAgICBpbnQgeDtcbiAgICAgIGludCB5O1xuICAgICAgaW50IHo7XG4gICAgICBpbnQgdztcbiAgICAgIGludCB1O1xuICAgIH07XG5cbiAgICBzdHJ1Y3QgaXZlYzZcbiAgICB7XG4gICAgICBpbnQgeDtcbiAgICAgIGludCB5O1xuICAgICAgaW50IHo7XG4gICAgICBpbnQgdztcbiAgICAgIGludCB1O1xuICAgICAgaW50IHY7XG4gICAgfTtcblxuICAgIHVuaWZvcm0gZmxvYXQgTkFOO1xuICAgICR7Z2xzbC5kZWZpbmVTcGVjaWFsTmFOfVxuICAgICR7Z2xzbC5kZWZpbmVTcGVjaWFsSW5mfVxuICAgICR7Z2xzbC5kZWZpbmVSb3VuZH1cblxuICAgIGludCBpbW9kKGludCB4LCBpbnQgeSkge1xuICAgICAgcmV0dXJuIHggLSB5ICogKHggLyB5KTtcbiAgICB9XG5cbiAgICBpbnQgaWRpdihpbnQgYSwgaW50IGIsIGZsb2F0IHNpZ24pIHtcbiAgICAgIGludCByZXMgPSBhIC8gYjtcbiAgICAgIGludCBtb2QgPSBpbW9kKGEsIGIpO1xuICAgICAgaWYgKHNpZ24gPCAwLiAmJiBtb2QgIT0gMCkge1xuICAgICAgICByZXMgLT0gMTtcbiAgICAgIH1cbiAgICAgIHJldHVybiByZXM7XG4gICAgfVxuXG4gICAgLy9CYXNlZCBvbiB0aGUgd29yayBvZiBEYXZlIEhvc2tpbnNcbiAgICAvL2h0dHBzOi8vd3d3LnNoYWRlcnRveS5jb20vdmlldy80ZGpTUldcbiAgICAjZGVmaW5lIEhBU0hTQ0FMRTEgNDQzLjg5NzVcbiAgICBmbG9hdCByYW5kb20oZmxvYXQgc2VlZCl7XG4gICAgICB2ZWMyIHAgPSByZXN1bHRVViAqIHNlZWQ7XG4gICAgICB2ZWMzIHAzICA9IGZyYWN0KHZlYzMocC54eXgpICogSEFTSFNDQUxFMSk7XG4gICAgICBwMyArPSBkb3QocDMsIHAzLnl6eCArIDE5LjE5KTtcbiAgICAgIHJldHVybiBmcmFjdCgocDMueCArIHAzLnkpICogcDMueik7XG4gICAgfVxuXG4gICAgJHtTQU1QTEVfMURfU05JUFBFVH1cbiAgICAke1NBTVBMRV8yRF9TTklQUEVUfVxuICAgICR7U0FNUExFXzNEX1NOSVBQRVR9XG4gIGA7XG5cbiAgcmV0dXJuIFNIQURFUl9QUkVGSVg7XG59XG5cbmNvbnN0IFNBTVBMRV8xRF9TTklQUEVUID0gYFxudmVjMiB1dkZyb21GbGF0KGludCB0ZXhOdW1SLCBpbnQgdGV4TnVtQywgaW50IGluZGV4KSB7XG4gIGludCB0ZXhSID0gaW5kZXggLyB0ZXhOdW1DO1xuICBpbnQgdGV4QyA9IGluZGV4IC0gdGV4UiAqIHRleE51bUM7XG4gIHJldHVybiAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgLyB2ZWMyKHRleE51bUMsIHRleE51bVIpO1xufVxudmVjMiBwYWNrZWRVVmZyb20xRChpbnQgdGV4TnVtUiwgaW50IHRleE51bUMsIGludCBpbmRleCkge1xuICBpbnQgdGV4ZWxJbmRleCA9IGluZGV4IC8gMjtcbiAgaW50IHRleFIgPSB0ZXhlbEluZGV4IC8gdGV4TnVtQztcbiAgaW50IHRleEMgPSB0ZXhlbEluZGV4IC0gdGV4UiAqIHRleE51bUM7XG4gIHJldHVybiAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgLyB2ZWMyKHRleE51bUMsIHRleE51bVIpO1xufVxuYDtcblxuY29uc3QgU0FNUExFXzJEX1NOSVBQRVQgPSBgXG52ZWMyIHBhY2tlZFVWZnJvbTJEKGludCB0ZXhlbHNJbkxvZ2ljYWxSb3csIGludCB0ZXhOdW1SLFxuICBpbnQgdGV4TnVtQywgaW50IHJvdywgaW50IGNvbCkge1xuICBpbnQgdGV4ZWxJbmRleCA9IChyb3cgLyAyKSAqIHRleGVsc0luTG9naWNhbFJvdyArIChjb2wgLyAyKTtcbiAgaW50IHRleFIgPSB0ZXhlbEluZGV4IC8gdGV4TnVtQztcbiAgaW50IHRleEMgPSB0ZXhlbEluZGV4IC0gdGV4UiAqIHRleE51bUM7XG4gIHJldHVybiAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgLyB2ZWMyKHRleE51bUMsIHRleE51bVIpO1xufVxuYDtcblxuY29uc3QgU0FNUExFXzNEX1NOSVBQRVQgPSBgXG52ZWMyIHBhY2tlZFVWZnJvbTNEKGludCB0ZXhOdW1SLCBpbnQgdGV4TnVtQyxcbiAgICBpbnQgdGV4ZWxzSW5CYXRjaCwgaW50IHRleGVsc0luTG9naWNhbFJvdywgaW50IGIsXG4gICAgaW50IHJvdywgaW50IGNvbCkge1xuICBpbnQgaW5kZXggPSBiICogdGV4ZWxzSW5CYXRjaCArIChyb3cgLyAyKSAqIHRleGVsc0luTG9naWNhbFJvdyArIChjb2wgLyAyKTtcbiAgaW50IHRleFIgPSBpbmRleCAvIHRleE51bUM7XG4gIGludCB0ZXhDID0gaW5kZXggLSB0ZXhSICogdGV4TnVtQztcbiAgcmV0dXJuICh2ZWMyKHRleEMsIHRleFIpICsgaGFsZkNSKSAvIHZlYzIodGV4TnVtQywgdGV4TnVtUik7XG59XG5gO1xuXG5jb25zdCBTSEFERVJfUEFDS0VEX1BSRUZJWCA9IGBcbiAgZmxvYXQgZ2V0Q2hhbm5lbCh2ZWM0IGZyYWcsIHZlYzIgaW5uZXJEaW1zKSB7XG4gICAgdmVjMiBtb2RDb29yZCA9IG1vZChpbm5lckRpbXMsIDIuKTtcbiAgICByZXR1cm4gbW9kQ29vcmQueCA9PSAwLiA/XG4gICAgICAobW9kQ29vcmQueSA9PSAwLiA/IGZyYWcuciA6IGZyYWcuZykgOlxuICAgICAgKG1vZENvb3JkLnkgPT0gMC4gPyBmcmFnLmIgOiBmcmFnLmEpO1xuICB9XG4gIGZsb2F0IGdldENoYW5uZWwodmVjNCBmcmFnLCBpbnQgZGltKSB7XG4gICAgZmxvYXQgbW9kQ29vcmQgPSBtb2QoZmxvYXQoZGltKSwgMi4pO1xuICAgIHJldHVybiBtb2RDb29yZCA9PSAwLiA/IGZyYWcuciA6IGZyYWcuZztcbiAgfVxuYDtcblxuZnVuY3Rpb24gZ2V0T3V0cHV0U2NhbGFyQ29vcmRzKCkge1xuICByZXR1cm4gYFxuICAgIGludCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICByZXR1cm4gMDtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldE91dHB1dFBhY2tlZDFEQ29vcmRzKFxuICAgIHNoYXBlOiBbbnVtYmVyXSwgdGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgZW5hYmxlU2hhcGVVbmlmb3JtczogYm9vbGVhbik6IHN0cmluZyB7XG4gIGNvbnN0IHBhY2tlZFRleFNoYXBlID1cbiAgICAgIFtNYXRoLmNlaWwodGV4U2hhcGVbMF0gLyAyKSwgTWF0aC5jZWlsKHRleFNoYXBlWzFdIC8gMildO1xuICBpZiAocGFja2VkVGV4U2hhcGVbMF0gPT09IDEpIHtcbiAgICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgICAgcmV0dXJuIGBcbiAgICAgIGludCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiAyICogaW50KHJlc3VsdFVWLnggKiBjZWlsKGZsb2F0KG91dFRleFNoYXBlWzFdKSAvIDIuMCkpO1xuICAgICAgfVxuICAgIGA7XG4gICAgfVxuXG4gICAgcmV0dXJuIGBcbiAgICAgIGludCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiAyICogaW50KHJlc3VsdFVWLnggKiAke3BhY2tlZFRleFNoYXBlWzFdfS4wKTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG5cbiAgaWYgKHBhY2tlZFRleFNoYXBlWzFdID09PSAxKSB7XG4gICAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICAgIHJldHVybiBgXG4gICAgICBpbnQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgICByZXR1cm4gMiAqIGludChyZXN1bHRVVi55ICogY2VpbChmbG9hdChvdXRUZXhTaGFwZVswXSkgLyAyLjApKTtcbiAgICAgIH1cbiAgICBgO1xuICAgIH1cblxuICAgIHJldHVybiBgXG4gICAgICBpbnQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgICByZXR1cm4gMiAqIGludChyZXN1bHRVVi55ICogJHtwYWNrZWRUZXhTaGFwZVswXX0uMCk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuXG4gIGlmIChlbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgcmV0dXJuIGBcbiAgICBpbnQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcGFja2VkVGV4U2hhcGUgPSBpdmVjMihjZWlsKGZsb2F0KG91dFRleFNoYXBlWzBdKSAvIDIuMCksIGNlaWwoZmxvYXQob3V0VGV4U2hhcGVbMV0pIC8gMi4wKSk7XG4gICAgICBpdmVjMiByZXNUZXhSQyA9IGl2ZWMyKHJlc3VsdFVWLnl4ICpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdmVjMihwYWNrZWRUZXhTaGFwZVswXSwgcGFja2VkVGV4U2hhcGVbMV0pKTtcbiAgICAgIHJldHVybiAyICogKHJlc1RleFJDLnggKiBwYWNrZWRUZXhTaGFwZVsxXSArIHJlc1RleFJDLnkpO1xuICAgIH1cbiAgYDtcbiAgfVxuXG4gIHJldHVybiBgXG4gICAgaW50IGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgIGl2ZWMyIHJlc1RleFJDID0gaXZlYzIocmVzdWx0VVYueXggKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2ZWMyKCR7cGFja2VkVGV4U2hhcGVbMF19LCAke3BhY2tlZFRleFNoYXBlWzFdfSkpO1xuICAgICAgcmV0dXJuIDIgKiAocmVzVGV4UkMueCAqICR7cGFja2VkVGV4U2hhcGVbMV19ICsgcmVzVGV4UkMueSk7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRPdXRwdXQxRENvb3JkcyhcbiAgICBzaGFwZTogW251bWJlcl0sIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgIGVuYWJsZVNoYXBlVW5pZm9ybXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBpZiAodGV4U2hhcGVbMF0gPT09IDEpIHtcbiAgICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgICAgcmV0dXJuIGBcbiAgICAgIGludCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiBpbnQocmVzdWx0VVYueCAqIGZsb2F0KG91dFRleFNoYXBlWzFdKSk7XG4gICAgICB9XG4gICAgYDtcbiAgICB9XG4gICAgcmV0dXJuIGBcbiAgICAgIGludCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiBpbnQocmVzdWx0VVYueCAqICR7dGV4U2hhcGVbMV19LjApO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgaWYgKHRleFNoYXBlWzFdID09PSAxKSB7XG4gICAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICAgIHJldHVybiBgXG4gICAgICBpbnQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgICByZXR1cm4gaW50KHJlc3VsdFVWLnkgKiBmbG9hdChvdXRUZXhTaGFwZVswXSkpO1xuICAgICAgfVxuICAgIGA7XG4gICAgfVxuICAgIHJldHVybiBgXG4gICAgICBpbnQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgICByZXR1cm4gaW50KHJlc3VsdFVWLnkgKiAke3RleFNoYXBlWzBdfS4wKTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIGlmIChlbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgcmV0dXJuIGBcbiAgICBpbnQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIob3V0VGV4U2hhcGVbMF0sIG91dFRleFNoYXBlWzFdKSk7XG4gICAgICByZXR1cm4gcmVzVGV4UkMueCAqIG91dFRleFNoYXBlWzFdICsgcmVzVGV4UkMueTtcbiAgICB9XG4gIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICBpbnQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIoJHt0ZXhTaGFwZVswXX0sICR7dGV4U2hhcGVbMV19KSk7XG4gICAgICByZXR1cm4gcmVzVGV4UkMueCAqICR7dGV4U2hhcGVbMV19ICsgcmVzVGV4UkMueTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldE91dHB1dFBhY2tlZDNEQ29vcmRzKFxuICAgIHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgIGVuYWJsZVNoYXBlVW5pZm9ybXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgIHJldHVybiBgXG4gICAgaXZlYzMgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcGFja2VkVGV4U2hhcGUgPSBpdmVjMihjZWlsKGZsb2F0KG91dFRleFNoYXBlWzBdKSAvIDIuMCksIGNlaWwoZmxvYXQob3V0VGV4U2hhcGVbMV0pIC8gMi4wKSk7XG4gICAgICBpbnQgdGV4ZWxzSW5Mb2dpY2FsUm93ID0gaW50KGNlaWwoZmxvYXQob3V0U2hhcGVbMl0pIC8gMi4wKSk7XG4gICAgICBpbnQgdGV4ZWxzSW5CYXRjaCA9IHRleGVsc0luTG9naWNhbFJvdyAqIGludChjZWlsKGZsb2F0KG91dFNoYXBlWzFdKSAvIDIuMCkpO1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIocGFja2VkVGV4U2hhcGVbMF0sIHBhY2tlZFRleFNoYXBlWzFdKSk7XG4gICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogcGFja2VkVGV4U2hhcGVbMV0gKyByZXNUZXhSQy55O1xuXG4gICAgICBpbnQgYiA9IGluZGV4IC8gdGV4ZWxzSW5CYXRjaDtcbiAgICAgIGluZGV4IC09IGIgKiB0ZXhlbHNJbkJhdGNoO1xuXG4gICAgICBpbnQgciA9IDIgKiAoaW5kZXggLyB0ZXhlbHNJbkxvZ2ljYWxSb3cpO1xuICAgICAgaW50IGMgPSBpbW9kKGluZGV4LCB0ZXhlbHNJbkxvZ2ljYWxSb3cpICogMjtcblxuICAgICAgcmV0dXJuIGl2ZWMzKGIsIHIsIGMpO1xuICAgIH1cbiAgYDtcbiAgfVxuXG4gIGNvbnN0IHBhY2tlZFRleFNoYXBlID1cbiAgICAgIFtNYXRoLmNlaWwodGV4U2hhcGVbMF0gLyAyKSwgTWF0aC5jZWlsKHRleFNoYXBlWzFdIC8gMildO1xuICBjb25zdCB0ZXhlbHNJbkxvZ2ljYWxSb3cgPSBNYXRoLmNlaWwoc2hhcGVbMl0gLyAyKTtcbiAgY29uc3QgdGV4ZWxzSW5CYXRjaCA9IHRleGVsc0luTG9naWNhbFJvdyAqIE1hdGguY2VpbChzaGFwZVsxXSAvIDIpO1xuXG4gIHJldHVybiBgXG4gICAgaXZlYzMgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIoJHtwYWNrZWRUZXhTaGFwZVswXX0sICR7cGFja2VkVGV4U2hhcGVbMV19KSk7XG4gICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogJHtwYWNrZWRUZXhTaGFwZVsxXX0gKyByZXNUZXhSQy55O1xuXG4gICAgICBpbnQgYiA9IGluZGV4IC8gJHt0ZXhlbHNJbkJhdGNofTtcbiAgICAgIGluZGV4IC09IGIgKiAke3RleGVsc0luQmF0Y2h9O1xuXG4gICAgICBpbnQgciA9IDIgKiAoaW5kZXggLyAke3RleGVsc0luTG9naWNhbFJvd30pO1xuICAgICAgaW50IGMgPSBpbW9kKGluZGV4LCAke3RleGVsc0luTG9naWNhbFJvd30pICogMjtcblxuICAgICAgcmV0dXJuIGl2ZWMzKGIsIHIsIGMpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0T3V0cHV0M0RDb29yZHMoXG4gICAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgdGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgZW5hYmxlU2hhcGVVbmlmb3JtczogYm9vbGVhbik6IHN0cmluZyB7XG4gIGlmIChlbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgY29uc3QgY29vcmRzRnJvbUluZGV4U25pcHBldCA9XG4gICAgICAgIHNoYWRlcl91dGlsLmdldE91dHB1dExvZ2ljYWxDb29yZGluYXRlc0Zyb21GbGF0SW5kZXhCeVVuaWZvcm0oXG4gICAgICAgICAgICBbJ3InLCAnYycsICdkJ10sIHNoYXBlKTtcblxuICAgIHJldHVybiBgXG4gIGl2ZWMzIGdldE91dHB1dENvb3JkcygpIHtcbiAgICBpdmVjMiByZXNUZXhSQyA9IGl2ZWMyKHJlc3VsdFVWLnl4ICpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIob3V0VGV4U2hhcGVbMF0sIG91dFRleFNoYXBlWzFdKSk7XG4gICAgaW50IGluZGV4ID0gcmVzVGV4UkMueCAqIG91dFRleFNoYXBlWzFdICsgcmVzVGV4UkMueTtcbiAgICAke2Nvb3Jkc0Zyb21JbmRleFNuaXBwZXR9XG4gICAgcmV0dXJuIGl2ZWMzKHIsIGMsIGQpO1xuICB9XG5gO1xuICB9XG4gIGNvbnN0IGNvb3Jkc0Zyb21JbmRleFNuaXBwZXQgPVxuICAgICAgc2hhZGVyX3V0aWwuZ2V0TG9naWNhbENvb3JkaW5hdGVzRnJvbUZsYXRJbmRleChbJ3InLCAnYycsICdkJ10sIHNoYXBlKTtcblxuICByZXR1cm4gYFxuICAgIGl2ZWMzIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgIGl2ZWMyIHJlc1RleFJDID0gaXZlYzIocmVzdWx0VVYueXggKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2ZWMyKCR7dGV4U2hhcGVbMF19LCAke3RleFNoYXBlWzFdfSkpO1xuICAgICAgaW50IGluZGV4ID0gcmVzVGV4UkMueCAqICR7dGV4U2hhcGVbMV19ICsgcmVzVGV4UkMueTtcbiAgICAgICR7Y29vcmRzRnJvbUluZGV4U25pcHBldH1cbiAgICAgIHJldHVybiBpdmVjMyhyLCBjLCBkKTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldE91dHB1dFBhY2tlZE5EQ29vcmRzKFxuICAgIHNoYXBlOiBudW1iZXJbXSwgdGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgZW5hYmxlU2hhcGVVbmlmb3JtczogYm9vbGVhbik6IHN0cmluZyB7XG4gIGlmIChlbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgLy8gVE9ETzogc3VwcG9ydCA1ZCBhbmQgNmRcbiAgICByZXR1cm4gYFxuICAgIGl2ZWM0IGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgIGl2ZWMyIHBhY2tlZFRleFNoYXBlID0gaXZlYzIoY2VpbChmbG9hdChvdXRUZXhTaGFwZVswXSkgLyAyLjApLCBjZWlsKGZsb2F0KG91dFRleFNoYXBlWzFdKSAvIDIuMCkpO1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIocGFja2VkVGV4U2hhcGVbMF0sIHBhY2tlZFRleFNoYXBlWzFdKSk7XG4gICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogcGFja2VkVGV4U2hhcGVbMV0gKyByZXNUZXhSQy55O1xuXG4gICAgICBpbnQgdGV4ZWxzSW5Mb2dpY2FsUm93ID0gaW50KGNlaWwoZmxvYXQob3V0U2hhcGVbM10pIC8gMi4wKSk7XG4gICAgICBpbnQgdGV4ZWxzSW5CYXRjaCA9IHRleGVsc0luTG9naWNhbFJvdyAqIGludChjZWlsKGZsb2F0KG91dFNoYXBlWzJdKSAvIDIuMCkpO1xuICAgICAgaW50IHRleGVsc0luQmF0Y2hOID0gdGV4ZWxzSW5CYXRjaCAqIG91dFNoYXBlWzFdO1xuXG4gICAgICBpbnQgYjIgPSBpbmRleCAvIHRleGVsc0luQmF0Y2hOO1xuICAgICAgaW5kZXggLT0gYjIgKiB0ZXhlbHNJbkJhdGNoTjtcblxuICAgICAgaW50IGIgPSBpbmRleCAvIHRleGVsc0luQmF0Y2g7XG4gICAgICBpbmRleCAtPSBiICogdGV4ZWxzSW5CYXRjaDtcblxuICAgICAgaW50IHIgPSAyICogKGluZGV4IC8gdGV4ZWxzSW5Mb2dpY2FsUm93KTtcbiAgICAgIGludCBjID0gaW1vZChpbmRleCwgdGV4ZWxzSW5Mb2dpY2FsUm93KSAqIDI7XG5cbiAgICAgIHJldHVybiBpdmVjNChiMiwgYiwgciwgYyk7XG4gICAgfVxuICBgO1xuICB9XG4gIGNvbnN0IHBhY2tlZFRleFNoYXBlID1cbiAgICAgIFtNYXRoLmNlaWwodGV4U2hhcGVbMF0gLyAyKSwgTWF0aC5jZWlsKHRleFNoYXBlWzFdIC8gMildO1xuXG4gIGNvbnN0IHRleGVsc0luTG9naWNhbFJvdyA9IE1hdGguY2VpbChzaGFwZVtzaGFwZS5sZW5ndGggLSAxXSAvIDIpO1xuICBjb25zdCB0ZXhlbHNJbkJhdGNoID1cbiAgICAgIHRleGVsc0luTG9naWNhbFJvdyAqIE1hdGguY2VpbChzaGFwZVtzaGFwZS5sZW5ndGggLSAyXSAvIDIpO1xuICBsZXQgdGV4ZWxzSW5CYXRjaE4gPSB0ZXhlbHNJbkJhdGNoO1xuICBsZXQgYmF0Y2hlcyA9IGBgO1xuICBsZXQgY29vcmRzID0gJ2IsIHIsIGMnO1xuXG4gIGZvciAobGV0IGIgPSAyOyBiIDwgc2hhcGUubGVuZ3RoIC0gMTsgYisrKSB7XG4gICAgdGV4ZWxzSW5CYXRjaE4gKj0gc2hhcGVbc2hhcGUubGVuZ3RoIC0gYiAtIDFdO1xuICAgIGJhdGNoZXMgPSBgXG4gICAgICBpbnQgYiR7Yn0gPSBpbmRleCAvICR7dGV4ZWxzSW5CYXRjaE59O1xuICAgICAgaW5kZXggLT0gYiR7Yn0gKiAke3RleGVsc0luQmF0Y2hOfTtcbiAgICBgICsgYmF0Y2hlcztcbiAgICBjb29yZHMgPSBgYiR7Yn0sIGAgKyBjb29yZHM7XG4gIH1cblxuICByZXR1cm4gYFxuICAgIGl2ZWMke3NoYXBlLmxlbmd0aH0gZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIoJHtwYWNrZWRUZXhTaGFwZVswXX0sICR7cGFja2VkVGV4U2hhcGVbMV19KSk7XG4gICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogJHtwYWNrZWRUZXhTaGFwZVsxXX0gKyByZXNUZXhSQy55O1xuXG4gICAgICAke2JhdGNoZXN9XG5cbiAgICAgIGludCBiID0gaW5kZXggLyAke3RleGVsc0luQmF0Y2h9O1xuICAgICAgaW5kZXggLT0gYiAqICR7dGV4ZWxzSW5CYXRjaH07XG5cbiAgICAgIGludCByID0gMiAqIChpbmRleCAvICR7dGV4ZWxzSW5Mb2dpY2FsUm93fSk7XG4gICAgICBpbnQgYyA9IGltb2QoaW5kZXgsICR7dGV4ZWxzSW5Mb2dpY2FsUm93fSkgKiAyO1xuXG4gICAgICByZXR1cm4gaXZlYyR7c2hhcGUubGVuZ3RofSgke2Nvb3Jkc30pO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0T3V0cHV0NERDb29yZHMoXG4gICAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSxcbiAgICBlbmFibGVTaGFwZVVuaWZvcm1zOiBib29sZWFuKTogc3RyaW5nIHtcbiAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICBjb25zdCBjb29yZHNGcm9tSW5kZXhTbmlwcGV0ID1cbiAgICAgICAgc2hhZGVyX3V0aWwuZ2V0T3V0cHV0TG9naWNhbENvb3JkaW5hdGVzRnJvbUZsYXRJbmRleEJ5VW5pZm9ybShcbiAgICAgICAgICAgIFsncicsICdjJywgJ2QnLCAnZDInXSwgc2hhcGUpO1xuXG4gICAgcmV0dXJuIGBcbiAgICBpdmVjNCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICBpdmVjMiByZXNUZXhSQyA9IGl2ZWMyKHJlc3VsdFVWLnl4ICpcbiAgICAgICAgdmVjMihvdXRUZXhTaGFwZVswXSwgb3V0VGV4U2hhcGVbMV0pKTtcbiAgICAgIGludCBpbmRleCA9IHJlc1RleFJDLnggKiBvdXRUZXhTaGFwZVsxXSArIHJlc1RleFJDLnk7XG4gICAgICAke2Nvb3Jkc0Zyb21JbmRleFNuaXBwZXR9XG4gICAgICByZXR1cm4gaXZlYzQociwgYywgZCwgZDIpO1xuICAgIH1cbiAgYDtcbiAgfVxuICBjb25zdCBjb29yZHNGcm9tSW5kZXhTbmlwcGV0ID0gc2hhZGVyX3V0aWwuZ2V0TG9naWNhbENvb3JkaW5hdGVzRnJvbUZsYXRJbmRleChcbiAgICAgIFsncicsICdjJywgJ2QnLCAnZDInXSwgc2hhcGUpO1xuXG4gIHJldHVybiBgXG4gICAgaXZlYzQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgIHZlYzIoJHt0ZXhTaGFwZVswXX0sICR7dGV4U2hhcGVbMV19KSk7XG4gICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogJHt0ZXhTaGFwZVsxXX0gKyByZXNUZXhSQy55O1xuICAgICAgJHtjb29yZHNGcm9tSW5kZXhTbmlwcGV0fVxuICAgICAgcmV0dXJuIGl2ZWM0KHIsIGMsIGQsIGQyKTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldE91dHB1dDVEQ29vcmRzKFxuICAgIHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3QgY29vcmRzRnJvbUluZGV4U25pcHBldCA9IHNoYWRlcl91dGlsLmdldExvZ2ljYWxDb29yZGluYXRlc0Zyb21GbGF0SW5kZXgoXG4gICAgICBbJ3InLCAnYycsICdkJywgJ2QyJywgJ2QzJ10sIHNoYXBlKTtcblxuICByZXR1cm4gYFxuICAgIGl2ZWM1IGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgIGl2ZWMyIHJlc1RleFJDID0gaXZlYzIocmVzdWx0VVYueXggKiB2ZWMyKCR7dGV4U2hhcGVbMF19LFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAke3RleFNoYXBlWzFdfSkpO1xuXG4gICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogJHt0ZXhTaGFwZVsxXX0gKyByZXNUZXhSQy55O1xuXG4gICAgICAke2Nvb3Jkc0Zyb21JbmRleFNuaXBwZXR9XG5cbiAgICAgIGl2ZWM1IG91dFNoYXBlID0gaXZlYzUociwgYywgZCwgZDIsIGQzKTtcbiAgICAgIHJldHVybiBvdXRTaGFwZTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldE91dHB1dDZEQ29vcmRzKFxuICAgIHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgdGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0pOiBzdHJpbmcge1xuICBjb25zdCBjb29yZHNGcm9tSW5kZXhTbmlwcGV0ID0gc2hhZGVyX3V0aWwuZ2V0TG9naWNhbENvb3JkaW5hdGVzRnJvbUZsYXRJbmRleChcbiAgICAgIFsncicsICdjJywgJ2QnLCAnZDInLCAnZDMnLCAnZDQnXSwgc2hhcGUpO1xuXG4gIHJldHVybiBgXG4gICAgaXZlYzYgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgIHZlYzIoJHt0ZXhTaGFwZVswXX0sICR7dGV4U2hhcGVbMV19KSk7XG4gICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogJHt0ZXhTaGFwZVsxXX0gKyByZXNUZXhSQy55O1xuXG4gICAgICAke2Nvb3Jkc0Zyb21JbmRleFNuaXBwZXR9XG5cbiAgICAgIGl2ZWM2IHJlc3VsdCA9IGl2ZWM2KHIsIGMsIGQsIGQyLCBkMywgZDQpO1xuICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldE91dHB1dFBhY2tlZDJEQ29vcmRzKFxuICAgIHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSxcbiAgICBlbmFibGVTaGFwZVVuaWZvcm1zOiBib29sZWFuKTogc3RyaW5nIHtcbiAgY29uc3QgcGFja2VkVGV4U2hhcGUgPVxuICAgICAgW01hdGguY2VpbCh0ZXhTaGFwZVswXSAvIDIpLCBNYXRoLmNlaWwodGV4U2hhcGVbMV0gLyAyKV07XG4gIGlmICh1dGlsLmFycmF5c0VxdWFsKHNoYXBlLCB0ZXhTaGFwZSkpIHtcbiAgICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgICAgcmV0dXJuIGBcbiAgICAgIGl2ZWMyIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgICAgaXZlYzIgcGFja2VkVGV4U2hhcGUgPSBpdmVjMihjZWlsKGZsb2F0KG91dFRleFNoYXBlWzBdKSAvIDIuMCksIGNlaWwoZmxvYXQob3V0VGV4U2hhcGVbMV0pIC8gMi4wKSk7XG4gICAgICAgIHJldHVybiAyICogaXZlYzIocmVzdWx0VVYueXggKiB2ZWMyKHBhY2tlZFRleFNoYXBlWzBdLCBwYWNrZWRUZXhTaGFwZVsxXSkpO1xuICAgICAgfVxuICAgIGA7XG4gICAgfVxuXG4gICAgcmV0dXJuIGBcbiAgICAgIGl2ZWMyIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgICAgcmV0dXJuIDIgKiBpdmVjMihyZXN1bHRVVi55eCAqIHZlYzIoJHtwYWNrZWRUZXhTaGFwZVswXX0sICR7XG4gICAgICAgIHBhY2tlZFRleFNoYXBlWzFdfSkpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICAvLyB0ZXhlbHMgbmVlZGVkIHRvIGFjY29tbW9kYXRlIGEgbG9naWNhbCByb3dcbiAgY29uc3QgdGV4ZWxzSW5Mb2dpY2FsUm93ID0gTWF0aC5jZWlsKHNoYXBlWzFdIC8gMik7XG5cbiAgLyoqXG4gICAqIGdldE91dHB1dENvb3Jkc1xuICAgKlxuICAgKiByZXNUZXhSQzogVGhlIHJvd3MgYW5kIGNvbHVtbnMgb2YgdGhlIHRleGVscy4gSWYgeW91IG1vdmUgb3ZlciBvbmVcbiAgICogdGV4ZWwgdG8gdGhlIHJpZ2h0IGluIHRoZSBwYWNrZWQgdGV4dHVyZSwgeW91IGFyZSBtb3Zpbmcgb3ZlciBvbmUgY29sdW1uXG4gICAqIChub3QgdHdvKS5cbiAgICpcbiAgICogaW5kZXg6IFRoZSB0ZXhlbCBpbmRleFxuICAgKi9cbiAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICByZXR1cm4gYFxuICAgIGl2ZWMyIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgIGl2ZWMyIHBhY2tlZFRleFNoYXBlID0gaXZlYzIoY2VpbChmbG9hdChvdXRUZXhTaGFwZVswXSkgLyAyLjApLCBjZWlsKGZsb2F0KG91dFRleFNoYXBlWzFdKSAvIDIuMCkpO1xuICAgICAgaW50IHRleGVsc0luTG9naWNhbFJvdyA9IGludChjZWlsKGZsb2F0KG91dFNoYXBlWzFdKSAvIDIuMCkpO1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIocGFja2VkVGV4U2hhcGVbMF0sIHBhY2tlZFRleFNoYXBlWzFdKSk7XG5cbiAgICAgIGludCBpbmRleCA9IHJlc1RleFJDLnggKiBwYWNrZWRUZXhTaGFwZVsxXSArIHJlc1RleFJDLnk7XG4gICAgICBpbnQgciA9IDIgKiAoaW5kZXggLyB0ZXhlbHNJbkxvZ2ljYWxSb3cpO1xuICAgICAgaW50IGMgPSBpbW9kKGluZGV4LCB0ZXhlbHNJbkxvZ2ljYWxSb3cpICogMjtcblxuICAgICAgcmV0dXJuIGl2ZWMyKHIsIGMpO1xuICAgIH1cbiAgYDtcbiAgfVxuXG4gIHJldHVybiBgXG4gICAgaXZlYzIgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIoJHtwYWNrZWRUZXhTaGFwZVswXX0sICR7cGFja2VkVGV4U2hhcGVbMV19KSk7XG5cbiAgICAgIGludCBpbmRleCA9IHJlc1RleFJDLnggKiAke3BhY2tlZFRleFNoYXBlWzFdfSArIHJlc1RleFJDLnk7XG4gICAgICBpbnQgciA9IDIgKiAoaW5kZXggLyAke3RleGVsc0luTG9naWNhbFJvd30pO1xuICAgICAgaW50IGMgPSBpbW9kKGluZGV4LCAke3RleGVsc0luTG9naWNhbFJvd30pICogMjtcblxuICAgICAgcmV0dXJuIGl2ZWMyKHIsIGMpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0T3V0cHV0MkRDb29yZHMoXG4gICAgc2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgIGVuYWJsZVNoYXBlVW5pZm9ybXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBpZiAodXRpbC5hcnJheXNFcXVhbChzaGFwZSwgdGV4U2hhcGUpKSB7XG4gICAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICAgIHJldHVybiBgXG4gICAgICBpdmVjMiBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiBpdmVjMihyZXN1bHRVVi55eCAqIHZlYzIob3V0VGV4U2hhcGVbMF0sIG91dFRleFNoYXBlWzFdKSk7XG4gICAgICB9XG4gICAgYDtcbiAgICB9XG4gICAgcmV0dXJuIGBcbiAgICAgIGl2ZWMyIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgICAgcmV0dXJuIGl2ZWMyKHJlc3VsdFVWLnl4ICogdmVjMigke3RleFNoYXBlWzBdfSwgJHt0ZXhTaGFwZVsxXX0pKTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIGlmIChzaGFwZVsxXSA9PT0gMSkge1xuICAgIGlmIChlbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgICByZXR1cm4gYFxuICAgICAgaXZlYzIgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgICBpdmVjMiByZXNUZXhSQyA9IGl2ZWMyKHJlc3VsdFVWLnl4ICpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2ZWMyKG91dFRleFNoYXBlWzBdLCBvdXRUZXhTaGFwZVsxXSkpO1xuICAgICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogb3V0VGV4U2hhcGVbMV0gKyByZXNUZXhSQy55O1xuICAgICAgICByZXR1cm4gaXZlYzIoaW5kZXgsIDApO1xuICAgICAgfVxuICAgIGA7XG4gICAgfVxuICAgIHJldHVybiBgXG4gICAgICBpdmVjMiBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIGl2ZWMyIHJlc1RleFJDID0gaXZlYzIocmVzdWx0VVYueXggKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIoJHt0ZXhTaGFwZVswXX0sICR7dGV4U2hhcGVbMV19KSk7XG4gICAgICAgIGludCBpbmRleCA9IHJlc1RleFJDLnggKiAke3RleFNoYXBlWzFdfSArIHJlc1RleFJDLnk7XG4gICAgICAgIHJldHVybiBpdmVjMihpbmRleCwgMCk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICBpZiAoc2hhcGVbMF0gPT09IDEpIHtcbiAgICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgICAgcmV0dXJuIGBcbiAgICAgIGl2ZWMyIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdmVjMihvdXRUZXhTaGFwZVswXSwgb3V0VGV4U2hhcGVbMV0pKTtcbiAgICAgICAgaW50IGluZGV4ID0gcmVzVGV4UkMueCAqIG91dFRleFNoYXBlWzFdICsgcmVzVGV4UkMueTtcbiAgICAgICAgcmV0dXJuIGl2ZWMyKDAsIGluZGV4KTtcbiAgICAgIH1cbiAgICBgO1xuICAgIH1cbiAgICByZXR1cm4gYFxuICAgICAgaXZlYzIgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgICBpdmVjMiByZXNUZXhSQyA9IGl2ZWMyKHJlc3VsdFVWLnl4ICpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2ZWMyKCR7dGV4U2hhcGVbMF19LCAke3RleFNoYXBlWzFdfSkpO1xuICAgICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogJHt0ZXhTaGFwZVsxXX0gKyByZXNUZXhSQy55O1xuICAgICAgICByZXR1cm4gaXZlYzIoMCwgaW5kZXgpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICByZXR1cm4gYFxuICAgIGl2ZWMyIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgIGl2ZWMyIHJlc1RleFJDID0gaXZlYzIocmVzdWx0VVYueXggKlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICB2ZWMyKG91dFRleFNoYXBlWzBdLCBvdXRUZXhTaGFwZVsxXSkpO1xuICAgICAgaW50IGluZGV4ID0gcmVzVGV4UkMueCAqIG91dFRleFNoYXBlWzFdICsgcmVzVGV4UkMueTtcbiAgICAgIGludCByID0gaW5kZXggLyBvdXRTaGFwZVsxXTtcbiAgICAgIGludCBjID0gaW5kZXggLSByICogb3V0U2hhcGVbMV07XG4gICAgICByZXR1cm4gaXZlYzIociwgYyk7XG4gICAgfVxuICBgO1xuICB9XG4gIHJldHVybiBgXG4gICAgaXZlYzIgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgaXZlYzIgcmVzVGV4UkMgPSBpdmVjMihyZXN1bHRVVi55eCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzIoJHt0ZXhTaGFwZVswXX0sICR7dGV4U2hhcGVbMV19KSk7XG4gICAgICBpbnQgaW5kZXggPSByZXNUZXhSQy54ICogJHt0ZXhTaGFwZVsxXX0gKyByZXNUZXhSQy55O1xuICAgICAgaW50IHIgPSBpbmRleCAvICR7c2hhcGVbMV19O1xuICAgICAgaW50IGMgPSBpbmRleCAtIHIgKiAke3NoYXBlWzFdfTtcbiAgICAgIHJldHVybiBpdmVjMihyLCBjKTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldEZsYXRPZmZzZXRVbmlmb3JtTmFtZSh0ZXhOYW1lOiBzdHJpbmcpOiBzdHJpbmcge1xuICByZXR1cm4gYG9mZnNldCR7dGV4TmFtZX1gO1xufVxuXG5mdW5jdGlvbiBnZXRQYWNrZWRTYW1wbGVyU2NhbGFyKGlucHV0SW5mbzogSW5wdXRJbmZvKTogc3RyaW5nIHtcbiAgY29uc3QgdGV4TmFtZSA9IGlucHV0SW5mby5uYW1lO1xuICBjb25zdCBmdW5jTmFtZSA9ICdnZXQnICsgdGV4TmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSArIHRleE5hbWUuc2xpY2UoMSk7XG4gIGNvbnN0IGdsc2wgPSBnZXRHbHNsRGlmZmVyZW5jZXMoKTtcbiAgcmV0dXJuIGBcbiAgICB2ZWM0ICR7ZnVuY05hbWV9KCkge1xuICAgICAgcmV0dXJuICR7Z2xzbC50ZXh0dXJlMkR9KCR7dGV4TmFtZX0sIGhhbGZDUik7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyU2NhbGFyKFxuICAgIGlucHV0SW5mbzogSW5wdXRJbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zOiBib29sZWFuKTogc3RyaW5nIHtcbiAgY29uc3QgdGV4TmFtZSA9IGlucHV0SW5mby5uYW1lO1xuICBjb25zdCBmdW5jTmFtZSA9ICdnZXQnICsgdGV4TmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSArIHRleE5hbWUuc2xpY2UoMSk7XG4gIGlmIChpbnB1dEluZm8uc2hhcGVJbmZvLmlzVW5pZm9ybSkge1xuICAgIHJldHVybiBgZmxvYXQgJHtmdW5jTmFtZX0oKSB7cmV0dXJuICR7dGV4TmFtZX07fWA7XG4gIH1cbiAgY29uc3QgW3RleE51bVIsIHRleE51bUNdID0gaW5wdXRJbmZvLnNoYXBlSW5mby50ZXhTaGFwZTtcbiAgaWYgKHRleE51bVIgPT09IDEgJiYgdGV4TnVtQyA9PT0gMSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfSgpIHtcbiAgICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgaGFsZkNSKTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG5cbiAgY29uc3Qgb2Zmc2V0ID0gZ2V0RmxhdE9mZnNldFVuaWZvcm1OYW1lKHRleE5hbWUpO1xuICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oKSB7XG4gICAgICB2ZWMyIHV2ID0gdXZGcm9tRmxhdCgke3RleE5hbWV9VGV4U2hhcGVbMF0sICR7dGV4TmFtZX1UZXhTaGFwZVsxXSwgJHtcbiAgICAgICAgb2Zmc2V0fSk7XG4gICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xuICB9XG5cbiAgY29uc3QgW3ROdW1SLCB0TnVtQ10gPSBpbnB1dEluZm8uc2hhcGVJbmZvLnRleFNoYXBlO1xuICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KCkge1xuICAgICAgdmVjMiB1diA9IHV2RnJvbUZsYXQoJHt0TnVtUn0sICR7dE51bUN9LCAke29mZnNldH0pO1xuICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0UGFja2VkU2FtcGxlcjFEKFxuICAgIGlucHV0SW5mbzogSW5wdXRJbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zOiBib29sZWFuKTogc3RyaW5nIHtcbiAgY29uc3QgdGV4TmFtZSA9IGlucHV0SW5mby5uYW1lO1xuICBjb25zdCBmdW5jTmFtZSA9ICdnZXQnICsgdGV4TmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSArIHRleE5hbWUuc2xpY2UoMSk7XG4gIGNvbnN0IHRleFNoYXBlID0gaW5wdXRJbmZvLnNoYXBlSW5mby50ZXhTaGFwZTtcbiAgY29uc3QgZ2xzbCA9IGdldEdsc2xEaWZmZXJlbmNlcygpO1xuICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgIHJldHVybiBgXG4gICAgdmVjNCAke2Z1bmNOYW1lfShpbnQgaW5kZXgpIHtcbiAgICAgIGl2ZWMyIHBhY2tlZFRleFNoYXBlID0gaXZlYzIoY2VpbChmbG9hdCgke1xuICAgICAgICB0ZXhOYW1lfVRleFNoYXBlWzBdKSAvIDIuMCksIGNlaWwoZmxvYXQoJHt0ZXhOYW1lfVRleFNoYXBlWzFdKSAvIDIuMCkpO1xuICAgICAgdmVjMiB1diA9IHBhY2tlZFVWZnJvbTFEKFxuICAgICAgICBwYWNrZWRUZXhTaGFwZVswXSwgcGFja2VkVGV4U2hhcGVbMV0sIGluZGV4KTtcbiAgICAgIHJldHVybiAke2dsc2wudGV4dHVyZTJEfSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xuICB9XG4gIGNvbnN0IHBhY2tlZFRleFNoYXBlID1cbiAgICAgIFtNYXRoLmNlaWwodGV4U2hhcGVbMF0gLyAyKSwgTWF0aC5jZWlsKHRleFNoYXBlWzFdIC8gMildO1xuICByZXR1cm4gYFxuICAgIHZlYzQgJHtmdW5jTmFtZX0oaW50IGluZGV4KSB7XG4gICAgICB2ZWMyIHV2ID0gcGFja2VkVVZmcm9tMUQoXG4gICAgICAgICR7cGFja2VkVGV4U2hhcGVbMF19LCAke3BhY2tlZFRleFNoYXBlWzFdfSwgaW5kZXgpO1xuICAgICAgcmV0dXJuICR7Z2xzbC50ZXh0dXJlMkR9KCR7dGV4TmFtZX0sIHV2KTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldFNhbXBsZXIxRChcbiAgICBpbnB1dEluZm86IElucHV0SW5mbywgZW5hYmxlU2hhcGVVbmlmb3JtczogYm9vbGVhbik6IHN0cmluZyB7XG4gIGNvbnN0IHRleE5hbWUgPSBpbnB1dEluZm8ubmFtZTtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuXG4gIGlmIChpbnB1dEluZm8uc2hhcGVJbmZvLmlzVW5pZm9ybSkge1xuICAgIC8vIFVuaWZvcm0gYXJyYXlzIHdpbGwgYmUgbGVzcyB0aGFuIDY1NTA1IChubyByaXNrIG9mIGZsb2F0MTYgb3ZlcmZsb3cpLlxuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgaW5kZXgpIHtcbiAgICAgICAgJHtnZXRVbmlmb3JtU2FtcGxlcihpbnB1dEluZm8pfVxuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICBjb25zdCB0ZXhTaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8udGV4U2hhcGU7XG4gIGNvbnN0IHROdW1SID0gdGV4U2hhcGVbMF07XG4gIGNvbnN0IHROdW1DID0gdGV4U2hhcGVbMV07XG5cbiAgaWYgKHROdW1DID09PSAxICYmIHROdW1SID09PSAxKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCBpbmRleCkge1xuICAgICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCBoYWxmQ1IpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgY29uc3Qgb2Zmc2V0ID0gZ2V0RmxhdE9mZnNldFVuaWZvcm1OYW1lKHRleE5hbWUpO1xuICBpZiAodE51bUMgPT09IDEpIHtcbiAgICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCBpbmRleCkge1xuICAgICAgICB2ZWMyIHV2ID0gdmVjMigwLjUsIChmbG9hdChpbmRleCArICR7b2Zmc2V0fSkgKyAwLjUpIC8gZmxvYXQoJHtcbiAgICAgICAgICB0ZXhOYW1lfVRleFNoYXBlWzBdKSk7XG4gICAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICAgIH1cblxuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgaW5kZXgpIHtcbiAgICAgICAgdmVjMiB1diA9IHZlYzIoMC41LCAoZmxvYXQoaW5kZXggKyAke29mZnNldH0pICsgMC41KSAvICR7dE51bVJ9LjApO1xuICAgICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICBpZiAodE51bVIgPT09IDEpIHtcbiAgICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCBpbmRleCkge1xuICAgICAgICB2ZWMyIHV2ID0gdmVjMigoZmxvYXQoaW5kZXggKyAke29mZnNldH0pICsgMC41KSAvIGZsb2F0KCR7XG4gICAgICAgICAgdGV4TmFtZX1UZXhTaGFwZVsxXSksIDAuNSk7XG4gICAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICAgIH1cblxuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgaW5kZXgpIHtcbiAgICAgICAgdmVjMiB1diA9IHZlYzIoKGZsb2F0KGluZGV4ICsgJHtvZmZzZXR9KSArIDAuNSkgLyAke3ROdW1DfS4wLCAwLjUpO1xuICAgICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuXG4gIGlmIChlbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgcmV0dXJuIGBcbiAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgaW5kZXgpIHtcbiAgICAgIHZlYzIgdXYgPSB1dkZyb21GbGF0KCR7dGV4TmFtZX1UZXhTaGFwZVswXSwgJHtcbiAgICAgICAgdGV4TmFtZX1UZXhTaGFwZVsxXSwgaW5kZXggKyAke29mZnNldH0pO1xuICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbiAgfVxuXG4gIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IGluZGV4KSB7XG4gICAgICB2ZWMyIHV2ID0gdXZGcm9tRmxhdCgke3ROdW1SfSwgJHt0TnVtQ30sIGluZGV4ICsgJHtvZmZzZXR9KTtcbiAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldFBhY2tlZFNhbXBsZXIyRChcbiAgICBpbnB1dEluZm86IElucHV0SW5mbywgZW5hYmxlU2hhcGVVbmlmb3JtczogYm9vbGVhbik6IHN0cmluZyB7XG4gIGNvbnN0IHNoYXBlID0gaW5wdXRJbmZvLnNoYXBlSW5mby5sb2dpY2FsU2hhcGU7XG4gIGNvbnN0IHRleE5hbWUgPSBpbnB1dEluZm8ubmFtZTtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0ZXhTaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8udGV4U2hhcGU7XG5cbiAgY29uc3QgdGV4TnVtUiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0ZXhOdW1DID0gdGV4U2hhcGVbMV07XG4gIGNvbnN0IGdsc2wgPSBnZXRHbHNsRGlmZmVyZW5jZXMoKTtcbiAgaWYgKHRleFNoYXBlICE9IG51bGwgJiYgdXRpbC5hcnJheXNFcXVhbChzaGFwZSwgdGV4U2hhcGUpKSB7XG4gICAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICAgIHJldHVybiBgXG4gICAgICB2ZWM0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wpIHtcbiAgICAgICAgdmVjMiB1diA9ICh2ZWMyKGNvbCwgcm93KSArIGhhbGZDUikgLyB2ZWMyKCR7dGV4TmFtZX1UZXhTaGFwZVsxXSwgJHtcbiAgICAgICAgICB0ZXhOYW1lfVRleFNoYXBlWzBdKTtcblxuICAgICAgICByZXR1cm4gJHtnbHNsLnRleHR1cmUyRH0oJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gICAgfVxuICAgIHJldHVybiBgXG4gICAgICB2ZWM0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wpIHtcbiAgICAgICAgdmVjMiB1diA9ICh2ZWMyKGNvbCwgcm93KSArIGhhbGZDUikgLyB2ZWMyKCR7dGV4TnVtQ30uMCwgJHt0ZXhOdW1SfS4wKTtcblxuICAgICAgICByZXR1cm4gJHtnbHNsLnRleHR1cmUyRH0oJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgIHJldHVybiBgXG4gICAgdmVjNCAke2Z1bmNOYW1lfShpbnQgcm93LCBpbnQgY29sKSB7XG4gICAgICBpdmVjMiBwYWNrZWRUZXhTaGFwZSA9IGl2ZWMyKGNlaWwoZmxvYXQoJHtcbiAgICAgICAgdGV4TmFtZX1UZXhTaGFwZVswXSkgLyAyLjApLCBjZWlsKGZsb2F0KCR7dGV4TmFtZX1UZXhTaGFwZVsxXSkgLyAyLjApKTtcbiAgICAgIGludCB2YWx1ZXNQZXJSb3cgPSBpbnQoY2VpbChmbG9hdCgke3RleE5hbWV9U2hhcGVbMV0pIC8gMi4wKSk7XG4gICAgICB2ZWMyIHV2ID0gcGFja2VkVVZmcm9tMkQodmFsdWVzUGVyUm93LCBwYWNrZWRUZXhTaGFwZVswXSwgcGFja2VkVGV4U2hhcGVbMV0sIHJvdywgY29sKTtcbiAgICAgIHJldHVybiAke2dsc2wudGV4dHVyZTJEfSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xuICB9XG4gIGNvbnN0IHBhY2tlZFRleFNoYXBlID1cbiAgICAgIFtNYXRoLmNlaWwodGV4U2hhcGVbMF0gLyAyKSwgTWF0aC5jZWlsKHRleFNoYXBlWzFdIC8gMildO1xuICBjb25zdCB2YWx1ZXNQZXJSb3cgPSBNYXRoLmNlaWwoc2hhcGVbMV0gLyAyKTtcblxuICByZXR1cm4gYFxuICAgIHZlYzQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCkge1xuICAgICAgdmVjMiB1diA9IHBhY2tlZFVWZnJvbTJEKCR7dmFsdWVzUGVyUm93fSwgJHtwYWNrZWRUZXhTaGFwZVswXX0sICR7XG4gICAgICBwYWNrZWRUZXhTaGFwZVsxXX0sIHJvdywgY29sKTtcbiAgICAgIHJldHVybiAke2dsc2wudGV4dHVyZTJEfSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyMkQoXG4gICAgaW5wdXRJbmZvOiBJbnB1dEluZm8sIGVuYWJsZVNoYXBlVW5pZm9ybXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBjb25zdCBzaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlO1xuICBjb25zdCB0ZXhOYW1lID0gaW5wdXRJbmZvLm5hbWU7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKTtcbiAgY29uc3QgdGV4U2hhcGUgPSBpbnB1dEluZm8uc2hhcGVJbmZvLnRleFNoYXBlO1xuXG4gIGlmICh0ZXhTaGFwZSAhPSBudWxsICYmIHV0aWwuYXJyYXlzRXF1YWwoc2hhcGUsIHRleFNoYXBlKSkge1xuICAgIGlmIChlbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCkge1xuICAgICAgICB2ZWMyIHV2ID0gKHZlYzIoY29sLCByb3cpICsgaGFsZkNSKSAvIHZlYzIoJHt0ZXhOYW1lfVRleFNoYXBlWzFdLCAke1xuICAgICAgICAgIHRleE5hbWV9VGV4U2hhcGVbMF0pO1xuICAgICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgICB9XG5cbiAgICBjb25zdCB0ZXhOdW1SID0gdGV4U2hhcGVbMF07XG4gICAgY29uc3QgdGV4TnVtQyA9IHRleFNoYXBlWzFdO1xuICAgIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCkge1xuICAgICAgdmVjMiB1diA9ICh2ZWMyKGNvbCwgcm93KSArIGhhbGZDUikgLyB2ZWMyKCR7dGV4TnVtQ30uMCwgJHt0ZXhOdW1SfS4wKTtcbiAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICB9XG4gIGA7XG4gIH1cblxuICBjb25zdCB7bmV3U2hhcGUsIGtlcHREaW1zfSA9IHV0aWwuc3F1ZWV6ZVNoYXBlKHNoYXBlKTtcbiAgY29uc3Qgc3F1ZWV6ZWRTaGFwZSA9IG5ld1NoYXBlO1xuICBpZiAoc3F1ZWV6ZWRTaGFwZS5sZW5ndGggPCBzaGFwZS5sZW5ndGgpIHtcbiAgICBjb25zdCBuZXdJbnB1dEluZm8gPSBzcXVlZXplSW5wdXRJbmZvKGlucHV0SW5mbywgc3F1ZWV6ZWRTaGFwZSk7XG4gICAgY29uc3QgcGFyYW1zID0gWydyb3cnLCAnY29sJ107XG4gICAgcmV0dXJuIGBcbiAgICAgICR7Z2V0U2FtcGxlckZyb21JbkluZm8obmV3SW5wdXRJbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zKX1cbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wpIHtcbiAgICAgICAgcmV0dXJuICR7ZnVuY05hbWV9KCR7Z2V0U3F1ZWV6ZWRQYXJhbXMocGFyYW1zLCBrZXB0RGltcyl9KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG5cbiAgaWYgKGlucHV0SW5mby5zaGFwZUluZm8uaXNVbmlmb3JtKSB7XG4gICAgLy8gVW5pZm9ybSBhcnJheXMgd2lsbCBiZSBsZXNzIHRoYW4gNjU1MDUgKG5vIHJpc2sgb2YgZmxvYXQxNiBvdmVyZmxvdykuXG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wpIHtcbiAgICAgICAgaW50IGluZGV4ID0gcm91bmQoZG90KHZlYzIocm93LCBjb2wpLCB2ZWMyKCR7c2hhcGVbMV19LCAxKSkpO1xuICAgICAgICAke2dldFVuaWZvcm1TYW1wbGVyKGlucHV0SW5mbyl9XG4gICAgICB9XG4gICAgYDtcbiAgfVxuXG4gIGNvbnN0IHRleE51bVIgPSB0ZXhTaGFwZVswXTtcbiAgY29uc3QgdGV4TnVtQyA9IHRleFNoYXBlWzFdO1xuICBjb25zdCBvZmZzZXQgPSBnZXRGbGF0T2Zmc2V0VW5pZm9ybU5hbWUodGV4TmFtZSk7XG4gIGlmICh0ZXhOdW1DID09PSAxKSB7XG4gICAgLy8gaW5kZXggaXMgdXNlZCBkaXJlY3RseSBhcyBwaHlzaWNhbCAobm8gcmlzayBvZiBmbG9hdDE2IG92ZXJmbG93KS5cbiAgICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wpIHtcbiAgICAgICAgZmxvYXQgaW5kZXggPSBkb3QodmVjMyhyb3csIGNvbCwgJHtvZmZzZXR9KSwgdmVjMygke1xuICAgICAgICAgIHRleE5hbWV9U2hhcGVbMV0sIDEsIDEpKTtcbiAgICAgICAgdmVjMiB1diA9IHZlYzIoMC41LCAoaW5kZXggKyAwLjUpIC8gZmxvYXQoJHt0ZXhOYW1lfVRleFNoYXBlWzBdKSk7XG4gICAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICAgIH1cbiAgICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wpIHtcbiAgICAgIGZsb2F0IGluZGV4ID0gZG90KHZlYzMocm93LCBjb2wsICR7b2Zmc2V0fSksIHZlYzMoJHtzaGFwZVsxXX0sIDEsIDEpKTtcbiAgICAgIHZlYzIgdXYgPSB2ZWMyKDAuNSwgKGluZGV4ICsgMC41KSAvICR7dGV4TnVtUn0uMCk7XG4gICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xuICB9XG4gIGlmICh0ZXhOdW1SID09PSAxKSB7XG4gICAgLy8gaW5kZXggaXMgdXNlZCBkaXJlY3RseSBhcyBwaHlzaWNhbCAobm8gcmlzayBvZiBmbG9hdDE2IG92ZXJmbG93KS5cbiAgICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wpIHtcbiAgICAgICAgZmxvYXQgaW5kZXggPSBkb3QodmVjMyhyb3csIGNvbCwgJHtvZmZzZXR9KSwgdmVjMygke1xuICAgICAgICAgIHRleE5hbWV9U2hhcGVbMV0sIDEsIDEpKTtcbiAgICAgICAgdmVjMiB1diA9IHZlYzIoKGluZGV4ICsgMC41KSAvIGZsb2F0KCR7dGV4TmFtZX1UZXhTaGFwZVsxXSksIDAuNSk7XG4gICAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICAgIH1cbiAgICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wpIHtcbiAgICAgIGZsb2F0IGluZGV4ID0gZG90KHZlYzMocm93LCBjb2wsICR7b2Zmc2V0fSksIHZlYzMoJHtzaGFwZVsxXX0sIDEsIDEpKTtcbiAgICAgIHZlYzIgdXYgPSB2ZWMyKChpbmRleCArIDAuNSkgLyAke3RleE51bUN9LjAsIDAuNSk7XG4gICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xuICB9XG5cbiAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCkge1xuICAgICAgICAvLyBFeHBsaWNpdGx5IHVzZSBpbnRlZ2VyIG9wZXJhdGlvbnMgYXMgZG90KCkgb25seSB3b3JrcyBvbiBmbG9hdHMuXG4gICAgICAgIGludCBpbmRleCA9IHJvdyAqICR7dGV4TmFtZX1TaGFwZVsxXSArIGNvbCArICR7b2Zmc2V0fTtcbiAgICAgICAgdmVjMiB1diA9IHV2RnJvbUZsYXQoJHt0ZXhOYW1lfVRleFNoYXBlWzBdLCAke1xuICAgICAgICB0ZXhOYW1lfVRleFNoYXBlWzFdLCBpbmRleCk7XG4gICAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIHJldHVybiBgXG4gIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wpIHtcbiAgICAvLyBFeHBsaWNpdGx5IHVzZSBpbnRlZ2VyIG9wZXJhdGlvbnMgYXMgZG90KCkgb25seSB3b3JrcyBvbiBmbG9hdHMuXG4gICAgaW50IGluZGV4ID0gcm93ICogJHtzaGFwZVsxXX0gKyBjb2wgKyAke29mZnNldH07XG4gICAgdmVjMiB1diA9IHV2RnJvbUZsYXQoJHt0ZXhOdW1SfSwgJHt0ZXhOdW1DfSwgaW5kZXgpO1xuICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgfVxuYDtcbn1cblxuZnVuY3Rpb24gZ2V0UGFja2VkU2FtcGxlcjNEKFxuICAgIGlucHV0SW5mbzogSW5wdXRJbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zOiBib29sZWFuKTogc3RyaW5nIHtcbiAgY29uc3Qgc2hhcGUgPSBpbnB1dEluZm8uc2hhcGVJbmZvLmxvZ2ljYWxTaGFwZTtcbiAgY29uc3QgdGV4TmFtZSA9IGlucHV0SW5mby5uYW1lO1xuICBjb25zdCBmdW5jTmFtZSA9ICdnZXQnICsgdGV4TmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSArIHRleE5hbWUuc2xpY2UoMSk7XG4gIGNvbnN0IHRleFNoYXBlID0gaW5wdXRJbmZvLnNoYXBlSW5mby50ZXhTaGFwZTtcbiAgY29uc3QgcGFja2VkVGV4U2hhcGUgPVxuICAgICAgW01hdGguY2VpbCh0ZXhTaGFwZVswXSAvIDIpLCBNYXRoLmNlaWwodGV4U2hhcGVbMV0gLyAyKV07XG5cbiAgaWYgKHNoYXBlWzBdID09PSAxKSB7XG4gICAgY29uc3Qgc3F1ZWV6ZWRTaGFwZSA9IHNoYXBlLnNsaWNlKDEpO1xuICAgIGNvbnN0IGtlcHREaW1zID0gWzEsIDJdO1xuICAgIGNvbnN0IG5ld0lucHV0SW5mbyA9IHNxdWVlemVJbnB1dEluZm8oaW5wdXRJbmZvLCBzcXVlZXplZFNoYXBlKTtcbiAgICBjb25zdCBwYXJhbXMgPSBbJ2InLCAncm93JywgJ2NvbCddO1xuICAgIHJldHVybiBgXG4gICAgICAgICR7Z2V0UGFja2VkU2FtcGxlckZyb21JbkluZm8obmV3SW5wdXRJbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zKX1cbiAgICAgICAgdmVjNCAke2Z1bmNOYW1lfShpbnQgYiwgaW50IHJvdywgaW50IGNvbCkge1xuICAgICAgICAgIHJldHVybiAke2Z1bmNOYW1lfSgke2dldFNxdWVlemVkUGFyYW1zKHBhcmFtcywga2VwdERpbXMpfSk7XG4gICAgICAgIH1cbiAgICAgIGA7XG4gIH1cblxuICBjb25zdCBnbHNsID0gZ2V0R2xzbERpZmZlcmVuY2VzKCk7XG4gIGlmIChlbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgcmV0dXJuIGBcbiAgICB2ZWM0ICR7ZnVuY05hbWV9KGludCBiLCBpbnQgcm93LCBpbnQgY29sKSB7XG4gICAgICBpdmVjMiBwYWNrZWRUZXhTaGFwZSA9IGl2ZWMyKGNlaWwoZmxvYXQoJHtcbiAgICAgICAgdGV4TmFtZX1UZXhTaGFwZVswXSkgLyAyLjApLCBjZWlsKGZsb2F0KCR7dGV4TmFtZX1UZXhTaGFwZVsxXSkgLyAyLjApKTtcbiAgICAgIGludCB2YWx1ZXNQZXJSb3cgPSBpbnQoY2VpbChmbG9hdCgke3RleE5hbWV9U2hhcGVbMl0pIC8gMi4wKSk7XG4gICAgICBpbnQgdGV4ZWxzSW5CYXRjaCA9IHZhbHVlc1BlclJvdyAqIGludChjZWlsKGZsb2F0KCR7XG4gICAgICAgIHRleE5hbWV9U2hhcGVbMV0pIC8gMi4wKSk7XG4gICAgICB2ZWMyIHV2ID0gcGFja2VkVVZmcm9tM0QoXG4gICAgICAgIHBhY2tlZFRleFNoYXBlWzBdLCBwYWNrZWRUZXhTaGFwZVsxXSwgdGV4ZWxzSW5CYXRjaCwgdmFsdWVzUGVyUm93LCBiLCByb3csIGNvbCk7XG4gICAgICByZXR1cm4gJHtnbHNsLnRleHR1cmUyRH0oJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbiAgfVxuXG4gIGNvbnN0IHRleE51bVIgPSBwYWNrZWRUZXhTaGFwZVswXTtcbiAgY29uc3QgdGV4TnVtQyA9IHBhY2tlZFRleFNoYXBlWzFdO1xuXG4gIGNvbnN0IHZhbHVlc1BlclJvdyA9IE1hdGguY2VpbChzaGFwZVsyXSAvIDIpO1xuICBjb25zdCB0ZXhlbHNJbkJhdGNoID0gdmFsdWVzUGVyUm93ICogTWF0aC5jZWlsKHNoYXBlWzFdIC8gMik7XG5cbiAgcmV0dXJuIGBcbiAgICB2ZWM0ICR7ZnVuY05hbWV9KGludCBiLCBpbnQgcm93LCBpbnQgY29sKSB7XG4gICAgICB2ZWMyIHV2ID0gcGFja2VkVVZmcm9tM0QoXG4gICAgICAgICR7dGV4TnVtUn0sICR7dGV4TnVtQ30sICR7dGV4ZWxzSW5CYXRjaH0sICR7dmFsdWVzUGVyUm93fSwgYiwgcm93LCBjb2wpO1xuICAgICAgcmV0dXJuICR7Z2xzbC50ZXh0dXJlMkR9KCR7dGV4TmFtZX0sIHV2KTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldFNhbXBsZXIzRChcbiAgICBpbnB1dEluZm86IElucHV0SW5mbywgZW5hYmxlU2hhcGVVbmlmb3JtczogYm9vbGVhbik6IHN0cmluZyB7XG4gIGNvbnN0IHNoYXBlID0gaW5wdXRJbmZvLnNoYXBlSW5mby5sb2dpY2FsU2hhcGU7XG4gIGNvbnN0IHRleE5hbWUgPSBpbnB1dEluZm8ubmFtZTtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCBzdHJpZGUwID0gc2hhcGVbMV0gKiBzaGFwZVsyXTtcbiAgY29uc3Qgc3RyaWRlMSA9IHNoYXBlWzJdO1xuXG4gIGNvbnN0IHtuZXdTaGFwZSwga2VwdERpbXN9ID0gdXRpbC5zcXVlZXplU2hhcGUoc2hhcGUpO1xuICBjb25zdCBzcXVlZXplZFNoYXBlID0gbmV3U2hhcGU7XG4gIGlmIChzcXVlZXplZFNoYXBlLmxlbmd0aCA8IHNoYXBlLmxlbmd0aCkge1xuICAgIGNvbnN0IG5ld0lucHV0SW5mbyA9IHNxdWVlemVJbnB1dEluZm8oaW5wdXRJbmZvLCBzcXVlZXplZFNoYXBlKTtcbiAgICBjb25zdCBwYXJhbXMgPSBbJ3JvdycsICdjb2wnLCAnZGVwdGgnXTtcbiAgICByZXR1cm4gYFxuICAgICAgICAke2dldFNhbXBsZXJGcm9tSW5JbmZvKG5ld0lucHV0SW5mbywgZW5hYmxlU2hhcGVVbmlmb3Jtcyl9XG4gICAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCkge1xuICAgICAgICAgIHJldHVybiAke2Z1bmNOYW1lfSgke2dldFNxdWVlemVkUGFyYW1zKHBhcmFtcywga2VwdERpbXMpfSk7XG4gICAgICAgIH1cbiAgICAgIGA7XG4gIH1cblxuICBpZiAoaW5wdXRJbmZvLnNoYXBlSW5mby5pc1VuaWZvcm0pIHtcbiAgICAvLyBVbmlmb3JtIGFycmF5cyB3aWxsIGJlIGxlc3MgdGhhbiA2NTUwNSAobm8gcmlzayBvZiBmbG9hdDE2IG92ZXJmbG93KS5cbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCwgaW50IGRlcHRoKSB7XG4gICAgICAgIGludCBpbmRleCA9IHJvdW5kKGRvdCh2ZWMzKHJvdywgY29sLCBkZXB0aCksXG4gICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzMoJHtzdHJpZGUwfSwgJHtzdHJpZGUxfSwgMSkpKTtcbiAgICAgICAgJHtnZXRVbmlmb3JtU2FtcGxlcihpbnB1dEluZm8pfVxuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICBjb25zdCB0ZXhTaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8udGV4U2hhcGU7XG4gIGNvbnN0IHRleE51bVIgPSB0ZXhTaGFwZVswXTtcbiAgY29uc3QgdGV4TnVtQyA9IHRleFNoYXBlWzFdO1xuICBjb25zdCBmbGF0T2Zmc2V0ID0gaW5wdXRJbmZvLnNoYXBlSW5mby5mbGF0T2Zmc2V0O1xuICBpZiAodGV4TnVtQyA9PT0gc3RyaWRlMCAmJiBmbGF0T2Zmc2V0ID09IG51bGwpIHtcbiAgICAvLyB0ZXhDIGlzIHVzZWQgZGlyZWN0bHkgYXMgcGh5c2ljYWwgKG5vIHJpc2sgb2YgZmxvYXQxNiBvdmVyZmxvdykuXG4gICAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgcm93LCBpbnQgY29sLCBpbnQgZGVwdGgpIHtcbiAgICAgICAgaW50IHN0cmlkZTEgPSAke3RleE5hbWV9U2hhcGVbMl07XG4gICAgICAgIGZsb2F0IHRleFIgPSBmbG9hdChyb3cpO1xuICAgICAgICBmbG9hdCB0ZXhDID0gZG90KHZlYzIoY29sLCBkZXB0aCksIHZlYzIoc3RyaWRlMSwgMSkpO1xuICAgICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC9cbiAgICAgICAgICAgICAgICAgICB2ZWMyKCR7dGV4TmFtZX1UZXhTaGFwZVsxXSwgJHt0ZXhOYW1lfVRleFNoYXBlWzBdKTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gICAgfVxuICAgIHJldHVybiBgXG4gICAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCkge1xuICAgICAgICAgIGZsb2F0IHRleFIgPSBmbG9hdChyb3cpO1xuICAgICAgICAgIGZsb2F0IHRleEMgPSBkb3QodmVjMihjb2wsIGRlcHRoKSwgdmVjMigke3N0cmlkZTF9LCAxKSk7XG4gICAgICAgICAgdmVjMiB1diA9ICh2ZWMyKHRleEMsIHRleFIpICsgaGFsZkNSKSAvXG4gICAgICAgICAgICAgICAgICAgICB2ZWMyKCR7dGV4TnVtQ30uMCwgJHt0ZXhOdW1SfS4wKTtcbiAgICAgICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgICAgIH1cbiAgICAgIGA7XG4gIH1cblxuICBpZiAodGV4TnVtQyA9PT0gc3RyaWRlMSAmJiBmbGF0T2Zmc2V0ID09IG51bGwpIHtcbiAgICAvLyB0ZXhSIGlzIHVzZWQgZGlyZWN0bHkgYXMgcGh5c2ljYWwgKG5vIHJpc2sgb2YgZmxvYXQxNiBvdmVyZmxvdykuXG4gICAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgcm93LCBpbnQgY29sLCBpbnQgZGVwdGgpIHtcbiAgICAgICAgZmxvYXQgdGV4UiA9IGRvdCh2ZWMyKHJvdywgY29sKSwgdmVjMigke3RleE5hbWV9U2hhcGVbMV0sIDEpKTtcbiAgICAgICAgZmxvYXQgdGV4QyA9IGZsb2F0KGRlcHRoKTtcbiAgICAgICAgdmVjMiB1diA9ICh2ZWMyKHRleEMsIHRleFIpICsgaGFsZkNSKSAvIHZlYzIoJHt0ZXhOYW1lfVRleFNoYXBlWzFdLCAke1xuICAgICAgICAgIHRleE5hbWV9VGV4U2hhcGVbMF0pO1xuICAgICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgICB9XG4gICAgcmV0dXJuIGBcbiAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgcm93LCBpbnQgY29sLCBpbnQgZGVwdGgpIHtcbiAgICAgIGZsb2F0IHRleFIgPSBkb3QodmVjMihyb3csIGNvbCksIHZlYzIoJHtzaGFwZVsxXX0sIDEpKTtcbiAgICAgIGZsb2F0IHRleEMgPSBmbG9hdChkZXB0aCk7XG4gICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMigke3RleE51bUN9LjAsICR7dGV4TnVtUn0uMCk7XG4gICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xuICB9XG5cbiAgY29uc3Qgb2Zmc2V0ID0gZ2V0RmxhdE9mZnNldFVuaWZvcm1OYW1lKHRleE5hbWUpO1xuICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCwgaW50IGRlcHRoKSB7XG4gICAgICAvLyBFeHBsaWNpdGx5IHVzZSBpbnRlZ2VyIG9wZXJhdGlvbnMgYXMgZG90KCkgb25seSB3b3JrcyBvbiBmbG9hdHMuXG4gICAgICBpbnQgc3RyaWRlMCA9ICR7dGV4TmFtZX1TaGFwZVsxXSAqICR7dGV4TmFtZX1TaGFwZVsyXTtcbiAgICAgIGludCBzdHJpZGUxID0gJHt0ZXhOYW1lfVNoYXBlWzJdO1xuICAgICAgaW50IGluZGV4ID0gcm93ICogJHtzdHJpZGUwfSArIGNvbCAqICR7c3RyaWRlMX0gKyBkZXB0aCArICR7b2Zmc2V0fTtcbiAgICAgIHZlYzIgdXYgPSB1dkZyb21GbGF0KCR7dGV4TmFtZX1UZXhTaGFwZVswXSwgJHt0ZXhOYW1lfVRleFNoYXBlWzFdLCBpbmRleCk7XG4gICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCkge1xuICAgICAgICAvLyBFeHBsaWNpdGx5IHVzZSBpbnRlZ2VyIG9wZXJhdGlvbnMgYXMgZG90KCkgb25seSB3b3JrcyBvbiBmbG9hdHMuXG4gICAgICAgIGludCBpbmRleCA9IHJvdyAqICR7c3RyaWRlMH0gKyBjb2wgKiAke3N0cmlkZTF9ICsgZGVwdGggKyAke29mZnNldH07XG4gICAgICAgIHZlYzIgdXYgPSB1dkZyb21GbGF0KCR7dGV4TnVtUn0sICR7dGV4TnVtQ30sIGluZGV4KTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRQYWNrZWRTYW1wbGVyTkQoXG4gICAgaW5wdXRJbmZvOiBJbnB1dEluZm8sIGVuYWJsZVNoYXBlVW5pZm9ybXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBjb25zdCB0ZXhOYW1lID0gaW5wdXRJbmZvLm5hbWU7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKTtcbiAgY29uc3QgZ2xzbCA9IGdldEdsc2xEaWZmZXJlbmNlcygpO1xuICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgIC8vIFRPRE86IHN1cHBvcnQgNWQgYW5kIDZkXG4gICAgcmV0dXJuIGBcbiAgICB2ZWM0ICR7ZnVuY05hbWV9KGludCBiMiwgaW50IGIsIGludCByb3csIGludCBjb2wpIHtcbiAgICAgIGludCB2YWx1ZXNQZXJSb3cgPSBpbnQoY2VpbChmbG9hdCgke3RleE5hbWV9U2hhcGVbM10pIC8gMi4wKSk7XG4gICAgICBpbnQgdGV4ZWxzSW5CYXRjaCA9IHZhbHVlc1BlclJvdyAqIGludChjZWlsKGZsb2F0KCR7XG4gICAgICAgIHRleE5hbWV9U2hhcGVbMl0pIC8gMi4wKSk7XG4gICAgICBpbnQgaW5kZXggPSBiICogdGV4ZWxzSW5CYXRjaCArIChyb3cgLyAyKSAqIHZhbHVlc1BlclJvdyArIChjb2wgLyAyKTtcbiAgICAgIHRleGVsc0luQmF0Y2ggKj0gJHt0ZXhOYW1lfVNoYXBlWzFdO1xuICAgICAgaW5kZXggPSBiMiAqIHRleGVsc0luQmF0Y2ggKyBpbmRleDtcbiAgICAgIGl2ZWMyIHBhY2tlZFRleFNoYXBlID0gaXZlYzIoY2VpbChmbG9hdCgke1xuICAgICAgICB0ZXhOYW1lfVRleFNoYXBlWzBdKSAvIDIuMCksIGNlaWwoZmxvYXQoJHt0ZXhOYW1lfVRleFNoYXBlWzFdKSAvIDIuMCkpO1xuICAgICAgaW50IHRleFIgPSBpbmRleCAvIHBhY2tlZFRleFNoYXBlWzFdO1xuICAgICAgaW50IHRleEMgPSBpbmRleCAtIHRleFIgKiBwYWNrZWRUZXhTaGFwZVsxXTtcbiAgICAgIHZlYzIgdXYgPSAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgLyB2ZWMyKHBhY2tlZFRleFNoYXBlWzFdLCBwYWNrZWRUZXhTaGFwZVswXSk7IHJldHVybiAke1xuICAgICAgICBnbHNsLnRleHR1cmUyRH0oJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbiAgfVxuICBjb25zdCBzaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlO1xuICBjb25zdCByYW5rID0gc2hhcGUubGVuZ3RoO1xuICBjb25zdCB0ZXhTaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8udGV4U2hhcGU7XG4gIGNvbnN0IHBhY2tlZFRleFNoYXBlID1cbiAgICAgIFtNYXRoLmNlaWwodGV4U2hhcGVbMF0gLyAyKSwgTWF0aC5jZWlsKHRleFNoYXBlWzFdIC8gMildO1xuICBjb25zdCB0ZXhOdW1SID0gcGFja2VkVGV4U2hhcGVbMF07XG4gIGNvbnN0IHRleE51bUMgPSBwYWNrZWRUZXhTaGFwZVsxXTtcblxuICBjb25zdCB2YWx1ZXNQZXJSb3cgPSBNYXRoLmNlaWwoc2hhcGVbcmFuayAtIDFdIC8gMik7XG4gIGxldCB0ZXhlbHNJbkJhdGNoID0gdmFsdWVzUGVyUm93ICogTWF0aC5jZWlsKHNoYXBlW3JhbmsgLSAyXSAvIDIpO1xuICBsZXQgcGFyYW1zID0gYGludCBiLCBpbnQgcm93LCBpbnQgY29sYDtcbiAgbGV0IGluZGV4ID0gYGIgKiAke3RleGVsc0luQmF0Y2h9ICsgKHJvdyAvIDIpICogJHt2YWx1ZXNQZXJSb3d9ICsgKGNvbCAvIDIpYDtcbiAgZm9yIChsZXQgYiA9IDI7IGIgPCByYW5rIC0gMTsgYisrKSB7XG4gICAgcGFyYW1zID0gYGludCBiJHtifSwgYCArIHBhcmFtcztcbiAgICB0ZXhlbHNJbkJhdGNoICo9IHNoYXBlW3JhbmsgLSBiIC0gMV07XG4gICAgaW5kZXggPSBgYiR7Yn0gKiAke3RleGVsc0luQmF0Y2h9ICsgYCArIGluZGV4O1xuICB9XG4gIHJldHVybiBgXG4gICAgdmVjNCAke2Z1bmNOYW1lfSgke3BhcmFtc30pIHtcbiAgICAgIGludCBpbmRleCA9ICR7aW5kZXh9O1xuICAgICAgaW50IHRleFIgPSBpbmRleCAvICR7dGV4TnVtQ307XG4gICAgICBpbnQgdGV4QyA9IGluZGV4IC0gdGV4UiAqICR7dGV4TnVtQ307XG4gICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMigke3RleE51bUN9LCAke3RleE51bVJ9KTtcbiAgICAgIHJldHVybiAke2dsc2wudGV4dHVyZTJEfSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyNEQoXG4gICAgaW5wdXRJbmZvOiBJbnB1dEluZm8sIGVuYWJsZVNoYXBlVW5pZm9ybXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBjb25zdCBzaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlO1xuICBjb25zdCB0ZXhOYW1lID0gaW5wdXRJbmZvLm5hbWU7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKTtcbiAgY29uc3Qgc3RyaWRlMiA9IHNoYXBlWzNdO1xuICBjb25zdCBzdHJpZGUxID0gc2hhcGVbMl0gKiBzdHJpZGUyO1xuICBjb25zdCBzdHJpZGUwID0gc2hhcGVbMV0gKiBzdHJpZGUxO1xuXG4gIGNvbnN0IHtuZXdTaGFwZSwga2VwdERpbXN9ID0gdXRpbC5zcXVlZXplU2hhcGUoc2hhcGUpO1xuICBpZiAobmV3U2hhcGUubGVuZ3RoIDwgc2hhcGUubGVuZ3RoKSB7XG4gICAgY29uc3QgbmV3SW5wdXRJbmZvID0gc3F1ZWV6ZUlucHV0SW5mbyhpbnB1dEluZm8sIG5ld1NoYXBlKTtcbiAgICBjb25zdCBwYXJhbXMgPSBbJ3JvdycsICdjb2wnLCAnZGVwdGgnLCAnZGVwdGgyJ107XG4gICAgcmV0dXJuIGBcbiAgICAgICR7Z2V0U2FtcGxlckZyb21JbkluZm8obmV3SW5wdXRJbmZvLCBlbmFibGVTaGFwZVVuaWZvcm1zKX1cbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCwgaW50IGRlcHRoMikge1xuICAgICAgICByZXR1cm4gJHtmdW5jTmFtZX0oJHtnZXRTcXVlZXplZFBhcmFtcyhwYXJhbXMsIGtlcHREaW1zKX0pO1xuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICBpZiAoaW5wdXRJbmZvLnNoYXBlSW5mby5pc1VuaWZvcm0pIHtcbiAgICAvLyBVbmlmb3JtIGFycmF5cyB3aWxsIGJlIGxlc3MgdGhhbiA2NTUwNSAobm8gcmlzayBvZiBmbG9hdDE2IG92ZXJmbG93KS5cbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCwgaW50IGRlcHRoLCBpbnQgZGVwdGgyKSB7XG4gICAgICAgIGludCBpbmRleCA9IHJvdW5kKGRvdCh2ZWM0KHJvdywgY29sLCBkZXB0aCwgZGVwdGgyKSxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgdmVjNCgke3N0cmlkZTB9LCAke3N0cmlkZTF9LCAke3N0cmlkZTJ9LCAxKSkpO1xuICAgICAgICAke2dldFVuaWZvcm1TYW1wbGVyKGlucHV0SW5mbyl9XG4gICAgICB9XG4gICAgYDtcbiAgfVxuXG4gIGNvbnN0IGZsYXRPZmZzZXQgPSBpbnB1dEluZm8uc2hhcGVJbmZvLmZsYXRPZmZzZXQ7XG4gIGNvbnN0IHRleFNoYXBlID0gaW5wdXRJbmZvLnNoYXBlSW5mby50ZXhTaGFwZTtcbiAgY29uc3QgdGV4TnVtUiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0ZXhOdW1DID0gdGV4U2hhcGVbMV07XG5cbiAgY29uc3Qgc3RyaWRlMlN0ciA9IGBpbnQgc3RyaWRlMiA9ICR7dGV4TmFtZX1TaGFwZVszXTtgO1xuICBjb25zdCBzdHJpZGUxU3RyID0gYGludCBzdHJpZGUxID0gJHt0ZXhOYW1lfVNoYXBlWzJdICogc3RyaWRlMjtgO1xuICBjb25zdCBzdHJpZGUwU3RyID0gYGludCBzdHJpZGUwID0gJHt0ZXhOYW1lfVNoYXBlWzFdICogc3RyaWRlMTtgO1xuICBpZiAodGV4TnVtQyA9PT0gc3RyaWRlMCAmJiBmbGF0T2Zmc2V0ID09IG51bGwpIHtcbiAgICAvLyB0ZXhDIGlzIHVzZWQgZGlyZWN0bHkgYXMgcGh5c2ljYWwgKG5vIHJpc2sgb2YgZmxvYXQxNiBvdmVyZmxvdykuXG4gICAgaWYgKGVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgcm93LCBpbnQgY29sLCBpbnQgZGVwdGgsIGludCBkZXB0aDIpIHtcbiAgICAgICAgJHtzdHJpZGUyU3RyfVxuICAgICAgICAke3N0cmlkZTFTdHJ9XG4gICAgICAgIGZsb2F0IHRleFIgPSBmbG9hdChyb3cpO1xuICAgICAgICBmbG9hdCB0ZXhDID1cbiAgICAgICAgICAgIGRvdCh2ZWMzKGNvbCwgZGVwdGgsIGRlcHRoMiksXG4gICAgICAgICAgICAgICAgdmVjMyhzdHJpZGUxLCBzdHJpZGUyLCAxKSk7XG4gICAgICAgIHZlYzIgdXYgPSAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgL1xuICAgICAgICAgICAgICAgICAgIHZlYzIoJHt0ZXhOYW1lfVRleFNoYXBlWzFdLCAke3RleE5hbWV9VGV4U2hhcGVbMF0pO1xuICAgICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgICB9XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCwgaW50IGRlcHRoMikge1xuICAgICAgICBmbG9hdCB0ZXhSID0gZmxvYXQocm93KTtcbiAgICAgICAgZmxvYXQgdGV4QyA9XG4gICAgICAgICAgICBkb3QodmVjMyhjb2wsIGRlcHRoLCBkZXB0aDIpLFxuICAgICAgICAgICAgICAgIHZlYzMoJHtzdHJpZGUxfSwgJHtzdHJpZGUyfSwgMSkpO1xuICAgICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC9cbiAgICAgICAgICAgICAgICAgICB2ZWMyKCR7dGV4TnVtQ30uMCwgJHt0ZXhOdW1SfS4wKTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgaWYgKHRleE51bUMgPT09IHN0cmlkZTIgJiYgZmxhdE9mZnNldCA9PSBudWxsKSB7XG4gICAgLy8gdGV4UiBpcyB1c2VkIGRpcmVjdGx5IGFzIHBoeXNpY2FsIChubyByaXNrIG9mIGZsb2F0MTYgb3ZlcmZsb3cpLlxuICAgIGlmIChlbmFibGVTaGFwZVVuaWZvcm1zKSB7XG4gICAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCwgaW50IGRlcHRoLCBpbnQgZGVwdGgyKSB7XG4gICAgICAgIGZsb2F0IHRleFIgPSBkb3QodmVjMyhyb3csIGNvbCwgZGVwdGgpLFxuICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzMoJHt0ZXhOYW1lfVNoYXBlWzFdICogJHt0ZXhOYW1lfVNoYXBlWzJdLCAke1xuICAgICAgICAgIHRleE5hbWV9U2hhcGVbMl0sIDEpKTtcbiAgICAgICAgZmxvYXQgdGV4QyA9IGZsb2F0KGRlcHRoMik7XG4gICAgICAgIHZlYzIgdXYgPSAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgL1xuICAgICAgICAgICAgICAgICAgdmVjMigke3RleE5hbWV9VGV4U2hhcGVbMV0sICR7dGV4TmFtZX1UZXhTaGFwZVswXSk7XG4gICAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICAgIH1cbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCwgaW50IGRlcHRoLCBpbnQgZGVwdGgyKSB7XG4gICAgICAgIGZsb2F0IHRleFIgPSBkb3QodmVjMyhyb3csIGNvbCwgZGVwdGgpLFxuICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzMoJHtzaGFwZVsxXSAqIHNoYXBlWzJdfSwgJHtzaGFwZVsyXX0sIDEpKTtcbiAgICAgICAgZmxvYXQgdGV4QyA9IGZsb2F0KGRlcHRoMik7XG4gICAgICAgIHZlYzIgdXYgPSAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgL1xuICAgICAgICAgICAgICAgICAgdmVjMigke3RleE51bUN9LjAsICR7dGV4TnVtUn0uMCk7XG4gICAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG5cbiAgY29uc3Qgb2Zmc2V0ID0gZ2V0RmxhdE9mZnNldFVuaWZvcm1OYW1lKHRleE5hbWUpO1xuICBpZiAoZW5hYmxlU2hhcGVVbmlmb3Jtcykge1xuICAgIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCwgaW50IGRlcHRoLCBpbnQgZGVwdGgyKSB7XG4gICAgICAvLyBFeHBsaWNpdGx5IHVzZSBpbnRlZ2VyIG9wZXJhdGlvbnMgYXMgZG90KCkgb25seSB3b3JrcyBvbiBmbG9hdHMuXG4gICAgICAke3N0cmlkZTJTdHJ9XG4gICAgICAke3N0cmlkZTFTdHJ9XG4gICAgICAke3N0cmlkZTBTdHJ9XG4gICAgICBpbnQgaW5kZXggPSByb3cgKiBzdHJpZGUwICsgY29sICogc3RyaWRlMSArXG4gICAgICAgICAgZGVwdGggKiBzdHJpZGUyICsgZGVwdGgyO1xuICAgICAgdmVjMiB1diA9IHV2RnJvbUZsYXQoJHt0ZXhOYW1lfVRleFNoYXBlWzBdLCAke1xuICAgICAgICB0ZXhOYW1lfVRleFNoYXBlWzFdLCBpbmRleCArICR7b2Zmc2V0fSk7XG4gICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xuICB9XG4gIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCwgaW50IGRlcHRoLCBpbnQgZGVwdGgyKSB7XG4gICAgICAvLyBFeHBsaWNpdGx5IHVzZSBpbnRlZ2VyIG9wZXJhdGlvbnMgYXMgZG90KCkgb25seSB3b3JrcyBvbiBmbG9hdHMuXG4gICAgICBpbnQgaW5kZXggPSByb3cgKiAke3N0cmlkZTB9ICsgY29sICogJHtzdHJpZGUxfSArXG4gICAgICAgICAgZGVwdGggKiAke3N0cmlkZTJ9ICsgZGVwdGgyO1xuICAgICAgdmVjMiB1diA9IHV2RnJvbUZsYXQoJHt0ZXhOdW1SfSwgJHt0ZXhOdW1DfSwgaW5kZXggKyAke29mZnNldH0pO1xuICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0U2FtcGxlcjVEKGlucHV0SW5mbzogSW5wdXRJbmZvKTogc3RyaW5nIHtcbiAgY29uc3Qgc2hhcGUgPSBpbnB1dEluZm8uc2hhcGVJbmZvLmxvZ2ljYWxTaGFwZTtcbiAgY29uc3QgdGV4TmFtZSA9IGlucHV0SW5mby5uYW1lO1xuICBjb25zdCBmdW5jTmFtZSA9ICdnZXQnICsgdGV4TmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSArIHRleE5hbWUuc2xpY2UoMSk7XG4gIGNvbnN0IHN0cmlkZTMgPSBzaGFwZVs0XTtcbiAgY29uc3Qgc3RyaWRlMiA9IHNoYXBlWzNdICogc3RyaWRlMztcbiAgY29uc3Qgc3RyaWRlMSA9IHNoYXBlWzJdICogc3RyaWRlMjtcbiAgY29uc3Qgc3RyaWRlMCA9IHNoYXBlWzFdICogc3RyaWRlMTtcblxuICBjb25zdCB7bmV3U2hhcGUsIGtlcHREaW1zfSA9IHV0aWwuc3F1ZWV6ZVNoYXBlKHNoYXBlKTtcbiAgaWYgKG5ld1NoYXBlLmxlbmd0aCA8IHNoYXBlLmxlbmd0aCkge1xuICAgIGNvbnN0IG5ld0lucHV0SW5mbyA9IHNxdWVlemVJbnB1dEluZm8oaW5wdXRJbmZvLCBuZXdTaGFwZSk7XG4gICAgY29uc3QgcGFyYW1zID0gWydyb3cnLCAnY29sJywgJ2RlcHRoJywgJ2RlcHRoMicsICdkZXB0aDMnXTtcbiAgICByZXR1cm4gYFxuICAgICAgJHtnZXRTYW1wbGVyRnJvbUluSW5mbyhuZXdJbnB1dEluZm8pfVxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCwgaW50IGRlcHRoLCBpbnQgZGVwdGgyLCBpbnQgZGVwdGgzKSB7XG4gICAgICAgIHJldHVybiAke2Z1bmNOYW1lfSgke2dldFNxdWVlemVkUGFyYW1zKHBhcmFtcywga2VwdERpbXMpfSk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuXG4gIGlmIChpbnB1dEluZm8uc2hhcGVJbmZvLmlzVW5pZm9ybSkge1xuICAgIC8vIFVuaWZvcm0gYXJyYXlzIHdpbGwgYmUgbGVzcyB0aGFuIDY1NTA1IChubyByaXNrIG9mIGZsb2F0MTYgb3ZlcmZsb3cpLlxuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgcm93LCBpbnQgY29sLCBpbnQgZGVwdGgsIGludCBkZXB0aDIsIGludCBkZXB0aDMpIHtcbiAgICAgICAgZmxvYXQgaW5kZXggPSBkb3QoXG4gICAgICAgICAgdmVjNChyb3csIGNvbCwgZGVwdGgsIGRlcHRoMiksXG4gICAgICAgICAgdmVjNCgke3N0cmlkZTB9LCAke3N0cmlkZTF9LCAke3N0cmlkZTJ9LCAke3N0cmlkZTN9KSkgK1xuICAgICAgICAgIGRlcHRoMztcbiAgICAgICAgJHtnZXRVbmlmb3JtU2FtcGxlcihpbnB1dEluZm8pfVxuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICBjb25zdCBmbGF0T2Zmc2V0ID0gaW5wdXRJbmZvLnNoYXBlSW5mby5mbGF0T2Zmc2V0O1xuICBjb25zdCB0ZXhTaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8udGV4U2hhcGU7XG4gIGNvbnN0IHRleE51bVIgPSB0ZXhTaGFwZVswXTtcbiAgY29uc3QgdGV4TnVtQyA9IHRleFNoYXBlWzFdO1xuXG4gIGlmICh0ZXhOdW1DID09PSBzdHJpZGUwICYmIGZsYXRPZmZzZXQgPT0gbnVsbCkge1xuICAgIC8vIHRleEMgaXMgdXNlZCBkaXJlY3RseSBhcyBwaHlzaWNhbCAobm8gcmlzayBvZiBmbG9hdDE2IG92ZXJmbG93KS5cbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oaW50IHJvdywgaW50IGNvbCwgaW50IGRlcHRoLCBpbnQgZGVwdGgyLCBpbnQgZGVwdGgzKSB7XG4gICAgICAgIGludCB0ZXhSID0gcm93O1xuICAgICAgICBmbG9hdCB0ZXhDID0gZG90KHZlYzQoY29sLCBkZXB0aCwgZGVwdGgyLCBkZXB0aDMpLFxuICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzQoJHtzdHJpZGUxfSwgJHtzdHJpZGUyfSwgJHtzdHJpZGUzfSwgMSkpO1xuICAgICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC9cbiAgICAgICAgICAgICAgICAgICB2ZWMyKCR7dGV4TnVtQ30uMCwgJHt0ZXhOdW1SfS4wKTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICBpZiAodGV4TnVtQyA9PT0gc3RyaWRlMyAmJiBmbGF0T2Zmc2V0ID09IG51bGwpIHtcbiAgICAvLyB0ZXhSIGlzIHVzZWQgZGlyZWN0bHkgYXMgcGh5c2ljYWwgKG5vIHJpc2sgb2YgZmxvYXQxNiBvdmVyZmxvdykuXG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCwgaW50IGRlcHRoMiwgaW50IGRlcHRoMykge1xuICAgICAgICBmbG9hdCB0ZXhSID0gZG90KFxuICAgICAgICAgIHZlYzQocm93LCBjb2wsIGRlcHRoLCBkZXB0aDIpLFxuICAgICAgICAgIHZlYzQoJHtzaGFwZVsxXSAqIHNoYXBlWzJdICogc2hhcGVbM119LFxuICAgICAgICAgICAgICAgJHtzaGFwZVsyXSAqIHNoYXBlWzNdfSwgJHtzaGFwZVszXX0sIDEpKTtcbiAgICAgICAgaW50IHRleEMgPSBkZXB0aDM7XG4gICAgICAgIHZlYzIgdXYgPSAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgL1xuICAgICAgICAgICAgICAgICAgdmVjMigke3RleE51bUN9LjAsICR7dGV4TnVtUn0uMCk7XG4gICAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG5cbiAgY29uc3Qgb2Zmc2V0ID0gZ2V0RmxhdE9mZnNldFVuaWZvcm1OYW1lKHRleE5hbWUpO1xuICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCwgaW50IGRlcHRoMiwgaW50IGRlcHRoMykge1xuICAgICAgLy8gRXhwbGljaXRseSB1c2UgaW50ZWdlciBvcGVyYXRpb25zIGFzIGRvdCgpIG9ubHkgd29ya3Mgb24gZmxvYXRzLlxuICAgICAgaW50IGluZGV4ID0gcm93ICogJHtzdHJpZGUwfSArIGNvbCAqICR7c3RyaWRlMX0gKyBkZXB0aCAqICR7c3RyaWRlMn0gK1xuICAgICAgICAgIGRlcHRoMiAqICR7c3RyaWRlM30gKyBkZXB0aDMgKyAke29mZnNldH07XG4gICAgICB2ZWMyIHV2ID0gdXZGcm9tRmxhdCgke3RleE51bVJ9LCAke3RleE51bUN9LCBpbmRleCk7XG4gICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyNkQoaW5wdXRJbmZvOiBJbnB1dEluZm8pOiBzdHJpbmcge1xuICBjb25zdCBzaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlO1xuICBjb25zdCB0ZXhOYW1lID0gaW5wdXRJbmZvLm5hbWU7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKTtcblxuICBjb25zdCB7bmV3U2hhcGUsIGtlcHREaW1zfSA9IHV0aWwuc3F1ZWV6ZVNoYXBlKHNoYXBlKTtcbiAgaWYgKG5ld1NoYXBlLmxlbmd0aCA8IHNoYXBlLmxlbmd0aCkge1xuICAgIGNvbnN0IG5ld0lucHV0SW5mbyA9IHNxdWVlemVJbnB1dEluZm8oaW5wdXRJbmZvLCBuZXdTaGFwZSk7XG4gICAgY29uc3QgcGFyYW1zID0gWydyb3cnLCAnY29sJywgJ2RlcHRoJywgJ2RlcHRoMicsICdkZXB0aDMnLCAnZGVwdGg0J107XG4gICAgcmV0dXJuIGBcbiAgICAgICR7Z2V0U2FtcGxlckZyb21JbkluZm8obmV3SW5wdXRJbmZvKX1cbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCxcbiAgICAgICAgICAgICAgICAgICAgaW50IGRlcHRoMiwgaW50IGRlcHRoMywgaW50IGRlcHRoNCkge1xuICAgICAgICByZXR1cm4gJHtmdW5jTmFtZX0oJHtnZXRTcXVlZXplZFBhcmFtcyhwYXJhbXMsIGtlcHREaW1zKX0pO1xuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICBjb25zdCBzdHJpZGU0ID0gc2hhcGVbNV07XG4gIGNvbnN0IHN0cmlkZTMgPSBzaGFwZVs0XSAqIHN0cmlkZTQ7XG4gIGNvbnN0IHN0cmlkZTIgPSBzaGFwZVszXSAqIHN0cmlkZTM7XG4gIGNvbnN0IHN0cmlkZTEgPSBzaGFwZVsyXSAqIHN0cmlkZTI7XG4gIGNvbnN0IHN0cmlkZTAgPSBzaGFwZVsxXSAqIHN0cmlkZTE7XG5cbiAgaWYgKGlucHV0SW5mby5zaGFwZUluZm8uaXNVbmlmb3JtKSB7XG4gICAgLy8gVW5pZm9ybSBhcnJheXMgd2lsbCBiZSBsZXNzIHRoYW4gNjU1MDUgKG5vIHJpc2sgb2YgZmxvYXQxNiBvdmVyZmxvdykuXG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCxcbiAgICAgICAgICAgICAgICAgIGludCBkZXB0aDIsIGludCBkZXB0aDMsIGludCBkZXB0aDQpIHtcbiAgICAgICAgaW50IGluZGV4ID0gcm91bmQoZG90KFxuICAgICAgICAgIHZlYzQocm93LCBjb2wsIGRlcHRoLCBkZXB0aDIpLFxuICAgICAgICAgIHZlYzQoJHtzdHJpZGUwfSwgJHtzdHJpZGUxfSwgJHtzdHJpZGUyfSwgJHtzdHJpZGUzfSkpICtcbiAgICAgICAgICBkb3QoXG4gICAgICAgICAgICB2ZWMyKGRlcHRoMywgZGVwdGg0KSxcbiAgICAgICAgICAgIHZlYzIoJHtzdHJpZGU0fSwgMSkpKTtcbiAgICAgICAgJHtnZXRVbmlmb3JtU2FtcGxlcihpbnB1dEluZm8pfVxuICAgICAgfVxuICAgIGA7XG4gIH1cblxuICBjb25zdCBmbGF0T2Zmc2V0ID0gaW5wdXRJbmZvLnNoYXBlSW5mby5mbGF0T2Zmc2V0O1xuICBjb25zdCB0ZXhTaGFwZSA9IGlucHV0SW5mby5zaGFwZUluZm8udGV4U2hhcGU7XG4gIGNvbnN0IHRleE51bVIgPSB0ZXhTaGFwZVswXTtcbiAgY29uc3QgdGV4TnVtQyA9IHRleFNoYXBlWzFdO1xuICBpZiAodGV4TnVtQyA9PT0gc3RyaWRlMCAmJiBmbGF0T2Zmc2V0ID09IG51bGwpIHtcbiAgICAvLyB0ZXhDIGlzIHVzZWQgZGlyZWN0bHkgYXMgcGh5c2ljYWwgKG5vIHJpc2sgb2YgZmxvYXQxNiBvdmVyZmxvdykuXG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCxcbiAgICAgICAgICAgICAgICAgICAgaW50IGRlcHRoMiwgaW50IGRlcHRoMywgaW50IGRlcHRoNCkge1xuICAgICAgICBpbnQgdGV4UiA9IHJvdztcbiAgICAgICAgZmxvYXQgdGV4QyA9IGRvdCh2ZWM0KGNvbCwgZGVwdGgsIGRlcHRoMiwgZGVwdGgzKSxcbiAgICAgICAgICB2ZWM0KCR7c3RyaWRlMX0sICR7c3RyaWRlMn0sICR7c3RyaWRlM30sICR7c3RyaWRlNH0pKSArXG4gICAgICAgICAgICAgICBmbG9hdChkZXB0aDQpO1xuICAgICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC9cbiAgICAgICAgICAgICAgICAgICB2ZWMyKCR7dGV4TnVtQ30uMCwgJHt0ZXhOdW1SfS4wKTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgaWYgKHRleE51bUMgPT09IHN0cmlkZTQgJiYgZmxhdE9mZnNldCA9PSBudWxsKSB7XG4gICAgLy8gdGV4UiBpcyB1c2VkIGRpcmVjdGx5IGFzIHBoeXNpY2FsIChubyByaXNrIG9mIGZsb2F0MTYgb3ZlcmZsb3cpLlxuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShpbnQgcm93LCBpbnQgY29sLCBpbnQgZGVwdGgsXG4gICAgICAgICAgICAgICAgICAgIGludCBkZXB0aDIsIGludCBkZXB0aDMsIGludCBkZXB0aDQpIHtcbiAgICAgICAgZmxvYXQgdGV4UiA9IGRvdCh2ZWM0KHJvdywgY29sLCBkZXB0aCwgZGVwdGgyKSxcbiAgICAgICAgICB2ZWM0KCR7c2hhcGVbMV0gKiBzaGFwZVsyXSAqIHNoYXBlWzNdICogc2hhcGVbNF19LFxuICAgICAgICAgICAgICAgJHtzaGFwZVsyXSAqIHNoYXBlWzNdICogc2hhcGVbNF19LFxuICAgICAgICAgICAgICAgJHtzaGFwZVszXSAqIHNoYXBlWzRdfSxcbiAgICAgICAgICAgICAgICR7c2hhcGVbNF19KSkgKyBmbG9hdChkZXB0aDMpO1xuICAgICAgICBpbnQgdGV4QyA9IGRlcHRoNDtcbiAgICAgICAgdmVjMiB1diA9ICh2ZWMyKHRleEMsIHRleFIpICsgaGFsZkNSKSAvXG4gICAgICAgICAgICAgICAgICB2ZWMyKCR7dGV4TnVtQ30uMCwgJHt0ZXhOdW1SfS4wKTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZVRleHR1cmUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgY29uc3Qgb2Zmc2V0ID0gZ2V0RmxhdE9mZnNldFVuaWZvcm1OYW1lKHRleE5hbWUpO1xuICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KGludCByb3csIGludCBjb2wsIGludCBkZXB0aCxcbiAgICAgICAgICAgICAgICAgIGludCBkZXB0aDIsIGludCBkZXB0aDMsIGludCBkZXB0aDQpIHtcbiAgICAgIC8vIEV4cGxpY2l0bHkgdXNlIGludGVnZXIgb3BlcmF0aW9ucyBhcyBkb3QoKSBvbmx5IHdvcmtzIG9uIGZsb2F0cy5cbiAgICAgIGludCBpbmRleCA9IHJvdyAqICR7c3RyaWRlMH0gKyBjb2wgKiAke3N0cmlkZTF9ICsgZGVwdGggKiAke3N0cmlkZTJ9ICtcbiAgICAgICAgICBkZXB0aDIgKiAke3N0cmlkZTN9ICsgZGVwdGgzICogJHtzdHJpZGU0fSArIGRlcHRoNCArICR7b2Zmc2V0fTtcbiAgICAgIHZlYzIgdXYgPSB1dkZyb21GbGF0KCR7dGV4TnVtUn0sICR7dGV4TnVtQ30sIGluZGV4KTtcbiAgICAgIHJldHVybiBzYW1wbGVUZXh0dXJlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldFVuaWZvcm1TYW1wbGVyKGlucHV0SW5mbzogSW5wdXRJbmZvKTogc3RyaW5nIHtcbiAgY29uc3QgdGV4TmFtZSA9IGlucHV0SW5mby5uYW1lO1xuICBjb25zdCBpblNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoaW5wdXRJbmZvLnNoYXBlSW5mby5sb2dpY2FsU2hhcGUpO1xuXG4gIGlmIChpblNpemUgPCAyKSB7XG4gICAgcmV0dXJuIGByZXR1cm4gJHt0ZXhOYW1lfTtgO1xuICB9XG5cbiAgcmV0dXJuIGBcbiAgICBmb3IgKGludCBpID0gMDsgaSA8ICR7aW5TaXplfTsgaSsrKSB7XG4gICAgICBpZiAoaSA9PSBpbmRleCkge1xuICAgICAgICByZXR1cm4gJHt0ZXhOYW1lfVtpXTtcbiAgICAgIH1cbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldFBhY2tlZFNhbXBsZXJBdE91dHB1dENvb3JkcyhcbiAgICBpbnB1dEluZm86IElucHV0SW5mbywgb3V0U2hhcGVJbmZvOiBTaGFwZUluZm8pIHtcbiAgY29uc3QgdGV4TmFtZSA9IGlucHV0SW5mby5uYW1lO1xuICBjb25zdCB0ZXhGdW5jU25pcHBldCA9IHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCBmdW5jTmFtZSA9ICdnZXQnICsgdGV4RnVuY1NuaXBwZXQgKyAnQXRPdXRDb29yZHMnO1xuICBjb25zdCBpblJhbmsgPSBpbnB1dEluZm8uc2hhcGVJbmZvLmxvZ2ljYWxTaGFwZS5sZW5ndGg7XG4gIGNvbnN0IG91dFJhbmsgPSBvdXRTaGFwZUluZm8ubG9naWNhbFNoYXBlLmxlbmd0aDtcblxuICBjb25zdCBicm9hZGNhc3REaW1zID0gZ2V0QnJvYWRjYXN0RGltcyhcbiAgICAgIGlucHV0SW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlLCBvdXRTaGFwZUluZm8ubG9naWNhbFNoYXBlKTtcblxuICBjb25zdCB0eXBlID0gZ2V0Q29vcmRzRGF0YVR5cGUob3V0UmFuayk7XG4gIGNvbnN0IHJhbmtEaWZmID0gb3V0UmFuayAtIGluUmFuaztcbiAgbGV0IGNvb3Jkc1NuaXBwZXQ6IHN0cmluZztcbiAgY29uc3QgZmllbGRzID0gWyd4JywgJ3knLCAneicsICd3JywgJ3UnLCAndiddO1xuXG4gIGlmIChpblJhbmsgPT09IDApIHtcbiAgICBjb29yZHNTbmlwcGV0ID0gJyc7XG4gIH0gZWxzZSBpZiAob3V0UmFuayA8IDIgJiYgYnJvYWRjYXN0RGltcy5sZW5ndGggPj0gMSkge1xuICAgIGNvb3Jkc1NuaXBwZXQgPSAnY29vcmRzID0gMDsnO1xuICB9IGVsc2Uge1xuICAgIGNvb3Jkc1NuaXBwZXQgPVxuICAgICAgICBicm9hZGNhc3REaW1zLm1hcChkID0+IGBjb29yZHMuJHtmaWVsZHNbZCArIHJhbmtEaWZmXX0gPSAwO2ApXG4gICAgICAgICAgICAuam9pbignXFxuJyk7XG4gIH1cbiAgbGV0IHVucGFja2VkQ29vcmRzU25pcHBldCA9ICcnO1xuICBpZiAob3V0UmFuayA8IDIgJiYgaW5SYW5rID4gMCkge1xuICAgIHVucGFja2VkQ29vcmRzU25pcHBldCA9ICdjb29yZHMnO1xuICB9IGVsc2Uge1xuICAgIHVucGFja2VkQ29vcmRzU25pcHBldCA9IGlucHV0SW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC5tYXAoKHMsIGkpID0+IGBjb29yZHMuJHtmaWVsZHNbaSArIHJhbmtEaWZmXX1gKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAuam9pbignLCAnKTtcbiAgfVxuXG4gIGxldCBvdXRwdXQgPSBgcmV0dXJuIG91dHB1dFZhbHVlO2A7XG4gIGNvbnN0IGluU2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShpbnB1dEluZm8uc2hhcGVJbmZvLmxvZ2ljYWxTaGFwZSk7XG4gIGNvbnN0IGlzSW5wdXRTY2FsYXIgPSBpblNpemUgPT09IDE7XG4gIGNvbnN0IG91dFNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUob3V0U2hhcGVJbmZvLmxvZ2ljYWxTaGFwZSk7XG4gIGNvbnN0IGlzT3V0cHV0U2NhbGFyID0gb3V0U2l6ZSA9PT0gMTtcblxuICBpZiAoaW5SYW5rID09PSAxICYmICFpc0lucHV0U2NhbGFyICYmICFpc091dHB1dFNjYWxhcikge1xuICAgIG91dHB1dCA9IGBcbiAgICAgIHJldHVybiB2ZWM0KG91dHB1dFZhbHVlLnh5LCBvdXRwdXRWYWx1ZS54eSk7XG4gICAgYDtcbiAgfSBlbHNlIGlmIChpc0lucHV0U2NhbGFyICYmICFpc091dHB1dFNjYWxhcikge1xuICAgIGlmIChvdXRSYW5rID09PSAxKSB7XG4gICAgICBvdXRwdXQgPSBgXG4gICAgICAgIHJldHVybiB2ZWM0KG91dHB1dFZhbHVlLngsIG91dHB1dFZhbHVlLngsIDAuLCAwLik7XG4gICAgICBgO1xuICAgIH0gZWxzZSB7XG4gICAgICBvdXRwdXQgPSBgXG4gICAgICAgIHJldHVybiB2ZWM0KG91dHB1dFZhbHVlLngpO1xuICAgICAgYDtcbiAgICB9XG4gIH0gZWxzZSBpZiAoYnJvYWRjYXN0RGltcy5sZW5ndGgpIHtcbiAgICBjb25zdCByb3dzID0gaW5SYW5rIC0gMjtcbiAgICBjb25zdCBjb2xzID0gaW5SYW5rIC0gMTtcblxuICAgIGlmIChicm9hZGNhc3REaW1zLmluZGV4T2Yocm93cykgPiAtMSAmJiBicm9hZGNhc3REaW1zLmluZGV4T2YoY29scykgPiAtMSkge1xuICAgICAgb3V0cHV0ID0gYHJldHVybiB2ZWM0KG91dHB1dFZhbHVlLngpO2A7XG4gICAgfSBlbHNlIGlmIChicm9hZGNhc3REaW1zLmluZGV4T2Yocm93cykgPiAtMSkge1xuICAgICAgb3V0cHV0ID0gYHJldHVybiB2ZWM0KG91dHB1dFZhbHVlLngsIG91dHB1dFZhbHVlLnksIGAgK1xuICAgICAgICAgIGBvdXRwdXRWYWx1ZS54LCBvdXRwdXRWYWx1ZS55KTtgO1xuICAgIH0gZWxzZSBpZiAoYnJvYWRjYXN0RGltcy5pbmRleE9mKGNvbHMpID4gLTEpIHtcbiAgICAgIG91dHB1dCA9IGByZXR1cm4gdmVjNChvdXRwdXRWYWx1ZS54eCwgb3V0cHV0VmFsdWUuenopO2A7XG4gICAgfVxuICB9XG5cbiAgcmV0dXJuIGBcbiAgICB2ZWM0ICR7ZnVuY05hbWV9KCkge1xuICAgICAgJHt0eXBlfSBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICR7Y29vcmRzU25pcHBldH1cbiAgICAgIHZlYzQgb3V0cHV0VmFsdWUgPSBnZXQke3RleEZ1bmNTbmlwcGV0fSgke3VucGFja2VkQ29vcmRzU25pcHBldH0pO1xuICAgICAgJHtvdXRwdXR9XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyQXRPdXRwdXRDb29yZHMoXG4gICAgaW5wdXRJbmZvOiBJbnB1dEluZm8sIG91dFNoYXBlSW5mbzogU2hhcGVJbmZvKSB7XG4gIGNvbnN0IHRleE5hbWUgPSBpbnB1dEluZm8ubmFtZTtcbiAgY29uc3QgdGV4RnVuY1NuaXBwZXQgPSB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKTtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleEZ1bmNTbmlwcGV0ICsgJ0F0T3V0Q29vcmRzJztcbiAgY29uc3Qgb3V0VGV4U2hhcGUgPSBvdXRTaGFwZUluZm8udGV4U2hhcGU7XG4gIGNvbnN0IGluVGV4U2hhcGUgPSBpbnB1dEluZm8uc2hhcGVJbmZvLnRleFNoYXBlO1xuICBjb25zdCBpblJhbmsgPSBpbnB1dEluZm8uc2hhcGVJbmZvLmxvZ2ljYWxTaGFwZS5sZW5ndGg7XG4gIGNvbnN0IG91dFJhbmsgPSBvdXRTaGFwZUluZm8ubG9naWNhbFNoYXBlLmxlbmd0aDtcblxuICBpZiAoIWlucHV0SW5mby5zaGFwZUluZm8uaXNVbmlmb3JtICYmIGluUmFuayA9PT0gb3V0UmFuayAmJlxuICAgICAgaW5wdXRJbmZvLnNoYXBlSW5mby5mbGF0T2Zmc2V0ID09IG51bGwgJiZcbiAgICAgIHV0aWwuYXJyYXlzRXF1YWwoaW5UZXhTaGFwZSwgb3V0VGV4U2hhcGUpKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KCkge1xuICAgICAgICByZXR1cm4gc2FtcGxlVGV4dHVyZSgke3RleE5hbWV9LCByZXN1bHRVVik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuXG4gIGNvbnN0IHR5cGUgPSBnZXRDb29yZHNEYXRhVHlwZShvdXRSYW5rKTtcbiAgY29uc3QgYnJvYWRjYXN0RGltcyA9IGdldEJyb2FkY2FzdERpbXMoXG4gICAgICBpbnB1dEluZm8uc2hhcGVJbmZvLmxvZ2ljYWxTaGFwZSwgb3V0U2hhcGVJbmZvLmxvZ2ljYWxTaGFwZSk7XG4gIGNvbnN0IHJhbmtEaWZmID0gb3V0UmFuayAtIGluUmFuaztcbiAgbGV0IGNvb3Jkc1NuaXBwZXQ6IHN0cmluZztcbiAgY29uc3QgZmllbGRzID0gWyd4JywgJ3knLCAneicsICd3JywgJ3UnLCAndiddO1xuXG4gIGlmIChpblJhbmsgPT09IDApIHtcbiAgICBjb29yZHNTbmlwcGV0ID0gJyc7XG4gIH0gZWxzZSBpZiAob3V0UmFuayA8IDIgJiYgYnJvYWRjYXN0RGltcy5sZW5ndGggPj0gMSkge1xuICAgIGNvb3Jkc1NuaXBwZXQgPSAnY29vcmRzID0gMDsnO1xuICB9IGVsc2Uge1xuICAgIGNvb3Jkc1NuaXBwZXQgPVxuICAgICAgICBicm9hZGNhc3REaW1zLm1hcChkID0+IGBjb29yZHMuJHtmaWVsZHNbZCArIHJhbmtEaWZmXX0gPSAwO2ApXG4gICAgICAgICAgICAuam9pbignXFxuJyk7XG4gIH1cbiAgbGV0IHVucGFja2VkQ29vcmRzU25pcHBldCA9ICcnO1xuICBpZiAob3V0UmFuayA8IDIgJiYgaW5SYW5rID4gMCkge1xuICAgIHVucGFja2VkQ29vcmRzU25pcHBldCA9ICdjb29yZHMnO1xuICB9IGVsc2Uge1xuICAgIHVucGFja2VkQ29vcmRzU25pcHBldCA9IGlucHV0SW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC5tYXAoKHMsIGkpID0+IGBjb29yZHMuJHtmaWVsZHNbaSArIHJhbmtEaWZmXX1gKVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAuam9pbignLCAnKTtcbiAgfVxuXG4gIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oKSB7XG4gICAgICAke3R5cGV9IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgJHtjb29yZHNTbmlwcGV0fVxuICAgICAgcmV0dXJuIGdldCR7dGV4RnVuY1NuaXBwZXR9KCR7dW5wYWNrZWRDb29yZHNTbmlwcGV0fSk7XG4gICAgfVxuICBgO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0Q29vcmRzRGF0YVR5cGUocmFuazogbnVtYmVyKTogc3RyaW5nIHtcbiAgaWYgKHJhbmsgPD0gMSkge1xuICAgIHJldHVybiAnaW50JztcbiAgfSBlbHNlIGlmIChyYW5rID09PSAyKSB7XG4gICAgcmV0dXJuICdpdmVjMic7XG4gIH0gZWxzZSBpZiAocmFuayA9PT0gMykge1xuICAgIHJldHVybiAnaXZlYzMnO1xuICB9IGVsc2UgaWYgKHJhbmsgPT09IDQpIHtcbiAgICByZXR1cm4gJ2l2ZWM0JztcbiAgfSBlbHNlIGlmIChyYW5rID09PSA1KSB7XG4gICAgcmV0dXJuICdpdmVjNSc7XG4gIH0gZWxzZSBpZiAocmFuayA9PT0gNikge1xuICAgIHJldHVybiAnaXZlYzYnO1xuICB9IGVsc2Uge1xuICAgIHRocm93IEVycm9yKGBHUFUgZm9yIHJhbmsgJHtyYW5rfSBpcyBub3QgeWV0IHN1cHBvcnRlZGApO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRVbmlmb3JtSW5mb0Zyb21TaGFwZShcbiAgICBpc1BhY2tlZDogYm9vbGVhbiwgc2hhcGU6IG51bWJlcltdLCB0ZXhTaGFwZTogbnVtYmVyW10pIHtcbiAgY29uc3Qge25ld1NoYXBlLCBrZXB0RGltc30gPSB1dGlsLnNxdWVlemVTaGFwZShzaGFwZSk7XG4gIGNvbnN0IHJhbmsgPSBzaGFwZS5sZW5ndGg7XG4gIGNvbnN0IHVzZVNxdWVlemVQYWNrZWRTaGFwZSA9IGlzUGFja2VkICYmIHJhbmsgPT09IDMgJiYgc2hhcGVbMF0gPT09IDE7XG4gIGNvbnN0IHNxdWVlemVTaGFwZSA9IHVzZVNxdWVlemVQYWNrZWRTaGFwZSA/IHNoYXBlLnNsaWNlKDEpIDogbmV3U2hhcGU7XG4gIGNvbnN0IHVzZVNxdWVlemVTaGFwZSA9XG4gICAgICAoIWlzUGFja2VkICYmIHJhbmsgPiAxICYmICF1dGlsLmFycmF5c0VxdWFsKHNoYXBlLCB0ZXhTaGFwZSkgJiZcbiAgICAgICBuZXdTaGFwZS5sZW5ndGggPCByYW5rKSB8fFxuICAgICAgdXNlU3F1ZWV6ZVBhY2tlZFNoYXBlO1xuICBjb25zdCB1bmlmb3JtU2hhcGUgPSB1c2VTcXVlZXplU2hhcGUgPyBzcXVlZXplU2hhcGUgOiBzaGFwZTtcbiAgcmV0dXJuIHt1c2VTcXVlZXplU2hhcGUsIHVuaWZvcm1TaGFwZSwga2VwdERpbXN9O1xufVxuXG4vKiogUmV0dXJucyBhIG5ldyBpbnB1dCBpbmZvIChhIGNvcHkpIHRoYXQgaGFzIGEgc3F1ZWV6ZWQgbG9naWNhbCBzaGFwZS4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzcXVlZXplSW5wdXRJbmZvKFxuICAgIGluSW5mbzogSW5wdXRJbmZvLCBzcXVlZXplZFNoYXBlOiBudW1iZXJbXSk6IElucHV0SW5mbyB7XG4gIC8vIERlZXAgY29weS5cbiAgY29uc3QgbmV3SW5wdXRJbmZvOiBJbnB1dEluZm8gPSBKU09OLnBhcnNlKEpTT04uc3RyaW5naWZ5KGluSW5mbykpO1xuICBuZXdJbnB1dEluZm8uc2hhcGVJbmZvLmxvZ2ljYWxTaGFwZSA9IHNxdWVlemVkU2hhcGU7XG4gIHJldHVybiBuZXdJbnB1dEluZm87XG59XG5cbmZ1bmN0aW9uIGdldFNxdWVlemVkUGFyYW1zKHBhcmFtczogc3RyaW5nW10sIGtlcHREaW1zOiBudW1iZXJbXSk6IHN0cmluZyB7XG4gIHJldHVybiBrZXB0RGltcy5tYXAoZCA9PiBwYXJhbXNbZF0pLmpvaW4oJywgJyk7XG59XG4iXX0=