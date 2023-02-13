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
import { env, util } from '@tensorflow/tfjs-core';
import { getWebGLContext } from './canvas_util';
import { getTextureConfig } from './tex_util';
export function callAndCheck(gl, func) {
    const returnValue = func();
    if (env().getBool('DEBUG')) {
        checkWebGLError(gl);
    }
    return returnValue;
}
function checkWebGLError(gl) {
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
        throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
    }
}
// https://en.wikipedia.org/wiki/Half-precision_floating-point_format
const MIN_FLOAT16 = 5.96e-8;
const MAX_FLOAT16 = 65504;
export function canBeRepresented(num) {
    if (env().getBool('WEBGL_RENDER_FLOAT32_ENABLED') || num === 0 ||
        (MIN_FLOAT16 < Math.abs(num) && Math.abs(num) < MAX_FLOAT16)) {
        return true;
    }
    return false;
}
export function getWebGLErrorMessage(gl, status) {
    switch (status) {
        case gl.NO_ERROR:
            return 'NO_ERROR';
        case gl.INVALID_ENUM:
            return 'INVALID_ENUM';
        case gl.INVALID_VALUE:
            return 'INVALID_VALUE';
        case gl.INVALID_OPERATION:
            return 'INVALID_OPERATION';
        case gl.INVALID_FRAMEBUFFER_OPERATION:
            return 'INVALID_FRAMEBUFFER_OPERATION';
        case gl.OUT_OF_MEMORY:
            return 'OUT_OF_MEMORY';
        case gl.CONTEXT_LOST_WEBGL:
            return 'CONTEXT_LOST_WEBGL';
        default:
            return `Unknown error code ${status}`;
    }
}
export function getExtensionOrThrow(gl, extensionName) {
    return throwIfNull(gl, () => gl.getExtension(extensionName), 'Extension "' + extensionName + '" not supported on this browser.');
}
export function createVertexShader(gl, vertexShaderSource) {
    const vertexShader = throwIfNull(gl, () => gl.createShader(gl.VERTEX_SHADER), 'Unable to create vertex WebGLShader.');
    callAndCheck(gl, () => gl.shaderSource(vertexShader, vertexShaderSource));
    callAndCheck(gl, () => gl.compileShader(vertexShader));
    if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) === false) {
        console.log(gl.getShaderInfoLog(vertexShader));
        throw new Error('Failed to compile vertex shader.');
    }
    return vertexShader;
}
export function createFragmentShader(gl, fragmentShaderSource) {
    const fragmentShader = throwIfNull(gl, () => gl.createShader(gl.FRAGMENT_SHADER), 'Unable to create fragment WebGLShader.');
    callAndCheck(gl, () => gl.shaderSource(fragmentShader, fragmentShaderSource));
    callAndCheck(gl, () => gl.compileShader(fragmentShader));
    if (env().get('ENGINE_COMPILE_ONLY')) {
        return fragmentShader;
    }
    if (gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS) === false) {
        logShaderSourceAndInfoLog(fragmentShaderSource, gl.getShaderInfoLog(fragmentShader));
        throw new Error('Failed to compile fragment shader.');
    }
    return fragmentShader;
}
const lineNumberRegex = /ERROR: [0-9]+:([0-9]+):/g;
export function logShaderSourceAndInfoLog(shaderSource, shaderInfoLog) {
    const lineNumberRegexResult = lineNumberRegex.exec(shaderInfoLog);
    if (lineNumberRegexResult == null) {
        console.log(`Couldn't parse line number in error: ${shaderInfoLog}`);
        console.log(shaderSource);
        return;
    }
    const lineNumber = +lineNumberRegexResult[1];
    const shaderLines = shaderSource.split('\n');
    const pad = shaderLines.length.toString().length + 2;
    const linesWithLineNumbers = shaderLines.map((line, lineNumber) => util.rightPad((lineNumber + 1).toString(), pad) + line);
    let maxLineLength = 0;
    for (let i = 0; i < linesWithLineNumbers.length; i++) {
        maxLineLength = Math.max(linesWithLineNumbers[i].length, maxLineLength);
    }
    const beforeErrorLines = linesWithLineNumbers.slice(0, lineNumber - 1);
    const errorLine = linesWithLineNumbers.slice(lineNumber - 1, lineNumber);
    const afterErrorLines = linesWithLineNumbers.slice(lineNumber);
    console.log(beforeErrorLines.join('\n'));
    console.log(shaderInfoLog.split('\n')[0]);
    console.log(`%c ${util.rightPad(errorLine[0], maxLineLength)}`, 'border:1px solid red; background-color:#e3d2d2; color:#a61717');
    console.log(afterErrorLines.join('\n'));
}
export function createProgram(gl) {
    return throwIfNull(gl, () => gl.createProgram(), 'Unable to create WebGLProgram.');
}
export function linkProgram(gl, program) {
    callAndCheck(gl, () => gl.linkProgram(program));
    if (env().get('ENGINE_COMPILE_ONLY')) {
        return;
    }
    if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Failed to link vertex and fragment shaders.');
    }
}
export function validateProgram(gl, program) {
    callAndCheck(gl, () => gl.validateProgram(program));
    if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Shader program validation failed.');
    }
}
export function createStaticVertexBuffer(gl, data) {
    const buffer = throwIfNull(gl, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
    callAndCheck(gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
    callAndCheck(gl, () => gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW));
    return buffer;
}
export function createStaticIndexBuffer(gl, data) {
    const buffer = throwIfNull(gl, () => gl.createBuffer(), 'Unable to create WebGLBuffer');
    callAndCheck(gl, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer));
    callAndCheck(gl, () => gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW));
    return buffer;
}
export function getNumChannels() {
    if (env().getNumber('WEBGL_VERSION') === 2) {
        return 1;
    }
    return 4;
}
export function createTexture(gl) {
    return throwIfNull(gl, () => gl.createTexture(), 'Unable to create WebGLTexture.');
}
export function validateTextureSize(width, height) {
    const maxTextureSize = env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
    if ((width <= 0) || (height <= 0)) {
        const requested = `[${width}x${height}]`;
        throw new Error('Requested texture size ' + requested + ' is invalid.');
    }
    if ((width > maxTextureSize) || (height > maxTextureSize)) {
        const requested = `[${width}x${height}]`;
        const max = `[${maxTextureSize}x${maxTextureSize}]`;
        throw new Error('Requested texture size ' + requested +
            ' greater than WebGL maximum on this browser / GPU ' + max + '.');
    }
}
export function createFramebuffer(gl) {
    return throwIfNull(gl, () => gl.createFramebuffer(), 'Unable to create WebGLFramebuffer.');
}
export function bindVertexBufferToProgramAttribute(gl, program, attribute, buffer, arrayEntriesPerItem, itemStrideInBytes, itemOffsetInBytes) {
    const loc = gl.getAttribLocation(program, attribute);
    if (loc === -1) {
        // The GPU compiler decided to strip out this attribute because it's unused,
        // thus no need to bind.
        return false;
    }
    callAndCheck(gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, buffer));
    callAndCheck(gl, () => gl.vertexAttribPointer(loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes, itemOffsetInBytes));
    callAndCheck(gl, () => gl.enableVertexAttribArray(loc));
    return true;
}
export function bindTextureUnit(gl, texture, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, () => gl.activeTexture(gl.TEXTURE0 + textureUnit));
    callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, texture));
}
export function unbindTextureUnit(gl, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, () => gl.activeTexture(gl.TEXTURE0 + textureUnit));
    callAndCheck(gl, () => gl.bindTexture(gl.TEXTURE_2D, null));
}
export function getProgramUniformLocationOrThrow(gl, program, uniformName) {
    return throwIfNull(gl, () => gl.getUniformLocation(program, uniformName), 'uniform "' + uniformName + '" not present in program.');
}
export function getProgramUniformLocation(gl, program, uniformName) {
    return gl.getUniformLocation(program, uniformName);
}
export function bindTextureToProgramUniformSampler(gl, texture, uniformSamplerLocation, textureUnit) {
    callAndCheck(gl, () => bindTextureUnit(gl, texture, textureUnit));
    callAndCheck(gl, () => gl.uniform1i(uniformSamplerLocation, textureUnit));
}
export function bindCanvasToFramebuffer(gl) {
    callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
    callAndCheck(gl, () => gl.viewport(0, 0, gl.canvas.width, gl.canvas.height));
    callAndCheck(gl, () => gl.scissor(0, 0, gl.canvas.width, gl.canvas.height));
}
export function bindColorTextureToFramebuffer(gl, texture, framebuffer) {
    callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
    callAndCheck(gl, () => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0));
}
export function unbindColorTextureFromFramebuffer(gl, framebuffer) {
    callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer));
    callAndCheck(gl, () => gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0));
}
export function validateFramebuffer(gl) {
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Error binding framebuffer: ' + getFramebufferErrorMessage(gl, status));
    }
}
export function getFramebufferErrorMessage(gl, status) {
    switch (status) {
        case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
            return 'FRAMEBUFFER_INCOMPLETE_DIMENSIONS';
        case gl.FRAMEBUFFER_UNSUPPORTED:
            return 'FRAMEBUFFER_UNSUPPORTED';
        default:
            return `unknown error ${status}`;
    }
}
function throwIfNull(gl, returnTOrNull, failureMessage) {
    const tOrNull = callAndCheck(gl, () => returnTOrNull());
    if (tOrNull == null) {
        throw new Error(failureMessage);
    }
    return tOrNull;
}
function validateTextureUnit(gl, textureUnit) {
    const maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
    const glTextureUnit = textureUnit + gl.TEXTURE0;
    if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
        const textureUnitRange = `[gl.TEXTURE0, gl.TEXTURE${maxTextureUnit}]`;
        throw new Error(`textureUnit must be in ${textureUnitRange}.`);
    }
}
export function getBatchDim(shape, dimsToSkip = 2) {
    return util.sizeFromShape(shape.slice(0, shape.length - dimsToSkip));
}
export function getRowsCols(shape) {
    if (shape.length === 0) {
        throw Error('Cannot get rows and columns of an empty shape array.');
    }
    return [
        shape.length > 1 ? shape[shape.length - 2] : 1, shape[shape.length - 1]
    ];
}
export function getShapeAs3D(shape) {
    let shapeAs3D = [1, 1, 1];
    const isScalar = shape.length === 0 || (shape.length === 1 && shape[0] === 1);
    if (!isScalar) {
        shapeAs3D =
            [getBatchDim(shape), ...getRowsCols(shape)];
    }
    return shapeAs3D;
}
export function getTextureShapeFromLogicalShape(logShape, isPacked = false) {
    let maxTexSize = env().getNumber('WEBGL_MAX_TEXTURE_SIZE');
    if (isPacked) {
        maxTexSize = maxTexSize * 2;
        // This logic ensures we accurately count the number of packed texels needed
        // to accommodate the tensor. We can only pack values in the same texel if
        // they are from adjacent pairs of rows/cols within the same batch. So if a
        // tensor has 3 rows, we pretend it has 4 rows in order to account for the
        // fact that the texels containing the third row are half empty.
        logShape = logShape.map((d, i) => i >= logShape.length - 2 ?
            util.nearestLargerEven(logShape[i]) :
            logShape[i]);
        // Packed texture height is at least 2 (the channel height of a single
        // texel).
        if (logShape.length === 1) {
            logShape = [2, logShape[0]];
        }
    }
    // If logical shape is 2, we don't squeeze, since we want to match physical.
    if (logShape.length !== 2) {
        const squeezeResult = util.squeezeShape(logShape);
        logShape = squeezeResult.newShape;
    }
    let size = util.sizeFromShape(logShape);
    if (logShape.length <= 1 && size <= maxTexSize) {
        return [1, size];
    }
    else if (logShape.length === 2 && logShape[0] <= maxTexSize &&
        logShape[1] <= maxTexSize) {
        return logShape;
    }
    else if (logShape.length === 3 && logShape[0] * logShape[1] <= maxTexSize &&
        logShape[2] <= maxTexSize) {
        return [logShape[0] * logShape[1], logShape[2]];
    }
    else if (logShape.length === 3 && logShape[0] <= maxTexSize &&
        logShape[1] * logShape[2] <= maxTexSize) {
        return [logShape[0], logShape[1] * logShape[2]];
    }
    else if (logShape.length === 4 &&
        logShape[0] * logShape[1] * logShape[2] <= maxTexSize &&
        logShape[3] <= maxTexSize) {
        return [logShape[0] * logShape[1] * logShape[2], logShape[3]];
    }
    else if (logShape.length === 4 && logShape[0] <= maxTexSize &&
        logShape[1] * logShape[2] * logShape[3] <= maxTexSize) {
        return [logShape[0], logShape[1] * logShape[2] * logShape[3]];
    }
    else {
        if (isPacked) {
            // For packed textures size equals the number of channels required to
            // accommodate the texture data. However in order to squarify such that
            // inner dimensions stay even, we rewrite size to equal the number of
            // texels. Then in the return statement we rehydrate the squarified
            // dimensions to channel units.
            const batchDim = getBatchDim(logShape);
            let rows = 2, cols = 2;
            if (logShape.length) {
                [rows, cols] = getRowsCols(logShape);
            }
            size = batchDim * (rows / 2) * (cols / 2);
            return util.sizeToSquarishShape(size).map(d => d * 2);
        }
        return util.sizeToSquarishShape(size);
    }
}
function isEven(n) {
    return n % 2 === 0;
}
/**
 * This determines whether reshaping a packed texture requires rearranging
 * the data within the texture, assuming 2x2 packing.
 */
export function isReshapeFree(shape1, shape2) {
    shape1 = shape1.slice(-2);
    shape2 = shape2.slice(-2);
    if (util.arraysEqual(shape1, shape2)) {
        return true;
    }
    if (!shape1.length || !shape2.length) { // One of the shapes is a scalar.
        return true;
    }
    if (shape1[0] === 0 || shape1[1] === 0 || shape2[0] === 0 ||
        shape2[1] === 0) {
        return true;
    }
    if (shape1.length !== shape2.length) { // One of the shapes is a vector.
        const shape1Cols = shape1.slice(-1)[0];
        const shape2Cols = shape2.slice(-1)[0];
        if (shape1Cols === shape2Cols) {
            return true;
        }
        if (isEven(shape1Cols) && isEven(shape2Cols) &&
            (shape1[0] === 1 || shape2[0] === 1)) {
            return true;
        }
    }
    return shape1[1] === shape2[1] && isEven(shape1[0]) && isEven(shape2[0]);
}
// We cache webgl params because the environment gets reset between
// unit tests and we don't want to constantly query the WebGLContext for
// MAX_TEXTURE_SIZE.
let MAX_TEXTURE_SIZE;
let MAX_TEXTURES_IN_SHADER;
export function getWebGLMaxTextureSize(webGLVersion) {
    if (MAX_TEXTURE_SIZE == null) {
        const gl = getWebGLContext(webGLVersion);
        MAX_TEXTURE_SIZE = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    }
    return MAX_TEXTURE_SIZE;
}
export function resetMaxTextureSize() {
    MAX_TEXTURE_SIZE = null;
}
export function resetMaxTexturesInShader() {
    MAX_TEXTURES_IN_SHADER = null;
}
export function getMaxTexturesInShader(webGLVersion) {
    if (MAX_TEXTURES_IN_SHADER == null) {
        const gl = getWebGLContext(webGLVersion);
        MAX_TEXTURES_IN_SHADER = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
    }
    // We cap at 16 to avoid spurious runtime "memory exhausted" error.
    return Math.min(16, MAX_TEXTURES_IN_SHADER);
}
export function getWebGLDisjointQueryTimerVersion(webGLVersion) {
    if (webGLVersion === 0) {
        return 0;
    }
    let queryTimerVersion;
    const gl = getWebGLContext(webGLVersion);
    if (hasExtension(gl, 'EXT_disjoint_timer_query_webgl2') &&
        webGLVersion === 2) {
        queryTimerVersion = 2;
    }
    else if (hasExtension(gl, 'EXT_disjoint_timer_query')) {
        queryTimerVersion = 1;
    }
    else {
        queryTimerVersion = 0;
    }
    return queryTimerVersion;
}
export function hasExtension(gl, extensionName) {
    const ext = gl.getExtension(extensionName);
    return ext != null;
}
export function isWebGLVersionEnabled(webGLVersion) {
    try {
        const gl = getWebGLContext(webGLVersion);
        if (gl != null) {
            return true;
        }
    }
    catch (e) {
        console.log('Error when getting WebGL context: ', e);
        return false;
    }
    return false;
}
export function isCapableOfRenderingToFloatTexture(webGLVersion) {
    if (webGLVersion === 0) {
        return false;
    }
    const gl = getWebGLContext(webGLVersion);
    if (webGLVersion === 1) {
        if (!hasExtension(gl, 'OES_texture_float')) {
            return false;
        }
    }
    else {
        if (!hasExtension(gl, 'EXT_color_buffer_float')) {
            return false;
        }
    }
    const isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
    return isFrameBufferComplete;
}
/**
 * Check if we can download values from a float/half-float texture.
 *
 * Note that for performance reasons we use binding a texture to a framebuffer
 * as a proxy for ability to download float values later using readPixels. The
 * texture params of this texture will not match those in readPixels exactly
 * but if we are unable to bind some kind of float texture to the frameBuffer
 * then we definitely will not be able to read float values from it.
 */
export function isDownloadFloatTextureEnabled(webGLVersion) {
    if (webGLVersion === 0) {
        return false;
    }
    const gl = getWebGLContext(webGLVersion);
    if (webGLVersion === 1) {
        if (!hasExtension(gl, 'OES_texture_float')) {
            return false;
        }
        if (!hasExtension(gl, 'WEBGL_color_buffer_float')) {
            return false;
        }
    }
    else {
        if (hasExtension(gl, 'EXT_color_buffer_float')) {
            return createFloatTextureAndBindToFramebuffer(gl);
        }
        const COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
        if (hasExtension(gl, COLOR_BUFFER_HALF_FLOAT)) {
            const textureHalfFloatExtension = gl.getExtension(COLOR_BUFFER_HALF_FLOAT);
            return createHalfFloatTextureAndBindToFramebuffer(gl, textureHalfFloatExtension);
        }
        return false;
    }
    const isFrameBufferComplete = createFloatTextureAndBindToFramebuffer(gl);
    return isFrameBufferComplete;
}
function createFloatTextureAndBindToFramebuffer(gl) {
    const texConfig = getTextureConfig(gl);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const width = 1;
    const height = 1;
    gl.texImage2D(gl.TEXTURE_2D, 0, texConfig.internalFormatFloat, width, height, 0, texConfig.textureFormatFloat, texConfig.textureTypeFloat, null);
    const frameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const isFrameBufferComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(frameBuffer);
    return isFrameBufferComplete;
}
function createHalfFloatTextureAndBindToFramebuffer(
// tslint:disable-next-line:no-any
gl, textureHalfFloatExtension) {
    const texConfig = getTextureConfig(gl, textureHalfFloatExtension);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const width = 1;
    const height = 1;
    gl.texImage2D(gl.TEXTURE_2D, 0, texConfig.internalFormatHalfFloat, width, height, 0, texConfig.textureFormatFloat, texConfig.textureTypeHalfFloat, null);
    const frameBuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const isFrameBufferComplete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(frameBuffer);
    return isFrameBufferComplete;
}
export function isWebGLFenceEnabled(webGLVersion) {
    if (webGLVersion !== 2) {
        return false;
    }
    const gl = getWebGLContext(webGLVersion);
    // tslint:disable-next-line:no-any
    const isEnabled = gl.fenceSync != null;
    return isEnabled;
}
export function assertNotComplex(tensor, opName) {
    if (!Array.isArray(tensor)) {
        tensor = [tensor];
    }
    tensor.forEach(t => {
        if (t != null) {
            util.assert(t.dtype !== 'complex64', () => `${opName} does not support complex64 tensors ` +
                'in the WebGL backend.');
        }
    });
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoid2ViZ2xfdXRpbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvd2ViZ2xfdXRpbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsR0FBRyxFQUFjLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRTVELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFDOUMsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRTVDLE1BQU0sVUFBVSxZQUFZLENBQUksRUFBeUIsRUFBRSxJQUFhO0lBQ3RFLE1BQU0sV0FBVyxHQUFHLElBQUksRUFBRSxDQUFDO0lBQzNCLElBQUksR0FBRyxFQUFFLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxFQUFFO1FBQzFCLGVBQWUsQ0FBQyxFQUFFLENBQUMsQ0FBQztLQUNyQjtJQUNELE9BQU8sV0FBVyxDQUFDO0FBQ3JCLENBQUM7QUFFRCxTQUFTLGVBQWUsQ0FBQyxFQUF5QjtJQUNoRCxNQUFNLEtBQUssR0FBRyxFQUFFLENBQUMsUUFBUSxFQUFFLENBQUM7SUFDNUIsSUFBSSxLQUFLLEtBQUssRUFBRSxDQUFDLFFBQVEsRUFBRTtRQUN6QixNQUFNLElBQUksS0FBSyxDQUFDLGVBQWUsR0FBRyxvQkFBb0IsQ0FBQyxFQUFFLEVBQUUsS0FBSyxDQUFDLENBQUMsQ0FBQztLQUNwRTtBQUNILENBQUM7QUFFRCxxRUFBcUU7QUFDckUsTUFBTSxXQUFXLEdBQUcsT0FBTyxDQUFDO0FBQzVCLE1BQU0sV0FBVyxHQUFHLEtBQUssQ0FBQztBQUUxQixNQUFNLFVBQVUsZ0JBQWdCLENBQUMsR0FBVztJQUMxQyxJQUFJLEdBQUcsRUFBRSxDQUFDLE9BQU8sQ0FBQyw4QkFBOEIsQ0FBQyxJQUFJLEdBQUcsS0FBSyxDQUFDO1FBQzFELENBQUMsV0FBVyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLElBQUksSUFBSSxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsR0FBRyxXQUFXLENBQUMsRUFBRTtRQUNoRSxPQUFPLElBQUksQ0FBQztLQUNiO0lBQ0QsT0FBTyxLQUFLLENBQUM7QUFDZixDQUFDO0FBRUQsTUFBTSxVQUFVLG9CQUFvQixDQUNoQyxFQUF5QixFQUFFLE1BQWM7SUFDM0MsUUFBUSxNQUFNLEVBQUU7UUFDZCxLQUFLLEVBQUUsQ0FBQyxRQUFRO1lBQ2QsT0FBTyxVQUFVLENBQUM7UUFDcEIsS0FBSyxFQUFFLENBQUMsWUFBWTtZQUNsQixPQUFPLGNBQWMsQ0FBQztRQUN4QixLQUFLLEVBQUUsQ0FBQyxhQUFhO1lBQ25CLE9BQU8sZUFBZSxDQUFDO1FBQ3pCLEtBQUssRUFBRSxDQUFDLGlCQUFpQjtZQUN2QixPQUFPLG1CQUFtQixDQUFDO1FBQzdCLEtBQUssRUFBRSxDQUFDLDZCQUE2QjtZQUNuQyxPQUFPLCtCQUErQixDQUFDO1FBQ3pDLEtBQUssRUFBRSxDQUFDLGFBQWE7WUFDbkIsT0FBTyxlQUFlLENBQUM7UUFDekIsS0FBSyxFQUFFLENBQUMsa0JBQWtCO1lBQ3hCLE9BQU8sb0JBQW9CLENBQUM7UUFDOUI7WUFDRSxPQUFPLHNCQUFzQixNQUFNLEVBQUUsQ0FBQztLQUN6QztBQUNILENBQUM7QUFFRCxNQUFNLFVBQVUsbUJBQW1CLENBQy9CLEVBQXlCLEVBQUUsYUFBcUI7SUFDbEQsT0FBTyxXQUFXLENBQ2QsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxZQUFZLENBQUMsYUFBYSxDQUFDLEVBQ3hDLGFBQWEsR0FBRyxhQUFhLEdBQUcsa0NBQWtDLENBQUMsQ0FBQztBQUMxRSxDQUFDO0FBRUQsTUFBTSxVQUFVLGtCQUFrQixDQUM5QixFQUF5QixFQUFFLGtCQUEwQjtJQUN2RCxNQUFNLFlBQVksR0FBZ0IsV0FBVyxDQUN6QyxFQUFFLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsYUFBYSxDQUFDLEVBQzNDLHNDQUFzQyxDQUFDLENBQUM7SUFDNUMsWUFBWSxDQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsWUFBWSxDQUFDLFlBQVksRUFBRSxrQkFBa0IsQ0FBQyxDQUFDLENBQUM7SUFDMUUsWUFBWSxDQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsYUFBYSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUM7SUFDdkQsSUFBSSxFQUFFLENBQUMsa0JBQWtCLENBQUMsWUFBWSxFQUFFLEVBQUUsQ0FBQyxjQUFjLENBQUMsS0FBSyxLQUFLLEVBQUU7UUFDcEUsT0FBTyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsZ0JBQWdCLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQztRQUMvQyxNQUFNLElBQUksS0FBSyxDQUFDLGtDQUFrQyxDQUFDLENBQUM7S0FDckQ7SUFDRCxPQUFPLFlBQVksQ0FBQztBQUN0QixDQUFDO0FBRUQsTUFBTSxVQUFVLG9CQUFvQixDQUNoQyxFQUF5QixFQUFFLG9CQUE0QjtJQUN6RCxNQUFNLGNBQWMsR0FBZ0IsV0FBVyxDQUMzQyxFQUFFLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLFlBQVksQ0FBQyxFQUFFLENBQUMsZUFBZSxDQUFDLEVBQzdDLHdDQUF3QyxDQUFDLENBQUM7SUFDOUMsWUFBWSxDQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsWUFBWSxDQUFDLGNBQWMsRUFBRSxvQkFBb0IsQ0FBQyxDQUFDLENBQUM7SUFDOUUsWUFBWSxDQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsYUFBYSxDQUFDLGNBQWMsQ0FBQyxDQUFDLENBQUM7SUFDekQsSUFBSSxHQUFHLEVBQUUsQ0FBQyxHQUFHLENBQUMscUJBQXFCLENBQUMsRUFBRTtRQUNwQyxPQUFPLGNBQWMsQ0FBQztLQUN2QjtJQUNELElBQUksRUFBRSxDQUFDLGtCQUFrQixDQUFDLGNBQWMsRUFBRSxFQUFFLENBQUMsY0FBYyxDQUFDLEtBQUssS0FBSyxFQUFFO1FBQ3RFLHlCQUF5QixDQUNyQixvQkFBb0IsRUFBRSxFQUFFLENBQUMsZ0JBQWdCLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQztRQUMvRCxNQUFNLElBQUksS0FBSyxDQUFDLG9DQUFvQyxDQUFDLENBQUM7S0FDdkQ7SUFDRCxPQUFPLGNBQWMsQ0FBQztBQUN4QixDQUFDO0FBRUQsTUFBTSxlQUFlLEdBQUcsMEJBQTBCLENBQUM7QUFDbkQsTUFBTSxVQUFVLHlCQUF5QixDQUNyQyxZQUFvQixFQUFFLGFBQXFCO0lBQzdDLE1BQU0scUJBQXFCLEdBQUcsZUFBZSxDQUFDLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUNsRSxJQUFJLHFCQUFxQixJQUFJLElBQUksRUFBRTtRQUNqQyxPQUFPLENBQUMsR0FBRyxDQUFDLHdDQUF3QyxhQUFhLEVBQUUsQ0FBQyxDQUFDO1FBQ3JFLE9BQU8sQ0FBQyxHQUFHLENBQUMsWUFBWSxDQUFDLENBQUM7UUFDMUIsT0FBTztLQUNSO0lBRUQsTUFBTSxVQUFVLEdBQUcsQ0FBQyxxQkFBcUIsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUU3QyxNQUFNLFdBQVcsR0FBRyxZQUFZLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQzdDLE1BQU0sR0FBRyxHQUFHLFdBQVcsQ0FBQyxNQUFNLENBQUMsUUFBUSxFQUFFLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQztJQUNyRCxNQUFNLG9CQUFvQixHQUFHLFdBQVcsQ0FBQyxHQUFHLENBQ3hDLENBQUMsSUFBSSxFQUFFLFVBQVUsRUFBRSxFQUFFLENBQ2pCLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxVQUFVLEdBQUcsQ0FBQyxDQUFDLENBQUMsUUFBUSxFQUFFLEVBQUUsR0FBRyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUM7SUFDaEUsSUFBSSxhQUFhLEdBQUcsQ0FBQyxDQUFDO0lBQ3RCLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxvQkFBb0IsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUU7UUFDcEQsYUFBYSxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsb0JBQW9CLENBQUMsQ0FBQyxDQUFDLENBQUMsTUFBTSxFQUFFLGFBQWEsQ0FBQyxDQUFDO0tBQ3pFO0lBRUQsTUFBTSxnQkFBZ0IsR0FBRyxvQkFBb0IsQ0FBQyxLQUFLLENBQUMsQ0FBQyxFQUFFLFVBQVUsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUN2RSxNQUFNLFNBQVMsR0FBRyxvQkFBb0IsQ0FBQyxLQUFLLENBQUMsVUFBVSxHQUFHLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQztJQUN6RSxNQUFNLGVBQWUsR0FBRyxvQkFBb0IsQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLENBQUM7SUFFL0QsT0FBTyxDQUFDLEdBQUcsQ0FBQyxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztJQUN6QyxPQUFPLENBQUMsR0FBRyxDQUFDLGFBQWEsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUMxQyxPQUFPLENBQUMsR0FBRyxDQUNQLE1BQU0sSUFBSSxDQUFDLFFBQVEsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLEVBQUUsYUFBYSxDQUFDLEVBQUUsRUFDbEQsK0RBQStELENBQUMsQ0FBQztJQUNyRSxPQUFPLENBQUMsR0FBRyxDQUFDLGVBQWUsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztBQUMxQyxDQUFDO0FBRUQsTUFBTSxVQUFVLGFBQWEsQ0FBQyxFQUF5QjtJQUNyRCxPQUFPLFdBQVcsQ0FDZCxFQUFFLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLGFBQWEsRUFBRSxFQUFFLGdDQUFnQyxDQUFDLENBQUM7QUFDdEUsQ0FBQztBQUVELE1BQU0sVUFBVSxXQUFXLENBQUMsRUFBeUIsRUFBRSxPQUFxQjtJQUMxRSxZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxXQUFXLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNoRCxJQUFJLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxxQkFBcUIsQ0FBQyxFQUFFO1FBQ3BDLE9BQU87S0FDUjtJQUNELElBQUksRUFBRSxDQUFDLG1CQUFtQixDQUFDLE9BQU8sRUFBRSxFQUFFLENBQUMsV0FBVyxDQUFDLEtBQUssS0FBSyxFQUFFO1FBQzdELE9BQU8sQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLGlCQUFpQixDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7UUFDM0MsTUFBTSxJQUFJLEtBQUssQ0FBQyw2Q0FBNkMsQ0FBQyxDQUFDO0tBQ2hFO0FBQ0gsQ0FBQztBQUVELE1BQU0sVUFBVSxlQUFlLENBQzNCLEVBQXlCLEVBQUUsT0FBcUI7SUFDbEQsWUFBWSxDQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsZUFBZSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDcEQsSUFBSSxFQUFFLENBQUMsbUJBQW1CLENBQUMsT0FBTyxFQUFFLEVBQUUsQ0FBQyxlQUFlLENBQUMsS0FBSyxLQUFLLEVBQUU7UUFDakUsT0FBTyxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsaUJBQWlCLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUMzQyxNQUFNLElBQUksS0FBSyxDQUFDLG1DQUFtQyxDQUFDLENBQUM7S0FDdEQ7QUFDSCxDQUFDO0FBRUQsTUFBTSxVQUFVLHdCQUF3QixDQUNwQyxFQUF5QixFQUFFLElBQWtCO0lBQy9DLE1BQU0sTUFBTSxHQUFnQixXQUFXLENBQ25DLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLEVBQUUsOEJBQThCLENBQUMsQ0FBQztJQUNqRSxZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLFlBQVksRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDO0lBQy9ELFlBQVksQ0FBQyxFQUFFLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLFVBQVUsQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLElBQUksRUFBRSxFQUFFLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQztJQUM3RSxPQUFPLE1BQU0sQ0FBQztBQUNoQixDQUFDO0FBRUQsTUFBTSxVQUFVLHVCQUF1QixDQUNuQyxFQUF5QixFQUFFLElBQWlCO0lBQzlDLE1BQU0sTUFBTSxHQUFnQixXQUFXLENBQ25DLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsWUFBWSxFQUFFLEVBQUUsOEJBQThCLENBQUMsQ0FBQztJQUNqRSxZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLG9CQUFvQixFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUM7SUFDdkUsWUFBWSxDQUNSLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsVUFBVSxDQUFDLEVBQUUsQ0FBQyxvQkFBb0IsRUFBRSxJQUFJLEVBQUUsRUFBRSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUM7SUFDNUUsT0FBTyxNQUFNLENBQUM7QUFDaEIsQ0FBQztBQUVELE1BQU0sVUFBVSxjQUFjO0lBQzVCLElBQUksR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLGVBQWUsQ0FBQyxLQUFLLENBQUMsRUFBRTtRQUMxQyxPQUFPLENBQUMsQ0FBQztLQUNWO0lBQ0QsT0FBTyxDQUFDLENBQUM7QUFDWCxDQUFDO0FBRUQsTUFBTSxVQUFVLGFBQWEsQ0FBQyxFQUF5QjtJQUNyRCxPQUFPLFdBQVcsQ0FDZCxFQUFFLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLGFBQWEsRUFBRSxFQUFFLGdDQUFnQyxDQUFDLENBQUM7QUFDdEUsQ0FBQztBQUVELE1BQU0sVUFBVSxtQkFBbUIsQ0FBQyxLQUFhLEVBQUUsTUFBYztJQUMvRCxNQUFNLGNBQWMsR0FBRyxHQUFHLEVBQUUsQ0FBQyxTQUFTLENBQUMsd0JBQXdCLENBQUMsQ0FBQztJQUNqRSxJQUFJLENBQUMsS0FBSyxJQUFJLENBQUMsQ0FBQyxJQUFJLENBQUMsTUFBTSxJQUFJLENBQUMsQ0FBQyxFQUFFO1FBQ2pDLE1BQU0sU0FBUyxHQUFHLElBQUksS0FBSyxJQUFJLE1BQU0sR0FBRyxDQUFDO1FBQ3pDLE1BQU0sSUFBSSxLQUFLLENBQUMseUJBQXlCLEdBQUcsU0FBUyxHQUFHLGNBQWMsQ0FBQyxDQUFDO0tBQ3pFO0lBQ0QsSUFBSSxDQUFDLEtBQUssR0FBRyxjQUFjLENBQUMsSUFBSSxDQUFDLE1BQU0sR0FBRyxjQUFjLENBQUMsRUFBRTtRQUN6RCxNQUFNLFNBQVMsR0FBRyxJQUFJLEtBQUssSUFBSSxNQUFNLEdBQUcsQ0FBQztRQUN6QyxNQUFNLEdBQUcsR0FBRyxJQUFJLGNBQWMsSUFBSSxjQUFjLEdBQUcsQ0FBQztRQUNwRCxNQUFNLElBQUksS0FBSyxDQUNYLHlCQUF5QixHQUFHLFNBQVM7WUFDckMsb0RBQW9ELEdBQUcsR0FBRyxHQUFHLEdBQUcsQ0FBQyxDQUFDO0tBQ3ZFO0FBQ0gsQ0FBQztBQUVELE1BQU0sVUFBVSxpQkFBaUIsQ0FBQyxFQUF5QjtJQUN6RCxPQUFPLFdBQVcsQ0FDZCxFQUFFLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLGlCQUFpQixFQUFFLEVBQUUsb0NBQW9DLENBQUMsQ0FBQztBQUM5RSxDQUFDO0FBRUQsTUFBTSxVQUFVLGtDQUFrQyxDQUM5QyxFQUF5QixFQUFFLE9BQXFCLEVBQUUsU0FBaUIsRUFDbkUsTUFBbUIsRUFBRSxtQkFBMkIsRUFBRSxpQkFBeUIsRUFDM0UsaUJBQXlCO0lBQzNCLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxpQkFBaUIsQ0FBQyxPQUFPLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDckQsSUFBSSxHQUFHLEtBQUssQ0FBQyxDQUFDLEVBQUU7UUFDZCw0RUFBNEU7UUFDNUUsd0JBQXdCO1FBQ3hCLE9BQU8sS0FBSyxDQUFDO0tBQ2Q7SUFDRCxZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLFlBQVksRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDO0lBQy9ELFlBQVksQ0FDUixFQUFFLEVBQ0YsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLG1CQUFtQixDQUN4QixHQUFHLEVBQUUsbUJBQW1CLEVBQUUsRUFBRSxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsaUJBQWlCLEVBQzVELGlCQUFpQixDQUFDLENBQUMsQ0FBQztJQUM1QixZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyx1QkFBdUIsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBQ3hELE9BQU8sSUFBSSxDQUFDO0FBQ2QsQ0FBQztBQUVELE1BQU0sVUFBVSxlQUFlLENBQzNCLEVBQXlCLEVBQUUsT0FBcUIsRUFBRSxXQUFtQjtJQUN2RSxtQkFBbUIsQ0FBQyxFQUFFLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFDckMsWUFBWSxDQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsYUFBYSxDQUFDLEVBQUUsQ0FBQyxRQUFRLEdBQUcsV0FBVyxDQUFDLENBQUMsQ0FBQztJQUNwRSxZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxXQUFXLENBQUMsRUFBRSxDQUFDLFVBQVUsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO0FBQ2pFLENBQUM7QUFFRCxNQUFNLFVBQVUsaUJBQWlCLENBQzdCLEVBQXlCLEVBQUUsV0FBbUI7SUFDaEQsbUJBQW1CLENBQUMsRUFBRSxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBQ3JDLFlBQVksQ0FBQyxFQUFFLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLGFBQWEsQ0FBQyxFQUFFLENBQUMsUUFBUSxHQUFHLFdBQVcsQ0FBQyxDQUFDLENBQUM7SUFDcEUsWUFBWSxDQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsV0FBVyxDQUFDLEVBQUUsQ0FBQyxVQUFVLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQztBQUM5RCxDQUFDO0FBRUQsTUFBTSxVQUFVLGdDQUFnQyxDQUM1QyxFQUF5QixFQUFFLE9BQXFCLEVBQ2hELFdBQW1CO0lBQ3JCLE9BQU8sV0FBVyxDQUNkLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsa0JBQWtCLENBQUMsT0FBTyxFQUFFLFdBQVcsQ0FBQyxFQUNyRCxXQUFXLEdBQUcsV0FBVyxHQUFHLDJCQUEyQixDQUFDLENBQUM7QUFDL0QsQ0FBQztBQUVELE1BQU0sVUFBVSx5QkFBeUIsQ0FDckMsRUFBeUIsRUFBRSxPQUFxQixFQUNoRCxXQUFtQjtJQUNyQixPQUFPLEVBQUUsQ0FBQyxrQkFBa0IsQ0FBQyxPQUFPLEVBQUUsV0FBVyxDQUFDLENBQUM7QUFDckQsQ0FBQztBQUVELE1BQU0sVUFBVSxrQ0FBa0MsQ0FDOUMsRUFBeUIsRUFBRSxPQUFxQixFQUNoRCxzQkFBNEMsRUFBRSxXQUFtQjtJQUNuRSxZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLGVBQWUsQ0FBQyxFQUFFLEVBQUUsT0FBTyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUM7SUFDbEUsWUFBWSxDQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsU0FBUyxDQUFDLHNCQUFzQixFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUM7QUFDNUUsQ0FBQztBQUVELE1BQU0sVUFBVSx1QkFBdUIsQ0FBQyxFQUF5QjtJQUMvRCxZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxlQUFlLENBQUMsRUFBRSxDQUFDLFdBQVcsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO0lBQ2pFLFlBQVksQ0FBQyxFQUFFLEVBQUUsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLFFBQVEsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxNQUFNLENBQUMsS0FBSyxFQUFFLEVBQUUsQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQztJQUM3RSxZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxPQUFPLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsTUFBTSxDQUFDLEtBQUssRUFBRSxFQUFFLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUM7QUFDOUUsQ0FBQztBQUVELE1BQU0sVUFBVSw2QkFBNkIsQ0FDekMsRUFBeUIsRUFBRSxPQUFxQixFQUNoRCxXQUE2QjtJQUMvQixZQUFZLENBQUMsRUFBRSxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxlQUFlLENBQUMsRUFBRSxDQUFDLFdBQVcsRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDO0lBQ3hFLFlBQVksQ0FDUixFQUFFLEVBQ0YsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLG9CQUFvQixDQUN6QixFQUFFLENBQUMsV0FBVyxFQUFFLEVBQUUsQ0FBQyxpQkFBaUIsRUFBRSxFQUFFLENBQUMsVUFBVSxFQUFFLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQzVFLENBQUM7QUFFRCxNQUFNLFVBQVUsaUNBQWlDLENBQzdDLEVBQXlCLEVBQUUsV0FBNkI7SUFDMUQsWUFBWSxDQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsQ0FBQyxFQUFFLENBQUMsZUFBZSxDQUFDLEVBQUUsQ0FBQyxXQUFXLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQztJQUN4RSxZQUFZLENBQ1IsRUFBRSxFQUNGLEdBQUcsRUFBRSxDQUFDLEVBQUUsQ0FBQyxvQkFBb0IsQ0FDekIsRUFBRSxDQUFDLFdBQVcsRUFBRSxFQUFFLENBQUMsaUJBQWlCLEVBQUUsRUFBRSxDQUFDLFVBQVUsRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUN6RSxDQUFDO0FBRUQsTUFBTSxVQUFVLG1CQUFtQixDQUFDLEVBQXlCO0lBQzNELE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxzQkFBc0IsQ0FBQyxFQUFFLENBQUMsV0FBVyxDQUFDLENBQUM7SUFDekQsSUFBSSxNQUFNLEtBQUssRUFBRSxDQUFDLG9CQUFvQixFQUFFO1FBQ3RDLE1BQU0sSUFBSSxLQUFLLENBQ1gsNkJBQTZCLEdBQUcsMEJBQTBCLENBQUMsRUFBRSxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUM7S0FDN0U7QUFDSCxDQUFDO0FBRUQsTUFBTSxVQUFVLDBCQUEwQixDQUN0QyxFQUF5QixFQUFFLE1BQWM7SUFDM0MsUUFBUSxNQUFNLEVBQUU7UUFDZCxLQUFLLEVBQUUsQ0FBQyxpQ0FBaUM7WUFDdkMsT0FBTyxtQ0FBbUMsQ0FBQztRQUM3QyxLQUFLLEVBQUUsQ0FBQyx5Q0FBeUM7WUFDL0MsT0FBTywyQ0FBMkMsQ0FBQztRQUNyRCxLQUFLLEVBQUUsQ0FBQyxpQ0FBaUM7WUFDdkMsT0FBTyxtQ0FBbUMsQ0FBQztRQUM3QyxLQUFLLEVBQUUsQ0FBQyx1QkFBdUI7WUFDN0IsT0FBTyx5QkFBeUIsQ0FBQztRQUNuQztZQUNFLE9BQU8saUJBQWlCLE1BQU0sRUFBRSxDQUFDO0tBQ3BDO0FBQ0gsQ0FBQztBQUVELFNBQVMsV0FBVyxDQUNoQixFQUF5QixFQUFFLGFBQTZCLEVBQ3hELGNBQXNCO0lBQ3hCLE1BQU0sT0FBTyxHQUFXLFlBQVksQ0FBQyxFQUFFLEVBQUUsR0FBRyxFQUFFLENBQUMsYUFBYSxFQUFFLENBQUMsQ0FBQztJQUNoRSxJQUFJLE9BQU8sSUFBSSxJQUFJLEVBQUU7UUFDbkIsTUFBTSxJQUFJLEtBQUssQ0FBQyxjQUFjLENBQUMsQ0FBQztLQUNqQztJQUNELE9BQU8sT0FBTyxDQUFDO0FBQ2pCLENBQUM7QUFFRCxTQUFTLG1CQUFtQixDQUFDLEVBQXlCLEVBQUUsV0FBbUI7SUFDekUsTUFBTSxjQUFjLEdBQUcsRUFBRSxDQUFDLGdDQUFnQyxHQUFHLENBQUMsQ0FBQztJQUMvRCxNQUFNLGFBQWEsR0FBRyxXQUFXLEdBQUcsRUFBRSxDQUFDLFFBQVEsQ0FBQztJQUNoRCxJQUFJLGFBQWEsR0FBRyxFQUFFLENBQUMsUUFBUSxJQUFJLGFBQWEsR0FBRyxjQUFjLEVBQUU7UUFDakUsTUFBTSxnQkFBZ0IsR0FBRywyQkFBMkIsY0FBYyxHQUFHLENBQUM7UUFDdEUsTUFBTSxJQUFJLEtBQUssQ0FBQywwQkFBMEIsZ0JBQWdCLEdBQUcsQ0FBQyxDQUFDO0tBQ2hFO0FBQ0gsQ0FBQztBQUVELE1BQU0sVUFBVSxXQUFXLENBQUMsS0FBZSxFQUFFLFVBQVUsR0FBRyxDQUFDO0lBQ3pELE9BQU8sSUFBSSxDQUFDLGFBQWEsQ0FBQyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxLQUFLLENBQUMsTUFBTSxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUM7QUFDdkUsQ0FBQztBQUVELE1BQU0sVUFBVSxXQUFXLENBQUMsS0FBZTtJQUN6QyxJQUFJLEtBQUssQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1FBQ3RCLE1BQU0sS0FBSyxDQUFDLHNEQUFzRCxDQUFDLENBQUM7S0FDckU7SUFFRCxPQUFPO1FBQ0wsS0FBSyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsS0FBSyxDQUFDLEtBQUssQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDO0tBQ3hFLENBQUM7QUFDSixDQUFDO0FBRUQsTUFBTSxVQUFVLFlBQVksQ0FBQyxLQUFlO0lBQzFDLElBQUksU0FBUyxHQUE2QixDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDcEQsTUFBTSxRQUFRLEdBQUcsS0FBSyxDQUFDLE1BQU0sS0FBSyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsTUFBTSxLQUFLLENBQUMsSUFBSSxLQUFLLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7SUFDOUUsSUFBSSxDQUFDLFFBQVEsRUFBRTtRQUNiLFNBQVM7WUFDTCxDQUFDLFdBQVcsQ0FBQyxLQUFLLENBQUMsRUFBRSxHQUFHLFdBQVcsQ0FBQyxLQUFLLENBQUMsQ0FBNkIsQ0FBQztLQUM3RTtJQUNELE9BQU8sU0FBUyxDQUFDO0FBQ25CLENBQUM7QUFFRCxNQUFNLFVBQVUsK0JBQStCLENBQzNDLFFBQWtCLEVBQUUsUUFBUSxHQUFHLEtBQUs7SUFDdEMsSUFBSSxVQUFVLEdBQUcsR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLHdCQUF3QixDQUFDLENBQUM7SUFDM0QsSUFBSSxRQUFRLEVBQUU7UUFDWixVQUFVLEdBQUcsVUFBVSxHQUFHLENBQUMsQ0FBQztRQUU1Qiw0RUFBNEU7UUFDNUUsMEVBQTBFO1FBQzFFLDJFQUEyRTtRQUMzRSwwRUFBMEU7UUFDMUUsZ0VBQWdFO1FBQ2hFLFFBQVEsR0FBRyxRQUFRLENBQUMsR0FBRyxDQUNuQixDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsSUFBSSxRQUFRLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQyxDQUFDO1lBQ2hDLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3JDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRXJCLHNFQUFzRTtRQUN0RSxVQUFVO1FBQ1YsSUFBSSxRQUFRLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRTtZQUN6QixRQUFRLEdBQUcsQ0FBQyxDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FDN0I7S0FDRjtJQUVELDRFQUE0RTtJQUM1RSxJQUFJLFFBQVEsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1FBQ3pCLE1BQU0sYUFBYSxHQUFHLElBQUksQ0FBQyxZQUFZLENBQUMsUUFBUSxDQUFDLENBQUM7UUFDbEQsUUFBUSxHQUFHLGFBQWEsQ0FBQyxRQUFRLENBQUM7S0FDbkM7SUFFRCxJQUFJLElBQUksR0FBRyxJQUFJLENBQUMsYUFBYSxDQUFDLFFBQVEsQ0FBQyxDQUFDO0lBQ3hDLElBQUksUUFBUSxDQUFDLE1BQU0sSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLFVBQVUsRUFBRTtRQUM5QyxPQUFPLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDO0tBQ2xCO1NBQU0sSUFDSCxRQUFRLENBQUMsTUFBTSxLQUFLLENBQUMsSUFBSSxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksVUFBVTtRQUNsRCxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksVUFBVSxFQUFFO1FBQzdCLE9BQU8sUUFBNEIsQ0FBQztLQUNyQztTQUFNLElBQ0gsUUFBUSxDQUFDLE1BQU0sS0FBSyxDQUFDLElBQUksUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxVQUFVO1FBQ2hFLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxVQUFVLEVBQUU7UUFDN0IsT0FBTyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7S0FDakQ7U0FBTSxJQUNILFFBQVEsQ0FBQyxNQUFNLEtBQUssQ0FBQyxJQUFJLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxVQUFVO1FBQ2xELFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksVUFBVSxFQUFFO1FBQzNDLE9BQU8sQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQ2pEO1NBQU0sSUFDSCxRQUFRLENBQUMsTUFBTSxLQUFLLENBQUM7UUFDckIsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksVUFBVTtRQUNyRCxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksVUFBVSxFQUFFO1FBQzdCLE9BQU8sQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFD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QUFJLEVBQUU7WUFDYixJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxLQUFLLEtBQUssV0FBVyxFQUN2QixHQUFHLEVBQUUsQ0FBQyxHQUFHLE1BQU0sc0NBQXNDO2dCQUNqRCx1QkFBdUIsQ0FBQyxDQUFDO1NBQ2xDO0lBQ0gsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2VudiwgVGVuc29ySW5mbywgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtnZXRXZWJHTENvbnRleHR9IGZyb20gJy4vY2FudmFzX3V0aWwnO1xuaW1wb3J0IHtnZXRUZXh0dXJlQ29uZmlnfSBmcm9tICcuL3RleF91dGlsJztcblxuZXhwb3J0IGZ1bmN0aW9uIGNhbGxBbmRDaGVjazxUPihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBmdW5jOiAoKSA9PiBUKTogVCB7XG4gIGNvbnN0IHJldHVyblZhbHVlID0gZnVuYygpO1xuICBpZiAoZW52KCkuZ2V0Qm9vbCgnREVCVUcnKSkge1xuICAgIGNoZWNrV2ViR0xFcnJvcihnbCk7XG4gIH1cbiAgcmV0dXJuIHJldHVyblZhbHVlO1xufVxuXG5mdW5jdGlvbiBjaGVja1dlYkdMRXJyb3IoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICBjb25zdCBlcnJvciA9IGdsLmdldEVycm9yKCk7XG4gIGlmIChlcnJvciAhPT0gZ2wuTk9fRVJST1IpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ1dlYkdMIEVycm9yOiAnICsgZ2V0V2ViR0xFcnJvck1lc3NhZ2UoZ2wsIGVycm9yKSk7XG4gIH1cbn1cblxuLy8gaHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvSGFsZi1wcmVjaXNpb25fZmxvYXRpbmctcG9pbnRfZm9ybWF0XG5jb25zdCBNSU5fRkxPQVQxNiA9IDUuOTZlLTg7XG5jb25zdCBNQVhfRkxPQVQxNiA9IDY1NTA0O1xuXG5leHBvcnQgZnVuY3Rpb24gY2FuQmVSZXByZXNlbnRlZChudW06IG51bWJlcik6IGJvb2xlYW4ge1xuICBpZiAoZW52KCkuZ2V0Qm9vbCgnV0VCR0xfUkVOREVSX0ZMT0FUMzJfRU5BQkxFRCcpIHx8IG51bSA9PT0gMCB8fFxuICAgICAgKE1JTl9GTE9BVDE2IDwgTWF0aC5hYnMobnVtKSAmJiBNYXRoLmFicyhudW0pIDwgTUFYX0ZMT0FUMTYpKSB7XG4gICAgcmV0dXJuIHRydWU7XG4gIH1cbiAgcmV0dXJuIGZhbHNlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0V2ViR0xFcnJvck1lc3NhZ2UoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgc3RhdHVzOiBudW1iZXIpOiBzdHJpbmcge1xuICBzd2l0Y2ggKHN0YXR1cykge1xuICAgIGNhc2UgZ2wuTk9fRVJST1I6XG4gICAgICByZXR1cm4gJ05PX0VSUk9SJztcbiAgICBjYXNlIGdsLklOVkFMSURfRU5VTTpcbiAgICAgIHJldHVybiAnSU5WQUxJRF9FTlVNJztcbiAgICBjYXNlIGdsLklOVkFMSURfVkFMVUU6XG4gICAgICByZXR1cm4gJ0lOVkFMSURfVkFMVUUnO1xuICAgIGNhc2UgZ2wuSU5WQUxJRF9PUEVSQVRJT046XG4gICAgICByZXR1cm4gJ0lOVkFMSURfT1BFUkFUSU9OJztcbiAgICBjYXNlIGdsLklOVkFMSURfRlJBTUVCVUZGRVJfT1BFUkFUSU9OOlxuICAgICAgcmV0dXJuICdJTlZBTElEX0ZSQU1FQlVGRkVSX09QRVJBVElPTic7XG4gICAgY2FzZSBnbC5PVVRfT0ZfTUVNT1JZOlxuICAgICAgcmV0dXJuICdPVVRfT0ZfTUVNT1JZJztcbiAgICBjYXNlIGdsLkNPTlRFWFRfTE9TVF9XRUJHTDpcbiAgICAgIHJldHVybiAnQ09OVEVYVF9MT1NUX1dFQkdMJztcbiAgICBkZWZhdWx0OlxuICAgICAgcmV0dXJuIGBVbmtub3duIGVycm9yIGNvZGUgJHtzdGF0dXN9YDtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RXh0ZW5zaW9uT3JUaHJvdyhcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBleHRlbnNpb25OYW1lOiBzdHJpbmcpOiB7fSB7XG4gIHJldHVybiB0aHJvd0lmTnVsbDx7fT4oXG4gICAgICBnbCwgKCkgPT4gZ2wuZ2V0RXh0ZW5zaW9uKGV4dGVuc2lvbk5hbWUpLFxuICAgICAgJ0V4dGVuc2lvbiBcIicgKyBleHRlbnNpb25OYW1lICsgJ1wiIG5vdCBzdXBwb3J0ZWQgb24gdGhpcyBicm93c2VyLicpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlVmVydGV4U2hhZGVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHZlcnRleFNoYWRlclNvdXJjZTogc3RyaW5nKTogV2ViR0xTaGFkZXIge1xuICBjb25zdCB2ZXJ0ZXhTaGFkZXI6IFdlYkdMU2hhZGVyID0gdGhyb3dJZk51bGw8V2ViR0xTaGFkZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZVNoYWRlcihnbC5WRVJURVhfU0hBREVSKSxcbiAgICAgICdVbmFibGUgdG8gY3JlYXRlIHZlcnRleCBXZWJHTFNoYWRlci4nKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5zaGFkZXJTb3VyY2UodmVydGV4U2hhZGVyLCB2ZXJ0ZXhTaGFkZXJTb3VyY2UpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5jb21waWxlU2hhZGVyKHZlcnRleFNoYWRlcikpO1xuICBpZiAoZ2wuZ2V0U2hhZGVyUGFyYW1ldGVyKHZlcnRleFNoYWRlciwgZ2wuQ09NUElMRV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFNoYWRlckluZm9Mb2codmVydGV4U2hhZGVyKSk7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gY29tcGlsZSB2ZXJ0ZXggc2hhZGVyLicpO1xuICB9XG4gIHJldHVybiB2ZXJ0ZXhTaGFkZXI7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVGcmFnbWVudFNoYWRlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBmcmFnbWVudFNoYWRlclNvdXJjZTogc3RyaW5nKTogV2ViR0xTaGFkZXIge1xuICBjb25zdCBmcmFnbWVudFNoYWRlcjogV2ViR0xTaGFkZXIgPSB0aHJvd0lmTnVsbDxXZWJHTFNoYWRlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlU2hhZGVyKGdsLkZSQUdNRU5UX1NIQURFUiksXG4gICAgICAnVW5hYmxlIHRvIGNyZWF0ZSBmcmFnbWVudCBXZWJHTFNoYWRlci4nKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5zaGFkZXJTb3VyY2UoZnJhZ21lbnRTaGFkZXIsIGZyYWdtZW50U2hhZGVyU291cmNlKSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuY29tcGlsZVNoYWRlcihmcmFnbWVudFNoYWRlcikpO1xuICBpZiAoZW52KCkuZ2V0KCdFTkdJTkVfQ09NUElMRV9PTkxZJykpIHtcbiAgICByZXR1cm4gZnJhZ21lbnRTaGFkZXI7XG4gIH1cbiAgaWYgKGdsLmdldFNoYWRlclBhcmFtZXRlcihmcmFnbWVudFNoYWRlciwgZ2wuQ09NUElMRV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGxvZ1NoYWRlclNvdXJjZUFuZEluZm9Mb2coXG4gICAgICAgIGZyYWdtZW50U2hhZGVyU291cmNlLCBnbC5nZXRTaGFkZXJJbmZvTG9nKGZyYWdtZW50U2hhZGVyKSk7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gY29tcGlsZSBmcmFnbWVudCBzaGFkZXIuJyk7XG4gIH1cbiAgcmV0dXJuIGZyYWdtZW50U2hhZGVyO1xufVxuXG5jb25zdCBsaW5lTnVtYmVyUmVnZXggPSAvRVJST1I6IFswLTldKzooWzAtOV0rKTovZztcbmV4cG9ydCBmdW5jdGlvbiBsb2dTaGFkZXJTb3VyY2VBbmRJbmZvTG9nKFxuICAgIHNoYWRlclNvdXJjZTogc3RyaW5nLCBzaGFkZXJJbmZvTG9nOiBzdHJpbmcpIHtcbiAgY29uc3QgbGluZU51bWJlclJlZ2V4UmVzdWx0ID0gbGluZU51bWJlclJlZ2V4LmV4ZWMoc2hhZGVySW5mb0xvZyk7XG4gIGlmIChsaW5lTnVtYmVyUmVnZXhSZXN1bHQgPT0gbnVsbCkge1xuICAgIGNvbnNvbGUubG9nKGBDb3VsZG4ndCBwYXJzZSBsaW5lIG51bWJlciBpbiBlcnJvcjogJHtzaGFkZXJJbmZvTG9nfWApO1xuICAgIGNvbnNvbGUubG9nKHNoYWRlclNvdXJjZSk7XG4gICAgcmV0dXJuO1xuICB9XG5cbiAgY29uc3QgbGluZU51bWJlciA9ICtsaW5lTnVtYmVyUmVnZXhSZXN1bHRbMV07XG5cbiAgY29uc3Qgc2hhZGVyTGluZXMgPSBzaGFkZXJTb3VyY2Uuc3BsaXQoJ1xcbicpO1xuICBjb25zdCBwYWQgPSBzaGFkZXJMaW5lcy5sZW5ndGgudG9TdHJpbmcoKS5sZW5ndGggKyAyO1xuICBjb25zdCBsaW5lc1dpdGhMaW5lTnVtYmVycyA9IHNoYWRlckxpbmVzLm1hcChcbiAgICAgIChsaW5lLCBsaW5lTnVtYmVyKSA9PlxuICAgICAgICAgIHV0aWwucmlnaHRQYWQoKGxpbmVOdW1iZXIgKyAxKS50b1N0cmluZygpLCBwYWQpICsgbGluZSk7XG4gIGxldCBtYXhMaW5lTGVuZ3RoID0gMDtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBsaW5lc1dpdGhMaW5lTnVtYmVycy5sZW5ndGg7IGkrKykge1xuICAgIG1heExpbmVMZW5ndGggPSBNYXRoLm1heChsaW5lc1dpdGhMaW5lTnVtYmVyc1tpXS5sZW5ndGgsIG1heExpbmVMZW5ndGgpO1xuICB9XG5cbiAgY29uc3QgYmVmb3JlRXJyb3JMaW5lcyA9IGxpbmVzV2l0aExpbmVOdW1iZXJzLnNsaWNlKDAsIGxpbmVOdW1iZXIgLSAxKTtcbiAgY29uc3QgZXJyb3JMaW5lID0gbGluZXNXaXRoTGluZU51bWJlcnMuc2xpY2UobGluZU51bWJlciAtIDEsIGxpbmVOdW1iZXIpO1xuICBjb25zdCBhZnRlckVycm9yTGluZXMgPSBsaW5lc1dpdGhMaW5lTnVtYmVycy5zbGljZShsaW5lTnVtYmVyKTtcblxuICBjb25zb2xlLmxvZyhiZWZvcmVFcnJvckxpbmVzLmpvaW4oJ1xcbicpKTtcbiAgY29uc29sZS5sb2coc2hhZGVySW5mb0xvZy5zcGxpdCgnXFxuJylbMF0pO1xuICBjb25zb2xlLmxvZyhcbiAgICAgIGAlYyAke3V0aWwucmlnaHRQYWQoZXJyb3JMaW5lWzBdLCBtYXhMaW5lTGVuZ3RoKX1gLFxuICAgICAgJ2JvcmRlcjoxcHggc29saWQgcmVkOyBiYWNrZ3JvdW5kLWNvbG9yOiNlM2QyZDI7IGNvbG9yOiNhNjE3MTcnKTtcbiAgY29uc29sZS5sb2coYWZ0ZXJFcnJvckxpbmVzLmpvaW4oJ1xcbicpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVByb2dyYW0oZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMUHJvZ3JhbSB7XG4gIHJldHVybiB0aHJvd0lmTnVsbDxXZWJHTFByb2dyYW0+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZVByb2dyYW0oKSwgJ1VuYWJsZSB0byBjcmVhdGUgV2ViR0xQcm9ncmFtLicpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbGlua1Byb2dyYW0oZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtKSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wubGlua1Byb2dyYW0ocHJvZ3JhbSkpO1xuICBpZiAoZW52KCkuZ2V0KCdFTkdJTkVfQ09NUElMRV9PTkxZJykpIHtcbiAgICByZXR1cm47XG4gIH1cbiAgaWYgKGdsLmdldFByb2dyYW1QYXJhbWV0ZXIocHJvZ3JhbSwgZ2wuTElOS19TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFByb2dyYW1JbmZvTG9nKHByb2dyYW0pKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBsaW5rIHZlcnRleCBhbmQgZnJhZ21lbnQgc2hhZGVycy4nKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gdmFsaWRhdGVQcm9ncmFtKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSkge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnZhbGlkYXRlUHJvZ3JhbShwcm9ncmFtKSk7XG4gIGlmIChnbC5nZXRQcm9ncmFtUGFyYW1ldGVyKHByb2dyYW0sIGdsLlZBTElEQVRFX1NUQVRVUykgPT09IGZhbHNlKSB7XG4gICAgY29uc29sZS5sb2coZ2wuZ2V0UHJvZ3JhbUluZm9Mb2cocHJvZ3JhbSkpO1xuICAgIHRocm93IG5ldyBFcnJvcignU2hhZGVyIHByb2dyYW0gdmFsaWRhdGlvbiBmYWlsZWQuJyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN0YXRpY1ZlcnRleEJ1ZmZlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBkYXRhOiBGbG9hdDMyQXJyYXkpOiBXZWJHTEJ1ZmZlciB7XG4gIGNvbnN0IGJ1ZmZlcjogV2ViR0xCdWZmZXIgPSB0aHJvd0lmTnVsbDxXZWJHTEJ1ZmZlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlQnVmZmVyKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMQnVmZmVyJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5BUlJBWV9CVUZGRVIsIGJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJ1ZmZlckRhdGEoZ2wuQVJSQVlfQlVGRkVSLCBkYXRhLCBnbC5TVEFUSUNfRFJBVykpO1xuICByZXR1cm4gYnVmZmVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU3RhdGljSW5kZXhCdWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZGF0YTogVWludDE2QXJyYXkpOiBXZWJHTEJ1ZmZlciB7XG4gIGNvbnN0IGJ1ZmZlcjogV2ViR0xCdWZmZXIgPSB0aHJvd0lmTnVsbDxXZWJHTEJ1ZmZlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlQnVmZmVyKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMQnVmZmVyJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5FTEVNRU5UX0FSUkFZX0JVRkZFUiwgYnVmZmVyKSk7XG4gIGNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC5idWZmZXJEYXRhKGdsLkVMRU1FTlRfQVJSQVlfQlVGRkVSLCBkYXRhLCBnbC5TVEFUSUNfRFJBVykpO1xuICByZXR1cm4gYnVmZmVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0TnVtQ2hhbm5lbHMoKTogbnVtYmVyIHtcbiAgaWYgKGVudigpLmdldE51bWJlcignV0VCR0xfVkVSU0lPTicpID09PSAyKSB7XG4gICAgcmV0dXJuIDE7XG4gIH1cbiAgcmV0dXJuIDQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVUZXh0dXJlKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpOiBXZWJHTFRleHR1cmUge1xuICByZXR1cm4gdGhyb3dJZk51bGw8V2ViR0xUZXh0dXJlPihcbiAgICAgIGdsLCAoKSA9PiBnbC5jcmVhdGVUZXh0dXJlKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMVGV4dHVyZS4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHZhbGlkYXRlVGV4dHVyZVNpemUod2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIpIHtcbiAgY29uc3QgbWF4VGV4dHVyZVNpemUgPSBlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX01BWF9URVhUVVJFX1NJWkUnKTtcbiAgaWYgKCh3aWR0aCA8PSAwKSB8fCAoaGVpZ2h0IDw9IDApKSB7XG4gICAgY29uc3QgcmVxdWVzdGVkID0gYFske3dpZHRofXgke2hlaWdodH1dYDtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ1JlcXVlc3RlZCB0ZXh0dXJlIHNpemUgJyArIHJlcXVlc3RlZCArICcgaXMgaW52YWxpZC4nKTtcbiAgfVxuICBpZiAoKHdpZHRoID4gbWF4VGV4dHVyZVNpemUpIHx8IChoZWlnaHQgPiBtYXhUZXh0dXJlU2l6ZSkpIHtcbiAgICBjb25zdCByZXF1ZXN0ZWQgPSBgWyR7d2lkdGh9eCR7aGVpZ2h0fV1gO1xuICAgIGNvbnN0IG1heCA9IGBbJHttYXhUZXh0dXJlU2l6ZX14JHttYXhUZXh0dXJlU2l6ZX1dYDtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdSZXF1ZXN0ZWQgdGV4dHVyZSBzaXplICcgKyByZXF1ZXN0ZWQgK1xuICAgICAgICAnIGdyZWF0ZXIgdGhhbiBXZWJHTCBtYXhpbXVtIG9uIHRoaXMgYnJvd3NlciAvIEdQVSAnICsgbWF4ICsgJy4nKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlRnJhbWVidWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMRnJhbWVidWZmZXIge1xuICByZXR1cm4gdGhyb3dJZk51bGw8V2ViR0xGcmFtZWJ1ZmZlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlRnJhbWVidWZmZXIoKSwgJ1VuYWJsZSB0byBjcmVhdGUgV2ViR0xGcmFtZWJ1ZmZlci4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRWZXJ0ZXhCdWZmZXJUb1Byb2dyYW1BdHRyaWJ1dGUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtLCBhdHRyaWJ1dGU6IHN0cmluZyxcbiAgICBidWZmZXI6IFdlYkdMQnVmZmVyLCBhcnJheUVudHJpZXNQZXJJdGVtOiBudW1iZXIsIGl0ZW1TdHJpZGVJbkJ5dGVzOiBudW1iZXIsXG4gICAgaXRlbU9mZnNldEluQnl0ZXM6IG51bWJlcik6IGJvb2xlYW4ge1xuICBjb25zdCBsb2MgPSBnbC5nZXRBdHRyaWJMb2NhdGlvbihwcm9ncmFtLCBhdHRyaWJ1dGUpO1xuICBpZiAobG9jID09PSAtMSkge1xuICAgIC8vIFRoZSBHUFUgY29tcGlsZXIgZGVjaWRlZCB0byBzdHJpcCBvdXQgdGhpcyBhdHRyaWJ1dGUgYmVjYXVzZSBpdCdzIHVudXNlZCxcbiAgICAvLyB0aHVzIG5vIG5lZWQgdG8gYmluZC5cbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkFSUkFZX0JVRkZFUiwgYnVmZmVyKSk7XG4gIGNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wudmVydGV4QXR0cmliUG9pbnRlcihcbiAgICAgICAgICBsb2MsIGFycmF5RW50cmllc1Blckl0ZW0sIGdsLkZMT0FULCBmYWxzZSwgaXRlbVN0cmlkZUluQnl0ZXMsXG4gICAgICAgICAgaXRlbU9mZnNldEluQnl0ZXMpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5lbmFibGVWZXJ0ZXhBdHRyaWJBcnJheShsb2MpKTtcbiAgcmV0dXJuIHRydWU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kVGV4dHVyZVVuaXQoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIHZhbGlkYXRlVGV4dHVyZVVuaXQoZ2wsIHRleHR1cmVVbml0KTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5hY3RpdmVUZXh0dXJlKGdsLlRFWFRVUkUwICsgdGV4dHVyZVVuaXQpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0ZXh0dXJlKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1bmJpbmRUZXh0dXJlVW5pdChcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIHZhbGlkYXRlVGV4dHVyZVVuaXQoZ2wsIHRleHR1cmVVbml0KTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5hY3RpdmVUZXh0dXJlKGdsLlRFWFRVUkUwICsgdGV4dHVyZVVuaXQpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRQcm9ncmFtVW5pZm9ybUxvY2F0aW9uT3JUaHJvdyhcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sXG4gICAgdW5pZm9ybU5hbWU6IHN0cmluZyk6IFdlYkdMVW5pZm9ybUxvY2F0aW9uIHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPFdlYkdMVW5pZm9ybUxvY2F0aW9uPihcbiAgICAgIGdsLCAoKSA9PiBnbC5nZXRVbmlmb3JtTG9jYXRpb24ocHJvZ3JhbSwgdW5pZm9ybU5hbWUpLFxuICAgICAgJ3VuaWZvcm0gXCInICsgdW5pZm9ybU5hbWUgKyAnXCIgbm90IHByZXNlbnQgaW4gcHJvZ3JhbS4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFByb2dyYW1Vbmlmb3JtTG9jYXRpb24oXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtLFxuICAgIHVuaWZvcm1OYW1lOiBzdHJpbmcpOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbiB7XG4gIHJldHVybiBnbC5nZXRVbmlmb3JtTG9jYXRpb24ocHJvZ3JhbSwgdW5pZm9ybU5hbWUpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZFRleHR1cmVUb1Byb2dyYW1Vbmlmb3JtU2FtcGxlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsXG4gICAgdW5pZm9ybVNhbXBsZXJMb2NhdGlvbjogV2ViR0xVbmlmb3JtTG9jYXRpb24sIHRleHR1cmVVbml0OiBudW1iZXIpIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBiaW5kVGV4dHVyZVVuaXQoZ2wsIHRleHR1cmUsIHRleHR1cmVVbml0KSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wudW5pZm9ybTFpKHVuaWZvcm1TYW1wbGVyTG9jYXRpb24sIHRleHR1cmVVbml0KSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kQ2FudmFzVG9GcmFtZWJ1ZmZlcihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBudWxsKSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wudmlld3BvcnQoMCwgMCwgZ2wuY2FudmFzLndpZHRoLCBnbC5jYW52YXMuaGVpZ2h0KSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuc2Npc3NvcigwLCAwLCBnbC5jYW52YXMud2lkdGgsIGdsLmNhbnZhcy5oZWlnaHQpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRDb2xvclRleHR1cmVUb0ZyYW1lYnVmZmVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSxcbiAgICBmcmFtZWJ1ZmZlcjogV2ViR0xGcmFtZWJ1ZmZlcikge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgZnJhbWVidWZmZXIpKTtcbiAgY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC5mcmFtZWJ1ZmZlclRleHR1cmUyRChcbiAgICAgICAgICBnbC5GUkFNRUJVRkZFUiwgZ2wuQ09MT1JfQVRUQUNITUVOVDAsIGdsLlRFWFRVUkVfMkQsIHRleHR1cmUsIDApKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHVuYmluZENvbG9yVGV4dHVyZUZyb21GcmFtZWJ1ZmZlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBmcmFtZWJ1ZmZlcjogV2ViR0xGcmFtZWJ1ZmZlcikge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgZnJhbWVidWZmZXIpKTtcbiAgY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC5mcmFtZWJ1ZmZlclRleHR1cmUyRChcbiAgICAgICAgICBnbC5GUkFNRUJVRkZFUiwgZ2wuQ09MT1JfQVRUQUNITUVOVDAsIGdsLlRFWFRVUkVfMkQsIG51bGwsIDApKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHZhbGlkYXRlRnJhbWVidWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICBjb25zdCBzdGF0dXMgPSBnbC5jaGVja0ZyYW1lYnVmZmVyU3RhdHVzKGdsLkZSQU1FQlVGRkVSKTtcbiAgaWYgKHN0YXR1cyAhPT0gZ2wuRlJBTUVCVUZGRVJfQ09NUExFVEUpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdFcnJvciBiaW5kaW5nIGZyYW1lYnVmZmVyOiAnICsgZ2V0RnJhbWVidWZmZXJFcnJvck1lc3NhZ2UoZ2wsIHN0YXR1cykpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFtZWJ1ZmZlckVycm9yTWVzc2FnZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBzdGF0dXM6IG51bWJlcik6IHN0cmluZyB7XG4gIHN3aXRjaCAoc3RhdHVzKSB7XG4gICAgY2FzZSBnbC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0FUVEFDSE1FTlQ6XG4gICAgICByZXR1cm4gJ0ZSQU1FQlVGRkVSX0lOQ09NUExFVEVfQVRUQUNITUVOVCc7XG4gICAgY2FzZSBnbC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX01JU1NJTkdfQVRUQUNITUVOVDpcbiAgICAgIHJldHVybiAnRlJBTUVCVUZGRVJfSU5DT01QTEVURV9NSVNTSU5HX0FUVEFDSE1FTlQnO1xuICAgIGNhc2UgZ2wuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9ESU1FTlNJT05TOlxuICAgICAgcmV0dXJuICdGUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0RJTUVOU0lPTlMnO1xuICAgIGNhc2UgZ2wuRlJBTUVCVUZGRVJfVU5TVVBQT1JURUQ6XG4gICAgICByZXR1cm4gJ0ZSQU1FQlVGRkVSX1VOU1VQUE9SVEVEJztcbiAgICBkZWZhdWx0OlxuICAgICAgcmV0dXJuIGB1bmtub3duIGVycm9yICR7c3RhdHVzfWA7XG4gIH1cbn1cblxuZnVuY3Rpb24gdGhyb3dJZk51bGw8VD4oXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcmV0dXJuVE9yTnVsbDogKCkgPT4gVCB8IG51bGwsXG4gICAgZmFpbHVyZU1lc3NhZ2U6IHN0cmluZyk6IFQge1xuICBjb25zdCB0T3JOdWxsOiBUfG51bGwgPSBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IHJldHVyblRPck51bGwoKSk7XG4gIGlmICh0T3JOdWxsID09IG51bGwpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoZmFpbHVyZU1lc3NhZ2UpO1xuICB9XG4gIHJldHVybiB0T3JOdWxsO1xufVxuXG5mdW5jdGlvbiB2YWxpZGF0ZVRleHR1cmVVbml0KGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmVVbml0OiBudW1iZXIpIHtcbiAgY29uc3QgbWF4VGV4dHVyZVVuaXQgPSBnbC5NQVhfQ09NQklORURfVEVYVFVSRV9JTUFHRV9VTklUUyAtIDE7XG4gIGNvbnN0IGdsVGV4dHVyZVVuaXQgPSB0ZXh0dXJlVW5pdCArIGdsLlRFWFRVUkUwO1xuICBpZiAoZ2xUZXh0dXJlVW5pdCA8IGdsLlRFWFRVUkUwIHx8IGdsVGV4dHVyZVVuaXQgPiBtYXhUZXh0dXJlVW5pdCkge1xuICAgIGNvbnN0IHRleHR1cmVVbml0UmFuZ2UgPSBgW2dsLlRFWFRVUkUwLCBnbC5URVhUVVJFJHttYXhUZXh0dXJlVW5pdH1dYDtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYHRleHR1cmVVbml0IG11c3QgYmUgaW4gJHt0ZXh0dXJlVW5pdFJhbmdlfS5gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0QmF0Y2hEaW0oc2hhcGU6IG51bWJlcltdLCBkaW1zVG9Ta2lwID0gMik6IG51bWJlciB7XG4gIHJldHVybiB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUuc2xpY2UoMCwgc2hhcGUubGVuZ3RoIC0gZGltc1RvU2tpcCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0Um93c0NvbHMoc2hhcGU6IG51bWJlcltdKTogW251bWJlciwgbnVtYmVyXSB7XG4gIGlmIChzaGFwZS5sZW5ndGggPT09IDApIHtcbiAgICB0aHJvdyBFcnJvcignQ2Fubm90IGdldCByb3dzIGFuZCBjb2x1bW5zIG9mIGFuIGVtcHR5IHNoYXBlIGFycmF5LicpO1xuICB9XG5cbiAgcmV0dXJuIFtcbiAgICBzaGFwZS5sZW5ndGggPiAxID8gc2hhcGVbc2hhcGUubGVuZ3RoIC0gMl0gOiAxLCBzaGFwZVtzaGFwZS5sZW5ndGggLSAxXVxuICBdO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0U2hhcGVBczNEKHNoYXBlOiBudW1iZXJbXSk6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSB7XG4gIGxldCBzaGFwZUFzM0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFsxLCAxLCAxXTtcbiAgY29uc3QgaXNTY2FsYXIgPSBzaGFwZS5sZW5ndGggPT09IDAgfHwgKHNoYXBlLmxlbmd0aCA9PT0gMSAmJiBzaGFwZVswXSA9PT0gMSk7XG4gIGlmICghaXNTY2FsYXIpIHtcbiAgICBzaGFwZUFzM0QgPVxuICAgICAgICBbZ2V0QmF0Y2hEaW0oc2hhcGUpLCAuLi5nZXRSb3dzQ29scyhzaGFwZSldIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgfVxuICByZXR1cm4gc2hhcGVBczNEO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0VGV4dHVyZVNoYXBlRnJvbUxvZ2ljYWxTaGFwZShcbiAgICBsb2dTaGFwZTogbnVtYmVyW10sIGlzUGFja2VkID0gZmFsc2UpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgbGV0IG1heFRleFNpemUgPSBlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX01BWF9URVhUVVJFX1NJWkUnKTtcbiAgaWYgKGlzUGFja2VkKSB7XG4gICAgbWF4VGV4U2l6ZSA9IG1heFRleFNpemUgKiAyO1xuXG4gICAgLy8gVGhpcyBsb2dpYyBlbnN1cmVzIHdlIGFjY3VyYXRlbHkgY291bnQgdGhlIG51bWJlciBvZiBwYWNrZWQgdGV4ZWxzIG5lZWRlZFxuICAgIC8vIHRvIGFjY29tbW9kYXRlIHRoZSB0ZW5zb3IuIFdlIGNhbiBvbmx5IHBhY2sgdmFsdWVzIGluIHRoZSBzYW1lIHRleGVsIGlmXG4gICAgLy8gdGhleSBhcmUgZnJvbSBhZGphY2VudCBwYWlycyBvZiByb3dzL2NvbHMgd2l0aGluIHRoZSBzYW1lIGJhdGNoLiBTbyBpZiBhXG4gICAgLy8gdGVuc29yIGhhcyAzIHJvd3MsIHdlIHByZXRlbmQgaXQgaGFzIDQgcm93cyBpbiBvcmRlciB0byBhY2NvdW50IGZvciB0aGVcbiAgICAvLyBmYWN0IHRoYXQgdGhlIHRleGVscyBjb250YWluaW5nIHRoZSB0aGlyZCByb3cgYXJlIGhhbGYgZW1wdHkuXG4gICAgbG9nU2hhcGUgPSBsb2dTaGFwZS5tYXAoXG4gICAgICAgIChkLCBpKSA9PiBpID49IGxvZ1NoYXBlLmxlbmd0aCAtIDIgP1xuICAgICAgICAgICAgdXRpbC5uZWFyZXN0TGFyZ2VyRXZlbihsb2dTaGFwZVtpXSkgOlxuICAgICAgICAgICAgbG9nU2hhcGVbaV0pO1xuXG4gICAgLy8gUGFja2VkIHRleHR1cmUgaGVpZ2h0IGlzIGF0IGxlYXN0IDIgKHRoZSBjaGFubmVsIGhlaWdodCBvZiBhIHNpbmdsZVxuICAgIC8vIHRleGVsKS5cbiAgICBpZiAobG9nU2hhcGUubGVuZ3RoID09PSAxKSB7XG4gICAgICBsb2dTaGFwZSA9IFsyLCBsb2dTaGFwZVswXV07XG4gICAgfVxuICB9XG5cbiAgLy8gSWYgbG9naWNhbCBzaGFwZSBpcyAyLCB3ZSBkb24ndCBzcXVlZXplLCBzaW5jZSB3ZSB3YW50IHRvIG1hdGNoIHBoeXNpY2FsLlxuICBpZiAobG9nU2hhcGUubGVuZ3RoICE9PSAyKSB7XG4gICAgY29uc3Qgc3F1ZWV6ZVJlc3VsdCA9IHV0aWwuc3F1ZWV6ZVNoYXBlKGxvZ1NoYXBlKTtcbiAgICBsb2dTaGFwZSA9IHNxdWVlemVSZXN1bHQubmV3U2hhcGU7XG4gIH1cblxuICBsZXQgc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShsb2dTaGFwZSk7XG4gIGlmIChsb2dTaGFwZS5sZW5ndGggPD0gMSAmJiBzaXplIDw9IG1heFRleFNpemUpIHtcbiAgICByZXR1cm4gWzEsIHNpemVdO1xuICB9IGVsc2UgaWYgKFxuICAgICAgbG9nU2hhcGUubGVuZ3RoID09PSAyICYmIGxvZ1NoYXBlWzBdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ1NoYXBlWzFdIDw9IG1heFRleFNpemUpIHtcbiAgICByZXR1cm4gbG9nU2hhcGUgYXMgW251bWJlciwgbnVtYmVyXTtcbiAgfSBlbHNlIGlmIChcbiAgICAgIGxvZ1NoYXBlLmxlbmd0aCA9PT0gMyAmJiBsb2dTaGFwZVswXSAqIGxvZ1NoYXBlWzFdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ1NoYXBlWzJdIDw9IG1heFRleFNpemUpIHtcbiAgICByZXR1cm4gW2xvZ1NoYXBlWzBdICogbG9nU2hhcGVbMV0sIGxvZ1NoYXBlWzJdXTtcbiAgfSBlbHNlIGlmIChcbiAgICAgIGxvZ1NoYXBlLmxlbmd0aCA9PT0gMyAmJiBsb2dTaGFwZVswXSA8PSBtYXhUZXhTaXplICYmXG4gICAgICBsb2dTaGFwZVsxXSAqIGxvZ1NoYXBlWzJdIDw9IG1heFRleFNpemUpIHtcbiAgICByZXR1cm4gW2xvZ1NoYXBlWzBdLCBsb2dTaGFwZVsxXSAqIGxvZ1NoYXBlWzJdXTtcbiAgfSBlbHNlIGlmIChcbiAgICAgIGxvZ1NoYXBlLmxlbmd0aCA9PT0gNCAmJlxuICAgICAgbG9nU2hhcGVbMF0gKiBsb2dTaGFwZVsxXSAqIGxvZ1NoYXBlWzJdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ1NoYXBlWzNdIDw9IG1heFRleFNpemUpIHtcbiAgICByZXR1cm4gW2xvZ1NoYXBlWzBdICogbG9nU2hhcGVbMV0gKiBsb2dTaGFwZVsyXSwgbG9nU2hhcGVbM11dO1xuICB9IGVsc2UgaWYgKFxuICAgICAgbG9nU2hhcGUubGVuZ3RoID09PSA0ICYmIGxvZ1NoYXBlWzBdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ1NoYXBlWzFdICogbG9nU2hhcGVbMl0gKiBsb2dTaGFwZVszXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgcmV0dXJuIFtsb2dTaGFwZVswXSwgbG9nU2hhcGVbMV0gKiBsb2dTaGFwZVsyXSAqIGxvZ1NoYXBlWzNdXTtcbiAgfSBlbHNlIHtcbiAgICBpZiAoaXNQYWNrZWQpIHtcbiAgICAgIC8vIEZvciBwYWNrZWQgdGV4dHVyZXMgc2l6ZSBlcXVhbHMgdGhlIG51bWJlciBvZiBjaGFubmVscyByZXF1aXJlZCB0b1xuICAgICAgLy8gYWNjb21tb2RhdGUgdGhlIHRleHR1cmUgZGF0YS4gSG93ZXZlciBpbiBvcmRlciB0byBzcXVhcmlmeSBzdWNoIHRoYXRcbiAgICAgIC8vIGlubmVyIGRpbWVuc2lvbnMgc3RheSBldmVuLCB3ZSByZXdyaXRlIHNpemUgdG8gZXF1YWwgdGhlIG51bWJlciBvZlxuICAgICAgLy8gdGV4ZWxzLiBUaGVuIGluIHRoZSByZXR1cm4gc3RhdGVtZW50IHdlIHJlaHlkcmF0ZSB0aGUgc3F1YXJpZmllZFxuICAgICAgLy8gZGltZW5zaW9ucyB0byBjaGFubmVsIHVuaXRzLlxuXG4gICAgICBjb25zdCBiYXRjaERpbSA9IGdldEJhdGNoRGltKGxvZ1NoYXBlKTtcbiAgICAgIGxldCByb3dzID0gMiwgY29scyA9IDI7XG4gICAgICBpZiAobG9nU2hhcGUubGVuZ3RoKSB7XG4gICAgICAgIFtyb3dzLCBjb2xzXSA9IGdldFJvd3NDb2xzKGxvZ1NoYXBlKTtcbiAgICAgIH1cbiAgICAgIHNpemUgPSBiYXRjaERpbSAqIChyb3dzIC8gMikgKiAoY29scyAvIDIpO1xuICAgICAgcmV0dXJuIHV0aWwuc2l6ZVRvU3F1YXJpc2hTaGFwZShzaXplKS5tYXAoZCA9PiBkICogMikgYXMgW251bWJlciwgbnVtYmVyXTtcbiAgICB9XG4gICAgcmV0dXJuIHV0aWwuc2l6ZVRvU3F1YXJpc2hTaGFwZShzaXplKTtcbiAgfVxufVxuXG5mdW5jdGlvbiBpc0V2ZW4objogbnVtYmVyKTogYm9vbGVhbiB7XG4gIHJldHVybiBuICUgMiA9PT0gMDtcbn1cblxuLyoqXG4gKiBUaGlzIGRldGVybWluZXMgd2hldGhlciByZXNoYXBpbmcgYSBwYWNrZWQgdGV4dHVyZSByZXF1aXJlcyByZWFycmFuZ2luZ1xuICogdGhlIGRhdGEgd2l0aGluIHRoZSB0ZXh0dXJlLCBhc3N1bWluZyAyeDIgcGFja2luZy5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGlzUmVzaGFwZUZyZWUoc2hhcGUxOiBudW1iZXJbXSwgc2hhcGUyOiBudW1iZXJbXSk6IGJvb2xlYW4ge1xuICBzaGFwZTEgPSBzaGFwZTEuc2xpY2UoLTIpO1xuICBzaGFwZTIgPSBzaGFwZTIuc2xpY2UoLTIpO1xuXG4gIGlmICh1dGlsLmFycmF5c0VxdWFsKHNoYXBlMSwgc2hhcGUyKSkge1xuICAgIHJldHVybiB0cnVlO1xuICB9XG5cbiAgaWYgKCFzaGFwZTEubGVuZ3RoIHx8ICFzaGFwZTIubGVuZ3RoKSB7ICAvLyBPbmUgb2YgdGhlIHNoYXBlcyBpcyBhIHNjYWxhci5cbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuXG4gIGlmIChzaGFwZTFbMF0gPT09IDAgfHwgc2hhcGUxWzFdID09PSAwIHx8IHNoYXBlMlswXSA9PT0gMCB8fFxuICAgICAgc2hhcGUyWzFdID09PSAwKSB7XG4gICAgcmV0dXJuIHRydWU7XG4gIH1cblxuICBpZiAoc2hhcGUxLmxlbmd0aCAhPT0gc2hhcGUyLmxlbmd0aCkgeyAgLy8gT25lIG9mIHRoZSBzaGFwZXMgaXMgYSB2ZWN0b3IuXG4gICAgY29uc3Qgc2hhcGUxQ29scyA9IHNoYXBlMS5zbGljZSgtMSlbMF07XG4gICAgY29uc3Qgc2hhcGUyQ29scyA9IHNoYXBlMi5zbGljZSgtMSlbMF07XG4gICAgaWYgKHNoYXBlMUNvbHMgPT09IHNoYXBlMkNvbHMpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIGlmIChpc0V2ZW4oc2hhcGUxQ29scykgJiYgaXNFdmVuKHNoYXBlMkNvbHMpICYmXG4gICAgICAgIChzaGFwZTFbMF0gPT09IDEgfHwgc2hhcGUyWzBdID09PSAxKSkge1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuICB9XG4gIHJldHVybiBzaGFwZTFbMV0gPT09IHNoYXBlMlsxXSAmJiBpc0V2ZW4oc2hhcGUxWzBdKSAmJiBpc0V2ZW4oc2hhcGUyWzBdKTtcbn1cblxuLy8gV2UgY2FjaGUgd2ViZ2wgcGFyYW1zIGJlY2F1c2UgdGhlIGVudmlyb25tZW50IGdldHMgcmVzZXQgYmV0d2VlblxuLy8gdW5pdCB0ZXN0cyBhbmQgd2UgZG9uJ3Qgd2FudCB0byBjb25zdGFudGx5IHF1ZXJ5IHRoZSBXZWJHTENvbnRleHQgZm9yXG4vLyBNQVhfVEVYVFVSRV9TSVpFLlxubGV0IE1BWF9URVhUVVJFX1NJWkU6IG51bWJlcjtcbmxldCBNQVhfVEVYVFVSRVNfSU5fU0hBREVSOiBudW1iZXI7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRXZWJHTE1heFRleHR1cmVTaXplKHdlYkdMVmVyc2lvbjogbnVtYmVyKTogbnVtYmVyIHtcbiAgaWYgKE1BWF9URVhUVVJFX1NJWkUgPT0gbnVsbCkge1xuICAgIGNvbnN0IGdsID0gZ2V0V2ViR0xDb250ZXh0KHdlYkdMVmVyc2lvbik7XG4gICAgTUFYX1RFWFRVUkVfU0laRSA9IGdsLmdldFBhcmFtZXRlcihnbC5NQVhfVEVYVFVSRV9TSVpFKTtcbiAgfVxuICByZXR1cm4gTUFYX1RFWFRVUkVfU0laRTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHJlc2V0TWF4VGV4dHVyZVNpemUoKSB7XG4gIE1BWF9URVhUVVJFX1NJWkUgPSBudWxsO1xufVxuZXhwb3J0IGZ1bmN0aW9uIHJlc2V0TWF4VGV4dHVyZXNJblNoYWRlcigpIHtcbiAgTUFYX1RFWFRVUkVTX0lOX1NIQURFUiA9IG51bGw7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRNYXhUZXh0dXJlc0luU2hhZGVyKHdlYkdMVmVyc2lvbjogbnVtYmVyKTogbnVtYmVyIHtcbiAgaWYgKE1BWF9URVhUVVJFU19JTl9TSEFERVIgPT0gbnVsbCkge1xuICAgIGNvbnN0IGdsID0gZ2V0V2ViR0xDb250ZXh0KHdlYkdMVmVyc2lvbik7XG4gICAgTUFYX1RFWFRVUkVTX0lOX1NIQURFUiA9IGdsLmdldFBhcmFtZXRlcihnbC5NQVhfVEVYVFVSRV9JTUFHRV9VTklUUyk7XG4gIH1cbiAgLy8gV2UgY2FwIGF0IDE2IHRvIGF2b2lkIHNwdXJpb3VzIHJ1bnRpbWUgXCJtZW1vcnkgZXhoYXVzdGVkXCIgZXJyb3IuXG4gIHJldHVybiBNYXRoLm1pbigxNiwgTUFYX1RFWFRVUkVTX0lOX1NIQURFUik7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRXZWJHTERpc2pvaW50UXVlcnlUaW1lclZlcnNpb24od2ViR0xWZXJzaW9uOiBudW1iZXIpOlxuICAgIG51bWJlciB7XG4gIGlmICh3ZWJHTFZlcnNpb24gPT09IDApIHtcbiAgICByZXR1cm4gMDtcbiAgfVxuXG4gIGxldCBxdWVyeVRpbWVyVmVyc2lvbjogbnVtYmVyO1xuICBjb25zdCBnbCA9IGdldFdlYkdMQ29udGV4dCh3ZWJHTFZlcnNpb24pO1xuXG4gIGlmIChoYXNFeHRlbnNpb24oZ2wsICdFWFRfZGlzam9pbnRfdGltZXJfcXVlcnlfd2ViZ2wyJykgJiZcbiAgICAgIHdlYkdMVmVyc2lvbiA9PT0gMikge1xuICAgIHF1ZXJ5VGltZXJWZXJzaW9uID0gMjtcbiAgfSBlbHNlIGlmIChoYXNFeHRlbnNpb24oZ2wsICdFWFRfZGlzam9pbnRfdGltZXJfcXVlcnknKSkge1xuICAgIHF1ZXJ5VGltZXJWZXJzaW9uID0gMTtcbiAgfSBlbHNlIHtcbiAgICBxdWVyeVRpbWVyVmVyc2lvbiA9IDA7XG4gIH1cbiAgcmV0dXJuIHF1ZXJ5VGltZXJWZXJzaW9uO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaGFzRXh0ZW5zaW9uKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGV4dGVuc2lvbk5hbWU6IHN0cmluZykge1xuICBjb25zdCBleHQgPSBnbC5nZXRFeHRlbnNpb24oZXh0ZW5zaW9uTmFtZSk7XG4gIHJldHVybiBleHQgIT0gbnVsbDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGlzV2ViR0xWZXJzaW9uRW5hYmxlZCh3ZWJHTFZlcnNpb246IDF8Mikge1xuICB0cnkge1xuICAgIGNvbnN0IGdsID0gZ2V0V2ViR0xDb250ZXh0KHdlYkdMVmVyc2lvbik7XG4gICAgaWYgKGdsICE9IG51bGwpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cbiAgfSBjYXRjaCAoZSkge1xuICAgIGNvbnNvbGUubG9nKCdFcnJvciB3aGVuIGdldHRpbmcgV2ViR0wgY29udGV4dDogJywgZSk7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIHJldHVybiBmYWxzZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGlzQ2FwYWJsZU9mUmVuZGVyaW5nVG9GbG9hdFRleHR1cmUod2ViR0xWZXJzaW9uOiBudW1iZXIpOlxuICAgIGJvb2xlYW4ge1xuICBpZiAod2ViR0xWZXJzaW9uID09PSAwKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG5cbiAgY29uc3QgZ2wgPSBnZXRXZWJHTENvbnRleHQod2ViR0xWZXJzaW9uKTtcblxuICBpZiAod2ViR0xWZXJzaW9uID09PSAxKSB7XG4gICAgaWYgKCFoYXNFeHRlbnNpb24oZ2wsICdPRVNfdGV4dHVyZV9mbG9hdCcpKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICB9IGVsc2Uge1xuICAgIGlmICghaGFzRXh0ZW5zaW9uKGdsLCAnRVhUX2NvbG9yX2J1ZmZlcl9mbG9hdCcpKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICB9XG5cbiAgY29uc3QgaXNGcmFtZUJ1ZmZlckNvbXBsZXRlID0gY3JlYXRlRmxvYXRUZXh0dXJlQW5kQmluZFRvRnJhbWVidWZmZXIoZ2wpO1xuICByZXR1cm4gaXNGcmFtZUJ1ZmZlckNvbXBsZXRlO1xufVxuXG4vKipcbiAqIENoZWNrIGlmIHdlIGNhbiBkb3dubG9hZCB2YWx1ZXMgZnJvbSBhIGZsb2F0L2hhbGYtZmxvYXQgdGV4dHVyZS5cbiAqXG4gKiBOb3RlIHRoYXQgZm9yIHBlcmZvcm1hbmNlIHJlYXNvbnMgd2UgdXNlIGJpbmRpbmcgYSB0ZXh0dXJlIHRvIGEgZnJhbWVidWZmZXJcbiAqIGFzIGEgcHJveHkgZm9yIGFiaWxpdHkgdG8gZG93bmxvYWQgZmxvYXQgdmFsdWVzIGxhdGVyIHVzaW5nIHJlYWRQaXhlbHMuIFRoZVxuICogdGV4dHVyZSBwYXJhbXMgb2YgdGhpcyB0ZXh0dXJlIHdpbGwgbm90IG1hdGNoIHRob3NlIGluIHJlYWRQaXhlbHMgZXhhY3RseVxuICogYnV0IGlmIHdlIGFyZSB1bmFibGUgdG8gYmluZCBzb21lIGtpbmQgb2YgZmxvYXQgdGV4dHVyZSB0byB0aGUgZnJhbWVCdWZmZXJcbiAqIHRoZW4gd2UgZGVmaW5pdGVseSB3aWxsIG5vdCBiZSBhYmxlIHRvIHJlYWQgZmxvYXQgdmFsdWVzIGZyb20gaXQuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpc0Rvd25sb2FkRmxvYXRUZXh0dXJlRW5hYmxlZCh3ZWJHTFZlcnNpb246IG51bWJlcik6IGJvb2xlYW4ge1xuICBpZiAod2ViR0xWZXJzaW9uID09PSAwKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG5cbiAgY29uc3QgZ2wgPSBnZXRXZWJHTENvbnRleHQod2ViR0xWZXJzaW9uKTtcblxuICBpZiAod2ViR0xWZXJzaW9uID09PSAxKSB7XG4gICAgaWYgKCFoYXNFeHRlbnNpb24oZ2wsICdPRVNfdGV4dHVyZV9mbG9hdCcpKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICAgIGlmICghaGFzRXh0ZW5zaW9uKGdsLCAnV0VCR0xfY29sb3JfYnVmZmVyX2Zsb2F0JykpIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gIH0gZWxzZSB7XG4gICAgaWYgKGhhc0V4dGVuc2lvbihnbCwgJ0VYVF9jb2xvcl9idWZmZXJfZmxvYXQnKSkge1xuICAgICAgcmV0dXJuIGNyZWF0ZUZsb2F0VGV4dHVyZUFuZEJpbmRUb0ZyYW1lYnVmZmVyKGdsKTtcbiAgICB9XG5cbiAgICBjb25zdCBDT0xPUl9CVUZGRVJfSEFMRl9GTE9BVCA9ICdFWFRfY29sb3JfYnVmZmVyX2hhbGZfZmxvYXQnO1xuICAgIGlmIChoYXNFeHRlbnNpb24oZ2wsIENPTE9SX0JVRkZFUl9IQUxGX0ZMT0FUKSkge1xuICAgICAgY29uc3QgdGV4dHVyZUhhbGZGbG9hdEV4dGVuc2lvbiA9XG4gICAgICAgICAgZ2wuZ2V0RXh0ZW5zaW9uKENPTE9SX0JVRkZFUl9IQUxGX0ZMT0FUKTtcbiAgICAgIHJldHVybiBjcmVhdGVIYWxmRmxvYXRUZXh0dXJlQW5kQmluZFRvRnJhbWVidWZmZXIoXG4gICAgICAgICAgZ2wsIHRleHR1cmVIYWxmRmxvYXRFeHRlbnNpb24pO1xuICAgIH1cblxuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIGNvbnN0IGlzRnJhbWVCdWZmZXJDb21wbGV0ZSA9IGNyZWF0ZUZsb2F0VGV4dHVyZUFuZEJpbmRUb0ZyYW1lYnVmZmVyKGdsKTtcbiAgcmV0dXJuIGlzRnJhbWVCdWZmZXJDb21wbGV0ZTtcbn1cblxuZnVuY3Rpb24gY3JlYXRlRmxvYXRUZXh0dXJlQW5kQmluZFRvRnJhbWVidWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6XG4gICAgYm9vbGVhbiB7XG4gIGNvbnN0IHRleENvbmZpZyA9IGdldFRleHR1cmVDb25maWcoZ2wpO1xuXG4gIGNvbnN0IHRleHR1cmUgPSBnbC5jcmVhdGVUZXh0dXJlKCk7XG4gIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRleHR1cmUpO1xuXG4gIGNvbnN0IHdpZHRoID0gMTtcbiAgY29uc3QgaGVpZ2h0ID0gMTtcbiAgZ2wudGV4SW1hZ2UyRChcbiAgICAgIGdsLlRFWFRVUkVfMkQsIDAsIHRleENvbmZpZy5pbnRlcm5hbEZvcm1hdEZsb2F0LCB3aWR0aCwgaGVpZ2h0LCAwLFxuICAgICAgdGV4Q29uZmlnLnRleHR1cmVGb3JtYXRGbG9hdCwgdGV4Q29uZmlnLnRleHR1cmVUeXBlRmxvYXQsIG51bGwpO1xuXG4gIGNvbnN0IGZyYW1lQnVmZmVyID0gZ2wuY3JlYXRlRnJhbWVidWZmZXIoKTtcbiAgZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBmcmFtZUJ1ZmZlcik7XG4gIGdsLmZyYW1lYnVmZmVyVGV4dHVyZTJEKFxuICAgICAgZ2wuRlJBTUVCVUZGRVIsIGdsLkNPTE9SX0FUVEFDSE1FTlQwLCBnbC5URVhUVVJFXzJELCB0ZXh0dXJlLCAwKTtcblxuICBjb25zdCBpc0ZyYW1lQnVmZmVyQ29tcGxldGUgPVxuICAgICAgZ2wuY2hlY2tGcmFtZWJ1ZmZlclN0YXR1cyhnbC5GUkFNRUJVRkZFUikgPT09IGdsLkZSQU1FQlVGRkVSX0NPTVBMRVRFO1xuXG4gIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIG51bGwpO1xuICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIG51bGwpO1xuICBnbC5kZWxldGVUZXh0dXJlKHRleHR1cmUpO1xuICBnbC5kZWxldGVGcmFtZWJ1ZmZlcihmcmFtZUJ1ZmZlcik7XG5cbiAgcmV0dXJuIGlzRnJhbWVCdWZmZXJDb21wbGV0ZTtcbn1cblxuZnVuY3Rpb24gY3JlYXRlSGFsZkZsb2F0VGV4dHVyZUFuZEJpbmRUb0ZyYW1lYnVmZmVyKFxuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlSGFsZkZsb2F0RXh0ZW5zaW9uOiBhbnkpOiBib29sZWFuIHtcbiAgY29uc3QgdGV4Q29uZmlnID0gZ2V0VGV4dHVyZUNvbmZpZyhnbCwgdGV4dHVyZUhhbGZGbG9hdEV4dGVuc2lvbik7XG4gIGNvbnN0IHRleHR1cmUgPSBnbC5jcmVhdGVUZXh0dXJlKCk7XG4gIGdsLmJpbmRUZXh0dXJlKGdsLlRFWFRVUkVfMkQsIHRleHR1cmUpO1xuXG4gIGNvbnN0IHdpZHRoID0gMTtcbiAgY29uc3QgaGVpZ2h0ID0gMTtcbiAgZ2wudGV4SW1hZ2UyRChcbiAgICAgIGdsLlRFWFRVUkVfMkQsIDAsIHRleENvbmZpZy5pbnRlcm5hbEZvcm1hdEhhbGZGbG9hdCwgd2lkdGgsIGhlaWdodCwgMCxcbiAgICAgIHRleENvbmZpZy50ZXh0dXJlRm9ybWF0RmxvYXQsIHRleENvbmZpZy50ZXh0dXJlVHlwZUhhbGZGbG9hdCwgbnVsbCk7XG5cbiAgY29uc3QgZnJhbWVCdWZmZXIgPSBnbC5jcmVhdGVGcmFtZWJ1ZmZlcigpO1xuICBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIGZyYW1lQnVmZmVyKTtcbiAgZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoXG4gICAgICBnbC5GUkFNRUJVRkZFUiwgZ2wuQ09MT1JfQVRUQUNITUVOVDAsIGdsLlRFWFRVUkVfMkQsIHRleHR1cmUsIDApO1xuXG4gIGNvbnN0IGlzRnJhbWVCdWZmZXJDb21wbGV0ZSA9XG4gICAgICBnbC5jaGVja0ZyYW1lYnVmZmVyU3RhdHVzKGdsLkZSQU1FQlVGRkVSKSA9PT0gZ2wuRlJBTUVCVUZGRVJfQ09NUExFVEU7XG5cbiAgZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgbnVsbCk7XG4gIGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgbnVsbCk7XG4gIGdsLmRlbGV0ZVRleHR1cmUodGV4dHVyZSk7XG4gIGdsLmRlbGV0ZUZyYW1lYnVmZmVyKGZyYW1lQnVmZmVyKTtcblxuICByZXR1cm4gaXNGcmFtZUJ1ZmZlckNvbXBsZXRlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNXZWJHTEZlbmNlRW5hYmxlZCh3ZWJHTFZlcnNpb246IG51bWJlcikge1xuICBpZiAod2ViR0xWZXJzaW9uICE9PSAyKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGNvbnN0IGdsID0gZ2V0V2ViR0xDb250ZXh0KHdlYkdMVmVyc2lvbik7XG5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICBjb25zdCBpc0VuYWJsZWQgPSAoZ2wgYXMgYW55KS5mZW5jZVN5bmMgIT0gbnVsbDtcbiAgcmV0dXJuIGlzRW5hYmxlZDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGFzc2VydE5vdENvbXBsZXgoXG4gICAgdGVuc29yOiBUZW5zb3JJbmZvfFRlbnNvckluZm9bXSwgb3BOYW1lOiBzdHJpbmcpOiB2b2lkIHtcbiAgaWYgKCFBcnJheS5pc0FycmF5KHRlbnNvcikpIHtcbiAgICB0ZW5zb3IgPSBbdGVuc29yXTtcbiAgfVxuICB0ZW5zb3IuZm9yRWFjaCh0ID0+IHtcbiAgICBpZiAodCAhPSBudWxsKSB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICB0LmR0eXBlICE9PSAnY29tcGxleDY0JyxcbiAgICAgICAgICAoKSA9PiBgJHtvcE5hbWV9IGRvZXMgbm90IHN1cHBvcnQgY29tcGxleDY0IHRlbnNvcnMgYCArXG4gICAgICAgICAgICAgICdpbiB0aGUgV2ViR0wgYmFja2VuZC4nKTtcbiAgICB9XG4gIH0pO1xufVxuIl19