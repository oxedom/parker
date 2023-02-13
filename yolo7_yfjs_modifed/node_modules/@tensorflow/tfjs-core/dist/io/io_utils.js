/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { complex } from '../ops/complex';
import { tensor } from '../ops/tensor';
import { sizeFromShape } from '../util';
import { DTYPE_VALUE_SIZE_MAP } from './types';
/** Number of bytes reserved for the length of the string. (32bit integer). */
const NUM_BYTES_STRING_LENGTH = 4;
/**
 * Encode a map from names to weight values as an ArrayBuffer, along with an
 * `Array` of `WeightsManifestEntry` as specification of the encoded weights.
 *
 * This function does not perform sharding.
 *
 * This function is the reverse of `decodeWeights`.
 *
 * @param tensors A map ("dict") from names to tensors.
 * @param group Group to which the weights belong (optional).
 * @returns A `Promise` of
 *   - A flat `ArrayBuffer` with all the binary values of the `Tensor`s
 *     concatenated.
 *   - An `Array` of `WeightManifestEntry`s, carrying information including
 *     tensor names, `dtype`s and shapes.
 * @throws Error: on unsupported tensor `dtype`.
 */
export async function encodeWeights(tensors, group) {
    // TODO(adarob, cais): Support quantization.
    const specs = [];
    const dataPromises = [];
    const names = Array.isArray(tensors) ?
        tensors.map(tensor => tensor.name) :
        Object.keys(tensors);
    for (let i = 0; i < names.length; ++i) {
        const name = names[i];
        const t = Array.isArray(tensors) ? tensors[i].tensor : tensors[name];
        if (t.dtype !== 'float32' && t.dtype !== 'int32' && t.dtype !== 'bool' &&
            t.dtype !== 'string' && t.dtype !== 'complex64') {
            throw new Error(`Unsupported dtype in weight '${name}': ${t.dtype}`);
        }
        const spec = { name, shape: t.shape, dtype: t.dtype };
        if (t.dtype === 'string') {
            const utf8bytes = new Promise(async (resolve) => {
                const vals = await t.bytes();
                const totalNumBytes = vals.reduce((p, c) => p + c.length, 0) +
                    NUM_BYTES_STRING_LENGTH * vals.length;
                const bytes = new Uint8Array(totalNumBytes);
                let offset = 0;
                for (let i = 0; i < vals.length; i++) {
                    const val = vals[i];
                    const bytesOfLength = new Uint8Array(new Uint32Array([val.length]).buffer);
                    bytes.set(bytesOfLength, offset);
                    offset += NUM_BYTES_STRING_LENGTH;
                    bytes.set(val, offset);
                    offset += val.length;
                }
                resolve(bytes);
            });
            dataPromises.push(utf8bytes);
        }
        else {
            dataPromises.push(t.data());
        }
        if (group != null) {
            spec.group = group;
        }
        specs.push(spec);
    }
    const tensorValues = await Promise.all(dataPromises);
    return { data: concatenateTypedArrays(tensorValues), specs };
}
/**
 * Decode flat ArrayBuffer as weights.
 *
 * This function does not handle sharding.
 *
 * This function is the reverse of `encodeWeights`.
 *
 * @param buffer A flat ArrayBuffer carrying the binary values of the tensors
 *   concatenated in the order specified in `specs`.
 * @param specs Specifications of the names, dtypes and shapes of the tensors
 *   whose value are encoded by `buffer`.
 * @return A map from tensor name to tensor value, with the names corresponding
 *   to names in `specs`.
 * @throws Error, if any of the tensors has unsupported dtype.
 */
export function decodeWeights(buffer, specs) {
    // TODO(adarob, cais): Support quantization.
    const out = {};
    let float16Decode;
    let offset = 0;
    for (const spec of specs) {
        const name = spec.name;
        const dtype = spec.dtype;
        const shape = spec.shape;
        const size = sizeFromShape(shape);
        let values;
        if ('quantization' in spec) {
            const quantization = spec.quantization;
            if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
                if (!('min' in quantization && 'scale' in quantization)) {
                    throw new Error(`Weight ${spec.name} with quantization ${quantization.dtype} ` +
                        `doesn't have corresponding metadata min and scale.`);
                }
            }
            else if (quantization.dtype === 'float16') {
                if (dtype !== 'float32') {
                    throw new Error(`Weight ${spec.name} is quantized with ${quantization.dtype} ` +
                        `which only supports weights of type float32 not ${dtype}.`);
                }
            }
            else {
                throw new Error(`Weight ${spec.name} has unknown ` +
                    `quantization dtype ${quantization.dtype}. ` +
                    `Supported quantization dtypes are: ` +
                    `'uint8', 'uint16', and 'float16'.`);
            }
            const quantizationSizeFactor = DTYPE_VALUE_SIZE_MAP[quantization.dtype];
            const byteBuffer = buffer.slice(offset, offset + size * quantizationSizeFactor);
            const quantizedArray = (quantization.dtype === 'uint8') ?
                new Uint8Array(byteBuffer) :
                new Uint16Array(byteBuffer);
            if (dtype === 'float32') {
                if (quantization.dtype === 'uint8' || quantization.dtype === 'uint16') {
                    values = new Float32Array(quantizedArray.length);
                    for (let i = 0; i < quantizedArray.length; i++) {
                        const v = quantizedArray[i];
                        values[i] = v * quantization.scale + quantization.min;
                    }
                }
                else if (quantization.dtype === 'float16') {
                    if (float16Decode === undefined) {
                        float16Decode = getFloat16Decoder();
                    }
                    values = float16Decode(quantizedArray);
                }
                else {
                    throw new Error(`Unsupported quantization type ${quantization.dtype} ` +
                        `for weight type float32.`);
                }
            }
            else if (dtype === 'int32') {
                if (quantization.dtype !== 'uint8' && quantization.dtype !== 'uint16') {
                    throw new Error(`Unsupported quantization type ${quantization.dtype} ` +
                        `for weight type int32.`);
                }
                values = new Int32Array(quantizedArray.length);
                for (let i = 0; i < quantizedArray.length; i++) {
                    const v = quantizedArray[i];
                    values[i] = Math.round(v * quantization.scale + quantization.min);
                }
            }
            else {
                throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
            }
            offset += size * quantizationSizeFactor;
        }
        else if (dtype === 'string') {
            const size = sizeFromShape(spec.shape);
            values = [];
            for (let i = 0; i < size; i++) {
                const byteLength = new Uint32Array(buffer.slice(offset, offset + NUM_BYTES_STRING_LENGTH))[0];
                offset += NUM_BYTES_STRING_LENGTH;
                const bytes = new Uint8Array(buffer.slice(offset, offset + byteLength));
                values.push(bytes);
                offset += byteLength;
            }
        }
        else {
            const dtypeFactor = DTYPE_VALUE_SIZE_MAP[dtype];
            const byteBuffer = buffer.slice(offset, offset + size * dtypeFactor);
            if (dtype === 'float32') {
                values = new Float32Array(byteBuffer);
            }
            else if (dtype === 'int32') {
                values = new Int32Array(byteBuffer);
            }
            else if (dtype === 'bool') {
                values = new Uint8Array(byteBuffer);
            }
            else if (dtype === 'complex64') {
                values = new Float32Array(byteBuffer);
                const real = new Float32Array(values.length / 2);
                const image = new Float32Array(values.length / 2);
                for (let i = 0; i < real.length; i++) {
                    real[i] = values[i * 2];
                    image[i] = values[i * 2 + 1];
                }
                const realTensor = tensor(real, shape, 'float32');
                const imageTensor = tensor(image, shape, 'float32');
                out[name] = complex(realTensor, imageTensor);
                realTensor.dispose();
                imageTensor.dispose();
            }
            else {
                throw new Error(`Unsupported dtype in weight '${name}': ${dtype}`);
            }
            offset += size * dtypeFactor;
        }
        if (dtype !== 'complex64') {
            out[name] = tensor(values, shape, dtype);
        }
    }
    return out;
}
/**
 * Concatenate TypedArrays into an ArrayBuffer.
 */
export function concatenateTypedArrays(xs) {
    // TODO(adarob, cais): Support quantization.
    if (xs === null) {
        throw new Error(`Invalid input value: ${JSON.stringify(xs)}`);
    }
    let totalByteLength = 0;
    // `normalizedXs` is here for this reason: a `TypedArray`'s `buffer'
    // can have a different byte length from that of the `TypedArray` itself,
    // for example, when the `TypedArray` is created from an offset in an
    // `ArrayBuffer`. `normliazedXs` holds `TypedArray`s whose `buffer`s match
    // the `TypedArray` in byte length. If an element of `xs` does not show
    // this property, a new `TypedArray` that satisfy this property will be
    // constructed and pushed into `normalizedXs`.
    const normalizedXs = [];
    xs.forEach((x) => {
        totalByteLength += x.byteLength;
        // tslint:disable:no-any
        normalizedXs.push(x.byteLength === x.buffer.byteLength ? x :
            new x.constructor(x));
        if (!(x instanceof Float32Array || x instanceof Int32Array ||
            x instanceof Uint8Array)) {
            throw new Error(`Unsupported TypedArray subtype: ${x.constructor.name}`);
        }
        // tslint:enable:no-any
    });
    const y = new Uint8Array(totalByteLength);
    let offset = 0;
    normalizedXs.forEach((x) => {
        y.set(new Uint8Array(x.buffer), offset);
        offset += x.byteLength;
    });
    return y.buffer;
}
// Use Buffer on Node.js instead of Blob/atob/btoa
const useNodeBuffer = typeof Buffer !== 'undefined' &&
    (typeof Blob === 'undefined' || typeof atob === 'undefined' ||
        typeof btoa === 'undefined');
/**
 * Calculate the byte length of a JavaScript string.
 *
 * Note that a JavaScript string can contain wide characters, therefore the
 * length of the string is not necessarily equal to the byte length.
 *
 * @param str Input string.
 * @returns Byte length.
 */
export function stringByteLength(str) {
    if (useNodeBuffer) {
        return Buffer.byteLength(str);
    }
    return new Blob([str]).size;
}
/**
 * Encode an ArrayBuffer as a base64 encoded string.
 *
 * @param buffer `ArrayBuffer` to be converted.
 * @returns A string that base64-encodes `buffer`.
 */
export function arrayBufferToBase64String(buffer) {
    if (useNodeBuffer) {
        return Buffer.from(buffer).toString('base64');
    }
    const buf = new Uint8Array(buffer);
    let s = '';
    for (let i = 0, l = buf.length; i < l; i++) {
        s += String.fromCharCode(buf[i]);
    }
    return btoa(s);
}
/**
 * Decode a base64 string as an ArrayBuffer.
 *
 * @param str Base64 string.
 * @returns Decoded `ArrayBuffer`.
 */
export function base64StringToArrayBuffer(str) {
    if (useNodeBuffer) {
        const buf = Buffer.from(str, 'base64');
        return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    const s = atob(str);
    const buffer = new Uint8Array(s.length);
    for (let i = 0; i < s.length; ++i) {
        buffer.set([s.charCodeAt(i)], i);
    }
    return buffer.buffer;
}
/**
 * Concatenate a number of ArrayBuffers into one.
 *
 * @param buffers A number of array buffers to concatenate.
 * @returns Result of concatenating `buffers` in order.
 */
export function concatenateArrayBuffers(buffers) {
    if (buffers.length === 1) {
        return buffers[0];
    }
    let totalByteLength = 0;
    buffers.forEach((buffer) => {
        totalByteLength += buffer.byteLength;
    });
    const temp = new Uint8Array(totalByteLength);
    let offset = 0;
    buffers.forEach((buffer) => {
        temp.set(new Uint8Array(buffer), offset);
        offset += buffer.byteLength;
    });
    return temp.buffer;
}
/**
 * Get the basename of a path.
 *
 * Behaves in a way analogous to Linux's basename command.
 *
 * @param path
 */
export function basename(path) {
    const SEPARATOR = '/';
    path = path.trim();
    while (path.endsWith(SEPARATOR)) {
        path = path.slice(0, path.length - 1);
    }
    const items = path.split(SEPARATOR);
    return items[items.length - 1];
}
/**
 * Create `ModelJSON` from `ModelArtifacts`.
 *
 * @param artifacts Model artifacts, describing the model and its weights.
 * @param manifest Weight manifest, describing where the weights of the
 *     `ModelArtifacts` are stored, and some metadata about them.
 * @returns Object representing the `model.json` file describing the model
 *     artifacts and weights
 */
export function getModelJSONForModelArtifacts(artifacts, manifest) {
    const result = {
        modelTopology: artifacts.modelTopology,
        format: artifacts.format,
        generatedBy: artifacts.generatedBy,
        convertedBy: artifacts.convertedBy,
        weightsManifest: manifest
    };
    if (artifacts.signature != null) {
        result.signature = artifacts.signature;
    }
    if (artifacts.userDefinedMetadata != null) {
        result.userDefinedMetadata = artifacts.userDefinedMetadata;
    }
    if (artifacts.modelInitializer != null) {
        result.modelInitializer = artifacts.modelInitializer;
    }
    if (artifacts.trainingConfig != null) {
        result.trainingConfig = artifacts.trainingConfig;
    }
    return result;
}
/**
 * Create `ModelArtifacts` from a JSON file.
 *
 * @param modelJSON Object containing the parsed JSON of `model.json`
 * @param loadWeights Function that takes the JSON file's weights manifest,
 *     reads weights from the listed path(s), and returns a Promise of the
 *     weight manifest entries along with the weights data.
 * @returns A Promise of the `ModelArtifacts`, as described by the JSON file.
 */
export async function getModelArtifactsForJSON(modelJSON, loadWeights) {
    const modelArtifacts = {
        modelTopology: modelJSON.modelTopology,
        format: modelJSON.format,
        generatedBy: modelJSON.generatedBy,
        convertedBy: modelJSON.convertedBy
    };
    if (modelJSON.trainingConfig != null) {
        modelArtifacts.trainingConfig = modelJSON.trainingConfig;
    }
    if (modelJSON.weightsManifest != null) {
        const [weightSpecs, weightData] = await loadWeights(modelJSON.weightsManifest);
        modelArtifacts.weightSpecs = weightSpecs;
        modelArtifacts.weightData = weightData;
    }
    if (modelJSON.signature != null) {
        modelArtifacts.signature = modelJSON.signature;
    }
    if (modelJSON.userDefinedMetadata != null) {
        modelArtifacts.userDefinedMetadata = modelJSON.userDefinedMetadata;
    }
    if (modelJSON.modelInitializer != null) {
        modelArtifacts.modelInitializer = modelJSON.modelInitializer;
    }
    return modelArtifacts;
}
/**
 * Populate ModelArtifactsInfo fields for a model with JSON topology.
 * @param modelArtifacts
 * @returns A ModelArtifactsInfo object.
 */
export function getModelArtifactsInfoForJSON(modelArtifacts) {
    if (modelArtifacts.modelTopology instanceof ArrayBuffer) {
        throw new Error('Expected JSON model topology, received ArrayBuffer.');
    }
    return {
        dateSaved: new Date(),
        modelTopologyType: 'JSON',
        modelTopologyBytes: modelArtifacts.modelTopology == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.modelTopology)),
        weightSpecsBytes: modelArtifacts.weightSpecs == null ?
            0 :
            stringByteLength(JSON.stringify(modelArtifacts.weightSpecs)),
        weightDataBytes: modelArtifacts.weightData == null ?
            0 :
            modelArtifacts.weightData.byteLength,
    };
}
/**
 * Computes mantisa table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 2048 mantissa lookup values.
 */
function computeFloat16MantisaTable() {
    const convertMantissa = (i) => {
        let m = i << 13;
        let e = 0;
        while ((m & 0x00800000) === 0) {
            e -= 0x00800000;
            m <<= 1;
        }
        m &= ~0x00800000;
        e += 0x38800000;
        return m | e;
    };
    const mantisaTable = new Uint32Array(2048);
    mantisaTable[0] = 0;
    for (let i = 1; i < 1024; i++) {
        mantisaTable[i] = convertMantissa(i);
    }
    for (let i = 1024; i < 2048; i++) {
        mantisaTable[i] = 0x38000000 + ((i - 1024) << 13);
    }
    return mantisaTable;
}
/**
 * Computes exponent table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 64 exponent lookup values.
 */
function computeFloat16ExponentTable() {
    const exponentTable = new Uint32Array(64);
    exponentTable[0] = 0;
    exponentTable[31] = 0x47800000;
    exponentTable[32] = 0x80000000;
    exponentTable[63] = 0xc7800000;
    for (let i = 1; i < 31; i++) {
        exponentTable[i] = i << 23;
    }
    for (let i = 33; i < 63; i++) {
        exponentTable[i] = 0x80000000 + ((i - 32) << 23);
    }
    return exponentTable;
}
/**
 * Computes offset table for casting Float16 to Float32
 * See http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
 *
 * @returns Uint32Array, 6d offset values.
 */
function computeFloat16OffsetTable() {
    const offsetTable = new Uint32Array(64);
    for (let i = 0; i < 64; i++) {
        offsetTable[i] = 1024;
    }
    offsetTable[0] = offsetTable[32] = 0;
    return offsetTable;
}
/**
 * Retrieve a Float16 decoder which will decode a ByteArray of Float16 values
 * to a Float32Array.
 *
 * @returns Function (buffer: Uint16Array) => Float32Array which decodes
 *          the Uint16Array of Float16 bytes to a Float32Array.
 */
export function getFloat16Decoder() {
    // Algorithm is based off of
    // http://www.fox-toolkit.org/ftp/fasthalffloatconversion.pdf
    // Cache lookup tables
    const mantisaTable = computeFloat16MantisaTable();
    const exponentTable = computeFloat16ExponentTable();
    const offsetTable = computeFloat16OffsetTable();
    return (quantizedArray) => {
        const buffer = new ArrayBuffer(4 * quantizedArray.length);
        const bufferUint32View = new Uint32Array(buffer);
        for (let index = 0; index < quantizedArray.length; index++) {
            const float16Bits = quantizedArray[index];
            const float32Bits = mantisaTable[offsetTable[float16Bits >> 10] + (float16Bits & 0x3ff)] +
                exponentTable[float16Bits >> 10];
            bufferUint32View[index] = float32Bits;
        }
        return new Float32Array(buffer);
    };
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW9fdXRpbHMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL2lvL2lvX3V0aWxzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxnQkFBZ0IsQ0FBQztBQUN2QyxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sZUFBZSxDQUFDO0FBR3JDLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFdEMsT0FBTyxFQUFDLG9CQUFvQixFQUEwRyxNQUFNLFNBQVMsQ0FBQztBQUV0Siw4RUFBOEU7QUFDOUUsTUFBTSx1QkFBdUIsR0FBRyxDQUFDLENBQUM7QUFFbEM7Ozs7Ozs7Ozs7Ozs7Ozs7R0FnQkc7QUFDSCxNQUFNLENBQUMsS0FBSyxVQUFVLGFBQWEsQ0FDL0IsT0FBcUMsRUFBRSxLQUFtQjtJQUU1RCw0Q0FBNEM7SUFDNUMsTUFBTSxLQUFLLEdBQTJCLEVBQUUsQ0FBQztJQUN6QyxNQUFNLFlBQVksR0FBK0IsRUFBRSxDQUFDO0lBRXBELE1BQU0sS0FBSyxHQUFhLEtBQUssQ0FBQyxPQUFPLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUM1QyxPQUFPLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7UUFDcEMsTUFBTSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQztJQUV6QixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsS0FBSyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtRQUNyQyxNQUFNLElBQUksR0FBRyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdEIsTUFBTSxDQUFDLEdBQUcsS0FBSyxDQUFDLE9BQU8sQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMs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SSxXQUFXLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDakQsS0FBSyxJQUFJLEtBQUssR0FBRyxDQUFDLEVBQUUsS0FBSyxHQUFHLGNBQWMsQ0FBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLEVBQUU7WUFDMUQsTUFBTSxXQUFXLEdBQUcsY0FBYyxDQUFDLEtBQUssQ0FBQyxDQUFDO1lBQzFDLE1BQU0sV0FBVyxHQUNiLFlBQVksQ0FBQyxXQUFXLENBQUMsV0FBVyxJQUFJLEVBQUUsQ0FBQyxHQUFHLENBQUMsV0FBVyxHQUFHLEtBQUssQ0FBQyxDQUFDO2dCQUNwRSxhQUFhLENBQUMsV0FBVyxJQUFJLEVBQUUsQ0FBQyxDQUFDO1lBQ3JDLGdCQUFnQixDQUFDLEtBQUssQ0FBQyxHQUFHLFdBQVcsQ0FBQztTQUN2QztRQUNELE9BQU8sSUFBSSxZQUFZLENBQUMsTUFBTSxDQUFDLENBQUM7SUFDbEMsQ0FBQyxDQUFDO0FBQ0osQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtjb21wbGV4fSBmcm9tICcuLi9vcHMvY29tcGxleCc7XG5pbXBvcnQge3RlbnNvcn0gZnJvbSAnLi4vb3BzL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yLCBOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7VHlwZWRBcnJheX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHtzaXplRnJvbVNoYXBlfSBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtEVFlQRV9WQUxVRV9TSVpFX01BUCwgTW9kZWxBcnRpZmFjdHMsIE1vZGVsQXJ0aWZhY3RzSW5mbywgTW9kZWxKU09OLCBXZWlnaHRHcm91cCwgV2VpZ2h0c01hbmlmZXN0Q29uZmlnLCBXZWlnaHRzTWFuaWZlc3RFbnRyeX0gZnJvbSAnLi90eXBlcyc7XG5cbi8qKiBOdW1iZXIgb2YgYnl0ZXMgcmVzZXJ2ZWQgZm9yIHRoZSBsZW5ndGggb2YgdGhlIHN0cmluZy4gKDMyYml0IGludGVnZXIpLiAqL1xuY29uc3QgTlVNX0JZVEVTX1NUUklOR19MRU5HVEggPSA0O1xuXG4vKipcbiAqIEVuY29kZSBhIG1hcCBmcm9tIG5hbWVzIHRvIHdlaWdodCB2YWx1ZXMgYXMgYW4gQXJyYXlCdWZmZXIsIGFsb25nIHdpdGggYW5cbiAqIGBBcnJheWAgb2YgYFdlaWdodHNNYW5pZmVzdEVudHJ5YCBhcyBzcGVjaWZpY2F0aW9uIG9mIHRoZSBlbmNvZGVkIHdlaWdodHMuXG4gKlxuICogVGhpcyBmdW5jdGlvbiBkb2VzIG5vdCBwZXJmb3JtIHNoYXJkaW5nLlxuICpcbiAqIFRoaXMgZnVuY3Rpb24gaXMgdGhlIHJldmVyc2Ugb2YgYGRlY29kZVdlaWdodHNgLlxuICpcbiAqIEBwYXJhbSB0ZW5zb3JzIEEgbWFwIChcImRpY3RcIikgZnJvbSBuYW1lcyB0byB0ZW5zb3JzLlxuICogQHBhcmFtIGdyb3VwIEdyb3VwIHRvIHdoaWNoIHRoZSB3ZWlnaHRzIGJlbG9uZyAob3B0aW9uYWwpLlxuICogQHJldHVybnMgQSBgUHJvbWlzZWAgb2ZcbiAqICAgLSBBIGZsYXQgYEFycmF5QnVmZmVyYCB3aXRoIGFsbCB0aGUgYmluYXJ5IHZhbHVlcyBvZiB0aGUgYFRlbnNvcmBzXG4gKiAgICAgY29uY2F0ZW5hdGVkLlxuICogICAtIEFuIGBBcnJheWAgb2YgYFdlaWdodE1hbmlmZXN0RW50cnlgcywgY2FycnlpbmcgaW5mb3JtYXRpb24gaW5jbHVkaW5nXG4gKiAgICAgdGVuc29yIG5hbWVzLCBgZHR5cGVgcyBhbmQgc2hhcGVzLlxuICogQHRocm93cyBFcnJvcjogb24gdW5zdXBwb3J0ZWQgdGVuc29yIGBkdHlwZWAuXG4gKi9cbmV4cG9ydCBhc3luYyBmdW5jdGlvbiBlbmNvZGVXZWlnaHRzKFxuICAgIHRlbnNvcnM6IE5hbWVkVGVuc29yTWFwfE5hbWVkVGVuc29yW10sIGdyb3VwPzogV2VpZ2h0R3JvdXApOlxuICAgIFByb21pc2U8e2RhdGE6IEFycmF5QnVmZmVyLCBzcGVjczogV2VpZ2h0c01hbmlmZXN0RW50cnlbXX0+IHtcbiAgLy8gVE9ETyhhZGFyb2IsIGNhaXMpOiBTdXBwb3J0IHF1YW50aXphdGlvbi5cbiAgY29uc3Qgc3BlY3M6IFdlaWdodHNNYW5pZmVzdEVudHJ5W10gPSBbXTtcbiAgY29uc3QgZGF0YVByb21pc2VzOiBBcnJheTxQcm9taXNlPFR5cGVkQXJyYXk+PiA9IFtdO1xuXG4gIGNvbnN0IG5hbWVzOiBzdHJpbmdbXSA9IEFycmF5LmlzQXJyYXkodGVuc29ycykgP1xuICAgICAgdGVuc29ycy5tYXAodGVuc29yID0+IHRlbnNvci5uYW1lKSA6XG4gICAgICBPYmplY3Qua2V5cyh0ZW5zb3JzKTtcblxuICBmb3IgKGxldCBpID0gMDsgaSA8IG5hbWVzLmxlbmd0aDsgKytpKSB7XG4gICAgY29uc3QgbmFtZSA9IG5hbWVzW2ldO1xuICAgIGNvbnN0IHQgPSBBcnJheS5pc0FycmF5KHRlbnNvcnMpID8gdGVuc29yc1tpXS50ZW5zb3IgOiB0ZW5zb3JzW25hbWVdO1xuICAgIGlmICh0LmR0eXBlICE9PSAnZmxvYXQzMicgJiYgdC5kdHlwZSAhPT0gJ2ludDMyJyAmJiB0LmR0eXBlICE9PSAnYm9vbCcgJiZcbiAgICAgICAgdC5kdHlwZSAhPT0gJ3N0cmluZycgJiYgdC5kdHlwZSAhPT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVW5zdXBwb3J0ZWQgZHR5cGUgaW4gd2VpZ2h0ICcke25hbWV9JzogJHt0LmR0eXBlfWApO1xuICAgIH1cbiAgICBjb25zdCBzcGVjOiBXZWlnaHRzTWFuaWZlc3RFbnRyeSA9IHtuYW1lLCBzaGFwZTogdC5zaGFwZSwgZHR5cGU6IHQuZHR5cGV9O1xuICAgIGlmICh0LmR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgY29uc3QgdXRmOGJ5dGVzID0gbmV3IFByb21pc2U8VHlwZWRBcnJheT4oYXN5bmMgcmVzb2x2ZSA9PiB7XG4gICAgICAgIGNvbnN0IHZhbHMgPSBhd2FpdCB0LmJ5dGVzKCkgYXMgVWludDhBcnJheVtdO1xuICAgICAgICBjb25zdCB0b3RhbE51bUJ5dGVzID0gdmFscy5yZWR1Y2UoKHAsIGMpID0+IHAgKyBjLmxlbmd0aCwgMCkgK1xuICAgICAgICAgICAgTlVNX0JZVEVTX1NUUklOR19MRU5HVEggKiB2YWxzLmxlbmd0aDtcbiAgICAgICAgY29uc3QgYnl0ZXMgPSBuZXcgVWludDhBcnJheSh0b3RhbE51bUJ5dGVzKTtcbiAgICAgICAgbGV0IG9mZnNldCA9IDA7XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFscy5sZW5ndGg7IGkrKykge1xuICAgICAgICAgIGNvbnN0IHZhbCA9IHZhbHNbaV07XG4gICAgICAgICAgY29uc3QgYnl0ZXNPZkxlbmd0aCA9XG4gICAgICAgICAgICAgIG5ldyBVaW50OEFycmF5KG5ldyBVaW50MzJBcnJheShbdmFsLmxlbmd0aF0pLmJ1ZmZlcik7XG4gICAgICAgICAgYnl0ZXMuc2V0KGJ5dGVzT2ZMZW5ndGgsIG9mZnNldCk7XG4gICAgICAgICAgb2Zmc2V0ICs9IE5VTV9CWVRFU19TVFJJTkdfTEVOR1RIO1xuICAgICAgICAgIGJ5dGVzLnNldCh2YWwsIG9mZnNldCk7XG4gICAgICAgICAgb2Zmc2V0ICs9IHZhbC5sZW5ndGg7XG4gICAgICAgIH1cbiAgICAgICAgcmVzb2x2ZShieXRlcyk7XG4gICAgICB9KTtcbiAgICAgIGRhdGFQcm9taXNlcy5wdXNoKHV0ZjhieXRlcyk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGRhdGFQcm9taXNlcy5wdXNoKHQuZGF0YSgpKTtcbiAgICB9XG4gICAgaWYgKGdyb3VwICE9IG51bGwpIHtcbiAgICAgIHNwZWMuZ3JvdXAgPSBncm91cDtcbiAgICB9XG4gICAgc3BlY3MucHVzaChzcGVjKTtcbiAgfVxuXG4gIGNvbnN0IHRlbnNvclZhbHVlcyA9IGF3YWl0IFByb21pc2UuYWxsKGRhdGFQcm9taXNlcyk7XG4gIHJldHVybiB7ZGF0YTogY29uY2F0ZW5hdGVUeXBlZEFycmF5cyh0ZW5zb3JWYWx1ZXMpLCBzcGVjc307XG59XG5cbi8qKlxuICogRGVjb2RlIGZsYXQgQXJyYXlCdWZmZXIgYXMgd2VpZ2h0cy5cbiAqXG4gKiBUaGlzIGZ1bmN0aW9uIGRvZXMgbm90IGhhbmRsZSBzaGFyZGluZy5cbiAqXG4gKiBUaGlzIGZ1bmN0aW9uIGlzIHRoZSByZXZlcnNlIG9mIGBlbmNvZGVXZWlnaHRzYC5cbiAqXG4gKiBAcGFyYW0gYnVmZmVyIEEgZmxhdCBBcnJheUJ1ZmZlciBjYXJyeWluZyB0aGUgYmluYXJ5IHZhbHVlcyBvZiB0aGUgdGVuc29yc1xuICogICBjb25jYXRlbmF0ZWQgaW4gdGhlIG9yZGVyIHNwZWNpZmllZCBpbiBgc3BlY3NgLlxuICogQHBhcmFtIHNwZWNzIFNwZWNpZmljYXRpb25zIG9mIHRoZSBuYW1lcywgZHR5cGVzIGFuZCBzaGFwZXMgb2YgdGhlIHRlbnNvcnNcbiAqICAgd2hvc2UgdmFsdWUgYXJlIGVuY29kZWQgYnkgYGJ1ZmZlcmAuXG4gKiBAcmV0dXJuIEEgbWFwIGZyb20gdGVuc29yIG5hbWUgdG8gdGVuc29yIHZhbHVlLCB3aXRoIHRoZSBuYW1lcyBjb3JyZXNwb25kaW5nXG4gKiAgIHRvIG5hbWVzIGluIGBzcGVjc2AuXG4gKiBAdGhyb3dzIEVycm9yLCBpZiBhbnkgb2YgdGhlIHRlbnNvcnMgaGFzIHVuc3VwcG9ydGVkIGR0eXBlLlxuICovXG5leHBvcnQgZnVuY3Rpb24gZGVjb2RlV2VpZ2h0cyhcbiAgICBidWZmZXI6IEFycmF5QnVmZmVyLCBzcGVjczogV2VpZ2h0c01hbmlmZXN0RW50cnlbXSk6IE5hbWVkVGVuc29yTWFwIHtcbiAgLy8gVE9ETyhhZGFyb2IsIGNhaXMpOiBTdXBwb3J0IHF1YW50aXphdGlvbi5cbiAgY29uc3Qgb3V0OiBOYW1lZFRlbnNvck1hcCA9IHt9O1xuICBsZXQgZmxvYXQxNkRlY29kZTogKGJ1ZmZlcjogVWludDE2QXJyYXkpID0+IEZsb2F0MzJBcnJheSB8IHVuZGVmaW5lZDtcbiAgbGV0IG9mZnNldCA9IDA7XG4gIGZvciAoY29uc3Qgc3BlYyBvZiBzcGVjcykge1xuICAgIGNvbnN0IG5hbWUgPSBzcGVjLm5hbWU7XG4gICAgY29uc3QgZHR5cGUgPSBzcGVjLmR0eXBlO1xuICAgIGNvbnN0IHNoYXBlID0gc3BlYy5zaGFwZTtcbiAgICBjb25zdCBzaXplID0gc2l6ZUZyb21TaGFwZShzaGFwZSk7XG4gICAgbGV0IHZhbHVlczogVHlwZWRBcnJheXxzdHJpbmdbXXxVaW50OEFycmF5W107XG5cbiAgICBpZiAoJ3F1YW50aXphdGlvbicgaW4gc3BlYykge1xuICAgICAgY29uc3QgcXVhbnRpemF0aW9uID0gc3BlYy5xdWFudGl6YXRpb247XG4gICAgICBpZiAocXVhbnRpemF0aW9uLmR0eXBlID09PSAndWludDgnIHx8IHF1YW50aXphdGlvbi5kdHlwZSA9PT0gJ3VpbnQxNicpIHtcbiAgICAgICAgaWYgKCEoJ21pbicgaW4gcXVhbnRpemF0aW9uICYmICdzY2FsZScgaW4gcXVhbnRpemF0aW9uKSkge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgYFdlaWdodCAke3NwZWMubmFtZX0gd2l0aCBxdWFudGl6YXRpb24gJHtxdWFudGl6YXRpb24uZHR5cGV9IGAgK1xuICAgICAgICAgICAgICBgZG9lc24ndCBoYXZlIGNvcnJlc3BvbmRpbmcgbWV0YWRhdGEgbWluIGFuZCBzY2FsZS5gKTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIGlmIChxdWFudGl6YXRpb24uZHR5cGUgPT09ICdmbG9hdDE2Jykge1xuICAgICAgICBpZiAoZHR5cGUgIT09ICdmbG9hdDMyJykge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgYFdlaWdodCAke3NwZWMubmFtZX0gaXMgcXVhbnRpemVkIHdpdGggJHtxdWFudGl6YXRpb24uZHR5cGV9IGAgK1xuICAgICAgICAgICAgICBgd2hpY2ggb25seSBzdXBwb3J0cyB3ZWlnaHRzIG9mIHR5cGUgZmxvYXQzMiBub3QgJHtkdHlwZX0uYCk7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgIGBXZWlnaHQgJHtzcGVjLm5hbWV9IGhhcyB1bmtub3duIGAgK1xuICAgICAgICAgICAgYHF1YW50aXphdGlvbiBkdHlwZSAke3F1YW50aXphdGlvbi5kdHlwZX0uIGAgK1xuICAgICAgICAgICAgYFN1cHBvcnRlZCBxdWFudGl6YXRpb24gZHR5cGVzIGFyZTogYCArXG4gICAgICAgICAgICBgJ3VpbnQ4JywgJ3VpbnQxNicsIGFuZCAnZmxvYXQxNicuYCk7XG4gICAgICB9XG4gICAgICBjb25zdCBxdWFudGl6YXRpb25TaXplRmFjdG9yID0gRFRZUEVfVkFMVUVfU0laRV9NQVBbcXVhbnRpemF0aW9uLmR0eXBlXTtcbiAgICAgIGNvbnN0IGJ5dGVCdWZmZXIgPVxuICAgICAgICAgIGJ1ZmZlci5zbGljZShvZmZzZXQsIG9mZnNldCArIHNpemUgKiBxdWFudGl6YXRpb25TaXplRmFjdG9yKTtcbiAgICAgIGNvbnN0IHF1YW50aXplZEFycmF5ID0gKHF1YW50aXphdGlvbi5kdHlwZSA9PT0gJ3VpbnQ4JykgP1xuICAgICAgICAgIG5ldyBVaW50OEFycmF5KGJ5dGVCdWZmZXIpIDpcbiAgICAgICAgICBuZXcgVWludDE2QXJyYXkoYnl0ZUJ1ZmZlcik7XG4gICAgICBpZiAoZHR5cGUgPT09ICdmbG9hdDMyJykge1xuICAgICAgICBpZiAocXVhbnRpemF0aW9uLmR0eXBlID09PSAndWludDgnIHx8IHF1YW50aXphdGlvbi5kdHlwZSA9PT0gJ3VpbnQxNicpIHtcbiAgICAgICAgICB2YWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHF1YW50aXplZEFycmF5Lmxlbmd0aCk7XG4gICAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBxdWFudGl6ZWRBcnJheS5sZW5ndGg7IGkrKykge1xuICAgICAgICAgICAgY29uc3QgdiA9IHF1YW50aXplZEFycmF5W2ldO1xuICAgICAgICAgICAgdmFsdWVzW2ldID0gdiAqIHF1YW50aXphdGlvbi5zY2FsZSArIHF1YW50aXphdGlvbi5taW47XG4gICAgICAgICAgfVxuICAgICAgICB9IGVsc2UgaWYgKHF1YW50aXphdGlvbi5kdHlwZSA9PT0gJ2Zsb2F0MTYnKSB7XG4gICAgICAgICAgaWYgKGZsb2F0MTZEZWNvZGUgPT09IHVuZGVmaW5lZCkge1xuICAgICAgICAgICAgZmxvYXQxNkRlY29kZSA9IGdldEZsb2F0MTZEZWNvZGVyKCk7XG4gICAgICAgICAgfVxuICAgICAgICAgIHZhbHVlcyA9IGZsb2F0MTZEZWNvZGUocXVhbnRpemVkQXJyYXkgYXMgVWludDE2QXJyYXkpO1xuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgYFVuc3VwcG9ydGVkIHF1YW50aXphdGlvbiB0eXBlICR7cXVhbnRpemF0aW9uLmR0eXBlfSBgICtcbiAgICAgICAgICAgICAgYGZvciB3ZWlnaHQgdHlwZSBmbG9hdDMyLmApO1xuICAgICAgICB9XG4gICAgICB9IGVsc2UgaWYgKGR0eXBlID09PSAnaW50MzInKSB7XG4gICAgICAgIGlmIChxdWFudGl6YXRpb24uZHR5cGUgIT09ICd1aW50OCcgJiYgcXVhbnRpemF0aW9uLmR0eXBlICE9PSAndWludDE2Jykge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgYFVuc3VwcG9ydGVkIHF1YW50aXphdGlvbiB0eXBlICR7cXVhbnRpemF0aW9uLmR0eXBlfSBgICtcbiAgICAgICAgICAgICAgYGZvciB3ZWlnaHQgdHlwZSBpbnQzMi5gKTtcbiAgICAgICAgfVxuICAgICAgICB2YWx1ZXMgPSBuZXcgSW50MzJBcnJheShxdWFudGl6ZWRBcnJheS5sZW5ndGgpO1xuICAgICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHF1YW50aXplZEFycmF5Lmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgY29uc3QgdiA9IHF1YW50aXplZEFycmF5W2ldO1xuICAgICAgICAgIHZhbHVlc1tpXSA9IE1hdGgucm91bmQodiAqIHF1YW50aXphdGlvbi5zY2FsZSArIHF1YW50aXphdGlvbi5taW4pO1xuICAgICAgICB9XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoYFVuc3VwcG9ydGVkIGR0eXBlIGluIHdlaWdodCAnJHtuYW1lfSc6ICR7ZHR5cGV9YCk7XG4gICAgICB9XG4gICAgICBvZmZzZXQgKz0gc2l6ZSAqIHF1YW50aXphdGlvblNpemVGYWN0b3I7XG4gICAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIGNvbnN0IHNpemUgPSBzaXplRnJvbVNoYXBlKHNwZWMuc2hhcGUpO1xuICAgICAgdmFsdWVzID0gW107XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IHNpemU7IGkrKykge1xuICAgICAgICBjb25zdCBieXRlTGVuZ3RoID0gbmV3IFVpbnQzMkFycmF5KFxuICAgICAgICAgICAgYnVmZmVyLnNsaWNlKG9mZnNldCwgb2Zmc2V0ICsgTlVNX0JZVEVTX1NUUklOR19MRU5HVEgpKVswXTtcbiAgICAgICAgb2Zmc2V0ICs9IE5VTV9CWVRFU19TVFJJTkdfTEVOR1RIO1xuICAgICAgICBjb25zdCBieXRlcyA9IG5ldyBVaW50OEFycmF5KGJ1ZmZlci5zbGljZShvZmZzZXQsIG9mZnNldCArIGJ5dGVMZW5ndGgpKTtcbiAgICAgICAgKHZhbHVlcyBhcyBVaW50OEFycmF5W10pLnB1c2goYnl0ZXMpO1xuICAgICAgICBvZmZzZXQgKz0gYnl0ZUxlbmd0aDtcbiAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgZHR5cGVGYWN0b3IgPSBEVFlQRV9WQUxVRV9TSVpFX01BUFtkdHlwZV07XG4gICAgICBjb25zdCBieXRlQnVmZmVyID0gYnVmZmVyLnNsaWNlKG9mZnNldCwgb2Zmc2V0ICsgc2l6ZSAqIGR0eXBlRmFjdG9yKTtcblxuICAgICAgaWYgKGR0eXBlID09PSAnZmxvYXQzMicpIHtcbiAgICAgICAgdmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShieXRlQnVmZmVyKTtcbiAgICAgIH0gZWxzZSBpZiAoZHR5cGUgPT09ICdpbnQzMicpIHtcbiAgICAgICAgdmFsdWVzID0gbmV3IEludDMyQXJyYXkoYnl0ZUJ1ZmZlcik7XG4gICAgICB9IGVsc2UgaWYgKGR0eXBlID09PSAnYm9vbCcpIHtcbiAgICAgICAgdmFsdWVzID0gbmV3IFVpbnQ4QXJyYXkoYnl0ZUJ1ZmZlcik7XG4gICAgICB9IGVsc2UgaWYgKGR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICAgICB2YWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGJ5dGVCdWZmZXIpO1xuICAgICAgICBjb25zdCByZWFsID0gbmV3IEZsb2F0MzJBcnJheSh2YWx1ZXMubGVuZ3RoIC8gMik7XG4gICAgICAgIGNvbnN0IGltYWdlID0gbmV3IEZsb2F0MzJBcnJheSh2YWx1ZXMubGVuZ3RoIC8gMik7XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcmVhbC5sZW5ndGg7IGkrKykge1xuICAgICAgICAgIHJlYWxbaV0gPSB2YWx1ZXNbaSAqIDJdO1xuICAgICAgICAgIGltYWdlW2ldID0gdmFsdWVzW2kgKiAyICsgMV07XG4gICAgICAgIH1cbiAgICAgICAgY29uc3QgcmVhbFRlbnNvciA9IHRlbnNvcihyZWFsLCBzaGFwZSwgJ2Zsb2F0MzInKTtcbiAgICAgICAgY29uc3QgaW1hZ2VUZW5zb3IgPSB0ZW5zb3IoaW1hZ2UsIHNoYXBlLCAnZmxvYXQzMicpO1xuICAgICAgICBvdXRbbmFtZV0gPSBjb21wbGV4KHJlYWxUZW5zb3IsIGltYWdlVGVuc29yKTtcbiAgICAgICAgcmVhbFRlbnNvci5kaXNwb3NlKCk7XG4gICAgICAgIGltYWdlVGVuc29yLmRpc3Bvc2UoKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihgVW5zdXBwb3J0ZWQgZHR5cGUgaW4gd2VpZ2h0ICcke25hbWV9JzogJHtkdHlwZX1gKTtcbiAgICAgIH1cbiAgICAgIG9mZnNldCArPSBzaXplICogZHR5cGVGYWN0b3I7XG4gICAgfVxuICAgIGlmIChkdHlwZSAhPT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIG91dFtuYW1lXSA9IHRlbnNvcih2YWx1ZXMsIHNoYXBlLCBkdHlwZSk7XG4gICAgfVxuICB9XG4gIHJldHVybiBvdXQ7XG59XG5cbi8qKlxuICogQ29uY2F0ZW5hdGUgVHlwZWRBcnJheXMgaW50byBhbiBBcnJheUJ1ZmZlci5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbmNhdGVuYXRlVHlwZWRBcnJheXMoeHM6IFR5cGVkQXJyYXlbXSk6IEFycmF5QnVmZmVyIHtcbiAgLy8gVE9ETyhhZGFyb2IsIGNhaXMpOiBTdXBwb3J0IHF1YW50aXphdGlvbi5cbiAgaWYgKHhzID09PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBJbnZhbGlkIGlucHV0IHZhbHVlOiAke0pTT04uc3RyaW5naWZ5KHhzKX1gKTtcbiAgfVxuXG4gIGxldCB0b3RhbEJ5dGVMZW5ndGggPSAwO1xuXG4gIC8vIGBub3JtYWxpemVkWHNgIGlzIGhlcmUgZm9yIHRoaXMgcmVhc29uOiBhIGBUeXBlZEFycmF5YCdzIGBidWZmZXInXG4gIC8vIGNhbiBoYXZlIGEgZGlmZmVyZW50IGJ5dGUgbGVuZ3RoIGZyb20gdGhhdCBvZiB0aGUgYFR5cGVkQXJyYXlgIGl0c2VsZixcbiAgLy8gZm9yIGV4YW1wbGUsIHdoZW4gdGhlIGBUeXBlZEFycmF5YCBpcyBjcmVhdGVkIGZyb20gYW4gb2Zmc2V0IGluIGFuXG4gIC8vIGBBcnJheUJ1ZmZlcmAuIGBub3JtbGlhemVkWHNgIGhvbGRzIGBUeXBlZEFycmF5YHMgd2hvc2UgYGJ1ZmZlcmBzIG1hdGNoXG4gIC8vIHRoZSBgVHlwZWRBcnJheWAgaW4gYnl0ZSBsZW5ndGguIElmIGFuIGVsZW1lbnQgb2YgYHhzYCBkb2VzIG5vdCBzaG93XG4gIC8vIHRoaXMgcHJvcGVydHksIGEgbmV3IGBUeXBlZEFycmF5YCB0aGF0IHNhdGlzZnkgdGhpcyBwcm9wZXJ0eSB3aWxsIGJlXG4gIC8vIGNvbnN0cnVjdGVkIGFuZCBwdXNoZWQgaW50byBgbm9ybWFsaXplZFhzYC5cbiAgY29uc3Qgbm9ybWFsaXplZFhzOiBUeXBlZEFycmF5W10gPSBbXTtcbiAgeHMuZm9yRWFjaCgoeDogVHlwZWRBcnJheSkgPT4ge1xuICAgIHRvdGFsQnl0ZUxlbmd0aCArPSB4LmJ5dGVMZW5ndGg7XG4gICAgLy8gdHNsaW50OmRpc2FibGU6bm8tYW55XG4gICAgbm9ybWFsaXplZFhzLnB1c2goXG4gICAgICAgIHguYnl0ZUxlbmd0aCA9PT0geC5idWZmZXIuYnl0ZUxlbmd0aCA/IHggOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBuZXcgKHguY29uc3RydWN0b3IgYXMgYW55KSh4KSk7XG4gICAgaWYgKCEoeCBhcyBhbnkgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkgfHwgeCBhcyBhbnkgaW5zdGFuY2VvZiBJbnQzMkFycmF5IHx8XG4gICAgICAgICAgeCBhcyBhbnkgaW5zdGFuY2VvZiBVaW50OEFycmF5KSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBVbnN1cHBvcnRlZCBUeXBlZEFycmF5IHN1YnR5cGU6ICR7eC5jb25zdHJ1Y3Rvci5uYW1lfWApO1xuICAgIH1cbiAgICAvLyB0c2xpbnQ6ZW5hYmxlOm5vLWFueVxuICB9KTtcblxuICBjb25zdCB5ID0gbmV3IFVpbnQ4QXJyYXkodG90YWxCeXRlTGVuZ3RoKTtcbiAgbGV0IG9mZnNldCA9IDA7XG4gIG5vcm1hbGl6ZWRYcy5mb3JFYWNoKCh4OiBUeXBlZEFycmF5KSA9PiB7XG4gICAgeS5zZXQobmV3IFVpbnQ4QXJyYXkoeC5idWZmZXIpLCBvZmZzZXQpO1xuICAgIG9mZnNldCArPSB4LmJ5dGVMZW5ndGg7XG4gIH0pO1xuXG4gIHJldHVybiB5LmJ1ZmZlcjtcbn1cblxuLy8gVXNlIEJ1ZmZlciBvbiBOb2RlLmpzIGluc3RlYWQgb2YgQmxvYi9hdG9iL2J0b2FcbmNvbnN0IHVzZU5vZGVCdWZmZXIgPSB0eXBlb2YgQnVmZmVyICE9PSAndW5kZWZpbmVkJyAmJlxuICAgICh0eXBlb2YgQmxvYiA9PT0gJ3VuZGVmaW5lZCcgfHwgdHlwZW9mIGF0b2IgPT09ICd1bmRlZmluZWQnIHx8XG4gICAgIHR5cGVvZiBidG9hID09PSAndW5kZWZpbmVkJyk7XG5cbi8qKlxuICogQ2FsY3VsYXRlIHRoZSBieXRlIGxlbmd0aCBvZiBhIEphdmFTY3JpcHQgc3RyaW5nLlxuICpcbiAqIE5vdGUgdGhhdCBhIEphdmFTY3JpcHQgc3RyaW5nIGNhbiBjb250YWluIHdpZGUgY2hhcmFjdGVycywgdGhlcmVmb3JlIHRoZVxuICogbGVuZ3RoIG9mIHRoZSBzdHJpbmcgaXMgbm90IG5lY2Vzc2FyaWx5IGVxdWFsIHRvIHRoZSBieXRlIGxlbmd0aC5cbiAqXG4gKiBAcGFyYW0gc3RyIElucHV0IHN0cmluZy5cbiAqIEByZXR1cm5zIEJ5dGUgbGVuZ3RoLlxuICovXG5leHBvcnQgZnVuY3Rpb24gc3RyaW5nQnl0ZUxlbmd0aChzdHI6IHN0cmluZyk6IG51bWJlciB7XG4gIGlmICh1c2VOb2RlQnVmZmVyKSB7XG4gICAgcmV0dXJuIEJ1ZmZlci5ieXRlTGVuZ3RoKHN0cik7XG4gIH1cbiAgcmV0dXJuIG5ldyBCbG9iKFtzdHJdKS5zaXplO1xufVxuXG4vKipcbiAqIEVuY29kZSBhbiBBcnJheUJ1ZmZlciBhcyBhIGJhc2U2NCBlbmNvZGVkIHN0cmluZy5cbiAqXG4gKiBAcGFyYW0gYnVmZmVyIGBBcnJheUJ1ZmZlcmAgdG8gYmUgY29udmVydGVkLlxuICogQHJldHVybnMgQSBzdHJpbmcgdGhhdCBiYXNlNjQtZW5jb2RlcyBgYnVmZmVyYC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGFycmF5QnVmZmVyVG9CYXNlNjRTdHJpbmcoYnVmZmVyOiBBcnJheUJ1ZmZlcik6IHN0cmluZyB7XG4gIGlmICh1c2VOb2RlQnVmZmVyKSB7XG4gICAgcmV0dXJuIEJ1ZmZlci5mcm9tKGJ1ZmZlcikudG9TdHJpbmcoJ2Jhc2U2NCcpO1xuICB9XG4gIGNvbnN0IGJ1ZiA9IG5ldyBVaW50OEFycmF5KGJ1ZmZlcik7XG4gIGxldCBzID0gJyc7XG4gIGZvciAobGV0IGkgPSAwLCBsID0gYnVmLmxlbmd0aDsgaSA8IGw7IGkrKykge1xuICAgIHMgKz0gU3RyaW5nLmZyb21DaGFyQ29kZShidWZbaV0pO1xuICB9XG4gIHJldHVybiBidG9hKHMpO1xufVxuXG4vKipcbiAqIERlY29kZSBhIGJhc2U2NCBzdHJpbmcgYXMgYW4gQXJyYXlCdWZmZXIuXG4gKlxuICogQHBhcmFtIHN0ciBCYXNlNjQgc3RyaW5nLlxuICogQHJldHVybnMgRGVjb2RlZCBgQXJyYXlCdWZmZXJgLlxuICovXG5leHBvcnQgZnVuY3Rpb24gYmFzZTY0U3RyaW5nVG9BcnJheUJ1ZmZlcihzdHI6IHN0cmluZyk6IEFycmF5QnVmZmVyIHtcbiAgaWYgKHVzZU5vZGVCdWZmZXIpIHtcbiAgICBjb25zdCBidWYgPSBCdWZmZXIuZnJvbShzdHIsICdiYXNlNjQnKTtcbiAgICByZXR1cm4gYnVmLmJ1ZmZlci5zbGljZShidWYuYnl0ZU9mZnNldCwgYnVmLmJ5dGVPZmZzZXQgKyBidWYuYnl0ZUxlbmd0aCk7XG4gIH1cbiAgY29uc3QgcyA9IGF0b2Ioc3RyKTtcbiAgY29uc3QgYnVmZmVyID0gbmV3IFVpbnQ4QXJyYXkocy5sZW5ndGgpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IHMubGVuZ3RoOyArK2kpIHtcbiAgICBidWZmZXIuc2V0KFtzLmNoYXJDb2RlQXQoaSldLCBpKTtcbiAgfVxuICByZXR1cm4gYnVmZmVyLmJ1ZmZlcjtcbn1cblxuLyoqXG4gKiBDb25jYXRlbmF0ZSBhIG51bWJlciBvZiBBcnJheUJ1ZmZlcnMgaW50byBvbmUuXG4gKlxuICogQHBhcmFtIGJ1ZmZlcnMgQSBudW1iZXIgb2YgYXJyYXkgYnVmZmVycyB0byBjb25jYXRlbmF0ZS5cbiAqIEByZXR1cm5zIFJlc3VsdCBvZiBjb25jYXRlbmF0aW5nIGBidWZmZXJzYCBpbiBvcmRlci5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbmNhdGVuYXRlQXJyYXlCdWZmZXJzKGJ1ZmZlcnM6IEFycmF5QnVmZmVyW10pOiBBcnJheUJ1ZmZlciB7XG4gIGlmIChidWZmZXJzLmxlbmd0aCA9PT0gMSkge1xuICAgIHJldHVybiBidWZmZXJzWzBdO1xuICB9XG5cbiAgbGV0IHRvdGFsQnl0ZUxlbmd0aCA9IDA7XG4gIGJ1ZmZlcnMuZm9yRWFjaCgoYnVmZmVyOiBBcnJheUJ1ZmZlcikgPT4ge1xuICAgIHRvdGFsQnl0ZUxlbmd0aCArPSBidWZmZXIuYnl0ZUxlbmd0aDtcbiAgfSk7XG5cbiAgY29uc3QgdGVtcCA9IG5ldyBVaW50OEFycmF5KHRvdGFsQnl0ZUxlbmd0aCk7XG4gIGxldCBvZmZzZXQgPSAwO1xuICBidWZmZXJzLmZvckVhY2goKGJ1ZmZlcjogQXJyYXlCdWZmZXIpID0+IHtcbiAgICB0ZW1wLnNldChuZXcgVWludDhBcnJheShidWZmZXIpLCBvZmZzZXQpO1xuICAgIG9mZnNldCArPSBidWZmZXIuYnl0ZUxlbmd0aDtcbiAgfSk7XG4gIHJldHVybiB0ZW1wLmJ1ZmZlcjtcbn1cblxuLyoqXG4gKiBHZXQgdGhlIGJhc2VuYW1lIG9mIGEgcGF0aC5cbiAqXG4gKiBCZWhhdmVzIGluIGEgd2F5IGFuYWxvZ291cyB0byBMaW51eCdzIGJhc2VuYW1lIGNvbW1hbmQuXG4gKlxuICogQHBhcmFtIHBhdGhcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGJhc2VuYW1lKHBhdGg6IHN0cmluZyk6IHN0cmluZyB7XG4gIGNvbnN0IFNFUEFSQVRPUiA9ICcvJztcbiAgcGF0aCA9IHBhdGgudHJpbSgpO1xuICB3aGlsZSAocGF0aC5lbmRzV2l0aChTRVBBUkFUT1IpKSB7XG4gICAgcGF0aCA9IHBhdGguc2xpY2UoMCwgcGF0aC5sZW5ndGggLSAxKTtcbiAgfVxuICBjb25zdCBpdGVtcyA9IHBhdGguc3BsaXQoU0VQQVJBVE9SKTtcbiAgcmV0dXJuIGl0ZW1zW2l0ZW1zLmxlbmd0aCAtIDFdO1xufVxuXG4vKipcbiAqIENyZWF0ZSBgTW9kZWxKU09OYCBmcm9tIGBNb2RlbEFydGlmYWN0c2AuXG4gKlxuICogQHBhcmFtIGFydGlmYWN0cyBNb2RlbCBhcnRpZmFjdHMsIGRlc2NyaWJpbmcgdGhlIG1vZGVsIGFuZCBpdHMgd2VpZ2h0cy5cbiAqIEBwYXJhbSBtYW5pZmVzdCBXZWlnaHQgbWFuaWZlc3QsIGRlc2NyaWJpbmcgd2hlcmUgdGhlIHdlaWdodHMgb2YgdGhlXG4gKiAgICAgYE1vZGVsQXJ0aWZhY3RzYCBhcmUgc3RvcmVkLCBhbmQgc29tZSBtZXRhZGF0YSBhYm91dCB0aGVtLlxuICogQHJldHVybnMgT2JqZWN0IHJlcHJlc2VudGluZyB0aGUgYG1vZGVsLmpzb25gIGZpbGUgZGVzY3JpYmluZyB0aGUgbW9kZWxcbiAqICAgICBhcnRpZmFjdHMgYW5kIHdlaWdodHNcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGdldE1vZGVsSlNPTkZvck1vZGVsQXJ0aWZhY3RzKFxuICAgIGFydGlmYWN0czogTW9kZWxBcnRpZmFjdHMsIG1hbmlmZXN0OiBXZWlnaHRzTWFuaWZlc3RDb25maWcpOiBNb2RlbEpTT04ge1xuICBjb25zdCByZXN1bHQ6IE1vZGVsSlNPTiA9IHtcbiAgICBtb2RlbFRvcG9sb2d5OiBhcnRpZmFjdHMubW9kZWxUb3BvbG9neSxcbiAgICBmb3JtYXQ6IGFydGlmYWN0cy5mb3JtYXQsXG4gICAgZ2VuZXJhdGVkQnk6IGFydGlmYWN0cy5nZW5lcmF0ZWRCeSxcbiAgICBjb252ZXJ0ZWRCeTogYXJ0aWZhY3RzLmNvbnZlcnRlZEJ5LFxuICAgIHdlaWdodHNNYW5pZmVzdDogbWFuaWZlc3RcbiAgfTtcbiAgaWYgKGFydGlmYWN0cy5zaWduYXR1cmUgIT0gbnVsbCkge1xuICAgIHJlc3VsdC5zaWduYXR1cmUgPSBhcnRpZmFjdHMuc2lnbmF0dXJlO1xuICB9XG4gIGlmIChhcnRpZmFjdHMudXNlckRlZmluZWRNZXRhZGF0YSAhPSBudWxsKSB7XG4gICAgcmVzdWx0LnVzZXJEZWZpbmVkTWV0YWRhdGEgPSBhcnRpZmFjdHMudXNlckRlZmluZWRNZXRhZGF0YTtcbiAgfVxuICBpZiAoYXJ0aWZhY3RzLm1vZGVsSW5pdGlhbGl6ZXIgIT0gbnVsbCkge1xuICAgIHJlc3VsdC5tb2RlbEluaXRpYWxpemVyID0gYXJ0aWZhY3RzLm1vZGVsSW5pdGlhbGl6ZXI7XG4gIH1cbiAgaWYgKGFydGlmYWN0cy50cmFpbmluZ0NvbmZpZyAhPSBudWxsKSB7XG4gICAgcmVzdWx0LnRyYWluaW5nQ29uZmlnID0gYXJ0aWZhY3RzLnRyYWluaW5nQ29uZmlnO1xuICB9XG4gIHJldHVybiByZXN1bHQ7XG59XG5cbi8qKlxuICogQ3JlYXRlIGBNb2RlbEFydGlmYWN0c2AgZnJvbSBhIEpTT04gZmlsZS5cbiAqXG4gKiBAcGFyYW0gbW9kZWxKU09OIE9iamVjdCBjb250YWluaW5nIHRoZSBwYXJzZWQgSlNPTiBvZiBgbW9kZWwuanNvbmBcbiAqIEBwYXJhbSBsb2FkV2VpZ2h0cyBGdW5jdGlvbiB0aGF0IHRha2VzIHRoZSBKU09OIGZpbGUncyB3ZWlnaHRzIG1hbmlmZXN0LFxuICogICAgIHJlYWRzIHdlaWdodHMgZnJvbSB0aGUgbGlzdGVkIHBhdGgocyksIGFuZCByZXR1cm5zIGEgUHJvbWlzZSBvZiB0aGVcbiAqICAgICB3ZWlnaHQgbWFuaWZlc3QgZW50cmllcyBhbG9uZyB3aXRoIHRoZSB3ZWlnaHRzIGRhdGEuXG4gKiBAcmV0dXJucyBBIFByb21pc2Ugb2YgdGhlIGBNb2RlbEFydGlmYWN0c2AsIGFzIGRlc2NyaWJlZCBieSB0aGUgSlNPTiBmaWxlLlxuICovXG5leHBvcnQgYXN5bmMgZnVuY3Rpb24gZ2V0TW9kZWxBcnRpZmFjdHNGb3JKU09OKFxuICAgIG1vZGVsSlNPTjogTW9kZWxKU09OLFxuICAgIGxvYWRXZWlnaHRzOiAod2VpZ2h0c01hbmlmZXN0OiBXZWlnaHRzTWFuaWZlc3RDb25maWcpID0+IFByb21pc2U8W1xuICAgICAgLyogd2VpZ2h0U3BlY3MgKi8gV2VpZ2h0c01hbmlmZXN0RW50cnlbXSwgLyogd2VpZ2h0RGF0YSAqLyBBcnJheUJ1ZmZlclxuICAgIF0+KTogUHJvbWlzZTxNb2RlbEFydGlmYWN0cz4ge1xuICBjb25zdCBtb2RlbEFydGlmYWN0czogTW9kZWxBcnRpZmFjdHMgPSB7XG4gICAgbW9kZWxUb3BvbG9neTogbW9kZWxKU09OLm1vZGVsVG9wb2xvZ3ksXG4gICAgZm9ybWF0OiBtb2RlbEpTT04uZm9ybWF0LFxuICAgIGdlbmVyYXRlZEJ5OiBtb2RlbEpTT04uZ2VuZXJhdGVkQnksXG4gICAgY29udmVydGVkQnk6IG1vZGVsSlNPTi5jb252ZXJ0ZWRCeVxuICB9O1xuXG4gIGlmIChtb2RlbEpTT04udHJhaW5pbmdDb25maWcgIT0gbnVsbCkge1xuICAgIG1vZGVsQXJ0aWZhY3RzLnRyYWluaW5nQ29uZmlnID0gbW9kZWxKU09OLnRyYWluaW5nQ29uZmlnO1xuICB9XG4gIGlmIChtb2RlbEpTT04ud2VpZ2h0c01hbmlmZXN0ICE9IG51bGwpIHtcbiAgICBjb25zdCBbd2VpZ2h0U3BlY3MsIHdlaWdodERhdGFdID1cbiAgICAgICAgYXdhaXQgbG9hZFdlaWdodHMobW9kZWxKU09OLndlaWdodHNNYW5pZmVzdCk7XG4gICAgbW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MgPSB3ZWlnaHRTcGVjcztcbiAgICBtb2RlbEFydGlmYWN0cy53ZWlnaHREYXRhID0gd2VpZ2h0RGF0YTtcbiAgfVxuICBpZiAobW9kZWxKU09OLnNpZ25hdHVyZSAhPSBudWxsKSB7XG4gICAgbW9kZWxBcnRpZmFjdHMuc2lnbmF0dXJlID0gbW9kZWxKU09OLnNpZ25hdHVyZTtcbiAgfVxuICBpZiAobW9kZWxKU09OLnVzZXJEZWZpbmVkTWV0YWRhdGEgIT0gbnVsbCkge1xuICAgIG1vZGVsQXJ0aWZhY3RzLnVzZXJEZWZpbmVkTWV0YWRhdGEgPSBtb2RlbEpTT04udXNlckRlZmluZWRNZXRhZGF0YTtcbiAgfVxuICBpZiAobW9kZWxKU09OLm1vZGVsSW5pdGlhbGl6ZXIgIT0gbnVsbCkge1xuICAgIG1vZGVsQXJ0aWZhY3RzLm1vZGVsSW5pdGlhbGl6ZXIgPSBtb2RlbEpTT04ubW9kZWxJbml0aWFsaXplcjtcbiAgfVxuXG4gIHJldHVybiBtb2RlbEFydGlmYWN0cztcbn1cblxuLyoqXG4gKiBQb3B1bGF0ZSBNb2RlbEFydGlmYWN0c0luZm8gZmllbGRzIGZvciBhIG1vZGVsIHdpdGggSlNPTiB0b3BvbG9neS5cbiAqIEBwYXJhbSBtb2RlbEFydGlmYWN0c1xuICogQHJldHVybnMgQSBNb2RlbEFydGlmYWN0c0luZm8gb2JqZWN0LlxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2V0TW9kZWxBcnRpZmFjdHNJbmZvRm9ySlNPTihtb2RlbEFydGlmYWN0czogTW9kZWxBcnRpZmFjdHMpOlxuICAgIE1vZGVsQXJ0aWZhY3RzSW5mbyB7XG4gIGlmIChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5IGluc3RhbmNlb2YgQXJyYXlCdWZmZXIpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0V4cGVjdGVkIEpTT04gbW9kZWwgdG9wb2xvZ3ksIHJlY2VpdmVkIEFycmF5QnVmZmVyLicpO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICBkYXRlU2F2ZWQ6IG5ldyBEYXRlKCksXG4gICAgbW9kZWxUb3BvbG9neVR5cGU6ICdKU09OJyxcbiAgICBtb2RlbFRvcG9sb2d5Qnl0ZXM6IG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kgPT0gbnVsbCA/XG4gICAgICAgIDAgOlxuICAgICAgICBzdHJpbmdCeXRlTGVuZ3RoKEpTT04uc3RyaW5naWZ5KG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kpKSxcbiAgICB3ZWlnaHRTcGVjc0J5dGVzOiBtb2RlbEFydGlmYWN0cy53ZWlnaHRTcGVjcyA9PSBudWxsID9cbiAgICAgICAgMCA6XG4gICAgICAgIHN0cmluZ0J5dGVMZW5ndGgoSlNPTi5zdHJpbmdpZnkobW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MpKSxcbiAgICB3ZWlnaHREYXRhQnl0ZXM6IG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEgPT0gbnVsbCA/XG4gICAgICAgIDAgOlxuICAgICAgICBtb2RlbEFydGlmYWN0cy53ZWlnaHREYXRhLmJ5dGVMZW5ndGgsXG4gIH07XG59XG5cbi8qKlxuICogQ29tcHV0ZXMgbWFudGlzYSB0YWJsZSBmb3IgY2FzdGluZyBGbG9hdDE2IHRvIEZsb2F0MzJcbiAqIFNlZSBodHRwOi8vd3d3LmZveC10b29sa2l0Lm9yZy9mdHAvZmFzdGhhbGZmbG9hdGNvbnZlcnNpb24ucGRmXG4gKlxuICogQHJldHVybnMgVWludDMyQXJyYXksIDIwNDggbWFudGlzc2EgbG9va3VwIHZhbHVlcy5cbiAqL1xuZnVuY3Rpb24gY29tcHV0ZUZsb2F0MTZNYW50aXNhVGFibGUoKTogVWludDMyQXJyYXkge1xuICBjb25zdCBjb252ZXJ0TWFudGlzc2EgPSAoaTogbnVtYmVyKTogbnVtYmVyID0+IHtcbiAgICBsZXQgbSA9IGkgPDwgMTM7XG4gICAgbGV0IGUgPSAwO1xuXG4gICAgd2hpbGUgKChtICYgMHgwMDgwMDAwMCkgPT09IDApIHtcbiAgICAgIGUgLT0gMHgwMDgwMDAwMDtcbiAgICAgIG0gPDw9IDE7XG4gICAgfVxuICAgIG0gJj0gfjB4MDA4MDAwMDA7XG4gICAgZSArPSAweDM4ODAwMDAwO1xuXG4gICAgcmV0dXJuIG0gfCBlO1xuICB9O1xuXG4gIGNvbnN0IG1hbnRpc2FUYWJsZSA9IG5ldyBVaW50MzJBcnJheSgyMDQ4KTtcblxuICBtYW50aXNhVGFibGVbMF0gPSAwO1xuICBmb3IgKGxldCBpID0gMTsgaSA8IDEwMjQ7IGkrKykge1xuICAgIG1hbnRpc2FUYWJsZVtpXSA9IGNvbnZlcnRNYW50aXNzYShpKTtcbiAgfVxuICBmb3IgKGxldCBpID0gMTAyNDsgaSA8IDIwNDg7IGkrKykge1xuICAgIG1hbnRpc2FUYWJsZVtpXSA9IDB4MzgwMDAwMDAgKyAoKGkgLSAxMDI0KSA8PCAxMyk7XG4gIH1cblxuICByZXR1cm4gbWFudGlzYVRhYmxlO1xufVxuXG4vKipcbiAqIENvbXB1dGVzIGV4cG9uZW50IHRhYmxlIGZvciBjYXN0aW5nIEZsb2F0MTYgdG8gRmxvYXQzMlxuICogU2VlIGh0dHA6Ly93d3cuZm94LXRvb2xraXQub3JnL2Z0cC9mYXN0aGFsZmZsb2F0Y29udmVyc2lvbi5wZGZcbiAqXG4gKiBAcmV0dXJucyBVaW50MzJBcnJheSwgNjQgZXhwb25lbnQgbG9va3VwIHZhbHVlcy5cbiAqL1xuZnVuY3Rpb24gY29tcHV0ZUZsb2F0MTZFeHBvbmVudFRhYmxlKCk6IFVpbnQzMkFycmF5IHtcbiAgY29uc3QgZXhwb25lbnRUYWJsZSA9IG5ldyBVaW50MzJBcnJheSg2NCk7XG5cbiAgZXhwb25lbnRUYWJsZVswXSA9IDA7XG4gIGV4cG9uZW50VGFibGVbMzFdID0gMHg0NzgwMDAwMDtcbiAgZXhwb25lbnRUYWJsZVszMl0gPSAweDgwMDAwMDAwO1xuICBleHBvbmVudFRhYmxlWzYzXSA9IDB4Yzc4MDAwMDA7XG4gIGZvciAobGV0IGkgPSAxOyBpIDwgMzE7IGkrKykge1xuICAgIGV4cG9uZW50VGFibGVbaV0gPSBpIDw8IDIzO1xuICB9XG4gIGZvciAobGV0IGkgPSAzMzsgaSA8IDYzOyBpKyspIHtcbiAgICBleHBvbmVudFRhYmxlW2ldID0gMHg4MDAwMDAwMCArICgoaSAtIDMyKSA8PCAyMyk7XG4gIH1cblxuICByZXR1cm4gZXhwb25lbnRUYWJsZTtcbn1cblxuLyoqXG4gKiBDb21wdXRlcyBvZmZzZXQgdGFibGUgZm9yIGNhc3RpbmcgRmxvYXQxNiB0byBGbG9hdDMyXG4gKiBTZWUgaHR0cDovL3d3dy5mb3gtdG9vbGtpdC5vcmcvZnRwL2Zhc3RoYWxmZmxvYXRjb252ZXJzaW9uLnBkZlxuICpcbiAqIEByZXR1cm5zIFVpbnQzMkFycmF5LCA2ZCBvZmZzZXQgdmFsdWVzLlxuICovXG5mdW5jdGlvbiBjb21wdXRlRmxvYXQxNk9mZnNldFRhYmxlKCk6IFVpbnQzMkFycmF5IHtcbiAgY29uc3Qgb2Zmc2V0VGFibGUgPSBuZXcgVWludDMyQXJyYXkoNjQpO1xuXG4gIGZvciAobGV0IGkgPSAwOyBpIDwgNjQ7IGkrKykge1xuICAgIG9mZnNldFRhYmxlW2ldID0gMTAyNDtcbiAgfVxuICBvZmZzZXRUYWJsZVswXSA9IG9mZnNldFRhYmxlWzMyXSA9IDA7XG5cbiAgcmV0dXJuIG9mZnNldFRhYmxlO1xufVxuXG4vKipcbiAqIFJldHJpZXZlIGEgRmxvYXQxNiBkZWNvZGVyIHdoaWNoIHdpbGwgZGVjb2RlIGEgQnl0ZUFycmF5IG9mIEZsb2F0MTYgdmFsdWVzXG4gKiB0byBhIEZsb2F0MzJBcnJheS5cbiAqXG4gKiBAcmV0dXJucyBGdW5jdGlvbiAoYnVmZmVyOiBVaW50MTZBcnJheSkgPT4gRmxvYXQzMkFycmF5IHdoaWNoIGRlY29kZXNcbiAqICAgICAgICAgIHRoZSBVaW50MTZBcnJheSBvZiBGbG9hdDE2IGJ5dGVzIHRvIGEgRmxvYXQzMkFycmF5LlxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2V0RmxvYXQxNkRlY29kZXIoKTogKGJ1ZmZlcjogVWludDE2QXJyYXkpID0+IEZsb2F0MzJBcnJheSB7XG4gIC8vIEFsZ29yaXRobSBpcyBiYXNlZCBvZmYgb2ZcbiAgLy8gaHR0cDovL3d3dy5mb3gtdG9vbGtpdC5vcmcvZnRwL2Zhc3RoYWxmZmxvYXRjb252ZXJzaW9uLnBkZlxuXG4gIC8vIENhY2hlIGxvb2t1cCB0YWJsZXNcbiAgY29uc3QgbWFudGlzYVRhYmxlID0gY29tcHV0ZUZsb2F0MTZNYW50aXNhVGFibGUoKTtcbiAgY29uc3QgZXhwb25lbnRUYWJsZSA9IGNvbXB1dGVGbG9hdDE2RXhwb25lbnRUYWJsZSgpO1xuICBjb25zdCBvZmZzZXRUYWJsZSA9IGNvbXB1dGVGbG9hdDE2T2Zmc2V0VGFibGUoKTtcblxuICByZXR1cm4gKHF1YW50aXplZEFycmF5OiBVaW50MTZBcnJheSkgPT4ge1xuICAgIGNvbnN0IGJ1ZmZlciA9IG5ldyBBcnJheUJ1ZmZlcig0ICogcXVhbnRpemVkQXJyYXkubGVuZ3RoKTtcbiAgICBjb25zdCBidWZmZXJVaW50MzJWaWV3ID0gbmV3IFVpbnQzMkFycmF5KGJ1ZmZlcik7XG4gICAgZm9yIChsZXQgaW5kZXggPSAwOyBpbmRleCA8IHF1YW50aXplZEFycmF5Lmxlbmd0aDsgaW5kZXgrKykge1xuICAgICAgY29uc3QgZmxvYXQxNkJpdHMgPSBxdWFudGl6ZWRBcnJheVtpbmRleF07XG4gICAgICBjb25zdCBmbG9hdDMyQml0cyA9XG4gICAgICAgICAgbWFudGlzYVRhYmxlW29mZnNldFRhYmxlW2Zsb2F0MTZCaXRzID4+IDEwXSArIChmbG9hdDE2Qml0cyAmIDB4M2ZmKV0gK1xuICAgICAgICAgIGV4cG9uZW50VGFibGVbZmxvYXQxNkJpdHMgPj4gMTBdO1xuICAgICAgYnVmZmVyVWludDMyVmlld1tpbmRleF0gPSBmbG9hdDMyQml0cztcbiAgICB9XG4gICAgcmV0dXJuIG5ldyBGbG9hdDMyQXJyYXkoYnVmZmVyKTtcbiAgfTtcbn1cbiJdfQ==