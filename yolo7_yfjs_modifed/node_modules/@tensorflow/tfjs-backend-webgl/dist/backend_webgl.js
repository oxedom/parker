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
// Import webgl flags.
import './flags_webgl';
import { backend_util, buffer, DataStorage, engine, env, kernel_impls, KernelBackend, nextFrame, scalar, tidy, util } from '@tensorflow/tfjs-core';
import { getWebGLContext } from './canvas_util';
import { DecodeMatrixProgram } from './decode_matrix_gpu';
import { DecodeMatrixPackedProgram } from './decode_matrix_packed_gpu';
import { EncodeFloatProgram } from './encode_float_gpu';
import { EncodeFloatPackedProgram } from './encode_float_packed_gpu';
import { EncodeMatrixProgram } from './encode_matrix_gpu';
import { EncodeMatrixPackedProgram } from './encode_matrix_packed_gpu';
import { GPGPUContext } from './gpgpu_context';
import * as gpgpu_math from './gpgpu_math';
import { getUniformLocations } from './gpgpu_math';
import { simpleAbsImplCPU } from './kernel_utils/shared';
import { PackProgram } from './pack_gpu';
import { ReshapePackedProgram } from './reshape_packed_gpu';
import * as tex_util from './tex_util';
import { TextureUsage } from './tex_util';
import { TextureManager } from './texture_manager';
import * as unary_op from './unaryop_gpu';
import { UnaryOpProgram } from './unaryop_gpu';
import { UnaryOpPackedProgram } from './unaryop_packed_gpu';
import { UnpackProgram } from './unpack_gpu';
import * as webgl_util from './webgl_util';
const whereImpl = kernel_impls.whereImpl;
export const EPSILON_FLOAT32 = 1e-7;
export const EPSILON_FLOAT16 = 1e-4;
const binaryCaches = {};
export function getBinaryCache(webGLVersion) {
    if (webGLVersion in binaryCaches) {
        return binaryCaches[webGLVersion];
    }
    binaryCaches[webGLVersion] = {};
    return binaryCaches[webGLVersion];
}
// Empirically determined constant used to determine size threshold for handing
// off execution to the CPU.
const CPU_HANDOFF_SIZE_THRESHOLD = env().getNumber('CPU_HANDOFF_SIZE_THRESHOLD');
// Empirically determined constant used to decide the number of MB on GPU
// before we warn about high memory use. The MB are this constant * screen area
// * dpi / 1024 / 1024.
const BEFORE_PAGING_CONSTANT = 600;
function numMBBeforeWarning() {
    if (env().global.screen == null) {
        return 1024; // 1 GB.
    }
    return (env().global.screen.height * env().global.screen.width *
        window.devicePixelRatio) *
        BEFORE_PAGING_CONSTANT / 1024 / 1024;
}
export class MathBackendWebGL extends KernelBackend {
    constructor(gpuResource) {
        super();
        // Maps data ids that have a pending read operation, to list of subscribers.
        this.pendingRead = new WeakMap();
        // List of data ids that are scheduled for disposal, but are waiting on a
        // pending read operation.
        this.pendingDisposal = new WeakSet();
        // Used to count the number of 'shallow' sliced tensors that point to the
        // same data id.
        this.dataRefCount = new WeakMap();
        this.numBytesInGPU = 0;
        // Accumulated time spent (including blocking) in uploading data to webgl.
        this.uploadWaitMs = 0;
        // Accumulated time spent (including blocking in downloading data from webgl.
        this.downloadWaitMs = 0;
        // record the last manual GL Flush time.
        this.lastGlFlushTime = 0;
        this.warnedAboutMemory = false;
        this.pendingDeletes = 0;
        this.disposed = false;
        if (!env().getBool('HAS_WEBGL')) {
            throw new Error('WebGL is not supported on this device');
        }
        let newGPGPU;
        if (gpuResource != null) {
            if (gpuResource instanceof GPGPUContext) {
                newGPGPU = gpuResource;
            }
            else {
                const gl = getWebGLContext(env().getNumber('WEBGL_VERSION'), gpuResource);
                newGPGPU = new GPGPUContext(gl);
            }
            this.binaryCache = {};
            this.gpgpuCreatedLocally = false;
        }
        else {
            const gl = getWebGLContext(env().getNumber('WEBGL_VERSION'));
            newGPGPU = new GPGPUContext(gl);
            this.binaryCache = getBinaryCache(env().getNumber('WEBGL_VERSION'));
            this.gpgpuCreatedLocally = true;
        }
        this.gpgpu = newGPGPU;
        this.canvas = this.gpgpu.gl.canvas;
        this.textureManager = new TextureManager(this.gpgpu);
        this.numMBBeforeWarning = numMBBeforeWarning();
        this.texData = new DataStorage(this, engine());
    }
    nextDataId() {
        return MathBackendWebGL.nextDataId++;
    }
    numDataIds() {
        return this.texData.numDataIds() - this.pendingDeletes;
    }
    write(values, shape, dtype) {
        if (env().getBool('WEBGL_CHECK_NUMERICAL_PROBLEMS') ||
            env().getBool('DEBUG')) {
            this.checkNumericalProblems(values);
        }
        if (dtype === 'complex64' && values != null) {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        const dataId = { id: this.nextDataId() };
        this.texData.set(dataId, { shape, dtype, values, usage: TextureUsage.UPLOAD, refCount: 1 });
        return dataId;
    }
    /** Return refCount of a `TensorData`. */
    refCount(dataId) {
        if (this.texData.has(dataId)) {
            const tensorData = this.texData.get(dataId);
            return tensorData.refCount;
        }
        return 0;
    }
    /** Increase refCount of a `TextureData`. */
    incRef(dataId) {
        const texData = this.texData.get(dataId);
        texData.refCount++;
    }
    /** Decrease refCount of a `TextureData`. */
    decRef(dataId) {
        if (this.texData.has(dataId)) {
            const texData = this.texData.get(dataId);
            texData.refCount--;
        }
    }
    move(dataId, values, shape, dtype, refCount) {
        if (env().getBool('DEBUG')) {
            this.checkNumericalProblems(values);
        }
        if (dtype === 'complex64') {
            throw new Error(`Cannot write to a complex64 dtype. ` +
                `Please use tf.complex(real, imag).`);
        }
        this.texData.set(dataId, { shape, dtype, values, usage: TextureUsage.UPLOAD, refCount });
    }
    disposeIntermediateTensorInfo(tensorInfo) {
        this.disposeData(tensorInfo.dataId);
    }
    readSync(dataId) {
        const texData = this.texData.get(dataId);
        const { values, dtype, complexTensorInfos, slice, shape, isPacked } = texData;
        // The presence of `slice` indicates this tensor is a shallow slice of a
        // different tensor, and is using that original tensor's texture. Run
        // `clone` in order to copy that texture and read from it.
        if (slice != null) {
            let program;
            if (isPacked) {
                program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
            }
            else {
                program = new UnaryOpProgram(shape, unary_op.CLONE);
            }
            const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
            const data = this.readSync(res.dataId);
            this.disposeIntermediateTensorInfo(res);
            return data;
        }
        if (values != null) {
            return this.convertAndCacheOnCPU(dataId);
        }
        if (dtype === 'string') {
            return values;
        }
        const shouldTimeProgram = this.activeTimers != null;
        let start;
        if (shouldTimeProgram) {
            start = util.now();
        }
        let result;
        if (dtype === 'complex64') {
            const realValues = this.readSync(complexTensorInfos.real.dataId);
            const imagValues = this.readSync(complexTensorInfos.imag.dataId);
            result = backend_util.mergeRealAndImagArrays(realValues, imagValues);
        }
        else {
            result = this.getValuesFromTexture(dataId);
        }
        if (shouldTimeProgram) {
            this.downloadWaitMs += util.now() - start;
        }
        return this.convertAndCacheOnCPU(dataId, result);
    }
    async read(dataId) {
        if (this.pendingRead.has(dataId)) {
            const subscribers = this.pendingRead.get(dataId);
            return new Promise(resolve => subscribers.push(resolve));
        }
        const texData = this.texData.get(dataId);
        const { values, shape, slice, dtype, complexTensorInfos, isPacked } = texData;
        // The presence of `slice` indicates this tensor is a shallow slice of a
        // different tensor, and is using that original tensor's texture. Run
        // `clone` in order to copy that texture and read from it.
        if (slice != null) {
            let program;
            if (isPacked) {
                program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
            }
            else {
                program = new UnaryOpProgram(shape, unary_op.CLONE);
            }
            const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
            const data = this.read(res.dataId);
            this.disposeIntermediateTensorInfo(res);
            return data;
        }
        if (values != null) {
            return this.convertAndCacheOnCPU(dataId);
        }
        if (env().getBool('DEBUG')) {
            // getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') caused a blocking GPU call.
            // For performance reason, only check it for debugging. In production,
            // it doesn't handle this use case anyway, so behavior is not changed.
            if (!env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED') &&
                env().getNumber('WEBGL_VERSION') === 2) {
                throw new Error(`tensor.data() with WEBGL_DOWNLOAD_FLOAT_ENABLED=false and ` +
                    `WEBGL_VERSION=2 not yet supported.`);
            }
        }
        let buffer = null;
        let tmpDownloadTarget;
        if (dtype !== 'complex64' && env().get('WEBGL_BUFFER_SUPPORTED')) {
            // Possibly copy the texture into a buffer before inserting a fence.
            tmpDownloadTarget = this.decode(dataId);
            const tmpData = this.texData.get(tmpDownloadTarget.dataId);
            buffer = this.gpgpu.createBufferFromTexture(tmpData.texture.texture, ...tex_util.getDenseTexShape(shape));
        }
        this.pendingRead.set(dataId, []);
        if (dtype !== 'complex64') {
            // Create a fence and wait for it to resolve.
            await this.gpgpu.createAndWaitForFence();
        }
        // Download the values from the GPU.
        let vals;
        if (dtype === 'complex64') {
            const ps = await Promise.all([
                this.read(complexTensorInfos.real.dataId),
                this.read(complexTensorInfos.imag.dataId)
            ]);
            const realValues = ps[0];
            const imagValues = ps[1];
            vals = backend_util.mergeRealAndImagArrays(realValues, imagValues);
        }
        else if (buffer == null) {
            vals = this.getValuesFromTexture(dataId);
        }
        else {
            const size = util.sizeFromShape(shape);
            vals = this.gpgpu.downloadFloat32MatrixFromBuffer(buffer, size);
        }
        if (tmpDownloadTarget != null) {
            this.disposeIntermediateTensorInfo(tmpDownloadTarget);
        }
        if (buffer != null) {
            const gl = this.gpgpu.gl;
            webgl_util.callAndCheck(gl, () => gl.deleteBuffer(buffer));
        }
        const dTypeVals = this.convertAndCacheOnCPU(dataId, vals);
        const subscribers = this.pendingRead.get(dataId);
        this.pendingRead.delete(dataId);
        // Notify all pending reads.
        subscribers.forEach(resolve => resolve(dTypeVals));
        if (this.pendingDisposal.has(dataId)) {
            this.pendingDisposal.delete(dataId);
            if (this.disposeData(dataId)) {
                engine().removeDataId(dataId, this);
            }
            this.pendingDeletes--;
        }
        return dTypeVals;
    }
    /**
     * Read tensor to a new texture that is densely packed for ease of use.
     * @param dataId The source tensor.
     * @param options
     *     customTexShape: Optional. If set, will use the user defined texture
     *     shape to create the texture.
     */
    readToGPU(dataId, options = {}) {
        const texData = this.texData.get(dataId);
        const { values, shape, slice, dtype, isPacked, texture } = texData;
        if (dtype === 'complex64') {
            throw new Error('Does not support reading texture for complex64 dtype.');
        }
        // The presence of `slice` indicates this tensor is a shallow slice of a
        // different tensor, and is using that original tensor's texture. Run
        // `clone` in order to copy that texture and read from it.
        if (slice != null) {
            let program;
            if (isPacked) {
                program = new UnaryOpPackedProgram(shape, unary_op.CLONE);
            }
            else {
                program = new UnaryOpProgram(shape, unary_op.CLONE);
            }
            const res = this.runWebGLProgram(program, [{ dataId, shape, dtype }], dtype);
            const gpuResouorce = this.readToGPU(res, options);
            this.disposeIntermediateTensorInfo(res);
            return gpuResouorce;
        }
        if (texture == null) {
            if (values != null) {
                throw new Error('Data is not on GPU but on CPU.');
            }
            else {
                throw new Error('There is no data on GPU or CPU.');
            }
        }
        // Decode the texture so that it is stored densely (using four channels).
        const tmpTarget = this.decode(dataId, options.customTexShape);
        // Make engine track this tensor, so that we can dispose it later.
        const tensorRef = engine().makeTensorFromDataId(tmpTarget.dataId, tmpTarget.shape, tmpTarget.dtype);
        const tmpData = this.texData.get(tmpTarget.dataId);
        return Object.assign({ tensorRef }, tmpData.texture);
    }
    bufferSync(t) {
        const data = this.readSync(t.dataId);
        let decodedData = data;
        if (t.dtype === 'string') {
            try {
                // Decode the bytes into string.
                decodedData = data.map(d => util.decodeString(d));
            }
            catch (_a) {
                throw new Error('Failed to decode encoded string bytes into utf-8');
            }
        }
        return buffer(t.shape, t.dtype, decodedData);
    }
    checkNumericalProblems(values) {
        if (values == null) {
            return;
        }
        for (let i = 0; i < values.length; i++) {
            const num = values[i];
            if (!webgl_util.canBeRepresented(num)) {
                if (env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')) {
                    throw Error(`The value ${num} cannot be represented with your ` +
                        `current settings. Consider enabling float32 rendering: ` +
                        `'tf.env().set('WEBGL_RENDER_FLOAT32_ENABLED', true);'`);
                }
                throw Error(`The value ${num} cannot be represented on this device.`);
            }
        }
    }
    getValuesFromTexture(dataId) {
        const { shape, dtype, isPacked } = this.texData.get(dataId);
        const size = util.sizeFromShape(shape);
        if (env().getBool('WEBGL_DOWNLOAD_FLOAT_ENABLED')) {
            const tmpTarget = this.decode(dataId);
            const tmpData = this.texData.get(tmpTarget.dataId);
            const vals = this.gpgpu
                .downloadMatrixFromPackedTexture(tmpData.texture.texture, ...tex_util.getDenseTexShape(shape))
                .subarray(0, size);
            this.disposeIntermediateTensorInfo(tmpTarget);
            return vals;
        }
        const shouldUsePackedProgram = env().getBool('WEBGL_PACK') && isPacked === true;
        const outputShape = shouldUsePackedProgram ? webgl_util.getShapeAs3D(shape) : shape;
        const program = shouldUsePackedProgram ?
            new EncodeFloatPackedProgram(outputShape) :
            new EncodeFloatProgram(outputShape);
        const output = this.runWebGLProgram(program, [{ shape: outputShape, dtype, dataId }], 'float32');
        const tmpData = this.texData.get(output.dataId);
        const vals = this.gpgpu
            .downloadByteEncodedFloatMatrixFromOutputTexture(tmpData.texture.texture, tmpData.texShape[0], tmpData.texShape[1])
            .subarray(0, size);
        this.disposeIntermediateTensorInfo(output);
        return vals;
    }
    timerAvailable() {
        return env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0;
    }
    time(f) {
        const oldActiveTimers = this.activeTimers;
        const newActiveTimers = [];
        let outerMostTime = false;
        if (this.programTimersStack == null) {
            this.programTimersStack = newActiveTimers;
            outerMostTime = true;
        }
        else {
            this.activeTimers.push(newActiveTimers);
        }
        this.activeTimers = newActiveTimers;
        f();
        // needing to split these up because util.flatten only accepts certain types
        const flattenedActiveTimerQueries = util.flatten(this.activeTimers.map((d) => d.query))
            .filter(d => d != null);
        const flattenedActiveTimerNames = util.flatten(this.activeTimers.map((d) => d.name))
            .filter(d => d != null);
        this.activeTimers = oldActiveTimers;
        if (outerMostTime) {
            this.programTimersStack = null;
        }
        const res = {
            uploadWaitMs: this.uploadWaitMs,
            downloadWaitMs: this.downloadWaitMs,
            kernelMs: null,
            wallMs: null // will be filled by the engine
        };
        return (async () => {
            if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') >
                0) {
                const kernelMs = await Promise.all(flattenedActiveTimerQueries);
                res['kernelMs'] = util.sum(kernelMs);
                res['getExtraProfileInfo'] = () => kernelMs
                    .map((d, i) => ({ name: flattenedActiveTimerNames[i], ms: d }))
                    .map(d => `${d.name}: ${d.ms}`)
                    .join(', ');
            }
            else {
                res['kernelMs'] = {
                    error: 'WebGL query timers are not supported in this environment.'
                };
            }
            this.uploadWaitMs = 0;
            this.downloadWaitMs = 0;
            return res;
        })();
    }
    memory() {
        return {
            unreliable: false,
            numBytesInGPU: this.numBytesInGPU,
            numBytesInGPUAllocated: this.textureManager.numBytesAllocated,
            numBytesInGPUFree: this.textureManager.numBytesFree
        };
    }
    startTimer() {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            return this.gpgpu.beginQuery();
        }
        return { startMs: util.now(), endMs: null };
    }
    endTimer(query) {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            this.gpgpu.endQuery();
            return query;
        }
        query.endMs = util.now();
        return query;
    }
    async getQueryTime(query) {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE') > 0) {
            return this.gpgpu.waitForQueryAndGetTime(query);
        }
        const timerQuery = query;
        return timerQuery.endMs - timerQuery.startMs;
    }
    /**
     * Decrease the RefCount on the dataId and dispose the memory if the dataId
     * has 0 refCount. If there are pending read on the data, the disposal would
     * added to the pending delete queue. Return true if the dataId is removed
     * from backend or the backend does not contain the dataId, false if the
     * dataId is not removed. Memory may or may not be released even when dataId
     * is removed, which also depends on dataRefCount, see `releaseGPU`.
     * @param dataId
     * @oaram force Optional, remove the data regardless of refCount
     */
    disposeData(dataId, force = false) {
        if (this.pendingDisposal.has(dataId)) {
            return false;
        }
        // No-op if already disposed.
        if (!this.texData.has(dataId)) {
            return true;
        }
        // if force flag is set, change refCount to 0, this would ensure disposal
        // when added to the pendingDisposal queue. Memory may or may not be
        // released, which also depends on dataRefCount, see `releaseGPU`.
        if (force) {
            this.texData.get(dataId).refCount = 0;
        }
        else {
            this.texData.get(dataId).refCount--;
        }
        if (!force && this.texData.get(dataId).refCount > 0) {
            return false;
        }
        if (this.pendingRead.has(dataId)) {
            this.pendingDisposal.add(dataId);
            this.pendingDeletes++;
            return false;
        }
        this.releaseGPUData(dataId);
        const { complexTensorInfos } = this.texData.get(dataId);
        if (complexTensorInfos != null) {
            this.disposeData(complexTensorInfos.real.dataId, force);
            this.disposeData(complexTensorInfos.imag.dataId, force);
        }
        this.texData.delete(dataId);
        return true;
    }
    releaseGPUData(dataId) {
        const { texture, dtype, texShape, usage, isPacked, slice } = this.texData.get(dataId);
        const key = slice && slice.origDataId || dataId;
        const refCount = this.dataRefCount.get(key);
        if (refCount > 1) {
            this.dataRefCount.set(key, refCount - 1);
        }
        else {
            this.dataRefCount.delete(key);
            if (texture != null) {
                this.numBytesInGPU -= this.computeBytes(texShape, dtype);
                this.textureManager.releaseTexture(texture, texShape, usage, isPacked);
            }
        }
        const texData = this.texData.get(dataId);
        texData.texture = null;
        texData.texShape = null;
        texData.isPacked = false;
        texData.slice = null;
    }
    getTexture(dataId) {
        this.uploadToGPU(dataId);
        return this.texData.get(dataId).texture.texture;
    }
    /**
     * Returns internal information for the specific data bucket. Used in unit
     * tests.
     */
    getDataInfo(dataId) {
        return this.texData.get(dataId);
    }
    /*
    Tests whether all the inputs to an op are small and on the CPU. This heuristic
    determines when it would be faster to execute a kernel on the CPU. WebGL
    kernels opt into running this check and forwarding when appropriate.
    TODO(https://github.com/tensorflow/tfjs/issues/872): Develop a more
    sustainable strategy for optimizing backend execution of ops.
     */
    shouldExecuteOnCPU(inputs, sizeThreshold = CPU_HANDOFF_SIZE_THRESHOLD) {
        return env().getBool('WEBGL_CPU_FORWARD') &&
            inputs.every(input => this.texData.get(input.dataId).texture == null &&
                util.sizeFromShape(input.shape) < sizeThreshold);
    }
    getGPGPUContext() {
        return this.gpgpu;
    }
    where(condition) {
        backend_util.warn('tf.where() in webgl locks the UI thread. ' +
            'Call tf.whereAsync() instead');
        const condVals = condition.dataSync();
        return whereImpl(condition.shape, condVals);
    }
    packedUnaryOp(x, op, dtype) {
        const program = new UnaryOpPackedProgram(x.shape, op);
        const outInfo = this.compileAndRun(program, [x], dtype);
        return engine().makeTensorFromDataId(outInfo.dataId, outInfo.shape, outInfo.dtype);
    }
    // TODO(msoulanille) remove this once the backend has been modularized
    // a copy is needed here to break a circular dependency.
    // Also remove the op from unary_op.
    abs(x) {
        // TODO: handle cases when x is complex.
        if (this.shouldExecuteOnCPU([x]) && x.dtype !== 'complex64') {
            const outValues = simpleAbsImplCPU(this.texData.get(x.dataId).values);
            return this.makeOutput(x.shape, x.dtype, outValues);
        }
        if (env().getBool('WEBGL_PACK_UNARY_OPERATIONS')) {
            return this.packedUnaryOp(x, unary_op.ABS, x.dtype);
        }
        const program = new UnaryOpProgram(x.shape, unary_op.ABS);
        const outInfo = this.compileAndRun(program, [x]);
        return engine().makeTensorFromDataId(outInfo.dataId, outInfo.shape, outInfo.dtype);
    }
    makeTensorInfo(shape, dtype, values) {
        let dataId;
        if (dtype === 'string' && values != null && values.length > 0 &&
            util.isString(values[0])) {
            const encodedValues = values.map(d => util.encodeString(d));
            dataId = this.write(encodedValues, shape, dtype);
        }
        else {
            dataId = this.write(values, shape, dtype);
        }
        this.texData.get(dataId).usage = null;
        return { dataId, shape, dtype };
    }
    makeOutput(shape, dtype, values) {
        const { dataId } = this.makeTensorInfo(shape, dtype, values);
        return engine().makeTensorFromDataId(dataId, shape, dtype, this);
    }
    unpackTensor(input) {
        const program = new UnpackProgram(input.shape);
        return this.runWebGLProgram(program, [input], input.dtype);
    }
    packTensor(input) {
        const program = new PackProgram(input.shape);
        const preventEagerUnpackingOutput = true;
        return this.runWebGLProgram(program, [input], input.dtype, null /* customUniformValues */, preventEagerUnpackingOutput);
    }
    packedReshape(input, afterShape) {
        const input3DShape = [
            webgl_util.getBatchDim(input.shape),
            ...webgl_util.getRowsCols(input.shape)
        ];
        const input3D = {
            dtype: input.dtype,
            shape: input3DShape,
            dataId: input.dataId
        };
        const afterShapeAs3D = [
            webgl_util.getBatchDim(afterShape), ...webgl_util.getRowsCols(afterShape)
        ];
        const program = new ReshapePackedProgram(afterShapeAs3D, input3DShape);
        const preventEagerUnpackingOfOutput = true;
        const customValues = [input3DShape];
        const output = this.runWebGLProgram(program, [input3D], input.dtype, customValues, preventEagerUnpackingOfOutput);
        return { dataId: output.dataId, shape: afterShape, dtype: output.dtype };
    }
    decode(dataId, customTexShape) {
        const texData = this.texData.get(dataId);
        const { isPacked, shape, dtype } = texData;
        if (customTexShape != null) {
            const size = util.sizeFromShape(shape);
            const texSize = customTexShape[0] * customTexShape[1] * 4;
            util.assert(size <= texSize, () => 'customTexShape is too small. ' +
                'Row * Column * 4 should be equal or larger than the ' +
                'size of the tensor data.');
        }
        const shapeAs3D = webgl_util.getShapeAs3D(shape);
        let program;
        if (isPacked) {
            program = new DecodeMatrixPackedProgram(shapeAs3D);
        }
        else {
            program = new DecodeMatrixProgram(shapeAs3D);
        }
        const preventEagerUnpackingOfOutput = true;
        const customValues = [customTexShape != null ? customTexShape :
                tex_util.getDenseTexShape(shapeAs3D)];
        const out = this.runWebGLProgram(program, [{ shape: shapeAs3D, dtype, dataId }], dtype, customValues, preventEagerUnpackingOfOutput, customTexShape);
        return { dtype, shape, dataId: out.dataId };
    }
    runWebGLProgram(program, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput = false, customTexShape) {
        const output = this.makeTensorInfo(program.outputShape, outputDtype);
        const outData = this.texData.get(output.dataId);
        if (program.packedOutput) {
            outData.isPacked = true;
        }
        if (program.outPackingScheme === tex_util.PackingScheme.DENSE) {
            const texelShape = customTexShape != null ?
                customTexShape :
                tex_util.getDenseTexShape(program.outputShape);
            // For a densely packed output, we explicitly set texShape
            // so it doesn't get assigned later according to our typical packing
            // scheme wherein a single texel can only contain values from adjacent
            // rows/cols.
            outData.texShape = texelShape.map(d => d * 2);
        }
        if (program.outTexUsage != null) {
            outData.usage = program.outTexUsage;
        }
        if (util.sizeFromShape(output.shape) === 0) {
            // Short-circuit the computation since the result is empty (has 0 in its
            // shape).
            outData.values =
                util.getTypedArrayFromDType(output.dtype, 0);
            return output;
        }
        const dataToDispose = [];
        const inputsData = inputs.map(input => {
            if (input.dtype === 'complex64') {
                throw new Error(`GPGPUProgram does not support complex64 input. For complex64 ` +
                    `dtypes, please separate the program into real and imaginary ` +
                    `parts.`);
            }
            let texData = this.texData.get(input.dataId);
            if (texData.texture == null) {
                if (!program.packedInputs &&
                    util.sizeFromShape(input.shape) <=
                        env().getNumber('WEBGL_SIZE_UPLOAD_UNIFORM')) {
                    // Upload small tensors that live on the CPU as uniforms, not as
                    // textures. Do this only when the environment supports 32bit floats
                    // due to problems when comparing 16bit floats with 32bit floats.
                    // TODO(https://github.com/tensorflow/tfjs/issues/821): Make it
                    // possible for packed shaders to sample from uniforms.
                    return {
                        shape: input.shape,
                        texData: null,
                        isUniform: true,
                        uniformValues: texData.values
                    };
                }
                // This ensures that if a packed program's inputs have not yet been
                // uploaded to the GPU, they get uploaded as packed right off the bat.
                if (program.packedInputs) {
                    texData.isPacked = true;
                    texData.shape = input.shape;
                }
            }
            this.uploadToGPU(input.dataId);
            if (!!texData.isPacked !== !!program.packedInputs) {
                input = texData.isPacked ? this.unpackTensor(input) :
                    this.packTensor(input);
                dataToDispose.push(input);
                texData = this.texData.get(input.dataId);
            }
            else if (texData.isPacked &&
                !webgl_util.isReshapeFree(texData.shape, input.shape)) {
                // This is a special case where a texture exists for a tensor
                // but the shapes are incompatible (due to packing constraints) because
                // the tensor did not have a chance to go through the packed reshape
                // shader. This only happens when we reshape the *same* tensor to form
                // *distinct* inputs to an op, e.g. dotting a vector with itself. This
                // case will disappear once packed uploading is the default.
                const savedInput = input;
                const targetShape = input.shape;
                input.shape = texData.shape;
                input = this.packedReshape(input, targetShape);
                dataToDispose.push(input);
                texData = this.texData.get(input.dataId);
                savedInput.shape = targetShape;
            }
            return { shape: input.shape, texData, isUniform: false };
        });
        this.uploadToGPU(output.dataId);
        const outputData = { shape: output.shape, texData: outData, isUniform: false };
        const key = gpgpu_math.makeShaderKey(program, inputsData, outputData);
        const binary = this.getAndSaveBinary(key, () => {
            return gpgpu_math.compileProgram(this.gpgpu, program, inputsData, outputData);
        });
        const shouldTimeProgram = this.activeTimers != null;
        let query;
        if (shouldTimeProgram) {
            query = this.startTimer();
        }
        if (!env().get('ENGINE_COMPILE_ONLY')) {
            gpgpu_math.runProgram(this.gpgpu, binary, inputsData, outputData, customUniformValues);
        }
        dataToDispose.forEach(info => this.disposeIntermediateTensorInfo(info));
        if (shouldTimeProgram) {
            query = this.endTimer(query);
            this.activeTimers.push({ name: program.constructor.name, query: this.getQueryTime(query) });
        }
        const glFlushThreshold = env().get('WEBGL_FLUSH_THRESHOLD');
        // Manually GL flush requested
        if (glFlushThreshold > 0) {
            const time = util.now();
            if ((time - this.lastGlFlushTime) > glFlushThreshold) {
                this.gpgpu.gl.flush();
                this.lastGlFlushTime = time;
            }
        }
        if (!env().getBool('WEBGL_LAZILY_UNPACK') && outData.isPacked &&
            preventEagerUnpackingOfOutput === false) {
            const unpacked = this.unpackTensor(output);
            this.disposeIntermediateTensorInfo(output);
            return unpacked;
        }
        return output;
    }
    compileAndRun(program, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput = false) {
        outputDtype = outputDtype || inputs[0].dtype;
        const outInfo = this.runWebGLProgram(program, inputs, outputDtype, customUniformValues, preventEagerUnpackingOfOutput);
        return outInfo;
    }
    getAndSaveBinary(key, getBinary) {
        if (!(key in this.binaryCache)) {
            this.binaryCache[key] = getBinary();
        }
        return this.binaryCache[key];
    }
    getTextureManager() {
        return this.textureManager;
    }
    dispose() {
        if (this.disposed) {
            return;
        }
        // Avoid disposing the compiled webgl programs during unit testing because
        // it slows down test execution.
        if (!env().getBool('IS_TEST')) {
            const allKeys = Object.keys(this.binaryCache);
            allKeys.forEach(key => {
                this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
                delete this.binaryCache[key];
            });
        }
        this.textureManager.dispose();
        if (this.canvas != null &&
            (typeof (HTMLCanvasElement) !== 'undefined' &&
                this.canvas instanceof HTMLCanvasElement)) {
            this.canvas.remove();
        }
        else {
            this.canvas = null;
        }
        if (this.gpgpuCreatedLocally) {
            this.gpgpu.program = null;
            this.gpgpu.dispose();
        }
        this.disposed = true;
    }
    floatPrecision() {
        if (this.floatPrecisionValue == null) {
            this.floatPrecisionValue = tidy(() => {
                if (!env().get('WEBGL_RENDER_FLOAT32_ENABLED')) {
                    // Momentarily switching DEBUG flag to false so we don't throw an
                    // error trying to upload a small value.
                    const debugFlag = env().getBool('DEBUG');
                    env().set('DEBUG', false);
                    const underflowCheckValue = this.abs(scalar(1e-8)).dataSync()[0];
                    env().set('DEBUG', debugFlag);
                    if (underflowCheckValue > 0) {
                        return 32;
                    }
                }
                return 16;
            });
        }
        return this.floatPrecisionValue;
    }
    /** Returns the smallest representable number.  */
    epsilon() {
        return this.floatPrecision() === 32 ? EPSILON_FLOAT32 : EPSILON_FLOAT16;
    }
    uploadToGPU(dataId) {
        const texData = this.texData.get(dataId);
        const { shape, dtype, values, texture, usage, isPacked } = texData;
        if (texture != null) {
            // Array is already on GPU. No-op.
            return;
        }
        const shouldTimeProgram = this.activeTimers != null;
        let start;
        if (shouldTimeProgram) {
            start = util.now();
        }
        let texShape = texData.texShape;
        if (texShape == null) {
            // This texShape may not be the final texture shape. For packed or dense
            // textures, the texShape will be changed when textures are created.
            texShape = webgl_util.getTextureShapeFromLogicalShape(shape, isPacked);
            texData.texShape = texShape;
        }
        if (values != null) {
            const shapeAs3D = webgl_util.getShapeAs3D(shape);
            let program;
            let width = texShape[1], height = texShape[0];
            const isByteArray = values instanceof Uint8Array || values instanceof Uint8ClampedArray;
            // texture for float array is PhysicalTextureType.PACKED_2X2_FLOAT32, we
            // need to make sure the upload uses the same packed size
            if (isPacked || !isByteArray) {
                [width, height] = tex_util.getPackedMatrixTextureShapeWidthHeight(texShape[0], texShape[1]);
            }
            if (isPacked) {
                program = new EncodeMatrixPackedProgram(shapeAs3D, isByteArray);
            }
            else {
                program = new EncodeMatrixProgram(shapeAs3D, isByteArray);
            }
            // TexShape for float array needs to be the original shape, which byte
            // array needs to be packed size. This allow the data upload shape to be
            // matched with texture creation logic.
            const tempDenseInputTexShape = isByteArray ? [height, width] : texShape;
            const tempDenseInputHandle = this.makeTensorInfo(tempDenseInputTexShape, dtype);
            const tempDenseInputTexData = this.texData.get(tempDenseInputHandle.dataId);
            if (isByteArray) {
                tempDenseInputTexData.usage = TextureUsage.PIXELS;
            }
            else {
                tempDenseInputTexData.usage = TextureUsage.UPLOAD;
            }
            tempDenseInputTexData.texShape = tempDenseInputTexShape;
            this.gpgpu.uploadDenseMatrixToTexture(this.getTexture(tempDenseInputHandle.dataId), width, height, values);
            const customValues = [[height, width]];
            // We want the output to remain packed regardless of the value of
            // WEBGL_PACK.
            const preventEagerUnpacking = true;
            const encodedOutputTarget = this.runWebGLProgram(program, [tempDenseInputHandle], dtype, customValues, preventEagerUnpacking);
            // Have the original texture assume the identity of the encoded output.
            const outputTexData = this.texData.get(encodedOutputTarget.dataId);
            texData.texShape = outputTexData.texShape;
            texData.isPacked = outputTexData.isPacked;
            texData.usage = outputTexData.usage;
            if (!env().get('ENGINE_COMPILE_ONLY')) {
                texData.texture = outputTexData.texture;
                // Once uploaded, don't store the values on cpu.
                texData.values = null;
                this.texData.delete(encodedOutputTarget.dataId);
            }
            else {
                this.disposeData(encodedOutputTarget.dataId);
            }
            this.disposeIntermediateTensorInfo(tempDenseInputHandle);
            if (shouldTimeProgram) {
                this.uploadWaitMs += util.now() - start;
            }
        }
        else {
            const newTexture = this.acquireTexture(texShape, usage, dtype, isPacked);
            texData.texture = newTexture;
        }
    }
    convertAndCacheOnCPU(dataId, float32Values) {
        const texData = this.texData.get(dataId);
        const { dtype } = texData;
        this.releaseGPUData(dataId);
        if (float32Values != null) {
            texData.values = float32ToTypedArray(float32Values, dtype);
        }
        return texData.values;
    }
    acquireTexture(texShape, texType, dtype, isPacked) {
        this.numBytesInGPU += this.computeBytes(texShape, dtype);
        if (!this.warnedAboutMemory &&
            this.numBytesInGPU > this.numMBBeforeWarning * 1024 * 1024) {
            const mb = (this.numBytesInGPU / 1024 / 1024).toFixed(2);
            this.warnedAboutMemory = true;
            console.warn(`High memory usage in GPU: ${mb} MB, ` +
                `most likely due to a memory leak`);
        }
        return this.textureManager.acquireTexture(texShape, texType, isPacked);
    }
    computeBytes(shape, dtype) {
        return shape[0] * shape[1] * util.bytesPerElement(dtype);
    }
    checkCompileCompletion() {
        for (const [, binary] of Object.entries(this.binaryCache)) {
            this.checkCompletion_(binary);
        }
    }
    async checkCompileCompletionAsync() {
        const ps = [];
        if (this.gpgpu.parallelCompilationExtension) {
            for (const [, binary] of Object.entries(this.binaryCache)) {
                ps.push(this.checkCompletionAsync_(binary));
            }
            return Promise.all(ps);
        }
        else {
            for (const [, binary] of Object.entries(this.binaryCache)) {
                const p = new Promise((resolve) => {
                    try {
                        this.checkCompletion_(binary);
                        resolve(true);
                    }
                    catch (error) {
                        throw error;
                    }
                });
                ps.push(p);
            }
            return Promise.all(ps);
        }
    }
    async checkCompletionAsync_(binary) {
        if (this.gpgpu.gl.getProgramParameter(binary.webGLProgram, this.gpgpu.parallelCompilationExtension.COMPLETION_STATUS_KHR)) {
            return this.checkCompletion_(binary);
        }
        else {
            await nextFrame();
            return this.checkCompletionAsync_(binary);
        }
    }
    checkCompletion_(binary) {
        if (this.gpgpu.gl.getProgramParameter(binary.webGLProgram, this.gpgpu.gl.LINK_STATUS) === false) {
            console.log(this.gpgpu.gl.getProgramInfoLog(binary.webGLProgram));
            if (this.gpgpu.gl.getShaderParameter(binary.fragmentShader, this.gpgpu.gl.COMPILE_STATUS) === false) {
                webgl_util.logShaderSourceAndInfoLog(binary.source, this.gpgpu.gl.getShaderInfoLog(binary.fragmentShader));
                throw new Error('Failed to compile fragment shader.');
            }
            throw new Error('Failed to link vertex and fragment shaders.');
        }
        return true;
    }
    getUniformLocations() {
        for (const [, binary] of Object.entries(this.binaryCache)) {
            const { uniformLocations, customUniformLocations, infLoc, nanLoc, inShapesLocations, inTexShapesLocations, outShapeLocation, outShapeStridesLocation, outTexShapeLocation } = getUniformLocations(this.gpgpu, binary.program, binary.webGLProgram);
            binary.uniformLocations = uniformLocations;
            binary.customUniformLocations = customUniformLocations;
            binary.infLoc = infLoc;
            binary.nanLoc = nanLoc;
            binary.inShapesLocations = inShapesLocations;
            binary.inTexShapesLocations = inTexShapesLocations;
            binary.outShapeLocation = outShapeLocation;
            binary.outShapeStridesLocation = outShapeStridesLocation;
            binary.outTexShapeLocation = outTexShapeLocation;
        }
    }
}
MathBackendWebGL.nextDataId = 0;
function float32ToTypedArray(a, dtype) {
    if (dtype === 'float32' || dtype === 'complex64') {
        return a;
    }
    else if (dtype === 'int32' || dtype === 'bool') {
        const result = (dtype === 'int32') ? new Int32Array(a.length) :
            new Uint8Array(a.length);
        for (let i = 0; i < result.length; ++i) {
            result[i] = Math.round(a[i]);
        }
        return result;
    }
    else {
        throw new Error(`Unknown dtype ${dtype}`);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFja2VuZF93ZWJnbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvYmFja2VuZF93ZWJnbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxzQkFBc0I7QUFDdEIsT0FBTyxlQUFlLENBQUM7QUFHdkIsT0FBTyxFQUFDLFlBQVksRUFBaUIsTUFBTSxFQUFVLFdBQVcsRUFBOEMsTUFBTSxFQUFFLEdBQUcsRUFBVyxZQUFZLEVBQUUsYUFBYSxFQUFjLFNBQVMsRUFBeUMsTUFBTSxFQUF3RCxJQUFJLEVBQTBCLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQzlWLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFDOUMsT0FBTyxFQUFDLG1CQUFtQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFDeEQsT0FBTyxFQUFDLHlCQUF5QixFQUFDLE1BQU0sNEJBQTRCLENBQUM7QUFDckUsT0FBTyxFQUFDLGtCQUFrQixFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDdEQsT0FBTyxFQUFDLHdCQUF3QixFQUFDLE1BQU0sMkJBQTJCLENBQUM7QUFDbkUsT0FBTyxFQUFDLG1CQUFtQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFDeEQsT0FBTyxFQUFDLHlCQUF5QixFQUFDLE1BQU0sNEJBQTRCLENBQUM7QUFDckUsT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBQzdDLE9BQU8sS0FBSyxVQUFVLE1BQU0sY0FBYyxDQUFDO0FBQzNDLE9BQU8sRUFBQyxtQkFBbUIsRUFBd0MsTUFBTSxjQUFjLENBQUM7QUFDeEYsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFDdkQsT0FBTyxFQUFDLFdBQVcsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUN2QyxPQUFPLEVBQUMsb0JBQW9CLEVBQUMs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QU0sRUFBRTtRQUNoRCxNQUFNLE1BQU0sR0FBRyxDQUFDLEtBQUssS0FBSyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUM7WUFDMUIsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQzlELEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxNQUFNLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFO1lBQ3RDLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1NBQzlCO1FBQ0QsT0FBTyxNQUEyQixDQUFDO0tBQ3BDO1NBQU07UUFDTCxNQUFNLElBQUksS0FBSyxDQUFDLGlCQUFpQixLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzNDO0FBQ0gsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE3IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLy8gSW1wb3J0IHdlYmdsIGZsYWdzLlxuaW1wb3J0ICcuL2ZsYWdzX3dlYmdsJztcblxuaW1wb3J0ICogYXMgdGYgZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7YmFja2VuZF91dGlsLCBCYWNrZW5kVmFsdWVzLCBidWZmZXIsIERhdGFJZCwgRGF0YVN0b3JhZ2UsIERhdGFUb0dQVVdlYkdMT3B0aW9uLCBEYXRhVHlwZSwgRGF0YVZhbHVlcywgZW5naW5lLCBlbnYsIEdQVURhdGEsIGtlcm5lbF9pbXBscywgS2VybmVsQmFja2VuZCwgTWVtb3J5SW5mbywgbmV4dEZyYW1lLCBOdW1lcmljRGF0YVR5cGUsIFJhbmssIFJlY3Vyc2l2ZUFycmF5LCBzY2FsYXIsIFNoYXBlTWFwLCBUZW5zb3IsIFRlbnNvcjJELCBUZW5zb3JCdWZmZXIsIFRlbnNvckluZm8sIHRpZHksIFRpbWluZ0luZm8sIFR5cGVkQXJyYXksIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQge2dldFdlYkdMQ29udGV4dH0gZnJvbSAnLi9jYW52YXNfdXRpbCc7XG5pbXBvcnQge0RlY29kZU1hdHJpeFByb2dyYW19IGZyb20gJy4vZGVjb2RlX21hdHJpeF9ncHUnO1xuaW1wb3J0IHtEZWNvZGVNYXRyaXhQYWNrZWRQcm9ncmFtfSBmcm9tICcuL2RlY29kZV9tYXRyaXhfcGFja2VkX2dwdSc7XG5pbXBvcnQge0VuY29kZUZsb2F0UHJvZ3JhbX0gZnJvbSAnLi9lbmNvZGVfZmxvYXRfZ3B1JztcbmltcG9ydCB7RW5jb2RlRmxvYXRQYWNrZWRQcm9ncmFtfSBmcm9tICcuL2VuY29kZV9mbG9hdF9wYWNrZWRfZ3B1JztcbmltcG9ydCB7RW5jb2RlTWF0cml4UHJvZ3JhbX0gZnJvbSAnLi9lbmNvZGVfbWF0cml4X2dwdSc7XG5pbXBvcnQge0VuY29kZU1hdHJpeFBhY2tlZFByb2dyYW19IGZyb20gJy4vZW5jb2RlX21hdHJpeF9wYWNrZWRfZ3B1JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgZ3BncHVfbWF0aCBmcm9tICcuL2dwZ3B1X21hdGgnO1xuaW1wb3J0IHtnZXRVbmlmb3JtTG9jYXRpb25zLCBHUEdQVUJpbmFyeSwgR1BHUFVQcm9ncmFtLCBUZW5zb3JEYXRhfSBmcm9tICcuL2dwZ3B1X21hdGgnO1xuaW1wb3J0IHtzaW1wbGVBYnNJbXBsQ1BVfSBmcm9tICcuL2tlcm5lbF91dGlscy9zaGFyZWQnO1xuaW1wb3J0IHtQYWNrUHJvZ3JhbX0gZnJvbSAnLi9wYWNrX2dwdSc7XG5pbXBvcnQge1Jlc2hhcGVQYWNrZWRQcm9ncmFtfSBmcm9tICcuL3Jlc2hhcGVfcGFja2VkX2dwdSc7XG5pbXBvcnQgKiBhcyB0ZXhfdXRpbCBmcm9tICcuL3RleF91dGlsJztcbmltcG9ydCB7VGV4dHVyZSwgVGV4dHVyZURhdGEsIFRleHR1cmVVc2FnZX0gZnJvbSAnLi90ZXhfdXRpbCc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuL3RleHR1cmVfbWFuYWdlcic7XG5pbXBvcnQgKiBhcyB1bmFyeV9vcCBmcm9tICcuL3VuYXJ5b3BfZ3B1JztcbmltcG9ydCB7VW5hcnlPcFByb2dyYW19IGZyb20gJy4vdW5hcnlvcF9ncHUnO1xuaW1wb3J0IHtVbmFyeU9wUGFja2VkUHJvZ3JhbX0gZnJvbSAnLi91bmFyeW9wX3BhY2tlZF9ncHUnO1xuaW1wb3J0IHtVbnBhY2tQcm9ncmFtfSBmcm9tICcuL3VucGFja19ncHUnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5jb25zdCB3aGVyZUltcGwgPSBrZXJuZWxfaW1wbHMud2hlcmVJbXBsO1xuXG5leHBvcnQgY29uc3QgRVBTSUxPTl9GTE9BVDMyID0gMWUtNztcbmV4cG9ydCBjb25zdCBFUFNJTE9OX0ZMT0FUMTYgPSAxZS00O1xuXG50eXBlIEtlcm5lbEluZm8gPSB7XG4gIG5hbWU6IHN0cmluZzsgcXVlcnk6IFByb21pc2U8bnVtYmVyPjtcbn07XG5cbmV4cG9ydCB0eXBlIFRpbWVyTm9kZSA9IFJlY3Vyc2l2ZUFycmF5PEtlcm5lbEluZm8+fEtlcm5lbEluZm87XG5leHBvcnQgaW50ZXJmYWNlIENQVVRpbWVyUXVlcnkge1xuICBzdGFydE1zOiBudW1iZXI7XG4gIGVuZE1zPzogbnVtYmVyO1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIFdlYkdMTWVtb3J5SW5mbyBleHRlbmRzIE1lbW9yeUluZm8ge1xuICBudW1CeXRlc0luR1BVOiBudW1iZXI7XG4gIC8vIFRyYWNrcyB0aGUgdG90YWwgbnVtYmVyIG9mIGJ5dGVzIGFsbG9jYXRlZCBvbiB0aGUgR1BVLCBhY2NvdW50aW5nIGZvciB0aGVcbiAgLy8gcGh5c2ljYWwgdGV4dHVyZSB0eXBlLlxuICBudW1CeXRlc0luR1BVQWxsb2NhdGVkOiBudW1iZXI7XG4gIC8vIFRyYWNrcyBieXRlIHNpemUgb2YgdGV4dHVyZXMgdGhhdCB3ZXJlIGNyZWF0ZWQgYW5kIHRoZW4gbWFkZSBhdmFpbGFibGUgZm9yXG4gIC8vIHJldXNlIChkaXNwb3NlZCkuXG4gIG51bUJ5dGVzSW5HUFVGcmVlOiBudW1iZXI7XG4gIHVucmVsaWFibGU6IGJvb2xlYW47XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgV2ViR0xUaW1pbmdJbmZvIGV4dGVuZHMgVGltaW5nSW5mbyB7XG4gIHVwbG9hZFdhaXRNczogbnVtYmVyO1xuICBkb3dubG9hZFdhaXRNczogbnVtYmVyO1xufVxuXG5jb25zdCBiaW5hcnlDYWNoZXM6IHtbd2ViR0xWZXJzaW9uOiBzdHJpbmddOiB7W2tleTogc3RyaW5nXTogR1BHUFVCaW5hcnl9fSA9IHt9O1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0QmluYXJ5Q2FjaGUod2ViR0xWZXJzaW9uOiBudW1iZXIpIHtcbiAgaWYgKHdlYkdMVmVyc2lvbiBpbiBiaW5hcnlDYWNoZXMpIHtcbiAgICByZXR1cm4gYmluYXJ5Q2FjaGVzW3dlYkdMVmVyc2lvbl07XG4gIH1cbiAgYmluYXJ5Q2FjaGVzW3dlYkdMVmVyc2lvbl0gPSB7fTtcbiAgcmV0dXJuIGJpbmFyeUNhY2hlc1t3ZWJHTFZlcnNpb25dO1xufVxuXG4vLyBFbXBpcmljYWxseSBkZXRlcm1pbmVkIGNvbnN0YW50IHVzZWQgdG8gZGV0ZXJtaW5lIHNpemUgdGhyZXNob2xkIGZvciBoYW5kaW5nXG4vLyBvZmYgZXhlY3V0aW9uIHRvIHRoZSBDUFUuXG5jb25zdCBDUFVfSEFORE9GRl9TSVpFX1RIUkVTSE9MRCA9XG4gICAgZW52KCkuZ2V0TnVtYmVyKCdDUFVfSEFORE9GRl9TSVpFX1RIUkVTSE9MRCcpO1xuXG4vLyBFbXBpcmljYWxseSBkZXRlcm1pbmVkIGNvbnN0YW50IHVzZWQgdG8gZGVjaWRlIHRoZSBudW1iZXIgb2YgTUIgb24gR1BVXG4vLyBiZWZvcmUgd2Ugd2FybiBhYm91dCBoaWdoIG1lbW9yeSB1c2UuIFRoZSBNQiBhcmUgdGhpcyBjb25zdGFudCAqIHNjcmVlbiBhcmVhXG4vLyAqIGRwaSAvIDEwMjQgLyAxMDI0LlxuY29uc3QgQkVGT1JFX1BBR0lOR19DT05TVEFOVCA9IDYwMDtcbmZ1bmN0aW9uIG51bU1CQmVmb3JlV2FybmluZygpOiBudW1iZXIge1xuICBpZiAoZW52KCkuZ2xvYmFsLnNjcmVlbiA9PSBudWxsKSB7XG4gICAgcmV0dXJuIDEwMjQ7ICAvLyAxIEdCLlxuICB9XG4gIHJldHVybiAoZW52KCkuZ2xvYmFsLnNjcmVlbi5oZWlnaHQgKiBlbnYoKS5nbG9iYWwuc2NyZWVuLndpZHRoICpcbiAgICAgICAgICB3aW5kb3cuZGV2aWNlUGl4ZWxSYXRpbykgKlxuICAgICAgQkVGT1JFX1BBR0lOR19DT05TVEFOVCAvIDEwMjQgLyAxMDI0O1xufVxuXG5leHBvcnQgY2xhc3MgTWF0aEJhY2tlbmRXZWJHTCBleHRlbmRzIEtlcm5lbEJhY2tlbmQge1xuICB0ZXhEYXRhOiBEYXRhU3RvcmFnZTxUZXh0dXJlRGF0YT47XG4gIGdwZ3B1OiBHUEdQVUNvbnRleHQ7XG5cbiAgcHJpdmF0ZSBzdGF0aWMgbmV4dERhdGFJZCA9IDA7XG4gIHByaXZhdGUgbmV4dERhdGFJZCgpOiBudW1iZXIge1xuICAgIHJldHVybiBNYXRoQmFja2VuZFdlYkdMLm5leHREYXRhSWQrKztcbiAgfVxuICAvLyBNYXBzIGRhdGEgaWRzIHRoYXQgaGF2ZSBhIHBlbmRpbmcgcmVhZCBvcGVyYXRpb24sIHRvIGxpc3Qgb2Ygc3Vic2NyaWJlcnMuXG4gIHByaXZhdGUgcGVuZGluZ1JlYWQgPSBuZXcgV2Vha01hcDxEYXRhSWQsIEFycmF5PChhcnI6IFR5cGVkQXJyYXkpID0+IHZvaWQ+PigpO1xuICAvLyBMaXN0IG9mIGRhdGEgaWRzIHRoYXQgYXJlIHNjaGVkdWxlZCBmb3IgZGlzcG9zYWwsIGJ1dCBhcmUgd2FpdGluZyBvbiBhXG4gIC8vIHBlbmRpbmcgcmVhZCBvcGVyYXRpb24uXG4gIHByaXZhdGUgcGVuZGluZ0Rpc3Bvc2FsID0gbmV3IFdlYWtTZXQ8RGF0YUlkPigpO1xuXG4gIC8vIFVzZWQgdG8gY291bnQgdGhlIG51bWJlciBvZiAnc2hhbGxvdycgc2xpY2VkIHRlbnNvcnMgdGhhdCBwb2ludCB0byB0aGVcbiAgLy8gc2FtZSBkYXRhIGlkLlxuICBkYXRhUmVmQ291bnQgPSBuZXcgV2Vha01hcDxEYXRhSWQsIG51bWJlcj4oKTtcbiAgcHJpdmF0ZSBudW1CeXRlc0luR1BVID0gMDtcblxuICBwcml2YXRlIGNhbnZhczogSFRNTENhbnZhc0VsZW1lbnR8T2Zmc2NyZWVuQ2FudmFzO1xuXG4gIHByaXZhdGUgcHJvZ3JhbVRpbWVyc1N0YWNrOiBUaW1lck5vZGVbXTtcbiAgcHJpdmF0ZSBhY3RpdmVUaW1lcnM6IFRpbWVyTm9kZVtdO1xuICAvLyBBY2N1bXVsYXRlZCB0aW1lIHNwZW50IChpbmNsdWRpbmcgYmxvY2tpbmcpIGluIHVwbG9hZGluZyBkYXRhIHRvIHdlYmdsLlxuICBwcml2YXRlIHVwbG9hZFdhaXRNcyA9IDA7XG4gIC8vIEFjY3VtdWxhdGVkIHRpbWUgc3BlbnQgKGluY2x1ZGluZyBibG9ja2luZyBpbiBkb3dubG9hZGluZyBkYXRhIGZyb20gd2ViZ2wuXG4gIHByaXZhdGUgZG93bmxvYWRXYWl0TXMgPSAwO1xuXG4gIC8vIHJlY29yZCB0aGUgbGFzdCBtYW51YWwgR0wgRmx1c2ggdGltZS5cbiAgcHJpdmF0ZSBsYXN0R2xGbHVzaFRpbWUgPSAwO1xuXG4gIC8vIE51bWJlciBvZiBiaXRzIG9mIHByZWNpc2lvbiBvZiB0aGlzIGJhY2tlbmQuXG4gIHByaXZhdGUgZmxvYXRQcmVjaXNpb25WYWx1ZTogMzJ8MTY7XG5cbiAgcHJpdmF0ZSB0ZXh0dXJlTWFuYWdlcjogVGV4dHVyZU1hbmFnZXI7XG4gIHByaXZhdGUgYmluYXJ5Q2FjaGU6IHtba2V5OiBzdHJpbmddOiBHUEdQVUJpbmFyeX07XG4gIHByaXZhdGUgZ3BncHVDcmVhdGVkTG9jYWxseTogYm9vbGVhbjtcbiAgcHJpdmF0ZSBudW1NQkJlZm9yZVdhcm5pbmc6IG51bWJlcjtcbiAgcHJpdmF0ZSB3YXJuZWRBYm91dE1lbW9yeSA9IGZhbHNlO1xuXG4gIGNvbnN0cnVjdG9yKGdwdVJlc291cmNlPzogR1BHUFVDb250ZXh0fEhUTUxDYW52YXNFbGVtZW50fE9mZnNjcmVlbkNhbnZhcykge1xuICAgIHN1cGVyKCk7XG4gICAgaWYgKCFlbnYoKS5nZXRCb29sKCdIQVNfV0VCR0wnKSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdXZWJHTCBpcyBub3Qgc3VwcG9ydGVkIG9uIHRoaXMgZGV2aWNlJyk7XG4gICAgfVxuXG4gICAgbGV0IG5ld0dQR1BVO1xuICAgIGlmIChncHVSZXNvdXJjZSAhPSBudWxsKSB7XG4gICAgICBpZiAoZ3B1UmVzb3VyY2UgaW5zdGFuY2VvZiBHUEdQVUNvbnRleHQpIHtcbiAgICAgICAgbmV3R1BHUFUgPSBncHVSZXNvdXJjZTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIGNvbnN0IGdsID1cbiAgICAgICAgICAgIGdldFdlYkdMQ29udGV4dChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1ZFUlNJT04nKSwgZ3B1UmVzb3VyY2UpO1xuICAgICAgICBuZXdHUEdQVSA9IG5ldyBHUEdQVUNvbnRleHQoZ2wpO1xuICAgICAgfVxuICAgICAgdGhpcy5iaW5hcnlDYWNoZSA9IHt9O1xuICAgICAgdGhpcy5ncGdwdUNyZWF0ZWRMb2NhbGx5ID0gZmFsc2U7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGdsID0gZ2V0V2ViR0xDb250ZXh0KGVudigpLmdldE51bWJlcignV0VCR0xfVkVSU0lPTicpKTtcbiAgICAgIG5ld0dQR1BVID0gbmV3IEdQR1BVQ29udGV4dChnbCk7XG4gICAgICB0aGlzLmJpbmFyeUNhY2hlID0gZ2V0QmluYXJ5Q2FjaGUoZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9WRVJTSU9OJykpO1xuICAgICAgdGhpcy5ncGdwdUNyZWF0ZWRMb2NhbGx5ID0gdHJ1ZTtcbiAgICB9XG5cbiAgICB0aGlzLmdwZ3B1ID0gbmV3R1BHUFU7XG4gICAgdGhpcy5jYW52YXMgPSB0aGlzLmdwZ3B1LmdsLmNhbnZhcztcbiAgICB0aGlzLnRleHR1cmVNYW5hZ2VyID0gbmV3IFRleHR1cmVNYW5hZ2VyKHRoaXMuZ3BncHUpO1xuICAgIHRoaXMubnVtTUJCZWZvcmVXYXJuaW5nID0gbnVtTUJCZWZvcmVXYXJuaW5nKCk7XG4gICAgdGhpcy50ZXhEYXRhID0gbmV3IERhdGFTdG9yYWdlKHRoaXMsIGVuZ2luZSgpKTtcbiAgfVxuXG4gIG51bURhdGFJZHMoKSB7XG4gICAgcmV0dXJuIHRoaXMudGV4RGF0YS5udW1EYXRhSWRzKCkgLSB0aGlzLnBlbmRpbmdEZWxldGVzO1xuICB9XG5cbiAgd3JpdGUodmFsdWVzOiBCYWNrZW5kVmFsdWVzLCBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSk6IERhdGFJZCB7XG4gICAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdMX0NIRUNLX05VTUVSSUNBTF9QUk9CTEVNUycpIHx8XG4gICAgICAgIGVudigpLmdldEJvb2woJ0RFQlVHJykpIHtcbiAgICAgIHRoaXMuY2hlY2tOdW1lcmljYWxQcm9ibGVtcyh2YWx1ZXMpO1xuICAgIH1cbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnICYmIHZhbHVlcyAhPSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYENhbm5vdCB3cml0ZSB0byBhIGNvbXBsZXg2NCBkdHlwZS4gYCArXG4gICAgICAgICAgYFBsZWFzZSB1c2UgdGYuY29tcGxleChyZWFsLCBpbWFnKS5gKTtcbiAgICB9XG4gICAgY29uc3QgZGF0YUlkID0ge2lkOiB0aGlzLm5leHREYXRhSWQoKX07XG4gICAgdGhpcy50ZXhEYXRhLnNldChcbiAgICAgICAgZGF0YUlkLFxuICAgICAgICB7c2hhcGUsIGR0eXBlLCB2YWx1ZXMsIHVzYWdlOiBUZXh0dXJlVXNhZ2UuVVBMT0FELCByZWZDb3VudDogMX0pO1xuICAgIHJldHVybiBkYXRhSWQ7XG4gIH1cblxuICAvKiogUmV0dXJuIHJlZkNvdW50IG9mIGEgYFRlbnNvckRhdGFgLiAqL1xuICByZWZDb3VudChkYXRhSWQ6IERhdGFJZCk6IG51bWJlciB7XG4gICAgaWYgKHRoaXMudGV4RGF0YS5oYXMoZGF0YUlkKSkge1xuICAgICAgY29uc3QgdGVuc29yRGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICAgIHJldHVybiB0ZW5zb3JEYXRhLnJlZkNvdW50O1xuICAgIH1cbiAgICByZXR1cm4gMDtcbiAgfVxuXG4gIC8qKiBJbmNyZWFzZSByZWZDb3VudCBvZiBhIGBUZXh0dXJlRGF0YWAuICovXG4gIGluY1JlZihkYXRhSWQ6IERhdGFJZCk6IHZvaWQge1xuICAgIGNvbnN0IHRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gICAgdGV4RGF0YS5yZWZDb3VudCsrO1xuICB9XG5cbiAgLyoqIERlY3JlYXNlIHJlZkNvdW50IG9mIGEgYFRleHR1cmVEYXRhYC4gKi9cbiAgZGVjUmVmKGRhdGFJZDogRGF0YUlkKTogdm9pZCB7XG4gICAgaWYgKHRoaXMudGV4RGF0YS5oYXMoZGF0YUlkKSkge1xuICAgICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICAgIHRleERhdGEucmVmQ291bnQtLTtcbiAgICB9XG4gIH1cblxuICBtb3ZlKFxuICAgICAgZGF0YUlkOiBEYXRhSWQsIHZhbHVlczogQmFja2VuZFZhbHVlcywgc2hhcGU6IG51bWJlcltdLCBkdHlwZTogRGF0YVR5cGUsXG4gICAgICByZWZDb3VudDogbnVtYmVyKTogdm9pZCB7XG4gICAgaWYgKGVudigpLmdldEJvb2woJ0RFQlVHJykpIHtcbiAgICAgIHRoaXMuY2hlY2tOdW1lcmljYWxQcm9ibGVtcyh2YWx1ZXMpO1xuICAgIH1cbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYENhbm5vdCB3cml0ZSB0byBhIGNvbXBsZXg2NCBkdHlwZS4gYCArXG4gICAgICAgICAgYFBsZWFzZSB1c2UgdGYuY29tcGxleChyZWFsLCBpbWFnKS5gKTtcbiAgICB9XG4gICAgdGhpcy50ZXhEYXRhLnNldChcbiAgICAgICAgZGF0YUlkLCB7c2hhcGUsIGR0eXBlLCB2YWx1ZXMsIHVzYWdlOiBUZXh0dXJlVXNhZ2UuVVBMT0FELCByZWZDb3VudH0pO1xuICB9XG5cbiAgZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8odGVuc29ySW5mbzogVGVuc29ySW5mbyk6IHZvaWQge1xuICAgIHRoaXMuZGlzcG9zZURhdGEodGVuc29ySW5mby5kYXRhSWQpO1xuICB9XG5cbiAgcmVhZFN5bmMoZGF0YUlkOiBEYXRhSWQpOiBCYWNrZW5kVmFsdWVzIHtcbiAgICBjb25zdCB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHt2YWx1ZXMsIGR0eXBlLCBjb21wbGV4VGVuc29ySW5mb3MsIHNsaWNlLCBzaGFwZSwgaXNQYWNrZWR9ID0gdGV4RGF0YTtcblxuICAgIC8vIFRoZSBwcmVzZW5jZSBvZiBgc2xpY2VgIGluZGljYXRlcyB0aGlzIHRlbnNvciBpcyBhIHNoYWxsb3cgc2xpY2Ugb2YgYVxuICAgIC8vIGRpZmZlcmVudCB0ZW5zb3IsIGFuZCBpcyB1c2luZyB0aGF0IG9yaWdpbmFsIHRlbnNvcidzIHRleHR1cmUuIFJ1blxuICAgIC8vIGBjbG9uZWAgaW4gb3JkZXIgdG8gY29weSB0aGF0IHRleHR1cmUgYW5kIHJlYWQgZnJvbSBpdC5cbiAgICBpZiAoc2xpY2UgIT0gbnVsbCkge1xuICAgICAgbGV0IHByb2dyYW07XG4gICAgICBpZiAoaXNQYWNrZWQpIHtcbiAgICAgICAgcHJvZ3JhbSA9IG5ldyBVbmFyeU9wUGFja2VkUHJvZ3JhbShzaGFwZSwgdW5hcnlfb3AuQ0xPTkUpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcHJvZ3JhbSA9IG5ldyBVbmFyeU9wUHJvZ3JhbShzaGFwZSwgdW5hcnlfb3AuQ0xPTkUpO1xuICAgICAgfVxuICAgICAgY29uc3QgcmVzID1cbiAgICAgICAgICB0aGlzLnJ1bldlYkdMUHJvZ3JhbShwcm9ncmFtLCBbe2RhdGFJZCwgc2hhcGUsIGR0eXBlfV0sIGR0eXBlKTtcbiAgICAgIGNvbnN0IGRhdGEgPSB0aGlzLnJlYWRTeW5jKHJlcy5kYXRhSWQpO1xuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhyZXMpO1xuICAgICAgcmV0dXJuIGRhdGE7XG4gICAgfVxuICAgIGlmICh2YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgcmV0dXJuIHRoaXMuY29udmVydEFuZENhY2hlT25DUFUoZGF0YUlkKTtcbiAgICB9XG4gICAgaWYgKGR0eXBlID09PSAnc3RyaW5nJykge1xuICAgICAgcmV0dXJuIHZhbHVlcztcbiAgICB9XG4gICAgY29uc3Qgc2hvdWxkVGltZVByb2dyYW0gPSB0aGlzLmFjdGl2ZVRpbWVycyAhPSBudWxsO1xuICAgIGxldCBzdGFydDogbnVtYmVyO1xuICAgIGlmIChzaG91bGRUaW1lUHJvZ3JhbSkge1xuICAgICAgc3RhcnQgPSB1dGlsLm5vdygpO1xuICAgIH1cblxuICAgIGxldCByZXN1bHQ6IEZsb2F0MzJBcnJheTtcbiAgICBpZiAoZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICBjb25zdCByZWFsVmFsdWVzID1cbiAgICAgICAgICB0aGlzLnJlYWRTeW5jKGNvbXBsZXhUZW5zb3JJbmZvcy5yZWFsLmRhdGFJZCkgYXMgRmxvYXQzMkFycmF5O1xuICAgICAgY29uc3QgaW1hZ1ZhbHVlcyA9XG4gICAgICAgICAgdGhpcy5yZWFkU3luYyhjb21wbGV4VGVuc29ySW5mb3MuaW1hZy5kYXRhSWQpIGFzIEZsb2F0MzJBcnJheTtcbiAgICAgIHJlc3VsdCA9IGJhY2tlbmRfdXRpbC5tZXJnZVJlYWxBbmRJbWFnQXJyYXlzKHJlYWxWYWx1ZXMsIGltYWdWYWx1ZXMpO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXN1bHQgPSB0aGlzLmdldFZhbHVlc0Zyb21UZXh0dXJlKGRhdGFJZCk7XG4gICAgfVxuXG4gICAgaWYgKHNob3VsZFRpbWVQcm9ncmFtKSB7XG4gICAgICB0aGlzLmRvd25sb2FkV2FpdE1zICs9IHV0aWwubm93KCkgLSBzdGFydDtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuY29udmVydEFuZENhY2hlT25DUFUoZGF0YUlkLCByZXN1bHQpO1xuICB9XG5cbiAgYXN5bmMgcmVhZChkYXRhSWQ6IERhdGFJZCk6IFByb21pc2U8QmFja2VuZFZhbHVlcz4ge1xuICAgIGlmICh0aGlzLnBlbmRpbmdSZWFkLmhhcyhkYXRhSWQpKSB7XG4gICAgICBjb25zdCBzdWJzY3JpYmVycyA9IHRoaXMucGVuZGluZ1JlYWQuZ2V0KGRhdGFJZCk7XG4gICAgICByZXR1cm4gbmV3IFByb21pc2U8VHlwZWRBcnJheT4ocmVzb2x2ZSA9PiBzdWJzY3JpYmVycy5wdXNoKHJlc29sdmUpKTtcbiAgICB9XG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBjb25zdCB7dmFsdWVzLCBzaGFwZSwgc2xpY2UsIGR0eXBlLCBjb21wbGV4VGVuc29ySW5mb3MsIGlzUGFja2VkfSA9IHRleERhdGE7XG5cbiAgICAvLyBUaGUgcHJlc2VuY2Ugb2YgYHNsaWNlYCBpbmRpY2F0ZXMgdGhpcyB0ZW5zb3IgaXMgYSBzaGFsbG93IHNsaWNlIG9mIGFcbiAgICAvLyBkaWZmZXJlbnQgdGVuc29yLCBhbmQgaXMgdXNpbmcgdGhhdCBvcmlnaW5hbCB0ZW5zb3IncyB0ZXh0dXJlLiBSdW5cbiAgICAvLyBgY2xvbmVgIGluIG9yZGVyIHRvIGNvcHkgdGhhdCB0ZXh0dXJlIGFuZCByZWFkIGZyb20gaXQuXG4gICAgaWYgKHNsaWNlICE9IG51bGwpIHtcbiAgICAgIGxldCBwcm9ncmFtO1xuICAgICAgaWYgKGlzUGFja2VkKSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgVW5hcnlPcFBhY2tlZFByb2dyYW0oc2hhcGUsIHVuYXJ5X29wLkNMT05FKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgVW5hcnlPcFByb2dyYW0oc2hhcGUsIHVuYXJ5X29wLkNMT05FKTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHJlcyA9XG4gICAgICAgICAgdGhpcy5ydW5XZWJHTFByb2dyYW0ocHJvZ3JhbSwgW3tkYXRhSWQsIHNoYXBlLCBkdHlwZX1dLCBkdHlwZSk7XG4gICAgICBjb25zdCBkYXRhID0gdGhpcy5yZWFkKHJlcy5kYXRhSWQpO1xuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhyZXMpO1xuICAgICAgcmV0dXJuIGRhdGE7XG4gICAgfVxuXG4gICAgaWYgKHZhbHVlcyAhPSBudWxsKSB7XG4gICAgICByZXR1cm4gdGhpcy5jb252ZXJ0QW5kQ2FjaGVPbkNQVShkYXRhSWQpO1xuICAgIH1cblxuICAgIGlmIChlbnYoKS5nZXRCb29sKCdERUJVRycpKSB7XG4gICAgICAvLyBnZXRCb29sKCdXRUJHTF9ET1dOTE9BRF9GTE9BVF9FTkFCTEVEJykgY2F1c2VkIGEgYmxvY2tpbmcgR1BVIGNhbGwuXG4gICAgICAvLyBGb3IgcGVyZm9ybWFuY2UgcmVhc29uLCBvbmx5IGNoZWNrIGl0IGZvciBkZWJ1Z2dpbmcuIEluIHByb2R1Y3Rpb24sXG4gICAgICAvLyBpdCBkb2Vzbid0IGhhbmRsZSB0aGlzIHVzZSBjYXNlIGFueXdheSwgc28gYmVoYXZpb3IgaXMgbm90IGNoYW5nZWQuXG4gICAgICBpZiAoIWVudigpLmdldEJvb2woJ1dFQkdMX0RPV05MT0FEX0ZMT0FUX0VOQUJMRUQnKSAmJlxuICAgICAgICAgIGVudigpLmdldE51bWJlcignV0VCR0xfVkVSU0lPTicpID09PSAyKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgIGB0ZW5zb3IuZGF0YSgpIHdpdGggV0VCR0xfRE9XTkxPQURfRkxPQVRfRU5BQkxFRD1mYWxzZSBhbmQgYCArXG4gICAgICAgICAgICBgV0VCR0xfVkVSU0lPTj0yIG5vdCB5ZXQgc3VwcG9ydGVkLmApO1xuICAgICAgfVxuICAgIH1cblxuICAgIGxldCBidWZmZXI6IFdlYkdMQnVmZmVyID0gbnVsbDtcbiAgICBsZXQgdG1wRG93bmxvYWRUYXJnZXQ6IFRlbnNvckluZm87XG5cbiAgICBpZiAoZHR5cGUgIT09ICdjb21wbGV4NjQnICYmIGVudigpLmdldCgnV0VCR0xfQlVGRkVSX1NVUFBPUlRFRCcpKSB7XG4gICAgICAvLyBQb3NzaWJseSBjb3B5IHRoZSB0ZXh0dXJlIGludG8gYSBidWZmZXIgYmVmb3JlIGluc2VydGluZyBhIGZlbmNlLlxuICAgICAgdG1wRG93bmxvYWRUYXJnZXQgPSB0aGlzLmRlY29kZShkYXRhSWQpO1xuICAgICAgY29uc3QgdG1wRGF0YSA9IHRoaXMudGV4RGF0YS5nZXQodG1wRG93bmxvYWRUYXJnZXQuZGF0YUlkKTtcblxuICAgICAgYnVmZmVyID0gdGhpcy5ncGdwdS5jcmVhdGVCdWZmZXJGcm9tVGV4dHVyZShcbiAgICAgICAgICB0bXBEYXRhLnRleHR1cmUudGV4dHVyZSwgLi4udGV4X3V0aWwuZ2V0RGVuc2VUZXhTaGFwZShzaGFwZSkpO1xuICAgIH1cblxuICAgIHRoaXMucGVuZGluZ1JlYWQuc2V0KGRhdGFJZCwgW10pO1xuXG4gICAgaWYgKGR0eXBlICE9PSAnY29tcGxleDY0Jykge1xuICAgICAgLy8gQ3JlYXRlIGEgZmVuY2UgYW5kIHdhaXQgZm9yIGl0IHRvIHJlc29sdmUuXG4gICAgICBhd2FpdCB0aGlzLmdwZ3B1LmNyZWF0ZUFuZFdhaXRGb3JGZW5jZSgpO1xuICAgIH1cblxuICAgIC8vIERvd25sb2FkIHRoZSB2YWx1ZXMgZnJvbSB0aGUgR1BVLlxuICAgIGxldCB2YWxzOiBGbG9hdDMyQXJyYXk7XG4gICAgaWYgKGR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICAgY29uc3QgcHMgPSBhd2FpdCBQcm9taXNlLmFsbChbXG4gICAgICAgIHRoaXMucmVhZChjb21wbGV4VGVuc29ySW5mb3MucmVhbC5kYXRhSWQpLFxuICAgICAgICB0aGlzLnJlYWQoY29tcGxleFRlbnNvckluZm9zLmltYWcuZGF0YUlkKVxuICAgICAgXSk7XG5cbiAgICAgIGNvbnN0IHJlYWxWYWx1ZXMgPSBwc1swXTtcbiAgICAgIGNvbnN0IGltYWdWYWx1ZXMgPSBwc1sxXTtcbiAgICAgIHZhbHMgPSBiYWNrZW5kX3V0aWwubWVyZ2VSZWFsQW5kSW1hZ0FycmF5cyhcbiAgICAgICAgICByZWFsVmFsdWVzIGFzIEZsb2F0MzJBcnJheSwgaW1hZ1ZhbHVlcyBhcyBGbG9hdDMyQXJyYXkpO1xuICAgIH0gZWxzZSBpZiAoYnVmZmVyID09IG51bGwpIHtcbiAgICAgIHZhbHMgPSB0aGlzLmdldFZhbHVlc0Zyb21UZXh0dXJlKGRhdGFJZCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgICAgdmFscyA9IHRoaXMuZ3BncHUuZG93bmxvYWRGbG9hdDMyTWF0cml4RnJvbUJ1ZmZlcihidWZmZXIsIHNpemUpO1xuICAgIH1cbiAgICBpZiAodG1wRG93bmxvYWRUYXJnZXQgIT0gbnVsbCkge1xuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyh0bXBEb3dubG9hZFRhcmdldCk7XG4gICAgfVxuICAgIGlmIChidWZmZXIgIT0gbnVsbCkge1xuICAgICAgY29uc3QgZ2wgPSB0aGlzLmdwZ3B1LmdsO1xuICAgICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRlbGV0ZUJ1ZmZlcihidWZmZXIpKTtcbiAgICB9XG4gICAgY29uc3QgZFR5cGVWYWxzID0gdGhpcy5jb252ZXJ0QW5kQ2FjaGVPbkNQVShkYXRhSWQsIHZhbHMpO1xuXG4gICAgY29uc3Qgc3Vic2NyaWJlcnMgPSB0aGlzLnBlbmRpbmdSZWFkLmdldChkYXRhSWQpO1xuICAgIHRoaXMucGVuZGluZ1JlYWQuZGVsZXRlKGRhdGFJZCk7XG5cbiAgICAvLyBOb3RpZnkgYWxsIHBlbmRpbmcgcmVhZHMuXG4gICAgc3Vic2NyaWJlcnMuZm9yRWFjaChyZXNvbHZlID0+IHJlc29sdmUoZFR5cGVWYWxzKSk7XG4gICAgaWYgKHRoaXMucGVuZGluZ0Rpc3Bvc2FsLmhhcyhkYXRhSWQpKSB7XG4gICAgICB0aGlzLnBlbmRpbmdEaXNwb3NhbC5kZWxldGUoZGF0YUlkKTtcbiAgICAgIGlmICh0aGlzLmRpc3Bvc2VEYXRhKGRhdGFJZCkpIHtcbiAgICAgICAgZW5naW5lKCkucmVtb3ZlRGF0YUlkKGRhdGFJZCwgdGhpcyk7XG4gICAgICB9XG4gICAgICB0aGlzLnBlbmRpbmdEZWxldGVzLS07XG4gICAgfVxuICAgIHJldHVybiBkVHlwZVZhbHM7XG4gIH1cblxuICAvKipcbiAgICogUmVhZCB0ZW5zb3IgdG8gYSBuZXcgdGV4dHVyZSB0aGF0IGlzIGRlbnNlbHkgcGFja2VkIGZvciBlYXNlIG9mIHVzZS5cbiAgICogQHBhcmFtIGRhdGFJZCBUaGUgc291cmNlIHRlbnNvci5cbiAgICogQHBhcmFtIG9wdGlvbnNcbiAgICogICAgIGN1c3RvbVRleFNoYXBlOiBPcHRpb25hbC4gSWYgc2V0LCB3aWxsIHVzZSB0aGUgdXNlciBkZWZpbmVkIHRleHR1cmVcbiAgICogICAgIHNoYXBlIHRvIGNyZWF0ZSB0aGUgdGV4dHVyZS5cbiAgICovXG4gIHJlYWRUb0dQVShkYXRhSWQ6IERhdGFJZCwgb3B0aW9uczogRGF0YVRvR1BVV2ViR0xPcHRpb24gPSB7fSk6IEdQVURhdGEge1xuICAgIGNvbnN0IHRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gICAgY29uc3Qge3ZhbHVlcywgc2hhcGUsIHNsaWNlLCBkdHlwZSwgaXNQYWNrZWQsIHRleHR1cmV9ID0gdGV4RGF0YTtcblxuICAgIGlmIChkdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignRG9lcyBub3Qgc3VwcG9ydCByZWFkaW5nIHRleHR1cmUgZm9yIGNvbXBsZXg2NCBkdHlwZS4nKTtcbiAgICB9XG5cbiAgICAvLyBUaGUgcHJlc2VuY2Ugb2YgYHNsaWNlYCBpbmRpY2F0ZXMgdGhpcyB0ZW5zb3IgaXMgYSBzaGFsbG93IHNsaWNlIG9mIGFcbiAgICAvLyBkaWZmZXJlbnQgdGVuc29yLCBhbmQgaXMgdXNpbmcgdGhhdCBvcmlnaW5hbCB0ZW5zb3IncyB0ZXh0dXJlLiBSdW5cbiAgICAvLyBgY2xvbmVgIGluIG9yZGVyIHRvIGNvcHkgdGhhdCB0ZXh0dXJlIGFuZCByZWFkIGZyb20gaXQuXG4gICAgaWYgKHNsaWNlICE9IG51bGwpIHtcbiAgICAgIGxldCBwcm9ncmFtO1xuICAgICAgaWYgKGlzUGFja2VkKSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgVW5hcnlPcFBhY2tlZFByb2dyYW0oc2hhcGUsIHVuYXJ5X29wLkNMT05FKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgVW5hcnlPcFByb2dyYW0oc2hhcGUsIHVuYXJ5X29wLkNMT05FKTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHJlcyA9XG4gICAgICAgICAgdGhpcy5ydW5XZWJHTFByb2dyYW0ocHJvZ3JhbSwgW3tkYXRhSWQsIHNoYXBlLCBkdHlwZX1dLCBkdHlwZSk7XG4gICAgICBjb25zdCBncHVSZXNvdW9yY2UgPSB0aGlzLnJlYWRUb0dQVShyZXMsIG9wdGlvbnMpO1xuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhyZXMpO1xuICAgICAgcmV0dXJuIGdwdVJlc291b3JjZTtcbiAgICB9XG5cbiAgICBpZiAodGV4dHVyZSA9PSBudWxsKSB7XG4gICAgICBpZiAodmFsdWVzICE9IG51bGwpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdEYXRhIGlzIG5vdCBvbiBHUFUgYnV0IG9uIENQVS4nKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcignVGhlcmUgaXMgbm8gZGF0YSBvbiBHUFUgb3IgQ1BVLicpO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIERlY29kZSB0aGUgdGV4dHVyZSBzbyB0aGF0IGl0IGlzIHN0b3JlZCBkZW5zZWx5ICh1c2luZyBmb3VyIGNoYW5uZWxzKS5cbiAgICBjb25zdCB0bXBUYXJnZXQgPSB0aGlzLmRlY29kZShkYXRhSWQsIG9wdGlvbnMuY3VzdG9tVGV4U2hhcGUpO1xuXG4gICAgLy8gTWFrZSBlbmdpbmUgdHJhY2sgdGhpcyB0ZW5zb3IsIHNvIHRoYXQgd2UgY2FuIGRpc3Bvc2UgaXQgbGF0ZXIuXG4gICAgY29uc3QgdGVuc29yUmVmID0gZW5naW5lKCkubWFrZVRlbnNvckZyb21EYXRhSWQoXG4gICAgICAgIHRtcFRhcmdldC5kYXRhSWQsIHRtcFRhcmdldC5zaGFwZSwgdG1wVGFyZ2V0LmR0eXBlKTtcblxuICAgIGNvbnN0IHRtcERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KHRtcFRhcmdldC5kYXRhSWQpO1xuICAgIHJldHVybiB7dGVuc29yUmVmLCAuLi50bXBEYXRhLnRleHR1cmV9O1xuICB9XG5cbiAgYnVmZmVyU3luYzxSIGV4dGVuZHMgUmFuaz4odDogVGVuc29ySW5mbyk6IFRlbnNvckJ1ZmZlcjxSPiB7XG4gICAgY29uc3QgZGF0YSA9IHRoaXMucmVhZFN5bmModC5kYXRhSWQpO1xuICAgIGxldCBkZWNvZGVkRGF0YSA9IGRhdGEgYXMgRGF0YVZhbHVlcztcbiAgICBpZiAodC5kdHlwZSA9PT0gJ3N0cmluZycpIHtcbiAgICAgIHRyeSB7XG4gICAgICAgIC8vIERlY29kZSB0aGUgYnl0ZXMgaW50byBzdHJpbmcuXG4gICAgICAgIGRlY29kZWREYXRhID0gKGRhdGEgYXMgVWludDhBcnJheVtdKS5tYXAoZCA9PiB1dGlsLmRlY29kZVN0cmluZyhkKSk7XG4gICAgICB9IGNhdGNoIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gZGVjb2RlIGVuY29kZWQgc3RyaW5nIGJ5dGVzIGludG8gdXRmLTgnKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGJ1ZmZlcih0LnNoYXBlIGFzIFNoYXBlTWFwW1JdLCB0LmR0eXBlLCBkZWNvZGVkRGF0YSkgYXNcbiAgICAgICAgVGVuc29yQnVmZmVyPFI+O1xuICB9XG5cbiAgcHJpdmF0ZSBjaGVja051bWVyaWNhbFByb2JsZW1zKHZhbHVlczogQmFja2VuZFZhbHVlcyk6IHZvaWQge1xuICAgIGlmICh2YWx1ZXMgPT0gbnVsbCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgbnVtID0gdmFsdWVzW2ldIGFzIG51bWJlcjtcbiAgICAgIGlmICghd2ViZ2xfdXRpbC5jYW5CZVJlcHJlc2VudGVkKG51bSkpIHtcbiAgICAgICAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdMX1JFTkRFUl9GTE9BVDMyX0NBUEFCTEUnKSkge1xuICAgICAgICAgIHRocm93IEVycm9yKFxuICAgICAgICAgICAgICBgVGhlIHZhbHVlICR7bnVtfSBjYW5ub3QgYmUgcmVwcmVzZW50ZWQgd2l0aCB5b3VyIGAgK1xuICAgICAgICAgICAgICBgY3VycmVudCBzZXR0aW5ncy4gQ29uc2lkZXIgZW5hYmxpbmcgZmxvYXQzMiByZW5kZXJpbmc6IGAgK1xuICAgICAgICAgICAgICBgJ3RmLmVudigpLnNldCgnV0VCR0xfUkVOREVSX0ZMT0FUMzJfRU5BQkxFRCcsIHRydWUpOydgKTtcbiAgICAgICAgfVxuICAgICAgICB0aHJvdyBFcnJvcihgVGhlIHZhbHVlICR7bnVtfSBjYW5ub3QgYmUgcmVwcmVzZW50ZWQgb24gdGhpcyBkZXZpY2UuYCk7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBnZXRWYWx1ZXNGcm9tVGV4dHVyZShkYXRhSWQ6IERhdGFJZCk6IEZsb2F0MzJBcnJheSB7XG4gICAgY29uc3Qge3NoYXBlLCBkdHlwZSwgaXNQYWNrZWR9ID0gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgIGlmIChlbnYoKS5nZXRCb29sKCdXRUJHTF9ET1dOTE9BRF9GTE9BVF9FTkFCTEVEJykpIHtcbiAgICAgIGNvbnN0IHRtcFRhcmdldCA9IHRoaXMuZGVjb2RlKGRhdGFJZCk7XG4gICAgICBjb25zdCB0bXBEYXRhID0gdGhpcy50ZXhEYXRhLmdldCh0bXBUYXJnZXQuZGF0YUlkKTtcbiAgICAgIGNvbnN0IHZhbHMgPVxuICAgICAgICAgIHRoaXMuZ3BncHVcbiAgICAgICAgICAgICAgLmRvd25sb2FkTWF0cml4RnJvbVBhY2tlZFRleHR1cmUoXG4gICAgICAgICAgICAgICAgICB0bXBEYXRhLnRleHR1cmUudGV4dHVyZSwgLi4udGV4X3V0aWwuZ2V0RGVuc2VUZXhTaGFwZShzaGFwZSkpXG4gICAgICAgICAgICAgIC5zdWJhcnJheSgwLCBzaXplKTtcblxuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyh0bXBUYXJnZXQpO1xuXG4gICAgICByZXR1cm4gdmFscztcbiAgICB9XG5cbiAgICBjb25zdCBzaG91bGRVc2VQYWNrZWRQcm9ncmFtID1cbiAgICAgICAgZW52KCkuZ2V0Qm9vbCgnV0VCR0xfUEFDSycpICYmIGlzUGFja2VkID09PSB0cnVlO1xuICAgIGNvbnN0IG91dHB1dFNoYXBlID1cbiAgICAgICAgc2hvdWxkVXNlUGFja2VkUHJvZ3JhbSA/IHdlYmdsX3V0aWwuZ2V0U2hhcGVBczNEKHNoYXBlKSA6IHNoYXBlO1xuICAgIGNvbnN0IHByb2dyYW0gPSBzaG91bGRVc2VQYWNrZWRQcm9ncmFtID9cbiAgICAgICAgbmV3IEVuY29kZUZsb2F0UGFja2VkUHJvZ3JhbShvdXRwdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pIDpcbiAgICAgICAgbmV3IEVuY29kZUZsb2F0UHJvZ3JhbShvdXRwdXRTaGFwZSk7XG4gICAgY29uc3Qgb3V0cHV0ID0gdGhpcy5ydW5XZWJHTFByb2dyYW0oXG4gICAgICAgIHByb2dyYW0sIFt7c2hhcGU6IG91dHB1dFNoYXBlLCBkdHlwZSwgZGF0YUlkfV0sICdmbG9hdDMyJyk7XG4gICAgY29uc3QgdG1wRGF0YSA9IHRoaXMudGV4RGF0YS5nZXQob3V0cHV0LmRhdGFJZCk7XG4gICAgY29uc3QgdmFscyA9IHRoaXMuZ3BncHVcbiAgICAgICAgICAgICAgICAgICAgIC5kb3dubG9hZEJ5dGVFbmNvZGVkRmxvYXRNYXRyaXhGcm9tT3V0cHV0VGV4dHVyZShcbiAgICAgICAgICAgICAgICAgICAgICAgICB0bXBEYXRhLnRleHR1cmUudGV4dHVyZSwgdG1wRGF0YS50ZXhTaGFwZVswXSxcbiAgICAgICAgICAgICAgICAgICAgICAgICB0bXBEYXRhLnRleFNoYXBlWzFdKVxuICAgICAgICAgICAgICAgICAgICAgLnN1YmFycmF5KDAsIHNpemUpO1xuICAgIHRoaXMuZGlzcG9zZUludGVybWVkaWF0ZVRlbnNvckluZm8ob3V0cHV0KTtcblxuICAgIHJldHVybiB2YWxzO1xuICB9XG5cbiAgdGltZXJBdmFpbGFibGUoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIGVudigpLmdldE51bWJlcignV0VCR0xfRElTSk9JTlRfUVVFUllfVElNRVJfRVhURU5TSU9OX1JFTElBQkxFJykgPiAwO1xuICB9XG5cbiAgdGltZShmOiAoKSA9PiB2b2lkKTogUHJvbWlzZTxXZWJHTFRpbWluZ0luZm8+IHtcbiAgICBjb25zdCBvbGRBY3RpdmVUaW1lcnMgPSB0aGlzLmFjdGl2ZVRpbWVycztcbiAgICBjb25zdCBuZXdBY3RpdmVUaW1lcnM6IFRpbWVyTm9kZVtdID0gW107XG5cbiAgICBsZXQgb3V0ZXJNb3N0VGltZSA9IGZhbHNlO1xuICAgIGlmICh0aGlzLnByb2dyYW1UaW1lcnNTdGFjayA9PSBudWxsKSB7XG4gICAgICB0aGlzLnByb2dyYW1UaW1lcnNTdGFjayA9IG5ld0FjdGl2ZVRpbWVycztcbiAgICAgIG91dGVyTW9zdFRpbWUgPSB0cnVlO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmFjdGl2ZVRpbWVycy5wdXNoKG5ld0FjdGl2ZVRpbWVycyk7XG4gICAgfVxuICAgIHRoaXMuYWN0aXZlVGltZXJzID0gbmV3QWN0aXZlVGltZXJzO1xuXG4gICAgZigpO1xuXG4gICAgLy8gbmVlZGluZyB0byBzcGxpdCB0aGVzZSB1cCBiZWNhdXNlIHV0aWwuZmxhdHRlbiBvbmx5IGFjY2VwdHMgY2VydGFpbiB0eXBlc1xuICAgIGNvbnN0IGZsYXR0ZW5lZEFjdGl2ZVRpbWVyUXVlcmllcyA9XG4gICAgICAgIHV0aWwuZmxhdHRlbih0aGlzLmFjdGl2ZVRpbWVycy5tYXAoKGQ6IEtlcm5lbEluZm8pID0+IGQucXVlcnkpKVxuICAgICAgICAgICAgLmZpbHRlcihkID0+IGQgIT0gbnVsbCk7XG4gICAgY29uc3QgZmxhdHRlbmVkQWN0aXZlVGltZXJOYW1lcyA9XG4gICAgICAgIHV0aWwuZmxhdHRlbih0aGlzLmFjdGl2ZVRpbWVycy5tYXAoKGQ6IEtlcm5lbEluZm8pID0+IGQubmFtZSkpXG4gICAgICAgICAgICAuZmlsdGVyKGQgPT4gZCAhPSBudWxsKTtcblxuICAgIHRoaXMuYWN0aXZlVGltZXJzID0gb2xkQWN0aXZlVGltZXJzO1xuXG4gICAgaWYgKG91dGVyTW9zdFRpbWUpIHtcbiAgICAgIHRoaXMucHJvZ3JhbVRpbWVyc1N0YWNrID0gbnVsbDtcbiAgICB9XG5cbiAgICBjb25zdCByZXM6IFdlYkdMVGltaW5nSW5mbyA9IHtcbiAgICAgIHVwbG9hZFdhaXRNczogdGhpcy51cGxvYWRXYWl0TXMsXG4gICAgICBkb3dubG9hZFdhaXRNczogdGhpcy5kb3dubG9hZFdhaXRNcyxcbiAgICAgIGtlcm5lbE1zOiBudWxsLFxuICAgICAgd2FsbE1zOiBudWxsICAvLyB3aWxsIGJlIGZpbGxlZCBieSB0aGUgZW5naW5lXG4gICAgfTtcblxuICAgIHJldHVybiAoYXN5bmMgKCkgPT4ge1xuICAgICAgaWYgKGVudigpLmdldE51bWJlcignV0VCR0xfRElTSk9JTlRfUVVFUllfVElNRVJfRVhURU5TSU9OX1JFTElBQkxFJykgPlxuICAgICAgICAgIDApIHtcbiAgICAgICAgY29uc3Qga2VybmVsTXMgPSBhd2FpdCBQcm9taXNlLmFsbChmbGF0dGVuZWRBY3RpdmVUaW1lclF1ZXJpZXMpO1xuXG4gICAgICAgIHJlc1sna2VybmVsTXMnXSA9IHV0aWwuc3VtKGtlcm5lbE1zKTtcbiAgICAgICAgcmVzWydnZXRFeHRyYVByb2ZpbGVJbmZvJ10gPSAoKSA9PlxuICAgICAgICAgICAga2VybmVsTXNcbiAgICAgICAgICAgICAgICAubWFwKChkLCBpKSA9PiAoe25hbWU6IGZsYXR0ZW5lZEFjdGl2ZVRpbWVyTmFtZXNbaV0sIG1zOiBkfSkpXG4gICAgICAgICAgICAgICAgLm1hcChkID0+IGAke2QubmFtZX06ICR7ZC5tc31gKVxuICAgICAgICAgICAgICAgIC5qb2luKCcsICcpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgcmVzWydrZXJuZWxNcyddID0ge1xuICAgICAgICAgIGVycm9yOiAnV2ViR0wgcXVlcnkgdGltZXJzIGFyZSBub3Qgc3VwcG9ydGVkIGluIHRoaXMgZW52aXJvbm1lbnQuJ1xuICAgICAgICB9O1xuICAgICAgfVxuXG4gICAgICB0aGlzLnVwbG9hZFdhaXRNcyA9IDA7XG4gICAgICB0aGlzLmRvd25sb2FkV2FpdE1zID0gMDtcbiAgICAgIHJldHVybiByZXM7XG4gICAgfSkoKTtcbiAgfVxuICBtZW1vcnkoKTogV2ViR0xNZW1vcnlJbmZvIHtcbiAgICByZXR1cm4ge1xuICAgICAgdW5yZWxpYWJsZTogZmFsc2UsXG4gICAgICBudW1CeXRlc0luR1BVOiB0aGlzLm51bUJ5dGVzSW5HUFUsXG4gICAgICBudW1CeXRlc0luR1BVQWxsb2NhdGVkOiB0aGlzLnRleHR1cmVNYW5hZ2VyLm51bUJ5dGVzQWxsb2NhdGVkLFxuICAgICAgbnVtQnl0ZXNJbkdQVUZyZWU6IHRoaXMudGV4dHVyZU1hbmFnZXIubnVtQnl0ZXNGcmVlXG4gICAgfSBhcyBXZWJHTE1lbW9yeUluZm87XG4gIH1cblxuICBwcml2YXRlIHN0YXJ0VGltZXIoKTogV2ViR0xRdWVyeXxDUFVUaW1lclF1ZXJ5IHtcbiAgICBpZiAoZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fUkVMSUFCTEUnKSA+IDApIHtcbiAgICAgIHJldHVybiB0aGlzLmdwZ3B1LmJlZ2luUXVlcnkoKTtcbiAgICB9XG4gICAgcmV0dXJuIHtzdGFydE1zOiB1dGlsLm5vdygpLCBlbmRNczogbnVsbH07XG4gIH1cblxuICBwcml2YXRlIGVuZFRpbWVyKHF1ZXJ5OiBXZWJHTFF1ZXJ5fENQVVRpbWVyUXVlcnkpOiBXZWJHTFF1ZXJ5fENQVVRpbWVyUXVlcnkge1xuICAgIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX0RJU0pPSU5UX1FVRVJZX1RJTUVSX0VYVEVOU0lPTl9SRUxJQUJMRScpID4gMCkge1xuICAgICAgdGhpcy5ncGdwdS5lbmRRdWVyeSgpO1xuICAgICAgcmV0dXJuIHF1ZXJ5O1xuICAgIH1cbiAgICAocXVlcnkgYXMgQ1BVVGltZXJRdWVyeSkuZW5kTXMgPSB1dGlsLm5vdygpO1xuICAgIHJldHVybiBxdWVyeTtcbiAgfVxuXG4gIHByaXZhdGUgYXN5bmMgZ2V0UXVlcnlUaW1lKHF1ZXJ5OiBXZWJHTFF1ZXJ5fENQVVRpbWVyUXVlcnkpOiBQcm9taXNlPG51bWJlcj4ge1xuICAgIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX0RJU0pPSU5UX1FVRVJZX1RJTUVSX0VYVEVOU0lPTl9SRUxJQUJMRScpID4gMCkge1xuICAgICAgcmV0dXJuIHRoaXMuZ3BncHUud2FpdEZvclF1ZXJ5QW5kR2V0VGltZShxdWVyeSBhcyBXZWJHTFF1ZXJ5KTtcbiAgICB9XG4gICAgY29uc3QgdGltZXJRdWVyeSA9IHF1ZXJ5IGFzIENQVVRpbWVyUXVlcnk7XG4gICAgcmV0dXJuIHRpbWVyUXVlcnkuZW5kTXMgLSB0aW1lclF1ZXJ5LnN0YXJ0TXM7XG4gIH1cblxuICBwcml2YXRlIHBlbmRpbmdEZWxldGVzID0gMDtcblxuICAvKipcbiAgICogRGVjcmVhc2UgdGhlIFJlZkNvdW50IG9uIHRoZSBkYXRhSWQgYW5kIGRpc3Bvc2UgdGhlIG1lbW9yeSBpZiB0aGUgZGF0YUlkXG4gICAqIGhhcyAwIHJlZkNvdW50LiBJZiB0aGVyZSBhcmUgcGVuZGluZyByZWFkIG9uIHRoZSBkYXRhLCB0aGUgZGlzcG9zYWwgd291bGRcbiAgICogYWRkZWQgdG8gdGhlIHBlbmRpbmcgZGVsZXRlIHF1ZXVlLiBSZXR1cm4gdHJ1ZSBpZiB0aGUgZGF0YUlkIGlzIHJlbW92ZWRcbiAgICogZnJvbSBiYWNrZW5kIG9yIHRoZSBiYWNrZW5kIGRvZXMgbm90IGNvbnRhaW4gdGhlIGRhdGFJZCwgZmFsc2UgaWYgdGhlXG4gICAqIGRhdGFJZCBpcyBub3QgcmVtb3ZlZC4gTWVtb3J5IG1heSBvciBtYXkgbm90IGJlIHJlbGVhc2VkIGV2ZW4gd2hlbiBkYXRhSWRcbiAgICogaXMgcmVtb3ZlZCwgd2hpY2ggYWxzbyBkZXBlbmRzIG9uIGRhdGFSZWZDb3VudCwgc2VlIGByZWxlYXNlR1BVYC5cbiAgICogQHBhcmFtIGRhdGFJZFxuICAgKiBAb2FyYW0gZm9yY2UgT3B0aW9uYWwsIHJlbW92ZSB0aGUgZGF0YSByZWdhcmRsZXNzIG9mIHJlZkNvdW50XG4gICAqL1xuICBkaXNwb3NlRGF0YShkYXRhSWQ6IERhdGFJZCwgZm9yY2UgPSBmYWxzZSk6IGJvb2xlYW4ge1xuICAgIGlmICh0aGlzLnBlbmRpbmdEaXNwb3NhbC5oYXMoZGF0YUlkKSkge1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cblxuICAgIC8vIE5vLW9wIGlmIGFscmVhZHkgZGlzcG9zZWQuXG4gICAgaWYgKCF0aGlzLnRleERhdGEuaGFzKGRhdGFJZCkpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cblxuICAgIC8vIGlmIGZvcmNlIGZsYWcgaXMgc2V0LCBjaGFuZ2UgcmVmQ291bnQgdG8gMCwgdGhpcyB3b3VsZCBlbnN1cmUgZGlzcG9zYWxcbiAgICAvLyB3aGVuIGFkZGVkIHRvIHRoZSBwZW5kaW5nRGlzcG9zYWwgcXVldWUuIE1lbW9yeSBtYXkgb3IgbWF5IG5vdCBiZVxuICAgIC8vIHJlbGVhc2VkLCB3aGljaCBhbHNvIGRlcGVuZHMgb24gZGF0YVJlZkNvdW50LCBzZWUgYHJlbGVhc2VHUFVgLlxuICAgIGlmIChmb3JjZSkge1xuICAgICAgdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpLnJlZkNvdW50ID0gMDtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpLnJlZkNvdW50LS07XG4gICAgfVxuXG4gICAgaWYgKCFmb3JjZSAmJiB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCkucmVmQ291bnQgPiAwKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuXG4gICAgaWYgKHRoaXMucGVuZGluZ1JlYWQuaGFzKGRhdGFJZCkpIHtcbiAgICAgIHRoaXMucGVuZGluZ0Rpc3Bvc2FsLmFkZChkYXRhSWQpO1xuICAgICAgdGhpcy5wZW5kaW5nRGVsZXRlcysrO1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cblxuICAgIHRoaXMucmVsZWFzZUdQVURhdGEoZGF0YUlkKTtcbiAgICBjb25zdCB7Y29tcGxleFRlbnNvckluZm9zfSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICBpZiAoY29tcGxleFRlbnNvckluZm9zICE9IG51bGwpIHtcbiAgICAgIHRoaXMuZGlzcG9zZURhdGEoY29tcGxleFRlbnNvckluZm9zLnJlYWwuZGF0YUlkLCBmb3JjZSk7XG4gICAgICB0aGlzLmRpc3Bvc2VEYXRhKGNvbXBsZXhUZW5zb3JJbmZvcy5pbWFnLmRhdGFJZCwgZm9yY2UpO1xuICAgIH1cblxuICAgIHRoaXMudGV4RGF0YS5kZWxldGUoZGF0YUlkKTtcblxuICAgIHJldHVybiB0cnVlO1xuICB9XG5cbiAgcHJpdmF0ZSByZWxlYXNlR1BVRGF0YShkYXRhSWQ6IERhdGFJZCk6IHZvaWQge1xuICAgIGNvbnN0IHt0ZXh0dXJlLCBkdHlwZSwgdGV4U2hhcGUsIHVzYWdlLCBpc1BhY2tlZCwgc2xpY2V9ID1cbiAgICAgICAgdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IGtleSA9IHNsaWNlICYmIHNsaWNlLm9yaWdEYXRhSWQgfHwgZGF0YUlkO1xuICAgIGNvbnN0IHJlZkNvdW50ID0gdGhpcy5kYXRhUmVmQ291bnQuZ2V0KGtleSk7XG5cbiAgICBpZiAocmVmQ291bnQgPiAxKSB7XG4gICAgICB0aGlzLmRhdGFSZWZDb3VudC5zZXQoa2V5LCByZWZDb3VudCAtIDEpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmRhdGFSZWZDb3VudC5kZWxldGUoa2V5KTtcbiAgICAgIGlmICh0ZXh0dXJlICE9IG51bGwpIHtcbiAgICAgICAgdGhpcy5udW1CeXRlc0luR1BVIC09IHRoaXMuY29tcHV0ZUJ5dGVzKHRleFNoYXBlLCBkdHlwZSk7XG4gICAgICAgIHRoaXMudGV4dHVyZU1hbmFnZXIucmVsZWFzZVRleHR1cmUodGV4dHVyZSwgdGV4U2hhcGUsIHVzYWdlLCBpc1BhY2tlZCk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgY29uc3QgdGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKTtcbiAgICB0ZXhEYXRhLnRleHR1cmUgPSBudWxsO1xuICAgIHRleERhdGEudGV4U2hhcGUgPSBudWxsO1xuICAgIHRleERhdGEuaXNQYWNrZWQgPSBmYWxzZTtcbiAgICB0ZXhEYXRhLnNsaWNlID0gbnVsbDtcbiAgfVxuXG4gIGdldFRleHR1cmUoZGF0YUlkOiBEYXRhSWQpOiBXZWJHTFRleHR1cmUge1xuICAgIHRoaXMudXBsb2FkVG9HUFUoZGF0YUlkKTtcbiAgICByZXR1cm4gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpLnRleHR1cmUudGV4dHVyZTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXR1cm5zIGludGVybmFsIGluZm9ybWF0aW9uIGZvciB0aGUgc3BlY2lmaWMgZGF0YSBidWNrZXQuIFVzZWQgaW4gdW5pdFxuICAgKiB0ZXN0cy5cbiAgICovXG4gIGdldERhdGFJbmZvKGRhdGFJZDogRGF0YUlkKTogVGV4dHVyZURhdGEge1xuICAgIHJldHVybiB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gIH1cblxuICAvKlxuICBUZXN0cyB3aGV0aGVyIGFsbCB0aGUgaW5wdXRzIHRvIGFuIG9wIGFyZSBzbWFsbCBhbmQgb24gdGhlIENQVS4gVGhpcyBoZXVyaXN0aWNcbiAgZGV0ZXJtaW5lcyB3aGVuIGl0IHdvdWxkIGJlIGZhc3RlciB0byBleGVjdXRlIGEga2VybmVsIG9uIHRoZSBDUFUuIFdlYkdMXG4gIGtlcm5lbHMgb3B0IGludG8gcnVubmluZyB0aGlzIGNoZWNrIGFuZCBmb3J3YXJkaW5nIHdoZW4gYXBwcm9wcmlhdGUuXG4gIFRPRE8oaHR0cHM6Ly9naXRodWIuY29tL3RlbnNvcmZsb3cvdGZqcy9pc3N1ZXMvODcyKTogRGV2ZWxvcCBhIG1vcmVcbiAgc3VzdGFpbmFibGUgc3RyYXRlZ3kgZm9yIG9wdGltaXppbmcgYmFja2VuZCBleGVjdXRpb24gb2Ygb3BzLlxuICAgKi9cbiAgc2hvdWxkRXhlY3V0ZU9uQ1BVKFxuICAgICAgaW5wdXRzOiBUZW5zb3JJbmZvW10sXG4gICAgICBzaXplVGhyZXNob2xkID0gQ1BVX0hBTkRPRkZfU0laRV9USFJFU0hPTEQpOiBib29sZWFuIHtcbiAgICByZXR1cm4gZW52KCkuZ2V0Qm9vbCgnV0VCR0xfQ1BVX0ZPUldBUkQnKSAmJlxuICAgICAgICBpbnB1dHMuZXZlcnkoXG4gICAgICAgICAgICBpbnB1dCA9PiB0aGlzLnRleERhdGEuZ2V0KGlucHV0LmRhdGFJZCkudGV4dHVyZSA9PSBudWxsICYmXG4gICAgICAgICAgICAgICAgdXRpbC5zaXplRnJvbVNoYXBlKGlucHV0LnNoYXBlKSA8IHNpemVUaHJlc2hvbGQpO1xuICB9XG5cbiAgZ2V0R1BHUFVDb250ZXh0KCk6IEdQR1BVQ29udGV4dCB7XG4gICAgcmV0dXJuIHRoaXMuZ3BncHU7XG4gIH1cblxuICB3aGVyZShjb25kaXRpb246IFRlbnNvcik6IFRlbnNvcjJEIHtcbiAgICBiYWNrZW5kX3V0aWwud2FybihcbiAgICAgICAgJ3RmLndoZXJlKCkgaW4gd2ViZ2wgbG9ja3MgdGhlIFVJIHRocmVhZC4gJyArXG4gICAgICAgICdDYWxsIHRmLndoZXJlQXN5bmMoKSBpbnN0ZWFkJyk7XG4gICAgY29uc3QgY29uZFZhbHMgPSBjb25kaXRpb24uZGF0YVN5bmMoKTtcbiAgICByZXR1cm4gd2hlcmVJbXBsKGNvbmRpdGlvbi5zaGFwZSwgY29uZFZhbHMpO1xuICB9XG5cbiAgcHJpdmF0ZSBwYWNrZWRVbmFyeU9wKHg6IFRlbnNvckluZm8sIG9wOiBzdHJpbmcsIGR0eXBlOiBEYXRhVHlwZSkge1xuICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgVW5hcnlPcFBhY2tlZFByb2dyYW0oeC5zaGFwZSwgb3ApO1xuICAgIGNvbnN0IG91dEluZm8gPSB0aGlzLmNvbXBpbGVBbmRSdW4ocHJvZ3JhbSwgW3hdLCBkdHlwZSk7XG4gICAgcmV0dXJuIGVuZ2luZSgpLm1ha2VUZW5zb3JGcm9tRGF0YUlkKFxuICAgICAgICBvdXRJbmZvLmRhdGFJZCwgb3V0SW5mby5zaGFwZSwgb3V0SW5mby5kdHlwZSk7XG4gIH1cblxuICAvLyBUT0RPKG1zb3VsYW5pbGxlKSByZW1vdmUgdGhpcyBvbmNlIHRoZSBiYWNrZW5kIGhhcyBiZWVuIG1vZHVsYXJpemVkXG4gIC8vIGEgY29weSBpcyBuZWVkZWQgaGVyZSB0byBicmVhayBhIGNpcmN1bGFyIGRlcGVuZGVuY3kuXG4gIC8vIEFsc28gcmVtb3ZlIHRoZSBvcCBmcm9tIHVuYXJ5X29wLlxuICBhYnM8VCBleHRlbmRzIFRlbnNvcj4oeDogVCk6IFQge1xuICAgIC8vIFRPRE86IGhhbmRsZSBjYXNlcyB3aGVuIHggaXMgY29tcGxleC5cbiAgICBpZiAodGhpcy5zaG91bGRFeGVjdXRlT25DUFUoW3hdKSAmJiB4LmR0eXBlICE9PSAnY29tcGxleDY0Jykge1xuICAgICAgY29uc3Qgb3V0VmFsdWVzID1cbiAgICAgICAgICBzaW1wbGVBYnNJbXBsQ1BVKHRoaXMudGV4RGF0YS5nZXQoeC5kYXRhSWQpLnZhbHVlcyBhcyBUeXBlZEFycmF5KTtcbiAgICAgIHJldHVybiB0aGlzLm1ha2VPdXRwdXQoeC5zaGFwZSwgeC5kdHlwZSwgb3V0VmFsdWVzKTtcbiAgICB9XG5cbiAgICBpZiAoZW52KCkuZ2V0Qm9vbCgnV0VCR0xfUEFDS19VTkFSWV9PUEVSQVRJT05TJykpIHtcbiAgICAgIHJldHVybiB0aGlzLnBhY2tlZFVuYXJ5T3AoeCwgdW5hcnlfb3AuQUJTLCB4LmR0eXBlKSBhcyBUO1xuICAgIH1cblxuICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgVW5hcnlPcFByb2dyYW0oeC5zaGFwZSwgdW5hcnlfb3AuQUJTKTtcbiAgICBjb25zdCBvdXRJbmZvID0gdGhpcy5jb21waWxlQW5kUnVuKHByb2dyYW0sIFt4XSk7XG4gICAgcmV0dXJuIGVuZ2luZSgpLm1ha2VUZW5zb3JGcm9tRGF0YUlkKFxuICAgICAgICAgICAgICAgb3V0SW5mby5kYXRhSWQsIG91dEluZm8uc2hhcGUsIG91dEluZm8uZHR5cGUpIGFzIFQ7XG4gIH1cblxuICBtYWtlVGVuc29ySW5mbyhcbiAgICAgIHNoYXBlOiBudW1iZXJbXSwgZHR5cGU6IERhdGFUeXBlLFxuICAgICAgdmFsdWVzPzogQmFja2VuZFZhbHVlc3xzdHJpbmdbXSk6IFRlbnNvckluZm8ge1xuICAgIGxldCBkYXRhSWQ7XG4gICAgaWYgKGR0eXBlID09PSAnc3RyaW5nJyAmJiB2YWx1ZXMgIT0gbnVsbCAmJiB2YWx1ZXMubGVuZ3RoID4gMCAmJlxuICAgICAgICB1dGlsLmlzU3RyaW5nKHZhbHVlc1swXSkpIHtcbiAgICAgIGNvbnN0IGVuY29kZWRWYWx1ZXMgPVxuICAgICAgICAgICh2YWx1ZXMgYXMge30gYXMgc3RyaW5nW10pLm1hcChkID0+IHV0aWwuZW5jb2RlU3RyaW5nKGQpKTtcblxuICAgICAgZGF0YUlkID0gdGhpcy53cml0ZShlbmNvZGVkVmFsdWVzLCBzaGFwZSwgZHR5cGUpO1xuICAgIH0gZWxzZSB7XG4gICAgICBkYXRhSWQgPSB0aGlzLndyaXRlKHZhbHVlcyBhcyBUeXBlZEFycmF5LCBzaGFwZSwgZHR5cGUpO1xuICAgIH1cblxuICAgIHRoaXMudGV4RGF0YS5nZXQoZGF0YUlkKS51c2FnZSA9IG51bGw7XG4gICAgcmV0dXJuIHtkYXRhSWQsIHNoYXBlLCBkdHlwZX07XG4gIH1cblxuICBwcml2YXRlIG1ha2VPdXRwdXQ8VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgICBzaGFwZTogbnVtYmVyW10sIGR0eXBlOiBEYXRhVHlwZSwgdmFsdWVzPzogQmFja2VuZFZhbHVlcyk6IFQge1xuICAgIGNvbnN0IHtkYXRhSWR9ID0gdGhpcy5tYWtlVGVuc29ySW5mbyhzaGFwZSwgZHR5cGUsIHZhbHVlcyk7XG4gICAgcmV0dXJuIGVuZ2luZSgpLm1ha2VUZW5zb3JGcm9tRGF0YUlkKGRhdGFJZCwgc2hhcGUsIGR0eXBlLCB0aGlzKSBhcyBUO1xuICB9XG5cbiAgdW5wYWNrVGVuc29yKGlucHV0OiBUZW5zb3JJbmZvKTogVGVuc29ySW5mbyB7XG4gICAgY29uc3QgcHJvZ3JhbSA9IG5ldyBVbnBhY2tQcm9ncmFtKGlucHV0LnNoYXBlKTtcbiAgICByZXR1cm4gdGhpcy5ydW5XZWJHTFByb2dyYW0ocHJvZ3JhbSwgW2lucHV0XSwgaW5wdXQuZHR5cGUpO1xuICB9XG5cbiAgcGFja1RlbnNvcihpbnB1dDogVGVuc29ySW5mbyk6IFRlbnNvckluZm8ge1xuICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgUGFja1Byb2dyYW0oaW5wdXQuc2hhcGUpO1xuICAgIGNvbnN0IHByZXZlbnRFYWdlclVucGFja2luZ091dHB1dCA9IHRydWU7XG4gICAgcmV0dXJuIHRoaXMucnVuV2ViR0xQcm9ncmFtKFxuICAgICAgICBwcm9ncmFtLCBbaW5wdXRdLCBpbnB1dC5kdHlwZSwgbnVsbCAvKiBjdXN0b21Vbmlmb3JtVmFsdWVzICovLFxuICAgICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPdXRwdXQpO1xuICB9XG5cbiAgcHJpdmF0ZSBwYWNrZWRSZXNoYXBlKGlucHV0OiBUZW5zb3JJbmZvLCBhZnRlclNoYXBlOiBudW1iZXJbXSk6IFRlbnNvckluZm8ge1xuICAgIGNvbnN0IGlucHV0M0RTaGFwZSA9IFtcbiAgICAgIHdlYmdsX3V0aWwuZ2V0QmF0Y2hEaW0oaW5wdXQuc2hhcGUpLFxuICAgICAgLi4ud2ViZ2xfdXRpbC5nZXRSb3dzQ29scyhpbnB1dC5zaGFwZSlcbiAgICBdIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgICBjb25zdCBpbnB1dDNEOiBUZW5zb3JJbmZvID0ge1xuICAgICAgZHR5cGU6IGlucHV0LmR0eXBlLFxuICAgICAgc2hhcGU6IGlucHV0M0RTaGFwZSxcbiAgICAgIGRhdGFJZDogaW5wdXQuZGF0YUlkXG4gICAgfTtcbiAgICBjb25zdCBhZnRlclNoYXBlQXMzRCA9IFtcbiAgICAgIHdlYmdsX3V0aWwuZ2V0QmF0Y2hEaW0oYWZ0ZXJTaGFwZSksIC4uLndlYmdsX3V0aWwuZ2V0Um93c0NvbHMoYWZ0ZXJTaGFwZSlcbiAgICBdIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcblxuICAgIGNvbnN0IHByb2dyYW0gPSBuZXcgUmVzaGFwZVBhY2tlZFByb2dyYW0oYWZ0ZXJTaGFwZUFzM0QsIGlucHV0M0RTaGFwZSk7XG4gICAgY29uc3QgcHJldmVudEVhZ2VyVW5wYWNraW5nT2ZPdXRwdXQgPSB0cnVlO1xuICAgIGNvbnN0IGN1c3RvbVZhbHVlcyA9IFtpbnB1dDNEU2hhcGVdO1xuICAgIGNvbnN0IG91dHB1dCA9IHRoaXMucnVuV2ViR0xQcm9ncmFtKFxuICAgICAgICBwcm9ncmFtLCBbaW5wdXQzRF0sIGlucHV0LmR0eXBlLCBjdXN0b21WYWx1ZXMsXG4gICAgICAgIHByZXZlbnRFYWdlclVucGFja2luZ09mT3V0cHV0KTtcbiAgICByZXR1cm4ge2RhdGFJZDogb3V0cHV0LmRhdGFJZCwgc2hhcGU6IGFmdGVyU2hhcGUsIGR0eXBlOiBvdXRwdXQuZHR5cGV9O1xuICB9XG5cbiAgcHJpdmF0ZSBkZWNvZGUoZGF0YUlkOiBEYXRhSWQsIGN1c3RvbVRleFNoYXBlPzogW251bWJlciwgbnVtYmVyXSk6XG4gICAgICBUZW5zb3JJbmZvIHtcbiAgICBjb25zdCB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHtpc1BhY2tlZCwgc2hhcGUsIGR0eXBlfSA9IHRleERhdGE7XG4gICAgaWYgKGN1c3RvbVRleFNoYXBlICE9IG51bGwpIHtcbiAgICAgIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgICAgY29uc3QgdGV4U2l6ZSA9IGN1c3RvbVRleFNoYXBlWzBdICogY3VzdG9tVGV4U2hhcGVbMV0gKiA0O1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgc2l6ZSA8PSB0ZXhTaXplLFxuICAgICAgICAgICgpID0+ICdjdXN0b21UZXhTaGFwZSBpcyB0b28gc21hbGwuICcgK1xuICAgICAgICAgICAgICAnUm93ICogQ29sdW1uICogNCBzaG91bGQgYmUgZXF1YWwgb3IgbGFyZ2VyIHRoYW4gdGhlICcgK1xuICAgICAgICAgICAgICAnc2l6ZSBvZiB0aGUgdGVuc29yIGRhdGEuJyk7XG4gICAgfVxuICAgIGNvbnN0IHNoYXBlQXMzRCA9XG4gICAgICAgIHdlYmdsX3V0aWwuZ2V0U2hhcGVBczNEKHNoYXBlKSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gICAgbGV0IHByb2dyYW07XG4gICAgaWYgKGlzUGFja2VkKSB7XG4gICAgICBwcm9ncmFtID0gbmV3IERlY29kZU1hdHJpeFBhY2tlZFByb2dyYW0oc2hhcGVBczNEKTtcbiAgICB9IGVsc2Uge1xuICAgICAgcHJvZ3JhbSA9IG5ldyBEZWNvZGVNYXRyaXhQcm9ncmFtKHNoYXBlQXMzRCk7XG4gICAgfVxuICAgIGNvbnN0IHByZXZlbnRFYWdlclVucGFja2luZ09mT3V0cHV0ID0gdHJ1ZTtcbiAgICBjb25zdCBjdXN0b21WYWx1ZXMgPVxuICAgICAgICBbY3VzdG9tVGV4U2hhcGUgIT0gbnVsbCA/IGN1c3RvbVRleFNoYXBlIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0ZXhfdXRpbC5nZXREZW5zZVRleFNoYXBlKHNoYXBlQXMzRCldO1xuICAgIGNvbnN0IG91dCA9IHRoaXMucnVuV2ViR0xQcm9ncmFtKFxuICAgICAgICBwcm9ncmFtLCBbe3NoYXBlOiBzaGFwZUFzM0QsIGR0eXBlLCBkYXRhSWR9XSwgZHR5cGUsIGN1c3RvbVZhbHVlcyxcbiAgICAgICAgcHJldmVudEVhZ2VyVW5wYWNraW5nT2ZPdXRwdXQsIGN1c3RvbVRleFNoYXBlKTtcbiAgICByZXR1cm4ge2R0eXBlLCBzaGFwZSwgZGF0YUlkOiBvdXQuZGF0YUlkfTtcbiAgfVxuXG4gIHJ1bldlYkdMUHJvZ3JhbShcbiAgICAgIHByb2dyYW06IEdQR1BVUHJvZ3JhbSwgaW5wdXRzOiBUZW5zb3JJbmZvW10sIG91dHB1dER0eXBlOiBEYXRhVHlwZSxcbiAgICAgIGN1c3RvbVVuaWZvcm1WYWx1ZXM/OiBudW1iZXJbXVtdLCBwcmV2ZW50RWFnZXJVbnBhY2tpbmdPZk91dHB1dCA9IGZhbHNlLFxuICAgICAgY3VzdG9tVGV4U2hhcGU/OiBbbnVtYmVyLCBudW1iZXJdKTogVGVuc29ySW5mbyB7XG4gICAgY29uc3Qgb3V0cHV0ID0gdGhpcy5tYWtlVGVuc29ySW5mbyhwcm9ncmFtLm91dHB1dFNoYXBlLCBvdXRwdXREdHlwZSk7XG4gICAgY29uc3Qgb3V0RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQob3V0cHV0LmRhdGFJZCk7XG4gICAgaWYgKHByb2dyYW0ucGFja2VkT3V0cHV0KSB7XG4gICAgICBvdXREYXRhLmlzUGFja2VkID0gdHJ1ZTtcbiAgICB9XG4gICAgaWYgKHByb2dyYW0ub3V0UGFja2luZ1NjaGVtZSA9PT0gdGV4X3V0aWwuUGFja2luZ1NjaGVtZS5ERU5TRSkge1xuICAgICAgY29uc3QgdGV4ZWxTaGFwZSA9IGN1c3RvbVRleFNoYXBlICE9IG51bGwgP1xuICAgICAgICAgIGN1c3RvbVRleFNoYXBlIDpcbiAgICAgICAgICB0ZXhfdXRpbC5nZXREZW5zZVRleFNoYXBlKHByb2dyYW0ub3V0cHV0U2hhcGUpO1xuICAgICAgLy8gRm9yIGEgZGVuc2VseSBwYWNrZWQgb3V0cHV0LCB3ZSBleHBsaWNpdGx5IHNldCB0ZXhTaGFwZVxuICAgICAgLy8gc28gaXQgZG9lc24ndCBnZXQgYXNzaWduZWQgbGF0ZXIgYWNjb3JkaW5nIHRvIG91ciB0eXBpY2FsIHBhY2tpbmdcbiAgICAgIC8vIHNjaGVtZSB3aGVyZWluIGEgc2luZ2xlIHRleGVsIGNhbiBvbmx5IGNvbnRhaW4gdmFsdWVzIGZyb20gYWRqYWNlbnRcbiAgICAgIC8vIHJvd3MvY29scy5cbiAgICAgIG91dERhdGEudGV4U2hhcGUgPSB0ZXhlbFNoYXBlLm1hcChkID0+IGQgKiAyKSBhcyBbbnVtYmVyLCBudW1iZXJdO1xuICAgIH1cbiAgICBpZiAocHJvZ3JhbS5vdXRUZXhVc2FnZSAhPSBudWxsKSB7XG4gICAgICBvdXREYXRhLnVzYWdlID0gcHJvZ3JhbS5vdXRUZXhVc2FnZTtcbiAgICB9XG5cbiAgICBpZiAodXRpbC5zaXplRnJvbVNoYXBlKG91dHB1dC5zaGFwZSkgPT09IDApIHtcbiAgICAgIC8vIFNob3J0LWNpcmN1aXQgdGhlIGNvbXB1dGF0aW9uIHNpbmNlIHRoZSByZXN1bHQgaXMgZW1wdHkgKGhhcyAwIGluIGl0c1xuICAgICAgLy8gc2hhcGUpLlxuICAgICAgb3V0RGF0YS52YWx1ZXMgPVxuICAgICAgICAgIHV0aWwuZ2V0VHlwZWRBcnJheUZyb21EVHlwZShvdXRwdXQuZHR5cGUgYXMgJ2Zsb2F0MzInLCAwKTtcbiAgICAgIHJldHVybiBvdXRwdXQ7XG4gICAgfVxuXG4gICAgY29uc3QgZGF0YVRvRGlzcG9zZTogVGVuc29ySW5mb1tdID0gW107XG4gICAgY29uc3QgaW5wdXRzRGF0YTogVGVuc29yRGF0YVtdID0gaW5wdXRzLm1hcChpbnB1dCA9PiB7XG4gICAgICBpZiAoaW5wdXQuZHR5cGUgPT09ICdjb21wbGV4NjQnKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgIGBHUEdQVVByb2dyYW0gZG9lcyBub3Qgc3VwcG9ydCBjb21wbGV4NjQgaW5wdXQuIEZvciBjb21wbGV4NjQgYCArXG4gICAgICAgICAgICBgZHR5cGVzLCBwbGVhc2Ugc2VwYXJhdGUgdGhlIHByb2dyYW0gaW50byByZWFsIGFuZCBpbWFnaW5hcnkgYCArXG4gICAgICAgICAgICBgcGFydHMuYCk7XG4gICAgICB9XG5cbiAgICAgIGxldCB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChpbnB1dC5kYXRhSWQpO1xuXG4gICAgICBpZiAodGV4RGF0YS50ZXh0dXJlID09IG51bGwpIHtcbiAgICAgICAgaWYgKCFwcm9ncmFtLnBhY2tlZElucHV0cyAmJlxuICAgICAgICAgICAgdXRpbC5zaXplRnJvbVNoYXBlKGlucHV0LnNoYXBlKSA8PVxuICAgICAgICAgICAgICAgIGVudigpLmdldE51bWJlcignV0VCR0xfU0laRV9VUExPQURfVU5JRk9STScpKSB7XG4gICAgICAgICAgLy8gVXBsb2FkIHNtYWxsIHRlbnNvcnMgdGhhdCBsaXZlIG9uIHRoZSBDUFUgYXMgdW5pZm9ybXMsIG5vdCBhc1xuICAgICAgICAgIC8vIHRleHR1cmVzLiBEbyB0aGlzIG9ubHkgd2hlbiB0aGUgZW52aXJvbm1lbnQgc3VwcG9ydHMgMzJiaXQgZmxvYXRzXG4gICAgICAgICAgLy8gZHVlIHRvIHByb2JsZW1zIHdoZW4gY29tcGFyaW5nIDE2Yml0IGZsb2F0cyB3aXRoIDMyYml0IGZsb2F0cy5cbiAgICAgICAgICAvLyBUT0RPKGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzgyMSk6IE1ha2UgaXRcbiAgICAgICAgICAvLyBwb3NzaWJsZSBmb3IgcGFja2VkIHNoYWRlcnMgdG8gc2FtcGxlIGZyb20gdW5pZm9ybXMuXG4gICAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICAgIHNoYXBlOiBpbnB1dC5zaGFwZSxcbiAgICAgICAgICAgIHRleERhdGE6IG51bGwsXG4gICAgICAgICAgICBpc1VuaWZvcm06IHRydWUsXG4gICAgICAgICAgICB1bmlmb3JtVmFsdWVzOiB0ZXhEYXRhLnZhbHVlcyBhcyBUeXBlZEFycmF5XG4gICAgICAgICAgfTtcbiAgICAgICAgfVxuXG4gICAgICAgIC8vIFRoaXMgZW5zdXJlcyB0aGF0IGlmIGEgcGFja2VkIHByb2dyYW0ncyBpbnB1dHMgaGF2ZSBub3QgeWV0IGJlZW5cbiAgICAgICAgLy8gdXBsb2FkZWQgdG8gdGhlIEdQVSwgdGhleSBnZXQgdXBsb2FkZWQgYXMgcGFja2VkIHJpZ2h0IG9mZiB0aGUgYmF0LlxuICAgICAgICBpZiAocHJvZ3JhbS5wYWNrZWRJbnB1dHMpIHtcbiAgICAgICAgICB0ZXhEYXRhLmlzUGFja2VkID0gdHJ1ZTtcbiAgICAgICAgICB0ZXhEYXRhLnNoYXBlID0gaW5wdXQuc2hhcGU7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgdGhpcy51cGxvYWRUb0dQVShpbnB1dC5kYXRhSWQpO1xuICAgICAgaWYgKCEhdGV4RGF0YS5pc1BhY2tlZCAhPT0gISFwcm9ncmFtLnBhY2tlZElucHV0cykge1xuICAgICAgICBpbnB1dCA9IHRleERhdGEuaXNQYWNrZWQgPyB0aGlzLnVucGFja1RlbnNvcihpbnB1dCkgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aGlzLnBhY2tUZW5zb3IoaW5wdXQpO1xuICAgICAgICBkYXRhVG9EaXNwb3NlLnB1c2goaW5wdXQpO1xuICAgICAgICB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChpbnB1dC5kYXRhSWQpO1xuICAgICAgfSBlbHNlIGlmIChcbiAgICAgICAgICB0ZXhEYXRhLmlzUGFja2VkICYmXG4gICAgICAgICAgIXdlYmdsX3V0aWwuaXNSZXNoYXBlRnJlZSh0ZXhEYXRhLnNoYXBlLCBpbnB1dC5zaGFwZSkpIHtcbiAgICAgICAgLy8gVGhpcyBpcyBhIHNwZWNpYWwgY2FzZSB3aGVyZSBhIHRleHR1cmUgZXhpc3RzIGZvciBhIHRlbnNvclxuICAgICAgICAvLyBidXQgdGhlIHNoYXBlcyBhcmUgaW5jb21wYXRpYmxlIChkdWUgdG8gcGFja2luZyBjb25zdHJhaW50cykgYmVjYXVzZVxuICAgICAgICAvLyB0aGUgdGVuc29yIGRpZCBub3QgaGF2ZSBhIGNoYW5jZSB0byBnbyB0aHJvdWdoIHRoZSBwYWNrZWQgcmVzaGFwZVxuICAgICAgICAvLyBzaGFkZXIuIFRoaXMgb25seSBoYXBwZW5zIHdoZW4gd2UgcmVzaGFwZSB0aGUgKnNhbWUqIHRlbnNvciB0byBmb3JtXG4gICAgICAgIC8vICpkaXN0aW5jdCogaW5wdXRzIHRvIGFuIG9wLCBlLmcuIGRvdHRpbmcgYSB2ZWN0b3Igd2l0aCBpdHNlbGYuIFRoaXNcbiAgICAgICAgLy8gY2FzZSB3aWxsIGRpc2FwcGVhciBvbmNlIHBhY2tlZCB1cGxvYWRpbmcgaXMgdGhlIGRlZmF1bHQuXG5cbiAgICAgICAgY29uc3Qgc2F2ZWRJbnB1dCA9IGlucHV0O1xuICAgICAgICBjb25zdCB0YXJnZXRTaGFwZSA9IGlucHV0LnNoYXBlO1xuXG4gICAgICAgIGlucHV0LnNoYXBlID0gdGV4RGF0YS5zaGFwZTtcbiAgICAgICAgaW5wdXQgPSB0aGlzLnBhY2tlZFJlc2hhcGUoaW5wdXQgYXMgVGVuc29yLCB0YXJnZXRTaGFwZSk7XG4gICAgICAgIGRhdGFUb0Rpc3Bvc2UucHVzaChpbnB1dCk7XG4gICAgICAgIHRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGlucHV0LmRhdGFJZCk7XG5cbiAgICAgICAgc2F2ZWRJbnB1dC5zaGFwZSA9IHRhcmdldFNoYXBlO1xuICAgICAgfVxuXG4gICAgICByZXR1cm4ge3NoYXBlOiBpbnB1dC5zaGFwZSwgdGV4RGF0YSwgaXNVbmlmb3JtOiBmYWxzZX07XG4gICAgfSk7XG5cbiAgICB0aGlzLnVwbG9hZFRvR1BVKG91dHB1dC5kYXRhSWQpO1xuICAgIGNvbnN0IG91dHB1dERhdGE6XG4gICAgICAgIFRlbnNvckRhdGEgPSB7c2hhcGU6IG91dHB1dC5zaGFwZSwgdGV4RGF0YTogb3V0RGF0YSwgaXNVbmlmb3JtOiBmYWxzZX07XG4gICAgY29uc3Qga2V5ID0gZ3BncHVfbWF0aC5tYWtlU2hhZGVyS2V5KHByb2dyYW0sIGlucHV0c0RhdGEsIG91dHB1dERhdGEpO1xuICAgIGNvbnN0IGJpbmFyeSA9IHRoaXMuZ2V0QW5kU2F2ZUJpbmFyeShrZXksICgpID0+IHtcbiAgICAgIHJldHVybiBncGdwdV9tYXRoLmNvbXBpbGVQcm9ncmFtKFxuICAgICAgICAgIHRoaXMuZ3BncHUsIHByb2dyYW0sIGlucHV0c0RhdGEsIG91dHB1dERhdGEpO1xuICAgIH0pO1xuICAgIGNvbnN0IHNob3VsZFRpbWVQcm9ncmFtID0gdGhpcy5hY3RpdmVUaW1lcnMgIT0gbnVsbDtcbiAgICBsZXQgcXVlcnk6IFdlYkdMUXVlcnl8Q1BVVGltZXJRdWVyeTtcbiAgICBpZiAoc2hvdWxkVGltZVByb2dyYW0pIHtcbiAgICAgIHF1ZXJ5ID0gdGhpcy5zdGFydFRpbWVyKCk7XG4gICAgfVxuXG4gICAgaWYgKCFlbnYoKS5nZXQoJ0VOR0lORV9DT01QSUxFX09OTFknKSkge1xuICAgICAgZ3BncHVfbWF0aC5ydW5Qcm9ncmFtKFxuICAgICAgICAgIHRoaXMuZ3BncHUsIGJpbmFyeSwgaW5wdXRzRGF0YSwgb3V0cHV0RGF0YSwgY3VzdG9tVW5pZm9ybVZhbHVlcyk7XG4gICAgfVxuXG4gICAgZGF0YVRvRGlzcG9zZS5mb3JFYWNoKGluZm8gPT4gdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhpbmZvKSk7XG5cbiAgICBpZiAoc2hvdWxkVGltZVByb2dyYW0pIHtcbiAgICAgIHF1ZXJ5ID0gdGhpcy5lbmRUaW1lcihxdWVyeSk7XG4gICAgICB0aGlzLmFjdGl2ZVRpbWVycy5wdXNoKFxuICAgICAgICAgIHtuYW1lOiBwcm9ncmFtLmNvbnN0cnVjdG9yLm5hbWUsIHF1ZXJ5OiB0aGlzLmdldFF1ZXJ5VGltZShxdWVyeSl9KTtcbiAgICB9XG5cbiAgICBjb25zdCBnbEZsdXNoVGhyZXNob2xkID0gZW52KCkuZ2V0KCdXRUJHTF9GTFVTSF9USFJFU0hPTEQnKTtcbiAgICAvLyBNYW51YWxseSBHTCBmbHVzaCByZXF1ZXN0ZWRcbiAgICBpZiAoZ2xGbHVzaFRocmVzaG9sZCA+IDApIHtcbiAgICAgIGNvbnN0IHRpbWUgPSB1dGlsLm5vdygpO1xuICAgICAgaWYgKCh0aW1lIC0gdGhpcy5sYXN0R2xGbHVzaFRpbWUpID4gZ2xGbHVzaFRocmVzaG9sZCkge1xuICAgICAgICB0aGlzLmdwZ3B1LmdsLmZsdXNoKCk7XG4gICAgICAgIHRoaXMubGFzdEdsRmx1c2hUaW1lID0gdGltZTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICBpZiAoIWVudigpLmdldEJvb2woJ1dFQkdMX0xBWklMWV9VTlBBQ0snKSAmJiBvdXREYXRhLmlzUGFja2VkICYmXG4gICAgICAgIHByZXZlbnRFYWdlclVucGFja2luZ09mT3V0cHV0ID09PSBmYWxzZSkge1xuICAgICAgY29uc3QgdW5wYWNrZWQgPSB0aGlzLnVucGFja1RlbnNvcihvdXRwdXQpO1xuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhvdXRwdXQpO1xuICAgICAgcmV0dXJuIHVucGFja2VkO1xuICAgIH1cbiAgICByZXR1cm4gb3V0cHV0O1xuICB9XG5cbiAgY29tcGlsZUFuZFJ1bihcbiAgICAgIHByb2dyYW06IEdQR1BVUHJvZ3JhbSwgaW5wdXRzOiBUZW5zb3JJbmZvW10sIG91dHB1dER0eXBlPzogRGF0YVR5cGUsXG4gICAgICBjdXN0b21Vbmlmb3JtVmFsdWVzPzogbnVtYmVyW11bXSxcbiAgICAgIHByZXZlbnRFYWdlclVucGFja2luZ09mT3V0cHV0ID0gZmFsc2UpOiBUZW5zb3JJbmZvIHtcbiAgICBvdXRwdXREdHlwZSA9IG91dHB1dER0eXBlIHx8IGlucHV0c1swXS5kdHlwZTtcbiAgICBjb25zdCBvdXRJbmZvID0gdGhpcy5ydW5XZWJHTFByb2dyYW0oXG4gICAgICAgIHByb2dyYW0sIGlucHV0cywgb3V0cHV0RHR5cGUsIGN1c3RvbVVuaWZvcm1WYWx1ZXMsXG4gICAgICAgIHByZXZlbnRFYWdlclVucGFja2luZ09mT3V0cHV0KTtcbiAgICByZXR1cm4gb3V0SW5mbztcbiAgfVxuXG4gIHByaXZhdGUgZ2V0QW5kU2F2ZUJpbmFyeShrZXk6IHN0cmluZywgZ2V0QmluYXJ5OiAoKSA9PiBHUEdQVUJpbmFyeSk6XG4gICAgICBHUEdQVUJpbmFyeSB7XG4gICAgaWYgKCEoa2V5IGluIHRoaXMuYmluYXJ5Q2FjaGUpKSB7XG4gICAgICB0aGlzLmJpbmFyeUNhY2hlW2tleV0gPSBnZXRCaW5hcnkoKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuYmluYXJ5Q2FjaGVba2V5XTtcbiAgfVxuXG4gIGdldFRleHR1cmVNYW5hZ2VyKCk6IFRleHR1cmVNYW5hZ2VyIHtcbiAgICByZXR1cm4gdGhpcy50ZXh0dXJlTWFuYWdlcjtcbiAgfVxuXG4gIHByaXZhdGUgZGlzcG9zZWQgPSBmYWxzZTtcblxuICBkaXNwb3NlKCkge1xuICAgIGlmICh0aGlzLmRpc3Bvc2VkKSB7XG4gICAgICByZXR1cm47XG4gICAgfVxuICAgIC8vIEF2b2lkIGRpc3Bvc2luZyB0aGUgY29tcGlsZWQgd2ViZ2wgcHJvZ3JhbXMgZHVyaW5nIHVuaXQgdGVzdGluZyBiZWNhdXNlXG4gICAgLy8gaXQgc2xvd3MgZG93biB0ZXN0IGV4ZWN1dGlvbi5cbiAgICBpZiAoIWVudigpLmdldEJvb2woJ0lTX1RFU1QnKSkge1xuICAgICAgY29uc3QgYWxsS2V5cyA9IE9iamVjdC5rZXlzKHRoaXMuYmluYXJ5Q2FjaGUpO1xuICAgICAgYWxsS2V5cy5mb3JFYWNoKGtleSA9PiB7XG4gICAgICAgIHRoaXMuZ3BncHUuZGVsZXRlUHJvZ3JhbSh0aGlzLmJpbmFyeUNhY2hlW2tleV0ud2ViR0xQcm9ncmFtKTtcbiAgICAgICAgZGVsZXRlIHRoaXMuYmluYXJ5Q2FjaGVba2V5XTtcbiAgICAgIH0pO1xuICAgIH1cbiAgICB0aGlzLnRleHR1cmVNYW5hZ2VyLmRpc3Bvc2UoKTtcbiAgICBpZiAodGhpcy5jYW52YXMgIT0gbnVsbCAmJlxuICAgICAgICAodHlwZW9mIChIVE1MQ2FudmFzRWxlbWVudCkgIT09ICd1bmRlZmluZWQnICYmXG4gICAgICAgICB0aGlzLmNhbnZhcyBpbnN0YW5jZW9mIEhUTUxDYW52YXNFbGVtZW50KSkge1xuICAgICAgdGhpcy5jYW52YXMucmVtb3ZlKCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuY2FudmFzID0gbnVsbDtcbiAgICB9XG4gICAgaWYgKHRoaXMuZ3BncHVDcmVhdGVkTG9jYWxseSkge1xuICAgICAgdGhpcy5ncGdwdS5wcm9ncmFtID0gbnVsbDtcbiAgICAgIHRoaXMuZ3BncHUuZGlzcG9zZSgpO1xuICAgIH1cbiAgICB0aGlzLmRpc3Bvc2VkID0gdHJ1ZTtcbiAgfVxuXG4gIGZsb2F0UHJlY2lzaW9uKCk6IDE2fDMyIHtcbiAgICBpZiAodGhpcy5mbG9hdFByZWNpc2lvblZhbHVlID09IG51bGwpIHtcbiAgICAgIHRoaXMuZmxvYXRQcmVjaXNpb25WYWx1ZSA9IHRpZHkoKCkgPT4ge1xuICAgICAgICBpZiAoIWVudigpLmdldCgnV0VCR0xfUkVOREVSX0ZMT0FUMzJfRU5BQkxFRCcpKSB7XG4gICAgICAgICAgLy8gTW9tZW50YXJpbHkgc3dpdGNoaW5nIERFQlVHIGZsYWcgdG8gZmFsc2Ugc28gd2UgZG9uJ3QgdGhyb3cgYW5cbiAgICAgICAgICAvLyBlcnJvciB0cnlpbmcgdG8gdXBsb2FkIGEgc21hbGwgdmFsdWUuXG4gICAgICAgICAgY29uc3QgZGVidWdGbGFnID0gZW52KCkuZ2V0Qm9vbCgnREVCVUcnKTtcbiAgICAgICAgICBlbnYoKS5zZXQoJ0RFQlVHJywgZmFsc2UpO1xuICAgICAgICAgIGNvbnN0IHVuZGVyZmxvd0NoZWNrVmFsdWUgPSB0aGlzLmFicyhzY2FsYXIoMWUtOCkpLmRhdGFTeW5jKClbMF07XG4gICAgICAgICAgZW52KCkuc2V0KCdERUJVRycsIGRlYnVnRmxhZyk7XG5cbiAgICAgICAgICBpZiAodW5kZXJmbG93Q2hlY2tWYWx1ZSA+IDApIHtcbiAgICAgICAgICAgIHJldHVybiAzMjtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIDE2O1xuICAgICAgfSk7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmZsb2F0UHJlY2lzaW9uVmFsdWU7XG4gIH1cblxuICAvKiogUmV0dXJucyB0aGUgc21hbGxlc3QgcmVwcmVzZW50YWJsZSBudW1iZXIuICAqL1xuICBlcHNpbG9uKCk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuZmxvYXRQcmVjaXNpb24oKSA9PT0gMzIgPyBFUFNJTE9OX0ZMT0FUMzIgOiBFUFNJTE9OX0ZMT0FUMTY7XG4gIH1cblxuICB1cGxvYWRUb0dQVShkYXRhSWQ6IERhdGFJZCk6IHZvaWQge1xuICAgIGNvbnN0IHRleERhdGEgPSB0aGlzLnRleERhdGEuZ2V0KGRhdGFJZCk7XG4gICAgY29uc3Qge3NoYXBlLCBkdHlwZSwgdmFsdWVzLCB0ZXh0dXJlLCB1c2FnZSwgaXNQYWNrZWR9ID0gdGV4RGF0YTtcblxuICAgIGlmICh0ZXh0dXJlICE9IG51bGwpIHtcbiAgICAgIC8vIEFycmF5IGlzIGFscmVhZHkgb24gR1BVLiBOby1vcC5cbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3Qgc2hvdWxkVGltZVByb2dyYW0gPSB0aGlzLmFjdGl2ZVRpbWVycyAhPSBudWxsO1xuICAgIGxldCBzdGFydDogbnVtYmVyO1xuICAgIGlmIChzaG91bGRUaW1lUHJvZ3JhbSkge1xuICAgICAgc3RhcnQgPSB1dGlsLm5vdygpO1xuICAgIH1cblxuICAgIGxldCB0ZXhTaGFwZSA9IHRleERhdGEudGV4U2hhcGU7XG4gICAgaWYgKHRleFNoYXBlID09IG51bGwpIHtcbiAgICAgIC8vIFRoaXMgdGV4U2hhcGUgbWF5IG5vdCBiZSB0aGUgZmluYWwgdGV4dHVyZSBzaGFwZS4gRm9yIHBhY2tlZCBvciBkZW5zZVxuICAgICAgLy8gdGV4dHVyZXMsIHRoZSB0ZXhTaGFwZSB3aWxsIGJlIGNoYW5nZWQgd2hlbiB0ZXh0dXJlcyBhcmUgY3JlYXRlZC5cbiAgICAgIHRleFNoYXBlID0gd2ViZ2xfdXRpbC5nZXRUZXh0dXJlU2hhcGVGcm9tTG9naWNhbFNoYXBlKHNoYXBlLCBpc1BhY2tlZCk7XG4gICAgICB0ZXhEYXRhLnRleFNoYXBlID0gdGV4U2hhcGU7XG4gICAgfVxuXG4gICAgaWYgKHZhbHVlcyAhPSBudWxsKSB7XG4gICAgICBjb25zdCBzaGFwZUFzM0QgPSB3ZWJnbF91dGlsLmdldFNoYXBlQXMzRChzaGFwZSk7XG5cbiAgICAgIGxldCBwcm9ncmFtO1xuICAgICAgbGV0IHdpZHRoID0gdGV4U2hhcGVbMV0sIGhlaWdodCA9IHRleFNoYXBlWzBdO1xuICAgICAgY29uc3QgaXNCeXRlQXJyYXkgPVxuICAgICAgICAgIHZhbHVlcyBpbnN0YW5jZW9mIFVpbnQ4QXJyYXkgfHwgdmFsdWVzIGluc3RhbmNlb2YgVWludDhDbGFtcGVkQXJyYXk7XG5cbiAgICAgIC8vIHRleHR1cmUgZm9yIGZsb2F0IGFycmF5IGlzIFBoeXNpY2FsVGV4dHVyZVR5cGUuUEFDS0VEXzJYMl9GTE9BVDMyLCB3ZVxuICAgICAgLy8gbmVlZCB0byBtYWtlIHN1cmUgdGhlIHVwbG9hZCB1c2VzIHRoZSBzYW1lIHBhY2tlZCBzaXplXG4gICAgICBpZiAoaXNQYWNrZWQgfHwgIWlzQnl0ZUFycmF5KSB7XG4gICAgICAgIFt3aWR0aCwgaGVpZ2h0XSA9IHRleF91dGlsLmdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KFxuICAgICAgICAgICAgdGV4U2hhcGVbMF0sIHRleFNoYXBlWzFdKTtcbiAgICAgIH1cblxuICAgICAgaWYgKGlzUGFja2VkKSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgRW5jb2RlTWF0cml4UGFja2VkUHJvZ3JhbShzaGFwZUFzM0QsIGlzQnl0ZUFycmF5KTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHByb2dyYW0gPSBuZXcgRW5jb2RlTWF0cml4UHJvZ3JhbShzaGFwZUFzM0QsIGlzQnl0ZUFycmF5KTtcbiAgICAgIH1cblxuICAgICAgLy8gVGV4U2hhcGUgZm9yIGZsb2F0IGFycmF5IG5lZWRzIHRvIGJlIHRoZSBvcmlnaW5hbCBzaGFwZSwgd2hpY2ggYnl0ZVxuICAgICAgLy8gYXJyYXkgbmVlZHMgdG8gYmUgcGFja2VkIHNpemUuIFRoaXMgYWxsb3cgdGhlIGRhdGEgdXBsb2FkIHNoYXBlIHRvIGJlXG4gICAgICAvLyBtYXRjaGVkIHdpdGggdGV4dHVyZSBjcmVhdGlvbiBsb2dpYy5cbiAgICAgIGNvbnN0IHRlbXBEZW5zZUlucHV0VGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0gPVxuICAgICAgICAgIGlzQnl0ZUFycmF5ID8gW2hlaWdodCwgd2lkdGhdIDogdGV4U2hhcGU7XG4gICAgICBjb25zdCB0ZW1wRGVuc2VJbnB1dEhhbmRsZSA9XG4gICAgICAgICAgdGhpcy5tYWtlVGVuc29ySW5mbyh0ZW1wRGVuc2VJbnB1dFRleFNoYXBlLCBkdHlwZSk7XG4gICAgICBjb25zdCB0ZW1wRGVuc2VJbnB1dFRleERhdGEgPVxuICAgICAgICAgIHRoaXMudGV4RGF0YS5nZXQodGVtcERlbnNlSW5wdXRIYW5kbGUuZGF0YUlkKTtcbiAgICAgIGlmIChpc0J5dGVBcnJheSkge1xuICAgICAgICB0ZW1wRGVuc2VJbnB1dFRleERhdGEudXNhZ2UgPSBUZXh0dXJlVXNhZ2UuUElYRUxTO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGVtcERlbnNlSW5wdXRUZXhEYXRhLnVzYWdlID0gVGV4dHVyZVVzYWdlLlVQTE9BRDtcbiAgICAgIH1cbiAgICAgIHRlbXBEZW5zZUlucHV0VGV4RGF0YS50ZXhTaGFwZSA9IHRlbXBEZW5zZUlucHV0VGV4U2hhcGU7XG4gICAgICB0aGlzLmdwZ3B1LnVwbG9hZERlbnNlTWF0cml4VG9UZXh0dXJlKFxuICAgICAgICAgIHRoaXMuZ2V0VGV4dHVyZSh0ZW1wRGVuc2VJbnB1dEhhbmRsZS5kYXRhSWQpLCB3aWR0aCwgaGVpZ2h0LFxuICAgICAgICAgIHZhbHVlcyBhcyBUeXBlZEFycmF5KTtcblxuICAgICAgY29uc3QgY3VzdG9tVmFsdWVzID0gW1toZWlnaHQsIHdpZHRoXV07XG4gICAgICAvLyBXZSB3YW50IHRoZSBvdXRwdXQgdG8gcmVtYWluIHBhY2tlZCByZWdhcmRsZXNzIG9mIHRoZSB2YWx1ZSBvZlxuICAgICAgLy8gV0VCR0xfUEFDSy5cbiAgICAgIGNvbnN0IHByZXZlbnRFYWdlclVucGFja2luZyA9IHRydWU7XG4gICAgICBjb25zdCBlbmNvZGVkT3V0cHV0VGFyZ2V0ID0gdGhpcy5ydW5XZWJHTFByb2dyYW0oXG4gICAgICAgICAgcHJvZ3JhbSwgW3RlbXBEZW5zZUlucHV0SGFuZGxlXSwgZHR5cGUsIGN1c3RvbVZhbHVlcyxcbiAgICAgICAgICBwcmV2ZW50RWFnZXJVbnBhY2tpbmcpO1xuXG4gICAgICAvLyBIYXZlIHRoZSBvcmlnaW5hbCB0ZXh0dXJlIGFzc3VtZSB0aGUgaWRlbnRpdHkgb2YgdGhlIGVuY29kZWQgb3V0cHV0LlxuICAgICAgY29uc3Qgb3V0cHV0VGV4RGF0YSA9IHRoaXMudGV4RGF0YS5nZXQoZW5jb2RlZE91dHB1dFRhcmdldC5kYXRhSWQpO1xuICAgICAgdGV4RGF0YS50ZXhTaGFwZSA9IG91dHB1dFRleERhdGEudGV4U2hhcGU7XG4gICAgICB0ZXhEYXRhLmlzUGFja2VkID0gb3V0cHV0VGV4RGF0YS5pc1BhY2tlZDtcbiAgICAgIHRleERhdGEudXNhZ2UgPSBvdXRwdXRUZXhEYXRhLnVzYWdlO1xuXG4gICAgICBpZiAoIWVudigpLmdldCgnRU5HSU5FX0NPTVBJTEVfT05MWScpKSB7XG4gICAgICAgIHRleERhdGEudGV4dHVyZSA9IG91dHB1dFRleERhdGEudGV4dHVyZTtcbiAgICAgICAgLy8gT25jZSB1cGxvYWRlZCwgZG9uJ3Qgc3RvcmUgdGhlIHZhbHVlcyBvbiBjcHUuXG4gICAgICAgIHRleERhdGEudmFsdWVzID0gbnVsbDtcbiAgICAgICAgdGhpcy50ZXhEYXRhLmRlbGV0ZShlbmNvZGVkT3V0cHV0VGFyZ2V0LmRhdGFJZCk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aGlzLmRpc3Bvc2VEYXRhKGVuY29kZWRPdXRwdXRUYXJnZXQuZGF0YUlkKTtcbiAgICAgIH1cblxuICAgICAgdGhpcy5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyh0ZW1wRGVuc2VJbnB1dEhhbmRsZSk7XG5cbiAgICAgIGlmIChzaG91bGRUaW1lUHJvZ3JhbSkge1xuICAgICAgICB0aGlzLnVwbG9hZFdhaXRNcyArPSB1dGlsLm5vdygpIC0gc3RhcnQ7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IG5ld1RleHR1cmUgPSB0aGlzLmFjcXVpcmVUZXh0dXJlKHRleFNoYXBlLCB1c2FnZSwgZHR5cGUsIGlzUGFja2VkKTtcbiAgICAgIHRleERhdGEudGV4dHVyZSA9IG5ld1RleHR1cmU7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBjb252ZXJ0QW5kQ2FjaGVPbkNQVShkYXRhSWQ6IERhdGFJZCwgZmxvYXQzMlZhbHVlcz86IEZsb2F0MzJBcnJheSk6XG4gICAgICBUeXBlZEFycmF5IHtcbiAgICBjb25zdCB0ZXhEYXRhID0gdGhpcy50ZXhEYXRhLmdldChkYXRhSWQpO1xuICAgIGNvbnN0IHtkdHlwZX0gPSB0ZXhEYXRhO1xuXG4gICAgdGhpcy5yZWxlYXNlR1BVRGF0YShkYXRhSWQpO1xuXG4gICAgaWYgKGZsb2F0MzJWYWx1ZXMgIT0gbnVsbCkge1xuICAgICAgdGV4RGF0YS52YWx1ZXMgPSBmbG9hdDMyVG9UeXBlZEFycmF5KGZsb2F0MzJWYWx1ZXMsIGR0eXBlIGFzICdmbG9hdDMyJyk7XG4gICAgfVxuICAgIHJldHVybiB0ZXhEYXRhLnZhbHVlcyBhcyBUeXBlZEFycmF5O1xuICB9XG5cbiAgcHJpdmF0ZSBhY3F1aXJlVGV4dHVyZShcbiAgICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCB0ZXhUeXBlOiBUZXh0dXJlVXNhZ2UsIGR0eXBlOiBEYXRhVHlwZSxcbiAgICAgIGlzUGFja2VkOiBib29sZWFuKTogVGV4dHVyZSB7XG4gICAgdGhpcy5udW1CeXRlc0luR1BVICs9IHRoaXMuY29tcHV0ZUJ5dGVzKHRleFNoYXBlLCBkdHlwZSk7XG4gICAgaWYgKCF0aGlzLndhcm5lZEFib3V0TWVtb3J5ICYmXG4gICAgICAgIHRoaXMubnVtQnl0ZXNJbkdQVSA+IHRoaXMubnVtTUJCZWZvcmVXYXJuaW5nICogMTAyNCAqIDEwMjQpIHtcbiAgICAgIGNvbnN0IG1iID0gKHRoaXMubnVtQnl0ZXNJbkdQVSAvIDEwMjQgLyAxMDI0KS50b0ZpeGVkKDIpO1xuICAgICAgdGhpcy53YXJuZWRBYm91dE1lbW9yeSA9IHRydWU7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYEhpZ2ggbWVtb3J5IHVzYWdlIGluIEdQVTogJHttYn0gTUIsIGAgK1xuICAgICAgICAgIGBtb3N0IGxpa2VseSBkdWUgdG8gYSBtZW1vcnkgbGVha2ApO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy50ZXh0dXJlTWFuYWdlci5hY3F1aXJlVGV4dHVyZSh0ZXhTaGFwZSwgdGV4VHlwZSwgaXNQYWNrZWQpO1xuICB9XG5cbiAgcHJpdmF0ZSBjb21wdXRlQnl0ZXMoc2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIGR0eXBlOiBEYXRhVHlwZSkge1xuICAgIHJldHVybiBzaGFwZVswXSAqIHNoYXBlWzFdICogdXRpbC5ieXRlc1BlckVsZW1lbnQoZHR5cGUpO1xuICB9XG5cbiAgY2hlY2tDb21waWxlQ29tcGxldGlvbigpIHtcbiAgICBmb3IgKGNvbnN0IFssIGJpbmFyeV0gb2YgT2JqZWN0LmVudHJpZXModGhpcy5iaW5hcnlDYWNoZSkpIHtcbiAgICAgIHRoaXMuY2hlY2tDb21wbGV0aW9uXyhiaW5hcnkpO1xuICAgIH1cbiAgfVxuXG4gIGFzeW5jIGNoZWNrQ29tcGlsZUNvbXBsZXRpb25Bc3luYygpOiBQcm9taXNlPGJvb2xlYW5bXT4ge1xuICAgIGNvbnN0IHBzID0gW107XG4gICAgaWYgKHRoaXMuZ3BncHUucGFyYWxsZWxDb21waWxhdGlvbkV4dGVuc2lvbikge1xuICAgICAgZm9yIChjb25zdCBbLCBiaW5hcnldIG9mIE9iamVjdC5lbnRyaWVzKHRoaXMuYmluYXJ5Q2FjaGUpKSB7XG4gICAgICAgIHBzLnB1c2godGhpcy5jaGVja0NvbXBsZXRpb25Bc3luY18oYmluYXJ5KSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gUHJvbWlzZS5hbGwocHMpO1xuICAgIH0gZWxzZSB7XG4gICAgICBmb3IgKGNvbnN0IFssIGJpbmFyeV0gb2YgT2JqZWN0LmVudHJpZXModGhpcy5iaW5hcnlDYWNoZSkpIHtcbiAgICAgICAgY29uc3QgcDogUHJvbWlzZTxib29sZWFuPiA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7XG4gICAgICAgICAgdHJ5IHtcbiAgICAgICAgICAgIHRoaXMuY2hlY2tDb21wbGV0aW9uXyhiaW5hcnkpO1xuICAgICAgICAgICAgcmVzb2x2ZSh0cnVlKTtcbiAgICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgICAgfVxuICAgICAgICB9KTtcbiAgICAgICAgcHMucHVzaChwKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiBQcm9taXNlLmFsbChwcyk7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBhc3luYyBjaGVja0NvbXBsZXRpb25Bc3luY18oYmluYXJ5OiBHUEdQVUJpbmFyeSk6IFByb21pc2U8Ym9vbGVhbj4ge1xuICAgIGlmICh0aGlzLmdwZ3B1LmdsLmdldFByb2dyYW1QYXJhbWV0ZXIoXG4gICAgICAgICAgICBiaW5hcnkud2ViR0xQcm9ncmFtLFxuICAgICAgICAgICAgdGhpcy5ncGdwdS5wYXJhbGxlbENvbXBpbGF0aW9uRXh0ZW5zaW9uLkNPTVBMRVRJT05fU1RBVFVTX0tIUikpIHtcbiAgICAgIHJldHVybiB0aGlzLmNoZWNrQ29tcGxldGlvbl8oYmluYXJ5KTtcbiAgICB9IGVsc2Uge1xuICAgICAgYXdhaXQgbmV4dEZyYW1lKCk7XG4gICAgICByZXR1cm4gdGhpcy5jaGVja0NvbXBsZXRpb25Bc3luY18oYmluYXJ5KTtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIGNoZWNrQ29tcGxldGlvbl8oYmluYXJ5OiBHUEdQVUJpbmFyeSk6IGJvb2xlYW4ge1xuICAgIGlmICh0aGlzLmdwZ3B1LmdsLmdldFByb2dyYW1QYXJhbWV0ZXIoXG4gICAgICAgICAgICBiaW5hcnkud2ViR0xQcm9ncmFtLCB0aGlzLmdwZ3B1LmdsLkxJTktfU1RBVFVTKSA9PT0gZmFsc2UpIHtcbiAgICAgIGNvbnNvbGUubG9nKHRoaXMuZ3BncHUuZ2wuZ2V0UHJvZ3JhbUluZm9Mb2coYmluYXJ5LndlYkdMUHJvZ3JhbSkpO1xuICAgICAgaWYgKHRoaXMuZ3BncHUuZ2wuZ2V0U2hhZGVyUGFyYW1ldGVyKFxuICAgICAgICAgICAgICBiaW5hcnkuZnJhZ21lbnRTaGFkZXIsIHRoaXMuZ3BncHUuZ2wuQ09NUElMRV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgICAgICB3ZWJnbF91dGlsLmxvZ1NoYWRlclNvdXJjZUFuZEluZm9Mb2coXG4gICAgICAgICAgICBiaW5hcnkuc291cmNlLFxuICAgICAgICAgICAgdGhpcy5ncGdwdS5nbC5nZXRTaGFkZXJJbmZvTG9nKGJpbmFyeS5mcmFnbWVudFNoYWRlcikpO1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBjb21waWxlIGZyYWdtZW50IHNoYWRlci4nKTtcbiAgICAgIH1cbiAgICAgIHRocm93IG5ldyBFcnJvcignRmFpbGVkIHRvIGxpbmsgdmVydGV4IGFuZCBmcmFnbWVudCBzaGFkZXJzLicpO1xuICAgIH1cbiAgICByZXR1cm4gdHJ1ZTtcbiAgfVxuXG4gIGdldFVuaWZvcm1Mb2NhdGlvbnMoKSB7XG4gICAgZm9yIChjb25zdCBbLCBiaW5hcnldIG9mIE9iamVjdC5lbnRyaWVzKHRoaXMuYmluYXJ5Q2FjaGUpKSB7XG4gICAgICBjb25zdCB7XG4gICAgICAgIHVuaWZvcm1Mb2NhdGlvbnMsXG4gICAgICAgIGN1c3RvbVVuaWZvcm1Mb2NhdGlvbnMsXG4gICAgICAgIGluZkxvYyxcbiAgICAgICAgbmFuTG9jLFxuICAgICAgICBpblNoYXBlc0xvY2F0aW9ucyxcbiAgICAgICAgaW5UZXhTaGFwZXNMb2NhdGlvbnMsXG4gICAgICAgIG91dFNoYXBlTG9jYXRpb24sXG4gICAgICAgIG91dFNoYXBlU3RyaWRlc0xvY2F0aW9uLFxuICAgICAgICBvdXRUZXhTaGFwZUxvY2F0aW9uXG4gICAgICB9ID0gZ2V0VW5pZm9ybUxvY2F0aW9ucyh0aGlzLmdwZ3B1LCBiaW5hcnkucHJvZ3JhbSwgYmluYXJ5LndlYkdMUHJvZ3JhbSk7XG4gICAgICBiaW5hcnkudW5pZm9ybUxvY2F0aW9ucyA9IHVuaWZvcm1Mb2NhdGlvbnM7XG4gICAgICBiaW5hcnkuY3VzdG9tVW5pZm9ybUxvY2F0aW9ucyA9IGN1c3RvbVVuaWZvcm1Mb2NhdGlvbnM7XG4gICAgICBiaW5hcnkuaW5mTG9jID0gaW5mTG9jO1xuICAgICAgYmluYXJ5Lm5hbkxvYyA9IG5hbkxvYztcbiAgICAgIGJpbmFyeS5pblNoYXBlc0xvY2F0aW9ucyA9IGluU2hhcGVzTG9jYXRpb25zO1xuICAgICAgYmluYXJ5LmluVGV4U2hhcGVzTG9jYXRpb25zID0gaW5UZXhTaGFwZXNMb2NhdGlvbnM7XG4gICAgICBiaW5hcnkub3V0U2hhcGVMb2NhdGlvbiA9IG91dFNoYXBlTG9jYXRpb247XG4gICAgICBiaW5hcnkub3V0U2hhcGVTdHJpZGVzTG9jYXRpb24gPSBvdXRTaGFwZVN0cmlkZXNMb2NhdGlvbjtcbiAgICAgIGJpbmFyeS5vdXRUZXhTaGFwZUxvY2F0aW9uID0gb3V0VGV4U2hhcGVMb2NhdGlvbjtcbiAgICB9XG4gIH1cbn1cblxuZnVuY3Rpb24gZmxvYXQzMlRvVHlwZWRBcnJheTxEIGV4dGVuZHMgTnVtZXJpY0RhdGFUeXBlPihcbiAgICBhOiBGbG9hdDMyQXJyYXksIGR0eXBlOiBEKTogdGYuRGF0YVR5cGVNYXBbRF0ge1xuICBpZiAoZHR5cGUgPT09ICdmbG9hdDMyJyB8fCBkdHlwZSA9PT0gJ2NvbXBsZXg2NCcpIHtcbiAgICByZXR1cm4gYSBhcyB0Zi5EYXRhVHlwZU1hcFtEXTtcbiAgfSBlbHNlIGlmIChkdHlwZSA9PT0gJ2ludDMyJyB8fCBkdHlwZSA9PT0gJ2Jvb2wnKSB7XG4gICAgY29uc3QgcmVzdWx0ID0gKGR0eXBlID09PSAnaW50MzInKSA/IG5ldyBJbnQzMkFycmF5KGEubGVuZ3RoKSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG5ldyBVaW50OEFycmF5KGEubGVuZ3RoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHJlc3VsdC5sZW5ndGg7ICsraSkge1xuICAgICAgcmVzdWx0W2ldID0gTWF0aC5yb3VuZChhW2ldKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdCBhcyB0Zi5EYXRhVHlwZU1hcFtEXTtcbiAgfSBlbHNlIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVua25vd24gZHR5cGUgJHtkdHlwZX1gKTtcbiAgfVxufVxuIl19