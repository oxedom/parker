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
import { getWebGLContext, setWebGLContext } from './canvas_util';
import * as gpgpu_util from './gpgpu_util';
import * as tex_util from './tex_util';
import * as webgl_util from './webgl_util';
export class GPGPUContext {
    constructor(gl) {
        this.outputTexture = null;
        this.program = null;
        this.disposed = false;
        this.vertexAttrsAreBound = false;
        this.itemsToPoll = [];
        const glVersion = env().getNumber('WEBGL_VERSION');
        if (gl != null) {
            this.gl = gl;
            setWebGLContext(glVersion, gl);
        }
        else {
            this.gl = getWebGLContext(glVersion);
        }
        // WebGL 2.0 enables texture floats without an extension.
        let COLOR_BUFFER_FLOAT = 'WEBGL_color_buffer_float';
        const COLOR_BUFFER_HALF_FLOAT = 'EXT_color_buffer_half_float';
        this.parallelCompilationExtension =
            this.gl.getExtension('KHR_parallel_shader_compile');
        if (env().getNumber('WEBGL_VERSION') === 1) {
            const TEXTURE_FLOAT = 'OES_texture_float';
            const TEXTURE_HALF_FLOAT = 'OES_texture_half_float';
            this.textureFloatExtension =
                webgl_util.getExtensionOrThrow(this.gl, TEXTURE_FLOAT);
            if (webgl_util.hasExtension(this.gl, TEXTURE_HALF_FLOAT)) {
                this.textureHalfFloatExtension =
                    webgl_util.getExtensionOrThrow(this.gl, TEXTURE_HALF_FLOAT);
            }
            else if (env().get('WEBGL_FORCE_F16_TEXTURES')) {
                throw new Error('GL context does not support half float textures, yet the ' +
                    'environment flag WEBGL_FORCE_F16_TEXTURES is set to true.');
            }
            this.colorBufferFloatExtension = this.gl.getExtension(COLOR_BUFFER_FLOAT);
            if (webgl_util.hasExtension(this.gl, COLOR_BUFFER_HALF_FLOAT)) {
                this.colorBufferHalfFloatExtension =
                    webgl_util.getExtensionOrThrow(this.gl, COLOR_BUFFER_HALF_FLOAT);
            }
            else if (env().get('WEBGL_FORCE_F16_TEXTURES')) {
                throw new Error('GL context does not support color renderable half floats, yet ' +
                    'the environment flag WEBGL_FORCE_F16_TEXTURES is set to true.');
            }
        }
        else {
            COLOR_BUFFER_FLOAT = 'EXT_color_buffer_float';
            if (webgl_util.hasExtension(this.gl, COLOR_BUFFER_FLOAT)) {
                this.colorBufferFloatExtension =
                    this.gl.getExtension(COLOR_BUFFER_FLOAT);
            }
            else if (webgl_util.hasExtension(this.gl, COLOR_BUFFER_HALF_FLOAT)) {
                this.colorBufferHalfFloatExtension =
                    this.gl.getExtension(COLOR_BUFFER_HALF_FLOAT);
            }
            else {
                throw new Error('GL context does not support color renderable floats');
            }
        }
        this.vertexBuffer = gpgpu_util.createVertexBuffer(this.gl);
        this.indexBuffer = gpgpu_util.createIndexBuffer(this.gl);
        this.framebuffer = webgl_util.createFramebuffer(this.gl);
        this.textureConfig =
            tex_util.getTextureConfig(this.gl, this.textureHalfFloatExtension);
    }
    get debug() {
        return env().getBool('DEBUG');
    }
    dispose() {
        if (this.disposed) {
            return;
        }
        if (this.program != null) {
            console.warn('Disposing a GPGPUContext that still has a bound WebGLProgram.' +
                ' This is probably a resource leak, delete the program with ' +
                'GPGPUContext.deleteProgram before disposing.');
        }
        if (this.outputTexture != null) {
            console.warn('Disposing a GPGPUContext that still has a bound output matrix ' +
                'texture.  This is probably a resource leak, delete the output ' +
                'matrix texture with GPGPUContext.deleteMatrixTexture before ' +
                'disposing.');
        }
        const gl = this.gl;
        webgl_util.callAndCheck(gl, () => gl.finish());
        webgl_util.callAndCheck(gl, () => gl.bindFramebuffer(gl.FRAMEBUFFER, null));
        webgl_util.callAndCheck(gl, () => gl.deleteFramebuffer(this.framebuffer));
        webgl_util.callAndCheck(gl, () => gl.bindBuffer(gl.ARRAY_BUFFER, null));
        webgl_util.callAndCheck(gl, () => gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null));
        webgl_util.callAndCheck(gl, () => gl.deleteBuffer(this.indexBuffer));
        this.disposed = true;
    }
    createFloat32MatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createFloat32MatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    createFloat16MatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createFloat16MatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    createUnsignedBytesMatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createUnsignedBytesMatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    uploadPixelDataToTexture(texture, pixels) {
        this.throwIfDisposed();
        gpgpu_util.uploadPixelDataToTexture(this.gl, texture, pixels);
    }
    uploadDenseMatrixToTexture(texture, width, height, data) {
        this.throwIfDisposed();
        gpgpu_util.uploadDenseMatrixToTexture(this.gl, texture, width, height, data, this.textureConfig);
    }
    createFloat16PackedMatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createFloat16PackedMatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    createPackedMatrixTexture(rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createPackedMatrixTexture(this.gl, rows, columns, this.textureConfig);
    }
    deleteMatrixTexture(texture) {
        this.throwIfDisposed();
        if (this.outputTexture === texture) {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
            this.outputTexture = null;
        }
        webgl_util.callAndCheck(this.gl, () => this.gl.deleteTexture(texture));
    }
    downloadByteEncodedFloatMatrixFromOutputTexture(texture, rows, columns) {
        return this.downloadMatrixDriver(texture, () => gpgpu_util.downloadByteEncodedFloatMatrixFromOutputTexture(this.gl, rows, columns, this.textureConfig));
    }
    downloadPackedMatrixFromBuffer(buffer, batch, rows, columns, physicalRows, physicalCols) {
        return gpgpu_util.downloadPackedMatrixFromBuffer(this.gl, buffer, batch, rows, columns, physicalRows, physicalCols, this.textureConfig);
    }
    downloadFloat32MatrixFromBuffer(buffer, size) {
        return gpgpu_util.downloadFloat32MatrixFromBuffer(this.gl, buffer, size);
    }
    createBufferFromTexture(texture, rows, columns) {
        this.bindTextureToFrameBuffer(texture);
        const result = gpgpu_util.createBufferFromOutputTexture(this.gl, rows, columns, this.textureConfig);
        this.unbindTextureToFrameBuffer();
        return result;
    }
    createAndWaitForFence() {
        const fenceContext = this.createFence(this.gl);
        return this.pollFence(fenceContext);
    }
    createFence(gl) {
        let query;
        let isFencePassed;
        if (env().getBool('WEBGL_FENCE_API_ENABLED')) {
            const gl2 = gl;
            const sync = gl2.fenceSync(gl2.SYNC_GPU_COMMANDS_COMPLETE, 0);
            gl.flush();
            isFencePassed = () => {
                const status = gl2.clientWaitSync(sync, 0, 0);
                return status === gl2.ALREADY_SIGNALED ||
                    status === gl2.CONDITION_SATISFIED;
            };
            query = sync;
        }
        else if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') > 0) {
            query = this.beginQuery();
            this.endQuery();
            isFencePassed = () => this.isQueryAvailable(query, env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'));
        }
        else {
            // If we have no way to fence, return true immediately. This will fire in
            // WebGL 1.0 when there is no disjoint query timer. In this case, because
            // the fence passes immediately, we'll immediately ask for a download of
            // the texture, which will cause the UI thread to hang.
            isFencePassed = () => true;
        }
        return { query, isFencePassed };
    }
    downloadMatrixFromPackedTexture(texture, physicalRows, physicalCols) {
        return this.downloadMatrixDriver(texture, () => gpgpu_util.downloadMatrixFromPackedOutputTexture(this.gl, physicalRows, physicalCols));
    }
    createProgram(fragmentShader) {
        this.throwIfDisposed();
        const gl = this.gl;
        if (this.vertexShader == null) {
            this.vertexShader = gpgpu_util.createVertexShader(gl);
        }
        const program = webgl_util.createProgram(gl);
        webgl_util.callAndCheck(gl, () => gl.attachShader(program, this.vertexShader));
        webgl_util.callAndCheck(gl, () => gl.attachShader(program, fragmentShader));
        webgl_util.linkProgram(gl, program);
        if (this.debug) {
            webgl_util.validateProgram(gl, program);
        }
        if (!this.vertexAttrsAreBound) {
            this.setProgram(program);
            this.vertexAttrsAreBound = gpgpu_util.bindVertexProgramAttributeStreams(gl, this.program, this.vertexBuffer);
        }
        return program;
    }
    deleteProgram(program) {
        this.throwIfDisposed();
        if (program === this.program) {
            this.program = null;
        }
        if (program != null) {
            webgl_util.callAndCheck(this.gl, () => this.gl.deleteProgram(program));
        }
    }
    setProgram(program) {
        this.throwIfDisposed();
        this.program = program;
        if ((this.program != null) && this.debug) {
            webgl_util.validateProgram(this.gl, this.program);
        }
        webgl_util.callAndCheck(this.gl, () => this.gl.useProgram(program));
    }
    getUniformLocation(program, uniformName, shouldThrow = true) {
        this.throwIfDisposed();
        if (shouldThrow) {
            return webgl_util.getProgramUniformLocationOrThrow(this.gl, program, uniformName);
        }
        else {
            return webgl_util.getProgramUniformLocation(this.gl, program, uniformName);
        }
    }
    getAttributeLocation(program, attribute) {
        this.throwIfDisposed();
        return webgl_util.callAndCheck(this.gl, () => this.gl.getAttribLocation(program, attribute));
    }
    getUniformLocationNoThrow(program, uniformName) {
        this.throwIfDisposed();
        return this.gl.getUniformLocation(program, uniformName);
    }
    setInputMatrixTexture(inputMatrixTexture, uniformLocation, textureUnit) {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        webgl_util.bindTextureToProgramUniformSampler(this.gl, inputMatrixTexture, uniformLocation, textureUnit);
    }
    setOutputMatrixTexture(outputMatrixTexture, rows, columns) {
        this.setOutputMatrixTextureDriver(outputMatrixTexture, columns, rows);
    }
    setOutputPackedMatrixTexture(outputPackedMatrixTexture, rows, columns) {
        this.throwIfDisposed();
        const [width, height] = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns);
        this.setOutputMatrixTextureDriver(outputPackedMatrixTexture, width, height);
    }
    setOutputMatrixWriteRegion(startRow, numRows, startColumn, numColumns) {
        this.setOutputMatrixWriteRegionDriver(startColumn, startRow, numColumns, numRows);
    }
    setOutputPackedMatrixWriteRegion(startRow, numRows, startColumn, numColumns) {
        throw new Error('setOutputPackedMatrixWriteRegion not implemented.');
    }
    debugValidate() {
        if (this.program != null) {
            webgl_util.validateProgram(this.gl, this.program);
        }
        webgl_util.validateFramebuffer(this.gl);
    }
    executeProgram() {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        const gl = this.gl;
        if (this.debug) {
            this.debugValidate();
        }
        webgl_util.callAndCheck(gl, () => gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0));
    }
    blockUntilAllProgramsCompleted() {
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, () => this.gl.finish());
    }
    getQueryTimerExtension() {
        if (this.disjointQueryTimerExtension == null) {
            this.disjointQueryTimerExtension =
                webgl_util.getExtensionOrThrow(this.gl, env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2 ?
                    'EXT_disjoint_timer_query_webgl2' :
                    'EXT_disjoint_timer_query');
        }
        return this.disjointQueryTimerExtension;
    }
    getQueryTimerExtensionWebGL2() {
        return this.getQueryTimerExtension();
    }
    getQueryTimerExtensionWebGL1() {
        return this.getQueryTimerExtension();
    }
    beginQuery() {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
            const gl2 = this.gl;
            const ext = this.getQueryTimerExtensionWebGL2();
            const query = gl2.createQuery();
            gl2.beginQuery(ext.TIME_ELAPSED_EXT, query);
            return query;
        }
        const ext = this.getQueryTimerExtensionWebGL1();
        const query = ext.createQueryEXT();
        ext.beginQueryEXT(ext.TIME_ELAPSED_EXT, query);
        return query;
    }
    endQuery() {
        if (env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION') === 2) {
            const gl2 = this.gl;
            const ext = this.getQueryTimerExtensionWebGL2();
            gl2.endQuery(ext.TIME_ELAPSED_EXT);
            return;
        }
        const ext = this.getQueryTimerExtensionWebGL1();
        ext.endQueryEXT(ext.TIME_ELAPSED_EXT);
    }
    async waitForQueryAndGetTime(query) {
        await util.repeatedTry(() => this.disposed || // while testing contexts are created / disposed
            // in rapid succession, so without this check we
            // may poll for the query timer indefinitely
            this.isQueryAvailable(query, env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION')));
        return this.getQueryTime(query, env().getNumber('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_VERSION'));
    }
    getQueryTime(query, queryTimerVersion) {
        if (queryTimerVersion === 0) {
            return null;
        }
        if (queryTimerVersion === 2) {
            const gl2 = this.gl;
            const timeElapsedNanos = gl2.getQueryParameter(query, gl2.QUERY_RESULT);
            // Return milliseconds.
            return timeElapsedNanos / 1000000;
        }
        else {
            const ext = this.getQueryTimerExtensionWebGL1();
            const timeElapsedNanos = ext.getQueryObjectEXT(query, ext.QUERY_RESULT_EXT);
            // Return milliseconds.
            return timeElapsedNanos / 1000000;
        }
    }
    isQueryAvailable(query, queryTimerVersion) {
        if (queryTimerVersion === 0) {
            return true;
        }
        if (queryTimerVersion === 2) {
            const gl2 = this.gl;
            const ext = this.getQueryTimerExtensionWebGL2();
            const available = gl2.getQueryParameter(query, gl2.QUERY_RESULT_AVAILABLE);
            if (this.disjoint == null) {
                this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
            }
            return available && !this.disjoint;
        }
        else {
            const ext = this.getQueryTimerExtensionWebGL1();
            const available = ext.getQueryObjectEXT(query, ext.QUERY_RESULT_AVAILABLE_EXT);
            if (this.disjoint == null) {
                this.disjoint = this.gl.getParameter(ext.GPU_DISJOINT_EXT);
            }
            return available && !this.disjoint;
        }
    }
    pollFence(fenceContext) {
        return new Promise(resolve => {
            this.addItemToPoll(() => fenceContext.isFencePassed(), () => resolve());
        });
    }
    pollItems() {
        // Find the last query that has finished.
        const index = linearSearchLastTrue(this.itemsToPoll.map(x => x.isDoneFn));
        for (let i = 0; i <= index; ++i) {
            const { resolveFn } = this.itemsToPoll[i];
            resolveFn();
        }
        this.itemsToPoll = this.itemsToPoll.slice(index + 1);
    }
    addItemToPoll(isDoneFn, resolveFn) {
        this.itemsToPoll.push({ isDoneFn, resolveFn });
        if (this.itemsToPoll.length > 1) {
            // We already have a running loop that polls.
            return;
        }
        // Start a new loop that polls.
        util.repeatedTry(() => {
            this.pollItems();
            // End the loop if no more items to poll.
            return this.itemsToPoll.length === 0;
        });
    }
    bindTextureToFrameBuffer(texture) {
        this.throwIfDisposed();
        webgl_util.bindColorTextureToFramebuffer(this.gl, texture, this.framebuffer);
        if (this.debug) {
            webgl_util.validateFramebuffer(this.gl);
        }
    }
    unbindTextureToFrameBuffer() {
        if (this.outputTexture != null) {
            webgl_util.bindColorTextureToFramebuffer(this.gl, this.outputTexture, this.framebuffer);
            if (this.debug) {
                webgl_util.validateFramebuffer(this.gl);
            }
        }
        else {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
        }
    }
    downloadMatrixDriver(texture, downloadAndDecode) {
        this.bindTextureToFrameBuffer(texture);
        const result = downloadAndDecode();
        this.unbindTextureToFrameBuffer();
        return result;
    }
    setOutputMatrixTextureDriver(outputMatrixTextureMaybePacked, width, height) {
        this.throwIfDisposed();
        const gl = this.gl;
        webgl_util.bindColorTextureToFramebuffer(gl, outputMatrixTextureMaybePacked, this.framebuffer);
        if (this.debug) {
            webgl_util.validateFramebuffer(gl);
        }
        this.outputTexture = outputMatrixTextureMaybePacked;
        webgl_util.callAndCheck(gl, () => gl.viewport(0, 0, width, height));
        webgl_util.callAndCheck(gl, () => gl.scissor(0, 0, width, height));
    }
    setOutputMatrixWriteRegionDriver(x, y, width, height) {
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, () => this.gl.scissor(x, y, width, height));
    }
    throwIfDisposed() {
        if (this.disposed) {
            throw new Error('Attempted to use disposed GPGPUContext.');
        }
    }
    throwIfNoProgram() {
        if (this.program == null) {
            throw new Error('No GPU program is currently set.');
        }
    }
}
/**
 * Finds the index of the last true element using linear search.
 * Note: We can't do binary search because Chrome expects us to explicitly
 * test all fences before download:
 * https://github.com/tensorflow/tfjs/issues/1145
 */
export function linearSearchLastTrue(arr) {
    let i = 0;
    for (; i < arr.length; ++i) {
        const isDone = arr[i]();
        if (!isDone) {
            break;
        }
    }
    return i - 1;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3BncHVfY29udGV4dC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvZ3BncHVfY29udGV4dC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsR0FBRyxFQUF5QixJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV2RSxPQUFPLEVBQUMsZUFBZSxFQUFFLGVBQWUsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUMvRCxPQUFPLEtBQUssVUFBVSxNQUFNLGNBQWMsQ0FBQztBQUMzQyxPQUFPLEtBQUssUUFBUSxNQUFNLFlBQVksQ0FBQztBQUd2QyxPQUFPLEtBQUssVUFBVSxNQUFNLGNBQWMsQ0FBQztBQU8zQyxNQUFNLE9BQU8sWUFBWTtJQW1CdkIsWUFBWSxFQUEwQjtRQVB0QyxrQkFBYSxHQUFzQixJQUFJLENBQUM7UUFDeEMsWUFBTyxHQUFzQixJQUFJLENBQUM7UUFDMUIsYUFBUSxHQUFHLEtBQUssQ0FBQztRQXNPakIsd0JBQW1CLEdBQUcsS0FBSyxDQUFDO1FBdVA1QixnQkFBVyxHQUFlLEVBQUUsQ0FBQztRQXZkbkMsTUFBTSxTQUFTLEdBQUcsR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLGVBQWUsQ0FBQyxDQUFDO1FBQ25ELElBQUksRUFBRSxJQUFJLElBQUksRUFBRTtZQUNkLElBQUksQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDO1lBQ2IsZUFBZSxDQUFDLFNBQVMsRUFBRSxFQUFFLENBQUMsQ0FBQztTQUNoQzthQUFNO1lBQ0wsSUFBSSxDQUFDLEVBQUUsR0FBRyxlQUFlLENBQUMsU0FBUyxDQUFDLENBQUM7U0FDdEM7UUFDRCx5REFBeUQ7UUFDekQsSUFBSSxrQkFBa0IsR0FBRywwQkFBMEIsQ0FBQztRQUNwRCxNQUFNLHVCQUF1QixHQUFHLDZCQUE2QixDQUFDO1FBQzlELElBQUksQ0FBQyw0QkFBNEI7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LElBQUksQ0FBQyxRQUFRLEVBQUU7WUFDakIsTUFBTSxJQUFJLEtBQUssQ0FBQyx5Q0FBeUMsQ0FBQyxDQUFDO1NBQzVEO0lBQ0gsQ0FBQztJQUVPLGdCQUFnQjtRQUN0QixJQUFJLElBQUksQ0FBQyxPQUFPLElBQUksSUFBSSxFQUFFO1lBQ3hCLE1BQU0sSUFBSSxLQUFLLENBQUMsa0NBQWtDLENBQUMsQ0FBQztTQUNyRDtJQUNILENBQUM7Q0FDRjtBQU9EOzs7OztHQUtHO0FBQ0gsTUFBTSxVQUFVLG9CQUFvQixDQUFDLEdBQXlCO0lBQzVELElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQztJQUNWLE9BQU8sQ0FBQyxHQUFHLEdBQUcsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7UUFDMUIsTUFBTSxNQUFNLEdBQUcsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUM7UUFDeEIsSUFBSSxDQUFDLE1BQU0sRUFBRTtZQUNYLE1BQU07U0FDUDtLQUNGO0lBQ0QsT0FBTyxDQUFDLEdBQUcsQ0FBQyxDQUFDO0FBQ2YsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE3IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtlbnYsIFBpeGVsRGF0YSwgVHlwZWRBcnJheSwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtnZXRXZWJHTENvbnRleHQsIHNldFdlYkdMQ29udGV4dH0gZnJvbSAnLi9jYW52YXNfdXRpbCc7XG5pbXBvcnQgKiBhcyBncGdwdV91dGlsIGZyb20gJy4vZ3BncHVfdXRpbCc7XG5pbXBvcnQgKiBhcyB0ZXhfdXRpbCBmcm9tICcuL3RleF91dGlsJztcbmltcG9ydCB7VGV4dHVyZSwgVGV4dHVyZUNvbmZpZ30gZnJvbSAnLi90ZXhfdXRpbCc7XG5pbXBvcnQge1dlYkdMMURpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvbiwgV2ViR0wyRGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9uLCBXZWJHTFBhcmFsbGVsQ29tcGlsYXRpb25FeHRlbnNpb259IGZyb20gJy4vd2ViZ2xfdHlwZXMnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5leHBvcnQgaW50ZXJmYWNlIEZlbmNlQ29udGV4dCB7XG4gIHF1ZXJ5OiBXZWJHTFF1ZXJ5fFdlYkdMU3luYztcbiAgaXNGZW5jZVBhc3NlZCgpOiBib29sZWFuO1xufVxuXG5leHBvcnQgY2xhc3MgR1BHUFVDb250ZXh0IHtcbiAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dDtcbiAgdGV4dHVyZUZsb2F0RXh0ZW5zaW9uOiB7fTtcbiAgdGV4dHVyZUhhbGZGbG9hdEV4dGVuc2lvbjoge307XG4gIGNvbG9yQnVmZmVyRmxvYXRFeHRlbnNpb246IHt9O1xuICBjb2xvckJ1ZmZlckhhbGZGbG9hdEV4dGVuc2lvbjoge307XG4gIGRpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvbjogV2ViR0wyRGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9ufFxuICAgICAgV2ViR0wxRGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9uO1xuICBwYXJhbGxlbENvbXBpbGF0aW9uRXh0ZW5zaW9uOiBXZWJHTFBhcmFsbGVsQ29tcGlsYXRpb25FeHRlbnNpb247XG4gIHZlcnRleEJ1ZmZlcjogV2ViR0xCdWZmZXI7XG4gIGluZGV4QnVmZmVyOiBXZWJHTEJ1ZmZlcjtcbiAgZnJhbWVidWZmZXI6IFdlYkdMRnJhbWVidWZmZXI7XG4gIG91dHB1dFRleHR1cmU6IFdlYkdMVGV4dHVyZXxudWxsID0gbnVsbDtcbiAgcHJvZ3JhbTogV2ViR0xQcm9ncmFtfG51bGwgPSBudWxsO1xuICBwcml2YXRlIGRpc3Bvc2VkID0gZmFsc2U7XG4gIHByaXZhdGUgZGlzam9pbnQ6IGJvb2xlYW47XG4gIHByaXZhdGUgdmVydGV4U2hhZGVyOiBXZWJHTFNoYWRlcjtcbiAgdGV4dHVyZUNvbmZpZzogVGV4dHVyZUNvbmZpZztcblxuICBjb25zdHJ1Y3RvcihnbD86IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICAgIGNvbnN0IGdsVmVyc2lvbiA9IGVudigpLmdldE51bWJlcignV0VCR0xfVkVSU0lPTicpO1xuICAgIGlmIChnbCAhPSBudWxsKSB7XG4gICAgICB0aGlzLmdsID0gZ2w7XG4gICAgICBzZXRXZWJHTENvbnRleHQoZ2xWZXJzaW9uLCBnbCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuZ2wgPSBnZXRXZWJHTENvbnRleHQoZ2xWZXJzaW9uKTtcbiAgICB9XG4gICAgLy8gV2ViR0wgMi4wIGVuYWJsZXMgdGV4dHVyZSBmbG9hdHMgd2l0aG91dCBhbiBleHRlbnNpb24uXG4gICAgbGV0IENPTE9SX0JVRkZFUl9GTE9BVCA9ICdXRUJHTF9jb2xvcl9idWZmZXJfZmxvYXQnO1xuICAgIGNvbnN0IENPTE9SX0JVRkZFUl9IQUxGX0ZMT0FUID0gJ0VYVF9jb2xvcl9idWZmZXJfaGFsZl9mbG9hdCc7XG4gICAgdGhpcy5wYXJhbGxlbENvbXBpbGF0aW9uRXh0ZW5zaW9uID1cbiAgICAgICAgdGhpcy5nbC5nZXRFeHRlbnNpb24oJ0tIUl9wYXJhbGxlbF9zaGFkZXJfY29tcGlsZScpO1xuICAgIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1ZFUlNJT04nKSA9PT0gMSkge1xuICAgICAgY29uc3QgVEVYVFVSRV9GTE9BVCA9ICdPRVNfdGV4dHVyZV9mbG9hdCc7XG4gICAgICBjb25zdCBURVhUVVJFX0hBTEZfRkxPQVQgPSAnT0VTX3RleHR1cmVfaGFsZl9mbG9hdCc7XG5cbiAgICAgIHRoaXMudGV4dHVyZUZsb2F0RXh0ZW5zaW9uID1cbiAgICAgICAgICB3ZWJnbF91dGlsLmdldEV4dGVuc2lvbk9yVGhyb3codGhpcy5nbCwgVEVYVFVSRV9GTE9BVCk7XG4gICAgICBpZiAod2ViZ2xfdXRpbC5oYXNFeHRlbnNpb24odGhpcy5nbCwgVEVYVFVSRV9IQUxGX0ZMT0FUKSkge1xuICAgICAgICB0aGlzLnRleHR1cmVIYWxmRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgICAgd2ViZ2xfdXRpbC5nZXRFeHRlbnNpb25PclRocm93KHRoaXMuZ2wsIFRFWFRVUkVfSEFMRl9GTE9BVCk7XG4gICAgICB9IGVsc2UgaWYgKGVudigpLmdldCgnV0VCR0xfRk9SQ0VfRjE2X1RFWFRVUkVTJykpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgJ0dMIGNvbnRleHQgZG9lcyBub3Qgc3VwcG9ydCBoYWxmIGZsb2F0IHRleHR1cmVzLCB5ZXQgdGhlICcgK1xuICAgICAgICAgICAgJ2Vudmlyb25tZW50IGZsYWcgV0VCR0xfRk9SQ0VfRjE2X1RFWFRVUkVTIGlzIHNldCB0byB0cnVlLicpO1xuICAgICAgfVxuXG4gICAgICB0aGlzLmNvbG9yQnVmZmVyRmxvYXRFeHRlbnNpb24gPSB0aGlzLmdsLmdldEV4dGVuc2lvbihDT0xPUl9CVUZGRVJfRkxPQVQpO1xuICAgICAgaWYgKHdlYmdsX3V0aWwuaGFzRXh0ZW5zaW9uKHRoaXMuZ2wsIENPTE9SX0JVRkZFUl9IQUxGX0ZMT0FUKSkge1xuICAgICAgICB0aGlzLmNvbG9yQnVmZmVySGFsZkZsb2F0RXh0ZW5zaW9uID1cbiAgICAgICAgICAgIHdlYmdsX3V0aWwuZ2V0RXh0ZW5zaW9uT3JUaHJvdyh0aGlzLmdsLCBDT0xPUl9CVUZGRVJfSEFMRl9GTE9BVCk7XG4gICAgICB9IGVsc2UgaWYgKGVudigpLmdldCgnV0VCR0xfRk9SQ0VfRjE2X1RFWFRVUkVTJykpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgJ0dMIGNvbnRleHQgZG9lcyBub3Qgc3VwcG9ydCBjb2xvciByZW5kZXJhYmxlIGhhbGYgZmxvYXRzLCB5ZXQgJyArXG4gICAgICAgICAgICAndGhlIGVudmlyb25tZW50IGZsYWcgV0VCR0xfRk9SQ0VfRjE2X1RFWFRVUkVTIGlzIHNldCB0byB0cnVlLicpO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICBDT0xPUl9CVUZGRVJfRkxPQVQgPSAnRVhUX2NvbG9yX2J1ZmZlcl9mbG9hdCc7XG4gICAgICBpZiAod2ViZ2xfdXRpbC5oYXNFeHRlbnNpb24odGhpcy5nbCwgQ09MT1JfQlVGRkVSX0ZMT0FUKSkge1xuICAgICAgICB0aGlzLmNvbG9yQnVmZmVyRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgICAgdGhpcy5nbC5nZXRFeHRlbnNpb24oQ09MT1JfQlVGRkVSX0ZMT0FUKTtcbiAgICAgIH0gZWxzZSBpZiAod2ViZ2xfdXRpbC5oYXNFeHRlbnNpb24odGhpcy5nbCwgQ09MT1JfQlVGRkVSX0hBTEZfRkxPQVQpKSB7XG4gICAgICAgIHRoaXMuY29sb3JCdWZmZXJIYWxmRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgICAgdGhpcy5nbC5nZXRFeHRlbnNpb24oQ09MT1JfQlVGRkVSX0hBTEZfRkxPQVQpO1xuICAgICAgfSBlbHNlIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCdHTCBjb250ZXh0IGRvZXMgbm90IHN1cHBvcnQgY29sb3IgcmVuZGVyYWJsZSBmbG9hdHMnKTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB0aGlzLnZlcnRleEJ1ZmZlciA9IGdwZ3B1X3V0aWwuY3JlYXRlVmVydGV4QnVmZmVyKHRoaXMuZ2wpO1xuICAgIHRoaXMuaW5kZXhCdWZmZXIgPSBncGdwdV91dGlsLmNyZWF0ZUluZGV4QnVmZmVyKHRoaXMuZ2wpO1xuICAgIHRoaXMuZnJhbWVidWZmZXIgPSB3ZWJnbF91dGlsLmNyZWF0ZUZyYW1lYnVmZmVyKHRoaXMuZ2wpO1xuXG4gICAgdGhpcy50ZXh0dXJlQ29uZmlnID1cbiAgICAgICAgdGV4X3V0aWwuZ2V0VGV4dHVyZUNvbmZpZyh0aGlzLmdsLCB0aGlzLnRleHR1cmVIYWxmRmxvYXRFeHRlbnNpb24pO1xuICB9XG5cbiAgcHJpdmF0ZSBnZXQgZGVidWcoKTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIGVudigpLmdldEJvb2woJ0RFQlVHJyk7XG4gIH1cblxuICBwdWJsaWMgZGlzcG9zZSgpIHtcbiAgICBpZiAodGhpcy5kaXNwb3NlZCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBpZiAodGhpcy5wcm9ncmFtICE9IG51bGwpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAnRGlzcG9zaW5nIGEgR1BHUFVDb250ZXh0IHRoYXQgc3RpbGwgaGFzIGEgYm91bmQgV2ViR0xQcm9ncmFtLicgK1xuICAgICAgICAgICcgVGhpcyBpcyBwcm9iYWJseSBhIHJlc291cmNlIGxlYWssIGRlbGV0ZSB0aGUgcHJvZ3JhbSB3aXRoICcgK1xuICAgICAgICAgICdHUEdQVUNvbnRleHQuZGVsZXRlUHJvZ3JhbSBiZWZvcmUgZGlzcG9zaW5nLicpO1xuICAgIH1cbiAgICBpZiAodGhpcy5vdXRwdXRUZXh0dXJlICE9IG51bGwpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAnRGlzcG9zaW5nIGEgR1BHUFVDb250ZXh0IHRoYXQgc3RpbGwgaGFzIGEgYm91bmQgb3V0cHV0IG1hdHJpeCAnICtcbiAgICAgICAgICAndGV4dHVyZS4gIFRoaXMgaXMgcHJvYmFibHkgYSByZXNvdXJjZSBsZWFrLCBkZWxldGUgdGhlIG91dHB1dCAnICtcbiAgICAgICAgICAnbWF0cml4IHRleHR1cmUgd2l0aCBHUEdQVUNvbnRleHQuZGVsZXRlTWF0cml4VGV4dHVyZSBiZWZvcmUgJyArXG4gICAgICAgICAgJ2Rpc3Bvc2luZy4nKTtcbiAgICB9XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5maW5pc2goKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgbnVsbCkpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kZWxldGVGcmFtZWJ1ZmZlcih0aGlzLmZyYW1lYnVmZmVyKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuQVJSQVlfQlVGRkVSLCBudWxsKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkVMRU1FTlRfQVJSQVlfQlVGRkVSLCBudWxsKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRlbGV0ZUJ1ZmZlcih0aGlzLmluZGV4QnVmZmVyKSk7XG4gICAgdGhpcy5kaXNwb3NlZCA9IHRydWU7XG4gIH1cblxuICBwdWJsaWMgY3JlYXRlRmxvYXQzMk1hdHJpeFRleHR1cmUocm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBUZXh0dXJlIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiBncGdwdV91dGlsLmNyZWF0ZUZsb2F0MzJNYXRyaXhUZXh0dXJlKFxuICAgICAgICB0aGlzLmdsLCByb3dzLCBjb2x1bW5zLCB0aGlzLnRleHR1cmVDb25maWcpO1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZUZsb2F0MTZNYXRyaXhUZXh0dXJlKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogVGV4dHVyZSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICByZXR1cm4gZ3BncHVfdXRpbC5jcmVhdGVGbG9hdDE2TWF0cml4VGV4dHVyZShcbiAgICAgICAgdGhpcy5nbCwgcm93cywgY29sdW1ucywgdGhpcy50ZXh0dXJlQ29uZmlnKTtcbiAgfVxuXG4gIHB1YmxpYyBjcmVhdGVVbnNpZ25lZEJ5dGVzTWF0cml4VGV4dHVyZShyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6XG4gICAgICBUZXh0dXJlIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiBncGdwdV91dGlsLmNyZWF0ZVVuc2lnbmVkQnl0ZXNNYXRyaXhUZXh0dXJlKFxuICAgICAgICB0aGlzLmdsLCByb3dzLCBjb2x1bW5zLCB0aGlzLnRleHR1cmVDb25maWcpO1xuICB9XG5cbiAgcHVibGljIHVwbG9hZFBpeGVsRGF0YVRvVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSxcbiAgICAgIHBpeGVsczogUGl4ZWxEYXRhfEltYWdlRGF0YXxIVE1MSW1hZ2VFbGVtZW50fEhUTUxDYW52YXNFbGVtZW50fFxuICAgICAgSW1hZ2VCaXRtYXApIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGdwZ3B1X3V0aWwudXBsb2FkUGl4ZWxEYXRhVG9UZXh0dXJlKHRoaXMuZ2wsIHRleHR1cmUsIHBpeGVscyk7XG4gIH1cblxuICBwdWJsaWMgdXBsb2FkRGVuc2VNYXRyaXhUb1RleHR1cmUoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBkYXRhOiBUeXBlZEFycmF5KSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBncGdwdV91dGlsLnVwbG9hZERlbnNlTWF0cml4VG9UZXh0dXJlKFxuICAgICAgICB0aGlzLmdsLCB0ZXh0dXJlLCB3aWR0aCwgaGVpZ2h0LCBkYXRhLCB0aGlzLnRleHR1cmVDb25maWcpO1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZUZsb2F0MTZQYWNrZWRNYXRyaXhUZXh0dXJlKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTpcbiAgICAgIFRleHR1cmUge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuY3JlYXRlRmxvYXQxNlBhY2tlZE1hdHJpeFRleHR1cmUoXG4gICAgICAgIHRoaXMuZ2wsIHJvd3MsIGNvbHVtbnMsIHRoaXMudGV4dHVyZUNvbmZpZyk7XG4gIH1cblxuICBwdWJsaWMgY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFRleHR1cmUge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShcbiAgICAgICAgdGhpcy5nbCwgcm93cywgY29sdW1ucywgdGhpcy50ZXh0dXJlQ29uZmlnKTtcbiAgfVxuXG4gIHB1YmxpYyBkZWxldGVNYXRyaXhUZXh0dXJlKHRleHR1cmU6IFdlYkdMVGV4dHVyZSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgaWYgKHRoaXMub3V0cHV0VGV4dHVyZSA9PT0gdGV4dHVyZSkge1xuICAgICAgd2ViZ2xfdXRpbC51bmJpbmRDb2xvclRleHR1cmVGcm9tRnJhbWVidWZmZXIodGhpcy5nbCwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgICB0aGlzLm91dHB1dFRleHR1cmUgPSBudWxsO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLmRlbGV0ZVRleHR1cmUodGV4dHVyZSkpO1xuICB9XG5cbiAgcHVibGljIGRvd25sb2FkQnl0ZUVuY29kZWRGbG9hdE1hdHJpeEZyb21PdXRwdXRUZXh0dXJlKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gICAgcmV0dXJuIHRoaXMuZG93bmxvYWRNYXRyaXhEcml2ZXIoXG4gICAgICAgIHRleHR1cmUsXG4gICAgICAgICgpID0+IGdwZ3B1X3V0aWwuZG93bmxvYWRCeXRlRW5jb2RlZEZsb2F0TWF0cml4RnJvbU91dHB1dFRleHR1cmUoXG4gICAgICAgICAgICB0aGlzLmdsLCByb3dzLCBjb2x1bW5zLCB0aGlzLnRleHR1cmVDb25maWcpKTtcbiAgfVxuXG4gIHB1YmxpYyBkb3dubG9hZFBhY2tlZE1hdHJpeEZyb21CdWZmZXIoXG4gICAgICBidWZmZXI6IFdlYkdMQnVmZmVyLCBiYXRjaDogbnVtYmVyLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcixcbiAgICAgIHBoeXNpY2FsUm93czogbnVtYmVyLCBwaHlzaWNhbENvbHM6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuZG93bmxvYWRQYWNrZWRNYXRyaXhGcm9tQnVmZmVyKFxuICAgICAgICB0aGlzLmdsLCBidWZmZXIsIGJhdGNoLCByb3dzLCBjb2x1bW5zLCBwaHlzaWNhbFJvd3MsIHBoeXNpY2FsQ29scyxcbiAgICAgICAgdGhpcy50ZXh0dXJlQ29uZmlnKTtcbiAgfVxuXG4gIHB1YmxpYyBkb3dubG9hZEZsb2F0MzJNYXRyaXhGcm9tQnVmZmVyKGJ1ZmZlcjogV2ViR0xCdWZmZXIsIHNpemU6IG51bWJlcik6XG4gICAgICBGbG9hdDMyQXJyYXkge1xuICAgIHJldHVybiBncGdwdV91dGlsLmRvd25sb2FkRmxvYXQzMk1hdHJpeEZyb21CdWZmZXIodGhpcy5nbCwgYnVmZmVyLCBzaXplKTtcbiAgfVxuXG4gIHB1YmxpYyBjcmVhdGVCdWZmZXJGcm9tVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBXZWJHTEJ1ZmZlciB7XG4gICAgdGhpcy5iaW5kVGV4dHVyZVRvRnJhbWVCdWZmZXIodGV4dHVyZSk7XG4gICAgY29uc3QgcmVzdWx0ID0gZ3BncHVfdXRpbC5jcmVhdGVCdWZmZXJGcm9tT3V0cHV0VGV4dHVyZShcbiAgICAgICAgdGhpcy5nbCBhcyBXZWJHTDJSZW5kZXJpbmdDb250ZXh0LCByb3dzLCBjb2x1bW5zLCB0aGlzLnRleHR1cmVDb25maWcpO1xuICAgIHRoaXMudW5iaW5kVGV4dHVyZVRvRnJhbWVCdWZmZXIoKTtcbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZUFuZFdhaXRGb3JGZW5jZSgpOiBQcm9taXNlPHZvaWQ+IHtcbiAgICBjb25zdCBmZW5jZUNvbnRleHQgPSB0aGlzLmNyZWF0ZUZlbmNlKHRoaXMuZ2wpO1xuICAgIHJldHVybiB0aGlzLnBvbGxGZW5jZShmZW5jZUNvbnRleHQpO1xuICB9XG5cbiAgcHJpdmF0ZSBjcmVhdGVGZW5jZShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogRmVuY2VDb250ZXh0IHtcbiAgICBsZXQgcXVlcnk6IFdlYkdMUXVlcnl8V2ViR0xTeW5jO1xuICAgIGxldCBpc0ZlbmNlUGFzc2VkOiAoKSA9PiBib29sZWFuO1xuXG4gICAgaWYgKGVudigpLmdldEJvb2woJ1dFQkdMX0ZFTkNFX0FQSV9FTkFCTEVEJykpIHtcbiAgICAgIGNvbnN0IGdsMiA9IGdsIGFzIFdlYkdMMlJlbmRlcmluZ0NvbnRleHQ7XG5cbiAgICAgIGNvbnN0IHN5bmMgPSBnbDIuZmVuY2VTeW5jKGdsMi5TWU5DX0dQVV9DT01NQU5EU19DT01QTEVURSwgMCk7XG4gICAgICBnbC5mbHVzaCgpO1xuXG4gICAgICBpc0ZlbmNlUGFzc2VkID0gKCkgPT4ge1xuICAgICAgICBjb25zdCBzdGF0dXMgPSBnbDIuY2xpZW50V2FpdFN5bmMoc3luYywgMCwgMCk7XG4gICAgICAgIHJldHVybiBzdGF0dXMgPT09IGdsMi5BTFJFQURZX1NJR05BTEVEIHx8XG4gICAgICAgICAgICBzdGF0dXMgPT09IGdsMi5DT05ESVRJT05fU0FUSVNGSUVEO1xuICAgICAgfTtcblxuICAgICAgcXVlcnkgPSBzeW5jO1xuICAgIH0gZWxzZSBpZiAoXG4gICAgICAgIGVudigpLmdldE51bWJlcignV0VCR0xfRElTSk9JTlRfUVVFUllfVElNRVJfRVhURU5TSU9OX1ZFUlNJT04nKSA+IDApIHtcbiAgICAgIHF1ZXJ5ID0gdGhpcy5iZWdpblF1ZXJ5KCk7XG4gICAgICB0aGlzLmVuZFF1ZXJ5KCk7XG4gICAgICBpc0ZlbmNlUGFzc2VkID0gKCkgPT4gdGhpcy5pc1F1ZXJ5QXZhaWxhYmxlKFxuICAgICAgICAgIHF1ZXJ5LFxuICAgICAgICAgIGVudigpLmdldE51bWJlcignV0VCR0xfRElTSk9JTlRfUVVFUllfVElNRVJfRVhURU5TSU9OX1ZFUlNJT04nKSk7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIElmIHdlIGhhdmUgbm8gd2F5IHRvIGZlbmNlLCByZXR1cm4gdHJ1ZSBpbW1lZGlhdGVseS4gVGhpcyB3aWxsIGZpcmUgaW5cbiAgICAgIC8vIFdlYkdMIDEuMCB3aGVuIHRoZXJlIGlzIG5vIGRpc2pvaW50IHF1ZXJ5IHRpbWVyLiBJbiB0aGlzIGNhc2UsIGJlY2F1c2VcbiAgICAgIC8vIHRoZSBmZW5jZSBwYXNzZXMgaW1tZWRpYXRlbHksIHdlJ2xsIGltbWVkaWF0ZWx5IGFzayBmb3IgYSBkb3dubG9hZCBvZlxuICAgICAgLy8gdGhlIHRleHR1cmUsIHdoaWNoIHdpbGwgY2F1c2UgdGhlIFVJIHRocmVhZCB0byBoYW5nLlxuICAgICAgaXNGZW5jZVBhc3NlZCA9ICgpID0+IHRydWU7XG4gICAgfVxuXG4gICAgcmV0dXJuIHtxdWVyeSwgaXNGZW5jZVBhc3NlZH07XG4gIH1cblxuICBwdWJsaWMgZG93bmxvYWRNYXRyaXhGcm9tUGFja2VkVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcGh5c2ljYWxSb3dzOiBudW1iZXIsXG4gICAgICBwaHlzaWNhbENvbHM6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gICAgcmV0dXJuIHRoaXMuZG93bmxvYWRNYXRyaXhEcml2ZXIoXG4gICAgICAgIHRleHR1cmUsXG4gICAgICAgICgpID0+IGdwZ3B1X3V0aWwuZG93bmxvYWRNYXRyaXhGcm9tUGFja2VkT3V0cHV0VGV4dHVyZShcbiAgICAgICAgICAgIHRoaXMuZ2wsIHBoeXNpY2FsUm93cywgcGh5c2ljYWxDb2xzKSk7XG4gIH1cblxuICBwcml2YXRlIHZlcnRleEF0dHJzQXJlQm91bmQgPSBmYWxzZTtcblxuICBwdWJsaWMgY3JlYXRlUHJvZ3JhbShmcmFnbWVudFNoYWRlcjogV2ViR0xTaGFkZXIpOiBXZWJHTFByb2dyYW0ge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIGlmICh0aGlzLnZlcnRleFNoYWRlciA9PSBudWxsKSB7XG4gICAgICB0aGlzLnZlcnRleFNoYWRlciA9IGdwZ3B1X3V0aWwuY3JlYXRlVmVydGV4U2hhZGVyKGdsKTtcbiAgICB9XG4gICAgY29uc3QgcHJvZ3JhbTogV2ViR0xQcm9ncmFtID0gd2ViZ2xfdXRpbC5jcmVhdGVQcm9ncmFtKGdsKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgICAgZ2wsICgpID0+IGdsLmF0dGFjaFNoYWRlcihwcm9ncmFtLCB0aGlzLnZlcnRleFNoYWRlcikpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5hdHRhY2hTaGFkZXIocHJvZ3JhbSwgZnJhZ21lbnRTaGFkZXIpKTtcbiAgICB3ZWJnbF91dGlsLmxpbmtQcm9ncmFtKGdsLCBwcm9ncmFtKTtcbiAgICBpZiAodGhpcy5kZWJ1Zykge1xuICAgICAgd2ViZ2xfdXRpbC52YWxpZGF0ZVByb2dyYW0oZ2wsIHByb2dyYW0pO1xuICAgIH1cbiAgICBpZiAoIXRoaXMudmVydGV4QXR0cnNBcmVCb3VuZCkge1xuICAgICAgdGhpcy5zZXRQcm9ncmFtKHByb2dyYW0pO1xuICAgICAgdGhpcy52ZXJ0ZXhBdHRyc0FyZUJvdW5kID0gZ3BncHVfdXRpbC5iaW5kVmVydGV4UHJvZ3JhbUF0dHJpYnV0ZVN0cmVhbXMoXG4gICAgICAgICAgZ2wsIHRoaXMucHJvZ3JhbSwgdGhpcy52ZXJ0ZXhCdWZmZXIpO1xuICAgIH1cbiAgICByZXR1cm4gcHJvZ3JhbTtcbiAgfVxuXG4gIHB1YmxpYyBkZWxldGVQcm9ncmFtKHByb2dyYW06IFdlYkdMUHJvZ3JhbSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgaWYgKHByb2dyYW0gPT09IHRoaXMucHJvZ3JhbSkge1xuICAgICAgdGhpcy5wcm9ncmFtID0gbnVsbDtcbiAgICB9XG4gICAgaWYgKHByb2dyYW0gIT0gbnVsbCkge1xuICAgICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2sodGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5kZWxldGVQcm9ncmFtKHByb2dyYW0pKTtcbiAgICB9XG4gIH1cblxuICBwdWJsaWMgc2V0UHJvZ3JhbShwcm9ncmFtOiBXZWJHTFByb2dyYW18bnVsbCkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy5wcm9ncmFtID0gcHJvZ3JhbTtcbiAgICBpZiAoKHRoaXMucHJvZ3JhbSAhPSBudWxsKSAmJiB0aGlzLmRlYnVnKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlUHJvZ3JhbSh0aGlzLmdsLCB0aGlzLnByb2dyYW0pO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLnVzZVByb2dyYW0ocHJvZ3JhbSkpO1xuICB9XG5cbiAgcHVibGljIGdldFVuaWZvcm1Mb2NhdGlvbihcbiAgICAgIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgdW5pZm9ybU5hbWU6IHN0cmluZyxcbiAgICAgIHNob3VsZFRocm93ID0gdHJ1ZSk6IFdlYkdMVW5pZm9ybUxvY2F0aW9uIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGlmIChzaG91bGRUaHJvdykge1xuICAgICAgcmV0dXJuIHdlYmdsX3V0aWwuZ2V0UHJvZ3JhbVVuaWZvcm1Mb2NhdGlvbk9yVGhyb3coXG4gICAgICAgICAgdGhpcy5nbCwgcHJvZ3JhbSwgdW5pZm9ybU5hbWUpO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gd2ViZ2xfdXRpbC5nZXRQcm9ncmFtVW5pZm9ybUxvY2F0aW9uKFxuICAgICAgICAgIHRoaXMuZ2wsIHByb2dyYW0sIHVuaWZvcm1OYW1lKTtcbiAgICB9XG4gIH1cblxuICBwdWJsaWMgZ2V0QXR0cmlidXRlTG9jYXRpb24ocHJvZ3JhbTogV2ViR0xQcm9ncmFtLCBhdHRyaWJ1dGU6IHN0cmluZyk6XG4gICAgICBudW1iZXIge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgICB0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLmdldEF0dHJpYkxvY2F0aW9uKHByb2dyYW0sIGF0dHJpYnV0ZSkpO1xuICB9XG5cbiAgcHVibGljIGdldFVuaWZvcm1Mb2NhdGlvbk5vVGhyb3cocHJvZ3JhbTogV2ViR0xQcm9ncmFtLCB1bmlmb3JtTmFtZTogc3RyaW5nKTpcbiAgICAgIFdlYkdMVW5pZm9ybUxvY2F0aW9uIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiB0aGlzLmdsLmdldFVuaWZvcm1Mb2NhdGlvbihwcm9ncmFtLCB1bmlmb3JtTmFtZSk7XG4gIH1cblxuICBwdWJsaWMgc2V0SW5wdXRNYXRyaXhUZXh0dXJlKFxuICAgICAgaW5wdXRNYXRyaXhUZXh0dXJlOiBXZWJHTFRleHR1cmUsIHVuaWZvcm1Mb2NhdGlvbjogV2ViR0xVbmlmb3JtTG9jYXRpb24sXG4gICAgICB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB0aGlzLnRocm93SWZOb1Byb2dyYW0oKTtcbiAgICB3ZWJnbF91dGlsLmJpbmRUZXh0dXJlVG9Qcm9ncmFtVW5pZm9ybVNhbXBsZXIoXG4gICAgICAgIHRoaXMuZ2wsIGlucHV0TWF0cml4VGV4dHVyZSwgdW5pZm9ybUxvY2F0aW9uLCB0ZXh0dXJlVW5pdCk7XG4gIH1cblxuICBwdWJsaWMgc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIG91dHB1dE1hdHJpeFRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpIHtcbiAgICB0aGlzLnNldE91dHB1dE1hdHJpeFRleHR1cmVEcml2ZXIob3V0cHV0TWF0cml4VGV4dHVyZSwgY29sdW1ucywgcm93cyk7XG4gIH1cblxuICBwdWJsaWMgc2V0T3V0cHV0UGFja2VkTWF0cml4VGV4dHVyZShcbiAgICAgIG91dHB1dFBhY2tlZE1hdHJpeFRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IFt3aWR0aCwgaGVpZ2h0XSA9XG4gICAgICAgIHRleF91dGlsLmdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICAgIHRoaXMuc2V0T3V0cHV0TWF0cml4VGV4dHVyZURyaXZlcihvdXRwdXRQYWNrZWRNYXRyaXhUZXh0dXJlLCB3aWR0aCwgaGVpZ2h0KTtcbiAgfVxuXG4gIHB1YmxpYyBzZXRPdXRwdXRNYXRyaXhXcml0ZVJlZ2lvbihcbiAgICAgIHN0YXJ0Um93OiBudW1iZXIsIG51bVJvd3M6IG51bWJlciwgc3RhcnRDb2x1bW46IG51bWJlcixcbiAgICAgIG51bUNvbHVtbnM6IG51bWJlcikge1xuICAgIHRoaXMuc2V0T3V0cHV0TWF0cml4V3JpdGVSZWdpb25Ecml2ZXIoXG4gICAgICAgIHN0YXJ0Q29sdW1uLCBzdGFydFJvdywgbnVtQ29sdW1ucywgbnVtUm93cyk7XG4gIH1cblxuICBwdWJsaWMgc2V0T3V0cHV0UGFja2VkTWF0cml4V3JpdGVSZWdpb24oXG4gICAgICBzdGFydFJvdzogbnVtYmVyLCBudW1Sb3dzOiBudW1iZXIsIHN0YXJ0Q29sdW1uOiBudW1iZXIsXG4gICAgICBudW1Db2x1bW5zOiBudW1iZXIpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ3NldE91dHB1dFBhY2tlZE1hdHJpeFdyaXRlUmVnaW9uIG5vdCBpbXBsZW1lbnRlZC4nKTtcbiAgfVxuXG4gIHB1YmxpYyBkZWJ1Z1ZhbGlkYXRlKCkge1xuICAgIGlmICh0aGlzLnByb2dyYW0gIT0gbnVsbCkge1xuICAgICAgd2ViZ2xfdXRpbC52YWxpZGF0ZVByb2dyYW0odGhpcy5nbCwgdGhpcy5wcm9ncmFtKTtcbiAgICB9XG4gICAgd2ViZ2xfdXRpbC52YWxpZGF0ZUZyYW1lYnVmZmVyKHRoaXMuZ2wpO1xuICB9XG5cbiAgcHVibGljIGV4ZWN1dGVQcm9ncmFtKCkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy50aHJvd0lmTm9Qcm9ncmFtKCk7XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIGlmICh0aGlzLmRlYnVnKSB7XG4gICAgICB0aGlzLmRlYnVnVmFsaWRhdGUoKTtcbiAgICB9XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIGdsLCAoKSA9PiBnbC5kcmF3RWxlbWVudHMoZ2wuVFJJQU5HTEVTLCA2LCBnbC5VTlNJR05FRF9TSE9SVCwgMCkpO1xuICB9XG5cbiAgcHVibGljIGJsb2NrVW50aWxBbGxQcm9ncmFtc0NvbXBsZXRlZCgpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKHRoaXMuZ2wsICgpID0+IHRoaXMuZ2wuZmluaXNoKCkpO1xuICB9XG5cbiAgcHJpdmF0ZSBnZXRRdWVyeVRpbWVyRXh0ZW5zaW9uKCk6IFdlYkdMMURpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvblxuICAgICAgfFdlYkdMMkRpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvbiB7XG4gICAgaWYgKHRoaXMuZGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9uID09IG51bGwpIHtcbiAgICAgIHRoaXMuZGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9uID1cbiAgICAgICAgICB3ZWJnbF91dGlsLmdldEV4dGVuc2lvbk9yVGhyb3coXG4gICAgICAgICAgICAgIHRoaXMuZ2wsXG4gICAgICAgICAgICAgIGVudigpLmdldE51bWJlcihcbiAgICAgICAgICAgICAgICAgICdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fVkVSU0lPTicpID09PSAyID9cbiAgICAgICAgICAgICAgICAgICdFWFRfZGlzam9pbnRfdGltZXJfcXVlcnlfd2ViZ2wyJyA6XG4gICAgICAgICAgICAgICAgICAnRVhUX2Rpc2pvaW50X3RpbWVyX3F1ZXJ5JykgYXNcbiAgICAgICAgICAgICAgV2ViR0wxRGlzam9pbnRRdWVyeVRpbWVyRXh0ZW5zaW9uIHxcbiAgICAgICAgICBXZWJHTDJEaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb247XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmRpc2pvaW50UXVlcnlUaW1lckV4dGVuc2lvbjtcbiAgfVxuXG4gIHByaXZhdGUgZ2V0UXVlcnlUaW1lckV4dGVuc2lvbldlYkdMMigpOiBXZWJHTDJEaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb24ge1xuICAgIHJldHVybiB0aGlzLmdldFF1ZXJ5VGltZXJFeHRlbnNpb24oKTtcbiAgfVxuXG4gIHByaXZhdGUgZ2V0UXVlcnlUaW1lckV4dGVuc2lvbldlYkdMMSgpOiBXZWJHTDFEaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb24ge1xuICAgIHJldHVybiB0aGlzLmdldFF1ZXJ5VGltZXJFeHRlbnNpb24oKSBhcyBXZWJHTDFEaXNqb2ludFF1ZXJ5VGltZXJFeHRlbnNpb247XG4gIH1cblxuICBiZWdpblF1ZXJ5KCk6IFdlYkdMUXVlcnkge1xuICAgIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX0RJU0pPSU5UX1FVRVJZX1RJTUVSX0VYVEVOU0lPTl9WRVJTSU9OJykgPT09IDIpIHtcbiAgICAgIGNvbnN0IGdsMiA9IHRoaXMuZ2wgYXMgV2ViR0wyUmVuZGVyaW5nQ29udGV4dDtcbiAgICAgIGNvbnN0IGV4dCA9IHRoaXMuZ2V0UXVlcnlUaW1lckV4dGVuc2lvbldlYkdMMigpO1xuXG4gICAgICBjb25zdCBxdWVyeSA9IGdsMi5jcmVhdGVRdWVyeSgpO1xuICAgICAgZ2wyLmJlZ2luUXVlcnkoZXh0LlRJTUVfRUxBUFNFRF9FWFQsIHF1ZXJ5KTtcbiAgICAgIHJldHVybiBxdWVyeTtcbiAgICB9XG4gICAgY29uc3QgZXh0ID0gdGhpcy5nZXRRdWVyeVRpbWVyRXh0ZW5zaW9uV2ViR0wxKCk7XG4gICAgY29uc3QgcXVlcnkgPSBleHQuY3JlYXRlUXVlcnlFWFQoKSBhcyBXZWJHTFF1ZXJ5O1xuICAgIGV4dC5iZWdpblF1ZXJ5RVhUKGV4dC5USU1FX0VMQVBTRURfRVhULCBxdWVyeSk7XG4gICAgcmV0dXJuIHF1ZXJ5O1xuICB9XG5cbiAgZW5kUXVlcnkoKSB7XG4gICAgaWYgKGVudigpLmdldE51bWJlcignV0VCR0xfRElTSk9JTlRfUVVFUllfVElNRVJfRVhURU5TSU9OX1ZFUlNJT04nKSA9PT0gMikge1xuICAgICAgY29uc3QgZ2wyID0gdGhpcy5nbCBhcyBXZWJHTDJSZW5kZXJpbmdDb250ZXh0O1xuICAgICAgY29uc3QgZXh0ID0gdGhpcy5nZXRRdWVyeVRpbWVyRXh0ZW5zaW9uV2ViR0wyKCk7XG4gICAgICBnbDIuZW5kUXVlcnkoZXh0LlRJTUVfRUxBUFNFRF9FWFQpO1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCBleHQgPSB0aGlzLmdldFF1ZXJ5VGltZXJFeHRlbnNpb25XZWJHTDEoKTtcbiAgICBleHQuZW5kUXVlcnlFWFQoZXh0LlRJTUVfRUxBUFNFRF9FWFQpO1xuICB9XG5cbiAgcHVibGljIGFzeW5jIHdhaXRGb3JRdWVyeUFuZEdldFRpbWUocXVlcnk6IFdlYkdMUXVlcnkpOiBQcm9taXNlPG51bWJlcj4ge1xuICAgIGF3YWl0IHV0aWwucmVwZWF0ZWRUcnkoXG4gICAgICAgICgpID0+IHRoaXMuZGlzcG9zZWQgfHwgIC8vIHdoaWxlIHRlc3RpbmcgY29udGV4dHMgYXJlIGNyZWF0ZWQgLyBkaXNwb3NlZFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAvLyBpbiByYXBpZCBzdWNjZXNzaW9uLCBzbyB3aXRob3V0IHRoaXMgY2hlY2sgd2VcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgLy8gbWF5IHBvbGwgZm9yIHRoZSBxdWVyeSB0aW1lciBpbmRlZmluaXRlbHlcbiAgICAgICAgICAgIHRoaXMuaXNRdWVyeUF2YWlsYWJsZShcbiAgICAgICAgICAgICAgICBxdWVyeSxcbiAgICAgICAgICAgICAgICBlbnYoKS5nZXROdW1iZXIoXG4gICAgICAgICAgICAgICAgICAgICdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fVkVSU0lPTicpKSk7XG4gICAgcmV0dXJuIHRoaXMuZ2V0UXVlcnlUaW1lKFxuICAgICAgICBxdWVyeSwgZW52KCkuZ2V0TnVtYmVyKCdXRUJHTF9ESVNKT0lOVF9RVUVSWV9USU1FUl9FWFRFTlNJT05fVkVSU0lPTicpKTtcbiAgfVxuXG4gIHByaXZhdGUgZ2V0UXVlcnlUaW1lKHF1ZXJ5OiBXZWJHTFF1ZXJ5LCBxdWVyeVRpbWVyVmVyc2lvbjogbnVtYmVyKTogbnVtYmVyIHtcbiAgICBpZiAocXVlcnlUaW1lclZlcnNpb24gPT09IDApIHtcbiAgICAgIHJldHVybiBudWxsO1xuICAgIH1cblxuICAgIGlmIChxdWVyeVRpbWVyVmVyc2lvbiA9PT0gMikge1xuICAgICAgY29uc3QgZ2wyID0gdGhpcy5nbCBhcyBXZWJHTDJSZW5kZXJpbmdDb250ZXh0O1xuXG4gICAgICBjb25zdCB0aW1lRWxhcHNlZE5hbm9zID0gZ2wyLmdldFF1ZXJ5UGFyYW1ldGVyKHF1ZXJ5LCBnbDIuUVVFUllfUkVTVUxUKTtcbiAgICAgIC8vIFJldHVybiBtaWxsaXNlY29uZHMuXG4gICAgICByZXR1cm4gdGltZUVsYXBzZWROYW5vcyAvIDEwMDAwMDA7XG4gICAgfSBlbHNlIHtcbiAgICAgIGNvbnN0IGV4dCA9IHRoaXMuZ2V0UXVlcnlUaW1lckV4dGVuc2lvbldlYkdMMSgpO1xuXG4gICAgICBjb25zdCB0aW1lRWxhcHNlZE5hbm9zID1cbiAgICAgICAgICBleHQuZ2V0UXVlcnlPYmplY3RFWFQocXVlcnksIGV4dC5RVUVSWV9SRVNVTFRfRVhUKTtcbiAgICAgIC8vIFJldHVybiBtaWxsaXNlY29uZHMuXG4gICAgICByZXR1cm4gdGltZUVsYXBzZWROYW5vcyAvIDEwMDAwMDA7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBpc1F1ZXJ5QXZhaWxhYmxlKHF1ZXJ5OiBXZWJHTFF1ZXJ5LCBxdWVyeVRpbWVyVmVyc2lvbjogbnVtYmVyKTpcbiAgICAgIGJvb2xlYW4ge1xuICAgIGlmIChxdWVyeVRpbWVyVmVyc2lvbiA9PT0gMCkge1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuXG4gICAgaWYgKHF1ZXJ5VGltZXJWZXJzaW9uID09PSAyKSB7XG4gICAgICBjb25zdCBnbDIgPSB0aGlzLmdsIGFzIFdlYkdMMlJlbmRlcmluZ0NvbnRleHQ7XG4gICAgICBjb25zdCBleHQgPSB0aGlzLmdldFF1ZXJ5VGltZXJFeHRlbnNpb25XZWJHTDIoKTtcblxuICAgICAgY29uc3QgYXZhaWxhYmxlID1cbiAgICAgICAgICBnbDIuZ2V0UXVlcnlQYXJhbWV0ZXIocXVlcnksIGdsMi5RVUVSWV9SRVNVTFRfQVZBSUxBQkxFKTtcbiAgICAgIGlmICh0aGlzLmRpc2pvaW50ID09IG51bGwpIHtcbiAgICAgICAgdGhpcy5kaXNqb2ludCA9IHRoaXMuZ2wuZ2V0UGFyYW1ldGVyKGV4dC5HUFVfRElTSk9JTlRfRVhUKTtcbiAgICAgIH1cblxuICAgICAgcmV0dXJuIGF2YWlsYWJsZSAmJiAhdGhpcy5kaXNqb2ludDtcbiAgICB9IGVsc2Uge1xuICAgICAgY29uc3QgZXh0ID0gdGhpcy5nZXRRdWVyeVRpbWVyRXh0ZW5zaW9uV2ViR0wxKCk7XG5cbiAgICAgIGNvbnN0IGF2YWlsYWJsZSA9XG4gICAgICAgICAgZXh0LmdldFF1ZXJ5T2JqZWN0RVhUKHF1ZXJ5LCBleHQuUVVFUllfUkVTVUxUX0FWQUlMQUJMRV9FWFQpO1xuICAgICAgaWYgKHRoaXMuZGlzam9pbnQgPT0gbnVsbCkge1xuICAgICAgICB0aGlzLmRpc2pvaW50ID0gdGhpcy5nbC5nZXRQYXJhbWV0ZXIoZXh0LkdQVV9ESVNKT0lOVF9FWFQpO1xuICAgICAgfVxuXG4gICAgICByZXR1cm4gYXZhaWxhYmxlICYmICF0aGlzLmRpc2pvaW50O1xuICAgIH1cbiAgfVxuXG4gIHBvbGxGZW5jZShmZW5jZUNvbnRleHQ6IEZlbmNlQ29udGV4dCkge1xuICAgIHJldHVybiBuZXcgUHJvbWlzZTx2b2lkPihyZXNvbHZlID0+IHtcbiAgICAgIHRoaXMuYWRkSXRlbVRvUG9sbCgoKSA9PiBmZW5jZUNvbnRleHQuaXNGZW5jZVBhc3NlZCgpLCAoKSA9PiByZXNvbHZlKCkpO1xuICAgIH0pO1xuICB9XG5cbiAgcHJpdmF0ZSBpdGVtc1RvUG9sbDogUG9sbEl0ZW1bXSA9IFtdO1xuXG4gIHBvbGxJdGVtcygpOiB2b2lkIHtcbiAgICAvLyBGaW5kIHRoZSBsYXN0IHF1ZXJ5IHRoYXQgaGFzIGZpbmlzaGVkLlxuICAgIGNvbnN0IGluZGV4ID0gbGluZWFyU2VhcmNoTGFzdFRydWUodGhpcy5pdGVtc1RvUG9sbC5tYXAoeCA9PiB4LmlzRG9uZUZuKSk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPD0gaW5kZXg7ICsraSkge1xuICAgICAgY29uc3Qge3Jlc29sdmVGbn0gPSB0aGlzLml0ZW1zVG9Qb2xsW2ldO1xuICAgICAgcmVzb2x2ZUZuKCk7XG4gICAgfVxuICAgIHRoaXMuaXRlbXNUb1BvbGwgPSB0aGlzLml0ZW1zVG9Qb2xsLnNsaWNlKGluZGV4ICsgMSk7XG4gIH1cblxuICBwcml2YXRlIGFkZEl0ZW1Ub1BvbGwoaXNEb25lRm46ICgpID0+IGJvb2xlYW4sIHJlc29sdmVGbjogKCkgPT4gdm9pZCkge1xuICAgIHRoaXMuaXRlbXNUb1BvbGwucHVzaCh7aXNEb25lRm4sIHJlc29sdmVGbn0pO1xuICAgIGlmICh0aGlzLml0ZW1zVG9Qb2xsLmxlbmd0aCA+IDEpIHtcbiAgICAgIC8vIFdlIGFscmVhZHkgaGF2ZSBhIHJ1bm5pbmcgbG9vcCB0aGF0IHBvbGxzLlxuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICAvLyBTdGFydCBhIG5ldyBsb29wIHRoYXQgcG9sbHMuXG4gICAgdXRpbC5yZXBlYXRlZFRyeSgoKSA9PiB7XG4gICAgICB0aGlzLnBvbGxJdGVtcygpO1xuICAgICAgLy8gRW5kIHRoZSBsb29wIGlmIG5vIG1vcmUgaXRlbXMgdG8gcG9sbC5cbiAgICAgIHJldHVybiB0aGlzLml0ZW1zVG9Qb2xsLmxlbmd0aCA9PT0gMDtcbiAgICB9KTtcbiAgfVxuXG4gIHByaXZhdGUgYmluZFRleHR1cmVUb0ZyYW1lQnVmZmVyKHRleHR1cmU6IFdlYkdMVGV4dHVyZSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgd2ViZ2xfdXRpbC5iaW5kQ29sb3JUZXh0dXJlVG9GcmFtZWJ1ZmZlcihcbiAgICAgICAgdGhpcy5nbCwgdGV4dHVyZSwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgaWYgKHRoaXMuZGVidWcpIHtcbiAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVGcmFtZWJ1ZmZlcih0aGlzLmdsKTtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIHVuYmluZFRleHR1cmVUb0ZyYW1lQnVmZmVyKCkge1xuICAgIGlmICh0aGlzLm91dHB1dFRleHR1cmUgIT0gbnVsbCkge1xuICAgICAgd2ViZ2xfdXRpbC5iaW5kQ29sb3JUZXh0dXJlVG9GcmFtZWJ1ZmZlcihcbiAgICAgICAgICB0aGlzLmdsLCB0aGlzLm91dHB1dFRleHR1cmUsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgICAgaWYgKHRoaXMuZGVidWcpIHtcbiAgICAgICAgd2ViZ2xfdXRpbC52YWxpZGF0ZUZyYW1lYnVmZmVyKHRoaXMuZ2wpO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICB3ZWJnbF91dGlsLnVuYmluZENvbG9yVGV4dHVyZUZyb21GcmFtZWJ1ZmZlcih0aGlzLmdsLCB0aGlzLmZyYW1lYnVmZmVyKTtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIGRvd25sb2FkTWF0cml4RHJpdmVyKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgICAgZG93bmxvYWRBbmREZWNvZGU6ICgpID0+IEZsb2F0MzJBcnJheSk6IEZsb2F0MzJBcnJheSB7XG4gICAgdGhpcy5iaW5kVGV4dHVyZVRvRnJhbWVCdWZmZXIodGV4dHVyZSk7XG4gICAgY29uc3QgcmVzdWx0ID0gZG93bmxvYWRBbmREZWNvZGUoKTtcbiAgICB0aGlzLnVuYmluZFRleHR1cmVUb0ZyYW1lQnVmZmVyKCk7XG5cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJpdmF0ZSBzZXRPdXRwdXRNYXRyaXhUZXh0dXJlRHJpdmVyKFxuICAgICAgb3V0cHV0TWF0cml4VGV4dHVyZU1heWJlUGFja2VkOiBXZWJHTFRleHR1cmUsIHdpZHRoOiBudW1iZXIsXG4gICAgICBoZWlnaHQ6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIHdlYmdsX3V0aWwuYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgICAgIGdsLCBvdXRwdXRNYXRyaXhUZXh0dXJlTWF5YmVQYWNrZWQsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgIGlmICh0aGlzLmRlYnVnKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlRnJhbWVidWZmZXIoZ2wpO1xuICAgIH1cbiAgICB0aGlzLm91dHB1dFRleHR1cmUgPSBvdXRwdXRNYXRyaXhUZXh0dXJlTWF5YmVQYWNrZWQ7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnZpZXdwb3J0KDAsIDAsIHdpZHRoLCBoZWlnaHQpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuc2Npc3NvcigwLCAwLCB3aWR0aCwgaGVpZ2h0KSk7XG4gIH1cblxuICBwcml2YXRlIHNldE91dHB1dE1hdHJpeFdyaXRlUmVnaW9uRHJpdmVyKFxuICAgICAgeDogbnVtYmVyLCB5OiBudW1iZXIsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgICAgdGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5zY2lzc29yKHgsIHksIHdpZHRoLCBoZWlnaHQpKTtcbiAgfVxuXG4gIHByaXZhdGUgdGhyb3dJZkRpc3Bvc2VkKCkge1xuICAgIGlmICh0aGlzLmRpc3Bvc2VkKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0F0dGVtcHRlZCB0byB1c2UgZGlzcG9zZWQgR1BHUFVDb250ZXh0LicpO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgdGhyb3dJZk5vUHJvZ3JhbSgpIHtcbiAgICBpZiAodGhpcy5wcm9ncmFtID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignTm8gR1BVIHByb2dyYW0gaXMgY3VycmVudGx5IHNldC4nKTtcbiAgICB9XG4gIH1cbn1cblxudHlwZSBQb2xsSXRlbSA9IHtcbiAgaXNEb25lRm46ICgpID0+IGJvb2xlYW4sXG4gIHJlc29sdmVGbjogKCkgPT4gdm9pZFxufTtcblxuLyoqXG4gKiBGaW5kcyB0aGUgaW5kZXggb2YgdGhlIGxhc3QgdHJ1ZSBlbGVtZW50IHVzaW5nIGxpbmVhciBzZWFyY2guXG4gKiBOb3RlOiBXZSBjYW4ndCBkbyBiaW5hcnkgc2VhcmNoIGJlY2F1c2UgQ2hyb21lIGV4cGVjdHMgdXMgdG8gZXhwbGljaXRseVxuICogdGVzdCBhbGwgZmVuY2VzIGJlZm9yZSBkb3dubG9hZDpcbiAqIGh0dHBzOi8vZ2l0aHViLmNvbS90ZW5zb3JmbG93L3RmanMvaXNzdWVzLzExNDVcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGxpbmVhclNlYXJjaExhc3RUcnVlKGFycjogQXJyYXk8KCkgPT4gYm9vbGVhbj4pOiBudW1iZXIge1xuICBsZXQgaSA9IDA7XG4gIGZvciAoOyBpIDwgYXJyLmxlbmd0aDsgKytpKSB7XG4gICAgY29uc3QgaXNEb25lID0gYXJyW2ldKCk7XG4gICAgaWYgKCFpc0RvbmUpIHtcbiAgICAgIGJyZWFrO1xuICAgIH1cbiAgfVxuICByZXR1cm4gaSAtIDE7XG59XG4iXX0=