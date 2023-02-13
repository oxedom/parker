/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import { ENGINE } from '../engine';
import { env } from '../environment';
import { FromPixels } from '../kernel_names';
import { getKernel } from '../kernel_registry';
import { Tensor } from '../tensor';
import { convertToTensor } from '../tensor_util_env';
import { cast } from './cast';
import { op } from './operation';
import { tensor3d } from './tensor3d';
let fromPixels2DContext;
/**
 * Creates a `tf.Tensor` from an image.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * tf.browser.fromPixels(image).print();
 * ```
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8Array; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @returns A Tensor3D with the shape `[height, width, numChannels]`.
 *
 * Note: fromPixels can be lossy in some cases, same image may result in
 * slightly different tensor values, if rendered by different rendering
 * engines. This means that results from different browsers, or even same
 * browser with CPU and GPU rendering engines can be different. See discussion
 * in details:
 * https://github.com/tensorflow/tfjs/issues/5482
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
function fromPixels_(pixels, numChannels = 3) {
    // Sanity checks.
    if (numChannels > 4) {
        throw new Error('Cannot construct Tensor with more than 4 channels from pixels.');
    }
    if (pixels == null) {
        throw new Error('pixels passed to tf.browser.fromPixels() can not be null');
    }
    let isPixelData = false;
    let isImageData = false;
    let isVideo = false;
    let isImage = false;
    let isCanvasLike = false;
    let isImageBitmap = false;
    if (pixels.data instanceof Uint8Array) {
        isPixelData = true;
    }
    else if (typeof (ImageData) !== 'undefined' && pixels instanceof ImageData) {
        isImageData = true;
    }
    else if (typeof (HTMLVideoElement) !== 'undefined' &&
        pixels instanceof HTMLVideoElement) {
        isVideo = true;
    }
    else if (typeof (HTMLImageElement) !== 'undefined' &&
        pixels instanceof HTMLImageElement) {
        isImage = true;
        // tslint:disable-next-line: no-any
    }
    else if (pixels.getContext != null) {
        isCanvasLike = true;
    }
    else if (typeof (ImageBitmap) !== 'undefined' && pixels instanceof ImageBitmap) {
        isImageBitmap = true;
    }
    else {
        throw new Error('pixels passed to tf.browser.fromPixels() must be either an ' +
            `HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData ` +
            `in browser, or OffscreenCanvas, ImageData in webworker` +
            ` or {data: Uint32Array, width: number, height: number}, ` +
            `but was ${pixels.constructor.name}`);
    }
    if (isVideo) {
        const HAVE_CURRENT_DATA_READY_STATE = 2;
        if (isVideo &&
            pixels.readyState <
                HAVE_CURRENT_DATA_READY_STATE) {
            throw new Error('The video element has not loaded data yet. Please wait for ' +
                '`loadeddata` event on the <video> element.');
        }
    }
    // If the current backend has 'FromPixels' registered, it has a more
    // efficient way of handling pixel uploads, so we call that.
    const kernel = getKernel(FromPixels, ENGINE.backendName);
    if (kernel != null) {
        const inputs = { pixels };
        const attrs = { numChannels };
        return ENGINE.runKernel(FromPixels, inputs, attrs);
    }
    const [width, height] = isVideo ?
        [
            pixels.videoWidth,
            pixels.videoHeight
        ] :
        [pixels.width, pixels.height];
    let vals;
    if (isCanvasLike) {
        vals =
            // tslint:disable-next-line:no-any
            pixels.getContext('2d').getImageData(0, 0, width, height).data;
    }
    else if (isImageData || isPixelData) {
        vals = pixels.data;
    }
    else if (isImage || isVideo || isImageBitmap) {
        if (fromPixels2DContext == null) {
            if (typeof document === 'undefined') {
                if (typeof OffscreenCanvas !== 'undefined' &&
                    typeof OffscreenCanvasRenderingContext2D !== 'undefined') {
                    // @ts-ignore
                    fromPixels2DContext = new OffscreenCanvas(1, 1).getContext('2d');
                }
                else {
                    throw new Error('Cannot parse input in current context. ' +
                        'Reason: OffscreenCanvas Context2D rendering is not supported.');
                }
            }
            else {
                fromPixels2DContext = document.createElement('canvas').getContext('2d');
            }
        }
        fromPixels2DContext.canvas.width = width;
        fromPixels2DContext.canvas.height = height;
        fromPixels2DContext.drawImage(pixels, 0, 0, width, height);
        vals = fromPixels2DContext.getImageData(0, 0, width, height).data;
    }
    let values;
    if (numChannels === 4) {
        values = new Int32Array(vals);
    }
    else {
        const numPixels = width * height;
        values = new Int32Array(numPixels * numChannels);
        for (let i = 0; i < numPixels; i++) {
            for (let channel = 0; channel < numChannels; ++channel) {
                values[i * numChannels + channel] = vals[i * 4 + channel];
            }
        }
    }
    const outShape = [height, width, numChannels];
    return tensor3d(values, outShape, 'int32');
}
// Helper functions for |fromPixelsAsync| to check whether the input can
// be wrapped into imageBitmap.
function isPixelData(pixels) {
    return (pixels != null) && (pixels.data instanceof Uint8Array);
}
function isImageBitmapFullySupported() {
    return typeof window !== 'undefined' &&
        typeof (ImageBitmap) !== 'undefined' &&
        window.hasOwnProperty('createImageBitmap');
}
function isNonEmptyPixels(pixels) {
    return pixels != null && pixels.width !== 0 && pixels.height !== 0;
}
function canWrapPixelsToImageBitmap(pixels) {
    return isImageBitmapFullySupported() && !(pixels instanceof ImageBitmap) &&
        isNonEmptyPixels(pixels) && !isPixelData(pixels);
}
/**
 * Creates a `tf.Tensor` from an image in async way.
 *
 * ```js
 * const image = new ImageData(1, 1);
 * image.data[0] = 100;
 * image.data[1] = 150;
 * image.data[2] = 200;
 * image.data[3] = 255;
 *
 * (await tf.browser.fromPixelsAsync(image)).print();
 * ```
 * This API is the async version of fromPixels. The API will first
 * check |WRAP_TO_IMAGEBITMAP| flag, and try to wrap the input to
 * imageBitmap if the flag is set to true.
 *
 * @param pixels The input image to construct the tensor from. The
 * supported image types are all 4-channel. You can also pass in an image
 * object with following attributes:
 * `{data: Uint8Array; width: number; height: number}`
 * @param numChannels The number of channels of the output tensor. A
 * numChannels value less than 4 allows you to ignore channels. Defaults to
 * 3 (ignores alpha channel of input image).
 *
 * @doc {heading: 'Browser', namespace: 'browser', ignoreCI: true}
 */
export async function fromPixelsAsync(pixels, numChannels = 3) {
    let inputs = null;
    // Check whether the backend needs to wrap |pixels| to imageBitmap and
    // whether |pixels| can be wrapped to imageBitmap.
    if (env().getBool('WRAP_TO_IMAGEBITMAP') &&
        canWrapPixelsToImageBitmap(pixels)) {
        // Force the imageBitmap creation to not do any premultiply alpha
        // ops.
        let imageBitmap;
        try {
            // wrap in try-catch block, because createImageBitmap may not work
            // properly in some browsers, e.g.
            // https://bugzilla.mozilla.org/show_bug.cgi?id=1335594
            // tslint:disable-next-line: no-any
            imageBitmap = await createImageBitmap(pixels, { premultiplyAlpha: 'none' });
        }
        catch (e) {
            imageBitmap = null;
        }
        // createImageBitmap will clip the source size.
        // In some cases, the input will have larger size than its content.
        // E.g. new Image(10, 10) but with 1 x 1 content. Using
        // createImageBitmap will clip the size from 10 x 10 to 1 x 1, which
        // is not correct. We should avoid wrapping such resouce to
        // imageBitmap.
        if (imageBitmap != null && imageBitmap.width === pixels.width &&
            imageBitmap.height === pixels.height) {
            inputs = imageBitmap;
        }
        else {
            inputs = pixels;
        }
    }
    else {
        inputs = pixels;
    }
    return fromPixels_(inputs, numChannels);
}
/**
 * Draws a `tf.Tensor` of pixel values to a byte array or optionally a
 * canvas.
 *
 * When the dtype of the input is 'float32', we assume values in the range
 * [0-1]. Otherwise, when input is 'int32', we assume values in the range
 * [0-255].
 *
 * Returns a promise that resolves when the canvas has been drawn to.
 *
 * @param img A rank-2 tensor with shape `[height, width]`, or a rank-3 tensor
 * of shape `[height, width, numChannels]`. If rank-2, draws grayscale. If
 * rank-3, must have depth of 1, 3 or 4. When depth of 1, draws
 * grayscale. When depth of 3, we draw with the first three components of
 * the depth dimension corresponding to r, g, b and alpha = 1. When depth of
 * 4, all four components of the depth dimension correspond to r, g, b, a.
 * @param canvas The canvas to draw to.
 *
 * @doc {heading: 'Browser', namespace: 'browser'}
 */
export async function toPixels(img, canvas) {
    let $img = convertToTensor(img, 'img', 'toPixels');
    if (!(img instanceof Tensor)) {
        // Assume int32 if user passed a native array.
        const originalImgTensor = $img;
        $img = cast(originalImgTensor, 'int32');
        originalImgTensor.dispose();
    }
    if ($img.rank !== 2 && $img.rank !== 3) {
        throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${$img.rank}.`);
    }
    const [height, width] = $img.shape.slice(0, 2);
    const depth = $img.rank === 2 ? 1 : $img.shape[2];
    if (depth > 4 || depth === 2) {
        throw new Error(`toPixels only supports depth of size ` +
            `1, 3 or 4 but got ${depth}`);
    }
    if ($img.dtype !== 'float32' && $img.dtype !== 'int32') {
        throw new Error(`Unsupported type for toPixels: ${$img.dtype}.` +
            ` Please use float32 or int32 tensors.`);
    }
    const data = await $img.data();
    const multiplier = $img.dtype === 'float32' ? 255 : 1;
    const bytes = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < height * width; ++i) {
        const rgba = [0, 0, 0, 255];
        for (let d = 0; d < depth; d++) {
            const value = data[i * depth + d];
            if ($img.dtype === 'float32') {
                if (value < 0 || value > 1) {
                    throw new Error(`Tensor values for a float32 Tensor must be in the ` +
                        `range [0 - 1] but encountered ${value}.`);
                }
            }
            else if ($img.dtype === 'int32') {
                if (value < 0 || value > 255) {
                    throw new Error(`Tensor values for a int32 Tensor must be in the ` +
                        `range [0 - 255] but encountered ${value}.`);
                }
            }
            if (depth === 1) {
                rgba[0] = value * multiplier;
                rgba[1] = value * multiplier;
                rgba[2] = value * multiplier;
            }
            else {
                rgba[d] = value * multiplier;
            }
        }
        const j = i * 4;
        bytes[j + 0] = Math.round(rgba[0]);
        bytes[j + 1] = Math.round(rgba[1]);
        bytes[j + 2] = Math.round(rgba[2]);
        bytes[j + 3] = Math.round(rgba[3]);
    }
    if (canvas != null) {
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        const imageData = new ImageData(bytes, width, height);
        ctx.putImageData(imageData, 0, 0);
    }
    if ($img !== img) {
        $img.dispose();
    }
    return bytes;
}
export const fromPixels = op({ fromPixels_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYnJvd3Nlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2Jyb3dzZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDbkMsT0FBTyxFQUFDLFVBQVUsRUFBb0MsTUFBTSxpQkFBaUIsQ0FBQztBQUM5RSxPQUFPLEVBQUMsU0FBUyxFQUFlLE1BQU0sb0JBQW9CLENBQUM7QUFDM0QsT0FBTyxFQUFDLE1BQU0sRUFBcUIsTUFBTSxXQUFXLENBQUM7QUFFckQsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBR25ELE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDNUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsUUFBUSxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRXBDLElBQUksbUJBQTZDLENBQUM7QUFFbEQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0ErQkc7QUFDSCxTQUFTLFdBQVcsQ0FDaEIsTUFDNEIsRUFDNUIsV0FBVyxHQUFHLENBQUM7SUFDakIsaUJBQWlCO0lBQ2pCLElBQUksV0FBVyxHQUFHLENBQUMsRUFBRTtRQUNuQixNQUFNLElBQUksS0FBSyxDQUNYLGdFQUFnRSxDQUFDLENBQUM7S0FDdkU7SUFDRCxJQUFJLE1BQU0sSUFBSSxJQUFJLEVBQUU7UUFDbEIsTUFBTSxJQUFJLEtBQUssQ0FBQywwREFBMEQsQ0FBQyxDQUFDO0tBQzdFO0lBQ0QsSUFBSSxXQUFXLEdBQUcsS0FBSyxDQUFDO0lBQ3hCLElBQUksV0FBVyxHQUFHLEtBQUssQ0FBQztJQUN4QixJQUFJLE9BQU8sR0FB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QUNoQixJQUFJLENBQUMsT0FBTyxFQUFFLENBQUM7S0FDaEI7SUFDRCxPQUFPLEtBQUssQ0FBQztBQUNmLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxVQUFVLEdBQUcsRUFBRSxDQUFDLEVBQUMsV0FBVyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE5IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge2Vudn0gZnJvbSAnLi4vZW52aXJvbm1lbnQnO1xuaW1wb3J0IHtGcm9tUGl4ZWxzLCBGcm9tUGl4ZWxzQXR0cnMsIEZyb21QaXhlbHNJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge2dldEtlcm5lbCwgTmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3IsIFRlbnNvcjJELCBUZW5zb3IzRH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7UGl4ZWxEYXRhLCBUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7Y2FzdH0gZnJvbSAnLi9jYXN0JztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7dGVuc29yM2R9IGZyb20gJy4vdGVuc29yM2QnO1xuXG5sZXQgZnJvbVBpeGVsczJEQ29udGV4dDogQ2FudmFzUmVuZGVyaW5nQ29udGV4dDJEO1xuXG4vKipcbiAqIENyZWF0ZXMgYSBgdGYuVGVuc29yYCBmcm9tIGFuIGltYWdlLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBpbWFnZSA9IG5ldyBJbWFnZURhdGEoMSwgMSk7XG4gKiBpbWFnZS5kYXRhWzBdID0gMTAwO1xuICogaW1hZ2UuZGF0YVsxXSA9IDE1MDtcbiAqIGltYWdlLmRhdGFbMl0gPSAyMDA7XG4gKiBpbWFnZS5kYXRhWzNdID0gMjU1O1xuICpcbiAqIHRmLmJyb3dzZXIuZnJvbVBpeGVscyhpbWFnZSkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSBwaXhlbHMgVGhlIGlucHV0IGltYWdlIHRvIGNvbnN0cnVjdCB0aGUgdGVuc29yIGZyb20uIFRoZVxuICogc3VwcG9ydGVkIGltYWdlIHR5cGVzIGFyZSBhbGwgNC1jaGFubmVsLiBZb3UgY2FuIGFsc28gcGFzcyBpbiBhbiBpbWFnZVxuICogb2JqZWN0IHdpdGggZm9sbG93aW5nIGF0dHJpYnV0ZXM6XG4gKiBge2RhdGE6IFVpbnQ4QXJyYXk7IHdpZHRoOiBudW1iZXI7IGhlaWdodDogbnVtYmVyfWBcbiAqIEBwYXJhbSBudW1DaGFubmVscyBUaGUgbnVtYmVyIG9mIGNoYW5uZWxzIG9mIHRoZSBvdXRwdXQgdGVuc29yLiBBXG4gKiBudW1DaGFubmVscyB2YWx1ZSBsZXNzIHRoYW4gNCBhbGxvd3MgeW91IHRvIGlnbm9yZSBjaGFubmVscy4gRGVmYXVsdHMgdG9cbiAqIDMgKGlnbm9yZXMgYWxwaGEgY2hhbm5lbCBvZiBpbnB1dCBpbWFnZSkuXG4gKlxuICogQHJldHVybnMgQSBUZW5zb3IzRCB3aXRoIHRoZSBzaGFwZSBgW2hlaWdodCwgd2lkdGgsIG51bUNoYW5uZWxzXWAuXG4gKlxuICogTm90ZTogZnJvbVBpeGVscyBjYW4gYmUgbG9zc3kgaW4gc29tZSBjYXNlcywgc2FtZSBpbWFnZSBtYXkgcmVzdWx0IGluXG4gKiBzbGlnaHRseSBkaWZmZXJlbnQgdGVuc29yIHZhbHVlcywgaWYgcmVuZGVyZWQgYnkgZGlmZmVyZW50IHJlbmRlcmluZ1xuICogZW5naW5lcy4gVGhpcyBtZWFucyB0aGF0IHJlc3VsdHMgZnJvbSBkaWZmZXJlbnQgYnJvd3NlcnMsIG9yIGV2ZW4gc2FtZVxuICogYnJvd3NlciB3aXRoIENQVSBhbmQgR1BVIHJlbmRlcmluZyBlbmdpbmVzIGNhbiBiZSBkaWZmZXJlbnQuIFNlZSBkaXNjdXNzaW9uXG4gKiBpbiBkZXRhaWxzOlxuICogaHR0cHM6Ly9naXRodWIuY29tL3RlbnNvcmZsb3cvdGZqcy9pc3N1ZXMvNTQ4MlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdCcm93c2VyJywgbmFtZXNwYWNlOiAnYnJvd3NlcicsIGlnbm9yZUNJOiB0cnVlfVxuICovXG5mdW5jdGlvbiBmcm9tUGl4ZWxzXyhcbiAgICBwaXhlbHM6IFBpeGVsRGF0YXxJbWFnZURhdGF8SFRNTEltYWdlRWxlbWVudHxIVE1MQ2FudmFzRWxlbWVudHxcbiAgICBIVE1MVmlkZW9FbGVtZW50fEltYWdlQml0bWFwLFxuICAgIG51bUNoYW5uZWxzID0gMyk6IFRlbnNvcjNEIHtcbiAgLy8gU2FuaXR5IGNoZWNrcy5cbiAgaWYgKG51bUNoYW5uZWxzID4gNCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ0Nhbm5vdCBjb25zdHJ1Y3QgVGVuc29yIHdpdGggbW9yZSB0aGFuIDQgY2hhbm5lbHMgZnJvbSBwaXhlbHMuJyk7XG4gIH1cbiAgaWYgKHBpeGVscyA9PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdwaXhlbHMgcGFzc2VkIHRvIHRmLmJyb3dzZXIuZnJvbVBpeGVscygpIGNhbiBub3QgYmUgbnVsbCcpO1xuICB9XG4gIGxldCBpc1BpeGVsRGF0YSA9IGZhbHNlO1xuICBsZXQgaXNJbWFnZURhdGEgPSBmYWxzZTtcbiAgbGV0IGlzVmlkZW8gPSBmYWxzZTtcbiAgbGV0IGlzSW1hZ2UgPSBmYWxzZTtcbiAgbGV0IGlzQ2FudmFzTGlrZSA9IGZhbHNlO1xuICBsZXQgaXNJbWFnZUJpdG1hcCA9IGZhbHNlO1xuICBpZiAoKHBpeGVscyBhcyBQaXhlbERhdGEpLmRhdGEgaW5zdGFuY2VvZiBVaW50OEFycmF5KSB7XG4gICAgaXNQaXhlbERhdGEgPSB0cnVlO1xuICB9IGVsc2UgaWYgKFxuICAgICAgdHlwZW9mIChJbWFnZURhdGEpICE9PSAndW5kZWZpbmVkJyAmJiBwaXhlbHMgaW5zdGFuY2VvZiBJbWFnZURhdGEpIHtcbiAgICBpc0ltYWdlRGF0YSA9IHRydWU7XG4gIH0gZWxzZSBpZiAoXG4gICAgICB0eXBlb2YgKEhUTUxWaWRlb0VsZW1lbnQpICE9PSAndW5kZWZpbmVkJyAmJlxuICAgICAgcGl4ZWxzIGluc3RhbmNlb2YgSFRNTFZpZGVvRWxlbWVudCkge1xuICAgIGlzVmlkZW8gPSB0cnVlO1xuICB9IGVsc2UgaWYgKFxuICAgICAgdHlwZW9mIChIVE1MSW1hZ2VFbGVtZW50KSAhPT0gJ3VuZGVmaW5lZCcgJiZcbiAgICAgIHBpeGVscyBpbnN0YW5jZW9mIEhUTUxJbWFnZUVsZW1lbnQpIHtcbiAgICBpc0ltYWdlID0gdHJ1ZTtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLWFueVxuICB9IGVsc2UgaWYgKChwaXhlbHMgYXMgYW55KS5nZXRDb250ZXh0ICE9IG51bGwpIHtcbiAgICBpc0NhbnZhc0xpa2UgPSB0cnVlO1xuICB9IGVsc2UgaWYgKFxuICAgICAgdHlwZW9mIChJbWFnZUJpdG1hcCkgIT09ICd1bmRlZmluZWQnICYmIHBpeGVscyBpbnN0YW5jZW9mIEltYWdlQml0bWFwKSB7XG4gICAgaXNJbWFnZUJpdG1hcCA9IHRydWU7XG4gIH0gZWxzZSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAncGl4ZWxzIHBhc3NlZCB0byB0Zi5icm93c2VyLmZyb21QaXhlbHMoKSBtdXN0IGJlIGVpdGhlciBhbiAnICtcbiAgICAgICAgYEhUTUxWaWRlb0VsZW1lbnQsIEhUTUxJbWFnZUVsZW1lbnQsIEhUTUxDYW52YXNFbGVtZW50LCBJbWFnZURhdGEgYCArXG4gICAgICAgIGBpbiBicm93c2VyLCBvciBPZmZzY3JlZW5DYW52YXMsIEltYWdlRGF0YSBpbiB3ZWJ3b3JrZXJgICtcbiAgICAgICAgYCBvciB7ZGF0YTogVWludDMyQXJyYXksIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyfSwgYCArXG4gICAgICAgIGBidXQgd2FzICR7KHBpeGVscyBhcyB7fSkuY29uc3RydWN0b3IubmFtZX1gKTtcbiAgfVxuICBpZiAoaXNWaWRlbykge1xuICAgIGNvbnN0IEhBVkVfQ1VSUkVOVF9EQVRBX1JFQURZX1NUQVRFID0gMjtcbiAgICBpZiAoaXNWaWRlbyAmJlxuICAgICAgICAocGl4ZWxzIGFzIEhUTUxWaWRlb0VsZW1lbnQpLnJlYWR5U3RhdGUgPFxuICAgICAgICAgICAgSEFWRV9DVVJSRU5UX0RBVEFfUkVBRFlfU1RBVEUpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAnVGhlIHZpZGVvIGVsZW1lbnQgaGFzIG5vdCBsb2FkZWQgZGF0YSB5ZXQuIFBsZWFzZSB3YWl0IGZvciAnICtcbiAgICAgICAgICAnYGxvYWRlZGRhdGFgIGV2ZW50IG9uIHRoZSA8dmlkZW8+IGVsZW1lbnQuJyk7XG4gICAgfVxuICB9XG4gIC8vIElmIHRoZSBjdXJyZW50IGJhY2tlbmQgaGFzICdGcm9tUGl4ZWxzJyByZWdpc3RlcmVkLCBpdCBoYXMgYSBtb3JlXG4gIC8vIGVmZmljaWVudCB3YXkgb2YgaGFuZGxpbmcgcGl4ZWwgdXBsb2Fkcywgc28gd2UgY2FsbCB0aGF0LlxuICBjb25zdCBrZXJuZWwgPSBnZXRLZXJuZWwoRnJvbVBpeGVscywgRU5HSU5FLmJhY2tlbmROYW1lKTtcbiAgaWYgKGtlcm5lbCAhPSBudWxsKSB7XG4gICAgY29uc3QgaW5wdXRzOiBGcm9tUGl4ZWxzSW5wdXRzID0ge3BpeGVsc307XG4gICAgY29uc3QgYXR0cnM6IEZyb21QaXhlbHNBdHRycyA9IHtudW1DaGFubmVsc307XG4gICAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgIEZyb21QaXhlbHMsIGlucHV0cyBhcyB7fSBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgYXR0cnMgYXMge30gYXMgTmFtZWRBdHRyTWFwKTtcbiAgfVxuXG4gIGNvbnN0IFt3aWR0aCwgaGVpZ2h0XSA9IGlzVmlkZW8gP1xuICAgICAgW1xuICAgICAgICAocGl4ZWxzIGFzIEhUTUxWaWRlb0VsZW1lbnQpLnZpZGVvV2lkdGgsXG4gICAgICAgIChwaXhlbHMgYXMgSFRNTFZpZGVvRWxlbWVudCkudmlkZW9IZWlnaHRcbiAgICAgIF0gOlxuICAgICAgW3BpeGVscy53aWR0aCwgcGl4ZWxzLmhlaWdodF07XG4gIGxldCB2YWxzOiBVaW50OENsYW1wZWRBcnJheXxVaW50OEFycmF5O1xuXG4gIGlmIChpc0NhbnZhc0xpa2UpIHtcbiAgICB2YWxzID1cbiAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICAocGl4ZWxzIGFzIGFueSkuZ2V0Q29udGV4dCgnMmQnKS5nZXRJbWFnZURhdGEoMCwgMCwgd2lkdGgsIGhlaWdodCkuZGF0YTtcbiAgfSBlbHNlIGlmIChpc0ltYWdlRGF0YSB8fCBpc1BpeGVsRGF0YSkge1xuICAgIHZhbHMgPSAocGl4ZWxzIGFzIFBpeGVsRGF0YSB8IEltYWdlRGF0YSkuZGF0YTtcbiAgfSBlbHNlIGlmIChpc0ltYWdlIHx8IGlzVmlkZW8gfHwgaXNJbWFnZUJpdG1hcCkge1xuICAgIGlmIChmcm9tUGl4ZWxzMkRDb250ZXh0ID09IG51bGwpIHtcbiAgICAgIGlmICh0eXBlb2YgZG9jdW1lbnQgPT09ICd1bmRlZmluZWQnKSB7XG4gICAgICAgIGlmICh0eXBlb2YgT2Zmc2NyZWVuQ2FudmFzICE9PSAndW5kZWZpbmVkJyAmJlxuICAgICAgICAgICAgdHlwZW9mIE9mZnNjcmVlbkNhbnZhc1JlbmRlcmluZ0NvbnRleHQyRCAhPT0gJ3VuZGVmaW5lZCcpIHtcbiAgICAgICAgICAvLyBAdHMtaWdub3JlXG4gICAgICAgICAgZnJvbVBpeGVsczJEQ29udGV4dCA9IG5ldyBPZmZzY3JlZW5DYW52YXMoMSwgMSkuZ2V0Q29udGV4dCgnMmQnKTtcbiAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAgICdDYW5ub3QgcGFyc2UgaW5wdXQgaW4gY3VycmVudCBjb250ZXh0LiAnICtcbiAgICAgICAgICAgICAgJ1JlYXNvbjogT2Zmc2NyZWVuQ2FudmFzIENvbnRleHQyRCByZW5kZXJpbmcgaXMgbm90IHN1cHBvcnRlZC4nKTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIHtcbiAgICAgICAgZnJvbVBpeGVsczJEQ29udGV4dCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2NhbnZhcycpLmdldENvbnRleHQoJzJkJyk7XG4gICAgICB9XG4gICAgfVxuICAgIGZyb21QaXhlbHMyRENvbnRleHQuY2FudmFzLndpZHRoID0gd2lkdGg7XG4gICAgZnJvbVBpeGVsczJEQ29udGV4dC5jYW52YXMuaGVpZ2h0ID0gaGVpZ2h0O1xuICAgIGZyb21QaXhlbHMyRENvbnRleHQuZHJhd0ltYWdlKFxuICAgICAgICBwaXhlbHMgYXMgSFRNTFZpZGVvRWxlbWVudCwgMCwgMCwgd2lkdGgsIGhlaWdodCk7XG4gICAgdmFscyA9IGZyb21QaXhlbHMyRENvbnRleHQuZ2V0SW1hZ2VEYXRhKDAsIDAsIHdpZHRoLCBoZWlnaHQpLmRhdGE7XG4gIH1cbiAgbGV0IHZhbHVlczogSW50MzJBcnJheTtcbiAgaWYgKG51bUNoYW5uZWxzID09PSA0KSB7XG4gICAgdmFsdWVzID0gbmV3IEludDMyQXJyYXkodmFscyk7XG4gIH0gZWxzZSB7XG4gICAgY29uc3QgbnVtUGl4ZWxzID0gd2lkdGggKiBoZWlnaHQ7XG4gICAgdmFsdWVzID0gbmV3IEludDMyQXJyYXkobnVtUGl4ZWxzICogbnVtQ2hhbm5lbHMpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbnVtUGl4ZWxzOyBpKyspIHtcbiAgICAgIGZvciAobGV0IGNoYW5uZWwgPSAwOyBjaGFubmVsIDwgbnVtQ2hhbm5lbHM7ICsrY2hhbm5lbCkge1xuICAgICAgICB2YWx1ZXNbaSAqIG51bUNoYW5uZWxzICsgY2hhbm5lbF0gPSB2YWxzW2kgKiA0ICsgY2hhbm5lbF07XG4gICAgICB9XG4gICAgfVxuICB9XG4gIGNvbnN0IG91dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbaGVpZ2h0LCB3aWR0aCwgbnVtQ2hhbm5lbHNdO1xuICByZXR1cm4gdGVuc29yM2QodmFsdWVzLCBvdXRTaGFwZSwgJ2ludDMyJyk7XG59XG5cbi8vIEhlbHBlciBmdW5jdGlvbnMgZm9yIHxmcm9tUGl4ZWxzQXN5bmN8IHRvIGNoZWNrIHdoZXRoZXIgdGhlIGlucHV0IGNhblxuLy8gYmUgd3JhcHBlZCBpbnRvIGltYWdlQml0bWFwLlxuZnVuY3Rpb24gaXNQaXhlbERhdGEocGl4ZWxzOiBQaXhlbERhdGF8SW1hZ2VEYXRhfEhUTUxJbWFnZUVsZW1lbnR8XG4gICAgICAgICAgICAgICAgICAgICBIVE1MQ2FudmFzRWxlbWVudHxIVE1MVmlkZW9FbGVtZW50fFxuICAgICAgICAgICAgICAgICAgICAgSW1hZ2VCaXRtYXApOiBwaXhlbHMgaXMgUGl4ZWxEYXRhIHtcbiAgcmV0dXJuIChwaXhlbHMgIT0gbnVsbCkgJiYgKChwaXhlbHMgYXMgUGl4ZWxEYXRhKS5kYXRhIGluc3RhbmNlb2YgVWludDhBcnJheSk7XG59XG5cbmZ1bmN0aW9uIGlzSW1hZ2VCaXRtYXBGdWxseVN1cHBvcnRlZCgpIHtcbiAgcmV0dXJuIHR5cGVvZiB3aW5kb3cgIT09ICd1bmRlZmluZWQnICYmXG4gICAgICB0eXBlb2YgKEltYWdlQml0bWFwKSAhPT0gJ3VuZGVmaW5lZCcgJiZcbiAgICAgIHdpbmRvdy5oYXNPd25Qcm9wZXJ0eSgnY3JlYXRlSW1hZ2VCaXRtYXAnKTtcbn1cblxuZnVuY3Rpb24gaXNOb25FbXB0eVBpeGVscyhwaXhlbHM6IFBpeGVsRGF0YXxJbWFnZURhdGF8SFRNTEltYWdlRWxlbWVudHxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgSFRNTENhbnZhc0VsZW1lbnR8SFRNTFZpZGVvRWxlbWVudHxJbWFnZUJpdG1hcCkge1xuICByZXR1cm4gcGl4ZWxzICE9IG51bGwgJiYgcGl4ZWxzLndpZHRoICE9PSAwICYmIHBpeGVscy5oZWlnaHQgIT09IDA7XG59XG5cbmZ1bmN0aW9uIGNhbldyYXBQaXhlbHNUb0ltYWdlQml0bWFwKHBpeGVsczogUGl4ZWxEYXRhfEltYWdlRGF0YXxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEhUTUxJbWFnZUVsZW1lbnR8SFRNTENhbnZhc0VsZW1lbnR8XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBIVE1MVmlkZW9FbGVtZW50fEltYWdlQml0bWFwKSB7XG4gIHJldHVybiBpc0ltYWdlQml0bWFwRnVsbHlTdXBwb3J0ZWQoKSAmJiAhKHBpeGVscyBpbnN0YW5jZW9mIEltYWdlQml0bWFwKSAmJlxuICAgICAgaXNOb25FbXB0eVBpeGVscyhwaXhlbHMpICYmICFpc1BpeGVsRGF0YShwaXhlbHMpO1xufVxuXG4vKipcbiAqIENyZWF0ZXMgYSBgdGYuVGVuc29yYCBmcm9tIGFuIGltYWdlIGluIGFzeW5jIHdheS5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgaW1hZ2UgPSBuZXcgSW1hZ2VEYXRhKDEsIDEpO1xuICogaW1hZ2UuZGF0YVswXSA9IDEwMDtcbiAqIGltYWdlLmRhdGFbMV0gPSAxNTA7XG4gKiBpbWFnZS5kYXRhWzJdID0gMjAwO1xuICogaW1hZ2UuZGF0YVszXSA9IDI1NTtcbiAqXG4gKiAoYXdhaXQgdGYuYnJvd3Nlci5mcm9tUGl4ZWxzQXN5bmMoaW1hZ2UpKS5wcmludCgpO1xuICogYGBgXG4gKiBUaGlzIEFQSSBpcyB0aGUgYXN5bmMgdmVyc2lvbiBvZiBmcm9tUGl4ZWxzLiBUaGUgQVBJIHdpbGwgZmlyc3RcbiAqIGNoZWNrIHxXUkFQX1RPX0lNQUdFQklUTUFQfCBmbGFnLCBhbmQgdHJ5IHRvIHdyYXAgdGhlIGlucHV0IHRvXG4gKiBpbWFnZUJpdG1hcCBpZiB0aGUgZmxhZyBpcyBzZXQgdG8gdHJ1ZS5cbiAqXG4gKiBAcGFyYW0gcGl4ZWxzIFRoZSBpbnB1dCBpbWFnZSB0byBjb25zdHJ1Y3QgdGhlIHRlbnNvciBmcm9tLiBUaGVcbiAqIHN1cHBvcnRlZCBpbWFnZSB0eXBlcyBhcmUgYWxsIDQtY2hhbm5lbC4gWW91IGNhbiBhbHNvIHBhc3MgaW4gYW4gaW1hZ2VcbiAqIG9iamVjdCB3aXRoIGZvbGxvd2luZyBhdHRyaWJ1dGVzOlxuICogYHtkYXRhOiBVaW50OEFycmF5OyB3aWR0aDogbnVtYmVyOyBoZWlnaHQ6IG51bWJlcn1gXG4gKiBAcGFyYW0gbnVtQ2hhbm5lbHMgVGhlIG51bWJlciBvZiBjaGFubmVscyBvZiB0aGUgb3V0cHV0IHRlbnNvci4gQVxuICogbnVtQ2hhbm5lbHMgdmFsdWUgbGVzcyB0aGFuIDQgYWxsb3dzIHlvdSB0byBpZ25vcmUgY2hhbm5lbHMuIERlZmF1bHRzIHRvXG4gKiAzIChpZ25vcmVzIGFscGhhIGNoYW5uZWwgb2YgaW5wdXQgaW1hZ2UpLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdCcm93c2VyJywgbmFtZXNwYWNlOiAnYnJvd3NlcicsIGlnbm9yZUNJOiB0cnVlfVxuICovXG5leHBvcnQgYXN5bmMgZnVuY3Rpb24gZnJvbVBpeGVsc0FzeW5jKFxuICAgIHBpeGVsczogUGl4ZWxEYXRhfEltYWdlRGF0YXxIVE1MSW1hZ2VFbGVtZW50fEhUTUxDYW52YXNFbGVtZW50fFxuICAgIEhUTUxWaWRlb0VsZW1lbnR8SW1hZ2VCaXRtYXAsXG4gICAgbnVtQ2hhbm5lbHMgPSAzKSB7XG4gIGxldCBpbnB1dHM6IFBpeGVsRGF0YXxJbWFnZURhdGF8SFRNTEltYWdlRWxlbWVudHxIVE1MQ2FudmFzRWxlbWVudHxcbiAgICAgIEhUTUxWaWRlb0VsZW1lbnR8SW1hZ2VCaXRtYXAgPSBudWxsO1xuXG4gIC8vIENoZWNrIHdoZXRoZXIgdGhlIGJhY2tlbmQgbmVlZHMgdG8gd3JhcCB8cGl4ZWxzfCB0byBpbWFnZUJpdG1hcCBhbmRcbiAgLy8gd2hldGhlciB8cGl4ZWxzfCBjYW4gYmUgd3JhcHBlZCB0byBpbWFnZUJpdG1hcC5cbiAgaWYgKGVudigpLmdldEJvb2woJ1dSQVBfVE9fSU1BR0VCSVRNQVAnKSAmJlxuICAgICAgY2FuV3JhcFBpeGVsc1RvSW1hZ2VCaXRtYXAocGl4ZWxzKSkge1xuICAgIC8vIEZvcmNlIHRoZSBpbWFnZUJpdG1hcCBjcmVhdGlvbiB0byBub3QgZG8gYW55IHByZW11bHRpcGx5IGFscGhhXG4gICAgLy8gb3BzLlxuICAgIGxldCBpbWFnZUJpdG1hcDtcblxuICAgIHRyeSB7XG4gICAgICAvLyB3cmFwIGluIHRyeS1jYXRjaCBibG9jaywgYmVjYXVzZSBjcmVhdGVJbWFnZUJpdG1hcCBtYXkgbm90IHdvcmtcbiAgICAgIC8vIHByb3Blcmx5IGluIHNvbWUgYnJvd3NlcnMsIGUuZy5cbiAgICAgIC8vIGh0dHBzOi8vYnVnemlsbGEubW96aWxsYS5vcmcvc2hvd19idWcuY2dpP2lkPTEzMzU1OTRcbiAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tYW55XG4gICAgICBpbWFnZUJpdG1hcCA9IGF3YWl0IChjcmVhdGVJbWFnZUJpdG1hcCBhcyBhbnkpKFxuICAgICAgICAgIHBpeGVscyBhcyBJbWFnZUJpdG1hcFNvdXJjZSwge3ByZW11bHRpcGx5QWxwaGE6ICdub25lJ30pO1xuICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgIGltYWdlQml0bWFwID0gbnVsbDtcbiAgICB9XG5cbiAgICAvLyBjcmVhdGVJbWFnZUJpdG1hcCB3aWxsIGNsaXAgdGhlIHNvdXJjZSBzaXplLlxuICAgIC8vIEluIHNvbWUgY2FzZXMsIHRoZSBpbnB1dCB3aWxsIGhhdmUgbGFyZ2VyIHNpemUgdGhhbiBpdHMgY29udGVudC5cbiAgICAvLyBFLmcuIG5ldyBJbWFnZSgxMCwgMTApIGJ1dCB3aXRoIDEgeCAxIGNvbnRlbnQuIFVzaW5nXG4gICAgLy8gY3JlYXRlSW1hZ2VCaXRtYXAgd2lsbCBjbGlwIHRoZSBzaXplIGZyb20gMTAgeCAxMCB0byAxIHggMSwgd2hpY2hcbiAgICAvLyBpcyBub3QgY29ycmVjdC4gV2Ugc2hvdWxkIGF2b2lkIHdyYXBwaW5nIHN1Y2ggcmVzb3VjZSB0b1xuICAgIC8vIGltYWdlQml0bWFwLlxuICAgIGlmIChpbWFnZUJpdG1hcCAhPSBudWxsICYmIGltYWdlQml0bWFwLndpZHRoID09PSBwaXhlbHMud2lkdGggJiZcbiAgICAgICAgaW1hZ2VCaXRtYXAuaGVpZ2h0ID09PSBwaXhlbHMuaGVpZ2h0KSB7XG4gICAgICBpbnB1dHMgPSBpbWFnZUJpdG1hcDtcbiAgICB9IGVsc2Uge1xuICAgICAgaW5wdXRzID0gcGl4ZWxzO1xuICAgIH1cbiAgfSBlbHNlIHtcbiAgICBpbnB1dHMgPSBwaXhlbHM7XG4gIH1cblxuICByZXR1cm4gZnJvbVBpeGVsc18oaW5wdXRzLCBudW1DaGFubmVscyk7XG59XG5cbi8qKlxuICogRHJhd3MgYSBgdGYuVGVuc29yYCBvZiBwaXhlbCB2YWx1ZXMgdG8gYSBieXRlIGFycmF5IG9yIG9wdGlvbmFsbHkgYVxuICogY2FudmFzLlxuICpcbiAqIFdoZW4gdGhlIGR0eXBlIG9mIHRoZSBpbnB1dCBpcyAnZmxvYXQzMicsIHdlIGFzc3VtZSB2YWx1ZXMgaW4gdGhlIHJhbmdlXG4gKiBbMC0xXS4gT3RoZXJ3aXNlLCB3aGVuIGlucHV0IGlzICdpbnQzMicsIHdlIGFzc3VtZSB2YWx1ZXMgaW4gdGhlIHJhbmdlXG4gKiBbMC0yNTVdLlxuICpcbiAqIFJldHVybnMgYSBwcm9taXNlIHRoYXQgcmVzb2x2ZXMgd2hlbiB0aGUgY2FudmFzIGhhcyBiZWVuIGRyYXduIHRvLlxuICpcbiAqIEBwYXJhbSBpbWcgQSByYW5rLTIgdGVuc29yIHdpdGggc2hhcGUgYFtoZWlnaHQsIHdpZHRoXWAsIG9yIGEgcmFuay0zIHRlbnNvclxuICogb2Ygc2hhcGUgYFtoZWlnaHQsIHdpZHRoLCBudW1DaGFubmVsc11gLiBJZiByYW5rLTIsIGRyYXdzIGdyYXlzY2FsZS4gSWZcbiAqIHJhbmstMywgbXVzdCBoYXZlIGRlcHRoIG9mIDEsIDMgb3IgNC4gV2hlbiBkZXB0aCBvZiAxLCBkcmF3c1xuICogZ3JheXNjYWxlLiBXaGVuIGRlcHRoIG9mIDMsIHdlIGRyYXcgd2l0aCB0aGUgZmlyc3QgdGhyZWUgY29tcG9uZW50cyBvZlxuICogdGhlIGRlcHRoIGRpbWVuc2lvbiBjb3JyZXNwb25kaW5nIHRvIHIsIGcsIGIgYW5kIGFscGhhID0gMS4gV2hlbiBkZXB0aCBvZlxuICogNCwgYWxsIGZvdXIgY29tcG9uZW50cyBvZiB0aGUgZGVwdGggZGltZW5zaW9uIGNvcnJlc3BvbmQgdG8gciwgZywgYiwgYS5cbiAqIEBwYXJhbSBjYW52YXMgVGhlIGNhbnZhcyB0byBkcmF3IHRvLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdCcm93c2VyJywgbmFtZXNwYWNlOiAnYnJvd3Nlcid9XG4gKi9cbmV4cG9ydCBhc3luYyBmdW5jdGlvbiB0b1BpeGVscyhcbiAgICBpbWc6IFRlbnNvcjJEfFRlbnNvcjNEfFRlbnNvckxpa2UsXG4gICAgY2FudmFzPzogSFRNTENhbnZhc0VsZW1lbnQpOiBQcm9taXNlPFVpbnQ4Q2xhbXBlZEFycmF5PiB7XG4gIGxldCAkaW1nID0gY29udmVydFRvVGVuc29yKGltZywgJ2ltZycsICd0b1BpeGVscycpO1xuICBpZiAoIShpbWcgaW5zdGFuY2VvZiBUZW5zb3IpKSB7XG4gICAgLy8gQXNzdW1lIGludDMyIGlmIHVzZXIgcGFzc2VkIGEgbmF0aXZlIGFycmF5LlxuICAgIGNvbnN0IG9yaWdpbmFsSW1nVGVuc29yID0gJGltZztcbiAgICAkaW1nID0gY2FzdChvcmlnaW5hbEltZ1RlbnNvciwgJ2ludDMyJyk7XG4gICAgb3JpZ2luYWxJbWdUZW5zb3IuZGlzcG9zZSgpO1xuICB9XG4gIGlmICgkaW1nLnJhbmsgIT09IDIgJiYgJGltZy5yYW5rICE9PSAzKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgdG9QaXhlbHMgb25seSBzdXBwb3J0cyByYW5rIDIgb3IgMyB0ZW5zb3JzLCBnb3QgcmFuayAkeyRpbWcucmFua30uYCk7XG4gIH1cbiAgY29uc3QgW2hlaWdodCwgd2lkdGhdID0gJGltZy5zaGFwZS5zbGljZSgwLCAyKTtcbiAgY29uc3QgZGVwdGggPSAkaW1nLnJhbmsgPT09IDIgPyAxIDogJGltZy5zaGFwZVsyXTtcblxuICBpZiAoZGVwdGggPiA0IHx8IGRlcHRoID09PSAyKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgdG9QaXhlbHMgb25seSBzdXBwb3J0cyBkZXB0aCBvZiBzaXplIGAgK1xuICAgICAgICBgMSwgMyBvciA0IGJ1dCBnb3QgJHtkZXB0aH1gKTtcbiAgfVxuXG4gIGlmICgkaW1nLmR0eXBlICE9PSAnZmxvYXQzMicgJiYgJGltZy5kdHlwZSAhPT0gJ2ludDMyJykge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYFVuc3VwcG9ydGVkIHR5cGUgZm9yIHRvUGl4ZWxzOiAkeyRpbWcuZHR5cGV9LmAgK1xuICAgICAgICBgIFBsZWFzZSB1c2UgZmxvYXQzMiBvciBpbnQzMiB0ZW5zb3JzLmApO1xuICB9XG5cbiAgY29uc3QgZGF0YSA9IGF3YWl0ICRpbWcuZGF0YSgpO1xuICBjb25zdCBtdWx0aXBsaWVyID0gJGltZy5kdHlwZSA9PT0gJ2Zsb2F0MzInID8gMjU1IDogMTtcbiAgY29uc3QgYnl0ZXMgPSBuZXcgVWludDhDbGFtcGVkQXJyYXkod2lkdGggKiBoZWlnaHQgKiA0KTtcblxuICBmb3IgKGxldCBpID0gMDsgaSA8IGhlaWdodCAqIHdpZHRoOyArK2kpIHtcbiAgICBjb25zdCByZ2JhID0gWzAsIDAsIDAsIDI1NV07XG5cbiAgICBmb3IgKGxldCBkID0gMDsgZCA8IGRlcHRoOyBkKyspIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gZGF0YVtpICogZGVwdGggKyBkXTtcblxuICAgICAgaWYgKCRpbWcuZHR5cGUgPT09ICdmbG9hdDMyJykge1xuICAgICAgICBpZiAodmFsdWUgPCAwIHx8IHZhbHVlID4gMSkge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICAgYFRlbnNvciB2YWx1ZXMgZm9yIGEgZmxvYXQzMiBUZW5zb3IgbXVzdCBiZSBpbiB0aGUgYCArXG4gICAgICAgICAgICAgIGByYW5nZSBbMCAtIDFdIGJ1dCBlbmNvdW50ZXJlZCAke3ZhbHVlfS5gKTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIGlmICgkaW1nLmR0eXBlID09PSAnaW50MzInKSB7XG4gICAgICAgIGlmICh2YWx1ZSA8IDAgfHwgdmFsdWUgPiAyNTUpIHtcbiAgICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAgIGBUZW5zb3IgdmFsdWVzIGZvciBhIGludDMyIFRlbnNvciBtdXN0IGJlIGluIHRoZSBgICtcbiAgICAgICAgICAgICAgYHJhbmdlIFswIC0gMjU1XSBidXQgZW5jb3VudGVyZWQgJHt2YWx1ZX0uYCk7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgaWYgKGRlcHRoID09PSAxKSB7XG4gICAgICAgIHJnYmFbMF0gPSB2YWx1ZSAqIG11bHRpcGxpZXI7XG4gICAgICAgIHJnYmFbMV0gPSB2YWx1ZSAqIG11bHRpcGxpZXI7XG4gICAgICAgIHJnYmFbMl0gPSB2YWx1ZSAqIG11bHRpcGxpZXI7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZ2JhW2RdID0gdmFsdWUgKiBtdWx0aXBsaWVyO1xuICAgICAgfVxuICAgIH1cblxuICAgIGNvbnN0IGogPSBpICogNDtcbiAgICBieXRlc1tqICsgMF0gPSBNYXRoLnJvdW5kKHJnYmFbMF0pO1xuICAgIGJ5dGVzW2ogKyAxXSA9IE1hdGgucm91bmQocmdiYVsxXSk7XG4gICAgYnl0ZXNbaiArIDJdID0gTWF0aC5yb3VuZChyZ2JhWzJdKTtcbiAgICBieXRlc1tqICsgM10gPSBNYXRoLnJvdW5kKHJnYmFbM10pO1xuICB9XG5cbiAgaWYgKGNhbnZhcyAhPSBudWxsKSB7XG4gICAgY2FudmFzLndpZHRoID0gd2lkdGg7XG4gICAgY2FudmFzLmhlaWdodCA9IGhlaWdodDtcbiAgICBjb25zdCBjdHggPSBjYW52YXMuZ2V0Q29udGV4dCgnMmQnKTtcbiAgICBjb25zdCBpbWFnZURhdGEgPSBuZXcgSW1hZ2VEYXRhKGJ5dGVzLCB3aWR0aCwgaGVpZ2h0KTtcbiAgICBjdHgucHV0SW1hZ2VEYXRhKGltYWdlRGF0YSwgMCwgMCk7XG4gIH1cbiAgaWYgKCRpbWcgIT09IGltZykge1xuICAgICRpbWcuZGlzcG9zZSgpO1xuICB9XG4gIHJldHVybiBieXRlcztcbn1cblxuZXhwb3J0IGNvbnN0IGZyb21QaXhlbHMgPSBvcCh7ZnJvbVBpeGVsc199KTtcbiJdfQ==