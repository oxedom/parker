/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { ENGINE } from '../../engine';
import { ResizeNearestNeighbor } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { op } from '../operation';
import { reshape } from '../reshape';
/**
 * NearestNeighbor resize a batch of 3D images to a new shape.
 *
 * @param images The images, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param size The new shape `[newHeight, newWidth]` to resize the
 *     images to. Each channel is resized individually.
 * @param alignCorners Defaults to False. If true, rescale
 *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
 *     corners of images and resized images. If false, rescale by
 *     `new_height / height`. Treat similarly the width dimension.
 * @param halfPixelCenters Defaults to `false`. Whether to assumes pixels are of
 *      half the actual dimensions, and yields more accurate resizes. This flag
 *      would also make the floating point coordinates of the top left pixel
 *      0.5, 0.5.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function resizeNearestNeighbor_(images, size, alignCorners = false, halfPixelCenters = false) {
    const $images = convertToTensor(images, 'images', 'resizeNearestNeighbor');
    util.assert($images.rank === 3 || $images.rank === 4, () => `Error in resizeNearestNeighbor: x must be rank 3 or 4, but got ` +
        `rank ${$images.rank}.`);
    util.assert(size.length === 2, () => `Error in resizeNearestNeighbor: new shape must 2D, but got shape ` +
        `${size}.`);
    util.assert($images.dtype === 'float32' || $images.dtype === 'int32', () => '`images` must have `int32` or `float32` as dtype');
    util.assert(halfPixelCenters === false || alignCorners === false, () => `Error in resizeNearestNeighbor: If halfPixelCenters is true, ` +
        `alignCorners must be false.`);
    let batchImages = $images;
    let reshapedTo4D = false;
    if ($images.rank === 3) {
        reshapedTo4D = true;
        batchImages = reshape($images, [1, $images.shape[0], $images.shape[1], $images.shape[2]]);
    }
    const [] = size;
    const inputs = { images: batchImages };
    const attrs = { alignCorners, halfPixelCenters, size };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(ResizeNearestNeighbor, inputs, attrs);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
export const resizeNearestNeighbor = op({ resizeNearestNeighbor_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVzaXplX25lYXJlc3RfbmVpZ2hib3IuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9pbWFnZS9yZXNpemVfbmVhcmVzdF9uZWlnaGJvci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ3BDLE9BQU8sRUFBQyxxQkFBcUIsRUFBMEQsTUFBTSxvQkFBb0IsQ0FBQztBQUlsSCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxLQUFLLElBQUksTUFBTSxZQUFZLENBQUM7QUFFbkMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRW5DOzs7Ozs7Ozs7Ozs7Ozs7OztHQWlCRztBQUNILFNBQVMsc0JBQXNCLENBQzNCLE1BQW9CLEVBQUUsSUFBc0IsRUFBRSxZQUFZLEdBQUcsS0FBSyxFQUNsRSxnQkFBZ0IsR0FBRyxLQUFLO0lBQzFCLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLHVCQUF1QixDQUFDLENBQUM7SUFFM0UsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDeEMsR0FBRyxFQUFFLENBQUMsaUVBQWlFO1FBQ25FLFFBQVEsT0FBTyxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDakMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDakIsR0FBRyxFQUFFLENBQ0QsbUVBQW1FO1FBQ25FLEdBQUcsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUNwQixJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxLQUFLLEtBQUssU0FBUyxJQUFJLE9BQU8sQ0FBQyxLQUFLLEtBQUssT0FBTyxFQUN4RCxHQUFHLEVBQUUsQ0FBQyxrREFBa0QsQ0FBQyxDQUFDO0lBQzlELElBQUksQ0FBQyxNQUFNLENBQ1AsZ0JBQWdCLEtBQUssS0FBSyxJQUFJLFlBQVksS0FBSyxLQUFLLEVBQ3BELEdBQUcsRUFBRSxDQUFDLCtEQUErRDtRQUNqRSw2QkFBNkIsQ0FBQyxDQUFDO0lBQ3ZDLElBQUksV0FBVyxHQUFHLE9BQW1CLENBQUM7SUFDdEMsSUFBSSxZQUFZLEdBQUcsS0FBSyxDQUFDO0lBQ3pCLElBQUksT0FBTyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDdEIsWUFBWSxHQUFHLElBQUksQ0FBQztRQUNwQixXQUFXLEdBQUcsT0FBTyxDQUNqQixPQUFPLEVBQUUsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQ3pFO0lBQ0QsTUFBTSxFQUFFLEdBQUcsSUFBSSxDQUFDO0lBRWhCLE1BQU0sTUFBTSxHQUFnQyxFQUFDLE1BQU0sRUFBRSxXQUFXLEVBQUMsQ0FBQztJQUNsRSxNQUFNLEtBQUssR0FDc0IsRUFBQyxZQUFZLEVBQUUsZ0JBQWdCLEVBQUUsSUFBSSxFQUFDLENBQUM7SUFFeEUsMERBQTBEO0lBQzFELE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ1oscUJBQXFCLEVBQUUsTUFBOEIsRUFDckQsS0FBMkIsQ0FBTSxDQUFDO0lBRWxELElBQUksWUFBWSxFQUFFO1FBQ2hCLE9BQU8sT0FBTyxDQUFDLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQU0sQ0FBQztLQUN0RTtJQUNELE9BQU8sR0FBRyxDQUFDO0FBQ2IsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLHFCQUFxQixHQUFHLEVBQUUsQ0FBQyxFQUFDLHNCQUFzQixFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uLy4uL2VuZ2luZSc7XG5pbXBvcnQge1Jlc2l6ZU5lYXJlc3ROZWlnaGJvciwgUmVzaXplTmVhcmVzdE5laWdoYm9yQXR0cnMsIFJlc2l6ZU5lYXJlc3ROZWlnaGJvcklucHV0c30gZnJvbSAnLi4vLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi8uLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3IzRCwgVGVuc29yNER9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi8uLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uLy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4uL3Jlc2hhcGUnO1xuXG4vKipcbiAqIE5lYXJlc3ROZWlnaGJvciByZXNpemUgYSBiYXRjaCBvZiAzRCBpbWFnZXMgdG8gYSBuZXcgc2hhcGUuXG4gKlxuICogQHBhcmFtIGltYWdlcyBUaGUgaW1hZ2VzLCBvZiByYW5rIDQgb3IgcmFuayAzLCBvZiBzaGFwZVxuICogICAgIGBbYmF0Y2gsIGhlaWdodCwgd2lkdGgsIGluQ2hhbm5lbHNdYC4gSWYgcmFuayAzLCBiYXRjaCBvZiAxIGlzIGFzc3VtZWQuXG4gKiBAcGFyYW0gc2l6ZSBUaGUgbmV3IHNoYXBlIGBbbmV3SGVpZ2h0LCBuZXdXaWR0aF1gIHRvIHJlc2l6ZSB0aGVcbiAqICAgICBpbWFnZXMgdG8uIEVhY2ggY2hhbm5lbCBpcyByZXNpemVkIGluZGl2aWR1YWxseS5cbiAqIEBwYXJhbSBhbGlnbkNvcm5lcnMgRGVmYXVsdHMgdG8gRmFsc2UuIElmIHRydWUsIHJlc2NhbGVcbiAqICAgICBpbnB1dCBieSBgKG5ld19oZWlnaHQgLSAxKSAvIChoZWlnaHQgLSAxKWAsIHdoaWNoIGV4YWN0bHkgYWxpZ25zIHRoZSA0XG4gKiAgICAgY29ybmVycyBvZiBpbWFnZXMgYW5kIHJlc2l6ZWQgaW1hZ2VzLiBJZiBmYWxzZSwgcmVzY2FsZSBieVxuICogICAgIGBuZXdfaGVpZ2h0IC8gaGVpZ2h0YC4gVHJlYXQgc2ltaWxhcmx5IHRoZSB3aWR0aCBkaW1lbnNpb24uXG4gKiBAcGFyYW0gaGFsZlBpeGVsQ2VudGVycyBEZWZhdWx0cyB0byBgZmFsc2VgLiBXaGV0aGVyIHRvIGFzc3VtZXMgcGl4ZWxzIGFyZSBvZlxuICogICAgICBoYWxmIHRoZSBhY3R1YWwgZGltZW5zaW9ucywgYW5kIHlpZWxkcyBtb3JlIGFjY3VyYXRlIHJlc2l6ZXMuIFRoaXMgZmxhZ1xuICogICAgICB3b3VsZCBhbHNvIG1ha2UgdGhlIGZsb2F0aW5nIHBvaW50IGNvb3JkaW5hdGVzIG9mIHRoZSB0b3AgbGVmdCBwaXhlbFxuICogICAgICAwLjUsIDAuNS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdJbWFnZXMnLCBuYW1lc3BhY2U6ICdpbWFnZSd9XG4gKi9cbmZ1bmN0aW9uIHJlc2l6ZU5lYXJlc3ROZWlnaGJvcl88VCBleHRlbmRzIFRlbnNvcjNEfFRlbnNvcjREPihcbiAgICBpbWFnZXM6IFR8VGVuc29yTGlrZSwgc2l6ZTogW251bWJlciwgbnVtYmVyXSwgYWxpZ25Db3JuZXJzID0gZmFsc2UsXG4gICAgaGFsZlBpeGVsQ2VudGVycyA9IGZhbHNlKTogVCB7XG4gIGNvbnN0ICRpbWFnZXMgPSBjb252ZXJ0VG9UZW5zb3IoaW1hZ2VzLCAnaW1hZ2VzJywgJ3Jlc2l6ZU5lYXJlc3ROZWlnaGJvcicpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJGltYWdlcy5yYW5rID09PSAzIHx8ICRpbWFnZXMucmFuayA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiByZXNpemVOZWFyZXN0TmVpZ2hib3I6IHggbXVzdCBiZSByYW5rIDMgb3IgNCwgYnV0IGdvdCBgICtcbiAgICAgICAgICBgcmFuayAkeyRpbWFnZXMucmFua30uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgc2l6ZS5sZW5ndGggPT09IDIsXG4gICAgICAoKSA9PlxuICAgICAgICAgIGBFcnJvciBpbiByZXNpemVOZWFyZXN0TmVpZ2hib3I6IG5ldyBzaGFwZSBtdXN0IDJELCBidXQgZ290IHNoYXBlIGAgK1xuICAgICAgICAgIGAke3NpemV9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgICRpbWFnZXMuZHR5cGUgPT09ICdmbG9hdDMyJyB8fCAkaW1hZ2VzLmR0eXBlID09PSAnaW50MzInLFxuICAgICAgKCkgPT4gJ2BpbWFnZXNgIG11c3QgaGF2ZSBgaW50MzJgIG9yIGBmbG9hdDMyYCBhcyBkdHlwZScpO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGhhbGZQaXhlbENlbnRlcnMgPT09IGZhbHNlIHx8IGFsaWduQ29ybmVycyA9PT0gZmFsc2UsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gcmVzaXplTmVhcmVzdE5laWdoYm9yOiBJZiBoYWxmUGl4ZWxDZW50ZXJzIGlzIHRydWUsIGAgK1xuICAgICAgICAgIGBhbGlnbkNvcm5lcnMgbXVzdCBiZSBmYWxzZS5gKTtcbiAgbGV0IGJhdGNoSW1hZ2VzID0gJGltYWdlcyBhcyBUZW5zb3I0RDtcbiAgbGV0IHJlc2hhcGVkVG80RCA9IGZhbHNlO1xuICBpZiAoJGltYWdlcy5yYW5rID09PSAzKSB7XG4gICAgcmVzaGFwZWRUbzREID0gdHJ1ZTtcbiAgICBiYXRjaEltYWdlcyA9IHJlc2hhcGUoXG4gICAgICAgICRpbWFnZXMsIFsxLCAkaW1hZ2VzLnNoYXBlWzBdLCAkaW1hZ2VzLnNoYXBlWzFdLCAkaW1hZ2VzLnNoYXBlWzJdXSk7XG4gIH1cbiAgY29uc3QgW10gPSBzaXplO1xuXG4gIGNvbnN0IGlucHV0czogUmVzaXplTmVhcmVzdE5laWdoYm9ySW5wdXRzID0ge2ltYWdlczogYmF0Y2hJbWFnZXN9O1xuICBjb25zdCBhdHRyczpcbiAgICAgIFJlc2l6ZU5lYXJlc3ROZWlnaGJvckF0dHJzID0ge2FsaWduQ29ybmVycywgaGFsZlBpeGVsQ2VudGVycywgc2l6ZX07XG5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICBjb25zdCByZXMgPSBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICAgICAgICAgICAgUmVzaXplTmVhcmVzdE5laWdoYm9yLCBpbnB1dHMgYXMge30gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgICAgICBhdHRycyBhcyB7fSBhcyBOYW1lZEF0dHJNYXApIGFzIFQ7XG5cbiAgaWYgKHJlc2hhcGVkVG80RCkge1xuICAgIHJldHVybiByZXNoYXBlKHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM11dKSBhcyBUO1xuICB9XG4gIHJldHVybiByZXM7XG59XG5cbmV4cG9ydCBjb25zdCByZXNpemVOZWFyZXN0TmVpZ2hib3IgPSBvcCh7cmVzaXplTmVhcmVzdE5laWdoYm9yX30pO1xuIl19