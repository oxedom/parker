/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
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
import { Transform } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { op } from '../operation';
/**
 * Applies the given transform(s) to the image(s).
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 * @param transforms Projective transform matrix/matrices. A tensor1d of length
 *     8 or tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0
 *     b1, b2, c0, c1], then it maps the output point (x, y) to a transformed
 *     input point (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
 *     where k = c0 x + c1 y + 1. The transforms are inverted compared to the
 *     transform mapping input points to output points.
 * @param interpolation Interpolation mode.
 *     Supported values: 'nearest', 'bilinear'. Default to 'nearest'.
 * @param fillMode Points outside the boundaries of the input are filled
 *     according to the given mode, one of 'constant', 'reflect', 'wrap',
 *     'nearest'. Default to 'constant'.
 *     'reflect': (d c b a | a b c d | d c b a ) The input is extended by
 *     reflecting about the edge of the last pixel.
 *     'constant': (k k k k | a b c d | k k k k) The input is extended by
 *     filling all values beyond the edge with the same constant value k.
 *     'wrap': (a b c d | a b c d | a b c d) The input is extended by
 *     wrapping around to the opposite edge.
 *     'nearest': (a a a a | a b c d | d d d d) The input is extended by
 *     the nearest pixel.
 * @param fillValue A float represents the value to be filled outside the
 *     boundaries when fillMode is 'constant'.
 * @param Output dimension after the transform, [height, width]. If undefined,
 *     output is the same size as input image.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function transform_(image, transforms, interpolation = 'nearest', fillMode = 'constant', fillValue = 0, outputShape) {
    const $image = convertToTensor(image, 'image', 'transform', 'float32');
    const $transforms = convertToTensor(transforms, 'transforms', 'transform', 'float32');
    util.assert($image.rank === 4, () => 'Error in transform: image must be rank 4,' +
        `but got rank ${$image.rank}.`);
    util.assert($transforms.rank === 2 &&
        ($transforms.shape[0] === $image.shape[0] ||
            $transforms.shape[0] === 1) &&
        $transforms.shape[1] === 8, () => `Error in transform: Input transform should be batch x 8 or 1 x 8`);
    util.assert(outputShape == null || outputShape.length === 2, () => 'Error in transform: outputShape must be [height, width] or null, ' +
        `but got ${outputShape}.`);
    const inputs = { image: $image, transforms: $transforms };
    const attrs = { interpolation, fillMode, fillValue, outputShape };
    return ENGINE.runKernel(Transform, inputs, attrs);
}
export const transform = op({ transform_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhbnNmb3JtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvaW1hZ2UvdHJhbnNmb3JtLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLFNBQVMsRUFBa0MsTUFBTSxvQkFBb0IsQ0FBQztBQUk5RSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxLQUFLLElBQUksTUFBTSxZQUFZLENBQUM7QUFFbkMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUVoQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0E2Qkc7QUFDSCxTQUFTLFVBQVUsQ0FDZixLQUEwQixFQUFFLFVBQStCLEVBQzNELGdCQUFzQyxTQUFTLEVBQy9DLFdBQWtELFVBQVUsRUFBRSxTQUFTLEdBQUcsQ0FBQyxFQUMzRSxXQUE4QjtJQUNoQyxNQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxXQUFXLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDdkUsTUFBTSxXQUFXLEdBQ2IsZUFBZSxDQUFDLFVBQVUsRUFBRSxZQUFZLEVBQUUsV0FBVyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBRXRFLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLEdBQUcsRUFBRSxDQUFDLDJDQUEyQztRQUM3QyxnQkFBZ0IsTUFBTSxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFFeEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxXQUFXLENBQUMsSUFBSSxLQUFLLENBQUM7UUFDbEIsQ0FBQyxXQUFXLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO1lBQ3hDLFdBQVcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQzVCLFdBQVcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxFQUM5QixHQUFHLEVBQUUsQ0FBQyxrRUFBa0UsQ0FBQyxDQUFDO0lBRTlFLElBQUksQ0FBQyxNQUFNLENBQ1AsV0FBVyxJQUFJLElBQUksSUFBSSxXQUFXLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDL0MsR0FBRyxFQUFFLENBQ0QsbUVBQW1FO1FBQ25FLFdBQVcsV0FBVyxHQUFHLENBQUMsQ0FBQztJQUVuQyxNQUFNLE1BQU0sR0FBb0IsRUFBQyxLQUFLLEVBQUUsTUFBTSxFQUFFLFVBQVUsRUFBRSxXQUFXLEVBQUMsQ0FBQztJQUN6RSxNQUFNLEtBQUssR0FDVSxFQUFDLGFBQWEsRUFBRSxRQUFRLEVBQUUsU0FBUyxFQUFFLFdBQVcsRUFBQyxDQUFDO0lBRXZFLE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsU0FBUyxFQUFFLE1BQThCLEVBQUUsS0FBMkIsQ0FBQyxDQUFDO0FBQzlFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQUcsRUFBRSxDQUFDLEVBQUMsVUFBVSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtUcmFuc2Zvcm0sIFRyYW5zZm9ybUF0dHJzLCBUcmFuc2Zvcm1JbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4vLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yMkQsIFRlbnNvcjREfSBmcm9tICcuLi8uLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi8uLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi8uLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uLy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIEFwcGxpZXMgdGhlIGdpdmVuIHRyYW5zZm9ybShzKSB0byB0aGUgaW1hZ2UocykuXG4gKlxuICogQHBhcmFtIGltYWdlIDRkIHRlbnNvciBvZiBzaGFwZSBgW2JhdGNoLCBpbWFnZUhlaWdodCwgaW1hZ2VXaWR0aCwgZGVwdGhdYC5cbiAqIEBwYXJhbSB0cmFuc2Zvcm1zIFByb2plY3RpdmUgdHJhbnNmb3JtIG1hdHJpeC9tYXRyaWNlcy4gQSB0ZW5zb3IxZCBvZiBsZW5ndGhcbiAqICAgICA4IG9yIHRlbnNvciBvZiBzaXplIE4geCA4LiBJZiBvbmUgcm93IG9mIHRyYW5zZm9ybXMgaXMgW2EwLCBhMSwgYTIsIGIwXG4gKiAgICAgYjEsIGIyLCBjMCwgYzFdLCB0aGVuIGl0IG1hcHMgdGhlIG91dHB1dCBwb2ludCAoeCwgeSkgdG8gYSB0cmFuc2Zvcm1lZFxuICogICAgIGlucHV0IHBvaW50ICh4JywgeScpID0gKChhMCB4ICsgYTEgeSArIGEyKSAvIGssIChiMCB4ICsgYjEgeSArIGIyKSAvIGspLFxuICogICAgIHdoZXJlIGsgPSBjMCB4ICsgYzEgeSArIDEuIFRoZSB0cmFuc2Zvcm1zIGFyZSBpbnZlcnRlZCBjb21wYXJlZCB0byB0aGVcbiAqICAgICB0cmFuc2Zvcm0gbWFwcGluZyBpbnB1dCBwb2ludHMgdG8gb3V0cHV0IHBvaW50cy5cbiAqIEBwYXJhbSBpbnRlcnBvbGF0aW9uIEludGVycG9sYXRpb24gbW9kZS5cbiAqICAgICBTdXBwb3J0ZWQgdmFsdWVzOiAnbmVhcmVzdCcsICdiaWxpbmVhcicuIERlZmF1bHQgdG8gJ25lYXJlc3QnLlxuICogQHBhcmFtIGZpbGxNb2RlIFBvaW50cyBvdXRzaWRlIHRoZSBib3VuZGFyaWVzIG9mIHRoZSBpbnB1dCBhcmUgZmlsbGVkXG4gKiAgICAgYWNjb3JkaW5nIHRvIHRoZSBnaXZlbiBtb2RlLCBvbmUgb2YgJ2NvbnN0YW50JywgJ3JlZmxlY3QnLCAnd3JhcCcsXG4gKiAgICAgJ25lYXJlc3QnLiBEZWZhdWx0IHRvICdjb25zdGFudCcuXG4gKiAgICAgJ3JlZmxlY3QnOiAoZCBjIGIgYSB8IGEgYiBjIGQgfCBkIGMgYiBhICkgVGhlIGlucHV0IGlzIGV4dGVuZGVkIGJ5XG4gKiAgICAgcmVmbGVjdGluZyBhYm91dCB0aGUgZWRnZSBvZiB0aGUgbGFzdCBwaXhlbC5cbiAqICAgICAnY29uc3RhbnQnOiAoayBrIGsgayB8IGEgYiBjIGQgfCBrIGsgayBrKSBUaGUgaW5wdXQgaXMgZXh0ZW5kZWQgYnlcbiAqICAgICBmaWxsaW5nIGFsbCB2YWx1ZXMgYmV5b25kIHRoZSBlZGdlIHdpdGggdGhlIHNhbWUgY29uc3RhbnQgdmFsdWUgay5cbiAqICAgICAnd3JhcCc6IChhIGIgYyBkIHwgYSBiIGMgZCB8IGEgYiBjIGQpIFRoZSBpbnB1dCBpcyBleHRlbmRlZCBieVxuICogICAgIHdyYXBwaW5nIGFyb3VuZCB0byB0aGUgb3Bwb3NpdGUgZWRnZS5cbiAqICAgICAnbmVhcmVzdCc6IChhIGEgYSBhIHwgYSBiIGMgZCB8IGQgZCBkIGQpIFRoZSBpbnB1dCBpcyBleHRlbmRlZCBieVxuICogICAgIHRoZSBuZWFyZXN0IHBpeGVsLlxuICogQHBhcmFtIGZpbGxWYWx1ZSBBIGZsb2F0IHJlcHJlc2VudHMgdGhlIHZhbHVlIHRvIGJlIGZpbGxlZCBvdXRzaWRlIHRoZVxuICogICAgIGJvdW5kYXJpZXMgd2hlbiBmaWxsTW9kZSBpcyAnY29uc3RhbnQnLlxuICogQHBhcmFtIE91dHB1dCBkaW1lbnNpb24gYWZ0ZXIgdGhlIHRyYW5zZm9ybSwgW2hlaWdodCwgd2lkdGhdLiBJZiB1bmRlZmluZWQsXG4gKiAgICAgb3V0cHV0IGlzIHRoZSBzYW1lIHNpemUgYXMgaW5wdXQgaW1hZ2UuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnSW1hZ2VzJywgbmFtZXNwYWNlOiAnaW1hZ2UnfVxuICovXG5mdW5jdGlvbiB0cmFuc2Zvcm1fKFxuICAgIGltYWdlOiBUZW5zb3I0RHxUZW5zb3JMaWtlLCB0cmFuc2Zvcm1zOiBUZW5zb3IyRHxUZW5zb3JMaWtlLFxuICAgIGludGVycG9sYXRpb246ICduZWFyZXN0J3wnYmlsaW5lYXInID0gJ25lYXJlc3QnLFxuICAgIGZpbGxNb2RlOiAnY29uc3RhbnQnfCdyZWZsZWN0J3wnd3JhcCd8J25lYXJlc3QnID0gJ2NvbnN0YW50JywgZmlsbFZhbHVlID0gMCxcbiAgICBvdXRwdXRTaGFwZT86IFtudW1iZXIsIG51bWJlcl0pOiBUZW5zb3I0RCB7XG4gIGNvbnN0ICRpbWFnZSA9IGNvbnZlcnRUb1RlbnNvcihpbWFnZSwgJ2ltYWdlJywgJ3RyYW5zZm9ybScsICdmbG9hdDMyJyk7XG4gIGNvbnN0ICR0cmFuc2Zvcm1zID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcih0cmFuc2Zvcm1zLCAndHJhbnNmb3JtcycsICd0cmFuc2Zvcm0nLCAnZmxvYXQzMicpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJGltYWdlLnJhbmsgPT09IDQsXG4gICAgICAoKSA9PiAnRXJyb3IgaW4gdHJhbnNmb3JtOiBpbWFnZSBtdXN0IGJlIHJhbmsgNCwnICtcbiAgICAgICAgICBgYnV0IGdvdCByYW5rICR7JGltYWdlLnJhbmt9LmApO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHRyYW5zZm9ybXMucmFuayA9PT0gMiAmJlxuICAgICAgICAgICgkdHJhbnNmb3Jtcy5zaGFwZVswXSA9PT0gJGltYWdlLnNoYXBlWzBdIHx8XG4gICAgICAgICAgICR0cmFuc2Zvcm1zLnNoYXBlWzBdID09PSAxKSAmJlxuICAgICAgICAgICR0cmFuc2Zvcm1zLnNoYXBlWzFdID09PSA4LFxuICAgICAgKCkgPT4gYEVycm9yIGluIHRyYW5zZm9ybTogSW5wdXQgdHJhbnNmb3JtIHNob3VsZCBiZSBiYXRjaCB4IDggb3IgMSB4IDhgKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgIG91dHB1dFNoYXBlID09IG51bGwgfHwgb3V0cHV0U2hhcGUubGVuZ3RoID09PSAyLFxuICAgICAgKCkgPT5cbiAgICAgICAgICAnRXJyb3IgaW4gdHJhbnNmb3JtOiBvdXRwdXRTaGFwZSBtdXN0IGJlIFtoZWlnaHQsIHdpZHRoXSBvciBudWxsLCAnICtcbiAgICAgICAgICBgYnV0IGdvdCAke291dHB1dFNoYXBlfS5gKTtcblxuICBjb25zdCBpbnB1dHM6IFRyYW5zZm9ybUlucHV0cyA9IHtpbWFnZTogJGltYWdlLCB0cmFuc2Zvcm1zOiAkdHJhbnNmb3Jtc307XG4gIGNvbnN0IGF0dHJzOlxuICAgICAgVHJhbnNmb3JtQXR0cnMgPSB7aW50ZXJwb2xhdGlvbiwgZmlsbE1vZGUsIGZpbGxWYWx1ZSwgb3V0cHV0U2hhcGV9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgVHJhbnNmb3JtLCBpbnB1dHMgYXMge30gYXMgTmFtZWRUZW5zb3JNYXAsIGF0dHJzIGFzIHt9IGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCB0cmFuc2Zvcm0gPSBvcCh7dHJhbnNmb3JtX30pO1xuIl19