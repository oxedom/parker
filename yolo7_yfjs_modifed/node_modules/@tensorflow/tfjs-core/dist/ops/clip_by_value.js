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
import { ENGINE } from '../engine';
import { ClipByValue } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Clips values element-wise. `max(min(x, clipValueMax), clipValueMin)`
 *
 * ```js
 * const x = tf.tensor1d([-1, 2, -3, 4]);
 *
 * x.clipByValue(-2, 3).print();  // or tf.clipByValue(x, -2, 3)
 * ```
 * @param x The input tensor.
 * @param clipValueMin Lower-bound of range to be clipped to.
 * @param clipValueMax Upper-bound of range to be clipped to.
 *
 * @doc {heading: 'Operations', subheading: 'Basic math'}
 */
function clipByValue_(x, clipValueMin, clipValueMax) {
    const $x = convertToTensor(x, 'x', 'clipByValue');
    util.assert((clipValueMin <= clipValueMax), () => `Error in clip: min (${clipValueMin}) must be ` +
        `less than or equal to max (${clipValueMax}).`);
    const inputs = { x: $x };
    const attrs = { clipValueMin, clipValueMax };
    return ENGINE.runKernel(ClipByValue, inputs, attrs);
}
export const clipByValue = op({ clipByValue_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY2xpcF9ieV92YWx1ZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2NsaXBfYnlfdmFsdWUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsV0FBVyxFQUFzQyxNQUFNLGlCQUFpQixDQUFDO0FBSWpGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7O0dBYUc7QUFDSCxTQUFTLFlBQVksQ0FDakIsQ0FBZSxFQUFFLFlBQW9CLEVBQUUsWUFBb0I7SUFDN0QsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsYUFBYSxDQUFDLENBQUM7SUFDbEQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLFlBQVksSUFBSSxZQUFZLENBQUMsRUFDOUIsR0FBRyxFQUFFLENBQUMsdUJBQXVCLFlBQVksWUFBWTtRQUNqRCw4QkFBOEIsWUFBWSxJQUFJLENBQUMsQ0FBQztJQUV4RCxNQUFNLE1BQU0sR0FBc0IsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDMUMsTUFBTSxLQUFLLEdBQXFCLEVBQUMsWUFBWSxFQUFFLFlBQVksRUFBQyxDQUFDO0lBRTdELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsV0FBVyxFQUFFLE1BQThCLEVBQUUsS0FBMkIsQ0FBQyxDQUFDO0FBQ2hGLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxXQUFXLEdBQUcsRUFBRSxDQUFDLEVBQUMsWUFBWSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtDbGlwQnlWYWx1ZSwgQ2xpcEJ5VmFsdWVBdHRycywgQ2xpcEJ5VmFsdWVJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogQ2xpcHMgdmFsdWVzIGVsZW1lbnQtd2lzZS4gYG1heChtaW4oeCwgY2xpcFZhbHVlTWF4KSwgY2xpcFZhbHVlTWluKWBcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFstMSwgMiwgLTMsIDRdKTtcbiAqXG4gKiB4LmNsaXBCeVZhbHVlKC0yLCAzKS5wcmludCgpOyAgLy8gb3IgdGYuY2xpcEJ5VmFsdWUoeCwgLTIsIDMpXG4gKiBgYGBcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IuXG4gKiBAcGFyYW0gY2xpcFZhbHVlTWluIExvd2VyLWJvdW5kIG9mIHJhbmdlIHRvIGJlIGNsaXBwZWQgdG8uXG4gKiBAcGFyYW0gY2xpcFZhbHVlTWF4IFVwcGVyLWJvdW5kIG9mIHJhbmdlIHRvIGJlIGNsaXBwZWQgdG8uXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnQmFzaWMgbWF0aCd9XG4gKi9cbmZ1bmN0aW9uIGNsaXBCeVZhbHVlXzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICB4OiBUfFRlbnNvckxpa2UsIGNsaXBWYWx1ZU1pbjogbnVtYmVyLCBjbGlwVmFsdWVNYXg6IG51bWJlcik6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdjbGlwQnlWYWx1ZScpO1xuICB1dGlsLmFzc2VydChcbiAgICAgIChjbGlwVmFsdWVNaW4gPD0gY2xpcFZhbHVlTWF4KSxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjbGlwOiBtaW4gKCR7Y2xpcFZhbHVlTWlufSkgbXVzdCBiZSBgICtcbiAgICAgICAgICBgbGVzcyB0aGFuIG9yIGVxdWFsIHRvIG1heCAoJHtjbGlwVmFsdWVNYXh9KS5gKTtcblxuICBjb25zdCBpbnB1dHM6IENsaXBCeVZhbHVlSW5wdXRzID0ge3g6ICR4fTtcbiAgY29uc3QgYXR0cnM6IENsaXBCeVZhbHVlQXR0cnMgPSB7Y2xpcFZhbHVlTWluLCBjbGlwVmFsdWVNYXh9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgQ2xpcEJ5VmFsdWUsIGlucHV0cyBhcyB7fSBhcyBOYW1lZFRlbnNvck1hcCwgYXR0cnMgYXMge30gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGNsaXBCeVZhbHVlID0gb3Aoe2NsaXBCeVZhbHVlX30pO1xuIl19