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
import { ENGINE } from '../engine';
import { Tile } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Construct a tensor by repeating it the number of times given by reps.
 *
 * This operation creates a new tensor by replicating `input` `reps`
 * times. The output tensor's i'th dimension has `input.shape[i] *
 * reps[i]` elements, and the values of `input` are replicated
 * `reps[i]` times along the i'th dimension. For example, tiling
 * `[a, b, c, d]` by `[2]` produces `[a, b, c, d, a, b, c, d]`.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 *
 * a.tile([2]).print();    // or a.tile([2])
 * ```
 *
 * ```js
 * const a = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * a.tile([1, 2]).print();  // or a.tile([1, 2])
 * ```
 * @param x The tensor to tile.
 * @param reps Determines the number of replications per dimension.
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function tile_(x, reps) {
    const $x = convertToTensor(x, 'x', 'tile', 'string_or_numeric');
    util.assert($x.rank === reps.length, () => `Error in transpose: rank of input ${$x.rank} ` +
        `must match length of reps ${reps}.`);
    const inputs = { x: $x };
    const attrs = { reps };
    return ENGINE.runKernel(Tile, inputs, attrs);
}
export const tile = op({ tile_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGlsZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3RpbGUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsSUFBSSxFQUF3QixNQUFNLGlCQUFpQixDQUFDO0FBSTVELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0F3Qkc7QUFDSCxTQUFTLEtBQUssQ0FBbUIsQ0FBZSxFQUFFLElBQWM7SUFDOUQsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsTUFBTSxFQUFFLG1CQUFtQixDQUFDLENBQUM7SUFDaEUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxLQUFLLElBQUksQ0FBQyxNQUFNLEVBQ3ZCLEdBQUcsRUFBRSxDQUFDLHFDQUFxQyxFQUFFLENBQUMsSUFBSSxHQUFHO1FBQ2pELDZCQUE2QixJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBRTlDLE1BQU0sTUFBTSxHQUFlLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQ25DLE1BQU0sS0FBSyxHQUFjLEVBQUMsSUFBSSxFQUFDLENBQUM7SUFFaEMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixJQUFJLEVBQUUsTUFBbUMsRUFDekMsS0FBZ0MsQ0FBQyxDQUFDO0FBQ3hDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxJQUFJLEdBQUcsRUFBRSxDQUFDLEVBQUMsS0FBSyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1RpbGUsIFRpbGVBdHRycywgVGlsZUlucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb25zdHJ1Y3QgYSB0ZW5zb3IgYnkgcmVwZWF0aW5nIGl0IHRoZSBudW1iZXIgb2YgdGltZXMgZ2l2ZW4gYnkgcmVwcy5cbiAqXG4gKiBUaGlzIG9wZXJhdGlvbiBjcmVhdGVzIGEgbmV3IHRlbnNvciBieSByZXBsaWNhdGluZyBgaW5wdXRgIGByZXBzYFxuICogdGltZXMuIFRoZSBvdXRwdXQgdGVuc29yJ3MgaSd0aCBkaW1lbnNpb24gaGFzIGBpbnB1dC5zaGFwZVtpXSAqXG4gKiByZXBzW2ldYCBlbGVtZW50cywgYW5kIHRoZSB2YWx1ZXMgb2YgYGlucHV0YCBhcmUgcmVwbGljYXRlZFxuICogYHJlcHNbaV1gIHRpbWVzIGFsb25nIHRoZSBpJ3RoIGRpbWVuc2lvbi4gRm9yIGV4YW1wbGUsIHRpbGluZ1xuICogYFthLCBiLCBjLCBkXWAgYnkgYFsyXWAgcHJvZHVjZXMgYFthLCBiLCBjLCBkLCBhLCBiLCBjLCBkXWAuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgMl0pO1xuICpcbiAqIGEudGlsZShbMl0pLnByaW50KCk7ICAgIC8vIG9yIGEudGlsZShbMl0pXG4gKiBgYGBcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCA0XSwgWzIsIDJdKTtcbiAqXG4gKiBhLnRpbGUoWzEsIDJdKS5wcmludCgpOyAgLy8gb3IgYS50aWxlKFsxLCAyXSlcbiAqIGBgYFxuICogQHBhcmFtIHggVGhlIHRlbnNvciB0byB0aWxlLlxuICogQHBhcmFtIHJlcHMgRGV0ZXJtaW5lcyB0aGUgbnVtYmVyIG9mIHJlcGxpY2F0aW9ucyBwZXIgZGltZW5zaW9uLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ1NsaWNpbmcgYW5kIEpvaW5pbmcnfVxuICovXG5mdW5jdGlvbiB0aWxlXzxUIGV4dGVuZHMgVGVuc29yPih4OiBUfFRlbnNvckxpa2UsIHJlcHM6IG51bWJlcltdKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ3RpbGUnLCAnc3RyaW5nX29yX251bWVyaWMnKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkeC5yYW5rID09PSByZXBzLmxlbmd0aCxcbiAgICAgICgpID0+IGBFcnJvciBpbiB0cmFuc3Bvc2U6IHJhbmsgb2YgaW5wdXQgJHskeC5yYW5rfSBgICtcbiAgICAgICAgICBgbXVzdCBtYXRjaCBsZW5ndGggb2YgcmVwcyAke3JlcHN9LmApO1xuXG4gIGNvbnN0IGlucHV0czogVGlsZUlucHV0cyA9IHt4OiAkeH07XG4gIGNvbnN0IGF0dHJzOiBUaWxlQXR0cnMgPSB7cmVwc307XG5cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBUaWxlLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IHRpbGUgPSBvcCh7dGlsZV99KTtcbiJdfQ==