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
import { SparseSegmentSum } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import { op } from '../operation';
/**
 * Computes the sum along sparse segments of a tensor.
 *
 * ```js
 * const c = tf.tensor2d([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]]);
 * // Select two rows, one segment.
 * const result1 = tf.sparse.sparseSegmentSum(c,
 *                                           tf.tensor1d([0, 1], 'int32'),
 *                                           tf.tensor1d([0, 0], 'int32'));
 * result1.print(); // [[0, 0, 0, 0]]
 *
 * // Select two rows, two segment.
 * const result2 = tf.sparse.sparseSegmentSum(c,
 *                                           tf.tensor1d([0, 1], 'int32'),
 *                                           tf.tensor1d([0, 1], 'int32'));
 * result2.print(); // [[1, 2, 3, 4], [-1, -2, -3, -4]]
 *
 * // Select all rows, two segments.
 * const result3 = tf.sparse.sparseSegmentSum(c,
 *                                           tf.tensor1d([0, 1, 2], 'int32'),
 *                                           tf.tensor1d([0, 0, 1], 'int32'));
 * result3.print(); // [[0, 0, 0, 0], [5, 6, 7, 8]]
 * ```
 * @param data: A Tensor of at least one dimension with data that will be
 *     assembled in the output.
 * @param indices: A 1-D Tensor with indices into data. Has same rank as
 *     segmentIds.
 * @param segmentIds: A 1-D Tensor with indices into the output Tensor. Values
 *     should be sorted and can be repeated.
 * @return Has same shape as data, except for dimension 0 which has equal to
 *         the number of segments.
 *
 * @doc {heading: 'Operations', subheading: 'Sparse'}
 */
function sparseSegmentSum_(data, indices, segmentIds) {
    const $data = convertToTensor(data, 'data', 'sparseSegmentSum');
    const $indices = convertToTensor(indices, 'indices', 'sparseSegmentSum', 'int32');
    const $segmentIds = convertToTensor(segmentIds, 'segmentIds', 'sparseSegmentSum', 'int32');
    if ($data.rank < 1) {
        throw new Error(`Data should be at least 1 dimensional but received scalar`);
    }
    if ($indices.rank !== 1) {
        throw new Error(`Indices should be Tensor1D but received shape
         ${$indices.shape}`);
    }
    if ($segmentIds.rank !== 1) {
        throw new Error(`Segment ids should be Tensor1D but received shape
         ${$segmentIds.shape}`);
    }
    const inputs = {
        data: $data,
        indices: $indices,
        segmentIds: $segmentIds
    };
    return ENGINE.runKernel(SparseSegmentSum, inputs);
}
export const sparseSegmentSum = op({ sparseSegmentSum_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic3BhcnNlX3NlZ21lbnRfc3VtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvc3BhcnNlL3NwYXJzZV9zZWdtZW50X3N1bS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ3BDLE9BQU8sRUFBQyxnQkFBZ0IsRUFBeUIsTUFBTSxvQkFBb0IsQ0FBQztBQUU1RSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUVoQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBaUNHO0FBQ0gsU0FBUyxpQkFBaUIsQ0FDdEIsSUFBdUIsRUFBRSxPQUE0QixFQUNyRCxVQUErQjtJQUNqQyxNQUFNLEtBQUssR0FBRyxlQUFlLENBQUMsSUFBSSxFQUFFLE1BQU0sRUFBRSxrQkFBa0IsQ0FBQyxDQUFDO0lBQ2hFLE1BQU0sUUFBUSxHQUNWLGVBQWUsQ0FBQyxPQUFPLEVBQUUsU0FBUyxFQUFFLGtCQUFrQixFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ3JFLE1BQU0sV0FBVyxHQUNiLGVBQWUsQ0FBQyxVQUFVLEVBQUUsWUFBWSxFQUFFLGtCQUFrQixFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBRTNFLElBQUksS0FBSyxDQUFDLElBQUksR0FBRyxDQUFDLEVBQUU7UUFDbEIsTUFBTSxJQUFJLEtBQUssQ0FDWCwyREFBMkQsQ0FBQyxDQUFDO0tBQ2xFO0lBQ0QsSUFBSSxRQUFRLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUN2QixNQUFNLElBQUksS0FBSyxDQUFDO1dBQ1QsUUFBUSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7S0FDMUI7SUFDRCxJQUFJLFdBQVcsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQzFCLE1BQU0sSUFBSSxLQUFLLENBQUM7V0FDVCxXQUFXLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUM3QjtJQUVELE1BQU0sTUFBTSxHQUEyQjtRQUNyQyxJQUFJLEVBQUUsS0FBSztRQUNYLE9BQU8sRUFBRSxRQUFRO1FBQ2pCLFVBQVUsRUFBRSxXQUFXO0tBQ3hCLENBQUM7SUFFRixPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQUMsZ0JBQWdCLEVBQUUsTUFBWSxDQUFDLENBQUM7QUFDMUQsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGdCQUFnQixHQUFHLEVBQUUsQ0FBQyxFQUFDLGlCQUFpQixFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIxIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uLy4uL2VuZ2luZSc7XG5pbXBvcnQge1NwYXJzZVNlZ21lbnRTdW0sIFNwYXJzZVNlZ21lbnRTdW1JbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge1RlbnNvciwgVGVuc29yMUR9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgc3VtIGFsb25nIHNwYXJzZSBzZWdtZW50cyBvZiBhIHRlbnNvci5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYyA9IHRmLnRlbnNvcjJkKFtbMSwyLDMsNF0sIFstMSwtMiwtMywtNF0sIFs1LDYsNyw4XV0pO1xuICogLy8gU2VsZWN0IHR3byByb3dzLCBvbmUgc2VnbWVudC5cbiAqIGNvbnN0IHJlc3VsdDEgPSB0Zi5zcGFyc2Uuc3BhcnNlU2VnbWVudFN1bShjLFxuICogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGYudGVuc29yMWQoWzAsIDFdLCAnaW50MzInKSxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjFkKFswLCAwXSwgJ2ludDMyJykpO1xuICogcmVzdWx0MS5wcmludCgpOyAvLyBbWzAsIDAsIDAsIDBdXVxuICpcbiAqIC8vIFNlbGVjdCB0d28gcm93cywgdHdvIHNlZ21lbnQuXG4gKiBjb25zdCByZXN1bHQyID0gdGYuc3BhcnNlLnNwYXJzZVNlZ21lbnRTdW0oYyxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjFkKFswLCAxXSwgJ2ludDMyJyksXG4gKiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0Zi50ZW5zb3IxZChbMCwgMV0sICdpbnQzMicpKTtcbiAqIHJlc3VsdDIucHJpbnQoKTsgLy8gW1sxLCAyLCAzLCA0XSwgWy0xLCAtMiwgLTMsIC00XV1cbiAqXG4gKiAvLyBTZWxlY3QgYWxsIHJvd3MsIHR3byBzZWdtZW50cy5cbiAqIGNvbnN0IHJlc3VsdDMgPSB0Zi5zcGFyc2Uuc3BhcnNlU2VnbWVudFN1bShjLFxuICogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgdGYudGVuc29yMWQoWzAsIDEsIDJdLCAnaW50MzInKSxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHRmLnRlbnNvcjFkKFswLCAwLCAxXSwgJ2ludDMyJykpO1xuICogcmVzdWx0My5wcmludCgpOyAvLyBbWzAsIDAsIDAsIDBdLCBbNSwgNiwgNywgOF1dXG4gKiBgYGBcbiAqIEBwYXJhbSBkYXRhOiBBIFRlbnNvciBvZiBhdCBsZWFzdCBvbmUgZGltZW5zaW9uIHdpdGggZGF0YSB0aGF0IHdpbGwgYmVcbiAqICAgICBhc3NlbWJsZWQgaW4gdGhlIG91dHB1dC5cbiAqIEBwYXJhbSBpbmRpY2VzOiBBIDEtRCBUZW5zb3Igd2l0aCBpbmRpY2VzIGludG8gZGF0YS4gSGFzIHNhbWUgcmFuayBhc1xuICogICAgIHNlZ21lbnRJZHMuXG4gKiBAcGFyYW0gc2VnbWVudElkczogQSAxLUQgVGVuc29yIHdpdGggaW5kaWNlcyBpbnRvIHRoZSBvdXRwdXQgVGVuc29yLiBWYWx1ZXNcbiAqICAgICBzaG91bGQgYmUgc29ydGVkIGFuZCBjYW4gYmUgcmVwZWF0ZWQuXG4gKiBAcmV0dXJuIEhhcyBzYW1lIHNoYXBlIGFzIGRhdGEsIGV4Y2VwdCBmb3IgZGltZW5zaW9uIDAgd2hpY2ggaGFzIGVxdWFsIHRvXG4gKiAgICAgICAgIHRoZSBudW1iZXIgb2Ygc2VnbWVudHMuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnU3BhcnNlJ31cbiAqL1xuZnVuY3Rpb24gc3BhcnNlU2VnbWVudFN1bV8oXG4gICAgZGF0YTogVGVuc29yfFRlbnNvckxpa2UsIGluZGljZXM6IFRlbnNvcjFEfFRlbnNvckxpa2UsXG4gICAgc2VnbWVudElkczogVGVuc29yMUR8VGVuc29yTGlrZSk6IFRlbnNvciB7XG4gIGNvbnN0ICRkYXRhID0gY29udmVydFRvVGVuc29yKGRhdGEsICdkYXRhJywgJ3NwYXJzZVNlZ21lbnRTdW0nKTtcbiAgY29uc3QgJGluZGljZXMgPVxuICAgICAgY29udmVydFRvVGVuc29yKGluZGljZXMsICdpbmRpY2VzJywgJ3NwYXJzZVNlZ21lbnRTdW0nLCAnaW50MzInKTtcbiAgY29uc3QgJHNlZ21lbnRJZHMgPVxuICAgICAgY29udmVydFRvVGVuc29yKHNlZ21lbnRJZHMsICdzZWdtZW50SWRzJywgJ3NwYXJzZVNlZ21lbnRTdW0nLCAnaW50MzInKTtcblxuICBpZiAoJGRhdGEucmFuayA8IDEpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBEYXRhIHNob3VsZCBiZSBhdCBsZWFzdCAxIGRpbWVuc2lvbmFsIGJ1dCByZWNlaXZlZCBzY2FsYXJgKTtcbiAgfVxuICBpZiAoJGluZGljZXMucmFuayAhPT0gMSkge1xuICAgIHRocm93IG5ldyBFcnJvcihgSW5kaWNlcyBzaG91bGQgYmUgVGVuc29yMUQgYnV0IHJlY2VpdmVkIHNoYXBlXG4gICAgICAgICAkeyRpbmRpY2VzLnNoYXBlfWApO1xuICB9XG4gIGlmICgkc2VnbWVudElkcy5yYW5rICE9PSAxKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBTZWdtZW50IGlkcyBzaG91bGQgYmUgVGVuc29yMUQgYnV0IHJlY2VpdmVkIHNoYXBlXG4gICAgICAgICAkeyRzZWdtZW50SWRzLnNoYXBlfWApO1xuICB9XG5cbiAgY29uc3QgaW5wdXRzOiBTcGFyc2VTZWdtZW50U3VtSW5wdXRzID0ge1xuICAgIGRhdGE6ICRkYXRhLFxuICAgIGluZGljZXM6ICRpbmRpY2VzLFxuICAgIHNlZ21lbnRJZHM6ICRzZWdtZW50SWRzXG4gIH07XG5cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoU3BhcnNlU2VnbWVudFN1bSwgaW5wdXRzIGFzIHt9KTtcbn1cblxuZXhwb3J0IGNvbnN0IHNwYXJzZVNlZ21lbnRTdW0gPSBvcCh7c3BhcnNlU2VnbWVudFN1bV99KTtcbiJdfQ==