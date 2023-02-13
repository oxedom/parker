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
import { convertToTensor } from '../../tensor_util_env';
import { assert } from '../../util';
import { greaterEqual } from '../greater_equal';
import { lessEqual } from '../less_equal';
import { logicalAnd } from '../logical_and';
import { op } from '../operation';
import { range } from '../range';
import { reshape } from '../reshape';
import { scalar } from '../scalar';
import { stack } from '../stack';
import { sub } from '../sub';
import { unstack } from '../unstack';
import { where } from '../where';
import { zeros } from '../zeros';
/**
 * Copy a tensor setting everything outside a central band in each innermost
 * matrix to zero.
 *
 * The band part is computed as follows: Assume input has `k` dimensions
 * `[I, J, K, ..., M, N]`, then the output is a tensor with the same shape where
 * `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.
 * The indicator function
 * `in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower))`
 * `&& (num_upper < 0 || (n-m) <= num_upper)`
 *
 * ```js
 * const x = tf.tensor2d([[ 0,  1,  2, 3],
 *                        [-1,  0,  1, 2],
 *                        [-2, -1,  0, 1],
 *                        [-3, -2, -1, 0]]);
 * let y = tf.linalg.bandPart(x, 1, -1);
 * y.print(); // [[ 0,  1,  2, 3],
 *            //  [-1,  0,  1, 2],
 *            //  [ 0, -1,  0, 1],
 *            //  [ 0, 0 , -1, 0]]
 * let z = tf.linalg.bandPart(x, 2, 1);
 * z.print(); // [[ 0,  1,  0, 0],
 *            //  [-1,  0,  1, 0],
 *            //  [-2, -1,  0, 1],
 *            //  [ 0, -2, -1, 0]]
 * ```
 *
 * @param x Rank `k` tensor
 * @param numLower Number of subdiagonals to keep.
 *   If negative, keep entire lower triangle.
 * @param numUpper Number of subdiagonals to keep.
 *   If negative, keep entire upper triangle.
 * @returns Rank `k` tensor of the same shape as input.
 *   The extracted banded tensor.
 *
 * @doc {heading:'Operations', subheading:'Linear Algebra', namespace:'linalg'}
 */
function bandPart_(a, numLower, numUpper) {
    assert(numLower % 1 === 0, () => `bandPart(): numLower must be an integer, got ${numLower}.`);
    assert(numUpper % 1 === 0, () => `bandPart(): numUpper must be an integer, got ${numUpper}.`);
    const $a = convertToTensor(a, 'a', 'bandPart');
    assert($a.rank >= 2, () => `bandPart(): Rank must be at least 2, got ${$a.rank}.`);
    const shape = $a.shape;
    const [M, N] = $a.shape.slice(-2);
    if (!(numLower <= M)) {
        throw new Error(`bandPart(): numLower (${numLower})` +
            ` must not be greater than the number of rows (${M}).`);
    }
    if (!(numUpper <= N)) {
        throw new Error(`bandPart(): numUpper (${numUpper})` +
            ` must not be greater than the number of columns (${N}).`);
    }
    if (numLower < 0) {
        numLower = M;
    }
    if (numUpper < 0) {
        numUpper = N;
    }
    const i = reshape(range(0, M, 1, 'int32'), [-1, 1]);
    const j = range(0, N, 1, 'int32');
    const ij = sub(i, j);
    const inBand = logicalAnd(lessEqual(ij, scalar(+numLower, 'int32')), greaterEqual(ij, scalar(-numUpper, 'int32')));
    const zero = zeros([M, N], $a.dtype);
    return reshape(stack(unstack(reshape($a, [-1, M, N]))
        .map(mat => where(inBand, mat, zero))), shape);
}
export const bandPart = op({ bandPart_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFuZF9wYXJ0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvbGluYWxnL2JhbmRfcGFydC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFlBQVksQ0FBQztBQUVsQyxPQUFPLEVBQUMsWUFBWSxFQUFDLE1BQU0sa0JBQWtCLENBQUM7QUFDOUMsT0FBTyxFQUFDLFNBQVMsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUN4QyxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDMUMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDbkMsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFDM0IsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNuQyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FxQ0c7QUFDSCxTQUFTLFNBQVMsQ0FDZCxDQUFlLEVBQUUsUUFBZ0IsRUFBRSxRQUFnQjtJQUNyRCxNQUFNLENBQ0YsUUFBUSxHQUFHLENBQUMsS0FBSyxDQUFDLEVBQ2xCLEdBQUcsRUFBRSxDQUFDLGdEQUFnRCxRQUFRLEdBQUcsQ0FBQyxDQUFDO0lBQ3ZFLE1BQU0sQ0FDRixRQUFRLEdBQUcsQ0FBQyxLQUFLLENBQUMsRUFDbEIsR0FBRyxFQUFFLENBQUMsZ0RBQWdELFFBQVEsR0FBRyxDQUFDLENBQUM7SUFFdkUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsVUFBVSxDQUFDLENBQUM7SUFFL0MsTUFBTSxDQUNGLEVBQUUsQ0FBQyxJQUFJLElBQUksQ0FBQyxFQUNaLEdBQUcsRUFBRSxDQUFDLDRDQUE0QyxFQUFFLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUVsRSxNQUFNLEtBQUssR0FBRyxFQUFFLENBQUMsS0FBSyxDQUFDO0lBQ3ZCLE1BQU0sQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUVsQyxJQUFJLENBQUMsQ0FBQyxRQUFRLElBQUksQ0FBQyxDQUFDLEVBQUU7UUFDcEIsTUFBTSxJQUFJLEtBQUssQ0FDWCx5QkFBeUIsUUFBUSxHQUFHO1lBQ3BDLGlEQUFpRCxDQUFDLElBQUksQ0FBQyxDQUFDO0tBQzdEO0lBQ0QsSUFBSSxDQUFDLENBQUMsUUFBUSxJQUFJLENBQUMsQ0FBQyxFQUFFO1FBQ3BCLE1BQU0sSUFBSSxLQUFLLENBQ1gseUJBQXlCLFFBQVEsR0FBRztZQUNwQyxvREFBb0QsQ0FBQyxJQUFJLENBQUMsQ0FBQztLQUNoRTtJQUVELElBQUksUUFBUSxHQUFHLENBQUMsRUFBRTtRQUNoQixRQUFRLEdBQUcsQ0FBQyxDQUFDO0tBQ2Q7SUFDRCxJQUFJLFFBQVEsR0FBRyxDQUFDLEVBQUU7UUFDaEIsUUFBUSxHQUFHLENBQUMsQ0FBQztLQUNkO0lBRUQsTUFBTSxDQUFDLEdBQUcsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxPQUFPLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDcEQsTUFBTSxDQUFDLEdBQUcsS0FBSyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ2xDLE1BQU0sRUFBRSxHQUFHLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFFckIsTUFBTSxNQUFNLEdBQUcsVUFBVSxDQUNyQixTQUFTLENBQUMsRUFBRSxFQUFFLE1BQU0sQ0FBQyxDQUFDLFFBQVEsRUFBRSxPQUFPLENBQUMsQ0FBQyxFQUN6QyxZQUFZLENBQUMsRUFBRSxFQUFFLE1BQU0sQ0FBQyxDQUFDLFFBQVEsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFFbEQsTUFBTSxJQUFJLEdBQUcsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUVyQyxPQUFPLE9BQU8sQ0FDSCxLQUFLLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUMzQixHQUFHLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxLQUFLLENBQUMsTUFBTSxFQUFFLEdBQUcsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLEVBQ2hELEtBQUssQ0FBTSxDQUFDO0FBQ3pCLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxRQUFRLEdBQUcsRUFBRSxDQUFDLEVBQUMsU0FBUyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0IHthc3NlcnR9IGZyb20gJy4uLy4uL3V0aWwnO1xuXG5pbXBvcnQge2dyZWF0ZXJFcXVhbH0gZnJvbSAnLi4vZ3JlYXRlcl9lcXVhbCc7XG5pbXBvcnQge2xlc3NFcXVhbH0gZnJvbSAnLi4vbGVzc19lcXVhbCc7XG5pbXBvcnQge2xvZ2ljYWxBbmR9IGZyb20gJy4uL2xvZ2ljYWxfYW5kJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5pbXBvcnQge3JhbmdlfSBmcm9tICcuLi9yYW5nZSc7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4uL3Jlc2hhcGUnO1xuaW1wb3J0IHtzY2FsYXJ9IGZyb20gJy4uL3NjYWxhcic7XG5pbXBvcnQge3N0YWNrfSBmcm9tICcuLi9zdGFjayc7XG5pbXBvcnQge3N1Yn0gZnJvbSAnLi4vc3ViJztcbmltcG9ydCB7dW5zdGFja30gZnJvbSAnLi4vdW5zdGFjayc7XG5pbXBvcnQge3doZXJlfSBmcm9tICcuLi93aGVyZSc7XG5pbXBvcnQge3plcm9zfSBmcm9tICcuLi96ZXJvcyc7XG5cbi8qKlxuICogQ29weSBhIHRlbnNvciBzZXR0aW5nIGV2ZXJ5dGhpbmcgb3V0c2lkZSBhIGNlbnRyYWwgYmFuZCBpbiBlYWNoIGlubmVybW9zdFxuICogbWF0cml4IHRvIHplcm8uXG4gKlxuICogVGhlIGJhbmQgcGFydCBpcyBjb21wdXRlZCBhcyBmb2xsb3dzOiBBc3N1bWUgaW5wdXQgaGFzIGBrYCBkaW1lbnNpb25zXG4gKiBgW0ksIEosIEssIC4uLiwgTSwgTl1gLCB0aGVuIHRoZSBvdXRwdXQgaXMgYSB0ZW5zb3Igd2l0aCB0aGUgc2FtZSBzaGFwZSB3aGVyZVxuICogYGJhbmRbaSwgaiwgaywgLi4uLCBtLCBuXSA9IGluX2JhbmQobSwgbikgKiBpbnB1dFtpLCBqLCBrLCAuLi4sIG0sIG5dYC5cbiAqIFRoZSBpbmRpY2F0b3IgZnVuY3Rpb25cbiAqIGBpbl9iYW5kKG0sIG4pID0gKG51bV9sb3dlciA8IDAgfHwgKG0tbikgPD0gbnVtX2xvd2VyKSlgXG4gKiBgJiYgKG51bV91cHBlciA8IDAgfHwgKG4tbSkgPD0gbnVtX3VwcGVyKWBcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFtbIDAsICAxLCAgMiwgM10sXG4gKiAgICAgICAgICAgICAgICAgICAgICAgIFstMSwgIDAsICAxLCAyXSxcbiAqICAgICAgICAgICAgICAgICAgICAgICAgWy0yLCAtMSwgIDAsIDFdLFxuICogICAgICAgICAgICAgICAgICAgICAgICBbLTMsIC0yLCAtMSwgMF1dKTtcbiAqIGxldCB5ID0gdGYubGluYWxnLmJhbmRQYXJ0KHgsIDEsIC0xKTtcbiAqIHkucHJpbnQoKTsgLy8gW1sgMCwgIDEsICAyLCAzXSxcbiAqICAgICAgICAgICAgLy8gIFstMSwgIDAsICAxLCAyXSxcbiAqICAgICAgICAgICAgLy8gIFsgMCwgLTEsICAwLCAxXSxcbiAqICAgICAgICAgICAgLy8gIFsgMCwgMCAsIC0xLCAwXV1cbiAqIGxldCB6ID0gdGYubGluYWxnLmJhbmRQYXJ0KHgsIDIsIDEpO1xuICogei5wcmludCgpOyAvLyBbWyAwLCAgMSwgIDAsIDBdLFxuICogICAgICAgICAgICAvLyAgWy0xLCAgMCwgIDEsIDBdLFxuICogICAgICAgICAgICAvLyAgWy0yLCAtMSwgIDAsIDFdLFxuICogICAgICAgICAgICAvLyAgWyAwLCAtMiwgLTEsIDBdXVxuICogYGBgXG4gKlxuICogQHBhcmFtIHggUmFuayBga2AgdGVuc29yXG4gKiBAcGFyYW0gbnVtTG93ZXIgTnVtYmVyIG9mIHN1YmRpYWdvbmFscyB0byBrZWVwLlxuICogICBJZiBuZWdhdGl2ZSwga2VlcCBlbnRpcmUgbG93ZXIgdHJpYW5nbGUuXG4gKiBAcGFyYW0gbnVtVXBwZXIgTnVtYmVyIG9mIHN1YmRpYWdvbmFscyB0byBrZWVwLlxuICogICBJZiBuZWdhdGl2ZSwga2VlcCBlbnRpcmUgdXBwZXIgdHJpYW5nbGUuXG4gKiBAcmV0dXJucyBSYW5rIGBrYCB0ZW5zb3Igb2YgdGhlIHNhbWUgc2hhcGUgYXMgaW5wdXQuXG4gKiAgIFRoZSBleHRyYWN0ZWQgYmFuZGVkIHRlbnNvci5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOidPcGVyYXRpb25zJywgc3ViaGVhZGluZzonTGluZWFyIEFsZ2VicmEnLCBuYW1lc3BhY2U6J2xpbmFsZyd9XG4gKi9cbmZ1bmN0aW9uIGJhbmRQYXJ0XzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICBhOiBUfFRlbnNvckxpa2UsIG51bUxvd2VyOiBudW1iZXIsIG51bVVwcGVyOiBudW1iZXIpOiBUIHtcbiAgYXNzZXJ0KFxuICAgICAgbnVtTG93ZXIgJSAxID09PSAwLFxuICAgICAgKCkgPT4gYGJhbmRQYXJ0KCk6IG51bUxvd2VyIG11c3QgYmUgYW4gaW50ZWdlciwgZ290ICR7bnVtTG93ZXJ9LmApO1xuICBhc3NlcnQoXG4gICAgICBudW1VcHBlciAlIDEgPT09IDAsXG4gICAgICAoKSA9PiBgYmFuZFBhcnQoKTogbnVtVXBwZXIgbXVzdCBiZSBhbiBpbnRlZ2VyLCBnb3QgJHtudW1VcHBlcn0uYCk7XG5cbiAgY29uc3QgJGEgPSBjb252ZXJ0VG9UZW5zb3IoYSwgJ2EnLCAnYmFuZFBhcnQnKTtcblxuICBhc3NlcnQoXG4gICAgICAkYS5yYW5rID49IDIsXG4gICAgICAoKSA9PiBgYmFuZFBhcnQoKTogUmFuayBtdXN0IGJlIGF0IGxlYXN0IDIsIGdvdCAkeyRhLnJhbmt9LmApO1xuXG4gIGNvbnN0IHNoYXBlID0gJGEuc2hhcGU7XG4gIGNvbnN0IFtNLCBOXSA9ICRhLnNoYXBlLnNsaWNlKC0yKTtcblxuICBpZiAoIShudW1Mb3dlciA8PSBNKSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYGJhbmRQYXJ0KCk6IG51bUxvd2VyICgke251bUxvd2VyfSlgICtcbiAgICAgICAgYCBtdXN0IG5vdCBiZSBncmVhdGVyIHRoYW4gdGhlIG51bWJlciBvZiByb3dzICgke019KS5gKTtcbiAgfVxuICBpZiAoIShudW1VcHBlciA8PSBOKSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYGJhbmRQYXJ0KCk6IG51bVVwcGVyICgke251bVVwcGVyfSlgICtcbiAgICAgICAgYCBtdXN0IG5vdCBiZSBncmVhdGVyIHRoYW4gdGhlIG51bWJlciBvZiBjb2x1bW5zICgke059KS5gKTtcbiAgfVxuXG4gIGlmIChudW1Mb3dlciA8IDApIHtcbiAgICBudW1Mb3dlciA9IE07XG4gIH1cbiAgaWYgKG51bVVwcGVyIDwgMCkge1xuICAgIG51bVVwcGVyID0gTjtcbiAgfVxuXG4gIGNvbnN0IGkgPSByZXNoYXBlKHJhbmdlKDAsIE0sIDEsICdpbnQzMicpLCBbLTEsIDFdKTtcbiAgY29uc3QgaiA9IHJhbmdlKDAsIE4sIDEsICdpbnQzMicpO1xuICBjb25zdCBpaiA9IHN1YihpLCBqKTtcblxuICBjb25zdCBpbkJhbmQgPSBsb2dpY2FsQW5kKFxuICAgICAgbGVzc0VxdWFsKGlqLCBzY2FsYXIoK251bUxvd2VyLCAnaW50MzInKSksXG4gICAgICBncmVhdGVyRXF1YWwoaWosIHNjYWxhcigtbnVtVXBwZXIsICdpbnQzMicpKSk7XG5cbiAgY29uc3QgemVybyA9IHplcm9zKFtNLCBOXSwgJGEuZHR5cGUpO1xuXG4gIHJldHVybiByZXNoYXBlKFxuICAgICAgICAgICAgIHN0YWNrKHVuc3RhY2socmVzaGFwZSgkYSwgWy0xLCBNLCBOXSkpXG4gICAgICAgICAgICAgICAgICAgICAgIC5tYXAobWF0ID0+IHdoZXJlKGluQmFuZCwgbWF0LCB6ZXJvKSkpLFxuICAgICAgICAgICAgIHNoYXBlKSBhcyBUO1xufVxuXG5leHBvcnQgY29uc3QgYmFuZFBhcnQgPSBvcCh7YmFuZFBhcnRffSk7XG4iXX0=