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
import { assert } from '../../util';
import { div } from '../div';
import { mul } from '../mul';
import { norm } from '../norm';
import { op } from '../operation';
import { split } from '../split';
import { squeeze } from '../squeeze';
import { stack } from '../stack';
import { sub } from '../sub';
import { sum } from '../sum';
/**
 * Gram-Schmidt orthogonalization.
 *
 * ```js
 * const x = tf.tensor2d([[1, 2], [3, 4]]);
 * let y = tf.linalg.gramSchmidt(x);
 * y.print();
 * console.log('Othogonalized:');
 * y.dot(y.transpose()).print();  // should be nearly the identity matrix.
 * console.log('First row direction maintained:');
 * const data = await y.array();
 * console.log(data[0][1] / data[0][0]);  // should be nearly 2.
 * ```
 *
 * @param xs The vectors to be orthogonalized, in one of the two following
 *   formats:
 *   - An Array of `tf.Tensor1D`.
 *   - A `tf.Tensor2D`, i.e., a matrix, in which case the vectors are the rows
 *     of `xs`.
 *   In each case, all the vectors must have the same length and the length
 *   must be greater than or equal to the number of vectors.
 * @returns The orthogonalized and normalized vectors or matrix.
 *   Orthogonalization means that the vectors or the rows of the matrix
 *   are orthogonal (zero inner products). Normalization means that each
 *   vector or each row of the matrix has an L2 norm that equals `1`.
 *
 * @doc {heading:'Operations', subheading:'Linear Algebra', namespace:'linalg'}
 */
function gramSchmidt_(xs) {
    let inputIsTensor2D;
    if (Array.isArray(xs)) {
        inputIsTensor2D = false;
        assert(xs != null && xs.length > 0, () => 'Gram-Schmidt process: input must not be null, undefined, or ' +
            'empty');
        const dim = xs[0].shape[0];
        for (let i = 1; i < xs.length; ++i) {
            assert(xs[i].shape[0] === dim, () => 'Gram-Schmidt: Non-unique lengths found in the input vectors: ' +
                `(${xs[i].shape[0]} vs. ${dim})`);
        }
    }
    else {
        inputIsTensor2D = true;
        xs = split(xs, xs.shape[0], 0).map(x => squeeze(x, [0]));
    }
    assert(xs.length <= xs[0].shape[0], () => `Gram-Schmidt: Number of vectors (${xs.length}) exceeds ` +
        `number of dimensions (${xs[0].shape[0]}).`);
    const ys = [];
    const xs1d = xs;
    for (let i = 0; i < xs.length; ++i) {
        ys.push(ENGINE.tidy(() => {
            let x = xs1d[i];
            if (i > 0) {
                for (let j = 0; j < i; ++j) {
                    const proj = mul(sum(mul(ys[j], x)), ys[j]);
                    x = sub(x, proj);
                }
            }
            return div(x, norm(x, 'euclidean'));
        }));
    }
    if (inputIsTensor2D) {
        return stack(ys, 0);
    }
    else {
        return ys;
    }
}
export const gramSchmidt = op({ gramSchmidt_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ3JhbV9zY2htaWR0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvbGluYWxnL2dyYW1fc2NobWlkdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRXBDLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFFbEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUMzQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxJQUFJLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDN0IsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDbkMsT0FBTyxFQUFDLEtBQUssRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUMvQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxRQUFRLENBQUM7QUFFM0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTJCRztBQUNILFNBQVMsWUFBWSxDQUFDLEVBQXVCO0lBQzNDLElBQUksZUFBd0IsQ0FBQztJQUM3QixJQUFJLEtBQUssQ0FBQyxPQUFPLENBQUMsRUFBRSxDQUFDLEVBQUU7UUFDckIsZUFBZSxHQUFHLEtBQUssQ0FBQztRQUN4QixNQUFNLENBQ0YsRUFBRSxJQUFJLElBQUksSUFBSSxFQUFFLENBQUMsTUFBTSxHQUFHLENBQUMsRUFDM0IsR0FBRyxFQUFFLENBQUMsOERBQThEO1lBQ2hFLE9BQU8sQ0FBQyxDQUFDO1FBQ2pCLE1BQU0sR0FBRyxHQUFHLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDM0IsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUU7WUFDbEMsTUFBTSxDQUNGLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEtBQUssR0FBRyxFQUN0QixHQUFHLEVBQUUsQ0FDRCwrREFBK0Q7Z0JBQy9ELElBQUssRUFBaUIsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLFFBQVEsR0FBRyxHQUFHLENBQUMsQ0FBQztTQUMzRDtLQUNGO1NBQU07UUFDTCxlQUFlLEdBQUcsSUFBSSxDQUFDO1FBQ3ZCLEVBQUUsR0FBRyxLQUFLLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsT0FBTyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUMxRDtJQUVELE1BQU0sQ0FDRixFQUFFLENBQUMsTUFBTSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQzNCLEdBQUcsRUFBRSxDQUFDLG9DQUNLLEVBQWlCLENBQUMsTUFBTSxZQUFZO1FBQzNDLHlCQUEwQixFQUFpQixDQUFDLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7SUFFckUsTUFBTSxFQUFFLEdBQWUsRUFBRSxDQUFDO0lBQzFCLE1BQU0sSUFBSSxHQUFHLEVBQUUsQ0FBQztJQUNoQixLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRTtRQUNsQyxFQUFFLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsR0FBRyxFQUFFO1lBQ3ZCLElBQUksQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNoQixJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUU7Z0JBQ1QsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRTtvQkFDMUIsTUFBTSxJQUFJLEdBQUcsR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7b0JBQzVDLENBQUMsR0FBRyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDO2lCQUNsQjthQUNGO1lBQ0QsT0FBTyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQztRQUN0QyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQ0w7SUFFRCxJQUFJLGVBQWUsRUFBRTtRQUNuQixPQUFPLEtBQUssQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFhLENBQUM7S0FDakM7U0FBTTtRQUNMLE9BQU8sRUFBRSxDQUFDO0tBQ1g7QUFDSCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sV0FBVyxHQUFHLEVBQUUsQ0FBQyxFQUFDLFlBQVksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtUZW5zb3IxRCwgVGVuc29yMkR9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge2Fzc2VydH0gZnJvbSAnLi4vLi4vdXRpbCc7XG5cbmltcG9ydCB7ZGl2fSBmcm9tICcuLi9kaXYnO1xuaW1wb3J0IHttdWx9IGZyb20gJy4uL211bCc7XG5pbXBvcnQge25vcm19IGZyb20gJy4uL25vcm0nO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi4vb3BlcmF0aW9uJztcbmltcG9ydCB7c3BsaXR9IGZyb20gJy4uL3NwbGl0JztcbmltcG9ydCB7c3F1ZWV6ZX0gZnJvbSAnLi4vc3F1ZWV6ZSc7XG5pbXBvcnQge3N0YWNrfSBmcm9tICcuLi9zdGFjayc7XG5pbXBvcnQge3N1Yn0gZnJvbSAnLi4vc3ViJztcbmltcG9ydCB7c3VtfSBmcm9tICcuLi9zdW0nO1xuXG4vKipcbiAqIEdyYW0tU2NobWlkdCBvcnRob2dvbmFsaXphdGlvbi5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFtbMSwgMl0sIFszLCA0XV0pO1xuICogbGV0IHkgPSB0Zi5saW5hbGcuZ3JhbVNjaG1pZHQoeCk7XG4gKiB5LnByaW50KCk7XG4gKiBjb25zb2xlLmxvZygnT3Rob2dvbmFsaXplZDonKTtcbiAqIHkuZG90KHkudHJhbnNwb3NlKCkpLnByaW50KCk7ICAvLyBzaG91bGQgYmUgbmVhcmx5IHRoZSBpZGVudGl0eSBtYXRyaXguXG4gKiBjb25zb2xlLmxvZygnRmlyc3Qgcm93IGRpcmVjdGlvbiBtYWludGFpbmVkOicpO1xuICogY29uc3QgZGF0YSA9IGF3YWl0IHkuYXJyYXkoKTtcbiAqIGNvbnNvbGUubG9nKGRhdGFbMF1bMV0gLyBkYXRhWzBdWzBdKTsgIC8vIHNob3VsZCBiZSBuZWFybHkgMi5cbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4cyBUaGUgdmVjdG9ycyB0byBiZSBvcnRob2dvbmFsaXplZCwgaW4gb25lIG9mIHRoZSB0d28gZm9sbG93aW5nXG4gKiAgIGZvcm1hdHM6XG4gKiAgIC0gQW4gQXJyYXkgb2YgYHRmLlRlbnNvcjFEYC5cbiAqICAgLSBBIGB0Zi5UZW5zb3IyRGAsIGkuZS4sIGEgbWF0cml4LCBpbiB3aGljaCBjYXNlIHRoZSB2ZWN0b3JzIGFyZSB0aGUgcm93c1xuICogICAgIG9mIGB4c2AuXG4gKiAgIEluIGVhY2ggY2FzZSwgYWxsIHRoZSB2ZWN0b3JzIG11c3QgaGF2ZSB0aGUgc2FtZSBsZW5ndGggYW5kIHRoZSBsZW5ndGhcbiAqICAgbXVzdCBiZSBncmVhdGVyIHRoYW4gb3IgZXF1YWwgdG8gdGhlIG51bWJlciBvZiB2ZWN0b3JzLlxuICogQHJldHVybnMgVGhlIG9ydGhvZ29uYWxpemVkIGFuZCBub3JtYWxpemVkIHZlY3RvcnMgb3IgbWF0cml4LlxuICogICBPcnRob2dvbmFsaXphdGlvbiBtZWFucyB0aGF0IHRoZSB2ZWN0b3JzIG9yIHRoZSByb3dzIG9mIHRoZSBtYXRyaXhcbiAqICAgYXJlIG9ydGhvZ29uYWwgKHplcm8gaW5uZXIgcHJvZHVjdHMpLiBOb3JtYWxpemF0aW9uIG1lYW5zIHRoYXQgZWFjaFxuICogICB2ZWN0b3Igb3IgZWFjaCByb3cgb2YgdGhlIG1hdHJpeCBoYXMgYW4gTDIgbm9ybSB0aGF0IGVxdWFscyBgMWAuXG4gKlxuICogQGRvYyB7aGVhZGluZzonT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6J0xpbmVhciBBbGdlYnJhJywgbmFtZXNwYWNlOidsaW5hbGcnfVxuICovXG5mdW5jdGlvbiBncmFtU2NobWlkdF8oeHM6IFRlbnNvcjFEW118VGVuc29yMkQpOiBUZW5zb3IxRFtdfFRlbnNvcjJEIHtcbiAgbGV0IGlucHV0SXNUZW5zb3IyRDogYm9vbGVhbjtcbiAgaWYgKEFycmF5LmlzQXJyYXkoeHMpKSB7XG4gICAgaW5wdXRJc1RlbnNvcjJEID0gZmFsc2U7XG4gICAgYXNzZXJ0KFxuICAgICAgICB4cyAhPSBudWxsICYmIHhzLmxlbmd0aCA+IDAsXG4gICAgICAgICgpID0+ICdHcmFtLVNjaG1pZHQgcHJvY2VzczogaW5wdXQgbXVzdCBub3QgYmUgbnVsbCwgdW5kZWZpbmVkLCBvciAnICtcbiAgICAgICAgICAgICdlbXB0eScpO1xuICAgIGNvbnN0IGRpbSA9IHhzWzBdLnNoYXBlWzBdO1xuICAgIGZvciAobGV0IGkgPSAxOyBpIDwgeHMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGFzc2VydChcbiAgICAgICAgICB4c1tpXS5zaGFwZVswXSA9PT0gZGltLFxuICAgICAgICAgICgpID0+XG4gICAgICAgICAgICAgICdHcmFtLVNjaG1pZHQ6IE5vbi11bmlxdWUgbGVuZ3RocyBmb3VuZCBpbiB0aGUgaW5wdXQgdmVjdG9yczogJyArXG4gICAgICAgICAgICAgIGAoJHsoeHMgYXMgVGVuc29yMURbXSlbaV0uc2hhcGVbMF19IHZzLiAke2RpbX0pYCk7XG4gICAgfVxuICB9IGVsc2Uge1xuICAgIGlucHV0SXNUZW5zb3IyRCA9IHRydWU7XG4gICAgeHMgPSBzcGxpdCh4cywgeHMuc2hhcGVbMF0sIDApLm1hcCh4ID0+IHNxdWVlemUoeCwgWzBdKSk7XG4gIH1cblxuICBhc3NlcnQoXG4gICAgICB4cy5sZW5ndGggPD0geHNbMF0uc2hhcGVbMF0sXG4gICAgICAoKSA9PiBgR3JhbS1TY2htaWR0OiBOdW1iZXIgb2YgdmVjdG9ycyAoJHtcbiAgICAgICAgICAgICAgICAoeHMgYXMgVGVuc29yMURbXSkubGVuZ3RofSkgZXhjZWVkcyBgICtcbiAgICAgICAgICBgbnVtYmVyIG9mIGRpbWVuc2lvbnMgKCR7KHhzIGFzIFRlbnNvcjFEW10pWzBdLnNoYXBlWzBdfSkuYCk7XG5cbiAgY29uc3QgeXM6IFRlbnNvcjFEW10gPSBbXTtcbiAgY29uc3QgeHMxZCA9IHhzO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IHhzLmxlbmd0aDsgKytpKSB7XG4gICAgeXMucHVzaChFTkdJTkUudGlkeSgoKSA9PiB7XG4gICAgICBsZXQgeCA9IHhzMWRbaV07XG4gICAgICBpZiAoaSA+IDApIHtcbiAgICAgICAgZm9yIChsZXQgaiA9IDA7IGogPCBpOyArK2opIHtcbiAgICAgICAgICBjb25zdCBwcm9qID0gbXVsKHN1bShtdWwoeXNbal0sIHgpKSwgeXNbal0pO1xuICAgICAgICAgIHggPSBzdWIoeCwgcHJvaik7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIHJldHVybiBkaXYoeCwgbm9ybSh4LCAnZXVjbGlkZWFuJykpO1xuICAgIH0pKTtcbiAgfVxuXG4gIGlmIChpbnB1dElzVGVuc29yMkQpIHtcbiAgICByZXR1cm4gc3RhY2soeXMsIDApIGFzIFRlbnNvcjJEO1xuICB9IGVsc2Uge1xuICAgIHJldHVybiB5cztcbiAgfVxufVxuXG5leHBvcnQgY29uc3QgZ3JhbVNjaG1pZHQgPSBvcCh7Z3JhbVNjaG1pZHRffSk7XG4iXX0=