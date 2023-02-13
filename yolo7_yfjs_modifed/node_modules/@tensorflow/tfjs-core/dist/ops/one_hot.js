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
import { OneHot } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Creates a one-hot `tf.Tensor`. The locations represented by `indices` take
 * value `onValue` (defaults to 1), while all other locations take value
 * `offValue` (defaults to 0). If `indices` is rank `R`, the output has rank
 * `R+1` with the last axis of size `depth`.
 * `indices` used to encode prediction class must start from 0. For example,
 *  if you have 3 classes of data, class 1 should be encoded as 0, class 2
 *  should be 1, and class 3 should be 2.
 *
 * ```js
 * tf.oneHot(tf.tensor1d([0, 1], 'int32'), 3).print();
 * ```
 *
 * @param indices `tf.Tensor` of indices with dtype `int32`. Indices must
 * start from 0.
 * @param depth The depth of the one hot dimension.
 * @param onValue A number used to fill in the output when the index matches
 * the location.
 * @param offValue A number used to fill in the output when the index does
 *     not match the location.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function oneHot_(indices, depth, onValue = 1, offValue = 0) {
    if (depth < 2) {
        throw new Error(`Error in oneHot: depth must be >=2, but it is ${depth}`);
    }
    const $indices = convertToTensor(indices, 'indices', 'oneHot', 'int32');
    const inputs = { indices: $indices };
    const attrs = { depth, onValue, offValue };
    return ENGINE.runKernel(OneHot, inputs, attrs);
}
export const oneHot = op({ oneHot_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoib25lX2hvdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL29uZV9ob3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsTUFBTSxFQUE0QixNQUFNLGlCQUFpQixDQUFDO0FBSWxFLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBc0JHO0FBQ0gsU0FBUyxPQUFPLENBQ1osT0FBMEIsRUFBRSxLQUFhLEVBQUUsT0FBTyxHQUFHLENBQUMsRUFDdEQsUUFBUSxHQUFHLENBQUM7SUFDZCxJQUFJLEtBQUssR0FBRyxDQUFDLEVBQUU7UUFDYixNQUFNLElBQUksS0FBSyxDQUFDLGlEQUFpRCxLQUFLLEVBQUUsQ0FBQyxDQUFDO0tBQzNFO0lBQ0QsTUFBTSxRQUFRLEdBQUcsZUFBZSxDQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUsUUFBUSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBRXhFLE1BQU0sTUFBTSxHQUFpQixFQUFDLE9BQU8sRUFBRSxRQUFRLEVBQUMsQ0FBQztJQUNqRCxNQUFNLEtBQUssR0FBZ0IsRUFBQyxLQUFLLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBQyxDQUFDO0lBRXRELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsTUFBTSxFQUFFLE1BQW1DLEVBQzNDLEtBQWdDLENBQUMsQ0FBQztBQUN4QyxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxFQUFDLE9BQU8sRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtPbmVIb3QsIE9uZUhvdEF0dHJzLCBPbmVIb3RJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDcmVhdGVzIGEgb25lLWhvdCBgdGYuVGVuc29yYC4gVGhlIGxvY2F0aW9ucyByZXByZXNlbnRlZCBieSBgaW5kaWNlc2AgdGFrZVxuICogdmFsdWUgYG9uVmFsdWVgIChkZWZhdWx0cyB0byAxKSwgd2hpbGUgYWxsIG90aGVyIGxvY2F0aW9ucyB0YWtlIHZhbHVlXG4gKiBgb2ZmVmFsdWVgIChkZWZhdWx0cyB0byAwKS4gSWYgYGluZGljZXNgIGlzIHJhbmsgYFJgLCB0aGUgb3V0cHV0IGhhcyByYW5rXG4gKiBgUisxYCB3aXRoIHRoZSBsYXN0IGF4aXMgb2Ygc2l6ZSBgZGVwdGhgLiBcbiAqIGBpbmRpY2VzYCB1c2VkIHRvIGVuY29kZSBwcmVkaWN0aW9uIGNsYXNzIG11c3Qgc3RhcnQgZnJvbSAwLiBGb3IgZXhhbXBsZSxcbiAqICBpZiB5b3UgaGF2ZSAzIGNsYXNzZXMgb2YgZGF0YSwgY2xhc3MgMSBzaG91bGQgYmUgZW5jb2RlZCBhcyAwLCBjbGFzcyAyXG4gKiAgc2hvdWxkIGJlIDEsIGFuZCBjbGFzcyAzIHNob3VsZCBiZSAyLiBcbiAqXG4gKiBgYGBqc1xuICogdGYub25lSG90KHRmLnRlbnNvcjFkKFswLCAxXSwgJ2ludDMyJyksIDMpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gaW5kaWNlcyBgdGYuVGVuc29yYCBvZiBpbmRpY2VzIHdpdGggZHR5cGUgYGludDMyYC4gSW5kaWNlcyBtdXN0IFxuICogc3RhcnQgZnJvbSAwLlxuICogQHBhcmFtIGRlcHRoIFRoZSBkZXB0aCBvZiB0aGUgb25lIGhvdCBkaW1lbnNpb24uXG4gKiBAcGFyYW0gb25WYWx1ZSBBIG51bWJlciB1c2VkIHRvIGZpbGwgaW4gdGhlIG91dHB1dCB3aGVuIHRoZSBpbmRleCBtYXRjaGVzXG4gKiB0aGUgbG9jYXRpb24uXG4gKiBAcGFyYW0gb2ZmVmFsdWUgQSBudW1iZXIgdXNlZCB0byBmaWxsIGluIHRoZSBvdXRwdXQgd2hlbiB0aGUgaW5kZXggZG9lc1xuICogICAgIG5vdCBtYXRjaCB0aGUgbG9jYXRpb24uXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnQ3JlYXRpb24nfVxuICovXG5mdW5jdGlvbiBvbmVIb3RfKFxuICAgIGluZGljZXM6IFRlbnNvcnxUZW5zb3JMaWtlLCBkZXB0aDogbnVtYmVyLCBvblZhbHVlID0gMSxcbiAgICBvZmZWYWx1ZSA9IDApOiBUZW5zb3Ige1xuICBpZiAoZGVwdGggPCAyKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBFcnJvciBpbiBvbmVIb3Q6IGRlcHRoIG11c3QgYmUgPj0yLCBidXQgaXQgaXMgJHtkZXB0aH1gKTtcbiAgfVxuICBjb25zdCAkaW5kaWNlcyA9IGNvbnZlcnRUb1RlbnNvcihpbmRpY2VzLCAnaW5kaWNlcycsICdvbmVIb3QnLCAnaW50MzInKTtcblxuICBjb25zdCBpbnB1dHM6IE9uZUhvdElucHV0cyA9IHtpbmRpY2VzOiAkaW5kaWNlc307XG4gIGNvbnN0IGF0dHJzOiBPbmVIb3RBdHRycyA9IHtkZXB0aCwgb25WYWx1ZSwgb2ZmVmFsdWV9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgT25lSG90LCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IG9uZUhvdCA9IG9wKHtvbmVIb3RffSk7XG4iXX0=