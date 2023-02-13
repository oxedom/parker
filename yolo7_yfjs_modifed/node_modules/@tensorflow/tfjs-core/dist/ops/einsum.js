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
import { ENGINE } from '../engine';
import { Einsum } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Tensor contraction over specified indices and outer product.
 *
 * `einsum` allows defining Tensors by defining their element-wise computation.
 * This computation is based on
 * [Einstein summation](https://en.wikipedia.org/wiki/Einstein_notation).
 *
 * Some special cases include:
 *
 * Matrix multiplication:
 * ```js
 * const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tf.tensor2d([[0, 1], [2, 3], [4, 5]]);
 * x.print();
 * y.print();
 * tf.einsum('ij,jk->ik', x, y).print();
 * ```
 *
 * Dot product:
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 * const y = tf.tensor1d([0, 1, 2]);
 * x.print();
 * y.print();
 * tf.einsum('i,i->', x, y).print();
 * ```
 *
 * Batch dot product:
 * ```js
 * const x = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
 * const y = tf.tensor2d([[0, 1, 2], [3, 4, 5]]);
 * x.print();
 * y.print();
 * tf.einsum('bi,bi->b', x, y).print();
 * ```
 *
 * Outer prouduct:
 * ```js
 * const x = tf.tensor1d([1, 3, 5]);
 * const y = tf.tensor1d([2, 4, 6]);
 * x.print();
 * y.print();
 * tf.einsum('i,j->ij', x, y).print();
 * ```
 *
 * Matrix transpose:
 * ```js
 * const x = tf.tensor2d([[1, 2], [3, 4]]);
 * x.print();
 * tf.einsum('ij->ji', x).print();
 * ```
 *
 * Batch matrix transpose:
 * ```js
 * const x = tf.tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]);
 * x.print();
 * tf.einsum('bij->bji', x).print();
 * ```
 *
 * Limitations:
 *
 * This implementation of einsum has the following limitations:
 *
 * - Does not support >2 input tensors.
 * - Does not support duplicate axes for any given input tensor. E.g., equation
 *   'ii->' is not suppoted.
 * - The `...` notation is not supported.
 *
 * @param equation a string describing the contraction, in the same format as
 * [numpy.einsum](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html).
 * @param tensors the input(s) to contract (each one a Tensor), whose shapes
 *     should be consistent with equation.
 * @returns The output tensor.
 *
 * @doc {heading: 'Tensors', subheading: 'Matrices'}
 */
export function einsum_(equation, ...tensors) {
    const $tensors = tensors.map((t, i) => convertToTensor(t, `tensors${i}`, 'einsum'));
    const attrs = { equation };
    return ENGINE.runKernel(Einsum, $tensors, attrs);
}
export const einsum = op({ einsum_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZWluc3VtLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvZWluc3VtLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLE1BQU0sRUFBYyxNQUFNLGlCQUFpQixDQUFDO0FBSXBELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0EyRUc7QUFDSCxNQUFNLFVBQVUsT0FBTyxDQUFDLFFBQWdCLEVBQUUsR0FBRyxPQUFpQjtJQUM1RCxNQUFNLFFBQVEsR0FDVixPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsZUFBZSxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsRUFBRSxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUM7SUFDdkUsTUFBTSxLQUFLLEdBQWdCLEVBQUMsUUFBUSxFQUFDLENBQUM7SUFDdEMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixNQUFNLEVBQUUsUUFBZ0MsRUFBRSxLQUEyQixDQUFDLENBQUM7QUFDN0UsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLE1BQU0sR0FBRyxFQUFFLENBQUMsRUFBQyxPQUFPLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjEgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7RWluc3VtLCBFaW5zdW1BdHRyc30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBUZW5zb3IgY29udHJhY3Rpb24gb3ZlciBzcGVjaWZpZWQgaW5kaWNlcyBhbmQgb3V0ZXIgcHJvZHVjdC5cbiAqXG4gKiBgZWluc3VtYCBhbGxvd3MgZGVmaW5pbmcgVGVuc29ycyBieSBkZWZpbmluZyB0aGVpciBlbGVtZW50LXdpc2UgY29tcHV0YXRpb24uXG4gKiBUaGlzIGNvbXB1dGF0aW9uIGlzIGJhc2VkIG9uXG4gKiBbRWluc3RlaW4gc3VtbWF0aW9uXShodHRwczovL2VuLndpa2lwZWRpYS5vcmcvd2lraS9FaW5zdGVpbl9ub3RhdGlvbikuXG4gKlxuICogU29tZSBzcGVjaWFsIGNhc2VzIGluY2x1ZGU6XG4gKlxuICogTWF0cml4IG11bHRpcGxpY2F0aW9uOlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IyZChbWzEsIDIsIDNdLCBbNCwgNSwgNl1dKTtcbiAqIGNvbnN0IHkgPSB0Zi50ZW5zb3IyZChbWzAsIDFdLCBbMiwgM10sIFs0LCA1XV0pO1xuICogeC5wcmludCgpO1xuICogeS5wcmludCgpO1xuICogdGYuZWluc3VtKCdpaixqay0+aWsnLCB4LCB5KS5wcmludCgpO1xuICogYGBgXG4gKlxuICogRG90IHByb2R1Y3Q6XG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gKiBjb25zdCB5ID0gdGYudGVuc29yMWQoWzAsIDEsIDJdKTtcbiAqIHgucHJpbnQoKTtcbiAqIHkucHJpbnQoKTtcbiAqIHRmLmVpbnN1bSgnaSxpLT4nLCB4LCB5KS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQmF0Y2ggZG90IHByb2R1Y3Q6XG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFtbMSwgMiwgM10sIFs0LCA1LCA2XV0pO1xuICogY29uc3QgeSA9IHRmLnRlbnNvcjJkKFtbMCwgMSwgMl0sIFszLCA0LCA1XV0pO1xuICogeC5wcmludCgpO1xuICogeS5wcmludCgpO1xuICogdGYuZWluc3VtKCdiaSxiaS0+YicsIHgsIHkpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBPdXRlciBwcm91ZHVjdDpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDMsIDVdKTtcbiAqIGNvbnN0IHkgPSB0Zi50ZW5zb3IxZChbMiwgNCwgNl0pO1xuICogeC5wcmludCgpO1xuICogeS5wcmludCgpO1xuICogdGYuZWluc3VtKCdpLGotPmlqJywgeCwgeSkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIE1hdHJpeCB0cmFuc3Bvc2U6XG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFtbMSwgMl0sIFszLCA0XV0pO1xuICogeC5wcmludCgpO1xuICogdGYuZWluc3VtKCdpai0+amknLCB4KS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQmF0Y2ggbWF0cml4IHRyYW5zcG9zZTpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yM2QoW1tbMSwgMl0sIFszLCA0XV0sIFtbLTEsIC0yXSwgWy0zLCAtNF1dXSk7XG4gKiB4LnByaW50KCk7XG4gKiB0Zi5laW5zdW0oJ2Jpai0+YmppJywgeCkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIExpbWl0YXRpb25zOlxuICpcbiAqIFRoaXMgaW1wbGVtZW50YXRpb24gb2YgZWluc3VtIGhhcyB0aGUgZm9sbG93aW5nIGxpbWl0YXRpb25zOlxuICpcbiAqIC0gRG9lcyBub3Qgc3VwcG9ydCA+MiBpbnB1dCB0ZW5zb3JzLlxuICogLSBEb2VzIG5vdCBzdXBwb3J0IGR1cGxpY2F0ZSBheGVzIGZvciBhbnkgZ2l2ZW4gaW5wdXQgdGVuc29yLiBFLmcuLCBlcXVhdGlvblxuICogICAnaWktPicgaXMgbm90IHN1cHBvdGVkLlxuICogLSBUaGUgYC4uLmAgbm90YXRpb24gaXMgbm90IHN1cHBvcnRlZC5cbiAqXG4gKiBAcGFyYW0gZXF1YXRpb24gYSBzdHJpbmcgZGVzY3JpYmluZyB0aGUgY29udHJhY3Rpb24sIGluIHRoZSBzYW1lIGZvcm1hdCBhc1xuICogW251bXB5LmVpbnN1bV0oaHR0cHM6Ly9udW1weS5vcmcvZG9jL3N0YWJsZS9yZWZlcmVuY2UvZ2VuZXJhdGVkL251bXB5LmVpbnN1bS5odG1sKS5cbiAqIEBwYXJhbSB0ZW5zb3JzIHRoZSBpbnB1dChzKSB0byBjb250cmFjdCAoZWFjaCBvbmUgYSBUZW5zb3IpLCB3aG9zZSBzaGFwZXNcbiAqICAgICBzaG91bGQgYmUgY29uc2lzdGVudCB3aXRoIGVxdWF0aW9uLlxuICogQHJldHVybnMgVGhlIG91dHB1dCB0ZW5zb3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnTWF0cmljZXMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gZWluc3VtXyhlcXVhdGlvbjogc3RyaW5nLCAuLi50ZW5zb3JzOiBUZW5zb3JbXSk6IFRlbnNvciB7XG4gIGNvbnN0ICR0ZW5zb3JzID1cbiAgICAgIHRlbnNvcnMubWFwKCh0LCBpKSA9PiBjb252ZXJ0VG9UZW5zb3IodCwgYHRlbnNvcnMke2l9YCwgJ2VpbnN1bScpKTtcbiAgY29uc3QgYXR0cnM6IEVpbnN1bUF0dHJzID0ge2VxdWF0aW9ufTtcbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICBFaW5zdW0sICR0ZW5zb3JzIGFzIHt9IGFzIE5hbWVkVGVuc29yTWFwLCBhdHRycyBhcyB7fSBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3QgZWluc3VtID0gb3Aoe2VpbnN1bV99KTtcbiJdfQ==