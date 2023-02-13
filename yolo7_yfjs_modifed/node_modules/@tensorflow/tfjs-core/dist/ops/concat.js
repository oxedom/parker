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
import { Concat } from '../kernel_names';
import { convertToTensorArray } from '../tensor_util_env';
import { assert } from '../util';
import { clone } from './clone';
import { op } from './operation';
/**
 * Concatenates a list of `tf.Tensor`s along a given axis.
 *
 * The tensors ranks and types must match, and their sizes must match in all
 * dimensions except `axis`.
 *
 * Also available are stricter rank-specific methods that assert that
 * `tensors` are of the given rank:
 *   - `tf.concat1d`
 *   - `tf.concat2d`
 *   - `tf.concat3d`
 *   - `tf.concat4d`
 *
 * Except `tf.concat1d` (which does not have axis param), all methods have
 * same signature as this method.
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * a.concat(b).print();  // or a.concat(b)
 * ```
 *
 * ```js
 * const a = tf.tensor1d([1, 2]);
 * const b = tf.tensor1d([3, 4]);
 * const c = tf.tensor1d([5, 6]);
 * tf.concat([a, b, c]).print();
 * ```
 *
 * ```js
 * const a = tf.tensor2d([[1, 2], [10, 20]]);
 * const b = tf.tensor2d([[3, 4], [30, 40]]);
 * const axis = 1;
 * tf.concat([a, b], axis).print();
 * ```
 * @param tensors A list of tensors to concatenate.
 * @param axis The axis to concate along. Defaults to 0 (the first dim).
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function concat_(tensors, axis = 0) {
    assert(tensors.length >= 1, () => 'Pass at least one tensor to concat');
    const $tensors = convertToTensorArray(tensors, 'tensors', 'concat', 'string_or_numeric');
    if ($tensors[0].dtype === 'complex64') {
        $tensors.forEach(tensor => {
            if (tensor.dtype !== 'complex64') {
                throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${tensor.dtype}. `);
            }
        });
    }
    if ($tensors.length === 1) {
        return clone($tensors[0]);
    }
    const inputs = $tensors;
    const attr = { axis };
    return ENGINE.runKernel(Concat, inputs, attr);
}
export const concat = op({ concat_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29uY2F0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvY29uY2F0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLE1BQU0sRUFBNEIsTUFBTSxpQkFBaUIsQ0FBQztBQUlsRSxPQUFPLEVBQUMsb0JBQW9CLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUV4RCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBRS9CLE9BQU8sRUFBQyxLQUFLLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFDOUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBdUNHO0FBQ0gsU0FBUyxPQUFPLENBQW1CLE9BQTRCLEVBQUUsSUFBSSxHQUFHLENBQUM7SUFDdkUsTUFBTSxDQUFDLE9BQU8sQ0FBQyxNQUFNLElBQUksQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLG9DQUFvQyxDQUFDLENBQUM7SUFFeEUsTUFBTSxRQUFRLEdBQ1Ysb0JBQW9CLENBQUMsT0FBTyxFQUFFLFNBQVMsRUFBRSxRQUFRLEVBQUUsbUJBQW1CLENBQUMsQ0FBQztJQUU1RSxJQUFJLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLEtBQUssV0FBVyxFQUFFO1FBQ3JDLFFBQVEsQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLEVBQUU7WUFDeEIsSUFBSSxNQUFNLENBQUMsS0FBSyxLQUFLLFdBQVcsRUFBRTtnQkFDaEMsTUFBTSxJQUFJLEtBQUssQ0FBQzt1QkFDRCxNQUFNLENBQUMsS0FBSyxJQUFJLENBQUMsQ0FBQzthQUNsQztRQUNILENBQUMsQ0FBQyxDQUFDO0tBQ0o7SUFFRCxJQUFJLFFBQVEsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1FBQ3pCLE9BQU8sS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQzNCO0lBRUQsTUFBTSxNQUFNLEdBQWlCLFFBQVEsQ0FBQztJQUN0QyxNQUFNLElBQUksR0FBZ0IsRUFBQyxJQUFJLEVBQUMsQ0FBQztJQUVqQyxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQ25CLE1BQU0sRUFBRSxNQUE4QixFQUFFLElBQTBCLENBQUMsQ0FBQztBQUMxRSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sTUFBTSxHQUFHLEVBQUUsQ0FBQyxFQUFDLE9BQU8sRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7Q29uY2F0LCBDb25jYXRBdHRycywgQ29uY2F0SW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvckFycmF5fSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2Fzc2VydH0gZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7Y2xvbmV9IGZyb20gJy4vY2xvbmUnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIENvbmNhdGVuYXRlcyBhIGxpc3Qgb2YgYHRmLlRlbnNvcmBzIGFsb25nIGEgZ2l2ZW4gYXhpcy5cbiAqXG4gKiBUaGUgdGVuc29ycyByYW5rcyBhbmQgdHlwZXMgbXVzdCBtYXRjaCwgYW5kIHRoZWlyIHNpemVzIG11c3QgbWF0Y2ggaW4gYWxsXG4gKiBkaW1lbnNpb25zIGV4Y2VwdCBgYXhpc2AuXG4gKlxuICogQWxzbyBhdmFpbGFibGUgYXJlIHN0cmljdGVyIHJhbmstc3BlY2lmaWMgbWV0aG9kcyB0aGF0IGFzc2VydCB0aGF0XG4gKiBgdGVuc29yc2AgYXJlIG9mIHRoZSBnaXZlbiByYW5rOlxuICogICAtIGB0Zi5jb25jYXQxZGBcbiAqICAgLSBgdGYuY29uY2F0MmRgXG4gKiAgIC0gYHRmLmNvbmNhdDNkYFxuICogICAtIGB0Zi5jb25jYXQ0ZGBcbiAqXG4gKiBFeGNlcHQgYHRmLmNvbmNhdDFkYCAod2hpY2ggZG9lcyBub3QgaGF2ZSBheGlzIHBhcmFtKSwgYWxsIG1ldGhvZHMgaGF2ZVxuICogc2FtZSBzaWduYXR1cmUgYXMgdGhpcyBtZXRob2QuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMSwgMl0pO1xuICogY29uc3QgYiA9IHRmLnRlbnNvcjFkKFszLCA0XSk7XG4gKiBhLmNvbmNhdChiKS5wcmludCgpOyAgLy8gb3IgYS5jb25jYXQoYilcbiAqIGBgYFxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBhID0gdGYudGVuc29yMWQoWzEsIDJdKTtcbiAqIGNvbnN0IGIgPSB0Zi50ZW5zb3IxZChbMywgNF0pO1xuICogY29uc3QgYyA9IHRmLnRlbnNvcjFkKFs1LCA2XSk7XG4gKiB0Zi5jb25jYXQoW2EsIGIsIGNdKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IyZChbWzEsIDJdLCBbMTAsIDIwXV0pO1xuICogY29uc3QgYiA9IHRmLnRlbnNvcjJkKFtbMywgNF0sIFszMCwgNDBdXSk7XG4gKiBjb25zdCBheGlzID0gMTtcbiAqIHRmLmNvbmNhdChbYSwgYl0sIGF4aXMpLnByaW50KCk7XG4gKiBgYGBcbiAqIEBwYXJhbSB0ZW5zb3JzIEEgbGlzdCBvZiB0ZW5zb3JzIHRvIGNvbmNhdGVuYXRlLlxuICogQHBhcmFtIGF4aXMgVGhlIGF4aXMgdG8gY29uY2F0ZSBhbG9uZy4gRGVmYXVsdHMgdG8gMCAodGhlIGZpcnN0IGRpbSkuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnU2xpY2luZyBhbmQgSm9pbmluZyd9XG4gKi9cbmZ1bmN0aW9uIGNvbmNhdF88VCBleHRlbmRzIFRlbnNvcj4odGVuc29yczogQXJyYXk8VHxUZW5zb3JMaWtlPiwgYXhpcyA9IDApOiBUIHtcbiAgYXNzZXJ0KHRlbnNvcnMubGVuZ3RoID49IDEsICgpID0+ICdQYXNzIGF0IGxlYXN0IG9uZSB0ZW5zb3IgdG8gY29uY2F0Jyk7XG5cbiAgY29uc3QgJHRlbnNvcnMgPVxuICAgICAgY29udmVydFRvVGVuc29yQXJyYXkodGVuc29ycywgJ3RlbnNvcnMnLCAnY29uY2F0JywgJ3N0cmluZ19vcl9udW1lcmljJyk7XG5cbiAgaWYgKCR0ZW5zb3JzWzBdLmR0eXBlID09PSAnY29tcGxleDY0Jykge1xuICAgICR0ZW5zb3JzLmZvckVhY2godGVuc29yID0+IHtcbiAgICAgIGlmICh0ZW5zb3IuZHR5cGUgIT09ICdjb21wbGV4NjQnKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihgQ2Fubm90IGNvbmNhdGVuYXRlIGNvbXBsZXg2NCB0ZW5zb3JzIHdpdGggYSB0ZW5zb3JcbiAgICAgICAgICB3aXRoIGR0eXBlICR7dGVuc29yLmR0eXBlfS4gYCk7XG4gICAgICB9XG4gICAgfSk7XG4gIH1cblxuICBpZiAoJHRlbnNvcnMubGVuZ3RoID09PSAxKSB7XG4gICAgcmV0dXJuIGNsb25lKCR0ZW5zb3JzWzBdKTtcbiAgfVxuXG4gIGNvbnN0IGlucHV0czogQ29uY2F0SW5wdXRzID0gJHRlbnNvcnM7XG4gIGNvbnN0IGF0dHI6IENvbmNhdEF0dHJzID0ge2F4aXN9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgQ29uY2F0LCBpbnB1dHMgYXMge30gYXMgTmFtZWRUZW5zb3JNYXAsIGF0dHIgYXMge30gYXMgTmFtZWRBdHRyTWFwKTtcbn1cblxuZXhwb3J0IGNvbnN0IGNvbmNhdCA9IG9wKHtjb25jYXRffSk7XG4iXX0=