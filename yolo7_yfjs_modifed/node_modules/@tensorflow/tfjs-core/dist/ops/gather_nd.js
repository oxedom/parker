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
import { GatherNd } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Gather slices from input tensor into a Tensor with shape specified by
 * `indices`.
 *
 * `indices` is an K-dimensional integer tensor, best thought of as a
 * (K-1)-dimensional tensor of indices into input, where each element defines a
 * slice of input:
 * output[\\(i_0, ..., i_{K-2}\\)] = input[indices[\\(i_0, ..., i_{K-2}\\)]]
 *
 * Whereas in `tf.gather`, `indices` defines slices into the first dimension of
 * input, in `tf.gatherND`, `indices` defines slices into the first N dimensions
 * of input, where N = indices.shape[-1].
 *
 * The last dimension of indices can be at most the rank of input:
 * indices.shape[-1] <= input.rank
 *
 * The last dimension of `indices` corresponds to elements
 * (if indices.shape[-1] == input.rank) or slices
 * (if indices.shape[-1] < input.rank) along dimension indices.shape[-1] of
 * input.
 * The output tensor has shape
 * indices.shape[:-1] + input.shape[indices.shape[-1]:]
 *
 * Note that on CPU, if an out of bound index is found, an error is returned. On
 * GPU, if an out of bound index is found, a 0 is stored in the corresponding
 * output value.
 *
 * ```js
 * const indices = tf.tensor2d([0, 1, 1, 0], [2,2], 'int32');
 * const input = tf.tensor2d([9, 10, 11, 12], [2, 2]);
 * tf.gatherND(input, indices).print() // [10, 11]
 * ```
 *
 * @param x The tensor from which to gather values.
 * @param indices Index tensor, must be of type int32.
 *
 * @doc {heading: 'Operations', subheading: 'Slicing and Joining'}
 */
function gatherND_(x, indices) {
    const $indices = convertToTensor(indices, 'indices', 'gatherND', 'int32');
    const $x = convertToTensor(x, 'x', 'gatherND', 'string_or_numeric');
    const inputs = { params: $x, indices: $indices };
    return ENGINE.runKernel(GatherNd, inputs);
}
export const gatherND = op({ gatherND_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ2F0aGVyX25kLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvZ2F0aGVyX25kLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLFFBQVEsRUFBaUIsTUFBTSxpQkFBaUIsQ0FBQztBQUd6RCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXFDRztBQUNILFNBQVMsU0FBUyxDQUFDLENBQW9CLEVBQUUsT0FBMEI7SUFDakUsTUFBTSxRQUFRLEdBQUcsZUFBZSxDQUFDLE9BQU8sRUFBRSxTQUFTLEVBQUUsVUFBVSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQzFFLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFVBQVUsRUFBRSxtQkFBbUIsQ0FBQyxDQUFDO0lBRXBFLE1BQU0sTUFBTSxHQUFtQixFQUFDLE1BQU0sRUFBRSxFQUFFLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBQyxDQUFDO0lBRS9ELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FBQyxRQUFRLEVBQUUsTUFBOEIsQ0FBQyxDQUFDO0FBQ3BFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxRQUFRLEdBQUcsRUFBRSxDQUFDLEVBQUMsU0FBUyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtHYXRoZXJOZCwgR2F0aGVyTmRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIEdhdGhlciBzbGljZXMgZnJvbSBpbnB1dCB0ZW5zb3IgaW50byBhIFRlbnNvciB3aXRoIHNoYXBlIHNwZWNpZmllZCBieVxuICogYGluZGljZXNgLlxuICpcbiAqIGBpbmRpY2VzYCBpcyBhbiBLLWRpbWVuc2lvbmFsIGludGVnZXIgdGVuc29yLCBiZXN0IHRob3VnaHQgb2YgYXMgYVxuICogKEstMSktZGltZW5zaW9uYWwgdGVuc29yIG9mIGluZGljZXMgaW50byBpbnB1dCwgd2hlcmUgZWFjaCBlbGVtZW50IGRlZmluZXMgYVxuICogc2xpY2Ugb2YgaW5wdXQ6XG4gKiBvdXRwdXRbXFxcXChpXzAsIC4uLiwgaV97Sy0yfVxcXFwpXSA9IGlucHV0W2luZGljZXNbXFxcXChpXzAsIC4uLiwgaV97Sy0yfVxcXFwpXV1cbiAqXG4gKiBXaGVyZWFzIGluIGB0Zi5nYXRoZXJgLCBgaW5kaWNlc2AgZGVmaW5lcyBzbGljZXMgaW50byB0aGUgZmlyc3QgZGltZW5zaW9uIG9mXG4gKiBpbnB1dCwgaW4gYHRmLmdhdGhlck5EYCwgYGluZGljZXNgIGRlZmluZXMgc2xpY2VzIGludG8gdGhlIGZpcnN0IE4gZGltZW5zaW9uc1xuICogb2YgaW5wdXQsIHdoZXJlIE4gPSBpbmRpY2VzLnNoYXBlWy0xXS5cbiAqXG4gKiBUaGUgbGFzdCBkaW1lbnNpb24gb2YgaW5kaWNlcyBjYW4gYmUgYXQgbW9zdCB0aGUgcmFuayBvZiBpbnB1dDpcbiAqIGluZGljZXMuc2hhcGVbLTFdIDw9IGlucHV0LnJhbmtcbiAqXG4gKiBUaGUgbGFzdCBkaW1lbnNpb24gb2YgYGluZGljZXNgIGNvcnJlc3BvbmRzIHRvIGVsZW1lbnRzXG4gKiAoaWYgaW5kaWNlcy5zaGFwZVstMV0gPT0gaW5wdXQucmFuaykgb3Igc2xpY2VzXG4gKiAoaWYgaW5kaWNlcy5zaGFwZVstMV0gPCBpbnB1dC5yYW5rKSBhbG9uZyBkaW1lbnNpb24gaW5kaWNlcy5zaGFwZVstMV0gb2ZcbiAqIGlucHV0LlxuICogVGhlIG91dHB1dCB0ZW5zb3IgaGFzIHNoYXBlXG4gKiBpbmRpY2VzLnNoYXBlWzotMV0gKyBpbnB1dC5zaGFwZVtpbmRpY2VzLnNoYXBlWy0xXTpdXG4gKlxuICogTm90ZSB0aGF0IG9uIENQVSwgaWYgYW4gb3V0IG9mIGJvdW5kIGluZGV4IGlzIGZvdW5kLCBhbiBlcnJvciBpcyByZXR1cm5lZC4gT25cbiAqIEdQVSwgaWYgYW4gb3V0IG9mIGJvdW5kIGluZGV4IGlzIGZvdW5kLCBhIDAgaXMgc3RvcmVkIGluIHRoZSBjb3JyZXNwb25kaW5nXG4gKiBvdXRwdXQgdmFsdWUuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGluZGljZXMgPSB0Zi50ZW5zb3IyZChbMCwgMSwgMSwgMF0sIFsyLDJdLCAnaW50MzInKTtcbiAqIGNvbnN0IGlucHV0ID0gdGYudGVuc29yMmQoWzksIDEwLCAxMSwgMTJdLCBbMiwgMl0pO1xuICogdGYuZ2F0aGVyTkQoaW5wdXQsIGluZGljZXMpLnByaW50KCkgLy8gWzEwLCAxMV1cbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4IFRoZSB0ZW5zb3IgZnJvbSB3aGljaCB0byBnYXRoZXIgdmFsdWVzLlxuICogQHBhcmFtIGluZGljZXMgSW5kZXggdGVuc29yLCBtdXN0IGJlIG9mIHR5cGUgaW50MzIuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnU2xpY2luZyBhbmQgSm9pbmluZyd9XG4gKi9cbmZ1bmN0aW9uIGdhdGhlck5EXyh4OiBUZW5zb3J8VGVuc29yTGlrZSwgaW5kaWNlczogVGVuc29yfFRlbnNvckxpa2UpOiBUZW5zb3Ige1xuICBjb25zdCAkaW5kaWNlcyA9IGNvbnZlcnRUb1RlbnNvcihpbmRpY2VzLCAnaW5kaWNlcycsICdnYXRoZXJORCcsICdpbnQzMicpO1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdnYXRoZXJORCcsICdzdHJpbmdfb3JfbnVtZXJpYycpO1xuXG4gIGNvbnN0IGlucHV0czogR2F0aGVyTmRJbnB1dHMgPSB7cGFyYW1zOiAkeCwgaW5kaWNlczogJGluZGljZXN9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKEdhdGhlck5kLCBpbnB1dHMgYXMge30gYXMgTmFtZWRUZW5zb3JNYXApO1xufVxuXG5leHBvcnQgY29uc3QgZ2F0aGVyTkQgPSBvcCh7Z2F0aGVyTkRffSk7XG4iXX0=