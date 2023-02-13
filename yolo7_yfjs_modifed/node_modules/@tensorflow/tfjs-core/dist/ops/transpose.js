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
import { Transpose } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Transposes the `tf.Tensor`. Permutes the dimensions according to `perm`.
 *
 * The returned `tf.Tensor`'s dimension `i` will correspond to the input
 * dimension `perm[i]`. If `perm` is not given, it is set to `[n-1...0]`,
 * where `n` is the rank of the input `tf.Tensor`. Hence by default, this
 * operation performs a regular matrix transpose on 2-D input `tf.Tensor`s.
 *
 * ```js
 * const a = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3]);
 *
 * a.transpose().print();  // or tf.transpose(a)
 * ```
 *
 * @param x The tensor to transpose.
 * @param perm The permutation of the dimensions of a.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function transpose_(x, perm) {
    const $x = convertToTensor(x, 'x', 'transpose');
    if (perm == null) {
        perm = $x.shape.map((s, i) => i).reverse();
    }
    util.assert($x.rank === perm.length, () => `Error in transpose: rank of input ${$x.rank} ` +
        `must match length of perm ${perm}.`);
    perm.forEach(axis => {
        util.assert(axis >= 0 && axis < $x.rank, () => `All entries in 'perm' must be between 0 and ${$x.rank - 1}` +
            ` but got ${perm}`);
    });
    if ($x.rank <= 1) {
        return $x.clone();
    }
    const inputs = { x: $x };
    const attrs = { perm };
    return ENGINE.runKernel(Transpose, inputs, attrs);
}
export const transpose = op({ transpose_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidHJhbnNwb3NlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvdHJhbnNwb3NlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLFNBQVMsRUFBa0MsTUFBTSxpQkFBaUIsQ0FBQztBQUkzRSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBa0JHO0FBQ0gsU0FBUyxVQUFVLENBQW1CLENBQWUsRUFBRSxJQUFlO0lBQ3BFLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0lBRWhELElBQUksSUFBSSxJQUFJLElBQUksRUFBRTtRQUNoQixJQUFJLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLEVBQUUsQ0FBQztLQUM1QztJQUNELElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxJQUFJLENBQUMsTUFBTSxFQUN2QixHQUFHLEVBQUUsQ0FBQyxxQ0FBcUMsRUFBRSxDQUFDLElBQUksR0FBRztRQUNqRCw2QkFBNkIsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUM5QyxJQUFJLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxFQUFFO1FBQ2xCLElBQUksQ0FBQyxNQUFNLENBQ1AsSUFBSSxJQUFJLENBQUMsSUFBSSxJQUFJLEdBQUcsRUFBRSxDQUFDLElBQUksRUFDM0IsR0FBRyxFQUFFLENBQUMsK0NBQStDLEVBQUUsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxFQUFFO1lBQzlELFlBQVksSUFBSSxFQUFFLENBQUMsQ0FBQztJQUM5QixDQUFDLENBQUMsQ0FBQztJQUVILElBQUksRUFBRSxDQUFDLElBQUksSUFBSSxDQUFDLEVBQUU7UUFDaEIsT0FBTyxFQUFFLENBQUMsS0FBSyxFQUFFLENBQUM7S0FDbkI7SUFFRCxNQUFNLE1BQU0sR0FBb0IsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDeEMsTUFBTSxLQUFLLEdBQW1CLEVBQUMsSUFBSSxFQUFDLENBQUM7SUFFckMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixTQUFTLEVBQUUsTUFBOEIsRUFBRSxLQUEyQixDQUFDLENBQUM7QUFDOUUsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFNBQVMsR0FBRyxFQUFFLENBQUMsRUFBQyxVQUFVLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7VHJhbnNwb3NlLCBUcmFuc3Bvc2VBdHRycywgVHJhbnNwb3NlSW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIFRyYW5zcG9zZXMgdGhlIGB0Zi5UZW5zb3JgLiBQZXJtdXRlcyB0aGUgZGltZW5zaW9ucyBhY2NvcmRpbmcgdG8gYHBlcm1gLlxuICpcbiAqIFRoZSByZXR1cm5lZCBgdGYuVGVuc29yYCdzIGRpbWVuc2lvbiBgaWAgd2lsbCBjb3JyZXNwb25kIHRvIHRoZSBpbnB1dFxuICogZGltZW5zaW9uIGBwZXJtW2ldYC4gSWYgYHBlcm1gIGlzIG5vdCBnaXZlbiwgaXQgaXMgc2V0IHRvIGBbbi0xLi4uMF1gLFxuICogd2hlcmUgYG5gIGlzIHRoZSByYW5rIG9mIHRoZSBpbnB1dCBgdGYuVGVuc29yYC4gSGVuY2UgYnkgZGVmYXVsdCwgdGhpc1xuICogb3BlcmF0aW9uIHBlcmZvcm1zIGEgcmVndWxhciBtYXRyaXggdHJhbnNwb3NlIG9uIDItRCBpbnB1dCBgdGYuVGVuc29yYHMuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgNCwgNSwgNl0sIFsyLCAzXSk7XG4gKlxuICogYS50cmFuc3Bvc2UoKS5wcmludCgpOyAgLy8gb3IgdGYudHJhbnNwb3NlKGEpXG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geCBUaGUgdGVuc29yIHRvIHRyYW5zcG9zZS5cbiAqIEBwYXJhbSBwZXJtIFRoZSBwZXJtdXRhdGlvbiBvZiB0aGUgZGltZW5zaW9ucyBvZiBhLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ01hdHJpY2VzJ31cbiAqL1xuZnVuY3Rpb24gdHJhbnNwb3NlXzxUIGV4dGVuZHMgVGVuc29yPih4OiBUfFRlbnNvckxpa2UsIHBlcm0/OiBudW1iZXJbXSk6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICd0cmFuc3Bvc2UnKTtcblxuICBpZiAocGVybSA9PSBudWxsKSB7XG4gICAgcGVybSA9ICR4LnNoYXBlLm1hcCgocywgaSkgPT4gaSkucmV2ZXJzZSgpO1xuICB9XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHgucmFuayA9PT0gcGVybS5sZW5ndGgsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gdHJhbnNwb3NlOiByYW5rIG9mIGlucHV0ICR7JHgucmFua30gYCArXG4gICAgICAgICAgYG11c3QgbWF0Y2ggbGVuZ3RoIG9mIHBlcm0gJHtwZXJtfS5gKTtcbiAgcGVybS5mb3JFYWNoKGF4aXMgPT4ge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBheGlzID49IDAgJiYgYXhpcyA8ICR4LnJhbmssXG4gICAgICAgICgpID0+IGBBbGwgZW50cmllcyBpbiAncGVybScgbXVzdCBiZSBiZXR3ZWVuIDAgYW5kICR7JHgucmFuayAtIDF9YCArXG4gICAgICAgICAgICBgIGJ1dCBnb3QgJHtwZXJtfWApO1xuICB9KTtcblxuICBpZiAoJHgucmFuayA8PSAxKSB7XG4gICAgcmV0dXJuICR4LmNsb25lKCk7XG4gIH1cblxuICBjb25zdCBpbnB1dHM6IFRyYW5zcG9zZUlucHV0cyA9IHt4OiAkeH07XG4gIGNvbnN0IGF0dHJzOiBUcmFuc3Bvc2VBdHRycyA9IHtwZXJtfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIFRyYW5zcG9zZSwgaW5wdXRzIGFzIHt9IGFzIE5hbWVkVGVuc29yTWFwLCBhdHRycyBhcyB7fSBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3QgdHJhbnNwb3NlID0gb3Aoe3RyYW5zcG9zZV99KTtcbiJdfQ==