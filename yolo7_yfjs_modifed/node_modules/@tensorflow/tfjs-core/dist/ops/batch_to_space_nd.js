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
import { BatchToSpaceND } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * This operation reshapes the "batch" dimension 0 into `M + 1` dimensions of
 * shape `blockShape + [batch]`, interleaves these blocks back into the grid
 * defined by the spatial dimensions `[1, ..., M]`, to obtain a result with
 * the same rank as the input. The spatial dimensions of this intermediate
 * result are then optionally cropped according to `crops` to produce the
 * output. This is the reverse of `tf.spaceToBatchND`. See below for a precise
 * description.
 *
 * ```js
 * const x = tf.tensor4d([1, 2, 3, 4], [4, 1, 1, 1]);
 * const blockShape = [2, 2];
 * const crops = [[0, 0], [0, 0]];
 *
 * x.batchToSpaceND(blockShape, crops).print();
 * ```
 *
 * @param x A `tf.Tensor`. N-D with `x.shape` = `[batch] + spatialShape +
 * remainingShape`, where spatialShape has `M` dimensions.
 * @param blockShape A 1-D array. Must have shape `[M]`, all values must
 * be >= 1.
 * @param crops A 2-D array.  Must have shape `[M, 2]`, all values must be >= 0.
 * `crops[i] = [cropStart, cropEnd]` specifies the amount to crop from input
 * dimension `i + 1`, which corresponds to spatial dimension `i`. It is required
 * that `cropStart[i] + cropEnd[i] <= blockShape[i] * inputShape[i + 1]`
 *
 * This operation is equivalent to the following steps:
 *
 * 1. Reshape `x` to `reshaped` of shape: `[blockShape[0], ...,
 * blockShape[M-1], batch / prod(blockShape), x.shape[1], ...,
 * x.shape[N-1]]`
 *
 * 2. Permute dimensions of `reshaped`to produce `permuted` of shape `[batch /
 * prod(blockShape),x.shape[1], blockShape[0], ..., x.shape[M],
 * blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
 *
 * 3. Reshape `permuted` to produce `reshapedPermuted` of shape `[batch /
 * prod(blockShape),x.shape[1] * blockShape[0], ..., x.shape[M] *
 * blockShape[M-1],x.shape[M+1], ..., x.shape[N-1]]`
 *
 * 4. Crop the start and end of dimensions `[1, ..., M]` of `reshapedPermuted`
 * according to `crops` to produce the output of shape: `[batch /
 * prod(blockShape),x.shape[1] * blockShape[0] - crops[0,0] - crops[0,1],
 * ..., x.shape[M] * blockShape[M-1] - crops[M-1,0] -
 * crops[M-1,1],x.shape[M+1], ..., x.shape[N-1]]`
 *
 * @doc {heading: 'Tensors', subheading: 'Transformations'}
 */
function batchToSpaceND_(x, blockShape, crops) {
    const $x = convertToTensor(x, 'x', 'batchToSpaceND');
    const prod = blockShape.reduce((a, b) => a * b);
    util.assert($x.rank >= 1 + blockShape.length, () => `input rank is ${$x.rank} but should be > than blockShape.length ${blockShape.length}`);
    util.assert(crops.length === blockShape.length, () => `crops.length is ${crops.length} but should be equal to blockShape.length  ${blockShape.length}`);
    util.assert($x.shape[0] % prod === 0, () => `input tensor batch is ${$x.shape[0]} but is not divisible by the product of ` +
        `the elements of blockShape ${blockShape.join(' * ')} === ${prod}`);
    const inputs = { x: $x };
    const attrs = { blockShape, crops };
    return ENGINE.runKernel(BatchToSpaceND, inputs, attrs);
}
export const batchToSpaceND = op({ batchToSpaceND_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmF0Y2hfdG9fc3BhY2VfbmQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9iYXRjaF90b19zcGFjZV9uZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxjQUFjLEVBQTRDLE1BQU0saUJBQWlCLENBQUM7QUFJMUYsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBK0NHO0FBQ0gsU0FBUyxlQUFlLENBQ3BCLENBQWUsRUFBRSxVQUFvQixFQUFFLEtBQWlCO0lBQzFELE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLGdCQUFnQixDQUFDLENBQUM7SUFDckQsTUFBTSxJQUFJLEdBQUcsVUFBVSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUVoRCxJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLElBQUksQ0FBQyxHQUFHLFVBQVUsQ0FBQyxNQUFNLEVBQ2hDLEdBQUcsRUFBRSxDQUFDLGlCQUFpQixFQUFFLENBQUMsSUFBSSwyQ0FDMUIsVUFBVSxDQUFDLE1BQU0sRUFBRSxDQUFDLENBQUM7SUFFN0IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxLQUFLLENBQUMsTUFBTSxLQUFLLFVBQVUsQ0FBQyxNQUFNLEVBQ2xDLEdBQUcsRUFBRSxDQUFDLG1CQUNGLEtBQUssQ0FBQyxNQUFNLDhDQUNaLFVBQVUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDO0lBRTdCLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLEtBQUssQ0FBQyxFQUN4QixHQUFHLEVBQUUsQ0FBQyx5QkFDSSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQywwQ0FBMEM7UUFDM0QsOEJBQThCLFVBQVUsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLFFBQVEsSUFBSSxFQUFFLENBQUMsQ0FBQztJQUU1RSxNQUFNLE1BQU0sR0FBeUIsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDN0MsTUFBTSxLQUFLLEdBQXdCLEVBQUMsVUFBVSxFQUFFLEtBQUssRUFBQyxDQUFDO0lBRXZELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsY0FBYyxFQUFFLE1BQThCLEVBQzlDLEtBQTJCLENBQUMsQ0FBQztBQUNuQyxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sY0FBYyxHQUFHLEVBQUUsQ0FBQyxFQUFDLGVBQWUsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtCYXRjaFRvU3BhY2VORCwgQmF0Y2hUb1NwYWNlTkRBdHRycywgQmF0Y2hUb1NwYWNlTkRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogVGhpcyBvcGVyYXRpb24gcmVzaGFwZXMgdGhlIFwiYmF0Y2hcIiBkaW1lbnNpb24gMCBpbnRvIGBNICsgMWAgZGltZW5zaW9ucyBvZlxuICogc2hhcGUgYGJsb2NrU2hhcGUgKyBbYmF0Y2hdYCwgaW50ZXJsZWF2ZXMgdGhlc2UgYmxvY2tzIGJhY2sgaW50byB0aGUgZ3JpZFxuICogZGVmaW5lZCBieSB0aGUgc3BhdGlhbCBkaW1lbnNpb25zIGBbMSwgLi4uLCBNXWAsIHRvIG9idGFpbiBhIHJlc3VsdCB3aXRoXG4gKiB0aGUgc2FtZSByYW5rIGFzIHRoZSBpbnB1dC4gVGhlIHNwYXRpYWwgZGltZW5zaW9ucyBvZiB0aGlzIGludGVybWVkaWF0ZVxuICogcmVzdWx0IGFyZSB0aGVuIG9wdGlvbmFsbHkgY3JvcHBlZCBhY2NvcmRpbmcgdG8gYGNyb3BzYCB0byBwcm9kdWNlIHRoZVxuICogb3V0cHV0LiBUaGlzIGlzIHRoZSByZXZlcnNlIG9mIGB0Zi5zcGFjZVRvQmF0Y2hORGAuIFNlZSBiZWxvdyBmb3IgYSBwcmVjaXNlXG4gKiBkZXNjcmlwdGlvbi5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjRkKFsxLCAyLCAzLCA0XSwgWzQsIDEsIDEsIDFdKTtcbiAqIGNvbnN0IGJsb2NrU2hhcGUgPSBbMiwgMl07XG4gKiBjb25zdCBjcm9wcyA9IFtbMCwgMF0sIFswLCAwXV07XG4gKlxuICogeC5iYXRjaFRvU3BhY2VORChibG9ja1NoYXBlLCBjcm9wcykucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB4IEEgYHRmLlRlbnNvcmAuIE4tRCB3aXRoIGB4LnNoYXBlYCA9IGBbYmF0Y2hdICsgc3BhdGlhbFNoYXBlICtcbiAqIHJlbWFpbmluZ1NoYXBlYCwgd2hlcmUgc3BhdGlhbFNoYXBlIGhhcyBgTWAgZGltZW5zaW9ucy5cbiAqIEBwYXJhbSBibG9ja1NoYXBlIEEgMS1EIGFycmF5LiBNdXN0IGhhdmUgc2hhcGUgYFtNXWAsIGFsbCB2YWx1ZXMgbXVzdFxuICogYmUgPj0gMS5cbiAqIEBwYXJhbSBjcm9wcyBBIDItRCBhcnJheS4gIE11c3QgaGF2ZSBzaGFwZSBgW00sIDJdYCwgYWxsIHZhbHVlcyBtdXN0IGJlID49IDAuXG4gKiBgY3JvcHNbaV0gPSBbY3JvcFN0YXJ0LCBjcm9wRW5kXWAgc3BlY2lmaWVzIHRoZSBhbW91bnQgdG8gY3JvcCBmcm9tIGlucHV0XG4gKiBkaW1lbnNpb24gYGkgKyAxYCwgd2hpY2ggY29ycmVzcG9uZHMgdG8gc3BhdGlhbCBkaW1lbnNpb24gYGlgLiBJdCBpcyByZXF1aXJlZFxuICogdGhhdCBgY3JvcFN0YXJ0W2ldICsgY3JvcEVuZFtpXSA8PSBibG9ja1NoYXBlW2ldICogaW5wdXRTaGFwZVtpICsgMV1gXG4gKlxuICogVGhpcyBvcGVyYXRpb24gaXMgZXF1aXZhbGVudCB0byB0aGUgZm9sbG93aW5nIHN0ZXBzOlxuICpcbiAqIDEuIFJlc2hhcGUgYHhgIHRvIGByZXNoYXBlZGAgb2Ygc2hhcGU6IGBbYmxvY2tTaGFwZVswXSwgLi4uLFxuICogYmxvY2tTaGFwZVtNLTFdLCBiYXRjaCAvIHByb2QoYmxvY2tTaGFwZSksIHguc2hhcGVbMV0sIC4uLixcbiAqIHguc2hhcGVbTi0xXV1gXG4gKlxuICogMi4gUGVybXV0ZSBkaW1lbnNpb25zIG9mIGByZXNoYXBlZGB0byBwcm9kdWNlIGBwZXJtdXRlZGAgb2Ygc2hhcGUgYFtiYXRjaCAvXG4gKiBwcm9kKGJsb2NrU2hhcGUpLHguc2hhcGVbMV0sIGJsb2NrU2hhcGVbMF0sIC4uLiwgeC5zaGFwZVtNXSxcbiAqIGJsb2NrU2hhcGVbTS0xXSx4LnNoYXBlW00rMV0sIC4uLiwgeC5zaGFwZVtOLTFdXWBcbiAqXG4gKiAzLiBSZXNoYXBlIGBwZXJtdXRlZGAgdG8gcHJvZHVjZSBgcmVzaGFwZWRQZXJtdXRlZGAgb2Ygc2hhcGUgYFtiYXRjaCAvXG4gKiBwcm9kKGJsb2NrU2hhcGUpLHguc2hhcGVbMV0gKiBibG9ja1NoYXBlWzBdLCAuLi4sIHguc2hhcGVbTV0gKlxuICogYmxvY2tTaGFwZVtNLTFdLHguc2hhcGVbTSsxXSwgLi4uLCB4LnNoYXBlW04tMV1dYFxuICpcbiAqIDQuIENyb3AgdGhlIHN0YXJ0IGFuZCBlbmQgb2YgZGltZW5zaW9ucyBgWzEsIC4uLiwgTV1gIG9mIGByZXNoYXBlZFBlcm11dGVkYFxuICogYWNjb3JkaW5nIHRvIGBjcm9wc2AgdG8gcHJvZHVjZSB0aGUgb3V0cHV0IG9mIHNoYXBlOiBgW2JhdGNoIC9cbiAqIHByb2QoYmxvY2tTaGFwZSkseC5zaGFwZVsxXSAqIGJsb2NrU2hhcGVbMF0gLSBjcm9wc1swLDBdIC0gY3JvcHNbMCwxXSxcbiAqIC4uLiwgeC5zaGFwZVtNXSAqIGJsb2NrU2hhcGVbTS0xXSAtIGNyb3BzW00tMSwwXSAtXG4gKiBjcm9wc1tNLTEsMV0seC5zaGFwZVtNKzFdLCAuLi4sIHguc2hhcGVbTi0xXV1gXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ1RlbnNvcnMnLCBzdWJoZWFkaW5nOiAnVHJhbnNmb3JtYXRpb25zJ31cbiAqL1xuZnVuY3Rpb24gYmF0Y2hUb1NwYWNlTkRfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFR8VGVuc29yTGlrZSwgYmxvY2tTaGFwZTogbnVtYmVyW10sIGNyb3BzOiBudW1iZXJbXVtdKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2JhdGNoVG9TcGFjZU5EJyk7XG4gIGNvbnN0IHByb2QgPSBibG9ja1NoYXBlLnJlZHVjZSgoYSwgYikgPT4gYSAqIGIpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHgucmFuayA+PSAxICsgYmxvY2tTaGFwZS5sZW5ndGgsXG4gICAgICAoKSA9PiBgaW5wdXQgcmFuayBpcyAkeyR4LnJhbmt9IGJ1dCBzaG91bGQgYmUgPiB0aGFuIGJsb2NrU2hhcGUubGVuZ3RoICR7XG4gICAgICAgICAgYmxvY2tTaGFwZS5sZW5ndGh9YCk7XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICBjcm9wcy5sZW5ndGggPT09IGJsb2NrU2hhcGUubGVuZ3RoLFxuICAgICAgKCkgPT4gYGNyb3BzLmxlbmd0aCBpcyAke1xuICAgICAgICAgIGNyb3BzLmxlbmd0aH0gYnV0IHNob3VsZCBiZSBlcXVhbCB0byBibG9ja1NoYXBlLmxlbmd0aCAgJHtcbiAgICAgICAgICBibG9ja1NoYXBlLmxlbmd0aH1gKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgICR4LnNoYXBlWzBdICUgcHJvZCA9PT0gMCxcbiAgICAgICgpID0+IGBpbnB1dCB0ZW5zb3IgYmF0Y2ggaXMgJHtcbiAgICAgICAgICAgICAgICAkeC5zaGFwZVswXX0gYnV0IGlzIG5vdCBkaXZpc2libGUgYnkgdGhlIHByb2R1Y3Qgb2YgYCArXG4gICAgICAgICAgYHRoZSBlbGVtZW50cyBvZiBibG9ja1NoYXBlICR7YmxvY2tTaGFwZS5qb2luKCcgKiAnKX0gPT09ICR7cHJvZH1gKTtcblxuICBjb25zdCBpbnB1dHM6IEJhdGNoVG9TcGFjZU5ESW5wdXRzID0ge3g6ICR4fTtcbiAgY29uc3QgYXR0cnM6IEJhdGNoVG9TcGFjZU5EQXR0cnMgPSB7YmxvY2tTaGFwZSwgY3JvcHN9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgQmF0Y2hUb1NwYWNlTkQsIGlucHV0cyBhcyB7fSBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgIGF0dHJzIGFzIHt9IGFzIE5hbWVkQXR0ck1hcCk7XG59XG5cbmV4cG9ydCBjb25zdCBiYXRjaFRvU3BhY2VORCA9IG9wKHtiYXRjaFRvU3BhY2VORF99KTtcbiJdfQ==