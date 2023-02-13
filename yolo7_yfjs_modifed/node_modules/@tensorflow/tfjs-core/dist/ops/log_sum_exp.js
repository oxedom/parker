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
import { convertToTensor } from '../tensor_util_env';
import { parseAxisParam } from '../util';
import { add } from './add';
import { expandShapeToKeepDim } from './axis_util';
import { exp } from './exp';
import { log } from './log';
import { max } from './max';
import { op } from './operation';
import { reshape } from './reshape';
import { sub } from './sub';
import { sum } from './sum';
/**
 * Computes the log(sum(exp(elements across the reduction dimensions)).
 *
 * Reduces the input along the dimensions given in `axis`. Unless `keepDims`
 * is true, the rank of the array is reduced by 1 for each entry in `axis`.
 * If `keepDims` is true, the reduced dimensions are retained with length 1.
 * If `axis` has no entries, all dimensions are reduced, and an array with a
 * single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.logSumExp().print();  // or tf.logSumExp(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.logSumExp(axis).print();  // or tf.logSumExp(a, axis)
 * ```
 * @param x The input tensor.
 * @param axis The dimension(s) to reduce. If null (the default),
 *     reduces all dimensions.
 * @param keepDims If true, retains reduced dimensions with length
 *     of 1. Defaults to false.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function logSumExp_(x, axis = null, keepDims = false) {
    const $x = convertToTensor(x, 'x', 'logSumExp');
    const axes = parseAxisParam(axis, $x.shape);
    const xMax = max($x, axes, true /* keepDims */);
    const a = sub($x, xMax);
    const b = exp(a);
    const c = sum(b, axes);
    const d = log(c);
    const res = add(reshape(xMax, d.shape), d);
    if (keepDims) {
        const newShape = expandShapeToKeepDim(res.shape, axes);
        return reshape(res, newShape);
    }
    return res;
}
export const logSumExp = op({ logSumExp_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibG9nX3N1bV9leHAuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9sb2dfc3VtX2V4cC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUV2QyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxvQkFBb0IsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUNqRCxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBRTFCOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBNEJHO0FBQ0gsU0FBUyxVQUFVLENBQ2YsQ0FBb0IsRUFBRSxPQUF3QixJQUFJLEVBQUUsUUFBUSxHQUFHLEtBQUs7SUFDdEUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFFaEQsTUFBTSxJQUFJLEdBQUcsY0FBYyxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDNUMsTUFBTSxJQUFJLEdBQUcsR0FBRyxDQUFDLEVBQUUsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDO0lBQ2hELE1BQU0sQ0FBQyxHQUFHLEdBQUcsQ0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEIsTUFBTSxDQUFDLEdBQUcsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ2pCLE1BQU0sQ0FBQyxHQUFHLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDdkIsTUFBTSxDQUFDLEdBQUcsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ2pCLE1BQU0sR0FBRyxHQUFHLEdBQUcsQ0FBQyxPQUFPLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUUzQyxJQUFJLFFBQVEsRUFBRTtRQUNaLE1BQU0sUUFBUSxHQUFHLG9CQUFvQixDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDdkQsT0FBTyxPQUFPLENBQUMsR0FBRyxFQUFFLFFBQVEsQ0FBTSxDQUFDO0tBQ3BDO0lBQ0QsT0FBTyxHQUFRLENBQUM7QUFDbEIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFNBQVMsR0FBRyxFQUFFLENBQUMsRUFBQyxVQUFVLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge3BhcnNlQXhpc1BhcmFtfSBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHthZGR9IGZyb20gJy4vYWRkJztcbmltcG9ydCB7ZXhwYW5kU2hhcGVUb0tlZXBEaW19IGZyb20gJy4vYXhpc191dGlsJztcbmltcG9ydCB7ZXhwfSBmcm9tICcuL2V4cCc7XG5pbXBvcnQge2xvZ30gZnJvbSAnLi9sb2cnO1xuaW1wb3J0IHttYXh9IGZyb20gJy4vbWF4JztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9yZXNoYXBlJztcbmltcG9ydCB7c3VifSBmcm9tICcuL3N1Yic7XG5pbXBvcnQge3N1bX0gZnJvbSAnLi9zdW0nO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBsb2coc3VtKGV4cChlbGVtZW50cyBhY3Jvc3MgdGhlIHJlZHVjdGlvbiBkaW1lbnNpb25zKSkuXG4gKlxuICogUmVkdWNlcyB0aGUgaW5wdXQgYWxvbmcgdGhlIGRpbWVuc2lvbnMgZ2l2ZW4gaW4gYGF4aXNgLiBVbmxlc3MgYGtlZXBEaW1zYFxuICogaXMgdHJ1ZSwgdGhlIHJhbmsgb2YgdGhlIGFycmF5IGlzIHJlZHVjZWQgYnkgMSBmb3IgZWFjaCBlbnRyeSBpbiBgYXhpc2AuXG4gKiBJZiBga2VlcERpbXNgIGlzIHRydWUsIHRoZSByZWR1Y2VkIGRpbWVuc2lvbnMgYXJlIHJldGFpbmVkIHdpdGggbGVuZ3RoIDEuXG4gKiBJZiBgYXhpc2AgaGFzIG5vIGVudHJpZXMsIGFsbCBkaW1lbnNpb25zIGFyZSByZWR1Y2VkLCBhbmQgYW4gYXJyYXkgd2l0aCBhXG4gKiBzaW5nbGUgZWxlbWVudCBpcyByZXR1cm5lZC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gKlxuICogeC5sb2dTdW1FeHAoKS5wcmludCgpOyAgLy8gb3IgdGYubG9nU3VtRXhwKHgpXG4gKiBgYGBcbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFsxLCAyLCAzLCA0XSwgWzIsIDJdKTtcbiAqXG4gKiBjb25zdCBheGlzID0gMTtcbiAqIHgubG9nU3VtRXhwKGF4aXMpLnByaW50KCk7ICAvLyBvciB0Zi5sb2dTdW1FeHAoYSwgYXhpcylcbiAqIGBgYFxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvci5cbiAqIEBwYXJhbSBheGlzIFRoZSBkaW1lbnNpb24ocykgdG8gcmVkdWNlLiBJZiBudWxsICh0aGUgZGVmYXVsdCksXG4gKiAgICAgcmVkdWNlcyBhbGwgZGltZW5zaW9ucy5cbiAqIEBwYXJhbSBrZWVwRGltcyBJZiB0cnVlLCByZXRhaW5zIHJlZHVjZWQgZGltZW5zaW9ucyB3aXRoIGxlbmd0aFxuICogICAgIG9mIDEuIERlZmF1bHRzIHRvIGZhbHNlLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ1JlZHVjdGlvbid9XG4gKi9cbmZ1bmN0aW9uIGxvZ1N1bUV4cF88VCBleHRlbmRzIFRlbnNvcj4oXG4gICAgeDogVGVuc29yfFRlbnNvckxpa2UsIGF4aXM6IG51bWJlcnxudW1iZXJbXSA9IG51bGwsIGtlZXBEaW1zID0gZmFsc2UpOiBUIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnbG9nU3VtRXhwJyk7XG5cbiAgY29uc3QgYXhlcyA9IHBhcnNlQXhpc1BhcmFtKGF4aXMsICR4LnNoYXBlKTtcbiAgY29uc3QgeE1heCA9IG1heCgkeCwgYXhlcywgdHJ1ZSAvKiBrZWVwRGltcyAqLyk7XG4gIGNvbnN0IGEgPSBzdWIoJHgsIHhNYXgpO1xuICBjb25zdCBiID0gZXhwKGEpO1xuICBjb25zdCBjID0gc3VtKGIsIGF4ZXMpO1xuICBjb25zdCBkID0gbG9nKGMpO1xuICBjb25zdCByZXMgPSBhZGQocmVzaGFwZSh4TWF4LCBkLnNoYXBlKSwgZCk7XG5cbiAgaWYgKGtlZXBEaW1zKSB7XG4gICAgY29uc3QgbmV3U2hhcGUgPSBleHBhbmRTaGFwZVRvS2VlcERpbShyZXMuc2hhcGUsIGF4ZXMpO1xuICAgIHJldHVybiByZXNoYXBlKHJlcywgbmV3U2hhcGUpIGFzIFQ7XG4gIH1cbiAgcmV0dXJuIHJlcyBhcyBUO1xufVxuXG5leHBvcnQgY29uc3QgbG9nU3VtRXhwID0gb3Aoe2xvZ1N1bUV4cF99KTtcbiJdfQ==