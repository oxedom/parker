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
import { convertToTensor } from '../tensor_util_env';
import { parseAxisParam } from '../util';
import { abs } from './abs';
import * as axis_util from './axis_util';
import { max } from './max';
import { min } from './min';
import { op } from './operation';
import { pow } from './pow';
import { reshape } from './reshape';
import { scalar } from './scalar';
import { sqrt } from './sqrt';
import { square } from './square';
import { sum } from './sum';
/**
 * Computes the norm of scalar, vectors, and matrices.
 * This function can compute several different vector norms (the 1-norm, the
 * Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0)
 * and matrix norms (Frobenius, 1-norm, and inf-norm).
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * x.norm().print();  // or tf.norm(x)
 * ```
 *
 * @param x The input array.
 * @param ord Optional. Order of the norm. Supported norm types are
 * following:
 *
 *  | ord        | norm for matrices         | norm for vectors
 *  |------------|---------------------------|---------------------
 *  |'euclidean' |Frobenius norm             |2-norm
 *  |'fro'       |Frobenius norm	           |
 *  |Infinity    |max(sum(abs(x), axis=1))   |max(abs(x))
 *  |-Infinity   |min(sum(abs(x), axis=1))   |min(abs(x))
 *  |1           |max(sum(abs(x), axis=0))   |sum(abs(x))
 *  |2           |                           |sum(abs(x)^2)^1/2*
 *
 * @param axis Optional. If axis is null (the default), the input is
 * considered a vector and a single vector norm is computed over the entire
 * set of values in the Tensor, i.e. norm(x, ord) is equivalent
 * to norm(x.reshape([-1]), ord). If axis is a integer, the input
 * is considered a batch of vectors, and axis determines the axis in x
 * over which to compute vector norms. If axis is a 2-tuple of integer it is
 * considered a batch of matrices and axis determines the axes in NDArray
 * over which to compute a matrix norm.
 * @param keepDims Optional. If true, the norm have the same dimensionality
 * as the input.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function norm_(x, ord = 'euclidean', axis = null, keepDims = false) {
    x = convertToTensor(x, 'x', 'norm');
    const norm = normImpl(x, ord, axis);
    let keepDimsShape = norm.shape;
    if (keepDims) {
        const axes = parseAxisParam(axis, x.shape);
        keepDimsShape = axis_util.expandShapeToKeepDim(norm.shape, axes);
    }
    return reshape(norm, keepDimsShape);
}
function normImpl(x, p, axis = null) {
    if (x.rank === 0) {
        return abs(x);
    }
    // consider vector when no axis is specified
    if (x.rank !== 1 && axis === null) {
        return normImpl(reshape(x, [-1]), p, axis);
    }
    // vector
    if (x.rank === 1 || typeof axis === 'number' ||
        Array.isArray(axis) && axis.length === 1) {
        if (p === 1) {
            return sum(abs(x), axis);
        }
        if (p === Infinity) {
            return max(abs(x), axis);
        }
        if (p === -Infinity) {
            return min(abs(x), axis);
        }
        if (p === 'euclidean' || p === 2) {
            // norm(x, 2) = sum(abs(xi) ^ 2) ^ 1/2
            return sqrt(sum(pow(abs(x), scalar(2, 'int32')), axis));
        }
        throw new Error(`Error in norm: invalid ord value: ${p}`);
    }
    // matrix (assumption axis[0] < axis[1])
    if (Array.isArray(axis) && axis.length === 2) {
        if (p === 1) {
            return max(sum(abs(x), axis[0]), axis[1] - 1);
        }
        if (p === Infinity) {
            return max(sum(abs(x), axis[1]), axis[0]);
        }
        if (p === -Infinity) {
            return min(sum(abs(x), axis[1]), axis[0]);
        }
        if (p === 'fro' || p === 'euclidean') {
            // norm(x) = sqrt(sum(pow(x, 2)))
            return sqrt(sum(square(x), axis));
        }
        throw new Error(`Error in norm: invalid ord value: ${p}`);
    }
    throw new Error(`Error in norm: invalid axis: ${axis}`);
}
export const norm = op({ norm_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibm9ybS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL25vcm0udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFdkMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEtBQUssU0FBUyxNQUFNLGFBQWEsQ0FBQztBQUN6QyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEMsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUNoQyxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzVCLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDaEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUUxQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXFDRztBQUNILFNBQVMsS0FBSyxDQUNWLENBQW9CLEVBQUUsTUFBZ0MsV0FBVyxFQUNqRSxPQUF3QixJQUFJLEVBQUUsUUFBUSxHQUFHLEtBQUs7SUFDaEQsQ0FBQyxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBRXBDLE1BQU0sSUFBSSxHQUFHLFFBQVEsQ0FBQyxD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QUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FDM0M7UUFDRCxJQUFJLENBQUMsS0FBSyxLQUFLLElBQUksQ0FBQyxLQUFLLFdBQVcsRUFBRTtZQUNwQyxpQ0FBaUM7WUFDakMsT0FBTyxJQUFJLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO1NBQ25DO1FBRUQsTUFBTSxJQUFJLEtBQUssQ0FBQyxxQ0FBcUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztLQUMzRDtJQUVELE1BQU0sSUFBSSxLQUFLLENBQUMsZ0NBQWdDLElBQUksRUFBRSxDQUFDLENBQUM7QUFDMUQsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLElBQUksR0FBRyxFQUFFLENBQUMsRUFBQyxLQUFLLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge3BhcnNlQXhpc1BhcmFtfSBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHthYnN9IGZyb20gJy4vYWJzJztcbmltcG9ydCAqIGFzIGF4aXNfdXRpbCBmcm9tICcuL2F4aXNfdXRpbCc7XG5pbXBvcnQge21heH0gZnJvbSAnLi9tYXgnO1xuaW1wb3J0IHttaW59IGZyb20gJy4vbWluJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cG93fSBmcm9tICcuL3Bvdyc7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vcmVzaGFwZSc7XG5pbXBvcnQge3NjYWxhcn0gZnJvbSAnLi9zY2FsYXInO1xuaW1wb3J0IHtzcXJ0fSBmcm9tICcuL3NxcnQnO1xuaW1wb3J0IHtzcXVhcmV9IGZyb20gJy4vc3F1YXJlJztcbmltcG9ydCB7c3VtfSBmcm9tICcuL3N1bSc7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIG5vcm0gb2Ygc2NhbGFyLCB2ZWN0b3JzLCBhbmQgbWF0cmljZXMuXG4gKiBUaGlzIGZ1bmN0aW9uIGNhbiBjb21wdXRlIHNldmVyYWwgZGlmZmVyZW50IHZlY3RvciBub3JtcyAodGhlIDEtbm9ybSwgdGhlXG4gKiBFdWNsaWRlYW4gb3IgMi1ub3JtLCB0aGUgaW5mLW5vcm0sIGFuZCBpbiBnZW5lcmFsIHRoZSBwLW5vcm0gZm9yIHAgPiAwKVxuICogYW5kIG1hdHJpeCBub3JtcyAoRnJvYmVuaXVzLCAxLW5vcm0sIGFuZCBpbmYtbm9ybSkuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMiwgMywgNF0pO1xuICpcbiAqIHgubm9ybSgpLnByaW50KCk7ICAvLyBvciB0Zi5ub3JtKHgpXG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgYXJyYXkuXG4gKiBAcGFyYW0gb3JkIE9wdGlvbmFsLiBPcmRlciBvZiB0aGUgbm9ybS4gU3VwcG9ydGVkIG5vcm0gdHlwZXMgYXJlXG4gKiBmb2xsb3dpbmc6XG4gKlxuICogIHwgb3JkICAgICAgICB8IG5vcm0gZm9yIG1hdHJpY2VzICAgICAgICAgfCBub3JtIGZvciB2ZWN0b3JzXG4gKiAgfC0tLS0tLS0tLS0tLXwtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS18LS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gKiAgfCdldWNsaWRlYW4nIHxGcm9iZW5pdXMgbm9ybSAgICAgICAgICAgICB8Mi1ub3JtXG4gKiAgfCdmcm8nICAgICAgIHxGcm9iZW5pdXMgbm9ybVx0ICAgICAgICAgICB8XG4gKiAgfEluZmluaXR5ICAgIHxtYXgoc3VtKGFicyh4KSwgYXhpcz0xKSkgICB8bWF4KGFicyh4KSlcbiAqICB8LUluZmluaXR5ICAgfG1pbihzdW0oYWJzKHgpLCBheGlzPTEpKSAgIHxtaW4oYWJzKHgpKVxuICogIHwxICAgICAgICAgICB8bWF4KHN1bShhYnMoeCksIGF4aXM9MCkpICAgfHN1bShhYnMoeCkpXG4gKiAgfDIgICAgICAgICAgIHwgICAgICAgICAgICAgICAgICAgICAgICAgICB8c3VtKGFicyh4KV4yKV4xLzIqXG4gKlxuICogQHBhcmFtIGF4aXMgT3B0aW9uYWwuIElmIGF4aXMgaXMgbnVsbCAodGhlIGRlZmF1bHQpLCB0aGUgaW5wdXQgaXNcbiAqIGNvbnNpZGVyZWQgYSB2ZWN0b3IgYW5kIGEgc2luZ2xlIHZlY3RvciBub3JtIGlzIGNvbXB1dGVkIG92ZXIgdGhlIGVudGlyZVxuICogc2V0IG9mIHZhbHVlcyBpbiB0aGUgVGVuc29yLCBpLmUuIG5vcm0oeCwgb3JkKSBpcyBlcXVpdmFsZW50XG4gKiB0byBub3JtKHgucmVzaGFwZShbLTFdKSwgb3JkKS4gSWYgYXhpcyBpcyBhIGludGVnZXIsIHRoZSBpbnB1dFxuICogaXMgY29uc2lkZXJlZCBhIGJhdGNoIG9mIHZlY3RvcnMsIGFuZCBheGlzIGRldGVybWluZXMgdGhlIGF4aXMgaW4geFxuICogb3ZlciB3aGljaCB0byBjb21wdXRlIHZlY3RvciBub3Jtcy4gSWYgYXhpcyBpcyBhIDItdHVwbGUgb2YgaW50ZWdlciBpdCBpc1xuICogY29uc2lkZXJlZCBhIGJhdGNoIG9mIG1hdHJpY2VzIGFuZCBheGlzIGRldGVybWluZXMgdGhlIGF4ZXMgaW4gTkRBcnJheVxuICogb3ZlciB3aGljaCB0byBjb21wdXRlIGEgbWF0cml4IG5vcm0uXG4gKiBAcGFyYW0ga2VlcERpbXMgT3B0aW9uYWwuIElmIHRydWUsIHRoZSBub3JtIGhhdmUgdGhlIHNhbWUgZGltZW5zaW9uYWxpdHlcbiAqIGFzIHRoZSBpbnB1dC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnT3BlcmF0aW9ucycsIHN1YmhlYWRpbmc6ICdNYXRyaWNlcyd9XG4gKi9cbmZ1bmN0aW9uIG5vcm1fKFxuICAgIHg6IFRlbnNvcnxUZW5zb3JMaWtlLCBvcmQ6IG51bWJlcnwnZXVjbGlkZWFuJ3wnZnJvJyA9ICdldWNsaWRlYW4nLFxuICAgIGF4aXM6IG51bWJlcnxudW1iZXJbXSA9IG51bGwsIGtlZXBEaW1zID0gZmFsc2UpOiBUZW5zb3Ige1xuICB4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ25vcm0nKTtcblxuICBjb25zdCBub3JtID0gbm9ybUltcGwoeCwgb3JkLCBheGlzKTtcbiAgbGV0IGtlZXBEaW1zU2hhcGUgPSBub3JtLnNoYXBlO1xuICBpZiAoa2VlcERpbXMpIHtcbiAgICBjb25zdCBheGVzID0gcGFyc2VBeGlzUGFyYW0oYXhpcywgeC5zaGFwZSk7XG4gICAga2VlcERpbXNTaGFwZSA9IGF4aXNfdXRpbC5leHBhbmRTaGFwZVRvS2VlcERpbShub3JtLnNoYXBlLCBheGVzKTtcbiAgfVxuICByZXR1cm4gcmVzaGFwZShub3JtLCBrZWVwRGltc1NoYXBlKTtcbn1cblxuZnVuY3Rpb24gbm9ybUltcGwoXG4gICAgeDogVGVuc29yLCBwOiBudW1iZXJ8c3RyaW5nLCBheGlzOiBudW1iZXJ8bnVtYmVyW10gPSBudWxsKTogVGVuc29yIHtcbiAgaWYgKHgucmFuayA9PT0gMCkge1xuICAgIHJldHVybiBhYnMoeCk7XG4gIH1cblxuICAvLyBjb25zaWRlciB2ZWN0b3Igd2hlbiBubyBheGlzIGlzIHNwZWNpZmllZFxuICBpZiAoeC5yYW5rICE9PSAxICYmIGF4aXMgPT09IG51bGwpIHtcbiAgICByZXR1cm4gbm9ybUltcGwocmVzaGFwZSh4LCBbLTFdKSwgcCwgYXhpcyk7XG4gIH1cblxuICAvLyB2ZWN0b3JcbiAgaWYgKHgucmFuayA9PT0gMSB8fCB0eXBlb2YgYXhpcyA9PT0gJ251bWJlcicgfHxcbiAgICAgIEFycmF5LmlzQXJyYXkoYXhpcykgJiYgYXhpcy5sZW5ndGggPT09IDEpIHtcbiAgICBpZiAocCA9PT0gMSkge1xuICAgICAgcmV0dXJuIHN1bShhYnMoeCksIGF4aXMpO1xuICAgIH1cbiAgICBpZiAocCA9PT0gSW5maW5pdHkpIHtcbiAgICAgIHJldHVybiBtYXgoYWJzKHgpLCBheGlzKTtcbiAgICB9XG4gICAgaWYgKHAgPT09IC1JbmZpbml0eSkge1xuICAgICAgcmV0dXJuIG1pbihhYnMoeCksIGF4aXMpO1xuICAgIH1cbiAgICBpZiAocCA9PT0gJ2V1Y2xpZGVhbicgfHwgcCA9PT0gMikge1xuICAgICAgLy8gbm9ybSh4LCAyKSA9IHN1bShhYnMoeGkpIF4gMikgXiAxLzJcbiAgICAgIHJldHVybiBzcXJ0KHN1bShwb3coYWJzKHgpLCBzY2FsYXIoMiwgJ2ludDMyJykpLCBheGlzKSk7XG4gICAgfVxuXG4gICAgdGhyb3cgbmV3IEVycm9yKGBFcnJvciBpbiBub3JtOiBpbnZhbGlkIG9yZCB2YWx1ZTogJHtwfWApO1xuICB9XG5cbiAgLy8gbWF0cml4IChhc3N1bXB0aW9uIGF4aXNbMF0gPCBheGlzWzFdKVxuICBpZiAoQXJyYXkuaXNBcnJheShheGlzKSAmJiBheGlzLmxlbmd0aCA9PT0gMikge1xuICAgIGlmIChwID09PSAxKSB7XG4gICAgICByZXR1cm4gbWF4KHN1bShhYnMoeCksIGF4aXNbMF0pLCBheGlzWzFdIC0gMSk7XG4gICAgfVxuICAgIGlmIChwID09PSBJbmZpbml0eSkge1xuICAgICAgcmV0dXJuIG1heChzdW0oYWJzKHgpLCBheGlzWzFdKSwgYXhpc1swXSk7XG4gICAgfVxuICAgIGlmIChwID09PSAtSW5maW5pdHkpIHtcbiAgICAgIHJldHVybiBtaW4oc3VtKGFicyh4KSwgYXhpc1sxXSksIGF4aXNbMF0pO1xuICAgIH1cbiAgICBpZiAocCA9PT0gJ2ZybycgfHwgcCA9PT0gJ2V1Y2xpZGVhbicpIHtcbiAgICAgIC8vIG5vcm0oeCkgPSBzcXJ0KHN1bShwb3coeCwgMikpKVxuICAgICAgcmV0dXJuIHNxcnQoc3VtKHNxdWFyZSh4KSwgYXhpcykpO1xuICAgIH1cblxuICAgIHRocm93IG5ldyBFcnJvcihgRXJyb3IgaW4gbm9ybTogaW52YWxpZCBvcmQgdmFsdWU6ICR7cH1gKTtcbiAgfVxuXG4gIHRocm93IG5ldyBFcnJvcihgRXJyb3IgaW4gbm9ybTogaW52YWxpZCBheGlzOiAke2F4aXN9YCk7XG59XG5cbmV4cG9ydCBjb25zdCBub3JtID0gb3Aoe25vcm1ffSk7XG4iXX0=