/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
// tslint:disable-next-line:max-line-length
import { Constant, GlorotNormal, GlorotUniform, HeNormal, HeUniform, Identity, LeCunNormal, LeCunUniform, Ones, Orthogonal, RandomNormal, RandomUniform, TruncatedNormal, VarianceScaling, Zeros } from './initializers';
/**
 * Initializer that generates tensors initialized to 0.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function zeros() {
    return new Zeros();
}
/**
 * Initializer that generates tensors initialized to 1.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function ones() {
    return new Ones();
}
/**
 * Initializer that generates values initialized to some constant.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function constant(args) {
    return new Constant(args);
}
/**
 * Initializer that generates random values initialized to a uniform
 * distribution.
 *
 * Values will be distributed uniformly between the configured minval and
 * maxval.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function randomUniform(args) {
    return new RandomUniform(args);
}
/**
 * Initializer that generates random values initialized to a normal
 * distribution.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function randomNormal(args) {
    return new RandomNormal(args);
}
/**
 * Initializer that generates random values initialized to a truncated normal.
 * distribution.
 *
 * These values are similar to values from a `RandomNormal` except that values
 * more than two standard deviations from the mean are discarded and re-drawn.
 * This is the recommended initializer for neural network weights and filters.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function truncatedNormal(args) {
    return new TruncatedNormal(args);
}
/**
 * Initializer that generates the identity matrix.
 * Only use for square 2D matrices.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function identity(args) {
    return new Identity(args);
}
/**
 * Initializer capable of adapting its scale to the shape of weights.
 * With distribution=NORMAL, samples are drawn from a truncated normal
 * distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:
 *   - number of input units in the weight tensor, if mode = FAN_IN.
 *   - number of output units, if mode = FAN_OUT.
 *   - average of the numbers of input and output units, if mode = FAN_AVG.
 * With distribution=UNIFORM,
 * samples are drawn from a uniform distribution
 * within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
 *
 * @doc {heading: 'Initializers',namespace: 'initializers'}
 */
export function varianceScaling(config) {
    return new VarianceScaling(config);
}
/**
 * Glorot uniform initializer, also called Xavier uniform initializer.
 * It draws samples from a uniform distribution within [-limit, limit]
 * where `limit` is `sqrt(6 / (fan_in + fan_out))`
 * where `fan_in` is the number of input units in the weight tensor
 * and `fan_out` is the number of output units in the weight tensor
 *
 * Reference:
 *   Glorot & Bengio, AISTATS 2010
 *       http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function glorotUniform(args) {
    return new GlorotUniform(args);
}
/**
 * Glorot normal initializer, also called Xavier normal initializer.
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(2 / (fan_in + fan_out))`
 * where `fan_in` is the number of input units in the weight tensor
 * and `fan_out` is the number of output units in the weight tensor.
 *
 * Reference:
 *   Glorot & Bengio, AISTATS 2010
 *       http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function glorotNormal(args) {
    return new GlorotNormal(args);
}
/**
 * He normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(2 / fanIn)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * Reference:
 *     He et al., http://arxiv.org/abs/1502.01852
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function heNormal(args) {
    return new HeNormal(args);
}
/**
 * He uniform initializer.
 *
 * It draws samples from a uniform distribution within [-limit, limit]
 * where `limit` is `sqrt(6 / fan_in)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * Reference:
 *     He et al., http://arxiv.org/abs/1502.01852
 *
 * @doc {heading: 'Initializers',namespace: 'initializers'}
 */
export function heUniform(args) {
    return new HeUniform(args);
}
/**
 * LeCun normal initializer.
 *
 * It draws samples from a truncated normal distribution centered on 0
 * with `stddev = sqrt(1 / fanIn)`
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * References:
 *   [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
 *   [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function leCunNormal(args) {
    return new LeCunNormal(args);
}
/**
 * LeCun uniform initializer.
 *
 * It draws samples from a uniform distribution in the interval
 * `[-limit, limit]` with `limit = sqrt(3 / fanIn)`,
 * where `fanIn` is the number of input units in the weight tensor.
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function leCunUniform(args) {
    return new LeCunUniform(args);
}
/**
 * Initializer that generates a random orthogonal matrix.
 *
 * Reference:
 * [Saxe et al., http://arxiv.org/abs/1312.6120](http://arxiv.org/abs/1312.6120)
 *
 * @doc {heading: 'Initializers', namespace: 'initializers'}
 */
export function orthogonal(args) {
    return new Orthogonal(args);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZXhwb3J0c19pbml0aWFsaXplcnMuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWxheWVycy9zcmMvZXhwb3J0c19pbml0aWFsaXplcnMudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7O0dBUUc7QUFDSCwyQ0FBMkM7QUFDM0MsT0FBTyxFQUFDLFFBQVEsRUFBZ0IsWUFBWSxFQUFFLGFBQWEsRUFBRSxRQUFRLEVBQUUsU0FBUyxFQUFFLFFBQVEsRUFBNkIsV0FBVyxFQUFFLFlBQVksRUFBRSxJQUFJLEVBQUUsVUFBVSxFQUFrQixZQUFZLEVBQW9CLGFBQWEsRUFBOEMsZUFBZSxFQUF1QixlQUFlLEVBQXVCLEtBQUssRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBRXhYOzs7O0dBSUc7QUFDSCxNQUFNLFVBQVUsS0FBSztJQUNuQixPQUFPLElBQUksS0FBSyxFQUFFLENBQUM7QUFDckIsQ0FBQztBQUVEOzs7O0dBSUc7QUFDSCxNQUFNLFVBQVUsSUFBSTtJQUNsQixPQUFPLElBQUksSUFBSSxFQUFFLENBQUM7QUFDcEIsQ0FBQztBQUVEOzs7O0dBSUc7QUFDSCxNQUFNLFVBQVUsUUFBUSxDQUFDLElBQWtCO0lBQ3pDLE9BQU8sSUFBSSxRQUFRLENBQUMsSUFBSSxDQUFDLENBQUM7QUFDNUIsQ0FBQztBQUVEOzs7Ozs7OztHQVFHO0FBQ0gsTUFBTSxVQUFVLGFBQWEsQ0FBQyxJQUF1QjtJQUNuRCxPQUFPLElBQUksYUFBYSxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQ2pDLENBQUM7QUFFRDs7Ozs7R0FLRztBQUNILE1BQU0sVUFBVSxZQUFZLENBQUMsSUFBc0I7SUFDakQsT0FBTyxJQUFJLFlBQVksQ0FBQyxJQUFJLENBQUMsQ0FBQztBQUNoQyxDQUFDO0FBRUQ7Ozs7Ozs7OztHQVNHO0FBQ0gsTUFBTSxVQUFVLGVBQWUsQ0FBQyxJQUF5QjtJQUN2RCxPQUFPLElBQUksZUFBZSxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQ25DLENBQUM7QUFFRDs7Ozs7R0FLRztBQUNILE1BQU0sVUFBVSxRQUFRLENBQUMsSUFBa0I7SUFDekMsT0FBTyxJQUFJLFFBQVEsQ0FBQyxJQUFJLENBQUMsQ0FBQztBQUM1QixDQUFDO0FBRUQ7Ozs7Ozs7Ozs7OztHQVlHO0FBQ0gsTUFBTSxVQUFVLGVBQWUsQ0FBQyxNQUEyQjtJQUN6RCxPQUFPLElBQUksZUFBZSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0FBQ3JDLENBQUM7QUFFRDs7Ozs7Ozs7Ozs7O0dBWUc7QUFDSCxNQUFNLFVBQVUsYUFBYSxDQUFDLElBQTZCO0lBQ3pELE9BQU8sSUFBSSxhQUFhLENBQUMsSUFBSSxDQUFDLENBQUM7QUFDakMsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7R0FZRztBQUNILE1BQU0sVUFBVSxZQUFZLENBQUMsSUFBNkI7SUFDeEQsT0FBTyxJQUFJLFlBQVksQ0FBQyxJQUFJLENBQUMsQ0FBQztBQUNoQyxDQUFDO0FBRUQ7Ozs7Ozs7Ozs7O0dBV0c7QUFDSCxNQUFNLFVBQVUsUUFBUSxDQUFDLElBQTZCO0lBQ3BELE9BQU8sSUFBSSxRQUFRLENBQUMsSUFBSSxDQUFDLENBQUM7QUFDNUIsQ0FBQztBQUVEOzs7Ozs7Ozs7OztHQVdHO0FBQ0gsTUFBTSxVQUFVLFNBQVMsQ0FBQyxJQUE2QjtJQUNyRCxPQUFPLElBQUksU0FBUyxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQzdCLENBQUM7QUFFRDs7Ozs7Ozs7Ozs7O0dBWUc7QUFDSCxNQUFNLFVBQVUsV0FBVyxDQUFDLElBQTZCO0lBQ3ZELE9BQU8sSUFBSSxXQUFXLENBQUMsSUFBSSxDQUFDLENBQUM7QUFDL0IsQ0FBQztBQUVEOzs7Ozs7OztHQVFHO0FBQ0gsTUFBTSxVQUFVLFlBQVksQ0FBQyxJQUE2QjtJQUN4RCxPQUFPLElBQUksWUFBWSxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQ2hDLENBQUM7QUFFRDs7Ozs7OztHQU9HO0FBQ0gsTUFBTSxVQUFVLFVBQVUsQ0FBQyxJQUFvQjtJQUM3QyxPQUFPLElBQUksVUFBVSxDQUFDLElBQUksQ0FBQyxDQUFDO0FBQzlCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlXG4gKiBsaWNlbnNlIHRoYXQgY2FuIGJlIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgb3IgYXRcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm1heC1saW5lLWxlbmd0aFxuaW1wb3J0IHtDb25zdGFudCwgQ29uc3RhbnRBcmdzLCBHbG9yb3ROb3JtYWwsIEdsb3JvdFVuaWZvcm0sIEhlTm9ybWFsLCBIZVVuaWZvcm0sIElkZW50aXR5LCBJZGVudGl0eUFyZ3MsIEluaXRpYWxpemVyLCBMZUN1bk5vcm1hbCwgTGVDdW5Vbmlmb3JtLCBPbmVzLCBPcnRob2dvbmFsLCBPcnRob2dvbmFsQXJncywgUmFuZG9tTm9ybWFsLCBSYW5kb21Ob3JtYWxBcmdzLCBSYW5kb21Vbmlmb3JtLCBSYW5kb21Vbmlmb3JtQXJncywgU2VlZE9ubHlJbml0aWFsaXplckFyZ3MsIFRydW5jYXRlZE5vcm1hbCwgVHJ1bmNhdGVkTm9ybWFsQXJncywgVmFyaWFuY2VTY2FsaW5nLCBWYXJpYW5jZVNjYWxpbmdBcmdzLCBaZXJvc30gZnJvbSAnLi9pbml0aWFsaXplcnMnO1xuXG4vKipcbiAqIEluaXRpYWxpemVyIHRoYXQgZ2VuZXJhdGVzIHRlbnNvcnMgaW5pdGlhbGl6ZWQgdG8gMC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnSW5pdGlhbGl6ZXJzJywgbmFtZXNwYWNlOiAnaW5pdGlhbGl6ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHplcm9zKCk6IFplcm9zIHtcbiAgcmV0dXJuIG5ldyBaZXJvcygpO1xufVxuXG4vKipcbiAqIEluaXRpYWxpemVyIHRoYXQgZ2VuZXJhdGVzIHRlbnNvcnMgaW5pdGlhbGl6ZWQgdG8gMS5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnSW5pdGlhbGl6ZXJzJywgbmFtZXNwYWNlOiAnaW5pdGlhbGl6ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG9uZXMoKTogSW5pdGlhbGl6ZXIge1xuICByZXR1cm4gbmV3IE9uZXMoKTtcbn1cblxuLyoqXG4gKiBJbml0aWFsaXplciB0aGF0IGdlbmVyYXRlcyB2YWx1ZXMgaW5pdGlhbGl6ZWQgdG8gc29tZSBjb25zdGFudC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnSW5pdGlhbGl6ZXJzJywgbmFtZXNwYWNlOiAnaW5pdGlhbGl6ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbnN0YW50KGFyZ3M6IENvbnN0YW50QXJncyk6IEluaXRpYWxpemVyIHtcbiAgcmV0dXJuIG5ldyBDb25zdGFudChhcmdzKTtcbn1cblxuLyoqXG4gKiBJbml0aWFsaXplciB0aGF0IGdlbmVyYXRlcyByYW5kb20gdmFsdWVzIGluaXRpYWxpemVkIHRvIGEgdW5pZm9ybVxuICogZGlzdHJpYnV0aW9uLlxuICpcbiAqIFZhbHVlcyB3aWxsIGJlIGRpc3RyaWJ1dGVkIHVuaWZvcm1seSBiZXR3ZWVuIHRoZSBjb25maWd1cmVkIG1pbnZhbCBhbmRcbiAqIG1heHZhbC5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnSW5pdGlhbGl6ZXJzJywgbmFtZXNwYWNlOiAnaW5pdGlhbGl6ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHJhbmRvbVVuaWZvcm0oYXJnczogUmFuZG9tVW5pZm9ybUFyZ3MpOiBJbml0aWFsaXplciB7XG4gIHJldHVybiBuZXcgUmFuZG9tVW5pZm9ybShhcmdzKTtcbn1cblxuLyoqXG4gKiBJbml0aWFsaXplciB0aGF0IGdlbmVyYXRlcyByYW5kb20gdmFsdWVzIGluaXRpYWxpemVkIHRvIGEgbm9ybWFsXG4gKiBkaXN0cmlidXRpb24uXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0luaXRpYWxpemVycycsIG5hbWVzcGFjZTogJ2luaXRpYWxpemVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiByYW5kb21Ob3JtYWwoYXJnczogUmFuZG9tTm9ybWFsQXJncyk6IEluaXRpYWxpemVyIHtcbiAgcmV0dXJuIG5ldyBSYW5kb21Ob3JtYWwoYXJncyk7XG59XG5cbi8qKlxuICogSW5pdGlhbGl6ZXIgdGhhdCBnZW5lcmF0ZXMgcmFuZG9tIHZhbHVlcyBpbml0aWFsaXplZCB0byBhIHRydW5jYXRlZCBub3JtYWwuXG4gKiBkaXN0cmlidXRpb24uXG4gKlxuICogVGhlc2UgdmFsdWVzIGFyZSBzaW1pbGFyIHRvIHZhbHVlcyBmcm9tIGEgYFJhbmRvbU5vcm1hbGAgZXhjZXB0IHRoYXQgdmFsdWVzXG4gKiBtb3JlIHRoYW4gdHdvIHN0YW5kYXJkIGRldmlhdGlvbnMgZnJvbSB0aGUgbWVhbiBhcmUgZGlzY2FyZGVkIGFuZCByZS1kcmF3bi5cbiAqIFRoaXMgaXMgdGhlIHJlY29tbWVuZGVkIGluaXRpYWxpemVyIGZvciBuZXVyYWwgbmV0d29yayB3ZWlnaHRzIGFuZCBmaWx0ZXJzLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdJbml0aWFsaXplcnMnLCBuYW1lc3BhY2U6ICdpbml0aWFsaXplcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gdHJ1bmNhdGVkTm9ybWFsKGFyZ3M6IFRydW5jYXRlZE5vcm1hbEFyZ3MpOiBJbml0aWFsaXplciB7XG4gIHJldHVybiBuZXcgVHJ1bmNhdGVkTm9ybWFsKGFyZ3MpO1xufVxuXG4vKipcbiAqIEluaXRpYWxpemVyIHRoYXQgZ2VuZXJhdGVzIHRoZSBpZGVudGl0eSBtYXRyaXguXG4gKiBPbmx5IHVzZSBmb3Igc3F1YXJlIDJEIG1hdHJpY2VzLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdJbml0aWFsaXplcnMnLCBuYW1lc3BhY2U6ICdpbml0aWFsaXplcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gaWRlbnRpdHkoYXJnczogSWRlbnRpdHlBcmdzKTogSW5pdGlhbGl6ZXIge1xuICByZXR1cm4gbmV3IElkZW50aXR5KGFyZ3MpO1xufVxuXG4vKipcbiAqIEluaXRpYWxpemVyIGNhcGFibGUgb2YgYWRhcHRpbmcgaXRzIHNjYWxlIHRvIHRoZSBzaGFwZSBvZiB3ZWlnaHRzLlxuICogV2l0aCBkaXN0cmlidXRpb249Tk9STUFMLCBzYW1wbGVzIGFyZSBkcmF3biBmcm9tIGEgdHJ1bmNhdGVkIG5vcm1hbFxuICogZGlzdHJpYnV0aW9uIGNlbnRlcmVkIG9uIHplcm8sIHdpdGggYHN0ZGRldiA9IHNxcnQoc2NhbGUgLyBuKWAgd2hlcmUgbiBpczpcbiAqICAgLSBudW1iZXIgb2YgaW5wdXQgdW5pdHMgaW4gdGhlIHdlaWdodCB0ZW5zb3IsIGlmIG1vZGUgPSBGQU5fSU4uXG4gKiAgIC0gbnVtYmVyIG9mIG91dHB1dCB1bml0cywgaWYgbW9kZSA9IEZBTl9PVVQuXG4gKiAgIC0gYXZlcmFnZSBvZiB0aGUgbnVtYmVycyBvZiBpbnB1dCBhbmQgb3V0cHV0IHVuaXRzLCBpZiBtb2RlID0gRkFOX0FWRy5cbiAqIFdpdGggZGlzdHJpYnV0aW9uPVVOSUZPUk0sXG4gKiBzYW1wbGVzIGFyZSBkcmF3biBmcm9tIGEgdW5pZm9ybSBkaXN0cmlidXRpb25cbiAqIHdpdGhpbiBbLWxpbWl0LCBsaW1pdF0sIHdpdGggYGxpbWl0ID0gc3FydCgzICogc2NhbGUgLyBuKWAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0luaXRpYWxpemVycycsbmFtZXNwYWNlOiAnaW5pdGlhbGl6ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHZhcmlhbmNlU2NhbGluZyhjb25maWc6IFZhcmlhbmNlU2NhbGluZ0FyZ3MpOiBJbml0aWFsaXplciB7XG4gIHJldHVybiBuZXcgVmFyaWFuY2VTY2FsaW5nKGNvbmZpZyk7XG59XG5cbi8qKlxuICogR2xvcm90IHVuaWZvcm0gaW5pdGlhbGl6ZXIsIGFsc28gY2FsbGVkIFhhdmllciB1bmlmb3JtIGluaXRpYWxpemVyLlxuICogSXQgZHJhd3Mgc2FtcGxlcyBmcm9tIGEgdW5pZm9ybSBkaXN0cmlidXRpb24gd2l0aGluIFstbGltaXQsIGxpbWl0XVxuICogd2hlcmUgYGxpbWl0YCBpcyBgc3FydCg2IC8gKGZhbl9pbiArIGZhbl9vdXQpKWBcbiAqIHdoZXJlIGBmYW5faW5gIGlzIHRoZSBudW1iZXIgb2YgaW5wdXQgdW5pdHMgaW4gdGhlIHdlaWdodCB0ZW5zb3JcbiAqIGFuZCBgZmFuX291dGAgaXMgdGhlIG51bWJlciBvZiBvdXRwdXQgdW5pdHMgaW4gdGhlIHdlaWdodCB0ZW5zb3JcbiAqXG4gKiBSZWZlcmVuY2U6XG4gKiAgIEdsb3JvdCAmIEJlbmdpbywgQUlTVEFUUyAyMDEwXG4gKiAgICAgICBodHRwOi8vam1sci5vcmcvcHJvY2VlZGluZ3MvcGFwZXJzL3Y5L2dsb3JvdDEwYS9nbG9yb3QxMGEucGRmLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdJbml0aWFsaXplcnMnLCBuYW1lc3BhY2U6ICdpbml0aWFsaXplcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gZ2xvcm90VW5pZm9ybShhcmdzOiBTZWVkT25seUluaXRpYWxpemVyQXJncyk6IEluaXRpYWxpemVyIHtcbiAgcmV0dXJuIG5ldyBHbG9yb3RVbmlmb3JtKGFyZ3MpO1xufVxuXG4vKipcbiAqIEdsb3JvdCBub3JtYWwgaW5pdGlhbGl6ZXIsIGFsc28gY2FsbGVkIFhhdmllciBub3JtYWwgaW5pdGlhbGl6ZXIuXG4gKiBJdCBkcmF3cyBzYW1wbGVzIGZyb20gYSB0cnVuY2F0ZWQgbm9ybWFsIGRpc3RyaWJ1dGlvbiBjZW50ZXJlZCBvbiAwXG4gKiB3aXRoIGBzdGRkZXYgPSBzcXJ0KDIgLyAoZmFuX2luICsgZmFuX291dCkpYFxuICogd2hlcmUgYGZhbl9pbmAgaXMgdGhlIG51bWJlciBvZiBpbnB1dCB1bml0cyBpbiB0aGUgd2VpZ2h0IHRlbnNvclxuICogYW5kIGBmYW5fb3V0YCBpcyB0aGUgbnVtYmVyIG9mIG91dHB1dCB1bml0cyBpbiB0aGUgd2VpZ2h0IHRlbnNvci5cbiAqXG4gKiBSZWZlcmVuY2U6XG4gKiAgIEdsb3JvdCAmIEJlbmdpbywgQUlTVEFUUyAyMDEwXG4gKiAgICAgICBodHRwOi8vam1sci5vcmcvcHJvY2VlZGluZ3MvcGFwZXJzL3Y5L2dsb3JvdDEwYS9nbG9yb3QxMGEucGRmXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0luaXRpYWxpemVycycsIG5hbWVzcGFjZTogJ2luaXRpYWxpemVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBnbG9yb3ROb3JtYWwoYXJnczogU2VlZE9ubHlJbml0aWFsaXplckFyZ3MpOiBJbml0aWFsaXplciB7XG4gIHJldHVybiBuZXcgR2xvcm90Tm9ybWFsKGFyZ3MpO1xufVxuXG4vKipcbiAqIEhlIG5vcm1hbCBpbml0aWFsaXplci5cbiAqXG4gKiBJdCBkcmF3cyBzYW1wbGVzIGZyb20gYSB0cnVuY2F0ZWQgbm9ybWFsIGRpc3RyaWJ1dGlvbiBjZW50ZXJlZCBvbiAwXG4gKiB3aXRoIGBzdGRkZXYgPSBzcXJ0KDIgLyBmYW5JbilgXG4gKiB3aGVyZSBgZmFuSW5gIGlzIHRoZSBudW1iZXIgb2YgaW5wdXQgdW5pdHMgaW4gdGhlIHdlaWdodCB0ZW5zb3IuXG4gKlxuICogUmVmZXJlbmNlOlxuICogICAgIEhlIGV0IGFsLiwgaHR0cDovL2FyeGl2Lm9yZy9hYnMvMTUwMi4wMTg1MlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdJbml0aWFsaXplcnMnLCBuYW1lc3BhY2U6ICdpbml0aWFsaXplcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gaGVOb3JtYWwoYXJnczogU2VlZE9ubHlJbml0aWFsaXplckFyZ3MpOiBJbml0aWFsaXplciB7XG4gIHJldHVybiBuZXcgSGVOb3JtYWwoYXJncyk7XG59XG5cbi8qKlxuICogSGUgdW5pZm9ybSBpbml0aWFsaXplci5cbiAqXG4gKiBJdCBkcmF3cyBzYW1wbGVzIGZyb20gYSB1bmlmb3JtIGRpc3RyaWJ1dGlvbiB3aXRoaW4gWy1saW1pdCwgbGltaXRdXG4gKiB3aGVyZSBgbGltaXRgIGlzIGBzcXJ0KDYgLyBmYW5faW4pYFxuICogd2hlcmUgYGZhbkluYCBpcyB0aGUgbnVtYmVyIG9mIGlucHV0IHVuaXRzIGluIHRoZSB3ZWlnaHQgdGVuc29yLlxuICpcbiAqIFJlZmVyZW5jZTpcbiAqICAgICBIZSBldCBhbC4sIGh0dHA6Ly9hcnhpdi5vcmcvYWJzLzE1MDIuMDE4NTJcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnSW5pdGlhbGl6ZXJzJyxuYW1lc3BhY2U6ICdpbml0aWFsaXplcnMnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gaGVVbmlmb3JtKGFyZ3M6IFNlZWRPbmx5SW5pdGlhbGl6ZXJBcmdzKTogSW5pdGlhbGl6ZXIge1xuICByZXR1cm4gbmV3IEhlVW5pZm9ybShhcmdzKTtcbn1cblxuLyoqXG4gKiBMZUN1biBub3JtYWwgaW5pdGlhbGl6ZXIuXG4gKlxuICogSXQgZHJhd3Mgc2FtcGxlcyBmcm9tIGEgdHJ1bmNhdGVkIG5vcm1hbCBkaXN0cmlidXRpb24gY2VudGVyZWQgb24gMFxuICogd2l0aCBgc3RkZGV2ID0gc3FydCgxIC8gZmFuSW4pYFxuICogd2hlcmUgYGZhbkluYCBpcyB0aGUgbnVtYmVyIG9mIGlucHV0IHVuaXRzIGluIHRoZSB3ZWlnaHQgdGVuc29yLlxuICpcbiAqIFJlZmVyZW5jZXM6XG4gKiAgIFtTZWxmLU5vcm1hbGl6aW5nIE5ldXJhbCBOZXR3b3Jrc10oaHR0cHM6Ly9hcnhpdi5vcmcvYWJzLzE3MDYuMDI1MTUpXG4gKiAgIFtFZmZpY2llbnQgQmFja3Byb3BdKGh0dHA6Ly95YW5uLmxlY3VuLmNvbS9leGRiL3B1Ymxpcy9wZGYvbGVjdW4tOThiLnBkZilcbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnSW5pdGlhbGl6ZXJzJywgbmFtZXNwYWNlOiAnaW5pdGlhbGl6ZXJzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGxlQ3VuTm9ybWFsKGFyZ3M6IFNlZWRPbmx5SW5pdGlhbGl6ZXJBcmdzKTogSW5pdGlhbGl6ZXIge1xuICByZXR1cm4gbmV3IExlQ3VuTm9ybWFsKGFyZ3MpO1xufVxuXG4vKipcbiAqIExlQ3VuIHVuaWZvcm0gaW5pdGlhbGl6ZXIuXG4gKlxuICogSXQgZHJhd3Mgc2FtcGxlcyBmcm9tIGEgdW5pZm9ybSBkaXN0cmlidXRpb24gaW4gdGhlIGludGVydmFsXG4gKiBgWy1saW1pdCwgbGltaXRdYCB3aXRoIGBsaW1pdCA9IHNxcnQoMyAvIGZhbkluKWAsXG4gKiB3aGVyZSBgZmFuSW5gIGlzIHRoZSBudW1iZXIgb2YgaW5wdXQgdW5pdHMgaW4gdGhlIHdlaWdodCB0ZW5zb3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0luaXRpYWxpemVycycsIG5hbWVzcGFjZTogJ2luaXRpYWxpemVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBsZUN1blVuaWZvcm0oYXJnczogU2VlZE9ubHlJbml0aWFsaXplckFyZ3MpOiBJbml0aWFsaXplciB7XG4gIHJldHVybiBuZXcgTGVDdW5Vbmlmb3JtKGFyZ3MpO1xufVxuXG4vKipcbiAqIEluaXRpYWxpemVyIHRoYXQgZ2VuZXJhdGVzIGEgcmFuZG9tIG9ydGhvZ29uYWwgbWF0cml4LlxuICpcbiAqIFJlZmVyZW5jZTpcbiAqIFtTYXhlIGV0IGFsLiwgaHR0cDovL2FyeGl2Lm9yZy9hYnMvMTMxMi42MTIwXShodHRwOi8vYXJ4aXYub3JnL2Ficy8xMzEyLjYxMjApXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0luaXRpYWxpemVycycsIG5hbWVzcGFjZTogJ2luaXRpYWxpemVycyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBvcnRob2dvbmFsKGFyZ3M6IE9ydGhvZ29uYWxBcmdzKTogSW5pdGlhbGl6ZXIge1xuICByZXR1cm4gbmV3IE9ydGhvZ29uYWwoYXJncyk7XG59XG4iXX0=