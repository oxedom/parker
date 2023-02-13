import * as losses from './losses';
import * as metrics from './metrics';
/**
 * Binary accuracy metric function.
 *
 * `yTrue` and `yPred` can have 0-1 values. Example:
 * ```js
 * const x = tf.tensor2d([[1, 1, 1, 1], [0, 0, 0, 0]], [2, 4]);
 * const y = tf.tensor2d([[1, 0, 1, 0], [0, 0, 0, 1]], [2, 4]);
 * const accuracy = tf.metrics.binaryAccuracy(x, y);
 * accuracy.print();
 * ```
 *
 * `yTrue` and `yPred` can also have floating-number values between 0 and 1, in
 * which case the values will be thresholded at 0.5 to yield 0-1 values (i.e.,
 * a value >= 0.5 and <= 1.0 is interpreted as 1.
 * )
 * Example:
 * ```js
 * const x = tf.tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
 * const y = tf.tensor1d([0.2, 0.4, 0.6, 0.8, 0.2, 0.3, 0.4, 0.7]);
 * const accuracy = tf.metrics.binaryAccuracy(x, y);
 * accuracy.print();
 * ```
 *
 * @param yTrue Binary Tensor of truth.
 * @param yPred Binary Tensor of prediction.
 * @return Accuracy Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function binaryAccuracy(yTrue, yPred) {
    return metrics.binaryAccuracy(yTrue, yPred);
}
/**
 * Binary crossentropy metric function.
 *
 * Example:
 * ```js
 * const x = tf.tensor2d([[0], [1], [1], [1]]);
 * const y = tf.tensor2d([[0], [0], [0.5], [1]]);
 * const crossentropy = tf.metrics.binaryCrossentropy(x, y);
 * crossentropy.print();
 * ```
 *
 * @param yTrue Binary Tensor of truth.
 * @param yPred Binary Tensor of prediction, probabilities for the `1` case.
 * @return Accuracy Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function binaryCrossentropy(yTrue, yPred) {
    return metrics.binaryCrossentropy(yTrue, yPred);
}
/**
 * Sparse categorical accuracy metric function.
 *
 * Example:
 * ```js
 *
 * const yTrue = tf.tensor1d([1, 1, 2, 2, 0]);
 * const yPred = tf.tensor2d(
 *      [[0, 1, 0], [1, 0, 0], [0, 0.4, 0.6], [0, 0.6, 0.4], [0.7, 0.3, 0]]);
 * const crossentropy = tf.metrics.sparseCategoricalAccuracy(yTrue, yPred);
 * crossentropy.print();
 * ```
 *
 * @param yTrue True labels: indices.
 * @param yPred Predicted probabilities or logits.
 * @returns Accuracy tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function sparseCategoricalAccuracy(yTrue, yPred) {
    return metrics.sparseCategoricalAccuracy(yTrue, yPred);
}
/**
 * Categorical accuracy metric function.
 *
 * Example:
 * ```js
 * const x = tf.tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]]);
 * const y = tf.tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]]);
 * const accuracy = tf.metrics.categoricalAccuracy(x, y);
 * accuracy.print();
 * ```
 *
 * @param yTrue Binary Tensor of truth: one-hot encoding of categories.
 * @param yPred Binary Tensor of prediction: probabilities or logits for the
 *   same categories as in `yTrue`.
 * @return Accuracy Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function categoricalAccuracy(yTrue, yPred) {
    return metrics.categoricalAccuracy(yTrue, yPred);
}
/**
 * Categorical crossentropy between an output tensor and a target tensor.
 *
 * @param target A tensor of the same shape as `output`.
 * @param output A tensor resulting from a softmax (unless `fromLogits` is
 *  `true`, in which case `output` is expected to be the logits).
 * @param fromLogits Boolean, whether `output` is the result of a softmax, or is
 *   a tensor of logits.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function categoricalCrossentropy(yTrue, yPred) {
    return metrics.categoricalCrossentropy(yTrue, yPred);
}
/**
 * Computes the precision of the predictions with respect to the labels.
 *
 * Example:
 * ```js
 * const x = tf.tensor2d(
 *    [
 *      [0, 0, 0, 1],
 *      [0, 1, 0, 0],
 *      [0, 0, 0, 1],
 *      [1, 0, 0, 0],
 *      [0, 0, 1, 0]
 *    ]
 * );
 *
 * const y = tf.tensor2d(
 *    [
 *      [0, 0, 1, 0],
 *      [0, 1, 0, 0],
 *      [0, 0, 0, 1],
 *      [0, 1, 0, 0],
 *      [0, 1, 0, 0]
 *    ]
 * );
 *
 * const precision = tf.metrics.precision(x, y);
 * precision.print();
 * ```
 *
 * @param yTrue The ground truth values. Expected to be contain only 0-1 values.
 * @param yPred The predicted values. Expected to be contain only 0-1 values.
 * @return Precision Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function precision(yTrue, yPred) {
    return metrics.precision(yTrue, yPred);
}
/**
 * Computes the recall of the predictions with respect to the labels.
 *
 * Example:
 * ```js
 * const x = tf.tensor2d(
 *    [
 *      [0, 0, 0, 1],
 *      [0, 1, 0, 0],
 *      [0, 0, 0, 1],
 *      [1, 0, 0, 0],
 *      [0, 0, 1, 0]
 *    ]
 * );
 *
 * const y = tf.tensor2d(
 *    [
 *      [0, 0, 1, 0],
 *      [0, 1, 0, 0],
 *      [0, 0, 0, 1],
 *      [0, 1, 0, 0],
 *      [0, 1, 0, 0]
 *    ]
 * );
 *
 * const recall = tf.metrics.recall(x, y);
 * recall.print();
 * ```
 *
 * @param yTrue The ground truth values. Expected to be contain only 0-1 values.
 * @param yPred The predicted values. Expected to be contain only 0-1 values.
 * @return Recall Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function recall(yTrue, yPred) {
    return metrics.recall(yTrue, yPred);
}
/**
 * Loss or metric function: Cosine proximity.
 *
 * Mathematically, cosine proximity is defined as:
 *   `-sum(l2Normalize(yTrue) * l2Normalize(yPred))`,
 * wherein `l2Normalize()` normalizes the L2 norm of the input to 1 and `*`
 * represents element-wise multiplication.
 *
 * ```js
 * const yTrue = tf.tensor2d([[1, 0], [1, 0]]);
 * const yPred = tf.tensor2d([[1 / Math.sqrt(2), 1 / Math.sqrt(2)], [0, 1]]);
 * const proximity = tf.metrics.cosineProximity(yTrue, yPred);
 * proximity.print();
 * ```
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Cosine proximity Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function cosineProximity(yTrue, yPred) {
    return losses.cosineProximity(yTrue, yPred);
}
/**
 * Loss or metric function: Mean absolute error.
 *
 * Mathematically, mean absolute error is defined as:
 *   `mean(abs(yPred - yTrue))`,
 * wherein the `mean` is applied over feature dimensions.
 *
 * ```js
 * const yTrue = tf.tensor2d([[0, 1], [0, 0], [2, 3]]);
 * const yPred = tf.tensor2d([[0, 1], [0, 1], [-2, -3]]);
 * const mse = tf.metrics.meanAbsoluteError(yTrue, yPred);
 * mse.print();
 * ```
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Mean absolute error Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function meanAbsoluteError(yTrue, yPred) {
    return losses.meanAbsoluteError(yTrue, yPred);
}
/**
 * Loss or metric function: Mean absolute percentage error.
 *
 * ```js
 * const yTrue = tf.tensor2d([[0, 1], [10, 20]]);
 * const yPred = tf.tensor2d([[0, 1], [11, 24]]);
 * const mse = tf.metrics.meanAbsolutePercentageError(yTrue, yPred);
 * mse.print();
 * ```
 *
 * Aliases: `tf.metrics.MAPE`, `tf.metrics.mape`.
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Mean absolute percentage error Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function meanAbsolutePercentageError(yTrue, yPred) {
    return losses.meanAbsolutePercentageError(yTrue, yPred);
}
export function MAPE(yTrue, yPred) {
    return losses.meanAbsolutePercentageError(yTrue, yPred);
}
export function mape(yTrue, yPred) {
    return losses.meanAbsolutePercentageError(yTrue, yPred);
}
/**
 * Loss or metric function: Mean squared error.
 *
 * ```js
 * const yTrue = tf.tensor2d([[0, 1], [3, 4]]);
 * const yPred = tf.tensor2d([[0, 1], [-3, -4]]);
 * const mse = tf.metrics.meanSquaredError(yTrue, yPred);
 * mse.print();
 * ```
 *
 * Aliases: `tf.metrics.MSE`, `tf.metrics.mse`.
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Mean squared error Tensor.
 *
 * @doc {heading: 'Metrics', namespace: 'metrics'}
 */
export function meanSquaredError(yTrue, yPred) {
    return losses.meanSquaredError(yTrue, yPred);
}
export function MSE(yTrue, yPred) {
    return losses.meanSquaredError(yTrue, yPred);
}
export function mse(yTrue, yPred) {
    return losses.meanSquaredError(yTrue, yPred);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZXhwb3J0c19tZXRyaWNzLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1sYXllcnMvc3JjL2V4cG9ydHNfbWV0cmljcy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFXQSxPQUFPLEtBQUssTUFBTSxNQUFNLFVBQVUsQ0FBQztBQUNuQyxPQUFPLEtBQUssT0FBTyxNQUFNLFdBQVcsQ0FBQztBQUVyQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTRCRztBQUNILE1BQU0sVUFBVSxjQUFjLENBQUMsS0FBYSxFQUFFLEtBQWE7SUFDekQsT0FBTyxPQUFPLENBQUMsY0FBYyxDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQztBQUM5QyxDQUFDO0FBRUQ7Ozs7Ozs7Ozs7Ozs7Ozs7R0FnQkc7QUFDSCxNQUFNLFVBQVUsa0JBQWtCLENBQUMsS0FBYSxFQUFFLEtBQWE7SUFDN0QsT0FBTyxPQUFPLENBQUMsa0JBQWtCLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDO0FBQ2xELENBQUM7QUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBa0JHO0FBQ0gsTUFBTSxVQUFVLHlCQUF5QixDQUNyQyxLQUFhLEVBQUUsS0FBYTtJQUM5QixPQUFPLE9BQU8sQ0FBQyx5QkFBeUIsQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUM7QUFDekQsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7OztHQWlCRztBQUNILE1BQU0sVUFBVSxtQkFBbUIsQ0FBQyxLQUFhLEVBQUUsS0FBYTtJQUM5RCxPQUFPLE9BQU8sQ0FBQyxtQkFBbUIsQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUM7QUFDbkQsQ0FBQztBQUVEOzs7Ozs7Ozs7O0dBVUc7QUFDSCxNQUFNLFVBQVUsdUJBQXVCLENBQUMsS0FBYSxFQUFFLEtBQWE7SUFDbEUsT0FBTyxPQUFPLENBQUMsdUJBQXVCLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDO0FBQ3ZELENBQUM7QUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQWtDRztBQUNILE1BQU0sVUFBVSxTQUFTLENBQUMsS0FBYSxFQUFFLEtBQWE7SUFDcEQsT0FBTyxPQUFPLENBQUMsU0FBUyxDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQztBQUN6QyxDQUFDO0FBRUQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FrQ0c7QUFDSCxNQUFNLFVBQVUsTUFBTSxDQUFDLEtBQWEsRUFBRSxLQUFhO0lBQ2pELE9BQU8sT0FBTyxDQUFDLE1BQU0sQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUM7QUFDdEMsQ0FBQztBQUVEOzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQW9CRztBQUNILE1BQU0sVUFBVSxlQUFlLENBQUMsS0FBYSxFQUFFLEtBQWE7SUFDMUQsT0FBTyxNQUFNLENBQUMsZUFBZSxDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQztBQUM5QyxDQUFDO0FBRUQ7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FtQkc7QUFDSCxNQUFNLFVBQVUsaUJBQWlCLENBQUMsS0FBYSxFQUFFLEtBQWE7SUFDNUQsT0FBTyxNQUFNLENBQUMsaUJBQWlCLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDO0FBQ2hELENBQUM7QUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FpQkc7QUFDSCxNQUFNLFVBQVUsMkJBQTJCLENBQ3ZDLEtBQWEsRUFBRSxLQUFhO0lBQzlCLE9BQU8sTUFBTSxDQUFDLDJCQUEyQixDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQztBQUMxRCxDQUFDO0FBRUQsTUFBTSxVQUFVLElBQUksQ0FBQyxLQUFhLEVBQUUsS0FBYTtJQUMvQyxPQUFPLE1BQU0sQ0FBQywyQkFBMkIsQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUM7QUFDMUQsQ0FBQztBQUVELE1BQU0sVUFBVSxJQUFJLENBQUMsS0FBYSxFQUFFLEtBQWE7SUFDL0MsT0FBTyxNQUFNLENBQUMsMkJBQTJCLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDO0FBQzFELENBQUM7QUFFRDs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FpQkc7QUFDSCxNQUFNLFVBQVUsZ0JBQWdCLENBQUMsS0FBYSxFQUFFLEtBQWE7SUFDM0QsT0FBTyxNQUFNLENBQUMsZ0JBQWdCLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDO0FBQy9DLENBQUM7QUFFRCxNQUFNLFVBQVUsR0FBRyxDQUFDLEtBQWEsRUFBRSxLQUFhO0lBQzlDLE9BQU8sTUFBTSxDQUFDLGdCQUFnQixDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQztBQUMvQyxDQUFDO0FBRUQsTUFBTSxVQUFVLEdBQUcsQ0FBQyxLQUFhLEVBQUUsS0FBYTtJQUM5QyxPQUFPLE1BQU0sQ0FBQyxnQkFBZ0IsQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUM7QUFDL0MsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0ICogYXMgbG9zc2VzIGZyb20gJy4vbG9zc2VzJztcbmltcG9ydCAqIGFzIG1ldHJpY3MgZnJvbSAnLi9tZXRyaWNzJztcblxuLyoqXG4gKiBCaW5hcnkgYWNjdXJhY3kgbWV0cmljIGZ1bmN0aW9uLlxuICpcbiAqIGB5VHJ1ZWAgYW5kIGB5UHJlZGAgY2FuIGhhdmUgMC0xIHZhbHVlcy4gRXhhbXBsZTpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMmQoW1sxLCAxLCAxLCAxXSwgWzAsIDAsIDAsIDBdXSwgWzIsIDRdKTtcbiAqIGNvbnN0IHkgPSB0Zi50ZW5zb3IyZChbWzEsIDAsIDEsIDBdLCBbMCwgMCwgMCwgMV1dLCBbMiwgNF0pO1xuICogY29uc3QgYWNjdXJhY3kgPSB0Zi5tZXRyaWNzLmJpbmFyeUFjY3VyYWN5KHgsIHkpO1xuICogYWNjdXJhY3kucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIGB5VHJ1ZWAgYW5kIGB5UHJlZGAgY2FuIGFsc28gaGF2ZSBmbG9hdGluZy1udW1iZXIgdmFsdWVzIGJldHdlZW4gMCBhbmQgMSwgaW5cbiAqIHdoaWNoIGNhc2UgdGhlIHZhbHVlcyB3aWxsIGJlIHRocmVzaG9sZGVkIGF0IDAuNSB0byB5aWVsZCAwLTEgdmFsdWVzIChpLmUuLFxuICogYSB2YWx1ZSA+PSAwLjUgYW5kIDw9IDEuMCBpcyBpbnRlcnByZXRlZCBhcyAxLlxuICogKVxuICogRXhhbXBsZTpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDEsIDEsIDEsIDAsIDAsIDAsIDBdKTtcbiAqIGNvbnN0IHkgPSB0Zi50ZW5zb3IxZChbMC4yLCAwLjQsIDAuNiwgMC44LCAwLjIsIDAuMywgMC40LCAwLjddKTtcbiAqIGNvbnN0IGFjY3VyYWN5ID0gdGYubWV0cmljcy5iaW5hcnlBY2N1cmFjeSh4LCB5KTtcbiAqIGFjY3VyYWN5LnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geVRydWUgQmluYXJ5IFRlbnNvciBvZiB0cnV0aC5cbiAqIEBwYXJhbSB5UHJlZCBCaW5hcnkgVGVuc29yIG9mIHByZWRpY3Rpb24uXG4gKiBAcmV0dXJuIEFjY3VyYWN5IFRlbnNvci5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTWV0cmljcycsIG5hbWVzcGFjZTogJ21ldHJpY3MnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gYmluYXJ5QWNjdXJhY3koeVRydWU6IFRlbnNvciwgeVByZWQ6IFRlbnNvcik6IFRlbnNvciB7XG4gIHJldHVybiBtZXRyaWNzLmJpbmFyeUFjY3VyYWN5KHlUcnVlLCB5UHJlZCk7XG59XG5cbi8qKlxuICogQmluYXJ5IGNyb3NzZW50cm9weSBtZXRyaWMgZnVuY3Rpb24uXG4gKlxuICogRXhhbXBsZTpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMmQoW1swXSwgWzFdLCBbMV0sIFsxXV0pO1xuICogY29uc3QgeSA9IHRmLnRlbnNvcjJkKFtbMF0sIFswXSwgWzAuNV0sIFsxXV0pO1xuICogY29uc3QgY3Jvc3NlbnRyb3B5ID0gdGYubWV0cmljcy5iaW5hcnlDcm9zc2VudHJvcHkoeCwgeSk7XG4gKiBjcm9zc2VudHJvcHkucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB5VHJ1ZSBCaW5hcnkgVGVuc29yIG9mIHRydXRoLlxuICogQHBhcmFtIHlQcmVkIEJpbmFyeSBUZW5zb3Igb2YgcHJlZGljdGlvbiwgcHJvYmFiaWxpdGllcyBmb3IgdGhlIGAxYCBjYXNlLlxuICogQHJldHVybiBBY2N1cmFjeSBUZW5zb3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ01ldHJpY3MnLCBuYW1lc3BhY2U6ICdtZXRyaWNzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGJpbmFyeUNyb3NzZW50cm9weSh5VHJ1ZTogVGVuc29yLCB5UHJlZDogVGVuc29yKTogVGVuc29yIHtcbiAgcmV0dXJuIG1ldHJpY3MuYmluYXJ5Q3Jvc3NlbnRyb3B5KHlUcnVlLCB5UHJlZCk7XG59XG5cbi8qKlxuICogU3BhcnNlIGNhdGVnb3JpY2FsIGFjY3VyYWN5IG1ldHJpYyBmdW5jdGlvbi5cbiAqXG4gKiBFeGFtcGxlOlxuICogYGBganNcbiAqXG4gKiBjb25zdCB5VHJ1ZSA9IHRmLnRlbnNvcjFkKFsxLCAxLCAyLCAyLCAwXSk7XG4gKiBjb25zdCB5UHJlZCA9IHRmLnRlbnNvcjJkKFxuICogICAgICBbWzAsIDEsIDBdLCBbMSwgMCwgMF0sIFswLCAwLjQsIDAuNl0sIFswLCAwLjYsIDAuNF0sIFswLjcsIDAuMywgMF1dKTtcbiAqIGNvbnN0IGNyb3NzZW50cm9weSA9IHRmLm1ldHJpY3Muc3BhcnNlQ2F0ZWdvcmljYWxBY2N1cmFjeSh5VHJ1ZSwgeVByZWQpO1xuICogY3Jvc3NlbnRyb3B5LnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geVRydWUgVHJ1ZSBsYWJlbHM6IGluZGljZXMuXG4gKiBAcGFyYW0geVByZWQgUHJlZGljdGVkIHByb2JhYmlsaXRpZXMgb3IgbG9naXRzLlxuICogQHJldHVybnMgQWNjdXJhY3kgdGVuc29yLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdNZXRyaWNzJywgbmFtZXNwYWNlOiAnbWV0cmljcyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBzcGFyc2VDYXRlZ29yaWNhbEFjY3VyYWN5KFxuICAgIHlUcnVlOiBUZW5zb3IsIHlQcmVkOiBUZW5zb3IpOiBUZW5zb3Ige1xuICByZXR1cm4gbWV0cmljcy5zcGFyc2VDYXRlZ29yaWNhbEFjY3VyYWN5KHlUcnVlLCB5UHJlZCk7XG59XG5cbi8qKlxuICogQ2F0ZWdvcmljYWwgYWNjdXJhY3kgbWV0cmljIGZ1bmN0aW9uLlxuICpcbiAqIEV4YW1wbGU6XG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFtbMCwgMCwgMCwgMV0sIFswLCAwLCAwLCAxXV0pO1xuICogY29uc3QgeSA9IHRmLnRlbnNvcjJkKFtbMC4xLCAwLjgsIDAuMDUsIDAuMDVdLCBbMC4xLCAwLjA1LCAwLjA1LCAwLjhdXSk7XG4gKiBjb25zdCBhY2N1cmFjeSA9IHRmLm1ldHJpY3MuY2F0ZWdvcmljYWxBY2N1cmFjeSh4LCB5KTtcbiAqIGFjY3VyYWN5LnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geVRydWUgQmluYXJ5IFRlbnNvciBvZiB0cnV0aDogb25lLWhvdCBlbmNvZGluZyBvZiBjYXRlZ29yaWVzLlxuICogQHBhcmFtIHlQcmVkIEJpbmFyeSBUZW5zb3Igb2YgcHJlZGljdGlvbjogcHJvYmFiaWxpdGllcyBvciBsb2dpdHMgZm9yIHRoZVxuICogICBzYW1lIGNhdGVnb3JpZXMgYXMgaW4gYHlUcnVlYC5cbiAqIEByZXR1cm4gQWNjdXJhY3kgVGVuc29yLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdNZXRyaWNzJywgbmFtZXNwYWNlOiAnbWV0cmljcyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjYXRlZ29yaWNhbEFjY3VyYWN5KHlUcnVlOiBUZW5zb3IsIHlQcmVkOiBUZW5zb3IpOiBUZW5zb3Ige1xuICByZXR1cm4gbWV0cmljcy5jYXRlZ29yaWNhbEFjY3VyYWN5KHlUcnVlLCB5UHJlZCk7XG59XG5cbi8qKlxuICogQ2F0ZWdvcmljYWwgY3Jvc3NlbnRyb3B5IGJldHdlZW4gYW4gb3V0cHV0IHRlbnNvciBhbmQgYSB0YXJnZXQgdGVuc29yLlxuICpcbiAqIEBwYXJhbSB0YXJnZXQgQSB0ZW5zb3Igb2YgdGhlIHNhbWUgc2hhcGUgYXMgYG91dHB1dGAuXG4gKiBAcGFyYW0gb3V0cHV0IEEgdGVuc29yIHJlc3VsdGluZyBmcm9tIGEgc29mdG1heCAodW5sZXNzIGBmcm9tTG9naXRzYCBpc1xuICogIGB0cnVlYCwgaW4gd2hpY2ggY2FzZSBgb3V0cHV0YCBpcyBleHBlY3RlZCB0byBiZSB0aGUgbG9naXRzKS5cbiAqIEBwYXJhbSBmcm9tTG9naXRzIEJvb2xlYW4sIHdoZXRoZXIgYG91dHB1dGAgaXMgdGhlIHJlc3VsdCBvZiBhIHNvZnRtYXgsIG9yIGlzXG4gKiAgIGEgdGVuc29yIG9mIGxvZ2l0cy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTWV0cmljcycsIG5hbWVzcGFjZTogJ21ldHJpY3MnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gY2F0ZWdvcmljYWxDcm9zc2VudHJvcHkoeVRydWU6IFRlbnNvciwgeVByZWQ6IFRlbnNvcik6IFRlbnNvciB7XG4gIHJldHVybiBtZXRyaWNzLmNhdGVnb3JpY2FsQ3Jvc3NlbnRyb3B5KHlUcnVlLCB5UHJlZCk7XG59XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIHByZWNpc2lvbiBvZiB0aGUgcHJlZGljdGlvbnMgd2l0aCByZXNwZWN0IHRvIHRoZSBsYWJlbHMuXG4gKlxuICogRXhhbXBsZTpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMmQoXG4gKiAgICBbXG4gKiAgICAgIFswLCAwLCAwLCAxXSxcbiAqICAgICAgWzAsIDEsIDAsIDBdLFxuICogICAgICBbMCwgMCwgMCwgMV0sXG4gKiAgICAgIFsxLCAwLCAwLCAwXSxcbiAqICAgICAgWzAsIDAsIDEsIDBdXG4gKiAgICBdXG4gKiApO1xuICpcbiAqIGNvbnN0IHkgPSB0Zi50ZW5zb3IyZChcbiAqICAgIFtcbiAqICAgICAgWzAsIDAsIDEsIDBdLFxuICogICAgICBbMCwgMSwgMCwgMF0sXG4gKiAgICAgIFswLCAwLCAwLCAxXSxcbiAqICAgICAgWzAsIDEsIDAsIDBdLFxuICogICAgICBbMCwgMSwgMCwgMF1cbiAqICAgIF1cbiAqICk7XG4gKlxuICogY29uc3QgcHJlY2lzaW9uID0gdGYubWV0cmljcy5wcmVjaXNpb24oeCwgeSk7XG4gKiBwcmVjaXNpb24ucHJpbnQoKTtcbiAqIGBgYFxuICpcbiAqIEBwYXJhbSB5VHJ1ZSBUaGUgZ3JvdW5kIHRydXRoIHZhbHVlcy4gRXhwZWN0ZWQgdG8gYmUgY29udGFpbiBvbmx5IDAtMSB2YWx1ZXMuXG4gKiBAcGFyYW0geVByZWQgVGhlIHByZWRpY3RlZCB2YWx1ZXMuIEV4cGVjdGVkIHRvIGJlIGNvbnRhaW4gb25seSAwLTEgdmFsdWVzLlxuICogQHJldHVybiBQcmVjaXNpb24gVGVuc29yLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdNZXRyaWNzJywgbmFtZXNwYWNlOiAnbWV0cmljcyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwcmVjaXNpb24oeVRydWU6IFRlbnNvciwgeVByZWQ6IFRlbnNvcik6IFRlbnNvciB7XG4gIHJldHVybiBtZXRyaWNzLnByZWNpc2lvbih5VHJ1ZSwgeVByZWQpO1xufVxuXG4vKipcbiAqIENvbXB1dGVzIHRoZSByZWNhbGwgb2YgdGhlIHByZWRpY3Rpb25zIHdpdGggcmVzcGVjdCB0byB0aGUgbGFiZWxzLlxuICpcbiAqIEV4YW1wbGU6XG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjJkKFxuICogICAgW1xuICogICAgICBbMCwgMCwgMCwgMV0sXG4gKiAgICAgIFswLCAxLCAwLCAwXSxcbiAqICAgICAgWzAsIDAsIDAsIDFdLFxuICogICAgICBbMSwgMCwgMCwgMF0sXG4gKiAgICAgIFswLCAwLCAxLCAwXVxuICogICAgXVxuICogKTtcbiAqXG4gKiBjb25zdCB5ID0gdGYudGVuc29yMmQoXG4gKiAgICBbXG4gKiAgICAgIFswLCAwLCAxLCAwXSxcbiAqICAgICAgWzAsIDEsIDAsIDBdLFxuICogICAgICBbMCwgMCwgMCwgMV0sXG4gKiAgICAgIFswLCAxLCAwLCAwXSxcbiAqICAgICAgWzAsIDEsIDAsIDBdXG4gKiAgICBdXG4gKiApO1xuICpcbiAqIGNvbnN0IHJlY2FsbCA9IHRmLm1ldHJpY3MucmVjYWxsKHgsIHkpO1xuICogcmVjYWxsLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geVRydWUgVGhlIGdyb3VuZCB0cnV0aCB2YWx1ZXMuIEV4cGVjdGVkIHRvIGJlIGNvbnRhaW4gb25seSAwLTEgdmFsdWVzLlxuICogQHBhcmFtIHlQcmVkIFRoZSBwcmVkaWN0ZWQgdmFsdWVzLiBFeHBlY3RlZCB0byBiZSBjb250YWluIG9ubHkgMC0xIHZhbHVlcy5cbiAqIEByZXR1cm4gUmVjYWxsIFRlbnNvci5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnTWV0cmljcycsIG5hbWVzcGFjZTogJ21ldHJpY3MnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gcmVjYWxsKHlUcnVlOiBUZW5zb3IsIHlQcmVkOiBUZW5zb3IpOiBUZW5zb3Ige1xuICByZXR1cm4gbWV0cmljcy5yZWNhbGwoeVRydWUsIHlQcmVkKTtcbn1cblxuLyoqXG4gKiBMb3NzIG9yIG1ldHJpYyBmdW5jdGlvbjogQ29zaW5lIHByb3hpbWl0eS5cbiAqXG4gKiBNYXRoZW1hdGljYWxseSwgY29zaW5lIHByb3hpbWl0eSBpcyBkZWZpbmVkIGFzOlxuICogICBgLXN1bShsMk5vcm1hbGl6ZSh5VHJ1ZSkgKiBsMk5vcm1hbGl6ZSh5UHJlZCkpYCxcbiAqIHdoZXJlaW4gYGwyTm9ybWFsaXplKClgIG5vcm1hbGl6ZXMgdGhlIEwyIG5vcm0gb2YgdGhlIGlucHV0IHRvIDEgYW5kIGAqYFxuICogcmVwcmVzZW50cyBlbGVtZW50LXdpc2UgbXVsdGlwbGljYXRpb24uXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHlUcnVlID0gdGYudGVuc29yMmQoW1sxLCAwXSwgWzEsIDBdXSk7XG4gKiBjb25zdCB5UHJlZCA9IHRmLnRlbnNvcjJkKFtbMSAvIE1hdGguc3FydCgyKSwgMSAvIE1hdGguc3FydCgyKV0sIFswLCAxXV0pO1xuICogY29uc3QgcHJveGltaXR5ID0gdGYubWV0cmljcy5jb3NpbmVQcm94aW1pdHkoeVRydWUsIHlQcmVkKTtcbiAqIHByb3hpbWl0eS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHlUcnVlIFRydXRoIFRlbnNvci5cbiAqIEBwYXJhbSB5UHJlZCBQcmVkaWN0aW9uIFRlbnNvci5cbiAqIEByZXR1cm4gQ29zaW5lIHByb3hpbWl0eSBUZW5zb3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ01ldHJpY3MnLCBuYW1lc3BhY2U6ICdtZXRyaWNzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvc2luZVByb3hpbWl0eSh5VHJ1ZTogVGVuc29yLCB5UHJlZDogVGVuc29yKTogVGVuc29yIHtcbiAgcmV0dXJuIGxvc3Nlcy5jb3NpbmVQcm94aW1pdHkoeVRydWUsIHlQcmVkKTtcbn1cblxuLyoqXG4gKiBMb3NzIG9yIG1ldHJpYyBmdW5jdGlvbjogTWVhbiBhYnNvbHV0ZSBlcnJvci5cbiAqXG4gKiBNYXRoZW1hdGljYWxseSwgbWVhbiBhYnNvbHV0ZSBlcnJvciBpcyBkZWZpbmVkIGFzOlxuICogICBgbWVhbihhYnMoeVByZWQgLSB5VHJ1ZSkpYCxcbiAqIHdoZXJlaW4gdGhlIGBtZWFuYCBpcyBhcHBsaWVkIG92ZXIgZmVhdHVyZSBkaW1lbnNpb25zLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB5VHJ1ZSA9IHRmLnRlbnNvcjJkKFtbMCwgMV0sIFswLCAwXSwgWzIsIDNdXSk7XG4gKiBjb25zdCB5UHJlZCA9IHRmLnRlbnNvcjJkKFtbMCwgMV0sIFswLCAxXSwgWy0yLCAtM11dKTtcbiAqIGNvbnN0IG1zZSA9IHRmLm1ldHJpY3MubWVhbkFic29sdXRlRXJyb3IoeVRydWUsIHlQcmVkKTtcbiAqIG1zZS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHlUcnVlIFRydXRoIFRlbnNvci5cbiAqIEBwYXJhbSB5UHJlZCBQcmVkaWN0aW9uIFRlbnNvci5cbiAqIEByZXR1cm4gTWVhbiBhYnNvbHV0ZSBlcnJvciBUZW5zb3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ01ldHJpY3MnLCBuYW1lc3BhY2U6ICdtZXRyaWNzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG1lYW5BYnNvbHV0ZUVycm9yKHlUcnVlOiBUZW5zb3IsIHlQcmVkOiBUZW5zb3IpOiBUZW5zb3Ige1xuICByZXR1cm4gbG9zc2VzLm1lYW5BYnNvbHV0ZUVycm9yKHlUcnVlLCB5UHJlZCk7XG59XG5cbi8qKlxuICogTG9zcyBvciBtZXRyaWMgZnVuY3Rpb246IE1lYW4gYWJzb2x1dGUgcGVyY2VudGFnZSBlcnJvci5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeVRydWUgPSB0Zi50ZW5zb3IyZChbWzAsIDFdLCBbMTAsIDIwXV0pO1xuICogY29uc3QgeVByZWQgPSB0Zi50ZW5zb3IyZChbWzAsIDFdLCBbMTEsIDI0XV0pO1xuICogY29uc3QgbXNlID0gdGYubWV0cmljcy5tZWFuQWJzb2x1dGVQZXJjZW50YWdlRXJyb3IoeVRydWUsIHlQcmVkKTtcbiAqIG1zZS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQWxpYXNlczogYHRmLm1ldHJpY3MuTUFQRWAsIGB0Zi5tZXRyaWNzLm1hcGVgLlxuICpcbiAqIEBwYXJhbSB5VHJ1ZSBUcnV0aCBUZW5zb3IuXG4gKiBAcGFyYW0geVByZWQgUHJlZGljdGlvbiBUZW5zb3IuXG4gKiBAcmV0dXJuIE1lYW4gYWJzb2x1dGUgcGVyY2VudGFnZSBlcnJvciBUZW5zb3IuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ01ldHJpY3MnLCBuYW1lc3BhY2U6ICdtZXRyaWNzJ31cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIG1lYW5BYnNvbHV0ZVBlcmNlbnRhZ2VFcnJvcihcbiAgICB5VHJ1ZTogVGVuc29yLCB5UHJlZDogVGVuc29yKTogVGVuc29yIHtcbiAgcmV0dXJuIGxvc3Nlcy5tZWFuQWJzb2x1dGVQZXJjZW50YWdlRXJyb3IoeVRydWUsIHlQcmVkKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIE1BUEUoeVRydWU6IFRlbnNvciwgeVByZWQ6IFRlbnNvcik6IFRlbnNvciB7XG4gIHJldHVybiBsb3NzZXMubWVhbkFic29sdXRlUGVyY2VudGFnZUVycm9yKHlUcnVlLCB5UHJlZCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYXBlKHlUcnVlOiBUZW5zb3IsIHlQcmVkOiBUZW5zb3IpOiBUZW5zb3Ige1xuICByZXR1cm4gbG9zc2VzLm1lYW5BYnNvbHV0ZVBlcmNlbnRhZ2VFcnJvcih5VHJ1ZSwgeVByZWQpO1xufVxuXG4vKipcbiAqIExvc3Mgb3IgbWV0cmljIGZ1bmN0aW9uOiBNZWFuIHNxdWFyZWQgZXJyb3IuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHlUcnVlID0gdGYudGVuc29yMmQoW1swLCAxXSwgWzMsIDRdXSk7XG4gKiBjb25zdCB5UHJlZCA9IHRmLnRlbnNvcjJkKFtbMCwgMV0sIFstMywgLTRdXSk7XG4gKiBjb25zdCBtc2UgPSB0Zi5tZXRyaWNzLm1lYW5TcXVhcmVkRXJyb3IoeVRydWUsIHlQcmVkKTtcbiAqIG1zZS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQWxpYXNlczogYHRmLm1ldHJpY3MuTVNFYCwgYHRmLm1ldHJpY3MubXNlYC5cbiAqXG4gKiBAcGFyYW0geVRydWUgVHJ1dGggVGVuc29yLlxuICogQHBhcmFtIHlQcmVkIFByZWRpY3Rpb24gVGVuc29yLlxuICogQHJldHVybiBNZWFuIHNxdWFyZWQgZXJyb3IgVGVuc29yLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdNZXRyaWNzJywgbmFtZXNwYWNlOiAnbWV0cmljcyd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBtZWFuU3F1YXJlZEVycm9yKHlUcnVlOiBUZW5zb3IsIHlQcmVkOiBUZW5zb3IpOiBUZW5zb3Ige1xuICByZXR1cm4gbG9zc2VzLm1lYW5TcXVhcmVkRXJyb3IoeVRydWUsIHlQcmVkKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIE1TRSh5VHJ1ZTogVGVuc29yLCB5UHJlZDogVGVuc29yKTogVGVuc29yIHtcbiAgcmV0dXJuIGxvc3Nlcy5tZWFuU3F1YXJlZEVycm9yKHlUcnVlLCB5UHJlZCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtc2UoeVRydWU6IFRlbnNvciwgeVByZWQ6IFRlbnNvcik6IFRlbnNvciB7XG4gIHJldHVybiBsb3NzZXMubWVhblNxdWFyZWRFcnJvcih5VHJ1ZSwgeVByZWQpO1xufVxuIl19