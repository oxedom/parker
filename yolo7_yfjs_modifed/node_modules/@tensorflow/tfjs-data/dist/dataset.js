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
 *
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs-core';
import * as seedrandom from 'seedrandom';
import { iteratorFromConcatenated, iteratorFromFunction, iteratorFromItems, iteratorFromZipped, ZipMismatchMode } from './iterators/lazy_iterator';
import { canTensorify, deepMapAndAwaitAll, isIterable } from './util/deep_map';
// TODO(soergel): consider vectorized operations within the pipeline.
/**
 * Represents a potentially large list of independent data elements (typically
 * 'samples' or 'examples').
 *
 * A 'data example' may be a primitive, an array, a map from string keys to
 * values, or any nested structure of these.
 *
 * A `Dataset` represents an ordered collection of elements, together with a
 * chain of transformations to be performed on those elements. Each
 * transformation is a method of `Dataset` that returns another `Dataset`, so
 * these may be chained, e.g.
 * `const processedDataset = rawDataset.filter(...).map(...).batch(...)`.
 *
 * Data loading and transformation is done in a lazy, streaming fashion.  The
 * dataset may be iterated over multiple times; each iteration starts the data
 * loading anew and recapitulates the transformations.
 *
 * A `Dataset` is typically processed as a stream of unbatched examples --i.e.,
 * its transformations are applied one example at a time. Batching produces a
 * new `Dataset` where each element is a batch. Batching should usually come
 * last in a pipeline, because data transformations are easier to express on a
 * per-example basis than on a per-batch basis.
 *
 * The following code examples are calling `await dataset.forEachAsync(...)` to
 * iterate once over the entire dataset in order to print out the data.
 *
 * @doc {heading: 'Data', subheading: 'Classes', namespace: 'data'}
 */
export class Dataset {
    constructor() {
        this.size = null;
    }
    // TODO(soergel): Make Datasets report whether repeated iterator() calls
    // produce the same result (e.g., reading from a file) or different results
    // (e.g., from the webcam).  Currently we don't make this distinction but it
    // could be important for the user to know.
    // abstract isDeterministic(): boolean;
    /**
     * Groups elements into batches.
     *
     * It is assumed that each of the incoming dataset elements has the same
     * structure-- i.e. the same set of keys at each location in an object
     * hierarchy.  For each key, the resulting `Dataset` provides a batched
     * element collecting all of the incoming values for that key.
     *
     *  * Incoming primitives are grouped into a 1-D Tensor.
     *  * Incoming Tensors are grouped into a new Tensor where the 0'th axis is
     *    the batch dimension.
     *  * Incoming arrays are converted to Tensor and then batched.
     *  * A nested array is interpreted as an n-D Tensor, so the batched result
     *    has n+1 dimensions.
     *  * An array that cannot be converted to Tensor produces an error.
     *
     * If an array should not be batched as a unit, it should first be converted
     * to an object with integer keys.
     *
     * Here are a few examples:
     *
     * Batch a dataset of numbers:
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6, 7, 8]).batch(4);
     * await a.forEachAsync(e => e.print());
     * ```
     *
     * Batch a dataset of arrays:
     * ```js
     * const b = tf.data.array([[1], [2], [3], [4], [5], [6], [7], [8]]).batch(4);
     * await b.forEachAsync(e => e.print());
     * ```
     *
     * Batch a dataset of objects:
     * ```js
     * const c = tf.data.array([{a: 1, b: 11}, {a: 2, b: 12}, {a: 3, b: 13},
     *   {a: 4, b: 14}, {a: 5, b: 15}, {a: 6, b: 16}, {a: 7, b: 17},
     *   {a: 8, b: 18}]).batch(4);
     * await c.forEachAsync(e => {
     *   console.log('{');
     *   for(var key in e) {
     *     console.log(key+':');
     *     e[key].print();
     *   }
     *   console.log('}');
     * })
     * ```
     *
     * @param batchSize The number of elements desired per batch.
     * @param smallLastBatch Whether to emit the final batch when it has fewer
     *   than batchSize elements. Default true.
     * @returns A `Dataset`, from which a stream of batches can be obtained.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    batch(batchSize, smallLastBatch = true) {
        const base = this;
        tf.util.assert(batchSize > 0, () => `batchSize needs to be positive, but it is
      ${batchSize}`);
        let size;
        if (this.size === Infinity || this.size == null) {
            // If the size of this dataset is infinity or null, the new size keeps the
            // same.
            size = this.size;
        }
        else if (smallLastBatch) {
            // If the size of this dataset is known and include small last batch, the
            // new size is full batch count plus last batch.
            size = Math.ceil(this.size / batchSize);
        }
        else {
            // If the size of this dataset is known and not include small last batch,
            // the new size is full batch count.
            size = Math.floor(this.size / batchSize);
        }
        return datasetFromIteratorFn(async () => {
            return (await base.iterator())
                .columnMajorBatch(batchSize, smallLastBatch, deepBatchConcat);
        }, size);
    }
    /**
     * Concatenates this `Dataset` with another.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]);
     * const b = tf.data.array([4, 5, 6]);
     * const c = a.concatenate(b);
     * await c.forEachAsync(e => console.log(e));
     * ```
     *
     * @param dataset A `Dataset` to be concatenated onto this one.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    concatenate(dataset) {
        const base = this;
        let size;
        if (this.size === Infinity || dataset.size === Infinity) {
            // If the size of any of these two dataset is infinity, new size is
            // infinity.
            size = Infinity;
        }
        else if (this.size != null && dataset.size != null) {
            // If the size of both datasets are known and not infinity, new size is
            // sum the size of these two datasets.
            size = this.size + dataset.size;
        }
        else {
            // If neither of these two datasets has infinite size and any of these two
            // datasets' size is null, the new size is null.
            size = null;
        }
        return datasetFromIteratorFn(async () => (await base.iterator()).concatenate(await dataset.iterator()), size);
    }
    /**
     * Filters this dataset according to `predicate`.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
     *   .filter(x => x%2 === 0);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param predicate A function mapping a dataset element to a boolean or a
     * `Promise` for one.
     *
     * @returns A `Dataset` of elements for which the predicate was true.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    filter(predicate) {
        const base = this;
        let size;
        if (this.size === Infinity) {
            // If the size of this dataset is infinity, new size is infinity
            size = Infinity;
        }
        else {
            // If this dataset has limited elements, new size is null because it might
            // exhausted randomly.
            size = null;
        }
        return datasetFromIteratorFn(async () => {
            return (await base.iterator()).filter(x => tf.tidy(() => predicate(x)));
        }, size);
    }
    /**
     * Apply a function to every element of the dataset.
     *
     * After the function is applied to a dataset element, any Tensors contained
     * within that element are disposed.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param f A function to apply to each dataset element.
     * @returns A `Promise` that resolves after all elements have been processed.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    async forEachAsync(f) {
        return (await this.iterator()).forEachAsync(f);
    }
    /**
     * Maps this dataset through a 1-to-1 transform.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]).map(x => x*x);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param transform A function mapping a dataset element to a transformed
     *   dataset element.
     *
     * @returns A `Dataset` of transformed elements.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    map(transform) {
        const base = this;
        return datasetFromIteratorFn(async () => {
            return (await base.iterator()).map(x => tf.tidy(() => transform(x)));
        }, this.size);
    }
    /**
     * Maps this dataset through an async 1-to-1 transform.
     *
     * ```js
     * const a =
     *  tf.data.array([1, 2, 3]).mapAsync(x => new Promise(function(resolve){
     *    setTimeout(() => {
     *      resolve(x * x);
     *    }, Math.random()*1000 + 500);
     *  }));
     * console.log(await a.toArray());
     * ```
     *
     * @param transform A function mapping a dataset element to a `Promise` for a
     *   transformed dataset element.  This transform is responsible for disposing
     *   any intermediate `Tensor`s, i.e. by wrapping its computation in
     *   `tf.tidy()`; that cannot be automated here (as it is in the synchronous
     *   `map()` case).
     *
     * @returns A `Dataset` of transformed elements.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    mapAsync(transform) {
        const base = this;
        return datasetFromIteratorFn(async () => {
            return (await base.iterator()).mapAsync(transform);
        }, this.size);
    }
    /**
     *  Creates a `Dataset` that prefetches elements from this dataset.
     *
     * @param bufferSize: An integer specifying the number of elements to be
     *   prefetched.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    prefetch(bufferSize) {
        if (bufferSize == null) {
            throw new RangeError('`Dataset.prefetch()` requires bufferSize to be specified.');
        }
        const base = this;
        return datasetFromIteratorFn(async () => (await base.iterator()).prefetch(bufferSize), this.size);
    }
    /**
     * Repeats this dataset `count` times.
     *
     * NOTE: If this dataset is a function of global state (e.g. a random number
     * generator), then different repetitions may produce different elements.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3]).repeat(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param count: (Optional) An integer, representing the number of times
     *   the dataset should be repeated. The default behavior (if `count` is
     *   `undefined` or negative) is for the dataset be repeated indefinitely.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    repeat(count) {
        const base = this;
        let size;
        if (this.size != null && count > 0) {
            // If this dataset has size and count is positive, new size is current
            // size multiply count. This also covers the case that current size is
            // infinity.
            size = this.size * count;
        }
        else if (count === 0) {
            // If count is 0, new size is 0.
            size = 0;
        }
        else if (this.size != null && (count === undefined || count < 0)) {
            // If this dataset has size and count is undefined or negative, the
            // dataset will be repeated indefinitely and new size is infinity.
            size = Infinity;
        }
        else {
            // If the size of this dataset is null, the new dataset's size is null.
            size = null;
        }
        return datasetFromIteratorFn(async () => {
            const iteratorIterator = iteratorFromFunction(async () => ({ value: await base.iterator(), done: false }));
            return iteratorFromConcatenated(iteratorIterator.take(count));
        }, size);
    }
    /**
     * Creates a `Dataset` that skips `count` initial elements from this dataset.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]).skip(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param count: The number of elements of this dataset that should be skipped
     *   to form the new dataset.  If `count` is greater than the size of this
     *   dataset, the new dataset will contain no elements.  If `count`
     *   is `undefined` or negative, skips the entire dataset.
     *
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    skip(count) {
        const base = this;
        let size;
        if (this.size != null && count >= 0 && this.size >= count) {
            // If the size of this dataset is greater than count, the new dataset's
            // size is current size minus skipped size.This also covers the case that
            // current size is infinity.
            size = this.size - count;
        }
        else if (this.size != null &&
            (this.size < count || count === undefined || count < 0)) {
            // If the size of this dataset is smaller than count, or count is
            // undefined or negative, skips the entire dataset and the new size is 0.
            size = 0;
        }
        else {
            // If the size of this dataset is null, the new dataset's size is null.
            size = null;
        }
        return datasetFromIteratorFn(async () => (await base.iterator()).skip(count), size);
    }
    /**
     * Pseudorandomly shuffles the elements of this dataset. This is done in a
     * streaming manner, by sampling from a given number of prefetched elements.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]).shuffle(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param bufferSize: An integer specifying the number of elements from this
     *   dataset from which the new dataset will sample.
     * @param seed: (Optional) An integer specifying the random seed that will
     *   be used to create the distribution.
     * @param reshuffleEachIteration: (Optional) A boolean, which if true
     *   indicates that the dataset should be pseudorandomly reshuffled each time
     *   it is iterated over. If false, elements will be returned in the same
     *   shuffled order on each iteration. (Defaults to `true`.)
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    shuffle(bufferSize, seed, reshuffleEachIteration = true) {
        if (bufferSize == null || bufferSize < 0) {
            if (this.size == null) {
                throw new RangeError('`Dataset.shuffle()` requires bufferSize to be specified.');
            }
            else {
                throw new RangeError('`Dataset.shuffle()` requires bufferSize to be specified.  ' +
                    'If your data fits in main memory (for regular JS objects), ' +
                    'and/or GPU memory (for `tf.Tensor`s), consider setting ' +
                    `bufferSize to the dataset size (${this.size} elements)`);
            }
        }
        const base = this;
        const random = seedrandom.alea(seed || tf.util.now().toString());
        return datasetFromIteratorFn(async () => {
            let seed2 = random.int32();
            if (reshuffleEachIteration) {
                seed2 += random.int32();
            }
            return (await base.iterator()).shuffle(bufferSize, seed2.toString());
        }, this.size);
    }
    /**
     * Creates a `Dataset` with at most `count` initial elements from this
     * dataset.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]).take(3);
     * await a.forEachAsync(e => console.log(e));
     * ```
     *
     * @param count: The number of elements of this dataset that should be taken
     *   to form the new dataset.  If `count` is `undefined` or negative, or if
     *   `count` is greater than the size of this dataset, the new dataset will
     *   contain all elements of this dataset.
     * @returns A `Dataset`.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    take(count) {
        const base = this;
        let size;
        if (this.size != null && this.size > count) {
            // If the size of this dataset is greater than count, the new dataset's
            // size is count.
            size = count;
        }
        else if (this.size != null && this.size <= count) {
            // If the size of this dataset is equal or smaller than count, the new
            // dataset's size is the size of this dataset.
            size = this.size;
        }
        else {
            // If the size of this dataset is null, the new dataset's size is null.
            size = null;
        }
        return datasetFromIteratorFn(async () => (await base.iterator()).take(count), size);
    }
    /**
     * Collect all elements of this dataset into an array.
     *
     * Obviously this will succeed only for small datasets that fit in memory.
     * Useful for testing and generally should be avoided if possible.
     *
     * ```js
     * const a = tf.data.array([1, 2, 3, 4, 5, 6]);
     * console.log(await a.toArray());
     * ```
     *
     * @returns A Promise for an array of elements, which will resolve
     *   when a new stream has been obtained and fully consumed.
     *
     * @doc {heading: 'Data', subheading: 'Classes'}
     */
    async toArray() {
        if (this.size === Infinity) {
            throw new Error('Can not convert infinite data stream to array.');
        }
        return (await this.iterator()).toArray();
    }
    /**
     * Collect all elements of this dataset into an array with prefetching 100
     * elements. This is useful for testing, because the prefetch changes the
     * order in which the Promises are resolved along the processing pipeline.
     * This may help expose bugs where results are dependent on the order of
     * Promise resolution rather than on the logical order of the stream (i.e.,
     * due to hidden mutable state).
     *
     * @returns A Promise for an array of elements, which will resolve
     *   when a new stream has been obtained and fully consumed.
     */
    async toArrayForTest() {
        if (this.size === Infinity) {
            throw new Error('Can not convert infinite data stream to array.');
        }
        return (await this.iterator()).toArrayForTest();
    }
}
// TODO(soergel): deep sharded shuffle, where supported
Dataset.MAX_BUFFER_SIZE = 10000;
/**
 * Create a `Dataset` defined by a provided iterator() function.
 *
 * ```js
 * let i = -1;
 * const func = () =>
 *    ++i < 5 ? {value: i, done: false} : {value: null, done: true};
 * const iter = tf.data.iteratorFromFunction(func);
 * const ds = tf.data.datasetFromIteratorFn(iter);
 * await ds.forEachAsync(e => console.log(e));
 * ```
 */
export function datasetFromIteratorFn(iteratorFn, size = null) {
    return new class extends Dataset {
        constructor() {
            super(...arguments);
            this.size = size;
        }
        /*
         * Provide a new stream of elements.  Note this will also start new streams
         * from any underlying `Dataset`s.
         */
        async iterator() {
            return iteratorFn();
        }
    }();
}
/**
 * Create a `Dataset` from an array of elements.
 *
 * Create a Dataset from an array of objects:
 * ```js
 * const a = tf.data.array([{'item': 1}, {'item': 2}, {'item': 3}]);
 * await a.forEachAsync(e => console.log(e));
 * ```
 *
 * Create a Dataset from an array of numbers:
 * ```js
 * const a = tf.data.array([4, 5, 6]);
 * await a.forEachAsync(e => console.log(e));
 * ```
 * @param items An array of elements that will be parsed as items in a dataset.
 *
 * @doc {heading: 'Data', subheading: 'Creation', namespace: 'data'}
 */
export function array(items) {
    return datasetFromIteratorFn(async () => iteratorFromItems(items), items.length);
}
/**
 * Create a `Dataset` by zipping together an array, dict, or nested
 * structure of `Dataset`s (and perhaps additional constants).
 * The underlying datasets must provide elements in a consistent order such that
 * they correspond.
 *
 * The number of elements in the resulting dataset is the same as the size of
 * the smallest dataset in datasets.
 *
 * The nested structure of the `datasets` argument determines the
 * structure of elements in the resulting iterator.
 *
 * Note this means that, given an array of two datasets that produce dict
 * elements, the result is a dataset that produces elements that are arrays
 * of two dicts:
 *
 * Zip an array of datasets:
 * ```js
 * console.log('Zip two datasets of objects:');
 * const ds1 = tf.data.array([{a: 1}, {a: 2}, {a: 3}]);
 * const ds2 = tf.data.array([{b: 4}, {b: 5}, {b: 6}]);
 * const ds3 = tf.data.zip([ds1, ds2]);
 * await ds3.forEachAsync(e => console.log(JSON.stringify(e)));
 *
 * // If the goal is to merge the dicts in order to produce elements like
 * // {a: ..., b: ...}, this requires a second step such as:
 * console.log('Merge the objects:');
 * const ds4 = ds3.map(x => {return {a: x[0].a, b: x[1].b}});
 * await ds4.forEachAsync(e => console.log(e));
 * ```
 *
 * Zip a dict of datasets:
 * ```js
 * const a = tf.data.array([{a: 1}, {a: 2}, {a: 3}]);
 * const b = tf.data.array([{b: 4}, {b: 5}, {b: 6}]);
 * const c = tf.data.zip({c: a, d: b});
 * await c.forEachAsync(e => console.log(JSON.stringify(e)));
 * ```
 *
 * @doc {heading: 'Data', subheading: 'Operations', namespace: 'data'}
 */
export function zip(datasets) {
    // manually type-check the argument for JS users
    if (!isIterable(datasets)) {
        throw new Error('The argument to zip() must be an object or array.');
    }
    let size;
    if (Array.isArray(datasets)) {
        for (let i = 0; i < datasets.length; i++) {
            size = size == null ? datasets[i].size :
                Math.min(size, datasets[i].size);
        }
    }
    else if (datasets instanceof Object) {
        for (const ds in datasets) {
            size = size == null ? datasets[ds].size :
                Math.min(size, datasets[ds].size);
        }
    }
    return datasetFromIteratorFn(async () => {
        const streams = await deepMapAndAwaitAll(datasets, d => {
            if (d instanceof Dataset) {
                return { value: d.iterator(), recurse: false };
            }
            else if (isIterable(d)) {
                return { value: null, recurse: true };
            }
            else {
                throw new Error('Leaves of the structure passed to zip() must be Datasets, ' +
                    'not primitives.');
            }
        });
        return iteratorFromZipped(streams, ZipMismatchMode.SHORTEST);
    }, size);
}
/**
 * A zip function for use with deepZip, passed via the columnMajorBatch call.
 *
 * Accepts an array of identically-structured nested elements and either batches
 * them (if they are primitives, numeric arrays, or Tensors) or requests
 * recursion (if not).
 */
// tslint:disable-next-line:no-any
function deepBatchConcat(rows) {
    if (rows === null) {
        return null;
    }
    // use the first item to decide whether to recurse or batch here.
    const exampleRow = rows[0];
    if (canTensorify(exampleRow)) {
        // rows is an array of primitives, Tensors, or arrays.  Batch them.
        const value = batchConcat(rows);
        return { value, recurse: false };
    }
    // the example row is an object, so recurse into it.
    return { value: null, recurse: true };
}
/**
 * Assembles a list of same-shaped numbers, number arrays, or Tensors
 * into a single new Tensor where axis 0 is the batch dimension.
 */
function batchConcat(arrays) {
    if (arrays.length === 0) {
        // We can't return an empty Tensor because we don't know the element shape.
        throw new Error('Can\'t make a batch of zero elements.');
    }
    if (arrays[0] instanceof tf.Tensor) {
        // Input is an array of Tensors
        return tf.stack(arrays);
    }
    else {
        // Input is a possibly-nested array of numbers.
        return tf.tensor(arrays);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGF0YXNldC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtZGF0YS9zcmMvZGF0YXNldC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7OztHQWdCRztBQUVILE9BQU8sS0FBSyxFQUFFLE1BQU0sdUJBQXVCLENBQUM7QUFFNUMsT0FBTyxLQUFLLFVBQVUsTUFBTSxZQUFZLENBQUM7QUFFekMsT0FBTyxFQUFDLHdCQUF3QixFQUFFLG9CQUFvQixFQUFFLGlCQUFpQixFQUFFLGtCQUFrQixFQUFnQixlQUFlLEVBQUMsTUFBTSwyQkFBMkIsQ0FBQztBQUUvSixPQUFPLEVBQUMsWUFBWSxFQUFFLGtCQUFrQixFQUFpQixVQUFVLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQU81RixxRUFBcUU7QUFFckU7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQTJCRztBQUNILE1BQU0sT0FBZ0IsT0FBTztJQUE3QjtRQVdXLFNBQUksR0FBVyxJQUFJLENBQUM7SUEyYy9CLENBQUM7SUF6Y0Msd0VBQXdFO0lBQ3hFLDJFQUEyRTtJQUMzRSw0RUFBNEU7SUFDNUUsMkNBQTJDO0lBQzNDLHVDQUF1QztJQUV2Qzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O09Bc0RHO0lBQ0gsS0FBSyxDQUFDLFNBQWlCLEVBQUUsY0FBYyxHQUFHLElBQUk7UUFDNUMsTUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDO1FBQ2xCLEVBQUUsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUNWLFNBQVMsR0FBRyxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUM7UUFDckIsU0FBUyxFQUFFLENBQUMsQ0FBQztRQUNqQixJQUFJLElBQUksQ0FBQztRQUNULElBQUksSUFBSSxDQUFDLElBQUksS0FBSyxRQUFRLElBQUksSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLEVBQUU7WUFDL0MsMEVBQTBF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K0M7UUFDL0MsT0FBTyxFQUFFLENBQUMsTUFBTSxDQUFDLE1BQW9CLENBQUMsQ0FBQztLQUN4QztBQUNILENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgKiBhcyB0ZiBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0IHtUZW5zb3JDb250YWluZXIsIFRlbnNvckxpa2V9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5pbXBvcnQgKiBhcyBzZWVkcmFuZG9tIGZyb20gJ3NlZWRyYW5kb20nO1xuXG5pbXBvcnQge2l0ZXJhdG9yRnJvbUNvbmNhdGVuYXRlZCwgaXRlcmF0b3JGcm9tRnVuY3Rpb24sIGl0ZXJhdG9yRnJvbUl0ZW1zLCBpdGVyYXRvckZyb21aaXBwZWQsIExhenlJdGVyYXRvciwgWmlwTWlzbWF0Y2hNb2RlfSBmcm9tICcuL2l0ZXJhdG9ycy9sYXp5X2l0ZXJhdG9yJztcbmltcG9ydCB7Q29udGFpbmVyfSBmcm9tICcuL3R5cGVzJztcbmltcG9ydCB7Y2FuVGVuc29yaWZ5LCBkZWVwTWFwQW5kQXdhaXRBbGwsIERlZXBNYXBSZXN1bHQsIGlzSXRlcmFibGV9IGZyb20gJy4vdXRpbC9kZWVwX21hcCc7XG5cbi8qKlxuICogQSBuZXN0ZWQgc3RydWN0dXJlIG9mIERhdGFzZXRzLCB1c2VkIGFzIHRoZSBpbnB1dCB0byB6aXAoKS5cbiAqL1xuZXhwb3J0IHR5cGUgRGF0YXNldENvbnRhaW5lciA9IENvbnRhaW5lcjxEYXRhc2V0PFRlbnNvckNvbnRhaW5lcj4+O1xuXG4vLyBUT0RPKHNvZXJnZWwpOiBjb25zaWRlciB2ZWN0b3JpemVkIG9wZXJhdGlvbnMgd2l0aGluIHRoZSBwaXBlbGluZS5cblxuLyoqXG4gKiBSZXByZXNlbnRzIGEgcG90ZW50aWFsbHkgbGFyZ2UgbGlzdCBvZiBpbmRlcGVuZGVudCBkYXRhIGVsZW1lbnRzICh0eXBpY2FsbHlcbiAqICdzYW1wbGVzJyBvciAnZXhhbXBsZXMnKS5cbiAqXG4gKiBBICdkYXRhIGV4YW1wbGUnIG1heSBiZSBhIHByaW1pdGl2ZSwgYW4gYXJyYXksIGEgbWFwIGZyb20gc3RyaW5nIGtleXMgdG9cbiAqIHZhbHVlcywgb3IgYW55IG5lc3RlZCBzdHJ1Y3R1cmUgb2YgdGhlc2UuXG4gKlxuICogQSBgRGF0YXNldGAgcmVwcmVzZW50cyBhbiBvcmRlcmVkIGNvbGxlY3Rpb24gb2YgZWxlbWVudHMsIHRvZ2V0aGVyIHdpdGggYVxuICogY2hhaW4gb2YgdHJhbnNmb3JtYXRpb25zIHRvIGJlIHBlcmZvcm1lZCBvbiB0aG9zZSBlbGVtZW50cy4gRWFjaFxuICogdHJhbnNmb3JtYXRpb24gaXMgYSBtZXRob2Qgb2YgYERhdGFzZXRgIHRoYXQgcmV0dXJucyBhbm90aGVyIGBEYXRhc2V0YCwgc29cbiAqIHRoZXNlIG1heSBiZSBjaGFpbmVkLCBlLmcuXG4gKiBgY29uc3QgcHJvY2Vzc2VkRGF0YXNldCA9IHJhd0RhdGFzZXQuZmlsdGVyKC4uLikubWFwKC4uLikuYmF0Y2goLi4uKWAuXG4gKlxuICogRGF0YSBsb2FkaW5nIGFuZCB0cmFuc2Zvcm1hdGlvbiBpcyBkb25lIGluIGEgbGF6eSwgc3RyZWFtaW5nIGZhc2hpb24uICBUaGVcbiAqIGRhdGFzZXQgbWF5IGJlIGl0ZXJhdGVkIG92ZXIgbXVsdGlwbGUgdGltZXM7IGVhY2ggaXRlcmF0aW9uIHN0YXJ0cyB0aGUgZGF0YVxuICogbG9hZGluZyBhbmV3IGFuZCByZWNhcGl0dWxhdGVzIHRoZSB0cmFuc2Zvcm1hdGlvbnMuXG4gKlxuICogQSBgRGF0YXNldGAgaXMgdHlwaWNhbGx5IHByb2Nlc3NlZCBhcyBhIHN0cmVhbSBvZiB1bmJhdGNoZWQgZXhhbXBsZXMgLS1pLmUuLFxuICogaXRzIHRyYW5zZm9ybWF0aW9ucyBhcmUgYXBwbGllZCBvbmUgZXhhbXBsZSBhdCBhIHRpbWUuIEJhdGNoaW5nIHByb2R1Y2VzIGFcbiAqIG5ldyBgRGF0YXNldGAgd2hlcmUgZWFjaCBlbGVtZW50IGlzIGEgYmF0Y2guIEJhdGNoaW5nIHNob3VsZCB1c3VhbGx5IGNvbWVcbiAqIGxhc3QgaW4gYSBwaXBlbGluZSwgYmVjYXVzZSBkYXRhIHRyYW5zZm9ybWF0aW9ucyBhcmUgZWFzaWVyIHRvIGV4cHJlc3Mgb24gYVxuICogcGVyLWV4YW1wbGUgYmFzaXMgdGhhbiBvbiBhIHBlci1iYXRjaCBiYXNpcy5cbiAqXG4gKiBUaGUgZm9sbG93aW5nIGNvZGUgZXhhbXBsZXMgYXJlIGNhbGxpbmcgYGF3YWl0IGRhdGFzZXQuZm9yRWFjaEFzeW5jKC4uLilgIHRvXG4gKiBpdGVyYXRlIG9uY2Ugb3ZlciB0aGUgZW50aXJlIGRhdGFzZXQgaW4gb3JkZXIgdG8gcHJpbnQgb3V0IHRoZSBkYXRhLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdEYXRhJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnLCBuYW1lc3BhY2U6ICdkYXRhJ31cbiAqL1xuZXhwb3J0IGFic3RyYWN0IGNsYXNzIERhdGFzZXQ8VCBleHRlbmRzIHRmLlRlbnNvckNvbnRhaW5lcj4ge1xuICAvKlxuICAgKiBQcm92aWRlIGEgbmV3IHN0cmVhbSBvZiBlbGVtZW50cy4gIE5vdGUgdGhpcyB3aWxsIGFsc28gc3RhcnQgbmV3IHN0cmVhbXNcbiAgICogZnJvbSBhbnkgdW5kZXJseWluZyBgRGF0YXNldGBzLlxuICAgKlxuICAgKiBDQVVUSU9OOiBBbnkgVGVuc29ycyBjb250YWluZWQgd2l0aGluIHRoZSBlbGVtZW50cyByZXR1cm5lZCBmcm9tXG4gICAqIHRoaXMgc3RyZWFtICptdXN0KiBiZSBtYW51YWxseSBkaXNwb3NlZCB0byBhdm9pZCBhIEdQVSBtZW1vcnkgbGVhay5cbiAgICogVGhlIHRmLnRpZHkoKSBhcHByb2FjaCBjYW5ub3QgYmUgdXNlZCBpbiBhbiBhc3luY2hyb25vdXMgY29udGV4dC5cbiAgICovXG4gIGFic3RyYWN0IGFzeW5jIGl0ZXJhdG9yKCk6IFByb21pc2U8TGF6eUl0ZXJhdG9yPFQ+PjtcblxuICByZWFkb25seSBzaXplOiBudW1iZXIgPSBudWxsO1xuXG4gIC8vIFRPRE8oc29lcmdlbCk6IE1ha2UgRGF0YXNldHMgcmVwb3J0IHdoZXRoZXIgcmVwZWF0ZWQgaXRlcmF0b3IoKSBjYWxsc1xuICAvLyBwcm9kdWNlIHRoZSBzYW1lIHJlc3VsdCAoZS5nLiwgcmVhZGluZyBmcm9tIGEgZmlsZSkgb3IgZGlmZmVyZW50IHJlc3VsdHNcbiAgLy8gKGUuZy4sIGZyb20gdGhlIHdlYmNhbSkuICBDdXJyZW50bHkgd2UgZG9uJ3QgbWFrZSB0aGlzIGRpc3RpbmN0aW9uIGJ1dCBpdFxuICAvLyBjb3VsZCBiZSBpbXBvcnRhbnQgZm9yIHRoZSB1c2VyIHRvIGtub3cuXG4gIC8vIGFic3RyYWN0IGlzRGV0ZXJtaW5pc3RpYygpOiBib29sZWFuO1xuXG4gIC8qKlxuICAgKiBHcm91cHMgZWxlbWVudHMgaW50byBiYXRjaGVzLlxuICAgKlxuICAgKiBJdCBpcyBhc3N1bWVkIHRoYXQgZWFjaCBvZiB0aGUgaW5jb21pbmcgZGF0YXNldCBlbGVtZW50cyBoYXMgdGhlIHNhbWVcbiAgICogc3RydWN0dXJlLS0gaS5lLiB0aGUgc2FtZSBzZXQgb2Yga2V5cyBhdCBlYWNoIGxvY2F0aW9uIGluIGFuIG9iamVjdFxuICAgKiBoaWVyYXJjaHkuICBGb3IgZWFjaCBrZXksIHRoZSByZXN1bHRpbmcgYERhdGFzZXRgIHByb3ZpZGVzIGEgYmF0Y2hlZFxuICAgKiBlbGVtZW50IGNvbGxlY3RpbmcgYWxsIG9mIHRoZSBpbmNvbWluZyB2YWx1ZXMgZm9yIHRoYXQga2V5LlxuICAgKlxuICAgKiAgKiBJbmNvbWluZyBwcmltaXRpdmVzIGFyZSBncm91cGVkIGludG8gYSAxLUQgVGVuc29yLlxuICAgKiAgKiBJbmNvbWluZyBUZW5zb3JzIGFyZSBncm91cGVkIGludG8gYSBuZXcgVGVuc29yIHdoZXJlIHRoZSAwJ3RoIGF4aXMgaXNcbiAgICogICAgdGhlIGJhdGNoIGRpbWVuc2lvbi5cbiAgICogICogSW5jb21pbmcgYXJyYXlzIGFyZSBjb252ZXJ0ZWQgdG8gVGVuc29yIGFuZCB0aGVuIGJhdGNoZWQuXG4gICAqICAqIEEgbmVzdGVkIGFycmF5IGlzIGludGVycHJldGVkIGFzIGFuIG4tRCBUZW5zb3IsIHNvIHRoZSBiYXRjaGVkIHJlc3VsdFxuICAgKiAgICBoYXMgbisxIGRpbWVuc2lvbnMuXG4gICAqICAqIEFuIGFycmF5IHRoYXQgY2Fubm90IGJlIGNvbnZlcnRlZCB0byBUZW5zb3IgcHJvZHVjZXMgYW4gZXJyb3IuXG4gICAqXG4gICAqIElmIGFuIGFycmF5IHNob3VsZCBub3QgYmUgYmF0Y2hlZCBhcyBhIHVuaXQsIGl0IHNob3VsZCBmaXJzdCBiZSBjb252ZXJ0ZWRcbiAgICogdG8gYW4gb2JqZWN0IHdpdGggaW50ZWdlciBrZXlzLlxuICAgKlxuICAgKiBIZXJlIGFyZSBhIGZldyBleGFtcGxlczpcbiAgICpcbiAgICogQmF0Y2ggYSBkYXRhc2V0IG9mIG51bWJlcnM6XG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGEgPSB0Zi5kYXRhLmFycmF5KFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4XSkuYmF0Y2goNCk7XG4gICAqIGF3YWl0IGEuZm9yRWFjaEFzeW5jKGUgPT4gZS5wcmludCgpKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEJhdGNoIGEgZGF0YXNldCBvZiBhcnJheXM6XG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGIgPSB0Zi5kYXRhLmFycmF5KFtbMV0sIFsyXSwgWzNdLCBbNF0sIFs1XSwgWzZdLCBbN10sIFs4XV0pLmJhdGNoKDQpO1xuICAgKiBhd2FpdCBiLmZvckVhY2hBc3luYyhlID0+IGUucHJpbnQoKSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBCYXRjaCBhIGRhdGFzZXQgb2Ygb2JqZWN0czpcbiAgICogYGBganNcbiAgICogY29uc3QgYyA9IHRmLmRhdGEuYXJyYXkoW3thOiAxLCBiOiAxMX0sIHthOiAyLCBiOiAxMn0sIHthOiAzLCBiOiAxM30sXG4gICAqICAge2E6IDQsIGI6IDE0fSwge2E6IDUsIGI6IDE1fSwge2E6IDYsIGI6IDE2fSwge2E6IDcsIGI6IDE3fSxcbiAgICogICB7YTogOCwgYjogMTh9XSkuYmF0Y2goNCk7XG4gICAqIGF3YWl0IGMuZm9yRWFjaEFzeW5jKGUgPT4ge1xuICAgKiAgIGNvbnNvbGUubG9nKCd7Jyk7XG4gICAqICAgZm9yKHZhciBrZXkgaW4gZSkge1xuICAgKiAgICAgY29uc29sZS5sb2coa2V5Kyc6Jyk7XG4gICAqICAgICBlW2tleV0ucHJpbnQoKTtcbiAgICogICB9XG4gICAqICAgY29uc29sZS5sb2coJ30nKTtcbiAgICogfSlcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSBiYXRjaFNpemUgVGhlIG51bWJlciBvZiBlbGVtZW50cyBkZXNpcmVkIHBlciBiYXRjaC5cbiAgICogQHBhcmFtIHNtYWxsTGFzdEJhdGNoIFdoZXRoZXIgdG8gZW1pdCB0aGUgZmluYWwgYmF0Y2ggd2hlbiBpdCBoYXMgZmV3ZXJcbiAgICogICB0aGFuIGJhdGNoU2l6ZSBlbGVtZW50cy4gRGVmYXVsdCB0cnVlLlxuICAgKiBAcmV0dXJucyBBIGBEYXRhc2V0YCwgZnJvbSB3aGljaCBhIHN0cmVhbSBvZiBiYXRjaGVzIGNhbiBiZSBvYnRhaW5lZC5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ0RhdGEnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBiYXRjaChiYXRjaFNpemU6IG51bWJlciwgc21hbGxMYXN0QmF0Y2ggPSB0cnVlKTogRGF0YXNldDx0Zi5UZW5zb3JDb250YWluZXI+IHtcbiAgICBjb25zdCBiYXNlID0gdGhpcztcbiAgICB0Zi51dGlsLmFzc2VydChcbiAgICAgICAgYmF0Y2hTaXplID4gMCwgKCkgPT4gYGJhdGNoU2l6ZSBuZWVkcyB0byBiZSBwb3NpdGl2ZSwgYnV0IGl0IGlzXG4gICAgICAke2JhdGNoU2l6ZX1gKTtcbiAgICBsZXQgc2l6ZTtcbiAgICBpZiAodGhpcy5zaXplID09PSBJbmZpbml0eSB8fCB0aGlzLnNpemUgPT0gbnVsbCkge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0IGlzIGluZmluaXR5IG9yIG51bGwsIHRoZSBuZXcgc2l6ZSBrZWVwcyB0aGVcbiAgICAgIC8vIHNhbWUuXG4gICAgICBzaXplID0gdGhpcy5zaXplO1xuICAgIH0gZWxzZSBpZiAoc21hbGxMYXN0QmF0Y2gpIHtcbiAgICAgIC8vIElmIHRoZSBzaXplIG9mIHRoaXMgZGF0YXNldCBpcyBrbm93biBhbmQgaW5jbHVkZSBzbWFsbCBsYXN0IGJhdGNoLCB0aGVcbiAgICAgIC8vIG5ldyBzaXplIGlzIGZ1bGwgYmF0Y2ggY291bnQgcGx1cyBsYXN0IGJhdGNoLlxuICAgICAgc2l6ZSA9IE1hdGguY2VpbCh0aGlzLnNpemUgLyBiYXRjaFNpemUpO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBJZiB0aGUgc2l6ZSBvZiB0aGlzIGRhdGFzZXQgaXMga25vd24gYW5kIG5vdCBpbmNsdWRlIHNtYWxsIGxhc3QgYmF0Y2gsXG4gICAgICAvLyB0aGUgbmV3IHNpemUgaXMgZnVsbCBiYXRjaCBjb3VudC5cbiAgICAgIHNpemUgPSBNYXRoLmZsb29yKHRoaXMuc2l6ZSAvIGJhdGNoU2l6ZSk7XG4gICAgfVxuICAgIHJldHVybiBkYXRhc2V0RnJvbUl0ZXJhdG9yRm4oYXN5bmMgKCkgPT4ge1xuICAgICAgcmV0dXJuIChhd2FpdCBiYXNlLml0ZXJhdG9yKCkpXG4gICAgICAgICAgLmNvbHVtbk1ham9yQmF0Y2goYmF0Y2hTaXplLCBzbWFsbExhc3RCYXRjaCwgZGVlcEJhdGNoQ29uY2F0KTtcbiAgICB9LCBzaXplKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb25jYXRlbmF0ZXMgdGhpcyBgRGF0YXNldGAgd2l0aCBhbm90aGVyLlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBhID0gdGYuZGF0YS5hcnJheShbMSwgMiwgM10pO1xuICAgKiBjb25zdCBiID0gdGYuZGF0YS5hcnJheShbNCwgNSwgNl0pO1xuICAgKiBjb25zdCBjID0gYS5jb25jYXRlbmF0ZShiKTtcbiAgICogYXdhaXQgYy5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhlKSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0gZGF0YXNldCBBIGBEYXRhc2V0YCB0byBiZSBjb25jYXRlbmF0ZWQgb250byB0aGlzIG9uZS5cbiAgICogQHJldHVybnMgQSBgRGF0YXNldGAuXG4gICAqXG4gICAqIEBkb2Mge2hlYWRpbmc6ICdEYXRhJywgc3ViaGVhZGluZzogJ0NsYXNzZXMnfVxuICAgKi9cbiAgY29uY2F0ZW5hdGUoZGF0YXNldDogRGF0YXNldDxUPik6IERhdGFzZXQ8VD4ge1xuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIGxldCBzaXplO1xuICAgIGlmICh0aGlzLnNpemUgPT09IEluZmluaXR5IHx8IGRhdGFzZXQuc2l6ZSA9PT0gSW5maW5pdHkpIHtcbiAgICAgIC8vIElmIHRoZSBzaXplIG9mIGFueSBvZiB0aGVzZSB0d28gZGF0YXNldCBpcyBpbmZpbml0eSwgbmV3IHNpemUgaXNcbiAgICAgIC8vIGluZmluaXR5LlxuICAgICAgc2l6ZSA9IEluZmluaXR5O1xuICAgIH0gZWxzZSBpZiAodGhpcy5zaXplICE9IG51bGwgJiYgZGF0YXNldC5zaXplICE9IG51bGwpIHtcbiAgICAgIC8vIElmIHRoZSBzaXplIG9mIGJvdGggZGF0YXNldHMgYXJlIGtub3duIGFuZCBub3QgaW5maW5pdHksIG5ldyBzaXplIGlzXG4gICAgICAvLyBzdW0gdGhlIHNpemUgb2YgdGhlc2UgdHdvIGRhdGFzZXRzLlxuICAgICAgc2l6ZSA9IHRoaXMuc2l6ZSArIGRhdGFzZXQuc2l6ZTtcbiAgICB9IGVsc2Uge1xuICAgICAgLy8gSWYgbmVpdGhlciBvZiB0aGVzZSB0d28gZGF0YXNldHMgaGFzIGluZmluaXRlIHNpemUgYW5kIGFueSBvZiB0aGVzZSB0d29cbiAgICAgIC8vIGRhdGFzZXRzJyBzaXplIGlzIG51bGwsIHRoZSBuZXcgc2l6ZSBpcyBudWxsLlxuICAgICAgc2l6ZSA9IG51bGw7XG4gICAgfVxuICAgIHJldHVybiBkYXRhc2V0RnJvbUl0ZXJhdG9yRm4oXG4gICAgICAgIGFzeW5jICgpID0+XG4gICAgICAgICAgICAoYXdhaXQgYmFzZS5pdGVyYXRvcigpKS5jb25jYXRlbmF0ZShhd2FpdCBkYXRhc2V0Lml0ZXJhdG9yKCkpLFxuICAgICAgICBzaXplKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBGaWx0ZXJzIHRoaXMgZGF0YXNldCBhY2NvcmRpbmcgdG8gYHByZWRpY2F0ZWAuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGEgPSB0Zi5kYXRhLmFycmF5KFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5LCAxMF0pXG4gICAqICAgLmZpbHRlcih4ID0+IHglMiA9PT0gMCk7XG4gICAqIGF3YWl0IGEuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIHByZWRpY2F0ZSBBIGZ1bmN0aW9uIG1hcHBpbmcgYSBkYXRhc2V0IGVsZW1lbnQgdG8gYSBib29sZWFuIG9yIGFcbiAgICogYFByb21pc2VgIGZvciBvbmUuXG4gICAqXG4gICAqIEByZXR1cm5zIEEgYERhdGFzZXRgIG9mIGVsZW1lbnRzIGZvciB3aGljaCB0aGUgcHJlZGljYXRlIHdhcyB0cnVlLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGZpbHRlcihwcmVkaWNhdGU6ICh2YWx1ZTogVCkgPT4gYm9vbGVhbik6IERhdGFzZXQ8VD4ge1xuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIGxldCBzaXplO1xuICAgIGlmICh0aGlzLnNpemUgPT09IEluZmluaXR5KSB7XG4gICAgICAvLyBJZiB0aGUgc2l6ZSBvZiB0aGlzIGRhdGFzZXQgaXMgaW5maW5pdHksIG5ldyBzaXplIGlzIGluZmluaXR5XG4gICAgICBzaXplID0gSW5maW5pdHk7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIElmIHRoaXMgZGF0YXNldCBoYXMgbGltaXRlZCBlbGVtZW50cywgbmV3IHNpemUgaXMgbnVsbCBiZWNhdXNlIGl0IG1pZ2h0XG4gICAgICAvLyBleGhhdXN0ZWQgcmFuZG9tbHkuXG4gICAgICBzaXplID0gbnVsbDtcbiAgICB9XG4gICAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbihhc3luYyAoKSA9PiB7XG4gICAgICByZXR1cm4gKGF3YWl0IGJhc2UuaXRlcmF0b3IoKSkuZmlsdGVyKHggPT4gdGYudGlkeSgoKSA9PiBwcmVkaWNhdGUoeCkpKTtcbiAgICB9LCBzaXplKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBBcHBseSBhIGZ1bmN0aW9uIHRvIGV2ZXJ5IGVsZW1lbnQgb2YgdGhlIGRhdGFzZXQuXG4gICAqXG4gICAqIEFmdGVyIHRoZSBmdW5jdGlvbiBpcyBhcHBsaWVkIHRvIGEgZGF0YXNldCBlbGVtZW50LCBhbnkgVGVuc29ycyBjb250YWluZWRcbiAgICogd2l0aGluIHRoYXQgZWxlbWVudCBhcmUgZGlzcG9zZWQuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGEgPSB0Zi5kYXRhLmFycmF5KFsxLCAyLCAzXSk7XG4gICAqIGF3YWl0IGEuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIGYgQSBmdW5jdGlvbiB0byBhcHBseSB0byBlYWNoIGRhdGFzZXQgZWxlbWVudC5cbiAgICogQHJldHVybnMgQSBgUHJvbWlzZWAgdGhhdCByZXNvbHZlcyBhZnRlciBhbGwgZWxlbWVudHMgaGF2ZSBiZWVuIHByb2Nlc3NlZC5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ0RhdGEnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBhc3luYyBmb3JFYWNoQXN5bmMoZjogKGlucHV0OiBUKSA9PiB2b2lkKTogUHJvbWlzZTx2b2lkPiB7XG4gICAgcmV0dXJuIChhd2FpdCB0aGlzLml0ZXJhdG9yKCkpLmZvckVhY2hBc3luYyhmKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBNYXBzIHRoaXMgZGF0YXNldCB0aHJvdWdoIGEgMS10by0xIHRyYW5zZm9ybS5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoWzEsIDIsIDNdKS5tYXAoeCA9PiB4KngpO1xuICAgKiBhd2FpdCBhLmZvckVhY2hBc3luYyhlID0+IGNvbnNvbGUubG9nKGUpKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSB0cmFuc2Zvcm0gQSBmdW5jdGlvbiBtYXBwaW5nIGEgZGF0YXNldCBlbGVtZW50IHRvIGEgdHJhbnNmb3JtZWRcbiAgICogICBkYXRhc2V0IGVsZW1lbnQuXG4gICAqXG4gICAqIEByZXR1cm5zIEEgYERhdGFzZXRgIG9mIHRyYW5zZm9ybWVkIGVsZW1lbnRzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIG1hcDxPIGV4dGVuZHMgdGYuVGVuc29yQ29udGFpbmVyPih0cmFuc2Zvcm06ICh2YWx1ZTogVCkgPT4gTyk6IERhdGFzZXQ8Tz4ge1xuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIHJldHVybiBkYXRhc2V0RnJvbUl0ZXJhdG9yRm4oYXN5bmMgKCkgPT4ge1xuICAgICAgcmV0dXJuIChhd2FpdCBiYXNlLml0ZXJhdG9yKCkpLm1hcCh4ID0+IHRmLnRpZHkoKCkgPT4gdHJhbnNmb3JtKHgpKSk7XG4gICAgfSwgdGhpcy5zaXplKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBNYXBzIHRoaXMgZGF0YXNldCB0aHJvdWdoIGFuIGFzeW5jIDEtdG8tMSB0cmFuc2Zvcm0uXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGEgPVxuICAgKiAgdGYuZGF0YS5hcnJheShbMSwgMiwgM10pLm1hcEFzeW5jKHggPT4gbmV3IFByb21pc2UoZnVuY3Rpb24ocmVzb2x2ZSl7XG4gICAqICAgIHNldFRpbWVvdXQoKCkgPT4ge1xuICAgKiAgICAgIHJlc29sdmUoeCAqIHgpO1xuICAgKiAgICB9LCBNYXRoLnJhbmRvbSgpKjEwMDAgKyA1MDApO1xuICAgKiAgfSkpO1xuICAgKiBjb25zb2xlLmxvZyhhd2FpdCBhLnRvQXJyYXkoKSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0gdHJhbnNmb3JtIEEgZnVuY3Rpb24gbWFwcGluZyBhIGRhdGFzZXQgZWxlbWVudCB0byBhIGBQcm9taXNlYCBmb3IgYVxuICAgKiAgIHRyYW5zZm9ybWVkIGRhdGFzZXQgZWxlbWVudC4gIFRoaXMgdHJhbnNmb3JtIGlzIHJlc3BvbnNpYmxlIGZvciBkaXNwb3NpbmdcbiAgICogICBhbnkgaW50ZXJtZWRpYXRlIGBUZW5zb3JgcywgaS5lLiBieSB3cmFwcGluZyBpdHMgY29tcHV0YXRpb24gaW5cbiAgICogICBgdGYudGlkeSgpYDsgdGhhdCBjYW5ub3QgYmUgYXV0b21hdGVkIGhlcmUgKGFzIGl0IGlzIGluIHRoZSBzeW5jaHJvbm91c1xuICAgKiAgIGBtYXAoKWAgY2FzZSkuXG4gICAqXG4gICAqIEByZXR1cm5zIEEgYERhdGFzZXRgIG9mIHRyYW5zZm9ybWVkIGVsZW1lbnRzLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIG1hcEFzeW5jPE8gZXh0ZW5kcyB0Zi5UZW5zb3JDb250YWluZXI+KHRyYW5zZm9ybTogKHZhbHVlOiBUKSA9PiBQcm9taXNlPE8+KTpcbiAgICAgIERhdGFzZXQ8Tz4ge1xuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIHJldHVybiBkYXRhc2V0RnJvbUl0ZXJhdG9yRm4oYXN5bmMgKCkgPT4ge1xuICAgICAgcmV0dXJuIChhd2FpdCBiYXNlLml0ZXJhdG9yKCkpLm1hcEFzeW5jKHRyYW5zZm9ybSk7XG4gICAgfSwgdGhpcy5zaXplKTtcbiAgfVxuXG4gIC8qKlxuICAgKiAgQ3JlYXRlcyBhIGBEYXRhc2V0YCB0aGF0IHByZWZldGNoZXMgZWxlbWVudHMgZnJvbSB0aGlzIGRhdGFzZXQuXG4gICAqXG4gICAqIEBwYXJhbSBidWZmZXJTaXplOiBBbiBpbnRlZ2VyIHNwZWNpZnlpbmcgdGhlIG51bWJlciBvZiBlbGVtZW50cyB0byBiZVxuICAgKiAgIHByZWZldGNoZWQuXG4gICAqIEByZXR1cm5zIEEgYERhdGFzZXRgLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHByZWZldGNoKGJ1ZmZlclNpemU6IG51bWJlcik6IERhdGFzZXQ8VD4ge1xuICAgIGlmIChidWZmZXJTaXplID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBSYW5nZUVycm9yKFxuICAgICAgICAgICdgRGF0YXNldC5wcmVmZXRjaCgpYCByZXF1aXJlcyBidWZmZXJTaXplIHRvIGJlIHNwZWNpZmllZC4nKTtcbiAgICB9XG5cbiAgICBjb25zdCBiYXNlID0gdGhpcztcbiAgICByZXR1cm4gZGF0YXNldEZyb21JdGVyYXRvckZuKFxuICAgICAgICBhc3luYyAoKSA9PiAoYXdhaXQgYmFzZS5pdGVyYXRvcigpKS5wcmVmZXRjaChidWZmZXJTaXplKSwgdGhpcy5zaXplKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXBlYXRzIHRoaXMgZGF0YXNldCBgY291bnRgIHRpbWVzLlxuICAgKlxuICAgKiBOT1RFOiBJZiB0aGlzIGRhdGFzZXQgaXMgYSBmdW5jdGlvbiBvZiBnbG9iYWwgc3RhdGUgKGUuZy4gYSByYW5kb20gbnVtYmVyXG4gICAqIGdlbmVyYXRvciksIHRoZW4gZGlmZmVyZW50IHJlcGV0aXRpb25zIG1heSBwcm9kdWNlIGRpZmZlcmVudCBlbGVtZW50cy5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoWzEsIDIsIDNdKS5yZXBlYXQoMyk7XG4gICAqIGF3YWl0IGEuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIGNvdW50OiAoT3B0aW9uYWwpIEFuIGludGVnZXIsIHJlcHJlc2VudGluZyB0aGUgbnVtYmVyIG9mIHRpbWVzXG4gICAqICAgdGhlIGRhdGFzZXQgc2hvdWxkIGJlIHJlcGVhdGVkLiBUaGUgZGVmYXVsdCBiZWhhdmlvciAoaWYgYGNvdW50YCBpc1xuICAgKiAgIGB1bmRlZmluZWRgIG9yIG5lZ2F0aXZlKSBpcyBmb3IgdGhlIGRhdGFzZXQgYmUgcmVwZWF0ZWQgaW5kZWZpbml0ZWx5LlxuICAgKiBAcmV0dXJucyBBIGBEYXRhc2V0YC5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ0RhdGEnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICByZXBlYXQoY291bnQ/OiBudW1iZXIpOiBEYXRhc2V0PFQ+IHtcbiAgICBjb25zdCBiYXNlID0gdGhpcztcbiAgICBsZXQgc2l6ZTtcbiAgICBpZiAodGhpcy5zaXplICE9IG51bGwgJiYgY291bnQgPiAwKSB7XG4gICAgICAvLyBJZiB0aGlzIGRhdGFzZXQgaGFzIHNpemUgYW5kIGNvdW50IGlzIHBvc2l0aXZlLCBuZXcgc2l6ZSBpcyBjdXJyZW50XG4gICAgICAvLyBzaXplIG11bHRpcGx5IGNvdW50LiBUaGlzIGFsc28gY292ZXJzIHRoZSBjYXNlIHRoYXQgY3VycmVudCBzaXplIGlzXG4gICAgICAvLyBpbmZpbml0eS5cbiAgICAgIHNpemUgPSB0aGlzLnNpemUgKiBjb3VudDtcbiAgICB9IGVsc2UgaWYgKGNvdW50ID09PSAwKSB7XG4gICAgICAvLyBJZiBjb3VudCBpcyAwLCBuZXcgc2l6ZSBpcyAwLlxuICAgICAgc2l6ZSA9IDA7XG4gICAgfSBlbHNlIGlmICh0aGlzLnNpemUgIT0gbnVsbCAmJiAoY291bnQgPT09IHVuZGVmaW5lZCB8fCBjb3VudCA8IDApKSB7XG4gICAgICAvLyBJZiB0aGlzIGRhdGFzZXQgaGFzIHNpemUgYW5kIGNvdW50IGlzIHVuZGVmaW5lZCBvciBuZWdhdGl2ZSwgdGhlXG4gICAgICAvLyBkYXRhc2V0IHdpbGwgYmUgcmVwZWF0ZWQgaW5kZWZpbml0ZWx5IGFuZCBuZXcgc2l6ZSBpcyBpbmZpbml0eS5cbiAgICAgIHNpemUgPSBJbmZpbml0eTtcbiAgICB9IGVsc2Uge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0IGlzIG51bGwsIHRoZSBuZXcgZGF0YXNldCdzIHNpemUgaXMgbnVsbC5cbiAgICAgIHNpemUgPSBudWxsO1xuICAgIH1cbiAgICByZXR1cm4gZGF0YXNldEZyb21JdGVyYXRvckZuKGFzeW5jICgpID0+IHtcbiAgICAgIGNvbnN0IGl0ZXJhdG9ySXRlcmF0b3IgPSBpdGVyYXRvckZyb21GdW5jdGlvbihcbiAgICAgICAgICBhc3luYyAoKSA9PiAoe3ZhbHVlOiBhd2FpdCBiYXNlLml0ZXJhdG9yKCksIGRvbmU6IGZhbHNlfSkpO1xuICAgICAgcmV0dXJuIGl0ZXJhdG9yRnJvbUNvbmNhdGVuYXRlZChpdGVyYXRvckl0ZXJhdG9yLnRha2UoY291bnQpKTtcbiAgICB9LCBzaXplKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDcmVhdGVzIGEgYERhdGFzZXRgIHRoYXQgc2tpcHMgYGNvdW50YCBpbml0aWFsIGVsZW1lbnRzIGZyb20gdGhpcyBkYXRhc2V0LlxuICAgKlxuICAgKiBgYGBqc1xuICAgKiBjb25zdCBhID0gdGYuZGF0YS5hcnJheShbMSwgMiwgMywgNCwgNSwgNl0pLnNraXAoMyk7XG4gICAqIGF3YWl0IGEuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHBhcmFtIGNvdW50OiBUaGUgbnVtYmVyIG9mIGVsZW1lbnRzIG9mIHRoaXMgZGF0YXNldCB0aGF0IHNob3VsZCBiZSBza2lwcGVkXG4gICAqICAgdG8gZm9ybSB0aGUgbmV3IGRhdGFzZXQuICBJZiBgY291bnRgIGlzIGdyZWF0ZXIgdGhhbiB0aGUgc2l6ZSBvZiB0aGlzXG4gICAqICAgZGF0YXNldCwgdGhlIG5ldyBkYXRhc2V0IHdpbGwgY29udGFpbiBubyBlbGVtZW50cy4gIElmIGBjb3VudGBcbiAgICogICBpcyBgdW5kZWZpbmVkYCBvciBuZWdhdGl2ZSwgc2tpcHMgdGhlIGVudGlyZSBkYXRhc2V0LlxuICAgKlxuICAgKiBAcmV0dXJucyBBIGBEYXRhc2V0YC5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ0RhdGEnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBza2lwKGNvdW50OiBudW1iZXIpOiBEYXRhc2V0PFQ+IHtcbiAgICBjb25zdCBiYXNlID0gdGhpcztcbiAgICBsZXQgc2l6ZTtcbiAgICBpZiAodGhpcy5zaXplICE9IG51bGwgJiYgY291bnQgPj0gMCAmJiB0aGlzLnNpemUgPj0gY291bnQpIHtcbiAgICAgIC8vIElmIHRoZSBzaXplIG9mIHRoaXMgZGF0YXNldCBpcyBncmVhdGVyIHRoYW4gY291bnQsIHRoZSBuZXcgZGF0YXNldCdzXG4gICAgICAvLyBzaXplIGlzIGN1cnJlbnQgc2l6ZSBtaW51cyBza2lwcGVkIHNpemUuVGhpcyBhbHNvIGNvdmVycyB0aGUgY2FzZSB0aGF0XG4gICAgICAvLyBjdXJyZW50IHNpemUgaXMgaW5maW5pdHkuXG4gICAgICBzaXplID0gdGhpcy5zaXplIC0gY291bnQ7XG4gICAgfSBlbHNlIGlmIChcbiAgICAgICAgdGhpcy5zaXplICE9IG51bGwgJiZcbiAgICAgICAgKHRoaXMuc2l6ZSA8IGNvdW50IHx8IGNvdW50ID09PSB1bmRlZmluZWQgfHwgY291bnQgPCAwKSkge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0IGlzIHNtYWxsZXIgdGhhbiBjb3VudCwgb3IgY291bnQgaXNcbiAgICAgIC8vIHVuZGVmaW5lZCBvciBuZWdhdGl2ZSwgc2tpcHMgdGhlIGVudGlyZSBkYXRhc2V0IGFuZCB0aGUgbmV3IHNpemUgaXMgMC5cbiAgICAgIHNpemUgPSAwO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBJZiB0aGUgc2l6ZSBvZiB0aGlzIGRhdGFzZXQgaXMgbnVsbCwgdGhlIG5ldyBkYXRhc2V0J3Mgc2l6ZSBpcyBudWxsLlxuICAgICAgc2l6ZSA9IG51bGw7XG4gICAgfVxuICAgIHJldHVybiBkYXRhc2V0RnJvbUl0ZXJhdG9yRm4oXG4gICAgICAgIGFzeW5jICgpID0+IChhd2FpdCBiYXNlLml0ZXJhdG9yKCkpLnNraXAoY291bnQpLCBzaXplKTtcbiAgfVxuXG4gIC8vIFRPRE8oc29lcmdlbCk6IGRlZXAgc2hhcmRlZCBzaHVmZmxlLCB3aGVyZSBzdXBwb3J0ZWRcblxuICBzdGF0aWMgcmVhZG9ubHkgTUFYX0JVRkZFUl9TSVpFID0gMTAwMDA7XG5cbiAgLyoqXG4gICAqIFBzZXVkb3JhbmRvbWx5IHNodWZmbGVzIHRoZSBlbGVtZW50cyBvZiB0aGlzIGRhdGFzZXQuIFRoaXMgaXMgZG9uZSBpbiBhXG4gICAqIHN0cmVhbWluZyBtYW5uZXIsIGJ5IHNhbXBsaW5nIGZyb20gYSBnaXZlbiBudW1iZXIgb2YgcHJlZmV0Y2hlZCBlbGVtZW50cy5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoWzEsIDIsIDMsIDQsIDUsIDZdKS5zaHVmZmxlKDMpO1xuICAgKiBhd2FpdCBhLmZvckVhY2hBc3luYyhlID0+IGNvbnNvbGUubG9nKGUpKTtcbiAgICogYGBgXG4gICAqXG4gICAqIEBwYXJhbSBidWZmZXJTaXplOiBBbiBpbnRlZ2VyIHNwZWNpZnlpbmcgdGhlIG51bWJlciBvZiBlbGVtZW50cyBmcm9tIHRoaXNcbiAgICogICBkYXRhc2V0IGZyb20gd2hpY2ggdGhlIG5ldyBkYXRhc2V0IHdpbGwgc2FtcGxlLlxuICAgKiBAcGFyYW0gc2VlZDogKE9wdGlvbmFsKSBBbiBpbnRlZ2VyIHNwZWNpZnlpbmcgdGhlIHJhbmRvbSBzZWVkIHRoYXQgd2lsbFxuICAgKiAgIGJlIHVzZWQgdG8gY3JlYXRlIHRoZSBkaXN0cmlidXRpb24uXG4gICAqIEBwYXJhbSByZXNodWZmbGVFYWNoSXRlcmF0aW9uOiAoT3B0aW9uYWwpIEEgYm9vbGVhbiwgd2hpY2ggaWYgdHJ1ZVxuICAgKiAgIGluZGljYXRlcyB0aGF0IHRoZSBkYXRhc2V0IHNob3VsZCBiZSBwc2V1ZG9yYW5kb21seSByZXNodWZmbGVkIGVhY2ggdGltZVxuICAgKiAgIGl0IGlzIGl0ZXJhdGVkIG92ZXIuIElmIGZhbHNlLCBlbGVtZW50cyB3aWxsIGJlIHJldHVybmVkIGluIHRoZSBzYW1lXG4gICAqICAgc2h1ZmZsZWQgb3JkZXIgb24gZWFjaCBpdGVyYXRpb24uIChEZWZhdWx0cyB0byBgdHJ1ZWAuKVxuICAgKiBAcmV0dXJucyBBIGBEYXRhc2V0YC5cbiAgICpcbiAgICogQGRvYyB7aGVhZGluZzogJ0RhdGEnLCBzdWJoZWFkaW5nOiAnQ2xhc3Nlcyd9XG4gICAqL1xuICBzaHVmZmxlKGJ1ZmZlclNpemU6IG51bWJlciwgc2VlZD86IHN0cmluZywgcmVzaHVmZmxlRWFjaEl0ZXJhdGlvbiA9IHRydWUpOlxuICAgICAgRGF0YXNldDxUPiB7XG4gICAgaWYgKGJ1ZmZlclNpemUgPT0gbnVsbCB8fCBidWZmZXJTaXplIDwgMCkge1xuICAgICAgaWYgKHRoaXMuc2l6ZSA9PSBudWxsKSB7XG4gICAgICAgIHRocm93IG5ldyBSYW5nZUVycm9yKFxuICAgICAgICAgICAgJ2BEYXRhc2V0LnNodWZmbGUoKWAgcmVxdWlyZXMgYnVmZmVyU2l6ZSB0byBiZSBzcGVjaWZpZWQuJyk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aHJvdyBuZXcgUmFuZ2VFcnJvcihcbiAgICAgICAgICAgICdgRGF0YXNldC5zaHVmZmxlKClgIHJlcXVpcmVzIGJ1ZmZlclNpemUgdG8gYmUgc3BlY2lmaWVkLiAgJyArXG4gICAgICAgICAgICAnSWYgeW91ciBkYXRhIGZpdHMgaW4gbWFpbiBtZW1vcnkgKGZvciByZWd1bGFyIEpTIG9iamVjdHMpLCAnICtcbiAgICAgICAgICAgICdhbmQvb3IgR1BVIG1lbW9yeSAoZm9yIGB0Zi5UZW5zb3JgcyksIGNvbnNpZGVyIHNldHRpbmcgJyArXG4gICAgICAgICAgICBgYnVmZmVyU2l6ZSB0byB0aGUgZGF0YXNldCBzaXplICgke3RoaXMuc2l6ZX0gZWxlbWVudHMpYCk7XG4gICAgICB9XG4gICAgfVxuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIGNvbnN0IHJhbmRvbSA9IHNlZWRyYW5kb20uYWxlYShzZWVkIHx8IHRmLnV0aWwubm93KCkudG9TdHJpbmcoKSk7XG4gICAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbihhc3luYyAoKSA9PiB7XG4gICAgICBsZXQgc2VlZDIgPSByYW5kb20uaW50MzIoKTtcbiAgICAgIGlmIChyZXNodWZmbGVFYWNoSXRlcmF0aW9uKSB7XG4gICAgICAgIHNlZWQyICs9IHJhbmRvbS5pbnQzMigpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIChhd2FpdCBiYXNlLml0ZXJhdG9yKCkpLnNodWZmbGUoYnVmZmVyU2l6ZSwgc2VlZDIudG9TdHJpbmcoKSk7XG4gICAgfSwgdGhpcy5zaXplKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDcmVhdGVzIGEgYERhdGFzZXRgIHdpdGggYXQgbW9zdCBgY291bnRgIGluaXRpYWwgZWxlbWVudHMgZnJvbSB0aGlzXG4gICAqIGRhdGFzZXQuXG4gICAqXG4gICAqIGBgYGpzXG4gICAqIGNvbnN0IGEgPSB0Zi5kYXRhLmFycmF5KFsxLCAyLCAzLCA0LCA1LCA2XSkudGFrZSgzKTtcbiAgICogYXdhaXQgYS5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhlKSk7XG4gICAqIGBgYFxuICAgKlxuICAgKiBAcGFyYW0gY291bnQ6IFRoZSBudW1iZXIgb2YgZWxlbWVudHMgb2YgdGhpcyBkYXRhc2V0IHRoYXQgc2hvdWxkIGJlIHRha2VuXG4gICAqICAgdG8gZm9ybSB0aGUgbmV3IGRhdGFzZXQuICBJZiBgY291bnRgIGlzIGB1bmRlZmluZWRgIG9yIG5lZ2F0aXZlLCBvciBpZlxuICAgKiAgIGBjb3VudGAgaXMgZ3JlYXRlciB0aGFuIHRoZSBzaXplIG9mIHRoaXMgZGF0YXNldCwgdGhlIG5ldyBkYXRhc2V0IHdpbGxcbiAgICogICBjb250YWluIGFsbCBlbGVtZW50cyBvZiB0aGlzIGRhdGFzZXQuXG4gICAqIEByZXR1cm5zIEEgYERhdGFzZXRgLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIHRha2UoY291bnQ6IG51bWJlcik6IERhdGFzZXQ8VD4ge1xuICAgIGNvbnN0IGJhc2UgPSB0aGlzO1xuICAgIGxldCBzaXplO1xuICAgIGlmICh0aGlzLnNpemUgIT0gbnVsbCAmJiB0aGlzLnNpemUgPiBjb3VudCkge1xuICAgICAgLy8gSWYgdGhlIHNpemUgb2YgdGhpcyBkYXRhc2V0IGlzIGdyZWF0ZXIgdGhhbiBjb3VudCwgdGhlIG5ldyBkYXRhc2V0J3NcbiAgICAgIC8vIHNpemUgaXMgY291bnQuXG4gICAgICBzaXplID0gY291bnQ7XG4gICAgfSBlbHNlIGlmICh0aGlzLnNpemUgIT0gbnVsbCAmJiB0aGlzLnNpemUgPD0gY291bnQpIHtcbiAgICAgIC8vIElmIHRoZSBzaXplIG9mIHRoaXMgZGF0YXNldCBpcyBlcXVhbCBvciBzbWFsbGVyIHRoYW4gY291bnQsIHRoZSBuZXdcbiAgICAgIC8vIGRhdGFzZXQncyBzaXplIGlzIHRoZSBzaXplIG9mIHRoaXMgZGF0YXNldC5cbiAgICAgIHNpemUgPSB0aGlzLnNpemU7XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIElmIHRoZSBzaXplIG9mIHRoaXMgZGF0YXNldCBpcyBudWxsLCB0aGUgbmV3IGRhdGFzZXQncyBzaXplIGlzIG51bGwuXG4gICAgICBzaXplID0gbnVsbDtcbiAgICB9XG4gICAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbihcbiAgICAgICAgYXN5bmMgKCkgPT4gKGF3YWl0IGJhc2UuaXRlcmF0b3IoKSkudGFrZShjb3VudCksIHNpemUpO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbGxlY3QgYWxsIGVsZW1lbnRzIG9mIHRoaXMgZGF0YXNldCBpbnRvIGFuIGFycmF5LlxuICAgKlxuICAgKiBPYnZpb3VzbHkgdGhpcyB3aWxsIHN1Y2NlZWQgb25seSBmb3Igc21hbGwgZGF0YXNldHMgdGhhdCBmaXQgaW4gbWVtb3J5LlxuICAgKiBVc2VmdWwgZm9yIHRlc3RpbmcgYW5kIGdlbmVyYWxseSBzaG91bGQgYmUgYXZvaWRlZCBpZiBwb3NzaWJsZS5cbiAgICpcbiAgICogYGBganNcbiAgICogY29uc3QgYSA9IHRmLmRhdGEuYXJyYXkoWzEsIDIsIDMsIDQsIDUsIDZdKTtcbiAgICogY29uc29sZS5sb2coYXdhaXQgYS50b0FycmF5KCkpO1xuICAgKiBgYGBcbiAgICpcbiAgICogQHJldHVybnMgQSBQcm9taXNlIGZvciBhbiBhcnJheSBvZiBlbGVtZW50cywgd2hpY2ggd2lsbCByZXNvbHZlXG4gICAqICAgd2hlbiBhIG5ldyBzdHJlYW0gaGFzIGJlZW4gb2J0YWluZWQgYW5kIGZ1bGx5IGNvbnN1bWVkLlxuICAgKlxuICAgKiBAZG9jIHtoZWFkaW5nOiAnRGF0YScsIHN1YmhlYWRpbmc6ICdDbGFzc2VzJ31cbiAgICovXG4gIGFzeW5jIHRvQXJyYXkoKSB7XG4gICAgaWYgKHRoaXMuc2l6ZSA9PT0gSW5maW5pdHkpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignQ2FuIG5vdCBjb252ZXJ0IGluZmluaXRlIGRhdGEgc3RyZWFtIHRvIGFycmF5LicpO1xuICAgIH1cbiAgICByZXR1cm4gKGF3YWl0IHRoaXMuaXRlcmF0b3IoKSkudG9BcnJheSgpO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbGxlY3QgYWxsIGVsZW1lbnRzIG9mIHRoaXMgZGF0YXNldCBpbnRvIGFuIGFycmF5IHdpdGggcHJlZmV0Y2hpbmcgMTAwXG4gICAqIGVsZW1lbnRzLiBUaGlzIGlzIHVzZWZ1bCBmb3IgdGVzdGluZywgYmVjYXVzZSB0aGUgcHJlZmV0Y2ggY2hhbmdlcyB0aGVcbiAgICogb3JkZXIgaW4gd2hpY2ggdGhlIFByb21pc2VzIGFyZSByZXNvbHZlZCBhbG9uZyB0aGUgcHJvY2Vzc2luZyBwaXBlbGluZS5cbiAgICogVGhpcyBtYXkgaGVscCBleHBvc2UgYnVncyB3aGVyZSByZXN1bHRzIGFyZSBkZXBlbmRlbnQgb24gdGhlIG9yZGVyIG9mXG4gICAqIFByb21pc2UgcmVzb2x1dGlvbiByYXRoZXIgdGhhbiBvbiB0aGUgbG9naWNhbCBvcmRlciBvZiB0aGUgc3RyZWFtIChpLmUuLFxuICAgKiBkdWUgdG8gaGlkZGVuIG11dGFibGUgc3RhdGUpLlxuICAgKlxuICAgKiBAcmV0dXJucyBBIFByb21pc2UgZm9yIGFuIGFycmF5IG9mIGVsZW1lbnRzLCB3aGljaCB3aWxsIHJlc29sdmVcbiAgICogICB3aGVuIGEgbmV3IHN0cmVhbSBoYXMgYmVlbiBvYnRhaW5lZCBhbmQgZnVsbHkgY29uc3VtZWQuXG4gICAqL1xuICBhc3luYyB0b0FycmF5Rm9yVGVzdCgpIHtcbiAgICBpZiAodGhpcy5zaXplID09PSBJbmZpbml0eSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdDYW4gbm90IGNvbnZlcnQgaW5maW5pdGUgZGF0YSBzdHJlYW0gdG8gYXJyYXkuJyk7XG4gICAgfVxuICAgIHJldHVybiAoYXdhaXQgdGhpcy5pdGVyYXRvcigpKS50b0FycmF5Rm9yVGVzdCgpO1xuICB9XG59XG5cbi8qKlxuICogQ3JlYXRlIGEgYERhdGFzZXRgIGRlZmluZWQgYnkgYSBwcm92aWRlZCBpdGVyYXRvcigpIGZ1bmN0aW9uLlxuICpcbiAqIGBgYGpzXG4gKiBsZXQgaSA9IC0xO1xuICogY29uc3QgZnVuYyA9ICgpID0+XG4gKiAgICArK2kgPCA1ID8ge3ZhbHVlOiBpLCBkb25lOiBmYWxzZX0gOiB7dmFsdWU6IG51bGwsIGRvbmU6IHRydWV9O1xuICogY29uc3QgaXRlciA9IHRmLmRhdGEuaXRlcmF0b3JGcm9tRnVuY3Rpb24oZnVuYyk7XG4gKiBjb25zdCBkcyA9IHRmLmRhdGEuZGF0YXNldEZyb21JdGVyYXRvckZuKGl0ZXIpO1xuICogYXdhaXQgZHMuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICogYGBgXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBkYXRhc2V0RnJvbUl0ZXJhdG9yRm48VCBleHRlbmRzIHRmLlRlbnNvckNvbnRhaW5lcj4oXG4gICAgaXRlcmF0b3JGbjogKCkgPT4gUHJvbWlzZTxMYXp5SXRlcmF0b3I8VD4+LFxuICAgIHNpemU6IG51bWJlciA9IG51bGwpOiBEYXRhc2V0PFQ+IHtcbiAgcmV0dXJuIG5ldyBjbGFzcyBleHRlbmRzIERhdGFzZXQ8VD4ge1xuICAgIHNpemUgPSBzaXplO1xuXG4gICAgLypcbiAgICAgKiBQcm92aWRlIGEgbmV3IHN0cmVhbSBvZiBlbGVtZW50cy4gIE5vdGUgdGhpcyB3aWxsIGFsc28gc3RhcnQgbmV3IHN0cmVhbXNcbiAgICAgKiBmcm9tIGFueSB1bmRlcmx5aW5nIGBEYXRhc2V0YHMuXG4gICAgICovXG4gICAgYXN5bmMgaXRlcmF0b3IoKTogUHJvbWlzZTxMYXp5SXRlcmF0b3I8VD4+IHtcbiAgICAgIHJldHVybiBpdGVyYXRvckZuKCk7XG4gICAgfVxuICB9XG4gICgpO1xufVxuXG4vKipcbiAqIENyZWF0ZSBhIGBEYXRhc2V0YCBmcm9tIGFuIGFycmF5IG9mIGVsZW1lbnRzLlxuICpcbiAqIENyZWF0ZSBhIERhdGFzZXQgZnJvbSBhbiBhcnJheSBvZiBvYmplY3RzOlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi5kYXRhLmFycmF5KFt7J2l0ZW0nOiAxfSwgeydpdGVtJzogMn0sIHsnaXRlbSc6IDN9XSk7XG4gKiBhd2FpdCBhLmZvckVhY2hBc3luYyhlID0+IGNvbnNvbGUubG9nKGUpKTtcbiAqIGBgYFxuICpcbiAqIENyZWF0ZSBhIERhdGFzZXQgZnJvbSBhbiBhcnJheSBvZiBudW1iZXJzOlxuICogYGBganNcbiAqIGNvbnN0IGEgPSB0Zi5kYXRhLmFycmF5KFs0LCA1LCA2XSk7XG4gKiBhd2FpdCBhLmZvckVhY2hBc3luYyhlID0+IGNvbnNvbGUubG9nKGUpKTtcbiAqIGBgYFxuICogQHBhcmFtIGl0ZW1zIEFuIGFycmF5IG9mIGVsZW1lbnRzIHRoYXQgd2lsbCBiZSBwYXJzZWQgYXMgaXRlbXMgaW4gYSBkYXRhc2V0LlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdEYXRhJywgc3ViaGVhZGluZzogJ0NyZWF0aW9uJywgbmFtZXNwYWNlOiAnZGF0YSd9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBhcnJheTxUIGV4dGVuZHMgdGYuVGVuc29yQ29udGFpbmVyPihpdGVtczogVFtdKTogRGF0YXNldDxUPiB7XG4gIHJldHVybiBkYXRhc2V0RnJvbUl0ZXJhdG9yRm4oXG4gICAgICBhc3luYyAoKSA9PiBpdGVyYXRvckZyb21JdGVtcyhpdGVtcyksIGl0ZW1zLmxlbmd0aCk7XG59XG5cbi8qKlxuICogQ3JlYXRlIGEgYERhdGFzZXRgIGJ5IHppcHBpbmcgdG9nZXRoZXIgYW4gYXJyYXksIGRpY3QsIG9yIG5lc3RlZFxuICogc3RydWN0dXJlIG9mIGBEYXRhc2V0YHMgKGFuZCBwZXJoYXBzIGFkZGl0aW9uYWwgY29uc3RhbnRzKS5cbiAqIFRoZSB1bmRlcmx5aW5nIGRhdGFzZXRzIG11c3QgcHJvdmlkZSBlbGVtZW50cyBpbiBhIGNvbnNpc3RlbnQgb3JkZXIgc3VjaCB0aGF0XG4gKiB0aGV5IGNvcnJlc3BvbmQuXG4gKlxuICogVGhlIG51bWJlciBvZiBlbGVtZW50cyBpbiB0aGUgcmVzdWx0aW5nIGRhdGFzZXQgaXMgdGhlIHNhbWUgYXMgdGhlIHNpemUgb2ZcbiAqIHRoZSBzbWFsbGVzdCBkYXRhc2V0IGluIGRhdGFzZXRzLlxuICpcbiAqIFRoZSBuZXN0ZWQgc3RydWN0dXJlIG9mIHRoZSBgZGF0YXNldHNgIGFyZ3VtZW50IGRldGVybWluZXMgdGhlXG4gKiBzdHJ1Y3R1cmUgb2YgZWxlbWVudHMgaW4gdGhlIHJlc3VsdGluZyBpdGVyYXRvci5cbiAqXG4gKiBOb3RlIHRoaXMgbWVhbnMgdGhhdCwgZ2l2ZW4gYW4gYXJyYXkgb2YgdHdvIGRhdGFzZXRzIHRoYXQgcHJvZHVjZSBkaWN0XG4gKiBlbGVtZW50cywgdGhlIHJlc3VsdCBpcyBhIGRhdGFzZXQgdGhhdCBwcm9kdWNlcyBlbGVtZW50cyB0aGF0IGFyZSBhcnJheXNcbiAqIG9mIHR3byBkaWN0czpcbiAqXG4gKiBaaXAgYW4gYXJyYXkgb2YgZGF0YXNldHM6XG4gKiBgYGBqc1xuICogY29uc29sZS5sb2coJ1ppcCB0d28gZGF0YXNldHMgb2Ygb2JqZWN0czonKTtcbiAqIGNvbnN0IGRzMSA9IHRmLmRhdGEuYXJyYXkoW3thOiAxfSwge2E6IDJ9LCB7YTogM31dKTtcbiAqIGNvbnN0IGRzMiA9IHRmLmRhdGEuYXJyYXkoW3tiOiA0fSwge2I6IDV9LCB7YjogNn1dKTtcbiAqIGNvbnN0IGRzMyA9IHRmLmRhdGEuemlwKFtkczEsIGRzMl0pO1xuICogYXdhaXQgZHMzLmZvckVhY2hBc3luYyhlID0+IGNvbnNvbGUubG9nKEpTT04uc3RyaW5naWZ5KGUpKSk7XG4gKlxuICogLy8gSWYgdGhlIGdvYWwgaXMgdG8gbWVyZ2UgdGhlIGRpY3RzIGluIG9yZGVyIHRvIHByb2R1Y2UgZWxlbWVudHMgbGlrZVxuICogLy8ge2E6IC4uLiwgYjogLi4ufSwgdGhpcyByZXF1aXJlcyBhIHNlY29uZCBzdGVwIHN1Y2ggYXM6XG4gKiBjb25zb2xlLmxvZygnTWVyZ2UgdGhlIG9iamVjdHM6Jyk7XG4gKiBjb25zdCBkczQgPSBkczMubWFwKHggPT4ge3JldHVybiB7YTogeFswXS5hLCBiOiB4WzFdLmJ9fSk7XG4gKiBhd2FpdCBkczQuZm9yRWFjaEFzeW5jKGUgPT4gY29uc29sZS5sb2coZSkpO1xuICogYGBgXG4gKlxuICogWmlwIGEgZGljdCBvZiBkYXRhc2V0czpcbiAqIGBgYGpzXG4gKiBjb25zdCBhID0gdGYuZGF0YS5hcnJheShbe2E6IDF9LCB7YTogMn0sIHthOiAzfV0pO1xuICogY29uc3QgYiA9IHRmLmRhdGEuYXJyYXkoW3tiOiA0fSwge2I6IDV9LCB7YjogNn1dKTtcbiAqIGNvbnN0IGMgPSB0Zi5kYXRhLnppcCh7YzogYSwgZDogYn0pO1xuICogYXdhaXQgYy5mb3JFYWNoQXN5bmMoZSA9PiBjb25zb2xlLmxvZyhKU09OLnN0cmluZ2lmeShlKSkpO1xuICogYGBgXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ0RhdGEnLCBzdWJoZWFkaW5nOiAnT3BlcmF0aW9ucycsIG5hbWVzcGFjZTogJ2RhdGEnfVxuICovXG5leHBvcnQgZnVuY3Rpb24gemlwPE8gZXh0ZW5kcyB0Zi5UZW5zb3JDb250YWluZXI+KGRhdGFzZXRzOiBEYXRhc2V0Q29udGFpbmVyKTpcbiAgICBEYXRhc2V0PE8+IHtcbiAgLy8gbWFudWFsbHkgdHlwZS1jaGVjayB0aGUgYXJndW1lbnQgZm9yIEpTIHVzZXJzXG4gIGlmICghaXNJdGVyYWJsZShkYXRhc2V0cykpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ1RoZSBhcmd1bWVudCB0byB6aXAoKSBtdXN0IGJlIGFuIG9iamVjdCBvciBhcnJheS4nKTtcbiAgfVxuICBsZXQgc2l6ZTtcbiAgaWYgKEFycmF5LmlzQXJyYXkoZGF0YXNldHMpKSB7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBkYXRhc2V0cy5sZW5ndGg7IGkrKykge1xuICAgICAgc2l6ZSA9IHNpemUgPT0gbnVsbCA/IChkYXRhc2V0c1tpXSBhcyBEYXRhc2V0PE8+KS5zaXplIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICBNYXRoLm1pbihzaXplLCAoZGF0YXNldHNbaV0gYXMgRGF0YXNldDxPPikuc2l6ZSk7XG4gICAgfVxuICB9IGVsc2UgaWYgKGRhdGFzZXRzIGluc3RhbmNlb2YgT2JqZWN0KSB7XG4gICAgZm9yIChjb25zdCBkcyBpbiBkYXRhc2V0cykge1xuICAgICAgc2l6ZSA9IHNpemUgPT0gbnVsbCA/IChkYXRhc2V0c1tkc10gYXMgRGF0YXNldDxPPikuc2l6ZSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgTWF0aC5taW4oc2l6ZSwgKGRhdGFzZXRzW2RzXSBhcyBEYXRhc2V0PE8+KS5zaXplKTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIGRhdGFzZXRGcm9tSXRlcmF0b3JGbjxPPihhc3luYyAoKSA9PiB7XG4gICAgY29uc3Qgc3RyZWFtcyA9IGF3YWl0IGRlZXBNYXBBbmRBd2FpdEFsbChkYXRhc2V0cywgZCA9PiB7XG4gICAgICBpZiAoZCBpbnN0YW5jZW9mIERhdGFzZXQpIHtcbiAgICAgICAgcmV0dXJuIHt2YWx1ZTogZC5pdGVyYXRvcigpLCByZWN1cnNlOiBmYWxzZX07XG4gICAgICB9IGVsc2UgaWYgKGlzSXRlcmFibGUoZCkpIHtcbiAgICAgICAgcmV0dXJuIHt2YWx1ZTogbnVsbCwgcmVjdXJzZTogdHJ1ZX07XG4gICAgICB9IGVsc2Uge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAnTGVhdmVzIG9mIHRoZSBzdHJ1Y3R1cmUgcGFzc2VkIHRvIHppcCgpIG11c3QgYmUgRGF0YXNldHMsICcgK1xuICAgICAgICAgICAgJ25vdCBwcmltaXRpdmVzLicpO1xuICAgICAgfVxuICAgIH0pO1xuICAgIHJldHVybiBpdGVyYXRvckZyb21aaXBwZWQ8Tz4oc3RyZWFtcywgWmlwTWlzbWF0Y2hNb2RlLlNIT1JURVNUKTtcbiAgfSwgc2l6ZSk7XG59XG5cbi8qKlxuICogQSB6aXAgZnVuY3Rpb24gZm9yIHVzZSB3aXRoIGRlZXBaaXAsIHBhc3NlZCB2aWEgdGhlIGNvbHVtbk1ham9yQmF0Y2ggY2FsbC5cbiAqXG4gKiBBY2NlcHRzIGFuIGFycmF5IG9mIGlkZW50aWNhbGx5LXN0cnVjdHVyZWQgbmVzdGVkIGVsZW1lbnRzIGFuZCBlaXRoZXIgYmF0Y2hlc1xuICogdGhlbSAoaWYgdGhleSBhcmUgcHJpbWl0aXZlcywgbnVtZXJpYyBhcnJheXMsIG9yIFRlbnNvcnMpIG9yIHJlcXVlc3RzXG4gKiByZWN1cnNpb24gKGlmIG5vdCkuXG4gKi9cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbmZ1bmN0aW9uIGRlZXBCYXRjaENvbmNhdChyb3dzOiBhbnlbXSk6IERlZXBNYXBSZXN1bHQge1xuICBpZiAocm93cyA9PT0gbnVsbCkge1xuICAgIHJldHVybiBudWxsO1xuICB9XG5cbiAgLy8gdXNlIHRoZSBmaXJzdCBpdGVtIHRvIGRlY2lkZSB3aGV0aGVyIHRvIHJlY3Vyc2Ugb3IgYmF0Y2ggaGVyZS5cbiAgY29uc3QgZXhhbXBsZVJvdyA9IHJvd3NbMF07XG5cbiAgaWYgKGNhblRlbnNvcmlmeShleGFtcGxlUm93KSkge1xuICAgIC8vIHJvd3MgaXMgYW4gYXJyYXkgb2YgcHJpbWl0aXZlcywgVGVuc29ycywgb3IgYXJyYXlzLiAgQmF0Y2ggdGhlbS5cbiAgICBjb25zdCB2YWx1ZSA9IGJhdGNoQ29uY2F0KHJvd3MpO1xuICAgIHJldHVybiB7dmFsdWUsIHJlY3Vyc2U6IGZhbHNlfTtcbiAgfVxuXG4gIC8vIHRoZSBleGFtcGxlIHJvdyBpcyBhbiBvYmplY3QsIHNvIHJlY3Vyc2UgaW50byBpdC5cbiAgcmV0dXJuIHt2YWx1ZTogbnVsbCwgcmVjdXJzZTogdHJ1ZX07XG59XG5cbi8qKlxuICogQXNzZW1ibGVzIGEgbGlzdCBvZiBzYW1lLXNoYXBlZCBudW1iZXJzLCBudW1iZXIgYXJyYXlzLCBvciBUZW5zb3JzXG4gKiBpbnRvIGEgc2luZ2xlIG5ldyBUZW5zb3Igd2hlcmUgYXhpcyAwIGlzIHRoZSBiYXRjaCBkaW1lbnNpb24uXG4gKi9cbmZ1bmN0aW9uIGJhdGNoQ29uY2F0PFQgZXh0ZW5kcyhUZW5zb3JMaWtlIHwgdGYuVGVuc29yKT4oYXJyYXlzOiBUW10pOlxuICAgIHRmLlRlbnNvciB7XG4gIGlmIChhcnJheXMubGVuZ3RoID09PSAwKSB7XG4gICAgLy8gV2UgY2FuJ3QgcmV0dXJuIGFuIGVtcHR5IFRlbnNvciBiZWNhdXNlIHdlIGRvbid0IGtub3cgdGhlIGVsZW1lbnQgc2hhcGUuXG4gICAgdGhyb3cgbmV3IEVycm9yKCdDYW5cXCd0IG1ha2UgYSBiYXRjaCBvZiB6ZXJvIGVsZW1lbnRzLicpO1xuICB9XG5cbiAgaWYgKGFycmF5c1swXSBpbnN0YW5jZW9mIHRmLlRlbnNvcikge1xuICAgIC8vIElucHV0IGlzIGFuIGFycmF5IG9mIFRlbnNvcnNcbiAgICByZXR1cm4gdGYuc3RhY2soYXJyYXlzIGFzIHRmLlRlbnNvcltdKTtcbiAgfSBlbHNlIHtcbiAgICAvLyBJbnB1dCBpcyBhIHBvc3NpYmx5LW5lc3RlZCBhcnJheSBvZiBudW1iZXJzLlxuICAgIHJldHVybiB0Zi50ZW5zb3IoYXJyYXlzIGFzIFRlbnNvckxpa2UpO1xuICB9XG59XG4iXX0=