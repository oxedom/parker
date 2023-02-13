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
import { concat, keep, reshape, scalar, slice, stack, tensor, tidy, unstack } from '@tensorflow/tfjs-core';
import { assertShapesMatchAllowUndefinedSize, inferElementShape, mergeElementShape } from './tensor_utils';
/**
 * TensorList stores a container of `tf.Tensor` objects, which are accessible
 * via tensors field.
 *
 * In order to get a copy of the underlying list, use the copy method:
 * ```
 *    TensorList b = a.copy();
 *    b.tensors().pushBack(t);  // This does not modify a.tensors().
 * ```
 *
 * Note that this is not a deep copy: the memory locations of the underlying
 * tensors will still point to the same locations of the corresponding tensors
 * in the original.
 */
export class TensorList {
    /**
     *
     * @param tensors list of tensors
     * @param elementShape shape of each tensor, this can be a single number (any
     * shape is allowed) or partial shape (dim = -1).
     * @param elementDtype data type of each tensor
     * @param maxNumElements The maximum allowed size of `tensors`. Defaults to -1
     *   meaning that the size of `tensors` is unbounded.
     */
    constructor(tensors, elementShape, elementDtype, maxNumElements = -1) {
        this.tensors = tensors;
        this.elementShape = elementShape;
        this.elementDtype = elementDtype;
        if (tensors != null) {
            tensors.forEach(tensor => {
                if (elementDtype !== tensor.dtype) {
                    throw new Error(`Invalid data types; op elements ${elementDtype}, but list elements ${tensor.dtype}`);
                }
                assertShapesMatchAllowUndefinedSize(elementShape, tensor.shape, 'TensorList shape mismatch: ');
                keep(tensor);
            });
        }
        this.idTensor = scalar(0);
        this.maxNumElements = maxNumElements;
        keep(this.idTensor);
    }
    get id() {
        return this.idTensor.id;
    }
    /**
     * Get a new TensorList containing a copy of the underlying tensor container.
     */
    copy() {
        return new TensorList([...this.tensors], this.elementShape, this.elementDtype);
    }
    /**
     * Dispose the tensors and idTensor and clear the tensor list.
     */
    clearAndClose(keepIds) {
        this.tensors.forEach(tensor => {
            if (keepIds == null || !keepIds.has(tensor.id)) {
                tensor.dispose();
            }
        });
        this.tensors.length = 0;
        this.idTensor.dispose();
    }
    /**
     * The size of the tensors in the tensor list.
     */
    size() {
        return this.tensors.length;
    }
    /**
     * Return a tensor that stacks a list of rank-R tf.Tensors into one rank-(R+1)
     * tf.Tensor.
     * @param elementShape shape of each tensor
     * @param elementDtype data type of each tensor
     * @param numElements the number of elements to stack
     */
    stack(elementShape, elementDtype, numElements = -1) {
        if (elementDtype !== this.elementDtype) {
            throw new Error(`Invalid data types; op elements ${elementDtype}, but list elements ${this.elementDtype}`);
        }
        if (numElements !== -1 && this.tensors.length !== numElements) {
            throw new Error(`Operation expected a list with ${numElements} elements but got a list with ${this.tensors.length} elements.`);
        }
        assertShapesMatchAllowUndefinedSize(elementShape, this.elementShape, 'TensorList shape mismatch: ');
        const outputElementShape = inferElementShape(this.elementShape, this.tensors, elementShape);
        return tidy(() => {
            const reshapedTensors = this.tensors.map(tensor => reshape(tensor, outputElementShape));
            return stack(reshapedTensors, 0);
        });
    }
    /**
     * Pop a tensor from the end of the list.
     * @param elementShape shape of the tensor
     * @param elementDtype data type of the tensor
     */
    popBack(elementShape, elementDtype) {
        if (elementDtype !== this.elementDtype) {
            throw new Error(`Invalid data types; op elements ${elementDtype}, but list elements ${this.elementDtype}`);
        }
        if (this.size() === 0) {
            throw new Error('Trying to pop from an empty list.');
        }
        const outputElementShape = inferElementShape(this.elementShape, this.tensors, elementShape);
        const tensor = this.tensors.pop();
        assertShapesMatchAllowUndefinedSize(tensor.shape, elementShape, 'TensorList shape mismatch: ');
        return reshape(tensor, outputElementShape);
    }
    /**
     * Push a tensor to the end of the list.
     * @param tensor Tensor to be pushed.
     */
    pushBack(tensor) {
        if (tensor.dtype !== this.elementDtype) {
            throw new Error(`Invalid data types; op elements ${tensor.dtype}, but list elements ${this.elementDtype}`);
        }
        assertShapesMatchAllowUndefinedSize(tensor.shape, this.elementShape, 'TensorList shape mismatch: ');
        if (this.maxNumElements === this.size()) {
            throw new Error(`Trying to push element into a full list.`);
        }
        keep(tensor);
        this.tensors.push(tensor);
    }
    /**
     * Update the size of the list.
     * @param size the new size of the list.
     */
    resize(size) {
        if (size < 0) {
            throw new Error(`TensorListResize expects size to be non-negative. Got: ${size}`);
        }
        if (this.maxNumElements !== -1 && size > this.maxNumElements) {
            throw new Error(`TensorListResize input size ${size} is greater maxNumElement ${this.maxNumElements}.`);
        }
        this.tensors.length = size;
    }
    /**
     * Retrieve the element at the provided index
     * @param elementShape shape of the tensor
     * @param elementDtype dtype of the tensor
     * @param elementIndex index of the tensor
     */
    getItem(elementIndex, elementShape, elementDtype) {
        if (elementDtype !== this.elementDtype) {
            throw new Error(`Invalid data types; op elements ${elementDtype}, but list elements ${this.elementDtype}`);
        }
        if (elementIndex < 0 || elementIndex > this.tensors.length) {
            throw new Error(`Trying to access element ${elementIndex} in a list with ${this.tensors.length} elements.`);
        }
        if (this.tensors[elementIndex] == null) {
            throw new Error(`element at index ${elementIndex} is null.`);
        }
        assertShapesMatchAllowUndefinedSize(this.tensors[elementIndex].shape, elementShape, 'TensorList shape mismatch: ');
        const outputElementShape = inferElementShape(this.elementShape, this.tensors, elementShape);
        return reshape(this.tensors[elementIndex], outputElementShape);
    }
    /**
     * Set the tensor at the index
     * @param elementIndex index of the tensor
     * @param tensor the tensor to be inserted into the list
     */
    setItem(elementIndex, tensor) {
        if (tensor.dtype !== this.elementDtype) {
            throw new Error(`Invalid data types; op elements ${tensor.dtype}, but list elements ${this.elementDtype}`);
        }
        if (elementIndex < 0 ||
            this.maxNumElements !== -1 && elementIndex >= this.maxNumElements) {
            throw new Error(`Trying to set element ${elementIndex} in a list with max ${this.maxNumElements} elements.`);
        }
        assertShapesMatchAllowUndefinedSize(this.elementShape, tensor.shape, 'TensorList shape mismatch: ');
        keep(tensor);
        this.tensors[elementIndex] = tensor;
    }
    /**
     * Return selected values in the TensorList as a stacked Tensor. All of
     * selected values must have been written and their shapes must all match.
     * @param indices indices of tensors to gather
     * @param elementDtype output tensor dtype
     * @param elementShape output tensor element shape
     */
    gather(indices, elementDtype, elementShape) {
        if (elementDtype !== this.elementDtype) {
            throw new Error(`Invalid data types; op elements ${elementDtype}, but list elements ${this.elementDtype}`);
        }
        assertShapesMatchAllowUndefinedSize(this.elementShape, elementShape, 'TensorList shape mismatch: ');
        // When indices is greater than the size of the list, indices beyond the
        // size of the list are ignored.
        indices = indices.slice(0, this.size());
        const outputElementShape = inferElementShape(this.elementShape, this.tensors, elementShape);
        if (indices.length === 0) {
            return tensor([], [0].concat(outputElementShape));
        }
        return tidy(() => {
            const tensors = indices.map(i => reshape(this.tensors[i], outputElementShape));
            return stack(tensors, 0);
        });
    }
    /**
     * Return the values in the TensorList as a concatenated Tensor.
     * @param elementDtype output tensor dtype
     * @param elementShape output tensor element shape
     */
    concat(elementDtype, elementShape) {
        if (!!elementDtype && elementDtype !== this.elementDtype) {
            throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${elementDtype}`);
        }
        assertShapesMatchAllowUndefinedSize(this.elementShape, elementShape, 'TensorList shape mismatch: ');
        const outputElementShape = inferElementShape(this.elementShape, this.tensors, elementShape);
        if (this.size() === 0) {
            return tensor([], [0].concat(outputElementShape));
        }
        return tidy(() => {
            const tensors = this.tensors.map(t => reshape(t, outputElementShape));
            return concat(tensors, 0);
        });
    }
}
/**
 * Creates a TensorList which, when stacked, has the value of tensor.
 * @param tensor from tensor
 * @param elementShape output tensor element shape
 */
export function fromTensor(tensor, elementShape, elementDtype) {
    const dtype = tensor.dtype;
    if (tensor.shape.length < 1) {
        throw new Error(`Tensor must be at least a vector, but saw shape: ${tensor.shape}`);
    }
    if (tensor.dtype !== elementDtype) {
        throw new Error(`Invalid data types; op elements ${tensor.dtype}, but list elements ${elementDtype}`);
    }
    const tensorElementShape = tensor.shape.slice(1);
    assertShapesMatchAllowUndefinedSize(tensorElementShape, elementShape, 'TensorList shape mismatch: ');
    const tensorList = unstack(tensor);
    return new TensorList(tensorList, elementShape, dtype);
}
/**
 * Return a TensorList of the given size with empty elements.
 * @param elementShape the shape of the future elements of the list
 * @param elementDtype the desired type of elements in the list
 * @param numElements the number of elements to reserve
 */
export function reserve(elementShape, elementDtype, numElements) {
    return new TensorList([], elementShape, elementDtype, numElements);
}
/**
 * Put tensors at specific indices of a stacked tensor into a TensorList.
 * @param indices list of indices on how to scatter the tensor.
 * @param tensor input tensor.
 * @param elementShape the shape of the future elements of the list
 * @param numElements the number of elements to scatter
 */
export function scatter(tensor, indices, elementShape, numElements) {
    if (indices.length !== tensor.shape[0]) {
        throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${indices.length} vs. ${tensor.shape[0]}`);
    }
    const maxIndex = Math.max(...indices);
    if (numElements != null && numElements !== -1 && maxIndex >= numElements) {
        throw new Error(`Max index must be < array size (${maxIndex}  vs. ${numElements})`);
    }
    const list = new TensorList([], elementShape, tensor.dtype, numElements);
    const tensors = unstack(tensor, 0);
    indices.forEach((value, index) => {
        list.setItem(value, tensors[index]);
    });
    return list;
}
/**
 * Split the values of a Tensor into a TensorList.
 * @param length the lengths to use when splitting value along
 *    its first dimension.
 * @param tensor the tensor to split.
 * @param elementShape the shape of the future elements of the list
 */
export function split(tensor, length, elementShape) {
    let totalLength = 0;
    const cumulativeLengths = length.map(len => {
        totalLength += len;
        return totalLength;
    });
    if (totalLength !== tensor.shape[0]) {
        throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${totalLength}, and tensor's shape is: ${tensor.shape}`);
    }
    const shapeWithoutFirstDim = tensor.shape.slice(1);
    const outputElementShape = mergeElementShape(shapeWithoutFirstDim, elementShape);
    const elementPerRow = totalLength === 0 ? 0 : tensor.size / totalLength;
    const tensors = tidy(() => {
        const tensors = [];
        tensor = reshape(tensor, [1, totalLength, elementPerRow]);
        for (let i = 0; i < length.length; ++i) {
            const previousLength = (i === 0) ? 0 : cumulativeLengths[i - 1];
            const indices = [0, previousLength, 0];
            const sizes = [1, length[i], elementPerRow];
            tensors[i] = reshape(slice(tensor, indices, sizes), outputElementShape);
        }
        tensor.dispose();
        return tensors;
    });
    const list = new TensorList([], elementShape, tensor.dtype, length.length);
    for (let i = 0; i < tensors.length; i++) {
        list.setItem(i, tensors[i]);
    }
    return list;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidGVuc29yX2xpc3QuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvbnZlcnRlci9zcmMvZXhlY3V0b3IvdGVuc29yX2xpc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBWSxJQUFJLEVBQUUsT0FBTyxFQUFFLE1BQU0sRUFBRSxLQUFLLEVBQUUsS0FBSyxFQUFVLE1BQU0sRUFBRSxJQUFJLEVBQUUsT0FBTyxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFM0gsT0FBTyxFQUFDLG1DQUFtQyxFQUFFLGlCQUFpQixFQUFFLGlCQUFpQixFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFFekc7Ozs7Ozs7Ozs7Ozs7R0FhRztBQUVILE1BQU0sT0FBTyxVQUFVO0lBT3JCOzs7Ozs7OztPQVFHO0lBQ0gsWUFDYSxPQUFpQixFQUFXLFlBQTZCLEVBQ3pELFlBQXNCLEVBQUUsY0FBYyxHQUFHLENBQUMsQ0FBQztRQUQzQyxZQUFPLEdBQVAsT0FBTyxDQUFVO1FBQVcsaUJBQVksR0FBWixZQUFZLENBQWlCO1FBQ3pELGlCQUFZLEdBQVosWUFBWSxDQUFVO1FBQ2pDLElBQUksT0FBTyxJQUFJLElBQUksRUFBRTtZQUNuQixPQUFPLENBQUMsT0FBTyxDQUFDLE1BQU0sQ0FBQyxFQUFFO2dCQUN2QixJQUFJLFlBQVksS0FBSyxNQUFNLENBQUMsS0FBSyxFQUFFO29CQUNqQyxNQUFNLElBQUksS0FBSyxDQUFDLG1DQUNaLFlBQVksdUJBQXVCLE1BQU0sQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDO2lCQUN4RDtnQkFDRCxtQ0FBbUMsQ0FDL0IsWUFBWSxFQUFFLE1BQU0sQ0FBQyxLQUFLLEVBQUUsNkJBQTZCLENBQUMsQ0FBQztnQkFFL0QsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQ2YsQ0FBQyxDQUFDLENBQUM7U0FDSjtRQUNELElBQUks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QyxDQUFDO1lBQzVDLE9BQU8sQ0FBQyxDQUFDLENBQUMsR0FBRyxPQUFPLENBQ2hCLEtBQUssQ0FBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssQ0FBQyxFQUFFLGtCQUE4QixDQUFDLENBQUM7U0FDcEU7UUFDRCxNQUFNLENBQUMsT0FBTyxFQUFFLENBQUM7UUFDakIsT0FBTyxPQUFPLENBQUM7SUFDakIsQ0FBQyxDQUFDLENBQUM7SUFFSCxNQUFNLElBQUksR0FBRyxJQUFJLFVBQVUsQ0FBQyxFQUFFLEVBQUUsWUFBWSxFQUFFLE1BQU0sQ0FBQyxLQUFLLEVBQUUsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBRTNFLEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFO1FBQ3ZDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQzdCO0lBQ0QsT0FBTyxJQUFJLENBQUM7QUFDZCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2NvbmNhdCwgRGF0YVR5cGUsIGtlZXAsIHJlc2hhcGUsIHNjYWxhciwgc2xpY2UsIHN0YWNrLCBUZW5zb3IsIHRlbnNvciwgdGlkeSwgdW5zdGFja30gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHthc3NlcnRTaGFwZXNNYXRjaEFsbG93VW5kZWZpbmVkU2l6ZSwgaW5mZXJFbGVtZW50U2hhcGUsIG1lcmdlRWxlbWVudFNoYXBlfSBmcm9tICcuL3RlbnNvcl91dGlscyc7XG5cbi8qKlxuICogVGVuc29yTGlzdCBzdG9yZXMgYSBjb250YWluZXIgb2YgYHRmLlRlbnNvcmAgb2JqZWN0cywgd2hpY2ggYXJlIGFjY2Vzc2libGVcbiAqIHZpYSB0ZW5zb3JzIGZpZWxkLlxuICpcbiAqIEluIG9yZGVyIHRvIGdldCBhIGNvcHkgb2YgdGhlIHVuZGVybHlpbmcgbGlzdCwgdXNlIHRoZSBjb3B5IG1ldGhvZDpcbiAqIGBgYFxuICogICAgVGVuc29yTGlzdCBiID0gYS5jb3B5KCk7XG4gKiAgICBiLnRlbnNvcnMoKS5wdXNoQmFjayh0KTsgIC8vIFRoaXMgZG9lcyBub3QgbW9kaWZ5IGEudGVuc29ycygpLlxuICogYGBgXG4gKlxuICogTm90ZSB0aGF0IHRoaXMgaXMgbm90IGEgZGVlcCBjb3B5OiB0aGUgbWVtb3J5IGxvY2F0aW9ucyBvZiB0aGUgdW5kZXJseWluZ1xuICogdGVuc29ycyB3aWxsIHN0aWxsIHBvaW50IHRvIHRoZSBzYW1lIGxvY2F0aW9ucyBvZiB0aGUgY29ycmVzcG9uZGluZyB0ZW5zb3JzXG4gKiBpbiB0aGUgb3JpZ2luYWwuXG4gKi9cblxuZXhwb3J0IGNsYXNzIFRlbnNvckxpc3Qge1xuICByZWFkb25seSBpZFRlbnNvcjogVGVuc29yO1xuICBtYXhOdW1FbGVtZW50czogbnVtYmVyO1xuXG4gIGdldCBpZCgpIHtcbiAgICByZXR1cm4gdGhpcy5pZFRlbnNvci5pZDtcbiAgfVxuICAvKipcbiAgICpcbiAgICogQHBhcmFtIHRlbnNvcnMgbGlzdCBvZiB0ZW5zb3JzXG4gICAqIEBwYXJhbSBlbGVtZW50U2hhcGUgc2hhcGUgb2YgZWFjaCB0ZW5zb3IsIHRoaXMgY2FuIGJlIGEgc2luZ2xlIG51bWJlciAoYW55XG4gICAqIHNoYXBlIGlzIGFsbG93ZWQpIG9yIHBhcnRpYWwgc2hhcGUgKGRpbSA9IC0xKS5cbiAgICogQHBhcmFtIGVsZW1lbnREdHlwZSBkYXRhIHR5cGUgb2YgZWFjaCB0ZW5zb3JcbiAgICogQHBhcmFtIG1heE51bUVsZW1lbnRzIFRoZSBtYXhpbXVtIGFsbG93ZWQgc2l6ZSBvZiBgdGVuc29yc2AuIERlZmF1bHRzIHRvIC0xXG4gICAqICAgbWVhbmluZyB0aGF0IHRoZSBzaXplIG9mIGB0ZW5zb3JzYCBpcyB1bmJvdW5kZWQuXG4gICAqL1xuICBjb25zdHJ1Y3RvcihcbiAgICAgIHJlYWRvbmx5IHRlbnNvcnM6IFRlbnNvcltdLCByZWFkb25seSBlbGVtZW50U2hhcGU6IG51bWJlcnxudW1iZXJbXSxcbiAgICAgIHJlYWRvbmx5IGVsZW1lbnREdHlwZTogRGF0YVR5cGUsIG1heE51bUVsZW1lbnRzID0gLTEpIHtcbiAgICBpZiAodGVuc29ycyAhPSBudWxsKSB7XG4gICAgICB0ZW5zb3JzLmZvckVhY2godGVuc29yID0+IHtcbiAgICAgICAgaWYgKGVsZW1lbnREdHlwZSAhPT0gdGVuc29yLmR0eXBlKSB7XG4gICAgICAgICAgdGhyb3cgbmV3IEVycm9yKGBJbnZhbGlkIGRhdGEgdHlwZXM7IG9wIGVsZW1lbnRzICR7XG4gICAgICAgICAgICAgIGVsZW1lbnREdHlwZX0sIGJ1dCBsaXN0IGVsZW1lbnRzICR7dGVuc29yLmR0eXBlfWApO1xuICAgICAgICB9XG4gICAgICAgIGFzc2VydFNoYXBlc01hdGNoQWxsb3dVbmRlZmluZWRTaXplKFxuICAgICAgICAgICAgZWxlbWVudFNoYXBlLCB0ZW5zb3Iuc2hhcGUsICdUZW5zb3JMaXN0IHNoYXBlIG1pc21hdGNoOiAnKTtcblxuICAgICAgICBrZWVwKHRlbnNvcik7XG4gICAgICB9KTtcbiAgICB9XG4gICAgdGhpcy5pZFRlbnNvciA9IHNjYWxhcigwKTtcbiAgICB0aGlzLm1heE51bUVsZW1lbnRzID0gbWF4TnVtRWxlbWVudHM7XG4gICAga2VlcCh0aGlzLmlkVGVuc29yKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBHZXQgYSBuZXcgVGVuc29yTGlzdCBjb250YWluaW5nIGEgY29weSBvZiB0aGUgdW5kZXJseWluZyB0ZW5zb3IgY29udGFpbmVyLlxuICAgKi9cbiAgY29weSgpOiBUZW5zb3JMaXN0IHtcbiAgICByZXR1cm4gbmV3IFRlbnNvckxpc3QoXG4gICAgICAgIFsuLi50aGlzLnRlbnNvcnNdLCB0aGlzLmVsZW1lbnRTaGFwZSwgdGhpcy5lbGVtZW50RHR5cGUpO1xuICB9XG5cbiAgLyoqXG4gICAqIERpc3Bvc2UgdGhlIHRlbnNvcnMgYW5kIGlkVGVuc29yIGFuZCBjbGVhciB0aGUgdGVuc29yIGxpc3QuXG4gICAqL1xuICBjbGVhckFuZENsb3NlKGtlZXBJZHM/OiBTZXQ8bnVtYmVyPikge1xuICAgIHRoaXMudGVuc29ycy5mb3JFYWNoKHRlbnNvciA9PiB7XG4gICAgICBpZiAoa2VlcElkcyA9PSBudWxsIHx8ICFrZWVwSWRzLmhhcyh0ZW5zb3IuaWQpKSB7XG4gICAgICAgIHRlbnNvci5kaXNwb3NlKCk7XG4gICAgICB9XG4gICAgfSk7XG4gICAgdGhpcy50ZW5zb3JzLmxlbmd0aCA9IDA7XG4gICAgdGhpcy5pZFRlbnNvci5kaXNwb3NlKCk7XG4gIH1cbiAgLyoqXG4gICAqIFRoZSBzaXplIG9mIHRoZSB0ZW5zb3JzIGluIHRoZSB0ZW5zb3IgbGlzdC5cbiAgICovXG4gIHNpemUoKSB7XG4gICAgcmV0dXJuIHRoaXMudGVuc29ycy5sZW5ndGg7XG4gIH1cblxuICAvKipcbiAgICogUmV0dXJuIGEgdGVuc29yIHRoYXQgc3RhY2tzIGEgbGlzdCBvZiByYW5rLVIgdGYuVGVuc29ycyBpbnRvIG9uZSByYW5rLShSKzEpXG4gICAqIHRmLlRlbnNvci5cbiAgICogQHBhcmFtIGVsZW1lbnRTaGFwZSBzaGFwZSBvZiBlYWNoIHRlbnNvclxuICAgKiBAcGFyYW0gZWxlbWVudER0eXBlIGRhdGEgdHlwZSBvZiBlYWNoIHRlbnNvclxuICAgKiBAcGFyYW0gbnVtRWxlbWVudHMgdGhlIG51bWJlciBvZiBlbGVtZW50cyB0byBzdGFja1xuICAgKi9cbiAgc3RhY2soZWxlbWVudFNoYXBlOiBudW1iZXJbXSwgZWxlbWVudER0eXBlOiBEYXRhVHlwZSwgbnVtRWxlbWVudHMgPSAtMSk6XG4gICAgICBUZW5zb3Ige1xuICAgIGlmIChlbGVtZW50RHR5cGUgIT09IHRoaXMuZWxlbWVudER0eXBlKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYEludmFsaWQgZGF0YSB0eXBlczsgb3AgZWxlbWVudHMgJHtcbiAgICAgICAgICBlbGVtZW50RHR5cGV9LCBidXQgbGlzdCBlbGVtZW50cyAke3RoaXMuZWxlbWVudER0eXBlfWApO1xuICAgIH1cbiAgICBpZiAobnVtRWxlbWVudHMgIT09IC0xICYmIHRoaXMudGVuc29ycy5sZW5ndGggIT09IG51bUVsZW1lbnRzKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYE9wZXJhdGlvbiBleHBlY3RlZCBhIGxpc3Qgd2l0aCAke1xuICAgICAgICAgIG51bUVsZW1lbnRzfSBlbGVtZW50cyBidXQgZ290IGEgbGlzdCB3aXRoICR7XG4gICAgICAgICAgdGhpcy50ZW5zb3JzLmxlbmd0aH0gZWxlbWVudHMuYCk7XG4gICAgfVxuICAgIGFzc2VydFNoYXBlc01hdGNoQWxsb3dVbmRlZmluZWRTaXplKFxuICAgICAgICBlbGVtZW50U2hhcGUsIHRoaXMuZWxlbWVudFNoYXBlLCAnVGVuc29yTGlzdCBzaGFwZSBtaXNtYXRjaDogJyk7XG4gICAgY29uc3Qgb3V0cHV0RWxlbWVudFNoYXBlID1cbiAgICAgICAgaW5mZXJFbGVtZW50U2hhcGUodGhpcy5lbGVtZW50U2hhcGUsIHRoaXMudGVuc29ycywgZWxlbWVudFNoYXBlKTtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBjb25zdCByZXNoYXBlZFRlbnNvcnMgPVxuICAgICAgICAgIHRoaXMudGVuc29ycy5tYXAodGVuc29yID0+IHJlc2hhcGUodGVuc29yLCBvdXRwdXRFbGVtZW50U2hhcGUpKTtcbiAgICAgIHJldHVybiBzdGFjayhyZXNoYXBlZFRlbnNvcnMsIDApO1xuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIFBvcCBhIHRlbnNvciBmcm9tIHRoZSBlbmQgb2YgdGhlIGxpc3QuXG4gICAqIEBwYXJhbSBlbGVtZW50U2hhcGUgc2hhcGUgb2YgdGhlIHRlbnNvclxuICAgKiBAcGFyYW0gZWxlbWVudER0eXBlIGRhdGEgdHlwZSBvZiB0aGUgdGVuc29yXG4gICAqL1xuICBwb3BCYWNrKGVsZW1lbnRTaGFwZTogbnVtYmVyW10sIGVsZW1lbnREdHlwZTogRGF0YVR5cGUpOiBUZW5zb3Ige1xuICAgIGlmIChlbGVtZW50RHR5cGUgIT09IHRoaXMuZWxlbWVudER0eXBlKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYEludmFsaWQgZGF0YSB0eXBlczsgb3AgZWxlbWVudHMgJHtcbiAgICAgICAgICBlbGVtZW50RHR5cGV9LCBidXQgbGlzdCBlbGVtZW50cyAke3RoaXMuZWxlbWVudER0eXBlfWApO1xuICAgIH1cblxuICAgIGlmICh0aGlzLnNpemUoKSA9PT0gMCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdUcnlpbmcgdG8gcG9wIGZyb20gYW4gZW1wdHkgbGlzdC4nKTtcbiAgICB9XG4gICAgY29uc3Qgb3V0cHV0RWxlbWVudFNoYXBlID1cbiAgICAgICAgaW5mZXJFbGVtZW50U2hhcGUodGhpcy5lbGVtZW50U2hhcGUsIHRoaXMudGVuc29ycywgZWxlbWVudFNoYXBlKTtcbiAgICBjb25zdCB0ZW5zb3IgPSB0aGlzLnRlbnNvcnMucG9wKCk7XG5cbiAgICBhc3NlcnRTaGFwZXNNYXRjaEFsbG93VW5kZWZpbmVkU2l6ZShcbiAgICAgICAgdGVuc29yLnNoYXBlLCBlbGVtZW50U2hhcGUsICdUZW5zb3JMaXN0IHNoYXBlIG1pc21hdGNoOiAnKTtcblxuICAgIHJldHVybiByZXNoYXBlKHRlbnNvciwgb3V0cHV0RWxlbWVudFNoYXBlKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBQdXNoIGEgdGVuc29yIHRvIHRoZSBlbmQgb2YgdGhlIGxpc3QuXG4gICAqIEBwYXJhbSB0ZW5zb3IgVGVuc29yIHRvIGJlIHB1c2hlZC5cbiAgICovXG4gIHB1c2hCYWNrKHRlbnNvcjogVGVuc29yKSB7XG4gICAgaWYgKHRlbnNvci5kdHlwZSAhPT0gdGhpcy5lbGVtZW50RHR5cGUpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgSW52YWxpZCBkYXRhIHR5cGVzOyBvcCBlbGVtZW50cyAke1xuICAgICAgICAgIHRlbnNvci5kdHlwZX0sIGJ1dCBsaXN0IGVsZW1lbnRzICR7dGhpcy5lbGVtZW50RHR5cGV9YCk7XG4gICAgfVxuXG4gICAgYXNzZXJ0U2hhcGVzTWF0Y2hBbGxvd1VuZGVmaW5lZFNpemUoXG4gICAgICAgIHRlbnNvci5zaGFwZSwgdGhpcy5lbGVtZW50U2hhcGUsICdUZW5zb3JMaXN0IHNoYXBlIG1pc21hdGNoOiAnKTtcblxuICAgIGlmICh0aGlzLm1heE51bUVsZW1lbnRzID09PSB0aGlzLnNpemUoKSkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBUcnlpbmcgdG8gcHVzaCBlbGVtZW50IGludG8gYSBmdWxsIGxpc3QuYCk7XG4gICAgfVxuICAgIGtlZXAodGVuc29yKTtcbiAgICB0aGlzLnRlbnNvcnMucHVzaCh0ZW5zb3IpO1xuICB9XG5cbiAgLyoqXG4gICAqIFVwZGF0ZSB0aGUgc2l6ZSBvZiB0aGUgbGlzdC5cbiAgICogQHBhcmFtIHNpemUgdGhlIG5ldyBzaXplIG9mIHRoZSBsaXN0LlxuICAgKi9cbiAgcmVzaXplKHNpemU6IG51bWJlcikge1xuICAgIGlmIChzaXplIDwgMCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBUZW5zb3JMaXN0UmVzaXplIGV4cGVjdHMgc2l6ZSB0byBiZSBub24tbmVnYXRpdmUuIEdvdDogJHtzaXplfWApO1xuICAgIH1cblxuICAgIGlmICh0aGlzLm1heE51bUVsZW1lbnRzICE9PSAtMSAmJiBzaXplID4gdGhpcy5tYXhOdW1FbGVtZW50cykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKGBUZW5zb3JMaXN0UmVzaXplIGlucHV0IHNpemUgJHtcbiAgICAgICAgICBzaXplfSBpcyBncmVhdGVyIG1heE51bUVsZW1lbnQgJHt0aGlzLm1heE51bUVsZW1lbnRzfS5gKTtcbiAgICB9XG4gICAgdGhpcy50ZW5zb3JzLmxlbmd0aCA9IHNpemU7XG4gIH1cblxuICAvKipcbiAgICogUmV0cmlldmUgdGhlIGVsZW1lbnQgYXQgdGhlIHByb3ZpZGVkIGluZGV4XG4gICAqIEBwYXJhbSBlbGVtZW50U2hhcGUgc2hhcGUgb2YgdGhlIHRlbnNvclxuICAgKiBAcGFyYW0gZWxlbWVudER0eXBlIGR0eXBlIG9mIHRoZSB0ZW5zb3JcbiAgICogQHBhcmFtIGVsZW1lbnRJbmRleCBpbmRleCBvZiB0aGUgdGVuc29yXG4gICAqL1xuICBnZXRJdGVtKGVsZW1lbnRJbmRleDogbnVtYmVyLCBlbGVtZW50U2hhcGU6IG51bWJlcltdLCBlbGVtZW50RHR5cGU6IERhdGFUeXBlKTpcbiAgICAgIFRlbnNvciB7XG4gICAgaWYgKGVsZW1lbnREdHlwZSAhPT0gdGhpcy5lbGVtZW50RHR5cGUpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgSW52YWxpZCBkYXRhIHR5cGVzOyBvcCBlbGVtZW50cyAke1xuICAgICAgICAgIGVsZW1lbnREdHlwZX0sIGJ1dCBsaXN0IGVsZW1lbnRzICR7dGhpcy5lbGVtZW50RHR5cGV9YCk7XG4gICAgfVxuICAgIGlmIChlbGVtZW50SW5kZXggPCAwIHx8IGVsZW1lbnRJbmRleCA+IHRoaXMudGVuc29ycy5sZW5ndGgpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVHJ5aW5nIHRvIGFjY2VzcyBlbGVtZW50ICR7XG4gICAgICAgICAgZWxlbWVudEluZGV4fSBpbiBhIGxpc3Qgd2l0aCAke3RoaXMudGVuc29ycy5sZW5ndGh9IGVsZW1lbnRzLmApO1xuICAgIH1cblxuICAgIGlmICh0aGlzLnRlbnNvcnNbZWxlbWVudEluZGV4XSA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoYGVsZW1lbnQgYXQgaW5kZXggJHtlbGVtZW50SW5kZXh9IGlzIG51bGwuYCk7XG4gICAgfVxuXG4gICAgYXNzZXJ0U2hhcGVzTWF0Y2hBbGxvd1VuZGVmaW5lZFNpemUoXG4gICAgICAgIHRoaXMudGVuc29yc1tlbGVtZW50SW5kZXhdLnNoYXBlLCBlbGVtZW50U2hhcGUsXG4gICAgICAgICdUZW5zb3JMaXN0IHNoYXBlIG1pc21hdGNoOiAnKTtcbiAgICBjb25zdCBvdXRwdXRFbGVtZW50U2hhcGUgPVxuICAgICAgICBpbmZlckVsZW1lbnRTaGFwZSh0aGlzLmVsZW1lbnRTaGFwZSwgdGhpcy50ZW5zb3JzLCBlbGVtZW50U2hhcGUpO1xuICAgIHJldHVybiByZXNoYXBlKHRoaXMudGVuc29yc1tlbGVtZW50SW5kZXhdLCBvdXRwdXRFbGVtZW50U2hhcGUpO1xuICB9XG5cbiAgLyoqXG4gICAqIFNldCB0aGUgdGVuc29yIGF0IHRoZSBpbmRleFxuICAgKiBAcGFyYW0gZWxlbWVudEluZGV4IGluZGV4IG9mIHRoZSB0ZW5zb3JcbiAgICogQHBhcmFtIHRlbnNvciB0aGUgdGVuc29yIHRvIGJlIGluc2VydGVkIGludG8gdGhlIGxpc3RcbiAgICovXG4gIHNldEl0ZW0oZWxlbWVudEluZGV4OiBudW1iZXIsIHRlbnNvcjogVGVuc29yKSB7XG4gICAgaWYgKHRlbnNvci5kdHlwZSAhPT0gdGhpcy5lbGVtZW50RHR5cGUpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgSW52YWxpZCBkYXRhIHR5cGVzOyBvcCBlbGVtZW50cyAke1xuICAgICAgICAgIHRlbnNvci5kdHlwZX0sIGJ1dCBsaXN0IGVsZW1lbnRzICR7dGhpcy5lbGVtZW50RHR5cGV9YCk7XG4gICAgfVxuXG4gICAgaWYgKGVsZW1lbnRJbmRleCA8IDAgfHxcbiAgICAgICAgdGhpcy5tYXhOdW1FbGVtZW50cyAhPT0gLTEgJiYgZWxlbWVudEluZGV4ID49IHRoaXMubWF4TnVtRWxlbWVudHMpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVHJ5aW5nIHRvIHNldCBlbGVtZW50ICR7XG4gICAgICAgICAgZWxlbWVudEluZGV4fSBpbiBhIGxpc3Qgd2l0aCBtYXggJHt0aGlzLm1heE51bUVsZW1lbnRzfSBlbGVtZW50cy5gKTtcbiAgICB9XG5cbiAgICBhc3NlcnRTaGFwZXNNYXRjaEFsbG93VW5kZWZpbmVkU2l6ZShcbiAgICAgICAgdGhpcy5lbGVtZW50U2hhcGUsIHRlbnNvci5zaGFwZSwgJ1RlbnNvckxpc3Qgc2hhcGUgbWlzbWF0Y2g6ICcpO1xuICAgIGtlZXAodGVuc29yKTtcbiAgICB0aGlzLnRlbnNvcnNbZWxlbWVudEluZGV4XSA9IHRlbnNvcjtcbiAgfVxuXG4gIC8qKlxuICAgKiBSZXR1cm4gc2VsZWN0ZWQgdmFsdWVzIGluIHRoZSBUZW5zb3JMaXN0IGFzIGEgc3RhY2tlZCBUZW5zb3IuIEFsbCBvZlxuICAgKiBzZWxlY3RlZCB2YWx1ZXMgbXVzdCBoYXZlIGJlZW4gd3JpdHRlbiBhbmQgdGhlaXIgc2hhcGVzIG11c3QgYWxsIG1hdGNoLlxuICAgKiBAcGFyYW0gaW5kaWNlcyBpbmRpY2VzIG9mIHRlbnNvcnMgdG8gZ2F0aGVyXG4gICAqIEBwYXJhbSBlbGVtZW50RHR5cGUgb3V0cHV0IHRlbnNvciBkdHlwZVxuICAgKiBAcGFyYW0gZWxlbWVudFNoYXBlIG91dHB1dCB0ZW5zb3IgZWxlbWVudCBzaGFwZVxuICAgKi9cbiAgZ2F0aGVyKGluZGljZXM6IG51bWJlcltdLCBlbGVtZW50RHR5cGU6IERhdGFUeXBlLCBlbGVtZW50U2hhcGU6IG51bWJlcltdKTpcbiAgICAgIFRlbnNvciB7XG4gICAgaWYgKGVsZW1lbnREdHlwZSAhPT0gdGhpcy5lbGVtZW50RHR5cGUpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgSW52YWxpZCBkYXRhIHR5cGVzOyBvcCBlbGVtZW50cyAke1xuICAgICAgICAgIGVsZW1lbnREdHlwZX0sIGJ1dCBsaXN0IGVsZW1lbnRzICR7dGhpcy5lbGVtZW50RHR5cGV9YCk7XG4gICAgfVxuXG4gICAgYXNzZXJ0U2hhcGVzTWF0Y2hBbGxvd1VuZGVmaW5lZFNpemUoXG4gICAgICAgIHRoaXMuZWxlbWVudFNoYXBlLCBlbGVtZW50U2hhcGUsICdUZW5zb3JMaXN0IHNoYXBlIG1pc21hdGNoOiAnKTtcblxuICAgIC8vIFdoZW4gaW5kaWNlcyBpcyBncmVhdGVyIHRoYW4gdGhlIHNpemUgb2YgdGhlIGxpc3QsIGluZGljZXMgYmV5b25kIHRoZVxuICAgIC8vIHNpemUgb2YgdGhlIGxpc3QgYXJlIGlnbm9yZWQuXG4gICAgaW5kaWNlcyA9IGluZGljZXMuc2xpY2UoMCwgdGhpcy5zaXplKCkpO1xuICAgIGNvbnN0IG91dHB1dEVsZW1lbnRTaGFwZSA9XG4gICAgICAgIGluZmVyRWxlbWVudFNoYXBlKHRoaXMuZWxlbWVudFNoYXBlLCB0aGlzLnRlbnNvcnMsIGVsZW1lbnRTaGFwZSk7XG4gICAgaWYgKGluZGljZXMubGVuZ3RoID09PSAwKSB7XG4gICAgICByZXR1cm4gdGVuc29yKFtdLCBbMF0uY29uY2F0KG91dHB1dEVsZW1lbnRTaGFwZSkpO1xuICAgIH1cblxuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IHRlbnNvcnMgPVxuICAgICAgICAgIGluZGljZXMubWFwKGkgPT4gcmVzaGFwZSh0aGlzLnRlbnNvcnNbaV0sIG91dHB1dEVsZW1lbnRTaGFwZSkpO1xuICAgICAgcmV0dXJuIHN0YWNrKHRlbnNvcnMsIDApO1xuICAgIH0pO1xuICB9XG5cbiAgLyoqXG4gICAqIFJldHVybiB0aGUgdmFsdWVzIGluIHRoZSBUZW5zb3JMaXN0IGFzIGEgY29uY2F0ZW5hdGVkIFRlbnNvci5cbiAgICogQHBhcmFtIGVsZW1lbnREdHlwZSBvdXRwdXQgdGVuc29yIGR0eXBlXG4gICAqIEBwYXJhbSBlbGVtZW50U2hhcGUgb3V0cHV0IHRlbnNvciBlbGVtZW50IHNoYXBlXG4gICAqL1xuICBjb25jYXQoZWxlbWVudER0eXBlOiBEYXRhVHlwZSwgZWxlbWVudFNoYXBlOiBudW1iZXJbXSk6IFRlbnNvciB7XG4gICAgaWYgKCEhZWxlbWVudER0eXBlICYmIGVsZW1lbnREdHlwZSAhPT0gdGhpcy5lbGVtZW50RHR5cGUpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcihgVGVuc29yTGlzdCBkdHlwZSBpcyAke1xuICAgICAgICAgIHRoaXMuZWxlbWVudER0eXBlfSBidXQgY29uY2F0IHJlcXVlc3RlZCBkdHlwZSAke2VsZW1lbnREdHlwZX1gKTtcbiAgICB9XG5cbiAgICBhc3NlcnRTaGFwZXNNYXRjaEFsbG93VW5kZWZpbmVkU2l6ZShcbiAgICAgICAgdGhpcy5lbGVtZW50U2hhcGUsIGVsZW1lbnRTaGFwZSwgJ1RlbnNvckxpc3Qgc2hhcGUgbWlzbWF0Y2g6ICcpO1xuICAgIGNvbnN0IG91dHB1dEVsZW1lbnRTaGFwZSA9XG4gICAgICAgIGluZmVyRWxlbWVudFNoYXBlKHRoaXMuZWxlbWVudFNoYXBlLCB0aGlzLnRlbnNvcnMsIGVsZW1lbnRTaGFwZSk7XG5cbiAgICBpZiAodGhpcy5zaXplKCkgPT09IDApIHtcbiAgICAgIHJldHVybiB0ZW5zb3IoW10sIFswXS5jb25jYXQob3V0cHV0RWxlbWVudFNoYXBlKSk7XG4gICAgfVxuICAgIHJldHVybiB0aWR5KCgpID0+IHtcbiAgICAgIGNvbnN0IHRlbnNvcnMgPSB0aGlzLnRlbnNvcnMubWFwKHQgPT4gcmVzaGFwZSh0LCBvdXRwdXRFbGVtZW50U2hhcGUpKTtcbiAgICAgIHJldHVybiBjb25jYXQodGVuc29ycywgMCk7XG4gICAgfSk7XG4gIH1cbn1cblxuLyoqXG4gKiBDcmVhdGVzIGEgVGVuc29yTGlzdCB3aGljaCwgd2hlbiBzdGFja2VkLCBoYXMgdGhlIHZhbHVlIG9mIHRlbnNvci5cbiAqIEBwYXJhbSB0ZW5zb3IgZnJvbSB0ZW5zb3JcbiAqIEBwYXJhbSBlbGVtZW50U2hhcGUgb3V0cHV0IHRlbnNvciBlbGVtZW50IHNoYXBlXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBmcm9tVGVuc29yKFxuICAgIHRlbnNvcjogVGVuc29yLCBlbGVtZW50U2hhcGU6IG51bWJlcltdLCBlbGVtZW50RHR5cGU6IERhdGFUeXBlKSB7XG4gIGNvbnN0IGR0eXBlID0gdGVuc29yLmR0eXBlO1xuICBpZiAodGVuc29yLnNoYXBlLmxlbmd0aCA8IDEpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgIGBUZW5zb3IgbXVzdCBiZSBhdCBsZWFzdCBhIHZlY3RvciwgYnV0IHNhdyBzaGFwZTogJHt0ZW5zb3Iuc2hhcGV9YCk7XG4gIH1cbiAgaWYgKHRlbnNvci5kdHlwZSAhPT0gZWxlbWVudER0eXBlKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBJbnZhbGlkIGRhdGEgdHlwZXM7IG9wIGVsZW1lbnRzICR7XG4gICAgICAgIHRlbnNvci5kdHlwZX0sIGJ1dCBsaXN0IGVsZW1lbnRzICR7ZWxlbWVudER0eXBlfWApO1xuICB9XG4gIGNvbnN0IHRlbnNvckVsZW1lbnRTaGFwZSA9IHRlbnNvci5zaGFwZS5zbGljZSgxKTtcbiAgYXNzZXJ0U2hhcGVzTWF0Y2hBbGxvd1VuZGVmaW5lZFNpemUoXG4gICAgICB0ZW5zb3JFbGVtZW50U2hhcGUsIGVsZW1lbnRTaGFwZSwgJ1RlbnNvckxpc3Qgc2hhcGUgbWlzbWF0Y2g6ICcpO1xuICBjb25zdCB0ZW5zb3JMaXN0OiBUZW5zb3JbXSA9IHVuc3RhY2sodGVuc29yKTtcbiAgcmV0dXJuIG5ldyBUZW5zb3JMaXN0KHRlbnNvckxpc3QsIGVsZW1lbnRTaGFwZSwgZHR5cGUpO1xufVxuXG4vKipcbiAqIFJldHVybiBhIFRlbnNvckxpc3Qgb2YgdGhlIGdpdmVuIHNpemUgd2l0aCBlbXB0eSBlbGVtZW50cy5cbiAqIEBwYXJhbSBlbGVtZW50U2hhcGUgdGhlIHNoYXBlIG9mIHRoZSBmdXR1cmUgZWxlbWVudHMgb2YgdGhlIGxpc3RcbiAqIEBwYXJhbSBlbGVtZW50RHR5cGUgdGhlIGRlc2lyZWQgdHlwZSBvZiBlbGVtZW50cyBpbiB0aGUgbGlzdFxuICogQHBhcmFtIG51bUVsZW1lbnRzIHRoZSBudW1iZXIgb2YgZWxlbWVudHMgdG8gcmVzZXJ2ZVxuICovXG5leHBvcnQgZnVuY3Rpb24gcmVzZXJ2ZShcbiAgICBlbGVtZW50U2hhcGU6IG51bWJlcltdLCBlbGVtZW50RHR5cGU6IERhdGFUeXBlLCBudW1FbGVtZW50czogbnVtYmVyKSB7XG4gIHJldHVybiBuZXcgVGVuc29yTGlzdChbXSwgZWxlbWVudFNoYXBlLCBlbGVtZW50RHR5cGUsIG51bUVsZW1lbnRzKTtcbn1cblxuLyoqXG4gKiBQdXQgdGVuc29ycyBhdCBzcGVjaWZpYyBpbmRpY2VzIG9mIGEgc3RhY2tlZCB0ZW5zb3IgaW50byBhIFRlbnNvckxpc3QuXG4gKiBAcGFyYW0gaW5kaWNlcyBsaXN0IG9mIGluZGljZXMgb24gaG93IHRvIHNjYXR0ZXIgdGhlIHRlbnNvci5cbiAqIEBwYXJhbSB0ZW5zb3IgaW5wdXQgdGVuc29yLlxuICogQHBhcmFtIGVsZW1lbnRTaGFwZSB0aGUgc2hhcGUgb2YgdGhlIGZ1dHVyZSBlbGVtZW50cyBvZiB0aGUgbGlzdFxuICogQHBhcmFtIG51bUVsZW1lbnRzIHRoZSBudW1iZXIgb2YgZWxlbWVudHMgdG8gc2NhdHRlclxuICovXG5leHBvcnQgZnVuY3Rpb24gc2NhdHRlcihcbiAgICB0ZW5zb3I6IFRlbnNvciwgaW5kaWNlczogbnVtYmVyW10sIGVsZW1lbnRTaGFwZTogbnVtYmVyW10sXG4gICAgbnVtRWxlbWVudHM/OiBudW1iZXIpOiBUZW5zb3JMaXN0IHtcbiAgaWYgKGluZGljZXMubGVuZ3RoICE9PSB0ZW5zb3Iuc2hhcGVbMF0pIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYEV4cGVjdGVkIGxlbihpbmRpY2VzKSA9PSB0ZW5zb3Iuc2hhcGVbMF0sIGJ1dCBzYXc6ICR7XG4gICAgICAgIGluZGljZXMubGVuZ3RofSB2cy4gJHt0ZW5zb3Iuc2hhcGVbMF19YCk7XG4gIH1cblxuICBjb25zdCBtYXhJbmRleCA9IE1hdGgubWF4KC4uLmluZGljZXMpO1xuXG4gIGlmIChudW1FbGVtZW50cyAhPSBudWxsICYmIG51bUVsZW1lbnRzICE9PSAtMSAmJiBtYXhJbmRleCA+PSBudW1FbGVtZW50cykge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYE1heCBpbmRleCBtdXN0IGJlIDwgYXJyYXkgc2l6ZSAoJHttYXhJbmRleH0gIHZzLiAke251bUVsZW1lbnRzfSlgKTtcbiAgfVxuXG4gIGNvbnN0IGxpc3QgPSBuZXcgVGVuc29yTGlzdChbXSwgZWxlbWVudFNoYXBlLCB0ZW5zb3IuZHR5cGUsIG51bUVsZW1lbnRzKTtcbiAgY29uc3QgdGVuc29ycyA9IHVuc3RhY2sodGVuc29yLCAwKTtcbiAgaW5kaWNlcy5mb3JFYWNoKCh2YWx1ZSwgaW5kZXgpID0+IHtcbiAgICBsaXN0LnNldEl0ZW0odmFsdWUsIHRlbnNvcnNbaW5kZXhdKTtcbiAgfSk7XG4gIHJldHVybiBsaXN0O1xufVxuXG4vKipcbiAqIFNwbGl0IHRoZSB2YWx1ZXMgb2YgYSBUZW5zb3IgaW50byBhIFRlbnNvckxpc3QuXG4gKiBAcGFyYW0gbGVuZ3RoIHRoZSBsZW5ndGhzIHRvIHVzZSB3aGVuIHNwbGl0dGluZyB2YWx1ZSBhbG9uZ1xuICogICAgaXRzIGZpcnN0IGRpbWVuc2lvbi5cbiAqIEBwYXJhbSB0ZW5zb3IgdGhlIHRlbnNvciB0byBzcGxpdC5cbiAqIEBwYXJhbSBlbGVtZW50U2hhcGUgdGhlIHNoYXBlIG9mIHRoZSBmdXR1cmUgZWxlbWVudHMgb2YgdGhlIGxpc3RcbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHNwbGl0KFxuICAgIHRlbnNvcjogVGVuc29yLCBsZW5ndGg6IG51bWJlcltdLCBlbGVtZW50U2hhcGU6IG51bWJlcltdKSB7XG4gIGxldCB0b3RhbExlbmd0aCA9IDA7XG4gIGNvbnN0IGN1bXVsYXRpdmVMZW5ndGhzID0gbGVuZ3RoLm1hcChsZW4gPT4ge1xuICAgIHRvdGFsTGVuZ3RoICs9IGxlbjtcbiAgICByZXR1cm4gdG90YWxMZW5ndGg7XG4gIH0pO1xuXG4gIGlmICh0b3RhbExlbmd0aCAhPT0gdGVuc29yLnNoYXBlWzBdKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGBFeHBlY3RlZCBzdW0gb2YgbGVuZ3RocyB0byBiZSBlcXVhbCB0b1xuICAgICAgICAgIHRlbnNvci5zaGFwZVswXSwgYnV0IHN1bSBvZiBsZW5ndGhzIGlzXG4gICAgICAgICR7dG90YWxMZW5ndGh9LCBhbmQgdGVuc29yJ3Mgc2hhcGUgaXM6ICR7dGVuc29yLnNoYXBlfWApO1xuICB9XG5cbiAgY29uc3Qgc2hhcGVXaXRob3V0Rmlyc3REaW0gPSB0ZW5zb3Iuc2hhcGUuc2xpY2UoMSk7XG4gIGNvbnN0IG91dHB1dEVsZW1lbnRTaGFwZSA9XG4gICAgICBtZXJnZUVsZW1lbnRTaGFwZShzaGFwZVdpdGhvdXRGaXJzdERpbSwgZWxlbWVudFNoYXBlKTtcbiAgY29uc3QgZWxlbWVudFBlclJvdyA9IHRvdGFsTGVuZ3RoID09PSAwID8gMCA6IHRlbnNvci5zaXplIC8gdG90YWxMZW5ndGg7XG4gIGNvbnN0IHRlbnNvcnM6IFRlbnNvcltdID0gdGlkeSgoKSA9PiB7XG4gICAgY29uc3QgdGVuc29ycyA9IFtdO1xuICAgIHRlbnNvciA9IHJlc2hhcGUodGVuc29yLCBbMSwgdG90YWxMZW5ndGgsIGVsZW1lbnRQZXJSb3ddKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGxlbmd0aC5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgcHJldmlvdXNMZW5ndGggPSAoaSA9PT0gMCkgPyAwIDogY3VtdWxhdGl2ZUxlbmd0aHNbaSAtIDFdO1xuICAgICAgY29uc3QgaW5kaWNlcyA9IFswLCBwcmV2aW91c0xlbmd0aCwgMF07XG4gICAgICBjb25zdCBzaXplcyA9IFsxLCBsZW5ndGhbaV0sIGVsZW1lbnRQZXJSb3ddO1xuICAgICAgdGVuc29yc1tpXSA9IHJlc2hhcGUoXG4gICAgICAgICAgc2xpY2UodGVuc29yLCBpbmRpY2VzLCBzaXplcyksIG91dHB1dEVsZW1lbnRTaGFwZSBhcyBudW1iZXJbXSk7XG4gICAgfVxuICAgIHRlbnNvci5kaXNwb3NlKCk7XG4gICAgcmV0dXJuIHRlbnNvcnM7XG4gIH0pO1xuXG4gIGNvbnN0IGxpc3QgPSBuZXcgVGVuc29yTGlzdChbXSwgZWxlbWVudFNoYXBlLCB0ZW5zb3IuZHR5cGUsIGxlbmd0aC5sZW5ndGgpO1xuXG4gIGZvciAobGV0IGkgPSAwOyBpIDwgdGVuc29ycy5sZW5ndGg7IGkrKykge1xuICAgIGxpc3Quc2V0SXRlbShpLCB0ZW5zb3JzW2ldKTtcbiAgfVxuICByZXR1cm4gbGlzdDtcbn1cbiJdfQ==