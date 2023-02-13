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
import { scalar } from '@tensorflow/tfjs-core';
import { TensorArray } from '../../executor/tensor_array';
import { fromTensor, reserve, scatter, split } from '../../executor/tensor_list';
import { cloneTensor, getParamValue, getTensor } from './utils';
export const executeOp = async (node, tensorMap, context) => {
    switch (node.op) {
        case 'If':
        case 'StatelessIf': {
            const thenFunc = getParamValue('thenBranch', node, tensorMap, context);
            const elseFunc = getParamValue('elseBranch', node, tensorMap, context);
            const cond = getParamValue('cond', node, tensorMap, context);
            const args = getParamValue('args', node, tensorMap, context);
            const condValue = await cond.data();
            if (condValue[0]) {
                return context.functionMap[thenFunc].executeFunctionAsync(args, context.tensorArrayMap, context.tensorListMap);
            }
            else {
                return context.functionMap[elseFunc].executeFunctionAsync(args, context.tensorArrayMap, context.tensorListMap);
            }
        }
        case 'While':
        case 'StatelessWhile': {
            const bodyFunc = getParamValue('body', node, tensorMap, context);
            const condFunc = getParamValue('cond', node, tensorMap, context);
            const args = getParamValue('args', node, tensorMap, context);
            // Calculate the condition of the loop
            const condResult = (await context.functionMap[condFunc].executeFunctionAsync(args, context.tensorArrayMap, context.tensorListMap));
            const argIds = args.map(tensor => tensor.id);
            let condValue = await condResult[0].data();
            // Dispose the intermediate tensors for condition function
            condResult.forEach(tensor => {
                if (!tensor.kept && argIds.indexOf(tensor.id) === -1) {
                    tensor.dispose();
                }
            });
            let result = args;
            while (condValue[0]) {
                // Record the previous result for intermediate tensor tracking
                const origResult = result;
                // Execution the body of the loop
                result = await context.functionMap[bodyFunc].executeFunctionAsync(result, context.tensorArrayMap, context.tensorListMap);
                const resultIds = result.map(tensor => tensor.id);
                // Dispose the intermediate tensor for body function that is not global
                // kept, not input/output of the body function
                origResult.forEach(tensor => {
                    if (!tensor.kept && argIds.indexOf(tensor.id) === -1 &&
                        resultIds.indexOf(tensor.id) === -1) {
                        tensor.dispose();
                    }
                });
                // Recalcuate the condition of the loop using the latest results.
                const condResult = (await context.functionMap[condFunc].executeFunctionAsync(result, context.tensorArrayMap, context.tensorListMap));
                condValue = await condResult[0].data();
                // Dispose the intermediate tensors for condition function
                condResult.forEach(tensor => {
                    if (!tensor.kept && argIds.indexOf(tensor.id) === -1 &&
                        resultIds.indexOf(tensor.id) === -1) {
                        tensor.dispose();
                    }
                });
            }
            return result;
        }
        case 'LoopCond': {
            const pred = getParamValue('pred', node, tensorMap, context);
            return [cloneTensor(pred)];
        }
        case 'Switch': {
            const pred = getParamValue('pred', node, tensorMap, context);
            let data = getParamValue('data', node, tensorMap, context);
            if (!data.kept) {
                data = cloneTensor(data);
            }
            // Outputs nodes :0 => false, :1 => true
            return (await pred.data())[0] ? [undefined, data] : [data, undefined];
        }
        case 'Merge': {
            const inputName = node.inputNames.find(name => getTensor(name, tensorMap, context) !== undefined);
            if (inputName) {
                const data = getTensor(inputName, tensorMap, context);
                return [cloneTensor(data)];
            }
            return undefined;
        }
        case 'Enter': {
            const frameId = getParamValue('frameName', node, tensorMap, context);
            const data = getParamValue('tensor', node, tensorMap, context);
            context.enterFrame(frameId);
            return [cloneTensor(data)];
        }
        case 'Exit': {
            const data = getParamValue('tensor', node, tensorMap, context);
            context.exitFrame();
            return [cloneTensor(data)];
        }
        case 'NextIteration': {
            const data = getParamValue('tensor', node, tensorMap, context);
            context.nextIteration();
            return [cloneTensor(data)];
        }
        case 'TensorArrayV3': {
            const size = getParamValue('size', node, tensorMap, context);
            const dtype = getParamValue('dtype', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const dynamicSize = getParamValue('dynamicSize', node, tensorMap, context);
            const clearAfterRead = getParamValue('clearAfterRead', node, tensorMap, context);
            const identicalElementShapes = getParamValue('identicalElementShapes', node, tensorMap, context);
            const name = getParamValue('name', node, tensorMap, context);
            const tensorArray = new TensorArray(name, dtype, size, elementShape, identicalElementShapes, dynamicSize, clearAfterRead);
            context.addTensorArray(tensorArray);
            return [tensorArray.idTensor, scalar(1.0)];
        }
        case 'TensorArrayWriteV3': {
            const id = getParamValue('tensorArrayId', node, tensorMap, context);
            const index = getParamValue('index', node, tensorMap, context);
            const writeTensor = getParamValue('tensor', node, tensorMap, context);
            const writeTensorArray = context.getTensorArray(id.id);
            writeTensorArray.write(index, writeTensor);
            return [writeTensorArray.idTensor];
        }
        case 'TensorArrayReadV3': {
            const readId = getParamValue('tensorArrayId', node, tensorMap, context);
            const readIndex = getParamValue('index', node, tensorMap, context);
            const readTensorArray = context.getTensorArray(readId.id);
            return [readTensorArray.read(readIndex)];
        }
        case 'TensorArrayGatherV3': {
            const gatherId = getParamValue('tensorArrayId', node, tensorMap, context);
            const gatherIndices = getParamValue('indices', node, tensorMap, context);
            const gatherDtype = getParamValue('dtype', node, tensorMap, context);
            const gatherTensorArray = context.getTensorArray(gatherId.id);
            return [gatherTensorArray.gather(gatherIndices, gatherDtype)];
        }
        case 'TensorArrayScatterV3': {
            const scatterId = getParamValue('tensorArrayId', node, tensorMap, context);
            const scatterIndices = getParamValue('indices', node, tensorMap, context);
            const scatterTensor = getParamValue('tensor', node, tensorMap, context);
            const scatterTensorArray = context.getTensorArray(scatterId.id);
            scatterTensorArray.scatter(scatterIndices, scatterTensor);
            return [scatterTensorArray.idTensor];
        }
        case 'TensorArrayConcatV3': {
            const concatId = getParamValue('tensorArrayId', node, tensorMap, context);
            const concatTensorArray = context.getTensorArray(concatId.id);
            const concatDtype = getParamValue('dtype', node, tensorMap, context);
            return [concatTensorArray.concat(concatDtype)];
        }
        case 'TensorArraySplitV3': {
            const splitId = getParamValue('tensorArrayId', node, tensorMap, context);
            const splitTensor = getParamValue('tensor', node, tensorMap, context);
            const lengths = getParamValue('lengths', node, tensorMap, context);
            const splitTensorArray = context.getTensorArray(splitId.id);
            splitTensorArray.split(lengths, splitTensor);
            return [splitTensorArray.idTensor];
        }
        case 'TensorArraySizeV3': {
            const sizeId = getParamValue('tensorArrayId', node, tensorMap, context);
            const sizeTensorArray = context.getTensorArray(sizeId.id);
            return [scalar(sizeTensorArray.size(), 'int32')];
        }
        case 'TensorArrayCloseV3': {
            const closeId = getParamValue('tensorArrayId', node, tensorMap, context);
            const closeTensorArray = context.getTensorArray(closeId.id);
            closeTensorArray.clearAndClose();
            return [closeTensorArray.idTensor];
        }
        case 'TensorListSetItem': {
            const idTensor = getParamValue('tensorListId', node, tensorMap, context);
            const index = getParamValue('index', node, tensorMap, context);
            const writeTensor = getParamValue('tensor', node, tensorMap, context);
            const tensorList = context.getTensorList(idTensor.id);
            tensorList.setItem(index, writeTensor);
            return [tensorList.idTensor];
        }
        case 'TensorListGetItem': {
            const idTensor = getParamValue('tensorListId', node, tensorMap, context);
            const readIndex = getParamValue('index', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const elementDType = getParamValue('elementDType', node, tensorMap, context);
            const tensorList = context.getTensorList(idTensor.id);
            return [tensorList.getItem(readIndex, elementShape, elementDType)];
        }
        case 'TensorListScatterV2':
        case 'TensorListScatter': {
            const scatterIndices = getParamValue('indices', node, tensorMap, context);
            const scatterTensor = getParamValue('tensor', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const numElements = getParamValue('numElements', node, tensorMap, context);
            const tensorList = scatter(scatterTensor, scatterIndices, elementShape, numElements);
            context.addTensorList(tensorList);
            return [tensorList.idTensor];
        }
        case 'TensorListReserve':
        case 'EmptyTensorList': {
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const elementDtype = getParamValue('elementDType', node, tensorMap, context);
            let numElementsParam;
            if (node.op === 'TensorListReserve') {
                numElementsParam = 'numElements';
            }
            else {
                numElementsParam = 'maxNumElements';
            }
            const numElements = getParamValue(numElementsParam, node, tensorMap, context);
            const tensorList = reserve(elementShape, elementDtype, numElements);
            context.addTensorList(tensorList);
            return [tensorList.idTensor];
        }
        case 'TensorListGather': {
            const gatherId = getParamValue('tensorListId', node, tensorMap, context);
            const gatherIndices = getParamValue('indices', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const elementDtype = getParamValue('elementDType', node, tensorMap, context);
            const tensorList = context.getTensorList(gatherId.id);
            return [tensorList.gather(gatherIndices, elementDtype, elementShape)];
        }
        case 'TensorListStack': {
            const idTensor = getParamValue('tensorListId', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const elementDtype = getParamValue('elementDType', node, tensorMap, context);
            const numElements = getParamValue('numElements', node, tensorMap, context);
            const tensorList = context.getTensorList(idTensor.id);
            return [tensorList.stack(elementShape, elementDtype, numElements)];
        }
        case 'TensorListFromTensor': {
            const tensor = getParamValue('tensor', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const elementDtype = getParamValue('elementDType', node, tensorMap, context);
            const tensorList = fromTensor(tensor, elementShape, elementDtype);
            context.addTensorList(tensorList);
            return [tensorList.idTensor];
        }
        case 'TensorListConcat': {
            const concatId = getParamValue('tensorListId', node, tensorMap, context);
            const tensorList = context.getTensorList(concatId.id);
            const concatDtype = getParamValue('dtype', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            return [tensorList.concat(concatDtype, elementShape)];
        }
        case 'TensorListPushBack': {
            const idTensor = getParamValue('tensorListId', node, tensorMap, context);
            const writeTensor = getParamValue('tensor', node, tensorMap, context);
            const tensorList = context.getTensorList(idTensor.id);
            tensorList.pushBack(writeTensor);
            return [tensorList.idTensor];
        }
        case 'TensorListPopBack': {
            const idTensor = getParamValue('tensorListId', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const elementDType = getParamValue('elementDType', node, tensorMap, context);
            const tensorList = context.getTensorList(idTensor.id);
            return [tensorList.popBack(elementShape, elementDType)];
        }
        case 'TensorListSplit': {
            const splitTensor = getParamValue('tensor', node, tensorMap, context);
            const elementShape = getParamValue('elementShape', node, tensorMap, context);
            const lengths = getParamValue('lengths', node, tensorMap, context);
            const tensorList = split(splitTensor, lengths, elementShape);
            context.addTensorList(tensorList);
            return [tensorList.idTensor];
        }
        default:
            throw TypeError(`Node type ${node.op} is not implemented`);
    }
};
export const CATEGORY = 'control';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udHJvbF9leGVjdXRvci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29udmVydGVyL3NyYy9vcGVyYXRpb25zL2V4ZWN1dG9ycy9jb250cm9sX2V4ZWN1dG9yLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBVyxNQUFNLEVBQVMsTUFBTSx1QkFBdUIsQ0FBQztBQUkvRCxPQUFPLEVBQUMsV0FBVyxFQUFDLE1BQU0sNkJBQTZCLENBQUM7QUFDeEQsT0FBTyxFQUFDLFVBQVUsRUFBRSxPQUFPLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxNQUFNLDRCQUE0QixDQUFDO0FBRy9FLE9BQU8sRUFBQyxXQUFXLEVBQUUsYUFBYSxFQUFFLFNBQVMsRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUU5RCxNQUFNLENBQUMsTUFBTSxTQUFTLEdBQTRCLEtBQUssRUFDbkQsSUFBVSxFQUFFLFNBQTBCLEVBQ3RDLE9BQXlCLEVBQXFCLEVBQUU7SUFDbEQsUUFBUSxJQUFJLENBQUMsRUFBRSxFQUFFO1FBQ2YsS0FBSyxJQUFJLENBQUM7UUFDVixLQUFLLGFBQWEsQ0FBQyxDQUFDO1lBQ2xCLE1BQU0sUUFBUSxHQUNWLGFBQWEsQ0FBQyxZQUFZLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUNwRSxNQUFNLFFBQVEsR0FDVixhQUFhLENBQUMsWUFBWSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFXLENBQUM7WUFDcEUsTUFBTSxJQUFJLEdBQUcsYUFBYSxDQUFDLE1BQU0sRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBVyxDQUFDO1lBQ3ZFLE1BQU0sSUFBSSxHQUFHLGFBQWEsQ0FBQyxNQUFNLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQWEsQ0FBQztZQUN6RSxNQUFNLFNBQVMsR0FB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QU8sQ0FBYSxDQUFDO1lBQ3hFLE1BQU0sVUFBVSxHQUFHLE9BQU8sQ0FBQyxhQUFhLENBQUMsUUFBUSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1lBQ3RELE9BQU8sQ0FBQyxVQUFVLENBQUMsT0FBTyxDQUFDLFlBQVksRUFBRSxZQUFZLENBQUMsQ0FBQyxDQUFDO1NBQ3pEO1FBQ0QsS0FBSyxpQkFBaUIsQ0FBQyxDQUFDO1lBQ3RCLE1BQU0sV0FBVyxHQUNiLGFBQWEsQ0FBQyxRQUFRLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxPQUFPLENBQVcsQ0FBQztZQUNoRSxNQUFNLFlBQVksR0FDZCxhQUFhLENBQUMsY0FBYyxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsT0FBTyxDQUFhLENBQUM7WUFDeEUsTUFBTSxPQUFPLEdBQ1QsYUFBYSxDQUFDLFNBQVMsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBYSxDQUFDO1lBRW5FLE1BQU0sVUFBVSxHQUFHLEtBQUssQ0FBQyxXQUFXLEVBQUUsT0FBTyxFQUFFLFlBQVksQ0FBQyxDQUFDO1lBQzdELE9BQU8sQ0FBQyxhQUFhLENBQUMsVUFBVSxDQUFDLENBQUM7WUFDbEMsT0FBTyxDQUFDLFVBQVUsQ0FBQyxRQUFRLENBQUMsQ0FBQztTQUM5QjtRQUNEO1lBQ0UsTUFBTSxTQUFTLENBQUMsYUFBYSxJQUFJLENBQUMsRUFBRSxxQkFBcUIsQ0FBQyxDQUFDO0tBQzlEO0FBQ0gsQ0FBQyxDQUFDO0FBRUYsTUFBTSxDQUFDLE1BQU0sUUFBUSxHQUFHLFNBQVMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtEYXRhVHlwZSwgc2NhbGFyLCBUZW5zb3J9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TmFtZWRUZW5zb3JzTWFwfSBmcm9tICcuLi8uLi9kYXRhL3R5cGVzJztcbmltcG9ydCB7RXhlY3V0aW9uQ29udGV4dH0gZnJvbSAnLi4vLi4vZXhlY3V0b3IvZXhlY3V0aW9uX2NvbnRleHQnO1xuaW1wb3J0IHtUZW5zb3JBcnJheX0gZnJvbSAnLi4vLi4vZXhlY3V0b3IvdGVuc29yX2FycmF5JztcbmltcG9ydCB7ZnJvbVRlbnNvciwgcmVzZXJ2ZSwgc2NhdHRlciwgc3BsaXR9IGZyb20gJy4uLy4uL2V4ZWN1dG9yL3RlbnNvcl9saXN0JztcbmltcG9ydCB7SW50ZXJuYWxPcEFzeW5jRXhlY3V0b3IsIE5vZGV9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtjbG9uZVRlbnNvciwgZ2V0UGFyYW1WYWx1ZSwgZ2V0VGVuc29yfSBmcm9tICcuL3V0aWxzJztcblxuZXhwb3J0IGNvbnN0IGV4ZWN1dGVPcDogSW50ZXJuYWxPcEFzeW5jRXhlY3V0b3IgPSBhc3luYyhcbiAgICBub2RlOiBOb2RlLCB0ZW5zb3JNYXA6IE5hbWVkVGVuc29yc01hcCxcbiAgICBjb250ZXh0OiBFeGVjdXRpb25Db250ZXh0KTogUHJvbWlzZTxUZW5zb3JbXT4gPT4ge1xuICBzd2l0Y2ggKG5vZGUub3ApIHtcbiAgICBjYXNlICdJZic6XG4gICAgY2FzZSAnU3RhdGVsZXNzSWYnOiB7XG4gICAgICBjb25zdCB0aGVuRnVuYyA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndGhlbkJyYW5jaCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgc3RyaW5nO1xuICAgICAgY29uc3QgZWxzZUZ1bmMgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2Vsc2VCcmFuY2gnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIHN0cmluZztcbiAgICAgIGNvbnN0IGNvbmQgPSBnZXRQYXJhbVZhbHVlKCdjb25kJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb25zdCBhcmdzID0gZ2V0UGFyYW1WYWx1ZSgnYXJncycsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yW107XG4gICAgICBjb25zdCBjb25kVmFsdWUgPSBhd2FpdCBjb25kLmRhdGEoKTtcbiAgICAgIGlmIChjb25kVmFsdWVbMF0pIHtcbiAgICAgICAgcmV0dXJuIGNvbnRleHQuZnVuY3Rpb25NYXBbdGhlbkZ1bmNdLmV4ZWN1dGVGdW5jdGlvbkFzeW5jKFxuICAgICAgICAgICAgYXJncywgY29udGV4dC50ZW5zb3JBcnJheU1hcCwgY29udGV4dC50ZW5zb3JMaXN0TWFwKTtcbiAgICAgIH0gZWxzZSB7XG4gICAgICAgIHJldHVybiBjb250ZXh0LmZ1bmN0aW9uTWFwW2Vsc2VGdW5jXS5leGVjdXRlRnVuY3Rpb25Bc3luYyhcbiAgICAgICAgICAgIGFyZ3MsIGNvbnRleHQudGVuc29yQXJyYXlNYXAsIGNvbnRleHQudGVuc29yTGlzdE1hcCk7XG4gICAgICB9XG4gICAgfVxuICAgIGNhc2UgJ1doaWxlJzpcbiAgICBjYXNlICdTdGF0ZWxlc3NXaGlsZSc6IHtcbiAgICAgIGNvbnN0IGJvZHlGdW5jID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdib2R5Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBzdHJpbmc7XG4gICAgICBjb25zdCBjb25kRnVuYyA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnY29uZCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgc3RyaW5nO1xuICAgICAgY29uc3QgYXJncyA9IGdldFBhcmFtVmFsdWUoJ2FyZ3MnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcltdO1xuXG4gICAgICAvLyBDYWxjdWxhdGUgdGhlIGNvbmRpdGlvbiBvZiB0aGUgbG9vcFxuICAgICAgY29uc3QgY29uZFJlc3VsdCA9XG4gICAgICAgICAgKGF3YWl0IGNvbnRleHQuZnVuY3Rpb25NYXBbY29uZEZ1bmNdLmV4ZWN1dGVGdW5jdGlvbkFzeW5jKFxuICAgICAgICAgICAgICBhcmdzLCBjb250ZXh0LnRlbnNvckFycmF5TWFwLCBjb250ZXh0LnRlbnNvckxpc3RNYXApKTtcbiAgICAgIGNvbnN0IGFyZ0lkcyA9IGFyZ3MubWFwKHRlbnNvciA9PiB0ZW5zb3IuaWQpO1xuICAgICAgbGV0IGNvbmRWYWx1ZSA9IGF3YWl0IGNvbmRSZXN1bHRbMF0uZGF0YSgpO1xuICAgICAgLy8gRGlzcG9zZSB0aGUgaW50ZXJtZWRpYXRlIHRlbnNvcnMgZm9yIGNvbmRpdGlvbiBmdW5jdGlvblxuICAgICAgY29uZFJlc3VsdC5mb3JFYWNoKHRlbnNvciA9PiB7XG4gICAgICAgIGlmICghdGVuc29yLmtlcHQgJiYgYXJnSWRzLmluZGV4T2YodGVuc29yLmlkKSA9PT0gLTEpIHtcbiAgICAgICAgICB0ZW5zb3IuZGlzcG9zZSgpO1xuICAgICAgICB9XG4gICAgICB9KTtcblxuICAgICAgbGV0IHJlc3VsdDogVGVuc29yW10gPSBhcmdzO1xuXG4gICAgICB3aGlsZSAoY29uZFZhbHVlWzBdKSB7XG4gICAgICAgIC8vIFJlY29yZCB0aGUgcHJldmlvdXMgcmVzdWx0IGZvciBpbnRlcm1lZGlhdGUgdGVuc29yIHRyYWNraW5nXG4gICAgICAgIGNvbnN0IG9yaWdSZXN1bHQgPSByZXN1bHQ7XG4gICAgICAgIC8vIEV4ZWN1dGlvbiB0aGUgYm9keSBvZiB0aGUgbG9vcFxuICAgICAgICByZXN1bHQgPSBhd2FpdCBjb250ZXh0LmZ1bmN0aW9uTWFwW2JvZHlGdW5jXS5leGVjdXRlRnVuY3Rpb25Bc3luYyhcbiAgICAgICAgICAgIHJlc3VsdCwgY29udGV4dC50ZW5zb3JBcnJheU1hcCwgY29udGV4dC50ZW5zb3JMaXN0TWFwKTtcbiAgICAgICAgY29uc3QgcmVzdWx0SWRzID0gcmVzdWx0Lm1hcCh0ZW5zb3IgPT4gdGVuc29yLmlkKTtcblxuICAgICAgICAvLyBEaXNwb3NlIHRoZSBpbnRlcm1lZGlhdGUgdGVuc29yIGZvciBib2R5IGZ1bmN0aW9uIHRoYXQgaXMgbm90IGdsb2JhbFxuICAgICAgICAvLyBrZXB0LCBub3QgaW5wdXQvb3V0cHV0IG9mIHRoZSBib2R5IGZ1bmN0aW9uXG4gICAgICAgIG9yaWdSZXN1bHQuZm9yRWFjaCh0ZW5zb3IgPT4ge1xuICAgICAgICAgIGlmICghdGVuc29yLmtlcHQgJiYgYXJnSWRzLmluZGV4T2YodGVuc29yLmlkKSA9PT0gLTEgJiZcbiAgICAgICAgICAgICAgcmVzdWx0SWRzLmluZGV4T2YodGVuc29yLmlkKSA9PT0gLTEpIHtcbiAgICAgICAgICAgIHRlbnNvci5kaXNwb3NlKCk7XG4gICAgICAgICAgfVxuICAgICAgICB9KTtcblxuICAgICAgICAvLyBSZWNhbGN1YXRlIHRoZSBjb25kaXRpb24gb2YgdGhlIGxvb3AgdXNpbmcgdGhlIGxhdGVzdCByZXN1bHRzLlxuICAgICAgICBjb25zdCBjb25kUmVzdWx0ID1cbiAgICAgICAgICAgIChhd2FpdCBjb250ZXh0LmZ1bmN0aW9uTWFwW2NvbmRGdW5jXS5leGVjdXRlRnVuY3Rpb25Bc3luYyhcbiAgICAgICAgICAgICAgICByZXN1bHQsIGNvbnRleHQudGVuc29yQXJyYXlNYXAsIGNvbnRleHQudGVuc29yTGlzdE1hcCkpO1xuICAgICAgICBjb25kVmFsdWUgPSBhd2FpdCBjb25kUmVzdWx0WzBdLmRhdGEoKTtcbiAgICAgICAgLy8gRGlzcG9zZSB0aGUgaW50ZXJtZWRpYXRlIHRlbnNvcnMgZm9yIGNvbmRpdGlvbiBmdW5jdGlvblxuICAgICAgICBjb25kUmVzdWx0LmZvckVhY2godGVuc29yID0+IHtcbiAgICAgICAgICBpZiAoIXRlbnNvci5rZXB0ICYmIGFyZ0lkcy5pbmRleE9mKHRlbnNvci5pZCkgPT09IC0xICYmXG4gICAgICAgICAgICAgIHJlc3VsdElkcy5pbmRleE9mKHRlbnNvci5pZCkgPT09IC0xKSB7XG4gICAgICAgICAgICB0ZW5zb3IuZGlzcG9zZSgpO1xuICAgICAgICAgIH1cbiAgICAgICAgfSk7XG4gICAgICB9XG4gICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cbiAgICBjYXNlICdMb29wQ29uZCc6IHtcbiAgICAgIGNvbnN0IHByZWQgPSBnZXRQYXJhbVZhbHVlKCdwcmVkJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICByZXR1cm4gW2Nsb25lVGVuc29yKHByZWQpXTtcbiAgICB9XG4gICAgY2FzZSAnU3dpdGNoJzoge1xuICAgICAgY29uc3QgcHJlZCA9IGdldFBhcmFtVmFsdWUoJ3ByZWQnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjtcbiAgICAgIGxldCBkYXRhID0gZ2V0UGFyYW1WYWx1ZSgnZGF0YScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgaWYgKCFkYXRhLmtlcHQpIHtcbiAgICAgICAgZGF0YSA9IGNsb25lVGVuc29yKGRhdGEpO1xuICAgICAgfVxuICAgICAgLy8gT3V0cHV0cyBub2RlcyA6MCA9PiBmYWxzZSwgOjEgPT4gdHJ1ZVxuICAgICAgcmV0dXJuIChhd2FpdCBwcmVkLmRhdGEoKSlbMF0gPyBbdW5kZWZpbmVkLCBkYXRhXSA6IFtkYXRhLCB1bmRlZmluZWRdO1xuICAgIH1cbiAgICBjYXNlICdNZXJnZSc6IHtcbiAgICAgIGNvbnN0IGlucHV0TmFtZSA9IG5vZGUuaW5wdXROYW1lcy5maW5kKFxuICAgICAgICAgIG5hbWUgPT4gZ2V0VGVuc29yKG5hbWUsIHRlbnNvck1hcCwgY29udGV4dCkgIT09IHVuZGVmaW5lZCk7XG4gICAgICBpZiAoaW5wdXROYW1lKSB7XG4gICAgICAgIGNvbnN0IGRhdGEgPSBnZXRUZW5zb3IoaW5wdXROYW1lLCB0ZW5zb3JNYXAsIGNvbnRleHQpO1xuICAgICAgICByZXR1cm4gW2Nsb25lVGVuc29yKGRhdGEpXTtcbiAgICAgIH1cbiAgICAgIHJldHVybiB1bmRlZmluZWQ7XG4gICAgfVxuICAgIGNhc2UgJ0VudGVyJzoge1xuICAgICAgY29uc3QgZnJhbWVJZCA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnZnJhbWVOYW1lJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBzdHJpbmc7XG4gICAgICBjb25zdCBkYXRhID0gZ2V0UGFyYW1WYWx1ZSgndGVuc29yJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb250ZXh0LmVudGVyRnJhbWUoZnJhbWVJZCk7XG4gICAgICByZXR1cm4gW2Nsb25lVGVuc29yKGRhdGEpXTtcbiAgICB9XG4gICAgY2FzZSAnRXhpdCc6IHtcbiAgICAgIGNvbnN0IGRhdGEgPSBnZXRQYXJhbVZhbHVlKCd0ZW5zb3InLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjtcbiAgICAgIGNvbnRleHQuZXhpdEZyYW1lKCk7XG4gICAgICByZXR1cm4gW2Nsb25lVGVuc29yKGRhdGEpXTtcbiAgICB9XG4gICAgY2FzZSAnTmV4dEl0ZXJhdGlvbic6IHtcbiAgICAgIGNvbnN0IGRhdGEgPSBnZXRQYXJhbVZhbHVlKCd0ZW5zb3InLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjtcbiAgICAgIGNvbnRleHQubmV4dEl0ZXJhdGlvbigpO1xuICAgICAgcmV0dXJuIFtjbG9uZVRlbnNvcihkYXRhKV07XG4gICAgfVxuICAgIGNhc2UgJ1RlbnNvckFycmF5VjMnOiB7XG4gICAgICBjb25zdCBzaXplID0gZ2V0UGFyYW1WYWx1ZSgnc2l6ZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyO1xuICAgICAgY29uc3QgZHR5cGUgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2R0eXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBEYXRhVHlwZTtcbiAgICAgIGNvbnN0IGVsZW1lbnRTaGFwZSA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnZWxlbWVudFNoYXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXJbXTtcbiAgICAgIGNvbnN0IGR5bmFtaWNTaXplID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdkeW5hbWljU2l6ZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgYm9vbGVhbjtcbiAgICAgIGNvbnN0IGNsZWFyQWZ0ZXJSZWFkID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdjbGVhckFmdGVyUmVhZCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgYm9vbGVhbjtcbiAgICAgIGNvbnN0IGlkZW50aWNhbEVsZW1lbnRTaGFwZXMgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2lkZW50aWNhbEVsZW1lbnRTaGFwZXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzXG4gICAgICAgICAgYm9vbGVhbjtcbiAgICAgIGNvbnN0IG5hbWUgPSBnZXRQYXJhbVZhbHVlKCduYW1lJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBzdHJpbmc7XG4gICAgICBjb25zdCB0ZW5zb3JBcnJheSA9IG5ldyBUZW5zb3JBcnJheShcbiAgICAgICAgICBuYW1lLCBkdHlwZSwgc2l6ZSwgZWxlbWVudFNoYXBlLCBpZGVudGljYWxFbGVtZW50U2hhcGVzLCBkeW5hbWljU2l6ZSxcbiAgICAgICAgICBjbGVhckFmdGVyUmVhZCk7XG4gICAgICBjb250ZXh0LmFkZFRlbnNvckFycmF5KHRlbnNvckFycmF5KTtcbiAgICAgIHJldHVybiBbdGVuc29yQXJyYXkuaWRUZW5zb3IsIHNjYWxhcigxLjApXTtcbiAgICB9XG4gICAgY2FzZSAnVGVuc29yQXJyYXlXcml0ZVYzJzoge1xuICAgICAgY29uc3QgaWQgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3RlbnNvckFycmF5SWQnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjtcbiAgICAgIGNvbnN0IGluZGV4ID0gZ2V0UGFyYW1WYWx1ZSgnaW5kZXgnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcjtcbiAgICAgIGNvbnN0IHdyaXRlVGVuc29yID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd0ZW5zb3InLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjtcbiAgICAgIGNvbnN0IHdyaXRlVGVuc29yQXJyYXkgPSBjb250ZXh0LmdldFRlbnNvckFycmF5KGlkLmlkKTtcbiAgICAgIHdyaXRlVGVuc29yQXJyYXkud3JpdGUoaW5kZXgsIHdyaXRlVGVuc29yKTtcbiAgICAgIHJldHVybiBbd3JpdGVUZW5zb3JBcnJheS5pZFRlbnNvcl07XG4gICAgfVxuICAgIGNhc2UgJ1RlbnNvckFycmF5UmVhZFYzJzoge1xuICAgICAgY29uc3QgcmVhZElkID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd0ZW5zb3JBcnJheUlkJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb25zdCByZWFkSW5kZXggPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2luZGV4Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICBjb25zdCByZWFkVGVuc29yQXJyYXkgPSBjb250ZXh0LmdldFRlbnNvckFycmF5KHJlYWRJZC5pZCk7XG4gICAgICByZXR1cm4gW3JlYWRUZW5zb3JBcnJheS5yZWFkKHJlYWRJbmRleCldO1xuICAgIH1cbiAgICBjYXNlICdUZW5zb3JBcnJheUdhdGhlclYzJzoge1xuICAgICAgY29uc3QgZ2F0aGVySWQgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3RlbnNvckFycmF5SWQnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjtcbiAgICAgIGNvbnN0IGdhdGhlckluZGljZXMgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2luZGljZXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgY29uc3QgZ2F0aGVyRHR5cGUgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2R0eXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBEYXRhVHlwZTtcbiAgICAgIGNvbnN0IGdhdGhlclRlbnNvckFycmF5ID0gY29udGV4dC5nZXRUZW5zb3JBcnJheShnYXRoZXJJZC5pZCk7XG4gICAgICByZXR1cm4gW2dhdGhlclRlbnNvckFycmF5LmdhdGhlcihnYXRoZXJJbmRpY2VzLCBnYXRoZXJEdHlwZSldO1xuICAgIH1cbiAgICBjYXNlICdUZW5zb3JBcnJheVNjYXR0ZXJWMyc6IHtcbiAgICAgIGNvbnN0IHNjYXR0ZXJJZCA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndGVuc29yQXJyYXlJZCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgY29uc3Qgc2NhdHRlckluZGljZXMgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2luZGljZXMnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgY29uc3Qgc2NhdHRlclRlbnNvciA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndGVuc29yJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb25zdCBzY2F0dGVyVGVuc29yQXJyYXkgPSBjb250ZXh0LmdldFRlbnNvckFycmF5KHNjYXR0ZXJJZC5pZCk7XG4gICAgICBzY2F0dGVyVGVuc29yQXJyYXkuc2NhdHRlcihzY2F0dGVySW5kaWNlcywgc2NhdHRlclRlbnNvcik7XG4gICAgICByZXR1cm4gW3NjYXR0ZXJUZW5zb3JBcnJheS5pZFRlbnNvcl07XG4gICAgfVxuICAgIGNhc2UgJ1RlbnNvckFycmF5Q29uY2F0VjMnOiB7XG4gICAgICBjb25zdCBjb25jYXRJZCA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndGVuc29yQXJyYXlJZCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgY29uc3QgY29uY2F0VGVuc29yQXJyYXkgPSBjb250ZXh0LmdldFRlbnNvckFycmF5KGNvbmNhdElkLmlkKTtcbiAgICAgIGNvbnN0IGNvbmNhdER0eXBlID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdkdHlwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgRGF0YVR5cGU7XG4gICAgICByZXR1cm4gW2NvbmNhdFRlbnNvckFycmF5LmNvbmNhdChjb25jYXREdHlwZSldO1xuICAgIH1cbiAgICBjYXNlICdUZW5zb3JBcnJheVNwbGl0VjMnOiB7XG4gICAgICBjb25zdCBzcGxpdElkID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd0ZW5zb3JBcnJheUlkJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb25zdCBzcGxpdFRlbnNvciA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndGVuc29yJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb25zdCBsZW5ndGhzID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdsZW5ndGhzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXJbXTtcbiAgICAgIGNvbnN0IHNwbGl0VGVuc29yQXJyYXkgPSBjb250ZXh0LmdldFRlbnNvckFycmF5KHNwbGl0SWQuaWQpO1xuICAgICAgc3BsaXRUZW5zb3JBcnJheS5zcGxpdChsZW5ndGhzLCBzcGxpdFRlbnNvcik7XG4gICAgICByZXR1cm4gW3NwbGl0VGVuc29yQXJyYXkuaWRUZW5zb3JdO1xuICAgIH1cbiAgICBjYXNlICdUZW5zb3JBcnJheVNpemVWMyc6IHtcbiAgICAgIGNvbnN0IHNpemVJZCA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndGVuc29yQXJyYXlJZCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgY29uc3Qgc2l6ZVRlbnNvckFycmF5ID0gY29udGV4dC5nZXRUZW5zb3JBcnJheShzaXplSWQuaWQpO1xuICAgICAgcmV0dXJuIFtzY2FsYXIoc2l6ZVRlbnNvckFycmF5LnNpemUoKSwgJ2ludDMyJyldO1xuICAgIH1cbiAgICBjYXNlICdUZW5zb3JBcnJheUNsb3NlVjMnOiB7XG4gICAgICBjb25zdCBjbG9zZUlkID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd0ZW5zb3JBcnJheUlkJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb25zdCBjbG9zZVRlbnNvckFycmF5ID0gY29udGV4dC5nZXRUZW5zb3JBcnJheShjbG9zZUlkLmlkKTtcbiAgICAgIGNsb3NlVGVuc29yQXJyYXkuY2xlYXJBbmRDbG9zZSgpO1xuICAgICAgcmV0dXJuIFtjbG9zZVRlbnNvckFycmF5LmlkVGVuc29yXTtcbiAgICB9XG4gICAgY2FzZSAnVGVuc29yTGlzdFNldEl0ZW0nOiB7XG4gICAgICBjb25zdCBpZFRlbnNvciA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndGVuc29yTGlzdElkJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb25zdCBpbmRleCA9IGdldFBhcmFtVmFsdWUoJ2luZGV4Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICBjb25zdCB3cml0ZVRlbnNvciA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndGVuc29yJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb25zdCB0ZW5zb3JMaXN0ID0gY29udGV4dC5nZXRUZW5zb3JMaXN0KGlkVGVuc29yLmlkKTtcbiAgICAgIHRlbnNvckxpc3Quc2V0SXRlbShpbmRleCwgd3JpdGVUZW5zb3IpO1xuICAgICAgcmV0dXJuIFt0ZW5zb3JMaXN0LmlkVGVuc29yXTtcbiAgICB9XG4gICAgY2FzZSAnVGVuc29yTGlzdEdldEl0ZW0nOiB7XG4gICAgICBjb25zdCBpZFRlbnNvciA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgndGVuc29yTGlzdElkJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBUZW5zb3I7XG4gICAgICBjb25zdCByZWFkSW5kZXggPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2luZGV4Jywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICBjb25zdCBlbGVtZW50U2hhcGUgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2VsZW1lbnRTaGFwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyW107XG5cbiAgICAgIGNvbnN0IGVsZW1lbnREVHlwZSA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnZWxlbWVudERUeXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBEYXRhVHlwZTtcbiAgICAgIGNvbnN0IHRlbnNvckxpc3QgPSBjb250ZXh0LmdldFRlbnNvckxpc3QoaWRUZW5zb3IuaWQpO1xuICAgICAgcmV0dXJuIFt0ZW5zb3JMaXN0LmdldEl0ZW0ocmVhZEluZGV4LCBlbGVtZW50U2hhcGUsIGVsZW1lbnREVHlwZSldO1xuICAgIH1cbiAgICBjYXNlICdUZW5zb3JMaXN0U2NhdHRlclYyJzpcbiAgICBjYXNlICdUZW5zb3JMaXN0U2NhdHRlcic6IHtcbiAgICAgIGNvbnN0IHNjYXR0ZXJJbmRpY2VzID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdpbmRpY2VzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXJbXTtcbiAgICAgIGNvbnN0IHNjYXR0ZXJUZW5zb3IgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3RlbnNvcicsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgY29uc3QgZWxlbWVudFNoYXBlID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdlbGVtZW50U2hhcGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgY29uc3QgbnVtRWxlbWVudHMgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ251bUVsZW1lbnRzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICBjb25zdCB0ZW5zb3JMaXN0ID1cbiAgICAgICAgICBzY2F0dGVyKHNjYXR0ZXJUZW5zb3IsIHNjYXR0ZXJJbmRpY2VzLCBlbGVtZW50U2hhcGUsIG51bUVsZW1lbnRzKTtcbiAgICAgIGNvbnRleHQuYWRkVGVuc29yTGlzdCh0ZW5zb3JMaXN0KTtcbiAgICAgIHJldHVybiBbdGVuc29yTGlzdC5pZFRlbnNvcl07XG4gICAgfVxuICAgIGNhc2UgJ1RlbnNvckxpc3RSZXNlcnZlJzpcbiAgICBjYXNlICdFbXB0eVRlbnNvckxpc3QnOiB7XG4gICAgICBjb25zdCBlbGVtZW50U2hhcGUgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2VsZW1lbnRTaGFwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyW107XG4gICAgICBjb25zdCBlbGVtZW50RHR5cGUgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2VsZW1lbnREVHlwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgRGF0YVR5cGU7XG4gICAgICBsZXQgbnVtRWxlbWVudHNQYXJhbTtcblxuICAgICAgaWYgKG5vZGUub3AgPT09ICdUZW5zb3JMaXN0UmVzZXJ2ZScpIHtcbiAgICAgICAgbnVtRWxlbWVudHNQYXJhbSA9ICdudW1FbGVtZW50cyc7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBudW1FbGVtZW50c1BhcmFtID0gJ21heE51bUVsZW1lbnRzJztcbiAgICAgIH1cblxuICAgICAgY29uc3QgbnVtRWxlbWVudHMgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUobnVtRWxlbWVudHNQYXJhbSwgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG5cbiAgICAgIGNvbnN0IHRlbnNvckxpc3QgPSByZXNlcnZlKGVsZW1lbnRTaGFwZSwgZWxlbWVudER0eXBlLCBudW1FbGVtZW50cyk7XG4gICAgICBjb250ZXh0LmFkZFRlbnNvckxpc3QodGVuc29yTGlzdCk7XG4gICAgICByZXR1cm4gW3RlbnNvckxpc3QuaWRUZW5zb3JdO1xuICAgIH1cbiAgICBjYXNlICdUZW5zb3JMaXN0R2F0aGVyJzoge1xuICAgICAgY29uc3QgZ2F0aGVySWQgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3RlbnNvckxpc3RJZCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgY29uc3QgZ2F0aGVySW5kaWNlcyA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnaW5kaWNlcycsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyW107XG4gICAgICBjb25zdCBlbGVtZW50U2hhcGUgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2VsZW1lbnRTaGFwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyW107XG4gICAgICBjb25zdCBlbGVtZW50RHR5cGUgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ2VsZW1lbnREVHlwZScsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgRGF0YVR5cGU7XG4gICAgICBjb25zdCB0ZW5zb3JMaXN0ID0gY29udGV4dC5nZXRUZW5zb3JMaXN0KGdhdGhlcklkLmlkKTtcbiAgICAgIHJldHVybiBbdGVuc29yTGlzdC5nYXRoZXIoZ2F0aGVySW5kaWNlcywgZWxlbWVudER0eXBlLCBlbGVtZW50U2hhcGUpXTtcbiAgICB9XG4gICAgY2FzZSAnVGVuc29yTGlzdFN0YWNrJzoge1xuICAgICAgY29uc3QgaWRUZW5zb3IgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3RlbnNvckxpc3RJZCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgY29uc3QgZWxlbWVudFNoYXBlID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdlbGVtZW50U2hhcGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgY29uc3QgZWxlbWVudER0eXBlID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdlbGVtZW50RFR5cGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIERhdGFUeXBlO1xuICAgICAgY29uc3QgbnVtRWxlbWVudHMgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ251bUVsZW1lbnRzJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXI7XG4gICAgICBjb25zdCB0ZW5zb3JMaXN0ID0gY29udGV4dC5nZXRUZW5zb3JMaXN0KGlkVGVuc29yLmlkKTtcbiAgICAgIHJldHVybiBbdGVuc29yTGlzdC5zdGFjayhlbGVtZW50U2hhcGUsIGVsZW1lbnREdHlwZSwgbnVtRWxlbWVudHMpXTtcbiAgICB9XG4gICAgY2FzZSAnVGVuc29yTGlzdEZyb21UZW5zb3InOiB7XG4gICAgICBjb25zdCB0ZW5zb3IgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3RlbnNvcicsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgY29uc3QgZWxlbWVudFNoYXBlID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdlbGVtZW50U2hhcGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgY29uc3QgZWxlbWVudER0eXBlID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdlbGVtZW50RFR5cGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIERhdGFUeXBlO1xuICAgICAgY29uc3QgdGVuc29yTGlzdCA9IGZyb21UZW5zb3IodGVuc29yLCBlbGVtZW50U2hhcGUsIGVsZW1lbnREdHlwZSk7XG4gICAgICBjb250ZXh0LmFkZFRlbnNvckxpc3QodGVuc29yTGlzdCk7XG4gICAgICByZXR1cm4gW3RlbnNvckxpc3QuaWRUZW5zb3JdO1xuICAgIH1cbiAgICBjYXNlICdUZW5zb3JMaXN0Q29uY2F0Jzoge1xuICAgICAgY29uc3QgY29uY2F0SWQgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3RlbnNvckxpc3RJZCcsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgY29uc3QgdGVuc29yTGlzdCA9IGNvbnRleHQuZ2V0VGVuc29yTGlzdChjb25jYXRJZC5pZCk7XG4gICAgICBjb25zdCBjb25jYXREdHlwZSA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnZHR5cGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIERhdGFUeXBlO1xuICAgICAgY29uc3QgZWxlbWVudFNoYXBlID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdlbGVtZW50U2hhcGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgcmV0dXJuIFt0ZW5zb3JMaXN0LmNvbmNhdChjb25jYXREdHlwZSwgZWxlbWVudFNoYXBlKV07XG4gICAgfVxuICAgIGNhc2UgJ1RlbnNvckxpc3RQdXNoQmFjayc6IHtcbiAgICAgIGNvbnN0IGlkVGVuc29yID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd0ZW5zb3JMaXN0SWQnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjtcbiAgICAgIGNvbnN0IHdyaXRlVGVuc29yID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd0ZW5zb3InLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjtcbiAgICAgIGNvbnN0IHRlbnNvckxpc3QgPSBjb250ZXh0LmdldFRlbnNvckxpc3QoaWRUZW5zb3IuaWQpO1xuICAgICAgdGVuc29yTGlzdC5wdXNoQmFjayh3cml0ZVRlbnNvcik7XG4gICAgICByZXR1cm4gW3RlbnNvckxpc3QuaWRUZW5zb3JdO1xuICAgIH1cbiAgICBjYXNlICdUZW5zb3JMaXN0UG9wQmFjayc6IHtcbiAgICAgIGNvbnN0IGlkVGVuc29yID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCd0ZW5zb3JMaXN0SWQnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIFRlbnNvcjtcbiAgICAgIGNvbnN0IGVsZW1lbnRTaGFwZSA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnZWxlbWVudFNoYXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBudW1iZXJbXTtcbiAgICAgIGNvbnN0IGVsZW1lbnREVHlwZSA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnZWxlbWVudERUeXBlJywgbm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSBhcyBEYXRhVHlwZTtcbiAgICAgIGNvbnN0IHRlbnNvckxpc3QgPSBjb250ZXh0LmdldFRlbnNvckxpc3QoaWRUZW5zb3IuaWQpO1xuICAgICAgcmV0dXJuIFt0ZW5zb3JMaXN0LnBvcEJhY2soZWxlbWVudFNoYXBlLCBlbGVtZW50RFR5cGUpXTtcbiAgICB9XG4gICAgY2FzZSAnVGVuc29yTGlzdFNwbGl0Jzoge1xuICAgICAgY29uc3Qgc3BsaXRUZW5zb3IgPVxuICAgICAgICAgIGdldFBhcmFtVmFsdWUoJ3RlbnNvcicsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgVGVuc29yO1xuICAgICAgY29uc3QgZWxlbWVudFNoYXBlID1cbiAgICAgICAgICBnZXRQYXJhbVZhbHVlKCdlbGVtZW50U2hhcGUnLCBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpIGFzIG51bWJlcltdO1xuICAgICAgY29uc3QgbGVuZ3RocyA9XG4gICAgICAgICAgZ2V0UGFyYW1WYWx1ZSgnbGVuZ3RocycsIG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkgYXMgbnVtYmVyW107XG5cbiAgICAgIGNvbnN0IHRlbnNvckxpc3QgPSBzcGxpdChzcGxpdFRlbnNvciwgbGVuZ3RocywgZWxlbWVudFNoYXBlKTtcbiAgICAgIGNvbnRleHQuYWRkVGVuc29yTGlzdCh0ZW5zb3JMaXN0KTtcbiAgICAgIHJldHVybiBbdGVuc29yTGlzdC5pZFRlbnNvcl07XG4gICAgfVxuICAgIGRlZmF1bHQ6XG4gICAgICB0aHJvdyBUeXBlRXJyb3IoYE5vZGUgdHlwZSAke25vZGUub3B9IGlzIG5vdCBpbXBsZW1lbnRlZGApO1xuICB9XG59O1xuXG5leHBvcnQgY29uc3QgQ0FURUdPUlkgPSAnY29udHJvbCc7XG4iXX0=