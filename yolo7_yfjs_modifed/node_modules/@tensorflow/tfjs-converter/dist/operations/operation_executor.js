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
import * as tfc from '@tensorflow/tfjs-core';
import { NodeValueImpl } from './custom_op/node_value_impl';
import { getRegisteredOp } from './custom_op/register';
import * as arithmetic from './executors/arithmetic_executor';
import * as basicMath from './executors/basic_math_executor';
import * as control from './executors/control_executor';
import * as convolution from './executors/convolution_executor';
import * as creation from './executors/creation_executor';
import * as dynamic from './executors/dynamic_executor';
import * as evaluation from './executors/evaluation_executor';
import * as graph from './executors/graph_executor';
import * as hashTable from './executors/hash_table_executor';
import * as image from './executors/image_executor';
import * as logical from './executors/logical_executor';
import * as matrices from './executors/matrices_executor';
import * as normalization from './executors/normalization_executor';
import * as reduction from './executors/reduction_executor';
import * as sliceJoin from './executors/slice_join_executor';
import * as sparse from './executors/sparse_executor';
import * as spectral from './executors/spectral_executor';
import * as string from './executors/string_executor';
import * as transformation from './executors/transformation_executor';
/**
 * Executes the op defined by the node object.
 * @param node
 * @param tensorMap contains tensors for executed nodes and weights
 * @param context contains tensors and information for running the current node.
 * @param resourceManager Optional. Contains global resources of the model.
 */
export function executeOp(node, tensorMap, context, resourceManager) {
    const value = ((node, tensorMap, context) => {
        switch (node.category) {
            case 'arithmetic':
                return tfc.tidy(() => arithmetic.executeOp(node, tensorMap, context));
            case 'basic_math':
                return tfc.tidy(() => basicMath.executeOp(node, tensorMap, context));
            case 'control':
                return control.executeOp(node, tensorMap, context);
            case 'convolution':
                return tfc.tidy(() => convolution.executeOp(node, tensorMap, context));
            case 'creation':
                return tfc.tidy(() => creation.executeOp(node, tensorMap, context));
            case 'dynamic':
                return dynamic.executeOp(node, tensorMap, context);
            case 'evaluation':
                return tfc.tidy(() => evaluation.executeOp(node, tensorMap, context));
            case 'image':
                return tfc.tidy(() => image.executeOp(node, tensorMap, context));
            case 'graph':
                return tfc.tidy(() => graph.executeOp(node, tensorMap, context));
            case 'logical':
                return tfc.tidy(() => logical.executeOp(node, tensorMap, context));
            case 'matrices':
                return tfc.tidy(() => matrices.executeOp(node, tensorMap, context));
            case 'normalization':
                return tfc.tidy(() => normalization.executeOp(node, tensorMap, context));
            case 'reduction':
                return tfc.tidy(() => reduction.executeOp(node, tensorMap, context));
            case 'slice_join':
                return tfc.tidy(() => sliceJoin.executeOp(node, tensorMap, context));
            case 'sparse':
                return tfc.tidy(() => sparse.executeOp(node, tensorMap, context));
            case 'spectral':
                return tfc.tidy(() => spectral.executeOp(node, tensorMap, context));
            case 'string':
                return tfc.tidy(() => string.executeOp(node, tensorMap, context));
            case 'transformation':
                return tfc.tidy(() => transformation.executeOp(node, tensorMap, context));
            case 'hash_table':
                return hashTable.executeOp(node, tensorMap, context, resourceManager);
            case 'custom':
                const opMapper = getRegisteredOp(node.op);
                if (opMapper && opMapper.customExecutor) {
                    return opMapper.customExecutor(new NodeValueImpl(node, tensorMap, context));
                }
                else {
                    throw TypeError(`Custom op ${node.op} is not registered.`);
                }
            default:
                throw TypeError(`Unknown op '${node.op}'. File an issue at ` +
                    `https://github.com/tensorflow/tfjs/issues so we can add it` +
                    `, or register a custom execution with tf.registerOp()`);
        }
    })(node, tensorMap, context);
    if (tfc.util.isPromise(value)) {
        return value.then((data) => [].concat(data));
    }
    return [].concat(value);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoib3BlcmF0aW9uX2V4ZWN1dG9yLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb252ZXJ0ZXIvc3JjL29wZXJhdGlvbnMvb3BlcmF0aW9uX2V4ZWN1dG9yLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sS0FBSyxHQUFHLE1BQU0sdUJBQXVCLENBQUM7QUFNN0MsT0FBTyxFQUFDLGFBQWEsRUFBQyxNQUFNLDZCQUE2QixDQUFDO0FBQzFELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUNyRCxPQUFPLEtBQUssVUFBVSxNQUFNLGlDQUFpQyxDQUFDO0FBQzlELE9BQU8sS0FBSyxTQUFTLE1BQU0saUNBQWlDLENBQUM7QUFDN0QsT0FBTyxLQUFLLE9BQU8sTUFBTSw4QkFBOEIsQ0FBQztBQUN4RCxPQUFPLEtBQUssV0FBVyxNQUFNLGtDQUFrQyxDQUFDO0FBQ2hFLE9BQU8sS0FBSyxRQUFRLE1BQU0sK0JBQStCLENBQUM7QUFDMUQsT0FBTyxLQUFLLE9BQU8sTUFBTSw4QkFBOEIsQ0FBQztBQUN4RCxPQUFPLEtBQUssVUFBVSxNQUFNLGlDQUFpQyxDQUFDO0FBQzlELE9BQU8sS0FBSyxLQUFLLE1BQU0sNEJBQTRCLENBQUM7QUFDcEQsT0FBTyxLQUFLLFNBQVMsTUFBTSxpQ0FBaUMsQ0FBQztBQUM3RCxPQUFPLEtBQUssS0FBSyxNQUFNLDRCQUE0QixDQUFDO0FBQ3BELE9BQU8sS0FBSyxPQUFPLE1BQU0sOEJBQThCLENBQUM7QUFDeEQsT0FBTyxLQUFLLFFBQVEsTUFBTSwrQkFBK0IsQ0FBQztBQUMxRCxPQUFPLEtBQUssYUFBYSxNQUFNLG9DQUFvQyxDQUFDO0FBQ3BFLE9BQU8sS0FBSyxTQUFTLE1BQU0sZ0NBQWdDLENBQUM7QUFDNUQsT0FBTyxLQUFLLFNBQVMsTUFBTSxpQ0FBaUMsQ0FBQztBQUM3RCxPQUFPLEtBQUss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RSxTQUFTLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztpQkFDbEQ7cUJBQU07b0JBQ0wsTUFBTSxTQUFTLENBQUMsYUFBYSxJQUFJLENBQUMsRUFBRSxxQkFBcUIsQ0FBQyxDQUFDO2lCQUM1RDtZQUNIO2dCQUNFLE1BQU0sU0FBUyxDQUNYLGVBQWUsSUFBSSxDQUFDLEVBQUUsc0JBQXNCO29CQUM1Qyw0REFBNEQ7b0JBQzVELHVEQUF1RCxDQUFDLENBQUM7U0FDaEU7SUFDSCxDQUFDLENBQUMsQ0FBQyxJQUFJLEVBQUUsU0FBUyxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ2pDLElBQUksR0FBRyxDQUFDLElBQUksQ0FBQyxTQUFTLENBQUMsS0FBSyxDQUFDLEVBQUU7UUFDN0IsT0FBUSxLQUE2QixDQUFDLElBQUksQ0FBQyxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDO0tBQ3ZFO0lBQ0QsT0FBTyxFQUFFLENBQUMsTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDO0FBQzFCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCAqIGFzIHRmYyBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge05hbWVkVGVuc29yc01hcH0gZnJvbSAnLi4vZGF0YS90eXBlcyc7XG5pbXBvcnQge0V4ZWN1dGlvbkNvbnRleHR9IGZyb20gJy4uL2V4ZWN1dG9yL2V4ZWN1dGlvbl9jb250ZXh0JztcbmltcG9ydCB7UmVzb3VyY2VNYW5hZ2VyfSBmcm9tICcuLi9leGVjdXRvci9yZXNvdXJjZV9tYW5hZ2VyJztcblxuaW1wb3J0IHtOb2RlVmFsdWVJbXBsfSBmcm9tICcuL2N1c3RvbV9vcC9ub2RlX3ZhbHVlX2ltcGwnO1xuaW1wb3J0IHtnZXRSZWdpc3RlcmVkT3B9IGZyb20gJy4vY3VzdG9tX29wL3JlZ2lzdGVyJztcbmltcG9ydCAqIGFzIGFyaXRobWV0aWMgZnJvbSAnLi9leGVjdXRvcnMvYXJpdGhtZXRpY19leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBiYXNpY01hdGggZnJvbSAnLi9leGVjdXRvcnMvYmFzaWNfbWF0aF9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBjb250cm9sIGZyb20gJy4vZXhlY3V0b3JzL2NvbnRyb2xfZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgY29udm9sdXRpb24gZnJvbSAnLi9leGVjdXRvcnMvY29udm9sdXRpb25fZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgY3JlYXRpb24gZnJvbSAnLi9leGVjdXRvcnMvY3JlYXRpb25fZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgZHluYW1pYyBmcm9tICcuL2V4ZWN1dG9ycy9keW5hbWljX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIGV2YWx1YXRpb24gZnJvbSAnLi9leGVjdXRvcnMvZXZhbHVhdGlvbl9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBncmFwaCBmcm9tICcuL2V4ZWN1dG9ycy9ncmFwaF9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBoYXNoVGFibGUgZnJvbSAnLi9leGVjdXRvcnMvaGFzaF90YWJsZV9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBpbWFnZSBmcm9tICcuL2V4ZWN1dG9ycy9pbWFnZV9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBsb2dpY2FsIGZyb20gJy4vZXhlY3V0b3JzL2xvZ2ljYWxfZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgbWF0cmljZXMgZnJvbSAnLi9leGVjdXRvcnMvbWF0cmljZXNfZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgbm9ybWFsaXphdGlvbiBmcm9tICcuL2V4ZWN1dG9ycy9ub3JtYWxpemF0aW9uX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIHJlZHVjdGlvbiBmcm9tICcuL2V4ZWN1dG9ycy9yZWR1Y3Rpb25fZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgc2xpY2VKb2luIGZyb20gJy4vZXhlY3V0b3JzL3NsaWNlX2pvaW5fZXhlY3V0b3InO1xuaW1wb3J0ICogYXMgc3BhcnNlIGZyb20gJy4vZXhlY3V0b3JzL3NwYXJzZV9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBzcGVjdHJhbCBmcm9tICcuL2V4ZWN1dG9ycy9zcGVjdHJhbF9leGVjdXRvcic7XG5pbXBvcnQgKiBhcyBzdHJpbmcgZnJvbSAnLi9leGVjdXRvcnMvc3RyaW5nX2V4ZWN1dG9yJztcbmltcG9ydCAqIGFzIHRyYW5zZm9ybWF0aW9uIGZyb20gJy4vZXhlY3V0b3JzL3RyYW5zZm9ybWF0aW9uX2V4ZWN1dG9yJztcbmltcG9ydCB7Tm9kZX0gZnJvbSAnLi90eXBlcyc7XG5cbi8qKlxuICogRXhlY3V0ZXMgdGhlIG9wIGRlZmluZWQgYnkgdGhlIG5vZGUgb2JqZWN0LlxuICogQHBhcmFtIG5vZGVcbiAqIEBwYXJhbSB0ZW5zb3JNYXAgY29udGFpbnMgdGVuc29ycyBmb3IgZXhlY3V0ZWQgbm9kZXMgYW5kIHdlaWdodHNcbiAqIEBwYXJhbSBjb250ZXh0IGNvbnRhaW5zIHRlbnNvcnMgYW5kIGluZm9ybWF0aW9uIGZvciBydW5uaW5nIHRoZSBjdXJyZW50IG5vZGUuXG4gKiBAcGFyYW0gcmVzb3VyY2VNYW5hZ2VyIE9wdGlvbmFsLiBDb250YWlucyBnbG9iYWwgcmVzb3VyY2VzIG9mIHRoZSBtb2RlbC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGV4ZWN1dGVPcChcbiAgICBub2RlOiBOb2RlLCB0ZW5zb3JNYXA6IE5hbWVkVGVuc29yc01hcCwgY29udGV4dDogRXhlY3V0aW9uQ29udGV4dCxcbiAgICByZXNvdXJjZU1hbmFnZXI/OiBSZXNvdXJjZU1hbmFnZXIpOiB0ZmMuVGVuc29yW118UHJvbWlzZTx0ZmMuVGVuc29yW10+IHtcbiAgY29uc3QgdmFsdWUgPVxuICAgICAgKChub2RlOiBOb2RlLCB0ZW5zb3JNYXA6IE5hbWVkVGVuc29yc01hcCwgY29udGV4dDogRXhlY3V0aW9uQ29udGV4dCkgPT4ge1xuICAgICAgICBzd2l0Y2ggKG5vZGUuY2F0ZWdvcnkpIHtcbiAgICAgICAgICBjYXNlICdhcml0aG1ldGljJzpcbiAgICAgICAgICAgIHJldHVybiB0ZmMudGlkeShcbiAgICAgICAgICAgICAgICAoKSA9PiBhcml0aG1ldGljLmV4ZWN1dGVPcChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpKTtcbiAgICAgICAgICBjYXNlICdiYXNpY19tYXRoJzpcbiAgICAgICAgICAgIHJldHVybiB0ZmMudGlkeShcbiAgICAgICAgICAgICAgICAoKSA9PiBiYXNpY01hdGguZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ2NvbnRyb2wnOlxuICAgICAgICAgICAgcmV0dXJuIGNvbnRyb2wuZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCk7XG4gICAgICAgICAgY2FzZSAnY29udm9sdXRpb24nOlxuICAgICAgICAgICAgcmV0dXJuIHRmYy50aWR5KFxuICAgICAgICAgICAgICAgICgpID0+IGNvbnZvbHV0aW9uLmV4ZWN1dGVPcChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpKTtcbiAgICAgICAgICBjYXNlICdjcmVhdGlvbic6XG4gICAgICAgICAgICByZXR1cm4gdGZjLnRpZHkoKCkgPT4gY3JlYXRpb24uZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ2R5bmFtaWMnOlxuICAgICAgICAgICAgcmV0dXJuIGR5bmFtaWMuZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCk7XG4gICAgICAgICAgY2FzZSAnZXZhbHVhdGlvbic6XG4gICAgICAgICAgICByZXR1cm4gdGZjLnRpZHkoXG4gICAgICAgICAgICAgICAgKCkgPT4gZXZhbHVhdGlvbi5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAnaW1hZ2UnOlxuICAgICAgICAgICAgcmV0dXJuIHRmYy50aWR5KCgpID0+IGltYWdlLmV4ZWN1dGVPcChub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpKTtcbiAgICAgICAgICBjYXNlICdncmFwaCc6XG4gICAgICAgICAgICByZXR1cm4gdGZjLnRpZHkoKCkgPT4gZ3JhcGguZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ2xvZ2ljYWwnOlxuICAgICAgICAgICAgcmV0dXJuIHRmYy50aWR5KCgpID0+IGxvZ2ljYWwuZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ21hdHJpY2VzJzpcbiAgICAgICAgICAgIHJldHVybiB0ZmMudGlkeSgoKSA9PiBtYXRyaWNlcy5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAnbm9ybWFsaXphdGlvbic6XG4gICAgICAgICAgICByZXR1cm4gdGZjLnRpZHkoXG4gICAgICAgICAgICAgICAgKCkgPT4gbm9ybWFsaXphdGlvbi5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAncmVkdWN0aW9uJzpcbiAgICAgICAgICAgIHJldHVybiB0ZmMudGlkeShcbiAgICAgICAgICAgICAgICAoKSA9PiByZWR1Y3Rpb24uZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ3NsaWNlX2pvaW4nOlxuICAgICAgICAgICAgcmV0dXJuIHRmYy50aWR5KFxuICAgICAgICAgICAgICAgICgpID0+IHNsaWNlSm9pbi5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAnc3BhcnNlJzpcbiAgICAgICAgICAgIHJldHVybiB0ZmMudGlkeSgoKSA9PiBzcGFyc2UuZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ3NwZWN0cmFsJzpcbiAgICAgICAgICAgIHJldHVybiB0ZmMudGlkeSgoKSA9PiBzcGVjdHJhbC5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAnc3RyaW5nJzpcbiAgICAgICAgICAgIHJldHVybiB0ZmMudGlkeSgoKSA9PiBzdHJpbmcuZXhlY3V0ZU9wKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgIGNhc2UgJ3RyYW5zZm9ybWF0aW9uJzpcbiAgICAgICAgICAgIHJldHVybiB0ZmMudGlkeShcbiAgICAgICAgICAgICAgICAoKSA9PiB0cmFuc2Zvcm1hdGlvbi5leGVjdXRlT3Aobm9kZSwgdGVuc29yTWFwLCBjb250ZXh0KSk7XG4gICAgICAgICAgY2FzZSAnaGFzaF90YWJsZSc6XG4gICAgICAgICAgICByZXR1cm4gaGFzaFRhYmxlLmV4ZWN1dGVPcChcbiAgICAgICAgICAgICAgICBub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQsIHJlc291cmNlTWFuYWdlcik7XG4gICAgICAgICAgY2FzZSAnY3VzdG9tJzpcbiAgICAgICAgICAgIGNvbnN0IG9wTWFwcGVyID0gZ2V0UmVnaXN0ZXJlZE9wKG5vZGUub3ApO1xuICAgICAgICAgICAgaWYgKG9wTWFwcGVyICYmIG9wTWFwcGVyLmN1c3RvbUV4ZWN1dG9yKSB7XG4gICAgICAgICAgICAgIHJldHVybiBvcE1hcHBlci5jdXN0b21FeGVjdXRvcihcbiAgICAgICAgICAgICAgICAgIG5ldyBOb2RlVmFsdWVJbXBsKG5vZGUsIHRlbnNvck1hcCwgY29udGV4dCkpO1xuICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgdGhyb3cgVHlwZUVycm9yKGBDdXN0b20gb3AgJHtub2RlLm9wfSBpcyBub3QgcmVnaXN0ZXJlZC5gKTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICBkZWZhdWx0OlxuICAgICAgICAgICAgdGhyb3cgVHlwZUVycm9yKFxuICAgICAgICAgICAgICAgIGBVbmtub3duIG9wICcke25vZGUub3B9Jy4gRmlsZSBhbiBpc3N1ZSBhdCBgICtcbiAgICAgICAgICAgICAgICBgaHR0cHM6Ly9naXRodWIuY29tL3RlbnNvcmZsb3cvdGZqcy9pc3N1ZXMgc28gd2UgY2FuIGFkZCBpdGAgK1xuICAgICAgICAgICAgICAgIGAsIG9yIHJlZ2lzdGVyIGEgY3VzdG9tIGV4ZWN1dGlvbiB3aXRoIHRmLnJlZ2lzdGVyT3AoKWApO1xuICAgICAgICB9XG4gICAgICB9KShub2RlLCB0ZW5zb3JNYXAsIGNvbnRleHQpO1xuICBpZiAodGZjLnV0aWwuaXNQcm9taXNlKHZhbHVlKSkge1xuICAgIHJldHVybiAodmFsdWUgYXMgUHJvbWlzZTx0ZmMuVGVuc29yPikudGhlbigoZGF0YSkgPT4gW10uY29uY2F0KGRhdGEpKTtcbiAgfVxuICByZXR1cm4gW10uY29uY2F0KHZhbHVlKTtcbn1cbiJdfQ==