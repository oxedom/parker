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
/**
 * Unit tests for passthrough IOHandlers.
 */
import * as tf from '../index';
import { BROWSER_ENVS, describeWithFlags } from '../jasmine_util';
const modelTopology1 = {
    'class_name': 'Sequential',
    'keras_version': '2.1.4',
    'config': [{
            'class_name': 'Dense',
            'config': {
                'kernel_initializer': {
                    'class_name': 'VarianceScaling',
                    'config': {
                        'distribution': 'uniform',
                        'scale': 1.0,
                        'seed': null,
                        'mode': 'fan_avg'
                    }
                },
                'name': 'dense',
                'kernel_constraint': null,
                'bias_regularizer': null,
                'bias_constraint': null,
                'dtype': 'float32',
                'activation': 'linear',
                'trainable': true,
                'kernel_regularizer': null,
                'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                'units': 1,
                'batch_input_shape': [null, 3],
                'use_bias': true,
                'activity_regularizer': null
            }
        }],
    'backend': 'tensorflow'
};
const weightSpecs1 = [
    {
        name: 'dense/kernel',
        shape: [3, 1],
        dtype: 'float32',
    },
    {
        name: 'dense/bias',
        shape: [1],
        dtype: 'float32',
    }
];
const weightData1 = new ArrayBuffer(16);
const artifacts1 = {
    modelTopology: modelTopology1,
    weightSpecs: weightSpecs1,
    weightData: weightData1,
};
describeWithFlags('Passthrough Saver', BROWSER_ENVS, () => {
    it('passes provided arguments through on save', async () => {
        const testStartDate = new Date();
        let savedArtifacts = null;
        async function saveHandler(artifacts) {
            savedArtifacts = artifacts;
            return {
                modelArtifactsInfo: {
                    dateSaved: testStartDate,
                    modelTopologyType: 'JSON',
                    modelTopologyBytes: JSON.stringify(modelTopology1).length,
                    weightSpecsBytes: JSON.stringify(weightSpecs1).length,
                    weightDataBytes: weightData1.byteLength,
                }
            };
        }
        const saveTrigger = tf.io.withSaveHandler(saveHandler);
        const saveResult = await saveTrigger.save(artifacts1);
        expect(saveResult.errors).toEqual(undefined);
        const artifactsInfo = saveResult.modelArtifactsInfo;
        expect(artifactsInfo.dateSaved.getTime())
            .toBeGreaterThanOrEqual(testStartDate.getTime());
        expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
            .toEqual(JSON.stringify(modelTopology1).length);
        expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
            .toEqual(JSON.stringify(weightSpecs1).length);
        expect(saveResult.modelArtifactsInfo.weightDataBytes).toEqual(16);
        expect(savedArtifacts.modelTopology).toEqual(modelTopology1);
        expect(savedArtifacts.weightSpecs).toEqual(weightSpecs1);
        expect(savedArtifacts.weightData).toEqual(weightData1);
    });
});
describeWithFlags('Passthrough Loader', BROWSER_ENVS, () => {
    it('load topology and weights: legacy signature', async () => {
        const passthroughHandler = tf.io.fromMemory(modelTopology1, weightSpecs1, weightData1);
        const modelArtifacts = await passthroughHandler.load();
        expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
        expect(modelArtifacts.weightSpecs).toEqual(weightSpecs1);
        expect(modelArtifacts.weightData).toEqual(weightData1);
        expect(modelArtifacts.userDefinedMetadata).toEqual(undefined);
    });
    it('load topology and weights', async () => {
        const passthroughHandler = tf.io.fromMemory({
            modelTopology: modelTopology1,
            weightSpecs: weightSpecs1,
            weightData: weightData1
        });
        const modelArtifacts = await passthroughHandler.load();
        expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
        expect(modelArtifacts.weightSpecs).toEqual(weightSpecs1);
        expect(modelArtifacts.weightData).toEqual(weightData1);
        expect(modelArtifacts.userDefinedMetadata).toEqual(undefined);
    });
    it('load model topology only: legacy signature', async () => {
        const passthroughHandler = tf.io.fromMemory(modelTopology1);
        const modelArtifacts = await passthroughHandler.load();
        expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
        expect(modelArtifacts.weightSpecs).toEqual(undefined);
        expect(modelArtifacts.weightData).toEqual(undefined);
        expect(modelArtifacts.userDefinedMetadata).toEqual(undefined);
    });
    it('load model topology only', async () => {
        const passthroughHandler = tf.io.fromMemory({ modelTopology: modelTopology1 });
        const modelArtifacts = await passthroughHandler.load();
        expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
        expect(modelArtifacts.weightSpecs).toEqual(undefined);
        expect(modelArtifacts.weightData).toEqual(undefined);
        expect(modelArtifacts.userDefinedMetadata).toEqual(undefined);
    });
    it('load topology, weights, and user-defined metadata', async () => {
        const userDefinedMetadata = { 'fooField': 'fooValue' };
        const passthroughHandler = tf.io.fromMemory({
            modelTopology: modelTopology1,
            weightSpecs: weightSpecs1,
            weightData: weightData1,
            userDefinedMetadata
        });
        const modelArtifacts = await passthroughHandler.load();
        expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
        expect(modelArtifacts.weightSpecs).toEqual(weightSpecs1);
        expect(modelArtifacts.weightData).toEqual(weightData1);
        expect(modelArtifacts.userDefinedMetadata).toEqual(userDefinedMetadata);
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicGFzc3Rocm91Z2hfdGVzdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvaW8vcGFzc3Rocm91Z2hfdGVzdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSDs7R0FFRztBQUVILE9BQU8sS0FBSyxFQUFFLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxZQUFZLEVBQUUsaUJBQWlCLEVBQUMsTUFBTSxpQkFBaUIsQ0FBQztBQUVoRSxNQUFNLGNBQWMsR0FBTztJQUN6QixZQUFZLEVBQUUsWUFBWTtJQUMxQixlQUFlLEVBQUUsT0FBTztJQUN4QixRQUFRLEVBQUUsQ0FBQztZQUNULFlBQVksRUFBRSxPQUFPO1lBQ3JCLFFBQVEsRUFBRTtnQkFDUixvQkFBb0IsRUFBRTtvQkFDcEIsWUFBWSxFQUFFLGlCQUFpQjtvQkFDL0IsUUFBUSxFQUFFO3dCQUNSLGNBQWMsRUFBRSxTQUFTO3dCQUN6QixPQUFPLEVBQUUsR0FBRzt3QkFDWixNQUFNLEVBQUUsSUFBSTt3QkFDWixNQUFNLEVBQUUsU0FBUztxQkFDbEI7aUJBQ0Y7Z0JBQ0QsTUFBTSxFQUFFLE9BQU87Z0JBQ2YsbUJBQW1CLEVBQUUsSUFBSTtnQkFDekIsa0JBQWtCLEVBQUUsSUFBSTtnQkFDeEIsaUJBQWlCLEVBQUUsSUFBSTtnQkFDdkIsT0FBTyxFQUFFLFNBQVM7Z0JBQ2xCLFlBQVksRUFBRSxRQUFRO2dCQUN0QixXQUFXLEVBQUUsSUFBSTtnQkFDakIsb0JBQW9CLEVBQUUsSUFBSTtnQkFDMUIsa0JBQWtCLEVBQUUsRUFBQyxZQUFZLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBRSxFQUFFLEVBQUM7Z0JBQ3pELE9BQU8sRUFBRSxDQUFDO2dCQUNWLG1CQUFtQixFQUFFLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQztnQkFDOUIsVUFBVSxFQUFFLElBQUk7Z0JBQ2hCLHNCQUFzQixF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QyxDQUFDLE9BQU8sQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUNyRCxNQUFNLENBQUMsY0FBYyxDQUFDLG1CQUFtQixDQUFDLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxDQUFDO0lBQ2hFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLG1EQUFtRCxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ2pFLE1BQU0sbUJBQW1CLEdBQU8sRUFBQyxVQUFVLEVBQUUsVUFBVSxFQUFDLENBQUM7UUFDekQsTUFBTSxrQkFBa0IsR0FBRyxFQUFFLENBQUMsRUFBRSxDQUFDLFVBQVUsQ0FBQztZQUMxQyxhQUFhLEVBQUUsY0FBYztZQUM3QixXQUFXLEVBQUUsWUFBWTtZQUN6QixVQUFVLEVBQUUsV0FBVztZQUN2QixtQkFBbUI7U0FDcEIsQ0FBQyxDQUFDO1FBQ0gsTUFBTSxjQUFjLEdBQUcsTUFBTSxrQkFBa0IsQ0FBQyxJQUFJLEVBQUUsQ0FBQztRQUN2RCxNQUFNLENBQUMsY0FBYyxDQUFDLGFBQWEsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxjQUFjLENBQUMsQ0FBQztRQUM3RCxNQUFNLENBQUMsY0FBYyxDQUFDLFdBQVcsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxZQUFZLENBQUMsQ0FBQztRQUN6RCxNQUFNLENBQUMsY0FBYyxDQUFDLFVBQVUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxXQUFXLENBQUMsQ0FBQztRQUN2RCxNQUFNLENBQUMsY0FBYyxDQUFDLG1CQUFtQixDQUFDLENBQUMsT0FBTyxDQUFDLG1CQUFtQixDQUFDLENBQUM7SUFDMUUsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuLyoqXG4gKiBVbml0IHRlc3RzIGZvciBwYXNzdGhyb3VnaCBJT0hhbmRsZXJzLlxuICovXG5cbmltcG9ydCAqIGFzIHRmIGZyb20gJy4uL2luZGV4JztcbmltcG9ydCB7QlJPV1NFUl9FTlZTLCBkZXNjcmliZVdpdGhGbGFnc30gZnJvbSAnLi4vamFzbWluZV91dGlsJztcblxuY29uc3QgbW9kZWxUb3BvbG9neTE6IHt9ID0ge1xuICAnY2xhc3NfbmFtZSc6ICdTZXF1ZW50aWFsJyxcbiAgJ2tlcmFzX3ZlcnNpb24nOiAnMi4xLjQnLFxuICAnY29uZmlnJzogW3tcbiAgICAnY2xhc3NfbmFtZSc6ICdEZW5zZScsXG4gICAgJ2NvbmZpZyc6IHtcbiAgICAgICdrZXJuZWxfaW5pdGlhbGl6ZXInOiB7XG4gICAgICAgICdjbGFzc19uYW1lJzogJ1ZhcmlhbmNlU2NhbGluZycsXG4gICAgICAgICdjb25maWcnOiB7XG4gICAgICAgICAgJ2Rpc3RyaWJ1dGlvbic6ICd1bmlmb3JtJyxcbiAgICAgICAgICAnc2NhbGUnOiAxLjAsXG4gICAgICAgICAgJ3NlZWQnOiBudWxsLFxuICAgICAgICAgICdtb2RlJzogJ2Zhbl9hdmcnXG4gICAgICAgIH1cbiAgICAgIH0sXG4gICAgICAnbmFtZSc6ICdkZW5zZScsXG4gICAgICAna2VybmVsX2NvbnN0cmFpbnQnOiBudWxsLFxuICAgICAgJ2JpYXNfcmVndWxhcml6ZXInOiBudWxsLFxuICAgICAgJ2JpYXNfY29uc3RyYWludCc6IG51bGwsXG4gICAgICAnZHR5cGUnOiAnZmxvYXQzMicsXG4gICAgICAnYWN0aXZhdGlvbic6ICdsaW5lYXInLFxuICAgICAgJ3RyYWluYWJsZSc6IHRydWUsXG4gICAgICAna2VybmVsX3JlZ3VsYXJpemVyJzogbnVsbCxcbiAgICAgICdiaWFzX2luaXRpYWxpemVyJzogeydjbGFzc19uYW1lJzogJ1plcm9zJywgJ2NvbmZpZyc6IHt9fSxcbiAgICAgICd1bml0cyc6IDEsXG4gICAgICAnYmF0Y2hfaW5wdXRfc2hhcGUnOiBbbnVsbCwgM10sXG4gICAgICAndXNlX2JpYXMnOiB0cnVlLFxuICAgICAgJ2FjdGl2aXR5X3JlZ3VsYXJpemVyJzogbnVsbFxuICAgIH1cbiAgfV0sXG4gICdiYWNrZW5kJzogJ3RlbnNvcmZsb3cnXG59O1xuXG5jb25zdCB3ZWlnaHRTcGVjczE6IHRmLmlvLldlaWdodHNNYW5pZmVzdEVudHJ5W10gPSBbXG4gIHtcbiAgICBuYW1lOiAnZGVuc2Uva2VybmVsJyxcbiAgICBzaGFwZTogWzMsIDFdLFxuICAgIGR0eXBlOiAnZmxvYXQzMicsXG4gIH0sXG4gIHtcbiAgICBuYW1lOiAnZGVuc2UvYmlhcycsXG4gICAgc2hhcGU6IFsxXSxcbiAgICBkdHlwZTogJ2Zsb2F0MzInLFxuICB9XG5dO1xuXG5jb25zdCB3ZWlnaHREYXRhMSA9IG5ldyBBcnJheUJ1ZmZlcigxNik7XG5jb25zdCBhcnRpZmFjdHMxOiB0Zi5pby5Nb2RlbEFydGlmYWN0cyA9IHtcbiAgbW9kZWxUb3BvbG9neTogbW9kZWxUb3BvbG9neTEsXG4gIHdlaWdodFNwZWNzOiB3ZWlnaHRTcGVjczEsXG4gIHdlaWdodERhdGE6IHdlaWdodERhdGExLFxufTtcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ1Bhc3N0aHJvdWdoIFNhdmVyJywgQlJPV1NFUl9FTlZTLCAoKSA9PiB7XG4gIGl0KCdwYXNzZXMgcHJvdmlkZWQgYXJndW1lbnRzIHRocm91Z2ggb24gc2F2ZScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCB0ZXN0U3RhcnREYXRlID0gbmV3IERhdGUoKTtcbiAgICBsZXQgc2F2ZWRBcnRpZmFjdHM6IHRmLmlvLk1vZGVsQXJ0aWZhY3RzID0gbnVsbDtcblxuICAgIGFzeW5jIGZ1bmN0aW9uIHNhdmVIYW5kbGVyKGFydGlmYWN0czogdGYuaW8uTW9kZWxBcnRpZmFjdHMpOlxuICAgICAgICBQcm9taXNlPHRmLmlvLlNhdmVSZXN1bHQ+IHtcbiAgICAgIHNhdmVkQXJ0aWZhY3RzID0gYXJ0aWZhY3RzO1xuICAgICAgcmV0dXJuIHtcbiAgICAgICAgbW9kZWxBcnRpZmFjdHNJbmZvOiB7XG4gICAgICAgICAgZGF0ZVNhdmVkOiB0ZXN0U3RhcnREYXRlLFxuICAgICAgICAgIG1vZGVsVG9wb2xvZ3lUeXBlOiAnSlNPTicsXG4gICAgICAgICAgbW9kZWxUb3BvbG9neUJ5dGVzOiBKU09OLnN0cmluZ2lmeShtb2RlbFRvcG9sb2d5MSkubGVuZ3RoLFxuICAgICAgICAgIHdlaWdodFNwZWNzQnl0ZXM6IEpTT04uc3RyaW5naWZ5KHdlaWdodFNwZWNzMSkubGVuZ3RoLFxuICAgICAgICAgIHdlaWdodERhdGFCeXRlczogd2VpZ2h0RGF0YTEuYnl0ZUxlbmd0aCxcbiAgICAgICAgfVxuICAgICAgfTtcbiAgICB9XG5cbiAgICBjb25zdCBzYXZlVHJpZ2dlciA9IHRmLmlvLndpdGhTYXZlSGFuZGxlcihzYXZlSGFuZGxlcik7XG4gICAgY29uc3Qgc2F2ZVJlc3VsdCA9IGF3YWl0IHNhdmVUcmlnZ2VyLnNhdmUoYXJ0aWZhY3RzMSk7XG5cbiAgICBleHBlY3Qoc2F2ZVJlc3VsdC5lcnJvcnMpLnRvRXF1YWwodW5kZWZpbmVkKTtcbiAgICBjb25zdCBhcnRpZmFjdHNJbmZvID0gc2F2ZVJlc3VsdC5tb2RlbEFydGlmYWN0c0luZm87XG4gICAgZXhwZWN0KGFydGlmYWN0c0luZm8uZGF0ZVNhdmVkLmdldFRpbWUoKSlcbiAgICAgICAgLnRvQmVHcmVhdGVyVGhhbk9yRXF1YWwodGVzdFN0YXJ0RGF0ZS5nZXRUaW1lKCkpO1xuICAgIGV4cGVjdChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby5tb2RlbFRvcG9sb2d5Qnl0ZXMpXG4gICAgICAgIC50b0VxdWFsKEpTT04uc3RyaW5naWZ5KG1vZGVsVG9wb2xvZ3kxKS5sZW5ndGgpO1xuICAgIGV4cGVjdChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby53ZWlnaHRTcGVjc0J5dGVzKVxuICAgICAgICAudG9FcXVhbChKU09OLnN0cmluZ2lmeSh3ZWlnaHRTcGVjczEpLmxlbmd0aCk7XG4gICAgZXhwZWN0KHNhdmVSZXN1bHQubW9kZWxBcnRpZmFjdHNJbmZvLndlaWdodERhdGFCeXRlcykudG9FcXVhbCgxNik7XG5cbiAgICBleHBlY3Qoc2F2ZWRBcnRpZmFjdHMubW9kZWxUb3BvbG9neSkudG9FcXVhbChtb2RlbFRvcG9sb2d5MSk7XG4gICAgZXhwZWN0KHNhdmVkQXJ0aWZhY3RzLndlaWdodFNwZWNzKS50b0VxdWFsKHdlaWdodFNwZWNzMSk7XG4gICAgZXhwZWN0KHNhdmVkQXJ0aWZhY3RzLndlaWdodERhdGEpLnRvRXF1YWwod2VpZ2h0RGF0YTEpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZVdpdGhGbGFncygnUGFzc3Rocm91Z2ggTG9hZGVyJywgQlJPV1NFUl9FTlZTLCAoKSA9PiB7XG4gIGl0KCdsb2FkIHRvcG9sb2d5IGFuZCB3ZWlnaHRzOiBsZWdhY3kgc2lnbmF0dXJlJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHBhc3N0aHJvdWdoSGFuZGxlciA9XG4gICAgICAgIHRmLmlvLmZyb21NZW1vcnkobW9kZWxUb3BvbG9neTEsIHdlaWdodFNwZWNzMSwgd2VpZ2h0RGF0YTEpO1xuICAgIGNvbnN0IG1vZGVsQXJ0aWZhY3RzID0gYXdhaXQgcGFzc3Rocm91Z2hIYW5kbGVyLmxvYWQoKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMubW9kZWxUb3BvbG9neSkudG9FcXVhbChtb2RlbFRvcG9sb2d5MSk7XG4gICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzKS50b0VxdWFsKHdlaWdodFNwZWNzMSk7XG4gICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEpLnRvRXF1YWwod2VpZ2h0RGF0YTEpO1xuICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy51c2VyRGVmaW5lZE1ldGFkYXRhKS50b0VxdWFsKHVuZGVmaW5lZCk7XG4gIH0pO1xuXG4gIGl0KCdsb2FkIHRvcG9sb2d5IGFuZCB3ZWlnaHRzJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHBhc3N0aHJvdWdoSGFuZGxlciA9IHRmLmlvLmZyb21NZW1vcnkoe1xuICAgICAgbW9kZWxUb3BvbG9neTogbW9kZWxUb3BvbG9neTEsXG4gICAgICB3ZWlnaHRTcGVjczogd2VpZ2h0U3BlY3MxLFxuICAgICAgd2VpZ2h0RGF0YTogd2VpZ2h0RGF0YTFcbiAgICB9KTtcbiAgICBjb25zdCBtb2RlbEFydGlmYWN0cyA9IGF3YWl0IHBhc3N0aHJvdWdoSGFuZGxlci5sb2FkKCk7XG4gICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kpLnRvRXF1YWwobW9kZWxUb3BvbG9neTEpO1xuICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy53ZWlnaHRTcGVjcykudG9FcXVhbCh3ZWlnaHRTcGVjczEpO1xuICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy53ZWlnaHREYXRhKS50b0VxdWFsKHdlaWdodERhdGExKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMudXNlckRlZmluZWRNZXRhZGF0YSkudG9FcXVhbCh1bmRlZmluZWQpO1xuICB9KTtcblxuICBpdCgnbG9hZCBtb2RlbCB0b3BvbG9neSBvbmx5OiBsZWdhY3kgc2lnbmF0dXJlJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHBhc3N0aHJvdWdoSGFuZGxlciA9IHRmLmlvLmZyb21NZW1vcnkobW9kZWxUb3BvbG9neTEpO1xuICAgIGNvbnN0IG1vZGVsQXJ0aWZhY3RzID0gYXdhaXQgcGFzc3Rocm91Z2hIYW5kbGVyLmxvYWQoKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMubW9kZWxUb3BvbG9neSkudG9FcXVhbChtb2RlbFRvcG9sb2d5MSk7XG4gICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzKS50b0VxdWFsKHVuZGVmaW5lZCk7XG4gICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEpLnRvRXF1YWwodW5kZWZpbmVkKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMudXNlckRlZmluZWRNZXRhZGF0YSkudG9FcXVhbCh1bmRlZmluZWQpO1xuICB9KTtcblxuICBpdCgnbG9hZCBtb2RlbCB0b3BvbG9neSBvbmx5JywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHBhc3N0aHJvdWdoSGFuZGxlciA9XG4gICAgICAgIHRmLmlvLmZyb21NZW1vcnkoe21vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxfSk7XG4gICAgY29uc3QgbW9kZWxBcnRpZmFjdHMgPSBhd2FpdCBwYXNzdGhyb3VnaEhhbmRsZXIubG9hZCgpO1xuICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5KS50b0VxdWFsKG1vZGVsVG9wb2xvZ3kxKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MpLnRvRXF1YWwodW5kZWZpbmVkKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YSkudG9FcXVhbCh1bmRlZmluZWQpO1xuICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy51c2VyRGVmaW5lZE1ldGFkYXRhKS50b0VxdWFsKHVuZGVmaW5lZCk7XG4gIH0pO1xuXG4gIGl0KCdsb2FkIHRvcG9sb2d5LCB3ZWlnaHRzLCBhbmQgdXNlci1kZWZpbmVkIG1ldGFkYXRhJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHVzZXJEZWZpbmVkTWV0YWRhdGE6IHt9ID0geydmb29GaWVsZCc6ICdmb29WYWx1ZSd9O1xuICAgIGNvbnN0IHBhc3N0aHJvdWdoSGFuZGxlciA9IHRmLmlvLmZyb21NZW1vcnkoe1xuICAgICAgbW9kZWxUb3BvbG9neTogbW9kZWxUb3BvbG9neTEsXG4gICAgICB3ZWlnaHRTcGVjczogd2VpZ2h0U3BlY3MxLFxuICAgICAgd2VpZ2h0RGF0YTogd2VpZ2h0RGF0YTEsXG4gICAgICB1c2VyRGVmaW5lZE1ldGFkYXRhXG4gICAgfSk7XG4gICAgY29uc3QgbW9kZWxBcnRpZmFjdHMgPSBhd2FpdCBwYXNzdGhyb3VnaEhhbmRsZXIubG9hZCgpO1xuICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5KS50b0VxdWFsKG1vZGVsVG9wb2xvZ3kxKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MpLnRvRXF1YWwod2VpZ2h0U3BlY3MxKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YSkudG9FcXVhbCh3ZWlnaHREYXRhMSk7XG4gICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLnVzZXJEZWZpbmVkTWV0YWRhdGEpLnRvRXF1YWwodXNlckRlZmluZWRNZXRhZGF0YSk7XG4gIH0pO1xufSk7XG4iXX0=