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
import * as tf from '../index';
import { BROWSER_ENVS, describeWithFlags } from '../jasmine_util';
import { expectArraysClose, expectArraysEqual } from '../test_util';
describeWithFlags('loadWeights', BROWSER_ENVS, () => {
    const setupFakeWeightFiles = (fileBufferMap) => {
        spyOn(tf.env().platform, 'fetch').and.callFake((path) => {
            return new Response(fileBufferMap[path], { headers: { 'Content-type': 'application/octet-stream' } });
        });
    };
    it('1 group, 1 weight, 1 requested weight', async () => {
        setupFakeWeightFiles({ './weightfile0': new Float32Array([1, 2, 3]) });
        const manifest = [{
                'paths': ['weightfile0'],
                'weights': [{ 'name': 'weight0', 'dtype': 'float32', 'shape': [3] }]
            }];
        const weightsNamesToFetch = ['weight0'];
        const weights = await tf.io.loadWeights(manifest, './', weightsNamesToFetch);
        expect(tf.env().platform.fetch.calls.count()).toBe(1);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(weightsNamesToFetch.length);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [1, 2, 3]);
        expect(weight0.shape).toEqual([3]);
        expect(weight0.dtype).toEqual('float32');
    });
    it('1 group, 2 weights, fetch 1st weight', async () => {
        setupFakeWeightFiles({ './weightfile0': new Float32Array([1, 2, 3, 4, 5]) });
        const manifest = [{
                'paths': ['weightfile0'],
                'weights': [
                    { 'name': 'weight0', 'dtype': 'float32', 'shape': [2] },
                    { 'name': 'weight1', 'dtype': 'float32', 'shape': [3] }
                ]
            }];
        // Load the first weight.
        const weights = await tf.io.loadWeights(manifest, './', ['weight0']);
        expect(tf.env().platform.fetch.calls.count()).toBe(1);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(1);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [1, 2]);
        expect(weight0.shape).toEqual([2]);
        expect(weight0.dtype).toEqual('float32');
    });
    it('1 group, 2 weights, fetch 2nd weight', async () => {
        setupFakeWeightFiles({ './weightfile0': new Float32Array([1, 2, 3, 4, 5]) });
        const manifest = [{
                'paths': ['weightfile0'],
                'weights': [
                    { 'name': 'weight0', 'dtype': 'float32', 'shape': [2] },
                    { 'name': 'weight1', 'dtype': 'float32', 'shape': [3] }
                ]
            }];
        // Load the second weight.
        const weights = await tf.io.loadWeights(manifest, './', ['weight1']);
        expect(tf.env().platform.fetch.calls.count()).toBe(1);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(1);
        const weight1 = weights['weight1'];
        expectArraysClose(await weight1.data(), [3, 4, 5]);
        expect(weight1.shape).toEqual([3]);
        expect(weight1.dtype).toEqual('float32');
    });
    it('1 group, 2 weights, fetch all weights', async () => {
        setupFakeWeightFiles({ './weightfile0': new Float32Array([1, 2, 3, 4, 5]) });
        const manifest = [{
                'paths': ['weightfile0'],
                'weights': [
                    { 'name': 'weight0', 'dtype': 'float32', 'shape': [2] },
                    { 'name': 'weight1', 'dtype': 'float32', 'shape': [3] }
                ]
            }];
        // Load all weights.
        const weights = await tf.io.loadWeights(manifest, './', ['weight0', 'weight1']);
        expect(tf.env().platform.fetch.calls.count()).toBe(1);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(2);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [1, 2]);
        expect(weight0.shape).toEqual([2]);
        expect(weight0.dtype).toEqual('float32');
        const weight1 = weights['weight1'];
        expectArraysClose(await weight1.data(), [3, 4, 5]);
        expect(weight1.shape).toEqual([3]);
        expect(weight1.dtype).toEqual('float32');
    });
    it('1 group, multiple weights, different dtypes', async () => {
        const buffer = new ArrayBuffer(5 * 4 + 1);
        const view = new DataView(buffer);
        view.setInt32(0, 1, true);
        view.setInt32(4, 2, true);
        view.setUint8(8, 1);
        view.setFloat32(9, 3., true);
        view.setFloat32(13, 4., true);
        view.setFloat32(17, 5., true);
        setupFakeWeightFiles({ './weightfile0': buffer });
        const manifest = [{
                'paths': ['weightfile0'],
                'weights': [
                    { 'name': 'weight0', 'dtype': 'int32', 'shape': [2] },
                    { 'name': 'weight1', 'dtype': 'bool', 'shape': [] },
                    { 'name': 'weight2', 'dtype': 'float32', 'shape': [3] },
                ]
            }];
        // Load all weights.
        const weights = await tf.io.loadWeights(manifest, './', ['weight0', 'weight1', 'weight2']);
        expect(tf.env().platform.fetch.calls.count()).toBe(1);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(3);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [1, 2]);
        expect(weight0.shape).toEqual([2]);
        expect(weight0.dtype).toEqual('int32');
        const weight1 = weights['weight1'];
        expectArraysClose(await weight1.data(), [1]);
        expect(weight1.shape).toEqual([]);
        expect(weight1.dtype).toEqual('bool');
        const weight2 = weights['weight2'];
        expectArraysClose(await weight2.data(), [3, 4, 5]);
        expect(weight2.shape).toEqual([3]);
        expect(weight2.dtype).toEqual('float32');
    });
    it('1 group, sharded 1 weight across multiple files', async () => {
        const shard0 = new Float32Array([1, 2, 3, 4, 5]);
        const shard1 = new Float32Array([1.1, 2.2]);
        const shard2 = new Float32Array([10, 20, 30]);
        setupFakeWeightFiles({
            './weightfile0': shard0,
            './weightsfile1': shard1,
            './weightsfile2': shard2
        });
        const manifest = [{
                'paths': ['weightfile0', 'weightsfile1', 'weightsfile2'],
                'weights': [{ 'name': 'weight0', 'dtype': 'float32', 'shape': [5, 2] }]
            }];
        const weights = await tf.io.loadWeights(manifest, './', ['weight0']);
        expect(tf.env().platform.fetch.calls.count()).toBe(3);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(1);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [1, 2, 3, 4, 5, 1.1, 2.2, 10, 20, 30]);
        expect(weight0.shape).toEqual([5, 2]);
        expect(weight0.dtype).toEqual('float32');
    });
    it('1 group, sharded 2 weights across multiple files', async () => {
        const shard0 = new Int32Array([1, 2, 3, 4, 5]);
        // shard1 contains part of the first weight and part of the second.
        const shard1 = new ArrayBuffer(5 * 4);
        const intBuffer = new Int32Array(shard1, 0, 2);
        intBuffer.set([10, 20]);
        const floatBuffer = new Float32Array(shard1, intBuffer.byteLength, 3);
        floatBuffer.set([3.0, 4.0, 5.0]);
        const shard2 = new Float32Array([10, 20, 30]);
        setupFakeWeightFiles({
            './weightfile0': shard0,
            './weightsfile1': shard1,
            './weightsfile2': shard2
        });
        const manifest = [{
                'paths': ['weightfile0', 'weightsfile1', 'weightsfile2'],
                'weights': [
                    { 'name': 'weight0', 'dtype': 'int32', 'shape': [7, 1] },
                    { 'name': 'weight1', 'dtype': 'float32', 'shape': [3, 2] }
                ]
            }];
        const weights = await tf.io.loadWeights(manifest, './', ['weight0', 'weight1']);
        expect(tf.env().platform.fetch.calls.count()).toBe(3);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(2);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [1, 2, 3, 4, 5, 10, 20]);
        expect(weight0.shape).toEqual([7, 1]);
        expect(weight0.dtype).toEqual('int32');
        const weight1 = weights['weight1'];
        expectArraysClose(await weight1.data(), [3.0, 4.0, 5.0, 10, 20, 30]);
        expect(weight1.shape).toEqual([3, 2]);
        expect(weight1.dtype).toEqual('float32');
    });
    it('2 group, 4 weights, fetches one group', async () => {
        setupFakeWeightFiles({
            './weightfile0': new Float32Array([1, 2, 3, 4, 5]),
            './weightfile1': new Float32Array([6, 7, 8, 9])
        });
        const manifest = [
            {
                'paths': ['weightfile0'],
                'weights': [
                    { 'name': 'weight0', 'dtype': 'float32', 'shape': [2] },
                    { 'name': 'weight1', 'dtype': 'float32', 'shape': [3] }
                ]
            },
            {
                'paths': ['weightfile1'],
                'weights': [
                    { 'name': 'weight2', 'dtype': 'float32', 'shape': [3, 1] },
                    { 'name': 'weight3', 'dtype': 'float32', 'shape': [] }
                ]
            }
        ];
        const weights = await tf.io.loadWeights(manifest, './', ['weight0', 'weight1']);
        // Only the first group should be fetched.
        expect(tf.env().platform.fetch.calls.count()).toBe(1);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(2);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [1, 2]);
        expect(weight0.shape).toEqual([2]);
        expect(weight0.dtype).toEqual('float32');
        const weight1 = weights['weight1'];
        expectArraysClose(await weight1.data(), [3, 4, 5]);
        expect(weight1.shape).toEqual([3]);
        expect(weight1.dtype).toEqual('float32');
    });
    it('2 group, 4 weights, one weight from each group', async () => {
        setupFakeWeightFiles({
            './weightfile0': new Float32Array([1, 2, 3, 4, 5]),
            './weightfile1': new Float32Array([6, 7, 8, 9])
        });
        const manifest = [
            {
                'paths': ['weightfile0'],
                'weights': [
                    { 'name': 'weight0', 'dtype': 'float32', 'shape': [2] },
                    { 'name': 'weight1', 'dtype': 'float32', 'shape': [3] }
                ]
            },
            {
                'paths': ['weightfile1'],
                'weights': [
                    { 'name': 'weight2', 'dtype': 'float32', 'shape': [3, 1] },
                    { 'name': 'weight3', 'dtype': 'float32', 'shape': [] }
                ]
            }
        ];
        const weights = await tf.io.loadWeights(manifest, './', ['weight0', 'weight2']);
        // Both groups need to be fetched.
        expect(tf.env().platform.fetch.calls.count()).toBe(2);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(2);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [1, 2]);
        expect(weight0.shape).toEqual([2]);
        expect(weight0.dtype).toEqual('float32');
        const weight2 = weights['weight2'];
        expectArraysClose(await weight2.data(), [6, 7, 8]);
        expect(weight2.shape).toEqual([3, 1]);
        expect(weight2.dtype).toEqual('float32');
    });
    it('2 group, 4 weights, dont specify weights fetchs all', async () => {
        setupFakeWeightFiles({
            './weightfile0': new Float32Array([1, 2, 3, 4, 5]),
            './weightfile1': new Float32Array([6, 7, 8, 9])
        });
        const manifest = [
            {
                'paths': ['weightfile0'],
                'weights': [
                    { 'name': 'weight0', 'dtype': 'float32', 'shape': [2] },
                    { 'name': 'weight1', 'dtype': 'float32', 'shape': [3] }
                ]
            },
            {
                'paths': ['weightfile1'],
                'weights': [
                    { 'name': 'weight2', 'dtype': 'float32', 'shape': [3, 1] },
                    { 'name': 'weight3', 'dtype': 'float32', 'shape': [] }
                ]
            }
        ];
        // Don't pass a third argument to loadWeights to load all weights.
        const weights = await tf.io.loadWeights(manifest, './');
        // Both groups need to be fetched.
        expect(tf.env().platform.fetch.calls.count()).toBe(2);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(4);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [1, 2]);
        expect(weight0.shape).toEqual([2]);
        expect(weight0.dtype).toEqual('float32');
        const weight1 = weights['weight1'];
        expectArraysClose(await weight1.data(), [3, 4, 5]);
        expect(weight1.shape).toEqual([3]);
        expect(weight1.dtype).toEqual('float32');
        const weight2 = weights['weight2'];
        expectArraysClose(await weight2.data(), [6, 7, 8]);
        expect(weight2.shape).toEqual([3, 1]);
        expect(weight2.dtype).toEqual('float32');
        const weight3 = weights['weight3'];
        expectArraysClose(await weight3.data(), [9]);
        expect(weight3.shape).toEqual([]);
        expect(weight3.dtype).toEqual('float32');
    });
    it('throws if requested weight not found', async () => {
        setupFakeWeightFiles({ './weightfile0': new Float32Array([1, 2, 3]) });
        const manifest = [{
                'paths': ['weightfile0'],
                'weights': [{ 'name': 'weight0', 'dtype': 'float32', 'shape': [3] }]
            }];
        const weightsNamesToFetch = ['doesntexist'];
        try {
            await tf.io.loadWeights(manifest, './', weightsNamesToFetch);
            fail();
        }
        catch (e) {
            expect(e.message).toContain('Could not find weights');
        }
    });
    it('throws if requested weight has unknown dtype', async () => {
        setupFakeWeightFiles({ './weightfile0': new Float32Array([1, 2, 3]) });
        const manifest = [{
                'paths': ['weightfile0'],
                'weights': [{
                        'name': 'weight0',
                        // tslint:disable-next-line:no-any
                        'dtype': 'null',
                        'shape': [3]
                    }]
            }];
        const weightsNamesToFetch = ['weight0'];
        try {
            await tf.io.loadWeights(manifest, './', weightsNamesToFetch);
            fail();
        }
        catch (e) {
            expect(e.message).toContain('Unsupported dtype');
        }
    });
    it('should use request option', async () => {
        setupFakeWeightFiles({ './weightfile0': new Float32Array([1, 2, 3]) });
        const manifest = [{
                'paths': ['weightfile0'],
                'weights': [{ 'name': 'weight0', 'dtype': 'float32', 'shape': [3] }]
            }];
        const weightsNamesToFetch = ['weight0'];
        await tf.io.loadWeights(manifest, './', weightsNamesToFetch, { credentials: 'include' });
        expect(tf.env().platform.fetch.calls.count()).toBe(1);
        expect(tf.env().platform.fetch)
            .toHaveBeenCalledWith('./weightfile0', { credentials: 'include' }, { isBinary: true });
    });
    const quantizationTest = async (quantizationDtype) => {
        const arrayType = quantizationDtype === 'uint8' ? Uint8Array : Uint16Array;
        setupFakeWeightFiles({ './weightfile0': new arrayType([0, 48, 255, 0, 48, 255]) });
        const manifest = [{
                'paths': ['weightfile0'],
                'weights': [
                    {
                        'name': 'weight0',
                        'dtype': 'float32',
                        'shape': [3],
                        'quantization': { 'min': -1, 'scale': 0.1, 'dtype': quantizationDtype }
                    },
                    {
                        'name': 'weight1',
                        'dtype': 'int32',
                        'shape': [3],
                        'quantization': { 'min': -1, 'scale': 0.1, 'dtype': quantizationDtype }
                    }
                ]
            }];
        const weightsNamesToFetch = ['weight0', 'weight1'];
        const weights = await tf.io.loadWeights(manifest, './', weightsNamesToFetch);
        expect(tf.env().platform.fetch.calls.count()).toBe(1);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(weightsNamesToFetch.length);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [-1, 3.8, 24.5]);
        expect(weight0.shape).toEqual([3]);
        expect(weight0.dtype).toEqual('float32');
        const weight1 = weights['weight1'];
        expectArraysEqual(await weight1.data(), [-1, 4, 25]);
        expect(weight1.shape).toEqual([3]);
        expect(weight1.dtype).toEqual('int32');
    };
    it('quantized weights (uint8)', async () => {
        await quantizationTest('uint8');
    });
    it('quantized weights (uint16)', async () => {
        await quantizationTest('uint16');
    });
    it('2 groups, 1 quantized, 1 unquantized', async () => {
        setupFakeWeightFiles({
            './weightfile0': new Uint8Array([0, 48, 255, 0, 48, 255]),
            './weightfile1': new Float32Array([6, 7, 8, 9])
        });
        const manifest = [
            {
                'paths': ['weightfile0'],
                'weights': [
                    {
                        'name': 'weight0',
                        'dtype': 'float32',
                        'shape': [3],
                        'quantization': { 'min': -1, 'scale': 0.1, 'dtype': 'uint8' }
                    },
                    {
                        'name': 'weight1',
                        'dtype': 'int32',
                        'shape': [3],
                        'quantization': { 'min': -1, 'scale': 0.1, 'dtype': 'uint8' }
                    }
                ]
            },
            {
                'paths': ['weightfile1'],
                'weights': [
                    { 'name': 'weight2', 'dtype': 'float32', 'shape': [3, 1] },
                    { 'name': 'weight3', 'dtype': 'float32', 'shape': [] }
                ]
            }
        ];
        const weights = await tf.io.loadWeights(manifest, './', ['weight0', 'weight2']);
        // Both groups need to be fetched.
        expect(tf.env().platform.fetch.calls.count()).toBe(2);
        const weightNames = Object.keys(weights);
        expect(weightNames.length).toEqual(2);
        const weight0 = weights['weight0'];
        expectArraysClose(await weight0.data(), [-1, 3.8, 24.5]);
        expect(weight0.shape).toEqual([3]);
        expect(weight0.dtype).toEqual('float32');
        const weight2 = weights['weight2'];
        expectArraysClose(await weight2.data(), [6, 7, 8]);
        expect(weight2.shape).toEqual([3, 1]);
        expect(weight2.dtype).toEqual('float32');
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoid2VpZ2h0c19sb2FkZXJfdGVzdC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvaW8vd2VpZ2h0c19sb2FkZXJfdGVzdC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLEtBQUssRUFBRSxNQUFNLFVBQVUsQ0FBQztBQUMvQixPQUFPLEVBQUMsWUFBWSxFQUFFLGlCQUFpQixFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFDaEUsT0FBTyxFQUFDLGlCQUFpQixFQUFFLGlCQUFpQixFQUFDLE1BQU0sY0FBYyxDQUFDO0FBR2xFLGlCQUFpQixDQUFDLGFBQWEsRUFBRSxZQUFZLEVBQUUsR0FBRyxFQUFFO0lBQ2xELE1BQU0sb0JBQW9CLEdBQUcsQ0FBQyxhQUc3QixFQUFFLEVBQUU7UUFDSCxLQUFLLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLFFBQVEsRUFBRSxPQUFPLENBQUMsQ0FBQyxHQUFHLENBQUMsUUFBUSxDQUFDLENBQUMsSUFBWSxFQUFFLEVBQUU7WUFDOUQsT0FBTyxJQUFJLFFBQVEsQ0FDZixhQUFhLENBQUMsSUFBSSxDQUFDLEVBQ25CLEVBQUMsT0FBTyxFQUFFLEVBQUMsY0FBYyxFQUFFLDBCQUEwQixFQUFDLEVBQUMsQ0FBQyxDQUFDO1FBQy9ELENBQUMsQ0FBQyxDQUFDO0lBQ0wsQ0FBQyxDQUFDO0lBRUYsRUFBRSxDQUFDLHVDQUF1QyxFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3JELG9CQUFvQixDQUFDLEVBQUMsZUFBZSxFQUFFLElBQUksWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFDLENBQUMsQ0FBQztRQUVyRSxNQUFNLFFBQVEsR0FBMEIsQ0FBQztnQkFDdkMsT0FBTyxFQUFFLENBQUMsYUFBYSxDQUFDO2dCQUN4QixTQUFTLEVBQUUsQ0FBQyxFQUFDLE1B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QVMsQ0FBQyxDQUFDO1FBRXpDLE1BQU0sT0FBTyxHQUFHLE9BQU8sQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUNuQyxpQkFBaUIsQ0FBQyxNQUFNLE9BQU8sQ0FBQyxJQUFJLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNuRCxNQUFNLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3RDLE1BQU0sQ0FBQyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsT0FBTyxDQUFDLFNBQVMsQ0FBQyxDQUFDO0lBQzNDLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQgKiBhcyB0ZiBmcm9tICcuLi9pbmRleCc7XG5pbXBvcnQge0JST1dTRVJfRU5WUywgZGVzY3JpYmVXaXRoRmxhZ3N9IGZyb20gJy4uL2phc21pbmVfdXRpbCc7XG5pbXBvcnQge2V4cGVjdEFycmF5c0Nsb3NlLCBleHBlY3RBcnJheXNFcXVhbH0gZnJvbSAnLi4vdGVzdF91dGlsJztcbmltcG9ydCB7V2VpZ2h0c01hbmlmZXN0Q29uZmlnfSBmcm9tICcuL3R5cGVzJztcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ2xvYWRXZWlnaHRzJywgQlJPV1NFUl9FTlZTLCAoKSA9PiB7XG4gIGNvbnN0IHNldHVwRmFrZVdlaWdodEZpbGVzID0gKGZpbGVCdWZmZXJNYXA6IHtcbiAgICBbZmlsZW5hbWU6IHN0cmluZ106IEZsb2F0MzJBcnJheXxJbnQzMkFycmF5fEFycmF5QnVmZmVyfFVpbnQ4QXJyYXl8XG4gICAgVWludDE2QXJyYXlcbiAgfSkgPT4ge1xuICAgIHNweU9uKHRmLmVudigpLnBsYXRmb3JtLCAnZmV0Y2gnKS5hbmQuY2FsbEZha2UoKHBhdGg6IHN0cmluZykgPT4ge1xuICAgICAgcmV0dXJuIG5ldyBSZXNwb25zZShcbiAgICAgICAgICBmaWxlQnVmZmVyTWFwW3BhdGhdLFxuICAgICAgICAgIHtoZWFkZXJzOiB7J0NvbnRlbnQtdHlwZSc6ICdhcHBsaWNhdGlvbi9vY3RldC1zdHJlYW0nfX0pO1xuICAgIH0pO1xuICB9O1xuXG4gIGl0KCcxIGdyb3VwLCAxIHdlaWdodCwgMSByZXF1ZXN0ZWQgd2VpZ2h0JywgYXN5bmMgKCkgPT4ge1xuICAgIHNldHVwRmFrZVdlaWdodEZpbGVzKHsnLi93ZWlnaHRmaWxlMCc6IG5ldyBGbG9hdDMyQXJyYXkoWzEsIDIsIDNdKX0pO1xuXG4gICAgY29uc3QgbWFuaWZlc3Q6IFdlaWdodHNNYW5pZmVzdENvbmZpZyA9IFt7XG4gICAgICAncGF0aHMnOiBbJ3dlaWdodGZpbGUwJ10sXG4gICAgICAnd2VpZ2h0cyc6IFt7J25hbWUnOiAnd2VpZ2h0MCcsICdkdHlwZSc6ICdmbG9hdDMyJywgJ3NoYXBlJzogWzNdfV1cbiAgICB9XTtcblxuICAgIGNvbnN0IHdlaWdodHNOYW1lc1RvRmV0Y2ggPSBbJ3dlaWdodDAnXTtcbiAgICBjb25zdCB3ZWlnaHRzID1cbiAgICAgICAgYXdhaXQgdGYuaW8ubG9hZFdlaWdodHMobWFuaWZlc3QsICcuLycsIHdlaWdodHNOYW1lc1RvRmV0Y2gpO1xuICAgIGV4cGVjdCgodGYuZW52KCkucGxhdGZvcm0uZmV0Y2ggYXMgamFzbWluZS5TcHkpLmNhbGxzLmNvdW50KCkpLnRvQmUoMSk7XG5cbiAgICBjb25zdCB3ZWlnaHROYW1lcyA9IE9iamVjdC5rZXlzKHdlaWdodHMpO1xuICAgIGV4cGVjdCh3ZWlnaHROYW1lcy5sZW5ndGgpLnRvRXF1YWwod2VpZ2h0c05hbWVzVG9GZXRjaC5sZW5ndGgpO1xuXG4gICAgY29uc3Qgd2VpZ2h0MCA9IHdlaWdodHNbJ3dlaWdodDAnXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB3ZWlnaHQwLmRhdGEoKSwgWzEsIDIsIDNdKTtcbiAgICBleHBlY3Qod2VpZ2h0MC5zaGFwZSkudG9FcXVhbChbM10pO1xuICAgIGV4cGVjdCh3ZWlnaHQwLmR0eXBlKS50b0VxdWFsKCdmbG9hdDMyJyk7XG4gIH0pO1xuXG4gIGl0KCcxIGdyb3VwLCAyIHdlaWdodHMsIGZldGNoIDFzdCB3ZWlnaHQnLCBhc3luYyAoKSA9PiB7XG4gICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoeycuL3dlaWdodGZpbGUwJzogbmV3IEZsb2F0MzJBcnJheShbMSwgMiwgMywgNCwgNV0pfSk7XG5cbiAgICBjb25zdCBtYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgICdwYXRocyc6IFsnd2VpZ2h0ZmlsZTAnXSxcbiAgICAgICd3ZWlnaHRzJzogW1xuICAgICAgICB7J25hbWUnOiAnd2VpZ2h0MCcsICdkdHlwZSc6ICdmbG9hdDMyJywgJ3NoYXBlJzogWzJdfSxcbiAgICAgICAgeyduYW1lJzogJ3dlaWdodDEnLCAnZHR5cGUnOiAnZmxvYXQzMicsICdzaGFwZSc6IFszXX1cbiAgICAgIF1cbiAgICB9XTtcblxuICAgIC8vIExvYWQgdGhlIGZpcnN0IHdlaWdodC5cbiAgICBjb25zdCB3ZWlnaHRzID0gYXdhaXQgdGYuaW8ubG9hZFdlaWdodHMobWFuaWZlc3QsICcuLycsIFsnd2VpZ2h0MCddKTtcbiAgICBleHBlY3QoKHRmLmVudigpLnBsYXRmb3JtLmZldGNoIGFzIGphc21pbmUuU3B5KS5jYWxscy5jb3VudCgpKS50b0JlKDEpO1xuXG4gICAgY29uc3Qgd2VpZ2h0TmFtZXMgPSBPYmplY3Qua2V5cyh3ZWlnaHRzKTtcbiAgICBleHBlY3Qod2VpZ2h0TmFtZXMubGVuZ3RoKS50b0VxdWFsKDEpO1xuXG4gICAgY29uc3Qgd2VpZ2h0MCA9IHdlaWdodHNbJ3dlaWdodDAnXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB3ZWlnaHQwLmRhdGEoKSwgWzEsIDJdKTtcbiAgICBleHBlY3Qod2VpZ2h0MC5zaGFwZSkudG9FcXVhbChbMl0pO1xuICAgIGV4cGVjdCh3ZWlnaHQwLmR0eXBlKS50b0VxdWFsKCdmbG9hdDMyJyk7XG4gIH0pO1xuXG4gIGl0KCcxIGdyb3VwLCAyIHdlaWdodHMsIGZldGNoIDJuZCB3ZWlnaHQnLCBhc3luYyAoKSA9PiB7XG4gICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoeycuL3dlaWdodGZpbGUwJzogbmV3IEZsb2F0MzJBcnJheShbMSwgMiwgMywgNCwgNV0pfSk7XG5cbiAgICBjb25zdCBtYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgICdwYXRocyc6IFsnd2VpZ2h0ZmlsZTAnXSxcbiAgICAgICd3ZWlnaHRzJzogW1xuICAgICAgICB7J25hbWUnOiAnd2VpZ2h0MCcsICdkdHlwZSc6ICdmbG9hdDMyJywgJ3NoYXBlJzogWzJdfSxcbiAgICAgICAgeyduYW1lJzogJ3dlaWdodDEnLCAnZHR5cGUnOiAnZmxvYXQzMicsICdzaGFwZSc6IFszXX1cbiAgICAgIF1cbiAgICB9XTtcblxuICAgIC8vIExvYWQgdGhlIHNlY29uZCB3ZWlnaHQuXG4gICAgY29uc3Qgd2VpZ2h0cyA9IGF3YWl0IHRmLmlvLmxvYWRXZWlnaHRzKG1hbmlmZXN0LCAnLi8nLCBbJ3dlaWdodDEnXSk7XG4gICAgZXhwZWN0KCh0Zi5lbnYoKS5wbGF0Zm9ybS5mZXRjaCBhcyBqYXNtaW5lLlNweSkuY2FsbHMuY291bnQoKSkudG9CZSgxKTtcblxuICAgIGNvbnN0IHdlaWdodE5hbWVzID0gT2JqZWN0LmtleXMod2VpZ2h0cyk7XG4gICAgZXhwZWN0KHdlaWdodE5hbWVzLmxlbmd0aCkudG9FcXVhbCgxKTtcblxuICAgIGNvbnN0IHdlaWdodDEgPSB3ZWlnaHRzWyd3ZWlnaHQxJ107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgd2VpZ2h0MS5kYXRhKCksIFszLCA0LCA1XSk7XG4gICAgZXhwZWN0KHdlaWdodDEuc2hhcGUpLnRvRXF1YWwoWzNdKTtcbiAgICBleHBlY3Qod2VpZ2h0MS5kdHlwZSkudG9FcXVhbCgnZmxvYXQzMicpO1xuICB9KTtcblxuICBpdCgnMSBncm91cCwgMiB3ZWlnaHRzLCBmZXRjaCBhbGwgd2VpZ2h0cycsIGFzeW5jICgpID0+IHtcbiAgICBzZXR1cEZha2VXZWlnaHRGaWxlcyh7Jy4vd2VpZ2h0ZmlsZTAnOiBuZXcgRmxvYXQzMkFycmF5KFsxLCAyLCAzLCA0LCA1XSl9KTtcblxuICAgIGNvbnN0IG1hbmlmZXN0OiBXZWlnaHRzTWFuaWZlc3RDb25maWcgPSBbe1xuICAgICAgJ3BhdGhzJzogWyd3ZWlnaHRmaWxlMCddLFxuICAgICAgJ3dlaWdodHMnOiBbXG4gICAgICAgIHsnbmFtZSc6ICd3ZWlnaHQwJywgJ2R0eXBlJzogJ2Zsb2F0MzInLCAnc2hhcGUnOiBbMl19LFxuICAgICAgICB7J25hbWUnOiAnd2VpZ2h0MScsICdkdHlwZSc6ICdmbG9hdDMyJywgJ3NoYXBlJzogWzNdfVxuICAgICAgXVxuICAgIH1dO1xuXG4gICAgLy8gTG9hZCBhbGwgd2VpZ2h0cy5cbiAgICBjb25zdCB3ZWlnaHRzID1cbiAgICAgICAgYXdhaXQgdGYuaW8ubG9hZFdlaWdodHMobWFuaWZlc3QsICcuLycsIFsnd2VpZ2h0MCcsICd3ZWlnaHQxJ10pO1xuICAgIGV4cGVjdCgodGYuZW52KCkucGxhdGZvcm0uZmV0Y2ggYXMgamFzbWluZS5TcHkpLmNhbGxzLmNvdW50KCkpLnRvQmUoMSk7XG5cbiAgICBjb25zdCB3ZWlnaHROYW1lcyA9IE9iamVjdC5rZXlzKHdlaWdodHMpO1xuICAgIGV4cGVjdCh3ZWlnaHROYW1lcy5sZW5ndGgpLnRvRXF1YWwoMik7XG5cbiAgICBjb25zdCB3ZWlnaHQwID0gd2VpZ2h0c1snd2VpZ2h0MCddO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHdlaWdodDAuZGF0YSgpLCBbMSwgMl0pO1xuICAgIGV4cGVjdCh3ZWlnaHQwLnNoYXBlKS50b0VxdWFsKFsyXSk7XG4gICAgZXhwZWN0KHdlaWdodDAuZHR5cGUpLnRvRXF1YWwoJ2Zsb2F0MzInKTtcblxuICAgIGNvbnN0IHdlaWdodDEgPSB3ZWlnaHRzWyd3ZWlnaHQxJ107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgd2VpZ2h0MS5kYXRhKCksIFszLCA0LCA1XSk7XG4gICAgZXhwZWN0KHdlaWdodDEuc2hhcGUpLnRvRXF1YWwoWzNdKTtcbiAgICBleHBlY3Qod2VpZ2h0MS5kdHlwZSkudG9FcXVhbCgnZmxvYXQzMicpO1xuICB9KTtcblxuICBpdCgnMSBncm91cCwgbXVsdGlwbGUgd2VpZ2h0cywgZGlmZmVyZW50IGR0eXBlcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBidWZmZXIgPSBuZXcgQXJyYXlCdWZmZXIoNSAqIDQgKyAxKTtcbiAgICBjb25zdCB2aWV3ID0gbmV3IERhdGFWaWV3KGJ1ZmZlcik7XG4gICAgdmlldy5zZXRJbnQzMigwLCAxLCB0cnVlKTtcbiAgICB2aWV3LnNldEludDMyKDQsIDIsIHRydWUpO1xuICAgIHZpZXcuc2V0VWludDgoOCwgMSk7XG4gICAgdmlldy5zZXRGbG9hdDMyKDksIDMuLCB0cnVlKTtcbiAgICB2aWV3LnNldEZsb2F0MzIoMTMsIDQuLCB0cnVlKTtcbiAgICB2aWV3LnNldEZsb2F0MzIoMTcsIDUuLCB0cnVlKTtcbiAgICBzZXR1cEZha2VXZWlnaHRGaWxlcyh7Jy4vd2VpZ2h0ZmlsZTAnOiBidWZmZXJ9KTtcblxuICAgIGNvbnN0IG1hbmlmZXN0OiBXZWlnaHRzTWFuaWZlc3RDb25maWcgPSBbe1xuICAgICAgJ3BhdGhzJzogWyd3ZWlnaHRmaWxlMCddLFxuICAgICAgJ3dlaWdodHMnOiBbXG4gICAgICAgIHsnbmFtZSc6ICd3ZWlnaHQwJywgJ2R0eXBlJzogJ2ludDMyJywgJ3NoYXBlJzogWzJdfSxcbiAgICAgICAgeyduYW1lJzogJ3dlaWdodDEnLCAnZHR5cGUnOiAnYm9vbCcsICdzaGFwZSc6IFtdfSxcbiAgICAgICAgeyduYW1lJzogJ3dlaWdodDInLCAnZHR5cGUnOiAnZmxvYXQzMicsICdzaGFwZSc6IFszXX0sXG4gICAgICBdXG4gICAgfV07XG5cbiAgICAvLyBMb2FkIGFsbCB3ZWlnaHRzLlxuICAgIGNvbnN0IHdlaWdodHMgPSBhd2FpdCB0Zi5pby5sb2FkV2VpZ2h0cyhcbiAgICAgICAgbWFuaWZlc3QsICcuLycsIFsnd2VpZ2h0MCcsICd3ZWlnaHQxJywgJ3dlaWdodDInXSk7XG4gICAgZXhwZWN0KCh0Zi5lbnYoKS5wbGF0Zm9ybS5mZXRjaCBhcyBqYXNtaW5lLlNweSkuY2FsbHMuY291bnQoKSkudG9CZSgxKTtcblxuICAgIGNvbnN0IHdlaWdodE5hbWVzID0gT2JqZWN0LmtleXMod2VpZ2h0cyk7XG4gICAgZXhwZWN0KHdlaWdodE5hbWVzLmxlbmd0aCkudG9FcXVhbCgzKTtcblxuICAgIGNvbnN0IHdlaWdodDAgPSB3ZWlnaHRzWyd3ZWlnaHQwJ107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgd2VpZ2h0MC5kYXRhKCksIFsxLCAyXSk7XG4gICAgZXhwZWN0KHdlaWdodDAuc2hhcGUpLnRvRXF1YWwoWzJdKTtcbiAgICBleHBlY3Qod2VpZ2h0MC5kdHlwZSkudG9FcXVhbCgnaW50MzInKTtcblxuICAgIGNvbnN0IHdlaWdodDEgPSB3ZWlnaHRzWyd3ZWlnaHQxJ107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgd2VpZ2h0MS5kYXRhKCksIFsxXSk7XG4gICAgZXhwZWN0KHdlaWdodDEuc2hhcGUpLnRvRXF1YWwoW10pO1xuICAgIGV4cGVjdCh3ZWlnaHQxLmR0eXBlKS50b0VxdWFsKCdib29sJyk7XG5cbiAgICBjb25zdCB3ZWlnaHQyID0gd2VpZ2h0c1snd2VpZ2h0MiddO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHdlaWdodDIuZGF0YSgpLCBbMywgNCwgNV0pO1xuICAgIGV4cGVjdCh3ZWlnaHQyLnNoYXBlKS50b0VxdWFsKFszXSk7XG4gICAgZXhwZWN0KHdlaWdodDIuZHR5cGUpLnRvRXF1YWwoJ2Zsb2F0MzInKTtcbiAgfSk7XG5cbiAgaXQoJzEgZ3JvdXAsIHNoYXJkZWQgMSB3ZWlnaHQgYWNyb3NzIG11bHRpcGxlIGZpbGVzJywgYXN5bmMgKCkgPT4ge1xuICAgIGNvbnN0IHNoYXJkMCA9IG5ldyBGbG9hdDMyQXJyYXkoWzEsIDIsIDMsIDQsIDVdKTtcbiAgICBjb25zdCBzaGFyZDEgPSBuZXcgRmxvYXQzMkFycmF5KFsxLjEsIDIuMl0pO1xuICAgIGNvbnN0IHNoYXJkMiA9IG5ldyBGbG9hdDMyQXJyYXkoWzEwLCAyMCwgMzBdKTtcblxuICAgIHNldHVwRmFrZVdlaWdodEZpbGVzKHtcbiAgICAgICcuL3dlaWdodGZpbGUwJzogc2hhcmQwLFxuICAgICAgJy4vd2VpZ2h0c2ZpbGUxJzogc2hhcmQxLFxuICAgICAgJy4vd2VpZ2h0c2ZpbGUyJzogc2hhcmQyXG4gICAgfSk7XG5cbiAgICBjb25zdCBtYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgICdwYXRocyc6IFsnd2VpZ2h0ZmlsZTAnLCAnd2VpZ2h0c2ZpbGUxJywgJ3dlaWdodHNmaWxlMiddLFxuICAgICAgJ3dlaWdodHMnOiBbeyduYW1lJzogJ3dlaWdodDAnLCAnZHR5cGUnOiAnZmxvYXQzMicsICdzaGFwZSc6IFs1LCAyXX1dXG4gICAgfV07XG5cbiAgICBjb25zdCB3ZWlnaHRzID0gYXdhaXQgdGYuaW8ubG9hZFdlaWdodHMobWFuaWZlc3QsICcuLycsIFsnd2VpZ2h0MCddKTtcbiAgICBleHBlY3QoKHRmLmVudigpLnBsYXRmb3JtLmZldGNoIGFzIGphc21pbmUuU3B5KS5jYWxscy5jb3VudCgpKS50b0JlKDMpO1xuXG4gICAgY29uc3Qgd2VpZ2h0TmFtZXMgPSBPYmplY3Qua2V5cyh3ZWlnaHRzKTtcbiAgICBleHBlY3Qod2VpZ2h0TmFtZXMubGVuZ3RoKS50b0VxdWFsKDEpO1xuXG4gICAgY29uc3Qgd2VpZ2h0MCA9IHdlaWdodHNbJ3dlaWdodDAnXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShcbiAgICAgICAgYXdhaXQgd2VpZ2h0MC5kYXRhKCksIFsxLCAyLCAzLCA0LCA1LCAxLjEsIDIuMiwgMTAsIDIwLCAzMF0pO1xuICAgIGV4cGVjdCh3ZWlnaHQwLnNoYXBlKS50b0VxdWFsKFs1LCAyXSk7XG4gICAgZXhwZWN0KHdlaWdodDAuZHR5cGUpLnRvRXF1YWwoJ2Zsb2F0MzInKTtcbiAgfSk7XG5cbiAgaXQoJzEgZ3JvdXAsIHNoYXJkZWQgMiB3ZWlnaHRzIGFjcm9zcyBtdWx0aXBsZSBmaWxlcycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCBzaGFyZDAgPSBuZXcgSW50MzJBcnJheShbMSwgMiwgMywgNCwgNV0pO1xuXG4gICAgLy8gc2hhcmQxIGNvbnRhaW5zIHBhcnQgb2YgdGhlIGZpcnN0IHdlaWdodCBhbmQgcGFydCBvZiB0aGUgc2Vjb25kLlxuICAgIGNvbnN0IHNoYXJkMSA9IG5ldyBBcnJheUJ1ZmZlcig1ICogNCk7XG4gICAgY29uc3QgaW50QnVmZmVyID0gbmV3IEludDMyQXJyYXkoc2hhcmQxLCAwLCAyKTtcbiAgICBpbnRCdWZmZXIuc2V0KFsxMCwgMjBdKTtcbiAgICBjb25zdCBmbG9hdEJ1ZmZlciA9IG5ldyBGbG9hdDMyQXJyYXkoc2hhcmQxLCBpbnRCdWZmZXIuYnl0ZUxlbmd0aCwgMyk7XG4gICAgZmxvYXRCdWZmZXIuc2V0KFszLjAsIDQuMCwgNS4wXSk7XG5cbiAgICBjb25zdCBzaGFyZDIgPSBuZXcgRmxvYXQzMkFycmF5KFsxMCwgMjAsIDMwXSk7XG5cbiAgICBzZXR1cEZha2VXZWlnaHRGaWxlcyh7XG4gICAgICAnLi93ZWlnaHRmaWxlMCc6IHNoYXJkMCxcbiAgICAgICcuL3dlaWdodHNmaWxlMSc6IHNoYXJkMSxcbiAgICAgICcuL3dlaWdodHNmaWxlMic6IHNoYXJkMlxuICAgIH0pO1xuXG4gICAgY29uc3QgbWFuaWZlc3Q6IFdlaWdodHNNYW5pZmVzdENvbmZpZyA9IFt7XG4gICAgICAncGF0aHMnOiBbJ3dlaWdodGZpbGUwJywgJ3dlaWdodHNmaWxlMScsICd3ZWlnaHRzZmlsZTInXSxcbiAgICAgICd3ZWlnaHRzJzogW1xuICAgICAgICB7J25hbWUnOiAnd2VpZ2h0MCcsICdkdHlwZSc6ICdpbnQzMicsICdzaGFwZSc6IFs3LCAxXX0sXG4gICAgICAgIHsnbmFtZSc6ICd3ZWlnaHQxJywgJ2R0eXBlJzogJ2Zsb2F0MzInLCAnc2hhcGUnOiBbMywgMl19XG4gICAgICBdXG4gICAgfV07XG5cbiAgICBjb25zdCB3ZWlnaHRzID1cbiAgICAgICAgYXdhaXQgdGYuaW8ubG9hZFdlaWdodHMobWFuaWZlc3QsICcuLycsIFsnd2VpZ2h0MCcsICd3ZWlnaHQxJ10pO1xuICAgIGV4cGVjdCgodGYuZW52KCkucGxhdGZvcm0uZmV0Y2ggYXMgamFzbWluZS5TcHkpLmNhbGxzLmNvdW50KCkpLnRvQmUoMyk7XG5cbiAgICBjb25zdCB3ZWlnaHROYW1lcyA9IE9iamVjdC5rZXlzKHdlaWdodHMpO1xuICAgIGV4cGVjdCh3ZWlnaHROYW1lcy5sZW5ndGgpLnRvRXF1YWwoMik7XG5cbiAgICBjb25zdCB3ZWlnaHQwID0gd2VpZ2h0c1snd2VpZ2h0MCddO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHdlaWdodDAuZGF0YSgpLCBbMSwgMiwgMywgNCwgNSwgMTAsIDIwXSk7XG4gICAgZXhwZWN0KHdlaWdodDAuc2hhcGUpLnRvRXF1YWwoWzcsIDFdKTtcbiAgICBleHBlY3Qod2VpZ2h0MC5kdHlwZSkudG9FcXVhbCgnaW50MzInKTtcblxuICAgIGNvbnN0IHdlaWdodDEgPSB3ZWlnaHRzWyd3ZWlnaHQxJ107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgd2VpZ2h0MS5kYXRhKCksIFszLjAsIDQuMCwgNS4wLCAxMCwgMjAsIDMwXSk7XG4gICAgZXhwZWN0KHdlaWdodDEuc2hhcGUpLnRvRXF1YWwoWzMsIDJdKTtcbiAgICBleHBlY3Qod2VpZ2h0MS5kdHlwZSkudG9FcXVhbCgnZmxvYXQzMicpO1xuICB9KTtcblxuICBpdCgnMiBncm91cCwgNCB3ZWlnaHRzLCBmZXRjaGVzIG9uZSBncm91cCcsIGFzeW5jICgpID0+IHtcbiAgICBzZXR1cEZha2VXZWlnaHRGaWxlcyh7XG4gICAgICAnLi93ZWlnaHRmaWxlMCc6IG5ldyBGbG9hdDMyQXJyYXkoWzEsIDIsIDMsIDQsIDVdKSxcbiAgICAgICcuL3dlaWdodGZpbGUxJzogbmV3IEZsb2F0MzJBcnJheShbNiwgNywgOCwgOV0pXG4gICAgfSk7XG5cbiAgICBjb25zdCBtYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW1xuICAgICAge1xuICAgICAgICAncGF0aHMnOiBbJ3dlaWdodGZpbGUwJ10sXG4gICAgICAgICd3ZWlnaHRzJzogW1xuICAgICAgICAgIHsnbmFtZSc6ICd3ZWlnaHQwJywgJ2R0eXBlJzogJ2Zsb2F0MzInLCAnc2hhcGUnOiBbMl19LFxuICAgICAgICAgIHsnbmFtZSc6ICd3ZWlnaHQxJywgJ2R0eXBlJzogJ2Zsb2F0MzInLCAnc2hhcGUnOiBbM119XG4gICAgICAgIF1cbiAgICAgIH0sXG4gICAgICB7XG4gICAgICAgICdwYXRocyc6IFsnd2VpZ2h0ZmlsZTEnXSxcbiAgICAgICAgJ3dlaWdodHMnOiBbXG4gICAgICAgICAgeyduYW1lJzogJ3dlaWdodDInLCAnZHR5cGUnOiAnZmxvYXQzMicsICdzaGFwZSc6IFszLCAxXX0sXG4gICAgICAgICAgeyduYW1lJzogJ3dlaWdodDMnLCAnZHR5cGUnOiAnZmxvYXQzMicsICdzaGFwZSc6IFtdfVxuICAgICAgICBdXG4gICAgICB9XG4gICAgXTtcblxuICAgIGNvbnN0IHdlaWdodHMgPVxuICAgICAgICBhd2FpdCB0Zi5pby5sb2FkV2VpZ2h0cyhtYW5pZmVzdCwgJy4vJywgWyd3ZWlnaHQwJywgJ3dlaWdodDEnXSk7XG4gICAgLy8gT25seSB0aGUgZmlyc3QgZ3JvdXAgc2hvdWxkIGJlIGZldGNoZWQuXG4gICAgZXhwZWN0KCh0Zi5lbnYoKS5wbGF0Zm9ybS5mZXRjaCBhcyBqYXNtaW5lLlNweSkuY2FsbHMuY291bnQoKSkudG9CZSgxKTtcblxuICAgIGNvbnN0IHdlaWdodE5hbWVzID0gT2JqZWN0LmtleXMod2VpZ2h0cyk7XG4gICAgZXhwZWN0KHdlaWdodE5hbWVzLmxlbmd0aCkudG9FcXVhbCgyKTtcblxuICAgIGNvbnN0IHdlaWdodDAgPSB3ZWlnaHRzWyd3ZWlnaHQwJ107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgd2VpZ2h0MC5kYXRhKCksIFsxLCAyXSk7XG4gICAgZXhwZWN0KHdlaWdodDAuc2hhcGUpLnRvRXF1YWwoWzJdKTtcbiAgICBleHBlY3Qod2VpZ2h0MC5kdHlwZSkudG9FcXVhbCgnZmxvYXQzMicpO1xuXG4gICAgY29uc3Qgd2VpZ2h0MSA9IHdlaWdodHNbJ3dlaWdodDEnXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB3ZWlnaHQxLmRhdGEoKSwgWzMsIDQsIDVdKTtcbiAgICBleHBlY3Qod2VpZ2h0MS5zaGFwZSkudG9FcXVhbChbM10pO1xuICAgIGV4cGVjdCh3ZWlnaHQxLmR0eXBlKS50b0VxdWFsKCdmbG9hdDMyJyk7XG4gIH0pO1xuXG4gIGl0KCcyIGdyb3VwLCA0IHdlaWdodHMsIG9uZSB3ZWlnaHQgZnJvbSBlYWNoIGdyb3VwJywgYXN5bmMgKCkgPT4ge1xuICAgIHNldHVwRmFrZVdlaWdodEZpbGVzKHtcbiAgICAgICcuL3dlaWdodGZpbGUwJzogbmV3IEZsb2F0MzJBcnJheShbMSwgMiwgMywgNCwgNV0pLFxuICAgICAgJy4vd2VpZ2h0ZmlsZTEnOiBuZXcgRmxvYXQzMkFycmF5KFs2LCA3LCA4LCA5XSlcbiAgICB9KTtcblxuICAgIGNvbnN0IG1hbmlmZXN0OiBXZWlnaHRzTWFuaWZlc3RDb25maWcgPSBbXG4gICAgICB7XG4gICAgICAgICdwYXRocyc6IFsnd2VpZ2h0ZmlsZTAnXSxcbiAgICAgICAgJ3dlaWdodHMnOiBbXG4gICAgICAgICAgeyduYW1lJzogJ3dlaWdodDAnLCAnZHR5cGUnOiAnZmxvYXQzMicsICdzaGFwZSc6IFsyXX0sXG4gICAgICAgICAgeyduYW1lJzogJ3dlaWdodDEnLCAnZHR5cGUnOiAnZmxvYXQzMicsICdzaGFwZSc6IFszXX1cbiAgICAgICAgXVxuICAgICAgfSxcbiAgICAgIHtcbiAgICAgICAgJ3BhdGhzJzogWyd3ZWlnaHRmaWxlMSddLFxuICAgICAgICAnd2VpZ2h0cyc6IFtcbiAgICAgICAgICB7J25hbWUnOiAnd2VpZ2h0MicsICdkdHlwZSc6ICdmbG9hdDMyJywgJ3NoYXBlJzogWzMsIDFdfSxcbiAgICAgICAgICB7J25hbWUnOiAnd2VpZ2h0MycsICdkdHlwZSc6ICdmbG9hdDMyJywgJ3NoYXBlJzogW119XG4gICAgICAgIF1cbiAgICAgIH1cbiAgICBdO1xuXG4gICAgY29uc3Qgd2VpZ2h0cyA9XG4gICAgICAgIGF3YWl0IHRmLmlvLmxvYWRXZWlnaHRzKG1hbmlmZXN0LCAnLi8nLCBbJ3dlaWdodDAnLCAnd2VpZ2h0MiddKTtcbiAgICAvLyBCb3RoIGdyb3VwcyBuZWVkIHRvIGJlIGZldGNoZWQuXG4gICAgZXhwZWN0KCh0Zi5lbnYoKS5wbGF0Zm9ybS5mZXRjaCBhcyBqYXNtaW5lLlNweSkuY2FsbHMuY291bnQoKSkudG9CZSgyKTtcblxuICAgIGNvbnN0IHdlaWdodE5hbWVzID0gT2JqZWN0LmtleXMod2VpZ2h0cyk7XG4gICAgZXhwZWN0KHdlaWdodE5hbWVzLmxlbmd0aCkudG9FcXVhbCgyKTtcblxuICAgIGNvbnN0IHdlaWdodDAgPSB3ZWlnaHRzWyd3ZWlnaHQwJ107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgd2VpZ2h0MC5kYXRhKCksIFsxLCAyXSk7XG4gICAgZXhwZWN0KHdlaWdodDAuc2hhcGUpLnRvRXF1YWwoWzJdKTtcbiAgICBleHBlY3Qod2VpZ2h0MC5kdHlwZSkudG9FcXVhbCgnZmxvYXQzMicpO1xuXG4gICAgY29uc3Qgd2VpZ2h0MiA9IHdlaWdodHNbJ3dlaWdodDInXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB3ZWlnaHQyLmRhdGEoKSwgWzYsIDcsIDhdKTtcbiAgICBleHBlY3Qod2VpZ2h0Mi5zaGFwZSkudG9FcXVhbChbMywgMV0pO1xuICAgIGV4cGVjdCh3ZWlnaHQyLmR0eXBlKS50b0VxdWFsKCdmbG9hdDMyJyk7XG4gIH0pO1xuXG4gIGl0KCcyIGdyb3VwLCA0IHdlaWdodHMsIGRvbnQgc3BlY2lmeSB3ZWlnaHRzIGZldGNocyBhbGwnLCBhc3luYyAoKSA9PiB7XG4gICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoe1xuICAgICAgJy4vd2VpZ2h0ZmlsZTAnOiBuZXcgRmxvYXQzMkFycmF5KFsxLCAyLCAzLCA0LCA1XSksXG4gICAgICAnLi93ZWlnaHRmaWxlMSc6IG5ldyBGbG9hdDMyQXJyYXkoWzYsIDcsIDgsIDldKVxuICAgIH0pO1xuXG4gICAgY29uc3QgbWFuaWZlc3Q6IFdlaWdodHNNYW5pZmVzdENvbmZpZyA9IFtcbiAgICAgIHtcbiAgICAgICAgJ3BhdGhzJzogWyd3ZWlnaHRmaWxlMCddLFxuICAgICAgICAnd2VpZ2h0cyc6IFtcbiAgICAgICAgICB7J25hbWUnOiAnd2VpZ2h0MCcsICdkdHlwZSc6ICdmbG9hdDMyJywgJ3NoYXBlJzogWzJdfSxcbiAgICAgICAgICB7J25hbWUnOiAnd2VpZ2h0MScsICdkdHlwZSc6ICdmbG9hdDMyJywgJ3NoYXBlJzogWzNdfVxuICAgICAgICBdXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAncGF0aHMnOiBbJ3dlaWdodGZpbGUxJ10sXG4gICAgICAgICd3ZWlnaHRzJzogW1xuICAgICAgICAgIHsnbmFtZSc6ICd3ZWlnaHQyJywgJ2R0eXBlJzogJ2Zsb2F0MzInLCAnc2hhcGUnOiBbMywgMV19LFxuICAgICAgICAgIHsnbmFtZSc6ICd3ZWlnaHQzJywgJ2R0eXBlJzogJ2Zsb2F0MzInLCAnc2hhcGUnOiBbXX1cbiAgICAgICAgXVxuICAgICAgfVxuICAgIF07XG5cbiAgICAvLyBEb24ndCBwYXNzIGEgdGhpcmQgYXJndW1lbnQgdG8gbG9hZFdlaWdodHMgdG8gbG9hZCBhbGwgd2VpZ2h0cy5cbiAgICBjb25zdCB3ZWlnaHRzID0gYXdhaXQgdGYuaW8ubG9hZFdlaWdodHMobWFuaWZlc3QsICcuLycpO1xuICAgIC8vIEJvdGggZ3JvdXBzIG5lZWQgdG8gYmUgZmV0Y2hlZC5cbiAgICBleHBlY3QoKHRmLmVudigpLnBsYXRmb3JtLmZldGNoIGFzIGphc21pbmUuU3B5KS5jYWxscy5jb3VudCgpKS50b0JlKDIpO1xuXG4gICAgY29uc3Qgd2VpZ2h0TmFtZXMgPSBPYmplY3Qua2V5cyh3ZWlnaHRzKTtcbiAgICBleHBlY3Qod2VpZ2h0TmFtZXMubGVuZ3RoKS50b0VxdWFsKDQpO1xuXG4gICAgY29uc3Qgd2VpZ2h0MCA9IHdlaWdodHNbJ3dlaWdodDAnXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB3ZWlnaHQwLmRhdGEoKSwgWzEsIDJdKTtcbiAgICBleHBlY3Qod2VpZ2h0MC5zaGFwZSkudG9FcXVhbChbMl0pO1xuICAgIGV4cGVjdCh3ZWlnaHQwLmR0eXBlKS50b0VxdWFsKCdmbG9hdDMyJyk7XG5cbiAgICBjb25zdCB3ZWlnaHQxID0gd2VpZ2h0c1snd2VpZ2h0MSddO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHdlaWdodDEuZGF0YSgpLCBbMywgNCwgNV0pO1xuICAgIGV4cGVjdCh3ZWlnaHQxLnNoYXBlKS50b0VxdWFsKFszXSk7XG4gICAgZXhwZWN0KHdlaWdodDEuZHR5cGUpLnRvRXF1YWwoJ2Zsb2F0MzInKTtcblxuICAgIGNvbnN0IHdlaWdodDIgPSB3ZWlnaHRzWyd3ZWlnaHQyJ107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgd2VpZ2h0Mi5kYXRhKCksIFs2LCA3LCA4XSk7XG4gICAgZXhwZWN0KHdlaWdodDIuc2hhcGUpLnRvRXF1YWwoWzMsIDFdKTtcbiAgICBleHBlY3Qod2VpZ2h0Mi5kdHlwZSkudG9FcXVhbCgnZmxvYXQzMicpO1xuXG4gICAgY29uc3Qgd2VpZ2h0MyA9IHdlaWdodHNbJ3dlaWdodDMnXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB3ZWlnaHQzLmRhdGEoKSwgWzldKTtcbiAgICBleHBlY3Qod2VpZ2h0My5zaGFwZSkudG9FcXVhbChbXSk7XG4gICAgZXhwZWN0KHdlaWdodDMuZHR5cGUpLnRvRXF1YWwoJ2Zsb2F0MzInKTtcbiAgfSk7XG5cbiAgaXQoJ3Rocm93cyBpZiByZXF1ZXN0ZWQgd2VpZ2h0IG5vdCBmb3VuZCcsIGFzeW5jICgpID0+IHtcbiAgICBzZXR1cEZha2VXZWlnaHRGaWxlcyh7Jy4vd2VpZ2h0ZmlsZTAnOiBuZXcgRmxvYXQzMkFycmF5KFsxLCAyLCAzXSl9KTtcblxuICAgIGNvbnN0IG1hbmlmZXN0OiBXZWlnaHRzTWFuaWZlc3RDb25maWcgPSBbe1xuICAgICAgJ3BhdGhzJzogWyd3ZWlnaHRmaWxlMCddLFxuICAgICAgJ3dlaWdodHMnOiBbeyduYW1lJzogJ3dlaWdodDAnLCAnZHR5cGUnOiAnZmxvYXQzMicsICdzaGFwZSc6IFszXX1dXG4gICAgfV07XG5cbiAgICBjb25zdCB3ZWlnaHRzTmFtZXNUb0ZldGNoID0gWydkb2VzbnRleGlzdCddO1xuICAgIHRyeSB7XG4gICAgICBhd2FpdCB0Zi5pby5sb2FkV2VpZ2h0cyhtYW5pZmVzdCwgJy4vJywgd2VpZ2h0c05hbWVzVG9GZXRjaCk7XG4gICAgICBmYWlsKCk7XG4gICAgfSBjYXRjaCAoZSkge1xuICAgICAgZXhwZWN0KGUubWVzc2FnZSkudG9Db250YWluKCdDb3VsZCBub3QgZmluZCB3ZWlnaHRzJyk7XG4gICAgfVxuICB9KTtcblxuICBpdCgndGhyb3dzIGlmIHJlcXVlc3RlZCB3ZWlnaHQgaGFzIHVua25vd24gZHR5cGUnLCBhc3luYyAoKSA9PiB7XG4gICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoeycuL3dlaWdodGZpbGUwJzogbmV3IEZsb2F0MzJBcnJheShbMSwgMiwgM10pfSk7XG5cbiAgICBjb25zdCBtYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgICdwYXRocyc6IFsnd2VpZ2h0ZmlsZTAnXSxcbiAgICAgICd3ZWlnaHRzJzogW3tcbiAgICAgICAgJ25hbWUnOiAnd2VpZ2h0MCcsXG4gICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgJ2R0eXBlJzogJ251bGwnIGFzIGFueSxcbiAgICAgICAgJ3NoYXBlJzogWzNdXG4gICAgICB9XVxuICAgIH1dO1xuXG4gICAgY29uc3Qgd2VpZ2h0c05hbWVzVG9GZXRjaCA9IFsnd2VpZ2h0MCddO1xuICAgIHRyeSB7XG4gICAgICBhd2FpdCB0Zi5pby5sb2FkV2VpZ2h0cyhtYW5pZmVzdCwgJy4vJywgd2VpZ2h0c05hbWVzVG9GZXRjaCk7XG4gICAgICBmYWlsKCk7XG4gICAgfSBjYXRjaCAoZSkge1xuICAgICAgZXhwZWN0KGUubWVzc2FnZSkudG9Db250YWluKCdVbnN1cHBvcnRlZCBkdHlwZScpO1xuICAgIH1cbiAgfSk7XG5cbiAgaXQoJ3Nob3VsZCB1c2UgcmVxdWVzdCBvcHRpb24nLCBhc3luYyAoKSA9PiB7XG4gICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoeycuL3dlaWdodGZpbGUwJzogbmV3IEZsb2F0MzJBcnJheShbMSwgMiwgM10pfSk7XG5cbiAgICBjb25zdCBtYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgICdwYXRocyc6IFsnd2VpZ2h0ZmlsZTAnXSxcbiAgICAgICd3ZWlnaHRzJzogW3snbmFtZSc6ICd3ZWlnaHQwJywgJ2R0eXBlJzogJ2Zsb2F0MzInLCAnc2hhcGUnOiBbM119XVxuICAgIH1dO1xuXG4gICAgY29uc3Qgd2VpZ2h0c05hbWVzVG9GZXRjaCA9IFsnd2VpZ2h0MCddO1xuICAgIGF3YWl0IHRmLmlvLmxvYWRXZWlnaHRzKFxuICAgICAgICBtYW5pZmVzdCwgJy4vJywgd2VpZ2h0c05hbWVzVG9GZXRjaCwge2NyZWRlbnRpYWxzOiAnaW5jbHVkZSd9KTtcbiAgICBleHBlY3QoKHRmLmVudigpLnBsYXRmb3JtLmZldGNoIGFzIGphc21pbmUuU3B5KS5jYWxscy5jb3VudCgpKS50b0JlKDEpO1xuICAgIGV4cGVjdCh0Zi5lbnYoKS5wbGF0Zm9ybS5mZXRjaClcbiAgICAgICAgLnRvSGF2ZUJlZW5DYWxsZWRXaXRoKFxuICAgICAgICAgICAgJy4vd2VpZ2h0ZmlsZTAnLCB7Y3JlZGVudGlhbHM6ICdpbmNsdWRlJ30sIHtpc0JpbmFyeTogdHJ1ZX0pO1xuICB9KTtcblxuICBjb25zdCBxdWFudGl6YXRpb25UZXN0ID0gYXN5bmMgKHF1YW50aXphdGlvbkR0eXBlOiAndWludDgnfCd1aW50MTYnKSA9PiB7XG4gICAgY29uc3QgYXJyYXlUeXBlID0gcXVhbnRpemF0aW9uRHR5cGUgPT09ICd1aW50OCcgPyBVaW50OEFycmF5IDogVWludDE2QXJyYXk7XG4gICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoXG4gICAgICAgIHsnLi93ZWlnaHRmaWxlMCc6IG5ldyBhcnJheVR5cGUoWzAsIDQ4LCAyNTUsIDAsIDQ4LCAyNTVdKX0pO1xuXG4gICAgY29uc3QgbWFuaWZlc3Q6IFdlaWdodHNNYW5pZmVzdENvbmZpZyA9IFt7XG4gICAgICAncGF0aHMnOiBbJ3dlaWdodGZpbGUwJ10sXG4gICAgICAnd2VpZ2h0cyc6IFtcbiAgICAgICAge1xuICAgICAgICAgICduYW1lJzogJ3dlaWdodDAnLFxuICAgICAgICAgICdkdHlwZSc6ICdmbG9hdDMyJyxcbiAgICAgICAgICAnc2hhcGUnOiBbM10sXG4gICAgICAgICAgJ3F1YW50aXphdGlvbic6IHsnbWluJzogLTEsICdzY2FsZSc6IDAuMSwgJ2R0eXBlJzogcXVhbnRpemF0aW9uRHR5cGV9XG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICAnbmFtZSc6ICd3ZWlnaHQxJyxcbiAgICAgICAgICAnZHR5cGUnOiAnaW50MzInLFxuICAgICAgICAgICdzaGFwZSc6IFszXSxcbiAgICAgICAgICAncXVhbnRpemF0aW9uJzogeydtaW4nOiAtMSwgJ3NjYWxlJzogMC4xLCAnZHR5cGUnOiBxdWFudGl6YXRpb25EdHlwZX1cbiAgICAgICAgfVxuICAgICAgXVxuICAgIH1dO1xuXG4gICAgY29uc3Qgd2VpZ2h0c05hbWVzVG9GZXRjaCA9IFsnd2VpZ2h0MCcsICd3ZWlnaHQxJ107XG4gICAgY29uc3Qgd2VpZ2h0cyA9XG4gICAgICAgIGF3YWl0IHRmLmlvLmxvYWRXZWlnaHRzKG1hbmlmZXN0LCAnLi8nLCB3ZWlnaHRzTmFtZXNUb0ZldGNoKTtcbiAgICBleHBlY3QoKHRmLmVudigpLnBsYXRmb3JtLmZldGNoIGFzIGphc21pbmUuU3B5KS5jYWxscy5jb3VudCgpKS50b0JlKDEpO1xuXG4gICAgY29uc3Qgd2VpZ2h0TmFtZXMgPSBPYmplY3Qua2V5cyh3ZWlnaHRzKTtcbiAgICBleHBlY3Qod2VpZ2h0TmFtZXMubGVuZ3RoKS50b0VxdWFsKHdlaWdodHNOYW1lc1RvRmV0Y2gubGVuZ3RoKTtcblxuICAgIGNvbnN0IHdlaWdodDAgPSB3ZWlnaHRzWyd3ZWlnaHQwJ107XG4gICAgZXhwZWN0QXJyYXlzQ2xvc2UoYXdhaXQgd2VpZ2h0MC5kYXRhKCksIFstMSwgMy44LCAyNC41XSk7XG4gICAgZXhwZWN0KHdlaWdodDAuc2hhcGUpLnRvRXF1YWwoWzNdKTtcbiAgICBleHBlY3Qod2VpZ2h0MC5kdHlwZSkudG9FcXVhbCgnZmxvYXQzMicpO1xuXG4gICAgY29uc3Qgd2VpZ2h0MSA9IHdlaWdodHNbJ3dlaWdodDEnXTtcbiAgICBleHBlY3RBcnJheXNFcXVhbChhd2FpdCB3ZWlnaHQxLmRhdGEoKSwgWy0xLCA0LCAyNV0pO1xuICAgIGV4cGVjdCh3ZWlnaHQxLnNoYXBlKS50b0VxdWFsKFszXSk7XG4gICAgZXhwZWN0KHdlaWdodDEuZHR5cGUpLnRvRXF1YWwoJ2ludDMyJyk7XG4gIH07XG5cbiAgaXQoJ3F1YW50aXplZCB3ZWlnaHRzICh1aW50OCknLCBhc3luYyAoKSA9PiB7XG4gICAgYXdhaXQgcXVhbnRpemF0aW9uVGVzdCgndWludDgnKTtcbiAgfSk7XG5cbiAgaXQoJ3F1YW50aXplZCB3ZWlnaHRzICh1aW50MTYpJywgYXN5bmMgKCkgPT4ge1xuICAgIGF3YWl0IHF1YW50aXphdGlvblRlc3QoJ3VpbnQxNicpO1xuICB9KTtcblxuICBpdCgnMiBncm91cHMsIDEgcXVhbnRpemVkLCAxIHVucXVhbnRpemVkJywgYXN5bmMgKCkgPT4ge1xuICAgIHNldHVwRmFrZVdlaWdodEZpbGVzKHtcbiAgICAgICcuL3dlaWdodGZpbGUwJzogbmV3IFVpbnQ4QXJyYXkoWzAsIDQ4LCAyNTUsIDAsIDQ4LCAyNTVdKSxcbiAgICAgICcuL3dlaWdodGZpbGUxJzogbmV3IEZsb2F0MzJBcnJheShbNiwgNywgOCwgOV0pXG4gICAgfSk7XG5cbiAgICBjb25zdCBtYW5pZmVzdDogV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW1xuICAgICAge1xuICAgICAgICAncGF0aHMnOiBbJ3dlaWdodGZpbGUwJ10sXG4gICAgICAgICd3ZWlnaHRzJzogW1xuICAgICAgICAgIHtcbiAgICAgICAgICAgICduYW1lJzogJ3dlaWdodDAnLFxuICAgICAgICAgICAgJ2R0eXBlJzogJ2Zsb2F0MzInLFxuICAgICAgICAgICAgJ3NoYXBlJzogWzNdLFxuICAgICAgICAgICAgJ3F1YW50aXphdGlvbic6IHsnbWluJzogLTEsICdzY2FsZSc6IDAuMSwgJ2R0eXBlJzogJ3VpbnQ4J31cbiAgICAgICAgICB9LFxuICAgICAgICAgIHtcbiAgICAgICAgICAgICduYW1lJzogJ3dlaWdodDEnLFxuICAgICAgICAgICAgJ2R0eXBlJzogJ2ludDMyJyxcbiAgICAgICAgICAgICdzaGFwZSc6IFszXSxcbiAgICAgICAgICAgICdxdWFudGl6YXRpb24nOiB7J21pbic6IC0xLCAnc2NhbGUnOiAwLjEsICdkdHlwZSc6ICd1aW50OCd9XG4gICAgICAgICAgfVxuICAgICAgICBdXG4gICAgICB9LFxuICAgICAge1xuICAgICAgICAncGF0aHMnOiBbJ3dlaWdodGZpbGUxJ10sXG4gICAgICAgICd3ZWlnaHRzJzogW1xuICAgICAgICAgIHsnbmFtZSc6ICd3ZWlnaHQyJywgJ2R0eXBlJzogJ2Zsb2F0MzInLCAnc2hhcGUnOiBbMywgMV19LFxuICAgICAgICAgIHsnbmFtZSc6ICd3ZWlnaHQzJywgJ2R0eXBlJzogJ2Zsb2F0MzInLCAnc2hhcGUnOiBbXX1cbiAgICAgICAgXVxuICAgICAgfVxuICAgIF07XG5cbiAgICBjb25zdCB3ZWlnaHRzID1cbiAgICAgICAgYXdhaXQgdGYuaW8ubG9hZFdlaWdodHMobWFuaWZlc3QsICcuLycsIFsnd2VpZ2h0MCcsICd3ZWlnaHQyJ10pO1xuICAgIC8vIEJvdGggZ3JvdXBzIG5lZWQgdG8gYmUgZmV0Y2hlZC5cbiAgICBleHBlY3QoKHRmLmVudigpLnBsYXRmb3JtLmZldGNoIGFzIGphc21pbmUuU3B5KS5jYWxscy5jb3VudCgpKS50b0JlKDIpO1xuXG4gICAgY29uc3Qgd2VpZ2h0TmFtZXMgPSBPYmplY3Qua2V5cyh3ZWlnaHRzKTtcbiAgICBleHBlY3Qod2VpZ2h0TmFtZXMubGVuZ3RoKS50b0VxdWFsKDIpO1xuXG4gICAgY29uc3Qgd2VpZ2h0MCA9IHdlaWdodHNbJ3dlaWdodDAnXTtcbiAgICBleHBlY3RBcnJheXNDbG9zZShhd2FpdCB3ZWlnaHQwLmRhdGEoKSwgWy0xLCAzLjgsIDI0LjVdKTtcbiAgICBleHBlY3Qod2VpZ2h0MC5zaGFwZSkudG9FcXVhbChbM10pO1xuICAgIGV4cGVjdCh3ZWlnaHQwLmR0eXBlKS50b0VxdWFsKCdmbG9hdDMyJyk7XG5cbiAgICBjb25zdCB3ZWlnaHQyID0gd2VpZ2h0c1snd2VpZ2h0MiddO1xuICAgIGV4cGVjdEFycmF5c0Nsb3NlKGF3YWl0IHdlaWdodDIuZGF0YSgpLCBbNiwgNywgOF0pO1xuICAgIGV4cGVjdCh3ZWlnaHQyLnNoYXBlKS50b0VxdWFsKFszLCAxXSk7XG4gICAgZXhwZWN0KHdlaWdodDIuZHR5cGUpLnRvRXF1YWwoJ2Zsb2F0MzInKTtcbiAgfSk7XG59KTtcbiJdfQ==