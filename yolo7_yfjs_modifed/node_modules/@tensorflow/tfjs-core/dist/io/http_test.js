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
import { BROWSER_ENVS, CHROME_ENVS, describeWithFlags, NODE_ENVS } from '../jasmine_util';
import { HTTPRequest, httpRouter, parseUrl } from './http';
// Test data.
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
const trainingConfig1 = {
    loss: 'categorical_crossentropy',
    metrics: ['accuracy'],
    optimizer_config: { class_name: 'SGD', config: { learningRate: 0.1 } }
};
let fetchSpy;
const fakeResponse = (body, contentType, path) => ({
    ok: true,
    json() {
        return Promise.resolve(JSON.parse(body));
    },
    arrayBuffer() {
        const buf = body.buffer ?
            body.buffer :
            body;
        return Promise.resolve(buf);
    },
    headers: { get: (key) => contentType },
    url: path
});
const setupFakeWeightFiles = (fileBufferMap, requestInits) => {
    fetchSpy = spyOn(tf.env().platform, 'fetch')
        .and.callFake((path, init) => {
        if (fileBufferMap[path]) {
            requestInits[path] = init;
            return Promise.resolve(fakeResponse(fileBufferMap[path].data, fileBufferMap[path].contentType, path));
        }
        else {
            return Promise.reject('path not found');
        }
    });
};
describeWithFlags('http-load fetch', NODE_ENVS, () => {
    let requestInits;
    // tslint:disable-next-line:no-any
    let originalFetch;
    // simulate a fetch polyfill, this needs to be non-null for spyOn to work
    beforeEach(() => {
        // tslint:disable-next-line:no-any
        originalFetch = global.fetch;
        // tslint:disable-next-line:no-any
        global.fetch = () => { };
        requestInits = {};
    });
    afterAll(() => {
        // tslint:disable-next-line:no-any
        global.fetch = originalFetch;
    });
    it('1 group, 2 weights, 1 path', async () => {
        const weightManifest1 = [{
                paths: ['weightfile0'],
                weights: [
                    {
                        name: 'dense/kernel',
                        shape: [3, 1],
                        dtype: 'float32',
                    },
                    {
                        name: 'dense/bias',
                        shape: [2],
                        dtype: 'float32',
                    }
                ]
            }];
        const floatData = new Float32Array([1, 3, 3, 7, 4]);
        setupFakeWeightFiles({
            './model.json': {
                data: JSON.stringify({
                    modelTopology: modelTopology1,
                    weightsManifest: weightManifest1,
                    format: 'tfjs-layers',
                    generatedBy: '1.15',
                    convertedBy: '1.3.1',
                    signature: null,
                    userDefinedMetadata: {}
                }),
                contentType: 'application/json'
            },
            './weightfile0': { data: floatData, contentType: 'application/octet-stream' },
        }, requestInits);
        const handler = tf.io.http('./model.json');
        const modelArtifacts = await handler.load();
        expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
        expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
        expect(modelArtifacts.format).toEqual('tfjs-layers');
        expect(modelArtifacts.generatedBy).toEqual('1.15');
        expect(modelArtifacts.convertedBy).toEqual('1.3.1');
        expect(modelArtifacts.userDefinedMetadata).toEqual({});
        expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);
    });
    it('throw exception if no fetch polyfill', () => {
        // tslint:disable-next-line:no-any
        delete global.fetch;
        try {
            tf.io.http('./model.json');
        }
        catch (err) {
            expect(err.message).toMatch(/Unable to find fetch polyfill./);
        }
    });
});
// Turned off for other browsers due to:
// https://github.com/tensorflow/tfjs/issues/426
describeWithFlags('http-save', CHROME_ENVS, () => {
    // Test data.
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
        format: 'layers-model',
        generatedBy: 'TensorFlow.js v0.0.0',
        convertedBy: null,
        signature: null,
        userDefinedMetadata: {},
        modelInitializer: {},
        trainingConfig: trainingConfig1
    };
    let requestInits = [];
    beforeEach(() => {
        requestInits = [];
        spyOn(tf.env().platform, 'fetch')
            .and.callFake((path, init) => {
            if (path === 'model-upload-test' ||
                path === 'http://model-upload-test') {
                requestInits.push(init);
                return Promise.resolve(new Response(null, { status: 200 }));
            }
            else {
                return Promise.reject(new Response(null, { status: 404 }));
            }
        });
    });
    it('Save topology and weights, default POST method', (done) => {
        const testStartDate = new Date();
        const handler = tf.io.getSaveHandlers('http://model-upload-test')[0];
        handler.save(artifacts1)
            .then(saveResult => {
            expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
                .toBeGreaterThanOrEqual(testStartDate.getTime());
            // Note: The following two assertions work only because there is no
            //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
            expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
                .toEqual(JSON.stringify(modelTopology1).length);
            expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
                .toEqual(JSON.stringify(weightSpecs1).length);
            expect(saveResult.modelArtifactsInfo.weightDataBytes)
                .toEqual(weightData1.byteLength);
            expect(requestInits.length).toEqual(1);
            const init = requestInits[0];
            expect(init.method).toEqual('POST');
            const body = init.body;
            const jsonFile = body.get('model.json');
            const jsonFileReader = new FileReader();
            jsonFileReader.onload = (event) => {
                const modelJSON = 
                // tslint:disable-next-line:no-any
                JSON.parse(event.target.result);
                expect(modelJSON.modelTopology).toEqual(modelTopology1);
                expect(modelJSON.weightsManifest.length).toEqual(1);
                expect(modelJSON.weightsManifest[0].weights).toEqual(weightSpecs1);
                expect(modelJSON.trainingConfig).toEqual(trainingConfig1);
                const weightsFile = body.get('model.weights.bin');
                const weightsFileReader = new FileReader();
                weightsFileReader.onload = (event) => {
                    // tslint:disable-next-line:no-any
                    const weightData = event.target.result;
                    expect(new Uint8Array(weightData))
                        .toEqual(new Uint8Array(weightData1));
                    done();
                };
                weightsFileReader.onerror = ev => {
                    done.fail(weightsFileReader.error.message);
                };
                weightsFileReader.readAsArrayBuffer(weightsFile);
            };
            jsonFileReader.onerror = ev => {
                done.fail(jsonFileReader.error.message);
            };
            jsonFileReader.readAsText(jsonFile);
        })
            .catch(err => {
            done.fail(err.stack);
        });
    });
    it('Save topology only, default POST method', (done) => {
        const testStartDate = new Date();
        const handler = tf.io.getSaveHandlers('http://model-upload-test')[0];
        const topologyOnlyArtifacts = { modelTopology: modelTopology1 };
        handler.save(topologyOnlyArtifacts)
            .then(saveResult => {
            expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
                .toBeGreaterThanOrEqual(testStartDate.getTime());
            // Note: The following two assertions work only because there is no
            //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
            expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
                .toEqual(JSON.stringify(modelTopology1).length);
            expect(saveResult.modelArtifactsInfo.weightSpecsBytes).toEqual(0);
            expect(saveResult.modelArtifactsInfo.weightDataBytes).toEqual(0);
            expect(requestInits.length).toEqual(1);
            const init = requestInits[0];
            expect(init.method).toEqual('POST');
            const body = init.body;
            const jsonFile = body.get('model.json');
            const jsonFileReader = new FileReader();
            jsonFileReader.onload = (event) => {
                // tslint:disable-next-line:no-any
                const modelJSON = JSON.parse(event.target.result);
                expect(modelJSON.modelTopology).toEqual(modelTopology1);
                // No weights should have been sent to the server.
                expect(body.get('model.weights.bin')).toEqual(null);
                done();
            };
            jsonFileReader.onerror = event => {
                done.fail(jsonFileReader.error.message);
            };
            jsonFileReader.readAsText(jsonFile);
        })
            .catch(err => {
            done.fail(err.stack);
        });
    });
    it('Save topology and weights, PUT method, extra headers', (done) => {
        const testStartDate = new Date();
        const handler = tf.io.http('model-upload-test', {
            requestInit: {
                method: 'PUT',
                headers: { 'header_key_1': 'header_value_1', 'header_key_2': 'header_value_2' }
            }
        });
        handler.save(artifacts1)
            .then(saveResult => {
            expect(saveResult.modelArtifactsInfo.dateSaved.getTime())
                .toBeGreaterThanOrEqual(testStartDate.getTime());
            // Note: The following two assertions work only because there is no
            //   non-ASCII characters in `modelTopology1` and `weightSpecs1`.
            expect(saveResult.modelArtifactsInfo.modelTopologyBytes)
                .toEqual(JSON.stringify(modelTopology1).length);
            expect(saveResult.modelArtifactsInfo.weightSpecsBytes)
                .toEqual(JSON.stringify(weightSpecs1).length);
            expect(saveResult.modelArtifactsInfo.weightDataBytes)
                .toEqual(weightData1.byteLength);
            expect(requestInits.length).toEqual(1);
            const init = requestInits[0];
            expect(init.method).toEqual('PUT');
            // Check headers.
            expect(init.headers).toEqual({
                'header_key_1': 'header_value_1',
                'header_key_2': 'header_value_2'
            });
            const body = init.body;
            const jsonFile = body.get('model.json');
            const jsonFileReader = new FileReader();
            jsonFileReader.onload = (event) => {
                const modelJSON = 
                // tslint:disable-next-line:no-any
                JSON.parse(event.target.result);
                expect(modelJSON.format).toEqual('layers-model');
                expect(modelJSON.generatedBy).toEqual('TensorFlow.js v0.0.0');
                expect(modelJSON.convertedBy).toEqual(null);
                expect(modelJSON.modelTopology).toEqual(modelTopology1);
                expect(modelJSON.modelInitializer).toEqual({});
                expect(modelJSON.weightsManifest.length).toEqual(1);
                expect(modelJSON.weightsManifest[0].weights).toEqual(weightSpecs1);
                expect(modelJSON.trainingConfig).toEqual(trainingConfig1);
                const weightsFile = body.get('model.weights.bin');
                const weightsFileReader = new FileReader();
                weightsFileReader.onload = (event) => {
                    // tslint:disable-next-line:no-any
                    const weightData = event.target.result;
                    expect(new Uint8Array(weightData))
                        .toEqual(new Uint8Array(weightData1));
                    done();
                };
                weightsFileReader.onerror = event => {
                    done.fail(weightsFileReader.error.message);
                };
                weightsFileReader.readAsArrayBuffer(weightsFile);
            };
            jsonFileReader.onerror = event => {
                done.fail(jsonFileReader.error.message);
            };
            jsonFileReader.readAsText(jsonFile);
        })
            .catch(err => {
            done.fail(err.stack);
        });
    });
    it('404 response causes Error', (done) => {
        const handler = tf.io.getSaveHandlers('http://invalid/path')[0];
        handler.save(artifacts1)
            .then(saveResult => {
            done.fail('Calling http at invalid URL succeeded ' +
                'unexpectedly');
        })
            .catch(err => {
            done();
        });
    });
    it('getLoadHandlers with one URL string', () => {
        const handlers = tf.io.getLoadHandlers('http://foo/model.json');
        expect(handlers.length).toEqual(1);
        expect(handlers[0] instanceof HTTPRequest).toEqual(true);
    });
    it('Existing body leads to Error', () => {
        expect(() => tf.io.http('model-upload-test', {
            requestInit: { body: 'existing body' }
        })).toThrowError(/requestInit is expected to have no pre-existing body/);
    });
    it('Empty, null or undefined URL paths lead to Error', () => {
        expect(() => tf.io.http(null))
            .toThrowError(/must not be null, undefined or empty/);
        expect(() => tf.io.http(undefined))
            .toThrowError(/must not be null, undefined or empty/);
        expect(() => tf.io.http(''))
            .toThrowError(/must not be null, undefined or empty/);
    });
    it('router', () => {
        expect(httpRouter('http://bar/foo') instanceof HTTPRequest).toEqual(true);
        expect(httpRouter('https://localhost:5000/upload') instanceof HTTPRequest)
            .toEqual(true);
        expect(httpRouter('localhost://foo')).toBeNull();
        expect(httpRouter('foo:5000/bar')).toBeNull();
    });
});
describeWithFlags('parseUrl', BROWSER_ENVS, () => {
    it('should parse url with no suffix', () => {
        const url = 'http://google.com/file';
        const [prefix, suffix] = parseUrl(url);
        expect(prefix).toEqual('http://google.com/');
        expect(suffix).toEqual('');
    });
    it('should parse url with suffix', () => {
        const url = 'http://google.com/file?param=1';
        const [prefix, suffix] = parseUrl(url);
        expect(prefix).toEqual('http://google.com/');
        expect(suffix).toEqual('?param=1');
    });
    it('should parse url with multiple serach params', () => {
        const url = 'http://google.com/a?x=1/file?param=1';
        const [prefix, suffix] = parseUrl(url);
        expect(prefix).toEqual('http://google.com/a?x=1/');
        expect(suffix).toEqual('?param=1');
    });
});
describeWithFlags('http-load', BROWSER_ENVS, () => {
    describe('JSON model', () => {
        let requestInits;
        beforeEach(() => {
            requestInits = {};
        });
        it('1 group, 2 weights, 1 path', async () => {
            const weightManifest1 = [{
                    paths: ['weightfile0'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            const floatData = new Float32Array([1, 3, 3, 7, 4]);
            setupFakeWeightFiles({
                './model.json': {
                    data: JSON.stringify({
                        modelTopology: modelTopology1,
                        weightsManifest: weightManifest1,
                        format: 'tfjs-graph-model',
                        generatedBy: '1.15',
                        convertedBy: '1.3.1',
                        signature: null,
                        userDefinedMetadata: {},
                        modelInitializer: {}
                    }),
                    contentType: 'application/json'
                },
                './weightfile0': { data: floatData, contentType: 'application/octet-stream' },
            }, requestInits);
            const handler = tf.io.http('./model.json');
            const modelArtifacts = await handler.load();
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
            expect(modelArtifacts.format).toEqual('tfjs-graph-model');
            expect(modelArtifacts.generatedBy).toEqual('1.15');
            expect(modelArtifacts.convertedBy).toEqual('1.3.1');
            expect(modelArtifacts.userDefinedMetadata).toEqual({});
            expect(modelArtifacts.modelInitializer).toEqual({});
            expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);
            expect(Object.keys(requestInits).length).toEqual(2);
            // Assert that fetch is invoked with `window` as the context.
            expect(fetchSpy.calls.mostRecent().object).toEqual(window);
        });
        it('1 group, 2 weights, 1 path, with requestInit', async () => {
            const weightManifest1 = [{
                    paths: ['weightfile0'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            const floatData = new Float32Array([1, 3, 3, 7, 4]);
            setupFakeWeightFiles({
                './model.json': {
                    data: JSON.stringify({
                        modelTopology: modelTopology1,
                        weightsManifest: weightManifest1
                    }),
                    contentType: 'application/json'
                },
                './weightfile0': { data: floatData, contentType: 'application/octet-stream' },
            }, requestInits);
            const handler = tf.io.http('./model.json', { requestInit: { headers: { 'header_key_1': 'header_value_1' } } });
            const modelArtifacts = await handler.load();
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);
            expect(Object.keys(requestInits).length).toEqual(2);
            expect(Object.keys(requestInits).length).toEqual(2);
            expect(requestInits['./model.json'].headers['header_key_1'])
                .toEqual('header_value_1');
            expect(requestInits['./weightfile0'].headers['header_key_1'])
                .toEqual('header_value_1');
            expect(fetchSpy.calls.mostRecent().object).toEqual(window);
        });
        it('1 group, 2 weight, 2 paths', async () => {
            const weightManifest1 = [{
                    paths: ['weightfile0', 'weightfile1'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            const floatData1 = new Float32Array([1, 3, 3]);
            const floatData2 = new Float32Array([7, 4]);
            setupFakeWeightFiles({
                './model.json': {
                    data: JSON.stringify({
                        modelTopology: modelTopology1,
                        weightsManifest: weightManifest1
                    }),
                    contentType: 'application/json'
                },
                './weightfile0': { data: floatData1, contentType: 'application/octet-stream' },
                './weightfile1': { data: floatData2, contentType: 'application/octet-stream' }
            }, requestInits);
            const handler = tf.io.http('./model.json');
            const modelArtifacts = await handler.load();
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(new Float32Array([1, 3, 3, 7, 4]));
        });
        it('2 groups, 2 weight, 2 paths', async () => {
            const weightsManifest = [
                {
                    paths: ['weightfile0'],
                    weights: [{
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        }]
                },
                {
                    paths: ['weightfile1'],
                    weights: [{
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }],
                }
            ];
            const floatData1 = new Float32Array([1, 3, 3]);
            const floatData2 = new Float32Array([7, 4]);
            setupFakeWeightFiles({
                './model.json': {
                    data: JSON.stringify({ modelTopology: modelTopology1, weightsManifest }),
                    contentType: 'application/json'
                },
                './weightfile0': { data: floatData1, contentType: 'application/octet-stream' },
                './weightfile1': { data: floatData2, contentType: 'application/octet-stream' }
            }, requestInits);
            const handler = tf.io.http('./model.json');
            const modelArtifacts = await handler.load();
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightsManifest[0].weights.concat(weightsManifest[1].weights));
            expect(new Float32Array(modelArtifacts.weightData))
                .toEqual(new Float32Array([1, 3, 3, 7, 4]));
        });
        it('2 groups, 2 weight, 2 paths, Int32 and Uint8 Data', async () => {
            const weightsManifest = [
                {
                    paths: ['weightfile0'],
                    weights: [{
                            name: 'fooWeight',
                            shape: [3, 1],
                            dtype: 'int32',
                        }]
                },
                {
                    paths: ['weightfile1'],
                    weights: [{
                            name: 'barWeight',
                            shape: [2],
                            dtype: 'bool',
                        }],
                }
            ];
            const floatData1 = new Int32Array([1, 3, 3]);
            const floatData2 = new Uint8Array([7, 4]);
            setupFakeWeightFiles({
                'path1/model.json': {
                    data: JSON.stringify({ modelTopology: modelTopology1, weightsManifest }),
                    contentType: 'application/json'
                },
                'path1/weightfile0': { data: floatData1, contentType: 'application/octet-stream' },
                'path1/weightfile1': { data: floatData2, contentType: 'application/octet-stream' }
            }, requestInits);
            const handler = tf.io.http('path1/model.json');
            const modelArtifacts = await handler.load();
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightsManifest[0].weights.concat(weightsManifest[1].weights));
            expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                .toEqual(new Int32Array([1, 3, 3]));
            expect(new Uint8Array(modelArtifacts.weightData.slice(12, 14)))
                .toEqual(new Uint8Array([7, 4]));
        });
        it('topology only', async () => {
            setupFakeWeightFiles({
                './model.json': {
                    data: JSON.stringify({ modelTopology: modelTopology1 }),
                    contentType: 'application/json'
                },
            }, requestInits);
            const handler = tf.io.http('./model.json');
            const modelArtifacts = await handler.load();
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs).toBeUndefined();
            expect(modelArtifacts.weightData).toBeUndefined();
        });
        it('weights only', async () => {
            const weightsManifest = [
                {
                    paths: ['weightfile0'],
                    weights: [{
                            name: 'fooWeight',
                            shape: [3, 1],
                            dtype: 'int32',
                        }]
                },
                {
                    paths: ['weightfile1'],
                    weights: [{
                            name: 'barWeight',
                            shape: [2],
                            dtype: 'float32',
                        }],
                }
            ];
            const floatData1 = new Int32Array([1, 3, 3]);
            const floatData2 = new Float32Array([-7, -4]);
            setupFakeWeightFiles({
                'path1/model.json': {
                    data: JSON.stringify({ weightsManifest }),
                    contentType: 'application/json'
                },
                'path1/weightfile0': { data: floatData1, contentType: 'application/octet-stream' },
                'path1/weightfile1': { data: floatData2, contentType: 'application/octet-stream' }
            }, requestInits);
            const handler = tf.io.http('path1/model.json');
            const modelArtifacts = await handler.load();
            expect(modelArtifacts.modelTopology).toBeUndefined();
            expect(modelArtifacts.weightSpecs)
                .toEqual(weightsManifest[0].weights.concat(weightsManifest[1].weights));
            expect(new Int32Array(modelArtifacts.weightData.slice(0, 12)))
                .toEqual(new Int32Array([1, 3, 3]));
            expect(new Float32Array(modelArtifacts.weightData.slice(12, 20)))
                .toEqual(new Float32Array([-7, -4]));
        });
        it('Missing modelTopology and weightsManifest leads to error', async () => {
            setupFakeWeightFiles({
                'path1/model.json': { data: JSON.stringify({}), contentType: 'application/json' }
            }, requestInits);
            const handler = tf.io.http('path1/model.json');
            handler.load()
                .then(modelTopology1 => {
                fail('Loading from missing modelTopology and weightsManifest ' +
                    'succeeded unexpectedly.');
            })
                .catch(err => {
                expect(err.message)
                    .toMatch(/contains neither model topology or manifest/);
            });
        });
        it('with fetch rejection leads to error', async () => {
            setupFakeWeightFiles({
                'path1/model.json': { data: JSON.stringify({}), contentType: 'text/html' }
            }, requestInits);
            const handler = tf.io.http('path2/model.json');
            try {
                const data = await handler.load();
                expect(data).toBeDefined();
                fail('Loading with fetch rejection succeeded unexpectedly.');
            }
            catch (err) {
                // This error is mocked in beforeEach
                expect(err).toEqual('path not found');
            }
        });
        it('Provide WeightFileTranslateFunc', async () => {
            const weightManifest1 = [{
                    paths: ['weightfile0'],
                    weights: [
                        {
                            name: 'dense/kernel',
                            shape: [3, 1],
                            dtype: 'float32',
                        },
                        {
                            name: 'dense/bias',
                            shape: [2],
                            dtype: 'float32',
                        }
                    ]
                }];
            const floatData = new Float32Array([1, 3, 3, 7, 4]);
            setupFakeWeightFiles({
                './model.json': {
                    data: JSON.stringify({
                        modelTopology: modelTopology1,
                        weightsManifest: weightManifest1
                    }),
                    contentType: 'application/json'
                },
                'auth_weightfile0': { data: floatData, contentType: 'application/octet-stream' },
            }, requestInits);
            async function prefixWeightUrlConverter(weightFile) {
                // Add 'auth_' prefix to the weight file url.
                return new Promise(resolve => setTimeout(resolve, 1, 'auth_' + weightFile));
            }
            const handler = tf.io.http('./model.json', {
                requestInit: { headers: { 'header_key_1': 'header_value_1' } },
                weightUrlConverter: prefixWeightUrlConverter
            });
            const modelArtifacts = await handler.load();
            expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
            expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
            expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);
            expect(Object.keys(requestInits).length).toEqual(2);
            expect(Object.keys(requestInits).length).toEqual(2);
            expect(requestInits['./model.json'].headers['header_key_1'])
                .toEqual('header_value_1');
            expect(requestInits['auth_weightfile0'].headers['header_key_1'])
                .toEqual('header_value_1');
            expect(fetchSpy.calls.mostRecent().object).toEqual(window);
        });
    });
    it('Overriding BrowserHTTPRequest fetchFunc', async () => {
        const weightManifest1 = [{
                paths: ['weightfile0'],
                weights: [
                    {
                        name: 'dense/kernel',
                        shape: [3, 1],
                        dtype: 'float32',
                    },
                    {
                        name: 'dense/bias',
                        shape: [2],
                        dtype: 'float32',
                    }
                ]
            }];
        const floatData = new Float32Array([1, 3, 3, 7, 4]);
        const fetchInputs = [];
        const fetchInits = [];
        async function customFetch(input, init) {
            fetchInputs.push(input);
            fetchInits.push(init);
            if (input === './model.json') {
                return new Response(JSON.stringify({
                    modelTopology: modelTopology1,
                    weightsManifest: weightManifest1,
                    trainingConfig: trainingConfig1
                }), { status: 200, headers: { 'content-type': 'application/json' } });
            }
            else if (input === './weightfile0') {
                return new Response(floatData, {
                    status: 200,
                    headers: { 'content-type': 'application/octet-stream' }
                });
            }
            else {
                return new Response(null, { status: 404 });
            }
        }
        const handler = tf.io.http('./model.json', { requestInit: { credentials: 'include' }, fetchFunc: customFetch });
        const modelArtifacts = await handler.load();
        expect(modelArtifacts.modelTopology).toEqual(modelTopology1);
        expect(modelArtifacts.trainingConfig).toEqual(trainingConfig1);
        expect(modelArtifacts.weightSpecs).toEqual(weightManifest1[0].weights);
        expect(new Float32Array(modelArtifacts.weightData)).toEqual(floatData);
        expect(fetchInputs).toEqual(['./model.json', './weightfile0']);
        expect(fetchInits.length).toEqual(2);
        expect(fetchInits[0].credentials).toEqual('include');
        expect(fetchInits[1].credentials).toEqual('include');
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaHR0cF90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9pby9odHRwX3Rlc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxLQUFLLEVBQUUsTUFBTSxVQUFVLENBQUM7QUFDL0IsT0FBTyxFQUFDLFlBQVksRUFBRSxXQUFXLEVBQUUsaUJBQWlCLEVBQUUsU0FBUyxFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFDeEYsT0FBTyxFQUFDLFdBQVcsRUFBRSxVQUFVLEVBQUUsUUFBUSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBRXpELGFBQWE7QUFDYixNQUFNLGNBQWMsR0FBTztJQUN6QixZQUFZLEVBQUUsWUFBWTtJQUMxQixlQUFlLEVBQUUsT0FBTztJQUN4QixRQUFRLEVBQUUsQ0FBQztZQUNULFlBQVksRUFBRSxPQUFPO1lBQ3JCLFFBQVEsRUFBRTtnQkFDUixvQkFBb0IsRUFBRTtvQkFDcEIsWUFBWSxFQUFFLGlCQUFpQjtvQkFDL0IsUUFBUSxFQUFFO3dCQUNSLGNBQWMsRUFBRSxTQUFTO3dCQUN6QixPQUFPLEVBQUUsR0FBRzt3QkFDWixNQUFNLEVBQUUsSUFBSTt3QkFDWixNQUFNLEVBQUUsU0FBUztxQkFDbEI7aUJBQ0Y7Z0JBQ0QsTUFBTSxFQUFFLE9BQU87Z0JBQ2YsbUJBQW1CLEVBQUUsSUFBSTtnQkFDekIsa0JBQWtCLEVBQUUsSUFBSTtnQkFDeEIsaUJBQWlCLEVBQUUsSUFBSTtnQkFDdkIsT0FBTyxFQUFFLFNBQVM7Z0JBQ2xCLFlBQVksRUFBRSxRQUFRO2dCQUN0QixXQUFXLEVBQUUsSUFBSTtnQkFDakIsb0JBQW9CLEVBQUUsSUFBSTtnQkFDMUIsa0JBQWtCLEVBQUUsRUFBQyxZQUFZLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBRSxFQUFFLEVBQUM7Z0JBQ3pELE9BQU8sRUFBRSxDQUFDO2dCQUNWLG1CQUFtQixF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QyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDckMsTUFBTSxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxXQUFXLENBQUMsQ0FBQyxPQUFPLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDckQsTUFBTSxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxXQUFXLENBQUMsQ0FBQyxPQUFPLENBQUMsU0FBUyxDQUFDLENBQUM7SUFDdkQsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICogYXMgdGYgZnJvbSAnLi4vaW5kZXgnO1xuaW1wb3J0IHtCUk9XU0VSX0VOVlMsIENIUk9NRV9FTlZTLCBkZXNjcmliZVdpdGhGbGFncywgTk9ERV9FTlZTfSBmcm9tICcuLi9qYXNtaW5lX3V0aWwnO1xuaW1wb3J0IHtIVFRQUmVxdWVzdCwgaHR0cFJvdXRlciwgcGFyc2VVcmx9IGZyb20gJy4vaHR0cCc7XG5cbi8vIFRlc3QgZGF0YS5cbmNvbnN0IG1vZGVsVG9wb2xvZ3kxOiB7fSA9IHtcbiAgJ2NsYXNzX25hbWUnOiAnU2VxdWVudGlhbCcsXG4gICdrZXJhc192ZXJzaW9uJzogJzIuMS40JyxcbiAgJ2NvbmZpZyc6IFt7XG4gICAgJ2NsYXNzX25hbWUnOiAnRGVuc2UnLFxuICAgICdjb25maWcnOiB7XG4gICAgICAna2VybmVsX2luaXRpYWxpemVyJzoge1xuICAgICAgICAnY2xhc3NfbmFtZSc6ICdWYXJpYW5jZVNjYWxpbmcnLFxuICAgICAgICAnY29uZmlnJzoge1xuICAgICAgICAgICdkaXN0cmlidXRpb24nOiAndW5pZm9ybScsXG4gICAgICAgICAgJ3NjYWxlJzogMS4wLFxuICAgICAgICAgICdzZWVkJzogbnVsbCxcbiAgICAgICAgICAnbW9kZSc6ICdmYW5fYXZnJ1xuICAgICAgICB9XG4gICAgICB9LFxuICAgICAgJ25hbWUnOiAnZGVuc2UnLFxuICAgICAgJ2tlcm5lbF9jb25zdHJhaW50JzogbnVsbCxcbiAgICAgICdiaWFzX3JlZ3VsYXJpemVyJzogbnVsbCxcbiAgICAgICdiaWFzX2NvbnN0cmFpbnQnOiBudWxsLFxuICAgICAgJ2R0eXBlJzogJ2Zsb2F0MzInLFxuICAgICAgJ2FjdGl2YXRpb24nOiAnbGluZWFyJyxcbiAgICAgICd0cmFpbmFibGUnOiB0cnVlLFxuICAgICAgJ2tlcm5lbF9yZWd1bGFyaXplcic6IG51bGwsXG4gICAgICAnYmlhc19pbml0aWFsaXplcic6IHsnY2xhc3NfbmFtZSc6ICdaZXJvcycsICdjb25maWcnOiB7fX0sXG4gICAgICAndW5pdHMnOiAxLFxuICAgICAgJ2JhdGNoX2lucHV0X3NoYXBlJzogW251bGwsIDNdLFxuICAgICAgJ3VzZV9iaWFzJzogdHJ1ZSxcbiAgICAgICdhY3Rpdml0eV9yZWd1bGFyaXplcic6IG51bGxcbiAgICB9XG4gIH1dLFxuICAnYmFja2VuZCc6ICd0ZW5zb3JmbG93J1xufTtcbmNvbnN0IHRyYWluaW5nQ29uZmlnMTogdGYuaW8uVHJhaW5pbmdDb25maWcgPSB7XG4gIGxvc3M6ICdjYXRlZ29yaWNhbF9jcm9zc2VudHJvcHknLFxuICBtZXRyaWNzOiBbJ2FjY3VyYWN5J10sXG4gIG9wdGltaXplcl9jb25maWc6IHtjbGFzc19uYW1lOiAnU0dEJywgY29uZmlnOiB7bGVhcm5pbmdSYXRlOiAwLjF9fVxufTtcblxubGV0IGZldGNoU3B5OiBqYXNtaW5lLlNweTtcblxudHlwZSBUeXBlZEFycmF5cyA9IEZsb2F0MzJBcnJheXxJbnQzMkFycmF5fFVpbnQ4QXJyYXl8VWludDE2QXJyYXk7XG5jb25zdCBmYWtlUmVzcG9uc2UgPVxuICAgIChib2R5OiBzdHJpbmd8VHlwZWRBcnJheXN8QXJyYXlCdWZmZXIsIGNvbnRlbnRUeXBlOiBzdHJpbmcsIHBhdGg6IHN0cmluZykgPT5cbiAgICAgICAgKHtcbiAgICAgICAgICBvazogdHJ1ZSxcbiAgICAgICAgICBqc29uKCkge1xuICAgICAgICAgICAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShKU09OLnBhcnNlKGJvZHkgYXMgc3RyaW5nKSk7XG4gICAgICAgICAgfSxcbiAgICAgICAgICBhcnJheUJ1ZmZlcigpIHtcbiAgICAgICAgICAgIGNvbnN0IGJ1ZjogQXJyYXlCdWZmZXIgPSAoYm9keSBhcyBUeXBlZEFycmF5cykuYnVmZmVyID9cbiAgICAgICAgICAgICAgICAoYm9keSBhcyBUeXBlZEFycmF5cykuYnVmZmVyIDpcbiAgICAgICAgICAgICAgICBib2R5IGFzIEFycmF5QnVmZmVyO1xuICAgICAgICAgICAgcmV0dXJuIFByb21pc2UucmVzb2x2ZShidWYpO1xuICAgICAgICAgIH0sXG4gICAgICAgICAgaGVhZGVyczoge2dldDogKGtleTogc3RyaW5nKSA9PiBjb250ZW50VHlwZX0sXG4gICAgICAgICAgdXJsOiBwYXRoXG4gICAgICAgIH0pO1xuXG5jb25zdCBzZXR1cEZha2VXZWlnaHRGaWxlcyA9XG4gICAgKGZpbGVCdWZmZXJNYXA6IHtcbiAgICAgIFtmaWxlbmFtZTogc3RyaW5nXToge1xuICAgICAgICBkYXRhOiBzdHJpbmd8RmxvYXQzMkFycmF5fEludDMyQXJyYXl8QXJyYXlCdWZmZXJ8VWludDhBcnJheXxVaW50MTZBcnJheSxcbiAgICAgICAgY29udGVudFR5cGU6IHN0cmluZ1xuICAgICAgfVxuICAgIH0sXG4gICAgIHJlcXVlc3RJbml0czoge1trZXk6IHN0cmluZ106IFJlcXVlc3RJbml0fSkgPT4ge1xuICAgICAgZmV0Y2hTcHkgPSBzcHlPbih0Zi5lbnYoKS5wbGF0Zm9ybSwgJ2ZldGNoJylcbiAgICAgICAgICAgICAgICAgICAgIC5hbmQuY2FsbEZha2UoKHBhdGg6IHN0cmluZywgaW5pdDogUmVxdWVzdEluaXQpID0+IHtcbiAgICAgICAgICAgICAgICAgICAgICAgaWYgKGZpbGVCdWZmZXJNYXBbcGF0aF0pIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICByZXF1ZXN0SW5pdHNbcGF0aF0gPSBpbml0O1xuICAgICAgICAgICAgICAgICAgICAgICAgIHJldHVybiBQcm9taXNlLnJlc29sdmUoZmFrZVJlc3BvbnNlKFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICBmaWxlQnVmZmVyTWFwW3BhdGhdLmRhdGEsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgIGZpbGVCdWZmZXJNYXBbcGF0aF0uY29udGVudFR5cGUsIHBhdGgpKTtcbiAgICAgICAgICAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICAgICAgICAgICByZXR1cm4gUHJvbWlzZS5yZWplY3QoJ3BhdGggbm90IGZvdW5kJyk7XG4gICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgIH0pO1xuICAgIH07XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdodHRwLWxvYWQgZmV0Y2gnLCBOT0RFX0VOVlMsICgpID0+IHtcbiAgbGV0IHJlcXVlc3RJbml0czoge1trZXk6IHN0cmluZ106IHtoZWFkZXJzOiB7W2tleTogc3RyaW5nXTogc3RyaW5nfX19O1xuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gIGxldCBvcmlnaW5hbEZldGNoOiBhbnk7XG4gIC8vIHNpbXVsYXRlIGEgZmV0Y2ggcG9seWZpbGwsIHRoaXMgbmVlZHMgdG8gYmUgbm9uLW51bGwgZm9yIHNweU9uIHRvIHdvcmtcbiAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIG9yaWdpbmFsRmV0Y2ggPSAoZ2xvYmFsIGFzIGFueSkuZmV0Y2g7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIChnbG9iYWwgYXMgYW55KS5mZXRjaCA9ICgpID0+IHt9O1xuICAgIHJlcXVlc3RJbml0cyA9IHt9O1xuICB9KTtcblxuICBhZnRlckFsbCgoKSA9PiB7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIChnbG9iYWwgYXMgYW55KS5mZXRjaCA9IG9yaWdpbmFsRmV0Y2g7XG4gIH0pO1xuXG4gIGl0KCcxIGdyb3VwLCAyIHdlaWdodHMsIDEgcGF0aCcsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCB3ZWlnaHRNYW5pZmVzdDE6IHRmLmlvLldlaWdodHNNYW5pZmVzdENvbmZpZyA9IFt7XG4gICAgICBwYXRoczogWyd3ZWlnaHRmaWxlMCddLFxuICAgICAgd2VpZ2h0czogW1xuICAgICAgICB7XG4gICAgICAgICAgbmFtZTogJ2RlbnNlL2tlcm5lbCcsXG4gICAgICAgICAgc2hhcGU6IFszLCAxXSxcbiAgICAgICAgICBkdHlwZTogJ2Zsb2F0MzInLFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgbmFtZTogJ2RlbnNlL2JpYXMnLFxuICAgICAgICAgIHNoYXBlOiBbMl0sXG4gICAgICAgICAgZHR5cGU6ICdmbG9hdDMyJyxcbiAgICAgICAgfVxuICAgICAgXVxuICAgIH1dO1xuICAgIGNvbnN0IGZsb2F0RGF0YSA9IG5ldyBGbG9hdDMyQXJyYXkoWzEsIDMsIDMsIDcsIDRdKTtcbiAgICBzZXR1cEZha2VXZWlnaHRGaWxlcyhcbiAgICAgICAge1xuICAgICAgICAgICcuL21vZGVsLmpzb24nOiB7XG4gICAgICAgICAgICBkYXRhOiBKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgIG1vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxLFxuICAgICAgICAgICAgICB3ZWlnaHRzTWFuaWZlc3Q6IHdlaWdodE1hbmlmZXN0MSxcbiAgICAgICAgICAgICAgZm9ybWF0OiAndGZqcy1sYXllcnMnLFxuICAgICAgICAgICAgICBnZW5lcmF0ZWRCeTogJzEuMTUnLFxuICAgICAgICAgICAgICBjb252ZXJ0ZWRCeTogJzEuMy4xJyxcbiAgICAgICAgICAgICAgc2lnbmF0dXJlOiBudWxsLFxuICAgICAgICAgICAgICB1c2VyRGVmaW5lZE1ldGFkYXRhOiB7fVxuICAgICAgICAgICAgfSksXG4gICAgICAgICAgICBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nXG4gICAgICAgICAgfSxcbiAgICAgICAgICAnLi93ZWlnaHRmaWxlMCc6XG4gICAgICAgICAgICAgIHtkYXRhOiBmbG9hdERhdGEsIGNvbnRlbnRUeXBlOiAnYXBwbGljYXRpb24vb2N0ZXQtc3RyZWFtJ30sXG4gICAgICAgIH0sXG4gICAgICAgIHJlcXVlc3RJbml0cyk7XG5cbiAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uaHR0cCgnLi9tb2RlbC5qc29uJyk7XG4gICAgY29uc3QgbW9kZWxBcnRpZmFjdHMgPSBhd2FpdCBoYW5kbGVyLmxvYWQoKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMubW9kZWxUb3BvbG9neSkudG9FcXVhbChtb2RlbFRvcG9sb2d5MSk7XG4gICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzKS50b0VxdWFsKHdlaWdodE1hbmlmZXN0MVswXS53ZWlnaHRzKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMuZm9ybWF0KS50b0VxdWFsKCd0ZmpzLWxheWVycycpO1xuICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5nZW5lcmF0ZWRCeSkudG9FcXVhbCgnMS4xNScpO1xuICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5jb252ZXJ0ZWRCeSkudG9FcXVhbCgnMS4zLjEnKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMudXNlckRlZmluZWRNZXRhZGF0YSkudG9FcXVhbCh7fSk7XG4gICAgZXhwZWN0KG5ldyBGbG9hdDMyQXJyYXkobW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YSkpLnRvRXF1YWwoZmxvYXREYXRhKTtcbiAgfSk7XG5cbiAgaXQoJ3Rocm93IGV4Y2VwdGlvbiBpZiBubyBmZXRjaCBwb2x5ZmlsbCcsICgpID0+IHtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgZGVsZXRlIChnbG9iYWwgYXMgYW55KS5mZXRjaDtcbiAgICB0cnkge1xuICAgICAgdGYuaW8uaHR0cCgnLi9tb2RlbC5qc29uJyk7XG4gICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICBleHBlY3QoZXJyLm1lc3NhZ2UpLnRvTWF0Y2goL1VuYWJsZSB0byBmaW5kIGZldGNoIHBvbHlmaWxsLi8pO1xuICAgIH1cbiAgfSk7XG59KTtcblxuLy8gVHVybmVkIG9mZiBmb3Igb3RoZXIgYnJvd3NlcnMgZHVlIHRvOlxuLy8gaHR0cHM6Ly9naXRodWIuY29tL3RlbnNvcmZsb3cvdGZqcy9pc3N1ZXMvNDI2XG5kZXNjcmliZVdpdGhGbGFncygnaHR0cC1zYXZlJywgQ0hST01FX0VOVlMsICgpID0+IHtcbiAgLy8gVGVzdCBkYXRhLlxuICBjb25zdCB3ZWlnaHRTcGVjczE6IHRmLmlvLldlaWdodHNNYW5pZmVzdEVudHJ5W10gPSBbXG4gICAge1xuICAgICAgbmFtZTogJ2RlbnNlL2tlcm5lbCcsXG4gICAgICBzaGFwZTogWzMsIDFdLFxuICAgICAgZHR5cGU6ICdmbG9hdDMyJyxcbiAgICB9LFxuICAgIHtcbiAgICAgIG5hbWU6ICdkZW5zZS9iaWFzJyxcbiAgICAgIHNoYXBlOiBbMV0sXG4gICAgICBkdHlwZTogJ2Zsb2F0MzInLFxuICAgIH1cbiAgXTtcbiAgY29uc3Qgd2VpZ2h0RGF0YTEgPSBuZXcgQXJyYXlCdWZmZXIoMTYpO1xuICBjb25zdCBhcnRpZmFjdHMxOiB0Zi5pby5Nb2RlbEFydGlmYWN0cyA9IHtcbiAgICBtb2RlbFRvcG9sb2d5OiBtb2RlbFRvcG9sb2d5MSxcbiAgICB3ZWlnaHRTcGVjczogd2VpZ2h0U3BlY3MxLFxuICAgIHdlaWdodERhdGE6IHdlaWdodERhdGExLFxuICAgIGZvcm1hdDogJ2xheWVycy1tb2RlbCcsXG4gICAgZ2VuZXJhdGVkQnk6ICdUZW5zb3JGbG93LmpzIHYwLjAuMCcsXG4gICAgY29udmVydGVkQnk6IG51bGwsXG4gICAgc2lnbmF0dXJlOiBudWxsLFxuICAgIHVzZXJEZWZpbmVkTWV0YWRhdGE6IHt9LFxuICAgIG1vZGVsSW5pdGlhbGl6ZXI6IHt9LFxuICAgIHRyYWluaW5nQ29uZmlnOiB0cmFpbmluZ0NvbmZpZzFcbiAgfTtcblxuICBsZXQgcmVxdWVzdEluaXRzOiBSZXF1ZXN0SW5pdFtdID0gW107XG5cbiAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgcmVxdWVzdEluaXRzID0gW107XG4gICAgc3B5T24odGYuZW52KCkucGxhdGZvcm0sICdmZXRjaCcpXG4gICAgICAgIC5hbmQuY2FsbEZha2UoKHBhdGg6IHN0cmluZywgaW5pdDogUmVxdWVzdEluaXQpID0+IHtcbiAgICAgICAgICBpZiAocGF0aCA9PT0gJ21vZGVsLXVwbG9hZC10ZXN0JyB8fFxuICAgICAgICAgICAgICBwYXRoID09PSAnaHR0cDovL21vZGVsLXVwbG9hZC10ZXN0Jykge1xuICAgICAgICAgICAgcmVxdWVzdEluaXRzLnB1c2goaW5pdCk7XG4gICAgICAgICAgICByZXR1cm4gUHJvbWlzZS5yZXNvbHZlKG5ldyBSZXNwb25zZShudWxsLCB7c3RhdHVzOiAyMDB9KSk7XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIHJldHVybiBQcm9taXNlLnJlamVjdChuZXcgUmVzcG9uc2UobnVsbCwge3N0YXR1czogNDA0fSkpO1xuICAgICAgICAgIH1cbiAgICAgICAgfSk7XG4gIH0pO1xuXG4gIGl0KCdTYXZlIHRvcG9sb2d5IGFuZCB3ZWlnaHRzLCBkZWZhdWx0IFBPU1QgbWV0aG9kJywgKGRvbmUpID0+IHtcbiAgICBjb25zdCB0ZXN0U3RhcnREYXRlID0gbmV3IERhdGUoKTtcbiAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uZ2V0U2F2ZUhhbmRsZXJzKCdodHRwOi8vbW9kZWwtdXBsb2FkLXRlc3QnKVswXTtcbiAgICBoYW5kbGVyLnNhdmUoYXJ0aWZhY3RzMSlcbiAgICAgICAgLnRoZW4oc2F2ZVJlc3VsdCA9PiB7XG4gICAgICAgICAgZXhwZWN0KHNhdmVSZXN1bHQubW9kZWxBcnRpZmFjdHNJbmZvLmRhdGVTYXZlZC5nZXRUaW1lKCkpXG4gICAgICAgICAgICAgIC50b0JlR3JlYXRlclRoYW5PckVxdWFsKHRlc3RTdGFydERhdGUuZ2V0VGltZSgpKTtcbiAgICAgICAgICAvLyBOb3RlOiBUaGUgZm9sbG93aW5nIHR3byBhc3NlcnRpb25zIHdvcmsgb25seSBiZWNhdXNlIHRoZXJlIGlzIG5vXG4gICAgICAgICAgLy8gICBub24tQVNDSUkgY2hhcmFjdGVycyBpbiBgbW9kZWxUb3BvbG9neTFgIGFuZCBgd2VpZ2h0U3BlY3MxYC5cbiAgICAgICAgICBleHBlY3Qoc2F2ZVJlc3VsdC5tb2RlbEFydGlmYWN0c0luZm8ubW9kZWxUb3BvbG9neUJ5dGVzKVxuICAgICAgICAgICAgICAudG9FcXVhbChKU09OLnN0cmluZ2lmeShtb2RlbFRvcG9sb2d5MSkubGVuZ3RoKTtcbiAgICAgICAgICBleHBlY3Qoc2F2ZVJlc3VsdC5tb2RlbEFydGlmYWN0c0luZm8ud2VpZ2h0U3BlY3NCeXRlcylcbiAgICAgICAgICAgICAgLnRvRXF1YWwoSlNPTi5zdHJpbmdpZnkod2VpZ2h0U3BlY3MxKS5sZW5ndGgpO1xuICAgICAgICAgIGV4cGVjdChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby53ZWlnaHREYXRhQnl0ZXMpXG4gICAgICAgICAgICAgIC50b0VxdWFsKHdlaWdodERhdGExLmJ5dGVMZW5ndGgpO1xuXG4gICAgICAgICAgZXhwZWN0KHJlcXVlc3RJbml0cy5sZW5ndGgpLnRvRXF1YWwoMSk7XG4gICAgICAgICAgY29uc3QgaW5pdCA9IHJlcXVlc3RJbml0c1swXTtcbiAgICAgICAgICBleHBlY3QoaW5pdC5tZXRob2QpLnRvRXF1YWwoJ1BPU1QnKTtcbiAgICAgICAgICBjb25zdCBib2R5ID0gaW5pdC5ib2R5IGFzIEZvcm1EYXRhO1xuICAgICAgICAgIGNvbnN0IGpzb25GaWxlID0gYm9keS5nZXQoJ21vZGVsLmpzb24nKSBhcyBGaWxlO1xuICAgICAgICAgIGNvbnN0IGpzb25GaWxlUmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTtcbiAgICAgICAgICBqc29uRmlsZVJlYWRlci5vbmxvYWQgPSAoZXZlbnQ6IEV2ZW50KSA9PiB7XG4gICAgICAgICAgICBjb25zdCBtb2RlbEpTT04gPVxuICAgICAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgICAgICAgICBKU09OLnBhcnNlKChldmVudC50YXJnZXQgYXMgYW55KS5yZXN1bHQpIGFzIHRmLmlvLk1vZGVsSlNPTjtcbiAgICAgICAgICAgIGV4cGVjdChtb2RlbEpTT04ubW9kZWxUb3BvbG9neSkudG9FcXVhbChtb2RlbFRvcG9sb2d5MSk7XG4gICAgICAgICAgICBleHBlY3QobW9kZWxKU09OLndlaWdodHNNYW5pZmVzdC5sZW5ndGgpLnRvRXF1YWwoMSk7XG4gICAgICAgICAgICBleHBlY3QobW9kZWxKU09OLndlaWdodHNNYW5pZmVzdFswXS53ZWlnaHRzKS50b0VxdWFsKHdlaWdodFNwZWNzMSk7XG4gICAgICAgICAgICBleHBlY3QobW9kZWxKU09OLnRyYWluaW5nQ29uZmlnKS50b0VxdWFsKHRyYWluaW5nQ29uZmlnMSk7XG5cbiAgICAgICAgICAgIGNvbnN0IHdlaWdodHNGaWxlID0gYm9keS5nZXQoJ21vZGVsLndlaWdodHMuYmluJykgYXMgRmlsZTtcbiAgICAgICAgICAgIGNvbnN0IHdlaWdodHNGaWxlUmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTtcbiAgICAgICAgICAgIHdlaWdodHNGaWxlUmVhZGVyLm9ubG9hZCA9IChldmVudDogRXZlbnQpID0+IHtcbiAgICAgICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICAgICAgICBjb25zdCB3ZWlnaHREYXRhID0gKGV2ZW50LnRhcmdldCBhcyBhbnkpLnJlc3VsdCBhcyBBcnJheUJ1ZmZlcjtcbiAgICAgICAgICAgICAgZXhwZWN0KG5ldyBVaW50OEFycmF5KHdlaWdodERhdGEpKVxuICAgICAgICAgICAgICAgICAgLnRvRXF1YWwobmV3IFVpbnQ4QXJyYXkod2VpZ2h0RGF0YTEpKTtcbiAgICAgICAgICAgICAgZG9uZSgpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHdlaWdodHNGaWxlUmVhZGVyLm9uZXJyb3IgPSBldiA9PiB7XG4gICAgICAgICAgICAgIGRvbmUuZmFpbCh3ZWlnaHRzRmlsZVJlYWRlci5lcnJvci5tZXNzYWdlKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICB3ZWlnaHRzRmlsZVJlYWRlci5yZWFkQXNBcnJheUJ1ZmZlcih3ZWlnaHRzRmlsZSk7XG4gICAgICAgICAgfTtcbiAgICAgICAgICBqc29uRmlsZVJlYWRlci5vbmVycm9yID0gZXYgPT4ge1xuICAgICAgICAgICAgZG9uZS5mYWlsKGpzb25GaWxlUmVhZGVyLmVycm9yLm1lc3NhZ2UpO1xuICAgICAgICAgIH07XG4gICAgICAgICAganNvbkZpbGVSZWFkZXIucmVhZEFzVGV4dChqc29uRmlsZSk7XG4gICAgICAgIH0pXG4gICAgICAgIC5jYXRjaChlcnIgPT4ge1xuICAgICAgICAgIGRvbmUuZmFpbChlcnIuc3RhY2spO1xuICAgICAgICB9KTtcbiAgfSk7XG5cbiAgaXQoJ1NhdmUgdG9wb2xvZ3kgb25seSwgZGVmYXVsdCBQT1NUIG1ldGhvZCcsIChkb25lKSA9PiB7XG4gICAgY29uc3QgdGVzdFN0YXJ0RGF0ZSA9IG5ldyBEYXRlKCk7XG4gICAgY29uc3QgaGFuZGxlciA9IHRmLmlvLmdldFNhdmVIYW5kbGVycygnaHR0cDovL21vZGVsLXVwbG9hZC10ZXN0JylbMF07XG4gICAgY29uc3QgdG9wb2xvZ3lPbmx5QXJ0aWZhY3RzID0ge21vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxfTtcbiAgICBoYW5kbGVyLnNhdmUodG9wb2xvZ3lPbmx5QXJ0aWZhY3RzKVxuICAgICAgICAudGhlbihzYXZlUmVzdWx0ID0+IHtcbiAgICAgICAgICBleHBlY3Qoc2F2ZVJlc3VsdC5tb2RlbEFydGlmYWN0c0luZm8uZGF0ZVNhdmVkLmdldFRpbWUoKSlcbiAgICAgICAgICAgICAgLnRvQmVHcmVhdGVyVGhhbk9yRXF1YWwodGVzdFN0YXJ0RGF0ZS5nZXRUaW1lKCkpO1xuICAgICAgICAgIC8vIE5vdGU6IFRoZSBmb2xsb3dpbmcgdHdvIGFzc2VydGlvbnMgd29yayBvbmx5IGJlY2F1c2UgdGhlcmUgaXMgbm9cbiAgICAgICAgICAvLyAgIG5vbi1BU0NJSSBjaGFyYWN0ZXJzIGluIGBtb2RlbFRvcG9sb2d5MWAgYW5kIGB3ZWlnaHRTcGVjczFgLlxuICAgICAgICAgIGV4cGVjdChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby5tb2RlbFRvcG9sb2d5Qnl0ZXMpXG4gICAgICAgICAgICAgIC50b0VxdWFsKEpTT04uc3RyaW5naWZ5KG1vZGVsVG9wb2xvZ3kxKS5sZW5ndGgpO1xuICAgICAgICAgIGV4cGVjdChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby53ZWlnaHRTcGVjc0J5dGVzKS50b0VxdWFsKDApO1xuICAgICAgICAgIGV4cGVjdChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby53ZWlnaHREYXRhQnl0ZXMpLnRvRXF1YWwoMCk7XG5cbiAgICAgICAgICBleHBlY3QocmVxdWVzdEluaXRzLmxlbmd0aCkudG9FcXVhbCgxKTtcbiAgICAgICAgICBjb25zdCBpbml0ID0gcmVxdWVzdEluaXRzWzBdO1xuICAgICAgICAgIGV4cGVjdChpbml0Lm1ldGhvZCkudG9FcXVhbCgnUE9TVCcpO1xuICAgICAgICAgIGNvbnN0IGJvZHkgPSBpbml0LmJvZHkgYXMgRm9ybURhdGE7XG4gICAgICAgICAgY29uc3QganNvbkZpbGUgPSBib2R5LmdldCgnbW9kZWwuanNvbicpIGFzIEZpbGU7XG4gICAgICAgICAgY29uc3QganNvbkZpbGVSZWFkZXIgPSBuZXcgRmlsZVJlYWRlcigpO1xuICAgICAgICAgIGpzb25GaWxlUmVhZGVyLm9ubG9hZCA9IChldmVudDogRXZlbnQpID0+IHtcbiAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgICAgIGNvbnN0IG1vZGVsSlNPTiA9IEpTT04ucGFyc2UoKGV2ZW50LnRhcmdldCBhcyBhbnkpLnJlc3VsdCk7XG4gICAgICAgICAgICBleHBlY3QobW9kZWxKU09OLm1vZGVsVG9wb2xvZ3kpLnRvRXF1YWwobW9kZWxUb3BvbG9neTEpO1xuICAgICAgICAgICAgLy8gTm8gd2VpZ2h0cyBzaG91bGQgaGF2ZSBiZWVuIHNlbnQgdG8gdGhlIHNlcnZlci5cbiAgICAgICAgICAgIGV4cGVjdChib2R5LmdldCgnbW9kZWwud2VpZ2h0cy5iaW4nKSkudG9FcXVhbChudWxsKTtcbiAgICAgICAgICAgIGRvbmUoKTtcbiAgICAgICAgICB9O1xuICAgICAgICAgIGpzb25GaWxlUmVhZGVyLm9uZXJyb3IgPSBldmVudCA9PiB7XG4gICAgICAgICAgICBkb25lLmZhaWwoanNvbkZpbGVSZWFkZXIuZXJyb3IubWVzc2FnZSk7XG4gICAgICAgICAgfTtcbiAgICAgICAgICBqc29uRmlsZVJlYWRlci5yZWFkQXNUZXh0KGpzb25GaWxlKTtcbiAgICAgICAgfSlcbiAgICAgICAgLmNhdGNoKGVyciA9PiB7XG4gICAgICAgICAgZG9uZS5mYWlsKGVyci5zdGFjayk7XG4gICAgICAgIH0pO1xuICB9KTtcblxuICBpdCgnU2F2ZSB0b3BvbG9neSBhbmQgd2VpZ2h0cywgUFVUIG1ldGhvZCwgZXh0cmEgaGVhZGVycycsIChkb25lKSA9PiB7XG4gICAgY29uc3QgdGVzdFN0YXJ0RGF0ZSA9IG5ldyBEYXRlKCk7XG4gICAgY29uc3QgaGFuZGxlciA9IHRmLmlvLmh0dHAoJ21vZGVsLXVwbG9hZC10ZXN0Jywge1xuICAgICAgcmVxdWVzdEluaXQ6IHtcbiAgICAgICAgbWV0aG9kOiAnUFVUJyxcbiAgICAgICAgaGVhZGVyczpcbiAgICAgICAgICAgIHsnaGVhZGVyX2tleV8xJzogJ2hlYWRlcl92YWx1ZV8xJywgJ2hlYWRlcl9rZXlfMic6ICdoZWFkZXJfdmFsdWVfMid9XG4gICAgICB9XG4gICAgfSk7XG4gICAgaGFuZGxlci5zYXZlKGFydGlmYWN0czEpXG4gICAgICAgIC50aGVuKHNhdmVSZXN1bHQgPT4ge1xuICAgICAgICAgIGV4cGVjdChzYXZlUmVzdWx0Lm1vZGVsQXJ0aWZhY3RzSW5mby5kYXRlU2F2ZWQuZ2V0VGltZSgpKVxuICAgICAgICAgICAgICAudG9CZUdyZWF0ZXJUaGFuT3JFcXVhbCh0ZXN0U3RhcnREYXRlLmdldFRpbWUoKSk7XG4gICAgICAgICAgLy8gTm90ZTogVGhlIGZvbGxvd2luZyB0d28gYXNzZXJ0aW9ucyB3b3JrIG9ubHkgYmVjYXVzZSB0aGVyZSBpcyBub1xuICAgICAgICAgIC8vICAgbm9uLUFTQ0lJIGNoYXJhY3RlcnMgaW4gYG1vZGVsVG9wb2xvZ3kxYCBhbmQgYHdlaWdodFNwZWNzMWAuXG4gICAgICAgICAgZXhwZWN0KHNhdmVSZXN1bHQubW9kZWxBcnRpZmFjdHNJbmZvLm1vZGVsVG9wb2xvZ3lCeXRlcylcbiAgICAgICAgICAgICAgLnRvRXF1YWwoSlNPTi5zdHJpbmdpZnkobW9kZWxUb3BvbG9neTEpLmxlbmd0aCk7XG4gICAgICAgICAgZXhwZWN0KHNhdmVSZXN1bHQubW9kZWxBcnRpZmFjdHNJbmZvLndlaWdodFNwZWNzQnl0ZXMpXG4gICAgICAgICAgICAgIC50b0VxdWFsKEpTT04uc3RyaW5naWZ5KHdlaWdodFNwZWNzMSkubGVuZ3RoKTtcbiAgICAgICAgICBleHBlY3Qoc2F2ZVJlc3VsdC5tb2RlbEFydGlmYWN0c0luZm8ud2VpZ2h0RGF0YUJ5dGVzKVxuICAgICAgICAgICAgICAudG9FcXVhbCh3ZWlnaHREYXRhMS5ieXRlTGVuZ3RoKTtcblxuICAgICAgICAgIGV4cGVjdChyZXF1ZXN0SW5pdHMubGVuZ3RoKS50b0VxdWFsKDEpO1xuICAgICAgICAgIGNvbnN0IGluaXQgPSByZXF1ZXN0SW5pdHNbMF07XG4gICAgICAgICAgZXhwZWN0KGluaXQubWV0aG9kKS50b0VxdWFsKCdQVVQnKTtcblxuICAgICAgICAgIC8vIENoZWNrIGhlYWRlcnMuXG4gICAgICAgICAgZXhwZWN0KGluaXQuaGVhZGVycykudG9FcXVhbCh7XG4gICAgICAgICAgICAnaGVhZGVyX2tleV8xJzogJ2hlYWRlcl92YWx1ZV8xJyxcbiAgICAgICAgICAgICdoZWFkZXJfa2V5XzInOiAnaGVhZGVyX3ZhbHVlXzInXG4gICAgICAgICAgfSk7XG5cbiAgICAgICAgICBjb25zdCBib2R5ID0gaW5pdC5ib2R5IGFzIEZvcm1EYXRhO1xuICAgICAgICAgIGNvbnN0IGpzb25GaWxlID0gYm9keS5nZXQoJ21vZGVsLmpzb24nKSBhcyBGaWxlO1xuICAgICAgICAgIGNvbnN0IGpzb25GaWxlUmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTtcbiAgICAgICAgICBqc29uRmlsZVJlYWRlci5vbmxvYWQgPSAoZXZlbnQ6IEV2ZW50KSA9PiB7XG4gICAgICAgICAgICBjb25zdCBtb2RlbEpTT04gPVxuICAgICAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgICAgICAgICBKU09OLnBhcnNlKChldmVudC50YXJnZXQgYXMgYW55KS5yZXN1bHQpIGFzIHRmLmlvLk1vZGVsSlNPTjtcbiAgICAgICAgICAgIGV4cGVjdChtb2RlbEpTT04uZm9ybWF0KS50b0VxdWFsKCdsYXllcnMtbW9kZWwnKTtcbiAgICAgICAgICAgIGV4cGVjdChtb2RlbEpTT04uZ2VuZXJhdGVkQnkpLnRvRXF1YWwoJ1RlbnNvckZsb3cuanMgdjAuMC4wJyk7XG4gICAgICAgICAgICBleHBlY3QobW9kZWxKU09OLmNvbnZlcnRlZEJ5KS50b0VxdWFsKG51bGwpO1xuICAgICAgICAgICAgZXhwZWN0KG1vZGVsSlNPTi5tb2RlbFRvcG9sb2d5KS50b0VxdWFsKG1vZGVsVG9wb2xvZ3kxKTtcbiAgICAgICAgICAgIGV4cGVjdChtb2RlbEpTT04ubW9kZWxJbml0aWFsaXplcikudG9FcXVhbCh7fSk7XG4gICAgICAgICAgICBleHBlY3QobW9kZWxKU09OLndlaWdodHNNYW5pZmVzdC5sZW5ndGgpLnRvRXF1YWwoMSk7XG4gICAgICAgICAgICBleHBlY3QobW9kZWxKU09OLndlaWdodHNNYW5pZmVzdFswXS53ZWlnaHRzKS50b0VxdWFsKHdlaWdodFNwZWNzMSk7XG4gICAgICAgICAgICBleHBlY3QobW9kZWxKU09OLnRyYWluaW5nQ29uZmlnKS50b0VxdWFsKHRyYWluaW5nQ29uZmlnMSk7XG5cbiAgICAgICAgICAgIGNvbnN0IHdlaWdodHNGaWxlID0gYm9keS5nZXQoJ21vZGVsLndlaWdodHMuYmluJykgYXMgRmlsZTtcbiAgICAgICAgICAgIGNvbnN0IHdlaWdodHNGaWxlUmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTtcbiAgICAgICAgICAgIHdlaWdodHNGaWxlUmVhZGVyLm9ubG9hZCA9IChldmVudDogRXZlbnQpID0+IHtcbiAgICAgICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICAgICAgICBjb25zdCB3ZWlnaHREYXRhID0gKGV2ZW50LnRhcmdldCBhcyBhbnkpLnJlc3VsdCBhcyBBcnJheUJ1ZmZlcjtcbiAgICAgICAgICAgICAgZXhwZWN0KG5ldyBVaW50OEFycmF5KHdlaWdodERhdGEpKVxuICAgICAgICAgICAgICAgICAgLnRvRXF1YWwobmV3IFVpbnQ4QXJyYXkod2VpZ2h0RGF0YTEpKTtcbiAgICAgICAgICAgICAgZG9uZSgpO1xuICAgICAgICAgICAgfTtcbiAgICAgICAgICAgIHdlaWdodHNGaWxlUmVhZGVyLm9uZXJyb3IgPSBldmVudCA9PiB7XG4gICAgICAgICAgICAgIGRvbmUuZmFpbCh3ZWlnaHRzRmlsZVJlYWRlci5lcnJvci5tZXNzYWdlKTtcbiAgICAgICAgICAgIH07XG4gICAgICAgICAgICB3ZWlnaHRzRmlsZVJlYWRlci5yZWFkQXNBcnJheUJ1ZmZlcih3ZWlnaHRzRmlsZSk7XG4gICAgICAgICAgfTtcbiAgICAgICAgICBqc29uRmlsZVJlYWRlci5vbmVycm9yID0gZXZlbnQgPT4ge1xuICAgICAgICAgICAgZG9uZS5mYWlsKGpzb25GaWxlUmVhZGVyLmVycm9yLm1lc3NhZ2UpO1xuICAgICAgICAgIH07XG4gICAgICAgICAganNvbkZpbGVSZWFkZXIucmVhZEFzVGV4dChqc29uRmlsZSk7XG4gICAgICAgIH0pXG4gICAgICAgIC5jYXRjaChlcnIgPT4ge1xuICAgICAgICAgIGRvbmUuZmFpbChlcnIuc3RhY2spO1xuICAgICAgICB9KTtcbiAgfSk7XG5cbiAgaXQoJzQwNCByZXNwb25zZSBjYXVzZXMgRXJyb3InLCAoZG9uZSkgPT4ge1xuICAgIGNvbnN0IGhhbmRsZXIgPSB0Zi5pby5nZXRTYXZlSGFuZGxlcnMoJ2h0dHA6Ly9pbnZhbGlkL3BhdGgnKVswXTtcbiAgICBoYW5kbGVyLnNhdmUoYXJ0aWZhY3RzMSlcbiAgICAgICAgLnRoZW4oc2F2ZVJlc3VsdCA9PiB7XG4gICAgICAgICAgZG9uZS5mYWlsKFxuICAgICAgICAgICAgICAnQ2FsbGluZyBodHRwIGF0IGludmFsaWQgVVJMIHN1Y2NlZWRlZCAnICtcbiAgICAgICAgICAgICAgJ3VuZXhwZWN0ZWRseScpO1xuICAgICAgICB9KVxuICAgICAgICAuY2F0Y2goZXJyID0+IHtcbiAgICAgICAgICBkb25lKCk7XG4gICAgICAgIH0pO1xuICB9KTtcblxuICBpdCgnZ2V0TG9hZEhhbmRsZXJzIHdpdGggb25lIFVSTCBzdHJpbmcnLCAoKSA9PiB7XG4gICAgY29uc3QgaGFuZGxlcnMgPSB0Zi5pby5nZXRMb2FkSGFuZGxlcnMoJ2h0dHA6Ly9mb28vbW9kZWwuanNvbicpO1xuICAgIGV4cGVjdChoYW5kbGVycy5sZW5ndGgpLnRvRXF1YWwoMSk7XG4gICAgZXhwZWN0KGhhbmRsZXJzWzBdIGluc3RhbmNlb2YgSFRUUFJlcXVlc3QpLnRvRXF1YWwodHJ1ZSk7XG4gIH0pO1xuXG4gIGl0KCdFeGlzdGluZyBib2R5IGxlYWRzIHRvIEVycm9yJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB0Zi5pby5odHRwKCdtb2RlbC11cGxvYWQtdGVzdCcsIHtcbiAgICAgIHJlcXVlc3RJbml0OiB7Ym9keTogJ2V4aXN0aW5nIGJvZHknfVxuICAgIH0pKS50b1Rocm93RXJyb3IoL3JlcXVlc3RJbml0IGlzIGV4cGVjdGVkIHRvIGhhdmUgbm8gcHJlLWV4aXN0aW5nIGJvZHkvKTtcbiAgfSk7XG5cbiAgaXQoJ0VtcHR5LCBudWxsIG9yIHVuZGVmaW5lZCBVUkwgcGF0aHMgbGVhZCB0byBFcnJvcicsICgpID0+IHtcbiAgICBleHBlY3QoKCkgPT4gdGYuaW8uaHR0cChudWxsKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigvbXVzdCBub3QgYmUgbnVsbCwgdW5kZWZpbmVkIG9yIGVtcHR5Lyk7XG4gICAgZXhwZWN0KCgpID0+IHRmLmlvLmh0dHAodW5kZWZpbmVkKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigvbXVzdCBub3QgYmUgbnVsbCwgdW5kZWZpbmVkIG9yIGVtcHR5Lyk7XG4gICAgZXhwZWN0KCgpID0+IHRmLmlvLmh0dHAoJycpKVxuICAgICAgICAudG9UaHJvd0Vycm9yKC9tdXN0IG5vdCBiZSBudWxsLCB1bmRlZmluZWQgb3IgZW1wdHkvKTtcbiAgfSk7XG5cbiAgaXQoJ3JvdXRlcicsICgpID0+IHtcbiAgICBleHBlY3QoaHR0cFJvdXRlcignaHR0cDovL2Jhci9mb28nKSBpbnN0YW5jZW9mIEhUVFBSZXF1ZXN0KS50b0VxdWFsKHRydWUpO1xuICAgIGV4cGVjdChodHRwUm91dGVyKCdodHRwczovL2xvY2FsaG9zdDo1MDAwL3VwbG9hZCcpIGluc3RhbmNlb2YgSFRUUFJlcXVlc3QpXG4gICAgICAgIC50b0VxdWFsKHRydWUpO1xuICAgIGV4cGVjdChodHRwUm91dGVyKCdsb2NhbGhvc3Q6Ly9mb28nKSkudG9CZU51bGwoKTtcbiAgICBleHBlY3QoaHR0cFJvdXRlcignZm9vOjUwMDAvYmFyJykpLnRvQmVOdWxsKCk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdwYXJzZVVybCcsIEJST1dTRVJfRU5WUywgKCkgPT4ge1xuICBpdCgnc2hvdWxkIHBhcnNlIHVybCB3aXRoIG5vIHN1ZmZpeCcsICgpID0+IHtcbiAgICBjb25zdCB1cmwgPSAnaHR0cDovL2dvb2dsZS5jb20vZmlsZSc7XG4gICAgY29uc3QgW3ByZWZpeCwgc3VmZml4XSA9IHBhcnNlVXJsKHVybCk7XG4gICAgZXhwZWN0KHByZWZpeCkudG9FcXVhbCgnaHR0cDovL2dvb2dsZS5jb20vJyk7XG4gICAgZXhwZWN0KHN1ZmZpeCkudG9FcXVhbCgnJyk7XG4gIH0pO1xuICBpdCgnc2hvdWxkIHBhcnNlIHVybCB3aXRoIHN1ZmZpeCcsICgpID0+IHtcbiAgICBjb25zdCB1cmwgPSAnaHR0cDovL2dvb2dsZS5jb20vZmlsZT9wYXJhbT0xJztcbiAgICBjb25zdCBbcHJlZml4LCBzdWZmaXhdID0gcGFyc2VVcmwodXJsKTtcbiAgICBleHBlY3QocHJlZml4KS50b0VxdWFsKCdodHRwOi8vZ29vZ2xlLmNvbS8nKTtcbiAgICBleHBlY3Qoc3VmZml4KS50b0VxdWFsKCc/cGFyYW09MScpO1xuICB9KTtcbiAgaXQoJ3Nob3VsZCBwYXJzZSB1cmwgd2l0aCBtdWx0aXBsZSBzZXJhY2ggcGFyYW1zJywgKCkgPT4ge1xuICAgIGNvbnN0IHVybCA9ICdodHRwOi8vZ29vZ2xlLmNvbS9hP3g9MS9maWxlP3BhcmFtPTEnO1xuICAgIGNvbnN0IFtwcmVmaXgsIHN1ZmZpeF0gPSBwYXJzZVVybCh1cmwpO1xuICAgIGV4cGVjdChwcmVmaXgpLnRvRXF1YWwoJ2h0dHA6Ly9nb29nbGUuY29tL2E/eD0xLycpO1xuICAgIGV4cGVjdChzdWZmaXgpLnRvRXF1YWwoJz9wYXJhbT0xJyk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCdodHRwLWxvYWQnLCBCUk9XU0VSX0VOVlMsICgpID0+IHtcbiAgZGVzY3JpYmUoJ0pTT04gbW9kZWwnLCAoKSA9PiB7XG4gICAgbGV0IHJlcXVlc3RJbml0czoge1trZXk6IHN0cmluZ106IHtoZWFkZXJzOiB7W2tleTogc3RyaW5nXTogc3RyaW5nfX19O1xuXG4gICAgYmVmb3JlRWFjaCgoKSA9PiB7XG4gICAgICByZXF1ZXN0SW5pdHMgPSB7fTtcbiAgICB9KTtcblxuICAgIGl0KCcxIGdyb3VwLCAyIHdlaWdodHMsIDEgcGF0aCcsIGFzeW5jICgpID0+IHtcbiAgICAgIGNvbnN0IHdlaWdodE1hbmlmZXN0MTogdGYuaW8uV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgICAgcGF0aHM6IFsnd2VpZ2h0ZmlsZTAnXSxcbiAgICAgICAgd2VpZ2h0czogW1xuICAgICAgICAgIHtcbiAgICAgICAgICAgIG5hbWU6ICdkZW5zZS9rZXJuZWwnLFxuICAgICAgICAgICAgc2hhcGU6IFszLCAxXSxcbiAgICAgICAgICAgIGR0eXBlOiAnZmxvYXQzMicsXG4gICAgICAgICAgfSxcbiAgICAgICAgICB7XG4gICAgICAgICAgICBuYW1lOiAnZGVuc2UvYmlhcycsXG4gICAgICAgICAgICBzaGFwZTogWzJdLFxuICAgICAgICAgICAgZHR5cGU6ICdmbG9hdDMyJyxcbiAgICAgICAgICB9XG4gICAgICAgIF1cbiAgICAgIH1dO1xuICAgICAgY29uc3QgZmxvYXREYXRhID0gbmV3IEZsb2F0MzJBcnJheShbMSwgMywgMywgNywgNF0pO1xuICAgICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoXG4gICAgICAgICAge1xuICAgICAgICAgICAgJy4vbW9kZWwuanNvbic6IHtcbiAgICAgICAgICAgICAgZGF0YTogSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgIG1vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxLFxuICAgICAgICAgICAgICAgIHdlaWdodHNNYW5pZmVzdDogd2VpZ2h0TWFuaWZlc3QxLFxuICAgICAgICAgICAgICAgIGZvcm1hdDogJ3RmanMtZ3JhcGgtbW9kZWwnLFxuICAgICAgICAgICAgICAgIGdlbmVyYXRlZEJ5OiAnMS4xNScsXG4gICAgICAgICAgICAgICAgY29udmVydGVkQnk6ICcxLjMuMScsXG4gICAgICAgICAgICAgICAgc2lnbmF0dXJlOiBudWxsLFxuICAgICAgICAgICAgICAgIHVzZXJEZWZpbmVkTWV0YWRhdGE6IHt9LFxuICAgICAgICAgICAgICAgIG1vZGVsSW5pdGlhbGl6ZXI6IHt9XG4gICAgICAgICAgICAgIH0pLFxuICAgICAgICAgICAgICBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nXG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgJy4vd2VpZ2h0ZmlsZTAnOlxuICAgICAgICAgICAgICAgIHtkYXRhOiBmbG9hdERhdGEsIGNvbnRlbnRUeXBlOiAnYXBwbGljYXRpb24vb2N0ZXQtc3RyZWFtJ30sXG4gICAgICAgICAgfSxcbiAgICAgICAgICByZXF1ZXN0SW5pdHMpO1xuXG4gICAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uaHR0cCgnLi9tb2RlbC5qc29uJyk7XG4gICAgICBjb25zdCBtb2RlbEFydGlmYWN0cyA9IGF3YWl0IGhhbmRsZXIubG9hZCgpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kpLnRvRXF1YWwobW9kZWxUb3BvbG9neTEpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzKS50b0VxdWFsKHdlaWdodE1hbmlmZXN0MVswXS53ZWlnaHRzKTtcbiAgICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5mb3JtYXQpLnRvRXF1YWwoJ3RmanMtZ3JhcGgtbW9kZWwnKTtcbiAgICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5nZW5lcmF0ZWRCeSkudG9FcXVhbCgnMS4xNScpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLmNvbnZlcnRlZEJ5KS50b0VxdWFsKCcxLjMuMScpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLnVzZXJEZWZpbmVkTWV0YWRhdGEpLnRvRXF1YWwoe30pO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLm1vZGVsSW5pdGlhbGl6ZXIpLnRvRXF1YWwoe30pO1xuXG4gICAgICBleHBlY3QobmV3IEZsb2F0MzJBcnJheShtb2RlbEFydGlmYWN0cy53ZWlnaHREYXRhKSkudG9FcXVhbChmbG9hdERhdGEpO1xuICAgICAgZXhwZWN0KE9iamVjdC5rZXlzKHJlcXVlc3RJbml0cykubGVuZ3RoKS50b0VxdWFsKDIpO1xuICAgICAgLy8gQXNzZXJ0IHRoYXQgZmV0Y2ggaXMgaW52b2tlZCB3aXRoIGB3aW5kb3dgIGFzIHRoZSBjb250ZXh0LlxuICAgICAgZXhwZWN0KGZldGNoU3B5LmNhbGxzLm1vc3RSZWNlbnQoKS5vYmplY3QpLnRvRXF1YWwod2luZG93KTtcbiAgICB9KTtcblxuICAgIGl0KCcxIGdyb3VwLCAyIHdlaWdodHMsIDEgcGF0aCwgd2l0aCByZXF1ZXN0SW5pdCcsIGFzeW5jICgpID0+IHtcbiAgICAgIGNvbnN0IHdlaWdodE1hbmlmZXN0MTogdGYuaW8uV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgICAgcGF0aHM6IFsnd2VpZ2h0ZmlsZTAnXSxcbiAgICAgICAgd2VpZ2h0czogW1xuICAgICAgICAgIHtcbiAgICAgICAgICAgIG5hbWU6ICdkZW5zZS9rZXJuZWwnLFxuICAgICAgICAgICAgc2hhcGU6IFszLCAxXSxcbiAgICAgICAgICAgIGR0eXBlOiAnZmxvYXQzMicsXG4gICAgICAgICAgfSxcbiAgICAgICAgICB7XG4gICAgICAgICAgICBuYW1lOiAnZGVuc2UvYmlhcycsXG4gICAgICAgICAgICBzaGFwZTogWzJdLFxuICAgICAgICAgICAgZHR5cGU6ICdmbG9hdDMyJyxcbiAgICAgICAgICB9XG4gICAgICAgIF1cbiAgICAgIH1dO1xuICAgICAgY29uc3QgZmxvYXREYXRhID0gbmV3IEZsb2F0MzJBcnJheShbMSwgMywgMywgNywgNF0pO1xuICAgICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoXG4gICAgICAgICAge1xuICAgICAgICAgICAgJy4vbW9kZWwuanNvbic6IHtcbiAgICAgICAgICAgICAgZGF0YTogSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgIG1vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxLFxuICAgICAgICAgICAgICAgIHdlaWdodHNNYW5pZmVzdDogd2VpZ2h0TWFuaWZlc3QxXG4gICAgICAgICAgICAgIH0pLFxuICAgICAgICAgICAgICBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nXG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgJy4vd2VpZ2h0ZmlsZTAnOlxuICAgICAgICAgICAgICAgIHtkYXRhOiBmbG9hdERhdGEsIGNvbnRlbnRUeXBlOiAnYXBwbGljYXRpb24vb2N0ZXQtc3RyZWFtJ30sXG4gICAgICAgICAgfSxcbiAgICAgICAgICByZXF1ZXN0SW5pdHMpO1xuXG4gICAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uaHR0cChcbiAgICAgICAgICAnLi9tb2RlbC5qc29uJyxcbiAgICAgICAgICB7cmVxdWVzdEluaXQ6IHtoZWFkZXJzOiB7J2hlYWRlcl9rZXlfMSc6ICdoZWFkZXJfdmFsdWVfMSd9fX0pO1xuICAgICAgY29uc3QgbW9kZWxBcnRpZmFjdHMgPSBhd2FpdCBoYW5kbGVyLmxvYWQoKTtcbiAgICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5KS50b0VxdWFsKG1vZGVsVG9wb2xvZ3kxKTtcbiAgICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy53ZWlnaHRTcGVjcykudG9FcXVhbCh3ZWlnaHRNYW5pZmVzdDFbMF0ud2VpZ2h0cyk7XG4gICAgICBleHBlY3QobmV3IEZsb2F0MzJBcnJheShtb2RlbEFydGlmYWN0cy53ZWlnaHREYXRhKSkudG9FcXVhbChmbG9hdERhdGEpO1xuICAgICAgZXhwZWN0KE9iamVjdC5rZXlzKHJlcXVlc3RJbml0cykubGVuZ3RoKS50b0VxdWFsKDIpO1xuICAgICAgZXhwZWN0KE9iamVjdC5rZXlzKHJlcXVlc3RJbml0cykubGVuZ3RoKS50b0VxdWFsKDIpO1xuICAgICAgZXhwZWN0KHJlcXVlc3RJbml0c1snLi9tb2RlbC5qc29uJ10uaGVhZGVyc1snaGVhZGVyX2tleV8xJ10pXG4gICAgICAgICAgLnRvRXF1YWwoJ2hlYWRlcl92YWx1ZV8xJyk7XG4gICAgICBleHBlY3QocmVxdWVzdEluaXRzWycuL3dlaWdodGZpbGUwJ10uaGVhZGVyc1snaGVhZGVyX2tleV8xJ10pXG4gICAgICAgICAgLnRvRXF1YWwoJ2hlYWRlcl92YWx1ZV8xJyk7XG5cbiAgICAgIGV4cGVjdChmZXRjaFNweS5jYWxscy5tb3N0UmVjZW50KCkub2JqZWN0KS50b0VxdWFsKHdpbmRvdyk7XG4gICAgfSk7XG5cbiAgICBpdCgnMSBncm91cCwgMiB3ZWlnaHQsIDIgcGF0aHMnLCBhc3luYyAoKSA9PiB7XG4gICAgICBjb25zdCB3ZWlnaHRNYW5pZmVzdDE6IHRmLmlvLldlaWdodHNNYW5pZmVzdENvbmZpZyA9IFt7XG4gICAgICAgIHBhdGhzOiBbJ3dlaWdodGZpbGUwJywgJ3dlaWdodGZpbGUxJ10sXG4gICAgICAgIHdlaWdodHM6IFtcbiAgICAgICAgICB7XG4gICAgICAgICAgICBuYW1lOiAnZGVuc2Uva2VybmVsJyxcbiAgICAgICAgICAgIHNoYXBlOiBbMywgMV0sXG4gICAgICAgICAgICBkdHlwZTogJ2Zsb2F0MzInLFxuICAgICAgICAgIH0sXG4gICAgICAgICAge1xuICAgICAgICAgICAgbmFtZTogJ2RlbnNlL2JpYXMnLFxuICAgICAgICAgICAgc2hhcGU6IFsyXSxcbiAgICAgICAgICAgIGR0eXBlOiAnZmxvYXQzMicsXG4gICAgICAgICAgfVxuICAgICAgICBdXG4gICAgICB9XTtcbiAgICAgIGNvbnN0IGZsb2F0RGF0YTEgPSBuZXcgRmxvYXQzMkFycmF5KFsxLCAzLCAzXSk7XG4gICAgICBjb25zdCBmbG9hdERhdGEyID0gbmV3IEZsb2F0MzJBcnJheShbNywgNF0pO1xuICAgICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoXG4gICAgICAgICAge1xuICAgICAgICAgICAgJy4vbW9kZWwuanNvbic6IHtcbiAgICAgICAgICAgICAgZGF0YTogSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgIG1vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxLFxuICAgICAgICAgICAgICAgIHdlaWdodHNNYW5pZmVzdDogd2VpZ2h0TWFuaWZlc3QxXG4gICAgICAgICAgICAgIH0pLFxuICAgICAgICAgICAgICBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nXG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgJy4vd2VpZ2h0ZmlsZTAnOlxuICAgICAgICAgICAgICAgIHtkYXRhOiBmbG9hdERhdGExLCBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL29jdGV0LXN0cmVhbSd9LFxuICAgICAgICAgICAgJy4vd2VpZ2h0ZmlsZTEnOlxuICAgICAgICAgICAgICAgIHtkYXRhOiBmbG9hdERhdGEyLCBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL29jdGV0LXN0cmVhbSd9XG4gICAgICAgICAgfSxcbiAgICAgICAgICByZXF1ZXN0SW5pdHMpO1xuXG4gICAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uaHR0cCgnLi9tb2RlbC5qc29uJyk7XG4gICAgICBjb25zdCBtb2RlbEFydGlmYWN0cyA9IGF3YWl0IGhhbmRsZXIubG9hZCgpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kpLnRvRXF1YWwobW9kZWxUb3BvbG9neTEpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzKS50b0VxdWFsKHdlaWdodE1hbmlmZXN0MVswXS53ZWlnaHRzKTtcbiAgICAgIGV4cGVjdChuZXcgRmxvYXQzMkFycmF5KG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEpKVxuICAgICAgICAgIC50b0VxdWFsKG5ldyBGbG9hdDMyQXJyYXkoWzEsIDMsIDMsIDcsIDRdKSk7XG4gICAgfSk7XG5cbiAgICBpdCgnMiBncm91cHMsIDIgd2VpZ2h0LCAyIHBhdGhzJywgYXN5bmMgKCkgPT4ge1xuICAgICAgY29uc3Qgd2VpZ2h0c01hbmlmZXN0OiB0Zi5pby5XZWlnaHRzTWFuaWZlc3RDb25maWcgPSBbXG4gICAgICAgIHtcbiAgICAgICAgICBwYXRoczogWyd3ZWlnaHRmaWxlMCddLFxuICAgICAgICAgIHdlaWdodHM6IFt7XG4gICAgICAgICAgICBuYW1lOiAnZGVuc2Uva2VybmVsJyxcbiAgICAgICAgICAgIHNoYXBlOiBbMywgMV0sXG4gICAgICAgICAgICBkdHlwZTogJ2Zsb2F0MzInLFxuICAgICAgICAgIH1dXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBwYXRoczogWyd3ZWlnaHRmaWxlMSddLFxuICAgICAgICAgIHdlaWdodHM6IFt7XG4gICAgICAgICAgICBuYW1lOiAnZGVuc2UvYmlhcycsXG4gICAgICAgICAgICBzaGFwZTogWzJdLFxuICAgICAgICAgICAgZHR5cGU6ICdmbG9hdDMyJyxcbiAgICAgICAgICB9XSxcbiAgICAgICAgfVxuICAgICAgXTtcbiAgICAgIGNvbnN0IGZsb2F0RGF0YTEgPSBuZXcgRmxvYXQzMkFycmF5KFsxLCAzLCAzXSk7XG4gICAgICBjb25zdCBmbG9hdERhdGEyID0gbmV3IEZsb2F0MzJBcnJheShbNywgNF0pO1xuICAgICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoXG4gICAgICAgICAge1xuICAgICAgICAgICAgJy4vbW9kZWwuanNvbic6IHtcbiAgICAgICAgICAgICAgZGF0YTogSlNPTi5zdHJpbmdpZnkoXG4gICAgICAgICAgICAgICAgICB7bW9kZWxUb3BvbG9neTogbW9kZWxUb3BvbG9neTEsIHdlaWdodHNNYW5pZmVzdH0pLFxuICAgICAgICAgICAgICBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nXG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgJy4vd2VpZ2h0ZmlsZTAnOlxuICAgICAgICAgICAgICAgIHtkYXRhOiBmbG9hdERhdGExLCBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL29jdGV0LXN0cmVhbSd9LFxuICAgICAgICAgICAgJy4vd2VpZ2h0ZmlsZTEnOlxuICAgICAgICAgICAgICAgIHtkYXRhOiBmbG9hdERhdGEyLCBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL29jdGV0LXN0cmVhbSd9XG4gICAgICAgICAgfSxcbiAgICAgICAgICByZXF1ZXN0SW5pdHMpO1xuXG4gICAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uaHR0cCgnLi9tb2RlbC5qc29uJyk7XG4gICAgICBjb25zdCBtb2RlbEFydGlmYWN0cyA9IGF3YWl0IGhhbmRsZXIubG9hZCgpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kpLnRvRXF1YWwobW9kZWxUb3BvbG9neTEpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzKVxuICAgICAgICAgIC50b0VxdWFsKFxuICAgICAgICAgICAgICB3ZWlnaHRzTWFuaWZlc3RbMF0ud2VpZ2h0cy5jb25jYXQod2VpZ2h0c01hbmlmZXN0WzFdLndlaWdodHMpKTtcbiAgICAgIGV4cGVjdChuZXcgRmxvYXQzMkFycmF5KG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEpKVxuICAgICAgICAgIC50b0VxdWFsKG5ldyBGbG9hdDMyQXJyYXkoWzEsIDMsIDMsIDcsIDRdKSk7XG4gICAgfSk7XG5cbiAgICBpdCgnMiBncm91cHMsIDIgd2VpZ2h0LCAyIHBhdGhzLCBJbnQzMiBhbmQgVWludDggRGF0YScsIGFzeW5jICgpID0+IHtcbiAgICAgIGNvbnN0IHdlaWdodHNNYW5pZmVzdDogdGYuaW8uV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW1xuICAgICAgICB7XG4gICAgICAgICAgcGF0aHM6IFsnd2VpZ2h0ZmlsZTAnXSxcbiAgICAgICAgICB3ZWlnaHRzOiBbe1xuICAgICAgICAgICAgbmFtZTogJ2Zvb1dlaWdodCcsXG4gICAgICAgICAgICBzaGFwZTogWzMsIDFdLFxuICAgICAgICAgICAgZHR5cGU6ICdpbnQzMicsXG4gICAgICAgICAgfV1cbiAgICAgICAgfSxcbiAgICAgICAge1xuICAgICAgICAgIHBhdGhzOiBbJ3dlaWdodGZpbGUxJ10sXG4gICAgICAgICAgd2VpZ2h0czogW3tcbiAgICAgICAgICAgIG5hbWU6ICdiYXJXZWlnaHQnLFxuICAgICAgICAgICAgc2hhcGU6IFsyXSxcbiAgICAgICAgICAgIGR0eXBlOiAnYm9vbCcsXG4gICAgICAgICAgfV0sXG4gICAgICAgIH1cbiAgICAgIF07XG4gICAgICBjb25zdCBmbG9hdERhdGExID0gbmV3IEludDMyQXJyYXkoWzEsIDMsIDNdKTtcbiAgICAgIGNvbnN0IGZsb2F0RGF0YTIgPSBuZXcgVWludDhBcnJheShbNywgNF0pO1xuICAgICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoXG4gICAgICAgICAge1xuICAgICAgICAgICAgJ3BhdGgxL21vZGVsLmpzb24nOiB7XG4gICAgICAgICAgICAgIGRhdGE6IEpTT04uc3RyaW5naWZ5KFxuICAgICAgICAgICAgICAgICAge21vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxLCB3ZWlnaHRzTWFuaWZlc3R9KSxcbiAgICAgICAgICAgICAgY29udGVudFR5cGU6ICdhcHBsaWNhdGlvbi9qc29uJ1xuICAgICAgICAgICAgfSxcbiAgICAgICAgICAgICdwYXRoMS93ZWlnaHRmaWxlMCc6XG4gICAgICAgICAgICAgICAge2RhdGE6IGZsb2F0RGF0YTEsIGNvbnRlbnRUeXBlOiAnYXBwbGljYXRpb24vb2N0ZXQtc3RyZWFtJ30sXG4gICAgICAgICAgICAncGF0aDEvd2VpZ2h0ZmlsZTEnOlxuICAgICAgICAgICAgICAgIHtkYXRhOiBmbG9hdERhdGEyLCBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL29jdGV0LXN0cmVhbSd9XG4gICAgICAgICAgfSxcbiAgICAgICAgICByZXF1ZXN0SW5pdHMpO1xuXG4gICAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uaHR0cCgncGF0aDEvbW9kZWwuanNvbicpO1xuICAgICAgY29uc3QgbW9kZWxBcnRpZmFjdHMgPSBhd2FpdCBoYW5kbGVyLmxvYWQoKTtcbiAgICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5KS50b0VxdWFsKG1vZGVsVG9wb2xvZ3kxKTtcbiAgICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy53ZWlnaHRTcGVjcylcbiAgICAgICAgICAudG9FcXVhbChcbiAgICAgICAgICAgICAgd2VpZ2h0c01hbmlmZXN0WzBdLndlaWdodHMuY29uY2F0KHdlaWdodHNNYW5pZmVzdFsxXS53ZWlnaHRzKSk7XG4gICAgICBleHBlY3QobmV3IEludDMyQXJyYXkobW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YS5zbGljZSgwLCAxMikpKVxuICAgICAgICAgIC50b0VxdWFsKG5ldyBJbnQzMkFycmF5KFsxLCAzLCAzXSkpO1xuICAgICAgZXhwZWN0KG5ldyBVaW50OEFycmF5KG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEuc2xpY2UoMTIsIDE0KSkpXG4gICAgICAgICAgLnRvRXF1YWwobmV3IFVpbnQ4QXJyYXkoWzcsIDRdKSk7XG4gICAgfSk7XG5cbiAgICBpdCgndG9wb2xvZ3kgb25seScsIGFzeW5jICgpID0+IHtcbiAgICAgIHNldHVwRmFrZVdlaWdodEZpbGVzKFxuICAgICAgICAgIHtcbiAgICAgICAgICAgICcuL21vZGVsLmpzb24nOiB7XG4gICAgICAgICAgICAgIGRhdGE6IEpTT04uc3RyaW5naWZ5KHttb2RlbFRvcG9sb2d5OiBtb2RlbFRvcG9sb2d5MX0pLFxuICAgICAgICAgICAgICBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nXG4gICAgICAgICAgICB9LFxuICAgICAgICAgIH0sXG4gICAgICAgICAgcmVxdWVzdEluaXRzKTtcblxuICAgICAgY29uc3QgaGFuZGxlciA9IHRmLmlvLmh0dHAoJy4vbW9kZWwuanNvbicpO1xuICAgICAgY29uc3QgbW9kZWxBcnRpZmFjdHMgPSBhd2FpdCBoYW5kbGVyLmxvYWQoKTtcbiAgICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5KS50b0VxdWFsKG1vZGVsVG9wb2xvZ3kxKTtcbiAgICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy53ZWlnaHRTcGVjcykudG9CZVVuZGVmaW5lZCgpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEpLnRvQmVVbmRlZmluZWQoKTtcbiAgICB9KTtcblxuICAgIGl0KCd3ZWlnaHRzIG9ubHknLCBhc3luYyAoKSA9PiB7XG4gICAgICBjb25zdCB3ZWlnaHRzTWFuaWZlc3Q6IHRmLmlvLldlaWdodHNNYW5pZmVzdENvbmZpZyA9IFtcbiAgICAgICAge1xuICAgICAgICAgIHBhdGhzOiBbJ3dlaWdodGZpbGUwJ10sXG4gICAgICAgICAgd2VpZ2h0czogW3tcbiAgICAgICAgICAgIG5hbWU6ICdmb29XZWlnaHQnLFxuICAgICAgICAgICAgc2hhcGU6IFszLCAxXSxcbiAgICAgICAgICAgIGR0eXBlOiAnaW50MzInLFxuICAgICAgICAgIH1dXG4gICAgICAgIH0sXG4gICAgICAgIHtcbiAgICAgICAgICBwYXRoczogWyd3ZWlnaHRmaWxlMSddLFxuICAgICAgICAgIHdlaWdodHM6IFt7XG4gICAgICAgICAgICBuYW1lOiAnYmFyV2VpZ2h0JyxcbiAgICAgICAgICAgIHNoYXBlOiBbMl0sXG4gICAgICAgICAgICBkdHlwZTogJ2Zsb2F0MzInLFxuICAgICAgICAgIH1dLFxuICAgICAgICB9XG4gICAgICBdO1xuICAgICAgY29uc3QgZmxvYXREYXRhMSA9IG5ldyBJbnQzMkFycmF5KFsxLCAzLCAzXSk7XG4gICAgICBjb25zdCBmbG9hdERhdGEyID0gbmV3IEZsb2F0MzJBcnJheShbLTcsIC00XSk7XG4gICAgICBzZXR1cEZha2VXZWlnaHRGaWxlcyhcbiAgICAgICAgICB7XG4gICAgICAgICAgICAncGF0aDEvbW9kZWwuanNvbic6IHtcbiAgICAgICAgICAgICAgZGF0YTogSlNPTi5zdHJpbmdpZnkoe3dlaWdodHNNYW5pZmVzdH0pLFxuICAgICAgICAgICAgICBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nXG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgJ3BhdGgxL3dlaWdodGZpbGUwJzpcbiAgICAgICAgICAgICAgICB7ZGF0YTogZmxvYXREYXRhMSwgY29udGVudFR5cGU6ICdhcHBsaWNhdGlvbi9vY3RldC1zdHJlYW0nfSxcbiAgICAgICAgICAgICdwYXRoMS93ZWlnaHRmaWxlMSc6XG4gICAgICAgICAgICAgICAge2RhdGE6IGZsb2F0RGF0YTIsIGNvbnRlbnRUeXBlOiAnYXBwbGljYXRpb24vb2N0ZXQtc3RyZWFtJ31cbiAgICAgICAgICB9LFxuICAgICAgICAgIHJlcXVlc3RJbml0cyk7XG5cbiAgICAgIGNvbnN0IGhhbmRsZXIgPSB0Zi5pby5odHRwKCdwYXRoMS9tb2RlbC5qc29uJyk7XG4gICAgICBjb25zdCBtb2RlbEFydGlmYWN0cyA9IGF3YWl0IGhhbmRsZXIubG9hZCgpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kpLnRvQmVVbmRlZmluZWQoKTtcbiAgICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy53ZWlnaHRTcGVjcylcbiAgICAgICAgICAudG9FcXVhbChcbiAgICAgICAgICAgICAgd2VpZ2h0c01hbmlmZXN0WzBdLndlaWdodHMuY29uY2F0KHdlaWdodHNNYW5pZmVzdFsxXS53ZWlnaHRzKSk7XG4gICAgICBleHBlY3QobmV3IEludDMyQXJyYXkobW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YS5zbGljZSgwLCAxMikpKVxuICAgICAgICAgIC50b0VxdWFsKG5ldyBJbnQzMkFycmF5KFsxLCAzLCAzXSkpO1xuICAgICAgZXhwZWN0KG5ldyBGbG9hdDMyQXJyYXkobW9kZWxBcnRpZmFjdHMud2VpZ2h0RGF0YS5zbGljZSgxMiwgMjApKSlcbiAgICAgICAgICAudG9FcXVhbChuZXcgRmxvYXQzMkFycmF5KFstNywgLTRdKSk7XG4gICAgfSk7XG5cbiAgICBpdCgnTWlzc2luZyBtb2RlbFRvcG9sb2d5IGFuZCB3ZWlnaHRzTWFuaWZlc3QgbGVhZHMgdG8gZXJyb3InLCBhc3luYyAoKSA9PiB7XG4gICAgICBzZXR1cEZha2VXZWlnaHRGaWxlcyhcbiAgICAgICAgICB7XG4gICAgICAgICAgICAncGF0aDEvbW9kZWwuanNvbic6XG4gICAgICAgICAgICAgICAge2RhdGE6IEpTT04uc3RyaW5naWZ5KHt9KSwgY29udGVudFR5cGU6ICdhcHBsaWNhdGlvbi9qc29uJ31cbiAgICAgICAgICB9LFxuICAgICAgICAgIHJlcXVlc3RJbml0cyk7XG4gICAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uaHR0cCgncGF0aDEvbW9kZWwuanNvbicpO1xuICAgICAgaGFuZGxlci5sb2FkKClcbiAgICAgICAgICAudGhlbihtb2RlbFRvcG9sb2d5MSA9PiB7XG4gICAgICAgICAgICBmYWlsKFxuICAgICAgICAgICAgICAgICdMb2FkaW5nIGZyb20gbWlzc2luZyBtb2RlbFRvcG9sb2d5IGFuZCB3ZWlnaHRzTWFuaWZlc3QgJyArXG4gICAgICAgICAgICAgICAgJ3N1Y2NlZWRlZCB1bmV4cGVjdGVkbHkuJyk7XG4gICAgICAgICAgfSlcbiAgICAgICAgICAuY2F0Y2goZXJyID0+IHtcbiAgICAgICAgICAgIGV4cGVjdChlcnIubWVzc2FnZSlcbiAgICAgICAgICAgICAgICAudG9NYXRjaCgvY29udGFpbnMgbmVpdGhlciBtb2RlbCB0b3BvbG9neSBvciBtYW5pZmVzdC8pO1xuICAgICAgICAgIH0pO1xuICAgIH0pO1xuXG4gICAgaXQoJ3dpdGggZmV0Y2ggcmVqZWN0aW9uIGxlYWRzIHRvIGVycm9yJywgYXN5bmMgKCkgPT4ge1xuICAgICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoXG4gICAgICAgICAge1xuICAgICAgICAgICAgJ3BhdGgxL21vZGVsLmpzb24nOlxuICAgICAgICAgICAgICAgIHtkYXRhOiBKU09OLnN0cmluZ2lmeSh7fSksIGNvbnRlbnRUeXBlOiAndGV4dC9odG1sJ31cbiAgICAgICAgICB9LFxuICAgICAgICAgIHJlcXVlc3RJbml0cyk7XG4gICAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uaHR0cCgncGF0aDIvbW9kZWwuanNvbicpO1xuICAgICAgdHJ5IHtcbiAgICAgICAgY29uc3QgZGF0YSA9IGF3YWl0IGhhbmRsZXIubG9hZCgpO1xuICAgICAgICBleHBlY3QoZGF0YSkudG9CZURlZmluZWQoKTtcbiAgICAgICAgZmFpbCgnTG9hZGluZyB3aXRoIGZldGNoIHJlamVjdGlvbiBzdWNjZWVkZWQgdW5leHBlY3RlZGx5LicpO1xuICAgICAgfSBjYXRjaCAoZXJyKSB7XG4gICAgICAgIC8vIFRoaXMgZXJyb3IgaXMgbW9ja2VkIGluIGJlZm9yZUVhY2hcbiAgICAgICAgZXhwZWN0KGVycikudG9FcXVhbCgncGF0aCBub3QgZm91bmQnKTtcbiAgICAgIH1cbiAgICB9KTtcbiAgICBpdCgnUHJvdmlkZSBXZWlnaHRGaWxlVHJhbnNsYXRlRnVuYycsIGFzeW5jICgpID0+IHtcbiAgICAgIGNvbnN0IHdlaWdodE1hbmlmZXN0MTogdGYuaW8uV2VpZ2h0c01hbmlmZXN0Q29uZmlnID0gW3tcbiAgICAgICAgcGF0aHM6IFsnd2VpZ2h0ZmlsZTAnXSxcbiAgICAgICAgd2VpZ2h0czogW1xuICAgICAgICAgIHtcbiAgICAgICAgICAgIG5hbWU6ICdkZW5zZS9rZXJuZWwnLFxuICAgICAgICAgICAgc2hhcGU6IFszLCAxXSxcbiAgICAgICAgICAgIGR0eXBlOiAnZmxvYXQzMicsXG4gICAgICAgICAgfSxcbiAgICAgICAgICB7XG4gICAgICAgICAgICBuYW1lOiAnZGVuc2UvYmlhcycsXG4gICAgICAgICAgICBzaGFwZTogWzJdLFxuICAgICAgICAgICAgZHR5cGU6ICdmbG9hdDMyJyxcbiAgICAgICAgICB9XG4gICAgICAgIF1cbiAgICAgIH1dO1xuICAgICAgY29uc3QgZmxvYXREYXRhID0gbmV3IEZsb2F0MzJBcnJheShbMSwgMywgMywgNywgNF0pO1xuICAgICAgc2V0dXBGYWtlV2VpZ2h0RmlsZXMoXG4gICAgICAgICAge1xuICAgICAgICAgICAgJy4vbW9kZWwuanNvbic6IHtcbiAgICAgICAgICAgICAgZGF0YTogSlNPTi5zdHJpbmdpZnkoe1xuICAgICAgICAgICAgICAgIG1vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxLFxuICAgICAgICAgICAgICAgIHdlaWdodHNNYW5pZmVzdDogd2VpZ2h0TWFuaWZlc3QxXG4gICAgICAgICAgICAgIH0pLFxuICAgICAgICAgICAgICBjb250ZW50VHlwZTogJ2FwcGxpY2F0aW9uL2pzb24nXG4gICAgICAgICAgICB9LFxuICAgICAgICAgICAgJ2F1dGhfd2VpZ2h0ZmlsZTAnOlxuICAgICAgICAgICAgICAgIHtkYXRhOiBmbG9hdERhdGEsIGNvbnRlbnRUeXBlOiAnYXBwbGljYXRpb24vb2N0ZXQtc3RyZWFtJ30sXG4gICAgICAgICAgfSxcbiAgICAgICAgICByZXF1ZXN0SW5pdHMpO1xuICAgICAgYXN5bmMgZnVuY3Rpb24gcHJlZml4V2VpZ2h0VXJsQ29udmVydGVyKHdlaWdodEZpbGU6IHN0cmluZyk6XG4gICAgICAgICAgUHJvbWlzZTxzdHJpbmc+IHtcbiAgICAgICAgLy8gQWRkICdhdXRoXycgcHJlZml4IHRvIHRoZSB3ZWlnaHQgZmlsZSB1cmwuXG4gICAgICAgIHJldHVybiBuZXcgUHJvbWlzZShcbiAgICAgICAgICAgIHJlc29sdmUgPT4gc2V0VGltZW91dChyZXNvbHZlLCAxLCAnYXV0aF8nICsgd2VpZ2h0RmlsZSkpO1xuICAgICAgfVxuXG4gICAgICBjb25zdCBoYW5kbGVyID0gdGYuaW8uaHR0cCgnLi9tb2RlbC5qc29uJywge1xuICAgICAgICByZXF1ZXN0SW5pdDoge2hlYWRlcnM6IHsnaGVhZGVyX2tleV8xJzogJ2hlYWRlcl92YWx1ZV8xJ319LFxuICAgICAgICB3ZWlnaHRVcmxDb252ZXJ0ZXI6IHByZWZpeFdlaWdodFVybENvbnZlcnRlclxuICAgICAgfSk7XG4gICAgICBjb25zdCBtb2RlbEFydGlmYWN0cyA9IGF3YWl0IGhhbmRsZXIubG9hZCgpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLm1vZGVsVG9wb2xvZ3kpLnRvRXF1YWwobW9kZWxUb3BvbG9neTEpO1xuICAgICAgZXhwZWN0KG1vZGVsQXJ0aWZhY3RzLndlaWdodFNwZWNzKS50b0VxdWFsKHdlaWdodE1hbmlmZXN0MVswXS53ZWlnaHRzKTtcbiAgICAgIGV4cGVjdChuZXcgRmxvYXQzMkFycmF5KG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEpKS50b0VxdWFsKGZsb2F0RGF0YSk7XG4gICAgICBleHBlY3QoT2JqZWN0LmtleXMocmVxdWVzdEluaXRzKS5sZW5ndGgpLnRvRXF1YWwoMik7XG4gICAgICBleHBlY3QoT2JqZWN0LmtleXMocmVxdWVzdEluaXRzKS5sZW5ndGgpLnRvRXF1YWwoMik7XG4gICAgICBleHBlY3QocmVxdWVzdEluaXRzWycuL21vZGVsLmpzb24nXS5oZWFkZXJzWydoZWFkZXJfa2V5XzEnXSlcbiAgICAgICAgICAudG9FcXVhbCgnaGVhZGVyX3ZhbHVlXzEnKTtcbiAgICAgIGV4cGVjdChyZXF1ZXN0SW5pdHNbJ2F1dGhfd2VpZ2h0ZmlsZTAnXS5oZWFkZXJzWydoZWFkZXJfa2V5XzEnXSlcbiAgICAgICAgICAudG9FcXVhbCgnaGVhZGVyX3ZhbHVlXzEnKTtcblxuICAgICAgZXhwZWN0KGZldGNoU3B5LmNhbGxzLm1vc3RSZWNlbnQoKS5vYmplY3QpLnRvRXF1YWwod2luZG93KTtcbiAgICB9KTtcbiAgfSk7XG5cbiAgaXQoJ092ZXJyaWRpbmcgQnJvd3NlckhUVFBSZXF1ZXN0IGZldGNoRnVuYycsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCB3ZWlnaHRNYW5pZmVzdDE6IHRmLmlvLldlaWdodHNNYW5pZmVzdENvbmZpZyA9IFt7XG4gICAgICBwYXRoczogWyd3ZWlnaHRmaWxlMCddLFxuICAgICAgd2VpZ2h0czogW1xuICAgICAgICB7XG4gICAgICAgICAgbmFtZTogJ2RlbnNlL2tlcm5lbCcsXG4gICAgICAgICAgc2hhcGU6IFszLCAxXSxcbiAgICAgICAgICBkdHlwZTogJ2Zsb2F0MzInLFxuICAgICAgICB9LFxuICAgICAgICB7XG4gICAgICAgICAgbmFtZTogJ2RlbnNlL2JpYXMnLFxuICAgICAgICAgIHNoYXBlOiBbMl0sXG4gICAgICAgICAgZHR5cGU6ICdmbG9hdDMyJyxcbiAgICAgICAgfVxuICAgICAgXVxuICAgIH1dO1xuICAgIGNvbnN0IGZsb2F0RGF0YSA9IG5ldyBGbG9hdDMyQXJyYXkoWzEsIDMsIDMsIDcsIDRdKTtcblxuICAgIGNvbnN0IGZldGNoSW5wdXRzOiBSZXF1ZXN0SW5mb1tdID0gW107XG4gICAgY29uc3QgZmV0Y2hJbml0czogUmVxdWVzdEluaXRbXSA9IFtdO1xuICAgIGFzeW5jIGZ1bmN0aW9uIGN1c3RvbUZldGNoKFxuICAgICAgICBpbnB1dDogUmVxdWVzdEluZm8sIGluaXQ/OiBSZXF1ZXN0SW5pdCk6IFByb21pc2U8UmVzcG9uc2U+IHtcbiAgICAgIGZldGNoSW5wdXRzLnB1c2goaW5wdXQpO1xuICAgICAgZmV0Y2hJbml0cy5wdXNoKGluaXQpO1xuXG4gICAgICBpZiAoaW5wdXQgPT09ICcuL21vZGVsLmpzb24nKSB7XG4gICAgICAgIHJldHVybiBuZXcgUmVzcG9uc2UoXG4gICAgICAgICAgICBKU09OLnN0cmluZ2lmeSh7XG4gICAgICAgICAgICAgIG1vZGVsVG9wb2xvZ3k6IG1vZGVsVG9wb2xvZ3kxLFxuICAgICAgICAgICAgICB3ZWlnaHRzTWFuaWZlc3Q6IHdlaWdodE1hbmlmZXN0MSxcbiAgICAgICAgICAgICAgdHJhaW5pbmdDb25maWc6IHRyYWluaW5nQ29uZmlnMVxuICAgICAgICAgICAgfSksXG4gICAgICAgICAgICB7c3RhdHVzOiAyMDAsIGhlYWRlcnM6IHsnY29udGVudC10eXBlJzogJ2FwcGxpY2F0aW9uL2pzb24nfX0pO1xuICAgICAgfSBlbHNlIGlmIChpbnB1dCA9PT0gJy4vd2VpZ2h0ZmlsZTAnKSB7XG4gICAgICAgIHJldHVybiBuZXcgUmVzcG9uc2UoZmxvYXREYXRhLCB7XG4gICAgICAgICAgc3RhdHVzOiAyMDAsXG4gICAgICAgICAgaGVhZGVyczogeydjb250ZW50LXR5cGUnOiAnYXBwbGljYXRpb24vb2N0ZXQtc3RyZWFtJ31cbiAgICAgICAgfSk7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICByZXR1cm4gbmV3IFJlc3BvbnNlKG51bGwsIHtzdGF0dXM6IDQwNH0pO1xuICAgICAgfVxuICAgIH1cblxuICAgIGNvbnN0IGhhbmRsZXIgPSB0Zi5pby5odHRwKFxuICAgICAgICAnLi9tb2RlbC5qc29uJyxcbiAgICAgICAge3JlcXVlc3RJbml0OiB7Y3JlZGVudGlhbHM6ICdpbmNsdWRlJ30sIGZldGNoRnVuYzogY3VzdG9tRmV0Y2h9KTtcbiAgICBjb25zdCBtb2RlbEFydGlmYWN0cyA9IGF3YWl0IGhhbmRsZXIubG9hZCgpO1xuICAgIGV4cGVjdChtb2RlbEFydGlmYWN0cy5tb2RlbFRvcG9sb2d5KS50b0VxdWFsKG1vZGVsVG9wb2xvZ3kxKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMudHJhaW5pbmdDb25maWcpLnRvRXF1YWwodHJhaW5pbmdDb25maWcxKTtcbiAgICBleHBlY3QobW9kZWxBcnRpZmFjdHMud2VpZ2h0U3BlY3MpLnRvRXF1YWwod2VpZ2h0TWFuaWZlc3QxWzBdLndlaWdodHMpO1xuICAgIGV4cGVjdChuZXcgRmxvYXQzMkFycmF5KG1vZGVsQXJ0aWZhY3RzLndlaWdodERhdGEpKS50b0VxdWFsKGZsb2F0RGF0YSk7XG5cbiAgICBleHBlY3QoZmV0Y2hJbnB1dHMpLnRvRXF1YWwoWycuL21vZGVsLmpzb24nLCAnLi93ZWlnaHRmaWxlMCddKTtcbiAgICBleHBlY3QoZmV0Y2hJbml0cy5sZW5ndGgpLnRvRXF1YWwoMik7XG4gICAgZXhwZWN0KGZldGNoSW5pdHNbMF0uY3JlZGVudGlhbHMpLnRvRXF1YWwoJ2luY2x1ZGUnKTtcbiAgICBleHBlY3QoZmV0Y2hJbml0c1sxXS5jcmVkZW50aWFscykudG9FcXVhbCgnaW5jbHVkZScpO1xuICB9KTtcbn0pO1xuIl19