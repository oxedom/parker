/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
import * as tf from './index';
import { ALL_ENVS, describeWithFlags } from './jasmine_util';
import { complex, scalar, tensor2d } from './ops/ops';
import { inferShape } from './tensor_util_env';
import * as util from './util';
describe('Util', () => {
    it('Correctly gets size from shape', () => {
        expect(util.sizeFromShape([1, 2, 3, 4])).toEqual(24);
    });
    it('Correctly identifies scalars', () => {
        expect(util.isScalarShape([])).toBe(true);
        expect(util.isScalarShape([1, 2])).toBe(false);
        expect(util.isScalarShape([1])).toBe(false);
    });
    it('Number arrays equal', () => {
        expect(util.arraysEqual([1, 2, 3, 6], [1, 2, 3, 6])).toBe(true);
        expect(util.arraysEqual([1, 2], [1, 2, 3])).toBe(false);
        expect(util.arraysEqual([1, 2, 5], [1, 2])).toBe(false);
    });
    it('Arrays shuffle randomly', () => {
        // Create 1000 numbers ordered
        const a = Array.apply(0, { length: 1000 }).map(Number.call, Number).slice(1);
        const b = [].concat(a); // copy ES5 style
        util.shuffle(a);
        expect(a).not.toEqual(b);
        expect(a.length).toEqual(b.length);
    });
    it('Multiple arrays shuffle together', () => {
        // Create 1000 numbers ordered
        const a = Array.apply(0, { length: 1000 }).map(Number.call, Number).slice(1);
        const b = [].concat(a); // copies
        const c = [].concat(a);
        util.shuffleCombo(a, b);
        expect(a).not.toEqual(c);
        expect(a).toEqual(b);
        expect(a.length).toEqual(c.length);
    });
    it('Is integer', () => {
        expect(util.isInt(0.5)).toBe(false);
        expect(util.isInt(1)).toBe(true);
    });
    it('Size to squarish shape (perfect square)', () => {
        expect(util.sizeToSquarishShape(9)).toEqual([3, 3]);
    });
    it('Size to squarish shape (prime number)', () => {
        expect(util.sizeToSquarishShape(11)).toEqual([4, 3]);
    });
    it('Size to squarish shape (almost square)', () => {
        expect(util.sizeToSquarishShape(35)).toEqual([6, 6]);
    });
    it('Size of 1 to squarish shape', () => {
        expect(util.sizeToSquarishShape(1)).toEqual([1, 1]);
    });
    it('infer shape single number', () => {
        expect(inferShape(4)).toEqual([]);
    });
    it('infer shape 1d array', () => {
        expect(inferShape([1, 2, 5])).toEqual([3]);
    });
    it('infer shape 2d array', () => {
        expect(inferShape([[1, 2, 5], [5, 4, 1]])).toEqual([2, 3]);
    });
    it('infer shape 3d array', () => {
        const a = [[[1, 2], [2, 3], [5, 6]], [[5, 6], [4, 5], [1, 2]]];
        expect(inferShape(a)).toEqual([2, 3, 2]);
    });
    it('infer shape 4d array', () => {
        const a = [
            [[[1], [2]], [[2], [3]], [[5], [6]]], [[[5], [6]], [[4], [5]], [[1], [2]]]
        ];
        expect(inferShape(a)).toEqual([2, 3, 2, 1]);
    });
    it('infer shape of typed array', () => {
        const a = new Float32Array([1, 2, 3, 4, 5]);
        expect(inferShape(a)).toEqual([5]);
    });
    it('infer shape of clamped typed array', () => {
        const a = new Uint8ClampedArray([1, 2, 3, 4, 5]);
        expect(inferShape(a)).toEqual([5]);
    });
    it('infer shape of Uint8Array[], string tensor', () => {
        const a = [new Uint8Array([1, 2]), new Uint8Array([3, 4])];
        expect(inferShape(a, 'string')).toEqual([2]);
    });
    it('infer shape of Uint8Array[][], string tensor', () => {
        const a = [
            [new Uint8Array([1]), new Uint8Array([2])],
            [new Uint8Array([1]), new Uint8Array([2])]
        ];
        expect(inferShape(a, 'string')).toEqual([2, 2]);
    });
    it('infer shape of Uint8Array[][][], string tensor', () => {
        const a = [
            [[new Uint8Array([1, 2])], [new Uint8Array([2, 1])]],
            [[new Uint8Array([1, 2])], [new Uint8Array([2, 1])]]
        ];
        expect(inferShape(a, 'string')).toEqual([2, 2, 1]);
    });
});
describe('util.flatten', () => {
    it('nested number arrays', () => {
        expect(util.flatten([[1, 2, 3], [4, 5, 6]])).toEqual([1, 2, 3, 4, 5, 6]);
        expect(util.flatten([[[1, 2], [3, 4], [5, 6], [7, 8]]])).toEqual([
            1, 2, 3, 4, 5, 6, 7, 8
        ]);
        expect(util.flatten([1, 2, 3, 4, 5, 6])).toEqual([1, 2, 3, 4, 5, 6]);
    });
    it('nested string arrays', () => {
        expect(util.flatten([['a', 'b'], ['c', [['d']]]])).toEqual([
            'a', 'b', 'c', 'd'
        ]);
        expect(util.flatten([['a', ['b']], ['c', [['d']], 'e']])).toEqual([
            'a', 'b', 'c', 'd', 'e'
        ]);
    });
    it('mixed TypedArray and number[]', () => {
        const data = [new Float32Array([1, 2]), 3, [4, 5, new Float32Array([6, 7])]];
        expect(util.flatten(data)).toEqual([1, 2, 3, 4, 5, 6, 7]);
    });
    it('nested Uint8Arrays, skipTypedArray=true', () => {
        const data = [
            [new Uint8Array([1, 2]), new Uint8Array([3, 4])],
            [new Uint8Array([5, 6]), new Uint8Array([7, 8])]
        ];
        expect(util.flatten(data, [], true)).toEqual([
            new Uint8Array([1, 2]), new Uint8Array([3, 4]), new Uint8Array([5, 6]),
            new Uint8Array([7, 8])
        ]);
    });
});
function encodeStrings(a) {
    return a.map(s => util.encodeString(s));
}
describe('util.bytesFromStringArray', () => {
    it('count bytes after utf8 encoding', () => {
        expect(util.bytesFromStringArray(encodeStrings(['a', 'bb', 'ccc'])))
            .toBe(6);
        expect(util.bytesFromStringArray(encodeStrings(['a', 'bb', 'cccddd'])))
            .toBe(9);
        expect(util.bytesFromStringArray(encodeStrings(['даниел']))).toBe(6 * 2);
    });
});
describe('util.inferDtype', () => {
    it('a single string => string', () => {
        expect(util.inferDtype('hello')).toBe('string');
    });
    it('a single boolean => bool', () => {
        expect(util.inferDtype(true)).toBe('bool');
        expect(util.inferDtype(false)).toBe('bool');
    });
    it('a single number => float32', () => {
        expect(util.inferDtype(0)).toBe('float32');
        expect(util.inferDtype(34)).toBe('float32');
    });
    it('a list of strings => string', () => {
        // Flat.
        expect(util.inferDtype(['a', 'b', 'c'])).toBe('string');
        // Nested.
        expect(util.inferDtype([
            [['a']], [['b']], [['c']], [['d']]
        ])).toBe('string');
    });
    it('a list of bools => float32', () => {
        // Flat.
        expect(util.inferDtype([false, true, false])).toBe('bool');
        // Nested.
        expect(util.inferDtype([
            [[true]], [[false]], [[true]], [[true]]
        ])).toBe('bool');
    });
    it('a list of numbers => float32', () => {
        // Flat.
        expect(util.inferDtype([0, 1, 2])).toBe('float32');
        // Nested.
        expect(util.inferDtype([[[0]], [[1]], [[2]], [[3]]])).toBe('float32');
    });
});
describe('util.repeatedTry', () => {
    it('resolves', (doneFn) => {
        let counter = 0;
        const checkFn = () => {
            counter++;
            if (counter === 2) {
                return true;
            }
            return false;
        };
        util.repeatedTry(checkFn).then(doneFn).catch(() => {
            throw new Error('Rejected backoff.');
        });
    });
    it('rejects', (doneFn) => {
        const checkFn = () => false;
        util.repeatedTry(checkFn, () => 0, 5)
            .then(() => {
            throw new Error('Backoff resolved');
        })
            .catch(doneFn);
    });
});
describe('util.inferFromImplicitShape', () => {
    it('empty shape', () => {
        const result = util.inferFromImplicitShape([], 0);
        expect(result).toEqual([]);
    });
    it('[2, 3, 4] -> [2, 3, 4]', () => {
        const result = util.inferFromImplicitShape([2, 3, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, -1, 4] -> [2, 3, 4], size=24', () => {
        const result = util.inferFromImplicitShape([2, -1, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[-1, 3, 4] -> [2, 3, 4], size=24', () => {
        const result = util.inferFromImplicitShape([-1, 3, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, 3, -1] -> [2, 3, 4], size=24', () => {
        const result = util.inferFromImplicitShape([2, 3, -1], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, -1, -1] throws error', () => {
        expect(() => util.inferFromImplicitShape([2, -1, -1], 24)).toThrowError();
    });
    it('[2, 3, -1] size=13 throws error', () => {
        expect(() => util.inferFromImplicitShape([2, 3, -1], 13)).toThrowError();
    });
    it('[2, 3, 4] size=25 (should be 24) throws error', () => {
        expect(() => util.inferFromImplicitShape([2, 3, 4], 25)).toThrowError();
    });
});
describe('util parseAxisParam', () => {
    it('axis=null returns no axes for scalar', () => {
        const axis = null;
        const shape = [];
        expect(util.parseAxisParam(axis, shape)).toEqual([]);
    });
    it('axis=null returns 0 axis for Tensor1D', () => {
        const axis = null;
        const shape = [4];
        expect(util.parseAxisParam(axis, shape)).toEqual([0]);
    });
    it('axis=null returns all axes for Tensor3D', () => {
        const axis = null;
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([0, 1, 2]);
    });
    it('axis as a single number', () => {
        const axis = 1;
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([1]);
    });
    it('axis as single negative number', () => {
        const axis = -1;
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([2]);
        const axis2 = -2;
        expect(util.parseAxisParam(axis2, shape)).toEqual([1]);
        const axis3 = -3;
        expect(util.parseAxisParam(axis3, shape)).toEqual([0]);
    });
    it('axis as list of negative numbers', () => {
        const axis = [-1, -3];
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([2, 0]);
    });
    it('axis as list of positive numbers', () => {
        const axis = [0, 2];
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([0, 2]);
    });
    it('axis as combo of positive and negative numbers', () => {
        const axis = [0, -1];
        const shape = [3, 1, 2];
        expect(util.parseAxisParam(axis, shape)).toEqual([0, 2]);
    });
    it('axis out of range throws error', () => {
        const axis = -4;
        const shape = [3, 1, 2];
        expect(() => util.parseAxisParam(axis, shape)).toThrowError();
        const axis2 = 4;
        expect(() => util.parseAxisParam(axis2, shape)).toThrowError();
    });
    it('axis a list with one number out of range throws error', () => {
        const axis = [0, 4];
        const shape = [3, 1, 2];
        expect(() => util.parseAxisParam(axis, shape)).toThrowError();
    });
    it('axis with decimal value throws error', () => {
        const axis = 0.5;
        const shape = [3, 1, 2];
        expect(() => util.parseAxisParam(axis, shape)).toThrowError();
    });
});
describe('util.squeezeShape', () => {
    it('scalar', () => {
        const { newShape, keptDims } = util.squeezeShape([]);
        expect(newShape).toEqual([]);
        expect(keptDims).toEqual([]);
    });
    it('1x1 reduced to scalar', () => {
        const { newShape, keptDims } = util.squeezeShape([1, 1]);
        expect(newShape).toEqual([]);
        expect(keptDims).toEqual([]);
    });
    it('1x3x1 reduced to [3]', () => {
        const { newShape, keptDims } = util.squeezeShape([1, 3, 1]);
        expect(newShape).toEqual([3]);
        expect(keptDims).toEqual([1]);
    });
    it('1x1x4 reduced to [4]', () => {
        const { newShape, keptDims } = util.squeezeShape([1, 1, 4]);
        expect(newShape).toEqual([4]);
        expect(keptDims).toEqual([2]);
    });
    it('2x3x4 not reduction', () => {
        const { newShape, keptDims } = util.squeezeShape([2, 3, 4]);
        expect(newShape).toEqual([2, 3, 4]);
        expect(keptDims).toEqual([0, 1, 2]);
    });
    describe('with axis', () => {
        it('should only reduce dimensions specified by axis', () => {
            const { newShape, keptDims } = util.squeezeShape([1, 1, 1, 1, 4], [1, 2]);
            expect(newShape).toEqual([1, 1, 4]);
            expect(keptDims).toEqual([0, 3, 4]);
        });
        it('should only reduce dimensions specified by negative axis', () => {
            const { newShape, keptDims } = util.squeezeShape([1, 1, 1, 1, 4], [-2, -3]);
            expect(newShape).toEqual([1, 1, 4]);
            expect(keptDims).toEqual([0, 1, 4]);
        });
        it('should only reduce dimensions specified by negative axis', () => {
            const axis = [-2, -3];
            util.squeezeShape([1, 1, 1, 1, 4], axis);
            expect(axis).toEqual([-2, -3]);
        });
        it('throws error when specified axis is not squeezable', () => {
            expect(() => util.squeezeShape([1, 1, 2, 1, 4], [1, 2])).toThrowError();
        });
        it('throws error when specified negative axis is not squeezable', () => {
            expect(() => util.squeezeShape([1, 1, 2, 1, 4], [-1, -2])).toThrowError();
        });
        it('throws error when specified axis is out of range', () => {
            expect(() => util.squeezeShape([1, 1, 2, 1, 4], [11, 22])).toThrowError();
        });
        it('throws error when specified negative axis is out of range', () => {
            expect(() => util.squeezeShape([1, 1, 2, 1, 4], [
                -11, -22
            ])).toThrowError();
        });
    });
});
describe('util.checkConversionForErrors', () => {
    it('Float32Array has NaN', () => {
        expect(() => util.checkConversionForErrors(new Float32Array([1, 2, 3, NaN, 4, 255]), 'float32'))
            .toThrowError();
    });
    it('Float32Array has Infinity', () => {
        expect(() => util.checkConversionForErrors(new Float32Array([1, 2, 3, Infinity, 4, 255]), 'float32'))
            .toThrowError();
    });
    it('Int32Array has NaN', () => {
        expect(() => util.checkConversionForErrors([1, 2, 3, 4, NaN], 'int32'))
            .toThrowError();
    });
});
describe('util.hasEncodingLoss', () => {
    it('complex64 to any', () => {
        expect(util.hasEncodingLoss('complex64', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('complex64', 'float32')).toBe(true);
        expect(util.hasEncodingLoss('complex64', 'int32')).toBe(true);
        expect(util.hasEncodingLoss('complex64', 'bool')).toBe(true);
    });
    it('any to complex64', () => {
        expect(util.hasEncodingLoss('bool', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('int32', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('float32', 'complex64')).toBe(false);
        expect(util.hasEncodingLoss('complex64', 'complex64')).toBe(false);
    });
    it('any to float32', () => {
        expect(util.hasEncodingLoss('bool', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('int32', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('complex64', 'float32')).toBe(true);
    });
    it('float32 to any', () => {
        expect(util.hasEncodingLoss('float32', 'float32')).toBe(false);
        expect(util.hasEncodingLoss('float32', 'int32')).toBe(true);
        expect(util.hasEncodingLoss('float32', 'bool')).toBe(true);
        expect(util.hasEncodingLoss('float32', 'complex64')).toBe(false);
    });
    it('int32 to lower', () => {
        expect(util.hasEncodingLoss('int32', 'int32')).toBe(false);
        expect(util.hasEncodingLoss('int32', 'bool')).toBe(true);
    });
    it('lower to int32', () => {
        expect(util.hasEncodingLoss('bool', 'int32')).toBe(false);
    });
    it('bool to bool', () => {
        expect(util.hasEncodingLoss('bool', 'bool')).toBe(false);
    });
});
describeWithFlags('util.toNestedArray', ALL_ENVS, () => {
    it('2 dimensions', () => {
        const a = new Float32Array([1, 2, 3, 4, 5, 6]);
        expect(util.toNestedArray([2, 3], a)).toEqual([[1, 2, 3], [4, 5, 6]]);
    });
    it('3 dimensions (2x2x3)', () => {
        const a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        expect(util.toNestedArray([2, 2, 3], a)).toEqual([
            [[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]
        ]);
    });
    it('3 dimensions (3x2x2)', () => {
        const a = new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        expect(util.toNestedArray([3, 2, 2], a)).toEqual([
            [[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8, 9], [10, 11]]
        ]);
    });
    it('invalid dimension', () => {
        const a = new Float32Array([1, 2, 3]);
        expect(() => util.toNestedArray([2, 2], a)).toThrowError();
    });
    it('tensor to nested array', async () => {
        const x = tensor2d([1, 2, 3, 4], [2, 2]);
        expect(util.toNestedArray(x.shape, await x.data())).toEqual([
            [1, 2], [3, 4]
        ]);
    });
    it('scalar to nested array', async () => {
        const x = scalar(1);
        expect(util.toNestedArray(x.shape, await x.data())).toEqual(1);
    });
    it('tensor with zero shape', () => {
        const a = new Float32Array([0, 1]);
        expect(util.toNestedArray([1, 0, 2], a)).toEqual([]);
    });
});
describeWithFlags('util.toNestedArray for a complex tensor', ALL_ENVS, () => {
    it('2 dimensions', () => {
        const a = new Float32Array([1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16]);
        expect(util.toNestedArray([2, 3], a, true)).toEqual([
            [1, 11, 2, 12, 3, 13], [4, 14, 5, 15, 6, 16]
        ]);
    });
    it('3 dimensions (2x2x3)', () => {
        const a = new Float32Array([
            0, 50, 1, 51, 2, 52, 3, 53, 4, 54, 5, 55,
            6, 56, 7, 57, 8, 58, 9, 59, 10, 60, 11, 61
        ]);
        expect(util.toNestedArray([2, 2, 3], a, true)).toEqual([
            [[0, 50, 1, 51, 2, 52], [3, 53, 4, 54, 5, 55]],
            [[6, 56, 7, 57, 8, 58], [9, 59, 10, 60, 11, 61]]
        ]);
    });
    it('3 dimensions (3x2x2)', () => {
        const a = new Float32Array([
            0, 50, 1, 51, 2, 52, 3, 53, 4, 54, 5, 55,
            6, 56, 7, 57, 8, 58, 9, 59, 10, 60, 11, 61
        ]);
        expect(util.toNestedArray([3, 2, 2], a, true)).toEqual([
            [[0, 50, 1, 51], [2, 52, 3, 53]], [[4, 54, 5, 55], [6, 56, 7, 57]],
            [[8, 58, 9, 59], [10, 60, 11, 61]]
        ]);
    });
    it('invalid dimension', () => {
        const a = new Float32Array([1, 11, 2, 12, 3, 13]);
        expect(() => util.toNestedArray([2, 2], a, true)).toThrowError();
    });
    it('tensor to nested array', async () => {
        const x = complex([[1, 2], [3, 4]], [[11, 12], [13, 14]]);
        expect(util.toNestedArray(x.shape, await x.data(), true)).toEqual([
            [1, 11, 2, 12], [3, 13, 4, 14]
        ]);
    });
});
describe('util.fetch', () => {
    it('should call the platform fetch', () => {
        spyOn(tf.env().platform, 'fetch').and.callFake(() => { });
        util.fetch('test/path', { method: 'GET' });
        expect(tf.env().platform.fetch).toHaveBeenCalledWith('test/path', {
            method: 'GET'
        });
    });
});
describe('util.encodeString', () => {
    it('Encode an empty string, default encoding', () => {
        const res = util.encodeString('');
        expect(res).toEqual(new Uint8Array([]));
    });
    it('Encode an empty string, utf-8 encoding', () => {
        const res = util.encodeString('', 'utf-8');
        expect(res).toEqual(new Uint8Array([]));
    });
    it('Encode an empty string, invalid decoding', () => {
        expect(() => util.encodeString('', 'foobarbax')).toThrowError();
    });
    it('Encode cyrillic letters', () => {
        const res = util.encodeString('Kaкo стe');
        expect(res).toEqual(new Uint8Array([75, 97, 208, 186, 111, 32, 209, 129, 209, 130, 101]));
    });
    it('Encode ascii letters', () => {
        const res = util.encodeString('hello');
        expect(res).toEqual(new Uint8Array([104, 101, 108, 108, 111]));
    });
});
describe('util.decodeString', () => {
    it('decode an empty string', () => {
        const s = util.decodeString(new Uint8Array([]));
        expect(s).toEqual('');
    });
    it('decode ascii', () => {
        const s = util.decodeString(new Uint8Array([104, 101, 108, 108, 111]));
        expect(s).toEqual('hello');
    });
    it('decode cyrillic', () => {
        const s = util.decodeString(new Uint8Array([75, 97, 208, 186, 111, 32, 209, 129, 209, 130, 101]));
        expect(s).toEqual('Kaкo стe');
    });
    it('decode utf-16', () => {
        const s = util.decodeString(new Uint8Array([255, 254, 237, 139, 0, 138, 4, 89, 6, 116]), 'utf-16');
        // UTF-16 allows optional presence of byte-order-mark (BOM)
        // Construct string for '语言处理', with and without BOM
        const expected = String.fromCodePoint(0x8bed, 0x8a00, 0x5904, 0x7406);
        const expectedBOM = String.fromCodePoint(0xfeff, 0x8bed, 0x8a00, 0x5904, 0x7406);
        if (s.codePointAt(0) === 0xfeff) {
            expect(s).toEqual(expectedBOM);
        }
        else {
            expect(s).toEqual(expected);
        }
    });
    it('assert promise', () => {
        const promise = new Promise(() => { });
        expect(util.isPromise(promise)).toBeTruthy();
        const promise2 = { then: () => { } };
        expect(util.isPromise(promise2)).toBeTruthy();
        const promise3 = {};
        expect(util.isPromise(promise3)).toBeFalsy();
    });
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidXRpbF90ZXN0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy91dGlsX3Rlc3QudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxLQUFLLEVBQUUsTUFBTSxTQUFTLENBQUM7QUFDOUIsT0FBTyxFQUFDLFFBQVEsRUFBRSxpQkFBaUIsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQzNELE9BQU8sRUFBQyxPQUFPLEVBQUUsTUFBTSxFQUFFLFFBQVEsRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNwRCxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFDN0MsT0FBTyxLQUFLLElBQUksTUFBTSxRQUFRLENBQUM7QUFFL0IsUUFBUSxDQUFDLE1BQU0sRUFBRSxHQUFHLEVBQUU7SUFDcEIsRUFBRSxDQUFDLGdDQUFnQyxFQUFFLEdBQUcsRUFBRTtRQUN4QyxNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxDQUFDLENBQUM7SUFDdkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsOEJBQThCLEVBQUUsR0FBRyxFQUFFO1FBQ3RDLE1BQU0sQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDL0MsTUFBTSxDQUFDLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUss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QyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDN0MsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsc0JBQXNCLEVBQUUsR0FBRyxFQUFFO1FBQzlCLE1BQU0sQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzdELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNCQUFzQixFQUFFLEdBQUcsRUFBRTtRQUM5QixNQUFNLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMvRCxNQUFNLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzNDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNCQUFzQixFQUFFLEdBQUcsRUFBRTtRQUM5QixNQUFNLENBQUMsR0FBRztZQUNSLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMs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Q0FBQyxDQUFDLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDckQsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQztBQUVILFFBQVEsQ0FBQyxjQUFjLEVBQUUsR0FBRyxFQUFFO0lBQzVCLEVBQUUsQ0FBQyxzQkFBc0IsRUFBRSxHQUFHLEVBQUU7UUFDOUIsTUFBTSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN6RSxNQUFNLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUM7WUFDL0QsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUM7U0FDdkIsQ0FBQyxDQUFDO1FBQ0gsTUFBTSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDdkUsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsc0JBQXNCLEVBQUUsR0FBRyxFQUFFO1FBQzlCLE1BQU0sQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxHQUFHLEVBQUUsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUM7WUFDekQsR0FBRyxFQUFFLEdBQUcsRUFBRSxHQUFHLEVBQUUsR0FBRztTQUNuQixDQUFDLENBQUM7UUFDSCxNQUFNLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUM7WUFDaEUsR0FBRyxFQUFFLEdBQUcsRUFBRSxHQUFHLEVBQUUsR0FBRyxFQUFFLEdBQUc7U0FDeEIsQ0FBQyxDQUFDO0lBQ0wsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsK0JBQStCLEVBQUUsR0FBRyxFQUFFO1FBQ3ZDLE1BQU0sSUFBSSxHQUNOLENBQUMsSUFBSSxZQUFZLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLElBQUksWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3BFLE1BQU0sQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM1RCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyx5Q0FBeUMsRUFBRSxHQUFHLEVBQUU7UUFDakQsTUFBTSxJQUFJLEdBQUc7WUFDWCxDQUFDLElBQUksVUFBVSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNoRCxDQUFDLElBQUksVUFBVSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztTQUNqRCxDQUFDO1FBQ0YsTUFBTSxDQUFDLElBQUksQ0FBQyxPQUFPLENBQUMsSUFBSSxFQUFFLEVBQUUsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQztZQUMzQyxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLElBQUksVUFBVSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7WUFDdEUsSUFBSSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7U0FDdkIsQ0FBQyxDQUFDO0lBQ0wsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQztBQUVILFNBQVMsYUFBYSxDQUFDLENBQVc7SUFDaEMsT0FBTyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQzFDLENBQUM7QUFFRCxRQUFRLENBQUMsMkJBQTJCLEVBQUUsR0FBRyxFQUFFO0lBQ3pDLEVBQUUsQ0FBQyxpQ0FBaUMsRUFBRSxHQUFHLEVBQUU7UUFDekMsTUFBTSxDQUFDLElBQUksQ0FBQyxvQkFBb0IsQ0FBQyxhQUFhLENBQUMsQ0FBQyxHQUFHLEVBQUUsSUFBSSxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUMvRCxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDYixNQUFNLENBQUMsSUFBSSxDQUFDLG9CQUFvQixDQUFDLGFBQWEsQ0FBQyxDQUFDLEdBQUcsRUFBRSxJQUFJLEVBQUUsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQ2xFLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNiLE1BQU0sQ0FBQyxJQUFJLENBQUMsb0JBQW9CLENBQUMsYUFBYSxDQUFDLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUMzRSxDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDO0FBRUgsUUFBUSxDQUFDLGlCQUFpQixFQUFFLEdBQUcsRUFBRTtJQUMvQixFQUFFLENBQUMsMkJBQTJCLEVBQUUsR0FBRyxFQUFFO1FBQ25DLE1BQU0sQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDO0lBQ2xELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDBCQUEwQixFQUFFLEdBQUcsRUFBRTtRQUNsQyxNQUFNLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUMzQyxNQUFNLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztJQUM5QyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyw0QkFBNEIsRUFBRSxHQUFHLEVBQUU7UUFDcEMsTUFBTSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDM0MsTUFBTSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7SUFDOUMsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsNkJBQTZCLEVBQUUsR0FBRyxFQUFFO1FBQ3JDLFFBQVE7UUFDUixNQUFNLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQyxDQUFDLEdBQUcsRUFBRSxHQUFHLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQztRQUN4RCxVQUFVO1FBQ1YsTUFBTSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUM7WUFDckIsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDO1NBQ25DLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQztJQUNyQixDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyw0QkFBNEIsRUFBRSxHQUFHLEVBQUU7UUFDcEMsUUFBUTtRQUNSLE1BQU0sQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDLENBQUMsS0FBSyxFQUFFLElBQUksRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQzNELFVBQVU7UUFDVixNQUFNLENBQUMsSUFBSSxDQUFDLFVBQVUsQ0FBQztZQUNyQixDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7U0FDeEMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQ25CLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDhCQUE4QixFQUFFLEdBQUcsRUFBRTtRQUN0QyxRQUFRO1FBQ1IsTUFBTSxDQUFDLElBQUksQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLENBQUM7UUFDbkQsVUFBVTtRQUNWLE1BQU0sQ0FBQyxJQUFJLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDO0lBQ3hFLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUM7QUFFSCxRQUFRLENBQUMsa0JBQWtCLEVBQUUsR0FBRyxFQUFFO0lBQ2hDLEVBQUUsQ0FBQyxVQUFVLEVBQUUsQ0FBQyxNQUFNLEVBQUUsRUFBRTtRQUN4QixJQUFJLE9BQU8sR0FBRyxDQUFDLENBQUM7UUFDaEIsTUFBTSxPQUFPLEdBQUcsR0FBRyxFQUFFO1lBQ25CLE9BQU8sRUFBRSxDQUFDO1lBQ1YsSUFBSSxPQUFPLEtBQUssQ0FBQyxFQUFFO2dCQUNqQixPQUFPLElBQUksQ0FBQzthQUNiO1lBQ0QsT0FBTyxLQUFLLENBQUM7UUFDZixDQUFDLENBQUM7UUFFRixJQUFJLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxLQUFLLENBQUMsR0FBRyxFQUFFO1lBQ2hELE1BQU0sSUFBSSxLQUFLLENBQUMsbUJBQW1CLENBQUMsQ0FBQztRQUN2QyxDQUFDLENBQUMsQ0FBQztJQUNMLENBQUMsQ0FBQyxDQUFDO0lBQ0gsRUFBRSxDQUFDLFNBQVMsRUFBRSxDQUFDLE1BQU0sRUFBRSxFQUFFO1FBQ3ZCLE1BQU0sT0FBTyxHQUFHLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQztRQUU1QixJQUFJLENBQUMsV0FBVyxDQUFDLE9BQU8sRUFBRSxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDO2FBQ2hDLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDVCxNQUFNLElBQUksS0FBSyxDQUFDLGtCQUFrQixDQUFDLENBQUM7UUFDdEMsQ0FBQyxDQUFDO2FBQ0QsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDO0lBQ3JCLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUM7QUFFSCxRQUFRLENBQUMsNkJBQTZCLEVBQUUsR0FBRyxFQUFFO0lBQzNDLEVBQUUsQ0FBQyxhQUFhLEVBQUUsR0FBRyxFQUFFO1FBQ3JCLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxzQkFBc0IsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDbEQsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLENBQUMsQ0FBQztJQUM3QixDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyx3QkFBd0IsRUFBRSxHQUFHLEVBQUU7UUFDaEMsTUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLHNCQUFzQixDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQztRQUMxRCxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3BDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGtDQUFrQyxFQUFFLEdBQUcsRUFBRTtRQUMxQyxNQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsc0JBQXNCLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUM7UUFDM0QsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNwQyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxrQ0FBa0MsRUFBRSxHQUFHLEVBQUU7UUFDMUMsTUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLHNCQUFzQixDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxDQUFDO1FBQzNELE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDcEMsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsa0NBQWtDLEVBQUUsR0FBRyxFQUFFO1FBQzFDLE1BQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxzQkFBc0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQztRQUMzRCxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3BDLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLDBCQUEwQixFQUFFLEdBQUcsRUFBRTtRQUNsQyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLHNCQUFzQixDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQyxZQUFZLEVBQUUsQ0FBQztJQUM1RSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxpQ0FBaUMsRUFBRSxHQUFHLEVBQUU7UUFDekMsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxzQkFBc0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLFlBQVksRUFBRSxDQUFDO0lBQzNFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLCtDQUErQyxFQUFFLEdBQUcsRUFBRTtRQUN2RCxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLHNCQUFzQixDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLFlBQVksRUFBRSxDQUFDO0lBQzFFLENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUM7QUFFSCxRQUFRLENBQUMscUJBQXFCLEVBQUUsR0FBRyxFQUFFO0lBQ25DLEVBQUUsQ0FBQyxzQ0FBc0MsRUFBRSxHQUFHLEVBQUU7UUFDOUMsTUFBTSxJQUFJLEdBQVcsSUFBSSxDQUFDO1FBQzFCLE1BQU0sS0FBSyxHQUFhLEVBQUUsQ0FBQztRQUMzQixNQUFNLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxJQUFJLEVBQUUsS0FBSyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxDQUFDLENBQUM7SUFDdkQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsdUNBQXVDLEVBQUUsR0FBRyxFQUFFO1FBQy9DLE1BQU0sSUFBSSxHQUFXLElBQUksQ0FBQztRQUMxQixNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2xCLE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLElBQUksRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDeEQsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMseUNBQXlDLEVBQUUsR0FBRyxFQUFFO1FBQ2pELE1BQU0sSUFBSSxHQUFhLElBQUksQ0FBQztRQUM1QixNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzlELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHlCQUF5QixFQUFFLEdBQUcsRUFBRTtRQUNqQyxNQUFNLElBQUksR0FBRyxDQUFDLENBQUM7UUFDZixNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN4RCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxnQ0FBZ0MsRUFBRSxHQUFHLEVBQUU7UUFDeEMsTUFBTSxJQUFJLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFDaEIsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLElBQUksRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFdEQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFDakIsTUFBTSxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUV2RCxNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsQ0FBQztRQUNqQixNQUFNLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxLQUFLLEVBQUUsS0FBSyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGtDQUFrQyxFQUFFLEdBQUcsRUFBRTtRQUMxQyxNQUFNLElBQUksR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdEIsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLElBQUksRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzNELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGtDQUFrQyxFQUFFLEdBQUcsRUFBRTtRQUMxQyxNQUFNLElBQUksR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNwQixNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDM0QsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsZ0RBQWdELEVBQUUsR0FBRyxFQUFFO1FBQ3hELE1BQU0sSUFBSSxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDckIsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLElBQUksRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzNELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGdDQUFnQyxFQUFFLEdBQUcsRUFBRTtRQUN4QyxNQUFNLElBQUksR0FBRyxDQUFDLENBQUMsQ0FBQztRQUNoQixNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsWUFBWSxFQUFFLENBQUM7UUFFOUQsTUFBTSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2hCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQyxDQUFDLFlBQVksRUFBRSxDQUFDO0lBQ2pFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHVEQUF1RCxFQUFFLEdBQUcsRUFBRTtRQUMvRCxNQUFNLElBQUksR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNwQixNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsWUFBWSxFQUFFLENBQUM7SUFDaEUsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsc0NBQXNDLEVBQUUsR0FBRyxFQUFFO1FBQzlDLE1BQU0sSUFBSSxHQUFHLEdBQUcsQ0FBQztRQUNqQixNQUFNLEtBQUssR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFDeEIsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxFQUFFLEtBQUssQ0FBQyxDQUFDLENBQUMsWUFBWSxFQUFFLENBQUM7SUFDaEUsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQztBQUVILFFBQVEsQ0FBQyxtQkFBbUIsRUFBRSxHQUFHLEVBQUU7SUFDakMsRUFBRSxDQUFDLFFBQVEsRUFBRSxHQUFHLEVBQUU7UUFDaEIsTUFBTSxFQUFDLFFBQVEsRUFBRSxRQUFRLEVBQUMsR0FBRyxJQUFJLENBQUMsWUFBWSxDQUFDLEVBQUUsQ0FBQyxDQUFDO1FBQ25ELE1BQU0sQ0FBQyxRQUFRLENBQUMsQ0FBQyxPQUFPLENBQUMsRUFBRSxDQUFDLENBQUM7UUFDN0IsTUFBTSxDQUFDLFFBQVEsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLENBQUMsQ0FBQztJQUMvQixDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyx1QkFBdUIsRUFBRSxHQUFHLEVBQUU7UUFDL0IsTUFBTSxFQUFDLFFBQVEsRUFBRSxRQUFRLEVBQUMsR0FBRyxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDdkQsTUFBTSxDQUFDLFFBQVEsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUM3QixNQUFNLENBQUMsUUFBUSxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsQ0FBQyxDQUFDO0lBQy9CLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNCQUFzQixFQUFFLEdBQUcsRUFBRTtRQUM5QixNQUFNLEVBQUMsUUFBUSxFQUFFLFFBQVEsRUFBQyxHQUFHLElBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDMUQsTUFBTSxDQUFDLFFBQVEsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDOUIsTUFBTSxDQUFDLFFBQVEsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDaEMsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsc0JBQXNCLEVBQUUsR0FBRyxFQUFFO1FBQzlCLE1BQU0sRUFBQyxRQUFRLEVBQUUsUUFBUSxFQUFDLEdBQUcsSUFBSSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMxRCxNQUFNLENBQUMsUUFBUSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUM5QixNQUFNLENBQUMsUUFBUSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNoQyxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxxQkFBcUIsRUFBRSxHQUFHLEVBQUU7UUFDN0IsTUFBTSxFQUFDLFFBQVEsRUFBRSxRQUFRLEVBQUMsR0FBRyxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzFELE1BQU0sQ0FBQyxRQUFRLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDcEMsTUFBTSxDQUFDLFFBQVEsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN0QyxDQUFDLENBQUMsQ0FBQztJQUVILFFBQVEsQ0FBQyxXQUFXLEVBQUUsR0FBRyxFQUFFO1FBQ3pCLEVBQUUsQ0FBQyxpREFBaUQsRUFBRSxHQUFHLEVBQUU7WUFDekQsTUFBTSxFQUFDLFFBQVEsRUFBRSxRQUFRLEVBQUMsR0FBRyxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDeEUsTUFBTSxDQUFDLFFBQVEsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNwQyxNQUFNLENBQUMsUUFBUSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3RDLENBQUMsQ0FBQyxDQUFDO1FBQ0gsRUFBRSxDQUFDLDBEQUEwRCxFQUFFLEdBQUcsRUFBRTtZQUNsRSxNQUFNLEVBQUMsUUFBUSxFQUFFLFFBQVEsRUFBQyxHQUFHLElBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDMUUsTUFBTSxDQUFDLFFBQVEsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNwQyxNQUFNLENBQUMsUUFBUSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3RDLENBQUMsQ0FBQyxDQUFDO1FBQ0gsRUFBRSxDQUFDLDBEQUEwRCxFQUFFLEdBQUcsRUFBRTtZQUNsRSxNQUFNLElBQUksR0FBRyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdEIsSUFBSSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQztZQUN6QyxNQUFNLENBQUMsSUFBSSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pDLENBQUMsQ0FBQyxDQUFDO1FBQ0gsRUFBRSxDQUFDLG9EQUFvRCxFQUFFLEdBQUcsRUFBRTtZQUM1RCxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsWUFBWSxFQUFFLENBQUM7UUFDMUUsQ0FBQyxDQUFDLENBQUM7UUFDSCxFQUFFLENBQUMsNkRBQTZELEVBQUUsR0FBRyxFQUFFO1lBQ3JFLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsWUFBWSxFQUFFLENBQUM7UUFDNUUsQ0FBQyxDQUFDLENBQUM7UUFDSCxFQUFFLENBQUMsa0RBQWtELEVBQUUsR0FBRyxFQUFFO1lBQzFELE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxZQUFZLEVBQUUsQ0FBQztRQUM1RSxDQUFDLENBQUMsQ0FBQztRQUNILEVBQUUsQ0FBQywyREFBMkQsRUFBRSxHQUFHLEVBQUU7WUFDbkUsTUFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUU7Z0JBQzlDLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRTthQUNULENBQUMsQ0FBQyxDQUFDLFlBQVksRUFBRSxDQUFDO1FBQ3JCLENBQUMsQ0FBQyxDQUFDO0lBQ0wsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQztBQUVILFFBQVEsQ0FBQywrQkFBK0IsRUFBRSxHQUFHLEVBQUU7SUFDN0MsRUFBRSxDQUFDLHNCQUFzQixFQUFFLEdBQUcsRUFBRTtRQUM5QixNQUFNLENBQ0YsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLHdCQUF3QixDQUMvQixJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUMsRUFBRSxTQUFTLENBQUMsQ0FBQzthQUN4RCxZQUFZLEVBQUUsQ0FBQztJQUN0QixDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQywyQkFBMkIsRUFBRSxHQUFHLEVBQUU7UUFDbkMsTUFBTSxDQUNGLEdBQUcsRUFBRSxDQUFDLElBQUksQ0FBQyx3QkFBd0IsQ0FDL0IsSUFBSSxZQUFZLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxRQUFRLEVBQUUsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUM7YUFDN0QsWUFBWSxFQUFFLENBQUM7SUFDdEIsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsb0JBQW9CLEVBQUUsR0FBRyxFQUFFO1FBQzVCLE1BQU0sQ0FBQyxHQUFHLEVBQUUsQ0FBQyxJQUFJLENBQUMsd0JBQXdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsR0FBRyxDQUFDLEVBQUUsT0FBTyxDQUFDLENBQUM7YUFDbEUsWUFBWSxFQUFFLENBQUM7SUFDdEIsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQztBQUVILFFBQVEsQ0FBQyxzQkFBc0IsRUFBRSxHQUFHLEVBQUU7SUFDcEMsRUFBRSxDQUFDLGtCQUFrQixFQUFFLEdBQUcsRUFBRTtRQUMxQixNQUFNLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxXQUFXLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDbkUsTUFBTSxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsV0FBVyxFQUFFLFNBQVMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQ2hFLE1BQU0sQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLFdBQVcsRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUM5RCxNQUFNLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxXQUFXLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDL0QsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsa0JBQWtCLEVBQUUsR0FBRyxFQUFFO1FBQzFCLE1BQU0sQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLE1BQU0sRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUM5RCxNQUFNLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxPQUFPLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDL0QsTUFBTSxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsU0FBUyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ2pFLE1BQU0sQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLFdBQVcsRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUNyRSxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxnQkFBZ0IsRUFBRSxHQUFHLEVBQUU7UUFDeEIsTUFBTSxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsTUFBTSxFQUFFLFNBQVMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQzVELE1BQU0sQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLE9BQU8sRUFBRSxTQUFTLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUM3RCxNQUFNLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxTQUFTLEVBQUUsU0FBUyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDL0QsTUFBTSxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsV0FBVyxFQUFFLFNBQVMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQ2xFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLGdCQUFnQixFQUFFLEdBQUcsRUFBRTtRQUN4QixNQUFNLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxTQUFTLEVBQUUsU0FBUyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDL0QsTUFBTSxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsU0FBUyxFQUFFLE9BQU8sQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDO1FBQzVELE1BQU0sQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLFNBQVMsRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUMzRCxNQUFNLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxTQUFTLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDbkUsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsZ0JBQWdCLEVBQUUsR0FBRyxFQUFFO1FBQ3hCLE1BQU0sQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLE9BQU8sRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUMzRCxNQUFNLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxPQUFPLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDM0QsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsZ0JBQWdCLEVBQUUsR0FBRyxFQUFFO1FBQ3hCLE1BQU0sQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLE1BQU0sRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUM1RCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxjQUFjLEVBQUUsR0FBRyxFQUFFO1FBQ3RCLE1BQU0sQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLE1BQU0sRUFBRSxNQUFNLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUMzRCxDQUFDLENBQUMsQ0FBQztBQUNMLENBQUMsQ0FBQyxDQUFDO0FBRUgsaUJBQWlCLENBQUMsb0JBQW9CLEVBQUUsUUFBUSxFQUFFLEdBQUcsRUFBRTtJQUNyRCxFQUFFLENBQUMsY0FBYyxFQUFFLEdBQUcsRUFBRTtRQUN0QixNQUFNLENBQUMsR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUMvQyxNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3hFLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNCQUFzQixFQUFFLEdBQUcsRUFBRTtRQUM5QixNQUFNLENBQUMsR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNuRSxNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUM7WUFDL0MsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQ2pELENBQUMsQ0FBQztJQUNMLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHNCQUFzQixFQUFFLEdBQUcsRUFBRTtRQUM5QixNQUFNLENBQUMsR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUMsQ0FBQztRQUNuRSxNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUM7WUFDL0MsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO1NBQ3ZELENBQUMsQ0FBQztJQUNMLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLG1CQUFtQixFQUFFLEdBQUcsRUFBRTtRQUMzQixNQUFNLENBQUMsR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN0QyxNQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLFlBQVksRUFBRSxDQUFDO0lBQzdELENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHdCQUF3QixFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3RDLE1BQU0sQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDekMsTUFBTSxDQUFDLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxNQUFNLENBQUMsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDO1lBQzFELENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztTQUNmLENBQUMsQ0FBQztJQUNMLENBQUMsQ0FBQyxDQUFDO0lBRUgsRUFBRSxDQUFDLHdCQUF3QixFQUFFLEtBQUssSUFBSSxFQUFFO1FBQ3RDLE1BQU0sQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNwQixNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLE1BQU0sQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDakUsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsd0JBQXdCLEVBQUUsR0FBRyxFQUFFO1FBQ2hDLE1BQU0sQ0FBQyxHQUFHLElBQUksWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDbkMsTUFBTSxDQUFDLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLEVBQUUsQ0FBQyxDQUFDO0lBQ3ZELENBQUMsQ0FBQyxDQUFDO0FBQ0wsQ0FBQyxDQUFDLENBQUM7QUFFSCxpQkFBaUIsQ0FBQyx5Q0FBeUMsRUFBRSxRQUFRLEVBQUUsR0FBRyxFQUFFO0lBQzFFLEVBQUUsQ0FBQyxjQUFjLEVBQUUsR0FBRyxFQUFFO1FBQ3RCLE1BQU0sQ0FBQyxHQUFHLElBQUksWUFBWSxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQ3ZFLE1BQU0sQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDLE9BQU8sQ0FBQztZQUNsRCxDQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztTQUM3QyxDQUFDLENBQUM7SUFDTCxDQUFDLENBQUMsQ0FBQztJQUVILEVBQUUsQ0FBQyxzQkFBc0IsRUFBRSxHQUFHLEVBQUU7UUFDOUIsTUFBTSxDQUFDLEdBQUcsSUFBSSxZQUFZLENBQUM7WUFDekIsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUcsRUFBRSxFQUFFLENBQUMsRUFBRyxFQUFFO1lBQzFDLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRTtTQUMzQyxDQUFDLENBQUM7UUFDSCxNQUFNLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDO1lBQ3JELENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQztZQUM5QyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7U0FDakQsQ0FBQyxDQUFDO0lBQ0wsQ0FBQyxDQUFDLENBQUM7SUFFSCxFQUFFLENBQUMsc0JBQXNCLEVBQUUsR0FBRyxFQUFFO1FBQzlCLE1BQU0sQ0FBQyxHQUFHLElBQUksWUFBWSxDQUFDO1lBQ3pCLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFHLEVBQUUsRUFBRSxDQUFDLEVBQUcsRUFBRTtZQUMxQyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxFQUFFLEVBQUU7U0FDM0MsQ0FBQyxDQUFDO1FBQ0gsTUFBTSxDQUFDLElBQUksQ0FBQyxhQUFhLENBQUMsQ0FBQyxD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LENBQUM7UUFDOUMsTUFBTSxRQUFRLEdBQUcsRUFBRSxDQUFDO1FBQ3BCLE1BQU0sQ0FBQyxJQUFJLENBQUMsU0FBUyxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUMsU0FBUyxFQUFFLENBQUM7SUFDL0MsQ0FBQyxDQUFDLENBQUM7QUFDTCxDQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE3IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0ICogYXMgdGYgZnJvbSAnLi9pbmRleCc7XG5pbXBvcnQge0FMTF9FTlZTLCBkZXNjcmliZVdpdGhGbGFnc30gZnJvbSAnLi9qYXNtaW5lX3V0aWwnO1xuaW1wb3J0IHtjb21wbGV4LCBzY2FsYXIsIHRlbnNvcjJkfSBmcm9tICcuL29wcy9vcHMnO1xuaW1wb3J0IHtpbmZlclNoYXBlfSBmcm9tICcuL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4vdXRpbCc7XG5cbmRlc2NyaWJlKCdVdGlsJywgKCkgPT4ge1xuICBpdCgnQ29ycmVjdGx5IGdldHMgc2l6ZSBmcm9tIHNoYXBlJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLnNpemVGcm9tU2hhcGUoWzEsIDIsIDMsIDRdKSkudG9FcXVhbCgyNCk7XG4gIH0pO1xuXG4gIGl0KCdDb3JyZWN0bHkgaWRlbnRpZmllcyBzY2FsYXJzJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmlzU2NhbGFyU2hhcGUoW10pKS50b0JlKHRydWUpO1xuICAgIGV4cGVjdCh1dGlsLmlzU2NhbGFyU2hhcGUoWzEsIDJdKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaXNTY2FsYXJTaGFwZShbMV0pKS50b0JlKGZhbHNlKTtcbiAgfSk7XG5cbiAgaXQoJ051bWJlciBhcnJheXMgZXF1YWwnLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuYXJyYXlzRXF1YWwoWzEsIDIsIDMsIDZdLCBbMSwgMiwgMywgNl0pKS50b0JlKHRydWUpO1xuICAgIGV4cGVjdCh1dGlsLmFycmF5c0VxdWFsKFsxLCAyXSwgWzEsIDIsIDNdKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuYXJyYXlzRXF1YWwoWzEsIDIsIDVdLCBbMSwgMl0pKS50b0JlKGZhbHNlKTtcbiAgfSk7XG5cbiAgaXQoJ0FycmF5cyBzaHVmZmxlIHJhbmRvbWx5JywgKCkgPT4ge1xuICAgIC8vIENyZWF0ZSAxMDAwIG51bWJlcnMgb3JkZXJlZFxuICAgIGNvbnN0IGEgPSBBcnJheS5hcHBseSgwLCB7bGVuZ3RoOiAxMDAwfSkubWFwKE51bWJlci5jYWxsLCBOdW1iZXIpLnNsaWNlKDEpO1xuICAgIGNvbnN0IGIgPSBbXS5jb25jYXQoYSk7ICAvLyBjb3B5IEVTNSBzdHlsZVxuICAgIHV0aWwuc2h1ZmZsZShhKTtcbiAgICBleHBlY3QoYSkubm90LnRvRXF1YWwoYik7XG4gICAgZXhwZWN0KGEubGVuZ3RoKS50b0VxdWFsKGIubGVuZ3RoKTtcbiAgfSk7XG5cbiAgaXQoJ011bHRpcGxlIGFycmF5cyBzaHVmZmxlIHRvZ2V0aGVyJywgKCkgPT4ge1xuICAgIC8vIENyZWF0ZSAxMDAwIG51bWJlcnMgb3JkZXJlZFxuICAgIGNvbnN0IGEgPSBBcnJheS5hcHBseSgwLCB7bGVuZ3RoOiAxMDAwfSkubWFwKE51bWJlci5jYWxsLCBOdW1iZXIpLnNsaWNlKDEpO1xuICAgIGNvbnN0IGIgPSBbXS5jb25jYXQoYSk7ICAvLyBjb3BpZXNcbiAgICBjb25zdCBjID0gW10uY29uY2F0KGEpO1xuICAgIHV0aWwuc2h1ZmZsZUNvbWJvKGEsIGIpO1xuICAgIGV4cGVjdChhKS5ub3QudG9FcXVhbChjKTtcbiAgICBleHBlY3QoYSkudG9FcXVhbChiKTtcbiAgICBleHBlY3QoYS5sZW5ndGgpLnRvRXF1YWwoYy5sZW5ndGgpO1xuICB9KTtcblxuICBpdCgnSXMgaW50ZWdlcicsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5pc0ludCgwLjUpKS50b0JlKGZhbHNlKTtcbiAgICBleHBlY3QodXRpbC5pc0ludCgxKSkudG9CZSh0cnVlKTtcbiAgfSk7XG5cbiAgaXQoJ1NpemUgdG8gc3F1YXJpc2ggc2hhcGUgKHBlcmZlY3Qgc3F1YXJlKScsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5zaXplVG9TcXVhcmlzaFNoYXBlKDkpKS50b0VxdWFsKFszLCAzXSk7XG4gIH0pO1xuXG4gIGl0KCdTaXplIHRvIHNxdWFyaXNoIHNoYXBlIChwcmltZSBudW1iZXIpJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLnNpemVUb1NxdWFyaXNoU2hhcGUoMTEpKS50b0VxdWFsKFs0LCAzXSk7XG4gIH0pO1xuXG4gIGl0KCdTaXplIHRvIHNxdWFyaXNoIHNoYXBlIChhbG1vc3Qgc3F1YXJlKScsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5zaXplVG9TcXVhcmlzaFNoYXBlKDM1KSkudG9FcXVhbChbNiwgNl0pO1xuICB9KTtcblxuICBpdCgnU2l6ZSBvZiAxIHRvIHNxdWFyaXNoIHNoYXBlJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLnNpemVUb1NxdWFyaXNoU2hhcGUoMSkpLnRvRXF1YWwoWzEsIDFdKTtcbiAgfSk7XG5cbiAgaXQoJ2luZmVyIHNoYXBlIHNpbmdsZSBudW1iZXInLCAoKSA9PiB7XG4gICAgZXhwZWN0KGluZmVyU2hhcGUoNCkpLnRvRXF1YWwoW10pO1xuICB9KTtcblxuICBpdCgnaW5mZXIgc2hhcGUgMWQgYXJyYXknLCAoKSA9PiB7XG4gICAgZXhwZWN0KGluZmVyU2hhcGUoWzEsIDIsIDVdKSkudG9FcXVhbChbM10pO1xuICB9KTtcblxuICBpdCgnaW5mZXIgc2hhcGUgMmQgYXJyYXknLCAoKSA9PiB7XG4gICAgZXhwZWN0KGluZmVyU2hhcGUoW1sxLCAyLCA1XSwgWzUsIDQsIDFdXSkpLnRvRXF1YWwoWzIsIDNdKTtcbiAgfSk7XG5cbiAgaXQoJ2luZmVyIHNoYXBlIDNkIGFycmF5JywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBbW1sxLCAyXSwgWzIsIDNdLCBbNSwgNl1dLCBbWzUsIDZdLCBbNCwgNV0sIFsxLCAyXV1dO1xuICAgIGV4cGVjdChpbmZlclNoYXBlKGEpKS50b0VxdWFsKFsyLCAzLCAyXSk7XG4gIH0pO1xuXG4gIGl0KCdpbmZlciBzaGFwZSA0ZCBhcnJheScsICgpID0+IHtcbiAgICBjb25zdCBhID0gW1xuICAgICAgW1tbMV0sIFsyXV0sIFtbMl0sIFszXV0sIFtbNV0sIFs2XV1dLCBbW1s1XSwgWzZdXSwgW1s0XSwgWzVdXSwgW1sxXSwgWzJdXV1cbiAgICBdO1xuICAgIGV4cGVjdChpbmZlclNoYXBlKGEpKS50b0VxdWFsKFsyLCAzLCAyLCAxXSk7XG4gIH0pO1xuXG4gIGl0KCdpbmZlciBzaGFwZSBvZiB0eXBlZCBhcnJheScsICgpID0+IHtcbiAgICBjb25zdCBhID0gbmV3IEZsb2F0MzJBcnJheShbMSwgMiwgMywgNCwgNV0pO1xuICAgIGV4cGVjdChpbmZlclNoYXBlKGEpKS50b0VxdWFsKFs1XSk7XG4gIH0pO1xuXG4gIGl0KCdpbmZlciBzaGFwZSBvZiBjbGFtcGVkIHR5cGVkIGFycmF5JywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBuZXcgVWludDhDbGFtcGVkQXJyYXkoWzEsIDIsIDMsIDQsIDVdKTtcbiAgICBleHBlY3QoaW5mZXJTaGFwZShhKSkudG9FcXVhbChbNV0pO1xuICB9KTtcblxuICBpdCgnaW5mZXIgc2hhcGUgb2YgVWludDhBcnJheVtdLCBzdHJpbmcgdGVuc29yJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBbbmV3IFVpbnQ4QXJyYXkoWzEsIDJdKSwgbmV3IFVpbnQ4QXJyYXkoWzMsIDRdKV07XG4gICAgZXhwZWN0KGluZmVyU2hhcGUoYSwgJ3N0cmluZycpKS50b0VxdWFsKFsyXSk7XG4gIH0pO1xuXG4gIGl0KCdpbmZlciBzaGFwZSBvZiBVaW50OEFycmF5W11bXSwgc3RyaW5nIHRlbnNvcicsICgpID0+IHtcbiAgICBjb25zdCBhID0gW1xuICAgICAgW25ldyBVaW50OEFycmF5KFsxXSksIG5ldyBVaW50OEFycmF5KFsyXSldLFxuICAgICAgW25ldyBVaW50OEFycmF5KFsxXSksIG5ldyBVaW50OEFycmF5KFsyXSldXG4gICAgXTtcbiAgICBleHBlY3QoaW5mZXJTaGFwZShhLCAnc3RyaW5nJykpLnRvRXF1YWwoWzIsIDJdKTtcbiAgfSk7XG5cbiAgaXQoJ2luZmVyIHNoYXBlIG9mIFVpbnQ4QXJyYXlbXVtdW10sIHN0cmluZyB0ZW5zb3InLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IFtcbiAgICAgIFtbbmV3IFVpbnQ4QXJyYXkoWzEsIDJdKV0sIFtuZXcgVWludDhBcnJheShbMiwgMV0pXV0sXG4gICAgICBbW25ldyBVaW50OEFycmF5KFsxLCAyXSldLCBbbmV3IFVpbnQ4QXJyYXkoWzIsIDFdKV1dXG4gICAgXTtcbiAgICBleHBlY3QoaW5mZXJTaGFwZShhLCAnc3RyaW5nJykpLnRvRXF1YWwoWzIsIDIsIDFdKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmUoJ3V0aWwuZmxhdHRlbicsICgpID0+IHtcbiAgaXQoJ25lc3RlZCBudW1iZXIgYXJyYXlzJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmZsYXR0ZW4oW1sxLCAyLCAzXSwgWzQsIDUsIDZdXSkpLnRvRXF1YWwoWzEsIDIsIDMsIDQsIDUsIDZdKTtcbiAgICBleHBlY3QodXRpbC5mbGF0dGVuKFtbWzEsIDJdLCBbMywgNF0sIFs1LCA2XSwgWzcsIDhdXV0pKS50b0VxdWFsKFtcbiAgICAgIDEsIDIsIDMsIDQsIDUsIDYsIDcsIDhcbiAgICBdKTtcbiAgICBleHBlY3QodXRpbC5mbGF0dGVuKFsxLCAyLCAzLCA0LCA1LCA2XSkpLnRvRXF1YWwoWzEsIDIsIDMsIDQsIDUsIDZdKTtcbiAgfSk7XG5cbiAgaXQoJ25lc3RlZCBzdHJpbmcgYXJyYXlzJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmZsYXR0ZW4oW1snYScsICdiJ10sIFsnYycsIFtbJ2QnXV1dXSkpLnRvRXF1YWwoW1xuICAgICAgJ2EnLCAnYicsICdjJywgJ2QnXG4gICAgXSk7XG4gICAgZXhwZWN0KHV0aWwuZmxhdHRlbihbWydhJywgWydiJ11dLCBbJ2MnLCBbWydkJ11dLCAnZSddXSkpLnRvRXF1YWwoW1xuICAgICAgJ2EnLCAnYicsICdjJywgJ2QnLCAnZSdcbiAgICBdKTtcbiAgfSk7XG5cbiAgaXQoJ21peGVkIFR5cGVkQXJyYXkgYW5kIG51bWJlcltdJywgKCkgPT4ge1xuICAgIGNvbnN0IGRhdGEgPVxuICAgICAgICBbbmV3IEZsb2F0MzJBcnJheShbMSwgMl0pLCAzLCBbNCwgNSwgbmV3IEZsb2F0MzJBcnJheShbNiwgN10pXV07XG4gICAgZXhwZWN0KHV0aWwuZmxhdHRlbihkYXRhKSkudG9FcXVhbChbMSwgMiwgMywgNCwgNSwgNiwgN10pO1xuICB9KTtcblxuICBpdCgnbmVzdGVkIFVpbnQ4QXJyYXlzLCBza2lwVHlwZWRBcnJheT10cnVlJywgKCkgPT4ge1xuICAgIGNvbnN0IGRhdGEgPSBbXG4gICAgICBbbmV3IFVpbnQ4QXJyYXkoWzEsIDJdKSwgbmV3IFVpbnQ4QXJyYXkoWzMsIDRdKV0sXG4gICAgICBbbmV3IFVpbnQ4QXJyYXkoWzUsIDZdKSwgbmV3IFVpbnQ4QXJyYXkoWzcsIDhdKV1cbiAgICBdO1xuICAgIGV4cGVjdCh1dGlsLmZsYXR0ZW4oZGF0YSwgW10sIHRydWUpKS50b0VxdWFsKFtcbiAgICAgIG5ldyBVaW50OEFycmF5KFsxLCAyXSksIG5ldyBVaW50OEFycmF5KFszLCA0XSksIG5ldyBVaW50OEFycmF5KFs1LCA2XSksXG4gICAgICBuZXcgVWludDhBcnJheShbNywgOF0pXG4gICAgXSk7XG4gIH0pO1xufSk7XG5cbmZ1bmN0aW9uIGVuY29kZVN0cmluZ3MoYTogc3RyaW5nW10pOiBVaW50OEFycmF5W10ge1xuICByZXR1cm4gYS5tYXAocyA9PiB1dGlsLmVuY29kZVN0cmluZyhzKSk7XG59XG5cbmRlc2NyaWJlKCd1dGlsLmJ5dGVzRnJvbVN0cmluZ0FycmF5JywgKCkgPT4ge1xuICBpdCgnY291bnQgYnl0ZXMgYWZ0ZXIgdXRmOCBlbmNvZGluZycsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5ieXRlc0Zyb21TdHJpbmdBcnJheShlbmNvZGVTdHJpbmdzKFsnYScsICdiYicsICdjY2MnXSkpKVxuICAgICAgICAudG9CZSg2KTtcbiAgICBleHBlY3QodXRpbC5ieXRlc0Zyb21TdHJpbmdBcnJheShlbmNvZGVTdHJpbmdzKFsnYScsICdiYicsICdjY2NkZGQnXSkpKVxuICAgICAgICAudG9CZSg5KTtcbiAgICBleHBlY3QodXRpbC5ieXRlc0Zyb21TdHJpbmdBcnJheShlbmNvZGVTdHJpbmdzKFsn0LTQsNC90LjQtdC7J10pKSkudG9CZSg2ICogMik7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlKCd1dGlsLmluZmVyRHR5cGUnLCAoKSA9PiB7XG4gIGl0KCdhIHNpbmdsZSBzdHJpbmcgPT4gc3RyaW5nJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmluZmVyRHR5cGUoJ2hlbGxvJykpLnRvQmUoJ3N0cmluZycpO1xuICB9KTtcblxuICBpdCgnYSBzaW5nbGUgYm9vbGVhbiA9PiBib29sJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmluZmVyRHR5cGUodHJ1ZSkpLnRvQmUoJ2Jvb2wnKTtcbiAgICBleHBlY3QodXRpbC5pbmZlckR0eXBlKGZhbHNlKSkudG9CZSgnYm9vbCcpO1xuICB9KTtcblxuICBpdCgnYSBzaW5nbGUgbnVtYmVyID0+IGZsb2F0MzInLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuaW5mZXJEdHlwZSgwKSkudG9CZSgnZmxvYXQzMicpO1xuICAgIGV4cGVjdCh1dGlsLmluZmVyRHR5cGUoMzQpKS50b0JlKCdmbG9hdDMyJyk7XG4gIH0pO1xuXG4gIGl0KCdhIGxpc3Qgb2Ygc3RyaW5ncyA9PiBzdHJpbmcnLCAoKSA9PiB7XG4gICAgLy8gRmxhdC5cbiAgICBleHBlY3QodXRpbC5pbmZlckR0eXBlKFsnYScsICdiJywgJ2MnXSkpLnRvQmUoJ3N0cmluZycpO1xuICAgIC8vIE5lc3RlZC5cbiAgICBleHBlY3QodXRpbC5pbmZlckR0eXBlKFtcbiAgICAgIFtbJ2EnXV0sIFtbJ2InXV0sIFtbJ2MnXV0sIFtbJ2QnXV1cbiAgICBdKSkudG9CZSgnc3RyaW5nJyk7XG4gIH0pO1xuXG4gIGl0KCdhIGxpc3Qgb2YgYm9vbHMgPT4gZmxvYXQzMicsICgpID0+IHtcbiAgICAvLyBGbGF0LlxuICAgIGV4cGVjdCh1dGlsLmluZmVyRHR5cGUoW2ZhbHNlLCB0cnVlLCBmYWxzZV0pKS50b0JlKCdib29sJyk7XG4gICAgLy8gTmVzdGVkLlxuICAgIGV4cGVjdCh1dGlsLmluZmVyRHR5cGUoW1xuICAgICAgW1t0cnVlXV0sIFtbZmFsc2VdXSwgW1t0cnVlXV0sIFtbdHJ1ZV1dXG4gICAgXSkpLnRvQmUoJ2Jvb2wnKTtcbiAgfSk7XG5cbiAgaXQoJ2EgbGlzdCBvZiBudW1iZXJzID0+IGZsb2F0MzInLCAoKSA9PiB7XG4gICAgLy8gRmxhdC5cbiAgICBleHBlY3QodXRpbC5pbmZlckR0eXBlKFswLCAxLCAyXSkpLnRvQmUoJ2Zsb2F0MzInKTtcbiAgICAvLyBOZXN0ZWQuXG4gICAgZXhwZWN0KHV0aWwuaW5mZXJEdHlwZShbW1swXV0sIFtbMV1dLCBbWzJdXSwgW1szXV1dKSkudG9CZSgnZmxvYXQzMicpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZSgndXRpbC5yZXBlYXRlZFRyeScsICgpID0+IHtcbiAgaXQoJ3Jlc29sdmVzJywgKGRvbmVGbikgPT4ge1xuICAgIGxldCBjb3VudGVyID0gMDtcbiAgICBjb25zdCBjaGVja0ZuID0gKCkgPT4ge1xuICAgICAgY291bnRlcisrO1xuICAgICAgaWYgKGNvdW50ZXIgPT09IDIpIHtcbiAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICB9XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfTtcblxuICAgIHV0aWwucmVwZWF0ZWRUcnkoY2hlY2tGbikudGhlbihkb25lRm4pLmNhdGNoKCgpID0+IHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignUmVqZWN0ZWQgYmFja29mZi4nKTtcbiAgICB9KTtcbiAgfSk7XG4gIGl0KCdyZWplY3RzJywgKGRvbmVGbikgPT4ge1xuICAgIGNvbnN0IGNoZWNrRm4gPSAoKSA9PiBmYWxzZTtcblxuICAgIHV0aWwucmVwZWF0ZWRUcnkoY2hlY2tGbiwgKCkgPT4gMCwgNSlcbiAgICAgICAgLnRoZW4oKCkgPT4ge1xuICAgICAgICAgIHRocm93IG5ldyBFcnJvcignQmFja29mZiByZXNvbHZlZCcpO1xuICAgICAgICB9KVxuICAgICAgICAuY2F0Y2goZG9uZUZuKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmUoJ3V0aWwuaW5mZXJGcm9tSW1wbGljaXRTaGFwZScsICgpID0+IHtcbiAgaXQoJ2VtcHR5IHNoYXBlJywgKCkgPT4ge1xuICAgIGNvbnN0IHJlc3VsdCA9IHV0aWwuaW5mZXJGcm9tSW1wbGljaXRTaGFwZShbXSwgMCk7XG4gICAgZXhwZWN0KHJlc3VsdCkudG9FcXVhbChbXSk7XG4gIH0pO1xuXG4gIGl0KCdbMiwgMywgNF0gLT4gWzIsIDMsIDRdJywgKCkgPT4ge1xuICAgIGNvbnN0IHJlc3VsdCA9IHV0aWwuaW5mZXJGcm9tSW1wbGljaXRTaGFwZShbMiwgMywgNF0sIDI0KTtcbiAgICBleHBlY3QocmVzdWx0KS50b0VxdWFsKFsyLCAzLCA0XSk7XG4gIH0pO1xuXG4gIGl0KCdbMiwgLTEsIDRdIC0+IFsyLCAzLCA0XSwgc2l6ZT0yNCcsICgpID0+IHtcbiAgICBjb25zdCByZXN1bHQgPSB1dGlsLmluZmVyRnJvbUltcGxpY2l0U2hhcGUoWzIsIC0xLCA0XSwgMjQpO1xuICAgIGV4cGVjdChyZXN1bHQpLnRvRXF1YWwoWzIsIDMsIDRdKTtcbiAgfSk7XG5cbiAgaXQoJ1stMSwgMywgNF0gLT4gWzIsIDMsIDRdLCBzaXplPTI0JywgKCkgPT4ge1xuICAgIGNvbnN0IHJlc3VsdCA9IHV0aWwuaW5mZXJGcm9tSW1wbGljaXRTaGFwZShbLTEsIDMsIDRdLCAyNCk7XG4gICAgZXhwZWN0KHJlc3VsdCkudG9FcXVhbChbMiwgMywgNF0pO1xuICB9KTtcblxuICBpdCgnWzIsIDMsIC0xXSAtPiBbMiwgMywgNF0sIHNpemU9MjQnLCAoKSA9PiB7XG4gICAgY29uc3QgcmVzdWx0ID0gdXRpbC5pbmZlckZyb21JbXBsaWNpdFNoYXBlKFsyLCAzLCAtMV0sIDI0KTtcbiAgICBleHBlY3QocmVzdWx0KS50b0VxdWFsKFsyLCAzLCA0XSk7XG4gIH0pO1xuXG4gIGl0KCdbMiwgLTEsIC0xXSB0aHJvd3MgZXJyb3InLCAoKSA9PiB7XG4gICAgZXhwZWN0KCgpID0+IHV0aWwuaW5mZXJGcm9tSW1wbGljaXRTaGFwZShbMiwgLTEsIC0xXSwgMjQpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ1syLCAzLCAtMV0gc2l6ZT0xMyB0aHJvd3MgZXJyb3InLCAoKSA9PiB7XG4gICAgZXhwZWN0KCgpID0+IHV0aWwuaW5mZXJGcm9tSW1wbGljaXRTaGFwZShbMiwgMywgLTFdLCAxMykpLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcblxuICBpdCgnWzIsIDMsIDRdIHNpemU9MjUgKHNob3VsZCBiZSAyNCkgdGhyb3dzIGVycm9yJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLmluZmVyRnJvbUltcGxpY2l0U2hhcGUoWzIsIDMsIDRdLCAyNSkpLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZSgndXRpbCBwYXJzZUF4aXNQYXJhbScsICgpID0+IHtcbiAgaXQoJ2F4aXM9bnVsbCByZXR1cm5zIG5vIGF4ZXMgZm9yIHNjYWxhcicsICgpID0+IHtcbiAgICBjb25zdCBheGlzOiBudW1iZXIgPSBudWxsO1xuICAgIGNvbnN0IHNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICAgIGV4cGVjdCh1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHNoYXBlKSkudG9FcXVhbChbXSk7XG4gIH0pO1xuXG4gIGl0KCdheGlzPW51bGwgcmV0dXJucyAwIGF4aXMgZm9yIFRlbnNvcjFEJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXM6IG51bWJlciA9IG51bGw7XG4gICAgY29uc3Qgc2hhcGUgPSBbNF07XG4gICAgZXhwZWN0KHV0aWwucGFyc2VBeGlzUGFyYW0oYXhpcywgc2hhcGUpKS50b0VxdWFsKFswXSk7XG4gIH0pO1xuXG4gIGl0KCdheGlzPW51bGwgcmV0dXJucyBhbGwgYXhlcyBmb3IgVGVuc29yM0QnLCAoKSA9PiB7XG4gICAgY29uc3QgYXhpczogbnVtYmVyW10gPSBudWxsO1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDEsIDJdO1xuICAgIGV4cGVjdCh1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHNoYXBlKSkudG9FcXVhbChbMCwgMSwgMl0pO1xuICB9KTtcblxuICBpdCgnYXhpcyBhcyBhIHNpbmdsZSBudW1iZXInLCAoKSA9PiB7XG4gICAgY29uc3QgYXhpcyA9IDE7XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgMSwgMl07XG4gICAgZXhwZWN0KHV0aWwucGFyc2VBeGlzUGFyYW0oYXhpcywgc2hhcGUpKS50b0VxdWFsKFsxXSk7XG4gIH0pO1xuXG4gIGl0KCdheGlzIGFzIHNpbmdsZSBuZWdhdGl2ZSBudW1iZXInLCAoKSA9PiB7XG4gICAgY29uc3QgYXhpcyA9IC0xO1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDEsIDJdO1xuICAgIGV4cGVjdCh1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHNoYXBlKSkudG9FcXVhbChbMl0pO1xuXG4gICAgY29uc3QgYXhpczIgPSAtMjtcbiAgICBleHBlY3QodXRpbC5wYXJzZUF4aXNQYXJhbShheGlzMiwgc2hhcGUpKS50b0VxdWFsKFsxXSk7XG5cbiAgICBjb25zdCBheGlzMyA9IC0zO1xuICAgIGV4cGVjdCh1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMzLCBzaGFwZSkpLnRvRXF1YWwoWzBdKTtcbiAgfSk7XG5cbiAgaXQoJ2F4aXMgYXMgbGlzdCBvZiBuZWdhdGl2ZSBudW1iZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXMgPSBbLTEsIC0zXTtcbiAgICBjb25zdCBzaGFwZSA9IFszLCAxLCAyXTtcbiAgICBleHBlY3QodXRpbC5wYXJzZUF4aXNQYXJhbShheGlzLCBzaGFwZSkpLnRvRXF1YWwoWzIsIDBdKTtcbiAgfSk7XG5cbiAgaXQoJ2F4aXMgYXMgbGlzdCBvZiBwb3NpdGl2ZSBudW1iZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXMgPSBbMCwgMl07XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgMSwgMl07XG4gICAgZXhwZWN0KHV0aWwucGFyc2VBeGlzUGFyYW0oYXhpcywgc2hhcGUpKS50b0VxdWFsKFswLCAyXSk7XG4gIH0pO1xuXG4gIGl0KCdheGlzIGFzIGNvbWJvIG9mIHBvc2l0aXZlIGFuZCBuZWdhdGl2ZSBudW1iZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXMgPSBbMCwgLTFdO1xuICAgIGNvbnN0IHNoYXBlID0gWzMsIDEsIDJdO1xuICAgIGV4cGVjdCh1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMsIHNoYXBlKSkudG9FcXVhbChbMCwgMl0pO1xuICB9KTtcblxuICBpdCgnYXhpcyBvdXQgb2YgcmFuZ2UgdGhyb3dzIGVycm9yJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXMgPSAtNDtcbiAgICBjb25zdCBzaGFwZSA9IFszLCAxLCAyXTtcbiAgICBleHBlY3QoKCkgPT4gdXRpbC5wYXJzZUF4aXNQYXJhbShheGlzLCBzaGFwZSkpLnRvVGhyb3dFcnJvcigpO1xuXG4gICAgY29uc3QgYXhpczIgPSA0O1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLnBhcnNlQXhpc1BhcmFtKGF4aXMyLCBzaGFwZSkpLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcblxuICBpdCgnYXhpcyBhIGxpc3Qgd2l0aCBvbmUgbnVtYmVyIG91dCBvZiByYW5nZSB0aHJvd3MgZXJyb3InLCAoKSA9PiB7XG4gICAgY29uc3QgYXhpcyA9IFswLCA0XTtcbiAgICBjb25zdCBzaGFwZSA9IFszLCAxLCAyXTtcbiAgICBleHBlY3QoKCkgPT4gdXRpbC5wYXJzZUF4aXNQYXJhbShheGlzLCBzaGFwZSkpLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcblxuICBpdCgnYXhpcyB3aXRoIGRlY2ltYWwgdmFsdWUgdGhyb3dzIGVycm9yJywgKCkgPT4ge1xuICAgIGNvbnN0IGF4aXMgPSAwLjU7XG4gICAgY29uc3Qgc2hhcGUgPSBbMywgMSwgMl07XG4gICAgZXhwZWN0KCgpID0+IHV0aWwucGFyc2VBeGlzUGFyYW0oYXhpcywgc2hhcGUpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmUoJ3V0aWwuc3F1ZWV6ZVNoYXBlJywgKCkgPT4ge1xuICBpdCgnc2NhbGFyJywgKCkgPT4ge1xuICAgIGNvbnN0IHtuZXdTaGFwZSwga2VwdERpbXN9ID0gdXRpbC5zcXVlZXplU2hhcGUoW10pO1xuICAgIGV4cGVjdChuZXdTaGFwZSkudG9FcXVhbChbXSk7XG4gICAgZXhwZWN0KGtlcHREaW1zKS50b0VxdWFsKFtdKTtcbiAgfSk7XG5cbiAgaXQoJzF4MSByZWR1Y2VkIHRvIHNjYWxhcicsICgpID0+IHtcbiAgICBjb25zdCB7bmV3U2hhcGUsIGtlcHREaW1zfSA9IHV0aWwuc3F1ZWV6ZVNoYXBlKFsxLCAxXSk7XG4gICAgZXhwZWN0KG5ld1NoYXBlKS50b0VxdWFsKFtdKTtcbiAgICBleHBlY3Qoa2VwdERpbXMpLnRvRXF1YWwoW10pO1xuICB9KTtcblxuICBpdCgnMXgzeDEgcmVkdWNlZCB0byBbM10nLCAoKSA9PiB7XG4gICAgY29uc3Qge25ld1NoYXBlLCBrZXB0RGltc30gPSB1dGlsLnNxdWVlemVTaGFwZShbMSwgMywgMV0pO1xuICAgIGV4cGVjdChuZXdTaGFwZSkudG9FcXVhbChbM10pO1xuICAgIGV4cGVjdChrZXB0RGltcykudG9FcXVhbChbMV0pO1xuICB9KTtcblxuICBpdCgnMXgxeDQgcmVkdWNlZCB0byBbNF0nLCAoKSA9PiB7XG4gICAgY29uc3Qge25ld1NoYXBlLCBrZXB0RGltc30gPSB1dGlsLnNxdWVlemVTaGFwZShbMSwgMSwgNF0pO1xuICAgIGV4cGVjdChuZXdTaGFwZSkudG9FcXVhbChbNF0pO1xuICAgIGV4cGVjdChrZXB0RGltcykudG9FcXVhbChbMl0pO1xuICB9KTtcblxuICBpdCgnMngzeDQgbm90IHJlZHVjdGlvbicsICgpID0+IHtcbiAgICBjb25zdCB7bmV3U2hhcGUsIGtlcHREaW1zfSA9IHV0aWwuc3F1ZWV6ZVNoYXBlKFsyLCAzLCA0XSk7XG4gICAgZXhwZWN0KG5ld1NoYXBlKS50b0VxdWFsKFsyLCAzLCA0XSk7XG4gICAgZXhwZWN0KGtlcHREaW1zKS50b0VxdWFsKFswLCAxLCAyXSk7XG4gIH0pO1xuXG4gIGRlc2NyaWJlKCd3aXRoIGF4aXMnLCAoKSA9PiB7XG4gICAgaXQoJ3Nob3VsZCBvbmx5IHJlZHVjZSBkaW1lbnNpb25zIHNwZWNpZmllZCBieSBheGlzJywgKCkgPT4ge1xuICAgICAgY29uc3Qge25ld1NoYXBlLCBrZXB0RGltc30gPSB1dGlsLnNxdWVlemVTaGFwZShbMSwgMSwgMSwgMSwgNF0sIFsxLCAyXSk7XG4gICAgICBleHBlY3QobmV3U2hhcGUpLnRvRXF1YWwoWzEsIDEsIDRdKTtcbiAgICAgIGV4cGVjdChrZXB0RGltcykudG9FcXVhbChbMCwgMywgNF0pO1xuICAgIH0pO1xuICAgIGl0KCdzaG91bGQgb25seSByZWR1Y2UgZGltZW5zaW9ucyBzcGVjaWZpZWQgYnkgbmVnYXRpdmUgYXhpcycsICgpID0+IHtcbiAgICAgIGNvbnN0IHtuZXdTaGFwZSwga2VwdERpbXN9ID0gdXRpbC5zcXVlZXplU2hhcGUoWzEsIDEsIDEsIDEsIDRdLCBbLTIsIC0zXSk7XG4gICAgICBleHBlY3QobmV3U2hhcGUpLnRvRXF1YWwoWzEsIDEsIDRdKTtcbiAgICAgIGV4cGVjdChrZXB0RGltcykudG9FcXVhbChbMCwgMSwgNF0pO1xuICAgIH0pO1xuICAgIGl0KCdzaG91bGQgb25seSByZWR1Y2UgZGltZW5zaW9ucyBzcGVjaWZpZWQgYnkgbmVnYXRpdmUgYXhpcycsICgpID0+IHtcbiAgICAgIGNvbnN0IGF4aXMgPSBbLTIsIC0zXTtcbiAgICAgIHV0aWwuc3F1ZWV6ZVNoYXBlKFsxLCAxLCAxLCAxLCA0XSwgYXhpcyk7XG4gICAgICBleHBlY3QoYXhpcykudG9FcXVhbChbLTIsIC0zXSk7XG4gICAgfSk7XG4gICAgaXQoJ3Rocm93cyBlcnJvciB3aGVuIHNwZWNpZmllZCBheGlzIGlzIG5vdCBzcXVlZXphYmxlJywgKCkgPT4ge1xuICAgICAgZXhwZWN0KCgpID0+IHV0aWwuc3F1ZWV6ZVNoYXBlKFsxLCAxLCAyLCAxLCA0XSwgWzEsIDJdKSkudG9UaHJvd0Vycm9yKCk7XG4gICAgfSk7XG4gICAgaXQoJ3Rocm93cyBlcnJvciB3aGVuIHNwZWNpZmllZCBuZWdhdGl2ZSBheGlzIGlzIG5vdCBzcXVlZXphYmxlJywgKCkgPT4ge1xuICAgICAgZXhwZWN0KCgpID0+IHV0aWwuc3F1ZWV6ZVNoYXBlKFsxLCAxLCAyLCAxLCA0XSwgWy0xLCAtMl0pKS50b1Rocm93RXJyb3IoKTtcbiAgICB9KTtcbiAgICBpdCgndGhyb3dzIGVycm9yIHdoZW4gc3BlY2lmaWVkIGF4aXMgaXMgb3V0IG9mIHJhbmdlJywgKCkgPT4ge1xuICAgICAgZXhwZWN0KCgpID0+IHV0aWwuc3F1ZWV6ZVNoYXBlKFsxLCAxLCAyLCAxLCA0XSwgWzExLCAyMl0pKS50b1Rocm93RXJyb3IoKTtcbiAgICB9KTtcbiAgICBpdCgndGhyb3dzIGVycm9yIHdoZW4gc3BlY2lmaWVkIG5lZ2F0aXZlIGF4aXMgaXMgb3V0IG9mIHJhbmdlJywgKCkgPT4ge1xuICAgICAgZXhwZWN0KCgpID0+IHV0aWwuc3F1ZWV6ZVNoYXBlKFsxLCAxLCAyLCAxLCA0XSwgW1xuICAgICAgICAtMTEsIC0yMlxuICAgICAgXSkpLnRvVGhyb3dFcnJvcigpO1xuICAgIH0pO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZSgndXRpbC5jaGVja0NvbnZlcnNpb25Gb3JFcnJvcnMnLCAoKSA9PiB7XG4gIGl0KCdGbG9hdDMyQXJyYXkgaGFzIE5hTicsICgpID0+IHtcbiAgICBleHBlY3QoXG4gICAgICAgICgpID0+IHV0aWwuY2hlY2tDb252ZXJzaW9uRm9yRXJyb3JzKFxuICAgICAgICAgICAgbmV3IEZsb2F0MzJBcnJheShbMSwgMiwgMywgTmFOLCA0LCAyNTVdKSwgJ2Zsb2F0MzInKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcblxuICBpdCgnRmxvYXQzMkFycmF5IGhhcyBJbmZpbml0eScsICgpID0+IHtcbiAgICBleHBlY3QoXG4gICAgICAgICgpID0+IHV0aWwuY2hlY2tDb252ZXJzaW9uRm9yRXJyb3JzKFxuICAgICAgICAgICAgbmV3IEZsb2F0MzJBcnJheShbMSwgMiwgMywgSW5maW5pdHksIDQsIDI1NV0pLCAnZmxvYXQzMicpKVxuICAgICAgICAudG9UaHJvd0Vycm9yKCk7XG4gIH0pO1xuXG4gIGl0KCdJbnQzMkFycmF5IGhhcyBOYU4nLCAoKSA9PiB7XG4gICAgZXhwZWN0KCgpID0+IHV0aWwuY2hlY2tDb252ZXJzaW9uRm9yRXJyb3JzKFsxLCAyLCAzLCA0LCBOYU5dLCAnaW50MzInKSlcbiAgICAgICAgLnRvVGhyb3dFcnJvcigpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZSgndXRpbC5oYXNFbmNvZGluZ0xvc3MnLCAoKSA9PiB7XG4gIGl0KCdjb21wbGV4NjQgdG8gYW55JywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnY29tcGxleDY0JywgJ2NvbXBsZXg2NCcpKS50b0JlKGZhbHNlKTtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2NvbXBsZXg2NCcsICdmbG9hdDMyJykpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdjb21wbGV4NjQnLCAnaW50MzInKSkudG9CZSh0cnVlKTtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2NvbXBsZXg2NCcsICdib29sJykpLnRvQmUodHJ1ZSk7XG4gIH0pO1xuXG4gIGl0KCdhbnkgdG8gY29tcGxleDY0JywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnYm9vbCcsICdjb21wbGV4NjQnKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdpbnQzMicsICdjb21wbGV4NjQnKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdmbG9hdDMyJywgJ2NvbXBsZXg2NCcpKS50b0JlKGZhbHNlKTtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2NvbXBsZXg2NCcsICdjb21wbGV4NjQnKSkudG9CZShmYWxzZSk7XG4gIH0pO1xuXG4gIGl0KCdhbnkgdG8gZmxvYXQzMicsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2Jvb2wnLCAnZmxvYXQzMicpKS50b0JlKGZhbHNlKTtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2ludDMyJywgJ2Zsb2F0MzInKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdmbG9hdDMyJywgJ2Zsb2F0MzInKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdjb21wbGV4NjQnLCAnZmxvYXQzMicpKS50b0JlKHRydWUpO1xuICB9KTtcblxuICBpdCgnZmxvYXQzMiB0byBhbnknLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdmbG9hdDMyJywgJ2Zsb2F0MzInKSkudG9CZShmYWxzZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdmbG9hdDMyJywgJ2ludDMyJykpLnRvQmUodHJ1ZSk7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdmbG9hdDMyJywgJ2Jvb2wnKSkudG9CZSh0cnVlKTtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2Zsb2F0MzInLCAnY29tcGxleDY0JykpLnRvQmUoZmFsc2UpO1xuICB9KTtcblxuICBpdCgnaW50MzIgdG8gbG93ZXInLCAoKSA9PiB7XG4gICAgZXhwZWN0KHV0aWwuaGFzRW5jb2RpbmdMb3NzKCdpbnQzMicsICdpbnQzMicpKS50b0JlKGZhbHNlKTtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2ludDMyJywgJ2Jvb2wnKSkudG9CZSh0cnVlKTtcbiAgfSk7XG5cbiAgaXQoJ2xvd2VyIHRvIGludDMyJywgKCkgPT4ge1xuICAgIGV4cGVjdCh1dGlsLmhhc0VuY29kaW5nTG9zcygnYm9vbCcsICdpbnQzMicpKS50b0JlKGZhbHNlKTtcbiAgfSk7XG5cbiAgaXQoJ2Jvb2wgdG8gYm9vbCcsICgpID0+IHtcbiAgICBleHBlY3QodXRpbC5oYXNFbmNvZGluZ0xvc3MoJ2Jvb2wnLCAnYm9vbCcpKS50b0JlKGZhbHNlKTtcbiAgfSk7XG59KTtcblxuZGVzY3JpYmVXaXRoRmxhZ3MoJ3V0aWwudG9OZXN0ZWRBcnJheScsIEFMTF9FTlZTLCAoKSA9PiB7XG4gIGl0KCcyIGRpbWVuc2lvbnMnLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IG5ldyBGbG9hdDMyQXJyYXkoWzEsIDIsIDMsIDQsIDUsIDZdKTtcbiAgICBleHBlY3QodXRpbC50b05lc3RlZEFycmF5KFsyLCAzXSwgYSkpLnRvRXF1YWwoW1sxLCAyLCAzXSwgWzQsIDUsIDZdXSk7XG4gIH0pO1xuXG4gIGl0KCczIGRpbWVuc2lvbnMgKDJ4MngzKScsICgpID0+IHtcbiAgICBjb25zdCBhID0gbmV3IEZsb2F0MzJBcnJheShbMCwgMSwgMiwgMywgNCwgNSwgNiwgNywgOCwgOSwgMTAsIDExXSk7XG4gICAgZXhwZWN0KHV0aWwudG9OZXN0ZWRBcnJheShbMiwgMiwgM10sIGEpKS50b0VxdWFsKFtcbiAgICAgIFtbMCwgMSwgMl0sIFszLCA0LCA1XV0sIFtbNiwgNywgOF0sIFs5LCAxMCwgMTFdXVxuICAgIF0pO1xuICB9KTtcblxuICBpdCgnMyBkaW1lbnNpb25zICgzeDJ4MiknLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IG5ldyBGbG9hdDMyQXJyYXkoWzAsIDEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEwLCAxMV0pO1xuICAgIGV4cGVjdCh1dGlsLnRvTmVzdGVkQXJyYXkoWzMsIDIsIDJdLCBhKSkudG9FcXVhbChbXG4gICAgICBbWzAsIDFdLCBbMiwgM11dLCBbWzQsIDVdLCBbNiwgN11dLCBbWzgsIDldLCBbMTAsIDExXV1cbiAgICBdKTtcbiAgfSk7XG5cbiAgaXQoJ2ludmFsaWQgZGltZW5zaW9uJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBuZXcgRmxvYXQzMkFycmF5KFsxLCAyLCAzXSk7XG4gICAgZXhwZWN0KCgpID0+IHV0aWwudG9OZXN0ZWRBcnJheShbMiwgMl0sIGEpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ3RlbnNvciB0byBuZXN0ZWQgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgeCA9IHRlbnNvcjJkKFsxLCAyLCAzLCA0XSwgWzIsIDJdKTtcbiAgICBleHBlY3QodXRpbC50b05lc3RlZEFycmF5KHguc2hhcGUsIGF3YWl0IHguZGF0YSgpKSkudG9FcXVhbChbXG4gICAgICBbMSwgMl0sIFszLCA0XVxuICAgIF0pO1xuICB9KTtcblxuICBpdCgnc2NhbGFyIHRvIG5lc3RlZCBhcnJheScsIGFzeW5jICgpID0+IHtcbiAgICBjb25zdCB4ID0gc2NhbGFyKDEpO1xuICAgIGV4cGVjdCh1dGlsLnRvTmVzdGVkQXJyYXkoeC5zaGFwZSwgYXdhaXQgeC5kYXRhKCkpKS50b0VxdWFsKDEpO1xuICB9KTtcblxuICBpdCgndGVuc29yIHdpdGggemVybyBzaGFwZScsICgpID0+IHtcbiAgICBjb25zdCBhID0gbmV3IEZsb2F0MzJBcnJheShbMCwgMV0pO1xuICAgIGV4cGVjdCh1dGlsLnRvTmVzdGVkQXJyYXkoWzEsIDAsIDJdLCBhKSkudG9FcXVhbChbXSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlV2l0aEZsYWdzKCd1dGlsLnRvTmVzdGVkQXJyYXkgZm9yIGEgY29tcGxleCB0ZW5zb3InLCBBTExfRU5WUywgKCkgPT4ge1xuICBpdCgnMiBkaW1lbnNpb25zJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBuZXcgRmxvYXQzMkFycmF5KFsxLCAxMSwgMiwgMTIsIDMsIDEzLCA0LCAxNCwgNSwgMTUsIDYsIDE2XSk7XG4gICAgZXhwZWN0KHV0aWwudG9OZXN0ZWRBcnJheShbMiwgM10sIGEsIHRydWUpKS50b0VxdWFsKFtcbiAgICAgIFsxLCAxMSwgMiwgMTIsIDMsIDEzXSwgWzQsIDE0LCA1LCAxNSwgNiwgMTZdXG4gICAgXSk7XG4gIH0pO1xuXG4gIGl0KCczIGRpbWVuc2lvbnMgKDJ4MngzKScsICgpID0+IHtcbiAgICBjb25zdCBhID0gbmV3IEZsb2F0MzJBcnJheShbXG4gICAgICAwLCA1MCwgMSwgNTEsIDIsIDUyLCAzLCA1MywgNCwgIDU0LCA1LCAgNTUsXG4gICAgICA2LCA1NiwgNywgNTcsIDgsIDU4LCA5LCA1OSwgMTAsIDYwLCAxMSwgNjFcbiAgICBdKTtcbiAgICBleHBlY3QodXRpbC50b05lc3RlZEFycmF5KFsyLCAyLCAzXSwgYSwgdHJ1ZSkpLnRvRXF1YWwoW1xuICAgICAgW1swLCA1MCwgMSwgNTEsIDIsIDUyXSwgWzMsIDUzLCA0LCA1NCwgNSwgNTVdXSxcbiAgICAgIFtbNiwgNTYsIDcsIDU3LCA4LCA1OF0sIFs5LCA1OSwgMTAsIDYwLCAxMSwgNjFdXVxuICAgIF0pO1xuICB9KTtcblxuICBpdCgnMyBkaW1lbnNpb25zICgzeDJ4MiknLCAoKSA9PiB7XG4gICAgY29uc3QgYSA9IG5ldyBGbG9hdDMyQXJyYXkoW1xuICAgICAgMCwgNTAsIDEsIDUxLCAyLCA1MiwgMywgNTMsIDQsICA1NCwgNSwgIDU1LFxuICAgICAgNiwgNTYsIDcsIDU3LCA4LCA1OCwgOSwgNTksIDEwLCA2MCwgMTEsIDYxXG4gICAgXSk7XG4gICAgZXhwZWN0KHV0aWwudG9OZXN0ZWRBcnJheShbMywgMiwgMl0sIGEsIHRydWUpKS50b0VxdWFsKFtcbiAgICAgIFtbMCwgNTAsIDEsIDUxXSwgWzIsIDUyLCAzLCA1M11dLCBbWzQsIDU0LCA1LCA1NV0sIFs2LCA1NiwgNywgNTddXSxcbiAgICAgIFtbOCwgNTgsIDksIDU5XSwgWzEwLCA2MCwgMTEsIDYxXV1cbiAgICBdKTtcbiAgfSk7XG5cbiAgaXQoJ2ludmFsaWQgZGltZW5zaW9uJywgKCkgPT4ge1xuICAgIGNvbnN0IGEgPSBuZXcgRmxvYXQzMkFycmF5KFsxLCAxMSwgMiwgMTIsIDMsIDEzXSk7XG4gICAgZXhwZWN0KCgpID0+IHV0aWwudG9OZXN0ZWRBcnJheShbMiwgMl0sIGEsIHRydWUpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ3RlbnNvciB0byBuZXN0ZWQgYXJyYXknLCBhc3luYyAoKSA9PiB7XG4gICAgY29uc3QgeCA9IGNvbXBsZXgoW1sxLCAyXSwgWzMsIDRdXSwgW1sxMSwgMTJdLCBbMTMsIDE0XV0pO1xuICAgIGV4cGVjdCh1dGlsLnRvTmVzdGVkQXJyYXkoeC5zaGFwZSwgYXdhaXQgeC5kYXRhKCksIHRydWUpKS50b0VxdWFsKFtcbiAgICAgIFsxLCAxMSwgMiwgMTJdLCBbMywgMTMsIDQsIDE0XVxuICAgIF0pO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZSgndXRpbC5mZXRjaCcsICgpID0+IHtcbiAgaXQoJ3Nob3VsZCBjYWxsIHRoZSBwbGF0Zm9ybSBmZXRjaCcsICgpID0+IHtcbiAgICBzcHlPbih0Zi5lbnYoKS5wbGF0Zm9ybSwgJ2ZldGNoJykuYW5kLmNhbGxGYWtlKCgpID0+IHt9KTtcblxuICAgIHV0aWwuZmV0Y2goJ3Rlc3QvcGF0aCcsIHttZXRob2Q6ICdHRVQnfSk7XG5cbiAgICBleHBlY3QodGYuZW52KCkucGxhdGZvcm0uZmV0Y2gpLnRvSGF2ZUJlZW5DYWxsZWRXaXRoKCd0ZXN0L3BhdGgnLCB7XG4gICAgICBtZXRob2Q6ICdHRVQnXG4gICAgfSk7XG4gIH0pO1xufSk7XG5cbmRlc2NyaWJlKCd1dGlsLmVuY29kZVN0cmluZycsICgpID0+IHtcbiAgaXQoJ0VuY29kZSBhbiBlbXB0eSBzdHJpbmcsIGRlZmF1bHQgZW5jb2RpbmcnLCAoKSA9PiB7XG4gICAgY29uc3QgcmVzID0gdXRpbC5lbmNvZGVTdHJpbmcoJycpO1xuICAgIGV4cGVjdChyZXMpLnRvRXF1YWwobmV3IFVpbnQ4QXJyYXkoW10pKTtcbiAgfSk7XG5cbiAgaXQoJ0VuY29kZSBhbiBlbXB0eSBzdHJpbmcsIHV0Zi04IGVuY29kaW5nJywgKCkgPT4ge1xuICAgIGNvbnN0IHJlcyA9IHV0aWwuZW5jb2RlU3RyaW5nKCcnLCAndXRmLTgnKTtcbiAgICBleHBlY3QocmVzKS50b0VxdWFsKG5ldyBVaW50OEFycmF5KFtdKSk7XG4gIH0pO1xuXG4gIGl0KCdFbmNvZGUgYW4gZW1wdHkgc3RyaW5nLCBpbnZhbGlkIGRlY29kaW5nJywgKCkgPT4ge1xuICAgIGV4cGVjdCgoKSA9PiB1dGlsLmVuY29kZVN0cmluZygnJywgJ2Zvb2JhcmJheCcpKS50b1Rocm93RXJyb3IoKTtcbiAgfSk7XG5cbiAgaXQoJ0VuY29kZSBjeXJpbGxpYyBsZXR0ZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IHJlcyA9IHV0aWwuZW5jb2RlU3RyaW5nKCdLYdC6byDRgdGCZScpO1xuICAgIGV4cGVjdChyZXMpLnRvRXF1YWwoXG4gICAgICAgIG5ldyBVaW50OEFycmF5KFs3NSwgOTcsIDIwOCwgMTg2LCAxMTEsIDMyLCAyMDksIDEyOSwgMjA5LCAxMzAsIDEwMV0pKTtcbiAgfSk7XG5cbiAgaXQoJ0VuY29kZSBhc2NpaSBsZXR0ZXJzJywgKCkgPT4ge1xuICAgIGNvbnN0IHJlcyA9IHV0aWwuZW5jb2RlU3RyaW5nKCdoZWxsbycpO1xuICAgIGV4cGVjdChyZXMpLnRvRXF1YWwobmV3IFVpbnQ4QXJyYXkoWzEwNCwgMTAxLCAxMDgsIDEwOCwgMTExXSkpO1xuICB9KTtcbn0pO1xuXG5kZXNjcmliZSgndXRpbC5kZWNvZGVTdHJpbmcnLCAoKSA9PiB7XG4gIGl0KCdkZWNvZGUgYW4gZW1wdHkgc3RyaW5nJywgKCkgPT4ge1xuICAgIGNvbnN0IHMgPSB1dGlsLmRlY29kZVN0cmluZyhuZXcgVWludDhBcnJheShbXSkpO1xuICAgIGV4cGVjdChzKS50b0VxdWFsKCcnKTtcbiAgfSk7XG5cbiAgaXQoJ2RlY29kZSBhc2NpaScsICgpID0+IHtcbiAgICBjb25zdCBzID0gdXRpbC5kZWNvZGVTdHJpbmcobmV3IFVpbnQ4QXJyYXkoWzEwNCwgMTAxLCAxMDgsIDEwOCwgMTExXSkpO1xuICAgIGV4cGVjdChzKS50b0VxdWFsKCdoZWxsbycpO1xuICB9KTtcblxuICBpdCgnZGVjb2RlIGN5cmlsbGljJywgKCkgPT4ge1xuICAgIGNvbnN0IHMgPSB1dGlsLmRlY29kZVN0cmluZyhcbiAgICAgICAgbmV3IFVpbnQ4QXJyYXkoWzc1LCA5NywgMjA4LCAxODYsIDExMSwgMzIsIDIwOSwgMTI5LCAyMDksIDEzMCwgMTAxXSkpO1xuICAgIGV4cGVjdChzKS50b0VxdWFsKCdLYdC6byDRgdGCZScpO1xuICB9KTtcblxuICBpdCgnZGVjb2RlIHV0Zi0xNicsICgpID0+IHtcbiAgICBjb25zdCBzID0gdXRpbC5kZWNvZGVTdHJpbmcoXG4gICAgICAgIG5ldyBVaW50OEFycmF5KFsyNTUsIDI1NCwgMjM3LCAxMzksIDAsIDEzOCwgNCwgODksIDYsIDExNl0pLCAndXRmLTE2Jyk7XG5cbiAgICAvLyBVVEYtMTYgYWxsb3dzIG9wdGlvbmFsIHByZXNlbmNlIG9mIGJ5dGUtb3JkZXItbWFyayAoQk9NKVxuICAgIC8vIENvbnN0cnVjdCBzdHJpbmcgZm9yICfor63oqIDlpITnkIYnLCB3aXRoIGFuZCB3aXRob3V0IEJPTVxuICAgIGNvbnN0IGV4cGVjdGVkID0gU3RyaW5nLmZyb21Db2RlUG9pbnQoMHg4YmVkLCAweDhhMDAsIDB4NTkwNCwgMHg3NDA2KTtcbiAgICBjb25zdCBleHBlY3RlZEJPTSA9XG4gICAgICAgIFN0cmluZy5mcm9tQ29kZVBvaW50KDB4ZmVmZiwgMHg4YmVkLCAweDhhMDAsIDB4NTkwNCwgMHg3NDA2KTtcblxuICAgIGlmIChzLmNvZGVQb2ludEF0KDApID09PSAweGZlZmYpIHtcbiAgICAgIGV4cGVjdChzKS50b0VxdWFsKGV4cGVjdGVkQk9NKTtcbiAgICB9IGVsc2Uge1xuICAgICAgZXhwZWN0KHMpLnRvRXF1YWwoZXhwZWN0ZWQpO1xuICAgIH1cbiAgfSk7XG5cbiAgaXQoJ2Fzc2VydCBwcm9taXNlJywgKCkgPT4ge1xuICAgIGNvbnN0IHByb21pc2UgPSBuZXcgUHJvbWlzZSgoKSA9PiB7fSk7XG4gICAgZXhwZWN0KHV0aWwuaXNQcm9taXNlKHByb21pc2UpKS50b0JlVHJ1dGh5KCk7XG4gICAgY29uc3QgcHJvbWlzZTIgPSB7dGhlbjogKCkgPT4ge319O1xuICAgIGV4cGVjdCh1dGlsLmlzUHJvbWlzZShwcm9taXNlMikpLnRvQmVUcnV0aHkoKTtcbiAgICBjb25zdCBwcm9taXNlMyA9IHt9O1xuICAgIGV4cGVjdCh1dGlsLmlzUHJvbWlzZShwcm9taXNlMykpLnRvQmVGYWxzeSgpO1xuICB9KTtcbn0pO1xuIl19