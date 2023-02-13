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
import { util } from '@tensorflow/tfjs-core';
import { useShapeUniforms } from './gpgpu_math';
export class DepthwiseConvPacked2DProgram {
    constructor(convInfo, addBias = false, activation = null, hasPreluActivation = false, hasLeakyReluAlpha = false) {
        this.variableNames = ['x', 'W'];
        this.packedInputs = true;
        this.packedOutput = true;
        this.customUniforms = [
            { name: 'pads', type: 'ivec2' },
            { name: 'strides', type: 'ivec2' },
            { name: 'dilations', type: 'ivec2' },
            { name: 'inDims', type: 'ivec2' },
        ];
        this.outputShape = convInfo.outShape;
        this.enableShapeUniforms = useShapeUniforms(this.outputShape.length);
        const channelMul = convInfo.outChannels / convInfo.inChannels;
        const padLeft = convInfo.padInfo.left;
        const strideWidth = convInfo.strideWidth;
        const dilationWidth = convInfo.dilationWidth;
        const filterHeight = convInfo.filterHeight;
        const filterWidth = convInfo.filterWidth;
        const texelsAcross = filterWidth;
        let mainLoop = `
      int xR; int xC; int xCOffset;
      vec4 wTexel; vec4 previous; vec4 final;`;
        for (let c = 0; c < filterWidth; c++) {
            mainLoop += `
          vec4 xTexelC${c * 2};
          int xTexelC${c * 2}Ready;
          vec4 xTexelC${c * 2 + 1};
          int xTexelC${c * 2 + 1}Ready;
          vec4 xC${c};`;
        }
        /**
         * This vectorized implementation works by gathering the values needed for
         * each output channel's dot product into vec4's and then multiplying them
         * all together (this happens in the final double for-loop below). Most of
         * the main loop consists of constructing these vec4's with the minimum
         * number of texture2D calls, which means making use of all four returned
         * values from a texture2D call at once.
         */
        mainLoop += `
    for (int r = 0; r < ${filterHeight}; r++) {
      `;
        for (let c = 0; c < filterWidth; c++) {
            mainLoop += `
          xTexelC${c * 2} = vec4(0.0);
          xTexelC${c * 2}Ready = 0;
          xTexelC${c * 2 + 1} = vec4(0.0);
          xTexelC${c * 2 + 1}Ready = 0;
          xC${c} = vec4(0.0);`;
        }
        mainLoop += `
        xR = xRCorner + r * dilations[0];
        if (xR >=0 && xR < inDims[0]) {
      `;
        for (let texelC = 0; texelC < (texelsAcross + 1) / 2; texelC++) {
            const colIndex = texelC * 2;
            mainLoop += `
          xC = xCCorner + ${colIndex * dilationWidth};
          `;
            if (strideWidth === 1) {
                if (colIndex < filterWidth) {
                    // If padding is odd, the outer texels have to be composed.
                    if (padLeft % 2 === 1) {
                        // TODO: Ensure vec4 previous does not result in redundant sample,
                        // and avoid setting xTexelRC's that exceed the boundary in the
                        // first place rather than resetting them to vec4(0)).
                        // To compute xCOffset:
                        // - If padding is odd, we must add 1 to ensure we ask for an
                        // even-numbered row.
                        // - We subtract 2 to access the previous texel.
                        mainLoop += `
                xCOffset = xC + 1;
                if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${colIndex}Ready == 0) {
                  xTexelC${colIndex} = getX(batch, xR, xCOffset, d1);

                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${colIndex}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex}Ready = 1;
                }
              `;
                        // This texel has been read in previous iteration if the dilation
                        // is 1.
                        if (dilationWidth === 1 && colIndex > 0) {
                            mainLoop += `
                xC${colIndex} = vec4(xTexelC${colIndex - 2}.zw, xTexelC${colIndex}.xy);
                `;
                        }
                        else {
                            mainLoop += `
                  xCOffset = xC + 1 - 2;

                  if (xCOffset >= 0 && xCOffset < inDims[1]) {
                    previous = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      previous.zw = vec2(0.0);
                    }

                    xC${colIndex} = vec4(previous.zw, xTexelC${colIndex}.xy);
                  } else {
                    xC${colIndex} = vec4(0.0, 0.0, xTexelC${colIndex}.xy);
                  }
                  `;
                        }
                    }
                    else {
                        // Padding is even, so xRC corresponds to a single texel.
                        mainLoop += `
                if (xC >= 0 && xC < inDims[1] && xTexelC${colIndex}Ready == 0) {
                  xTexelC${colIndex} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${colIndex}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex}Ready = 1;
                }

                xC${colIndex} = xTexelC${colIndex};
                `;
                    }
                    if (colIndex + 1 < filterWidth) {
                        // If dilation is even, the second entry should match the first
                        // (either both are composed or both are single samples). But if
                        // dilation is odd, then the second entry should be the opposite
                        // of the first (if the first is composed, the second is a single
                        // sample, and vice versa.)
                        const nextTexelOffset = padLeft % 2 === 0 ?
                            util.nearestLargerEven(dilationWidth) :
                            dilationWidth;
                        if ((dilationWidth % 2 === 0 && padLeft % 2 === 1) ||
                            (dilationWidth % 2 !== 0 && padLeft % 2 !== 1)) {
                            mainLoop += `
                  xCOffset = xC + imod(pads[1], 2) + ${nextTexelOffset};

                  if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${colIndex + 1}Ready == 0) {
                    xTexelC${colIndex + 1} = getX(batch, xR, xCOffset, d1);

                    // Need to manually clear unused channels in case
                    // we're reading from recycled texture.
                    if (xCOffset + 1 >= inDims[1]) {
                      xTexelC${colIndex + 1}.zw = vec2(0.0);
                    }
                    xTexelC${colIndex + 1}Ready = 1;
                  }
                  `;
                            // If dilation > 1 then the xRC's will not be able to share any
                            // values, so each xRC will require two unique calls to getX.
                            if (dilationWidth > 1) {
                                mainLoop += `
                    xCOffset -= 2;
                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${colIndex}Ready == 0) {
                      xTexelC${colIndex} = getX(batch, xR, xCOffset, d1);
                      xTexelC${colIndex}Ready = 1;
                    }
                    `;
                            }
                            mainLoop += `
                  xC${colIndex + 1} = vec4(xTexelC${colIndex}.zw, xTexelC${colIndex + 1}.xy);
                  `;
                        }
                        else {
                            // If dilation is 1 and padding is odd, we have already read the
                            // texel when constructing the previous x value. Here we can
                            // simply skip the texture read.
                            if (nextTexelOffset === 1) {
                                mainLoop += `
                    xC${colIndex + 1} = xTexelC${colIndex};
                    `;
                            }
                            else {
                                mainLoop += `
                    xCOffset = xC + ${nextTexelOffset};

                    if (xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${colIndex + 1}Ready == 0) {
                      xTexelC${colIndex + 1} = getX(batch, xR, xCOffset, d1);
                      if (xCOffset + 1 >= inDims[1]) {
                        xTexelC${colIndex + 1}.zw = vec2(0.0);
                      }
                      xTexelC${colIndex + 1}Ready = 1;
                    }

                    xC${colIndex + 1} = xTexelC${colIndex + 1};
                    `;
                            }
                        }
                    }
                }
            }
            else { // stride === 2
                if (colIndex < filterWidth) {
                    // Depending on whether padLeft is even or odd, we want either the
                    // xy or zw channels from X texels for xC${colIndex}. If padLeft is
                    // even, xC${colIndex +1} is simply the zw channels of texels we've
                    // already sampled. But if padLeft is odd, xC{$c + 1}.zw will
                    // need to come from the xy channels of a new texel, hence the `
                    // vec4
                    // final` initialized below.
                    if (padLeft % 2 === 1) {
                        mainLoop += `
                xCOffset = xC + 1 - strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${colIndex}Ready == 0) {
                  xTexelC${colIndex} = getX(batch, xR, xCOffset, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${colIndex}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex}Ready = 1;
                }

                if(xC + 1 >= 0 && xC + 1 < inDims[1] && xTexelC${colIndex + 1}Ready == 0) {
                  xTexelC${colIndex + 1} = getX(batch, xR, xC + 1, d1);
                  // Need to manually clear unused channels in case
                  // we're reading from recycled texture.
                  if (xC + 2 >= inDims[1]) {
                    xTexelC${colIndex + 1}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex + 1}Ready = 1;
                }

                xC${colIndex} = vec4(xTexelC${colIndex}.zw, xTexelC${colIndex + 1}.zw);
              `;
                        if (colIndex + 1 < filterWidth) {
                            mainLoop += `
                  final = vec4(0.0);
                  xCOffset = xC + 1 + strides[1];
                  if(xCOffset >= 0 && xCOffset < inDims[1]) {
                    final = getX(batch, xR, xCOffset, d1);
                  }
                  xC${colIndex + 1} = vec4(xTexelC${colIndex + 1}.xy, final.xy);
                `;
                        }
                    }
                    else {
                        mainLoop += `
                if(xC >= 0 && xC < inDims[1] && xTexelC${colIndex}Ready == 0) {
                  xTexelC${colIndex} = getX(batch, xR, xC, d1);
                  if (xC + 1 >= inDims[1]) {
                    xTexelC${colIndex}.zw = vec2(0.0);
                  }
                  xTexelC${colIndex}Ready = 1;
                }

                xCOffset = xC + strides[1];
                if(xCOffset >= 0 && xCOffset < inDims[1] && xTexelC${colIndex + 1}Ready == 0) {
                  xTexelC${colIndex + 1} = getX(batch, xR, xCOffset, d1);
                  if (xCOffset + 1 >= inDims[1]) {
                    xTexelC${colIndex + 1}.zw = vec2(0.);
                  }
                  xTexelC${colIndex + 1}Ready = 1;
                }

                xC${colIndex} = vec4(
                  xTexelC${colIndex}.xy, xTexelC${colIndex + 1}.xy);
              `;
                        if (colIndex + 1 < filterWidth) {
                            mainLoop += `
                  xC${colIndex + 1} = vec4(xTexelC${colIndex}.zw, xTexelC${colIndex + 1}.zw);
                `;
                        }
                    }
                }
            }
            // localize the dotProd accumulation within the loop, the theory is for
            // GPU with limited cache, accumulate sum across large amount of
            // veriables will cause lots of cache misses. (i.e. 5x5 filter will have
            // 50 variables)
            if (colIndex < filterWidth) {
                mainLoop += `
            wTexel = getW(r, ${colIndex}, d1, q);
            dotProd += xC${colIndex} * vec4(wTexel.xz, wTexel.xz);
          `;
                if (colIndex + 1 < filterWidth) {
                    mainLoop += `
              wTexel = getW(r, ${colIndex + 1}, d1, q);
              dotProd += xC${colIndex + 1} * vec4(wTexel.xz, wTexel.xz);
            `;
                }
            }
        }
        mainLoop += `
    }
  `;
        mainLoop += `
      }
    `;
        let activationSnippet = '', applyActivationSnippet = '';
        if (activation) {
            if (hasPreluActivation) {
                activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getPreluActivationWeightsAtOutCoords();
          ${activation}
        }`;
            }
            else if (hasLeakyReluAlpha) {
                activationSnippet = `vec4 activation(vec4 a) {
          vec4 b = getLeakyreluAlphaAtOutCoords();
          ${activation}
        }`;
            }
            else {
                activationSnippet = `vec4 activation(vec4 x) {
          ${activation}
        }`;
            }
            applyActivationSnippet = `result = activation(result);`;
        }
        const addBiasSnippet = addBias ? 'result += getBiasAtOutCoords();' : '';
        if (addBias) {
            this.variableNames.push('bias');
        }
        if (hasPreluActivation) {
            this.variableNames.push('preluActivationWeights');
        }
        if (hasLeakyReluAlpha) {
            this.variableNames.push('leakyreluAlpha');
        }
        this.userCode = `
      ${activationSnippet}

      void main() {
        ivec4 coords = getOutputCoords();
        int batch = coords.x;
        ivec2 xRCCorner = coords.yz * strides - pads;
        int d2 = coords.w;
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};
        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        //intialize dotProd with a small epsilon seems to reduce GPU accuracy loss.
        vec4 dotProd = vec4(0.000000000000001);

        ${mainLoop}

        vec4 result = dotProd - vec4(0.000000000000001);
        ${addBiasSnippet}
        ${applyActivationSnippet}
        setOutput(result);
      }
    `;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udl9wYWNrZWRfZ3B1X2RlcHRod2lzZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvY29udl9wYWNrZWRfZ3B1X2RlcHRod2lzZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQWUsSUFBSSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFekQsT0FBTyxFQUFlLGdCQUFnQixFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRTVELE1BQU0sT0FBTyw0QkFBNEI7SUFjdkMsWUFDSSxRQUFpQyxFQUFFLE9BQU8sR0FBRyxLQUFLLEVBQ2xELGFBQXFCLElBQUksRUFBRSxrQkFBa0IsR0FBRyxLQUFLLEVBQ3JELGlCQUFpQixHQUFHLEtBQUs7UUFoQjdCLGtCQUFhLEdBQUcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxDQUFDLENBQUM7UUFDM0IsaUJBQVksR0FBRyxJQUFJLENBQUM7UUFDcEIsaUJBQVksR0FBRyxJQUFJLENBQUM7UUFJcEIsbUJBQWMsR0FBRztZQUNmLEVBQUMsSUFBSSxFQUFFLE1BQU0sRUFBRSxJQUFJLEVBQUUsT0FBZ0IsRUFBRTtZQUN2QyxFQUFDLElBQUksRUFBRSxTQUFTLEVBQUUsSUFBSSxFQUFFLE9BQWdCLEVBQUU7WUFDMUMsRUFBQyxJQUFJLEVBQUUsV0FBVyxFQUFFLElBQUksRUFBRSxPQUFnQixFQUFFO1lBQzVDLEVBQUMsSUFBSSxFQUFFLFFBQVEsRUFBRSxJQUFJLEVBQUUsT0FBZ0IsRUFBRTtTQUMxQyxDQUFDO1FBTUEsSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUMsUUFBUSxDQUFDO1FBQ3JDLElBQUksQ0FBQyxtQkFBbUIsR0FBRyxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ3JFLE1BQU0sVUFBVSxHQUFHLFFBQVEsQ0FBQyxXQUFXLEdBQUcs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O1FBQ0QsSUFBSSxpQkFBaUIsRUFBRTtZQUNyQixJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDO1NBQzNDO1FBRUQsSUFBSSxDQUFDLFFBQVEsR0FBRztRQUNaLGlCQUFpQjs7Ozs7Ozt3QkFPRCxVQUFVOzRCQUNOLFVBQVU7Ozs7Ozs7VUFPNUIsUUFBUTs7O1VBR1IsY0FBYztVQUNkLHNCQUFzQjs7O0tBRzNCLENBQUM7SUFDSixDQUFDO0NBQ0YiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCB1dGlsfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge0dQR1BVUHJvZ3JhbSwgdXNlU2hhcGVVbmlmb3Jtc30gZnJvbSAnLi9ncGdwdV9tYXRoJztcblxuZXhwb3J0IGNsYXNzIERlcHRod2lzZUNvbnZQYWNrZWQyRFByb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ1cnXTtcbiAgcGFja2VkSW5wdXRzID0gdHJ1ZTtcbiAgcGFja2VkT3V0cHV0ID0gdHJ1ZTtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuICBlbmFibGVTaGFwZVVuaWZvcm1zOiBib29sZWFuO1xuICBjdXN0b21Vbmlmb3JtcyA9IFtcbiAgICB7bmFtZTogJ3BhZHMnLCB0eXBlOiAnaXZlYzInIGFzIGNvbnN0IH0sXG4gICAge25hbWU6ICdzdHJpZGVzJywgdHlwZTogJ2l2ZWMyJyBhcyBjb25zdCB9LFxuICAgIHtuYW1lOiAnZGlsYXRpb25zJywgdHlwZTogJ2l2ZWMyJyBhcyBjb25zdCB9LFxuICAgIHtuYW1lOiAnaW5EaW1zJywgdHlwZTogJ2l2ZWMyJyBhcyBjb25zdCB9LFxuICBdO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgY29udkluZm86IGJhY2tlbmRfdXRpbC5Db252MkRJbmZvLCBhZGRCaWFzID0gZmFsc2UsXG4gICAgICBhY3RpdmF0aW9uOiBzdHJpbmcgPSBudWxsLCBoYXNQcmVsdUFjdGl2YXRpb24gPSBmYWxzZSxcbiAgICAgIGhhc0xlYWt5UmVsdUFscGhhID0gZmFsc2UpIHtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gY29udkluZm8ub3V0U2hhcGU7XG4gICAgdGhpcy5lbmFibGVTaGFwZVVuaWZvcm1zID0gdXNlU2hhcGVVbmlmb3Jtcyh0aGlzLm91dHB1dFNoYXBlLmxlbmd0aCk7XG4gICAgY29uc3QgY2hhbm5lbE11bCA9IGNvbnZJbmZvLm91dENoYW5uZWxzIC8gY29udkluZm8uaW5DaGFubmVscztcbiAgICBjb25zdCBwYWRMZWZ0ID0gY29udkluZm8ucGFkSW5mby5sZWZ0O1xuICAgIGNvbnN0IHN0cmlkZVdpZHRoID0gY29udkluZm8uc3RyaWRlV2lkdGg7XG4gICAgY29uc3QgZGlsYXRpb25XaWR0aCA9IGNvbnZJbmZvLmRpbGF0aW9uV2lkdGg7XG4gICAgY29uc3QgZmlsdGVySGVpZ2h0ID0gY29udkluZm8uZmlsdGVySGVpZ2h0O1xuICAgIGNvbnN0IGZpbHRlcldpZHRoID0gY29udkluZm8uZmlsdGVyV2lkdGg7XG4gICAgY29uc3QgdGV4ZWxzQWNyb3NzID0gZmlsdGVyV2lkdGg7XG5cbiAgICBsZXQgbWFpbkxvb3AgPSBgXG4gICAgICBpbnQgeFI7IGludCB4QzsgaW50IHhDT2Zmc2V0O1xuICAgICAgdmVjNCB3VGV4ZWw7IHZlYzQgcHJldmlvdXM7IHZlYzQgZmluYWw7YDtcblxuICAgIGZvciAobGV0IGMgPSAwOyBjIDwgZmlsdGVyV2lkdGg7IGMrKykge1xuICAgICAgbWFpbkxvb3AgKz0gYFxuICAgICAgICAgIHZlYzQgeFRleGVsQyR7YyAqIDJ9O1xuICAgICAgICAgIGludCB4VGV4ZWxDJHtjICogMn1SZWFkeTtcbiAgICAgICAgICB2ZWM0IHhUZXhlbEMke2MgKiAyICsgMX07XG4gICAgICAgICAgaW50IHhUZXhlbEMke2MgKiAyICsgMX1SZWFkeTtcbiAgICAgICAgICB2ZWM0IHhDJHtjfTtgO1xuICAgIH1cblxuICAgIC8qKlxuICAgICAqIFRoaXMgdmVjdG9yaXplZCBpbXBsZW1lbnRhdGlvbiB3b3JrcyBieSBnYXRoZXJpbmcgdGhlIHZhbHVlcyBuZWVkZWQgZm9yXG4gICAgICogZWFjaCBvdXRwdXQgY2hhbm5lbCdzIGRvdCBwcm9kdWN0IGludG8gdmVjNCdzIGFuZCB0aGVuIG11bHRpcGx5aW5nIHRoZW1cbiAgICAgKiBhbGwgdG9nZXRoZXIgKHRoaXMgaGFwcGVucyBpbiB0aGUgZmluYWwgZG91YmxlIGZvci1sb29wIGJlbG93KS4gTW9zdCBvZlxuICAgICAqIHRoZSBtYWluIGxvb3AgY29uc2lzdHMgb2YgY29uc3RydWN0aW5nIHRoZXNlIHZlYzQncyB3aXRoIHRoZSBtaW5pbXVtXG4gICAgICogbnVtYmVyIG9mIHRleHR1cmUyRCBjYWxscywgd2hpY2ggbWVhbnMgbWFraW5nIHVzZSBvZiBhbGwgZm91ciByZXR1cm5lZFxuICAgICAqIHZhbHVlcyBmcm9tIGEgdGV4dHVyZTJEIGNhbGwgYXQgb25jZS5cbiAgICAgKi9cbiAgICBtYWluTG9vcCArPSBgXG4gICAgZm9yIChpbnQgciA9IDA7IHIgPCAke2ZpbHRlckhlaWdodH07IHIrKykge1xuICAgICAgYDtcbiAgICBmb3IgKGxldCBjID0gMDsgYyA8IGZpbHRlcldpZHRoOyBjKyspIHtcbiAgICAgIG1haW5Mb29wICs9IGBcbiAgICAgICAgICB4VGV4ZWxDJHtjICogMn0gPSB2ZWM0KDAuMCk7XG4gICAgICAgICAgeFRleGVsQyR7YyAqIDJ9UmVhZHkgPSAwO1xuICAgICAgICAgIHhUZXhlbEMke2MgKiAyICsgMX0gPSB2ZWM0KDAuMCk7XG4gICAgICAgICAgeFRleGVsQyR7YyAqIDIgKyAxfVJlYWR5ID0gMDtcbiAgICAgICAgICB4QyR7Y30gPSB2ZWM0KDAuMCk7YDtcbiAgICB9XG4gICAgbWFpbkxvb3AgKz0gYFxuICAgICAgICB4UiA9IHhSQ29ybmVyICsgciAqIGRpbGF0aW9uc1swXTtcbiAgICAgICAgaWYgKHhSID49MCAmJiB4UiA8IGluRGltc1swXSkge1xuICAgICAgYDtcblxuICAgIGZvciAobGV0IHRleGVsQyA9IDA7IHRleGVsQyA8ICh0ZXhlbHNBY3Jvc3MgKyAxKSAvIDI7IHRleGVsQysrKSB7XG4gICAgICBjb25zdCBjb2xJbmRleCA9IHRleGVsQyAqIDI7XG5cbiAgICAgIG1haW5Mb29wICs9IGBcbiAgICAgICAgICB4QyA9IHhDQ29ybmVyICsgJHtjb2xJbmRleCAqIGRpbGF0aW9uV2lkdGh9O1xuICAgICAgICAgIGA7XG5cbiAgICAgIGlmIChzdHJpZGVXaWR0aCA9PT0gMSkge1xuICAgICAgICBpZiAoY29sSW5kZXggPCBmaWx0ZXJXaWR0aCkge1xuICAgICAgICAgIC8vIElmIHBhZGRpbmcgaXMgb2RkLCB0aGUgb3V0ZXIgdGV4ZWxzIGhhdmUgdG8gYmUgY29tcG9zZWQuXG4gICAgICAgICAgaWYgKHBhZExlZnQgJSAyID09PSAxKSB7XG4gICAgICAgICAgICAvLyBUT0RPOiBFbnN1cmUgdmVjNCBwcmV2aW91cyBkb2VzIG5vdCByZXN1bHQgaW4gcmVkdW5kYW50IHNhbXBsZSxcbiAgICAgICAgICAgIC8vIGFuZCBhdm9pZCBzZXR0aW5nIHhUZXhlbFJDJ3MgdGhhdCBleGNlZWQgdGhlIGJvdW5kYXJ5IGluIHRoZVxuICAgICAgICAgICAgLy8gZmlyc3QgcGxhY2UgcmF0aGVyIHRoYW4gcmVzZXR0aW5nIHRoZW0gdG8gdmVjNCgwKSkuXG5cbiAgICAgICAgICAgIC8vIFRvIGNvbXB1dGUgeENPZmZzZXQ6XG4gICAgICAgICAgICAvLyAtIElmIHBhZGRpbmcgaXMgb2RkLCB3ZSBtdXN0IGFkZCAxIHRvIGVuc3VyZSB3ZSBhc2sgZm9yIGFuXG4gICAgICAgICAgICAvLyBldmVuLW51bWJlcmVkIHJvdy5cbiAgICAgICAgICAgIC8vIC0gV2Ugc3VidHJhY3QgMiB0byBhY2Nlc3MgdGhlIHByZXZpb3VzIHRleGVsLlxuXG4gICAgICAgICAgICBtYWluTG9vcCArPSBgXG4gICAgICAgICAgICAgICAgeENPZmZzZXQgPSB4QyArIDE7XG4gICAgICAgICAgICAgICAgaWYgKHhDT2Zmc2V0ID49IDAgJiYgeENPZmZzZXQgPCBpbkRpbXNbMV0gJiYgeFRleGVsQyR7XG4gICAgICAgICAgICAgICAgY29sSW5kZXh9UmVhZHkgPT0gMCkge1xuICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXh9ID0gZ2V0WChiYXRjaCwgeFIsIHhDT2Zmc2V0LCBkMSk7XG5cbiAgICAgICAgICAgICAgICAgIC8vIE5lZWQgdG8gbWFudWFsbHkgY2xlYXIgdW51c2VkIGNoYW5uZWxzIGluIGNhc2VcbiAgICAgICAgICAgICAgICAgIC8vIHdlJ3JlIHJlYWRpbmcgZnJvbSByZWN5Y2xlZCB0ZXh0dXJlLlxuICAgICAgICAgICAgICAgICAgaWYgKHhDT2Zmc2V0ICsgMSA+PSBpbkRpbXNbMV0pIHtcbiAgICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXh9Lnp3ID0gdmVjMigwLjApO1xuICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXh9UmVhZHkgPSAxO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgYDtcbiAgICAgICAgICAgIC8vIFRoaXMgdGV4ZWwgaGFzIGJlZW4gcmVhZCBpbiBwcmV2aW91cyBpdGVyYXRpb24gaWYgdGhlIGRpbGF0aW9uXG4gICAgICAgICAgICAvLyBpcyAxLlxuICAgICAgICAgICAgaWYgKGRpbGF0aW9uV2lkdGggPT09IDEgJiYgY29sSW5kZXggPiAwKSB7XG4gICAgICAgICAgICAgIG1haW5Mb29wICs9IGBcbiAgICAgICAgICAgICAgICB4QyR7Y29sSW5kZXh9ID0gdmVjNCh4VGV4ZWxDJHtjb2xJbmRleCAtIDJ9Lnp3LCB4VGV4ZWxDJHtcbiAgICAgICAgICAgICAgICAgIGNvbEluZGV4fS54eSk7XG4gICAgICAgICAgICAgICAgYDtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIG1haW5Mb29wICs9IGBcbiAgICAgICAgICAgICAgICAgIHhDT2Zmc2V0ID0geEMgKyAxIC0gMjtcblxuICAgICAgICAgICAgICAgICAgaWYgKHhDT2Zmc2V0ID49IDAgJiYgeENPZmZzZXQgPCBpbkRpbXNbMV0pIHtcbiAgICAgICAgICAgICAgICAgICAgcHJldmlvdXMgPSBnZXRYKGJhdGNoLCB4UiwgeENPZmZzZXQsIGQxKTtcblxuICAgICAgICAgICAgICAgICAgICAvLyBOZWVkIHRvIG1hbnVhbGx5IGNsZWFyIHVudXNlZCBjaGFubmVscyBpbiBjYXNlXG4gICAgICAgICAgICAgICAgICAgIC8vIHdlJ3JlIHJlYWRpbmcgZnJvbSByZWN5Y2xlZCB0ZXh0dXJlLlxuICAgICAgICAgICAgICAgICAgICBpZiAoeENPZmZzZXQgKyAxID49IGluRGltc1sxXSkge1xuICAgICAgICAgICAgICAgICAgICAgIHByZXZpb3VzLnp3ID0gdmVjMigwLjApO1xuICAgICAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICAgICAgeEMke2NvbEluZGV4fSA9IHZlYzQocHJldmlvdXMuencsIHhUZXhlbEMke2NvbEluZGV4fS54eSk7XG4gICAgICAgICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAgICAgICB4QyR7Y29sSW5kZXh9ID0gdmVjNCgwLjAsIDAuMCwgeFRleGVsQyR7Y29sSW5kZXh9Lnh5KTtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgIGA7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIC8vIFBhZGRpbmcgaXMgZXZlbiwgc28geFJDIGNvcnJlc3BvbmRzIHRvIGEgc2luZ2xlIHRleGVsLlxuICAgICAgICAgICAgbWFpbkxvb3AgKz0gYFxuICAgICAgICAgICAgICAgIGlmICh4QyA+PSAwICYmIHhDIDwgaW5EaW1zWzFdICYmIHhUZXhlbEMke2NvbEluZGV4fVJlYWR5ID09IDApIHtcbiAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4fSA9IGdldFgoYmF0Y2gsIHhSLCB4QywgZDEpO1xuICAgICAgICAgICAgICAgICAgaWYgKHhDICsgMSA+PSBpbkRpbXNbMV0pIHtcbiAgICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXh9Lnp3ID0gdmVjMigwLjApO1xuICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXh9UmVhZHkgPSAxO1xuICAgICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICAgIHhDJHtjb2xJbmRleH0gPSB4VGV4ZWxDJHtjb2xJbmRleH07XG4gICAgICAgICAgICAgICAgYDtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBpZiAoY29sSW5kZXggKyAxIDwgZmlsdGVyV2lkdGgpIHtcbiAgICAgICAgICAgIC8vIElmIGRpbGF0aW9uIGlzIGV2ZW4sIHRoZSBzZWNvbmQgZW50cnkgc2hvdWxkIG1hdGNoIHRoZSBmaXJzdFxuICAgICAgICAgICAgLy8gKGVpdGhlciBib3RoIGFyZSBjb21wb3NlZCBvciBib3RoIGFyZSBzaW5nbGUgc2FtcGxlcykuIEJ1dCBpZlxuICAgICAgICAgICAgLy8gZGlsYXRpb24gaXMgb2RkLCB0aGVuIHRoZSBzZWNvbmQgZW50cnkgc2hvdWxkIGJlIHRoZSBvcHBvc2l0ZVxuICAgICAgICAgICAgLy8gb2YgdGhlIGZpcnN0IChpZiB0aGUgZmlyc3QgaXMgY29tcG9zZWQsIHRoZSBzZWNvbmQgaXMgYSBzaW5nbGVcbiAgICAgICAgICAgIC8vIHNhbXBsZSwgYW5kIHZpY2UgdmVyc2EuKVxuXG4gICAgICAgICAgICBjb25zdCBuZXh0VGV4ZWxPZmZzZXQgPSBwYWRMZWZ0ICUgMiA9PT0gMCA/XG4gICAgICAgICAgICAgICAgdXRpbC5uZWFyZXN0TGFyZ2VyRXZlbihkaWxhdGlvbldpZHRoKSA6XG4gICAgICAgICAgICAgICAgZGlsYXRpb25XaWR0aDtcblxuICAgICAgICAgICAgaWYgKChkaWxhdGlvbldpZHRoICUgMiA9PT0gMCAmJiBwYWRMZWZ0ICUgMiA9PT0gMSkgfHxcbiAgICAgICAgICAgICAgICAoZGlsYXRpb25XaWR0aCAlIDIgIT09IDAgJiYgcGFkTGVmdCAlIDIgIT09IDEpKSB7XG4gICAgICAgICAgICAgIG1haW5Mb29wICs9IGBcbiAgICAgICAgICAgICAgICAgIHhDT2Zmc2V0ID0geEMgKyBpbW9kKHBhZHNbMV0sIDIpICsgJHtuZXh0VGV4ZWxPZmZzZXR9O1xuXG4gICAgICAgICAgICAgICAgICBpZiAoeENPZmZzZXQgPj0gMCAmJiB4Q09mZnNldCA8IGluRGltc1sxXSAmJiB4VGV4ZWxDJHtcbiAgICAgICAgICAgICAgICAgIGNvbEluZGV4ICsgMX1SZWFkeSA9PSAwKSB7XG4gICAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4ICsgMX0gPSBnZXRYKGJhdGNoLCB4UiwgeENPZmZzZXQsIGQxKTtcblxuICAgICAgICAgICAgICAgICAgICAvLyBOZWVkIHRvIG1hbnVhbGx5IGNsZWFyIHVudXNlZCBjaGFubmVscyBpbiBjYXNlXG4gICAgICAgICAgICAgICAgICAgIC8vIHdlJ3JlIHJlYWRpbmcgZnJvbSByZWN5Y2xlZCB0ZXh0dXJlLlxuICAgICAgICAgICAgICAgICAgICBpZiAoeENPZmZzZXQgKyAxID49IGluRGltc1sxXSkge1xuICAgICAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4ICsgMX0uencgPSB2ZWMyKDAuMCk7XG4gICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXggKyAxfVJlYWR5ID0gMTtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgIGA7XG5cbiAgICAgICAgICAgICAgLy8gSWYgZGlsYXRpb24gPiAxIHRoZW4gdGhlIHhSQydzIHdpbGwgbm90IGJlIGFibGUgdG8gc2hhcmUgYW55XG4gICAgICAgICAgICAgIC8vIHZhbHVlcywgc28gZWFjaCB4UkMgd2lsbCByZXF1aXJlIHR3byB1bmlxdWUgY2FsbHMgdG8gZ2V0WC5cbiAgICAgICAgICAgICAgaWYgKGRpbGF0aW9uV2lkdGggPiAxKSB7XG4gICAgICAgICAgICAgICAgbWFpbkxvb3AgKz0gYFxuICAgICAgICAgICAgICAgICAgICB4Q09mZnNldCAtPSAyO1xuICAgICAgICAgICAgICAgICAgICBpZiAoeENPZmZzZXQgPj0gMCAmJiB4Q09mZnNldCA8IGluRGltc1sxXSAmJiB4VGV4ZWxDJHtcbiAgICAgICAgICAgICAgICAgICAgY29sSW5kZXh9UmVhZHkgPT0gMCkge1xuICAgICAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4fSA9IGdldFgoYmF0Y2gsIHhSLCB4Q09mZnNldCwgZDEpO1xuICAgICAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4fVJlYWR5ID0gMTtcbiAgICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgICBgO1xuICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgbWFpbkxvb3AgKz0gYFxuICAgICAgICAgICAgICAgICAgeEMke2NvbEluZGV4ICsgMX0gPSB2ZWM0KHhUZXhlbEMke2NvbEluZGV4fS56dywgeFRleGVsQyR7XG4gICAgICAgICAgICAgICAgICBjb2xJbmRleCArIDF9Lnh5KTtcbiAgICAgICAgICAgICAgICAgIGA7XG4gICAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgICAvLyBJZiBkaWxhdGlvbiBpcyAxIGFuZCBwYWRkaW5nIGlzIG9kZCwgd2UgaGF2ZSBhbHJlYWR5IHJlYWQgdGhlXG4gICAgICAgICAgICAgIC8vIHRleGVsIHdoZW4gY29uc3RydWN0aW5nIHRoZSBwcmV2aW91cyB4IHZhbHVlLiBIZXJlIHdlIGNhblxuICAgICAgICAgICAgICAvLyBzaW1wbHkgc2tpcCB0aGUgdGV4dHVyZSByZWFkLlxuICAgICAgICAgICAgICBpZiAobmV4dFRleGVsT2Zmc2V0ID09PSAxKSB7XG4gICAgICAgICAgICAgICAgbWFpbkxvb3AgKz0gYFxuICAgICAgICAgICAgICAgICAgICB4QyR7Y29sSW5kZXggKyAxfSA9IHhUZXhlbEMke2NvbEluZGV4fTtcbiAgICAgICAgICAgICAgICAgICAgYDtcbiAgICAgICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgICAgICBtYWluTG9vcCArPSBgXG4gICAgICAgICAgICAgICAgICAgIHhDT2Zmc2V0ID0geEMgKyAke25leHRUZXhlbE9mZnNldH07XG5cbiAgICAgICAgICAgICAgICAgICAgaWYgKHhDT2Zmc2V0ID49IDAgJiYgeENPZmZzZXQgPCBpbkRpbXNbMV0gJiYgeFRleGVsQyR7XG4gICAgICAgICAgICAgICAgICAgIGNvbEluZGV4ICsgMX1SZWFkeSA9PSAwKSB7XG4gICAgICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXggKyAxfSA9IGdldFgoYmF0Y2gsIHhSLCB4Q09mZnNldCwgZDEpO1xuICAgICAgICAgICAgICAgICAgICAgIGlmICh4Q09mZnNldCArIDEgPj0gaW5EaW1zWzFdKSB7XG4gICAgICAgICAgICAgICAgICAgICAgICB4VGV4ZWxDJHtjb2xJbmRleCArIDF9Lnp3ID0gdmVjMigwLjApO1xuICAgICAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgICAgICB4VGV4ZWxDJHtjb2xJbmRleCArIDF9UmVhZHkgPSAxO1xuICAgICAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICAgICAgeEMke2NvbEluZGV4ICsgMX0gPSB4VGV4ZWxDJHtjb2xJbmRleCArIDF9O1xuICAgICAgICAgICAgICAgICAgICBgO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9IGVsc2UgeyAgLy8gc3RyaWRlID09PSAyXG4gICAgICAgIGlmIChjb2xJbmRleCA8IGZpbHRlcldpZHRoKSB7XG4gICAgICAgICAgLy8gRGVwZW5kaW5nIG9uIHdoZXRoZXIgcGFkTGVmdCBpcyBldmVuIG9yIG9kZCwgd2Ugd2FudCBlaXRoZXIgdGhlXG4gICAgICAgICAgLy8geHkgb3IgencgY2hhbm5lbHMgZnJvbSBYIHRleGVscyBmb3IgeEMke2NvbEluZGV4fS4gSWYgcGFkTGVmdCBpc1xuICAgICAgICAgIC8vIGV2ZW4sIHhDJHtjb2xJbmRleCArMX0gaXMgc2ltcGx5IHRoZSB6dyBjaGFubmVscyBvZiB0ZXhlbHMgd2UndmVcbiAgICAgICAgICAvLyBhbHJlYWR5IHNhbXBsZWQuIEJ1dCBpZiBwYWRMZWZ0IGlzIG9kZCwgeEN7JGMgKyAxfS56dyB3aWxsXG4gICAgICAgICAgLy8gbmVlZCB0byBjb21lIGZyb20gdGhlIHh5IGNoYW5uZWxzIG9mIGEgbmV3IHRleGVsLCBoZW5jZSB0aGUgYFxuICAgICAgICAgIC8vIHZlYzRcbiAgICAgICAgICAvLyBmaW5hbGAgaW5pdGlhbGl6ZWQgYmVsb3cuXG4gICAgICAgICAgaWYgKHBhZExlZnQgJSAyID09PSAxKSB7XG4gICAgICAgICAgICBtYWluTG9vcCArPSBgXG4gICAgICAgICAgICAgICAgeENPZmZzZXQgPSB4QyArIDEgLSBzdHJpZGVzWzFdO1xuICAgICAgICAgICAgICAgIGlmKHhDT2Zmc2V0ID49IDAgJiYgeENPZmZzZXQgPCBpbkRpbXNbMV0gJiYgeFRleGVsQyR7XG4gICAgICAgICAgICAgICAgY29sSW5kZXh9UmVhZHkgPT0gMCkge1xuICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXh9ID0gZ2V0WChiYXRjaCwgeFIsIHhDT2Zmc2V0LCBkMSk7XG4gICAgICAgICAgICAgICAgICAvLyBOZWVkIHRvIG1hbnVhbGx5IGNsZWFyIHVudXNlZCBjaGFubmVscyBpbiBjYXNlXG4gICAgICAgICAgICAgICAgICAvLyB3ZSdyZSByZWFkaW5nIGZyb20gcmVjeWNsZWQgdGV4dHVyZS5cbiAgICAgICAgICAgICAgICAgIGlmICh4Q09mZnNldCArIDEgPj0gaW5EaW1zWzFdKSB7XG4gICAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4fS56dyA9IHZlYzIoMC4wKTtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4fVJlYWR5ID0gMTtcbiAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICBpZih4QyArIDEgPj0gMCAmJiB4QyArIDEgPCBpbkRpbXNbMV0gJiYgeFRleGVsQyR7XG4gICAgICAgICAgICAgICAgY29sSW5kZXggKyAxfVJlYWR5ID09IDApIHtcbiAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4ICsgMX0gPSBnZXRYKGJhdGNoLCB4UiwgeEMgKyAxLCBkMSk7XG4gICAgICAgICAgICAgICAgICAvLyBOZWVkIHRvIG1hbnVhbGx5IGNsZWFyIHVudXNlZCBjaGFubmVscyBpbiBjYXNlXG4gICAgICAgICAgICAgICAgICAvLyB3ZSdyZSByZWFkaW5nIGZyb20gcmVjeWNsZWQgdGV4dHVyZS5cbiAgICAgICAgICAgICAgICAgIGlmICh4QyArIDIgPj0gaW5EaW1zWzFdKSB7XG4gICAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4ICsgMX0uencgPSB2ZWMyKDAuMCk7XG4gICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICB4VGV4ZWxDJHtjb2xJbmRleCArIDF9UmVhZHkgPSAxO1xuICAgICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICAgIHhDJHtjb2xJbmRleH0gPSB2ZWM0KHhUZXhlbEMke2NvbEluZGV4fS56dywgeFRleGVsQyR7XG4gICAgICAgICAgICAgICAgY29sSW5kZXggKyAxfS56dyk7XG4gICAgICAgICAgICAgIGA7XG5cbiAgICAgICAgICAgIGlmIChjb2xJbmRleCArIDEgPCBmaWx0ZXJXaWR0aCkge1xuICAgICAgICAgICAgICBtYWluTG9vcCArPSBgXG4gICAgICAgICAgICAgICAgICBmaW5hbCA9IHZlYzQoMC4wKTtcbiAgICAgICAgICAgICAgICAgIHhDT2Zmc2V0ID0geEMgKyAxICsgc3RyaWRlc1sxXTtcbiAgICAgICAgICAgICAgICAgIGlmKHhDT2Zmc2V0ID49IDAgJiYgeENPZmZzZXQgPCBpbkRpbXNbMV0pIHtcbiAgICAgICAgICAgICAgICAgICAgZmluYWwgPSBnZXRYKGJhdGNoLCB4UiwgeENPZmZzZXQsIGQxKTtcbiAgICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICAgIHhDJHtjb2xJbmRleCArIDF9ID0gdmVjNCh4VGV4ZWxDJHtjb2xJbmRleCArIDF9Lnh5LCBmaW5hbC54eSk7XG4gICAgICAgICAgICAgICAgYDtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgbWFpbkxvb3AgKz0gYFxuICAgICAgICAgICAgICAgIGlmKHhDID49IDAgJiYgeEMgPCBpbkRpbXNbMV0gJiYgeFRleGVsQyR7Y29sSW5kZXh9UmVhZHkgPT0gMCkge1xuICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXh9ID0gZ2V0WChiYXRjaCwgeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgICAgICBpZiAoeEMgKyAxID49IGluRGltc1sxXSkge1xuICAgICAgICAgICAgICAgICAgICB4VGV4ZWxDJHtjb2xJbmRleH0uencgPSB2ZWMyKDAuMCk7XG4gICAgICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgICAgICB4VGV4ZWxDJHtjb2xJbmRleH1SZWFkeSA9IDE7XG4gICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgeENPZmZzZXQgPSB4QyArIHN0cmlkZXNbMV07XG4gICAgICAgICAgICAgICAgaWYoeENPZmZzZXQgPj0gMCAmJiB4Q09mZnNldCA8IGluRGltc1sxXSAmJiB4VGV4ZWxDJHtcbiAgICAgICAgICAgICAgICBjb2xJbmRleCArIDF9UmVhZHkgPT0gMCkge1xuICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXggKyAxfSA9IGdldFgoYmF0Y2gsIHhSLCB4Q09mZnNldCwgZDEpO1xuICAgICAgICAgICAgICAgICAgaWYgKHhDT2Zmc2V0ICsgMSA+PSBpbkRpbXNbMV0pIHtcbiAgICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXggKyAxfS56dyA9IHZlYzIoMC4pO1xuICAgICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAgICAgeFRleGVsQyR7Y29sSW5kZXggKyAxfVJlYWR5ID0gMTtcbiAgICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgICB4QyR7Y29sSW5kZXh9ID0gdmVjNChcbiAgICAgICAgICAgICAgICAgIHhUZXhlbEMke2NvbEluZGV4fS54eSwgeFRleGVsQyR7Y29sSW5kZXggKyAxfS54eSk7XG4gICAgICAgICAgICAgIGA7XG5cbiAgICAgICAgICAgIGlmIChjb2xJbmRleCArIDEgPCBmaWx0ZXJXaWR0aCkge1xuICAgICAgICAgICAgICBtYWluTG9vcCArPSBgXG4gICAgICAgICAgICAgICAgICB4QyR7Y29sSW5kZXggKyAxfSA9IHZlYzQoeFRleGVsQyR7Y29sSW5kZXh9Lnp3LCB4VGV4ZWxDJHtcbiAgICAgICAgICAgICAgICAgIGNvbEluZGV4ICsgMX0uencpO1xuICAgICAgICAgICAgICAgIGA7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIC8vIGxvY2FsaXplIHRoZSBkb3RQcm9kIGFjY3VtdWxhdGlvbiB3aXRoaW4gdGhlIGxvb3AsIHRoZSB0aGVvcnkgaXMgZm9yXG4gICAgICAvLyBHUFUgd2l0aCBsaW1pdGVkIGNhY2hlLCBhY2N1bXVsYXRlIHN1bSBhY3Jvc3MgbGFyZ2UgYW1vdW50IG9mXG4gICAgICAvLyB2ZXJpYWJsZXMgd2lsbCBjYXVzZSBsb3RzIG9mIGNhY2hlIG1pc3Nlcy4gKGkuZS4gNXg1IGZpbHRlciB3aWxsIGhhdmVcbiAgICAgIC8vIDUwIHZhcmlhYmxlcylcbiAgICAgIGlmIChjb2xJbmRleCA8IGZpbHRlcldpZHRoKSB7XG4gICAgICAgIG1haW5Mb29wICs9IGBcbiAgICAgICAgICAgIHdUZXhlbCA9IGdldFcociwgJHtjb2xJbmRleH0sIGQxLCBxKTtcbiAgICAgICAgICAgIGRvdFByb2QgKz0geEMke2NvbEluZGV4fSAqIHZlYzQod1RleGVsLnh6LCB3VGV4ZWwueHopO1xuICAgICAgICAgIGA7XG5cbiAgICAgICAgaWYgKGNvbEluZGV4ICsgMSA8IGZpbHRlcldpZHRoKSB7XG4gICAgICAgICAgbWFpbkxvb3AgKz0gYFxuICAgICAgICAgICAgICB3VGV4ZWwgPSBnZXRXKHIsICR7Y29sSW5kZXggKyAxfSwgZDEsIHEpO1xuICAgICAgICAgICAgICBkb3RQcm9kICs9IHhDJHtjb2xJbmRleCArIDF9ICogdmVjNCh3VGV4ZWwueHosIHdUZXhlbC54eik7XG4gICAgICAgICAgICBgO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIG1haW5Mb29wICs9IGBcbiAgICB9XG4gIGA7XG4gICAgbWFpbkxvb3AgKz0gYFxuICAgICAgfVxuICAgIGA7XG5cbiAgICBsZXQgYWN0aXZhdGlvblNuaXBwZXQgPSAnJywgYXBwbHlBY3RpdmF0aW9uU25pcHBldCA9ICcnO1xuICAgIGlmIChhY3RpdmF0aW9uKSB7XG4gICAgICBpZiAoaGFzUHJlbHVBY3RpdmF0aW9uKSB7XG4gICAgICAgIGFjdGl2YXRpb25TbmlwcGV0ID0gYHZlYzQgYWN0aXZhdGlvbih2ZWM0IGEpIHtcbiAgICAgICAgICB2ZWM0IGIgPSBnZXRQcmVsdUFjdGl2YXRpb25XZWlnaHRzQXRPdXRDb29yZHMoKTtcbiAgICAgICAgICAke2FjdGl2YXRpb259XG4gICAgICAgIH1gO1xuICAgICAgfSBlbHNlIGlmIChoYXNMZWFreVJlbHVBbHBoYSkge1xuICAgICAgICBhY3RpdmF0aW9uU25pcHBldCA9IGB2ZWM0IGFjdGl2YXRpb24odmVjNCBhKSB7XG4gICAgICAgICAgdmVjNCBiID0gZ2V0TGVha3lyZWx1QWxwaGFBdE91dENvb3JkcygpO1xuICAgICAgICAgICR7YWN0aXZhdGlvbn1cbiAgICAgICAgfWA7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBhY3RpdmF0aW9uU25pcHBldCA9IGB2ZWM0IGFjdGl2YXRpb24odmVjNCB4KSB7XG4gICAgICAgICAgJHthY3RpdmF0aW9ufVxuICAgICAgICB9YDtcbiAgICAgIH1cblxuICAgICAgYXBwbHlBY3RpdmF0aW9uU25pcHBldCA9IGByZXN1bHQgPSBhY3RpdmF0aW9uKHJlc3VsdCk7YDtcbiAgICB9XG5cbiAgICBjb25zdCBhZGRCaWFzU25pcHBldCA9IGFkZEJpYXMgPyAncmVzdWx0ICs9IGdldEJpYXNBdE91dENvb3JkcygpOycgOiAnJztcbiAgICBpZiAoYWRkQmlhcykge1xuICAgICAgdGhpcy52YXJpYWJsZU5hbWVzLnB1c2goJ2JpYXMnKTtcbiAgICB9XG5cbiAgICBpZiAoaGFzUHJlbHVBY3RpdmF0aW9uKSB7XG4gICAgICB0aGlzLnZhcmlhYmxlTmFtZXMucHVzaCgncHJlbHVBY3RpdmF0aW9uV2VpZ2h0cycpO1xuICAgIH1cbiAgICBpZiAoaGFzTGVha3lSZWx1QWxwaGEpIHtcbiAgICAgIHRoaXMudmFyaWFibGVOYW1lcy5wdXNoKCdsZWFreXJlbHVBbHBoYScpO1xuICAgIH1cblxuICAgIHRoaXMudXNlckNvZGUgPSBgXG4gICAgICAke2FjdGl2YXRpb25TbmlwcGV0fVxuXG4gICAgICB2b2lkIG1haW4oKSB7XG4gICAgICAgIGl2ZWM0IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICBpbnQgYmF0Y2ggPSBjb29yZHMueDtcbiAgICAgICAgaXZlYzIgeFJDQ29ybmVyID0gY29vcmRzLnl6ICogc3RyaWRlcyAtIHBhZHM7XG4gICAgICAgIGludCBkMiA9IGNvb3Jkcy53O1xuICAgICAgICBpbnQgZDEgPSBkMiAvICR7Y2hhbm5lbE11bH07XG4gICAgICAgIGludCBxID0gZDIgLSBkMSAqICR7Y2hhbm5lbE11bH07XG4gICAgICAgIGludCB4UkNvcm5lciA9IHhSQ0Nvcm5lci54O1xuICAgICAgICBpbnQgeENDb3JuZXIgPSB4UkNDb3JuZXIueTtcblxuICAgICAgICAvL2ludGlhbGl6ZSBkb3RQcm9kIHdpdGggYSBzbWFsbCBlcHNpbG9uIHNlZW1zIHRvIHJlZHVjZSBHUFUgYWNjdXJhY3kgbG9zcy5cbiAgICAgICAgdmVjNCBkb3RQcm9kID0gdmVjNCgwLjAwMDAwMDAwMDAwMDAwMSk7XG5cbiAgICAgICAgJHttYWluTG9vcH1cblxuICAgICAgICB2ZWM0IHJlc3VsdCA9IGRvdFByb2QgLSB2ZWM0KDAuMDAwMDAwMDAwMDAwMDAxKTtcbiAgICAgICAgJHthZGRCaWFzU25pcHBldH1cbiAgICAgICAgJHthcHBseUFjdGl2YXRpb25TbmlwcGV0fVxuICAgICAgICBzZXRPdXRwdXQocmVzdWx0KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG59XG4iXX0=