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
import { backend_util, buffer, MaxPool3DGrad } from '@tensorflow/tfjs-core';
import { assertNotComplex } from '../cpu_util';
import { maxPool3dPositions } from '../utils/pool_utils';
export function maxPool3DGrad(args) {
    const { inputs, backend, attrs } = args;
    const { dy, input } = inputs;
    const { filterSize, strides, pad, dimRoundingMode } = attrs;
    assertNotComplex([dy, input], 'maxPool3DGrad');
    const convInfo = backend_util.computePool3DInfo(input.shape, filterSize, strides, 1 /* dilations */, pad, dimRoundingMode);
    const inputBuf = backend.bufferSync(input);
    const maxPosBuf = maxPool3dPositions(inputBuf, convInfo);
    const strideDepth = convInfo.strideDepth;
    const strideHeight = convInfo.strideHeight;
    const strideWidth = convInfo.strideWidth;
    const dilationDepth = convInfo.dilationDepth;
    const dilationHeight = convInfo.dilationHeight;
    const dilationWidth = convInfo.dilationWidth;
    const effectiveFilterDepth = convInfo.effectiveFilterDepth;
    const effectiveFilterHeight = convInfo.effectiveFilterHeight;
    const effectiveFilterWidth = convInfo.effectiveFilterWidth;
    const padFront = effectiveFilterDepth - 1 - convInfo.padInfo.front;
    const padLeft = effectiveFilterWidth - 1 - convInfo.padInfo.left;
    const padTop = effectiveFilterHeight - 1 - convInfo.padInfo.top;
    const dx = buffer(input.shape, 'float32');
    const dyBuf = backend.bufferSync(dy);
    for (let batch = 0; batch < convInfo.batchSize; ++batch) {
        for (let channel = 0; channel < convInfo.inChannels; ++channel) {
            for (let dxDepth = 0; dxDepth < convInfo.inDepth; ++dxDepth) {
                for (let dxRow = 0; dxRow < convInfo.inHeight; ++dxRow) {
                    for (let dxCol = 0; dxCol < convInfo.inWidth; ++dxCol) {
                        // Shader code begins
                        const dyDepthCorner = dxDepth - padFront;
                        const dyRowCorner = dxRow - padTop;
                        const dyColCorner = dxCol - padLeft;
                        let dotProd = 0;
                        for (let wDepth = 0; wDepth < effectiveFilterDepth; wDepth += dilationDepth) {
                            const dyDepth = (dyDepthCorner + wDepth) / strideDepth;
                            if (dyDepth < 0 || dyDepth >= convInfo.outDepth ||
                                Math.floor(dyDepth) !== dyDepth) {
                                continue;
                            }
                            for (let wRow = 0; wRow < effectiveFilterHeight; wRow += dilationHeight) {
                                const dyRow = (dyRowCorner + wRow) / strideHeight;
                                if (dyRow < 0 || dyRow >= convInfo.outHeight ||
                                    Math.floor(dyRow) !== dyRow) {
                                    continue;
                                }
                                for (let wCol = 0; wCol < effectiveFilterWidth; wCol += dilationWidth) {
                                    const dyCol = (dyColCorner + wCol) / strideWidth;
                                    if (dyCol < 0 || dyCol >= convInfo.outWidth ||
                                        Math.floor(dyCol) !== dyCol) {
                                        continue;
                                    }
                                    const maxPos = effectiveFilterDepth * effectiveFilterHeight *
                                        effectiveFilterWidth -
                                        1 -
                                        maxPosBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                                    const curPos = wDepth * effectiveFilterHeight * effectiveFilterWidth +
                                        wRow * effectiveFilterWidth + wCol;
                                    const mask = maxPos === curPos ? 1 : 0;
                                    if (mask === 0) {
                                        continue;
                                    }
                                    const pixel = dyBuf.get(batch, dyDepth, dyRow, dyCol, channel);
                                    dotProd += pixel * mask;
                                }
                            }
                        }
                        dx.set(dotProd, batch, dxDepth, dxRow, dxCol, channel);
                    }
                }
            }
        }
    }
    return backend.makeTensorInfo(dx.shape, dx.dtype, dx.values);
}
export const maxPool3DGradConfig = {
    kernelName: MaxPool3DGrad,
    backendName: 'cpu',
    kernelFunc: maxPool3DGrad
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiTWF4UG9vbDNER3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC1jcHUvc3JjL2tlcm5lbHMvTWF4UG9vbDNER3JhZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUFFLE1BQU0sRUFBNEIsYUFBYSxFQUFzRCxNQUFNLHVCQUF1QixDQUFDO0FBR3pKLE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUM3QyxPQUFPLEVBQUMsa0JBQWtCLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUV2RCxNQUFNLFVBQVUsYUFBYSxDQUFDLElBSTdCO0lBQ0MsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxFQUFFLEVBQUUsS0FBSyxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQzNCLE1BQU0sRUFBQyxVQUFVLEVBQUUsT0FBTyxFQUFFLEdBQUcsRUFBRSxlQUFlLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFMUQsZ0JBQWdCLENBQUMsQ0FBQyxFQUFFLEVBQUUsS0FBSyxDQUFDLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFFL0MsTUFBTSxRQUFRLEdBQUcsWUFBWSxDQUFDLGlCQUFpQixDQUMzQyxLQUFLLENBQUMsS0FBaUQsRUFBRSxVQUFVLEVBQ25FLE9BQU8sRUFBRSxDQUFDLENBQUMsZUFBZSxFQUFFLEdBQUcsRUFBRSxlQUFlLENBQUMsQ0FBQztJQUV0RCxNQUFNLFFBQVEsR0FBRyxPQUFPLENBQUMsVUFBVSxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQzNDLE1BQU0sU0FBUyxHQUFHLGtCQUFrQixDQUFDLFFBQVEsRUFBRSxRQUFRLENBQUMsQ0FBQztJQUN6RCxNQUFNLFdBQVcsR0FBRyxRQUFRLENBQUMsV0FBVyxDQUFDO0lBQ3pD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QUVELE9BQU8sT0FBTyxDQUFDLGNBQWMsQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLEVBQUUsQ0FBQyxLQUFLLEVBQUUsRUFBRSxDQUFDLE1BQU0sQ0FBQyxDQUFDO0FBQy9ELENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxtQkFBbUIsR0FBaUI7SUFDL0MsVUFBVSxFQUFFLGFBQWE7SUFDekIsV0FBVyxFQUFFLEtBQUs7SUFDbEIsVUFBVSxFQUFFLGFBQWlDO0NBQzlDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7YmFja2VuZF91dGlsLCBidWZmZXIsIEtlcm5lbENvbmZpZywgS2VybmVsRnVuYywgTWF4UG9vbDNER3JhZCwgTWF4UG9vbDNER3JhZEF0dHJzLCBNYXhQb29sM0RHcmFkSW5wdXRzLCBUZW5zb3JJbmZvfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge01hdGhCYWNrZW5kQ1BVfSBmcm9tICcuLi9iYWNrZW5kX2NwdSc7XG5pbXBvcnQge2Fzc2VydE5vdENvbXBsZXh9IGZyb20gJy4uL2NwdV91dGlsJztcbmltcG9ydCB7bWF4UG9vbDNkUG9zaXRpb25zfSBmcm9tICcuLi91dGlscy9wb29sX3V0aWxzJztcblxuZXhwb3J0IGZ1bmN0aW9uIG1heFBvb2wzREdyYWQoYXJnczoge1xuICBpbnB1dHM6IE1heFBvb2wzREdyYWRJbnB1dHMsXG4gIGJhY2tlbmQ6IE1hdGhCYWNrZW5kQ1BVLFxuICBhdHRyczogTWF4UG9vbDNER3JhZEF0dHJzXG59KTogVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtkeSwgaW5wdXR9ID0gaW5wdXRzO1xuICBjb25zdCB7ZmlsdGVyU2l6ZSwgc3RyaWRlcywgcGFkLCBkaW1Sb3VuZGluZ01vZGV9ID0gYXR0cnM7XG5cbiAgYXNzZXJ0Tm90Q29tcGxleChbZHksIGlucHV0XSwgJ21heFBvb2wzREdyYWQnKTtcblxuICBjb25zdCBjb252SW5mbyA9IGJhY2tlbmRfdXRpbC5jb21wdXRlUG9vbDNESW5mbyhcbiAgICAgIGlucHV0LnNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpbHRlclNpemUsXG4gICAgICBzdHJpZGVzLCAxIC8qIGRpbGF0aW9ucyAqLywgcGFkLCBkaW1Sb3VuZGluZ01vZGUpO1xuXG4gIGNvbnN0IGlucHV0QnVmID0gYmFja2VuZC5idWZmZXJTeW5jKGlucHV0KTtcbiAgY29uc3QgbWF4UG9zQnVmID0gbWF4UG9vbDNkUG9zaXRpb25zKGlucHV0QnVmLCBjb252SW5mbyk7XG4gIGNvbnN0IHN0cmlkZURlcHRoID0gY29udkluZm8uc3RyaWRlRGVwdGg7XG4gIGNvbnN0IHN0cmlkZUhlaWdodCA9IGNvbnZJbmZvLnN0cmlkZUhlaWdodDtcbiAgY29uc3Qgc3RyaWRlV2lkdGggPSBjb252SW5mby5zdHJpZGVXaWR0aDtcbiAgY29uc3QgZGlsYXRpb25EZXB0aCA9IGNvbnZJbmZvLmRpbGF0aW9uRGVwdGg7XG4gIGNvbnN0IGRpbGF0aW9uSGVpZ2h0ID0gY29udkluZm8uZGlsYXRpb25IZWlnaHQ7XG4gIGNvbnN0IGRpbGF0aW9uV2lkdGggPSBjb252SW5mby5kaWxhdGlvbldpZHRoO1xuICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJEZXB0aCA9IGNvbnZJbmZvLmVmZmVjdGl2ZUZpbHRlckRlcHRoO1xuICBjb25zdCBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgPSBjb252SW5mby5lZmZlY3RpdmVGaWx0ZXJIZWlnaHQ7XG4gIGNvbnN0IGVmZmVjdGl2ZUZpbHRlcldpZHRoID0gY29udkluZm8uZWZmZWN0aXZlRmlsdGVyV2lkdGg7XG4gIGNvbnN0IHBhZEZyb250ID0gZWZmZWN0aXZlRmlsdGVyRGVwdGggLSAxIC0gY29udkluZm8ucGFkSW5mby5mcm9udDtcbiAgY29uc3QgcGFkTGVmdCA9IGVmZmVjdGl2ZUZpbHRlcldpZHRoIC0gMSAtIGNvbnZJbmZvLnBhZEluZm8ubGVmdDtcbiAgY29uc3QgcGFkVG9wID0gZWZmZWN0aXZlRmlsdGVySGVpZ2h0IC0gMSAtIGNvbnZJbmZvLnBhZEluZm8udG9wO1xuICBjb25zdCBkeCA9IGJ1ZmZlcihpbnB1dC5zaGFwZSwgJ2Zsb2F0MzInKTtcblxuICBjb25zdCBkeUJ1ZiA9IGJhY2tlbmQuYnVmZmVyU3luYyhkeSk7XG5cbiAgZm9yIChsZXQgYmF0Y2ggPSAwOyBiYXRjaCA8IGNvbnZJbmZvLmJhdGNoU2l6ZTsgKytiYXRjaCkge1xuICAgIGZvciAobGV0IGNoYW5uZWwgPSAwOyBjaGFubmVsIDwgY29udkluZm8uaW5DaGFubmVsczsgKytjaGFubmVsKSB7XG4gICAgICBmb3IgKGxldCBkeERlcHRoID0gMDsgZHhEZXB0aCA8IGNvbnZJbmZvLmluRGVwdGg7ICsrZHhEZXB0aCkge1xuICAgICAgICBmb3IgKGxldCBkeFJvdyA9IDA7IGR4Um93IDwgY29udkluZm8uaW5IZWlnaHQ7ICsrZHhSb3cpIHtcbiAgICAgICAgICBmb3IgKGxldCBkeENvbCA9IDA7IGR4Q29sIDwgY29udkluZm8uaW5XaWR0aDsgKytkeENvbCkge1xuICAgICAgICAgICAgLy8gU2hhZGVyIGNvZGUgYmVnaW5zXG4gICAgICAgICAgICBjb25zdCBkeURlcHRoQ29ybmVyID0gZHhEZXB0aCAtIHBhZEZyb250O1xuICAgICAgICAgICAgY29uc3QgZHlSb3dDb3JuZXIgPSBkeFJvdyAtIHBhZFRvcDtcbiAgICAgICAgICAgIGNvbnN0IGR5Q29sQ29ybmVyID0gZHhDb2wgLSBwYWRMZWZ0O1xuICAgICAgICAgICAgbGV0IGRvdFByb2QgPSAwO1xuICAgICAgICAgICAgZm9yIChsZXQgd0RlcHRoID0gMDsgd0RlcHRoIDwgZWZmZWN0aXZlRmlsdGVyRGVwdGg7XG4gICAgICAgICAgICAgICAgIHdEZXB0aCArPSBkaWxhdGlvbkRlcHRoKSB7XG4gICAgICAgICAgICAgIGNvbnN0IGR5RGVwdGggPSAoZHlEZXB0aENvcm5lciArIHdEZXB0aCkgLyBzdHJpZGVEZXB0aDtcbiAgICAgICAgICAgICAgaWYgKGR5RGVwdGggPCAwIHx8IGR5RGVwdGggPj0gY29udkluZm8ub3V0RGVwdGggfHxcbiAgICAgICAgICAgICAgICAgIE1hdGguZmxvb3IoZHlEZXB0aCkgIT09IGR5RGVwdGgpIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICBmb3IgKGxldCB3Um93ID0gMDsgd1JvdyA8IGVmZmVjdGl2ZUZpbHRlckhlaWdodDtcbiAgICAgICAgICAgICAgICAgICB3Um93ICs9IGRpbGF0aW9uSGVpZ2h0KSB7XG4gICAgICAgICAgICAgICAgY29uc3QgZHlSb3cgPSAoZHlSb3dDb3JuZXIgKyB3Um93KSAvIHN0cmlkZUhlaWdodDtcbiAgICAgICAgICAgICAgICBpZiAoZHlSb3cgPCAwIHx8IGR5Um93ID49IGNvbnZJbmZvLm91dEhlaWdodCB8fFxuICAgICAgICAgICAgICAgICAgICBNYXRoLmZsb29yKGR5Um93KSAhPT0gZHlSb3cpIHtcbiAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgICBmb3IgKGxldCB3Q29sID0gMDsgd0NvbCA8IGVmZmVjdGl2ZUZpbHRlcldpZHRoO1xuICAgICAgICAgICAgICAgICAgICAgd0NvbCArPSBkaWxhdGlvbldpZHRoKSB7XG4gICAgICAgICAgICAgICAgICBjb25zdCBkeUNvbCA9IChkeUNvbENvcm5lciArIHdDb2wpIC8gc3RyaWRlV2lkdGg7XG4gICAgICAgICAgICAgICAgICBpZiAoZHlDb2wgPCAwIHx8IGR5Q29sID49IGNvbnZJbmZvLm91dFdpZHRoIHx8XG4gICAgICAgICAgICAgICAgICAgICAgTWF0aC5mbG9vcihkeUNvbCkgIT09IGR5Q29sKSB7XG4gICAgICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgICAgICBjb25zdCBtYXhQb3MgPSBlZmZlY3RpdmVGaWx0ZXJEZXB0aCAqIGVmZmVjdGl2ZUZpbHRlckhlaWdodCAqXG4gICAgICAgICAgICAgICAgICAgICAgICAgIGVmZmVjdGl2ZUZpbHRlcldpZHRoIC1cbiAgICAgICAgICAgICAgICAgICAgICAxIC1cbiAgICAgICAgICAgICAgICAgICAgICAobWF4UG9zQnVmLmdldChiYXRjaCwgZHlEZXB0aCwgZHlSb3csIGR5Q29sLCBjaGFubmVsKSBhc1xuICAgICAgICAgICAgICAgICAgICAgICBudW1iZXIpO1xuICAgICAgICAgICAgICAgICAgY29uc3QgY3VyUG9zID1cbiAgICAgICAgICAgICAgICAgICAgICB3RGVwdGggKiBlZmZlY3RpdmVGaWx0ZXJIZWlnaHQgKiBlZmZlY3RpdmVGaWx0ZXJXaWR0aCArXG4gICAgICAgICAgICAgICAgICAgICAgd1JvdyAqIGVmZmVjdGl2ZUZpbHRlcldpZHRoICsgd0NvbDtcblxuICAgICAgICAgICAgICAgICAgY29uc3QgbWFzayA9IG1heFBvcyA9PT0gY3VyUG9zID8gMSA6IDA7XG4gICAgICAgICAgICAgICAgICBpZiAobWFzayA9PT0gMCkge1xuICAgICAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgICAgICAgY29uc3QgcGl4ZWwgPVxuICAgICAgICAgICAgICAgICAgICAgIGR5QnVmLmdldChiYXRjaCwgZHlEZXB0aCwgZHlSb3csIGR5Q29sLCBjaGFubmVsKTtcbiAgICAgICAgICAgICAgICAgIGRvdFByb2QgKz0gcGl4ZWwgKiBtYXNrO1xuICAgICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZHguc2V0KGRvdFByb2QsIGJhdGNoLCBkeERlcHRoLCBkeFJvdywgZHhDb2wsIGNoYW5uZWwpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIHJldHVybiBiYWNrZW5kLm1ha2VUZW5zb3JJbmZvKGR4LnNoYXBlLCBkeC5kdHlwZSwgZHgudmFsdWVzKTtcbn1cblxuZXhwb3J0IGNvbnN0IG1heFBvb2wzREdyYWRDb25maWc6IEtlcm5lbENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogTWF4UG9vbDNER3JhZCxcbiAgYmFja2VuZE5hbWU6ICdjcHUnLFxuICBrZXJuZWxGdW5jOiBtYXhQb29sM0RHcmFkIGFzIHt9IGFzIEtlcm5lbEZ1bmNcbn07XG4iXX0=