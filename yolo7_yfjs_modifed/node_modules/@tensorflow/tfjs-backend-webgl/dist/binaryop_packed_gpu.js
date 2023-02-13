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
import { backend_util, util } from '@tensorflow/tfjs-core';
import { useShapeUniforms } from './gpgpu_math';
import { getChannels } from './packing_util';
import { getCoordsDataType } from './shader_compiler';
export const CHECK_NAN_SNIPPET = `
  result.r = isNaN.r > 0. ? NAN : result.r;
  result.g = isNaN.g > 0. ? NAN : result.g;
  result.b = isNaN.b > 0. ? NAN : result.b;
  result.a = isNaN.a > 0. ? NAN : result.a;
`;
export const ELU_DER = `
  vec4 bGTEZero = vec4(greaterThanEqual(b, vec4(0.)));
  return (bGTEZero * a) + ((vec4(1.0) - bGTEZero) * (a * (b + vec4(1.0))));
`;
export const NOT_EQUAL = `
  return vec4(notEqual(a, b));
`;
export class BinaryOpPackedProgram {
    constructor(op, aShape, bShape, checkOutOfBounds = false) {
        this.variableNames = ['A', 'B'];
        this.supportsBroadcasting = true;
        this.packedInputs = true;
        this.packedOutput = true;
        this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
        const rank = this.outputShape.length;
        this.enableShapeUniforms = useShapeUniforms(rank);
        let checkOutOfBoundsString = '';
        if (checkOutOfBounds) {
            if (rank === 0 || util.sizeFromShape(this.outputShape) === 1) {
                checkOutOfBoundsString = `
          result.y = 0.;
          result.z = 0.;
          result.w = 0.;
        `;
            }
            else {
                const dtype = getCoordsDataType(rank);
                checkOutOfBoundsString = `
          ${dtype} coords = getOutputCoords();
        `;
                if (rank === 1) {
                    if (this.enableShapeUniforms) {
                        checkOutOfBoundsString += `
            result.y = (coords + 1) >= outShape ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;
                    }
                    else {
                        checkOutOfBoundsString += `
            result.y = (coords + 1) >= ${this.outputShape[0]} ? 0. : result.y;
            result.z = 0.;
            result.w = 0.;
          `;
                    }
                }
                else {
                    const channels = getChannels('coords', rank);
                    if (this.enableShapeUniforms) {
                        checkOutOfBoundsString += `
            bool nextRowOutOfBounds =
              (${channels[rank - 2]} + 1) >= outShape[${rank} - 2];
            bool nextColOutOfBounds =
              (${channels[rank - 1]} + 1) >= outShape[${rank} - 1];
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `;
                    }
                    else {
                        checkOutOfBoundsString += `
            bool nextRowOutOfBounds =
              (${channels[rank - 2]} + 1) >= ${this.outputShape[rank - 2]};
            bool nextColOutOfBounds =
              (${channels[rank - 1]} + 1) >= ${this.outputShape[rank - 1]};
            result.y = nextColOutOfBounds ? 0. : result.y;
            result.z = nextRowOutOfBounds ? 0. : result.z;
            result.w = nextColOutOfBounds || nextRowOutOfBounds ? 0. : result.w;
          `;
                    }
                }
            }
        }
        this.userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${op}
      }

      void main() {
        vec4 a = getAAtOutCoords();
        vec4 b = getBAtOutCoords();

        vec4 result = binaryOperation(a, b);
        ${checkOutOfBoundsString}

        setOutput(result);
      }
    `;
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmluYXJ5b3BfcGFja2VkX2dwdS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3RmanMtYmFja2VuZC13ZWJnbC9zcmMvYmluYXJ5b3BfcGFja2VkX2dwdS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsWUFBWSxFQUFFLElBQUksRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBRXpELE9BQU8sRUFBZSxnQkFBZ0IsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUM1RCxPQUFPLEVBQUMsV0FBVyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDM0MsT0FBTyxFQUFDLGlCQUFpQixFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFFcEQsTUFBTSxDQUFDLE1BQU0saUJBQWlCLEdBQUc7Ozs7O0NBS2hDLENBQUM7QUFFRixNQUFNLENBQUMsTUFBTSxPQUFPLEdBQUc7OztDQUd0QixDQUFDO0FBRUYsTUFBTSxDQUFDLE1BQU0sU0FBUyxHQUFHOztDQUV4QixDQUFDO0FBRUYsTUFBTSxPQUFPLHFCQUFxQjtJQVNoQyxZQUNJLEVBQVUsRUFBRSxNQUFnQixFQUFFLE1BQWdCLEVBQzlDLGdCQUFnQixHQUFHLEtBQUs7UUFWNUIsa0JBQWEsR0FBRyxDQUFDLEdBQUcsRUFBRSxHQUFHLENBQUMsQ0FBQztRQUczQix5QkFBb0IsR0FBRyxJQUFJLENBQUM7UUFDNUIsaUJBQVksR0FBRyxJQUFJLENBQUM7UUFDcEIsaUJBQVksR0FBRyxJQUFJLENBQUM7UUFNbEIsSUFBSSxDQUFDLFdBQVcsR0FBRyxZQUFZLENBQUMsMEJBQTBCLENBQUMsTUFBTSxFQUFFLE1BQU0sQ0FBQyxDQUFDO1FBQzNFLE1BQU0sSUFBSSxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxDQUFDO1FBQ3JDLElBQUksQ0FBQyxtQkFBbUIsR0FBRyxnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUNsRCxJQUFJLHNCQUFzQixHQUFHLEVBQUUsQ0FBQztRQUNoQyxJQUFJLGdCQUFnQixFQUFFO1lBQ3BCLElBQUksSUFBSSxLQUFLLENBQUMsSUFBSSxJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsS0FBSyxDQUFDLEVBQUU7Z0JBQzVELHNCQUFzQixHQUFHOzs7O1NBSXhCLENBQUM7YUFDSDtpQkFBTTtnQkFDTCxNQUFNLEtBQUssR0FBRyxpQkFBaUIsQ0FBQyxJQUFJLENBQUMsQ0FBQztnQkFDdEMsc0JBQXNCLEdBQUc7WUFDckIsS0FBSztTQUNSLENBQUM7Z0JBQ0YsSUFBSSxJQUFJLEtBQUssQ0FBQyxFQUFFO29CQUNkLElBQUksSUFBSSxDQUFDLG1CQUFtQixFQUFFO3dCQUM1QixzQkFBc0IsSUFBSTs7OztXQUkzQixDQUFDO3FCQUNEO3lCQUFNO3dCQUNMLHNCQUFzQixJQUFJO3lDQUNHLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDOzs7V0FHakQsQ0FBQztxQkFDRDtpQkFDRjtxQkFBTTtvQkFDTCxNQUFNLFFBQVEsR0FBRyxXQUFXLENBQUMsUUFBUSxFQUFFLElBQUksQ0FBQyxDQUFDO29CQUM3QyxJQUFJLElBQUksQ0FBQyxtQkFBbUIsRUFBRTt3QkFDNUIsc0JBQXNCLElBQUk7O2lCQUVyQixRQUFRLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQyxxQkFBcUIsSUFBSTs7aUJBRTNDLFFBQVEsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDLHFCQUFxQixJQUFJOzs7O1dBSWpELENBQUM7cUJBQ0Q7eUJBQU07d0JBQ0wsc0JBQXNCLElBQUk7O2lCQUVyQixRQUFRLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQyxZQUFZLElBQUksQ0FBQyxXQUFXLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQzs7aUJBRXhELFFBQVEsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDLFlBQVksSUFBSSxDQUFDLFdBQVcsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDOzs7O1dBSTlELENBQUM7cUJBQ0Q7aUJBQ0Y7YUFDRjtTQUNGO1FBRUQsSUFBSSxDQUFDLFFBQVEsR0FBRzs7VUFFVixFQUFFOzs7Ozs7OztVQVFGLHNCQUFzQjs7OztLQUkzQixDQUFDO0lBQ0osQ0FBQztDQUNGIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2JhY2tlbmRfdXRpbCwgdXRpbH0gZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcblxuaW1wb3J0IHtHUEdQVVByb2dyYW0sIHVzZVNoYXBlVW5pZm9ybXN9IGZyb20gJy4vZ3BncHVfbWF0aCc7XG5pbXBvcnQge2dldENoYW5uZWxzfSBmcm9tICcuL3BhY2tpbmdfdXRpbCc7XG5pbXBvcnQge2dldENvb3Jkc0RhdGFUeXBlfSBmcm9tICcuL3NoYWRlcl9jb21waWxlcic7XG5cbmV4cG9ydCBjb25zdCBDSEVDS19OQU5fU05JUFBFVCA9IGBcbiAgcmVzdWx0LnIgPSBpc05hTi5yID4gMC4gPyBOQU4gOiByZXN1bHQucjtcbiAgcmVzdWx0LmcgPSBpc05hTi5nID4gMC4gPyBOQU4gOiByZXN1bHQuZztcbiAgcmVzdWx0LmIgPSBpc05hTi5iID4gMC4gPyBOQU4gOiByZXN1bHQuYjtcbiAgcmVzdWx0LmEgPSBpc05hTi5hID4gMC4gPyBOQU4gOiByZXN1bHQuYTtcbmA7XG5cbmV4cG9ydCBjb25zdCBFTFVfREVSID0gYFxuICB2ZWM0IGJHVEVaZXJvID0gdmVjNChncmVhdGVyVGhhbkVxdWFsKGIsIHZlYzQoMC4pKSk7XG4gIHJldHVybiAoYkdURVplcm8gKiBhKSArICgodmVjNCgxLjApIC0gYkdURVplcm8pICogKGEgKiAoYiArIHZlYzQoMS4wKSkpKTtcbmA7XG5cbmV4cG9ydCBjb25zdCBOT1RfRVFVQUwgPSBgXG4gIHJldHVybiB2ZWM0KG5vdEVxdWFsKGEsIGIpKTtcbmA7XG5cbmV4cG9ydCBjbGFzcyBCaW5hcnlPcFBhY2tlZFByb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWydBJywgJ0InXTtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuICBzdXBwb3J0c0Jyb2FkY2FzdGluZyA9IHRydWU7XG4gIHBhY2tlZElucHV0cyA9IHRydWU7XG4gIHBhY2tlZE91dHB1dCA9IHRydWU7XG4gIGVuYWJsZVNoYXBlVW5pZm9ybXM6IGJvb2xlYW47XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICBvcDogc3RyaW5nLCBhU2hhcGU6IG51bWJlcltdLCBiU2hhcGU6IG51bWJlcltdLFxuICAgICAgY2hlY2tPdXRPZkJvdW5kcyA9IGZhbHNlKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGJhY2tlbmRfdXRpbC5hc3NlcnRBbmRHZXRCcm9hZGNhc3RTaGFwZShhU2hhcGUsIGJTaGFwZSk7XG4gICAgY29uc3QgcmFuayA9IHRoaXMub3V0cHV0U2hhcGUubGVuZ3RoO1xuICAgIHRoaXMuZW5hYmxlU2hhcGVVbmlmb3JtcyA9IHVzZVNoYXBlVW5pZm9ybXMocmFuayk7XG4gICAgbGV0IGNoZWNrT3V0T2ZCb3VuZHNTdHJpbmcgPSAnJztcbiAgICBpZiAoY2hlY2tPdXRPZkJvdW5kcykge1xuICAgICAgaWYgKHJhbmsgPT09IDAgfHwgdXRpbC5zaXplRnJvbVNoYXBlKHRoaXMub3V0cHV0U2hhcGUpID09PSAxKSB7XG4gICAgICAgIGNoZWNrT3V0T2ZCb3VuZHNTdHJpbmcgPSBgXG4gICAgICAgICAgcmVzdWx0LnkgPSAwLjtcbiAgICAgICAgICByZXN1bHQueiA9IDAuO1xuICAgICAgICAgIHJlc3VsdC53ID0gMC47XG4gICAgICAgIGA7XG4gICAgICB9IGVsc2Uge1xuICAgICAgICBjb25zdCBkdHlwZSA9IGdldENvb3Jkc0RhdGFUeXBlKHJhbmspO1xuICAgICAgICBjaGVja091dE9mQm91bmRzU3RyaW5nID0gYFxuICAgICAgICAgICR7ZHR5cGV9IGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICBgO1xuICAgICAgICBpZiAocmFuayA9PT0gMSkge1xuICAgICAgICAgIGlmICh0aGlzLmVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICAgICAgICAgIGNoZWNrT3V0T2ZCb3VuZHNTdHJpbmcgKz0gYFxuICAgICAgICAgICAgcmVzdWx0LnkgPSAoY29vcmRzICsgMSkgPj0gb3V0U2hhcGUgPyAwLiA6IHJlc3VsdC55O1xuICAgICAgICAgICAgcmVzdWx0LnogPSAwLjtcbiAgICAgICAgICAgIHJlc3VsdC53ID0gMC47XG4gICAgICAgICAgYDtcbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgY2hlY2tPdXRPZkJvdW5kc1N0cmluZyArPSBgXG4gICAgICAgICAgICByZXN1bHQueSA9IChjb29yZHMgKyAxKSA+PSAke3RoaXMub3V0cHV0U2hhcGVbMF19ID8gMC4gOiByZXN1bHQueTtcbiAgICAgICAgICAgIHJlc3VsdC56ID0gMC47XG4gICAgICAgICAgICByZXN1bHQudyA9IDAuO1xuICAgICAgICAgIGA7XG4gICAgICAgICAgfVxuICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgIGNvbnN0IGNoYW5uZWxzID0gZ2V0Q2hhbm5lbHMoJ2Nvb3JkcycsIHJhbmspO1xuICAgICAgICAgIGlmICh0aGlzLmVuYWJsZVNoYXBlVW5pZm9ybXMpIHtcbiAgICAgICAgICAgIGNoZWNrT3V0T2ZCb3VuZHNTdHJpbmcgKz0gYFxuICAgICAgICAgICAgYm9vbCBuZXh0Um93T3V0T2ZCb3VuZHMgPVxuICAgICAgICAgICAgICAoJHtjaGFubmVsc1tyYW5rIC0gMl19ICsgMSkgPj0gb3V0U2hhcGVbJHtyYW5rfSAtIDJdO1xuICAgICAgICAgICAgYm9vbCBuZXh0Q29sT3V0T2ZCb3VuZHMgPVxuICAgICAgICAgICAgICAoJHtjaGFubmVsc1tyYW5rIC0gMV19ICsgMSkgPj0gb3V0U2hhcGVbJHtyYW5rfSAtIDFdO1xuICAgICAgICAgICAgcmVzdWx0LnkgPSBuZXh0Q29sT3V0T2ZCb3VuZHMgPyAwLiA6IHJlc3VsdC55O1xuICAgICAgICAgICAgcmVzdWx0LnogPSBuZXh0Um93T3V0T2ZCb3VuZHMgPyAwLiA6IHJlc3VsdC56O1xuICAgICAgICAgICAgcmVzdWx0LncgPSBuZXh0Q29sT3V0T2ZCb3VuZHMgfHwgbmV4dFJvd091dE9mQm91bmRzID8gMC4gOiByZXN1bHQudztcbiAgICAgICAgICBgO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBjaGVja091dE9mQm91bmRzU3RyaW5nICs9IGBcbiAgICAgICAgICAgIGJvb2wgbmV4dFJvd091dE9mQm91bmRzID1cbiAgICAgICAgICAgICAgKCR7Y2hhbm5lbHNbcmFuayAtIDJdfSArIDEpID49ICR7dGhpcy5vdXRwdXRTaGFwZVtyYW5rIC0gMl19O1xuICAgICAgICAgICAgYm9vbCBuZXh0Q29sT3V0T2ZCb3VuZHMgPVxuICAgICAgICAgICAgICAoJHtjaGFubmVsc1tyYW5rIC0gMV19ICsgMSkgPj0gJHt0aGlzLm91dHB1dFNoYXBlW3JhbmsgLSAxXX07XG4gICAgICAgICAgICByZXN1bHQueSA9IG5leHRDb2xPdXRPZkJvdW5kcyA/IDAuIDogcmVzdWx0Lnk7XG4gICAgICAgICAgICByZXN1bHQueiA9IG5leHRSb3dPdXRPZkJvdW5kcyA/IDAuIDogcmVzdWx0Lno7XG4gICAgICAgICAgICByZXN1bHQudyA9IG5leHRDb2xPdXRPZkJvdW5kcyB8fCBuZXh0Um93T3V0T2ZCb3VuZHMgPyAwLiA6IHJlc3VsdC53O1xuICAgICAgICAgIGA7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgdGhpcy51c2VyQ29kZSA9IGBcbiAgICAgIHZlYzQgYmluYXJ5T3BlcmF0aW9uKHZlYzQgYSwgdmVjNCBiKSB7XG4gICAgICAgICR7b3B9XG4gICAgICB9XG5cbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgdmVjNCBhID0gZ2V0QUF0T3V0Q29vcmRzKCk7XG4gICAgICAgIHZlYzQgYiA9IGdldEJBdE91dENvb3JkcygpO1xuXG4gICAgICAgIHZlYzQgcmVzdWx0ID0gYmluYXJ5T3BlcmF0aW9uKGEsIGIpO1xuICAgICAgICAke2NoZWNrT3V0T2ZCb3VuZHNTdHJpbmd9XG5cbiAgICAgICAgc2V0T3V0cHV0KHJlc3VsdCk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuIl19