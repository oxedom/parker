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
import { env } from '@tensorflow/tfjs-core';
export function getGlslDifferences() {
    let version;
    let attribute;
    let varyingVs;
    let varyingFs;
    let texture2D;
    let output;
    let defineOutput;
    let defineSpecialNaN;
    let defineSpecialInf;
    let defineRound;
    if (env().getNumber('WEBGL_VERSION') === 2) {
        version = '#version 300 es';
        attribute = 'in';
        varyingVs = 'out';
        varyingFs = 'in';
        texture2D = 'texture';
        output = 'outputColor';
        defineOutput = 'out vec4 outputColor;';
        // Use custom isnan definition to work across differences between
        // implementations on various platforms. While this should happen in ANGLE
        // we still see differences between android and windows (on chrome) when
        // using isnan directly. Since WebGL2 supports uint type and
        // floatBitsToUinT built-in function, we could implment isnan following
        // IEEE 754 rules.
        // NaN defination in IEEE 754-1985 is :
        //   - sign = either 0 or 1.
        //   - biased exponent = all 1 bits.
        //   - fraction = anything except all 0 bits (since all 0 bits represents
        //   infinity).
        // https://en.wikipedia.org/wiki/IEEE_754-1985#Representation_of_non-numbers
        defineSpecialNaN = `
      bool isnan_custom(float val) {
        uint floatToUint = floatBitsToUint(val);
        return (floatToUint & 0x7fffffffu) > 0x7f800000u;
      }

      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan_custom(val.x),
          isnan_custom(val.y), isnan_custom(val.z), isnan_custom(val.w));
      }

      #define isnan(value) isnan_custom(value)
    `;
        // In webgl 2 we do not need to specify a custom isinf so there is no
        // need for a special INFINITY constant.
        defineSpecialInf = ``;
        defineRound = `
      #define round(value) newRound(value)
      int newRound(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 newRound(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `;
    }
    else {
        version = '';
        attribute = 'attribute';
        varyingVs = 'varying';
        varyingFs = 'varying';
        texture2D = 'texture2D';
        output = 'gl_FragColor';
        defineOutput = '';
        // WebGL1 has no built in isnan so we define one here.
        defineSpecialNaN = `
      #define isnan(value) isnan_custom(value)
      bool isnan_custom(float val) {
        return (val > 0. || val < 1. || val == 0.) ? false : true;
      }
      bvec4 isnan_custom(vec4 val) {
        return bvec4(isnan(val.x), isnan(val.y), isnan(val.z), isnan(val.w));
      }
    `;
        defineSpecialInf = `
      uniform float INFINITY;

      bool isinf(float val) {
        return abs(val) == INFINITY;
      }
      bvec4 isinf(vec4 val) {
        return equal(abs(val), vec4(INFINITY));
      }
    `;
        defineRound = `
      int round(float value) {
        return int(floor(value + 0.5));
      }

      ivec4 round(vec4 value) {
        return ivec4(floor(value + vec4(0.5)));
      }
    `;
    }
    return {
        version,
        attribute,
        varyingVs,
        varyingFs,
        texture2D,
        output,
        defineOutput,
        defineSpecialNaN,
        defineSpecialInf,
        defineRound
    };
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZ2xzbF92ZXJzaW9uLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9nbHNsX3ZlcnNpb24udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBZTFDLE1BQU0sVUFBVSxrQkFBa0I7SUFDaEMsSUFBSSxPQUFlLENBQUM7SUFDcEIsSUFBSSxTQUFpQixDQUFDO0lBQ3RCLElBQUksU0FBaUIsQ0FBQztJQUN0QixJQUFJLFNBQWlCLENBQUM7SUFDdEIsSUFBSSxTQUFpQixDQUFDO0lBQ3RCLElBQUksTUFBYyxDQUFDO0lBQ25CLElBQUksWUFBb0IsQ0FBQztJQUN6QixJQUFJLGdCQUF3QixDQUFDO0lBQzdCLElBQUksZ0JBQXdCLENBQUM7SUFDN0IsSUFBSSxXQUFtQixDQUFDO0lBRXhCLElBQUksR0FBRyxFQUFFLENBQUMsU0FBUyxDQUFDLGVBQWUsQ0FBQyxLQUFLLENBQUMsRUFBRTtRQUMxQyxPQUFPLEdBQUcsaUJBQWlCLENBQUM7UUFDNUIsU0FBUyxHQUFHLElBQUksQ0FBQztRQUNqQixTQUFTLEdBQUcsS0FBSyxDQUFDO1FBQ2xCLFNBQVMsR0FBRyxJQUFJLENBQUM7UUFDakIsU0FBUyxHQUFHLFNBQVMsQ0FBQztRQUN0QixNQUFNLEdBQUcsYUFBYSxDQUFDO1FBQ3ZCLFlBQVksR0FBRyx1QkFBdUIsQ0FBQztRQUV2QyxpRUFBaUU7UUFDakUsMEVBQTBFO1FBQzFFLHdFQUF3RTtRQUN4RSw0REFBNEQ7UUFDNUQsdUVBQXVFO1FBQ3ZFLGtCQUFrQjtRQUNsQix1Q0FBdUM7UUFDdkMsNEJBQTRCO1FBQzVCLG9DQUFvQztRQUNwQyx5RUFBeUU7UUFDekUsZUFBZTtRQUNmLDRFQUE0RTtRQUM1RSxnQkFBZ0IsR0FBRzs7Ozs7Ozs7Ozs7O0tBWWxCLENBQUM7UUFDRixxRUFBcUU7UUFDckUsd0NBQXdDO1FBQ3hDLGdCQUFnQixHQUFHLEVBQUUsQ0FBQztRQUN0QixXQUFXLEdBQUc7Ozs7Ozs7OztLQVNiLENBQUM7S0FDSDtTQUFNO1FBQ0wsT0FBTyxHQUFHLEVBQUUsQ0FBQztRQUNiLFNBQVMsR0FBRyxXQUFXLENBQUM7UUFDeEIsU0FBUyxHQUFHLFNBQVMsQ0FBQztRQUN0QixTQUFTLEdBQUcsU0FBUyxDQUFDO1FBQ3RCLFNBQVMsR0FBRyxXQUFXLENBQUM7UUFDeEIsTUFBTSxHQUFHLGNBQWMsQ0FBQztRQUN4QixZQUFZLEdBQUcsRUFBRSxDQUFDO1FBQ2xCLHNEQUFzRDtRQUN0RCxnQkFBZ0IsR0FBRzs7Ozs7Ozs7S0FRbEIsQ0FBQztRQUNGLGdCQUFnQixHQUFHOzs7Ozs7Ozs7S0FTbEIsQ0FBQztRQUNGLFdBQVcsR0FBRzs7Ozs7Ozs7S0FRYixDQUFDO0tBQ0g7SUFFRCxPQUFPO1FBQ0wsT0FBTztRQUNQLFNBQVM7UUFDVCxTQUFTO1FBQ1QsU0FBUztRQUNULFNBQVM7UUFDVCxNQUFNO1FBQ04sWUFBWTtRQUNaLGdCQUFnQjtRQUNoQixnQkFBZ0I7UUFDaEIsV0FBVztLQUNaLENBQUM7QUFDSixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuaW1wb3J0IHtlbnZ9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmV4cG9ydCB0eXBlIEdMU0wgPSB7XG4gIHZlcnNpb246IHN0cmluZyxcbiAgYXR0cmlidXRlOiBzdHJpbmcsXG4gIHZhcnlpbmdWczogc3RyaW5nLFxuICB2YXJ5aW5nRnM6IHN0cmluZyxcbiAgdGV4dHVyZTJEOiBzdHJpbmcsXG4gIG91dHB1dDogc3RyaW5nLFxuICBkZWZpbmVPdXRwdXQ6IHN0cmluZyxcbiAgZGVmaW5lU3BlY2lhbE5hTjogc3RyaW5nLFxuICBkZWZpbmVTcGVjaWFsSW5mOiBzdHJpbmcsXG4gIGRlZmluZVJvdW5kOiBzdHJpbmdcbn07XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRHbHNsRGlmZmVyZW5jZXMoKTogR0xTTCB7XG4gIGxldCB2ZXJzaW9uOiBzdHJpbmc7XG4gIGxldCBhdHRyaWJ1dGU6IHN0cmluZztcbiAgbGV0IHZhcnlpbmdWczogc3RyaW5nO1xuICBsZXQgdmFyeWluZ0ZzOiBzdHJpbmc7XG4gIGxldCB0ZXh0dXJlMkQ6IHN0cmluZztcbiAgbGV0IG91dHB1dDogc3RyaW5nO1xuICBsZXQgZGVmaW5lT3V0cHV0OiBzdHJpbmc7XG4gIGxldCBkZWZpbmVTcGVjaWFsTmFOOiBzdHJpbmc7XG4gIGxldCBkZWZpbmVTcGVjaWFsSW5mOiBzdHJpbmc7XG4gIGxldCBkZWZpbmVSb3VuZDogc3RyaW5nO1xuXG4gIGlmIChlbnYoKS5nZXROdW1iZXIoJ1dFQkdMX1ZFUlNJT04nKSA9PT0gMikge1xuICAgIHZlcnNpb24gPSAnI3ZlcnNpb24gMzAwIGVzJztcbiAgICBhdHRyaWJ1dGUgPSAnaW4nO1xuICAgIHZhcnlpbmdWcyA9ICdvdXQnO1xuICAgIHZhcnlpbmdGcyA9ICdpbic7XG4gICAgdGV4dHVyZTJEID0gJ3RleHR1cmUnO1xuICAgIG91dHB1dCA9ICdvdXRwdXRDb2xvcic7XG4gICAgZGVmaW5lT3V0cHV0ID0gJ291dCB2ZWM0IG91dHB1dENvbG9yOyc7XG5cbiAgICAvLyBVc2UgY3VzdG9tIGlzbmFuIGRlZmluaXRpb24gdG8gd29yayBhY3Jvc3MgZGlmZmVyZW5jZXMgYmV0d2VlblxuICAgIC8vIGltcGxlbWVudGF0aW9ucyBvbiB2YXJpb3VzIHBsYXRmb3Jtcy4gV2hpbGUgdGhpcyBzaG91bGQgaGFwcGVuIGluIEFOR0xFXG4gICAgLy8gd2Ugc3RpbGwgc2VlIGRpZmZlcmVuY2VzIGJldHdlZW4gYW5kcm9pZCBhbmQgd2luZG93cyAob24gY2hyb21lKSB3aGVuXG4gICAgLy8gdXNpbmcgaXNuYW4gZGlyZWN0bHkuIFNpbmNlIFdlYkdMMiBzdXBwb3J0cyB1aW50IHR5cGUgYW5kXG4gICAgLy8gZmxvYXRCaXRzVG9VaW5UIGJ1aWx0LWluIGZ1bmN0aW9uLCB3ZSBjb3VsZCBpbXBsbWVudCBpc25hbiBmb2xsb3dpbmdcbiAgICAvLyBJRUVFIDc1NCBydWxlcy5cbiAgICAvLyBOYU4gZGVmaW5hdGlvbiBpbiBJRUVFIDc1NC0xOTg1IGlzIDpcbiAgICAvLyAgIC0gc2lnbiA9IGVpdGhlciAwIG9yIDEuXG4gICAgLy8gICAtIGJpYXNlZCBleHBvbmVudCA9IGFsbCAxIGJpdHMuXG4gICAgLy8gICAtIGZyYWN0aW9uID0gYW55dGhpbmcgZXhjZXB0IGFsbCAwIGJpdHMgKHNpbmNlIGFsbCAwIGJpdHMgcmVwcmVzZW50c1xuICAgIC8vICAgaW5maW5pdHkpLlxuICAgIC8vIGh0dHBzOi8vZW4ud2lraXBlZGlhLm9yZy93aWtpL0lFRUVfNzU0LTE5ODUjUmVwcmVzZW50YXRpb25fb2Zfbm9uLW51bWJlcnNcbiAgICBkZWZpbmVTcGVjaWFsTmFOID0gYFxuICAgICAgYm9vbCBpc25hbl9jdXN0b20oZmxvYXQgdmFsKSB7XG4gICAgICAgIHVpbnQgZmxvYXRUb1VpbnQgPSBmbG9hdEJpdHNUb1VpbnQodmFsKTtcbiAgICAgICAgcmV0dXJuIChmbG9hdFRvVWludCAmIDB4N2ZmZmZmZmZ1KSA+IDB4N2Y4MDAwMDB1O1xuICAgICAgfVxuXG4gICAgICBidmVjNCBpc25hbl9jdXN0b20odmVjNCB2YWwpIHtcbiAgICAgICAgcmV0dXJuIGJ2ZWM0KGlzbmFuX2N1c3RvbSh2YWwueCksXG4gICAgICAgICAgaXNuYW5fY3VzdG9tKHZhbC55KSwgaXNuYW5fY3VzdG9tKHZhbC56KSwgaXNuYW5fY3VzdG9tKHZhbC53KSk7XG4gICAgICB9XG5cbiAgICAgICNkZWZpbmUgaXNuYW4odmFsdWUpIGlzbmFuX2N1c3RvbSh2YWx1ZSlcbiAgICBgO1xuICAgIC8vIEluIHdlYmdsIDIgd2UgZG8gbm90IG5lZWQgdG8gc3BlY2lmeSBhIGN1c3RvbSBpc2luZiBzbyB0aGVyZSBpcyBub1xuICAgIC8vIG5lZWQgZm9yIGEgc3BlY2lhbCBJTkZJTklUWSBjb25zdGFudC5cbiAgICBkZWZpbmVTcGVjaWFsSW5mID0gYGA7XG4gICAgZGVmaW5lUm91bmQgPSBgXG4gICAgICAjZGVmaW5lIHJvdW5kKHZhbHVlKSBuZXdSb3VuZCh2YWx1ZSlcbiAgICAgIGludCBuZXdSb3VuZChmbG9hdCB2YWx1ZSkge1xuICAgICAgICByZXR1cm4gaW50KGZsb29yKHZhbHVlICsgMC41KSk7XG4gICAgICB9XG5cbiAgICAgIGl2ZWM0IG5ld1JvdW5kKHZlYzQgdmFsdWUpIHtcbiAgICAgICAgcmV0dXJuIGl2ZWM0KGZsb29yKHZhbHVlICsgdmVjNCgwLjUpKSk7XG4gICAgICB9XG4gICAgYDtcbiAgfSBlbHNlIHtcbiAgICB2ZXJzaW9uID0gJyc7XG4gICAgYXR0cmlidXRlID0gJ2F0dHJpYnV0ZSc7XG4gICAgdmFyeWluZ1ZzID0gJ3ZhcnlpbmcnO1xuICAgIHZhcnlpbmdGcyA9ICd2YXJ5aW5nJztcbiAgICB0ZXh0dXJlMkQgPSAndGV4dHVyZTJEJztcbiAgICBvdXRwdXQgPSAnZ2xfRnJhZ0NvbG9yJztcbiAgICBkZWZpbmVPdXRwdXQgPSAnJztcbiAgICAvLyBXZWJHTDEgaGFzIG5vIGJ1aWx0IGluIGlzbmFuIHNvIHdlIGRlZmluZSBvbmUgaGVyZS5cbiAgICBkZWZpbmVTcGVjaWFsTmFOID0gYFxuICAgICAgI2RlZmluZSBpc25hbih2YWx1ZSkgaXNuYW5fY3VzdG9tKHZhbHVlKVxuICAgICAgYm9vbCBpc25hbl9jdXN0b20oZmxvYXQgdmFsKSB7XG4gICAgICAgIHJldHVybiAodmFsID4gMC4gfHwgdmFsIDwgMS4gfHwgdmFsID09IDAuKSA/IGZhbHNlIDogdHJ1ZTtcbiAgICAgIH1cbiAgICAgIGJ2ZWM0IGlzbmFuX2N1c3RvbSh2ZWM0IHZhbCkge1xuICAgICAgICByZXR1cm4gYnZlYzQoaXNuYW4odmFsLngpLCBpc25hbih2YWwueSksIGlzbmFuKHZhbC56KSwgaXNuYW4odmFsLncpKTtcbiAgICAgIH1cbiAgICBgO1xuICAgIGRlZmluZVNwZWNpYWxJbmYgPSBgXG4gICAgICB1bmlmb3JtIGZsb2F0IElORklOSVRZO1xuXG4gICAgICBib29sIGlzaW5mKGZsb2F0IHZhbCkge1xuICAgICAgICByZXR1cm4gYWJzKHZhbCkgPT0gSU5GSU5JVFk7XG4gICAgICB9XG4gICAgICBidmVjNCBpc2luZih2ZWM0IHZhbCkge1xuICAgICAgICByZXR1cm4gZXF1YWwoYWJzKHZhbCksIHZlYzQoSU5GSU5JVFkpKTtcbiAgICAgIH1cbiAgICBgO1xuICAgIGRlZmluZVJvdW5kID0gYFxuICAgICAgaW50IHJvdW5kKGZsb2F0IHZhbHVlKSB7XG4gICAgICAgIHJldHVybiBpbnQoZmxvb3IodmFsdWUgKyAwLjUpKTtcbiAgICAgIH1cblxuICAgICAgaXZlYzQgcm91bmQodmVjNCB2YWx1ZSkge1xuICAgICAgICByZXR1cm4gaXZlYzQoZmxvb3IodmFsdWUgKyB2ZWM0KDAuNSkpKTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG5cbiAgcmV0dXJuIHtcbiAgICB2ZXJzaW9uLFxuICAgIGF0dHJpYnV0ZSxcbiAgICB2YXJ5aW5nVnMsXG4gICAgdmFyeWluZ0ZzLFxuICAgIHRleHR1cmUyRCxcbiAgICBvdXRwdXQsXG4gICAgZGVmaW5lT3V0cHV0LFxuICAgIGRlZmluZVNwZWNpYWxOYU4sXG4gICAgZGVmaW5lU3BlY2lhbEluZixcbiAgICBkZWZpbmVSb3VuZFxuICB9O1xufVxuIl19