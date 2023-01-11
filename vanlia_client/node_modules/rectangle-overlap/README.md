# rectangle-overlap

[![TypeScript support](https://img.shields.io/npm/types/rectangle-overlap.svg)](https://github.com/lovasoa/rectangle-overlap/blob/master/index.ts)
[![License](https://img.shields.io/npm/l/rectangle-overlap.svg)](https://github.com/lovasoa/rectangle-overlap/blob/master/LICENSE)

Fastly compute the intersection of two rectangles.

## Usage

```js
const intersection = require("rectangle-overlap");

let rect1 = {x: 0, y: 0, width: 10, height: 10};
let rect2 = {x: 2, y: 3, width: 42, height: 42};

const overlap = intersection(rect1, rect2);

if (overlap) {
  console.log(`The rectangles overlap over an area of ${overlap.area}`);
  console.log(
    `Intersection coordinates: x=${overlap.x}, y=${overlap.y}, width=${overlap.width}, height=${overlap.height}`,
  )
} else {
  console.log("The rectangles do not overlap");
}
```
