const intersection = require("rectangle-overlap");



function getLines(rect) {
  const xLine = rect.cords.left_x - rect.cords.right_x;
  const yLine = rect.cords.top_y - rect.cords.bottom_y;

  return { xLine, yLine };
}

function rectangleArea(rect) {
  const width = rect.cords.right_x - rect.cords.left_x;
  const height = rect.cords.bottom_y - rect.cords.top_y;
  return Math.abs(width * height);
}

function fillArrayWithFalse(width, height) {
  const picture = [];
  for (let y = 0; y < height; y++) {
    let row = new Array(width).fill(false);
    picture.push(row);
  }

  return picture;
}

function fillArrayWithTrue(width, height) {
  const picture = [];
  for (let y = 0; y < height; y++) {
    let row = new Array(width).fill(true);
    picture.push(row);
  }

  return picture;
}







let getOverlap = (rectangle1,rectangle2) => 
{
  const intersectionX1 = Math.max(rectangle1.right_x, rectangle2.right_x);
  const intersectionX2 = Math.min(rectangle1.right_x + rectangle1.width, rectangle2.right_x + rectangle2.width);
  if (intersectionX2 < intersectionX1) {
    return null;
  }
  const intersectionY1 = Math.max(rectangle1.top_y, rectangle2.top_y);
  const intersectionY2 = Math.min(rectangle1.top_y + rectangle1.height, rectangle2.top_y + rectangle2.height);
  if (intersectionY2 < intersectionY1) {
    return null;
  }

  return {
    right_x:intersectionX1,
    top_y:intersectionY1,
    width: intersectionX2 - intersectionX1,
    height:  intersectionY2 - intersectionY1,
    area: ((intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1))
  }


}





module.exports = { getLines, fillArrayWithTrue, fillArrayWithFalse, paint };
