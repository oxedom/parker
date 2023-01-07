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




function paint(rect, image) {
  const biggestY = Math.max(rect.cords.top_y,rect.cords.bottom_y );
  const smallestY = Math.min(rect.cords.top_y,rect.cords.bottom_y );

  const biggestX = Math.max(rect.cords.right_x, rect.cords.left_x);
  const smallestX = Math.min(rect.cords.right_x, rect.cords.left_x);

  for (let r = smallestY; r < biggestY; r++) {
      for (let c = smallestX; c < biggestX; c++) {
        image[r][c] = true
      }
  }
  return image;
}




 
const roi = 
{
  cords:
  {
    "right_x":1,
    "top_y": 4,
    "left_x": 0,
    "bottom_y": 0
}
}

const FALSEARRAY = fillArrayWithFalse(4,4)
console.table(paint(roi, FALSEARRAY))














module.exports = { getLines, fillArrayWithFalse, fillArrayWithTrue };
