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




function paint(rect, image) {

  const let = 
  {
    "height": 314,
    "right_x": 2,
    "top_y": 2,
    "width": 404
}
  
}
  


return painted
}













module.exports = { getLines, fillArrayWithTrue, fillArrayWithFalse, paint };
