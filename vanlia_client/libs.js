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


const fourByFour = fillArrayWithFalse(4,4)


const roi = {label: 'anything',
    cords: {
      right_x: 4,
      top_y: 4 ,
      left_x: 0 ,
      bottom_y: 0,}}



function paint(roi, image) {
  for (let x = roi.cords.left_x; x <= roi.cords.right_x-1; x++) {
    for (let y = roi.cords.top_y-1; y <= roi.cords.bottom_y-1; y++) {
      image[x][y] = true;
    }
  }
  return image;
}


 


console.table(paint(roi,fourByFour))














module.exports = { getLines, fillArrayWithFalse, fillPictureWithRect };
