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

let {bottom_y, left_x, right_x, top_y} = rect
let painted = image

for (let y = 0; y < image.length; y++) {
  for (let x = 0; x < image[0].length; x++) {


    if(((left_x-1) <= x <= (right_x-1)))
    {
      console.log('I have been printed');
    }
    
  }
  
}
  


return painted
}













module.exports = { getLines, fillArrayWithTrue, fillArrayWithFalse, paint };
