

function getOverlap(rectangle1, rectangle2) {
  const intersectionX1 = Math.max(rectangle1.right_x, rectangle2.right_x);
  const intersectionX2 = Math.min(
    rectangle1.right_x + rectangle1.width,
    rectangle2.right_x + rectangle2.width
  );
  if (intersectionX2 < intersectionX1) {
    return null;
  }
  const intersectionY1 = Math.max(rectangle1.top_y, rectangle2.top_y);
  const intersectionY2 = Math.min(
    rectangle1.top_y + rectangle1.height,
    rectangle2.top_y + rectangle2.height
  );
  if (intersectionY2 < intersectionY1) {
    return null;
  }

  return {
    right_x: intersectionX1,
    top_y: intersectionY1,
    width: intersectionX2 - intersectionX1,
    height: intersectionY2 - intersectionY1,
    area: (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1),
  };
}

async function capturedImageToBuffer(capturedImage) {
  const imagePhoto = await capturedImage.takePhoto();

  let imageBuffer = await imagePhoto.arrayBuffer();

  imageBuffer = new Uint8Array(imageBuffer);

  return imageBuffer;
}

export async function capturedImageServer(capturedImage) {
  const imageBuffer = await capturedImageToBuffer(capturedImage);

  const res = await fetch(flask_url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ buffer: [...imageBuffer] }),
  });

  const data = await res.json();

  return data;
}



export function checkOverlapArrays(detectionsArr, selectedArr) {
  let overlaps = [];
  detectionsArr.forEach((d) => {
    selectedArr.forEach((s) => {
      let overlapCords = getOverlap(d.cords, s.cords);
      if (overlapCords != null) {
        let overlap = {
          ...s,
          cords: overlapCords,
        };

        overlaps.push(overlap);
      }
    });
  });
  return overlaps;
}

export function checkRectOverlap(rect, detectionsArr) {
  let answer = false;
  detectionsArr.forEach((d) => {
    //If answer is already true return answer
    if(answer == true) { return answer}
        ///Overlap calculation
    let overlapCords = getOverlap(d.cords, rect.cords);
    //If overlapcords is null the squares don't intersect
    if (overlapCords == null) {
      return;
    } else {
      let overlapArea_rounded = Math.round(overlapCords.area);
      let rectArea_rounded = Math.round(rect.area);

      //If the overlap is the same size (The decection is bigger than the selection)
      if (overlapArea_rounded == rectArea_rounded) {
        answer = true;
      }

      //Overlap rounded should be smaller than rectArea, so we calcualte
      //how much % of the sqaure it's overlapping
      let percentDiff = overlapArea_rounded / rectArea_rounded;

      //If it overlaps more than 40% of the square return true, else false
      if (percentDiff > 0.4) {
        answer = true;
      } else {
        return;
      }
    }
  });
  return answer;
}






export function isVehicle(label) {
if(label === 'car' || label === 'truck' ||  "label" == 'motorcycle'  || label ===  'bus'  ) { return true} 
else { return false}

 }