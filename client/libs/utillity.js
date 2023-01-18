const flask_url = "http://127.0.0.1:5000/api/cv/yolo";

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

function rectangleArea(rect) {
  const width = rect.cords.right_x - rect.cords.left_x;
  const height = rect.cords.bottom_y - rect.cords.top_y;
  return Math.abs(width * height);
}

export function finalName(name, arrayLength) {
  if (name === "") {
    return `ROI NUMBER: ${arrayLength + 1}`;
  } else return name;
}
