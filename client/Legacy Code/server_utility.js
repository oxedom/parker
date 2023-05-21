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