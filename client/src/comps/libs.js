function isValidBase64Image(str) {
  const regex =
    /^data:image\/(png|jpeg|jpg|gif|bmp|webp);base64,[A-Za-z0-9+/]+=*$/;
  return regex.test(str);
}


