import labels from "./labels.json";

export function xywh2xyxy(x) {
  //Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
  var y = [];
  y[0] = x[0] - x[2] / 2; //top left x
  y[1] = x[1] - x[3] / 2; //top left y
  y[2] = x[0] + x[2] / 2; //bottom right x
  y[3] = x[1] + x[3] / 2; //bottom right y
  return y;
}

export const renderBoxes = (
  canvasRef,
  threshold,
  boxes_data,
  scores_data,
  classes_data
) => {
  canvasRef.clearRect(0, 0, canvasRef.canvas.width, canvasRef.canvas.height); // clean canvas

  // font configs
  const font = "18px sans-serif";
  canvasRef.font = font;
  canvasRef.textBaseline = "top";

  for (let i = 0; i < scores_data.length; ++i) {
    //console.log('scores_data[i]: ', scores_data[i])
    if (scores_data[i] > threshold) {
      const klass = labels[classes_data[i]];
      const score = (scores_data[i] * 100).toFixed(1);

      let [x1, y1, x2, y2] = xywh2xyxy(boxes_data[i]);

      const width = x2 - x1;
      const height = y2 - y1;

      // Draw the bounding box.
      canvasRef.strokeStyle = "#B033FF";
      canvasRef.lineWidth = 2;
      canvasRef.strokeRect(x1, y1, width, height);

      // Draw the label background.
      canvasRef.fillStyle = "#B033FF";
      const textWidth = canvasRef.measureText(
        klass + " - " + score + "%"
      ).width;
      const textHeight = parseInt(font, 10); // base 10
      canvasRef.fillRect(
        x1 - 1,
        y1 - (textHeight + 2),
        textWidth + 2,
        textHeight + 2
      );

      // Draw labels
      canvasRef.fillStyle = "#ffffff";
      canvasRef.fillText(
        klass + " - " + score + "%",
        x1 - 1,
        y1 - (textHeight + 2)
      );
    }
  }
};
