const Canvas = (props) => {
  console.log(props);
  return (
    <canvas
      width={props.imageWidth}
      height={props.imageHeight}
      className="absolute"
    ></canvas>
  );
};

export default Canvas;
