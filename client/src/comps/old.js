//   const outputCtx = output.getContext('2d')
//   resJson.meta_data.detections.forEach(detection =>
//     {
//         const { pt1, pt2} = detection
//         const { img_width, img_height } = resJson.meta_data
//         output.width = img_width
//         output.height = img_height
//         outputCtx.strokeStyle = 'rgb(255, 255, 255)';
//         ctx.lineWidth = 2;
//         outputCtx.fillStyle = 'red'
//         ctx.beginPath();
//         ctx.strokeRect(pt1.x1, pt1.y1, pt2.x2, pt2.y2);
//         // ctx.fill();
//     })
