res = res.arraySync()[0];
//Filtering only detections > conf_thres
const conf_thres = 0.75;
res = res.filter(dataRow => dataRow[4]>=conf_thres);

var boxes = [];
var class_detect = [];
var scores = [];
res.forEach(process_pred);

function process_pred(res){
  var box = res.slice(0,4);

  const roiObj = 
  {
    cords : {top_x: undefined,
    left_x: undefined,
    width: undefined,
    height: undefined },
    label: undefined,
    score: undefined
  }

  const cls_detections = res.slice(5, 85);
  var max_score_index = cls_detections.reduce((imax, x, i, arr) => x > arr[imax] ? i : imax, 0);
  roiObj.score = max_score_index
  roiObj.label = labels[max_score_index]

  // let top_y = Math.abs(box[0])
  // let left_x = Math.abs(box[1])

  // let bottom_y = Math.abs(box[2])
  // let right_x = Math.abs(box[3])
  // let width = right_x - left_x
  // let height = top_y - bottom_y


  const right_x =box[0];
  const top_y =box[1];
  const width = box[2];
  const height =box[3];

  roiObj.cords  = 
  {
     top_y,
    left_x ,
   bottom_y,
    right_x,
    width,
    height,
  
  }
  roiObj.area = width * height
  console.log(roiObj)
//   const search_index = class_detect.indexOf(max_score_index);
//   if (search_index != -1){
//     if(scores[search_index] < res[max_score_index + 5]){
//       boxes[search_index] = box;
//       scores[search_index] = res[max_score_index + 5];
//     }
//   }
//   else{
//     boxes.push(box)
//     class_detect.push(max_score_index);
//     scores.push(res[max_score_index + 5]);
//   }
// }
// renderBoxes(canvasRef, threshold, boxes, scores, class_detect);
tf.dispose(res);
}
});