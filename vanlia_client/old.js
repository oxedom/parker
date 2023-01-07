const outputCtx = output.getContext("2d");
resJson.meta_data.detections.forEach((detection) => {
  const { pt1, pt2 } = detection;
  const { img_width, img_height } = resJson.meta_data;
  output.width = img_width;
  output.height = img_height;
  outputCtx.strokeStyle = "rgb(255, 255, 255)";
  ctx.lineWidth = 2;
  outputCtx.fillStyle = "red";
  ctx.beginPath();
  ctx.strokeRect(pt1.x1, pt1.y1, pt2.x2, pt2.y2);
  // ctx.fill();
});

const getCapabilitiesOfDevice = async () => {
  const devices = await navigator.mediaDevices.enumerateDevices();
  console.log(devices);
  const video_devices = devices.filter((d) => {
    return d.kind === "videoinput";
  });
  const videoDevice = video_devices[0];
  const videoDeviceCapabilities = await videoDevice.getCapabilities();
  console.log(videoDeviceCapabilities);
  return videoDeviceCapabilities;
};




[
  {
      "label": "anything",
      "cords": {
          "right_x": 354,
          "top_y": 359,
          "left_x": 680,
          "bottom_y": 72
      }
  },
  {
      "label": "anything",
      "cords": {
          "right_x": 73,
          "top_y": 289,
          "left_x": 239,
          "bottom_y": 116
      }
  },
  {
      "label": "anything",
      "cords": {
          "right_x": 202,
          "top_y": 319,
          "left_x": 394,
          "bottom_y": 173
      }
  },
  {
      "label": "anything",
      "cords": {
          "right_x": 283,
          "top_y": 305,
          "left_x": 428,
          "bottom_y": 175
      }
  },
  {
      "label": "anything",
      "cords": {
          "right_x": 212,
          "top_y": 302,
          "left_x": 498,
          "bottom_y": 106
      }
  }
]


[
  {
      "label": "anything",
      "cords": {
          "right_x": 30,
          "top_y": 175,
          "left_x": 138,
          "bottom_y": 36
      }
  },
  {
      "label": "anything",
      "cords": {
          "right_x": 205,
          "top_y": 373,
          "left_x": 324,
          "bottom_y": 214
      }
  },
  {
      "label": "anything",
      "cords": {
          "right_x": 562,
          "top_y": 308,
          "left_x": 707,
          "bottom_y": 85
      }
  },
  {
      "label": "anything",
      "cords": {
          "right_x": 306,
          "top_y": 263,
          "left_x": 537,
          "bottom_y": 63
      }
  }
]


