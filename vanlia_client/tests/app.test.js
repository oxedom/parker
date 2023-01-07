const {
  getLines,
  fillArrayWithFalse,
  fillPictureWithRect,
} = require("../libs");

describe("getLines", () => {
  test("#1 Get lines", () => {
    const bigger = {
      label: "anything",
      cords: {
        right_x: 100,
        top_y: 450,
        left_x: 300,
        bottom_y: 150,
      },
    };

    const answer = { xLine: 200, yLine: 300 };

    expect(getLines(bigger)).toEqual(answer);
  });

  test("#2 Get lines  ", () => {
    const rect = {
      label: "anything",
      cords: {
        right_x: 312,
        top_y: 31,
        left_x: 22,
        bottom_y: 150,
      },
    };

    const answer = { xLine: -290, yLine: -119 };

    expect(getLines(rect)).toEqual(answer);
  });
});

describe("Fill all picture with false", () => {
  test("#1 Fill picture with false ", () => {
    const width = 2;
    const height = 2;
    const answer = [
      [false, false],
      [false, false],
    ];

    expect(fillArrayWithFalse(width, height)).toEqual(answer);
  });

  test("#2 Fill picture with false", () => {
    const width = 1;
    const height = 10;
    const answer = [
      [false],
      [false],
      [false],
      [false],
      [false],
      [false],
      [false],
      [false],
      [false],
      [false],
    ];
    expect(fillArrayWithFalse(width, height)).toEqual(answer);
  });

  test("#3 Fill picture with false", () => {
    const width = 10;
    const height = 1;
    const answer = [
      [false, false, false, false, false, false, false, false, false, false],
    ];

    expect(fillArrayWithFalse(width, height)).toEqual(answer);
  });
});


describe("Fill rectangle in rectangle", () => {
  test("#1", () => {
  
    
    expect(fillPictureWithRect(falsePictureArray, rect)).toEqual(answer);
  }); 

})


