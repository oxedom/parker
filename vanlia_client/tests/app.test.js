const {
  getLines,
  fillArrayWithFalse,
  paint,
  fillArrayWithTrue,
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

describe("Paint", () => {
  test("1# Give the parent cords a larger bbox than itself and make the parent 100% true", () => {
    let roi = fillArrayWithFalse(10, 10);
    let trueROi = fillArrayWithTrue(10, 10);
    let cords = {
      top_y: 698,
      left_x: 54,
      right_x: 1086,
      bottom_y: 82,
    };
    let paintedRoi = paint(cords, roi);

    expect(paintedRoi).toEqual(trueROi);
  });

  test("3# Paint top half with true", () => {
    let roi = fillArrayWithFalse(4, 4);
    let answer = [
      [true, true, true, true],
      [true, true, true, true],
      [false, false, false, false],
      [false, false, false, false],
    ];

    // let neededChanges = [
    //   (0,0),(0,1)(0,2), (0,3)
    //   (1,0), (1,1), (1,2), (1,3)
    // ]

    let rect = {
      top_y: 2,
      bottom_y: 0,
      left_x: 0,
      right_x: 4,
    };

    let paintedRoi = paint(rect, roi);
    expect(paintedRoi).toEqual(answer);
  });

  test("4# Paint bottom half with true", () => {
    let roi = fillArrayWithFalse(4, 4);
    let answer = [
      [false, false, false, false],
      [false, false, false, false],
      [true, true, true, true],
      [true, true, true, true],
    ];

    let rect = {
      top_y: 2,
      bottom_y: 0,
      left_x: 0,
      right_x: 4,
    };

    let paintedRoi = paint(rect, roi);
    expect(paintedRoi).toEqual(answer);
  });

  test("#5 paint half left with true", () => {
    let roi = fillArrayWithFalse(4, 4);
    let answer = [
      [true, true, false, false],
      [true, true, false, false],
      [true, true, false, false],
      [true, true, false, false],
    ];
    let rect = {
      top_y: 4,
      bottom_y: 2,
      left_x: 0,
      right_x: 2,
    };
    let paintedRoi = paint(rect, roi);
    expect(paintedRoi).toEqual(answer);
  });

  test("#6 paint half right with true", () => {
    let roi = fillArrayWithFalse(4, 4);
    let answer = [
      [false, false, true, true],
      [false, false, true, true],
      [false, false, true, true],
      [false, false, true, true],
    ];

    let rect = {
      top_y: 4,
      bottom_y: 0,
      left_x: 2,
      right_x: 4,
    };

    let paintedRoi = paint(rect, roi);

    expect(paintedRoi).toEqual(answer);
  });

  test("#7 Painting it all", () => {
    let roi = fillArrayWithFalse(4, 4);
    let answer = [
      [true, true, true, true],
      [true, true, true, true],
      [true, true, true, true],
      [true, true, true, true],
    ];

    let rect = {
      top_y: 100,
      bottom_y: 0,
      left_x: 0,
      right_x: 100,
    };

    let paintedRoi = paint(rect, roi);

    expect(paintedRoi).toEqual(answer);
  });
});
