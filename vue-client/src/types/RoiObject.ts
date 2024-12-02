export interface Rectangle {
  bottom_y: number | null
  left_x: number | null
  top_y: number | null
  right_x: number | null
  width: number | null
  height: number | null
}

export interface RoiObject {
  cords: Rectangle
  confidenceLevel: number | null
  label: string | null
  area: number | null
}
