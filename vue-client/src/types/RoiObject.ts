export interface RoiObject {
  cords: {
    bottom_y: number
    left_x: number
    top_y: number
    right_x: number
    width: number
    height: number
  }
  confidenceLevel: number
  label: string
  area: number
}
