export interface RoiCords {
  right_x: number;
  top_y: number;
  width: number;
  height: number;
  bottom_y?: number;
  left_x?: number;
}

export type RoiEventName = "initialized" | "occupied" | "available";

export interface RoiEvent {
  eventName: RoiEventName;
  timeMarked: number;
  duration: number | null;
  cycle?: number;
}

export interface SelectedRoi {
  uid: string;
  label: string;
  cords: RoiCords;
  area: number;
  firstSeen: number | null;
  lastSeen: number | null;
  occupied: boolean | null;
  evaluating: boolean;
  hover: boolean | null;
  cycleCount: number;
  events: RoiEvent[];
  confidenceLevel?: number;
}

export interface DetectionRoi {
  cords: RoiCords;
  label: string;
  confidenceLevel: number;
  area: number;
  hover?: boolean | null;
  evaluating?: boolean;
  occupied?: boolean | null;
}

export type VideoSource = "demo" | "webcam" | "rtc" | null;

export interface OccupationCounts {
  OccupiedCount: number;
  availableCount: number;
}
