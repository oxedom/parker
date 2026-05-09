<script setup lang="ts">
import DefaultLayout from "@/components/DefaultLayout.vue";
import AppQuestion from "@/components/AppQuestion.vue";
import AppInstruction from "@/components/AppInstruction.vue";

const baseUrl = import.meta.env.BASE_URL;

const questions = [
  {
    question: "How does Parker work?",
    answer:
      "Under the hood, Parker uses tensorflow.js to run machine learning models entirely in the browser. Parker uses YOLO7 (You Only Look Once v7), a real-time object detection model. The model's output is mapped to a custom data structure suitable for a parking application.",
  },
  {
    question: "How does remote communication work?",
    answer:
      "Parker implements remote communication through a P2P solution using PeerJS, a JavaScript library that wraps WebRTC and uses PeerServer Cloud services and Google STUN servers.",
  },
  {
    question: "What kind of video input does the application take?",
    answer:
      "Parker can take webcam footage or remote video communication through another device with a built-in camera and modern browser.",
  },
  {
    question: "What data is collected from the application?",
    answer:
      "Other than basic site analytics, none. Parker processes all video footage on the client side, and remote communication is peer-to-peer and private.",
  },
  {
    question: "How was the auto-detect parking implemented?",
    answer:
      "Auto-detect collects a snapshot of all detected vehicles for a few seconds and then suppresses any that have not been in a consistent position. On an empty lot no selections will be made; on a full lot all the parking spaces will be marked.",
  },
  {
    question: "Could the processing be moved to a backend server?",
    answer:
      "Yes — earlier versions of Parker used a Flask server that received base64/blob frames from the client and processed them with YOLOv3 + OpenCV. It works at small scale but needs additional configuration (powerful GPU, autoscaling) to grow.",
  },
];
</script>

<template>
  <DefaultLayout>
    <div class="flex-1 flex mx-auto flex-col p-10 max-w-[800px] gap-5 mt-5">
      <main>
        <div class="text-lg">
          <h2
            class="text-5xl text-center flex justify-center items-center gap-5 font-bold rounded-lg mb-10 text-orange-500"
          >
            About Parker
            <a href="https://github.com/oxedom/parker">
              <img
                :src="`${baseUrl}static/icons/github-mark.png`"
                alt="github"
                width="50"
              />
            </a>
          </h2>

          <section
            class="p-5 border-b-8 border-orange-500 rounded-md flex gap-2 flex-col text-xl"
          >
            <p>
              <span class="text-2xl text-gray-700 font-bold">Parker</span> is an
              open-source parking browser application that enables you to monitor
              parking spots using a webcam, cellphone camera, or any virtual
              webcam. The tool uses computer vision object detection and all
              computation is processed inside the browser using TensorFlow.js.
              Communication between a remote cellphone and the browser is enabled
              through WebRTC (PeerJS uses PeerServer for session metadata and
              candidate signaling, as well as Google STUN servers).
            </p>
            <p class="text-4xl my-5 text-gray-700">Built with</p>
            <ul class="grid grid-cols-none sm:grid-cols-2 gap-2">
              <li class="flex gap-2">
                <img
                  :src="`${baseUrl}static/tesnorflow.png`"
                  alt="tensorflow"
                  width="30"
                  height="30"
                />
                <a class="text-blue-500" href="https://www.tensorflow.org/js"
                  >TensorFlow.js</a
                >
              </li>
              <li class="flex gap-2">
                <span>Vue 3 + Vite</span>
              </li>
              <li class="flex gap-2">
                <img :src="`${baseUrl}static/webRTC.png`" alt="webrtc" width="30" />
                <span>WebRTC (PeerJS)</span>
              </li>
              <li class="flex gap-2 items-center">
                <a
                  class="text-blue-500"
                  href="https://github.com/hugozanini/yolov7-tfjs"
                  >YOLO7-tfjs</a
                >
              </li>
              <li class="flex gap-2 items-center">
                <img :src="`${baseUrl}static/Tailwind.png`" alt="tailwind" width="30" />
                <a class="text-blue-500" href="https://tailwindcss.com"
                  >tailwindcss</a
                >
              </li>
              <li class="flex gap-2 items-center">
                <a class="text-blue-500" href="https://pinia.vuejs.org">Pinia</a>
              </li>
            </ul>
          </section>

          <section class="border-b-8 border-orange-500">
            <h3 class="text-gray-700 text-3xl my-5 p-2">F.A.Q</h3>
            <AppQuestion
              v-for="q in questions"
              :key="q.question"
              :question="q.question"
              :answer="q.answer"
            />
            <AppInstruction title="Where can I find the source code?">
              <a
                class="text-blue-500 text-center border-b p-1 border-black"
                href="https://github.com/oxedom/parker"
                >In this repository</a
              >
            </AppInstruction>
          </section>

          <section class="flex flex-col gap-5">
            <h3 class="text-3xl text-gray-800 my-5 p-2">Instructions</h3>

            <AppInstruction title="Mobile phone instructions">
              <div class="space-y-2">
                <p>
                  Navigate to the "Vision" page, press the Remote button, and use
                  your mobile device to scan the QR code. Once the page loads,
                  locate the "Call" button and allow camera access when prompted.
                </p>
                <p>1. Make sure your phone is streaming video in landscape mode.</p>
                <p>
                  2. Disable autolock so the video stream isn't interrupted.
                </p>
                <p>3. You may need to press Call a few times to establish a connection.</p>
              </div>
            </AppInstruction>

            <AppInstruction title="Connect CCTV/IP cameras with OBS">
              <p>
                <strong>If your IP/CCTV cameras are not directly connected: </strong>
                Stream their video using
                <a class="text-blue-500" href="https://obsproject.com/">OBS</a>'s
                Window Capture feature and create a virtual webcam on your PC.
              </p>
              <p>
                <strong>If your IP/CCTV camera is on your local network: </strong>
                Set OBS video capture settings to a local IP address.
              </p>
            </AppInstruction>

            <AppInstruction title="Webcam instructions">
              <p>
                Open the vision page, press the Webcam button, allow webcam
                access, and point it where you want.
              </p>
            </AppInstruction>

            <AppInstruction title="Settings explanation">
              <section class="flex gap-2 flex-col">
                <p>TFJS: Run the TFJS object detection model.</p>
                <p>Show Boxes: Render bboxes on detected items.</p>
                <p>Vehicle Only: Only detect vehicles.</p>
                <p>Detection Threshold: Minimum score for a detection to be valid.</p>
                <p>IOU Threshold: IoU threshold between bboxes.</p>
                <p>Render Rate: Render a frame every N seconds.</p>
              </section>
            </AppInstruction>

            <div class="my-10">
              <h4 class="text-3xl mb-5">Special thanks</h4>
              <ul class="flex flex-col gap-2">
                <li>Hugo Zanini, for porting YOLO7 to tfjs and sharing his code.</li>
                <li>Jason Mayes, for advice and guidance.</li>
                <li>The TensorFlow community.</li>
              </ul>
            </div>
          </section>
        </div>
      </main>
    </div>
  </DefaultLayout>
</template>
