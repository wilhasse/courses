import "./style.css";
import { getVideo, drawVideo } from "./src/camera.js";

(async function run() {
  const button = document.querySelector("button");
  const video = await getVideo();
  const canvas = document.querySelector("canvas");

  button.addEventListener("click", () => {
    drawVideo(video, canvas);
  });
})();
