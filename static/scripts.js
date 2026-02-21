let mediaRecorder;
let audioChunks = [];
const video = document.getElementById("video");

async function init() {

    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
    });

    video.srcObject = stream;

    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = e => {
        audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {

        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        audioChunks = [];

        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0);

        const imageData = canvas.toDataURL("image/png");

        const formData = new FormData();
        formData.append("face", imageData);
        formData.append("audio", audioBlob);

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        document.getElementById("result").innerHTML =
            "Face: " + data.face_emotion +
            "<br>Audio: " + data.audio_emotion +
            "<br><b>Final: " + data.final_emotion + "</b>";
    };
}

function captureAndRecord() {
    audioChunks = [];
    mediaRecorder.start();

    setTimeout(() => {
        mediaRecorder.stop();
    }, 3000);
}

init();
