
function showErrorMessage(logText) {
    console.log(logText);
    $('#videoInfo').text(logText);
}

function addRemoteVideo(remoteStream) {
    // check if not duplicate
    if (connectedStreamIds.includes(remoteStream.id)) {
        console.log('already has the stream in view', remoteStream.id);
        return;
    }
    console.log('info: Remote peer connected');
    connectedStreamIds.push(remoteStream.id);
    console.log(connectedStreamIds);

    // Show stream in some video/canvas element.
    var video = document.getElementById('remoteVideo');
    video.srcObject = remoteStream;
    video.onloadedmetadata = function(e) {
        video.play();
    };
}

function createEmptyStream() {
    function createEmptyAudioTrack() {
        const ctx = new AudioContext();
        const oscillator = ctx.createOscillator();
        const dst = oscillator.connect(ctx.createMediaStreamDestination());
        oscillator.start();
        const track = dst.stream.getAudioTracks()[0];
        return Object.assign(track, {
            enabled: false
        });
    };

    function createEmptyVideoTrack(width, height) {
        const canvas = Object.assign(document.createElement('canvas'), {
            width,
            height
        });
        canvas.getContext('2d').fillRect(0, 0, width, height);

        const stream = canvas.captureStream();
        const track = stream.getVideoTracks()[0];

        return Object.assign(track, {
            enabled: false
        });
    };

    audioTrack = createEmptyAudioTrack();

    videoTrack = createEmptyVideoTrack(640, 480);
    return new MediaStream([audioTrack, videoTrack]);
}

function connectToPeer(pid) {
    window.peer = new Peer();
    window.peer.on('open', function(id) {
        console.log('my peer ID is: ' + id);
        console.log('connecting to peer', pid);
        const nullstream = window.createEmptyStream();
        var call = window.peer.call(pid, nullstream);
        call.on('stream', function(remoteStream) {
            window.stream = remoteStream;
            window.showErrorMessage('Connected to remote stream.');
            $('#btnRecordStart').removeAttr('disabled');
            addRemoteVideo(remoteStream);
        });
    });
    window.peer.on('close', function(id) {
        showErrorMessage('You are disconnected.');
    });
}



var peer;
var stream;
var connectedStreamIds = [];

// recording
var recording = false;
var recorder;
var recording_time = 0;
var recording_time_int;

function connect() {
    $.ajax({
        type: "POST",
        url: './verify-passcode.php',
        data: {
            hash: '9b1c8533dfd522a81ddd8b38b81976c6',
            passcode: $('#f-sharing-passcode').val(),
        },
        success: function(json) {
            var resp = JSON.parse(json);
            if (resp.result) {
                connectToPeer(resp.pid);
                $('#pass-container').hide('slow');
            }
            window.showErrorMessage(resp.message);
        }
    });
}
