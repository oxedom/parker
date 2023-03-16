export  async function callPeer(peerID, videoEl, peer) 
{
    // let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    let emptyStream = new MediaStream()
    const call = peer.call(peerID, emptyStream)

    call.on('stream', (remoteStream) => {
    console.log('REMOTE STREAM FROM INPUT')
    console.log(remoteStream)
    videoEl.current.srcObject = remoteStream
    videoEl.current.play();
  });

  
}

