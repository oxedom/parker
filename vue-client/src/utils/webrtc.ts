
interface MediaStreamAudioDestinationNode extends AudioNode {
  stream?: MediaStream;
}

export function createEmptyStream() {
  function createEmptyAudioTrack() {
    const ctx = new AudioContext()
    const oscillator = ctx.createOscillator()
    const dst : MediaStreamAudioDestinationNode = oscillator.connect(ctx.createMediaStreamDestination())
    oscillator.start()
    if(!dst.stream) throw new Error('Error: AudioNode is missing stream property, most likley a browser limitation')
    const track = dst.stream.getAudioTracks()[0]
    return Object.assign(track, {
      enabled: false
    })
  }

  function createEmptyVideoTrack(width : number, height: number) {
    const canvas = Object.assign(document.createElement('canvas'), {
      width,
      height
    })
    const context = canvas.getContext('2d')
    if(context !== null) context.fillRect(0, 0, width, height)
    else throw new Error('Error: Context was null, can not call fillRect method')

    const stream = canvas.captureStream()
    const track = stream.getVideoTracks()[0]

    return Object.assign(track, {
      enabled: false
    })
  }

  const audioTrack = createEmptyAudioTrack()

  const videoTrack = createEmptyVideoTrack(640, 480)
  return new MediaStream([audioTrack, videoTrack])
}
