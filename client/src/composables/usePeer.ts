import { onUnmounted, ref, shallowRef } from "vue";
import type { DataConnection, MediaConnection, Peer as PeerType } from "peerjs";

type Handler<T> = (payload: T) => void;

export function usePeer() {
  const peer = shallowRef<PeerType | null>(null);
  const peerId = ref<string>("");
  const remoteConn = shallowRef<DataConnection | null>(null);
  const incomingCall = shallowRef<MediaConnection | null>(null);
  const ready = ref(false);

  let onCallStreamCb: Handler<MediaStream> | null = null;
  let onConnectionCb: Handler<DataConnection> | null = null;

  async function init() {
    if (peer.value) return;
    const { default: Peer } = await import("peerjs");
    const p = new Peer();
    peer.value = p;

    p.on("open", (id) => {
      peerId.value = id;
      ready.value = true;
    });
    p.on("connection", (conn) => {
      remoteConn.value = conn;
      onConnectionCb?.(conn);
    });
    p.on("call", (call) => {
      incomingCall.value = call;
      call.on("stream", (stream) => onCallStreamCb?.(stream));
    });
    p.on("error", (err) => console.error("Peer error", err));
  }

  function answerWith(stream: MediaStream) {
    incomingCall.value?.answer(stream);
  }

  function send(data: unknown) {
    remoteConn.value?.send(data as object);
  }

  function connectTo(remoteId: string): DataConnection | null {
    if (!peer.value) return null;
    return peer.value.connect(remoteId);
  }

  function call(remoteId: string, stream: MediaStream): MediaConnection | null {
    if (!peer.value) return null;
    return peer.value.call(remoteId, stream);
  }

  function onIncomingStream(cb: Handler<MediaStream>) {
    onCallStreamCb = cb;
  }
  function onIncomingConnection(cb: Handler<DataConnection>) {
    onConnectionCb = cb;
  }

  function destroy() {
    peer.value?.destroy();
    peer.value = null;
    ready.value = false;
  }

  onUnmounted(destroy);

  return {
    peer,
    peerId,
    ready,
    remoteConn,
    init,
    answerWith,
    send,
    connectTo,
    call,
    onIncomingStream,
    onIncomingConnection,
    destroy,
  };
}
