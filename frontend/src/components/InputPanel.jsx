import { useEffect, useRef, useState } from "react";

export default function InputPanel({ onSubmit, loading }) {
  const [text, setText] = useState("");
  const [image, setImage] = useState(null);
  const [video, setVideo] = useState(null);
  const [preview, setPreview] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [micSupported, setMicSupported] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [micError, setMicError] = useState("");
  const fileRef = useRef();
  const videoRef = useRef();
  const recognitionRef = useRef(null);

  useEffect(() => {
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    setMicSupported(true);
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = true;

    recognition.onresult = (event) => {
      let transcript = "";
      for (let i = 0; i < event.results.length; i += 1) {
        transcript += event.results[i][0]?.transcript || "";
      }
      setText(transcript.trim());
    };

    recognition.onstart = () => {
      setMicError("");
      setIsListening(true);
    };
    recognition.onend = () => setIsListening(false);
    recognition.onerror = (event) => {
      setIsListening(false);
      setMicError(`Mic error: ${event.error}`);
    };

    recognitionRef.current = recognition;
    return () => {
      try {
        recognition.stop();
      } catch {
        /* no-op */
      }
    };
  }, []);

  function handleFile(file) {
    if (!file) return;
    if (file.type.startsWith("image/")) {
      setImage(file);
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target.result);
      reader.readAsDataURL(file);
    } else if (file.type.startsWith("video/")) {
      handleVideoFile(file);
    }
  }

  function handleVideoFile(file) {
    if (!file || !file.type.startsWith("video/")) return;
    setVideo(file);
    setVideoPreview(URL.createObjectURL(file));
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!text.trim() && !image && !video) return;
    onSubmit({ text, image, video });
  }

  function clearImage() {
    setImage(null);
    setPreview(null);
    if (fileRef.current) fileRef.current.value = "";
  }

  function clearVideo() {
    if (videoPreview) URL.revokeObjectURL(videoPreview);
    setVideo(null);
    setVideoPreview(null);
    if (videoRef.current) videoRef.current.value = "";
  }

  function toggleMic() {
    if (!recognitionRef.current) return;
    if (isListening) {
      recognitionRef.current.stop();
      return;
    }
    setMicError("");
    try {
      recognitionRef.current.start();
    } catch {
      setMicError("Microphone could not start. Check browser permissions.");
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div>
        <label className="block text-sm font-medium text-slate-400 mb-2">
          Describe the task
        </label>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-slate-500">
              {isListening ? "Listening..." : "Type or use mic"}
            </span>
            {isListening && (
              <span
                aria-label="live microphone indicator"
                className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full border border-danger/40 bg-danger/15"
              >
                <span className="w-2 h-2 rounded-full bg-danger animate-pulse-dot" />
                <span className="w-1 h-3 rounded-full bg-danger/80 animate-pulse-dot" style={{ animationDelay: "0.1s" }} />
                <span className="w-1 h-4 rounded-full bg-danger animate-pulse-dot" style={{ animationDelay: "0.2s" }} />
                <span className="w-1 h-2 rounded-full bg-danger/70 animate-pulse-dot" style={{ animationDelay: "0.3s" }} />
              </span>
            )}
          </div>
          {micSupported ? (
            <button
              type="button"
              onClick={toggleMic}
              className={`text-xs px-2.5 py-1 rounded border transition-colors ${
                isListening
                  ? "border-danger/40 text-danger bg-danger/10"
                  : "border-dark-500 text-slate-400 hover:text-slate-300 hover:border-dark-400"
              }`}
            >
              <span className="inline-flex items-center gap-1.5">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 3a3 3 0 00-3 3v6a3 3 0 106 0V6a3 3 0 00-3-3zm0 18v-3m-6-6a6 6 0 0012 0"
                  />
                </svg>
                {isListening ? "Stop Mic" : "Mic"}
              </span>
            </button>
          ) : (
            <span className="text-[11px] text-slate-600">Mic unavailable</span>
          )}
        </div>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="e.g. Inspect Zone B loading dock for safety violations..."
          rows={4}
          className="w-full bg-dark-700 border border-dark-500 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-accent/50 focus:border-accent resize-none transition-all"
        />
        {micError && (
          <p className="text-[11px] text-warning mt-2">{micError}</p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-400 mb-2">
          Upload image
        </label>
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
          className={`relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all ${
            dragOver
              ? "drop-zone-active"
              : "border-dark-500 hover:border-dark-400"
          }`}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />

          {preview ? (
            <div className="relative">
              <img
                src={preview}
                alt="Preview"
                className="max-h-48 mx-auto rounded-lg object-contain"
              />
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); clearImage(); }}
                className="absolute top-1 right-1 bg-dark-900/80 hover:bg-danger text-white rounded-full w-7 h-7 flex items-center justify-center text-sm transition-colors"
              >
                ×
              </button>
              <p className="text-xs text-slate-500 mt-2">{image?.name}</p>
            </div>
          ) : (
            <div className="text-slate-500">
              <svg className="w-10 h-10 mx-auto mb-2 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <p className="text-sm">Drop an image here or click to browse</p>
            </div>
          )}
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-400 mb-2">
          Upload video
        </label>
        <div
          onClick={() => videoRef.current?.click()}
          className="relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all border-dark-500 hover:border-dark-400"
        >
          <input
            ref={videoRef}
            type="file"
            accept="video/mp4,video/webm,video/quicktime,video/x-msvideo"
            className="hidden"
            onChange={(e) => handleVideoFile(e.target.files[0])}
          />

          {videoPreview ? (
            <div className="relative">
              <video
                src={videoPreview}
                className="max-h-48 mx-auto rounded-lg"
                controls
                muted
              />
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); clearVideo(); }}
                className="absolute top-1 right-1 bg-dark-900/80 hover:bg-danger text-white rounded-full w-7 h-7 flex items-center justify-center text-sm transition-colors"
              >
                ×
              </button>
              <p className="text-xs text-slate-500 mt-2">{video?.name}</p>
            </div>
          ) : (
            <div className="text-slate-500">
              <svg className="w-10 h-10 mx-auto mb-2 opacity-40" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <p className="text-sm">Upload a video for frame-by-frame analysis</p>
              <p className="text-[11px] text-slate-600 mt-1">MP4, WebM, MOV, AVI</p>
            </div>
          )}
        </div>
      </div>

      <button
        type="submit"
        disabled={loading || (!text.trim() && !image && !video)}
        className="w-full py-3 px-6 bg-accent hover:bg-accent-light disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-all flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Running Pipeline...
          </>
        ) : (
          <>
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Run Agent
          </>
        )}
      </button>
    </form>
  );
}
