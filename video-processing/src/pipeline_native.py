"""Arm A — Gemini-native: whole video -> Gemini samples + summarizes; Q&A reuses
the uploaded file handle + chat history. No local frame extraction, no FAISS.
"""

from __future__ import annotations

import os

_SUMMARY_INSTRUCTION = (
    "You are a video analysis system. Watch the entire video and produce a "
    "detailed summary. First give timestamped bullet points of the key events in "
    "the format start_time:end_time: description (use HH:MM:SS.s). Then add a short "
    "'Overall summary:' paragraph (2-4 sentences) describing the video as a whole."
)


class NativePipeline:
    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg
        self._file = None

    def prepare(self, report):
        """Upload the video once; reused by summarize + every Q&A turn."""
        if self._file is None:
            with report.stage("upload") as st:
                self._file = self.client.upload_video(self.cfg.video)
                report.counter("file_size_mb", round(os.path.getsize(self.cfg.video) / 1e6, 2))
        return self._file

    def summarize(self, report) -> str:
        f = self.prepare(report)
        with report.stage("vlm_summarize") as st:
            r = self.client.generate(
                self.cfg.vlm_model, [_SUMMARY_INSTRUCTION, f],
                temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                max_output_tokens=self.cfg.max_output_tokens)
            st.set_tokens(r.in_tokens, r.out_tokens, r.cost_usd)
        return r.text

    def answer(self, question: str, history: list[tuple[str, str]], report) -> str:
        f = self.prepare(report)
        contents = [f]
        if history:
            hist = "\n".join(f"User: {q}\nAI: {a}" for q, a in history[-6:])
            contents.append("Conversation so far:\n" + hist)
        contents.append(
            "Answer this question about the video. Only use what is visible/audible "
            "in the video; if unknown, say so.\nQuestion: " + question)
        with report.stage("qa_generate") as st:
            r = self.client.generate(
                self.cfg.vlm_model, contents,
                temperature=self.cfg.temperature, top_p=self.cfg.top_p,
                max_output_tokens=self.cfg.max_output_tokens)
            st.set_tokens(r.in_tokens, r.out_tokens, r.cost_usd)
        return r.text

    def cleanup(self):
        if self._file is not None:
            self.client.delete_file(self._file)
            self._file = None
