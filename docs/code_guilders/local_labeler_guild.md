Online login link:  [http://100.89.98.89:7861/login](http://100.89.98.89:7861/login "http://100.89.98.89:7861/login")

# Play QA videos from your laptop (slow-internet mode)

If the QA website is slow because video takes forever to load, you can **copy your
assigned videos onto your laptop once** (at home, on fast wifi) and have the editor
play them **from your own disk** at the office. Everything else — your assignment
list, the captions, saving your edits — still talks to the server as normal. Only
the heavy video bytes come from local.

---

## How it works (the short version)

There are two computers involved:

- **The server** — `100.89.98.89`, runs the website and holds all the videos.
- **Your laptop** — where you want the videos to live so they play fast.

```
your browser ──► http://localhost:8090 ──► local_video_server.py ──► ~/caption_videos/<video>.mp4
                 (a local "door" the browser is allowed to use)        (the files on your disk)
```

A webpage isn't allowed to read files off your disk directly, so a tiny program
(`local_video_server.py`) runs on your laptop and hands the videos to your browser
over `http://localhost`. `localhost` means your own machine, so it never touches
the slow office internet.

---

## One-time setup — get `local_video_server.py` onto your laptop

You only need the website (no SSH required):

1. Log in and open the editor at `http://100.89.98.89:7861`.
2. Click the **⚙ Settings** button in the top toolbar.
3. Click **⬇ Local player script** — this downloads `local_video_server.py` to
   your laptop. Save it somewhere you'll remember, e.g. your home folder.

That's it. No special software needed beyond **Python 3**. (If `python` isn't
found later, use `python3` instead.)

> Already have SSH access to the server? You can skip the button and
> `scp tho2@100.89.98.89:/home/tho2/web_demo_bao/scripts/local_video_server.py .`
> instead — same file either way.

---

## Step 1 — At home (fast internet): download your videos

1. In the editor, click **⚙ Settings** → **⬇ Download videos**.
2. This zips up your whole not-done queue (assigned + in-progress) and downloads
   it as `<you>_assigned_videos.zip`. For a full queue (~1000 episodes) this is a
   few GB, so it can take a while — **don't close the tab until the browser shows
   the download finished**.
3. Unzip it into `~/caption_videos` (the folder structure inside the zip already
   matches what `local_video_server.py` expects — just extract in place, don't
   flatten it).

**This is a single big download, not resumable.** If your connection drops
partway through, you'll need to click the button again and re-download the
whole zip — there's no "top up just the new ones" like before. If that's a
recurring problem on your connection, mention it; we can look at a
one-at-a-time option.

---

## Step 2 — At the office: play from your laptop

1. Start the local video server and **leave this terminal open**:

   ```bash
   python local_video_server.py ~/caption_videos
   ```

   You'll see `Serving N local videos … Listening on http://127.0.0.1:8090`.
2. Open the website (`http://100.89.98.89:7861`), log in, click **⚙ Settings**,
   and click **`Local: off`** so it becomes **`Local: on`**.
3. Review normally. Downloaded videos play instantly from your disk. Anything you
   *didn't* download quietly streams from the server (you'll see a brief
   "Not in local cache — streaming from server" note).

When you're done, press **Ctrl+C** in that terminal to stop the local server.

---

## Good to know

- **The toggle is sticky** — it stays on across page reloads. Just remember the
  local server has to be running, or you'll get the streaming fallback.
- **Your edits are never stuck on the laptop** — saving always goes to the server.
- **Re-download anytime** new videos are assigned to you (Step 1 again — it
  re-zips your whole current queue, so you'll get everything again, not just the
  new ones).
- **Disk space**: full-quality episodes are ~3–5 MB each; a few thousand is a few
  GB.

---

## Troubleshooting

| Problem                                        | Fix                                                                                                                                |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `python: command not found`                  | use`python3` instead                                                                                                             |
| Download zip seems stuck / very slow           | it's building server-side before sending — a ~1000-video queue can take a minute or two; leave the tab open                       |
| Toggle is on but videos still slow             | the local server isn't running — start`local_video_server.py`, or it's a video you didn't download (it falls back to streaming) |
| Local server says`Video root does not exist` | run Step 1 first, or check the`~/caption_videos` path matches                                                                    |
| Changed the server's`--port`                 | in the browser console:`localStorage.setItem('localVideoBase','http://localhost:<port>')`                                        |
