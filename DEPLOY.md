# Deploying the annotation UI on Render, with your own domain

This lets anyone with the link open the Pixel Annotation tool in a browser
(no install), and every "Save" is pushed straight to the `annotation-data`
GitHub branch — no manual file-passing needed.

## 1. Create a GitHub token

1. GitHub → Settings → Developer settings → Personal access tokens →
   **Fine-grained tokens** → Generate new token.
2. Repository access: only `Hedgescan-Prototype`.
3. Permissions: **Contents → Read and write**. Nothing else needed.
4. Copy the token now — you won't see it again.

## 2. Create the Render service

1. [render.com](https://render.com) → sign in with GitHub → New → Blueprint.
2. Pick the `Hedgescan-Prototype` repo. Render reads `render.yaml` (already
   in this repo) and proposes a web service — confirm the branch is
   `claude/hedgescan-phase-0-prototype-lyfmww` (or wherever this code lives).
3. When asked for `GITHUB_TOKEN`, paste the token from step 1. It's stored
   as a secret, never committed.
4. Deploy. First build takes a few minutes (installing OpenCV etc.).

## 3. Point your domain at it

1. In the Render service → Settings → Custom Domains → add your domain
   (or a subdomain, e.g. `annotate.yourdomain.com` — simpler, and keeps your
   root domain free for something else).
2. Render shows a CNAME target. Add that CNAME record at your domain
   registrar/DNS provider.
3. Wait for DNS to propagate (usually minutes, sometimes longer) and for
   Render's automatic HTTPS certificate to issue.

## 4. Share it

Send friends the URL. They open **Pixel annotation**, work through photos,
hit **Save annotation for this image** — each save becomes a commit on the
`annotation-data` branch, visible in GitHub right away.

The **Download annotation .txt** button next to Save always works too
(browser download, no GitHub needed) — a manual fallback if the GitHub push
ever fails for someone (e.g. a transient network error), so nobody loses
their work.

## Notes

- Render's free tier sleeps after inactivity and wakes on the next request
  (10-30s delay) — fine for occasional annotation sessions, mention it to
  friends so a slow first load isn't alarming.
- `GITHUB_REPO` and `GITHUB_ANNOTATION_BRANCH` are already set in
  `render.yaml`; only `GITHUB_TOKEN` needs to be entered by hand (Render
  won't let secrets live in a committed file).
- The **Dataset folder** field in the Pixel annotation tab can stay as
  `Dataset` even though that folder won't exist on Render's filesystem —
  the app automatically falls back to fetching photos from the `Dataset`
  GitHub branch via the API when the local folder is empty.
