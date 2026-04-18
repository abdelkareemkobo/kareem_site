---
title: "HyperRun Deep Dive: Streaming Chat Architecture with FastHTML, HTMX, and SSE"
description: How I built a real-time AI chat widget with zero React, zero WebSockets, and almost zero custom JavaScript. A deep dive into FastHTML's server-rendered approach — SSE streaming, DaisyUI modals with checkbox toggles, OOB swaps, session-based state, and an embed trick that works on any static site with one line of code.
author: kareem
date: 2026-03-19
draft: false
featured: false
categories:
  - blogging
  - til
  - blog/build/project
image: images/colgrep.jpg
---

## AI Widget with HTMX vs React JavaScript

Every AI chat widget I've seen is built the same way: 

React frontend + WebSocket connection +  custom state management +  a build step. 

For something that's basically "send text, get text back", that's a lot of machinery.

FastHTML + HTMX gives you a different deal:

- Server renders the HTML — no client-side framework

- HTMX handles interactions — no custom JS for fetching/swapping

- SSE handles streaming — simpler than WebSockets for one-directional data

- Python all the way down — the UI is just functions returning FT components

The entire HyperRun UI is one Python file.

**No `package.json`, no `node_modules`, no build step.**

****
```markdown

      ┌─────────────────────────────────────────────────────────┐
      │                    Doc Author's Site                    │
      │  (Quarto / nbdev / MkDocs / GitHub Pages)               │
      │                                                         │
      │  <script src="https://server/embed"></script>           │
      │         │                                               │
      │         ▼                                               │
      │  ┌─────────────┐                                        │
      │  │ 💬 Ask AI   │ ◄── floating button (injected by JS)   │
      │  └──────┬──────┘                                        │
      │         │ click                                         │
      │         ▼                                               │
      │  ┌─────────────────────────┐                            │
      │  │  iframe                 │                            │
      │  │  /chat-standalone       │──────────────────┐         │
      │  │                         │                  │         │
      │  └─────────────────────────┘                  │         │
      └───────────────────────────────────────────────┼─────────┘
                                                      │
                                                      ▼
                                        ┌────────────────────┐
                                        │  HyperRun Server   │
                                        │  (FastHTML + HTMX) │
                                        └────────────────────┘
```

## FastHTML App Setup

```markdown 

  User types "what does insert_article do?"
        │
        ▼
  ┌──────────┐  POST /ask   ┌──────────────┐
  │  Browser │────────────►│  FastHTML     │
  │  (HTMX)  │             │  /ask route   │
  └──────────┘              └──────┬───────┘
        ▲                          │
        │                          ▼
        │                   ┌──────────────┐
        │                   │ Returns:     │
        │                   │ • user bubble│
        │                   │ • ai bubble  │
        │                   │   (with SSE) │
        │                   │ • new input  │
        │                   │   (OOB swap) │
        │                   └──────┬───────┘
        │                          │
        │  SSE connects to         │
        │  /stream?query=...       ▼
        │                   ┌──────────────┐
        │  sse: message     │  /stream     │
        │◄──────────────────│  async gen   │
        │  sse: message     │      │       │
        │◄──────────────────│      ▼       │
        │  sse: message     │  ┌────────┐  │
        │◄──────────────────│  │ColGrep │  │
        │  sse: close       │  │search  │  │
        │◄──────────────────│  └───┬────┘  │
        │                   │      │       │
        ▼                   │      ▼       │
    marked.parse()          │  ┌────────┐  │
    renders markdown        │  │Lisette │  │
                            │  │AsyncChat│ │
                            │  │+ tools │  │
                            │  └────────┘  │
                            └──────────────┘
```

Setting up the app means picking your CSS framework and loading the right headers:

```python
from fasthtml.common import *
from fhdaisy import *
from fastlucide import SvgSprites, SvgStyle

icons = SvgSprites()

app, rt = fast_app(
    pico=False,
    hdrs=(
        *daisy_hdrs,
        SvgStyle(),
        CHAT_CSS,
        Script(src="https://unpkg.com/htmx-ext-sse@2.2.3/sse.js"),
        Script(src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"),
    ))
```

A few things to note:

- `pico=False` — FastHTML includes PicoCSS by default, we swap it for DaisyUI
- `daisy_hdrs` — from `fhdaisy`, loads Tailwind + DaisyUI via CDN
- `SvgSprites()` — from `fastlucide`, gives us Lucide icons as inline SVGs
- SSE extension and `marked.js` are the only external JS we load

That's the whole frontend stack. 

No bundler needed.

## The Chat Widget Architecture

### Modal with Checkbox Toggle

DaisyUI has a modal component that opens/closes with a hidden checkbox — pure CSS, no JS:

```python
def chat_widget():
    return Div(
        icons,
        Label(
            Span(cls="inline-flex items-center gap-2")(
                icons("message-circle", sz=20),
                Span("Ask AI", cls="font-semibold tracking-tight")),
            fr="chat-modal",
            cls="btn btn-primary rounded-full fixed bottom-6 right-6 z-50"),
        Input(type="checkbox", id="chat-modal", cls="modal-toggle"),
        Div(cls="modal")(
            chat_content(),
            Label(cls="modal-backdrop", fr="chat-modal")))
```

Click the `Label` → toggles the checkbox → CSS shows/hides the modal.

Click the backdrop → same checkbox → closes it. 
Zero JavaScript.

> **Gotcha:** The `Input(modal-toggle)` must be immediately before the `Div(modal)` — DaisyUI uses a CSS sibling selector.
>
> We learned this the hard way.

### Reusable Component

The same `chat_content()` function powers both the modal widget and the standalone page:

```python
def chat_content(standalone=False):
    box_cls = "w-full h-screen p-0 bg-base-100" if standalone else "modal-box w-11/12 max-w-4xl p-0 ..."
    return Div(
        # header, messages, input form
        cls=box_cls)
```

One component, two contexts — no duplication.

### Chat Bubbles

DaisyUI's `chat` component gives us styled message bubbles for free:

```python
def user_bubble(msg):
    return Div(cls="chat chat-end")(
        Div("You", cls="chat-header"),
        Div(msg, cls="chat-bubble chat-bubble-primary"))
```

`chat-end` = right-aligned (user).

`chat-start` = left-aligned (AI). 

DaisyUI handles the speech bubble shapes, spacing, everything.

## Streaming with SSE + HTMX

```markdown

**Streaming Life Cycle**

  Browser (HTMX)                          Server (FastHTML)
       │                                        │
       │  ── SSE connect /stream ──────────►    │
       │                                        │
       │                              AsyncChat(query, stream=True)
       │                                        │
       │  ◄── sse: message (Span("I'll")) ──    │
       │  ◄── sse: message (Span(" search"))──  │
       │  ◄── sse: message (Span(" the")) ──    │
       │  ◄── sse: message (Span(" code"))──    │
       │           ...                          │
       │  ◄── sse: message (Span("...")) ──     │
       │  ◄── sse: close (Span("")) ────────    │
       │                                        │
       │  disconnect                            │
       │                                        │
       ▼                                        
  hx_on__sse_close fires:
  this.innerHTML = marked.parse(this.textContent)
  
  Raw text ──► Rendered markdown HTML

```
Why SSE over WebSockets

Chat is one-directional — the server streams tokens to the client. 

WebSockets are bidirectional, which is overkill here.

SSE is:

- Built into browsers natively
- Supported by HTMX via a small extension
- Dead simple in FastHTML — just return an EventStream

### The SSE Lifecycle

The AI response bubble sets up the SSE connection:

```python
def ai_bubble_stream(query):
    return Div(
        Span(cls="loading loading-dots loading-sm"),
        hx_ext="sse",
        sse_connect=stream_response.to(query=query),
        sse_swap="message",
        sse_close="close",
        hx_swap="beforeend",
        hx_on__sse_close="this.innerHTML=marked.parse(this.textContent)",
        cls="chat-bubble bg-base-200")
```

Here's what each attribute does:

- hx_ext="sse" — activates the SSE extension on this element

- sse_connect — URL to connect to, built with .to() which handles URL encoding

- sse_swap="message" — swap in content when an event named message arrives

- sse_close="close" — disconnect when an event named close arrives

- hx_swap="beforeend" — append each chunk (don't replace)

- hx_on__sse_close — when stream ends, render the accumulated text as markdown

### The Server Side

FastHTML makes SSE trivial — an async generator that yields sse_message():

```python
@rt("/stream")
async def stream_response(query: str, sess):
    chat = chats[sid]

    async def generate():
        async for chunk in await chat(query, stream=True):
            text = chunk.choices[0].delta.content
            if text:
                yield sse_message(Span(text))
        yield sse_message(Span(""), event="close")

    return EventStream(generate())
```

Each sse_message(Span(text)) sends a chunk of HTML.
HTMX appends it into the bubble.

When we're done, event="close" tells HTMX to disconnect.

### Markdown Rendering

During streaming, the text arrives as plain chunks.

We can't render markdown mid-stream because a **bold** might be split across two chunks.

The solution: accumulate raw text during streaming, then render it all at once when the stream closes.

marked.parse() on the client converts the accumulated textContent to HTML.

>Gotcha: SSE auto-reconnects by default. Without sse_close="close", the connection reopens and you get triple responses.

## HTMX Patterns We Used

### OOB Swaps — Clearing the Input

After the user submits a question, the input should clear. 

The HTMX-idiomatic way is an out-of-band swap — return a fresh empty input from the server alongside the response:

```python
@rt("/ask")
def ask(query: str, sess):
    new_inp = Input(id="query", placeholder="Ask a follow-up question...",
                    hx_swap_oob="true")
    return Div(user_bubble(query), ai_bubble_stream(query)), new_inp
```

hx_swap_oob="true" tells HTMX: "find the element with this id on the page and replace it." No JavaScript needed. 

The main response goes to #messages, the input replacement happens out-of-band.

### Form Submit — Free Enter Key

Instead of putting hx_get on the button and wiring up keyboard events, wrap everything in a Form:

```python
Form(cls="join w-full", hx_post=ask, hx_target="#messages", hx_swap="beforeend")(
    Input(id="query", name="query"),
    Button(icons("send", sz=18)))
```

Forms submit on Enter automatically. 
One less thing to build.

### .to() for Safe URLs

FastHTML route functions have a .to() method that generates URL-encoded paths:

sse_connect=stream_response.to(query=query, msg_id=msg_id)

No manual urllib.parse.quote(). It handles spaces, special characters, everything.

### Session-Based State

FastHTML's session (signed cookie) stores per-user state — selected model, API key, cost tracking:

```python
@rt("/stream")
async def stream_response(query: str, sess):
    model = sess.get("model", "claude-sonnet-4-20250514")
    api_key = sess.get("api_key", None)
```

Chat history lives in a server-side dict keyed by session ID:

chats = {}  # session_id → AsyncChat instance

The AsyncChat from Lisette maintains conversation history internally — just reuse the same instance and it remembers everything.

## The Embed Trick

### Dynamic /embed Route

The hardest part of embedding a widget on external sites is the server URL — you don't want the doc author manually editing URLs. So HyperRun serves its own embed script:

```python
@rt("/embed")
def embed_js(req):
    server = f"{req.url.scheme}://{req.url.netloc}"
    js = f"""document.addEventListener('DOMContentLoaded',function(){{
      // creates floating button + iframe pointing to {server}/chat-standalone
    }});"""
    return Response(js, media_type='application/javascript; charset=utf-8')
```

The key: req.url.scheme and req.url.netloc give us the server's own URL. 

Deploy to `https://myapp.fly.dev` and the generated JS automatically uses that URL.

 Zero configuration for the doc author.

### /chat-standalone — Full Page for iframe

The same chat_content() component gets wrapped in a complete HTML page with its own HTMX and DaisyUI headers:

```python
@rt("/chat-standalone")
def chat_standalone():
    return Html(
        Head(*daisy_hdrs, SvgStyle(), CHAT_CSS,
             Script(src="<https://unpkg.com/htmx.org>"),
             Script(src="<https://unpkg.com/htmx-ext-sse@2.2.3/sse.js>"),
             Script(src="<https://cdn.jsdelivr.net/npm/marked/marked.min.js>")),
        Body(icons, chat_content(standalone=True)))
```

The iframe is fully self-contained — it doesn't depend on anything from the host page.

### Cross-Origin Close Button

In the modal version, the close button is a Label that toggles a checkbox.

In the iframe, there's no checkbox. 

So the standalone close button sends a postMessage to the parent page:

Button(icons("x"), onclick="window.parent.postMessage('hyperrun-close','*')")

And the /embed script listens for it:

```python
window.addEventListener('message', function(e) {
    if (e.data === 'hyperrun-close')
        document.getElementById('hr-frame').style.display = 'none'
});

```

### One Line to Embed Anywhere

The doc author's entire setup:

#### Quarto/nbdev (_quarto.yml)

include-after-body:

- text: '<script src="https://your-server/embed"></script>'

#### MkDocs (mkdocs.yml)

extra_javascript:

- <https://your-server/embed>

#### Any HTML page

<script src="https://your-server/embed"></script>

One line. Works everywhere.


## Lessons Learned Using Streaming HTMX

Building HyperRun was mostly smooth — but a few things bit us. 

Here's what to watch for.

### SSE Auto-Reconnect = Triple Responses

The HTMX SSE extension reconnects automatically when the connection closes. 

If you don't explicitly tell it to stop, it reconnects and the LLM response streams three times.

The fix: `sse_close="close"` on the element, and send `event="close"` as the last SSE message:

```python
yield sse_message(Span(""), event="close")
```

Simple, but took a while to figure out why every answer appeared three times.

### DaisyUI Modal Checkbox Ordering

DaisyUI's checkbox modal uses a CSS sibling selector — the `modal-toggle` input must be **immediately before** the `modal` div. 

Put anything between them and the modal silently breaks.

This doesn't work:

```python
Input(cls="modal-toggle")  # ✓ checkbox
Label(...)                  # ✗ this breaks the sibling selector
Div(cls="modal")(...)       # modal never opens
```

This works:

```python
Label(...)                  # button can go anywhere before
Input(cls="modal-toggle")  # ✓ immediately before modal
Div(cls="modal")(...)       # ✓ works
```

No error, no warning. Just a modal that doesn't open.

### Markdown Rendering Timing

You can't render markdown per-chunk during streaming — a `**bold**` might arrive as `**bo` in one chunk and `ld**` in the next.

We tried server-side rendering with mistletoe, client-side with `marked.js`, and various event-driven approaches. 

The simplest winner: accumulate plain text during streaming, then `marked.parse(this.textContent)` on `sse_close`. One line.

### `""` Returns Are Falsy in FastHTML

When we needed a route to clear content (like closing the chat panel), returning `""` didn't work — FastHTML treats empty strings as falsy.

Returning `Div()` instead gives you a proper empty element that HTMX swaps in correctly.

## Final Thoughts 

The stack — FastHTML + HTMX + SSE + DaisyUI — turned out to be surprisingly productive for building a real-time chat widget.
No build step, no React, no client-side state management.
Just Python functions returning HTML.

You can read more about other related blogs here: 

1. [HyperRun](https://kareemai.com/blog/posts/nlp/embedding_world/hyperrun.html)

2. [SeoRat](https://kareemai.com/blog/posts/seo/seo_rat_journey.html)

3. [BM25 Search](https://kareemai.com/blog/posts/nlp/embedding_world/sparse_embedding/bm25_from_scratch.html)

4. [BM25 Explained](https://kareemai.com/blog/posts/nlp/embedding_world/sparse_embedding/bm25_arabic_qdrant.html)

5. [BM25 Benchmark](https://kareemai.com/blog/posts/nlp/embedding_world/sparse_embedding/bm25_benchmark_full.html) 
