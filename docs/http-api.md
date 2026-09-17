# HTTP API

The server implements a subset of OpenAI-compatible Chat Completions and
Completions. [Launch it](launch.md) with a local bundle and use its configured
model name in requests.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness; successful when the model is ready |
| `GET /v1/models` | List the configured model |
| `GET /v1/models/{id}` | Retrieve that model |
| `POST /v1/chat/completions` | Text/image chat, buffered or SSE |
| `POST /v1/completions` | Raw text or token-ID prompt, buffered or SSE |
| `POST /v1/cache/prefill` | Prepare a prefix without generating output |
| `POST /v1/cache/finish` | Release named prefix retention |
| `GET /v1/cache/stats` | Cache capacity and use |
| `GET /v1/cache/index` | Retained prefixes and active executions |
| `GET /metrics` | Prometheus metrics |

## Generation

Chat accepts `messages` with system, developer, user, assistant, and tool roles.
Assistant history can carry `reasoning_content` and `tool_calls`; tool messages
must follow their assistant call and carry the matching `tool_call_id`. Raw
Completions accepts a string or one token-ID array in `prompt`. Each request
produces one choice. Supported sampling controls are `temperature`, `top_p`,
`top_k`, and unsigned 64-bit `seed`. Use `max_tokens`, or
`max_completion_tokens` for chat, to bound output. EOS stops generation.

Set `stream:true` for SSE. `stream_options.include_usage:true` requests a final
usage event; otherwise streaming omits usage. Buffered responses always include
usage, including `prompt_tokens_details.cached_tokens`. The total context
horizon is at most 262,144 processed tokens and can be lower for a particular
memory configuration.

`stop` accepts a string or an array of strings; matched stop text is removed
from the output. `chat_template_kwargs` accepts the booleans `enable_thinking`
and `preserve_thinking` and rejects every other key; rendering follows the
built-in Gemma 4 template described in the [bundle guide](models.md). Reasoning is
returned separately as `reasoning_content`. Chat also accepts `logprobs:true`
and `top_logprobs` from 0 through 20; supplying `top_logprobs` requires logprobs
to be enabled. Raw Completions' integer `logprobs` control is unsupported.

## Tools and JSON answers

Chat supports function definitions in `tools`, generated `tool_calls`, and
matching tool-result history. `tool_choice` supports `auto` and `none`;
required or named selection and strict argument schemas are not implemented.
The server never executes tools; the client runs them and submits the
results.

`response_format` supports `text`, `json_object`, and `json_schema`. Example:

```json
{
  "model": "gemma-4-31b",
  "messages": [{"role": "user", "content": "Return a short greeting."}],
  "max_tokens": 128,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "greeting",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
        "additionalProperties": false
      }
    }
  }
}
```

Constraints apply to the answer channel and work with MTP. Stop strings cannot
be combined with a constrained response format. If tool definitions are also
supplied, set `tool_choice:"none"`. Invalid or unsupported schemas receive a
request error. A response stopped by the token limit can still be incomplete.

## Images

User message content can contain text parts and `image_url` parts. Images must
be inline base64 PNG or JPEG data URLs; remote URLs and local file URLs are not
fetched. `image_url.detail` can be omitted or set to `auto`. Multiple images
and images in multiple user turns are supported.

```json
{"role":"user","content":[
  {"type":"text","text":"Describe this picture."},
  {"type":"image_url","image_url":{"url":"data:image/png;base64,BASE64_IMAGE_BYTES"}}
]}
```

The server decodes and prepares images natively. Body, pixel, tensor, and GPU
capacity limits apply. Image identity participates in prefix caching.

## Cache

Prefix reuse is automatic. The optional `cache` object supports `mode` (`auto`
or `reuse_only`) and named retention through `prompt_id`, `priority`, and
`finished`. `reuse_only` reuses existing prefixes without retaining new ones
and cannot carry owner controls. Named retention consumes the configured
cache budget; release it through `/v1/cache/finish` when finished.

`/v1/cache/prefill` accepts `prompt_id` inside `cache`, for example
`{"prompt":"Hello","cache":{"prompt_id":"document"}}`.
`/v1/cache/finish` instead takes `{"prompt_id":"document"}` at the top level.
The [cache guide](cache.md) explains matching, ownership, `reuse_only`, shared
prefill, checkpoint storage, GPU/CPU eviction, and usage accounting with diagrams
and complete examples.

## Transport and validation

The listener uses HTTP/1.1 with `Content-Length` request bodies and
`Connection: close` responses. Chunked uploads and persistent HTTP connections
are outside this subset. Disconnecting, including a request-side TCP
half-close, cancels the associated work. Slow output consumers are bounded by
the configured buffers and timeouts.

Recognized controls are accepted only at their neutral value: `n:1`, zero
`frequency_penalty`/`presence_penalty`, empty `logit_bias`, and, in chat,
`store:false`, `modalities:["text"]`, and `parallel_tool_calls:true`. Always
rejected are `functions`, `function_call`, `reasoning_effort`, and the
prompt-cleanup fields `audio`, `moderation`, `prediction`,
`prompt_cache_retention`, `service_tier`, `verbosity`, and
`web_search_options`. Each endpoint also rejects the other endpoint's fields:
chat rejects `prompt`, `echo`, `best_of`, and `suffix`; Completions rejects
`messages`, `max_completion_tokens`, `chat_template_kwargs`, `top_logprobs`,
`logprobs`, `response_format`, `tools`, `tool_choice`, `store`, `modalities`,
and `parallel_tool_calls`, and accepts `echo` and `best_of` only at their
defaults. Unknown top-level fields are ignored. Nested messages, tools,
schemas, and cache objects are validated. There is no `/v1/responses`
endpoint. Error responses include an `x-request-id` for matching server logs;
an error after streaming has started is reported within the stream.
