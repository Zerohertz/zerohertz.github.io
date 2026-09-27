/* global hexo */
"use strict";

// Fill page.description for open_graph() when front matter has none:
// posts/pages with content -> leading text of excerpt/content, others -> site default.
const { stripHTML } = require("hexo-util");

const DEFAULT_DESCRIPTION = "⚡️ Zerohertz's Tech Blog ⚡️";
const MIN_LENGTH = 50;
const MAX_LENGTH = 160;

function summarize(html) {
  const text = stripHTML(
    html
      .replace(/<figure class="highlight[\s\S]*?<\/figure>/g, " ")
      .replace(/<(pre|script|style|h[1-6])[\s\S]*?<\/\1>/g, " ")
      .replace(/<\/(p|li|div|blockquote|td|th)>|<br\s*\/?>/g, " "),
  )
    // refctl reference markers ($_[$$_{1}$$_]$), block math and inline math
    .replace(/\$_\[[\s\S]*?_\]\$/g, "")
    .replace(/\$\$[\s\S]*?\$\$/g, "")
    .replace(/\$[^$\n]{1,80}\$/g, "")
    .replace(/\s+/g, " ")
    .replace(/\s+([.,])/g, "$1")
    .trim();
  return text.length > MAX_LENGTH
    ? `${text.substring(0, MAX_LENGTH).trim()}…`
    : text;
}

hexo.extend.filter.register("template_locals", (locals) => {
  const { page } = locals;
  if (page.description) return locals;
  // Short excerpts ("실행결과") lose to the content; posts with little prose use the title.
  const [excerpt, content] = [page.excerpt, page.content].map((html) =>
    html ? summarize(html) : "",
  );
  page.description =
    [excerpt, content].find((text) => text.length >= MIN_LENGTH) ||
    page.title ||
    DEFAULT_DESCRIPTION;
  return locals;
});
