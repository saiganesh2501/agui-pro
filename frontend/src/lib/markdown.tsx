import React from "react";

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(text: string): string {
  let t = escapeHtml(text);
  t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  t = t.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );
  return t;
}

type Block = { type: "code"; lang: string; content: string } | { type: "html"; content: string };

function parse(src: string): Block[] {
  const blocks: Block[] = [];
  const lines = src.split("\n");
  let i = 0;
  let para: string[] = [];
  let list: string[] = [];
  const flushP = () => {
    if (para.length) {
      blocks.push({ type: "html", content: `<p>${inline(para.join(" "))}</p>` });
      para = [];
    }
  };
  const flushL = () => {
    if (list.length) {
      blocks.push({
        type: "html",
        content: `<ul>${list.map((li) => `<li>${inline(li)}</li>`).join("")}</ul>`,
      });
      list = [];
    }
  };
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^```(\w+)?\s*$/);
    if (fence) {
      flushP();
      flushL();
      const lang = fence[1] ?? "";
      const buf: string[] = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++;
      blocks.push({ type: "code", lang, content: buf.join("\n") });
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      flushP();
      flushL();
      blocks.push({ type: "html", content: `<h${h[1].length}>${inline(h[2])}</h${h[1].length}>` });
      i++;
      continue;
    }
    const li = line.match(/^[-*]\s+(.*)$/);
    if (li) {
      flushP();
      list.push(li[1]);
      i++;
      continue;
    }
    if (line.trim() === "") {
      flushP();
      flushL();
      i++;
      continue;
    }
    flushL();
    para.push(line);
    i++;
  }
  flushP();
  flushL();
  return blocks;
}

export function Markdown({ text }: { text: string }) {
  const blocks = parse(text);
  return (
    <div className="md">
      {blocks.map((b, i) =>
        b.type === "code" ? (
          <CodeBlock key={i} lang={b.lang} code={b.content} />
        ) : (
          <div key={i} dangerouslySetInnerHTML={{ __html: b.content }} />
        )
      )}
    </div>
  );
}

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <div className="codeblock">
      <div className="codeblock-head">
        <span>{lang || "code"}</span>
        <button
          onClick={() => {
            navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? "copied" : "copy"}
        </button>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}
