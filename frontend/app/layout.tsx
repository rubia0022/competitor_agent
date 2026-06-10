import "./globals.css";

export const metadata = {
  title: "AI 竞品分析 Agent 协作系统",
  description: "Multi-Agent competitor analysis with LangGraph + Doubao",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}